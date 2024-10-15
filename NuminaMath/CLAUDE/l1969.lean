import Mathlib

namespace NUMINAMATH_CALUDE_parabola_x_axis_intersection_l1969_196974

/-- The parabola defined by y = x^2 - 2x + 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem stating that (1, 0) is the only intersection point of the parabola and the x-axis -/
theorem parabola_x_axis_intersection :
  ∃! x : ℝ, parabola x = 0 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_x_axis_intersection_l1969_196974


namespace NUMINAMATH_CALUDE_circle_centers_distance_l1969_196943

theorem circle_centers_distance (r₁ r₂ : ℝ) (angle : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 95) (h₃ : angle = 60) :
  let distance := 2 * r₂ - 2 * r₁
  distance = 160 :=
sorry

end NUMINAMATH_CALUDE_circle_centers_distance_l1969_196943


namespace NUMINAMATH_CALUDE_apples_per_basket_l1969_196952

theorem apples_per_basket (baskets_per_tree : ℕ) (trees : ℕ) (total_apples : ℕ) :
  baskets_per_tree = 20 →
  trees = 10 →
  total_apples = 3000 →
  total_apples / (trees * baskets_per_tree) = 15 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_basket_l1969_196952


namespace NUMINAMATH_CALUDE_smallest_factor_sum_l1969_196904

theorem smallest_factor_sum (b : ℕ) (p q : ℤ) : 
  (∀ x, x^2 + b*x + 2040 = (x + p) * (x + q)) →
  (∀ b' : ℕ, b' < b → 
    ¬∃ p' q' : ℤ, ∀ x, x^2 + b'*x + 2040 = (x + p') * (x + q')) →
  b = 94 :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_sum_l1969_196904


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1969_196972

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = Real.sqrt 3 * x) →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ x = -6 ∧ y = 0) →
  (∃ x y : ℝ, y^2 = 2*x ∧ x = -6) →
  (∀ x y : ℝ, x^2 / 9 - y^2 / 27 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1969_196972


namespace NUMINAMATH_CALUDE_book_loss_percentage_l1969_196999

/-- Given that the cost price of 30 books equals the selling price of 40 books,
    prove that the loss percentage is 25%. -/
theorem book_loss_percentage (C S : ℝ) (h : C > 0) (h1 : 30 * C = 40 * S) : 
  (C - S) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_loss_percentage_l1969_196999


namespace NUMINAMATH_CALUDE_f_recursive_relation_l1969_196941

def f (n : ℕ) : ℕ := (Finset.range (2 * n + 1)).sum (λ i => i ^ 2)

theorem f_recursive_relation (k : ℕ) : f (k + 1) = f k + (2 * k + 1) ^ 2 + (2 * k + 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_recursive_relation_l1969_196941


namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l1969_196937

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l1969_196937


namespace NUMINAMATH_CALUDE_correct_propositions_l1969_196951

-- Define the propositions
def proposition1 : Prop := ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

def proposition2 : Prop := 
  (∃ x : ℝ, x^2 + 1 > 3*x) ∧ (∀ x : ℝ, x^2 - 1 < 3*x)

def proposition3 : Prop := 
  ∀ a b m : ℝ, (a < b) → (a*m^2 < b*m^2)

def proposition4 : Prop := 
  ∀ p q : Prop, (p → q) ∧ ¬(q → p) → (¬p → ¬q) ∧ ¬(¬q → ¬p)

-- Theorem stating which propositions are correct
theorem correct_propositions : 
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l1969_196951


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l1969_196950

/-- The number of poles needed to enclose a rectangular plot -/
def num_poles (length width long_spacing short_spacing : ℕ) : ℕ :=
  2 * ((length / long_spacing - 1) + (width / short_spacing - 1))

/-- Theorem stating the number of poles needed for the given rectangular plot -/
theorem rectangular_plot_poles : 
  num_poles 120 80 5 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l1969_196950


namespace NUMINAMATH_CALUDE_contradiction_assumption_l1969_196969

theorem contradiction_assumption (a b : ℝ) : 
  (¬(a > b → 3*a > 3*b) ↔ 3*a ≤ 3*b) :=
by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l1969_196969


namespace NUMINAMATH_CALUDE_sin_squared_sum_range_l1969_196945

theorem sin_squared_sum_range (α β : ℝ) :
  3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 = 2 * Real.sin α →
  ∃ (x : ℝ), x = (Real.sin α)^2 + (Real.sin β)^2 ∧ 0 ≤ x ∧ x ≤ 4/9 :=
by sorry

end NUMINAMATH_CALUDE_sin_squared_sum_range_l1969_196945


namespace NUMINAMATH_CALUDE_m_plus_n_is_zero_l1969_196903

-- Define the complex function f
def f (m n z : ℂ) : ℂ := z^2 + m*z + n

-- State the theorem
theorem m_plus_n_is_zero (m n : ℂ) 
  (h : ∀ z : ℂ, Complex.abs z = 1 → Complex.abs (f m n z) = 1) : 
  m + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_m_plus_n_is_zero_l1969_196903


namespace NUMINAMATH_CALUDE_second_number_value_l1969_196975

theorem second_number_value (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : a / b = 3 / 4)
  (ratio_bc : b / c = 2 / 5)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) : 
  b = 480 / 17 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1969_196975


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1969_196962

/-- Given a square with side length a and a circle circumscribed around it,
    the area of a square inscribed in one of the resulting segments is a²/25 -/
theorem inscribed_square_area (a : ℝ) (a_pos : 0 < a) :
  ∃ (x : ℝ), x > 0 ∧ x^2 = a^2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1969_196962


namespace NUMINAMATH_CALUDE_annika_hiking_time_l1969_196927

/-- Annika's hiking problem -/
theorem annika_hiking_time (rate : ℝ) (initial_distance : ℝ) (total_distance_east : ℝ) : 
  rate = 10 →
  initial_distance = 2.75 →
  total_distance_east = 3.625 →
  (2 * total_distance_east) * rate = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_annika_hiking_time_l1969_196927


namespace NUMINAMATH_CALUDE_office_employee_count_l1969_196924

/-- Proves that the total number of employees in an office is 100 given specific salary and employee count conditions. -/
theorem office_employee_count :
  let avg_salary : ℚ := 720
  let officer_salary : ℚ := 1320
  let manager_salary : ℚ := 840
  let worker_salary : ℚ := 600
  let officer_count : ℕ := 10
  let manager_count : ℕ := 20
  ∃ (worker_count : ℕ),
    (officer_count : ℚ) * officer_salary + (manager_count : ℚ) * manager_salary + (worker_count : ℚ) * worker_salary =
    ((officer_count + manager_count + worker_count) : ℚ) * avg_salary ∧
    officer_count + manager_count + worker_count = 100 :=
by sorry

end NUMINAMATH_CALUDE_office_employee_count_l1969_196924


namespace NUMINAMATH_CALUDE_find_k_l1969_196947

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 100}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 25}
def P : ℝ × ℝ := (6, 8)
def S (k : ℝ) : ℝ × ℝ := (0, k)
def QR : ℝ := 5

-- Theorem statement
theorem find_k :
  P ∈ larger_circle ∧
  ∀ k, S k ∈ smaller_circle →
  QR = 5 →
  ∃ k, S k ∈ smaller_circle ∧ k = 5 := by
sorry

end NUMINAMATH_CALUDE_find_k_l1969_196947


namespace NUMINAMATH_CALUDE_intersection_sum_l1969_196939

theorem intersection_sum (a b : ℚ) : 
  (2 = (1/3) * 3 + a) → 
  (3 = (1/5) * 2 + b) → 
  a + b = 18/5 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1969_196939


namespace NUMINAMATH_CALUDE_power_multiplication_division_equality_l1969_196914

theorem power_multiplication_division_equality : (15^2 * 8^3) / 256 = 450 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_division_equality_l1969_196914


namespace NUMINAMATH_CALUDE_cost_for_three_roofs_is_1215_l1969_196965

/-- Calculates the total cost of materials for building roofs with discounts applied --/
def total_cost_with_discounts (
  num_roofs : ℕ
  ) (
  metal_bars_per_roof : ℕ
  ) (
  wooden_beams_per_roof : ℕ
  ) (
  steel_rods_per_roof : ℕ
  ) (
  bars_per_set : ℕ
  ) (
  beams_per_set : ℕ
  ) (
  rods_per_set : ℕ
  ) (
  cost_per_bar : ℕ
  ) (
  cost_per_beam : ℕ
  ) (
  cost_per_rod : ℕ
  ) (
  discount_threshold : ℕ
  ) (
  discount_rate : ℚ
  ) : ℕ :=
  sorry

/-- Theorem stating that the total cost for building 3 roofs with given specifications is $1215 --/
theorem cost_for_three_roofs_is_1215 :
  total_cost_with_discounts 3 2 3 1 7 5 4 10 15 20 10 (1/10) = 1215 :=
  sorry

end NUMINAMATH_CALUDE_cost_for_three_roofs_is_1215_l1969_196965


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1969_196955

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 = 24 / 77 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1969_196955


namespace NUMINAMATH_CALUDE_student_subtraction_problem_l1969_196954

theorem student_subtraction_problem (x y : ℤ) : 
  x = 40 → 7 * x - y = 130 → y = 150 := by sorry

end NUMINAMATH_CALUDE_student_subtraction_problem_l1969_196954


namespace NUMINAMATH_CALUDE_probability_green_yellow_blue_l1969_196964

def total_balls : ℕ := 500
def green_balls : ℕ := 100
def yellow_balls : ℕ := 70
def blue_balls : ℕ := 50

theorem probability_green_yellow_blue :
  (green_balls + yellow_balls + blue_balls : ℚ) / total_balls = 220 / 500 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_yellow_blue_l1969_196964


namespace NUMINAMATH_CALUDE_equation_in_y_l1969_196968

theorem equation_in_y (x y : ℝ) 
  (eq1 : 3 * x^2 - 5 * x + 4 * y + 6 = 0)
  (eq2 : 3 * x - 2 * y + 1 = 0) :
  4 * y^2 - 2 * y + 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_in_y_l1969_196968


namespace NUMINAMATH_CALUDE_eggs_broken_count_l1969_196940

-- Define the number of brown eggs
def brown_eggs : ℕ := 10

-- Define the number of white eggs
def white_eggs : ℕ := 3 * brown_eggs

-- Define the total number of eggs before dropping
def total_eggs_before : ℕ := brown_eggs + white_eggs

-- Define the number of eggs left after dropping
def eggs_left_after : ℕ := 20

-- Theorem to prove
theorem eggs_broken_count : total_eggs_before - eggs_left_after = 20 := by
  sorry

end NUMINAMATH_CALUDE_eggs_broken_count_l1969_196940


namespace NUMINAMATH_CALUDE_selection_theorem_l1969_196984

/-- The number of students in the group -/
def total_students : Nat := 6

/-- The number of students to be selected -/
def selected_students : Nat := 4

/-- The number of subjects -/
def subjects : Nat := 4

/-- The number of students who cannot participate in a specific subject -/
def restricted_students : Nat := 2

/-- The number of different selection plans -/
def selection_plans : Nat := 240

theorem selection_theorem :
  (total_students.factorial / (total_students - selected_students).factorial) -
  (restricted_students * ((total_students - 1).factorial / (total_students - selected_students).factorial)) =
  selection_plans := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l1969_196984


namespace NUMINAMATH_CALUDE_painted_cubes_count_l1969_196966

/-- A cube-based construction with 5 layers -/
structure CubeConstruction where
  middle_layer : Nat
  other_layers : Nat
  unpainted_cubes : Nat

/-- The number of cubes with at least one face painted in the construction -/
def painted_cubes (c : CubeConstruction) : Nat :=
  c.middle_layer + 4 * c.other_layers - c.unpainted_cubes

/-- Theorem: In the given cube construction, 104 cubes have at least one face painted -/
theorem painted_cubes_count (c : CubeConstruction) 
  (h1 : c.middle_layer = 16)
  (h2 : c.other_layers = 24)
  (h3 : c.unpainted_cubes = 8) : 
  painted_cubes c = 104 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l1969_196966


namespace NUMINAMATH_CALUDE_cyclic_equation_solution_l1969_196949

def cyclic_index (n i : ℕ) : ℕ :=
  (i - 1) % n + 1

theorem cyclic_equation_solution (n : ℕ) (x : ℕ → ℝ) :
  (∀ i, 0 ≤ x i) →
  (∀ k, x k + x (cyclic_index n (k + 1)) = (x (cyclic_index n (k + 2)))^2) →
  (∀ i, x i = 0 ∨ x i = 2) :=
sorry

end NUMINAMATH_CALUDE_cyclic_equation_solution_l1969_196949


namespace NUMINAMATH_CALUDE_original_number_l1969_196979

theorem original_number : ∃ x : ℝ, 3 * (2 * x + 6) = 72 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1969_196979


namespace NUMINAMATH_CALUDE_balloon_arrangement_count_l1969_196908

/-- The number of letters in the word "BALLOON" -/
def n : ℕ := 7

/-- The number of times the letter 'L' appears in "BALLOON" -/
def l_count : ℕ := 2

/-- The number of times the letter 'O' appears in "BALLOON" -/
def o_count : ℕ := 2

/-- The number of unique arrangements of the letters in "BALLOON" -/
def balloon_arrangements : ℕ := n.factorial / (l_count.factorial * o_count.factorial)

theorem balloon_arrangement_count : balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangement_count_l1969_196908


namespace NUMINAMATH_CALUDE_line_separates_points_l1969_196906

/-- Given that the origin (0,0) and the point (1,1) are on opposite sides of the line x+y=a,
    prove that the range of values for a is 0 < a < 2. -/
theorem line_separates_points (a : ℝ) : 
  (0 + 0 - a) * (1 + 1 - a) < 0 → 0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_line_separates_points_l1969_196906


namespace NUMINAMATH_CALUDE_local_extrema_sum_l1969_196917

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 27

-- State the theorem
theorem local_extrema_sum (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b (-1) ≥ f a b x) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f a b 3 ≤ f a b x) →
  a + b = -12 := by
  sorry

end NUMINAMATH_CALUDE_local_extrema_sum_l1969_196917


namespace NUMINAMATH_CALUDE_arithmetic_progression_cube_sum_l1969_196920

theorem arithmetic_progression_cube_sum (k x y z : ℤ) :
  (x < y ∧ y < z) →  -- x, y, z form an increasing sequence
  (z - y = y - x) →  -- x, y, z form an arithmetic progression
  (k * y^3 = x^3 + z^3) →  -- given equation
  ∃ t : ℤ, k = 2 * (3 * t^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cube_sum_l1969_196920


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1969_196982

theorem intersection_of_sets (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
  let B : Set ℝ := {x | x^2 + 2*(a + 1)*x + (a^2 - 5) = 0}
  (A ∩ B = {2}) → (a = -1 ∨ a = -3) := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1969_196982


namespace NUMINAMATH_CALUDE_cube_product_divided_l1969_196960

theorem cube_product_divided : (12 : ℝ)^3 * 6^3 / 432 = 864 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_divided_l1969_196960


namespace NUMINAMATH_CALUDE_sum_of_first_100_factorials_mod_100_l1969_196938

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem sum_of_first_100_factorials_mod_100 :
  sum_of_factorials 100 % 100 = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_100_factorials_mod_100_l1969_196938


namespace NUMINAMATH_CALUDE_inequality_proof_l1969_196987

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 3) :
  (a * b / Real.sqrt (c^2 + 3)) + 
  (b * c / Real.sqrt (a^2 + 3)) + 
  (c * a / Real.sqrt (b^2 + 3)) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1969_196987


namespace NUMINAMATH_CALUDE_profit_percentage_cricket_bat_l1969_196970

/-- The profit percentage calculation for a cricket bat sale -/
theorem profit_percentage_cricket_bat (selling_price profit : ℝ)
  (h1 : selling_price = 850)
  (h2 : profit = 230) :
  ∃ (percentage : ℝ), abs (percentage - 37.10) < 0.01 ∧
  percentage = (profit / (selling_price - profit)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_cricket_bat_l1969_196970


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l1969_196916

theorem cubic_root_ratio (a b c d : ℝ) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 2 ∨ x = 3) → 
  c / d = -1 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l1969_196916


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1969_196948

/-- A rectangle with length thrice its breadth and area 75 square meters has a perimeter of 40 meters. -/
theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 75 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1969_196948


namespace NUMINAMATH_CALUDE_ones_digit_of_nine_to_27_l1969_196919

def ones_digit (n : ℕ) : ℕ := n % 10

def power_of_nine_ones_digit (n : ℕ) : ℕ :=
  if n % 2 = 1 then 9 else 1

theorem ones_digit_of_nine_to_27 :
  ones_digit (9^27) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_nine_to_27_l1969_196919


namespace NUMINAMATH_CALUDE_intersection_points_convex_ngon_l1969_196992

/-- The number of intersection points of the diagonals in a convex n-gon -/
def intersectionPoints (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- Theorem: The number of intersection points of the diagonals in a convex n-gon
    is equal to (n choose 4) for n ≥ 4 -/
theorem intersection_points_convex_ngon (n : ℕ) (h : n ≥ 4) :
  intersectionPoints n = Nat.choose n 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_convex_ngon_l1969_196992


namespace NUMINAMATH_CALUDE_frisbee_price_l1969_196934

/-- Given the conditions of frisbee sales, prove the price of non-$4 frisbees -/
theorem frisbee_price (total_frisbees : ℕ) (total_receipts : ℕ) (price_known : ℕ) (min_known_price : ℕ) :
  total_frisbees = 64 →
  total_receipts = 196 →
  price_known = 4 →
  min_known_price = 4 →
  ∃ (price_unknown : ℕ),
    price_unknown * (total_frisbees - min_known_price) + price_known * min_known_price = total_receipts ∧
    price_unknown = 3 := by
  sorry

#check frisbee_price

end NUMINAMATH_CALUDE_frisbee_price_l1969_196934


namespace NUMINAMATH_CALUDE_f_geq_2_iff_max_m_value_l1969_196935

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the first part of the problem
theorem f_geq_2_iff (x : ℝ) :
  f x ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2 :=
sorry

-- Theorem for the second part of the problem
theorem max_m_value :
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ -2*x^2 + m) ∧
  (∀ m : ℝ, (∀ x : ℝ, f x ≥ -2*x^2 + m) → m ≤ 5/2) :=
sorry

end NUMINAMATH_CALUDE_f_geq_2_iff_max_m_value_l1969_196935


namespace NUMINAMATH_CALUDE_saree_final_price_l1969_196988

/-- Calculates the final sale price of a saree after discounts, tax, and custom fee. -/
def saree_sale_price (original_price : ℝ) (discount1 discount2 discount3 tax : ℝ) (custom_fee : ℝ) : ℝ :=
  let price_after_discounts := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let price_after_tax := price_after_discounts * (1 + tax)
  price_after_tax + custom_fee

/-- Theorem stating that the final sale price of the saree is 773.2 -/
theorem saree_final_price :
  saree_sale_price 1200 0.25 0.20 0.15 0.10 100 = 773.2 := by
  sorry

#eval saree_sale_price 1200 0.25 0.20 0.15 0.10 100

end NUMINAMATH_CALUDE_saree_final_price_l1969_196988


namespace NUMINAMATH_CALUDE_b_value_l1969_196981

/-- The value of b that satisfies the given conditions -/
def find_b : ℝ := sorry

/-- The line equation y = b - x -/
def line_equation (x y : ℝ) : Prop := y = find_b - x

/-- P is the intersection point of the line with the y-axis -/
def P : ℝ × ℝ := (0, find_b)

/-- S is the intersection point of the line with x = 6 -/
def S : ℝ × ℝ := (6, find_b - 6)

/-- Q is the intersection point of the line with the x-axis -/
def Q : ℝ × ℝ := (find_b, 0)

/-- O is the origin -/
def O : ℝ × ℝ := (0, 0)

/-- R is the point (6, 0) -/
def R : ℝ × ℝ := (6, 0)

/-- The area of triangle QRS -/
def area_QRS : ℝ := sorry

/-- The area of triangle QOP -/
def area_QOP : ℝ := sorry

theorem b_value :
  0 < find_b ∧ 
  find_b < 6 ∧ 
  line_equation (P.1) (P.2) ∧
  line_equation (S.1) (S.2) ∧
  (area_QRS / area_QOP = 4 / 25) →
  ∃ ε > 0, |find_b - 4.3| < ε :=
sorry

end NUMINAMATH_CALUDE_b_value_l1969_196981


namespace NUMINAMATH_CALUDE_ratio_problem_l1969_196958

theorem ratio_problem (x y : ℚ) (h : (3 * x + 2 * y) / (2 * x - y) = 5 / 4) : 
  x / y = -13 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1969_196958


namespace NUMINAMATH_CALUDE_triangle_angle_range_l1969_196971

theorem triangle_angle_range (A B C : Real) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) (h5 : Real.log (Real.tan A) + Real.log (Real.tan C) = 2 * Real.log (Real.tan B)) : 
  π / 3 ≤ B ∧ B < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_range_l1969_196971


namespace NUMINAMATH_CALUDE_count_solutions_eq_338350_l1969_196901

/-- The number of distinct integer solutions to |x| + |y| < 100 -/
def count_solutions : ℕ :=
  (Finset.sum (Finset.range 100) (fun k => (k + 1)^2) : ℕ)

/-- Theorem stating the correct number of solutions -/
theorem count_solutions_eq_338350 : count_solutions = 338350 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_eq_338350_l1969_196901


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_neg_one_two_l1969_196928

/-- Given an angle α whose terminal side passes through the point (-1, 2), 
    prove that cos α = -√5 / 5 -/
theorem cos_alpha_for_point_neg_one_two (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) → 
  Real.cos α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_neg_one_two_l1969_196928


namespace NUMINAMATH_CALUDE_max_value_of_x_l1969_196910

theorem max_value_of_x (x : ℕ) : 
  x > 0 ∧ 
  ∃ k : ℕ, x = 4 * k ∧ 
  x^3 < 1728 →
  x ≤ 8 ∧ ∃ y : ℕ, y > 0 ∧ ∃ m : ℕ, y = 4 * m ∧ y^3 < 1728 ∧ y = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_l1969_196910


namespace NUMINAMATH_CALUDE_total_yen_calculation_l1969_196994

theorem total_yen_calculation (checking_account savings_account : ℕ) 
  (h1 : checking_account = 6359)
  (h2 : savings_account = 3485) :
  checking_account + savings_account = 9844 := by
  sorry

end NUMINAMATH_CALUDE_total_yen_calculation_l1969_196994


namespace NUMINAMATH_CALUDE_exists_solution_with_y_twelve_l1969_196912

theorem exists_solution_with_y_twelve :
  ∃ (x z t : ℕ+), x + 12 + z + t = 15 := by
sorry

end NUMINAMATH_CALUDE_exists_solution_with_y_twelve_l1969_196912


namespace NUMINAMATH_CALUDE_A_xor_B_equals_one_three_l1969_196918

-- Define the ⊕ operation
def setXor (M P : Set ℝ) : Set ℝ := {x | x ∈ M ∨ x ∈ P ∧ x ∉ M ∩ P}

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

-- Theorem statement
theorem A_xor_B_equals_one_three : setXor A B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_xor_B_equals_one_three_l1969_196918


namespace NUMINAMATH_CALUDE_equation_solution_l1969_196932

theorem equation_solution : ∃ x : ℝ, 2*x + 17 = 32 - 3*x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1969_196932


namespace NUMINAMATH_CALUDE_johns_phone_bill_l1969_196976

/-- Calculates the total phone bill given the monthly fee, per-minute rate, and minutes used. -/
def total_bill (monthly_fee : ℝ) (per_minute_rate : ℝ) (minutes_used : ℝ) : ℝ :=
  monthly_fee + per_minute_rate * minutes_used

/-- Theorem stating that John's phone bill is $12.02 given the specified conditions. -/
theorem johns_phone_bill :
  let monthly_fee : ℝ := 5
  let per_minute_rate : ℝ := 0.25
  let minutes_used : ℝ := 28.08
  total_bill monthly_fee per_minute_rate minutes_used = 12.02 := by
sorry


end NUMINAMATH_CALUDE_johns_phone_bill_l1969_196976


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1969_196995

/-- The number of ways to arrange books on a shelf --/
def arrange_books (num_math_books : ℕ) (num_history_books : ℕ) : ℕ :=
  let math_book_arrangements := (num_math_books.choose 2) * (2 * 2)
  let history_book_arrangements := num_history_books.factorial
  math_book_arrangements * history_book_arrangements

/-- Theorem: The number of ways to arrange 4 math books and 6 history books with 2 math books on each end is 17280 --/
theorem book_arrangement_theorem :
  arrange_books 4 6 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1969_196995


namespace NUMINAMATH_CALUDE_max_value_of_z_l1969_196973

-- Define the variables and the objective function
variables (x y : ℝ)
def z : ℝ → ℝ → ℝ := λ x y => 2 * x + y

-- Define the constraints
def constraint1 (x y : ℝ) : Prop := x + 2 * y ≤ 2
def constraint2 (x y : ℝ) : Prop := x + y ≥ 0
def constraint3 (x : ℝ) : Prop := x ≤ 4

-- Theorem statement
theorem max_value_of_z :
  ∀ x y : ℝ, constraint1 x y → constraint2 x y → constraint3 x →
  z x y ≤ 11 ∧ ∃ x₀ y₀ : ℝ, constraint1 x₀ y₀ ∧ constraint2 x₀ y₀ ∧ constraint3 x₀ ∧ z x₀ y₀ = 11 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l1969_196973


namespace NUMINAMATH_CALUDE_strawberry_to_fruit_ratio_l1969_196990

-- Define the total garden size
def garden_size : ℕ := 64

-- Define the fruit section size (half of the garden)
def fruit_section : ℕ := garden_size / 2

-- Define the strawberry section size
def strawberry_section : ℕ := 8

-- Theorem to prove the ratio of strawberry section to fruit section
theorem strawberry_to_fruit_ratio :
  (strawberry_section : ℚ) / fruit_section = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_to_fruit_ratio_l1969_196990


namespace NUMINAMATH_CALUDE_largest_a_value_l1969_196983

theorem largest_a_value : 
  ∀ a : ℝ, (3*a + 4)*(a - 2) = 9*a → a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_a_value_l1969_196983


namespace NUMINAMATH_CALUDE_factorial_base_representation_823_l1969_196997

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorial_base_coeff (n k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

theorem factorial_base_representation_823 :
  factorial_base_coeff 823 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_base_representation_823_l1969_196997


namespace NUMINAMATH_CALUDE_balloon_tank_capacity_l1969_196980

theorem balloon_tank_capacity 
  (num_balloons : ℕ) 
  (air_per_balloon : ℕ) 
  (num_tanks : ℕ) 
  (h1 : num_balloons = 1000)
  (h2 : air_per_balloon = 10)
  (h3 : num_tanks = 20) :
  (num_balloons * air_per_balloon) / num_tanks = 500 := by
  sorry

end NUMINAMATH_CALUDE_balloon_tank_capacity_l1969_196980


namespace NUMINAMATH_CALUDE_greeting_cards_exchange_l1969_196977

theorem greeting_cards_exchange (x : ℕ) : x > 0 → x * (x - 1) = 1980 → ∀ (i j : ℕ), i < x ∧ j < x ∧ i ≠ j → ∃ (total : ℕ), total = 1980 ∧ total = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_greeting_cards_exchange_l1969_196977


namespace NUMINAMATH_CALUDE_sin_cos_105_15_identity_l1969_196926

theorem sin_cos_105_15_identity : 
  Real.sin (105 * π / 180) * Real.sin (15 * π / 180) - 
  Real.cos (105 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_105_15_identity_l1969_196926


namespace NUMINAMATH_CALUDE_diana_wins_probability_l1969_196963

def standard_die := Finset.range 6

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (standard_die.product standard_die).filter (fun (d, a) => d > a)

theorem diana_wins_probability :
  (favorable_outcomes.card : ℚ) / (standard_die.card * standard_die.card) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_diana_wins_probability_l1969_196963


namespace NUMINAMATH_CALUDE_regular_pay_limit_l1969_196956

theorem regular_pay_limit (regular_rate : ℝ) (overtime_hours : ℝ) (total_pay : ℝ) 
  (h1 : regular_rate = 3)
  (h2 : overtime_hours = 13)
  (h3 : total_pay = 198) :
  ∃ (regular_hours : ℝ),
    regular_hours * regular_rate + overtime_hours * (2 * regular_rate) = total_pay ∧
    regular_hours = 40 := by
  sorry

end NUMINAMATH_CALUDE_regular_pay_limit_l1969_196956


namespace NUMINAMATH_CALUDE_function_symmetry_implies_a_equals_four_l1969_196946

/-- Given a quadratic function f(x) = 2x^2 - ax + 3, 
    if f(1-x) = f(1+x) for all real x, then a = 4 -/
theorem function_symmetry_implies_a_equals_four (a : ℝ) : 
  (∀ x : ℝ, 2*(1-x)^2 - a*(1-x) + 3 = 2*(1+x)^2 - a*(1+x) + 3) → 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_implies_a_equals_four_l1969_196946


namespace NUMINAMATH_CALUDE_set_operations_and_range_l1969_196911

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

theorem set_operations_and_range :
  ∀ a : ℝ, (B ∪ C a = C a) →
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 4}) ∧
  (A ∪ B = {x : ℝ | x ≥ 2}) ∧
  ((U \ (A ∪ B)) = {x : ℝ | 0 < x ∧ x < 2}) ∧
  ((U \ A) ∩ B = {x : ℝ | x ≥ 4}) ∧
  (∀ x : ℝ, x > -6 ↔ ∃ y : ℝ, y ∈ C x ∧ y ∉ B) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l1969_196911


namespace NUMINAMATH_CALUDE_extreme_value_implies_m_eq_two_l1969_196922

/-- The function f(x) = x³ - (3/2)x² + m has an extreme value of 3/2 in the interval (0, 2) -/
def has_extreme_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo 0 2, f x = 3/2 ∧ ∀ y ∈ Set.Ioo 0 2, f y ≤ f x

/-- The main theorem stating that if f(x) = x³ - (3/2)x² + m has an extreme value of 3/2 
    in the interval (0, 2), then m = 2 -/
theorem extreme_value_implies_m_eq_two :
  ∀ m : ℝ, has_extreme_value (fun x => x^3 - (3/2)*x^2 + m) m → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_m_eq_two_l1969_196922


namespace NUMINAMATH_CALUDE_james_chores_total_time_l1969_196929

/-- Given James' chore schedule, prove that the total time spent is 16.5 hours -/
theorem james_chores_total_time :
  let vacuuming_time : ℝ := 3
  let cleaning_time : ℝ := 3 * vacuuming_time
  let laundry_time : ℝ := cleaning_time / 2
  vacuuming_time + cleaning_time + laundry_time = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_james_chores_total_time_l1969_196929


namespace NUMINAMATH_CALUDE_total_fruits_is_112_l1969_196921

/-- The number of apples and pears satisfying the given conditions -/
def total_fruits (apples pears : ℕ) : Prop :=
  ∃ (bags : ℕ),
    (5 * bags + 4 = apples) ∧
    (3 * bags = pears - 12) ∧
    (7 * bags = apples) ∧
    (3 * bags + 12 = pears)

/-- Theorem stating that the total number of fruits is 112 -/
theorem total_fruits_is_112 :
  ∃ (apples pears : ℕ), total_fruits apples pears ∧ apples + pears = 112 :=
sorry

end NUMINAMATH_CALUDE_total_fruits_is_112_l1969_196921


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1969_196909

theorem arithmetic_mean_problem (x y : ℝ) : 
  ((x + 10) + 18 + 3*x + 12 + (3*x + 6) + y) / 6 = 26 → x = 90/7 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1969_196909


namespace NUMINAMATH_CALUDE_some_pens_not_vens_l1969_196902

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Pen Den Ven : U → Prop)

-- Define the hypotheses
variable (h1 : ∀ x, Pen x → Den x)
variable (h2 : ∃ x, Den x ∧ ¬Ven x)

-- State the theorem
theorem some_pens_not_vens : ∃ x, Pen x ∧ ¬Ven x :=
sorry

end NUMINAMATH_CALUDE_some_pens_not_vens_l1969_196902


namespace NUMINAMATH_CALUDE_water_pump_problem_l1969_196967

theorem water_pump_problem (t₁ t₂ t_combined : ℝ) 
  (h₁ : t₂ = 6)
  (h₂ : t_combined = 3.6)
  (h₃ : 1 / t₁ + 1 / t₂ = 1 / t_combined) :
  t₁ = 9 := by
sorry

end NUMINAMATH_CALUDE_water_pump_problem_l1969_196967


namespace NUMINAMATH_CALUDE_lantern_probability_l1969_196986

def total_large_lanterns : ℕ := 360
def total_small_lanterns : ℕ := 1200

def large_with_two_small (x : ℕ) : Prop := 
  x * 2 + (total_large_lanterns - x) * 4 = total_small_lanterns

def large_with_four_small (x : ℕ) : ℕ := total_large_lanterns - x

def total_combinations : ℕ := total_large_lanterns.choose 2

def favorable_outcomes (x : ℕ) : ℕ := 
  (large_with_four_small x).choose 2 + (large_with_four_small x).choose 1 * x.choose 1

theorem lantern_probability (x : ℕ) (h : large_with_two_small x) : 
  (favorable_outcomes x : ℚ) / total_combinations = 958 / 1077 := by sorry

end NUMINAMATH_CALUDE_lantern_probability_l1969_196986


namespace NUMINAMATH_CALUDE_initial_investment_is_200_l1969_196923

/-- Represents the simple interest calculation -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating that given the conditions, the initial investment is $200 -/
theorem initial_investment_is_200 
  (P : ℝ) 
  (h1 : simpleInterest P (1/15) 3 = 240) 
  (h2 : simpleInterest 150 (1/15) 6 = 210) : 
  P = 200 := by
  sorry

#check initial_investment_is_200

end NUMINAMATH_CALUDE_initial_investment_is_200_l1969_196923


namespace NUMINAMATH_CALUDE_decimal_parts_fraction_decimal_parts_fraction_proof_l1969_196930

theorem decimal_parts_fraction : ℝ → Prop := 
  fun x => let a : ℤ := ⌊2 + Real.sqrt 2⌋
           let b : ℝ := 2 + Real.sqrt 2 - a
           let c : ℤ := ⌊4 - Real.sqrt 2⌋
           let d : ℝ := 4 - Real.sqrt 2 - c
           (b + d) / (a * c) = 1/6

theorem decimal_parts_fraction_proof : decimal_parts_fraction 2 := by
  sorry

end NUMINAMATH_CALUDE_decimal_parts_fraction_decimal_parts_fraction_proof_l1969_196930


namespace NUMINAMATH_CALUDE_fence_painting_problem_l1969_196913

theorem fence_painting_problem (initial_people : ℕ) (initial_time : ℝ) (new_time : ℝ) :
  initial_people = 8 →
  initial_time = 3 →
  new_time = 2 →
  ∃ (new_people : ℕ), 
    (initial_people : ℝ) * initial_time = (new_people : ℝ) * new_time ∧ 
    new_people = 12 :=
by sorry

end NUMINAMATH_CALUDE_fence_painting_problem_l1969_196913


namespace NUMINAMATH_CALUDE_solution_equals_one_l1969_196978

theorem solution_equals_one (x y : ℝ) 
  (eq1 : 2 * x + y = 4) 
  (eq2 : x + 2 * y = 5) : 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_equals_one_l1969_196978


namespace NUMINAMATH_CALUDE_min_value_on_interval_l1969_196959

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The closed interval [2, 3] -/
def I : Set ℝ := Set.Icc 2 3

theorem min_value_on_interval :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ x ∈ I, f x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l1969_196959


namespace NUMINAMATH_CALUDE_smallest_third_side_l1969_196933

theorem smallest_third_side (a b : ℝ) (ha : a = 7.5) (hb : b = 11.5) :
  ∃ (s : ℕ), s = 5 ∧ 
  (a + s > b) ∧ (a + b > s) ∧ (b + s > a) ∧
  (∀ (t : ℕ), t < s → ¬(a + t > b ∧ a + b > t ∧ b + t > a)) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_third_side_l1969_196933


namespace NUMINAMATH_CALUDE_temple_shop_charge_l1969_196996

/-- The charge per object at the temple shop -/
def charge_per_object : ℕ → ℕ → ℕ → ℕ → ℚ
  | num_people, shoes_per_person, socks_per_person, mobiles_per_person =>
    let total_objects := num_people * (shoes_per_person + socks_per_person + mobiles_per_person)
    let total_cost := 165
    total_cost / total_objects

theorem temple_shop_charge :
  charge_per_object 3 2 2 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_temple_shop_charge_l1969_196996


namespace NUMINAMATH_CALUDE_jessica_candy_distribution_l1969_196942

/-- The number of candies Jessica must remove to distribute them equally among her friends -/
def candies_to_remove (total : Nat) (friends : Nat) : Nat :=
  total % friends

theorem jessica_candy_distribution :
  candies_to_remove 30 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_candy_distribution_l1969_196942


namespace NUMINAMATH_CALUDE_costs_equal_at_60_guests_l1969_196915

/-- The number of guests for which the costs of Caesar's and Venus Hall are equal -/
def equal_cost_guests : ℕ := 60

/-- Caesar's room rental cost -/
def caesars_rental : ℕ := 800

/-- Caesar's per-meal cost -/
def caesars_meal : ℕ := 30

/-- Venus Hall's room rental cost -/
def venus_rental : ℕ := 500

/-- Venus Hall's per-meal cost -/
def venus_meal : ℕ := 35

/-- Theorem stating that the costs are equal for the given number of guests -/
theorem costs_equal_at_60_guests : 
  caesars_rental + caesars_meal * equal_cost_guests = 
  venus_rental + venus_meal * equal_cost_guests :=
by sorry

end NUMINAMATH_CALUDE_costs_equal_at_60_guests_l1969_196915


namespace NUMINAMATH_CALUDE_remainder_not_always_power_of_four_l1969_196985

theorem remainder_not_always_power_of_four :
  ∃ n : ℕ, n ≥ 2 ∧ ∃ k : ℕ, (2^(2^n) : ℕ) % (2^n - 1) = k ∧ ¬∃ m : ℕ, k = 4^m := by
  sorry

end NUMINAMATH_CALUDE_remainder_not_always_power_of_four_l1969_196985


namespace NUMINAMATH_CALUDE_bogatyr_age_l1969_196925

/-- Represents the ages of five wine brands -/
structure WineAges where
  carlo_rosi : ℕ
  franzia : ℕ
  twin_valley : ℕ
  beaulieu_vineyard : ℕ
  bogatyr : ℕ

/-- Defines the relationships between wine ages -/
def valid_wine_ages (ages : WineAges) : Prop :=
  ages.carlo_rosi = 40 ∧
  ages.franzia = 3 * ages.carlo_rosi ∧
  ages.carlo_rosi = 4 * ages.twin_valley ∧
  ages.beaulieu_vineyard = ages.twin_valley / 2 ∧
  ages.bogatyr = 2 * ages.franzia

/-- Theorem: Given the relationships between wine ages, Bogatyr's age is 240 years -/
theorem bogatyr_age (ages : WineAges) (h : valid_wine_ages ages) : ages.bogatyr = 240 := by
  sorry

end NUMINAMATH_CALUDE_bogatyr_age_l1969_196925


namespace NUMINAMATH_CALUDE_zero_in_interval_l1969_196900

/-- The function f(x) = log_a x + x - b -/
noncomputable def f (a b x : ℝ) : ℝ := (Real.log x) / (Real.log a) + x - b

/-- The theorem stating that the zero of f(x) lies in (2, 3) -/
theorem zero_in_interval (a b : ℝ) (ha : 0 < a) (ha' : a ≠ 1) 
  (hab : 2 < a ∧ a < 3 ∧ 3 < b ∧ b < 4) :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f a b x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1969_196900


namespace NUMINAMATH_CALUDE_subset_union_equality_l1969_196989

theorem subset_union_equality (n : ℕ+) (A : Fin (n + 1) → Set (Fin n)) 
  (h : ∀ i, (A i).Nonempty) :
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) := by
  sorry

end NUMINAMATH_CALUDE_subset_union_equality_l1969_196989


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1969_196998

theorem smallest_number_of_eggs : ∀ (n : ℕ), 
  (n > 150) → 
  (∃ (c : ℕ), n = 15 * c - 5) → 
  (∀ (m : ℕ), m > 150 ∧ (∃ (d : ℕ), m = 15 * d - 5) → m ≥ n) → 
  n = 160 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1969_196998


namespace NUMINAMATH_CALUDE_f_min_max_l1969_196905

-- Define the function
def f (x : ℝ) : ℝ := 1 + 3*x - x^3

-- State the theorem
theorem f_min_max : 
  (∃ x : ℝ, f x = -1) ∧ 
  (∀ x : ℝ, f x ≥ -1) ∧ 
  (∃ x : ℝ, f x = 3) ∧ 
  (∀ x : ℝ, f x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_f_min_max_l1969_196905


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_containing_interval_l1969_196936

-- Define the functions f and g
def f (a x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part 1
theorem solution_set_when_a_is_one :
  let a := 1
  ∃ S : Set ℝ, S = {x | f a x ≥ g x} ∧ 
    S = Set.Icc (-1) (((-1 : ℝ) + Real.sqrt 17) / 2) :=
sorry

-- Theorem for part 2
theorem range_of_a_containing_interval :
  ∃ R : Set ℝ, R = {a | ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ g x} ∧
    R = Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_containing_interval_l1969_196936


namespace NUMINAMATH_CALUDE_program_cost_is_40_92_l1969_196907

/-- Represents the cost calculation for a computer program run -/
def program_cost_calculation (milliseconds_per_second : ℝ) 
                             (os_overhead : ℝ) 
                             (cost_per_millisecond : ℝ) 
                             (tape_mounting_cost : ℝ) 
                             (program_runtime_seconds : ℝ) : ℝ :=
  let total_milliseconds := program_runtime_seconds * milliseconds_per_second
  os_overhead + (cost_per_millisecond * total_milliseconds) + tape_mounting_cost

/-- Theorem stating that the total cost for the given program run is $40.92 -/
theorem program_cost_is_40_92 : 
  program_cost_calculation 1000 1.07 0.023 5.35 1.5 = 40.92 := by
  sorry

end NUMINAMATH_CALUDE_program_cost_is_40_92_l1969_196907


namespace NUMINAMATH_CALUDE_prob_two_girls_from_five_l1969_196931

/-- The probability of selecting 2 girls as representatives from a group of 5 students (2 boys and 3 girls) is 3/10. -/
theorem prob_two_girls_from_five (total : ℕ) (boys : ℕ) (girls : ℕ) (representatives : ℕ) :
  total = 5 →
  boys = 2 →
  girls = 3 →
  representatives = 2 →
  (Nat.choose girls representatives : ℚ) / (Nat.choose total representatives : ℚ) = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_prob_two_girls_from_five_l1969_196931


namespace NUMINAMATH_CALUDE_prime_power_implies_prime_n_l1969_196993

theorem prime_power_implies_prime_n (n : ℕ) (p : ℕ) (k : ℕ) :
  (∃ (p : ℕ), Prime p ∧ ∃ (k : ℕ), 3^n - 2^n = p^k) →
  Prime n :=
by sorry

end NUMINAMATH_CALUDE_prime_power_implies_prime_n_l1969_196993


namespace NUMINAMATH_CALUDE_number_relations_l1969_196953

theorem number_relations (A B C : ℝ) : 
  A - B = 1860 ∧ 
  0.075 * A = 0.125 * B ∧ 
  0.15 * B = 0.05 * C → 
  A = 4650 ∧ B = 2790 ∧ C = 8370 := by
sorry

end NUMINAMATH_CALUDE_number_relations_l1969_196953


namespace NUMINAMATH_CALUDE_upstream_distance_is_18_l1969_196961

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  still_speed : ℝ  -- Speed of the man in still water (km/h)
  downstream_distance : ℝ  -- Distance swam downstream (km)
  downstream_time : ℝ  -- Time spent swimming downstream (hours)
  upstream_time : ℝ  -- Time spent swimming upstream (hours)

/-- Calculates the upstream distance given a swimming scenario -/
def upstream_distance (s : SwimmingScenario) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating that for the given conditions, the upstream distance is 18 km -/
theorem upstream_distance_is_18 :
  let s : SwimmingScenario := {
    still_speed := 11.5,
    downstream_distance := 51,
    downstream_time := 3,
    upstream_time := 3
  }
  upstream_distance s = 18 := by
  sorry

end NUMINAMATH_CALUDE_upstream_distance_is_18_l1969_196961


namespace NUMINAMATH_CALUDE_right_triangle_sine_l1969_196944

theorem right_triangle_sine (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 9) (h3 : c = 15) :
  a / c = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sine_l1969_196944


namespace NUMINAMATH_CALUDE_distance_between_towns_l1969_196957

theorem distance_between_towns (total_distance : ℝ) : total_distance = 50 :=
  let petya_distance := 10 + (1/4) * (total_distance - 10)
  let kolya_distance := 20 + (1/3) * (total_distance - 20)
  have h1 : petya_distance + kolya_distance = total_distance := by sorry
  have h2 : petya_distance = 10 + (1/4) * (total_distance - 10) := by sorry
  have h3 : kolya_distance = 20 + (1/3) * (total_distance - 20) := by sorry
  sorry

end NUMINAMATH_CALUDE_distance_between_towns_l1969_196957


namespace NUMINAMATH_CALUDE_fraction_simplification_l1969_196991

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) :
  3 / (m^2 - 9) + m / (9 - m^2) = -1 / (m + 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1969_196991
