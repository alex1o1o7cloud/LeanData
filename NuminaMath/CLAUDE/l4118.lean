import Mathlib

namespace NUMINAMATH_CALUDE_chess_tournament_draws_l4118_411838

/-- Represents a chess tournament with a fixed number of participants. -/
structure ChessTournament where
  n : ℕ  -- number of participants
  lists : Fin n → Fin 12 → Set (Fin n)  -- lists[i][j] is the jth list of participant i
  
  list_rule_1 : ∀ i, lists i 0 = {i}
  list_rule_2 : ∀ i j, j > 0 → lists i j ⊇ lists i (j-1)
  list_rule_12 : ∀ i, lists i 11 ≠ lists i 10

/-- The number of draws in the tournament. -/
def num_draws (t : ChessTournament) : ℕ :=
  (t.n.choose 2) - t.n

theorem chess_tournament_draws (t : ChessTournament) (h : t.n = 12) : 
  num_draws t = 54 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_draws_l4118_411838


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_representation_l4118_411888

theorem quadratic_form_ratio_representation (x y u v : ℤ) :
  (∃ k : ℤ, (x^2 + 3*y^2) = k * (u^2 + 3*v^2)) →
  ∃ a b : ℤ, (x^2 + 3*y^2) / (u^2 + 3*v^2) = a^2 + 3*b^2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_representation_l4118_411888


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l4118_411821

/-- Calculates the perimeter of a rectangular field given its width and length ratio. --/
def field_perimeter (width : ℝ) (length_ratio : ℝ) : ℝ :=
  2 * (width + length_ratio * width)

/-- Theorem: The perimeter of a rectangular field with width 50 meters and length 7/5 times its width is 240 meters. --/
theorem rectangular_field_perimeter :
  field_perimeter 50 (7/5) = 240 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l4118_411821


namespace NUMINAMATH_CALUDE_min_total_distance_l4118_411851

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 13 ∧ dist B C = 14 ∧ dist C A = 15

-- Define the total distance function
def TotalDistance (A B C P : ℝ × ℝ) : ℝ :=
  dist A P + 5 * dist B P + 4 * dist C P

-- State the theorem
theorem min_total_distance (A B C : ℝ × ℝ) (h : Triangle A B C) :
  ∀ P : ℝ × ℝ, TotalDistance A B C P ≥ 69 ∧
  (TotalDistance A B C B = 69) :=
by sorry

end NUMINAMATH_CALUDE_min_total_distance_l4118_411851


namespace NUMINAMATH_CALUDE_hypotenuse_length_l4118_411862

-- Define a right triangle with side lengths a, b, and c
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : c^2 = a^2 + b^2

-- Define the theorem
theorem hypotenuse_length (t : RightTriangle) 
  (h : Real.sqrt ((t.a - 3)^2 + (t.b - 2)^2) = 0) :
  t.c = 3 ∨ t.c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l4118_411862


namespace NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l4118_411882

theorem half_plus_five_equals_fifteen (n : ℝ) : (1/2) * n + 5 = 15 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l4118_411882


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_domain_eq_reals_l4118_411893

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

theorem domain_eq_reals : Set.range (fun x => |x|) = Set.range (fun x => Real.sqrt (x^2)) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_domain_eq_reals_l4118_411893


namespace NUMINAMATH_CALUDE_remainder_problem_l4118_411896

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1259 % d = r) (h3 : 1567 % d = r) (h4 : 2257 % d = r) : 
  d - r = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4118_411896


namespace NUMINAMATH_CALUDE_linda_spent_25_dollars_l4118_411816

/-- The amount Linda spent on her purchases -/
def linda_total_spent (coloring_book_price : ℚ) (coloring_book_quantity : ℕ)
  (peanut_pack_price : ℚ) (peanut_pack_quantity : ℕ) (stuffed_animal_price : ℚ) : ℚ :=
  coloring_book_price * coloring_book_quantity +
  peanut_pack_price * peanut_pack_quantity +
  stuffed_animal_price

theorem linda_spent_25_dollars :
  linda_total_spent 4 2 (3/2) 4 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_linda_spent_25_dollars_l4118_411816


namespace NUMINAMATH_CALUDE_store_comparison_l4118_411854

/-- The number of soccer balls to be purchased -/
def soccer_balls : ℕ := 100

/-- The cost of each soccer ball in yuan -/
def soccer_ball_cost : ℕ := 200

/-- The cost of each basketball in yuan -/
def basketball_cost : ℕ := 80

/-- The cost function for Store A's discount plan -/
def cost_A (x : ℕ) : ℕ := 
  if x ≤ soccer_balls then soccer_balls * soccer_ball_cost
  else soccer_balls * soccer_ball_cost + basketball_cost * (x - soccer_balls)

/-- The cost function for Store B's discount plan -/
def cost_B (x : ℕ) : ℕ := 
  (soccer_balls * soccer_ball_cost + x * basketball_cost) * 4 / 5

theorem store_comparison (x : ℕ) :
  (x = 100 → cost_A x < cost_B x) ∧
  (x > 100 → cost_A x = 80 * x + 12000 ∧ cost_B x = 64 * x + 16000) ∧
  (x = 300 → min (cost_A x) (cost_B x) > 
    cost_A 100 + cost_B 200) := by sorry

#eval cost_A 100
#eval cost_B 100
#eval cost_A 300
#eval cost_B 300
#eval cost_A 100 + cost_B 200

end NUMINAMATH_CALUDE_store_comparison_l4118_411854


namespace NUMINAMATH_CALUDE_problem_solution_l4118_411827

theorem problem_solution (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*x^2 + 1
  let g : ℝ → ℝ := λ x ↦ -x^3 + 3*x^2 + x - 7
  (f x + g x = x - 6) → (g x = -x^3 + 3*x^2 + x - 7) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4118_411827


namespace NUMINAMATH_CALUDE_blanch_lunch_slices_l4118_411880

/-- The number of pizza slices Blanch ate during lunch -/
def lunch_slices (initial : ℕ) (breakfast : ℕ) (snack : ℕ) (dinner : ℕ) (remaining : ℕ) : ℕ :=
  initial - breakfast - snack - dinner - remaining

/-- Theorem stating that Blanch ate 2 slices during lunch -/
theorem blanch_lunch_slices :
  lunch_slices 15 4 2 5 2 = 2 := by sorry

end NUMINAMATH_CALUDE_blanch_lunch_slices_l4118_411880


namespace NUMINAMATH_CALUDE_min_value_xy_l4118_411889

theorem min_value_xy (x y : ℝ) (h : x > 0 ∧ y > 0) (eq : 1/x + 2/y = Real.sqrt (x*y)) : 
  x * y ≥ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_l4118_411889


namespace NUMINAMATH_CALUDE_students_in_class_g_l4118_411886

theorem students_in_class_g (total_students : ℕ) (class_a class_b class_c class_d class_e class_f class_g : ℕ) : 
  total_students = 1500 ∧
  class_a = 188 ∧
  class_b = 115 ∧
  class_c = class_b + 80 ∧
  class_d = 2 * class_b ∧
  class_e = class_a + class_b ∧
  class_f = (class_c + class_d) / 2 ∧
  class_g = total_students - (class_a + class_b + class_c + class_d + class_e + class_f) →
  class_g = 256 :=
by sorry

end NUMINAMATH_CALUDE_students_in_class_g_l4118_411886


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4118_411836

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4118_411836


namespace NUMINAMATH_CALUDE_parallelogram_reflection_l4118_411883

/-- Reflect a point across the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflect a point across the line y = -x -/
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)

/-- The final position of point C after two reflections -/
def final_position (C : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_neg_x (reflect_x_axis C)

theorem parallelogram_reflection :
  let A : ℝ × ℝ := (2, 5)
  let B : ℝ × ℝ := (4, 9)
  let C : ℝ × ℝ := (6, 5)
  let D : ℝ × ℝ := (4, 1)
  final_position C = (5, -6) := by sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_l4118_411883


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4118_411871

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2*b)/c + (2*a + c)/b + (b + 3*c)/a ≥ 6 * 12^(1/6) :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (a + 2*b)/c + (2*a + c)/b + (b + 3*c)/a = 6 * 12^(1/6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4118_411871


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l4118_411895

theorem trigonometric_inequality (x : ℝ) : 
  0 ≤ 5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ∧
  5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ≤ 18 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l4118_411895


namespace NUMINAMATH_CALUDE_hash_eight_three_l4118_411849

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- State the theorem
theorem hash_eight_three : hash 8 3 = 127 :=
  by
    sorry

-- Define the conditions
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + 2 * s + 1

end NUMINAMATH_CALUDE_hash_eight_three_l4118_411849


namespace NUMINAMATH_CALUDE_cos_sin_180_degrees_l4118_411840

theorem cos_sin_180_degrees :
  Real.cos (180 * π / 180) = -1 ∧ Real.sin (180 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_180_degrees_l4118_411840


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4118_411897

theorem imaginary_part_of_complex_fraction : 
  Complex.im (2 * Complex.I^3 / (1 - Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4118_411897


namespace NUMINAMATH_CALUDE_largest_difference_is_62_l4118_411823

/-- Given a list of four digits, returns the largest 2-digit number that can be formed --/
def largest_two_digit (digits : List Nat) : Nat :=
  sorry

/-- Given a list of four digits, returns the smallest 2-digit number that can be formed --/
def smallest_two_digit (digits : List Nat) : Nat :=
  sorry

/-- The set of digits to be used --/
def digit_set : List Nat := [2, 4, 6, 8]

theorem largest_difference_is_62 :
  largest_two_digit digit_set - smallest_two_digit digit_set = 62 :=
sorry

end NUMINAMATH_CALUDE_largest_difference_is_62_l4118_411823


namespace NUMINAMATH_CALUDE_max_d_value_l4118_411802

/-- Represents a 6-digit number of the form 6d6,33f -/
def sixDigitNumber (d f : ℕ) : ℕ := 600000 + 10000*d + 3300 + f

/-- Predicate for d and f being single digits -/
def areSingleDigits (d f : ℕ) : Prop := d < 10 ∧ f < 10

/-- Predicate for the number being divisible by 33 -/
def isDivisibleBy33 (d f : ℕ) : Prop :=
  (sixDigitNumber d f) % 33 = 0

theorem max_d_value :
  ∃ (d : ℕ), 
    (∃ (f : ℕ), areSingleDigits d f ∧ isDivisibleBy33 d f) ∧
    (∀ (d' f' : ℕ), areSingleDigits d' f' → isDivisibleBy33 d' f' → d' ≤ d) ∧
    d = 1 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l4118_411802


namespace NUMINAMATH_CALUDE_fixed_cost_calculation_l4118_411800

/-- The fixed cost for producing products given total cost, marginal cost, and number of products. -/
theorem fixed_cost_calculation (total_cost marginal_cost : ℝ) (n : ℕ) :
  total_cost = 16000 →
  marginal_cost = 200 →
  n = 20 →
  total_cost = (marginal_cost * n) + 12000 :=
by sorry

end NUMINAMATH_CALUDE_fixed_cost_calculation_l4118_411800


namespace NUMINAMATH_CALUDE_max_intersection_area_is_zero_l4118_411824

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a right prism with an equilateral triangle base -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ
  baseVertices : List Point3D

/-- Calculates the area of intersection between a plane and a right prism -/
def intersectionArea (prism : RightPrism) (plane : Plane) : ℝ :=
  sorry

/-- The theorem stating that the maximum area of intersection is 0 -/
theorem max_intersection_area_is_zero (h : ℝ) (s : ℝ) (A B C : Point3D) :
  h = 5 →
  s = 6 →
  A = ⟨3, 0, 0⟩ →
  B = ⟨-3, 0, 0⟩ →
  C = ⟨0, 3 * Real.sqrt 3, 0⟩ →
  let prism : RightPrism := ⟨h, s, [A, B, C]⟩
  let plane : Plane := ⟨2, -3, 6, 30⟩
  intersectionArea prism plane = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersection_area_is_zero_l4118_411824


namespace NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l4118_411860

/-- Given two points A and B in a 2D plane, where:
  - A is at (0, 0)
  - B is on the line y = 5
  - The slope of segment AB is 3/4
  Prove that the sum of the x- and y-coordinates of point B is 35/3 -/
theorem coordinate_sum_of_point_B (B : ℝ × ℝ) : 
  B.2 = 5 ∧ 
  (B.2 - 0) / (B.1 - 0) = 3 / 4 → 
  B.1 + B.2 = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l4118_411860


namespace NUMINAMATH_CALUDE_alice_gives_no_stickers_to_charlie_l4118_411804

/-- Represents the sticker distribution problem --/
def sticker_distribution (c : ℕ) : Prop :=
  let alice_initial := 12 * c
  let bob_initial := 3 * c
  let charlie_initial := c
  let dave_initial := c
  let alice_final := alice_initial - (2 * c - bob_initial) - (3 * c - dave_initial)
  let bob_final := 2 * c
  let charlie_final := c
  let dave_final := 3 * c
  (alice_final - alice_initial) / alice_initial = 0

/-- Theorem stating that Alice gives 0 fraction of her stickers to Charlie --/
theorem alice_gives_no_stickers_to_charlie (c : ℕ) (hc : c > 0) :
  sticker_distribution c :=
sorry

end NUMINAMATH_CALUDE_alice_gives_no_stickers_to_charlie_l4118_411804


namespace NUMINAMATH_CALUDE_no_right_triangle_with_sides_x_2x_3x_l4118_411864

theorem no_right_triangle_with_sides_x_2x_3x :
  ¬ ∃ (x : ℝ), x > 0 ∧ x^2 + (2*x)^2 = (3*x)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_sides_x_2x_3x_l4118_411864


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4118_411850

def A : Set ℝ := {x | x^2 - 1 ≤ 0}
def B : Set ℝ := {x | x ≠ 0 ∧ (x - 2) / x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4118_411850


namespace NUMINAMATH_CALUDE_berry_temperature_theorem_l4118_411892

theorem berry_temperature_theorem (temps : List Float) (avg : Float) : 
  temps.length = 6 ∧ 
  temps = [99.1, 98.2, 98.7, 99.3, 99, 98.9] ∧ 
  avg = 99 →
  (temps.sum + 99.8) / 7 = avg :=
by sorry

end NUMINAMATH_CALUDE_berry_temperature_theorem_l4118_411892


namespace NUMINAMATH_CALUDE_largest_number_in_sample_l4118_411878

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (total : ℕ) (sample_size : ℕ) (first_sample : ℕ) : ℕ :=
  first_sample + (sample_size - 1) * (total / sample_size)

/-- Theorem stating the largest number in the specific systematic sample -/
theorem largest_number_in_sample :
  largest_sample_number 120 10 7 = 115 := by
  sorry

#eval largest_sample_number 120 10 7

end NUMINAMATH_CALUDE_largest_number_in_sample_l4118_411878


namespace NUMINAMATH_CALUDE_positive_trig_expressions_l4118_411873

theorem positive_trig_expressions :
  (Real.sin (305 * π / 180) * Real.cos (460 * π / 180) > 0) ∧
  (Real.cos (378 * π / 180) * Real.sin (1100 * π / 180) > 0) ∧
  (Real.tan (188 * π / 180) * Real.cos (158 * π / 180) ≤ 0) ∧
  (Real.tan (400 * π / 180) * Real.tan (470 * π / 180) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_positive_trig_expressions_l4118_411873


namespace NUMINAMATH_CALUDE_fraction_evaluation_l4118_411814

theorem fraction_evaluation : (16 + 8) / (4 - 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l4118_411814


namespace NUMINAMATH_CALUDE_circle_center_l4118_411879

/-- The equation of a circle in the form x^2 - 6x + y^2 + 2y = 9 -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y = 9

/-- The center of a circle -/
def CircleCenter (h k : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x - h)^2 + (y - k)^2 = 19

/-- Theorem: The center of the circle with equation x^2 - 6x + y^2 + 2y = 9 is (3, -1) -/
theorem circle_center :
  CircleCenter 3 (-1) CircleEquation :=
sorry

end NUMINAMATH_CALUDE_circle_center_l4118_411879


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l4118_411861

/-- Given a line L1 with equation x - 2y - 2 = 0 and a point P(1, 0),
    the line L2 passing through P and perpendicular to L1 has the equation 2x + y - 2 = 0 -/
theorem perpendicular_line_equation (L1 : (ℝ × ℝ) → Prop) (P : ℝ × ℝ) :
  L1 = λ (x, y) => x - 2*y - 2 = 0 →
  P = (1, 0) →
  ∃ (L2 : (ℝ × ℝ) → Prop),
    (∀ (x y : ℝ), L2 (x, y) ↔ 2*x + y - 2 = 0) ∧
    L2 P ∧
    (∀ (v w : ℝ × ℝ), L1 v ∧ L1 w → L2 v ∧ L2 w →
      (v.1 - w.1) * (v.1 - w.1) + (v.2 - w.2) * (v.2 - w.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l4118_411861


namespace NUMINAMATH_CALUDE_y_exceeds_x_l4118_411876

theorem y_exceeds_x (x y : ℝ) (h : x = 0.75 * y) : (y - x) / x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_y_exceeds_x_l4118_411876


namespace NUMINAMATH_CALUDE_statement_consistency_l4118_411806

def Statement : Type := Bool

def statementA (a b c d e : Statement) : Prop :=
  (a = true ∨ b = true ∨ c = true ∨ d = true ∨ e = true) ∧
  ¬(a = true ∧ b = true) ∧ ¬(a = true ∧ c = true) ∧ ¬(a = true ∧ d = true) ∧ ¬(a = true ∧ e = true) ∧
  ¬(b = true ∧ c = true) ∧ ¬(b = true ∧ d = true) ∧ ¬(b = true ∧ e = true) ∧
  ¬(c = true ∧ d = true) ∧ ¬(c = true ∧ e = true) ∧
  ¬(d = true ∧ e = true)

def statementC (a b c d e : Statement) : Prop :=
  a = true ∧ b = true ∧ c = true ∧ d = true ∧ e = true

def statementE (a : Statement) : Prop :=
  a = true

theorem statement_consistency :
  ∀ (a b c d e : Statement),
  (statementA a b c d e ↔ a = true) →
  (statementC a b c d e ↔ c = true) →
  (statementE a ↔ e = true) →
  (a = false ∧ b = true ∧ c = false ∧ d = true ∧ e = false) :=
by sorry

end NUMINAMATH_CALUDE_statement_consistency_l4118_411806


namespace NUMINAMATH_CALUDE_perpendicular_to_same_plane_implies_parallel_perpendicular_to_two_planes_implies_parallel_l4118_411868

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem perpendicular_to_same_plane_implies_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel_lines m n :=
sorry

-- Theorem 2: If a line is perpendicular to two planes, then those planes are parallel
theorem perpendicular_to_two_planes_implies_parallel 
  (n : Line) (α β : Plane) :
  perpendicular n α → perpendicular n β → parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_same_plane_implies_parallel_perpendicular_to_two_planes_implies_parallel_l4118_411868


namespace NUMINAMATH_CALUDE_smallest_abcd_l4118_411826

/-- Represents a four-digit number ABCD -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_range : a ∈ Finset.range 10
  b_range : b ∈ Finset.range 10
  c_range : c ∈ Finset.range 10
  d_range : d ∈ Finset.range 10
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts a FourDigitNumber to its numerical value -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  ab : Nat
  a : Nat
  b : Nat
  ab_two_digit : ab ∈ Finset.range 100
  ab_eq : ab = 10 * a + b
  a_not_eq_b : a ≠ b
  result : FourDigitNumber
  multiplication_condition : ab * a = result.toNat

/-- The main theorem stating that the smallest ABCD satisfying the conditions is 2046 -/
theorem smallest_abcd (conditions : ProblemConditions) :
  ∀ other : FourDigitNumber,
    (∃ other_conditions : ProblemConditions, other_conditions.result = other) →
    conditions.result.toNat ≤ other.toNat ∧ conditions.result.toNat = 2046 := by
  sorry


end NUMINAMATH_CALUDE_smallest_abcd_l4118_411826


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l4118_411863

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : p ≠ r) (h2 : k ≠ 0) :
  k * p^2 - k * r^2 = 4 * (k * p - k * r) → p + r = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l4118_411863


namespace NUMINAMATH_CALUDE_inequality_proof_l4118_411801

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a^2 + 3 * b^2) + (b * c) / (b^2 + 3 * c^2) + (c * a) / (c^2 + 3 * a^2) ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4118_411801


namespace NUMINAMATH_CALUDE_trajectory_equation_l4118_411842

/-- 
Given a point P(x, y) in the Cartesian coordinate system,
if the product of its distances to the x-axis and y-axis equals 1,
then the equation of its trajectory is xy = ± 1.
-/
theorem trajectory_equation (x y : ℝ) : 
  (|x| * |y| = 1) → (x * y = 1 ∨ x * y = -1) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l4118_411842


namespace NUMINAMATH_CALUDE_train_crossing_time_l4118_411809

/-- Proves that the time taken for the first train to cross a telegraph post is 10 seconds,
    given the conditions of the problem. -/
theorem train_crossing_time
  (train_length : ℝ)
  (second_train_time : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : second_train_time = 15)
  (h3 : crossing_time = 12) :
  let second_train_speed := train_length / second_train_time
  let relative_speed := 2 * train_length / crossing_time
  let first_train_speed := relative_speed - second_train_speed
  train_length / first_train_speed = 10 :=
by sorry


end NUMINAMATH_CALUDE_train_crossing_time_l4118_411809


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_iff_m_eq_7_or_neg_1_l4118_411859

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number k such that
    ax^2 + bx + c = (√a * x + k)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + k)^2

/-- The main theorem stating that m = 7 or m = -1 if and only if
    x^2 + 2(m-3)x + 16 is a perfect square trinomial -/
theorem perfect_square_trinomial_iff_m_eq_7_or_neg_1 :
  ∀ m : ℝ, (m = 7 ∨ m = -1) ↔ is_perfect_square_trinomial 1 (2*(m-3)) 16 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_iff_m_eq_7_or_neg_1_l4118_411859


namespace NUMINAMATH_CALUDE_number_of_classes_l4118_411841

theorem number_of_classes (single_sided_per_class_per_day : ℕ)
                          (double_sided_per_class_per_day : ℕ)
                          (school_days_per_week : ℕ)
                          (total_single_sided_per_week : ℕ)
                          (total_double_sided_per_week : ℕ)
                          (h1 : single_sided_per_class_per_day = 175)
                          (h2 : double_sided_per_class_per_day = 75)
                          (h3 : school_days_per_week = 5)
                          (h4 : total_single_sided_per_week = 16000)
                          (h5 : total_double_sided_per_week = 7000) :
  ⌊(total_single_sided_per_week + total_double_sided_per_week : ℚ) /
   ((single_sided_per_class_per_day + double_sided_per_class_per_day) * school_days_per_week)⌋ = 18 :=
by sorry

end NUMINAMATH_CALUDE_number_of_classes_l4118_411841


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l4118_411810

theorem quadratic_solution_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ x^2 - x - (m + 1) = 0) →
  m ∈ Set.Icc (-5/4) 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l4118_411810


namespace NUMINAMATH_CALUDE_complex_modulus_range_l4118_411853

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z = a + Complex.I) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l4118_411853


namespace NUMINAMATH_CALUDE_ratio_problem_l4118_411874

theorem ratio_problem (p q r s : ℚ) 
  (h1 : p / q = 4)
  (h2 : q / r = 3)
  (h3 : r / s = 1 / 5) :
  s / p = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l4118_411874


namespace NUMINAMATH_CALUDE_perp_condition_for_parallel_l4118_411872

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the given lines and planes
variable (a b : Line) (α β : Plane)

-- State the theorem
theorem perp_condition_for_parallel 
  (h1 : perp a α) 
  (h2 : subset b β) :
  (∀ α β, parallel α β → perpLine a b) ∧ 
  (∃ α β, perpLine a b ∧ ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_perp_condition_for_parallel_l4118_411872


namespace NUMINAMATH_CALUDE_baseball_cards_packs_l4118_411833

/-- The number of people who bought baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person bought -/
def cards_per_person : ℕ := 540

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 20

/-- The total number of packs of baseball cards for all people -/
def total_packs : ℕ := (num_people * cards_per_person) / cards_per_pack

theorem baseball_cards_packs : total_packs = 108 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_packs_l4118_411833


namespace NUMINAMATH_CALUDE_bicycle_speed_problem_l4118_411829

/-- Proves that given a total distance of 350 km, where the first 200 km is traveled at 20 km/h,
    and the average speed for the entire trip is 17.5 km/h, the speed for the remaining distance is 15 km/h. -/
theorem bicycle_speed_problem (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 350 →
  first_part_distance = 200 →
  first_part_speed = 20 →
  average_speed = 17.5 →
  (total_distance - first_part_distance) / ((total_distance / average_speed) - (first_part_distance / first_part_speed)) = 15 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_speed_problem_l4118_411829


namespace NUMINAMATH_CALUDE_a_divides_next_squared_plus_next_plus_one_l4118_411805

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 5 * a (n + 1) - a n - 1

theorem a_divides_next_squared_plus_next_plus_one :
  ∀ n : ℕ, (a n) ∣ ((a (n + 1))^2 + a (n + 1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_a_divides_next_squared_plus_next_plus_one_l4118_411805


namespace NUMINAMATH_CALUDE_second_month_sale_l4118_411815

def sale_month1 : ℕ := 7435
def sale_month3 : ℕ := 7855
def sale_month4 : ℕ := 8230
def sale_month5 : ℕ := 7560
def sale_month6 : ℕ := 6000
def average_sale : ℕ := 7500
def num_months : ℕ := 6

theorem second_month_sale :
  sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6 + 7920 = average_sale * num_months :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l4118_411815


namespace NUMINAMATH_CALUDE_min_sum_inequality_min_sum_achievable_l4118_411820

theorem min_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (12 * a)) + ((a + b + c) / (5 * a * b * c)) ≥ 4 / (360 ^ (1/4 : ℝ)) :=
sorry

theorem min_sum_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b)) + (b / (6 * c)) + (c / (12 * a)) + ((a + b + c) / (5 * a * b * c)) = 4 / (360 ^ (1/4 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_inequality_min_sum_achievable_l4118_411820


namespace NUMINAMATH_CALUDE_perfect_square_difference_l4118_411819

theorem perfect_square_difference (a b c : ℕ) 
  (h1 : Nat.gcd a (Nat.gcd b c) = 1)
  (h2 : a * b = c * (a - b)) : 
  ∃ (k : ℕ), a - b = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l4118_411819


namespace NUMINAMATH_CALUDE_car_motorcycle_transaction_result_l4118_411870

theorem car_motorcycle_transaction_result :
  let car_selling_price : ℚ := 20000
  let motorcycle_selling_price : ℚ := 10000
  let car_loss_percentage : ℚ := 25 / 100
  let motorcycle_gain_percentage : ℚ := 25 / 100
  let car_cost : ℚ := car_selling_price / (1 - car_loss_percentage)
  let motorcycle_cost : ℚ := motorcycle_selling_price / (1 + motorcycle_gain_percentage)
  let total_cost : ℚ := car_cost + motorcycle_cost
  let total_selling_price : ℚ := car_selling_price + motorcycle_selling_price
  let transaction_result : ℚ := total_cost - total_selling_price
  transaction_result = 4667 / 1 := by sorry

end NUMINAMATH_CALUDE_car_motorcycle_transaction_result_l4118_411870


namespace NUMINAMATH_CALUDE_youth_entertainment_suitable_for_sampling_other_scenarios_not_suitable_for_sampling_l4118_411839

/-- Represents a survey scenario -/
inductive SurveyScenario
| CompanyHealthCheck
| EpidemicTemperatureCheck
| YouthEntertainment
| AirplaneSecurity

/-- Determines if a survey scenario is suitable for sampling -/
def isSuitableForSampling (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.YouthEntertainment => True
  | _ => False

/-- Theorem stating that the youth entertainment survey is suitable for sampling -/
theorem youth_entertainment_suitable_for_sampling :
  isSuitableForSampling SurveyScenario.YouthEntertainment :=
by sorry

/-- Theorem stating that other scenarios are not suitable for sampling -/
theorem other_scenarios_not_suitable_for_sampling (scenario : SurveyScenario) :
  scenario ≠ SurveyScenario.YouthEntertainment →
  ¬ (isSuitableForSampling scenario) :=
by sorry

end NUMINAMATH_CALUDE_youth_entertainment_suitable_for_sampling_other_scenarios_not_suitable_for_sampling_l4118_411839


namespace NUMINAMATH_CALUDE_simplify_expression_l4118_411848

theorem simplify_expression :
  3 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 3 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4118_411848


namespace NUMINAMATH_CALUDE_brownie_division_l4118_411867

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a brownie tray with its dimensions -/
def tray : Dimensions := ⟨24, 30⟩

/-- Represents a single brownie piece with its dimensions -/
def piece : Dimensions := ⟨3, 4⟩

/-- Theorem stating that the tray can be divided into exactly 60 brownie pieces -/
theorem brownie_division :
  (area tray) / (area piece) = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_division_l4118_411867


namespace NUMINAMATH_CALUDE_triangle_area_l4118_411808

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  A = 30 * π / 180 →
  B = 60 * π / 180 →
  C = π - A - B →
  b = a * Real.sin B / Real.sin A →
  (1/2) * a * b * Real.sin C = (9 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4118_411808


namespace NUMINAMATH_CALUDE_max_comic_books_l4118_411877

theorem max_comic_books (cost : ℚ) (budget : ℚ) (h1 : cost = 25/20) (h2 : budget = 10) :
  ⌊budget / cost⌋ = 8 := by
sorry

end NUMINAMATH_CALUDE_max_comic_books_l4118_411877


namespace NUMINAMATH_CALUDE_stock_transaction_profit_l4118_411846

theorem stock_transaction_profit
  (initial_price : ℝ)
  (profit_percentage : ℝ)
  (loss_percentage : ℝ)
  (final_sale_percentage : ℝ)
  (h1 : initial_price = 1000)
  (h2 : profit_percentage = 0.1)
  (h3 : loss_percentage = 0.1)
  (h4 : final_sale_percentage = 0.9) :
  let first_sale_price := initial_price * (1 + profit_percentage)
  let second_sale_price := first_sale_price * (1 - loss_percentage)
  let final_sale_price := second_sale_price * final_sale_percentage
  final_sale_price - initial_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_stock_transaction_profit_l4118_411846


namespace NUMINAMATH_CALUDE_test_score_calculation_l4118_411817

theorem test_score_calculation (total_questions : ℕ) (first_half : ℕ) (second_half : ℕ)
  (first_correct_rate : ℚ) (second_correct_rate : ℚ)
  (h1 : total_questions = 80)
  (h2 : first_half = 40)
  (h3 : second_half = 40)
  (h4 : first_correct_rate = 9/10)
  (h5 : second_correct_rate = 19/20)
  (h6 : total_questions = first_half + second_half) :
  ⌊first_correct_rate * first_half⌋ + ⌊second_correct_rate * second_half⌋ = 74 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l4118_411817


namespace NUMINAMATH_CALUDE_clay_target_sequences_l4118_411856

theorem clay_target_sequences (n : ℕ) (a b c : ℕ) 
  (h1 : n = 8) 
  (h2 : a = 3) 
  (h3 : b = 3) 
  (h4 : c = 2) 
  (h5 : a + b + c = n) : 
  (Nat.factorial n) / (Nat.factorial a * Nat.factorial b * Nat.factorial c) = 560 :=
by sorry

end NUMINAMATH_CALUDE_clay_target_sequences_l4118_411856


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l4118_411855

theorem vector_perpendicular_condition (k : ℝ) : 
  let a : ℝ × ℝ := (-1, k)
  let b : ℝ × ℝ := (3, 1)
  (a.1 + b.1, a.2 + b.2) • a = 0 → k = -2 ∨ k = 1 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l4118_411855


namespace NUMINAMATH_CALUDE_mikes_initial_amount_solve_mikes_initial_amount_l4118_411843

/-- Proves that Mike's initial amount is $90 given the conditions of the problem -/
theorem mikes_initial_amount (carol_initial : ℕ) (carol_weekly_savings : ℕ) 
  (mike_weekly_savings : ℕ) (weeks : ℕ) (mike_initial : ℕ) : Prop :=
  carol_initial = 60 →
  carol_weekly_savings = 9 →
  mike_weekly_savings = 3 →
  weeks = 5 →
  carol_initial + carol_weekly_savings * weeks = mike_initial + mike_weekly_savings * weeks →
  mike_initial = 90

/-- The main theorem that proves Mike's initial amount -/
theorem solve_mikes_initial_amount : 
  ∃ (mike_initial : ℕ), mikes_initial_amount 60 9 3 5 mike_initial :=
by
  sorry

end NUMINAMATH_CALUDE_mikes_initial_amount_solve_mikes_initial_amount_l4118_411843


namespace NUMINAMATH_CALUDE_quiz_show_probability_l4118_411869

-- Define the number of questions and choices
def num_questions : ℕ := 4
def num_choices : ℕ := 4

-- Define the minimum number of correct answers needed to win
def min_correct : ℕ := 3

-- Define the probability of guessing a single question correctly
def prob_correct : ℚ := 1 / num_choices

-- Define the probability of guessing a single question incorrectly
def prob_incorrect : ℚ := 1 - prob_correct

-- Define the function to calculate the probability of winning
def prob_win : ℚ :=
  (num_questions.choose min_correct) * (prob_correct ^ min_correct) * (prob_incorrect ^ (num_questions - min_correct)) +
  (prob_correct ^ num_questions)

-- State the theorem
theorem quiz_show_probability :
  prob_win = 13 / 256 := by sorry

end NUMINAMATH_CALUDE_quiz_show_probability_l4118_411869


namespace NUMINAMATH_CALUDE_leona_earnings_l4118_411811

/-- Given an hourly rate calculated from earning $24.75 for 3 hours,
    prove that the earnings for 5 hours at the same rate will be $41.25. -/
theorem leona_earnings (hourly_rate : ℝ) (h1 : hourly_rate * 3 = 24.75) :
  hourly_rate * 5 = 41.25 := by
  sorry

end NUMINAMATH_CALUDE_leona_earnings_l4118_411811


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l4118_411837

theorem smallest_n_congruence (n : ℕ) : n = 3 ↔ (
  n > 0 ∧
  17 * n ≡ 136 [ZMOD 5] ∧
  ∀ m : ℕ, m > 0 → m < n → ¬(17 * m ≡ 136 [ZMOD 5])
) := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l4118_411837


namespace NUMINAMATH_CALUDE_arithmetic_sum_odd_sequence_l4118_411875

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem arithmetic_sum_odd_sequence :
  let seq := arithmetic_sequence 1 2 11
  (∀ x ∈ seq, is_odd x) ∧
  (seq.length = 11) ∧
  (seq.getLast? = some 21) →
  seq.sum = 121 := by
  sorry

#eval arithmetic_sequence 1 2 11

end NUMINAMATH_CALUDE_arithmetic_sum_odd_sequence_l4118_411875


namespace NUMINAMATH_CALUDE_line_equation_l4118_411845

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Point M -/
def M : ℝ × ℝ := (1, 1)

/-- A line passing through point (1, 1) -/
def line_through_M (m : ℝ) (x y : ℝ) : Prop := x = m * (y - 1) + 1

/-- The line intersects the ellipse at two points -/
def intersects_twice (m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    ellipse (m * (y₁ - 1) + 1) y₁ ∧
    ellipse (m * (y₂ - 1) + 1) y₂

/-- M is the midpoint of the line segment AB -/
def M_is_midpoint (m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    ellipse (m * (y₁ - 1) + 1) y₁ ∧
    ellipse (m * (y₂ - 1) + 1) y₂ ∧
    (y₁ + y₂) / 2 = 1

/-- The main theorem -/
theorem line_equation :
  ∃ m : ℝ, ellipse M.1 M.2 ∧
    intersects_twice m ∧
    M_is_midpoint m ∧
    ∀ x y : ℝ, line_through_M m x y ↔ 3 * x + 4 * y - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l4118_411845


namespace NUMINAMATH_CALUDE_not_all_odd_divisible_by_3_l4118_411898

theorem not_all_odd_divisible_by_3 : ¬ (∀ n : ℕ, Odd n → 3 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_not_all_odd_divisible_by_3_l4118_411898


namespace NUMINAMATH_CALUDE_circle_center_proof_l4118_411834

/-- A line in the 2D plane represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a line --/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 = l.c

/-- Check if a circle is tangent to a line --/
def circleTangentToLine (c : Circle) (l : Line) : Prop :=
  abs (l.a * c.center.1 + l.b * c.center.2 - l.c) = c.radius * (l.a^2 + l.b^2).sqrt

theorem circle_center_proof :
  let line1 : Line := { a := 5, b := -2, c := 40 }
  let line2 : Line := { a := 5, b := -2, c := 10 }
  let line3 : Line := { a := 3, b := -4, c := 0 }
  let center : ℝ × ℝ := (50/7, 75/14)
  ∃ (r : ℝ), 
    let c : Circle := { center := center, radius := r }
    circleTangentToLine c line1 ∧ 
    circleTangentToLine c line2 ∧ 
    pointOnLine center line3 := by
  sorry


end NUMINAMATH_CALUDE_circle_center_proof_l4118_411834


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l4118_411899

/-- Converts a repeating decimal with a single repeating digit to a rational number -/
def repeating_decimal_to_rational (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  repeating_decimal_to_rational 6 + 
  repeating_decimal_to_rational 2 - 
  repeating_decimal_to_rational 4 + 
  repeating_decimal_to_rational 9 = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l4118_411899


namespace NUMINAMATH_CALUDE_leading_coeff_of_polynomial_l4118_411852

/-- Given a polynomial f such that f(x + 1) - f(x) = 8x^2 + 6x + 4 for all real x,
    the leading coefficient of f is 8/3 -/
theorem leading_coeff_of_polynomial (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 1) - f x = 8 * x^2 + 6 * x + 4) →
  ∃ (a b c d : ℝ), (∀ x : ℝ, f x = (8/3) * x^3 + a * x^2 + b * x + c) ∧ a ≠ (8/3) :=
by sorry

end NUMINAMATH_CALUDE_leading_coeff_of_polynomial_l4118_411852


namespace NUMINAMATH_CALUDE_cookie_distribution_l4118_411825

/-- Given 24 cookies, prove that 6 friends can share them if each friend receives 3 more cookies than the previous friend, with the first friend receiving at least 1 cookie. -/
theorem cookie_distribution (total_cookies : ℕ) (cookie_increment : ℕ) (n : ℕ) : 
  total_cookies = 24 →
  cookie_increment = 3 →
  (n : ℚ) * ((1 : ℚ) + (1 : ℚ) + (cookie_increment : ℚ) * ((n : ℚ) - 1)) / 2 = (total_cookies : ℚ) →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l4118_411825


namespace NUMINAMATH_CALUDE_twenty_four_shots_hit_ship_l4118_411803

/-- Represents a point on the grid -/
structure Point where
  x : Fin 10
  y : Fin 10

/-- Represents a 1x4 ship on the grid -/
structure Ship where
  start : Point
  horizontal : Bool

/-- The set of 24 shots -/
def shots : Set Point := sorry

/-- Predicate to check if a ship overlaps with a point -/
def shipOverlapsPoint (s : Ship) (p : Point) : Prop := sorry

theorem twenty_four_shots_hit_ship :
  ∀ s : Ship, ∃ p ∈ shots, shipOverlapsPoint s p := by sorry

end NUMINAMATH_CALUDE_twenty_four_shots_hit_ship_l4118_411803


namespace NUMINAMATH_CALUDE_max_value_problem_l4118_411894

theorem max_value_problem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 6 * y ≤ 12) : 
  ∀ a b : ℝ, 4 * a + 3 * b ≤ 10 → 3 * a + 6 * b ≤ 12 → 2 * x + y ≥ 2 * a + b :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l4118_411894


namespace NUMINAMATH_CALUDE_halfway_point_between_fractions_l4118_411812

theorem halfway_point_between_fractions :
  let a := (1 : ℚ) / 9
  let b := (1 : ℚ) / 11
  let midpoint := (a + b) / 2
  midpoint = 10 / 99 := by sorry

end NUMINAMATH_CALUDE_halfway_point_between_fractions_l4118_411812


namespace NUMINAMATH_CALUDE_bisecting_line_sum_l4118_411844

/-- A line that bisects the circumference of a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  bisects : ∀ (x y : ℝ), a * x + b * y + 1 = 0 → 
    (x^2 + y^2 + 2*x + 2*y - 1 = 0 → 
      ∃ (p q : ℝ), p^2 + q^2 + 2*p + 2*q - 1 = 0 ∧ 
        a * p + b * q + 1 = 0 ∧ (p, q) ≠ (x, y))

/-- Theorem: If a line ax + by + 1 = 0 bisects the circumference of 
    the circle x^2 + y^2 + 2x + 2y - 1 = 0, then a + b = 1 -/
theorem bisecting_line_sum (l : BisectingLine) : l.a + l.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l4118_411844


namespace NUMINAMATH_CALUDE_mississippi_permutations_count_l4118_411832

def mississippi_permutations : ℕ :=
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)

theorem mississippi_permutations_count : mississippi_permutations = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_permutations_count_l4118_411832


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l4118_411822

/-- Represents the number of female students in a stratified sample -/
def female_students_in_sample (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) : ℕ :=
  (total_female * sample_size) / (total_male + total_female)

/-- Theorem: In a stratified sampling by gender with 500 male students, 400 female students, 
    and a sample size of 45, the number of female students in the sample is 20 -/
theorem stratified_sample_female_count :
  female_students_in_sample 500 400 45 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_female_count_l4118_411822


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l4118_411858

theorem sum_of_a_and_c (a b c r : ℝ) 
  (sum_eq : a + b + c = 114)
  (product_eq : a * b * c = 46656)
  (b_eq : b = a * r)
  (c_eq : c = a * r^2) :
  a + c = 78 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l4118_411858


namespace NUMINAMATH_CALUDE_olivias_quarters_l4118_411831

theorem olivias_quarters (spent : ℕ) (left : ℕ) : spent = 4 → left = 7 → spent + left = 11 := by
  sorry

end NUMINAMATH_CALUDE_olivias_quarters_l4118_411831


namespace NUMINAMATH_CALUDE_fifteenth_triangular_less_than_square_l4118_411847

-- Define the triangular number function
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem statement
theorem fifteenth_triangular_less_than_square :
  triangular_number 15 < 15^2 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_less_than_square_l4118_411847


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l4118_411828

theorem circle_intersection_theorem (O₁ O₂ T A B : ℝ × ℝ) : 
  let d := Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2)
  let r₁ := 4
  let r₂ := 6
  d ≥ 6 →
  (∃ C : ℝ × ℝ, (C.1 - O₁.1)^2 + (C.2 - O₁.2)^2 = r₁^2 ∧ 
               (C.1 - O₂.1)^2 + (C.2 - O₂.2)^2 = r₂^2) →
  (A.1 - O₁.1)^2 + (A.2 - O₁.2)^2 = r₂^2 →
  (B.1 - O₁.1)^2 + (B.2 - O₁.2)^2 = r₁^2 →
  Real.sqrt ((A.1 - T.1)^2 + (A.2 - T.2)^2) = 
    1/3 * Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) →
  Real.sqrt ((B.1 - T.1)^2 + (B.2 - T.2)^2) = 
    2/3 * Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l4118_411828


namespace NUMINAMATH_CALUDE_multiples_of_six_or_eight_l4118_411807

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_of_six_or_eight (upper_bound : ℕ) (h : upper_bound = 151) : 
  (count_multiples upper_bound 6 + count_multiples upper_bound 8 - 2 * count_multiples upper_bound 24) = 31 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_six_or_eight_l4118_411807


namespace NUMINAMATH_CALUDE_train_speed_l4118_411865

/-- The speed of two trains crossing each other -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 120) (h2 : crossing_time = 16) :
  let relative_speed := 2 * train_length / crossing_time
  let train_speed := relative_speed / 2
  let train_speed_kmh := train_speed * 3.6
  train_speed_kmh = 27 := by sorry

end NUMINAMATH_CALUDE_train_speed_l4118_411865


namespace NUMINAMATH_CALUDE_even_function_implies_k_equals_one_l4118_411891

/-- A function f is even if f(-x) = f(x) for all x in its domain. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = kx^2 + (k-1)x + 2. -/
def f (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + (k - 1) * x + 2

/-- If f(x) = kx^2 + (k-1)x + 2 is an even function, then k = 1. -/
theorem even_function_implies_k_equals_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_k_equals_one_l4118_411891


namespace NUMINAMATH_CALUDE_pie_rows_theorem_l4118_411887

def pecan_pies : ℕ := 16
def apple_pies : ℕ := 14
def pies_per_row : ℕ := 5

theorem pie_rows_theorem : 
  (pecan_pies + apple_pies) / pies_per_row = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_rows_theorem_l4118_411887


namespace NUMINAMATH_CALUDE_total_salary_is_616_l4118_411830

/-- The salary of employee N in dollars per week -/
def salary_N : ℝ := 280

/-- The ratio of M's salary to N's salary -/
def salary_ratio : ℝ := 1.2

/-- The salary of employee M in dollars per week -/
def salary_M : ℝ := salary_ratio * salary_N

/-- The total amount paid to both employees per week -/
def total_salary : ℝ := salary_M + salary_N

theorem total_salary_is_616 : total_salary = 616 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_is_616_l4118_411830


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l4118_411890

/-- The function f(x) defined as x^2 + bx + 5 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 5

/-- Theorem stating that -3 is not in the range of f(x) if and only if b is in the open interval (-4√2, 4√2) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, f b x ≠ -3) ↔ b ∈ Set.Ioo (-4 * Real.sqrt 2) (4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l4118_411890


namespace NUMINAMATH_CALUDE_coin_container_total_l4118_411835

theorem coin_container_total : ∃ (x : ℕ),
  (x * 1 + x * 3 * 10 + x * 3 * 5 * 25) = 63000 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_container_total_l4118_411835


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4118_411885

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote of C
def asymptote (x y : ℝ) : Prop :=
  y = (Real.sqrt 5 / 2) * x

-- Define the ellipse that shares a focus with C
def ellipse (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y, asymptote x y → hyperbola a b x y) ∧
  (∃ x y, ellipse x y ∧ hyperbola a b x y) →
  a^2 = 4 ∧ b^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4118_411885


namespace NUMINAMATH_CALUDE_percent_difference_l4118_411857

theorem percent_difference (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.1 * y) : x - y = -10 := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_l4118_411857


namespace NUMINAMATH_CALUDE_range_of_f_l4118_411884

-- Define the function f
def f (x : ℝ) : ℝ := |x + 10| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-15) 25 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l4118_411884


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l4118_411866

theorem least_positive_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 1 ∧ 
  n % 4 = 2 ∧
  ∀ m : ℕ, m > 0 ∧ m % 2 = 0 ∧ m % 5 = 1 ∧ m % 4 = 2 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l4118_411866


namespace NUMINAMATH_CALUDE_max_value_function_l4118_411881

theorem max_value_function (t : ℝ) : (3^t - 4*t)*t / (9^t + t) ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_function_l4118_411881


namespace NUMINAMATH_CALUDE_function_lower_bound_l4118_411818

open Real

theorem function_lower_bound (x : ℝ) (h : x > 0) : Real.exp x - Real.log x > 2 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l4118_411818


namespace NUMINAMATH_CALUDE_total_shoes_l4118_411813

theorem total_shoes (brian_shoes : ℕ) (edward_shoes : ℕ) (jacob_shoes : ℕ) 
  (h1 : brian_shoes = 22)
  (h2 : edward_shoes = 3 * brian_shoes)
  (h3 : jacob_shoes = edward_shoes / 2) :
  brian_shoes + edward_shoes + jacob_shoes = 121 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l4118_411813
