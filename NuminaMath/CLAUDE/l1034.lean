import Mathlib

namespace two_numbers_sum_and_quotient_l1034_103429

theorem two_numbers_sum_and_quotient (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 432 → y / x = 5 → x = 72 ∧ y = 360 := by
  sorry

end two_numbers_sum_and_quotient_l1034_103429


namespace average_of_4_8_N_l1034_103405

theorem average_of_4_8_N (N : ℝ) (h : 7 < N ∧ N < 15) : 
  let avg := (4 + 8 + N) / 3
  avg = 7 ∨ avg = 9 := by
sorry

end average_of_4_8_N_l1034_103405


namespace intersection_of_P_and_Q_l1034_103499

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | |x - 1| ≤ 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 2, 3} := by
  sorry

end intersection_of_P_and_Q_l1034_103499


namespace first_discount_percentage_l1034_103421

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) 
  (second_discount : ℝ) (first_discount : ℝ) : 
  original_price = 200 →
  final_price = 152 →
  second_discount = 0.05 →
  final_price = original_price * (1 - first_discount) * (1 - second_discount) →
  first_discount = 0.20 := by
  sorry

#check first_discount_percentage

end first_discount_percentage_l1034_103421


namespace ordering_abc_l1034_103408

theorem ordering_abc (a b c : ℝ) : 
  a = 31/32 → b = Real.cos (1/4) → c = 4 * Real.sin (1/4) → c > b ∧ b > a := by sorry

end ordering_abc_l1034_103408


namespace algebraic_expression_value_l1034_103490

theorem algebraic_expression_value (x y : ℝ) (h : x - y - 3 = 0) :
  x^2 - y^2 - 6*y = 9 := by sorry

end algebraic_expression_value_l1034_103490


namespace square_difference_l1034_103415

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x - 2) * (x + 2) = 9797 := by
  sorry

end square_difference_l1034_103415


namespace subset_condition_l1034_103479

def P : Set ℝ := {x | x^2 ≠ 4}
def Q (a : ℝ) : Set ℝ := {x | a * x = 4}

theorem subset_condition (a : ℝ) : Q a ⊆ P ↔ a = 0 ∨ a = 2 ∨ a = -2 := by
  sorry

end subset_condition_l1034_103479


namespace vegetable_load_weight_l1034_103449

/-- Calculates the total weight of a load of vegetables given the weight of a crate, 
    the weight of a carton, and the number of crates and cartons. -/
def totalWeight (crateWeight cartonWeight : ℕ) (numCrates numCartons : ℕ) : ℕ :=
  crateWeight * numCrates + cartonWeight * numCartons

/-- Proves that the total weight of a specific load of vegetables is 96 kilograms. -/
theorem vegetable_load_weight :
  totalWeight 4 3 12 16 = 96 := by
  sorry

end vegetable_load_weight_l1034_103449


namespace line_parameterization_l1034_103440

/-- Given a line y = (3/4)x + 2 parameterized by [x; y] = [-8; s] + t[l; -6],
    prove that s = -4 and l = -8 -/
theorem line_parameterization (s l : ℝ) : 
  (∀ x y t : ℝ, y = (3/4) * x + 2 ↔ ∃ t, (x, y) = (-8 + t * l, s + t * (-6))) →
  s = -4 ∧ l = -8 := by
  sorry

end line_parameterization_l1034_103440


namespace tan_product_pi_ninths_l1034_103453

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = Real.sqrt 3 := by
  sorry

end tan_product_pi_ninths_l1034_103453


namespace parallel_lines_angle_theorem_l1034_103448

/-- Given a configuration of two parallel lines intersected by two other lines,
    if one angle is 70°, its adjacent angle is 40°, and the corresponding angle
    on the other parallel line is 110°, then the remaining angle is 40°. -/
theorem parallel_lines_angle_theorem (a b c d : Real) :
  a = 70 →
  b = 40 →
  c = 110 →
  a + b + c + d = 360 →
  d = 40 := by
  sorry

end parallel_lines_angle_theorem_l1034_103448


namespace red_markers_count_l1034_103417

/-- Given a total number of markers and a number of blue markers, 
    calculate the number of red markers. -/
def red_markers (total : ℝ) (blue : ℕ) : ℝ :=
  total - blue

/-- Prove that given 64.0 total markers and 23 blue markers, 
    the number of red markers is 41. -/
theorem red_markers_count : red_markers 64.0 23 = 41 := by
  sorry

end red_markers_count_l1034_103417


namespace smallest_dual_base_representation_l1034_103403

theorem smallest_dual_base_representation : ∃ (n : ℕ) (a b : ℕ), 
  a > 3 ∧ b > 3 ∧
  n = a + 3 ∧
  n = 3 * b + 1 ∧
  (∀ (m : ℕ) (c d : ℕ), c > 3 → d > 3 → m = c + 3 → m = 3 * d + 1 → m ≥ n) :=
by sorry

end smallest_dual_base_representation_l1034_103403


namespace point_on_curve_limit_at_one_l1034_103459

/-- The curve y = x² + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The point (1, 2) lies on the curve -/
theorem point_on_curve : f 1 = 2 := by sorry

/-- The limit of Δy/Δx as Δx approaches 0 at x = 1 is 2 -/
theorem limit_at_one : 
  ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 
    0 < |h| → |h| < δ → |(f (1 + h) - f 1) / h - 2| < ε := by sorry

end point_on_curve_limit_at_one_l1034_103459


namespace pie_eating_contest_l1034_103414

theorem pie_eating_contest (first_student second_student : ℚ) 
  (h1 : first_student = 7/8)
  (h2 : second_student = 5/6) :
  first_student - second_student = 1/24 := by
sorry

end pie_eating_contest_l1034_103414


namespace trail_mix_nuts_l1034_103401

theorem trail_mix_nuts (walnuts almonds : ℚ) 
  (hw : walnuts = 0.25)
  (ha : almonds = 0.25) :
  walnuts + almonds = 0.50 := by sorry

end trail_mix_nuts_l1034_103401


namespace people_who_left_line_l1034_103450

theorem people_who_left_line (initial_people : ℕ) (joined_people : ℕ) (people_who_left : ℕ) : 
  initial_people = 31 → 
  joined_people = 25 → 
  initial_people = (initial_people - people_who_left) + joined_people →
  people_who_left = 25 := by
sorry

end people_who_left_line_l1034_103450


namespace consecutive_squares_sum_181_l1034_103481

theorem consecutive_squares_sum_181 :
  ∃ k : ℕ, k^2 + (k+1)^2 = 181 ∧ k = 9 := by
  sorry

end consecutive_squares_sum_181_l1034_103481


namespace tracy_book_collection_l1034_103462

theorem tracy_book_collection (x : ℕ) (h : x + 10 * x = 99) : x = 9 := by
  sorry

end tracy_book_collection_l1034_103462


namespace circle_equations_l1034_103431

-- Define points
def A : ℝ × ℝ := (6, 5)
def B : ℝ × ℝ := (0, 1)
def P : ℝ × ℝ := (-2, 4)
def Q : ℝ × ℝ := (3, -1)

-- Define the line equation for the center
def center_line (x y : ℝ) : Prop := 3 * x + 10 * y + 9 = 0

-- Define the chord length on x-axis
def chord_length : ℝ := 6

-- Define circle equations
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 13
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

theorem circle_equations :
  ∃ (C : ℝ × ℝ),
    (center_line C.1 C.2) ∧
    (circle1 A.1 A.2 ∨ circle2 A.1 A.2) ∧
    (circle1 B.1 B.2 ∨ circle2 B.1 B.2) ∧
    (circle1 P.1 P.2 ∨ circle2 P.1 P.2) ∧
    (circle1 Q.1 Q.2 ∨ circle2 Q.1 Q.2) ∧
    (∃ (x1 x2 : ℝ), x2 - x1 = chord_length ∧
      ((circle1 x1 0 ∧ circle1 x2 0) ∨ (circle2 x1 0 ∧ circle2 x2 0))) :=
by sorry


end circle_equations_l1034_103431


namespace new_person_weight_l1034_103426

/-- Given a group of people where:
  * There are initially 4 persons
  * One person weighing 70 kg is replaced by a new person
  * The average weight increases by 3 kg after the replacement
  * The total combined weight of all five people after the change is 390 kg
  Prove that the weight of the new person is 58 kg -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℕ) 
  (avg_increase : ℕ) (total_weight : ℕ) :
  initial_count = 4 →
  replaced_weight = 70 →
  avg_increase = 3 →
  total_weight = 390 →
  ∃ (new_weight : ℕ),
    new_weight = 58 ∧
    (total_weight - new_weight + replaced_weight) / initial_count = 
    (total_weight - new_weight) / initial_count + avg_increase :=
by sorry

end new_person_weight_l1034_103426


namespace quadrupled_exponent_base_l1034_103439

theorem quadrupled_exponent_base (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) 
  (h : (4 * c)^(4 * d) = c^d * y^d) : y = 256 * c^3 := by
  sorry

end quadrupled_exponent_base_l1034_103439


namespace like_terms_exponent_product_l1034_103474

/-- 
Given two algebraic terms are like terms, prove that the product of their exponents is 6.
-/
theorem like_terms_exponent_product (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ), 5 * a^3 * b^n = k * (-3 * a^m * b^2)) → m * n = 6 := by
  sorry

end like_terms_exponent_product_l1034_103474


namespace first_train_length_is_30_l1034_103476

/-- The length of the second train in meters -/
def second_train_length : ℝ := 180

/-- The time taken by the first train to cross the stationary second train in seconds -/
def time_cross_stationary : ℝ := 18

/-- The length of the platform crossed by the first train in meters -/
def platform_length_first : ℝ := 250

/-- The time taken by the first train to cross its platform in seconds -/
def time_cross_platform_first : ℝ := 24

/-- The length of the platform crossed by the second train in meters -/
def platform_length_second : ℝ := 200

/-- The time taken by the second train to cross its platform in seconds -/
def time_cross_platform_second : ℝ := 22

/-- The length of the first train in meters -/
def first_train_length : ℝ := 30

theorem first_train_length_is_30 :
  (first_train_length + second_train_length) / time_cross_stationary =
  (first_train_length + platform_length_first) / time_cross_platform_first ∧
  first_train_length = 30 := by
  sorry

end first_train_length_is_30_l1034_103476


namespace fruit_difference_l1034_103437

theorem fruit_difference (watermelons peaches plums : ℕ) : 
  watermelons = 1 →
  peaches > watermelons →
  plums = 3 * peaches →
  watermelons + peaches + plums = 53 →
  peaches - watermelons = 12 :=
by sorry

end fruit_difference_l1034_103437


namespace quadratic_roots_property_l1034_103454

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d - 2) * (e - 2) = 13/3 := by
sorry

end quadratic_roots_property_l1034_103454


namespace exists_floating_polyhedron_with_properties_l1034_103423

/-- A convex polyhedron floating in water -/
structure FloatingPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  submergedVolume : ℝ
  surfaceAreaAboveWater : ℝ
  volume_pos : 0 < volume
  surfaceArea_pos : 0 < surfaceArea
  submergedVolume_le_volume : submergedVolume ≤ volume
  surfaceAreaAboveWater_le_surfaceArea : surfaceAreaAboveWater ≤ surfaceArea

/-- Theorem stating the existence of a floating polyhedron with specific properties -/
theorem exists_floating_polyhedron_with_properties :
  ∀ ε > 0, ∃ (P : FloatingPolyhedron),
    P.submergedVolume / P.volume > 1 - ε ∧
    P.surfaceAreaAboveWater / P.surfaceArea > 1/2 := by
  sorry

end exists_floating_polyhedron_with_properties_l1034_103423


namespace unique_k_for_perfect_square_and_cube_l1034_103445

theorem unique_k_for_perfect_square_and_cube (Z K : ℤ) 
  (h1 : 700 < Z) (h2 : Z < 1500) (h3 : K > 1) (h4 : Z = K^4) :
  (∃ a b : ℤ, Z = a^2 ∧ Z = b^3) ↔ K = 3 * Real.sqrt 3 :=
sorry

end unique_k_for_perfect_square_and_cube_l1034_103445


namespace whale_consumption_increase_l1034_103489

/-- Represents the whale's plankton consumption pattern -/
structure WhaleConsumption where
  initial : ℕ  -- Initial consumption in the first hour
  increase : ℕ  -- Constant increase each hour after the first
  duration : ℕ  -- Duration of the feeding frenzy in hours
  total : ℕ     -- Total accumulated consumption
  sixth_hour : ℕ -- Consumption in the sixth hour

/-- Theorem stating the whale's consumption increase -/
theorem whale_consumption_increase 
  (w : WhaleConsumption) 
  (h1 : w.duration = 9)
  (h2 : w.total = 450)
  (h3 : w.sixth_hour = 54)
  (h4 : w.initial + 5 * w.increase = w.sixth_hour)
  (h5 : (w.duration : ℕ) * w.initial + 
        (w.duration * (w.duration - 1) / 2) * w.increase = w.total) : 
  w.increase = 4 := by
  sorry

end whale_consumption_increase_l1034_103489


namespace price_adjustment_solution_l1034_103433

/-- Selling prices before and after adjustment in places A and B -/
structure Prices where
  a_before : ℝ
  b_before : ℝ
  a_after : ℝ
  b_after : ℝ

/-- Conditions of the price adjustment problem -/
def PriceAdjustmentConditions (p : Prices) : Prop :=
  p.a_after = p.a_before * 1.1 ∧
  p.b_after = p.b_before - 5 ∧
  p.b_before - p.a_before = 10 ∧
  p.b_after - p.a_after = 1

/-- Theorem stating the solution to the price adjustment problem -/
theorem price_adjustment_solution :
  ∃ (p : Prices), PriceAdjustmentConditions p ∧ p.a_before = 40 ∧ p.b_before = 50 := by
  sorry

end price_adjustment_solution_l1034_103433


namespace matt_points_l1034_103436

/-- Calculates the total points scored in basketball given the number of successful 2-point and 3-point shots -/
def total_points (two_point_shots : ℕ) (three_point_shots : ℕ) : ℕ :=
  2 * two_point_shots + 3 * three_point_shots

/-- Theorem stating that four 2-point shots and two 3-point shots result in 14 points -/
theorem matt_points : total_points 4 2 = 14 := by
  sorry

end matt_points_l1034_103436


namespace geometric_sequence_common_ratio_l1034_103482

/-- Given a geometric sequence {a_n} with a_1 = 1/8 and a_4 = -1, prove that the common ratio q is -2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 1/8) 
  (h_a4 : a 4 = -1) 
  (q : ℚ) : 
  q = -2 := by
sorry

end geometric_sequence_common_ratio_l1034_103482


namespace max_distance_between_circles_l1034_103444

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ with equation x² + y² - 4x - 5 = 0 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

/-- The maximum distance between any point on C₁ and any point on C₂ is 13 -/
theorem max_distance_between_circles :
  ∃ (m₁ m₂ n₁ n₂ : ℝ), C₁ m₁ m₂ ∧ C₂ n₁ n₂ ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ Real.sqrt ((m₁ - n₁)^2 + (m₂ - n₂)^2) ∧
  Real.sqrt ((m₁ - n₁)^2 + (m₂ - n₂)^2) = 13 :=
sorry

end max_distance_between_circles_l1034_103444


namespace smallest_quotient_three_digit_number_l1034_103402

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem smallest_quotient_three_digit_number :
  ∀ n : ℕ, is_three_digit_number n →
    (n : ℚ) / (digit_sum n : ℚ) ≥ 199 / 19 :=
by sorry

end smallest_quotient_three_digit_number_l1034_103402


namespace line_through_points_l1034_103456

/-- Given a line y = ax + b passing through points (3, 2) and (7, 26), prove that a - b = 22 -/
theorem line_through_points (a b : ℝ) : 
  (2 : ℝ) = a * 3 + b ∧ (26 : ℝ) = a * 7 + b → a - b = 22 := by
  sorry

end line_through_points_l1034_103456


namespace x_eq_one_sufficient_not_necessary_for_cubic_eq_l1034_103430

theorem x_eq_one_sufficient_not_necessary_for_cubic_eq :
  (∀ x : ℝ, x = 1 → x^3 - 2*x + 1 = 0) ∧
  (∃ x : ℝ, x ≠ 1 ∧ x^3 - 2*x + 1 = 0) := by
  sorry

end x_eq_one_sufficient_not_necessary_for_cubic_eq_l1034_103430


namespace min_value_of_function_l1034_103496

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  let y := x + 4 / (x - 1)
  (∀ z, z > 1 → y ≤ z + 4 / (z - 1)) ∧ y = 5 ↔ x = 3 :=
by sorry

end min_value_of_function_l1034_103496


namespace fish_caught_l1034_103451

theorem fish_caught (initial_fish : ℕ) (initial_tadpoles : ℕ) (fish_caught : ℕ) : 
  initial_fish = 50 →
  initial_tadpoles = 3 * initial_fish →
  initial_tadpoles / 2 = (initial_fish - fish_caught) + 32 →
  fish_caught = 7 :=
by
  sorry

end fish_caught_l1034_103451


namespace inequality_proof_l1034_103438

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (b + c) * (c + a) * (a + b) ≥ 4 * ((a + b + c) * ((a + b + c) / 3) ^ (1/8) - 1) := by
  sorry

end inequality_proof_l1034_103438


namespace fliers_calculation_l1034_103443

theorem fliers_calculation (initial_fliers : ℕ) : 
  (initial_fliers : ℚ) * (4/5) * (3/4) = 1800 → initial_fliers = 3000 := by
  sorry

end fliers_calculation_l1034_103443


namespace larger_rectangle_area_larger_rectangle_area_proof_l1034_103446

theorem larger_rectangle_area : ℝ → ℝ → ℝ → Prop :=
  fun (small_square_area : ℝ) (small_rect_length : ℝ) (small_rect_width : ℝ) =>
    small_square_area = 25 ∧
    small_rect_length = 3 * Real.sqrt small_square_area ∧
    small_rect_width = Real.sqrt small_square_area ∧
    2 * small_rect_width = small_rect_length →
    small_rect_length * (2 * small_rect_width) = 150

-- The proof goes here
theorem larger_rectangle_area_proof :
  ∃ (small_square_area small_rect_length small_rect_width : ℝ),
    larger_rectangle_area small_square_area small_rect_length small_rect_width :=
by
  sorry

end larger_rectangle_area_larger_rectangle_area_proof_l1034_103446


namespace rectangle_area_is_216_l1034_103461

/-- Represents a rectangle with given properties -/
structure Rectangle where
  length : ℝ
  breadth : ℝ
  perimeterToBreadthRatio : ℝ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.breadth

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.breadth)

/-- Theorem: A rectangle with length 18 and perimeter to breadth ratio of 5 has an area of 216 -/
theorem rectangle_area_is_216 (r : Rectangle) 
    (h1 : r.length = 18)
    (h2 : r.perimeterToBreadthRatio = 5)
    (h3 : perimeter r / r.breadth = r.perimeterToBreadthRatio) : 
  area r = 216 := by
  sorry


end rectangle_area_is_216_l1034_103461


namespace percentage_equality_l1034_103460

theorem percentage_equality (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 := by
  sorry

end percentage_equality_l1034_103460


namespace quadratic_equation_properties_specific_root_condition_l1034_103441

/-- Represents a quadratic equation of the form x^2 + 2(m+1)x + m^2 - 1 = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + 2*(m+1)*x + m^2 - 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  8*m + 8

/-- Condition for the roots of the quadratic equation -/
def root_condition (x₁ x₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 = 16 - x₁*x₂

theorem quadratic_equation_properties (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ x₁ ≠ x₂) ↔ m ≥ -1 :=
sorry

theorem specific_root_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ 
   x₁ ≠ x₂ ∧ root_condition x₁ x₂) → m = 1 :=
sorry

end quadratic_equation_properties_specific_root_condition_l1034_103441


namespace total_cost_theorem_l1034_103485

-- Define the given conditions
def cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cards_per_pack : ℕ := 50
def cost_per_pack : ℚ := 3

-- Define the total number of index cards needed
def total_cards_needed : ℕ := cards_per_student * students_per_class * periods_per_day

-- Define the number of packs needed
def packs_needed : ℕ := (total_cards_needed + cards_per_pack - 1) / cards_per_pack

-- State the theorem
theorem total_cost_theorem : 
  cost_per_pack * packs_needed = 108 := by sorry

end total_cost_theorem_l1034_103485


namespace radical_axis_is_line_l1034_103458

/-- The locus of points with equal power with respect to two non-concentric circles is a line -/
theorem radical_axis_is_line (R₁ R₂ a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, ∀ x y : ℝ, (x + a)^2 + y^2 - R₁^2 = (x - a)^2 + y^2 - R₂^2 ↔ x = k :=
sorry

end radical_axis_is_line_l1034_103458


namespace representable_as_product_of_three_l1034_103467

theorem representable_as_product_of_three : ∃ (a b c : ℕ), 
  a > 1 ∧ b > 1 ∧ c > 1 ∧ 2^58 + 1 = a * b * c := by
  sorry

end representable_as_product_of_three_l1034_103467


namespace trihedral_angle_existence_l1034_103404

/-- A trihedral angle -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

/-- Given three dihedral angles, there exists a trihedral angle with these angles -/
theorem trihedral_angle_existence (α β γ : Real) : 
  ∃ (T : TrihedralAngle), T.α = α ∧ T.β = β ∧ T.γ = γ := by
  sorry

end trihedral_angle_existence_l1034_103404


namespace range_of_m_l1034_103457

/-- Proposition p: the solution set for |x| + |x + 1| > m is ℝ -/
def p (m : ℝ) : Prop := ∀ x, |x| + |x + 1| > m

/-- Proposition q: the function f(x) = x^2 - 2mx + 1 is increasing on (2, +∞) -/
def q (m : ℝ) : Prop := ∀ x > 2, Monotone (fun x => x^2 - 2*m*x + 1)

/-- The range of real numbers m that satisfies the given conditions is [1, 2] -/
theorem range_of_m :
  ∀ m : ℝ, (∀ m', (p m' ∨ q m') ∧ ¬(p m' ∧ q m') → m' = m) → m ∈ Set.Icc 1 2 :=
sorry

end range_of_m_l1034_103457


namespace total_sugar_amount_l1034_103416

/-- The total amount of sugar the owner started with, given the number of packs,
    weight per pack, and leftover sugar. -/
theorem total_sugar_amount
  (num_packs : ℕ)
  (weight_per_pack : ℕ)
  (leftover_sugar : ℕ)
  (h1 : num_packs = 12)
  (h2 : weight_per_pack = 250)
  (h3 : leftover_sugar = 20) :
  num_packs * weight_per_pack + leftover_sugar = 3020 :=
by sorry

end total_sugar_amount_l1034_103416


namespace opposite_of_2023_l1034_103475

theorem opposite_of_2023 : 
  (2023 : ℤ) + (-2023) = 0 := by sorry

end opposite_of_2023_l1034_103475


namespace oneSeventhIncreaseAfterRemoval_l1034_103470

/-- The decimal representation of 1/7 -/
def oneSeventhDecimal : ℚ := 1 / 7

/-- The position of the digit to be removed -/
def digitPosition : ℕ := 2021

/-- The function that removes the digit at the specified position and shifts subsequent digits -/
def removeDigitAndShift (q : ℚ) (pos : ℕ) : ℚ :=
  sorry -- Implementation details omitted

/-- Theorem stating that removing the 2021st digit after the decimal point in 1/7 increases the value -/
theorem oneSeventhIncreaseAfterRemoval :
  removeDigitAndShift oneSeventhDecimal digitPosition > oneSeventhDecimal :=
sorry

end oneSeventhIncreaseAfterRemoval_l1034_103470


namespace goods_train_passing_time_l1034_103498

/-- The time taken for a goods train to pass a man in an opposing train -/
theorem goods_train_passing_time (man_speed goods_speed : ℝ) (goods_length : ℝ) : 
  man_speed = 70 →
  goods_speed = 42 →
  goods_length = 280 →
  ∃ t : ℝ, t > 0 ∧ t < 10 ∧ t * (man_speed + goods_speed) * (1000 / 3600) = goods_length :=
by sorry

end goods_train_passing_time_l1034_103498


namespace discrete_rv_distribution_l1034_103480

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  h₁ : x₂ > x₁
  h₂ : p₁ = 0.6
  h₃ : p₁ * x₁ + (1 - p₁) * x₂ = 1.4  -- Expected value
  h₄ : p₁ * (x₁ - 1.4)^2 + (1 - p₁) * (x₂ - 1.4)^2 = 0.24  -- Variance

/-- The probability distribution of the discrete random variable -/
def probability_distribution (X : DiscreteRV) : Prop :=
  X.x₁ = 1 ∧ X.x₂ = 2

theorem discrete_rv_distribution (X : DiscreteRV) :
  probability_distribution X := by
  sorry

end discrete_rv_distribution_l1034_103480


namespace min_value_y_l1034_103465

theorem min_value_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y * Real.log y = Real.exp (2 * x) - y * Real.log (2 * x)) : 
  (∀ z, z > 0 → z * Real.log z = Real.exp (2 * x) - z * Real.log (2 * x) → y ≤ z) ∧ y = Real.exp 1 :=
sorry

end min_value_y_l1034_103465


namespace existence_of_special_sequence_l1034_103483

theorem existence_of_special_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j : Fin 100, i < j → a i < a j) ∧ 
    (∀ k : Fin 100, 2 ≤ k.val → k.val ≤ 100 → 
      Nat.lcm (a ⟨k.val - 1, sorry⟩) (a k) > Nat.lcm (a k) (a ⟨k.val + 1, sorry⟩)) :=
by sorry

end existence_of_special_sequence_l1034_103483


namespace circle_equation_l1034_103435

-- Define the center and radius of the circle
def center : ℝ × ℝ := (2, -1)
def radius : ℝ := 4

-- State the theorem
theorem circle_equation :
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ 
  (x - 2)^2 + (y + 1)^2 = 16 :=
by sorry

end circle_equation_l1034_103435


namespace monotone_increasing_condition_l1034_103495

/-- The function f(x) = sin(2x) - a*cos(x) is monotonically increasing on [0, π] iff a ≥ 2 -/
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 Real.pi, MonotoneOn (fun x => Real.sin (2 * x) - a * Real.cos x) (Set.Icc 0 Real.pi)) ↔ 
  a ≥ 2 := by
  sorry

end monotone_increasing_condition_l1034_103495


namespace geometric_arithmetic_ratio_l1034_103412

/-- Given a geometric sequence {a_n} with common ratio q ≠ 1,
    if a_4, a_3, a_5 form an arithmetic sequence,
    then (a_3 + a_4) / (a_2 + a_3) = -2 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ, a (n + 1) = q * a n) →
  2 * a 3 = a 4 + a 5 →
  (a 3 + a 4) / (a 2 + a 3) = -2 := by
  sorry

end geometric_arithmetic_ratio_l1034_103412


namespace calculation_proof_l1034_103471

theorem calculation_proof : 3 * 8 * 9 + 18 / 3 - 2^3 = 214 := by
  sorry

end calculation_proof_l1034_103471


namespace waiting_time_problem_l1034_103409

/-- Proves that the waiting time for the man to catch up is 25 minutes -/
theorem waiting_time_problem (man_speed woman_speed : ℚ) (stop_time : ℚ) :
  man_speed = 5 →
  woman_speed = 25 →
  stop_time = 5 / 60 →
  let distance_traveled := woman_speed * stop_time
  let catch_up_time := distance_traveled / man_speed
  catch_up_time = 25 / 60 := by sorry

end waiting_time_problem_l1034_103409


namespace arithmetic_sequence_angles_l1034_103466

/-- Given five angles in an arithmetic sequence with the smallest angle 25° and the largest 105°,
    the common difference is 20°. -/
theorem arithmetic_sequence_angles (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ →  -- ensuring the sequence is increasing
  a₁ = 25 →  -- smallest angle is 25°
  a₅ = 105 →  -- largest angle is 105°
  ∃ d : ℝ, d = 20 ∧  -- common difference exists and equals 20°
    a₂ = a₁ + d ∧ 
    a₃ = a₂ + d ∧ 
    a₄ = a₃ + d ∧ 
    a₅ = a₄ + d :=
by sorry

end arithmetic_sequence_angles_l1034_103466


namespace car_speed_comparison_l1034_103452

/-- Proves that given a car traveling at 80 km/hour takes 5 seconds longer to travel 1 km than at another speed, the other speed is 90 km/hour. -/
theorem car_speed_comparison (v : ℝ) : 
  v > 0 →  -- Ensure speed is positive
  (1 / (80 / 3600)) - (1 / (v / 3600)) = 5 → 
  v = 90 :=
by sorry

end car_speed_comparison_l1034_103452


namespace scalar_projection_a_onto_b_l1034_103477

/-- The scalar projection of vector a (1, 2) onto vector b (3, 4) is 11/5 -/
theorem scalar_projection_a_onto_b :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (3, 4)
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = 11 / 5 := by
  sorry

end scalar_projection_a_onto_b_l1034_103477


namespace dragon_disc_reassembly_l1034_103400

/-- A circular disc with a dragon painted on it -/
structure DragonDisc where
  center : ℝ × ℝ
  radius : ℝ
  dragon_center : ℝ × ℝ

/-- Two discs are congruent if they have the same radius -/
def congruent (d1 d2 : DragonDisc) : Prop := d1.radius = d2.radius

/-- The dragon covers the center of the disc if its center coincides with the disc's center -/
def dragon_covers_center (d : DragonDisc) : Prop := d.center = d.dragon_center

/-- A disc can be cut and reassembled if there exists a line that divides it into two pieces -/
def can_cut_and_reassemble (d : DragonDisc) : Prop := 
  ∃ (line : ℝ × ℝ → ℝ × ℝ → Prop), 
    ∃ (piece1 piece2 : Set (ℝ × ℝ)), 
      piece1 ∪ piece2 = {p | (p.1 - d.center.1)^2 + (p.2 - d.center.2)^2 ≤ d.radius^2}

theorem dragon_disc_reassembly 
  (d1 d2 : DragonDisc)
  (h_congruent : congruent d1 d2)
  (h_d1_center : dragon_covers_center d1)
  (h_d2_offset : ¬dragon_covers_center d2) :
  can_cut_and_reassemble d2 ∧ 
  ∃ (d2_new : DragonDisc), congruent d2 d2_new ∧ dragon_covers_center d2_new :=
by sorry

end dragon_disc_reassembly_l1034_103400


namespace cube_volume_partition_l1034_103493

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  sideLength : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Given a cube and a plane passing through the midpoint of one edge and two points
    on opposite edges with the ratio 1:7 from the vertices, the smaller part of the
    volume separated by this plane is 25/192 of the cube's volume. -/
theorem cube_volume_partition (cube : Cube) (plane : Plane)
  (h1 : plane.a * (cube.sideLength / 2) + plane.b * 0 + plane.c * 0 = plane.d)
  (h2 : plane.a * 0 + plane.b * 0 + plane.c * (cube.sideLength / 8) = plane.d)
  (h3 : plane.a * cube.sideLength + plane.b * cube.sideLength + plane.c * (cube.sideLength / 8) = plane.d) :
  ∃ (smallerVolume : ℝ), smallerVolume = (25 / 192) * cube.sideLength ^ 3 := by
  sorry

end cube_volume_partition_l1034_103493


namespace log_inequality_l1034_103419

theorem log_inequality (x : ℝ) (h : 2 * Real.log x / Real.log 2 - 1 < 0) : 0 < x ∧ x < Real.sqrt 2 := by
  sorry

end log_inequality_l1034_103419


namespace carols_weight_l1034_103442

theorem carols_weight (alice_weight carol_weight : ℝ) 
  (h1 : alice_weight + carol_weight = 240)
  (h2 : carol_weight - alice_weight = 2/3 * carol_weight) : 
  carol_weight = 180 := by
sorry

end carols_weight_l1034_103442


namespace complex_equation_solution_l1034_103420

theorem complex_equation_solution :
  ∀ x : ℝ, (1 - 2*I) * (x + I) = 4 - 3*I → x = 2 := by
  sorry

end complex_equation_solution_l1034_103420


namespace num_selection_methods_l1034_103418

/-- The number of fleets --/
def num_fleets : ℕ := 7

/-- The total number of vehicles to be selected --/
def total_vehicles : ℕ := 10

/-- The minimum number of vehicles in each fleet --/
def min_vehicles_per_fleet : ℕ := 5

/-- Function to calculate the number of ways to select vehicles --/
def select_vehicles (n f t m : ℕ) : ℕ :=
  Nat.choose n 1 + n * (n - 1) + Nat.choose n 3

/-- Theorem stating the number of ways to select vehicles --/
theorem num_selection_methods :
  select_vehicles num_fleets num_fleets total_vehicles min_vehicles_per_fleet = 84 := by
  sorry


end num_selection_methods_l1034_103418


namespace personal_planner_cost_proof_l1034_103464

/-- The cost of a spiral notebook -/
def spiral_notebook_cost : ℝ := 15

/-- The number of spiral notebooks -/
def num_spiral_notebooks : ℕ := 4

/-- The number of personal planners -/
def num_personal_planners : ℕ := 8

/-- The discount rate -/
def discount_rate : ℝ := 0.2

/-- The total cost after discount -/
def total_cost_after_discount : ℝ := 112

/-- The cost of a personal planner -/
def personal_planner_cost : ℝ := 10

theorem personal_planner_cost_proof :
  let total_cost := spiral_notebook_cost * num_spiral_notebooks + personal_planner_cost * num_personal_planners
  total_cost * (1 - discount_rate) = total_cost_after_discount :=
by sorry

end personal_planner_cost_proof_l1034_103464


namespace min_value_expression_min_value_achievable_l1034_103472

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 + 3 / (a + b + c)^2 ≥ 2 :=
sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 + 3 / (a + b + c)^2 = 2 :=
sorry

end min_value_expression_min_value_achievable_l1034_103472


namespace cube_diff_divisibility_l1034_103491

theorem cube_diff_divisibility (m n k : ℕ) (hm : Odd m) (hn : Odd n) (hk : k > 0) :
  (2^k ∣ m^3 - n^3) ↔ (2^k ∣ m - n) := by
  sorry

end cube_diff_divisibility_l1034_103491


namespace quadratic_function_range_l1034_103413

/-- A quadratic function f(x) = a + bx - x^2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a + b * x - x^2

theorem quadratic_function_range (a b m : ℝ) :
  (∀ x, f a b (1 + x) = f a b (1 - x)) →
  (∀ x ≤ 4, Monotone (fun x ↦ f a b (x + m))) →
  m ≤ -3 := by
  sorry

end quadratic_function_range_l1034_103413


namespace given_number_eq_scientific_notation_l1034_103427

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_abs_coeff : 1 ≤ |coefficient|
  abs_coeff_lt_ten : |coefficient| < 10

/-- The given number in centimeters -/
def given_number : ℝ := 0.0000021

/-- The scientific notation representation of the given number -/
def scientific_notation : ScientificNotation := {
  coefficient := 2.1
  exponent := -6
  one_le_abs_coeff := sorry
  abs_coeff_lt_ten := sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_eq_scientific_notation : 
  given_number = scientific_notation.coefficient * (10 : ℝ) ^ scientific_notation.exponent := by
  sorry

end given_number_eq_scientific_notation_l1034_103427


namespace managers_salary_managers_salary_proof_l1034_103473

/-- The manager's salary problem -/
theorem managers_salary (num_employees : ℕ) (initial_avg_salary : ℕ) (salary_increase : ℕ) : ℕ :=
  let total_initial_salary := num_employees * initial_avg_salary
  let new_avg_salary := initial_avg_salary + salary_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_initial_salary

/-- Proof of the manager's salary -/
theorem managers_salary_proof :
  managers_salary 50 2500 1500 = 79000 := by
  sorry

end managers_salary_managers_salary_proof_l1034_103473


namespace salary_problem_l1034_103447

theorem salary_problem (salary_a salary_b : ℝ) : 
  salary_a + salary_b = 2000 →
  salary_a * 0.05 = salary_b * 0.15 →
  salary_a = 1500 := by
sorry

end salary_problem_l1034_103447


namespace f_properties_l1034_103455

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom cond1 : ∀ x, f (10 + x) = f (10 - x)
axiom cond2 : ∀ x, f (20 - x) = -f (20 + x)

-- Define oddness
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

-- Define periodicity
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Theorem statement
theorem f_properties : is_odd f ∧ is_periodic f 20 :=
  sorry

end f_properties_l1034_103455


namespace nines_squared_zeros_l1034_103492

theorem nines_squared_zeros (n : ℕ) :
  ∃ m : ℕ, (10^9 - 1)^2 = m * 10^8 ∧ m % 10 ≠ 0 :=
sorry

end nines_squared_zeros_l1034_103492


namespace emily_garden_seeds_l1034_103422

theorem emily_garden_seeds (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : small_gardens = 3)
  (h3 : seeds_per_small_garden = 4) :
  total_seeds - (small_gardens * seeds_per_small_garden) = 29 := by
  sorry

end emily_garden_seeds_l1034_103422


namespace point_on_bisector_value_l1034_103484

/-- A point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The bisector of the angle between the two coordinate axes in the first and third quadrants -/
def isOnBisector (p : Point) : Prop :=
  p.x = p.y

/-- The theorem statement -/
theorem point_on_bisector_value (a : ℝ) :
  let A : Point := ⟨a, 2*a + 3⟩
  isOnBisector A → a = -3 := by
  sorry

end point_on_bisector_value_l1034_103484


namespace special_triangle_third_side_l1034_103486

/-- Triangle sides satisfy the given conditions -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  side_condition : Real.sqrt (a - 9) + (b - 2)^2 = 0
  c_odd : ∃ (k : ℤ), c = 2 * k + 1

/-- The third side of the special triangle is 9 -/
theorem special_triangle_third_side (t : SpecialTriangle) : t.c = 9 := by
  sorry

end special_triangle_third_side_l1034_103486


namespace arithmetic_mean_relation_l1034_103410

theorem arithmetic_mean_relation (a b x : ℝ) : 
  (2 * x = a + b) → 
  (2 * x^2 = a^2 - b^2) → 
  (a = -b ∨ a = 3*b) := by
sorry

end arithmetic_mean_relation_l1034_103410


namespace abs_difference_equals_seven_l1034_103478

theorem abs_difference_equals_seven (a b : ℝ) 
  (ha : |a| = 4) 
  (hb : |b| = 3) 
  (hab : a * b < 0) : 
  |a - b| = 7 := by
sorry

end abs_difference_equals_seven_l1034_103478


namespace quadratic_distinct_roots_l1034_103468

theorem quadratic_distinct_roots (n : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + n*x + 9 = 0 ∧ y^2 + n*y + 9 = 0) ↔ 
  (n < -6 ∨ n > 6) :=
sorry

end quadratic_distinct_roots_l1034_103468


namespace parallel_planes_transitive_perpendicular_to_line_parallel_l1034_103424

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (parallel_to_line : Plane → Line → Prop)
variable (perpendicular_to_line : Plane → Line → Prop)
variable (coincident : Plane → Plane → Prop)

-- Theorem 1
theorem parallel_planes_transitive (α β γ : Plane) :
  parallel α γ → parallel β γ → ¬coincident α β → parallel α β := by sorry

-- Theorem 2
theorem perpendicular_to_line_parallel (α β : Plane) (l : Line) :
  perpendicular_to_line α l → perpendicular_to_line β l → ¬coincident α β → parallel α β := by sorry

end parallel_planes_transitive_perpendicular_to_line_parallel_l1034_103424


namespace wall_passing_skill_l1034_103411

theorem wall_passing_skill (n : ℕ) (h : 8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n))) :
  n = 63 := by
  sorry

end wall_passing_skill_l1034_103411


namespace OMM_MOO_not_synonyms_l1034_103432

/-- Represents a word in the Ancient Tribe language --/
inductive AncientWord
  | M : AncientWord
  | O : AncientWord
  | append : AncientWord → AncientWord → AncientWord

/-- Counts the number of 'M's in a word --/
def countM : AncientWord → Nat
  | AncientWord.M => 1
  | AncientWord.O => 0
  | AncientWord.append w1 w2 => countM w1 + countM w2

/-- Counts the number of 'O's in a word --/
def countO : AncientWord → Nat
  | AncientWord.M => 0
  | AncientWord.O => 1
  | AncientWord.append w1 w2 => countO w1 + countO w2

/-- Calculates the difference between 'M's and 'O's in a word --/
def letterDifference (w : AncientWord) : Int :=
  (countM w : Int) - (countO w : Int)

/-- Two words are synonyms if their letter differences are equal --/
def areSynonyms (w1 w2 : AncientWord) : Prop :=
  letterDifference w1 = letterDifference w2

/-- Construct the word "OMM" --/
def OMM : AncientWord :=
  AncientWord.append AncientWord.O (AncientWord.append AncientWord.M AncientWord.M)

/-- Construct the word "MOO" --/
def MOO : AncientWord :=
  AncientWord.append AncientWord.M (AncientWord.append AncientWord.O AncientWord.O)

/-- Theorem: "OMM" and "MOO" are not synonyms --/
theorem OMM_MOO_not_synonyms : ¬(areSynonyms OMM MOO) := by
  sorry


end OMM_MOO_not_synonyms_l1034_103432


namespace empty_solution_set_range_l1034_103487

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 2*a| + |x + 3| < 5)) ↔ (a ≤ -4 ∨ a ≥ 1) :=
sorry

end empty_solution_set_range_l1034_103487


namespace fraction_reduction_l1034_103494

theorem fraction_reduction (a x : ℝ) :
  (Real.sqrt (a^2 + x^2) - (x^2 - a^2) / Real.sqrt (a^2 + x^2)) / (a^2 + x^2) = 
  2 * a^2 / (a^2 + x^2)^(3/2) :=
by sorry

end fraction_reduction_l1034_103494


namespace cuboid_gluing_theorem_l1034_103425

/-- A cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ+
  width : ℕ+
  height : ℕ+
  different_dimensions : length ≠ width ∧ width ≠ height ∧ height ≠ length

/-- The volume of a cuboid -/
def volume (c : Cuboid) : ℕ := c.length * c.width * c.height

/-- Two cuboids can be glued if they share a face -/
def can_be_glued (c1 c2 : Cuboid) : Prop :=
  (c1.length = c2.length ∧ c1.width = c2.width) ∨
  (c1.length = c2.length ∧ c1.height = c2.height) ∨
  (c1.width = c2.width ∧ c1.height = c2.height)

/-- The resulting cuboid after gluing two cuboids -/
def glued_cuboid (c1 c2 : Cuboid) : Cuboid :=
  if c1.length = c2.length ∧ c1.width = c2.width then
    ⟨c1.length, c1.width, c1.height + c2.height, sorry⟩
  else if c1.length = c2.length ∧ c1.height = c2.height then
    ⟨c1.length, c1.width + c2.width, c1.height, sorry⟩
  else
    ⟨c1.length + c2.length, c1.width, c1.height, sorry⟩

theorem cuboid_gluing_theorem (c1 c2 : Cuboid) :
  volume c1 = 12 →
  volume c2 = 30 →
  can_be_glued c1 c2 →
  let c := glued_cuboid c1 c2
  (c.length = 1 ∧ c.width = 2 ∧ c.height = 21) ∨
  (c.length = 1 ∧ c.width = 3 ∧ c.height = 14) ∨
  (c.length = 1 ∧ c.width = 6 ∧ c.height = 7) :=
by sorry

end cuboid_gluing_theorem_l1034_103425


namespace turban_price_turban_price_proof_l1034_103428

/-- The price of a turban given the following conditions:
  - The total salary for one year is Rs. 90 plus one turban
  - The servant works for 9 months (3/4 of a year)
  - The servant receives Rs. 60 plus the turban for 9 months of work
-/
theorem turban_price : ℝ :=
  let yearly_salary : ℝ → ℝ := λ t => 90 + t
  let worked_fraction : ℝ := 3 / 4
  let received_salary : ℝ → ℝ := λ t => 60 + t
  30

theorem turban_price_proof (t : ℝ) : 
  (let yearly_salary : ℝ → ℝ := λ t => 90 + t
   let worked_fraction : ℝ := 3 / 4
   let received_salary : ℝ → ℝ := λ t => 60 + t
   worked_fraction * yearly_salary t = received_salary t) →
  t = 30 := by
sorry

end turban_price_turban_price_proof_l1034_103428


namespace manuscript_cost_l1034_103463

/-- Represents the cost structure for typing and revising pages -/
structure TypingRates :=
  (initial : ℕ)
  (first_revision : ℕ)
  (second_revision : ℕ)
  (subsequent_revisions : ℕ)

/-- Represents the manuscript details -/
structure Manuscript :=
  (total_pages : ℕ)
  (revised_once : ℕ)
  (revised_twice : ℕ)
  (revised_thrice : ℕ)

/-- Calculates the total cost of typing and revising a manuscript -/
def total_cost (rates : TypingRates) (manuscript : Manuscript) : ℕ :=
  rates.initial * manuscript.total_pages +
  rates.first_revision * manuscript.revised_once +
  rates.second_revision * manuscript.revised_twice +
  rates.subsequent_revisions * manuscript.revised_thrice

/-- The typing service rates -/
def service_rates : TypingRates :=
  { initial := 10
  , first_revision := 5
  , second_revision := 7
  , subsequent_revisions := 10 }

/-- The manuscript details -/
def manuscript : Manuscript :=
  { total_pages := 150
  , revised_once := 20
  , revised_twice := 30
  , revised_thrice := 10 }

/-- Theorem stating that the total cost for the given manuscript is 1910 -/
theorem manuscript_cost : total_cost service_rates manuscript = 1910 := by
  sorry

end manuscript_cost_l1034_103463


namespace range_of_a_l1034_103406

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 5| + 1) → 
  a ∈ Set.Ioo 4 6 := by
sorry

end range_of_a_l1034_103406


namespace circle_intersection_theorem_l1034_103497

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the line l
def line_l (x y m : ℝ) : Prop := x - y + m = 0

-- Define the perpendicularity condition
def perpendicular (xa ya xb yb xc yc : ℝ) : Prop :=
  (xa - xc) * (xb - xc) + (ya - yc) * (yb - yc) = 0

-- State the theorem
theorem circle_intersection_theorem (m : ℝ) :
  (∃ (xa ya xb yb : ℝ),
    circle_C xa ya ∧ circle_C xb yb ∧
    line_l xa ya m ∧ line_l xb yb m ∧
    perpendicular xa ya xb yb 3 1) →
  m = 1 ∨ m = -5 :=
by sorry

end circle_intersection_theorem_l1034_103497


namespace shorter_base_length_l1034_103469

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_segment : ℝ

/-- The trapezoid satisfies the given conditions -/
def trapezoid_conditions (t : Trapezoid) : Prop :=
  t.long_base = 125 ∧ t.midpoint_segment = 5

/-- Theorem: In a trapezoid satisfying the given conditions, the shorter base is 115 -/
theorem shorter_base_length (t : Trapezoid) (h : trapezoid_conditions t) : 
  t.short_base = 115 := by
  sorry

#check shorter_base_length

end shorter_base_length_l1034_103469


namespace natural_number_divisibility_l1034_103407

theorem natural_number_divisibility (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by sorry

end natural_number_divisibility_l1034_103407


namespace coprime_powers_of_primes_l1034_103488

def valid_n : Set ℕ := {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 18, 20, 24, 30, 42}

def is_power_of_prime (m : ℕ) : Prop :=
  ∃ p k, Nat.Prime p ∧ m = p ^ k

theorem coprime_powers_of_primes (n : ℕ) :
  (∀ m, 0 < m ∧ m < n ∧ Nat.Coprime m n → is_power_of_prime m) ↔ n ∈ valid_n := by
  sorry

end coprime_powers_of_primes_l1034_103488


namespace polynomial_expansion_problem_l1034_103434

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 ∧ q > 0 ∧ p + 2*q = 1 ∧ 
  (45 : ℝ) * p^8 * q^2 = (120 : ℝ) * p^7 * q^3 → 
  p = 4/7 := by
sorry

end polynomial_expansion_problem_l1034_103434
