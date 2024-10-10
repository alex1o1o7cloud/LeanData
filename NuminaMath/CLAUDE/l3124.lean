import Mathlib

namespace finite_crosses_in_circle_l3124_312416

/-- A cross formed by the diagonals of a square with side length 1 -/
def Cross : Type := Unit

/-- A circle with radius 100 -/
def Circle : Type := Unit

/-- The maximum number of non-overlapping crosses that can fit inside the circle -/
noncomputable def maxCrosses : ℕ := sorry

/-- The theorem stating that the number of non-overlapping crosses that can fit inside the circle is finite -/
theorem finite_crosses_in_circle : ∃ n : ℕ, maxCrosses ≤ n := by sorry

end finite_crosses_in_circle_l3124_312416


namespace expression_evaluation_l3124_312461

theorem expression_evaluation : 5 * 402 + 4 * 402 + 3 * 402 + 401 = 5225 := by
  sorry

end expression_evaluation_l3124_312461


namespace range_of_a_l3124_312481

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {-1, -3, a}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (Set.compl A ∩ B a).Nonempty → a ≥ 0 := by
  sorry

end range_of_a_l3124_312481


namespace triangle_angle_sum_triangle_angle_sum_is_540_l3124_312440

theorem triangle_angle_sum : ℝ → Prop :=
  fun total_sum =>
    ∃ (int_angles ext_angles : ℝ),
      (int_angles = 180) ∧
      (ext_angles = 360) ∧
      (total_sum = int_angles + ext_angles)

theorem triangle_angle_sum_is_540 : 
  triangle_angle_sum 540 := by sorry

end triangle_angle_sum_triangle_angle_sum_is_540_l3124_312440


namespace parallel_line_equation_l3124_312403

/-- A line passing through point (2, -3) and parallel to y = x has equation x - y = 5 -/
theorem parallel_line_equation : 
  ∀ (x y : ℝ), 
  (∃ (m b : ℝ), y = m * x + b ∧ m = 1) →  -- Line parallel to y = x
  (2, -3) ∈ {(x, y) | y = m * x + b} →    -- Line passes through (2, -3)
  x - y = 5 :=                            -- Equation of the line
by sorry

end parallel_line_equation_l3124_312403


namespace bottle_cap_distribution_l3124_312468

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) 
  (h1 : total_caps = 35)
  (h2 : num_groups = 7)
  (h3 : caps_per_group * num_groups = total_caps) :
  caps_per_group = 5 := by
  sorry

end bottle_cap_distribution_l3124_312468


namespace fuel_cost_savings_l3124_312498

theorem fuel_cost_savings
  (old_efficiency : ℝ)
  (old_fuel_cost : ℝ)
  (efficiency_increase : ℝ)
  (fuel_cost_increase : ℝ)
  (h1 : efficiency_increase = 0.6)
  (h2 : fuel_cost_increase = 0.3)
  : (1 - (1 + fuel_cost_increase) / (1 + efficiency_increase)) * 100 = 18.75 := by
  sorry

end fuel_cost_savings_l3124_312498


namespace selection_problem_l3124_312469

def total_students : ℕ := 10
def selected_students : ℕ := 3
def students_excluding_c : ℕ := 9
def students_excluding_abc : ℕ := 7

theorem selection_problem :
  (Nat.choose students_excluding_c selected_students) -
  (Nat.choose students_excluding_abc selected_students) = 49 := by
  sorry

end selection_problem_l3124_312469


namespace power_product_equality_l3124_312405

theorem power_product_equality (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 := by
  sorry

end power_product_equality_l3124_312405


namespace construction_paper_count_l3124_312421

/-- Represents the number of sheets in a pack of construction paper -/
structure ConstructionPaper where
  blue : ℕ
  red : ℕ

/-- Represents the daily usage of construction paper -/
structure DailyUsage where
  blue : ℕ
  red : ℕ

def initial_ratio (pack : ConstructionPaper) : Prop :=
  pack.blue * 7 = pack.red * 2

def daily_usage : DailyUsage :=
  { blue := 1, red := 3 }

def last_day_usage : DailyUsage :=
  { blue := 1, red := 3 }

def remaining_red : ℕ := 15

theorem construction_paper_count :
  ∃ (pack : ConstructionPaper),
    initial_ratio pack ∧
    ∃ (days : ℕ),
      pack.blue = daily_usage.blue * days + last_day_usage.blue ∧
      pack.red = daily_usage.red * days + last_day_usage.red + remaining_red ∧
      pack.blue + pack.red = 135 :=
sorry

end construction_paper_count_l3124_312421


namespace nested_average_calculation_l3124_312414

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- Theorem statement
theorem nested_average_calculation : 
  avg3 (avg3 2 4 1) (avg2 3 2) 5 = 59 / 18 := by
  sorry

end nested_average_calculation_l3124_312414


namespace line_passes_through_fixed_point_l3124_312436

/-- The line equation passes through a fixed point for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (2 + m) * (-1) + (1 - 2*m) * (-2) + 4 - 3*m = 0 := by
  sorry

end line_passes_through_fixed_point_l3124_312436


namespace meat_for_spring_rolls_l3124_312459

theorem meat_for_spring_rolls (initial_meat : ℝ) (meatball_fraction : ℝ) (remaining_meat : ℝ) : 
  initial_meat = 20 ∧ meatball_fraction = 1/4 ∧ remaining_meat = 12 →
  initial_meat - meatball_fraction * initial_meat - remaining_meat = 3 :=
by sorry

end meat_for_spring_rolls_l3124_312459


namespace max_value_ratio_l3124_312431

theorem max_value_ratio (x y k : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (h : x = k * y) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x y k, x ≠ 0 → y ≠ 0 → k ≠ 0 → x = k * y →
    |x + y| / (|x| + |y|) ≤ M ∧ ∃ x y k, x ≠ 0 ∧ y ≠ 0 ∧ k ≠ 0 ∧ x = k * y ∧ |x + y| / (|x| + |y|) = M :=
sorry

end max_value_ratio_l3124_312431


namespace age_difference_l3124_312412

theorem age_difference (A B C : ℤ) (h : A + B = B + C + 11) : A - C = 11 := by
  sorry

end age_difference_l3124_312412


namespace negation_of_sum_of_squares_zero_l3124_312441

theorem negation_of_sum_of_squares_zero (a b : ℝ) :
  ¬(a^2 + b^2 = 0) ↔ (a ≠ 0 ∧ b ≠ 0) := by
  sorry

end negation_of_sum_of_squares_zero_l3124_312441


namespace linear_function_through_points_l3124_312496

/-- Given a linear function y = ax + a where a is a constant, and the graph of this function 
    passes through the point (1,2), prove that the graph also passes through the point (-2,-1). -/
theorem linear_function_through_points (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f = λ x => a * x + a) → 
  (2 = a * 1 + a) → 
  (-1 = a * (-2) + a) :=
by sorry

end linear_function_through_points_l3124_312496


namespace number_between_5_and_9_greater_than_7_l3124_312464

theorem number_between_5_and_9_greater_than_7 : ∃! x : ℝ, 5 < x ∧ x < 9 ∧ 7 < x := by
  sorry

end number_between_5_and_9_greater_than_7_l3124_312464


namespace table_relationship_l3124_312497

def f (x : ℝ) : ℝ := -5 * x^2 - 10 * x

theorem table_relationship : 
  (f 0 = 0) ∧ 
  (f 1 = -15) ∧ 
  (f 2 = -40) ∧ 
  (f 3 = -75) ∧ 
  (f 4 = -120) := by
  sorry

end table_relationship_l3124_312497


namespace x_squared_minus_y_squared_l3124_312434

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/12) (h2 : x - y = 1/36) : x^2 - y^2 = 5/432 := by
  sorry

end x_squared_minus_y_squared_l3124_312434


namespace calculation_proof_l3124_312457

theorem calculation_proof : (-2)^3 - |2 - 5| / (-3) = -7 := by
  sorry

end calculation_proof_l3124_312457


namespace fixed_points_condition_l3124_312458

/-- A quadratic function with parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - x + c

/-- Theorem stating the condition on c for a quadratic function with specific fixed point properties -/
theorem fixed_points_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f c x₁ = x₁ ∧ f c x₂ = x₂ ∧ x₁ < 2 ∧ 2 < x₂) →
  c < 0 :=
sorry

end fixed_points_condition_l3124_312458


namespace ellipse_k_range_l3124_312430

/-- 
Given a real number k, if the equation x²/(9-k) + y²/(k-1) = 1 represents an ellipse 
with foci on the y-axis, then 5 < k < 9.
-/
theorem ellipse_k_range (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (9 - k) + y^2 / (k - 1) = 1) → -- equation represents an ellipse
  (9 - k > 0) →  -- condition for ellipse
  (k - 1 > 0) →  -- condition for ellipse
  (k - 1 > 9 - k) →  -- foci on y-axis condition
  (5 < k ∧ k < 9) := by
sorry

end ellipse_k_range_l3124_312430


namespace cube_surface_area_l3124_312485

/-- Given a cube with volume 27 cubic cm, its surface area is 54 square cm. -/
theorem cube_surface_area (cube : Set ℝ) (volume : ℝ) (surface_area : ℝ) : 
  volume = 27 →
  surface_area = 54 :=
by sorry

end cube_surface_area_l3124_312485


namespace sum_first_and_ninth_term_l3124_312466

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sum_first_and_ninth_term : a 1 + a 9 = 19 := by
  sorry

end sum_first_and_ninth_term_l3124_312466


namespace canoe_kayak_ratio_l3124_312442

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  total_revenue : ℕ
  canoe_kayak_difference : ℕ

/-- Theorem stating the ratio of canoes to kayaks rented --/
theorem canoe_kayak_ratio (rb : RentalBusiness)
  (h1 : rb.canoe_cost = 11)
  (h2 : rb.kayak_cost = 16)
  (h3 : rb.total_revenue = 460)
  (h4 : rb.canoe_kayak_difference = 5) :
  ∃ (c k : ℕ), c = k + rb.canoe_kayak_difference ∧ 
                rb.canoe_cost * c + rb.kayak_cost * k = rb.total_revenue ∧
                c * 3 = k * 4 := by
  sorry

end canoe_kayak_ratio_l3124_312442


namespace line_passes_through_point_min_triangle_area_min_area_line_equation_l3124_312444

/-- Definition of the line l with parameter k -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 4 * k + 2 = 0

/-- Theorem stating that the line l always passes through the point (-4, 2) -/
theorem line_passes_through_point (k : ℝ) : line_l k (-4) 2 := by sorry

/-- Definition of the area of the triangle formed by the line and coordinate axes -/
noncomputable def triangle_area (k : ℝ) : ℝ := sorry

/-- Theorem stating the minimum area of the triangle -/
theorem min_triangle_area : 
  ∃ (k : ℝ), triangle_area k = 16 ∧ ∀ (k' : ℝ), triangle_area k' ≥ 16 := by sorry

/-- Theorem stating the equation of the line when the area is minimum -/
theorem min_area_line_equation (k : ℝ) : 
  triangle_area k = 16 → line_l k x y ↔ x - 2 * y + 8 = 0 := by sorry

end line_passes_through_point_min_triangle_area_min_area_line_equation_l3124_312444


namespace reciprocal_sum_fractions_l3124_312422

theorem reciprocal_sum_fractions : 
  (1 / (1/4 + 1/6 + 1/9) : ℚ) = 36/19 := by sorry

end reciprocal_sum_fractions_l3124_312422


namespace isosceles_trapezoid_side_length_l3124_312472

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- The length of the side of an isosceles trapezoid -/
def side_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that for an isosceles trapezoid with bases 7 and 13 and area 40, the side length is 5 -/
theorem isosceles_trapezoid_side_length :
  let t : IsoscelesTrapezoid := ⟨7, 13, 40⟩
  side_length t = 5 := by sorry

end isosceles_trapezoid_side_length_l3124_312472


namespace only_D_positive_l3124_312439

theorem only_D_positive :
  let a := -3 + 7 - 5
  let b := (1 - 2) * 3
  let c := -16 / ((-3)^2)
  let d := -(2^4) * (-6)
  (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0 ∧ d > 0) := by sorry

end only_D_positive_l3124_312439


namespace descending_eight_digit_numbers_count_l3124_312455

/-- The number of eight-digit numbers where each digit (except the last one) 
    is greater than the following digit. -/
def count_descending_eight_digit_numbers : ℕ :=
  Nat.choose 10 2

/-- Theorem stating that the count of eight-digit numbers with descending digits
    is equal to choosing 2 from 10. -/
theorem descending_eight_digit_numbers_count :
  count_descending_eight_digit_numbers = 45 := by
  sorry

end descending_eight_digit_numbers_count_l3124_312455


namespace square_perimeter_quadrupled_l3124_312489

theorem square_perimeter_quadrupled (s : ℝ) (x : ℝ) :
  x = 4 * s →
  4 * x = 4 * (4 * s) :=
by sorry

end square_perimeter_quadrupled_l3124_312489


namespace correct_average_l3124_312486

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 21 ∧ 
  wrong_num = 26 ∧ 
  correct_num = 36 →
  (n : ℚ) * incorrect_avg + (correct_num - wrong_num) = n * 22 :=
by sorry

end correct_average_l3124_312486


namespace f_one_zero_range_l3124_312424

/-- The quadratic function f(x) = 3ax^2 - 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * a * x + 1

/-- The property that f has exactly one zero in the interval [-1, 1] -/
def has_one_zero_in_interval (a : ℝ) : Prop :=
  ∃! x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0

/-- The theorem stating the range of a for which f has exactly one zero in [-1, 1] -/
theorem f_one_zero_range :
  ∀ a : ℝ, has_one_zero_in_interval a ↔ a = 3 ∨ (-1 < a ∧ a ≤ -1/5) :=
sorry

end f_one_zero_range_l3124_312424


namespace function_equation_solution_l3124_312433

theorem function_equation_solution (f : ℚ → ℚ) 
  (h0 : f 0 = 0)
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
sorry

end function_equation_solution_l3124_312433


namespace consecutive_odd_sum_48_l3124_312437

theorem consecutive_odd_sum_48 (a b : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  (∃ m : ℤ, b = 2*m + 1) →  -- b is odd
  b = a + 2 →               -- b is the next consecutive odd after a
  a + b = 48 →              -- sum is 48
  b = 25 :=                 -- larger number is 25
by sorry

end consecutive_odd_sum_48_l3124_312437


namespace fraction_product_theorem_l3124_312401

theorem fraction_product_theorem : 
  (7 / 4 : ℚ) * (14 / 49 : ℚ) * (10 / 15 : ℚ) * (12 / 36 : ℚ) * 
  (21 / 14 : ℚ) * (40 / 80 : ℚ) * (33 / 22 : ℚ) * (16 / 64 : ℚ) = 1 / 12 := by
  sorry

end fraction_product_theorem_l3124_312401


namespace x_less_than_y_l3124_312445

theorem x_less_than_y (a b : ℝ) (h1 : 0 < a) (h2 : a < b) 
  (x y : ℝ) (hx : x = (0.1993 : ℝ)^b * (0.1997 : ℝ)^a) 
  (hy : y = (0.1993 : ℝ)^a * (0.1997 : ℝ)^b) : x < y := by
  sorry

end x_less_than_y_l3124_312445


namespace rosa_phone_book_pages_l3124_312420

/-- Rosa's phone book calling problem -/
theorem rosa_phone_book_pages : 
  let week1_pages : ℝ := 10.2
  let week2_pages : ℝ := 8.6
  let week3_pages : ℝ := 12.4
  week1_pages + week2_pages + week3_pages = 31.2 :=
by sorry

end rosa_phone_book_pages_l3124_312420


namespace linear_function_theorem_l3124_312463

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Define the domain and range conditions
def domain_condition (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1
def range_condition (y : ℝ) : Prop := 1 ≤ y ∧ y ≤ 9

-- Theorem statement
theorem linear_function_theorem (k b : ℝ) :
  (∀ x, domain_condition x → range_condition (linear_function k b x)) →
  ((k = 2 ∧ b = 7) ∨ (k = -2 ∧ b = 3)) :=
sorry

end linear_function_theorem_l3124_312463


namespace number_ordering_l3124_312435

theorem number_ordering : (4 : ℚ) / 5 < (801 : ℚ) / 1000 ∧ (801 : ℚ) / 1000 < 81 / 100 := by
  sorry

end number_ordering_l3124_312435


namespace unique_number_l3124_312448

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem unique_number :
  ∃! n : ℕ, is_two_digit n ∧ is_odd n ∧ is_multiple_of_9 n ∧ is_perfect_square (digits_product n) ∧ n = 99 :=
sorry

end unique_number_l3124_312448


namespace circle_parabola_height_difference_l3124_312484

/-- Given a circle inside the parabola y = 4x^2, tangent at two points,
    prove the height difference between the circle's center and tangency points. -/
theorem circle_parabola_height_difference (a : ℝ) : 
  let parabola (x : ℝ) := 4 * x^2
  let tangency_point := (a, parabola a)
  let circle_center := (0, a^2 + 1/8)
  circle_center.2 - tangency_point.2 = -3 * a^2 + 1/8 :=
by sorry

end circle_parabola_height_difference_l3124_312484


namespace work_completion_time_l3124_312415

theorem work_completion_time (a_half_time b_third_time : ℝ) 
  (ha : a_half_time = 70)
  (hb : b_third_time = 35) :
  let a_rate := 1 / (2 * a_half_time)
  let b_rate := 1 / (3 * b_third_time)
  1 / (a_rate + b_rate) = 60 := by sorry

end work_completion_time_l3124_312415


namespace quadrilateral_area_l3124_312456

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def vector_dot_product (v w : ℝ × ℝ) : ℝ := sorry

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := sorry

def vector_length (v : ℝ × ℝ) : ℝ := sorry

def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area 
  (q : Quadrilateral) 
  (h_convex : is_convex q) 
  (h_bd : vector_length (vector_add q.B (vector_add q.D (-q.B))) = 2) 
  (h_perp : vector_dot_product (vector_add q.A (vector_add q.C (-q.A))) 
                               (vector_add q.B (vector_add q.D (-q.B))) = 0) 
  (h_sum : vector_dot_product (vector_add (vector_add q.A (vector_add q.B (-q.A))) 
                                          (vector_add q.D (vector_add q.C (-q.D)))) 
                              (vector_add (vector_add q.B (vector_add q.C (-q.B))) 
                                          (vector_add q.A (vector_add q.D (-q.A)))) = 5) : 
  area q = 3 := by sorry

end quadrilateral_area_l3124_312456


namespace greater_number_proof_l3124_312483

theorem greater_number_proof (x y : ℝ) 
  (sum_eq : x + y = 30)
  (diff_eq : x - y = 6)
  (prod_eq : x * y = 216) :
  max x y = 18 := by
sorry

end greater_number_proof_l3124_312483


namespace pool_filling_time_l3124_312419

theorem pool_filling_time (pipe1 pipe2 pipe3 pipe4 : ℚ) 
  (h1 : pipe1 = 1)
  (h2 : pipe2 = 1/2)
  (h3 : pipe3 = 1/3)
  (h4 : pipe4 = 1/4) :
  1 / (pipe1 + pipe2 + pipe3 + pipe4) = 12/25 := by sorry

end pool_filling_time_l3124_312419


namespace last_two_digits_sum_l3124_312475

theorem last_two_digits_sum (n : ℕ) : (9^n + 11^n) % 100 = 0 :=
by
  sorry

end last_two_digits_sum_l3124_312475


namespace sin_45_equals_sqrt2_div_2_l3124_312490

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def angle_45 (x y : ℝ) : Prop := x = y ∧ x > 0 ∧ y > 0

def right_isosceles_triangle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1 ∧ x = y ∧ x > 0 ∧ y > 0

theorem sin_45_equals_sqrt2_div_2 :
  ∀ x y : ℝ, unit_circle x y → angle_45 x y → right_isosceles_triangle x y →
  Real.sin (45 * π / 180) = Real.sqrt 2 / 2 :=
by sorry

end sin_45_equals_sqrt2_div_2_l3124_312490


namespace intersection_with_complement_l3124_312447

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_with_complement : 
  A ∩ (𝒰 \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end intersection_with_complement_l3124_312447


namespace expression_evaluation_l3124_312476

def f (x : ℚ) : ℚ := (2 * x + 2) / (x - 2)

theorem expression_evaluation :
  let x : ℚ := 3
  let result := f (f x)
  result = 8 := by sorry

end expression_evaluation_l3124_312476


namespace min_value_x2_y2_l3124_312467

theorem min_value_x2_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 196) :
  ∃ (m : ℝ), m = 169 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 196 → x^2 + y^2 ≤ a^2 + b^2 := by
  sorry

end min_value_x2_y2_l3124_312467


namespace enclosed_area_is_one_l3124_312471

-- Define the curves
def curve (x : ℝ) : ℝ := x^2 + 2
def line (x : ℝ) : ℝ := 3*x

-- Define the boundaries
def left_boundary : ℝ := 0
def right_boundary : ℝ := 2

-- Define the area function
noncomputable def area : ℝ := ∫ x in left_boundary..right_boundary, max (curve x - line x) 0 + max (line x - curve x) 0

-- Theorem statement
theorem enclosed_area_is_one : area = 1 := by sorry

end enclosed_area_is_one_l3124_312471


namespace units_digit_of_30_factorial_l3124_312487

theorem units_digit_of_30_factorial (n : ℕ) : n = 30 → n.factorial % 10 = 0 := by
  sorry

end units_digit_of_30_factorial_l3124_312487


namespace quadratic_root_value_l3124_312417

theorem quadratic_root_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 9*x + d = 0 ↔ x = (-9 + Real.sqrt d) / 2 ∨ x = (-9 - Real.sqrt d) / 2) →
  d = 16.2 := by
sorry

end quadratic_root_value_l3124_312417


namespace expression_evaluation_l3124_312408

theorem expression_evaluation : ((15^15 / 15^14)^3 * 3^5) / 9^2 = 10120 := by
  sorry

end expression_evaluation_l3124_312408


namespace opposite_of_five_l3124_312406

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_five : opposite 5 = -5 := by
  -- The proof goes here
  sorry

-- Lemma to show that the opposite satisfies the required property
lemma opposite_property (a : ℝ) : a + opposite a = 0 := by
  -- The proof goes here
  sorry

end opposite_of_five_l3124_312406


namespace video_dislikes_l3124_312491

theorem video_dislikes (likes : ℕ) (initial_dislikes : ℕ) (additional_dislikes : ℕ) : 
  likes = 3000 → 
  initial_dislikes = likes / 2 + 100 → 
  additional_dislikes = 1000 → 
  initial_dislikes + additional_dislikes = 2600 :=
by sorry

end video_dislikes_l3124_312491


namespace perpendicular_lines_from_parallel_planes_l3124_312449

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes
  (α β : Plane) (m n : Line)
  (h1 : parallel_planes α β)
  (h2 : perpendicular_line_plane m α)
  (h3 : parallel_line_plane n β) :
  perpendicular_lines m n :=
sorry

end perpendicular_lines_from_parallel_planes_l3124_312449


namespace bounded_quadratic_coef_sum_l3124_312470

/-- A quadratic polynomial f(x) = ax² + bx + c with |f(x)| ≤ 1 for all x in [0, 2] -/
def BoundedQuadratic (a b c : ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |a * x^2 + b * x + c| ≤ 1

/-- The sum of absolute values of coefficients is at most 7 -/
theorem bounded_quadratic_coef_sum (a b c : ℝ) (h : BoundedQuadratic a b c) :
  |a| + |b| + |c| ≤ 7 :=
sorry

end bounded_quadratic_coef_sum_l3124_312470


namespace initial_mixture_volume_l3124_312402

/-- Proves that the initial volume of a milk-water mixture is 45 litres given specific conditions -/
theorem initial_mixture_volume (initial_milk : ℝ) (initial_water : ℝ) : 
  initial_milk / initial_water = 4 →
  initial_milk / (initial_water + 11) = 1.8 →
  initial_milk + initial_water = 45 :=
by
  sorry

#check initial_mixture_volume

end initial_mixture_volume_l3124_312402


namespace odd_function_derivative_range_l3124_312410

open Real

theorem odd_function_derivative_range (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ∈ Set.Ioo (-1) 1, deriv f x = 5 + cos x) →  -- f'(x) = 5 + cos(x) for x ∈ (-1, 1)
  (f (1 - t) + f (1 - t^2) < 0) →  -- given condition
  t ∈ Set.Ioo 1 (sqrt 2) :=  -- t ∈ (1, √2)
by sorry

end odd_function_derivative_range_l3124_312410


namespace negation_of_existence_leq_negation_of_proposition_l3124_312446

theorem negation_of_existence_leq (p : ℝ → Prop) :
  (¬ ∃ x₀ : ℝ, p x₀) ↔ (∀ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x₀ : ℝ, Real.exp x₀ - x₀ - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end negation_of_existence_leq_negation_of_proposition_l3124_312446


namespace max_value_theorem_l3124_312428

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 2) :
  ∃ (max : ℝ), max = 25/8 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 → 1/y * (2/x + 1) ≤ max :=
sorry

end max_value_theorem_l3124_312428


namespace bigger_part_of_division_l3124_312492

theorem bigger_part_of_division (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 60) (h4 : 10 * x + 22 * y = 780) : max x y = 45 := by
  sorry

end bigger_part_of_division_l3124_312492


namespace expression_equality_l3124_312462

theorem expression_equality : 3 * 257 + 4 * 257 + 2 * 257 + 258 = 2571 := by
  sorry

end expression_equality_l3124_312462


namespace h_function_iff_strictly_increasing_l3124_312407

/-- A function f is an H-function if for any two distinct real numbers x₁ and x₂,
    x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁ -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function f is strictly increasing if for any two real numbers x₁ and x₂,
    x₁ < x₂ implies f x₁ < f x₂ -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end h_function_iff_strictly_increasing_l3124_312407


namespace distribute_four_to_three_l3124_312400

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where each box must contain at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 36 ways to distribute 4 distinct objects into 3 distinct boxes,
    where each box must contain at least one object. -/
theorem distribute_four_to_three : distribute 4 3 = 36 := by sorry

end distribute_four_to_three_l3124_312400


namespace no_water_overflow_l3124_312479

/-- Represents the dimensions and properties of a cylindrical container and an iron block. -/
structure ContainerProblem where
  container_depth : ℝ
  container_outer_diameter : ℝ
  container_wall_thickness : ℝ
  water_depth : ℝ
  block_diameter : ℝ
  block_height : ℝ

/-- Calculates the volume of water that will overflow when an iron block is placed in a cylindrical container. -/
noncomputable def water_overflow (p : ContainerProblem) : ℝ :=
  let container_inner_radius := (p.container_outer_diameter - 2 * p.container_wall_thickness) / 2
  let initial_water_volume := Real.pi * container_inner_radius ^ 2 * p.water_depth
  let container_max_volume := Real.pi * container_inner_radius ^ 2 * p.container_depth
  let block_volume := Real.pi * (p.block_diameter / 2) ^ 2 * p.block_height
  let new_total_volume := container_max_volume - block_volume
  max (initial_water_volume - new_total_volume) 0

/-- Theorem stating that no water will overflow in the given problem. -/
theorem no_water_overflow : 
  let problem : ContainerProblem := {
    container_depth := 30,
    container_outer_diameter := 22,
    container_wall_thickness := 1,
    water_depth := 27.5,
    block_diameter := 10,
    block_height := 30
  }
  water_overflow problem = 0 := by sorry

end no_water_overflow_l3124_312479


namespace factorization_x4_plus_81_l3124_312427

theorem factorization_x4_plus_81 (x : ℂ) : x^4 + 81 = (x^2 + 9*I)*(x^2 - 9*I) := by
  sorry

end factorization_x4_plus_81_l3124_312427


namespace cuboid_dimensions_sum_l3124_312425

theorem cuboid_dimensions_sum (A B C : ℝ) (h1 : A * B = 45) (h2 : B * C = 80) (h3 : C * A = 180) :
  A + B + C = 145 / 9 := by
sorry

end cuboid_dimensions_sum_l3124_312425


namespace min_manhattan_distance_l3124_312474

-- Define the manhattan distance function
def manhattan_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

-- Define the line
def on_line (x y : ℝ) : Prop :=
  3 * x + 4 * y - 12 = 0

-- State the theorem
theorem min_manhattan_distance :
  ∃ (min_dist : ℝ),
    min_dist = (12 - Real.sqrt 34) / 4 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      on_ellipse x₁ y₁ → on_line x₂ y₂ →
      manhattan_distance x₁ y₁ x₂ y₂ ≥ min_dist :=
by
  sorry

end min_manhattan_distance_l3124_312474


namespace intersection_with_complement_l3124_312411

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_with_complement : A ∩ (Set.univ \ B) = Set.Icc 2 3 := by
  sorry

end intersection_with_complement_l3124_312411


namespace smallest_product_increase_l3124_312488

theorem smallest_product_increase (p q r s : ℝ) 
  (h_pos : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s) 
  (h_order : p < q ∧ q < r ∧ r < s) : 
  min (min (min ((p+1)*q*r*s) (p*(q+1)*r*s)) (p*q*(r+1)*s)) (p*q*r*(s+1)) = p*q*r*(s+1) := by
  sorry

end smallest_product_increase_l3124_312488


namespace handball_league_female_fraction_l3124_312451

/-- Represents the handball league participation data --/
structure LeagueData where
  male_last_year : ℕ
  total_increase_rate : ℚ
  male_increase_rate : ℚ
  female_increase_rate : ℚ

/-- Calculates the fraction of female participants in the current year --/
def female_fraction (data : LeagueData) : ℚ :=
  -- The actual calculation would go here
  13/27

/-- Theorem stating that given the specific conditions, the fraction of female participants is 13/27 --/
theorem handball_league_female_fraction :
  let data : LeagueData := {
    male_last_year := 25,
    total_increase_rate := 1/5,  -- 20% increase
    male_increase_rate := 1/10,  -- 10% increase
    female_increase_rate := 3/10 -- 30% increase
  }
  female_fraction data = 13/27 := by
  sorry


end handball_league_female_fraction_l3124_312451


namespace power_difference_equality_l3124_312426

theorem power_difference_equality : (3^2)^3 - (2^3)^2 = 665 := by sorry

end power_difference_equality_l3124_312426


namespace correct_average_l3124_312443

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = n * 16 := by
  sorry

end correct_average_l3124_312443


namespace parabola_shift_l3124_312453

/-- A parabola shifted 1 unit left and 4 units down -/
def shifted_parabola (x : ℝ) : ℝ := 3 * (x + 1)^2 - 4

/-- The original parabola -/
def original_parabola (x : ℝ) : ℝ := 3 * x^2

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) - 4 :=
by sorry

end parabola_shift_l3124_312453


namespace digital_earth_functions_l3124_312495

-- Define the concept of Digital Earth
structure DigitalEarth where
  integratesInfo : Bool
  displaysIn3D : Bool
  isDynamic : Bool
  providesExperimentalConditions : Bool

-- Define the correct description of Digital Earth functions
def correctDescription (de : DigitalEarth) : Prop :=
  de.integratesInfo ∧ de.displaysIn3D ∧ de.isDynamic ∧ de.providesExperimentalConditions

-- Theorem stating that the correct description accurately represents Digital Earth functions
theorem digital_earth_functions :
  ∀ (de : DigitalEarth), correctDescription de ↔ 
    (de.integratesInfo = true ∧ 
     de.displaysIn3D = true ∧ 
     de.isDynamic = true ∧ 
     de.providesExperimentalConditions = true) :=
by
  sorry

#check digital_earth_functions

end digital_earth_functions_l3124_312495


namespace tangent_circle_equation_l3124_312482

/-- A circle with radius 5, center on the x-axis, and tangent to the line x=3 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : (center.2 = 0)
  radius_is_5 : radius = 5
  tangent_to_x3 : |center.1 - 3| = 5

/-- The equation of the circle is (x-8)^2 + y^2 = 25 or (x+2)^2 + y^2 = 25 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y : ℝ, (x - 8)^2 + y^2 = 25 ∨ (x + 2)^2 + y^2 = 25 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end tangent_circle_equation_l3124_312482


namespace hyperbola_circle_intersection_eccentricity_l3124_312465

/-- Given a hyperbola and a circle that intersect to form a square, 
    prove that the eccentricity of the hyperbola is √(2 + √2) -/
theorem hyperbola_circle_intersection_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c = Real.sqrt (a^2 + b^2)) 
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → x^2 + y^2 = c^2 → x^2 = y^2) : 
  c / a = Real.sqrt (2 + Real.sqrt 2) := by
  sorry

end hyperbola_circle_intersection_eccentricity_l3124_312465


namespace arithmetic_sequence_2005_l3124_312454

/-- Given an arithmetic sequence {a_n} with first term a₁ = 1 and common difference d = 3,
    prove that the value of n for which aₙ = 2005 is 669. -/
theorem arithmetic_sequence_2005 (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = 3) →  -- Common difference is 3
  a 1 = 1 →                    -- First term is 1
  ∃ n : ℕ, a n = 2005 ∧ n = 669 :=
by sorry

end arithmetic_sequence_2005_l3124_312454


namespace quadratic_distinct_roots_l3124_312429

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < 1/4 := by
  sorry

end quadratic_distinct_roots_l3124_312429


namespace selection_count_theorem_l3124_312477

/-- Represents a grid of people -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a selection of people from the grid -/
structure Selection :=
  (grid : Grid)
  (num_selected : ℕ)

/-- Counts the number of valid selections -/
def count_valid_selections (s : Selection) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem selection_count_theorem (g : Grid) (s : Selection) :
  g.rows = 6 ∧ g.cols = 7 ∧ s.grid = g ∧ s.num_selected = 3 →
  count_valid_selections s = 4200 :=
sorry

end selection_count_theorem_l3124_312477


namespace composition_result_l3124_312450

-- Define the two operations
def op1 (x : ℝ) : ℝ := 8 - x
def op2 (x : ℝ) : ℝ := x - 8

-- Notation for the operations
notation:max x "&" => op1 x
prefix:max "&" => op2

-- Theorem statement
theorem composition_result : &(15&) = -15 := by sorry

end composition_result_l3124_312450


namespace encoded_bec_value_l3124_312413

/-- Represents the encoding of a base 7 digit --/
inductive Encoding
  | A | B | C | D | E | F | G

/-- Represents a number in the encoded form --/
def EncodedNumber := List Encoding

/-- Converts an EncodedNumber to its base 10 representation --/
def to_base_10 (n : EncodedNumber) : ℕ := sorry

/-- Checks if three EncodedNumbers are consecutive integers --/
def are_consecutive (a b c : EncodedNumber) : Prop := sorry

theorem encoded_bec_value :
  ∃ (encode : Fin 7 → Encoding),
    Function.Injective encode ∧
    (∃ (x : ℕ), 
      are_consecutive 
        [encode (x % 7), encode ((x + 1) % 7), encode ((x + 2) % 7)]
        [encode ((x + 1) % 7), encode ((x + 2) % 7), encode ((x + 3) % 7)]
        [encode ((x + 2) % 7), encode ((x + 3) % 7), encode ((x + 4) % 7)]) →
    to_base_10 [Encoding.B, Encoding.E, Encoding.C] = 336 :=
sorry

end encoded_bec_value_l3124_312413


namespace sarah_speed_calculation_l3124_312423

def eugene_speed : ℚ := 5

def carlos_speed_ratio : ℚ := 4/5

def sarah_speed_ratio : ℚ := 6/7

def carlos_speed : ℚ := eugene_speed * carlos_speed_ratio

def sarah_speed : ℚ := carlos_speed * sarah_speed_ratio

theorem sarah_speed_calculation : sarah_speed = 24/7 := by
  sorry

end sarah_speed_calculation_l3124_312423


namespace marie_messages_theorem_l3124_312404

/-- Calculates the number of days required to read all unread messages. -/
def daysToReadMessages (initialUnread : ℕ) (readPerDay : ℕ) (newPerDay : ℕ) : ℕ :=
  if readPerDay ≤ newPerDay then 0  -- Cannot finish if receiving more than reading
  else (initialUnread + (newPerDay - 1)) / (readPerDay - newPerDay)

theorem marie_messages_theorem :
  daysToReadMessages 98 20 6 = 7 := by
sorry

end marie_messages_theorem_l3124_312404


namespace manufacturing_degrees_l3124_312438

/-- Represents the number of degrees in a full circle. -/
def full_circle : ℝ := 360

/-- Represents the percentage of employees in manufacturing as a decimal. -/
def manufacturing_percentage : ℝ := 0.20

/-- Calculates the number of degrees in a circle graph for a given percentage. -/
def degrees_for_percentage (percentage : ℝ) : ℝ := full_circle * percentage

/-- Theorem: The manufacturing section in the circle graph takes up 72 degrees. -/
theorem manufacturing_degrees :
  degrees_for_percentage manufacturing_percentage = 72 := by
  sorry

end manufacturing_degrees_l3124_312438


namespace intersection_distance_is_sqrt_2_l3124_312473

-- Define the two equations
def equation1 (x y : ℝ) : Prop := x^2 + y = 12
def equation2 (x y : ℝ) : Prop := x + y = 12

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ equation1 x y ∧ equation2 x y}

-- State the theorem
theorem intersection_distance_is_sqrt_2 :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt 2 = Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) :=
sorry

end intersection_distance_is_sqrt_2_l3124_312473


namespace inscribed_triangle_area_bound_l3124_312494

/-- A convex polygon -/
structure ConvexPolygon where
  -- Define properties of a convex polygon
  area : ℝ
  is_convex : Bool

/-- A line in 2D space -/
structure Line where
  -- Define properties of a line

/-- A triangle inscribed in a polygon -/
structure InscribedTriangle (M : ConvexPolygon) where
  -- Define properties of an inscribed triangle
  area : ℝ
  side_parallel_to : Line

/-- Theorem statement -/
theorem inscribed_triangle_area_bound (M : ConvexPolygon) (l : Line) :
  (∃ T : InscribedTriangle M, T.side_parallel_to = l ∧ T.area ≥ 3/8 * M.area) ∧
  (∃ M' : ConvexPolygon, ∃ l' : Line, 
    ∀ T : InscribedTriangle M', T.side_parallel_to = l' → T.area ≤ 3/8 * M'.area) :=
by sorry

end inscribed_triangle_area_bound_l3124_312494


namespace ellipse_axis_endpoints_distance_l3124_312409

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 4(x-2)^2 + 16y^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoints_distance : 
  let ellipse := {p : ℝ × ℝ | 4 * (p.1 - 2)^2 + 16 * p.2^2 = 64}
  let major_axis_endpoint := {p : ℝ × ℝ | p ∈ ellipse ∧ p.2 = 0 ∧ p.1 ≠ 2}
  let minor_axis_endpoint := {p : ℝ × ℝ | p ∈ ellipse ∧ p.1 = 2 ∧ p.2 ≠ 0}
  ∀ C ∈ major_axis_endpoint, ∀ D ∈ minor_axis_endpoint, 
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end ellipse_axis_endpoints_distance_l3124_312409


namespace students_taking_neither_music_nor_art_l3124_312493

theorem students_taking_neither_music_nor_art 
  (total : ℕ) (music : ℕ) (art : ℕ) (both : ℕ) :
  total = 500 →
  music = 30 →
  art = 20 →
  both = 10 →
  total - (music + art - both) = 460 :=
by sorry

end students_taking_neither_music_nor_art_l3124_312493


namespace pecan_pies_count_l3124_312499

def total_pies : ℕ := 13
def apple_pies : ℕ := 2
def pumpkin_pies : ℕ := 7

theorem pecan_pies_count : total_pies - apple_pies - pumpkin_pies = 4 := by
  sorry

end pecan_pies_count_l3124_312499


namespace difference_even_odd_sums_l3124_312452

/-- Sum of first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumFirstOddIntegers (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : 
  (sumFirstEvenIntegers 25) - (sumFirstOddIntegers 20) = 250 := by
  sorry

end difference_even_odd_sums_l3124_312452


namespace basketball_team_selection_l3124_312418

theorem basketball_team_selection (n : ℕ) (k : ℕ) (twins : ℕ) : 
  n = 15 → k = 5 → twins = 2 →
  (Nat.choose n k) - (Nat.choose (n - twins) k) = 1716 := by
sorry

end basketball_team_selection_l3124_312418


namespace sum_inequality_l3124_312460

theorem sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_sum : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2) :
  (1 - a) / (a^2 - a + 1) + (1 - b) / (b^2 - b + 1) + 
  (1 - c) / (c^2 - c + 1) + (1 - d) / (d^2 - d + 1) ≥ 0 := by
  sorry

end sum_inequality_l3124_312460


namespace positive_x_solution_l3124_312432

/-- Given a system of equations, prove that the positive solution for x is 3 -/
theorem positive_x_solution (x y z : ℝ) 
  (eq1 : x * y = 6 - 2*x - 3*y)
  (eq2 : y * z = 6 - 4*y - 2*z)
  (eq3 : x * z = 30 - 4*x - 3*z)
  (x_pos : x > 0) :
  x = 3 := by
  sorry

end positive_x_solution_l3124_312432


namespace unique_positive_solution_l3124_312480

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ Real.sin (Real.arccos (Real.tanh (Real.arcsin x))) = x ∧ x = Real.sqrt (1/2) := by
  sorry

end unique_positive_solution_l3124_312480


namespace power_fraction_simplification_l3124_312478

theorem power_fraction_simplification :
  (6^5 * 3^5) / 18^4 = 18 := by
  sorry

end power_fraction_simplification_l3124_312478
