import Mathlib

namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l1713_171380

def f (x : ℝ) := x^2 + 12*x - 15

theorem solution_exists_in_interval :
  ∃ x ∈ Set.Ioo 1.1 1.2, f x = 0 :=
by
  have h1 : f 1.1 < 0 := by sorry
  have h2 : f 1.2 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_solution_exists_in_interval_l1713_171380


namespace NUMINAMATH_CALUDE_post_office_problem_l1713_171393

theorem post_office_problem (total_spent : ℚ) (letter_cost : ℚ) (package_cost : ℚ) 
  (h1 : total_spent = 449/100)
  (h2 : letter_cost = 37/100)
  (h3 : package_cost = 88/100)
  : ∃ (letters packages : ℕ), 
    letters = packages + 2 ∧ 
    letter_cost * letters + package_cost * packages = total_spent ∧
    letters = 5 := by
  sorry

end NUMINAMATH_CALUDE_post_office_problem_l1713_171393


namespace NUMINAMATH_CALUDE_student_rank_l1713_171390

theorem student_rank (total : Nat) (rank_right : Nat) (rank_left : Nat) : 
  total = 21 → rank_right = 17 → rank_left = total - rank_right + 1 → rank_left = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_l1713_171390


namespace NUMINAMATH_CALUDE_min_n_is_15_l1713_171314

/-- A type representing the vertices of a regular 9-sided polygon -/
inductive Vertex : Type
  | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9

/-- A function type representing an assignment of integers to vertices -/
def Assignment := Vertex → Fin 9

/-- Predicate to check if an assignment is valid (each integer used once) -/
def is_valid_assignment (f : Assignment) : Prop :=
  ∀ i j : Vertex, i ≠ j → f i ≠ f j

/-- Function to get the next vertex in cyclic order -/
def next_vertex : Vertex → Vertex
  | Vertex.v1 => Vertex.v2
  | Vertex.v2 => Vertex.v3
  | Vertex.v3 => Vertex.v4
  | Vertex.v4 => Vertex.v5
  | Vertex.v5 => Vertex.v6
  | Vertex.v6 => Vertex.v7
  | Vertex.v7 => Vertex.v8
  | Vertex.v8 => Vertex.v9
  | Vertex.v9 => Vertex.v1

/-- Predicate to check if the sum of any three consecutive vertices does not exceed n -/
def satisfies_sum_condition (f : Assignment) (n : ℕ) : Prop :=
  ∀ v : Vertex, (f v).val + 1 + (f (next_vertex v)).val + 1 + (f (next_vertex (next_vertex v))).val + 1 ≤ n

/-- The main theorem: the minimum value of n is 15 -/
theorem min_n_is_15 :
  ∃ (f : Assignment), is_valid_assignment f ∧ satisfies_sum_condition f 15 ∧
  ∀ (m : ℕ), m < 15 → ¬∃ (g : Assignment), is_valid_assignment g ∧ satisfies_sum_condition g m :=
sorry

end NUMINAMATH_CALUDE_min_n_is_15_l1713_171314


namespace NUMINAMATH_CALUDE_unique_solution_2a3b_7c_l1713_171366

theorem unique_solution_2a3b_7c : ∃! (a b c : ℕ+), 2^(a:ℕ) * 3^(b:ℕ) = 7^(c:ℕ) - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_2a3b_7c_l1713_171366


namespace NUMINAMATH_CALUDE_hockey_league_teams_l1713_171394

/-- The number of teams in a hockey league. -/
def num_teams : ℕ := 16

/-- The number of times each team faces every other team. -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season. -/
def total_games : ℕ := 1200

/-- Theorem stating that the number of teams is correct given the conditions. -/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) * games_per_pair) / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l1713_171394


namespace NUMINAMATH_CALUDE_second_car_speed_l1713_171361

/-- Proves that the speed of the second car is 70 km/h given the conditions of the problem -/
theorem second_car_speed (initial_distance : ℝ) (first_car_speed : ℝ) (time : ℝ) :
  initial_distance = 60 →
  first_car_speed = 90 →
  time = 3 →
  ∃ (second_car_speed : ℝ),
    second_car_speed * time + initial_distance = first_car_speed * time ∧
    second_car_speed = 70 :=
by
  sorry


end NUMINAMATH_CALUDE_second_car_speed_l1713_171361


namespace NUMINAMATH_CALUDE_line_y_intercept_l1713_171331

/-- Given a line with equation 3x - y + 6 = 0, prove that its y-intercept is 6 -/
theorem line_y_intercept (x y : ℝ) (h : 3 * x - y + 6 = 0) : y = 6 ↔ x = 0 :=
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l1713_171331


namespace NUMINAMATH_CALUDE_negation_of_conjunction_l1713_171302

theorem negation_of_conjunction (x y : ℝ) : 
  ¬(x = 2 ∧ y = 3) ↔ (x ≠ 2 ∨ y ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_conjunction_l1713_171302


namespace NUMINAMATH_CALUDE_anya_hair_growth_l1713_171321

/-- The number of hairs Anya washes down the drain -/
def hairs_washed : ℕ := 32

/-- The number of hairs Anya brushes out -/
def hairs_brushed : ℕ := hairs_washed / 2

/-- The number of hairs Anya needs to grow back -/
def hairs_to_grow : ℕ := 49

/-- The total number of additional hairs Anya wants to have -/
def additional_hairs : ℕ := hairs_washed + hairs_brushed + hairs_to_grow

theorem anya_hair_growth :
  additional_hairs = 97 := by sorry

end NUMINAMATH_CALUDE_anya_hair_growth_l1713_171321


namespace NUMINAMATH_CALUDE_classroom_ratio_l1713_171341

theorem classroom_ratio :
  ∀ (x y : ℕ),
    x + y = 15 →
    30 * x + 25 * y = 400 →
    x / 15 = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_l1713_171341


namespace NUMINAMATH_CALUDE_round_recurring_decimal_to_thousandth_l1713_171370

/-- The repeating decimal 36.3636... -/
def recurring_decimal : ℚ := 36 + 36 / 99

/-- Rounding a number to the nearest thousandth -/
def round_to_thousandth (x : ℚ) : ℚ := 
  (⌊x * 1000 + 0.5⌋) / 1000

/-- Proof that rounding 36.3636... to the nearest thousandth equals 36.363 -/
theorem round_recurring_decimal_to_thousandth : 
  round_to_thousandth recurring_decimal = 36363 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_round_recurring_decimal_to_thousandth_l1713_171370


namespace NUMINAMATH_CALUDE_number_count_l1713_171345

theorem number_count (avg_all : Real) (avg1 : Real) (avg2 : Real) (avg3 : Real) 
  (h1 : avg_all = 3.95)
  (h2 : avg1 = 3.8)
  (h3 : avg2 = 3.85)
  (h4 : avg3 = 4.200000000000001)
  (h5 : 2 * avg1 + 2 * avg2 + 2 * avg3 = avg_all * 6) :
  6 = (2 * avg1 + 2 * avg2 + 2 * avg3) / avg_all := by
  sorry

end NUMINAMATH_CALUDE_number_count_l1713_171345


namespace NUMINAMATH_CALUDE_committee_probability_l1713_171320

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 5

theorem committee_probability :
  let total_combinations := Nat.choose total_members committee_size
  let all_boys_or_all_girls := 2 * Nat.choose boys committee_size
  (total_combinations - all_boys_or_all_girls : ℚ) / total_combinations = 5115 / 5313 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l1713_171320


namespace NUMINAMATH_CALUDE_horner_method_op_count_for_f_l1713_171313

/-- Horner's method operation count for polynomial evaluation -/
def horner_op_count (coeffs : List ℝ) : ℕ :=
  match coeffs with
  | [] => 0
  | [_] => 0
  | _ :: tail => 2 * (tail.length)

/-- The polynomial f(x) = x^5 + 4x^4 + 3x^3 + 2x^2 + 1 -/
def f_coeffs : List ℝ := [1, 4, 3, 2, 0, 1]

theorem horner_method_op_count_for_f :
  horner_op_count f_coeffs = 8 := by sorry

end NUMINAMATH_CALUDE_horner_method_op_count_for_f_l1713_171313


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1713_171329

theorem rectangle_measurement_error (L W : ℝ) (p : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let measured_area := (1.05 * L) * (W * (1 - p))
  let actual_area := L * W
  let error_percent := |measured_area - actual_area| / actual_area
  error_percent = 0.008 → p = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1713_171329


namespace NUMINAMATH_CALUDE_calculate_product_l1713_171371

theorem calculate_product : 150 * 22.5 * (1.5^2) * 10 = 75937.5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_product_l1713_171371


namespace NUMINAMATH_CALUDE_zoes_flower_purchase_l1713_171369

theorem zoes_flower_purchase (flower_price : ℕ) (roses_bought : ℕ) (total_spent : ℕ) : 
  flower_price = 3 →
  roses_bought = 8 →
  total_spent = 30 →
  (total_spent - roses_bought * flower_price) / flower_price = 2 := by
sorry

end NUMINAMATH_CALUDE_zoes_flower_purchase_l1713_171369


namespace NUMINAMATH_CALUDE_unique_solution_system_l1713_171353

theorem unique_solution_system (x y : ℝ) : 
  (3 * x ≥ 2 * y + 16 ∧ 
   x^4 + 2 * x^2 * y^2 + y^4 + 25 - 26 * x^2 - 26 * y^2 = 72 * x * y) ↔ 
  (x = 6 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1713_171353


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l1713_171358

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 - y^2/m = 1

-- Define the focus point
def focus : ℝ × ℝ := (-3, 0)

-- Theorem statement
theorem hyperbola_m_value :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), hyperbola_equation x y m → True) ∧ 
    (focus.1^2 = 1 + m) →
    m = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l1713_171358


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l1713_171364

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Parabola y^2 = 4x -/
def isOnParabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- Line x = -1 -/
def isTangentToLine (c : Circle) : Prop :=
  c.center.x + c.radius = -1

/-- Check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem -/
theorem circle_passes_through_fixed_point :
  ∀ (c : Circle),
  isOnParabola c.center →
  isTangentToLine c →
  isOnCircle ⟨1, 0⟩ c :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l1713_171364


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1713_171399

/-- The focal length of a hyperbola with equation x²/m - y² = 1 (m > 0) 
    and asymptote √3x + my = 0 is 4 -/
theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / m - y^2 = 1
  let asymptote : ℝ → ℝ → Prop := λ x y => Real.sqrt 3 * x + m * y = 0
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    (∀ x y, C x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (∀ x y, asymptote x y ↔ y = -(Real.sqrt 3 / m) * x) ∧
    c^2 = a^2 + b^2 ∧
    2 * c = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1713_171399


namespace NUMINAMATH_CALUDE_circle_M_properties_l1713_171330

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the center of the circle
def center_M : ℝ × ℝ := (1, -2)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, -1)

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem circle_M_properties :
  (center_M.2 = -2 * center_M.1) ∧ 
  tangent_line point_P.1 point_P.2 ∧
  (∀ x y, tangent_line x y → ¬ circle_M x y) ∧
  circle_M point_P.1 point_P.2 →
  (∀ x y, circle_M x y → 
    Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2) ≥ Real.sqrt 2) ∧
  (∃ x y, circle_M x y ∧ 
    Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_M_properties_l1713_171330


namespace NUMINAMATH_CALUDE_blue_crayon_boxes_l1713_171384

/-- Given information about crayon boxes and their contents, prove the number of blue crayon boxes -/
theorem blue_crayon_boxes (total_crayons : ℕ) (orange_boxes : ℕ) (orange_per_box : ℕ) 
  (red_boxes : ℕ) (red_per_box : ℕ) (blue_per_box : ℕ) :
  total_crayons = 94 →
  orange_boxes = 6 →
  orange_per_box = 8 →
  red_boxes = 1 →
  red_per_box = 11 →
  blue_per_box = 5 →
  ∃ (blue_boxes : ℕ), 
    total_crayons = orange_boxes * orange_per_box + red_boxes * red_per_box + blue_boxes * blue_per_box ∧
    blue_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_blue_crayon_boxes_l1713_171384


namespace NUMINAMATH_CALUDE_shaded_square_area_l1713_171391

/- Define the structure of the lawn -/
structure Lawn :=
  (total_area : ℝ)
  (rectangle_area : ℝ)
  (is_square : Bool)
  (has_four_rectangles : Bool)
  (has_square_in_rectangle : Bool)

/- Define the properties of the lawn -/
def lawn_properties (l : Lawn) : Prop :=
  l.is_square ∧ 
  l.has_four_rectangles ∧ 
  l.rectangle_area = 40 ∧
  l.has_square_in_rectangle

/- Theorem statement -/
theorem shaded_square_area (l : Lawn) :
  lawn_properties l →
  ∃ (square_area : ℝ), square_area = 2500 / 441 :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_l1713_171391


namespace NUMINAMATH_CALUDE_sqrt_9800_simplification_l1713_171351

theorem sqrt_9800_simplification : Real.sqrt 9800 = 70 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_9800_simplification_l1713_171351


namespace NUMINAMATH_CALUDE_total_marks_math_physics_l1713_171318

/-- Proves that the total marks in mathematics and physics is 60 -/
theorem total_marks_math_physics (math physics chemistry : ℕ) : 
  chemistry = physics + 20 →
  (math + chemistry) / 2 = 40 →
  math + physics = 60 := by
sorry

end NUMINAMATH_CALUDE_total_marks_math_physics_l1713_171318


namespace NUMINAMATH_CALUDE_triangle_properties_l1713_171305

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- Given conditions for the specific triangle -/
def special_triangle (t : AcuteTriangle) : Prop :=
  t.a = 2 * t.b * Real.sin t.A ∧ t.a = 3 * Real.sqrt 3 ∧ t.c = 5

theorem triangle_properties (t : AcuteTriangle) (h : special_triangle t) : 
  t.B = π/6 ∧ t.b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1713_171305


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l1713_171368

theorem multiplication_puzzle :
  ∀ (A B E F : ℕ),
    A < 10 → B < 10 → E < 10 → F < 10 →
    A ≠ B → A ≠ E → A ≠ F → B ≠ E → B ≠ F → E ≠ F →
    (100 * A + 10 * B + E) * F = 1000 * E + 100 * A + 10 * E + A →
    A + B = 5 := by
sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l1713_171368


namespace NUMINAMATH_CALUDE_order_of_t_squared_t_neg_t_l1713_171375

theorem order_of_t_squared_t_neg_t (t : ℝ) (h : t^2 + t < 0) : t < t^2 ∧ t^2 < -t := by
  sorry

end NUMINAMATH_CALUDE_order_of_t_squared_t_neg_t_l1713_171375


namespace NUMINAMATH_CALUDE_range_of_x_l1713_171397

theorem range_of_x (M : Set ℝ) (h : M = {x ^ 2 | x : ℝ} ∪ {1}) :
  {x : ℝ | x ≠ 1 ∧ x ≠ -1} = {x : ℝ | ∃ y ∈ M, y = x ^ 2} := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1713_171397


namespace NUMINAMATH_CALUDE_cyclists_speed_cyclists_speed_is_10_l1713_171333

/-- Two cyclists traveling in opposite directions for 2.5 hours end up 50 km apart. -/
theorem cyclists_speed : ℝ → Prop :=
  fun speed : ℝ =>
    let time : ℝ := 2.5
    let distance : ℝ := 50
    2 * speed * time = distance

/-- The speed of each cyclist is 10 km/h. -/
theorem cyclists_speed_is_10 : cyclists_speed 10 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_speed_cyclists_speed_is_10_l1713_171333


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1713_171332

/-- A trinomial x^2 + kx + 9 is a perfect square if and only if k = 6 or k = -6 -/
theorem perfect_square_trinomial (k : ℝ) : 
  (∃ (a b : ℝ), ∀ x, x^2 + k*x + 9 = (a*x + b)^2) ↔ (k = 6 ∨ k = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1713_171332


namespace NUMINAMATH_CALUDE_email_difference_is_six_l1713_171373

/-- Calculates the difference between morning and afternoon emails --/
def email_difference (early_morning late_morning early_afternoon late_afternoon : ℕ) : ℕ :=
  (early_morning + late_morning) - (early_afternoon + late_afternoon)

/-- Theorem stating the difference between morning and afternoon emails is 6 --/
theorem email_difference_is_six :
  email_difference 10 15 7 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_is_six_l1713_171373


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1713_171363

noncomputable def g (x : ℝ) : ℤ :=
  if x > 3 then
    ⌈(1 : ℝ) / (x - 3)⌉
  else if x < 3 then
    ⌊(1 : ℝ) / (x - 3)⌋
  else
    0  -- This value doesn't matter as g is not defined at x = 3

theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ 3 → g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1713_171363


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l1713_171342

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_and_extrema :
  ∃ (tangent_line : ℝ → ℝ) (max_value min_value : ℝ),
    (∀ x, tangent_line x = 1) ∧
    (f 0 = max_value) ∧
    (f (Real.pi / 2) = min_value) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_value) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ min_value) ∧
    (max_value = 1) ∧
    (min_value = -Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l1713_171342


namespace NUMINAMATH_CALUDE_max_value_P_l1713_171396

theorem max_value_P (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 2) :
  let P := (Real.sqrt (b^2 + c^2)) / (3 - a) + (Real.sqrt (c^2 + a^2)) / (3 - b) + a + b - 2022 * c
  P ≤ 3 ∧ (P = 3 ↔ a = 1 ∧ b = 1 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_P_l1713_171396


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1713_171382

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 + 3*x - 4 = 0
def equation2 (x : ℝ) : Prop := 2*x^2 - 4*x - 1 = 0

-- Theorem for the solutions of the first equation
theorem solutions_equation1 : 
  (∃ x : ℝ, equation1 x) ↔ (equation1 1 ∧ equation1 (-4)) :=
sorry

-- Theorem for the solutions of the second equation
theorem solutions_equation2 : 
  (∃ x : ℝ, equation2 x) ↔ (equation2 (1 + Real.sqrt 6 / 2) ∧ equation2 (1 - Real.sqrt 6 / 2)) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1713_171382


namespace NUMINAMATH_CALUDE_parameterization_validity_l1713_171306

def is_valid_parameterization (a b c d : ℝ) : Prop :=
  b = 3 * a + 4 ∧ d = 3 * c

theorem parameterization_validity (a b c d : ℝ) :
  is_valid_parameterization a b c d ↔
  ∀ t : ℝ, (3 * (a + c * t) + 4 = b + d * t) :=
by sorry

end NUMINAMATH_CALUDE_parameterization_validity_l1713_171306


namespace NUMINAMATH_CALUDE_simplify_expression_l1713_171344

theorem simplify_expression (n : ℕ) : 
  (3 * 2^(n+5) - 5 * 2^n) / (4 * 2^(n+2)) = 91 / 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1713_171344


namespace NUMINAMATH_CALUDE_starting_lineups_count_l1713_171317

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 5 starters from a team of 15 players,
    including a set of 4 quadruplets, with exactly one of the quadruplets
    in the starting lineup -/
def startingLineups : ℕ :=
  4 * binomial 11 4

theorem starting_lineups_count :
  startingLineups = 1320 := by sorry

end NUMINAMATH_CALUDE_starting_lineups_count_l1713_171317


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l1713_171355

theorem factorial_fraction_equals_one :
  (3 * Nat.factorial 5 + 15 * Nat.factorial 4) / Nat.factorial 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l1713_171355


namespace NUMINAMATH_CALUDE_smallest_square_side_length_l1713_171326

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The problem statement -/
theorem smallest_square_side_length 
  (rect1 : Rectangle)
  (rect2 : Rectangle)
  (h1 : rect1.width = 2 ∧ rect1.height = 4)
  (h2 : rect2.width = 4 ∧ rect2.height = 5)
  (h3 : ∀ (s : ℝ), s ≥ 0 → 
    (∃ (x1 y1 x2 y2 : ℝ), 
      0 ≤ x1 ∧ x1 + rect1.width ≤ s ∧
      0 ≤ y1 ∧ y1 + rect1.height ≤ s ∧
      0 ≤ x2 ∧ x2 + rect2.width ≤ s ∧
      0 ≤ y2 ∧ y2 + rect2.height ≤ s ∧
      (x1 + rect1.width ≤ x2 ∨ x2 + rect2.width ≤ x1 ∨
       y1 + rect1.height ≤ y2 ∨ y2 + rect2.height ≤ y1))) :
  (∀ (s : ℝ), s ≥ 0 ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      0 ≤ x1 ∧ x1 + rect1.width ≤ s ∧
      0 ≤ y1 ∧ y1 + rect1.height ≤ s ∧
      0 ≤ x2 ∧ x2 + rect2.width ≤ s ∧
      0 ≤ y2 ∧ y2 + rect2.height ≤ s ∧
      (x1 + rect1.width ≤ x2 ∨ x2 + rect2.width ≤ x1 ∨
       y1 + rect1.height ≤ y2 ∨ y2 + rect2.height ≤ y1)) → s ≥ 6) ∧
  (∃ (x1 y1 x2 y2 : ℝ), 
    0 ≤ x1 ∧ x1 + rect1.width ≤ 6 ∧
    0 ≤ y1 ∧ y1 + rect1.height ≤ 6 ∧
    0 ≤ x2 ∧ x2 + rect2.width ≤ 6 ∧
    0 ≤ y2 ∧ y2 + rect2.height ≤ 6 ∧
    (x1 + rect1.width ≤ x2 ∨ x2 + rect2.width ≤ x1 ∨
     y1 + rect1.height ≤ y2 ∨ y2 + rect2.height ≤ y1)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_side_length_l1713_171326


namespace NUMINAMATH_CALUDE_abs_sum_values_l1713_171301

theorem abs_sum_values (a b : ℝ) (ha : |a| = 3) (hb : |b| = 1) :
  |a + b| = 4 ∨ |a + b| = 2 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_values_l1713_171301


namespace NUMINAMATH_CALUDE_common_root_condition_l1713_171385

theorem common_root_condition (m : ℝ) : 
  (∃ x : ℝ, m * x - 1000 = 1001 ∧ 1001 * x = m - 1000 * x) ↔ (m = 2001 ∨ m = -2001) := by
  sorry

end NUMINAMATH_CALUDE_common_root_condition_l1713_171385


namespace NUMINAMATH_CALUDE_show_charge_day3_l1713_171337

/-- The charge per person on the first day in rupees -/
def charge_day1 : ℚ := 15

/-- The charge per person on the second day in rupees -/
def charge_day2 : ℚ := 15/2

/-- The ratio of attendance on the first day -/
def ratio_day1 : ℕ := 2

/-- The ratio of attendance on the second day -/
def ratio_day2 : ℕ := 5

/-- The ratio of attendance on the third day -/
def ratio_day3 : ℕ := 13

/-- The average charge per person for the whole show in rupees -/
def average_charge : ℚ := 5

/-- The charge per person on the third day in rupees -/
def charge_day3 : ℚ := 5/2

theorem show_charge_day3 :
  let total_ratio := ratio_day1 + ratio_day2 + ratio_day3
  let total_charge := ratio_day1 * charge_day1 + ratio_day2 * charge_day2 + ratio_day3 * charge_day3
  average_charge = total_charge / total_ratio := by
  sorry

end NUMINAMATH_CALUDE_show_charge_day3_l1713_171337


namespace NUMINAMATH_CALUDE_add_decimal_numbers_l1713_171319

theorem add_decimal_numbers : 0.45 + 57.25 = 57.70 := by
  sorry

end NUMINAMATH_CALUDE_add_decimal_numbers_l1713_171319


namespace NUMINAMATH_CALUDE_x_axis_segment_range_l1713_171376

/-- Definition of a quadratic function -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ -a * x^2 + 2 * b * x - c

/-- Definition of the centrally symmetric function with respect to (0,0) -/
def CentrallySymmetricFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + 2 * b * x + c

/-- Theorem about the range of x-axis segment length for the centrally symmetric function -/
theorem x_axis_segment_range
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : a + b + c = 0)
  (h2 : (2*c + b - a) * (2*c + b + 3*a) < 0) :
  ∃ (x₁ x₂ : ℝ), CentrallySymmetricFunction a b c x₁ = 0 ∧
                 CentrallySymmetricFunction a b c x₂ = 0 ∧
                 Real.sqrt 3 < |x₁ - x₂| ∧
                 |x₁ - x₂| < 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_x_axis_segment_range_l1713_171376


namespace NUMINAMATH_CALUDE_a₁₂_eq_15_l1713_171346

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  a₄_eq_1 : a 4 = 1
  a₇_plus_a₉_eq_16 : a 7 + a 9 = 16

/-- The 12th term of the arithmetic sequence is 15 -/
theorem a₁₂_eq_15 (seq : ArithmeticSequence) : seq.a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_a₁₂_eq_15_l1713_171346


namespace NUMINAMATH_CALUDE_integer_points_on_line_l1713_171392

theorem integer_points_on_line (n : ℕ) (initial_sum final_sum shift : ℤ) 
  (h1 : initial_sum = 25)
  (h2 : final_sum = -35)
  (h3 : shift = 5)
  (h4 : final_sum = initial_sum - n * shift) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_integer_points_on_line_l1713_171392


namespace NUMINAMATH_CALUDE_divisor_problem_l1713_171310

theorem divisor_problem : ∃ d : ℕ, d > 1 ∧ (9671 - 1) % d = 0 ∧ 
  ∀ k : ℕ, k > 1 → (9671 - 1) % k = 0 → k ≤ d :=
sorry

end NUMINAMATH_CALUDE_divisor_problem_l1713_171310


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1713_171360

theorem complex_equation_solution (z : ℂ) : 
  z * (1 - 2*I) = 3 + 2*I → z = -1/5 + 8/5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1713_171360


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1713_171343

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1713_171343


namespace NUMINAMATH_CALUDE_range_intersection_l1713_171312

theorem range_intersection (x : ℝ) : 
  (x^2 - 7*x + 10 ≤ 0) ∧ ((x - 3)*(x + 1) ≤ 0) ↔ 2 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_intersection_l1713_171312


namespace NUMINAMATH_CALUDE_perfect_square_pairs_l1713_171349

theorem perfect_square_pairs (m n : ℤ) :
  (∃ a : ℤ, m^2 + n = a^2) ∧ (∃ b : ℤ, n^2 + m = b^2) →
  (m = 0 ∧ ∃ k : ℤ, n = k^2) ∨
  (n = 0 ∧ ∃ k : ℤ, m = k^2) ∨
  (m = 1 ∧ n = -1) ∨
  (m = -1 ∧ n = 1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_pairs_l1713_171349


namespace NUMINAMATH_CALUDE_encounter_twelve_trams_l1713_171387

/-- Represents the tram system with given parameters -/
structure TramSystem where
  departure_interval : ℕ  -- Interval between tram departures in minutes
  journey_duration : ℕ    -- Duration of a full journey in minutes

/-- Calculates the number of trams encountered during a journey -/
def count_encountered_trams (system : TramSystem) : ℕ :=
  2 * (system.journey_duration / system.departure_interval)

/-- Theorem stating that in the given tram system, a passenger will encounter 12 trams -/
theorem encounter_twelve_trams (system : TramSystem) 
  (h1 : system.departure_interval = 10)
  (h2 : system.journey_duration = 60) : 
  count_encountered_trams system = 12 := by
  sorry

#eval count_encountered_trams ⟨10, 60⟩

end NUMINAMATH_CALUDE_encounter_twelve_trams_l1713_171387


namespace NUMINAMATH_CALUDE_percentage_x_more_than_y_l1713_171323

theorem percentage_x_more_than_y : 
  ∀ (x y z : ℝ),
  y = 1.2 * z →
  z = 250 →
  x + y + z = 925 →
  (x - y) / y * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_x_more_than_y_l1713_171323


namespace NUMINAMATH_CALUDE_prob_two_queens_or_two_jacks_standard_deck_l1713_171388

/-- A standard deck of cards. -/
structure Deck :=
  (total_cards : ℕ)
  (queens : ℕ)
  (jacks : ℕ)

/-- The probability of drawing either two queens or at least two jacks
    when selecting 3 cards randomly from a standard deck. -/
def prob_two_queens_or_two_jacks (d : Deck) : ℚ :=
  -- Definition to be proved
  74 / 850

/-- Theorem stating the probability of drawing either two queens or at least two jacks
    when selecting 3 cards randomly from a standard 52-card deck. -/
theorem prob_two_queens_or_two_jacks_standard_deck :
  prob_two_queens_or_two_jacks ⟨52, 4, 4⟩ = 74 / 850 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_queens_or_two_jacks_standard_deck_l1713_171388


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1713_171315

-- Define the hyperbola
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  focus_on_y_axis : Bool
  semi_minor_axis : ℝ
  eccentricity : ℝ

-- Define the asymptote equation type
structure AsymptoticEquation where
  slope : ℝ

-- Theorem statement
theorem hyperbola_asymptotes 
  (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_focus : h.focus_on_y_axis = true)
  (h_semi_minor : h.semi_minor_axis = 4 * Real.sqrt 2)
  (h_eccentricity : h.eccentricity = 3) :
  ∃ (eq : AsymptoticEquation), eq.slope = Real.sqrt 2 / 4 ∨ eq.slope = -(Real.sqrt 2 / 4) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1713_171315


namespace NUMINAMATH_CALUDE_symmetric_point_l1713_171383

/-- Given a line l: x + y = 1 and two points P and Q, 
    this function checks if Q is symmetric to P with respect to l --/
def is_symmetric (P Q : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  (qy - py) / (qx - px) = -1 ∧ -- Perpendicular condition
  (px + qx) / 2 + (py + qy) / 2 = 1 -- Midpoint on the line condition

/-- Theorem stating that Q(-4, -1) is symmetric to P(2, 5) with respect to the line x + y = 1 --/
theorem symmetric_point : is_symmetric (2, 5) (-4, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_l1713_171383


namespace NUMINAMATH_CALUDE_concentric_circles_chord_count_l1713_171340

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle ABC is 80 degrees, then the number of segments needed to return to the starting point is 18. -/
theorem concentric_circles_chord_count (angle_ABC : ℝ) (n : ℕ) : 
  angle_ABC = 80 → n * 100 = 360 * (n / 18) → n = 18 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_chord_count_l1713_171340


namespace NUMINAMATH_CALUDE_max_value_M_l1713_171303

theorem max_value_M (x y z w : ℝ) (h : x + y + z + w = 1) :
  ∃ (max : ℝ), max = (3 : ℝ) / 2 ∧ 
  ∀ (a b c d : ℝ), a + b + c + d = 1 → 
  a * d + 2 * b * d + 3 * a * b + 3 * c * d + 4 * a * c + 5 * b * c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_M_l1713_171303


namespace NUMINAMATH_CALUDE_power_of_64_two_thirds_l1713_171379

theorem power_of_64_two_thirds : (64 : ℝ) ^ (2/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_two_thirds_l1713_171379


namespace NUMINAMATH_CALUDE_min_xy_value_l1713_171336

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4*x*y - x - 2*y = 4) :
  ∀ z, z = x*y → z ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l1713_171336


namespace NUMINAMATH_CALUDE_science_fiction_total_pages_l1713_171325

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each science fiction book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_total_pages : total_pages = 3824 := by
  sorry

end NUMINAMATH_CALUDE_science_fiction_total_pages_l1713_171325


namespace NUMINAMATH_CALUDE_kenny_book_purchase_l1713_171339

def lawn_price : ℕ := 15
def video_game_price : ℕ := 45
def book_price : ℕ := 5
def lawns_mowed : ℕ := 35
def video_games_wanted : ℕ := 5

def total_earned : ℕ := lawn_price * lawns_mowed
def video_games_cost : ℕ := video_game_price * video_games_wanted
def remaining_money : ℕ := total_earned - video_games_cost

theorem kenny_book_purchase :
  remaining_money / book_price = 60 := by sorry

end NUMINAMATH_CALUDE_kenny_book_purchase_l1713_171339


namespace NUMINAMATH_CALUDE_commodity_price_problem_l1713_171338

theorem commodity_price_problem (total_cost first_price second_price : ℕ) :
  total_cost = 827 →
  first_price = second_price + 127 →
  total_cost = first_price + second_price →
  first_price = 477 := by
  sorry

end NUMINAMATH_CALUDE_commodity_price_problem_l1713_171338


namespace NUMINAMATH_CALUDE_complex_division_equality_l1713_171374

theorem complex_division_equality : (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l1713_171374


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersections_nonagon_intersections_eq_choose_four_l1713_171381

/-- The number of intersection points of diagonals in a regular nonagon -/
theorem nonagon_diagonal_intersections : ℕ := by
  -- Define a regular nonagon
  sorry

/-- The number of ways to choose 4 vertices from 9 vertices -/
def choose_four_from_nine : ℕ := Nat.choose 9 4

/-- Theorem: The number of distinct interior points where two or more diagonals
    intersect in a regular nonagon is equal to choose_four_from_nine -/
theorem nonagon_intersections_eq_choose_four :
  nonagon_diagonal_intersections = choose_four_from_nine := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersections_nonagon_intersections_eq_choose_four_l1713_171381


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l1713_171309

-- Define the companion sequence
def is_companion_sequence (c : ℕ+ → ℝ) (d : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, c n > 0 ∧ d n > 0 ∧
  c (n + 1) = (c n + d n) / Real.sqrt ((c n)^2 + (d n)^2)

-- Part 1
theorem part_one (a b : ℕ+ → ℝ) :
  is_companion_sequence a b →
  (∀ n : ℕ+, b n = a n) →
  b 1 = Real.sqrt 2 →
  ∀ n : ℕ+, a n = Real.sqrt 2 :=
sorry

-- Part 2
theorem part_two (a b : ℕ+ → ℝ) :
  is_companion_sequence a b →
  (∀ n : ℕ+, b (n + 1) = 1 + b n / a n) →
  ∃ k : ℝ, b 1 / a 1 = k →
  ∃ d : ℝ, ∀ n : ℕ+, (b (n + 1) / a (n + 1))^2 = (b n / a n)^2 + d :=
sorry

-- Part 3
theorem part_three (a b : ℕ+ → ℝ) :
  is_companion_sequence a b →
  (∀ n : ℕ+, b (n + 1) = Real.sqrt 2 * b n / a n) →
  (∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n) →
  a 1 = Real.sqrt 2 ∧ b 1 = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l1713_171309


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l1713_171300

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l1713_171300


namespace NUMINAMATH_CALUDE_fraction_equality_l1713_171356

theorem fraction_equality : (2222 - 2121)^2 / 196 = 52 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1713_171356


namespace NUMINAMATH_CALUDE_circle_B_radius_l1713_171398

/-- The configuration of four circles A, B, C, and D with specific properties -/
structure CircleConfig where
  /-- Radius of circle A -/
  radius_A : ℝ
  /-- Radius of circle B -/
  radius_B : ℝ
  /-- Radius of circle C -/
  radius_C : ℝ
  /-- Radius of circle D -/
  radius_D : ℝ
  /-- Circles A, B, and C are externally tangent to each other -/
  externally_tangent : radius_A + radius_B + radius_C = radius_D
  /-- Circles B and C are congruent -/
  B_C_congruent : radius_B = radius_C
  /-- Circle A passes through the center of D -/
  A_through_D_center : radius_A = radius_D / 2
  /-- Circle A has a radius of 2 -/
  A_radius_2 : radius_A = 2

/-- The main theorem stating that given the circle configuration, the radius of circle B is approximately 0.923 -/
theorem circle_B_radius (config : CircleConfig) : 
  0.922 < config.radius_B ∧ config.radius_B < 0.924 := by
  sorry


end NUMINAMATH_CALUDE_circle_B_radius_l1713_171398


namespace NUMINAMATH_CALUDE_hotel_rooms_l1713_171354

theorem hotel_rooms (total_rooms : ℕ) (single_cost double_cost : ℕ) (total_revenue : ℕ) :
  total_rooms = 260 ∧
  single_cost = 35 ∧
  double_cost = 60 ∧
  total_revenue = 14000 →
  ∃ (single_rooms double_rooms : ℕ),
    single_rooms + double_rooms = total_rooms ∧
    single_cost * single_rooms + double_cost * double_rooms = total_revenue ∧
    single_rooms = 64 :=
by sorry

end NUMINAMATH_CALUDE_hotel_rooms_l1713_171354


namespace NUMINAMATH_CALUDE_handshakes_in_social_event_l1713_171334

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  totalPeople : Nat
  group1Size : Nat
  group2Size : Nat
  knownInGroup1 : Nat
  knownInGroup2 : Nat

/-- Calculates the number of handshakes in a social event -/
def calculateHandshakes (event : SocialEvent) : Nat :=
  let group1Handshakes := event.group1Size * (event.totalPeople - event.group1Size + event.knownInGroup1)
  let group2Handshakes := event.group2Size * (event.totalPeople - event.group2Size + event.knownInGroup2)
  (group1Handshakes + group2Handshakes) / 2

/-- Theorem stating that the number of handshakes in the given social event is 630 -/
theorem handshakes_in_social_event :
  let event : SocialEvent := {
    totalPeople := 40,
    group1Size := 25,
    group2Size := 15,
    knownInGroup1 := 18,
    knownInGroup2 := 4
  }
  calculateHandshakes event = 630 := by
  sorry


end NUMINAMATH_CALUDE_handshakes_in_social_event_l1713_171334


namespace NUMINAMATH_CALUDE_rational_cube_sum_representation_l1713_171352

theorem rational_cube_sum_representation (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_representation_l1713_171352


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1713_171372

-- Define set A
def A : Set ℝ := {x | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1713_171372


namespace NUMINAMATH_CALUDE_search_plans_count_l1713_171367

/-- Represents the number of children in the group -/
def total_children : ℕ := 8

/-- Represents whether Grace participates in the task -/
inductive GraceParticipation
| Participates
| DoesNotParticipate

/-- Calculates the number of ways to distribute children for the search task -/
def count_search_plans : ℕ :=
  let grace_participates := Nat.choose 7 3  -- Choose 3 out of 7 to go with Grace
  let grace_not_participates := 7 * Nat.choose 6 3  -- Choose 1 to stay, then distribute 6
  grace_participates + grace_not_participates

/-- Theorem stating that the number of different search plans is 175 -/
theorem search_plans_count :
  count_search_plans = 175 := by sorry

end NUMINAMATH_CALUDE_search_plans_count_l1713_171367


namespace NUMINAMATH_CALUDE_prime_between_squares_l1713_171311

theorem prime_between_squares : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  ∃ n : ℕ, n^2 = p - 9 ∧ (n+1)^2 = p + 8 := by
  sorry

end NUMINAMATH_CALUDE_prime_between_squares_l1713_171311


namespace NUMINAMATH_CALUDE_range_of_half_difference_l1713_171357

theorem range_of_half_difference (α β : ℝ) 
  (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) :
  ∀ x, x ∈ Set.Icc (-π/2) 0 ↔ ∃ α β, -π/2 ≤ α ∧ α < β ∧ β ≤ π/2 ∧ x = (α - β)/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_half_difference_l1713_171357


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l1713_171327

/-- The last digit of the decimal expansion of 1/3^15 is 0 -/
theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) : 
  n = 15 → (∃ (k : ℕ), (1 : ℚ) / 3^n = k * (1 / 10^n) + (1 / 10^n)) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l1713_171327


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_b_values_l1713_171378

noncomputable section

def f (a b x : ℝ) : ℝ := (a * Real.log x) / (x + 1) + b / x

def tangent_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

theorem tangent_line_implies_a_b_values (a b : ℝ) :
  (∀ x, tangent_line x (f a b x)) →
  (tangent_line 1 (f a b 1)) →
  (a = 1 ∧ b = 1) := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_implies_a_b_values_l1713_171378


namespace NUMINAMATH_CALUDE_inverse_relation_values_l1713_171324

/-- Represents the constant product of two inversely related quantities -/
def k : ℝ := 800 * 0.5

/-- Represents the relationship between inversely related quantities a and b -/
def inverse_relation (a b : ℝ) : Prop := a * b = k

theorem inverse_relation_values (a₁ a₂ : ℝ) (h₁ : inverse_relation 800 0.5) :
  (inverse_relation 1600 0.250) ∧ (inverse_relation 400 1.000) := by
  sorry

#check inverse_relation_values

end NUMINAMATH_CALUDE_inverse_relation_values_l1713_171324


namespace NUMINAMATH_CALUDE_solve_2a_plus_b_l1713_171328

theorem solve_2a_plus_b (a b : ℝ) 
  (h1 : 4 * a^2 - b^2 = 12) 
  (h2 : 2 * a - b = 4) : 
  2 * a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_solve_2a_plus_b_l1713_171328


namespace NUMINAMATH_CALUDE_perpendicular_chords_intersection_distance_l1713_171386

theorem perpendicular_chords_intersection_distance (d r : ℝ) (AB CD : ℝ) (h1 : d = 10) (h2 : r = d / 2) (h3 : AB = 9) (h4 : CD = 8) :
  let S := r^2 - (AB/2)^2
  let R := r^2 - (CD/2)^2
  (S + R).sqrt = (55 : ℝ).sqrt / 2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_chords_intersection_distance_l1713_171386


namespace NUMINAMATH_CALUDE_man_speed_man_speed_approx_6kmh_l1713_171365

/-- Calculates the speed of a man given the parameters of a train passing him --/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / passing_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- The speed of the man is approximately 6 km/h --/
theorem man_speed_approx_6kmh :
  ∃ ε > 0, |man_speed 160 90 6 - 6| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_approx_6kmh_l1713_171365


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l1713_171307

theorem quadratic_equation_value (y : ℝ) (h : y = 4) : 3 * y^2 + 4 * y + 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l1713_171307


namespace NUMINAMATH_CALUDE_selection_methods_count_l1713_171335

def total_students : ℕ := 9
def selected_students : ℕ := 4
def specific_students : ℕ := 3

def selection_methods : ℕ := 
  Nat.choose specific_students 2 * Nat.choose (total_students - specific_students) 2 +
  Nat.choose specific_students 3 * Nat.choose (total_students - specific_students) 1

theorem selection_methods_count : selection_methods = 51 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l1713_171335


namespace NUMINAMATH_CALUDE_water_per_day_per_man_l1713_171322

/-- Calculates the amount of water needed per day per man on a sea voyage --/
theorem water_per_day_per_man 
  (total_men : ℕ) 
  (miles_per_day : ℕ) 
  (total_miles : ℕ) 
  (total_water : ℕ) : 
  total_men = 25 → 
  miles_per_day = 200 → 
  total_miles = 4000 → 
  total_water = 250 → 
  (total_water : ℚ) / ((total_miles : ℚ) / (miles_per_day : ℚ)) / (total_men : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_water_per_day_per_man_l1713_171322


namespace NUMINAMATH_CALUDE_competition_result_competition_result_proof_l1713_171395

-- Define the type for students
inductive Student : Type
  | A | B | C | D | E

-- Define a type for the competition order
def CompetitionOrder := List Student

-- Define the first person's prediction
def firstPrediction : CompetitionOrder :=
  [Student.A, Student.B, Student.C, Student.D, Student.E]

-- Define the second person's prediction
def secondPrediction : CompetitionOrder :=
  [Student.D, Student.A, Student.E, Student.C, Student.B]

-- Function to check if a student is in the correct position
def correctPosition (actual : CompetitionOrder) (predicted : CompetitionOrder) (index : Nat) : Prop :=
  actual.get? index = predicted.get? index

-- Function to check if adjacent pairs are correct
def correctAdjacentPair (actual : CompetitionOrder) (predicted : CompetitionOrder) (index : Nat) : Prop :=
  actual.get? index = predicted.get? index ∧ actual.get? (index + 1) = predicted.get? (index + 1)

-- Main theorem
theorem competition_result (actual : CompetitionOrder) : Prop :=
  (actual.length = 5) ∧
  (∀ i, i < 5 → ¬correctPosition actual firstPrediction i) ∧
  (∀ i, i < 4 → ¬correctAdjacentPair actual firstPrediction i) ∧
  ((correctPosition actual secondPrediction 0 ∧ correctPosition actual secondPrediction 1) ∨
   (correctPosition actual secondPrediction 1 ∧ correctPosition actual secondPrediction 2) ∨
   (correctPosition actual secondPrediction 2 ∧ correctPosition actual secondPrediction 3) ∨
   (correctPosition actual secondPrediction 3 ∧ correctPosition actual secondPrediction 4)) ∧
  ((correctAdjacentPair actual secondPrediction 0 ∧ correctAdjacentPair actual secondPrediction 2) ∨
   (correctAdjacentPair actual secondPrediction 0 ∧ correctAdjacentPair actual secondPrediction 3) ∨
   (correctAdjacentPair actual secondPrediction 1 ∧ correctAdjacentPair actual secondPrediction 3)) ∧
  (actual = [Student.E, Student.D, Student.A, Student.C, Student.B])

-- Proof of the theorem
theorem competition_result_proof : ∃ actual, competition_result actual := by
  sorry


end NUMINAMATH_CALUDE_competition_result_competition_result_proof_l1713_171395


namespace NUMINAMATH_CALUDE_triangle_properties_l1713_171308

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  2 * c * Real.cos B = 2 * a + b →
  S = Real.sqrt 3 * (a + b) →
  S = 1 / 2 * a * b * Real.sin C →
  (C = 2 * π / 3 ∧ a * b ≥ 64) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1713_171308


namespace NUMINAMATH_CALUDE_triangle_angle_and_max_area_l1713_171389

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangleCondition (t : Triangle) : Prop :=
  cos t.B / cos t.C = -t.b / (2 * t.a + t.c)

theorem triangle_angle_and_max_area (t : Triangle) 
  (h : triangleCondition t) : 
  t.B = 2 * π / 3 ∧ 
  (t.b = 3 → ∃ (maxArea : ℝ), maxArea = 3 * sqrt 3 / 4 ∧ 
    ∀ (area : ℝ), area ≤ maxArea) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_and_max_area_l1713_171389


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1713_171350

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2*x ∧ x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1713_171350


namespace NUMINAMATH_CALUDE_soccer_balls_count_l1713_171359

/-- The number of soccer balls in the gym. -/
def soccer_balls : ℕ := 20

/-- The number of baseballs in the gym. -/
def baseballs : ℕ := 5 * soccer_balls

/-- The number of volleyballs in the gym. -/
def volleyballs : ℕ := 3 * soccer_balls

/-- Theorem stating that the number of soccer balls is 20, given the conditions of the problem. -/
theorem soccer_balls_count :
  soccer_balls = 20 ∧
  baseballs = 5 * soccer_balls ∧
  volleyballs = 3 * soccer_balls ∧
  baseballs + volleyballs = 160 :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_count_l1713_171359


namespace NUMINAMATH_CALUDE_percentage_of_women_in_survey_l1713_171304

theorem percentage_of_women_in_survey (w : ℝ) (m : ℝ) : 
  w + m = 100 →
  (3/4 : ℝ) * w + (9/10 : ℝ) * m = 84 →
  w = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_survey_l1713_171304


namespace NUMINAMATH_CALUDE_number_exceeds_value_l1713_171377

theorem number_exceeds_value (n : ℕ) (v : ℕ) (h : n = 69) : 
  n = v + 3 * (86 - n) → v = 18 := by
sorry

end NUMINAMATH_CALUDE_number_exceeds_value_l1713_171377


namespace NUMINAMATH_CALUDE_alex_grocery_delivery_l1713_171316

/-- Alex's grocery delivery problem -/
theorem alex_grocery_delivery 
  (savings : ℝ) 
  (car_cost : ℝ) 
  (trip_charge : ℝ) 
  (grocery_percentage : ℝ) 
  (num_trips : ℕ) 
  (h1 : savings = 14500)
  (h2 : car_cost = 14600)
  (h3 : trip_charge = 1.5)
  (h4 : grocery_percentage = 0.05)
  (h5 : num_trips = 40)
  : ∃ (grocery_worth : ℝ), 
    trip_charge * num_trips + grocery_percentage * grocery_worth = car_cost - savings ∧ 
    grocery_worth = 800 := by
  sorry

end NUMINAMATH_CALUDE_alex_grocery_delivery_l1713_171316


namespace NUMINAMATH_CALUDE_cubic_factorization_l1713_171347

theorem cubic_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1713_171347


namespace NUMINAMATH_CALUDE_store_earnings_l1713_171362

/-- Calculates the total earnings from selling shirts and jeans --/
def total_earnings (shirt_price : ℕ) (shirt_quantity : ℕ) (jeans_quantity : ℕ) : ℕ :=
  let jeans_price := 2 * shirt_price
  shirt_price * shirt_quantity + jeans_price * jeans_quantity

/-- Proves that the total earnings from selling 20 shirts at $10 each and 10 pairs of jeans at twice the price of a shirt is $400 --/
theorem store_earnings : total_earnings 10 20 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_store_earnings_l1713_171362


namespace NUMINAMATH_CALUDE_five_line_triangle_bounds_l1713_171348

/-- A line in a plane --/
structure Line where
  -- Add necessary fields here
  
/-- A region in a plane --/
structure Region where
  -- Add necessary fields here

/-- Represents a configuration of lines in a plane --/
structure PlaneConfiguration where
  lines : List Line
  regions : List Region

/-- Checks if lines are in general position --/
def is_general_position (config : PlaneConfiguration) : Prop :=
  sorry

/-- Counts the number of triangular regions --/
def count_triangles (config : PlaneConfiguration) : Nat :=
  sorry

/-- Main theorem about triangles in a plane divided by five lines --/
theorem five_line_triangle_bounds 
  (config : PlaneConfiguration) 
  (h1 : config.lines.length = 5)
  (h2 : config.regions.length = 16)
  (h3 : is_general_position config) :
  3 ≤ count_triangles config ∧ count_triangles config ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_five_line_triangle_bounds_l1713_171348
