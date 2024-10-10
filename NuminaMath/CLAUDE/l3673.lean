import Mathlib

namespace material_left_proof_l3673_367350

theorem material_left_proof (material1 material2 material_used : ℚ) :
  material1 = 4 / 17 →
  material2 = 3 / 10 →
  material_used = 0.23529411764705882 →
  material1 + material2 - material_used = 51 / 170 := by
  sorry

end material_left_proof_l3673_367350


namespace extreme_value_at_one_l3673_367302

-- Define the function f(x) = x³ - ax
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem extreme_value_at_one (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 3 := by
  sorry


end extreme_value_at_one_l3673_367302


namespace triangle_inequality_ac_not_fourteen_l3673_367325

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  c < a + b ∧ b < a + c ∧ a < b + c :=
sorry

theorem ac_not_fourteen (ab bc : ℝ) (hab : ab = 5) (hbc : bc = 8) :
  ¬ (∃ (ac : ℝ), ac = 14 ∧ 
    (ac < ab + bc ∧ bc < ab + ac ∧ ab < bc + ac) ∧
    (0 < ab ∧ 0 < bc ∧ 0 < ac)) :=
sorry

end triangle_inequality_ac_not_fourteen_l3673_367325


namespace trivia_game_win_probability_l3673_367367

def num_questions : ℕ := 4
def num_choices : ℕ := 4
def min_correct : ℕ := 3

def probability_correct_guess : ℚ := 1 / num_choices

def probability_win : ℚ :=
  (probability_correct_guess ^ num_questions) +
  (num_questions * (probability_correct_guess ^ (num_questions - 1)) * (1 - probability_correct_guess))

theorem trivia_game_win_probability :
  probability_win = 13 / 256 := by
  sorry

end trivia_game_win_probability_l3673_367367


namespace quadratic_inequality_l3673_367331

theorem quadratic_inequality (x : ℝ) : x^2 + 3*x - 18 > 0 ↔ x < -6 ∨ x > 3 := by
  sorry

end quadratic_inequality_l3673_367331


namespace intersection_A_B_l3673_367388

-- Define set A
def A : Set ℝ := {x | ∃ y, (x^2)/4 + (3*y^2)/4 = 1}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = x^2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 2} := by sorry

end intersection_A_B_l3673_367388


namespace function_value_at_2017_l3673_367307

/-- Given a function f(x) = x^2 - x * f'(0) - 1, prove that f(2017) = 2016 * 2018 -/
theorem function_value_at_2017 (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - x * (deriv f 0) - 1) : 
  f 2017 = 2016 * 2018 := by
sorry

end function_value_at_2017_l3673_367307


namespace tomatoes_eaten_by_birds_l3673_367319

theorem tomatoes_eaten_by_birds 
  (total_grown : ℕ) 
  (remaining : ℕ) 
  (h1 : total_grown = 127) 
  (h2 : remaining = 54) 
  (h3 : remaining * 2 = total_grown - (total_grown - remaining * 2)) : 
  total_grown - remaining * 2 = 19 := by
sorry

end tomatoes_eaten_by_birds_l3673_367319


namespace students_just_passed_l3673_367371

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 28 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_all_passed : first_div_percent + second_div_percent ≤ 1) :
  total - (total * (first_div_percent + second_div_percent)).floor = 54 := by
  sorry

end students_just_passed_l3673_367371


namespace trajectory_equation_l3673_367313

-- Define the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the distance difference condition
def distance_difference : ℝ := 4

-- Define the trajectory of point C
def trajectory_of_C (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1 ∧ x ≥ 2

-- State the theorem
theorem trajectory_equation :
  ∀ (C : ℝ × ℝ),
    (Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) -
     Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = distance_difference) →
    trajectory_of_C C.1 C.2 :=
by sorry

end trajectory_equation_l3673_367313


namespace train_travel_rate_l3673_367309

/-- Given a train's travel information, prove the rate of additional hours per mile -/
theorem train_travel_rate (initial_distance : ℝ) (initial_time : ℝ) 
  (additional_distance : ℝ) (additional_time : ℝ) 
  (h1 : initial_distance = 360) 
  (h2 : initial_time = 3) 
  (h3 : additional_distance = 240) 
  (h4 : additional_time = 2) :
  (additional_time / additional_distance) = 1 / 120 := by
  sorry

end train_travel_rate_l3673_367309


namespace one_prime_in_alternating_series_l3673_367387

/-- The nth number in the alternating 1-0 series -/
def A (n : ℕ) : ℕ := 
  (10^(2*n) - 1) / 99

/-- The series of alternating 1-0 numbers -/
def alternating_series : Set ℕ :=
  {x | ∃ n : ℕ, x = A n}

/-- Theorem: There is exactly one prime number in the alternating 1-0 series -/
theorem one_prime_in_alternating_series : 
  ∃! p, p ∈ alternating_series ∧ Nat.Prime p :=
sorry

end one_prime_in_alternating_series_l3673_367387


namespace cubic_function_range_l3673_367372

/-- Given a cubic function f(x) = ax³ + bx satisfying certain conditions,
    prove that its range is [-2, 18] -/
theorem cubic_function_range (a b : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a * x^3 + b * x)
    (h_point : f 2 = 2) (h_slope : (fun x ↦ 3 * a * x^2 + b) 2 = 9) :
    Set.range f = Set.Icc (-2) 18 := by
  sorry

end cubic_function_range_l3673_367372


namespace inches_to_represent_distance_l3673_367332

/-- Represents the scale of a map in miles per inch -/
def map_scale : ℝ := 28

/-- Represents the relationship between inches and miles on the map -/
theorem inches_to_represent_distance (D : ℝ) :
  ∃ I : ℝ, I * map_scale = D ∧ I = D / map_scale :=
sorry

end inches_to_represent_distance_l3673_367332


namespace sum_of_number_and_its_square_l3673_367335

theorem sum_of_number_and_its_square (x : ℝ) : x = 18 → x + x^2 = 342 := by
  sorry

end sum_of_number_and_its_square_l3673_367335


namespace max_sum_xy_l3673_367359

theorem max_sum_xy (x y a b : ℝ) (hx : x > 0) (hy : y > 0)
  (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a*x + b*y = 1) :
  x + y ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 2 ∧
    ∃ (a₀ b₀ : ℝ), 0 ≤ a₀ ∧ a₀ ≤ x₀ ∧ 0 ≤ b₀ ∧ b₀ ≤ y₀ ∧
      a₀^2 + y₀^2 = 2 ∧ b₀^2 + x₀^2 = 1 ∧ a₀*x₀ + b₀*y₀ = 1 :=
by sorry

end max_sum_xy_l3673_367359


namespace triangle_longer_segment_l3673_367397

theorem triangle_longer_segment (a b c h x : ℝ) : 
  a = 40 → b = 90 → c = 100 → 
  x^2 + h^2 = a^2 → 
  (c - x)^2 + h^2 = b^2 → 
  c - x = 82.5 :=
by sorry

end triangle_longer_segment_l3673_367397


namespace ninth_square_difference_l3673_367328

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := (2 * n) ^ 2

/-- The difference in tiles between the n-th and (n-1)-th squares -/
def tile_difference (n : ℕ) : ℕ := tiles_in_square n - tiles_in_square (n - 1)

theorem ninth_square_difference : tile_difference 9 = 68 := by
  sorry

end ninth_square_difference_l3673_367328


namespace marked_nodes_on_circle_l3673_367375

/-- Represents a node in the hexagon grid -/
structure Node where
  x : ℤ
  y : ℤ

/-- Represents a circle in the hexagon grid -/
structure Circle where
  center : Node
  radius : ℕ

/-- The side length of the regular hexagon -/
def hexagon_side_length : ℕ := 5

/-- The side length of the equilateral triangles -/
def triangle_side_length : ℕ := 1

/-- The total number of nodes in the hexagon -/
def total_nodes : ℕ := 91

/-- A function that determines if a node is marked -/
def is_marked : Node → Prop := sorry

/-- A function that determines if a node lies on a given circle -/
def on_circle : Node → Circle → Prop := sorry

/-- The main theorem to be proved -/
theorem marked_nodes_on_circle :
  (∃ (marked_nodes : Finset Node), 
    (∀ n ∈ marked_nodes, is_marked n) ∧ 
    (marked_nodes.card > total_nodes / 2)) →
  (∃ (c : Circle) (five_nodes : Finset Node),
    five_nodes.card = 5 ∧
    (∀ n ∈ five_nodes, is_marked n ∧ on_circle n c)) :=
by sorry

end marked_nodes_on_circle_l3673_367375


namespace final_sign_is_minus_l3673_367320

/-- Represents the state of the board with plus and minus signs -/
structure BoardState where
  plus_count : Nat
  minus_count : Nat

/-- Represents an operation on the board -/
inductive Operation
  | same_sign
  | different_sign

/-- Applies an operation to the board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.same_sign => 
      if state.plus_count ≥ 2 then 
        { plus_count := state.plus_count - 1, minus_count := state.minus_count }
      else 
        { plus_count := state.plus_count + 1, minus_count := state.minus_count - 2 }
  | Operation.different_sign => 
      { plus_count := state.plus_count - 1, minus_count := state.minus_count }

/-- Theorem: After 24 operations, the final sign is a minus sign -/
theorem final_sign_is_minus (initial_state : BoardState) 
    (h_initial : initial_state.plus_count = 10 ∧ initial_state.minus_count = 15) 
    (operations : List Operation) 
    (h_operations : operations.length = 24) : 
    (operations.foldl apply_operation initial_state).plus_count = 0 ∧ 
    (operations.foldl apply_operation initial_state).minus_count = 1 := by
  sorry

end final_sign_is_minus_l3673_367320


namespace equation_solution_l3673_367311

theorem equation_solution : ∃ x : ℚ, (x - 2)^2 - (x + 3)*(x - 3) = 4*x - 1 ∧ x = 7/4 := by
  sorry

end equation_solution_l3673_367311


namespace rectangular_box_width_l3673_367316

/-- Proves that the width of rectangular boxes is 5 cm given the conditions of the problem -/
theorem rectangular_box_width (wooden_length wooden_width wooden_height : ℕ)
                               (box_length box_height : ℕ)
                               (max_boxes : ℕ) :
  wooden_length = 800 →
  wooden_width = 1000 →
  wooden_height = 600 →
  box_length = 4 →
  box_height = 6 →
  max_boxes = 4000000 →
  ∃ (box_width : ℕ),
    box_width = 5 ∧
    wooden_length * wooden_width * wooden_height =
    max_boxes * (box_length * box_width * box_height) :=
by sorry

end rectangular_box_width_l3673_367316


namespace linear_coefficient_of_quadratic_l3673_367347

theorem linear_coefficient_of_quadratic (m : ℝ) : 
  m^2 - 2*m - 1 = 2 → 
  m - 3 ≠ 0 → 
  ∃ a b c, (m - 3)*x + 4*m^2 - 2*m - 1 - m*x + 6 = a*x^2 + b*x + c ∧ b = 1 :=
by sorry

end linear_coefficient_of_quadratic_l3673_367347


namespace rounding_effect_on_expression_l3673_367386

theorem rounding_effect_on_expression (a b c a' b' c' : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha' : a' ≥ a) (hb' : b' ≤ b) (hc' : c' ≤ c) : 
  2 * (a' / b') + 2 * c' > 2 * (a / b) + 2 * c :=
sorry

end rounding_effect_on_expression_l3673_367386


namespace num_intersection_points_is_correct_l3673_367365

/-- The number of distinct intersection points of two equations -/
def num_intersection_points : ℕ := 3

/-- First equation -/
def equation1 (x y : ℝ) : Prop :=
  (x - y + 3) * (2*x + 3*y - 9) = 0

/-- Second equation -/
def equation2 (x y : ℝ) : Prop :=
  (2*x - y + 2) * (x + 3*y - 6) = 0

/-- A point satisfies both equations -/
def is_intersection_point (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

theorem num_intersection_points_is_correct :
  ∃ (points : Finset (ℝ × ℝ)),
    points.card = num_intersection_points ∧
    (∀ p ∈ points, is_intersection_point p) ∧
    (∀ p : ℝ × ℝ, is_intersection_point p → p ∈ points) :=
  sorry

end num_intersection_points_is_correct_l3673_367365


namespace minimum_students_l3673_367336

theorem minimum_students (boys girls : ℕ) : 
  boys > 0 → 
  girls > 0 → 
  (3 * boys) / 4 = (2 * girls) / 3 → 
  ∃ (total : ℕ), total = boys + girls ∧ total ≥ 17 ∧ 
    ∀ (b g : ℕ), b > 0 → g > 0 → (3 * b) / 4 = (2 * g) / 3 → b + g ≥ total :=
by
  sorry

end minimum_students_l3673_367336


namespace angie_leftover_money_l3673_367342

def angie_finances (salary : ℕ) (necessities : ℕ) (taxes : ℕ) : ℕ :=
  salary - (necessities + taxes)

theorem angie_leftover_money :
  angie_finances 80 42 20 = 18 := by sorry

end angie_leftover_money_l3673_367342


namespace isosceles_triangle_perimeter_l3673_367389

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 4 → b = 4 → c = 2 →
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b > c ∧ b + c > a ∧ a + c > b →  -- triangle inequality
  a + b + c = 10 := by
  sorry

end isosceles_triangle_perimeter_l3673_367389


namespace fuel_distance_theorem_l3673_367391

/-- Represents the relationship between remaining fuel and distance traveled for a car -/
def fuel_distance_relation (initial_fuel : ℝ) (consumption_rate : ℝ) (x : ℝ) : ℝ :=
  initial_fuel - consumption_rate * x

/-- Theorem stating the relationship between remaining fuel and distance traveled -/
theorem fuel_distance_theorem (x : ℝ) :
  fuel_distance_relation 60 0.12 x = 60 - 0.12 * x := by
  sorry

end fuel_distance_theorem_l3673_367391


namespace geometric_sequence_property_l3673_367370

/-- A sequence is geometric if there exists a constant r such that a_{n+1} = r * a_n for all n -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The property that a_n^2 = a_{n-1} * a_{n+1} for all n -/
def HasSquareProperty (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_property :
  (∀ a : ℕ → ℝ, IsGeometricSequence a → HasSquareProperty a) ∧
  (∃ a : ℕ → ℝ, HasSquareProperty a ∧ ¬IsGeometricSequence a) :=
sorry

end geometric_sequence_property_l3673_367370


namespace chocolate_gain_percent_l3673_367317

theorem chocolate_gain_percent (C S : ℝ) (h : 24 * C = 16 * S) : 
  (S - C) / C * 100 = 50 := by
sorry

end chocolate_gain_percent_l3673_367317


namespace parallelogram_area_l3673_367382

def u : Fin 3 → ℝ := ![4, 2, -3]
def v : Fin 3 → ℝ := ![2, -4, 5]

theorem parallelogram_area : 
  Real.sqrt ((u 0 * v 1 - u 1 * v 0)^2 + (u 0 * v 2 - u 2 * v 0)^2 + (u 1 * v 2 - u 2 * v 1)^2) = 20 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l3673_367382


namespace function_derivative_at_one_l3673_367394

theorem function_derivative_at_one 
  (f : ℝ → ℝ) 
  (h_diff : DifferentiableOn ℝ f (Set.Ioi 0))
  (h_def : ∀ x : ℝ, f (Real.exp x) = x + Real.exp x) : 
  deriv f 1 = 2 := by sorry

end function_derivative_at_one_l3673_367394


namespace alex_age_problem_l3673_367303

/-- Alex's age problem -/
theorem alex_age_problem (A M : ℝ) : 
  (A - M = 3 * (A - 4 * M)) → A / M = 11 / 2 := by
  sorry

end alex_age_problem_l3673_367303


namespace unique_prime_triple_l3673_367366

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 0 → d ∣ n → d = 1 ∨ d = n

theorem unique_prime_triple :
  ∃! (p q r : ℕ), isPrime p ∧ isPrime q ∧ isPrime r ∧ p = q + 2 ∧ q = r + 2 :=
by sorry

end unique_prime_triple_l3673_367366


namespace value_added_after_doubling_l3673_367339

theorem value_added_after_doubling (x : ℝ) (v : ℝ) : 
  x = 4 → 2 * x + v = x / 2 + 20 → v = 14 := by
  sorry

end value_added_after_doubling_l3673_367339


namespace area_equals_perimeter_count_l3673_367381

/-- A structure representing a rectangle with integer sides -/
structure Rectangle where
  a : ℕ
  b : ℕ

/-- A structure representing a right triangle with integer sides -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The area of a rectangle is equal to its perimeter -/
def Rectangle.areaEqualsPerimeter (r : Rectangle) : Prop :=
  r.a * r.b = 2 * (r.a + r.b)

/-- The area of a right triangle is equal to its perimeter -/
def RightTriangle.areaEqualsPerimeter (t : RightTriangle) : Prop :=
  t.a * t.b = 2 * (t.a + t.b + t.c)

/-- The sides of a right triangle satisfy the Pythagorean theorem -/
def RightTriangle.isPythagorean (t : RightTriangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

/-- The main theorem stating the number of rectangles and right triangles that satisfy the conditions -/
theorem area_equals_perimeter_count :
  (∃! (rs : Finset Rectangle), ∀ r ∈ rs, r.areaEqualsPerimeter ∧ rs.card = 2) ∧
  (∃! (ts : Finset RightTriangle), ∀ t ∈ ts, t.areaEqualsPerimeter ∧ t.isPythagorean ∧ ts.card = 1) := by
  sorry


end area_equals_perimeter_count_l3673_367381


namespace arithmetic_sequence_a1_l3673_367352

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a1 (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 9)
  (h_a3_a2 : 2 * a 3 - a 2 = 6) :
  a 1 = -3 := by
sorry

end arithmetic_sequence_a1_l3673_367352


namespace equation_solution_l3673_367361

theorem equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 3 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 3 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 18 :=
by sorry

end equation_solution_l3673_367361


namespace problem_solution_l3673_367323

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 10

theorem problem_solution (m : ℝ) (h_m : m > 1) :
  (∀ x, f m x = x^2 - 2*m*x + 10) →
  (f m m = 1 → ∀ x, f m x = x^2 - 6*x + 10) ∧
  (((∀ x ≤ 2, ∀ y ≤ 2, x < y → f m x > f m y) ∧
    (∀ x ∈ Set.Icc 1 (m + 1), ∀ y ∈ Set.Icc 1 (m + 1), |f m x - f m y| ≤ 9)) →
   m ∈ Set.Icc 2 4) ∧
  ((∃ x ∈ Set.Icc 3 5, f m x = 0) →
   m ∈ Set.Icc (Real.sqrt 10) (7/2)) :=
by sorry

end problem_solution_l3673_367323


namespace fettuccine_tortellini_ratio_l3673_367312

/-- The ratio of students preferring fettuccine to those preferring tortellini -/
theorem fettuccine_tortellini_ratio 
  (total_students : ℕ) 
  (fettuccine_preference : ℕ) 
  (tortellini_preference : ℕ) 
  (h1 : total_students = 800)
  (h2 : fettuccine_preference = 200)
  (h3 : tortellini_preference = 160) : 
  (fettuccine_preference : ℚ) / tortellini_preference = 5 / 4 :=
by
  sorry

end fettuccine_tortellini_ratio_l3673_367312


namespace yaras_ship_speed_l3673_367322

/-- Prove that Yara's ship speed is 30 nautical miles per hour -/
theorem yaras_ship_speed (theons_speed : ℝ) (distance : ℝ) (time_difference : ℝ) :
  theons_speed = 15 →
  distance = 90 →
  time_difference = 3 →
  distance / (distance / theons_speed - time_difference) = 30 :=
by sorry

end yaras_ship_speed_l3673_367322


namespace hyperbola_m_value_l3673_367301

-- Define the hyperbola equation
def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop := x^2 - m*y^2 = 1

-- Define the condition for axis lengths
def axis_length_condition (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 = 1 ∧ b^2 = 1/m ∧ 2*a = 2*(2*b)

-- Theorem statement
theorem hyperbola_m_value (m : ℝ) :
  (∀ x y : ℝ, hyperbola_equation m x y) →
  axis_length_condition m →
  m = 4 := by
sorry

end hyperbola_m_value_l3673_367301


namespace geometric_sequence_common_ratio_l3673_367346

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_1 + a_3 = 10 and a_4 + a_6 = 5/4, then q = 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  q = 1/2 := by
sorry

end geometric_sequence_common_ratio_l3673_367346


namespace absolute_value_four_l3673_367344

theorem absolute_value_four (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by
  sorry

end absolute_value_four_l3673_367344


namespace f_of_3x_plus_2_l3673_367329

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_of_3x_plus_2 (x : ℝ) : f (3 * x + 2) = 9 * x^2 + 12 * x + 5 := by
  sorry

end f_of_3x_plus_2_l3673_367329


namespace work_completion_proof_l3673_367369

/-- The number of days A takes to complete the work alone -/
def a_days : ℕ := 45

/-- The number of days B takes to complete the work alone -/
def b_days : ℕ := 40

/-- The number of days B takes to complete the remaining work after A leaves -/
def b_remaining_days : ℕ := 23

/-- The number of days A works before leaving -/
def x : ℕ := 9

theorem work_completion_proof :
  let total_work := 1
  let a_rate := total_work / a_days
  let b_rate := total_work / b_days
  x * (a_rate + b_rate) + b_remaining_days * b_rate = total_work :=
by sorry

end work_completion_proof_l3673_367369


namespace toy_cars_in_second_box_l3673_367378

theorem toy_cars_in_second_box :
  let total_boxes : ℕ := 3
  let cars_in_first_box : ℕ := 21
  let cars_in_third_box : ℕ := 19
  let total_cars : ℕ := 71
  let cars_in_second_box : ℕ := total_cars - cars_in_first_box - cars_in_third_box
  cars_in_second_box = 31 := by
  sorry

end toy_cars_in_second_box_l3673_367378


namespace geometric_sequence_common_ratio_l3673_367318

/-- A geometric sequence with a_2 = 2 and a_5 = 1/4 has a common ratio of 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1))
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1/4) :
  a 1 = 1/2 := by
sorry

end geometric_sequence_common_ratio_l3673_367318


namespace four_integer_average_l3673_367355

theorem four_integer_average (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧ 
  d = 90 ∧ 
  a ≥ 37 → 
  (a + b + c + d) / 4 ≥ 51 := by
sorry

end four_integer_average_l3673_367355


namespace frog_jump_distance_l3673_367330

/-- The distance jumped by the grasshopper in inches -/
def grasshopper_jump : ℕ := 19

/-- The difference between the grasshopper's jump and the frog's jump in inches -/
def grasshopper_frog_diff : ℕ := 4

/-- The difference between the frog's jump and the mouse's jump in inches -/
def frog_mouse_diff : ℕ := 44

/-- The distance jumped by the frog in inches -/
def frog_jump : ℕ := grasshopper_jump - grasshopper_frog_diff

theorem frog_jump_distance : frog_jump = 15 := by sorry

end frog_jump_distance_l3673_367330


namespace white_spotted_mushrooms_count_l3673_367384

/-- The number of white-spotted mushrooms gathered by Bill and Ted -/
def white_spotted_mushrooms : ℕ :=
  let bill_red := 12
  let bill_brown := 6
  let ted_blue := 6
  let red_with_spots := (2 * bill_red) / 3
  let brown_with_spots := bill_brown
  let blue_with_spots := ted_blue / 2
  red_with_spots + brown_with_spots + blue_with_spots

/-- Theorem stating that the total number of white-spotted mushrooms is 17 -/
theorem white_spotted_mushrooms_count : white_spotted_mushrooms = 17 := by
  sorry

end white_spotted_mushrooms_count_l3673_367384


namespace recycling_problem_l3673_367353

/-- Given a total number of cans and a number of bags, calculates the number of cans per bag -/
def cans_per_bag (total_cans : ℕ) (num_bags : ℕ) : ℕ :=
  total_cans / num_bags

theorem recycling_problem (total_cans : ℕ) (num_bags : ℕ) 
  (h1 : total_cans = 122) (h2 : num_bags = 2) : 
  cans_per_bag total_cans num_bags = 61 := by
  sorry

end recycling_problem_l3673_367353


namespace tangent_circle_equation_l3673_367315

/-- A circle tangent to the coordinate axes and passing through (2, 1) -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = center.2
  passes_through : (2 - center.1)^2 + (1 - center.2)^2 = radius^2

/-- The equation of the circle -/
def circle_equation (c : TangentCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Theorem stating the possible equations of the circle -/
theorem tangent_circle_equation :
  ∀ c : TangentCircle,
  (∀ x y : ℝ, circle_equation c x y ↔ (x - 5)^2 + (y - 5)^2 = 25) ∨
  (∀ x y : ℝ, circle_equation c x y ↔ (x - 1)^2 + (y - 1)^2 = 1) :=
by sorry

end tangent_circle_equation_l3673_367315


namespace nine_pointed_star_sum_tip_angles_l3673_367327

/-- A 9-pointed star formed by connecting nine evenly spaced points on a circle -/
structure NinePointedStar where
  /-- The measure of the angle at each tip of the star -/
  tip_angle : ℝ
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The points are evenly spaced on the circle -/
  evenly_spaced : num_points = 9
  /-- The measure of the arc between two consecutive points -/
  arc_measure : ℝ
  /-- The arc measure is 360° divided by the number of points -/
  arc_measure_def : arc_measure = 360 / num_points
  /-- Each tip angle subtends an arc that spans 3 consecutive points -/
  tip_angle_subtends_three_arcs : tip_angle = 3 * arc_measure / 2

/-- The sum of the measures of all tip angles in a 9-pointed star is 540° -/
theorem nine_pointed_star_sum_tip_angles (star : NinePointedStar) :
  star.num_points * star.tip_angle = 540 := by
  sorry

end nine_pointed_star_sum_tip_angles_l3673_367327


namespace stamp_problem_l3673_367392

/-- Given stamps of denominations 5, n, and n+1 cents, 
    where n is a positive integer, 
    if 97 cents is the greatest postage that cannot be formed, 
    then n = 25 -/
theorem stamp_problem (n : ℕ) : 
  n > 0 → 
  (∀ k : ℕ, k > 97 → ∃ a b c : ℕ, k = 5*a + n*b + (n+1)*c) → 
  (∃ a b c : ℕ, 97 = 5*a + n*b + (n+1)*c → False) → 
  n = 25 := by
  sorry

end stamp_problem_l3673_367392


namespace percentage_runs_by_running_is_fifty_percent_l3673_367321

def total_runs : ℕ := 120
def boundaries : ℕ := 3
def sixes : ℕ := 8
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

def runs_from_boundaries_and_sixes : ℕ := boundaries * runs_per_boundary + sixes * runs_per_six

def runs_by_running : ℕ := total_runs - runs_from_boundaries_and_sixes

theorem percentage_runs_by_running_is_fifty_percent :
  (runs_by_running : ℚ) / total_runs * 100 = 50 := by sorry

end percentage_runs_by_running_is_fifty_percent_l3673_367321


namespace positive_A_value_l3673_367304

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + 3*B^2

-- Theorem statement
theorem positive_A_value :
  ∃ A : ℝ, A > 0 ∧ hash A 6 = 270 ∧ A = 9 * Real.sqrt 2 := by
  sorry

end positive_A_value_l3673_367304


namespace stating_river_width_determination_l3673_367377

/-- Represents the width of a river and the meeting points of two ferries --/
structure RiverCrossing where
  width : ℝ
  first_meeting : ℝ
  second_meeting : ℝ

/-- 
Theorem stating that if two ferries meet at specific points during their crossings, 
the river width can be determined.
-/
theorem river_width_determination (r : RiverCrossing) 
  (h1 : r.first_meeting = 720)
  (h2 : r.second_meeting = 400)
  (h3 : r.first_meeting + (r.width - r.first_meeting) = r.width)
  (h4 : r.width + r.second_meeting = 3 * r.first_meeting) :
  r.width = 1760 := by
  sorry

#check river_width_determination

end stating_river_width_determination_l3673_367377


namespace triangle_ratio_theorem_l3673_367356

/-- Given a triangle XYZ with points D on XY and E on YZ satisfying certain ratios,
    prove that DE:EF = 1:4 when DE intersects XZ at F. -/
theorem triangle_ratio_theorem (X Y Z D E F : ℝ × ℝ) : 
  -- Triangle XYZ exists
  (∃ (a b c : ℝ), X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X) →
  -- D is on XY with XD:DY = 4:1
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ D = (1 - t) • X + t • Y ∧ t = 1/5) →
  -- E is on YZ with YE:EZ = 4:1
  (∃ s : ℝ, s ∈ Set.Icc 0 1 ∧ E = (1 - s) • Y + s • Z ∧ s = 4/5) →
  -- DE intersects XZ at F
  (∃ r : ℝ, F = (1 - r) • D + r • E ∧ 
            ∃ q : ℝ, F = (1 - q) • X + q • Z) →
  -- Then DE:EF = 1:4
  ‖E - D‖ / ‖F - E‖ = 1/4 := by
sorry

end triangle_ratio_theorem_l3673_367356


namespace product_of_roots_l3673_367357

theorem product_of_roots (x : ℝ) : 
  let equation := (16 : ℝ) * x^2 + 60 * x - 200
  let product_of_roots := -200 / 16
  equation = 0 → product_of_roots = -(25 : ℝ) / 2 := by
  sorry

end product_of_roots_l3673_367357


namespace square_root_difference_limit_l3673_367351

theorem square_root_difference_limit : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |Real.sqrt (n + 1) - Real.sqrt n| < ε := by
sorry

end square_root_difference_limit_l3673_367351


namespace empty_quadratic_inequality_solution_set_l3673_367376

theorem empty_quadratic_inequality_solution_set
  (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) :=
sorry

end empty_quadratic_inequality_solution_set_l3673_367376


namespace p_plus_q_equals_twenty_l3673_367396

theorem p_plus_q_equals_twenty (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) →
  P + Q = 20 := by
sorry

end p_plus_q_equals_twenty_l3673_367396


namespace quadratic_equation_no_real_roots_l3673_367398

theorem quadratic_equation_no_real_roots 
  (a b c : ℝ) 
  (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  ∀ x : ℝ, x^2 + (a + b + c) * x + a^2 + b^2 + c^2 ≠ 0 :=
by sorry

end quadratic_equation_no_real_roots_l3673_367398


namespace chess_tournament_participants_l3673_367341

/-- The number of second-year students in the chess tournament --/
def n : ℕ := 7

/-- The total number of participants in the tournament --/
def total_participants : ℕ := n + 2

/-- The total number of games played in the tournament --/
def total_games : ℕ := (total_participants * (total_participants - 1)) / 2

/-- The total points scored in the tournament --/
def total_points : ℕ := total_games

/-- The points scored by the two freshmen --/
def freshman_points : ℕ := 8

/-- The points scored by all second-year students --/
def secondyear_points : ℕ := total_points - freshman_points

/-- The points scored by each second-year student --/
def points_per_secondyear : ℕ := secondyear_points / n

theorem chess_tournament_participants :
  n > 0 ∧
  total_participants = n + 2 ∧
  total_games = (total_participants * (total_participants - 1)) / 2 ∧
  total_points = total_games ∧
  freshman_points = 8 ∧
  secondyear_points = total_points - freshman_points ∧
  points_per_secondyear = secondyear_points / n ∧
  points_per_secondyear * n = secondyear_points ∧
  (∀ m : ℕ, m ≠ n → (m > 0 → 
    (m + 2) * (m + 1) / 2 - 8 ≠ ((m + 2) * (m + 1) / 2 - 8) / m * m)) :=
by sorry

end chess_tournament_participants_l3673_367341


namespace toby_work_hours_l3673_367393

/-- Given the working hours of Thomas, Toby, and Rebecca, prove that Toby worked 10 hours less than twice what Thomas worked. -/
theorem toby_work_hours (x : ℕ) : 
  -- Total hours worked
  x + (2 * x - 10) + 56 = 157 →
  -- Rebecca worked 56 hours
  56 = 56 →
  -- Rebecca worked 8 hours less than Toby
  56 = (2 * x - 10) - 8 →
  -- Toby worked 10 hours less than twice what Thomas worked
  (2 * x - (2 * x - 10)) = 10 := by
sorry

end toby_work_hours_l3673_367393


namespace smallest_prime_perimeter_scalene_triangle_l3673_367337

def is_odd_prime (n : ℕ) : Prop := Nat.Prime n ∧ n % 2 = 1

def is_scalene_triangle (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    is_odd_prime a →
    is_odd_prime b →
    is_odd_prime c →
    is_scalene_triangle a b c →
    Nat.Prime (a + b + c) →
    a + b + c ≥ 19 :=
sorry

end smallest_prime_perimeter_scalene_triangle_l3673_367337


namespace isosceles_triangle_properties_l3673_367360

/-- A triangle with sides 13, 13, and 10 units -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 13
  hb : b = 13
  hc : c = 10

/-- The sum of squares of medians of the triangle -/
def sumOfSquaresOfMedians (t : IsoscelesTriangle) : ℝ := sorry

/-- The area of the triangle -/
def triangleArea (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem stating the sum of squares of medians and area of the specific triangle -/
theorem isosceles_triangle_properties (t : IsoscelesTriangle) :
  sumOfSquaresOfMedians t = 278.5 ∧ triangleArea t = 60 := by sorry

end isosceles_triangle_properties_l3673_367360


namespace light_bulb_survey_not_appropriate_l3673_367326

-- Define the types of surveys
inductive SurveyMethod
| Sampling
| Comprehensive

-- Define the characteristics of a survey subject
structure SurveySubject where
  population_size : Nat
  requires_destruction : Bool

-- Define when a survey method is appropriate
def is_appropriate (method : SurveyMethod) (subject : SurveySubject) : Prop :=
  match method with
  | SurveyMethod.Sampling => subject.population_size > 100 ∨ subject.requires_destruction
  | SurveyMethod.Comprehensive => subject.population_size ≤ 100 ∧ ¬subject.requires_destruction

-- Theorem statement
theorem light_bulb_survey_not_appropriate :
  let light_bulbs : SurveySubject := ⟨1000, true⟩
  ¬(is_appropriate SurveyMethod.Comprehensive light_bulbs) :=
by sorry

end light_bulb_survey_not_appropriate_l3673_367326


namespace sales_at_540_l3673_367338

/-- Represents the sales model for a product -/
structure SalesModel where
  originalPrice : ℕ
  initialSales : ℕ
  reductionStep : ℕ
  salesIncreasePerStep : ℕ

/-- Calculates the sales volume given a price reduction -/
def salesVolume (model : SalesModel) (priceReduction : ℕ) : ℕ :=
  model.initialSales + (priceReduction / model.reductionStep) * model.salesIncreasePerStep

/-- Theorem stating the sales volume at a specific price point -/
theorem sales_at_540 (model : SalesModel) 
  (h1 : model.originalPrice = 600)
  (h2 : model.initialSales = 750)
  (h3 : model.reductionStep = 5)
  (h4 : model.salesIncreasePerStep = 30) :
  salesVolume model 60 = 1110 := by
  sorry

#eval salesVolume { originalPrice := 600, initialSales := 750, reductionStep := 5, salesIncreasePerStep := 30 } 60

end sales_at_540_l3673_367338


namespace extremum_point_condition_l3673_367362

open Real

theorem extremum_point_condition (a : ℝ) :
  (∀ b : ℝ, ∃! x : ℝ, x > 0 ∧ 
    (∀ y : ℝ, y > 0 → (exp (a * x) * (log x + b) ≥ exp (a * y) * (log y + b)) ∨
                      (exp (a * x) * (log x + b) ≤ exp (a * y) * (log y + b))))
  → a < 0 := by
sorry

end extremum_point_condition_l3673_367362


namespace additional_spend_for_free_delivery_l3673_367358

/-- The minimum amount required for free delivery -/
def min_for_free_delivery : ℚ := 35

/-- The price of chicken per pound -/
def chicken_price_per_pound : ℚ := 6

/-- The amount of chicken in pounds -/
def chicken_amount : ℚ := 3/2

/-- The price of lettuce -/
def lettuce_price : ℚ := 3

/-- The price of cherry tomatoes -/
def cherry_tomatoes_price : ℚ := 5/2

/-- The price of a sweet potato -/
def sweet_potato_price : ℚ := 3/4

/-- The number of sweet potatoes -/
def sweet_potato_count : ℕ := 4

/-- The price of a broccoli head -/
def broccoli_price : ℚ := 2

/-- The number of broccoli heads -/
def broccoli_count : ℕ := 2

/-- The price of Brussel sprouts -/
def brussel_sprouts_price : ℚ := 5/2

/-- The total cost of items in Alice's cart -/
def cart_total : ℚ :=
  chicken_price_per_pound * chicken_amount +
  lettuce_price +
  cherry_tomatoes_price +
  sweet_potato_price * sweet_potato_count +
  broccoli_price * broccoli_count +
  brussel_sprouts_price

/-- The theorem stating how much more Alice needs to spend for free delivery -/
theorem additional_spend_for_free_delivery :
  min_for_free_delivery - cart_total = 11 := by sorry

end additional_spend_for_free_delivery_l3673_367358


namespace circle_equation_m_range_l3673_367385

theorem circle_equation_m_range (m : ℝ) :
  (∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) →
  m < 1/2 :=
by sorry

end circle_equation_m_range_l3673_367385


namespace hyperbola_center_l3673_367348

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 81 * x - 16 * y^2 + 64 * y + 144 = 0

-- Define the center of a hyperbola
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), eq x y ↔ eq (x - c.1) (y - c.2)

-- Theorem statement
theorem hyperbola_center :
  is_center (9/2, 2) hyperbola_eq :=
sorry

end hyperbola_center_l3673_367348


namespace cubic_equation_result_l3673_367324

theorem cubic_equation_result (a : ℝ) (h : a^3 + 2*a = -2) :
  3*a^6 + 12*a^4 - a^3 + 12*a^2 - 2*a - 4 = 10 := by
  sorry

end cubic_equation_result_l3673_367324


namespace intersection_theorem_union_theorem_l3673_367333

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

theorem intersection_theorem : 
  A ∩ B a = {2} → a = -1 ∨ a = -3 := by sorry

theorem union_theorem : 
  A ∪ B a = A → a ≤ -3 := by sorry

end intersection_theorem_union_theorem_l3673_367333


namespace correct_product_l3673_367399

/- Define a function to reverse digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/- Main theorem -/
theorem correct_product (a b : ℕ) :
  a ≥ 10 ∧ a ≤ 99 ∧  -- a is a two-digit number
  0 < b ∧  -- b is positive
  (reverse_digits a) * b = 187 →
  a * b = 187 :=
by
  sorry


end correct_product_l3673_367399


namespace tile_arrangements_count_l3673_367306

def brown_tiles : ℕ := 2
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 2
def yellow_tiles : ℕ := 2
def orange_tiles : ℕ := 1

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles + orange_tiles

theorem tile_arrangements_count :
  (Nat.factorial total_tiles) / (Nat.factorial brown_tiles * Nat.factorial purple_tiles * 
   Nat.factorial green_tiles * Nat.factorial yellow_tiles * Nat.factorial orange_tiles) = 5040 := by
  sorry

end tile_arrangements_count_l3673_367306


namespace optimal_garden_dimensions_l3673_367345

/-- Represents the dimensions of a rectangular garden --/
structure GardenDimensions where
  perpendicular_side : ℝ
  parallel_side : ℝ

/-- Calculates the area of the garden given its dimensions --/
def garden_area (d : GardenDimensions) : ℝ :=
  d.perpendicular_side * d.parallel_side

/-- Represents the constraints of the garden problem --/
structure GardenConstraints where
  wall_length : ℝ
  fence_cost_per_foot : ℝ
  total_fence_cost : ℝ

/-- Theorem stating that the optimal garden dimensions maximize the area --/
theorem optimal_garden_dimensions (c : GardenConstraints)
  (h1 : c.wall_length = 300)
  (h2 : c.fence_cost_per_foot = 10)
  (h3 : c.total_fence_cost = 1500) :
  ∃ (d : GardenDimensions),
    d.parallel_side = 75 ∧
    ∀ (d' : GardenDimensions),
      d'.perpendicular_side + d'.perpendicular_side + d'.parallel_side = c.total_fence_cost / c.fence_cost_per_foot →
      garden_area d ≥ garden_area d' :=
sorry

end optimal_garden_dimensions_l3673_367345


namespace solve_for_d_l3673_367368

theorem solve_for_d (y : ℝ) (d : ℝ) (h1 : y > 0) (h2 : (6 * y) / 20 + (3 * y) / d = 0.60 * y) : d = 10 := by
  sorry

end solve_for_d_l3673_367368


namespace cattle_transport_speed_l3673_367310

/-- Proves that the speed of a truck transporting cattle is 60 miles per hour given specific conditions -/
theorem cattle_transport_speed (total_cattle : ℕ) (distance : ℕ) (truck_capacity : ℕ) (total_time : ℕ) :
  total_cattle = 400 →
  distance = 60 →
  truck_capacity = 20 →
  total_time = 40 →
  (distance * 2 * (total_cattle / truck_capacity)) / total_time = 60 := by
  sorry

#check cattle_transport_speed

end cattle_transport_speed_l3673_367310


namespace product_of_three_numbers_l3673_367383

theorem product_of_three_numbers (x y z : ℚ) 
  (sum_eq : x + y + z = 30)
  (first_eq : x = 3 * (y + z))
  (second_eq : y = 6 * z) :
  x * y * z = 23625 / 686 := by
  sorry

end product_of_three_numbers_l3673_367383


namespace bijection_property_l3673_367373

theorem bijection_property (k : ℕ) (f : ℤ → ℤ) 
  (h_bij : Function.Bijective f)
  (h_prop : ∀ i j : ℤ, |i - j| ≤ k → |f i - f j| ≤ k) :
  ∀ i j : ℤ, |f i - f j| = |i - j| := by
  sorry

end bijection_property_l3673_367373


namespace boat_cost_correct_l3673_367363

/-- The cost of taking a boat to the Island of Mysteries -/
def boat_cost : ℚ := 254

/-- The cost of taking a plane to the Island of Mysteries -/
def plane_cost : ℚ := 600

/-- The amount saved by taking the boat instead of the plane -/
def savings : ℚ := 346

/-- Theorem stating that the boat cost is correct given the plane cost and savings -/
theorem boat_cost_correct : boat_cost = plane_cost - savings := by sorry

end boat_cost_correct_l3673_367363


namespace garden_tiles_count_l3673_367305

/-- Represents a square garden covered with square tiles -/
structure SquareGarden where
  side_length : ℕ
  diagonal_tiles : ℕ

/-- The total number of tiles in a square garden -/
def total_tiles (garden : SquareGarden) : ℕ :=
  garden.side_length * garden.side_length

/-- The number of tiles on both diagonals of a square garden -/
def diagonal_tiles_count (garden : SquareGarden) : ℕ :=
  2 * garden.side_length - 1

theorem garden_tiles_count (garden : SquareGarden) 
  (h : diagonal_tiles_count garden = 25) : 
  total_tiles garden = 169 := by
  sorry

end garden_tiles_count_l3673_367305


namespace series_sum_l3673_367349

def series (n : ℕ) : ℚ :=
  if n = 0 then 2
  else if n = 1 then 3 + (1/2) * (series 0)
  else (1005 - n + 1 : ℚ) + (1/2) * (series (n-1))

theorem series_sum : series 1003 = 2008 := by sorry

end series_sum_l3673_367349


namespace triangle_right_angled_l3673_367364

theorem triangle_right_angled (A B C : Real) : 
  (Real.sin A) ^ 2 + (Real.sin B) ^ 2 + (Real.sin C) ^ 2 = 2 * ((Real.cos A) ^ 2 + (Real.cos B) ^ 2 + (Real.cos C) ^ 2) → 
  A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2 :=
by sorry

end triangle_right_angled_l3673_367364


namespace complex_number_problem_l3673_367300

theorem complex_number_problem (z₁ z₂ z : ℂ) : 
  z₁ = 1 - 2*I →
  z₂ = 4 + 3*I →
  Complex.abs z = 2 →
  Complex.im z = Complex.re (3*z₁ - z₂) →
  Complex.re z < 0 ∧ Complex.im z < 0 →
  z = -Real.sqrt 2 - I * Real.sqrt 2 :=
by sorry

end complex_number_problem_l3673_367300


namespace system_solution_l3673_367343

theorem system_solution (x y : ℝ) : 
  (2 * x + y = 5) ∧ (x - 3 * y = 6) ↔ (x = 3 ∧ y = -1) :=
by sorry

end system_solution_l3673_367343


namespace third_week_vegetable_intake_l3673_367390

/-- Represents the daily vegetable intake in pounds -/
structure DailyIntake where
  asparagus : ℝ
  broccoli : ℝ
  cauliflower : ℝ
  spinach : ℝ
  kale : ℝ
  zucchini : ℝ

/-- Calculates the total daily intake -/
def totalDailyIntake (intake : DailyIntake) : ℝ :=
  intake.asparagus + intake.broccoli + intake.cauliflower + intake.spinach + intake.kale + intake.zucchini

/-- Initial daily intake -/
def initialIntake : DailyIntake :=
  { asparagus := 0.25, broccoli := 0.25, cauliflower := 0.5, spinach := 0, kale := 0, zucchini := 0 }

/-- Daily intake after second week changes -/
def secondWeekIntake : DailyIntake :=
  { asparagus := initialIntake.asparagus * 2,
    broccoli := initialIntake.broccoli * 3,
    cauliflower := initialIntake.cauliflower * 1.75,
    spinach := 0.5,
    kale := 0,
    zucchini := 0 }

/-- Daily intake in the third week -/
def thirdWeekIntake : DailyIntake :=
  { asparagus := secondWeekIntake.asparagus,
    broccoli := secondWeekIntake.broccoli,
    cauliflower := secondWeekIntake.cauliflower,
    spinach := secondWeekIntake.spinach,
    kale := 0.5,  -- 1 pound every two days
    zucchini := 0.15 }  -- 0.3 pounds every two days

theorem third_week_vegetable_intake :
  totalDailyIntake thirdWeekIntake * 7 = 22.925 := by
  sorry

end third_week_vegetable_intake_l3673_367390


namespace value_of_y_minus_x_l3673_367379

theorem value_of_y_minus_x (x y : ℚ) 
  (h1 : x + y = 8) 
  (h2 : y - 3 * x = 7) : 
  y - x = 7.5 := by
sorry

end value_of_y_minus_x_l3673_367379


namespace p_necessary_not_sufficient_for_q_l3673_367314

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, (-1 < x ∧ x < 3) → x < 3) ∧
  (∃ x : ℝ, x < 3 ∧ ¬(-1 < x ∧ x < 3)) := by
  sorry

end p_necessary_not_sufficient_for_q_l3673_367314


namespace no_prime_sum_10003_l3673_367395

/-- A function that returns the number of ways to write n as the sum of two primes -/
def count_prime_sum_ways (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range n)).card

/-- Theorem stating that 10003 cannot be written as the sum of two primes -/
theorem no_prime_sum_10003 : count_prime_sum_ways 10003 = 0 := by
  sorry

end no_prime_sum_10003_l3673_367395


namespace math_books_prob_theorem_l3673_367380

/-- The probability of all three mathematics textbooks ending up in the same box -/
def math_books_same_box_prob (total_books n_math_books : ℕ) 
  (box_sizes : Fin 3 → ℕ) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem math_books_prob_theorem :
  let total_books : ℕ := 15
  let n_math_books : ℕ := 3
  let box_sizes : Fin 3 → ℕ := ![4, 5, 6]
  math_books_same_box_prob total_books n_math_books box_sizes = 9 / 121 :=
sorry

end math_books_prob_theorem_l3673_367380


namespace sequence_formula_l3673_367354

def geometric_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ k, k ≥ 1 → a (k + 1) - a k = (a 2 - a 1) * (2 ^ (k - 1))

theorem sequence_formula (a : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 →
  geometric_sequence a n →
  a 2 - a 1 = 2 →
  ∀ k, k ≥ 1 → a k = 2^k - 1 :=
sorry

end sequence_formula_l3673_367354


namespace circle_center_l3673_367374

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Theorem statement
theorem circle_center :
  ∃ (c : ℝ × ℝ), (c.1 = 1 ∧ c.2 = 0) ∧
  ∀ (x y : ℝ), circle_equation x y ↔ (x - c.1)^2 + (y - c.2)^2 = 1 :=
by
  sorry

end circle_center_l3673_367374


namespace largest_difference_l3673_367340

def A : ℕ := 3 * 2023^2024
def B : ℕ := 2023^2024
def C : ℕ := 2022 * 2023^2023
def D : ℕ := 3 * 2023^2023
def E : ℕ := 2023^2023
def F : ℕ := 2023^2022

theorem largest_difference : 
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) := by
  sorry

end largest_difference_l3673_367340


namespace unique_unbeatable_city_l3673_367308

/-- Represents a city with two bulldozers -/
structure City where
  leftBulldozer : ℕ
  rightBulldozer : ℕ

/-- Represents the road with n cities -/
def Road (n : ℕ) := Fin n → City

/-- A city i overtakes city j if its right bulldozer can reach j -/
def overtakes (road : Road n) (i j : Fin n) : Prop :=
  i < j ∧ ∀ k, i < k ∧ k ≤ j → (road i).rightBulldozer > (road k).leftBulldozer

/-- There exists a unique city that cannot be overtaken -/
theorem unique_unbeatable_city (n : ℕ) (road : Road n)
  (h1 : ∀ i j : Fin n, i ≠ j → (road i).leftBulldozer ≠ (road j).leftBulldozer)
  (h2 : ∀ i j : Fin n, i ≠ j → (road i).rightBulldozer ≠ (road j).rightBulldozer)
  (h3 : ∀ i : Fin n, (road i).leftBulldozer ≠ (road i).rightBulldozer) :
  ∃! i : Fin n, ∀ j : Fin n, j ≠ i → ¬(overtakes road j i) :=
sorry

end unique_unbeatable_city_l3673_367308


namespace ferris_wheel_capacity_l3673_367334

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 4

/-- The number of people that can sit in each seat -/
def people_per_seat : ℕ := 4

/-- The total number of people that can ride the Ferris wheel at the same time -/
def total_people : ℕ := num_seats * people_per_seat

theorem ferris_wheel_capacity : total_people = 16 := by
  sorry

end ferris_wheel_capacity_l3673_367334
