import Mathlib

namespace train_length_l3902_390273

theorem train_length (train_speed : Real) (bridge_length : Real) (crossing_time : Real) :
  train_speed = 45 * (1000 / 3600) ∧
  bridge_length = 220 ∧
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 155 := by
  sorry

end train_length_l3902_390273


namespace max_value_expression_l3902_390299

theorem max_value_expression (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : x^2 = 1) : 
  ∃ (m : ℝ), ∀ y, y^2 = 1 → x^2 + a + b + c * d * x ≤ m ∧ m = 2 := by
  sorry

end max_value_expression_l3902_390299


namespace lunks_needed_for_20_apples_l3902_390208

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℚ) : ℚ := (1/2) * l

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (k : ℚ) : ℚ := (5/3) * k

/-- The number of lunks required to purchase a given number of apples -/
def lunks_for_apples (a : ℚ) : ℚ := 
  let kunks := (3/5) * a
  (2/4) * kunks

theorem lunks_needed_for_20_apples : 
  lunks_for_apples 20 = 24 := by sorry

end lunks_needed_for_20_apples_l3902_390208


namespace percentage_increase_income_l3902_390200

/-- Calculate the percentage increase in combined weekly income --/
theorem percentage_increase_income (initial_job_income initial_side_income final_job_income final_side_income : ℝ) :
  initial_job_income = 50 →
  initial_side_income = 20 →
  final_job_income = 90 →
  final_side_income = 30 →
  let initial_total := initial_job_income + initial_side_income
  let final_total := final_job_income + final_side_income
  let increase := final_total - initial_total
  let percentage_increase := (increase / initial_total) * 100
  ∀ ε > 0, |percentage_increase - 71.43| < ε :=
by
  sorry

end percentage_increase_income_l3902_390200


namespace chess_team_size_l3902_390206

theorem chess_team_size (total_students : ℕ) (percentage : ℚ) (team_size : ℕ) : 
  total_students = 160 → percentage = 1/10 → team_size = (total_students : ℚ) * percentage → team_size = 16 := by
  sorry

end chess_team_size_l3902_390206


namespace system_solution_l3902_390286

theorem system_solution (x y : ℝ) (eq1 : 2*x + y = 5) (eq2 : x + 2*y = 4) : x + y = 3 := by
  sorry

end system_solution_l3902_390286


namespace tony_errands_halfway_distance_l3902_390290

theorem tony_errands_halfway_distance (groceries haircut doctor : ℕ) 
  (h1 : groceries = 10)
  (h2 : haircut = 15)
  (h3 : doctor = 5) :
  (groceries + haircut + doctor) / 2 = 15 :=
by sorry

end tony_errands_halfway_distance_l3902_390290


namespace unique_root_in_interval_l3902_390283

theorem unique_root_in_interval (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x, f x = -x^3 - x) →
  m ≤ n →
  f m * f n < 0 →
  ∃! x, m ≤ x ∧ x ≤ n ∧ f x = 0 :=
by sorry

end unique_root_in_interval_l3902_390283


namespace perpendicular_bisector_and_parallel_line_l3902_390284

-- Define points A, B, and P
def A : ℝ × ℝ := (6, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := x - 2*y - 8 = 0

-- Define the parallel line equation
def parallel_line (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_and_parallel_line :
  -- Part 1: Perpendicular bisector
  (∀ x y, perp_bisector x y ↔ 
    -- Midpoint condition
    ((x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 
     ((A.1 - B.1)/2)^2 + ((A.2 - B.2)/2)^2) ∧
    -- Perpendicularity condition
    ((y - A.2)*(B.1 - A.1) = -(x - A.1)*(B.2 - A.2))) ∧
  -- Part 2: Parallel line
  (∀ x y, parallel_line x y ↔
    -- Point P lies on the line
    (2*P.1 + P.2 - 1 = 0) ∧
    -- Parallel to AB
    ((y - P.2)/(x - P.1) = (B.2 - A.2)/(B.1 - A.1))) :=
sorry


end perpendicular_bisector_and_parallel_line_l3902_390284


namespace min_components_for_reliability_l3902_390271

/-- The probability of a single component working properly -/
def p : ℝ := 0.5

/-- The minimum required probability for the entire circuit to work properly -/
def min_prob : ℝ := 0.95

/-- The function that calculates the probability of the entire circuit working properly -/
def circuit_prob (n : ℕ) : ℝ := 1 - p^n

/-- Theorem stating the minimum number of components required -/
theorem min_components_for_reliability :
  ∃ n : ℕ, (∀ m : ℕ, m < n → circuit_prob m < min_prob) ∧ circuit_prob n ≥ min_prob :=
sorry

end min_components_for_reliability_l3902_390271


namespace total_views_and_likes_theorem_l3902_390259

def total_views_and_likes (
  initial_yt_views : ℕ) (initial_yt_likes : ℕ)
  (initial_other_views : ℕ) (initial_other_likes : ℕ)
  (yt_view_increase_factor : ℕ) (yt_like_increase_factor : ℕ)
  (other_view_increase_factor : ℕ) (other_like_increase_percent : ℕ)
  (additional_yt_views : ℕ) (additional_yt_likes : ℕ)
  (additional_other_views : ℕ) (additional_other_likes : ℕ) : ℕ × ℕ :=
  let yt_views_after_4_days := initial_yt_views + initial_yt_views * yt_view_increase_factor
  let yt_likes_after_4_days := initial_yt_likes + initial_yt_likes * (yt_like_increase_factor - 1)
  let other_views_after_4_days := initial_other_views + initial_other_views * (other_view_increase_factor - 1)
  let other_likes_after_4_days := initial_other_likes + initial_other_likes * other_like_increase_percent / 100
  let final_yt_views := yt_views_after_4_days + additional_yt_views
  let final_yt_likes := yt_likes_after_4_days + additional_yt_likes
  let final_other_views := other_views_after_4_days + additional_other_views
  let final_other_likes := other_likes_after_4_days + additional_other_likes
  (final_yt_views + final_other_views, final_yt_likes + final_other_likes)

theorem total_views_and_likes_theorem :
  total_views_and_likes 4000 500 2000 300 10 3 2 50 50000 2000 30000 500 = (130000, 5250) := by
  sorry

end total_views_and_likes_theorem_l3902_390259


namespace complex_equation_solution_l3902_390222

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I)^2 * z = 3 + 2 * Complex.I →
  z = -1 + (3/2) * Complex.I :=
by sorry

end complex_equation_solution_l3902_390222


namespace area_enclosed_by_cosine_and_lines_l3902_390204

theorem area_enclosed_by_cosine_and_lines :
  let f (x : ℝ) := Real.cos x
  let a : ℝ := -π/3
  let b : ℝ := π/3
  ∫ x in a..b, f x = Real.sqrt 3 :=
by sorry

end area_enclosed_by_cosine_and_lines_l3902_390204


namespace parabola_vertex_l3902_390238

/-- A parabola is defined by the equation y = -3(x-1)^2 - 2 -/
def parabola (x y : ℝ) : Prop := y = -3 * (x - 1)^2 - 2

/-- The vertex of a parabola is the point where it reaches its maximum or minimum -/
def is_vertex (x y : ℝ) : Prop := parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

/-- The vertex of the parabola y = -3(x-1)^2 - 2 is at (1, -2) -/
theorem parabola_vertex : is_vertex 1 (-2) := by sorry

end parabola_vertex_l3902_390238


namespace rectangle_dimension_change_l3902_390249

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.35 * L) (h2 : (L' * B') / (L * B) = 1.0665) : B' = 0.79 * B := by
  sorry

end rectangle_dimension_change_l3902_390249


namespace divide_by_fraction_twelve_divided_by_one_sixth_l3902_390237

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth : 12 / (1 / 6) = 72 := by sorry

end divide_by_fraction_twelve_divided_by_one_sixth_l3902_390237


namespace point_A_coordinates_l3902_390276

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point to the left -/
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

/-- Translate a point upwards -/
def translateUp (p : Point) (d : ℝ) : Point :=
  { x := p.x, y := p.y + d }

/-- The theorem stating the coordinates of point A -/
theorem point_A_coordinates (A : Point) 
  (hB : ∃ d : ℝ, translateLeft A d = Point.mk 1 2)
  (hC : ∃ d : ℝ, translateUp A d = Point.mk 3 4) : 
  A = Point.mk 3 2 := by sorry

end point_A_coordinates_l3902_390276


namespace quadratic_roots_sum_product_l3902_390214

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 10 ∧ x * y = 26) → 
  m + n = 108 := by
sorry

end quadratic_roots_sum_product_l3902_390214


namespace rs_value_l3902_390221

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs : 0 < s) 
  (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 5/8) : r * s = Real.sqrt 3 / 4 := by
  sorry

end rs_value_l3902_390221


namespace expand_and_simplify_l3902_390254

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end expand_and_simplify_l3902_390254


namespace solution_value_l3902_390235

theorem solution_value (a b : ℝ) (h : 2 * a - 3 * b - 5 = 0) : 2 * a - 3 * b + 3 = 8 := by
  sorry

end solution_value_l3902_390235


namespace parabola_focal_line_theorem_l3902_390201

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point2D | p.y^2 = 4 * p.x}

/-- The focal length of the parabola y^2 = 4x -/
def focal_length : ℝ := 2

/-- A line passing through the focus of the parabola -/
structure FocalLine where
  intersects_parabola : Point2D → Point2D → Prop

/-- The length of a line segment between two points -/
def line_segment_length (A B : Point2D) : ℝ := sorry

theorem parabola_focal_line_theorem (l : FocalLine) (A B : Point2D) 
  (h1 : A ∈ Parabola) (h2 : B ∈ Parabola) 
  (h3 : l.intersects_parabola A B) 
  (h4 : (A.x + B.x) / 2 = 3) :
  line_segment_length A B = 8 := by sorry

end parabola_focal_line_theorem_l3902_390201


namespace max_b_when_a_is_e_min_a_minus_b_l3902_390228

open Real

-- Define the condition that e^x ≥ ax + b for all x
def condition (a b : ℝ) : Prop := ∀ x, exp x ≥ a * x + b

theorem max_b_when_a_is_e :
  (condition e b) → b ≤ 0 :=
sorry

theorem min_a_minus_b :
  ∃ a b, condition a b ∧ ∀ a' b', condition a' b' → a - b ≤ a' - b' ∧ a - b = -1/e :=
sorry

end max_b_when_a_is_e_min_a_minus_b_l3902_390228


namespace parallelogram_perimeter_plus_area_l3902_390232

/-- A parallelogram with integer coordinates -/
structure Parallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  sorry

/-- The area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  sorry

theorem parallelogram_perimeter_plus_area :
  let p : Parallelogram := ⟨(1,1), (6,3), (9,3), (4,1)⟩
  perimeter p + area p = 2 * Real.sqrt 29 + 12 := by
  sorry

end parallelogram_perimeter_plus_area_l3902_390232


namespace original_paint_intensity_l3902_390264

/-- Proves that the original paint intensity was 50% given the mixing conditions --/
theorem original_paint_intensity 
  (replaced_fraction : ℚ) 
  (replacement_intensity : ℚ) 
  (final_intensity : ℚ) : 
  replaced_fraction = 2/3 → 
  replacement_intensity = 1/5 → 
  final_intensity = 3/10 → 
  (1 - replaced_fraction) * (1/2) + replaced_fraction * replacement_intensity = final_intensity := by
  sorry

#eval (1 - 2/3) * (1/2) + 2/3 * (1/5) == 3/10

end original_paint_intensity_l3902_390264


namespace juice_distribution_l3902_390279

theorem juice_distribution (C : ℝ) (h : C > 0) : 
  let juice_volume := (2/3) * C
  let cups := 4
  let juice_per_cup := juice_volume / cups
  juice_per_cup / C = 1/6 := by sorry

end juice_distribution_l3902_390279


namespace gcd_polynomial_and_b_l3902_390207

theorem gcd_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^4 + 2 * b^3 + 5 * b^2 + 9 * b + 95) b = 95 := by
  sorry

end gcd_polynomial_and_b_l3902_390207


namespace no_mode_in_set_l3902_390230

def number_set : Finset ℕ := {91, 85, 80, 83, 84}

def x : ℕ := 504 - (91 + 85 + 80 + 83 + 84)

def complete_set : Finset ℕ := number_set ∪ {x}

theorem no_mode_in_set :
  (Finset.card complete_set = 6) ∧
  (Finset.sum complete_set id / Finset.card complete_set = 84) →
  ∀ n : ℕ, (complete_set.filter (λ m => m = n)).card ≤ 1 :=
by sorry

end no_mode_in_set_l3902_390230


namespace infinitely_many_consecutive_right_triangles_l3902_390269

/-- A right triangle with integer sides where the hypotenuse and one side are consecutive. -/
structure ConsecutiveRightTriangle where
  a : ℕ  -- One side of the triangle
  b : ℕ  -- The other side of the triangle
  c : ℕ  -- The hypotenuse
  consecutive : c = a + 1
  pythagorean : a^2 + b^2 = c^2

/-- There exist infinitely many ConsecutiveRightTriangles. -/
theorem infinitely_many_consecutive_right_triangles :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ∃ t : ConsecutiveRightTriangle, t.c = m :=
by sorry

end infinitely_many_consecutive_right_triangles_l3902_390269


namespace sum_of_floors_l3902_390242

def floor (x : ℚ) : ℤ := Int.floor x

theorem sum_of_floors : 
  (floor (2017 * 3 / 11 : ℚ)) + 
  (floor (2017 * 4 / 11 : ℚ)) + 
  (floor (2017 * 5 / 11 : ℚ)) + 
  (floor (2017 * 6 / 11 : ℚ)) + 
  (floor (2017 * 7 / 11 : ℚ)) + 
  (floor (2017 * 8 / 11 : ℚ)) = 6048 := by
sorry

end sum_of_floors_l3902_390242


namespace arccos_one_equals_zero_l3902_390281

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_equals_zero_l3902_390281


namespace congruence_solution_l3902_390298

theorem congruence_solution (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 201) (h3 : 200 * n ≡ 144 [ZMOD 101]) :
  n ≡ 29 [ZMOD 101] := by
  sorry

end congruence_solution_l3902_390298


namespace complex_square_sum_positive_l3902_390250

theorem complex_square_sum_positive (z₁ z₂ z₃ : ℂ) :
  (z₁^2 + z₂^2 : ℂ).re > (-z₃^2 : ℂ).re → (z₁^2 + z₂^2 + z₃^2 : ℂ).re > 0 := by
  sorry

end complex_square_sum_positive_l3902_390250


namespace total_faces_painted_l3902_390263

/-- The number of cuboids Amelia painted -/
def num_cuboids : ℕ := 6

/-- The number of faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- Theorem stating the total number of faces painted by Amelia -/
theorem total_faces_painted : num_cuboids * faces_per_cuboid = 36 := by
  sorry

end total_faces_painted_l3902_390263


namespace minimum_computer_units_l3902_390223

theorem minimum_computer_units (x : ℕ) : x ≥ 105 ↔ 
  (5500 * 60 + 5000 * (x - 60) > 550000) := by sorry

end minimum_computer_units_l3902_390223


namespace taxi_distance_calculation_l3902_390287

/-- Calculates the total distance of a taxi ride given the fare structure and total charge --/
theorem taxi_distance_calculation (initial_charge : ℚ) (initial_distance : ℚ) 
  (additional_charge : ℚ) (additional_distance : ℚ) (total_charge : ℚ) :
  initial_charge = 2.5 →
  initial_distance = 1/5 →
  additional_charge = 0.4 →
  additional_distance = 1/5 →
  total_charge = 18.1 →
  ∃ (total_distance : ℚ), total_distance = 8 ∧
    total_charge = initial_charge + 
      (total_distance - initial_distance) / additional_distance * additional_charge :=
by
  sorry


end taxi_distance_calculation_l3902_390287


namespace train_passing_time_l3902_390291

/-- The time taken for two trains to completely pass each other -/
theorem train_passing_time (length_A length_B : ℝ) (speed_A speed_B : ℝ) 
  (h1 : length_A = 150)
  (h2 : length_B = 150)
  (h3 : speed_A = 54 * (5/18))
  (h4 : speed_B = 36 * (5/18))
  (h5 : speed_A > 0)
  (h6 : speed_B > 0) :
  (length_A + length_B) / (speed_A + speed_B) = 12 := by
  sorry

end train_passing_time_l3902_390291


namespace complex_sum_zero_implies_b_equals_two_l3902_390243

theorem complex_sum_zero_implies_b_equals_two (b : ℝ) : 
  (2 : ℂ) - Complex.I * b = (2 : ℂ) - Complex.I * b ∧ 
  (2 : ℝ) + (-b) = 0 → b = 2 := by
  sorry

end complex_sum_zero_implies_b_equals_two_l3902_390243


namespace tan_negative_23pi_over_3_l3902_390277

theorem tan_negative_23pi_over_3 : Real.tan (-23 * Real.pi / 3) = Real.sqrt 3 := by
  sorry

end tan_negative_23pi_over_3_l3902_390277


namespace books_on_shelf_correct_book_count_l3902_390266

theorem books_on_shelf (initial_figures : ℕ) (added_figures : ℕ) (extra_books : ℕ) : ℕ :=
  let total_figures := initial_figures + added_figures
  let total_books := total_figures + extra_books
  total_books

theorem correct_book_count : books_on_shelf 2 4 4 = 10 := by
  sorry

end books_on_shelf_correct_book_count_l3902_390266


namespace minimize_reciprocal_sum_l3902_390294

theorem minimize_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 30) :
  (1 / a + 4 / b ≥ 8 / 15) ∧
  (1 / a + 4 / b = 8 / 15 ↔ a = 15 / 4 ∧ b = 15) := by
sorry

end minimize_reciprocal_sum_l3902_390294


namespace problem_solution_l3902_390213

theorem problem_solution (x y z : ℚ) 
  (h1 : 2*x + y + z = 14)
  (h2 : 2*x + y = 7)
  (h3 : x + 2*y = 10) :
  (x + y - z) / 3 = -4/9 := by
  sorry

end problem_solution_l3902_390213


namespace monkey_rope_system_length_l3902_390297

/-- Represents the age and weight of a monkey and its mother, and the properties of a rope system -/
structure MonkeyRopeSystem where
  monkey_age : ℝ
  mother_age : ℝ
  rope_weight_per_foot : ℝ
  weight : ℝ

/-- The conditions of the monkey-rope system problem -/
def monkey_rope_system_conditions (s : MonkeyRopeSystem) : Prop :=
  s.monkey_age + s.mother_age = 4 ∧
  s.monkey_age = s.mother_age / 2 ∧
  s.rope_weight_per_foot = 1/4 ∧
  s.weight = s.mother_age

/-- The theorem stating that under the given conditions, the rope length is 5 feet -/
theorem monkey_rope_system_length
  (s : MonkeyRopeSystem)
  (h : monkey_rope_system_conditions s) :
  (s.weight + s.weight) / (3/4) = 5 :=
sorry

end monkey_rope_system_length_l3902_390297


namespace expression_evaluation_l3902_390231

theorem expression_evaluation : -2^3 + 36 / 3^2 * (-1/2) + |(-5)| = -5 := by
  sorry

end expression_evaluation_l3902_390231


namespace complex_power_six_l3902_390224

theorem complex_power_six : (2 + 3*I : ℂ)^6 = -845 + 2028*I := by sorry

end complex_power_six_l3902_390224


namespace correct_costs_l3902_390288

/-- The cost of a pen in yuan -/
def pen_cost : ℝ := 10

/-- The cost of an exercise book in yuan -/
def book_cost : ℝ := 1

/-- The total cost of 2 exercise books and 1 pen in yuan -/
def total_cost : ℝ := 12

theorem correct_costs :
  (2 * book_cost + pen_cost = total_cost) ∧
  (book_cost = 0.1 * pen_cost) ∧
  (pen_cost = 10) ∧
  (book_cost = 1) := by sorry

end correct_costs_l3902_390288


namespace estate_area_calculation_l3902_390209

/-- Represents the scale of the map in miles per inch -/
def scale : ℚ := 300 / 2

/-- The length of the first side of the rectangle on the map in inches -/
def side1_map : ℚ := 10

/-- The length of the second side of the rectangle on the map in inches -/
def side2_map : ℚ := 6

/-- Converts a length on the map to the actual length in miles -/
def map_to_miles (map_length : ℚ) : ℚ := map_length * scale

/-- Calculates the area of a rectangle given its side lengths -/
def rectangle_area (length width : ℚ) : ℚ := length * width

theorem estate_area_calculation :
  rectangle_area (map_to_miles side1_map) (map_to_miles side2_map) = 1350000 := by
  sorry

end estate_area_calculation_l3902_390209


namespace solution_set_of_inequality_l3902_390268

-- Define the function f(x) = ax + b
def f (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem solution_set_of_inequality 
  (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f a b x < f a b y) 
  (h_intersect : f a b 2 = 0) :
  {x : ℝ | b * x^2 - a * x > 0} = {x : ℝ | -1/2 < x ∧ x < 0} := by
sorry

end solution_set_of_inequality_l3902_390268


namespace function_upper_bound_l3902_390262

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)

/-- A function that is bounded on [0,1] -/
def IsBoundedOnUnitInterval (f : ℝ → ℝ) : Prop :=
  ∃ M > 0, ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M

theorem function_upper_bound
    (f : ℝ → ℝ)
    (h1 : SatisfiesInequality f)
    (h2 : IsBoundedOnUnitInterval f) :
    ∀ x, x ≥ 0 → f x ≤ (1/2) * x^2 := by
  sorry

end function_upper_bound_l3902_390262


namespace prime_divisors_of_50_factorial_l3902_390240

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of prime divisors of 50! is equal to the number of prime numbers less than or equal to 50. -/
theorem prime_divisors_of_50_factorial (p : ℕ → Prop) :
  (∃ (n : ℕ), p n ∧ n ∣ factorial 50) ↔ (∃ (n : ℕ), p n ∧ n ≤ 50) :=
sorry

end prime_divisors_of_50_factorial_l3902_390240


namespace intersection_complement_equal_set_l3902_390203

def M : Set ℝ := {-1, 0, 1, 3}
def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement_equal_set : M ∩ (Set.univ \ N) = {0, 1} := by sorry

end intersection_complement_equal_set_l3902_390203


namespace banana_arrangements_l3902_390219

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (letterFrequencies : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (letterFrequencies.map Nat.factorial).prod

/-- The number of distinct arrangements of the letters in "banana" -/
def bananaArrangements : ℕ :=
  distinctArrangements 6 [1, 2, 3]

theorem banana_arrangements :
  bananaArrangements = 60 := by
  sorry

#eval bananaArrangements

end banana_arrangements_l3902_390219


namespace parabola_axis_of_symmetry_l3902_390296

/-- The parabola function -/
def f (x : ℝ) : ℝ := (2 - x) * x

/-- The axis of symmetry of the parabola -/
def axis_of_symmetry : ℝ := 1

/-- Theorem: The axis of symmetry of the parabola y = (2-x)x is the line x = 1 -/
theorem parabola_axis_of_symmetry :
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
by sorry

end parabola_axis_of_symmetry_l3902_390296


namespace quadratic_local_symmetry_exponential_local_symmetry_range_l3902_390248

/-- A function f has a local symmetry point at x₀ if f(-x₀) = -f(x₀) -/
def has_local_symmetry_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f (-x₀) = -f x₀

/-- Theorem 1: The quadratic function ax² + bx - a has a local symmetry point -/
theorem quadratic_local_symmetry
  (a b : ℝ) (ha : a ≠ 0) :
  ∃ x₀ : ℝ, has_local_symmetry_point (fun x ↦ a * x^2 + b * x - a) x₀ :=
sorry

/-- Theorem 2: Range of m for which 4^x - m * 2^(n+1) + m - 3 has a local symmetry point -/
theorem exponential_local_symmetry_range (n : ℝ) :
  ∃ m_min m_max : ℝ,
    (∀ m : ℝ, (∃ x₀ : ℝ, has_local_symmetry_point (fun x ↦ 4^x - m * 2^(n+1) + m - 3) x₀)
              ↔ m_min ≤ m ∧ m ≤ m_max) ∧
    m_min = 1 - Real.sqrt 3 ∧
    m_max = 2 * Real.sqrt 2 :=
sorry

end quadratic_local_symmetry_exponential_local_symmetry_range_l3902_390248


namespace fraction_problem_l3902_390217

theorem fraction_problem (f : ℚ) : 
  (1 / 5 : ℚ)^4 * f^2 = 1 / (10 : ℚ)^4 → f = 1 / 4 := by
  sorry

end fraction_problem_l3902_390217


namespace mersenne_factor_square_plus_nine_l3902_390247

theorem mersenne_factor_square_plus_nine (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n.val = 2^k :=
sorry

end mersenne_factor_square_plus_nine_l3902_390247


namespace not_necessary_nor_sufficient_condition_l3902_390282

theorem not_necessary_nor_sufficient_condition (a : ℝ) :
  ¬(∀ x : ℝ, a * x^2 + a * x - 1 < 0 ↔ a < 0) :=
sorry

end not_necessary_nor_sufficient_condition_l3902_390282


namespace mary_trip_time_and_cost_l3902_390253

-- Define the problem parameters
def uber_to_house : ℕ := 10
def uber_cost : ℚ := 15
def airport_time_factor : ℕ := 5
def bag_check_time : ℕ := 15
def luggage_fee_eur : ℚ := 20
def security_time_factor : ℕ := 3
def boarding_wait : ℕ := 20
def takeoff_wait_factor : ℕ := 2
def first_layover : ℕ := 205  -- 3 hours 25 minutes in minutes
def flight_delay : ℕ := 45
def second_layover : ℕ := 110  -- 1 hour 50 minutes in minutes
def time_zone_change : ℕ := 3
def usd_to_eur : ℚ := 0.85
def usd_to_gbp : ℚ := 0.75
def meal_cost_gbp : ℚ := 10

-- Define the theorem
theorem mary_trip_time_and_cost :
  let total_time : ℕ := uber_to_house + (uber_to_house * airport_time_factor) + 
                        bag_check_time + (bag_check_time * security_time_factor) + 
                        boarding_wait + (boarding_wait * takeoff_wait_factor) + 
                        first_layover + flight_delay + second_layover
  let total_time_hours : ℕ := total_time / 60 + time_zone_change
  let total_cost : ℚ := uber_cost + (luggage_fee_eur / usd_to_eur) + (meal_cost_gbp / usd_to_gbp)
  total_time_hours = 12 ∧ total_cost = 51.86 := by sorry

end mary_trip_time_and_cost_l3902_390253


namespace inequality_proof_l3902_390234

theorem inequality_proof (a b c : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end inequality_proof_l3902_390234


namespace divisor_equation_solution_l3902_390229

def is_sixth_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (d1 d2 d3 d4 d5 : ℕ), d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧ d4 ∣ n ∧ d5 ∣ n ∧
    1 < d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 < d)

def is_seventh_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (d1 d2 d3 d4 d5 d6 : ℕ), d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧ d4 ∣ n ∧ d5 ∣ n ∧ d6 ∣ n ∧
    1 < d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 < d6 ∧ d6 < d)

theorem divisor_equation_solution (n : ℕ) :
  (∃ (d6 d7 : ℕ), is_sixth_divisor n d6 ∧ is_seventh_divisor n d7 ∧ n = d6^2 + d7^2 - 1) →
  n = 144 ∨ n = 1984 :=
by sorry

end divisor_equation_solution_l3902_390229


namespace students_in_both_competitions_l3902_390215

theorem students_in_both_competitions 
  (total_students : ℕ) 
  (math_participants : ℕ) 
  (physics_participants : ℕ) 
  (no_competition_participants : ℕ) 
  (h1 : total_students = 40)
  (h2 : math_participants = 31)
  (h3 : physics_participants = 20)
  (h4 : no_competition_participants = 8) :
  total_students = math_participants + physics_participants + no_competition_participants - 19 := by
sorry

end students_in_both_competitions_l3902_390215


namespace middle_manager_sample_size_l3902_390261

theorem middle_manager_sample_size
  (total_employees : ℕ) (total_middle_managers : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 1000)
  (h2 : total_middle_managers = 150)
  (h3 : sample_size = 200) :
  (total_middle_managers : ℚ) / total_employees * sample_size = 30 :=
by sorry

end middle_manager_sample_size_l3902_390261


namespace prime_divisibility_problem_l3902_390292

theorem prime_divisibility_problem (p q : ℕ) : 
  Prime p → Prime q → p < 2005 → q < 2005 → 
  (q ∣ p^2 + 4) → (p ∣ q^2 + 4) → 
  p = 2 ∧ q = 2 := by
sorry

end prime_divisibility_problem_l3902_390292


namespace cone_volume_l3902_390251

/-- The volume of a cone with slant height 5 cm and height 4 cm is 12π cm³ -/
theorem cone_volume (l h : ℝ) (hl : l = 5) (hh : h = 4) :
  let r := Real.sqrt (l^2 - h^2)
  (1/3 : ℝ) * Real.pi * r^2 * h = 12 * Real.pi := by
  sorry

end cone_volume_l3902_390251


namespace sqrt_equation_solution_l3902_390274

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end sqrt_equation_solution_l3902_390274


namespace segment_length_l3902_390270

/-- Given a line segment AB divided by points P and Q, prove that the length of AB is 135/7 -/
theorem segment_length (A B P Q : ℝ) : 
  (∃ x y : ℝ, 
    A < P ∧ P < Q ∧ Q < B ∧  -- P and Q are between A and B
    P - A = 3*x ∧ B - P = 2*x ∧  -- P divides AB in ratio 3:2
    Q - A = 4*y ∧ B - Q = 5*y ∧  -- Q divides AB in ratio 4:5
    Q - P = 3)  -- Distance between P and Q is 3
  → B - A = 135/7 := by
sorry

end segment_length_l3902_390270


namespace trig_identity_l3902_390236

theorem trig_identity : 
  2 * Real.sin (50 * π / 180) + 
  Real.sin (10 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) * 
  Real.sqrt (2 * Real.sin (80 * π / 180) ^ 2) = Real.sqrt 6 := by sorry

end trig_identity_l3902_390236


namespace fraction_equality_implies_values_l3902_390260

theorem fraction_equality_implies_values (a b : ℚ) : 
  (∀ n : ℕ, (1 : ℚ) / ((2 * n - 1) * (2 * n + 1)) = a / (2 * n - 1) + b / (2 * n + 1)) →
  a = 1 / 2 ∧ b = -1 / 2 := by
  sorry

end fraction_equality_implies_values_l3902_390260


namespace square_side_length_l3902_390258

theorem square_side_length (circle_area : ℝ) (h1 : circle_area = 100) :
  let square_perimeter := circle_area
  let square_side := square_perimeter / 4
  square_side = 25 := by sorry

end square_side_length_l3902_390258


namespace sum_of_digits_of_product_94_nines_94_fours_l3902_390212

/-- A number consisting of n repeated digits d -/
def repeatedDigits (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_product_94_nines_94_fours :
  sumOfDigits (repeatedDigits 94 9 * repeatedDigits 94 4) = 846 := by sorry

end sum_of_digits_of_product_94_nines_94_fours_l3902_390212


namespace william_land_percentage_l3902_390233

-- Define the total tax collected from the village
def total_tax : ℝ := 3840

-- Define Mr. William's tax payment
def william_tax : ℝ := 480

-- Define the percentage of cultivated land that is taxed
def tax_percentage : ℝ := 0.9

-- Theorem statement
theorem william_land_percentage :
  (william_tax / total_tax) * 100 = 12.5 := by
sorry

end william_land_percentage_l3902_390233


namespace simplify_polynomial_expression_l3902_390225

theorem simplify_polynomial_expression (x : ℝ) : 
  3 * ((5 * x^2 - 4 * x + 8) - (3 * x^2 - 2 * x + 6)) = 6 * x^2 - 6 * x + 6 := by
  sorry

end simplify_polynomial_expression_l3902_390225


namespace hyperbola_m_value_l3902_390218

/-- The equation of a hyperbola with one focus at (-3,0) -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 - y^2 / m = 1

/-- The focus of the hyperbola is at (-3,0) -/
def focus_at_minus_three : ℝ × ℝ := (-3, 0)

/-- Theorem stating that m = 8 for the given hyperbola -/
theorem hyperbola_m_value :
  ∃ (m : ℝ), (∀ (x y : ℝ), hyperbola_equation x y m) ∧ 
  (focus_at_minus_three.1 = -3) ∧ (focus_at_minus_three.2 = 0) →
  m = 8 :=
sorry

end hyperbola_m_value_l3902_390218


namespace solution_set_min_value_g_inequality_proof_l3902_390289

-- Define the absolute value function
def f (x : ℝ) : ℝ := |x|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- Statement for part 1
theorem solution_set (x : ℝ) :
  f ((1 / 2^x) - 2) ≤ 1 ↔ x ∈ Set.Icc (Real.log 3 / Real.log 2) 0 :=
sorry

-- Statement for part 2
theorem min_value_g :
  ∃ (m : ℝ), m = 1 ∧ ∀ x, g x ≥ m :=
sorry

-- Statement for part 3
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
sorry

end solution_set_min_value_g_inequality_proof_l3902_390289


namespace a_eq_one_sufficient_not_necessary_l3902_390210

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, (a = 1 → ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) :=
sorry

end a_eq_one_sufficient_not_necessary_l3902_390210


namespace angle_sum_sine_l3902_390256

/-- Given an angle θ with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (t, 2t) where t < 0,
    prove that sin(θ + π/3) = -(2√5 + √15)/10 -/
theorem angle_sum_sine (t : ℝ) (θ : ℝ) (h1 : t < 0) 
    (h2 : Real.cos θ = -1 / Real.sqrt 5) 
    (h3 : Real.sin θ = -2 / Real.sqrt 5) : 
  Real.sin (θ + π/3) = -(2 * Real.sqrt 5 + Real.sqrt 15) / 10 := by
sorry

end angle_sum_sine_l3902_390256


namespace least_integer_absolute_value_l3902_390226

theorem least_integer_absolute_value (x : ℤ) : (∀ y : ℤ, |3*y + 4| ≤ 18 → y ≥ -7) ∧ |3*(-7) + 4| ≤ 18 := by
  sorry

end least_integer_absolute_value_l3902_390226


namespace quartic_roots_polynomial_problem_l3902_390205

theorem quartic_roots_polynomial_problem (a b c d : ℂ) (P : ℂ → ℂ) :
  (a^4 + 4*a^3 + 6*a^2 + 8*a + 10 = 0) →
  (b^4 + 4*b^3 + 6*b^2 + 8*b + 10 = 0) →
  (c^4 + 4*c^3 + 6*c^2 + 8*c + 10 = 0) →
  (d^4 + 4*d^3 + 6*d^2 + 8*d + 10 = 0) →
  (P a = b + c + d) →
  (P b = a + c + d) →
  (P c = a + b + d) →
  (P d = a + b + c) →
  (P (a + b + c + d) = -20) →
  (∀ x, P x = -10/37*x^4 - 30/37*x^3 - 56/37*x^2 - 118/37*x - 148/37) :=
by sorry

end quartic_roots_polynomial_problem_l3902_390205


namespace regular_polygon_with_144_degree_angle_has_10_sides_l3902_390220

theorem regular_polygon_with_144_degree_angle_has_10_sides :
  ∀ n : ℕ,
  n ≥ 3 →
  (n - 2) * 180 / n = 144 →
  n = 10 := by
sorry

end regular_polygon_with_144_degree_angle_has_10_sides_l3902_390220


namespace unique_number_with_conditions_l3902_390280

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_three_digit n ∧ digit_sum n = 25 ∧ n % 5 = 0

theorem unique_number_with_conditions :
  ∃! n : ℕ, satisfies_conditions n :=
sorry

end unique_number_with_conditions_l3902_390280


namespace fractional_equation_m_range_l3902_390252

theorem fractional_equation_m_range : 
  ∀ (x m : ℝ), 
    ((x + m) / (x - 2) - (2 * m) / (x - 2) = 3) →
    (x > 0) →
    (m < 6 ∧ m ≠ 2) := by
  sorry

end fractional_equation_m_range_l3902_390252


namespace complex_sum_theorem_l3902_390246

theorem complex_sum_theorem :
  let A : ℂ := 3 + 2*I
  let O : ℂ := -1 - 2*I
  let P : ℂ := 2*I
  let S : ℂ := 1 + 3*I
  A - O + P + S = 5 + 9*I :=
by sorry

end complex_sum_theorem_l3902_390246


namespace negation_of_existence_negation_of_proposition_l3902_390216

theorem negation_of_existence (P : ℕ → Prop) : 
  (¬∃ n, P n) ↔ (∀ n, ¬P n) := by sorry

theorem negation_of_proposition :
  (¬∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end negation_of_existence_negation_of_proposition_l3902_390216


namespace polynomial_multiple_condition_l3902_390239

/-- A polynomial f(x) of the form x^4 + p x^2 + q x + a^2 is a multiple of (x^2 - 1) 
    if and only if p = -(a^2 + 1), q = 0, and the other factor is (x^2 - a^2) -/
theorem polynomial_multiple_condition (a : ℝ) :
  ∃ (p q : ℝ), ∀ (x : ℝ), 
    (x^4 + p*x^2 + q*x + a^2 = (x^2 - 1) * (x^2 - a^2)) ↔ 
    (p = -(a^2 + 1) ∧ q = 0) :=
by sorry

end polynomial_multiple_condition_l3902_390239


namespace dog_weight_l3902_390227

theorem dog_weight (d k r : ℚ) 
  (total_weight : d + k + r = 40)
  (dog_rabbit_weight : d + r = 2 * k)
  (dog_kitten_weight : d + k = r) : 
  d = 20 / 3 := by
sorry

end dog_weight_l3902_390227


namespace elderly_in_sample_l3902_390267

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  young : ℕ
  middle : ℕ
  elderly : ℕ

/-- Represents the sampled employees -/
structure SampledEmployees where
  young : ℕ
  elderly : ℕ

/-- Theorem stating the number of elderly employees in the sample -/
theorem elderly_in_sample
  (total : ℕ)
  (employees : EmployeeCount)
  (sample : SampledEmployees)
  (h1 : total = employees.young + employees.middle + employees.elderly)
  (h2 : employees.young = 160)
  (h3 : employees.middle = 2 * employees.elderly)
  (h4 : total = 430)
  (h5 : sample.young = 32)
  : sample.elderly = 18 := by
  sorry

end elderly_in_sample_l3902_390267


namespace square_division_l3902_390295

theorem square_division (s : ℝ) (w : ℝ) (h : w = 5) :
  ∃ (a b c d e : ℝ),
    s = 20 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    a + b + w = s ∧
    c + d = s ∧
    a * c = b * d ∧
    a * c = w * e ∧
    a * c = (s - c) * (s - a - b) :=
by sorry

#check square_division

end square_division_l3902_390295


namespace double_counted_page_l3902_390241

theorem double_counted_page (n : ℕ) (m : ℕ) : 
  n > 0 ∧ m > 0 ∧ m ≤ n ∧ (n * (n + 1)) / 2 + m = 2040 → m = 24 := by
  sorry

end double_counted_page_l3902_390241


namespace wang_liang_set_exists_l3902_390211

theorem wang_liang_set_exists : ∃ (a b : ℕ), 
  1 ≤ a ∧ a ≤ 13 ∧ 
  1 ≤ b ∧ b ≤ 13 ∧ 
  (a - a / b) * b = 24 ∧ 
  (a ≠ 4 ∨ b ≠ 7) := by
  sorry

end wang_liang_set_exists_l3902_390211


namespace commission_for_398_machines_l3902_390275

/-- Represents the commission structure and pricing model for machine sales -/
structure SalesModel where
  initialPrice : ℝ
  priceDecrease : ℝ
  commissionRate1 : ℝ
  commissionRate2 : ℝ
  commissionRate3 : ℝ
  threshold1 : ℕ
  threshold2 : ℕ

/-- Calculates the total commission for a given number of machines sold -/
def calculateCommission (model : SalesModel) (machinesSold : ℕ) : ℝ :=
  sorry

/-- The specific sales model for the problem -/
def problemModel : SalesModel :=
  { initialPrice := 10000
    priceDecrease := 500
    commissionRate1 := 0.03
    commissionRate2 := 0.04
    commissionRate3 := 0.05
    threshold1 := 150
    threshold2 := 250 }

theorem commission_for_398_machines :
  calculateCommission problemModel 398 = 150000 := by
  sorry

end commission_for_398_machines_l3902_390275


namespace binary_digit_difference_l3902_390244

/-- Returns the number of digits in the base-2 representation of a natural number -/
def binaryDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

/-- The difference in the number of digits between the base-2 representations of 1200 and 200 is 3 -/
theorem binary_digit_difference : binaryDigits 1200 - binaryDigits 200 = 3 := by
  sorry

end binary_digit_difference_l3902_390244


namespace triangle_dance_nine_people_l3902_390265

def triangle_dance_rounds (n : ℕ) : ℕ :=
  if n % 3 = 0 then
    (Nat.factorial n) / ((Nat.factorial 3)^3 * Nat.factorial (n / 3))
  else
    0

theorem triangle_dance_nine_people :
  triangle_dance_rounds 9 = 280 :=
by sorry

end triangle_dance_nine_people_l3902_390265


namespace f_range_implies_m_plus_n_range_l3902_390285

-- Define the function f(x)
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

-- Define the interval [m, n]
def interval (m n : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ n }

-- State the theorem
theorem f_range_implies_m_plus_n_range (m n : ℝ) :
  (∀ x ∈ interval m n, -6 ≤ f x ∧ f x ≤ 2) →
  (0 ≤ m + n ∧ m + n ≤ 4) :=
by sorry

end f_range_implies_m_plus_n_range_l3902_390285


namespace total_animals_hunted_l3902_390278

theorem total_animals_hunted (sam rob mark peter : ℕ) : 
  sam = 6 →
  rob = sam / 2 →
  mark = (sam + rob) / 3 →
  peter = 3 * mark →
  sam + rob + mark + peter = 21 :=
by
  sorry

end total_animals_hunted_l3902_390278


namespace parallel_not_sufficient_nor_necessary_l3902_390272

-- Define the types for planes and lines
def Plane : Type := sorry
def Line : Type := sorry

-- Define the parallel relation for planes
def parallel (α β : Plane) : Prop := sorry

-- Define the subset relation for a line and a plane
def subset_plane (m : Line) (β : Plane) : Prop := sorry

-- Theorem statement
theorem parallel_not_sufficient_nor_necessary 
  (α β : Plane) (m : Line) (h : parallel α β) :
  ¬(∀ m, subset_plane m β → parallel α β) ∧ 
  ¬(∀ m, parallel α β → subset_plane m β) := by
  sorry

end parallel_not_sufficient_nor_necessary_l3902_390272


namespace ali_and_leila_trip_cost_l3902_390202

/-- The total cost of a trip for two people with a given original price and discount. -/
def trip_cost (original_price discount : ℕ) : ℕ :=
  2 * (original_price - discount)

/-- Theorem stating that the trip cost for Ali and Leila is $266. -/
theorem ali_and_leila_trip_cost :
  trip_cost 147 14 = 266 := by
  sorry

#eval trip_cost 147 14  -- This line is optional, for verification purposes

end ali_and_leila_trip_cost_l3902_390202


namespace eight_items_four_categories_l3902_390257

/-- The number of ways to assign n distinguishable items to k distinct categories -/
def assignments (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 65536 ways to assign 8 distinguishable items to 4 distinct categories -/
theorem eight_items_four_categories : assignments 8 4 = 65536 := by
  sorry

end eight_items_four_categories_l3902_390257


namespace enthalpy_relationship_l3902_390245

/-- Represents the enthalpy change of a chemical reaction -/
structure EnthalpyChange where
  value : ℝ
  units : String

/-- Represents a chemical reaction with its enthalpy change -/
structure ChemicalReaction where
  equation : String
  enthalpyChange : EnthalpyChange

/-- Given chemical reactions and their enthalpy changes, prove that 2a = b < 0 -/
theorem enthalpy_relationship (
  reaction1 reaction2 reaction3 reaction4 : ChemicalReaction
) (h1 : reaction1.equation = "H₂(g) + ½O₂(g) → H₂O(g)")
  (h2 : reaction2.equation = "2H₂(g) + O₂(g) → 2H₂O(g)")
  (h3 : reaction3.equation = "H₂(g) + ½O₂(g) → H₂O(l)")
  (h4 : reaction4.equation = "2H₂(g) + O₂(g) → 2H₂O(l)")
  (h5 : reaction1.enthalpyChange.units = "KJ·mol⁻¹")
  (h6 : reaction2.enthalpyChange.units = "KJ·mol⁻¹")
  (h7 : reaction3.enthalpyChange.units = "KJ·mol⁻¹")
  (h8 : reaction4.enthalpyChange.units = "KJ·mol⁻¹")
  (h9 : reaction1.enthalpyChange.value = reaction3.enthalpyChange.value)
  (h10 : reaction2.enthalpyChange.value = reaction4.enthalpyChange.value) :
  2 * reaction1.enthalpyChange.value = reaction2.enthalpyChange.value ∧ 
  reaction2.enthalpyChange.value < 0 := by
  sorry


end enthalpy_relationship_l3902_390245


namespace hemisphere_surface_area_l3902_390293

theorem hemisphere_surface_area (r : ℝ) (h : r = 6) :
  let sphere_area := λ r : ℝ => 4 * π * r^2
  let base_area := π * r^2
  let hemisphere_area := sphere_area r / 2 + base_area
  hemisphere_area = 108 * π := by sorry

end hemisphere_surface_area_l3902_390293


namespace sine_sum_equality_l3902_390255

theorem sine_sum_equality : 
  Real.sin (7 * π / 30) + Real.sin (11 * π / 30) - Real.sin (π / 30) - Real.sin (13 * π / 30) = 1/2 := by
  sorry

end sine_sum_equality_l3902_390255
