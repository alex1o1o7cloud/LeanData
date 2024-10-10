import Mathlib

namespace rectangle_arrangement_exists_l2462_246292

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents an arrangement of rectangles -/
structure Arrangement where
  height : ℝ
  width : ℝ

/-- Theorem: It is possible to arrange five identical rectangles with perimeter 10
    to form a single rectangle with perimeter 22 -/
theorem rectangle_arrangement_exists : ∃ (small : Rectangle) (arr : Arrangement),
  perimeter small = 10 ∧
  arr.height = 5 * small.length ∧
  arr.width = small.width ∧
  2 * (arr.height + arr.width) = 22 := by
  sorry

end rectangle_arrangement_exists_l2462_246292


namespace work_completion_time_l2462_246259

/-- The time taken to complete a work given two workers with different rates and a specific work pattern. -/
theorem work_completion_time 
  (p_time q_time : ℝ) 
  (solo_time : ℝ) 
  (h1 : p_time > 0) 
  (h2 : q_time > 0) 
  (h3 : solo_time > 0) 
  (h4 : solo_time < p_time) :
  let p_rate := 1 / p_time
  let q_rate := 1 / q_time
  let work_done_solo := solo_time * p_rate
  let remaining_work := 1 - work_done_solo
  let combined_rate := p_rate + q_rate
  let remaining_time := remaining_work / combined_rate
  solo_time + remaining_time = 20 := by sorry

end work_completion_time_l2462_246259


namespace solution_set_f_shifted_empty_solution_set_l2462_246207

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for part 1
theorem solution_set_f_shifted (x : ℝ) :
  f (x + 2) ≥ 2 ↔ x ≤ -3/2 ∨ x ≥ 1/2 :=
sorry

-- Theorem for part 2
theorem empty_solution_set (a : ℝ) :
  (∀ x, f x ≥ a) ↔ a ≤ 1 :=
sorry

end solution_set_f_shifted_empty_solution_set_l2462_246207


namespace rohans_age_is_25_l2462_246283

/-- Rohan's current age in years -/
def rohans_current_age : ℕ := 25

/-- Rohan's age 15 years ago -/
def rohans_past_age : ℕ := rohans_current_age - 15

/-- Rohan's age 15 years from now -/
def rohans_future_age : ℕ := rohans_current_age + 15

/-- Theorem stating that Rohan's current age is 25, given the condition -/
theorem rohans_age_is_25 :
  rohans_current_age = 25 ∧
  rohans_future_age = 4 * rohans_past_age :=
by sorry

end rohans_age_is_25_l2462_246283


namespace base3_to_base10_conversion_l2462_246243

/-- Converts a base 3 number represented as a list of digits to its base 10 equivalent -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number 102012₃ -/
def base3Number : List Nat := [2, 1, 0, 2, 0, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 302 := by
  sorry

end base3_to_base10_conversion_l2462_246243


namespace rotate_d_180_degrees_l2462_246254

/-- Rotation of a point by 180° about the origin -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotate_d_180_degrees :
  let d : ℝ × ℝ := (2, -3)
  rotate180 d = (-2, 3) := by
  sorry

end rotate_d_180_degrees_l2462_246254


namespace cosine_sum_zero_l2462_246230

theorem cosine_sum_zero (n : ℤ) (h : n % 7 = 1 ∨ n % 7 = 3 ∨ n % 7 = 4) :
  Real.cos (n * π / 7 - 13 * π / 14) + 
  Real.cos (3 * n * π / 7 - 3 * π / 14) + 
  Real.cos (5 * n * π / 7 - 3 * π / 14) = 0 := by
  sorry

end cosine_sum_zero_l2462_246230


namespace square_brush_ratio_l2462_246216

theorem square_brush_ratio (s w : ℝ) (h : s > 0) (h' : w > 0) : 
  w^2 + ((s - w)^2) / 2 = s^2 / 3 → s / w = 3 := by
  sorry

end square_brush_ratio_l2462_246216


namespace jerry_always_escapes_l2462_246277

/-- Represents the square pool -/
structure Pool :=
  (side : ℝ)
  (is_positive : side > 0)

/-- Represents the speeds of Tom and Jerry -/
structure Speeds :=
  (jerry_swim : ℝ)
  (tom_run : ℝ)
  (speed_ratio : tom_run = 4 * jerry_swim)
  (positive_speeds : jerry_swim > 0 ∧ tom_run > 0)

/-- Represents a point in the pool -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defines whether a point is inside or on the edge of the pool -/
def in_pool (p : Point) (pool : Pool) : Prop :=
  0 ≤ p.x ∧ p.x ≤ pool.side ∧ 0 ≤ p.y ∧ p.y ≤ pool.side

/-- Defines whether Jerry can escape from Tom -/
def can_escape (pool : Pool) (speeds : Speeds) : Prop :=
  ∀ (jerry_start tom_start : Point),
    in_pool jerry_start pool →
    ¬in_pool tom_start pool →
    ∃ (escape_point : Point),
      in_pool escape_point pool ∧
      (escape_point.x = 0 ∨ escape_point.x = pool.side ∨
       escape_point.y = 0 ∨ escape_point.y = pool.side) ∧
      (escape_point.x - jerry_start.x) ^ 2 + (escape_point.y - jerry_start.y) ^ 2 <
      ((escape_point.x - tom_start.x) ^ 2 + (escape_point.y - tom_start.y) ^ 2) * (speeds.jerry_swim / speeds.tom_run) ^ 2

theorem jerry_always_escapes (pool : Pool) (speeds : Speeds) :
  can_escape pool speeds :=
sorry

end jerry_always_escapes_l2462_246277


namespace quadratic_equation_coefficients_l2462_246244

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ), 
    (∀ x, 3 * x^2 = 2 * x - 3 ↔ a * x^2 + b * x + c = 0) →
    a = 3 ∧ b = -2 ∧ c = 3 := by
  sorry

end quadratic_equation_coefficients_l2462_246244


namespace sequence_2007th_term_l2462_246256

theorem sequence_2007th_term (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 0 = 1 →
  (∀ n, a (n + 2) = 6 * a n - a (n + 1)) →
  a 2007 = 2^2007 := by
sorry

end sequence_2007th_term_l2462_246256


namespace empty_solution_set_range_min_value_distance_sum_l2462_246266

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ (-1 < a ∧ a < 0) := by
  sorry

theorem min_value_distance_sum : 
  ∀ x : ℝ, |x - 1| + |x - 2| ≥ 1 := by
  sorry

end empty_solution_set_range_min_value_distance_sum_l2462_246266


namespace fruit_cost_difference_l2462_246255

theorem fruit_cost_difference : 
  let grapes_kg : ℝ := 7
  let grapes_price : ℝ := 70
  let grapes_discount : ℝ := 0.10
  let grapes_tax : ℝ := 0.05

  let mangoes_kg : ℝ := 9
  let mangoes_price : ℝ := 55
  let mangoes_discount : ℝ := 0.05
  let mangoes_tax : ℝ := 0.07

  let apples_kg : ℝ := 5
  let apples_price : ℝ := 40
  let apples_discount : ℝ := 0.08
  let apples_tax : ℝ := 0.03

  let oranges_kg : ℝ := 3
  let oranges_price : ℝ := 30
  let oranges_discount : ℝ := 0.15
  let oranges_tax : ℝ := 0.06

  let mangoes_cost := mangoes_kg * mangoes_price * (1 - mangoes_discount) * (1 + mangoes_tax)
  let apples_cost := apples_kg * apples_price * (1 - apples_discount) * (1 + apples_tax)

  mangoes_cost - apples_cost = 313.6475 := by sorry

end fruit_cost_difference_l2462_246255


namespace circles_configuration_l2462_246282

-- Define the centers of the circles as points in a metric space
variable (X : Type) [MetricSpace X]
variable (P Q R : X)

-- Define the radii of the circles
variable (p q r : ℝ)

-- Define the distance between P and Q
variable (d : ℝ)

-- State the theorem
theorem circles_configuration (h1 : p > q) (h2 : q > r) 
  (h3 : dist R P < p) (h4 : dist R Q < q) (h5 : d = dist P Q) :
  ¬(p + r = d) := by
  sorry

end circles_configuration_l2462_246282


namespace zoe_pictures_before_dolphin_show_l2462_246233

/-- The number of pictures Zoe took at the dolphin show -/
def pictures_at_dolphin_show : ℕ := 16

/-- The total number of pictures Zoe has taken -/
def total_pictures : ℕ := 44

/-- The number of pictures Zoe took before the dolphin show -/
def pictures_before_dolphin_show : ℕ := total_pictures - pictures_at_dolphin_show

theorem zoe_pictures_before_dolphin_show :
  pictures_before_dolphin_show = 28 :=
by sorry

end zoe_pictures_before_dolphin_show_l2462_246233


namespace inequality_proof_l2462_246278

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end inequality_proof_l2462_246278


namespace inscribed_quadrilateral_incenters_form_rectangle_l2462_246279

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle in the Euclidean plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- A quadrilateral in the Euclidean plane -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The incenter of a triangle -/
def incenter (A B C : Point) : Point := sorry

/-- Predicate to check if a quadrilateral is inscribed in a circle -/
def is_inscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Predicate to check if a quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop := sorry

theorem inscribed_quadrilateral_incenters_form_rectangle 
  (ABCD : Quadrilateral) (c : Circle) :
  is_inscribed ABCD c →
  let I_A := incenter ABCD.B ABCD.C ABCD.D
  let I_B := incenter ABCD.C ABCD.D ABCD.A
  let I_C := incenter ABCD.D ABCD.A ABCD.B
  let I_D := incenter ABCD.A ABCD.B ABCD.C
  is_rectangle (Quadrilateral.mk I_A I_B I_C I_D) :=
sorry

end inscribed_quadrilateral_incenters_form_rectangle_l2462_246279


namespace range_equals_std_dev_l2462_246221

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation
  symmetric : Bool
  within_range : ℝ → ℝ  -- function that gives the proportion within a range
  less_than : ℝ → ℝ  -- function that gives the proportion less than a value

/-- Theorem stating the relationship between the range and standard deviation -/
theorem range_equals_std_dev (D : SymmetricDistribution) (R : ℝ) :
  D.symmetric = true →
  D.within_range R = 0.68 →
  D.less_than (D.μ + R) = 0.84 →
  R = D.σ :=
by sorry

end range_equals_std_dev_l2462_246221


namespace parallel_line_through_A_l2462_246293

-- Define the point A
def A : ℝ × ℝ × ℝ := (-2, 3, 1)

-- Define the planes that form the given line
def plane1 (x y z : ℝ) : Prop := x - 2*y - z - 2 = 0
def plane2 (x y z : ℝ) : Prop := 2*x + 3*y - z + 1 = 0

-- Define the direction vector of the given line
def direction_vector : ℝ × ℝ × ℝ := (5, -1, 7)

-- Define the equation of the parallel line passing through A
def parallel_line (x y z : ℝ) : Prop :=
  (x + 2) / 5 = (y - 3) / (-1) ∧ (y - 3) / (-1) = (z - 1) / 7

-- Theorem statement
theorem parallel_line_through_A :
  ∀ (x y z : ℝ), 
    (∃ (t : ℝ), x = -2 + 5*t ∧ y = 3 - t ∧ z = 1 + 7*t) →
    parallel_line x y z :=
sorry

end parallel_line_through_A_l2462_246293


namespace julians_comic_book_pages_l2462_246274

theorem julians_comic_book_pages 
  (frames_per_page : ℝ) 
  (total_frames : ℕ) 
  (h1 : frames_per_page = 143.0) 
  (h2 : total_frames = 1573) : 
  ⌊(total_frames : ℝ) / frames_per_page⌋ = 11 := by
sorry

end julians_comic_book_pages_l2462_246274


namespace product_of_sum_and_sum_of_cubes_l2462_246258

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) : 
  a + b = 5 → a^3 + b^3 = 35 → a * b = 6 := by
  sorry

end product_of_sum_and_sum_of_cubes_l2462_246258


namespace power_of_two_greater_than_linear_l2462_246253

theorem power_of_two_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2*n + 1 := by
  sorry

end power_of_two_greater_than_linear_l2462_246253


namespace max_sock_pairs_john_sock_problem_l2462_246251

theorem max_sock_pairs (initial_pairs : ℕ) (lost_socks : ℕ) : ℕ :=
  let total_socks := 2 * initial_pairs
  let remaining_socks := total_socks - lost_socks
  let guaranteed_pairs := initial_pairs - lost_socks
  let possible_new_pairs := (remaining_socks - 2 * guaranteed_pairs) / 2
  guaranteed_pairs + possible_new_pairs

theorem john_sock_problem :
  max_sock_pairs 10 5 = 7 := by
  sorry

end max_sock_pairs_john_sock_problem_l2462_246251


namespace pizza_cost_is_seven_l2462_246206

def pizza_problem (box_cost : ℚ) : Prop :=
  let num_boxes : ℕ := 5
  let tip_ratio : ℚ := 1 / 7
  let total_paid : ℚ := 40
  let pizza_cost : ℚ := box_cost * num_boxes
  let tip : ℚ := pizza_cost * tip_ratio
  pizza_cost + tip = total_paid

theorem pizza_cost_is_seven :
  ∃ (box_cost : ℚ), pizza_problem box_cost ∧ box_cost = 7 :=
by sorry

end pizza_cost_is_seven_l2462_246206


namespace derivative_even_implies_b_zero_l2462_246252

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- Define the derivative of f
def f_deriv (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- State the theorem
theorem derivative_even_implies_b_zero (a b c : ℝ) :
  (∀ x : ℝ, f_deriv a b c x = f_deriv a b c (-x)) →
  b = 0 := by sorry

end derivative_even_implies_b_zero_l2462_246252


namespace complex_number_theorem_l2462_246247

theorem complex_number_theorem (z : ℂ) : 
  (∃ (k : ℝ), z / (1 + Complex.I) = k * Complex.I) ∧ 
  Complex.abs (z / (1 + Complex.I)) = 1 → 
  z = -1 + Complex.I ∨ z = 1 - Complex.I :=
by sorry

end complex_number_theorem_l2462_246247


namespace relationship_holds_l2462_246236

def x : Fin 5 → ℕ
  | ⟨0, _⟩ => 1
  | ⟨1, _⟩ => 2
  | ⟨2, _⟩ => 3
  | ⟨3, _⟩ => 4
  | ⟨4, _⟩ => 5

def y : Fin 5 → ℕ
  | ⟨0, _⟩ => 4
  | ⟨1, _⟩ => 15
  | ⟨2, _⟩ => 40
  | ⟨3, _⟩ => 85
  | ⟨4, _⟩ => 156

theorem relationship_holds : ∀ i : Fin 5, y i = (x i)^3 + 2*(x i) + 1 := by
  sorry

end relationship_holds_l2462_246236


namespace total_distinct_plants_l2462_246270

def X : ℕ := 600
def Y : ℕ := 500
def Z : ℕ := 400
def XY : ℕ := 70
def XZ : ℕ := 80
def YZ : ℕ := 60
def XYZ : ℕ := 30

theorem total_distinct_plants : X + Y + Z - XY - XZ - YZ + XYZ = 1320 := by
  sorry

end total_distinct_plants_l2462_246270


namespace complex_number_in_second_quadrant_l2462_246299

theorem complex_number_in_second_quadrant :
  let z : ℂ := (3 + 4 * Complex.I) * Complex.I
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_in_second_quadrant_l2462_246299


namespace max_operation_value_l2462_246263

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The operation to be maximized -/
def operation (X Y Z : Digit) : ℕ := 99 * X.val + 9 * Y.val - 9 * Z.val

/-- The theorem statement -/
theorem max_operation_value :
  ∃ (X Y Z : Digit), X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧
    operation X Y Z = 900 ∧
    ∀ (A B C : Digit), A ≠ B → B ≠ C → A ≠ C →
      operation A B C ≤ 900 :=
sorry

end max_operation_value_l2462_246263


namespace hyperbola_equation_l2462_246268

theorem hyperbola_equation (ellipse : (ℝ × ℝ) → Prop) 
  (ellipse_eq : ∀ x y, ellipse (x, y) ↔ x^2/27 + y^2/36 = 1)
  (shared_foci : ∃ f1 f2 : ℝ × ℝ, (∀ x y, ellipse (x, y) → 
    (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 36) ∧
    (∀ x y, x^2/4 - y^2/5 = 1 → 
    (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 9))
  (point_on_hyperbola : (Real.sqrt 15)^2/4 - 4^2/5 = 1) :
  ∀ x y, x^2/4 - y^2/5 = 1 ↔ 
    ∃ f1 f2 : ℝ × ℝ, (∀ a b, ellipse (a, b) → 
    (a - f1.1)^2 + (b - f1.2)^2 - ((a - f2.1)^2 + (b - f2.2)^2) = 36) ∧
    (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 9 :=
sorry


end hyperbola_equation_l2462_246268


namespace area_TURS_l2462_246296

/-- Rectangle PQRS with trapezoid TURS inside -/
structure Geometry where
  /-- Width of rectangle PQRS -/
  width : ℝ
  /-- Height of rectangle PQRS -/
  height : ℝ
  /-- Area of rectangle PQRS -/
  area_PQRS : ℝ
  /-- Distance of T from S -/
  ST_distance : ℝ
  /-- Distance of U from R -/
  UR_distance : ℝ
  /-- Width is 6 units -/
  width_eq : width = 6
  /-- Height is 4 units -/
  height_eq : height = 4
  /-- Area of PQRS is 24 square units -/
  area_eq : area_PQRS = 24
  /-- ST distance is 1 unit -/
  ST_eq : ST_distance = 1
  /-- UR distance is 1 unit -/
  UR_eq : UR_distance = 1

/-- The area of trapezoid TURS is 20 square units -/
theorem area_TURS (g : Geometry) : Real.sqrt ((g.width - 2 * g.ST_distance) * g.height + g.ST_distance * g.height) = 20 := by
  sorry

end area_TURS_l2462_246296


namespace simple_interest_calculation_l2462_246215

def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem simple_interest_calculation :
  let principal : ℚ := 80325
  let rate : ℚ := 1
  let time : ℚ := 5
  simple_interest principal rate time = 4016.25 := by sorry

end simple_interest_calculation_l2462_246215


namespace weight_increase_percentage_l2462_246202

/-- The percentage increase in total weight of two people given their initial weight ratio and individual weight increases -/
theorem weight_increase_percentage 
  (ram_ratio : ℝ) 
  (shyam_ratio : ℝ) 
  (ram_increase : ℝ) 
  (shyam_increase : ℝ) 
  (new_total_weight : ℝ) 
  (h1 : ram_ratio = 2) 
  (h2 : shyam_ratio = 5) 
  (h3 : ram_increase = 0.1) 
  (h4 : shyam_increase = 0.17) 
  (h5 : new_total_weight = 82.8) : 
  ∃ (percentage_increase : ℝ), 
    abs (percentage_increase - 15.06) < 0.01 ∧ 
    percentage_increase = 
      (new_total_weight - (ram_ratio + shyam_ratio) * 
        (new_total_weight / (ram_ratio * (1 + ram_increase) + shyam_ratio * (1 + shyam_increase)))) / 
      ((ram_ratio + shyam_ratio) * 
        (new_total_weight / (ram_ratio * (1 + ram_increase) + shyam_ratio * (1 + shyam_increase)))) 
      * 100 := by
  sorry


end weight_increase_percentage_l2462_246202


namespace equal_numbers_exist_l2462_246217

/-- A 10x10 grid of integers -/
def Grid := Fin 10 → Fin 10 → ℤ

/-- Two cells are adjacent if they differ by 1 in exactly one coordinate -/
def adjacent (i j i' j' : Fin 10) : Prop :=
  (i = i' ∧ j.val + 1 = j'.val) ∨
  (i = i' ∧ j'.val + 1 = j.val) ∨
  (j = j' ∧ i.val + 1 = i'.val) ∨
  (j = j' ∧ i'.val + 1 = i.val)

/-- The property that adjacent cells differ by at most 5 -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j i' j', adjacent i j i' j' → |g i j - g i' j'| ≤ 5

theorem equal_numbers_exist (g : Grid) (h : valid_grid g) :
  ∃ i j i' j', (i ≠ i' ∨ j ≠ j') ∧ g i j = g i' j' :=
sorry

end equal_numbers_exist_l2462_246217


namespace part_one_part_two_l2462_246298

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem for part (I)
theorem part_one (a : ℝ) : p a → a ≤ 1 := by
  sorry

-- Theorem for part (II)
theorem part_two (a : ℝ) : ¬(p a ∧ q a) → a ∈ Set.union (Set.Ioo (-2) 1) (Set.Ioi 1) := by
  sorry

end part_one_part_two_l2462_246298


namespace interior_angles_sum_plus_three_l2462_246273

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: Given a convex polygon with n sides whose interior angles sum to 2340 degrees,
    the sum of interior angles of a convex polygon with n + 3 sides is 2880 degrees. -/
theorem interior_angles_sum_plus_three (n : ℕ) 
  (h : sum_interior_angles n = 2340) : 
  sum_interior_angles (n + 3) = 2880 := by
  sorry


end interior_angles_sum_plus_three_l2462_246273


namespace smallest_integer_with_remainder_one_l2462_246232

theorem smallest_integer_with_remainder_one : ∃ k : ℕ, 
  k > 1 ∧ 
  k % 10 = 1 ∧ 
  k % 15 = 1 ∧ 
  k % 9 = 1 ∧
  (∀ m : ℕ, m > 1 ∧ m % 10 = 1 ∧ m % 15 = 1 ∧ m % 9 = 1 → k ≤ m) ∧
  k = 91 :=
by sorry

end smallest_integer_with_remainder_one_l2462_246232


namespace num_persons_is_nine_l2462_246275

/-- The number of persons who went to the hotel -/
def num_persons : ℕ := 9

/-- The amount spent by each of the first 8 persons -/
def amount_per_person : ℕ := 12

/-- The additional amount spent by the 9th person above the average -/
def additional_amount : ℕ := 8

/-- The total expenditure of all persons -/
def total_expenditure : ℕ := 117

/-- Theorem stating that the number of persons who went to the hotel is 9 -/
theorem num_persons_is_nine :
  (num_persons - 1) * amount_per_person + 
  ((num_persons - 1) * amount_per_person + additional_amount) / num_persons + additional_amount = 
  total_expenditure :=
sorry

end num_persons_is_nine_l2462_246275


namespace work_completion_time_l2462_246201

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 9

/-- The number of days x needs to finish the remaining work after y left -/
def x_remaining : ℝ := 8

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 20

theorem work_completion_time :
  x_days = 20 :=
sorry

end work_completion_time_l2462_246201


namespace trig_expression_equals_sqrt_two_l2462_246250

/-- Proves that the given trigonometric expression equals √2 --/
theorem trig_expression_equals_sqrt_two :
  (Real.cos (10 * π / 180) - Real.sqrt 3 * Real.cos (-100 * π / 180)) /
  Real.sqrt (1 - Real.sin (10 * π / 180)) = Real.sqrt 2 := by
  sorry

end trig_expression_equals_sqrt_two_l2462_246250


namespace corner_square_probability_l2462_246248

-- Define the grid size
def gridSize : Nat := 4

-- Define the number of squares to be selected
def squaresSelected : Nat := 3

-- Define the number of corner squares
def cornerSquares : Nat := 4

-- Define the total number of squares
def totalSquares : Nat := gridSize * gridSize

-- Define the probability of selecting at least one corner square
def probabilityAtLeastOneCorner : Rat := 17 / 28

theorem corner_square_probability :
  (1 : Rat) - (Nat.choose (totalSquares - cornerSquares) squaresSelected : Rat) / 
  (Nat.choose totalSquares squaresSelected) = probabilityAtLeastOneCorner := by
  sorry

end corner_square_probability_l2462_246248


namespace part_one_part_two_part_three_l2462_246214

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 1) * x + 3

-- Part 1
theorem part_one (a b : ℝ) (ha : a ≠ 0) 
  (h_solution_set : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) :
  2 * a + b = -3 := by sorry

-- Part 2
theorem part_two (a b : ℝ) (ha : a ≠ 0) (hf1 : f a b 1 = 5) (hb : b > -1) :
  (∀ a' b', a' ≠ 0 → b' > -1 → f a' b' 1 = 5 → 
    1 / |a| + 4 * |a| / (b + 1) ≤ 1 / |a'| + 4 * |a'| / (b' + 1)) ∧
  1 / |a| + 4 * |a| / (b + 1) = 2 := by sorry

-- Part 3
theorem part_three (a : ℝ) (ha : a ≠ 0) :
  let b := -a - 3
  let solution_set := {x : ℝ | f a b x < -2 * x + 1}
  (a < 0 → solution_set = {x | x < 2/a ∨ x > 1}) ∧
  (0 < a ∧ a < 2 → solution_set = {x | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → solution_set = ∅) ∧
  (a > 2 → solution_set = {x | 2/a < x ∧ x < 1}) := by sorry

end part_one_part_two_part_three_l2462_246214


namespace dishonest_dealer_profit_l2462_246241

/-- Calculates the profit percentage for a dishonest dealer --/
theorem dishonest_dealer_profit (real_weight : ℝ) (cost_price : ℝ) 
  (h1 : real_weight > 0) (h2 : cost_price > 0) : 
  let counterfeit_weight := 0.8 * real_weight
  let impure_weight := counterfeit_weight * 1.15
  let selling_price := cost_price * (real_weight / impure_weight)
  let profit := selling_price - cost_price
  profit / cost_price = 0.25 := by sorry

end dishonest_dealer_profit_l2462_246241


namespace floor_greater_than_x_minus_one_l2462_246212

theorem floor_greater_than_x_minus_one (x : ℝ) : ⌊x⌋ > x - 1 := by sorry

end floor_greater_than_x_minus_one_l2462_246212


namespace second_artifact_time_multiple_l2462_246222

/-- Represents the time spent on artifact collection in months -/
structure ArtifactTime where
  research : ℕ
  expedition : ℕ

/-- The total time spent on both artifacts in months -/
def total_time : ℕ := 10 * 12

/-- Time spent on the first artifact -/
def first_artifact : ArtifactTime := { research := 6, expedition := 2 * 12 }

/-- Calculate the total time spent on an artifact -/
def total_artifact_time (a : ArtifactTime) : ℕ := a.research + a.expedition

/-- The multiple of time taken for the second artifact compared to the first -/
def time_multiple : ℚ :=
  (total_time - total_artifact_time first_artifact) / total_artifact_time first_artifact

theorem second_artifact_time_multiple :
  time_multiple = 3 := by sorry

end second_artifact_time_multiple_l2462_246222


namespace largest_divisor_of_difference_of_squares_l2462_246289

theorem largest_divisor_of_difference_of_squares (m n : ℤ) : 
  Odd m → Odd n → n < m → 
  (∃ k : ℤ, m ^ 2 - n ^ 2 = 8 * k) ∧ 
  (∀ d : ℤ, d > 8 → ∃ m' n' : ℤ, Odd m' ∧ Odd n' ∧ n' < m' ∧ ¬(d ∣ (m' ^ 2 - n' ^ 2))) :=
by sorry

end largest_divisor_of_difference_of_squares_l2462_246289


namespace notebook_duration_example_l2462_246295

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and pages used per day. -/
def notebook_duration (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, using 4 pages per day, last for 50 days. -/
theorem notebook_duration_example : notebook_duration 5 40 4 = 50 := by
  sorry

end notebook_duration_example_l2462_246295


namespace binary_multiplication_theorem_l2462_246264

/-- Converts a list of binary digits to its decimal representation -/
def binaryToDecimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its binary representation -/
def decimalToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let product := [true, true, true, false, false, true, true]  -- 1100111₂
  binaryToDecimal a * binaryToDecimal b = binaryToDecimal product := by
  sorry

end binary_multiplication_theorem_l2462_246264


namespace range_of_f_l2462_246286

def f (x : ℤ) : ℤ := x^2 - 1

def domain : Set ℤ := {-1, 0, 1}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0} := by sorry

end range_of_f_l2462_246286


namespace plant_mass_problem_l2462_246231

theorem plant_mass_problem (initial_mass : ℝ) : 
  (((initial_mass * 3 + 4) * 3 + 4) * 3 + 4 = 133) → initial_mass = 3 := by
sorry

end plant_mass_problem_l2462_246231


namespace milkshakes_bought_l2462_246265

def initial_amount : ℕ := 120
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 3
def hamburgers_bought : ℕ := 8
def final_amount : ℕ := 70

theorem milkshakes_bought :
  ∃ (m : ℕ), 
    initial_amount - (hamburger_cost * hamburgers_bought + milkshake_cost * m) = final_amount ∧
    m = 6 := by
  sorry

end milkshakes_bought_l2462_246265


namespace sum_squares_3005_odd_integers_units_digit_l2462_246223

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square (n : ℕ) : ℕ := n * n

def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_squares_3005_odd_integers_units_digit :
  units_digit (List.sum (List.map square (first_n_odd_integers 3005))) = 3 := by
  sorry

end sum_squares_3005_odd_integers_units_digit_l2462_246223


namespace unique_element_quadratic_l2462_246237

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 4 * x + 4 = 0}

-- State the theorem
theorem unique_element_quadratic (a : ℝ) : 
  (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by sorry

end unique_element_quadratic_l2462_246237


namespace cube_root_of_64_l2462_246257

theorem cube_root_of_64 (n : ℕ) (t : ℕ) : t = n * (n - 1) * (n + 1) + n → t = 64 → n = 4 := by
  sorry

end cube_root_of_64_l2462_246257


namespace systematic_sampling_l2462_246288

/-- Systematic sampling problem -/
theorem systematic_sampling
  (population_size : ℕ)
  (sample_size : ℕ)
  (last_sampled : ℕ)
  (h1 : population_size = 8000)
  (h2 : sample_size = 50)
  (h3 : last_sampled = 7894)
  (h4 : last_sampled < population_size) :
  let segment_size := population_size / sample_size
  let first_sampled := last_sampled - (segment_size - 1)
  first_sampled = 735 :=
by sorry

end systematic_sampling_l2462_246288


namespace football_banquet_food_consumption_l2462_246229

theorem football_banquet_food_consumption 
  (max_food_per_guest : ℝ) 
  (min_guests : ℕ) 
  (h1 : max_food_per_guest = 2) 
  (h2 : min_guests = 160) : 
  ∃ (total_food : ℝ), total_food = max_food_per_guest * min_guests ∧ total_food = 320 := by
  sorry

end football_banquet_food_consumption_l2462_246229


namespace olivias_carrots_l2462_246267

theorem olivias_carrots (mom_carrots : ℕ) (good_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : mom_carrots = 14)
  (h2 : good_carrots = 19)
  (h3 : bad_carrots = 15) :
  good_carrots + bad_carrots - mom_carrots = 20 := by
  sorry

end olivias_carrots_l2462_246267


namespace basketball_highlight_film_l2462_246225

theorem basketball_highlight_film (point_guard : ℕ) (shooting_guard : ℕ) (small_forward : ℕ) (power_forward : ℕ) :
  point_guard = 130 →
  shooting_guard = 145 →
  small_forward = 85 →
  power_forward = 60 →
  ∃ (center : ℕ),
    center = 180 ∧
    (point_guard + shooting_guard + small_forward + power_forward + center) / 5 = 120 :=
by sorry

end basketball_highlight_film_l2462_246225


namespace game_probability_difference_l2462_246211

def coin_prob_heads : ℚ := 2/3
def coin_prob_tails : ℚ := 1/3

def game_x_win_prob : ℚ :=
  3 * (coin_prob_heads^2 * coin_prob_tails) + coin_prob_heads^3

def game_y_win_prob : ℚ :=
  4 * (coin_prob_heads^3 * coin_prob_tails + coin_prob_tails^3 * coin_prob_heads) +
  coin_prob_heads^4 + coin_prob_tails^4

theorem game_probability_difference :
  game_x_win_prob - game_y_win_prob = 11/81 :=
sorry

end game_probability_difference_l2462_246211


namespace planar_graph_inequality_l2462_246242

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces

/-- For any planar graph, twice the number of edges is greater than or equal to
    three times the number of faces. -/
theorem planar_graph_inequality (G : PlanarGraph) : 2 * G.E ≥ 3 * G.F := by
  sorry

end planar_graph_inequality_l2462_246242


namespace min_tiles_for_2014_area_l2462_246284

/-- Represents the side length of a square tile in centimeters -/
inductive TileSize
  | Small : TileSize  -- 3 cm
  | Large : TileSize  -- 5 cm

/-- Calculates the area of a square tile given its size -/
def tileArea (size : TileSize) : ℕ :=
  match size with
  | TileSize.Small => 9   -- 3² = 9
  | TileSize.Large => 25  -- 5² = 25

/-- Represents a collection of tiles -/
structure TileCollection where
  smallCount : ℕ
  largeCount : ℕ

/-- Calculates the total area covered by a collection of tiles -/
def totalArea (tiles : TileCollection) : ℕ :=
  tiles.smallCount * tileArea TileSize.Small + tiles.largeCount * tileArea TileSize.Large

/-- Calculates the total number of tiles in a collection -/
def totalTiles (tiles : TileCollection) : ℕ :=
  tiles.smallCount + tiles.largeCount

theorem min_tiles_for_2014_area :
  ∃ (tiles : TileCollection),
    totalArea tiles = 2014 ∧
    (∀ (other : TileCollection), totalArea other = 2014 → totalTiles tiles ≤ totalTiles other) ∧
    totalTiles tiles = 94 :=
  sorry

end min_tiles_for_2014_area_l2462_246284


namespace arithmetic_geometric_sequence_product_l2462_246239

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The main theorem -/
theorem arithmetic_geometric_sequence_product (a : ℕ → ℝ) :
  ArithmeticGeometricSequence a →
  a 1 = 3 →
  a 1 + a 3 + a 5 = 21 →
  a 2 * a 4 = 36 := by
  sorry


end arithmetic_geometric_sequence_product_l2462_246239


namespace cost_of_traveling_specific_roads_l2462_246227

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
def cost_of_traveling_roads (lawn_length lawn_width road_width cost_per_sqm : ℕ) : ℕ :=
  let road1_area := road_width * lawn_width
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_road_area := road1_area + road2_area - intersection_area
  total_road_area * cost_per_sqm

/-- Proves that the cost of traveling two intersecting roads on a specific rectangular lawn is 6500. -/
theorem cost_of_traveling_specific_roads :
  cost_of_traveling_roads 80 60 10 5 = 6500 := by
  sorry

end cost_of_traveling_specific_roads_l2462_246227


namespace range_of_a_l2462_246260

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*a*x + 4 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x y : ℝ, (y + (a-1)*x + 2*a - 1 = 0) ∧ 
  ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ≤ -2 ∨ (1 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l2462_246260


namespace mike_is_18_l2462_246234

-- Define Mike's age and his uncle's age
def mike_age : ℕ := sorry
def uncle_age : ℕ := sorry

-- Define the conditions
axiom age_difference : mike_age = uncle_age - 18
axiom sum_of_ages : mike_age + uncle_age = 54

-- Theorem to prove
theorem mike_is_18 : mike_age = 18 := by sorry

end mike_is_18_l2462_246234


namespace midpoint_of_specific_segment_l2462_246246

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The midpoint of two polar points -/
def polarMidpoint (p1 p2 : PolarPoint) : PolarPoint :=
  sorry

theorem midpoint_of_specific_segment :
  let p1 : PolarPoint := ⟨6, π/6⟩
  let p2 : PolarPoint := ⟨2, -π/6⟩
  let m := polarMidpoint p1 p2
  0 ≤ m.θ ∧ m.θ < 2*π ∧ m.r > 0 ∧ m = ⟨Real.sqrt 13, π/6⟩ := by
  sorry

end midpoint_of_specific_segment_l2462_246246


namespace greatest_number_of_factors_l2462_246291

/-- The greatest number of positive factors for b^n given the conditions -/
def max_factors : ℕ := 561

/-- b is a positive integer less than or equal to 15 -/
def b : ℕ := 12

/-- n is a perfect square less than or equal to 16 -/
def n : ℕ := 16

/-- Theorem stating that max_factors is the greatest number of positive factors of b^n -/
theorem greatest_number_of_factors :
  ∀ (b' n' : ℕ), 
    b' > 0 → b' ≤ 15 → 
    n' > 0 → ∃ (k : ℕ), n' = k^2 → n' ≤ 16 →
    (Nat.factors (b'^n')).length ≤ max_factors :=
by sorry

end greatest_number_of_factors_l2462_246291


namespace sqrt_2_irrational_l2462_246203

theorem sqrt_2_irrational : Irrational (Real.sqrt 2) := by
  sorry

end sqrt_2_irrational_l2462_246203


namespace largest_certain_divisor_l2462_246204

/-- An eight-sided die with numbers 1 through 8 -/
def Die : Finset ℕ := Finset.range 8 

/-- The product of 7 visible numbers on the die -/
def Q (visible : Finset ℕ) : ℕ := 
  Finset.prod visible id

/-- The theorem stating that 192 is the largest number that always divides Q -/
theorem largest_certain_divisor : 
  ∀ visible : Finset ℕ, visible ⊆ Die → visible.card = 7 → 
    (∀ n : ℕ, n > 192 → ∃ visible : Finset ℕ, visible ⊆ Die ∧ visible.card = 7 ∧ ¬(n ∣ Q visible)) ∧
    (∀ visible : Finset ℕ, visible ⊆ Die → visible.card = 7 → 192 ∣ Q visible) :=
by sorry

end largest_certain_divisor_l2462_246204


namespace b_current_age_l2462_246287

/-- Given two people A and B, prove B's current age is 38 years old. -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) → -- A's age in 10 years = 2 * (B's age 10 years ago)
  (a = b + 8) →             -- A is currently 8 years older than B
  b = 38 :=                 -- B's current age is 38
by sorry

end b_current_age_l2462_246287


namespace sequence_has_unique_occurrence_l2462_246249

def is_unique_occurrence (s : ℕ → ℝ) (x : ℝ) : Prop :=
  ∃! n : ℕ, s n = x

theorem sequence_has_unique_occurrence
  (a : ℕ → ℝ)
  (h_inc : ∀ i j : ℕ, i < j → a i < a j)
  (h_bound : ∀ i : ℕ, 0 < a i ∧ a i < 1) :
  ∃ x : ℝ, is_unique_occurrence (λ i => a i / i) x :=
sorry

end sequence_has_unique_occurrence_l2462_246249


namespace round_trip_average_speed_l2462_246281

/-- Calculates the average speed of a round trip given the following conditions:
  * The total distance of the round trip is 4 miles
  * The outbound journey of 2 miles takes 1 hour
  * The return journey of 2 miles is completed at a speed of 6.000000000000002 miles/hour
-/
theorem round_trip_average_speed : 
  let total_distance : ℝ := 4
  let outbound_distance : ℝ := 2
  let outbound_time : ℝ := 1
  let return_speed : ℝ := 6.000000000000002
  let return_time : ℝ := outbound_distance / return_speed
  let total_time : ℝ := outbound_time + return_time
  total_distance / total_time = 3 := by
sorry

end round_trip_average_speed_l2462_246281


namespace factor_problem_l2462_246245

theorem factor_problem (x : ℝ) (f : ℝ) : 
  x = 6 → (2 * x + 9) * f = 63 → f = 3 := by
  sorry

end factor_problem_l2462_246245


namespace circle_center_proof_l2462_246208

theorem circle_center_proof (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) :
  -- The circle passes through (1,0)
  (1, 0) ∈ C →
  -- The circle is tangent to y = x^2 at (2,4)
  (2, 4) ∈ C →
  (∀ (x y : ℝ), (x, y) ∈ C → y ≠ x^2 ∨ (x = 2 ∧ y = 4)) →
  -- The circle is tangent to the x-axis
  (∃ (x : ℝ), (x, 0) ∈ C ∧ ∀ (y : ℝ), y ≠ 0 → (x, y) ∉ C) →
  -- C is a circle with center 'center'
  (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = (1 - center.1)^2 + center.2^2) →
  -- The center is (178/15, 53/15)
  center = (178/15, 53/15) := by
sorry

end circle_center_proof_l2462_246208


namespace southbound_cyclist_speed_l2462_246285

/-- 
Given two cyclists starting from the same point and traveling in opposite directions,
with one cyclist traveling north at 10 km/h, prove that the speed of the southbound
cyclist is 15 km/h if they are 50 km apart after 2 hours.
-/
theorem southbound_cyclist_speed 
  (north_speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ) 
  (h1 : north_speed = 10) 
  (h2 : time = 2) 
  (h3 : distance = 50) : 
  ∃ south_speed : ℝ, south_speed = 15 ∧ (north_speed + south_speed) * time = distance :=
sorry

end southbound_cyclist_speed_l2462_246285


namespace sally_picked_42_peaches_l2462_246271

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked (initial current : ℕ) : ℕ :=
  current - initial

/-- Proof that Sally picked 42 peaches from the orchard -/
theorem sally_picked_42_peaches (initial current : ℕ) 
  (h1 : initial = 13) 
  (h2 : current = 55) : 
  peaches_picked initial current = 42 := by
  sorry

end sally_picked_42_peaches_l2462_246271


namespace cooler_capacity_ratio_l2462_246228

/-- Given three coolers with specific capacities, prove the ratio of the third to the second is 1/2. -/
theorem cooler_capacity_ratio :
  ∀ (c₁ c₂ c₃ : ℝ),
  c₁ = 100 →
  c₂ = c₁ + 0.5 * c₁ →
  c₁ + c₂ + c₃ = 325 →
  c₃ / c₂ = 1 / 2 := by
sorry

end cooler_capacity_ratio_l2462_246228


namespace min_value_expression_l2462_246209

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 2) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 2 ∧
  (∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 2 → a' + b' = 2 →
    (a' * c' / b' + c' / (a' * b') - c' / 2 + Real.sqrt 5 / (c' - 2) ≥ Real.sqrt 10 + Real.sqrt 5)) ∧
  (x * (2 + Real.sqrt 2) / y + (2 + Real.sqrt 2) / (x * y) - (2 + Real.sqrt 2) / 2 + Real.sqrt 5 / Real.sqrt 2 = Real.sqrt 10 + Real.sqrt 5) :=
by sorry

end min_value_expression_l2462_246209


namespace smallest_advantageous_discount_l2462_246205

theorem smallest_advantageous_discount : ∃ n : ℕ,
  (n : ℝ) > 0 ∧
  (∀ m : ℕ, m < n →
    (1 - m / 100 : ℝ) > (1 - 0.20)^2 ∨
    (1 - m / 100 : ℝ) > (1 - 0.13)^3 ∨
    (1 - m / 100 : ℝ) > (1 - 0.30) * (1 - 0.10)) ∧
  (1 - n / 100 : ℝ) ≤ (1 - 0.20)^2 ∧
  (1 - n / 100 : ℝ) ≤ (1 - 0.13)^3 ∧
  (1 - n / 100 : ℝ) ≤ (1 - 0.30) * (1 - 0.10) :=
by
  sorry

end smallest_advantageous_discount_l2462_246205


namespace julie_delivered_600_newspapers_l2462_246262

/-- Represents Julie's earnings and expenses --/
structure JulieFinances where
  saved : ℕ
  bikeCost : ℕ
  lawnsMowed : ℕ
  lawnRate : ℕ
  dogsWalked : ℕ
  dogRate : ℕ
  newspaperRate : ℕ
  leftover : ℕ

/-- Calculates the number of newspapers Julie delivered --/
def newspapersDelivered (j : JulieFinances) : ℕ :=
  ((j.bikeCost + j.leftover) - (j.saved + j.lawnsMowed * j.lawnRate + j.dogsWalked * j.dogRate)) / j.newspaperRate

/-- Theorem stating that Julie delivered 600 newspapers --/
theorem julie_delivered_600_newspapers :
  let j : JulieFinances := {
    saved := 1500,
    bikeCost := 2345,
    lawnsMowed := 20,
    lawnRate := 20,
    dogsWalked := 24,
    dogRate := 15,
    newspaperRate := 40,  -- in cents
    leftover := 155
  }
  newspapersDelivered j = 600 := by sorry


end julie_delivered_600_newspapers_l2462_246262


namespace monotone_cubic_function_condition_l2462_246200

/-- Given a function f(x) = -x^3 + bx that is monotonically increasing on (0, 1),
    prove that b ≥ 3 -/
theorem monotone_cubic_function_condition (b : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, Monotone (fun x => -x^3 + b*x)) →
  b ≥ 3 := by
  sorry

end monotone_cubic_function_condition_l2462_246200


namespace cubic_function_unique_negative_zero_l2462_246294

/-- Given a cubic function f(x) = ax³ - 3x² + 1 with a unique zero point x₀ < 0, prove that a > 2 -/
theorem cubic_function_unique_negative_zero (a : ℝ) :
  (∃! x₀ : ℝ, a * x₀^3 - 3 * x₀^2 + 1 = 0) →
  (∀ x₀ : ℝ, a * x₀^3 - 3 * x₀^2 + 1 = 0 → x₀ < 0) →
  a > 2 := by
  sorry

end cubic_function_unique_negative_zero_l2462_246294


namespace x4_coefficient_zero_l2462_246240

theorem x4_coefficient_zero (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, (x^2 + a*x + 1) * (-6*x^3) = -6*x^5 + f x * x^4 + -6*x^3) ↔ a = 0 := by
  sorry

end x4_coefficient_zero_l2462_246240


namespace product_modulo_l2462_246219

theorem product_modulo : (2345 * 1554) % 700 = 630 := by
  sorry

end product_modulo_l2462_246219


namespace largest_product_sum_of_digits_l2462_246226

def is_single_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_sum_of_digits :
  ∃ (n d e : ℕ),
    is_single_digit d ∧
    is_prime d ∧
    is_single_digit e ∧
    is_odd e ∧
    ¬is_prime e ∧
    n = d * e * (d^2 + e) ∧
    (∀ (m : ℕ), m = d' * e' * (d'^2 + e') →
      is_single_digit d' →
      is_prime d' →
      is_single_digit e' →
      is_odd e' →
      ¬is_prime e' →
      m ≤ n) ∧
    sum_of_digits n = 9 :=
  sorry

end largest_product_sum_of_digits_l2462_246226


namespace chest_to_treadmill_ratio_l2462_246210

/-- The price of the treadmill in dollars -/
def treadmill_price : ℝ := 100

/-- The price of the television in dollars -/
def tv_price : ℝ := 3 * treadmill_price

/-- The total sum of money from the sale in dollars -/
def total_sum : ℝ := 600

/-- The price of the chest of drawers in dollars -/
def chest_price : ℝ := total_sum - treadmill_price - tv_price

/-- The theorem stating that the ratio of the chest price to the treadmill price is 2:1 -/
theorem chest_to_treadmill_ratio :
  chest_price / treadmill_price = 2 := by sorry

end chest_to_treadmill_ratio_l2462_246210


namespace age_multiple_problem_l2462_246220

theorem age_multiple_problem (a b : ℕ) (m : ℚ) : 
  a = b + 5 →
  a + b = 13 →
  m * (a + 7 : ℚ) = 4 * (b + 7 : ℚ) →
  m = 2.75 := by
  sorry

end age_multiple_problem_l2462_246220


namespace original_number_proof_l2462_246235

theorem original_number_proof (x : ℝ) : x * 1.5 = 120 → x = 80 := by
  sorry

end original_number_proof_l2462_246235


namespace songs_in_playlists_l2462_246218

theorem songs_in_playlists (n : ℕ) :
  ∃ (k : ℕ), n = 12 + 9 * k ↔ ∃ (m : ℕ), n = 9 * m + 3 :=
by sorry

end songs_in_playlists_l2462_246218


namespace stratified_sampling_theorem_l2462_246280

/-- Represents the sample size for each category of students -/
structure SampleSizes where
  junior : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Calculates the stratified sample sizes given the total population, category populations, and total sample size -/
def calculateSampleSizes (totalPopulation : ℕ) (juniorPopulation : ℕ) (undergradPopulation : ℕ) (sampleSize : ℕ) : SampleSizes :=
  let juniorSample := (juniorPopulation * sampleSize) / totalPopulation
  let undergradSample := (undergradPopulation * sampleSize) / totalPopulation
  let gradSample := sampleSize - juniorSample - undergradSample
  { junior := juniorSample,
    undergraduate := undergradSample,
    graduate := gradSample }

theorem stratified_sampling_theorem (totalPopulation : ℕ) (juniorPopulation : ℕ) (undergradPopulation : ℕ) (sampleSize : ℕ)
    (h1 : totalPopulation = 5600)
    (h2 : juniorPopulation = 1300)
    (h3 : undergradPopulation = 3000)
    (h4 : sampleSize = 280) :
    calculateSampleSizes totalPopulation juniorPopulation undergradPopulation sampleSize =
    { junior := 65, undergraduate := 150, graduate := 65 } := by
  sorry

#check stratified_sampling_theorem

end stratified_sampling_theorem_l2462_246280


namespace circle_to_ellipse_l2462_246213

/-- If z is a complex number tracing a circle centered at the origin with radius 3,
    then z + 1/z traces an ellipse. -/
theorem circle_to_ellipse (z : ℂ) (h : ∀ θ : ℝ, z = 3 * Complex.exp (Complex.I * θ)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ∀ θ : ℝ, ∃ x y : ℝ, 
    z + 1/z = Complex.mk x y ∧ 
    (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

end circle_to_ellipse_l2462_246213


namespace square_diff_div_four_xy_eq_one_l2462_246269

theorem square_diff_div_four_xy_eq_one (x y : ℝ) (h : x * y ≠ 0) :
  ((x + y)^2 - (x - y)^2) / (4 * x * y) = 1 := by
  sorry

end square_diff_div_four_xy_eq_one_l2462_246269


namespace freshman_sophomore_percentage_l2462_246276

theorem freshman_sophomore_percentage
  (total_students : ℕ)
  (pet_ownership_ratio : ℚ)
  (non_pet_owners : ℕ)
  (h1 : total_students = 400)
  (h2 : pet_ownership_ratio = 1/5)
  (h3 : non_pet_owners = 160) :
  (↑(total_students - non_pet_owners) / (1 - pet_ownership_ratio)) / total_students = 1/2 :=
sorry

end freshman_sophomore_percentage_l2462_246276


namespace bus_stop_problem_l2462_246290

/-- The number of children who got on the bus at the bus stop -/
def children_got_on : ℕ := sorry

/-- The initial number of children on the bus -/
def initial_children : ℕ := 22

/-- The number of children who got off the bus at the bus stop -/
def children_got_off : ℕ := 60

/-- The final number of children on the bus after the bus stop -/
def final_children : ℕ := 2

theorem bus_stop_problem :
  initial_children - children_got_off + children_got_on = final_children ∧
  children_got_on = 40 := by sorry

end bus_stop_problem_l2462_246290


namespace mappings_count_l2462_246297

/-- Set A with elements from 1 to 15 -/
def A : Finset ℕ := Finset.range 15

/-- Set B with elements 0 and 1 -/
def B : Finset ℕ := {0, 1}

/-- The number of mappings from A to B where 1 is the image of at least two elements of A -/
def num_mappings : ℕ := 2^15 - (1 + 15)

/-- Theorem stating that the number of mappings from A to B where 1 is the image of at least two elements of A is 32752 -/
theorem mappings_count : num_mappings = 32752 := by
  sorry

#eval num_mappings

end mappings_count_l2462_246297


namespace coin_tosses_properties_l2462_246272

/-- Two independent coin tosses where A is "first coin is heads" and B is "second coin is tails" -/
structure CoinTosses where
  /-- Probability of event A (first coin is heads) -/
  prob_A : ℝ
  /-- Probability of event B (second coin is tails) -/
  prob_B : ℝ
  /-- A and B are independent events -/
  independent : Prop
  /-- Both coins are fair -/
  fair_coins : prob_A = 1/2 ∧ prob_B = 1/2

/-- Properties of the coin tosses -/
theorem coin_tosses_properties (ct : CoinTosses) :
  ct.independent ∧ 
  (1 - (1 - ct.prob_A) * (1 - ct.prob_B) = 3/4) ∧
  ct.prob_A = ct.prob_B :=
sorry

end coin_tosses_properties_l2462_246272


namespace largest_prime_divisor_of_39_squared_plus_52_squared_l2462_246224

theorem largest_prime_divisor_of_39_squared_plus_52_squared : 
  (Nat.factors (39^2 + 52^2)).maximum? = some 13 := by
  sorry

end largest_prime_divisor_of_39_squared_plus_52_squared_l2462_246224


namespace hyperbola_asymptotes_l2462_246261

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the perimeter of the quadrilateral formed by lines parallel to its asymptotes
    drawn from its left and right foci is 8b, then the equation of its asymptotes is y = ±x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, 2 * b = Real.sqrt ((b^2 * c^2) / a^2 + c^2)) →
  (∀ x y : ℝ, (y = x ∨ y = -x) ↔ y^2 = x^2) :=
sorry

end hyperbola_asymptotes_l2462_246261


namespace molecular_weight_7_moles_KBrO3_l2462_246238

/-- The atomic weight of potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of oxygen atoms in KBrO3 -/
def num_oxygen_atoms : ℕ := 3

/-- The molecular weight of one mole of KBrO3 in g/mol -/
def molecular_weight_KBrO3 : ℝ :=
  atomic_weight_K + atomic_weight_Br + (atomic_weight_O * num_oxygen_atoms)

/-- The number of moles of KBrO3 -/
def num_moles : ℕ := 7

/-- Theorem: The molecular weight of 7 moles of KBrO3 is 1169.00 grams -/
theorem molecular_weight_7_moles_KBrO3 :
  molecular_weight_KBrO3 * num_moles = 1169.00 := by
  sorry

end molecular_weight_7_moles_KBrO3_l2462_246238
