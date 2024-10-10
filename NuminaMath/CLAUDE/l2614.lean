import Mathlib

namespace print_shop_charges_l2614_261475

/-- 
Given:
- Print shop X charges $1.25 per color copy
- Print shop Y charges $60 more than print shop X for 40 color copies

Prove that print shop Y charges $2.75 per color copy
-/
theorem print_shop_charges (x y : ℝ) : 
  x = 1.25 → 
  40 * y = 40 * x + 60 → 
  y = 2.75 := by
  sorry

end print_shop_charges_l2614_261475


namespace simplify_sqrt_sum_l2614_261437

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 3) + Real.sqrt (8 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end simplify_sqrt_sum_l2614_261437


namespace triangular_array_sum_l2614_261431

/-- Represents the sum of numbers in the nth row of the triangular array. -/
def f (n : ℕ) : ℕ :=
  4 * 2^(n-1) - 2*n

/-- The triangular array starts with 1 on top and increases by 1 for each subsequent outer number.
    Interior numbers are obtained by adding two adjacent numbers from the previous row. -/
theorem triangular_array_sum (n : ℕ) (h : n > 0) :
  f n = 4 * 2^(n-1) - 2*n :=
sorry

end triangular_array_sum_l2614_261431


namespace pie_distribution_l2614_261463

/-- Given a pie with 48 slices, prove that after distributing specific fractions, 2 slices remain -/
theorem pie_distribution (total_slices : ℕ) (joe_fraction darcy_fraction carl_fraction emily_fraction frank_percent : ℚ) : 
  total_slices = 48 →
  joe_fraction = 1/3 →
  darcy_fraction = 1/4 →
  carl_fraction = 1/6 →
  emily_fraction = 1/8 →
  frank_percent = 10/100 →
  total_slices - (total_slices * joe_fraction).floor - (total_slices * darcy_fraction).floor - 
  (total_slices * carl_fraction).floor - (total_slices * emily_fraction).floor - 
  (total_slices * frank_percent).floor = 2 := by
sorry

end pie_distribution_l2614_261463


namespace average_marks_proof_l2614_261426

theorem average_marks_proof (total_subjects : Nat) 
                             (avg_five_subjects : ℝ) 
                             (sixth_subject_marks : ℝ) : 
  total_subjects = 6 →
  avg_five_subjects = 74 →
  sixth_subject_marks = 50 →
  ((avg_five_subjects * 5 + sixth_subject_marks) / total_subjects : ℝ) = 70 := by
  sorry

end average_marks_proof_l2614_261426


namespace train_speed_calculation_l2614_261407

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation 
  (train_length : Real) 
  (bridge_length : Real) 
  (crossing_time : Real) 
  (h1 : train_length = 100) 
  (h2 : bridge_length = 275) 
  (h3 : crossing_time = 30) : 
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_calculation_l2614_261407


namespace rectangular_solid_width_l2614_261476

theorem rectangular_solid_width (length depth surface_area : ℝ) : 
  length = 10 →
  depth = 6 →
  surface_area = 408 →
  surface_area = 2 * length * width + 2 * length * depth + 2 * width * depth →
  width = 9 :=
by
  sorry

end rectangular_solid_width_l2614_261476


namespace polygon_sides_count_l2614_261482

theorem polygon_sides_count (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 2 * 360 ↔ n = 6 := by sorry

end polygon_sides_count_l2614_261482


namespace five_g_base_stations_scientific_notation_l2614_261445

theorem five_g_base_stations_scientific_notation :
  (819000 : ℝ) = 8.19 * (10 ^ 5) := by sorry

end five_g_base_stations_scientific_notation_l2614_261445


namespace unique_four_digit_reverse_l2614_261449

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem unique_four_digit_reverse : ∃! n : ℕ, is_four_digit n ∧ n + 8802 = reverse_digits n :=
  sorry

end unique_four_digit_reverse_l2614_261449


namespace angle_tangent_product_l2614_261454

theorem angle_tangent_product (A C : ℝ) (h : 5 * (Real.cos A + Real.cos C) + 4 * (Real.cos A * Real.cos C + 1) = 0) :
  Real.tan (A / 2) * Real.tan (C / 2) = 3 := by
sorry

end angle_tangent_product_l2614_261454


namespace max_video_game_hours_l2614_261427

/-- Proves that given the conditions of Max's video game playing schedule,
    he must have played 2 hours on Wednesday. -/
theorem max_video_game_hours :
  ∀ x : ℝ,
  (x + x + (x + 3)) / 3 = 3 →
  x = 2 := by
sorry

end max_video_game_hours_l2614_261427


namespace minimum_living_allowance_growth_l2614_261436

/-- The average annual growth rate of the minimum living allowance -/
def average_growth_rate : ℝ := 0.3

/-- The initial minimum living allowance in yuan -/
def initial_allowance : ℝ := 200

/-- The final minimum living allowance in yuan -/
def final_allowance : ℝ := 338

/-- The number of years -/
def years : ℕ := 2

theorem minimum_living_allowance_growth :
  initial_allowance * (1 + average_growth_rate) ^ years = final_allowance := by
  sorry

end minimum_living_allowance_growth_l2614_261436


namespace circle_passes_through_focus_l2614_261444

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 8 * p.1

-- Define the line
def line (x : ℝ) : Prop := x + 2 = 0

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency of a circle to a line
def tangent_to_line (c : Circle) : Prop :=
  c.radius = |c.center.1 + 2|

-- Main theorem
theorem circle_passes_through_focus :
  ∀ c : Circle,
  parabola c.center →
  tangent_to_line c →
  c.center.1^2 + c.center.2^2 = (2 - c.center.1)^2 + c.center.2^2 :=
sorry

end circle_passes_through_focus_l2614_261444


namespace percentage_of_x_l2614_261435

theorem percentage_of_x (x y : ℝ) (h1 : x / y = 4) (h2 : y ≠ 0) :
  (2 * x - y) / x * 100 = 175 := by
  sorry

end percentage_of_x_l2614_261435


namespace hyperbola_property_l2614_261413

/-- The hyperbola with equation x²/9 - y²/4 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 9) - (p.2^2 / 4) = 1}

/-- Left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- Right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- A point on the right branch of the hyperbola -/
def A : ℝ × ℝ := sorry

/-- Origin point -/
def O : ℝ × ℝ := (0, 0)

/-- Point P such that 2 * OP = OA + OF₁ -/
def P : ℝ × ℝ := sorry

/-- Point Q such that 2 * OQ = OA + OF₂ -/
def Q : ℝ × ℝ := sorry

theorem hyperbola_property (h₁ : A ∈ Hyperbola)
    (h₂ : 2 • (P - O) = (A - O) + (F₁ - O))
    (h₃ : 2 • (Q - O) = (A - O) + (F₂ - O)) :
  ‖Q - O‖ - ‖P - O‖ = 3 := by sorry

end hyperbola_property_l2614_261413


namespace monitor_height_l2614_261499

theorem monitor_height 
  (width : ℝ) 
  (pixel_density : ℝ) 
  (total_pixels : ℝ) 
  (h1 : width = 21)
  (h2 : pixel_density = 100)
  (h3 : total_pixels = 2520000) :
  (total_pixels / (width * pixel_density)) / pixel_density = 12 := by
  sorry

end monitor_height_l2614_261499


namespace connie_additional_money_l2614_261480

def additional_money_needed (savings : ℚ) (watch_cost : ℚ) (strap_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost_before_tax := watch_cost + strap_cost
  let tax_amount := total_cost_before_tax * tax_rate
  let total_cost_with_tax := total_cost_before_tax + tax_amount
  total_cost_with_tax - savings

theorem connie_additional_money :
  additional_money_needed 39 55 15 (8/100) = 366/10 := by
  sorry

end connie_additional_money_l2614_261480


namespace roots_magnitude_l2614_261451

theorem roots_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ ≠ r₂) →  -- r₁ and r₂ are distinct
  (r₁^2 + p*r₁ + 12 = 0) →  -- r₁ is a root of the equation
  (r₂^2 + p*r₂ + 12 = 0) →  -- r₂ is a root of the equation
  (abs r₁ > 3 ∨ abs r₂ > 3) := by
sorry

end roots_magnitude_l2614_261451


namespace larger_integer_problem_l2614_261434

theorem larger_integer_problem (x y : ℤ) 
  (h1 : y = 4 * x) 
  (h2 : (x + 12) / y = 1 / 2) : 
  y = 48 := by sorry

end larger_integer_problem_l2614_261434


namespace ellipse_foci_ratio_l2614_261474

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Right-angled triangle formed by P, F₁, and F₂ -/
def is_right_triangle (P F₁ F₂ : ℝ × ℝ) : Prop := sorry

theorem ellipse_foci_ratio :
  is_on_ellipse P.1 P.2 →
  is_right_triangle P F₁ F₂ →
  distance P F₁ > distance P F₂ →
  (distance P F₁ / distance P F₂ = 7/2) ∨ (distance P F₁ / distance P F₂ = 2) := by
  sorry

end ellipse_foci_ratio_l2614_261474


namespace parallel_vectors_x_value_l2614_261468

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ a + b x = k • (2 • a - b x)) → x = -4 :=
by sorry

end parallel_vectors_x_value_l2614_261468


namespace number_problem_l2614_261495

theorem number_problem : ∃ x : ℝ, (x / 6) * 12 = 13 ∧ x = 6.5 := by
  sorry

end number_problem_l2614_261495


namespace range_of_ab_l2614_261471

-- Define the polynomial
def P (a b x : ℝ) : ℝ := (x^2 - a*x + 1) * (x^2 - b*x + 1)

-- State the theorem
theorem range_of_ab (a b : ℝ) (q : ℝ) (h_q : q ∈ Set.Icc (1/3) 2) 
  (h_roots : ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (P a b r₁ = 0 ∧ P a b r₂ = 0 ∧ P a b r₃ = 0 ∧ P a b r₄ = 0) ∧ 
    (∃ (m : ℝ), r₁ = m ∧ r₂ = m*q ∧ r₃ = m*q^2 ∧ r₄ = m*q^3)) :
  a * b ∈ Set.Icc 4 (112/9) :=
sorry

end range_of_ab_l2614_261471


namespace parabola_intersection_l2614_261402

theorem parabola_intersection :
  let f (x : ℝ) := 4 * x^2 + 5 * x - 6
  let g (x : ℝ) := x^2 + 14
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
    f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
    x₁ = -4 ∧ x₂ = 5/3 ∧
    f x₁ = 38 ∧ f x₂ = 121/9 ∧
    ∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂ :=
by sorry

end parabola_intersection_l2614_261402


namespace intersection_distance_l2614_261465

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem intersection_distance : 
  ∃ (M N : ℝ × ℝ),
    (parabola M.1 M.2) ∧ 
    (parabola N.1 N.2) ∧
    (line M.1 M.2) ∧ 
    (line N.1 N.2) ∧
    (line focus.1 focus.2) ∧
    (M ≠ N) ∧
    (Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 8) :=
sorry

end intersection_distance_l2614_261465


namespace yulia_lemonade_expenses_l2614_261479

/-- Represents the financial data for Yulia's earnings --/
structure YuliaFinances where
  net_profit : ℝ
  lemonade_revenue : ℝ
  babysitting_earnings : ℝ

/-- Calculates the expenses for operating the lemonade stand --/
def lemonade_expenses (finances : YuliaFinances) : ℝ :=
  finances.lemonade_revenue + finances.babysitting_earnings - finances.net_profit

/-- Theorem stating that Yulia's lemonade stand expenses are $34 --/
theorem yulia_lemonade_expenses :
  let finances : YuliaFinances := {
    net_profit := 44,
    lemonade_revenue := 47,
    babysitting_earnings := 31
  }
  lemonade_expenses finances = 34 := by
  sorry

end yulia_lemonade_expenses_l2614_261479


namespace thirty_sided_polygon_diagonals_l2614_261490

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals : num_diagonals 30 = 405 := by
  sorry

end thirty_sided_polygon_diagonals_l2614_261490


namespace find_divisor_l2614_261418

theorem find_divisor (N D : ℕ) (h1 : N = D * 8) (h2 : N % 5 = 4) : D = 3 := by
  sorry

end find_divisor_l2614_261418


namespace cubic_inequality_solution_l2614_261440

def f (x : ℝ) := x^3 + 5*x^2 + 8*x + 4

theorem cubic_inequality_solution :
  {x : ℝ | f x ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} := by sorry

end cubic_inequality_solution_l2614_261440


namespace quadratic_roots_difference_l2614_261462

theorem quadratic_roots_difference (a b c : ℝ) (r₁ r₂ : ℝ) : 
  a * r₁^2 + b * r₁ + c = 0 →
  a * r₂^2 + b * r₂ + c = 0 →
  a = 1 →
  b = -8 →
  c = 15 →
  r₁ + r₂ = 8 →
  ∃ n : ℤ, (r₁ + r₂ : ℝ) = n^2 →
  r₁ - r₂ = 2 :=
by sorry

end quadratic_roots_difference_l2614_261462


namespace sum_of_coefficients_l2614_261416

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 65 / 3) + (5 / 3))

theorem sum_of_coefficients (a b c : ℕ+) : 
  y^120 = 3*y^117 + 17*y^114 + 13*y^112 - y^60 + (a:ℝ)*y^55 + (b:ℝ)*y^53 + (c:ℝ)*y^50 →
  a + b + c = 131 := by
  sorry

end sum_of_coefficients_l2614_261416


namespace cubic_root_difference_l2614_261442

/-- The cubic equation x³ - px² + (p² - 1)/4x = 0 has a difference of 1 between its largest and smallest roots -/
theorem cubic_root_difference (p : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - p*x^2 + (p^2 - 1)/4*x
  let roots := {x : ℝ | f x = 0}
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ ∀ c ∈ roots, a ≤ c ∧ c ≤ b ∧ b - a = 1 :=
sorry

end cubic_root_difference_l2614_261442


namespace bobbys_candy_problem_l2614_261424

/-- The problem of Bobby's candy consumption -/
theorem bobbys_candy_problem (initial_candy : ℕ) : 
  initial_candy + 42 = 70 → initial_candy = 28 := by
  sorry

end bobbys_candy_problem_l2614_261424


namespace f_of_3_equals_9_l2614_261423

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_of_3_equals_9 : f 3 = 9 := by
  sorry

end f_of_3_equals_9_l2614_261423


namespace imaginary_part_of_z_l2614_261469

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - 3 * Complex.I) :
  z.im = 5/2 := by sorry

end imaginary_part_of_z_l2614_261469


namespace star_polygon_interior_angles_sum_l2614_261401

/-- A star polygon with n angles -/
structure StarPolygon where
  n : ℕ
  h_n : n ≥ 5

/-- The sum of interior angles of a star polygon -/
def sum_interior_angles (sp : StarPolygon) : ℝ :=
  180 * (sp.n - 4)

/-- Theorem: The sum of interior angles of a star polygon is 180° * (n - 4) -/
theorem star_polygon_interior_angles_sum (sp : StarPolygon) :
  sum_interior_angles sp = 180 * (sp.n - 4) := by
  sorry

end star_polygon_interior_angles_sum_l2614_261401


namespace no_valid_acute_triangle_l2614_261464

def is_valid_angle (α : ℕ) : Prop :=
  α % 10 = 0 ∧ α ≠ 30 ∧ α ≠ 60 ∧ α > 0 ∧ α < 90

def is_acute_triangle (α β γ : ℕ) : Prop :=
  α + β + γ = 180 ∧ α < 90 ∧ β < 90 ∧ γ < 90

theorem no_valid_acute_triangle :
  ¬ ∃ (α β γ : ℕ), is_valid_angle α ∧ is_valid_angle β ∧ is_valid_angle γ ∧
  is_acute_triangle α β γ ∧ α ≠ β ∧ β ≠ γ ∧ α ≠ γ :=
sorry

end no_valid_acute_triangle_l2614_261464


namespace tan_equality_periodic_l2614_261430

theorem tan_equality_periodic (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (243 * π / 180) → n = 63 := by
  sorry

end tan_equality_periodic_l2614_261430


namespace polynomial_factor_l2614_261448

-- Define the polynomial
def f (b : ℝ) (x : ℝ) : ℝ := 3 * x^3 + b * x + 12

-- Define the quadratic factor
def g (p : ℝ) (x : ℝ) : ℝ := x^2 + p * x + 2

-- Theorem statement
theorem polynomial_factor (b : ℝ) :
  (∃ p : ℝ, ∀ x : ℝ, ∃ k : ℝ, f b x = g p x * (3 * x + 6)) →
  b = -6 := by sorry

end polynomial_factor_l2614_261448


namespace distribute_five_balls_four_boxes_l2614_261441

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end distribute_five_balls_four_boxes_l2614_261441


namespace smallest_perimeter_triangle_l2614_261496

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle formed by two rays from a vertex -/
structure Angle where
  vertex : Point
  ray1 : Point
  ray2 : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line passing through a point -/
structure Line where
  point : Point
  direction : ℝ

/-- Checks if a point is inside an angle -/
def isPointInsideAngle (p : Point) (a : Angle) : Prop := sorry

/-- Finds the larger inscribed circle passing through a point in an angle -/
def largerInscribedCircle (p : Point) (a : Angle) : Circle := sorry

/-- Checks if a line is tangent to a circle at a point -/
def isTangentLine (l : Line) (c : Circle) (p : Point) : Prop := sorry

/-- Calculates the perimeter of a triangle formed by a line intersecting an angle -/
def trianglePerimeter (l : Line) (a : Angle) : ℝ := sorry

/-- The main theorem -/
theorem smallest_perimeter_triangle 
  (M : Point) (KAL : Angle) 
  (h_inside : isPointInsideAngle M KAL) :
  let S := largerInscribedCircle M KAL
  let tangent_line := Line.mk M (sorry : ℝ)  -- Direction that makes it tangent
  ∀ (l : Line), 
    l.point = M → 
    isTangentLine tangent_line S M → 
    trianglePerimeter l KAL ≥ trianglePerimeter tangent_line KAL :=
by sorry

end smallest_perimeter_triangle_l2614_261496


namespace bicycle_problem_l2614_261460

theorem bicycle_problem (total_distance : ℝ) (walking_speed : ℝ) (cycling_speed : ℝ) 
  (h1 : total_distance = 20)
  (h2 : walking_speed = 4)
  (h3 : cycling_speed = 20) :
  ∃ (x : ℝ) (t : ℝ),
    0 < x ∧ x < total_distance ∧
    (x / cycling_speed + (total_distance - x) / walking_speed = 
     x / walking_speed + (total_distance - x) / cycling_speed) ∧
    x = 10 ∧
    t = 3 ∧
    t = x / cycling_speed + (total_distance - x) / walking_speed :=
by sorry

end bicycle_problem_l2614_261460


namespace factorization_of_3m_squared_minus_12_l2614_261421

theorem factorization_of_3m_squared_minus_12 (m : ℝ) : 3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := by
  sorry

end factorization_of_3m_squared_minus_12_l2614_261421


namespace vertex_of_quadratic_l2614_261409

/-- The quadratic function f(x) = (x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := (x - 3)^2 + 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-3)^2 + 1 is at the point (3,1) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
sorry

end vertex_of_quadratic_l2614_261409


namespace neighbor_cans_is_46_l2614_261411

/-- Represents the recycling problem Collin faces --/
structure RecyclingProblem where
  /-- The amount earned per aluminum can in dollars --/
  earnings_per_can : ℚ
  /-- The number of cans found at home --/
  cans_at_home : ℕ
  /-- The factor by which the number of cans at grandparents' house exceeds those at home --/
  grandparents_factor : ℕ
  /-- The number of cans brought by dad from the office --/
  cans_from_office : ℕ
  /-- The amount Collin has to put into savings in dollars --/
  savings_amount : ℚ

/-- Calculates the number of cans Collin's neighbor gave him --/
def neighbor_cans (p : RecyclingProblem) : ℕ :=
  sorry

/-- Theorem stating that the number of cans Collin's neighbor gave him is 46 --/
theorem neighbor_cans_is_46 (p : RecyclingProblem)
  (h1 : p.earnings_per_can = 1/4)
  (h2 : p.cans_at_home = 12)
  (h3 : p.grandparents_factor = 3)
  (h4 : p.cans_from_office = 250)
  (h5 : p.savings_amount = 43) :
  neighbor_cans p = 46 :=
  sorry

end neighbor_cans_is_46_l2614_261411


namespace binomial_12_choose_6_l2614_261447

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end binomial_12_choose_6_l2614_261447


namespace fraction_equality_l2614_261414

theorem fraction_equality : (1632^2 - 1625^2) / (1645^2 - 1612^2) = 7/33 := by
  sorry

end fraction_equality_l2614_261414


namespace part_probabilities_l2614_261446

/-- Given two machines producing parts with known quantities of standard parts,
    this theorem proves the probabilities of selecting a standard part overall
    and conditionally based on which machine produced it. -/
theorem part_probabilities
  (total_parts_1 : ℕ) (standard_parts_1 : ℕ)
  (total_parts_2 : ℕ) (standard_parts_2 : ℕ)
  (h1 : total_parts_1 = 200)
  (h2 : standard_parts_1 = 190)
  (h3 : total_parts_2 = 300)
  (h4 : standard_parts_2 = 280) :
  let total_parts := total_parts_1 + total_parts_2
  let total_standard := standard_parts_1 + standard_parts_2
  let p_A := total_standard / total_parts
  let p_A_given_B := standard_parts_1 / total_parts_1
  let p_A_given_not_B := standard_parts_2 / total_parts_2
  p_A = 47/50 ∧ p_A_given_B = 19/20 ∧ p_A_given_not_B = 14/15 :=
by sorry

end part_probabilities_l2614_261446


namespace unique_solution_for_all_y_l2614_261422

theorem unique_solution_for_all_y :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7.5 = 0 :=
by
  sorry

end unique_solution_for_all_y_l2614_261422


namespace angle_C_in_similar_triangles_l2614_261477

-- Define the triangles and their properties
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180

-- Define the similarity relation
def similar (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- Theorem statement
theorem angle_C_in_similar_triangles (ABC DEF : Triangle) 
  (h1 : similar ABC DEF) (h2 : ABC.A = 30) (h3 : DEF.B = 30) : ABC.C = 120 := by
  sorry


end angle_C_in_similar_triangles_l2614_261477


namespace expression_value_l2614_261425

theorem expression_value (x y z : ℝ) 
  (eq1 : 2*x - 3*y - z = 0)
  (eq2 : x + 3*y - 14*z = 0)
  (z_neq_zero : z ≠ 0) :
  (x^2 + 3*x*y) / (y^2 + z^2) = 7 := by
  sorry

end expression_value_l2614_261425


namespace union_of_sets_l2614_261467

def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {a + 2, 5}

theorem union_of_sets (a : ℕ) (h : A ∩ B a = {3}) : A ∪ B a = {1, 3, 5} := by
  sorry

end union_of_sets_l2614_261467


namespace ryan_weekly_commute_l2614_261488

/-- Represents the different routes Ryan can take --/
inductive Route
| A
| B

/-- Represents the days of the week --/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Represents the different transportation methods --/
inductive TransportMethod
| Bike
| Bus
| FriendRide
| Walk

/-- Function to calculate biking time based on the route --/
def bikingTime (route : Route) : ℕ :=
  match route with
  | Route.A => 30
  | Route.B => 40

/-- Function to calculate bus time based on the day --/
def busTime (day : Day) : ℕ :=
  match day with
  | Day.Tuesday => 50
  | _ => 40

/-- Function to calculate friend's ride time based on the day --/
def friendRideTime (day : Day) : ℕ :=
  match day with
  | Day.Wednesday => 25
  | _ => 10

/-- Function to calculate walking time --/
def walkingTime : ℕ := 90

/-- Function to calculate total weekly commuting time --/
def totalWeeklyCommutingTime : ℕ :=
  (bikingTime Route.A + bikingTime Route.B) +
  (busTime Day.Monday + busTime Day.Tuesday + busTime Day.Wednesday) +
  friendRideTime Day.Wednesday +
  walkingTime

/-- Theorem stating that Ryan's total weekly commuting time is 315 minutes --/
theorem ryan_weekly_commute : totalWeeklyCommutingTime = 315 := by
  sorry

end ryan_weekly_commute_l2614_261488


namespace price_reduction_proof_l2614_261459

/-- The original selling price -/
def original_price : ℝ := 40

/-- The cost price -/
def cost_price : ℝ := 30

/-- The initial daily sales volume -/
def initial_sales : ℝ := 48

/-- The price after two reductions -/
def reduced_price : ℝ := 32.4

/-- The additional sales per yuan of price reduction -/
def sales_increase_rate : ℝ := 8

/-- The target daily profit -/
def target_profit : ℝ := 504

/-- The percentage reduction in price -/
def reduction_percentage : ℝ := 0.1

/-- The price reduction amount -/
def price_reduction : ℝ := 3

theorem price_reduction_proof :
  (∃ x : ℝ, (1 - x)^2 * original_price = reduced_price ∧ x = reduction_percentage) ∧
  (∃ m : ℝ, (original_price - m - cost_price) * (initial_sales + sales_increase_rate * m) = target_profit ∧ m = price_reduction) := by
  sorry

end price_reduction_proof_l2614_261459


namespace quadratic_solution_sum_l2614_261408

theorem quadratic_solution_sum (c d : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 2 * x + 8 = 0 ↔ x = c + d * I ∨ x = c - d * I) → 
  c + d^2 = 44/25 := by
  sorry

end quadratic_solution_sum_l2614_261408


namespace estimate_fish_population_l2614_261481

/-- Estimate the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (marked_fish : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  marked_fish = 200 →
  second_catch = 100 →
  marked_in_second = 10 →
  (marked_fish * second_catch) / marked_in_second = 2000 :=
by
  sorry

#check estimate_fish_population

end estimate_fish_population_l2614_261481


namespace total_kites_sold_l2614_261472

-- Define the sequence
def kite_sequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

-- Define the sum of the sequence
def kite_sum (n : ℕ) : ℕ := 
  n * (kite_sequence 1 + kite_sequence n) / 2

-- Theorem statement
theorem total_kites_sold : kite_sum 15 = 345 := by
  sorry

end total_kites_sold_l2614_261472


namespace vertex_of_our_parabola_l2614_261412

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola -/
def our_parabola : Parabola :=
  ⟨λ x => 2 * (x - 1)^2 + 5⟩

theorem vertex_of_our_parabola :
  vertex our_parabola = (1, 5) := by sorry

end vertex_of_our_parabola_l2614_261412


namespace min_value_expression_l2614_261483

theorem min_value_expression (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : c ≠ 0) :
  ((a - c)^2 + (c - b)^2 + (b - a)^2) / c^2 ≥ 2/3 := by
  sorry

end min_value_expression_l2614_261483


namespace geometric_sum_specific_l2614_261466

/-- Sum of a finite geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: The sum of the first 6 terms of the geometric sequence with
    first term 1/5 and common ratio 1/5 is equal to 1953/7812 -/
theorem geometric_sum_specific : geometric_sum (1/5) (1/5) 6 = 1953/7812 := by
  sorry

end geometric_sum_specific_l2614_261466


namespace equation_represents_hyperbola_and_parabola_l2614_261406

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation y^4 - 9x^6 = 3y^2 - 1 -/
def equation (p : Point) : Prop :=
  p.y^4 - 9*p.x^6 = 3*p.y^2 - 1

/-- Represents a hyperbola -/
def is_hyperbola (S : Set Point) : Prop :=
  ∃ a b c d e f : ℝ, ∀ p ∈ S, a*p.x^2 + b*p.y^2 + c*p.x*p.y + d*p.x + e*p.y + f = 0 ∧ a*b < 0

/-- Represents a parabola -/
def is_parabola (S : Set Point) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ p ∈ S, p.y = a*p.x^2 + b*p.x + c ∨ p.x = a*p.y^2 + b*p.y + d

/-- The theorem to be proved -/
theorem equation_represents_hyperbola_and_parabola :
  ∃ S₁ S₂ : Set Point,
    (∀ p, p ∈ S₁ ∪ S₂ ↔ equation p) ∧
    is_hyperbola S₁ ∧
    is_parabola S₂ :=
sorry

end equation_represents_hyperbola_and_parabola_l2614_261406


namespace expression_values_l2614_261487

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 := by
sorry

end expression_values_l2614_261487


namespace prob_black_fourth_draw_l2614_261400

structure Box where
  red_balls : ℕ
  black_balls : ℕ

def initial_box : Box := { red_balls := 3, black_balls := 3 }

def total_balls (b : Box) : ℕ := b.red_balls + b.black_balls

def prob_black_first_draw (b : Box) : ℚ :=
  b.black_balls / (total_balls b)

theorem prob_black_fourth_draw (b : Box) :
  prob_black_first_draw b = 1/2 →
  (∃ (p : ℚ), p = prob_black_first_draw b ∧ p = 1/2) :=
by sorry

end prob_black_fourth_draw_l2614_261400


namespace max_common_segment_for_coprime_l2614_261450

/-- The maximum length of the common initial segment of two sequences with coprime periods -/
def max_common_segment (m n : ℕ) : ℕ :=
  m + n - 2

/-- Theorem: For coprime positive integers m and n, the maximum length of the common
    initial segment of two sequences with periods m and n respectively is m + n - 2 -/
theorem max_common_segment_for_coprime (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
    (h_coprime : Nat.Coprime m n) : 
  max_common_segment m n = m + n - 2 := by
  sorry

#check max_common_segment_for_coprime

end max_common_segment_for_coprime_l2614_261450


namespace woody_writing_time_l2614_261404

/-- Proves that Woody spent 1.5 years writing his book given the conditions -/
theorem woody_writing_time :
  ∀ (woody_months ivanka_months : ℕ),
  ivanka_months = woody_months + 3 →
  woody_months + ivanka_months = 39 →
  (woody_months : ℚ) / 12 = 3/2 := by
sorry

end woody_writing_time_l2614_261404


namespace subtraction_and_decimal_conversion_l2614_261473

theorem subtraction_and_decimal_conversion : 3/4 - 1/16 = 0.6875 := by
  sorry

end subtraction_and_decimal_conversion_l2614_261473


namespace min_sum_distances_l2614_261410

/-- Given a parabola and a line, prove the minimum sum of distances -/
theorem min_sum_distances (x y : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 8*x}
  let line := {(x, y) : ℝ × ℝ | x - y + 2 = 0}
  let d1 (p : ℝ × ℝ) := |p.1|  -- distance from point to y-axis
  let d2 (p : ℝ × ℝ) := |p.1 - p.2 + 2| / Real.sqrt 2  -- distance from point to line
  ∃ (min : ℝ), ∀ (p : ℝ × ℝ), p ∈ parabola → d1 p + d2 p ≥ min ∧ 
  ∃ (q : ℝ × ℝ), q ∈ parabola ∧ d1 q + d2 q = min ∧ min = 2 * Real.sqrt 2 + 2 :=
sorry

end min_sum_distances_l2614_261410


namespace boisjoli_farm_egg_boxes_l2614_261458

/-- Calculates the number of egg boxes filled per week given the number of hens,
    eggs per hen per day, eggs per box, and days per week. -/
def boxes_per_week (hens : ℕ) (eggs_per_hen_per_day : ℕ) (eggs_per_box : ℕ) (days_per_week : ℕ) : ℕ :=
  (hens * eggs_per_hen_per_day * days_per_week) / eggs_per_box

/-- Proves that given 270 hens, each laying one egg per day, packed in boxes of 6,
    collected 7 days a week, the total number of boxes filled per week is 315. -/
theorem boisjoli_farm_egg_boxes :
  boxes_per_week 270 1 6 7 = 315 := by
  sorry

end boisjoli_farm_egg_boxes_l2614_261458


namespace simplest_square_root_l2614_261420

/-- Given real numbers a and b, with a ≠ 0, prove that √(a^2 + b^2) is the simplest form among:
    √(16a), √(a^2 + b^2), √(b/a), and √45 -/
theorem simplest_square_root (a b : ℝ) (ha : a ≠ 0) :
  ∃ (f : ℝ → ℝ), f (Real.sqrt (a^2 + b^2)) = Real.sqrt (a^2 + b^2) ∧
    (∀ g : ℝ → ℝ, g (Real.sqrt (16*a)) ≠ Real.sqrt (16*a) ∨
                   g (Real.sqrt (b/a)) ≠ Real.sqrt (b/a) ∨
                   g (Real.sqrt 45) ≠ Real.sqrt 45 ∨
                   g = f) :=
by sorry

end simplest_square_root_l2614_261420


namespace campers_rowing_count_l2614_261498

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 15

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 17

/-- The total number of campers who went rowing that day -/
def total_campers : ℕ := morning_campers + afternoon_campers

theorem campers_rowing_count : total_campers = 32 := by
  sorry

end campers_rowing_count_l2614_261498


namespace suraj_average_l2614_261432

theorem suraj_average (initial_average : ℝ) (innings : ℕ) (new_score : ℝ) (average_increase : ℝ) : 
  innings = 14 →
  new_score = 140 →
  average_increase = 8 →
  (innings * initial_average + new_score) / (innings + 1) = initial_average + average_increase →
  initial_average + average_increase = 28 :=
by sorry

end suraj_average_l2614_261432


namespace positive_numbers_l2614_261470

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_numbers_l2614_261470


namespace total_project_hours_l2614_261443

def project_hours (kate_hours : ℝ) : ℝ × ℝ × ℝ := 
  let pat_hours := 2 * kate_hours
  let mark_hours := kate_hours + 65
  (pat_hours, kate_hours, mark_hours)

theorem total_project_hours : 
  ∃ (kate_hours : ℝ), 
    let (pat_hours, _, mark_hours) := project_hours kate_hours
    pat_hours = (1/3) * mark_hours ∧ 
    pat_hours + kate_hours + mark_hours = 117 := by
  sorry

end total_project_hours_l2614_261443


namespace circle_area_difference_l2614_261497

/-- The difference in areas between a circle with radius 25 inches and a circle with diameter 15 inches is 568.75π square inches. -/
theorem circle_area_difference : 
  let r1 : ℝ := 25
  let d2 : ℝ := 15
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 568.75 * π := by sorry

end circle_area_difference_l2614_261497


namespace compute_expression_l2614_261405

theorem compute_expression : 3 * 3^4 - 9^60 / 9^57 = -486 := by
  sorry

end compute_expression_l2614_261405


namespace angle_value_l2614_261457

theorem angle_value (θ : Real) (h : θ > 0 ∧ θ < 90) : 
  (∃ (x y : Real), x = Real.sin (10 * π / 180) ∧ 
                   y = 1 + Real.sin (80 * π / 180) ∧ 
                   x = Real.sin θ ∧ 
                   y = Real.cos θ) → 
  θ = 85 * π / 180 := by
sorry

end angle_value_l2614_261457


namespace simplify_expression_l2614_261428

theorem simplify_expression : (81 / 16) ^ (3 / 4) - (-1) ^ 0 = 19 / 8 := by
  sorry

end simplify_expression_l2614_261428


namespace hostel_mess_expenditure_decrease_l2614_261493

/-- Proves that the decrease in average expenditure per head is 1 rupee
    given the initial conditions of the hostel mess problem. -/
theorem hostel_mess_expenditure_decrease :
  let initial_students : ℕ := 35
  let new_students : ℕ := 7
  let total_students : ℕ := initial_students + new_students
  let initial_expenditure : ℕ := 420
  let expenditure_increase : ℕ := 42
  let new_expenditure : ℕ := initial_expenditure + expenditure_increase
  let initial_average : ℚ := initial_expenditure / initial_students
  let new_average : ℚ := new_expenditure / total_students
  initial_average - new_average = 1 := by
  sorry

end hostel_mess_expenditure_decrease_l2614_261493


namespace stratified_sampling_most_suitable_l2614_261494

-- Define the land types
inductive LandType
  | Mountainous
  | Hilly
  | Flat
  | LowLying

-- Define the village structure
structure Village where
  landAreas : LandType → ℕ
  totalArea : ℕ
  sampleSize : ℕ

-- Define the sampling methods
inductive SamplingMethod
  | Drawing
  | RandomNumberTable
  | Systematic
  | Stratified

-- Define the suitability of a sampling method
def isSuitable (v : Village) (m : SamplingMethod) : Prop :=
  m = SamplingMethod.Stratified

-- Theorem statement
theorem stratified_sampling_most_suitable (v : Village) 
  (h1 : v.landAreas LandType.Mountainous = 8000)
  (h2 : v.landAreas LandType.Hilly = 12000)
  (h3 : v.landAreas LandType.Flat = 24000)
  (h4 : v.landAreas LandType.LowLying = 4000)
  (h5 : v.totalArea = 48000)
  (h6 : v.sampleSize = 480) :
  isSuitable v SamplingMethod.Stratified :=
sorry

end stratified_sampling_most_suitable_l2614_261494


namespace smallest_sum_with_same_probability_l2614_261455

/-- Represents a symmetrical die with 6 faces --/
structure SymmetricalDie :=
  (faces : Fin 6)

/-- Represents a set of symmetrical dice --/
def DiceSet := List SymmetricalDie

/-- The probability of getting a specific sum --/
def probability (dice : DiceSet) (sum : Nat) : ℝ := sorry

theorem smallest_sum_with_same_probability 
  (dice : DiceSet) 
  (p : ℝ) 
  (h1 : p > 0) 
  (h2 : probability dice 2022 = p) : 
  ∃ (smallest_sum : Nat), 
    smallest_sum = 337 ∧ 
    probability dice smallest_sum = p ∧ 
    ∀ (other_sum : Nat), 
      other_sum < smallest_sum → probability dice other_sum ≠ p :=
sorry

end smallest_sum_with_same_probability_l2614_261455


namespace license_plate_count_l2614_261415

/-- The number of letters in the alphabet. -/
def alphabet_size : ℕ := 26

/-- The number of possible odd digits. -/
def odd_digits : ℕ := 5

/-- The number of possible even digits. -/
def even_digits : ℕ := 5

/-- The number of possible digits that are multiples of 3. -/
def multiples_of_three : ℕ := 4

/-- The total number of license plates with the given constraints. -/
def total_license_plates : ℕ := alphabet_size ^ 3 * odd_digits * even_digits * multiples_of_three

theorem license_plate_count :
  total_license_plates = 17576000 := by
  sorry

end license_plate_count_l2614_261415


namespace total_peppers_weight_l2614_261489

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_peppers : ℚ := 0.3333333333333333

/-- The weight of red peppers bought by Hannah's Vegetarian Restaurant -/
def red_peppers : ℚ := 0.3333333333333333

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_peppers : ℚ := green_peppers + red_peppers

/-- Theorem stating that the total weight of peppers is 0.6666666666666666 pounds -/
theorem total_peppers_weight : total_peppers = 0.6666666666666666 := by
  sorry

end total_peppers_weight_l2614_261489


namespace f_derivative_at_2_l2614_261452

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_derivative_at_2 (a b : ℝ) :
  f a b 1 = -2 → (deriv (f a b)) 1 = 0 → (deriv (f a b)) 2 = -1/2 := by
  sorry

end f_derivative_at_2_l2614_261452


namespace reciprocal_equation_solution_l2614_261403

theorem reciprocal_equation_solution (x : ℝ) : 
  2 - (1 / (2 - x)^3) = 1 / (2 - x)^3 → x = 1 := by
  sorry

end reciprocal_equation_solution_l2614_261403


namespace equation_positive_root_l2614_261453

/-- Given an equation (x / (x - 5) = 3 - a / (x - 5)) with a positive root, prove that a = -5 --/
theorem equation_positive_root (x a : ℝ) (h : x > 0) 
  (eq : x / (x - 5) = 3 - a / (x - 5)) : a = -5 := by
  sorry

end equation_positive_root_l2614_261453


namespace problems_per_page_is_three_l2614_261478

/-- The number of problems on each page of homework -/
def problems_per_page : ℕ := sorry

/-- The number of pages of math homework -/
def math_pages : ℕ := 6

/-- The number of pages of reading homework -/
def reading_pages : ℕ := 4

/-- The total number of problems -/
def total_problems : ℕ := 30

/-- Theorem stating that the number of problems per page is 3 -/
theorem problems_per_page_is_three :
  problems_per_page = 3 ∧
  (math_pages + reading_pages) * problems_per_page = total_problems :=
sorry

end problems_per_page_is_three_l2614_261478


namespace square_inequality_for_negatives_l2614_261491

theorem square_inequality_for_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end square_inequality_for_negatives_l2614_261491


namespace min_value_theorem_l2614_261461

theorem min_value_theorem (α₁ α₂ : ℝ) 
  (h : (1 / (2 + Real.sin α₁)) + (1 / (2 + Real.sin (2 * α₂))) = 2) :
  ∃ (k₁ k₂ : ℤ), ∀ (m₁ m₂ : ℤ), 
    |10 * Real.pi - α₁ - α₂| ≥ |10 * Real.pi - ((-π/2 + 2*↑k₁*π) + (-π/4 + ↑k₂*π))| ∧
    |10 * Real.pi - ((-π/2 + 2*↑k₁*π) + (-π/4 + ↑k₂*π))| = π/4 :=
by sorry

end min_value_theorem_l2614_261461


namespace range_of_m_l2614_261417

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Define the set of m values that satisfy the conditions
def S : Set ℝ := {m | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : S = Set.Ioo 1 2 ∪ Set.Ici 3 :=
sorry

end range_of_m_l2614_261417


namespace monthly_salary_calculation_l2614_261456

/-- Represents the monthly salary in Rupees -/
def monthly_salary : ℝ := sorry

/-- Represents the savings rate as a decimal -/
def savings_rate : ℝ := 0.2

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.2

/-- Represents the new monthly savings amount in Rupees -/
def new_savings : ℝ := 240

theorem monthly_salary_calculation :
  monthly_salary * (1 - (1 + expense_increase_rate) * (1 - savings_rate)) = new_savings ∧
  monthly_salary = 6000 := by sorry

end monthly_salary_calculation_l2614_261456


namespace circles_common_chord_l2614_261486

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem circles_common_chord :
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y →
  ∀ (x y : ℝ), (C₁ x y ∧ C₂ x y) ↔ common_chord x y :=
sorry

end circles_common_chord_l2614_261486


namespace problem_odometer_distance_l2614_261438

/-- Represents an odometer that skips certain digits -/
structure SkippingOdometer :=
  (skipped_digits : List Nat)
  (displayed_value : Nat)

/-- Calculates the actual distance for a skipping odometer -/
def actualDistance (o : SkippingOdometer) : Nat :=
  sorry

/-- The specific odometer from the problem -/
def problemOdometer : SkippingOdometer :=
  { skipped_digits := [4, 7],
    displayed_value := 3008 }

theorem problem_odometer_distance :
  actualDistance problemOdometer = 1542 :=
sorry

end problem_odometer_distance_l2614_261438


namespace equation_simplification_l2614_261492

theorem equation_simplification (Y : ℝ) : ((3.242 * 10 * Y) / 100) = 0.3242 * Y := by
  sorry

end equation_simplification_l2614_261492


namespace db_length_l2614_261429

/-- Triangle ABC with altitudes and median -/
structure TriangleABC where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  M : ℝ × ℝ
  -- CD is altitude to AB
  cd_altitude : (C.1 - D.1) * (B.1 - A.1) + (C.2 - D.2) * (B.2 - A.2) = 0
  -- AE is altitude to BC
  ae_altitude : (A.1 - E.1) * (C.1 - B.1) + (A.2 - E.2) * (C.2 - B.2) = 0
  -- AM is median to BC
  am_median : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  -- Given lengths
  ab_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 12
  cd_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 5
  ae_length : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 4
  am_length : Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 6

/-- The length of DB in the given triangle is 15 -/
theorem db_length (t : TriangleABC) : 
  Real.sqrt ((t.D.1 - t.B.1)^2 + (t.D.2 - t.B.2)^2) = 15 := by
  sorry

end db_length_l2614_261429


namespace unit_vector_parallel_to_a_l2614_261485

def vector_a : ℝ × ℝ := (12, 5)

theorem unit_vector_parallel_to_a :
  ∃ (u : ℝ × ℝ), (u.1 * u.1 + u.2 * u.2 = 1) ∧
  (∃ (k : ℝ), vector_a = (k * u.1, k * u.2)) ∧
  (u = (12/13, 5/13) ∨ u = (-12/13, -5/13)) :=
sorry

end unit_vector_parallel_to_a_l2614_261485


namespace evaluate_expression_l2614_261433

theorem evaluate_expression (y Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) :
  10 * (6 * y + 14 * Real.pi + y^2) = 4 * Q + 10 * y^2 := by
  sorry

end evaluate_expression_l2614_261433


namespace music_class_students_l2614_261484

theorem music_class_students :
  ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 ∧ n = 45 := by
sorry

end music_class_students_l2614_261484


namespace golden_ratio_trigonometric_identity_l2614_261439

theorem golden_ratio_trigonometric_identity :
  let m := 2 * Real.sin (18 * π / 180)
  (Real.sin (42 * π / 180) + m) / Real.cos (42 * π / 180) = Real.sqrt 3 := by
  sorry

end golden_ratio_trigonometric_identity_l2614_261439


namespace perpendicular_line_equation_l2614_261419

/-- A line passing through point (2,-1) and perpendicular to x+y-3=0 has the equation x-y-3=0 -/
theorem perpendicular_line_equation :
  let point : ℝ × ℝ := (2, -1)
  let perpendicular_to : ℝ → ℝ → ℝ := fun x y => x + y - 3
  let line_equation : ℝ → ℝ → ℝ := fun x y => x - y - 3
  (∀ x y, perpendicular_to x y = 0 → (line_equation x y = 0 ↔ 
    (x - point.1) * 1 = (y - point.2) * 1 ∧
    1 * (-1) = -1)) :=
by sorry

end perpendicular_line_equation_l2614_261419
