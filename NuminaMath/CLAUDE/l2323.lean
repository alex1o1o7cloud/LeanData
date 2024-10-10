import Mathlib

namespace max_value_on_ellipse_l2323_232335

theorem max_value_on_ellipse :
  ∃ (max : ℝ),
    (∀ x y : ℝ, (y^2 / 4 + x^2 / 3 = 1) → 2*x + y ≤ max) ∧
    (∃ x y : ℝ, (y^2 / 4 + x^2 / 3 = 1) ∧ 2*x + y = max) ∧
    max = 4 := by
  sorry

end max_value_on_ellipse_l2323_232335


namespace equidistant_point_y_axis_l2323_232344

/-- The y-coordinate of the point on the y-axis that is equidistant from A(-3, -2) and B(2, 3) is 0 -/
theorem equidistant_point_y_axis : ∃ y : ℝ, 
  (y = 0) ∧ 
  ((-3 - 0)^2 + (-2 - y)^2 = (2 - 0)^2 + (3 - y)^2) := by
  sorry

end equidistant_point_y_axis_l2323_232344


namespace decreasing_function_on_positive_reals_l2323_232381

/-- The function f(x) = -x(x+2) is decreasing on the interval (0, +∞) -/
theorem decreasing_function_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → -x * (x + 2) > -y * (y + 2) := by
  sorry

end decreasing_function_on_positive_reals_l2323_232381


namespace sin_less_than_x_l2323_232342

theorem sin_less_than_x :
  (∀ x : ℝ, 0 < x → x < π / 2 → Real.sin x < x) ∧
  (∀ x : ℝ, x > 0 → Real.sin x < x) := by
  sorry

end sin_less_than_x_l2323_232342


namespace roots_of_polynomial_l2323_232317

def p (x : ℝ) : ℝ := 3*x^4 + 16*x^3 - 36*x^2 + 8*x

theorem roots_of_polynomial :
  ∃ (a b : ℝ), a^2 = 17 ∧
  (p 0 = 0) ∧
  (p (1/3) = 0) ∧
  (p (-3 + 2*a) = 0) ∧
  (p (-3 - 2*a) = 0) ∧
  (∀ x : ℝ, p x = 0 → x = 0 ∨ x = 1/3 ∨ x = -3 + 2*a ∨ x = -3 - 2*a) :=
by sorry

end roots_of_polynomial_l2323_232317


namespace cylinder_equal_volume_increase_l2323_232336

/-- Theorem: For a cylinder with radius 6 inches and height 4 inches, 
    the value of x that satisfies the equation π(R+x)²H = πR²(H+2x) is 6 inches. -/
theorem cylinder_equal_volume_increase (π : ℝ) : 
  ∃ (x : ℝ), x = 6 ∧ π * (6 + x)^2 * 4 = π * 6^2 * (4 + 2*x) :=
by sorry

end cylinder_equal_volume_increase_l2323_232336


namespace greatest_common_multiple_under_120_l2323_232333

theorem greatest_common_multiple_under_120 : 
  ∀ n : ℕ, n < 120 → n % 10 = 0 → n % 15 = 0 → n ≤ 90 :=
by
  sorry

end greatest_common_multiple_under_120_l2323_232333


namespace meaningful_expression_l2323_232308

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y * y = x ∧ y ≥ 0) ∧ x ≠ 2 ↔ x ≥ 0 ∧ x ≠ 2 := by
sorry

end meaningful_expression_l2323_232308


namespace x_positive_sufficient_not_necessary_for_abs_x_eq_x_l2323_232364

theorem x_positive_sufficient_not_necessary_for_abs_x_eq_x :
  (∀ x : ℝ, x > 0 → |x| = x) ∧
  (∃ x : ℝ, |x| = x ∧ ¬(x > 0)) := by
  sorry

end x_positive_sufficient_not_necessary_for_abs_x_eq_x_l2323_232364


namespace fourth_root_over_seventh_root_of_seven_l2323_232347

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (h : x = 7) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by sorry

end fourth_root_over_seventh_root_of_seven_l2323_232347


namespace quadratic_sum_powers_divisibility_l2323_232393

/-- Represents a quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  p : ℤ
  q : ℤ

/-- Condition that the polynomial has a positive discriminant -/
def has_positive_discriminant (f : QuadraticPolynomial) : Prop :=
  f.p * f.p - 4 * f.q > 0

/-- Sum of the hundredth powers of the roots of a quadratic polynomial -/
noncomputable def sum_of_hundredth_powers (f : QuadraticPolynomial) : ℝ :=
  let α := (-f.p + Real.sqrt (f.p * f.p - 4 * f.q)) / 2
  let β := (-f.p - Real.sqrt (f.p * f.p - 4 * f.q)) / 2
  α^100 + β^100

/-- Main theorem statement -/
theorem quadratic_sum_powers_divisibility 
  (f : QuadraticPolynomial) 
  (h_disc : has_positive_discriminant f) 
  (h_p : f.p % 5 = 0) 
  (h_q : f.q % 5 = 0) : 
  ∃ (k : ℤ), sum_of_hundredth_powers f = k * (5^50 : ℝ) ∧ 
  ∀ (n : ℕ), n > 50 → ¬∃ (m : ℤ), sum_of_hundredth_powers f = m * (5^n : ℝ) :=
sorry

end quadratic_sum_powers_divisibility_l2323_232393


namespace evaluate_expression_l2323_232351

theorem evaluate_expression : 3 - 5 * (2^3 + 3) * 2 = -107 := by
  sorry

end evaluate_expression_l2323_232351


namespace consecutive_product_bound_l2323_232371

theorem consecutive_product_bound (π : Fin 90 → Fin 90) (h : Function.Bijective π) :
  ∃ i : Fin 89, (π i).val * (π (i + 1)).val ≥ 2014 ∨
    (π (Fin.last 89)).val * (π 0).val ≥ 2014 :=
by sorry

end consecutive_product_bound_l2323_232371


namespace P_intersect_Q_eq_P_l2323_232360

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem P_intersect_Q_eq_P : P ∩ Q = P := by sorry

end P_intersect_Q_eq_P_l2323_232360


namespace smallest_resolvable_debt_is_correct_l2323_232356

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 250

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = pig_value * p + goat_value * g

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 50

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, 0 < d → d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
sorry

end smallest_resolvable_debt_is_correct_l2323_232356


namespace tourist_assignment_count_l2323_232304

/-- The number of ways to assign tourists to scenic spots -/
def assignmentCount (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else if n = k then Nat.factorial k
  else (Nat.choose n 2) * (Nat.factorial k)

theorem tourist_assignment_count :
  assignmentCount 4 3 = 36 := by
  sorry

end tourist_assignment_count_l2323_232304


namespace halfway_between_fractions_l2323_232328

theorem halfway_between_fractions : 
  (2 : ℚ) / 9 + (1 : ℚ) / 3 = (5 : ℚ) / 9 ∧ (5 : ℚ) / 9 / 2 = (5 : ℚ) / 18 := by
  sorry

end halfway_between_fractions_l2323_232328


namespace complex_magnitude_problem_l2323_232341

theorem complex_magnitude_problem (z : ℂ) : z = 2 / (1 - I) + I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l2323_232341


namespace smallest_positive_integer_ending_in_3_divisible_by_11_l2323_232390

theorem smallest_positive_integer_ending_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 3 → m % 11 = 0 → n ≤ m :=
by
  use 33
  sorry

end smallest_positive_integer_ending_in_3_divisible_by_11_l2323_232390


namespace muffin_packs_per_case_l2323_232376

/-- Proves the number of packs per case for Nora's muffin sale -/
theorem muffin_packs_per_case 
  (total_amount : ℕ) 
  (price_per_muffin : ℕ) 
  (num_cases : ℕ) 
  (muffins_per_pack : ℕ) 
  (h1 : total_amount = 120)
  (h2 : price_per_muffin = 2)
  (h3 : num_cases = 5)
  (h4 : muffins_per_pack = 4) :
  (total_amount / price_per_muffin) / num_cases / muffins_per_pack = 3 := by
  sorry

#check muffin_packs_per_case

end muffin_packs_per_case_l2323_232376


namespace percentage_increase_sum_l2323_232378

theorem percentage_increase_sum (A B C x y : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A = 120 → B = 110 → C = 100 →
  A = C * (1 + x / 100) →
  B = C * (1 + y / 100) →
  x + y = 30 := by
sorry

end percentage_increase_sum_l2323_232378


namespace power_of_five_mod_eight_l2323_232301

theorem power_of_five_mod_eight : 5^1082 % 8 = 1 := by
  sorry

end power_of_five_mod_eight_l2323_232301


namespace second_candidate_percentage_l2323_232329

theorem second_candidate_percentage : ∀ (total_marks : ℕ) (passing_mark : ℕ),
  passing_mark = 160 →
  (0.4 : ℝ) * total_marks = passing_mark - 40 →
  let second_candidate_marks := passing_mark + 20
  ((second_candidate_marks : ℝ) / total_marks) * 100 = 60 := by
  sorry

end second_candidate_percentage_l2323_232329


namespace lee_cookies_l2323_232311

/-- Given that Lee can make 24 cookies with 4 cups of flour, 
    this theorem proves that he can make 30 cookies with 5 cups of flour. -/
theorem lee_cookies (cookies_per_4_cups : ℕ) (h : cookies_per_4_cups = 24) :
  (cookies_per_4_cups * 5 / 4 : ℚ) = 30 := by
  sorry


end lee_cookies_l2323_232311


namespace shane_sandwiches_l2323_232348

/-- The number of slices in each package of sliced ham -/
def slices_per_ham_package (
  bread_slices_per_package : ℕ)
  (num_bread_packages : ℕ)
  (num_ham_packages : ℕ)
  (leftover_bread_slices : ℕ)
  (bread_slices_per_sandwich : ℕ) : ℕ :=
  let total_bread_slices := bread_slices_per_package * num_bread_packages
  let used_bread_slices := total_bread_slices - leftover_bread_slices
  let num_sandwiches := used_bread_slices / bread_slices_per_sandwich
  num_sandwiches / num_ham_packages

theorem shane_sandwiches :
  slices_per_ham_package 20 2 2 8 2 = 8 := by
  sorry

end shane_sandwiches_l2323_232348


namespace orange_pyramid_count_l2323_232338

def pyramid_oranges (base_length : ℕ) (base_width : ℕ) (top_oranges : ℕ) : ℕ :=
  let layers := min base_length base_width
  (layers * (base_length + base_width - layers + 1) * (2 * base_length + 2 * base_width - 3 * layers + 1)) / 6 + top_oranges

theorem orange_pyramid_count : pyramid_oranges 7 10 3 = 227 := by
  sorry

end orange_pyramid_count_l2323_232338


namespace polynomial_simplification_l2323_232353

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 9 * x - 5) - (2 * x^2 + 4 * x - 15) = x^2 + 5 * x + 10 := by
  sorry

end polynomial_simplification_l2323_232353


namespace probability_multiple_3_or_5_l2323_232357

def is_multiple_of_3_or_5 (n : ℕ) : Bool :=
  n % 3 = 0 || n % 5 = 0

def count_multiples (max : ℕ) : ℕ :=
  (List.range max).filter is_multiple_of_3_or_5 |>.length

theorem probability_multiple_3_or_5 :
  (count_multiples 20 : ℚ) / 20 = 9 / 20 := by
  sorry

end probability_multiple_3_or_5_l2323_232357


namespace intersection_when_a_is_4_range_of_a_for_subset_l2323_232334

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 3) * (x - 3 * a - 5) < 0}
def B : Set ℝ := {x | -x^2 + 5*x + 14 > 0}

-- Part 1
theorem intersection_when_a_is_4 :
  A 4 ∩ B = {x | 3 < x ∧ x < 7} :=
sorry

-- Part 2
theorem range_of_a_for_subset :
  {a : ℝ | A a ⊆ B} = {a : ℝ | -7/3 ≤ a ∧ a ≤ 2/3} :=
sorry

end intersection_when_a_is_4_range_of_a_for_subset_l2323_232334


namespace vector_subtraction_magnitude_l2323_232343

/-- Given two 2D vectors a and b, where the angle between them is 45°,
    a = (-1, 1), and |b| = 1, prove that |a - 2b| = √2 -/
theorem vector_subtraction_magnitude (a b : ℝ × ℝ) :
  let angle := Real.pi / 4
  a.1 = -1 ∧ a.2 = 1 →
  Real.sqrt (b.1^2 + b.2^2) = 1 →
  a.1 * b.1 + a.2 * b.2 = Real.sqrt 2 / 2 →
  Real.sqrt ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2) = Real.sqrt 2 := by
  sorry

end vector_subtraction_magnitude_l2323_232343


namespace positive_solution_x_l2323_232368

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 8 - 2 * x - 3 * y)
  (eq2 : y * z = 8 - 4 * y - 2 * z)
  (eq3 : x * z = 40 - 4 * x - 3 * z)
  (pos : x > 0) :
  x = (7 * Real.sqrt 13 - 6) / 2 := by
sorry

end positive_solution_x_l2323_232368


namespace shaded_region_perimeter_l2323_232354

/-- The perimeter of a region consisting of two radii of length 5 and a 3/4 circular arc of a circle with radius 5 is equal to 10 + (15π/2). -/
theorem shaded_region_perimeter (r : ℝ) (h : r = 5) :
  2 * r + (3/4) * (2 * π * r) = 10 + (15 * π) / 2 :=
by sorry

end shaded_region_perimeter_l2323_232354


namespace pure_imaginary_condition_l2323_232337

/-- Given that i is the imaginary unit and (1+ai)/i is a pure imaginary number, prove that a = 0 -/
theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →  -- i is the imaginary unit
  (↑1 + a * Complex.I) / Complex.I = b * Complex.I →  -- (1+ai)/i is a pure imaginary number
  a = 0 := by sorry

end pure_imaginary_condition_l2323_232337


namespace distinguishable_triangles_count_l2323_232313

/-- Represents the number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- Represents the number of triangles in the large equilateral triangle -/
def num_triangles : ℕ := 6

/-- Represents the number of corner triangles -/
def num_corners : ℕ := 3

/-- Represents the number of edge triangles -/
def num_edges : ℕ := 2

/-- Represents the number of center triangles -/
def num_center : ℕ := 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of distinguishable large equilateral triangles -/
def num_distinguishable_triangles : ℕ :=
  -- Corner configurations
  (num_colors + 
   num_colors * (num_colors - 1) + 
   choose num_colors num_corners) *
  -- Edge configurations
  (num_colors ^ num_edges) *
  -- Center configurations
  num_colors

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 61440 :=
sorry

end distinguishable_triangles_count_l2323_232313


namespace scaling_transformation_for_given_points_l2323_232330

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  sx : ℝ  -- scaling factor for x
  sy : ℝ  -- scaling factor for y

/-- Apply a scaling transformation to a point -/
def apply_scaling (t : ScalingTransformation) (p : ℝ × ℝ) : ℝ × ℝ :=
  (t.sx * p.1, t.sy * p.2)

theorem scaling_transformation_for_given_points :
  ∃ (t : ScalingTransformation),
    apply_scaling t (-2, 2) = (-6, 1) ∧
    t.sx = 3 ∧
    t.sy = 1/2 := by
  sorry

end scaling_transformation_for_given_points_l2323_232330


namespace smallest_cube_for_cone_l2323_232366

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseDiameter : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- The volume of a cube -/
def cubeVolume (c : Cube) : ℝ :=
  c.sideLength ^ 3

/-- Predicate to check if a cube can contain a cone upright -/
def canContainCone (cube : Cube) (cone : Cone) : Prop :=
  cube.sideLength ≥ cone.height ∧ cube.sideLength ≥ cone.baseDiameter

theorem smallest_cube_for_cone (cone : Cone) 
    (h1 : cone.height = 15)
    (h2 : cone.baseDiameter = 8) :
    ∃ (cube : Cube), 
      canContainCone cube cone ∧ 
      cubeVolume cube = 3375 ∧
      ∀ (c : Cube), canContainCone c cone → cubeVolume c ≥ 3375 := by
  sorry

end smallest_cube_for_cone_l2323_232366


namespace complex_square_equation_l2323_232382

theorem complex_square_equation (a b : ℕ+) :
  (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I →
  a + b * Complex.I = 4 + 3 * Complex.I := by
sorry

end complex_square_equation_l2323_232382


namespace selection_problem_l2323_232385

theorem selection_problem (n m k l : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 3) (h4 : l = 2) : 
  (Nat.choose n m) - (Nat.choose k (l + 1) * Nat.choose (n - k) (m - l - 1)) = 756 := by
  sorry

end selection_problem_l2323_232385


namespace total_amount_is_454_5_l2323_232300

/-- Represents the share distribution problem -/
def ShareDistribution (w x y z p : ℚ) : Prop :=
  x = (3/2) * w ∧
  y = (1/3) * w ∧
  z = (3/4) * w ∧
  p = (5/8) * w ∧
  y = 36

/-- Theorem stating that the total amount is 454.5 rupees -/
theorem total_amount_is_454_5 (w x y z p : ℚ) 
  (h : ShareDistribution w x y z p) : 
  w + x + y + z + p = 454.5 := by
  sorry

#eval (454.5 : ℚ)

end total_amount_is_454_5_l2323_232300


namespace total_combinations_l2323_232307

/-- The number of color choices available for painting the box. -/
def num_colors : ℕ := 4

/-- The number of decoration choices available for the box. -/
def num_decorations : ℕ := 3

/-- The number of painting method choices available. -/
def num_methods : ℕ := 3

/-- Theorem stating the total number of combinations for painting the box. -/
theorem total_combinations : num_colors * num_decorations * num_methods = 36 := by
  sorry

end total_combinations_l2323_232307


namespace tens_digit_of_2023_power_minus_2025_squared_l2323_232384

theorem tens_digit_of_2023_power_minus_2025_squared : ∃ n : ℕ, 
  2023^2024 - 2025^2 = 100 * n + 16 :=
by sorry

end tens_digit_of_2023_power_minus_2025_squared_l2323_232384


namespace min_value_of_bisecting_line_l2323_232318

/-- A line that bisects the circumference of a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  bisects : ∀ x y : ℝ, a * x + 2 * b * y - 2 = 0 → x^2 + y^2 - 4*x - 2*y - 8 = 0

/-- The minimum value of 1/a + 2/b for a bisecting line -/
theorem min_value_of_bisecting_line (l : BisectingLine) : 
  ∃ (m : ℝ), (∀ a b : ℝ, a > 0 → b > 0 → 
    (∀ x y : ℝ, a * x + 2 * b * y - 2 = 0 → x^2 + y^2 - 4*x - 2*y - 8 = 0) → 
    1/a + 2/b ≥ m) ∧ 
  1/l.a + 2/l.b = m ∧ 
  m = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_bisecting_line_l2323_232318


namespace turner_tickets_l2323_232389

def rollercoaster_rides : ℕ := 3
def catapult_rides : ℕ := 2
def ferris_wheel_rides : ℕ := 1

def rollercoaster_cost : ℕ := 4
def catapult_cost : ℕ := 4
def ferris_wheel_cost : ℕ := 1

def total_tickets : ℕ := 
  rollercoaster_rides * rollercoaster_cost + 
  catapult_rides * catapult_cost + 
  ferris_wheel_rides * ferris_wheel_cost

theorem turner_tickets : total_tickets = 21 := by
  sorry

end turner_tickets_l2323_232389


namespace petya_cannot_guarantee_win_l2323_232310

/-- Represents a position on the 9x9 board -/
structure Position :=
  (x : Fin 9)
  (y : Fin 9)

/-- Represents a direction of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a player in the game -/
inductive Player
  | Petya
  | Vasya

/-- The game state -/
structure GameState :=
  (position : Position)
  (lastDirection : Direction)
  (currentPlayer : Player)

/-- Checks if a move is valid for a given player -/
def isValidMove (player : Player) (lastDir : Direction) (newDir : Direction) : Prop :=
  match player with
  | Player.Petya => newDir = lastDir ∨ newDir = Direction.Right
  | Player.Vasya => newDir = lastDir ∨ newDir = Direction.Left

/-- Checks if a position is on the board -/
def isOnBoard (pos : Position) : Prop :=
  0 ≤ pos.x ∧ pos.x < 9 ∧ 0 ≤ pos.y ∧ pos.y < 9

/-- Theorem stating that Petya cannot guarantee a win -/
theorem petya_cannot_guarantee_win :
  ∀ (strategy : GameState → Direction),
  ∃ (counterStrategy : GameState → Direction),
  ∃ (finalState : GameState),
  (finalState.currentPlayer = Player.Petya ∧ 
   ¬∃ (dir : Direction), isValidMove Player.Petya finalState.lastDirection dir ∧ 
                         isOnBoard (finalState.position)) :=
sorry

end petya_cannot_guarantee_win_l2323_232310


namespace volleyball_score_ratio_l2323_232305

theorem volleyball_score_ratio :
  let lizzie_score : ℕ := 4
  let nathalie_score : ℕ := lizzie_score + 3
  let combined_score : ℕ := lizzie_score + nathalie_score
  let team_total : ℕ := 50
  let teammates_score : ℕ := 17
  ∃ (m : ℕ), 
    m * combined_score = team_total - lizzie_score - nathalie_score - teammates_score ∧
    m * combined_score = 2 * combined_score :=
by
  sorry

#check volleyball_score_ratio

end volleyball_score_ratio_l2323_232305


namespace chessboard_coloring_theorem_l2323_232399

/-- Represents a coloring of a chessboard -/
def Coloring (n k : ℕ) := Fin (2*n) → Fin k → Fin n

/-- Checks if a coloring has a monochromatic rectangle -/
def has_monochromatic_rectangle (n k : ℕ) (c : Coloring n k) : Prop :=
  ∃ (r₁ r₂ : Fin (2*n)) (c₁ c₂ : Fin k),
    r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧
    c r₁ c₁ = c r₁ c₂ ∧ c r₁ c₁ = c r₂ c₁ ∧ c r₁ c₁ = c r₂ c₂

/-- The main theorem -/
theorem chessboard_coloring_theorem (n : ℕ) (h : n > 0) :
  ∀ k : ℕ, (k ≥ n*(2*n-1) + 1) →
    ∀ c : Coloring n k, has_monochromatic_rectangle n k c :=
sorry

end chessboard_coloring_theorem_l2323_232399


namespace cubic_roots_sum_l2323_232325

theorem cubic_roots_sum (a b c : ℝ) : 
  (3 * a^3 - 5 * a^2 + 90 * a - 2 = 0) →
  (3 * b^3 - 5 * b^2 + 90 * b - 2 = 0) →
  (3 * c^3 - 5 * c^2 + 90 * c - 2 = 0) →
  (a + b + 1)^3 + (b + c + 1)^3 + (c + a + 1)^3 = 259 + 1/3 := by
  sorry

end cubic_roots_sum_l2323_232325


namespace complex_number_real_l2323_232340

theorem complex_number_real (m : ℝ) :
  (m ≠ -5) →
  (∃ (z : ℂ), z = (m + 5)⁻¹ + (m^2 + 2*m - 15)*I ∧ z.im = 0) →
  m = 3 := by
sorry

end complex_number_real_l2323_232340


namespace decagon_triangle_probability_l2323_232370

/-- A decagon is a polygon with 10 vertices -/
def Decagon : ℕ := 10

/-- The number of ways to choose 3 distinct vertices from a decagon -/
def TotalChoices : ℕ := Nat.choose Decagon 3

/-- The number of ways to choose 3 distinct vertices that form a triangle with sides as edges -/
def FavorableChoices : ℕ := Decagon

/-- The probability of choosing 3 distinct vertices that form a triangle with sides as edges -/
def ProbabilityOfTriangle : ℚ := FavorableChoices / TotalChoices

theorem decagon_triangle_probability :
  ProbabilityOfTriangle = 1 / 12 := by
  sorry

end decagon_triangle_probability_l2323_232370


namespace beth_coin_ratio_l2323_232355

/-- Proves that the ratio of coins Beth sold to her total coins after receiving Carl's gift is 1:2 -/
theorem beth_coin_ratio :
  let initial_coins : ℕ := 125
  let gift_coins : ℕ := 35
  let sold_coins : ℕ := 80
  let total_coins : ℕ := initial_coins + gift_coins
  (sold_coins : ℚ) / total_coins = 1 / 2 := by
  sorry

end beth_coin_ratio_l2323_232355


namespace base9_to_base5_conversion_l2323_232394

/-- Converts a base-9 number to its decimal (base-10) representation -/
def base9ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Converts a decimal (base-10) number to its base-5 representation -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The base-9 representation of the number to be converted -/
def number_base9 : List Nat := [4, 2, 7]

theorem base9_to_base5_conversion :
  decimalToBase5 (base9ToDecimal number_base9) = [4, 3, 2, 4] :=
sorry

end base9_to_base5_conversion_l2323_232394


namespace initial_blue_marbles_l2323_232345

/-- Proves that the initial number of blue marbles is 30 given the conditions of the problem -/
theorem initial_blue_marbles (initial_red : ℕ) (removed_red : ℕ) (total_left : ℕ)
  (h1 : initial_red = 20)
  (h2 : removed_red = 3)
  (h3 : total_left = 35) :
  initial_red + (total_left + removed_red + 4 * removed_red - (initial_red - removed_red)) = 30 := by
  sorry

end initial_blue_marbles_l2323_232345


namespace optimal_strategy_l2323_232312

-- Define the warehouse options
structure Warehouse where
  monthly_rent : ℝ
  repossession_probability : ℝ
  repossession_time : ℕ

-- Define the company's parameters
structure Company where
  planning_horizon : ℕ
  moving_cost : ℝ

-- Define the purchase option
structure PurchaseOption where
  total_price : ℝ
  installment_period : ℕ

def calculate_total_cost (w : Warehouse) (c : Company) (years : ℕ) : ℝ :=
  sorry

def calculate_purchase_cost (p : PurchaseOption) : ℝ :=
  sorry

theorem optimal_strategy (w1 w2 : Warehouse) (c : Company) (p : PurchaseOption) :
  w1.monthly_rent = 80000 ∧
  w2.monthly_rent = 20000 ∧
  w2.repossession_probability = 0.5 ∧
  w2.repossession_time = 5 ∧
  c.planning_horizon = 60 ∧
  c.moving_cost = 150000 ∧
  p.total_price = 3000000 ∧
  p.installment_period = 36 →
  calculate_total_cost w2 c 1 + calculate_purchase_cost p <
  min (calculate_total_cost w1 c 5) (calculate_total_cost w2 c 5) :=
sorry

#check optimal_strategy

end optimal_strategy_l2323_232312


namespace relationship_proof_l2323_232349

theorem relationship_proof (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 := by
  sorry

end relationship_proof_l2323_232349


namespace complex_parts_l2323_232331

theorem complex_parts (z : ℂ) (h : z = 2 - 3*I) : 
  z.re = 2 ∧ z.im = -3 := by sorry

end complex_parts_l2323_232331


namespace olivia_change_l2323_232363

def basketball_card_price : ℕ := 3
def baseball_card_price : ℕ := 4
def num_basketball_packs : ℕ := 2
def num_baseball_decks : ℕ := 5
def bill_value : ℕ := 50

def total_cost : ℕ := num_basketball_packs * basketball_card_price + num_baseball_decks * baseball_card_price

theorem olivia_change : bill_value - total_cost = 24 := by
  sorry

end olivia_change_l2323_232363


namespace repeated_digit_sum_2_power_2004_l2323_232302

/-- The digit sum function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Repeated application of digit_sum until a single digit is reached -/
def repeated_digit_sum (n : ℕ) : ℕ := sorry

/-- 2^2004 mod 9 ≡ 1 -/
lemma power_two_2004_mod_9 : 2^2004 % 9 = 1 := sorry

/-- For any natural number n, n ≡ digit_sum(n) (mod 9) -/
lemma digit_sum_congruence (n : ℕ) : n % 9 = digit_sum n % 9 := sorry

/-- The main theorem -/
theorem repeated_digit_sum_2_power_2004 : 
  repeated_digit_sum (2^2004) = 1 := sorry

end repeated_digit_sum_2_power_2004_l2323_232302


namespace total_spent_correct_l2323_232320

def regular_fee : ℝ := 150
def discount_rate : ℝ := 0.075
def tax_rate : ℝ := 0.06
def total_teachers : ℕ := 22
def special_diet_teachers : ℕ := 3
def regular_food_allowance : ℝ := 10
def special_food_allowance : ℝ := 15

def total_spent : ℝ :=
  let discounted_fee := regular_fee * (1 - discount_rate) * total_teachers
  let taxed_fee := discounted_fee * (1 + tax_rate)
  let food_allowance := regular_food_allowance * (total_teachers - special_diet_teachers) +
                        special_food_allowance * special_diet_teachers
  taxed_fee + food_allowance

theorem total_spent_correct :
  total_spent = 3470.65 := by sorry

end total_spent_correct_l2323_232320


namespace exercise_books_count_l2323_232377

/-- Given a shop with pencils, pens, and exercise books in the ratio 10 : 2 : 3,
    prove that if there are 120 pencils, then there are 36 exercise books. -/
theorem exercise_books_count (pencils pens books : ℕ) : 
  pencils = 120 →
  pencils / 10 = pens / 2 →
  pencils / 10 = books / 3 →
  books = 36 := by
sorry

end exercise_books_count_l2323_232377


namespace problem_statement_l2323_232352

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_diff_xy : x ≠ y) (h_diff_xz : x ≠ z) (h_diff_yz : y ≠ z)
  (h_eq1 : (y + 1) / (x - z) = (x + y) / (z + 1))
  (h_eq2 : (y + 1) / (x - z) = x / (y + 1)) :
  x / (y + 1) = 2 := by
  sorry

end problem_statement_l2323_232352


namespace time_difference_per_question_l2323_232392

/-- Prove that the difference in time per question between the Math and English exams is 4 minutes -/
theorem time_difference_per_question (english_questions math_questions : ℕ) 
  (english_duration math_duration : ℚ) : 
  english_questions = 30 →
  math_questions = 15 →
  english_duration = 1 →
  math_duration = 3/2 →
  (math_duration * 60 / math_questions) - (english_duration * 60 / english_questions) = 4 := by
  sorry

end time_difference_per_question_l2323_232392


namespace inequality_equivalence_l2323_232372

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / (2 - x) ≥ 0 ↔ (3 - x) / (x - 2) ≥ 0 :=
by sorry

end inequality_equivalence_l2323_232372


namespace abc_inequality_l2323_232373

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) :
  0 ≤ a*b + b*c + c*a - a*b*c ∧ a*b + b*c + c*a - a*b*c ≤ 2 := by
  sorry

end abc_inequality_l2323_232373


namespace line_slope_45_degrees_l2323_232319

theorem line_slope_45_degrees (y : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (⟨4, y⟩ ∈ line) ∧ 
    (⟨2, -3⟩ ∈ line) ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), ⟨x₁, y₁⟩ ∈ line → ⟨x₂, y₂⟩ ∈ line → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 1)) →
  y = -1 := by
sorry

end line_slope_45_degrees_l2323_232319


namespace min_dot_product_on_locus_l2323_232327

/-- The locus of point P -/
def locus (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) - abs x = 1

/-- A line through F(1,0) with slope k -/
def line_through_F (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

/-- Two points on the locus -/
structure LocusPoint where
  x : ℝ
  y : ℝ
  on_locus : locus x y

/-- The dot product of two vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

theorem min_dot_product_on_locus :
  ∀ (k : ℝ),
  k ≠ 0 →
  ∃ (A B D E : LocusPoint),
  line_through_F k A.x A.y ∧
  line_through_F k B.x B.y ∧
  line_through_F (-1/k) D.x D.y ∧
  line_through_F (-1/k) E.x E.y →
  ∀ (AD_dot_EB : ℝ),
  AD_dot_EB = dot_product (D.x - A.x) (D.y - A.y) (B.x - E.x) (B.y - E.y) →
  AD_dot_EB ≥ 16 :=
sorry

end min_dot_product_on_locus_l2323_232327


namespace geometric_sequence_properties_l2323_232321

/-- Geometric sequence with specified properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_arithmetic_mean : 2 * a 1 = a 2 + a 3)
  (h_a1 : a 1 = 1) :
  (∃ q : ℝ, q = -2 ∧ ∀ n : ℕ, a (n + 1) = q * a n) ∧
  (∀ n : ℕ, (Finset.range n).sum (fun i => (i + 1 : ℝ) * a (i + 1)) = (1 - (1 + 3 * n) * (-2)^n) / 9) :=
sorry

end geometric_sequence_properties_l2323_232321


namespace third_test_score_l2323_232396

def test1 : ℝ := 85
def test2 : ℝ := 79
def test4 : ℝ := 84
def test5 : ℝ := 85
def targetAverage : ℝ := 85
def numTests : ℕ := 5

theorem third_test_score (test3 : ℝ) : 
  (test1 + test2 + test3 + test4 + test5) / numTests = targetAverage → 
  test3 = 92 := by
sorry

end third_test_score_l2323_232396


namespace existence_of_intersection_point_l2323_232332

/-- Represents a circle on a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a point on a plane -/
def Point : Type := ℝ × ℝ

/-- Checks if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 > c.radius^2

/-- Represents a line on a plane -/
structure Line where
  point : Point
  direction : ℝ × ℝ
  non_zero : direction ≠ (0, 0)

/-- Checks if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  ∃ t : ℝ, is_outside (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2) c = false

/-- Main theorem: There exists a point outside both circles such that 
    any line passing through it intersects at least one of the circles -/
theorem existence_of_intersection_point (c1 c2 : Circle) 
  (h : ∀ p : Point, ¬(is_outside p c1 ∧ is_outside p c2)) : 
  ∃ p : Point, is_outside p c1 ∧ is_outside p c2 ∧ 
    ∀ l : Line, l.point = p → (intersects l c1 ∨ intersects l c2) := by
  sorry

end existence_of_intersection_point_l2323_232332


namespace transaction_handling_l2323_232323

/-- Problem: Transaction Handling --/
theorem transaction_handling 
  (mabel_transactions : ℕ)
  (anthony_percentage : ℚ)
  (cal_fraction : ℚ)
  (jade_additional : ℕ)
  (h1 : mabel_transactions = 90)
  (h2 : anthony_percentage = 11/10)
  (h3 : cal_fraction = 2/3)
  (h4 : jade_additional = 18) :
  let anthony_transactions := mabel_transactions * anthony_percentage
  let cal_transactions := anthony_transactions * cal_fraction
  let jade_transactions := cal_transactions + jade_additional
  jade_transactions = 84 := by
sorry

end transaction_handling_l2323_232323


namespace solve_for_a_l2323_232397

theorem solve_for_a : ∀ a : ℚ, 
  (∃ x : ℚ, (2 * a * x + 3) / (a - x) = 3 / 4 ∧ x = 1) → 
  a = -3 := by
sorry

end solve_for_a_l2323_232397


namespace expression_simplification_l2323_232322

theorem expression_simplification (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) (hac : a ≠ c) :
  (c^2 - a^2) / (c * a) - (c * a - c^2) / (c * a - a^2) = (2 * c^2 - a^2) / (c * a) := by
  sorry

end expression_simplification_l2323_232322


namespace segments_covered_by_q_at_most_q_plus_one_l2323_232339

/-- A half-line on the real number line -/
structure HalfLine where
  endpoint : ℝ
  direction : Bool -- true for right-infinite, false for left-infinite

/-- A configuration of half-lines on the real number line -/
def Configuration := List HalfLine

/-- A segment on the real number line -/
structure Segment where
  left : ℝ
  right : ℝ

/-- Count the number of half-lines covering a given segment -/
def coverCount (config : Configuration) (seg : Segment) : ℕ :=
  sorry

/-- The segments formed by the endpoints of the half-lines -/
def segments (config : Configuration) : List Segment :=
  sorry

/-- The segments covered by exactly q half-lines -/
def segmentsCoveredByQ (config : Configuration) (q : ℕ) : List Segment :=
  sorry

/-- The main theorem -/
theorem segments_covered_by_q_at_most_q_plus_one (config : Configuration) (q : ℕ) :
  (segmentsCoveredByQ config q).length ≤ q + 1 :=
  sorry

end segments_covered_by_q_at_most_q_plus_one_l2323_232339


namespace emma_popsicle_production_l2323_232398

/-- Emma's popsicle production problem -/
theorem emma_popsicle_production 
  (p h : ℝ) 
  (h_positive : h > 0)
  (p_def : p = 3/2 * h) :
  p * h - (p + 2) * (h - 3) = 7/2 * h + 6 := by
  sorry

end emma_popsicle_production_l2323_232398


namespace dannys_remaining_bottle_caps_l2323_232395

def initial_bottle_caps : ℕ := 91
def lost_bottle_caps : ℕ := 66

theorem dannys_remaining_bottle_caps :
  initial_bottle_caps - lost_bottle_caps = 25 := by
  sorry

end dannys_remaining_bottle_caps_l2323_232395


namespace even_function_implies_m_equals_one_l2323_232391

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = -x^2 + 2(m-1)x + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(m-1)*x + 3

theorem even_function_implies_m_equals_one :
  ∀ m : ℝ, IsEven (f m) → m = 1 := by
  sorry

end even_function_implies_m_equals_one_l2323_232391


namespace equation_holds_l2323_232303

theorem equation_holds (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end equation_holds_l2323_232303


namespace max_regions_50_lines_20_parallel_l2323_232326

/-- The maximum number of regions created by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of additional regions created by m parallel lines intersecting n non-parallel lines -/
def parallel_regions (m n : ℕ) : ℕ := m * (n + 1)

/-- The maximum number of regions created by n lines in a plane, where m of them are parallel -/
def max_regions_with_parallel (n m : ℕ) : ℕ :=
  max_regions (n - m) + parallel_regions m (n - m)

theorem max_regions_50_lines_20_parallel :
  max_regions_with_parallel 50 20 = 1086 := by
  sorry

end max_regions_50_lines_20_parallel_l2323_232326


namespace local_maximum_at_two_l2323_232375

/-- The function f(x) = x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem local_maximum_at_two (c : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), f c x ≤ f c 2) →
  (f_derivative c 2 = 0) →
  (∀ x ∈ Set.Ioo (2 - δ) 2, f_derivative c x > 0) →
  (∀ x ∈ Set.Ioo 2 (2 + δ), f_derivative c x < 0) →
  c = 6 := by
  sorry

end local_maximum_at_two_l2323_232375


namespace factorization_difference_of_squares_l2323_232315

theorem factorization_difference_of_squares (a : ℝ) : 
  a^2 - 9 = (a + 3) * (a - 3) := by sorry

#check factorization_difference_of_squares

end factorization_difference_of_squares_l2323_232315


namespace no_solution_exists_l2323_232358

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_with_B (B : ℕ) : ℕ := 12345670 + B

theorem no_solution_exists :
  ¬ ∃ B : ℕ, is_digit B ∧ 
    (number_with_B B).mod 2 = 0 ∧
    (number_with_B B).mod 5 = 0 ∧
    (number_with_B B).mod 11 = 0 :=
sorry

end no_solution_exists_l2323_232358


namespace particle_probability_l2323_232361

/-- The probability of a particle reaching (0,0) from (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * P (x-1) y + (1/3) * P x (y-1) + (1/3) * P (x-1) (y-1)

theorem particle_probability :
  P 5 5 = 793 / 6561 :=
sorry

end particle_probability_l2323_232361


namespace sam_seashells_l2323_232314

/-- Given that Sam found 35 seashells and gave 18 to Joan, prove that he now has 17 seashells. -/
theorem sam_seashells (initial : ℕ) (given_away : ℕ) (h1 : initial = 35) (h2 : given_away = 18) :
  initial - given_away = 17 := by
  sorry

end sam_seashells_l2323_232314


namespace absolute_value_eq_four_sum_of_absolute_values_min_value_of_sum_min_value_is_three_l2323_232369

-- Problem 1
theorem absolute_value_eq_four (a : ℝ) : 
  |a + 2| = 4 ↔ a = -6 ∨ a = 2 := by sorry

-- Problem 2
theorem sum_of_absolute_values (a : ℝ) :
  -4 < a ∧ a < 2 → |a + 4| + |a - 2| = 6 := by sorry

-- Problem 3
theorem min_value_of_sum (a : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ |a - 1| + |a + 2|) ↔ -2 ≤ a ∧ a ≤ 1 := by sorry

theorem min_value_is_three (a : ℝ) :
  -2 ≤ a ∧ a ≤ 1 → |a - 1| + |a + 2| = 3 := by sorry

end absolute_value_eq_four_sum_of_absolute_values_min_value_of_sum_min_value_is_three_l2323_232369


namespace dorchester_daily_pay_l2323_232383

/-- Represents Dorchester's earnings at the puppy wash -/
structure PuppyWashEarnings where
  dailyPay : ℝ
  puppyWashRate : ℝ
  puppiesWashed : ℕ
  totalEarnings : ℝ

/-- Dorchester's earnings satisfy the given conditions -/
def dorchesterEarnings : PuppyWashEarnings where
  dailyPay := 40
  puppyWashRate := 2.25
  puppiesWashed := 16
  totalEarnings := 76

/-- Theorem: Dorchester's daily pay is $40 given the conditions -/
theorem dorchester_daily_pay :
  dorchesterEarnings.dailyPay = 40 ∧
  dorchesterEarnings.totalEarnings = dorchesterEarnings.dailyPay +
    dorchesterEarnings.puppyWashRate * dorchesterEarnings.puppiesWashed :=
by sorry

end dorchester_daily_pay_l2323_232383


namespace inequality_proof_l2323_232374

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x * y + y * z + z * x = 1) : 
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5 / 2 := by
  sorry

end inequality_proof_l2323_232374


namespace g_of_3_l2323_232380

def g (x : ℝ) : ℝ := 5 * x^4 + 4 * x^3 - 7 * x^2 + 3 * x - 2

theorem g_of_3 : g 3 = 401 := by
  sorry

end g_of_3_l2323_232380


namespace largest_package_size_l2323_232387

/-- The largest possible number of markers in a package given that Lucy bought 30 markers, 
    Mia bought 45 markers, and Noah bought 75 markers. -/
theorem largest_package_size : Nat.gcd 30 (Nat.gcd 45 75) = 15 := by
  sorry

end largest_package_size_l2323_232387


namespace sum_of_polynomials_l2323_232379

/-- Given polynomial functions f, g, h, and j, prove their sum equals 3x^2 + x - 2 -/
theorem sum_of_polynomials (x : ℝ) : 
  let f := fun (x : ℝ) => 2 * x^2 - 4 * x + 3
  let g := fun (x : ℝ) => -3 * x^2 + 7 * x - 6
  let h := fun (x : ℝ) => 3 * x^2 - 3 * x + 2
  let j := fun (x : ℝ) => x^2 + x - 1
  f x + g x + h x + j x = 3 * x^2 + x - 2 := by
  sorry

end sum_of_polynomials_l2323_232379


namespace inequality_solution_set_l2323_232359

theorem inequality_solution_set (x : ℝ) : 
  (2 * x^2 / (x + 2) ≥ 3 / (x - 2) + 6 / 4) ↔ 
  (x < -2 ∨ (-2 < x ∧ x < (1 - Real.sqrt 129) / 8) ∨ 
   (2 < x ∧ x < 3) ∨ 
   ((1 + Real.sqrt 129) / 8 < x)) :=
by sorry

end inequality_solution_set_l2323_232359


namespace solution_in_quadrant_III_l2323_232324

/-- 
Given a system of equations x - y = 4 and cx + y = 5, where c is a constant,
this theorem states that the solution (x, y) is in Quadrant III 
(i.e., x < 0 and y < 0) if and only if c < -1.
-/
theorem solution_in_quadrant_III (c : ℝ) :
  (∃ x y : ℝ, x - y = 4 ∧ c * x + y = 5 ∧ x < 0 ∧ y < 0) ↔ c < -1 :=
by sorry

end solution_in_quadrant_III_l2323_232324


namespace line_passes_through_fixed_point_l2323_232309

/-- The line equation passing through a fixed point for all values of a -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 2) * x + (a + 1) * y + 6 = 0

/-- The fixed point coordinates -/
def fixed_point : ℝ × ℝ := (2, -2)

/-- Theorem stating that the line passes through the fixed point for all a -/
theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), line_equation a (fixed_point.1) (fixed_point.2) :=
sorry

end line_passes_through_fixed_point_l2323_232309


namespace sum_of_coefficients_equals_value_at_one_l2323_232388

/-- The polynomial for which we want to find the sum of coefficients -/
def p (x : ℝ) : ℝ := 3*(x^8 - 2*x^5 + 4*x^3 - 6) - 5*(x^4 - 3*x + 7) + 2*(x^6 - 5)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_equals_value_at_one :
  p 1 = -42 := by sorry

end sum_of_coefficients_equals_value_at_one_l2323_232388


namespace shoes_savings_theorem_l2323_232346

/-- The number of weekends needed to save for shoes -/
def weekends_needed (shoe_cost : ℕ) (saved : ℕ) (earnings_per_lawn : ℕ) (lawns_per_weekend : ℕ) : ℕ :=
  let remaining := shoe_cost - saved
  let earnings_per_weekend := earnings_per_lawn * lawns_per_weekend
  (remaining + earnings_per_weekend - 1) / earnings_per_weekend

theorem shoes_savings_theorem (shoe_cost saved earnings_per_lawn lawns_per_weekend : ℕ) 
  (h1 : shoe_cost = 120)
  (h2 : saved = 30)
  (h3 : earnings_per_lawn = 5)
  (h4 : lawns_per_weekend = 3) :
  weekends_needed shoe_cost saved earnings_per_lawn lawns_per_weekend = 6 := by
  sorry

end shoes_savings_theorem_l2323_232346


namespace gcd_of_B_is_five_l2323_232367

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_five : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
  sorry

end gcd_of_B_is_five_l2323_232367


namespace aunt_gave_109_l2323_232362

/-- The amount of money Paula's aunt gave her -/
def money_from_aunt (shirt_cost shirt_count pant_cost money_left : ℕ) : ℕ :=
  shirt_cost * shirt_count + pant_cost + money_left

/-- Proof that Paula's aunt gave her $109 -/
theorem aunt_gave_109 :
  money_from_aunt 11 2 13 74 = 109 := by
  sorry

end aunt_gave_109_l2323_232362


namespace claire_shirts_count_l2323_232386

theorem claire_shirts_count :
  ∀ (brian_shirts andrew_shirts steven_shirts claire_shirts : ℕ),
    brian_shirts = 3 →
    andrew_shirts = 6 * brian_shirts →
    steven_shirts = 4 * andrew_shirts →
    claire_shirts = 5 * steven_shirts →
    claire_shirts = 360 := by
  sorry

end claire_shirts_count_l2323_232386


namespace combined_weight_theorem_l2323_232365

def leo_weight : ℝ := 80
def weight_gain : ℝ := 10

theorem combined_weight_theorem (kendra_weight : ℝ) 
  (h : leo_weight + weight_gain = 1.5 * kendra_weight) :
  leo_weight + kendra_weight = 140 := by
sorry

end combined_weight_theorem_l2323_232365


namespace expression_evaluation_l2323_232316

theorem expression_evaluation (y : ℝ) (h : y ≠ 1/2) :
  (2*y - 1)^0 / (6⁻¹ + 2⁻¹) = 3/2 := by
  sorry

end expression_evaluation_l2323_232316


namespace composite_sum_of_powers_l2323_232350

theorem composite_sum_of_powers (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x * y = a^2000 + b^2000 + c^2000 + d^2000 :=
by
  sorry

end composite_sum_of_powers_l2323_232350


namespace wall_width_calculation_l2323_232306

/-- Calculates the width of a wall given brick dimensions and wall specifications -/
theorem wall_width_calculation (brick_length brick_width brick_height : ℝ)
  (wall_length wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_length = 25 →
  wall_height = 2 →
  num_bricks = 25000 →
  ∃ (wall_width : ℝ), wall_width = 0.75 :=
by sorry

end wall_width_calculation_l2323_232306
