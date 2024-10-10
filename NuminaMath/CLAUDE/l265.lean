import Mathlib

namespace parabola_directrix_l265_26596

/-- For a parabola with equation y = 2x^2, its directrix has the equation y = -1/8 -/
theorem parabola_directrix (x y : ℝ) :
  y = 2 * x^2 → (∃ (k : ℝ), y = k ∧ k = -1/8) :=
by sorry

end parabola_directrix_l265_26596


namespace probability_four_white_balls_l265_26534

/-- The probability of drawing 4 white balls from a box containing 7 white and 8 black balls -/
theorem probability_four_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) :
  total_balls = white_balls + black_balls →
  total_balls = 15 →
  white_balls = 7 →
  black_balls = 8 →
  drawn_balls = 4 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 39 :=
by sorry

end probability_four_white_balls_l265_26534


namespace expression_value_l265_26548

theorem expression_value :
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 1
  (x^2 * y * z - x * y * z^2) = 6 := by
sorry

end expression_value_l265_26548


namespace three_roots_symmetric_about_two_l265_26518

/-- A function f: ℝ → ℝ that satisfies f(2+x) = f(2-x) for all x ∈ ℝ -/
def symmetric_about_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

/-- The set of roots of f -/
def roots (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = 0}

theorem three_roots_symmetric_about_two (f : ℝ → ℝ) :
  symmetric_about_two f →
  (∃ a b : ℝ, roots f = {0, a, b} ∧ a ≠ b ∧ a ≠ 0 ∧ b ≠ 0) →
  roots f = {0, 2, 4} :=
sorry

end three_roots_symmetric_about_two_l265_26518


namespace square_has_most_symmetry_axes_l265_26510

/-- The number of axes of symmetry for a square -/
def square_symmetry_axes : ℕ := 4

/-- The number of axes of symmetry for an equilateral triangle -/
def equilateral_triangle_symmetry_axes : ℕ := 3

/-- The number of axes of symmetry for an isosceles triangle -/
def isosceles_triangle_symmetry_axes : ℕ := 1

/-- The number of axes of symmetry for an isosceles trapezoid -/
def isosceles_trapezoid_symmetry_axes : ℕ := 1

/-- The shape with the most axes of symmetry -/
def shape_with_most_symmetry_axes : ℕ := square_symmetry_axes

theorem square_has_most_symmetry_axes :
  shape_with_most_symmetry_axes = square_symmetry_axes ∧
  shape_with_most_symmetry_axes > equilateral_triangle_symmetry_axes ∧
  shape_with_most_symmetry_axes > isosceles_triangle_symmetry_axes ∧
  shape_with_most_symmetry_axes > isosceles_trapezoid_symmetry_axes :=
by sorry

end square_has_most_symmetry_axes_l265_26510


namespace min_value_theorem_l265_26538

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9 * x + 1 / x^6 ≥ 10 ∧ (9 * x + 1 / x^6 = 10 ↔ x = 1) := by
  sorry

end min_value_theorem_l265_26538


namespace solve_cake_baking_l265_26502

def cake_baking_problem (jane_rate roy_rate : ℚ) (jane_remaining_time : ℚ) (jane_remaining_work : ℚ) : Prop :=
  let combined_rate := jane_rate + roy_rate
  let total_work := 1
  ∃ t : ℚ, 
    t > 0 ∧
    combined_rate * t + jane_remaining_work = total_work ∧
    jane_rate * jane_remaining_time = jane_remaining_work ∧
    t = 2

theorem solve_cake_baking :
  cake_baking_problem (1/4) (1/5) (2/5) (1/10) :=
sorry

end solve_cake_baking_l265_26502


namespace vector_parallel_problem_l265_26526

/-- Two 2D vectors are parallel if their components are proportional -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_parallel_problem (m : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-3, 0)
  parallel (2 • a + b) (a - m • b) →
  m = -1/2 := by
  sorry

end vector_parallel_problem_l265_26526


namespace fraction_equation_solution_l265_26540

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 2) ∧ x = -1/3 := by
  sorry

end fraction_equation_solution_l265_26540


namespace gcf_of_72_and_90_l265_26504

theorem gcf_of_72_and_90 : Nat.gcd 72 90 = 18 := by
  sorry

end gcf_of_72_and_90_l265_26504


namespace sales_growth_rate_l265_26588

theorem sales_growth_rate (initial_sales final_sales : ℝ) 
  (h1 : initial_sales = 2000000)
  (h2 : final_sales = 2880000)
  (h3 : ∃ r : ℝ, initial_sales * (1 + r)^2 = final_sales) :
  ∃ r : ℝ, initial_sales * (1 + r)^2 = final_sales ∧ r = 0.2 :=
sorry

end sales_growth_rate_l265_26588


namespace final_state_correct_l265_26511

/-- Represents the state of variables A, B, and C -/
structure State :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)

/-- Executes the assignment statements and returns the final state -/
def executeAssignments : State := by
  let s1 : State := { A := 0, B := 0, C := 2 }  -- C ← 2
  let s2 : State := { A := s1.A, B := 1, C := s1.C }  -- B ← 1
  let s3 : State := { A := 2, B := s2.B, C := s2.C }  -- A ← 2
  exact s3

/-- Theorem stating that the final values of A, B, and C are 2, 1, and 2 respectively -/
theorem final_state_correct : 
  let final := executeAssignments
  final.A = 2 ∧ final.B = 1 ∧ final.C = 2 := by
  sorry

end final_state_correct_l265_26511


namespace forgotten_angles_sum_l265_26506

theorem forgotten_angles_sum (n : ℕ) (partial_sum : ℝ) : 
  n ≥ 3 → 
  partial_sum = 2797 → 
  (n - 2) * 180 - partial_sum = 83 :=
by sorry

end forgotten_angles_sum_l265_26506


namespace exam_success_probability_l265_26586

theorem exam_success_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/2) (h2 : p2 = 1/4) (h3 : p3 = 1/5) :
  let at_least_two_success := 
    p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3 + p1 * p2 * p3
  at_least_two_success = 9/40 := by
  sorry

end exam_success_probability_l265_26586


namespace shifted_line_equation_l265_26592

/-- Given a line with equation y = -2x, shifting it one unit upwards
    results in the equation y = -2x + 1 -/
theorem shifted_line_equation (x y : ℝ) :
  (y = -2 * x) → (y + 1 = -2 * x + 1) := by sorry

end shifted_line_equation_l265_26592


namespace encryption_theorem_l265_26578

/-- Represents the encryption table --/
def encryption_table : Fin 16 → Fin 16 := sorry

/-- Applies the encryption once to a string of 16 characters --/
def apply_encryption (s : String) : String := sorry

/-- Applies the encryption n times to a string --/
def apply_encryption_n_times (s : String) (n : ℕ) : String := sorry

/-- The last three characters of a string --/
def last_three (s : String) : String := sorry

theorem encryption_theorem :
  ∀ s : String,
  last_three s = "уао" →
  apply_encryption_n_times (apply_encryption s) 2014 = s →
  ∃ t : String, last_three t = "чку" ∧ apply_encryption_n_times t 2015 = s :=
sorry

end encryption_theorem_l265_26578


namespace workshop_efficiency_l265_26528

theorem workshop_efficiency (x : ℝ) (h : x > 0) : 
  (3000 / x) - (3000 / (2.5 * x)) = (3 / 2) := by
  sorry

end workshop_efficiency_l265_26528


namespace negation_of_neither_even_l265_26579

theorem negation_of_neither_even (a b : ℤ) : 
  ¬(¬(Even a) ∧ ¬(Even b)) ↔ (Even a ∨ Even b) := by
  sorry

end negation_of_neither_even_l265_26579


namespace uniform_price_l265_26512

def full_year_salary : ℕ := 500
def months_worked : ℕ := 9
def payment_received : ℕ := 300

theorem uniform_price : 
  ∃ (uniform_price : ℕ), 
    (uniform_price + payment_received = (months_worked * full_year_salary) / 12) ∧
    uniform_price = 75 := by
  sorry

end uniform_price_l265_26512


namespace cylinder_packing_l265_26536

theorem cylinder_packing (n : ℕ) (d : ℝ) (h : d > 0) :
  let rectangular_width := 8 * d
  let hexagonal_width := n * d * (Real.sqrt 3 / 2) + d
  40 < n → n < 42 →
  hexagonal_width < rectangular_width ∧
  hexagonal_width > rectangular_width - d :=
by sorry

end cylinder_packing_l265_26536


namespace quadratic_root_difference_l265_26569

theorem quadratic_root_difference (x : ℝ) : 
  let roots := {r : ℝ | r^2 - 7*r + 11 = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ ≠ r₂ ∧ |r₁ - r₂| = Real.sqrt 5 :=
by sorry

end quadratic_root_difference_l265_26569


namespace divisibility_by_power_of_two_l265_26535

theorem divisibility_by_power_of_two (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := by
  sorry

end divisibility_by_power_of_two_l265_26535


namespace tom_candy_l265_26570

def candy_problem (initial : ℕ) (from_friend : ℕ) (bought : ℕ) : Prop :=
  initial + from_friend + bought = 19

theorem tom_candy : candy_problem 2 7 10 := by sorry

end tom_candy_l265_26570


namespace max_y_value_l265_26516

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -1) : y ≤ 2 := by
  sorry

end max_y_value_l265_26516


namespace cube_root_equation_solution_l265_26551

theorem cube_root_equation_solution :
  ∃! x : ℝ, (4 - x / 3) ^ (1/3 : ℝ) = -2 :=
by
  -- Proof goes here
  sorry

end cube_root_equation_solution_l265_26551


namespace division_problem_l265_26547

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by
  sorry

end division_problem_l265_26547


namespace speed_in_still_water_l265_26576

theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 26) 
  (h2 : downstream_speed = 30) : 
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end speed_in_still_water_l265_26576


namespace special_sequence_250th_term_l265_26572

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- Predicate to check if a number is a multiple of 3 -/
def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 3 * m

/-- The sequence of positive integers omitting perfect squares and multiples of 3 -/
def special_sequence : ℕ → ℕ :=
  sorry

/-- The 250th term of the special sequence is 350 -/
theorem special_sequence_250th_term :
  special_sequence 250 = 350 := by
  sorry

end special_sequence_250th_term_l265_26572


namespace exist_n_points_with_integer_distances_l265_26532

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Main theorem statement -/
theorem exist_n_points_with_integer_distances (n : ℕ) (h : n ≥ 2) :
  ∃ (points : Fin n → Point),
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬areCollinear (points i) (points j) (points k)) ∧
    (∀ (i j : Fin n), i ≠ j → ∃ (d : ℤ), squaredDistance (points i) (points j) = d^2) :=
by sorry

end exist_n_points_with_integer_distances_l265_26532


namespace min_value_of_expression_l265_26581

theorem min_value_of_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x*y/z + z*x/y + y*z/x) * (x/(y*z) + y/(z*x) + z/(x*y)) ≥ 9 := by
  sorry

end min_value_of_expression_l265_26581


namespace max_value_on_ellipse_l265_26554

def ellipse (x y : ℝ) : Prop := y^2/16 + x^2/4 = 1

theorem max_value_on_ellipse :
  ∃ (M : ℝ), ∀ (x y : ℝ), ellipse x y → |2*Real.sqrt 3*x + y - 1| ≤ M ∧
  ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧ |2*Real.sqrt 3*x₀ + y₀ - 1| = M :=
sorry

end max_value_on_ellipse_l265_26554


namespace total_is_300_l265_26521

/-- The number of pennies thrown by Rachelle, Gretchen, and Rocky -/
def penny_throwing (rachelle gretchen rocky : ℕ) : Prop :=
  rachelle = 180 ∧ 
  gretchen = rachelle / 2 ∧ 
  rocky = gretchen / 3

/-- The total number of pennies thrown by all three -/
def total_pennies (rachelle gretchen rocky : ℕ) : ℕ :=
  rachelle + gretchen + rocky

/-- Theorem stating that the total number of pennies thrown is 300 -/
theorem total_is_300 (rachelle gretchen rocky : ℕ) : 
  penny_throwing rachelle gretchen rocky → total_pennies rachelle gretchen rocky = 300 :=
by
  sorry

end total_is_300_l265_26521


namespace sqrt_two_irrational_l265_26522

theorem sqrt_two_irrational :
  ∃ (x : ℝ), Irrational x ∧ (x = Real.sqrt 2) ∧
  (∀ y : ℝ, (y = 1/3 ∨ y = 3.1415 ∨ y = -5) → ¬Irrational y) :=
by sorry

end sqrt_two_irrational_l265_26522


namespace angle_A_measure_l265_26565

theorem angle_A_measure (A B C : ℝ) (a b c : ℝ) : 
  a = Real.sqrt 7 →
  b = 3 →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  A = π / 3 := by
  sorry

end angle_A_measure_l265_26565


namespace project_completion_time_l265_26593

theorem project_completion_time (a_time b_time total_time : ℕ) 
  (h1 : a_time = 20)
  (h2 : b_time = 20)
  (h3 : total_time = 15) :
  ∃ (x : ℕ), 
    (1 : ℚ) / a_time + (1 : ℚ) / b_time = (1 : ℚ) / (total_time - x) + 
    ((1 : ℚ) / b_time) * (x : ℚ) / total_time ∧ 
    x = 10 := by
  sorry

#check project_completion_time

end project_completion_time_l265_26593


namespace arithmetic_geometric_sequence_sum_l265_26574

theorem arithmetic_geometric_sequence_sum (a b c d : ℝ) : 
  (∃ k : ℝ, a = 6 + k ∧ b = 6 + 2*k ∧ 48 = 6 + 3*k) →  -- arithmetic sequence condition
  (∃ q : ℝ, c = 6*q ∧ d = 6*q^2 ∧ 48 = 6*q^3) →        -- geometric sequence condition
  a + b + c + d = 111 := by
sorry

end arithmetic_geometric_sequence_sum_l265_26574


namespace quadratic_equation_conversion_l265_26507

theorem quadratic_equation_conversion :
  ∀ x : ℝ, (x - 3)^2 = 4 ↔ x^2 - 6*x + 5 = 0 :=
by sorry

end quadratic_equation_conversion_l265_26507


namespace four_lighthouses_cover_plane_l265_26545

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a 90-degree angle in the plane -/
inductive Quadrant
  | NE
  | SE
  | SW
  | NW

/-- Represents a lighthouse with its position and illumination direction -/
structure Lighthouse where
  position : Point
  direction : Quadrant

/-- Checks if a point is illuminated by a lighthouse -/
def isIlluminated (p : Point) (l : Lighthouse) : Prop :=
  sorry

/-- The main theorem: four lighthouses can illuminate the entire plane -/
theorem four_lighthouses_cover_plane (a b c d : Point) :
  ∃ (la lb lc ld : Lighthouse),
    la.position = a ∧ lb.position = b ∧ lc.position = c ∧ ld.position = d ∧
    ∀ p : Point, isIlluminated p la ∨ isIlluminated p lb ∨ isIlluminated p lc ∨ isIlluminated p ld :=
  sorry


end four_lighthouses_cover_plane_l265_26545


namespace no_quadratic_trinomial_with_odd_coeffs_and_2022th_root_l265_26584

theorem no_quadratic_trinomial_with_odd_coeffs_and_2022th_root :
  ¬ ∃ (a b c : ℤ), 
    (Odd a ∧ Odd b ∧ Odd c) ∧ 
    (a * (1 / 2022)^2 + b * (1 / 2022) + c = 0) := by
  sorry

end no_quadratic_trinomial_with_odd_coeffs_and_2022th_root_l265_26584


namespace bread_leftover_l265_26587

theorem bread_leftover (total_length : Real) (jimin_eats_cm : Real) (taehyung_eats_m : Real) :
  total_length = 30 ∧ jimin_eats_cm = 150 ∧ taehyung_eats_m = 1.65 →
  total_length - (jimin_eats_cm / 100 + taehyung_eats_m) = 26.85 := by
  sorry

end bread_leftover_l265_26587


namespace student_path_probability_l265_26553

/-- Represents the number of paths between two points given the number of eastward and southward moves -/
def num_paths (east south : ℕ) : ℕ := Nat.choose (east + south) east

/-- Represents the total number of paths from A to B -/
def total_paths : ℕ := num_paths 6 5

/-- Represents the number of paths from A to B that pass through C and D -/
def paths_through_C_and_D : ℕ := num_paths 3 2 * num_paths 2 1 * num_paths 1 2

/-- The probability of choosing a specific path given the number of moves -/
def path_probability (moves : ℕ) : ℚ := (1 / 2) ^ moves

theorem student_path_probability : 
  (paths_through_C_and_D : ℚ) / total_paths = 15 / 77 := by sorry

end student_path_probability_l265_26553


namespace special_numbers_count_l265_26558

/-- Sum of digits of a positive integer -/
def digit_sum (x : ℕ+) : ℕ := sorry

/-- Counts the number of three-digit positive integers satisfying the condition -/
def count_special_numbers : ℕ := sorry

/-- Main theorem -/
theorem special_numbers_count :
  count_special_numbers = 14 := by sorry

end special_numbers_count_l265_26558


namespace a_2021_eq_6_l265_26577

def a : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => 
    if n % 3 = 0 then a (n / 3)
    else a (n / 3) + 1

theorem a_2021_eq_6 : a 2021 = 6 := by
  sorry

end a_2021_eq_6_l265_26577


namespace sum_of_even_integers_202_to_300_l265_26530

def sum_of_first_n_even_integers (n : ℕ) : ℕ := n * (n + 1)

def count_even_numbers_in_range (first last : ℕ) : ℕ :=
  (last - first) / 2 + 1

def sum_of_arithmetic_sequence (n first last : ℕ) : ℕ :=
  n / 2 * (first + last)

theorem sum_of_even_integers_202_to_300 
  (h : sum_of_first_n_even_integers 50 = 2550) :
  sum_of_arithmetic_sequence 
    (count_even_numbers_in_range 202 300) 
    202 
    300 = 12550 := by
  sorry

end sum_of_even_integers_202_to_300_l265_26530


namespace mat_length_approximation_l265_26571

/-- Represents the setup of a circular table with place mats -/
structure TableSetup where
  tableRadius : ℝ
  numMats : ℕ
  matWidth : ℝ

/-- Calculates the length of place mats given a table setup -/
def calculateMatLength (setup : TableSetup) : ℝ :=
  sorry

/-- Theorem stating that for the given setup, the mat length is approximately 3.9308 meters -/
theorem mat_length_approximation (setup : TableSetup) 
  (h1 : setup.tableRadius = 6)
  (h2 : setup.numMats = 8)
  (h3 : setup.matWidth = 1.5) :
  abs (calculateMatLength setup - 3.9308) < 0.0001 := by
  sorry

end mat_length_approximation_l265_26571


namespace janessa_cards_ordered_l265_26583

/-- The number of cards Janessa ordered from eBay --/
def cards_ordered (initial_cards : ℕ) (father_cards : ℕ) (thrown_cards : ℕ) (given_cards : ℕ) (kept_cards : ℕ) : ℕ :=
  given_cards + kept_cards - (initial_cards + father_cards) + thrown_cards

theorem janessa_cards_ordered :
  cards_ordered 4 13 4 29 20 = 36 := by
  sorry

end janessa_cards_ordered_l265_26583


namespace arithmetic_problem_l265_26537

theorem arithmetic_problem : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end arithmetic_problem_l265_26537


namespace triangle_roots_range_l265_26515

theorem triangle_roots_range (m : ℝ) : 
  (∃ x y z : ℝ, (x - 1) * (x^2 - 2*x + m) = 0 ∧ 
                (y - 1) * (y^2 - 2*y + m) = 0 ∧ 
                (z - 1) * (z^2 - 2*z + m) = 0 ∧
                x + y > z ∧ y + z > x ∧ z + x > y) ↔ 
  (3/4 < m ∧ m ≤ 1) :=
sorry

end triangle_roots_range_l265_26515


namespace cells_after_three_divisions_l265_26561

/-- The number of cells after n divisions, given that each division doubles the number of cells -/
def num_cells (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of cells after 3 divisions is 8 -/
theorem cells_after_three_divisions : num_cells 3 = 8 := by
  sorry

end cells_after_three_divisions_l265_26561


namespace fraction_addition_l265_26560

theorem fraction_addition : (3 : ℚ) / 5 + (2 : ℚ) / 5 = 1 := by sorry

end fraction_addition_l265_26560


namespace money_problem_l265_26568

/-- Given three people A, B, and C with certain amounts of money, 
    prove that A and C together have 300 rupees. -/
theorem money_problem (a b c : ℕ) : 
  a + b + c = 700 →
  b + c = 600 →
  c = 200 →
  a + c = 300 := by
  sorry

end money_problem_l265_26568


namespace second_chapter_length_l265_26585

theorem second_chapter_length (total_pages first_chapter_pages : ℕ) 
  (h1 : total_pages = 94)
  (h2 : first_chapter_pages = 48) :
  total_pages - first_chapter_pages = 46 := by
  sorry

end second_chapter_length_l265_26585


namespace sally_peaches_count_l265_26575

def initial_peaches : ℕ := 13
def first_orchard_peaches : ℕ := 55

def peaches_after_giving : ℕ := initial_peaches - (initial_peaches / 2)
def peaches_after_first_orchard : ℕ := peaches_after_giving + first_orchard_peaches
def second_orchard_peaches : ℕ := 2 * first_orchard_peaches
def total_peaches : ℕ := peaches_after_first_orchard + second_orchard_peaches

theorem sally_peaches_count : total_peaches = 172 := by
  sorry

end sally_peaches_count_l265_26575


namespace problem_1_problem_2_l265_26591

-- Problem 1
theorem problem_1 : 
  2 * Real.cos (π / 4) + (3 - Real.pi) ^ 0 - |2 - Real.sqrt 8| - (-1/3)⁻¹ = 6 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : 
  let x : ℝ := Real.sqrt 27 + |-2| - 3 * Real.tan (π / 3)
  ((x^2 - 1) / (x^2 - 2*x + 1) - 1 / (x - 1)) / ((x + 2) / (x - 1)) = 1/2 := by
  sorry

end problem_1_problem_2_l265_26591


namespace equation_solution_l265_26524

-- Define the equation
def equation (m : ℝ) (x : ℝ) : Prop :=
  (2*m - 6) * x^(|m| - 2) = m^2

-- State the theorem
theorem equation_solution :
  ∃ (m : ℝ), ∀ (x : ℝ), equation m x ↔ x = -3/4 :=
sorry

end equation_solution_l265_26524


namespace robot_race_track_length_l265_26542

/-- Represents the race between three robots A, B, and C --/
structure RobotRace where
  track_length : ℝ
  va : ℝ
  vb : ℝ
  vc : ℝ

/-- The conditions of the race --/
def race_conditions (race : RobotRace) : Prop :=
  race.track_length > 0 ∧
  race.va > 0 ∧ race.vb > 0 ∧ race.vc > 0 ∧
  race.track_length / race.va = (race.track_length - 1) / race.vb ∧
  race.track_length / race.va = (race.track_length - 2) / race.vc ∧
  race.track_length / race.vb = (race.track_length - 1.01) / race.vc

theorem robot_race_track_length (race : RobotRace) :
  race_conditions race → race.track_length = 101 := by
  sorry

#check robot_race_track_length

end robot_race_track_length_l265_26542


namespace quadratic_is_square_of_binomial_l265_26519

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 18 * x + 9 = (r * x + s)^2) → a = 9 := by
sorry

end quadratic_is_square_of_binomial_l265_26519


namespace complex_arithmetic_equality_l265_26567

theorem complex_arithmetic_equality : ((-1 : ℤ) ^ 2024) + (-10 : ℤ) / (1 / 2 : ℚ) * 2 + (2 - (-3 : ℤ) ^ 3) = -10 := by
  sorry

end complex_arithmetic_equality_l265_26567


namespace tetrahedron_volume_EFGH_l265_26597

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (EF EG EH FG FH GH : ℝ) : ℝ :=
  sorry

/-- Theorem: The volume of tetrahedron EFGH with given edge lengths is √3/2 -/
theorem tetrahedron_volume_EFGH :
  tetrahedron_volume 5 (3 * Real.sqrt 2) (2 * Real.sqrt 3) 4 (Real.sqrt 37) 3 = Real.sqrt 3 / 2 :=
sorry

end tetrahedron_volume_EFGH_l265_26597


namespace complex_modulus_problem_l265_26505

theorem complex_modulus_problem (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : 
  Complex.abs z = 5 := by
sorry

end complex_modulus_problem_l265_26505


namespace justins_dogs_l265_26566

theorem justins_dogs (camden_dogs rico_dogs justin_dogs : ℕ) : 
  camden_dogs = (3 * rico_dogs) / 4 →
  rico_dogs = justin_dogs + 10 →
  camden_dogs * 4 = 72 →
  justin_dogs = 14 := by
  sorry

end justins_dogs_l265_26566


namespace correct_batteries_in_toys_l265_26539

/-- The number of batteries Tom used in his toys -/
def batteries_in_toys : ℕ := 15

/-- The number of batteries Tom used in his flashlights -/
def batteries_in_flashlights : ℕ := 2

/-- Theorem stating that the number of batteries in toys is correct -/
theorem correct_batteries_in_toys :
  batteries_in_toys = batteries_in_flashlights + 13 :=
by sorry

end correct_batteries_in_toys_l265_26539


namespace reverse_product_inequality_l265_26500

/-- Reverses the digits and decimal point of a positive real number with finitely many decimal places -/
noncomputable def reverse (x : ℝ) : ℝ := sorry

/-- Predicate to check if a real number has finitely many decimal places -/
def has_finite_decimals (x : ℝ) : Prop := sorry

/-- The main theorem to be proved -/
theorem reverse_product_inequality {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (hfx : has_finite_decimals x) (hfy : has_finite_decimals y) : 
  reverse (x * y) ≤ 10 * reverse x * reverse y := by sorry

end reverse_product_inequality_l265_26500


namespace cube_volume_surface_area_l265_26513

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 6*x ∧ 6*s^2 = 2*x) → x = 1/972 := by
  sorry

end cube_volume_surface_area_l265_26513


namespace b_plus_c_equals_nine_l265_26580

theorem b_plus_c_equals_nine (a b c d : ℤ) 
  (h1 : a + b = 11) 
  (h2 : c + d = 3) 
  (h3 : a + d = 5) : 
  b + c = 9 := by
  sorry

end b_plus_c_equals_nine_l265_26580


namespace absolute_value_equation_solution_l265_26503

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x + 5| = 3 * x - 2 :=
by
  use 7/2
  sorry

end absolute_value_equation_solution_l265_26503


namespace pencils_leftover_l265_26531

theorem pencils_leftover : Int.mod 33333332 8 = 4 := by
  sorry

end pencils_leftover_l265_26531


namespace complex_square_equality_l265_26529

theorem complex_square_equality (a b : ℕ+) :
  (a + b * Complex.I) ^ 2 = 3 + 4 * Complex.I →
  a + b * Complex.I = 2 + Complex.I := by
sorry

end complex_square_equality_l265_26529


namespace equation_equivalence_l265_26595

theorem equation_equivalence : ∃ (b c : ℝ), 
  (∀ x : ℝ, |x - 4| = 3 ↔ x = 1 ∨ x = 7) →
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 7) →
  b = -8 ∧ c = 7 := by
sorry

end equation_equivalence_l265_26595


namespace quadratic_roots_sum_l265_26598

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 1/3) → 
  a + b = -14 := by
sorry

end quadratic_roots_sum_l265_26598


namespace cherry_pricing_and_profit_l265_26562

/-- Represents the cost and quantity of cherries --/
structure CherryData where
  yellow_cost : ℝ
  red_cost : ℝ
  yellow_quantity : ℝ
  red_quantity : ℝ

/-- Represents the sales data for red light cherries --/
structure SalesData where
  week1_price : ℝ
  week1_quantity : ℝ
  week2_price_decrease : ℝ
  week2_quantity : ℝ
  week3_discount : ℝ

/-- Theorem stating the cost price of red light cherries and minimum value of m --/
theorem cherry_pricing_and_profit (data : CherryData) (sales : SalesData) :
  data.yellow_cost = 6000 ∧
  data.red_cost = 1000 ∧
  data.yellow_quantity = data.red_quantity + 100 ∧
  data.yellow_cost / data.yellow_quantity = 2 * (data.red_cost / data.red_quantity) ∧
  sales.week1_price = 40 ∧
  sales.week2_quantity = 20 ∧
  sales.week3_discount = 0.3 →
  data.red_cost / data.red_quantity = 20 ∧
  ∃ m : ℝ,
    m ≥ 5 ∧
    sales.week1_quantity = 3 * m ∧
    sales.week2_price_decrease = 0.5 * m ∧
    (40 - 20) * (3 * m) + 20 * (40 - 0.5 * m - 20) + (40 * 0.7 - 20) * (50 - 3 * m - 20) ≥ 770 ∧
    ∀ m' : ℝ,
      m' < 5 →
      (40 - 20) * (3 * m') + 20 * (40 - 0.5 * m' - 20) + (40 * 0.7 - 20) * (50 - 3 * m' - 20) < 770 :=
by sorry

end cherry_pricing_and_profit_l265_26562


namespace largest_of_eight_consecutive_integers_l265_26523

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h1 : n > 0) 
  (h2 : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 4328) : 
  n + 7 = 544 := by
  sorry

end largest_of_eight_consecutive_integers_l265_26523


namespace candy_distribution_l265_26564

theorem candy_distribution (total_candy : ℕ) (total_bags : ℕ) (chocolate_hearts_bags : ℕ) (chocolate_kisses_bags : ℕ) 
  (h1 : total_candy = 63)
  (h2 : total_bags = 9)
  (h3 : chocolate_hearts_bags = 2)
  (h4 : chocolate_kisses_bags = 3)
  (h5 : total_candy % total_bags = 0) :
  let candy_per_bag := total_candy / total_bags
  let chocolate_bags := chocolate_hearts_bags + chocolate_kisses_bags
  let non_chocolate_bags := total_bags - chocolate_bags
  non_chocolate_bags * candy_per_bag = 28 := by
sorry


end candy_distribution_l265_26564


namespace largest_common_value_l265_26582

theorem largest_common_value (n m : ℕ) : 
  (∃ n m : ℕ, 479 = 2 + 3 * n ∧ 479 = 3 + 7 * m) ∧ 
  (∀ k : ℕ, k < 500 → k > 479 → ¬(∃ p q : ℕ, k = 2 + 3 * p ∧ k = 3 + 7 * q)) := by
sorry

end largest_common_value_l265_26582


namespace problem_solution_l265_26509

theorem problem_solution (x y : ℝ) 
  (eq1 : x + Real.sin y = 2010)
  (eq2 : x + 2010 * Real.cos y + 3 * Real.sin y = 2005)
  (h : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2009 + Real.pi / 2 := by
sorry

end problem_solution_l265_26509


namespace min_value_theorem_l265_26525

theorem min_value_theorem (x y : ℝ) (h : x^2 + y^2 + x*y = 315) :
  x^2 + y^2 - x*y ≥ 105 := by
sorry

end min_value_theorem_l265_26525


namespace point_on_line_trig_identity_l265_26549

/-- 
Given a point P with coordinates (cos θ, sin θ) that lies on the line 2x + y = 0,
prove that cos 2θ + (1/2) sin 2θ = -1.
-/
theorem point_on_line_trig_identity (θ : Real) 
  (h : 2 * Real.cos θ + Real.sin θ = 0) : 
  Real.cos (2 * θ) + (1/2) * Real.sin (2 * θ) = -1 := by
  sorry

end point_on_line_trig_identity_l265_26549


namespace remainder_problem_l265_26533

theorem remainder_problem (n : ℤ) (h : n ≡ 16 [ZMOD 30]) : 2 * n ≡ 2 [ZMOD 15] := by
  sorry

end remainder_problem_l265_26533


namespace g_neg_two_equals_fifteen_l265_26517

theorem g_neg_two_equals_fifteen :
  let g : ℝ → ℝ := λ x ↦ x^2 - 4*x + 3
  g (-2) = 15 := by sorry

end g_neg_two_equals_fifteen_l265_26517


namespace cubic_roots_sum_l265_26541

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 8 = 0) →
  (b^3 - 15*b^2 + 25*b - 8 = 0) →
  (c^3 - 15*c^2 + 25*c - 8 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 175/9) :=
by sorry

end cubic_roots_sum_l265_26541


namespace new_train_distance_l265_26527

theorem new_train_distance (old_distance : ℝ) (percentage_increase : ℝ) : 
  old_distance = 300 → percentage_increase = 30 → 
  old_distance * (1 + percentage_increase / 100) = 390 := by
  sorry

end new_train_distance_l265_26527


namespace train_speed_problem_l265_26556

/-- Calculates the speed of train A given the conditions of the problem -/
theorem train_speed_problem (length_A length_B : ℝ) (speed_B : ℝ) (crossing_time : ℝ) :
  length_A = 150 →
  length_B = 150 →
  speed_B = 36 →
  crossing_time = 12 →
  (length_A + length_B) / crossing_time * 3.6 - speed_B = 54 :=
by sorry

end train_speed_problem_l265_26556


namespace inquisitive_tourist_ratio_l265_26508

/-- Represents a tour group with a number of people -/
structure TourGroup where
  people : ℕ

/-- Represents a day of tours -/
structure TourDay where
  groups : List TourGroup
  usualQuestionsPerTourist : ℕ
  totalQuestionsAnswered : ℕ
  inquisitiveTouristGroup : ℕ  -- Index of the group with the inquisitive tourist

def calculateRatio (day : TourDay) : ℚ :=
  let regularQuestions := day.groups.enum.foldl
    (fun acc (i, group) =>
      if i = day.inquisitiveTouristGroup
      then acc + (group.people - 1) * day.usualQuestionsPerTourist
      else acc + group.people * day.usualQuestionsPerTourist)
    0
  let inquisitiveQuestions := day.totalQuestionsAnswered - regularQuestions
  inquisitiveQuestions / day.usualQuestionsPerTourist

theorem inquisitive_tourist_ratio (day : TourDay)
  (h1 : day.groups = [⟨6⟩, ⟨11⟩, ⟨8⟩, ⟨7⟩])
  (h2 : day.usualQuestionsPerTourist = 2)
  (h3 : day.totalQuestionsAnswered = 68)
  (h4 : day.inquisitiveTouristGroup = 2)  -- 0-based index for the third group
  : calculateRatio day = 3 := by
  sorry

#eval calculateRatio {
  groups := [⟨6⟩, ⟨11⟩, ⟨8⟩, ⟨7⟩],
  usualQuestionsPerTourist := 2,
  totalQuestionsAnswered := 68,
  inquisitiveTouristGroup := 2
}

end inquisitive_tourist_ratio_l265_26508


namespace isosceles_triangle_base_length_l265_26599

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 22 cm has a base of 8 cm. -/
theorem isosceles_triangle_base_length : ∀ (base congruent_side : ℝ),
  congruent_side = 7 →
  base + 2 * congruent_side = 22 →
  base = 8 := by
  sorry

end isosceles_triangle_base_length_l265_26599


namespace hyperbola_equation_part1_hyperbola_equation_part2_l265_26594

-- Part 1
theorem hyperbola_equation_part1 (c : ℝ) (h1 : c = Real.sqrt 6) :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / (6 - a^2) = 1) ↔ 
  (x^2 / 5 - y^2 = 1)) ∧
  ((-5)^2 / a^2 - 2^2 / (6 - a^2) = 1) := by
sorry

-- Part 2
theorem hyperbola_equation_part2 (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 = 3 ∧ y1 = -4 * Real.sqrt 2)
  (h2 : x2 = 9/4 ∧ y2 = 5) :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧
  (∀ (x y : ℝ), (m * x^2 - n * y^2 = 1) ↔ 
  (y^2 / 16 - x^2 / 9 = 1)) ∧
  (m * x1^2 - n * y1^2 = 1) ∧
  (m * x2^2 - n * y2^2 = 1) := by
sorry

end hyperbola_equation_part1_hyperbola_equation_part2_l265_26594


namespace sum_of_digits_of_seven_to_seventeen_l265_26544

def sum_of_last_two_digits (n : ℕ) : ℕ :=
  (n % 100) / 10 + n % 10

theorem sum_of_digits_of_seven_to_seventeen (n : ℕ) (h : n = (3 + 4)^17) :
  sum_of_last_two_digits n = 7 := by
  sorry

end sum_of_digits_of_seven_to_seventeen_l265_26544


namespace circle_ellipse_ratio_l265_26546

/-- A circle with equation x^2 + (y+1)^2 = n -/
structure Circle where
  n : ℝ

/-- An ellipse with equation x^2 + my^2 = 1 -/
structure Ellipse where
  m : ℝ

/-- The theorem stating that for a circle C and an ellipse M satisfying certain conditions,
    the ratio of n/m equals 8 -/
theorem circle_ellipse_ratio (C : Circle) (M : Ellipse) 
  (h1 : C.n > 0) 
  (h2 : M.m > 0) 
  (h3 : ∃ (x y : ℝ), x^2 + (y+1)^2 = C.n ∧ x^2 + M.m * y^2 = 1) 
  (h4 : ∃ (x y : ℝ), x^2 + y^2 = C.n ∧ x^2 + M.m * y^2 = 1) : 
  C.n / M.m = 8 := by
sorry

end circle_ellipse_ratio_l265_26546


namespace square_sum_equals_eight_l265_26573

theorem square_sum_equals_eight (a b c : ℝ) 
  (sum_condition : a + b + c = 4)
  (product_sum_condition : a * b + b * c + a * c = 4) :
  a^2 + b^2 + c^2 = 8 := by
sorry

end square_sum_equals_eight_l265_26573


namespace radio_show_duration_is_three_hours_l265_26520

/-- Calculates the total duration of a radio show in hours -/
def radio_show_duration (
  talking_segment_duration : ℕ)
  (ad_break_duration : ℕ)
  (num_talking_segments : ℕ)
  (num_ad_breaks : ℕ)
  (song_duration : ℕ) : ℚ :=
  let total_minutes : ℕ := 
    talking_segment_duration * num_talking_segments +
    ad_break_duration * num_ad_breaks +
    song_duration
  (total_minutes : ℚ) / 60

/-- Proves that given the specified conditions, the radio show duration is 3 hours -/
theorem radio_show_duration_is_three_hours :
  radio_show_duration 10 5 3 5 125 = 3 := by
  sorry

end radio_show_duration_is_three_hours_l265_26520


namespace min_distinct_lines_for_31_links_l265_26559

/-- A polygonal chain in a plane -/
structure PolygonalChain where
  links : ℕ
  non_self_intersecting : Bool
  adjacent_links_not_collinear : Bool

/-- The minimum number of distinct lines needed to contain all links of a polygonal chain -/
def min_distinct_lines (chain : PolygonalChain) : ℕ := sorry

/-- Theorem: For a non-self-intersecting polygonal chain with 31 links where adjacent links are not collinear, 
    the minimum number of distinct lines that can contain all links is 9 -/
theorem min_distinct_lines_for_31_links : 
  ∀ (chain : PolygonalChain), 
    chain.links = 31 ∧ 
    chain.non_self_intersecting = true ∧ 
    chain.adjacent_links_not_collinear = true → 
    min_distinct_lines chain = 9 := by sorry

end min_distinct_lines_for_31_links_l265_26559


namespace no_intersection_eq_two_five_l265_26501

theorem no_intersection_eq_two_five : ¬∃ a : ℝ, 
  ({2, 4, a^3 - 2*a^2 - a + 7} : Set ℝ) ∩ 
  ({1, 5*a - 5, -1/2*a^2 + 3/2*a + 4, a^3 + a^2 + 3*a + 7} : Set ℝ) = 
  ({2, 5} : Set ℝ) := by
sorry

end no_intersection_eq_two_five_l265_26501


namespace geometric_sequence_first_term_l265_26557

theorem geometric_sequence_first_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r = 4) -- second term is 4
  (h2 : a * r^3 = 16) -- fourth term is 16
  : a = 2 := by
sorry

end geometric_sequence_first_term_l265_26557


namespace parabola_line_intersection_bounds_l265_26514

/-- Parabola P with equation y = 2x^2 -/
def P : ℝ → ℝ := λ x => 2 * x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- Line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x => m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

/-- Theorem stating the existence of r and s, and their sum -/
theorem parabola_line_intersection_bounds :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 80 :=
sorry

end parabola_line_intersection_bounds_l265_26514


namespace line_through_midpoint_of_ellipse_chord_l265_26555

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 16 = 1

/-- The line we're trying to find -/
def line (x y : ℝ) : Prop := x + 8*y - 17 = 0

/-- Theorem stating that the line passing through the midpoint (1, 2) of a chord of the given ellipse
    has the equation x + 8y - 17 = 0 -/
theorem line_through_midpoint_of_ellipse_chord :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧  -- The endpoints of the chord lie on the ellipse
    (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 2 ∧  -- (1, 2) is the midpoint of the chord
    ∀ (x y : ℝ), line x y ↔ y - 2 = (-1/8) * (x - 1) :=  -- The line equation is correct
by sorry

end line_through_midpoint_of_ellipse_chord_l265_26555


namespace surface_area_greater_when_contained_l265_26589

/-- A convex polyhedron in 3D space -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  surfaceArea : ℝ

/-- States that one polyhedron is completely contained within another -/
def IsContainedIn (inner outer : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem: The surface area of the outer polyhedron is greater than
    the surface area of the inner polyhedron when one is contained in the other -/
theorem surface_area_greater_when_contained
  (inner outer : ConvexPolyhedron)
  (h : IsContainedIn inner outer) :
  outer.surfaceArea > inner.surfaceArea :=
sorry

end surface_area_greater_when_contained_l265_26589


namespace smallest_fraction_l265_26543

theorem smallest_fraction (x : ℝ) (h : x = 5) : 
  min (min (min (min (8/x) (8/(x+1))) (8/(x-1))) (x/8)) ((x+1)/8) = x/8 := by
  sorry

end smallest_fraction_l265_26543


namespace ducks_in_lake_l265_26590

/-- The number of ducks initially in the lake -/
def initial_ducks : ℕ := 13

/-- The number of ducks that joined the lake -/
def joining_ducks : ℕ := 20

/-- The total number of ducks in the lake -/
def total_ducks : ℕ := initial_ducks + joining_ducks

theorem ducks_in_lake : total_ducks = 33 := by
  sorry

end ducks_in_lake_l265_26590


namespace equation_one_solutions_l265_26563

theorem equation_one_solutions (x : ℝ) : (5*x + 2) * (4 - x) = 0 ↔ x = -2/5 ∨ x = 4 := by
  sorry

#check equation_one_solutions

end equation_one_solutions_l265_26563


namespace chocolate_box_problem_l265_26552

theorem chocolate_box_problem (total : ℕ) (remaining : ℕ) : 
  (remaining = 36) →
  (total = (remaining : ℚ) * 3 / (1 - (4/15 : ℚ))) →
  (total = 98) := by
sorry

end chocolate_box_problem_l265_26552


namespace circus_investment_revenue_l265_26550

/-- A circus production investment problem -/
theorem circus_investment_revenue (overhead : ℕ) (production_cost : ℕ) (break_even_performances : ℕ) :
  overhead = 81000 →
  production_cost = 7000 →
  break_even_performances = 9 →
  (overhead + break_even_performances * production_cost) / break_even_performances = 16000 :=
by sorry

end circus_investment_revenue_l265_26550
