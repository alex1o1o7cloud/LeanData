import Mathlib

namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l3167_316782

theorem inverse_sum_reciprocal (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (x * z + y * z + x * y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l3167_316782


namespace NUMINAMATH_CALUDE_right_triangle_existence_l3167_316715

theorem right_triangle_existence (a : ℤ) (h : a ≥ 5) :
  ∃ b c : ℤ, c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l3167_316715


namespace NUMINAMATH_CALUDE_triangles_on_square_sides_l3167_316751

/-- The number of triangles formed by 12 points on the sides of a square -/
def num_triangles_on_square_sides : ℕ := 216

/-- The total number of points on the sides of the square -/
def total_points : ℕ := 12

/-- The number of sides of the square -/
def num_sides : ℕ := 4

/-- The number of points on each side of the square (excluding vertices) -/
def points_per_side : ℕ := 3

/-- Theorem stating the number of triangles formed by points on square sides -/
theorem triangles_on_square_sides :
  num_triangles_on_square_sides = 
    (total_points.choose 3) - (num_sides * points_per_side.choose 3) :=
by sorry

end NUMINAMATH_CALUDE_triangles_on_square_sides_l3167_316751


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3167_316798

/-- A triangle with two sides of length 2 and 4, and the third side being a solution of x^2 - 6x + 8 = 0 has a perimeter of 10 -/
theorem triangle_perimeter : ∀ a b c : ℝ,
  a = 2 →
  b = 4 →
  c^2 - 6*c + 8 = 0 →
  c > 0 →
  a + b > c →
  b + c > a →
  c + a > b →
  a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3167_316798


namespace NUMINAMATH_CALUDE_max_profit_at_36_l3167_316747

/-- Represents the daily sales quantity of product A in kg -/
def y (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the daily profit in yuan -/
def w (x : ℝ) : ℝ := -2 * x^2 + 160 * x - 2760

/-- The cost of product A in yuan per kg -/
def cost_A : ℝ := 20

/-- The maximum allowed price of product A (180% of cost) -/
def max_price_A : ℝ := cost_A * 1.8

theorem max_profit_at_36 :
  ∀ x : ℝ, cost_A ≤ x ∧ x ≤ max_price_A →
  w x ≤ w 36 ∧ w 36 = 408 := by
  sorry

#eval w 36

end NUMINAMATH_CALUDE_max_profit_at_36_l3167_316747


namespace NUMINAMATH_CALUDE_domino_arrangements_equals_binomial_coefficient_l3167_316736

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a domino with width and height -/
structure Domino :=
  (width : ℕ)
  (height : ℕ)

/-- The number of distinct arrangements of dominoes on a grid -/
def distinct_arrangements (g : Grid) (d : Domino) (num_dominoes : ℕ) : ℕ :=
  sorry

/-- The binomial coefficient (n choose k) -/
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem domino_arrangements_equals_binomial_coefficient :
  let g : Grid := { width := 5, height := 3 }
  let d : Domino := { width := 2, height := 1 }
  let num_dominoes : ℕ := 3
  distinct_arrangements g d num_dominoes = binomial_coefficient 6 2 :=
by sorry

end NUMINAMATH_CALUDE_domino_arrangements_equals_binomial_coefficient_l3167_316736


namespace NUMINAMATH_CALUDE_f_symmetry_l3167_316789

/-- A cubic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

/-- Theorem: If f(-2) = 0, then f(2) = 2 -/
theorem f_symmetry (a b : ℝ) (h : f a b (-2) = 0) : f a b 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l3167_316789


namespace NUMINAMATH_CALUDE_weight_of_B_l3167_316738

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 43)
  (h2 : (A + B) / 2 = 48)
  (h3 : (B + C) / 2 = 42) :
  B = 51 := by sorry

end NUMINAMATH_CALUDE_weight_of_B_l3167_316738


namespace NUMINAMATH_CALUDE_intersection_equals_B_l3167_316772

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

-- State the theorem
theorem intersection_equals_B (p : ℝ) : A ∩ B p = B p ↔ p ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_B_l3167_316772


namespace NUMINAMATH_CALUDE_modular_equivalence_problem_l3167_316790

theorem modular_equivalence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ -315 ≡ n [ZMOD 23] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_modular_equivalence_problem_l3167_316790


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3167_316729

theorem polynomial_factorization (x : ℝ) : 12 * x^2 + 8 * x = 4 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3167_316729


namespace NUMINAMATH_CALUDE_f_derivative_l3167_316758

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (2 * x)

theorem f_derivative :
  deriv f = λ x => (2 * x + 2 * x^2) * Real.exp (2 * x) :=
sorry

end NUMINAMATH_CALUDE_f_derivative_l3167_316758


namespace NUMINAMATH_CALUDE_hospital_staff_count_l3167_316706

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) 
  (h1 : total = 250)
  (h2 : doctor_ratio = 2)
  (h3 : nurse_ratio = 3) :
  (nurse_ratio * total) / (doctor_ratio + nurse_ratio) = 150 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l3167_316706


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3167_316711

/-- The number of games in a chess tournament -/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  (n * (n - 1) / 2) * games_per_pair

/-- Theorem: In a chess tournament with 30 players, where each player plays 
    5 times against every other player, the total number of games is 2175 -/
theorem chess_tournament_games : num_games 30 5 = 2175 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_l3167_316711


namespace NUMINAMATH_CALUDE_next_two_terms_l3167_316743

def arithmetic_sequence (a₀ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₀ + n * d

def is_arithmetic_sequence (seq : ℕ → ℕ) (a₀ d : ℕ) : Prop :=
  ∀ n, seq n = arithmetic_sequence a₀ d n

theorem next_two_terms
  (seq : ℕ → ℕ)
  (h : is_arithmetic_sequence seq 3 4)
  (h0 : seq 0 = 3)
  (h1 : seq 1 = 7)
  (h2 : seq 2 = 11)
  (h3 : seq 3 = 15)
  (h4 : seq 4 = 19)
  (h5 : seq 5 = 23) :
  seq 6 = 27 ∧ seq 7 = 31 := by
sorry

end NUMINAMATH_CALUDE_next_two_terms_l3167_316743


namespace NUMINAMATH_CALUDE_task_assignment_ways_l3167_316792

def number_of_students : ℕ := 30
def number_of_tasks : ℕ := 3

def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

theorem task_assignment_ways :
  permutations number_of_students number_of_tasks = 24360 := by
  sorry

end NUMINAMATH_CALUDE_task_assignment_ways_l3167_316792


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3167_316742

theorem triangle_angle_measure (A B C : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π) 
  (h7 : A + B + C = π) (h8 : Real.sqrt 3 / Real.sin A = 1 / Real.sin (π/6)) (h9 : B = π/6) : 
  A = π/3 ∨ A = 2*π/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3167_316742


namespace NUMINAMATH_CALUDE_a_2006_mod_7_l3167_316702

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => a n + (a (n + 1))^2

theorem a_2006_mod_7 : a 2006 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_a_2006_mod_7_l3167_316702


namespace NUMINAMATH_CALUDE_tan_negative_585_deg_l3167_316748

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem
theorem tan_negative_585_deg : tan_deg (-585) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_585_deg_l3167_316748


namespace NUMINAMATH_CALUDE_oldest_child_age_l3167_316728

theorem oldest_child_age (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) 
  (h3 : (a + b + c) / 3 = 9) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l3167_316728


namespace NUMINAMATH_CALUDE_percentage_of_green_caps_l3167_316705

def total_caps : ℕ := 125
def red_caps : ℕ := 50

theorem percentage_of_green_caps :
  (total_caps - red_caps : ℚ) / total_caps * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_green_caps_l3167_316705


namespace NUMINAMATH_CALUDE_evaluate_expression_l3167_316744

theorem evaluate_expression (x y : ℝ) (hx : x = 3) (hy : y = 0) : y * (y - 3 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3167_316744


namespace NUMINAMATH_CALUDE_inequality_proof_l3167_316733

theorem inequality_proof (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  a * x^2 + b * y^2 ≥ (a * x + b * y)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3167_316733


namespace NUMINAMATH_CALUDE_fruit_bags_weight_l3167_316703

theorem fruit_bags_weight (x y z : ℝ) 
  (h1 : x + y = 90) 
  (h2 : y + z = 100) 
  (h3 : z + x = 110) 
  (pos_x : x > 0) 
  (pos_y : y > 0) 
  (pos_z : z > 0) : 
  x + y + z = 150 := by
sorry

end NUMINAMATH_CALUDE_fruit_bags_weight_l3167_316703


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3167_316784

theorem sqrt_simplification :
  (Real.sqrt 24 - Real.sqrt 2) - (Real.sqrt 8 + Real.sqrt 6) = Real.sqrt 6 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3167_316784


namespace NUMINAMATH_CALUDE_unanswered_questions_count_l3167_316762

/-- Represents the scoring system for AHSME competition --/
structure ScoringSystem where
  correct : Int
  incorrect : Int
  unanswered : Int

/-- Represents the AHSME competition --/
structure AHSMECompetition where
  new_scoring : ScoringSystem
  old_scoring : ScoringSystem
  total_questions : Nat
  new_score : Int
  old_score : Int

/-- Theorem stating that the number of unanswered questions is 9 --/
theorem unanswered_questions_count (comp : AHSMECompetition)
  (h_new_scoring : comp.new_scoring = { correct := 5, incorrect := 0, unanswered := 2 })
  (h_old_scoring : comp.old_scoring = { correct := 4, incorrect := -1, unanswered := 0 })
  (h_old_base : comp.old_score - 30 = 4 * (comp.new_score / 5) - (comp.total_questions - (comp.new_score / 5) - 9))
  (h_total : comp.total_questions = 30)
  (h_new_score : comp.new_score = 93)
  (h_old_score : comp.old_score = 84) :
  ∃ (correct incorrect : Nat), 
    correct + incorrect + 9 = comp.total_questions ∧
    5 * correct + 2 * 9 = comp.new_score ∧
    4 * correct - incorrect = comp.old_score - 30 :=
by sorry


end NUMINAMATH_CALUDE_unanswered_questions_count_l3167_316762


namespace NUMINAMATH_CALUDE_orange_harvest_l3167_316741

theorem orange_harvest (total_days : ℕ) (total_sacks : ℕ) (h1 : total_days = 14) (h2 : total_sacks = 56) :
  total_sacks / total_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_l3167_316741


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_l3167_316750

/-- Given a square with side length 2a and an intersecting line y = x + a/2,
    the perimeter of one part divided by a equals (√17 + 8) / 2 -/
theorem square_intersection_perimeter (a : ℝ) (a_pos : a > 0) :
  let square_vertices := [(-a, -a), (a, -a), (-a, a), (a, a)]
  let intersecting_line (x : ℝ) := x + a / 2
  let intersection_points := [(-a, -a/2), (a, -a), (a/2, a), (-a, a)]
  let perimeter := Real.sqrt (17 * a^2) / 2 + 4 * a
  perimeter / a = (Real.sqrt 17 + 8) / 2 :=
by sorry

end NUMINAMATH_CALUDE_square_intersection_perimeter_l3167_316750


namespace NUMINAMATH_CALUDE_abs_even_and_increasing_l3167_316723

-- Define the absolute value function
def f (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_abs_even_and_increasing_l3167_316723


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3167_316720

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℚ),
  x₁ = -3/4 ∧ x₂ = -4/3 ∧ x₃ = 5/2 ∧
  x₁ * x₂ = 1 ∧
  24 * x₁^3 - 10 * x₁^2 - 101 * x₁ - 60 = 0 ∧
  24 * x₂^3 - 10 * x₂^2 - 101 * x₂ - 60 = 0 ∧
  24 * x₃^3 - 10 * x₃^2 - 101 * x₃ - 60 = 0 :=
by
  sorry

#check cubic_equation_roots

end NUMINAMATH_CALUDE_cubic_equation_roots_l3167_316720


namespace NUMINAMATH_CALUDE_problem_solution_l3167_316759

theorem problem_solution : (0.5 : ℝ)^3 - (0.1 : ℝ)^3 / (0.5 : ℝ)^2 + 0.05 + (0.1 : ℝ)^2 = 0.181 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3167_316759


namespace NUMINAMATH_CALUDE_f_range_is_1_2_5_l3167_316735

def f (x : Int) : Int := x^2 + 1

def domain : Set Int := {-1, 0, 1, 2}

theorem f_range_is_1_2_5 : 
  {y | ∃ x ∈ domain, f x = y} = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_f_range_is_1_2_5_l3167_316735


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3167_316717

theorem z_in_first_quadrant (z : ℂ) (h : (3 + 2*I)*z = 13*I) : 
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3167_316717


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3167_316700

theorem sum_of_fractions : 
  (19 / ((2^3 - 1) * (3^3 - 1)) + 
   37 / ((3^3 - 1) * (4^3 - 1)) + 
   61 / ((4^3 - 1) * (5^3 - 1)) + 
   91 / ((5^3 - 1) * (6^3 - 1))) = 208 / 1505 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3167_316700


namespace NUMINAMATH_CALUDE_students_per_minibus_l3167_316773

theorem students_per_minibus (total_vehicles : Nat) (num_vans : Nat) (num_minibusses : Nat)
  (students_per_van : Nat) (total_students : Nat) :
  total_vehicles = num_vans + num_minibusses →
  num_vans = 6 →
  num_minibusses = 4 →
  students_per_van = 10 →
  total_students = 156 →
  (total_students - num_vans * students_per_van) / num_minibusses = 24 := by
  sorry

#check students_per_minibus

end NUMINAMATH_CALUDE_students_per_minibus_l3167_316773


namespace NUMINAMATH_CALUDE_speed_calculation_l3167_316712

/-- Given a distance of 900 meters covered in 180 seconds, prove that the speed is 18 km/h -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 900) (h2 : time = 180) :
  (distance / 1000) / (time / 3600) = 18 := by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l3167_316712


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l3167_316757

theorem max_value_on_ellipse (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 13 ∧ x + y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l3167_316757


namespace NUMINAMATH_CALUDE_right_triangle_special_area_l3167_316713

theorem right_triangle_special_area (c : ℝ) (h : c > 0) : ∃ (S : ℝ),
  (∃ (x : ℝ), 0 < x ∧ x < c ∧ (c - x) / x = x / c) →
  S = (c^2 * Real.sqrt (Real.sqrt 5 - 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_special_area_l3167_316713


namespace NUMINAMATH_CALUDE_friendly_integers_in_range_two_not_friendly_l3167_316766

def friendly (a : ℕ) : Prop :=
  ∃ m n : ℕ+, (m^2 + n) * (n^2 + m) = a * (m - n)^3

theorem friendly_integers_in_range :
  ∃ S : Finset ℕ, S.card ≥ 500 ∧ ∀ a ∈ S, a ∈ Finset.range 2013 ∧ friendly a :=
sorry

theorem two_not_friendly : ¬ friendly 2 :=
sorry

end NUMINAMATH_CALUDE_friendly_integers_in_range_two_not_friendly_l3167_316766


namespace NUMINAMATH_CALUDE_cone_surface_area_l3167_316778

/-- The surface area of a cone formed by rotating a right triangle -/
theorem cone_surface_area (r h l : ℝ) (triangle_condition : r^2 + h^2 = l^2) :
  r = 3 → h = 4 → l = 5 → (π * r * l + π * r^2) = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3167_316778


namespace NUMINAMATH_CALUDE_y_equivalent_condition_l3167_316730

theorem y_equivalent_condition (x y : ℝ) :
  y = 2 * x + 4 →
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8) ↔ 
  (y ∈ Set.Icc (-6) 6 ∪ Set.Icc 14 26) :=
by sorry

end NUMINAMATH_CALUDE_y_equivalent_condition_l3167_316730


namespace NUMINAMATH_CALUDE_range_of_x_l3167_316768

def f (x : ℝ) : ℝ := 3 * x - 2

def assignment_process (x : ℝ) : ℕ → ℝ
| 0 => x
| n + 1 => f (assignment_process x n)

def process_stops (x : ℝ) (k : ℕ) : Prop :=
  assignment_process x (k - 1) ≤ 244 ∧ assignment_process x k > 244

theorem range_of_x (x : ℝ) (k : ℕ) (h : k > 0) (h_stop : process_stops x k) :
  x ∈ Set.Ioo (3^(5 - k) + 1) (3^(6 - k) + 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l3167_316768


namespace NUMINAMATH_CALUDE_not_perfect_square_l3167_316785

theorem not_perfect_square (k : ℕ+) : ¬ ∃ (n : ℕ), (16 * k + 8 : ℕ) = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3167_316785


namespace NUMINAMATH_CALUDE_min_value_expression_l3167_316707

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3167_316707


namespace NUMINAMATH_CALUDE_tennis_match_duration_l3167_316781

def minutes_per_hour : ℕ := 60

def hours : ℕ := 11
def additional_minutes : ℕ := 5

theorem tennis_match_duration : 
  hours * minutes_per_hour + additional_minutes = 665 := by
  sorry

end NUMINAMATH_CALUDE_tennis_match_duration_l3167_316781


namespace NUMINAMATH_CALUDE_distance_after_10_hours_l3167_316779

/-- The distance between two trains after a given time -/
def distance_between_trains (speed1 speed2 time : ℝ) : ℝ :=
  (speed2 - speed1) * time

/-- Theorem: The distance between two trains after 10 hours -/
theorem distance_after_10_hours :
  distance_between_trains 10 35 10 = 250 := by
  sorry

#eval distance_between_trains 10 35 10

end NUMINAMATH_CALUDE_distance_after_10_hours_l3167_316779


namespace NUMINAMATH_CALUDE_aaron_position_2023_l3167_316776

/-- Represents a point on the 2D plane -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Aaron's movement rules -/
def nextDirection (d : Direction) : Direction :=
  match d with
  | Direction.East => Direction.North
  | Direction.North => Direction.West
  | Direction.West => Direction.South
  | Direction.South => Direction.East

/-- Move one step in the given direction -/
def move (p : Point) (d : Direction) : Point :=
  match d with
  | Direction.East => { x := p.x + 1, y := p.y }
  | Direction.North => { x := p.x, y := p.y + 1 }
  | Direction.West => { x := p.x - 1, y := p.y }
  | Direction.South => { x := p.x, y := p.y - 1 }

/-- Aaron's position after n steps -/
def aaronPosition (n : Nat) : Point :=
  sorry  -- The actual implementation would go here

theorem aaron_position_2023 :
  aaronPosition 2023 = { x := 21, y := -22 } := by
  sorry


end NUMINAMATH_CALUDE_aaron_position_2023_l3167_316776


namespace NUMINAMATH_CALUDE_max_sum_is_fifty_l3167_316791

/-- A hexagonal prism with an added pyramid -/
structure HexagonalPrismWithPyramid where
  /-- Number of faces when pyramid is added to hexagonal face -/
  faces_hex : ℕ
  /-- Number of vertices when pyramid is added to hexagonal face -/
  vertices_hex : ℕ
  /-- Number of edges when pyramid is added to hexagonal face -/
  edges_hex : ℕ
  /-- Number of faces when pyramid is added to rectangular face -/
  faces_rect : ℕ
  /-- Number of vertices when pyramid is added to rectangular face -/
  vertices_rect : ℕ
  /-- Number of edges when pyramid is added to rectangular face -/
  edges_rect : ℕ

/-- The maximum sum of exterior faces, vertices, and edges -/
def max_sum (shape : HexagonalPrismWithPyramid) : ℕ :=
  max (shape.faces_hex + shape.vertices_hex + shape.edges_hex)
      (shape.faces_rect + shape.vertices_rect + shape.edges_rect)

/-- Theorem: The maximum sum of exterior faces, vertices, and edges is 50 -/
theorem max_sum_is_fifty (shape : HexagonalPrismWithPyramid) 
  (h1 : shape.faces_hex = 13)
  (h2 : shape.vertices_hex = 13)
  (h3 : shape.edges_hex = 24)
  (h4 : shape.faces_rect = 11)
  (h5 : shape.vertices_rect = 13)
  (h6 : shape.edges_rect = 22) :
  max_sum shape = 50 := by
  sorry


end NUMINAMATH_CALUDE_max_sum_is_fifty_l3167_316791


namespace NUMINAMATH_CALUDE_james_singing_lesson_payment_l3167_316771

/-- Calculates the amount James pays for singing lessons given the specified conditions. -/
def jamesSingingLessonCost (totalLessons : ℕ) (lessonCost : ℕ) (freeLesson : ℕ) (fullPaidLessons : ℕ) : ℕ := 
  let paidLessonsAfterFull := (totalLessons - freeLesson - fullPaidLessons) / 2
  let totalCost := (fullPaidLessons + paidLessonsAfterFull) * lessonCost
  totalCost / 2

/-- Theorem stating that James pays $35 for his singing lessons under the given conditions. -/
theorem james_singing_lesson_payment :
  jamesSingingLessonCost 20 5 1 10 = 35 := by
  sorry

#eval jamesSingingLessonCost 20 5 1 10

end NUMINAMATH_CALUDE_james_singing_lesson_payment_l3167_316771


namespace NUMINAMATH_CALUDE_all_equilateral_triangles_similar_l3167_316716

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Definition of similarity for triangles -/
def similar_triangles (t1 t2 : EquilateralTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.side = k * t2.side

/-- Theorem: All equilateral triangles are similar -/
theorem all_equilateral_triangles_similar (t1 t2 : EquilateralTriangle) :
  similar_triangles t1 t2 :=
sorry

end NUMINAMATH_CALUDE_all_equilateral_triangles_similar_l3167_316716


namespace NUMINAMATH_CALUDE_encrypted_text_is_cipher_of_problem_statement_l3167_316797

/-- Represents a character in the Russian alphabet -/
inductive RussianChar : Type
| vowel : RussianChar
| consonant : RussianChar

/-- Represents a string of Russian characters -/
def RussianString := List RussianChar

/-- The tarabar cipher function -/
def tarabarCipher : RussianString → RussianString := sorry

/-- The first sentence of the problem statement -/
def problemStatement : RussianString := sorry

/-- The given encrypted text -/
def encryptedText : RussianString := sorry

/-- Theorem stating that the encrypted text is a cipher of the problem statement -/
theorem encrypted_text_is_cipher_of_problem_statement :
  tarabarCipher problemStatement = encryptedText := by sorry

end NUMINAMATH_CALUDE_encrypted_text_is_cipher_of_problem_statement_l3167_316797


namespace NUMINAMATH_CALUDE_first_group_size_is_three_l3167_316709

/-- The number of people in the first group -/
def first_group_size : ℕ := 3

/-- The amount of work completed by the first group in 3 days -/
def first_group_work : ℕ := 3

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 3

/-- The number of people in the second group -/
def second_group_size : ℕ := 5

/-- The amount of work completed by the second group in 3 days -/
def second_group_work : ℕ := 5

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 3

theorem first_group_size_is_three :
  first_group_size * first_group_work * second_group_days =
  second_group_size * second_group_work * first_group_days :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_three_l3167_316709


namespace NUMINAMATH_CALUDE_smallest_r_for_B_subset_C_l3167_316787

def A : Set ℝ := {t | 0 < t ∧ t < 2 * Real.pi}

def B : Set (ℝ × ℝ) := {p | ∃ t ∈ A, p.1 = Real.sin t ∧ p.2 = 2 * Real.sin t * Real.cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

theorem smallest_r_for_B_subset_C :
  ∃! r : ℝ, (∀ s : ℝ, B ⊆ C s → r ≤ s) ∧ B ⊆ C r ∧ r = 5/4 := by sorry

end NUMINAMATH_CALUDE_smallest_r_for_B_subset_C_l3167_316787


namespace NUMINAMATH_CALUDE_square_of_binomial_l3167_316774

theorem square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → 
  a = 81 / 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3167_316774


namespace NUMINAMATH_CALUDE_distance_swum_back_l3167_316710

/-- The distance a person swims back against the current -/
def swim_distance (still_water_speed : ℝ) (water_speed : ℝ) (time : ℝ) : ℝ :=
  (still_water_speed - water_speed) * time

/-- Theorem: The distance swum back against the current is 8 km -/
theorem distance_swum_back (still_water_speed : ℝ) (water_speed : ℝ) (time : ℝ)
    (h1 : still_water_speed = 8)
    (h2 : water_speed = 4)
    (h3 : time = 2) :
    swim_distance still_water_speed water_speed time = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_swum_back_l3167_316710


namespace NUMINAMATH_CALUDE_fraction_equality_l3167_316732

theorem fraction_equality : (2523 - 2428)^2 / 121 = 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3167_316732


namespace NUMINAMATH_CALUDE_probability_third_ball_white_l3167_316794

-- Define the problem setup
theorem probability_third_ball_white (n : ℕ) (h : n > 2) :
  let bags := Finset.range n
  let balls_in_bag (k : ℕ) := k + (n - k)
  let prob_choose_bag := 1 / n
  let prob_white_third (k : ℕ) := (n - k) / n
  (bags.sum (λ k => prob_choose_bag * prob_white_third k)) = (n - 1) / (2 * n) :=
by sorry


end NUMINAMATH_CALUDE_probability_third_ball_white_l3167_316794


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3167_316770

-- Define the variables
def distance_day1 : ℝ := 100
def distance_day2 : ℝ := 175
def time_difference : ℝ := 3

-- Define the theorem
theorem average_speed_calculation :
  ∃ (v : ℝ), v > 0 ∧
    distance_day2 / v - distance_day1 / v = time_difference ∧
    v = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3167_316770


namespace NUMINAMATH_CALUDE_vector_collinearity_l3167_316746

theorem vector_collinearity (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![-1, x]
  let sum : Fin 2 → ℝ := ![a 0 + b 0, a 1 + b 1]
  let diff : Fin 2 → ℝ := ![a 0 - b 0, a 1 - b 1]
  (sum 0 * diff 0 + sum 1 * diff 1 = 0) → (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3167_316746


namespace NUMINAMATH_CALUDE_roots_sum_minus_product_l3167_316725

theorem roots_sum_minus_product (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ + 3 = 0) → 
  (x₂^2 - 4*x₂ + 3 = 0) → 
  x₁ + x₂ - x₁*x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_minus_product_l3167_316725


namespace NUMINAMATH_CALUDE_second_to_tallest_ratio_l3167_316775

/-- The heights of four buildings satisfying certain conditions -/
structure BuildingHeights where
  t : ℝ  -- height of the tallest building
  s : ℝ  -- height of the second tallest building
  u : ℝ  -- height of the third tallest building
  v : ℝ  -- height of the fourth tallest building
  h1 : t = 100  -- the tallest building is 100 feet tall
  h2 : u = s / 2  -- the third tallest is half as tall as the second
  h3 : v = u / 5  -- the fourth is one-fifth as tall as the third
  h4 : t + s + u + v = 180  -- all 4 buildings together are 180 feet tall

/-- The ratio of the second tallest building to the tallest is 1:2 -/
theorem second_to_tallest_ratio (b : BuildingHeights) : b.s / b.t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_second_to_tallest_ratio_l3167_316775


namespace NUMINAMATH_CALUDE_fuel_after_600km_distance_with_22L_left_l3167_316793

-- Define the relationship between distance and remaining fuel
def fuel_remaining (s : ℝ) : ℝ := 50 - 0.08 * s

-- Theorem 1: When distance is 600 km, remaining fuel is 2 L
theorem fuel_after_600km : fuel_remaining 600 = 2 := by sorry

-- Theorem 2: When remaining fuel is 22 L, distance traveled is 350 km
theorem distance_with_22L_left : ∃ s : ℝ, fuel_remaining s = 22 ∧ s = 350 := by sorry

end NUMINAMATH_CALUDE_fuel_after_600km_distance_with_22L_left_l3167_316793


namespace NUMINAMATH_CALUDE_power_inequality_l3167_316739

theorem power_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ^ b > b ^ a) (hbc : b ^ c > c ^ b) : 
  a ^ c > c ^ a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3167_316739


namespace NUMINAMATH_CALUDE_cost_of_18_pencils_13_notebooks_l3167_316731

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook -/
def notebook_cost : ℝ := sorry

/-- The first given condition: 9 pencils and 11 notebooks cost $6.05 -/
axiom condition1 : 9 * pencil_cost + 11 * notebook_cost = 6.05

/-- The second given condition: 6 pencils and 4 notebooks cost $2.68 -/
axiom condition2 : 6 * pencil_cost + 4 * notebook_cost = 2.68

/-- Theorem: The cost of 18 pencils and 13 notebooks is $8.45 -/
theorem cost_of_18_pencils_13_notebooks :
  18 * pencil_cost + 13 * notebook_cost = 8.45 := by sorry

end NUMINAMATH_CALUDE_cost_of_18_pencils_13_notebooks_l3167_316731


namespace NUMINAMATH_CALUDE_merry_saturday_boxes_l3167_316755

/-- The number of boxes Merry had on Sunday -/
def sunday_boxes : ℕ := 25

/-- The number of apples in each box -/
def apples_per_box : ℕ := 10

/-- The total number of apples sold on Saturday and Sunday -/
def total_apples_sold : ℕ := 720

/-- The number of boxes left after selling -/
def boxes_left : ℕ := 3

/-- The number of boxes Merry had on Saturday -/
def saturday_boxes : ℕ := 69

theorem merry_saturday_boxes :
  saturday_boxes = 69 :=
by sorry

end NUMINAMATH_CALUDE_merry_saturday_boxes_l3167_316755


namespace NUMINAMATH_CALUDE_tart_base_flour_calculation_l3167_316764

theorem tart_base_flour_calculation (original_bases : ℕ) (original_flour : ℚ) 
  (new_bases : ℕ) (new_flour : ℚ) : 
  original_bases = 40 → 
  original_flour = 1/8 → 
  new_bases = 25 → 
  original_bases * original_flour = new_bases * new_flour → 
  new_flour = 1/5 := by
sorry

end NUMINAMATH_CALUDE_tart_base_flour_calculation_l3167_316764


namespace NUMINAMATH_CALUDE_two_books_adjacent_probability_l3167_316719

theorem two_books_adjacent_probability (n : ℕ) (h : n = 10) :
  let total_arrangements := n.factorial
  let favorable_arrangements := ((n - 1).factorial * 2)
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_two_books_adjacent_probability_l3167_316719


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l3167_316722

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral (P : Type*) [MetricSpace P] :=
  (A B C D : P)
  (cyclic : ∃ (center : P) (radius : ℝ), dist center A = radius ∧ dist center B = radius ∧ dist center C = radius ∧ dist center D = radius)

/-- The inequality for cyclic quadrilaterals -/
theorem cyclic_quadrilateral_inequality {P : Type*} [MetricSpace P] (ABCD : CyclicQuadrilateral P) :
  |dist ABCD.A ABCD.B - dist ABCD.C ABCD.D| + |dist ABCD.A ABCD.D - dist ABCD.B ABCD.C| ≥ 2 * |dist ABCD.A ABCD.C - dist ABCD.B ABCD.D| :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l3167_316722


namespace NUMINAMATH_CALUDE_system_solution_l3167_316786

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1), (0, 0, 0)}

theorem system_solution (x y z : ℝ) :
  (x * y = z ∧ x * z = y ∧ y * z = x) ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3167_316786


namespace NUMINAMATH_CALUDE_system_solution_l3167_316737

theorem system_solution :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (y - 2 * Real.sqrt (x * y) - Real.sqrt (y / x) + 2 = 0) →
  (3 * x^2 * y^2 + y^4 = 84) →
  ((x = 1/3 ∧ y = 3) ∨ (x = (21/76)^(1/4) ∧ y = 2 * (84/19)^(1/4))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3167_316737


namespace NUMINAMATH_CALUDE_boyds_male_friends_percentage_l3167_316718

theorem boyds_male_friends_percentage 
  (julian_total : ℕ) 
  (julian_boys_percent : ℚ) 
  (boyd_total : ℕ) 
  (boyd_girls_multiplier : ℕ) : 
  julian_total = 80 → 
  julian_boys_percent = 60 / 100 → 
  boyd_total = 100 → 
  boyd_girls_multiplier = 2 → 
  (boyd_total - boyd_girls_multiplier * (julian_total * (1 - julian_boys_percent))) / boyd_total = 36 / 100 := by
  sorry

end NUMINAMATH_CALUDE_boyds_male_friends_percentage_l3167_316718


namespace NUMINAMATH_CALUDE_butterfingers_count_l3167_316754

theorem butterfingers_count (total : ℕ) (snickers : ℕ) (mars : ℕ) (butterfingers : ℕ) : 
  total = 12 → snickers = 3 → mars = 2 → total = snickers + mars + butterfingers →
  butterfingers = 7 := by
sorry

end NUMINAMATH_CALUDE_butterfingers_count_l3167_316754


namespace NUMINAMATH_CALUDE_adjacent_probability_l3167_316783

/-- The number of seats in the arrangement -/
def total_seats : ℕ := 9

/-- The number of students to be seated -/
def num_students : ℕ := 8

/-- The number of rows in the seating arrangement -/
def num_rows : ℕ := 3

/-- The number of columns in the seating arrangement -/
def num_columns : ℕ := 3

/-- Calculate the total number of possible seating arrangements -/
def total_arrangements : ℕ := Nat.factorial total_seats

/-- Calculate the number of favorable arrangements where Abby and Bridget are adjacent -/
def favorable_arrangements : ℕ :=
  (num_rows * (num_columns - 1) + num_columns * (num_rows - 1)) * 2 * Nat.factorial (num_students - 1)

/-- The probability that Abby and Bridget are adjacent in the same row or column -/
theorem adjacent_probability :
  (favorable_arrangements : ℚ) / total_arrangements = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_probability_l3167_316783


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l3167_316795

theorem congruence_solutions_count : ∃ (S : Finset ℕ), 
  (∀ x ∈ S, x < 100 ∧ x > 0 ∧ (x + 13) % 34 = 55 % 34) ∧ 
  (∀ x < 100, x > 0 → (x + 13) % 34 = 55 % 34 → x ∈ S) ∧
  Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l3167_316795


namespace NUMINAMATH_CALUDE_tv_price_increase_l3167_316734

theorem tv_price_increase (x : ℝ) : 
  (1 + 0.3) * (1 + x) = 1 + 0.5600000000000001 ↔ x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l3167_316734


namespace NUMINAMATH_CALUDE_angle_subtraction_theorem_polynomial_simplification_theorem_l3167_316796

-- Define angle type
def Angle := ℕ × ℕ -- (degrees, minutes)

-- Define angle subtraction
def angle_sub (a b : Angle) : Angle := sorry

-- Define polynomial expression
def poly_expr (m : ℝ) := 5*m^2 - (m^2 - 6*m) - 2*(-m + 3*m^2)

theorem angle_subtraction_theorem :
  angle_sub (34, 26) (25, 33) = (8, 53) := by sorry

theorem polynomial_simplification_theorem (m : ℝ) :
  poly_expr m = -2*m^2 + 8*m := by sorry

end NUMINAMATH_CALUDE_angle_subtraction_theorem_polynomial_simplification_theorem_l3167_316796


namespace NUMINAMATH_CALUDE_min_value_z_l3167_316753

theorem min_value_z (x y : ℝ) (h1 : x - y + 5 ≥ 0) (h2 : x + y ≥ 0) (h3 : x ≤ 3) :
  ∀ z : ℝ, z = (x + y + 2) / (x + 3) → z ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_min_value_z_l3167_316753


namespace NUMINAMATH_CALUDE_triangle_inequality_l3167_316708

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3167_316708


namespace NUMINAMATH_CALUDE_remainder_3_305_mod_13_l3167_316701

theorem remainder_3_305_mod_13 : 3^305 % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_305_mod_13_l3167_316701


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3167_316756

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (m n : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : c > 0) (h4 : m * n = 2 / 9) :
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2
  let asymptote (x : ℝ) := b / a * x
  let A : ℝ × ℝ := (c, asymptote c)
  let B : ℝ × ℝ := (c, -asymptote c)
  let P : ℝ × ℝ := ((m + n) * c, (m - n) * asymptote c)
  (f (P.1) (P.2) = 1) →
  (c / a = 3 * Real.sqrt 2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3167_316756


namespace NUMINAMATH_CALUDE_xy_cube_plus_cube_xy_l3167_316724

theorem xy_cube_plus_cube_xy (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = -4) :
  x * y^3 + x^3 * y = -68 := by
  sorry

end NUMINAMATH_CALUDE_xy_cube_plus_cube_xy_l3167_316724


namespace NUMINAMATH_CALUDE_h_max_at_72_l3167_316777

/-- The divisor function d(n) -/
def d (n : ℕ+) : ℕ := sorry

/-- The function h(n) = d(n)^2 / n^(1/4) -/
noncomputable def h (n : ℕ+) : ℝ := (d n)^2 / n.val^(1/4 : ℝ)

/-- The theorem stating that h(n) is maximized when n = 72 -/
theorem h_max_at_72 : ∀ n : ℕ+, n ≠ 72 → h n < h 72 := by sorry

end NUMINAMATH_CALUDE_h_max_at_72_l3167_316777


namespace NUMINAMATH_CALUDE_probability_of_losing_l3167_316745

theorem probability_of_losing (p_win p_draw : ℚ) (h1 : p_win = 1/3) (h2 : p_draw = 1/2) 
  (h3 : p_win + p_draw + p_lose = 1) : p_lose = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_losing_l3167_316745


namespace NUMINAMATH_CALUDE_all_f_zero_l3167_316763

-- Define the type for infinite sequences of integers
def T := ℕ → ℤ

-- Define the sum of two sequences
def seqSum (x y : T) : T := λ n => x n + y n

-- Define the property of having exactly one 1 and all others 0
def hasOneOne (x : T) : Prop :=
  ∃ i, x i = 1 ∧ ∀ j, j ≠ i → x j = 0

-- Define the function f with its properties
def isValidF (f : T → ℤ) : Prop :=
  (∀ x, hasOneOne x → f x = 0) ∧
  (∀ x y, f (seqSum x y) = f x + f y)

-- The theorem to prove
theorem all_f_zero (f : T → ℤ) (hf : isValidF f) :
  ∀ x : T, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_f_zero_l3167_316763


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3167_316752

/-- Given an ellipse with equation 4x^2 + y^2 = 16, its major axis has length 8 -/
theorem ellipse_major_axis_length :
  ∀ (x y : ℝ), 4 * x^2 + y^2 = 16 → ∃ (a b : ℝ), 
    a > b ∧ 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    2 * a = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3167_316752


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l3167_316769

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Nat)
  (diamonds : Nat)
  (hearts : Nat)
  (spades : Nat)

/-- The probability of drawing a diamond, then a heart, then a spade from a standard 52 card deck -/
def probability_diamond_heart_spade (d : Deck) : ℚ :=
  (d.diamonds : ℚ) / d.cards *
  (d.hearts : ℚ) / (d.cards - 1) *
  (d.spades : ℚ) / (d.cards - 2)

/-- A standard 52 card deck -/
def standard_deck : Deck :=
  { cards := 52
  , diamonds := 13
  , hearts := 13
  , spades := 13 }

theorem probability_in_standard_deck :
  probability_diamond_heart_spade standard_deck = 2197 / 132600 :=
by sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l3167_316769


namespace NUMINAMATH_CALUDE_final_red_probability_l3167_316727

-- Define the contents of each bag
def bagA : ℕ × ℕ := (5, 3)  -- (white, black)
def bagB : ℕ × ℕ := (4, 6)  -- (red, green)
def bagC : ℕ × ℕ := (3, 4)  -- (red, green)

-- Define the probability of drawing a specific marble from a bag
def probDraw (color : ℕ) (bag : ℕ × ℕ) : ℚ :=
  color / (bag.1 + bag.2)

-- Define the probability of the final marble being red
def probFinalRed : ℚ :=
  let probWhiteA := probDraw bagA.1 bagA
  let probBlackA := probDraw bagA.2 bagA
  let probGreenB := probDraw bagB.2 bagB
  let probRedB := probDraw bagB.1 bagB
  let probGreenC := probDraw bagC.2 bagC
  let probRedC := probDraw bagC.1 bagC
  probWhiteA * probGreenB * probRedB + probBlackA * probGreenC * probRedC

-- Theorem statement
theorem final_red_probability : probFinalRed = 79 / 980 := by
  sorry

end NUMINAMATH_CALUDE_final_red_probability_l3167_316727


namespace NUMINAMATH_CALUDE_vipers_count_l3167_316761

/-- The number of vipers in a swamp area -/
def num_vipers (num_crocodiles num_alligators total_animals : ℕ) : ℕ :=
  total_animals - (num_crocodiles + num_alligators)

/-- Theorem: The number of vipers in the swamp is 5 -/
theorem vipers_count : num_vipers 22 23 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vipers_count_l3167_316761


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3167_316740

theorem quadratic_roots_product (a b : ℝ) : 
  (3 * a^2 + 9 * a - 18 = 0) → 
  (3 * b^2 + 9 * b - 18 = 0) → 
  (3*a - 2) * (6*b - 9) = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3167_316740


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l3167_316704

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2400 →
  Nat.gcd a b = 30 →
  a = 150 →
  b = 480 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l3167_316704


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3167_316788

/-- A positive arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r q : ℝ, r > 0 ∧ q > 1 ∧ ∀ k, a (n + k) = a n * r^k * q^(k*(k-1)/2)

theorem arithmetic_geometric_sequence_sum (a : ℕ → ℝ) 
  (h_seq : ArithGeomSeq a) 
  (h_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) : 
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3167_316788


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3167_316726

theorem smallest_solution_of_equation :
  ∃ x : ℝ, x = 1 - Real.sqrt 10 ∧
  (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 12 ∧
  ∀ y : ℝ, (3 * y) / (y - 3) + (3 * y^2 - 27) / y = 12 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3167_316726


namespace NUMINAMATH_CALUDE_altitude_intersection_location_depends_on_shape_l3167_316765

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define the shape of a triangle
inductive TriangleShape
  | Acute
  | Right
  | Obtuse

-- Define the location of a point relative to a triangle
inductive PointLocation
  | Inside
  | OnVertex
  | Outside

-- Function to determine the shape of a triangle
def determineShape (t : Triangle) : TriangleShape :=
  sorry

-- Function to find the intersection point of altitudes
def altitudeIntersection (t : Triangle) : ℝ × ℝ :=
  sorry

-- Function to determine the location of a point relative to a triangle
def determinePointLocation (t : Triangle) (p : ℝ × ℝ) : PointLocation :=
  sorry

-- Theorem stating that the location of the altitude intersection depends on the triangle shape
theorem altitude_intersection_location_depends_on_shape (t : Triangle) :
  let shape := determineShape t
  let intersection := altitudeIntersection t
  let location := determinePointLocation t intersection
  (shape = TriangleShape.Acute → location = PointLocation.Inside) ∧
  (shape = TriangleShape.Right → location = PointLocation.OnVertex) ∧
  (shape = TriangleShape.Obtuse → location = PointLocation.Outside) :=
  sorry

end NUMINAMATH_CALUDE_altitude_intersection_location_depends_on_shape_l3167_316765


namespace NUMINAMATH_CALUDE_minimum_distance_triangle_warehouse_l3167_316714

theorem minimum_distance_triangle_warehouse (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 7) (h3 : c = 3) :
  ∃ (p : ℝ × ℝ),
    p.1 > 0 ∧ p.1 < c ∧ p.2 > 0 ∧
    p.2 < Real.sqrt (a^2 - p.1^2) ∧
    (∀ (q : ℝ × ℝ),
      q.1 > 0 ∧ q.1 < c ∧ q.2 > 0 ∧ q.2 < Real.sqrt (a^2 - q.1^2) →
      6 * (Real.sqrt ((0 - p.1)^2 + p.2^2) +
           Real.sqrt ((c - p.1)^2 + p.2^2) +
           Real.sqrt (p.1^2 + (a - p.2)^2))
      ≤ 6 * (Real.sqrt ((0 - q.1)^2 + q.2^2) +
             Real.sqrt ((c - q.1)^2 + q.2^2) +
             Real.sqrt (q.1^2 + (a - q.2)^2))) ∧
    6 * (Real.sqrt ((0 - p.1)^2 + p.2^2) +
         Real.sqrt ((c - p.1)^2 + p.2^2) +
         Real.sqrt (p.1^2 + (a - p.2)^2)) = 6 * Real.sqrt 19 :=
by sorry


end NUMINAMATH_CALUDE_minimum_distance_triangle_warehouse_l3167_316714


namespace NUMINAMATH_CALUDE_vector_magnitude_range_l3167_316721

/-- Given unit vectors e₁ and e₂ with an angle of 120° between them, 
    and x, y ∈ ℝ such that |x*e₁ + y*e₂| = √3, 
    prove that 1 ≤ |x*e₁ - y*e₂| ≤ 3 -/
theorem vector_magnitude_range (e₁ e₂ : ℝ × ℝ) (x y : ℝ) :
  (e₁.1^2 + e₁.2^2 = 1) →  -- e₁ is a unit vector
  (e₂.1^2 + e₂.2^2 = 1) →  -- e₂ is a unit vector
  (e₁.1 * e₂.1 + e₁.2 * e₂.2 = -1/2) →  -- angle between e₁ and e₂ is 120°
  ((x*e₁.1 + y*e₂.1)^2 + (x*e₁.2 + y*e₂.2)^2 = 3) →  -- |x*e₁ + y*e₂| = √3
  1 ≤ ((x*e₁.1 - y*e₂.1)^2 + (x*e₁.2 - y*e₂.2)^2) ∧ 
  ((x*e₁.1 - y*e₂.1)^2 + (x*e₁.2 - y*e₂.2)^2) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_range_l3167_316721


namespace NUMINAMATH_CALUDE_interior_angle_of_17_sided_polygon_l3167_316780

theorem interior_angle_of_17_sided_polygon (S : ℝ) (x : ℝ) : 
  S = (17 - 2) * 180 ∧ S - x = 2570 → x = 130 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_of_17_sided_polygon_l3167_316780


namespace NUMINAMATH_CALUDE_arrow_sequence_for_multiples_of_four_l3167_316749

def arrow_direction (n : ℕ) : Bool × Bool :=
  if n % 4 = 0 then (false, true) else (true, false)

theorem arrow_sequence_for_multiples_of_four (n : ℕ) (h : n % 4 = 0) :
  arrow_direction n = (false, true) := by sorry

end NUMINAMATH_CALUDE_arrow_sequence_for_multiples_of_four_l3167_316749


namespace NUMINAMATH_CALUDE_max_daily_profit_l3167_316760

/-- The daily profit function for a factory -/
def daily_profit (x : ℕ) : ℚ :=
  -4/3 * (x^3 : ℚ) + 3600 * (x : ℚ)

/-- The maximum daily production capacity -/
def max_production : ℕ := 40

/-- Theorem stating the maximum daily profit and the production quantity that achieves it -/
theorem max_daily_profit :
  ∃ (x : ℕ), x ≤ max_production ∧
    (∀ (y : ℕ), y ≤ max_production → daily_profit y ≤ daily_profit x) ∧
    x = 30 ∧ daily_profit x = 72000 := by
  sorry

end NUMINAMATH_CALUDE_max_daily_profit_l3167_316760


namespace NUMINAMATH_CALUDE_expand_difference_of_squares_l3167_316799

theorem expand_difference_of_squares (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_difference_of_squares_l3167_316799


namespace NUMINAMATH_CALUDE_multiple_sum_properties_l3167_316767

theorem multiple_sum_properties (x y : ℤ) 
  (hx : ∃ (m : ℤ), x = 6 * m) 
  (hy : ∃ (n : ℤ), y = 12 * n) : 
  (∃ (k : ℤ), x + y = 2 * k) ∧ (∃ (l : ℤ), x + y = 6 * l) := by
  sorry

end NUMINAMATH_CALUDE_multiple_sum_properties_l3167_316767
