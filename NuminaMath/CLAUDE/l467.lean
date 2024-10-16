import Mathlib

namespace NUMINAMATH_CALUDE_root_product_equality_l467_46701

theorem root_product_equality : 
  (16 : ℝ) ^ (1/5) * (64 : ℝ) ^ (1/6) = 2 * (16 : ℝ) ^ (1/5) :=
by sorry

end NUMINAMATH_CALUDE_root_product_equality_l467_46701


namespace NUMINAMATH_CALUDE_initial_girls_count_l467_46720

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 35 / 100 →
  ((initial_girls : ℚ) - 3) / (total : ℚ) = 25 / 100 →
  initial_girls = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l467_46720


namespace NUMINAMATH_CALUDE_power_of_three_product_fourth_root_l467_46778

theorem power_of_three_product_fourth_root (x : ℝ) : 
  (3^12 * 3^8)^(1/4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_product_fourth_root_l467_46778


namespace NUMINAMATH_CALUDE_polynomial_factorization_l467_46750

theorem polynomial_factorization (x : ℝ) : 
  x^2 - 6*x + 9 - 49*x^4 = (-7*x^2 + x - 3) * (7*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l467_46750


namespace NUMINAMATH_CALUDE_circle_center_proof_l467_46770

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if a point is equidistant from two parallel lines -/
def equidistantFromParallelLines (p : Point) (l1 l2 : Line) : Prop :=
  abs (l1.a * p.x + l1.b * p.y - l1.c) = abs (l2.a * p.x + l2.b * p.y - l2.c)

theorem circle_center_proof (l1 l2 l3 : Line) (p : Point) :
  l1.a = 3 ∧ l1.b = -4 ∧ l1.c = 12 ∧
  l2.a = 3 ∧ l2.b = -4 ∧ l2.c = -24 ∧
  l3.a = 1 ∧ l3.b = -2 ∧ l3.c = 0 ∧
  p.x = -6 ∧ p.y = -3 →
  pointOnLine p l3 ∧ equidistantFromParallelLines p l1 l2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_proof_l467_46770


namespace NUMINAMATH_CALUDE_min_value_expression_l467_46785

theorem min_value_expression (x : ℝ) : (x^2 + 7) / Real.sqrt (x^2 + 3) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l467_46785


namespace NUMINAMATH_CALUDE_vector_addition_l467_46798

/-- Given two vectors OA and AB in 2D space, prove that OB is their sum. -/
theorem vector_addition (OA AB : ℝ × ℝ) (h1 : OA = (-2, 3)) (h2 : AB = (-1, -4)) :
  OA + AB = (-3, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l467_46798


namespace NUMINAMATH_CALUDE_solve_for_t_l467_46742

theorem solve_for_t (s t : ℚ) (eq1 : 8 * s + 7 * t = 145) (eq2 : s = t + 3) : t = 121 / 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l467_46742


namespace NUMINAMATH_CALUDE_perpendicular_chords_theorem_l467_46794

/-- 
Given a circle with radius R and two perpendicular chords intersecting at point M,
this theorem proves two properties:
1. The sum of squares of the four segments formed by the intersection is 4R^2.
2. If the distance from the center to M is d, the sum of squares of chord lengths is 8R^2 - 4d^2.
-/
theorem perpendicular_chords_theorem (R d : ℝ) (h : d ≥ 0) :
  ∃ (AM MB CM MD : ℝ),
    (AM ≥ 0) ∧ (MB ≥ 0) ∧ (CM ≥ 0) ∧ (MD ≥ 0) ∧
    (AM^2 + MB^2 + CM^2 + MD^2 = 4 * R^2) ∧
    ∃ (AB CD : ℝ),
      (AB ≥ 0) ∧ (CD ≥ 0) ∧
      (AB^2 + CD^2 = 8 * R^2 - 4 * d^2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_chords_theorem_l467_46794


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l467_46796

/-- Given an examination where 740 students appeared and 481 failed,
    prove that 35% of students passed. -/
theorem exam_pass_percentage 
  (total_students : ℕ) 
  (failed_students : ℕ) 
  (h1 : total_students = 740)
  (h2 : failed_students = 481) : 
  (total_students - failed_students : ℚ) / total_students * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l467_46796


namespace NUMINAMATH_CALUDE_warehouse_total_boxes_l467_46733

/-- Represents the number of boxes in each warehouse --/
structure Warehouses where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ

/-- Conditions for the warehouse problem --/
def warehouseConditions (w : Warehouses) : Prop :=
  ∃ x : ℕ,
    w.A = x ∧
    w.B = 3 * x ∧
    w.C = (3 * x) / 2 + 100 ∧
    w.D = 3 * x + 150 ∧
    w.E = 4 * x - 50 ∧
    w.B = w.E + 300

/-- The theorem to be proved --/
theorem warehouse_total_boxes (w : Warehouses) :
  warehouseConditions w → w.A + w.B + w.C + w.D + w.E = 4575 := by
  sorry


end NUMINAMATH_CALUDE_warehouse_total_boxes_l467_46733


namespace NUMINAMATH_CALUDE_intersection_M_N_union_complements_M_N_l467_46781

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x ≥ 1}

-- Define set N
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- Theorem for the intersection of M and N
theorem intersection_M_N : M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 5} := by sorry

-- Theorem for the union of complements of M and N
theorem union_complements_M_N : (U \ M) ∪ (U \ N) = {x : ℝ | x < 1 ∨ x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_union_complements_M_N_l467_46781


namespace NUMINAMATH_CALUDE_parabola_vertex_l467_46728

/-- Given a quadratic function f(x) = -x^2 + cx + d where c and d are real numbers,
    and the solution to f(x) ≤ 0 is (-∞, -4] ∪ [6, ∞),
    prove that the vertex of the parabola is (1, 25). -/
theorem parabola_vertex (c d : ℝ) 
  (h : ∀ x, -x^2 + c*x + d ≤ 0 ↔ x ∈ Set.Iic (-4) ∪ Set.Ici 6) : 
  let f := fun x => -x^2 + c*x + d
  (1, f 1) = (1, 25) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l467_46728


namespace NUMINAMATH_CALUDE_fixed_points_satisfy_circle_l467_46715

def moving_circle (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*m*y + 6*m - 2 = 0

theorem fixed_points_satisfy_circle :
  (∀ m : ℝ, moving_circle 1 1 m) ∧ (∀ m : ℝ, moving_circle (1/5) (7/5) m) :=
by sorry

end NUMINAMATH_CALUDE_fixed_points_satisfy_circle_l467_46715


namespace NUMINAMATH_CALUDE_total_windows_l467_46737

theorem total_windows (installed : ℕ) (install_time : ℕ) (remaining_time : ℕ) : 
  installed = 8 → install_time = 8 → remaining_time = 48 → 
  installed + (remaining_time / install_time) = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_windows_l467_46737


namespace NUMINAMATH_CALUDE_tangent_line_constant_l467_46723

/-- The value of m for which y = -x + m is tangent to y = x^2 - 3ln(x) -/
theorem tangent_line_constant (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 
    x^2 - 3 * Real.log x = -x + m ∧ 
    2 * x - 3 / x = -1) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_constant_l467_46723


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l467_46702

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3)^2 - 3*(a 3) - 5 = 0 →
  (a 10)^2 - 3*(a 10) - 5 = 0 →
  a 5 + a 8 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l467_46702


namespace NUMINAMATH_CALUDE_min_sum_m_n_l467_46735

theorem min_sum_m_n (m n : ℕ+) (h : 45 * m = n^3) : 
  (∀ (m' n' : ℕ+), 45 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 90 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l467_46735


namespace NUMINAMATH_CALUDE_integral_x_plus_inverse_x_l467_46789

theorem integral_x_plus_inverse_x : ∫ x in (1 : ℝ)..2, (x + 1/x) = 3/2 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_inverse_x_l467_46789


namespace NUMINAMATH_CALUDE_range_of_a_l467_46795

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a + 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition for point M
def condition_M (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C a x y ∧ 
    (x^2 + (y - 2)^2) + (x^2 + y^2) = 10

-- The main theorem
theorem range_of_a :
  ∀ a : ℝ, condition_M a ↔ 0 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l467_46795


namespace NUMINAMATH_CALUDE_proposition_two_l467_46745

theorem proposition_two (a b c : ℝ) (h1 : c > 1) (h2 : 0 < b) (h3 : b < 2) :
  a^2 + a*b + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_proposition_two_l467_46745


namespace NUMINAMATH_CALUDE_fifth_day_sale_l467_46741

def average_sale : ℕ := 625
def num_days : ℕ := 5
def day1_sale : ℕ := 435
def day2_sale : ℕ := 927
def day3_sale : ℕ := 855
def day4_sale : ℕ := 230

theorem fifth_day_sale :
  ∃ (day5_sale : ℕ),
    day5_sale = average_sale * num_days - (day1_sale + day2_sale + day3_sale + day4_sale) ∧
    day5_sale = 678 := by
  sorry

end NUMINAMATH_CALUDE_fifth_day_sale_l467_46741


namespace NUMINAMATH_CALUDE_x_gt_2_necessary_not_sufficient_for_x_gt_3_l467_46729

theorem x_gt_2_necessary_not_sufficient_for_x_gt_3 :
  (∀ x : ℝ, x > 3 → x > 2) ∧ 
  (∃ x : ℝ, x > 2 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_2_necessary_not_sufficient_for_x_gt_3_l467_46729


namespace NUMINAMATH_CALUDE_complex_equation_solution_l467_46793

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I + a) * (1 + Complex.I) = b * Complex.I → a + b * Complex.I = 1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l467_46793


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l467_46740

theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum_positive : a + b > 0) : 
  f a + f b > f (-a) + f (-b) := by
sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l467_46740


namespace NUMINAMATH_CALUDE_find_a_value_l467_46753

theorem find_a_value (x y a : ℝ) 
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (a * x + 6 * y) / (x - 2 * y) = 27) :
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l467_46753


namespace NUMINAMATH_CALUDE_unique_a_value_l467_46754

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_value (a : ℝ) (h : 1 ∈ A a) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l467_46754


namespace NUMINAMATH_CALUDE_sqrt_two_power_2000_identity_l467_46714

theorem sqrt_two_power_2000_identity : 
  (Real.sqrt 2 + 1)^2000 * (Real.sqrt 2 - 1)^2000 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_power_2000_identity_l467_46714


namespace NUMINAMATH_CALUDE_x_equals_cos_alpha_l467_46752

/-- Given two squares with side length 1/2 inclined at an angle 2α, 
    x is the length of the line segment connecting the midpoints of 
    the non-intersecting sides of the squares -/
def x (α : Real) : Real :=
  sorry

theorem x_equals_cos_alpha (α : Real) : x α = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_x_equals_cos_alpha_l467_46752


namespace NUMINAMATH_CALUDE_carl_index_cards_cost_l467_46755

/-- Calculates the total cost of index cards for Carl's students --/
def total_cost_index_cards (
  cards_6th : ℕ)  -- Number of cards for each 6th grader
  (cards_7th : ℕ) -- Number of cards for each 7th grader
  (cards_8th : ℕ) -- Number of cards for each 8th grader
  (students_6th : ℕ) -- Number of 6th grade students per period
  (students_7th : ℕ) -- Number of 7th grade students per period
  (students_8th : ℕ) -- Number of 8th grade students per period
  (periods : ℕ) -- Number of periods per day
  (pack_size : ℕ) -- Number of cards per pack
  (cost_3x5 : ℕ) -- Cost of a pack of 3x5 cards in dollars
  (cost_4x6 : ℕ) -- Cost of a pack of 4x6 cards in dollars
  : ℕ :=
  let total_cards_6th := cards_6th * students_6th * periods
  let total_cards_7th := cards_7th * students_7th * periods
  let total_cards_8th := cards_8th * students_8th * periods
  let packs_6th := (total_cards_6th + pack_size - 1) / pack_size
  let packs_7th := (total_cards_7th + pack_size - 1) / pack_size
  let packs_8th := (total_cards_8th + pack_size - 1) / pack_size
  packs_6th * cost_3x5 + packs_7th * cost_3x5 + packs_8th * cost_4x6

theorem carl_index_cards_cost : 
  total_cost_index_cards 8 10 12 20 25 30 6 50 3 4 = 326 := by
  sorry

end NUMINAMATH_CALUDE_carl_index_cards_cost_l467_46755


namespace NUMINAMATH_CALUDE_chess_team_arrangement_l467_46713

/-- Represents the number of boys on the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls on the chess team -/
def num_girls : ℕ := 2

/-- Represents the total number of team members -/
def total_members : ℕ := num_boys + num_girls

/-- Represents the number of ways to arrange the team members according to the specified conditions -/
def arrangements : ℕ := num_girls.factorial * num_boys.factorial

theorem chess_team_arrangement : arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_arrangement_l467_46713


namespace NUMINAMATH_CALUDE_rectangle_combination_forms_square_l467_46705

theorem rectangle_combination_forms_square (n : Nat) (h : n = 100) :
  ∃ (square : Set (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ square → x < n ∧ y < n) ∧ 
    (∀ (x y : ℕ), (x, y) ∈ square → (x + 1, y) ∈ square ∨ (x, y + 1) ∈ square) ∧
    (∃ (x y : ℕ), 
      (x, y) ∈ square ∧ 
      (x + 1, y) ∈ square ∧ 
      (x, y + 1) ∈ square ∧ 
      (x + 1, y + 1) ∈ square) :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_combination_forms_square_l467_46705


namespace NUMINAMATH_CALUDE_video_game_lives_l467_46751

/-- The total number of lives for a group of friends in a video game -/
def totalLives (numFriends : ℕ) (livesPerFriend : ℕ) : ℕ :=
  numFriends * livesPerFriend

/-- Theorem: Given 15 friends, each with 25 lives, the total number of lives is 375 -/
theorem video_game_lives : totalLives 15 25 = 375 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l467_46751


namespace NUMINAMATH_CALUDE_game_probability_l467_46766

/-- The number of possible choices for each player -/
def num_choices : ℕ := 16

/-- The probability of not winning a prize in a single trial -/
def prob_not_winning : ℚ := 15 / 16

theorem game_probability :
  (1 : ℚ) - (num_choices : ℚ) / ((num_choices : ℚ) * (num_choices : ℚ)) = prob_not_winning :=
by sorry

end NUMINAMATH_CALUDE_game_probability_l467_46766


namespace NUMINAMATH_CALUDE_parabolas_intersection_l467_46707

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := 9 * x^2 + 6 * x + 2

/-- The set of intersection points of the two parabolas -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = g p.1 ∧ p.2 = f p.1}

theorem parabolas_intersection :
  intersection_points = {(0, 2), (-5/3, 17)} := by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l467_46707


namespace NUMINAMATH_CALUDE_condition1_correct_condition2_correct_condition3_correct_l467_46736

-- Define the number of teachers, male students, and female students
def num_teachers : ℕ := 2
def num_male_students : ℕ := 3
def num_female_students : ℕ := 3

-- Define the total number of people
def total_people : ℕ := num_teachers + num_male_students + num_female_students

-- Function to calculate the number of arrangements for condition 1
def arrangements_condition1 : ℕ := sorry

-- Function to calculate the number of arrangements for condition 2
def arrangements_condition2 : ℕ := sorry

-- Function to calculate the number of arrangements for condition 3
def arrangements_condition3 : ℕ := sorry

-- Theorem for condition 1
theorem condition1_correct : arrangements_condition1 = 4320 := by sorry

-- Theorem for condition 2
theorem condition2_correct : arrangements_condition2 = 30240 := by sorry

-- Theorem for condition 3
theorem condition3_correct : arrangements_condition3 = 6720 := by sorry

end NUMINAMATH_CALUDE_condition1_correct_condition2_correct_condition3_correct_l467_46736


namespace NUMINAMATH_CALUDE_percentage_increase_decrease_l467_46787

theorem percentage_increase_decrease (x : ℝ) (h : x > 0) :
  x * (1 + 0.25) * (1 - 0.20) = x := by
  sorry

#check percentage_increase_decrease

end NUMINAMATH_CALUDE_percentage_increase_decrease_l467_46787


namespace NUMINAMATH_CALUDE_floor_sum_equality_l467_46708

theorem floor_sum_equality (n : ℕ+) : 
  ∑' k : ℕ, ⌊(n + 2^k : ℝ) / 2^(k+1)⌋ = n := by sorry

end NUMINAMATH_CALUDE_floor_sum_equality_l467_46708


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_equation_l467_46748

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the distance from the right focus to the left vertex is equal to twice
    the distance from it to the asymptote, then the equation of its asymptote
    is 4x ± 3y = 0. -/
theorem hyperbola_asymptote_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_distance : ∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 →
    (∃ (c : ℝ), a + c = 2 * (b * c / Real.sqrt (a^2 + b^2)))) :
  ∃ (k : ℝ), k > 0 ∧ (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (4*x = 3*y ∨ 4*x = -3*y)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_equation_l467_46748


namespace NUMINAMATH_CALUDE_max_sum_xyz_l467_46765

theorem max_sum_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
  16 * x₀ * y₀ * z₀ = (x₀ + y₀)^2 * (x₀ + z₀)^2 ∧ x₀ + y₀ + z₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l467_46765


namespace NUMINAMATH_CALUDE_intersection_proof_l467_46704

def S : Set Nat := {0, 1, 3, 5, 7, 9}

theorem intersection_proof (A B : Set Nat) 
  (h1 : S = {0, 1, 3, 5, 7, 9})
  (h2 : (S \ A) = {0, 5, 9})
  (h3 : B = {3, 5, 7}) :
  A ∩ B = {3, 7} := by
sorry

end NUMINAMATH_CALUDE_intersection_proof_l467_46704


namespace NUMINAMATH_CALUDE_range_of_m_l467_46771

def α (x : ℝ) : Prop := x^2 - 3*x - 10 ≤ 0

def β (m x : ℝ) : Prop := m - 3 ≤ x ∧ x ≤ m + 6

theorem range_of_m :
  (∀ x, α x → ∃ m, β m x) →
  ∀ m, (∃ x, β m x) → -1 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l467_46771


namespace NUMINAMATH_CALUDE_parabola_intersection_l467_46747

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 4
def parabola2 (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 8

theorem parabola_intersection :
  ∀ x y : ℝ, parabola1 x = parabola2 x ∧ y = parabola1 x ↔ (x = 3 ∧ y = 20) ∨ (x = 4 ∧ y = 32) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l467_46747


namespace NUMINAMATH_CALUDE_tangent_circles_distance_l467_46732

/-- Given two circles of radius 5 that are externally tangent to each other
    and internally tangent to a circle of radius 13, the distance between
    their points of tangency with the larger circle is 2√39. -/
theorem tangent_circles_distance (r₁ r₂ R : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 5) (h₃ : R = 13) :
  let d := 2 * (R - r₁)  -- distance between centers of small circles and large circle
  let s := r₁ + r₂       -- distance between centers of small circles
  2 * Real.sqrt ((d ^ 2) - (s / 2) ^ 2) = 2 * Real.sqrt 39 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_distance_l467_46732


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l467_46760

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem first_term_of_geometric_sequence 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : geometric_sequence a r 4 = 24) 
  (h2 : geometric_sequence a r 5 = 48) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l467_46760


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_target_l467_46739

/-- A fraction with two-digit numerator and denominator -/
structure TwoDigitFraction :=
  (numerator : Nat)
  (denominator : Nat)
  (num_two_digit : 10 ≤ numerator ∧ numerator ≤ 99)
  (den_two_digit : 10 ≤ denominator ∧ denominator ≤ 99)

/-- The fraction 4/9 -/
def target : Rat := 4 / 9

/-- The fraction 41/92 -/
def smallest : TwoDigitFraction :=
  { numerator := 41
  , denominator := 92
  , num_two_digit := by sorry
  , den_two_digit := by sorry }

theorem smallest_fraction_greater_than_target :
  (smallest.numerator : Rat) / smallest.denominator > target ∧
  ∀ (f : TwoDigitFraction), 
    (f.numerator : Rat) / f.denominator > target → 
    (smallest.numerator : Rat) / smallest.denominator ≤ (f.numerator : Rat) / f.denominator :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_target_l467_46739


namespace NUMINAMATH_CALUDE_calculate_expression_solve_inequality_system_l467_46784

-- Part 1
theorem calculate_expression : (π - 3) ^ 0 + (-1) ^ 2023 - Real.sqrt 8 = -2 * Real.sqrt 2 := by
  sorry

-- Part 2
theorem solve_inequality_system {x : ℝ} : (4 * x - 3 > 9 ∧ 2 + x ≥ 0) ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_inequality_system_l467_46784


namespace NUMINAMATH_CALUDE_sequence_2023_l467_46786

theorem sequence_2023 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, a n > 0) →
  (∀ n, 2 * S n = a n * (a n + 1)) →
  a 2023 = 2023 := by
sorry

end NUMINAMATH_CALUDE_sequence_2023_l467_46786


namespace NUMINAMATH_CALUDE_blood_cells_in_first_sample_l467_46711

theorem blood_cells_in_first_sample
  (total_cells : ℕ)
  (second_sample_cells : ℕ)
  (h1 : total_cells = 7341)
  (h2 : second_sample_cells = 3120) :
  total_cells - second_sample_cells = 4221 := by
  sorry

end NUMINAMATH_CALUDE_blood_cells_in_first_sample_l467_46711


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_two_l467_46782

theorem log_expression_equals_negative_two :
  (Real.log 64 / Real.log 32) / (Real.log 2 / Real.log 32) -
  (Real.log 256 / Real.log 16) / (Real.log 2 / Real.log 16) = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_two_l467_46782


namespace NUMINAMATH_CALUDE_max_value_negative_x_min_value_greater_than_negative_one_l467_46768

-- Problem 1
theorem max_value_negative_x (x : ℝ) (hx : x < 0) :
  (x^2 + x + 1) / x ≤ -1 :=
by sorry

-- Problem 2
theorem min_value_greater_than_negative_one (x : ℝ) (hx : x > -1) :
  ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_negative_x_min_value_greater_than_negative_one_l467_46768


namespace NUMINAMATH_CALUDE_inequality_proof_l467_46790

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  Real.sqrt (1/a - a) + Real.sqrt (1/b - b) + Real.sqrt (1/c - c) ≥ 
  Real.sqrt (2*a) + Real.sqrt (2*b) + Real.sqrt (2*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l467_46790


namespace NUMINAMATH_CALUDE_grading_orders_mod_100_l467_46703

/-- The number of students --/
def num_students : ℕ := 40

/-- The number of problems per student --/
def problems_per_student : ℕ := 3

/-- The number of different grading orders --/
def N : ℕ := 2 * 3^(num_students - 2)

/-- Theorem stating the result of N modulo 100 --/
theorem grading_orders_mod_100 : N % 100 = 78 := by
  sorry

end NUMINAMATH_CALUDE_grading_orders_mod_100_l467_46703


namespace NUMINAMATH_CALUDE_paperclips_exceed_300_l467_46756

def paperclips (k : ℕ) : ℕ := 5 * 3^k

theorem paperclips_exceed_300 : 
  (∀ n < 4, paperclips n ≤ 300) ∧ paperclips 4 > 300 := by sorry

end NUMINAMATH_CALUDE_paperclips_exceed_300_l467_46756


namespace NUMINAMATH_CALUDE_divisibility_of_integer_part_l467_46762

theorem divisibility_of_integer_part (k : ℕ+) (n : ℕ) :
  let A : ℝ := k + 1/2 + Real.sqrt (k^2 + 1/4)
  (⌊A^n⌋ : ℤ) % k = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_integer_part_l467_46762


namespace NUMINAMATH_CALUDE_johann_mail_delivery_l467_46738

theorem johann_mail_delivery (total_mail : ℕ) (friend_mail : ℕ) (num_friends : ℕ) :
  total_mail = 180 →
  friend_mail = 41 →
  num_friends = 2 →
  total_mail - (friend_mail * num_friends) = 98 := by
  sorry

end NUMINAMATH_CALUDE_johann_mail_delivery_l467_46738


namespace NUMINAMATH_CALUDE_wax_needed_l467_46706

def total_wax : ℕ := 166
def current_wax : ℕ := 20

theorem wax_needed : total_wax - current_wax = 146 := by
  sorry

end NUMINAMATH_CALUDE_wax_needed_l467_46706


namespace NUMINAMATH_CALUDE_wuyang_football_school_runners_l467_46791

theorem wuyang_football_school_runners (x : ℕ) : 
  (x - 4) % 2 = 0 →
  (x - 5) % 3 = 0 →
  x % 5 = 0 →
  ∃ n : ℕ, x = n ^ 2 →
  250 - 10 ≤ x - 3 ∧ x - 3 ≤ 250 + 10 →
  x = 260 := by
sorry

end NUMINAMATH_CALUDE_wuyang_football_school_runners_l467_46791


namespace NUMINAMATH_CALUDE_inequality_solution_set_minimum_value_no_positive_solution_l467_46775

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2/3 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value :
  ∃ (k : ℝ), k = 1 ∧ ∀ (x : ℝ), f x ≥ k := by sorry

-- Theorem for non-existence of positive a and b
theorem no_positive_solution :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b = 1 ∧ 1/a + 2/b = 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_minimum_value_no_positive_solution_l467_46775


namespace NUMINAMATH_CALUDE_burt_basil_profit_l467_46780

/-- Calculate the net profit from Burt's basil plants -/
theorem burt_basil_profit :
  let seed_cost : ℕ := 200  -- in cents
  let soil_cost : ℕ := 800  -- in cents
  let total_plants : ℕ := 20
  let price_per_plant : ℕ := 500  -- in cents
  
  let total_cost : ℕ := seed_cost + soil_cost
  let total_revenue : ℕ := total_plants * price_per_plant
  let net_profit : ℤ := total_revenue - total_cost
  
  net_profit = 9000  -- 90.00 in cents
  := by sorry

end NUMINAMATH_CALUDE_burt_basil_profit_l467_46780


namespace NUMINAMATH_CALUDE_hexagon_coins_proof_l467_46719

/-- The number of coins needed to construct a hexagon with side length n -/
def hexagon_coins (n : ℕ) : ℕ := 3 * n * (n - 1) + 1

theorem hexagon_coins_proof :
  (hexagon_coins 2 = 7) ∧
  (hexagon_coins 3 = 19) ∧
  (hexagon_coins 10 = 271) :=
by sorry

end NUMINAMATH_CALUDE_hexagon_coins_proof_l467_46719


namespace NUMINAMATH_CALUDE_ellipse_equation_l467_46776

/-- The standard equation of an ellipse with given eccentricity and major axis length -/
theorem ellipse_equation (e : ℝ) (major_axis : ℝ) :
  e = 2/3 →
  major_axis = 6 →
  ∃ (a b : ℝ),
    a = major_axis / 2 ∧
    b^2 = a^2 * (1 - e^2) ∧
    ((∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1) ∨
     (∀ x y : ℝ, x^2/b^2 + y^2/a^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l467_46776


namespace NUMINAMATH_CALUDE_nines_in_range_70_l467_46761

def count_nines (n : ℕ) : ℕ :=
  (n / 10) + (if n % 10 ≥ 9 then 1 else 0)

theorem nines_in_range_70 : count_nines 70 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nines_in_range_70_l467_46761


namespace NUMINAMATH_CALUDE_application_outcomes_count_l467_46764

/-- The number of colleges available for applications -/
def num_colleges : ℕ := 3

/-- The number of students applying to colleges -/
def num_students : ℕ := 3

/-- The total number of possible application outcomes when three students apply to three colleges,
    with the condition that the first two students must apply to different colleges -/
def total_outcomes : ℕ := num_colleges * (num_colleges - 1) * num_colleges

theorem application_outcomes_count : total_outcomes = 18 := by
  sorry

end NUMINAMATH_CALUDE_application_outcomes_count_l467_46764


namespace NUMINAMATH_CALUDE_first_tract_width_l467_46730

/-- Given two rectangular tracts of land with specified dimensions and combined area,
    calculates the width of the first tract. -/
theorem first_tract_width (length1 : ℝ) (length2 width2 : ℝ) (combined_area : ℝ) : 
  length1 = 300 →
  length2 = 250 →
  width2 = 630 →
  combined_area = 307500 →
  combined_area = length1 * (combined_area - length2 * width2) / length1 + length2 * width2 →
  (combined_area - length2 * width2) / length1 = 500 := by
sorry

end NUMINAMATH_CALUDE_first_tract_width_l467_46730


namespace NUMINAMATH_CALUDE_seventh_term_is_seven_l467_46797

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Fourth term is 4
  fourth_term : a + 3*d = 4

/-- The seventh term of the arithmetic sequence is 7 -/
theorem seventh_term_is_seven (seq : ArithmeticSequence) : seq.a + 6*seq.d = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_seven_l467_46797


namespace NUMINAMATH_CALUDE_equation_solution_l467_46783

theorem equation_solution (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))) :
  x = (3 + Real.sqrt 13) / 2 ∧ y = (3 + Real.sqrt 13) / 2 ∧ z = (3 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l467_46783


namespace NUMINAMATH_CALUDE_sin_negative_390_degrees_l467_46734

theorem sin_negative_390_degrees : Real.sin (-(390 * π / 180)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_390_degrees_l467_46734


namespace NUMINAMATH_CALUDE_no_solution_for_sqrt_equation_l467_46759

theorem no_solution_for_sqrt_equation :
  ¬ ∃ x : ℝ, x > 9 ∧ Real.sqrt (x - 9) + 3 = Real.sqrt (x + 9) - 3 := by
  sorry


end NUMINAMATH_CALUDE_no_solution_for_sqrt_equation_l467_46759


namespace NUMINAMATH_CALUDE_unique_prime_203B21_l467_46710

/-- A function that generates a six-digit number of the form 203B21 given a single digit B -/
def generate_number (B : Nat) : Nat :=
  203000 + B * 100 + 21

/-- Predicate to check if a number is of the form 203B21 where B is a single digit -/
def is_valid_form (n : Nat) : Prop :=
  ∃ B : Nat, B < 10 ∧ n = generate_number B

theorem unique_prime_203B21 :
  ∃! n : Nat, is_valid_form n ∧ Nat.Prime n ∧ n = 203521 := by sorry

end NUMINAMATH_CALUDE_unique_prime_203B21_l467_46710


namespace NUMINAMATH_CALUDE_ellipse_m_range_l467_46777

/-- The equation of the curve -/
def curve_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 2) + y^2 / (6 - m) = 1

/-- Definition of an ellipse in terms of its equation -/
def is_ellipse (m : ℝ) : Prop :=
  (∀ x y, curve_equation x y m → x^2 / (m - 2) > 0 ∧ y^2 / (6 - m) > 0) ∧
  m - 2 ≠ 6 - m

/-- Theorem: The range of m for which the curve is an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ (2 < m ∧ m < 6 ∧ m ≠ 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l467_46777


namespace NUMINAMATH_CALUDE_seashell_solution_l467_46757

/-- The number of seashells found by Mary, Jessica, and Kevin -/
def seashell_problem (mary_shells jessica_shells : ℕ) (kevin_multiplier : ℕ) : Prop :=
  let kevin_shells := kevin_multiplier * mary_shells
  mary_shells + jessica_shells + kevin_shells = 113

/-- Theorem stating the solution to the seashell problem -/
theorem seashell_solution : seashell_problem 18 41 3 := by
  sorry

end NUMINAMATH_CALUDE_seashell_solution_l467_46757


namespace NUMINAMATH_CALUDE_card_draw_probability_l467_46758

def standard_deck := 52
def face_cards := 12
def hearts := 13
def tens := 4

theorem card_draw_probability : 
  let p1 := face_cards / standard_deck
  let p2 := hearts / (standard_deck - 1)
  let p3 := tens / (standard_deck - 2)
  p1 * p2 * p3 = 1 / 217 := by
  sorry

end NUMINAMATH_CALUDE_card_draw_probability_l467_46758


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l467_46721

theorem max_value_of_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (2 * x^2 + 3 * y^2 + 5) ≤ Real.sqrt 28 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (2 * x + 3 * y + 4) / Real.sqrt (2 * x^2 + 3 * y^2 + 5) = Real.sqrt 28 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l467_46721


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l467_46712

theorem sum_of_reciprocals (x y z a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : x * y / (x - y) = a)
  (h2 : x * z / (x - z) = b)
  (h3 : y * z / (y - z) = c) :
  1 / x + 1 / y + 1 / z = (1 / a + 1 / b + 1 / c) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l467_46712


namespace NUMINAMATH_CALUDE_polynomial_expansion_l467_46744

-- Define the polynomials
def p (x : ℝ) : ℝ := 3*x^2 - 4*x + 3
def q (x : ℝ) : ℝ := -2*x^2 + 3*x - 4

-- State the theorem
theorem polynomial_expansion :
  ∀ x : ℝ, p x * q x = -6*x^4 + 17*x^3 - 30*x^2 + 25*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l467_46744


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l467_46725

theorem complex_fraction_equality : 
  let x : ℂ := (1 + Complex.I * Real.sqrt 3) / 3
  1 / (x^2 + x) = 9/76 - (45 * Complex.I * Real.sqrt 3)/76 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l467_46725


namespace NUMINAMATH_CALUDE_rectangle_side_length_l467_46788

/-- 
Given a rectangular arrangement with all right angles, where the top length 
consists of segments 3 cm, 2 cm, Y cm, and 1 cm sequentially, and the total 
bottom length is 11 cm, prove that Y = 5 cm.
-/
theorem rectangle_side_length (Y : ℝ) : 
  (3 : ℝ) + 2 + Y + 1 = 11 → Y = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l467_46788


namespace NUMINAMATH_CALUDE_quadratic_form_simplification_l467_46746

theorem quadratic_form_simplification 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ K : ℝ, ∀ x : ℝ, 
    (x + a)^2 / ((a - b) * (a - c)) + 
    (x + b)^2 / ((b - a) * (b - c + 2)) + 
    (x + c)^2 / ((c - a) * (c - b)) = 
    x^2 - (a + b + c) * x + K :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_simplification_l467_46746


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l467_46709

-- Define the equation of the hyperbola
def hyperbola_eq (x y k : ℝ) : Prop :=
  x^2 / (3 - k) - y^2 / (k - 1) = 1

-- Define the condition for k to represent a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola_eq x y k

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ (1 < k ∧ k < 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l467_46709


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l467_46769

theorem two_digit_number_puzzle :
  ∃! n : ℕ, 
    10 ≤ n ∧ n < 100 ∧  -- n is a two-digit number
    (n / 10 = 2 * (n % 10)) ∧  -- tens digit is twice the units digit
    (n - ((n % 10) * 10 + (n / 10)) = 36)  -- swapping digits results in 36 less
  :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l467_46769


namespace NUMINAMATH_CALUDE_five_digit_palindrome_digits_l467_46772

/-- A function that calculates the number of 5-digit palindromes that can be formed
    using n distinct digits -/
def palindrome_count (n : ℕ) : ℕ := n * n * n

/-- The theorem stating that if there are 125 possible 5-digit palindromes formed
    using some distinct digits, then the number of distinct digits is 5 -/
theorem five_digit_palindrome_digits :
  (∃ (n : ℕ), n > 0 ∧ palindrome_count n = 125) →
  (∃ (n : ℕ), n > 0 ∧ palindrome_count n = 125 ∧ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_palindrome_digits_l467_46772


namespace NUMINAMATH_CALUDE_inequality_reverse_l467_46743

theorem inequality_reverse (a b : ℝ) (h : a > b) : -4 * a < -4 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_reverse_l467_46743


namespace NUMINAMATH_CALUDE_log_equation_solution_l467_46700

theorem log_equation_solution (m n : ℝ) (b : ℝ) (h : m > 0) (h' : n > 0) :
  Real.log m^2 / Real.log 10 = b - Real.log n^3 / Real.log 10 →
  m = (10^b / n^3)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l467_46700


namespace NUMINAMATH_CALUDE_number_equation_l467_46779

theorem number_equation (x : ℤ) : 45 - (x - (37 - (15 - 20))) = 59 ↔ x = 28 := by sorry

end NUMINAMATH_CALUDE_number_equation_l467_46779


namespace NUMINAMATH_CALUDE_quadruple_count_l467_46724

/-- The number of ordered quadruples of positive even integers that sum to 104 -/
def n : ℕ := sorry

/-- Predicate for a quadruple of positive even integers -/
def is_valid_quadruple (x₁ x₂ x₃ x₄ : ℕ) : Prop :=
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
  Even x₁ ∧ Even x₂ ∧ Even x₃ ∧ Even x₄ ∧
  x₁ + x₂ + x₃ + x₄ = 104

/-- The main theorem stating that n/100 equals 208.25 -/
theorem quadruple_count : (n : ℚ) / 100 = 208.25 := by sorry

end NUMINAMATH_CALUDE_quadruple_count_l467_46724


namespace NUMINAMATH_CALUDE_min_distance_to_2i_l467_46773

theorem min_distance_to_2i (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - 2*Complex.I) ≥ 1 ∧ ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - 2*Complex.I) = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_2i_l467_46773


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l467_46792

theorem hot_dogs_remainder : 25197631 % 17 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l467_46792


namespace NUMINAMATH_CALUDE_polynomial_root_sum_squares_l467_46726

theorem polynomial_root_sum_squares (a b c t : ℝ) : 
  (∀ x : ℝ, x^3 - 6*x^2 + 8*x - 2 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^4 - 12*t^2 - 4*t = -4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_squares_l467_46726


namespace NUMINAMATH_CALUDE_min_m_and_range_l467_46767

noncomputable section

def f (x : ℝ) := (1 + x)^2 - 2 * Real.log (1 + x)

theorem min_m_and_range (x₀ : ℝ) (h : x₀ ∈ Set.Icc 0 1) :
  (∀ x ∈ Set.Icc 0 1, f x - (4 - 2 * Real.log 2) ≤ 0) ∧
  (f x₀ - 1 ≤ 0 → ∀ m : ℝ, f x₀ - m ≤ 0 → m ≥ 1) :=
by sorry

end

end NUMINAMATH_CALUDE_min_m_and_range_l467_46767


namespace NUMINAMATH_CALUDE_f_max_value_l467_46716

open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_derivative (x : ℝ) : deriv f x = (1 / x^2 - 2 * f x) / x

axiom f_initial_value : f 1 = 2

-- State the theorem
theorem f_max_value :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x ∧ f x = Real.exp 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l467_46716


namespace NUMINAMATH_CALUDE_exists_perpendicular_plane_containing_line_l467_46717

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Predicate to check if a line intersects a plane -/
def intersects (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a plane contains a line -/
def contains_line (β : Plane3D) (l : Line3D) : Prop :=
  sorry

/-- Predicate to check if two planes are perpendicular -/
def perpendicular_planes (α β : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line intersects a plane but is not perpendicular to it,
    then there exists a plane containing the line that is perpendicular to the original plane -/
theorem exists_perpendicular_plane_containing_line
  (l : Line3D) (α : Plane3D)
  (h1 : intersects l α)
  (h2 : ¬perpendicular_line_plane l α) :
  ∃ β : Plane3D, contains_line β l ∧ perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_plane_containing_line_l467_46717


namespace NUMINAMATH_CALUDE_continued_fraction_value_l467_46731

theorem continued_fraction_value : ∃ x : ℝ, x = 3 + 4 / (2 + 4 / x) ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l467_46731


namespace NUMINAMATH_CALUDE_gus_buys_two_dozen_l467_46718

def golf_balls_per_dozen : ℕ := 12

def dans_dozens : ℕ := 5
def chris_golf_balls : ℕ := 48
def total_golf_balls : ℕ := 132

def gus_dozens : ℕ := total_golf_balls / golf_balls_per_dozen - dans_dozens - chris_golf_balls / golf_balls_per_dozen

theorem gus_buys_two_dozen : gus_dozens = 2 := by
  sorry

end NUMINAMATH_CALUDE_gus_buys_two_dozen_l467_46718


namespace NUMINAMATH_CALUDE_function_equality_implies_m_range_l467_46763

theorem function_equality_implies_m_range :
  ∀ (m : ℝ), m > 0 →
  (∃ (x₁ : ℝ), x₁ ∈ Set.Icc 0 3 ∧
    (∀ (x₂ : ℝ), x₂ ∈ Set.Icc 0 3 →
      x₁^2 - 4*x₁ + 3 = m*(x₂ - 1) + 2)) →
  m ∈ Set.Ioo 0 (1/2) := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_range_l467_46763


namespace NUMINAMATH_CALUDE_legs_on_queen_mary_ii_l467_46722

/-- Calculates the total number of legs on a ship with cats and humans. -/
def total_legs (total_heads : ℕ) (num_cats : ℕ) : ℕ :=
  let num_humans := total_heads - num_cats
  let cat_legs := num_cats * 4
  let human_legs := (num_humans - 1) * 2 + 1
  cat_legs + human_legs

/-- Theorem stating that the total number of legs is 45 under given conditions. -/
theorem legs_on_queen_mary_ii :
  total_legs 16 7 = 45 := by
  sorry

end NUMINAMATH_CALUDE_legs_on_queen_mary_ii_l467_46722


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l467_46749

/-- Given sets M and N in the real numbers, prove that the intersection of the complement of M and N is the set of all real numbers less than -2. -/
theorem complement_M_intersect_N (M N : Set ℝ) 
  (hM : M = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (hN : N = {x : ℝ | x < 1}) :
  (Mᶜ ∩ N) = {x : ℝ | x < -2} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l467_46749


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l467_46774

def total_amount_spent (food_price : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let price_with_tax := food_price * (1 + sales_tax_rate)
  let total := price_with_tax * (1 + tip_rate)
  total

theorem dining_bill_calculation :
  total_amount_spent 100 0.1 0.2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l467_46774


namespace NUMINAMATH_CALUDE_is_quadratic_equation_example_l467_46799

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation 2x^2 = 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- Theorem: The equation 2x^2 = 1 is a quadratic equation in one variable -/
theorem is_quadratic_equation_example : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_is_quadratic_equation_example_l467_46799


namespace NUMINAMATH_CALUDE_share_multiple_l467_46727

theorem share_multiple (total : ℕ) (c_share : ℕ) (k : ℕ) : 
  total = 880 → c_share = 160 → 
  (∃ (a_share b_share : ℕ), 
    a_share + b_share + c_share = total ∧ 
    4 * a_share = k * b_share ∧ 
    k * b_share = 10 * c_share) → 
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_share_multiple_l467_46727
