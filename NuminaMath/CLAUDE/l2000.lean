import Mathlib

namespace NUMINAMATH_CALUDE_intersection_exists_l2000_200064

-- Define a structure for a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for a set of 5 points
def FivePoints := Fin 5 → Point3D

-- Define a predicate for 4 points being non-coplanar
def nonCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a predicate for a line intersecting a triangle
def lineIntersectsTriangle (l1 l2 p1 p2 p3 : Point3D) : Prop := sorry

-- Main theorem
theorem intersection_exists (points : FivePoints) 
  (h : ∀ i j k l : Fin 5, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
       nonCoplanar (points i) (points j) (points k) (points l)) :
  ∃ i j k l m : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ k ≠ l ∧ k ≠ m ∧ l ≠ m ∧
    lineIntersectsTriangle (points i) (points j) (points k) (points l) (points m) :=
  sorry

end NUMINAMATH_CALUDE_intersection_exists_l2000_200064


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2000_200071

theorem min_value_squared_sum (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) : 
  ∃ (min : ℝ), min = 80 ∧ 
  ∀ (x : ℝ), x = (p*t)^2 + (q*u)^2 + (r*v)^2 + (s*w)^2 → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2000_200071


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2000_200032

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2000_200032


namespace NUMINAMATH_CALUDE_petya_cannot_win_l2000_200055

/-- Represents a chess tournament --/
structure ChessTournament where
  players : ℕ
  games_per_player : ℕ
  total_games : ℕ
  last_place_max_points : ℕ

/-- Creates a chess tournament with the given number of players --/
def create_tournament (n : ℕ) : ChessTournament :=
  { players := n
  , games_per_player := n - 1
  , total_games := n * (n - 1) / 2
  , last_place_max_points := (n * (n - 1) / 2) / n }

/-- Theorem: Petya cannot become the winner after disqualification --/
theorem petya_cannot_win (t : ChessTournament) 
  (h1 : t.players = 10) 
  (h2 : t = create_tournament 10) 
  (h3 : t.last_place_max_points ≤ 4) :
  ∃ (remaining_players : ℕ) (remaining_games : ℕ),
    remaining_players = t.players - 1 ∧
    remaining_games = remaining_players * (remaining_players - 1) / 2 ∧
    remaining_games / remaining_players ≥ t.last_place_max_points :=
by sorry

end NUMINAMATH_CALUDE_petya_cannot_win_l2000_200055


namespace NUMINAMATH_CALUDE_at_least_three_lines_intersect_l2000_200026

/-- A line that divides a square into two quadrilaterals -/
structure DividingLine where
  divides_square : Bool
  area_ratio : Rat
  intersects_point : Point

/-- A square with dividing lines -/
structure DividedSquare where
  side_length : ℝ
  dividing_lines : List DividingLine

/-- The theorem statement -/
theorem at_least_three_lines_intersect (square : DividedSquare) :
  square.side_length > 0 ∧
  square.dividing_lines.length = 9 ∧
  (∀ l ∈ square.dividing_lines, l.divides_square ∧ l.area_ratio = 2 / 3) →
  ∃ p : Point, (square.dividing_lines.filter (λ l => l.intersects_point = p)).length ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_at_least_three_lines_intersect_l2000_200026


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l2000_200072

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_sum : unitsDigit (42^2 + 25^3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l2000_200072


namespace NUMINAMATH_CALUDE_correct_notebooks_A_correct_min_full_price_sales_l2000_200042

/-- Represents the bookstore problem with notebooks of two types -/
structure BookstoreProblem where
  total_notebooks : ℕ
  cost_price_A : ℕ
  cost_price_B : ℕ
  total_cost : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ
  discount_A : ℚ
  profit_threshold : ℕ

/-- The specific instance of the bookstore problem -/
def problem : BookstoreProblem :=
  { total_notebooks := 350
  , cost_price_A := 12
  , cost_price_B := 15
  , total_cost := 4800
  , selling_price_A := 20
  , selling_price_B := 25
  , discount_A := 0.7
  , profit_threshold := 2348 }

/-- The number of type A notebooks purchased -/
def notebooks_A (p : BookstoreProblem) : ℕ := sorry

/-- The number of type B notebooks purchased -/
def notebooks_B (p : BookstoreProblem) : ℕ := sorry

/-- The minimum number of notebooks of each type that must be sold at full price -/
def min_full_price_sales (p : BookstoreProblem) : ℕ := sorry

/-- Theorem stating the correct number of type A notebooks -/
theorem correct_notebooks_A : notebooks_A problem = 150 := by sorry

/-- Theorem stating the correct minimum number of full-price sales -/
theorem correct_min_full_price_sales : min_full_price_sales problem = 128 := by sorry

end NUMINAMATH_CALUDE_correct_notebooks_A_correct_min_full_price_sales_l2000_200042


namespace NUMINAMATH_CALUDE_solve_equations_l2000_200076

theorem solve_equations :
  (∀ x : ℚ, (16 / 5) / x = (12 / 7) / (5 / 8) → x = 7 / 6) ∧
  (∀ x : ℚ, 2 * x + 3 * 0.9 = 24.7 → x = 11) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l2000_200076


namespace NUMINAMATH_CALUDE_special_subset_contains_all_rationals_l2000_200043

def is_special_subset (S : Set ℚ) : Prop :=
  (1/2 ∈ S) ∧ 
  (∀ x ∈ S, x/2 ∈ S) ∧ 
  (∀ x ∈ S, 1/(x+1) ∈ S)

theorem special_subset_contains_all_rationals (S : Set ℚ) 
  (h : is_special_subset S) :
  ∀ q ∈ Set.Ioo (0 : ℚ) 1, q ∈ S :=
by
  sorry

end NUMINAMATH_CALUDE_special_subset_contains_all_rationals_l2000_200043


namespace NUMINAMATH_CALUDE_joan_apples_total_l2000_200016

/-- The number of apples Joan has now, given her initial pick and Melanie's gift -/
def total_apples (initial_pick : ℕ) (melanie_gift : ℕ) : ℕ :=
  initial_pick + melanie_gift

/-- Theorem stating that Joan has 70 apples in total -/
theorem joan_apples_total :
  total_apples 43 27 = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_total_l2000_200016


namespace NUMINAMATH_CALUDE_worker_B_completion_time_l2000_200046

-- Define the time it takes for Worker A to complete the job
def worker_A_time : ℝ := 5

-- Define the time it takes for both workers together to complete the job
def combined_time : ℝ := 3.333333333333333

-- Define the time it takes for Worker B to complete the job
def worker_B_time : ℝ := 10

-- Theorem statement
theorem worker_B_completion_time :
  (1 / worker_A_time + 1 / worker_B_time = 1 / combined_time) →
  worker_B_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_worker_B_completion_time_l2000_200046


namespace NUMINAMATH_CALUDE_mike_lawn_mowing_earnings_l2000_200006

def mower_blade_cost : ℕ := 24
def game_cost : ℕ := 5
def num_games : ℕ := 9

theorem mike_lawn_mowing_earnings :
  ∃ (total_earnings : ℕ),
    total_earnings = mower_blade_cost + (game_cost * num_games) :=
by
  sorry

end NUMINAMATH_CALUDE_mike_lawn_mowing_earnings_l2000_200006


namespace NUMINAMATH_CALUDE_inequalities_solution_range_l2000_200089

theorem inequalities_solution_range (m : ℝ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, (3 * ↑x - m < 0 ∧ 7 - 2 * ↑x < 5) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄) →
  (15 < m ∧ m ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_solution_range_l2000_200089


namespace NUMINAMATH_CALUDE_power_sum_difference_l2000_200068

theorem power_sum_difference : 4^1 + 3^2 - 2^3 + 1^4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l2000_200068


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2000_200091

theorem interest_rate_calculation (total_investment : ℝ) (first_part : ℝ) (second_part_rate : ℝ) (total_interest : ℝ) : 
  total_investment = 3600 →
  first_part = 1800 →
  second_part_rate = 5 →
  total_interest = 144 →
  (first_part * (3 / 100)) + ((total_investment - first_part) * (second_part_rate / 100)) = total_interest :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2000_200091


namespace NUMINAMATH_CALUDE_even_square_operation_l2000_200052

theorem even_square_operation (x : ℕ) (h : x > 0) : ∃ k : ℕ, (2 * x)^2 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_even_square_operation_l2000_200052


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2000_200038

theorem quadratic_inequality (y : ℝ) : 
  y^2 + 3*y - 54 > 0 ↔ y < -9 ∨ y > 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2000_200038


namespace NUMINAMATH_CALUDE_volunteers_selection_ways_l2000_200092

/-- The number of ways to select volunteers for community service --/
def select_volunteers (n : ℕ) : ℕ :=
  let both_days := n  -- Select 1 person for both days
  let saturday := n - 1  -- Select 1 person for Saturday from remaining n-1
  let sunday := n - 2  -- Select 1 person for Sunday from remaining n-2
  both_days * saturday * sunday

/-- Theorem: The number of ways to select exactly one person to serve for both days
    out of 5 volunteers, with 2 people selected each day, is equal to 60 --/
theorem volunteers_selection_ways :
  select_volunteers 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_selection_ways_l2000_200092


namespace NUMINAMATH_CALUDE_custom_op_result_l2000_200078

/-- Define the custom operation ã — -/
def custom_op (a b : ℤ) : ℤ := 2*a - 3*b + a*b

/-- Theorem stating that (1 ã — 2) - 2 = -4 -/
theorem custom_op_result : custom_op 1 2 - 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l2000_200078


namespace NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l2000_200014

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_n_for_irreducible_fractions : 
  ∃ (n : ℕ), n = 28 ∧ 
  (∀ k : ℕ, 5 ≤ k → k ≤ 24 → is_coprime k (n + k + 1)) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, 5 ≤ k ∧ k ≤ 24 ∧ ¬is_coprime k (m + k + 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l2000_200014


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_three_l2000_200030

theorem no_solution_iff_m_eq_neg_three (m : ℝ) :
  (∀ x : ℝ, x ≠ -1 → (3 * x) / (x + 1) ≠ m / (x + 1) + 2) ↔ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_three_l2000_200030


namespace NUMINAMATH_CALUDE_point_on_graph_l2000_200085

/-- The function f(x) = -2x + 3 --/
def f (x : ℝ) : ℝ := -2 * x + 3

/-- The point (1, 1) --/
def point : ℝ × ℝ := (1, 1)

/-- Theorem: The point (1, 1) lies on the graph of f(x) = -2x + 3 --/
theorem point_on_graph : f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l2000_200085


namespace NUMINAMATH_CALUDE_vector_problem_l2000_200001

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem vector_problem (h1 : e₁ ≠ 0) (h2 : e₂ ≠ 0) (h3 : ¬ ∃ (r : ℝ), e₁ = r • e₂) 
  (h4 : B - A = 3 • e₁ + k • e₂) 
  (h5 : C - B = 4 • e₁ + e₂) 
  (h6 : D - C = 8 • e₁ - 9 • e₂)
  (h7 : ∃ (t : ℝ), D - A = t • (B - A)) :
  k = -2 := by sorry

end NUMINAMATH_CALUDE_vector_problem_l2000_200001


namespace NUMINAMATH_CALUDE_opposite_equal_implies_zero_abs_equal_implies_equal_or_opposite_sum_product_condition_implies_abs_equality_abs_plus_self_nonnegative_l2000_200087

-- Statement 1
theorem opposite_equal_implies_zero (x : ℝ) : x = -x → x = 0 := by sorry

-- Statement 2
theorem abs_equal_implies_equal_or_opposite (a b : ℝ) : |a| = |b| → a = b ∨ a = -b := by sorry

-- Statement 3
theorem sum_product_condition_implies_abs_equality (a b : ℝ) : 
  a + b < 0 → ab > 0 → |7*a + 3*b| = -(7*a + 3*b) := by sorry

-- Statement 4
theorem abs_plus_self_nonnegative (m : ℚ) : |m| + m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_opposite_equal_implies_zero_abs_equal_implies_equal_or_opposite_sum_product_condition_implies_abs_equality_abs_plus_self_nonnegative_l2000_200087


namespace NUMINAMATH_CALUDE_james_chores_total_time_l2000_200013

/-- Given James' chore schedule, prove that the total time spent is 16.5 hours -/
theorem james_chores_total_time :
  let vacuuming_time : ℝ := 3
  let cleaning_time : ℝ := 3 * vacuuming_time
  let laundry_time : ℝ := cleaning_time / 2
  vacuuming_time + cleaning_time + laundry_time = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_james_chores_total_time_l2000_200013


namespace NUMINAMATH_CALUDE_eggs_broken_count_l2000_200028

-- Define the number of brown eggs
def brown_eggs : ℕ := 10

-- Define the number of white eggs
def white_eggs : ℕ := 3 * brown_eggs

-- Define the total number of eggs before dropping
def total_eggs_before : ℕ := brown_eggs + white_eggs

-- Define the number of eggs left after dropping
def eggs_left_after : ℕ := 20

-- Theorem to prove
theorem eggs_broken_count : total_eggs_before - eggs_left_after = 20 := by
  sorry

end NUMINAMATH_CALUDE_eggs_broken_count_l2000_200028


namespace NUMINAMATH_CALUDE_evaluate_expression_l2000_200060

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) :
  y * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2000_200060


namespace NUMINAMATH_CALUDE_function_machine_output_l2000_200054

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 20 then
    step1 - 8
  else
    step1 + 10

theorem function_machine_output :
  function_machine 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l2000_200054


namespace NUMINAMATH_CALUDE_unique_solution_l2000_200045

theorem unique_solution : ∃! x : ℝ, 70 + 5 * 12 / (180 / x) = 71 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2000_200045


namespace NUMINAMATH_CALUDE_midpoint_complex_numbers_l2000_200003

theorem midpoint_complex_numbers : 
  let a : ℂ := (1 : ℂ) / (1 + Complex.I)
  let b : ℂ := (1 : ℂ) / (1 - Complex.I)
  let c : ℂ := (a + b) / 2
  c = (1 : ℂ) / 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_complex_numbers_l2000_200003


namespace NUMINAMATH_CALUDE_carter_baseball_cards_l2000_200075

/-- Given that Marcus has 210 baseball cards and 58 more than Carter,
    prove that Carter has 152 baseball cards. -/
theorem carter_baseball_cards :
  let marcus_cards : ℕ := 210
  let difference : ℕ := 58
  let carter_cards : ℕ := marcus_cards - difference
  carter_cards = 152 :=
by sorry

end NUMINAMATH_CALUDE_carter_baseball_cards_l2000_200075


namespace NUMINAMATH_CALUDE_f_max_value_l2000_200062

/-- The function f(x) = 5x - x^2 -/
def f (x : ℝ) : ℝ := 5 * x - x^2

/-- The maximum value of f(x) is 6.25 -/
theorem f_max_value : ∃ (c : ℝ), ∀ (x : ℝ), f x ≤ c ∧ ∃ (x₀ : ℝ), f x₀ = c :=
  sorry

end NUMINAMATH_CALUDE_f_max_value_l2000_200062


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2000_200066

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 6*x - 16 = 0) ∧
  (∃ x : ℝ, 2*x^2 - 3*x - 5 = 0) →
  (∃ x : ℝ, x = 8 ∨ x = -2) ∧
  (∃ x : ℝ, x = 5/2 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2000_200066


namespace NUMINAMATH_CALUDE_eulers_formula_3d_l2000_200063

/-- A space convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- Euler's formula for space convex polyhedra -/
theorem eulers_formula_3d (p : ConvexPolyhedron) : p.faces + p.vertices - p.edges = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_3d_l2000_200063


namespace NUMINAMATH_CALUDE_danielles_rooms_l2000_200041

theorem danielles_rooms (d h g : ℕ) : 
  h = 3 * d →  -- Heidi's apartment has 3 times as many rooms as Danielle's
  g = h / 9 →  -- Grant's apartment has 1/9 as many rooms as Heidi's
  g = 2 →      -- Grant's apartment has 2 rooms
  d = 6        -- Prove that Danielle's apartment has 6 rooms
:= by sorry

end NUMINAMATH_CALUDE_danielles_rooms_l2000_200041


namespace NUMINAMATH_CALUDE_tangent_lines_through_A_area_of_triangle_AOC_l2000_200057

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + 12 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 5)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the center of the circle C
def center : ℝ × ℝ := (2, 3)

-- Theorem for the tangent lines
theorem tangent_lines_through_A :
  ∃ (k : ℝ), 
    (∀ x y : ℝ, x = 3 → circle_equation x y) ∧
    (∀ x y : ℝ, y = k*x + (11/4) → circle_equation x y) ∧
    k = 3/4 :=
sorry

-- Theorem for the area of triangle AOC
theorem area_of_triangle_AOC :
  let A := point_A
  let O := origin
  let C := center
  (1/2 : ℝ) * ‖C - O‖ * ‖A - O‖ * (|C.1 * A.2 - C.2 * A.1| / (‖C - O‖ * ‖A - O‖)) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_through_A_area_of_triangle_AOC_l2000_200057


namespace NUMINAMATH_CALUDE_square_of_negative_product_l2000_200002

theorem square_of_negative_product (a b : ℝ) : (-a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l2000_200002


namespace NUMINAMATH_CALUDE_pickled_vegetables_grade_C_l2000_200058

/-- Represents the number of boxes of pickled vegetables in each grade -/
structure GradeBoxes where
  A : ℕ
  B : ℕ
  C : ℕ

/-- 
Given:
- There are 420 boxes of pickled vegetables in total
- The vegetables are classified into three grades: A, B, and C
- m, n, and t are the number of boxes sampled from grades A, B, and C, respectively
- 2t = m + n

Prove that the number of boxes classified as grade C is 140
-/
theorem pickled_vegetables_grade_C (boxes : GradeBoxes) 
  (total_boxes : boxes.A + boxes.B + boxes.C = 420)
  (sample_relation : ∃ (m n t : ℕ), 2 * t = m + n) :
  boxes.C = 140 := by
  sorry

end NUMINAMATH_CALUDE_pickled_vegetables_grade_C_l2000_200058


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_16_is_plus_minus_2_l2000_200097

theorem sqrt_of_sqrt_16_is_plus_minus_2 : 
  {x : ℝ | x^2 = Real.sqrt 16} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_16_is_plus_minus_2_l2000_200097


namespace NUMINAMATH_CALUDE_diamond_three_eight_l2000_200053

-- Define the ◇ operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y + 2

-- Theorem statement
theorem diamond_three_eight : diamond 3 8 = 62 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_eight_l2000_200053


namespace NUMINAMATH_CALUDE_attic_items_count_l2000_200029

theorem attic_items_count (total : ℝ) (useful_percent : ℝ) (heirloom_percent : ℝ) (junk_percent : ℝ) (junk_count : ℝ) :
  useful_percent = 0.20 →
  heirloom_percent = 0.10 →
  junk_percent = 0.70 →
  junk_count = 28 →
  junk_percent * total = junk_count →
  useful_percent * total = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_attic_items_count_l2000_200029


namespace NUMINAMATH_CALUDE_recreation_spending_ratio_l2000_200056

/-- Proves that if wages decrease by 25% and recreation spending decreases from 30% to 20%,
    then the new recreation spending is 50% of the original. -/
theorem recreation_spending_ratio (original_wages : ℝ) (original_wages_positive : original_wages > 0) :
  let new_wages := 0.75 * original_wages
  let original_recreation := 0.3 * original_wages
  let new_recreation := 0.2 * new_wages
  new_recreation / original_recreation = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_recreation_spending_ratio_l2000_200056


namespace NUMINAMATH_CALUDE_total_rabbits_count_l2000_200048

/-- The number of white rabbits in the school -/
def white_rabbits : ℕ := 15

/-- The number of black rabbits in the school -/
def black_rabbits : ℕ := 37

/-- The total number of rabbits in the school -/
def total_rabbits : ℕ := white_rabbits + black_rabbits

theorem total_rabbits_count : total_rabbits = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_rabbits_count_l2000_200048


namespace NUMINAMATH_CALUDE_system_solution_l2000_200070

theorem system_solution (x y u v : ℝ) : 
  (x^2 + y^2 + u^2 + v^2 = 4) ∧
  (x*u + y*v + x*v + y*u = 0) ∧
  (x*y*u + y*u*v + u*v*x + v*x*y = -2) ∧
  (x*y*u*v = -1) →
  ((x = 1 ∧ y = 1 ∧ u = 1 ∧ v = -1) ∨
   (x = -1 + Real.sqrt 2 ∧ y = -1 - Real.sqrt 2 ∧ u = 1 ∧ v = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2000_200070


namespace NUMINAMATH_CALUDE_xiao_ming_opening_probability_l2000_200019

/-- Represents a three-digit lock with digits from 0 to 9 -/
structure Lock :=
  (digit1 : Fin 10)
  (digit2 : Fin 10)
  (digit3 : Fin 10)

/-- Represents Xiao Ming's knowledge of the lock -/
structure XiaoMingKnowledge :=
  (knownDigit1 : Fin 10)
  (knownDigit3 : Fin 10)

/-- The probability of Xiao Ming opening the lock on the first try -/
def openingProbability (lock : Lock) (knowledge : XiaoMingKnowledge) : ℚ :=
  if lock.digit1 = knowledge.knownDigit1 && lock.digit3 = knowledge.knownDigit3
  then 1 / 10
  else 0

/-- Theorem stating that the probability of Xiao Ming opening the lock on the first try is 1/10 -/
theorem xiao_ming_opening_probability 
  (lock : Lock) (knowledge : XiaoMingKnowledge) 
  (h1 : lock.digit1 = knowledge.knownDigit1) 
  (h3 : lock.digit3 = knowledge.knownDigit3) : 
  openingProbability lock knowledge = 1 / 10 := by
  sorry


end NUMINAMATH_CALUDE_xiao_ming_opening_probability_l2000_200019


namespace NUMINAMATH_CALUDE_cannot_be_even_after_odd_operations_l2000_200080

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- Function to determine the parity of a number -/
def getParity (n : ℕ) : Parity :=
  if n % 2 = 0 then Parity.Even else Parity.Odd

/-- Function to toggle the parity -/
def toggleParity (p : Parity) : Parity :=
  match p with
  | Parity.Even => Parity.Odd
  | Parity.Odd => Parity.Even

theorem cannot_be_even_after_odd_operations
  (initial : ℕ)
  (operations : ℕ)
  (h_initial_even : getParity initial = Parity.Even)
  (h_operations_odd : getParity operations = Parity.Odd) :
  ∃ (final : ℕ), getParity final = Parity.Odd ∧
    ∃ (f : ℕ → ℕ), (∀ n, f n = n + 1 ∨ f n = n - 1) ∧
      (f^[operations] initial = final) :=
sorry

end NUMINAMATH_CALUDE_cannot_be_even_after_odd_operations_l2000_200080


namespace NUMINAMATH_CALUDE_retirement_fund_increment_l2000_200074

theorem retirement_fund_increment (y k : ℝ) 
  (h1 : k * Real.sqrt (y + 3) = k * Real.sqrt y + 15)
  (h2 : k * Real.sqrt (y + 5) = k * Real.sqrt y + 27)
  : k * Real.sqrt y = 810 := by
  sorry

end NUMINAMATH_CALUDE_retirement_fund_increment_l2000_200074


namespace NUMINAMATH_CALUDE_sum_in_base10_l2000_200079

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 13 -/
def C : ℕ := 12

theorem sum_in_base10 : 
  base14ToBase10 356 + base13ToBase10 409 = 1505 := by sorry

end NUMINAMATH_CALUDE_sum_in_base10_l2000_200079


namespace NUMINAMATH_CALUDE_youngest_son_park_visits_l2000_200059

theorem youngest_son_park_visits (season_pass_cost : ℝ) (oldest_son_visits : ℕ) (youngest_son_cost_per_trip : ℝ) :
  season_pass_cost = 100 →
  oldest_son_visits = 35 →
  youngest_son_cost_per_trip = 4 →
  ∃ (youngest_son_visits : ℕ), 
    (season_pass_cost / youngest_son_visits) = youngest_son_cost_per_trip ∧
    youngest_son_visits = 25 :=
by sorry

end NUMINAMATH_CALUDE_youngest_son_park_visits_l2000_200059


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2000_200051

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1}
def B : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 ∧ 0 ≤ p.1 ∧ p.1 ≤ 10}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2000_200051


namespace NUMINAMATH_CALUDE_some_pens_not_vens_l2000_200008

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Pen Den Ven : U → Prop)

-- Define the hypotheses
variable (h1 : ∀ x, Pen x → Den x)
variable (h2 : ∃ x, Den x ∧ ¬Ven x)

-- State the theorem
theorem some_pens_not_vens : ∃ x, Pen x ∧ ¬Ven x :=
sorry

end NUMINAMATH_CALUDE_some_pens_not_vens_l2000_200008


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l2000_200050

theorem toy_store_revenue_ratio :
  ∀ (december_revenue : ℚ),
  december_revenue > 0 →
  let november_revenue := (3 : ℚ) / 5 * december_revenue
  let january_revenue := (1 : ℚ) / 6 * november_revenue
  let average_revenue := (november_revenue + january_revenue) / 2
  december_revenue / average_revenue = 20 / 7 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l2000_200050


namespace NUMINAMATH_CALUDE_increasing_function_range_l2000_200069

/-- Given a ∈ (0,1) and f(x) = a^x + (1+a)^x is increasing on (0,+∞), 
    prove that a ∈ [((5^(1/2)) - 1)/2, 1) -/
theorem increasing_function_range (a : ℝ) 
  (h1 : 0 < a ∧ a < 1) 
  (h2 : ∀ x > 0, Monotone (fun x => a^x + (1+a)^x)) : 
  a ∈ Set.Icc ((Real.sqrt 5 - 1) / 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_range_l2000_200069


namespace NUMINAMATH_CALUDE_initial_members_family_b_l2000_200015

/-- Represents the number of members in each family in Indira Nagar -/
structure FamilyMembers where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The theorem stating the initial number of members in family b -/
theorem initial_members_family_b (fm : FamilyMembers) : 
  fm.a = 7 ∧ fm.c = 10 ∧ fm.d = 13 ∧ fm.e = 6 ∧ fm.f = 10 ∧
  (fm.a + fm.b + fm.c + fm.d + fm.e + fm.f - 6) / 6 = 8 →
  fm.b = 8 := by
  sorry

#check initial_members_family_b

end NUMINAMATH_CALUDE_initial_members_family_b_l2000_200015


namespace NUMINAMATH_CALUDE_first_month_sale_is_5266_l2000_200007

/-- Calculates the sale in the first month given the sales data for 6 months -/
def first_month_sale (average : ℕ) (month2 : ℕ) (month3 : ℕ) (month4 : ℕ) (month5 : ℕ) (month6 : ℕ) : ℕ :=
  6 * average - (month2 + month3 + month4 + month5 + month6)

/-- Theorem stating that the sale in the first month is 5266 given the problem conditions -/
theorem first_month_sale_is_5266 :
  first_month_sale 5600 5768 5922 5678 6029 4937 = 5266 := by
  sorry

#eval first_month_sale 5600 5768 5922 5678 6029 4937

end NUMINAMATH_CALUDE_first_month_sale_is_5266_l2000_200007


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2000_200047

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2000_200047


namespace NUMINAMATH_CALUDE_count_integer_pairs_l2000_200096

theorem count_integer_pairs : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + 2*p.2 < 40) (Finset.product (Finset.range 40) (Finset.range 40))).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l2000_200096


namespace NUMINAMATH_CALUDE_power_difference_evaluation_l2000_200009

theorem power_difference_evaluation : (3^4)^3 - (4^3)^4 = -16246775 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_evaluation_l2000_200009


namespace NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l2000_200065

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := b^2 - a*b

-- Theorem statement
theorem equation_has_two_distinct_real_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ otimes (m - 2) x₁ = m ∧ otimes (m - 2) x₂ = m :=
by sorry

end NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l2000_200065


namespace NUMINAMATH_CALUDE_garden_area_l2000_200025

/-- The area of a garden with square cutouts -/
theorem garden_area (garden_length : ℝ) (garden_width : ℝ) 
  (cutout1_side : ℝ) (cutout2_side : ℝ) : 
  garden_length = 20 ∧ garden_width = 18 ∧ 
  cutout1_side = 4 ∧ cutout2_side = 5 →
  garden_length * garden_width - cutout1_side^2 - cutout2_side^2 = 319 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l2000_200025


namespace NUMINAMATH_CALUDE_circumscribing_circle_diameter_l2000_200084

theorem circumscribing_circle_diameter (n : ℕ) (r : ℝ) :
  n = 8 ∧ r = 2 →
  let R := (2 * r) / (2 * Real.sin (π / n))
  2 * (R + r) = 2 * (4 / Real.sqrt (2 - Real.sqrt 2) + 2) := by sorry

end NUMINAMATH_CALUDE_circumscribing_circle_diameter_l2000_200084


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l2000_200020

def is_geometric_progression (a b c : ℕ) : Prop := b * b = a * c

theorem three_digit_number_problem (a b c : ℕ) 
  (h1 : is_geometric_progression a b c)
  (h2 : 100 * c + 10 * b + a = 100 * a + 10 * b + c - 594)
  (h3 : 10 * c + b - (10 * b + c) = 18)
  (h4 : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h5 : a ≠ 0) :
  100 * a + 10 * b + c = 842 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l2000_200020


namespace NUMINAMATH_CALUDE_sophie_uses_one_sheet_per_load_l2000_200061

-- Define the given conditions
def loads_per_week : ℕ := 4
def box_cost : ℚ := 5.5
def sheets_per_box : ℕ := 104
def yearly_savings : ℚ := 11

-- Define the function to calculate the number of dryer sheets per load
def dryer_sheets_per_load : ℚ :=
  (yearly_savings / box_cost) * sheets_per_box / (loads_per_week * 52)

-- Theorem statement
theorem sophie_uses_one_sheet_per_load : 
  dryer_sheets_per_load = 1 := by sorry

end NUMINAMATH_CALUDE_sophie_uses_one_sheet_per_load_l2000_200061


namespace NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l2000_200033

theorem sandwich_non_condiment_percentage
  (total_weight : ℝ)
  (condiment_weight : ℝ)
  (h1 : total_weight = 150)
  (h2 : condiment_weight = 45) :
  (total_weight - condiment_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l2000_200033


namespace NUMINAMATH_CALUDE_day_crew_load_fraction_l2000_200082

/-- Represents the fraction of boxes loaded by the day crew given the night crew's relative productivity and size -/
theorem day_crew_load_fraction (D W : ℚ) : 
  D > 0 → W > 0 → 
  (D * W) / ((D * W) + ((3/4 * D) * (3/4 * W))) = 16/25 := by
  sorry

end NUMINAMATH_CALUDE_day_crew_load_fraction_l2000_200082


namespace NUMINAMATH_CALUDE_unique_orthogonal_chord_l2000_200086

-- Define the quadratic function
def f (p q x : ℝ) : ℝ := x^2 - 2*p*x + q

-- State the theorem
theorem unique_orthogonal_chord (p q : ℝ) :
  p > 0 ∧ q > 0 ∧  -- p and q are positive
  (∀ x, f p q x ≠ 0) ∧  -- graph doesn't intersect x-axis
  (∃! a, a > 0 ∧
    f p q (p - a) = f p q (p + a) ∧  -- AB parallel to x-axis
    (p - a) * (p + a) + (f p q (p - a))^2 = 0)  -- angle AOB = π/2
  → q = 1/4 := by
sorry

end NUMINAMATH_CALUDE_unique_orthogonal_chord_l2000_200086


namespace NUMINAMATH_CALUDE_brothers_additional_lambs_l2000_200067

/-- The number of lambs Merry takes care of -/
def merrys_lambs : ℕ := 10

/-- The total number of lambs -/
def total_lambs : ℕ := 23

/-- The number of lambs Merry's brother takes care of -/
def brothers_lambs : ℕ := total_lambs - merrys_lambs

/-- The additional number of lambs Merry's brother takes care of compared to Merry -/
def additional_lambs : ℕ := brothers_lambs - merrys_lambs

theorem brothers_additional_lambs :
  additional_lambs = 3 ∧ brothers_lambs > merrys_lambs := by
  sorry

end NUMINAMATH_CALUDE_brothers_additional_lambs_l2000_200067


namespace NUMINAMATH_CALUDE_max_vertex_sum_l2000_200099

-- Define a cube with numbered faces
def Cube := Fin 6 → ℕ

-- Define a function to get the sum of three faces at a vertex
def vertexSum (c : Cube) (v : Fin 8) : ℕ :=
  sorry -- Implementation details omitted as per instructions

-- Theorem statement
theorem max_vertex_sum (c : Cube) : 
  (∀ v : Fin 8, vertexSum c v ≤ 14) ∧ (∃ v : Fin 8, vertexSum c v = 14) :=
sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l2000_200099


namespace NUMINAMATH_CALUDE_path_length_is_twelve_l2000_200040

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  side_a : ℝ
  side_b : ℝ
  hypotenuse : ℝ
  is_right : side_a^2 + side_b^2 = hypotenuse^2
  side_values : side_a = 9 ∧ side_b = 12 ∧ hypotenuse = 15

/-- A circle rolling inside the triangle -/
structure RollingCircle where
  radius : ℝ
  radius_value : radius = 2

/-- The path traced by the center of the rolling circle -/
def path_length (t : RightTriangle) (c : RollingCircle) : ℝ := 
  t.side_a + t.side_b + t.hypotenuse - 2 * (t.side_a + t.side_b + t.hypotenuse - 6 * c.radius)

/-- Theorem stating that the path length is 12 -/
theorem path_length_is_twelve (t : RightTriangle) (c : RollingCircle) : 
  path_length t c = 12 := by sorry

end NUMINAMATH_CALUDE_path_length_is_twelve_l2000_200040


namespace NUMINAMATH_CALUDE_number_difference_problem_l2000_200000

theorem number_difference_problem : ∃ x : ℚ, x - (3/5) * x = 60 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_problem_l2000_200000


namespace NUMINAMATH_CALUDE_large_bucket_relation_tank_capacity_is_21_l2000_200011

/-- The capacity of a small bucket in liters -/
def small_bucket_capacity : ℝ := 0.5

/-- The capacity of a large bucket in liters -/
def large_bucket_capacity : ℝ := 4

/-- The number of small buckets used to fill the tank -/
def num_small_buckets : ℕ := 2

/-- The number of large buckets used to fill the tank -/
def num_large_buckets : ℕ := 5

/-- The relationship between small and large bucket capacities -/
theorem large_bucket_relation : large_bucket_capacity = 2 * small_bucket_capacity + 3 := by sorry

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := num_small_buckets * small_bucket_capacity + num_large_buckets * large_bucket_capacity

theorem tank_capacity_is_21 : tank_capacity = 21 := by sorry

end NUMINAMATH_CALUDE_large_bucket_relation_tank_capacity_is_21_l2000_200011


namespace NUMINAMATH_CALUDE_apple_arrangements_l2000_200035

/-- The number of distinct arrangements of letters in a word with repeated letters -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "APPLE" has 5 letters with 'P' repeating twice -/
def appleWord : (ℕ × List ℕ) := (5, [2])

theorem apple_arrangements :
  distinctArrangements appleWord.1 appleWord.2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_apple_arrangements_l2000_200035


namespace NUMINAMATH_CALUDE_min_steps_ladder_l2000_200036

theorem min_steps_ladder (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let n := a + b - Nat.gcd a b
  ∀ m : ℕ, (∃ (x y : ℕ), x * a - y * b = m ∧ x * a - y * b = 0) → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_min_steps_ladder_l2000_200036


namespace NUMINAMATH_CALUDE_painting_payment_l2000_200034

theorem painting_payment (rate : ℚ) (rooms : ℚ) (h1 : rate = 13 / 3) (h2 : rooms = 8 / 5) :
  rate * rooms = 104 / 15 := by
sorry

end NUMINAMATH_CALUDE_painting_payment_l2000_200034


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l2000_200018

def base_5_digit (n : ℕ) : Prop := n < 5

theorem smallest_sum_B_plus_c :
  ∃ (B c : ℕ),
    base_5_digit B ∧
    c > 6 ∧
    (31 * B = 4 * c + 4) ∧
    ∀ (B' c' : ℕ),
      base_5_digit B' →
      c' > 6 →
      (31 * B' = 4 * c' + 4) →
      B + c ≤ B' + c' ∧
    B + c = 34 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l2000_200018


namespace NUMINAMATH_CALUDE_gcf_120_180_300_l2000_200095

theorem gcf_120_180_300 : Nat.gcd 120 (Nat.gcd 180 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_120_180_300_l2000_200095


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l2000_200077

theorem arithmetic_expression_equals_24 :
  (8 * 9 / 6) + 8 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l2000_200077


namespace NUMINAMATH_CALUDE_r_daily_earnings_l2000_200093

/-- Represents the daily earnings of individuals p, q, and r -/
structure DailyEarnings where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The conditions given in the problem -/
def problem_conditions (e : DailyEarnings) : Prop :=
  9 * (e.p + e.q + e.r) = 1710 ∧
  5 * (e.p + e.r) = 600 ∧
  7 * (e.q + e.r) = 910

/-- Theorem stating that given the problem conditions, r's daily earnings are 60 -/
theorem r_daily_earnings (e : DailyEarnings) :
  problem_conditions e → e.r = 60 := by
  sorry


end NUMINAMATH_CALUDE_r_daily_earnings_l2000_200093


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2000_200037

-- Define the quadratic function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the maximum value function h(a)
noncomputable def h (a b c : ℝ) : ℝ := 
  let x₀ := -b / (2 * a)
  f a b c x₀

-- Main theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a < 0)
  (hf : ∀ x, 1 < x ∧ x < 3 → f a b c x > -2 * x)
  (hz : ∃! x, f a b c x + 6 * a = 0)
  : 
  (a = -1/5 ∧ b = -6/5 ∧ c = -3/5) ∧
  (∀ a' b' c', h a' b' c' ≥ -2) ∧
  (∃ a₀ b₀ c₀, h a₀ b₀ c₀ = -2)
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2000_200037


namespace NUMINAMATH_CALUDE_water_moles_equal_cao_moles_l2000_200010

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product : String

-- Define the molar quantities
structure MolarQuantities where
  cao_moles : ℝ
  h2o_moles : ℝ
  caoh2_moles : ℝ

-- Define the problem parameters
def cao_mass : ℝ := 168
def cao_molar_mass : ℝ := 56.08
def target_caoh2_moles : ℝ := 3

-- Define the reaction
def calcium_hydroxide_reaction : Reaction :=
  { reactant1 := "CaO", reactant2 := "H2O", product := "Ca(OH)2" }

-- Theorem statement
theorem water_moles_equal_cao_moles 
  (reaction : Reaction) 
  (quantities : MolarQuantities) :
  reaction = calcium_hydroxide_reaction →
  quantities.caoh2_moles = target_caoh2_moles →
  quantities.cao_moles = cao_mass / cao_molar_mass →
  quantities.h2o_moles = quantities.cao_moles :=
by sorry

end NUMINAMATH_CALUDE_water_moles_equal_cao_moles_l2000_200010


namespace NUMINAMATH_CALUDE_smallest_inexpressible_is_eleven_l2000_200039

def expressible (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_inexpressible_is_eleven :
  (∀ m < 11, expressible m) ∧ ¬expressible 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_inexpressible_is_eleven_l2000_200039


namespace NUMINAMATH_CALUDE_product_of_distinct_non_trivial_primes_last_digit_l2000_200023

def is_non_trivial_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n > 10

def last_digit (n : ℕ) : ℕ :=
  n % 10

theorem product_of_distinct_non_trivial_primes_last_digit 
  (p q : ℕ) (hp : is_non_trivial_prime p) (hq : is_non_trivial_prime q) (hpq : p ≠ q) :
  ∃ d : ℕ, d ∈ [1, 3, 7, 9] ∧ last_digit (p * q) = d :=
sorry

end NUMINAMATH_CALUDE_product_of_distinct_non_trivial_primes_last_digit_l2000_200023


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_neg_one_two_l2000_200012

/-- Given an angle α whose terminal side passes through the point (-1, 2), 
    prove that cos α = -√5 / 5 -/
theorem cos_alpha_for_point_neg_one_two (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) → 
  Real.cos α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_neg_one_two_l2000_200012


namespace NUMINAMATH_CALUDE_river_width_l2000_200004

/-- Given a river with specified depth, flow rate, and volume per minute, prove its width. -/
theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) :
  depth = 2 →
  flow_rate = 3 →
  volume_per_minute = 4500 →
  (flow_rate * 1000 / 60) * depth * (volume_per_minute / (flow_rate * 1000 / 60) / depth) = 45 := by
  sorry

end NUMINAMATH_CALUDE_river_width_l2000_200004


namespace NUMINAMATH_CALUDE_softball_players_count_l2000_200024

theorem softball_players_count (cricket hockey football total : ℕ) 
  (h1 : cricket = 15)
  (h2 : hockey = 12)
  (h3 : football = 13)
  (h4 : total = 55) :
  total - (cricket + hockey + football) = 15 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_count_l2000_200024


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l2000_200098

theorem least_integer_absolute_value (x : ℤ) :
  (∀ y : ℤ, y < x → ∃ z : ℤ, z ≥ y ∧ z < x ∧ |3 * z^2 - 2 * z + 5| > 29) →
  |3 * x^2 - 2 * x + 5| ≤ 29 →
  x = -2 :=
sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l2000_200098


namespace NUMINAMATH_CALUDE_sum_of_digits_power_product_l2000_200027

def power_product (a b c : ℕ) : ℕ := a^2010 * b^2012 * c

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_power_product : sum_of_digits (power_product 2 5 7) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_product_l2000_200027


namespace NUMINAMATH_CALUDE_geometric_sequence_150th_term_l2000_200044

/-- Given a geometric sequence with first term 8 and second term -4, 
    the 150th term is equal to -8 * (1/2)^149 -/
theorem geometric_sequence_150th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n * (-1/2)) → 
    a 1 = 8 → 
    a 2 = -4 → 
    a 150 = -8 * (1/2)^149 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_150th_term_l2000_200044


namespace NUMINAMATH_CALUDE_adams_average_score_l2000_200088

/-- Given Adam's total score and number of rounds played, calculate the average points per round --/
theorem adams_average_score (total_score : ℕ) (num_rounds : ℕ) 
  (h1 : total_score = 283) (h2 : num_rounds = 4) :
  ∃ (avg : ℚ), avg = (total_score : ℚ) / (num_rounds : ℚ) ∧ 
  ∃ (rounded : ℕ), rounded = round avg ∧ rounded = 71 := by
  sorry

end NUMINAMATH_CALUDE_adams_average_score_l2000_200088


namespace NUMINAMATH_CALUDE_camel_zebra_ratio_l2000_200081

theorem camel_zebra_ratio : 
  ∀ (zebras camels monkeys giraffes : ℕ),
  zebras = 12 →
  monkeys = 4 * camels →
  giraffes = 2 →
  monkeys = giraffes + 22 →
  (camels : ℚ) / zebras = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_camel_zebra_ratio_l2000_200081


namespace NUMINAMATH_CALUDE_nature_of_c_l2000_200090

theorem nature_of_c (a c : ℝ) (h : (2*a - 1) / (-3) < -(c + 1) / (-4)) :
  (c < 0 ∨ c > 0) ∧ c ≠ -1 :=
sorry

end NUMINAMATH_CALUDE_nature_of_c_l2000_200090


namespace NUMINAMATH_CALUDE_betty_order_hair_color_cost_l2000_200094

/-- Given Betty's order details, prove the cost of each hair color. -/
theorem betty_order_hair_color_cost 
  (total_items : ℕ) 
  (slipper_count : ℕ) 
  (slipper_cost : ℚ) 
  (lipstick_count : ℕ) 
  (lipstick_cost : ℚ) 
  (hair_color_count : ℕ) 
  (total_paid : ℚ) 
  (h_total_items : total_items = 18) 
  (h_slipper_count : slipper_count = 6) 
  (h_slipper_cost : slipper_cost = 5/2) 
  (h_lipstick_count : lipstick_count = 4) 
  (h_lipstick_cost : lipstick_cost = 5/4) 
  (h_hair_color_count : hair_color_count = 8) 
  (h_total_paid : total_paid = 44) 
  (h_item_sum : total_items = slipper_count + lipstick_count + hair_color_count) : 
  (total_paid - (slipper_count * slipper_cost + lipstick_count * lipstick_cost)) / hair_color_count = 3 := by
  sorry


end NUMINAMATH_CALUDE_betty_order_hair_color_cost_l2000_200094


namespace NUMINAMATH_CALUDE_fgh_supermarket_difference_l2000_200083

/-- Prove that the difference between FGH supermarkets in the US and Canada is 14 -/
theorem fgh_supermarket_difference : ∀ (total us : ℕ),
  total = 84 →
  us = 49 →
  us > total - us →
  us - (total - us) = 14 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_difference_l2000_200083


namespace NUMINAMATH_CALUDE_tony_water_consumption_l2000_200005

/-- Calculates the daily water consumption given the bottle capacity, number of refills per week, and days in a week. -/
def daily_water_consumption (bottle_capacity : ℕ) (refills_per_week : ℕ) (days_in_week : ℕ) : ℚ :=
  (bottle_capacity * refills_per_week : ℚ) / days_in_week

/-- Proves that given a water bottle capacity of 84 ounces, filled 6 times per week, 
    and 7 days in a week, the daily water consumption is 72 ounces. -/
theorem tony_water_consumption :
  daily_water_consumption 84 6 7 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tony_water_consumption_l2000_200005


namespace NUMINAMATH_CALUDE_cubic_root_form_l2000_200073

theorem cubic_root_form (x : ℝ) : 
  27 * x^3 - 8 * x^2 - 8 * x - 1 = 0 →
  ∃ (a b c : ℕ+), 
    x = (Real.rpow a (1/3 : ℝ) + Real.rpow b (1/3 : ℝ) + 1) / c ∧
    a = 729 ∧ b = 27 ∧ c = 27 := by
  sorry

#check cubic_root_form

end NUMINAMATH_CALUDE_cubic_root_form_l2000_200073


namespace NUMINAMATH_CALUDE_karens_bonus_l2000_200049

/-- The bonus calculation problem for Karen's students' test scores. -/
theorem karens_bonus (total_students : ℕ) (max_score : ℕ) (bonus_threshold : ℕ) 
  (extra_bonus_per_point : ℕ) (graded_tests : ℕ) (graded_average : ℕ) 
  (last_two_tests_score : ℕ) (total_bonus : ℕ) :
  total_students = 10 →
  max_score = 150 →
  bonus_threshold = 75 →
  extra_bonus_per_point = 10 →
  graded_tests = 8 →
  graded_average = 70 →
  last_two_tests_score = 290 →
  total_bonus = 600 →
  (graded_tests * graded_average + last_two_tests_score) / total_students > bonus_threshold →
  total_bonus - (((graded_tests * graded_average + last_two_tests_score) / total_students - bonus_threshold) * extra_bonus_per_point) = 500 := by
  sorry

end NUMINAMATH_CALUDE_karens_bonus_l2000_200049


namespace NUMINAMATH_CALUDE_train_speed_problem_l2000_200021

/-- Proves that given the conditions of the train problem, the average speed of Train B is 43 miles per hour. -/
theorem train_speed_problem (initial_gap : ℝ) (train_a_speed : ℝ) (overtake_time : ℝ) (final_gap : ℝ) :
  initial_gap = 13 →
  train_a_speed = 37 →
  overtake_time = 5 →
  final_gap = 17 →
  (initial_gap + train_a_speed * overtake_time + final_gap) / overtake_time = 43 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2000_200021


namespace NUMINAMATH_CALUDE_function_values_l2000_200017

def is_prime (n : ℕ) : Prop := sorry

def coprime (a b : ℕ) : Prop := sorry

def number_theory_function (f : ℕ → ℕ) : Prop :=
  (∀ a b, coprime a b → f (a * b) = f a * f b) ∧
  (∀ p q, is_prime p → is_prime q → f (p + q) = f p + f q)

theorem function_values (f : ℕ → ℕ) (h : number_theory_function f) :
  f 2 = 2 ∧ f 3 = 3 ∧ f 1999 = 1999 := by sorry

end NUMINAMATH_CALUDE_function_values_l2000_200017


namespace NUMINAMATH_CALUDE_point_coordinates_l2000_200031

/-- A point in the second quadrant with specific x and y values -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  second_quadrant : x < 0 ∧ y > 0
  x_abs : |x| = 2
  y_squared : y^2 = 1

/-- The coordinates of the point P are (-2, 1) -/
theorem point_coordinates (P : SecondQuadrantPoint) : P.x = -2 ∧ P.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2000_200031


namespace NUMINAMATH_CALUDE_ellipse_min_distance_sum_l2000_200022

theorem ellipse_min_distance_sum (x y : ℝ) : 
  (x^2 / 2 + y^2 = 1) →  -- Point (x, y) is on the ellipse
  (∃ (min : ℝ), (∀ (x' y' : ℝ), x'^2 / 2 + y'^2 = 1 → 
    (x'^2 + y'^2) + ((x' + 1)^2 + y'^2) ≥ min) ∧ 
    min = 2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_min_distance_sum_l2000_200022
