import Mathlib

namespace NUMINAMATH_CALUDE_number_of_valid_arrangements_l3409_340991

-- Define the triangular arrangement
structure TriangularArrangement :=
  (cells : Fin 9 → Nat)

-- Define the condition for valid placement
def ValidPlacement (arr : TriangularArrangement) : Prop :=
  -- Each number from 1 to 9 is used exactly once
  (∀ n : Fin 9, ∃! i : Fin 9, arr.cells i = n.val + 1) ∧
  -- The sum in each four-cell triangle is 23
  (arr.cells 0 + arr.cells 1 + arr.cells 3 + arr.cells 4 = 23) ∧
  (arr.cells 1 + arr.cells 2 + arr.cells 4 + arr.cells 5 = 23) ∧
  (arr.cells 3 + arr.cells 4 + arr.cells 6 + arr.cells 7 = 23) ∧
  -- Specific placements as indicated by arrows
  (arr.cells 3 = 7 ∨ arr.cells 6 = 7) ∧
  (arr.cells 1 = 2 ∨ arr.cells 2 = 2 ∨ arr.cells 4 = 2 ∨ arr.cells 5 = 2)

-- The theorem to be proved
theorem number_of_valid_arrangements :
  ∃! (arrangements : Finset TriangularArrangement),
    (∀ arr ∈ arrangements, ValidPlacement arr) ∧
    arrangements.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_arrangements_l3409_340991


namespace NUMINAMATH_CALUDE_subset_condition_l3409_340931

theorem subset_condition (a : ℝ) : 
  {x : ℝ | a ≤ x ∧ x < 7} ⊆ {x : ℝ | 2 < x ∧ x < 10} ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3409_340931


namespace NUMINAMATH_CALUDE_max_chord_length_l3409_340983

-- Define the family of curves
def family_of_curves (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

-- Define the line y = 2x
def line (x y : ℝ) : Prop := y = 2 * x

-- Theorem statement
theorem max_chord_length :
  ∃ (max_length : ℝ),
    (∀ θ x₁ y₁ x₂ y₂ : ℝ,
      family_of_curves θ x₁ y₁ ∧
      family_of_curves θ x₂ y₂ ∧
      line x₁ y₁ ∧
      line x₂ y₂ →
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≤ max_length) ∧
    max_length = 8 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_chord_length_l3409_340983


namespace NUMINAMATH_CALUDE_age_calculation_l3409_340910

-- Define the current ages and time intervals
def luke_current_age : ℕ := 20
def years_to_future : ℕ := 8
def years_to_luke_future : ℕ := 4

-- Define the relationships between ages
def mr_bernard_future_age : ℕ := 3 * luke_current_age
def luke_future_age : ℕ := luke_current_age + years_to_future
def sarah_future_age : ℕ := 2 * (luke_current_age + years_to_luke_future)

-- Calculate the average future age
def average_future_age : ℚ := (mr_bernard_future_age + luke_future_age + sarah_future_age) / 3

-- Define the final result
def result : ℚ := average_future_age - 10

-- Theorem to prove
theorem age_calculation :
  result = 35 + 1/3 :=
sorry

end NUMINAMATH_CALUDE_age_calculation_l3409_340910


namespace NUMINAMATH_CALUDE_simplify_fraction_l3409_340967

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3409_340967


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l3409_340987

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem statement
theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l3409_340987


namespace NUMINAMATH_CALUDE_existence_of_unfactorable_number_l3409_340989

theorem existence_of_unfactorable_number (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ∃ y : ℕ, y < p / 2 ∧ ¬∃ (a b : ℕ), a > y ∧ b > y ∧ p * y + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_existence_of_unfactorable_number_l3409_340989


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l3409_340973

/-- Given information about ticket sales for a show, prove the cost of a children's ticket. -/
theorem childrens_ticket_cost
  (adult_ticket_cost : ℝ)
  (adult_count : ℕ)
  (total_receipts : ℝ)
  (h1 : adult_ticket_cost = 5.50)
  (h2 : adult_count = 152)
  (h3 : total_receipts = 1026)
  (h4 : adult_count = 2 * (adult_count / 2)) :
  ∃ (childrens_ticket_cost : ℝ),
    childrens_ticket_cost = 2.50 ∧
    total_receipts = adult_count * adult_ticket_cost + (adult_count / 2) * childrens_ticket_cost :=
by sorry

end NUMINAMATH_CALUDE_childrens_ticket_cost_l3409_340973


namespace NUMINAMATH_CALUDE_debate_team_girls_l3409_340955

theorem debate_team_girls (boys : ℕ) (groups : ℕ) (group_size : ℕ) (total : ℕ) :
  boys = 28 →
  groups = 8 →
  group_size = 4 →
  total = groups * group_size →
  total - boys = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_debate_team_girls_l3409_340955


namespace NUMINAMATH_CALUDE_rectangle_area_modification_l3409_340982

/-- Given a rectangle with initial dimensions 5 × 7 inches, if shortening one side by 2 inches
    results in an area of 21 square inches, then doubling the length of the other side
    will result in an area of 70 square inches. -/
theorem rectangle_area_modification (length width : ℝ) : 
  length = 5 ∧ width = 7 ∧ 
  ((length - 2) * width = 21 ∨ length * (width - 2) = 21) →
  length * (2 * width) = 70 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_modification_l3409_340982


namespace NUMINAMATH_CALUDE_final_comic_book_count_l3409_340927

def initial_books : ℕ := 22
def books_bought : ℕ := 6

theorem final_comic_book_count :
  (initial_books / 2 + books_bought : ℕ) = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_final_comic_book_count_l3409_340927


namespace NUMINAMATH_CALUDE_solve_dogwood_problem_l3409_340925

def dogwood_problem (initial_trees final_trees tomorrow_trees : ℕ) : Prop :=
  let total_new_trees := final_trees - initial_trees
  let today_trees := total_new_trees - tomorrow_trees
  today_trees = 5

theorem solve_dogwood_problem :
  dogwood_problem 7 16 4 := by sorry

end NUMINAMATH_CALUDE_solve_dogwood_problem_l3409_340925


namespace NUMINAMATH_CALUDE_add_2687_minutes_to_7am_l3409_340916

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem add_2687_minutes_to_7am (start : Time) (h : start.hours = 7) (m : start.minutes = 0) :
  addMinutes start 2687 = { hours := 3, minutes := 47, h_valid := sorry, m_valid := sorry } :=
sorry

end NUMINAMATH_CALUDE_add_2687_minutes_to_7am_l3409_340916


namespace NUMINAMATH_CALUDE_cricket_run_rate_l3409_340912

/-- Calculates the required run rate for the remaining overs in a cricket game. -/
def required_run_rate (total_overs : ℕ) (initial_overs : ℕ) (initial_run_rate : ℚ) (target_runs : ℕ) : ℚ :=
  let remaining_overs := total_overs - initial_overs
  let initial_runs := initial_run_rate * initial_overs
  let remaining_runs := target_runs - initial_runs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario. -/
theorem cricket_run_rate : required_run_rate 60 10 (32/10) 282 = 5 := by
  sorry


end NUMINAMATH_CALUDE_cricket_run_rate_l3409_340912


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3409_340976

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_surface_area := 6 * L^2
  let new_edge_length := 1.5 * L
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area * 100 = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3409_340976


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l3409_340947

theorem geometric_progression_problem (b₁ q : ℝ) 
  (h_decreasing : |q| < 1)
  (h_sum_diff : b₁ / (1 - q^2) - (b₁ * q) / (1 - q^2) = 10)
  (h_sum_squares_diff : b₁^2 / (1 - q^4) - (b₁^2 * q^2) / (1 - q^4) = 20) :
  b₁ = 5 ∧ q = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l3409_340947


namespace NUMINAMATH_CALUDE_division_practice_time_l3409_340905

-- Define the given conditions
def total_training_time : ℕ := 5 * 60  -- 5 hours in minutes
def training_days : ℕ := 10
def daily_multiplication_time : ℕ := 10

-- Define the theorem
theorem division_practice_time :
  (total_training_time - training_days * daily_multiplication_time) / training_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_practice_time_l3409_340905


namespace NUMINAMATH_CALUDE_olympiad_problem_selection_l3409_340936

theorem olympiad_problem_selection (total_initial : ℕ) (final_count : ℕ) :
  total_initial = 27 →
  final_count = 10 →
  ∃ (alina_problems masha_problems : ℕ),
    alina_problems + masha_problems = total_initial ∧
    alina_problems / 2 + 2 * masha_problems / 3 = total_initial - final_count ∧
    masha_problems - alina_problems = 15 :=
by sorry

end NUMINAMATH_CALUDE_olympiad_problem_selection_l3409_340936


namespace NUMINAMATH_CALUDE_friends_receiving_balls_l3409_340918

/-- The number of ping pong balls Eunji has -/
def total_balls : ℕ := 44

/-- The number of ping pong balls given to each friend -/
def balls_per_friend : ℕ := 4

/-- The number of friends who will receive ping pong balls -/
def num_friends : ℕ := total_balls / balls_per_friend

theorem friends_receiving_balls : num_friends = 11 := by
  sorry

end NUMINAMATH_CALUDE_friends_receiving_balls_l3409_340918


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l3409_340986

theorem largest_n_with_unique_k : 
  ∀ n : ℕ+, n ≤ 112 ↔ 
    (∃! k : ℤ, (8 : ℚ)/15 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/13) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l3409_340986


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l3409_340948

/-- Given two lines that intersect at a specific point, prove the value of k -/
theorem intersecting_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + 5) →  -- Line p equation
  (∀ x y : ℝ, y = k * x + 3) →  -- Line q equation
  -7 = 3 * (-4) + 5 →           -- Point (-4, -7) satisfies line p equation
  -7 = k * (-4) + 3 →           -- Point (-4, -7) satisfies line q equation
  k = 2.5 := by
sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l3409_340948


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3409_340945

theorem least_n_satisfying_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 15 → k ≥ n) ∧ 
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3409_340945


namespace NUMINAMATH_CALUDE_smallest_in_odd_set_l3409_340975

/-- A set of consecutive odd integers -/
def ConsecutiveOddIntegers : Set ℤ := sorry

/-- The median of a set of integers -/
def median (s : Set ℤ) : ℚ := sorry

/-- The greatest integer in a set -/
def greatest (s : Set ℤ) : ℤ := sorry

/-- The smallest integer in a set -/
def smallest (s : Set ℤ) : ℤ := sorry

theorem smallest_in_odd_set (s : Set ℤ) :
  s = ConsecutiveOddIntegers ∧
  median s = 152.5 ∧
  greatest s = 161 →
  smallest s = 138 := by sorry

end NUMINAMATH_CALUDE_smallest_in_odd_set_l3409_340975


namespace NUMINAMATH_CALUDE_total_cars_in_group_l3409_340924

/-- Given a group of cars with specific properties, we prove that the total number of cars is 137. -/
theorem total_cars_in_group (total : ℕ) 
  (no_ac : ℕ) 
  (with_stripes : ℕ) 
  (ac_no_stripes : ℕ) 
  (h1 : no_ac = 37)
  (h2 : with_stripes ≥ 51)
  (h3 : ac_no_stripes = 49)
  (h4 : total = no_ac + with_stripes + ac_no_stripes) :
  total = 137 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_in_group_l3409_340924


namespace NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l3409_340977

theorem minimum_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + 2*n = 1) :
  (1/m + 1/n) ≥ 3 + 2*Real.sqrt 2 ∧ ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m + 2*n = 1 ∧ 1/m + 1/n = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_reciprocal_sum_l3409_340977


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_range_of_a_l3409_340915

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 6}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 6} := by sorry

theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 8} := by sorry

theorem complement_A : (Aᶜ : Set ℝ) = {x : ℝ | x < 3 ∨ x ≥ 8} := by sorry

theorem range_of_a (h : A ⊆ C a) : a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_range_of_a_l3409_340915


namespace NUMINAMATH_CALUDE_units_digit_of_7_62_l3409_340968

theorem units_digit_of_7_62 : ∃ n : ℕ, 7^62 ≡ 9 [MOD 10] :=
by
  -- We'll use n = 9 to prove the existence
  use 9
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_62_l3409_340968


namespace NUMINAMATH_CALUDE_dodge_truck_count_l3409_340951

/-- The number of vehicles in the Taco Castle parking lot -/
structure VehicleCount where
  dodge : ℕ
  ford : ℕ
  toyota : ℕ
  volkswagen : ℕ
  honda : ℕ
  chevrolet : ℕ

/-- The relationships between different vehicle types in the parking lot -/
def valid_count (v : VehicleCount) : Prop :=
  v.ford = v.dodge / 3 ∧
  v.ford = 2 * v.toyota ∧
  v.volkswagen = v.toyota / 2 ∧
  v.honda = (3 * v.ford) / 4 ∧
  v.chevrolet = (2 * v.honda) / 3 ∧
  v.volkswagen = 5

theorem dodge_truck_count (v : VehicleCount) (h : valid_count v) : v.dodge = 60 := by
  sorry

end NUMINAMATH_CALUDE_dodge_truck_count_l3409_340951


namespace NUMINAMATH_CALUDE_color_congruent_triangle_l3409_340980

/-- A type representing the 1992 colors used to color the plane -/
def Color := Fin 1992

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle in the plane -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- A coloring of the plane -/
def Coloring := Point → Color

/-- Two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- A point is on the edge of a triangle (excluding vertices) -/
def on_edge (p : Point) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem color_congruent_triangle 
  (coloring : Coloring) 
  (color_exists : ∀ c : Color, ∃ p : Point, coloring p = c) 
  (T : Triangle) : 
  ∃ T' : Triangle, congruent T T' ∧ 
    ∀ (e1 e2 : Fin 3), ∃ (p1 p2 : Point) (c : Color), 
      on_edge p1 T' ∧ on_edge p2 T' ∧ 
      coloring p1 = c ∧ coloring p2 = c := by sorry

end NUMINAMATH_CALUDE_color_congruent_triangle_l3409_340980


namespace NUMINAMATH_CALUDE_remove_two_gives_eight_point_five_l3409_340909

def original_list : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

def remove_number (list : List ℕ) (n : ℕ) : List ℕ :=
  list.filter (· ≠ n)

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem remove_two_gives_eight_point_five :
  average (remove_number original_list 2) = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_remove_two_gives_eight_point_five_l3409_340909


namespace NUMINAMATH_CALUDE_domain_of_f_l3409_340932

noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - x) + Real.log (3 * x + 1) / Real.log 10

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l3409_340932


namespace NUMINAMATH_CALUDE_rainy_days_probability_l3409_340966

/-- The probability of rain on any given day -/
def p : ℚ := 1/5

/-- The number of days considered -/
def n : ℕ := 10

/-- The number of rainy days we're interested in -/
def k : ℕ := 3

/-- The probability of exactly k rainy days out of n days -/
def prob_k_rainy_days (p : ℚ) (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rainy_days_probability : 
  prob_k_rainy_days p n k = 1966080/9765625 := by sorry

end NUMINAMATH_CALUDE_rainy_days_probability_l3409_340966


namespace NUMINAMATH_CALUDE_b_100_mod_81_l3409_340978

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b_100_mod_81 : b 100 ≡ 38 [ZMOD 81] := by sorry

end NUMINAMATH_CALUDE_b_100_mod_81_l3409_340978


namespace NUMINAMATH_CALUDE_no_solution_for_coin_problem_l3409_340956

theorem no_solution_for_coin_problem : 
  ¬∃ (x y z : ℕ), x + y + z = 13 ∧ x + 3*y + 5*z = 200 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_coin_problem_l3409_340956


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3409_340959

theorem gcd_of_three_numbers :
  Nat.gcd 105 (Nat.gcd 1001 2436) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3409_340959


namespace NUMINAMATH_CALUDE_subset_condition_implies_a_range_l3409_340934

theorem subset_condition_implies_a_range (a : ℝ) : 
  (Finset.powerset {2 * a, a^2 - a}).card = 4 → a ≠ 0 ∧ a ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_implies_a_range_l3409_340934


namespace NUMINAMATH_CALUDE_equation_system_solution_l3409_340928

/-- Given a system of equations, prove the values of x and y, and the expression for 2p + q -/
theorem equation_system_solution (p q r x y : ℚ) 
  (eq1 : p / q = 6 / 7)
  (eq2 : p / r = 8 / 9)
  (eq3 : q / r = x / y) :
  x = 28 ∧ y = 27 ∧ 2 * p + q = 19 / 6 * p := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3409_340928


namespace NUMINAMATH_CALUDE_certain_number_proof_l3409_340984

theorem certain_number_proof (X : ℝ) : 0.8 * X - 0.35 * 300 = 31 → X = 170 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3409_340984


namespace NUMINAMATH_CALUDE_cosine_roots_l3409_340995

theorem cosine_roots (t : ℝ) : 
  (32 * (Real.cos (6 * π / 180))^5 - 40 * (Real.cos (6 * π / 180))^3 + 10 * Real.cos (6 * π / 180) - Real.sqrt 3 = 0) →
  (32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3 = 0 ↔ 
    t = Real.cos (66 * π / 180) ∨ 
    t = Real.cos (78 * π / 180) ∨ 
    t = Real.cos (138 * π / 180) ∨ 
    t = Real.cos (150 * π / 180) ∨ 
    t = Real.cos (6 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_roots_l3409_340995


namespace NUMINAMATH_CALUDE_exactly_two_successes_out_of_three_l3409_340906

/-- The probability of making a successful shot -/
def p : ℚ := 2 / 3

/-- The number of attempts -/
def n : ℕ := 3

/-- The number of successful shots -/
def k : ℕ := 2

/-- The probability of making exactly k successful shots out of n attempts -/
def probability_k_successes : ℚ := 
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem exactly_two_successes_out_of_three : 
  probability_k_successes = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_successes_out_of_three_l3409_340906


namespace NUMINAMATH_CALUDE_binomial_20_19_l3409_340917

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_20_19_l3409_340917


namespace NUMINAMATH_CALUDE_card_difference_l3409_340943

theorem card_difference (janet brenda mara : ℕ) : 
  janet > brenda →
  mara = 2 * janet →
  janet + brenda + mara = 211 →
  mara = 150 - 40 →
  janet - brenda = 9 := by
sorry

end NUMINAMATH_CALUDE_card_difference_l3409_340943


namespace NUMINAMATH_CALUDE_inequality_proof_l3409_340940

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2*y*z)) + (y^2 / (y^2 + 2*z*x)) + (z^2 / (z^2 + 2*x*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3409_340940


namespace NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l3409_340961

-- Define the angle α
def α : Real := sorry

-- Define the conditions
axiom cos_α : Real.cos α = -1/5
axiom sin_α : Real.sin α = 2 * Real.sqrt 6 / 5

-- Define the second quadrant
def second_quadrant (θ : Real) : Prop :=
  Real.cos θ < 0 ∧ Real.sin θ > 0

-- Theorem to prove
theorem terminal_side_in_second_quadrant : second_quadrant α := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l3409_340961


namespace NUMINAMATH_CALUDE_probability_prime_1_to_30_l3409_340926

def is_prime (n : ℕ) : Prop := sorry

def count_primes (a b : ℕ) : ℕ := sorry

theorem probability_prime_1_to_30 : 
  let n := 30
  let prime_count := count_primes 1 n
  (prime_count : ℚ) / n = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_prime_1_to_30_l3409_340926


namespace NUMINAMATH_CALUDE_piggy_bank_coins_l3409_340999

theorem piggy_bank_coins (sequence : Fin 6 → ℕ) 
  (h1 : sequence 0 = 72)
  (h2 : sequence 1 = 81)
  (h3 : sequence 2 = 90)
  (h5 : sequence 4 = 108)
  (h6 : sequence 5 = 117)
  (h_arithmetic : ∀ i : Fin 5, sequence (i + 1) - sequence i = sequence 1 - sequence 0) :
  sequence 3 = 99 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_coins_l3409_340999


namespace NUMINAMATH_CALUDE_apples_bought_is_three_l3409_340962

/-- Calculates the number of apples bought given the total cost, number of oranges,
    price difference between oranges and apples, and the cost of each fruit. -/
def apples_bought (total_cost orange_count price_diff fruit_cost : ℚ) : ℚ :=
  (total_cost - orange_count * (fruit_cost + price_diff)) / fruit_cost

/-- Theorem stating that under the given conditions, the number of apples bought is 3. -/
theorem apples_bought_is_three :
  let total_cost : ℚ := 456/100
  let orange_count : ℚ := 7
  let price_diff : ℚ := 28/100
  let fruit_cost : ℚ := 26/100
  apples_bought total_cost orange_count price_diff fruit_cost = 3 := by
  sorry

#eval apples_bought (456/100) 7 (28/100) (26/100)

end NUMINAMATH_CALUDE_apples_bought_is_three_l3409_340962


namespace NUMINAMATH_CALUDE_remainder_equality_l3409_340969

theorem remainder_equality (P P' D R R' r r' : ℕ) 
  (h1 : P > P') 
  (h2 : R = P % D) 
  (h3 : R' = P' % D) 
  (h4 : r = (P * P') % D) 
  (h5 : r' = (R * R') % D) : 
  r = r' := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l3409_340969


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l3409_340913

theorem garage_sale_pricing (total_items : ℕ) (n : ℕ) : 
  total_items = 34 →
  n = (total_items - 20) →
  n = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l3409_340913


namespace NUMINAMATH_CALUDE_expression_equivalence_l3409_340903

theorem expression_equivalence (a b : ℝ) 
  (h1 : (a + b) - (a - b) ≠ 0) 
  (h2 : a + b + a - b ≠ 0) : 
  let P := a + b
  let Q := a - b
  ((P + Q)^2 / (P - Q)^2) - ((P - Q)^2 / (P + Q)^2) = (a^2 + b^2) * (a^2 - b^2) / (a^2 * b^2) := by
sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3409_340903


namespace NUMINAMATH_CALUDE_tan_half_product_l3409_340965

theorem tan_half_product (a b : Real) : 
  3 * (Real.cos a + Real.sin b) + 7 * (Real.cos a * Real.cos b + 1) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = 3 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_product_l3409_340965


namespace NUMINAMATH_CALUDE_park_diameter_l3409_340997

/-- Given a circular park with a fountain, garden ring, and walking path, 
    prove that the diameter of the outer boundary is 38 feet. -/
theorem park_diameter (fountain_diameter walking_path_width garden_ring_width : ℝ) 
  (h1 : fountain_diameter = 10)
  (h2 : walking_path_width = 6)
  (h3 : garden_ring_width = 8) :
  2 * (fountain_diameter / 2 + garden_ring_width + walking_path_width) = 38 := by
  sorry


end NUMINAMATH_CALUDE_park_diameter_l3409_340997


namespace NUMINAMATH_CALUDE_b_speed_is_20_l3409_340919

/-- The speed of person A in km/h -/
def speed_a : ℝ := 10

/-- The head start time of person A in hours -/
def head_start : ℝ := 5

/-- The total distance traveled when B catches up with A in km -/
def total_distance : ℝ := 100

/-- The speed of person B in km/h -/
def speed_b : ℝ := 20

theorem b_speed_is_20 :
  speed_b = (total_distance - speed_a * head_start) / (total_distance / speed_a - head_start) :=
by sorry

end NUMINAMATH_CALUDE_b_speed_is_20_l3409_340919


namespace NUMINAMATH_CALUDE_line_parabola_intersection_range_l3409_340911

/-- The range of m for which a line and a parabola have exactly one common point -/
theorem line_parabola_intersection_range (m : ℝ) : 
  (∃! x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ 
   (2 * x - 2 * m = x^2 + m * x - 1)) ↔ 
  (-3/5 < m ∧ m < 5) :=
by sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_range_l3409_340911


namespace NUMINAMATH_CALUDE_unique_valid_number_l3409_340923

def is_valid_number (n : ℕ) : Prop :=
  n % 25 = 0 ∧ n % 35 = 0 ∧
  (∃ (a b c : ℕ), a * n ≤ 1050 ∧ b * n ≤ 1050 ∧ c * n ≤ 1050 ∧
   a < b ∧ b < c ∧
   ∀ (x : ℕ), x * n ≤ 1050 → x = a ∨ x = b ∨ x = c)

theorem unique_valid_number : 
  is_valid_number 350 ∧ ∀ (m : ℕ), is_valid_number m → m = 350 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3409_340923


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l3409_340942

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (3, -4) and (5, -8) is -24. -/
theorem midpoint_coordinate_product : 
  let x₁ : ℝ := 3
  let y₁ : ℝ := -4
  let x₂ : ℝ := 5
  let y₂ : ℝ := -8
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x * midpoint_y = -24 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l3409_340942


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l3409_340900

theorem power_zero_eq_one (x : ℝ) : x ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l3409_340900


namespace NUMINAMATH_CALUDE_solve_equations_l3409_340921

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x - 4 = 2 * x + 5
def equation2 (x : ℝ) : Prop := (x - 3) / 4 - (2 * x + 1) / 2 = 1

-- State the theorem
theorem solve_equations :
  (∃! x : ℝ, equation1 x) ∧ (∃! x : ℝ, equation2 x) ∧
  (∀ x : ℝ, equation1 x → x = 9) ∧
  (∀ x : ℝ, equation2 x → x = -6) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l3409_340921


namespace NUMINAMATH_CALUDE_sin_45_minus_sin_15_l3409_340958

theorem sin_45_minus_sin_15 : 
  Real.sin (45 * π / 180) - Real.sin (15 * π / 180) = (3 * Real.sqrt 2 - Real.sqrt 6) / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_45_minus_sin_15_l3409_340958


namespace NUMINAMATH_CALUDE_coefficient_of_y_l3409_340902

theorem coefficient_of_y (y : ℝ) : 
  let expression := 5 * (y - 6) + 6 * (9 - 3 * y^2 + 7 * y) - 10 * (3 * y - 2)
  ∃ a b c : ℝ, expression = a * y^2 + 17 * y + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l3409_340902


namespace NUMINAMATH_CALUDE_probability_empty_bottle_day14_expected_pills_taken_l3409_340990

/-- Represents the pill-taking scenario with two bottles --/
structure PillScenario where
  totalDays : ℕ
  pillsPerBottle : ℕ
  bottles : ℕ

/-- Calculates the probability of finding an empty bottle on a specific day --/
def probabilityEmptyBottle (scenario : PillScenario) (day : ℕ) : ℚ :=
  sorry

/-- Calculates the expected number of pills taken when discovering an empty bottle --/
def expectedPillsTaken (scenario : PillScenario) : ℚ :=
  sorry

/-- Theorem stating the probability of finding an empty bottle on the 14th day --/
theorem probability_empty_bottle_day14 (scenario : PillScenario) :
  scenario.totalDays = 14 ∧ scenario.pillsPerBottle = 10 ∧ scenario.bottles = 2 →
  probabilityEmptyBottle scenario 14 = 143 / 4096 :=
sorry

/-- Theorem stating the expected number of pills taken when discovering an empty bottle --/
theorem expected_pills_taken (scenario : PillScenario) (ε : ℚ) :
  scenario.pillsPerBottle = 10 ∧ scenario.bottles = 2 →
  ∃ n : ℕ, abs (expectedPillsTaken scenario - 173 / 10) < ε ∧ n > 0 :=
sorry

end NUMINAMATH_CALUDE_probability_empty_bottle_day14_expected_pills_taken_l3409_340990


namespace NUMINAMATH_CALUDE_f_properties_l3409_340974

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

theorem f_properties :
  let f : ℝ × ℝ → ℝ × ℝ := λ p ↦ (p.1 + p.2, p.1 * p.2)
  (f (1, -2) = (-1, -2)) ∧
  (f (2, -1) = (1, -2)) ∧
  (f (-1, 2) = (1, -2)) ∧
  (∀ a b : ℝ, f (a, b) = (1, -2) → (a = 2 ∧ b = -1) ∨ (a = -1 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3409_340974


namespace NUMINAMATH_CALUDE_intersection_equals_singleton_two_l3409_340908

def M : Set ℤ := {1, 2, 3, 4}
def N : Set ℤ := {-2, 2}

theorem intersection_equals_singleton_two : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_singleton_two_l3409_340908


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l3409_340957

theorem least_three_digit_multiple : ∃ n : ℕ,
  (n ≥ 100 ∧ n < 1000) ∧
  2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 3 ∣ n ∧
  ∀ m : ℕ, (m ≥ 100 ∧ m < 1000 ∧ 2 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 3 ∣ m) → n ≤ m :=
by
  use 210
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l3409_340957


namespace NUMINAMATH_CALUDE_final_racers_count_l3409_340954

/-- Calculates the number of racers remaining after each elimination round -/
def remaining_racers (initial : ℕ) (first_elim : ℕ) (second_elim_frac : ℚ) (third_elim_frac : ℚ) : ℕ :=
  let after_first := initial - first_elim
  let after_second := after_first - (after_first * second_elim_frac).floor
  (after_second - (after_second * third_elim_frac).floor).toNat

/-- Theorem stating that given the initial conditions, 30 racers remain for the final section -/
theorem final_racers_count :
  remaining_racers 100 10 (1/3) (1/2) = 30 := by
  sorry


end NUMINAMATH_CALUDE_final_racers_count_l3409_340954


namespace NUMINAMATH_CALUDE_sandra_share_l3409_340964

/-- Represents the amount of money each person receives -/
structure Share :=
  (amount : ℕ)

/-- Represents the ratio of money distribution -/
structure Ratio :=
  (sandra : ℕ)
  (amy : ℕ)
  (ruth : ℕ)

/-- Calculates the share based on the ratio and a known share -/
def calculateShare (ratio : Ratio) (knownShare : Share) (partInRatio : ℕ) : Share :=
  ⟨knownShare.amount * (ratio.sandra / partInRatio)⟩

theorem sandra_share (ratio : Ratio) (amyShare : Share) :
  ratio.sandra = 2 ∧ ratio.amy = 1 ∧ amyShare.amount = 50 →
  (calculateShare ratio amyShare ratio.amy).amount = 100 := by
  sorry

#check sandra_share

end NUMINAMATH_CALUDE_sandra_share_l3409_340964


namespace NUMINAMATH_CALUDE_smallest_norm_v_l3409_340922

theorem smallest_norm_v (v : ℝ × ℝ) (h : ‖v + (4, 2)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_v_l3409_340922


namespace NUMINAMATH_CALUDE_mary_seashells_count_l3409_340993

/-- The number of seashells found by Mary and Jessica together -/
def total_seashells : ℕ := 59

/-- The number of seashells found by Jessica -/
def jessica_seashells : ℕ := 41

/-- The number of seashells found by Mary -/
def mary_seashells : ℕ := total_seashells - jessica_seashells

theorem mary_seashells_count : mary_seashells = 18 := by
  sorry

end NUMINAMATH_CALUDE_mary_seashells_count_l3409_340993


namespace NUMINAMATH_CALUDE_license_plate_difference_l3409_340992

/-- The number of possible license plates for Sunland -/
def sunland_plates : ℕ := 1 * (10^3) * (26^2)

/-- The number of possible license plates for Moonland -/
def moonland_plates : ℕ := (10^2) * (26^2) * (10^2)

/-- The theorem stating the difference in the number of license plates -/
theorem license_plate_difference : moonland_plates - sunland_plates = 6084000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l3409_340992


namespace NUMINAMATH_CALUDE_megan_country_albums_l3409_340970

theorem megan_country_albums :
  ∀ (country_albums pop_albums total_songs songs_per_album : ℕ),
    pop_albums = 8 →
    songs_per_album = 7 →
    total_songs = 70 →
    total_songs = country_albums * songs_per_album + pop_albums * songs_per_album →
    country_albums = 2 := by
  sorry

end NUMINAMATH_CALUDE_megan_country_albums_l3409_340970


namespace NUMINAMATH_CALUDE_work_done_by_combined_forces_l3409_340920

/-- Work done by combined forces -/
theorem work_done_by_combined_forces
  (F₁ : ℝ × ℝ)
  (F₂ : ℝ × ℝ)
  (S : ℝ × ℝ)
  (h₁ : F₁ = (Real.log 2, Real.log 2))
  (h₂ : F₂ = (Real.log 5, Real.log 2))
  (h₃ : S = (2 * Real.log 5, 1)) :
  (F₁.1 + F₂.1) * S.1 + (F₁.2 + F₂.2) * S.2 = 2 := by
  sorry

#check work_done_by_combined_forces

end NUMINAMATH_CALUDE_work_done_by_combined_forces_l3409_340920


namespace NUMINAMATH_CALUDE_no_integer_solution_l3409_340979

theorem no_integer_solution : ¬∃ y : ℤ, (8 : ℝ)^3 + 4^3 + 2^10 = 2^y := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3409_340979


namespace NUMINAMATH_CALUDE_factorial_ratio_l3409_340949

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3409_340949


namespace NUMINAMATH_CALUDE_pick_two_different_colors_custom_deck_l3409_340952

/-- A custom deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)
  (cards_per_suit : ℕ)

/-- The number of ways to pick two different cards of different colors -/
def pick_two_different_colors (d : Deck) : ℕ :=
  d.total_cards * (d.cards_per_suit * d.red_suits)

/-- Theorem stating the number of ways to pick two different cards of different colors -/
theorem pick_two_different_colors_custom_deck :
  ∃ (d : Deck), 
    d.total_cards = 60 ∧
    d.num_suits = 4 ∧
    d.red_suits = 2 ∧
    d.black_suits = 2 ∧
    d.cards_per_suit = 15 ∧
    pick_two_different_colors d = 1800 := by
  sorry

end NUMINAMATH_CALUDE_pick_two_different_colors_custom_deck_l3409_340952


namespace NUMINAMATH_CALUDE_matrix_determinant_l3409_340950

def matrix : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, 0; 8, 5, -2; 3, -1, 6]

theorem matrix_determinant :
  Matrix.det matrix = 138 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l3409_340950


namespace NUMINAMATH_CALUDE_shaded_area_l3409_340996

/-- Given a square and a rhombus with shared side, calculates the area of the region inside the square but outside the rhombus -/
theorem shaded_area (square_area rhombus_area : ℝ) : 
  square_area = 25 →
  rhombus_area = 20 →
  ∃ (shaded_area : ℝ), shaded_area = square_area - (rhombus_area * 0.7) ∧ shaded_area = 11 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_l3409_340996


namespace NUMINAMATH_CALUDE_permutation_count_l3409_340963

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def isValidPermutation (π : Fin 10 → Fin 10) : Prop :=
  Function.Bijective π ∧
  ∀ m n : Fin 10, isPrime ((m : ℕ) + (n : ℕ)) → isPrime ((π m : ℕ) + (π n : ℕ))

theorem permutation_count :
  (∃! (count : ℕ), ∃ (perms : Finset (Fin 10 → Fin 10)),
    Finset.card perms = count ∧
    ∀ π ∈ perms, isValidPermutation π ∧
    ∀ π, isValidPermutation π → π ∈ perms) ∧
  (∃ (perms : Finset (Fin 10 → Fin 10)),
    Finset.card perms = 4 ∧
    ∀ π ∈ perms, isValidPermutation π ∧
    ∀ π, isValidPermutation π → π ∈ perms) :=
by sorry

end NUMINAMATH_CALUDE_permutation_count_l3409_340963


namespace NUMINAMATH_CALUDE_pencil_cost_l3409_340937

-- Define the cost of a pen and a pencil in cents
variable (p q : ℚ)

-- Define the conditions from the problem
def condition1 : Prop := 3 * p + 4 * q = 287
def condition2 : Prop := 5 * p + 2 * q = 236

-- Theorem to prove
theorem pencil_cost (h1 : condition1 p q) (h2 : condition2 p q) : q = 52 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l3409_340937


namespace NUMINAMATH_CALUDE_gcd_of_sides_gt_one_l3409_340907

/-- A triangle with integer sides -/
structure IntegerTriangle where
  a : ℕ  -- side BC
  b : ℕ  -- side CA
  c : ℕ  -- side AB
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem to be proved -/
theorem gcd_of_sides_gt_one
  (t : IntegerTriangle)
  (side_order : t.c < t.b)
  (tangent_intersect : ℕ)  -- AD, the intersection of tangent at A with BC
  : Nat.gcd t.b t.c > 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_sides_gt_one_l3409_340907


namespace NUMINAMATH_CALUDE_scientific_notation_361000000_l3409_340904

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_361000000 :
  toScientificNotation 361000000 = ScientificNotation.mk 3.61 8 sorry := by sorry

end NUMINAMATH_CALUDE_scientific_notation_361000000_l3409_340904


namespace NUMINAMATH_CALUDE_ellipse_intersection_l3409_340944

/-- Definition of the ellipse with given properties -/
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (0, 3)
  let F₂ : ℝ × ℝ := (4, 0)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 7

theorem ellipse_intersection :
  ellipse (0, 0) → 
  (∃ x : ℝ, x ≠ 0 ∧ ellipse (x, 0)) → 
  ellipse (56/11, 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_l3409_340944


namespace NUMINAMATH_CALUDE_symmetric_derivative_minimum_value_l3409_340994

-- Define the function f
def f (b c x : ℝ) : ℝ := x^3 + b*x^2 + c*x

-- Define the derivative of f
def f' (b c x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

-- State the theorem
theorem symmetric_derivative_minimum_value (b c : ℝ) :
  (∀ x : ℝ, f' b c (4 - x) = f' b c x) →  -- f' is symmetric about x = 2
  (∃ t : ℝ, ∀ x : ℝ, f b c t ≤ f b c x) →  -- f has a minimum value
  (b = -6) ∧  -- Part 1: value of b
  (∃ g : ℝ → ℝ, (∀ t > 2, g t = f b c t) ∧  -- Part 2: domain of g
                (∀ y : ℝ, (∃ t > 2, g t = y) ↔ y < 8))  -- Part 3: range of g
  := by sorry

end NUMINAMATH_CALUDE_symmetric_derivative_minimum_value_l3409_340994


namespace NUMINAMATH_CALUDE_total_is_sum_of_eaten_and_saved_l3409_340930

/-- The number of strawberries Micah picked in total -/
def total_strawberries : ℕ := sorry

/-- The number of strawberries Micah ate -/
def eaten_strawberries : ℕ := 6

/-- The number of strawberries Micah saved for his mom -/
def saved_strawberries : ℕ := 18

/-- Theorem stating that the total number of strawberries is the sum of eaten and saved strawberries -/
theorem total_is_sum_of_eaten_and_saved : 
  total_strawberries = eaten_strawberries + saved_strawberries :=
by sorry

end NUMINAMATH_CALUDE_total_is_sum_of_eaten_and_saved_l3409_340930


namespace NUMINAMATH_CALUDE_movie_theater_attendance_l3409_340914

theorem movie_theater_attendance 
  (total_seats : ℕ) 
  (empty_seats : ℕ) 
  (h1 : total_seats = 750) 
  (h2 : empty_seats = 218) : 
  total_seats - empty_seats = 532 := by
sorry

end NUMINAMATH_CALUDE_movie_theater_attendance_l3409_340914


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_l3409_340953

theorem tan_alpha_2_implies (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6/13 ∧
  3 * Real.sin α ^ 2 + 3 * Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_l3409_340953


namespace NUMINAMATH_CALUDE_max_product_of_focal_distances_l3409_340933

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Statement: The maximum value of |PF1| * |PF2| is 25 for any point P on the ellipse -/
theorem max_product_of_focal_distances :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
  ∃ M : ℝ, M = 25 ∧ ∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
  (distance P F1) * (distance P F2) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_product_of_focal_distances_l3409_340933


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3409_340985

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℚ) : Prop := m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
def slope_general_form (a b : ℚ) : ℚ := -a / b

/-- The slope of a line in the form y = mx + b is m -/
def slope_slope_intercept_form (m : ℚ) : ℚ := m

theorem parallel_lines_m_value :
  ∀ m : ℚ, 
  parallel (slope_general_form 2 m) (slope_slope_intercept_form 3) →
  m = -2/3 := by
sorry


end NUMINAMATH_CALUDE_parallel_lines_m_value_l3409_340985


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3409_340935

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) :
  π * r^2 = 225 * π → 2 * π * r^2 + π * r^2 = 675 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3409_340935


namespace NUMINAMATH_CALUDE_negation_equivalence_l3409_340981

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3409_340981


namespace NUMINAMATH_CALUDE_closure_of_M_union_N_l3409_340972

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem closure_of_M_union_N :
  closure (M ∪ N) = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_closure_of_M_union_N_l3409_340972


namespace NUMINAMATH_CALUDE_minimum_guests_l3409_340998

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 327) (h2 : max_per_guest = 2) :
  ∃ (min_guests : ℕ), min_guests = 164 ∧ min_guests * max_per_guest ≥ total_food ∧ 
  ∀ (n : ℕ), n * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l3409_340998


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3409_340939

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℝ, (x + 3) * (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) → 
  p = -31 ∧ q = -71 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3409_340939


namespace NUMINAMATH_CALUDE_reading_ratio_l3409_340988

/-- Given the reading speeds of Carter, Oliver, and Lucy, prove that the ratio of pages
    Carter can read to pages Lucy can read in 1 hour is 1/2. -/
theorem reading_ratio (carter_pages oliver_pages lucy_extra : ℕ) 
  (h1 : carter_pages = 30)
  (h2 : oliver_pages = 40)
  (h3 : lucy_extra = 20) :
  (carter_pages : ℚ) / ((oliver_pages : ℚ) + lucy_extra) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_reading_ratio_l3409_340988


namespace NUMINAMATH_CALUDE_ratio_x_to_2y_l3409_340941

theorem ratio_x_to_2y (x y : ℝ) (h : (7 * x + 8 * y) / (x - 2 * y) = 29) : 
  x / (2 * y) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_2y_l3409_340941


namespace NUMINAMATH_CALUDE_inequality_holds_l3409_340938

/-- A quadratic function with the given symmetry property -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The symmetry property of f -/
axiom symmetry_at_3 (b c : ℝ) : ∀ t : ℝ, f b c (3 + t) = f b c (3 - t)

/-- The main theorem stating the inequality -/
theorem inequality_holds (b c : ℝ) : f b c 3 < f b c 1 ∧ f b c 1 < f b c 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3409_340938


namespace NUMINAMATH_CALUDE_probability_three_one_is_five_ninths_l3409_340901

def total_balls : ℕ := 18
def blue_balls : ℕ := 10
def red_balls : ℕ := 8
def drawn_balls : ℕ := 4

def probability_three_one : ℚ :=
  let favorable_outcomes := Nat.choose blue_balls 3 * Nat.choose red_balls 1 +
                            Nat.choose blue_balls 1 * Nat.choose red_balls 3
  let total_outcomes := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_three_one_is_five_ninths :
  probability_three_one = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_one_is_five_ninths_l3409_340901


namespace NUMINAMATH_CALUDE_tv_screen_area_difference_l3409_340929

theorem tv_screen_area_difference : 
  let diagonal_large : ℝ := 22
  let diagonal_small : ℝ := 20
  let area_large := diagonal_large ^ 2
  let area_small := diagonal_small ^ 2
  area_large - area_small = 84 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_area_difference_l3409_340929


namespace NUMINAMATH_CALUDE_min_value_of_sum_squares_l3409_340960

theorem min_value_of_sum_squares (a b c m : ℝ) 
  (sum_eq_one : a + b + c = 1) 
  (m_def : m = a^2 + b^2 + c^2) : 
  m ≥ 1/3 ∧ ∃ (a b c : ℝ), a + b + c = 1 ∧ a^2 + b^2 + c^2 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_squares_l3409_340960


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3409_340946

theorem min_value_expression (a b c : ℝ) 
  (ha : -1 < a ∧ a < 1) 
  (hb : -1 < b ∧ b < 1) 
  (hc : -1 < c ∧ c < 1) :
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + 
  (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2))) ≥ 2 :=
by sorry

theorem min_value_achieved (a b c : ℝ) 
  (ha : a = 0) 
  (hb : b = 0) 
  (hc : c = 0) :
  (1 / ((1 - a^2) * (1 - b^2) * (1 - c^2))) + 
  (1 / ((1 + a^2) * (1 + b^2) * (1 + c^2))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3409_340946


namespace NUMINAMATH_CALUDE_integer_valued_poly_implies_24P_integer_coeffs_l3409_340971

/-- A polynomial of degree 4 that takes integer values for integer inputs -/
def IntegerValuedPolynomial (P : ℝ → ℝ) : Prop :=
  (∃ a b c d e : ℝ, ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧
  (∀ n : ℤ, ∃ m : ℤ, P n = m)

/-- The coefficients of 24P(x) are integers -/
def Coefficients24PAreIntegers (P : ℝ → ℝ) : Prop :=
  ∃ a' b' c' d' e' : ℤ, ∀ x, 24 * P x = a'*x^4 + b'*x^3 + c'*x^2 + d'*x + e'

theorem integer_valued_poly_implies_24P_integer_coeffs
  (P : ℝ → ℝ) (h : IntegerValuedPolynomial P) :
  Coefficients24PAreIntegers P :=
sorry

end NUMINAMATH_CALUDE_integer_valued_poly_implies_24P_integer_coeffs_l3409_340971
