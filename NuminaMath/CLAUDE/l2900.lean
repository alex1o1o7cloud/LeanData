import Mathlib

namespace NUMINAMATH_CALUDE_players_who_quit_l2900_290001

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8)
  (h2 : lives_per_player = 3)
  (h3 : total_lives = 15) :
  initial_players - (total_lives / lives_per_player) = 3 :=
by sorry

end NUMINAMATH_CALUDE_players_who_quit_l2900_290001


namespace NUMINAMATH_CALUDE_quadratic_form_only_trivial_solution_l2900_290086

theorem quadratic_form_only_trivial_solution (a b c d : ℤ) :
  a^2 + 5*b^2 - 2*c^2 - 2*c*d - 3*d^2 = 0 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_only_trivial_solution_l2900_290086


namespace NUMINAMATH_CALUDE_max_roads_removal_l2900_290005

/-- A graph representing the Empire of Westeros --/
structure WesterosGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  is_connected : Bool
  vertex_count : vertices.card = 1000
  edge_count : edges.card = 2017
  initial_connectivity : is_connected = true

/-- The result of removing roads from the graph --/
structure KingdomFormation where
  removed_roads : Nat
  kingdom_count : Nat

/-- The maximum number of roads that can be removed to form exactly 7 kingdoms --/
def max_removable_roads (g : WesterosGraph) : Nat :=
  993

/-- Theorem stating the maximum number of removable roads --/
theorem max_roads_removal (g : WesterosGraph) :
  ∃ (kf : KingdomFormation),
    kf.removed_roads = max_removable_roads g ∧
    kf.kingdom_count = 7 ∧
    ∀ (kf' : KingdomFormation),
      kf'.kingdom_count = 7 → kf'.removed_roads ≤ kf.removed_roads :=
sorry


end NUMINAMATH_CALUDE_max_roads_removal_l2900_290005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2900_290098

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (hd : d ≠ 0)
  (ha : arithmeticSequence a d)
  (hineq : (a 3)^2 ≠ (a 1) * (a 9)) :
  (a 3) / (a 6) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2900_290098


namespace NUMINAMATH_CALUDE_simplify_expressions_l2900_290029

theorem simplify_expressions :
  (3 * Real.sqrt 20 - Real.sqrt 45 + Real.sqrt (1/5) = 16 * Real.sqrt 5 / 5) ∧
  ((Real.sqrt 6 - 2 * Real.sqrt 3)^2 - (2 * Real.sqrt 5 + Real.sqrt 2) * (2 * Real.sqrt 5 - Real.sqrt 2) = -12 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2900_290029


namespace NUMINAMATH_CALUDE_graph_equation_two_lines_l2900_290023

theorem graph_equation_two_lines (x y : ℝ) :
  (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_graph_equation_two_lines_l2900_290023


namespace NUMINAMATH_CALUDE_batsman_average_after_15th_inning_l2900_290079

def batsman_average (total_innings : ℕ) (last_inning_score : ℕ) (average_increase : ℕ) : ℚ :=
  let previous_average := (total_innings - 1 : ℚ) * (average_increase : ℚ) + (last_inning_score : ℚ) / (total_innings : ℚ)
  previous_average + average_increase

theorem batsman_average_after_15th_inning :
  batsman_average 15 75 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_15th_inning_l2900_290079


namespace NUMINAMATH_CALUDE_total_bees_after_changes_l2900_290059

/-- Represents a bee hive with initial bees and changes in population --/
structure BeeHive where
  initial : ℕ
  fly_in : ℕ
  fly_out : ℕ

/-- Calculates the final number of bees in a hive after changes --/
def final_bees (hive : BeeHive) : ℕ :=
  hive.initial + hive.fly_in - hive.fly_out

/-- Represents the bee colony --/
def BeeColony : List BeeHive := [
  { initial := 45, fly_in := 12, fly_out := 8 },
  { initial := 60, fly_in := 15, fly_out := 20 },
  { initial := 75, fly_in := 10, fly_out := 5 }
]

/-- Theorem stating the total number of bees after changes --/
theorem total_bees_after_changes :
  (BeeColony.map final_bees).sum = 184 := by
  sorry

end NUMINAMATH_CALUDE_total_bees_after_changes_l2900_290059


namespace NUMINAMATH_CALUDE_company_layoff_payment_l2900_290002

theorem company_layoff_payment (total_employees : ℕ) (salary : ℕ) (layoff_fraction : ℚ) : 
  total_employees = 450 →
  salary = 2000 →
  layoff_fraction = 1/3 →
  (total_employees : ℚ) * (1 - layoff_fraction) * salary = 600000 := by
sorry

end NUMINAMATH_CALUDE_company_layoff_payment_l2900_290002


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l2900_290014

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = 370 →
    12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
    balcony_tickets - orchestra_tickets = 190 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l2900_290014


namespace NUMINAMATH_CALUDE_find_divisor_l2900_290089

def is_divisor (n : ℕ) (d : ℕ) : Prop :=
  (n / d : ℚ) + 8 = 61

theorem find_divisor :
  ∃ (d : ℕ), is_divisor 265 d ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2900_290089


namespace NUMINAMATH_CALUDE_sin_ratio_comparison_l2900_290030

theorem sin_ratio_comparison :
  (Real.sin (3 * Real.pi / 180)) / (Real.sin (4 * Real.pi / 180)) >
  (Real.sin (1 * Real.pi / 180)) / (Real.sin (2 * Real.pi / 180)) :=
by sorry

end NUMINAMATH_CALUDE_sin_ratio_comparison_l2900_290030


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2900_290082

def M : Set ℝ := {x | x^2 - 6*x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2900_290082


namespace NUMINAMATH_CALUDE_cricket_bat_profit_l2900_290054

/-- Calculates the profit amount for a cricket bat sale -/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 850 →
  profit_percentage = 42.857142857142854 →
  ∃ (cost_price : ℝ) (profit : ℝ),
    cost_price > 0 ∧
    profit > 0 ∧
    selling_price = cost_price + profit ∧
    profit_percentage = (profit / cost_price) * 100 ∧
    profit = 255 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_l2900_290054


namespace NUMINAMATH_CALUDE_lcm_of_9_12_18_l2900_290081

theorem lcm_of_9_12_18 : Nat.lcm (Nat.lcm 9 12) 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_18_l2900_290081


namespace NUMINAMATH_CALUDE_ronald_laundry_frequency_l2900_290092

/-- The number of days between Tim's laundry sessions -/
def tim_laundry_interval : ℕ := 9

/-- The number of days until Ronald and Tim do laundry together again -/
def joint_laundry_interval : ℕ := 18

/-- The number of days between Ronald's laundry sessions -/
def ronald_laundry_interval : ℕ := 3

theorem ronald_laundry_frequency :
  (joint_laundry_interval % tim_laundry_interval = 0) ∧
  (joint_laundry_interval % ronald_laundry_interval = 0) ∧
  (∀ n : ℕ, n < ronald_laundry_interval → joint_laundry_interval % n ≠ 0 ∨ n = 1) :=
sorry

end NUMINAMATH_CALUDE_ronald_laundry_frequency_l2900_290092


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2900_290076

/-- A quadratic function f(x) = (x + a)(bx + 2a) where a, b ∈ ℝ, 
    which is even and has a range of (-∞, 4] -/
def quadratic_function (a b : ℝ) : ℝ → ℝ := fun x ↦ (x + a) * (b * x + 2 * a)

/-- The property of being an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The range of a function -/
def has_range (f : ℝ → ℝ) (S : Set ℝ) : Prop := ∀ y, y ∈ S ↔ ∃ x, f x = y

theorem quadratic_function_theorem (a b : ℝ) :
  is_even (quadratic_function a b) ∧ 
  has_range (quadratic_function a b) {y | y ≤ 4} →
  quadratic_function a b = fun x ↦ -2 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2900_290076


namespace NUMINAMATH_CALUDE_sector_central_angle_l2900_290017

/-- Given a sector with radius 10 cm and perimeter 45 cm, its central angle is 2.5 radians. -/
theorem sector_central_angle (r : ℝ) (p : ℝ) (α : ℝ) : 
  r = 10 → p = 45 → α = (p - 2 * r) / r → α = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2900_290017


namespace NUMINAMATH_CALUDE_k_bound_l2900_290060

/-- A sequence a_n defined as n^2 - kn for positive integers n -/
def a (k : ℝ) (n : ℕ) : ℝ := n^2 - k * n

/-- The property that a sequence is monotonically increasing -/
def MonotonicallyIncreasing (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, f (n + 1) > f n

/-- Theorem: If the sequence a_n is monotonically increasing, then k < 3 -/
theorem k_bound (k : ℝ) (h : MonotonicallyIncreasing (a k)) : k < 3 := by
  sorry

end NUMINAMATH_CALUDE_k_bound_l2900_290060


namespace NUMINAMATH_CALUDE_A_less_than_B_l2900_290087

theorem A_less_than_B (x y : ℝ) : 
  let A := -y^2 + 4*x - 3
  let B := x^2 + 2*x + 2*y
  A < B := by sorry

end NUMINAMATH_CALUDE_A_less_than_B_l2900_290087


namespace NUMINAMATH_CALUDE_f_has_one_zero_a_equals_one_l2900_290038

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / (x^2)

theorem f_has_one_zero :
  ∃! x, f x = 0 :=
sorry

theorem a_equals_one (a : ℝ) :
  (∀ x > 0, f x ≥ (2 * a * Real.log x) / x^2 + a / x) ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_a_equals_one_l2900_290038


namespace NUMINAMATH_CALUDE_chocolate_bar_calculation_l2900_290061

/-- The number of chocolate bars in each box -/
def bars_per_box : ℕ := 5

/-- The number of boxes Tom needs to sell -/
def boxes_to_sell : ℕ := 170

/-- The total number of chocolate bars Tom needs to sell -/
def total_bars : ℕ := bars_per_box * boxes_to_sell

theorem chocolate_bar_calculation :
  total_bars = 850 := by sorry

end NUMINAMATH_CALUDE_chocolate_bar_calculation_l2900_290061


namespace NUMINAMATH_CALUDE_exist_consecutive_amazing_numbers_l2900_290085

/-- Definition of an amazing number -/
def is_amazing (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    n = (Nat.gcd b c) * (Nat.gcd a (b*c)) + 
        (Nat.gcd c a) * (Nat.gcd b (c*a)) + 
        (Nat.gcd a b) * (Nat.gcd c (a*b))

/-- Theorem: There exist 2011 consecutive amazing numbers -/
theorem exist_consecutive_amazing_numbers : 
  ∃ start : ℕ, ∀ i : ℕ, i < 2011 → is_amazing (start + i) := by
  sorry

end NUMINAMATH_CALUDE_exist_consecutive_amazing_numbers_l2900_290085


namespace NUMINAMATH_CALUDE_jokes_count_l2900_290047

/-- The total number of jokes told by Jessy and Alan over two Saturdays -/
def total_jokes (jessy_first : ℕ) (alan_first : ℕ) : ℕ :=
  let first_saturday := jessy_first + alan_first
  let second_saturday := 2 * jessy_first + 2 * alan_first
  first_saturday + second_saturday

/-- Theorem stating the total number of jokes told by Jessy and Alan -/
theorem jokes_count : total_jokes 11 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_jokes_count_l2900_290047


namespace NUMINAMATH_CALUDE_transportation_cost_comparison_l2900_290011

/-- The cost function for company A -/
def cost_A (x : ℝ) : ℝ := 0.6 * x

/-- The cost function for company B -/
def cost_B (x : ℝ) : ℝ := 0.3 * x + 750

theorem transportation_cost_comparison (x : ℝ) 
  (h_x_pos : 0 < x) (h_x_upper : x < 5000) :
  (x < 2500 → cost_A x < cost_B x) ∧
  (x > 2500 → cost_B x < cost_A x) ∧
  (x = 2500 → cost_A x = cost_B x) := by
  sorry


end NUMINAMATH_CALUDE_transportation_cost_comparison_l2900_290011


namespace NUMINAMATH_CALUDE_squirrel_journey_time_l2900_290071

/-- Calculates the total journey time in minutes for a squirrel gathering nuts -/
theorem squirrel_journey_time (distance_to_tree : ℝ) (speed_to_tree : ℝ) (speed_from_tree : ℝ) :
  distance_to_tree = 2 →
  speed_to_tree = 3 →
  speed_from_tree = 2 →
  (distance_to_tree / speed_to_tree + distance_to_tree / speed_from_tree) * 60 = 100 := by
  sorry

#check squirrel_journey_time

end NUMINAMATH_CALUDE_squirrel_journey_time_l2900_290071


namespace NUMINAMATH_CALUDE_A_prime_div_B_prime_l2900_290073

/-- The series A' as defined in the problem -/
noncomputable def A' : ℝ := ∑' n, if n % 5 ≠ 0 ∧ n % 2 ≠ 0 then ((-1) ^ ((n - 1) / 2 : ℕ)) / n^2 else 0

/-- The series B' as defined in the problem -/
noncomputable def B' : ℝ := ∑' n, if n % 5 = 0 ∧ n % 2 ≠ 0 then ((-1) ^ ((n / 5 - 1) / 2 : ℕ)) / n^2 else 0

/-- The main theorem stating that A' / B' = 26 -/
theorem A_prime_div_B_prime : A' / B' = 26 := by
  sorry

end NUMINAMATH_CALUDE_A_prime_div_B_prime_l2900_290073


namespace NUMINAMATH_CALUDE_problem_statement_l2900_290025

theorem problem_statement (a b : ℝ) (h : 2 * a - b + 3 = 0) :
  2 * (2 * a + b) - 4 * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2900_290025


namespace NUMINAMATH_CALUDE_tenth_digit_of_expression_l2900_290093

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def tenthDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem tenth_digit_of_expression : 
  tenthDigit ((factorial 5 * factorial 5 - factorial 5 * factorial 3) / 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tenth_digit_of_expression_l2900_290093


namespace NUMINAMATH_CALUDE_sum_of_ages_sum_of_ages_proof_l2900_290095

/-- Proves that the sum of James and Louise's current ages is 32 years. -/
theorem sum_of_ages : ℝ → ℝ → Prop :=
  fun james louise =>
    james = louise + 9 →
    james + 5 = 3 * (louise - 3) →
    james + louise = 32

-- The proof is omitted
theorem sum_of_ages_proof : ∃ (james louise : ℝ), sum_of_ages james louise :=
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_sum_of_ages_proof_l2900_290095


namespace NUMINAMATH_CALUDE_matches_for_128_teams_l2900_290062

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_teams : ℕ
  num_teams_positive : 0 < num_teams

/-- The number of matches required to determine the championship team -/
def matches_required (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a tournament with 128 teams, 127 matches are required -/
theorem matches_for_128_teams :
  ∀ t : Tournament, t.num_teams = 128 → matches_required t = 127 := by
  sorry

#check matches_for_128_teams

end NUMINAMATH_CALUDE_matches_for_128_teams_l2900_290062


namespace NUMINAMATH_CALUDE_cube_opposite_color_l2900_290031

/-- Represents the colors of the squares --/
inductive Color
  | P | C | M | S | L | K

/-- Represents the faces of a cube --/
inductive Face
  | Top | Bottom | Front | Back | Left | Right

/-- Represents a cube formed by six hinged squares --/
structure Cube where
  faces : Face → Color

/-- Defines the opposite face relationship --/
def opposite_face : Face → Face
  | Face.Top    => Face.Bottom
  | Face.Bottom => Face.Top
  | Face.Front  => Face.Back
  | Face.Back   => Face.Front
  | Face.Left   => Face.Right
  | Face.Right  => Face.Left

theorem cube_opposite_color (c : Cube) :
  c.faces Face.Top = Color.M →
  c.faces Face.Front = Color.L →
  c.faces (opposite_face Face.Front) = Color.K :=
by sorry

end NUMINAMATH_CALUDE_cube_opposite_color_l2900_290031


namespace NUMINAMATH_CALUDE_notebook_cost_l2900_290028

theorem notebook_cost (total_cost cover_cost notebook_cost : ℚ) : 
  total_cost = 2.4 →
  notebook_cost = cover_cost + 2 →
  total_cost = notebook_cost + cover_cost →
  notebook_cost = 2.2 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l2900_290028


namespace NUMINAMATH_CALUDE_max_largest_integer_l2900_290065

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 45 →
  (max a (max b (max c (max d e)))) - (min a (min b (min c (min d e)))) = 10 →
  (max a (max b (max c (max d e)))) ≤ 215 :=
by sorry

end NUMINAMATH_CALUDE_max_largest_integer_l2900_290065


namespace NUMINAMATH_CALUDE_division_problem_l2900_290008

theorem division_problem : 
  ∃ (q r : ℕ), 253 = (15 + 13 * 3 - 5) * q + r ∧ r < (15 + 13 * 3 - 5) ∧ q = 5 ∧ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2900_290008


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2900_290004

theorem closest_integer_to_cube_root : ∃ (n : ℤ), 
  n = 10 ∧ ∀ (m : ℤ), |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2900_290004


namespace NUMINAMATH_CALUDE_parabola_intersection_l2900_290091

-- Define the parabola
def parabola (k x : ℝ) : ℝ := x^2 - (k-1)*x - 3*k - 2

-- Define the intersection points
def α (k : ℝ) : ℝ := sorry
def β (k : ℝ) : ℝ := sorry

-- Theorem statement
theorem parabola_intersection (k : ℝ) : 
  (parabola k (α k) = 0) ∧ 
  (parabola k (β k) = 0) ∧ 
  ((α k)^2 + (β k)^2 = 17) → 
  k = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2900_290091


namespace NUMINAMATH_CALUDE_rectangle_division_l2900_290040

theorem rectangle_division (a b c d e f : ℕ) : 
  (∀ a b, 39 ≠ 5 * a + 11 * b) ∧ 
  (∃ c d, 27 = 5 * c + 11 * d) ∧ 
  (∃ e f, 55 = 5 * e + 11 * f) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l2900_290040


namespace NUMINAMATH_CALUDE_area_of_circumcenter_quadrilateral_area_of_ABCD_proof_l2900_290099

/-- The area of the quadrilateral formed by the circumcenters of four equilateral triangles
    erected on the sides of a unit square (one inside, three outside) -/
theorem area_of_circumcenter_quadrilateral : ℝ :=
  let square_side_length : ℝ := 1
  let triangle_side_length : ℝ := 1
  let inside_triangle_count : ℕ := 1
  let outside_triangle_count : ℕ := 3
  (3 + Real.sqrt 3) / 6

/-- Proof of the area of the quadrilateral ABCD -/
theorem area_of_ABCD_proof :
  area_of_circumcenter_quadrilateral = (3 + Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_circumcenter_quadrilateral_area_of_ABCD_proof_l2900_290099


namespace NUMINAMATH_CALUDE_division_problem_l2900_290067

theorem division_problem (n x : ℝ) (h1 : n = 4.5) (h2 : (n / x) * 12 = 9) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2900_290067


namespace NUMINAMATH_CALUDE_complex_coordinates_of_Z_l2900_290009

theorem complex_coordinates_of_Z : 
  let Z : ℂ := (2 + 4 * Complex.I) / (1 + Complex.I)
  Z = 3 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_coordinates_of_Z_l2900_290009


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2900_290055

theorem smallest_common_multiple_of_6_and_15 :
  ∃ b : ℕ+, (∀ n : ℕ+, (6 ∣ n) → (15 ∣ n) → b ≤ n) ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2900_290055


namespace NUMINAMATH_CALUDE_sum_of_cubes_equals_cube_l2900_290003

theorem sum_of_cubes_equals_cube : 57^6 + 95^6 + 109^6 = 228^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equals_cube_l2900_290003


namespace NUMINAMATH_CALUDE_sweet_potato_problem_l2900_290046

-- Define the problem parameters
def total_harvested : ℕ := 80
def sold_to_adams : ℕ := 20
def sold_to_lenon : ℕ := 15
def traded_for_pumpkins : ℕ := 10
def pumpkins_received : ℕ := 5
def pumpkin_weight : ℕ := 3
def donation_percentage : Rat := 5 / 100

-- Define the theorem
theorem sweet_potato_problem :
  let remaining_before_donation := total_harvested - (sold_to_adams + sold_to_lenon + traded_for_pumpkins)
  let donation := (remaining_before_donation : Rat) * donation_percentage
  let remaining_after_donation := remaining_before_donation - ⌈donation⌉
  remaining_after_donation = 33 ∧ pumpkins_received * pumpkin_weight = 15 := by
  sorry


end NUMINAMATH_CALUDE_sweet_potato_problem_l2900_290046


namespace NUMINAMATH_CALUDE_science_club_team_selection_l2900_290018

theorem science_club_team_selection (n : ℕ) (k : ℕ) :
  n = 22 → k = 8 → Nat.choose n k = 319770 := by
  sorry

end NUMINAMATH_CALUDE_science_club_team_selection_l2900_290018


namespace NUMINAMATH_CALUDE_square_number_divisible_by_5_between_20_and_110_l2900_290007

theorem square_number_divisible_by_5_between_20_and_110 (y : ℕ) :
  (∃ n : ℕ, y = n^2) →
  y % 5 = 0 →
  20 < y →
  y < 110 →
  (y = 25 ∨ y = 100) :=
by sorry

end NUMINAMATH_CALUDE_square_number_divisible_by_5_between_20_and_110_l2900_290007


namespace NUMINAMATH_CALUDE_total_amount_shared_l2900_290022

theorem total_amount_shared (z y x : ℝ) : 
  z = 150 →
  y = 1.2 * z →
  x = 1.25 * y →
  x + y + z = 555 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_shared_l2900_290022


namespace NUMINAMATH_CALUDE_triangle_existence_l2900_290034

theorem triangle_existence (k : ℕ) (a b c : ℝ) 
  (h_k : k ≥ 10) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ a = y + z ∧ b = z + x ∧ c = x + y :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l2900_290034


namespace NUMINAMATH_CALUDE_z_mod_nine_l2900_290012

theorem z_mod_nine (z : ℤ) (h : ∃ k : ℤ, (z + 3) / 9 = k) : z % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_z_mod_nine_l2900_290012


namespace NUMINAMATH_CALUDE_coffee_ounces_per_pot_l2900_290097

/-- Calculates the number of ounces per pot of coffee -/
def ounces_per_pot (ounces_per_donut : ℚ) (cost_per_pot : ℚ) (dozen_donuts : ℕ) (total_cost : ℚ) : ℚ :=
  let total_donuts := dozen_donuts * 12
  let total_ounces := total_donuts * ounces_per_donut
  let num_pots := total_cost / cost_per_pot
  total_ounces / num_pots

/-- Proves that the number of ounces per pot of coffee is 12 -/
theorem coffee_ounces_per_pot :
  ounces_per_pot 2 3 3 18 = 12 := by sorry

end NUMINAMATH_CALUDE_coffee_ounces_per_pot_l2900_290097


namespace NUMINAMATH_CALUDE_right_triangle_area_l2900_290083

theorem right_triangle_area (a c : ℝ) (h1 : a = 30) (h2 : c = 34) : 
  let b := Real.sqrt (c^2 - a^2)
  (1/2) * a * b = 240 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2900_290083


namespace NUMINAMATH_CALUDE_set_intersection_complement_problem_l2900_290041

theorem set_intersection_complement_problem :
  let U : Type := ℝ
  let A : Set U := {x | x ≤ 3}
  let B : Set U := {x | x ≤ 6}
  (Aᶜ ∩ B) = {x : U | 3 < x ∧ x ≤ 6} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_problem_l2900_290041


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l2900_290010

/-- The function f(x) defined as |x+1| + |x-4| - a -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |x - 4| - a

/-- Theorem stating the minimum value of f(x) when a = 1 -/
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 4 ∧ ∃ y : ℝ, f 1 y = 4 :=
sorry

/-- Theorem stating the range of a for which f(x) ≥ 4/a + 1 holds for all real x -/
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4/a + 1) ↔ (a < 0 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l2900_290010


namespace NUMINAMATH_CALUDE_problem_statement_l2900_290058

theorem problem_statement (a b : ℝ) (h : |a + 5| + (b - 2)^2 = 0) :
  (a + b)^2010 = 3^2010 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2900_290058


namespace NUMINAMATH_CALUDE_six_b_equals_twenty_l2900_290045

theorem six_b_equals_twenty (a b : ℚ) 
  (h1 : 10 * a = b) 
  (h2 : b = 20) 
  (h3 : 120 * a * b = 800) : 
  6 * b = 20 := by
sorry

end NUMINAMATH_CALUDE_six_b_equals_twenty_l2900_290045


namespace NUMINAMATH_CALUDE_pauls_strawberries_l2900_290074

/-- Given an initial count of strawberries and an additional number picked,
    calculate the total number of strawberries. -/
def total_strawberries (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem: Paul's total strawberries after picking more -/
theorem pauls_strawberries :
  total_strawberries 42 78 = 120 := by
  sorry

end NUMINAMATH_CALUDE_pauls_strawberries_l2900_290074


namespace NUMINAMATH_CALUDE_album_distribution_ways_l2900_290043

/-- The number of ways to distribute albums to friends -/
def distribute_albums (photo_albums : ℕ) (stamp_albums : ℕ) (friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to distribute the albums -/
theorem album_distribution_ways :
  distribute_albums 2 3 4 = 10 := by sorry

end NUMINAMATH_CALUDE_album_distribution_ways_l2900_290043


namespace NUMINAMATH_CALUDE_range_of_m_l2900_290048

-- Define propositions p and q as functions of x and m
def p (x m : ℝ) : Prop := (x - m)^2 > 3*(x - m)

def q (x : ℝ) : Prop := x^2 + 3*x - 4 < 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ (∃ x, p x m ∧ ¬q x)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, necessary_but_not_sufficient m ↔ (m ≥ 1 ∨ m ≤ -7) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2900_290048


namespace NUMINAMATH_CALUDE_factory_production_theorem_l2900_290051

/-- Represents a production line with its output and sample size -/
structure ProductionLine where
  output : ℕ
  sample : ℕ

/-- Represents the factory's production data -/
structure FactoryProduction where
  total_output : ℕ
  line_a : ProductionLine
  line_b : ProductionLine
  line_c : ProductionLine

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- The main theorem about the factory's production -/
theorem factory_production_theorem (f : FactoryProduction) :
  f.total_output = 16800 ∧
  isArithmeticSequence f.line_a.sample f.line_b.sample f.line_c.sample ∧
  f.line_a.output + f.line_b.output + f.line_c.output = f.total_output →
  f.line_b.output = 5600 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_theorem_l2900_290051


namespace NUMINAMATH_CALUDE_sum_of_digits_7_pow_23_l2900_290000

/-- Returns the ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- Returns the tens digit of a natural number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- The sum of the tens digit and the ones digit of 7^23 is 7 -/
theorem sum_of_digits_7_pow_23 :
  tensDigit (7^23) + onesDigit (7^23) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_7_pow_23_l2900_290000


namespace NUMINAMATH_CALUDE_custom_operation_solution_l2900_290096

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem custom_operation_solution :
  ∀ x : ℝ, star 3 x = 31 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_solution_l2900_290096


namespace NUMINAMATH_CALUDE_conference_handshakes_count_l2900_290088

/-- The number of unique handshakes in a conference with specified conditions -/
def conferenceHandshakes (numCompanies : ℕ) (repsPerCompany : ℕ) : ℕ :=
  let totalPeople := numCompanies * repsPerCompany
  let handshakesPerPerson := totalPeople - repsPerCompany - 1
  (totalPeople * handshakesPerPerson) / 2

/-- Theorem: The number of handshakes in the specified conference is 250 -/
theorem conference_handshakes_count :
  conferenceHandshakes 5 5 = 250 := by
  sorry

#eval conferenceHandshakes 5 5

end NUMINAMATH_CALUDE_conference_handshakes_count_l2900_290088


namespace NUMINAMATH_CALUDE_max_value_of_f_l2900_290033

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a x ≤ f a y) ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, 3 ≤ f a x) →
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a y ≤ f a x ∧ f a x = 57 :=
by
  sorry


end NUMINAMATH_CALUDE_max_value_of_f_l2900_290033


namespace NUMINAMATH_CALUDE_mental_math_competition_l2900_290036

theorem mental_math_competition :
  ∃! (numbers : Finset ℕ),
    numbers.card = 4 ∧
    (∀ n ∈ numbers,
      ∃ (M m : ℕ),
        n = 15 * M + 11 * m ∧
        M > 1 ∧ m > 1 ∧
        Odd M ∧ Odd m ∧
        (∀ d : ℕ, d > 1 → Odd d → d ∣ n → m ≤ d ∧ d ≤ M) ∧
        numbers = {528, 880, 1232, 1936}) :=
by sorry

end NUMINAMATH_CALUDE_mental_math_competition_l2900_290036


namespace NUMINAMATH_CALUDE_system_two_solutions_l2900_290072

theorem system_two_solutions (a : ℝ) :
  (∃! x y, a^2 - 2*a*x - 6*y + x^2 + y^2 = 0 ∧ (|x| - 4)^2 + (|y| - 3)^2 = 25) ↔
  a ∈ Set.Ioo (-12) (-6) ∪ {0} ∪ Set.Ioo 6 12 :=
by sorry

end NUMINAMATH_CALUDE_system_two_solutions_l2900_290072


namespace NUMINAMATH_CALUDE_cylinder_not_identical_views_l2900_290020

-- Define the basic shapes
structure Shape :=
  (name : String)

-- Define the views
inductive View
  | Top
  | Front
  | Side

-- Define a function to get the shape of a view
def getViewShape (object : Shape) (view : View) : Shape :=
  sorry

-- Define the property of having identical views
def hasIdenticalViews (object : Shape) : Prop :=
  ∀ v1 v2 : View, getViewShape object v1 = getViewShape object v2

-- Define specific shapes
def cylinder : Shape :=
  { name := "Cylinder" }

def cube : Shape :=
  { name := "Cube" }

-- State the theorem
theorem cylinder_not_identical_views :
  ¬(hasIdenticalViews cylinder) ∧ hasIdenticalViews cube :=
sorry

end NUMINAMATH_CALUDE_cylinder_not_identical_views_l2900_290020


namespace NUMINAMATH_CALUDE_dog_walker_base_charge_l2900_290042

/-- Represents the earnings of a dog walker given their base charge per dog and walking durations. -/
def dog_walker_earnings (base_charge : ℝ) : ℝ :=
  (base_charge + 10 * 1) +  -- One dog for 10 minutes
  (2 * base_charge + 2 * 7 * 1) +  -- Two dogs for 7 minutes each
  (3 * base_charge + 3 * 9 * 1)  -- Three dogs for 9 minutes each

/-- Theorem stating that if a dog walker earns $171 with the given walking schedule, 
    their base charge per dog must be $20. -/
theorem dog_walker_base_charge : 
  ∃ (x : ℝ), dog_walker_earnings x = 171 → x = 20 :=
sorry

end NUMINAMATH_CALUDE_dog_walker_base_charge_l2900_290042


namespace NUMINAMATH_CALUDE_parabola_properties_l2900_290064

def parabola (x : ℝ) : ℝ := -3 * x^2

theorem parabola_properties :
  (∀ x : ℝ, parabola x ≤ parabola 0) ∧
  (parabola 0 = 0) ∧
  (∀ x y : ℝ, x > 0 → y > x → parabola y < parabola x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2900_290064


namespace NUMINAMATH_CALUDE_investment_growth_period_l2900_290052

/-- The annual interest rate as a real number between 0 and 1 -/
def interest_rate : ℝ := 0.341

/-- The target multiple of the initial investment -/
def target_multiple : ℝ := 3

/-- The function to calculate the investment value after n years -/
def investment_value (n : ℕ) : ℝ := (1 + interest_rate) ^ n

/-- The smallest investment period in years -/
def smallest_period : ℕ := 4

theorem investment_growth_period :
  (∀ k : ℕ, k < smallest_period → investment_value k ≤ target_multiple) ∧
  target_multiple < investment_value smallest_period :=
sorry

end NUMINAMATH_CALUDE_investment_growth_period_l2900_290052


namespace NUMINAMATH_CALUDE_solve_system_l2900_290037

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 17) 
  (eq2 : 6 * p + 5 * q = 20) : 
  q = 2 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l2900_290037


namespace NUMINAMATH_CALUDE_parabola_focus_on_line_l2900_290044

/-- The value of p for a parabola y^2 = 2px whose focus lies on 2x + y - 2 = 0 -/
theorem parabola_focus_on_line : ∃ (p : ℝ), 
  p > 0 ∧ 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ 2*x + y - 2 = 0 ∧ x = p/2 ∧ y = 0) →
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_on_line_l2900_290044


namespace NUMINAMATH_CALUDE_average_sales_per_month_l2900_290094

def sales_data : List ℕ := [100, 60, 40, 120]

theorem average_sales_per_month :
  (List.sum sales_data) / (List.length sales_data) = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_sales_per_month_l2900_290094


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_l2900_290050

theorem quadratic_roots_distance (m : ℝ) : 
  (∃ α β : ℂ, (α^2 - 2 * Real.sqrt 2 * α + m = 0) ∧ 
              (β^2 - 2 * Real.sqrt 2 * β + m = 0) ∧ 
              (Complex.abs (α - β) = 3)) → 
  m = 17/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distance_l2900_290050


namespace NUMINAMATH_CALUDE_binomial_10_3_l2900_290039

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2900_290039


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2900_290068

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 324 * π →
    (4 / 3) * π * r^3 = 972 * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2900_290068


namespace NUMINAMATH_CALUDE_equation_solutions_parabola_properties_l2900_290024

-- Part 1: Equation solving
def equation (x : ℝ) : Prop := (x - 9)^2 = 2 * (x - 9)

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 9 ∧ x₂ = 11 ∧ equation x₁ ∧ equation x₂ ∧
  ∀ (x : ℝ), equation x → x = x₁ ∨ x = x₂ :=
sorry

-- Part 2: Parabola function
def parabola (x y : ℝ) : Prop := y = -x^2 - 6*x - 7

theorem parabola_properties :
  (parabola (-3) 2) ∧ (parabola (-1) (-2)) ∧
  ∀ (x y : ℝ), y = -(x + 3)^2 + 2 ↔ parabola x y :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_parabola_properties_l2900_290024


namespace NUMINAMATH_CALUDE_system_solution_l2900_290066

theorem system_solution (a b c d e f : ℝ) 
  (eq1 : 4 * a = (b + c + d + e)^4)
  (eq2 : 4 * b = (c + d + e + f)^4)
  (eq3 : 4 * c = (d + e + f + a)^4)
  (eq4 : 4 * d = (e + f + a + b)^4)
  (eq5 : 4 * e = (f + a + b + c)^4)
  (eq6 : 4 * f = (a + b + c + d)^4) :
  a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4 ∧ e = 1/4 ∧ f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2900_290066


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_in_pyramid_inscribed_sphere_radius_is_correct_l2900_290026

/-- The radius of the sphere inscribed in a pyramid PMKC, where:
  - PABCD is a regular quadrilateral pyramid
  - PO is the height of the pyramid and equals 4
  - ABCD is the base of the pyramid with side length 6
  - M is the midpoint of BC
  - K is the midpoint of CD
-/
theorem inscribed_sphere_radius_in_pyramid (PO : ℝ) (side_length : ℝ) : ℝ :=
  let PMKC_volume := (1/8) * (1/3) * side_length^2 * PO
  let CMK_area := (1/4) * (1/2) * side_length^2
  let ON := (1/4) * side_length * Real.sqrt 2
  let PN := Real.sqrt ((PO^2) + (ON^2))
  let OK := (1/2) * side_length
  let PK := Real.sqrt ((PO^2) + (OK^2))
  let PKC_area := (1/2) * OK * PK
  let PMK_area := (1/2) * (side_length * Real.sqrt 2 / 2) * PN
  let surface_area := 2 * PKC_area + PMK_area + CMK_area
  let radius := 3 * PMKC_volume / surface_area
  12 / (13 + Real.sqrt 41)

theorem inscribed_sphere_radius_is_correct (PO : ℝ) (side_length : ℝ)
  (h1 : PO = 4)
  (h2 : side_length = 6) :
  inscribed_sphere_radius_in_pyramid PO side_length = 12 / (13 + Real.sqrt 41) := by
  sorry

#check inscribed_sphere_radius_is_correct

end NUMINAMATH_CALUDE_inscribed_sphere_radius_in_pyramid_inscribed_sphere_radius_is_correct_l2900_290026


namespace NUMINAMATH_CALUDE_trapezium_height_l2900_290084

theorem trapezium_height (a b area : ℝ) (ha : a = 20) (hb : b = 18) (harea : area = 475) :
  (2 * area) / (a + b) = 25 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l2900_290084


namespace NUMINAMATH_CALUDE_paul_school_supplies_l2900_290006

theorem paul_school_supplies : 
  let initial_regular_erasers : ℕ := 307
  let initial_jumbo_erasers : ℕ := 150
  let initial_standard_crayons : ℕ := 317
  let initial_jumbo_crayons : ℕ := 300
  let lost_regular_erasers : ℕ := 52
  let used_standard_crayons : ℕ := 123
  let used_jumbo_crayons : ℕ := 198

  let remaining_regular_erasers : ℕ := initial_regular_erasers - lost_regular_erasers
  let remaining_jumbo_erasers : ℕ := initial_jumbo_erasers
  let remaining_standard_crayons : ℕ := initial_standard_crayons - used_standard_crayons
  let remaining_jumbo_crayons : ℕ := initial_jumbo_crayons - used_jumbo_crayons

  let total_remaining_erasers : ℕ := remaining_regular_erasers + remaining_jumbo_erasers
  let total_remaining_crayons : ℕ := remaining_standard_crayons + remaining_jumbo_crayons

  (total_remaining_crayons : ℤ) - (total_remaining_erasers : ℤ) = -109
  := by sorry

end NUMINAMATH_CALUDE_paul_school_supplies_l2900_290006


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2900_290075

theorem largest_divisor_five_consecutive_integers : 
  ∃ (k : ℕ), k = 60 ∧ 
  (∀ (n : ℤ), ∃ (m : ℤ), (n * (n+1) * (n+2) * (n+3) * (n+4)) = m * k) ∧
  (∀ (l : ℕ), l > k → ∃ (n : ℤ), ∀ (m : ℤ), (n * (n+1) * (n+2) * (n+3) * (n+4)) ≠ m * l) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2900_290075


namespace NUMINAMATH_CALUDE_prime_power_cube_plus_one_l2900_290032

theorem prime_power_cube_plus_one (p : ℕ) (x y : ℕ+) (h_prime : Nat.Prime p) :
  p ^ (x : ℕ) = y ^ 3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_prime_power_cube_plus_one_l2900_290032


namespace NUMINAMATH_CALUDE_division_problem_l2900_290057

theorem division_problem (x : ℕ+) (y : ℚ) (m : ℤ) 
  (h1 : (x : ℚ) = 11 * y + 4)
  (h2 : (2 * x : ℚ) = 8 * m * y + 3)
  (h3 : 13 * y - x = 1) :
  m = 3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2900_290057


namespace NUMINAMATH_CALUDE_pizzas_served_during_lunch_l2900_290053

theorem pizzas_served_during_lunch (total_pizzas dinner_pizzas lunch_pizzas : ℕ) : 
  total_pizzas = 15 → 
  dinner_pizzas = 6 → 
  lunch_pizzas = total_pizzas - dinner_pizzas → 
  lunch_pizzas = 9 := by
sorry

end NUMINAMATH_CALUDE_pizzas_served_during_lunch_l2900_290053


namespace NUMINAMATH_CALUDE_area_Ω_bound_l2900_290070

/-- Parabola C: y = (1/2)x^2 -/
def parabola_C (x y : ℝ) : Prop := y = (1/2) * x^2

/-- Circle D: x^2 + (y - 1/2)^2 = r^2, where r > 0 -/
def circle_D (x y r : ℝ) : Prop := x^2 + (y - 1/2)^2 = r^2 ∧ r > 0

/-- C and D have no common points -/
def no_intersection (r : ℝ) : Prop := ∀ x y : ℝ, parabola_C x y → ¬(circle_D x y r)

/-- Point A is on parabola C -/
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.1 A.2

/-- Region Ω formed by tangents from A to D -/
def region_Ω (r : ℝ) (A : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Area of region Ω -/
noncomputable def area_Ω (r : ℝ) (A : ℝ × ℝ) : ℝ := sorry

theorem area_Ω_bound (r : ℝ) (h : no_intersection r) :
  ∀ A : ℝ × ℝ, point_on_parabola A →
    0 < area_Ω r A ∧ area_Ω r A < π/16 := by sorry

end NUMINAMATH_CALUDE_area_Ω_bound_l2900_290070


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2900_290077

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 140 → 
  b = 210 → 
  c^2 = a^2 + b^2 → 
  c = 70 * Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2900_290077


namespace NUMINAMATH_CALUDE_part1_part2_l2900_290090

-- Define the given condition
def condition (x y : ℝ) : Prop :=
  |x - 4 - 2 * Real.sqrt 2| + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0

-- Define a rhombus
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

-- Theorem for part 1
theorem part1 {x y : ℝ} (h : condition x y) :
  x * y^2 - x^2 * y = -32 * Real.sqrt 2 := by
  sorry

-- Theorem for part 2
theorem part2 {x y : ℝ} (h : condition x y) :
  let r : Rhombus := ⟨x, y⟩
  (r.diagonal1 * r.diagonal2 / 2 = 4) ∧
  (r.diagonal1 * r.diagonal2 / (4 * Real.sqrt 3) = 2 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_l2900_290090


namespace NUMINAMATH_CALUDE_intersection_M_N_l2900_290021

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2900_290021


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2900_290069

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2900_290069


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l2900_290013

theorem hyperbola_asymptote_a_value :
  ∀ (a : ℝ), a > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 → 
    ((3 * x + 2 * y = 0) ∨ (3 * x - 2 * y = 0))) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l2900_290013


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l2900_290027

/-- A rectangular game board -/
structure GameBoard where
  m : ℝ
  n : ℝ

/-- A penny with radius 1 -/
structure Penny where
  center : ℝ × ℝ

/-- The game state -/
structure GameState where
  board : GameBoard
  pennies : List Penny

/-- Check if a new penny placement is valid -/
def is_valid_placement (state : GameState) (new_penny : Penny) : Prop :=
  ∀ p ∈ state.pennies, (new_penny.center.1 - p.center.1)^2 + (new_penny.center.2 - p.center.2)^2 > 4

/-- The winning condition for the first player -/
def first_player_wins (board : GameBoard) : Prop :=
  board.m ≥ 2 ∧ board.n ≥ 2

/-- The main theorem -/
theorem first_player_winning_strategy (board : GameBoard) :
  first_player_wins board ↔ ∃ (strategy : GameState → Penny), 
    ∀ (game : GameState), game.board = board → 
      (is_valid_placement game (strategy game) → 
        ∀ (opponent_move : Penny), is_valid_placement (GameState.mk board (strategy game :: game.pennies)) opponent_move → 
          ∃ (next_move : Penny), is_valid_placement (GameState.mk board (opponent_move :: strategy game :: game.pennies)) next_move) :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l2900_290027


namespace NUMINAMATH_CALUDE_face_card_then_heart_probability_l2900_290019

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of face cards in a standard deck -/
def FaceCards : ℕ := 12

/-- Number of hearts in a standard deck -/
def Hearts : ℕ := 13

/-- Number of face cards that are hearts -/
def FaceHearts : ℕ := 3

/-- Probability of drawing a face card followed by a heart from a standard deck -/
theorem face_card_then_heart_probability :
  (FaceCards / StandardDeck) * (Hearts / (StandardDeck - 1)) = 19 / 210 :=
sorry

end NUMINAMATH_CALUDE_face_card_then_heart_probability_l2900_290019


namespace NUMINAMATH_CALUDE_solution_pairs_l2900_290035

theorem solution_pairs (x y : ℝ) : 
  (4 * x^2 - y^2)^2 + (7 * x + 3 * y - 39)^2 = 0 ↔ (x = 3 ∧ y = 6) ∨ (x = 39 ∧ y = -78) := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l2900_290035


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l2900_290015

def linear_function (k b x : ℝ) : ℝ := k * x + b

theorem fixed_point_of_linear_function (k b : ℝ) 
  (h : 3 * k - b = 2) : 
  linear_function k b (-3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l2900_290015


namespace NUMINAMATH_CALUDE_valid_speaking_orders_l2900_290049

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 7

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of special students (A and B) -/
def special_students : ℕ := 2

theorem valid_speaking_orders : 
  (choose special_students 1 * choose (total_students - special_students) (selected_students - 1) * arrange selected_students selected_students) +
  (choose special_students 2 * choose (total_students - special_students) (selected_students - 2) * arrange selected_students selected_students) -
  (choose special_students 2 * choose (total_students - special_students) (selected_students - 2) * arrange (selected_students - 1) (selected_students - 1) * arrange 2 2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_valid_speaking_orders_l2900_290049


namespace NUMINAMATH_CALUDE_product_63_57_l2900_290063

theorem product_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_product_63_57_l2900_290063


namespace NUMINAMATH_CALUDE_yankees_to_mets_ratio_l2900_290080

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The given conditions of the problem -/
def baseball_town_conditions (fans : FanCounts) : Prop :=
  fans.yankees + fans.mets + fans.red_sox = 360 ∧
  fans.mets = 96 ∧
  5 * fans.mets = 4 * fans.red_sox

/-- The theorem to be proved -/
theorem yankees_to_mets_ratio (fans : FanCounts) 
  (h : baseball_town_conditions fans) : 
  3 * fans.mets = 2 * fans.yankees :=
sorry

end NUMINAMATH_CALUDE_yankees_to_mets_ratio_l2900_290080


namespace NUMINAMATH_CALUDE_equation_solution_l2900_290078

theorem equation_solution : ∃ x : ℝ, 1 - 1 / ((1 - x)^3) = 1 / (1 - x) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2900_290078


namespace NUMINAMATH_CALUDE_max_volume_right_prism_l2900_290016

/-- Given a right prism with rectangular base (sides a and b) and height h, 
    where the sum of areas of two lateral faces and one base is 40,
    the maximum volume of the prism is 80√30/9 -/
theorem max_volume_right_prism (a b h : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : h > 0)
  (h₄ : a * h + b * h + a * b = 40) : 
  a * b * h ≤ 80 * Real.sqrt 30 / 9 := by
  sorry

#check max_volume_right_prism

end NUMINAMATH_CALUDE_max_volume_right_prism_l2900_290016


namespace NUMINAMATH_CALUDE_prob_same_color_is_one_twentieth_l2900_290056

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- Represents the number of girls selecting marbles -/
def number_of_girls : ℕ := 3

/-- Calculates the probability of all girls selecting the same colored marble -/
def prob_same_color : ℚ :=
  2 * (marbles_per_color.factorial / (marbles_per_color + number_of_girls).factorial)

/-- Theorem stating that the probability of all girls selecting the same colored marble is 1/20 -/
theorem prob_same_color_is_one_twentieth : prob_same_color = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_one_twentieth_l2900_290056
