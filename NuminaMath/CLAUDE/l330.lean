import Mathlib

namespace NUMINAMATH_CALUDE_largest_common_term_l330_33042

/-- The largest common term of two arithmetic sequences -/
theorem largest_common_term : ∃ (n m : ℕ), 
  138 = 2 + 4 * n ∧ 
  138 = 5 + 5 * m ∧ 
  138 ≤ 150 ∧ 
  ∀ (k l : ℕ), (2 + 4 * k = 5 + 5 * l) → (2 + 4 * k ≤ 150) → (2 + 4 * k ≤ 138) :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l330_33042


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l330_33009

theorem rectangular_garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 507 →
  width = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l330_33009


namespace NUMINAMATH_CALUDE_walnut_distribution_l330_33082

/-- The total number of walnuts -/
def total_walnuts : ℕ := 55

/-- The number of walnuts in the first pile -/
def first_pile : ℕ := 7

/-- The number of walnuts in each of the other piles -/
def other_piles : ℕ := 12

/-- The number of piles -/
def num_piles : ℕ := 5

theorem walnut_distribution :
  (num_piles - 1) * other_piles + first_pile = total_walnuts ∧
  ∃ (equal_walnuts : ℕ), equal_walnuts * num_piles = total_walnuts :=
by sorry

end NUMINAMATH_CALUDE_walnut_distribution_l330_33082


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_to_plane_l330_33079

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_parallel_perpendicular_to_plane 
  (a b : Line) (α : Plane) :
  parallel a b → perpendicular b α → perpendicular a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_to_plane_l330_33079


namespace NUMINAMATH_CALUDE_nectarines_per_box_l330_33084

theorem nectarines_per_box (num_crates : ℕ) (oranges_per_crate : ℕ) (num_boxes : ℕ) (total_fruits : ℕ) :
  num_crates = 12 →
  oranges_per_crate = 150 →
  num_boxes = 16 →
  total_fruits = 2280 →
  (total_fruits - num_crates * oranges_per_crate) / num_boxes = 30 :=
by sorry

end NUMINAMATH_CALUDE_nectarines_per_box_l330_33084


namespace NUMINAMATH_CALUDE_anchuria_laws_theorem_l330_33077

variables (K N M : ℕ) (p : ℝ)

/-- The probability that exactly M laws are included in the Concept -/
def prob_M_laws_included : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- The expected number of laws included in the Concept -/
def expected_laws_included : ℝ :=
  K * (1 - (1 - p)^N)

/-- Theorem stating the correctness of the probability and expectation calculations -/
theorem anchuria_laws_theorem (h1 : 0 ≤ p) (h2 : p ≤ 1) (h3 : M ≤ K) :
  (prob_M_laws_included K N M p = (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)) ∧
  (expected_laws_included K N p = K * (1 - (1 - p)^N)) := by
  sorry

end NUMINAMATH_CALUDE_anchuria_laws_theorem_l330_33077


namespace NUMINAMATH_CALUDE_point_coordinates_l330_33031

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the Cartesian coordinate system -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (p : Point),
    second_quadrant p →
    distance_to_x_axis p = 3 →
    distance_to_y_axis p = 7 →
    p = Point.mk (-7) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l330_33031


namespace NUMINAMATH_CALUDE_three_students_two_groups_l330_33015

/-- The number of ways for students to sign up for activity groups. -/
def signUpWays (numStudents : ℕ) (numGroups : ℕ) : ℕ :=
  numGroups ^ numStudents

/-- Theorem: Three students signing up for two groups results in 8 ways. -/
theorem three_students_two_groups :
  signUpWays 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_students_two_groups_l330_33015


namespace NUMINAMATH_CALUDE_half_sqrt_is_one_l330_33073

theorem half_sqrt_is_one (x : ℝ) : (1/2 : ℝ) * Real.sqrt x = 1 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_half_sqrt_is_one_l330_33073


namespace NUMINAMATH_CALUDE_expression_equals_101_15_closest_integer_is_6_l330_33019

-- Define the expression
def expression : ℚ := (4 * 10^150 + 4 * 10^152) / (3 * 10^151 + 3 * 10^151)

-- Theorem stating that the expression equals 101/15
theorem expression_equals_101_15 : expression = 101 / 15 := by sorry

-- Function to find the closest integer to a rational number
def closest_integer (q : ℚ) : ℤ := 
  ⌊q + 1/2⌋

-- Theorem stating that the closest integer to the expression is 6
theorem closest_integer_is_6 : closest_integer expression = 6 := by sorry

end NUMINAMATH_CALUDE_expression_equals_101_15_closest_integer_is_6_l330_33019


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_one_l330_33056

theorem sin_50_plus_sqrt3_tan_10_equals_one :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_one_l330_33056


namespace NUMINAMATH_CALUDE_corner_stationery_sales_proportion_l330_33037

theorem corner_stationery_sales_proportion :
  let total_sales_percent : ℝ := 100
  let markers_percent : ℝ := 25
  let notebooks_percent : ℝ := 47
  total_sales_percent - (markers_percent + notebooks_percent) = 28 := by
sorry

end NUMINAMATH_CALUDE_corner_stationery_sales_proportion_l330_33037


namespace NUMINAMATH_CALUDE_product_abcd_l330_33006

theorem product_abcd (a b c d : ℚ) : 
  (2 * a + 3 * b + 5 * c + 7 * d = 42) →
  (4 * (d + c) = b) →
  (2 * b + 2 * c = a) →
  (c - 2 = d) →
  (a * b * c * d = -26880 / 729) := by
sorry

end NUMINAMATH_CALUDE_product_abcd_l330_33006


namespace NUMINAMATH_CALUDE_sara_pumpkins_l330_33002

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The number of pumpkins Sara has left -/
def pumpkins_left : ℕ := 20

/-- The original number of pumpkins Sara grew -/
def original_pumpkins : ℕ := pumpkins_eaten + pumpkins_left

theorem sara_pumpkins : original_pumpkins = 43 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l330_33002


namespace NUMINAMATH_CALUDE_l_shaped_tiling_exists_l330_33040

/-- An L-shaped piece consisting of three squares -/
inductive LPiece
| mk : LPiece

/-- A square grid of side length 2^n -/
def Square (n : ℕ) := Fin (2^n) × Fin (2^n)

/-- A cell in the square grid -/
def Cell (n : ℕ) := Square n

/-- A tiling of the square grid using L-shaped pieces -/
def Tiling (n : ℕ) := Square n → Option LPiece

/-- Predicate to check if a tiling is valid -/
def is_valid_tiling (n : ℕ) (t : Tiling n) (removed : Cell n) : Prop :=
  ∀ (c : Cell n), c ≠ removed → ∃ (piece : LPiece), t c = some piece

/-- The main theorem: for all n, there exists a valid tiling of a 2^n x 2^n square
    with one cell removed using L-shaped pieces -/
theorem l_shaped_tiling_exists (n : ℕ) :
  ∀ (removed : Cell n), ∃ (t : Tiling n), is_valid_tiling n t removed :=
sorry

end NUMINAMATH_CALUDE_l_shaped_tiling_exists_l330_33040


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l330_33022

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 3 * n ≡ 1356 [ZMOD 22]) → n ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l330_33022


namespace NUMINAMATH_CALUDE_max_difference_of_five_integers_l330_33069

theorem max_difference_of_five_integers (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℝ) / 5 = 50 →
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e →
  e ≤ 58 →
  e - a ≤ 34 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_of_five_integers_l330_33069


namespace NUMINAMATH_CALUDE_f_k_even_iff_l330_33043

/-- The number of valid coloring schemes for n points on a circle with at least one red point in any k consecutive points. -/
def f_k (k n : ℕ) : ℕ := sorry

/-- Theorem stating the necessary and sufficient conditions for f_k(n) to be even. -/
theorem f_k_even_iff (n k : ℕ) (h1 : n > k) (h2 : k ≥ 2) :
  Even (f_k k n) ↔ Even k ∧ (k + 1 ∣ n) := by sorry

end NUMINAMATH_CALUDE_f_k_even_iff_l330_33043


namespace NUMINAMATH_CALUDE_age_problem_l330_33024

theorem age_problem (mehki_age jordyn_age certain_age : ℕ) : 
  mehki_age = jordyn_age + 10 →
  jordyn_age = 2 * certain_age →
  mehki_age = 22 →
  certain_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l330_33024


namespace NUMINAMATH_CALUDE_mod_equivalence_2023_l330_33058

theorem mod_equivalence_2023 : ∃! n : ℕ, n ≤ 6 ∧ n ≡ -2023 [ZMOD 7] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_2023_l330_33058


namespace NUMINAMATH_CALUDE_perimeter_of_square_arrangement_l330_33003

theorem perimeter_of_square_arrangement (total_area : ℝ) (num_squares : ℕ) 
  (arrangement_width : ℕ) (arrangement_height : ℕ) :
  total_area = 216 →
  num_squares = 6 →
  arrangement_width = 3 →
  arrangement_height = 2 →
  let square_area := total_area / num_squares
  let side_length := Real.sqrt square_area
  let perimeter := 2 * (arrangement_width + arrangement_height) * side_length
  perimeter = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_square_arrangement_l330_33003


namespace NUMINAMATH_CALUDE_smallest_cut_length_for_triangle_l330_33088

theorem smallest_cut_length_for_triangle (a b c : ℕ) (ha : a = 12) (hb : b = 18) (hc : c = 20) :
  ∃ (x : ℕ), x = 10 ∧
  (∀ (y : ℕ), y < x → (a - y) + (b - y) > (c - y)) ∧
  (a - x) + (b - x) ≤ (c - x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cut_length_for_triangle_l330_33088


namespace NUMINAMATH_CALUDE_inverse_sum_mod_11_l330_33076

theorem inverse_sum_mod_11 : 
  (((2⁻¹ : ZMod 11) + (6⁻¹ : ZMod 11) + (10⁻¹ : ZMod 11))⁻¹ : ZMod 11) = 8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_11_l330_33076


namespace NUMINAMATH_CALUDE_fraction_domain_l330_33028

theorem fraction_domain (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_fraction_domain_l330_33028


namespace NUMINAMATH_CALUDE_length_of_mn_l330_33067

/-- Given four collinear points A, B, C, D in order on a line,
    with M as the midpoint of AC and N as the midpoint of BD,
    prove that the length of MN is 24 when AD = 68 and BC = 20. -/
theorem length_of_mn (A B C D M N : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order
  (M = (A + C) / 2) →            -- M is midpoint of AC
  (N = (B + D) / 2) →            -- N is midpoint of BD
  (D - A = 68) →                 -- AD = 68
  (C - B = 20) →                 -- BC = 20
  (N - M = 24) :=                -- MN = 24
by sorry

end NUMINAMATH_CALUDE_length_of_mn_l330_33067


namespace NUMINAMATH_CALUDE_volleyball_matches_l330_33097

theorem volleyball_matches (a : ℕ) : 
  (3 / 5 : ℚ) * a = (11 / 20 : ℚ) * ((7 / 6 : ℚ) * a) → a = 24 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_matches_l330_33097


namespace NUMINAMATH_CALUDE_linear_function_monotonicity_linear_function_parity_linear_function_y_intercept_linear_function_x_intercept_l330_33047

-- Define a linear function
def linearFunction (a b x : ℝ) : ℝ := a * x + b

-- Theorem about monotonicity of linear functions
theorem linear_function_monotonicity (a b : ℝ) :
  (∀ x y : ℝ, x < y → linearFunction a b x < linearFunction a b y) ↔ a > 0 :=
sorry

-- Theorem about parity of linear functions
theorem linear_function_parity (a b : ℝ) :
  (∀ x : ℝ, linearFunction a b (-x) = -linearFunction a b x + 2*b) ↔ b = 0 :=
sorry

-- Theorem about y-intercept of linear functions
theorem linear_function_y_intercept (a b : ℝ) :
  linearFunction a b 0 = b :=
sorry

-- Theorem about x-intercept of linear functions (when it exists)
theorem linear_function_x_intercept (a b : ℝ) (h : a ≠ 0) :
  linearFunction a b (-b/a) = 0 :=
sorry

end NUMINAMATH_CALUDE_linear_function_monotonicity_linear_function_parity_linear_function_y_intercept_linear_function_x_intercept_l330_33047


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l330_33093

theorem arithmetic_progression_sum (a d : ℚ) :
  (let S₂₀ := 20 * (2 * a + 19 * d) / 2
   let S₅₀ := 50 * (2 * a + 49 * d) / 2
   let S₇₀ := 70 * (2 * a + 69 * d) / 2
   S₂₀ = 200 ∧ S₅₀ = 150) →
  S₇₀ = -350 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l330_33093


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l330_33051

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 + a 2 = 40) →
  (a 3 + a 4 = 60) →
  (a 7 + a 8 = 135) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l330_33051


namespace NUMINAMATH_CALUDE_steves_commute_l330_33055

/-- The distance from Steve's house to work -/
def distance : ℝ := by sorry

/-- Steve's speed on the way to work -/
def speed_to_work : ℝ := by sorry

/-- Steve's speed on the way back from work -/
def speed_from_work : ℝ := 14

/-- The total time Steve spends on the roads -/
def total_time : ℝ := 6

theorem steves_commute :
  (speed_from_work = 2 * speed_to_work) →
  (distance / speed_to_work + distance / speed_from_work = total_time) →
  distance = 28 := by sorry

end NUMINAMATH_CALUDE_steves_commute_l330_33055


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l330_33089

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   x₁ ≠ x₂ ∧
   x₁^2 + (m+2)*x₁ + m + 5 = 0 ∧
   x₂^2 + (m+2)*x₂ + m + 5 = 0) →
  -5 < m ∧ m ≤ -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l330_33089


namespace NUMINAMATH_CALUDE_domain_v_correct_l330_33039

/-- The domain of v(x, y) = 1/√(x + y) where x and y are real numbers -/
def domain_v : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 > -p.1}

/-- The function v(x, y) = 1/√(x + y) -/
noncomputable def v (p : ℝ × ℝ) : ℝ :=
  1 / Real.sqrt (p.1 + p.2)

theorem domain_v_correct :
  ∀ p : ℝ × ℝ, p ∈ domain_v ↔ ∃ z : ℝ, v p = z :=
by sorry

end NUMINAMATH_CALUDE_domain_v_correct_l330_33039


namespace NUMINAMATH_CALUDE_expansion_coefficient_implies_k_value_l330_33083

theorem expansion_coefficient_implies_k_value (k : ℕ+) :
  (15 * k ^ 4 : ℕ) < 120 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_implies_k_value_l330_33083


namespace NUMINAMATH_CALUDE_triangle_side_length_l330_33048

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l330_33048


namespace NUMINAMATH_CALUDE_number_equation_solution_l330_33081

theorem number_equation_solution : ∃ x : ℝ, (3 * x - 6 = 2 * x) ∧ (x = 6) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l330_33081


namespace NUMINAMATH_CALUDE_road_trip_distance_l330_33063

/-- Calculates the total distance of a road trip given specific conditions --/
theorem road_trip_distance 
  (speed : ℝ) 
  (break_duration : ℝ) 
  (time_between_breaks : ℝ) 
  (hotel_search_time : ℝ) 
  (total_trip_time : ℝ) 
  (h1 : speed = 62) 
  (h2 : break_duration = 0.5) 
  (h3 : time_between_breaks = 5) 
  (h4 : hotel_search_time = 0.5) 
  (h5 : total_trip_time = 50) :
  ∃ (distance : ℝ), distance = 2790 := by
  sorry

#check road_trip_distance

end NUMINAMATH_CALUDE_road_trip_distance_l330_33063


namespace NUMINAMATH_CALUDE_senior_field_trip_l330_33034

theorem senior_field_trip :
  ∃! n : ℕ, n < 300 ∧ n % 17 = 15 ∧ n % 19 = 12 ∧ n = 202 := by
  sorry

end NUMINAMATH_CALUDE_senior_field_trip_l330_33034


namespace NUMINAMATH_CALUDE_anya_lost_games_l330_33029

/-- Represents a girl playing table tennis -/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents the state of a girl (playing or resting) -/
inductive State
| Playing
| Resting

/-- The number of games each girl played -/
def games_played (g : Girl) : Nat :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- The total number of games played -/
def total_games : Nat := 19

/-- Predicate to check if a girl lost a specific game -/
def lost_game (g : Girl) (game_number : Nat) : Prop := sorry

/-- Theorem stating that Anya lost in games 4, 8, 12, and 16 -/
theorem anya_lost_games :
  lost_game Girl.Anya 4 ∧
  lost_game Girl.Anya 8 ∧
  lost_game Girl.Anya 12 ∧
  lost_game Girl.Anya 16 :=
by sorry

end NUMINAMATH_CALUDE_anya_lost_games_l330_33029


namespace NUMINAMATH_CALUDE_intersection_equality_l330_33044

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.cos (Real.arccos p.1)}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.arccos (Real.cos p.1)}

-- Define the intersection set
def intersection_set : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 ∧ -1 ≤ p.1 ∧ p.1 ≤ 1}

-- Theorem statement
theorem intersection_equality : A ∩ B = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l330_33044


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l330_33036

theorem nested_fraction_evaluation :
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l330_33036


namespace NUMINAMATH_CALUDE_toris_initial_height_l330_33000

/-- Given Tori's growth and current height, prove her initial height --/
theorem toris_initial_height (growth : ℝ) (current_height : ℝ) 
  (h1 : growth = 2.86)
  (h2 : current_height = 7.26) :
  current_height - growth = 4.40 := by
  sorry

end NUMINAMATH_CALUDE_toris_initial_height_l330_33000


namespace NUMINAMATH_CALUDE_monday_temp_value_l330_33060

/-- The average temperature for a week -/
def average_temp : ℝ := 99

/-- The number of days in a week -/
def num_days : ℕ := 7

/-- The temperatures for 6 days of the week -/
def known_temps : List ℝ := [99.1, 98.7, 99.3, 99.8, 99, 98.9]

/-- The temperature on Monday -/
def monday_temp : ℝ := num_days * average_temp - known_temps.sum

theorem monday_temp_value : monday_temp = 98.2 := by sorry

end NUMINAMATH_CALUDE_monday_temp_value_l330_33060


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_school_l330_33075

/-- Represents the number of students in a school -/
structure School :=
  (students : ℕ)

/-- Represents a sampling strategy -/
structure Sampling :=
  (total_population : ℕ)
  (sample_size : ℕ)
  (schools : Vector School 3)

/-- Checks if the number of students in schools forms an arithmetic sequence -/
def is_arithmetic_sequence (schools : Vector School 3) : Prop :=
  schools[1].students - schools[0].students = schools[2].students - schools[1].students

/-- Theorem: In a stratified sampling of 120 students from 1500 students 
    distributed in an arithmetic sequence across 3 schools, 
    the number of students sampled from the middle school (B) is 40 -/
theorem stratified_sampling_middle_school 
  (sampling : Sampling) 
  (h1 : sampling.total_population = 1500)
  (h2 : sampling.sample_size = 120)
  (h3 : is_arithmetic_sequence sampling.schools)
  : (sampling.sample_size / 3 : ℕ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_school_l330_33075


namespace NUMINAMATH_CALUDE_shenille_score_l330_33049

/-- Represents the number of points Shenille scored in a basketball game -/
def points_scored (three_point_attempts : ℕ) (two_point_attempts : ℕ) : ℝ :=
  (0.6 : ℝ) * three_point_attempts + (0.6 : ℝ) * two_point_attempts

/-- Theorem stating that Shenille scored 18 points given the conditions -/
theorem shenille_score :
  ∀ x y : ℕ,
  x + y = 30 →
  points_scored x y = 18 :=
by sorry

end NUMINAMATH_CALUDE_shenille_score_l330_33049


namespace NUMINAMATH_CALUDE_divisibility_statements_l330_33065

theorem divisibility_statements : 
  (∃ n : ℤ, 25 = 5 * n) ∧ 
  (∃ n : ℤ, 209 = 19 * n) ∧ 
  ¬(∃ n : ℤ, 63 = 19 * n) ∧
  (∃ n : ℤ, 140 = 7 * n) ∧
  (∃ n : ℤ, 90 = 30 * n) ∧
  (∃ n : ℤ, 34 = 17 * n) ∧
  (∃ n : ℤ, 68 = 17 * n) := by
  sorry

#check divisibility_statements

end NUMINAMATH_CALUDE_divisibility_statements_l330_33065


namespace NUMINAMATH_CALUDE_hyperbola_point_outside_circle_l330_33074

theorem hyperbola_point_outside_circle 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (focus : c > 0)
  (eccentricity : c / a = 2)
  (x₁ x₂ : ℝ)
  (roots : a * x₁^2 + b * x₁ - c = 0 ∧ a * x₂^2 + b * x₂ - c = 0) :
  x₁^2 + x₂^2 > 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_point_outside_circle_l330_33074


namespace NUMINAMATH_CALUDE_jackson_deduction_l330_33064

/-- Calculates the deduction in cents given an hourly wage in dollars and a deduction rate. -/
def calculate_deduction (hourly_wage : ℚ) (deduction_rate : ℚ) : ℚ :=
  hourly_wage * 100 * deduction_rate

theorem jackson_deduction :
  let hourly_wage : ℚ := 25
  let deduction_rate : ℚ := 25 / 1000  -- 2.5% expressed as a rational number
  calculate_deduction hourly_wage deduction_rate = 62.5 := by
  sorry

#eval calculate_deduction 25 (25/1000)

end NUMINAMATH_CALUDE_jackson_deduction_l330_33064


namespace NUMINAMATH_CALUDE_chef_dinner_meals_l330_33099

/-- Calculates the number of meals prepared for dinner given lunch and dinner information -/
def meals_prepared_for_dinner (lunch_prepared : ℕ) (lunch_sold : ℕ) (dinner_total : ℕ) : ℕ :=
  dinner_total - (lunch_prepared - lunch_sold)

/-- Proves that the chef prepared 5 meals for dinner -/
theorem chef_dinner_meals :
  meals_prepared_for_dinner 17 12 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chef_dinner_meals_l330_33099


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l330_33030

/-- Given a quadratic equation 10x^2 + 15x - 25 = 0, 
    the sum of the squares of its roots is equal to 29/4 -/
theorem sum_of_squares_of_roots : 
  let a : ℚ := 10
  let b : ℚ := 15
  let c : ℚ := -25
  let x₁ : ℚ := (-b + (b^2 - 4*a*c).sqrt) / (2*a)
  let x₂ : ℚ := (-b - (b^2 - 4*a*c).sqrt) / (2*a)
  x₁^2 + x₂^2 = 29/4 := by
sorry


end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l330_33030


namespace NUMINAMATH_CALUDE_tomato_suggestion_count_l330_33008

theorem tomato_suggestion_count (total students_potatoes students_bacon : ℕ) 
  (h1 : total = 826)
  (h2 : students_potatoes = 324)
  (h3 : students_bacon = 374) :
  total - (students_potatoes + students_bacon) = 128 := by
sorry

end NUMINAMATH_CALUDE_tomato_suggestion_count_l330_33008


namespace NUMINAMATH_CALUDE_triangle_side_range_l330_33025

/-- Given a triangle ABC with side lengths a, b, and c, prove that if |a+b-6|+(a-b+4)^2=0, then 4 < c < 6 -/
theorem triangle_side_range (a b c : ℝ) (h : |a+b-6|+(a-b+4)^2=0) : 4 < c ∧ c < 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_range_l330_33025


namespace NUMINAMATH_CALUDE_fraction_problem_l330_33070

theorem fraction_problem (a b m : ℚ) : 
  (2 * (1/2) - b = 0) →  -- Fraction is undefined when x = 0.5
  ((-2 + a) / (2 * (-2) - b) = 0) →  -- Fraction equals 0 when x = -2
  ((m + a) / (2 * m - b) = 1) →  -- Fraction equals 1 when x = m
  m = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l330_33070


namespace NUMINAMATH_CALUDE_mapping_result_l330_33012

-- Define the set A (and B) as pairs of real numbers
def A : Type := ℝ × ℝ

-- Define the mapping f
def f (p : A) : A :=
  let (x, y) := p
  (x - y, x + y)

-- Theorem statement
theorem mapping_result : f (-1, 2) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_mapping_result_l330_33012


namespace NUMINAMATH_CALUDE_f_neg_a_eq_zero_l330_33072

noncomputable def f (x : ℝ) : ℝ := x * Real.log (Real.exp (2 * x) + 1) - x^2 + 1

theorem f_neg_a_eq_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_a_eq_zero_l330_33072


namespace NUMINAMATH_CALUDE_consecutive_integer_averages_l330_33023

theorem consecutive_integer_averages (a b : ℤ) (h_positive : a > 0) : 
  ((7 * a + 21) / 7 = b) → 
  ((7 * b + 21) / 7 = a + 6) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integer_averages_l330_33023


namespace NUMINAMATH_CALUDE_set_operation_equality_l330_33090

def U : Finset Nat := {1,2,3,4,5,6,7}
def A : Finset Nat := {2,4,5,7}
def B : Finset Nat := {3,4,5}

theorem set_operation_equality : (U \ A) ∪ (U \ B) = {1,2,3,6,7} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_equality_l330_33090


namespace NUMINAMATH_CALUDE_max_value_of_a_l330_33014

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem max_value_of_a :
  (∀ x : ℝ, determinant (x - 1) (a - 2) (a + 1) x ≥ 1) →
  a ≤ 3/2 ∧ ∃ a₀ : ℝ, a₀ ≤ 3/2 ∧ ∀ x : ℝ, determinant (x - 1) (a₀ - 2) (a₀ + 1) x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l330_33014


namespace NUMINAMATH_CALUDE_car_speed_problem_l330_33026

/-- Proves that given a car traveling 75% of a trip at 50 mph and the remaining 25% at speed s,
    if the average speed for the entire trip is 50 mph, then s must equal 50 mph. -/
theorem car_speed_problem (D : ℝ) (s : ℝ) (h1 : D > 0) (h2 : s > 0) : 
  (0.75 * D / 50 + 0.25 * D / s) = D / 50 → s = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l330_33026


namespace NUMINAMATH_CALUDE_three_by_three_min_cuts_l330_33092

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a straight-line cut on the grid -/
inductive Cut
  | Vertical : ℕ → Cut
  | Horizontal : ℕ → Cut

/-- Defines the minimum number of cuts required to divide a grid into unit squares -/
def min_cuts (g : Grid) : ℕ := sorry

/-- Theorem stating that a 3x3 grid requires exactly 4 cuts to be divided into unit squares -/
theorem three_by_three_min_cuts :
  ∀ (g : Grid), g.size = 3 → min_cuts g = 4 := by sorry

end NUMINAMATH_CALUDE_three_by_three_min_cuts_l330_33092


namespace NUMINAMATH_CALUDE_function_range_in_unit_interval_l330_33071

theorem function_range_in_unit_interval (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ z : ℝ, 0 ≤ f z ∧ f z ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_in_unit_interval_l330_33071


namespace NUMINAMATH_CALUDE_triangle_properties_l330_33033

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about properties of an acute triangle -/
theorem triangle_properties (t : Triangle) 
  (acute : t.A > 0 ∧ t.A < π ∧ t.B > 0 ∧ t.B < π ∧ t.C > 0 ∧ t.C < π)
  (m : ℝ × ℝ) (n : ℝ × ℝ)
  (h_m : m = (Real.sqrt 3, 2 * Real.sin t.A))
  (h_n : n = (t.c, t.a))
  (h_parallel : ∃ (k : ℝ), m.1 * n.2 = k * m.2 * n.1)
  (h_c : t.c = Real.sqrt 7)
  (h_area : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3 / 2) :
  t.C = π/3 ∧ t.a + t.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l330_33033


namespace NUMINAMATH_CALUDE_absolute_value_and_exponents_l330_33001

theorem absolute_value_and_exponents : |-3| + 2^2 - (Real.sqrt 3 - 1)^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponents_l330_33001


namespace NUMINAMATH_CALUDE_f_at_negative_one_l330_33045

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 150*x + c

-- State the theorem
theorem f_at_negative_one (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c (-1) = 3733.25 := by
sorry


end NUMINAMATH_CALUDE_f_at_negative_one_l330_33045


namespace NUMINAMATH_CALUDE_topological_minor_theorem_l330_33050

-- Define the average degree of a graph
def average_degree (G : Graph) : ℝ := sorry

-- Define what it means for a graph to contain another graph as a topological minor
def contains_topological_minor (G H : Graph) : Prop := sorry

-- Define the complete graph on r vertices
def complete_graph (r : ℕ) : Graph := sorry

theorem topological_minor_theorem :
  ∃ (c : ℝ), c = 10 ∧
  ∀ (r : ℕ) (G : Graph),
    average_degree G ≥ c * r^2 →
    contains_topological_minor G (complete_graph r) :=
sorry

end NUMINAMATH_CALUDE_topological_minor_theorem_l330_33050


namespace NUMINAMATH_CALUDE_f_equals_g_l330_33066

theorem f_equals_g (f g : Nat → Nat)
  (h1 : ∀ n : Nat, n > 0 → f (g n) = f n + 1)
  (h2 : ∀ n : Nat, n > 0 → g (f n) = g n + 1) :
  ∀ n : Nat, n > 0 → f n = g n :=
by sorry

end NUMINAMATH_CALUDE_f_equals_g_l330_33066


namespace NUMINAMATH_CALUDE_negation_of_implication_l330_33005

theorem negation_of_implication (A B : Set α) (a b : α) :
  ¬(a ∉ A → b ∈ B) ↔ (a ∉ A ∧ b ∉ B) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l330_33005


namespace NUMINAMATH_CALUDE_soccer_team_combinations_l330_33057

theorem soccer_team_combinations (n : ℕ) (k : ℕ) (h1 : n = 16) (h2 : k = 7) :
  Nat.choose n k = 11440 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_combinations_l330_33057


namespace NUMINAMATH_CALUDE_max_servings_is_56_l330_33054

/-- Represents the recipe requirements for one serving of salad -/
structure Recipe where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the warehouse -/
structure Warehouse where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of servings that can be made -/
def max_servings (recipe : Recipe) (warehouse : Warehouse) : ℕ :=
  min
    (warehouse.cucumbers / recipe.cucumbers)
    (min
      (warehouse.tomatoes / recipe.tomatoes)
      (min
        (warehouse.brynza / recipe.brynza)
        (warehouse.peppers / recipe.peppers)))

/-- Theorem: The maximum number of servings that can be made is 56 -/
theorem max_servings_is_56 :
  let recipe := Recipe.mk 2 2 75 1
  let warehouse := Warehouse.mk 117 116 4200 60
  max_servings recipe warehouse = 56 := by
  sorry

#eval max_servings (Recipe.mk 2 2 75 1) (Warehouse.mk 117 116 4200 60)

end NUMINAMATH_CALUDE_max_servings_is_56_l330_33054


namespace NUMINAMATH_CALUDE_population_trend_decreasing_l330_33053

theorem population_trend_decreasing 
  (k : ℝ) 
  (h1 : -1 < k) 
  (h2 : k < 0) 
  (P : ℝ) 
  (hP : P > 0) :
  ∀ n : ℕ, ∀ m : ℕ, n < m → P * (1 + k)^n > P * (1 + k)^m :=
sorry

end NUMINAMATH_CALUDE_population_trend_decreasing_l330_33053


namespace NUMINAMATH_CALUDE_base_nine_to_ten_l330_33021

theorem base_nine_to_ten : 
  (3 * 9^4 + 9 * 9^3 + 4 * 9^2 + 5 * 9^1 + 7 * 9^0) = 26620 := by
  sorry

end NUMINAMATH_CALUDE_base_nine_to_ten_l330_33021


namespace NUMINAMATH_CALUDE_hyperbola_properties_l330_33068

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the asymptote equation
def is_asymptote (m : ℝ) : Prop := ∀ x y : ℝ, hyperbola x y → (y = m * x ∨ y = -m * x)

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := ∃ a c : ℝ, a > 0 ∧ c > 0 ∧ e = c / a ∧ ∀ x y : ℝ, hyperbola x y → x^2 / a^2 - y^2 / (c^2 - a^2) = 1

-- Theorem statement
theorem hyperbola_properties : is_asymptote (Real.sqrt 2) ∧ eccentricity (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l330_33068


namespace NUMINAMATH_CALUDE_sum_of_integers_l330_33086

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 4)
  (eq4 : d - a + b = 3)
  (eq5 : a + b + c - d = 10) : 
  a + b + c + d = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l330_33086


namespace NUMINAMATH_CALUDE_mutually_exclusive_necessary_not_sufficient_l330_33094

open Set

universe u

variable {Ω : Type u} [MeasurableSpace Ω]
variable (A₁ A₂ : Set Ω)

def mutually_exclusive (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅

def complementary (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅ ∧ A₁ ∪ A₂ = univ

theorem mutually_exclusive_necessary_not_sufficient :
  (complementary A₁ A₂ → mutually_exclusive A₁ A₂) ∧
  ¬(mutually_exclusive A₁ A₂ → complementary A₁ A₂) := by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_necessary_not_sufficient_l330_33094


namespace NUMINAMATH_CALUDE_william_napkins_before_l330_33061

/-- The number of napkins William had before receiving napkins from Olivia and Amelia. -/
def napkins_before : ℕ := sorry

/-- The number of napkins Olivia gave to William. -/
def olivia_napkins : ℕ := 10

/-- The number of napkins Amelia gave to William. -/
def amelia_napkins : ℕ := 2 * olivia_napkins

/-- The total number of napkins William has now. -/
def total_napkins : ℕ := 45

theorem william_napkins_before :
  napkins_before = total_napkins - (olivia_napkins + amelia_napkins) :=
by sorry

end NUMINAMATH_CALUDE_william_napkins_before_l330_33061


namespace NUMINAMATH_CALUDE_final_amount_in_euros_l330_33095

/-- Represents the number of coins of each type -/
structure CoinCollection where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ
  half_dollars : ℕ
  one_dollar_coins : ℕ

/-- Calculates the total value of a coin collection in dollars -/
def collection_value (c : CoinCollection) : ℚ :=
  c.quarters * (1/4) + c.dimes * (1/10) + c.nickels * (1/20) + 
  c.pennies * (1/100) + c.half_dollars * (1/2) + c.one_dollar_coins

/-- Rob's initial coin collection -/
def initial_collection : CoinCollection := {
  quarters := 7,
  dimes := 3,
  nickels := 5,
  pennies := 12,
  half_dollars := 3,
  one_dollar_coins := 2
}

/-- Removes one coin of each type from the collection -/
def remove_one_each (c : CoinCollection) : CoinCollection := {
  quarters := c.quarters - 1,
  dimes := c.dimes - 1,
  nickels := c.nickels - 1,
  pennies := c.pennies - 1,
  half_dollars := c.half_dollars - 1,
  one_dollar_coins := c.one_dollar_coins - 1
}

/-- Exchanges three nickels for two dimes -/
def exchange_nickels_for_dimes (c : CoinCollection) : CoinCollection := {
  c with
  nickels := c.nickels - 3,
  dimes := c.dimes + 2
}

/-- Exchanges a half-dollar for a quarter and two dimes -/
def exchange_half_dollar (c : CoinCollection) : CoinCollection := {
  c with
  half_dollars := c.half_dollars - 1,
  quarters := c.quarters + 1,
  dimes := c.dimes + 2
}

/-- Exchanges a one-dollar coin for fifty pennies -/
def exchange_dollar_for_pennies (c : CoinCollection) : CoinCollection := {
  c with
  one_dollar_coins := c.one_dollar_coins - 1,
  pennies := c.pennies + 50
}

/-- Converts dollars to euros -/
def dollars_to_euros (dollars : ℚ) : ℚ :=
  dollars * (85/100)

/-- The main theorem stating the final amount in euros -/
theorem final_amount_in_euros : 
  dollars_to_euros (collection_value (
    exchange_dollar_for_pennies (
      exchange_half_dollar (
        exchange_nickels_for_dimes (
          remove_one_each initial_collection
        )
      )
    )
  )) = 2.9835 := by
  sorry


end NUMINAMATH_CALUDE_final_amount_in_euros_l330_33095


namespace NUMINAMATH_CALUDE_even_function_implies_a_value_l330_33011

def f (x a : ℝ) : ℝ := (x + 1) * (2 * x + 3 * a)

theorem even_function_implies_a_value :
  (∀ x, f x a = f (-x) a) → a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_value_l330_33011


namespace NUMINAMATH_CALUDE_system_solution_and_range_l330_33041

theorem system_solution_and_range (a x y : ℝ) : 
  (2 * x + y = 5 * a ∧ x - 3 * y = -a + 7) →
  (x = 2 * a + 1 ∧ y = a - 2) ∧
  (x ≥ 0 ∧ y < 0 ↔ -1/2 ≤ a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_and_range_l330_33041


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l330_33010

/-- Represents a rectangle with side lengths x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- Represents the configuration of rectangles and squares -/
structure CrossConfiguration where
  inner_square_side : ℝ
  outer_square_side : ℝ
  rectangle : Rectangle

/-- The cross configuration satisfies the given conditions -/
def valid_configuration (c : CrossConfiguration) : Prop :=
  c.outer_square_side = 3 * c.inner_square_side ∧
  c.rectangle.y = c.inner_square_side ∧
  c.rectangle.x + c.inner_square_side = c.outer_square_side

/-- The theorem stating the ratio of rectangle sides -/
theorem rectangle_side_ratio (c : CrossConfiguration) 
  (h : valid_configuration c) : 
  c.rectangle.x / c.rectangle.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l330_33010


namespace NUMINAMATH_CALUDE_share_of_a_l330_33080

theorem share_of_a (total : ℝ) (a b c : ℝ) : 
  total = 500 →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a = 125 := by
sorry

end NUMINAMATH_CALUDE_share_of_a_l330_33080


namespace NUMINAMATH_CALUDE_consecutive_circle_selections_l330_33032

/-- Represents the arrangement of circles in the figure -/
structure CircleArrangement where
  total_circles : Nat
  long_side_rows : Nat
  long_side_ways : Nat
  diagonal_ways : Nat

/-- The specific arrangement for our problem -/
def problem_arrangement : CircleArrangement :=
  { total_circles := 33
  , long_side_rows := 6
  , long_side_ways := 21
  , diagonal_ways := 18 }

/-- Calculates the total number of ways to select three consecutive circles -/
def count_consecutive_selections (arr : CircleArrangement) : Nat :=
  arr.long_side_ways + 2 * arr.diagonal_ways

/-- Theorem stating that there are 57 ways to select three consecutive circles -/
theorem consecutive_circle_selections :
  count_consecutive_selections problem_arrangement = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_circle_selections_l330_33032


namespace NUMINAMATH_CALUDE_apple_sales_loss_percentage_l330_33013

/-- Represents the shopkeeper's apple sales scenario -/
structure AppleSales where
  total_apples : ℝ
  sale_percentages : Fin 4 → ℝ
  profit_percentages : Fin 4 → ℝ
  unsold_percentage : ℝ
  storage_cost : ℝ
  packaging_cost : ℝ
  transportation_cost : ℝ

/-- Calculates the effective loss percentage for the given apple sales scenario -/
def effective_loss_percentage (sales : AppleSales) : ℝ :=
  sorry

/-- The given apple sales scenario -/
def given_scenario : AppleSales :=
  { total_apples := 150,
    sale_percentages := ![0.30, 0.25, 0.15, 0.10],
    profit_percentages := ![0.20, 0.30, 0.40, 0.35],
    unsold_percentage := 0.20,
    storage_cost := 15,
    packaging_cost := 10,
    transportation_cost := 25 }

/-- Theorem stating that the effective loss percentage for the given scenario is approximately 32.83% -/
theorem apple_sales_loss_percentage :
  abs (effective_loss_percentage given_scenario - 32.83) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_apple_sales_loss_percentage_l330_33013


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_10080_l330_33085

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem least_n_factorial_divisible_by_10080 :
  ∀ n : ℕ, n > 0 → (is_divisible (factorial n) 10080 → n ≥ 7) ∧
  (is_divisible (factorial 7) 10080) :=
sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_10080_l330_33085


namespace NUMINAMATH_CALUDE_cube_root_of_negative_27_l330_33004

theorem cube_root_of_negative_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_27_l330_33004


namespace NUMINAMATH_CALUDE_joshes_investment_l330_33096

/-- Proves that the initial investment is $2000 given the conditions of Josh's investment scenario -/
theorem joshes_investment
  (initial_wallet : ℝ)
  (final_wallet : ℝ)
  (stock_increase : ℝ)
  (h1 : initial_wallet = 300)
  (h2 : final_wallet = 2900)
  (h3 : stock_increase = 0.3)
  : ∃ (investment : ℝ), 
    investment = 2000 ∧ 
    final_wallet = initial_wallet + investment * (1 + stock_increase) :=
by sorry

end NUMINAMATH_CALUDE_joshes_investment_l330_33096


namespace NUMINAMATH_CALUDE_unique_base_solution_l330_33078

def base_to_decimal (n : ℕ) (b : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * b^2 + tens * b + ones

theorem unique_base_solution :
  ∃! (c : ℕ), c > 0 ∧ base_to_decimal 243 c + base_to_decimal 156 c = base_to_decimal 421 c :=
by sorry

end NUMINAMATH_CALUDE_unique_base_solution_l330_33078


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l330_33087

theorem modulus_of_complex_number :
  let z : ℂ := -1 + 3 * Complex.I
  Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l330_33087


namespace NUMINAMATH_CALUDE_polynomial_remainder_l330_33052

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^3 - 18 * x^2 + 24 * x - 26) % (4 * x - 8) = 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l330_33052


namespace NUMINAMATH_CALUDE_rabbit_population_estimate_l330_33007

/-- Calculates the approximate number of rabbits in a forest using the capture-recapture method. -/
def estimate_rabbit_population (initial_tagged : ℕ) (recaptured : ℕ) (tagged_in_recapture : ℕ) : ℕ :=
  (initial_tagged * recaptured) / tagged_in_recapture

/-- The approximate number of rabbits in the forest is 50. -/
theorem rabbit_population_estimate :
  let initial_tagged : ℕ := 10
  let recaptured : ℕ := 10
  let tagged_in_recapture : ℕ := 2
  estimate_rabbit_population initial_tagged recaptured tagged_in_recapture = 50 := by
  sorry

#eval estimate_rabbit_population 10 10 2

end NUMINAMATH_CALUDE_rabbit_population_estimate_l330_33007


namespace NUMINAMATH_CALUDE_cubic_derivative_equality_l330_33046

theorem cubic_derivative_equality (f : ℝ → ℝ) (x : ℝ) :
  (f = fun x ↦ x^3) →
  (deriv f x = 3) →
  (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_derivative_equality_l330_33046


namespace NUMINAMATH_CALUDE_total_money_value_l330_33017

def gold_value : ℕ := 75
def silver_value : ℕ := 40
def bronze_value : ℕ := 20
def titanium_value : ℕ := 10

def gold_count : ℕ := 6
def silver_count : ℕ := 8
def bronze_count : ℕ := 10
def titanium_count : ℕ := 4

def cash : ℕ := 45

theorem total_money_value :
  gold_value * gold_count +
  silver_value * silver_count +
  bronze_value * bronze_count +
  titanium_value * titanium_count +
  cash = 1055 := by sorry

end NUMINAMATH_CALUDE_total_money_value_l330_33017


namespace NUMINAMATH_CALUDE_first_triangle_isosceles_l330_33059

theorem first_triangle_isosceles (α β γ δ ε : ℝ) :
  α + β + γ = 180 →
  α + β = δ →
  β + γ = ε →
  0 < α ∧ 0 < β ∧ 0 < γ →
  0 < δ ∧ 0 < ε →
  ∃ (θ : ℝ), (α = θ ∧ γ = θ) ∨ (α = θ ∧ β = θ) ∨ (β = θ ∧ γ = θ) :=
by sorry

end NUMINAMATH_CALUDE_first_triangle_isosceles_l330_33059


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l330_33062

theorem intersection_points_theorem :
  let roots : Set ℝ := {1, 2}
  let eq_A : ℝ → ℝ → Prop := λ x y ↦ (y = x^2 ∨ y = 3*x)
  let eq_B : ℝ → ℝ → Prop := λ x y ↦ (y = x^2 - 3*x + 2 ∨ y = 2)
  let eq_C : ℝ → ℝ → Prop := λ x y ↦ (y = x ∨ y = x - 2)
  let eq_D : ℝ → ℝ → Prop := λ x y ↦ (y = x^2 - 3*x + 3 ∨ y = 3)
  (∀ x y, eq_A x y → x ∉ roots) ∧
  (∀ x y, eq_B x y → x ∉ roots) ∧
  (¬∃ x y, eq_C x y) ∧
  (∀ x y, eq_D x y → x ∉ roots) := by
  sorry


end NUMINAMATH_CALUDE_intersection_points_theorem_l330_33062


namespace NUMINAMATH_CALUDE_circle_standard_equation_l330_33091

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points A and B
def A : ℝ × ℝ := (1, -5)
def B : ℝ × ℝ := (2, -2)

-- Define the line equation
def line_equation (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0

-- Define the circle equation
def circle_equation (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_standard_equation :
  ∃ (c : Circle),
    circle_equation c A ∧
    circle_equation c B ∧
    line_equation c.center ∧
    c.center = (-3, -2) ∧
    c.radius = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_standard_equation_l330_33091


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l330_33027

-- Problem 1
theorem problem_1 : (1) - 1^2 + 16 / (-4)^2 * (-3 - 1) = -5 := by sorry

-- Problem 2
theorem problem_2 : 5 * (5/8) - 2 * (-5/8) - 7 * (5/8) = 0 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) : x - 3*y - (-3*x + 4*y) = 4*x - 7*y := by sorry

-- Problem 4
theorem problem_4 (a b : ℝ) : 3*a - 4*(a - 3/2*b) - 2*(4*b + 5*a) = -11*a - 2*b := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l330_33027


namespace NUMINAMATH_CALUDE_stevens_peaches_l330_33035

theorem stevens_peaches (jake steven jill : ℕ) 
  (h1 : jake = steven - 18)
  (h2 : steven = jill + 13)
  (h3 : jill = 6) : 
  steven = 19 := by
sorry

end NUMINAMATH_CALUDE_stevens_peaches_l330_33035


namespace NUMINAMATH_CALUDE_initial_hay_bales_l330_33038

theorem initial_hay_bales (better_quality_cost previous_cost : ℚ) 
  (cost_difference : ℚ) : 
  better_quality_cost = 18 →
  previous_cost = 15 →
  cost_difference = 210 →
  ∃ x : ℚ, x = 10 ∧ 2 * better_quality_cost * x - previous_cost * x = cost_difference :=
by sorry

end NUMINAMATH_CALUDE_initial_hay_bales_l330_33038


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l330_33016

theorem negative_fraction_comparison : -5/6 < -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l330_33016


namespace NUMINAMATH_CALUDE_rolling_semicircle_distance_l330_33018

/-- The distance traveled by the center of a rolling semi-circle -/
theorem rolling_semicircle_distance (r : ℝ) (h : r = 4 / Real.pi) :
  2 * Real.pi * r / 2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_rolling_semicircle_distance_l330_33018


namespace NUMINAMATH_CALUDE_box_volume_l330_33020

theorem box_volume (sheet_length sheet_width cut_length : ℝ) 
  (h1 : sheet_length = 46)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 8) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 4800 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l330_33020


namespace NUMINAMATH_CALUDE_trivia_team_size_l330_33098

theorem trivia_team_size :
  ∀ (original_members : ℕ),
  (original_members ≥ 2) →
  (4 * (original_members - 2) = 20) →
  original_members = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_team_size_l330_33098
