import Mathlib

namespace NUMINAMATH_CALUDE_dagger_example_l21_2186

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem dagger_example : dagger (3/7) (11/4) = 132/7 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l21_2186


namespace NUMINAMATH_CALUDE_equal_probabilities_l21_2141

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  green : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  red_box : Box
  green_box : Box

/-- Initial state of the boxes -/
def initial_state : BoxState :=
  { red_box := { red := 100, green := 0 },
    green_box := { red := 0, green := 100 } }

/-- State after transferring 8 red balls from red box to green box -/
def after_first_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red - 8, green := state.red_box.green },
    green_box := { red := state.green_box.red + 8, green := state.green_box.green } }

/-- State after transferring 8 balls from green box to red box -/
def after_second_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red + 8, green := state.red_box.green },
    green_box := { red := state.green_box.red - 8, green := state.green_box.green } }

/-- Final state after all transfers -/
def final_state : BoxState :=
  after_second_transfer (after_first_transfer initial_state)

/-- Probability of drawing a specific color ball from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => box.red / (box.red + box.green)
  | "green" => box.green / (box.red + box.green)
  | _ => 0

theorem equal_probabilities :
  prob_draw final_state.red_box "green" = prob_draw final_state.green_box "red" := by
  sorry

end NUMINAMATH_CALUDE_equal_probabilities_l21_2141


namespace NUMINAMATH_CALUDE_planes_perp_to_line_are_parallel_line_in_plane_perp_to_other_plane_implies_planes_perp_l21_2138

-- Define basic geometric objects
variable (P Q R : Plane) (L M : Line)

-- Define geometric relationships
def perpendicular (L : Line) (P : Plane) : Prop := sorry
def parallel (P Q : Plane) : Prop := sorry
def contains (P : Plane) (L : Line) : Prop := sorry

-- Theorem 1: Two planes perpendicular to the same line are parallel to each other
theorem planes_perp_to_line_are_parallel 
  (h1 : perpendicular L P) (h2 : perpendicular L Q) : parallel P Q := by sorry

-- Theorem 2: If a line within a plane is perpendicular to another plane, 
-- then these two planes are perpendicular to each other
theorem line_in_plane_perp_to_other_plane_implies_planes_perp 
  (h1 : contains P L) (h2 : perpendicular L Q) : perpendicular P Q := by sorry

end NUMINAMATH_CALUDE_planes_perp_to_line_are_parallel_line_in_plane_perp_to_other_plane_implies_planes_perp_l21_2138


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l21_2111

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- A parabola defined by a focus point and a directrix line -/
structure Parabola :=
  (focus : Point) (directrix : Line)

/-- Represents the intersection points between a line and a parabola -/
inductive Intersection
  | NoIntersection
  | OnePoint (p : Point)
  | TwoPoints (p1 p2 : Point)

/-- 
Given a point F (focus), a line L, and a line D (directrix) in a plane,
there exists a construction method to find the intersection points (if any)
between L and the parabola defined by focus F and directrix D.
-/
theorem parabola_line_intersection
  (F : Point) (L D : Line) :
  ∃ (construct : Point → Line → Line → Intersection),
    construct F L D = Intersection.NoIntersection ∨
    (∃ p, construct F L D = Intersection.OnePoint p) ∨
    (∃ p1 p2, construct F L D = Intersection.TwoPoints p1 p2) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l21_2111


namespace NUMINAMATH_CALUDE_volume_of_sphere_wedge_l21_2129

/-- Given a sphere with circumference 18π inches cut into six congruent wedges,
    prove that the volume of one wedge is 162π cubic inches. -/
theorem volume_of_sphere_wedge :
  ∀ (r : ℝ), 
    r > 0 →
    2 * Real.pi * r = 18 * Real.pi →
    (4 / 3 * Real.pi * r^3) / 6 = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_of_sphere_wedge_l21_2129


namespace NUMINAMATH_CALUDE_function_value_at_negative_l21_2133

/-- Given a function f(x) = ax³ + bx - c/x + 2, if f(2023) = 6, then f(-2023) = -2 -/
theorem function_value_at_negative (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^3 + b * x - c / x + 2
  f 2023 = 6 → f (-2023) = -2 := by sorry

end NUMINAMATH_CALUDE_function_value_at_negative_l21_2133


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l21_2107

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 1145 * n ≡ 1717 * n [ZMOD 36] ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬(1145 * m ≡ 1717 * m [ZMOD 36])) ∧ 
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l21_2107


namespace NUMINAMATH_CALUDE_attendance_decrease_l21_2198

/-- Proves that given a projected 25 percent increase in attendance and actual attendance being 64 percent of the projected attendance, the actual percent decrease in attendance is 20 percent. -/
theorem attendance_decrease (P : ℝ) (P_positive : P > 0) : 
  let projected_attendance := 1.25 * P
  let actual_attendance := 0.64 * projected_attendance
  let percent_decrease := (P - actual_attendance) / P * 100
  percent_decrease = 20 := by
  sorry

end NUMINAMATH_CALUDE_attendance_decrease_l21_2198


namespace NUMINAMATH_CALUDE_odometer_puzzle_l21_2135

theorem odometer_puzzle (a b c d : ℕ) (h1 : a ≥ 1) (h2 : a + b + c + d = 10)
  (h3 : ∃ (x : ℕ), 1000 * (d - a) + 100 * (c - b) + 10 * (b - c) + (a - d) = 65 * x) :
  a^2 + b^2 + c^2 + d^2 = 42 := by
sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l21_2135


namespace NUMINAMATH_CALUDE_tennis_players_count_l21_2185

theorem tennis_players_count (total : ℕ) (baseball : ℕ) (both : ℕ) (no_sport : ℕ) :
  total = 310 →
  baseball = 255 →
  both = 94 →
  no_sport = 11 →
  ∃ tennis : ℕ, tennis = 138 ∧ total = tennis + baseball - both + no_sport :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_count_l21_2185


namespace NUMINAMATH_CALUDE_compound_interest_proof_l21_2182

/-- Given a principal amount for which the simple interest over 2 years at 10% rate is $600,
    prove that the compound interest over 2 years at 10% rate is $630 --/
theorem compound_interest_proof (P : ℝ) : 
  P * 0.1 * 2 = 600 → P * (1 + 0.1)^2 - P = 630 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_proof_l21_2182


namespace NUMINAMATH_CALUDE_min_distance_to_line_l21_2164

theorem min_distance_to_line (x y : ℝ) :
  8 * x + 15 * y = 120 →
  x ≥ 0 →
  ∃ (min : ℝ), min = 120 / 17 ∧ ∀ (x' y' : ℝ), 8 * x' + 15 * y' = 120 → x' ≥ 0 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l21_2164


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_b_l21_2188

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- The direction vector of the first line -/
def v : ℝ × ℝ := (4, -9)

/-- The direction vector of the second line -/
def w (b : ℝ) : ℝ × ℝ := (b, 3)

/-- Theorem: If the direction vectors v and w(b) are perpendicular, then b = 27/4 -/
theorem perpendicular_vectors_imply_b (b : ℝ) :
  perpendicular v (w b) → b = 27/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_b_l21_2188


namespace NUMINAMATH_CALUDE_new_cylinder_volume_l21_2162

/-- Theorem: New volume of a cylinder after tripling radius and doubling height -/
theorem new_cylinder_volume (r h : ℝ) (h1 : r > 0) (h2 : h > 0) (h3 : π * r^2 * h = 15) :
  π * (3*r)^2 * (2*h) = 270 := by
  sorry

end NUMINAMATH_CALUDE_new_cylinder_volume_l21_2162


namespace NUMINAMATH_CALUDE_freds_allowance_l21_2181

theorem freds_allowance (allowance : ℝ) : 
  (allowance / 2 + 6 = 14) → allowance = 16 := by
  sorry

end NUMINAMATH_CALUDE_freds_allowance_l21_2181


namespace NUMINAMATH_CALUDE_typing_time_proof_l21_2195

def typing_speed : ℕ := 38
def paper_length : ℕ := 4560
def minutes_per_hour : ℕ := 60

theorem typing_time_proof :
  (paper_length / typing_speed : ℚ) / minutes_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_proof_l21_2195


namespace NUMINAMATH_CALUDE_small_branches_count_l21_2152

/-- Represents the structure of a plant with branches and small branches. -/
structure Plant where
  small_branches_per_branch : ℕ
  total_count : ℕ

/-- The plant satisfies the given conditions. -/
def valid_plant (p : Plant) : Prop :=
  p.total_count = 1 + p.small_branches_per_branch + p.small_branches_per_branch^2

/-- Theorem: Given the conditions, the number of small branches per branch is 9. -/
theorem small_branches_count (p : Plant) 
    (h : valid_plant p) 
    (h_total : p.total_count = 91) : 
  p.small_branches_per_branch = 9 := by
  sorry

#check small_branches_count

end NUMINAMATH_CALUDE_small_branches_count_l21_2152


namespace NUMINAMATH_CALUDE_stratified_sampling_best_l21_2120

structure Population where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  sample_size : ℕ

def is_equal_proportion (p : Population) : Prop :=
  p.group1 = p.group2 ∧ p.total = p.group1 + p.group2

def maintains_proportion (p : Population) (method : String) : Prop :=
  method = "stratified sampling"

theorem stratified_sampling_best (p : Population) 
  (h1 : is_equal_proportion p) 
  (h2 : p.sample_size < p.total) :
  ∃ (method : String), maintains_proportion p method :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_best_l21_2120


namespace NUMINAMATH_CALUDE_shortest_rope_length_l21_2109

theorem shortest_rope_length (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a : ℝ) / 4 = (b : ℝ) / 5 ∧ (b : ℝ) / 5 = (c : ℝ) / 6 →
  a + c = b + 100 →
  a = 80 := by
sorry

end NUMINAMATH_CALUDE_shortest_rope_length_l21_2109


namespace NUMINAMATH_CALUDE_closed_set_A_l21_2147

def f (x : ℚ) : ℚ := (1 + x) / (1 - x)

def A : Set ℚ := {2, -3, -1/2, 1/3}

theorem closed_set_A :
  (2 ∈ A) ∧
  (∀ x ∈ A, f x ∈ A) ∧
  (∀ S : Set ℚ, 2 ∈ S → (∀ x ∈ S, f x ∈ S) → A ⊆ S) :=
sorry

end NUMINAMATH_CALUDE_closed_set_A_l21_2147


namespace NUMINAMATH_CALUDE_calculate_expression_l21_2174

theorem calculate_expression : 
  2 / (-1/4) - |(-Real.sqrt 18)| + (1/5)⁻¹ = -3 * Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l21_2174


namespace NUMINAMATH_CALUDE_anhui_imports_exports_2012_l21_2140

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem anhui_imports_exports_2012 :
  toScientificNotation (39.33 * 10^9) = ScientificNotation.mk 3.933 10 sorry := by
  sorry

end NUMINAMATH_CALUDE_anhui_imports_exports_2012_l21_2140


namespace NUMINAMATH_CALUDE_range_of_a_l21_2165

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + x - 2 > 0
def q (x a : ℝ) : Prop := x > a

-- Define what it means for q to be sufficient but not necessary for p
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l21_2165


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l21_2112

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := fun i => if i = 0 then x else 1
def b (x : ℝ) : Fin 2 → ℝ := fun i => if i = 0 then 4 else x

-- Define the parallel condition
def are_parallel (x : ℝ) : Prop := ∃ k : ℝ, ∀ i : Fin 2, a x i = k * b x i

-- Statement: x = 2 is sufficient but not necessary for a and b to be parallel
theorem x_eq_2_sufficient_not_necessary :
  (are_parallel 2) ∧ (∃ y : ℝ, y ≠ 2 ∧ are_parallel y) := by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l21_2112


namespace NUMINAMATH_CALUDE_find_M_and_N_l21_2178

theorem find_M_and_N :
  ∀ M N : ℕ,
  0 < M ∧ M < 10 ∧ 0 < N ∧ N < 10 →
  8 * 10^7 + M * 10^6 + 420852 * 9 = N * 10^7 + 9889788 * 11 →
  M = 5 ∧ N = 6 := by
sorry

end NUMINAMATH_CALUDE_find_M_and_N_l21_2178


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l21_2122

theorem magnitude_of_complex_number : Complex.abs (5/6 + 2*Complex.I) = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l21_2122


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l21_2191

/-- Given a line segment with midpoint (3, 4) and one endpoint (-2, -5), 
    the other endpoint is (8, 13) -/
theorem other_endpoint_of_line_segment 
  (midpoint : ℝ × ℝ) 
  (endpoint1 : ℝ × ℝ) 
  (endpoint2 : ℝ × ℝ) : 
  midpoint = (3, 4) → 
  endpoint1 = (-2, -5) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) → 
  endpoint2 = (8, 13) := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l21_2191


namespace NUMINAMATH_CALUDE_g_is_even_l21_2137

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define g as the sum of f(x) and f(-x)
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem stating that g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by
  sorry

end NUMINAMATH_CALUDE_g_is_even_l21_2137


namespace NUMINAMATH_CALUDE_cats_given_away_l21_2104

/-- Proves that the number of cats given away is 14, given the initial and remaining cat counts -/
theorem cats_given_away (initial_cats : ℝ) (remaining_cats : ℕ) 
  (h1 : initial_cats = 17.0) (h2 : remaining_cats = 3) : 
  initial_cats - remaining_cats = 14 := by
  sorry

end NUMINAMATH_CALUDE_cats_given_away_l21_2104


namespace NUMINAMATH_CALUDE_sum_K_floor_quotient_100_l21_2160

/-- K(x) is the number of irreducible fractions a/b where 1 ≤ a < x and 1 ≤ b < x -/
def K (x : ℕ) : ℕ :=
  (Finset.range (x - 1)).sum (λ k => Nat.totient k)

/-- The sum of K(⌊100/k⌋) for k from 1 to 100 equals 9801 -/
theorem sum_K_floor_quotient_100 :
  (Finset.range 100).sum (λ k => K (100 / (k + 1))) = 9801 := by
  sorry

end NUMINAMATH_CALUDE_sum_K_floor_quotient_100_l21_2160


namespace NUMINAMATH_CALUDE_sin_585_degrees_l21_2189

theorem sin_585_degrees :
  let π : ℝ := Real.pi
  let deg_to_rad (x : ℝ) : ℝ := x * π / 180
  ∀ (sin : ℝ → ℝ),
    (∀ x, sin (x + 2 * π) = sin x) →  -- Periodicity of sine
    (∀ x, sin (x + π) = -sin x) →     -- Sine of sum property
    sin (deg_to_rad 45) = Real.sqrt 2 / 2 →  -- Value of sin 45°
    sin (deg_to_rad 585) = -Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l21_2189


namespace NUMINAMATH_CALUDE_counter_value_l21_2148

/-- Given a counter with 'a' beads in the tens place and 'b' beads in the ones place,
    the number represented by this counter is equal to 10a + b. -/
theorem counter_value (a b : ℕ) : 10 * a + b = 10 * a + b := by
  sorry

end NUMINAMATH_CALUDE_counter_value_l21_2148


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l21_2192

/-- Number of games required to determine a champion in a single-elimination tournament -/
def games_required (num_players : ℕ) : ℕ := num_players - 1

/-- Theorem: In a single-elimination tournament with 512 players, 511 games are required to determine the champion -/
theorem single_elimination_tournament_games (num_players : ℕ) (h : num_players = 512) :
  games_required num_players = 511 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l21_2192


namespace NUMINAMATH_CALUDE_pencil_pen_problem_l21_2153

theorem pencil_pen_problem (S : Finset Nat) (A B : Finset Nat) :
  S.card = 400 →
  A ⊆ S →
  B ⊆ S →
  A.card = 375 →
  B.card = 80 →
  S = A ∪ B →
  (A \ B).card = 320 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_problem_l21_2153


namespace NUMINAMATH_CALUDE_exists_sum_all_odd_digits_l21_2168

/-- A function that returns true if all digits of a natural number are odd -/
def allDigitsOdd (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

/-- A function that reverses the digits of a three-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 100 + (n / 10 % 10) * 10 + (n / 100)

/-- Theorem stating that there exists a three-digit number A such that
    A + reverseDigits(A) has all odd digits -/
theorem exists_sum_all_odd_digits :
  ∃ A : ℕ, 100 ≤ A ∧ A < 1000 ∧ allDigitsOdd (A + reverseDigits A) :=
sorry

end NUMINAMATH_CALUDE_exists_sum_all_odd_digits_l21_2168


namespace NUMINAMATH_CALUDE_equation_solutions_l21_2139

theorem equation_solutions (p : ℕ) (h_prime : Nat.Prime p) :
  ∃! (solutions : Finset (ℕ × ℕ)), 
    solutions.card = 3 ∧
    ∀ (x y : ℕ), (x, y) ∈ solutions ↔ 
      (x > 0 ∧ y > 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / p) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l21_2139


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l21_2197

def total_missiles : ℕ := 60
def selected_missiles : ℕ := 6

def systematic_sample (total : ℕ) (select : ℕ) : List ℕ :=
  let interval := total / select
  List.range select |>.map (fun i => i * interval + interval / 2 + 1)

theorem correct_systematic_sample :
  systematic_sample total_missiles selected_missiles = [3, 13, 23, 33, 43, 53] :=
sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l21_2197


namespace NUMINAMATH_CALUDE_shell_calculation_l21_2184

theorem shell_calculation (initial : Real) (add1 : Real) (add2 : Real) (subtract : Real) (final : Real) :
  initial = 5.2 ∧ add1 = 15.7 ∧ add2 = 17.5 ∧ subtract = 4.3 ∧ final = 102.3 →
  final = 3 * ((initial + add1 + add2 - subtract)) :=
by sorry

end NUMINAMATH_CALUDE_shell_calculation_l21_2184


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l21_2187

theorem fraction_zero_implies_x_equals_one (x : ℝ) : 
  (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l21_2187


namespace NUMINAMATH_CALUDE_max_candies_eaten_l21_2158

theorem max_candies_eaten (n : Nat) (h : n = 46) : 
  (n * (n - 1)) / 2 = 1035 := by
  sorry

#check max_candies_eaten

end NUMINAMATH_CALUDE_max_candies_eaten_l21_2158


namespace NUMINAMATH_CALUDE_sqrt_720_simplification_l21_2110

theorem sqrt_720_simplification : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_simplification_l21_2110


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l21_2177

-- Define the set M
def M : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

-- Define the set N
def N : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_equals_interval : {x : ℝ | 1 < x ∧ x ≤ 2} = M ∩ N := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l21_2177


namespace NUMINAMATH_CALUDE_unknown_rope_length_l21_2179

/-- Calculates the length of an unknown rope given other rope lengths and conditions --/
theorem unknown_rope_length
  (known_ropes : List ℝ)
  (knot_loss : ℝ)
  (final_length : ℝ)
  (h1 : known_ropes = [8, 20, 2, 2, 2])
  (h2 : knot_loss = 1.2)
  (h3 : final_length = 35) :
  ∃ x : ℝ, x = 5.8 ∧ 
    final_length + (known_ropes.length * knot_loss) = 
    (known_ropes.sum + x) := by
  sorry


end NUMINAMATH_CALUDE_unknown_rope_length_l21_2179


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l21_2132

theorem complex_magnitude_equality (n : ℝ) :
  n > 0 ∧ Complex.abs (2 + n * Complex.I) = 4 * Real.sqrt 5 ↔ n = 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l21_2132


namespace NUMINAMATH_CALUDE_inequality_equivalence_inequality_positive_reals_l21_2176

-- Problem 1
theorem inequality_equivalence (x : ℝ) : (x + 2) / (2 - 3 * x) > 1 ↔ 0 < x ∧ x < 2 / 3 := by sorry

-- Problem 2
theorem inequality_positive_reals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + a*c := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_inequality_positive_reals_l21_2176


namespace NUMINAMATH_CALUDE_omega_range_l21_2125

theorem omega_range (ω : ℝ) (f : ℝ → ℝ) : 
  ω > 0 → 
  (∀ x, f x = Real.sin (ω * x + π / 4)) →
  (∀ x y, π / 2 < x → x < y → y < π → f y < f x) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_omega_range_l21_2125


namespace NUMINAMATH_CALUDE_lcm_180_616_l21_2121

theorem lcm_180_616 : Nat.lcm 180 616 = 27720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_180_616_l21_2121


namespace NUMINAMATH_CALUDE_freds_salary_l21_2114

theorem freds_salary (mikes_current_salary : ℝ) (mikes_salary_ratio : ℝ) (salary_increase_percent : ℝ) :
  mikes_current_salary = 15400 ∧
  mikes_salary_ratio = 10 ∧
  salary_increase_percent = 40 →
  (mikes_current_salary / (1 + salary_increase_percent / 100) / mikes_salary_ratio) = 1100 :=
by sorry

end NUMINAMATH_CALUDE_freds_salary_l21_2114


namespace NUMINAMATH_CALUDE_fraction_irreducible_l21_2190

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l21_2190


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l21_2143

theorem complex_modulus_problem (z : ℂ) (i : ℂ) (h1 : i^2 = -1) (h2 : (1 - i) * z = 1) : 
  Complex.abs (4 * z - 3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l21_2143


namespace NUMINAMATH_CALUDE_simplify_expression_l21_2180

theorem simplify_expression : Real.sqrt ((-2)^6) - (-1)^0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l21_2180


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l21_2117

/-- Given an arithmetic sequence {a_n} where a_2 + a_8 = 16, prove that a_5 = 8 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : a 2 + a 8 = 16) : 
  a 5 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l21_2117


namespace NUMINAMATH_CALUDE_expression_bounds_l21_2157

-- Define the constraint function
def constraint (x y : ℝ) : Prop := (|x| - 3)^2 + (|y| - 2)^2 = 1

-- Define the expression to be minimized/maximized
def expression (x y : ℝ) : ℝ := |x + 2| + |y + 3|

-- Theorem statement
theorem expression_bounds :
  (∃ x y : ℝ, constraint x y) →
  (∃ min max : ℝ,
    (∀ x y : ℝ, constraint x y → expression x y ≥ min) ∧
    (∃ x y : ℝ, constraint x y ∧ expression x y = min) ∧
    (∀ x y : ℝ, constraint x y → expression x y ≤ max) ∧
    (∃ x y : ℝ, constraint x y ∧ expression x y = max) ∧
    min = 2 - Real.sqrt 2 ∧
    max = 10 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_expression_bounds_l21_2157


namespace NUMINAMATH_CALUDE_restaurant_tip_percentage_l21_2183

/-- Calculates the tip percentage given the total bill and tip amount -/
def tip_percentage (total_bill : ℚ) (tip_amount : ℚ) : ℚ :=
  (tip_amount / total_bill) * 100

/-- Proves that for a $40 bill and $4 tip, the tip percentage is 10% -/
theorem restaurant_tip_percentage : tip_percentage 40 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_tip_percentage_l21_2183


namespace NUMINAMATH_CALUDE_four_card_selection_three_suits_l21_2167

theorem four_card_selection_three_suits (deck_size : Nat) (suits : Nat) (cards_per_suit : Nat) 
  (selection_size : Nat) (suits_represented : Nat) (cards_from_main_suit : Nat) :
  deck_size = suits * cards_per_suit →
  selection_size = 4 →
  suits = 4 →
  cards_per_suit = 13 →
  suits_represented = 3 →
  cards_from_main_suit = 2 →
  (suits.choose 1) * (suits - 1).choose (suits_represented - 1) * 
  (cards_per_suit.choose cards_from_main_suit) * 
  (cards_per_suit.choose 1) * (cards_per_suit.choose 1) = 158184 := by
sorry

end NUMINAMATH_CALUDE_four_card_selection_three_suits_l21_2167


namespace NUMINAMATH_CALUDE_tank_emptying_time_l21_2196

/-- Given a tank with specified capacity, leak rate, and inlet rate, 
    calculate the time it takes to empty when both leak and inlet are open. -/
theorem tank_emptying_time 
  (tank_capacity : ℝ) 
  (leak_time : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (h1 : tank_capacity = 1440) 
  (h2 : leak_time = 3) 
  (h3 : inlet_rate_per_minute = 6) : 
  (tank_capacity / (tank_capacity / leak_time - inlet_rate_per_minute * 60)) = 12 :=
by
  sorry

#check tank_emptying_time

end NUMINAMATH_CALUDE_tank_emptying_time_l21_2196


namespace NUMINAMATH_CALUDE_inequality_equivalence_l21_2116

theorem inequality_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 : ℝ) ^ (1 / (x + 1/x)) > (3 : ℝ) ^ (1 / (y + 1/y)) ↔ 
  (x > 0 ∧ y < 0) ∨ 
  (x > y ∧ y > 0 ∧ x * y > 1) ∨ 
  (x < y ∧ y < 0 ∧ 0 < x * y ∧ x * y < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l21_2116


namespace NUMINAMATH_CALUDE_triangle_properties_l21_2172

/-- Given a triangle ABC with acute angles A and B, prove the following:
    1. If ∠C = π/3 and c = 2, then 2 + 2√3 < perimeter ≤ 6
    2. If sin²A + sin²B > sin²C, then sin²A + sin²B > 1 -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (C = π/3 ∧ c = 2 → 2 + 2 * Real.sqrt 3 < a + b + c ∧ a + b + c ≤ 6) ∧
  (Real.sin A ^ 2 + Real.sin B ^ 2 > Real.sin C ^ 2 → Real.sin A ^ 2 + Real.sin B ^ 2 > 1) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l21_2172


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l21_2119

-- Define the # operation
def hash (a b : ℝ) : ℝ := a * b + 1

-- Theorem stating that none of the laws hold
theorem no_distributive_laws_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) := by
  sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l21_2119


namespace NUMINAMATH_CALUDE_no_2008_special_progressions_l21_2134

theorem no_2008_special_progressions : ¬ ∃ (progressions : Fin 2008 → Set ℕ),
  -- Each set in progressions is an infinite arithmetic progression
  (∀ i, ∃ (a d : ℕ), d > 0 ∧ progressions i = {n : ℕ | ∃ k, n = a + k * d}) ∧
  -- There are finitely many positive integers not in any progression
  (∃ S : Finset ℕ, ∀ n, n ∉ S → ∃ i, n ∈ progressions i) ∧
  -- No two progressions intersect
  (∀ i j, i ≠ j → progressions i ∩ progressions j = ∅) ∧
  -- Each progression contains a prime number bigger than 2008
  (∀ i, ∃ p ∈ progressions i, p > 2008 ∧ Nat.Prime p) :=
by
  sorry

end NUMINAMATH_CALUDE_no_2008_special_progressions_l21_2134


namespace NUMINAMATH_CALUDE_snow_probability_l21_2105

theorem snow_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 5) :
  1 - (1 - p)^n = 1023/1024 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l21_2105


namespace NUMINAMATH_CALUDE_red_parrots_count_l21_2156

theorem red_parrots_count (total : ℕ) (green_fraction : ℚ) (blue_fraction : ℚ) 
  (h_total : total = 160)
  (h_green : green_fraction = 5/8)
  (h_blue : blue_fraction = 1/4)
  (h_sum : green_fraction + blue_fraction < 1) :
  total - (green_fraction * total).num - (blue_fraction * total).num = 20 := by
  sorry

end NUMINAMATH_CALUDE_red_parrots_count_l21_2156


namespace NUMINAMATH_CALUDE_periodic_properties_l21_2126

-- Define a periodic function
def Periodic (f : ℝ → ℝ) : Prop :=
  ∃ T > 0, ∀ x, f (x + T) = f x

-- Define a non-periodic function
def NonPeriodic (g : ℝ → ℝ) : Prop :=
  ¬ Periodic g

theorem periodic_properties
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : Periodic f) (hg : NonPeriodic g) :
  Periodic (fun x ↦ (f x)^2) ∧
  NonPeriodic (fun x ↦ Real.sqrt (g x)) ∧
  Periodic (g ∘ f) :=
sorry

end NUMINAMATH_CALUDE_periodic_properties_l21_2126


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l21_2113

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ := a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowtie_equation_solution :
  ∀ y : ℝ, bowtie 3 y = 27 → y = 72 := by
sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l21_2113


namespace NUMINAMATH_CALUDE_transformed_ellipse_equation_l21_2161

/-- The equation of the curve obtained by transforming points on the ellipse x²/4 + y² = 1
    by keeping the x-coordinate unchanged and doubling the y-coordinate -/
theorem transformed_ellipse_equation :
  ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 → (x^2 + (2*y)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_transformed_ellipse_equation_l21_2161


namespace NUMINAMATH_CALUDE_product_expansion_sum_l21_2193

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x : ℝ, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + 2 * c + d = -44 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l21_2193


namespace NUMINAMATH_CALUDE_m_less_than_one_sufficient_not_necessary_l21_2108

-- Define the function f(x) = x^2 + 2x + m
def f (x m : ℝ) : ℝ := x^2 + 2*x + m

-- Define what it means for f to have a root
def has_root (m : ℝ) : Prop := ∃ x : ℝ, f x m = 0

-- Statement: "m < 1" is a sufficient but not necessary condition for f to have a root
theorem m_less_than_one_sufficient_not_necessary :
  (∀ m : ℝ, m < 1 → has_root m) ∧ 
  (∃ m : ℝ, ¬(m < 1) ∧ has_root m) :=
sorry

end NUMINAMATH_CALUDE_m_less_than_one_sufficient_not_necessary_l21_2108


namespace NUMINAMATH_CALUDE_min_cubes_for_box_l21_2151

-- Define the box dimensions
def box_length : ℝ := 10
def box_width : ℝ := 18
def box_height : ℝ := 4

-- Define the volume of a single cube
def cube_volume : ℝ := 12

-- Theorem statement
theorem min_cubes_for_box :
  ⌈(box_length * box_width * box_height) / cube_volume⌉ = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_box_l21_2151


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l21_2175

theorem set_membership_implies_value (m : ℚ) : 
  let A : Set ℚ := {m + 2, 2 * m^2 + m}
  3 ∈ A → m = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l21_2175


namespace NUMINAMATH_CALUDE_ali_baba_treasure_max_value_l21_2130

/-- The maximum value problem for Ali Baba's treasure --/
theorem ali_baba_treasure_max_value :
  let f : ℝ → ℝ → ℝ := λ x y => 20 * x + 60 * y
  let S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 ≤ 100 ∧ p.1 + 5 * p.2 ≤ 200}
  ∃ (x y : ℝ), (x, y) ∈ S ∧ f x y = 3000 ∧ ∀ (x' y' : ℝ), (x', y') ∈ S → f x' y' ≤ 3000 :=
by sorry


end NUMINAMATH_CALUDE_ali_baba_treasure_max_value_l21_2130


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l21_2144

theorem arithmetic_geometric_mean_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  let A := (x + y) / 2
  let G := Real.sqrt (x * y)
  A / G = 5 / 4 → x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l21_2144


namespace NUMINAMATH_CALUDE_contrapositive_exponential_l21_2118

theorem contrapositive_exponential (a b : ℝ) :
  (∀ a b, a > b → 2^a > 2^b) ↔ (∀ a b, 2^a ≤ 2^b → a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_exponential_l21_2118


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l21_2123

theorem least_k_for_inequality : 
  ∃ k : ℕ+, (∀ a : ℝ, a ∈ Set.Icc 0 1 → ∀ n : ℕ+, (a^(k:ℝ) * (1 - a)^(n:ℝ) < 1 / ((n:ℝ) + 1)^3)) ∧ 
  (∀ k' : ℕ+, k' < k → ∃ a : ℝ, a ∈ Set.Icc 0 1 ∧ ∃ n : ℕ+, a^(k':ℝ) * (1 - a)^(n:ℝ) ≥ 1 / ((n:ℝ) + 1)^3) ∧
  k = 4 :=
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l21_2123


namespace NUMINAMATH_CALUDE_expression_factorization_l21_2169

theorem expression_factorization (a b c d p q r s : ℝ) :
  (a * p + b * q + c * r + d * s)^2 +
  (a * q - b * p + c * s - d * r)^2 +
  (a * r - b * s - c * p + d * q)^2 +
  (a * s + b * r - c * q - d * p)^2 =
  (a^2 + b^2 + c^2 + d^2) * (p^2 + q^2 + r^2 + s^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l21_2169


namespace NUMINAMATH_CALUDE_nail_sizes_l21_2150

theorem nail_sizes (fraction_2d : ℝ) (fraction_2d_or_4d : ℝ) (fraction_4d : ℝ) :
  fraction_2d = 0.25 →
  fraction_2d_or_4d = 0.75 →
  fraction_4d = fraction_2d_or_4d - fraction_2d →
  fraction_4d = 0.50 := by
sorry

end NUMINAMATH_CALUDE_nail_sizes_l21_2150


namespace NUMINAMATH_CALUDE_hamburger_sales_proof_l21_2194

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of hamburgers sold per day -/
def average_daily_sales : ℕ := 9

/-- The total number of hamburgers sold in a week -/
def total_weekly_sales : ℕ := days_in_week * average_daily_sales

theorem hamburger_sales_proof : total_weekly_sales = 63 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_sales_proof_l21_2194


namespace NUMINAMATH_CALUDE_min_value_equiv_k_l21_2106

/-- The polynomial function f(x, y, k) -/
def f (x y k : ℝ) : ℝ := 9*x^2 - 12*k*x*y + (2*k^2 + 3)*y^2 - 6*x - 9*y + 12

/-- The theorem stating the equivalence between the minimum value of f being 0 and k = √(3)/4 -/
theorem min_value_equiv_k (k : ℝ) : 
  (∀ x y : ℝ, f x y k ≥ 0) ∧ (∃ x y : ℝ, f x y k = 0) ↔ k = Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_equiv_k_l21_2106


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l21_2154

/-- The line y = kx + 2 intersects the hyperbola x^2 - y^2 = 2 at exactly one point if and only if k = ±1 or k = ±√3 -/
theorem line_hyperbola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.1^2 - p.2^2 = 2) ↔ 
  (k = 1 ∨ k = -1 ∨ k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l21_2154


namespace NUMINAMATH_CALUDE_negation_of_inequality_proposition_l21_2115

theorem negation_of_inequality_proposition :
  (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2*a*b) ↔ (∃ a b : ℝ, a^2 + b^2 < 2*a*b) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_inequality_proposition_l21_2115


namespace NUMINAMATH_CALUDE_min_occupied_seats_150_l21_2199

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure the next person must sit next to someone -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 150 seats, the minimum number of occupied seats
    required to ensure the next person must sit next to someone is 90 -/
theorem min_occupied_seats_150 : min_occupied_seats 150 = 90 := by
  sorry

end NUMINAMATH_CALUDE_min_occupied_seats_150_l21_2199


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l21_2131

-- Define the sets A, B, and C
def A : Set ℝ := {-6, -5, -4, -3}
def B : Set ℝ := {2/3, 3/4, 7/9, 2.5}
def C : Set ℝ := {5, 5.5, 6, 6.5}

-- Define the theorem
theorem greatest_integer_difference (a b c : ℝ) 
  (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) :
  ∃ (d : ℤ), d = 5 ∧ 
  ∀ (a' b' c' : ℝ), a' ∈ A → b' ∈ B → c' ∈ C → 
  (Int.floor (|c' - Real.sqrt b' - (a' + Real.sqrt b')|) : ℤ) ≤ d :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l21_2131


namespace NUMINAMATH_CALUDE_sampling_is_systematic_l21_2149

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | Lottery
  | Random
  | Systematic

/-- Represents a class in the freshman year -/
structure FreshmanClass where
  students : Fin 56

/-- Represents the freshman year -/
structure FreshmanYear where
  classes : Fin 35 → FreshmanClass

/-- Defines the sampling method used in the problem -/
def samplingMethodUsed (year : FreshmanYear) : SamplingMethod :=
  SamplingMethod.Systematic

/-- Theorem stating that the sampling method used is systematic sampling -/
theorem sampling_is_systematic (year : FreshmanYear) :
  samplingMethodUsed year = SamplingMethod.Systematic :=
sorry

end NUMINAMATH_CALUDE_sampling_is_systematic_l21_2149


namespace NUMINAMATH_CALUDE_three_digit_addition_l21_2136

theorem three_digit_addition (A B : Nat) : A < 10 → B < 10 → 
  600 + 10 * A + 5 + 100 + 10 * B = 748 → B = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_addition_l21_2136


namespace NUMINAMATH_CALUDE_max_sum_squares_l21_2155

theorem max_sum_squares (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + 2*b + 3*c = 1) :
  ∃ (max : ℝ), max = 1 ∧ a^2 + b^2 + c^2 ≤ max ∧ ∃ (a' b' c' : ℝ), 
    0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + 2*b' + 3*c' = 1 ∧ a'^2 + b'^2 + c'^2 = max :=
sorry

end NUMINAMATH_CALUDE_max_sum_squares_l21_2155


namespace NUMINAMATH_CALUDE_portfolio_calculations_l21_2100

/-- Represents a stock with its yield and quote -/
structure Stock where
  yield : ℝ
  quote : ℝ

/-- Calculates the weighted average yield of a portfolio -/
def weightedAverageYield (stocks : List Stock) (proportions : List ℝ) : ℝ :=
  sorry

/-- Calculates the overall quote of a portfolio -/
def overallQuote (stocks : List Stock) (proportions : List ℝ) (totalInvestment : ℝ) : ℝ :=
  sorry

/-- Theorem stating that weighted average yield and overall quote can be calculated -/
theorem portfolio_calculations 
  (stocks : List Stock) 
  (proportions : List ℝ) 
  (totalInvestment : ℝ) 
  (h1 : stocks.length = 3)
  (h2 : proportions.length = 3)
  (h3 : proportions.sum = 1)
  (h4 : totalInvestment > 0) :
  ∃ (avgYield overallQ : ℝ), 
    avgYield = weightedAverageYield stocks proportions ∧ 
    overallQ = overallQuote stocks proportions totalInvestment :=
  sorry

end NUMINAMATH_CALUDE_portfolio_calculations_l21_2100


namespace NUMINAMATH_CALUDE_function_equality_l21_2101

theorem function_equality (x : ℝ) : x = Real.log (Real.exp x) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l21_2101


namespace NUMINAMATH_CALUDE_side_angle_relation_l21_2163

theorem side_angle_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  (a > b ↔ Real.sin A > Real.sin B) :=
by sorry

end NUMINAMATH_CALUDE_side_angle_relation_l21_2163


namespace NUMINAMATH_CALUDE_fgh_supermarket_difference_l21_2146

theorem fgh_supermarket_difference (total : ℕ) (us : ℕ) (h1 : total = 70) (h2 : us = 42) (h3 : us < total) :
  us - (total - us) = 14 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_difference_l21_2146


namespace NUMINAMATH_CALUDE_circle_division_relationship_l21_2102

theorem circle_division_relationship (a k : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → (x = a ∨ x = -a ∨ y = k * x)) →
  a^2 * (k^2 + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_division_relationship_l21_2102


namespace NUMINAMATH_CALUDE_solution_system_l21_2173

theorem solution_system (x y m n : ℤ) : 
  x = 2 ∧ y = -3 ∧ x + y = m ∧ 2 * x - y = n → m - n = -8 := by
  sorry

end NUMINAMATH_CALUDE_solution_system_l21_2173


namespace NUMINAMATH_CALUDE_anya_erasers_difference_l21_2128

theorem anya_erasers_difference (andrea_erasers : ℕ) (anya_ratio : ℚ) : 
  andrea_erasers = 6 → 
  anya_ratio = 4.5 → 
  (anya_ratio * andrea_erasers : ℚ) - andrea_erasers = 21 := by
sorry

end NUMINAMATH_CALUDE_anya_erasers_difference_l21_2128


namespace NUMINAMATH_CALUDE_mrs_kaplan_pizza_slices_l21_2142

theorem mrs_kaplan_pizza_slices :
  ∀ (bobby_pizzas : ℕ) (slices_per_pizza : ℕ) (kaplan_fraction : ℚ),
    bobby_pizzas = 2 →
    slices_per_pizza = 6 →
    kaplan_fraction = 1 / 4 →
    (↑bobby_pizzas * ↑slices_per_pizza : ℚ) * kaplan_fraction = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_kaplan_pizza_slices_l21_2142


namespace NUMINAMATH_CALUDE_average_score_theorem_l21_2124

def max_score : ℕ := 900
def amar_percent : ℕ := 64
def bhavan_percent : ℕ := 36
def chetan_percent : ℕ := 44
def num_boys : ℕ := 3

theorem average_score_theorem :
  let amar_score := max_score * amar_percent / 100
  let bhavan_score := max_score * bhavan_percent / 100
  let chetan_score := max_score * chetan_percent / 100
  let total_score := amar_score + bhavan_score + chetan_score
  (total_score / num_boys : ℚ) = 432 := by sorry

end NUMINAMATH_CALUDE_average_score_theorem_l21_2124


namespace NUMINAMATH_CALUDE_two_color_line_exists_l21_2127

/-- Represents a color in the grid -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- Represents a point in the 2D grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point in the grid -/
def Coloring := Point → Color

/-- Predicate to check if four points form a unit square -/
def isUnitSquare (p1 p2 p3 p4 : Point) : Prop :=
  (p1.x = p2.x ∧ p1.y + 1 = p2.y) ∧
  (p2.x + 1 = p3.x ∧ p2.y = p3.y) ∧
  (p3.x = p4.x ∧ p3.y - 1 = p4.y) ∧
  (p4.x - 1 = p1.x ∧ p4.y = p1.y)

/-- Predicate to check if a coloring is valid (adjacent nodes in unit squares have different colors) -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ p1 p2 p3 p4 : Point, isUnitSquare p1 p2 p3 p4 →
    c p1 ≠ c p2 ∧ c p1 ≠ c p3 ∧ c p1 ≠ c p4 ∧
    c p2 ≠ c p3 ∧ c p2 ≠ c p4 ∧
    c p3 ≠ c p4

/-- Predicate to check if a line (horizontal or vertical) uses only two colors -/
def lineUsesTwoColors (c : Coloring) : Prop :=
  (∃ y : ℤ, ∃ c1 c2 : Color, ∀ x : ℤ, c ⟨x, y⟩ = c1 ∨ c ⟨x, y⟩ = c2) ∨
  (∃ x : ℤ, ∃ c1 c2 : Color, ∀ y : ℤ, c ⟨x, y⟩ = c1 ∨ c ⟨x, y⟩ = c2)

theorem two_color_line_exists (c : Coloring) (h : isValidColoring c) : lineUsesTwoColors c := by
  sorry

end NUMINAMATH_CALUDE_two_color_line_exists_l21_2127


namespace NUMINAMATH_CALUDE_square_fold_angle_l21_2103

/-- A square with side length 1 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1))

/-- The angle formed by two lines after folding a square along its diagonal -/
def dihedral_angle (s : Square) : ℝ := sorry

/-- Theorem: The dihedral angle formed by folding a square along its diagonal is 60° -/
theorem square_fold_angle (s : Square) : dihedral_angle s = 60 * π / 180 := by sorry

end NUMINAMATH_CALUDE_square_fold_angle_l21_2103


namespace NUMINAMATH_CALUDE_polynomial_simplification_l21_2166

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 5 * x^3 - 3 * x + 7) + (-x^4 + 4 * x^2 - 5 * x + 2) =
  x^4 + 5 * x^3 + 4 * x^2 - 8 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l21_2166


namespace NUMINAMATH_CALUDE_third_number_in_multiplication_l21_2145

theorem third_number_in_multiplication (p n : ℕ) : 
  (p = 125 * 243 * n / 405) → 
  (1000 ≤ p) → 
  (p < 10000) → 
  (∀ m : ℕ, m < n → 125 * 243 * m / 405 < 1000) →
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_third_number_in_multiplication_l21_2145


namespace NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l21_2170

theorem polygon_sides_from_interior_angle (interior_angle : ℝ) : 
  interior_angle = 140 → (360 / (180 - interior_angle) : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l21_2170


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l21_2159

theorem sum_of_roots_quadratic (x : ℝ) :
  x^2 - 6*x + 8 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 6 ∧ x = r₁ ∨ x = r₂ :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l21_2159


namespace NUMINAMATH_CALUDE_fraction_irreducible_l21_2171

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l21_2171
