import Mathlib

namespace NUMINAMATH_CALUDE_fish_apple_equivalence_l3285_328500

/-- Represents the value of one fish in terms of apples -/
def fish_value (f l r a : ℚ) : Prop :=
  5 * f = 3 * l ∧ l = 6 * r ∧ 3 * r = 2 * a ∧ f = 12/5 * a

/-- Theorem stating that under the given trading conditions, one fish is worth 12/5 apples -/
theorem fish_apple_equivalence :
  ∀ f l r a : ℚ, fish_value f l r a :=
by
  sorry

#check fish_apple_equivalence

end NUMINAMATH_CALUDE_fish_apple_equivalence_l3285_328500


namespace NUMINAMATH_CALUDE_unique_modulo_congruence_l3285_328591

theorem unique_modulo_congruence : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 99999 [ZMOD 11] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_modulo_congruence_l3285_328591


namespace NUMINAMATH_CALUDE_equation_holds_l3285_328509

theorem equation_holds (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l3285_328509


namespace NUMINAMATH_CALUDE_max_value_of_function_l3285_328513

theorem max_value_of_function (x y : ℝ) (h : x^2 + y^2 = 25) :
  (∀ a b : ℝ, a^2 + b^2 = 25 →
    Real.sqrt (8 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) ≥
    Real.sqrt (8 * b - 6 * a + 50) + Real.sqrt (8 * b + 6 * a + 50)) ∧
  (∃ a b : ℝ, a^2 + b^2 = 25 ∧
    Real.sqrt (8 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) =
    Real.sqrt (8 * b - 6 * a + 50) + Real.sqrt (8 * b + 6 * a + 50) ∧
    Real.sqrt (8 * b - 6 * a + 50) + Real.sqrt (8 * b + 6 * a + 50) = 6 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3285_328513


namespace NUMINAMATH_CALUDE_turning_process_terminates_l3285_328584

/-- Represents the direction a soldier is facing -/
inductive Direction
  | East
  | West

/-- Represents the state of the line of soldiers -/
def SoldierLine := List Direction

/-- Performs one step of the turning process -/
def turn_step (line : SoldierLine) : SoldierLine :=
  sorry

/-- Checks if the line is stable (no more turns needed) -/
def is_stable (line : SoldierLine) : Prop :=
  sorry

/-- The main theorem: the turning process will eventually stop -/
theorem turning_process_terminates (initial_line : SoldierLine) :
  ∃ (n : ℕ) (final_line : SoldierLine), 
    (n.iterate turn_step initial_line = final_line) ∧ is_stable final_line :=
  sorry

end NUMINAMATH_CALUDE_turning_process_terminates_l3285_328584


namespace NUMINAMATH_CALUDE_largest_product_of_three_l3285_328574

def S : Finset Int := {-5, -4, -1, 2, 6}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≤ 120 :=
sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l3285_328574


namespace NUMINAMATH_CALUDE_initial_players_l3285_328538

theorem initial_players (initial_players : ℕ) : 
  (∀ (players : ℕ), 
    (players = initial_players + 2) →
    (7 * players = 63)) →
  initial_players = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_players_l3285_328538


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3285_328507

/-- Given a geometric sequence {a_n} where (a_5 - a_1) / (a_3 - a_1) = 3,
    prove that (a_10 - a_2) / (a_6 + a_2) = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : (a 5 - a 1) / (a 3 - a 1) = 3)
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) :
  (a 10 - a 2) / (a 6 + a 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3285_328507


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l3285_328598

theorem six_digit_divisibility (a b : Nat) : 
  (a < 10 ∧ b < 10) →  -- Ensure a and b are single digits
  (201000 + 100 * a + 10 * b + 7) % 11 = 0 →  -- Divisible by 11
  (201000 + 100 * a + 10 * b + 7) % 13 = 0 →  -- Divisible by 13
  10 * a + b = 48 := by
sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l3285_328598


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l3285_328568

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a^3 + t.b^3 + t.c^3) / (t.a + t.b + t.c) = t.c^2

def condition2 (t : Triangle) : Prop :=
  Real.sin t.α * Real.sin t.β = (Real.sin t.γ)^2

-- Define the theorem
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) 
  (h2 : condition2 t) : 
  t.a = t.b ∧ t.b = t.c := by
  sorry


end NUMINAMATH_CALUDE_triangle_is_equilateral_l3285_328568


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3285_328533

theorem inequality_equivalence (x : ℝ) : (x - 1) / 3 > 2 ↔ x > 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3285_328533


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3285_328506

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → ℕ) :
  (∃ (d : ℕ), d ∣ (p 3 15 + p 11 21) ∧ Prime d) →
  (∀ (d : ℕ), d ∣ (p 3 15 + p 11 21) ∧ Prime d → d ≥ 2) →
  (∃ (d : ℕ), d ∣ (p 3 15 + p 11 21) ∧ Prime d ∧ d = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3285_328506


namespace NUMINAMATH_CALUDE_sequence_properties_l3285_328535

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_property (n : ℕ) : S n + n = 2 * sequence_a n

def sequence_b (n : ℕ) : ℝ := n * sequence_a n + n

def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, n > 0 → sequence_a n = 2^n - 1) ∧
  (∀ n : ℕ, n ≥ 11 → (T n - 2) / n > 2018) ∧
  (∀ n : ℕ, n < 11 → (T n - 2) / n ≤ 2018) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3285_328535


namespace NUMINAMATH_CALUDE_segment_length_l3285_328517

/-- Given a line segment AB with points P, Q, and R, prove that AB has length 567 -/
theorem segment_length (A B P Q R : Real) : 
  (P - A) / (B - P) = 3 / 4 →  -- P divides AB in ratio 3:4
  (Q - A) / (B - Q) = 4 / 5 →  -- Q divides AB in ratio 4:5
  (R - P) / (Q - R) = 1 / 2 →  -- R divides PQ in ratio 1:2
  R - P = 3 →                  -- Length of PR is 3 units
  B - A = 567 := by            -- Length of AB is 567 units
  sorry


end NUMINAMATH_CALUDE_segment_length_l3285_328517


namespace NUMINAMATH_CALUDE_janice_throw_ratio_l3285_328551

/-- The height of Christine's first throw in feet -/
def christine_first : ℕ := 20

/-- The height of Janice's first throw in feet -/
def janice_first : ℕ := christine_first - 4

/-- The height of Christine's second throw in feet -/
def christine_second : ℕ := christine_first + 10

/-- The height of Christine's third throw in feet -/
def christine_third : ℕ := christine_second + 4

/-- The height of Janice's third throw in feet -/
def janice_third : ℕ := christine_first + 17

/-- The height of the highest throw in feet -/
def highest_throw : ℕ := 37

/-- The height of Janice's second throw in feet -/
def janice_second : ℕ := 2 * janice_first

theorem janice_throw_ratio :
  janice_second = 2 * janice_first ∧
  janice_third = highest_throw ∧
  janice_second < christine_third ∧
  janice_second > janice_first :=
by sorry

#check janice_throw_ratio

end NUMINAMATH_CALUDE_janice_throw_ratio_l3285_328551


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3285_328580

-- Define the propositions
def p (x : ℝ) : Prop := -2 < x ∧ x < 0
def q (x : ℝ) : Prop := |x| < 2

-- Theorem statement
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3285_328580


namespace NUMINAMATH_CALUDE_students_without_A_l3285_328532

theorem students_without_A (total : ℕ) (history_A : ℕ) (math_A : ℕ) (both_A : ℕ) :
  total = 40 →
  history_A = 10 →
  math_A = 18 →
  both_A = 6 →
  total - ((history_A + math_A) - both_A) = 18 :=
by sorry

end NUMINAMATH_CALUDE_students_without_A_l3285_328532


namespace NUMINAMATH_CALUDE_g_neither_even_nor_odd_l3285_328577

noncomputable def g (x : ℝ) : ℝ := Real.log (x + 2 + Real.sqrt (1 + (x + 2)^2))

theorem g_neither_even_nor_odd : 
  (∀ x, g (-x) = g x) ∧ (∀ x, g (-x) = -g x) → False := by
  sorry

end NUMINAMATH_CALUDE_g_neither_even_nor_odd_l3285_328577


namespace NUMINAMATH_CALUDE_second_subject_grade_l3285_328542

theorem second_subject_grade (grade1 grade2 grade3 average : ℚ) : 
  grade1 = 50 →
  grade3 = 90 →
  average = 70 →
  (grade1 + grade2 + grade3) / 3 = average →
  grade2 = 70 := by
sorry

end NUMINAMATH_CALUDE_second_subject_grade_l3285_328542


namespace NUMINAMATH_CALUDE_ruth_total_score_l3285_328525

-- Define the given conditions
def dean_total_points : ℕ := 252
def dean_games : ℕ := 28
def games_difference : ℕ := 10
def average_difference : ℚ := 1/2

-- Define Ruth's games
def ruth_games : ℕ := dean_games - games_difference

-- Define Dean's average
def dean_average : ℚ := dean_total_points / dean_games

-- Define Ruth's average
def ruth_average : ℚ := dean_average + average_difference

-- Theorem to prove
theorem ruth_total_score : ℕ := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ruth_total_score_l3285_328525


namespace NUMINAMATH_CALUDE_negation_of_existence_l3285_328590

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l3285_328590


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3285_328566

def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

def inclination_angle (f : ℝ → ℝ → Prop) : ℝ := sorry

theorem line_inclination_angle :
  inclination_angle line_equation = π * (5/6) := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3285_328566


namespace NUMINAMATH_CALUDE_pencil_distribution_theorem_l3285_328501

/-- The number of ways to distribute pencils among friends -/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) (min_pencils : ℕ) (max_pencils : ℕ) : ℕ := 
  sorry

/-- Theorem stating the number of ways to distribute 10 pencils among 4 friends -/
theorem pencil_distribution_theorem :
  distribute_pencils 10 4 1 5 = 64 := by sorry

end NUMINAMATH_CALUDE_pencil_distribution_theorem_l3285_328501


namespace NUMINAMATH_CALUDE_certain_number_proof_l3285_328563

theorem certain_number_proof : ∃ x : ℝ, x * 16 = 3408 ∧ 16 * 21.3 = 340.8 → x = 213 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3285_328563


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l3285_328539

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (|2 * x₁ - 3| = 15) ∧ 
  (|2 * x₂ - 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 15) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l3285_328539


namespace NUMINAMATH_CALUDE_new_boarders_correct_l3285_328599

/-- The number of new boarders that joined the school -/
def new_boarders : ℕ := 15

/-- The initial number of boarders -/
def initial_boarders : ℕ := 60

/-- The initial ratio of boarders to day students -/
def initial_ratio : ℚ := 2 / 5

/-- The final ratio of boarders to day students -/
def final_ratio : ℚ := 1 / 2

/-- The theorem stating that the number of new boarders is correct -/
theorem new_boarders_correct :
  let initial_day_students := (initial_boarders : ℚ) / initial_ratio
  (initial_boarders + new_boarders : ℚ) / initial_day_students = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_new_boarders_correct_l3285_328599


namespace NUMINAMATH_CALUDE_w_squared_value_l3285_328570

theorem w_squared_value (w : ℚ) (h : (w + 16)^2 = (4*w + 9)*(3*w + 6)) : 
  w^2 = 5929 / 484 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l3285_328570


namespace NUMINAMATH_CALUDE_sum_of_roots_l3285_328543

theorem sum_of_roots (a b : ℝ) : 
  a * (a - 4) = 21 → 
  b * (b - 4) = 21 → 
  a ≠ b → 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3285_328543


namespace NUMINAMATH_CALUDE_inequality_constraint_on_a_l3285_328556

theorem inequality_constraint_on_a (a : ℝ) : 
  (∀ x : ℝ, (Real.exp x - a * x) * (x^2 - a * x + 1) ≥ 0) → 
  0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_constraint_on_a_l3285_328556


namespace NUMINAMATH_CALUDE_tangent_product_equals_two_l3285_328562

theorem tangent_product_equals_two :
  let tan17 := Real.tan (17 * π / 180)
  let tan28 := Real.tan (28 * π / 180)
  (∀ a b, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)) →
  17 + 28 = 45 →
  Real.tan (45 * π / 180) = 1 →
  (1 + tan17) * (1 + tan28) = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_product_equals_two_l3285_328562


namespace NUMINAMATH_CALUDE_more_students_than_pets_l3285_328587

theorem more_students_than_pets :
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 25
  let rabbits_per_classroom : ℕ := 3
  let guinea_pigs_per_classroom : ℕ := 3
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  let total_guinea_pigs : ℕ := num_classrooms * guinea_pigs_per_classroom
  let total_pets : ℕ := total_rabbits + total_guinea_pigs
  total_students - total_pets = 95 := by
  sorry

end NUMINAMATH_CALUDE_more_students_than_pets_l3285_328587


namespace NUMINAMATH_CALUDE_ant_count_approximation_l3285_328526

/-- Represents the dimensions and ant densities of a park -/
structure ParkInfo where
  width : ℝ
  length : ℝ
  mainDensity : ℝ
  squareSide : ℝ
  squareDensity : ℝ

/-- Calculates the total number of ants in the park -/
def totalAnts (park : ParkInfo) : ℝ :=
  let totalArea := park.width * park.length
  let squareArea := park.squareSide * park.squareSide
  let mainArea := totalArea - squareArea
  let mainAnts := mainArea * 144 * park.mainDensity  -- Convert to square inches
  let squareAnts := squareArea * park.squareDensity
  mainAnts + squareAnts

/-- The park information as given in the problem -/
def givenPark : ParkInfo := {
  width := 250
  length := 350
  mainDensity := 4
  squareSide := 50
  squareDensity := 6
}

/-- Theorem stating that the total number of ants is approximately 50 million -/
theorem ant_count_approximation :
  abs (totalAnts givenPark - 50000000) ≤ 1000000 := by
  sorry

end NUMINAMATH_CALUDE_ant_count_approximation_l3285_328526


namespace NUMINAMATH_CALUDE_tourist_assignment_count_l3285_328510

/-- The number of ways to assign tourists to scenic spots -/
def assignmentCount (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else if n = k then Nat.factorial k
  else (Nat.choose n 2) * (Nat.factorial k)

theorem tourist_assignment_count :
  assignmentCount 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_tourist_assignment_count_l3285_328510


namespace NUMINAMATH_CALUDE_perpendicular_line_tangent_cubic_l3285_328560

/-- Given a line ax - by - 2 = 0 perpendicular to the tangent of y = x^3 at (1,1), prove a/b = -1/3 -/
theorem perpendicular_line_tangent_cubic (a b : ℝ) : 
  (∀ x y : ℝ, a * x - b * y - 2 = 0 → 
    (x - 1) * (3 * (1 : ℝ)^2) + (y - 1) = 0) → 
  a / b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_tangent_cubic_l3285_328560


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l3285_328550

/-- The volume of the space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 7) (h_cylinder : r_cylinder = 4) :
  let h_cylinder := 2 * Real.sqrt 33
  let v_sphere := (4 / 3) * π * r_sphere ^ 3
  let v_cylinder := π * r_cylinder ^ 2 * h_cylinder
  v_sphere - v_cylinder = (1372 / 3 - 32 * Real.sqrt 33) * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l3285_328550


namespace NUMINAMATH_CALUDE_intersecting_sets_implies_a_equals_one_l3285_328515

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | a * x^2 - 1 = 0 ∧ a > 0}
def N : Set ℝ := {-1/2, 1/2, 1}

-- Define the "intersect" property
def intersect (A B : Set ℝ) : Prop :=
  (∃ x, x ∈ A ∧ x ∈ B) ∧ (¬(A ⊆ B) ∧ ¬(B ⊆ A))

-- State the theorem
theorem intersecting_sets_implies_a_equals_one :
  ∀ a : ℝ, intersect (M a) N → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_sets_implies_a_equals_one_l3285_328515


namespace NUMINAMATH_CALUDE_consecutive_odd_product_l3285_328524

theorem consecutive_odd_product (n : ℕ) : (2*n - 1) * (2*n + 1) = (2*n)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_l3285_328524


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3285_328527

theorem lcm_from_product_and_hcf (A B : ℕ+) :
  A * B = 84942 →
  Nat.gcd A B = 33 →
  Nat.lcm A B = 2574 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3285_328527


namespace NUMINAMATH_CALUDE_fixed_point_of_log_function_l3285_328573

-- Define the logarithm function for any base a > 0 and a ≠ 1
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f(x) = log_a(x+2) + 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 2) + 1

-- State the theorem
theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_log_function_l3285_328573


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l3285_328564

def v1 : ℝ × ℝ × ℝ := (1, 3, 4)
def v2 (k : ℝ) : ℝ × ℝ × ℝ := (2, k, 1)
def v3 (k : ℝ) : ℝ × ℝ × ℝ := (1, 1, k)

def volume (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  (a1 * b2 * c3 + a2 * b3 * c1 + a3 * b1 * c2) -
  (a3 * b2 * c1 + a1 * b3 * c2 + a2 * b1 * c3)

theorem parallelepiped_volume (k : ℝ) :
  k > 0 ∧ volume v1 (v2 k) (v3 k) = 12 ↔
  k = 5 + Real.sqrt 26 ∨ k = 5 + Real.sqrt 2 ∨ k = 5 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l3285_328564


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3285_328508

def f (x : ℝ) : ℝ := x^4 - 9*x^3 + 21*x^2 + x - 18

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, f x = (x - 4) * q x + 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3285_328508


namespace NUMINAMATH_CALUDE_ratio_p_to_q_l3285_328549

theorem ratio_p_to_q (p q : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) 
  (h3 : (p + q) / (p - q) = 4 / 3) : p / q = 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_q_l3285_328549


namespace NUMINAMATH_CALUDE_equation_solution_l3285_328559

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x^2 + 6*x + 6*x * Real.sqrt (x + 5) - 52
  ∃ (x₁ x₂ : ℝ), x₁ = (9 - Real.sqrt 5) / 2 ∧ x₂ = (9 + Real.sqrt 5) / 2 ∧
  f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3285_328559


namespace NUMINAMATH_CALUDE_min_cells_to_determine_covering_l3285_328545

/-- Represents a chessboard with its dimensions -/
structure Chessboard where
  rows : Nat
  cols : Nat

/-- Represents a domino with its dimensions -/
structure Domino where
  width : Nat
  height : Nat

/-- Calculates the total number of cells on a chessboard -/
def totalCells (board : Chessboard) : Nat :=
  board.rows * board.cols

/-- Calculates the number of cells covered by a domino -/
def cellsCoveredByDomino (domino : Domino) : Nat :=
  domino.width * domino.height

/-- Calculates the number of dominoes needed to cover the chessboard -/
def numberOfDominoes (board : Chessboard) (domino : Domino) : Nat :=
  (totalCells board) / (cellsCoveredByDomino domino)

/-- The main theorem stating the minimum number of cells needed to determine the entire covering -/
theorem min_cells_to_determine_covering (board : Chessboard) (domino : Domino)
    (h1 : board.rows = 1000)
    (h2 : board.cols = 1000)
    (h3 : domino.width = 1)
    (h4 : domino.height = 10) :
    numberOfDominoes board domino = 100000 := by
  sorry

end NUMINAMATH_CALUDE_min_cells_to_determine_covering_l3285_328545


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3285_328530

theorem quadratic_inequality_solution (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) > 0} = {x : ℝ | x < a ∨ x > 1/a} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3285_328530


namespace NUMINAMATH_CALUDE_equation_solutions_l3285_328592

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3) ∧
  (∀ y : ℝ, 2*y^2 + 4*y = y + 2 ↔ y = -2 ∨ y = 1/2) ∧
  (∀ y : ℝ, (2*y + 1)^2 - 25 = 0 ↔ y = -3 ∨ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3285_328592


namespace NUMINAMATH_CALUDE_snail_climb_problem_l3285_328567

/-- The number of days required for a snail to climb out of a well -/
def days_to_climb (well_height : ℕ) (day_climb : ℕ) (night_slip : ℕ) : ℕ :=
  sorry

theorem snail_climb_problem :
  let well_height : ℕ := 12
  let day_climb : ℕ := 3
  let night_slip : ℕ := 2
  days_to_climb well_height day_climb night_slip = 10 := by sorry

end NUMINAMATH_CALUDE_snail_climb_problem_l3285_328567


namespace NUMINAMATH_CALUDE_correct_subtraction_l3285_328558

theorem correct_subtraction (x : ℤ) (h : x - 21 = 52) : x - 40 = 33 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l3285_328558


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3285_328576

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_eq : a + b + c = 3) (prod_sum_eq : a * b + b * c + a * c = 2) :
  a + b ≥ (6 - 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3285_328576


namespace NUMINAMATH_CALUDE_base_b_square_l3285_328594

theorem base_b_square (b : ℕ) : 
  (2 * b + 4)^2 = 5 * b^2 + 5 * b + 4 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l3285_328594


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3285_328514

theorem complex_fraction_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3285_328514


namespace NUMINAMATH_CALUDE_sum_f_negative_l3285_328511

def f (x : ℝ) := -x - x^3

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l3285_328511


namespace NUMINAMATH_CALUDE_solution_uniqueness_l3285_328537

theorem solution_uniqueness (x y z : ℝ) :
  x^2 * y + y^2 * z + z^2 = 0 ∧
  z^3 + z^2 * y + z * y^3 + x^2 * y = 1/4 * (x^4 + y^4) →
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_uniqueness_l3285_328537


namespace NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l3285_328555

theorem exists_integer_sqrt_8m : ∃ m : ℕ+, ∃ k : ℕ, (8 * m.val : ℕ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l3285_328555


namespace NUMINAMATH_CALUDE_milk_ratio_is_two_fifths_l3285_328557

/-- The number of milk boxes Lolita drinks on weekdays -/
def weekday_boxes : ℕ := 3

/-- The number of milk boxes Lolita drinks on Sundays -/
def sunday_boxes : ℕ := 3 * weekday_boxes

/-- The total number of milk boxes Lolita drinks per week -/
def total_boxes : ℕ := 30

/-- The number of milk boxes Lolita drinks on Saturdays -/
def saturday_boxes : ℕ := total_boxes - (5 * weekday_boxes + sunday_boxes)

/-- The ratio of milk boxes on Saturdays to weekdays -/
def milk_ratio : ℚ := saturday_boxes / (5 * weekday_boxes)

theorem milk_ratio_is_two_fifths : milk_ratio = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_milk_ratio_is_two_fifths_l3285_328557


namespace NUMINAMATH_CALUDE_total_numbers_correction_l3285_328541

/-- Given an initial average of 15, where one number was misread as 26 instead of 36,
    and the correct average is 16, prove that the total number of numbers is 10. -/
theorem total_numbers_correction (initial_avg : ℚ) (misread : ℚ) (correct : ℚ) (correct_avg : ℚ)
  (h1 : initial_avg = 15)
  (h2 : misread = 26)
  (h3 : correct = 36)
  (h4 : correct_avg = 16) :
  ∃ (n : ℕ) (S : ℚ), n > 0 ∧ n = 10 ∧ 
    S / n + misread / n = initial_avg ∧
    S / n + correct / n = correct_avg :=
by sorry

end NUMINAMATH_CALUDE_total_numbers_correction_l3285_328541


namespace NUMINAMATH_CALUDE_line_through_points_l3285_328516

/-- A line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  eq_at_point1 : (a * point1.1 + b) = point1.2
  eq_at_point2 : (a * point2.1 + b) = point2.2

/-- Theorem stating that for a line y = ax + b passing through (2, 3) and (6, 19), a - b = 9 -/
theorem line_through_points (l : Line) 
    (h1 : l.point1 = (2, 3))
    (h2 : l.point2 = (6, 19)) : 
  l.a - l.b = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3285_328516


namespace NUMINAMATH_CALUDE_find_other_number_l3285_328569

theorem find_other_number (a b : ℕ) (ha : a = 36) 
  (hhcf : Nat.gcd a b = 20) (hlcm : Nat.lcm a b = 396) : b = 220 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3285_328569


namespace NUMINAMATH_CALUDE_decimal_multiplication_l3285_328572

theorem decimal_multiplication : (0.2 : ℝ) * 0.8 = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l3285_328572


namespace NUMINAMATH_CALUDE_coin_touch_black_probability_l3285_328586

/-- Represents the square layout with black regions -/
structure SquareLayout where
  side_length : ℝ
  corner_triangle_leg : ℝ
  center_circle_diameter : ℝ

/-- Represents a coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of the coin touching any black region -/
def probability_touch_black (layout : SquareLayout) (coin : Coin) : ℝ :=
  sorry

/-- Theorem statement for the probability problem -/
theorem coin_touch_black_probability
  (layout : SquareLayout)
  (coin : Coin)
  (h1 : layout.side_length = 6)
  (h2 : layout.corner_triangle_leg = 1)
  (h3 : layout.center_circle_diameter = 2)
  (h4 : coin.diameter = 2) :
  probability_touch_black layout coin = (2 + Real.pi) / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_touch_black_probability_l3285_328586


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourths_l3285_328512

theorem tan_seven_pi_fourths : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourths_l3285_328512


namespace NUMINAMATH_CALUDE_total_books_equals_135_l3285_328554

def first_day_books : ℕ := 54
def second_day_books : ℕ := 23
def third_day_multiplier : ℕ := 3

def total_books : ℕ :=
  first_day_books +
  (second_day_books + 1) / 2 +
  third_day_multiplier * second_day_books

theorem total_books_equals_135 : total_books = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_books_equals_135_l3285_328554


namespace NUMINAMATH_CALUDE_sequence_sum_times_three_l3285_328505

theorem sequence_sum_times_three (seq : List Nat) : 
  seq = [82, 84, 86, 88, 90, 92, 94, 96, 98, 100] →
  3 * (seq.sum) = 2730 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_times_three_l3285_328505


namespace NUMINAMATH_CALUDE_water_conservation_function_correct_water_conserved_third_year_correct_l3285_328596

/-- Represents the water conservation model for a city's tree planting program. -/
structure WaterConservationModel where
  /-- The number of trees planted annually (in millions) -/
  annual_trees : ℕ
  /-- The initial water conservation in 2009 (in million cubic meters) -/
  initial_conservation : ℕ
  /-- The water conservation by 2015 (in billion cubic meters) -/
  final_conservation : ℚ
  /-- The year considered as the first year -/
  start_year : ℕ
  /-- The year when the forest city construction will be completed -/
  end_year : ℕ

/-- The water conservation function for the city's tree planting program. -/
def water_conservation_function (model : WaterConservationModel) (x : ℚ) : ℚ :=
  (4/3) * x + (5/3)

/-- Theorem stating that the given water conservation function is correct for the model. -/
theorem water_conservation_function_correct (model : WaterConservationModel) 
  (h1 : model.annual_trees = 500)
  (h2 : model.initial_conservation = 300)
  (h3 : model.final_conservation = 11/10)
  (h4 : model.start_year = 2009)
  (h5 : model.end_year = 2015) :
  ∀ x : ℚ, 1 ≤ x ∧ x ≤ 7 →
    water_conservation_function model x = 
      (model.final_conservation - (model.initial_conservation / 1000)) / (model.end_year - model.start_year) * x + 
      (model.initial_conservation / 1000) := by
  sorry

/-- Theorem stating that the water conserved in the third year (2011) is correct. -/
theorem water_conserved_third_year_correct (model : WaterConservationModel) :
  water_conservation_function model 3 = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_water_conservation_function_correct_water_conserved_third_year_correct_l3285_328596


namespace NUMINAMATH_CALUDE_mary_dog_walking_earnings_l3285_328579

/-- 
Given:
- Mary earns $20 washing cars each month
- Mary earns D dollars walking dogs each month
- Mary saves half of her total earnings each month
- It takes Mary 5 months to save $150

Prove that D = $40
-/
theorem mary_dog_walking_earnings (D : ℝ) : 
  (5 : ℝ) * ((20 + D) / 2) = 150 → D = 40 := by sorry

end NUMINAMATH_CALUDE_mary_dog_walking_earnings_l3285_328579


namespace NUMINAMATH_CALUDE_negative_exponent_division_l3285_328552

theorem negative_exponent_division (m : ℝ) :
  (-m)^7 / (-m)^2 = -m^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_exponent_division_l3285_328552


namespace NUMINAMATH_CALUDE_pavan_journey_l3285_328519

theorem pavan_journey (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 11 →
  speed1 = 30 →
  speed2 = 25 →
  ∃ (total_distance : ℝ),
    total_distance / (2 * speed1) + total_distance / (2 * speed2) = total_time ∧
    total_distance = 300 :=
by sorry

end NUMINAMATH_CALUDE_pavan_journey_l3285_328519


namespace NUMINAMATH_CALUDE_divisibility_property_l3285_328536

theorem divisibility_property (a b c d m x y : ℤ) 
  (h1 : m = a * d - b * c)
  (h2 : Nat.gcd a.natAbs m.natAbs = 1)
  (h3 : Nat.gcd b.natAbs m.natAbs = 1)
  (h4 : Nat.gcd c.natAbs m.natAbs = 1)
  (h5 : Nat.gcd d.natAbs m.natAbs = 1)
  (h6 : m ∣ (a * x + b * y)) :
  m ∣ (c * x + d * y) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3285_328536


namespace NUMINAMATH_CALUDE_power_division_expression_simplification_l3285_328531

-- Problem 1
theorem power_division (a : ℝ) : a^6 / a^2 = a^4 := by sorry

-- Problem 2
theorem expression_simplification (m : ℝ) : m^2 * m^4 - (2*m^3)^2 = -3*m^6 := by sorry

end NUMINAMATH_CALUDE_power_division_expression_simplification_l3285_328531


namespace NUMINAMATH_CALUDE_rectangle_diagonal_triangle_l3285_328571

theorem rectangle_diagonal_triangle (a b : ℝ) (h1 : a = 10) (h2 : b = 6) :
  let c := Real.sqrt (a^2 + b^2)
  c = Real.sqrt 136 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_triangle_l3285_328571


namespace NUMINAMATH_CALUDE_range_of_a_l3285_328534

theorem range_of_a (a : ℝ) 
  (h : ∀ x ∈ Set.Icc 3 4, x^2 - 3 > a*x - a) : 
  a < 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3285_328534


namespace NUMINAMATH_CALUDE_square_area_200m_l3285_328528

/-- The area of a square with side length 200 meters is 40000 square meters. -/
theorem square_area_200m : 
  let side_length : ℝ := 200
  let area : ℝ := side_length * side_length
  area = 40000 := by sorry

end NUMINAMATH_CALUDE_square_area_200m_l3285_328528


namespace NUMINAMATH_CALUDE_aang_fish_count_l3285_328521

theorem aang_fish_count :
  ∀ (aang_fish : ℕ),
  let sokka_fish : ℕ := 5
  let toph_fish : ℕ := 12
  let total_people : ℕ := 3
  let average_fish : ℕ := 8
  (aang_fish + sokka_fish + toph_fish) / total_people = average_fish →
  aang_fish = 7 := by
sorry

end NUMINAMATH_CALUDE_aang_fish_count_l3285_328521


namespace NUMINAMATH_CALUDE_division_of_fractions_l3285_328561

theorem division_of_fractions : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l3285_328561


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3285_328595

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3285_328595


namespace NUMINAMATH_CALUDE_other_number_is_31_l3285_328553

theorem other_number_is_31 (a b : ℤ) (h1 : 3 * a + 2 * b = 140) (h2 : a = 26 ∨ b = 26) : (a = 26 ∧ b = 31) ∨ (a = 31 ∧ b = 26) :=
sorry

end NUMINAMATH_CALUDE_other_number_is_31_l3285_328553


namespace NUMINAMATH_CALUDE_factor_polynomial_l3285_328520

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = -27 * x^5 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3285_328520


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_n_value_l3285_328583

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.log (x + 1)

theorem tangent_perpendicular_implies_n_value :
  let f' : ℝ → ℝ := λ x ↦ Real.exp x + 1 / (x + 1)
  let tangent_slope : ℝ := f' 0
  let perpendicular_line_slope : ℝ → ℝ := λ n ↦ 1 / n
  ∀ n : ℝ, n ≠ 0 →
    (tangent_slope * perpendicular_line_slope n = -1) →
    n = -2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_n_value_l3285_328583


namespace NUMINAMATH_CALUDE_cos_30_tan_45_equality_l3285_328523

theorem cos_30_tan_45_equality : 2 * Real.cos (30 * π / 180) - Real.tan (45 * π / 180) = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_tan_45_equality_l3285_328523


namespace NUMINAMATH_CALUDE_min_ceiling_sum_squares_l3285_328503

theorem min_ceiling_sum_squares (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z : ℝ) 
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0) (hE : E ≠ 0) (hF : F ≠ 0) 
  (hG : G ≠ 0) (hH : H ≠ 0) (hI : I ≠ 0) (hJ : J ≠ 0) (hK : K ≠ 0) (hL : L ≠ 0) 
  (hM : M ≠ 0) (hN : N ≠ 0) (hO : O ≠ 0) (hP : P ≠ 0) (hQ : Q ≠ 0) (hR : R ≠ 0) 
  (hS : S ≠ 0) (hT : T ≠ 0) (hU : U ≠ 0) (hV : V ≠ 0) (hW : W ≠ 0) (hX : X ≠ 0) 
  (hY : Y ≠ 0) (hZ : Z ≠ 0) : 
  26 = ⌈(A^2 + B^2 + C^2 + D^2 + E^2 + F^2 + G^2 + H^2 + I^2 + J^2 + K^2 + L^2 + 
         M^2 + N^2 + O^2 + P^2 + Q^2 + R^2 + S^2 + T^2 + U^2 + V^2 + W^2 + X^2 + Y^2 + Z^2)⌉ :=
by sorry

end NUMINAMATH_CALUDE_min_ceiling_sum_squares_l3285_328503


namespace NUMINAMATH_CALUDE_ab_nonpositive_l3285_328589

theorem ab_nonpositive (a b : ℝ) : (∀ x, (2*a + b)*x - 1 ≠ 0) → a*b ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l3285_328589


namespace NUMINAMATH_CALUDE_jacob_age_problem_l3285_328529

theorem jacob_age_problem (x : ℕ) : 
  (40 + x : ℕ) = 3 * (10 + x) ∧ 
  (40 - x : ℕ) = 7 * (10 - x) → 
  x = 5 := by sorry

end NUMINAMATH_CALUDE_jacob_age_problem_l3285_328529


namespace NUMINAMATH_CALUDE_sqrt_two_inverse_plus_cos_45_l3285_328581

theorem sqrt_two_inverse_plus_cos_45 : (Real.sqrt 2)⁻¹ + Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_inverse_plus_cos_45_l3285_328581


namespace NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l3285_328502

/-- A sequence of zeros and ones -/
def BinarySequence := List Bool

/-- Count the number of (1,0) pairs with even number of digits between them -/
def countEvenPairs (seq : BinarySequence) : ℕ := sorry

/-- Count the number of (1,0) pairs with odd number of digits between them -/
def countOddPairs (seq : BinarySequence) : ℕ := sorry

/-- Theorem: In any binary sequence, the number of (1,0) pairs with even number
    of digits between them is greater than or equal to the number of (1,0) pairs
    with odd number of digits between them -/
theorem even_pairs_ge_odd_pairs (seq : BinarySequence) :
  countEvenPairs seq ≥ countOddPairs seq := by sorry

end NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l3285_328502


namespace NUMINAMATH_CALUDE_square_root_problem_l3285_328548

theorem square_root_problem (c d : ℕ) (h : 241 * c + 214 = d^2) : d = 334 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3285_328548


namespace NUMINAMATH_CALUDE_halley_21st_century_appearance_l3285_328588

/-- Represents the year of Halley's Comet's appearance -/
def halley_appearance (n : ℕ) : ℕ := 1682 + 76 * n

/-- Predicate to check if a year is in the 21st century -/
def is_21st_century (year : ℕ) : Prop := 2001 ≤ year ∧ year ≤ 2100

theorem halley_21st_century_appearance :
  ∃ n : ℕ, is_21st_century (halley_appearance n) ∧ halley_appearance n = 2062 :=
sorry

end NUMINAMATH_CALUDE_halley_21st_century_appearance_l3285_328588


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l3285_328582

theorem stratified_sampling_sample_size (grade10 grade11 grade12 : ℕ) (n : ℕ) : 
  grade10 + grade11 + grade12 > 0 →
  grade10 = 2 * grade12 / 5 →
  grade11 = 3 * grade12 / 5 →
  (150 : ℝ) / n = (grade12 : ℝ) / (grade10 + grade11 + grade12) →
  n = 300 := by
sorry


end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l3285_328582


namespace NUMINAMATH_CALUDE_network_connections_l3285_328565

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l3285_328565


namespace NUMINAMATH_CALUDE_middle_trapezoid_radius_l3285_328518

/-- Given a trapezoid divided into three similar trapezoids by lines parallel to the bases,
    each with an inscribed circle, this theorem proves that the radius of the middle circle
    is the geometric mean of the radii of the other two circles. -/
theorem middle_trapezoid_radius (R r x : ℝ) 
  (h_positive : R > 0 ∧ r > 0 ∧ x > 0) 
  (h_similar : r / x = x / R) : 
  x = Real.sqrt (r * R) := by
sorry

end NUMINAMATH_CALUDE_middle_trapezoid_radius_l3285_328518


namespace NUMINAMATH_CALUDE_parabola_and_triangle_area_l3285_328540

/-- Parabola C: y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on the parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * c.p * x

/-- Circle E: (x-1)² + y² = 1 -/
def CircleE (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

/-- Theorem about the parabola equation and minimum area of triangle -/
theorem parabola_and_triangle_area (c : Parabola) (m : PointOnParabola c)
    (h_dist : (m.x - 2)^2 + m.y^2 = 3) (h_x : m.x > 2) :
    c.p = 1 ∧ ∃ (a b : ℝ), CircleE 0 a ∧ CircleE 0 b ∧
    (∀ a' b' : ℝ, CircleE 0 a' ∧ CircleE 0 b' →
      1/2 * |a - b| * m.x ≤ 1/2 * |a' - b'| * m.x) ∧
    1/2 * |a - b| * m.x = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_triangle_area_l3285_328540


namespace NUMINAMATH_CALUDE_coefficient_x4_eq_21_l3285_328585

/-- The coefficient of x^4 in the binomial expansion of (x+1/x-1)^6 -/
def coefficient_x4 : ℕ :=
  (Nat.choose 6 0) * (Nat.choose 6 1) + (Nat.choose 6 2) * (Nat.choose 4 0)

/-- Theorem stating that the coefficient of x^4 in the binomial expansion of (x+1/x-1)^6 is 21 -/
theorem coefficient_x4_eq_21 : coefficient_x4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_eq_21_l3285_328585


namespace NUMINAMATH_CALUDE_chord_length_squared_l3285_328504

/-- Given three circles with radii 6, 9, and 15, where the circles with radii 6 and 9
    are externally tangent to each other and internally tangent to the circle with radius 15,
    this theorem states that the square of the length of the chord of the circle with radius 15,
    which is a common external tangent to the other two circles, is equal to 692.64. -/
theorem chord_length_squared (r₁ r₂ R : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 9) (h₃ : R = 15)
  (h₄ : r₁ + r₂ = R - r₁ - r₂) : -- Condition for external tangency of smaller circles and internal tangency with larger circle
  (2 * R * ((r₁ * r₂) / (r₁ + r₂)))^2 = 692.64 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l3285_328504


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l3285_328544

/-- The first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 1 + (n - 1) * n / 2

/-- The last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set of the sequence -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem stating that the sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l3285_328544


namespace NUMINAMATH_CALUDE_a_plus_b_equals_zero_l3285_328547

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M (a : ℝ) : Set ℝ := {x | x^2 + a*x ≤ 0}

-- Define the complement of M in U
def C_U_M (a b : ℝ) : Set ℝ := {x | x > b ∨ x < 0}

-- Theorem statement
theorem a_plus_b_equals_zero (a b : ℝ) : 
  (∀ x, x ∈ M a ↔ x ∉ C_U_M a b) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_zero_l3285_328547


namespace NUMINAMATH_CALUDE_noah_painting_sales_l3285_328597

/-- Noah's painting sales calculation -/
theorem noah_painting_sales :
  let large_price : ℕ := 60
  let small_price : ℕ := 30
  let last_month_large : ℕ := 8
  let last_month_small : ℕ := 4
  let this_month_multiplier : ℕ := 2
  
  (large_price * last_month_large + small_price * last_month_small) * this_month_multiplier = 1200 :=
by sorry

end NUMINAMATH_CALUDE_noah_painting_sales_l3285_328597


namespace NUMINAMATH_CALUDE_linear_function_theorem_l3285_328522

/-- A linear function that intersects the x-axis at (-2, 0) and forms a triangle with area 8 with the coordinate axes -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  x_intercept : k * (-2) + b = 0
  triangle_area : |k * 2 * b / 2| = 8

/-- The two possible linear functions satisfying the given conditions -/
def possible_functions : Set LinearFunction :=
  { f | f.k = 4 ∧ f.b = 8 } ∪ { f | f.k = -4 ∧ f.b = -8 }

/-- Theorem stating that the only linear functions satisfying the conditions are y = 4x + 8 or y = -4x - 8 -/
theorem linear_function_theorem :
  ∀ f : LinearFunction, f ∈ possible_functions :=
by sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l3285_328522


namespace NUMINAMATH_CALUDE_complex_power_eight_l3285_328578

theorem complex_power_eight : (Complex.I + 1 : ℂ) ^ 8 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_eight_l3285_328578


namespace NUMINAMATH_CALUDE_greatest_third_term_arithmetic_sequence_l3285_328546

theorem greatest_third_term_arithmetic_sequence :
  ∀ (a d : ℕ+), 
  (a : ℕ) + (a + d : ℕ) + (a + 2 * d : ℕ) + (a + 3 * d : ℕ) = 58 →
  ∀ (b e : ℕ+),
  (b : ℕ) + (b + e : ℕ) + (b + 2 * e : ℕ) + (b + 3 * e : ℕ) = 58 →
  (a + 2 * d : ℕ) ≤ 19 :=
by sorry

end NUMINAMATH_CALUDE_greatest_third_term_arithmetic_sequence_l3285_328546


namespace NUMINAMATH_CALUDE_calculator_display_after_50_presses_l3285_328593

def calculator_function (x : ℚ) : ℚ := 1 / (1 - x)

def iterate_function (f : ℚ → ℚ) (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_function f x n)

theorem calculator_display_after_50_presses :
  iterate_function calculator_function (1/2) 50 = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculator_display_after_50_presses_l3285_328593


namespace NUMINAMATH_CALUDE_committee_formation_l3285_328575

theorem committee_formation (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_l3285_328575
