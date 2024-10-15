import Mathlib

namespace NUMINAMATH_CALUDE_cylinder_radius_l3841_384125

theorem cylinder_radius (h : ℝ) (r : ℝ) : 
  h = 4 → 
  π * (r + 10)^2 * h = π * r^2 * (h + 10) → 
  r = 4 + 2 * Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_l3841_384125


namespace NUMINAMATH_CALUDE_combined_output_in_five_minutes_l3841_384151

/-- The rate at which Machine A fills boxes (boxes per minute) -/
def machine_a_rate : ℚ := 24 / 60

/-- The rate at which Machine B fills boxes (boxes per minute) -/
def machine_b_rate : ℚ := 36 / 60

/-- The combined rate of both machines (boxes per minute) -/
def combined_rate : ℚ := machine_a_rate + machine_b_rate

/-- The time period we're interested in (minutes) -/
def time_period : ℚ := 5

theorem combined_output_in_five_minutes :
  combined_rate * time_period = 5 := by sorry

end NUMINAMATH_CALUDE_combined_output_in_five_minutes_l3841_384151


namespace NUMINAMATH_CALUDE_sum_after_operations_l3841_384129

/-- Given two numbers x and y whose sum is T, prove that if 5 is added to each number
    and then each resulting number is tripled, the sum of the final two numbers is 3T + 30. -/
theorem sum_after_operations (x y T : ℝ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_operations_l3841_384129


namespace NUMINAMATH_CALUDE_cos_2B_gt_cos_2A_necessary_not_sufficient_l3841_384138

-- Define a structure for a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the main theorem
theorem cos_2B_gt_cos_2A_necessary_not_sufficient (t : Triangle) :
  (∀ t : Triangle, t.A > t.B → Real.cos (2 * t.B) > Real.cos (2 * t.A)) ∧
  ¬(∀ t : Triangle, Real.cos (2 * t.B) > Real.cos (2 * t.A) → t.A > t.B) := by
  sorry

end NUMINAMATH_CALUDE_cos_2B_gt_cos_2A_necessary_not_sufficient_l3841_384138


namespace NUMINAMATH_CALUDE_solve_for_z_l3841_384156

theorem solve_for_z (x z : ℝ) 
  (h1 : x = 102) 
  (h2 : x^4*z - 3*x^3*z + 2*x^2*z = 1075648000) : 
  z = 1.024 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l3841_384156


namespace NUMINAMATH_CALUDE_remainder_problem_l3841_384126

theorem remainder_problem (n : ℤ) (h : n % 7 = 2) : (5 * n + 3) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3841_384126


namespace NUMINAMATH_CALUDE_divisibility_by_eighteen_l3841_384139

theorem divisibility_by_eighteen (n : Nat) : n ≤ 9 → (315 * 10 + n) % 18 = 0 ↔ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eighteen_l3841_384139


namespace NUMINAMATH_CALUDE_fraction_equality_l3841_384148

theorem fraction_equality (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a / b + b / a = 4) :
  (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3841_384148


namespace NUMINAMATH_CALUDE_vector_equation_sum_l3841_384178

/-- Given vectors a, b, c in R², if a = xb + yc for some real x and y, then x + y = 0 -/
theorem vector_equation_sum (a b c : Fin 2 → ℝ)
    (ha : a = ![3, -1])
    (hb : b = ![-1, 2])
    (hc : c = ![2, 1])
    (x y : ℝ)
    (h : a = x • b + y • c) :
  x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_sum_l3841_384178


namespace NUMINAMATH_CALUDE_a_4k_plus_2_div_3_l3841_384107

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => a (n + 2) + a (n + 1)

theorem a_4k_plus_2_div_3 (k : ℕ) : ∃ m : ℕ, a (4 * k + 2) = 3 * m := by
  sorry

end NUMINAMATH_CALUDE_a_4k_plus_2_div_3_l3841_384107


namespace NUMINAMATH_CALUDE_total_drawing_time_l3841_384146

/-- Given Bianca's and Lucas's drawing times, prove their total drawing time is 86 minutes -/
theorem total_drawing_time 
  (bianca_school : Nat) 
  (bianca_home : Nat)
  (lucas_school : Nat)
  (lucas_home : Nat)
  (h1 : bianca_school = 22)
  (h2 : bianca_home = 19)
  (h3 : lucas_school = 10)
  (h4 : lucas_home = 35) :
  bianca_school + bianca_home + lucas_school + lucas_home = 86 := by
  sorry

#check total_drawing_time

end NUMINAMATH_CALUDE_total_drawing_time_l3841_384146


namespace NUMINAMATH_CALUDE_shirt_price_l3841_384185

/-- The cost of a pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of a shirt -/
def shirt_cost : ℝ := sorry

/-- The total cost of 3 pairs of jeans and 2 shirts is $69 -/
axiom first_purchase : 3 * jeans_cost + 2 * shirt_cost = 69

/-- The total cost of 2 pairs of jeans and 3 shirts is $86 -/
axiom second_purchase : 2 * jeans_cost + 3 * shirt_cost = 86

/-- The cost of one shirt is $24 -/
theorem shirt_price : shirt_cost = 24 := by sorry

end NUMINAMATH_CALUDE_shirt_price_l3841_384185


namespace NUMINAMATH_CALUDE_hash_equals_100_l3841_384100

def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_equals_100 (a b : ℕ) (h : a + b + 6 = 11) : hash a b = 100 := by
  sorry

end NUMINAMATH_CALUDE_hash_equals_100_l3841_384100


namespace NUMINAMATH_CALUDE_inverse_of_A_l3841_384171

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem inverse_of_A :
  let inv_A : Matrix (Fin 2) (Fin 2) ℝ := !![-1, -3; -2, -5]
  Matrix.det A ≠ 0 → A * inv_A = 1 ∧ inv_A * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_l3841_384171


namespace NUMINAMATH_CALUDE_uncoverable_3x7_and_7x3_other_boards_coverable_l3841_384119

/-- A board configuration -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)
  (removed : ℕ)

/-- Checks if a board can be completely covered by dominoes -/
def can_cover (b : Board) : Prop :=
  (b.rows * b.cols - b.removed) % 2 = 0

/-- Theorem: A 3x7 or 7x3 board cannot be completely covered by dominoes -/
theorem uncoverable_3x7_and_7x3 :
  ¬(can_cover ⟨3, 7, 0⟩) ∧ ¬(can_cover ⟨7, 3, 0⟩) :=
sorry

/-- Theorem: All other given board configurations can be covered by dominoes -/
theorem other_boards_coverable :
  can_cover ⟨2, 3, 0⟩ ∧
  can_cover ⟨4, 4, 4⟩ ∧
  can_cover ⟨5, 5, 1⟩ :=
sorry

end NUMINAMATH_CALUDE_uncoverable_3x7_and_7x3_other_boards_coverable_l3841_384119


namespace NUMINAMATH_CALUDE_average_problem_l3841_384163

theorem average_problem (t b c : ℝ) :
  (t + b + c + 14 + 15) / 5 = 12 →
  (t + b + c + 29) / 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l3841_384163


namespace NUMINAMATH_CALUDE_jackson_keeps_120_lollipops_l3841_384132

/-- The number of lollipops Jackson keeps for himself -/
def lollipops_kept (apple banana cherry dragon_fruit : ℕ) (friends : ℕ) : ℕ :=
  cherry

/-- Theorem stating that Jackson keeps 120 lollipops for himself -/
theorem jackson_keeps_120_lollipops :
  lollipops_kept 53 62 120 15 13 = 120 := by
  sorry

#eval lollipops_kept 53 62 120 15 13

end NUMINAMATH_CALUDE_jackson_keeps_120_lollipops_l3841_384132


namespace NUMINAMATH_CALUDE_early_arrival_time_l3841_384189

/-- 
Given a boy's usual time to reach school and his faster rate relative to his usual rate,
calculate how many minutes earlier he arrives when walking at the faster rate.
-/
theorem early_arrival_time (usual_time : ℝ) (faster_rate_ratio : ℝ) 
  (h1 : usual_time = 28)
  (h2 : faster_rate_ratio = 7/6) : 
  usual_time - (usual_time / faster_rate_ratio) = 4 := by
  sorry

end NUMINAMATH_CALUDE_early_arrival_time_l3841_384189


namespace NUMINAMATH_CALUDE_count_sevens_to_2017_l3841_384127

/-- Count of occurrences of a digit in a range of natural numbers -/
def countDigitOccurrences (digit : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The main theorem stating that the count of 7's from 1 to 2017 is 602 -/
theorem count_sevens_to_2017 : countDigitOccurrences 7 1 2017 = 602 := by
  sorry

end NUMINAMATH_CALUDE_count_sevens_to_2017_l3841_384127


namespace NUMINAMATH_CALUDE_expression_evaluation_l3841_384159

theorem expression_evaluation : (36 + 12) / (6 - (2 + 1)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3841_384159


namespace NUMINAMATH_CALUDE_jims_paycheck_l3841_384172

def gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def tax_deduction : ℝ := 100

def retirement_deduction : ℝ := gross_pay * retirement_rate

def net_pay : ℝ := gross_pay - retirement_deduction - tax_deduction

theorem jims_paycheck : net_pay = 740 := by
  sorry

end NUMINAMATH_CALUDE_jims_paycheck_l3841_384172


namespace NUMINAMATH_CALUDE_mean_height_is_70_74_l3841_384147

def player_heights : List ℕ := [58, 59, 60, 61, 62, 63, 65, 65, 68, 70, 71, 74, 76, 78, 79, 81, 83, 85, 86]

def mean_height (heights : List ℕ) : ℚ :=
  (heights.sum : ℚ) / heights.length

theorem mean_height_is_70_74 :
  mean_height player_heights = 70.74 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_is_70_74_l3841_384147


namespace NUMINAMATH_CALUDE_charlie_win_probability_l3841_384186

/-- The probability of rolling a six on a standard six-sided die -/
def probSix : ℚ := 1 / 6

/-- The probability of not rolling a six on a standard six-sided die -/
def probNotSix : ℚ := 5 / 6

/-- The number of players in the game -/
def numPlayers : ℕ := 3

/-- The probability that Charlie (the third player) wins the dice game -/
def probCharlieWins : ℚ := 125 / 546

theorem charlie_win_probability :
  probCharlieWins = probSix * (probNotSix^numPlayers / (1 - probNotSix^numPlayers)) :=
sorry

end NUMINAMATH_CALUDE_charlie_win_probability_l3841_384186


namespace NUMINAMATH_CALUDE_irrational_floor_congruence_l3841_384133

theorem irrational_floor_congruence (k : ℕ) (h : k ≥ 2) :
  ∃ r : ℝ, Irrational r ∧ ∀ m : ℕ, (⌊r^m⌋ : ℤ) ≡ -1 [ZMOD k] :=
sorry

end NUMINAMATH_CALUDE_irrational_floor_congruence_l3841_384133


namespace NUMINAMATH_CALUDE_doctors_who_quit_correct_number_of_doctors_quit_l3841_384175

theorem doctors_who_quit (initial_doctors : ℕ) (initial_nurses : ℕ) 
  (nurses_quit : ℕ) (final_total : ℕ) : ℕ :=
  let doctors_quit := initial_doctors + initial_nurses - nurses_quit - final_total
  doctors_quit

theorem correct_number_of_doctors_quit : 
  doctors_who_quit 11 18 2 22 = 5 := by sorry

end NUMINAMATH_CALUDE_doctors_who_quit_correct_number_of_doctors_quit_l3841_384175


namespace NUMINAMATH_CALUDE_six_year_olds_count_l3841_384111

/-- Represents the number of children in each age group -/
structure AgeGroups where
  three_year_olds : ℕ
  four_year_olds : ℕ
  five_year_olds : ℕ
  six_year_olds : ℕ

/-- Represents the Sunday school with its age groups and class information -/
structure SundaySchool where
  ages : AgeGroups
  avg_class_size : ℕ
  num_classes : ℕ

def SundaySchool.total_children (s : SundaySchool) : ℕ :=
  s.ages.three_year_olds + s.ages.four_year_olds + s.ages.five_year_olds + s.ages.six_year_olds

theorem six_year_olds_count (s : SundaySchool) 
  (h1 : s.ages.three_year_olds = 13)
  (h2 : s.ages.four_year_olds = 20)
  (h3 : s.ages.five_year_olds = 15)
  (h4 : s.avg_class_size = 35)
  (h5 : s.num_classes = 2)
  : s.ages.six_year_olds = 22 := by
  sorry

#check six_year_olds_count

end NUMINAMATH_CALUDE_six_year_olds_count_l3841_384111


namespace NUMINAMATH_CALUDE_neg_one_quad_residue_iff_prime_mod_four_infinite_primes_mod_four_l3841_384181

/-- For an odd prime p, -1 is a quadratic residue modulo p if and only if p ≡ 1 (mod 4) -/
theorem neg_one_quad_residue_iff_prime_mod_four (p : Nat) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∃ x, x^2 % p = (p - 1) % p) ↔ p % 4 = 1 := by sorry

/-- There are infinitely many prime numbers congruent to 1 modulo 4 -/
theorem infinite_primes_mod_four :
  ∀ n, ∃ p, p > n ∧ Nat.Prime p ∧ p % 4 = 1 := by sorry

end NUMINAMATH_CALUDE_neg_one_quad_residue_iff_prime_mod_four_infinite_primes_mod_four_l3841_384181


namespace NUMINAMATH_CALUDE_optimal_triangle_count_l3841_384190

/-- A configuration of points in space --/
structure PointConfiguration where
  total_points : Nat
  num_groups : Nat
  group_sizes : Fin num_groups → Nat
  non_collinear : Bool
  different_sizes : ∀ i j, i ≠ j → group_sizes i ≠ group_sizes j

/-- The number of triangles formed by selecting one point from each of three different groups --/
def num_triangles (config : PointConfiguration) : Nat :=
  sorry

/-- The optimal configuration for maximizing the number of triangles --/
def optimal_config : PointConfiguration where
  total_points := 1989
  num_groups := 30
  group_sizes := fun i => 
    if i.val < 6 then 51 + i.val
    else if i.val = 6 then 58
    else 59 + i.val - 7
  non_collinear := true
  different_sizes := sorry

theorem optimal_triangle_count (config : PointConfiguration) :
  config.total_points = 1989 →
  config.num_groups = 30 →
  config.non_collinear = true →
  num_triangles config ≤ num_triangles optimal_config :=
sorry

end NUMINAMATH_CALUDE_optimal_triangle_count_l3841_384190


namespace NUMINAMATH_CALUDE_u_2008_eq_4008_l3841_384167

/-- Defines the sequence u_n as described in the problem -/
def u : ℕ → ℕ :=
  sorry

/-- Theorem stating that the 2008th term of the sequence is 4008 -/
theorem u_2008_eq_4008 : u 2008 = 4008 := by
  sorry

end NUMINAMATH_CALUDE_u_2008_eq_4008_l3841_384167


namespace NUMINAMATH_CALUDE_complement_of_union_equals_interval_l3841_384193

-- Define the universal set U
def U : Set ℝ := {x | -5 < x ∧ x < 5}

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- State the theorem
theorem complement_of_union_equals_interval :
  (U \ (A ∪ B)) = {x | -5 < x ∧ x ≤ -2} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_interval_l3841_384193


namespace NUMINAMATH_CALUDE_min_people_for_condition_l3841_384104

/-- Represents a circular table with chairs and people seated. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any additional
    person must sit next to at least one other person. -/
def satisfies_condition (table : CircularTable) : Prop :=
  table.seated_people * 4 ≥ table.total_chairs

/-- The theorem stating the minimum number of people required for the given condition. -/
theorem min_people_for_condition (table : CircularTable) 
  (h1 : table.total_chairs = 80)
  (h2 : satisfies_condition table)
  (h3 : ∀ n < table.seated_people, ¬satisfies_condition ⟨table.total_chairs, n⟩) :
  table.seated_people = 20 := by
  sorry

#check min_people_for_condition

end NUMINAMATH_CALUDE_min_people_for_condition_l3841_384104


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l3841_384108

/-- Given an initial angle of 30 degrees rotated 450 degrees clockwise,
    the resulting new acute angle measures 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 30 →
  rotation = 450 →
  let effective_rotation := rotation % 360
  let new_angle := (initial_angle - effective_rotation) % 360
  let acute_angle := min new_angle (360 - new_angle)
  acute_angle = 60 := by
  sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l3841_384108


namespace NUMINAMATH_CALUDE_distance_to_gate_l3841_384140

theorem distance_to_gate (field_side : ℝ) (fence_length : ℝ) (gate_distance : ℝ) :
  field_side = 84 →
  fence_length = 91 →
  gate_distance^2 + field_side^2 = fence_length^2 →
  gate_distance = 35 := by
sorry

end NUMINAMATH_CALUDE_distance_to_gate_l3841_384140


namespace NUMINAMATH_CALUDE_tire_circumference_l3841_384174

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference (rotation_speed : ℝ) (car_velocity : ℝ) : 
  rotation_speed = 400 ∧ car_velocity = 96 → 
  (car_velocity * 1000 / 60) / rotation_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_tire_circumference_l3841_384174


namespace NUMINAMATH_CALUDE_july_birth_percentage_l3841_384199

theorem july_birth_percentage (total : ℕ) (july_births : ℕ) 
  (h1 : total = 120) (h2 : july_births = 16) : 
  (july_births : ℝ) / total * 100 = 13.33 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l3841_384199


namespace NUMINAMATH_CALUDE_arithmetic_sequences_prime_term_l3841_384184

/-- Two arithmetic sequences with their sums -/
def ArithmeticSequences (a b : ℕ → ℕ) (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → S n / T n = (2 * n + 6 : ℚ) / (n + 1 : ℚ)

/-- The m-th term of the second sequence is prime -/
def SecondSequencePrimeTerm (b : ℕ → ℕ) (m : ℕ) : Prop :=
  m > 0 ∧ Nat.Prime (b m)

theorem arithmetic_sequences_prime_term 
  (a b : ℕ → ℕ) (S T : ℕ → ℚ) (m : ℕ) :
  ArithmeticSequences a b S T →
  SecondSequencePrimeTerm b m →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_prime_term_l3841_384184


namespace NUMINAMATH_CALUDE_player_a_not_losing_probability_l3841_384168

theorem player_a_not_losing_probability 
  (p_win : ℝ) 
  (p_draw : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_draw = 0.2) : 
  p_win + p_draw = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_player_a_not_losing_probability_l3841_384168


namespace NUMINAMATH_CALUDE_steps_between_correct_l3841_384164

/-- The number of steps taken from the Empire State Building to Madison Square Garden -/
def steps_between (total_steps down_steps : ℕ) : ℕ :=
  total_steps - down_steps

/-- Theorem stating that the steps between buildings is the difference of total steps and steps down -/
theorem steps_between_correct (total_steps down_steps : ℕ) 
  (h1 : total_steps ≥ down_steps)
  (h2 : total_steps = 991)
  (h3 : down_steps = 676) : 
  steps_between total_steps down_steps = 315 := by
  sorry

end NUMINAMATH_CALUDE_steps_between_correct_l3841_384164


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3841_384145

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3841_384145


namespace NUMINAMATH_CALUDE_p_div_q_eq_7371_l3841_384187

/-- The number of balls -/
def n : ℕ := 30

/-- The number of bins -/
def k : ℕ := 10

/-- The probability that two bins have 2 balls each and eight bins have 3 balls each -/
noncomputable def p : ℝ := (Nat.choose k 2) * (Nat.choose n 2) * (Nat.choose (n - 2) 2) * 
  (Nat.choose (n - 4) 3) * (Nat.choose (n - 7) 3) * (Nat.choose (n - 10) 3) * 
  (Nat.choose (n - 13) 3) * (Nat.choose (n - 16) 3) * (Nat.choose (n - 19) 3) * 
  (Nat.choose (n - 22) 3) * (Nat.choose (n - 25) 3) / (k ^ n)

/-- The probability that every bin has 3 balls -/
noncomputable def q : ℝ := (Nat.choose n 3) * (Nat.choose (n - 3) 3) * (Nat.choose (n - 6) 3) * 
  (Nat.choose (n - 9) 3) * (Nat.choose (n - 12) 3) * (Nat.choose (n - 15) 3) * 
  (Nat.choose (n - 18) 3) * (Nat.choose (n - 21) 3) * (Nat.choose (n - 24) 3) * 
  (Nat.choose (n - 27) 3) / (k ^ n)

/-- The theorem stating that the ratio of p to q is 7371 -/
theorem p_div_q_eq_7371 : p / q = 7371 := by
  sorry

end NUMINAMATH_CALUDE_p_div_q_eq_7371_l3841_384187


namespace NUMINAMATH_CALUDE_min_v_value_l3841_384143

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the translated function g
def g (x u v : ℝ) : ℝ := (x-u)^3 - 3*(x-u) - v

-- Theorem statement
theorem min_v_value (u : ℝ) (h : u > 0) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = g x₁ u v → f x₂ ≠ g x₂ u v) →
  v ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_v_value_l3841_384143


namespace NUMINAMATH_CALUDE_solve_for_y_l3841_384196

theorem solve_for_y (x y : ℤ) (h1 : x - y = 12) (h2 : x + y = 6) : y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3841_384196


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l3841_384166

theorem number_puzzle_solution :
  ∃ (A B C D E : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    A > 5 ∧
    A % B = 0 ∧
    C + A = D ∧
    B + C + E = A ∧
    B + C < E ∧
    C + E < B + 5 ∧
    A = 8 ∧ B = 2 ∧ C = 1 ∧ D = 9 ∧ E = 5 :=
by sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l3841_384166


namespace NUMINAMATH_CALUDE_no_eulerian_path_in_problem_graph_l3841_384162

/-- A region in the planar graph --/
structure Region where
  edges : ℕ

/-- A planar graph representation --/
structure PlanarGraph where
  regions : List Region
  total_edges : ℕ

/-- Check if a planar graph has an Eulerian path --/
def has_eulerian_path (g : PlanarGraph) : Prop :=
  (g.regions.filter (λ r => r.edges % 2 = 1)).length ≤ 2

/-- The specific planar graph from the problem --/
def problem_graph : PlanarGraph :=
  { regions := [
      { edges := 5 },
      { edges := 5 },
      { edges := 4 },
      { edges := 5 },
      { edges := 4 },
      { edges := 4 },
      { edges := 4 }
    ],
    total_edges := 16
  }

theorem no_eulerian_path_in_problem_graph :
  ¬ (has_eulerian_path problem_graph) :=
by sorry

end NUMINAMATH_CALUDE_no_eulerian_path_in_problem_graph_l3841_384162


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3841_384113

/-- Given real numbers m and n where m < n, the quadratic inequality
    x^2 - (m + n)x + mn > 0 has the solution set (-∞, m) ∪ (n, +∞),
    and specifying a = 1 makes this representation unique. -/
theorem quadratic_inequality_solution_set (m n : ℝ) (h : m < n) :
  ∃ (a b c : ℝ), a = 1 ∧
    (∀ x, a * x^2 + b * x + c > 0 ↔ x < m ∨ x > n) ∧
    (b = -(m + n) ∧ c = m * n) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3841_384113


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l3841_384182

theorem sum_mod_thirteen : (9753 + 9754 + 9755 + 9756) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l3841_384182


namespace NUMINAMATH_CALUDE_book_sale_revenue_l3841_384155

theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) : 
  total_books > 0 ∧ 
  price_per_book > 0 ∧
  (total_books : ℚ) / 3 = 30 ∧ 
  price_per_book = 17/4 → 
  (2 : ℚ) / 3 * (total_books : ℚ) * price_per_book = 255 := by
sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l3841_384155


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_l3841_384101

theorem raffle_ticket_sales (total_money : ℕ) (ticket_cost : ℕ) (num_tickets : ℕ) :
  total_money = 620 →
  ticket_cost = 4 →
  total_money = ticket_cost * num_tickets →
  num_tickets = 155 := by
sorry

end NUMINAMATH_CALUDE_raffle_ticket_sales_l3841_384101


namespace NUMINAMATH_CALUDE_range_of_a_l3841_384160

theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - 8*x - 20 ≤ 0 → x^2 - 2*x + 1 - a^2 ≤ 0) ∧ 
  (∃ x, x^2 - 8*x - 20 ≤ 0 ∧ x^2 - 2*x + 1 - a^2 > 0) ∧
  a > 0 → 
  a ≥ 9 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3841_384160


namespace NUMINAMATH_CALUDE_square_circle_union_area_l3841_384109

/-- The area of the union of a square with side length 12 and a circle with radius 12
    centered at one of the square's vertices is equal to 144 + 108π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let quarter_circle_area : ℝ := circle_area / 4
  square_area + circle_area - quarter_circle_area = 144 + 108 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l3841_384109


namespace NUMINAMATH_CALUDE_tiffany_album_distribution_l3841_384197

/-- Calculates the number of pictures in each album given the total number of pictures and the number of albums. -/
def pictures_per_album (phone_pics camera_pics num_albums : ℕ) : ℕ :=
  (phone_pics + camera_pics) / num_albums

/-- Proves that given the conditions in the problem, the number of pictures in each album is 4. -/
theorem tiffany_album_distribution :
  let phone_pics := 7
  let camera_pics := 13
  let num_albums := 5
  pictures_per_album phone_pics camera_pics num_albums = 4 := by
sorry

#eval pictures_per_album 7 13 5

end NUMINAMATH_CALUDE_tiffany_album_distribution_l3841_384197


namespace NUMINAMATH_CALUDE_angle_BDC_is_20_l3841_384128

-- Define the angles in degrees
def angle_A : ℝ := 70
def angle_E : ℝ := 50
def angle_C : ℝ := 40

-- Theorem to prove
theorem angle_BDC_is_20 : 
  ∃ (angle_BDC : ℝ), angle_BDC = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_BDC_is_20_l3841_384128


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3841_384179

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 205) :
  a * b = 2460 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3841_384179


namespace NUMINAMATH_CALUDE_triangle_237_not_exists_triangle_555_exists_l3841_384194

/-- Triangle inequality theorem checker -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: A triangle with sides 2, 3, and 7 does not satisfy the triangle inequality -/
theorem triangle_237_not_exists : ¬ (satisfies_triangle_inequality 2 3 7) :=
sorry

/-- Theorem: A triangle with sides 5, 5, and 5 satisfies the triangle inequality -/
theorem triangle_555_exists : satisfies_triangle_inequality 5 5 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_237_not_exists_triangle_555_exists_l3841_384194


namespace NUMINAMATH_CALUDE_bearing_and_ring_problem_l3841_384176

theorem bearing_and_ring_problem :
  ∃ (x y : ℕ),
    (x = 25 ∧ y = 16) ∨ (x = 16 ∧ y = 25) ∧
    (x : ℤ) + 2 = y ∧
    (y : ℤ) = x + 2 ∧
    x * ((y : ℤ) - 2) + y * (x + 2) - 800 = 2 * (y - x) ∧
    x * x + y * y = 881 :=
by sorry

end NUMINAMATH_CALUDE_bearing_and_ring_problem_l3841_384176


namespace NUMINAMATH_CALUDE_color_change_probability_l3841_384114

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDurations where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of observing a color change in a traffic light cycle -/
def probabilityOfColorChange (d : TrafficLightDurations) : ℚ :=
  let totalCycleDuration := d.green + d.yellow + d.red
  let changeWindowDuration := 3 * d.yellow
  changeWindowDuration / totalCycleDuration

/-- The main theorem stating the probability of observing a color change -/
theorem color_change_probability (d : TrafficLightDurations) 
  (h1 : d.green = 45)
  (h2 : d.yellow = 5)
  (h3 : d.red = 45) :
  probabilityOfColorChange d = 3 / 19 := by
  sorry

#eval probabilityOfColorChange { green := 45, yellow := 5, red := 45 }

end NUMINAMATH_CALUDE_color_change_probability_l3841_384114


namespace NUMINAMATH_CALUDE_PQRS_equals_negative_one_l3841_384135

theorem PQRS_equals_negative_one :
  let P : ℝ := Real.sqrt 2007 + Real.sqrt 2008
  let Q : ℝ := -Real.sqrt 2007 - Real.sqrt 2008
  let R : ℝ := Real.sqrt 2007 - Real.sqrt 2008
  let S : ℝ := -Real.sqrt 2008 + Real.sqrt 2007
  P * Q * R * S = -1 := by sorry

end NUMINAMATH_CALUDE_PQRS_equals_negative_one_l3841_384135


namespace NUMINAMATH_CALUDE_partnership_profit_l3841_384169

/-- A partnership business problem -/
theorem partnership_profit (investment_ratio : ℕ) (time_ratio : ℕ) (profit_B : ℕ) : 
  investment_ratio = 5 →
  time_ratio = 3 →
  profit_B = 4000 →
  investment_ratio * time_ratio * profit_B + profit_B = 64000 :=
by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l3841_384169


namespace NUMINAMATH_CALUDE_meat_for_hamburgers_l3841_384191

/-- Given that 5 pounds of meat make 10 hamburgers, prove that 15 pounds of meat are needed for 30 hamburgers -/
theorem meat_for_hamburgers (meat_per_10 : ℕ) (hamburgers_per_5 : ℕ) 
  (h1 : meat_per_10 = 5) 
  (h2 : hamburgers_per_5 = 10) :
  (meat_per_10 * 3 : ℕ) = 15 ∧ (hamburgers_per_5 * 3 : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_hamburgers_l3841_384191


namespace NUMINAMATH_CALUDE_square_figure_perimeter_l3841_384170

/-- A figure composed of two rows of three consecutive unit squares, with the top row directly above the bottom row -/
structure SquareFigure where
  /-- The side length of each square -/
  side_length : ℝ
  /-- The number of squares in each row -/
  squares_per_row : ℕ
  /-- The number of rows -/
  rows : ℕ
  /-- The side length is 1 -/
  unit_side : side_length = 1
  /-- There are three squares in each row -/
  three_squares : squares_per_row = 3
  /-- There are two rows -/
  two_rows : rows = 2

/-- The perimeter of the SquareFigure -/
def perimeter (fig : SquareFigure) : ℝ :=
  2 * fig.side_length * fig.squares_per_row + 2 * fig.side_length * fig.rows

/-- Theorem stating that the perimeter of the SquareFigure is 16 -/
theorem square_figure_perimeter (fig : SquareFigure) : perimeter fig = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_figure_perimeter_l3841_384170


namespace NUMINAMATH_CALUDE_center_coordinates_sum_l3841_384118

/-- Given two points as the endpoints of a diameter of a circle, 
    prove that the sum of the coordinates of the center is 0. -/
theorem center_coordinates_sum (x₁ y₁ x₂ y₂ : ℝ) 
  (h : x₁ = 9 ∧ y₁ = -5 ∧ x₂ = -3 ∧ y₂ = -1) : 
  ((x₁ + x₂) / 2) + ((y₁ + y₂) / 2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_center_coordinates_sum_l3841_384118


namespace NUMINAMATH_CALUDE_marbles_exceed_200_on_friday_l3841_384122

def marbles (k : ℕ) : ℕ := 4 * 3^k

theorem marbles_exceed_200_on_friday :
  (∀ j : ℕ, j < 4 → marbles j ≤ 200) ∧ marbles 4 > 200 :=
sorry

end NUMINAMATH_CALUDE_marbles_exceed_200_on_friday_l3841_384122


namespace NUMINAMATH_CALUDE_carpeting_cost_specific_room_l3841_384112

/-- Calculates the cost of carpeting a room given its dimensions and carpet specifications. -/
def carpeting_cost (room_length room_breadth carpet_width_cm carpet_cost_paise : ℚ) : ℚ :=
  let room_area : ℚ := room_length * room_breadth
  let carpet_width_m : ℚ := carpet_width_cm / 100
  let carpet_length : ℚ := room_area / carpet_width_m
  let total_cost_paise : ℚ := carpet_length * carpet_cost_paise
  total_cost_paise / 100

/-- Theorem stating that the cost of carpeting a specific room is 36 rupees. -/
theorem carpeting_cost_specific_room :
  carpeting_cost 15 6 75 30 = 36 := by
  sorry

end NUMINAMATH_CALUDE_carpeting_cost_specific_room_l3841_384112


namespace NUMINAMATH_CALUDE_eliza_says_500_l3841_384131

-- Define the upper bound of the counting range
def upper_bound : ℕ := 500

-- Define the skipping pattern for each student
def alice_skip (n : ℕ) : Bool := n % 4 = 0
def barbara_skip (n : ℕ) : Bool := n % 12 = 4
def candice_skip (n : ℕ) : Bool := n % 16 = 0
def debbie_skip (n : ℕ) : Bool := n % 64 = 0

-- Define a function to check if a number is said by any of the first four students
def is_said_by_first_four (n : ℕ) : Bool :=
  ¬(alice_skip n) ∨ ¬(barbara_skip n) ∨ ¬(candice_skip n) ∨ ¬(debbie_skip n)

-- Theorem statement
theorem eliza_says_500 : 
  ∀ n : ℕ, n ≤ upper_bound → (n ≠ upper_bound → is_said_by_first_four n) ∧ ¬(is_said_by_first_four upper_bound) :=
by sorry

end NUMINAMATH_CALUDE_eliza_says_500_l3841_384131


namespace NUMINAMATH_CALUDE_discriminant_of_polynomial_l3841_384183

/-- The discriminant of a quadratic polynomial ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic polynomial 5x^2 + (5 + 1/5)x + 1/5 -/
def polynomial (x : ℚ) : ℚ := 5*x^2 + (5 + 1/5)*x + 1/5

theorem discriminant_of_polynomial :
  discriminant 5 (5 + 1/5) (1/5) = 576/25 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_polynomial_l3841_384183


namespace NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_l3841_384116

open Set
open Function

-- Define the concept of a monotonic function
def Monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y ∨ f y ≤ f x

-- Define the concept of having a maximum and minimum value on an interval
def HasMaxMin (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ max min : ℝ, (∀ x, a ≤ x ∧ x ≤ b → f x ≤ max) ∧
                 (∀ x, a ≤ x ∧ x ≤ b → min ≤ f x)

theorem monotonic_sufficient_not_necessary (a b : ℝ) (h : a ≤ b) :
  (∀ f : ℝ → ℝ, Monotonic f a b → HasMaxMin f a b) ∧
  (∃ f : ℝ → ℝ, HasMaxMin f a b ∧ ¬Monotonic f a b) :=
sorry

end NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_l3841_384116


namespace NUMINAMATH_CALUDE_remainder_of_87_pow_88_plus_7_mod_88_l3841_384195

theorem remainder_of_87_pow_88_plus_7_mod_88 : 87^88 + 7 ≡ 8 [MOD 88] := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_87_pow_88_plus_7_mod_88_l3841_384195


namespace NUMINAMATH_CALUDE_red_ball_probability_l3841_384130

-- Define the containers and their contents
structure Container where
  red : ℕ
  green : ℕ

def containerA : Container := ⟨10, 5⟩
def containerB : Container := ⟨6, 6⟩
def containerC : Container := ⟨3, 9⟩
def containerD : Container := ⟨4, 8⟩

-- Define the list of containers
def containers : List Container := [containerA, containerB, containerC, containerD]

-- Function to calculate the probability of selecting a red ball from a container
def redProbability (c : Container) : ℚ :=
  c.red / (c.red + c.green)

-- Theorem stating the probability of selecting a red ball
theorem red_ball_probability :
  (1 / (containers.length : ℚ)) * (containers.map redProbability).sum = 25 / 48 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l3841_384130


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l3841_384105

/-- Triangle side condition -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem -/
theorem triangle_inequality_squared (a b c : ℝ) 
  (h : is_triangle a b c) : 
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_squared_l3841_384105


namespace NUMINAMATH_CALUDE_smallest_label_on_final_position_l3841_384103

/-- The number of points on the circle -/
def n : ℕ := 70

/-- The function that calculates the position of a label -/
def position (k : ℕ) : ℕ := (k * (k + 1) / 2) % n

/-- The final label we're interested in -/
def final_label : ℕ := 2014

/-- The smallest label we claim to be on the same point as the final label -/
def smallest_label : ℕ := 5

theorem smallest_label_on_final_position :
  position final_label = position smallest_label ∧
  ∀ m : ℕ, m < smallest_label → position final_label ≠ position m :=
sorry

end NUMINAMATH_CALUDE_smallest_label_on_final_position_l3841_384103


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l3841_384177

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop :=
  x^2 - 16*y^2 - 8*x + 16 = 0

/-- The first line represented by the equation -/
def line1 (x y : ℝ) : Prop :=
  x = 4 + 4*y

/-- The second line represented by the equation -/
def line2 (x y : ℝ) : Prop :=
  x = 4 - 4*y

/-- Theorem stating that the equation represents two lines -/
theorem equation_represents_two_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l3841_384177


namespace NUMINAMATH_CALUDE_mikes_typing_speed_l3841_384124

/-- Mike's original typing speed in words per minute -/
def original_speed : ℕ := 65

/-- Mike's reduced typing speed in words per minute -/
def reduced_speed : ℕ := original_speed - 20

/-- Number of words in the document -/
def document_words : ℕ := 810

/-- Time taken to type the document at reduced speed, in minutes -/
def typing_time : ℕ := 18

theorem mikes_typing_speed :
  (reduced_speed * typing_time = document_words) ∧
  (original_speed = 65) := by
  sorry

end NUMINAMATH_CALUDE_mikes_typing_speed_l3841_384124


namespace NUMINAMATH_CALUDE_rational_inequality_equivalence_l3841_384161

theorem rational_inequality_equivalence (x : ℝ) :
  (2 * x - 1) / (x + 1) > 1 ↔ x < -1 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_equivalence_l3841_384161


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3841_384165

-- Define set A
def A : Set ℝ := {x | 2 * x ≤ 4}

-- Define set B (domain of lg(x-1))
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3841_384165


namespace NUMINAMATH_CALUDE_root_value_theorem_l3841_384136

theorem root_value_theorem (a : ℝ) : 
  (a^2 - 4*a - 6 = 0) → (a^2 - 4*a + 3 = 9) := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l3841_384136


namespace NUMINAMATH_CALUDE_max_renovation_days_l3841_384144

def turnkey_cost : ℕ := 50000
def materials_cost : ℕ := 20000
def husband_wage : ℕ := 2000
def wife_wage : ℕ := 1500

theorem max_renovation_days : 
  ∃ n : ℕ, n = 8 ∧ 
  n * (husband_wage + wife_wage) + materials_cost ≤ turnkey_cost ∧
  (n + 1) * (husband_wage + wife_wage) + materials_cost > turnkey_cost :=
sorry

end NUMINAMATH_CALUDE_max_renovation_days_l3841_384144


namespace NUMINAMATH_CALUDE_trail_mix_peanuts_weight_l3841_384188

theorem trail_mix_peanuts_weight (total_weight chocolate_weight raisin_weight : ℚ)
  (h1 : total_weight = 0.4166666666666667)
  (h2 : chocolate_weight = 0.16666666666666666)
  (h3 : raisin_weight = 0.08333333333333333) :
  total_weight - (chocolate_weight + raisin_weight) = 0.1666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_peanuts_weight_l3841_384188


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l3841_384180

/-- Represents the state of the game -/
structure GameState where
  score : ℕ
  remainingCards : List ℕ

/-- Defines a valid move in the game -/
def validMove (state : GameState) (card : ℕ) : Prop :=
  card ∈ state.remainingCards ∧ card ≥ 1 ∧ card ≤ 4

/-- Defines the winning condition -/
def isWinningMove (state : GameState) (card : ℕ) : Prop :=
  validMove state card ∧ (state.score + card = 22 ∨ state.score + card > 22)

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_winning_strategy :
  ∃ (initialCards : List ℕ),
    (initialCards.length = 16) ∧
    (∀ c ∈ initialCards, c ≥ 1 ∧ c ≤ 4) ∧
    (∃ (strategy : GameState → ℕ),
      ∀ (opponentStrategy : GameState → ℕ),
        let initialState : GameState := { score := 0, remainingCards := initialCards }
        let firstMove := strategy initialState
        validMove initialState firstMove ∧ firstMove = 1 →
        ∃ (finalState : GameState),
          isWinningMove finalState (strategy finalState)) :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l3841_384180


namespace NUMINAMATH_CALUDE_checkerboard_achievable_l3841_384198

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Color

/-- Initial grid configuration -/
def initial_grid : Grid :=
  λ i j => if j.val < 2 then Color.Black else Color.White

/-- Checkerboard pattern grid -/
def checkerboard : Grid :=
  λ i j => if (i.val + j.val) % 2 = 0 then Color.White else Color.Black

/-- Represents a rectangular subgrid -/
structure Rectangle where
  top_left : Fin 4 × Fin 4
  bottom_right : Fin 4 × Fin 4

/-- Toggles the color of cells within a rectangle -/
def toggle_rectangle (g : Grid) (r : Rectangle) : Grid :=
  λ i j => if i.val ≥ r.top_left.1.val && i.val ≤ r.bottom_right.1.val &&
             j.val ≥ r.top_left.2.val && j.val ≤ r.bottom_right.2.val
           then
             match g i j with
             | Color.Black => Color.White
             | Color.White => Color.Black
           else g i j

/-- Theorem stating that the checkerboard pattern is achievable in three operations -/
theorem checkerboard_achievable :
  ∃ (r1 r2 r3 : Rectangle),
    toggle_rectangle (toggle_rectangle (toggle_rectangle initial_grid r1) r2) r3 = checkerboard :=
  sorry

end NUMINAMATH_CALUDE_checkerboard_achievable_l3841_384198


namespace NUMINAMATH_CALUDE_square_side_length_l3841_384152

theorem square_side_length (perimeter : ℝ) (h : perimeter = 100) :
  perimeter / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3841_384152


namespace NUMINAMATH_CALUDE_hexagon_diagonals_from_vertex_l3841_384106

/-- The number of diagonals that can be drawn from one vertex of a hexagon -/
def diagonals_from_vertex_hexagon : ℕ := 3

/-- Theorem stating that the number of diagonals from one vertex of a hexagon is 3 -/
theorem hexagon_diagonals_from_vertex :
  diagonals_from_vertex_hexagon = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_from_vertex_l3841_384106


namespace NUMINAMATH_CALUDE_no_prime_factor_congruent_to_negative_one_mod_eight_l3841_384115

theorem no_prime_factor_congruent_to_negative_one_mod_eight (n : ℕ+) (p : ℕ) 
  (h_prime : Nat.Prime p) (h_cong : p % 8 = 7) : 
  ¬(p ∣ 2^(n.val.succ.succ) + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_factor_congruent_to_negative_one_mod_eight_l3841_384115


namespace NUMINAMATH_CALUDE_triangle_formation_l3841_384192

/-- Triangle inequality theorem: the sum of any two sides must be greater than the third side --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 5 7 ∧
  ¬can_form_triangle 1 3 4 ∧
  ¬can_form_triangle 2 2 7 ∧
  ¬can_form_triangle 3 3 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3841_384192


namespace NUMINAMATH_CALUDE_star_equation_solution_l3841_384121

/-- The "※" operation for positive real numbers -/
def star (a b : ℝ) : ℝ := a * b + a + b^2

/-- Theorem: If 1※k = 3, then k = 1 -/
theorem star_equation_solution (k : ℝ) (hk : k > 0) (h : star 1 k = 3) : k = 1 := by
  sorry

#check star_equation_solution

end NUMINAMATH_CALUDE_star_equation_solution_l3841_384121


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3841_384117

theorem sufficient_not_necessary_condition :
  (∀ x y : ℝ, x > 3 ∧ y > 3 → x + y > 6) ∧
  (∃ x y : ℝ, x + y > 6 ∧ ¬(x > 3 ∧ y > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3841_384117


namespace NUMINAMATH_CALUDE_elliptical_machine_cost_l3841_384173

/-- The cost of an elliptical machine -/
def machine_cost : ℝ := 120

/-- The daily minimum payment -/
def daily_minimum_payment : ℝ := 6

/-- The number of days to pay the remaining balance -/
def payment_days : ℕ := 10

/-- Theorem stating the cost of the elliptical machine -/
theorem elliptical_machine_cost :
  (machine_cost / 2 = daily_minimum_payment * payment_days) ∧
  (machine_cost / 2 = machine_cost - machine_cost / 2) := by
  sorry

#check elliptical_machine_cost

end NUMINAMATH_CALUDE_elliptical_machine_cost_l3841_384173


namespace NUMINAMATH_CALUDE_arthur_spent_fraction_l3841_384154

theorem arthur_spent_fraction (initial_amount : ℚ) (remaining_amount : ℚ) 
  (h1 : initial_amount = 200)
  (h2 : remaining_amount = 40) :
  (initial_amount - remaining_amount) / initial_amount = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_spent_fraction_l3841_384154


namespace NUMINAMATH_CALUDE_area_of_region_l3841_384142

-- Define the region
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 ≤ 2 ∧ abs p.2 ≤ 2 ∧ abs (abs p.1 - abs p.2) ≤ 1}

-- State the theorem
theorem area_of_region : MeasureTheory.volume R = 12 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l3841_384142


namespace NUMINAMATH_CALUDE_unique_function_property_l3841_384123

theorem unique_function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = f x + 1)
  (h2 : ∀ x, f (x^2) = (f x)^2) :
  ∀ x, f x = x :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l3841_384123


namespace NUMINAMATH_CALUDE_rocket_momentum_l3841_384102

/-- Given two rockets with masses m and 9m, subjected to the same constant force F 
    for the same distance d, if the rocket with mass m acquires momentum p, 
    then the rocket with mass 9m acquires momentum 3p. -/
theorem rocket_momentum 
  (m : ℝ) 
  (F : ℝ) 
  (d : ℝ) 
  (p : ℝ) 
  (h1 : m > 0) 
  (h2 : F > 0) 
  (h3 : d > 0) 
  (h4 : p = Real.sqrt (2 * d * m * F)) : 
  9 * m * Real.sqrt ((2 * F * d) / (9 * m)) = 3 * p := by
  sorry

end NUMINAMATH_CALUDE_rocket_momentum_l3841_384102


namespace NUMINAMATH_CALUDE_thomas_score_l3841_384110

theorem thomas_score (n : ℕ) (avg_without_thomas avg_with_thomas thomas_score : ℚ) :
  n = 20 →
  avg_without_thomas = 78 →
  avg_with_thomas = 80 →
  (n - 1) * avg_without_thomas + thomas_score = n * avg_with_thomas →
  thomas_score = 118 := by
sorry

end NUMINAMATH_CALUDE_thomas_score_l3841_384110


namespace NUMINAMATH_CALUDE_bert_ernie_stamp_ratio_l3841_384141

/-- The number of stamps Peggy has -/
def peggy_stamps : ℕ := 75

/-- The number of stamps Peggy needs to add to match Bert's collection -/
def stamps_to_add : ℕ := 825

/-- The number of stamps Ernie has -/
def ernie_stamps : ℕ := 3 * peggy_stamps

/-- The number of stamps Bert has -/
def bert_stamps : ℕ := peggy_stamps + stamps_to_add

/-- The ratio of Bert's stamps to Ernie's stamps -/
def stamp_ratio : ℚ := bert_stamps / ernie_stamps

theorem bert_ernie_stamp_ratio :
  stamp_ratio = 4 / 1 := by sorry

end NUMINAMATH_CALUDE_bert_ernie_stamp_ratio_l3841_384141


namespace NUMINAMATH_CALUDE_giraffe_count_l3841_384150

/-- The number of giraffes in the zoo -/
def num_giraffes : ℕ := 5

/-- The number of penguins in the zoo -/
def num_penguins : ℕ := 10

/-- The number of elephants in the zoo -/
def num_elephants : ℕ := 2

/-- The total number of animals in the zoo -/
def total_animals : ℕ := 50

theorem giraffe_count :
  (num_penguins = 2 * num_giraffes) ∧
  (num_penguins = (20 : ℕ) * total_animals / 100) ∧
  (num_elephants = (4 : ℕ) * total_animals / 100) ∧
  (num_elephants = 2) →
  num_giraffes = 5 := by
sorry

end NUMINAMATH_CALUDE_giraffe_count_l3841_384150


namespace NUMINAMATH_CALUDE_tonya_stamps_after_trade_l3841_384137

/-- Represents the trade of matchbooks for stamps between Jimmy and Tonya --/
def matchbook_stamp_trade (stamp_match_ratio : ℕ) (matches_per_book : ℕ) (tonya_initial_stamps : ℕ) (jimmy_matchbooks : ℕ) : ℕ :=
  let jimmy_total_matches := jimmy_matchbooks * matches_per_book
  let jimmy_stamps_worth := jimmy_total_matches / stamp_match_ratio
  tonya_initial_stamps - jimmy_stamps_worth

/-- Theorem stating that Tonya will have 3 stamps left after the trade --/
theorem tonya_stamps_after_trade :
  matchbook_stamp_trade 12 24 13 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tonya_stamps_after_trade_l3841_384137


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l3841_384134

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h_side : a * b = 20) 
  (h_front : b * c = 12) 
  (h_bottom : a * c = 15) : 
  a * b * c = 60 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l3841_384134


namespace NUMINAMATH_CALUDE_scaled_variance_l3841_384153

-- Define a dataset type
def Dataset := List Real

-- Define the standard deviation function
noncomputable def standardDeviation (data : Dataset) : Real :=
  sorry

-- Define the variance function
noncomputable def variance (data : Dataset) : Real :=
  sorry

-- Define a function to scale a dataset
def scaleDataset (data : Dataset) (scale : Real) : Dataset :=
  data.map (· * scale)

-- Theorem statement
theorem scaled_variance 
  (data : Dataset) 
  (h : standardDeviation data = 2) : 
  variance (scaleDataset data 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_scaled_variance_l3841_384153


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l3841_384157

def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_root_sum (a b c d : ℝ) :
  g a b c d (3*I) = 0 ∧ g a b c d (1 + I) = 0 → a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l3841_384157


namespace NUMINAMATH_CALUDE_parabola_translation_l3841_384149

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * dx + p.b
    c := p.a * dx^2 - p.b * dx + p.c - dy }

theorem parabola_translation :
  let original := Parabola.mk 1 0 1  -- y = x^2 + 1
  let translated := translate original 3 (-2)  -- 3 units right, 2 units down
  translated = Parabola.mk 1 (-6) (-1)  -- y = (x - 3)^2 - 1
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3841_384149


namespace NUMINAMATH_CALUDE_max_bisections_for_zero_approximation_l3841_384120

/-- Theorem: Maximum number of bisections for approximating a zero --/
theorem max_bisections_for_zero_approximation 
  (f : ℝ → ℝ) 
  (zero_exists : ∃ x, x ∈ (Set.Ioo 0 1) ∧ f x = 0) 
  (accuracy : ℝ := 0.01) :
  (∃ n : ℕ, n ≤ 7 ∧ 
    (1 : ℝ) / (2 ^ n) < accuracy ∧ 
    ∀ m : ℕ, m < n → (1 : ℝ) / (2 ^ m) ≥ accuracy) :=
sorry

end NUMINAMATH_CALUDE_max_bisections_for_zero_approximation_l3841_384120


namespace NUMINAMATH_CALUDE_reflection_of_D_l3841_384158

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)  -- Translate down by 1
  let p'' := (p'.2, p'.1)   -- Reflect over y = x
  (p''.1, p''.2 + 1)        -- Translate up by 1

def D : ℝ × ℝ := (4, 1)

theorem reflection_of_D : 
  reflect_y_eq_x_minus_1 (reflect_x D) = (-2, 5) := by sorry

end NUMINAMATH_CALUDE_reflection_of_D_l3841_384158
