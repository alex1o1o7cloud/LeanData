import Mathlib

namespace NUMINAMATH_CALUDE_line_length_calculation_line_length_proof_l1815_181546

theorem line_length_calculation (initial_length : ℕ) 
  (first_erasure : ℕ) (first_extension : ℕ) 
  (second_erasure : ℕ) (final_addition : ℕ) : ℕ :=
  let step1 := initial_length - first_erasure
  let step2 := step1 + first_extension
  let step3 := step2 - second_erasure
  let final_length := step3 + final_addition
  final_length

theorem line_length_proof :
  line_length_calculation 100 24 35 15 8 = 104 := by
  sorry

end NUMINAMATH_CALUDE_line_length_calculation_line_length_proof_l1815_181546


namespace NUMINAMATH_CALUDE_small_square_area_l1815_181533

theorem small_square_area (n : ℕ) : n > 0 → (
  let outer_square_area : ℝ := 1
  let small_square_area : ℝ := 1 / 1985
  let side_length : ℝ := 1 / n
  let diagonal_segment : ℝ := (n - 1) / n
  let small_square_side : ℝ := diagonal_segment / Real.sqrt 1985
  small_square_side * small_square_side = small_square_area
) ↔ n = 32 := by sorry

end NUMINAMATH_CALUDE_small_square_area_l1815_181533


namespace NUMINAMATH_CALUDE_odd_power_sum_divisible_l1815_181532

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ n : ℕ, n > 0 → Odd n → (x^n + y^n) % (x + y) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisible_l1815_181532


namespace NUMINAMATH_CALUDE_one_different_value_l1815_181596

/-- The standard exponentiation result for 2^(2^(2^2)) -/
def standard_result : ℕ := 65536

/-- The set of all possible values obtained by different parenthesizations of 2^2^2^2 -/
def all_results : Set ℕ :=
  {2^(2^(2^2)), 2^((2^2)^2), ((2^2)^2)^2, (2^(2^2))^2, (2^2)^(2^2)}

/-- The theorem stating that there is exactly one value different from the standard result -/
theorem one_different_value :
  ∃! x, x ∈ all_results ∧ x ≠ standard_result :=
sorry

end NUMINAMATH_CALUDE_one_different_value_l1815_181596


namespace NUMINAMATH_CALUDE_sequence_ratio_proof_l1815_181521

theorem sequence_ratio_proof (d : ℝ) (q : ℚ) (a b : ℕ → ℝ) :
  d ≠ 0 →
  (0 < q) →
  (q < 1) →
  (∀ n, a (n + 1) = a n + d) →
  (∀ n, b (n + 1) = q * b n) →
  a 1 = d →
  b 1 = d^2 →
  (∃ k : ℕ, (a 1)^2 + (a 2)^2 + (a 3)^2 = k * (b 1 + b 2 + b 3)) →
  q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_proof_l1815_181521


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l1815_181522

theorem excluded_students_average_mark 
  (N : ℕ) (A E : ℕ) (A_R : ℚ) (A_E : ℚ) 
  (h1 : N = 56)
  (h2 : A = 80)
  (h3 : E = 8)
  (h4 : A_R = 90)
  (h5 : N * A = E * A_E + (N - E) * A_R) :
  A_E = 20 := by
sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l1815_181522


namespace NUMINAMATH_CALUDE_greg_needs_33_more_l1815_181541

/-- The cost of the scooter in dollars -/
def scooter_cost : ℕ := 90

/-- The amount Greg has saved in dollars -/
def greg_savings : ℕ := 57

/-- The additional amount Greg needs to buy the scooter -/
def additional_amount_needed : ℕ := scooter_cost - greg_savings

/-- Theorem stating that the additional amount Greg needs is $33 -/
theorem greg_needs_33_more :
  additional_amount_needed = 33 :=
by sorry

end NUMINAMATH_CALUDE_greg_needs_33_more_l1815_181541


namespace NUMINAMATH_CALUDE_classroom_capacity_l1815_181578

/-- Calculates the total number of desks in a classroom with an arithmetic progression of desks per row -/
def totalDesks (rows : ℕ) (firstRowDesks : ℕ) (increment : ℕ) : ℕ :=
  rows * (2 * firstRowDesks + (rows - 1) * increment) / 2

/-- Theorem stating that a classroom with 8 rows, starting with 10 desks and increasing by 2 each row, can seat 136 students -/
theorem classroom_capacity :
  totalDesks 8 10 2 = 136 := by
  sorry

#eval totalDesks 8 10 2

end NUMINAMATH_CALUDE_classroom_capacity_l1815_181578


namespace NUMINAMATH_CALUDE_pupils_in_program_l1815_181582

/-- Given a program with a total of 238 people and 61 parents, prove that there were 177 pupils present. -/
theorem pupils_in_program (total_people : ℕ) (parents : ℕ) (h1 : total_people = 238) (h2 : parents = 61) :
  total_people - parents = 177 := by
  sorry

end NUMINAMATH_CALUDE_pupils_in_program_l1815_181582


namespace NUMINAMATH_CALUDE_tower_height_difference_l1815_181590

/-- Given the heights of three towers and their relationships, prove the height difference between two of them. -/
theorem tower_height_difference 
  (cn_tower_height : ℝ)
  (cn_space_needle_diff : ℝ)
  (eiffel_tower_height : ℝ)
  (h1 : cn_tower_height = 553)
  (h2 : cn_space_needle_diff = 369)
  (h3 : eiffel_tower_height = 330) :
  eiffel_tower_height - (cn_tower_height - cn_space_needle_diff) = 146 := by
  sorry

end NUMINAMATH_CALUDE_tower_height_difference_l1815_181590


namespace NUMINAMATH_CALUDE_remainder_of_power_minus_digit_l1815_181503

theorem remainder_of_power_minus_digit (x : ℕ) : 
  x < 10 → (Nat.pow 2 200 - x) % 7 = 1 → x = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_of_power_minus_digit_l1815_181503


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l1815_181561

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l1815_181561


namespace NUMINAMATH_CALUDE_fifth_smallest_odd_with_four_prime_factors_l1815_181525

def has_at_least_four_prime_factors (n : ℕ) : Prop :=
  ∃ (p q r s : ℕ), Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ p * q * r * s ∣ n

def is_fifth_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), P a ∧ P b ∧ P c ∧ P d ∧
    a < b ∧ b < c ∧ c < d ∧ d < n ∧
    (∀ m, P m → m ≥ n ∨ m = a ∨ m = b ∨ m = c ∨ m = d)

theorem fifth_smallest_odd_with_four_prime_factors :
  is_fifth_smallest (λ n => Odd n ∧ has_at_least_four_prime_factors n) 1925 :=
sorry

end NUMINAMATH_CALUDE_fifth_smallest_odd_with_four_prime_factors_l1815_181525


namespace NUMINAMATH_CALUDE_exists_ten_digit_number_divisible_by_11_with_all_digits_l1815_181539

def is_ten_digit_number (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def contains_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, (n / 10^k) % 10 = d

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem exists_ten_digit_number_divisible_by_11_with_all_digits :
  ∃ n : ℕ, is_ten_digit_number n ∧ contains_all_digits n ∧ is_divisible_by_11 n :=
sorry

end NUMINAMATH_CALUDE_exists_ten_digit_number_divisible_by_11_with_all_digits_l1815_181539


namespace NUMINAMATH_CALUDE_nine_point_circle_chords_l1815_181593

/-- The number of chords that can be drawn in a circle with n points on its circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords that can be drawn in a circle with 9 points on its circumference is 36 -/
theorem nine_point_circle_chords : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_circle_chords_l1815_181593


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1815_181567

theorem midpoint_sum_equals_vertex_sum (d e f : ℝ) : 
  let vertex_sum := d + e + f
  let midpoint_sum := (d + e) / 2 + (d + f) / 2 + (e + f) / 2
  vertex_sum = midpoint_sum := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1815_181567


namespace NUMINAMATH_CALUDE_more_spins_more_accurate_l1815_181515

/-- Represents a spinner used in random simulation -/
structure Spinner :=
  (radius : ℝ)

/-- Represents the result of a spinner simulation -/
structure SimulationResult :=
  (accuracy : ℝ)

/-- Represents a random simulation using a spinner -/
def SpinnerSimulation := Spinner → ℕ → SimulationResult

/-- Axiom: The spinner must be spun randomly for accurate estimation -/
axiom random_spinning_required (s : Spinner) (n : ℕ) (sim : SpinnerSimulation) :
  SimulationResult

/-- Axiom: The number of spins affects the estimation accuracy -/
axiom spins_affect_accuracy (s : Spinner) (n m : ℕ) (sim : SpinnerSimulation) :
  n ≠ m → sim s n ≠ sim s m

/-- Axiom: The spinner's radius does not affect the estimation accuracy -/
axiom radius_doesnt_affect_accuracy (s₁ s₂ : Spinner) (n : ℕ) (sim : SpinnerSimulation) :
  s₁.radius ≠ s₂.radius → sim s₁ n = sim s₂ n

/-- Theorem: Increasing the number of spins improves the accuracy of the estimation result -/
theorem more_spins_more_accurate (s : Spinner) (n m : ℕ) (sim : SpinnerSimulation) :
  n < m → (sim s m).accuracy > (sim s n).accuracy :=
sorry

end NUMINAMATH_CALUDE_more_spins_more_accurate_l1815_181515


namespace NUMINAMATH_CALUDE_inequality_proof_l1815_181523

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1815_181523


namespace NUMINAMATH_CALUDE_jam_weight_l1815_181520

/-- Calculates the weight of jam given the initial and final suitcase weights and other item weights --/
theorem jam_weight 
  (initial_weight : ℝ) 
  (final_weight : ℝ) 
  (perfume_weight : ℝ) 
  (perfume_count : ℕ) 
  (chocolate_weight : ℝ) 
  (soap_weight : ℝ) 
  (soap_count : ℕ) 
  (h1 : initial_weight = 5) 
  (h2 : final_weight = 11) 
  (h3 : perfume_weight = 1.2 / 16) 
  (h4 : perfume_count = 5) 
  (h5 : chocolate_weight = 4) 
  (h6 : soap_weight = 5 / 16) 
  (h7 : soap_count = 2) : 
  final_weight - (initial_weight + perfume_weight * perfume_count + chocolate_weight + soap_weight * soap_count) = 1 := by
  sorry

#check jam_weight

end NUMINAMATH_CALUDE_jam_weight_l1815_181520


namespace NUMINAMATH_CALUDE_equation_solution_l1815_181583

theorem equation_solution (x : ℝ) (h : x ≥ 0) : x + 2 * Real.sqrt x - 8 = 0 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1815_181583


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l1815_181597

def g (x : ℝ) := (x - 3)^2 + 6

theorem smallest_c_for_inverse :
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l1815_181597


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_of_24_l1815_181516

theorem smallest_sum_of_factors_of_24 :
  (∀ a b : ℕ, a * b = 24 → a + b ≥ 10) ∧
  (∃ a b : ℕ, a * b = 24 ∧ a + b = 10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_of_24_l1815_181516


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l1815_181592

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  fociAxis : ℝ × ℝ
  eccentricity : ℝ
  passingPoint : ℝ × ℝ

/-- The equation of an ellipse given its properties -/
def ellipseEquation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 45 + y^2 / 36 = 1

/-- Theorem stating that an ellipse with the given properties has the specified equation -/
theorem ellipse_equation_proof (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.fociAxis.2 = 0)
  (h3 : e.eccentricity = Real.sqrt 5 / 5)
  (h4 : e.passingPoint = (-5, 4)) :
  ellipseEquation e = fun x y => x^2 / 45 + y^2 / 36 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l1815_181592


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l1815_181576

/-- The number of x-intercepts of the parabola x = -3y^2 + 2y + 2 -/
theorem parabola_x_intercepts :
  let f : ℝ → ℝ := λ y => -3 * y^2 + 2 * y + 2
  ∃! x : ℝ, ∃ y : ℝ, f y = x ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l1815_181576


namespace NUMINAMATH_CALUDE_ball_selection_limit_l1815_181570

open Real

/-- The probability of selecting n₁ white balls and n₂ black balls without replacement from an urn -/
noncomputable def P (M₁ M₂ n₁ n₂ : ℕ) : ℝ :=
  (Nat.choose M₁ n₁ * Nat.choose M₂ n₂ : ℝ) / Nat.choose (M₁ + M₂) (n₁ + n₂)

/-- The limit of the probability as M and M₁ approach infinity -/
theorem ball_selection_limit (n₁ n₂ : ℕ) (p : ℝ) (h_p : 0 < p ∧ p < 1) :
  ∀ ε > 0, ∃ N : ℕ, ∀ M₁ M₂ : ℕ,
    M₁ ≥ N → M₂ ≥ N →
    |P M₁ M₂ n₁ n₂ - (Nat.choose (n₁ + n₂) n₁ : ℝ) * p^n₁ * (1 - p)^n₂| < ε :=
by sorry

end NUMINAMATH_CALUDE_ball_selection_limit_l1815_181570


namespace NUMINAMATH_CALUDE_stream_speed_l1815_181569

theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 78)
  (h2 : upstream_distance = 50)
  (h3 : time = 2) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1815_181569


namespace NUMINAMATH_CALUDE_soccer_league_games_l1815_181548

theorem soccer_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 10) (h2 : total_games = 45) :
  (n * (n - 1)) / 2 = total_games → ∃ k : ℕ, k = 1 ∧ k * (n * (n - 1)) / 2 = total_games :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1815_181548


namespace NUMINAMATH_CALUDE_college_graduates_scientific_notation_l1815_181502

theorem college_graduates_scientific_notation :
  ∃ (x : ℝ) (n : ℤ), 
    x ≥ 1 ∧ x < 10 ∧ 
    116000000 = x * (10 : ℝ) ^ n ∧
    x = 1.16 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_college_graduates_scientific_notation_l1815_181502


namespace NUMINAMATH_CALUDE_b_101_mod_49_l1815_181556

/-- The sequence b_n defined as 5^n + 7^n -/
def b (n : ℕ) : ℕ := 5^n + 7^n

/-- Theorem stating that b_101 is congruent to 12 modulo 49 -/
theorem b_101_mod_49 : b 101 ≡ 12 [MOD 49] := by
  sorry

end NUMINAMATH_CALUDE_b_101_mod_49_l1815_181556


namespace NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l1815_181555

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

/-- The number of factors of a natural number -/
def NumberOfFactors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem composite_has_at_least_three_factors (n : ℕ) (h : IsComposite n) :
    NumberOfFactors n ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l1815_181555


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1815_181554

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Define the point of tangency
def x₀ : ℝ := 2

-- Define the slope of the tangent line
def m : ℝ := 3 * x₀^2 - 2

-- Define the y-intercept of the tangent line
def b : ℝ := f x₀ - m * x₀

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * x + b ↔ y - f x₀ = m * (x - x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1815_181554


namespace NUMINAMATH_CALUDE_train_length_l1815_181508

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 265 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1815_181508


namespace NUMINAMATH_CALUDE_equation_solution_l1815_181549

theorem equation_solution (x : ℝ) : 
  (((1 - (Real.cos (3 * x))^15 * (Real.cos (5 * x))^2)^(1/4) = Real.sin (5 * x)) ∧ 
   (Real.sin (5 * x) ≥ 0)) ↔ 
  ((∃ n : ℤ, x = π / 10 + 2 * π * n / 5) ∨ 
   (∃ s : ℤ, x = 2 * π * s)) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1815_181549


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l1815_181526

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ + 5| = 20) ∧ (|x₂ + 5| = 20) ∧ (x₁ ≠ x₂) ∧ (|x₁ - x₂| = 40) := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l1815_181526


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1815_181573

def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x > 1}

theorem union_of_A_and_B : A ∪ B = {x | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1815_181573


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1815_181540

-- Define set A
def A : Set ℝ := {x | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1815_181540


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1815_181529

theorem complex_power_modulus : Complex.abs ((2 + Complex.I) ^ 8) = 625 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1815_181529


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1815_181557

/-- The maximum area of a rectangle with perimeter 40 is 100 -/
theorem max_area_rectangle (p : ℝ) (h : p = 40) : 
  (∃ (a : ℝ), a > 0 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = p / 2 → x * y ≤ a) ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = p / 2 ∧ x * y = 100) :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1815_181557


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l1815_181591

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (l m : Line) (α : Plane) :
  subset m α → 
  parallel_lines l m → 
  ¬subset l α → 
  parallel_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l1815_181591


namespace NUMINAMATH_CALUDE_probability_is_one_third_l1815_181535

-- Define the number of hot dishes
def num_dishes : ℕ := 3

-- Define the number of dishes a student can choose
def num_choices : ℕ := 2

-- Define the function to calculate combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of two students choosing the same two dishes
def probability_same_choices : ℚ :=
  (combinations num_dishes num_choices) / (combinations num_dishes num_choices * combinations num_dishes num_choices)

-- Theorem to prove
theorem probability_is_one_third :
  probability_same_choices = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l1815_181535


namespace NUMINAMATH_CALUDE_mark_change_in_nickels_l1815_181545

def bread_cost : ℚ := 4.20
def cheese_cost : ℚ := 2.05
def amount_given : ℚ := 7.00
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10
def nickel_value : ℚ := 0.05

theorem mark_change_in_nickels :
  let total_cost := bread_cost + cheese_cost
  let change := amount_given - total_cost
  let remaining_change := change - (quarter_value + dime_value)
  (remaining_change / nickel_value : ℚ) = 8 := by sorry

end NUMINAMATH_CALUDE_mark_change_in_nickels_l1815_181545


namespace NUMINAMATH_CALUDE_jane_score_is_14_l1815_181531

/-- Represents a mathematics competition with a scoring system. -/
structure MathCompetition where
  totalQuestions : ℕ
  correctAnswers : ℕ
  incorrectAnswers : ℕ
  unansweredQuestions : ℕ
  correctPoints : ℚ
  incorrectPenalty : ℚ

/-- Calculates the total score for a given math competition. -/
def calculateScore (comp : MathCompetition) : ℚ :=
  comp.correctAnswers * comp.correctPoints - comp.incorrectAnswers * comp.incorrectPenalty

/-- Theorem stating that Jane's score in the competition is 14 points. -/
theorem jane_score_is_14 (comp : MathCompetition)
  (h1 : comp.totalQuestions = 35)
  (h2 : comp.correctAnswers = 17)
  (h3 : comp.incorrectAnswers = 12)
  (h4 : comp.unansweredQuestions = 6)
  (h5 : comp.correctPoints = 1)
  (h6 : comp.incorrectPenalty = 1/4)
  (h7 : comp.totalQuestions = comp.correctAnswers + comp.incorrectAnswers + comp.unansweredQuestions) :
  calculateScore comp = 14 := by
  sorry

end NUMINAMATH_CALUDE_jane_score_is_14_l1815_181531


namespace NUMINAMATH_CALUDE_prob_two_cards_sum_17_l1815_181505

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of each specific card (8 or 9) in the deck
def cards_of_each_value : ℕ := 4

-- Define the probability of choosing two specific cards
def prob_two_specific_cards : ℚ := (cards_of_each_value : ℚ) / total_cards * (cards_of_each_value : ℚ) / (total_cards - 1)

-- Define the number of ways to choose two cards that sum to 17 (8+9 or 9+8)
def ways_to_sum_17 : ℕ := 2

theorem prob_two_cards_sum_17 : 
  prob_two_specific_cards * ways_to_sum_17 = 8 / 663 := by sorry

end NUMINAMATH_CALUDE_prob_two_cards_sum_17_l1815_181505


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1815_181507

theorem quadratic_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x - m = 0 ∧ y^2 - 6*y - m = 0) ↔ m > -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1815_181507


namespace NUMINAMATH_CALUDE_find_k_value_l1815_181510

theorem find_k_value (k : ℝ) (h : 32 / k = 4) : k = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l1815_181510


namespace NUMINAMATH_CALUDE_furniture_shop_pricing_l1815_181560

theorem furniture_shop_pricing (cost_price : ℝ) (markup_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 6525 →
  markup_percentage = 24 →
  selling_price = cost_price * (1 + markup_percentage / 100) →
  selling_price = 8091 := by
sorry

end NUMINAMATH_CALUDE_furniture_shop_pricing_l1815_181560


namespace NUMINAMATH_CALUDE_abs_eq_neg_self_iff_nonpositive_l1815_181528

theorem abs_eq_neg_self_iff_nonpositive (x : ℝ) : |x| = -x ↔ x ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_eq_neg_self_iff_nonpositive_l1815_181528


namespace NUMINAMATH_CALUDE_triangle_abc_isosceles_l1815_181513

theorem triangle_abc_isosceles (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_condition : Real.sin B = 2 * Real.cos C * Real.sin A) : A = C := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_isosceles_l1815_181513


namespace NUMINAMATH_CALUDE_only_rational_root_l1815_181550

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 + 4 * x^2 + 6 * x - 8

theorem only_rational_root :
  ∀ x : ℚ, polynomial x = 0 ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_only_rational_root_l1815_181550


namespace NUMINAMATH_CALUDE_barrels_of_pitch_day4_is_two_l1815_181562

/-- Represents the roadwork company's paving project --/
structure RoadworkProject where
  total_length : ℕ
  gravel_per_truck : ℕ
  gravel_to_pitch_ratio : ℕ
  day1_truckloads_per_mile : ℕ
  day1_miles_paved : ℕ
  day2_truckloads_per_mile : ℕ
  day2_miles_paved : ℕ
  day3_truckloads_per_mile : ℕ
  day3_miles_paved : ℕ
  day4_truckloads_per_mile : ℕ

/-- Calculates the number of barrels of pitch needed for the fourth day --/
def barrels_of_pitch_day4 (project : RoadworkProject) : ℕ :=
  let remaining_miles := project.total_length - (project.day1_miles_paved + project.day2_miles_paved + project.day3_miles_paved)
  let day4_truckloads := remaining_miles * project.day4_truckloads_per_mile
  let pitch_per_truck := project.gravel_per_truck / project.gravel_to_pitch_ratio
  let total_pitch := day4_truckloads * pitch_per_truck
  (total_pitch + 9) / 10  -- Round up to the nearest whole barrel

/-- Theorem stating that the number of barrels of pitch needed for the fourth day is 2 --/
theorem barrels_of_pitch_day4_is_two (project : RoadworkProject) 
  (h1 : project.total_length = 20)
  (h2 : project.gravel_per_truck = 2)
  (h3 : project.gravel_to_pitch_ratio = 5)
  (h4 : project.day1_truckloads_per_mile = 3)
  (h5 : project.day1_miles_paved = 4)
  (h6 : project.day2_truckloads_per_mile = 4)
  (h7 : project.day2_miles_paved = 7)
  (h8 : project.day3_truckloads_per_mile = 2)
  (h9 : project.day3_miles_paved = 5)
  (h10 : project.day4_truckloads_per_mile = 1) :
  barrels_of_pitch_day4 project = 2 := by
  sorry

end NUMINAMATH_CALUDE_barrels_of_pitch_day4_is_two_l1815_181562


namespace NUMINAMATH_CALUDE_third_grade_girls_l1815_181577

theorem third_grade_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 123 → boys = 66 → total = boys + girls → girls = 57 := by
  sorry

end NUMINAMATH_CALUDE_third_grade_girls_l1815_181577


namespace NUMINAMATH_CALUDE_coin_difference_is_nine_l1815_181538

def coin_denominations : List Nat := [5, 10, 25, 50]

def amount_to_pay : Nat := 55

def min_coins (denominations : List Nat) (amount : Nat) : Nat :=
  sorry

def max_coins (denominations : List Nat) (amount : Nat) : Nat :=
  sorry

theorem coin_difference_is_nine :
  max_coins coin_denominations amount_to_pay - min_coins coin_denominations amount_to_pay = 9 :=
by sorry

end NUMINAMATH_CALUDE_coin_difference_is_nine_l1815_181538


namespace NUMINAMATH_CALUDE_l_shaped_count_is_even_l1815_181504

/-- A centrally symmetric figure on a grid --/
structure CentrallySymmetricFigure where
  n : ℕ  -- number of "L-shaped" figures
  k : ℕ  -- number of 1 × 4 rectangles

/-- Theorem: The number of "L-shaped" figures in a centrally symmetric figure is even --/
theorem l_shaped_count_is_even (figure : CentrallySymmetricFigure) : Even figure.n := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_count_is_even_l1815_181504


namespace NUMINAMATH_CALUDE_max_at_two_implies_c_six_l1815_181571

/-- The function f(x) defined as x(x-c)^2 --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- Theorem stating that if f(x) has a maximum at x = 2, then c = 6 --/
theorem max_at_two_implies_c_six :
  ∀ c : ℝ, (∀ x : ℝ, f c x ≤ f c 2) → c = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_at_two_implies_c_six_l1815_181571


namespace NUMINAMATH_CALUDE_power_of_five_l1815_181518

theorem power_of_five (n : ℕ) : 5^n = 5 * 25^2 * 125^3 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l1815_181518


namespace NUMINAMATH_CALUDE_solution_of_inequality_1_solution_of_inequality_2_l1815_181598

-- Define the solution sets
def solution_set_1 : Set ℝ := {x | -5/2 < x ∧ x < 3}
def solution_set_2 : Set ℝ := {x | x < -2/3 ∨ x > 0}

-- Theorem for the first inequality
theorem solution_of_inequality_1 :
  {x : ℝ | 2*x^2 - x - 15 < 0} = solution_set_1 :=
by sorry

-- Theorem for the second inequality
theorem solution_of_inequality_2 :
  {x : ℝ | 2/x > -3} = solution_set_2 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_inequality_1_solution_of_inequality_2_l1815_181598


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1815_181558

theorem absolute_value_equation_solution (a : ℝ) : 
  50 - |a - 2| = |4 - a| → (a = -22 ∨ a = 28) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1815_181558


namespace NUMINAMATH_CALUDE_comparison_of_scientific_notation_l1815_181509

theorem comparison_of_scientific_notation :
  (1.9 : ℝ) * (10 : ℝ) ^ 5 > (9.1 : ℝ) * (10 : ℝ) ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_scientific_notation_l1815_181509


namespace NUMINAMATH_CALUDE_range_of_m_l1815_181585

def p (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a = 8 - m ∧ b = 2 * m - 1

def q (m : ℝ) : Prop :=
  (m + 1) * (m - 2) < 0

theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Ioo (-1 : ℝ) (1/2) ∪ Set.Ico 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1815_181585


namespace NUMINAMATH_CALUDE_squido_oysters_l1815_181575

theorem squido_oysters (squido crabby : ℕ) : 
  crabby ≥ 2 * squido →
  squido + crabby = 600 →
  squido = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_squido_oysters_l1815_181575


namespace NUMINAMATH_CALUDE_secretary_project_hours_l1815_181519

theorem secretary_project_hours (total_hours : ℕ) (ratio1 ratio2 ratio3 : ℕ) 
  (h1 : ratio1 = 3) (h2 : ratio2 = 7) (h3 : ratio3 = 13) 
  (h_total : total_hours = 253) 
  (h_ratio : ratio1 + ratio2 + ratio3 > 0) :
  (ratio3 * total_hours) / (ratio1 + ratio2 + ratio3) = 143 := by
  sorry

end NUMINAMATH_CALUDE_secretary_project_hours_l1815_181519


namespace NUMINAMATH_CALUDE_no_solution_exists_l1815_181511

theorem no_solution_exists (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : 0 < a₁) (h₂ : a₁ < a₂) (h₃ : a₂ < a₃) (h₄ : a₃ < a₄) :
  ¬ ∃ (k : ℝ) (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧
    x₁ + x₂ + x₃ + x₄ = 1 ∧
    a₁ * x₁ + a₂ * x₂ + a₃ * x₃ + a₄ * x₄ = k ∧
    a₁^2 * x₁ + a₂^2 * x₂ + a₃^2 * x₃ + a₄^2 * x₄ = k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1815_181511


namespace NUMINAMATH_CALUDE_power_24_mod_15_l1815_181537

theorem power_24_mod_15 : 24^2377 % 15 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_24_mod_15_l1815_181537


namespace NUMINAMATH_CALUDE_tax_rate_as_percent_l1815_181581

/-- Given a tax rate of $82 per $100.00, prove that the tax rate expressed as a percent is 82%. -/
theorem tax_rate_as_percent (tax_amount : ℝ) (base_amount : ℝ) :
  tax_amount = 82 ∧ base_amount = 100 →
  (tax_amount / base_amount) * 100 = 82 := by
sorry

end NUMINAMATH_CALUDE_tax_rate_as_percent_l1815_181581


namespace NUMINAMATH_CALUDE_sequence_general_term_l1815_181586

def sequence_property (a : ℕ+ → ℕ+) : Prop :=
  (∀ m k : ℕ+, a (m^2) = (a m)^2) ∧
  (∀ m k : ℕ+, a (m^2 + k^2) = a m * a k)

theorem sequence_general_term (a : ℕ+ → ℕ+) (h : sequence_property a) :
  ∀ n : ℕ+, a n = 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1815_181586


namespace NUMINAMATH_CALUDE_g_27_is_zero_l1815_181517

/-- A function satisfying the given property -/
def special_function (g : ℕ → ℕ) : Prop :=
  ∀ a b c : ℕ, 3 * g (a^2 + b^2 + c^2) = (g a)^2 + (g b)^2 + (g c)^2

/-- The theorem stating that g(27) = 0 for any function satisfying the special property -/
theorem g_27_is_zero (g : ℕ → ℕ) (h : special_function g) : g 27 = 0 := by
  sorry

#check g_27_is_zero

end NUMINAMATH_CALUDE_g_27_is_zero_l1815_181517


namespace NUMINAMATH_CALUDE_polynomial_characterization_l1815_181524

theorem polynomial_characterization (p : ℕ) (hp : Nat.Prime p) :
  ∀ (P : ℕ → ℕ),
    (∀ x : ℕ, x > 0 → P x > x) →
    (∀ m : ℕ, m > 0 → ∃ l : ℕ, m ∣ (Nat.iterate P l p)) →
    (∃ a b : ℕ → ℕ, (∀ x, P x = x + 1 ∨ P x = x + p) ∧
                    (∀ x, a x = x + 1) ∧
                    (∀ x, b x = x + p)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l1815_181524


namespace NUMINAMATH_CALUDE_M_remainder_81_l1815_181551

/-- The largest integer multiple of 9 with no repeated digits and all non-zero digits -/
def M : ℕ :=
  sorry

/-- M is a multiple of 9 -/
axiom M_multiple_of_9 : M % 9 = 0

/-- All digits of M are different -/
axiom M_distinct_digits : ∀ i j, i ≠ j → (M / 10^i % 10) ≠ (M / 10^j % 10)

/-- All digits of M are non-zero -/
axiom M_nonzero_digits : ∀ i, (M / 10^i % 10) ≠ 0

/-- M is the largest such number -/
axiom M_largest : ∀ n, n % 9 = 0 → (∀ i j, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)) → 
                  (∀ i, (n / 10^i % 10) ≠ 0) → n ≤ M

theorem M_remainder_81 : M % 100 = 81 :=
  sorry

end NUMINAMATH_CALUDE_M_remainder_81_l1815_181551


namespace NUMINAMATH_CALUDE_equation_solution_exists_l1815_181587

theorem equation_solution_exists : ∃ x : ℝ, 
  x * 3967 + 36990 - 204790 / 19852 = 322299 ∧ 
  abs (x - 71.924) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l1815_181587


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1815_181547

/-- Given two right triangles and inscribed squares, proves the ratio of their side lengths -/
theorem inscribed_squares_ratio : 
  ∀ (x y : ℝ),
  (∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 ∧ 
    x * (a + b - x) = a * b) →
  (∃ (p q r : ℝ), p = 5 ∧ q = 12 ∧ r = 13 ∧ p^2 + q^2 = r^2 ∧ 
    y * (r - y) = p * q) →
  x / y = 444 / 1183 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1815_181547


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l1815_181542

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) : 
  n ≥ 3 →
  a * b ≠ 0 →
  a = k^2 * b →
  k > 0 →
  (n.choose 2) * (a + b)^(n - 2) * a * b + (n.choose 3) * (a + b)^(n - 3) * a^2 * b = 0 →
  n = 3 * k + 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l1815_181542


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocals_l1815_181563

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  x + y ≤ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b + 1/a + 1/b = 5 ∧ a + b = 4 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_reciprocals_l1815_181563


namespace NUMINAMATH_CALUDE_sqrt_equality_l1815_181568

theorem sqrt_equality (m n : ℝ) (h1 : m > 0) (h2 : 0 ≤ n) (h3 : n ≤ 3*m) :
  Real.sqrt (6*m + 2*Real.sqrt (9*m^2 - n^2)) - Real.sqrt (6*m - 2*Real.sqrt (9*m^2 - n^2)) = 2 * Real.sqrt (3*m - n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_l1815_181568


namespace NUMINAMATH_CALUDE_r_fourth_plus_reciprocal_l1815_181589

theorem r_fourth_plus_reciprocal (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_reciprocal_l1815_181589


namespace NUMINAMATH_CALUDE_system_solvable_iff_l1815_181584

/-- The system of equations -/
def system (b a x y : ℝ) : Prop :=
  x^2 + y^2 + 2*b*(b + x + y) = 81 ∧ y = 5 / ((x - a)^2 + 1)

/-- The theorem stating the condition for the existence of a solution -/
theorem system_solvable_iff (b : ℝ) :
  (∃ a : ℝ, ∃ x y : ℝ, system b a x y) ↔ -14 < b ∧ b < 9 :=
sorry

end NUMINAMATH_CALUDE_system_solvable_iff_l1815_181584


namespace NUMINAMATH_CALUDE_roy_julia_multiple_l1815_181564

-- Define variables for current ages
variable (R J K : ℕ)

-- Define the multiple
variable (M : ℕ)

-- Roy is 6 years older than Julia
def roy_julia_diff : Prop := R = J + 6

-- Roy is half of 6 years older than Kelly
def roy_kelly_diff : Prop := R = K + 3

-- In 4 years, Roy will be some multiple of Julia's age
def future_age_multiple : Prop := R + 4 = M * (J + 4)

-- In 4 years, Roy's age multiplied by Kelly's age would be 108
def future_age_product : Prop := (R + 4) * (K + 4) = 108

theorem roy_julia_multiple
  (h1 : roy_julia_diff R J)
  (h2 : roy_kelly_diff R K)
  (h3 : future_age_multiple R J M)
  (h4 : future_age_product R K) :
  M = 2 := by sorry

end NUMINAMATH_CALUDE_roy_julia_multiple_l1815_181564


namespace NUMINAMATH_CALUDE_product_is_twice_square_l1815_181572

theorem product_is_twice_square (a b c d : ℕ+) (h : a * b = 2 * c * d) :
  ∃ (n : ℕ), a * b * c * d = 2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_product_is_twice_square_l1815_181572


namespace NUMINAMATH_CALUDE_ship_meetings_count_l1815_181506

/-- The number of ships sailing in each direction -/
def num_ships : ℕ := 5

/-- The total number of meetings between two groups of ships -/
def total_meetings (n : ℕ) : ℕ := n * n

/-- Theorem stating that the total number of meetings is 25 -/
theorem ship_meetings_count : total_meetings num_ships = 25 := by
  sorry

end NUMINAMATH_CALUDE_ship_meetings_count_l1815_181506


namespace NUMINAMATH_CALUDE_exactly_three_red_marbles_l1815_181530

def total_marbles : ℕ := 15
def red_marbles : ℕ := 8
def blue_marbles : ℕ := 7
def trials : ℕ := 6
def target_red : ℕ := 3

def probability_red : ℚ := red_marbles / total_marbles
def probability_blue : ℚ := blue_marbles / total_marbles

theorem exactly_three_red_marbles :
  (Nat.choose trials target_red : ℚ) *
  probability_red ^ target_red *
  probability_blue ^ (trials - target_red) =
  6881280 / 38107875 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_red_marbles_l1815_181530


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1815_181588

theorem expression_simplification_and_evaluation :
  let m : ℚ := 2
  let expr := (2 / (m - 3) + 1) / ((2 * m - 2) / (m^2 - 6 * m + 9))
  expr = -1/2 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1815_181588


namespace NUMINAMATH_CALUDE_problem_solution_l1815_181543

def f (x : ℝ) : ℝ := |x - 1|

def A : Set ℝ := {x | -1 < x ∧ x < 1}

theorem problem_solution (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) :
  (∀ x, x ∈ A ↔ f x < 3 - |2*x + 1|) ∧ f (a*b) > f a - f b := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1815_181543


namespace NUMINAMATH_CALUDE_stratified_sampling_properties_l1815_181565

structure School where
  first_year_students : ℕ
  second_year_students : ℕ

def stratified_sample (school : School) (sample_size : ℕ) : 
  ℕ × ℕ :=
  let total_students := school.first_year_students + school.second_year_students
  let first_year_sample := (school.first_year_students * sample_size) / total_students
  let second_year_sample := sample_size - first_year_sample
  (first_year_sample, second_year_sample)

theorem stratified_sampling_properties 
  (school : School)
  (sample_size : ℕ)
  (h1 : school.first_year_students = 1000)
  (h2 : school.second_year_students = 1080)
  (h3 : sample_size = 208) :
  let (first_sample, second_sample) := stratified_sample school sample_size
  -- 1. Students from different grades can be selected simultaneously
  (first_sample > 0 ∧ second_sample > 0) ∧
  -- 2. The number of students selected from each grade is proportional to the grade's population
  (first_sample = 100 ∧ second_sample = 108) ∧
  -- 3. The probability of selection for any student is equal across both grades
  (first_sample / school.first_year_students = second_sample / school.second_year_students) :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_properties_l1815_181565


namespace NUMINAMATH_CALUDE_log_expression_simplification_l1815_181553

theorem log_expression_simplification 
  (a b m : ℝ) 
  (h : m^2 = a^2 - b^2) 
  (h1 : a + b > 0) 
  (h2 : a - b > 0) 
  (h3 : m > 0) :
  Real.log m / Real.log (a + b) + Real.log m / Real.log (a - b) - 
  2 * (Real.log m / Real.log (a + b)) * (Real.log m / Real.log (a - b)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l1815_181553


namespace NUMINAMATH_CALUDE_complement_of_complement_l1815_181501

def V : Finset Nat := {1, 2, 3, 4, 5}

def C_VN : Finset Nat := {2, 4}

def N : Finset Nat := {1, 3, 5}

theorem complement_of_complement (V C_VN N : Finset Nat) 
  (hV : V = {1, 2, 3, 4, 5})
  (hC_VN : C_VN = {2, 4})
  (hN : N = {1, 3, 5}) :
  N = V \ C_VN :=
by sorry

end NUMINAMATH_CALUDE_complement_of_complement_l1815_181501


namespace NUMINAMATH_CALUDE_team_formation_count_l1815_181512

def male_doctors : ℕ := 5
def female_doctors : ℕ := 4
def team_size : ℕ := 3

def team_formations : ℕ := 
  (Nat.choose male_doctors 2 * Nat.choose female_doctors 1) + 
  (Nat.choose male_doctors 1 * Nat.choose female_doctors 2)

theorem team_formation_count : team_formations = 70 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_count_l1815_181512


namespace NUMINAMATH_CALUDE_digital_earth_functions_complete_l1815_181552

-- Define the type for Digital Earth functions
inductive DigitalEarthFunction
  | globalResearch
  | educationalInterface
  | crimeTracking
  | sustainableDevelopment

-- Define the set of all Digital Earth functions
def allFunctions : Set DigitalEarthFunction :=
  {DigitalEarthFunction.globalResearch,
   DigitalEarthFunction.educationalInterface,
   DigitalEarthFunction.crimeTracking,
   DigitalEarthFunction.sustainableDevelopment}

-- Theorem: The set of Digital Earth functions includes all four functions
theorem digital_earth_functions_complete :
  ∀ f : DigitalEarthFunction, f ∈ allFunctions :=
by
  sorry


end NUMINAMATH_CALUDE_digital_earth_functions_complete_l1815_181552


namespace NUMINAMATH_CALUDE_a_5_equals_18_l1815_181527

/-- For a sequence defined by a_n = n^2 - 2n + 3, a_5 = 18 -/
theorem a_5_equals_18 :
  let a : ℕ → ℤ := λ n => n^2 - 2*n + 3
  a 5 = 18 := by sorry

end NUMINAMATH_CALUDE_a_5_equals_18_l1815_181527


namespace NUMINAMATH_CALUDE_egyptian_fraction_odd_divisor_l1815_181580

theorem egyptian_fraction_odd_divisor (n : ℕ) (h_n : n > 1) (h_odd : Odd n) :
  (∃ x y : ℕ, (4 : ℚ) / n = 1 / x + 1 / y) ↔
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∃ k : ℕ, p = 4 * k - 1) :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_odd_divisor_l1815_181580


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1815_181534

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im ((i / (1 + i)) - (1 / (2 * i))) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1815_181534


namespace NUMINAMATH_CALUDE_chessboard_nail_configuration_l1815_181579

/-- Represents a point on the chessboard --/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Checks if three points are collinear --/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- A configuration of 16 points on the chessboard --/
def Configuration := Fin 16 → Point

/-- Predicate to check if a configuration is valid --/
def valid_configuration (config : Configuration) : Prop :=
  (∀ i j k : Fin 16, i ≠ j → j ≠ k → i ≠ k → ¬collinear (config i) (config j) (config k))

theorem chessboard_nail_configuration :
  ∃ (config : Configuration), valid_configuration config :=
sorry

end NUMINAMATH_CALUDE_chessboard_nail_configuration_l1815_181579


namespace NUMINAMATH_CALUDE_range_of_f_l1815_181559

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (x - 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1815_181559


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l1815_181500

/-- Given two parabolas that form a kite when intersecting the coordinate axes, 
    prove that the sum of their coefficients is 1/50 if the kite area is 20 -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    (a * x₁^2 + 3 = 0 ∧ 5 - b * x₁^2 = 0) ∧ 
    (a * x₂^2 + 3 = 0 ∧ 5 - b * x₂^2 = 0) ∧
    (y₁ = a * 0^2 + 3 ∧ y₂ = 5 - b * 0^2) ∧
    (1/2 * (x₂ - x₁) * (y₂ - y₁) = 20)) →
  a + b = 1/50 := by
sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l1815_181500


namespace NUMINAMATH_CALUDE_final_balance_proof_l1815_181566

def bank_transactions (initial_balance : ℚ) : ℚ :=
  let balance1 := initial_balance - 300
  let balance2 := balance1 - 150
  let balance3 := balance2 + (3/5 * balance2)
  let balance4 := balance3 - 250
  balance4 + (2/3 * balance4)

theorem final_balance_proof :
  ∃ (initial_balance : ℚ),
    (300 = (3/7) * initial_balance) ∧
    (150 = (1/3) * (initial_balance - 300)) ∧
    (250 = (1/4) * (initial_balance - 300 - 150 + (3/5) * (initial_balance - 300 - 150))) ∧
    (bank_transactions initial_balance = 1250) :=
by sorry

end NUMINAMATH_CALUDE_final_balance_proof_l1815_181566


namespace NUMINAMATH_CALUDE_alternating_arrangement_white_first_arrangement_group_formation_l1815_181536

/- Define the total number of balls -/
def total_balls : ℕ := 12

/- Define the number of white balls -/
def white_balls : ℕ := 6

/- Define the number of black balls -/
def black_balls : ℕ := 6

/- Define the function to calculate factorial -/
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

/- Define the function to calculate combinations -/
def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

/- Theorem for part (a) -/
theorem alternating_arrangement : 
  factorial white_balls * factorial black_balls = 518400 := by sorry

/- Theorem for part (b) -/
theorem white_first_arrangement : 
  factorial white_balls * factorial black_balls = 518400 := by sorry

/- Theorem for part (c) -/
theorem group_formation : 
  choose white_balls 4 * factorial 4 * choose black_balls 3 * factorial 3 = 43200 := by sorry

end NUMINAMATH_CALUDE_alternating_arrangement_white_first_arrangement_group_formation_l1815_181536


namespace NUMINAMATH_CALUDE_fifth_bounce_height_l1815_181594

/-- Calculates the height of a bouncing ball after a given number of bounces. -/
def bounceHeight (initialHeight : ℝ) (initialEfficiency : ℝ) (efficiencyDecrease : ℝ) (airResistanceLoss : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The height of the ball after the fifth bounce is approximately 0.82 feet. -/
theorem fifth_bounce_height :
  let initialHeight : ℝ := 96
  let initialEfficiency : ℝ := 0.5
  let efficiencyDecrease : ℝ := 0.05
  let airResistanceLoss : ℝ := 0.02
  let bounces : ℕ := 5
  abs (bounceHeight initialHeight initialEfficiency efficiencyDecrease airResistanceLoss bounces - 0.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_fifth_bounce_height_l1815_181594


namespace NUMINAMATH_CALUDE_diana_wins_probability_l1815_181599

def diana_die : ℕ := 6
def apollo_die : ℕ := 4

def favorable_outcomes : ℕ := 14
def total_outcomes : ℕ := diana_die * apollo_die

theorem diana_wins_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 12 :=
sorry

end NUMINAMATH_CALUDE_diana_wins_probability_l1815_181599


namespace NUMINAMATH_CALUDE_power_series_coefficient_l1815_181544

theorem power_series_coefficient (n : ℕ) (t : ℝ) (a : ℕ → ℝ) : 
  (∑' n, a n * t^n / n.factorial) = 
    (∑' k, t^(2*k) / (2*k).factorial)^2 * (∑' j, t^j / j.factorial)^3 → 
  a n = (5^n + 2 * 3^n + 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_power_series_coefficient_l1815_181544


namespace NUMINAMATH_CALUDE_sum_equals_rounded_sum_l1815_181514

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (n : ℕ) : ℕ :=
  let rem := n % 5
  if rem < 3 then n - rem else n + (5 - rem)

def sum_rounded_to_five (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_five (List.range n))

theorem sum_equals_rounded_sum :
  sum_to_n 100 = sum_rounded_to_five 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_rounded_sum_l1815_181514


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l1815_181595

/-- The charge difference for color copies between two print shops -/
theorem print_shop_charge_difference 
  (price_X : ℚ) -- Price per copy at shop X
  (price_Y : ℚ) -- Price per copy at shop Y
  (num_copies : ℕ) -- Number of copies
  (h1 : price_X = 1.25) -- Shop X charges $1.25 per copy
  (h2 : price_Y = 2.75) -- Shop Y charges $2.75 per copy
  (h3 : num_copies = 60) -- We're considering 60 copies
  : (price_Y - price_X) * num_copies = 90 := by
  sorry

#check print_shop_charge_difference

end NUMINAMATH_CALUDE_print_shop_charge_difference_l1815_181595


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1815_181574

theorem simplify_and_evaluate (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  5 * (3 * x^2 * y - x * y^2) - (x * y^2 + 3 * x^2 * y) = 36 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1815_181574
