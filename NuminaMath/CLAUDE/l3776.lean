import Mathlib

namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3776_377683

def is_in_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, k * Real.pi - Real.pi / 4 < α ∧ α < k * Real.pi

theorem half_angle_quadrant (α : Real) :
  is_in_fourth_quadrant α → is_in_second_or_fourth_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3776_377683


namespace NUMINAMATH_CALUDE_congruence_solution_sum_l3776_377614

theorem congruence_solution_sum (a m : ℕ) : 
  m ≥ 2 → 
  0 ≤ a → 
  a < m → 
  (∀ x : ℤ, (8 * x + 1) % 12 = 5 % 12 ↔ x % m = a % m) → 
  a + m = 5 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_sum_l3776_377614


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3776_377629

theorem sum_of_fifth_powers (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (sum_squares_condition : a^2 + b^2 + c^2 = 3)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 4) :
  a^5 + b^5 + c^5 = 11/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3776_377629


namespace NUMINAMATH_CALUDE_student_distribution_problem_l3776_377632

/-- The number of ways to distribute n distinguishable students among k distinguishable schools,
    with each school receiving at least one student. -/
def distribute_students (n k : ℕ) : ℕ :=
  if n < k then 0
  else (k.choose 2) * k.factorial

/-- The problem statement -/
theorem student_distribution_problem :
  distribute_students 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_problem_l3776_377632


namespace NUMINAMATH_CALUDE_two_rational_solutions_l3776_377665

-- Define the system of equations
def system (x y z : ℚ) : Prop :=
  x + y + z = 0 ∧ x * y * z + z = 0 ∧ x * y + y * z + x * z + y = 0

-- Theorem stating that there are exactly two rational solutions
theorem two_rational_solutions :
  ∃! (s : Finset (ℚ × ℚ × ℚ)), s.card = 2 ∧ ∀ (x y z : ℚ), (x, y, z) ∈ s ↔ system x y z :=
sorry

end NUMINAMATH_CALUDE_two_rational_solutions_l3776_377665


namespace NUMINAMATH_CALUDE_like_terms_imply_equation_l3776_377605

/-- Two monomials are like terms if their variables and corresponding exponents are the same -/
def are_like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 1

theorem like_terms_imply_equation (m n : ℕ) :
  are_like_terms m n → m - 2*n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_equation_l3776_377605


namespace NUMINAMATH_CALUDE_age_ratio_l3776_377685

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- Conditions on the ages -/
def validAges (a : Ages) : Prop :=
  a.roy = a.julia + 8 ∧
  a.roy + 2 = 3 * (a.julia + 2) ∧
  (a.roy + 2) * (a.kelly + 2) = 96

/-- The theorem to be proved -/
theorem age_ratio (a : Ages) (h : validAges a) :
  (a.roy - a.julia) / (a.roy - a.kelly) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l3776_377685


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l3776_377663

theorem magic_8_ball_probability :
  let n : ℕ := 6  -- total number of questions
  let k : ℕ := 3  -- number of positive answers we're looking for
  let p : ℚ := 1/3  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 160/729 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l3776_377663


namespace NUMINAMATH_CALUDE_exponent_equation_l3776_377646

theorem exponent_equation (a : ℝ) (m : ℝ) (h1 : a ≠ 0) (h2 : a^5 * (a^m)^3 = a^11) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_l3776_377646


namespace NUMINAMATH_CALUDE_set_operations_l3776_377630

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

theorem set_operations :
  (A ∪ B = {x | x ≥ 3}) ∧
  (A ∩ B = {x | 4 ≤ x ∧ x < 10}) ∧
  ((Aᶜ ∩ B) ∩ (A ∪ B) = {x | (3 ≤ x ∧ x < 4) ∨ x ≥ 10}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3776_377630


namespace NUMINAMATH_CALUDE_staircase_classroom_seats_l3776_377689

/-- Represents the number of seats in a row of the staircase classroom. -/
def seats (n : ℕ) (a : ℕ) : ℕ := 12 + (n - 1) * a

theorem staircase_classroom_seats :
  ∃ a : ℕ,
  (seats 15 a = 2 * seats 5 a) ∧ 
  (seats 21 a = 52) := by
  sorry

end NUMINAMATH_CALUDE_staircase_classroom_seats_l3776_377689


namespace NUMINAMATH_CALUDE_car_speed_problem_l3776_377625

/-- Proves that given the conditions, car R's average speed is 50 miles per hour -/
theorem car_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ) :
  distance = 800 →
  time_diff = 2 →
  speed_diff = 10 →
  ∃ (speed_R : ℝ),
    distance / speed_R - time_diff = distance / (speed_R + speed_diff) ∧
    speed_R = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3776_377625


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3776_377658

/-- The line y = mx + (2m + 1), where m ∈ ℝ, always passes through the point (-2, 1). -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ((-2 : ℝ) : ℝ) * m + (2 * m + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3776_377658


namespace NUMINAMATH_CALUDE_senior_to_child_ratio_l3776_377694

theorem senior_to_child_ratio 
  (adults : ℕ) 
  (children : ℕ) 
  (seniors : ℕ) 
  (total : ℕ) 
  (h1 : adults = 58)
  (h2 : children = adults - 35)
  (h3 : total = adults + children + seniors)
  (h4 : total = 127) :
  (seniors : ℚ) / children = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_senior_to_child_ratio_l3776_377694


namespace NUMINAMATH_CALUDE_min_value_theorem_l3776_377631

theorem min_value_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 →
    a / (Real.sin θ)^(3/2) + b / (Real.cos θ)^(3/2) ≥ (a^(4/7) + b^(4/7))^(7/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3776_377631


namespace NUMINAMATH_CALUDE_intersection_determines_B_l3776_377672

def A : Set ℝ := {0, 1, 2, 3}

def B (m : ℝ) : Set ℝ := {x | x^2 - 5*x + m = 0}

theorem intersection_determines_B :
  ∃ m : ℝ, (A ∩ B m = {1}) → (B m = {1, 4}) := by sorry

end NUMINAMATH_CALUDE_intersection_determines_B_l3776_377672


namespace NUMINAMATH_CALUDE_relationship_between_x_and_z_l3776_377660

theorem relationship_between_x_and_z (x y z : ℝ) 
  (h1 : x = y * 1.027)  -- x is 2.7% greater than y
  (h2 : y = z * 0.45)   -- y is 55% less than z
  : x = z * (1 - 0.53785) :=  -- x is 53.785% less than z
by sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_z_l3776_377660


namespace NUMINAMATH_CALUDE_total_score_is_210_l3776_377627

/-- Represents the test scores of three students -/
structure TestScores where
  total_questions : ℕ
  marks_per_question : ℕ
  jose_wrong_questions : ℕ
  meghan_diff : ℕ
  jose_alisson_diff : ℕ

/-- Calculates the total score for three students given their test performance -/
def calculate_total_score (scores : TestScores) : ℕ :=
  let total_marks := scores.total_questions * scores.marks_per_question
  let jose_score := total_marks - (scores.jose_wrong_questions * scores.marks_per_question)
  let meghan_score := jose_score - scores.meghan_diff
  let alisson_score := jose_score - scores.jose_alisson_diff
  jose_score + meghan_score + alisson_score

/-- Theorem stating that the total score for the three students is 210 marks -/
theorem total_score_is_210 (scores : TestScores) 
  (h1 : scores.total_questions = 50)
  (h2 : scores.marks_per_question = 2)
  (h3 : scores.jose_wrong_questions = 5)
  (h4 : scores.meghan_diff = 20)
  (h5 : scores.jose_alisson_diff = 40) :
  calculate_total_score scores = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_210_l3776_377627


namespace NUMINAMATH_CALUDE_fibonacci_13th_term_l3776_377601

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_13th_term : fibonacci 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_13th_term_l3776_377601


namespace NUMINAMATH_CALUDE_pine_tree_branches_l3776_377603

/-- The number of branches in a pine tree -/
def num_branches : ℕ := 23

/-- The movements of the squirrel from the middle branch to the top -/
def movements : List ℤ := [5, -7, 4, 9]

/-- The number of branches from the middle to the top -/
def branches_to_top : ℕ := (movements.sum).toNat

theorem pine_tree_branches :
  num_branches = 2 * branches_to_top + 1 :=
by sorry

end NUMINAMATH_CALUDE_pine_tree_branches_l3776_377603


namespace NUMINAMATH_CALUDE_pants_price_proof_l3776_377662

/-- Given the total cost of a pair of pants and a belt, and the price difference between them,
    prove that the price of the pants is as stated. -/
theorem pants_price_proof (total_cost belt_price pants_price : ℝ) : 
  total_cost = 70.93 →
  pants_price = belt_price - 2.93 →
  total_cost = belt_price + pants_price →
  pants_price = 34.00 := by
sorry

end NUMINAMATH_CALUDE_pants_price_proof_l3776_377662


namespace NUMINAMATH_CALUDE_square_of_negative_product_l3776_377686

theorem square_of_negative_product (b : ℝ) : (-3 * b)^2 = 9 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l3776_377686


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3776_377619

/-- Given a cone with base radius 5 and slant height 15, prove that the central angle of the sector in the unfolded lateral surface is 120 degrees -/
theorem cone_lateral_surface_angle (base_radius : ℝ) (slant_height : ℝ) (central_angle : ℝ) : 
  base_radius = 5 → 
  slant_height = 15 → 
  central_angle * slant_height / 180 * π = 2 * π * base_radius → 
  central_angle = 120 := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3776_377619


namespace NUMINAMATH_CALUDE_mean_home_runs_l3776_377640

def home_runs : List (Nat × Nat) := [(5, 5), (9, 3), (7, 4), (11, 2)]

theorem mean_home_runs :
  let total_home_runs := (home_runs.map (λ (hr, players) => hr * players)).sum
  let total_players := (home_runs.map (λ (_, players) => players)).sum
  (total_home_runs : ℚ) / total_players = 729/100 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3776_377640


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l3776_377623

theorem min_x_prime_factorization_sum : ∃ (x y p q r : ℕ+) (a b c : ℕ),
  (3 : ℚ) * (x : ℚ)^7 = 5 * (y : ℚ)^11 ∧
  x = p^a * q^b * r^c ∧
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  (∀ (x' : ℕ+), (3 : ℚ) * (x' : ℚ)^7 = 5 * (y : ℚ)^11 → x ≤ x') →
  p + q + r + a + b + c = 24 :=
sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l3776_377623


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3776_377684

/-- The lateral surface area of a cone with base radius 6 and slant height 15 is 90π. -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 6 → l = 15 → π * r * l = 90 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3776_377684


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3776_377677

theorem least_subtraction_for_divisibility : ∃! k : ℕ, 
  k ≤ 16 ∧ (762429836 - k) % 17 = 0 ∧ 
  ∀ m : ℕ, m < k → (762429836 - m) % 17 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3776_377677


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_three_l3776_377639

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The first line equation: ax + 3y + 4 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 4 = 0

/-- The second line equation: x + (a-2)y + a^2 - 5 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 2) * y + a^2 - 5 = 0

/-- Theorem: If the two lines are parallel, then a = 3 -/
theorem parallel_lines_a_equals_three :
  ∀ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) → a = 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_three_l3776_377639


namespace NUMINAMATH_CALUDE_voting_scenario_theorem_l3776_377628

/-- Represents the voting scenario in a certain city -/
structure VotingScenario where
  total_voters : ℝ
  dem_percent : ℝ
  rep_percent : ℝ
  dem_for_A_percent : ℝ
  total_for_A_percent : ℝ
  rep_for_A_percent : ℝ

/-- The theorem statement for the voting scenario problem -/
theorem voting_scenario_theorem (v : VotingScenario) :
  v.dem_percent = 0.6 ∧
  v.rep_percent = 0.4 ∧
  v.dem_for_A_percent = 0.75 ∧
  v.total_for_A_percent = 0.57 →
  v.rep_for_A_percent = 0.3 := by
  sorry


end NUMINAMATH_CALUDE_voting_scenario_theorem_l3776_377628


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3776_377682

theorem quadratic_equal_roots (b : ℝ) :
  (∃ x : ℝ, b * x^2 + 2 * b * x + 4 = 0 ∧
   ∀ y : ℝ, b * y^2 + 2 * b * y + 4 = 0 → y = x) →
  b = 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3776_377682


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l3776_377636

/-- Converts a natural number to its base 7 representation --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list --/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball placement process up to n steps --/
def ballPlacement (n : ℕ) : ℕ :=
  sorry

theorem ball_placement_theorem :
  ballPlacement 1729 = sumDigits (toBase7 1729) :=
sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l3776_377636


namespace NUMINAMATH_CALUDE_percentage_increase_l3776_377613

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 500 → final = 650 → (final - initial) / initial * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3776_377613


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3776_377617

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 5}

theorem complement_intersection_theorem :
  (A ∩ B)ᶜ = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3776_377617


namespace NUMINAMATH_CALUDE_wrapping_paper_ratio_l3776_377618

theorem wrapping_paper_ratio : 
  ∀ (p1 p2 p3 : ℝ),
  p1 = 2 →
  p3 = p1 + p2 →
  p1 + p2 + p3 = 7 →
  p2 / p1 = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_ratio_l3776_377618


namespace NUMINAMATH_CALUDE_stating_constant_sum_of_products_l3776_377692

/-- 
Represents the sum of all products of pile sizes during the division process
for n balls.
-/
def f (n : ℕ) : ℕ := sorry

/-- 
Theorem stating that the sum of all products of pile sizes during the division
process is constant for any division strategy.
-/
theorem constant_sum_of_products (n : ℕ) (h : n > 0) :
  ∀ (strategy1 strategy2 : ℕ → ℕ × ℕ),
  (∀ k, k ≤ n → (strategy1 k).1 + (strategy1 k).2 = k) →
  (∀ k, k ≤ n → (strategy2 k).1 + (strategy2 k).2 = k) →
  f n = f n :=
by sorry

/--
Lemma showing that f(n) equals n(n-1)/2 for all positive integers n.
This represents the insight from the solution, but is not directly
assumed from the problem statement.
-/
lemma f_equals_combinations (n : ℕ) (h : n > 0) :
  f n = n * (n - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_stating_constant_sum_of_products_l3776_377692


namespace NUMINAMATH_CALUDE_sum_distinct_remainders_divided_by_13_l3776_377616

def distinct_remainders (n : ℕ) : Finset ℕ :=
  (Finset.range n).image (λ i => (i + 1)^2 % 13)

theorem sum_distinct_remainders_divided_by_13 :
  (Finset.sum (distinct_remainders 12) id) / 13 = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_distinct_remainders_divided_by_13_l3776_377616


namespace NUMINAMATH_CALUDE_common_factor_is_gcf_l3776_377695

-- Define the expression
def expression (a b c : ℤ) : ℤ := 8 * a^3 * b^2 - 12 * a * b^3 * c + 2 * a * b

-- Define the common factor
def common_factor (a b : ℤ) : ℤ := 2 * a * b

-- Theorem statement
theorem common_factor_is_gcf (a b c : ℤ) :
  (∃ k₁ k₂ k₃ : ℤ, 
    expression a b c = common_factor a b * (k₁ + k₂ + k₃) ∧
    k₁ = 4 * a^2 * b ∧
    k₂ = -6 * b^2 * c ∧
    k₃ = 1) ∧
  (∀ d : ℤ, d ∣ expression a b c → d ∣ common_factor a b ∨ d = 1 ∨ d = -1) :=
sorry

end NUMINAMATH_CALUDE_common_factor_is_gcf_l3776_377695


namespace NUMINAMATH_CALUDE_three_digit_repeating_decimal_cube_l3776_377656

theorem three_digit_repeating_decimal_cube (n : ℕ) : 
  (n < 1000 ∧ n > 0) →
  (∃ (a b : ℕ), b > a ∧ a > 0 ∧ b > 0 ∧ (n : ℚ) / 999 = (a : ℚ) / b ^ 3) →
  (n = 037 ∨ n = 296) :=
sorry

end NUMINAMATH_CALUDE_three_digit_repeating_decimal_cube_l3776_377656


namespace NUMINAMATH_CALUDE_probability_two_females_selected_l3776_377607

/-- The probability of selecting 2 females out of 6 finalists (4 females and 2 males) -/
theorem probability_two_females_selected (total : Nat) (females : Nat) (selected : Nat) 
  (h1 : total = 6) 
  (h2 : females = 4)
  (h3 : selected = 2) : 
  (Nat.choose females selected : ℚ) / (Nat.choose total selected) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_selected_l3776_377607


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3776_377654

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 60 ∧ 
  white = 22 ∧ 
  green = 18 ∧ 
  red = 5 ∧ 
  purple = 7 ∧ 
  prob = 4/5 ∧ 
  (white + green + (total - white - green - red - purple) : ℚ) / total = prob →
  total - white - green - red - purple = 8 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3776_377654


namespace NUMINAMATH_CALUDE_jose_age_l3776_377670

/-- Given the ages of Jose, Zack, and Inez, prove that Jose is 21 years old -/
theorem jose_age (jose zack inez : ℕ) 
  (h1 : jose = zack + 5) 
  (h2 : zack = inez + 4) 
  (h3 : inez = 12) : 
  jose = 21 := by
  sorry

end NUMINAMATH_CALUDE_jose_age_l3776_377670


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l3776_377653

def alice_number : ℕ := 30

def has_all_prime_factors_except_7 (n : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → p ∣ alice_number → p ≠ 7 → p ∣ n

theorem smallest_number_with_conditions :
  ∃ (bob_number : ℕ), bob_number > 0 ∧
  has_all_prime_factors_except_7 bob_number ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors_except_7 m → bob_number ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l3776_377653


namespace NUMINAMATH_CALUDE_water_level_rise_l3776_377661

/-- Calculates the rise in water level when a cube is fully immersed in a rectangular vessel. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) :
  cube_edge = 17 →
  vessel_length = 20 →
  vessel_width = 15 →
  ∃ (water_rise : ℝ), abs (water_rise - (cube_edge^3 / (vessel_length * vessel_width))) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_l3776_377661


namespace NUMINAMATH_CALUDE_all_measurements_correct_l3776_377609

-- Define a structure for measurements
structure Measurement where
  value : Float
  unit : String

-- Define the measurements
def ruler_length : Measurement := { value := 2, unit := "decimeters" }
def truck_capacity : Measurement := { value := 5, unit := "tons" }
def bus_speed : Measurement := { value := 100, unit := "kilometers" }
def book_thickness : Measurement := { value := 7, unit := "millimeters" }
def backpack_weight : Measurement := { value := 4000, unit := "grams" }

-- Define propositions for correct units
def correct_ruler_unit (m : Measurement) : Prop := m.unit = "decimeters"
def correct_truck_unit (m : Measurement) : Prop := m.unit = "tons"
def correct_bus_unit (m : Measurement) : Prop := m.unit = "kilometers"
def correct_book_unit (m : Measurement) : Prop := m.unit = "millimeters"
def correct_backpack_unit (m : Measurement) : Prop := m.unit = "grams"

-- Theorem stating that all measurements have correct units
theorem all_measurements_correct : 
  correct_ruler_unit ruler_length ∧
  correct_truck_unit truck_capacity ∧
  correct_bus_unit bus_speed ∧
  correct_book_unit book_thickness ∧
  correct_backpack_unit backpack_weight :=
by sorry


end NUMINAMATH_CALUDE_all_measurements_correct_l3776_377609


namespace NUMINAMATH_CALUDE_solution_set_max_value_min_value_l3776_377693

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |2*x - 2|

-- Theorem 1: Solution set of f(x) ≥ x-1
theorem solution_set (x : ℝ) : f x ≥ x - 1 ↔ 0 ≤ x ∧ x ≤ 2 :=
sorry

-- Theorem 2: Maximum value of f
theorem max_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 2 :=
sorry

-- Theorem 3: Minimum value of expression
theorem min_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 2) :
  (b^2 / a) + (c^2 / b) + (a^2 / c) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_max_value_min_value_l3776_377693


namespace NUMINAMATH_CALUDE_right_triangle_altitude_reciprocal_squares_l3776_377696

/-- In a right triangle with sides a and b, hypotenuse c, and altitude x drawn on the hypotenuse,
    the following equation holds: 1/x² = 1/a² + 1/b² -/
theorem right_triangle_altitude_reciprocal_squares 
  (a b c x : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_altitude : a * b = c * x) : 
  1 / x^2 = 1 / a^2 + 1 / b^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_reciprocal_squares_l3776_377696


namespace NUMINAMATH_CALUDE_inequality_proof_l3776_377674

theorem inequality_proof (x : ℝ) : (x - 5) / ((x - 3)^2 + 1) < 0 ↔ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3776_377674


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3776_377688

theorem pipe_filling_time (p q r t : ℝ) (hp : p = 6) (hr : r = 24) (ht : t = 3.4285714285714284)
  (h_total : 1/p + 1/q + 1/r = 1/t) : q = 8 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l3776_377688


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3776_377604

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3776_377604


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l3776_377659

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y : ℝ, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l3776_377659


namespace NUMINAMATH_CALUDE_prime_sum_squares_l3776_377678

theorem prime_sum_squares (a b c d : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d →
  a > 3 →
  b > 6 →
  c > 12 →
  a^2 - b^2 + c^2 - d^2 = 1749 →
  a^2 + b^2 + c^2 + d^2 = 1999 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l3776_377678


namespace NUMINAMATH_CALUDE_complex_power_2015_l3776_377626

/-- Given a complex number i such that i^2 = -1, i^3 = -i, and i^4 = 1,
    prove that i^2015 = -i -/
theorem complex_power_2015 (i : ℂ) (hi2 : i^2 = -1) (hi3 : i^3 = -i) (hi4 : i^4 = 1) :
  i^2015 = -i := by sorry

end NUMINAMATH_CALUDE_complex_power_2015_l3776_377626


namespace NUMINAMATH_CALUDE_fermat_point_sum_l3776_377600

theorem fermat_point_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + y^2 + x*y = 1)
  (h2 : y^2 + z^2 + y*z = 2)
  (h3 : z^2 + x^2 + z*x = 3) :
  x + y + z = Real.sqrt (3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_fermat_point_sum_l3776_377600


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3776_377647

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (3 * a 3 ^ 2 - 11 * a 3 + 9 = 0) →
  (3 * a 9 ^ 2 - 11 * a 9 + 9 = 0) →
  (a 5 * a 6 * a 7 = 3 * Real.sqrt 3 ∨ a 5 * a 6 * a 7 = -3 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l3776_377647


namespace NUMINAMATH_CALUDE_cosine_A_in_special_triangle_l3776_377676

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem cosine_A_in_special_triangle (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)  -- Sum of angles in a triangle
  (h2 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)  -- Positive side lengths
  (h3 : Real.sin t.A / 4 = Real.sin t.B / 5)  -- Given ratio
  (h4 : Real.sin t.B / 5 = Real.sin t.C / 6)  -- Given ratio
  : Real.cos t.A = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_A_in_special_triangle_l3776_377676


namespace NUMINAMATH_CALUDE_relationship_abc_l3776_377681

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow 0.6 0.4)
  (hb : b = Real.rpow 0.4 0.6)
  (hc : c = Real.rpow 0.4 0.4) :
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3776_377681


namespace NUMINAMATH_CALUDE_coefficient_x6y4_in_expansion_l3776_377697

theorem coefficient_x6y4_in_expansion : ∀ x y : ℝ,
  (Nat.choose 10 4 : ℝ) = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6y4_in_expansion_l3776_377697


namespace NUMINAMATH_CALUDE_original_alcohol_percentage_l3776_377610

/-- Proves that a 20-litre mixture of alcohol and water, when mixed with 3 litres of water,
    resulting in a new mixture with 17.391304347826086% alcohol, must have originally
    contained 20% alcohol. -/
theorem original_alcohol_percentage
  (original_volume : ℝ)
  (added_water : ℝ)
  (new_percentage : ℝ)
  (h1 : original_volume = 20)
  (h2 : added_water = 3)
  (h3 : new_percentage = 17.391304347826086) :
  (original_volume * (100 / (original_volume + added_water)) * new_percentage / 100) = 20 :=
sorry

end NUMINAMATH_CALUDE_original_alcohol_percentage_l3776_377610


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l3776_377622

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 42

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 70

theorem fgh_supermarkets_count :
  us_supermarkets = 42 ∧
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 14 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l3776_377622


namespace NUMINAMATH_CALUDE_min_degree_of_g_l3776_377679

variable (x : ℝ)
variable (f g h : ℝ → ℝ)

def is_polynomial (p : ℝ → ℝ) : Prop := sorry

def degree (p : ℝ → ℝ) : ℕ := sorry

theorem min_degree_of_g 
  (hpoly : is_polynomial f ∧ is_polynomial g ∧ is_polynomial h)
  (heq : ∀ x, 2 * f x + 5 * g x = h x)
  (hf : degree f = 7)
  (hh : degree h = 10) :
  degree g ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_degree_of_g_l3776_377679


namespace NUMINAMATH_CALUDE_tom_july_books_l3776_377667

/-- The number of books Tom read in May -/
def may_books : ℕ := 2

/-- The number of books Tom read in June -/
def june_books : ℕ := 6

/-- The total number of books Tom read -/
def total_books : ℕ := 18

/-- The number of books Tom read in July -/
def july_books : ℕ := total_books - may_books - june_books

theorem tom_july_books : july_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_tom_july_books_l3776_377667


namespace NUMINAMATH_CALUDE_third_layer_sugar_l3776_377691

def sugar_for_cake (smallest_layer sugar_second_layer sugar_third_layer : ℕ) : Prop :=
  (sugar_second_layer = 2 * smallest_layer) ∧ 
  (sugar_third_layer = 3 * sugar_second_layer)

theorem third_layer_sugar : ∀ (smallest_layer sugar_second_layer sugar_third_layer : ℕ),
  smallest_layer = 2 →
  sugar_for_cake smallest_layer sugar_second_layer sugar_third_layer →
  sugar_third_layer = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_layer_sugar_l3776_377691


namespace NUMINAMATH_CALUDE_difference_of_squares_25_7_l3776_377668

theorem difference_of_squares_25_7 : (25 + 7)^2 - (25 - 7)^2 = 700 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_25_7_l3776_377668


namespace NUMINAMATH_CALUDE_toms_age_ratio_l3776_377680

theorem toms_age_ratio (T M : ℚ) : 
  (∃ (children_sum : ℚ), 
    children_sum = T ∧ 
    T - M = 3 * (children_sum - 4 * M)) → 
  T / M = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l3776_377680


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l3776_377645

def num_white_socks : Nat := 5
def num_brown_socks : Nat := 3
def num_blue_socks : Nat := 2
def num_red_socks : Nat := 2

def total_socks : Nat := num_white_socks + num_brown_socks + num_blue_socks + num_red_socks

def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem same_color_sock_pairs : 
  choose num_white_socks 2 + choose num_brown_socks 2 + choose num_blue_socks 2 + choose num_red_socks 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l3776_377645


namespace NUMINAMATH_CALUDE_pentadecagon_diagonals_l3776_377675

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a 15-sided polygon -/
def pentadecagon_sides : ℕ := 15

/-- Theorem: The number of diagonals in a convex pentadecagon is 90 -/
theorem pentadecagon_diagonals : 
  num_diagonals pentadecagon_sides = 90 := by sorry

end NUMINAMATH_CALUDE_pentadecagon_diagonals_l3776_377675


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3776_377652

def C : Set Nat := {64, 66, 67, 68, 71}

def has_smallest_prime_factor (n : Nat) (s : Set Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, ∃ p : Nat, Prime p ∧ p ∣ n ∧ ∀ q : Nat, Prime q → q ∣ m → p ≤ q

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 64 C ∧
  has_smallest_prime_factor 66 C ∧
  has_smallest_prime_factor 68 C :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3776_377652


namespace NUMINAMATH_CALUDE_root_condition_l3776_377620

open Real

theorem root_condition (a : ℝ) :
  (∃ x : ℝ, x ≥ (exp 1) ∧ a + log x = 0) →
  a ≤ -1 ∧
  (∃ a : ℝ, a ≤ -1 ∧ ∃ x : ℝ, x ≥ (exp 1) ∧ a + log x = 0) ∧
  (∃ a : ℝ, a > -1 ∧ ∀ x : ℝ, x ≥ (exp 1) → a + log x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_root_condition_l3776_377620


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_implies_value_l3776_377637

theorem absolute_value_sum_zero_implies_value (x y : ℝ) :
  |x - 4| + |5 + y| = 0 → 2*x + 3*y = -7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_implies_value_l3776_377637


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3776_377608

/-- Given a square with perimeter 100 units divided vertically into 4 congruent rectangles,
    the perimeter of one of these rectangles is 62.5 units. -/
theorem rectangle_perimeter (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 100) : 
  2 * (s + s / 4) = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3776_377608


namespace NUMINAMATH_CALUDE_binary_1101_to_base5_l3776_377649

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its base-5 representation -/
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: decimal_to_base5 (n / 5)

/-- The binary representation of the number we want to convert -/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_to_base5 :
  decimal_to_base5 (binary_to_decimal binary_1101) = [3, 2] :=
by sorry

end NUMINAMATH_CALUDE_binary_1101_to_base5_l3776_377649


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l3776_377634

theorem nested_sqrt_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l3776_377634


namespace NUMINAMATH_CALUDE_fibonacci_gcd_l3776_377602

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_gcd :
  Nat.gcd (fib 2017) (fib 99 * fib 101 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_gcd_l3776_377602


namespace NUMINAMATH_CALUDE_minimum_cut_length_l3776_377664

theorem minimum_cut_length (longer_strip shorter_strip : ℝ) 
  (h1 : longer_strip = 23)
  (h2 : shorter_strip = 15) : 
  ∃ x : ℝ, x ≥ 7 ∧ ∀ y : ℝ, y ≥ 0 → longer_strip - y ≥ 2 * (shorter_strip - y) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_minimum_cut_length_l3776_377664


namespace NUMINAMATH_CALUDE_odometer_puzzle_l3776_377611

theorem odometer_puzzle (a b c : ℕ) 
  (h1 : a ≥ 1) 
  (h2 : 100 ≤ a * b * c ∧ a * b * c ≤ 300)
  (h3 : 75 ∣ b)
  (h4 : (a * b * c) + b - a * b * c = b) :
  a^2 + b^2 + c^2 = 5635 := by
sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l3776_377611


namespace NUMINAMATH_CALUDE_damaged_glassware_count_l3776_377642

-- Define the constants from the problem
def total_glassware : ℕ := 1500
def undamaged_fee : ℚ := 5/2
def damaged_fee : ℕ := 3
def total_received : ℕ := 3618

-- Define the theorem
theorem damaged_glassware_count :
  ∃ x : ℕ, 
    x ≤ total_glassware ∧ 
    (undamaged_fee * (total_glassware - x) : ℚ) - (damaged_fee * x : ℚ) = total_received ∧
    x = 24 := by
  sorry

end NUMINAMATH_CALUDE_damaged_glassware_count_l3776_377642


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l3776_377612

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else (10 : ℝ) ^ x

-- State the theorem
theorem f_composition_negative_two :
  f (f (-2)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l3776_377612


namespace NUMINAMATH_CALUDE_function_properties_l3776_377644

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + a

-- Define the theorem
theorem function_properties (a : ℝ) :
  -- Condition: x ∈ [-π/6, π/3]
  (∀ x, -π/6 ≤ x ∧ x ≤ π/3 →
    -- Condition: sum of max and min values is 3/2
    (⨆ x, f x a) + (⨅ x, f x a) = 3/2) →
  -- 1. Smallest positive period is π
  (∀ x, f (x + π) a = f x a) ∧
  (∀ T, T > 0 ∧ (∀ x, f (x + T) a = f x a) → T ≥ π) ∧
  -- 2. Interval of monotonic decrease
  (∀ k : ℤ, ∀ x y, k * π + π/6 ≤ x ∧ x ≤ y ∧ y ≤ k * π + 2*π/3 →
    f y a ≤ f x a) ∧
  -- 3. Solution set of f(x) > 1
  (∀ x, 0 < x ∧ x < π/3 ↔ f x a > 1) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3776_377644


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l3776_377690

/-- Proves that the percentage increase in the price of a candy box is 25% --/
theorem candy_box_price_increase 
  (new_candy_price : ℝ) 
  (new_soda_price : ℝ) 
  (original_total : ℝ) 
  (h1 : new_candy_price = 15)
  (h2 : new_soda_price = 6)
  (h3 : new_soda_price = (3/2) * (original_total - new_candy_price + new_soda_price))
  (h4 : original_total = 16) :
  (new_candy_price - (original_total - (2/3) * new_soda_price)) / (original_total - (2/3) * new_soda_price) = 1/4 := by
  sorry

#check candy_box_price_increase

end NUMINAMATH_CALUDE_candy_box_price_increase_l3776_377690


namespace NUMINAMATH_CALUDE_expression_evaluation_l3776_377643

theorem expression_evaluation (x : ℝ) (h1 : x^5 + 1 ≠ 0) (h2 : x^5 - 1 ≠ 0) :
  let expr := (((x^2 - 2*x + 2)^2 * (x^3 - x^2 + 1)^2) / (x^5 + 1)^2)^2 *
               (((x^2 + 2*x + 2)^2 * (x^3 + x^2 + 1)^2) / (x^5 - 1)^2)^2
  expr = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3776_377643


namespace NUMINAMATH_CALUDE_sarah_cupcake_ratio_l3776_377621

theorem sarah_cupcake_ratio :
  ∀ (michael_cookies sarah_initial_cupcakes sarah_final_desserts : ℕ)
    (sarah_saved_cupcakes : ℕ),
  michael_cookies = 5 →
  sarah_initial_cupcakes = 9 →
  sarah_final_desserts = 11 →
  sarah_final_desserts = sarah_initial_cupcakes - sarah_saved_cupcakes + michael_cookies →
  (sarah_saved_cupcakes : ℚ) / sarah_initial_cupcakes = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sarah_cupcake_ratio_l3776_377621


namespace NUMINAMATH_CALUDE_sin_half_angle_second_quadrant_l3776_377641

theorem sin_half_angle_second_quadrant (θ : Real) 
  (h1 : π < θ ∧ θ < 3*π/2) 
  (h2 : 25 * Real.sin θ ^ 2 + Real.sin θ - 24 = 0) : 
  Real.sin (θ/2) = 4/5 ∨ Real.sin (θ/2) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_half_angle_second_quadrant_l3776_377641


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l3776_377650

/-- 
Given a quadratic polynomial x^2 + bx + 3024, this theorem states that
111 is the smallest positive integer b for which the polynomial factors
into a product of two binomials with integer coefficients.
-/
theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), ∃ (r s : ℤ), x^2 + b*x + 3024 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬(∀ (x : ℤ), ∃ (r s : ℤ), x^2 + b'*x + 3024 = (x + r) * (x + s))) ∧
  b = 111 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l3776_377650


namespace NUMINAMATH_CALUDE_average_marks_all_candidates_l3776_377671

/-- Proves that the average marks of all candidates is 35 given the specified conditions -/
theorem average_marks_all_candidates
  (total_candidates : ℕ)
  (passed_candidates : ℕ)
  (failed_candidates : ℕ)
  (avg_marks_passed : ℚ)
  (avg_marks_failed : ℚ)
  (h1 : total_candidates = 120)
  (h2 : passed_candidates = 100)
  (h3 : failed_candidates = total_candidates - passed_candidates)
  (h4 : avg_marks_passed = 39)
  (h5 : avg_marks_failed = 15) :
  (passed_candidates * avg_marks_passed + failed_candidates * avg_marks_failed) / total_candidates = 35 :=
by
  sorry

#check average_marks_all_candidates

end NUMINAMATH_CALUDE_average_marks_all_candidates_l3776_377671


namespace NUMINAMATH_CALUDE_cannot_row_against_fast_stream_l3776_377638

/-- A man rowing a boat in a stream -/
structure Rower where
  speedWithStream : ℝ
  speedInStillWater : ℝ

/-- Determine if a rower can go against the stream -/
def canRowAgainstStream (r : Rower) : Prop :=
  r.speedInStillWater > r.speedWithStream - r.speedInStillWater

/-- Theorem: A man cannot row against the stream if his speed in still water
    is less than the stream's speed -/
theorem cannot_row_against_fast_stream (r : Rower)
  (h1 : r.speedWithStream = 10)
  (h2 : r.speedInStillWater = 2) :
  ¬(canRowAgainstStream r) := by
  sorry

#check cannot_row_against_fast_stream

end NUMINAMATH_CALUDE_cannot_row_against_fast_stream_l3776_377638


namespace NUMINAMATH_CALUDE_jerry_showers_l3776_377699

/-- Represents the water usage scenario for Jerry's household --/
structure WaterUsage where
  total_allowance : ℕ
  drinking_cooking : ℕ
  shower_usage : ℕ
  pool_length : ℕ
  pool_width : ℕ
  pool_height : ℕ
  gallon_to_cubic_foot : ℕ

/-- Calculates the number of showers Jerry can take in July --/
def calculate_showers (w : WaterUsage) : ℕ :=
  let pool_volume := w.pool_length * w.pool_width * w.pool_height
  let remaining_water := w.total_allowance - w.drinking_cooking - pool_volume
  remaining_water / w.shower_usage

/-- Theorem stating that Jerry can take 15 showers in July --/
theorem jerry_showers :
  let w : WaterUsage := {
    total_allowance := 1000,
    drinking_cooking := 100,
    shower_usage := 20,
    pool_length := 10,
    pool_width := 10,
    pool_height := 6,
    gallon_to_cubic_foot := 1
  }
  calculate_showers w = 15 := by
  sorry

#eval calculate_showers {
  total_allowance := 1000,
  drinking_cooking := 100,
  shower_usage := 20,
  pool_length := 10,
  pool_width := 10,
  pool_height := 6,
  gallon_to_cubic_foot := 1
}

end NUMINAMATH_CALUDE_jerry_showers_l3776_377699


namespace NUMINAMATH_CALUDE_josh_marbles_l3776_377673

theorem josh_marbles (initial : ℕ) (lost : ℕ) (difference : ℕ) (found : ℕ) : 
  initial = 15 →
  lost = 23 →
  difference = 14 →
  lost = found + difference →
  found = 9 := by
sorry

end NUMINAMATH_CALUDE_josh_marbles_l3776_377673


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l3776_377651

theorem quadratic_is_perfect_square (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 21*x + c = (a*x + b)^2) → c = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l3776_377651


namespace NUMINAMATH_CALUDE_least_integer_with_nine_factors_l3776_377633

/-- A function that returns the number of distinct positive factors of a positive integer -/
def number_of_factors (n : ℕ+) : ℕ :=
  sorry

/-- A function that checks if a number has exactly nine distinct positive factors -/
def has_nine_factors (n : ℕ+) : Prop :=
  number_of_factors n = 9

theorem least_integer_with_nine_factors :
  ∃ (n : ℕ+), has_nine_factors n ∧ ∀ (m : ℕ+), has_nine_factors m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_nine_factors_l3776_377633


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3776_377635

theorem solve_linear_equation :
  ∃ x : ℚ, 3*x - 5*x + 9*x + 4 = 289 ∧ x = 285/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3776_377635


namespace NUMINAMATH_CALUDE_pizza_sharing_ratio_l3776_377615

theorem pizza_sharing_ratio (total_slices : ℕ) (waiter_slices : ℕ) : 
  total_slices = 78 → 
  waiter_slices - 20 = 28 → 
  (total_slices - waiter_slices) / waiter_slices = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_sharing_ratio_l3776_377615


namespace NUMINAMATH_CALUDE_last_two_digits_product_l3776_377698

theorem last_two_digits_product (A B : ℕ) : 
  A < 10 → B < 10 → A + B = 12 → (10 * A + B) % 3 = 0 → A * B = 35 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l3776_377698


namespace NUMINAMATH_CALUDE_equation_solution_l3776_377657

theorem equation_solution : ∃ x : ℝ, 
  (216 + Real.sqrt 41472 - 18 * x - Real.sqrt (648 * x^2) = 0) ∧ 
  (x = (140 * Real.sqrt 2 - 140) / 9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3776_377657


namespace NUMINAMATH_CALUDE_total_cars_produced_l3776_377624

/-- The total number of cars produced in North America, Europe, and Asia is 9972. -/
theorem total_cars_produced (north_america europe asia : ℕ) 
  (h1 : north_america = 3884)
  (h2 : europe = 2871)
  (h3 : asia = 3217) : 
  north_america + europe + asia = 9972 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_produced_l3776_377624


namespace NUMINAMATH_CALUDE_b_77_mod_40_l3776_377655

def b (n : ℕ) : ℕ := 5^n + 9^n

theorem b_77_mod_40 : b 77 ≡ 14 [MOD 40] := by
  sorry

end NUMINAMATH_CALUDE_b_77_mod_40_l3776_377655


namespace NUMINAMATH_CALUDE_product_of_fractions_and_powers_of_two_l3776_377687

theorem product_of_fractions_and_powers_of_two : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * 
  (1 / 1024 : ℚ) * 2048 * (1 / 4096 : ℚ) * 8192 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_and_powers_of_two_l3776_377687


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3776_377648

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3776_377648


namespace NUMINAMATH_CALUDE_decimal_multiplication_l3776_377669

theorem decimal_multiplication : (0.7 : ℝ) * 0.8 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l3776_377669


namespace NUMINAMATH_CALUDE_gas_volume_ranking_l3776_377606

/-- Gas volume per capita for a region -/
structure GasVolume where
  region : String
  volume : Float

/-- Theorem: Russia has the highest gas volume per capita, followed by Non-West, then West -/
theorem gas_volume_ranking (west non_west russia : GasVolume) 
  (h_west : west.region = "West" ∧ west.volume = 21428)
  (h_non_west : non_west.region = "Non-West" ∧ non_west.volume = 26848.55)
  (h_russia : russia.region = "Russia" ∧ russia.volume = 302790.13) :
  russia.volume > non_west.volume ∧ non_west.volume > west.volume :=
by sorry

end NUMINAMATH_CALUDE_gas_volume_ranking_l3776_377606


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3776_377666

/-- The asymptotes of the hyperbola -/
def asymptote1 (x : ℝ) : ℝ := 3 * x + 6
def asymptote2 (x : ℝ) : ℝ := -3 * x + 4

/-- The point through which the hyperbola passes -/
def point : ℝ × ℝ := (1, 10)

/-- The standard form of the hyperbola equation -/
def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

/-- The theorem to be proved -/
theorem hyperbola_sum (h k a b : ℝ) :
  (∀ x, asymptote1 x = asymptote2 x → x = -1/3 ∧ asymptote1 x = 5) →
  hyperbola_equation point.1 point.2 h k a b →
  a > 0 ∧ b > 0 →
  a + h = 8/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3776_377666
