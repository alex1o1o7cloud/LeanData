import Mathlib

namespace simplify_expression_l3705_370565

theorem simplify_expression : (3 * 2 + 4 + 6) / 3 - 2 / 3 = 14 / 3 := by
  sorry

end simplify_expression_l3705_370565


namespace not_square_or_cube_of_2pow_minus_1_l3705_370552

theorem not_square_or_cube_of_2pow_minus_1 (n : ℕ) (h : n > 1) :
  ¬∃ (a : ℤ), (2^n - 1 : ℤ) = a^2 ∨ (2^n - 1 : ℤ) = a^3 := by
  sorry

end not_square_or_cube_of_2pow_minus_1_l3705_370552


namespace digit_sum_problem_l3705_370599

theorem digit_sum_problem (P Q : ℕ) : 
  P < 10 → Q < 10 → 77 * P + 77 * Q = 1000 * P + 100 * P + 10 * P + 7 → P + Q = 14 := by
  sorry

end digit_sum_problem_l3705_370599


namespace at_most_one_root_l3705_370538

theorem at_most_one_root (f : ℝ → ℝ) (h : ∀ a b, a < b → f a < f b) :
  ∃! x, f x = 0 :=
sorry

end at_most_one_root_l3705_370538


namespace bottle_lasts_eight_months_l3705_370501

/-- Represents the number of pills in a bottle -/
def bottle_pills : ℕ := 60

/-- Represents the fraction of a pill consumed daily -/
def daily_consumption : ℚ := 1/4

/-- Represents the number of days in a month (approximation) -/
def days_per_month : ℕ := 30

/-- Calculates the number of months a bottle will last -/
def bottle_duration : ℚ := (bottle_pills : ℚ) / daily_consumption / days_per_month

theorem bottle_lasts_eight_months :
  bottle_duration = 8 := by sorry

end bottle_lasts_eight_months_l3705_370501


namespace arithmetic_geometric_sequence_l3705_370567

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

def geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 := by
  sorry

end arithmetic_geometric_sequence_l3705_370567


namespace eight_people_arrangement_l3705_370515

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where 2 specific people are together -/
def arrangementsTwoTogether (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

/-- The number of ways to arrange n people in a row where 2 specific people are not together -/
def arrangementsNotTogether (n : ℕ) : ℕ :=
  totalArrangements n - arrangementsTwoTogether n

theorem eight_people_arrangement :
  arrangementsNotTogether 8 = 30240 := by
  sorry

end eight_people_arrangement_l3705_370515


namespace product_sum_theorem_l3705_370556

theorem product_sum_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 5)
  (eq3 : a + c + d = 20)
  (eq4 : b + c + d = 15) :
  a * b + c * d = 1002 / 9 := by
sorry

end product_sum_theorem_l3705_370556


namespace base_10_144_equals_base_12_100_l3705_370547

def base_10_to_12 (n : ℕ) : List ℕ := sorry

theorem base_10_144_equals_base_12_100 :
  base_10_to_12 144 = [1, 0, 0] :=
sorry

end base_10_144_equals_base_12_100_l3705_370547


namespace modulus_of_complex_expression_l3705_370581

theorem modulus_of_complex_expression :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end modulus_of_complex_expression_l3705_370581


namespace find_k_l3705_370507

theorem find_k : ∃ k : ℚ, (32 / k = 4) ∧ (k = 8) := by
  sorry

end find_k_l3705_370507


namespace volleyball_team_girls_l3705_370536

/-- Given a volleyball team with the following properties:
  * The total number of team members is 30
  * 20 members attended the last meeting
  * One-third of the girls and all boys attended the meeting
  Prove that the number of girls on the team is 15 -/
theorem volleyball_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 20 →
  boys + girls = total →
  boys + (1/3 : ℚ) * girls = attended →
  girls = 15 := by
sorry

end volleyball_team_girls_l3705_370536


namespace sqrt_inequality_l3705_370551

theorem sqrt_inequality (n : ℕ+) : Real.sqrt (n + 1) - Real.sqrt n < 1 / (2 * Real.sqrt n) := by
  sorry

end sqrt_inequality_l3705_370551


namespace quadratic_polynomial_integer_root_exists_l3705_370510

/-- Represents a quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluates a quadratic polynomial at x = -1 -/
def evalAtNegativeOne (p : QuadraticPolynomial) : ℤ :=
  p.a + -p.b + p.c

/-- Represents a single step change in the polynomial -/
inductive PolynomialStep
  | ChangeX : (δ : ℤ) → PolynomialStep
  | ChangeConstant : (δ : ℤ) → PolynomialStep

/-- Applies a step to a polynomial -/
def applyStep (p : QuadraticPolynomial) (step : PolynomialStep) : QuadraticPolynomial :=
  match step with
  | PolynomialStep.ChangeX δ => ⟨p.a, p.b + δ, p.c⟩
  | PolynomialStep.ChangeConstant δ => ⟨p.a, p.b, p.c + δ⟩

theorem quadratic_polynomial_integer_root_exists 
  (initial : QuadraticPolynomial)
  (final : QuadraticPolynomial)
  (h_initial : initial = ⟨1, 10, 20⟩)
  (h_final : final = ⟨1, 20, 10⟩)
  (steps : List PolynomialStep)
  (h_steps : ∀ step ∈ steps, 
    (∃ δ, step = PolynomialStep.ChangeX δ ∧ (δ = 1 ∨ δ = -1)) ∨
    (∃ δ, step = PolynomialStep.ChangeConstant δ ∧ (δ = 1 ∨ δ = -1)))
  (h_transform : final = steps.foldl applyStep initial) :
  ∃ p : QuadraticPolynomial, p ∈ initial :: (List.scanl applyStep initial steps) ∧ 
    ∃ x : ℤ, p.a * x^2 + p.b * x + p.c = 0 :=
  sorry

end quadratic_polynomial_integer_root_exists_l3705_370510


namespace problem_statement_l3705_370588

theorem problem_statement (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_eq : a^2 + b^2 = a + b) : 
  ((a + b)^2 ≤ 2*(a^2 + b^2)) ∧ ((a + 1)*(b + 1) ≤ 4) := by
  sorry

end problem_statement_l3705_370588


namespace cubic_function_symmetry_l3705_370517

/-- Given a function f(x) = ax³ + bx - 2 where f(2017) = 10, prove that f(-2017) = -14 -/
theorem cubic_function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x - 2
  f 2017 = 10 → f (-2017) = -14 := by
sorry

end cubic_function_symmetry_l3705_370517


namespace elmo_sandwich_jam_cost_l3705_370589

/-- The cost of blackberry jam used in Elmo's sandwiches -/
theorem elmo_sandwich_jam_cost :
  ∀ (N B J : ℕ),
    N > 1 →
    B > 0 →
    J > 0 →
    N * (6 * B + 7 * J) = 396 →
    (N * J * 7 : ℚ) / 100 = 378 / 100 := by
  sorry

end elmo_sandwich_jam_cost_l3705_370589


namespace prime_squared_minus_five_not_divisible_by_eight_l3705_370597

theorem prime_squared_minus_five_not_divisible_by_eight (p : ℕ) 
  (h_prime : Nat.Prime p) (h_gt_two : p > 2) : ¬ (8 ∣ (p^2 - 5)) := by
  sorry

end prime_squared_minus_five_not_divisible_by_eight_l3705_370597


namespace simplify_expression_l3705_370511

theorem simplify_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) :
  (1 - 2 / (x - 1)) * ((x^2 - x) / (x^2 - 6*x + 9)) = x / (x - 3) := by
  sorry

end simplify_expression_l3705_370511


namespace trajectory_of_midpoint_l3705_370568

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- Point P is on the hyperbola -/
def P_on_hyperbola (px py : ℝ) : Prop := hyperbola px py

/-- M is the midpoint of OP -/
def M_is_midpoint (mx my px py : ℝ) : Prop := mx = px / 2 ∧ my = py / 2

/-- The trajectory equation for point M -/
def trajectory (x y : ℝ) : Prop := x^2 - 4*y^2 = 1

theorem trajectory_of_midpoint (mx my px py : ℝ) :
  P_on_hyperbola px py → M_is_midpoint mx my px py → trajectory mx my :=
by sorry

end trajectory_of_midpoint_l3705_370568


namespace train_speed_proof_l3705_370525

/-- Proves that the speed of a train is 23.4 km/hr given specific conditions -/
theorem train_speed_proof (train_length : Real) (crossing_time : Real) (total_length : Real) :
  train_length = 180 →
  crossing_time = 30 →
  total_length = 195 →
  (total_length / crossing_time) * 3.6 = 23.4 :=
by
  sorry

#check train_speed_proof

end train_speed_proof_l3705_370525


namespace bread_inventory_l3705_370572

def initial_loaves : ℕ := 2355
def sold_loaves : ℕ := 629
def delivered_loaves : ℕ := 489

theorem bread_inventory : 
  initial_loaves - sold_loaves + delivered_loaves = 2215 := by
  sorry

end bread_inventory_l3705_370572


namespace sin_alpha_minus_pi_third_l3705_370593

theorem sin_alpha_minus_pi_third (α : Real) 
  (h1 : -π/2 < α) (h2 : α < 0) (h3 : 2 * Real.tan α * Real.sin α = 3) : 
  Real.sin (α - π/3) = -Real.sqrt 3 / 2 := by
  sorry

end sin_alpha_minus_pi_third_l3705_370593


namespace smallest_valid_number_l3705_370524

def is_valid (n : ℕ+) : Prop :=
  (Finset.card (Nat.divisors n) = 144) ∧
  (∃ k : ℕ, ∀ i : Fin 10, (k + i) ∈ Nat.divisors n)

theorem smallest_valid_number : 
  (is_valid 110880) ∧ (∀ m : ℕ+, m < 110880 → ¬(is_valid m)) :=
sorry

end smallest_valid_number_l3705_370524


namespace spade_calculation_l3705_370548

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_calculation : spade (spade 3 5) (spade 6 9) = 1 := by
  sorry

end spade_calculation_l3705_370548


namespace unique_quadratic_function_l3705_370591

/-- A quadratic function satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, 
    (∀ x, f x = a * x^2 + b * x + c) ∧ 
    (f (-1) = 0) ∧ 
    (∀ x, x ≤ f x) ∧
    (∀ x, f x ≤ (1 + x^2) / 2)

/-- The unique quadratic function satisfying the given conditions -/
theorem unique_quadratic_function (f : ℝ → ℝ) (hf : QuadraticFunction f) : 
  ∀ x, f x = (1/4) * x^2 + (1/2) * x + 1/4 :=
sorry

end unique_quadratic_function_l3705_370591


namespace prime_square_mod_240_l3705_370534

theorem prime_square_mod_240 (p : Nat) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  ∃ (r₁ r₂ : Nat), r₁ ≠ r₂ ∧ r₁ < 240 ∧ r₂ < 240 ∧
  ∀ (q : Nat), Nat.Prime q → q > 5 → (q^2 % 240 = r₁ ∨ q^2 % 240 = r₂) :=
sorry

end prime_square_mod_240_l3705_370534


namespace range_of_a_l3705_370560

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- Define the property that ¬p is sufficient but not necessary for ¬q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x a)) ∧ (∃ x, ¬(q x a) ∧ p x)

-- Theorem statement
theorem range_of_a :
  (∀ a : ℝ, sufficient_not_necessary a) → (∀ a : ℝ, a ≥ 1) :=
sorry

end range_of_a_l3705_370560


namespace extreme_value_negative_a_one_zero_positive_a_l3705_370590

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 - (a + 1) * x

-- Theorem for the case when a < 0
theorem extreme_value_negative_a (a : ℝ) (ha : a < 0) :
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧
  (∀ x : ℝ, f a x ≥ -a - 1/2) ∧
  (¬∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) :=
sorry

-- Theorem for the case when a > 0
theorem one_zero_positive_a (a : ℝ) (ha : a > 0) :
  ∃! x : ℝ, f a x = 0 :=
sorry

end

end extreme_value_negative_a_one_zero_positive_a_l3705_370590


namespace unique_solution_exponential_equation_l3705_370522

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x+2) * (4 : ℝ)^(2*x+3) = (8 : ℝ)^(3*x+4) :=
by
  sorry

end unique_solution_exponential_equation_l3705_370522


namespace sqrt_product_property_sqrt_40_in_terms_of_a_b_l3705_370563

theorem sqrt_product_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  Real.sqrt x * Real.sqrt y = Real.sqrt (x * y) := by sorry

theorem sqrt_40_in_terms_of_a_b (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 10) :
  Real.sqrt 40 = Real.sqrt 2 * a * b := by sorry

end sqrt_product_property_sqrt_40_in_terms_of_a_b_l3705_370563


namespace expression_value_l3705_370584

theorem expression_value (m n x y : ℤ) 
  (h1 : m - n = 100) 
  (h2 : x + y = -1) : 
  (n + x) - (m - y) = -101 := by
sorry

end expression_value_l3705_370584


namespace intersection_empty_iff_m_range_union_equals_B_iff_m_range_l3705_370532

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}
def B : Set ℝ := {x | x < -6 ∨ x > 1}

-- Theorem for part (I)
theorem intersection_empty_iff_m_range (m : ℝ) :
  A m ∩ B = ∅ ↔ -6 ≤ m ∧ m ≤ 0 :=
sorry

-- Theorem for part (II)
theorem union_equals_B_iff_m_range (m : ℝ) :
  A m ∪ B = B ↔ m < -7 ∨ m > 1 :=
sorry

end intersection_empty_iff_m_range_union_equals_B_iff_m_range_l3705_370532


namespace percentage_increase_l3705_370592

theorem percentage_increase (original : ℝ) (new : ℝ) (percentage : ℝ) : 
  original = 80 →
  new = 88.8 →
  percentage = 11 →
  (new - original) / original * 100 = percentage :=
by sorry

end percentage_increase_l3705_370592


namespace cone_vertex_angle_l3705_370546

theorem cone_vertex_angle (l r : ℝ) (h : l > 0) (h2 : r > 0) : 
  (2 * π * l / 3 = 2 * π * r) → 
  (2 * Real.arcsin (1 / 3) : ℝ) = 2 * Real.arcsin (r / l) := by
sorry

end cone_vertex_angle_l3705_370546


namespace book_cost_problem_l3705_370545

theorem book_cost_problem (cost_of_three : ℝ) (h : cost_of_three = 45) :
  let cost_of_one : ℝ := cost_of_three / 3
  let cost_of_seven : ℝ := 7 * cost_of_one
  cost_of_seven = 105 := by
  sorry

end book_cost_problem_l3705_370545


namespace soldiers_on_first_side_l3705_370503

theorem soldiers_on_first_side (food_per_soldier_first : ℕ)
                               (food_difference : ℕ)
                               (soldier_difference : ℕ)
                               (total_food : ℕ) :
  food_per_soldier_first = 10 →
  food_difference = 2 →
  soldier_difference = 500 →
  total_food = 68000 →
  ∃ (x : ℕ), 
    x * food_per_soldier_first + 
    (x - soldier_difference) * (food_per_soldier_first - food_difference) = total_food ∧
    x = 4000 := by
  sorry

end soldiers_on_first_side_l3705_370503


namespace gold_award_winners_possibly_all_freshmen_l3705_370596

theorem gold_award_winners_possibly_all_freshmen 
  (total_winners : ℕ) 
  (selected_students : ℕ) 
  (selected_freshmen : ℕ) 
  (selected_gold : ℕ) 
  (h1 : total_winners = 120)
  (h2 : selected_students = 24)
  (h3 : selected_freshmen = 6)
  (h4 : selected_gold = 4) :
  ∃ (total_freshmen : ℕ) (total_gold : ℕ),
    total_freshmen ≤ total_winners ∧
    total_gold ≤ total_winners ∧
    total_gold ≤ total_freshmen :=
by sorry

end gold_award_winners_possibly_all_freshmen_l3705_370596


namespace largest_common_divisor_of_consecutive_odd_product_l3705_370550

theorem largest_common_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∀ m : ℕ, m > 315 → ∃ k : ℕ, k > 0 ∧ Even k ∧ 
    ¬(m ∣ (k+1)*(k+3)*(k+5)*(k+7)*(k+9)*(k+11)*(k+13))) ∧
  (315 ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end largest_common_divisor_of_consecutive_odd_product_l3705_370550


namespace one_third_of_seven_times_nine_l3705_370598

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l3705_370598


namespace plant_branches_theorem_l3705_370573

theorem plant_branches_theorem : ∃ (x : ℕ), x > 0 ∧ 1 + x + x^2 = 57 ∧ x = 7 := by
  sorry

end plant_branches_theorem_l3705_370573


namespace trajectory_and_max_area_l3705_370505

noncomputable section

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := p.1^2 / 2 + p.2^2 = 1

-- Define the relation between P and M
def P_relation (P M : ℝ × ℝ) : Prop := P.1 = 2 * M.1 ∧ P.2 = 2 * M.2

-- Define the trajectory C
def on_trajectory (p : ℝ × ℝ) : Prop := p.1^2 / 8 + p.2^2 / 4 = 1

-- Define the line l
def on_line (p : ℝ × ℝ) (m : ℝ) : Prop := p.2 = p.1 + m

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2)) / 2

theorem trajectory_and_max_area 
  (M : ℝ × ℝ) (P : ℝ × ℝ) (m : ℝ) (A B : ℝ × ℝ) :
  on_ellipse M → 
  P_relation P M → 
  m ≠ 0 →
  on_line A m →
  on_line B m →
  on_trajectory A →
  on_trajectory B →
  A ≠ B →
  (∀ P, P_relation P M → on_trajectory P) ∧
  (∀ X Y, on_trajectory X → on_trajectory Y → on_line X m → on_line Y m → 
    triangle_area O X Y ≤ 2 * Real.sqrt 2) :=
sorry

end trajectory_and_max_area_l3705_370505


namespace arithmetic_sequence_problem_l3705_370585

/-- An arithmetic sequence is monotonically increasing if its common difference is positive -/
def IsMonoIncreasingArithmeticSeq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h_mono : IsMonoIncreasingArithmeticSeq a)
  (h_a3 : a 3 = 1)
  (h_a2a4 : a 2 * a 4 = 3/4) :
  a 1 = 0 := by
  sorry

end arithmetic_sequence_problem_l3705_370585


namespace max_value_expression_l3705_370583

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  4*x*y*Real.sqrt 2 + 5*y*z + 3*x*z*Real.sqrt 3 ≤ (44*Real.sqrt 2 + 110 + 9*Real.sqrt 3) / 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ z₀ ≥ 0 ∧ x₀^2 + y₀^2 + z₀^2 = 1 ∧
    4*x₀*y₀*Real.sqrt 2 + 5*y₀*z₀ + 3*x₀*z₀*Real.sqrt 3 = (44*Real.sqrt 2 + 110 + 9*Real.sqrt 3) / 3 :=
by sorry

end max_value_expression_l3705_370583


namespace arithmetic_sequence_sum_l3705_370579

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 3 = 1) →
  (a 10 + a 11 = 9) →
  (a 5 + a 6 = 4) :=
by
  sorry

end arithmetic_sequence_sum_l3705_370579


namespace fathers_age_l3705_370504

/-- Given information about Sebastian, his sister, and their father's ages, prove the father's current age. -/
theorem fathers_age (sebastian_age : ℕ) (age_difference : ℕ) (years_ago : ℕ) (fraction : ℚ) : 
  sebastian_age = 40 →
  age_difference = 10 →
  years_ago = 5 →
  fraction = 3/4 →
  (sebastian_age - years_ago + (sebastian_age - age_difference - years_ago) : ℚ) = 
    fraction * (sebastian_age - years_ago + (sebastian_age - age_difference - years_ago) + years_ago) →
  sebastian_age - years_ago + (sebastian_age - age_difference - years_ago) + years_ago = 85 := by
sorry

end fathers_age_l3705_370504


namespace album_pages_count_l3705_370575

theorem album_pages_count : ∃ (x : ℕ) (y : ℕ), 
  x > 0 ∧ 
  y > 0 ∧ 
  20 * x < y ∧ 
  23 * x > y ∧ 
  21 * x + y = 500 ∧ 
  x = 12 := by
  sorry

end album_pages_count_l3705_370575


namespace theater_ticket_sales_l3705_370553

theorem theater_ticket_sales (total_tickets : ℕ) (adult_price senior_price : ℚ) (total_receipts : ℚ) :
  total_tickets = 510 →
  adult_price = 21 →
  senior_price = 15 →
  total_receipts = 8748 →
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end theater_ticket_sales_l3705_370553


namespace number_solution_l3705_370530

theorem number_solution : ∃ x : ℝ, (45 - 3 * x = 18) ∧ (x = 9) := by sorry

end number_solution_l3705_370530


namespace complex_power_magnitude_l3705_370523

theorem complex_power_magnitude : Complex.abs ((4/5 : ℂ) + (3/5 : ℂ) * Complex.I) ^ 8 = 1 := by
  sorry

end complex_power_magnitude_l3705_370523


namespace leap_year_53_sundays_probability_l3705_370513

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The number of possible combinations for the two extra days in a leap year -/
def extra_day_combinations : ℕ := 7

/-- The number of combinations that result in 53 Sundays -/
def favorable_combinations : ℕ := 2

/-- The probability of a leap year having 53 Sundays -/
def prob_53_sundays : ℚ := favorable_combinations / extra_day_combinations

theorem leap_year_53_sundays_probability :
  prob_53_sundays = 2 / 7 := by sorry

end leap_year_53_sundays_probability_l3705_370513


namespace cubic_equation_root_l3705_370544

theorem cubic_equation_root (a b : ℚ) : 
  (2 + Real.sqrt 3 : ℝ) ^ 3 + a * (2 + Real.sqrt 3 : ℝ) ^ 2 + b * (2 + Real.sqrt 3 : ℝ) - 15 = 0 → 
  b = -44 := by
sorry

end cubic_equation_root_l3705_370544


namespace second_quadrant_trig_simplification_l3705_370578

theorem second_quadrant_trig_simplification (α : Real) 
  (h : π/2 < α ∧ α < π) : 
  (Real.sqrt (1 + 2 * Real.sin (5 * π - α) * Real.cos (α - π))) / 
  (Real.sin (α - 3 * π / 2) - Real.sqrt (1 - Real.sin (3 * π / 2 + α)^2)) = -1 :=
by sorry

end second_quadrant_trig_simplification_l3705_370578


namespace meal_center_allocation_l3705_370580

/-- Represents the meal center's soup can allocation problem -/
theorem meal_center_allocation (total_cans : ℕ) (adults_per_can children_per_can : ℕ) 
  (children_to_feed : ℕ) (adults_fed : ℕ) :
  total_cans = 10 →
  adults_per_can = 4 →
  children_per_can = 7 →
  children_to_feed = 21 →
  adults_fed = (total_cans - (children_to_feed / children_per_can)) * adults_per_can →
  adults_fed = 28 := by
sorry

end meal_center_allocation_l3705_370580


namespace f_increasing_on_positive_reals_l3705_370520

/-- The function f(x) = x^2 / (x^2 + 1) is increasing on the interval (0, +∞) -/
theorem f_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → 
    (x₁^2 / (x₁^2 + 1)) < (x₂^2 / (x₂^2 + 1)) := by
  sorry

end f_increasing_on_positive_reals_l3705_370520


namespace reciprocal_of_x_l3705_370541

theorem reciprocal_of_x (x : ℝ) (h1 : x^3 - 2*x^2 = 0) (h2 : x ≠ 0) : 1/x = 1/2 := by
  sorry

end reciprocal_of_x_l3705_370541


namespace triangle_area_72_l3705_370555

theorem triangle_area_72 (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * (2*x) * x = 72) : x = 6 * Real.sqrt 2 := by
  sorry

end triangle_area_72_l3705_370555


namespace apple_rate_is_70_l3705_370569

-- Define the given quantities
def apple_quantity : ℕ := 8
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 45
def total_paid : ℕ := 965

-- Define the unknown apple rate
def apple_rate : ℕ := sorry

-- Theorem statement
theorem apple_rate_is_70 :
  apple_quantity * apple_rate + mango_quantity * mango_rate = total_paid →
  apple_rate = 70 := by
  sorry

end apple_rate_is_70_l3705_370569


namespace tony_remaining_money_l3705_370542

def initial_amount : ℕ := 20
def ticket_cost : ℕ := 8
def hotdog_cost : ℕ := 3

theorem tony_remaining_money :
  initial_amount - ticket_cost - hotdog_cost = 9 :=
by sorry

end tony_remaining_money_l3705_370542


namespace at_least_one_real_root_l3705_370512

theorem at_least_one_real_root (a b c : ℝ) : 
  (a - b)^2 - 4*(b - c) ≥ 0 ∨ 
  (b - c)^2 - 4*(c - a) ≥ 0 ∨ 
  (c - a)^2 - 4*(a - b) ≥ 0 := by
sorry

end at_least_one_real_root_l3705_370512


namespace extreme_value_implies_zero_derivative_converse_not_always_true_l3705_370561

-- Define a function that has an extreme value at a point
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x ≤ f x₀ ∨ f x ≥ f x₀

-- Theorem statement
theorem extreme_value_implies_zero_derivative
  (f : ℝ → ℝ) (x₀ : ℝ) (hf : Differentiable ℝ f) :
  has_extreme_value f x₀ → deriv f x₀ = 0 :=
sorry

-- Counter-example to show the converse is not always true
theorem converse_not_always_true :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ deriv f 0 = 0 ∧ ¬(has_extreme_value f 0) :=
sorry

end extreme_value_implies_zero_derivative_converse_not_always_true_l3705_370561


namespace sqrt_12_plus_sqrt_27_l3705_370594

theorem sqrt_12_plus_sqrt_27 : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end sqrt_12_plus_sqrt_27_l3705_370594


namespace q_age_is_40_l3705_370516

/-- Represents the ages of two people p and q --/
structure Ages where
  p : ℕ
  q : ℕ

/-- The condition stated by p --/
def age_condition (ages : Ages) : Prop :=
  ages.p = 3 * (ages.q - (ages.p - ages.q))

/-- The sum of their present ages is 100 --/
def age_sum (ages : Ages) : Prop :=
  ages.p + ages.q = 100

/-- Theorem stating that given the conditions, q's present age is 40 --/
theorem q_age_is_40 (ages : Ages) 
  (h1 : age_condition ages) 
  (h2 : age_sum ages) : 
  ages.q = 40 := by
  sorry

end q_age_is_40_l3705_370516


namespace wedge_volume_l3705_370570

/-- The volume of a wedge that represents one-third of a cylindrical cheese log -/
theorem wedge_volume (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cylinder_volume := π * r^2 * h
  let wedge_volume := (1/3) * cylinder_volume
  h = 8 ∧ r = 5 → wedge_volume = (200 * π) / 3 := by
  sorry

end wedge_volume_l3705_370570


namespace investment_problem_l3705_370564

/-- Proves that given the conditions of the investment problem, b's investment amount is 1000. -/
theorem investment_problem (a b c total_profit c_share : ℚ) : 
  a = 800 →
  c = 1200 →
  total_profit = 1000 →
  c_share = 400 →
  c_share / total_profit = c / (a + b + c) →
  b = 1000 := by
  sorry

end investment_problem_l3705_370564


namespace prime_product_theorem_l3705_370500

theorem prime_product_theorem (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ →
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ →
  2*p₁ + 3*p₂ + 5*p₃ + 7*p₄ = 162 →
  11*p₁ + 7*p₂ + 5*p₃ + 4*p₄ = 162 →
  p₁ * p₂ * p₃ * p₄ = 570 :=
by sorry

end prime_product_theorem_l3705_370500


namespace abs_eq_piecewise_l3705_370509

theorem abs_eq_piecewise (x : ℝ) : |x| = if x ≥ 0 then x else -x := by sorry

end abs_eq_piecewise_l3705_370509


namespace symmetric_points_y_axis_l3705_370533

/-- Given two points in R² that are symmetric about the y-axis, 
    prove that their x-coordinates are negatives of each other 
    and their y-coordinates are the same. -/
theorem symmetric_points_y_axis 
  (A B : ℝ × ℝ) 
  (h_symmetric : A.1 = -B.1 ∧ A.2 = B.2) 
  (h_A : A = (1, -2)) : 
  B = (-1, -2) := by
sorry

end symmetric_points_y_axis_l3705_370533


namespace sufficient_not_necessary_l3705_370519

/-- The inequality x^2 - 2ax + a > 0 has ℝ as its solution set -/
def has_real_solution_set (a : ℝ) : Prop :=
  ∀ x, x^2 - 2*a*x + a > 0

/-- 0 < a < 1 -/
def a_in_open_unit_interval (a : ℝ) : Prop :=
  0 < a ∧ a < 1

theorem sufficient_not_necessary :
  (∀ a : ℝ, has_real_solution_set a → a_in_open_unit_interval a) ∧
  (∃ a : ℝ, a_in_open_unit_interval a ∧ ¬has_real_solution_set a) :=
sorry

end sufficient_not_necessary_l3705_370519


namespace fifteenth_term_of_sequence_l3705_370531

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 13) (h₃ : a₃ = 23) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 143 := by
  sorry

end fifteenth_term_of_sequence_l3705_370531


namespace sphere_surface_volume_relation_l3705_370554

theorem sphere_surface_volume_relation :
  ∀ (r r' : ℝ) (A A' V V' : ℝ),
  (A = 4 * Real.pi * r^2) →
  (A' = 4 * A) →
  (V = (4/3) * Real.pi * r^3) →
  (V' = (4/3) * Real.pi * r'^3) →
  (A' = 4 * Real.pi * r'^2) →
  (V' = 8 * V) :=
by sorry

end sphere_surface_volume_relation_l3705_370554


namespace mod_eleven_fifth_power_l3705_370518

theorem mod_eleven_fifth_power (n : ℕ) : 
  11^5 ≡ n [ZMOD 9] → 0 ≤ n → n < 9 → n = 5 := by
  sorry

end mod_eleven_fifth_power_l3705_370518


namespace f_derivative_at_zero_l3705_370582

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.exp (x^2) - Real.cos x) / x else 0

theorem f_derivative_at_zero : 
  deriv f 0 = (3/2) := by sorry

end f_derivative_at_zero_l3705_370582


namespace bicycle_oil_requirement_l3705_370502

/-- The amount of oil needed to fix a bicycle -/
theorem bicycle_oil_requirement (wheel_count : ℕ) (oil_per_wheel : ℕ) (oil_for_rest : ℕ) : 
  wheel_count = 2 → oil_per_wheel = 10 → oil_for_rest = 5 →
  wheel_count * oil_per_wheel + oil_for_rest = 25 := by
  sorry

#check bicycle_oil_requirement

end bicycle_oil_requirement_l3705_370502


namespace last_day_is_monday_l3705_370521

/-- 
Given a year with 365 days, if the 15th day falls on a Monday,
then the 365th day also falls on a Monday.
-/
theorem last_day_is_monday (year : ℕ) : 
  year % 7 = 1 → -- Assuming Monday is represented by 1
  (365 % 7 = year % 7) → -- The last day falls on the same day as the first
  (15 % 7 = 1) → -- The 15th day is a Monday
  (365 % 7 = 1) -- The 365th day is also a Monday
:= by sorry

end last_day_is_monday_l3705_370521


namespace complement_of_A_in_U_l3705_370566

def U : Set Int := {x | (x + 1) * (x - 3) ≤ 0}

def A : Set Int := {0, 1, 2}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {-1, 3} := by
  sorry

end complement_of_A_in_U_l3705_370566


namespace baseball_gear_cost_l3705_370506

/-- Calculates the total cost of baseball gear including tax -/
def total_cost (birthday_money : ℚ) (glove_price : ℚ) (glove_discount : ℚ) 
  (baseball_price : ℚ) (bat_price : ℚ) (bat_discount : ℚ) (cleats_price : ℚ) 
  (cap_price : ℚ) (tax_rate : ℚ) : ℚ :=
  let discounted_glove := glove_price * (1 - glove_discount)
  let discounted_bat := bat_price * (1 - bat_discount)
  let subtotal := discounted_glove + baseball_price + discounted_bat + cleats_price + cap_price
  let total := subtotal * (1 + tax_rate)
  total

/-- Theorem stating the total cost of baseball gear -/
theorem baseball_gear_cost : 
  total_cost 120 35 0.2 15 50 0.1 30 10 0.07 = 136.96 := by
  sorry


end baseball_gear_cost_l3705_370506


namespace fraction_simplification_l3705_370528

theorem fraction_simplification :
  (1 / 2 + 1 / 3) / (3 / 7 - 1 / 5) = 175 / 48 := by
  sorry

end fraction_simplification_l3705_370528


namespace total_quantities_l3705_370535

theorem total_quantities (average : ℝ) (average_three : ℝ) (average_two : ℝ) : 
  average = 11 → average_three = 4 → average_two = 21.5 → 
  ∃ (n : ℕ), n = 5 ∧ 
    (n : ℝ) * average = 3 * average_three + 2 * average_two := by
  sorry

end total_quantities_l3705_370535


namespace set_operations_and_subset_l3705_370562

def A : Set ℝ := {x | (x - 3) / (x - 7) < 0}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

theorem set_operations_and_subset :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) ∧
  (∀ a : ℝ, C a ⊆ (A ∪ B) ↔ a ≤ 3) :=
sorry

end set_operations_and_subset_l3705_370562


namespace solve_equation_l3705_370574

theorem solve_equation (x : ℝ) (h : x ≠ 0) :
  (2 / x + (3 / x) / (6 / x) = 1.25) → x = 8 / 3 := by
  sorry

end solve_equation_l3705_370574


namespace cube_sum_of_roots_l3705_370571

theorem cube_sum_of_roots (r s t : ℝ) : 
  (r - (20 : ℝ)^(1/3)) * (r - (60 : ℝ)^(1/3)) * (r - (120 : ℝ)^(1/3)) = 1 →
  (s - (20 : ℝ)^(1/3)) * (s - (60 : ℝ)^(1/3)) * (s - (120 : ℝ)^(1/3)) = 1 →
  (t - (20 : ℝ)^(1/3)) * (t - (60 : ℝ)^(1/3)) * (t - (120 : ℝ)^(1/3)) = 1 →
  r ≠ s → r ≠ t → s ≠ t →
  r^3 + s^3 + t^3 = 203 := by
  sorry

end cube_sum_of_roots_l3705_370571


namespace oyster_consumption_l3705_370576

/-- The number of oysters Squido eats -/
def squido_oysters : ℕ := 200

/-- The number of oysters Crabby eats -/
def crabby_oysters : ℕ := 2 * squido_oysters

/-- The total number of oysters eaten by Crabby and Squido -/
def total_oysters : ℕ := squido_oysters + crabby_oysters

theorem oyster_consumption :
  total_oysters = 600 :=
by sorry

end oyster_consumption_l3705_370576


namespace average_of_numbers_l3705_370543

def numbers : List ℝ := [13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : ℝ) = 125830.8 := by
  sorry

end average_of_numbers_l3705_370543


namespace arithmetic_sequence_sum_l3705_370549

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The arithmetic sequence -/
def a : ℕ → ℝ := sorry

theorem arithmetic_sequence_sum :
  (S 7 = 28) → (S 11 = 66) → (S 9 = 45) := by
  sorry

end arithmetic_sequence_sum_l3705_370549


namespace minimum_nickels_needed_l3705_370558

/-- The cost of the sneakers in dollars -/
def sneaker_cost : ℚ := 45.5

/-- The number of $10 bills Chloe has -/
def ten_dollar_bills : ℕ := 4

/-- The number of quarters Chloe has -/
def quarters : ℕ := 5

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The minimum number of nickels needed -/
def min_nickels : ℕ := 85

theorem minimum_nickels_needed :
  ∀ n : ℕ,
  (n : ℚ) * nickel_value + (ten_dollar_bills * 10 : ℚ) + (quarters * 0.25 : ℚ) ≥ sneaker_cost →
  n ≥ min_nickels :=
by sorry

end minimum_nickels_needed_l3705_370558


namespace milk_water_ratio_in_first_vessel_l3705_370577

-- Define the volumes of the vessels
def vessel1_volume : ℚ := 3
def vessel2_volume : ℚ := 5

-- Define the milk to water ratio in the second vessel
def vessel2_milk_ratio : ℚ := 6
def vessel2_water_ratio : ℚ := 4

-- Define the mixed ratio
def mixed_ratio : ℚ := 1

-- Define the unknown ratio for the first vessel
def vessel1_milk_ratio : ℚ := 1
def vessel1_water_ratio : ℚ := 2

theorem milk_water_ratio_in_first_vessel :
  (vessel1_milk_ratio / vessel1_water_ratio = 1 / 2) ∧
  (vessel1_milk_ratio * vessel1_volume + vessel2_milk_ratio * vessel2_volume) /
  (vessel1_water_ratio * vessel1_volume + vessel2_water_ratio * vessel2_volume) = mixed_ratio :=
by sorry

end milk_water_ratio_in_first_vessel_l3705_370577


namespace smallest_c_value_l3705_370537

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_c_value (a b c d e : ℕ) :
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧
  is_perfect_square (b + c + d) ∧
  is_perfect_cube (a + b + c + d + e) →
  c ≥ 675 ∧ ∃ (a' b' c' d' e' : ℕ),
    a' + 1 = b' ∧ b' + 1 = c' ∧ c' + 1 = d' ∧ d' + 1 = e' ∧
    is_perfect_square (b' + c' + d') ∧
    is_perfect_cube (a' + b' + c' + d' + e') ∧
    c' = 675 :=
by sorry

end smallest_c_value_l3705_370537


namespace cost_of_bananas_l3705_370508

/-- The cost of bananas given the following conditions:
  * The cost of one banana is 800 won
  * The cost of one kiwi is 400 won
  * The total number of bananas and kiwis is 18
  * The total amount spent is 10,000 won
-/
theorem cost_of_bananas :
  let banana_cost : ℕ := 800
  let kiwi_cost : ℕ := 400
  let total_fruits : ℕ := 18
  let total_spent : ℕ := 10000
  ∃ (num_bananas : ℕ),
    num_bananas * banana_cost + (total_fruits - num_bananas) * kiwi_cost = total_spent ∧
    num_bananas * banana_cost = 5600 :=
by sorry

end cost_of_bananas_l3705_370508


namespace max_abs_z_l3705_370586

theorem max_abs_z (z : ℂ) (θ : ℝ) (h : z - 1 = Complex.cos θ + Complex.I * Complex.sin θ) :
  Complex.abs z ≤ 2 ∧ ∃ θ₀ : ℝ, Complex.abs (1 + Complex.cos θ₀ + Complex.I * Complex.sin θ₀) = 2 := by
  sorry

end max_abs_z_l3705_370586


namespace common_divisors_count_l3705_370526

/-- The number of positive divisors that 9240, 7920, and 8800 have in common -/
theorem common_divisors_count : Nat.card {d : ℕ | d > 0 ∧ d ∣ 9240 ∧ d ∣ 7920 ∧ d ∣ 8800} = 32 := by
  sorry

end common_divisors_count_l3705_370526


namespace school_trip_classrooms_l3705_370527

theorem school_trip_classrooms 
  (students_per_classroom : ℕ) 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (h1 : students_per_classroom = 66)
  (h2 : seats_per_bus = 6)
  (h3 : buses_needed = 737) :
  (buses_needed * seats_per_bus) / students_per_classroom = 67 := by
  sorry

end school_trip_classrooms_l3705_370527


namespace monkey_percentage_after_eating_l3705_370587

/-- The percentage of monkeys among animals after two monkeys each eat one bird -/
theorem monkey_percentage_after_eating (initial_monkeys initial_birds : ℕ) 
  (h1 : initial_monkeys = 6)
  (h2 : initial_birds = 6)
  (h3 : initial_monkeys > 0)
  (h4 : initial_birds ≥ 2) : 
  (initial_monkeys : ℚ) / (initial_monkeys + initial_birds - 2 : ℚ) = 3/5 := by
  sorry

#check monkey_percentage_after_eating

end monkey_percentage_after_eating_l3705_370587


namespace speed_equivalence_l3705_370540

/-- Proves that a speed of 0.8 km/h is equivalent to 8/36 m/s -/
theorem speed_equivalence : ∃ (speed : ℚ), 
  (speed = 8 / 36) ∧ 
  (speed * 3600 / 1000 = 0.8) := by
  sorry

end speed_equivalence_l3705_370540


namespace books_left_l3705_370557

def initial_books : ℕ := 75
def borrowed_books : ℕ := 18

theorem books_left : initial_books - borrowed_books = 57 := by
  sorry

end books_left_l3705_370557


namespace division_problem_l3705_370514

theorem division_problem (A : ℕ) : A = 1 → 23 = 13 * A + 10 := by
  sorry

end division_problem_l3705_370514


namespace x_cube_x_square_order_l3705_370539

theorem x_cube_x_square_order (x : ℝ) (h : -1 < x ∧ x < 0) : x < x^3 ∧ x^3 < x^2 := by
  sorry

end x_cube_x_square_order_l3705_370539


namespace derivative_at_alpha_l3705_370529

open Real

theorem derivative_at_alpha (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2 * cos α - sin x
  HasDerivAt f (-cos α) α := by
  sorry

end derivative_at_alpha_l3705_370529


namespace square_area_from_vertices_l3705_370595

/-- The area of a square with vertices P(2, 3), Q(-3, 4), R(-2, -1), and S(3, 0) is 26 square units -/
theorem square_area_from_vertices : 
  let P : ℝ × ℝ := (2, 3)
  let Q : ℝ × ℝ := (-3, 4)
  let R : ℝ × ℝ := (-2, -1)
  let S : ℝ × ℝ := (3, 0)
  let square_area := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^2
  square_area = 26 := by
  sorry

end square_area_from_vertices_l3705_370595


namespace constant_integral_equals_one_l3705_370559

theorem constant_integral_equals_one : ∫ x in (0:ℝ)..1, (1:ℝ) = 1 := by sorry

end constant_integral_equals_one_l3705_370559
