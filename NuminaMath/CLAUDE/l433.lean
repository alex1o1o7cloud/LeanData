import Mathlib

namespace gallon_paint_cost_l433_43324

def pints_needed : ℕ := 8
def pint_cost : ℚ := 8
def gallon_equivalent_pints : ℕ := 8
def savings : ℚ := 9

def total_pint_cost : ℚ := pints_needed * pint_cost

theorem gallon_paint_cost : 
  total_pint_cost - savings = 55 := by sorry

end gallon_paint_cost_l433_43324


namespace abs_minus_self_nonnegative_l433_43300

theorem abs_minus_self_nonnegative (a : ℚ) : |a| - a ≥ 0 := by
  sorry

end abs_minus_self_nonnegative_l433_43300


namespace power_five_remainder_l433_43352

theorem power_five_remainder (n : ℕ) : (5^1234 : ℕ) % 100 = 25 := by
  sorry

end power_five_remainder_l433_43352


namespace angle_420_equals_60_l433_43315

/-- The angle (in degrees) that represents a full rotation in a standard coordinate system -/
def full_rotation : ℝ := 360

/-- Two angles have the same terminal side if their difference is a multiple of a full rotation -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α - β = k * full_rotation

/-- Theorem: The angle 420° has the same terminal side as 60° -/
theorem angle_420_equals_60 : same_terminal_side 420 60 := by
  sorry

end angle_420_equals_60_l433_43315


namespace common_solution_z_values_l433_43332

theorem common_solution_z_values : 
  ∃ (z₁ z₂ : ℝ), 
    (∀ x : ℝ, x^2 + z₁^2 - 9 = 0 ∧ x^2 - 4*z₁ + 5 = 0) ∧
    (∀ x : ℝ, x^2 + z₂^2 - 9 = 0 ∧ x^2 - 4*z₂ + 5 = 0) ∧
    z₁ = -2 + 3 * Real.sqrt 2 ∧
    z₂ = -2 - 3 * Real.sqrt 2 ∧
    (∀ z : ℝ, (∃ x : ℝ, x^2 + z^2 - 9 = 0 ∧ x^2 - 4*z + 5 = 0) → (z = z₁ ∨ z = z₂)) :=
by sorry

end common_solution_z_values_l433_43332


namespace derivative_exp_sin_l433_43308

theorem derivative_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp (Real.sin x)) x = Real.exp (Real.sin x) * Real.cos x := by
  sorry

end derivative_exp_sin_l433_43308


namespace simplify_expression_l433_43397

theorem simplify_expression : (625 : ℝ) ^ (1/4 : ℝ) * (256 : ℝ) ^ (1/3 : ℝ) = 20 := by
  sorry

end simplify_expression_l433_43397


namespace max_pies_without_ingredients_l433_43387

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (chocolate_pies marshmallow_pies cayenne_pies walnut_pies : ℕ) :
  total_pies = 48 →
  chocolate_pies ≥ 16 →
  marshmallow_pies = 24 →
  cayenne_pies = 36 →
  walnut_pies ≥ 6 →
  ∃ (pies_without_ingredients : ℕ),
    pies_without_ingredients ≤ 12 ∧
    pies_without_ingredients + chocolate_pies + marshmallow_pies + cayenne_pies + walnut_pies ≥ total_pies :=
by sorry

end max_pies_without_ingredients_l433_43387


namespace sum_of_symmetric_points_zero_l433_43320

-- Define a function v that is symmetric under 180° rotation around the origin
def v_symmetric (v : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, v (-x) = -v x

-- Theorem statement
theorem sum_of_symmetric_points_zero (v : ℝ → ℝ) (h : v_symmetric v) :
  v (-2) + v (-1) + v 1 + v 2 = 0 := by
  sorry

end sum_of_symmetric_points_zero_l433_43320


namespace davids_weighted_average_l433_43346

-- Define the marks and weightages
def english_marks : ℝ := 96
def math_marks : ℝ := 95
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 97
def biology_marks : ℝ := 95

def english_weight : ℝ := 0.1
def math_weight : ℝ := 0.2
def physics_weight : ℝ := 0.3
def chemistry_weight : ℝ := 0.2
def biology_weight : ℝ := 0.2

-- Define the weighted average calculation
def weighted_average : ℝ :=
  english_marks * english_weight +
  math_marks * math_weight +
  physics_marks * physics_weight +
  chemistry_marks * chemistry_weight +
  biology_marks * biology_weight

-- Theorem statement
theorem davids_weighted_average :
  weighted_average = 91.6 := by sorry

end davids_weighted_average_l433_43346


namespace rectangle_square_division_l433_43311

theorem rectangle_square_division (n : ℕ) : 
  (∃ (a b c d : ℕ), 
    a * b = n ∧ 
    c * d = n + 76 ∧ 
    a * d = b * c) → 
  n = 324 := by sorry

end rectangle_square_division_l433_43311


namespace integer_inequalities_result_l433_43378

theorem integer_inequalities_result (n m : ℤ) 
  (h1 : 3*n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3*m - 2*n < 46) :
  2*n + m = 36 := by
  sorry

end integer_inequalities_result_l433_43378


namespace quadratic_square_completion_l433_43304

theorem quadratic_square_completion (x : ℝ) : 
  (x^2 + 10*x + 9 = 0) → (∃ c d : ℝ, (x + c)^2 = d ∧ d = 16) :=
by sorry

end quadratic_square_completion_l433_43304


namespace lowry_bonsai_sales_l433_43312

/-- The number of small bonsai sold by Lowry -/
def small_bonsai_sold : ℕ := 3

/-- The cost of a small bonsai in dollars -/
def small_bonsai_cost : ℕ := 30

/-- The cost of a big bonsai in dollars -/
def big_bonsai_cost : ℕ := 20

/-- The number of big bonsai sold -/
def big_bonsai_sold : ℕ := 5

/-- The total earnings in dollars -/
def total_earnings : ℕ := 190

theorem lowry_bonsai_sales :
  small_bonsai_sold * small_bonsai_cost + big_bonsai_sold * big_bonsai_cost = total_earnings :=
by sorry

end lowry_bonsai_sales_l433_43312


namespace orange_count_l433_43394

theorem orange_count (initial : ℕ) (thrown_away : ℕ) (added : ℕ) :
  initial ≥ thrown_away →
  initial - thrown_away + added = initial + added - thrown_away := by
  sorry

-- Example with given values
example : 31 - 9 + 38 = 60 := by
  sorry

end orange_count_l433_43394


namespace joes_journey_time_l433_43363

/-- Represents the problem of Joe's journey to school -/
theorem joes_journey_time :
  ∀ (d : ℝ) (r_w : ℝ),
  r_w > 0 →
  3 * r_w = 3 * d / 4 →
  (3 + 1 / 4 : ℝ) = 3 + (d / 4) / (4 * r_w) :=
by sorry

end joes_journey_time_l433_43363


namespace trigonometric_identity_l433_43340

theorem trigonometric_identity : 
  (Real.cos (28 * π / 180) * Real.cos (56 * π / 180)) / Real.sin (2 * π / 180) + 
  (Real.cos (2 * π / 180) * Real.cos (4 * π / 180)) / Real.sin (28 * π / 180) = 
  (Real.sqrt 3 * Real.sin (38 * π / 180)) / (4 * Real.sin (2 * π / 180) * Real.sin (28 * π / 180)) := by
  sorry

end trigonometric_identity_l433_43340


namespace inequality_proof_l433_43309

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_proof_l433_43309


namespace sandwich_composition_ham_cost_is_correct_l433_43318

/-- The cost of a slice of ham in a sandwich -/
def ham_cost : ℚ := 25 / 100

/-- The selling price of a sandwich -/
def sandwich_price : ℚ := 150 / 100

/-- The cost of a slice of bread -/
def bread_cost : ℚ := 15 / 100

/-- The cost of a slice of cheese -/
def cheese_cost : ℚ := 35 / 100

/-- The total cost to make a sandwich -/
def sandwich_cost : ℚ := 90 / 100

/-- A sandwich contains 2 slices of bread, 1 slice of ham, and 1 slice of cheese -/
theorem sandwich_composition (h : ℚ) :
  sandwich_cost = 2 * bread_cost + h + cheese_cost :=
sorry

/-- The cost of a slice of ham is $0.25 -/
theorem ham_cost_is_correct :
  ham_cost = sandwich_cost - 2 * bread_cost - cheese_cost :=
sorry

end sandwich_composition_ham_cost_is_correct_l433_43318


namespace orange_pill_cost_l433_43314

/-- Represents the cost of pills for Alice's treatment --/
structure PillCost where
  orange : ℝ
  blue : ℝ
  duration : ℕ
  daily_intake : ℕ
  total_cost : ℝ

/-- The cost of pills satisfies the given conditions --/
def is_valid_cost (cost : PillCost) : Prop :=
  cost.orange = cost.blue + 2 ∧
  cost.duration = 21 ∧
  cost.daily_intake = 1 ∧
  cost.total_cost = 735 ∧
  cost.duration * cost.daily_intake * (cost.orange + cost.blue) = cost.total_cost

/-- The theorem stating that the cost of one orange pill is $18.5 --/
theorem orange_pill_cost (cost : PillCost) (h : is_valid_cost cost) : cost.orange = 18.5 := by
  sorry

end orange_pill_cost_l433_43314


namespace matrix_equation_solution_l433_43350

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; 4, -1]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-12, 5; 8, -3]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![-0.8, -2.6; -2.0, 1.8]
  M * A = B := by sorry

end matrix_equation_solution_l433_43350


namespace geometric_sequence_a1_l433_43366

/-- A monotonically decreasing geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a1 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 = 1 →
  a 2 + a 4 = 5/2 →
  a 1 = 4 := by
sorry

end geometric_sequence_a1_l433_43366


namespace profit_sharing_ratio_l433_43359

/-- Represents the business partnership between A and B -/
structure Partnership where
  a_initial_investment : ℕ
  b_investment : ℕ
  a_investment_duration : ℕ
  b_investment_duration : ℕ

/-- Calculates the effective capital contribution -/
def effective_capital (investment : ℕ) (duration : ℕ) : ℕ :=
  investment * duration

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplify_ratio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- Theorem stating that the profit sharing ratio is 2:3 given the conditions -/
theorem profit_sharing_ratio (p : Partnership) 
  (h1 : p.a_initial_investment = 4500)
  (h2 : p.b_investment = 16200)
  (h3 : p.a_investment_duration = 12)
  (h4 : p.b_investment_duration = 5) :
  simplify_ratio 
    (effective_capital p.a_initial_investment p.a_investment_duration)
    (effective_capital p.b_investment p.b_investment_duration) = (2, 3) := by
  sorry


end profit_sharing_ratio_l433_43359


namespace probability_two_red_two_blue_l433_43391

/-- The probability of selecting 2 red and 2 blue marbles from a bag containing 12 red marbles
    and 8 blue marbles, when 4 marbles are selected at random without replacement. -/
theorem probability_two_red_two_blue (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ)
    (selected_marbles : ℕ) :
    total_marbles = red_marbles + blue_marbles →
    total_marbles = 20 →
    red_marbles = 12 →
    blue_marbles = 8 →
    selected_marbles = 4 →
    (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2 : ℚ) /
    (Nat.choose total_marbles selected_marbles) = 56 / 147 :=
by sorry

end probability_two_red_two_blue_l433_43391


namespace pet_store_kittens_l433_43342

theorem pet_store_kittens (initial : ℕ) : initial + 3 = 9 → initial = 6 := by
  sorry

end pet_store_kittens_l433_43342


namespace max_min_values_of_f_l433_43323

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 2, f x = max) ∧
    (∀ x ∈ Set.Icc 0 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 2, f x = min) ∧
    max = 5 ∧ min = -15 := by
  sorry

end max_min_values_of_f_l433_43323


namespace circle_line_intersection_l433_43384

/-- A circle defined by x^2 + y^2 + 2x + 4y + m = 0 has exactly two points at a distance of √2
    from the line x + y + 1 = 0 if and only if m ∈ (-3, 5) -/
theorem circle_line_intersection (m : ℝ) :
  (∃! (p q : ℝ × ℝ),
    p ≠ q ∧
    (p.1^2 + p.2^2 + 2*p.1 + 4*p.2 + m = 0) ∧
    (q.1^2 + q.2^2 + 2*q.1 + 4*q.2 + m = 0) ∧
    (p.1 + p.2 + 1 ≠ 0) ∧
    (q.1 + q.2 + 1 ≠ 0) ∧
    ((p.1 + p.2 + 1)^2 / 2 = 2) ∧
    ((q.1 + q.2 + 1)^2 / 2 = 2))
  ↔
  (-3 < m ∧ m < 5) :=
by sorry

end circle_line_intersection_l433_43384


namespace doubled_container_volume_l433_43389

/-- The volume of a container after doubling its dimensions -/
def doubled_volume (original_volume : ℝ) : ℝ := 8 * original_volume

/-- Theorem: Doubling the dimensions of a 3-gallon container results in a 24-gallon container -/
theorem doubled_container_volume : doubled_volume 3 = 24 := by
  sorry

end doubled_container_volume_l433_43389


namespace harmonic_mean_pairs_l433_43302

theorem harmonic_mean_pairs : 
  let count := Finset.filter (fun p : ℕ × ℕ => 
    p.1 < p.2 ∧ 
    (2 * p.1 * p.2 : ℚ) / (p.1 + p.2) = 4^30
  ) (Finset.range (2^61 + 1) ×ˢ Finset.range (2^61 + 1))
  
  count.card = 61 := by
  sorry

end harmonic_mean_pairs_l433_43302


namespace no_integer_solutions_l433_43322

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x ≠ 1 ∧ (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end no_integer_solutions_l433_43322


namespace correct_calculation_l433_43325

theorem correct_calculation (x : ℤ) (h : x + 5 = 43) : 5 * x = 190 := by
  sorry

end correct_calculation_l433_43325


namespace reservoir_capacity_l433_43377

theorem reservoir_capacity : 
  ∀ (C : ℝ), 
  (C / 3 + 150 = 3 * C / 4) → 
  C = 360 := by
sorry

end reservoir_capacity_l433_43377


namespace sons_age_l433_43327

theorem sons_age (son father : ℕ) 
  (h1 : son = (father / 4) - 1)
  (h2 : father = 5 * son - 5) : 
  son = 9 := by
sorry

end sons_age_l433_43327


namespace base_6_arithmetic_l433_43328

/-- Convert a base 6 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (λ acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a base 10 number to base 6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Theorem: 1254₆ - 432₆ + 221₆ = 1043₆ in base 6 --/
theorem base_6_arithmetic :
  to_base_6 (to_base_10 [4, 5, 2, 1] - to_base_10 [2, 3, 4] + to_base_10 [1, 2, 2]) = [3, 4, 0, 1] :=
sorry

end base_6_arithmetic_l433_43328


namespace smaller_rectangle_dimensions_l433_43341

theorem smaller_rectangle_dimensions 
  (square_side : ℝ) 
  (h_square_side : square_side = 10) 
  (small_length small_width : ℝ) 
  (h_rectangles : small_length + 2 * small_length = square_side) 
  (h_square : small_width = small_length) : 
  small_length = 10 / 3 ∧ small_width = 10 / 3 := by
sorry

end smaller_rectangle_dimensions_l433_43341


namespace sphere_radius_ratio_l433_43356

theorem sphere_radius_ratio : 
  ∀ (r R : ℝ), 
    (4 / 3 * π * r^3 = 36 * π) → 
    (4 / 3 * π * R^3 = 450 * π) → 
    r / R = 1 / Real.rpow 12.5 (1/3) := by
  sorry

end sphere_radius_ratio_l433_43356


namespace sum_of_digits_of_M_l433_43330

-- Define M as a positive integer
def M : ℕ+ := sorry

-- Define the condition M^2 = 36^49 * 49^36
axiom M_squared : (M : ℕ)^2 = 36^49 * 49^36

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_M : sum_of_digits (M : ℕ) = 21 := by sorry

end sum_of_digits_of_M_l433_43330


namespace problem_1_problem_2_l433_43326

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ :=
  if a ≥ b then b^2 else 2*a - b

-- Theorem statements
theorem problem_1 : triangle (-4) (-5) = 25 := by sorry

theorem problem_2 : triangle (triangle (-3) 2) (-9) = 81 := by sorry

end problem_1_problem_2_l433_43326


namespace smallest_element_mean_l433_43381

/-- The arithmetic mean of the smallest number in all r-element subsets of {1, 2, ..., n} -/
def f (r n : ℕ+) : ℚ :=
  (n + 1) / (r + 1)

/-- Theorem stating that f(r, n) is the arithmetic mean of the smallest number
    in all r-element subsets of {1, 2, ..., n} -/
theorem smallest_element_mean (r n : ℕ+) (h : r ≤ n) :
  f r n = (Finset.sum (Finset.range (n - r + 1)) (fun a => a * (Nat.choose (n - a) (r - 1)))) /
          (Nat.choose n r) :=
sorry

end smallest_element_mean_l433_43381


namespace system_of_equations_solutions_l433_43306

theorem system_of_equations_solutions :
  (∃ x y : ℚ, x + y = 3 ∧ x - y = 1 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℚ, 2*x + y = 3 ∧ x - 2*y = 1 ∧ x = 7/5 ∧ y = 1/5) := by
  sorry

end system_of_equations_solutions_l433_43306


namespace equation_solution_l433_43357

theorem equation_solution : 
  ∃ x : ℝ, (5 * 1.6 - (2 * x) / 1.3 = 4) ∧ (x = 2.6) := by
  sorry

end equation_solution_l433_43357


namespace smallest_m_for_integral_solutions_l433_43368

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), 
  (m > 0) ∧ 
  (∃ (x : ℤ), 12 * x^2 - m * x + 432 = 0) ∧
  (∀ (k : ℕ), k > 0 ∧ k < m → ¬∃ (y : ℤ), 12 * y^2 - k * y + 432 = 0) ∧
  m = 144 :=
by sorry

end smallest_m_for_integral_solutions_l433_43368


namespace pi_fourth_in_range_of_f_l433_43317

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem pi_fourth_in_range_of_f : ∃ (x : ℝ), f x = π / 4 := by
  sorry

end pi_fourth_in_range_of_f_l433_43317


namespace line_parallel_perp_plane_l433_43385

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perp_plane
  (m n : Line) (α : Plane)
  (h1 : m ≠ n)
  (h2 : parallel m n)
  (h3 : perp m α) :
  perp n α :=
sorry

end line_parallel_perp_plane_l433_43385


namespace halloween_candy_weight_l433_43319

/-- Represents the weight of different types of candy in pounds -/
structure CandyWeights where
  chocolate : ℝ
  gummyBears : ℝ
  caramels : ℝ
  hardCandy : ℝ

/-- Calculates the total weight of candy -/
def totalWeight (cw : CandyWeights) : ℝ :=
  cw.chocolate + cw.gummyBears + cw.caramels + cw.hardCandy

/-- Frank's candy weights -/
def frankCandy : CandyWeights := {
  chocolate := 3,
  gummyBears := 2,
  caramels := 1,
  hardCandy := 4
}

/-- Gwen's candy weights -/
def gwenCandy : CandyWeights := {
  chocolate := 2,
  gummyBears := 2.5,
  caramels := 1,
  hardCandy := 1.5
}

/-- Theorem stating that the total combined weight of Frank and Gwen's Halloween candy is 17 pounds -/
theorem halloween_candy_weight :
  totalWeight frankCandy + totalWeight gwenCandy = 17 := by
  sorry

end halloween_candy_weight_l433_43319


namespace initial_lions_l433_43365

/-- Proves that the initial number of lions is 100 given the conditions of the problem -/
theorem initial_lions (net_increase_per_month : ℕ) (total_increase : ℕ) (final_count : ℕ) : 
  net_increase_per_month = 4 → 
  total_increase = 48 → 
  final_count = 148 → 
  final_count - total_increase = 100 := by
sorry

end initial_lions_l433_43365


namespace max_product_partition_l433_43392

/-- Given positive integers k and n with k ≥ n, where k = nq + r (0 ≤ r < n),
    F(k) is the maximum product of n positive integers that sum to k. -/
def F (k n : ℕ+) (h : k ≥ n) : ℕ := by sorry

/-- The quotient when k is divided by n -/
def q (k n : ℕ+) : ℕ := k / n

/-- The remainder when k is divided by n -/
def r (k n : ℕ+) : ℕ := k % n

theorem max_product_partition (k n : ℕ+) (h : k ≥ n) :
  F k n h = (q k n) ^ (n - r k n) * ((q k n) + 1) ^ (r k n) := by sorry

end max_product_partition_l433_43392


namespace perfect_square_sum_l433_43344

theorem perfect_square_sum : ∃ k : ℕ, 2^8 + 2^11 + 2^12 = k^2 := by
  sorry

end perfect_square_sum_l433_43344


namespace cylinder_minus_cones_volume_l433_43331

/-- The volume of a cylinder minus the volume of two congruent cones --/
theorem cylinder_minus_cones_volume 
  (r : ℝ) 
  (h_cylinder : ℝ) 
  (h_cone : ℝ) 
  (h_cylinder_eq : h_cylinder = 30) 
  (h_cone_eq : h_cone = 15) 
  (r_eq : r = 10) : 
  π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end cylinder_minus_cones_volume_l433_43331


namespace given_number_equals_scientific_form_l433_43395

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  is_valid : 1 ≤ mantissa ∧ mantissa < 10

/-- The given number in decimal form -/
def given_number : ℝ := 0.000123

/-- The scientific notation representation of the given number -/
def scientific_form : ScientificNotation := {
  mantissa := 1.23,
  exponent := -4,
  is_valid := by sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_equals_scientific_form :
  given_number = scientific_form.mantissa * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end given_number_equals_scientific_form_l433_43395


namespace total_apples_l433_43355

theorem total_apples (marin_apples : ℕ) (david_apples : ℕ) (amanda_apples : ℕ) : 
  marin_apples = 6 →
  david_apples = 2 * marin_apples →
  amanda_apples = david_apples + 5 →
  marin_apples + david_apples + amanda_apples = 35 :=
by sorry

end total_apples_l433_43355


namespace problem_statement_l433_43383

theorem problem_statement (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 6) :
  a * b^2 - a^2 * b = -24 := by
sorry

end problem_statement_l433_43383


namespace unique_integer_satisfying_expression_l433_43376

def is_integer_expression (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.factorial (n^3 - 1)) = k * (Nat.factorial n)^(n^2)

theorem unique_integer_satisfying_expression :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ is_integer_expression n :=
sorry

end unique_integer_satisfying_expression_l433_43376


namespace total_yellow_marbles_l433_43367

/-- The total number of yellow marbles given the number of marbles each person has -/
def total_marbles (mary_marbles joan_marbles john_marbles : ℕ) : ℕ :=
  mary_marbles + joan_marbles + john_marbles

/-- Theorem stating that the total number of yellow marbles is 19 -/
theorem total_yellow_marbles :
  total_marbles 9 3 7 = 19 := by
  sorry

end total_yellow_marbles_l433_43367


namespace max_volume_at_five_l433_43360

def box_volume (x : ℝ) : ℝ := (30 - 2*x)^2 * x

def possible_x : Set ℝ := {4, 5, 6, 7}

theorem max_volume_at_five :
  ∀ x ∈ possible_x, x ≠ 5 → box_volume x ≤ box_volume 5 := by
  sorry

end max_volume_at_five_l433_43360


namespace divisors_of_2013_power_13_l433_43316

theorem divisors_of_2013_power_13 : 
  let n : ℕ := 2013^13
  ∀ (p : ℕ → Prop), 
    (∀ k, p k ↔ k ∣ n ∧ k > 0) →
    (2013 = 3 * 11 * 61) →
    (∃! (s : Finset ℕ), ∀ k, k ∈ s ↔ p k) →
    Finset.card s = 2744 := by
  sorry

end divisors_of_2013_power_13_l433_43316


namespace smallest_possible_a_l433_43374

theorem smallest_possible_a (a b c : ℝ) : 
  a > 0 → 
  (∃ n : ℤ, a + 2*b + 3*c = n) →
  (∀ x y : ℝ, y = a*x^2 + b*x + c ↔ y = a*(x - 1/2)^2 - 1/2) →
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, (∃ n : ℤ, a' + 2*b' + 3*c' = n) ∧
    (∀ x y : ℝ, y = a'*x^2 + b'*x + c' ↔ y = a'*(x - 1/2)^2 - 1/2)) →
    a ≤ a') →
  a = 2 :=
sorry

end smallest_possible_a_l433_43374


namespace smallest_sum_of_four_primes_l433_43396

/-- Given four positive prime numbers whose product equals the sum of 55 consecutive positive integers,
    the smallest possible sum of these four primes is 28. -/
theorem smallest_sum_of_four_primes (a b c d : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime c → Nat.Prime d →
  (∃ x : ℕ, a * b * c * d = (55 : ℕ) * (x + 27)) →
  (∀ w x y z : ℕ, Nat.Prime w → Nat.Prime x → Nat.Prime y → Nat.Prime z →
    (∃ n : ℕ, w * x * y * z = (55 : ℕ) * (n + 27)) →
    a + b + c + d ≤ w + x + y + z) →
  a + b + c + d = 28 :=
sorry

end smallest_sum_of_four_primes_l433_43396


namespace jack_remaining_notebooks_l433_43369

-- Define the initial number of notebooks for Gerald
def gerald_notebooks : ℕ := 8

-- Define Jack's initial number of notebooks relative to Gerald's
def jack_initial_notebooks : ℕ := gerald_notebooks + 13

-- Define the number of notebooks Jack gives to Paula
def notebooks_to_paula : ℕ := 5

-- Define the number of notebooks Jack gives to Mike
def notebooks_to_mike : ℕ := 6

-- Theorem: Jack has 10 notebooks left
theorem jack_remaining_notebooks :
  jack_initial_notebooks - (notebooks_to_paula + notebooks_to_mike) = 10 := by
  sorry

end jack_remaining_notebooks_l433_43369


namespace june_found_seventeen_eggs_l433_43361

/-- The number of eggs June found -/
def total_eggs : ℕ :=
  let nest1_eggs := 2 * 5  -- 2 nests with 5 eggs each in 1 tree
  let nest2_eggs := 1 * 3  -- 1 nest with 3 eggs in another tree
  let nest3_eggs := 1 * 4  -- 1 nest with 4 eggs in the front yard
  nest1_eggs + nest2_eggs + nest3_eggs

/-- Theorem stating that June found 17 eggs in total -/
theorem june_found_seventeen_eggs : total_eggs = 17 := by
  sorry

end june_found_seventeen_eggs_l433_43361


namespace no_solutions_prime_factorial_inequality_l433_43362

theorem no_solutions_prime_factorial_inequality :
  ¬ ∃ (n k : ℕ), Prime n ∧ n ≤ n! - k^n ∧ n! - k^n ≤ k * n :=
by sorry

end no_solutions_prime_factorial_inequality_l433_43362


namespace ellen_painted_ten_roses_l433_43380

/-- The time it takes to paint different types of flowers and vines --/
structure PaintingTimes where
  lily : ℕ
  rose : ℕ
  orchid : ℕ
  vine : ℕ

/-- The number of each type of flower and vine painted --/
structure FlowerCounts where
  lilies : ℕ
  roses : ℕ
  orchids : ℕ
  vines : ℕ

/-- Calculates the total time spent painting based on the painting times and flower counts --/
def totalPaintingTime (times : PaintingTimes) (counts : FlowerCounts) : ℕ :=
  times.lily * counts.lilies +
  times.rose * counts.roses +
  times.orchid * counts.orchids +
  times.vine * counts.vines

/-- Theorem: Given the painting times and flower counts, prove that Ellen painted 10 roses --/
theorem ellen_painted_ten_roses
  (times : PaintingTimes)
  (counts : FlowerCounts)
  (h1 : times.lily = 5)
  (h2 : times.rose = 7)
  (h3 : times.orchid = 3)
  (h4 : times.vine = 2)
  (h5 : counts.lilies = 17)
  (h6 : counts.orchids = 6)
  (h7 : counts.vines = 20)
  (h8 : totalPaintingTime times counts = 213) :
  counts.roses = 10 :=
sorry

end ellen_painted_ten_roses_l433_43380


namespace translation_result_l433_43313

/-- A line in the xy-plane is represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount. -/
def translateLine (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

/-- The original line y = 2x - 1 -/
def originalLine : Line :=
  { slope := 2, intercept := -1 }

/-- The amount of upward translation -/
def translationAmount : ℝ := 2

/-- The resulting line after translation -/
def resultingLine : Line := translateLine originalLine translationAmount

theorem translation_result :
  resultingLine.slope = 2 ∧ resultingLine.intercept = 1 := by
  sorry

end translation_result_l433_43313


namespace solve_inequality_part1_solve_inequality_part2_l433_43371

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

-- Part 1: Solve f(x) < 0 when a = -4
theorem solve_inequality_part1 : 
  ∀ x : ℝ, f (-4) x < 0 ↔ 1 < x ∧ x < 3 :=
sorry

-- Part 2: Find range of a when f(x) > 0 for all real x
theorem solve_inequality_part2 : 
  (∀ x : ℝ, f a x > 0) ↔ -2 * Real.sqrt 3 < a ∧ a < 2 * Real.sqrt 3 :=
sorry

end solve_inequality_part1_solve_inequality_part2_l433_43371


namespace distribute_6_5_l433_43373

def distribute (n m : ℕ) : ℕ := 
  Nat.choose (m - 1) (n - m)

theorem distribute_6_5 : distribute 6 5 = 5 := by
  sorry

end distribute_6_5_l433_43373


namespace trigonometric_simplification_l433_43358

theorem trigonometric_simplification :
  (Real.sin (7 * π / 180) + Real.cos (15 * π / 180) * Real.sin (8 * π / 180)) /
  (Real.cos (7 * π / 180) - Real.sin (15 * π / 180) * Real.sin (8 * π / 180)) = 2 - Real.sqrt 3 := by
  sorry

end trigonometric_simplification_l433_43358


namespace log_equality_l433_43305

theorem log_equality : Real.log 16 / Real.log 4096 = Real.log 4 / Real.log 64 := by
  sorry

#check log_equality

end log_equality_l433_43305


namespace proportion_problem_l433_43390

theorem proportion_problem (x : ℝ) : 
  (x / 5 = 0.96 / 8) → x = 0.6 := by
sorry

end proportion_problem_l433_43390


namespace triangle_inequality_l433_43386

theorem triangle_inequality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_triangle : (x + y - z) * (y + z - x) * (z + x - y) > 0) : 
  x * (y + z)^2 + y * (z + x)^2 + z * (x + y)^2 - (x^3 + y^3 + z^3) ≤ 9 * x * y * z := by
  sorry

end triangle_inequality_l433_43386


namespace sqrt_x_plus_one_meaningful_l433_43329

theorem sqrt_x_plus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by sorry

end sqrt_x_plus_one_meaningful_l433_43329


namespace problem_sum_value_l433_43338

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the geometric series (3/4)^k from k=1 to 12 -/
def problem_sum : ℚ := geometric_sum (3/4) (3/4) 12

theorem problem_sum_value : problem_sum = 48738225 / 16777216 := by
  sorry

end problem_sum_value_l433_43338


namespace estimate_value_l433_43382

theorem estimate_value : 
  3 < (2 * Real.sqrt 2 + Real.sqrt 6) * Real.sqrt (1/2) ∧ 
  (2 * Real.sqrt 2 + Real.sqrt 6) * Real.sqrt (1/2) < 4 := by
  sorry

end estimate_value_l433_43382


namespace chinese_count_l433_43321

theorem chinese_count (total : ℕ) (americans : ℕ) (australians : ℕ) 
  (h1 : total = 49)
  (h2 : americans = 16)
  (h3 : australians = 11) :
  total - (americans + australians) = 22 := by
sorry

end chinese_count_l433_43321


namespace transaction_fraction_l433_43399

theorem transaction_fraction (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ) : 
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + mabel_transactions / 10 →
  jade_transactions = 81 →
  jade_transactions = cal_transactions + 15 →
  cal_transactions * 3 = anthony_transactions * 2 := by
sorry

end transaction_fraction_l433_43399


namespace ellipse_condition_l433_43343

/-- Represents an ellipse with equation ax^2 + by^2 = 1 -/
structure Ellipse (a b : ℝ) where
  equation : ∀ x y : ℝ, a * x^2 + b * y^2 = 1
  is_ellipse : True  -- We assume it's an ellipse
  foci_on_x_axis : True  -- We assume foci are on x-axis

/-- 
If ax^2 + by^2 = 1 represents an ellipse with foci on the x-axis,
where a and b are real numbers, then b > a > 0.
-/
theorem ellipse_condition (a b : ℝ) (e : Ellipse a b) : b > a ∧ a > 0 := by
  sorry

end ellipse_condition_l433_43343


namespace arithmetic_geometric_sequence_property_l433_43370

/-- A positive arithmetic geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ (a (n + 1) - a n) = (a (n + 2) - a (n + 1))
    ∧ (a (n + 1))^2 = (a n) * (a (n + 2))

theorem arithmetic_geometric_sequence_property
  (a : ℕ → ℝ) (h : ArithmeticGeometricSequence a)
  (h_eq : a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) :
  a 3 + a 6 = 4 := by
sorry

end arithmetic_geometric_sequence_property_l433_43370


namespace greatest_two_digit_prime_saturated_is_98_l433_43347

/-- A number is prime saturated if the product of all its different positive prime factors
    is less than its square root -/
def IsPrimeSaturated (n : ℕ) : Prop :=
  (Finset.prod (Nat.factors n).toFinset id) < Real.sqrt (n : ℝ)

/-- The greatest two-digit prime saturated integer -/
def GreatestTwoDigitPrimeSaturated : ℕ := 98

theorem greatest_two_digit_prime_saturated_is_98 :
  IsPrimeSaturated GreatestTwoDigitPrimeSaturated ∧
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ IsPrimeSaturated n → n ≤ GreatestTwoDigitPrimeSaturated :=
by
  sorry

end greatest_two_digit_prime_saturated_is_98_l433_43347


namespace equation_represents_pair_of_lines_l433_43353

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space defined by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point satisfies the equation 9x^2 - 25y^2 = 0 -/
def satisfiesEquation (p : Point2D) : Prop :=
  9 * p.x^2 - 25 * p.y^2 = 0

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The two lines that form the solution -/
def line1 : Line2D := { a := 3, b := -5, c := 0 }
def line2 : Line2D := { a := 3, b := 5, c := 0 }

/-- Theorem stating that the equation represents a pair of straight lines -/
theorem equation_represents_pair_of_lines :
  ∀ p : Point2D, satisfiesEquation p ↔ (pointOnLine p line1 ∨ pointOnLine p line2) :=
sorry


end equation_represents_pair_of_lines_l433_43353


namespace exists_unique_solution_l433_43398

theorem exists_unique_solution : ∃! x : ℝ, 
  (0.86 : ℝ)^3 - (0.1 : ℝ)^3 / (0.86 : ℝ)^2 + x + (0.1 : ℝ)^2 = 0.76 := by
  sorry

end exists_unique_solution_l433_43398


namespace circus_tent_sections_l433_43336

theorem circus_tent_sections (section_capacity : ℕ) (total_capacity : ℕ) : 
  section_capacity = 246 → total_capacity = 984 → total_capacity / section_capacity = 4 := by
  sorry

end circus_tent_sections_l433_43336


namespace inverse_g_sum_l433_43303

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else x^2 - 4*x + 5

theorem inverse_g_sum : ∃ y₁ y₂ y₃ : ℝ,
  g y₁ = -1 ∧ g y₂ = 1 ∧ g y₃ = 4 ∧ y₁ + y₂ + y₃ = 4 :=
sorry

end inverse_g_sum_l433_43303


namespace f_extrema_l433_43364

-- Define the function f(x)
def f (p q x : ℝ) : ℝ := x^3 - p*x^2 - q*x

-- State the theorem
theorem f_extrema (p q : ℝ) :
  (f p q 1 = 0) →
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f p q x ≤ f p q x₀) ∧
  (f p q x₀ = 4/27) ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f p q x ≥ f p q x₁) ∧
  (f p q x₁ = -4) :=
by sorry

end f_extrema_l433_43364


namespace johnny_runs_four_times_l433_43339

theorem johnny_runs_four_times (block_length : ℝ) (average_distance : ℝ) :
  block_length = 200 →
  average_distance = 600 →
  ∃ (johnny_runs : ℕ),
    (average_distance = (block_length * johnny_runs + block_length * (johnny_runs / 2)) / 2) ∧
    johnny_runs = 4 :=
by sorry

end johnny_runs_four_times_l433_43339


namespace polynomial_factorization_l433_43307

/-- Proves that x³ - 2x²y + xy² = x(x-y)² for all real numbers x and y -/
theorem polynomial_factorization (x y : ℝ) : 
  x^3 - 2*x^2*y + x*y^2 = x*(x-y)^2 := by sorry

end polynomial_factorization_l433_43307


namespace first_hour_premium_l433_43349

/-- A psychologist charges different rates for the first hour and additional hours of therapy. -/
structure TherapyRates where
  /-- The charge for the first hour of therapy -/
  first_hour : ℝ
  /-- The charge for each additional hour of therapy -/
  additional_hour : ℝ
  /-- The total charge for 5 hours of therapy is $375 -/
  five_hour_total : first_hour + 4 * additional_hour = 375
  /-- The total charge for 2 hours of therapy is $174 -/
  two_hour_total : first_hour + additional_hour = 174

/-- The difference between the first hour charge and additional hour charge is $40 -/
theorem first_hour_premium (rates : TherapyRates) : 
  rates.first_hour - rates.additional_hour = 40 := by
  sorry

end first_hour_premium_l433_43349


namespace fraction_equality_l433_43337

theorem fraction_equality : 
  let f (x : ℕ) := x^4 + 324
  (∀ x, f x = (x^2 - 6*x + 18) * (x^2 + 6*x + 18)) →
  (f 64 * f 52 * f 40 * f 28 * f 16) / (f 58 * f 46 * f 34 * f 22 * f 10) = 137 / 1513 :=
by sorry

end fraction_equality_l433_43337


namespace total_paid_equals_143_l433_43375

def manicure_cost : ℝ := 30
def pedicure_cost : ℝ := 40
def hair_treatment_cost : ℝ := 50

def manicure_tip_rate : ℝ := 0.25
def pedicure_tip_rate : ℝ := 0.20
def hair_treatment_tip_rate : ℝ := 0.15

def total_cost (service_cost : ℝ) (tip_rate : ℝ) : ℝ :=
  service_cost * (1 + tip_rate)

theorem total_paid_equals_143 :
  total_cost manicure_cost manicure_tip_rate +
  total_cost pedicure_cost pedicure_tip_rate +
  total_cost hair_treatment_cost hair_treatment_tip_rate = 143 := by
  sorry

end total_paid_equals_143_l433_43375


namespace total_berets_is_eleven_l433_43348

def spools_per_beret : ℕ := 3

def red_spools : ℕ := 12
def black_spools : ℕ := 15
def blue_spools : ℕ := 6

def berets_from_spools (spools : ℕ) : ℕ := spools / spools_per_beret

theorem total_berets_is_eleven :
  berets_from_spools red_spools + berets_from_spools black_spools + berets_from_spools blue_spools = 11 := by
  sorry

end total_berets_is_eleven_l433_43348


namespace multiples_of_six_ending_in_four_l433_43354

theorem multiples_of_six_ending_in_four (n : ℕ) : 
  (∃ m : ℕ, m = 10) ↔ 
  (∀ k : ℕ, (6 * k < 600 ∧ (6 * k) % 10 = 4) → k ≤ n) ∧ 
  (∃ (k₁ k₂ : ℕ), k₁ ≤ n ∧ k₂ ≤ n ∧ k₁ ≠ k₂ ∧ 
    6 * k₁ < 600 ∧ (6 * k₁) % 10 = 4 ∧ 
    6 * k₂ < 600 ∧ (6 * k₂) % 10 = 4) :=
by sorry

end multiples_of_six_ending_in_four_l433_43354


namespace friday_attendance_l433_43310

/-- Calculates the percentage of students present on a given day -/
def students_present (initial_absenteeism : ℝ) (daily_increase : ℝ) (day : ℕ) : ℝ :=
  100 - (initial_absenteeism + daily_increase * day)

/-- Proves that the percentage of students present on Friday is 78% -/
theorem friday_attendance 
  (initial_absenteeism : ℝ) 
  (daily_increase : ℝ) 
  (h1 : initial_absenteeism = 14) 
  (h2 : daily_increase = 2) : 
  students_present initial_absenteeism daily_increase 4 = 78 := by
  sorry

#eval students_present 14 2 4

end friday_attendance_l433_43310


namespace ryanne_hezekiah_age_difference_l433_43301

/-- Given that Ryanne and Hezekiah's combined age is 15 and Hezekiah is 4 years old,
    prove that Ryanne is 7 years older than Hezekiah. -/
theorem ryanne_hezekiah_age_difference :
  ∀ (ryanne_age hezekiah_age : ℕ),
    ryanne_age + hezekiah_age = 15 →
    hezekiah_age = 4 →
    ryanne_age - hezekiah_age = 7 :=
by
  sorry

end ryanne_hezekiah_age_difference_l433_43301


namespace seashell_fraction_proof_l433_43334

def dozen : ℕ := 12

theorem seashell_fraction_proof 
  (mimi_shells : ℕ) 
  (kyle_shells : ℕ) 
  (leigh_shells : ℕ) :
  mimi_shells = 2 * dozen →
  kyle_shells = 2 * mimi_shells →
  leigh_shells = 16 →
  (leigh_shells : ℚ) / (kyle_shells : ℚ) = 1 / 3 := by
  sorry

end seashell_fraction_proof_l433_43334


namespace intersection_max_difference_zero_l433_43345

-- Define the polynomial functions
def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := x^2 + x^4

-- State the theorem
theorem intersection_max_difference_zero :
  (∀ x : ℝ, f x = g x → x = -1) →  -- Given condition: x = -1 is the only intersection
  (∃ x : ℝ, f x = g x) →           -- Ensure at least one intersection exists
  (∀ x y : ℝ, f x = g x ∧ f y = g y → |f x - f y| = 0) := by
  sorry

end intersection_max_difference_zero_l433_43345


namespace fraction_addition_l433_43333

theorem fraction_addition : (3 / 4) / (5 / 8) + 1 / 8 = 53 / 40 := by
  sorry

end fraction_addition_l433_43333


namespace amusement_park_admission_l433_43372

theorem amusement_park_admission (child_fee adult_fee : ℚ) 
  (total_people : ℕ) (total_fees : ℚ) :
  child_fee = 3/2 →
  adult_fee = 4 →
  total_people = 315 →
  total_fees = 810 →
  ∃ (children adults : ℕ),
    children + adults = total_people ∧
    child_fee * children + adult_fee * adults = total_fees ∧
    children = 180 := by
  sorry

end amusement_park_admission_l433_43372


namespace students_in_section_A_l433_43379

/-- The number of students in section A -/
def students_A : ℕ := 26

/-- The number of students in section B -/
def students_B : ℕ := 34

/-- The average weight of students in section A (in kg) -/
def avg_weight_A : ℚ := 50

/-- The average weight of students in section B (in kg) -/
def avg_weight_B : ℚ := 30

/-- The average weight of the whole class (in kg) -/
def avg_weight_total : ℚ := 38.67

theorem students_in_section_A : 
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = avg_weight_total := by
  sorry

end students_in_section_A_l433_43379


namespace power_seven_mod_twelve_l433_43351

theorem power_seven_mod_twelve : 7^93 % 12 = 7 := by
  sorry

end power_seven_mod_twelve_l433_43351


namespace white_balls_count_l433_43393

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 60 →
  green = 10 →
  yellow = 7 →
  red = 15 →
  purple = 6 →
  prob_not_red_purple = 13/20 →
  ∃ white : ℕ, white = 22 ∧ total = white + green + yellow + red + purple :=
by sorry

end white_balls_count_l433_43393


namespace ln_inequality_range_l433_43335

theorem ln_inequality_range (x : ℝ) (m : ℝ) (h : x > 0) :
  (∀ x > 0, Real.log x ≤ x * Real.exp (m^2 - m - 1)) ↔ (m ≤ 0 ∨ m ≥ 1) := by
  sorry

end ln_inequality_range_l433_43335


namespace sqrt_equation_solution_l433_43388

theorem sqrt_equation_solution : ∃ x : ℝ, x = 1225 / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 := by
  sorry

end sqrt_equation_solution_l433_43388
