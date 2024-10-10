import Mathlib

namespace original_number_proof_l2813_281302

def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

theorem original_number_proof :
  let original := 1453789
  let swapped := 8453719
  (∃ i j, swap_digits original i j = swapped) ∧
  (swapped > 3 * original) :=
by sorry

end original_number_proof_l2813_281302


namespace cards_distribution_l2813_281367

/-- Given a deck of 48 cards dealt as evenly as possible among 9 people,
    the number of people who receive fewer than 6 cards is 6. -/
theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 48) (h2 : num_people = 9) :
  let cards_per_person := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_fewer := num_people - people_with_extra
  people_with_fewer = 6 := by
  sorry

end cards_distribution_l2813_281367


namespace largest_n_for_sin_cos_inequality_l2813_281332

theorem largest_n_for_sin_cos_inequality : 
  ∃ n : ℕ+, (∀ m : ℕ+, m > n → ∃ x : ℝ, (Real.sin x + Real.cos x)^(m : ℝ) < 2 / (m : ℝ)) ∧
             (∀ x : ℝ, (Real.sin x + Real.cos x)^(n : ℝ) ≥ 2 / (n : ℝ)) ∧
             n = 4 := by
  sorry

end largest_n_for_sin_cos_inequality_l2813_281332


namespace no_trapezoid_solution_l2813_281337

theorem no_trapezoid_solution : ¬∃ (b₁ b₂ : ℕ), 
  b₁ > 0 ∧ b₂ > 0 ∧
  b₁ % 12 = 0 ∧ b₂ % 12 = 0 ∧
  80 * (b₁ + b₂) / 2 = 2800 :=
sorry

end no_trapezoid_solution_l2813_281337


namespace blue_face_probability_l2813_281339

/-- A cube with colored faces -/
structure ColoredCube where
  blue_faces : ℕ
  red_faces : ℕ

/-- The probability of rolling a specific color on a colored cube -/
def roll_probability (cube : ColoredCube) (color : String) : ℚ :=
  match color with
  | "blue" => cube.blue_faces / (cube.blue_faces + cube.red_faces)
  | "red" => cube.red_faces / (cube.blue_faces + cube.red_faces)
  | _ => 0

/-- Theorem: The probability of rolling a blue face on a cube with 3 blue faces and 3 red faces is 1/2 -/
theorem blue_face_probability :
  ∀ (cube : ColoredCube),
    cube.blue_faces = 3 →
    cube.red_faces = 3 →
    roll_probability cube "blue" = 1/2 :=
by
  sorry

end blue_face_probability_l2813_281339


namespace ratio_problem_l2813_281320

theorem ratio_problem (first_number second_number : ℝ) : 
  first_number / second_number = 20 → first_number = 200 → second_number = 10 := by
  sorry

end ratio_problem_l2813_281320


namespace log_ratio_squared_l2813_281383

theorem log_ratio_squared (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1) 
  (h_log : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (h_prod : x * y^2 = 243) : 
  (Real.log (x/y) / Real.log 3)^2 = 49/36 := by
sorry

end log_ratio_squared_l2813_281383


namespace smallest_n_congruence_l2813_281354

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 1456 [ZMOD 11]) → n ≥ 6 :=
by sorry

end smallest_n_congruence_l2813_281354


namespace cyclic_sum_minimum_l2813_281369

theorem cyclic_sum_minimum (a b c d : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (sum_eq_four : a + b + c + d = 4) : 
  ((b + 3) / (a^2 + 4) + 
   (c + 3) / (b^2 + 4) + 
   (d + 3) / (c^2 + 4) + 
   (a + 3) / (d^2 + 4)) ≥ 3 ∧ 
  ∃ a b c d, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a + b + c + d = 4 ∧ 
    ((b + 3) / (a^2 + 4) + 
     (c + 3) / (b^2 + 4) + 
     (d + 3) / (c^2 + 4) + 
     (a + 3) / (d^2 + 4)) = 3 :=
by sorry

end cyclic_sum_minimum_l2813_281369


namespace largest_digit_sum_quotient_l2813_281342

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

theorem largest_digit_sum_quotient :
  (∀ n : ThreeDigitNumber, (value n : ℚ) / (digitSum n : ℚ) ≤ 100) ∧
  (∃ n : ThreeDigitNumber, (value n : ℚ) / (digitSum n : ℚ) = 100) := by
  sorry

end largest_digit_sum_quotient_l2813_281342


namespace max_trig_sum_l2813_281379

theorem max_trig_sum (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) :
  Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + 
  Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₅ + 
  Real.cos θ₅ * Real.sin θ₁ ≤ 5/2 := by
sorry

end max_trig_sum_l2813_281379


namespace quadratic_equations_solutions_l2813_281356

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 1 = 0 ∧ x₂^2 - 1 = 0 ∧ x₁ = 1 ∧ x₂ = -1) ∧
  (∃ x₁ x₂ : ℝ, x₁^2 - 3*x₁ + 1 = 0 ∧ x₂^2 - 3*x₂ + 1 = 0 ∧
    x₁ = (3 + Real.sqrt 5) / 2 ∧ x₂ = (3 - Real.sqrt 5) / 2) :=
by sorry

end quadratic_equations_solutions_l2813_281356


namespace complex_product_magnitude_l2813_281314

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 7 →
  a * b = t - 6 * Complex.I →
  t = 9 * Real.sqrt 5 := by
sorry

end complex_product_magnitude_l2813_281314


namespace fencing_cost_theorem_l2813_281380

/-- The cost of fencing an irregularly shaped field -/
theorem fencing_cost_theorem (triangle_side1 triangle_side2 triangle_side3 circle_radius : ℝ)
  (triangle_cost_per_meter circle_cost_per_meter : ℝ)
  (h1 : triangle_side1 = 100)
  (h2 : triangle_side2 = 150)
  (h3 : triangle_side3 = 50)
  (h4 : circle_radius = 30)
  (h5 : triangle_cost_per_meter = 5)
  (h6 : circle_cost_per_meter = 7) :
  ∃ (total_cost : ℝ), 
    abs (total_cost - ((triangle_side1 + triangle_side2 + triangle_side3) * triangle_cost_per_meter +
    2 * Real.pi * circle_radius * circle_cost_per_meter)) < 1 ∧
    total_cost = 2819 :=
by sorry

end fencing_cost_theorem_l2813_281380


namespace double_price_increase_l2813_281333

theorem double_price_increase (original_price : ℝ) (increase_percentage : ℝ) :
  let first_increase := original_price * (1 + increase_percentage / 100)
  let second_increase := first_increase * (1 + increase_percentage / 100)
  increase_percentage = 15 →
  second_increase = original_price * (1 + 32.25 / 100) :=
by sorry

end double_price_increase_l2813_281333


namespace solution_set_equality_l2813_281399

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := {x : ℝ | x ≠ 0 ∧ 1 / x < 1 / 2}

-- State the theorem
theorem solution_set_equality : solution_set = Set.Ioi 2 ∪ Set.Iio 0 := by
  sorry

end solution_set_equality_l2813_281399


namespace no_five_solutions_and_divisibility_l2813_281315

theorem no_five_solutions_and_divisibility (k : ℤ) :
  (¬ ∃ (x₁ x₂ x₃ x₄ x₅ y₁ : ℤ),
    y₁^2 - k = x₁^3 ∧
    (y₁ - 1)^2 - k = x₂^3 ∧
    (y₁ - 2)^2 - k = x₃^3 ∧
    (y₁ - 3)^2 - k = x₄^3 ∧
    (y₁ - 4)^2 - k = x₅^3) ∧
  (∀ (x₁ x₂ x₃ x₄ y₁ : ℤ),
    y₁^2 - k = x₁^3 ∧
    (y₁ - 1)^2 - k = x₂^3 ∧
    (y₁ - 2)^2 - k = x₃^3 ∧
    (y₁ - 3)^2 - k = x₄^3 →
    63 ∣ (k - 17)) :=
by sorry

end no_five_solutions_and_divisibility_l2813_281315


namespace quadratic_function_and_area_l2813_281387

-- Define the quadratic function f
def f : ℝ → ℝ := fun x ↦ x^2 + 2*x + 1

-- Theorem statement
theorem quadratic_function_and_area :
  (∀ x, (deriv f) x = 2*x + 2) ∧ 
  (∃! x, f x = 0) ∧
  (∫ x in (-3)..0, ((-x^2 - 4*x + 1) - f x)) = 9 := by sorry

end quadratic_function_and_area_l2813_281387


namespace polynomial_simplification_l2813_281304

/-- Given two polynomials in p, prove that their difference simplifies to the given result. -/
theorem polynomial_simplification (p : ℝ) :
  (2 * p^4 - 3 * p^3 + 7 * p - 4) - (-6 * p^3 - 5 * p^2 + 4 * p + 3) =
  2 * p^4 + 3 * p^3 + 5 * p^2 + 3 * p - 7 := by
  sorry

end polynomial_simplification_l2813_281304


namespace triangle_inequality_two_points_l2813_281370

/-- Triangle inequality for two points in the plane of a triangle -/
theorem triangle_inequality_two_points (A B C P₁ P₂ : ℝ × ℝ) 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (a₁ : ℝ) (b₁ : ℝ) (c₁ : ℝ) 
  (a₂ : ℝ) (b₂ : ℝ) (c₂ : ℝ) 
  (ha : a = dist B C) 
  (hb : b = dist A C) 
  (hc : c = dist A B) 
  (ha₁ : a₁ = dist P₁ A) 
  (hb₁ : b₁ = dist P₁ B) 
  (hc₁ : c₁ = dist P₁ C) 
  (ha₂ : a₂ = dist P₂ A) 
  (hb₂ : b₂ = dist P₂ B) 
  (hc₂ : c₂ = dist P₂ C) : 
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c :=
sorry

#check triangle_inequality_two_points

end triangle_inequality_two_points_l2813_281370


namespace average_y_value_l2813_281375

def linear_regression (x : ℝ) : ℝ := 1.5 * x + 45

def x_values : List ℝ := [1, 7, 5, 13, 19]

theorem average_y_value (x_avg : ℝ) (h : x_avg = (List.sum x_values) / (List.length x_values)) :
  linear_regression x_avg = 58.5 := by
  sorry

end average_y_value_l2813_281375


namespace share_calculation_l2813_281355

theorem share_calculation (total_amount : ℕ) (ratio_parts : List ℕ) : 
  total_amount = 4800 → 
  ratio_parts = [2, 4, 6] → 
  (total_amount / (ratio_parts.sum)) * (ratio_parts.head!) = 800 := by
  sorry

end share_calculation_l2813_281355


namespace product_of_multiples_of_three_l2813_281378

theorem product_of_multiples_of_three : ∃ (a b : ℕ), 
  a = 22 * 3 ∧ 
  b = 23 * 3 ∧ 
  a < 100 ∧ 
  b < 100 ∧ 
  a * b = 4554 := by
  sorry

end product_of_multiples_of_three_l2813_281378


namespace cricket_innings_problem_l2813_281357

theorem cricket_innings_problem (initial_average : ℝ) (next_innings_runs : ℝ) (average_increase : ℝ) :
  initial_average = 32 →
  next_innings_runs = 137 →
  average_increase = 5 →
  ∃ n : ℕ, (n : ℝ) * initial_average + next_innings_runs = (n + 1 : ℝ) * (initial_average + average_increase) ∧ n = 20 :=
by sorry

end cricket_innings_problem_l2813_281357


namespace one_pair_probability_l2813_281312

/-- Represents the number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- Represents the number of socks to be selected -/
def socks_selected : ℕ := 4

/-- Calculates the probability of selecting exactly one pair of socks of the same color -/
def prob_one_pair : ℚ := 4 / 7

/-- Proves that the probability of selecting exactly one pair of socks of the same color
    when randomly choosing 4 socks from a set of 10 socks (2 of each of 5 colors) is 4/7 -/
theorem one_pair_probability : 
  total_socks = num_colors * socks_per_color ∧ 
  socks_selected = 4 → 
  prob_one_pair = 4 / 7 := by
  sorry

end one_pair_probability_l2813_281312


namespace sin_cos_difference_equals_half_l2813_281308

theorem sin_cos_difference_equals_half : 
  Real.sin (137 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (13 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_difference_equals_half_l2813_281308


namespace floor_tiling_l2813_281352

theorem floor_tiling (n : ℕ) (h1 : 10 < n) (h2 : n < 20) :
  (∃ x : ℕ, n^2 = 9*x) ↔ n = 12 ∨ n = 15 ∨ n = 18 := by
  sorry

end floor_tiling_l2813_281352


namespace cube_construction_count_l2813_281321

/-- The rotational symmetry group of a cube -/
def CubeRotationGroup : Type := Unit

/-- The order of the rotational symmetry group of a cube -/
def cubeRotationGroupOrder : ℕ := 26

/-- The number of ways to choose 13 items from 27 items -/
def chooseThirteenFromTwentySeven : ℕ := 2333606

/-- The number of configurations fixed by the identity rotation -/
def fixedByIdentity : ℕ := chooseThirteenFromTwentySeven

/-- The number of configurations fixed by rotations around centroids of faces -/
def fixedByCentroidRotations : ℕ := 2

/-- The number of configurations fixed by rotations around vertices and edges -/
def fixedByVertexEdgeRotations : ℕ := 1

/-- The total number of fixed configurations -/
def totalFixedConfigurations : ℕ := fixedByIdentity + fixedByCentroidRotations + fixedByVertexEdgeRotations

/-- The number of rotationally distinct ways to construct the cube -/
def distinctConstructions : ℕ := totalFixedConfigurations / cubeRotationGroupOrder

theorem cube_construction_count :
  distinctConstructions = 89754 :=
sorry

end cube_construction_count_l2813_281321


namespace mary_earnings_per_home_l2813_281309

/-- Mary's earnings per home, given total earnings and number of homes cleaned -/
def earnings_per_home (total_earnings : ℕ) (homes_cleaned : ℕ) : ℕ :=
  total_earnings / homes_cleaned

/-- Proof that Mary earns $46 per home -/
theorem mary_earnings_per_home :
  earnings_per_home 276 6 = 46 := by
  sorry

end mary_earnings_per_home_l2813_281309


namespace no_solution_arctan_equation_l2813_281358

theorem no_solution_arctan_equation :
  ¬ ∃ (x : ℝ), x > 0 ∧ Real.arctan (1 / x^2) + Real.arctan (1 / x^4) = π / 4 := by
sorry

end no_solution_arctan_equation_l2813_281358


namespace renata_final_balance_l2813_281385

/-- Represents Renata's financial transactions throughout the day -/
def renata_transactions : ℤ → ℤ
| 0 => 10                  -- Initial amount
| 1 => -4                  -- Charity ticket donation
| 2 => 90                  -- Charity draw winnings
| 3 => -50                 -- First slot machine loss
| 4 => -10                 -- Second slot machine loss
| 5 => -5                  -- Third slot machine loss
| 6 => -1                  -- Water bottle purchase
| 7 => -1                  -- Lottery ticket purchase
| 8 => 65                  -- Lottery winnings
| _ => 0                   -- No more transactions

/-- The final balance after all transactions -/
def final_balance : ℤ := (List.range 9).foldl (· + renata_transactions ·) 0

/-- Theorem stating that Renata's final balance is $94 -/
theorem renata_final_balance : final_balance = 94 := by
  sorry

end renata_final_balance_l2813_281385


namespace subtraction_of_decimals_l2813_281336

theorem subtraction_of_decimals : 7.25 - 3.1 - 1.05 = 3.10 := by
  sorry

end subtraction_of_decimals_l2813_281336


namespace best_shooter_D_l2813_281386

structure Shooter where
  name : String
  average_score : ℝ
  variance : ℝ

def is_best_shooter (s : Shooter) (shooters : List Shooter) : Prop :=
  (∀ t ∈ shooters, s.average_score ≥ t.average_score) ∧
  (∀ t ∈ shooters, s.average_score = t.average_score → s.variance ≤ t.variance)

theorem best_shooter_D :
  let shooters := [
    ⟨"A", 9, 1.2⟩,
    ⟨"B", 8, 0.4⟩,
    ⟨"C", 9, 1.8⟩,
    ⟨"D", 9, 0.4⟩
  ]
  let D := ⟨"D", 9, 0.4⟩
  is_best_shooter D shooters := by
  sorry

#check best_shooter_D

end best_shooter_D_l2813_281386


namespace sawing_time_l2813_281322

/-- Given that sawing a steel bar into 2 pieces takes 2 minutes,
    this theorem proves that sawing the same bar into 6 pieces takes 10 minutes. -/
theorem sawing_time (time_for_two_pieces : ℕ) (pieces : ℕ) : 
  time_for_two_pieces = 2 → pieces = 6 → (pieces - 1) * (time_for_two_pieces / (2 - 1)) = 10 := by
  sorry

end sawing_time_l2813_281322


namespace range_of_a_l2813_281350

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → x^2 + a*x + 3 ≥ a) → 
  a ∈ Set.Icc (-7) 2 := by
sorry

end range_of_a_l2813_281350


namespace x2y_plus_1_is_third_degree_binomial_l2813_281301

/-- A binomial is a polynomial with exactly two terms. -/
def is_binomial (p : Polynomial ℝ) : Prop :=
  p.support.card = 2

/-- The degree of a polynomial is the highest degree of any of its terms. -/
def polynomial_degree (p : Polynomial ℝ) : ℕ := p.natDegree

/-- A third-degree polynomial has a degree of 3. -/
def is_third_degree (p : Polynomial ℝ) : Prop :=
  polynomial_degree p = 3

theorem x2y_plus_1_is_third_degree_binomial :
  let p : Polynomial ℝ := X^2 * Y + 1
  is_binomial p ∧ is_third_degree p :=
sorry

end x2y_plus_1_is_third_degree_binomial_l2813_281301


namespace equal_rectangle_count_l2813_281376

def count_rectangles (perimeter : ℕ) : ℕ :=
  (perimeter / 2 - 1) / 2

theorem equal_rectangle_count :
  count_rectangles 1996 = count_rectangles 1998 :=
by sorry

end equal_rectangle_count_l2813_281376


namespace part_one_part_two_l2813_281323

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def C (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

-- Part I: Prove that if A = B, then a = 2
theorem part_one : A 2 = B :=
sorry

-- Part II: Prove that if B ∪ C = B, then m ∈ {-1/4, 0, 1/2}
theorem part_two (m : ℝ) : B ∪ C m = B → m ∈ ({-1/4, 0, 1/2} : Set ℝ) :=
sorry

end part_one_part_two_l2813_281323


namespace procedure_arrangement_count_l2813_281391

/-- The number of ways to arrange 6 procedures with specific constraints -/
def arrangement_count : ℕ := 96

/-- The number of procedures -/
def total_procedures : ℕ := 6

/-- The number of ways to place procedure A (first or last) -/
def a_placements : ℕ := 2

/-- The number of ways to arrange B and C within their unit -/
def bc_arrangements : ℕ := 2

/-- The number of elements to arrange (BC unit + 3 other procedures) -/
def elements_to_arrange : ℕ := 4

theorem procedure_arrangement_count :
  arrangement_count = 
    a_placements * elements_to_arrange.factorial * bc_arrangements :=
by sorry

end procedure_arrangement_count_l2813_281391


namespace maria_flour_calculation_l2813_281363

/-- The amount of flour needed for a given number of cookies -/
def flour_needed (cookies : ℕ) : ℚ :=
  (3 : ℚ) * cookies / 40

theorem maria_flour_calculation :
  flour_needed 120 = 9 := by sorry

end maria_flour_calculation_l2813_281363


namespace fraction_comparison_l2813_281382

theorem fraction_comparison (x : ℝ) : 
  x > 3/4 → x ≠ 3 → (9 - 3*x ≠ 0) → (5*x + 3 > 9 - 3*x) :=
by sorry

end fraction_comparison_l2813_281382


namespace problem_solution_l2813_281359

/-- Equation I: 2x + y + z = 47, where x, y, z are positive integers -/
def equation_I (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 2 * x + y + z = 47

/-- Equation II: 2x + y + z + w = 47, where x, y, z, w are positive integers -/
def equation_II (x y z w : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ 2 * x + y + z + w = 47

/-- Consecutive integers -/
def consecutive (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = a + 2

/-- Four consecutive integers -/
def consecutive_four (a b c d : ℕ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3

/-- Consecutive even integers -/
def consecutive_even (a b c : ℕ) : Prop :=
  ∃ k : ℕ, a = 2 * k ∧ b = 2 * (k + 1) ∧ c = 2 * (k + 2)

/-- Four consecutive even integers -/
def consecutive_even_four (a b c d : ℕ) : Prop :=
  ∃ k : ℕ, a = 2 * k ∧ b = 2 * (k + 1) ∧ c = 2 * (k + 2) ∧ d = 2 * (k + 3)

/-- Four consecutive odd integers -/
def consecutive_odd_four (a b c d : ℕ) : Prop :=
  ∃ k : ℕ, a = 2 * k + 1 ∧ b = 2 * k + 3 ∧ c = 2 * k + 5 ∧ d = 2 * k + 7

theorem problem_solution :
  (∃ x y z : ℕ, equation_I x y z ∧ consecutive x y z) ∧
  (∃ x y z w : ℕ, equation_II x y z w ∧ consecutive_four x y z w) ∧
  (¬ ∃ x y z : ℕ, equation_I x y z ∧ consecutive_even x y z) ∧
  (¬ ∃ x y z w : ℕ, equation_II x y z w ∧ consecutive_even_four x y z w) ∧
  (¬ ∃ x y z w : ℕ, equation_II x y z w ∧ consecutive_odd_four x y z w) :=
by sorry

end problem_solution_l2813_281359


namespace four_plus_five_result_l2813_281324

/-- Define the sequence operation for two consecutive integers -/
def seqOperation (a b : ℕ) : ℕ := (a + b)^2 + 1

/-- Theorem stating that 4 + 5 results in 82 in the given sequence -/
theorem four_plus_five_result :
  seqOperation 4 5 = 82 :=
by sorry

end four_plus_five_result_l2813_281324


namespace integer_root_count_l2813_281364

theorem integer_root_count : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, ∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) ∧ S.card = 12 :=
sorry

end integer_root_count_l2813_281364


namespace polynomial_with_triple_roots_l2813_281388

def p : ℝ → ℝ := fun x ↦ 12 * x^5 - 30 * x^4 + 20 * x^3 - 1

theorem polynomial_with_triple_roots :
  (∀ x : ℝ, (∃ q : ℝ → ℝ, p x + 1 = x^3 * q x)) ∧
  (∀ x : ℝ, (∃ r : ℝ → ℝ, p x - 1 = (x - 1)^3 * r x)) →
  ∀ x : ℝ, p x = 12 * x^5 - 30 * x^4 + 20 * x^3 - 1 :=
by sorry

end polynomial_with_triple_roots_l2813_281388


namespace complement_intersection_theorem_l2813_281353

universe u

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {1, 2, 5}

theorem complement_intersection_theorem : (U \ A) ∩ B = {1} := by
  sorry

end complement_intersection_theorem_l2813_281353


namespace sqrt_difference_approximation_l2813_281348

theorem sqrt_difference_approximation : |Real.sqrt 144 - Real.sqrt 140 - 0.17| < 0.01 := by
  sorry

end sqrt_difference_approximation_l2813_281348


namespace sam_container_capacity_l2813_281340

/-- Represents a rectangular container with dimensions and marble capacity. -/
structure Container where
  length : ℝ
  width : ℝ
  height : ℝ
  capacity : ℕ

/-- Calculates the volume of a container. -/
def containerVolume (c : Container) : ℝ :=
  c.length * c.width * c.height

/-- Theorem: Given Ellie's container dimensions and capacity, and the relative dimensions
    of Sam's container, Sam's container holds 1200 marbles. -/
theorem sam_container_capacity
  (ellie : Container)
  (h_ellie_dims : ellie.length = 2 ∧ ellie.width = 3 ∧ ellie.height = 4)
  (h_ellie_capacity : ellie.capacity = 200)
  (sam : Container)
  (h_sam_dims : sam.length = ellie.length ∧ 
                sam.width = 2 * ellie.width ∧ 
                sam.height = 3 * ellie.height) :
  sam.capacity = 1200 := by
sorry


end sam_container_capacity_l2813_281340


namespace divisibility_condition_solutions_l2813_281398

theorem divisibility_condition_solutions (n p : ℕ) (h_prime : Nat.Prime p) (h_range : 0 < n ∧ n ≤ 2 * p) :
  n^(p-1) ∣ (p-1)^n + 1 ↔ 
    (n = 1 ∧ p ≥ 2) ∨
    (n = 2 ∧ p = 2) ∨
    (n = 3 ∧ p = 3) :=
by sorry

end divisibility_condition_solutions_l2813_281398


namespace worker_production_equations_l2813_281351

/-- Represents the daily production of workers in a company -/
structure WorkerProduction where
  novice : ℕ
  experienced : ℕ

/-- The conditions of the worker production problem -/
class WorkerProductionProblem (w : WorkerProduction) where
  experience_difference : w.experienced - w.novice = 30
  total_production : w.novice + 2 * w.experienced = 180

/-- The theorem stating the correct system of equations for the worker production problem -/
theorem worker_production_equations (w : WorkerProduction) [WorkerProductionProblem w] :
  (w.experienced - w.novice = 30) ∧ (w.novice + 2 * w.experienced = 180) := by
  sorry

end worker_production_equations_l2813_281351


namespace time_to_paint_one_house_l2813_281377

/-- Given that 9 houses can be painted in 3 hours, prove that one house can be painted in 20 minutes. -/
theorem time_to_paint_one_house : 
  ∀ (total_houses : ℕ) (total_hours : ℕ) (minutes_per_hour : ℕ),
  total_houses = 9 →
  total_hours = 3 →
  minutes_per_hour = 60 →
  (total_hours * minutes_per_hour) / total_houses = 20 := by
  sorry

end time_to_paint_one_house_l2813_281377


namespace max_games_512_3_l2813_281328

/-- Represents a tournament where players must be defeated three times to be eliminated -/
structure Tournament where
  contestants : ℕ
  defeats_to_eliminate : ℕ

/-- Calculates the maximum number of games that could be played in the tournament -/
def max_games (t : Tournament) : ℕ :=
  (t.contestants - 1) * t.defeats_to_eliminate + 2

/-- Theorem stating that for a tournament with 512 contestants and 3 defeats to eliminate,
    the maximum number of games is 1535 -/
theorem max_games_512_3 :
  let t : Tournament := { contestants := 512, defeats_to_eliminate := 3 }
  max_games t = 1535 := by
  sorry

end max_games_512_3_l2813_281328


namespace greatest_integer_with_gcf_five_l2813_281362

theorem greatest_integer_with_gcf_five : ∃ n : ℕ, n < 200 ∧ Nat.gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 → Nat.gcd m 30 = 5 → m ≤ n :=
by
  -- Proof goes here
  sorry

end greatest_integer_with_gcf_five_l2813_281362


namespace line_slope_and_intercept_l2813_281327

/-- Given a line expressed as (3, -4) · ((x, y) - (-2, 8)) = 0, 
    prove that its slope is 3/4 and its y-intercept is 9.5 -/
theorem line_slope_and_intercept :
  let line := fun (x y : ℝ) => 3 * (x + 2) + (-4) * (y - 8) = 0
  ∃ (m b : ℝ), m = 3/4 ∧ b = 9.5 ∧ ∀ x y, line x y ↔ y = m * x + b :=
by sorry

end line_slope_and_intercept_l2813_281327


namespace shooting_training_equivalence_l2813_281334

-- Define the propositions
variable (p q : Prop)

-- Define "both shots hit the target"
def both_hit (p q : Prop) : Prop := p ∧ q

-- Define "exactly one shot hits the target"
def exactly_one_hit (p q : Prop) : Prop := (p ∧ ¬q) ∨ (¬p ∧ q)

-- Theorem stating the equivalence
theorem shooting_training_equivalence :
  (both_hit p q ↔ p ∧ q) ∧
  (exactly_one_hit p q ↔ (p ∧ ¬q) ∨ (¬p ∧ q)) :=
by sorry

end shooting_training_equivalence_l2813_281334


namespace triangle_perimeter_l2813_281310

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a + b + c = 6 := by
  sorry

end triangle_perimeter_l2813_281310


namespace electronic_items_loss_percentage_l2813_281325

/-- Calculate the overall loss percentage for three electronic items -/
theorem electronic_items_loss_percentage :
  let cost_prices : List ℚ := [1500, 2500, 800]
  let sale_prices : List ℚ := [1275, 2300, 700]
  let total_cost := cost_prices.sum
  let total_sale := sale_prices.sum
  let loss := total_cost - total_sale
  let loss_percentage := (loss / total_cost) * 100
  loss_percentage = 10.9375 := by
  sorry

end electronic_items_loss_percentage_l2813_281325


namespace closest_angles_to_2013_l2813_281331

theorem closest_angles_to_2013 (x : ℝ) :
  (2^(Real.sin x)^2 + 2^(Real.cos x)^2 = 2 * Real.sqrt 2) →
  (x = 1935 * π / 180 ∨ x = 2025 * π / 180) ∧
  ∀ y : ℝ, (2^(Real.sin y)^2 + 2^(Real.cos y)^2 = 2 * Real.sqrt 2) →
    (1935 * π / 180 < y ∧ y < 2025 * π / 180) →
    (y ≠ 1935 * π / 180 ∧ y ≠ 2025 * π / 180) →
    ¬(∃ n : ℤ, y = n * π / 180) :=
by sorry

end closest_angles_to_2013_l2813_281331


namespace min_side_triangle_l2813_281330

theorem min_side_triangle (S γ : ℝ) (hS : S > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (1/2 * a * b * Real.sin γ = S) ∧
  (∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 →
    1/2 * a' * b' * Real.sin γ = S →
    c' ≥ 2 * Real.sqrt (S * Real.tan (γ/2))) :=
sorry

end min_side_triangle_l2813_281330


namespace three_digit_cube_divisible_by_eight_l2813_281392

theorem three_digit_cube_divisible_by_eight :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3 ∧ 8 ∣ n := by sorry

end three_digit_cube_divisible_by_eight_l2813_281392


namespace total_cost_calculation_l2813_281372

-- Define the prices and quantities
def shirt_price : ℝ := 15
def shirt_quantity : ℕ := 4
def pants_price : ℝ := 40
def pants_quantity : ℕ := 2
def suit_price : ℝ := 150
def suit_quantity : ℕ := 1
def sweater_price : ℝ := 30
def sweater_quantity : ℕ := 2
def tie_price : ℝ := 20
def tie_quantity : ℕ := 3
def shoes_price : ℝ := 80
def shoes_quantity : ℕ := 1

-- Define the discounts
def shirt_discount : ℝ := 0.2
def pants_discount : ℝ := 0.3
def tie_discount : ℝ := 0.5
def shoes_discount : ℝ := 0.25
def coupon_discount : ℝ := 0.1

-- Define reward points
def reward_points : ℕ := 500
def reward_point_value : ℝ := 0.05

-- Define sales tax
def sales_tax_rate : ℝ := 0.05

-- Define the theorem
theorem total_cost_calculation :
  let shirt_total := shirt_price * shirt_quantity * (1 - shirt_discount)
  let pants_total := pants_price * pants_quantity * (1 - pants_discount)
  let suit_total := suit_price * suit_quantity
  let sweater_total := sweater_price * sweater_quantity
  let tie_total := tie_price * tie_quantity - tie_price * tie_discount
  let shoes_total := shoes_price * shoes_quantity * (1 - shoes_discount)
  let subtotal := shirt_total + pants_total + suit_total + sweater_total + tie_total + shoes_total
  let after_coupon := subtotal * (1 - coupon_discount)
  let after_rewards := after_coupon - (reward_points * reward_point_value)
  let final_total := after_rewards * (1 + sales_tax_rate)
  final_total = 374.43 := by sorry

end total_cost_calculation_l2813_281372


namespace initial_saree_purchase_l2813_281394

/-- The number of sarees in the initial purchase -/
def num_sarees : ℕ := 2

/-- The price of one saree -/
def saree_price : ℕ := 400

/-- The price of one shirt -/
def shirt_price : ℕ := 200

/-- Theorem stating that the number of sarees in the initial purchase is 2 -/
theorem initial_saree_purchase : 
  (∃ (X : ℕ), X * saree_price + 4 * shirt_price = 1600) ∧ 
  (saree_price + 6 * shirt_price = 1600) ∧
  (12 * shirt_price = 2400) →
  num_sarees = 2 := by
  sorry

end initial_saree_purchase_l2813_281394


namespace prime_sum_probability_l2813_281374

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Bool := sorry

/-- The number of dice being rolled -/
def numDice : ℕ := 7

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling numDice dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of outcomes that result in a prime sum -/
def primeOutcomes : ℕ := 80425

/-- The probability of rolling numDice dice and obtaining a prime sum -/
def primeProbability : ℚ := primeOutcomes / totalOutcomes

theorem prime_sum_probability :
  primeProbability = 26875 / 93312 := by sorry

end prime_sum_probability_l2813_281374


namespace fraction_power_four_l2813_281318

theorem fraction_power_four : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end fraction_power_four_l2813_281318


namespace age_sum_is_21_l2813_281326

/-- Given two people p and q, where 6 years ago p was half the age of q,
    and the ratio of their present ages is 3:4, prove that the sum of
    their present ages is 21 years. -/
theorem age_sum_is_21 (p q : ℕ) : 
  (p - 6 = (q - 6) / 2) →  -- 6 years ago, p was half of q in age
  (p : ℚ) / q = 3 / 4 →    -- The ratio of their present ages is 3:4
  p + q = 21 :=            -- The sum of their present ages is 21
by sorry

end age_sum_is_21_l2813_281326


namespace proportional_increase_l2813_281319

theorem proportional_increase (x y : ℝ) (c : ℝ) (h1 : y = c * x) :
  let x' := 1.3 * x
  let y' := c * x'
  y' = 2.6 * y →
  (y' - y) / y = 1.6 := by
sorry

end proportional_increase_l2813_281319


namespace proposition_d_is_false_l2813_281390

/-- Proposition D is false: There exist four mutually different non-zero vectors on a plane 
    such that the sum vector of any two vectors is perpendicular to the sum vector of 
    the remaining two vectors. -/
theorem proposition_d_is_false :
  ∃ (a b c d : ℝ × ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a.1 + b.1) * (c.1 + d.1) + (a.2 + b.2) * (c.2 + d.2) = 0 ∧
    (a.1 + c.1) * (b.1 + d.1) + (a.2 + c.2) * (b.2 + d.2) = 0 ∧
    (a.1 + d.1) * (b.1 + c.1) + (a.2 + d.2) * (b.2 + c.2) = 0 :=
by
  sorry


end proposition_d_is_false_l2813_281390


namespace triangle_side_length_l2813_281368

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  2 * b = a + c →  -- arithmetic sequence condition
  B = π / 6 →  -- 30° in radians
  (1 / 2) * a * c * Real.sin B = 3 / 2 →  -- area condition
  b = 1 + Real.sqrt 3 := by
  sorry

end triangle_side_length_l2813_281368


namespace kim_cousins_count_l2813_281306

theorem kim_cousins_count (gum_per_cousin : ℕ) (total_gum : ℕ) (h1 : gum_per_cousin = 5) (h2 : total_gum = 20) :
  total_gum / gum_per_cousin = 4 := by
  sorry

end kim_cousins_count_l2813_281306


namespace total_big_cats_l2813_281344

def feline_sanctuary (lions tigers : ℕ) : ℕ :=
  let cougars := (lions + tigers) / 2
  lions + tigers + cougars

theorem total_big_cats :
  feline_sanctuary 12 14 = 39 := by
  sorry

end total_big_cats_l2813_281344


namespace ab_plus_cd_value_l2813_281307

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -3)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := by
  sorry

end ab_plus_cd_value_l2813_281307


namespace sequence_properties_l2813_281316

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (ha_cond : 2 * a 5 - a 3 = 3)
  (hb_2 : b 2 = 1)
  (hb_4 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (∃ q : ℝ, (q = 2 ∨ q = -2) ∧ ∀ n : ℕ, b (n + 1) = b n * q) :=
by sorry

end sequence_properties_l2813_281316


namespace square_sum_zero_implies_both_zero_l2813_281365

theorem square_sum_zero_implies_both_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l2813_281365


namespace digit_207_is_8_l2813_281349

/-- The decimal representation of 3/7 as a sequence of digits -/
def decimal_rep_3_7 : ℕ → Fin 10
  | n => sorry

/-- The length of the repeating sequence in the decimal representation of 3/7 -/
def repeat_length : ℕ := 6

/-- The 207th digit beyond the decimal point in the decimal representation of 3/7 -/
def digit_207 : Fin 10 := decimal_rep_3_7 206

theorem digit_207_is_8 : digit_207 = 8 := by sorry

end digit_207_is_8_l2813_281349


namespace intersection_and_range_l2813_281371

def A : Set ℝ := {x | x^2 + 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + 2*a^2 - 2 = 0}

theorem intersection_and_range :
  (A ∩ B 1 = {-4}) ∧
  (∀ a : ℝ, A ∩ B a = B a ↔ a < -1 ∨ a > 3) := by
  sorry

end intersection_and_range_l2813_281371


namespace woods_area_calculation_l2813_281373

/-- The area of rectangular woods -/
def woods_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The area of woods with width 8 miles and length 3 miles is 24 square miles -/
theorem woods_area_calculation :
  woods_area 8 3 = 24 := by
  sorry

end woods_area_calculation_l2813_281373


namespace pentagonal_prism_sum_l2813_281361

/-- Definition of a pentagonal prism -/
structure PentagonalPrism where
  bases : ℕ := 2
  connecting_faces : ℕ := 5
  edges_per_base : ℕ := 5
  vertices_per_base : ℕ := 5

/-- Theorem: The sum of faces, edges, and vertices of a pentagonal prism is 32 -/
theorem pentagonal_prism_sum (p : PentagonalPrism) : 
  (p.bases + p.connecting_faces) + 
  (p.edges_per_base * 2 + p.edges_per_base) + 
  (p.vertices_per_base * 2) = 32 := by
  sorry

#check pentagonal_prism_sum

end pentagonal_prism_sum_l2813_281361


namespace bob_local_tax_cents_l2813_281300

/-- Bob's hourly wage in dollars -/
def bob_hourly_wage : ℝ := 25

/-- Local tax rate as a decimal -/
def local_tax_rate : ℝ := 0.025

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℝ := 100

/-- Theorem: The amount of Bob's hourly wage used for local taxes is 62.5 cents -/
theorem bob_local_tax_cents : 
  bob_hourly_wage * local_tax_rate * dollars_to_cents = 62.5 := by
  sorry

end bob_local_tax_cents_l2813_281300


namespace triangle_property_l2813_281395

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (Real.sin A + Real.sin B) * (a - b) = (Real.sin C - Real.sin B) * c →
  a = 4 →
  A = π / 3 ∧ (∀ b' c' : ℝ, b' > 0 → c' > 0 → 
    (Real.sin A + Real.sin B) * (a - b') = (Real.sin C - Real.sin B) * c' →
    1/2 * b' * c' * Real.sin A ≤ 4 * Real.sqrt 3) :=
by sorry

end triangle_property_l2813_281395


namespace inequality_proof_l2813_281360

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l2813_281360


namespace shaded_area_squares_l2813_281396

theorem shaded_area_squares (large_side small_side : ℝ) 
  (h1 : large_side = 14) 
  (h2 : small_side = 10) : 
  (large_side^2 - small_side^2) = 49 := by
  sorry

end shaded_area_squares_l2813_281396


namespace library_visits_l2813_281397

/-- Proves that William goes to the library 2 times per week given the conditions -/
theorem library_visits (jason_freq : ℕ) (william_freq : ℕ) (jason_total : ℕ) (weeks : ℕ) :
  jason_freq = 4 * william_freq →
  jason_total = 32 →
  weeks = 4 →
  jason_total = jason_freq * weeks →
  william_freq = 2 := by
  sorry

end library_visits_l2813_281397


namespace ice_cream_scoops_left_l2813_281384

/-- Represents the flavors of ice cream --/
inductive Flavor
  | Chocolate
  | Strawberry
  | Vanilla

/-- Represents a person --/
inductive Person
  | Ethan
  | Lucas
  | Danny
  | Connor
  | Olivia
  | Shannon

/-- The number of scoops in each carton --/
def scoops_per_carton : ℕ := 10

/-- The initial number of scoops for each flavor --/
def initial_scoops (f : Flavor) : ℕ := scoops_per_carton

/-- The number of scoops a person wants for each flavor --/
def scoops_wanted (p : Person) (f : Flavor) : ℕ :=
  match p, f with
  | Person.Ethan, Flavor.Chocolate => 1
  | Person.Ethan, Flavor.Vanilla => 1
  | Person.Lucas, Flavor.Chocolate => 2
  | Person.Danny, Flavor.Chocolate => 2
  | Person.Connor, Flavor.Chocolate => 2
  | Person.Olivia, Flavor.Strawberry => 1
  | Person.Olivia, Flavor.Vanilla => 1
  | Person.Shannon, Flavor.Strawberry => 2
  | Person.Shannon, Flavor.Vanilla => 2
  | _, _ => 0

/-- The total number of scoops taken for each flavor --/
def total_scoops_taken (f : Flavor) : ℕ :=
  (scoops_wanted Person.Ethan f) +
  (scoops_wanted Person.Lucas f) +
  (scoops_wanted Person.Danny f) +
  (scoops_wanted Person.Connor f) +
  (scoops_wanted Person.Olivia f) +
  (scoops_wanted Person.Shannon f)

/-- The number of scoops left for each flavor --/
def scoops_left (f : Flavor) : ℕ :=
  initial_scoops f - total_scoops_taken f

/-- The total number of scoops left --/
def total_scoops_left : ℕ :=
  (scoops_left Flavor.Chocolate) +
  (scoops_left Flavor.Strawberry) +
  (scoops_left Flavor.Vanilla)

theorem ice_cream_scoops_left : total_scoops_left = 16 := by
  sorry

end ice_cream_scoops_left_l2813_281384


namespace art_show_pricing_l2813_281313

/-- The price of a large painting that satisfies the given conditions -/
def large_painting_price : ℕ → ℕ → ℕ → ℕ → ℕ := λ small_price large_count small_count total_earnings =>
  (total_earnings - small_price * small_count) / large_count

theorem art_show_pricing (small_price large_count small_count total_earnings : ℕ) 
  (h1 : small_price = 80)
  (h2 : large_count = 5)
  (h3 : small_count = 8)
  (h4 : total_earnings = 1140) :
  large_painting_price small_price large_count small_count total_earnings = 100 := by
sorry

#eval large_painting_price 80 5 8 1140

end art_show_pricing_l2813_281313


namespace hyperbola_eccentricity_l2813_281393

/-- Hyperbola with foci and a special point -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ  -- Left focus
  F₂ : ℝ × ℝ  -- Right focus
  P : ℝ × ℝ   -- Special point on the right branch
  h₁ : a > b
  h₂ : b > 0
  h₃ : F₁.1 < 0 ∧ F₁.2 = 0  -- Left focus on negative x-axis
  h₄ : F₂.1 > 0 ∧ F₂.2 = 0  -- Right focus on positive x-axis
  h₅ : P.1 > 0  -- P is on the right branch
  h₆ : P.1^2 / a^2 - P.2^2 / b^2 = 1  -- P satisfies hyperbola equation
  h₇ : (P.1 + F₂.1) * (P.1 - F₂.1) + P.2 * P.2 = 0  -- Dot product condition
  h₈ : (P.1 - F₁.1)^2 + P.2^2 = 4 * ((P.1 - F₂.1)^2 + P.2^2)  -- Distance condition

/-- The eccentricity of a hyperbola with the given properties is √5 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  Real.sqrt ((h.F₂.1 - h.F₁.1)^2 / (4 * h.a^2)) = Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_l2813_281393


namespace jaydee_typing_time_l2813_281338

/-- Calculates the time needed to type a research paper given specific conditions. -/
def time_to_type_paper (words_per_minute : ℕ) (break_interval : ℕ) (break_duration : ℕ) 
  (words_per_mistake : ℕ) (mistake_correction_time : ℕ) (total_words : ℕ) : ℕ :=
  let typing_time := (total_words + words_per_minute - 1) / words_per_minute
  let breaks := typing_time / break_interval
  let break_time := breaks * break_duration
  let mistakes := (total_words + words_per_mistake - 1) / words_per_mistake
  let correction_time := mistakes * mistake_correction_time
  let total_minutes := typing_time + break_time + correction_time
  (total_minutes + 59) / 60

/-- Theorem stating that Jaydee will take 6 hours to type the research paper. -/
theorem jaydee_typing_time : 
  time_to_type_paper 32 25 5 100 1 7125 = 6 := by
  sorry

end jaydee_typing_time_l2813_281338


namespace perfect_square_condition_l2813_281381

theorem perfect_square_condition (n : ℕ) : 
  ∃ k : ℕ, n^2 + 3*n = k^2 ↔ n = 1 := by sorry

end perfect_square_condition_l2813_281381


namespace dice_roll_probability_l2813_281341

def roll_probability : ℚ := 1 / 12

theorem dice_roll_probability :
  (probability_first_die_three * probability_second_die_odd = roll_probability) :=
by
  sorry

where
  probability_first_die_three : ℚ := 1 / 6
  probability_second_die_odd : ℚ := 1 / 2

end dice_roll_probability_l2813_281341


namespace s_99_digits_l2813_281343

/-- s(n) is the n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- count_digits n returns the number of digits in the natural number n -/
def count_digits (n : ℕ) : ℕ := sorry

/-- The theorem states that s(99) has 189 digits -/
theorem s_99_digits : count_digits (s 99) = 189 := by sorry

end s_99_digits_l2813_281343


namespace function_characterization_l2813_281345

-- Define the property that f must satisfy
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f y + f (x + y) ≥ (y + 1) * f x + f y

-- Theorem statement
theorem function_characterization (f : ℝ → ℝ) 
  (h : SatisfiesInequality f) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x := by
  sorry

end function_characterization_l2813_281345


namespace circle_diameter_endpoint_l2813_281335

/-- Given a circle with center (5, -2) and one endpoint of a diameter at (2, 3),
    prove that the other endpoint of the diameter is at (8, -7). -/
theorem circle_diameter_endpoint (center : ℝ × ℝ) (endpoint1 : ℝ × ℝ) (endpoint2 : ℝ × ℝ) : 
  center = (5, -2) → endpoint1 = (2, 3) → endpoint2 = (8, -7) → 
  (center.1 - endpoint1.1 = endpoint2.1 - center.1 ∧ 
   center.2 - endpoint1.2 = endpoint2.2 - center.2) := by
sorry

end circle_diameter_endpoint_l2813_281335


namespace final_price_approx_l2813_281347

-- Define the initial cost price
def initial_cost : ℝ := 114.94

-- Define the profit percentages
def profit_A : ℝ := 0.35
def profit_B : ℝ := 0.45

-- Define the function to calculate selling price given cost price and profit percentage
def selling_price (cost : ℝ) (profit : ℝ) : ℝ := cost * (1 + profit)

-- Define the final selling price calculation
def final_price : ℝ := selling_price (selling_price initial_cost profit_A) profit_B

-- Theorem to prove
theorem final_price_approx :
  ∃ ε > 0, |final_price - 225| < ε :=
sorry

end final_price_approx_l2813_281347


namespace P_roots_count_l2813_281303

/-- Recursive definition of the polynomial sequence Pₙ(x) -/
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | 1, x => x
  | (n+2), x => x * P (n+1) x - P n x

/-- The number of distinct real roots of Pₙ(x) -/
def num_roots (n : ℕ) : ℕ := n

theorem P_roots_count (n : ℕ) : 
  (∃ (s : Finset ℝ), s.card = num_roots n ∧ 
   (∀ x ∈ s, P n x = 0) ∧
   (∀ x : ℝ, P n x = 0 → x ∈ s)) :=
sorry

end P_roots_count_l2813_281303


namespace triple_q_2000_power_l2813_281305

/-- Sum of digits function -/
def q (n : ℕ) : ℕ :=
  if n < 10 then n else q (n / 10) + n % 10

/-- Theorem: The triple application of q to 2000^2000 results in 4 -/
theorem triple_q_2000_power : q (q (q (2000^2000))) = 4 := by
  sorry

end triple_q_2000_power_l2813_281305


namespace cubic_integer_bound_l2813_281317

theorem cubic_integer_bound (a b c d : ℝ) (ha : a > 4/3) :
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |a * x^3 + b * x^2 + c * x + d| ≤ 1) ∧ Finset.card S ≤ 3 := by
  sorry

end cubic_integer_bound_l2813_281317


namespace two_cubic_feet_equals_3456_cubic_inches_l2813_281366

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Calculates the volume in cubic inches given the volume in cubic feet -/
def cubic_feet_to_cubic_inches (cf : ℚ) : ℚ :=
  cf * feet_to_inches^3

/-- Theorem stating that 2 cubic feet is equal to 3456 cubic inches -/
theorem two_cubic_feet_equals_3456_cubic_inches :
  cubic_feet_to_cubic_inches 2 = 3456 := by
  sorry

end two_cubic_feet_equals_3456_cubic_inches_l2813_281366


namespace shaded_area_proof_l2813_281311

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem shaded_area_proof : U \ (A ∪ B) = {0, 2} := by
  sorry

end shaded_area_proof_l2813_281311


namespace triangle_perimeter_is_six_l2813_281346

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C,
    this theorem proves that under certain conditions, the perimeter is 6. -/
theorem triangle_perimeter_is_six 
  (a b c : ℝ) 
  (A B C : ℝ)
  (h1 : a * Real.cos C + Real.sqrt 3 * a * Real.sin C - b - c = 0)
  (h2 : a = 2)
  (h3 : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  a + b + c = 6 := by
  sorry

end triangle_perimeter_is_six_l2813_281346


namespace smallest_sum_prime_set_l2813_281389

/-- A set of natural numbers uses each digit exactly once -/
def uses_each_digit_once (s : Finset ℕ) : Prop :=
  ∃ (digits : Finset ℕ), digits.card = 10 ∧
    ∀ d ∈ digits, 0 ≤ d ∧ d < 10 ∧
    ∀ n ∈ s, ∀ k, 0 ≤ k ∧ k < 10 → (n / 10^k % 10) ∈ digits

/-- The sum of a set of natural numbers -/
def set_sum (s : Finset ℕ) : ℕ := s.sum id

/-- The theorem to be proved -/
theorem smallest_sum_prime_set :
  ∃ (s : Finset ℕ),
    (∀ n ∈ s, Nat.Prime n) ∧
    uses_each_digit_once s ∧
    set_sum s = 4420 ∧
    (∀ t : Finset ℕ, (∀ n ∈ t, Nat.Prime n) → uses_each_digit_once t → set_sum s ≤ set_sum t) :=
sorry

end smallest_sum_prime_set_l2813_281389


namespace bike_ride_time_l2813_281329

/-- Given a constant speed where 2 miles are covered in 6 minutes,
    prove that the time required to travel 5 miles at the same speed is 15 minutes. -/
theorem bike_ride_time (speed : ℝ) (h1 : speed > 0) (h2 : 2 / speed = 6) : 5 / speed = 15 := by
  sorry

end bike_ride_time_l2813_281329
