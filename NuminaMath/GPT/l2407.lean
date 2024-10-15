import Mathlib

namespace NUMINAMATH_GPT_question1_question2_application_l2407_240771

theorem question1: (-4)^2 - (-3) * (-5) = 1 := by
  sorry

theorem question2 (a : ℝ) (h : a = -4) : a^2 - (a + 1) * (a - 1) = 1 := by
  sorry

theorem application (a : ℝ) (h : a = 1.35) : a * (a - 1) * 2 * a - a^3 - a * (a - 1)^2 = -1.35 := by
  sorry

end NUMINAMATH_GPT_question1_question2_application_l2407_240771


namespace NUMINAMATH_GPT_find_x_in_interval_l2407_240720

noncomputable def a : ℝ := Real.sqrt 2014 - Real.sqrt 2013

theorem find_x_in_interval :
  ∀ x : ℝ, (0 < x) → (x < Real.pi) →
  (a^(Real.tan x ^ 2) + (Real.sqrt 2014 + Real.sqrt 2013)^(-Real.tan x ^ 2) = 2 * a^3) →
  (x = Real.pi / 3 ∨ x = 2 * Real.pi / 3) := by
  -- add proof here
  sorry

end NUMINAMATH_GPT_find_x_in_interval_l2407_240720


namespace NUMINAMATH_GPT_polygon_interior_angles_sum_l2407_240706

theorem polygon_interior_angles_sum {n : ℕ} 
  (h1 : ∀ (k : ℕ), k > 2 → (360 = k * 40)) :
  180 * (9 - 2) = 1260 :=
by
  sorry

end NUMINAMATH_GPT_polygon_interior_angles_sum_l2407_240706


namespace NUMINAMATH_GPT_regular_triangular_pyramid_volume_l2407_240715

theorem regular_triangular_pyramid_volume (a γ : ℝ) : 
  ∃ V, V = (a^3 * Real.sin (γ / 2)^2) / (12 * Real.sqrt (1 - (Real.sin (γ / 2))^2)) := 
sorry

end NUMINAMATH_GPT_regular_triangular_pyramid_volume_l2407_240715


namespace NUMINAMATH_GPT_find_fraction_identity_l2407_240759

variable (x y z : ℝ)

theorem find_fraction_identity
 (h1 : 16 * y^2 = 15 * x * z)
 (h2 : y = 2 * x * z / (x + z)) :
 x / z + z / x = 34 / 15 := by
-- proof skipped
sorry

end NUMINAMATH_GPT_find_fraction_identity_l2407_240759


namespace NUMINAMATH_GPT_toy_store_restock_l2407_240704

theorem toy_store_restock 
  (initial_games : ℕ) (games_sold : ℕ) (after_restock_games : ℕ) 
  (initial_games_condition : initial_games = 95)
  (games_sold_condition : games_sold = 68)
  (after_restock_games_condition : after_restock_games = 74) :
  after_restock_games - (initial_games - games_sold) = 47 :=
by {
  sorry
}

end NUMINAMATH_GPT_toy_store_restock_l2407_240704


namespace NUMINAMATH_GPT_domain_of_function_l2407_240746

theorem domain_of_function (x : ℝ) :
  (x^2 - 5*x + 6 ≥ 0) → (x ≠ 2) → (x < 2 ∨ x ≥ 3) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_domain_of_function_l2407_240746


namespace NUMINAMATH_GPT_part1_part2_l2407_240748

noncomputable def f (x a : ℝ) := 5 - |x + a| - |x - 2|

theorem part1 : 
  (∀ x, f x 1 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
sorry

theorem part2 :
  (∀ a, (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2407_240748


namespace NUMINAMATH_GPT_tyrone_gave_marbles_to_eric_l2407_240732

theorem tyrone_gave_marbles_to_eric (initial_tyrone_marbles : ℕ) (initial_eric_marbles : ℕ) (marbles_given : ℕ) :
  initial_tyrone_marbles = 150 ∧ initial_eric_marbles = 30 ∧ (initial_tyrone_marbles - marbles_given = 3 * initial_eric_marbles) → marbles_given = 60 :=
by
  sorry

end NUMINAMATH_GPT_tyrone_gave_marbles_to_eric_l2407_240732


namespace NUMINAMATH_GPT_f_eq_zero_range_x_l2407_240789

-- Definition of the function f on domain ℝ*
def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_domain : ∀ x : ℝ, x ≠ 0 → f x = f x
axiom f_4 : f 4 = 1
axiom f_multiplicative : ∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → f (x1 * x2) = f x1 + f x2
axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y

-- Problem (1): Prove f(1) = 0
theorem f_eq_zero : f 1 = 0 :=
sorry

-- Problem (2): Prove range 3 < x ≤ 5 given the inequality condition
theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 :=
sorry

end NUMINAMATH_GPT_f_eq_zero_range_x_l2407_240789


namespace NUMINAMATH_GPT_simplify_fraction_l2407_240731

theorem simplify_fraction (k : ℤ) : 
  (∃ (a b : ℤ), a = 1 ∧ b = 2 ∧ (6 * k + 12) / 6 = a * k + b) → (1 / 2 : ℚ) = (1 / 2 : ℚ) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2407_240731


namespace NUMINAMATH_GPT_find_x_l2407_240766

theorem find_x (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2407_240766


namespace NUMINAMATH_GPT_option_B_valid_l2407_240737

-- Definitions derived from conditions
def at_least_one_black (balls : List Bool) : Prop :=
  ∃ b ∈ balls, b = true

def both_black (balls : List Bool) : Prop :=
  balls = [true, true]

def exactly_one_black (balls : List Bool) : Prop :=
  balls.count true = 1

def exactly_two_black (balls : List Bool) : Prop :=
  balls.count true = 2

def mutually_exclusive (P Q : Prop) : Prop :=
  P ∧ Q → False

def non_complementary (P Q : Prop) : Prop :=
  ¬(P → ¬Q) ∧ ¬(¬P → Q)

-- Balls: true represents a black ball, false represents a red ball.
def all_draws := [[true, true], [true, false], [false, true], [false, false]]

-- Proof statement
theorem option_B_valid :
  (mutually_exclusive (exactly_one_black [true, false]) (exactly_two_black [true, true])) ∧ 
  (non_complementary (exactly_one_black [true, false]) (exactly_two_black [true, true])) :=
  sorry

end NUMINAMATH_GPT_option_B_valid_l2407_240737


namespace NUMINAMATH_GPT_smallest_number_divisible_l2407_240781

theorem smallest_number_divisible (x y : ℕ) (h : x + y = 4728) 
  (h1 : (x + y) % 27 = 0) 
  (h2 : (x + y) % 35 = 0) 
  (h3 : (x + y) % 25 = 0) 
  (h4 : (x + y) % 21 = 0) : 
  x = 4725 := by 
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_l2407_240781


namespace NUMINAMATH_GPT_total_vessels_proof_l2407_240791

def cruise_ships : Nat := 4
def cargo_ships : Nat := cruise_ships * 2
def sailboats : Nat := cargo_ships + 6
def fishing_boats : Nat := sailboats / 7
def total_vessels : Nat := cruise_ships + cargo_ships + sailboats + fishing_boats

theorem total_vessels_proof : total_vessels = 28 := by
  sorry

end NUMINAMATH_GPT_total_vessels_proof_l2407_240791


namespace NUMINAMATH_GPT_q_evaluation_l2407_240747

def q (x y : ℤ) : ℤ :=
if x >= 0 ∧ y >= 0 then x - y
else if x < 0 ∧ y < 0 then x + 3 * y
else 2 * x + 2 * y

theorem q_evaluation : q (q 1 (-1)) (q (-2) (-3)) = -22 := by
sorry

end NUMINAMATH_GPT_q_evaluation_l2407_240747


namespace NUMINAMATH_GPT_probability_prime_sum_is_1_9_l2407_240762

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_sum (a b : ℕ) : Bool :=
  Nat.Prime (a + b)

def valid_prime_pairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)]

def total_pairs : ℕ :=
  45

def valid_pairs_count : ℕ :=
  valid_prime_pairs.length

def probability_prime_sum : ℚ :=
  valid_pairs_count / total_pairs

theorem probability_prime_sum_is_1_9 :
  probability_prime_sum = 1 / 9 := 
by
  sorry

end NUMINAMATH_GPT_probability_prime_sum_is_1_9_l2407_240762


namespace NUMINAMATH_GPT_smallest_two_digit_integer_l2407_240700

theorem smallest_two_digit_integer (n a b : ℕ) (h1 : n = 10 * a + b) (h2 : 2 * n = 10 * b + a + 5) (h3 : 1 ≤ a) (h4 : a ≤ 9) (h5 : 0 ≤ b) (h6 : b ≤ 9) : n = 69 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_two_digit_integer_l2407_240700


namespace NUMINAMATH_GPT_compound_interest_second_year_l2407_240793

theorem compound_interest_second_year
  (P: ℝ) (r: ℝ) (CI_3 : ℝ) (CI_2 : ℝ)
  (h1 : r = 0.06)
  (h2 : CI_3 = 1272)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1200 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_second_year_l2407_240793


namespace NUMINAMATH_GPT_difference_of_percentages_l2407_240718

variable (x y : ℝ)

theorem difference_of_percentages :
  (0.60 * (50 + x)) - (0.45 * (30 + y)) = 16.5 + 0.60 * x - 0.45 * y := 
sorry

end NUMINAMATH_GPT_difference_of_percentages_l2407_240718


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2407_240753

theorem hyperbola_eccentricity (x y : ℝ) :
  (∃ a b : ℝ, a^2 = 4 ∧ b^2 = 12 ∧ (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) ∧
  ∃ e : ℝ, e = c / a ∧ e = 2) :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2407_240753


namespace NUMINAMATH_GPT_initial_students_count_l2407_240797

theorem initial_students_count (n : ℕ) (W : ℝ) 
  (h1 : W = n * 28) 
  (h2 : W + 1 = (n + 1) * 27.1) : 
  n = 29 := by
  sorry

end NUMINAMATH_GPT_initial_students_count_l2407_240797


namespace NUMINAMATH_GPT_roots_sum_roots_product_algebraic_expression_l2407_240719

theorem roots_sum (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 + x2 = 1 :=
sorry

theorem roots_product (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 * x2 = -1 :=
sorry

theorem algebraic_expression (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1^2 + x2^2 = 3 :=
sorry

end NUMINAMATH_GPT_roots_sum_roots_product_algebraic_expression_l2407_240719


namespace NUMINAMATH_GPT_product_xy_min_value_x_plus_y_min_value_attained_l2407_240736

theorem product_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : x * y = 64 := 
sorry

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : 
  x + y = 18 := 
sorry

-- Additional theorem to prove that the minimum value is attained when x = 6 and y = 12
theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) :
  x = 6 ∧ y = 12 := 
sorry

end NUMINAMATH_GPT_product_xy_min_value_x_plus_y_min_value_attained_l2407_240736


namespace NUMINAMATH_GPT_max_a_l2407_240741

variable {a x : ℝ}

theorem max_a (h : x^2 - 2 * x - 3 > 0 → x < a ∧ ¬ (x < a → x^2 - 2 * x - 3 > 0)) : a = 3 :=
sorry

end NUMINAMATH_GPT_max_a_l2407_240741


namespace NUMINAMATH_GPT_find_initial_candies_l2407_240739

-- Definitions for the conditions
def initial_candies (x : ℕ) : Prop :=
  (3 * x) % 4 = 0 ∧
  (x % 2) = 0 ∧
  ∃ (k : ℕ), 2 ≤ k ∧ k ≤ 6 ∧ (1 * x) / 2 - 20 - k = 4

-- Theorems we need to prove
theorem find_initial_candies (x : ℕ) (h : initial_candies x) : x = 52 ∨ x = 56 ∨ x = 60 :=
sorry

end NUMINAMATH_GPT_find_initial_candies_l2407_240739


namespace NUMINAMATH_GPT_number_of_possible_values_l2407_240796

theorem number_of_possible_values (a b c : ℕ) (h : a + 11 * b + 111 * c = 1050) :
  ∃ (n : ℕ), 6 ≤ n ∧ n ≤ 1050 ∧ (n % 9 = 6) ∧ (n = a + 2 * b + 3 * c) :=
sorry

end NUMINAMATH_GPT_number_of_possible_values_l2407_240796


namespace NUMINAMATH_GPT_exists_nat_m_inequality_for_large_n_l2407_240717

section sequence_problem

-- Define the sequence
noncomputable def a (n : ℕ) : ℚ :=
if n = 7 then 16 / 3 else
if n < 7 then 0 else -- hands off values before a7 that are not needed
3 * a (n - 1) / (7 - a (n - 1) + 4)

-- Define the properties to be proven
theorem exists_nat_m {m : ℕ} :
  (∀ n, n > m → a n < 2) ∧ (∀ n, n ≤ m → a n > 2) :=
sorry

theorem inequality_for_large_n (n : ℕ) (hn : n ≥ 10) :
  (a (n - 1) + a n + 1) / 2 < a n :=
sorry

end sequence_problem

end NUMINAMATH_GPT_exists_nat_m_inequality_for_large_n_l2407_240717


namespace NUMINAMATH_GPT_hyperbola_center_is_equidistant_l2407_240795

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem hyperbola_center_is_equidistant (F1 F2 C : ℝ × ℝ) 
  (hF1 : F1 = (3, -2)) 
  (hF2 : F2 = (11, 6))
  (hC : C = ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2)) :
  C = (7, 2) ∧ distance C F1 = distance C F2 :=
by
  -- Fill in with the appropriate proofs
  sorry

end NUMINAMATH_GPT_hyperbola_center_is_equidistant_l2407_240795


namespace NUMINAMATH_GPT_apples_fraction_of_pears_l2407_240710

variables (A O P : ℕ)

-- Conditions
def oranges_condition := O = 3 * A
def pears_condition := P = 4 * O

-- Statement we need to prove
theorem apples_fraction_of_pears (A O P : ℕ) (h1 : O = 3 * A) (h2 : P = 4 * O) : (A : ℚ) / P = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_apples_fraction_of_pears_l2407_240710


namespace NUMINAMATH_GPT_range_of_a_l2407_240768

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → quadratic_function a x ≥ quadratic_function a y ∧ y ≤ 4) →
  a ≤ -5 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l2407_240768


namespace NUMINAMATH_GPT_product_of_three_consecutive_integers_is_multiple_of_6_l2407_240742

theorem product_of_three_consecutive_integers_is_multiple_of_6 (n : ℕ) (h : n > 0) :
    ∃ k : ℕ, n * (n + 1) * (n + 2) = 6 * k :=
by
  sorry

end NUMINAMATH_GPT_product_of_three_consecutive_integers_is_multiple_of_6_l2407_240742


namespace NUMINAMATH_GPT_total_loss_l2407_240703

theorem total_loss (P : ℝ) (A : ℝ) (L : ℝ) (h1 : A = (1/9) * P) (h2 : 603 = (P / (A + P)) * L) : 
  L = 670 :=
by
  sorry

end NUMINAMATH_GPT_total_loss_l2407_240703


namespace NUMINAMATH_GPT_unique_seq_l2407_240745

def seq (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j

theorem unique_seq (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n) : 
  seq a ↔ (∀ n, a n = n) := 
by
  intros
  sorry

end NUMINAMATH_GPT_unique_seq_l2407_240745


namespace NUMINAMATH_GPT_jovial_frogs_not_green_l2407_240770

variables {Frog : Type} (jovial green can_jump can_swim : Frog → Prop)

theorem jovial_frogs_not_green :
  (∀ frog, jovial frog → can_swim frog) →
  (∀ frog, green frog → ¬ can_jump frog) →
  (∀ frog, ¬ can_jump frog → ¬ can_swim frog) →
  (∀ frog, jovial frog → ¬ green frog) :=
by
  intros h1 h2 h3 frog hj
  sorry

end NUMINAMATH_GPT_jovial_frogs_not_green_l2407_240770


namespace NUMINAMATH_GPT_coefficient_x3_l2407_240714

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3 (n k : ℕ) (x : ℤ) :
  let expTerm : ℤ := 1 - x + (1 / x^2017)
  let expansion := fun (k : ℕ) => binomial n k • ((1 - x)^(n - k) * (1 / x^2017)^k)
  (n = 9) → (k = 3) →
  (expansion k) = -84 :=
  by
    intros
    sorry

end NUMINAMATH_GPT_coefficient_x3_l2407_240714


namespace NUMINAMATH_GPT_part1_part2_l2407_240716

-- Definitions for the conditions
def A : Set ℝ := {x : ℝ | 2 * x - 4 < 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}
def U : Set ℝ := Set.univ

-- The questions translated as Lean theorems
theorem part1 : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

theorem part2 : (U \ A) ∩ B = {x : ℝ | 2 ≤ x ∧ x < 5} := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2407_240716


namespace NUMINAMATH_GPT_find_percentage_l2407_240767

theorem find_percentage : 
  ∀ (P : ℕ), 
  (50 - 47 = (P / 100) * 15) →
  P = 20 := 
by
  intro P h
  sorry

end NUMINAMATH_GPT_find_percentage_l2407_240767


namespace NUMINAMATH_GPT_cost_per_mile_proof_l2407_240724

noncomputable def daily_rental_cost : ℝ := 50
noncomputable def daily_budget : ℝ := 88
noncomputable def max_miles : ℝ := 190.0

theorem cost_per_mile_proof : 
  (daily_budget - daily_rental_cost) / max_miles = 0.20 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_mile_proof_l2407_240724


namespace NUMINAMATH_GPT_solve_eq_l2407_240790

theorem solve_eq (x : ℝ) (h : 2 - 1 / (2 - x) = 1 / (2 - x)) : x = 1 := 
sorry

end NUMINAMATH_GPT_solve_eq_l2407_240790


namespace NUMINAMATH_GPT_scenario_a_scenario_b_scenario_c_scenario_d_scenario_e_scenario_f_l2407_240734

open Nat

-- Definitions for combinations and permutations
def binomial (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))
def variations (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

-- Scenario a: Each path can be used by at most one person and at most once
theorem scenario_a : binomial 5 2 * binomial 3 2 = 30 := by sorry

-- Scenario b: Each path can be used twice but only in different directions
theorem scenario_b : binomial 5 2 * binomial 5 2 = 100 := by sorry

-- Scenario c: No restrictions
theorem scenario_c : (5 * 5) * (5 * 5) = 625 := by sorry

-- Scenario d: Same as a) with two people distinguished
theorem scenario_d : variations 5 2 * variations 3 2 = 120 := by sorry

-- Scenario e: Same as b) with two people distinguished
theorem scenario_e : variations 5 2 * variations 5 2 = 400 := by sorry

-- Scenario f: Same as c) with two people distinguished
theorem scenario_f : (5 * 5) * (5 * 5) = 625 := by sorry

end NUMINAMATH_GPT_scenario_a_scenario_b_scenario_c_scenario_d_scenario_e_scenario_f_l2407_240734


namespace NUMINAMATH_GPT_convert_fraction_to_decimal_l2407_240779

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end NUMINAMATH_GPT_convert_fraction_to_decimal_l2407_240779


namespace NUMINAMATH_GPT_possible_slopes_of_line_intersecting_ellipse_l2407_240754

theorem possible_slopes_of_line_intersecting_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ (Set.Iic (-2/5) ∪ Set.Ici (2/5)) :=
by
  sorry

end NUMINAMATH_GPT_possible_slopes_of_line_intersecting_ellipse_l2407_240754


namespace NUMINAMATH_GPT_total_wet_surface_area_eq_l2407_240738

-- Definitions based on given conditions
def length_cistern : ℝ := 10
def width_cistern : ℝ := 6
def height_water : ℝ := 1.35

-- Problem statement: Prove the total wet surface area is as calculated
theorem total_wet_surface_area_eq :
  let area_bottom : ℝ := length_cistern * width_cistern
  let area_longer_sides : ℝ := 2 * (length_cistern * height_water)
  let area_shorter_sides : ℝ := 2 * (width_cistern * height_water)
  let total_wet_surface_area : ℝ := area_bottom + area_longer_sides + area_shorter_sides
  total_wet_surface_area = 103.2 :=
by
  -- Since we do not need the proof, we use sorry here
  sorry

end NUMINAMATH_GPT_total_wet_surface_area_eq_l2407_240738


namespace NUMINAMATH_GPT_largest_number_l2407_240784

theorem largest_number 
  (a b c : ℝ) (h1 : a = 0.8) (h2 : b = 1/2) (h3 : c = 0.9) (h4 : a ≤ 2) (h5 : b ≤ 2) (h6 : c ≤ 2) :
  max (max a b) c = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_l2407_240784


namespace NUMINAMATH_GPT_time_for_Dawson_l2407_240774

variable (D : ℝ)
variable (Henry_time : ℝ := 7)
variable (avg_time : ℝ := 22.5)

theorem time_for_Dawson (h : avg_time = (D + Henry_time) / 2) : D = 38 := 
by 
  sorry

end NUMINAMATH_GPT_time_for_Dawson_l2407_240774


namespace NUMINAMATH_GPT_combine_like_terms_l2407_240712

theorem combine_like_terms (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := 
  sorry

end NUMINAMATH_GPT_combine_like_terms_l2407_240712


namespace NUMINAMATH_GPT_albert_runs_track_l2407_240775

theorem albert_runs_track (x : ℕ) (track_distance : ℕ) (total_distance : ℕ) (additional_laps : ℕ) 
(h1 : track_distance = 9)
(h2 : total_distance = 99)
(h3 : additional_laps = 5)
(h4 : total_distance = track_distance * x + track_distance * additional_laps) :
x = 6 :=
by
  sorry

end NUMINAMATH_GPT_albert_runs_track_l2407_240775


namespace NUMINAMATH_GPT_minimal_range_of_observations_l2407_240735

variable {x1 x2 x3 x4 x5 : ℝ}

def arithmetic_mean (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  (x1 + x2 + x3 + x4 + x5) / 5 = 8

def median (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  x3 = 10 ∧ x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5

theorem minimal_range_of_observations 
  (h_mean : arithmetic_mean x1 x2 x3 x4 x5)
  (h_median : median x1 x2 x3 x4 x5) : 
  ∃ x1 x2 x3 x4 x5 : ℝ, (x1 + x2 + x3 + x4 + x5) = 40 ∧ x3 = 10 ∧ x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5 ∧ (x5 - x1) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_minimal_range_of_observations_l2407_240735


namespace NUMINAMATH_GPT_determine_n_eq_1_l2407_240709

theorem determine_n_eq_1 :
  ∃ n : ℝ, (∀ x : ℝ, (x = 2 → (x^3 - 3*x^2 + n = 2*x^3 - 6*x^2 + 5*n))) → n = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_n_eq_1_l2407_240709


namespace NUMINAMATH_GPT_pure_imaginary_iff_real_part_zero_l2407_240713

theorem pure_imaginary_iff_real_part_zero (a b : ℝ) : (∃ z : ℂ, z = a + bi ∧ z.im ≠ 0) ↔ (a = 0 ∧ b ≠ 0) :=
sorry

end NUMINAMATH_GPT_pure_imaginary_iff_real_part_zero_l2407_240713


namespace NUMINAMATH_GPT_height_of_C_l2407_240730

noncomputable def height_A_B_C (h_A h_B h_C : ℝ) : Prop := 
  (h_A + h_B + h_C) / 3 = 143 ∧ 
  h_A + 4.5 = (h_B + h_C) / 2 ∧ 
  h_B = h_C + 3

theorem height_of_C (h_A h_B h_C : ℝ) (h : height_A_B_C h_A h_B h_C) : h_C = 143 :=
  sorry

end NUMINAMATH_GPT_height_of_C_l2407_240730


namespace NUMINAMATH_GPT_solve_for_x_l2407_240783

theorem solve_for_x (x : ℝ) (h : (3 / 4) - (1 / 2) = 1 / x) : x = 4 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2407_240783


namespace NUMINAMATH_GPT_arithmetic_mean_of_reciprocals_is_correct_l2407_240778

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_reciprocals_is_correct_l2407_240778


namespace NUMINAMATH_GPT_sum_of_f1_possible_values_l2407_240726

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f1_possible_values :
  (∀ (x y : ℝ), f (f (x - y)) = f x * f y - f x + f y - 2 * x * y) →
  (f 1 = -1) := sorry

end NUMINAMATH_GPT_sum_of_f1_possible_values_l2407_240726


namespace NUMINAMATH_GPT_bacon_cost_l2407_240782

namespace PancakeBreakfast

def cost_of_stack_pancakes : ℝ := 4.0
def stacks_sold : ℕ := 60
def slices_bacon_sold : ℕ := 90
def total_revenue : ℝ := 420.0

theorem bacon_cost (B : ℝ) 
  (h1 : stacks_sold * cost_of_stack_pancakes + slices_bacon_sold * B = total_revenue) : 
  B = 2 :=
  by {
    sorry
  }

end PancakeBreakfast

end NUMINAMATH_GPT_bacon_cost_l2407_240782


namespace NUMINAMATH_GPT_gravel_cost_correct_l2407_240725

-- Definitions from the conditions
def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 60
def road_width : ℕ := 15
def gravel_cost_per_sq_m : ℕ := 3

-- Calculate areas of the roads
def area_road_length : ℕ := lawn_length * road_width
def area_road_breadth : ℕ := (lawn_breadth - road_width) * road_width

-- Total area to be graveled
def total_area : ℕ := area_road_length + area_road_breadth

-- Total cost
def total_cost : ℕ := total_area * gravel_cost_per_sq_m

-- Prove the total cost is 5625 Rs
theorem gravel_cost_correct : total_cost = 5625 := by
  sorry

end NUMINAMATH_GPT_gravel_cost_correct_l2407_240725


namespace NUMINAMATH_GPT_total_earnings_correct_l2407_240733

-- Define the earnings of each individual
def SalvadorEarnings := 1956
def SantoEarnings := SalvadorEarnings / 2
def MariaEarnings := 3 * SantoEarnings
def PedroEarnings := SantoEarnings + MariaEarnings

-- Define the total earnings calculation
def TotalEarnings := SalvadorEarnings + SantoEarnings + MariaEarnings + PedroEarnings

-- State the theorem to prove
theorem total_earnings_correct :
  TotalEarnings = 9780 :=
sorry

end NUMINAMATH_GPT_total_earnings_correct_l2407_240733


namespace NUMINAMATH_GPT_find_m_minus_n_l2407_240702

theorem find_m_minus_n (m n : ℝ) (h1 : -5 + 1 = m) (h2 : -5 * 1 = n) : m - n = 1 :=
sorry

end NUMINAMATH_GPT_find_m_minus_n_l2407_240702


namespace NUMINAMATH_GPT_max_value_2x_minus_y_l2407_240769

theorem max_value_2x_minus_y (x y : ℝ) (h : x^2 / 4 + y^2 / 9 = 1) : 2 * x - y ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_value_2x_minus_y_l2407_240769


namespace NUMINAMATH_GPT_value_of_m_l2407_240794

theorem value_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 / (x - 2) + (x + m) / (2 - x) = 1)) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l2407_240794


namespace NUMINAMATH_GPT_ring_stack_distance_l2407_240773

noncomputable def vertical_distance (rings : Nat) : Nat :=
  let diameters := List.range rings |>.map (λ i => 15 - 2 * i)
  let thickness := 1 * rings
  thickness

theorem ring_stack_distance :
  vertical_distance 7 = 58 :=
by 
  sorry

end NUMINAMATH_GPT_ring_stack_distance_l2407_240773


namespace NUMINAMATH_GPT_SquareArea_l2407_240752

theorem SquareArea (s : ℝ) (θ : ℝ) (h1 : s = 3) (h2 : θ = π / 4) : s * s = 9 := 
by 
  sorry

end NUMINAMATH_GPT_SquareArea_l2407_240752


namespace NUMINAMATH_GPT_mitch_earns_correctly_l2407_240740

noncomputable def mitch_weekly_earnings : ℝ :=
  let earnings_mw := 3 * (3 * 5 : ℝ) -- Monday to Wednesday
  let earnings_tf := 2 * (6 * 4 : ℝ) -- Thursday and Friday
  let earnings_sat := 4 * 6         -- Saturday
  let earnings_sun := 5 * 8         -- Sunday
  let total_earnings := earnings_mw + earnings_tf + earnings_sat + earnings_sun
  let after_expenses := total_earnings - 25
  let after_tax := after_expenses - 0.10 * after_expenses
  after_tax

theorem mitch_earns_correctly : mitch_weekly_earnings = 118.80 := by
  sorry

end NUMINAMATH_GPT_mitch_earns_correctly_l2407_240740


namespace NUMINAMATH_GPT_average_temps_l2407_240722

-- Define the temperature lists
def temps_C : List ℚ := [
  37.3, 37.2, 36.9, -- Sunday
  36.6, 36.9, 37.1, -- Monday
  37.1, 37.3, 37.2, -- Tuesday
  36.8, 37.3, 37.5, -- Wednesday
  37.1, 37.7, 37.3, -- Thursday
  37.5, 37.4, 36.9, -- Friday
  36.9, 37.0, 37.1  -- Saturday
]

def temps_K : List ℚ := [
  310.4, 310.3, 310.0, -- Sunday
  309.8, 310.0, 310.2, -- Monday
  310.2, 310.4, 310.3, -- Tuesday
  309.9, 310.4, 310.6, -- Wednesday
  310.2, 310.8, 310.4, -- Thursday
  310.6, 310.5, 310.0, -- Friday
  310.0, 310.1, 310.2  -- Saturday
]

def temps_R : List ℚ := [
  558.7, 558.6, 558.1, -- Sunday
  557.7, 558.1, 558.3, -- Monday
  558.3, 558.7, 558.6, -- Tuesday
  558.0, 558.7, 559.1, -- Wednesday
  558.3, 559.4, 558.7, -- Thursday
  559.1, 558.9, 558.1, -- Friday
  558.1, 558.2, 558.3  -- Saturday
]

-- Calculate the average of a list of temperatures
def average (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

-- Define the average temperatures
def avg_C := average temps_C
def avg_K := average temps_K
def avg_R := average temps_R

-- State that the computed averages are equal to the provided values
theorem average_temps :
  avg_C = 37.1143 ∧
  avg_K = 310.1619 ∧
  avg_R = 558.2524 :=
by
  -- Proof can be completed here
  sorry

end NUMINAMATH_GPT_average_temps_l2407_240722


namespace NUMINAMATH_GPT_problem_a_problem_b_l2407_240792
-- Import the entire math library to ensure all necessary functionality is included

-- Define the problem context
variables {x y z : ℝ}

-- State the conditions as definitions
def conditions (x y z : ℝ) : Prop :=
  (x ≤ y) ∧ (y ≤ z) ∧ (x + y + z = 12) ∧ (x^2 + y^2 + z^2 = 54)

-- State the formal proof problems
theorem problem_a (h : conditions x y z) : x ≤ 3 ∧ 5 ≤ z :=
sorry

theorem problem_b (h : conditions x y z) : 
  9 ≤ x * y ∧ x * y ≤ 25 ∧
  9 ≤ y * z ∧ y * z ≤ 25 ∧
  9 ≤ z * x ∧ z * x ≤ 25 :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_l2407_240792


namespace NUMINAMATH_GPT_lucy_age_l2407_240763

theorem lucy_age (Inez_age : ℕ) (Zack_age : ℕ) (Jose_age : ℕ) (Lucy_age : ℕ) 
  (h1 : Inez_age = 18) 
  (h2 : Zack_age = Inez_age + 4) 
  (h3 : Jose_age = Zack_age - 6) 
  (h4 : Lucy_age = Jose_age + 2) : 
  Lucy_age = 18 := by
sorry

end NUMINAMATH_GPT_lucy_age_l2407_240763


namespace NUMINAMATH_GPT_john_subtracts_79_l2407_240772

theorem john_subtracts_79 :
  let a := 40
  let b := 1
  let n := (a - b) * (a - b)
  n = a * a - 79
:= by
  sorry

end NUMINAMATH_GPT_john_subtracts_79_l2407_240772


namespace NUMINAMATH_GPT_cards_with_1_count_l2407_240749

theorem cards_with_1_count (m k : ℕ) 
  (h1 : k = m + 100) 
  (sum_of_products : (m * (m - 1) / 2) + (k * (k - 1) / 2) - m * k = 1000) : 
  m = 3950 :=
by
  sorry

end NUMINAMATH_GPT_cards_with_1_count_l2407_240749


namespace NUMINAMATH_GPT_question1_question2_l2407_240750

-- Condition: p
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0

-- Condition: q
def q (a : ℝ) (x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Question 1 statement: Given p is true and q is false when a = 0, find range of x
theorem question1 (x : ℝ) (h : p x ∧ ¬q 0 x) : -7/2 ≤ x ∧ x < -3 :=
sorry

-- Question 2 statement: If p is a sufficient condition for q, find range of a
theorem question2 (a : ℝ) (h : ∀ x, p x → q a x) : -5/2 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_GPT_question1_question2_l2407_240750


namespace NUMINAMATH_GPT_calculate_radius_l2407_240708

noncomputable def radius_of_wheel (D : ℝ) (N : ℕ) (π : ℝ) : ℝ :=
  D / (2 * π * N)

theorem calculate_radius : 
  radius_of_wheel 4224 3000 Real.pi = 0.224 :=
by
  sorry

end NUMINAMATH_GPT_calculate_radius_l2407_240708


namespace NUMINAMATH_GPT_simplify_abs_expression_l2407_240723

theorem simplify_abs_expression
  (a b : ℝ)
  (h1 : a < 0)
  (h2 : a * b < 0)
  : |a - b - 3| - |4 + b - a| = -1 := by
  sorry

end NUMINAMATH_GPT_simplify_abs_expression_l2407_240723


namespace NUMINAMATH_GPT_polynomial_has_no_real_roots_l2407_240777

theorem polynomial_has_no_real_roots :
  ∀ x : ℝ, x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + 5/2 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_has_no_real_roots_l2407_240777


namespace NUMINAMATH_GPT_total_tosses_correct_l2407_240787

def num_heads : Nat := 3
def num_tails : Nat := 7
def total_tosses : Nat := num_heads + num_tails

theorem total_tosses_correct : total_tosses = 10 := by
  sorry

end NUMINAMATH_GPT_total_tosses_correct_l2407_240787


namespace NUMINAMATH_GPT_math_problem_l2407_240721

variable {x a b : ℝ}

theorem math_problem (h1 : x < a) (h2 : a < 0) (h3 : b = -a) : x^2 > b^2 ∧ b^2 > 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_l2407_240721


namespace NUMINAMATH_GPT_digits_in_number_l2407_240705

def four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def contains_digits (n : ℕ) (d1 d2 d3 : ℕ) : Prop :=
  (n / 1000 = d1 ∨ n / 100 % 10 = d1 ∨ n / 10 % 10 = d1 ∨ n % 10 = d1) ∧
  (n / 1000 = d2 ∨ n / 100 % 10 = d2 ∨ n / 10 % 10 = d2 ∨ n % 10 = d2) ∧
  (n / 1000 = d3 ∨ n / 100 % 10 = d3 ∨ n / 10 % 10 = d3 ∨ n % 10 = d3)

def exactly_two_statements_true (s1 s2 s3 : Prop) : Prop :=
  (s1 ∧ s2 ∧ ¬s3) ∨ (s1 ∧ ¬s2 ∧ s3) ∨ (¬s1 ∧ s2 ∧ s3)

theorem digits_in_number (n : ℕ) 
  (h1 : four_digit_number n)
  (h2 : contains_digits n 1 4 5 ∨ contains_digits n 1 5 9 ∨ contains_digits n 7 8 9)
  (h3 : exactly_two_statements_true (contains_digits n 1 4 5) (contains_digits n 1 5 9) (contains_digits n 7 8 9)) :
  contains_digits n 1 4 5 ∧ contains_digits n 1 5 9 :=
sorry

end NUMINAMATH_GPT_digits_in_number_l2407_240705


namespace NUMINAMATH_GPT_center_of_circle_tangent_to_parallel_lines_l2407_240701

-- Define the line equations
def line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 40
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -20
def line3 (x y : ℝ) : Prop := x - 2 * y = 0

-- The proof problem
theorem center_of_circle_tangent_to_parallel_lines
  (x y : ℝ)
  (h1 : line1 x y → false)
  (h2 : line2 x y → false)
  (h3 : line3 x y) :
  x = 10 ∧ y = 5 := by
  sorry

end NUMINAMATH_GPT_center_of_circle_tangent_to_parallel_lines_l2407_240701


namespace NUMINAMATH_GPT_tangent_line_eq_monotonic_intervals_extremes_f_l2407_240757

variables {a x : ℝ}

noncomputable def f (a x : ℝ) : ℝ := -1/3 * x^3 + 2 * a * x^2 - 3 * a^2 * x
noncomputable def f' (a x : ℝ) : ℝ := -x^2 + 4 * a * x - 3 * a^2

theorem tangent_line_eq {a : ℝ} (h : a = -1) : (∃ y, y = f (-1) (-2) ∧ 3 * x - 3 * y + 8 = 0) := sorry

theorem monotonic_intervals_extremes {a : ℝ} (h : 0 < a) :
  (∀ x, (a < x ∧ x < 3 * a → 0 < f' a x) ∧ 
        (x < a ∨ 3 * a < x → f' a x < 0) ∧ 
        (f a (3 * a) = 0 ∧ f a a = -4/3 * a^3)) := sorry

theorem f'_inequality_range (h1 : ∀ x, 2 * a ≤ x ∧ x ≤ 2 * a + 2 → |f' a x| ≤ 3 * a) :
  (1 ≤ a ∧ a ≤ 3) := sorry

end NUMINAMATH_GPT_tangent_line_eq_monotonic_intervals_extremes_f_l2407_240757


namespace NUMINAMATH_GPT_ab_sum_l2407_240798

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -1 < x ∧ x < 5 }
def B : Set ℝ := { x | 2 < x ∧ x < 8 }
def C (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a - 2 }
def complement_A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 5 }
def complement_B : Set ℝ := { x | x ≤ 2 ∨ x ≥ 8 }
def complement_A_and_C (a b : ℝ) : Set ℝ := { x | 6 ≤ x ∧ x ≤ b }

theorem ab_sum (a b: ℝ) (h: (complement_A ∩ C a) = complement_A_and_C a b) : a + b = 13 :=
by
  sorry

end NUMINAMATH_GPT_ab_sum_l2407_240798


namespace NUMINAMATH_GPT_ratio_problem_l2407_240764

-- Define the conditions and the required proof
theorem ratio_problem (p q n : ℝ) (h1 : p / q = 5 / n) (h2 : 2 * p + q = 14) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_problem_l2407_240764


namespace NUMINAMATH_GPT_original_price_is_correct_l2407_240756

-- Given conditions as Lean definitions
def reduced_price : ℝ := 2468
def reduction_amount : ℝ := 161.46

-- To find the original price including the sales tax
def original_price_including_tax (P : ℝ) : Prop :=
  P - reduction_amount = reduced_price

-- The proof statement to show the price is 2629.46
theorem original_price_is_correct : original_price_including_tax 2629.46 :=
by
  sorry

end NUMINAMATH_GPT_original_price_is_correct_l2407_240756


namespace NUMINAMATH_GPT_kids_have_equal_eyes_l2407_240799

theorem kids_have_equal_eyes (mom_eyes dad_eyes kids_num total_eyes kids_eyes : ℕ) 
  (h_mom_eyes : mom_eyes = 1) 
  (h_dad_eyes : dad_eyes = 3) 
  (h_kids_num : kids_num = 3) 
  (h_total_eyes : total_eyes = 16) 
  (h_family_eyes : mom_eyes + dad_eyes + kids_num * kids_eyes = total_eyes) :
  kids_eyes = 4 :=
by
  sorry

end NUMINAMATH_GPT_kids_have_equal_eyes_l2407_240799


namespace NUMINAMATH_GPT_fraction_of_girls_in_debate_l2407_240786

theorem fraction_of_girls_in_debate (g b : ℕ) (h : g = b) :
  ((2 / 3) * g) / ((2 / 3) * g + (3 / 5) * b) = 30 / 57 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_girls_in_debate_l2407_240786


namespace NUMINAMATH_GPT_trapezoid_perimeter_is_correct_l2407_240728

noncomputable def trapezoid_perimeter_proof : ℝ :=
  let EF := 60
  let θ := Real.pi / 4 -- 45 degrees in radians
  let h := 30 * Real.sqrt 2
  let GH := EF + 2 * h / Real.tan θ
  let EG := h / Real.tan θ
  EF + GH + 2 * EG -- Perimeter calculation

theorem trapezoid_perimeter_is_correct :
  trapezoid_perimeter_proof = 180 + 60 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_is_correct_l2407_240728


namespace NUMINAMATH_GPT_ratio_area_rectangle_triangle_l2407_240776

-- Define the lengths L and W as positive real numbers
variables {L W : ℝ} (hL : L > 0) (hW : W > 0)

-- Define the area of the rectangle
noncomputable def area_rectangle (L W : ℝ) : ℝ := L * W

-- Define the area of the triangle with base L and height W
noncomputable def area_triangle (L W : ℝ) : ℝ := (1 / 2) * L * W

-- Define the ratio between the area of the rectangle and the area of the triangle
noncomputable def area_ratio (L W : ℝ) : ℝ := area_rectangle L W / area_triangle L W

-- Prove that this ratio is equal to 2
theorem ratio_area_rectangle_triangle : area_ratio L W = 2 := by sorry

end NUMINAMATH_GPT_ratio_area_rectangle_triangle_l2407_240776


namespace NUMINAMATH_GPT_n_product_expression_l2407_240707

theorem n_product_expression (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 :=
sorry

end NUMINAMATH_GPT_n_product_expression_l2407_240707


namespace NUMINAMATH_GPT_arithmetic_sequence_fraction_zero_l2407_240727

noncomputable def arithmetic_sequence_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_fraction_zero (a1 d : ℝ) 
    (h1 : a1 ≠ 0) (h9 : arithmetic_sequence_term a1 d 9 = 0) :
  (arithmetic_sequence_term a1 d 1 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 11 + 
   arithmetic_sequence_term a1 d 16) / 
  (arithmetic_sequence_term a1 d 7 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 14) = 0 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fraction_zero_l2407_240727


namespace NUMINAMATH_GPT_hot_dog_remainder_l2407_240751

theorem hot_dog_remainder : 35252983 % 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_hot_dog_remainder_l2407_240751


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l2407_240758

theorem sum_of_consecutive_integers (S : ℕ) (hS : S = 560):
  ∃ (N : ℕ), N = 11 ∧ 
  ∀ n (k : ℕ), 2 ≤ n → (n * (2 * k + n - 1)) = 1120 → N = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l2407_240758


namespace NUMINAMATH_GPT_sin_810_eq_one_l2407_240744

theorem sin_810_eq_one : Real.sin (810 * Real.pi / 180) = 1 :=
by
  -- You can add the proof here
  sorry

end NUMINAMATH_GPT_sin_810_eq_one_l2407_240744


namespace NUMINAMATH_GPT_roots_equation_l2407_240755

theorem roots_equation (α β : ℝ) (h1 : α^2 - 4 * α - 1 = 0) (h2 : β^2 - 4 * β - 1 = 0) :
  3 * α^3 + 4 * β^2 = 80 + 35 * α :=
by
  sorry

end NUMINAMATH_GPT_roots_equation_l2407_240755


namespace NUMINAMATH_GPT_remainder_divided_by_82_l2407_240788

theorem remainder_divided_by_82 (x : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) ↔ (∃ m : ℤ, x + 13 = 41 * m + 18) :=
by
  sorry

end NUMINAMATH_GPT_remainder_divided_by_82_l2407_240788


namespace NUMINAMATH_GPT_number_of_cupcakes_l2407_240780

theorem number_of_cupcakes (total gluten_free vegan gluten_free_vegan non_vegan : ℕ) 
    (h1 : gluten_free = total / 2)
    (h2 : vegan = 24)
    (h3 : gluten_free_vegan = vegan / 2)
    (h4 : non_vegan = 28)
    (h5 : gluten_free_vegan = gluten_free / 2) :
    total = 52 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cupcakes_l2407_240780


namespace NUMINAMATH_GPT_smallest_digit_divisible_by_11_l2407_240729

theorem smallest_digit_divisible_by_11 : ∃ d : ℕ, (0 ≤ d ∧ d ≤ 9) ∧ d = 6 ∧ (d + 7 - (4 + 3 + 6)) % 11 = 0 := by
  sorry

end NUMINAMATH_GPT_smallest_digit_divisible_by_11_l2407_240729


namespace NUMINAMATH_GPT_present_age_of_son_l2407_240743

variable (S M : ℕ)

theorem present_age_of_son :
  (M = S + 30) ∧ (M + 2 = 2 * (S + 2)) → S = 28 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_son_l2407_240743


namespace NUMINAMATH_GPT_angle_bisector_slope_l2407_240761

theorem angle_bisector_slope
  (m₁ m₂ : ℝ) (h₁ : m₁ = 2) (h₂ : m₂ = -1) (k : ℝ)
  (h_k : k = (m₁ + m₂ + Real.sqrt ((m₁ - m₂)^2 + 4)) / 2) :
  k = (1 + Real.sqrt 13) / 2 :=
by
  rw [h₁, h₂] at h_k
  sorry

end NUMINAMATH_GPT_angle_bisector_slope_l2407_240761


namespace NUMINAMATH_GPT_elena_butter_l2407_240711

theorem elena_butter (cups_flour butter : ℕ) (h1 : cups_flour * 4 = 28) (h2 : butter * 4 = 12) : butter = 3 := 
by
  sorry

end NUMINAMATH_GPT_elena_butter_l2407_240711


namespace NUMINAMATH_GPT_number_divisible_by_20p_l2407_240765

noncomputable def floor_expr (p : ℕ) : ℤ :=
  Int.floor ((2 + Real.sqrt 5) ^ p - 2 ^ (p + 1))

theorem number_divisible_by_20p (p : ℕ) (hp : Nat.Prime p ∧ p % 2 = 1) :
  ∃ k : ℤ, floor_expr p = k * 20 * p :=
by
  sorry

end NUMINAMATH_GPT_number_divisible_by_20p_l2407_240765


namespace NUMINAMATH_GPT_gcd_polynomial_l2407_240785

theorem gcd_polynomial (b : ℤ) (h : ∃ k : ℤ, b = 2 * 997 * k) : 
  Int.gcd (3 * b^2 + 34 * b + 102) (b + 21) = 21 := 
by
  -- Proof would go here, but is omitted as instructed
  sorry

end NUMINAMATH_GPT_gcd_polynomial_l2407_240785


namespace NUMINAMATH_GPT_truck_sand_amount_l2407_240760

theorem truck_sand_amount (initial_sand loss_sand final_sand : ℝ) (h1 : initial_sand = 4.1) (h2 : loss_sand = 2.4) :
  initial_sand - loss_sand = final_sand ↔ final_sand = 1.7 := 
by
  sorry

end NUMINAMATH_GPT_truck_sand_amount_l2407_240760
