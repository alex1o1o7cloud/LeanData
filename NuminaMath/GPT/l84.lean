import Mathlib

namespace average_age_of_women_l84_8433

theorem average_age_of_women (A : ℕ) (W1 W2 : ℕ) 
  (h1 : 7 * A - 26 - 30 + W1 + W2 = 7 * (A + 4)) : 
  (W1 + W2) / 2 = 42 := 
by 
  sorry

end average_age_of_women_l84_8433


namespace Adam_smiley_count_l84_8470

theorem Adam_smiley_count :
  ∃ (adam mojmir petr pavel : ℕ), adam + mojmir + petr + pavel = 52 ∧
  petr + pavel = 33 ∧ adam >= 1 ∧ mojmir >= 1 ∧ petr >= 1 ∧ pavel >= 1 ∧
  mojmir > max petr pavel ∧ adam = 1 :=
by
  sorry

end Adam_smiley_count_l84_8470


namespace average_rainfall_is_4_l84_8484

namespace VirginiaRainfall

def march_rainfall : ℝ := 3.79
def april_rainfall : ℝ := 4.5
def may_rainfall : ℝ := 3.95
def june_rainfall : ℝ := 3.09
def july_rainfall : ℝ := 4.67

theorem average_rainfall_is_4 :
  (march_rainfall + april_rainfall + may_rainfall + june_rainfall + july_rainfall) / 5 = 4 := by
  sorry

end VirginiaRainfall

end average_rainfall_is_4_l84_8484


namespace angle_value_l84_8459

theorem angle_value (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 360) 
(h3 : (Real.sin 215 * π / 180, Real.cos 215 * π / 180) = (Real.sin α, Real.cos α)) :
α = 235 :=
sorry

end angle_value_l84_8459


namespace total_string_length_l84_8426

theorem total_string_length 
  (circumference1 : ℝ) (height1 : ℝ) (loops1 : ℕ)
  (circumference2 : ℝ) (height2 : ℝ) (loops2 : ℕ)
  (h1 : circumference1 = 6) (h2 : height1 = 20) (h3 : loops1 = 5)
  (h4 : circumference2 = 3) (h5 : height2 = 10) (h6 : loops2 = 3)
  : (loops1 * Real.sqrt (circumference1 ^ 2 + (height1 / loops1) ^ 2) + loops2 * Real.sqrt (circumference2 ^ 2 + (height2 / loops2) ^ 2)) = (5 * Real.sqrt 52 + 3 * Real.sqrt 19.89) := 
by {
  sorry
}

end total_string_length_l84_8426


namespace find_x_value_l84_8449

-- Let's define the conditions
def equation (x y : ℝ) : Prop := x^2 - 4 * x + y = 0
def y_value : ℝ := 4

-- Define the theorem which states that x = 2 satisfies the conditions
theorem find_x_value (x : ℝ) (h : equation x y_value) : x = 2 :=
by
  sorry

end find_x_value_l84_8449


namespace part1_part2_l84_8432

def f (x a : ℝ) := |x - a| + 2 * |x + 1|

-- Part 1: Solve the inequality f(x) > 4 when a = 2
theorem part1 (x : ℝ) : f x 2 > 4 ↔ (x < -4/3 ∨ x > 0) := by
  sorry

-- Part 2: If the solution set of the inequality f(x) < 3x + 4 is {x | x > 2}, find the value of a.
theorem part2 (a : ℝ) : (∀ x : ℝ, (f x a < 3 * x + 4 ↔ x > 2)) → a = 6 := by
  sorry

end part1_part2_l84_8432


namespace q_minus_r_l84_8405

noncomputable def problem (x : ℝ) : Prop :=
  (5 * x - 15) / (x^2 + x - 20) = x + 3

def q_and_r (q r : ℝ) : Prop :=
  q ≠ r ∧ problem q ∧ problem r ∧ q > r

theorem q_minus_r (q r : ℝ) (h : q_and_r q r) : q - r = 2 :=
  sorry

end q_minus_r_l84_8405


namespace function_relation_l84_8498

theorem function_relation (f : ℝ → ℝ) 
  (h0 : ∀ x, f (-x) = f x)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) :
  f 0 < f (-6.5) ∧ f (-6.5) < f (-1) := 
by
  sorry

end function_relation_l84_8498


namespace area_of_enclosed_figure_l84_8463

theorem area_of_enclosed_figure:
  ∫ (x : ℝ) in (1/2)..2, x⁻¹ = 2 * Real.log 2 :=
by
  sorry

end area_of_enclosed_figure_l84_8463


namespace solution_correct_l84_8490

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop :=
  x^2 - 36 * x + 320 ≤ 16

theorem solution_correct (x : ℝ) : quadratic_inequality_solution x ↔ 16 ≤ x ∧ x ≤ 19 :=
by sorry

end solution_correct_l84_8490


namespace quotient_remainder_difference_l84_8423

theorem quotient_remainder_difference (N Q P R k : ℕ) (h1 : N = 75) (h2 : N = 5 * Q) (h3 : N = 34 * P + R) (h4 : Q = R + k) (h5 : k > 0) :
  Q - R = 8 :=
sorry

end quotient_remainder_difference_l84_8423


namespace lisa_max_non_a_quizzes_l84_8462

def lisa_goal : ℕ := 34
def quizzes_total : ℕ := 40
def quizzes_taken_first : ℕ := 25
def quizzes_with_a_first : ℕ := 20
def remaining_quizzes : ℕ := quizzes_total - quizzes_taken_first
def additional_a_needed : ℕ := lisa_goal - quizzes_with_a_first

theorem lisa_max_non_a_quizzes : 
  additional_a_needed ≤ remaining_quizzes → 
  remaining_quizzes - additional_a_needed ≤ 1 :=
by
  sorry

end lisa_max_non_a_quizzes_l84_8462


namespace A_intersection_B_complement_l84_8487

noncomputable
def universal_set : Set ℝ := Set.univ

def set_A : Set ℝ := {x | x > 1}

def set_B : Set ℝ := {y | -1 < y ∧ y < 2}

def B_complement : Set ℝ := {y | y <= -1 ∨ y >= 2}

def intersection : Set ℝ := {x | x >= 2}

theorem A_intersection_B_complement :
  (set_A ∩ B_complement) = intersection :=
  sorry

end A_intersection_B_complement_l84_8487


namespace semicircle_perimeter_l84_8483

/-- The perimeter of a semicircle with radius 6.3 cm is approximately 32.382 cm. -/
theorem semicircle_perimeter (r : ℝ) (h : r = 6.3) : 
  (π * r + 2 * r = 32.382) :=
by
  sorry

end semicircle_perimeter_l84_8483


namespace age_ratio_l84_8411

theorem age_ratio (A B C : ℕ) (h1 : A = B + 2) (h2 : A + B + C = 27) (h3 : B = 10) : B / C = 2 :=
by
  sorry

end age_ratio_l84_8411


namespace total_rainfall_correct_l84_8467

-- Define the individual rainfall amounts
def rainfall_mon1 : ℝ := 0.17
def rainfall_wed1 : ℝ := 0.42
def rainfall_fri : ℝ := 0.08
def rainfall_mon2 : ℝ := 0.37
def rainfall_wed2 : ℝ := 0.51

-- Define the total rainfall
def total_rainfall : ℝ := rainfall_mon1 + rainfall_wed1 + rainfall_fri + rainfall_mon2 + rainfall_wed2

-- Theorem statement to prove the total rainfall is 1.55 cm
theorem total_rainfall_correct : total_rainfall = 1.55 :=
by
  -- Proof goes here
  sorry

end total_rainfall_correct_l84_8467


namespace max_result_l84_8458

-- Define the expressions as Lean definitions
def expr1 : Int := 2 + (-2)
def expr2 : Int := 2 - (-2)
def expr3 : Int := 2 * (-2)
def expr4 : Int := 2 / (-2)

-- State the theorem
theorem max_result : 
  (expr2 = 4) ∧ (expr2 > expr1) ∧ (expr2 > expr3) ∧ (expr2 > expr4) :=
by
  sorry

end max_result_l84_8458


namespace equation_of_chord_line_l84_8444

theorem equation_of_chord_line (m n s t : ℝ)
  (h₀ : m > 0) (h₁ : n > 0) (h₂ : s > 0) (h₃ : t > 0)
  (h₄ : m + n = 3)
  (h₅ : m / s + n / t = 1)
  (h₆ : m < n)
  (h₇ : s + t = 3 + 2 * Real.sqrt 2)
  (h₈ : ∃ x1 x2 y1 y2 : ℝ, 
        (x1 + x2) / 2 = m ∧ (y1 + y2) / 2 = n ∧
        x1 ^ 2 / 4 + y1 ^ 2 / 16 = 1 ∧
        x2 ^ 2 / 4 + y2 ^ 2 / 16 = 1) 
  : 2 * m + n - 4 = 0 := sorry

end equation_of_chord_line_l84_8444


namespace fraction_transformation_half_l84_8403

theorem fraction_transformation_half (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  ((2 * a + 2 * b) / (4 * a^2 + 4 * b^2)) = (1 / 2) * ((a + b) / (a^2 + b^2)) :=
by sorry

end fraction_transformation_half_l84_8403


namespace arccos_cos_three_l84_8424

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi :=
  sorry

end arccos_cos_three_l84_8424


namespace odd_prime_power_condition_l84_8425

noncomputable def is_power_of (a b : ℕ) : Prop :=
  ∃ t : ℕ, b = a ^ t

theorem odd_prime_power_condition (n p x y k : ℕ) (hn : 1 < n) (hp_prime : Prime p) 
  (hp_odd : p % 2 = 1) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (hx_odd : x % 2 ≠ 0) 
  (hy_odd : y % 2 ≠ 0) (h_eq : x^n + y^n = p^k) :
  is_power_of p n :=
sorry

end odd_prime_power_condition_l84_8425


namespace number_of_buses_l84_8457

-- Definitions based on the given conditions
def vans : ℕ := 6
def people_per_van : ℕ := 6
def people_per_bus : ℕ := 18
def total_people : ℕ := 180

-- Theorem to prove the number of buses
theorem number_of_buses : 
  ∃ buses : ℕ, buses = (total_people - (vans * people_per_van)) / people_per_bus ∧ buses = 8 :=
by
  sorry

end number_of_buses_l84_8457


namespace tetrahedron_BC_squared_l84_8482

theorem tetrahedron_BC_squared (AB AC BC R r : ℝ) 
  (h1 : AB = 1) 
  (h2 : AC = 1) 
  (h3 : 1 ≤ BC) 
  (h4 : R = 4 * r) 
  (concentric : AB = AC ∧ R > 0 ∧ r > 0) :
  BC^2 = 1 + Real.sqrt (7 / 15) := 
by 
sorry

end tetrahedron_BC_squared_l84_8482


namespace ball_in_78th_position_is_green_l84_8469

-- Definition of colors in the sequence
inductive Color
| red
| yellow
| green
| blue
| violet

open Color

-- Function to compute the color of a ball at a given position within a cycle
def ball_color (n : Nat) : Color :=
  match n % 5 with
  | 0 => red    -- 78 % 5 == 3, hence 3 + 1 == 4 ==> Using 0 for red to 4 for violet
  | 1 => yellow
  | 2 => green
  | 3 => blue
  | 4 => violet
  | _ => red  -- default case, should not be reached

-- Theorem stating the desired proof problem
theorem ball_in_78th_position_is_green : ball_color 78 = green :=
by
  sorry

end ball_in_78th_position_is_green_l84_8469


namespace roots_cubic_eq_sum_fraction_l84_8471

theorem roots_cubic_eq_sum_fraction (p q r : ℝ)
  (h1 : p + q + r = 8)
  (h2 : p * q + p * r + q * r = 10)
  (h3 : p * q * r = 3) :
  p / (q * r + 2) + q / (p * r + 2) + r / (p * q + 2) = 8 / 69 := 
sorry

end roots_cubic_eq_sum_fraction_l84_8471


namespace deer_distribution_l84_8419

theorem deer_distribution :
  ∃ a : ℕ → ℚ,
    (a 1 + a 2 + a 3 + a 4 + a 5 = 5) ∧
    (a 4 = 2 / 3) ∧ 
    (a 3 = 1) ∧ 
    (a 1 = 5 / 3) :=
by
  sorry

end deer_distribution_l84_8419


namespace simplify_expression_l84_8486

variables (x y : ℝ)

theorem simplify_expression :
  (3 * x)^4 + (4 * x) * (x^3) + (5 * y)^2 = 85 * x^4 + 25 * y^2 :=
by
  sorry

end simplify_expression_l84_8486


namespace flour_amount_second_combination_l84_8421

-- Define given conditions as parameters
variables {sugar_cost flour_cost : ℝ} (sugar_per_pound flour_per_pound : ℝ)
variable (cost1 cost2 : ℝ)

axiom cost1_eq :
  40 * sugar_per_pound + 16 * flour_per_pound = cost1

axiom cost2_eq :
  30 * sugar_per_pound + flour_cost = cost2

axiom sugar_rate :
  sugar_per_pound = 0.45

axiom flour_rate :
  flour_per_pound = 0.45

-- Define the target theorem
theorem flour_amount_second_combination : ∃ flour_amount : ℝ, flour_amount = 28 := by
  sorry

end flour_amount_second_combination_l84_8421


namespace waffle_bowl_more_scoops_l84_8477

-- Definitions based on conditions
def single_cone_scoops : ℕ := 1
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def total_scoops : ℕ := 10
def remaining_scoops : ℕ := total_scoops - (banana_split_scoops + single_cone_scoops + double_cone_scoops)

-- Question: Prove that the waffle bowl has 1 more scoop than the banana split
theorem waffle_bowl_more_scoops : remaining_scoops - banana_split_scoops = 1 := by
  have h1 : single_cone_scoops = 1 := rfl
  have h2 : banana_split_scoops = 3 * single_cone_scoops := rfl
  have h3 : double_cone_scoops = 2 * single_cone_scoops := rfl
  have h4 : total_scoops = 10 := rfl
  have h5 : remaining_scoops = total_scoops - (banana_split_scoops + single_cone_scoops + double_cone_scoops) := rfl
  sorry

end waffle_bowl_more_scoops_l84_8477


namespace units_digit_of_147_pow_is_7_some_exponent_units_digit_l84_8499

theorem units_digit_of_147_pow_is_7 (n : ℕ) : (147 ^ 25) % 10 = 7 % 10 :=
by
  sorry

theorem some_exponent_units_digit (n : ℕ) (hn : n % 4 = 2) : ((147 ^ 25) ^ n) % 10 = 9 :=
by
  have base_units_digit := units_digit_of_147_pow_is_7 25
  sorry

end units_digit_of_147_pow_is_7_some_exponent_units_digit_l84_8499


namespace residue_mod_neg_935_mod_24_l84_8418

theorem residue_mod_neg_935_mod_24 : (-935) % 24 = 1 :=
by
  sorry

end residue_mod_neg_935_mod_24_l84_8418


namespace quadratic_inequality_false_iff_l84_8480

open Real

theorem quadratic_inequality_false_iff (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
sorry

end quadratic_inequality_false_iff_l84_8480


namespace parabola_equation_l84_8407

-- Define the conditions and the claim
theorem parabola_equation (p : ℝ) (hp : p > 0) (h_symmetry : -p / 2 = -1 / 2) : 
  (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = 2 * y) :=
by 
  sorry

end parabola_equation_l84_8407


namespace exists_100_digit_number_divisible_by_sum_of_digits_l84_8455

-- Definitions
def is_100_digit_number (n : ℕ) : Prop :=
  10^99 ≤ n ∧ n < 10^100

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_divisible_by_sum_of_digits (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

-- Main theorem statement
theorem exists_100_digit_number_divisible_by_sum_of_digits :
  ∃ n : ℕ, is_100_digit_number n ∧ no_zero_digits n ∧ is_divisible_by_sum_of_digits n :=
sorry

end exists_100_digit_number_divisible_by_sum_of_digits_l84_8455


namespace exist_pairs_sum_and_diff_l84_8474

theorem exist_pairs_sum_and_diff (N : ℕ) : ∃ a b c d : ℕ, 
  (a + b = c + d) ∧ (a * b + N = c * d ∨ a * b = c * d + N) := sorry

end exist_pairs_sum_and_diff_l84_8474


namespace canoe_row_probability_l84_8489

theorem canoe_row_probability :
  let p_left_works := 3 / 5
  let p_right_works := 3 / 5
  let p_left_breaks := 1 - p_left_works
  let p_right_breaks := 1 - p_right_works
  let p_can_still_row := (p_left_works * p_right_works) + (p_left_works * p_right_breaks) + (p_left_breaks * p_right_works)
  p_can_still_row = 21 / 25 :=
by
  sorry

end canoe_row_probability_l84_8489


namespace angle_sum_triangle_l84_8437

theorem angle_sum_triangle (A B C : Type) (angle_A angle_B angle_C : ℝ) 
(h1 : angle_A = 45) (h2 : angle_B = 25) 
(h3 : angle_A + angle_B + angle_C = 180) : 
angle_C = 110 := 
sorry

end angle_sum_triangle_l84_8437


namespace min_value_x_l84_8402

theorem min_value_x (a b x : ℝ) (ha : 0 < a) (hb : 0 < b)
(hcond : 4 * a + b * (1 - a) = 0)
(hineq : ∀ a b, 0 < a → 0 < b → 4 * a + b * (1 - a) = 0 → (1 / (a ^ 2) + 16 / (b ^ 2) ≥ 1 + x / 2 - x ^ 2)) :
  x = 1 :=
sorry

end min_value_x_l84_8402


namespace total_lives_l84_8431

theorem total_lives (initial_players additional_players lives_per_player : ℕ) (h1 : initial_players = 4) (h2 : additional_players = 5) (h3 : lives_per_player = 3) :
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end total_lives_l84_8431


namespace river_current_speed_l84_8434

/--
Given conditions:
- The rower realized the hat was missing 15 minutes after passing under the bridge.
- The rower caught the hat 15 minutes later.
- The total distance the hat traveled from the bridge is 1 kilometer.
Prove that the speed of the river current is 2 km/h.
-/
theorem river_current_speed (t1 t2 d : ℝ) (h_t1 : t1 = 15 / 60) (h_t2 : t2 = 15 / 60) (h_d : d = 1) : 
  d / (t1 + t2) = 2 := by
sorry

end river_current_speed_l84_8434


namespace sum_squares_l84_8409

theorem sum_squares (a b c : ℝ) (h1 : a + b + c = 22) (h2 : a * b + b * c + c * a = 116) : 
  (a^2 + b^2 + c^2 = 252) :=
by
  sorry

end sum_squares_l84_8409


namespace least_N_no_square_l84_8416

theorem least_N_no_square (N : ℕ) : 
  (∀ k, (1000 * N) ≤ k ∧ k ≤ (1000 * N + 999) → 
  ∃ m, ¬ (k = m^2)) ↔ N = 282 :=
by
  sorry

end least_N_no_square_l84_8416


namespace eval_expression_l84_8494

theorem eval_expression : 
  (20-19 + 18-17 + 16-15 + 14-13 + 12-11 + 10-9 + 8-7 + 6-5 + 4-3 + 2-1) / 
  (1-2 + 3-4 + 5-6 + 7-8 + 9-10 + 11-12 + 13-14 + 15-16 + 17-18 + 19-20) = -1 := by
  sorry

end eval_expression_l84_8494


namespace question_2024_polynomials_l84_8460

open Polynomial

noncomputable def P (x : ℝ) : Polynomial ℝ := sorry
noncomputable def Q (x : ℝ) : Polynomial ℝ := sorry

-- Main statement
theorem question_2024_polynomials (P Q : Polynomial ℝ) (hP : P.degree = 2024) (hQ : Q.degree = 2024)
    (hPm : P.leadingCoeff = 1) (hQm : Q.leadingCoeff = 1) (h : ∀ x : ℝ, P.eval x ≠ Q.eval x) :
    ∀ (α : ℝ), α ≠ 0 → ∃ x : ℝ, P.eval (x - α) = Q.eval (x + α) :=
by
  sorry

end question_2024_polynomials_l84_8460


namespace min_value_y1_minus_4y2_l84_8415

/-- 
Suppose a parabola C : y^2 = 4x intersects at points A(x1, y1) and B(x2, y2) with a line 
passing through its focus. Given that A is in the first quadrant, 
the minimum value of |y1 - 4y2| is 8.
--/
theorem min_value_y1_minus_4y2 (x1 y1 x2 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) 
  (h2 : y2^2 = 4 * x2)
  (h3 : x1 > 0) (h4 : y1 > 0) 
  (focus : (1, 0) ∈ {(x, y) | y^2 = 4 * x}) : 
  (|y1 - 4 * y2|) ≥ 8 :=
sorry

end min_value_y1_minus_4y2_l84_8415


namespace possible_integer_roots_l84_8422

-- Define the general polynomial
def polynomial (b2 b1 : ℤ) (x : ℤ) : ℤ := x ^ 3 + b2 * x ^ 2 + b1 * x - 30

-- Statement: Prove the set of possible integer roots includes exactly the divisors of -30
theorem possible_integer_roots (b2 b1 : ℤ) :
  {r : ℤ | polynomial b2 b1 r = 0} = 
  {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30} :=
sorry

end possible_integer_roots_l84_8422


namespace negation_example_l84_8414

open Real

theorem negation_example : 
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n ≥ x^2) ↔ ∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2 := 
  sorry

end negation_example_l84_8414


namespace ratio_b_c_l84_8454

theorem ratio_b_c (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : a * b * c / (d * e * f) = 0.1875)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) : 
  b / c = 3 :=
sorry

end ratio_b_c_l84_8454


namespace largest_non_sum_217_l84_8450

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end largest_non_sum_217_l84_8450


namespace find_complex_number_purely_imaginary_l84_8491

theorem find_complex_number_purely_imaginary :
  ∃ z : ℂ, (∃ b : ℝ, b ≠ 0 ∧ z = 1 + b * I) ∧ (∀ a b : ℝ, z = a + b * I → a^2 - b^2 + 3 = 0) :=
by
  -- Proof will go here
  sorry

end find_complex_number_purely_imaginary_l84_8491


namespace family_ages_l84_8438

theorem family_ages :
  ∃ (x j b m F M : ℕ), 
    (b = j - x) ∧
    (m = j - 2 * x) ∧
    (j * b = F) ∧
    (b * m = M) ∧
    (j + b + m + F + M = 90) ∧
    (F = M + x ∨ F = M - x) ∧
    (j = 6) ∧ 
    (b = 6) ∧ 
    (m = 6) ∧ 
    (F = 36) ∧ 
    (M = 36) :=
sorry

end family_ages_l84_8438


namespace cube_properties_l84_8430

theorem cube_properties (s y : ℝ) (h1 : s^3 = 8 * y) (h2 : 6 * s^2 = 6 * y) : y = 64 := by
  sorry

end cube_properties_l84_8430


namespace rectangle_is_square_l84_8420

theorem rectangle_is_square
  (a b: ℝ)  -- rectangle side lengths
  (h: a ≠ b)  -- initial assumption: rectangle not a square
  (shift_perpendicular: ∀ (P Q R S: ℝ × ℝ), (P ≠ Q → Q ≠ R → R ≠ S → S ≠ P) → (∀ (shift: ℝ × ℝ → ℝ × ℝ), ∀ (P₁: ℝ × ℝ), shift P₁ = P₁ + (0, 1) ∨ shift P₁ = P₁ + (1, 0)) → false):
  False := sorry

end rectangle_is_square_l84_8420


namespace percentage_of_democrats_l84_8453

variable (D R : ℝ)

theorem percentage_of_democrats (h1 : D + R = 100) (h2 : 0.75 * D + 0.20 * R = 53) :
  D = 60 :=
by
  sorry

end percentage_of_democrats_l84_8453


namespace purchased_both_books_l84_8400

theorem purchased_both_books: 
  ∀ (A B AB C : ℕ), A = 2 * B → AB = 2 * (B - AB) → C = 1000 → C = A - AB → AB = 500 := 
by
  intros A B AB C h1 h2 h3 h4
  sorry

end purchased_both_books_l84_8400


namespace mr_blues_yard_expectation_l84_8412

noncomputable def calculate_expected_harvest (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_feet := length_steps * step_length
  let width_feet := width_steps * step_length
  let area := length_feet * width_feet
  let total_yield := area * yield_per_sqft
  total_yield

theorem mr_blues_yard_expectation : calculate_expected_harvest 18 25 2.5 (3 / 4) = 2109.375 :=
by
  sorry

end mr_blues_yard_expectation_l84_8412


namespace escalator_steps_l84_8485

theorem escalator_steps (T : ℝ) (E : ℝ) (N : ℝ) (h1 : N - 11 = 2 * (N - 29)) : N = 47 :=
by
  sorry

end escalator_steps_l84_8485


namespace abs_inequality_solution_l84_8448

theorem abs_inequality_solution (x : ℝ) : |2 * x - 5| > 1 ↔ x < 2 ∨ x > 3 := sorry

end abs_inequality_solution_l84_8448


namespace polynomial_p_l84_8441

variable {a b c : ℝ}

theorem polynomial_p (a b c : ℝ) : 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) * 2 :=
by
  sorry

end polynomial_p_l84_8441


namespace tomatoes_left_l84_8475

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction : ℕ) (E1 : initial_tomatoes = 21) 
  (E2 : birds = 2) (E3 : fraction = 3) : 
  initial_tomatoes - initial_tomatoes / fraction = 14 :=
by 
  sorry

end tomatoes_left_l84_8475


namespace car_speed_l84_8451

theorem car_speed (v : ℝ) (h1 : 1 / 900 * 3600 = 4) (h2 : 1 / v * 3600 = 6) : v = 600 :=
by
  sorry

end car_speed_l84_8451


namespace find_number_l84_8496

theorem find_number (x : ℝ) (h : (3.242 * 16) / x = 0.051871999999999995) : x = 1000 :=
by
  sorry

end find_number_l84_8496


namespace product_sequence_eq_l84_8408

theorem product_sequence_eq :
  let seq := [ (1 : ℚ) / 2, 4 / 1, 1 / 8, 16 / 1, 1 / 32, 64 / 1,
               1 / 128, 256 / 1, 1 / 512, 1024 / 1, 1 / 2048, 4096 / 1 ]
  (seq.prod) * (3 / 4) = 1536 := by 
  -- expand and simplify the series of products
  sorry 

end product_sequence_eq_l84_8408


namespace shortest_track_length_l84_8428

open Nat

def Melanie_track_length := 8
def Martin_track_length := 20

theorem shortest_track_length :
  Nat.lcm Melanie_track_length Martin_track_length = 40 :=
by
  sorry

end shortest_track_length_l84_8428


namespace sum_is_integer_l84_8472

theorem sum_is_integer (x y z : ℝ) (h1 : x ^ 2 = y + 2) (h2 : y ^ 2 = z + 2) (h3 : z ^ 2 = x + 2) : ∃ n : ℤ, x + y + z = n :=
  sorry

end sum_is_integer_l84_8472


namespace number_of_possible_values_b_l84_8410

theorem number_of_possible_values_b : 
  ∃ n : ℕ, n = 2 ∧ 
    (∀ b : ℕ, b ≥ 2 → (b^3 ≤ 256) ∧ (256 < b^4) ↔ (b = 5 ∨ b = 6)) :=
by {
  sorry
}

end number_of_possible_values_b_l84_8410


namespace cookies_per_bag_l84_8435

-- Definitions of the given conditions
def c1 := 23  -- number of chocolate chip cookies
def c2 := 25  -- number of oatmeal cookies
def b := 8    -- number of baggies

-- Statement to prove
theorem cookies_per_bag : (c1 + c2) / b = 6 :=
by 
  sorry

end cookies_per_bag_l84_8435


namespace complex_roots_circle_radius_l84_8406

theorem complex_roots_circle_radius (z : ℂ) (h : (z + 2)^4 = 16 * z^4) :
  ∃ r : ℝ, (∀ z, (z + 2)^4 = 16 * z^4 → (z - (2/3))^2 + y^2 = r) ∧ r = 1 :=
sorry

end complex_roots_circle_radius_l84_8406


namespace ratio_e_to_f_l84_8476

theorem ratio_e_to_f {a b c d e f : ℝ}
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.75) :
  e / f = 0.5 :=
sorry

end ratio_e_to_f_l84_8476


namespace picnic_weather_condition_l84_8447

variables (P Q : Prop)

theorem picnic_weather_condition (h : ¬P → ¬Q) : Q → P := 
by sorry

end picnic_weather_condition_l84_8447


namespace giselle_initial_doves_l84_8473

theorem giselle_initial_doves (F : ℕ) (h1 : ∀ F, F > 0) (h2 : 3 * F * 3 / 4 + F = 65) : F = 20 :=
sorry

end giselle_initial_doves_l84_8473


namespace calculate_neg2_add3_l84_8479

theorem calculate_neg2_add3 : (-2) + 3 = 1 :=
  sorry

end calculate_neg2_add3_l84_8479


namespace true_statement_l84_8401

def statement_i (i : ℕ) (n : ℕ) : Prop := 
  (i = (n - 1))

theorem true_statement :
  ∃! n : ℕ, (n ≤ 100 ∧ ∀ i, (i ≠ n - 1) → statement_i i n = false) ∧ statement_i (n - 1) n = true :=
by
  sorry

end true_statement_l84_8401


namespace cannot_tile_surface_square_hexagon_l84_8493

-- Definitions of internal angles of the tile shapes
def internal_angle_triangle := 60
def internal_angle_square := 90
def internal_angle_hexagon := 120
def internal_angle_octagon := 135

-- The theorem to prove that square and hexagon cannot tile a surface without gaps or overlaps
theorem cannot_tile_surface_square_hexagon : ∀ (m n : ℕ), internal_angle_square * m + internal_angle_hexagon * n ≠ 360 := 
by sorry

end cannot_tile_surface_square_hexagon_l84_8493


namespace set_union_example_l84_8439

open Set

/-- Given sets A = {1, 2, 3} and B = {-1, 1}, prove that A ∪ B = {-1, 1, 2, 3} -/
theorem set_union_example : 
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  A ∪ B = ({-1, 1, 2, 3} : Set ℤ) :=
by
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  show A ∪ B = ({-1, 1, 2, 3} : Set ℤ)
  -- Proof to be provided here
  sorry

end set_union_example_l84_8439


namespace inequality_incorrect_l84_8456

theorem inequality_incorrect (a b : ℝ) (h : a > b) : ¬(-a + 2 > -b + 2) :=
by
  sorry

end inequality_incorrect_l84_8456


namespace find_x_when_y_neg_five_l84_8488

-- Definitions based on the conditions provided
variable (x y : ℝ)
def inversely_proportional (x y : ℝ) := ∃ (k : ℝ), x * y = k

-- Proving the main result
theorem find_x_when_y_neg_five (h_prop : inversely_proportional x y) (hx4 : x = 4) (hy2 : y = 2) :
    (y = -5) → x = - 8 / 5 := by
  sorry

end find_x_when_y_neg_five_l84_8488


namespace club_members_after_four_years_l84_8445

theorem club_members_after_four_years
  (b : ℕ → ℕ)
  (h_initial : b 0 = 20)
  (h_recursive : ∀ k, b (k + 1) = 3 * (b k) - 10) :
  b 4 = 1220 :=
sorry

end club_members_after_four_years_l84_8445


namespace greatest_integer_equality_l84_8478

theorem greatest_integer_equality (m : ℝ) (h : m ≥ 3) :
  Int.floor ((m * (m + 1)) / (2 * (2 * m - 1))) = Int.floor ((m + 1) / 4) :=
  sorry

end greatest_integer_equality_l84_8478


namespace one_minus_repeating_six_l84_8495

noncomputable def repeating_six : Real := 2 / 3

theorem one_minus_repeating_six : 1 - repeating_six = 1 / 3 :=
by
  sorry

end one_minus_repeating_six_l84_8495


namespace min_x2_y2_eq_16_then_product_zero_l84_8417

theorem min_x2_y2_eq_16_then_product_zero
  (x y : ℝ)
  (h1 : ∃ x y : ℝ, (x^2 + y^2 = 16 ∧ ∀ a b : ℝ, a^2 + b^2 ≥ 16) ) :
  (x + 4) * (y - 4) = 0 := 
sorry

end min_x2_y2_eq_16_then_product_zero_l84_8417


namespace exists_bound_for_expression_l84_8492

theorem exists_bound_for_expression :
  ∃ (C : ℝ), (∀ (k : ℤ), abs ((k^8 - 2*k + 1 : ℤ) / (k^4 - 3 : ℤ)) < C) := 
sorry

end exists_bound_for_expression_l84_8492


namespace percentage_of_students_who_received_certificates_l84_8481

theorem percentage_of_students_who_received_certificates
  (total_boys : ℕ)
  (total_girls : ℕ)
  (perc_boys_certificates : ℝ)
  (perc_girls_certificates : ℝ)
  (h1 : total_boys = 30)
  (h2 : total_girls = 20)
  (h3 : perc_boys_certificates = 0.1)
  (h4 : perc_girls_certificates = 0.2)
  : (3 + 4) / (30 + 20) * 100 = 14 :=
by
  sorry

end percentage_of_students_who_received_certificates_l84_8481


namespace farm_field_area_l84_8465

variable (A D : ℕ)

theorem farm_field_area
  (h1 : 160 * D = A)
  (h2 : 85 * (D + 2) + 40 = A) :
  A = 480 :=
by
  sorry

end farm_field_area_l84_8465


namespace find_constants_l84_8436

theorem find_constants (a b c : ℝ) (h_neg : a < 0) (h_amp : |a| = 3) (h_period : b > 0 ∧ (2 * π / b) = 8 * π) : 
a = -3 ∧ b = 0.5 :=
by
  sorry

end find_constants_l84_8436


namespace even_function_l84_8464

-- Definition of f and F with the given conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Condition that x is in the interval (-a, a)
def in_interval (a x : ℝ) : Prop := x > -a ∧ x < a

-- Definition of F(x)
def F (x : ℝ) : ℝ := f x + f (-x)

-- The proposition that we want to prove
theorem even_function (h : in_interval a x) : F f x = F f (-x) :=
by
  unfold F
  sorry

end even_function_l84_8464


namespace number_of_straight_A_students_l84_8452

-- Define the initial conditions and numbers
variables {x y : ℕ}

-- Define the initial student count and conditions on percentages
def initial_student_count := 25
def new_student_count := 7
def total_student_count := initial_student_count + new_student_count
def initial_percentage (x : ℕ) := (x : ℚ) / initial_student_count * 100
def new_percentage (x y : ℕ) := ((x + y : ℚ) / total_student_count) * 100

theorem number_of_straight_A_students
  (x y : ℕ)
  (h : initial_percentage x + 10 = new_percentage x y) :
  (x + y = 16) :=
sorry

end number_of_straight_A_students_l84_8452


namespace gcd_three_digit_palindromes_l84_8404

theorem gcd_three_digit_palindromes : 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → 
  ∃ d : ℕ, d = 1 ∧ ∀ n m : ℕ, (n = 101 * a + 10 * b) → (m = 101 * a + 10 * b) → gcd n m = d := 
by sorry

end gcd_three_digit_palindromes_l84_8404


namespace find_n_l84_8446

theorem find_n : ∃ n : ℕ, 2^7 * 3^3 * 5 * n = Nat.factorial 12 ∧ n = 27720 :=
by
  use 27720
  have h1 : 2^7 * 3^3 * 5 * 27720 = Nat.factorial 12 :=
  sorry -- This will be the place to prove the given equation eventually.
  exact ⟨h1, rfl⟩

end find_n_l84_8446


namespace sum_of_all_possible_values_of_N_with_equation_l84_8466

def satisfiesEquation (N : ℝ) : Prop :=
  N * (N - 4) = -7

theorem sum_of_all_possible_values_of_N_with_equation :
  (∀ N, satisfiesEquation N → N + (4 - N) = 4) :=
sorry

end sum_of_all_possible_values_of_N_with_equation_l84_8466


namespace drawing_two_black_balls_probability_equals_half_l84_8442

noncomputable def total_number_of_events : ℕ := 6

noncomputable def number_of_black_draw_events : ℕ := 3

noncomputable def probability_of_drawing_two_black_balls : ℚ :=
  number_of_black_draw_events / total_number_of_events

theorem drawing_two_black_balls_probability_equals_half :
  probability_of_drawing_two_black_balls = 1 / 2 :=
by
  sorry

end drawing_two_black_balls_probability_equals_half_l84_8442


namespace parabola_focus_line_ratio_l84_8497

noncomputable def ratio_AF_BF : ℝ := (Real.sqrt 5 + 3) / 2

theorem parabola_focus_line_ratio :
  ∀ (F A B : ℝ × ℝ), 
    F = (1, 0) ∧ 
    (A.2 = 2 * A.1 - 2 ∧ A.2^2 = 4 * A.1 ) ∧ 
    (B.2 = 2 * B.1 - 2 ∧ B.2^2 = 4 * B.1) ∧ 
    A.2 > 0 -> 
  |(A.1 - F.1) / (B.1 - F.1)| = ratio_AF_BF :=
by
  sorry

end parabola_focus_line_ratio_l84_8497


namespace find_water_and_bucket_weight_l84_8429

-- Define the original amount of water (x) and the weight of the bucket (y)
variables (x y : ℝ)

-- Given conditions described as hypotheses
def conditions (x y : ℝ) : Prop :=
  4 * x + y = 16 ∧ 6 * x + y = 22

-- The goal is to prove the values of x and y
theorem find_water_and_bucket_weight (h : conditions x y) : x = 3 ∧ y = 4 :=
by
  sorry

end find_water_and_bucket_weight_l84_8429


namespace apple_price_33kg_l84_8461

theorem apple_price_33kg
  (l q : ℝ)
  (h1 : 10 * l = 3.62)
  (h2 : 30 * l + 6 * q = 12.48) :
  30 * l + 3 * q = 11.67 :=
by
  sorry

end apple_price_33kg_l84_8461


namespace gcd_256_180_600_l84_8440

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l84_8440


namespace smallest_number_of_students_l84_8427

theorem smallest_number_of_students (n9 n7 n8 : ℕ) (h7 : 9 * n7 = 7 * n9) (h8 : 5 * n8 = 9 * n9) :
  n9 + n7 + n8 = 134 :=
by
  -- Skipping proof with sorry
  sorry

end smallest_number_of_students_l84_8427


namespace find_y_values_l84_8443

variable (x y : ℝ)

theorem find_y_values 
    (h1 : 3 * x^2 + 9 * x + 4 * y - 2 = 0)
    (h2 : 3 * x + 2 * y - 6 = 0) : 
    y^2 - 13 * y + 26 = 0 := by
  sorry

end find_y_values_l84_8443


namespace line_b_y_intercept_l84_8468

theorem line_b_y_intercept :
  ∃ c : ℝ, (∀ x : ℝ, (-3) * x + c = -3 * x + 7) ∧ ∃ p : ℝ × ℝ, (p = (5, -2)) → -3 * 5 + c = -2 →
  c = 13 :=
by
  sorry

end line_b_y_intercept_l84_8468


namespace square_area_l84_8413

-- Definition of the vertices' coordinates
def y_coords := ({-3, 2, 2, -3} : Set ℤ)
def x_coords_when_y2 := ({0, 5} : Set ℤ)

-- The statement we need to prove
theorem square_area (h1 : y_coords = {-3, 2, 2, -3}) 
                     (h2 : x_coords_when_y2 = {0, 5}) : 
                     ∃ s : ℤ, s^2 = 25 :=
by
  sorry

end square_area_l84_8413
