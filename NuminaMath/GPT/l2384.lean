import Mathlib

namespace NUMINAMATH_GPT_score_of_B_is_correct_l2384_238454

theorem score_of_B_is_correct (A B C D E : ℝ)
  (h1 : (A + B + C + D + E) / 5 = 90)
  (h2 : (A + B + C) / 3 = 86)
  (h3 : (B + D + E) / 3 = 95) : 
  B = 93 := 
by 
  sorry

end NUMINAMATH_GPT_score_of_B_is_correct_l2384_238454


namespace NUMINAMATH_GPT_triangle_inequality_x_not_2_l2384_238428

theorem triangle_inequality_x_not_2 (x : ℝ) (h1 : 2 < x) (h2 : x < 8) : x ≠ 2 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_inequality_x_not_2_l2384_238428


namespace NUMINAMATH_GPT_min_ball_count_required_l2384_238478

def is_valid_ball_count (n : ℕ) : Prop :=
  n >= 11 ∧ n ≠ 17 ∧ n % 6 ≠ 0

def distinct_list (l : List ℕ) : Prop :=
  ∀ i j, i < l.length → j < l.length → i ≠ j → l.nthLe i sorry ≠ l.nthLe j sorry

def valid_ball_counts_list (l : List ℕ) : Prop :=
  (l.length = 10) ∧ distinct_list l ∧ (∀ n ∈ l, is_valid_ball_count n)

theorem min_ball_count_required : ∃ l, valid_ball_counts_list l ∧ l.sum = 174 := sorry

end NUMINAMATH_GPT_min_ball_count_required_l2384_238478


namespace NUMINAMATH_GPT_gcd_12a_20b_min_value_l2384_238461

-- Define the conditions
def is_positive_integer (x : ℕ) : Prop := x > 0

def gcd_condition (a b d : ℕ) : Prop := gcd a b = d

-- State the problem
theorem gcd_12a_20b_min_value (a b : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_gcd_ab : gcd_condition a b 10) :
  ∃ (k : ℕ), k = gcd (12 * a) (20 * b) ∧ k = 40 :=
by
  sorry

end NUMINAMATH_GPT_gcd_12a_20b_min_value_l2384_238461


namespace NUMINAMATH_GPT_intersection_is_correct_l2384_238494

noncomputable def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
noncomputable def B := { x : ℝ | 0 < x ∧ x ≤ 3 }

theorem intersection_is_correct : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l2384_238494


namespace NUMINAMATH_GPT_compare_logarithmic_values_l2384_238497

theorem compare_logarithmic_values :
  let a := Real.log 3.4 / Real.log 2
  let b := Real.log 3.6 / Real.log 4
  let c := Real.log 0.3 / Real.log 3
  c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_GPT_compare_logarithmic_values_l2384_238497


namespace NUMINAMATH_GPT_find_fraction_of_cistern_l2384_238425

noncomputable def fraction_initially_full (x : ℝ) : Prop :=
  let rateA := (1 - x) / 12
  let rateB := (1 - x) / 8
  let combined_rate := 1 / 14.4
  combined_rate = rateA + rateB

theorem find_fraction_of_cistern {x : ℝ} (h : fraction_initially_full x) : x = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_of_cistern_l2384_238425


namespace NUMINAMATH_GPT_no_prime_sum_10003_l2384_238468

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end NUMINAMATH_GPT_no_prime_sum_10003_l2384_238468


namespace NUMINAMATH_GPT_part1_part2_l2384_238462

noncomputable def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

theorem part1 (x : ℝ) : f x < 4 ↔ -1 < x ∧ x < (5:ℝ)/3 := by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≥ |a + 1|) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2384_238462


namespace NUMINAMATH_GPT_sum_of_first_3n_terms_l2384_238488

variable {S : ℕ → ℝ}
variable {n : ℕ}
variable {a b : ℝ}

def arithmetic_sum (S : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, S (m + 1) = S m + (d * (m + 1))

theorem sum_of_first_3n_terms (h1 : S n = a) (h2 : S (2 * n) = b) 
  (h3 : arithmetic_sum S) : S (3 * n) = 3 * b - 2 * a :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_3n_terms_l2384_238488


namespace NUMINAMATH_GPT_binary_to_decimal_l2384_238415

theorem binary_to_decimal : (11010 : ℕ) = 26 := by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l2384_238415


namespace NUMINAMATH_GPT_directrix_parabola_y_eq_2x2_l2384_238458

theorem directrix_parabola_y_eq_2x2 : (∃ y : ℝ, y = 2 * x^2) → (∃ y : ℝ, y = -1/8) :=
by
  sorry

end NUMINAMATH_GPT_directrix_parabola_y_eq_2x2_l2384_238458


namespace NUMINAMATH_GPT_triangle_cannot_have_two_right_angles_l2384_238411

theorem triangle_cannot_have_two_right_angles (A B C : ℝ) (h : A + B + C = 180) : 
  ¬ (A = 90 ∧ B = 90) :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_cannot_have_two_right_angles_l2384_238411


namespace NUMINAMATH_GPT_sum_series_eq_two_l2384_238426

noncomputable def series_term (n : ℕ) : ℚ := (3 * n - 2) / (n * (n + 1) * (n + 2))

theorem sum_series_eq_two :
  ∑' n : ℕ, series_term (n + 1) = 2 :=
sorry

end NUMINAMATH_GPT_sum_series_eq_two_l2384_238426


namespace NUMINAMATH_GPT_relatively_prime_bound_l2384_238406

theorem relatively_prime_bound {m n : ℕ} {a : ℕ → ℕ} (h1 : 1 < m) (h2 : 1 < n) (h3 : m ≥ n)
  (h4 : ∀ i j, i ≠ j → a i = a j → False) (h5 : ∀ i, a i ≤ m) (h6 : ∀ i j, i ≠ j → a i ∣ a j → a i = 1) 
  (x : ℝ) : ∃ i, dist (a i * x) (round (a i * x)) ≥ 2 / (m * (m + 1)) * dist x (round x) :=
sorry

end NUMINAMATH_GPT_relatively_prime_bound_l2384_238406


namespace NUMINAMATH_GPT_find_b_minus_a_l2384_238410

theorem find_b_minus_a (a b : ℤ) (h1 : a * b = 2 * (a + b) + 11) (h2 : b = 7) : b - a = 2 :=
by sorry

end NUMINAMATH_GPT_find_b_minus_a_l2384_238410


namespace NUMINAMATH_GPT_sector_area_l2384_238480

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = 2 * Real.pi / 5) (hr : r = 20) :
  1 / 2 * r^2 * θ = 80 * Real.pi := by
  sorry

end NUMINAMATH_GPT_sector_area_l2384_238480


namespace NUMINAMATH_GPT_H2O_formed_l2384_238435

-- Definition of the balanced chemical equation
def balanced_eqn : Prop :=
  ∀ (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ), HCH3CO2 + NaOH = NaCH3CO2 + H2O

-- Statement of the problem
theorem H2O_formed (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ) 
  (h1 : HCH3CO2 = 1)
  (h2 : NaOH = 1)
  (balanced : balanced_eqn):
  H2O = 1 :=
by sorry

end NUMINAMATH_GPT_H2O_formed_l2384_238435


namespace NUMINAMATH_GPT_expected_number_of_own_hats_l2384_238419

-- Define the number of people
def num_people : ℕ := 2015

-- Define the expectation based on the problem description
noncomputable def expected_hats (n : ℕ) : ℝ := 1

-- The main theorem representing the problem statement
theorem expected_number_of_own_hats : expected_hats num_people = 1 := sorry

end NUMINAMATH_GPT_expected_number_of_own_hats_l2384_238419


namespace NUMINAMATH_GPT_jose_bottle_caps_l2384_238445

def jose_start : ℕ := 7
def rebecca_gives : ℕ := 2
def final_bottle_caps : ℕ := 9

theorem jose_bottle_caps :
  jose_start + rebecca_gives = final_bottle_caps :=
by
  sorry

end NUMINAMATH_GPT_jose_bottle_caps_l2384_238445


namespace NUMINAMATH_GPT_shifted_parabola_eq_l2384_238414

-- Definitions
def original_parabola (x y : ℝ) : Prop := y = 3 * x^2

def shifted_origin (x' y' x y : ℝ) : Prop :=
  (x' = x + 1) ∧ (y' = y + 1)

-- Target statement
theorem shifted_parabola_eq : ∀ (x y x' y' : ℝ),
  original_parabola x y →
  shifted_origin x' y' x y →
  y' = 3*(x' - 1)*(x' - 1) + 1 → 
  y = 3*(x + 1)*(x + 1) - 1 :=
by
  intros x y x' y' h_orig h_shifted h_new_eq
  sorry

end NUMINAMATH_GPT_shifted_parabola_eq_l2384_238414


namespace NUMINAMATH_GPT_increase_80_by_135_percent_l2384_238443

theorem increase_80_by_135_percent : 
  let original := 80 
  let increase := 1.35 
  original + (increase * original) = 188 := 
by
  sorry

end NUMINAMATH_GPT_increase_80_by_135_percent_l2384_238443


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l2384_238444

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (hroots : ∀ x, (x = p ∨ x = q ∨ x = r) ↔ (30*x^3 - 50*x^2 + 22*x - 1 = 0)) 
  (h0 : 0 < p ∧ p < 1) (h1 : 0 < q ∧ q < 1) (h2 : 0 < r ∧ r < 1) 
  (hdistinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (1 / (1 - p)) + (1 / (1 - q)) + (1 / (1 - r)) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l2384_238444


namespace NUMINAMATH_GPT_jenna_interest_l2384_238437

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

def interest_earned (P r : ℝ) (n : ℕ) : ℝ :=
  compound_interest P r n - P

theorem jenna_interest :
  interest_earned 1500 0.05 5 = 414.42 :=
by
  sorry

end NUMINAMATH_GPT_jenna_interest_l2384_238437


namespace NUMINAMATH_GPT_art_museum_visitors_l2384_238452

theorem art_museum_visitors 
  (V : ℕ)
  (H1 : ∃ (d : ℕ), d = 130)
  (H2 : ∃ (e u : ℕ), e = u)
  (H3 : ∃ (x : ℕ), x = (3 * V) / 4)
  (H4 : V = (3 * V) / 4 + 130) :
  V = 520 :=
sorry

end NUMINAMATH_GPT_art_museum_visitors_l2384_238452


namespace NUMINAMATH_GPT_evaluate_expression_l2384_238417

theorem evaluate_expression :
  500 * 997 * 0.0997 * 10^2 = 5 * (997:ℝ)^2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2384_238417


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2384_238447

theorem solution_set_of_quadratic_inequality (a : ℝ) (x : ℝ) :
  (∀ x, 0 < x - 0.5 ∧ x < 2 → ax^2 + 5 * x - 2 > 0) ∧ a = -2 →
  (∀ x, -3 < x ∧ x < 0.5 → a * x^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2384_238447


namespace NUMINAMATH_GPT_triangle_isosceles_l2384_238466

theorem triangle_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) :
  b = c → IsoscelesTriangle := 
by
  sorry

end NUMINAMATH_GPT_triangle_isosceles_l2384_238466


namespace NUMINAMATH_GPT_finite_discrete_points_3_to_15_l2384_238431

def goldfish_cost (n : ℕ) : ℕ := 18 * n

theorem finite_discrete_points_3_to_15 : 
  ∀ (n : ℕ), 3 ≤ n ∧ n ≤ 15 → 
  ∃ (C : ℕ), C = goldfish_cost n ∧ ∃ (x : ℕ), (n, C) = (x, goldfish_cost x) :=
by
  sorry

end NUMINAMATH_GPT_finite_discrete_points_3_to_15_l2384_238431


namespace NUMINAMATH_GPT_part1_part2_l2384_238465

-- Definitions
def A (x : ℝ) : Prop := (x + 2) / (x - 3 / 2) < 0
def B (x : ℝ) (m : ℝ) : Prop := x^2 - (m + 1) * x + m ≤ 0

-- Part (1): when m = 2, find A ∪ B
theorem part1 :
  (∀ x, A x ∨ B x 2) ↔ ∀ x, -2 < x ∧ x ≤ 2 := sorry

-- Part (2): find the range of real number m
theorem part2 :
  (∀ x, A x → B x m) ↔ (-2 < m ∧ m < 3 / 2) := sorry

end NUMINAMATH_GPT_part1_part2_l2384_238465


namespace NUMINAMATH_GPT_min_area_rectangle_l2384_238486

theorem min_area_rectangle (P : ℕ) (hP : P = 60) :
  ∃ (l w : ℕ), 2 * l + 2 * w = P ∧ l * w = 29 :=
by
  sorry

end NUMINAMATH_GPT_min_area_rectangle_l2384_238486


namespace NUMINAMATH_GPT_domain_w_l2384_238474

noncomputable def w (y : ℝ) : ℝ := (y - 3)^(1/3) + (15 - y)^(1/3)

theorem domain_w : ∀ y : ℝ, ∃ x : ℝ, w y = x := by
  sorry

end NUMINAMATH_GPT_domain_w_l2384_238474


namespace NUMINAMATH_GPT_volume_of_cuboid_l2384_238493

theorem volume_of_cuboid (a b c : ℕ) (h_a : a = 2) (h_b : b = 5) (h_c : c = 8) : 
  a * b * c = 80 := 
by 
  sorry

end NUMINAMATH_GPT_volume_of_cuboid_l2384_238493


namespace NUMINAMATH_GPT_baker_sold_more_cakes_than_pastries_l2384_238422

theorem baker_sold_more_cakes_than_pastries (cakes_sold pastries_sold : ℕ) 
  (h_cakes_sold : cakes_sold = 158) (h_pastries_sold : pastries_sold = 147) : 
  (cakes_sold - pastries_sold) = 11 := by
  sorry

end NUMINAMATH_GPT_baker_sold_more_cakes_than_pastries_l2384_238422


namespace NUMINAMATH_GPT_total_skateboarding_distance_l2384_238476

def skateboarded_to_park : ℕ := 16
def skateboarded_back_home : ℕ := 9

theorem total_skateboarding_distance : 
  skateboarded_to_park + skateboarded_back_home = 25 := by 
  sorry

end NUMINAMATH_GPT_total_skateboarding_distance_l2384_238476


namespace NUMINAMATH_GPT_g_f_neg3_eq_1741_l2384_238400

def f (x : ℤ) : ℤ := x^3 - 3
def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg3_eq_1741 : g (f (-3)) = 1741 := 
by 
  sorry

end NUMINAMATH_GPT_g_f_neg3_eq_1741_l2384_238400


namespace NUMINAMATH_GPT_sqrt_of_4_l2384_238495

theorem sqrt_of_4 : {x : ℤ | x^2 = 4} = {2, -2} :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_4_l2384_238495


namespace NUMINAMATH_GPT_problem_statement_l2384_238434

def approx_digit_place (num : ℕ) : ℕ :=
if num = 3020000 then 0 else sorry

theorem problem_statement :
  approx_digit_place (3 * 10^6 + 2 * 10^4) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2384_238434


namespace NUMINAMATH_GPT_length_of_AB_l2384_238429

-- Conditions:
-- The radius of the inscribed circle is 6 cm.
-- The triangle is a right triangle with a 60 degree angle at one vertex.
-- Question: Prove that the length of AB is 12 + 12√3 cm.

theorem length_of_AB (r : ℝ) (angle : ℝ) (h_radius : r = 6) (h_angle : angle = 60) :
  ∃ (AB : ℝ), AB = 12 + 12 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l2384_238429


namespace NUMINAMATH_GPT_find_fraction_l2384_238499

variable (x : ℝ) (f : ℝ)
axiom thirty_percent_of_x : 0.30 * x = 63.0000000000001
axiom fraction_condition : f = 0.40 * x + 12

theorem find_fraction : f = 96 := by
  sorry

end NUMINAMATH_GPT_find_fraction_l2384_238499


namespace NUMINAMATH_GPT_b_investment_l2384_238407

theorem b_investment (x : ℝ) (total_profit A_investment B_investment C_investment A_profit: ℝ)
  (h1 : A_investment = 6300)
  (h2 : B_investment = x)
  (h3 : C_investment = 10500)
  (h4 : total_profit = 12600)
  (h5 : A_profit = 3780)
  (ratio_eq : (A_investment / (A_investment + B_investment + C_investment)) = (A_profit / total_profit)) :
  B_investment = 13700 :=
  sorry

end NUMINAMATH_GPT_b_investment_l2384_238407


namespace NUMINAMATH_GPT_boys_without_calculators_l2384_238412

/-- In Mrs. Robinson's math class, there are 20 boys, and 30 of her students bring their calculators to class. 
    If 18 of the students who brought calculators are girls, then the number of boys who didn't bring their calculators is 8. -/
theorem boys_without_calculators (num_boys : ℕ) (num_students_with_calculators : ℕ) (num_girls_with_calculators : ℕ)
  (h1 : num_boys = 20)
  (h2 : num_students_with_calculators = 30)
  (h3 : num_girls_with_calculators = 18) :
  num_boys - (num_students_with_calculators - num_girls_with_calculators) = 8 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_boys_without_calculators_l2384_238412


namespace NUMINAMATH_GPT_anna_and_bob_play_together_l2384_238402

-- Definitions based on the conditions
def total_players := 12
def matches_per_week := 2
def players_per_match := 6
def anna_and_bob := 2
def other_players := total_players - anna_and_bob
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Lean statement based on the equivalent proof problem
theorem anna_and_bob_play_together :
  combination other_players (players_per_match - anna_and_bob) = 210 := by
  -- To use Binomial Theorem in Lean
  -- The mathematical equivalent is C(10, 4) = 210
  sorry

end NUMINAMATH_GPT_anna_and_bob_play_together_l2384_238402


namespace NUMINAMATH_GPT_LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l2384_238469

noncomputable section

-- Problem 1: Prove length ratios for simultaneous ignition
def LengthRatioSimultaneous (t : ℝ) : Prop :=
  let LA := 1 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosSimultaneous (t : ℝ) : LengthRatioSimultaneous t := sorry

-- Problem 2: Prove length ratios when one candle is lit 30 minutes earlier
def LengthRatioNonSimultaneous (t : ℝ) : Prop :=
  let LA := 5 / 6 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosNonSimultaneous (t : ℝ) : LengthRatioNonSimultaneous t := sorry

end NUMINAMATH_GPT_LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l2384_238469


namespace NUMINAMATH_GPT_share_price_increase_l2384_238457

theorem share_price_increase
  (P : ℝ)
  -- At the end of the first quarter, the share price was 20% higher than at the beginning of the year.
  (end_of_first_quarter : ℝ := 1.20 * P)
  -- The percent increase from the end of the first quarter to the end of the second quarter was 25%.
  (percent_increase_second_quarter : ℝ := 0.25)
  -- At the end of the second quarter, the share price
  (end_of_second_quarter : ℝ := end_of_first_quarter + percent_increase_second_quarter * end_of_first_quarter) :
  -- What is the percent increase in share price at the end of the second quarter compared to the beginning of the year?
  end_of_second_quarter = 1.50 * P :=
by
  sorry

end NUMINAMATH_GPT_share_price_increase_l2384_238457


namespace NUMINAMATH_GPT_part1_part2_part3_l2384_238475

variables (a b c : ℤ)
-- Condition: For all integer values of x, (ax^2 + bx + c) is a square number 
def quadratic_is_square_for_any_x (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

-- Question (1): Prove that 2a, 2b, c are all integers
theorem part1 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ m n : ℤ, 2 * a = m ∧ 2 * b = n ∧ ∃ k₁ : ℤ, c = k₁ :=
sorry

-- Question (2): Prove that a, b, c are all integers, and c is a square number
theorem part2 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2 :=
sorry

-- Question (3): Prove that if (2) holds, it does not necessarily mean that 
-- for all integer values of x, (ax^2 + bx + c) is always a square number.
theorem part3 (a b c : ℤ) (h : ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2) : 
  ¬ quadratic_is_square_for_any_x a b c :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l2384_238475


namespace NUMINAMATH_GPT_base_seven_to_ten_l2384_238472

theorem base_seven_to_ten :
  4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0 = 10738 :=
by
  sorry

end NUMINAMATH_GPT_base_seven_to_ten_l2384_238472


namespace NUMINAMATH_GPT_largest_value_l2384_238404

def value (word : List Char) : Nat :=
  word.foldr (fun c acc =>
    acc + match c with
      | 'A' => 1
      | 'B' => 2
      | 'C' => 3
      | 'D' => 4
      | 'E' => 5
      | _ => 0
    ) 0

theorem largest_value :
  value ['B', 'E', 'E'] > value ['D', 'A', 'D'] ∧
  value ['B', 'E', 'E'] > value ['B', 'A', 'D'] ∧
  value ['B', 'E', 'E'] > value ['C', 'A', 'B'] ∧
  value ['B', 'E', 'E'] > value ['B', 'E', 'D'] :=
by sorry

end NUMINAMATH_GPT_largest_value_l2384_238404


namespace NUMINAMATH_GPT_wendy_furniture_time_l2384_238460

variable (chairs tables pieces minutes total_time : ℕ)

theorem wendy_furniture_time (h1 : chairs = 4) (h2 : tables = 4) (h3 : pieces = chairs + tables) (h4 : minutes = 6) (h5 : total_time = pieces * minutes) : total_time = 48 :=
by
  sorry

end NUMINAMATH_GPT_wendy_furniture_time_l2384_238460


namespace NUMINAMATH_GPT_min_max_pieces_three_planes_l2384_238423

theorem min_max_pieces_three_planes : 
  ∃ (min max : ℕ), (min = 4) ∧ (max = 8) := by
  sorry

end NUMINAMATH_GPT_min_max_pieces_three_planes_l2384_238423


namespace NUMINAMATH_GPT_product_of_g_at_roots_l2384_238489

noncomputable def f (x : ℝ) : ℝ := x^5 + x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 2
noncomputable def roots : List ℝ := sorry -- To indicate the list of roots x_1, x_2, x_3, x_4, x_5 of the polynomial f(x)

theorem product_of_g_at_roots :
  (roots.map g).prod = -23 := sorry

end NUMINAMATH_GPT_product_of_g_at_roots_l2384_238489


namespace NUMINAMATH_GPT_fourth_term_in_arithmetic_sequence_l2384_238459

theorem fourth_term_in_arithmetic_sequence (a d : ℝ) (h : 2 * a + 6 * d = 20) : a + 3 * d = 10 :=
sorry

end NUMINAMATH_GPT_fourth_term_in_arithmetic_sequence_l2384_238459


namespace NUMINAMATH_GPT_sequence_properties_l2384_238479

-- Define the sequence formula
def a_n (n : ℤ) : ℤ := n^2 - 5 * n + 4

-- State the theorem about the sequence
theorem sequence_properties :
  -- Part 1: The number of negative terms in the sequence
  (∃ (S : Finset ℤ), ∀ n ∈ S, a_n n < 0 ∧ S.card = 2) ∧
  -- Part 2: The minimum value of the sequence and the value of n at minimum
  (∀ n : ℤ, (a_n n ≥ -9 / 4) ∧ (a_n (5 / 2) = -9 / 4)) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_properties_l2384_238479


namespace NUMINAMATH_GPT_hurdle_distance_l2384_238450

theorem hurdle_distance (d : ℝ) : 
  50 + 11 * d + 55 = 600 → d = 45 := by
  sorry

end NUMINAMATH_GPT_hurdle_distance_l2384_238450


namespace NUMINAMATH_GPT_carla_marbles_l2384_238470

theorem carla_marbles (before now bought : ℝ) (h_before : before = 187.0) (h_now : now = 321) : bought = 134 :=
by
  sorry

end NUMINAMATH_GPT_carla_marbles_l2384_238470


namespace NUMINAMATH_GPT_ages_total_l2384_238492

theorem ages_total (P Q : ℕ) (h1 : P - 8 = (1 / 2) * (Q - 8)) (h2 : P / Q = 3 / 4) : P + Q = 28 :=
by
  sorry

end NUMINAMATH_GPT_ages_total_l2384_238492


namespace NUMINAMATH_GPT_rate_times_base_eq_9000_l2384_238448

noncomputable def Rate : ℝ := 0.00015
noncomputable def BaseAmount : ℝ := 60000000

theorem rate_times_base_eq_9000 :
  Rate * BaseAmount = 9000 := 
  sorry

end NUMINAMATH_GPT_rate_times_base_eq_9000_l2384_238448


namespace NUMINAMATH_GPT_tom_batteries_used_total_l2384_238441

def batteries_used_in_flashlights : Nat := 2 * 3
def batteries_used_in_toys : Nat := 4 * 5
def batteries_used_in_controllers : Nat := 2 * 6
def total_batteries_used : Nat := batteries_used_in_flashlights + batteries_used_in_toys + batteries_used_in_controllers

theorem tom_batteries_used_total : total_batteries_used = 38 :=
by
  sorry

end NUMINAMATH_GPT_tom_batteries_used_total_l2384_238441


namespace NUMINAMATH_GPT_find_code_l2384_238477

theorem find_code (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 11 * (A + B + C) = 242) :
  A = 5 ∧ B = 8 ∧ C = 9 ∨ A = 5 ∧ B = 9 ∧ C = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_code_l2384_238477


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l2384_238456

-- Define the condition and the statement to be proved for the first equation
theorem solve_eq1 (x : ℝ) : 9 * x^2 - 25 = 0 → x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Define the condition and the statement to be proved for the second equation
theorem solve_eq2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 :=
by sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l2384_238456


namespace NUMINAMATH_GPT_geometric_sequence_b_l2384_238424

theorem geometric_sequence_b (b : ℝ) (h : b > 0) (s : ℝ) 
  (h1 : 30 * s = b) (h2 : b * s = 15 / 4) : 
  b = 15 * Real.sqrt 2 / 2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_l2384_238424


namespace NUMINAMATH_GPT_eval_expression_l2384_238442

-- Define the expression to evaluate
def expression : ℚ := 2 * 3 + 4 - (5 / 6)

-- Prove the equivalence of the evaluated expression to the expected result
theorem eval_expression : expression = 37 / 3 :=
by
  -- The detailed proof steps are omitted (relying on sorry)
  sorry

end NUMINAMATH_GPT_eval_expression_l2384_238442


namespace NUMINAMATH_GPT_students_ages_average_l2384_238409

variables (a b c : ℕ)

theorem students_ages_average (h1 : (14 * a + 13 * b + 12 * c) = 13 * (a + b + c)) : a = c :=
by
  sorry

end NUMINAMATH_GPT_students_ages_average_l2384_238409


namespace NUMINAMATH_GPT_students_transferred_l2384_238439

theorem students_transferred (students_before : ℕ) (total_students : ℕ) (students_equal : ℕ) 
  (h1 : students_before = 23) (h2 : total_students = 50) (h3 : students_equal = total_students / 2) : 
  (∃ x : ℕ, students_equal = students_before + x) → (∃ x : ℕ, x = 2) :=
by
  -- h1: students_before = 23
  -- h2: total_students = 50
  -- h3: students_equal = total_students / 2
  -- to prove: ∃ x : ℕ, students_equal = students_before + x → ∃ x : ℕ, x = 2
  sorry

end NUMINAMATH_GPT_students_transferred_l2384_238439


namespace NUMINAMATH_GPT_add_coefficients_l2384_238413

theorem add_coefficients (a : ℕ) : 2 * a + a = 3 * a :=
by 
  sorry

end NUMINAMATH_GPT_add_coefficients_l2384_238413


namespace NUMINAMATH_GPT_parallel_vectors_condition_l2384_238484

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2)

theorem parallel_vectors_condition (m : ℝ) :
  vectors_parallel (1, m + 1) (m, 2) ↔ m = -2 ∨ m = 1 := by
  sorry

end NUMINAMATH_GPT_parallel_vectors_condition_l2384_238484


namespace NUMINAMATH_GPT_polynomial_roots_l2384_238432

theorem polynomial_roots : (∃ x : ℝ, (4 * x ^ 4 + 11 * x ^ 3 - 37 * x ^ 2 + 18 * x = 0) ↔ (x = 0 ∨ x = 1 / 2 ∨ x = 3 / 2 ∨ x = -6)) :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_roots_l2384_238432


namespace NUMINAMATH_GPT_sum_of_roots_even_l2384_238463

theorem sum_of_roots_even (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
    (h_distinct : ∃ x y : ℤ, x ≠ y ∧ (x^2 - 2 * p * x + (p * q) = 0) ∧ (y^2 - 2 * p * y + (p * q) = 0)) :
    Even (2 * p) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_roots_even_l2384_238463


namespace NUMINAMATH_GPT_boar_sausages_left_l2384_238483

def boar_sausages_final_count(sausages_initial : ℕ) : ℕ :=
  let after_monday := sausages_initial - (2 / 5 * sausages_initial)
  let after_tuesday := after_monday - (1 / 2 * after_monday)
  let after_wednesday := after_tuesday - (1 / 4 * after_tuesday)
  let after_thursday := after_wednesday - (1 / 3 * after_wednesday)
  let after_sharing := after_thursday - (1 / 5 * after_thursday)
  let after_eating := after_sharing - (3 / 5 * after_sharing)
  after_eating

theorem boar_sausages_left : boar_sausages_final_count 1200 = 58 := 
  sorry

end NUMINAMATH_GPT_boar_sausages_left_l2384_238483


namespace NUMINAMATH_GPT_evaluate_expression_at_2_l2384_238446

-- Define the quadratic and linear components of the expression
def quadratic (x : ℝ) := 3 * x ^ 2 - 8 * x + 5
def linear (x : ℝ) := 4 * x - 7

-- State the proposition to evaluate the given expression at x = 2
theorem evaluate_expression_at_2 : quadratic 2 * linear 2 = 1 := by
  -- The proof is skipped by using sorry
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_2_l2384_238446


namespace NUMINAMATH_GPT_pentagon_same_parity_l2384_238473

open Classical

theorem pentagon_same_parity (vertices : Fin 5 → ℤ × ℤ) : 
  ∃ i j : Fin 5, i ≠ j ∧ (vertices i).1 % 2 = (vertices j).1 % 2 ∧ (vertices i).2 % 2 = (vertices j).2 % 2 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_same_parity_l2384_238473


namespace NUMINAMATH_GPT_ferry_P_travel_time_l2384_238401

-- Define the conditions based on the problem statement
variables (t : ℝ) -- travel time of ferry P
def speed_P := 6 -- speed of ferry P in km/h
def speed_Q := speed_P + 3 -- speed of ferry Q in km/h
def distance_P := speed_P * t -- distance traveled by ferry P in km
def distance_Q := 3 * distance_P -- distance traveled by ferry Q in km
def time_Q := t + 3 -- travel time of ferry Q

-- Theorem to prove that travel time t for ferry P is 3 hours
theorem ferry_P_travel_time : time_Q * speed_Q = distance_Q → t = 3 :=
by {
  -- Since you've mentioned to include the statement only and not the proof,
  -- Therefore, the proof body is left as an exercise or represented by sorry.
  sorry
}

end NUMINAMATH_GPT_ferry_P_travel_time_l2384_238401


namespace NUMINAMATH_GPT_find_b_l2384_238420

def oscillation_period (a b c d : ℝ) (oscillations : ℝ) : Prop :=
  oscillations = 5 * (2 * Real.pi) / b

theorem find_b
  (a b c d : ℝ)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (pos_d : d > 0)
  (osc_complexity: oscillation_period a b c d 5):
  b = 5 := by
  sorry

end NUMINAMATH_GPT_find_b_l2384_238420


namespace NUMINAMATH_GPT_total_students_correct_l2384_238485

def num_boys : ℕ := 272
def num_girls : ℕ := num_boys + 106
def total_students : ℕ := num_boys + num_girls

theorem total_students_correct : total_students = 650 :=
by
  sorry

end NUMINAMATH_GPT_total_students_correct_l2384_238485


namespace NUMINAMATH_GPT_compare_expressions_l2384_238449

theorem compare_expressions (x y : ℝ) (h1: x * y > 0) (h2: x ≠ y) : 
  x^4 + 6 * x^2 * y^2 + y^4 > 4 * x * y * (x^2 + y^2) :=
by
  sorry

end NUMINAMATH_GPT_compare_expressions_l2384_238449


namespace NUMINAMATH_GPT_pool_capacity_percentage_l2384_238481

theorem pool_capacity_percentage :
  let width := 60 
  let length := 150 
  let depth := 10 
  let drain_rate := 60 
  let time := 1200 
  let total_volume := width * length * depth
  let water_removed := drain_rate * time
  let capacity_percentage := (water_removed / total_volume : ℚ) * 100
  capacity_percentage = 80 := by
  sorry

end NUMINAMATH_GPT_pool_capacity_percentage_l2384_238481


namespace NUMINAMATH_GPT_problem_l2384_238464

noncomputable def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b

theorem problem (a b : ℝ) (h1 : f 1 a b = 0) (h2 : f 2 a b = 0) : f (-1) a b = 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2384_238464


namespace NUMINAMATH_GPT_faye_pencils_l2384_238487

theorem faye_pencils (rows : ℕ) (pencils_per_row : ℕ) (h_rows : rows = 30) (h_pencils_per_row : pencils_per_row = 24) :
  rows * pencils_per_row = 720 :=
by
  sorry

end NUMINAMATH_GPT_faye_pencils_l2384_238487


namespace NUMINAMATH_GPT_direct_proportion_inequality_l2384_238498

theorem direct_proportion_inequality (k x1 x2 y1 y2 : ℝ) (h_k : k < 0) (h_y1 : y1 = k * x1) (h_y2 : y2 = k * x2) (h_x : x1 < x2) : y1 > y2 :=
by
  -- The proof will be written here, currently leaving it as sorry
  sorry

end NUMINAMATH_GPT_direct_proportion_inequality_l2384_238498


namespace NUMINAMATH_GPT_pure_imaginary_sol_l2384_238408

theorem pure_imaginary_sol (m : ℝ) (h : (m^2 - m - 2) = 0 ∧ (m + 1) ≠ 0) : m = 2 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_sol_l2384_238408


namespace NUMINAMATH_GPT_eval_f_3_minus_f_neg_3_l2384_238427

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 7 * x

-- State the theorem
theorem eval_f_3_minus_f_neg_3 : f 3 - f (-3) = 690 := by
  sorry

end NUMINAMATH_GPT_eval_f_3_minus_f_neg_3_l2384_238427


namespace NUMINAMATH_GPT_hyperbola_transformation_l2384_238451

def equation_transform (x y : ℝ) : Prop :=
  y = (1 - 3 * x) / (2 * x - 1)

def coordinate_shift (x y X Y : ℝ) : Prop :=
  X = x - 0.5 ∧ Y = y + 1.5

theorem hyperbola_transformation (x y X Y : ℝ) :
  equation_transform x y →
  coordinate_shift x y X Y →
  (X * Y = -0.25) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_transformation_l2384_238451


namespace NUMINAMATH_GPT_solve_equation_l2384_238482

theorem solve_equation (x : ℝ) : 
  (3 * x + 2) * (x + 3) = x + 3 ↔ (x = -3 ∨ x = -1/3) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l2384_238482


namespace NUMINAMATH_GPT_keith_total_spent_l2384_238421

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tire_cost : ℝ := 112.46
def num_tires : ℕ := 4
def printer_cable_cost : ℝ := 14.85
def num_printer_cables : ℕ := 2
def blank_cd_pack_cost : ℝ := 0.98
def num_blank_cds : ℕ := 10
def sales_tax_rate : ℝ := 0.0825

theorem keith_total_spent : 
  speakers_cost +
  cd_player_cost +
  (num_tires * tire_cost) +
  (num_printer_cables * printer_cable_cost) +
  (num_blank_cds * blank_cd_pack_cost) *
  (1 + sales_tax_rate) = 827.87 := 
sorry

end NUMINAMATH_GPT_keith_total_spent_l2384_238421


namespace NUMINAMATH_GPT_women_science_majors_is_30_percent_l2384_238453

noncomputable def percentage_women_science_majors (ns_percent : ℝ) (m_percent : ℝ) (m_sci_percent : ℝ) : ℝ :=
  let w_percent := 1 - m_percent
  let m_sci_total := m_percent * m_sci_percent
  let total_sci := 1 - ns_percent
  let w_sci_total := total_sci - m_sci_total
  (w_sci_total / w_percent) * 100

theorem women_science_majors_is_30_percent :
  percentage_women_science_majors 0.60 0.40 0.55 = 30 := by
  sorry

end NUMINAMATH_GPT_women_science_majors_is_30_percent_l2384_238453


namespace NUMINAMATH_GPT_clothing_store_earnings_l2384_238467

-- Defining the conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def shirt_cost : ℕ := 10
def jeans_cost : ℕ := 2 * shirt_cost

-- Statement of the problem
theorem clothing_store_earnings :
  num_shirts * shirt_cost + num_jeans * jeans_cost = 400 :=
by
  sorry

end NUMINAMATH_GPT_clothing_store_earnings_l2384_238467


namespace NUMINAMATH_GPT_distance_traveled_l2384_238440

theorem distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 12) : D = 100 := 
sorry

end NUMINAMATH_GPT_distance_traveled_l2384_238440


namespace NUMINAMATH_GPT_smallest_integer_x_l2384_238403

theorem smallest_integer_x (x : ℤ) : (x^2 - 11 * x + 24 < 0) → x ≥ 4 ∧ x < 8 :=
by
sorry

end NUMINAMATH_GPT_smallest_integer_x_l2384_238403


namespace NUMINAMATH_GPT_mrs_doe_inheritance_l2384_238416

noncomputable def calculateInheritance (totalTaxes : ℝ) : ℝ :=
  totalTaxes / 0.3625

theorem mrs_doe_inheritance (h : 0.3625 * calculateInheritance 15000 = 15000) :
  calculateInheritance 15000 = 41379 :=
by
  unfold calculateInheritance
  field_simp
  norm_cast
  sorry

end NUMINAMATH_GPT_mrs_doe_inheritance_l2384_238416


namespace NUMINAMATH_GPT_limit_example_l2384_238438

open Real

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, (∀ x : ℝ, 0 < |x + 4| ∧ |x + 4| < δ → abs ((2 * x^2 + 6 * x - 8) / (x + 4) + 10) < ε) :=
by
  sorry

end NUMINAMATH_GPT_limit_example_l2384_238438


namespace NUMINAMATH_GPT_range_of_a_l2384_238490

def is_monotonically_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, (0 < x) → (x < y) → (f x ≤ f y)

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem range_of_a (a : ℝ) : 
  is_monotonically_increasing (f a) a → a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2384_238490


namespace NUMINAMATH_GPT_boxes_in_attic_l2384_238496

theorem boxes_in_attic (B : ℕ)
  (h1 : 6 ≤ B)
  (h2 : ∀ T : ℕ, T = (B - 6) / 2 ∧ T = 10)
  (h3 : ∀ O : ℕ, O = 180 + 2 * T ∧ O = 20 * T) :
  B = 26 :=
by
  sorry

end NUMINAMATH_GPT_boxes_in_attic_l2384_238496


namespace NUMINAMATH_GPT_find_perp_line_eq_l2384_238418

-- Line equation in the standard form
def line_eq (x y : ℝ) : Prop := 4 * x - 3 * y - 12 = 0

-- Equation of the required line that is perpendicular to the given line and has the same y-intercept
def perp_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 16 = 0

theorem find_perp_line_eq (x y : ℝ) :
  (∃ k : ℝ, line_eq 0 k ∧ perp_line_eq 0 k) →
  (∃ a b c : ℝ, perp_line_eq a b) :=
by
  sorry

end NUMINAMATH_GPT_find_perp_line_eq_l2384_238418


namespace NUMINAMATH_GPT_smallest_x_plus_y_l2384_238455

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end NUMINAMATH_GPT_smallest_x_plus_y_l2384_238455


namespace NUMINAMATH_GPT_combined_area_win_bonus_l2384_238433

theorem combined_area_win_bonus (r : ℝ) (P_win P_bonus : ℝ) : 
  r = 8 → P_win = 1 / 4 → P_bonus = 1 / 8 → 
  (P_win * (Real.pi * r^2) + P_bonus * (Real.pi * r^2) = 24 * Real.pi) :=
by
  intro h_r h_Pwin h_Pbonus
  rw [h_r, h_Pwin, h_Pbonus]
  -- Calculation is skipped as per the instructions
  sorry

end NUMINAMATH_GPT_combined_area_win_bonus_l2384_238433


namespace NUMINAMATH_GPT_ball_travel_distance_fourth_hit_l2384_238436

theorem ball_travel_distance_fourth_hit :
  let initial_height := 150
  let rebound_ratio := 1 / 3
  let distances := [initial_height, 
                    initial_height * rebound_ratio, 
                    initial_height * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    (initial_height * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio, 
                    ((initial_height * rebound_ratio) * rebound_ratio) * rebound_ratio]
  distances.sum = 294 + 1 / 3 := by
  sorry

end NUMINAMATH_GPT_ball_travel_distance_fourth_hit_l2384_238436


namespace NUMINAMATH_GPT_product_of_equal_numbers_l2384_238405

theorem product_of_equal_numbers (a b : ℕ) (mean : ℕ) (sum : ℕ)
  (h1 : mean = 20)
  (h2 : a = 22)
  (h3 : b = 34)
  (h4 : sum = 4 * mean)
  (h5 : sum - a - b = 2 * x)
  (h6 : sum = 80)
  (h7 : x = 12) 
  : x * x = 144 :=
by
  sorry

end NUMINAMATH_GPT_product_of_equal_numbers_l2384_238405


namespace NUMINAMATH_GPT_car_speed_l2384_238491

theorem car_speed (v : ℝ) (h : (1 / v) = (1 / 100 + 2 / 3600)) : v = 3600 / 38 := 
by
  sorry

end NUMINAMATH_GPT_car_speed_l2384_238491


namespace NUMINAMATH_GPT_steven_apples_peaches_difference_l2384_238430

def steven_apples := 19
def jake_apples (steven_apples : ℕ) := steven_apples + 4
def jake_peaches (steven_peaches : ℕ) := steven_peaches - 3

theorem steven_apples_peaches_difference (P : ℕ) :
  19 - P = steven_apples - P :=
by
  sorry

end NUMINAMATH_GPT_steven_apples_peaches_difference_l2384_238430


namespace NUMINAMATH_GPT_least_integer_value_x_l2384_238471

theorem least_integer_value_x (x : ℤ) : (3 * |2 * (x : ℤ) - 1| + 6 < 24) → x = -2 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_value_x_l2384_238471
