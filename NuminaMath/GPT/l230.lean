import Mathlib

namespace sum_of_fractions_l230_23075

theorem sum_of_fractions :
  (1/15 + 2/15 + 3/15 + 4/15 + 5/15 + 6/15 + 7/15 + 8/15 + 9/15 + 46/15) = (91/15) := by
  sorry

end sum_of_fractions_l230_23075


namespace perfect_square_for_x_l230_23067

def expr (x : ℝ) : ℝ := 11.98 * 11.98 + 11.98 * x + 0.02 * 0.02

theorem perfect_square_for_x : expr 0.04 = (11.98 + 0.02) ^ 2 :=
by
  sorry

end perfect_square_for_x_l230_23067


namespace convex_parallelogram_faces_1992_l230_23062

theorem convex_parallelogram_faces_1992 (n : ℕ) (h : n > 0) : (n * (n - 1) ≠ 1992) := 
by
  sorry

end convex_parallelogram_faces_1992_l230_23062


namespace product_remainder_l230_23082

theorem product_remainder (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) (h4 : (a + b + c) % 7 = 3) : 
  (a * b * c) % 7 = 2 := 
by sorry

end product_remainder_l230_23082


namespace prob_three_blue_is_correct_l230_23099

-- Definitions corresponding to the problem conditions
def total_jellybeans : ℕ := 20
def blue_jellybeans_start : ℕ := 10
def red_jellybeans : ℕ := 10

-- Probabilities calculation steps as definitions
def prob_first_blue : ℚ := blue_jellybeans_start / total_jellybeans
def prob_second_blue_given_first_blue : ℚ := (blue_jellybeans_start - 1) / (total_jellybeans - 1)
def prob_third_blue_given_first_two_blue : ℚ := (blue_jellybeans_start - 2) / (total_jellybeans - 2)

-- Total probability of drawing three blue jellybeans
def prob_three_blue : ℚ := 
  prob_first_blue *
  prob_second_blue_given_first_blue *
  prob_third_blue_given_first_two_blue

-- Formal statement of the proof problem
theorem prob_three_blue_is_correct : prob_three_blue = 2 / 19 :=
by
  -- Fill the proof here
  sorry

end prob_three_blue_is_correct_l230_23099


namespace Sequential_structure_not_conditional_l230_23000

-- Definitions based on provided conditions
def is_conditional (s : String) : Prop :=
  s = "Loop structure" ∨ s = "If structure" ∨ s = "Until structure"

-- Theorem stating that Sequential structure is the one that doesn't contain a conditional judgment box
theorem Sequential_structure_not_conditional :
  ¬ is_conditional "Sequential structure" :=
by
  intro h
  cases h <;> contradiction

end Sequential_structure_not_conditional_l230_23000


namespace PersonYs_speed_in_still_water_l230_23018

def speed_in_still_water (speed_X : ℕ) (t_1 t_2 : ℕ) (x : ℕ) : Prop :=
  ∀ y : ℤ, 4 * (6 - y + x + y) = 4 * 6 + 4 * x ∧ 16 * (x + y) = 16 * (6 + y) + 4 * (x - 6) →
  x = 10

theorem PersonYs_speed_in_still_water :
  speed_in_still_water 6 4 16 10 :=
by
  sorry

end PersonYs_speed_in_still_water_l230_23018


namespace problem_solution_l230_23034

variables {f : ℝ → ℝ}

-- f is monotonically decreasing on [1, 3]
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

-- f(x+3) is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 3) = f (3 - x)

-- Given conditions
axiom mono_dec : monotone_decreasing_on f 1 3
axiom even_f : even_function f

-- To prove: f(π) < f(2) < f(5)
theorem problem_solution : f π < f 2 ∧ f 2 < f 5 :=
by
  sorry

end problem_solution_l230_23034


namespace determine_m_l230_23027

theorem determine_m (x m : ℝ) (h₁ : 2 * x + m = 6) (h₂ : x = 2) : m = 2 := by
  sorry

end determine_m_l230_23027


namespace gcd_m_n_l230_23098

def m := 122^2 + 234^2 + 346^2 + 458^2
def n := 121^2 + 233^2 + 345^2 + 457^2

theorem gcd_m_n : Int.gcd m n = 1 := 
by sorry

end gcd_m_n_l230_23098


namespace calculate_E_l230_23069

theorem calculate_E (P J T B A E : ℝ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : B = 1.40 * J)
  (h4 : A = 0.85 * B)
  (h5 : T = P - (E / 100) * P)
  (h6 : E = 2 * ((P - A) / P) * 100) : 
  E = 21.5 := 
sorry

end calculate_E_l230_23069


namespace find_s_for_g_eq_0_l230_23095

def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 2 * x^2 - 5 * x + s

theorem find_s_for_g_eq_0 : ∃ (s : ℝ), g 3 s = 0 → s = -867 :=
by
  sorry

end find_s_for_g_eq_0_l230_23095


namespace proposition_2_proposition_4_l230_23074

variable {m n : Line}
variable {α β : Plane}

-- Define predicates for perpendicularity, parallelism, and containment
axiom line_parallel_plane (n : Line) (α : Plane) : Prop
axiom line_perp_plane (n : Line) (α : Plane) : Prop
axiom plane_perp_plane (α β : Plane) : Prop
axiom line_in_plane (m : Line) (β : Plane) : Prop

-- State the correct propositions
theorem proposition_2 (m n : Line) (α β : Plane)
  (h1 : line_perp_plane m n)
  (h2 : line_perp_plane n α)
  (h3 : line_perp_plane m β) :
  plane_perp_plane α β := sorry

theorem proposition_4 (n : Line) (α β : Plane)
  (h1 : line_perp_plane n β)
  (h2 : plane_perp_plane α β) :
  line_parallel_plane n α ∨ line_in_plane n α := sorry

end proposition_2_proposition_4_l230_23074


namespace arithmetic_expression_proof_l230_23010

theorem arithmetic_expression_proof : 4 * 6 * 8 + 18 / 3 ^ 2 = 194 := by
  sorry

end arithmetic_expression_proof_l230_23010


namespace angle_in_second_quadrant_l230_23065

/-- If α is an angle in the first quadrant, then π - α is an angle in the second quadrant -/
theorem angle_in_second_quadrant (α : Real) (h : 0 < α ∧ α < π / 2) : π - α > π / 2 ∧ π - α < π :=
by
  sorry

end angle_in_second_quadrant_l230_23065


namespace inscribed_circle_radius_l230_23037

theorem inscribed_circle_radius (a b c : ℝ) (R : ℝ) (r : ℝ) :
  a = 20 → b = 20 → d = 25 → r = 6 := 
by
  -- conditions of the problem
  sorry

end inscribed_circle_radius_l230_23037


namespace temperature_drop_change_l230_23023

theorem temperature_drop_change (T : ℝ) (h1 : T + 2 = T + 2) :
  (T - 4) - T = -4 :=
by
  sorry

end temperature_drop_change_l230_23023


namespace decimal_to_fraction_correct_l230_23061

-- Define a structure representing our initial decimal to fraction conversion
structure DecimalFractionConversion :=
  (decimal: ℚ)
  (vulgar_fraction: ℚ)
  (simplified_fraction: ℚ)

-- Define the conditions provided in the problem
def conversion_conditions : DecimalFractionConversion :=
  { decimal := 35 / 100,
    vulgar_fraction := 35 / 100,
    simplified_fraction := 7 / 20 }

-- State the theorem we aim to prove
theorem decimal_to_fraction_correct :
  conversion_conditions.simplified_fraction = 7 / 20 := by
  sorry

end decimal_to_fraction_correct_l230_23061


namespace termites_ate_black_squares_l230_23058

def chessboard_black_squares_eaten : Nat :=
  12

theorem termites_ate_black_squares :
  let rows := 8;
  let cols := 8;
  let total_squares := rows * cols / 2; -- This simplistically assumes half the squares are black.
  (total_squares = 32) → 
  chessboard_black_squares_eaten = 12 :=
by
  intros h
  sorry

end termites_ate_black_squares_l230_23058


namespace inequality_has_exactly_one_solution_l230_23011

-- Definitions based on the conditions
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 3 * a

-- The main theorem that encodes the proof problem
theorem inequality_has_exactly_one_solution (a : ℝ) : 
  (∃! x : ℝ, |f x a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end inequality_has_exactly_one_solution_l230_23011


namespace tina_first_hour_coins_l230_23096

variable (X : ℕ)

theorem tina_first_hour_coins :
  let first_hour_coins := X
  let second_third_hour_coins := 30 + 30
  let fourth_hour_coins := 40
  let fifth_hour_removed_coins := 20
  let total_coins := first_hour_coins + second_third_hour_coins + fourth_hour_coins - fifth_hour_removed_coins
  total_coins = 100 → X = 20 :=
by
  intro h
  sorry

end tina_first_hour_coins_l230_23096


namespace extra_coverage_calculation_l230_23085

/-- Define the conditions -/
def bag_coverage : ℕ := 500
def lawn_length : ℕ := 35
def lawn_width : ℕ := 48
def number_of_bags : ℕ := 6

/-- Define the main theorem to prove -/
theorem extra_coverage_calculation :
  number_of_bags * bag_coverage - (lawn_length * lawn_width) = 1320 := 
by
  sorry

end extra_coverage_calculation_l230_23085


namespace combined_return_percentage_l230_23087

theorem combined_return_percentage (investment1 investment2 : ℝ) 
  (return1_percent return2_percent : ℝ) (total_investment total_return : ℝ) :
  investment1 = 500 → 
  return1_percent = 0.07 → 
  investment2 = 1500 → 
  return2_percent = 0.09 → 
  total_investment = investment1 + investment2 → 
  total_return = investment1 * return1_percent + investment2 * return2_percent → 
  (total_return / total_investment) * 100 = 8.5 :=
by 
  sorry

end combined_return_percentage_l230_23087


namespace initial_population_l230_23007

theorem initial_population (P : ℝ) (h : (0.9 : ℝ)^2 * P = 4860) : P = 6000 :=
by
  sorry

end initial_population_l230_23007


namespace shared_total_l230_23012

theorem shared_total (total_amount : ℝ) (maggie_share : ℝ) (debby_percentage : ℝ)
  (h1 : debby_percentage = 0.25)
  (h2 : maggie_share = 4500)
  (h3 : maggie_share = (1 - debby_percentage) * total_amount) :
  total_amount = 6000 :=
by
  sorry

end shared_total_l230_23012


namespace equal_share_of_tea_l230_23044

def totalCups : ℕ := 10
def totalPeople : ℕ := 5
def cupsPerPerson : ℕ := totalCups / totalPeople

theorem equal_share_of_tea : cupsPerPerson = 2 := by
  sorry

end equal_share_of_tea_l230_23044


namespace ladder_base_l230_23063

theorem ladder_base (h : ℝ) (b : ℝ) (l : ℝ)
  (h_eq : h = 12) (l_eq : l = 15) : b = 9 :=
by
  have hypotenuse := l
  have height := h
  have base := b
  have pythagorean_theorem : height^2 + base^2 = hypotenuse^2 := by sorry 
  sorry

end ladder_base_l230_23063


namespace cost_price_of_watch_l230_23079

theorem cost_price_of_watch :
  ∃ (CP : ℝ), (CP * 1.07 = CP * 0.88 + 250) ∧ CP = 250 / 0.19 :=
sorry

end cost_price_of_watch_l230_23079


namespace smallest_possible_sector_angle_l230_23042

theorem smallest_possible_sector_angle : ∃ a₁ d : ℕ, 2 * a₁ + 9 * d = 72 ∧ a₁ = 9 :=
by
  sorry

end smallest_possible_sector_angle_l230_23042


namespace rate_percent_is_10_l230_23066

theorem rate_percent_is_10
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ) 
  (h1 : SI = 2500) (h2 : P = 5000) (h3 : T = 5) :
  R = 10 :=
by
  sorry

end rate_percent_is_10_l230_23066


namespace geometric_sequence_product_l230_23051

theorem geometric_sequence_product (b : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, b (n+1) = b n * r)
  (h_b9 : b 9 = (3 + 5) / 2) : b 1 * b 17 = 16 :=
by
  sorry

end geometric_sequence_product_l230_23051


namespace geometric_sequence_sum_l230_23060

theorem geometric_sequence_sum {a : ℕ → ℤ} (r : ℤ) (h1 : a 1 = 1) (h2 : r = -2) 
(h3 : ∀ n, a (n + 1) = a n * r) : 
  a 1 + |a 2| + |a 3| + a 4 = 15 := 
by sorry

end geometric_sequence_sum_l230_23060


namespace books_written_by_Zig_l230_23040

theorem books_written_by_Zig (F Z : ℕ) (h1 : Z = 4 * F) (h2 : F + Z = 75) : Z = 60 := by
  sorry

end books_written_by_Zig_l230_23040


namespace rationalize_denominator_sqrt_l230_23017

theorem rationalize_denominator_sqrt (x y : ℝ) (hx : x = 5) (hy : y = 12) :
  Real.sqrt (x / y) = Real.sqrt 15 / 6 :=
by
  rw [hx, hy]
  sorry

end rationalize_denominator_sqrt_l230_23017


namespace inequality_proof_l230_23049

theorem inequality_proof (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) : 
  2 * Real.sin α + Real.tan α > 3 * α := 
by
  sorry

end inequality_proof_l230_23049


namespace candy_store_truffle_price_l230_23031

def total_revenue : ℝ := 212
def fudge_revenue : ℝ := 20 * 2.5
def pretzels_revenue : ℝ := 3 * 12 * 2.0
def truffles_quantity : ℕ := 5 * 12

theorem candy_store_truffle_price (total_revenue fudge_revenue pretzels_revenue truffles_quantity : ℝ) : 
  (total_revenue - (fudge_revenue + pretzels_revenue)) / truffles_quantity = 1.50 := 
by 
  sorry

end candy_store_truffle_price_l230_23031


namespace find_13th_result_l230_23029

theorem find_13th_result
  (avg_25 : ℕ → ℕ)
  (avg_1_to_12 : ℕ → ℕ)
  (avg_14_to_25 : ℕ → ℕ)
  (h1 : avg_25 25 = 50)
  (h2 : avg_1_to_12 12 = 14)
  (h3 : avg_14_to_25 12 = 17) :
  ∃ (X : ℕ), X = 878 := sorry

end find_13th_result_l230_23029


namespace complex_power_difference_l230_23078

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i)^10 - (1 - i)^10 = 64 * i := 
by sorry

end complex_power_difference_l230_23078


namespace max_two_digit_number_divisible_by_23_l230_23033

theorem max_two_digit_number_divisible_by_23 :
  ∃ n : ℕ, 
    (n < 100) ∧ 
    (1000 ≤ n * 109) ∧ 
    (n * 109 < 10000) ∧ 
    (n % 23 = 0) ∧ 
    (n / 23 < 10) ∧ 
    (n = 69) :=
by {
  sorry
}

end max_two_digit_number_divisible_by_23_l230_23033


namespace minimum_flower_cost_l230_23038

def vertical_strip_width : ℝ := 3
def horizontal_strip_height : ℝ := 2
def bed_width : ℝ := 11
def bed_height : ℝ := 6

def easter_lily_cost : ℝ := 3
def dahlia_cost : ℝ := 2.5
def canna_cost : ℝ := 2

def vertical_strip_area : ℝ := vertical_strip_width * bed_height
def horizontal_strip_area : ℝ := horizontal_strip_height * bed_width
def overlap_area : ℝ := vertical_strip_width * horizontal_strip_height
def remaining_area : ℝ := (bed_width * bed_height) - vertical_strip_area - (horizontal_strip_area - overlap_area)

def easter_lily_area : ℝ := horizontal_strip_area - overlap_area
def dahlia_area : ℝ := vertical_strip_area
def canna_area : ℝ := remaining_area

def easter_lily_total_cost : ℝ := easter_lily_area * easter_lily_cost
def dahlia_total_cost : ℝ := dahlia_area * dahlia_cost
def canna_total_cost : ℝ := canna_area * canna_cost

def total_cost : ℝ := easter_lily_total_cost + dahlia_total_cost + canna_total_cost

theorem minimum_flower_cost : total_cost = 157 := by
  sorry

end minimum_flower_cost_l230_23038


namespace bald_eagle_dive_time_l230_23073

-- Definitions as per the conditions in the problem
def speed_bald_eagle : ℝ := 100
def speed_peregrine_falcon : ℝ := 2 * speed_bald_eagle
def time_peregrine_falcon : ℝ := 15

-- The theorem to prove
theorem bald_eagle_dive_time : (speed_bald_eagle * 30) = (speed_peregrine_falcon * time_peregrine_falcon) := by
  sorry

end bald_eagle_dive_time_l230_23073


namespace flat_rate_65_l230_23002

noncomputable def flat_rate_first_night (f n : ℝ) : Prop := 
  (f + 4 * n = 245) ∧ (f + 9 * n = 470)

theorem flat_rate_65 :
  ∃ (f n : ℝ), flat_rate_first_night f n ∧ f = 65 := 
by
  sorry

end flat_rate_65_l230_23002


namespace log_inequality_l230_23014

theorem log_inequality (x y : ℝ) :
  let log2 := Real.log 2
  let log5 := Real.log 5
  let log3 := Real.log 3
  let log2_3 := log3 / log2
  let log5_3 := log3 / log5
  (log2_3 ^ x - log5_3 ^ x ≥ log2_3 ^ (-y) - log5_3 ^ (-y)) → (x + y ≥ 0) :=
by
  intros h
  sorry

end log_inequality_l230_23014


namespace michael_height_l230_23009

theorem michael_height (flagpole_height flagpole_shadow michael_shadow : ℝ) 
                        (h1 : flagpole_height = 50) 
                        (h2 : flagpole_shadow = 25) 
                        (h3 : michael_shadow = 5) : 
                        (michael_shadow * (flagpole_height / flagpole_shadow) = 10) :=
by
  sorry

end michael_height_l230_23009


namespace solution_set_inequality_l230_23022

theorem solution_set_inequality (a c : ℝ)
  (h : ∀ x : ℝ, (ax^2 + 2*x + c < 0) ↔ (x < -1/3 ∨ x > 1/2)) :
  (∀ x : ℝ, (cx^2 - 2*x + a ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 3)) :=
sorry

end solution_set_inequality_l230_23022


namespace find_f_of_3_l230_23054

theorem find_f_of_3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x * f y - y) = x * y - f y) 
  (h2 : f 0 = 0) (h3 : ∀ x : ℝ, f (-x) = -f x) : f 3 = 3 :=
sorry

end find_f_of_3_l230_23054


namespace count_three_digit_integers_with_product_thirty_l230_23015

theorem count_three_digit_integers_with_product_thirty :
  (∃ S : Finset (ℕ × ℕ × ℕ),
      (∀ (a b c : ℕ), (a, b, c) ∈ S → a * b * c = 30 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9) 
    ∧ S.card = 12) :=
by
  sorry

end count_three_digit_integers_with_product_thirty_l230_23015


namespace p_adic_valuation_of_factorial_l230_23046

noncomputable def digit_sum (n p : ℕ) : ℕ :=
  -- Definition for sum of digits of n in base p
  sorry

def p_adic_valuation (n factorial : ℕ) (p : ℕ) : ℕ :=
  -- Representation of p-adic valuation of n!
  sorry

theorem p_adic_valuation_of_factorial (n p : ℕ) (hp: p > 1):
  p_adic_valuation n.factorial p = (n - digit_sum n p) / (p - 1) :=
sorry

end p_adic_valuation_of_factorial_l230_23046


namespace harmonic_mean_pairs_count_l230_23053

theorem harmonic_mean_pairs_count :
  ∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, p.1 < p.2 ∧ 2 * p.1 * p.2 = 4^15 * (p.1 + p.2)) ∧ s.card = 29 :=
sorry

end harmonic_mean_pairs_count_l230_23053


namespace solution_set_f_x_leq_x_range_of_a_l230_23070

-- Definition of the function f
def f (x : ℝ) : ℝ := |2 * x - 7| + 1

-- Proof Problem for Question (1):
-- Given: f(x) = |2x - 7| + 1
-- Prove: The solution set of the inequality f(x) <= x is {x | 8/3 <= x <= 6}
theorem solution_set_f_x_leq_x :
  { x : ℝ | f x ≤ x } = { x : ℝ | 8 / 3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Definition of the function g
def g (x : ℝ) : ℝ := f x - 2 * |x - 1|

-- Proof Problem for Question (2):
-- Given: f(x) = |2x - 7| + 1 and g(x) = f(x) - 2 * |x - 1|
-- Prove: If ∃ x, g(x) <= a, then a >= -4
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 :=
sorry

end solution_set_f_x_leq_x_range_of_a_l230_23070


namespace problem1_problem2_problem3_problem4_l230_23071

variable (a b c : ℝ)

theorem problem1 : a^4 * (a^2)^3 = a^10 :=
by
  sorry

theorem problem2 : 2 * a^3 * b^2 * c / (1 / 3 * a^2 * b) = 6 * a * b * c :=
by
  sorry

theorem problem3 : 6 * a * (1 / 3 * a * b - b) - (2 * a * b + b) * (a - 1) = -5 * a * b + b :=
by
  sorry

theorem problem4 : (a - 2)^2 - (3 * a + 2 * b) * (3 * a - 2 * b) = -8 * a^2 - 4 * a + 4 + 4 * b^2 :=
by
  sorry

end problem1_problem2_problem3_problem4_l230_23071


namespace zero_sum_of_squares_l230_23039

theorem zero_sum_of_squares {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end zero_sum_of_squares_l230_23039


namespace necessary_condition_for_inequality_l230_23019

-- Definitions based on the conditions in a)
variables (A B C D : ℝ)

-- Main statement translating c) into Lean
theorem necessary_condition_for_inequality (h : C < D) : A > B :=
by sorry

end necessary_condition_for_inequality_l230_23019


namespace average_speed_whole_journey_l230_23013

theorem average_speed_whole_journey (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 54
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  let V_avg := total_distance / total_time
  V_avg = 64.8 :=
by
  sorry

end average_speed_whole_journey_l230_23013


namespace mean_of_combined_sets_l230_23005

theorem mean_of_combined_sets (A : Finset ℝ) (B : Finset ℝ)
  (hA_len : A.card = 7) (hB_len : B.card = 8)
  (hA_mean : (A.sum id) / 7 = 15) (hB_mean : (B.sum id) / 8 = 22) :
  (A.sum id + B.sum id) / 15 = 18.73 :=
by sorry

end mean_of_combined_sets_l230_23005


namespace smallest_n_for_common_factor_l230_23048

theorem smallest_n_for_common_factor : ∃ n : ℕ, n > 0 ∧ (Nat.gcd (11 * n - 3) (8 * n + 4) > 1) ∧ n = 42 := 
by
  sorry

end smallest_n_for_common_factor_l230_23048


namespace circle_tangent_l230_23072

variables {O M : ℝ} {R : ℝ}

theorem circle_tangent
  (r : ℝ)
  (hOM_pos : O ≠ M)
  (hO : O > 0)
  (hR : R > 0)
  (h_distinct : ∀ (m n : ℝ), m ≠ n → abs (m - n) ≠ 0) :
  (r = abs (O - M) - R) ∨ (r = abs (O - M) + R) ∨ (r = R - abs (O - M)) →
  (abs ((O - M)^2 + r^2 - R^2) = 2 * R * r) :=
sorry

end circle_tangent_l230_23072


namespace right_triangle_min_perimeter_multiple_13_l230_23028

theorem right_triangle_min_perimeter_multiple_13 :
  ∃ (a b c : ℕ), 
    (a^2 + b^2 = c^2) ∧ 
    (a % 13 = 0 ∨ b % 13 = 0) ∧
    (a < b) ∧ 
    (a + b > c) ∧ 
    (a + b + c = 24) :=
sorry

end right_triangle_min_perimeter_multiple_13_l230_23028


namespace intersection_P_M_l230_23088

open Set Int

def P : Set ℤ := {x | 0 ≤ x ∧ x < 3}

def M : Set ℤ := {x | x^2 ≤ 9}

theorem intersection_P_M : P ∩ M = {0, 1, 2} := by
  sorry

end intersection_P_M_l230_23088


namespace sum_of_abc_l230_23057

theorem sum_of_abc (a b c : ℝ) (h : (a - 5)^2 + (b - 6)^2 + (c - 7)^2 = 0) :
  a + b + c = 18 :=
sorry

end sum_of_abc_l230_23057


namespace unique_solution_of_fraction_eq_l230_23004

theorem unique_solution_of_fraction_eq (x : ℝ) : (1 / (x - 1) = 2 / (x - 2)) ↔ (x = 0) :=
by
  sorry

end unique_solution_of_fraction_eq_l230_23004


namespace g_range_l230_23089

variable {R : Type*} [LinearOrderedRing R]

-- Let y = f(x) be a function defined on R with a period of 1
def periodic (f : R → R) : Prop :=
  ∀ x, f (x + 1) = f x

-- If g(x) = f(x) + 2x
def g (f : R → R) (x : R) : R := f x + 2 * x

-- If the range of g(x) on the interval [1,2] is [-1,5]
def rangeCondition (f : R → R) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → -1 ≤ g f x ∧ g f x ≤ 5

-- Then the range of the function g(x) on the interval [-2020,2020] is [-4043,4041]
theorem g_range (f : R → R) 
  (hf_periodic : periodic f) 
  (hf_range : rangeCondition f) : 
  ∀ x, -2020 ≤ x ∧ x ≤ 2020 → -4043 ≤ g f x ∧ g f x ≤ 4041 :=
sorry

end g_range_l230_23089


namespace area_of_square_on_AD_l230_23097

theorem area_of_square_on_AD :
  ∃ (AB BC CD AD : ℝ),
    (∃ AB_sq BC_sq CD_sq AD_sq : ℝ,
      AB_sq = 25 ∧ BC_sq = 49 ∧ CD_sq = 64 ∧ 
      AB = Real.sqrt AB_sq ∧ BC = Real.sqrt BC_sq ∧ CD = Real.sqrt CD_sq ∧
      AD_sq = AB^2 + BC^2 + CD^2 ∧ AD = Real.sqrt AD_sq ∧ AD_sq = 138
    ) :=
by
  sorry

end area_of_square_on_AD_l230_23097


namespace sum_of_infinite_series_l230_23003

theorem sum_of_infinite_series :
  ∑' n, (1 : ℝ) / ((2 * n + 1)^2 - (2 * n - 1)^2) * ((1 : ℝ) / (2 * n - 1)^2 - (1 : ℝ) / (2 * n + 1)^2) = 1 :=
sorry

end sum_of_infinite_series_l230_23003


namespace marble_selection_probability_l230_23006

theorem marble_selection_probability :
  let total_marbles := 9
  let selected_marbles := 4
  let total_ways := Nat.choose total_marbles selected_marbles
  let red_marbles := 3
  let blue_marbles := 3
  let green_marbles := 3
  let ways_one_red := Nat.choose red_marbles 1
  let ways_two_blue := Nat.choose blue_marbles 2
  let ways_one_green := Nat.choose green_marbles 1
  let favorable_outcomes := ways_one_red * ways_two_blue * ways_one_green
  (favorable_outcomes : ℚ) / total_ways = 3 / 14 :=
by
  sorry

end marble_selection_probability_l230_23006


namespace sharona_bought_more_pencils_l230_23035

-- Define constants for the amounts paid
def amount_paid_jamar : ℚ := 1.43
def amount_paid_sharona : ℚ := 1.87

-- Define the function that computes the number of pencils given the price per pencil and total amount paid
def num_pencils (amount_paid : ℚ) (price_per_pencil : ℚ) : ℚ := amount_paid / price_per_pencil

-- Define the theorem stating that Sharona bought 4 more pencils than Jamar
theorem sharona_bought_more_pencils {price_per_pencil : ℚ} (h_price : price_per_pencil > 0) :
  num_pencils amount_paid_sharona price_per_pencil = num_pencils amount_paid_jamar price_per_pencil + 4 :=
sorry

end sharona_bought_more_pencils_l230_23035


namespace sign_of_c_l230_23030

/-
Define the context and conditions as Lean axioms.
-/

variables (a b c : ℝ)

-- Axiom: The sum of coefficients is less than zero
axiom h1 : a + b + c < 0

-- Axiom: The quadratic equation has no real roots, thus the discriminant is less than zero
axiom h2 : (b^2 - 4*a*c) < 0

/-
Formal statement of the proof problem:
-/

theorem sign_of_c : c < 0 :=
by
  -- We state that the proof of c < 0 follows from the given axioms
  sorry

end sign_of_c_l230_23030


namespace solutions_of_system_l230_23081

theorem solutions_of_system (x y z : ℝ) :
    (x^2 - y = z^2) → (y^2 - z = x^2) → (z^2 - x = y^2) →
    (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
    (x = 1 ∧ y = 0 ∧ z = -1) ∨ 
    (x = 0 ∧ y = -1 ∧ z = 1) ∨ 
    (x = -1 ∧ y = 1 ∧ z = 0) := by
  sorry

end solutions_of_system_l230_23081


namespace abigail_fence_building_time_l230_23020

def abigail_time_per_fence (total_built: ℕ) (additional_hours: ℕ) (total_fences: ℕ): ℕ :=
  (additional_hours * 60) / (total_fences - total_built)

theorem abigail_fence_building_time :
  abigail_time_per_fence 10 8 26 = 30 :=
sorry

end abigail_fence_building_time_l230_23020


namespace liza_final_balance_l230_23084

theorem liza_final_balance :
  let monday_balance := 800
  let tuesday_rent := 450
  let wednesday_deposit := 1500
  let thursday_electric_bill := 117
  let thursday_internet_bill := 100
  let thursday_groceries (balance : ℝ) := 0.2 * balance
  let friday_interest (balance : ℝ) := 0.02 * balance
  let saturday_phone_bill := 70
  let saturday_additional_deposit := 300
  let tuesday_balance := monday_balance - tuesday_rent
  let wednesday_balance := tuesday_balance + wednesday_deposit
  let thursday_balance_before_groceries := wednesday_balance - thursday_electric_bill - thursday_internet_bill
  let thursday_balance_after_groceries := thursday_balance_before_groceries - thursday_groceries thursday_balance_before_groceries
  let friday_balance := thursday_balance_after_groceries + friday_interest thursday_balance_after_groceries
  let saturday_balance_after_phone := friday_balance - saturday_phone_bill
  let final_balance := saturday_balance_after_phone + saturday_additional_deposit
  final_balance = 1562.528 :=
by
  let monday_balance := 800
  let tuesday_rent := 450
  let wednesday_deposit := 1500
  let thursday_electric_bill := 117
  let thursday_internet_bill := 100
  let thursday_groceries := 0.2 * (800 - 450 + 1500 - 117 - 100)
  let friday_interest := 0.02 * (800 - 450 + 1500 - 117 - 100 - 0.2 * (800 - 450 + 1500 - 117 - 100))
  let final_balance := 800 - 450 + 1500 - 117 - 100 - thursday_groceries + friday_interest - 70 + 300
  sorry

end liza_final_balance_l230_23084


namespace quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l230_23093

theorem quad_eq1_solution (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  sorry

theorem quad_eq2_solution (x : ℝ) : 2 * x^2 - 7 * x + 5 = 0 → x = 5 / 2 ∨ x = 1 :=
by
  sorry

theorem quad_eq3_solution (x : ℝ) : (x + 3)^2 - 2 * (x + 3) = 0 → x = -3 ∨ x = -1 :=
by
  sorry

end quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l230_23093


namespace remainder_expression_mod_l230_23001

/-- 
Let the positive integers s, t, u, and v leave remainders of 6, 9, 13, and 17, respectively, 
when divided by 23. Also, let s > t > u > v.
We want to prove that the remainder when 2 * (s - t) - 3 * (t - u) + 4 * (u - v) is divided by 23 is 12.
-/
theorem remainder_expression_mod (s t u v : ℕ) (hs : s % 23 = 6) (ht : t % 23 = 9) (hu : u % 23 = 13) (hv : v % 23 = 17)
  (h_gt : s > t ∧ t > u ∧ u > v) : (2 * (s - t) - 3 * (t - u) + 4 * (u - v)) % 23 = 12 :=
by
  sorry

end remainder_expression_mod_l230_23001


namespace horner_value_v2_l230_23083

def poly (x : ℤ) : ℤ := 208 + 9 * x^2 + 6 * x^4 + x^6

theorem horner_value_v2 : poly (-4) = ((((0 + -4) * -4 + 6) * -4 + 9) * -4 + 208) :=
by
  sorry

end horner_value_v2_l230_23083


namespace find_number_l230_23068

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
by
  sorry

end find_number_l230_23068


namespace quadrilateral_possible_rods_l230_23076

theorem quadrilateral_possible_rods (rods : Finset ℕ) (a b c : ℕ) (ha : a = 3) (hb : b = 7) (hc : c = 15)
  (hrods : rods = (Finset.range 31 \ {3, 7, 15})) :
  ∃ d, d ∈ rods ∧ 5 < d ∧ d < 25 ∧ rods.card - 2 = 17 := 
by
  sorry

end quadrilateral_possible_rods_l230_23076


namespace megan_average_speed_l230_23094

theorem megan_average_speed :
  ∃ s : ℕ, s = 100 / 3 ∧ ∃ (o₁ o₂ : ℕ), o₁ = 27472 ∧ o₂ = 27572 ∧ o₂ - o₁ = 100 :=
by
  sorry

end megan_average_speed_l230_23094


namespace password_problem_l230_23024

theorem password_problem (n : ℕ) :
  (n^4 - n * (n - 1) * (n - 2) * (n - 3) = 936) → n = 6 :=
by
  sorry

end password_problem_l230_23024


namespace rectangle_area_is_588_l230_23059

-- Definitions based on the conditions of the problem
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- The statement to prove that the area of the rectangle is 588
theorem rectangle_area_is_588 : length * width = 588 :=
by
  -- Omitted proof
  sorry

end rectangle_area_is_588_l230_23059


namespace quadratic_two_distinct_real_roots_l230_23050

theorem quadratic_two_distinct_real_roots (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + 2 * x1 - 3 = 0) ∧ (a * x2^2 + 2 * x2 - 3 = 0)) ↔ a > -1 / 3 := by
  sorry

end quadratic_two_distinct_real_roots_l230_23050


namespace vasya_claim_false_l230_23077

theorem vasya_claim_false :
  ∀ (weights : List ℕ), weights = [1, 2, 3, 4, 5, 6, 7] →
  (¬ ∃ (subset : List ℕ), subset.length = 3 ∧ 1 ∈ subset ∧
  ((weights.sum - subset.sum) = 14) ∧ (14 = 14)) :=
by
  sorry

end vasya_claim_false_l230_23077


namespace approximate_number_of_fish_in_pond_l230_23047

theorem approximate_number_of_fish_in_pond :
  (∃ N : ℕ, 
  (∃ tagged1 tagged2 : ℕ, tagged1 = 50 ∧ tagged2 = 10) ∧
  (∃ caught1 caught2 : ℕ, caught1 = 50 ∧ caught2 = 50) ∧
  ((tagged2 : ℝ) / caught2 = (tagged1 : ℝ) / (N : ℝ)) ∧
  N = 250) :=
sorry

end approximate_number_of_fish_in_pond_l230_23047


namespace find_smaller_number_l230_23026

-- Define the two numbers such that one is 3 times the other
def numbers (x : ℝ) := (x, 3 * x)

-- Define the condition that the sum of the two numbers is 14
def sum_condition (x y : ℝ) : Prop := x + y = 14

-- The theorem we want to prove
theorem find_smaller_number (x : ℝ) (hx : sum_condition x (3 * x)) : x = 3.5 :=
by
  -- Proof goes here
  sorry

end find_smaller_number_l230_23026


namespace parts_of_diagonal_in_rectangle_l230_23086

/-- Proving that a 24x60 rectangle divided by its diagonal results in 1512 parts --/

theorem parts_of_diagonal_in_rectangle :
  let m := 24
  let n := 60
  let gcd_mn := gcd m n
  let unit_squares := m * n
  let diagonal_intersections := m + n - gcd_mn
  unit_squares + diagonal_intersections = 1512 :=
by
  sorry

end parts_of_diagonal_in_rectangle_l230_23086


namespace volume_ratio_of_cubes_l230_23052

-- Given conditions
def edge_length_smaller_cube : ℝ := 6
def edge_length_larger_cube : ℝ := 12

-- Problem statement
theorem volume_ratio_of_cubes : 
  (edge_length_smaller_cube / edge_length_larger_cube) ^ 3 = (1 / 8) := 
by
  sorry

end volume_ratio_of_cubes_l230_23052


namespace cross_section_perimeter_l230_23041

-- Define the lengths of the diagonals AC and BD.
def length_AC : ℝ := 8
def length_BD : ℝ := 12

-- Define the perimeter calculation for the cross-section quadrilateral
-- that passes through the midpoint E of AB and is parallel to BD and AC.
theorem cross_section_perimeter :
  let side1 := length_AC / 2
  let side2 := length_BD / 2
  let perimeter := 2 * (side1 + side2)
  perimeter = 20 :=
by
  sorry

end cross_section_perimeter_l230_23041


namespace distance_is_correct_l230_23025

noncomputable def distance_from_home_to_forest_park : ℝ := 11  -- distance in kilometers

structure ProblemData where
  v : ℝ                  -- Xiao Wu's bicycling speed (in meters per minute)
  t_catch_up : ℝ          -- time it takes for father to catch up (in minutes)
  d_forest : ℝ            -- distance from catch-up point to forest park (in kilometers)
  t_remaining : ℝ        -- time remaining for Wu to reach park after wallet delivered (in minutes)
  bike_speed_factor : ℝ   -- speed factor of father's car compared to Wu's bike
  
open ProblemData

def problem_conditions : ProblemData :=
  { v := 350,
    t_catch_up := 7.5,
    d_forest := 3.5,
    t_remaining := 10,
    bike_speed_factor := 5 }

theorem distance_is_correct (data : ProblemData) :
  data.v = 350 →
  data.t_catch_up = 7.5 →
  data.d_forest = 3.5 →
  data.t_remaining = 10 →
  data.bike_speed_factor = 5 →
  distance_from_home_to_forest_park = 11 := 
by
  intros
  sorry

end distance_is_correct_l230_23025


namespace find_c_for_two_solutions_in_real_l230_23092

noncomputable def system_two_solutions (x y c : ℝ) : Prop := (|x + y| = 2007 ∧ |x - y| = c)

theorem find_c_for_two_solutions_in_real : ∃ c : ℝ, (∀ x y : ℝ, system_two_solutions x y c) ↔ (c = 0) :=
by
  sorry

end find_c_for_two_solutions_in_real_l230_23092


namespace distance_proof_l230_23091

/-- Maxwell's walking speed in km/h. -/
def Maxwell_speed := 4

/-- Time Maxwell walks before meeting Brad in hours. -/
def Maxwell_time := 10

/-- Brad's running speed in km/h. -/
def Brad_speed := 6

/-- Time Brad runs before meeting Maxwell in hours. -/
def Brad_time := 9

/-- Distance between Maxwell and Brad's homes in km. -/
def distance_between_homes : ℕ := 94

/-- Prove the distance between their homes is 94 km given the conditions. -/
theorem distance_proof 
  (h1 : Maxwell_speed * Maxwell_time = 40)
  (h2 : Brad_speed * Brad_time = 54) :
  Maxwell_speed * Maxwell_time + Brad_speed * Brad_time = distance_between_homes := 
by 
  sorry

end distance_proof_l230_23091


namespace quotient_remainder_difference_l230_23016

theorem quotient_remainder_difference :
  ∀ (N Q Q' R : ℕ), 
    N = 75 →
    N = 5 * Q →
    N = 34 * Q' + R →
    Q > R →
    Q - R = 8 :=
by
  intros N Q Q' R hN hDiv5 hDiv34 hGt
  sorry

end quotient_remainder_difference_l230_23016


namespace satisfy_eqn_l230_23021

/-- 
  Prove that the integer pairs (0, 1), (0, -1), (1, 0), (-1, 0), (2, 2), (-2, -2)
  are the only pairs that satisfy x^5 + y^5 = (x + y)^3
-/
theorem satisfy_eqn (x y : ℤ) : 
  (x, y) = (0, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (1, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (2, 2) ∨ (x, y) = (-2, -2) ↔ 
  x^5 + y^5 = (x + y)^3 := 
by 
  sorry

end satisfy_eqn_l230_23021


namespace max_value_sin2x_cos2x_l230_23090

open Real

theorem max_value_sin2x_cos2x (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) :
  (sin (2 * x) + cos (2 * x) ≤ sqrt 2) ∧
  (∃ y, (0 ≤ y ∧ y ≤ π / 2) ∧ (sin (2 * y) + cos (2 * y) = sqrt 2)) :=
by
  sorry

end max_value_sin2x_cos2x_l230_23090


namespace cupcakes_sold_l230_23056

theorem cupcakes_sold (initial_made sold additional final : ℕ) (h1 : initial_made = 42) (h2 : additional = 39) (h3 : final = 59) :
  (initial_made - sold + additional = final) -> sold = 22 :=
by
  intro h
  rw [h1, h2, h3] at h
  sorry

end cupcakes_sold_l230_23056


namespace evaluate_expression_eq_neg_one_evaluate_expression_only_value_l230_23036

variable (a y : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : a ≠ 2 * y)
variable (h3 : a ≠ -2 * y)

theorem evaluate_expression_eq_neg_one
  (h : y = -a / 3) :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) ) = -1 := 
sorry

theorem evaluate_expression_only_value :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) = -1 ) ↔ 
  y = -a / 3 := 
sorry

end evaluate_expression_eq_neg_one_evaluate_expression_only_value_l230_23036


namespace percentage_students_on_trip_l230_23045

variable (total_students : ℕ)
variable (students_more_than_100 : ℕ)
variable (students_on_trip : ℕ)
variable (percentage_more_than_100 : ℝ)
variable (percentage_not_more_than_100 : ℝ)

-- Given conditions
def condition_1 := percentage_more_than_100 = 0.16
def condition_2 := percentage_not_more_than_100 = 0.75

-- The final proof statement
theorem percentage_students_on_trip :
  percentage_more_than_100 * (total_students : ℝ) /
  ((1 - percentage_not_more_than_100)) / (total_students : ℝ) * 100 = 64 :=
by
  sorry

end percentage_students_on_trip_l230_23045


namespace final_selling_price_l230_23043

-- Conditions
variable (x : ℝ)
def original_price : ℝ := x
def first_discount : ℝ := 0.8 * x
def additional_reduction : ℝ := 10

-- Statement of the problem
theorem final_selling_price (x : ℝ) : (0.8 * x) - 10 = 0.8 * x - 10 :=
by sorry

end final_selling_price_l230_23043


namespace certain_number_l230_23008

-- Define the conditions as variables
variables {x : ℝ}

-- Define the proof problem
theorem certain_number (h : 0.15 * x = 0.025 * 450) : x = 75 :=
sorry

end certain_number_l230_23008


namespace problem_lean_statement_l230_23055

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x + π / 6)

theorem problem_lean_statement : 
  (∀ x, g x = 2 * cos (2 * x)) ∧ (∀ x, g (x) = g (-x)) ∧ (∀ x, g (x + π) = g (x)) :=
  sorry

end problem_lean_statement_l230_23055


namespace average_weight_of_whole_class_l230_23032

theorem average_weight_of_whole_class :
  let students_A := 26
  let students_B := 34
  let avg_weight_A := 50
  let avg_weight_B := 30
  let total_weight_A := avg_weight_A * students_A
  let total_weight_B := avg_weight_B * students_B
  let total_weight_class := total_weight_A + total_weight_B
  let total_students_class := students_A + students_B
  let avg_weight_class := total_weight_class / total_students_class
  avg_weight_class = 38.67 :=
by {
  sorry -- Proof is not required as per instructions
}

end average_weight_of_whole_class_l230_23032


namespace train_passing_time_l230_23080

theorem train_passing_time
  (length_A : ℝ) (length_B : ℝ) (time_A : ℝ) (speed_B : ℝ) 
  (Dir_opposite : true) 
  (passenger_on_A_time : time_A = 10)
  (length_of_A : length_A = 150)
  (length_of_B : length_B = 200)
  (relative_speed : speed_B = length_B / time_A) :
  ∃ x : ℝ, length_A / x = length_B / time_A ∧ x = 7.5 :=
by
  -- conditions stated
  sorry

end train_passing_time_l230_23080


namespace geometric_progression_terms_l230_23064

theorem geometric_progression_terms (a b r : ℝ) (n : ℕ) (h1 : 0 < r) (h2: a ≠ 0) (h3 : b = a * r^(n-1)) :
  n = 1 + (Real.log (b / a)) / (Real.log r) :=
by sorry

end geometric_progression_terms_l230_23064
