import Mathlib

namespace NUMINAMATH_GPT_joan_games_attended_l646_64640
-- Mathematical definitions based on the provided conditions

def total_games_played : ℕ := 864
def games_missed_by_Joan : ℕ := 469

-- Theorem statement
theorem joan_games_attended : total_games_played - games_missed_by_Joan = 395 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_joan_games_attended_l646_64640


namespace NUMINAMATH_GPT_general_term_of_sequence_l646_64667

theorem general_term_of_sequence (n : ℕ) :
  ∃ (a : ℕ → ℚ),
    a 1 = 1 / 2 ∧ 
    a 2 = -2 ∧ 
    a 3 = 9 / 2 ∧ 
    a 4 = -8 ∧ 
    a 5 = 25 / 2 ∧ 
    ∀ n, a n = (-1) ^ (n + 1) * (n ^ 2 / 2) := 
by
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l646_64667


namespace NUMINAMATH_GPT_thirteen_pow_2023_mod_1000_l646_64645

theorem thirteen_pow_2023_mod_1000 :
  (13^2023) % 1000 = 99 :=
sorry

end NUMINAMATH_GPT_thirteen_pow_2023_mod_1000_l646_64645


namespace NUMINAMATH_GPT_find_original_number_l646_64687

noncomputable def three_digit_number (d e f : ℕ) := 100 * d + 10 * e + f

/-- Given conditions and the sum S, determine the original three-digit number -/
theorem find_original_number (S : ℕ) (d e f : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9)
  (h2 : 0 ≤ e ∧ e ≤ 9) (h3 : 0 ≤ f ∧ f ≤ 9) (h4 : S = 4321) :
  three_digit_number d e f = 577 :=
sorry


end NUMINAMATH_GPT_find_original_number_l646_64687


namespace NUMINAMATH_GPT_find_angle_x_l646_64635

theorem find_angle_x (x : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ)
  (h₁ : α = 45)
  (h₂ : β = 3 * x)
  (h₃ : γ = x)
  (h₄ : α + β + γ = 180) :
  x = 33.75 :=
sorry

end NUMINAMATH_GPT_find_angle_x_l646_64635


namespace NUMINAMATH_GPT_slope_reciprocal_and_a_bounds_l646_64631

theorem slope_reciprocal_and_a_bounds (x : ℝ) (f g : ℝ → ℝ) 
    (h1 : ∀ x, f x = Real.log x - a * (x - 1)) 
    (h2 : ∀ x, g x = Real.exp x) :
    ((∀ k₁ k₂, (∃ x₁, k₁ = deriv f x₁) ∧ (∃ x₂, k₂ = deriv g x₂) ∧ k₁ * k₂ = 1) 
    ↔ (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 ∨ a = 0) :=
by
  sorry

end NUMINAMATH_GPT_slope_reciprocal_and_a_bounds_l646_64631


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l646_64658

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : 2 * a - 3 * b + 5 = 0) (h₂ : 2 * a + 3 * b - 13 = 0) :
  ∃ p : ℝ, p = 7 ∨ p = 8 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l646_64658


namespace NUMINAMATH_GPT_trigonometric_identity_l646_64611

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l646_64611


namespace NUMINAMATH_GPT_sin_value_l646_64657

theorem sin_value (α : ℝ) (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (α + Real.pi / 6) = -3 / 5) : 
  Real.sin (2 * α + Real.pi / 12) = -17 * Real.sqrt 2 / 50 := 
sorry

end NUMINAMATH_GPT_sin_value_l646_64657


namespace NUMINAMATH_GPT_sum_first_12_terms_l646_64654

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 ^ (n - 1) else 2 * n - 1

def S (n : ℕ) : ℕ := 
  (Finset.range n).sum a

theorem sum_first_12_terms : S 12 = 1443 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_12_terms_l646_64654


namespace NUMINAMATH_GPT_no_prime_for_equation_l646_64671

theorem no_prime_for_equation (x k : ℕ) (p : ℕ) (h_prime : p.Prime) (h_eq : x^5 + 2 * x + 3 = p^k) : False := 
sorry

end NUMINAMATH_GPT_no_prime_for_equation_l646_64671


namespace NUMINAMATH_GPT_elective_course_schemes_l646_64607

theorem elective_course_schemes : Nat.choose 4 2 = 6 := by
  sorry

end NUMINAMATH_GPT_elective_course_schemes_l646_64607


namespace NUMINAMATH_GPT_min_distance_sq_l646_64688

theorem min_distance_sq (x y : ℝ) (h : x - y - 1 = 0) : (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_min_distance_sq_l646_64688


namespace NUMINAMATH_GPT_good_apples_count_l646_64694

theorem good_apples_count (total_apples : ℕ) (rotten_percentage : ℝ) (good_apples : ℕ) (h1 : total_apples = 75) (h2 : rotten_percentage = 0.12) :
  good_apples = (1 - rotten_percentage) * total_apples := by
  sorry

end NUMINAMATH_GPT_good_apples_count_l646_64694


namespace NUMINAMATH_GPT_prove_value_of_expressions_l646_64647

theorem prove_value_of_expressions (a b : ℕ) 
  (h₁ : 2^a = 8^b) 
  (h₂ : a + 2 * b = 5) : 
  2^a + 8^b = 16 := 
by 
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_prove_value_of_expressions_l646_64647


namespace NUMINAMATH_GPT_equal_saturdays_and_sundays_l646_64604

theorem equal_saturdays_and_sundays (start_day : ℕ) (h : start_day < 7) :
  ∃! d, (d < 7 ∧ ((d + 2) % 7 = 0 → (d = 5))) :=
by
  sorry

end NUMINAMATH_GPT_equal_saturdays_and_sundays_l646_64604


namespace NUMINAMATH_GPT_functional_eq_zero_function_l646_64673

theorem functional_eq_zero_function (f : ℝ → ℝ) (k : ℝ) (h : ∀ x y : ℝ, f (f x + f y + k * x * y) = x * f y + y * f x) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_functional_eq_zero_function_l646_64673


namespace NUMINAMATH_GPT_round_table_legs_l646_64621

theorem round_table_legs:
  ∀ (chairs tables disposed chairs_legs tables_legs : ℕ) (total_legs : ℕ),
  chairs = 80 →
  chairs_legs = 5 →
  tables = 20 →
  disposed = 40 * chairs / 100 →
  total_legs = 300 →
  total_legs - (chairs - disposed) * chairs_legs = tables * tables_legs →
  tables_legs = 3 :=
by 
  intros chairs tables disposed chairs_legs tables_legs total_legs
  sorry

end NUMINAMATH_GPT_round_table_legs_l646_64621


namespace NUMINAMATH_GPT_age_of_20th_student_l646_64649

theorem age_of_20th_student (avg_age_20 : ℕ) (avg_age_9 : ℕ) (avg_age_10 : ℕ) :
  (avg_age_20 = 20) →
  (avg_age_9 = 11) →
  (avg_age_10 = 24) →
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  (age_20th = 61) :=
by
  intros h1 h2 h3
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  sorry

end NUMINAMATH_GPT_age_of_20th_student_l646_64649


namespace NUMINAMATH_GPT_count_divisors_of_100000_l646_64682

theorem count_divisors_of_100000 : 
  ∃ n : ℕ, n = 36 ∧ ∀ k : ℕ, (k ∣ 100000) → ∃ (i j : ℕ), 0 ≤ i ∧ i ≤ 5 ∧ 0 ≤ j ∧ j ≤ 5 ∧ k = 2^i * 5^j := by
  sorry

end NUMINAMATH_GPT_count_divisors_of_100000_l646_64682


namespace NUMINAMATH_GPT_original_number_l646_64677

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000

def permutations_sum (a b c : ℕ) : ℕ :=
  let abc := 100 * a + 10 * b + c
  let acb := 100 * a + 10 * c + b
  let bac := 100 * b + 10 * a + c
  let bca := 100 * b + 10 * c + a
  let cab := 100 * c + 10 * a + b
  let cba := 100 * c + 10 * b + a
  abc + acb + bac + bca + cab + cba

theorem original_number (abc : ℕ) (a b c : ℕ) :
  is_three_digit abc →
  abc = 100 * a + 10 * b + c →
  permutations_sum a b c = 3194 →
  abc = 358 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l646_64677


namespace NUMINAMATH_GPT_find_arrays_l646_64681

theorem find_arrays (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a ∣ b * c * d - 1 ∧ b ∣ a * c * d - 1 ∧ c ∣ a * b * d - 1 ∧ d ∣ a * b * c - 1 →
  (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨
  (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) := by
  sorry

end NUMINAMATH_GPT_find_arrays_l646_64681


namespace NUMINAMATH_GPT_Greg_more_than_Sharon_l646_64678

-- Define the harvest amounts
def Greg_harvest : ℝ := 0.4
def Sharon_harvest : ℝ := 0.1

-- Show that Greg harvested 0.3 more acres than Sharon
theorem Greg_more_than_Sharon : Greg_harvest - Sharon_harvest = 0.3 := by
  sorry

end NUMINAMATH_GPT_Greg_more_than_Sharon_l646_64678


namespace NUMINAMATH_GPT_total_students_playing_one_sport_l646_64697

noncomputable def students_playing_at_least_one_sport (total_students B S Ba C B_S B_Ba B_C S_Ba C_S C_Ba B_C_S: ℕ) : ℕ :=
  B + S + Ba + C - B_S - B_Ba - B_C - S_Ba - C_S - C_Ba + B_C_S

theorem total_students_playing_one_sport : 
  students_playing_at_least_one_sport 200 50 60 35 80 10 15 20 25 30 5 10 = 130 := by
  sorry

end NUMINAMATH_GPT_total_students_playing_one_sport_l646_64697


namespace NUMINAMATH_GPT_max_workers_l646_64636

theorem max_workers (S a n : ℕ) (h1 : n > 0) (h2 : S > 0) (h3 : a > 0)
  (h4 : (S:ℚ) / (a * n) > (3 * S:ℚ) / (a * (n + 5))) :
  2 * n + 5 = 9 := 
by
  sorry

end NUMINAMATH_GPT_max_workers_l646_64636


namespace NUMINAMATH_GPT_find_x_l646_64675

namespace MathProof

variables {a b x : ℝ}
variables (h1 : a > 0) (h2 : b > 0)

theorem find_x (h3 : (a^2)^(2 * b) = a^b * x^b) : x = a^3 :=
by sorry

end MathProof

end NUMINAMATH_GPT_find_x_l646_64675


namespace NUMINAMATH_GPT_count_even_numbers_l646_64630

theorem count_even_numbers (a b : ℕ) (h1 : a > 300) (h2 : b ≤ 600) (h3 : ∀ n, 300 < n ∧ n ≤ 600 → n % 2 = 0) : 
  ∃ c : ℕ, c = 150 :=
by
  sorry

end NUMINAMATH_GPT_count_even_numbers_l646_64630


namespace NUMINAMATH_GPT_first_reduction_percentage_l646_64606

theorem first_reduction_percentage (P : ℝ) (x : ℝ) (h : 0.30 * (1 - x / 100) * P = 0.225 * P) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_first_reduction_percentage_l646_64606


namespace NUMINAMATH_GPT_Katy_jellybeans_l646_64627

variable (Matt Matilda Steve Katy : ℕ)

def jellybean_relationship (Matt Matilda Steve Katy : ℕ) : Prop :=
  (Matt = 10 * Steve) ∧
  (Matilda = Matt / 2) ∧
  (Steve = 84) ∧
  (Katy = 3 * Matilda) ∧
  (Katy = Matt / 2)

theorem Katy_jellybeans : ∃ Katy, jellybean_relationship Matt Matilda Steve Katy ∧ Katy = 1260 := by
  sorry

end NUMINAMATH_GPT_Katy_jellybeans_l646_64627


namespace NUMINAMATH_GPT_total_votes_l646_64622

theorem total_votes (V : ℕ) (h1 : ∃ c : ℕ, c = 84) (h2 : ∃ m : ℕ, m = 476) (h3 : ∃ d : ℕ, d = ((84 * V - 16 * V) / 100)) : 
  V = 700 := 
by 
  sorry 

end NUMINAMATH_GPT_total_votes_l646_64622


namespace NUMINAMATH_GPT_investment_A_l646_64613

-- Define constants B and C's investment values, C's share, and total profit.
def B_investment : ℕ := 8000
def C_investment : ℕ := 9000
def C_share : ℕ := 36000
def total_profit : ℕ := 88000

-- Problem statement to prove
theorem investment_A (A_investment : ℕ) : 
  (A_investment + B_investment + C_investment = 17000) → 
  (C_investment * total_profit = C_share * (A_investment + B_investment + C_investment)) →
  A_investment = 5000 :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_investment_A_l646_64613


namespace NUMINAMATH_GPT_sqrt_of_9_is_3_l646_64690

theorem sqrt_of_9_is_3 {x : ℝ} (h₁ : x * x = 9) (h₂ : x ≥ 0) : x = 3 := sorry

end NUMINAMATH_GPT_sqrt_of_9_is_3_l646_64690


namespace NUMINAMATH_GPT_max_blocks_fit_in_box_l646_64655

def box_dimensions : ℕ × ℕ × ℕ := (4, 6, 2)
def block_dimensions : ℕ × ℕ × ℕ := (3, 2, 1)
def block_volume := 6
def box_volume := 48

theorem max_blocks_fit_in_box (box_dimensions : ℕ × ℕ × ℕ)
    (block_dimensions : ℕ × ℕ × ℕ) : 
  (box_volume / block_volume = 8) := 
by
  sorry

end NUMINAMATH_GPT_max_blocks_fit_in_box_l646_64655


namespace NUMINAMATH_GPT_votes_for_winning_candidate_l646_64618

-- Define the variables and conditions
variable (V : ℝ) -- Total number of votes
variable (W : ℝ) -- Votes for the winner

-- Condition 1: The winner received 75% of the votes
axiom winner_votes: W = 0.75 * V

-- Condition 2: The winner won by 500 votes
axiom win_by_500: W - 0.25 * V = 500

-- The statement we want to prove
theorem votes_for_winning_candidate : W = 750 :=
by sorry

end NUMINAMATH_GPT_votes_for_winning_candidate_l646_64618


namespace NUMINAMATH_GPT_tangent_line_parabola_l646_64619

theorem tangent_line_parabola (k : ℝ) (tangent : ∀ y : ℝ, ∃ x : ℝ, 4 * x + 3 * y + k = 0 ∧ y^2 = 12 * x) : 
  k = 27 / 4 :=
sorry

end NUMINAMATH_GPT_tangent_line_parabola_l646_64619


namespace NUMINAMATH_GPT_garden_area_increase_l646_64623

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_increase_l646_64623


namespace NUMINAMATH_GPT_olivia_initial_money_l646_64672

theorem olivia_initial_money (spent_supermarket : ℕ) (spent_showroom : ℕ) (left_money : ℕ) (initial_money : ℕ) :
  spent_supermarket = 31 → spent_showroom = 49 → left_money = 26 → initial_money = spent_supermarket + spent_showroom + left_money → initial_money = 106 :=
by
  intros h_supermarket h_showroom h_left h_initial 
  rw [h_supermarket, h_showroom, h_left] at h_initial
  exact h_initial

end NUMINAMATH_GPT_olivia_initial_money_l646_64672


namespace NUMINAMATH_GPT_barbara_total_cost_l646_64639

-- Definitions based on the given conditions
def steak_cost_per_pound : ℝ := 15.00
def steak_quantity : ℝ := 4.5
def chicken_cost_per_pound : ℝ := 8.00
def chicken_quantity : ℝ := 1.5

def expected_total_cost : ℝ := 42.00

-- The main proposition we need to prove
theorem barbara_total_cost :
  steak_cost_per_pound * steak_quantity + chicken_cost_per_pound * chicken_quantity = expected_total_cost :=
by
  sorry

end NUMINAMATH_GPT_barbara_total_cost_l646_64639


namespace NUMINAMATH_GPT_math_proof_problem_l646_64632

theorem math_proof_problem : 
  (325 - Real.sqrt 125) / 425 = 65 - 5 := 
by sorry

end NUMINAMATH_GPT_math_proof_problem_l646_64632


namespace NUMINAMATH_GPT_symmetric_axis_of_quadratic_fn_l646_64698

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 + 8 * x + 9 

-- State the theorem that the axis of symmetry for the quadratic function y = x^2 + 8x + 9 is x = -4
theorem symmetric_axis_of_quadratic_fn : ∃ h : ℝ, h = -4 ∧ ∀ x, quadratic_function x = quadratic_function (2 * h - x) :=
by sorry

end NUMINAMATH_GPT_symmetric_axis_of_quadratic_fn_l646_64698


namespace NUMINAMATH_GPT_sum_first_n_terms_l646_64616

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_n_terms
  (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h_a2a4 : a 2 + a 4 = 8)
  (h_common_diff : ∀ n : ℕ, a (n + 1) = a n + 2) :
  ∃ S_n : ℕ → ℝ, ∀ n : ℕ, S_n n = n^2 - n :=
by 
  sorry

end NUMINAMATH_GPT_sum_first_n_terms_l646_64616


namespace NUMINAMATH_GPT_magician_trick_success_l646_64651

theorem magician_trick_success {n : ℕ} (T_pos : ℕ) (deck_size : ℕ := 52) (discard_count : ℕ := 51):
  (T_pos = 1 ∨ T_pos = deck_size) → ∃ strategy : Type, ∀ spectator_choice : ℕ, (spectator_choice ≤ deck_size) → 
                          ((T_pos = 1 → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)
                          ∧ (T_pos = deck_size → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)) :=
sorry

end NUMINAMATH_GPT_magician_trick_success_l646_64651


namespace NUMINAMATH_GPT_min_value_of_sequence_l646_64660

theorem min_value_of_sequence :
  ∃ (a : ℕ → ℤ), a 1 = 0 ∧ (∀ n : ℕ, n ≥ 2 → |a n| = |a (n - 1) + 1|) ∧ (a 1 + a 2 + a 3 + a 4 = -2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_sequence_l646_64660


namespace NUMINAMATH_GPT_smaller_square_area_percentage_l646_64605

noncomputable def area_percentage_of_smaller_square :=
  let side_length_large_square : ℝ := 4
  let area_large_square := side_length_large_square ^ 2
  let side_length_smaller_square := side_length_large_square / 5
  let area_smaller_square := side_length_smaller_square ^ 2
  (area_smaller_square / area_large_square) * 100
theorem smaller_square_area_percentage :
  area_percentage_of_smaller_square = 4 := 
sorry

end NUMINAMATH_GPT_smaller_square_area_percentage_l646_64605


namespace NUMINAMATH_GPT_cart_total_distance_l646_64670

-- Definitions for the conditions
def first_section_distance := (15/2) * (8 + (8 + 14 * 10))
def second_section_distance := (15/2) * (148 + (148 + 14 * 6))

-- Combining both distances
def total_distance := first_section_distance + second_section_distance

-- Statement to be proved
theorem cart_total_distance:
  total_distance = 4020 :=
by
  sorry

end NUMINAMATH_GPT_cart_total_distance_l646_64670


namespace NUMINAMATH_GPT_cuboid_volume_l646_64674

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : b * c = 5) (h3 : a * c = 15) : a * b * c = 15 :=
sorry

end NUMINAMATH_GPT_cuboid_volume_l646_64674


namespace NUMINAMATH_GPT_emma_bank_account_balance_l646_64661

def initial_amount : ℝ := 230
def withdrawn_amount : ℝ := 60
def deposit_amount : ℝ := 2 * withdrawn_amount
def final_amount : ℝ := initial_amount - withdrawn_amount + deposit_amount

theorem emma_bank_account_balance : final_amount = 290 := 
by 
  -- Definitions have already been stated; the proof is not required
  sorry

end NUMINAMATH_GPT_emma_bank_account_balance_l646_64661


namespace NUMINAMATH_GPT_proof_problem_l646_64642

theorem proof_problem (x : ℕ) (h : 320 / (x + 26) = 4) : x = 54 := 
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l646_64642


namespace NUMINAMATH_GPT_xyz_value_l646_64679

variable (x y z : ℝ)

theorem xyz_value :
  (x + y + z) * (x*y + x*z + y*z) = 36 →
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 24 →
  x * y * z = 4 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_xyz_value_l646_64679


namespace NUMINAMATH_GPT_fraction_value_l646_64665

theorem fraction_value
  (x y z : ℝ)
  (h1 : x / 2 = y / 3)
  (h2 : y / 3 = z / 5)
  (h3 : 2 * x + y ≠ 0) :
  (x + y - 3 * z) / (2 * x + y) = -10 / 7 := by
  -- Add sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_fraction_value_l646_64665


namespace NUMINAMATH_GPT_left_vertex_of_ellipse_l646_64676

theorem left_vertex_of_ellipse : 
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 6*x + 8 = 0 ∧ x = a - 5) ∧
  2 * b = 8 → left_vertex = (-5, 0) :=
sorry

end NUMINAMATH_GPT_left_vertex_of_ellipse_l646_64676


namespace NUMINAMATH_GPT_matt_assignment_problems_l646_64610

theorem matt_assignment_problems (P : ℕ) (h : 5 * P - 2 * P = 60) : P = 20 :=
by
  sorry

end NUMINAMATH_GPT_matt_assignment_problems_l646_64610


namespace NUMINAMATH_GPT_garden_perimeter_is_48_l646_64620

def square_garden_perimeter (pond_area garden_remaining_area : ℕ) : ℕ :=
  let garden_area := pond_area + garden_remaining_area
  let side_length := Int.natAbs (Int.sqrt garden_area)
  4 * side_length

theorem garden_perimeter_is_48 :
  square_garden_perimeter 20 124 = 48 :=
  by
  sorry

end NUMINAMATH_GPT_garden_perimeter_is_48_l646_64620


namespace NUMINAMATH_GPT_least_number_subtracted_divisible_l646_64666

theorem least_number_subtracted_divisible (n : ℕ) (d : ℕ) (h : n = 1234567) (k : d = 37) :
  n % d = 13 :=
by 
  rw [h, k]
  sorry

end NUMINAMATH_GPT_least_number_subtracted_divisible_l646_64666


namespace NUMINAMATH_GPT_triangle_leg_ratio_l646_64643

theorem triangle_leg_ratio :
  ∀ (a b : ℝ) (h₁ : a = 4) (h₂ : b = 2 * Real.sqrt 5),
    ((a / b) = (2 * Real.sqrt 5) / 5) :=
by
  intros a b h₁ h₂
  sorry

end NUMINAMATH_GPT_triangle_leg_ratio_l646_64643


namespace NUMINAMATH_GPT_chocolate_and_gum_l646_64656

/--
Kolya says that two chocolate bars are more expensive than five gum sticks, 
while Sasha claims that three chocolate bars are more expensive than eight gum sticks. 
When this was checked, only one of them was right. Is it true that seven chocolate bars 
are more expensive than nineteen gum sticks?
-/
theorem chocolate_and_gum (c g : ℝ) (hk : 2 * c > 5 * g) (hs : 3 * c > 8 * g) (only_one_correct : ¬((2 * c > 5 * g) ∧ (3 * c > 8 * g)) ∧ (2 * c > 5 * g ∨ 3 * c > 8 * g)) : 7 * c < 19 * g :=
by
  sorry

end NUMINAMATH_GPT_chocolate_and_gum_l646_64656


namespace NUMINAMATH_GPT_determine_b_for_constant_remainder_l646_64650

theorem determine_b_for_constant_remainder (b : ℚ) :
  ∃ r : ℚ, ∀ x : ℚ,  (12 * x^3 - 9 * x^2 + b * x + 8) / (3 * x^2 - 4 * x + 2) = r ↔ b = -4 / 3 :=
by sorry

end NUMINAMATH_GPT_determine_b_for_constant_remainder_l646_64650


namespace NUMINAMATH_GPT_sum_series_eq_three_l646_64668

theorem sum_series_eq_three : 
  ∑' (k : ℕ), (k^2 : ℝ) / (2^k : ℝ) = 3 := sorry

end NUMINAMATH_GPT_sum_series_eq_three_l646_64668


namespace NUMINAMATH_GPT_minimize_expression_l646_64648

theorem minimize_expression (x y : ℝ) (k : ℝ) (h : k = -1) : (xy + k)^2 + (x - y)^2 ≥ 0 ∧ (∀ x y : ℝ, (xy + k)^2 + (x - y)^2 = 0 ↔ k = -1) := 
by {
  sorry
}

end NUMINAMATH_GPT_minimize_expression_l646_64648


namespace NUMINAMATH_GPT_min_value_geom_seq_l646_64663

theorem min_value_geom_seq (a : ℕ → ℝ) (r m n : ℕ) (h_geom : ∃ r, ∀ i, a (i + 1) = a i * r)
  (h_ratio : r = 2) (h_a_m : 4 * a 1 = a m) :
  ∃ (m n : ℕ), (m + n = 6) → (1 / m + 4 / n) = 3 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_geom_seq_l646_64663


namespace NUMINAMATH_GPT_roots_of_quadratic_l646_64609

theorem roots_of_quadratic (m n : ℝ) (h₁ : m + n = -2) (h₂ : m * n = -2022) (h₃ : ∀ x, x^2 + 2 * x - 2022 = 0 → x = m ∨ x = n) :
  m^2 + 3 * m + n = 2020 :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_l646_64609


namespace NUMINAMATH_GPT_sandy_balloons_l646_64662

def balloons_problem (A S T : ℕ) : ℕ :=
  T - (A + S)

theorem sandy_balloons : balloons_problem 37 39 104 = 28 := by
  sorry

end NUMINAMATH_GPT_sandy_balloons_l646_64662


namespace NUMINAMATH_GPT_price_of_scooter_l646_64628

-- Assume upfront_payment and percentage_upfront are given
def upfront_payment : ℝ := 240
def percentage_upfront : ℝ := 0.20

noncomputable
def total_price (upfront_payment : ℝ) (percentage_upfront : ℝ) : ℝ :=
  (upfront_payment / percentage_upfront)

theorem price_of_scooter : total_price upfront_payment percentage_upfront = 1200 :=
  by
    sorry

end NUMINAMATH_GPT_price_of_scooter_l646_64628


namespace NUMINAMATH_GPT_frog_probability_l646_64680

noncomputable def frog_escape_prob (P : ℕ → ℚ) : Prop :=
  P 0 = 0 ∧
  P 11 = 1 ∧
  (∀ N, 0 < N ∧ N < 11 → 
    P N = (N + 1) / 12 * P (N - 1) + (1 - (N + 1) / 12) * P (N + 1)) ∧
  P 2 = 72 / 167

theorem frog_probability : ∃ P : ℕ → ℚ, frog_escape_prob P :=
sorry

end NUMINAMATH_GPT_frog_probability_l646_64680


namespace NUMINAMATH_GPT_number_of_non_congruent_triangles_l646_64693

theorem number_of_non_congruent_triangles :
  ∃ q : ℕ, q = 3 ∧ 
    (∀ (a b : ℕ), (a ≤ 2 ∧ 2 ≤ b) → (a + 2 > b) ∧ (a + b > 2) ∧ (2 + b > a) →
    (q = 3)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_non_congruent_triangles_l646_64693


namespace NUMINAMATH_GPT_find_a_l646_64624

-- Define the variables and conditions
variable (a x y : ℤ)

-- Given conditions
def x_value := (x = 2)
def y_value := (y = 1)
def equation := (a * x - y = 3)

-- The theorem to prove
theorem find_a : x_value x → y_value y → equation a x y → a = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_a_l646_64624


namespace NUMINAMATH_GPT_function_has_local_minimum_at_zero_l646_64608

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * (x - 1))

def is_local_minimum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, abs (y - x) < ε → f x ≤ f y

theorem function_has_local_minimum_at_zero :
  -4 < 0 ∧ 0 < 1 ∧ is_local_minimum f 0 := 
sorry

end NUMINAMATH_GPT_function_has_local_minimum_at_zero_l646_64608


namespace NUMINAMATH_GPT_gcd_polynomial_eval_l646_64603

theorem gcd_polynomial_eval (b : ℤ) (h : ∃ (k : ℤ), b = 570 * k) :
  Int.gcd (4 * b ^ 3 + b ^ 2 + 5 * b + 95) b = 95 := by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_eval_l646_64603


namespace NUMINAMATH_GPT_intervals_of_monotonicity_l646_64600

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos (x + Real.pi / 3)

theorem intervals_of_monotonicity :
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi) → (f x ≤ f (7 * Real.pi / 12 + k * Real.pi)))) ∧
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (-5 * Real.pi / 12 + k * Real.pi) (Real.pi / 12 + k * Real.pi) → (f x ≥ f (Real.pi / 12 + k * Real.pi)))) ∧
  (f (Real.pi / 2) = -Real.sqrt 3) ∧
  (f (Real.pi / 12) = 1 - Real.sqrt 3 / 2) := sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_l646_64600


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l646_64614

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * sqrt 3 * cos x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, 2 * cos x)

noncomputable def f (x : ℝ) : ℝ := 
  let a_dot_b := (a x).1 * (b x).1 + (a x).2 * (b x).2
  let b_norm_sq := (b x).1 ^ 2 + (b x).2 ^ 2
  a_dot_b + b_norm_sq + 3 / 2

theorem smallest_positive_period_of_f :
  ∀ x, f (x + π) = f x := sorry

theorem symmetry_center_of_f :
  ∃ k : ℤ, ∀ x, f x = 5 ↔ x = (-π / 12 + k * (π / 2) : ℝ) := sorry

theorem range_of_f_in_interval :
  ∀ x, (π / 6 ≤ x ∧ x ≤ π / 2) → (5 / 2 ≤ f x ∧ f x ≤ 10) := sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l646_64614


namespace NUMINAMATH_GPT_domain_correct_l646_64625

noncomputable def domain_function (x : ℝ) : Prop :=
  (4 * x - 3 > 0) ∧ (Real.log (4 * x - 3) / Real.log 0.5 > 0)

theorem domain_correct : {x : ℝ | domain_function x} = {x : ℝ | (3 / 4 : ℝ) < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_correct_l646_64625


namespace NUMINAMATH_GPT_least_positive_integer_l646_64601

theorem least_positive_integer (n : ℕ) : 
  (530 + n) % 4 = 0 → n = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_positive_integer_l646_64601


namespace NUMINAMATH_GPT_max_min_value_sum_l646_64626

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x) * Real.sin (x - 2) + x + 1

theorem max_min_value_sum (M m : ℝ) 
  (hM : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≤ M)
  (hm : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≥ m)
  (hM_max : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = M)
  (hm_min : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = m)
  : M + m = 6 :=
sorry

end NUMINAMATH_GPT_max_min_value_sum_l646_64626


namespace NUMINAMATH_GPT_greatest_integer_x_l646_64652

theorem greatest_integer_x (x : ℤ) : 
  (∃ n : ℤ, (x^2 + 4*x + 10) = n * (x - 4)) → x ≤ 46 := 
by
  sorry

end NUMINAMATH_GPT_greatest_integer_x_l646_64652


namespace NUMINAMATH_GPT_inequality_a6_b6_l646_64696

theorem inequality_a6_b6 (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end NUMINAMATH_GPT_inequality_a6_b6_l646_64696


namespace NUMINAMATH_GPT_sarah_marry_age_l646_64644

/-- Sarah is 9 years old. -/
def Sarah_age : ℕ := 9

/-- Sarah's name has 5 letters. -/
def Sarah_name_length : ℕ := 5

/-- The game's rule is to add the number of letters in the player's name 
    to twice the player's age. -/
def game_rule (name_length age : ℕ) : ℕ :=
  name_length + 2 * age

/-- Prove that Sarah will get married at the age of 23. -/
theorem sarah_marry_age : game_rule Sarah_name_length Sarah_age = 23 := 
  sorry

end NUMINAMATH_GPT_sarah_marry_age_l646_64644


namespace NUMINAMATH_GPT_range_of_a_l646_64685

theorem range_of_a {a : ℝ} (h1 : ∀ x : ℝ, x - a ≥ 0 → 2 * x - 10 < 0) :
  3 < a ∧ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l646_64685


namespace NUMINAMATH_GPT_base_triangle_not_equilateral_l646_64689

-- Define the lengths of the lateral edges
def SA := 1
def SB := 2
def SC := 4

-- Main theorem: the base triangle is not equilateral
theorem base_triangle_not_equilateral 
  (a : ℝ)
  (equilateral : a = a)
  (triangle_inequality1 : SA + SB > a)
  (triangle_inequality2 : SA + a > SC) : 
  a ≠ a :=
by 
  sorry

end NUMINAMATH_GPT_base_triangle_not_equilateral_l646_64689


namespace NUMINAMATH_GPT_smallest_c_value_l646_64699

theorem smallest_c_value :
  ∃ a b c : ℕ, a * b * c = 3990 ∧ a + b + c = 56 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
by {
  -- Skipping proof as instructed
  sorry
}

end NUMINAMATH_GPT_smallest_c_value_l646_64699


namespace NUMINAMATH_GPT_while_loop_output_correct_do_while_loop_output_correct_l646_64664

def while_loop (a : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (7 - i)).map (λ n => (i + n, a + n + 1))

def do_while_loop (x : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (10 - i + 1)).map (λ n => (i + n, x + (n + 1) * 10))

theorem while_loop_output_correct : while_loop 2 1 = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)] := 
sorry

theorem do_while_loop_output_correct : do_while_loop 100 1 = [(1, 110), (2, 120), (3, 130), (4, 140), (5, 150), (6, 160), (7, 170), (8, 180), (9, 190), (10, 200)] :=
sorry

end NUMINAMATH_GPT_while_loop_output_correct_do_while_loop_output_correct_l646_64664


namespace NUMINAMATH_GPT_inequality_proof_l646_64692

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (habc : a * b * c = 1)

theorem inequality_proof :
  (a + 1 / b)^2 + (b + 1 / c)^2 + (c + 1 / a)^2 ≥ 3 * (a + b + c + 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l646_64692


namespace NUMINAMATH_GPT_intersection_points_polar_coords_l646_64641

theorem intersection_points_polar_coords :
  (∀ (x y : ℝ), ((x - 4)^2 + (y - 5)^2 = 25 ∧ (x^2 + y^2 - 2*y = 0)) →
  (∃ (ρ θ : ℝ), ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
    ((x, y) = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧
    ((ρ = 2 ∧ θ = Real.pi / 2) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4)))) :=
sorry

end NUMINAMATH_GPT_intersection_points_polar_coords_l646_64641


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l646_64633

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l646_64633


namespace NUMINAMATH_GPT_black_white_ratio_l646_64684

theorem black_white_ratio 
  (x y : ℕ) 
  (h1 : (y - 1) * 7 = x * 9) 
  (h2 : y * 5 = (x - 1) * 7) : 
  y - x = 7 := 
by 
  sorry

end NUMINAMATH_GPT_black_white_ratio_l646_64684


namespace NUMINAMATH_GPT_find_balcony_seat_cost_l646_64686

-- Definitions based on conditions
variable (O B : ℕ) -- Number of orchestra tickets and cost of balcony ticket
def orchestra_ticket_cost : ℕ := 12
def total_tickets : ℕ := 370
def total_cost : ℕ := 3320
def tickets_difference : ℕ := 190

-- Lean statement to prove the cost of a balcony seat
theorem find_balcony_seat_cost :
  (2 * O + tickets_difference = total_tickets) ∧
  (orchestra_ticket_cost * O + B * (O + tickets_difference) = total_cost) →
  B = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_balcony_seat_cost_l646_64686


namespace NUMINAMATH_GPT_total_distance_total_distance_alt_l646_64617

variable (D : ℝ) -- declare the variable for the total distance

-- defining the conditions
def speed_walking : ℝ := 4 -- speed in km/hr when walking
def speed_running : ℝ := 8 -- speed in km/hr when running
def total_time : ℝ := 3.75 -- total time in hours

-- proving that D = 10 given the conditions
theorem total_distance 
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time) : 
    D = 10 := 
sorry

-- Alternative theorem version declaring variables directly
theorem total_distance_alt
    (speed_walking speed_running total_time : ℝ) -- declaring variables
    (D : ℝ) -- the total distance
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time)
    (hw : speed_walking = 4)
    (hr : speed_running = 8)
    (ht : total_time = 3.75) : 
    D = 10 := 
sorry

end NUMINAMATH_GPT_total_distance_total_distance_alt_l646_64617


namespace NUMINAMATH_GPT_simplify_expression_l646_64683

theorem simplify_expression (y : ℝ) : 5 * y + 7 * y + 8 * y = 20 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l646_64683


namespace NUMINAMATH_GPT_inv_f_of_neg3_l646_64691

def f (x : Real) : Real := 5 - 2 * x

theorem inv_f_of_neg3 : f⁻¹ (-3) = 4 :=
by
  sorry

end NUMINAMATH_GPT_inv_f_of_neg3_l646_64691


namespace NUMINAMATH_GPT_multiply_then_divide_eq_multiply_l646_64659

theorem multiply_then_divide_eq_multiply (x : ℚ) :
  (x * (2 / 5)) / (3 / 7) = x * (14 / 15) :=
by
  sorry

end NUMINAMATH_GPT_multiply_then_divide_eq_multiply_l646_64659


namespace NUMINAMATH_GPT_projected_increase_l646_64602

theorem projected_increase (R : ℝ) (P : ℝ) 
  (h1 : ∃ P, ∀ (R : ℝ), 0.9 * R = 0.75 * (R + (P / 100) * R)) 
  (h2 : ∀ (R : ℝ), R > 0) :
  P = 20 :=
by
  sorry

end NUMINAMATH_GPT_projected_increase_l646_64602


namespace NUMINAMATH_GPT_find_length_DE_l646_64653

theorem find_length_DE (AB AC BC : ℝ) (angleA : ℝ) 
                         (DE DF EF : ℝ) (angleD : ℝ) :
  AB = 9 → AC = 11 → BC = 7 →
  angleA = 60 → DE = 3 → DF = 5.5 → EF = 2.5 →
  angleD = 60 →
  DE = 9 * 2.5 / 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_find_length_DE_l646_64653


namespace NUMINAMATH_GPT_remainder_div_82_l646_64629

theorem remainder_div_82 (x : ℤ) (h : ∃ k : ℤ, x + 17 = 41 * k + 22) : (x % 82 = 5) :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_82_l646_64629


namespace NUMINAMATH_GPT_length_ab_square_l646_64612

theorem length_ab_square (s a : ℝ) (h_square : s = 2 * a) (h_area : 3000 = 1/2 * (s + (s - 2 * a)) * s) : 
  s = 20 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_length_ab_square_l646_64612


namespace NUMINAMATH_GPT_small_triangle_area_ratio_l646_64637

theorem small_triangle_area_ratio (a b n : ℝ) 
  (h₀ : a > 0) 
  (h₁ : b > 0) 
  (h₂ : n > 0) 
  (h₃ : ∃ (r s : ℝ), r > 0 ∧ s > 0 ∧ (1/2) * a * r = n * a * b ∧ s = (b^2) / (2 * n * b)) :
  (b^2 / (4 * n)) / (a * b) = 1 / (4 * n) :=
by sorry

end NUMINAMATH_GPT_small_triangle_area_ratio_l646_64637


namespace NUMINAMATH_GPT_range_of_omega_l646_64695

noncomputable def function_with_highest_points (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + Real.pi / 4)

theorem range_of_omega (ω : ℝ) (hω : ω > 0)
  (h : ∀ x ∈ Set.Icc 0 1, 2 * Real.sin (ω * x + Real.pi / 4) = 2) :
  Set.Icc (17 * Real.pi / 4) (25 * Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_omega_l646_64695


namespace NUMINAMATH_GPT_probability_P_plus_S_mod_7_correct_l646_64638

noncomputable def probability_P_plus_S_mod_7 : ℚ :=
  let n := 60
  let total_ways := (n * (n - 1)) / 2
  let num_special_pairs := total_ways - ((52 * 51) / 2)
  num_special_pairs / total_ways

theorem probability_P_plus_S_mod_7_correct :
  probability_P_plus_S_mod_7 = 148 / 590 :=
by
  rw [probability_P_plus_S_mod_7]
  sorry

end NUMINAMATH_GPT_probability_P_plus_S_mod_7_correct_l646_64638


namespace NUMINAMATH_GPT_intersection_with_x_axis_l646_64669

theorem intersection_with_x_axis (t : ℝ) (x y : ℝ) 
  (h1 : x = -2 + 5 * t) 
  (h2 : y = 1 - 2 * t) 
  (h3 : y = 0) : x = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_intersection_with_x_axis_l646_64669


namespace NUMINAMATH_GPT_horner_method_multiplications_additions_count_l646_64646

-- Define the polynomial function
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - 2 * x^2 + 4 * x - 6

-- Define the property we want to prove
theorem horner_method_multiplications_additions_count : 
  ∃ (multiplications additions : ℕ), multiplications = 4 ∧ additions = 4 := 
by
  sorry

end NUMINAMATH_GPT_horner_method_multiplications_additions_count_l646_64646


namespace NUMINAMATH_GPT_radioactive_decay_minimum_years_l646_64634

noncomputable def min_years (a : ℝ) (n : ℕ) : Prop :=
  (a * (1 - 3 / 4) ^ n ≤ a * 1 / 100)

theorem radioactive_decay_minimum_years (a : ℝ) (h : 0 < a) : ∃ n : ℕ, min_years a n ∧ n = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_radioactive_decay_minimum_years_l646_64634


namespace NUMINAMATH_GPT_gcd_1021_2729_l646_64615

theorem gcd_1021_2729 : Int.gcd 1021 2729 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1021_2729_l646_64615
