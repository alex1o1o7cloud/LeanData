import Mathlib

namespace problems_per_worksheet_l280_28040

theorem problems_per_worksheet (total_worksheets : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) (h1 : total_worksheets = 15) (h2 : graded_worksheets = 7) (h3 : remaining_problems = 24) : (remaining_problems / (total_worksheets - graded_worksheets)) = 3 :=
by {
  sorry
}

end problems_per_worksheet_l280_28040


namespace simplify_and_evaluate_l280_28069

-- Define the variables
variables (x y : ℝ)

-- Define the expression
def expression := 2 * x * y + (3 * x * y - 2 * y^2) - 2 * (x * y - y^2)

-- Introduce the conditions
theorem simplify_and_evaluate : 
  (x = -1) → (y = 2) → expression x y = -6 := 
by 
  intro hx hy 
  sorry

end simplify_and_evaluate_l280_28069


namespace dvd_sold_168_l280_28010

/-- 
Proof that the number of DVDs sold (D) is 168 given the conditions:
1) D = 1.6 * C
2) D + C = 273 
-/
theorem dvd_sold_168 (C D : ℝ) (h1 : D = 1.6 * C) (h2 : D + C = 273) : D = 168 := 
sorry

end dvd_sold_168_l280_28010


namespace green_chameleon_increase_l280_28013

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l280_28013


namespace average_of_consecutive_integers_l280_28049

variable (c : ℕ)
variable (d : ℕ)

-- Given condition: d == (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7
def condition1 : Prop := d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7

-- The theorem to prove
theorem average_of_consecutive_integers : condition1 c d → 
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7 + d + 8 + d + 9) / 10 = c + 9 :=
sorry

end average_of_consecutive_integers_l280_28049


namespace money_distribution_l280_28031

theorem money_distribution (A B C : ℝ) (h1 : A + B + C = 1000) (h2 : B + C = 600) (h3 : C = 300) : A + C = 700 := by
  sorry

end money_distribution_l280_28031


namespace determine_digit_l280_28077

theorem determine_digit (Θ : ℚ) (h : 312 / Θ = 40 + 2 * Θ) : Θ = 6 :=
sorry

end determine_digit_l280_28077


namespace volume_of_cut_pyramid_l280_28012

theorem volume_of_cut_pyramid
  (base_length : ℝ)
  (slant_length : ℝ)
  (cut_height : ℝ)
  (original_base_area : ℝ)
  (original_height : ℝ)
  (new_base_area : ℝ)
  (volume : ℝ)
  (h_base_length : base_length = 8 * Real.sqrt 2)
  (h_slant_length : slant_length = 10)
  (h_cut_height : cut_height = 3)
  (h_original_base_area : original_base_area = (base_length ^ 2) / 2)
  (h_original_height : original_height = Real.sqrt (slant_length ^ 2 - (base_length / Real.sqrt 2) ^ 2))
  (h_new_base_area : new_base_area = original_base_area / 4)
  (h_volume : volume = (1 / 3) * new_base_area * cut_height) :
  volume = 32 :=
by
  sorry

end volume_of_cut_pyramid_l280_28012


namespace range_of_a_l280_28033

structure PropositionP (a : ℝ) : Prop :=
  (h : 2 * a + 1 > 5)

structure PropositionQ (a : ℝ) : Prop :=
  (h : -1 ≤ a ∧ a ≤ 3)

theorem range_of_a (a : ℝ) (hp : PropositionP a ∨ PropositionQ a) (hq : ¬(PropositionP a ∧ PropositionQ a)) :
  (-1 ≤ a ∧ a ≤ 2) ∨ (a > 3) :=
sorry

end range_of_a_l280_28033


namespace speed_faster_train_correct_l280_28088

noncomputable def speed_faster_train_proof
  (time_seconds : ℝ) 
  (speed_slower_train : ℝ)
  (train_length_meters : ℝ) :
  Prop :=
  let time_hours := time_seconds / 3600
  let train_length_km := train_length_meters / 1000
  let total_distance_km := train_length_km + train_length_km
  let relative_speed_km_hr := total_distance_km / time_hours
  let speed_faster_train := relative_speed_km_hr + speed_slower_train
  speed_faster_train = 46

theorem speed_faster_train_correct :
  speed_faster_train_proof 36.00001 36 50.000013888888894 :=
by 
  -- proof steps would go here
  sorry

end speed_faster_train_correct_l280_28088


namespace incorrect_independence_test_conclusion_l280_28078

-- Definitions for each condition
def independence_test_principle_of_small_probability (A : Prop) : Prop :=
A  -- Statement A: The independence test is based on the principle of small probability.

def independence_test_conclusion_variability (C : Prop) : Prop :=
C  -- Statement C: Different samples may lead to different conclusions in the independence test.

def independence_test_not_the_only_method (D : Prop) : Prop :=
D  -- Statement D: The independence test is not the only method to determine whether two categorical variables are related.

-- Incorrect statement B
def independence_test_conclusion_always_correct (B : Prop) : Prop :=
B  -- Statement B: The conclusion drawn from the independence test is always correct.

-- Prove that statement B is incorrect given conditions A, C, and D
theorem incorrect_independence_test_conclusion (A B C D : Prop) 
  (hA : independence_test_principle_of_small_probability A)
  (hC : independence_test_conclusion_variability C)
  (hD : independence_test_not_the_only_method D) :
  ¬ independence_test_conclusion_always_correct B :=
sorry

end incorrect_independence_test_conclusion_l280_28078


namespace solution_l280_28007

noncomputable def problem (x : ℝ) : Prop :=
  0 < x ∧ (1/2 * (4 * x^2 - 1) = (x^2 - 50 * x - 20) * (x^2 + 25 * x + 10))

theorem solution (x : ℝ) (h : problem x) : x = 26 + Real.sqrt 677 :=
by
  sorry

end solution_l280_28007


namespace max_students_l280_28028

-- Definitions for the conditions
noncomputable def courses := ["Mathematics", "Physics", "Biology", "Music", "History", "Geography"]

def most_preferred (ranking : List String) : Prop :=
  "Mathematics" ∈ (ranking.take 2) ∨ "Mathematics" ∈ (ranking.take 3)

def least_preferred (ranking : List String) : Prop :=
  "Music" ∉ ranking.drop (ranking.length - 2)

def preference_constraints (ranking : List String) : Prop :=
  ranking.indexOf "History" < ranking.indexOf "Geography" ∧
  ranking.indexOf "Physics" < ranking.indexOf "Biology"

def all_rankings_unique (rankings : List (List String)) : Prop :=
  ∀ (r₁ r₂ : List String), r₁ ≠ r₂ → r₁ ∈ rankings → r₂ ∈ rankings → r₁ ≠ r₂

-- The goal statement
theorem max_students : 
  ∃ (rankings : List (List String)), 
  (∀ r ∈ rankings, most_preferred r) ∧
  (∀ r ∈ rankings, least_preferred r) ∧
  (∀ r ∈ rankings, preference_constraints r) ∧
  all_rankings_unique rankings ∧
  rankings.length = 44 :=
sorry

end max_students_l280_28028


namespace percentage_of_a_l280_28066

theorem percentage_of_a (x a : ℝ) (paise_in_rupee : ℝ := 100) (a_value : a = 160 * paise_in_rupee) (h : (x / 100) * a = 80) : x = 0.5 :=
by sorry

end percentage_of_a_l280_28066


namespace penny_difference_l280_28017

variables (p : ℕ)

/-- Liam and Mia have certain numbers of fifty-cent coins. This theorem proves the difference 
    in their total value in pennies. 
-/
theorem penny_difference:
  (3 * p + 2) * 50 - (2 * p + 7) * 50 = 50 * p - 250 :=
by
  sorry

end penny_difference_l280_28017


namespace smaller_triangle_area_14_365_l280_28098

noncomputable def smaller_triangle_area (A : ℝ) (H_reduction : ℝ) : ℝ :=
  A * (H_reduction)^2

theorem smaller_triangle_area_14_365 :
  smaller_triangle_area 34 0.65 = 14.365 :=
by
  -- Proof will be provided here
  sorry

end smaller_triangle_area_14_365_l280_28098


namespace greatest_number_of_pieces_leftover_l280_28073

theorem greatest_number_of_pieces_leftover (y : ℕ) (q r : ℕ) 
  (h : y = 6 * q + r) (hrange : r < 6) : r = 5 := sorry

end greatest_number_of_pieces_leftover_l280_28073


namespace points_on_circle_l280_28086

theorem points_on_circle (t : ℝ) : (∃ (x y : ℝ), x = Real.cos (2 * t) ∧ y = Real.sin (2 * t) ∧ (x^2 + y^2 = 1)) := by
  sorry

end points_on_circle_l280_28086


namespace tangent_line_eq_at_0_max_min_values_l280_28079

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_eq_at_0 : ∀ x : ℝ, x = 0 → f x = 1 :=
by
  sorry

theorem max_min_values : (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f 0 ≥ f x) ∧ (f (Real.pi / 2) = -Real.pi / 2) :=
by
  sorry

end tangent_line_eq_at_0_max_min_values_l280_28079


namespace range_of_a_l280_28046

def p (a : ℝ) : Prop := ∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ (x^2 + (y^2) / a = 1)
def q (a : ℝ) : Prop := ∃ x0 : ℝ, 4^x0 - 2^x0 - a ≤ 0

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → -1/4 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l280_28046


namespace negation_equivalence_l280_28032

-- Declare the condition for real solutions of a quadratic equation
def has_real_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a * x + 1 = 0

-- Define the proposition p
def prop_p : Prop :=
  ∀ a : ℝ, a ≥ 0 → has_real_solutions a

-- Define the negation of p
def neg_prop_p : Prop :=
  ∃ a : ℝ, a ≥ 0 ∧ ¬ has_real_solutions a

-- The theorem stating the equivalence of p's negation to its formulated negation.
theorem negation_equivalence : neg_prop_p = ¬ prop_p := by
  sorry

end negation_equivalence_l280_28032


namespace no_integer_pairs_satisfy_equation_l280_28075

def equation_satisfaction (m n : ℤ) : Prop :=
  m^3 + 3 * m^2 + 2 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ (m n : ℤ), equation_satisfaction m n :=
by
  sorry

end no_integer_pairs_satisfy_equation_l280_28075


namespace cosine_neg_alpha_l280_28053

theorem cosine_neg_alpha (alpha : ℝ) (h : Real.sin (π/2 + alpha) = -3/5) : Real.cos (-alpha) = -3/5 :=
sorry

end cosine_neg_alpha_l280_28053


namespace naomi_saw_wheels_l280_28044

theorem naomi_saw_wheels :
  let regular_bikes := 7
  let children's_bikes := 11
  let wheels_per_regular_bike := 2
  let wheels_per_children_bike := 4
  let total_wheels := regular_bikes * wheels_per_regular_bike + children's_bikes * wheels_per_children_bike
  total_wheels = 58 := by
  sorry

end naomi_saw_wheels_l280_28044


namespace jessica_flowers_problem_l280_28003

theorem jessica_flowers_problem
(initial_roses initial_daisies : ℕ)
(thrown_roses thrown_daisies : ℕ)
(current_roses current_daisies : ℕ)
(cut_roses cut_daisies : ℕ)
(h_initial_roses : initial_roses = 21)
(h_initial_daisies : initial_daisies = 17)
(h_thrown_roses : thrown_roses = 34)
(h_thrown_daisies : thrown_daisies = 25)
(h_current_roses : current_roses = 15)
(h_current_daisies : current_daisies = 10)
(h_cut_roses : cut_roses = (thrown_roses - initial_roses) + current_roses)
(h_cut_daisies : cut_daisies = (thrown_daisies - initial_daisies) + current_daisies) :
thrown_roses + thrown_daisies - (cut_roses + cut_daisies) = 13 := by
  sorry

end jessica_flowers_problem_l280_28003


namespace stratified_sampling_female_students_l280_28002

-- Definitions from conditions
def male_students : ℕ := 800
def female_students : ℕ := 600
def drawn_male_students : ℕ := 40
def total_students : ℕ := 1400

-- Proof statement
theorem stratified_sampling_female_students : 
  (female_students * drawn_male_students) / male_students = 30 :=
by
  -- substitute and simplify
  sorry

end stratified_sampling_female_students_l280_28002


namespace copies_made_in_half_hour_l280_28039

theorem copies_made_in_half_hour :
  let copies_per_minute_machine1 := 40
  let copies_per_minute_machine2 := 55
  let time_minutes := 30
  (copies_per_minute_machine1 * time_minutes) + (copies_per_minute_machine2 * time_minutes) = 2850 := by
    sorry

end copies_made_in_half_hour_l280_28039


namespace total_emails_vacation_l280_28030

def day_1_emails : ℕ := 16
def day_2_emails : ℕ := day_1_emails / 2
def day_3_emails : ℕ := day_2_emails / 2
def day_4_emails : ℕ := day_3_emails / 2

def total_emails : ℕ := day_1_emails + day_2_emails + day_3_emails + day_4_emails

theorem total_emails_vacation : total_emails = 30 := by
  -- Use "sorry" to skip the proof as per instructions.
  sorry

end total_emails_vacation_l280_28030


namespace integer_cubed_fraction_l280_28009

theorem integer_cubed_fraction
  (a b : ℕ)
  (hab : 0 < b ∧ 0 < a)
  (h : (a^2 + b^2) % (a - b)^2 = 0) :
  (a^3 + b^3) % (a - b)^3 = 0 :=
by sorry

end integer_cubed_fraction_l280_28009


namespace first_term_to_common_difference_ratio_l280_28006

theorem first_term_to_common_difference_ratio (a d : ℝ) 
  (h : (14 / 2) * (2 * a + 13 * d) = 3 * (7 / 2) * (2 * a + 6 * d)) :
  a / d = 4 :=
by
  sorry

end first_term_to_common_difference_ratio_l280_28006


namespace second_number_removed_l280_28062

theorem second_number_removed (S : ℝ) (X : ℝ) (h1 : S / 50 = 38) (h2 : (S - 45 - X) / 48 = 37.5) : X = 55 :=
by
  sorry

end second_number_removed_l280_28062


namespace tan_alpha_l280_28016

variable (α : ℝ)
variable (H_cos : Real.cos α = 12/13)
variable (H_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)

theorem tan_alpha :
  Real.tan α = -5/12 :=
sorry

end tan_alpha_l280_28016


namespace problem_statement_l280_28091

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem problem_statement : star A B = {1, 7} := by
  sorry

end problem_statement_l280_28091


namespace largest_of_four_consecutive_primes_l280_28042

noncomputable def sum_of_primes_is_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ Prime (p1 + p2 + p3 + p4)

theorem largest_of_four_consecutive_primes :
  ∃ p1 p2 p3 p4, 
  sum_of_primes_is_prime p1 p2 p3 p4 ∧ 
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ 
  (p1, p2, p3, p4) = (2, 3, 5, 7) ∧ 
  max p1 (max p2 (max p3 p4)) = 7 :=
by {
  sorry                                 -- solve this in Lean
}

end largest_of_four_consecutive_primes_l280_28042


namespace geometric_common_ratio_arithmetic_sequence_l280_28067

theorem geometric_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : S 3 = a 1 * (1 - q^3) / (1 - q)) (h2 : S 3 = 3 * a 1) :
  q = 2 ∨ q^3 = - (1 / 2) := by
  sorry

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h : S 3 = a 1 * (1 - q^3) / (1 - q))
  (h3 : 2 * S 9 = S 3 + S 6) (h4 : q ≠ 1) :
  a 2 + a 5 = 2 * a 8 := by
  sorry

end geometric_common_ratio_arithmetic_sequence_l280_28067


namespace calculate_expression_l280_28095

theorem calculate_expression (a b c : ℤ) (ha : a = 3) (hb : b = 7) (hc : c = 2) :
  ((a * b - c) - (a + b * c)) - ((a * c - b) - (a - b * c)) = -8 :=
by
  rw [ha, hb, hc]  -- Substitute a, b, c with 3, 7, 2 respectively
  sorry  -- Placeholder for the proof

end calculate_expression_l280_28095


namespace min_value_proof_l280_28056

noncomputable def min_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2

theorem min_value_proof :
  ∃ α β : ℝ, min_value_expression α β = 48 := by
  sorry

end min_value_proof_l280_28056


namespace opposite_of_neg_three_l280_28080

def opposite (x : Int) : Int := -x

theorem opposite_of_neg_three : opposite (-3) = 3 := by
  -- To be proven using Lean
  sorry

end opposite_of_neg_three_l280_28080


namespace length_of_AB_is_1_l280_28050

variables {A B C : ℝ} -- Points defining the triangle vertices
variables {a b c : ℝ} -- Lengths of triangle sides opposite to angles A, B, C respectively
variables {α β γ : ℝ} -- Angles at points A B C
variables {s₁ s₂ s₃ : ℝ} -- Sin values of the angles

noncomputable def length_of_AB (a b c : ℝ) : ℝ :=
  if a + b + c = 4 ∧ a + b = 3 * c then 1 else 0

theorem length_of_AB_is_1 : length_of_AB a b c = 1 :=
by
  have h_perimeter : a + b + c = 4 := sorry
  have h_sin_condition : a + b = 3 * c := sorry
  simp [length_of_AB, h_perimeter, h_sin_condition]
  sorry

end length_of_AB_is_1_l280_28050


namespace largest_avg_5_l280_28082

def arithmetic_avg (a l : ℕ) : ℚ :=
  (a + l) / 2

def multiples_avg_2 (n : ℕ) : ℚ :=
  arithmetic_avg 2 (n - (n % 2))

def multiples_avg_3 (n : ℕ) : ℚ :=
  arithmetic_avg 3 (n - (n % 3))

def multiples_avg_4 (n : ℕ) : ℚ :=
  arithmetic_avg 4 (n - (n % 4))

def multiples_avg_5 (n : ℕ) : ℚ :=
  arithmetic_avg 5 (n - (n % 5))

def multiples_avg_6 (n : ℕ) : ℚ :=
  arithmetic_avg 6 (n - (n % 6))

theorem largest_avg_5 (n : ℕ) (h : n = 101) : 
  multiples_avg_5 n > multiples_avg_2 n ∧ 
  multiples_avg_5 n > multiples_avg_3 n ∧ 
  multiples_avg_5 n > multiples_avg_4 n ∧ 
  multiples_avg_5 n > multiples_avg_6 n :=
by
  sorry

end largest_avg_5_l280_28082


namespace value_of_a_plus_d_l280_28041

variable {R : Type} [LinearOrderedField R]
variables {a b c d : R}

theorem value_of_a_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 13 := by
  sorry

end value_of_a_plus_d_l280_28041


namespace directrix_of_parabola_l280_28047

-- Define the condition given in the problem
def parabola_eq (x y : ℝ) : Prop := x^2 = 2 * y

-- Define the directrix equation property we want to prove
theorem directrix_of_parabola (x : ℝ) :
  (∃ y : ℝ, parabola_eq x y) → (∃ y : ℝ, y = -1 / 2) :=
by sorry

end directrix_of_parabola_l280_28047


namespace tangent_alpha_l280_28081

open Real

noncomputable def a (α : ℝ) : ℝ × ℝ := (sin α, 2)
noncomputable def b (α : ℝ) : ℝ × ℝ := (-cos α, 1)

theorem tangent_alpha (α : ℝ) (h : ∀ k : ℝ, a α = (k • b α)) : tan α = -2 := by
  have h1 : sin α / -cos α = 2 := by sorry
  have h2 : tan α = -2 := by sorry
  exact h2

end tangent_alpha_l280_28081


namespace find_unknown_number_l280_28087

theorem find_unknown_number (x : ℤ) (h : (20 + 40 + 60) / 3 = 9 + (10 + 70 + x) / 3) : x = 13 :=
by
  sorry

end find_unknown_number_l280_28087


namespace perpendicular_condition_l280_28076

theorem perpendicular_condition (m : ℝ) : 
  (2 * (m + 1) * (m - 3) + 2 * (m - 3) = 0) ↔ (m = 3 ∨ m = -3) :=
by
  sorry

end perpendicular_condition_l280_28076


namespace unique_set_property_l280_28085

theorem unique_set_property (a b c : ℕ) (h1: 1 < a) (h2: 1 < b) (h3: 1 < c) 
    (gcd_ab_c: (Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1))
    (property_abc: (a * b) % c = (a * c) % b ∧ (a * c) % b = (b * c) % a) : 
    (a = 2 ∧ b = 3 ∧ c = 5) ∨ 
    (a = 2 ∧ b = 5 ∧ c = 3) ∨ 
    (a = 3 ∧ b = 2 ∧ c = 5) ∨ 
    (a = 3 ∧ b = 5 ∧ c = 2) ∨ 
    (a = 5 ∧ b = 2 ∧ c = 3) ∨ 
    (a = 5 ∧ b = 3 ∧ c = 2) := sorry

end unique_set_property_l280_28085


namespace units_digit_sum_l280_28059

theorem units_digit_sum (n : ℕ) (h : n > 0) : (35^n % 10) + (93^45 % 10) = 8 :=
by
  -- Since the units digit of 35^n is always 5 
  have h1 : 35^n % 10 = 5 := sorry
  -- Since the units digit of 93^45 is 3 (since 45 mod 4 = 1 and the pattern repeats every 4),
  have h2 : 93^45 % 10 = 3 := sorry
  -- Therefore, combining the units digits
  calc
    (35^n % 10) + (93^45 % 10)
    = 5 + 3 := by rw [h1, h2]
    _ = 8 := by norm_num

end units_digit_sum_l280_28059


namespace sandy_worked_days_l280_28063

-- Definitions based on the conditions
def total_hours_worked : ℕ := 45
def hours_per_day : ℕ := 9

-- The theorem that we need to prove
theorem sandy_worked_days : total_hours_worked / hours_per_day = 5 :=
by sorry

end sandy_worked_days_l280_28063


namespace number_of_women_per_table_l280_28093

theorem number_of_women_per_table
  (tables : ℕ) (men_per_table : ℕ) 
  (total_customers : ℕ) (total_tables : tables = 9) 
  (men_at_each_table : men_per_table = 3) 
  (customers : total_customers = 90) 
  (total_men : 3 * 9 = 27) 
  (total_women : 90 - 27 = 63) :
  (63 / 9 = 7) :=
by
  sorry

end number_of_women_per_table_l280_28093


namespace quadratic_inequality_l280_28097

theorem quadratic_inequality 
  (a b c : ℝ) 
  (h₁ : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1)
  (x : ℝ) 
  (hx : |x| ≤ 1) : 
  |c * x^2 + b * x + a| ≤ 2 := 
sorry

end quadratic_inequality_l280_28097


namespace annual_decrease_rate_l280_28092

theorem annual_decrease_rate
  (P0 : ℕ := 8000)
  (P2 : ℕ := 6480) :
  ∃ r : ℝ, 8000 * (1 - r / 100)^2 = 6480 ∧ r = 10 :=
by
  use 10
  sorry

end annual_decrease_rate_l280_28092


namespace find_speed_from_p_to_q_l280_28023

noncomputable def speed_from_p_to_q (v : ℝ) (d : ℝ) : Prop :=
  let return_speed := 1.5 * v
  let avg_speed := 75
  let total_distance := 2 * d
  let total_time := d / v + d / return_speed
  avg_speed = total_distance / total_time

theorem find_speed_from_p_to_q (v : ℝ) (d : ℝ) : speed_from_p_to_q v d → v = 62.5 :=
by
  intro h
  sorry

end find_speed_from_p_to_q_l280_28023


namespace factorize_problem_1_factorize_problem_2_l280_28014

theorem factorize_problem_1 (a b : ℝ) : -3 * a ^ 3 + 12 * a ^ 2 * b - 12 * a * b ^ 2 = -3 * a * (a - 2 * b) ^ 2 := 
sorry

theorem factorize_problem_2 (m n : ℝ) : 9 * (m + n) ^ 2 - (m - n) ^ 2 = 4 * (2 * m + n) * (m + 2 * n) := 
sorry

end factorize_problem_1_factorize_problem_2_l280_28014


namespace kerry_age_l280_28024

theorem kerry_age (cost_per_box : ℝ) (boxes_bought : ℕ) (candles_per_box : ℕ) (cakes : ℕ) 
  (total_cost : ℝ) (total_candles : ℕ) (candles_per_cake : ℕ) (age : ℕ) :
  cost_per_box = 2.5 →
  boxes_bought = 2 →
  candles_per_box = 12 →
  cakes = 3 →
  total_cost = 5 →
  total_cost = boxes_bought * cost_per_box →
  total_candles = boxes_bought * candles_per_box →
  candles_per_cake = total_candles / cakes →
  age = candles_per_cake →
  age = 8 :=
by
  intros
  sorry

end kerry_age_l280_28024


namespace expression_values_l280_28070

noncomputable def sign (x : ℝ) : ℝ := 
if x > 0 then 1 else -1

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ v ∈ ({-4, 0, 4} : Set ℝ), 
    sign a + sign b + sign c + sign (a * b * c) = v := by
  sorry

end expression_values_l280_28070


namespace find_f_2_l280_28026

theorem find_f_2 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 2 = 5 :=
sorry

end find_f_2_l280_28026


namespace bathroom_width_l280_28000

def length : ℝ := 4
def area : ℝ := 8
def width : ℝ := 2

theorem bathroom_width :
  area = length * width :=
by
  sorry

end bathroom_width_l280_28000


namespace common_points_count_l280_28051

variable (x y : ℝ)

def curve1 : Prop := x^2 + 4 * y^2 = 4
def curve2 : Prop := 4 * x^2 + y^2 = 4
def curve3 : Prop := x^2 + y^2 = 1

theorem common_points_count : ∀ (x y : ℝ), curve1 x y ∧ curve2 x y ∧ curve3 x y → false := by
  intros
  sorry

end common_points_count_l280_28051


namespace cubic_polynomial_value_at_3_and_neg3_l280_28020

variable (Q : ℝ → ℝ)
variable (a b c d m : ℝ)
variable (h1 : Q 1 = 5 * m)
variable (h0 : Q 0 = 2 * m)
variable (h_1 : Q (-1) = 6 * m)
variable (hQ : ∀ x, Q x = a * x^3 + b * x^2 + c * x + d)

theorem cubic_polynomial_value_at_3_and_neg3 :
  Q 3 + Q (-3) = 67 * m := by
  -- sorry is used to skip the proof
  sorry

end cubic_polynomial_value_at_3_and_neg3_l280_28020


namespace max_GREECE_val_l280_28068

variables (V E R I A G C : ℕ)
noncomputable def verify : Prop :=
  (V * 100 + E * 10 + R - (I * 10 + A)) = G^(R^E) * (G * 100 + R * 10 + E + E * 100 + C * 10 + E) ∧
  G ≠ 0 ∧ E ≠ 0 ∧ V ≠ 0 ∧ I ≠ 0 ∧
  V ≠ E ∧ V ≠ R ∧ V ≠ I ∧ V ≠ A ∧ V ≠ G ∧ V ≠ C ∧
  E ≠ R ∧ E ≠ I ∧ E ≠ A ∧ E ≠ G ∧ E ≠ C ∧
  R ≠ I ∧ R ≠ A ∧ R ≠ G ∧ R ≠ C ∧
  I ≠ A ∧ I ≠ G ∧ I ≠ C ∧
  A ≠ G ∧ A ≠ C ∧
  G ≠ C

theorem max_GREECE_val : ∃ V E R I A G C : ℕ, verify V E R I A G C ∧ (G * 100000 + R * 10000 + E * 1000 + E * 100 + C * 10 + E = 196646) :=
sorry

end max_GREECE_val_l280_28068


namespace length_of_picture_frame_l280_28021

theorem length_of_picture_frame (P W : ℕ) (hP : P = 30) (hW : W = 10) : ∃ L : ℕ, 2 * (L + W) = P ∧ L = 5 :=
by
  sorry

end length_of_picture_frame_l280_28021


namespace dogwood_trees_tomorrow_l280_28038

def initial_dogwood_trees : Nat := 7
def trees_planted_today : Nat := 3
def final_total_dogwood_trees : Nat := 12

def trees_after_today : Nat := initial_dogwood_trees + trees_planted_today
def trees_planted_tomorrow : Nat := final_total_dogwood_trees - trees_after_today

theorem dogwood_trees_tomorrow :
  trees_planted_tomorrow = 2 :=
by
  sorry

end dogwood_trees_tomorrow_l280_28038


namespace rice_mixing_ratio_l280_28054

-- Definitions based on conditions
def rice_1_price : ℝ := 6
def rice_2_price : ℝ := 8.75
def mixture_price : ℝ := 7.50

-- Proof of the required ratio
theorem rice_mixing_ratio (x y : ℝ) (h : (rice_1_price * x + rice_2_price * y) / (x + y) = mixture_price) :
  y / x = 6 / 5 :=
by 
  sorry

end rice_mixing_ratio_l280_28054


namespace conic_section_eccentricity_l280_28084

noncomputable def eccentricity (m : ℝ) : ℝ :=
if m = 2 then 1 / Real.sqrt 2 else
if m = -2 then Real.sqrt 3 else
0

theorem conic_section_eccentricity (m : ℝ) (h : 4 * 1 = m * m) :
  eccentricity m = 1 / Real.sqrt 2 ∨ eccentricity m = Real.sqrt 3 :=
by
  sorry

end conic_section_eccentricity_l280_28084


namespace glens_speed_is_37_l280_28036

/-!
# Problem Statement
Glen and Hannah drive at constant speeds toward each other on a highway. Glen drives at a certain speed G km/h. At some point, they pass by each other, and keep driving away from each other, maintaining their constant speeds. 
Glen is 130 km away from Hannah at 6 am and again at 11 am. Hannah is driving at 15 kilometers per hour.
Prove that Glen's speed is 37 km/h.
-/

def glens_speed (G : ℝ) : Prop :=
  ∃ G: ℝ, 
    (∃ H_speed : ℝ, H_speed = 15) ∧ -- Hannah's speed
    (∃ distance : ℝ, distance = 130) ∧ -- distance at 6 am and 11 am
    G + 15 = 260 / 5 -- derived equation from conditions

theorem glens_speed_is_37 : glens_speed 37 :=
by {
  sorry -- proof to be filled in
}

end glens_speed_is_37_l280_28036


namespace cos_difference_simplification_l280_28005

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  x - y = -1 / (2 * Real.sqrt 5) :=
sorry

end cos_difference_simplification_l280_28005


namespace find_rectangle_length_l280_28045

-- Define the problem conditions
def length_is_three_times_breadth (l b : ℕ) : Prop := l = 3 * b
def area_of_rectangle (l b : ℕ) : Prop := l * b = 6075

-- Define the theorem to prove the length of the rectangle given the conditions
theorem find_rectangle_length (l b : ℕ) (h1 : length_is_three_times_breadth l b) (h2 : area_of_rectangle l b) : l = 135 := 
sorry

end find_rectangle_length_l280_28045


namespace sum_xyz_eq_11sqrt5_l280_28029

noncomputable def x : ℝ :=
sorry

noncomputable def y : ℝ :=
sorry

noncomputable def z : ℝ :=
sorry

axiom pos_x : x > 0
axiom pos_y : y > 0
axiom pos_z : z > 0

axiom xy_eq_30 : x * y = 30
axiom xz_eq_60 : x * z = 60
axiom yz_eq_90 : y * z = 90

theorem sum_xyz_eq_11sqrt5 : x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_eq_11sqrt5_l280_28029


namespace arrange_animals_adjacent_l280_28074

theorem arrange_animals_adjacent:
  let chickens := 5
  let dogs := 3
  let cats := 6
  let rabbits := 4
  let total_animals := 18
  let group_orderings := 24 -- 4!
  let chicken_orderings := 120 -- 5!
  let dog_orderings := 6 -- 3!
  let cat_orderings := 720 -- 6!
  let rabbit_orderings := 24 -- 4!
  total_animals = chickens + dogs + cats + rabbits →
  chickens > 0 ∧ dogs > 0 ∧ cats > 0 ∧ rabbits > 0 →
  group_orderings * chicken_orderings * dog_orderings * cat_orderings * rabbit_orderings = 17863680 :=
  by intros; sorry

end arrange_animals_adjacent_l280_28074


namespace students_in_all_classes_l280_28034

theorem students_in_all_classes (total_students : ℕ) (students_photography : ℕ) (students_music : ℕ) (students_theatre : ℕ) (students_dance : ℕ) (students_at_least_two : ℕ) (students_in_all : ℕ) :
  total_students = 30 →
  students_photography = 15 →
  students_music = 18 →
  students_theatre = 12 →
  students_dance = 10 →
  students_at_least_two = 18 →
  students_in_all = 4 :=
by
  intros
  sorry

end students_in_all_classes_l280_28034


namespace system_solution_l280_28019

theorem system_solution (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -2 / 5 :=
by
  sorry

end system_solution_l280_28019


namespace new_students_l280_28071

theorem new_students (S_i : ℕ) (L : ℕ) (S_f : ℕ) (N : ℕ) 
  (h₁ : S_i = 11) 
  (h₂ : L = 6) 
  (h₃ : S_f = 47) 
  (h₄ : S_f = S_i - L + N) : 
  N = 42 :=
by 
  rw [h₁, h₂, h₃] at h₄
  sorry

end new_students_l280_28071


namespace Andy_and_Carlos_tie_for_first_l280_28027

def AndyLawnArea (A : ℕ) := 3 * A
def CarlosLawnArea (A : ℕ) := A / 4
def BethMowingRate := 90
def CarlosMowingRate := BethMowingRate / 3
def AndyMowingRate := BethMowingRate * 4

theorem Andy_and_Carlos_tie_for_first (A : ℕ) (hA_nonzero : 0 < A) :
  (AndyLawnArea A / AndyMowingRate) = (CarlosLawnArea A / CarlosMowingRate) ∧
  (AndyLawnArea A / AndyMowingRate) < (A / BethMowingRate) :=
by
  unfold AndyLawnArea CarlosLawnArea BethMowingRate CarlosMowingRate AndyMowingRate
  sorry

end Andy_and_Carlos_tie_for_first_l280_28027


namespace carrots_picked_by_mother_l280_28018

-- Define the conditions
def faye_picked : ℕ := 23
def good_carrots : ℕ := 12
def bad_carrots : ℕ := 16

-- Define the problem of the total number of carrots
def total_carrots : ℕ := good_carrots + bad_carrots

-- Define the mother's picked carrots
def mother_picked (total_faye : ℕ) (total : ℕ) := total - total_faye

-- State the theorem
theorem carrots_picked_by_mother (faye_picked : ℕ) (total_carrots : ℕ) : mother_picked faye_picked total_carrots = 5 := by
  sorry

end carrots_picked_by_mother_l280_28018


namespace containers_needed_l280_28004

-- Define the conditions: 
def weight_in_pounds : ℚ := 25 / 2
def ounces_per_pound : ℚ := 16
def ounces_per_container : ℚ := 50

-- Define the total weight in ounces
def total_weight_in_ounces := weight_in_pounds * ounces_per_pound

-- Theorem statement: Number of containers.
theorem containers_needed : total_weight_in_ounces / ounces_per_container = 4 := 
by
  -- Write the proof here
  sorry

end containers_needed_l280_28004


namespace rotation_volumes_l280_28083

theorem rotation_volumes (a b c V1 V2 V3 : ℝ) (h : a^2 + b^2 = c^2)
    (hV1 : V1 = (1 / 3) * Real.pi * a^2 * b^2 / c)
    (hV2 : V2 = (1 / 3) * Real.pi * b^2 * a)
    (hV3 : V3 = (1 / 3) * Real.pi * a^2 * b) : 
    (1 / V1^2) = (1 / V2^2) + (1 / V3^2) :=
sorry

end rotation_volumes_l280_28083


namespace perimeter_trapezoid_l280_28061

theorem perimeter_trapezoid 
(E F G H : Point)
(EF GH : ℝ)
(HJ EI FG EH : ℝ)
(h_eq1 : EF = GH)
(h_FG : FG = 10)
(h_EH : EH = 20)
(h_EI : EI = 5)
(h_HJ : HJ = 5)
(h_EF_HG : EF = Real.sqrt (EI^2 + ((EH - FG) / 2)^2)) :
  2 * EF + FG + EH = 30 + 10 * Real.sqrt 2 :=
by
  sorry

end perimeter_trapezoid_l280_28061


namespace equation_roots_l280_28099

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l280_28099


namespace find_value_of_expression_l280_28052

variable (α : ℝ)

theorem find_value_of_expression 
  (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / (Real.cos α)^2) = 6 := 
by 
  sorry

end find_value_of_expression_l280_28052


namespace fifi_pink_hangers_l280_28060

theorem fifi_pink_hangers :
  ∀ (g b y p : ℕ), 
  g = 4 →
  b = g - 1 →
  y = b - 1 →
  16 = g + b + y + p →
  p = 7 :=
by
  intros
  sorry

end fifi_pink_hangers_l280_28060


namespace cash_realized_without_brokerage_l280_28043

theorem cash_realized_without_brokerage
  (C : ℝ)
  (h1 : (1 / 4) * (1 / 100) = 1 / 400)
  (h2 : C + (C / 400) = 108) :
  C = 43200 / 401 :=
by
  sorry

end cash_realized_without_brokerage_l280_28043


namespace average_last_four_numbers_l280_28065

theorem average_last_four_numbers (numbers : List ℝ) 
  (h1 : numbers.length = 7)
  (h2 : (numbers.sum / 7) = 62)
  (h3 : (numbers.take 3).sum / 3 = 58) : 
  ((numbers.drop 3).sum / 4) = 65 :=
by
  sorry

end average_last_four_numbers_l280_28065


namespace fifth_term_sequence_l280_28035

theorem fifth_term_sequence : 2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 := 
by 
  sorry

end fifth_term_sequence_l280_28035


namespace ordered_pairs_satisfy_equation_l280_28094

theorem ordered_pairs_satisfy_equation :
  (∃ (a : ℝ) (b : ℤ), a > 0 ∧ 3 ≤ b ∧ b ≤ 203 ∧ (Real.log a / Real.log b) ^ 2021 = Real.log (a ^ 2021) / Real.log b) :=
sorry

end ordered_pairs_satisfy_equation_l280_28094


namespace fewer_twos_for_100_l280_28058

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l280_28058


namespace min_length_intersection_l280_28008

def set_with_length (a b : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ b}
def length_of_set (a b : ℝ) := b - a
def M (m : ℝ) := set_with_length m (m + 3/4)
def N (n : ℝ) := set_with_length (n - 1/3) n

theorem min_length_intersection (m n : ℝ) (h₁ : 0 ≤ m) (h₂ : m + 3/4 ≤ 1) (h₃ : 0 ≤ n - 1/3) (h₄ : n ≤ 1) : 
  length_of_set (max m (n - 1/3)) (min (m + 3/4) n) = 1/12 :=
by
  sorry

end min_length_intersection_l280_28008


namespace minimum_rectangle_length_l280_28022

theorem minimum_rectangle_length (a x y : ℝ) (h : x * y = a^2) : x ≥ a ∨ y ≥ a :=
sorry

end minimum_rectangle_length_l280_28022


namespace fraction_equality_l280_28048

variables {R : Type*} [Field R] {m n p q : R}

theorem fraction_equality 
  (h1 : m / n = 15)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 10) :
  m / q = 1 / 2 :=
sorry

end fraction_equality_l280_28048


namespace find_last_even_number_l280_28011

theorem find_last_even_number (n : ℕ) (h : 4 * (n * (n + 1) * (2 * n + 1) / 6) = 560) : 2 * n = 14 :=
by
  sorry

end find_last_even_number_l280_28011


namespace range_of_p_l280_28072

theorem range_of_p (a b : ℝ) :
  (∀ x y p q : ℝ, p + q = 1 → (p * (x^2 + a * x + b) + q * (y^2 + a * y + b) ≥ ((p * x + q * y)^2 + a * (p * x + q * y) + b))) →
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 1) :=
sorry

end range_of_p_l280_28072


namespace volume_box_l280_28089

theorem volume_box (x y : ℝ) :
  (16 - 2 * x) * (12 - 2 * y) * y = 4 * x * y ^ 2 - 24 * x * y + 192 * y - 32 * y ^ 2 :=
by sorry

end volume_box_l280_28089


namespace students_solved_only_B_l280_28001

variable (A B C : Prop)
variable (n x y b c d : ℕ)

-- Conditions given in the problem
axiom h1 : n = 25
axiom h2 : x + y + b + c + d = n
axiom h3 : b + d = 2 * (c + d)
axiom h4 : x = y + 1
axiom h5 : x + b + c = 2 * (b + c)

-- Theorem to be proved
theorem students_solved_only_B : b = 6 :=
by
  sorry

end students_solved_only_B_l280_28001


namespace eliminate_duplicates_3n_2m1_l280_28096

theorem eliminate_duplicates_3n_2m1 :
  ∀ k: ℤ, ∃ n m: ℤ, 3 * n ≠ 2 * m + 1 ↔ 2 * m + 1 = 12 * k + 1 ∨ 2 * m + 1 = 12 * k + 5 :=
by
  sorry

end eliminate_duplicates_3n_2m1_l280_28096


namespace total_operations_l280_28015

-- Define the process of iterative multiplication and division as described in the problem
def process (start : Nat) : Nat :=
  let m1 := 3 * start
  let m2 := 3 * m1
  let m3 := 3 * m2
  let m4 := 3 * m3
  let m5 := 3 * m4
  let d1 := m5 / 2
  let d2 := d1 / 2
  let d3 := d2 / 2
  let d4 := d3 / 2
  let d5 := d4 / 2
  let d6 := d5 / 2
  let d7 := d6 / 2
  d7

theorem total_operations : process 1 = 1 ∧ 5 + 7 = 12 :=
by
  sorry

end total_operations_l280_28015


namespace deductive_reasoning_example_l280_28055

-- Definitions for the conditions
def Metal (x : Type) : Prop := sorry
def ConductsElectricity (x : Type) : Prop := sorry
def Iron : Type := sorry

-- The problem statement
theorem deductive_reasoning_example (H1 : ∀ x, Metal x → ConductsElectricity x) (H2 : Metal Iron) : ConductsElectricity Iron :=
by sorry

end deductive_reasoning_example_l280_28055


namespace difference_eq_neg_subtrahend_implies_minuend_zero_l280_28037

theorem difference_eq_neg_subtrahend_implies_minuend_zero {x y : ℝ} (h : x - y = -y) : x = 0 :=
sorry

end difference_eq_neg_subtrahend_implies_minuend_zero_l280_28037


namespace right_triangle_area_l280_28025

theorem right_triangle_area {a r R : ℝ} (hR : R = (5 / 2) * r) (h_leg : ∃ BC, BC = a) :
  (∃ area, area = (2 * a^2 / 3) ∨ area = (3 * a^2 / 8)) :=
sorry

end right_triangle_area_l280_28025


namespace problem_l280_28064

-- Definitions for conditions
def countMultiplesOf (n upperLimit : ℕ) : ℕ :=
  (upperLimit - 1) / n

def a : ℕ := countMultiplesOf 4 40
def b : ℕ := countMultiplesOf 4 40

-- Statement to prove
theorem problem : (a + b)^2 = 324 := by
  sorry

end problem_l280_28064


namespace booth_visibility_correct_l280_28090

noncomputable def booth_visibility (L : ℝ) : ℝ × ℝ :=
  let ρ_min := L
  let ρ_max := (1 + Real.sqrt 2) / 2 * L
  (ρ_min, ρ_max)

theorem booth_visibility_correct (L : ℝ) (hL : L > 0) :
  booth_visibility L = (L, (1 + Real.sqrt 2) / 2 * L) :=
by
  sorry

end booth_visibility_correct_l280_28090


namespace ceil_e_add_pi_l280_28057

theorem ceil_e_add_pi : ⌈Real.exp 1 + Real.pi⌉ = 6 := by
  sorry

end ceil_e_add_pi_l280_28057
