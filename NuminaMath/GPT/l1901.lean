import Mathlib

namespace NUMINAMATH_GPT_fraction_of_journey_by_rail_l1901_190116

theorem fraction_of_journey_by_rail :
  ∀ (x : ℝ), x * 130 + (17 / 20) * 130 + 6.5 = 130 → x = 1 / 10 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_fraction_of_journey_by_rail_l1901_190116


namespace NUMINAMATH_GPT_percentage_of_360_equals_115_2_l1901_190142

theorem percentage_of_360_equals_115_2 (p : ℝ) (h : (p / 100) * 360 = 115.2) : p = 32 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_360_equals_115_2_l1901_190142


namespace NUMINAMATH_GPT_root_difference_l1901_190114

theorem root_difference (p : ℝ) (r s : ℝ) 
  (h₁ : r + s = 2 * p) 
  (h₂ : r * s = (p^2 - 4) / 3) : 
  r - s = 2 * (Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_root_difference_l1901_190114


namespace NUMINAMATH_GPT_solve_a_l1901_190139

-- Defining sets A and B
def set_A (a : ℤ) : Set ℤ := {a^2, a + 1, -3}
def set_B (a : ℤ) : Set ℤ := {a - 3, 2 * a - 1, a^2 + 1}

-- Defining the condition of intersection
def intersection_condition (a : ℤ) : Prop :=
  (set_A a) ∩ (set_B a) = {-3}

-- Stating the theorem
theorem solve_a (a : ℤ) (h : intersection_condition a) : a = -1 :=
sorry

end NUMINAMATH_GPT_solve_a_l1901_190139


namespace NUMINAMATH_GPT_solve_inequality_l1901_190117

def numerator (x : ℝ) : ℝ := x ^ 2 - 4 * x + 3
def denominator (x : ℝ) : ℝ := (x - 2) ^ 2

theorem solve_inequality : { x : ℝ | numerator x / denominator x < 0 } = { x : ℝ | 1 < x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1901_190117


namespace NUMINAMATH_GPT_number_of_trees_planted_l1901_190104

def current_trees : ℕ := 34
def final_trees : ℕ := 83
def planted_trees : ℕ := final_trees - current_trees

theorem number_of_trees_planted : planted_trees = 49 :=
by
  -- proof goes here, but it is skipped for now
  sorry

end NUMINAMATH_GPT_number_of_trees_planted_l1901_190104


namespace NUMINAMATH_GPT_sum_of_consecutive_page_numbers_l1901_190134

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20250) : n + (n + 1) = 285 := 
sorry

end NUMINAMATH_GPT_sum_of_consecutive_page_numbers_l1901_190134


namespace NUMINAMATH_GPT_negation_if_positive_then_square_positive_l1901_190179

theorem negation_if_positive_then_square_positive :
  (¬ (∀ x : ℝ, x > 0 → x^2 > 0)) ↔ (∀ x : ℝ, x ≤ 0 → x^2 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_if_positive_then_square_positive_l1901_190179


namespace NUMINAMATH_GPT_cos_double_angle_l1901_190108

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1901_190108


namespace NUMINAMATH_GPT_sqrt_fraction_eq_half_l1901_190129

-- Define the problem statement in a Lean 4 theorem:
theorem sqrt_fraction_eq_half : Real.sqrt ((25 / 36 : ℚ) - (4 / 9 : ℚ)) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_eq_half_l1901_190129


namespace NUMINAMATH_GPT_sculpture_and_base_height_l1901_190178

theorem sculpture_and_base_height :
  let sculpture_height_in_feet := 2
  let sculpture_height_in_inches := 10
  let base_height_in_inches := 2
  let total_height_in_inches := (sculpture_height_in_feet * 12) + sculpture_height_in_inches + base_height_in_inches
  let total_height_in_feet := total_height_in_inches / 12
  total_height_in_feet = 3 :=
by
  sorry

end NUMINAMATH_GPT_sculpture_and_base_height_l1901_190178


namespace NUMINAMATH_GPT_product_divisible_by_4_l1901_190181

noncomputable def biased_die_prob_divisible_by_4 : ℚ :=
  let q := 1/4  -- probability of rolling a number divisible by 3
  let p4 := 2 * q -- probability of rolling a number divisible by 4
  let p_neither := (1 - p4) * (1 - p4) -- probability of neither roll being divisible by 4
  1 - p_neither -- probability that at least one roll is divisible by 4

theorem product_divisible_by_4 :
  biased_die_prob_divisible_by_4 = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_product_divisible_by_4_l1901_190181


namespace NUMINAMATH_GPT_find_a₈_l1901_190118

noncomputable def a₃ : ℝ := -11 / 6
noncomputable def a₅ : ℝ := -13 / 7

theorem find_a₈ (h : ∃ d : ℝ, ∀ n : ℕ, (1 / (a₃ + 2)) + (n-2) * d = (1 / (a_n + 2)))
  : a_n = -32 / 17 := sorry

end NUMINAMATH_GPT_find_a₈_l1901_190118


namespace NUMINAMATH_GPT_money_lent_years_l1901_190144

noncomputable def compound_interest_time (A P r n : ℝ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem money_lent_years :
  compound_interest_time 740 671.2018140589569 0.05 1 = 2 := by
  sorry

end NUMINAMATH_GPT_money_lent_years_l1901_190144


namespace NUMINAMATH_GPT_pairs_of_powers_of_two_l1901_190194

theorem pairs_of_powers_of_two (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (h3 : ∃ a : ℕ, m + n = 2^a) (h4 : ∃ b : ℕ, mn + 1 = 2^b) :
  (∃ a : ℕ, m = 2^a - 1 ∧ n = 1) ∨ 
  (∃ a : ℕ, m = 2^(a-1) + 1 ∧ n = 2^(a-1) - 1) :=
sorry

end NUMINAMATH_GPT_pairs_of_powers_of_two_l1901_190194


namespace NUMINAMATH_GPT_inequality_for_positive_reals_l1901_190112

theorem inequality_for_positive_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
    ≥ (Real.sqrt (3 / 2) * Real.sqrt (x + y + z)) := 
sorry

end NUMINAMATH_GPT_inequality_for_positive_reals_l1901_190112


namespace NUMINAMATH_GPT_oates_reunion_attendees_l1901_190102

noncomputable def total_guests : ℕ := 100
noncomputable def hall_attendees : ℕ := 70
noncomputable def both_reunions_attendees : ℕ := 10

theorem oates_reunion_attendees :
  ∃ O : ℕ, total_guests = O + hall_attendees - both_reunions_attendees ∧ O = 40 :=
by
  sorry

end NUMINAMATH_GPT_oates_reunion_attendees_l1901_190102


namespace NUMINAMATH_GPT_polynomial_identity_l1901_190133

theorem polynomial_identity (a b c d e f : ℤ)
  (h_eq : ∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 770 := by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l1901_190133


namespace NUMINAMATH_GPT_odd_function_a_increasing_function_a_l1901_190196

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem odd_function_a (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - (f x a)) → a = -1 :=
by sorry

theorem increasing_function_a (a : ℝ) :
  (∀ x : ℝ, (Real.exp x - a * Real.exp (-x)) ≥ 0) → a ∈ Set.Iic 0 :=
by sorry

end NUMINAMATH_GPT_odd_function_a_increasing_function_a_l1901_190196


namespace NUMINAMATH_GPT_value_of_b_l1901_190182

variable (a b : ℤ)

theorem value_of_b : a = 105 ∧ a ^ 3 = 21 * 49 * 45 * b → b = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_b_l1901_190182


namespace NUMINAMATH_GPT_proof_inequalities_equivalence_max_f_value_l1901_190106

-- Definitions for the conditions
def inequality1 (x: ℝ) := |x - 2| > 1
def inequality2 (x: ℝ) := x^2 - 4 * x + 3 > 0

-- The main statements to prove
theorem proof_inequalities_equivalence : 
  {x : ℝ | inequality1 x} = {x : ℝ | inequality2 x} := 
sorry

noncomputable def f (x: ℝ) := 4 * Real.sqrt (x - 3) + 3 * Real.sqrt (5 - x)

theorem max_f_value : 
  ∃ x : ℝ, (3 ≤ x ∧ x ≤ 5) ∧ (f x = 5 * Real.sqrt 2) ∧ ∀ y : ℝ, ((3 ≤ y ∧ y ≤ 5) → f y ≤ 5 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_proof_inequalities_equivalence_max_f_value_l1901_190106


namespace NUMINAMATH_GPT_lara_bouncy_house_time_l1901_190172

theorem lara_bouncy_house_time :
  let run1_time := (3 * 60 + 45) + (2 * 60 + 10) + (1 * 60 + 28)
  let door_time := 73
  let run2_time := (2 * 60 + 55) + (1 * 60 + 48) + (1 * 60 + 15)
  run1_time + door_time + run2_time = 874 := by
    let run1_time := 225 + 130 + 88
    let door_time := 73
    let run2_time := 175 + 108 + 75
    sorry

end NUMINAMATH_GPT_lara_bouncy_house_time_l1901_190172


namespace NUMINAMATH_GPT_girls_count_l1901_190154

variable (B G : ℕ)

theorem girls_count (h1: B = 387) (h2: G = (B + (54 * B) / 100)) : G = 596 := 
by 
  sorry

end NUMINAMATH_GPT_girls_count_l1901_190154


namespace NUMINAMATH_GPT_James_beat_old_record_by_296_points_l1901_190127

def touchdowns_per_game := 4
def points_per_touchdown := 6
def number_of_games := 15
def two_point_conversions := 6
def points_per_two_point_conversion := 2
def field_goals := 8
def points_per_field_goal := 3
def extra_point_attempts := 20
def points_per_extra_point := 1
def consecutive_touchdowns := 3
def games_with_consecutive_touchdowns := 5
def bonus_multiplier := 2
def old_record := 300

def James_points : ℕ :=
  (touchdowns_per_game * number_of_games * points_per_touchdown) + 
  ((consecutive_touchdowns * games_with_consecutive_touchdowns) * points_per_touchdown * bonus_multiplier) +
  (two_point_conversions * points_per_two_point_conversion) +
  (field_goals * points_per_field_goal) +
  (extra_point_attempts * points_per_extra_point)

def points_above_old_record := James_points - old_record

theorem James_beat_old_record_by_296_points : points_above_old_record = 296 := by
  -- here would be the proof
  sorry

end NUMINAMATH_GPT_James_beat_old_record_by_296_points_l1901_190127


namespace NUMINAMATH_GPT_solve_for_m_l1901_190155

theorem solve_for_m (m : ℝ) :
  (1 * m + (3 + m) * 2 = 0) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_m_l1901_190155


namespace NUMINAMATH_GPT_g_800_eq_768_l1901_190137

noncomputable def g : ℕ → ℕ := sorry

axiom g_condition1 (n : ℕ) : g (g n) = 2 * n
axiom g_condition2 (n : ℕ) : g (4 * n + 3) = 4 * n + 1

theorem g_800_eq_768 : g 800 = 768 := by
  sorry

end NUMINAMATH_GPT_g_800_eq_768_l1901_190137


namespace NUMINAMATH_GPT_inequality_holds_for_positive_y_l1901_190140

theorem inequality_holds_for_positive_y (y : ℝ) (hy : y > 0) : y^2 ≥ 2 * y - 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_positive_y_l1901_190140


namespace NUMINAMATH_GPT_intersecting_lines_sum_l1901_190151

theorem intersecting_lines_sum (a b : ℝ) (h1 : 2 = (1/3) * 4 + a) (h2 : 4 = (1/3) * 2 + b) : a + b = 4 :=
sorry

end NUMINAMATH_GPT_intersecting_lines_sum_l1901_190151


namespace NUMINAMATH_GPT_Bridget_weight_is_correct_l1901_190111

-- Definitions based on conditions
def Martha_weight : ℕ := 2
def weight_difference : ℕ := 37

-- Bridget's weight based on the conditions
def Bridget_weight : ℕ := Martha_weight + weight_difference

-- Proof problem: Prove that Bridget's weight is 39
theorem Bridget_weight_is_correct : Bridget_weight = 39 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Bridget_weight_is_correct_l1901_190111


namespace NUMINAMATH_GPT_probability_at_least_one_passes_l1901_190125

theorem probability_at_least_one_passes (prob_pass : ℚ) (prob_fail : ℚ) (p_all_fail: ℚ):
  (prob_pass = 1/3) →
  (prob_fail = 1 - prob_pass) →
  (p_all_fail = prob_fail ^ 3) →
  (1 - p_all_fail = 19/27) :=
by
  intros hpp hpf hpaf
  sorry

end NUMINAMATH_GPT_probability_at_least_one_passes_l1901_190125


namespace NUMINAMATH_GPT_particle_probability_l1901_190136

theorem particle_probability 
  (P : ℕ → ℝ) (n : ℕ)
  (h1 : P 0 = 1)
  (h2 : P 1 = 2 / 3)
  (h3 : ∀ n ≥ 3, P n = 2 / 3 * P (n-1) + 1 / 3 * P (n-2)) :
  P n = 2 / 3 + 1 / 12 * (1 - (-1 / 3)^(n-1)) := 
sorry

end NUMINAMATH_GPT_particle_probability_l1901_190136


namespace NUMINAMATH_GPT_number_of_trees_l1901_190184

theorem number_of_trees (initial_trees planted_trees : ℕ)
  (h1 : initial_trees = 13)
  (h2 : planted_trees = 12) :
  initial_trees + planted_trees = 25 := by
  sorry

end NUMINAMATH_GPT_number_of_trees_l1901_190184


namespace NUMINAMATH_GPT_price_of_first_metal_l1901_190130

theorem price_of_first_metal (x : ℝ) 
  (h1 : (x + 96) / 2 = 82) : 
  x = 68 :=
by sorry

end NUMINAMATH_GPT_price_of_first_metal_l1901_190130


namespace NUMINAMATH_GPT_find_a_from_inequality_solution_set_l1901_190109

theorem find_a_from_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (x^2 - a*x + 4 < 0) ↔ (1 < x ∧ x < 4)) -> a = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_from_inequality_solution_set_l1901_190109


namespace NUMINAMATH_GPT_people_with_fewer_than_7_cards_l1901_190180

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end NUMINAMATH_GPT_people_with_fewer_than_7_cards_l1901_190180


namespace NUMINAMATH_GPT_cubic_polynomials_common_roots_c_d_l1901_190113

theorem cubic_polynomials_common_roots_c_d (c d : ℝ) :
  (∀ (r s : ℝ), r ≠ s ∧
     (r^3 + c*r^2 + 12*r + 7 = 0) ∧ (s^3 + c*s^2 + 12*s + 7 = 0) ∧
     (r^3 + d*r^2 + 15*r + 9 = 0) ∧ (s^3 + d*s^2 + 15*s + 9 = 0)) →
  (c = -5 ∧ d = -6) := 
by
  sorry

end NUMINAMATH_GPT_cubic_polynomials_common_roots_c_d_l1901_190113


namespace NUMINAMATH_GPT_total_biscuits_l1901_190162

-- Define the number of dogs and biscuits per dog
def num_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem stating the total number of biscuits needed
theorem total_biscuits : num_dogs * biscuits_per_dog = 6 := by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_biscuits_l1901_190162


namespace NUMINAMATH_GPT_inequality_proof_l1901_190141

variables (x y z : ℝ)

theorem inequality_proof (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  2 ≤ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ∧ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ≤ (1 + x) * (1 + y) * (1 + z) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1901_190141


namespace NUMINAMATH_GPT_winning_strategy_l1901_190138

noncomputable def winning_player (n : ℕ) (h : n ≥ 2) : String :=
if n = 2 ∨ n = 4 ∨ n = 8 then "Ariane" else "Bérénice"

theorem winning_strategy (n : ℕ) (h : n ≥ 2) :
  (winning_player n h = "Ariane" ↔ (n = 2 ∨ n = 4 ∨ n = 8)) ∧
  (winning_player n h = "Bérénice" ↔ ¬ (n = 2 ∨ n = 4 ∨ n = 8)) :=
sorry

end NUMINAMATH_GPT_winning_strategy_l1901_190138


namespace NUMINAMATH_GPT_p_is_necessary_but_not_sufficient_for_q_l1901_190100

def p (x : ℝ) : Prop := |2 * x - 3| < 1
def q (x : ℝ) : Prop := x * (x - 3) < 0

theorem p_is_necessary_but_not_sufficient_for_q :
  (∀ x : ℝ, q x → p x) ∧ ¬(∀ x : ℝ, p x → q x) :=
by sorry

end NUMINAMATH_GPT_p_is_necessary_but_not_sufficient_for_q_l1901_190100


namespace NUMINAMATH_GPT_penny_makes_from_cheesecakes_l1901_190167

-- Definitions based on the conditions
def slices_per_pie : ℕ := 6
def cost_per_slice : ℕ := 7
def pies_sold : ℕ := 7

-- The mathematical equivalent proof problem
theorem penny_makes_from_cheesecakes : slices_per_pie * cost_per_slice * pies_sold = 294 := by
  sorry

end NUMINAMATH_GPT_penny_makes_from_cheesecakes_l1901_190167


namespace NUMINAMATH_GPT_x2_plus_y2_lt_1_l1901_190153

theorem x2_plus_y2_lt_1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : x^2 + y^2 < 1 :=
sorry

end NUMINAMATH_GPT_x2_plus_y2_lt_1_l1901_190153


namespace NUMINAMATH_GPT_parabola_sum_l1901_190132

-- Define the quadratic equation
noncomputable def quadratic_eq (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Given conditions
variables (a b c : ℝ)
variables (h1 : (∀ x y : ℝ, y = quadratic_eq a b c x → y = a * (x - 6)^2 - 2))
variables (h2 : quadratic_eq a b c 3 = 0)

-- Prove the sum a + b + c
theorem parabola_sum :
  a + b + c = 14 / 9 :=
sorry

end NUMINAMATH_GPT_parabola_sum_l1901_190132


namespace NUMINAMATH_GPT_new_cost_percentage_l1901_190188

def cost (t b : ℝ) := t * b^5

theorem new_cost_percentage (t b : ℝ) : 
  let C := cost t b
  let W := cost (3 * t) (2 * b)
  W = 96 * C :=
by
  sorry

end NUMINAMATH_GPT_new_cost_percentage_l1901_190188


namespace NUMINAMATH_GPT_problem_proof_l1901_190149

open Real

noncomputable def angle_B (A C : ℝ) : ℝ := π / 3

noncomputable def area_triangle (a b c : ℝ) : ℝ := 
  (1/2) * a * c * (sqrt 3 / 2)

theorem problem_proof (A B C a b c : ℝ)
  (h1 : 2 * cos A * cos C * (tan A * tan C - 1) = 1)
  (h2 : a + c = sqrt 15)
  (h3 : b = sqrt 3)
  (h4 : B = π / 3) :
  (B = angle_B A C) ∧ 
  (area_triangle a b c = sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1901_190149


namespace NUMINAMATH_GPT_football_count_white_patches_count_l1901_190101

theorem football_count (x : ℕ) (footballs : ℕ) (students : ℕ) (h1 : students - 9 = footballs + 9) (h2 : students = 2 * footballs + 9) : footballs = 27 :=
sorry

theorem white_patches_count (white_patches : ℕ) (h : 2 * 12 * 5 = 6 * white_patches) : white_patches = 20 :=
sorry

end NUMINAMATH_GPT_football_count_white_patches_count_l1901_190101


namespace NUMINAMATH_GPT_converse_angles_complements_l1901_190190

theorem converse_angles_complements (α β : ℝ) (h : ∀γ : ℝ, α + γ = 90 ∧ β + γ = 90 → α = β) : 
  ∀ δ, α + δ = 90 ∧ β + δ = 90 → α = β :=
by 
  sorry

end NUMINAMATH_GPT_converse_angles_complements_l1901_190190


namespace NUMINAMATH_GPT_value_of_a_l1901_190146

theorem value_of_a (a : ℝ) : (|a| - 1 = 1) ∧ (a - 2 ≠ 0) → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1901_190146


namespace NUMINAMATH_GPT_marked_percentage_above_cost_l1901_190177

theorem marked_percentage_above_cost (CP SP : ℝ) (discount_percentage MP : ℝ) 
  (h1 : CP = 540) 
  (h2 : SP = 457) 
  (h3 : discount_percentage = 26.40901771336554) 
  (h4 : SP = MP * (1 - discount_percentage / 100)) : 
  ((MP - CP) / CP) * 100 = 15 :=
by
  sorry

end NUMINAMATH_GPT_marked_percentage_above_cost_l1901_190177


namespace NUMINAMATH_GPT_eggs_in_fridge_l1901_190199

theorem eggs_in_fridge (total_eggs : ℕ) (eggs_per_cake : ℕ) (num_cakes : ℕ) (eggs_used : ℕ) (eggs_in_fridge : ℕ)
  (h1 : total_eggs = 60)
  (h2 : eggs_per_cake = 5)
  (h3 : num_cakes = 10)
  (h4 : eggs_used = eggs_per_cake * num_cakes)
  (h5 : eggs_in_fridge = total_eggs - eggs_used) :
  eggs_in_fridge = 10 :=
by
  sorry

end NUMINAMATH_GPT_eggs_in_fridge_l1901_190199


namespace NUMINAMATH_GPT_geometric_series_sum_l1901_190145

theorem geometric_series_sum :
  ∑' i : ℕ, (2 / 3) ^ (i + 1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1901_190145


namespace NUMINAMATH_GPT_minimum_f_value_minimum_fraction_value_l1901_190175

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem minimum_f_value : ∃ x : ℝ, f x = 2 :=
by
  -- proof skipped, please insert proof here
  sorry

theorem minimum_fraction_value (a b : ℝ) (h : a^2 + b^2 = 2) : 
  (1 / (a^2 + 1)) + (4 / (b^2 + 1)) = 9 / 4 :=
by
  -- proof skipped, please insert proof here
  sorry

end NUMINAMATH_GPT_minimum_f_value_minimum_fraction_value_l1901_190175


namespace NUMINAMATH_GPT_classics_section_books_l1901_190169

-- Define the number of authors
def num_authors : Nat := 6

-- Define the number of books per author
def books_per_author : Nat := 33

-- Define the total number of books
def total_books : Nat := num_authors * books_per_author

-- Prove that the total number of books is 198
theorem classics_section_books : total_books = 198 := by
  sorry

end NUMINAMATH_GPT_classics_section_books_l1901_190169


namespace NUMINAMATH_GPT_Penelope_daily_savings_l1901_190166

theorem Penelope_daily_savings
  (total_savings : ℝ)
  (days_in_year : ℕ)
  (h1 : total_savings = 8760)
  (h2 : days_in_year = 365) :
  total_savings / days_in_year = 24 :=
by
  sorry

end NUMINAMATH_GPT_Penelope_daily_savings_l1901_190166


namespace NUMINAMATH_GPT_complex_number_identity_l1901_190198

theorem complex_number_identity (z : ℂ) (h : z = 1 + (1 : ℂ) * I) : z^2 + z = 1 + 3 * I := 
sorry

end NUMINAMATH_GPT_complex_number_identity_l1901_190198


namespace NUMINAMATH_GPT_smallest_value_l1901_190124

noncomputable def smallest_possible_sum (a b : ℝ) : ℝ :=
  a + b

theorem smallest_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) :
  smallest_possible_sum a b = 6.5 :=
sorry

end NUMINAMATH_GPT_smallest_value_l1901_190124


namespace NUMINAMATH_GPT_collinear_vectors_l1901_190158

-- Definitions
def a : ℝ × ℝ := (2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, 6)

-- Proof statement
theorem collinear_vectors (x : ℝ) (h : ∃ k : ℝ, b x = k • a) : x = 3 :=
by sorry

end NUMINAMATH_GPT_collinear_vectors_l1901_190158


namespace NUMINAMATH_GPT_coefficients_sum_eq_zero_l1901_190147

theorem coefficients_sum_eq_zero 
  (a b c : ℝ)
  (f g h : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : ∀ x, g x = b * x^2 + c * x + a)
  (h3 : ∀ x, h x = c * x^2 + a * x + b)
  (h4 : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) :
  a + b + c = 0 := 
sorry

end NUMINAMATH_GPT_coefficients_sum_eq_zero_l1901_190147


namespace NUMINAMATH_GPT_mileage_on_city_streets_l1901_190163

-- Defining the given conditions
def distance_on_highways : ℝ := 210
def mileage_on_highways : ℝ := 35
def total_gas_used : ℝ := 9
def distance_on_city_streets : ℝ := 54

-- Proving the mileage on city streets
theorem mileage_on_city_streets :
  ∃ x : ℝ, 
    (distance_on_highways / mileage_on_highways + distance_on_city_streets / x = total_gas_used)
    ∧ x = 18 :=
by
  sorry

end NUMINAMATH_GPT_mileage_on_city_streets_l1901_190163


namespace NUMINAMATH_GPT_minoxidil_percentage_l1901_190121

-- Define the conditions
variable (x : ℝ) -- percentage of Minoxidil in the solution to add
def pharmacist_scenario (x : ℝ) : Prop :=
  let amt_2_percent_solution := 70 -- 70 ml of 2% solution
  let percent_in_2_percent := 0.02
  let amt_of_2_percent := percent_in_2_percent * amt_2_percent_solution
  let amt_added_solution := 35 -- 35 ml of solution to add
  let total_volume := amt_2_percent_solution + amt_added_solution -- 105 ml in total
  let desired_percent := 0.03
  let desired_amt := desired_percent * total_volume
  amt_of_2_percent + (x / 100) * amt_added_solution = desired_amt

-- Define the proof problem statement
theorem minoxidil_percentage : pharmacist_scenario 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_minoxidil_percentage_l1901_190121


namespace NUMINAMATH_GPT_right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l1901_190197

theorem right_triangle_arithmetic_progression_is_345 (a b c : ℕ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ d, b = a + d ∧ c = a + 2 * d)
  : (a, b, c) = (3, 4, 5) :=
by
  sorry

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

noncomputable def sqrt_golden_ratio_div_2 := Real.sqrt ((1 + Real.sqrt 5) / 2)

theorem right_triangle_geometric_progression 
  (a b c : ℝ)
  (h1 : a * a + b * b = c * c)
  (h2 : ∃ r, b = a * r ∧ c = a * r * r)
  : (a, b, c) = (1, sqrt_golden_ratio_div_2, golden_ratio) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_arithmetic_progression_is_345_right_triangle_geometric_progression_l1901_190197


namespace NUMINAMATH_GPT_ant_paths_l1901_190152

theorem ant_paths (n m : ℕ) : 
  ∃ paths : ℕ, paths = Nat.choose (n + m) m := sorry

end NUMINAMATH_GPT_ant_paths_l1901_190152


namespace NUMINAMATH_GPT_ball_arrangement_l1901_190185

theorem ball_arrangement :
  (Nat.factorial 9) / ((Nat.factorial 2) * (Nat.factorial 3) * (Nat.factorial 4)) = 1260 := 
by
  sorry

end NUMINAMATH_GPT_ball_arrangement_l1901_190185


namespace NUMINAMATH_GPT_min_value_f_in_interval_l1901_190171

def f (x : ℝ) : ℝ := x^4 + 2 * x^2 - 1

theorem min_value_f_in_interval : 
  ∃ x ∈ (Set.Icc (-1 : ℝ) 1), f x = -1 :=
by
  sorry


end NUMINAMATH_GPT_min_value_f_in_interval_l1901_190171


namespace NUMINAMATH_GPT_second_valve_rate_difference_l1901_190191

theorem second_valve_rate_difference (V1 V2 : ℝ) 
  (h1 : V1 = 12000 / 120)
  (h2 : V1 + V2 = 12000 / 48) :
  V2 - V1 = 50 :=
by
  -- Since h1: V1 = 100
  -- And V1 + V2 = 250 from h2
  -- Therefore V2 = 250 - 100 = 150
  -- And V2 - V1 = 150 - 100 = 50
  sorry

end NUMINAMATH_GPT_second_valve_rate_difference_l1901_190191


namespace NUMINAMATH_GPT_inclination_angle_of_line_l1901_190189

open Real

theorem inclination_angle_of_line (x y : ℝ) (h : x + y - 3 = 0) : 
  ∃ θ : ℝ, θ = 3 * π / 4 :=
by
  sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l1901_190189


namespace NUMINAMATH_GPT_percentage_of_y_l1901_190168

theorem percentage_of_y (x y : ℝ) (h1 : x = 4 * y) (h2 : 0.80 * x = (P / 100) * y) : P = 320 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_of_y_l1901_190168


namespace NUMINAMATH_GPT_suresh_wifes_speed_l1901_190110

-- Define conditions
def circumference_of_track : ℝ := 0.726 -- track circumference in kilometers
def suresh_speed : ℝ := 4.5 -- Suresh's speed in km/hr
def meeting_time_in_hours : ℝ := 0.088 -- time till they meet in hours

-- Define the question and expected answer
theorem suresh_wifes_speed : ∃ (V : ℝ), V = 3.75 :=
  by
    -- Let Distance_covered_by_both = circumference_of_track
    let Distance_covered_by_suresh : ℝ := suresh_speed * meeting_time_in_hours
    let Distance_covered_by_suresh_wife : ℝ := circumference_of_track - Distance_covered_by_suresh
    let suresh_wifes_speed : ℝ := Distance_covered_by_suresh_wife / meeting_time_in_hours
    -- Expected answer
    existsi suresh_wifes_speed
    sorry

end NUMINAMATH_GPT_suresh_wifes_speed_l1901_190110


namespace NUMINAMATH_GPT_abs_x_ge_abs_4ax_l1901_190160

theorem abs_x_ge_abs_4ax (a : ℝ) (h : ∀ x : ℝ, abs x ≥ 4 * a * x) : abs a ≤ 1 / 4 :=
sorry

end NUMINAMATH_GPT_abs_x_ge_abs_4ax_l1901_190160


namespace NUMINAMATH_GPT_seokgi_initial_money_l1901_190176

theorem seokgi_initial_money (X : ℝ) (h1 : X / 2 - X / 4 = 1250) : X = 5000 := by
  sorry

end NUMINAMATH_GPT_seokgi_initial_money_l1901_190176


namespace NUMINAMATH_GPT_fraction_identity_l1901_190173

theorem fraction_identity (m n r t : ℚ) 
  (h₁ : m / n = 3 / 5) 
  (h₂ : r / t = 8 / 9) :
  (3 * m^2 * r - n * t^2) / (5 * n * t^2 - 9 * m^2 * r) = -1 := 
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1901_190173


namespace NUMINAMATH_GPT_set_intersection_eq_l1901_190187

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5}
def ComplementU (S : Set ℕ) : Set ℕ := U \ S

theorem set_intersection_eq : 
  A ∩ (ComplementU B) = {1, 3} := 
by
  sorry

end NUMINAMATH_GPT_set_intersection_eq_l1901_190187


namespace NUMINAMATH_GPT_find_formula_and_range_l1901_190123

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := 4^x + a * 2^x + b

theorem find_formula_and_range
  (a b : ℝ)
  (h₀ : f 0 a b = 1)
  (h₁ : f (-1) a b = -5 / 4) :
  f x (-3) 3 = 4^x - 3 * 2^x + 3 ∧ 
  (∀ x, 0 ≤ x ∧ x ≤ 2 → 1 ≤ f x (-3) 3 ∧ f x (-3) 3 ≤ 25) :=
by
  sorry

end NUMINAMATH_GPT_find_formula_and_range_l1901_190123


namespace NUMINAMATH_GPT_figure_total_area_l1901_190122

theorem figure_total_area :
  let height_left_rect := 6
  let width_base_left_rect := 5
  let height_top_left_rect := 3
  let width_top_left_rect := 5
  let height_top_center_rect := 3
  let width_sum_center_rect := 10
  let height_top_right_rect := 8
  let width_top_right_rect := 2
  let area_total := (height_left_rect * width_base_left_rect) + (height_top_left_rect * width_top_left_rect) + (height_top_center_rect * width_sum_center_rect) + (height_top_right_rect * width_top_right_rect)
  area_total = 91
:= sorry

end NUMINAMATH_GPT_figure_total_area_l1901_190122


namespace NUMINAMATH_GPT_correct_inequality_l1901_190170

variable (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b)

theorem correct_inequality : (1 / (a * b^2)) < (1 / (a^2 * b)) :=
by
  sorry

end NUMINAMATH_GPT_correct_inequality_l1901_190170


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1901_190126

noncomputable def A : Set ℕ := {x | 2 ≤ x ∧ x ≤ 4}
def B : Set ℕ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1901_190126


namespace NUMINAMATH_GPT_uphill_distance_is_100_l1901_190120

def speed_uphill := 30  -- km/hr
def speed_downhill := 60  -- km/hr
def distance_downhill := 50  -- km
def avg_speed := 36  -- km/hr

-- Let d be the distance traveled uphill
variable (d : ℕ)

-- total distance is d + 50 km
def total_distance := d + distance_downhill

-- total time is (time uphill) + (time downhill)
def total_time := (d / speed_uphill) + (distance_downhill / speed_downhill)

theorem uphill_distance_is_100 (d : ℕ) (h : avg_speed = total_distance / total_time) : d = 100 :=
by
  sorry  -- proof is omitted

end NUMINAMATH_GPT_uphill_distance_is_100_l1901_190120


namespace NUMINAMATH_GPT_simplify_fraction_l1901_190159

theorem simplify_fraction :
  (3^100 + 3^98) / (3^100 - 3^98) = 5 / 4 := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1901_190159


namespace NUMINAMATH_GPT_proof_of_truth_values_l1901_190165

open Classical

variables (x : ℝ)

-- Original proposition: If x = 1, then x^2 = 1.
def original_proposition : Prop := (x = 1) → (x^2 = 1)

-- Converse of the original proposition: If x^2 = 1, then x = 1.
def converse_proposition : Prop := (x^2 = 1) → (x = 1)

-- Inverse of the original proposition: If x ≠ 1, then x^2 ≠ 1.
def inverse_proposition : Prop := (x ≠ 1) → (x^2 ≠ 1)

-- Contrapositive of the original proposition: If x^2 ≠ 1, then x ≠ 1.
def contrapositive_proposition : Prop := (x^2 ≠ 1) → (x ≠ 1)

-- Negation of the original proposition: If x = 1, then x^2 ≠ 1.
def negation_proposition : Prop := (x = 1) → (x^2 ≠ 1)

theorem proof_of_truth_values :
  (original_proposition x) ∧
  (converse_proposition x = False) ∧
  (inverse_proposition x = False) ∧
  (contrapositive_proposition x) ∧
  (negation_proposition x = False) := by
  sorry

end NUMINAMATH_GPT_proof_of_truth_values_l1901_190165


namespace NUMINAMATH_GPT_dogs_count_l1901_190161

namespace PetStore

-- Definitions derived from the conditions
def ratio_cats_dogs := 3 / 4
def num_cats := 18
def num_groups := num_cats / 3
def num_dogs := 4 * num_groups

-- The statement to prove
theorem dogs_count : num_dogs = 24 :=
by
  sorry

end PetStore

end NUMINAMATH_GPT_dogs_count_l1901_190161


namespace NUMINAMATH_GPT_cows_relationship_l1901_190119

theorem cows_relationship (H : ℕ) (W : ℕ) (T : ℕ) (hcows : W = 17) (tcows : T = 70) (together : H + W = T) : H = 53 :=
by
  rw [hcows, tcows] at together
  linarith
  -- sorry

end NUMINAMATH_GPT_cows_relationship_l1901_190119


namespace NUMINAMATH_GPT_abs_x_plus_one_ge_one_l1901_190183

theorem abs_x_plus_one_ge_one {x : ℝ} : |x + 1| ≥ 1 ↔ x ≤ -2 ∨ x ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_abs_x_plus_one_ge_one_l1901_190183


namespace NUMINAMATH_GPT_gcd_40_56_l1901_190131

theorem gcd_40_56 : Nat.gcd 40 56 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_40_56_l1901_190131


namespace NUMINAMATH_GPT_consecutive_primes_sum_square_is_prime_l1901_190186

-- Defining what it means for three numbers to be consecutive primes
def consecutive_primes (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  ((p < q ∧ q < r) ∨ (p < q ∧ q < r ∧ r < p) ∨ 
   (r < p ∧ p < q) ∨ (q < p ∧ p < r) ∨ 
   (q < r ∧ r < p) ∨ (r < q ∧ q < p))

-- Defining our main problem statement
theorem consecutive_primes_sum_square_is_prime :
  ∀ p q r : ℕ, consecutive_primes p q r → Nat.Prime (p^2 + q^2 + r^2) ↔ (p = 3 ∧ q = 5 ∧ r = 7) :=
by
  -- Sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_consecutive_primes_sum_square_is_prime_l1901_190186


namespace NUMINAMATH_GPT_find_x2_plus_y2_l1901_190115

noncomputable def xy : ℝ := 12
noncomputable def eq2 (x y : ℝ) : Prop := x^2 * y + x * y^2 + x + y = 120

theorem find_x2_plus_y2 (x y : ℝ) (h1 : xy = 12) (h2 : eq2 x y) : 
  x^2 + y^2 = 10344 / 169 :=
sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l1901_190115


namespace NUMINAMATH_GPT_T_shape_perimeter_l1901_190157

/-- Two rectangles each measuring 3 inch × 5 inch are placed to form the letter T.
The overlapping area between the two rectangles is 1.5 inch. -/
theorem T_shape_perimeter:
  let l := 5 -- inches
  let w := 3 -- inches
  let overlap := 1.5 -- inches
  -- perimeter of one rectangle
  let P := 2 * (l + w)
  -- total perimeter accounting for overlap
  let total_perimeter := 2 * P - 2 * overlap
  total_perimeter = 29 :=
by
  sorry

end NUMINAMATH_GPT_T_shape_perimeter_l1901_190157


namespace NUMINAMATH_GPT_number_of_students_l1901_190148

theorem number_of_students (y c r n : ℕ) (h1 : y = 730) (h2 : c = 17) (h3 : r = 16) :
  y - r = n * c ↔ n = 42 :=
by
  have h4 : 730 - 16 = 714 := by norm_num
  have h5 : 714 / 17 = 42 := by norm_num
  sorry

end NUMINAMATH_GPT_number_of_students_l1901_190148


namespace NUMINAMATH_GPT_increasing_sequences_count_with_modulo_l1901_190107

theorem increasing_sequences_count_with_modulo : 
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sequences_count % mod_value = k :=
by
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sorry

end NUMINAMATH_GPT_increasing_sequences_count_with_modulo_l1901_190107


namespace NUMINAMATH_GPT_range_of_a_l1901_190143

theorem range_of_a (a : ℝ) :
  let A := {x | x^2 + 4 * x = 0}
  let B := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}
  A ∩ B = B → (a = 1 ∨ a ≤ -1) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1901_190143


namespace NUMINAMATH_GPT_total_income_percentage_l1901_190103

-- Define the base income of Juan
def juan_base_income (J : ℝ) := J

-- Define Tim's base income
def tim_base_income (J : ℝ) := 0.70 * J

-- Define Mary's total income
def mary_total_income (J : ℝ) := 1.232 * J

-- Define Lisa's total income
def lisa_total_income (J : ℝ) := 0.6489 * J

-- Define Nina's total income
def nina_total_income (J : ℝ) := 1.3375 * J

-- Define the sum of the total incomes of Mary, Lisa, and Nina
def sum_income (J : ℝ) := mary_total_income J + lisa_total_income J + nina_total_income J

-- Define the statement we need to prove: the percentage of Juan's total income
theorem total_income_percentage (J : ℝ) (hJ : J ≠ 0) :
  ((sum_income J / juan_base_income J) * 100) = 321.84 :=
by
  unfold juan_base_income sum_income mary_total_income lisa_total_income nina_total_income
  sorry

end NUMINAMATH_GPT_total_income_percentage_l1901_190103


namespace NUMINAMATH_GPT_min_value_fraction_l1901_190128

theorem min_value_fraction (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (h_sum : a + 3 * b = 1) :
  (∀ x y : ℝ, (0 < x) → (0 < y) → x + 3 * y = 1 → 16 ≤ 1 / x + 3 / y) :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l1901_190128


namespace NUMINAMATH_GPT_ratio_of_areas_l1901_190105

noncomputable def side_length_S : ℝ := sorry
noncomputable def side_length_longer_R : ℝ := 1.2 * side_length_S
noncomputable def side_length_shorter_R : ℝ := 0.8 * side_length_S
noncomputable def area_S : ℝ := side_length_S ^ 2
noncomputable def area_R : ℝ := side_length_longer_R * side_length_shorter_R

theorem ratio_of_areas :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1901_190105


namespace NUMINAMATH_GPT_no_real_solution_3x2_plus_9x_le_neg12_l1901_190192

/-- There are no real values of x such that 3x^2 + 9x ≤ -12. -/
theorem no_real_solution_3x2_plus_9x_le_neg12 (x : ℝ) : ¬(3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_3x2_plus_9x_le_neg12_l1901_190192


namespace NUMINAMATH_GPT_profitable_allocation_2015_l1901_190150

theorem profitable_allocation_2015 :
  ∀ (initial_price : ℝ) (final_price : ℝ)
    (annual_interest_2015 : ℝ) (two_year_interest : ℝ) (annual_interest_2016 : ℝ),
  initial_price = 70 ∧ final_price = 85 ∧ annual_interest_2015 = 0.16 ∧
  two_year_interest = 0.15 ∧ annual_interest_2016 = 0.10 →
  (initial_price * (1 + annual_interest_2015) * (1 + annual_interest_2016) > final_price) ∨
  (initial_price * (1 + two_year_interest)^2 > final_price) :=
by
  intros initial_price final_price annual_interest_2015 two_year_interest annual_interest_2016
  intro h
  sorry

end NUMINAMATH_GPT_profitable_allocation_2015_l1901_190150


namespace NUMINAMATH_GPT_not_perfect_square_l1901_190135

theorem not_perfect_square (a b : ℤ) (h1 : a > b) (h2 : Int.gcd (ab - 1) (a + b) = 1) (h3 : Int.gcd (ab + 1) (a - b) = 1) :
  ¬ ∃ c : ℤ, (a + b)^2 + (ab - 1)^2 = c^2 := 
  sorry

end NUMINAMATH_GPT_not_perfect_square_l1901_190135


namespace NUMINAMATH_GPT_fifth_student_guess_l1901_190195

theorem fifth_student_guess (s1 s2 s3 s4 s5 : ℕ) 
(h1 : s1 = 100)
(h2 : s2 = 8 * s1)
(h3 : s3 = s2 - 200)
(h4 : s4 = (s1 + s2 + s3) / 3 + 25)
(h5 : s5 = s4 + s4 / 5) : 
s5 = 630 :=
sorry

end NUMINAMATH_GPT_fifth_student_guess_l1901_190195


namespace NUMINAMATH_GPT_Alyssa_total_spent_l1901_190193

-- define the amounts spent on grapes and cherries
def costGrapes: ℝ := 12.08
def costCherries: ℝ := 9.85

-- define the total cost based on the given conditions
def totalCost: ℝ := costGrapes + costCherries

-- prove that the total cost equals 21.93
theorem Alyssa_total_spent:
  totalCost = 21.93 := 
  by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_Alyssa_total_spent_l1901_190193


namespace NUMINAMATH_GPT_Q_div_P_l1901_190174

theorem Q_div_P (P Q : ℤ) (h : ∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 →
  P / (x + 7) + Q / (x * (x - 6)) = (x^2 - x + 15) / (x^3 + x^2 - 42 * x)) :
  Q / P = 7 :=
sorry

end NUMINAMATH_GPT_Q_div_P_l1901_190174


namespace NUMINAMATH_GPT_common_solutions_form_segment_length_one_l1901_190164

theorem common_solutions_form_segment_length_one (a : ℝ) (h₁ : ∀ x : ℝ, x^2 - 4 * x + 2 - a ≤ 0) 
  (h₂ : ∀ x : ℝ, x^2 - 5 * x + 2 * a + 8 ≤ 0) : 
  (a = -1 ∨ a = -7 / 4) :=
by
  sorry

end NUMINAMATH_GPT_common_solutions_form_segment_length_one_l1901_190164


namespace NUMINAMATH_GPT_expression_value_l1901_190156

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem expression_value :
  let numerator := factorial 10
  let denominator := (1 + 2) * (3 + 4) * (5 + 6) * (7 + 8) * (9 + 10)
  numerator / denominator = 660 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1901_190156
