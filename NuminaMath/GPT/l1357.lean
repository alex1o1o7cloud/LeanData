import Mathlib

namespace cos_double_angle_at_origin_l1357_135719

noncomputable def vertex : ℝ × ℝ := (0, 0)
noncomputable def initial_side : ℝ × ℝ := (1, 0)
noncomputable def terminal_side : ℝ × ℝ := (-1, 3)
noncomputable def cos2alpha (v i t : ℝ × ℝ) : ℝ :=
  2 * ((t.1) / (Real.sqrt (t.1 ^ 2 + t.2 ^ 2))) ^ 2 - 1

theorem cos_double_angle_at_origin :
  cos2alpha vertex initial_side terminal_side = -4 / 5 :=
by
  sorry

end cos_double_angle_at_origin_l1357_135719


namespace find_minutes_per_mile_l1357_135781

-- Conditions
def num_of_movies : ℕ := 2
def avg_length_of_movie_hours : ℝ := 1.5
def total_distance_miles : ℝ := 15

-- Question and proof target
theorem find_minutes_per_mile :
  (num_of_movies * avg_length_of_movie_hours * 60) / total_distance_miles = 12 :=
by
  -- Insert the proof here (not required as per the task instructions)
  sorry

end find_minutes_per_mile_l1357_135781


namespace find_a_and_union_l1357_135780

noncomputable def A (a : ℝ) : Set ℝ := { -4, 2 * a - 1, a ^ 2 }
noncomputable def B (a : ℝ) : Set ℝ := { a - 5, 1 - a, 9 }

theorem find_a_and_union {a : ℝ}
  (h : A a ∩ B a = {9}): 
  a = -3 ∧ A a ∪ B a = {-8, -7, -4, 4, 9} :=
by
  sorry

end find_a_and_union_l1357_135780


namespace smallest_integer_n_condition_l1357_135711

theorem smallest_integer_n_condition :
  (∃ n : ℕ, n > 0 ∧ (∀ (m : ℤ), (1 ≤ m ∧ m ≤ 1992) → (∃ (k : ℤ), (m : ℚ) / 1993 < k / n ∧ k / n < (m + 1 : ℚ) / 1994))) ↔ n = 3987 :=
sorry

end smallest_integer_n_condition_l1357_135711


namespace find_prime_pairs_l1357_135721

open Nat

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def valid_prime_pairs (p q : ℕ): Prop :=
  is_prime p ∧ is_prime q ∧ divides p (30 * q - 1) ∧ divides q (30 * p - 1)

theorem find_prime_pairs :
  { (p, q) | valid_prime_pairs p q } = { (7, 11), (11, 7), (59, 61), (61, 59) } :=
sorry

end find_prime_pairs_l1357_135721


namespace compare_xyz_l1357_135774

theorem compare_xyz
  (a b c d : ℝ) (h : a < b ∧ b < c ∧ c < d)
  (x : ℝ) (hx : x = (a + b) * (c + d))
  (y : ℝ) (hy : y = (a + c) * (b + d))
  (z : ℝ) (hz : z = (a + d) * (b + c)) :
  x < y ∧ y < z :=
by sorry

end compare_xyz_l1357_135774


namespace joan_gave_28_seashells_to_sam_l1357_135700

/-- 
Given:
- Joan found 70 seashells on the beach.
- After giving away some seashells, she has 27 left.
- She gave twice as many seashells to Sam as she gave to her friend Lily.

Show that:
- Joan gave 28 seashells to Sam.
-/
theorem joan_gave_28_seashells_to_sam (L S : ℕ) 
  (h1 : S = 2 * L) 
  (h2 : 70 - 27 = 43) 
  (h3 : L + S = 43) :
  S = 28 :=
by
  sorry

end joan_gave_28_seashells_to_sam_l1357_135700


namespace arithmetic_sequence_a6_l1357_135701

theorem arithmetic_sequence_a6 (a : ℕ → ℝ)
  (h4_8 : ∃ a4 a8, (a 4 = a4) ∧ (a 8 = a8) ∧ a4^2 - 6*a4 + 5 = 0 ∧ a8^2 - 6*a8 + 5 = 0) :
  a 6 = 3 := by 
  sorry

end arithmetic_sequence_a6_l1357_135701


namespace cars_selected_l1357_135772

theorem cars_selected (num_cars num_clients selections_made total_selections : ℕ)
  (h1 : num_cars = 16)
  (h2 : num_clients = 24)
  (h3 : selections_made = 2)
  (h4 : total_selections = num_clients * selections_made) :
  num_cars * (total_selections / num_cars) = 48 :=
by
  sorry

end cars_selected_l1357_135772


namespace average_visitors_in_month_of_30_days_starting_with_sunday_l1357_135754

def average_visitors_per_day (sundays_visitors : ℕ) (other_days_visitors : ℕ) (num_sundays : ℕ) (num_other_days : ℕ) : ℕ :=
  (sundays_visitors * num_sundays + other_days_visitors * num_other_days) / (num_sundays + num_other_days)

theorem average_visitors_in_month_of_30_days_starting_with_sunday :
  average_visitors_per_day 1000 700 5 25 = 750 := sorry

end average_visitors_in_month_of_30_days_starting_with_sunday_l1357_135754


namespace Hay_s_Linens_sales_l1357_135708

theorem Hay_s_Linens_sales :
  ∃ (n : ℕ), 500 ≤ 52 * n ∧ 52 * n ≤ 700 ∧
             ∀ m, (500 ≤ 52 * m ∧ 52 * m ≤ 700) → n ≤ m :=
sorry

end Hay_s_Linens_sales_l1357_135708


namespace a_2013_is_4_l1357_135756

theorem a_2013_is_4
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 7)
  (h3 : ∀ n : ℕ, a (n+2) = (a n * a (n+1)) % 10) :
  a 2013 = 4 :=
sorry

end a_2013_is_4_l1357_135756


namespace total_triangles_correct_l1357_135743

-- Define the rectangle and additional constructions
structure Rectangle :=
  (A B C D : Type)
  (midpoint_AB midpoint_BC midpoint_CD midpoint_DA : Type)
  (AC BD diagonals : Type)

-- Hypothesize the structure
variables (rect : Rectangle)

-- Define the number of triangles
def number_of_triangles (r : Rectangle) : Nat := 16

-- The theorem statement
theorem total_triangles_correct : number_of_triangles rect = 16 :=
by
  sorry

end total_triangles_correct_l1357_135743


namespace greatest_integer_solution_l1357_135762

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 36 ≤ 0) : n ≤ 9 :=
by
  sorry

end greatest_integer_solution_l1357_135762


namespace twelve_hens_lay_48_eggs_in_twelve_days_l1357_135767

theorem twelve_hens_lay_48_eggs_in_twelve_days :
  (∀ (hens eggs days : ℕ), hens = 3 → eggs = 3 → days = 3 → eggs / (hens * days) = 1/3) → 
  ∀ (hens days : ℕ), hens = 12 → days = 12 → hens * days * (1/3) = 48 :=
by
  sorry

end twelve_hens_lay_48_eggs_in_twelve_days_l1357_135767


namespace cos_330_eq_sqrt_3_div_2_l1357_135759

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l1357_135759


namespace possible_winning_scores_count_l1357_135726

def total_runners := 15
def total_score := (total_runners * (total_runners + 1)) / 2

def min_score := 15
def max_potential_score := 39

def is_valid_winning_score (score : ℕ) : Prop :=
  min_score ≤ score ∧ score ≤ max_potential_score

theorem possible_winning_scores_count : 
  ∃ scores : Finset ℕ, ∀ score ∈ scores, is_valid_winning_score score ∧ Finset.card scores = 25 := 
sorry

end possible_winning_scores_count_l1357_135726


namespace find_value_l1357_135771

variable {a b : ℝ}

theorem find_value (h : 2 * a + b + 1 = 0) : 1 + 4 * a + 2 * b = -1 := 
by
  sorry

end find_value_l1357_135771


namespace probability_not_same_level_is_four_fifths_l1357_135765

-- Definitions of the conditions
def nobility_levels := 5
def total_outcomes := nobility_levels * nobility_levels
def same_level_outcomes := nobility_levels

-- Definition of the probability
def probability_not_same_level := 1 - (same_level_outcomes / total_outcomes : ℚ)

-- The theorem statement
theorem probability_not_same_level_is_four_fifths :
  probability_not_same_level = 4 / 5 := 
  by sorry

end probability_not_same_level_is_four_fifths_l1357_135765


namespace total_number_of_workers_l1357_135753

theorem total_number_of_workers 
  (W : ℕ) 
  (avg_all : ℕ) 
  (n_technicians : ℕ) 
  (avg_technicians : ℕ) 
  (avg_non_technicians : ℕ) :
  avg_all * W = avg_technicians * n_technicians + avg_non_technicians * (W - n_technicians) →
  avg_all = 8000 →
  n_technicians = 7 →
  avg_technicians = 12000 →
  avg_non_technicians = 6000 →
  W = 21 :=
by 
  intro h1 h2 h3 h4 h5
  sorry

end total_number_of_workers_l1357_135753


namespace gcd_of_324_and_135_l1357_135720

theorem gcd_of_324_and_135 : Nat.gcd 324 135 = 27 :=
by
  sorry

end gcd_of_324_and_135_l1357_135720


namespace range_of_a_l1357_135763

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (a * x - 1) / x > 2 * a) ↔ a ∈ (Set.Ici (-1/2) : Set ℝ) :=
by
  sorry

end range_of_a_l1357_135763


namespace min_value_expression_l1357_135795

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  ∃ c, c = (1/(a+1) + 4/(b+1)) ∧ c ≥ 9/4 :=
by
  sorry

end min_value_expression_l1357_135795


namespace shopkeepers_total_profit_percentage_l1357_135725

noncomputable def calculateProfitPercentage : ℝ :=
  let oranges := 1000
  let bananas := 800
  let apples := 750
  let rotten_oranges_percentage := 0.12
  let rotten_bananas_percentage := 0.05
  let rotten_apples_percentage := 0.10
  let profit_oranges_percentage := 0.20
  let profit_bananas_percentage := 0.25
  let profit_apples_percentage := 0.15
  let cost_per_orange := 2.5
  let cost_per_banana := 1.5
  let cost_per_apple := 2.0

  let rotten_oranges := rotten_oranges_percentage * oranges
  let rotten_bananas := rotten_bananas_percentage * bananas
  let rotten_apples := rotten_apples_percentage * apples

  let good_oranges := oranges - rotten_oranges
  let good_bananas := bananas - rotten_bananas
  let good_apples := apples - rotten_apples

  let cost_oranges := cost_per_orange * oranges
  let cost_bananas := cost_per_banana * bananas
  let cost_apples := cost_per_apple * apples

  let total_cost := cost_oranges + cost_bananas + cost_apples

  let selling_price_oranges := cost_per_orange * (1 + profit_oranges_percentage) * good_oranges
  let selling_price_bananas := cost_per_banana * (1 + profit_bananas_percentage) * good_bananas
  let selling_price_apples := cost_per_apple * (1 + profit_apples_percentage) * good_apples

  let total_selling_price := selling_price_oranges + selling_price_bananas + selling_price_apples

  let total_profit := total_selling_price - total_cost

  (total_profit / total_cost) * 100

theorem shopkeepers_total_profit_percentage :
  calculateProfitPercentage = 8.03 := sorry

end shopkeepers_total_profit_percentage_l1357_135725


namespace max_value_2x_plus_y_l1357_135704

theorem max_value_2x_plus_y (x y : ℝ) (h : y^2 / 4 + x^2 / 3 = 1) : 2 * x + y ≤ 4 :=
by
  sorry

end max_value_2x_plus_y_l1357_135704


namespace ammonium_bromide_total_weight_l1357_135733

noncomputable def nitrogen_weight : ℝ := 14.01
noncomputable def hydrogen_weight : ℝ := 1.01
noncomputable def bromine_weight : ℝ := 79.90
noncomputable def ammonium_bromide_weight : ℝ := nitrogen_weight + 4 * hydrogen_weight + bromine_weight
noncomputable def moles : ℝ := 5
noncomputable def total_weight : ℝ := moles * ammonium_bromide_weight

theorem ammonium_bromide_total_weight :
  total_weight = 489.75 :=
by
  -- The proof is omitted.
  sorry

end ammonium_bromide_total_weight_l1357_135733


namespace price_verification_l1357_135748

noncomputable def price_on_hot_day : ℚ :=
  let P : ℚ := 225 / 172
  1.25 * P

theorem price_verification :
  (32 * 7 * (225 / 172) + 32 * 3 * (1.25 * (225 / 172)) - (32 * 10 * 0.75)) = 210 :=
sorry

end price_verification_l1357_135748


namespace francis_had_2_muffins_l1357_135723

noncomputable def cost_of_francis_breakfast (m : ℕ) : ℕ := 2 * m + 6
noncomputable def cost_of_kiera_breakfast : ℕ := 4 + 3
noncomputable def total_cost (m : ℕ) : ℕ := cost_of_francis_breakfast m + cost_of_kiera_breakfast

theorem francis_had_2_muffins (m : ℕ) : total_cost m = 17 → m = 2 :=
by
  -- Sorry is used here to leave the proof steps blank.
  sorry

end francis_had_2_muffins_l1357_135723


namespace range_of_x_l1357_135728

-- Defining the vectors as given in the conditions
def a (x : ℝ) : ℝ × ℝ := (x, 3)
def b : ℝ × ℝ := (2, -1)

-- Defining the condition that the angle is obtuse
def is_obtuse (x : ℝ) : Prop := 
  let dot_product := (a x).1 * b.1 + (a x).2 * b.2
  dot_product < 0

-- Defining the condition that vectors are not in opposite directions
def not_opposite_directions (x : ℝ) : Prop := x ≠ -6

-- Proving the required range of x
theorem range_of_x (x : ℝ) :
  is_obtuse x → not_opposite_directions x → x < 3 / 2 :=
sorry

end range_of_x_l1357_135728


namespace arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l1357_135788

-- Arithmetic Progression
theorem arithmetic_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 - x2 = x2 - x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = (2 * a^3 + 27 * c) / (9 * a)) :=
sorry

-- Geometric Progression
theorem geometric_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x2 / x1 = x3 / x2 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = a * c^(1/3)) :=
sorry

-- Harmonic Sequence
theorem harmonic_sequence_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x1 - x2) / (x2 - x3) = x1 / x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (a = (2 * b^3 + 27 * c) / (9 * b^2)) :=
sorry

end arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l1357_135788


namespace probability_of_log2N_is_integer_and_N_is_even_l1357_135714

-- Defining the range of N as a four-digit number in base four
def is_base4_four_digit (N : ℕ) : Prop := 64 ≤ N ∧ N ≤ 255

-- Defining the condition that log_2 N is an integer
def is_power_of_two (N : ℕ) : Prop := ∃ k : ℕ, N = 2^k

-- Defining the condition that N is even
def is_even (N : ℕ) : Prop := N % 2 = 0

-- Combining all conditions
def meets_conditions (N : ℕ) : Prop := is_base4_four_digit N ∧ is_power_of_two N ∧ is_even N

-- Total number of four-digit numbers in base four
def total_base4_four_digits : ℕ := 192

-- Set of N values that meet the conditions
def valid_N_values : Finset ℕ := {64, 128}

-- The probability calculation
def calculated_probability : ℚ := valid_N_values.card / total_base4_four_digits

-- The final proof statement
theorem probability_of_log2N_is_integer_and_N_is_even : calculated_probability = 1 / 96 :=
by
  -- Prove the equality here (matching the solution given)
  sorry

end probability_of_log2N_is_integer_and_N_is_even_l1357_135714


namespace asha_remaining_money_l1357_135731

-- Define the borrowed amounts, gift, and savings
def borrowed_from_brother : ℤ := 20
def borrowed_from_father : ℤ := 40
def borrowed_from_mother : ℤ := 30
def gift_from_granny : ℤ := 70
def savings : ℤ := 100

-- Total amount of money Asha has
def total_amount : ℤ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

-- Amount spent by Asha
def amount_spent : ℤ := (3 * total_amount) / 4

-- Amount of money Asha remains with
def amount_left : ℤ := total_amount - amount_spent

-- The proof statement
theorem asha_remaining_money : amount_left = 65 := by
  sorry

end asha_remaining_money_l1357_135731


namespace sin_cos_theta_l1357_135727

open Real

theorem sin_cos_theta (θ : ℝ) (H1 : θ > π / 2 ∧ θ < π) (H2 : tan (θ + π / 4) = 1 / 2) :
  sin θ + cos θ = -sqrt 10 / 5 :=
by
  sorry

end sin_cos_theta_l1357_135727


namespace alayas_fruit_salads_l1357_135787

theorem alayas_fruit_salads (A : ℕ) (H1 : 2 * A + A = 600) : A = 200 := 
by
  sorry

end alayas_fruit_salads_l1357_135787


namespace initial_salary_increase_l1357_135735

theorem initial_salary_increase :
  ∃ x : ℝ, 5000 * (1 + x/100) * 0.95 = 5225 := by
  sorry

end initial_salary_increase_l1357_135735


namespace scientific_notation_of_population_l1357_135766

theorem scientific_notation_of_population : (85000000 : ℝ) = 8.5 * 10^7 := 
by
  sorry

end scientific_notation_of_population_l1357_135766


namespace max_number_of_eligible_ages_l1357_135707

-- Definitions based on the problem conditions
def average_age : ℝ := 31
def std_dev : ℝ := 5
def acceptable_age_range (a : ℝ) : Prop := 26 ≤ a ∧ a ≤ 36
def has_masters_degree : Prop := 24 ≤ 26  -- simplified for context indicated in problem
def has_work_experience : Prop := 26 ≥ 26

-- Define the maximum number of different ages of the eligible applicants
noncomputable def max_diff_ages : ℕ := 36 - 26 + 1  -- This matches the solution step directly

-- The theorem stating the result
theorem max_number_of_eligible_ages :
  max_diff_ages = 11 :=
by {
  sorry
}

end max_number_of_eligible_ages_l1357_135707


namespace ninth_term_of_geometric_sequence_l1357_135715

theorem ninth_term_of_geometric_sequence :
  let a1 := (5 : ℚ)
  let r := (3 / 4 : ℚ)
  (a1 * r^8) = (32805 / 65536 : ℚ) :=
by {
  sorry
}

end ninth_term_of_geometric_sequence_l1357_135715


namespace find_FC_l1357_135744

variable (DC : ℝ) (CB : ℝ) (AB AD ED : ℝ)
variable (FC : ℝ)
variable (h1 : DC = 9)
variable (h2 : CB = 6)
variable (h3 : AB = (1/3) * AD)
variable (h4 : ED = (2/3) * AD)

theorem find_FC : FC = 9 :=
by sorry

end find_FC_l1357_135744


namespace probability_three_specific_cards_l1357_135798

theorem probability_three_specific_cards :
  let total_deck := 52
  let total_spades := 13
  let total_tens := 4
  let total_queens := 4
  let p_case1 := ((12:ℚ) / total_deck) * (total_tens / (total_deck - 1)) * (total_queens / (total_deck - 2))
  let p_case2 := ((1:ℚ) / total_deck) * ((total_tens - 1) / (total_deck - 1)) * (total_queens / (total_deck - 2))
  p_case1 + p_case2 = (17:ℚ) / 11050 :=
by
  sorry

end probability_three_specific_cards_l1357_135798


namespace remaining_macaroons_weight_is_103_l1357_135776

-- Definitions based on the conditions
def coconutMacaroonsInitialCount := 12
def coconutMacaroonWeight := 5
def coconutMacaroonsBags := 4

def almondMacaroonsInitialCount := 8
def almondMacaroonWeight := 8
def almondMacaroonsBags := 2

def whiteChocolateMacaroonsInitialCount := 2
def whiteChocolateMacaroonWeight := 10

def steveAteCoconutMacaroons := coconutMacaroonsInitialCount / coconutMacaroonsBags
def steveAteAlmondMacaroons := (almondMacaroonsInitialCount / almondMacaroonsBags) / 2
def steveAteWhiteChocolateMacaroons := 1

-- Calculation of remaining macaroons weights
def remainingCoconutMacaroonsCount := coconutMacaroonsInitialCount - steveAteCoconutMacaroons
def remainingAlmondMacaroonsCount := almondMacaroonsInitialCount - steveAteAlmondMacaroons
def remainingWhiteChocolateMacaroonsCount := whiteChocolateMacaroonsInitialCount - steveAteWhiteChocolateMacaroons

-- Calculation of total remaining weight
def remainingCoconutMacaroonsWeight := remainingCoconutMacaroonsCount * coconutMacaroonWeight
def remainingAlmondMacaroonsWeight := remainingAlmondMacaroonsCount * almondMacaroonWeight
def remainingWhiteChocolateMacaroonsWeight := remainingWhiteChocolateMacaroonsCount * whiteChocolateMacaroonWeight

def totalRemainingWeight := remainingCoconutMacaroonsWeight + remainingAlmondMacaroonsWeight + remainingWhiteChocolateMacaroonsWeight

-- Statement to be proved
theorem remaining_macaroons_weight_is_103 :
  totalRemainingWeight = 103 := by
  sorry

end remaining_macaroons_weight_is_103_l1357_135776


namespace wool_usage_l1357_135710

def total_balls_of_wool_used (scarves_aaron sweaters_aaron sweaters_enid : ℕ) (wool_per_scarf wool_per_sweater : ℕ) : ℕ :=
  (scarves_aaron * wool_per_scarf) + (sweaters_aaron * wool_per_sweater) + (sweaters_enid * wool_per_sweater)

theorem wool_usage :
  total_balls_of_wool_used 10 5 8 3 4 = 82 :=
by
  -- calculations done in solution steps
  -- total_balls_of_wool_used (10 scarves * 3 balls/scarf) + (5 sweaters * 4 balls/sweater) + (8 sweaters * 4 balls/sweater)
  -- total_balls_of_wool_used (30) + (20) + (32)
  -- total_balls_of_wool_used = 30 + 20 + 32 = 82
  sorry

end wool_usage_l1357_135710


namespace smallest_n_logarithm_l1357_135761

theorem smallest_n_logarithm :
  ∃ n : ℕ, 0 < n ∧ 
  (Real.log (Real.log n / Real.log 3) / Real.log 3^2 =
  Real.log (Real.log n / Real.log 2) / Real.log 2^3) ∧ 
  n = 9 :=
by
  sorry

end smallest_n_logarithm_l1357_135761


namespace min_value_48_l1357_135770

noncomputable def min_value {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : ℝ :=
  1 / a + 27 / b

theorem min_value_48 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : 
  min_value ha hb h = 48 := 
sorry

end min_value_48_l1357_135770


namespace negation_of_universal_proposition_l1357_135764

open Classical

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^2 > x) ↔ (∃ x : ℕ, x^2 ≤ x) :=
by
  sorry

end negation_of_universal_proposition_l1357_135764


namespace recycling_target_l1357_135713

/-- Six Grade 4 sections launched a recycling drive where they collect old newspapers to recycle.
Each section collected 280 kilos in two weeks. After the third week, they found that they need 320 kilos more to reach their target.
  How many kilos of the newspaper is their target? -/
theorem recycling_target (sections : ℕ) (kilos_collected_2_weeks : ℕ) (additional_kilos : ℕ) : 
  sections = 6 ∧ kilos_collected_2_weeks = 280 ∧ additional_kilos = 320 → 
  (sections * (kilos_collected_2_weeks / 2) * 3 + additional_kilos) = 2840 :=
by
  sorry

end recycling_target_l1357_135713


namespace angle_bisectors_and_median_inequality_l1357_135745

open Real

variables (A B C : Point)
variables (a b c : ℝ) -- sides of the triangle
variables (p : ℝ) -- semi-perimeter of the triangle
variables (la lb mc : ℝ) -- angle bisectors and median lengths

-- Assume the given conditions
axiom angle_bisector_la (A B C : Point) : ℝ -- lengths of the angle bisector of ∠BAC
axiom angle_bisector_lb (A B C : Point) : ℝ -- lengths of the angle bisector of ∠ABC
axiom median_mc (A B C : Point) : ℝ -- length of the median from vertex C
axiom semi_perimeter (a b c : ℝ) : ℝ -- semi-perimeter of the triangle

-- The statement of the theorem
theorem angle_bisectors_and_median_inequality (la lb mc p : ℝ) :
  la + lb + mc ≤ sqrt 3 * p :=
sorry

end angle_bisectors_and_median_inequality_l1357_135745


namespace arithmetic_seq_a8_l1357_135784

theorem arithmetic_seq_a8 : ∀ (a : ℕ → ℤ), 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → 
  (a 5 + a 6 = 22) → 
  (a 3 = 7) → 
  a 8 = 15 :=
by
  intros a ha_arithmetic hsum h3
  sorry

end arithmetic_seq_a8_l1357_135784


namespace rational_solution_counts_l1357_135778

theorem rational_solution_counts :
  (∃ (x y : ℚ), x^2 + y^2 = 2) ∧ 
  (¬ ∃ (x y : ℚ), x^2 + y^2 = 3) := 
by 
  sorry

end rational_solution_counts_l1357_135778


namespace find_x_between_0_and_180_l1357_135739

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l1357_135739


namespace base_b_square_of_15_l1357_135718

theorem base_b_square_of_15 (b : ℕ) (h : (b + 5) * (b + 5) = 4 * b^2 + 3 * b + 6) : b = 8 :=
sorry

end base_b_square_of_15_l1357_135718


namespace min_value_expression_l1357_135779

theorem min_value_expression : ∃ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 6 * x + 4 * y + 5 = 2 := 
sorry

end min_value_expression_l1357_135779


namespace average_of_multiples_of_9_l1357_135702

-- Define the problem in Lean
theorem average_of_multiples_of_9 :
  let pos_multiples := [9, 18, 27, 36, 45]
  let neg_multiples := [-9, -18, -27, -36, -45]
  (pos_multiples.sum + neg_multiples.sum) / 2 = 0 :=
by
  sorry

end average_of_multiples_of_9_l1357_135702


namespace find_r_condition_l1357_135746

variable {x y z w r : ℝ}

axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : z ≠ 0
axiom h4 : w ≠ 0
axiom h5 : (x ≠ y) ∧ (x ≠ z) ∧ (x ≠ w) ∧ (y ≠ z) ∧ (y ≠ w) ∧ (z ≠ w)

noncomputable def is_geometric_progression (a b c d : ℝ) (r : ℝ) : Prop :=
  b = a * r ∧ c = a * r^2 ∧ d = a * r^3

theorem find_r_condition :
  is_geometric_progression (x * (y - z)) (y * (z - x)) (z * (x - y)) (w * (y - x)) r →
  r^3 + r^2 + r + 1 = 0 :=
by
  intros
  sorry

end find_r_condition_l1357_135746


namespace marble_189_is_gray_l1357_135703

def marble_color (n : ℕ) : String :=
  let cycle_length := 14
  let gray_thres := 5
  let white_thres := 9
  let black_thres := 12
  let position := (n - 1) % cycle_length + 1
  if position ≤ gray_thres then "gray"
  else if position ≤ white_thres then "white"
  else if position ≤ black_thres then "black"
  else "blue"

theorem marble_189_is_gray : marble_color 189 = "gray" :=
by {
  -- We assume the necessary definitions and steps discussed above.
  sorry
}

end marble_189_is_gray_l1357_135703


namespace permutations_of_BANANA_l1357_135751

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l1357_135751


namespace polynomial_remainder_l1357_135732

theorem polynomial_remainder :
  ∀ (x : ℝ), (x^4 + 2 * x^3 - 3 * x^2 + 4 * x - 5) % (x^2 - 3 * x + 2) = (24 * x - 25) :=
by
  sorry

end polynomial_remainder_l1357_135732


namespace smallest_angle_of_triangle_l1357_135794

theorem smallest_angle_of_triangle (k : ℕ) (h : 4 * k + 5 * k + 9 * k = 180) : 4 * k = 40 :=
by {
  sorry
}

end smallest_angle_of_triangle_l1357_135794


namespace volume_of_cube_is_correct_surface_area_of_cube_is_correct_l1357_135769

-- Define the conditions: total edge length of the cube frame
def total_edge_length : ℕ := 60
def number_of_edges : ℕ := 12

-- Define the edge length of the cube
def edge_length (total_edge_length number_of_edges : ℕ) : ℕ := total_edge_length / number_of_edges

-- Define the volume of the cube
def cube_volume (a : ℕ) : ℕ := a ^ 3

-- Define the surface area of the cube
def cube_surface_area (a : ℕ) : ℕ := 6 * (a ^ 2)

-- Volume Proof Statement
theorem volume_of_cube_is_correct : cube_volume (edge_length total_edge_length number_of_edges) = 125 :=
by
  sorry

-- Surface Area Proof Statement
theorem surface_area_of_cube_is_correct : cube_surface_area (edge_length total_edge_length number_of_edges) = 150 :=
by
  sorry

end volume_of_cube_is_correct_surface_area_of_cube_is_correct_l1357_135769


namespace sin_double_angle_identity_l1357_135734

open Real 

theorem sin_double_angle_identity 
  (A : ℝ) 
  (h1 : 0 < A) 
  (h2 : A < π / 2) 
  (h3 : cos A = 3 / 5) : 
  sin (2 * A) = 24 / 25 :=
by 
  sorry

end sin_double_angle_identity_l1357_135734


namespace largest_square_plots_l1357_135706

theorem largest_square_plots (width length pathway_material : Nat) (width_eq : width = 30) (length_eq : length = 60) (pathway_material_eq : pathway_material = 2010) : ∃ (n : Nat), n * (2 * n) = 578 := 
by
  sorry

end largest_square_plots_l1357_135706


namespace minimum_cost_of_candies_l1357_135797

variable (Orange Apple Grape Strawberry : ℕ)

-- Conditions
def CandyRelation1 := Apple = 2 * Orange
def CandyRelation2 := Strawberry = 2 * Grape
def CandyRelation3 := Apple = 2 * Strawberry
def TotalCandies := Orange + Apple + Grape + Strawberry = 90
def CandyCost := 0.1

-- Question
theorem minimum_cost_of_candies :
  CandyRelation1 Orange Apple → 
  CandyRelation2 Grape Strawberry → 
  CandyRelation3 Apple Strawberry → 
  TotalCandies Orange Apple Grape Strawberry → 
  Orange ≥ 3 ∧ Apple ≥ 3 ∧ Grape ≥ 3 ∧ Strawberry ≥ 3 →
  (5 * CandyCost + 3 * CandyCost + 3 * CandyCost + 3 * CandyCost = 1.4) :=
sorry

end minimum_cost_of_candies_l1357_135797


namespace solve_system_l1357_135738

-- The system of equations as conditions in Lean
def system1 (x y : ℤ) : Prop := 5 * x + 2 * y = 25
def system2 (x y : ℤ) : Prop := 3 * x + 4 * y = 15

-- The statement that asserts the solution is (x = 5, y = 0)
theorem solve_system : ∃ x y : ℤ, system1 x y ∧ system2 x y ∧ x = 5 ∧ y = 0 :=
by
  sorry

end solve_system_l1357_135738


namespace union_of_sets_l1357_135705

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (hA : A = {x, y}) (hB : B = {x + 1, 5}) (h_inter : A ∩ B = {2}) :
  A ∪ B = {1, 2, 5} :=
by
  sorry

end union_of_sets_l1357_135705


namespace boys_from_school_A_study_science_l1357_135716

theorem boys_from_school_A_study_science (total_boys school_A_percent non_science_boys school_A_boys study_science_boys: ℕ) 
(h1 : total_boys = 300)
(h2 : school_A_percent = 20)
(h3 : non_science_boys = 42)
(h4 : school_A_boys = (school_A_percent * total_boys) / 100)
(h5 : study_science_boys = school_A_boys - non_science_boys) :
(study_science_boys * 100 / school_A_boys) = 30 :=
by
  sorry

end boys_from_school_A_study_science_l1357_135716


namespace maria_total_eggs_l1357_135736

def total_eggs (boxes : ℕ) (eggs_per_box : ℕ) : ℕ :=
  boxes * eggs_per_box

theorem maria_total_eggs :
  total_eggs 3 7 = 21 :=
by
  -- Here, you would normally show the steps of computation
  -- which we can skip with sorry
  sorry

end maria_total_eggs_l1357_135736


namespace problem_statement_l1357_135758

theorem problem_statement (x y : ℝ) (h : -x + 2 * y = 5) :
  5 * (x - 2 * y) ^ 2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  sorry

end problem_statement_l1357_135758


namespace original_volume_of_ice_cube_l1357_135747

theorem original_volume_of_ice_cube
  (V : ℝ)
  (h1 : V * (1/2) * (2/3) * (3/4) * (4/5) = 30)
  : V = 150 :=
sorry

end original_volume_of_ice_cube_l1357_135747


namespace solve_m_l1357_135792

theorem solve_m (m : ℝ) : (m + 1) / 6 = m / 1 → m = 1 / 5 :=
by
  intro h
  sorry

end solve_m_l1357_135792


namespace necessary_but_not_sufficient_condition_for_a_lt_neg_one_l1357_135799

theorem necessary_but_not_sufficient_condition_for_a_lt_neg_one (a : ℝ) : 
  (1 / a > -1) ↔ (a < -1) :=
by sorry

end necessary_but_not_sufficient_condition_for_a_lt_neg_one_l1357_135799


namespace angle_coterminal_l1357_135775

theorem angle_coterminal (k : ℤ) : 
  ∃ α : ℝ, α = 30 + k * 360 :=
sorry

end angle_coterminal_l1357_135775


namespace reciprocal_sum_neg_l1357_135741

theorem reciprocal_sum_neg (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c = 8) : (1/a) + (1/b) + (1/c) < 0 := 
sorry

end reciprocal_sum_neg_l1357_135741


namespace parabola_focus_directrix_distance_l1357_135785

theorem parabola_focus_directrix_distance {a : ℝ} (h₀ : a > 0):
  (∃ (b : ℝ), ∃ (x1 x2 : ℝ), (x1 + x2 = 1 / a) ∧ (1 / (2 * a) = 1)) → 
  (1 / (2 * a) / 2 = 1 / 4) :=
by
  sorry

end parabola_focus_directrix_distance_l1357_135785


namespace simplify_expr_l1357_135760

theorem simplify_expr : 2 - 2 / (1 + Real.sqrt 2) - 2 / (1 - Real.sqrt 2) = -2 := by
  sorry

end simplify_expr_l1357_135760


namespace sam_earnings_difference_l1357_135783

def hours_per_dollar := 1 / 10  -- Sam earns $10 per hour, so it takes 1/10 hour per dollar earned.

theorem sam_earnings_difference
  (hours_per_dollar : ℝ := 1 / 10)
  (E1 : ℝ := 200)  -- Earnings in the first month are $200.
  (total_hours : ℝ := 55)  -- Total hours he worked over two months.
  (total_hourly_earning : ℝ := total_hours / hours_per_dollar)  -- Total earnings over two months.
  (E2 : ℝ := total_hourly_earning - E1) :  -- Earnings in the second month.

  E2 - E1 = 150 :=  -- The difference in earnings between the second month and the first month is $150.
sorry

end sam_earnings_difference_l1357_135783


namespace marbles_given_to_juan_l1357_135709

def initial : ℕ := 776
def left : ℕ := 593

theorem marbles_given_to_juan : initial - left = 183 :=
by sorry

end marbles_given_to_juan_l1357_135709


namespace simplify_and_substitute_l1357_135717

theorem simplify_and_substitute (x : ℝ) (h1 : x ≠ 1) (h3 : x ≠ 3) : 
  ((1 - (2 / (x - 1))) * ((x^2 - x) / (x^2 - 6*x + 9))) = (x / (x - 3)) ∧ 
  (2 / (2 - 3)) = -2 := by
  sorry

end simplify_and_substitute_l1357_135717


namespace find_number_of_math_problems_l1357_135777

-- Define the number of social studies problems
def social_studies_problems : ℕ := 6

-- Define the number of science problems
def science_problems : ℕ := 10

-- Define the time to solve each type of problem in minutes
def time_per_math_problem : ℝ := 2
def time_per_social_studies_problem : ℝ := 0.5
def time_per_science_problem : ℝ := 1.5

-- Define the total time to solve all problems in minutes
def total_time : ℝ := 48

-- Define the theorem to find the number of math problems
theorem find_number_of_math_problems (M : ℕ) :
  time_per_math_problem * M + time_per_social_studies_problem * social_studies_problems + time_per_science_problem * science_problems = total_time → 
  M = 15 :=
by {
  -- proof is not required to be written, hence expressing the unresolved part
  sorry
}

end find_number_of_math_problems_l1357_135777


namespace age_difference_l1357_135737

variable (A B C : ℕ)

-- Conditions: C is 11 years younger than A
axiom h1 : C = A - 11

-- Statement: Prove the difference (A + B) - (B + C) is 11
theorem age_difference : (A + B) - (B + C) = 11 := by
  sorry

end age_difference_l1357_135737


namespace find_positive_number_l1357_135789

noncomputable def solve_number (x : ℝ) : Prop :=
  (2/3 * x = 64/216 * (1/x)) ∧ (x > 0)

theorem find_positive_number (x : ℝ) : solve_number x → x = (2/9) * Real.sqrt 3 :=
  by
  sorry

end find_positive_number_l1357_135789


namespace eating_time_proof_l1357_135752

noncomputable def combined_eating_time (time_fat time_thin weight : ℝ) : ℝ :=
  let rate_fat := 1 / time_fat
  let rate_thin := 1 / time_thin
  let combined_rate := rate_fat + rate_thin
  weight / combined_rate

theorem eating_time_proof :
  let time_fat := 12
  let time_thin := 40
  let weight := 5
  combined_eating_time time_fat time_thin weight = (600 / 13) :=
by
  -- placeholder for the proof
  sorry

end eating_time_proof_l1357_135752


namespace prob_two_girls_is_one_fourth_l1357_135749

-- Define the probability of giving birth to a girl
def prob_girl : ℚ := 1 / 2

-- Define the probability of having two girls
def prob_two_girls : ℚ := prob_girl * prob_girl

-- Theorem statement: The probability of having two girls is 1/4
theorem prob_two_girls_is_one_fourth : prob_two_girls = 1 / 4 :=
by sorry

end prob_two_girls_is_one_fourth_l1357_135749


namespace total_runs_of_a_b_c_l1357_135712

/-- Suppose a, b, and c are the runs scored by three players in a cricket match. The ratios of the runs are given as a : b = 1 : 3 and b : c = 1 : 5. Additionally, c scored 75 runs. Prove that the total runs scored by all of them is 95. -/
theorem total_runs_of_a_b_c (a b c : ℕ) (h1 : a * 3 = b) (h2 : b * 5 = c) (h3 : c = 75) : a + b + c = 95 := 
by sorry

end total_runs_of_a_b_c_l1357_135712


namespace consumer_installment_credit_l1357_135722

theorem consumer_installment_credit (A C : ℝ) (h1 : A = 0.36 * C) (h2 : 35 = (1 / 3) * A) :
  C = 291.67 :=
by 
  -- The proof should go here
  sorry

end consumer_installment_credit_l1357_135722


namespace hypotenuse_of_454590_triangle_l1357_135755

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end hypotenuse_of_454590_triangle_l1357_135755


namespace motorcycles_in_anytown_l1357_135729

variable (t s m : ℕ) -- t: number of trucks, s: number of sedans, m: number of motorcycles
variable (r_trucks r_sedans r_motorcycles : ℕ) -- r_trucks : truck ratio, r_sedans : sedan ratio, r_motorcycles : motorcycle ratio
variable (n_sedans : ℕ) -- n_sedans: number of sedans

theorem motorcycles_in_anytown
  (h1 : r_trucks = 3) -- ratio of trucks
  (h2 : r_sedans = 7) -- ratio of sedans
  (h3 : r_motorcycles = 2) -- ratio of motorcycles
  (h4 : s = 9100) -- number of sedans
  (h5 : s = (r_sedans * n_sedans)) -- relationship between sedans and parts
  (h6 : t = (r_trucks * n_sedans)) -- relationship between trucks and parts
  (h7 : m = (r_motorcycles * n_sedans)) -- relationship between motorcycles and parts
  : m = 2600 := by
    sorry

end motorcycles_in_anytown_l1357_135729


namespace arc_length_of_sector_l1357_135740

theorem arc_length_of_sector 
  (R : ℝ) (θ : ℝ) (hR : R = Real.pi) (hθ : θ = 2 * Real.pi / 3) : 
  (R * θ = 2 * Real.pi^2 / 3) := 
by
  rw [hR, hθ]
  sorry

end arc_length_of_sector_l1357_135740


namespace max_n_for_regular_polygons_l1357_135724

theorem max_n_for_regular_polygons (m n : ℕ) (h1 : m ≥ n) (h2 : n ≥ 3)
  (h3 : (7 * (m - 2) * n) = (8 * (n - 2) * m)) : 
  n ≤ 112 ∧ (∃ m, (14 * n = (n - 16) * m)) :=
by
  sorry

end max_n_for_regular_polygons_l1357_135724


namespace cost_of_four_stamps_l1357_135793

theorem cost_of_four_stamps (cost_one_stamp : ℝ) (h : cost_one_stamp = 0.34) : 4 * cost_one_stamp = 1.36 := 
by
  rw [h]
  norm_num

end cost_of_four_stamps_l1357_135793


namespace number_of_ways_to_fill_l1357_135750

-- Definitions and conditions
def triangular_array (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the triangular array structure
  sorry 

def sum_based (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the sum-based condition
  sorry 

def valid_filling (x : Fin 13 → ℕ) :=
  (∀ i, x i = 0 ∨ x i = 1) ∧
  (x 0 + x 12) % 5 = 0

theorem number_of_ways_to_fill (x : Fin 13 → ℕ) :
  triangular_array 13 1 → sum_based 13 1 →
  valid_filling x → 
  (∃ (count : ℕ), count = 4096) :=
sorry

end number_of_ways_to_fill_l1357_135750


namespace select_4_people_arrangement_3_day_new_year_l1357_135768

def select_4_people_arrangement (n k : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.factorial (n - 2) / Nat.factorial 2

theorem select_4_people_arrangement_3_day_new_year :
  select_4_people_arrangement 7 4 = 420 :=
by
  -- proof to be filled in
  sorry

end select_4_people_arrangement_3_day_new_year_l1357_135768


namespace different_meal_combinations_l1357_135786

-- Defining the conditions explicitly
def items_on_menu : ℕ := 12

-- A function representing possible combinations of choices for Yann and Camille
def meal_combinations (menu_items : ℕ) : ℕ :=
  menu_items * (menu_items - 1)

-- Theorem stating that given 12 items on the menu, the different combinations of meals is 132
theorem different_meal_combinations : meal_combinations items_on_menu = 132 :=
by
  sorry

end different_meal_combinations_l1357_135786


namespace min_value_abs_plus_2023_proof_l1357_135742

noncomputable def min_value_abs_plus_2023 (a : ℚ) : Prop :=
  |a| + 2023 ≥ 2023

theorem min_value_abs_plus_2023_proof (a : ℚ) : min_value_abs_plus_2023 a :=
  by
  sorry

end min_value_abs_plus_2023_proof_l1357_135742


namespace symmetric_sum_l1357_135791

theorem symmetric_sum (m n : ℤ) (hA : n = 3) (hB : m = -2) : m + n = 1 :=
by
  rw [hA, hB]
  exact rfl

end symmetric_sum_l1357_135791


namespace calculate_cherry_pies_l1357_135796

-- Definitions for the conditions
def total_pies : ℕ := 40
def ratio_parts_apple : ℕ := 2
def ratio_parts_blueberry : ℕ := 5
def ratio_parts_cherry : ℕ := 3
def total_ratio_parts := ratio_parts_apple + ratio_parts_blueberry + ratio_parts_cherry

-- Calculating the number of pies per part and then the number of cherry pies
def pies_per_part : ℕ := total_pies / total_ratio_parts
def cherry_pies : ℕ := ratio_parts_cherry * pies_per_part

-- Proof statement
theorem calculate_cherry_pies : cherry_pies = 12 :=
by
  -- Lean proof goes here
  sorry

end calculate_cherry_pies_l1357_135796


namespace range_of_a_intersection_l1357_135730

theorem range_of_a_intersection (a : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x - 2 * k + 2 ∧ y = a * x^2 - 2 * a * x - 3 * a) ↔ (a ≤ -2/3 ∨ a > 0) := by
  sorry

end range_of_a_intersection_l1357_135730


namespace expected_number_of_digits_on_fair_icosahedral_die_l1357_135773

noncomputable def expected_digits_fair_icosahedral_die : ℚ :=
  let prob_one_digit := (9 : ℚ) / 20
  let prob_two_digits := (11 : ℚ) / 20
  (prob_one_digit * 1) + (prob_two_digits * 2)

theorem expected_number_of_digits_on_fair_icosahedral_die : expected_digits_fair_icosahedral_die = 1.55 := by
  sorry

end expected_number_of_digits_on_fair_icosahedral_die_l1357_135773


namespace total_spent_is_49_l1357_135782

-- Define the prices of items
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define Paula's purchases
def paula_bracelets := 3
def paula_keychains := 2
def paula_coloring_book := 1
def paula_stickers := 4

-- Define Olive's purchases
def olive_bracelets := 2
def olive_coloring_book := 1
def olive_toy_car := 1
def olive_stickers := 3

-- Calculate total expenses
def paula_total := paula_bracelets * price_bracelet + paula_keychains * price_keychain + paula_coloring_book * price_coloring_book + paula_stickers * price_sticker
def olive_total := olive_coloring_book * price_coloring_book + olive_bracelets * price_bracelet + olive_toy_car * price_toy_car + olive_stickers * price_sticker
def total_expense := paula_total + olive_total

-- Prove the total expenses amount to $49
theorem total_spent_is_49 : total_expense = 49 :=
by
  have : paula_total = (3 * 4) + (2 * 5) + (1 * 3) + (4 * 1) := rfl
  have : olive_total = (1 * 3) + (2 * 4) + (1 *6) + (3 * 1) := rfl
  have : paula_total = 29 := rfl
  have : olive_total = 20 := rfl
  have : total_expense = 29 + 20 := rfl
  exact rfl

end total_spent_is_49_l1357_135782


namespace find_a_l1357_135757

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}) (h1 : 1 ∈ A) : a = -1 :=
by
  sorry

end find_a_l1357_135757


namespace quadratic_roots_eq_l1357_135790

theorem quadratic_roots_eq (a : ℝ) (b : ℝ) :
  (∀ x, (2 * x^2 - 3 * x - 8 = 0) → 
         ((x + 3)^2 + a * (x + 3) + b = 0)) → 
  b = 9.5 :=
by
  sorry

end quadratic_roots_eq_l1357_135790
