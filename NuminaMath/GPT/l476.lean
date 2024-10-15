import Mathlib

namespace NUMINAMATH_GPT_midpoint_one_sixth_one_ninth_l476_47605

theorem midpoint_one_sixth_one_ninth : (1 / 6 + 1 / 9) / 2 = 5 / 36 := by
  sorry

end NUMINAMATH_GPT_midpoint_one_sixth_one_ninth_l476_47605


namespace NUMINAMATH_GPT_sequence_divisibility_l476_47608

theorem sequence_divisibility (a b c : ℤ) (u v : ℕ → ℤ) (N : ℕ)
  (hu0 : u 0 = 1) (hu1 : u 1 = 1)
  (hu : ∀ n ≥ 2, u n = 2 * u (n - 1) - 3 * u (n - 2))
  (hv0 : v 0 = a) (hv1 : v 1 = b) (hv2 : v 2 = c)
  (hv : ∀ n ≥ 3, v n = v (n - 1) - 3 * v (n - 2) + 27 * v (n - 3))
  (hdiv : ∀ n ≥ N, u n ∣ v n) : 3 * a = 2 * b + c :=
by
  sorry

end NUMINAMATH_GPT_sequence_divisibility_l476_47608


namespace NUMINAMATH_GPT_dave_apps_files_difference_l476_47688

theorem dave_apps_files_difference :
  let initial_apps := 15
  let initial_files := 24
  let final_apps := 21
  let final_files := 4
  final_apps - final_files = 17 :=
by
  intros
  sorry

end NUMINAMATH_GPT_dave_apps_files_difference_l476_47688


namespace NUMINAMATH_GPT_infinite_k_values_l476_47699

theorem infinite_k_values (k : ℕ) : (∃ k, ∀ (a b c : ℕ),
  (a = 64 ∧ b ≥ 0 ∧ c = 0 ∧ k = 2^a * 3^b * 5^c) ↔
  Nat.lcm (Nat.lcm (2^8) (2^24 * 3^12)) k = 2^64) →
  ∃ (b : ℕ), true :=
by
  sorry

end NUMINAMATH_GPT_infinite_k_values_l476_47699


namespace NUMINAMATH_GPT_average_age_across_rooms_l476_47606

theorem average_age_across_rooms :
  let room_a_people := 8
  let room_a_average_age := 35
  let room_b_people := 5
  let room_b_average_age := 30
  let room_c_people := 7
  let room_c_average_age := 25
  let total_people := room_a_people + room_b_people + room_c_people
  let total_age := (room_a_people * room_a_average_age) + (room_b_people * room_b_average_age) + (room_c_people * room_c_average_age)
  let average_age := total_age / total_people
  average_age = 30.25 := by
{
  sorry
}

end NUMINAMATH_GPT_average_age_across_rooms_l476_47606


namespace NUMINAMATH_GPT_parabola_passing_through_4_neg2_l476_47650

theorem parabola_passing_through_4_neg2 :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ y = -2 ∧ x = 4 ∧ (y^2 = x)) ∨
  (∃ p : ℝ, x^2 = -2 * p * y ∧ y = -2 ∧ x = 4 ∧ (x^2 = -8 * y)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_passing_through_4_neg2_l476_47650


namespace NUMINAMATH_GPT_pugs_working_together_l476_47693

theorem pugs_working_together (P : ℕ) (H1 : P * 45 = 15 * 12) : P = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_pugs_working_together_l476_47693


namespace NUMINAMATH_GPT_education_expenses_l476_47609

noncomputable def totalSalary (savings : ℝ) (savingsPercentage : ℝ) : ℝ :=
  savings / savingsPercentage

def totalExpenses (rent milk groceries petrol misc : ℝ) : ℝ :=
  rent + milk + groceries + petrol + misc

def amountSpentOnEducation (totalSalary totalExpenses savings : ℝ) : ℝ :=
  totalSalary - (totalExpenses + savings)

theorem education_expenses :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let petrol := 2000
  let misc := 700
  let savings := 1800
  let savingsPercentage := 0.10
  amountSpentOnEducation (totalSalary savings savingsPercentage) 
                          (totalExpenses rent milk groceries petrol misc) 
                          savings = 2500 :=
by
  sorry

end NUMINAMATH_GPT_education_expenses_l476_47609


namespace NUMINAMATH_GPT_games_went_this_year_l476_47613

theorem games_went_this_year (t l : ℕ) (h1 : t = 13) (h2 : l = 9) : (t - l = 4) :=
by
  sorry

end NUMINAMATH_GPT_games_went_this_year_l476_47613


namespace NUMINAMATH_GPT_metal_contest_winner_l476_47610

theorem metal_contest_winner (x y : ℕ) (hx : 95 * x + 74 * y = 2831) : x = 15 ∧ y = 19 ∧ 95 * 15 > 74 * 19 := by
  sorry

end NUMINAMATH_GPT_metal_contest_winner_l476_47610


namespace NUMINAMATH_GPT_percent_increase_output_l476_47685

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end NUMINAMATH_GPT_percent_increase_output_l476_47685


namespace NUMINAMATH_GPT_smallest_c_minus_a_l476_47691

theorem smallest_c_minus_a (a b c : ℕ) (h1 : a * b * c = 720) (h2 : a < b) (h3 : b < c) : c - a ≥ 24 :=
sorry

end NUMINAMATH_GPT_smallest_c_minus_a_l476_47691


namespace NUMINAMATH_GPT_problem_sol_l476_47669

-- Defining the operations as given
def operation_hash (a b c : ℤ) : ℤ := 4 * a ^ 3 + 4 * b ^ 3 + 8 * a ^ 2 * b + c
def operation_star (a b d : ℤ) : ℤ := 2 * a ^ 2 - 3 * b ^ 2 + d ^ 3

-- Main theorem statement
theorem problem_sol (a b x c d : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (hc : c > 0) (hd : d > 0) 
  (h3 : operation_hash a x c = 250)
  (h4 : operation_star a b d + x = 50) :
  False := sorry

end NUMINAMATH_GPT_problem_sol_l476_47669


namespace NUMINAMATH_GPT_total_spent_l476_47690

theorem total_spent (cost_per_deck : ℕ) (decks_frank : ℕ) (decks_friend : ℕ) (total : ℕ) : 
  cost_per_deck = 7 → 
  decks_frank = 3 → 
  decks_friend = 2 → 
  total = (decks_frank * cost_per_deck) + (decks_friend * cost_per_deck) → 
  total = 35 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_l476_47690


namespace NUMINAMATH_GPT_range_of_a_l476_47652

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.exp x - 2 * a * x - a ^ 2 + 3

theorem range_of_a (h : ∀ x, x ≥ 0 → f x a - x ^ 2 ≥ 0) :
  -Real.sqrt 5 ≤ a ∧ a ≤ 3 - Real.log 3 := sorry

end NUMINAMATH_GPT_range_of_a_l476_47652


namespace NUMINAMATH_GPT_intersection_P_Q_l476_47666

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l476_47666


namespace NUMINAMATH_GPT_luke_clothing_distribution_l476_47647

theorem luke_clothing_distribution (total_clothing: ℕ) (first_load: ℕ) (num_loads: ℕ) 
  (remaining_clothing : total_clothing - first_load = 30)
  (equal_load_per_small_load: (total_clothing - first_load) / num_loads = 6) : 
  total_clothing = 47 ∧ first_load = 17 ∧ num_loads = 5 :=
by
  have h1 : total_clothing - first_load = 30 := remaining_clothing
  have h2 : (total_clothing - first_load) / num_loads = 6 := equal_load_per_small_load
  sorry

end NUMINAMATH_GPT_luke_clothing_distribution_l476_47647


namespace NUMINAMATH_GPT_fraction_simplification_l476_47674

theorem fraction_simplification 
  (a b c : ℝ)
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0) 
  (h₃ : a^2 + b^2 + c^2 ≠ 0) :
  (a^2 * b^2 + 2 * a^2 * b * c + a^2 * c^2 - b^4) / (a^4 - b^2 * c^2 + 2 * a * b * c^2 + c^4) =
  ((a * b + a * c + b^2) * (a * b + a * c - b^2)) / ((a^2 + b^2 - c^2) * (a^2 - b^2 + c^2)) :=
sorry

end NUMINAMATH_GPT_fraction_simplification_l476_47674


namespace NUMINAMATH_GPT_trapezoid_longer_side_length_l476_47621

theorem trapezoid_longer_side_length (x : ℝ) (h₁ : 4 = 2*2) (h₂ : ∃ AP DQ O : ℝ, ∀ (S : ℝ), 
  S = (1/2) * (x + 2) * 1 → S = 2) : 
  x = 2 :=
by sorry

end NUMINAMATH_GPT_trapezoid_longer_side_length_l476_47621


namespace NUMINAMATH_GPT_least_number_divisible_by_digits_and_5_l476_47659

/-- Define a predicate to check if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10 % 10, n / 10 % 10, n % 10]
  ∀ d ∈ digits, d ≠ 0 → n % d = 0

/-- Define the main theorem stating the least four-digit number divisible by 5 and each of its digits is 1425 -/
theorem least_number_divisible_by_digits_and_5 
  (n : ℕ) (hn : 1000 ≤ n ∧ n < 10000)
  (hd : (∀ i j : ℕ, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)))
  (hdiv5 : n % 5 = 0)
  (hdiv_digits : divisible_by_digits n) 
  : n = 1425 :=
sorry

end NUMINAMATH_GPT_least_number_divisible_by_digits_and_5_l476_47659


namespace NUMINAMATH_GPT_opposite_of_2023_l476_47629

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l476_47629


namespace NUMINAMATH_GPT_eliza_irons_dress_in_20_minutes_l476_47645

def eliza_iron_time : Prop :=
∃ d : ℕ, 
  (d ≠ 0 ∧  -- To avoid division by zero
  8 + 180 / d = 17 ∧
  d = 20)

theorem eliza_irons_dress_in_20_minutes : eliza_iron_time :=
sorry

end NUMINAMATH_GPT_eliza_irons_dress_in_20_minutes_l476_47645


namespace NUMINAMATH_GPT_three_digit_2C4_not_multiple_of_5_l476_47644

theorem three_digit_2C4_not_multiple_of_5 : ∀ C : ℕ, C < 10 → ¬(∃ n : ℕ, 2 * 100 + C * 10 + 4 = 5 * n) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_2C4_not_multiple_of_5_l476_47644


namespace NUMINAMATH_GPT_min_value_of_squares_l476_47679

theorem min_value_of_squares (x y z : ℝ) (h : x + y + z = 1) : x^2 + y^2 + z^2 ≥ 1 / 3 := sorry

end NUMINAMATH_GPT_min_value_of_squares_l476_47679


namespace NUMINAMATH_GPT_chris_wins_l476_47631

noncomputable def chris_heads : ℚ := 1 / 4
noncomputable def drew_heads : ℚ := 1 / 3
noncomputable def both_tails : ℚ := (1 - chris_heads) * (1 - drew_heads)

/-- The probability that Chris wins comparing with relatively prime -/
theorem chris_wins (p q : ℕ) (hpq : Nat.Coprime p q) (hq0 : q ≠ 0) :
  (chris_heads * (1 + both_tails)) = (p : ℚ) / q ∧ (q - p = 1) :=
sorry

end NUMINAMATH_GPT_chris_wins_l476_47631


namespace NUMINAMATH_GPT_charles_ate_no_bananas_l476_47646

theorem charles_ate_no_bananas (W C B : ℝ) (h1 : W = 48) (h2 : C = 35) (h3 : W + C = 83) : B = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_charles_ate_no_bananas_l476_47646


namespace NUMINAMATH_GPT_no_int_solutions_for_quadratics_l476_47642

theorem no_int_solutions_for_quadratics :
  ¬ ∃ a b c : ℤ, (∃ x1 x2 : ℤ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧
                (∃ y1 y2 : ℤ, (a + 1) * y1^2 + (b + 1) * y1 + (c + 1) = 0 ∧ 
                              (a + 1) * y2^2 + (b + 1) * y2 + (c + 1) = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_int_solutions_for_quadratics_l476_47642


namespace NUMINAMATH_GPT_difference_is_693_l476_47632

noncomputable def one_tenth_of_seven_thousand : ℕ := 1 / 10 * 7000
noncomputable def one_tenth_percent_of_seven_thousand : ℕ := (1 / 10 / 100) * 7000
noncomputable def difference : ℕ := one_tenth_of_seven_thousand - one_tenth_percent_of_seven_thousand

theorem difference_is_693 :
  difference = 693 :=
by
  sorry

end NUMINAMATH_GPT_difference_is_693_l476_47632


namespace NUMINAMATH_GPT_articles_produced_l476_47696

theorem articles_produced (x y z w : ℕ) :
  (x ≠ 0) → (y ≠ 0) → (z ≠ 0) → (w ≠ 0) →
  ((x * x * x * (1 / x^2) = x) →
  y * z * w * (1 / x^2) = y * z * w / x^2) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_articles_produced_l476_47696


namespace NUMINAMATH_GPT_john_horizontal_distance_l476_47626

-- Define the conditions and the question
def elevation_initial : ℕ := 100
def elevation_final : ℕ := 1450
def vertical_to_horizontal_ratio (v h : ℕ) : Prop := v * 2 = h

-- Define the proof problem: the horizontal distance John moves
theorem john_horizontal_distance : ∃ h, vertical_to_horizontal_ratio (elevation_final - elevation_initial) h ∧ h = 2700 := 
by 
  sorry

end NUMINAMATH_GPT_john_horizontal_distance_l476_47626


namespace NUMINAMATH_GPT_factorize_x_cubed_minus_9x_l476_47682

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end NUMINAMATH_GPT_factorize_x_cubed_minus_9x_l476_47682


namespace NUMINAMATH_GPT_area_of_region_l476_47686

theorem area_of_region (x y : ℝ) : |4 * x - 24| + |3 * y + 10| ≤ 6 → ∃ A : ℝ, A = 12 :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_l476_47686


namespace NUMINAMATH_GPT_log_eq_condition_pq_l476_47612

theorem log_eq_condition_pq :
  ∀ (p q : ℝ), p > 0 → q > 0 → (Real.log p + Real.log q = Real.log (2 * p + q)) → p = 3 ∧ q = 3 :=
by
  intros p q hp hq hlog
  sorry

end NUMINAMATH_GPT_log_eq_condition_pq_l476_47612


namespace NUMINAMATH_GPT_cricket_team_members_l476_47655

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℕ := 27)
  (wk_age : ℕ := captain_age + 1)
  (total_avg_age : ℕ := 23)
  (remaining_avg_age : ℕ := total_avg_age - 1)
  (total_age : ℕ := n * total_avg_age)
  (captain_and_wk_age : ℕ := captain_age + wk_age)
  (remaining_age : ℕ := (n - 2) * remaining_avg_age) : n = 11 := 
by
  sorry

end NUMINAMATH_GPT_cricket_team_members_l476_47655


namespace NUMINAMATH_GPT_systematic_sampling_selects_616_l476_47627

theorem systematic_sampling_selects_616 (n : ℕ) (h₁ : n = 1000) (h₂ : (∀ i : ℕ, ∃ j : ℕ, i = 46 + j * 10) → True) :
  (∃ m : ℕ, m = 616) :=
  by
  sorry

end NUMINAMATH_GPT_systematic_sampling_selects_616_l476_47627


namespace NUMINAMATH_GPT_family_age_problem_l476_47622

theorem family_age_problem (T y : ℕ)
  (h1 : T = 5 * 17)
  (h2 : (T + 5 * y + 2) = 6 * 17)
  : y = 3 := by
  sorry

end NUMINAMATH_GPT_family_age_problem_l476_47622


namespace NUMINAMATH_GPT_equal_chord_segments_l476_47656

theorem equal_chord_segments 
  (a x y : ℝ) 
  (AM CM : ℝ → ℝ → Prop) 
  (AB CD : ℝ → Prop)
  (intersect_chords_theorem : AM x (a - x) = CM y (a - y)) :
  x = y ∨ x = a - y :=
by
  sorry

end NUMINAMATH_GPT_equal_chord_segments_l476_47656


namespace NUMINAMATH_GPT_precision_of_rounded_value_l476_47667

-- Definition of the original problem in Lean 4
def original_value := 27390000000

-- Proof statement to check the precision of the rounded value to the million place
theorem precision_of_rounded_value :
  (original_value % 1000000 = 0) :=
sorry

end NUMINAMATH_GPT_precision_of_rounded_value_l476_47667


namespace NUMINAMATH_GPT_part_a_part_b_l476_47649

noncomputable def sequence_a (n : ℕ) : ℝ :=
  Real.cos (10^n * Real.pi / 180)

theorem part_a (h : 100 > 2) : sequence_a 100 > 0 := by
  sorry

theorem part_b : |sequence_a 100| < 0.18 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l476_47649


namespace NUMINAMATH_GPT_linear_coefficient_l476_47641

theorem linear_coefficient (m x : ℝ) (h1 : (m - 3) * x ^ (m^2 - 2 * m - 1) - m * x + 6 = 0) (h2 : (m^2 - 2 * m - 1 = 2)) (h3 : m ≠ 3) : 
  ∃ a b c : ℝ, a * x ^ 2 + b * x + c = 0 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_linear_coefficient_l476_47641


namespace NUMINAMATH_GPT_interest_rate_10_percent_l476_47615

-- Definitions for the problem
variables (P : ℝ) (R : ℝ) (T : ℝ)

-- Condition that the money doubles in 10 years on simple interest
def money_doubles_in_10_years (P R : ℝ) : Prop :=
  P = (P * R * 10) / 100

-- Statement that R is 10% if the money doubles in 10 years
theorem interest_rate_10_percent {P : ℝ} (h : money_doubles_in_10_years P R) : R = 10 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_10_percent_l476_47615


namespace NUMINAMATH_GPT_log_prime_factor_inequality_l476_47653

open Real

noncomputable def num_prime_factors (n : ℕ) : ℕ := sorry 

theorem log_prime_factor_inequality (n : ℕ) (h : 0 < n) : 
  log n ≥ num_prime_factors n * log 2 := 
sorry

end NUMINAMATH_GPT_log_prime_factor_inequality_l476_47653


namespace NUMINAMATH_GPT_find_x_l476_47602

theorem find_x (x : ℝ) (h : x - 1/10 = x / 10) : x = 1 / 9 := 
  sorry

end NUMINAMATH_GPT_find_x_l476_47602


namespace NUMINAMATH_GPT_range_of_m_for_false_p_and_q_l476_47636

theorem range_of_m_for_false_p_and_q (m : ℝ) :
  (¬ (∀ x y : ℝ, (x^2 / (1 - m) + y^2 / (m + 2) = 1) ∧ ∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (2 - m) = 1))) →
  (m ≤ 1 ∨ m ≥ 2) :=
sorry

end NUMINAMATH_GPT_range_of_m_for_false_p_and_q_l476_47636


namespace NUMINAMATH_GPT_number_of_cutlery_pieces_added_l476_47620

-- Define the initial conditions
def forks_initial := 6
def knives_initial := forks_initial + 9
def spoons_initial := 2 * knives_initial
def teaspoons_initial := forks_initial / 2
def total_initial_cutlery := forks_initial + knives_initial + spoons_initial + teaspoons_initial
def total_final_cutlery := 62

-- Define the total number of cutlery pieces added
def cutlery_added := total_final_cutlery - total_initial_cutlery

-- Define the theorem to prove
theorem number_of_cutlery_pieces_added : cutlery_added = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_cutlery_pieces_added_l476_47620


namespace NUMINAMATH_GPT_sum_of_roots_l476_47657

theorem sum_of_roots : 
  let a := 1
  let b := 2001
  let c := -2002
  ∀ x y: ℝ, (x^2 + b*x + c = 0) ∧ (y^2 + b*y + c = 0) -> (x + y = -b) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l476_47657


namespace NUMINAMATH_GPT_rhombus_area_2sqrt2_l476_47675

structure Rhombus (α : Type _) :=
  (side_length : ℝ)
  (angle : ℝ)

theorem rhombus_area_2sqrt2 (R : Rhombus ℝ) (h_side : R.side_length = 2) (h_angle : R.angle = 45) :
  ∃ A : ℝ, A = 2 * Real.sqrt 2 :=
by
  let A := 2 * Real.sqrt 2
  existsi A
  sorry

end NUMINAMATH_GPT_rhombus_area_2sqrt2_l476_47675


namespace NUMINAMATH_GPT_flowers_bees_butterflies_comparison_l476_47637

def num_flowers : ℕ := 12
def num_bees : ℕ := 7
def num_butterflies : ℕ := 4
def difference_flowers_bees : ℕ := num_flowers - num_bees

theorem flowers_bees_butterflies_comparison :
  difference_flowers_bees - num_butterflies = 1 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_flowers_bees_butterflies_comparison_l476_47637


namespace NUMINAMATH_GPT_remainder_18_l476_47683

theorem remainder_18 (x : ℤ) (k : ℤ) (h : x = 62 * k + 7) :
  (x + 11) % 31 = 18 :=
by
  sorry

end NUMINAMATH_GPT_remainder_18_l476_47683


namespace NUMINAMATH_GPT_sin_160_eq_sin_20_l476_47681

theorem sin_160_eq_sin_20 : Real.sin (160 * Real.pi / 180) = Real.sin (20 * Real.pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_sin_160_eq_sin_20_l476_47681


namespace NUMINAMATH_GPT_math_problem_l476_47639

def cond1 (R r a b c p : ℝ) : Prop := R * r = (a * b * c) / (4 * p)
def cond2 (a b c p : ℝ) : Prop := a * b * c ≤ 8 * p^3
def cond3 (a b c p : ℝ) : Prop := p^2 ≤ (3 * (a^2 + b^2 + c^2)) / 4
def cond4 (m_a m_b m_c R : ℝ) : Prop := m_a^2 + m_b^2 + m_c^2 ≤ (27 * R^2) / 4

theorem math_problem (R r a b c p m_a m_b m_c : ℝ) 
  (h1 : cond1 R r a b c p)
  (h2 : cond2 a b c p)
  (h3 : cond3 a b c p)
  (h4 : cond4 m_a m_b m_c R) : 
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ (27 * R^2) / 2 :=
by 
  sorry

end NUMINAMATH_GPT_math_problem_l476_47639


namespace NUMINAMATH_GPT_a_perpendicular_to_a_minus_b_l476_47677

def vector := ℝ × ℝ

def dot_product (u v : vector) : ℝ := u.1 * v.1 + u.2 * v.2

def a : vector := (-2, 1)
def b : vector := (-1, 3)

def a_minus_b : vector := (a.1 - b.1, a.2 - b.2) 

theorem a_perpendicular_to_a_minus_b : dot_product a a_minus_b = 0 := by
  sorry

end NUMINAMATH_GPT_a_perpendicular_to_a_minus_b_l476_47677


namespace NUMINAMATH_GPT_simplify_fraction_expression_l476_47661

theorem simplify_fraction_expression :
  5 * (12 / 7) * (49 / (-60)) = -7 := 
sorry

end NUMINAMATH_GPT_simplify_fraction_expression_l476_47661


namespace NUMINAMATH_GPT_tan_A_area_triangle_ABC_l476_47624
open Real

-- Define the given conditions
def conditions (A : ℝ) (AC AB : ℝ) : Prop :=
  (sin A + cos A = sqrt 2 / 2) ∧ (AC = 2) ∧ (AB = 3)

-- State the first proof problem for tan A
theorem tan_A (A : ℝ) (hcond : conditions A 2 3) : tan A = -(2 + sqrt 3) := 
by 
  -- sorry for the proof placeholder
  sorry

-- State the second proof problem for the area of triangle ABC
theorem area_triangle_ABC (A B C : ℝ) (C_eq : C = 90) 
  (hcond : conditions A 2 3)
  (hBC : BC = sqrt ((AC^2) + (AB^2) - 2 * AC * AB * cos B)) : 
  (1/2) * AC * AB * sin A = (3 / 4) * (sqrt 6 + sqrt 2) := 
by 
  -- sorry for the proof placeholder
  sorry

end NUMINAMATH_GPT_tan_A_area_triangle_ABC_l476_47624


namespace NUMINAMATH_GPT_initial_amount_proof_l476_47635

noncomputable def initial_amount (A B : ℝ) : ℝ :=
  A + B

theorem initial_amount_proof :
  ∃ (A B : ℝ), B = 4000.0000000000005 ∧ 
               (A * 0.15 * 2 = B * 0.18 * 2 + 360) ∧ 
               initial_amount A B = 10000.000000000002 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_proof_l476_47635


namespace NUMINAMATH_GPT_jacob_age_l476_47664

/- Conditions:
1. Rehana's current age is 25.
2. In five years, Rehana's age is three times Phoebe's age.
3. Jacob's current age is 3/5 of Phoebe's current age.

Prove that Jacob's current age is 3.
-/

theorem jacob_age (R P J : ℕ) (h1 : R = 25) (h2 : R + 5 = 3 * (P + 5)) (h3 : J = 3 / 5 * P) : J = 3 :=
by
  sorry

end NUMINAMATH_GPT_jacob_age_l476_47664


namespace NUMINAMATH_GPT_solution_set_inequality_l476_47648

theorem solution_set_inequality (x : ℝ) : (x ≠ 1) → 
  ((x - 3) * (x + 2) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l476_47648


namespace NUMINAMATH_GPT_neg_p_exists_x_l476_47663

-- Let p be the proposition: For all x in ℝ, x^2 - 3x + 3 > 0
def p : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 > 0

-- Prove that the negation of p implies that there exists some x in ℝ such that x^2 - 3x + 3 ≤ 0
theorem neg_p_exists_x : ¬p ↔ ∃ x : ℝ, x^2 - 3 * x + 3 ≤ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_neg_p_exists_x_l476_47663


namespace NUMINAMATH_GPT_pieces_of_paper_picked_up_l476_47654

theorem pieces_of_paper_picked_up (Olivia : ℕ) (Edward : ℕ) (h₁ : Olivia = 16) (h₂ : Edward = 3) : Olivia + Edward = 19 :=
by
  sorry

end NUMINAMATH_GPT_pieces_of_paper_picked_up_l476_47654


namespace NUMINAMATH_GPT_lower_limit_of_range_with_multiples_l476_47658

theorem lower_limit_of_range_with_multiples (n : ℕ) (h : 2000 - n ≥ 198 * 10 ∧ n % 10 = 0 ∧ n + 1980 ≤ 2000) :
  n = 30 :=
by
  sorry

end NUMINAMATH_GPT_lower_limit_of_range_with_multiples_l476_47658


namespace NUMINAMATH_GPT_find_f_2017_l476_47634

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_2017 (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_func_eq : ∀ x : ℝ, f (x + 3) * f x = -1)
  (h_val : f (-1) = 2) :
  f 2017 = -2 := sorry

end NUMINAMATH_GPT_find_f_2017_l476_47634


namespace NUMINAMATH_GPT_sin_cos_identity_l476_47672

theorem sin_cos_identity :
  (Real.sin (75 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) -
   Real.sin (15 * Real.pi / 180) * Real.sin (150 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l476_47672


namespace NUMINAMATH_GPT_waiter_tables_l476_47694

/-
Problem:
A waiter had 22 customers in his section.
14 of them left.
The remaining customers were seated at tables with 4 people per table.
Prove the number of tables is 2.
-/

theorem waiter_tables:
  ∃ (tables : ℤ), 
    (∀ (customers_initial customers_remaining people_per_table tables_calculated : ℤ), 
      customers_initial = 22 →
      customers_remaining = customers_initial - 14 →
      people_per_table = 4 →
      tables_calculated = customers_remaining / people_per_table →
      tables = tables_calculated) →
    tables = 2 :=
by
  sorry

end NUMINAMATH_GPT_waiter_tables_l476_47694


namespace NUMINAMATH_GPT_induction_step_n_eq_1_l476_47640

theorem induction_step_n_eq_1 : (1 + 2 + 3 = (1+1)*(2*1+1)) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_induction_step_n_eq_1_l476_47640


namespace NUMINAMATH_GPT_sum_modified_midpoint_coordinates_l476_47698

theorem sum_modified_midpoint_coordinates :
  let p1 : (ℝ × ℝ) := (10, 3)
  let p2 : (ℝ × ℝ) := (-4, 7)
  let midpoint : (ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let modified_x := 2 * midpoint.1 
  (modified_x + midpoint.2) = 11 := by
  sorry

end NUMINAMATH_GPT_sum_modified_midpoint_coordinates_l476_47698


namespace NUMINAMATH_GPT_morleys_theorem_l476_47616

def is_trisector (A B C : Point) (p : Point) : Prop :=
sorry -- Definition that this point p is on one of the trisectors of ∠BAC

def triangle (A B C : Point) : Prop :=
sorry -- Definition that points A, B, C form a triangle

def equilateral (A B C : Point) : Prop :=
sorry -- Definition that triangle ABC is equilateral

theorem morleys_theorem (A B C D E F : Point)
  (hABC : triangle A B C)
  (hD : is_trisector A B C D)
  (hE : is_trisector B C A E)
  (hF : is_trisector C A B F) :
  equilateral D E F :=
sorry

end NUMINAMATH_GPT_morleys_theorem_l476_47616


namespace NUMINAMATH_GPT_not_entire_field_weedy_l476_47601

-- Define the conditions
def field_divided_into_100_plots : Prop :=
  ∃ (a b : ℕ), a * b = 100

def initial_weedy_plots : Prop :=
  ∃ (weedy_plots : Finset (ℕ × ℕ)), weedy_plots.card = 9

def plot_becomes_weedy (weedy_plots : Finset (ℕ × ℕ)) (p : ℕ × ℕ) : Prop :=
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots)

-- Theorem statement
theorem not_entire_field_weedy :
  field_divided_into_100_plots →
  initial_weedy_plots →
  (∀ weedy_plots : Finset (ℕ × ℕ), (∀ p : ℕ × ℕ, plot_becomes_weedy weedy_plots p → weedy_plots ∪ {p} = weedy_plots) → weedy_plots.card < 100) :=
  sorry

end NUMINAMATH_GPT_not_entire_field_weedy_l476_47601


namespace NUMINAMATH_GPT_product_remainder_l476_47662

theorem product_remainder (a b : ℕ) (m n : ℤ) (ha : a = 3 * m + 2) (hb : b = 3 * n + 2) : 
  (a * b) % 3 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_product_remainder_l476_47662


namespace NUMINAMATH_GPT_multiplication_addition_l476_47697

theorem multiplication_addition :
  23 * 37 + 16 = 867 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_addition_l476_47697


namespace NUMINAMATH_GPT_total_books_l476_47611

noncomputable def num_books_on_shelf : ℕ := 8

theorem total_books (p h s : ℕ) (assump1 : p = 2) (assump2 : h = 6) (assump3 : s = 36) :
  p + h = num_books_on_shelf :=
by {
  -- leaving the proof construction out as per instructions
  sorry
}

end NUMINAMATH_GPT_total_books_l476_47611


namespace NUMINAMATH_GPT_kwik_e_tax_revenue_l476_47628

def price_federal : ℕ := 50
def price_state : ℕ := 30
def price_quarterly : ℕ := 80

def num_federal : ℕ := 60
def num_state : ℕ := 20
def num_quarterly : ℕ := 10

def revenue_federal := num_federal * price_federal
def revenue_state := num_state * price_state
def revenue_quarterly := num_quarterly * price_quarterly

def total_revenue := revenue_federal + revenue_state + revenue_quarterly

theorem kwik_e_tax_revenue : total_revenue = 4400 := by
  sorry

end NUMINAMATH_GPT_kwik_e_tax_revenue_l476_47628


namespace NUMINAMATH_GPT_drink_costs_l476_47643

theorem drink_costs (cost_of_steak_per_person : ℝ) (total_tip_paid : ℝ) (tip_percentage : ℝ) (billy_tip_coverage_percentage : ℝ) (total_tip_percentage : ℝ) :
  cost_of_steak_per_person = 20 → 
  total_tip_paid = 8 → 
  tip_percentage = 0.20 → 
  billy_tip_coverage_percentage = 0.80 → 
  total_tip_percentage = 0.20 → 
  ∃ (cost_of_drink : ℝ), cost_of_drink = 1.60 :=
by
  intros
  sorry

end NUMINAMATH_GPT_drink_costs_l476_47643


namespace NUMINAMATH_GPT_probability_at_least_one_bean_distribution_of_X_expectation_of_X_l476_47651

noncomputable def total_ways := Nat.choose 6 3
noncomputable def ways_select_2_egg_1_bean := (Nat.choose 4 2) * (Nat.choose 2 1)
noncomputable def ways_select_1_egg_2_bean := (Nat.choose 4 1) * (Nat.choose 2 2)
noncomputable def at_least_one_bean_probability := (ways_select_2_egg_1_bean + ways_select_1_egg_2_bean) / total_ways

theorem probability_at_least_one_bean : at_least_one_bean_probability = 4 / 5 :=
by sorry

noncomputable def p_X_eq_0 := (Nat.choose 4 3) / total_ways
noncomputable def p_X_eq_1 := ways_select_2_egg_1_bean / total_ways
noncomputable def p_X_eq_2 := ways_select_1_egg_2_bean / total_ways

theorem distribution_of_X : p_X_eq_0 = 1 / 5 ∧ p_X_eq_1 = 3 / 5 ∧ p_X_eq_2 = 1 / 5 :=
by sorry

noncomputable def E_X := (0 * p_X_eq_0) + (1 * p_X_eq_1) + (2 * p_X_eq_2)

theorem expectation_of_X : E_X = 1 :=
by sorry

end NUMINAMATH_GPT_probability_at_least_one_bean_distribution_of_X_expectation_of_X_l476_47651


namespace NUMINAMATH_GPT_solution_proof_l476_47607

def count_multiples (n : ℕ) (m : ℕ) (limit : ℕ) : ℕ :=
  (limit - 1) / m + 1

def problem_statement : Prop :=
  let multiples_of_10 := count_multiples 1 10 300
  let multiples_of_10_and_6 := count_multiples 1 30 300
  let multiples_of_10_and_11 := count_multiples 1 110 300
  let unwanted_multiples := multiples_of_10_and_6 + multiples_of_10_and_11
  multiples_of_10 - unwanted_multiples = 20

theorem solution_proof : problem_statement :=
  by {
    sorry
  }

end NUMINAMATH_GPT_solution_proof_l476_47607


namespace NUMINAMATH_GPT_lily_distance_from_start_l476_47695

open Real

def north_south_net := 40 - 10 -- 30 meters south
def east_west_net := 30 - 15 -- 15 meters east

theorem lily_distance_from_start : 
  ∀ (north_south : ℝ) (east_west : ℝ), 
    north_south = north_south_net → 
    east_west = east_west_net → 
    distance = Real.sqrt ((north_south * north_south) + (east_west * east_west)) → 
    distance = 15 * Real.sqrt 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lily_distance_from_start_l476_47695


namespace NUMINAMATH_GPT_probability_is_correct_l476_47676

noncomputable def probability_total_more_than_7 : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 15
  favorable_outcomes / total_outcomes

theorem probability_is_correct :
  probability_total_more_than_7 = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_is_correct_l476_47676


namespace NUMINAMATH_GPT_turban_as_part_of_salary_l476_47680

-- Definitions of the given conditions
def annual_salary (T : ℕ) : ℕ := 90 + 70 * T
def nine_month_salary (T : ℕ) : ℕ := 3 * (90 + 70 * T) / 4
def leaving_amount : ℕ := 50 + 70

-- Proof problem statement in Lean 4
theorem turban_as_part_of_salary (T : ℕ) (h : nine_month_salary T = leaving_amount) : T = 1 := 
sorry

end NUMINAMATH_GPT_turban_as_part_of_salary_l476_47680


namespace NUMINAMATH_GPT_find_x_minus_y_l476_47689

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l476_47689


namespace NUMINAMATH_GPT_correct_statement_is_B_l476_47619

def coefficient_of_x : Int := 1
def is_monomial (t : String) : Bool := t = "1x^0"
def coefficient_of_neg_3x : Int := -3
def degree_of_5x2y : Int := 3

theorem correct_statement_is_B :
  (coefficient_of_x = 0) = false ∧ 
  (is_monomial "1x^0" = true) ∧ 
  (coefficient_of_neg_3x = 3) = false ∧ 
  (degree_of_5x2y = 2) = false ∧ 
  (B = "1 is a monomial") :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_statement_is_B_l476_47619


namespace NUMINAMATH_GPT_member_sum_of_two_others_l476_47630

def numMembers : Nat := 1978
def numCountries : Nat := 6

theorem member_sum_of_two_others :
  ∃ m : ℕ, m ∈ Finset.range numMembers.succ ∧
  ∃ a b : ℕ, a ∈ Finset.range numMembers.succ ∧ b ∈ Finset.range numMembers.succ ∧ 
  ∃ country : Fin (numCountries + 1), (a = m + b ∧ country = country) :=
by
  sorry

end NUMINAMATH_GPT_member_sum_of_two_others_l476_47630


namespace NUMINAMATH_GPT_selection_methods_eq_total_students_l476_47603

def num_boys := 36
def num_girls := 28
def total_students : ℕ := num_boys + num_girls

theorem selection_methods_eq_total_students :
    total_students = 64 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_selection_methods_eq_total_students_l476_47603


namespace NUMINAMATH_GPT_compute_g_neg_101_l476_47614

noncomputable def g (x : ℝ) : ℝ := sorry

theorem compute_g_neg_101 (g_condition : ∀ x y : ℝ, g (x * y) + x = x * g y + g x)
                         (g1 : g 1 = 7) :
    g (-101) = -95 := 
by 
  sorry

end NUMINAMATH_GPT_compute_g_neg_101_l476_47614


namespace NUMINAMATH_GPT_unique_triple_solution_l476_47670

theorem unique_triple_solution :
  ∃! (x y z : ℕ), (y > 1) ∧ Prime y ∧
                  (¬(3 ∣ z ∧ y ∣ z)) ∧
                  (x^3 - y^3 = z^2) ∧
                  (x = 8 ∧ y = 7 ∧ z = 13) :=
by
  sorry

end NUMINAMATH_GPT_unique_triple_solution_l476_47670


namespace NUMINAMATH_GPT_P_eq_Q_l476_47600

open Set Real

def P : Set ℝ := {m | -1 < m ∧ m ≤ 0}
def Q : Set ℝ := {m | ∀ (x : ℝ), m * x^2 + 4 * m * x - 4 < 0}

theorem P_eq_Q : P = Q :=
by
  sorry

end NUMINAMATH_GPT_P_eq_Q_l476_47600


namespace NUMINAMATH_GPT_minimum_y_l476_47671

theorem minimum_y (x : ℝ) (h : x > 1) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y = 3) :=
by
  sorry

end NUMINAMATH_GPT_minimum_y_l476_47671


namespace NUMINAMATH_GPT_initially_caught_and_tagged_fish_l476_47617

theorem initially_caught_and_tagged_fish (N T : ℕ) (hN : N = 800) (h_ratio : 2 / 40 = T / N) : T = 40 :=
by
  have hN : N = 800 := hN
  have h_ratio : 2 / 40 = T / 800 := by rw [hN] at h_ratio; exact h_ratio
  sorry

end NUMINAMATH_GPT_initially_caught_and_tagged_fish_l476_47617


namespace NUMINAMATH_GPT_solve_quadratic_l476_47678

theorem solve_quadratic (x : ℝ) : x^2 - 4 * x + 3 = 0 ↔ (x = 1 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l476_47678


namespace NUMINAMATH_GPT_natural_numbers_between_sqrt_100_and_101_l476_47604

theorem natural_numbers_between_sqrt_100_and_101 :
  ∃ (n : ℕ), n = 200 ∧ (∀ k : ℕ, 100 < Real.sqrt k ∧ Real.sqrt k < 101 -> 10000 < k ∧ k < 10201) := 
by
  sorry

end NUMINAMATH_GPT_natural_numbers_between_sqrt_100_and_101_l476_47604


namespace NUMINAMATH_GPT_carl_candy_bars_l476_47692

/-- 
Carl earns $0.75 every week for taking out his neighbor's trash. 
Carl buys a candy bar every time he earns $0.50. 
After four weeks, Carl will be able to buy 6 candy bars.
-/
theorem carl_candy_bars :
  (0.75 * 4) / 0.50 = 6 := 
  by
    sorry

end NUMINAMATH_GPT_carl_candy_bars_l476_47692


namespace NUMINAMATH_GPT_polynomial_roots_correct_l476_47684

theorem polynomial_roots_correct :
  (∃ (s : Finset ℝ), s = {1, 2, 4} ∧ (∀ x, x ∈ s ↔ (Polynomial.eval x (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 7 * Polynomial.X^2 + Polynomial.C 14 * Polynomial.X - Polynomial.C 8) = 0))) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_correct_l476_47684


namespace NUMINAMATH_GPT_fraction_equivalent_to_decimal_l476_47638

theorem fraction_equivalent_to_decimal : 
  (0.4 -- using appropriate representation for repeating decimal 0.4\overline{13}
      + 13 / 990) = 409 / 990 ∧ Nat.gcd 409 990 = 1 := 
sorry

end NUMINAMATH_GPT_fraction_equivalent_to_decimal_l476_47638


namespace NUMINAMATH_GPT_Levi_has_5_lemons_l476_47625

theorem Levi_has_5_lemons
  (Levi Jayden Eli Ian : ℕ)
  (h1 : Jayden = Levi + 6)
  (h2 : Eli = 3 * Jayden)
  (h3 : Ian = 2 * Eli)
  (h4 : Levi + Jayden + Eli + Ian = 115) :
  Levi = 5 := 
sorry

end NUMINAMATH_GPT_Levi_has_5_lemons_l476_47625


namespace NUMINAMATH_GPT_evaluate_expression_l476_47687

theorem evaluate_expression (a : ℤ) : ((a + 10) - a + 3) * ((a + 10) - a - 2) = 104 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l476_47687


namespace NUMINAMATH_GPT_minimum_value_of_f_on_neg_interval_l476_47623

theorem minimum_value_of_f_on_neg_interval (f : ℝ → ℝ) 
    (h_even : ∀ x, f (-x) = f x) 
    (h_increasing : ∀ x y, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y) 
  : ∀ x, -2 ≤ x → x ≤ -1 → f (-1) ≤ f x := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_on_neg_interval_l476_47623


namespace NUMINAMATH_GPT_region_area_l476_47618

noncomputable def area_of_region := 4 * Real.pi

theorem region_area :
  (∃ x y, x^2 + y^2 - 4 * x + 2 * y + 1 = 0) →
  Real.pi * 4 = area_of_region :=
by
  sorry

end NUMINAMATH_GPT_region_area_l476_47618


namespace NUMINAMATH_GPT_range_of_a_l476_47665

noncomputable def f (a x : ℝ) : ℝ := min (Real.exp x - 2) (Real.exp (2 * x) - a * Real.exp x + a + 24)

def has_three_zeros (f : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0

theorem range_of_a (a : ℝ) :
  has_three_zeros (f a) ↔ 12 < a ∧ a < 28 :=
sorry

end NUMINAMATH_GPT_range_of_a_l476_47665


namespace NUMINAMATH_GPT_inequality_proof_l476_47633

variable (a b : Real)
variable (θ : Real)

-- Line equation and point condition
def line_eq := ∀ x y, x / a + y / b = 1 → (x, y) = (Real.cos θ, Real.sin θ)
-- Main theorem to prove
theorem inequality_proof : (line_eq a b θ) → 1 / (a^2) + 1 / (b^2) ≥ 1 := sorry

end NUMINAMATH_GPT_inequality_proof_l476_47633


namespace NUMINAMATH_GPT_average_ratio_one_l476_47660

theorem average_ratio_one (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / 50)
  let scores_with_averages := scores ++ [A, A]
  let A' := (scores_with_averages.sum / 52)
  A' = A :=
by
  sorry

end NUMINAMATH_GPT_average_ratio_one_l476_47660


namespace NUMINAMATH_GPT_relationship_l476_47668

noncomputable def a : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def b : ℝ := (3 / 5) ^ (2 / 5)
noncomputable def c : ℝ := Real.logb (3 / 5) (2 / 5)

theorem relationship : a < b ∧ b < c :=
by
  -- proof will go here
  sorry


end NUMINAMATH_GPT_relationship_l476_47668


namespace NUMINAMATH_GPT_toy_robot_shipment_l476_47673

-- Define the conditions provided in the problem
def thirty_percent_displayed (total: ℕ) : ℕ := (3 * total) / 10
def seventy_percent_stored (total: ℕ) : ℕ := (7 * total) / 10

-- The main statement to prove: if 70% of the toy robots equal 140, then the total number of toy robots is 200
theorem toy_robot_shipment (total : ℕ) (h : seventy_percent_stored total = 140) : total = 200 :=
by
  -- We will fill in the proof here
  sorry

end NUMINAMATH_GPT_toy_robot_shipment_l476_47673
