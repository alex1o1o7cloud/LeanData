import Mathlib

namespace intersection_eq_singleton_zero_l1291_129108

-- Definition of the sets M and N
def M : Set ℤ := {0, 1}
def N : Set ℤ := { x | ∃ n : ℤ, x = 2 * n }

-- The theorem stating that the intersection of M and N is {0}
theorem intersection_eq_singleton_zero : M ∩ N = {0} :=
by
  sorry

end intersection_eq_singleton_zero_l1291_129108


namespace person_a_age_l1291_129111

theorem person_a_age (A B : ℕ) (h1 : A + B = 43) (h2 : A + 4 = B + 7) : A = 23 :=
by sorry

end person_a_age_l1291_129111


namespace range_g_l1291_129136

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x * Real.cos x + (Real.cos x)^2 - 1/2

noncomputable def g (x : ℝ) : ℝ := 
  let h (x : ℝ) := (Real.sin (2 * x + Real.pi))
  h (x - (5 * Real.pi / 12))

theorem range_g :
  (Set.image g (Set.Icc (-Real.pi/12) (Real.pi/3))) = Set.Icc (-1) (1/2) :=
  sorry

end range_g_l1291_129136


namespace rain_difference_l1291_129174

theorem rain_difference (r_m r_t : ℝ) (h_monday : r_m = 0.9) (h_tuesday : r_t = 0.2) : r_m - r_t = 0.7 :=
by sorry

end rain_difference_l1291_129174


namespace ratio_of_speeds_l1291_129192

theorem ratio_of_speeds (P R : ℝ) (total_time : ℝ) (time_rickey : ℝ)
  (h1 : total_time = 70)
  (h2 : time_rickey = 40)
  (h3 : total_time - time_rickey = 30) :
  P / R = 3 / 4 :=
by
  sorry

end ratio_of_speeds_l1291_129192


namespace integer_xyz_zero_l1291_129102

theorem integer_xyz_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end integer_xyz_zero_l1291_129102


namespace solution_set_real_implies_conditions_l1291_129146

variable {a b c : ℝ}

theorem solution_set_real_implies_conditions (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c < 0) : a < 0 ∧ (b^2 - 4 * a * c) < 0 := 
sorry

end solution_set_real_implies_conditions_l1291_129146


namespace job_completion_time_l1291_129101

theorem job_completion_time (h1 : ∀ {a d : ℝ}, 4 * (1/a + 1/d) = 1)
                             (h2 : ∀ d : ℝ, d = 11.999999999999998) :
                             (∀ a : ℝ, a = 6) :=
by
  sorry

end job_completion_time_l1291_129101


namespace expected_number_of_2s_when_three_dice_rolled_l1291_129127

def probability_of_rolling_2 : ℚ := 1 / 6
def probability_of_not_rolling_2 : ℚ := 5 / 6

theorem expected_number_of_2s_when_three_dice_rolled :
  (0 * (probability_of_not_rolling_2)^3 + 
   1 * 3 * (probability_of_rolling_2) * (probability_of_not_rolling_2)^2 + 
   2 * 3 * (probability_of_rolling_2)^2 * (probability_of_not_rolling_2) + 
   3 * (probability_of_rolling_2)^3) = 
   1 / 2 :=
by
  sorry

end expected_number_of_2s_when_three_dice_rolled_l1291_129127


namespace find_original_price_of_dish_l1291_129142

noncomputable def original_price_of_dish (P : ℝ) : Prop :=
  let john_paid := (0.9 * P) + (0.15 * P)
  let jane_paid := (0.9 * P) + (0.135 * P)
  john_paid = jane_paid + 0.60 → P = 40

theorem find_original_price_of_dish (P : ℝ) (h : original_price_of_dish P) : P = 40 := by
  sorry

end find_original_price_of_dish_l1291_129142


namespace find_a_given_difference_l1291_129154

theorem find_a_given_difference (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : |a - a^2| = 6) : a = 3 :=
sorry

end find_a_given_difference_l1291_129154


namespace common_solution_ys_l1291_129193

theorem common_solution_ys : 
  {y : ℝ | ∃ x : ℝ, x^2 + y^2 = 9 ∧ x^2 + 2*y = 7} = {1 + Real.sqrt 3, 1 - Real.sqrt 3} :=
sorry

end common_solution_ys_l1291_129193


namespace solve_for_a_l1291_129103

variable (a u : ℝ)

def eq1 := (3 / a) + (1 / u) = 7 / 2
def eq2 := (2 / a) - (3 / u) = 6

theorem solve_for_a (h1 : eq1 a u) (h2 : eq2 a u) : a = 2 / 3 := 
by
  sorry

end solve_for_a_l1291_129103


namespace truth_values_set1_truth_values_set2_l1291_129163

-- Definitions for set (1)
def p1 : Prop := Prime 3
def q1 : Prop := Even 3

-- Definitions for set (2)
def p2 (x : Int) : Prop := x = -2 ∧ (x^2 + x - 2 = 0)
def q2 (x : Int) : Prop := x = 1 ∧ (x^2 + x - 2 = 0)

-- Theorem for set (1)
theorem truth_values_set1 : 
  (p1 ∨ q1) = true ∧ (p1 ∧ q1) = false ∧ (¬p1) = false := by sorry

-- Theorem for set (2)
theorem truth_values_set2 (x : Int) :
  (p2 x ∨ q2 x) = true ∧ (p2 x ∧ q2 x) = true ∧ (¬p2 x) = false := by sorry

end truth_values_set1_truth_values_set2_l1291_129163


namespace owner_overtakes_thief_l1291_129156

theorem owner_overtakes_thief :
  ∀ (speed_thief speed_owner : ℕ) (time_theft_discovered : ℝ), 
    speed_thief = 45 →
    speed_owner = 50 →
    time_theft_discovered = 0.5 →
    (time_theft_discovered + (45 * 0.5) / (speed_owner - speed_thief)) = 5 := 
by
  intros speed_thief speed_owner time_theft_discovered h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end owner_overtakes_thief_l1291_129156


namespace rod_length_l1291_129159

theorem rod_length (num_pieces : ℝ) (length_per_piece : ℝ) (h1 : num_pieces = 118.75) (h2 : length_per_piece = 0.40) : 
  num_pieces * length_per_piece = 47.5 := by
  sorry

end rod_length_l1291_129159


namespace derivative_of_f_l1291_129186

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * (Real.cos x + Real.sin x)

theorem derivative_of_f (x : ℝ) : deriv f x = -2 * Real.exp (-x) * Real.sin x :=
by sorry

end derivative_of_f_l1291_129186


namespace rectangle_side_divisible_by_4_l1291_129175

theorem rectangle_side_divisible_by_4 (a b : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ a → i % 4 = 0)
  (h2 : ∀ j, 1 ≤ j ∧ j ≤ b → j % 4 = 0): 
  (a % 4 = 0) ∨ (b % 4 = 0) :=
sorry

end rectangle_side_divisible_by_4_l1291_129175


namespace pastries_more_than_cakes_l1291_129132

def cakes_made : ℕ := 19
def pastries_made : ℕ := 131

theorem pastries_more_than_cakes : pastries_made - cakes_made = 112 :=
by {
  -- Proof will be inserted here
  sorry
}

end pastries_more_than_cakes_l1291_129132


namespace jeremy_gifted_37_goats_l1291_129150

def initial_horses := 100
def initial_sheep := 29
def initial_chickens := 9

def total_initial_animals := initial_horses + initial_sheep + initial_chickens
def animals_bought_by_brian := total_initial_animals / 2
def animals_left_after_brian := total_initial_animals - animals_bought_by_brian

def total_male_animals := 53
def total_female_animals := 53
def total_remaining_animals := total_male_animals + total_female_animals

def goats_gifted_by_jeremy := total_remaining_animals - animals_left_after_brian

theorem jeremy_gifted_37_goats :
  goats_gifted_by_jeremy = 37 := 
by 
  sorry

end jeremy_gifted_37_goats_l1291_129150


namespace geographic_info_tech_helps_western_development_l1291_129196

namespace GeographicInfoTech

def monitors_three_gorges_project : Prop :=
  -- Point ①
  true

def monitors_ecological_environment_meteorological_changes_and_provides_accurate_info : Prop :=
  -- Point ②
  true

def tracks_migration_tibetan_antelopes : Prop :=
  -- Point ③
  true

def addresses_ecological_environment_issues_in_southwest : Prop :=
  -- Point ④
  true

noncomputable def provides_services_for_development_western_regions : Prop :=
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes -- A (①②③)

-- Theorem stating that geographic information technology helps in ①, ②, ③ given its role
theorem geographic_info_tech_helps_western_development (h : provides_services_for_development_western_regions) :
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes := 
by
  exact h

end GeographicInfoTech

end geographic_info_tech_helps_western_development_l1291_129196


namespace find_a_b_sum_l1291_129158

-- Definitions for the conditions
def equation1 (a : ℝ) : Prop := 3 = (1 / 3) * 6 + a
def equation2 (b : ℝ) : Prop := 6 = (1 / 3) * 3 + b

theorem find_a_b_sum : 
  ∃ (a b : ℝ), equation1 a ∧ equation2 b ∧ (a + b = 6) :=
sorry

end find_a_b_sum_l1291_129158


namespace equal_clubs_and_students_l1291_129166

theorem equal_clubs_and_students (S C : ℕ) 
  (h1 : ∀ c : ℕ, c < C → ∃ (m : ℕ → Prop), (∃ p, m p ∧ p = 3))
  (h2 : ∀ s : ℕ, s < S → ∃ (n : ℕ → Prop), (∃ p, n p ∧ p = 3)) :
  S = C := 
by
  sorry

end equal_clubs_and_students_l1291_129166


namespace bucky_savings_excess_l1291_129140

def cost_of_game := 60
def saved_amount := 15
def fish_earnings_weekends (fish : String) : ℕ :=
  match fish with
  | "trout" => 5
  | "bluegill" => 4
  | "bass" => 7
  | "catfish" => 6
  | _ => 0

def fish_earnings_weekdays (fish : String) : ℕ :=
  match fish with
  | "trout" => 10
  | "bluegill" => 8
  | "bass" => 14
  | "catfish" => 12
  | _ => 0

def sunday_fish := 10
def weekday_fish := 3
def weekdays := 2

def sunday_fish_distribution := [
  ("trout", 3),
  ("bluegill", 2),
  ("bass", 4),
  ("catfish", 1)
]

noncomputable def sunday_earnings : ℕ :=
  sunday_fish_distribution.foldl (λ acc (fish, count) =>
    acc + count * fish_earnings_weekends fish) 0

noncomputable def weekday_earnings : ℕ :=
  weekdays * weekday_fish * (
    fish_earnings_weekdays "trout" +
    fish_earnings_weekdays "bluegill" +
    fish_earnings_weekdays "bass")

noncomputable def total_earnings : ℕ :=
  sunday_earnings + weekday_earnings

noncomputable def total_savings : ℕ :=
  total_earnings + saved_amount

theorem bucky_savings_excess :
  total_savings - cost_of_game = 76 :=
by sorry

end bucky_savings_excess_l1291_129140


namespace burger_cost_l1291_129181

theorem burger_cost (days_in_june : ℕ) (burgers_per_day : ℕ) (total_spent : ℕ) (h1 : days_in_june = 30) (h2 : burgers_per_day = 2) (h3 : total_spent = 720) : 
  total_spent / (burgers_per_day * days_in_june) = 12 :=
by
  -- We will prove this in Lean, but skipping the proof here
  sorry

end burger_cost_l1291_129181


namespace fifth_dog_weight_l1291_129187

theorem fifth_dog_weight (y : ℝ) (h : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y) / 5) : y = 31 :=
by
  sorry

end fifth_dog_weight_l1291_129187


namespace Rajesh_days_to_complete_l1291_129198

theorem Rajesh_days_to_complete (Mahesh_days : ℕ) (Rajesh_days : ℕ) (Total_days : ℕ)
  (h1 : Mahesh_days = 45) (h2 : Total_days - 20 = Rajesh_days) (h3 : Total_days = 54) :
  Rajesh_days = 34 :=
by
  sorry

end Rajesh_days_to_complete_l1291_129198


namespace union_complements_eq_l1291_129144

-- Definitions for the universal set U and subsets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5, 7}
def B : Set ℕ := {3, 4, 5}

-- Definition of the complements of A and B with respect to U
def complement_U_A : Set ℕ := {x ∈ U | x ∉ A}
def complement_U_B : Set ℕ := {x ∈ U | x ∉ B}

-- The union of the two complements
def union_complements : Set ℕ := complement_U_A ∪ complement_U_B

-- The target proof statement
theorem union_complements_eq : union_complements = {1, 2, 3, 6, 7} := by
  sorry

end union_complements_eq_l1291_129144


namespace germination_rate_proof_l1291_129184

def random_number_table := [[78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279],
                            [43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820],
                            [61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636],
                            [63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421],
                            [42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983]]

noncomputable def first_4_tested_seeds : List Nat :=
  let numbers_in_random_table := [390, 737, 924, 220, 372]
  numbers_in_random_table.filter (λ x => x < 850) |>.take 4

theorem germination_rate_proof :
  first_4_tested_seeds = [390, 737, 220, 372] := 
by 
  sorry

end germination_rate_proof_l1291_129184


namespace gasoline_added_correct_l1291_129199

def tank_capacity := 48
def initial_fraction := 3 / 4
def final_fraction := 9 / 10

def gasoline_at_initial_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_at_final_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_added (initial: ℝ) (final: ℝ) : ℝ := final - initial

theorem gasoline_added_correct (capacity: ℝ) (initial_fraction: ℝ) (final_fraction: ℝ)
  (h_capacity : capacity = 48) (h_initial : initial_fraction = 3 / 4) (h_final : final_fraction = 9 / 10) :
  gasoline_added (gasoline_at_initial_fraction capacity initial_fraction) (gasoline_at_final_fraction capacity final_fraction) = 7.2 :=
by
  sorry

end gasoline_added_correct_l1291_129199


namespace volume_PQRS_is_48_39_cm3_l1291_129155

noncomputable def area_of_triangle (a h : ℝ) : ℝ := 0.5 * a * h

noncomputable def volume_of_tetrahedron (base_area height : ℝ) : ℝ := (1/3) * base_area * height

noncomputable def height_from_area (area base : ℝ) : ℝ := (2 * area) / base

noncomputable def volume_of_tetrahedron_PQRS : ℝ :=
  let PQ := 5
  let area_PQR := 18
  let area_PQS := 16
  let angle_PQ := 45
  let h_PQR := height_from_area area_PQR PQ
  let h_PQS := height_from_area area_PQS PQ
  let h := h_PQS * (Real.sin (angle_PQ * Real.pi / 180))
  volume_of_tetrahedron area_PQR h

theorem volume_PQRS_is_48_39_cm3 : volume_of_tetrahedron_PQRS = 48.39 := by
  sorry

end volume_PQRS_is_48_39_cm3_l1291_129155


namespace bianca_deleted_text_files_l1291_129143

theorem bianca_deleted_text_files (pictures songs total : ℕ) (h₁ : pictures = 2) (h₂ : songs = 8) (h₃ : total = 17) :
  total - (pictures + songs) = 7 :=
by {
  sorry
}

end bianca_deleted_text_files_l1291_129143


namespace correct_average_l1291_129131

theorem correct_average 
  (n : ℕ) (initial_average : ℚ) (wrong_number : ℚ) (correct_number : ℚ) (wrong_average : ℚ)
  (h_n : n = 10) 
  (h_initial : initial_average = 14) 
  (h_wrong_number : wrong_number = 26) 
  (h_correct_number : correct_number = 36) 
  (h_wrong_average : wrong_average = 14) : 
  (initial_average * n - wrong_number + correct_number) / n = 15 := 
by
  sorry

end correct_average_l1291_129131


namespace cranberries_left_in_bog_l1291_129110

theorem cranberries_left_in_bog
  (total_cranberries : ℕ)
  (percentage_harvested_by_humans : ℕ)
  (cranberries_eaten_by_elk : ℕ)
  (initial_count : total_cranberries = 60000)
  (harvested_percentage : percentage_harvested_by_humans = 40)
  (eaten_by_elk : cranberries_eaten_by_elk = 20000) :
  total_cranberries - (total_cranberries * percentage_harvested_by_humans / 100) - cranberries_eaten_by_elk = 16000 :=
by
  sorry

end cranberries_left_in_bog_l1291_129110


namespace sum_of_fourth_powers_l1291_129178

theorem sum_of_fourth_powers
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2)
  (h3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25 / 6 := 
sorry

end sum_of_fourth_powers_l1291_129178


namespace emily_subtracts_99_from_50sq_to_get_49sq_l1291_129147

-- Define the identity for squares
theorem emily_subtracts_99_from_50sq_to_get_49sq :
  ∀ (x : ℕ), (49 : ℕ) = (50 - 1) → (x = 50 → 49^2 = 50^2 - 99) := by
  intro x h1 h2
  sorry

end emily_subtracts_99_from_50sq_to_get_49sq_l1291_129147


namespace rectangle_area_l1291_129133

noncomputable def area_of_rectangle (radius : ℝ) (ab ad : ℝ) : ℝ :=
  ab * ad

theorem rectangle_area (radius : ℝ) (ad : ℝ) (ab : ℝ) 
  (h_radius : radius = Real.sqrt 5)
  (h_ab_ad_relation : ab = 4 * ad) : 
  area_of_rectangle radius ab ad = 16 / 5 :=
by
  sorry

end rectangle_area_l1291_129133


namespace cone_from_sector_l1291_129117

theorem cone_from_sector
  (r : ℝ) (slant_height : ℝ)
  (radius_circle : ℝ := 10)
  (angle_sector : ℝ := 252) :
  (r = 7 ∧ slant_height = 10) :=
by
  sorry

end cone_from_sector_l1291_129117


namespace no_bounded_sequence_a1_gt_2015_l1291_129134

theorem no_bounded_sequence_a1_gt_2015 (a1 : ℚ) (h_a1 : a1 > 2015) : 
  ∀ (a_n : ℕ → ℚ), a_n 1 = a1 → 
  (∀ (n : ℕ), ∃ (p_n q_n : ℕ), p_n > 0 ∧ q_n > 0 ∧ (p_n.gcd q_n = 1) ∧ (a_n n = p_n / q_n) ∧ 
  (a_n (n + 1) = (p_n^2 + 2015) / (p_n * q_n))) → 
  ∃ (M : ℚ), ∀ (n : ℕ), a_n n ≤ M → 
  False :=
sorry

end no_bounded_sequence_a1_gt_2015_l1291_129134


namespace find_x_l1291_129123

theorem find_x : 
  (∃ x : ℝ, 
    2.5 * ((3.6 * 0.48 * 2.5) / (0.12 * x * 0.5)) = 2000.0000000000002) → 
  x = 0.225 :=
by
  sorry

end find_x_l1291_129123


namespace trajectory_curve_point_F_exists_l1291_129149

noncomputable def curve_C := { p : ℝ × ℝ | (p.1 - 1/2)^2 + (p.2 - 1/2)^2 = 4 }

theorem trajectory_curve (M : ℝ × ℝ) (p : ℝ × ℝ) (q : ℝ × ℝ) :
    M = ((p.1 + q.1) / 2, (p.2 + q.2) / 2) → 
    p.1^2 + p.2^2 = 9 → 
    q.1^2 + q.2^2 = 9 →
    (p.1 - 1)^2 + (p.2 - 1)^2 > 0 → 
    (q.1 - 1)^2 + (q.2 - 1)^2 > 0 → 
    ((p.1 - 1) * (q.1 - 1) + (p.2 - 1) * (q.2 - 1) = 0) →
    (M.1 - 1/2)^2 + (M.2 - 1/2)^2 = 4 :=
sorry

theorem point_F_exists (E D : ℝ × ℝ) (F : ℝ × ℝ) (H : ℝ × ℝ) :
    E = (9/2, 1/2) → D = (1/2, 1/2) → F.2 = 1/2 → 
    (∃ t : ℝ, t ≠ 9/2 ∧ F.1 = t) →
    (H ∈ curve_C) →
    ((H.1 - 9/2)^2 + (H.2 - 1/2)^2) / ((H.1 - F.1)^2 + (H.2 - 1/2)^2) = 24 * (15 - 8 * H.1) / ((t^2 + 15/4) * (24)) :=
sorry

end trajectory_curve_point_F_exists_l1291_129149


namespace words_per_page_l1291_129170

theorem words_per_page 
    (p : ℕ) 
    (h1 : 150 > 0) 
    (h2 : 150 * p ≡ 200 [MOD 221]) :
    p = 118 := 
by sorry

end words_per_page_l1291_129170


namespace expression_value_l1291_129160

theorem expression_value {a b c d m : ℝ} (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 1) : 
  (a + b) * c * d - 2014 * m = -2014 ∨ (a + b) * c * d - 2014 * m = 2014 := 
by
  sorry

end expression_value_l1291_129160


namespace sum_of_arithmetic_sequence_l1291_129165

noncomputable def arithmetic_sequence_sum (a_1 d : ℝ) (n : ℕ) : ℝ :=
n * a_1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a_1 d : ℝ) (p q : ℕ) (h₁ : p ≠ q) (h₂ : arithmetic_sequence_sum a_1 d p = q) (h₃ : arithmetic_sequence_sum a_1 d q = p) : 
arithmetic_sequence_sum a_1 d (p + q) = - (p + q) := sorry

end sum_of_arithmetic_sequence_l1291_129165


namespace master_craftsman_total_parts_l1291_129177

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l1291_129177


namespace smallest_integer_n_satisfying_inequality_l1291_129190

theorem smallest_integer_n_satisfying_inequality 
  (x y z : ℝ) : 
  (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4) :=
sorry

end smallest_integer_n_satisfying_inequality_l1291_129190


namespace angle_D_is_90_l1291_129119

theorem angle_D_is_90 (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : B = 130) (h5 : C + D = 180) :
  D = 90 :=
by
  sorry

end angle_D_is_90_l1291_129119


namespace qualified_weight_example_l1291_129138

-- Define the range of qualified weights
def is_qualified_weight (w : ℝ) : Prop :=
  9.9 ≤ w ∧ w ≤ 10.1

-- State the problem: show that 10 kg is within the qualified range
theorem qualified_weight_example : is_qualified_weight 10 :=
  by
    sorry

end qualified_weight_example_l1291_129138


namespace rationalize_denominator_sum_equals_49_l1291_129106

open Real

noncomputable def A : ℚ := -1
noncomputable def B : ℚ := -3
noncomputable def C : ℚ := 1
noncomputable def D : ℚ := 2
noncomputable def E : ℚ := 33
noncomputable def F : ℚ := 17

theorem rationalize_denominator_sum_equals_49 :
  let expr := (A * sqrt 3 + B * sqrt 5 + C * sqrt 11 + D * sqrt E) / F
  49 = A + B + C + D + E + F :=
by {
  -- The proof will go here.
  exact sorry
}

end rationalize_denominator_sum_equals_49_l1291_129106


namespace matrix_count_l1291_129137

-- A definition for the type of 3x3 matrices with 1's on the diagonal and * can be 0 or 1
def valid_matrix (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 0 = 1 ∧ 
  m 1 1 = 1 ∧ 
  m 2 2 = 1 ∧ 
  (m 0 1 = 0 ∨ m 0 1 = 1) ∧
  (m 0 2 = 0 ∨ m 0 2 = 1) ∧
  (m 1 0 = 0 ∨ m 1 0 = 1) ∧
  (m 1 2 = 0 ∨ m 1 2 = 1) ∧
  (m 2 0 = 0 ∨ m 2 0 = 1) ∧
  (m 2 1 = 0 ∨ m 2 1 = 1)

-- A definition to check that rows are distinct
def distinct_rows (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 ≠ m 1 ∧ m 1 ≠ m 2 ∧ m 0 ≠ m 2

-- Complete proof problem statement
theorem matrix_count : ∃ (n : ℕ), 
  (∀ m : Matrix (Fin 3) (Fin 3) ℕ, valid_matrix m → distinct_rows m) ∧ 
  n = 45 :=
by
  sorry

end matrix_count_l1291_129137


namespace age_ratio_l1291_129128

theorem age_ratio (darcie_age : ℕ) (father_age : ℕ) (mother_ratio : ℚ) (mother_fraction : ℚ)
  (h1 : darcie_age = 4)
  (h2 : father_age = 30)
  (h3 : mother_ratio = 4/5)
  (h4 : mother_fraction = mother_ratio * father_age)
  (h5 : mother_fraction = 24) :
  (darcie_age : ℚ) / mother_fraction = 1 / 6 :=
by
  sorry

end age_ratio_l1291_129128


namespace radius_of_circumscribed_circle_l1291_129189

-- Definitions based on conditions
def sector (radius : ℝ) (central_angle : ℝ) : Prop :=
  central_angle = 120 ∧ radius = 10

-- Statement of the theorem we want to prove
theorem radius_of_circumscribed_circle (r R : ℝ) (h : sector r 120) : R = 20 := 
by
  sorry

end radius_of_circumscribed_circle_l1291_129189


namespace area_is_prime_number_l1291_129162

open Real Int

noncomputable def area_of_triangle (a : Int) : Real :=
  (a * a : Real) / 20

theorem area_is_prime_number 
  (a : Int) 
  (h1 : ∃ p : ℕ, Nat.Prime p ∧ p = ((a * a) / 20 : Real)) :
  ((a * a) / 20 : Real) = 5 :=
by 
  sorry

end area_is_prime_number_l1291_129162


namespace race_speeds_l1291_129126

theorem race_speeds (x y : ℕ) 
  (h1 : 5 * x + 10 = 5 * y) 
  (h2 : 6 * x = 4 * y) :
  x = 4 ∧ y = 6 :=
by {
  -- Proof will go here, but for now we skip it.
  sorry
}

end race_speeds_l1291_129126


namespace induction_base_case_not_necessarily_one_l1291_129176

theorem induction_base_case_not_necessarily_one :
  (∀ (P : ℕ → Prop) (n₀ : ℕ), (P n₀) → (∀ n, n ≥ n₀ → P n → P (n + 1)) → ∀ n, n ≥ n₀ → P n) ↔
  (∃ n₀ : ℕ, n₀ ≠ 1) :=
sorry

end induction_base_case_not_necessarily_one_l1291_129176


namespace football_team_progress_l1291_129188

theorem football_team_progress (lost_yards gained_yards : Int) : lost_yards = -5 → gained_yards = 13 → lost_yards + gained_yards = 8 := 
by
  intros h_lost h_gained
  rw [h_lost, h_gained]
  sorry

end football_team_progress_l1291_129188


namespace arithmetic_sequence_a6_eq_4_l1291_129115

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Condition: a_n is an arithmetic sequence, so a_(n+1) = a_n + d
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: a_2 = 2
def a_2_eq_2 (a : ℕ → ℝ) : Prop :=
  a 2 = 2

-- Condition: S_4 = 9, where S_n is the sum of first n terms of the sequence
def sum_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1))

def S_4_eq_9 (S : ℕ → ℝ) : Prop :=
  S 4 = 9

-- Proof: a_6 = 4
theorem arithmetic_sequence_a6_eq_4 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a_2_eq_2 a)
  (h3 : sum_S_n a S) 
  (h4 : S_4_eq_9 S) :
  a 6 = 4 := 
sorry

end arithmetic_sequence_a6_eq_4_l1291_129115


namespace greatest_gcd_f_l1291_129113

def f (n : ℕ) : ℕ := 70 + n^2

def g (n : ℕ) : ℕ := Nat.gcd (f n) (f (n + 1))

theorem greatest_gcd_f (n : ℕ) (h : 0 < n) : g n = 281 :=
  sorry

end greatest_gcd_f_l1291_129113


namespace percentage_of_students_with_same_grades_l1291_129145

noncomputable def same_grade_percentage (students_class : ℕ) (grades_A : ℕ) (grades_B : ℕ) (grades_C : ℕ) (grades_D : ℕ) (grades_E : ℕ) : ℚ :=
  ((grades_A + grades_B + grades_C + grades_D + grades_E : ℚ) / students_class) * 100

theorem percentage_of_students_with_same_grades :
  let students_class := 40
  let grades_A := 3
  let grades_B := 5
  let grades_C := 6
  let grades_D := 2
  let grades_E := 1
  same_grade_percentage students_class grades_A grades_B grades_C grades_D grades_E = 42.5 := by
  sorry

end percentage_of_students_with_same_grades_l1291_129145


namespace common_sum_of_matrix_l1291_129180

theorem common_sum_of_matrix :
  let S := (1 / 2 : ℝ) * 25 * (10 + 34)
  let adjusted_total := S + 10
  let common_sum := adjusted_total / 6
  common_sum = 93.33 :=
by
  sorry

end common_sum_of_matrix_l1291_129180


namespace no_real_m_for_parallel_lines_l1291_129135

theorem no_real_m_for_parallel_lines : 
  ∀ (m : ℝ), ∃ (l1 l2 : ℝ × ℝ × ℝ), 
  (l1 = (2, (m + 1), 4)) ∧ (l2 = (m, 3, 4)) ∧ 
  ( ∀ (m : ℝ), -2 / (m + 1) = -m / 3 → false ) :=
by sorry

end no_real_m_for_parallel_lines_l1291_129135


namespace range_of_a_add_b_l1291_129109

-- Define the problem and assumptions
variables (a b : ℝ)
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom ab_eq_a_add_b_add_3 : a * b = a + b + 3

-- Define the theorem to prove
theorem range_of_a_add_b : a + b ≥ 6 :=
sorry

end range_of_a_add_b_l1291_129109


namespace valid_numbers_count_l1291_129167

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 5 ∧ d < 10

def count_valid_numbers : ℕ :=
  let first_digit_choices := 8 -- from 1 to 9 excluding 5
  let second_digit_choices := 8 -- from the digits (0-9 excluding 5 and first digit)
  let third_digit_choices := 7 -- from the digits (0-9 excluding 5 and first two digits)
  let fourth_digit_choices := 6 -- from the digits (0-9 excluding 5 and first three digits)
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem valid_numbers_count : count_valid_numbers = 2688 :=
  by
  sorry

end valid_numbers_count_l1291_129167


namespace number_of_cds_l1291_129121

-- Define the constants
def total_money : ℕ := 37
def cd_price : ℕ := 14
def cassette_price : ℕ := 9

theorem number_of_cds (total_money cd_price cassette_price : ℕ) (h_total_money : total_money = 37) (h_cd_price : cd_price = 14) (h_cassette_price : cassette_price = 9) :
  ∃ n : ℕ, n * cd_price + cassette_price = total_money ∧ n = 2 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end number_of_cds_l1291_129121


namespace ratio_cost_price_selling_price_l1291_129107

theorem ratio_cost_price_selling_price (CP SP : ℝ) (h : SP = 1.5 * CP) : CP / SP = 2 / 3 :=
by
  sorry

end ratio_cost_price_selling_price_l1291_129107


namespace sixth_bar_placement_l1291_129141

theorem sixth_bar_placement (f : ℕ → ℕ) (h1 : f 1 = 1) (h2 : f 2 = 121) :
  (∃ n, f 6 = n ∧ (n = 16 ∨ n = 46 ∨ n = 76 ∨ n = 106)) :=
sorry

end sixth_bar_placement_l1291_129141


namespace marikas_father_twice_her_age_l1291_129182

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l1291_129182


namespace prob_three_students_exactly_two_absent_l1291_129153

def prob_absent : ℚ := 1 / 30
def prob_present : ℚ := 29 / 30

theorem prob_three_students_exactly_two_absent :
  (prob_absent * prob_absent * prob_present) * 3 = 29 / 9000 := by
  sorry

end prob_three_students_exactly_two_absent_l1291_129153


namespace trip_duration_60_mph_l1291_129124

noncomputable def time_at_new_speed (initial_time : ℚ) (initial_speed : ℚ) (new_speed : ℚ) : ℚ :=
  initial_time * (initial_speed / new_speed)

theorem trip_duration_60_mph :
  time_at_new_speed (9 / 2) 70 60 = 5.25 := 
by
  sorry

end trip_duration_60_mph_l1291_129124


namespace num_possible_values_of_M_l1291_129172

theorem num_possible_values_of_M :
  ∃ n : ℕ, n = 8 ∧
  ∃ (a b : ℕ), (10 <= 10*a + b) ∧ (10*a + b < 100) ∧ (9*(a - b) ∈ {k : ℕ | ∃ m : ℕ, k = m^2}) := sorry

end num_possible_values_of_M_l1291_129172


namespace least_positive_four_digit_multiple_of_6_l1291_129179

theorem least_positive_four_digit_multiple_of_6 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 6 = 0 → n ≤ m) := 
sorry

end least_positive_four_digit_multiple_of_6_l1291_129179


namespace blue_paint_needed_l1291_129161

/-- 
If the ratio of blue paint to green paint is \(4:1\), and Sarah wants to make 40 cans of the mixture,
prove that the number of cans of blue paint needed is 32.
-/
theorem blue_paint_needed (r: ℕ) (total_cans: ℕ) (h_ratio: r = 4) (h_total: total_cans = 40) : 
  ∃ b: ℕ, b = 4 / 5 * total_cans ∧ b = 32 :=
by
  sorry

end blue_paint_needed_l1291_129161


namespace soccer_league_games_l1291_129151

theorem soccer_league_games (n_teams games_played : ℕ) (h1 : n_teams = 10) (h2 : games_played = 45) :
  ∃ k : ℕ, (n_teams * (n_teams - 1)) / 2 = games_played ∧ k = 1 :=
by
  sorry

end soccer_league_games_l1291_129151


namespace compare_y1_y2_l1291_129185

noncomputable def quadratic (x : ℝ) : ℝ := -x^2 + 2

theorem compare_y1_y2 :
  let y1 := quadratic 1
  let y2 := quadratic 3
  y1 > y2 :=
by
  let y1 := quadratic 1
  let y2 := quadratic 3
  sorry

end compare_y1_y2_l1291_129185


namespace water_left_l1291_129195

-- Conditions
def initial_water : ℚ := 3
def water_used : ℚ := 11 / 8

-- Proposition to be proven
theorem water_left :
  initial_water - water_used = 13 / 8 := by
  sorry

end water_left_l1291_129195


namespace unique_solution_l1291_129183

theorem unique_solution (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hn : 2 ≤ n) (h_y_bound : y ≤ 5 * 2^(2*n)) :
  x^(2*n+1) - y^(2*n+1) = x * y * z + 2^(2*n+1) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  sorry

end unique_solution_l1291_129183


namespace car_cost_l1291_129122

theorem car_cost (days_in_week : ℕ) (sue_days : ℕ) (sister_days : ℕ) 
  (sue_payment : ℕ) (car_cost : ℕ) 
  (h1 : days_in_week = 7)
  (h2 : sue_days = days_in_week - sister_days)
  (h3 : sister_days = 4)
  (h4 : sue_payment = 900)
  (h5 : sue_payment * days_in_week = sue_days * car_cost) :
  car_cost = 2100 := 
by {
  sorry
}

end car_cost_l1291_129122


namespace smallest_positive_angle_l1291_129194

theorem smallest_positive_angle :
  ∀ (x : ℝ), 12 * (Real.sin x)^3 * (Real.cos x)^3 - 2 * (Real.sin x)^3 * (Real.cos x)^3 = 1 → 
  x = 15 * (Real.pi / 180) :=
by
  intros x h
  sorry

end smallest_positive_angle_l1291_129194


namespace max_blue_points_l1291_129116

theorem max_blue_points (n : ℕ) (r b : ℕ)
  (h1 : n = 2009)
  (h2 : b + r = n)
  (h3 : ∀(k : ℕ), b ≤ k * (k - 1) / 2 → r ≥ k) :
  b = 1964 :=
by
  sorry

end max_blue_points_l1291_129116


namespace solve_equation_l1291_129164

theorem solve_equation (m x : ℝ) (hm_pos : m > 0) (hm_ne_one : m ≠ 1) :
  7.320 * m^(1 + Real.log x / Real.log 3) + m^(1 - Real.log x / Real.log 3) = m^2 + 1 ↔ x = 3 ∨ x = 1/3 :=
by
  sorry

end solve_equation_l1291_129164


namespace fraction_equality_l1291_129171

theorem fraction_equality (a b : ℝ) (h : (1 / a) - (1 / b) = 4) :
  (a - 2 * a * b - b) / (2 * a - 2 * b + 7 * a * b) = 6 :=
by
  sorry

end fraction_equality_l1291_129171


namespace simplify_and_evaluate_l1291_129118

theorem simplify_and_evaluate 
  (a b : ℤ)
  (h1 : a = 2)
  (h2 : b = -1) : 
  (2 * a^2 * b - 4 * a * b^2) - 2 * (a * b^2 + a^2 * b) = -12 := 
by
  rw [h1, h2]
  sorry

end simplify_and_evaluate_l1291_129118


namespace carA_speed_calc_l1291_129168

-- Defining the conditions of the problem
def carA_time : ℕ := 8
def carB_speed : ℕ := 25
def carB_time : ℕ := 4
def distance_ratio : ℕ := 4
def carB_distance : ℕ := carB_speed * carB_time
def carA_distance : ℕ := distance_ratio * carB_distance

-- Mathematical statement to be proven
theorem carA_speed_calc : carA_distance / carA_time = 50 := by
  sorry

end carA_speed_calc_l1291_129168


namespace ratio_flow_chart_to_total_time_l1291_129125

noncomputable def T := 48
noncomputable def D := 18
noncomputable def C := (3 / 8) * T
noncomputable def F := T - C - D

theorem ratio_flow_chart_to_total_time : (F / T) = (1 / 4) := by
  sorry

end ratio_flow_chart_to_total_time_l1291_129125


namespace maura_classroom_students_l1291_129129

theorem maura_classroom_students (T : ℝ) (h1 : Tina_students = T) (h2 : Maura_students = T) (h3 : Zack_students = T / 2) (h4 : Tina_students + Maura_students + Zack_students = 69) : T = 138 / 5 := by
  sorry

end maura_classroom_students_l1291_129129


namespace jo_reading_time_l1291_129173

structure Book :=
  (totalPages : Nat)
  (currentPage : Nat)
  (pageOneHourAgo : Nat)

def readingTime (b : Book) : Nat :=
  let pagesRead := b.currentPage - b.pageOneHourAgo
  let pagesLeft := b.totalPages - b.currentPage
  pagesLeft / pagesRead

theorem jo_reading_time :
  ∀ (b : Book), b.totalPages = 210 → b.currentPage = 90 → b.pageOneHourAgo = 60 → readingTime b = 4 :=
by
  intro b h1 h2 h3
  sorry

end jo_reading_time_l1291_129173


namespace find_a_l1291_129157

noncomputable def A := {x : ℝ | x^2 - 8 * x + 15 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 1 = 0}

theorem find_a (a : ℝ) : (A ∩ B a = B a) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by
  sorry

end find_a_l1291_129157


namespace sally_total_fries_is_50_l1291_129130

-- Definitions for the conditions
def sally_initial_fries : ℕ := 14
def mark_initial_fries : ℕ := 3 * 12
def mark_fraction_given_to_sally : ℕ := mark_initial_fries / 3
def jessica_total_cm_of_fries : ℕ := 240
def fry_length_cm : ℕ := 5
def jessica_total_fries : ℕ := jessica_total_cm_of_fries / fry_length_cm
def jessica_fraction_given_to_sally : ℕ := jessica_total_fries / 2

-- Definition for the question
def total_fries_sally_has (sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally : ℕ) : ℕ :=
  sally_initial_fries + mark_fraction_given_to_sally + jessica_fraction_given_to_sally

-- The theorem to be proved
theorem sally_total_fries_is_50 :
  total_fries_sally_has sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally = 50 :=
sorry

end sally_total_fries_is_50_l1291_129130


namespace area_under_curve_l1291_129152

theorem area_under_curve : 
  ∫ x in (1/2 : ℝ)..(2 : ℝ), (1 / x) = 2 * Real.log 2 := by
  sorry

end area_under_curve_l1291_129152


namespace cost_of_bananas_l1291_129139

theorem cost_of_bananas
  (apple_cost : ℕ)
  (orange_cost : ℕ)
  (banana_cost : ℕ)
  (num_apples : ℕ)
  (num_oranges : ℕ)
  (num_bananas : ℕ)
  (total_paid : ℕ) 
  (discount_threshold : ℕ)
  (discount_amount : ℕ)
  (total_fruits : ℕ)
  (total_without_discount : ℕ) :
  apple_cost = 1 → 
  orange_cost = 2 → 
  num_apples = 5 → 
  num_oranges = 3 → 
  num_bananas = 2 → 
  total_paid = 15 → 
  discount_threshold = 5 → 
  discount_amount = 1 → 
  total_fruits = num_apples + num_oranges + num_bananas →
  total_without_discount = (num_apples * apple_cost) + (num_oranges * orange_cost) + (num_bananas * banana_cost) →
  (total_without_discount - (discount_amount * (total_fruits / discount_threshold))) = total_paid →
  banana_cost = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end cost_of_bananas_l1291_129139


namespace p_as_percentage_of_x_l1291_129114

-- Given conditions
variables (x y z w t u p : ℝ)
variables (h1 : 0.37 * z = 0.84 * y)
variables (h2 : y = 0.62 * x)
variables (h3 : 0.47 * w = 0.73 * z)
variables (h4 : w = t - u)
variables (h5 : u = 0.25 * t)
variables (h6 : p = z + t + u)

-- Prove that p is 505.675% of x
theorem p_as_percentage_of_x : p = 5.05675 * x := by
  sorry

end p_as_percentage_of_x_l1291_129114


namespace total_money_l1291_129120

variable (A B C: ℕ)
variable (h1: A + C = 200) 
variable (h2: B + C = 350)
variable (h3: C = 200)

theorem total_money : A + B + C = 350 :=
by
  sorry

end total_money_l1291_129120


namespace population_doubling_time_l1291_129100

open Real

noncomputable def net_growth_rate (birth_rate : ℝ) (death_rate : ℝ) : ℝ :=
birth_rate - death_rate

noncomputable def percentage_growth_rate (net_growth_rate : ℝ) (population_base : ℝ) : ℝ :=
(net_growth_rate / population_base) * 100

noncomputable def doubling_time (percentage_growth_rate : ℝ) : ℝ :=
70 / percentage_growth_rate

theorem population_doubling_time :
    let birth_rate := 39.4
    let death_rate := 19.4
    let population_base := 1000
    let net_growth := net_growth_rate birth_rate death_rate
    let percentage_growth := percentage_growth_rate net_growth population_base
    doubling_time percentage_growth = 35 := 
by
    sorry

end population_doubling_time_l1291_129100


namespace Bobby_candy_chocolate_sum_l1291_129105

/-
  Bobby ate 33 pieces of candy, then ate 4 more, and he also ate 14 pieces of chocolate.
  Prove that the total number of pieces of candy and chocolate he ate altogether is 51.
-/

theorem Bobby_candy_chocolate_sum :
  let initial_candy := 33
  let more_candy := 4
  let chocolate := 14
  let total_candy := initial_candy + more_candy
  total_candy + chocolate = 51 :=
by
  -- The theorem asserts the problem; apologies, the proof is not required here.
  sorry

end Bobby_candy_chocolate_sum_l1291_129105


namespace solve_for_k_l1291_129169

theorem solve_for_k :
  ∀ (k : ℝ), (∃ x : ℝ, (3*x + 8)*(x - 6) = -50 + k*x) ↔
    k = -10 + 2*Real.sqrt 6 ∨ k = -10 - 2*Real.sqrt 6 := by
  sorry

end solve_for_k_l1291_129169


namespace evaluate_double_sum_l1291_129197

theorem evaluate_double_sum :
  ∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m + 1) ^ 2 / (n + 1) / (m + n + 3) = 1 := by
  sorry

end evaluate_double_sum_l1291_129197


namespace length_of_equal_pieces_l1291_129191

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end length_of_equal_pieces_l1291_129191


namespace ab_conditions_l1291_129148

theorem ab_conditions (a b : ℝ) : ¬((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by 
  sorry

end ab_conditions_l1291_129148


namespace number_of_rabbits_l1291_129112

-- Given conditions
variable (r c : ℕ)
variable (cond1 : r + c = 51)
variable (cond2 : 4 * r = 3 * (2 * c) + 4)

-- To prove
theorem number_of_rabbits : r = 31 :=
sorry

end number_of_rabbits_l1291_129112


namespace grades_calculation_l1291_129104

-- Defining the conditions
def total_students : ℕ := 22800
def students_per_grade : ℕ := 75

-- Stating the theorem to be proved
theorem grades_calculation : total_students / students_per_grade = 304 := sorry

end grades_calculation_l1291_129104
