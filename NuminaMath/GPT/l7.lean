import Mathlib

namespace truck_and_trailer_total_weight_l7_7402

def truck_weight : ℝ := 4800
def trailer_weight (truck_weight : ℝ) : ℝ := 0.5 * truck_weight - 200
def total_weight (truck_weight trailer_weight : ℝ) : ℝ := truck_weight + trailer_weight 

theorem truck_and_trailer_total_weight : 
  total_weight truck_weight (trailer_weight truck_weight) = 7000 :=
by 
  sorry

end truck_and_trailer_total_weight_l7_7402


namespace sum_two_triangular_numbers_iff_l7_7818

theorem sum_two_triangular_numbers_iff (m : ℕ) : 
  (∃ a b : ℕ, m = (a * (a + 1)) / 2 + (b * (b + 1)) / 2) ↔ 
  (∃ x y : ℕ, 4 * m + 1 = x * x + y * y) :=
by sorry

end sum_two_triangular_numbers_iff_l7_7818


namespace kyle_gas_and_maintenance_expense_l7_7477

def monthly_income : ℝ := 3200
def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous_expenses : ℝ := 200
def car_payment : ℝ := 350

def total_bills : ℝ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous_expenses

theorem kyle_gas_and_maintenance_expense :
  monthly_income - total_bills - car_payment = 350 :=
by
  sorry

end kyle_gas_and_maintenance_expense_l7_7477


namespace albums_total_l7_7943

noncomputable def miriam_albums (katrina_albums : ℕ) := 5 * katrina_albums
noncomputable def katrina_albums (bridget_albums : ℕ) := 6 * bridget_albums
noncomputable def bridget_albums (adele_albums : ℕ) := adele_albums - 15
noncomputable def total_albums (adele_albums : ℕ) (bridget_albums : ℕ) (katrina_albums : ℕ) (miriam_albums : ℕ) := 
  adele_albums + bridget_albums + katrina_albums + miriam_albums

theorem albums_total (adele_has_30 : adele_albums = 30) : 
  total_albums 30 (bridget_albums 30) (katrina_albums (bridget_albums 30)) (miriam_albums (katrina_albums (bridget_albums 30))) = 585 := 
by 
  -- In Lean, replace the term 'sorry' below with the necessary steps to finish the proof
  sorry

end albums_total_l7_7943


namespace minimal_team_members_l7_7106

theorem minimal_team_members (n : ℕ) : 
  (n ≡ 1 [MOD 6]) ∧ (n ≡ 2 [MOD 8]) ∧ (n ≡ 3 [MOD 9]) → n = 343 := 
by
  sorry

end minimal_team_members_l7_7106


namespace systematic_classic_equations_l7_7304

theorem systematic_classic_equations (x y : ℕ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔
  (if (exists p q : ℕ, p = 7 * x + 7 ∧ q = 9 * (x - 1)) 
  then x = x ∧ y = 9 * (x - 1) 
  else false) :=
by 
  sorry

end systematic_classic_equations_l7_7304


namespace second_storm_duration_l7_7364

theorem second_storm_duration
  (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : 30 * x + 15 * y = 975) :
  y = 25 := 
sorry

end second_storm_duration_l7_7364


namespace Fran_same_distance_speed_l7_7116

noncomputable def Joann_rides (v_j t_j : ℕ) : ℕ := v_j * t_j

def Fran_speed (d t_f : ℕ) : ℕ := d / t_f

theorem Fran_same_distance_speed
  (v_j t_j t_f : ℕ) (hj: v_j = 15) (tj: t_j = 4) (tf: t_f = 5) : Fran_speed (Joann_rides v_j t_j) t_f = 12 := by
  have hj_dist: Joann_rides v_j t_j = 60 := by
    rw [hj, tj]
    sorry -- proof of Joann's distance
  have d_j: ℕ := 60
  have hf: Fran_speed d_j t_f = Fran_speed 60 5 := by
    rw ←hj_dist
    sorry -- proof to equate d_j with Joann's distance
  show Fran_speed 60 5 = 12
  sorry -- Final computation proof

end Fran_same_distance_speed_l7_7116


namespace calculate_initial_income_l7_7668

noncomputable def initial_income : Float := 151173.52

theorem calculate_initial_income :
  let I := initial_income
  let children_distribution := 0.30 * I
  let eldest_child_share := (children_distribution / 6) + 0.05 * I
  let remaining_for_wife := 0.40 * I
  let remaining_after_distribution := I - (children_distribution + remaining_for_wife)
  let donation_to_orphanage := 0.10 * remaining_after_distribution
  let remaining_after_donation := remaining_after_distribution - donation_to_orphanage
  let federal_tax := 0.02 * remaining_after_donation
  let final_amount := remaining_after_donation - federal_tax
  final_amount = 40000 :=
by
  sorry

end calculate_initial_income_l7_7668


namespace estimated_red_balls_l7_7358

theorem estimated_red_balls
  (total_balls : ℕ)
  (total_draws : ℕ)
  (red_draws : ℕ)
  (h_total_balls : total_balls = 12)
  (h_total_draws : total_draws = 200)
  (h_red_draws : red_draws = 50) :
  red_draws * total_balls = total_draws * 3 :=
by
  sorry

end estimated_red_balls_l7_7358


namespace probability_red_first_given_black_second_l7_7375

open ProbabilityTheory MeasureTheory

-- Definitions for Urn A and Urn B ball quantities
def urnA := (white : 4, red : 2)
def urnB := (red : 3, black : 3)

-- Event of drawing a red ball first and a black ball second
def eventRedFirst := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "red") ∨ (urn = 2 ∧ ball = "red")
def eventBlackSecond := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "black") ∨ (urn = 2 ∧ ball = "black")

-- Probability function definition
noncomputable def P := sorry -- Probability function placeholder

-- Conditional Probability
theorem probability_red_first_given_black_second :
  P(eventRedFirst | eventBlackSecond) = 2 / 5 := sorry

end probability_red_first_given_black_second_l7_7375


namespace james_muffins_l7_7054

theorem james_muffins (arthur_muffins : ℕ) (times : ℕ) (james_muffins : ℕ) 
  (h1 : arthur_muffins = 115) 
  (h2 : times = 12) 
  (h3 : james_muffins = arthur_muffins * times) : 
  james_muffins = 1380 := 
by 
  sorry

end james_muffins_l7_7054


namespace probability_of_exactly_nine_correct_matches_is_zero_l7_7206

theorem probability_of_exactly_nine_correct_matches_is_zero :
  let n := 10 in
  let match_probability (correct: Fin n → Fin n) (guess: Fin n → Fin n) (right_count: Nat) :=
    (Finset.univ.filter (λ i => correct i = guess i)).card = right_count in
  ∀ (correct_guessing: Fin n → Fin n), 
    ∀ (random_guessing: Fin n → Fin n),
      match_probability correct_guessing random_guessing 9 → 
        match_probability correct_guessing random_guessing 10 :=
begin
  sorry -- This skips the proof part
end

end probability_of_exactly_nine_correct_matches_is_zero_l7_7206


namespace sqrt_equiv_1715_l7_7835

noncomputable def sqrt_five_squared_times_seven_sixth : ℕ := 
  Nat.sqrt (5^2 * 7^6)

theorem sqrt_equiv_1715 : sqrt_five_squared_times_seven_sixth = 1715 := by
  sorry

end sqrt_equiv_1715_l7_7835


namespace sqrt_of_expression_l7_7832

theorem sqrt_of_expression : Real.sqrt (5^2 * 7^6) = 1715 := 
by
  sorry

end sqrt_of_expression_l7_7832


namespace solve_equation_l7_7634

theorem solve_equation (y : ℝ) : 
  5 * (y + 2) + 9 = 3 * (1 - y) ↔ y = -2 := 
by 
  sorry

end solve_equation_l7_7634


namespace volleyball_problem_correct_l7_7403

noncomputable def volleyball_problem : Nat :=
  let total_players := 16
  let triplets : Finset String := {"Alicia", "Amanda", "Anna"}
  let twins : Finset String := {"Beth", "Brenda"}
  let remaining_players := total_players - triplets.card - twins.card
  let no_triplets_no_twins := Nat.choose remaining_players 6
  let one_triplet_no_twins := triplets.card * Nat.choose remaining_players 5
  let no_triplets_one_twin := twins.card * Nat.choose remaining_players 5
  no_triplets_no_twins + one_triplet_no_twins + no_triplets_one_twin

theorem volleyball_problem_correct : volleyball_problem = 2772 := by
  sorry

end volleyball_problem_correct_l7_7403


namespace slice_of_bread_area_l7_7815

theorem slice_of_bread_area (total_area : ℝ) (number_of_parts : ℕ) (h1 : total_area = 59.6) (h2 : number_of_parts = 4) : 
  total_area / number_of_parts = 14.9 :=
by
  rw [h1, h2]
  norm_num


end slice_of_bread_area_l7_7815


namespace average_operating_time_l7_7553

-- Definition of problem conditions
def cond1 : Nat := 5 -- originally had 5 air conditioners
def cond2 : Nat := 6 -- after installing 1 more
def total_hours : Nat := 24 * 5 -- total operating hours allowable in 24 hours

-- Formalize the average operating time calculation
theorem average_operating_time : (total_hours / cond2) = 20 := by
  sorry

end average_operating_time_l7_7553


namespace initial_distance_between_Seonghyeon_and_Jisoo_l7_7155

theorem initial_distance_between_Seonghyeon_and_Jisoo 
  (D : ℝ)
  (h1 : 2000 = (D - 200) + 1000) : 
  D = 1200 :=
by
  sorry

end initial_distance_between_Seonghyeon_and_Jisoo_l7_7155


namespace domain_lg_tan_minus_sqrt3_l7_7346

open Real

theorem domain_lg_tan_minus_sqrt3 :
  {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} =
    {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} :=
by
  sorry

end domain_lg_tan_minus_sqrt3_l7_7346


namespace max_marks_l7_7655

theorem max_marks (total_marks : ℕ) (obtained_marks : ℕ) (failed_by : ℕ) 
    (passing_percentage : ℝ) (passing_marks : ℝ) (H1 : obtained_marks = 125)
    (H2 : failed_by = 40) (H3 : passing_percentage = 0.33) 
    (H4 : passing_marks = obtained_marks + failed_by) 
    (H5 : passing_marks = passing_percentage * total_marks) : total_marks = 500 := by
  sorry

end max_marks_l7_7655


namespace difference_in_ages_l7_7972

/-- Definitions: --/
def sum_of_ages (B J : ℕ) := B + J = 70
def jennis_age (J : ℕ) := J = 19

/-- Theorem: --/
theorem difference_in_ages : ∀ (B J : ℕ), sum_of_ages B J → jennis_age J → B - J = 32 :=
by
  intros B J hsum hJ
  rw [jennis_age] at hJ
  rw [sum_of_ages] at hsum
  sorry

end difference_in_ages_l7_7972


namespace number_of_students_l7_7342

theorem number_of_students (n : ℕ)
  (h_avg : 100 * n = total_marks_unknown)
  (h_wrong_marks : total_marks_wrong = total_marks_unknown + 50)
  (h_correct_avg : total_marks_correct / n = 95)
  (h_corrected_marks : total_marks_correct = total_marks_wrong - 50) :
  n = 10 :=
by
  sorry

end number_of_students_l7_7342


namespace simplify_expression_l7_7059

variable (x y : ℝ)

theorem simplify_expression : 3 * x^2 * y * (2 / (9 * x^3 * y)) = 2 / (3 * x) :=
by sorry

end simplify_expression_l7_7059


namespace annie_diorama_time_l7_7414

theorem annie_diorama_time (P B : ℕ) (h1 : B = 3 * P - 5) (h2 : B = 49) : P + B = 67 :=
sorry

end annie_diorama_time_l7_7414


namespace smallest_common_students_l7_7194

theorem smallest_common_students 
    (z : ℕ) (k : ℕ) (j : ℕ) 
    (hz : z = k ∧ k = j) 
    (hz_ratio : ∃ x : ℕ, z = 3 * x ∧ k = 2 * x ∧ j = 5 * x)
    (hz_group : ∃ y : ℕ, z = 14 * y) 
    (hk_group : ∃ w : ℕ, k = 10 * w) 
    (hj_group : ∃ v : ℕ, j = 15 * v) : 
    z = 630 ∧ k = 420 ∧ j = 1050 :=
    sorry

end smallest_common_students_l7_7194


namespace probability_no_adjacent_same_color_l7_7798

noncomputable def beadArrangements : ℕ := nat.factorial 6 / (nat.factorial 3 * nat.factorial 2 * nat.factorial 1)

theorem probability_no_adjacent_same_color : 
  let totalArrangements := beadArrangements in
  let validArrangements := 10 in
  (validArrangements / totalArrangements : ℚ) = 1 / 6 :=
by
  let totalArrangements := beadArrangements
  let validArrangements := 10
  have h : totalArrangements = 60 := by sorry
  rw [h]
  norm_num
  sorry

end probability_no_adjacent_same_color_l7_7798


namespace length_AB_indeterminate_l7_7084

theorem length_AB_indeterminate
  (A B C : Type)
  (AC : ℝ) (BC : ℝ)
  (AC_eq_1 : AC = 1)
  (BC_eq_3 : BC = 3) :
  (2 < AB ∧ AB < 4) ∨ (AB = 2 ∨ AB = 4) → false :=
by sorry

end length_AB_indeterminate_l7_7084


namespace minimize_sum_of_squares_of_perpendiculars_l7_7302

open Real

variable {α β c : ℝ} -- angles and side length

theorem minimize_sum_of_squares_of_perpendiculars
    (habc : α + β = π)
    (P : ℝ)
    (AP BP : ℝ)
    (x : AP + BP = c)
    (u : ℝ)
    (v : ℝ)
    (hAP : AP = P)
    (hBP : BP = c - P)
    (hu : u = P * sin α)
    (hv : v = (c - P) * sin β)
    (f : ℝ)
    (hf : f = (P * sin α)^2 + ((c - P) * sin β)^2):
  (AP / BP = (sin β)^2 / (sin α)^2) := sorry

end minimize_sum_of_squares_of_perpendiculars_l7_7302


namespace gcd_1043_2295_eq_1_l7_7569

theorem gcd_1043_2295_eq_1 : Nat.gcd 1043 2295 = 1 := by
  sorry

end gcd_1043_2295_eq_1_l7_7569


namespace escalator_time_l7_7411

theorem escalator_time (escalator_speed person_speed length : ℕ) 
    (h1 : escalator_speed = 12) 
    (h2 : person_speed = 2) 
    (h3 : length = 196) : 
    (length / (escalator_speed + person_speed) = 14) :=
by
  sorry

end escalator_time_l7_7411


namespace keith_attended_games_l7_7515

-- Definitions based on the given conditions
def total_games : ℕ := 8
def missed_games : ℕ := 4

-- The proof goal: Keith's attendance
def attended_games : ℕ := total_games - missed_games

-- Main statement to prove the total games Keith attended
theorem keith_attended_games : attended_games = 4 := by
  -- Sorry is a placeholder for the proof
  sorry

end keith_attended_games_l7_7515


namespace first_ball_red_given_second_black_l7_7370

open ProbabilityTheory

noncomputable def urn_A : Finset (Finset ℕ) := { {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 1, 2}, ... }
noncomputable def urn_B : Finset (Finset ℕ) := { {1, 1, 1, 2, 2, 2}, {1, 1, 2, 2, 2, 2}, ... }

noncomputable def prob_draw_red : ℕ := 7 / 15

theorem first_ball_red_given_second_black :
  (∑ A_Burn_selection in ({0, 1} : Finset ℕ), 
     ((∑ ball_draw from A_Burn_selection,
           if A_Burn_selection = 0 then (∑ red in urn_A, if red = 1 then 1 else 0) / 6 / 2
           else (∑ red in urn_B, if red = 1 then 1 else 0) / 6 / 2) *
     ((∑ second_urn_selection in ({0, 1} : Finset ℕ),
           if second_urn_selection = 0 and A_Burn_selection = 0 then 
              ∑ black in urn_A, if black = 1 then 1 else 0 / 6 / 2 
           else 
              ∑ black in urn_B, if black = 1 then 1 else 0 / 6 / 2))) = 7 / 15 :=
sorry

end first_ball_red_given_second_black_l7_7370


namespace A_subset_B_l7_7080

def A (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 ≤ 5 / 4

def B (x y : ℝ) (a : ℝ) : Prop :=
  abs (x - 1) + 2 * abs (y - 2) ≤ a

theorem A_subset_B (a : ℝ) (h : a ≥ 5 / 2) : 
  ∀ x y : ℝ, A x y → B x y a := 
sorry

end A_subset_B_l7_7080


namespace service_fee_calculation_l7_7606

-- Problem definitions based on conditions
def cost_food : ℝ := 50
def tip : ℝ := 5
def total_spent : ℝ := 61
def service_fee_percentage (x : ℝ) : Prop := x = (12 / 50) * 100

-- The main statement to be proven, showing that the service fee percentage is 24%
theorem service_fee_calculation : service_fee_percentage 24 :=
by {
  sorry
}

end service_fee_calculation_l7_7606


namespace coprime_divisors_imply_product_divisor_l7_7128

theorem coprime_divisors_imply_product_divisor 
  (a b n : ℕ) (h_coprime : Nat.gcd a b = 1)
  (h_a_div_n : a ∣ n) (h_b_div_n : b ∣ n) : a * b ∣ n :=
by
  sorry

end coprime_divisors_imply_product_divisor_l7_7128


namespace square_side_length_in_right_triangle_l7_7951

theorem square_side_length_in_right_triangle :
  ∀ (DE EF DF : ℝ) (s : ℝ),
  DE = 5 ∧ EF = 12 ∧ DF = 13 ∧
  (∃ P Q S R, 
    P ∈ DF ∧ Q ∈ DF ∧ 
    S ∈ DE ∧ 
    R ∈ EF ∧ 
    square PQRS ∧ 
    ℓ PQ = s) →
  s = 780 / 169 :=
by {
  sorry
}

end square_side_length_in_right_triangle_l7_7951


namespace find_extrema_l7_7258

theorem find_extrema (x y : ℝ) (h1 : x < 0) (h2 : -1 < y) (h3 : y < 0) : 
  max (max x (x*y)) (x*y^2) = x*y ∧ min (min x (x*y)) (x*y^2) = x :=
by sorry

end find_extrema_l7_7258


namespace money_left_after_purchase_l7_7248

noncomputable def total_cost : ℝ := 250 + 25 + 35 + 45 + 90

def savings_erika : ℝ := 155

noncomputable def savings_rick : ℝ := total_cost / 2

def savings_sam : ℝ := 175

def combined_cost_cake_flowers_skincare : ℝ := 25 + 35 + 45

noncomputable def savings_amy : ℝ := 2 * combined_cost_cake_flowers_skincare

noncomputable def total_savings : ℝ := savings_erika + savings_rick + savings_sam + savings_amy

noncomputable def money_left : ℝ := total_savings - total_cost

theorem money_left_after_purchase : money_left = 317.5 := by
  sorry

end money_left_after_purchase_l7_7248


namespace continuous_stripe_probability_l7_7994

notation "ℙ" => ProbabilityTheory.ProbabilityMeasure

-- Define the problem setup.
def stripe_colors (cube : ℕ) : set (set ℕ) :=
  { colors | ∀ face ∈ 𝒰{1,2,3,4,5,6}, colors ∈ {0,1} }

-- Define the probability space.
noncomputable def tower_probability_space : ProbabilityTheory.ProbabilitySpace :=
  ProbabilityTheory.probability_space_of_finset ℙ (stripe_colors 3) sorry

-- Define the event of continuous stripe.
def continuous_stripe_event : Set (Set ℕ) :=
  {colors | ∃ (striped_faces : Finset ℕ), striped_faces.card = 1 ∧ -- One stripe connects top to bottom
  ∀ cube ∈ {1,2,3}, striped_faces ⊆ {face | face ∈ striped_faces }}

-- Statement of the theorem.
theorem continuous_stripe_probability : ℙ(tower_probability_space, continuous_stripe_event) = 1 / 4096 :=
sorry

end continuous_stripe_probability_l7_7994


namespace equation_of_line_passing_through_center_and_perpendicular_to_l_l7_7723

theorem equation_of_line_passing_through_center_and_perpendicular_to_l (a : ℝ) : 
  let C_center := (-2, 1)
  let l_slope := 1
  let m_slope := -1
  ∃ (b : ℝ), ∀ x y : ℝ, (x + y + 1 = 0) := 
by 
  let C_center := (-2, 1)
  let l_slope := 1
  let m_slope := -1
  use 1
  sorry

end equation_of_line_passing_through_center_and_perpendicular_to_l_l7_7723


namespace find_exponent_l7_7905

theorem find_exponent 
  (h1 : (1 : ℝ) / 9 = 3 ^ (-2 : ℝ))
  (h2 : (3 ^ (20 : ℝ) : ℝ) / 9 = 3 ^ x) : 
  x = 18 :=
by sorry

end find_exponent_l7_7905


namespace partitions_of_6_into_4_indistinguishable_boxes_l7_7287

theorem partitions_of_6_into_4_indistinguishable_boxes : 
  ∃ (X : Finset (Multiset ℕ)), X.card = 9 ∧ 
  ∀ p ∈ X, p.sum = 6 ∧ p.card ≤ 4 := 
sorry

end partitions_of_6_into_4_indistinguishable_boxes_l7_7287


namespace expected_number_of_groups_l7_7176

-- Define the conditions
variables (k m : ℕ) (h : 0 < k ∧ 0 < m)

-- Expected value of groups in the sequence
theorem expected_number_of_groups : 
  ∀ k m, (0 < k) → (0 < m) → 
  let total_groups := 1 + (2 * k * m) / (k + m) in total_groups = 1 + (2 * k * m) / (k + m) :=
by
  intros k m hk hm
  let total_groups := 1 + (2 * k * m) / (k + m)
  exact (rfl : total_groups = 1 + (2 * k * m) / (k + m))

end expected_number_of_groups_l7_7176


namespace train_speed_l7_7810

-- Define the platform length in meters and the time taken to cross in seconds
def platform_length : ℝ := 260
def time_crossing : ℝ := 26

-- Define the length of the goods train in meters
def train_length : ℝ := 260.0416

-- Define the total distance covered by the train when crossing the platform
def total_distance : ℝ := platform_length + train_length

-- Define the speed of the train in meters per second
def speed_m_s : ℝ := total_distance / time_crossing

-- Define the conversion factor from meters per second to kilometers per hour
def conversion_factor : ℝ := 3.6

-- Define the speed of the train in kilometers per hour
def speed_km_h : ℝ := speed_m_s * conversion_factor

-- State the theorem to be proved
theorem train_speed : speed_km_h = 72.00576 :=
by
  sorry

end train_speed_l7_7810


namespace probability_is_correct_l7_7391

variables (total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items : ℕ)

-- Setting up the problem according to the given conditions
def conditions := (total_items = 10) ∧ 
                  (truckA_first_class = 2) ∧ (truckA_second_class = 2) ∧ 
                  (truckB_first_class = 4) ∧ (truckB_second_class = 2) ∧ 
                  (brokenA = 1) ∧ (brokenB = 1) ∧
                  (remaining_items = 8)

-- Calculating the probability of selecting a first-class item from the remaining items
def probability_of_first_class : ℚ :=
  1/3 * 1/2 + 1/6 * 5/8 + 1/3 * 5/8 + 1/6 * 3/4

-- The theorem to be proved
theorem probability_is_correct : 
  conditions total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items →
  probability_of_first_class = 29/48 :=
sorry

end probability_is_correct_l7_7391


namespace john_needs_total_planks_l7_7440

theorem john_needs_total_planks : 
  let large_planks := 12
  let small_planks := 17
  large_planks + small_planks = 29 :=
by
  sorry

end john_needs_total_planks_l7_7440


namespace negation_proposition_p_l7_7948

theorem negation_proposition_p (x y : ℝ) : (¬ ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → (x ≠ 1 ∨ y ≠ 2)) :=
by
  sorry

end negation_proposition_p_l7_7948


namespace smallest_sum_of_squares_l7_7435

theorem smallest_sum_of_squares :
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 ≥ 36 ∧ y^2 ≥ 36 ∧ x^2 + y^2 = 625 :=
by
  sorry

end smallest_sum_of_squares_l7_7435


namespace peter_remaining_walk_time_l7_7149

-- Define the parameters and conditions
def total_distance : ℝ := 2.5
def time_per_mile : ℝ := 20
def distance_walked : ℝ := 1

-- Define the remaining distance
def remaining_distance : ℝ := total_distance - distance_walked

-- Define the remaining time Peter needs to walk
def remaining_time_to_walk (d : ℝ) (t : ℝ) : ℝ := d * t

-- State the problem we want to prove
theorem peter_remaining_walk_time :
  remaining_time_to_walk remaining_distance time_per_mile = 30 :=
by
  -- Placeholder for the proof
  sorry

end peter_remaining_walk_time_l7_7149


namespace original_price_l7_7667

theorem original_price (P : ℝ) 
  (h1 : 1.40 * P = P + 700) : P = 1750 :=
by sorry

end original_price_l7_7667


namespace probability_red_given_black_l7_7367

noncomputable def urn_A := {white := 4, red := 2}
noncomputable def urn_B := {red := 3, black := 3}

-- Define the probabilities as required in the conditions
def prob_urn_A := 1 / 2
def prob_urn_B := 1 / 2

def draw_red_from_A := 2 / 6
def draw_black_from_B := 3 / 6
def draw_red_from_B := 3 / 6
def draw_black_from_B_after_red := 3 / 5
def draw_black_from_B_after_black := 2 / 5

def probability_first_red_second_black :=
  (prob_urn_A * draw_red_from_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_black)

def probability_second_black :=
  (prob_urn_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_black_from_B * prob_urn_B * draw_black_from_B_after_black)

theorem probability_red_given_black :
  probability_first_red_second_black / probability_second_black = 7 / 15 :=
sorry

end probability_red_given_black_l7_7367


namespace rectangle_side_difference_l7_7102

theorem rectangle_side_difference (p d x y : ℝ) (h1 : 2 * x + 2 * y = p)
                                   (h2 : x^2 + y^2 = d^2)
                                   (h3 : x = 2 * y) :
    x - y = p / 6 := 
sorry

end rectangle_side_difference_l7_7102


namespace some_employees_not_managers_l7_7055

-- Definitions of the conditions
def isEmployee : Type := sorry
def isManager : isEmployee → Prop := sorry
def isShareholder : isEmployee → Prop := sorry
def isPunctual : isEmployee → Prop := sorry

-- Given conditions
axiom some_employees_not_punctual : ∃ e : isEmployee, ¬isPunctual e
axiom all_managers_punctual : ∀ m : isEmployee, isManager m → isPunctual m
axiom some_managers_shareholders : ∃ m : isEmployee, isManager m ∧ isShareholder m

-- The statement to be proved
theorem some_employees_not_managers : ∃ e : isEmployee, ¬isManager e :=
by sorry

end some_employees_not_managers_l7_7055


namespace sin_angle_identity_l7_7247

theorem sin_angle_identity : 
  (Real.sin (Real.pi / 4) * Real.sin (7 * Real.pi / 12) + Real.sin (Real.pi / 4) * Real.sin (Real.pi / 12)) = Real.sqrt 3 / 2 := 
by 
  sorry

end sin_angle_identity_l7_7247


namespace determinant_scaling_l7_7881

variable (p q r s : ℝ)

theorem determinant_scaling 
  (h : Matrix.det ![![p, q], ![r, s]] = 3) : 
  Matrix.det ![![2 * p, 2 * p + 5 * q], ![2 * r, 2 * r + 5 * s]] = 30 :=
by 
  sorry

end determinant_scaling_l7_7881


namespace extended_pattern_ratio_l7_7568

noncomputable def original_black_tiles : ℕ := 12
noncomputable def original_white_tiles : ℕ := 24
noncomputable def original_total_tiles : ℕ := 36
noncomputable def extended_total_tiles : ℕ := 64
noncomputable def border_black_tiles : ℕ := 24 /- The new border adds 24 black tiles -/
noncomputable def extended_black_tiles : ℕ := 36
noncomputable def extended_white_tiles := original_white_tiles

theorem extended_pattern_ratio :
  (extended_black_tiles : ℚ) / extended_white_tiles = 3 / 2 :=
by
  sorry

end extended_pattern_ratio_l7_7568


namespace solve_system_nat_l7_7336

open Nat

theorem solve_system_nat (x y z t : ℕ) :
  (x + y = z * t ∧ z + t = x * y) ↔ (x, y, z, t) = (1, 5, 2, 3) ∨ (x, y, z, t) = (2, 2, 2, 2) :=
by
  sorry

end solve_system_nat_l7_7336


namespace trihedral_angle_properties_l7_7950

-- Definitions for the problem's conditions
variables {α β γ : ℝ}
variables {A B C S : Type}
variables (angle_ASB angle_BSC angle_CSA : ℝ)

-- Given the conditions of the trihedral angle and the dihedral angles
theorem trihedral_angle_properties 
  (h1 : angle_ASB + angle_BSC + angle_CSA < 2 * Real.pi)
  (h2 : α + β + γ > Real.pi) : 
  true := 
by
  sorry

end trihedral_angle_properties_l7_7950


namespace smallest_relatively_prime_210_l7_7710

theorem smallest_relatively_prime_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ (∀ y : ℕ, y > 1 → y < x → Nat.gcd y 210 ≠ 1) :=
sorry

end smallest_relatively_prime_210_l7_7710


namespace vector_subtraction_l7_7892

def vector_a : ℝ × ℝ := (3, 5)
def vector_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction :
  vector_a - 2 • vector_b = (7, 3) :=
sorry

end vector_subtraction_l7_7892


namespace find_z_l7_7874

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l7_7874


namespace candy_problem_l7_7044

theorem candy_problem (a : ℕ) (h₁ : a % 10 = 6) (h₂ : a % 15 = 11) (h₃ : 200 ≤ a) (h₄ : a ≤ 250) :
  a = 206 ∨ a = 236 :=
sorry

end candy_problem_l7_7044


namespace max_value_4x_plus_y_l7_7483

theorem max_value_4x_plus_y (x y : ℝ) (h : 16 * x^2 + y^2 + 4 * x * y = 3) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (u : ℝ), (∃ (x y : ℝ), 16 * x^2 + y^2 + 4 * x * y = 3 ∧ u = 4 * x + y) → u ≤ M :=
by
  use 2
  sorry

end max_value_4x_plus_y_l7_7483


namespace prove_A_plus_B_l7_7616

variable (A B : ℝ)

theorem prove_A_plus_B (h : ∀ x : ℝ, x ≠ 2 → (A / (x - 2) + B * (x + 3) = (-5 * x^2 + 20 * x + 34) / (x - 2))) : A + B = 9 := by
  sorry

end prove_A_plus_B_l7_7616


namespace solve_for_z_l7_7869

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l7_7869


namespace number_of_articles_sold_at_cost_price_l7_7009

-- Let C be the cost price of one article.
-- Let S be the selling price of one article.
-- Let X be the number of articles sold at cost price.

variables (C S : ℝ) (X : ℕ)

-- Condition 1: The cost price of X articles is equal to the selling price of 32 articles.
axiom condition1 : (X : ℝ) * C = 32 * S

-- Condition 2: The profit is 25%, so the selling price S is 1.25 times the cost price C.
axiom condition2 : S = 1.25 * C

-- The theorem we need to prove
theorem number_of_articles_sold_at_cost_price : X = 40 :=
by
  -- Proof here
  sorry

end number_of_articles_sold_at_cost_price_l7_7009


namespace rectangle_area_is_12_l7_7235

noncomputable def rectangle_area_proof (w l y : ℝ) : Prop :=
  l = 3 * w ∧ 2 * (l + w) = 16 ∧ (l^2 + w^2 = y^2) → l * w = 12

theorem rectangle_area_is_12 (y : ℝ) : ∃ (w l : ℝ), rectangle_area_proof w l y :=
by
  -- Introducing variables
  exists 2
  exists 6
  -- Constructing proof steps (skipped here with sorry)
  sorry

end rectangle_area_is_12_l7_7235


namespace expression1_expression2_expression3_expression4_l7_7240

theorem expression1 : 12 - (-10) + 7 = 29 := 
by
  sorry

theorem expression2 : 1 + (-2) * abs (-2 - 3) - 5 = -14 :=
by
  sorry

theorem expression3 : (-8 * (-1 / 6 + 3 / 4 - 1 / 12)) / (1 / 6) = -24 :=
by
  sorry

theorem expression4 : -1 ^ 2 - (2 - (-2) ^ 3) / (-2 / 5) * (5 / 2) = 123 / 2 := 
by
  sorry

end expression1_expression2_expression3_expression4_l7_7240


namespace domain_of_f_l7_7246

noncomputable def f (x : ℝ) : ℝ := (x^3 - 125) / (x + 5)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≠ -5} := 
by
  sorry

end domain_of_f_l7_7246


namespace central_angle_radian_l7_7785

-- Define the context of the sector and conditions
def sector (r θ : ℝ) :=
  θ = r * 6 ∧ 1/2 * r^2 * θ = 6

-- Define the radian measure of the central angle
theorem central_angle_radian (r : ℝ) (θ : ℝ) (h : sector r θ) : θ = 3 :=
by
  sorry

end central_angle_radian_l7_7785


namespace child_ticket_cost_l7_7330

/-- Defining the conditions and proving the cost of a child's ticket --/
theorem child_ticket_cost:
  (∀ c: ℕ, 
      -- Revenue from Monday
      (7 * c + 5 * 4) + 
      -- Revenue from Tuesday
      (4 * c + 2 * 4) = 
      -- Total revenue for both days
      61 
    ) → 
    -- Proving c
    (c = 3) :=
by
  sorry

end child_ticket_cost_l7_7330


namespace problem_distribution_count_l7_7408

theorem problem_distribution_count : 12^6 = 2985984 := 
by
  sorry

end problem_distribution_count_l7_7408


namespace interest_rate_for_4000_investment_l7_7376

theorem interest_rate_for_4000_investment
      (total_money : ℝ := 9000)
      (invested_at_9_percent : ℝ := 5000)
      (total_interest : ℝ := 770)
      (invested_at_unknown_rate : ℝ := 4000) :
  ∃ r : ℝ, invested_at_unknown_rate * r = total_interest - (invested_at_9_percent * 0.09) ∧ r = 0.08 :=
by {
  -- Proof is not required based on instruction, so we use sorry.
  sorry
}

end interest_rate_for_4000_investment_l7_7376


namespace find_n_from_binomial_terms_l7_7439

theorem find_n_from_binomial_terms (x a : ℕ) (n : ℕ) 
  (h1 : n.choose 1 * x^(n-1) * a = 56) 
  (h2 : n.choose 2 * x^(n-2) * a^2 = 168) 
  (h3 : n.choose 3 * x^(n-3) * a^3 = 336) : 
  n = 5 :=
by
  sorry

end find_n_from_binomial_terms_l7_7439


namespace trig_identity_proof_l7_7559

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def sin_30 := Real.sin (Real.pi / 6)
noncomputable def cos_60 := Real.cos (Real.pi / 3)

theorem trig_identity_proof :
  (1 - (1 / cos_30)) * (1 + (2 / sin_60)) * (1 - (1 / sin_30)) * (1 + (2 / cos_60)) = (25 - 10 * Real.sqrt 3) / 3 := by
  sorry

end trig_identity_proof_l7_7559


namespace drinking_problem_solution_l7_7737

def drinking_rate (name : String) (hours : ℕ) (total_liters : ℕ) : ℚ :=
  total_liters / hours

def total_wine_consumed_in_x_hours (x : ℚ) :=
  x * (
  drinking_rate "assistant1" 12 40 +
  drinking_rate "assistant2" 10 40 +
  drinking_rate "assistant3" 8 40
  )

theorem drinking_problem_solution : 
  (∃ x : ℚ, total_wine_consumed_in_x_hours x = 40) →
  ∃ x : ℚ, x = 120 / 37 :=
by 
  sorry

end drinking_problem_solution_l7_7737


namespace add_num_denom_fraction_l7_7524

theorem add_num_denom_fraction (n : ℚ) : (2 + n) / (7 + n) = 3 / 5 ↔ n = 11 / 2 := 
by
  sorry

end add_num_denom_fraction_l7_7524


namespace max_remainder_l7_7043

theorem max_remainder : ∃ n r, 2013 ≤ n ∧ n ≤ 2156 ∧ (n % 5 = r) ∧ (n % 11 = r) ∧ (n % 13 = r) ∧ (r ≤ 4) ∧ (∀ m, 2013 ≤ m ∧ m ≤ 2156 ∧ (m % 5 = r) ∧ (m % 11 = r) ∧ (m % 13 = r) ∧ (m ≤ n) ∧ (r ≤ 4) → r ≤ 4) := sorry

end max_remainder_l7_7043


namespace fraction_meaningful_l7_7969

-- Define the condition for the fraction being meaningful
def denominator_not_zero (x : ℝ) : Prop := x + 1 ≠ 0

-- Define the statement to be proved
theorem fraction_meaningful (x : ℝ) : denominator_not_zero x ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_l7_7969


namespace indistinguishable_balls_boxes_l7_7284

open Finset

def partitions (n : ℕ) (k : ℕ) : ℕ :=
  (univ : Finset (Multiset ℕ)).filter (λ p, p.sum = n ∧ p.card ≤ k).card

theorem indistinguishable_balls_boxes : partitions 6 4 = 9 :=
sorry

end indistinguishable_balls_boxes_l7_7284


namespace observation_count_l7_7516

theorem observation_count (n : ℤ) (mean_initial : ℝ) (erroneous_value correct_value : ℝ) (mean_corrected : ℝ) :
  mean_initial = 36 →
  erroneous_value = 20 →
  correct_value = 34 →
  mean_corrected = 36.45 →
  n ≥ 0 →
  ∃ n : ℤ, (n * mean_initial + (correct_value - erroneous_value) = n * mean_corrected) ∧ (n = 31) :=
by
  intros h1 h2 h3 h4 h5
  use 31
  sorry

end observation_count_l7_7516


namespace candy_problem_solution_l7_7048

theorem candy_problem_solution :
  ∃ (a : ℕ), a % 10 = 6 ∧ a % 15 = 11 ∧ 200 ≤ a ∧ a ≤ 250 ∧ (a = 206 ∨ a = 236) :=
begin
  sorry
end

end candy_problem_solution_l7_7048


namespace supply_without_leak_last_for_20_days_l7_7816

variable (C V : ℝ)

-- Condition 1: if there is a 10-liter leak per day, the supply lasts for 15 days
axiom h1 : C = 15 * (V + 10)

-- Condition 2: if there is a 20-liter leak per day, the supply lasts for 12 days
axiom h2 : C = 12 * (V + 20)

-- The problem to prove: without any leak, the tank can supply water to the village for 20 days
theorem supply_without_leak_last_for_20_days (C V : ℝ) (h1 : C = 15 * (V + 10)) (h2 : C = 12 * (V + 20)) : C / V = 20 := 
by 
  sorry

end supply_without_leak_last_for_20_days_l7_7816


namespace peter_walks_more_time_l7_7144

-- Define the total distance Peter has to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def distance_walked : ℝ := 1.0

-- Define Peter's walking pace in minutes per mile
def walking_pace : ℝ := 20.0

-- Prove that Peter has to walk 30 more minutes to reach the grocery store
theorem peter_walks_more_time : walking_pace * (total_distance - distance_walked) = 30 :=
by
  sorry

end peter_walks_more_time_l7_7144


namespace syllogistic_reasoning_problem_l7_7184

theorem syllogistic_reasoning_problem
  (H1 : ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I)
  (H2 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.re z = 2)
  (H3 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.im z = 3) :
  (¬ ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I) → "The conclusion is wrong due to the incorrect major premise" = "A" :=
sorry

end syllogistic_reasoning_problem_l7_7184


namespace line_passing_quadrants_l7_7350

theorem line_passing_quadrants (a k : ℝ) (a_nonzero : a ≠ 0)
  (x1 x2 y1 y2 : ℝ) (hx1 : y1 = a * x1^2 - a) (hx2 : y2 = a * x2^2 - a)
  (hx1_y1 : y1 = k * x1) (hx2_y2 : y2 = k * x2) 
  (sum_x : x1 + x2 < 0) : 
  ∃ (q1 q4 : (ℝ × ℝ)), 
  (q1.1 > 0 ∧ q1.2 > 0 ∧ q1.2 = a * q1.1 + k) ∧ (q4.1 > 0 ∧ q4.2 < 0 ∧ q4.2 = a * q4.1 + k) := 
sorry

end line_passing_quadrants_l7_7350


namespace find_m_l7_7053

def is_good (n : ℤ) : Prop :=
  ¬ (∃ k : ℤ, |n| = k^2)

theorem find_m (m : ℤ) : (m % 4 = 3) → 
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_good a ∧ is_good b ∧ is_good c ∧ (a * b * c) % 2 = 1 ∧ a + b + c = m) :=
sorry

end find_m_l7_7053


namespace unoccupied_cylinder_volume_l7_7800

theorem unoccupied_cylinder_volume (r h : ℝ) (V_cylinder V_cone : ℝ) :
  r = 15 ∧ h = 30 ∧ V_cylinder = π * r^2 * h ∧ V_cone = (1/3) * π * r^2 * (r / 2) →
  V_cylinder - 2 * V_cone = 4500 * π :=
by
  intros h1
  sorry

end unoccupied_cylinder_volume_l7_7800


namespace ratio_x_y_z_l7_7678

variables (x y z : ℝ)

theorem ratio_x_y_z (h1 : 0.60 * x = 0.30 * y) 
                    (h2 : 0.80 * z = 0.40 * x) 
                    (h3 : z = 2 * y) : 
                    x / y = 4 ∧ y / y = 1 ∧ z / y = 2 :=
by
  sorry

end ratio_x_y_z_l7_7678


namespace determinant_of_matrixA_l7_7560

def matrixA : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![3, 0, -2],
  ![5, 6, -4],
  ![1, 3, 7]
]

theorem determinant_of_matrixA : Matrix.det matrixA = 144 := by
  sorry

end determinant_of_matrixA_l7_7560


namespace andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l7_7413

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that André wins the book is 1/4. -/
theorem andre_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let probability := (black_balls : ℚ) / total_balls
  probability = 1 / 4 := 
by 
  sorry

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that Dalva wins the book is 1/4. -/
theorem dalva_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let andre_white := (3 / 4 : ℚ)
  let bianca_white := (2 / 3 : ℚ)
  let carlos_white := (1 / 2 : ℚ)
  let probability := andre_white * bianca_white * carlos_white * (black_balls / (total_balls - 3))
  probability = 1 / 4 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that André wins the book is 5/14. -/
theorem andre_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_first_black := (black_balls : ℚ) / total_balls
  let andre_fifth_black := (((6 / 8 : ℚ) * (5 / 7 : ℚ) * (4 / 6 : ℚ) * (3 / 5 : ℚ)) * black_balls / (total_balls - 4))
  let probability := andre_first_black + andre_fifth_black
  probability = 5 / 14 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that Dalva wins the book is 1/7. -/
theorem dalva_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_white := (6 / 8 : ℚ)
  let bianca_white := (5 / 7 : ℚ)
  let carlos_white := (4 / 6 : ℚ)
  let dalva_black := (black_balls / (total_balls - 3))
  let probability := andre_white * bianca_white * carlos_white * dalva_black
  probability = 1 / 7 := 
by 
  sorry

end andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l7_7413


namespace vincent_rope_length_l7_7649

def rope_length : Nat := 72
def pieces_count : Nat := 12
def shortened_length : Nat := 1
def tied_pieces : Nat := 3

theorem vincent_rope_length : 
  let piece_length := rope_length / pieces_count
  let shortened_piece_length := piece_length - shortened_length
  let final_length := shortened_piece_length * tied_pieces
  final_length = 15 := by
  let piece_length := rope_length / pieces_count
  let shortened_piece_length := piece_length - shortened_length
  let final_length := shortened_piece_length * tied_pieces
  show final_length = 15 from sorry

end vincent_rope_length_l7_7649


namespace smallest_angle_between_a_c_l7_7924

noncomputable def a : EuclideanSpace ℝ (Fin 3) := ![2, 0, 0]
noncomputable def b : EuclideanSpace ℝ (Fin 3) := ![1, 0, 0]
noncomputable def c : EuclideanSpace ℝ (Fin 3) := ![1, 1, √5 - 1]

noncomputable def k : EuclideanSpace ℝ (Fin 3) := ![2, -1, 0]

theorem smallest_angle_between_a_c :
  ∥a∥ = 2 →
  ∥b∥ = 2 →
  ∥c∥ = 3 →
  (a × (a × c) + b = k) →
  (∀ θ: ℝ, θ = 30) :=
by
  intros h1 h2 h3 h4
  sorry

end smallest_angle_between_a_c_l7_7924


namespace system_of_equations_correct_l7_7305

theorem system_of_equations_correct (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
begin
  -- sorry, proof placeholder
  sorry
end

end system_of_equations_correct_l7_7305


namespace sqrt_product_eq_l7_7829

theorem sqrt_product_eq : Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end sqrt_product_eq_l7_7829


namespace expected_lifetime_flashlight_l7_7321

section
variables {Ω : Type} [ProbabilitySpace Ω]
variables (ξ η : Ω → ℝ)
variables (h_ξ_expect : E[ξ] = 2)

-- Define the minimum of ξ and η
def min_ξ_η (ω : Ω) : ℝ := min (ξ ω) (η ω)

theorem expected_lifetime_flashlight : E[min_ξ_η ξ η] ≤ 2 :=
by
  sorry
end

end expected_lifetime_flashlight_l7_7321


namespace minimum_value_expression_l7_7127

theorem minimum_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ z, (z = a^2 + b^2 + 1 / a^2 + 2 * b / a) ∧ z ≥ 2 :=
sorry

end minimum_value_expression_l7_7127


namespace eq_squares_diff_l7_7428

theorem eq_squares_diff {x y z : ℝ} :
  x = (y - z)^2 ∧ y = (x - z)^2 ∧ z = (x - y)^2 →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 1) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end eq_squares_diff_l7_7428


namespace min_value_z_l7_7379

theorem min_value_z : ∀ (x y : ℝ), ∃ z, z = 3 * x^2 + y^2 + 12 * x - 6 * y + 40 ∧ z = 19 :=
by
  intro x y
  use 3 * x^2 + y^2 + 12 * x - 6 * y + 40 -- Define z
  sorry -- Proof is skipped for now

end min_value_z_l7_7379


namespace complete_square_form_l7_7029

theorem complete_square_form (x : ℝ) (a : ℝ) 
  (h : x^2 - 2 * x - 4 = 0) : (x - 1)^2 = a ↔ a = 5 :=
by
  sorry

end complete_square_form_l7_7029


namespace min_diff_between_y_and_x_l7_7912

theorem min_diff_between_y_and_x (x y z : ℤ)
    (h1 : x < y)
    (h2 : y < z)
    (h3 : Even x)
    (h4 : Odd y)
    (h5 : Odd z)
    (h6 : z - x = 9) :
    y - x = 1 := 
  by sorry

end min_diff_between_y_and_x_l7_7912


namespace hiker_speed_correct_l7_7811

variable (hikerSpeed : ℝ)
variable (cyclistSpeed : ℝ := 15)
variable (cyclistTravelTime : ℝ := 5 / 60)  -- Converted 5 minutes to hours
variable (hikerCatchUpTime : ℝ := 13.75 / 60)  -- Converted 13.75 minutes to hours
variable (cyclistDistance : ℝ := cyclistSpeed * cyclistTravelTime)

theorem hiker_speed_correct :
  (hikerSpeed * hikerCatchUpTime = cyclistDistance) →
  hikerSpeed = 60 / 11 :=
by
  intro hiker_eq_cyclist_distance
  sorry

end hiker_speed_correct_l7_7811


namespace classA_wins_championship_distribution_expectation_scoreB_l7_7361

-- Define the probabilities of winning each game for Class A
def probBasketballA : ℝ := 0.4
def probSoccerA : ℝ := 0.8
def probBadmintonA : ℝ := 0.6

-- Define the conditions of independence
variable (indepEvents : indep_event probBasketballA probSoccerA probBadmintonA)

-- Define the point scoring system
def points_win := 8
def points_loss := 0

-- Define the total probability calculation for Class A winning at least two events
noncomputable def probClassAWinChampionship : ℝ := 
  probBasketballA * probSoccerA * probBadmintonA + -- All three events
  (1 - probBasketballA) * probSoccerA * probBadmintonA + -- Soccer and Badminton
  probBasketballA * (1 - probSoccerA) * probBadmintonA + -- Basketball and Badminton
  probBasketballA * probSoccerA * (1 - probBadmintonA) -- Basketball and Soccer

-- Verify the expected probability
theorem classA_wins_championship : probClassAWinChampionship = 0.656 :=
by
  unfold probClassAWinChampionship
  exact sorry

-- Distribution table for Class B's total score X
def probScore0 : ℝ := 0.4 * 0.8 * 0.6
def probScore10 : ℝ := 0.6 * 0.8 * 0.6 + 0.4 * 0.2 * 0.6 + 0.4 * 0.8 * 0.4
def probScore20 : ℝ := 0.6 * 0.2 * 0.6 + 0.6 * 0.8 * 0.4 + 0.4 * 0.2 * 0.4
def probScore30 : ℝ := 0.6 * 0.2 * 0.4

theorem distribution_expectation_scoreB : 
  ∑ p in [0, 10, 20, 30], p * probability_score p = 12 :=
by
  unfold probability_score
  exact sorry

end classA_wins_championship_distribution_expectation_scoreB_l7_7361


namespace min_value_k_l7_7752

variables (x : ℕ → ℚ) (k n c : ℚ)

theorem min_value_k
  (k_gt_one : k > 1) -- condition that k > 1
  (n_gt_2018 : n > 2018) -- condition that n > 2018
  (n_odd : n % 2 = 1) -- condition that n is odd
  (non_zero_rational : ∀ i : ℕ, x i ≠ 0) -- non-zero rational numbers x₁, x₂, ..., xₙ
  (not_all_equal : ∃ i j : ℕ, x i ≠ x j) -- they are not all equal
  (relations : ∀ i : ℕ, x i + k / x (i + 1) = c) -- given relations
  : k = 4 :=
sorry

end min_value_k_l7_7752


namespace solve_for_z_l7_7859

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l7_7859


namespace solution_set_of_inequality_l7_7640

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_inequality 
  (hf_even : ∀ x : ℝ, f x = f (|x|))
  (hf_increasing : ∀ x y : ℝ, x < y → x < 0 → y < 0 → f x < f y)
  (hf_value : f 3 = 1) :
  {x : ℝ | f (x - 1) < 1} = {x : ℝ | x > 4 ∨ x < -2} := 
sorry

end solution_set_of_inequality_l7_7640


namespace find_z_l7_7868

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l7_7868


namespace candy_problem_solution_l7_7049

theorem candy_problem_solution :
  ∃ (a : ℕ), a % 10 = 6 ∧ a % 15 = 11 ∧ 200 ≤ a ∧ a ≤ 250 ∧ (a = 206 ∨ a = 236) :=
begin
  sorry
end

end candy_problem_solution_l7_7049


namespace num_distinct_prime_factors_330_l7_7276

theorem num_distinct_prime_factors_330 : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, Nat.Prime x ∧ 330 % x = 0 := 
sorry

end num_distinct_prime_factors_330_l7_7276


namespace find_x_when_parallel_l7_7925

-- Given vectors
def a : ℝ × ℝ := (-2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Conditional statement: parallel vectors
def parallel_vectors (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

-- Proof statement
theorem find_x_when_parallel (x : ℝ) (h : parallel_vectors a (b x)) : x = 1 := 
  sorry

end find_x_when_parallel_l7_7925


namespace fiona_reaches_goal_l7_7795

-- Define the set of lily pads
def pads : Finset ℕ := Finset.range 15

-- Define the start, predator, and goal pads
def start_pad : ℕ := 0
def predator_pads : Finset ℕ := {4, 8}
def goal_pad : ℕ := 13

-- Define the hop probabilities
def hop_next : ℚ := 1/3
def hop_two : ℚ := 1/3
def hop_back : ℚ := 1/3

-- Define the transition probabilities (excluding jumps to negative pads)
def transition (current next : ℕ) : ℚ :=
  if next = current + 1 ∨ next = current + 2 ∨ (next = current - 1 ∧ current > 0)
  then 1/3 else 0

-- Define the function to check if a pad is safe
def is_safe (pad : ℕ) : Prop := ¬ (pad ∈ predator_pads)

-- Define the probability that Fiona reaches pad 13 without landing on 4 or 8
noncomputable def probability_reach_13 : ℚ :=
  -- Function to recursively calculate the probability
  sorry

-- Statement to prove
theorem fiona_reaches_goal : probability_reach_13 = 16 / 177147 := 
sorry

end fiona_reaches_goal_l7_7795


namespace divisible_by_six_l7_7096

theorem divisible_by_six (n a b : ℕ) (h1 : 2^n = 10 * a + b) (h2 : n > 3) (h3 : b > 0) (h4 : b < 10) : 6 ∣ (a * b) := 
sorry

end divisible_by_six_l7_7096


namespace probability_of_sine_inequality_l7_7719

open Set Real

noncomputable def probability_sine_inequality (x : ℝ) : Prop :=
  ∃ (μ : MeasureTheory.Measure ℝ), μ (Ioc (-3) 3) = 1 ∧
    μ {x | sin (π / 6 * x) ≥ 1 / 2} = 1 / 3

theorem probability_of_sine_inequality : probability_sine_inequality x :=
by
  sorry

end probability_of_sine_inequality_l7_7719


namespace change_received_l7_7131

-- Define the given conditions
def num_apples : ℕ := 5
def cost_per_apple : ℝ := 0.75
def amount_paid : ℝ := 10.00

-- Prove the change is equal to $6.25
theorem change_received :
  amount_paid - (num_apples * cost_per_apple) = 6.25 :=
by
  sorry

end change_received_l7_7131


namespace temperature_decrease_l7_7908

theorem temperature_decrease (rise_1_degC : ℝ) (decrease_2_degC : ℝ) 
  (h : rise_1_degC = 1) : decrease_2_degC = -2 :=
by 
  -- This is the statement with the condition and problem to be proven:
  sorry

end temperature_decrease_l7_7908


namespace can_cut_one_more_square_l7_7984

theorem can_cut_one_more_square (G : Finset (Fin 29 × Fin 29)) (hG : G.card = 99) :
  (∃ S : Finset (Fin 29 × Fin 29), S.card = 4 ∧ (S ⊆ G) ∧ (∀ s1 s2 : Fin 29 × Fin 29, s1 ∈ S → s2 ∈ S → s1 ≠ s2 → (|s1.1 - s2.1| > 2 ∨ |s1.2 - s2.2| > 2))) :=
sorry

end can_cut_one_more_square_l7_7984


namespace solve_inequality_l7_7249

theorem solve_inequality :
  {x : ℝ | (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 2)} =
  {x : ℝ | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)} :=
by
  sorry

end solve_inequality_l7_7249


namespace angles_identity_l7_7448
open Real

theorem angles_identity (α β : ℝ) (hα : 0 < α ∧ α < (π / 2)) (hβ : 0 < β ∧ β < (π / 2))
  (h1 : 3 * (sin α)^2 + 2 * (sin β)^2 = 1)
  (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end angles_identity_l7_7448


namespace p_is_contradictory_to_q_l7_7268

variable (a : ℝ)

def p := a > 0 → a^2 ≠ 0
def q := a ≤ 0 → a^2 = 0

theorem p_is_contradictory_to_q : (p a) ↔ ¬ (q a) :=
by
  sorry

end p_is_contradictory_to_q_l7_7268


namespace david_reading_time_l7_7062

def total_time : ℕ := 180
def math_homework : ℕ := 25
def spelling_homework : ℕ := 30
def history_assignment : ℕ := 20
def science_project : ℕ := 15
def piano_practice : ℕ := 30
def study_breaks : ℕ := 2 * 10

def time_other_activities : ℕ := math_homework + spelling_homework + history_assignment + science_project + piano_practice + study_breaks

theorem david_reading_time : total_time - time_other_activities = 40 :=
by
  -- Calculation steps would go here, not provided for the theorem statement.
  sorry

end david_reading_time_l7_7062


namespace mixed_operations_with_decimals_false_l7_7840

-- Definitions and conditions
def operations_same_level_with_decimals : Prop :=
  ∀ (a b c : ℝ), a + b - c = (a + b) - c

def calculate_left_to_right_with_decimals : Prop :=
  ∀ (a b c : ℝ), (a - b + c) = a - b + c ∧ (a + b - c) = a + b - c

-- Proposition we're proving
theorem mixed_operations_with_decimals_false :
  ¬ ∀ (a b c : ℝ), (a + b - c) ≠ (a - b + c) :=
by
  intro h
  sorry

end mixed_operations_with_decimals_false_l7_7840


namespace number_of_distinct_triangles_l7_7729

-- Definition of the grid
def grid_points : List (ℕ × ℕ) := 
  [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)]

-- Definition involving combination logic
def binomial (n k : ℕ) : ℕ := n.choose k

-- Count all possible combinations of 3 points
def total_combinations : ℕ := binomial 8 3

-- Count the degenerate cases (collinear points) in the grid
def degenerate_cases : ℕ := 2 * binomial 4 3

-- The required value of distinct triangles
def distinct_triangles : ℕ := total_combinations - degenerate_cases

theorem number_of_distinct_triangles :
  distinct_triangles = 48 :=
by
  sorry

end number_of_distinct_triangles_l7_7729


namespace max_remainder_l7_7533

-- Definition of the problem
def max_remainder_condition (x : ℕ) (y : ℕ) : Prop :=
  x % 7 = y

theorem max_remainder (y : ℕ) :
  (max_remainder_condition (7 * 102 + y) y ∧ y < 7) → (y = 6 ∧ 7 * 102 + 6 = 720) :=
by
  sorry

end max_remainder_l7_7533


namespace total_distance_l7_7656

theorem total_distance (D : ℕ) 
  (h1 : (1 / 2 * D : ℝ) + (1 / 4 * (1 / 2 * D : ℝ)) + 105 = D) : 
  D = 280 :=
by
  sorry

end total_distance_l7_7656


namespace john_speed_when_runs_alone_l7_7749

theorem john_speed_when_runs_alone (x : ℝ) : 
  (6 * (1/2) + x * (1/2) = 5) → x = 4 :=
by
  intro h
  linarith

end john_speed_when_runs_alone_l7_7749


namespace no_ordered_triples_exist_l7_7897

theorem no_ordered_triples_exist :
  ¬ ∃ (x y z : ℤ), 
    (x^2 - 3 * x * y + 2 * y^2 - z^2 = 39) ∧
    (-x^2 + 6 * y * z + 2 * z^2 = 40) ∧
    (x^2 + x * y + 8 * z^2 = 96) :=
sorry

end no_ordered_triples_exist_l7_7897


namespace time_for_A_and_C_to_complete_work_l7_7214

variable (A_rate B_rate C_rate : ℝ)

theorem time_for_A_and_C_to_complete_work
  (hA : A_rate = 1 / 4)
  (hBC : 1 / 3 = B_rate + C_rate)
  (hB : B_rate = 1 / 12) :
  1 / (A_rate + C_rate) = 2 :=
by
  -- Here would be the proof logic
  sorry

end time_for_A_and_C_to_complete_work_l7_7214


namespace smallest_coprime_to_210_l7_7706

theorem smallest_coprime_to_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 210 = 1 → y ≥ x :=
by
  sorry

end smallest_coprime_to_210_l7_7706


namespace calculate_percentage_increase_l7_7415

variable (fish_first_round : ℕ) (fish_second_round : ℕ) (fish_total : ℕ) (fish_last_round : ℕ) (increase : ℚ) (percentage_increase : ℚ)

theorem calculate_percentage_increase
  (h1 : fish_first_round = 8)
  (h2 : fish_second_round = fish_first_round + 12)
  (h3 : fish_total = 60)
  (h4 : fish_last_round = fish_total - (fish_first_round + fish_second_round))
  (h5 : increase = fish_last_round - fish_second_round)
  (h6 : percentage_increase = (increase / fish_second_round) * 100) :
  percentage_increase = 60 := by
  sorry

end calculate_percentage_increase_l7_7415


namespace balloons_remaining_l7_7607

-- Define the initial conditions
def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2

-- State the theorem
theorem balloons_remaining : initial_balloons - lost_balloons = 7 := by
  -- Add the solution proof steps here
  sorry

end balloons_remaining_l7_7607


namespace quadratic_inequality_l7_7579

theorem quadratic_inequality (a : ℝ) (h : 0 ≤ a ∧ a < 4) : ∀ x : ℝ, a * x^2 - a * x + 1 > 0 :=
by
  sorry

end quadratic_inequality_l7_7579


namespace calculate_expression_l7_7674

theorem calculate_expression (x : ℕ) (h : x = 3) : x + x * x^(x - 1) = 30 := by
  rw [h]
  -- Proof steps would go here but we are including only the statement
  sorry

end calculate_expression_l7_7674


namespace arithmetic_sequence_sum_eight_l7_7793

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sum (a₁ a₈ : α) (n : α) : α := (n * (a₁ + a₈)) / 2

theorem arithmetic_sequence_sum_eight {a₄ a₅ : α} (h₄₅ : a₄ + a₅ = 10) :
  let a₁ := a₄ - 3 * ((a₅ - a₄) / 1) -- a₁ in terms of a₄ and a₅
  let a₈ := a₄ + 4 * ((a₅ - a₄) / 1) -- a₈ in terms of a₄ and a₅
  arithmetic_sum a₁ a₈ 8 = 40 :=
by
  sorry

end arithmetic_sequence_sum_eight_l7_7793


namespace molecular_weight_H_of_H2CrO4_is_correct_l7_7252

-- Define the atomic weight of hydrogen
def atomic_weight_H : ℝ := 1.008

-- Define the number of hydrogen atoms in H2CrO4
def num_H_atoms_in_H2CrO4 : ℕ := 2

-- Define the molecular weight of the compound H2CrO4
def molecular_weight_H2CrO4 : ℝ := 118

-- Define the molecular weight of the hydrogen part (H2)
def molecular_weight_H2 : ℝ := atomic_weight_H * num_H_atoms_in_H2CrO4

-- The statement to prove
theorem molecular_weight_H_of_H2CrO4_is_correct : molecular_weight_H2 = 2.016 :=
by
  sorry

end molecular_weight_H_of_H2CrO4_is_correct_l7_7252


namespace num_supervisors_correct_l7_7104

theorem num_supervisors_correct (S : ℕ) 
  (avg_sal_total : ℕ) (avg_sal_supervisor : ℕ) (avg_sal_laborer : ℕ) (num_laborers : ℕ)
  (h1 : avg_sal_total = 1250) 
  (h2 : avg_sal_supervisor = 2450) 
  (h3 : avg_sal_laborer = 950) 
  (h4 : num_laborers = 42) 
  (h5 : avg_sal_total = (39900 + S * avg_sal_supervisor) / (num_laborers + S)) : 
  S = 10 := by sorry

end num_supervisors_correct_l7_7104


namespace time_for_A_and_C_to_complete_work_l7_7213

variable (A_rate B_rate C_rate : ℝ)

theorem time_for_A_and_C_to_complete_work
  (hA : A_rate = 1 / 4)
  (hBC : 1 / 3 = B_rate + C_rate)
  (hB : B_rate = 1 / 12) :
  1 / (A_rate + C_rate) = 2 :=
by
  -- Here would be the proof logic
  sorry

end time_for_A_and_C_to_complete_work_l7_7213


namespace probability_of_rolling_2_4_6_l7_7028

def fair_eight_sided_die : ℕ := 8
def successful_outcomes : set ℕ := {2, 4, 6}
def num_successful_outcomes : ℕ := successful_outcomes.to_finset.card

theorem probability_of_rolling_2_4_6 :
  (num_successful_outcomes : ℚ) / fair_eight_sided_die = 3 / 8 :=
by
  -- Note: The proof is omitted by using 'sorry'
  sorry

end probability_of_rolling_2_4_6_l7_7028


namespace circle_center_l7_7786

theorem circle_center (x y : ℝ) :
  x^2 + y^2 - 4 * x - 2 * y - 5 = 0 → (x - 2)^2 + (y - 1)^2 = 10 :=
by sorry

end circle_center_l7_7786


namespace range_of_m_l7_7089

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x

theorem range_of_m (m : ℝ) : f m > 1 → m < 0 := by
  sorry

end range_of_m_l7_7089


namespace adamek_marbles_l7_7820

theorem adamek_marbles : ∃ n : ℕ, (∀ k : ℕ, n = 4 * k ∧ n = 3 * (k + 8)) → n = 96 :=
by
  sorry

end adamek_marbles_l7_7820


namespace find_percentage_l7_7210

noncomputable def percentage_condition (P : ℝ) : Prop :=
  9000 + (P / 100) * 9032 = 10500

theorem find_percentage (P : ℝ) (h : percentage_condition P) : P = 16.61 :=
sorry

end find_percentage_l7_7210


namespace two_b_is_16667_percent_of_a_l7_7005

theorem two_b_is_16667_percent_of_a {a b : ℝ} (h : a = 1.2 * b) : (2 * b / a) = 5 / 3 := by
  sorry

end two_b_is_16667_percent_of_a_l7_7005


namespace markeesha_sales_l7_7768

variable (Friday_sales : ℕ)
variable (Saturday_sales : ℕ)
variable (Sunday_sales : ℕ)

def Total_sales : ℕ :=
  Friday_sales + Saturday_sales + Sunday_sales

theorem markeesha_sales :
  Friday_sales = 30 →
  Saturday_sales = 2 * Friday_sales →
  Sunday_sales = Saturday_sales - 15 →
  Total_sales Friday_sales Saturday_sales Sunday_sales = 135 :=
by
  intros h1 h2 h3
  simp [Total_sales, h1, h2, h3]
  sorry

end markeesha_sales_l7_7768


namespace Fran_speed_l7_7115

-- Definitions needed for statements
def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 5

-- Formalize the problem in Lean
theorem Fran_speed (Joann_distance : ℝ) (Fran_speed : ℝ) : 
  Joann_distance = Joann_speed * Joann_time →
  Fran_speed * Fran_time = Joann_distance →
  Fran_speed = 12 :=
by
  -- assume the conditions about distances
  intros h1 h2
  -- prove the goal
  sorry

end Fran_speed_l7_7115


namespace ferry_travel_time_l7_7855

theorem ferry_travel_time:
  ∀ (v_P v_Q : ℝ) (d_P d_Q : ℝ) (t_P t_Q : ℝ),
    v_P = 8 →
    v_Q = v_P + 1 →
    d_Q = 3 * d_P →
    t_Q = t_P + 5 →
    d_P = v_P * t_P →
    d_Q = v_Q * t_Q →
    t_P = 3 := by
  sorry

end ferry_travel_time_l7_7855


namespace quadratic_expression_rewrite_l7_7352

theorem quadratic_expression_rewrite (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) → a + b + c = 171 :=
sorry

end quadratic_expression_rewrite_l7_7352


namespace tanya_work_days_l7_7776

theorem tanya_work_days (days_sakshi : ℕ) (efficiency_increase : ℚ) (work_rate_sakshi : ℚ) (work_rate_tanya : ℚ) (days_tanya : ℚ) :
  days_sakshi = 15 ->
  efficiency_increase = 1.25 ->
  work_rate_sakshi = 1 / days_sakshi ->
  work_rate_tanya = work_rate_sakshi * efficiency_increase ->
  days_tanya = 1 / work_rate_tanya ->
  days_tanya = 12 :=
by
  intros h_sakshi h_efficiency h_work_rate_sakshi h_work_rate_tanya h_days_tanya
  sorry

end tanya_work_days_l7_7776


namespace fence_poles_count_l7_7040

def length_path : ℕ := 900
def length_bridge : ℕ := 42
def distance_between_poles : ℕ := 6

theorem fence_poles_count :
  2 * (length_path - length_bridge) / distance_between_poles = 286 :=
by
  sorry

end fence_poles_count_l7_7040


namespace evaluate_expression_at_2_l7_7382

theorem evaluate_expression_at_2 : (3^2 - 2^3) = 1 := 
by
  sorry

end evaluate_expression_at_2_l7_7382


namespace probability_of_exactly_9_correct_matches_is_zero_l7_7201

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end probability_of_exactly_9_correct_matches_is_zero_l7_7201


namespace probability_of_exactly_9_correct_matches_is_zero_l7_7200

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end probability_of_exactly_9_correct_matches_is_zero_l7_7200


namespace pythagorean_triple_divisible_by_60_l7_7310

theorem pythagorean_triple_divisible_by_60 
  (a b c : ℕ) (h : a * a + b * b = c * c) : 60 ∣ (a * b * c) :=
sorry

end pythagorean_triple_divisible_by_60_l7_7310


namespace theater_ticket_area_l7_7401

theorem theater_ticket_area
  (P width : ℕ)
  (hP : P = 28)
  (hwidth : width = 6)
  (length : ℕ)
  (hlength : 2 * (length + width) = P) :
  length * width = 48 :=
by
  sorry

end theater_ticket_area_l7_7401


namespace total_area_of_colored_paper_l7_7895

-- Definitions
def num_pieces : ℝ := 3.2
def side_length : ℝ := 8.5

-- Theorem statement
theorem total_area_of_colored_paper : 
  let area_one_piece := side_length * side_length
  let total_area := area_one_piece * num_pieces
  total_area = 231.2 := by
  sorry

end total_area_of_colored_paper_l7_7895


namespace ratio_of_area_of_small_triangle_to_square_l7_7517

theorem ratio_of_area_of_small_triangle_to_square
  (n : ℕ)
  (square_area : ℝ)
  (A1 : square_area > 0)
  (ADF_area : ℝ)
  (H1 : ADF_area = n * square_area)
  (FEC_area : ℝ)
  (H2 : FEC_area = 1 / (4 * n)) :
  FEC_area / square_area = 1 / (4 * n) :=
by
  sorry

end ratio_of_area_of_small_triangle_to_square_l7_7517


namespace number_without_daughters_l7_7486

-- Given conditions
def Marilyn_daughters : Nat := 10
def total_women : Nat := 40
def daughters_with_daughters_women_have_each : Nat := 5

-- Helper definition representing the computation of granddaughters
def Marilyn_granddaughters : Nat := total_women - Marilyn_daughters

-- Proving the main statement
theorem number_without_daughters : 
  (Marilyn_daughters - (Marilyn_granddaughters / daughters_with_daughters_women_have_each)) + Marilyn_granddaughters = 34 := by
  sorry

end number_without_daughters_l7_7486


namespace fraction_expression_of_repeating_decimal_l7_7653

theorem fraction_expression_of_repeating_decimal :
  ∃ (x : ℕ), x = 79061333 ∧ (∀ y : ℚ, y = 0.71 + 264 * (1/999900) → x / 999900 = y) :=
by
  sorry

end fraction_expression_of_repeating_decimal_l7_7653


namespace LineChart_characteristics_and_applications_l7_7935

-- Definitions related to question and conditions
def LineChart : Type := sorry
def represents_amount (lc : LineChart) : Prop := sorry
def reflects_increase_or_decrease (lc : LineChart) : Prop := sorry

-- Theorem related to the correct answer
theorem LineChart_characteristics_and_applications (lc : LineChart) :
  represents_amount lc ∧ reflects_increase_or_decrease lc :=
sorry

end LineChart_characteristics_and_applications_l7_7935


namespace trajectory_of_G_l7_7085

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop :=
  9 * x^2 / 4 + 3 * y^2 = 1

-- State the theorem
theorem trajectory_of_G (P G : ℝ × ℝ) (hP : ellipse P.1 P.2) (hG_relation : ∃ k : ℝ, k = 2 ∧ P = (3 * G.1, 3 * G.2)) :
  trajectory G.1 G.2 :=
by
  sorry

end trajectory_of_G_l7_7085


namespace birds_nest_building_area_scientific_notation_l7_7354

theorem birds_nest_building_area_scientific_notation :
  (258000 : ℝ) = 2.58 * 10^5 :=
by sorry

end birds_nest_building_area_scientific_notation_l7_7354


namespace expression_value_l7_7025

theorem expression_value (x y z : ℤ) (h1 : x = 25) (h2 : y = 30) (h3 : z = 7) :
  (x - (y - z)) - ((x - y) - (z - 1)) = 13 :=
by
  sorry

end expression_value_l7_7025


namespace integral_sign_l7_7064

noncomputable def I : ℝ := ∫ x in -Real.pi..0, Real.sin x

theorem integral_sign : I < 0 := sorry

end integral_sign_l7_7064


namespace x_of_x35x_div_by_18_l7_7437

theorem x_of_x35x_div_by_18 (x : ℕ) (h₁ : 18 = 2 * 9) (h₂ : (2 * x + 8) % 9 = 0) (h₃ : ∃ k : ℕ, x = 2 * k) : x = 8 :=
sorry

end x_of_x35x_div_by_18_l7_7437


namespace lilies_per_centerpiece_l7_7758

theorem lilies_per_centerpiece (centerpieces roses orchids cost total_budget price_per_flower number_of_lilies_per_centerpiece : ℕ) 
  (h0 : centerpieces = 6)
  (h1 : roses = 8)
  (h2 : orchids = 2 * roses)
  (h3 : cost = total_budget)
  (h4 : total_budget = 2700)
  (h5 : price_per_flower = 15)
  (h6 : cost = (centerpieces * roses * price_per_flower) + (centerpieces * orchids * price_per_flower) + (centerpieces * number_of_lilies_per_centerpiece * price_per_flower))
  : number_of_lilies_per_centerpiece = 6 := 
by 
  sorry

end lilies_per_centerpiece_l7_7758


namespace partitions_of_6_into_4_indistinguishable_boxes_l7_7289

theorem partitions_of_6_into_4_indistinguishable_boxes : 
  ∃ (X : Finset (Multiset ℕ)), X.card = 9 ∧ 
  ∀ p ∈ X, p.sum = 6 ∧ p.card ≤ 4 := 
sorry

end partitions_of_6_into_4_indistinguishable_boxes_l7_7289


namespace nylon_cord_length_l7_7662

-- Let the length of cord be w
-- Dog runs 30 feet forming a semicircle, that is pi * w = 30
-- Prove that w is approximately 9.55

theorem nylon_cord_length (pi_approx : Real := 3.14) : Real :=
  let w := 30 / pi_approx
  w

end nylon_cord_length_l7_7662


namespace points_per_vegetable_correct_l7_7018

-- Given conditions
def total_points_needed : ℕ := 200
def number_of_students : ℕ := 25
def number_of_weeks : ℕ := 2
def veggies_per_student_per_week : ℕ := 2

-- Derived values
def total_veggies_eaten_by_class : ℕ :=
  number_of_students * number_of_weeks * veggies_per_student_per_week

def points_per_vegetable : ℕ :=
  total_points_needed / total_veggies_eaten_by_class

-- Theorem to be proven
theorem points_per_vegetable_correct :
  points_per_vegetable = 2 := by
sorry

end points_per_vegetable_correct_l7_7018


namespace sum_of_remainders_l7_7781

theorem sum_of_remainders (a b c d e : ℕ)
  (h1 : a % 13 = 3)
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7)
  (h4 : d % 13 = 9)
  (h5 : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 :=
by {
  sorry
}

end sum_of_remainders_l7_7781


namespace ratio_to_percent_l7_7016

theorem ratio_to_percent :
  (9 / 5 * 100) = 180 :=
by
  sorry

end ratio_to_percent_l7_7016


namespace diagonals_in_polygon_of_150_sides_l7_7676

-- Definition of the number of diagonals formula
def number_of_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Given condition: the polygon has 150 sides
def n : ℕ := 150

-- Statement to prove
theorem diagonals_in_polygon_of_150_sides : number_of_diagonals n = 11025 :=
by
  sorry

end diagonals_in_polygon_of_150_sides_l7_7676


namespace steven_weight_l7_7561

theorem steven_weight (danny_weight : ℝ) (steven_more : ℝ) (steven_weight : ℝ) 
  (h₁ : danny_weight = 40) 
  (h₂ : steven_more = 0.2 * danny_weight) 
  (h₃ : steven_weight = danny_weight + steven_more) : 
  steven_weight = 48 := 
  by 
  sorry

end steven_weight_l7_7561


namespace tan_double_angle_third_quadrant_l7_7074

theorem tan_double_angle_third_quadrant
  (α : ℝ)
  (sin_alpha : Real.sin α = -3/5)
  (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.tan (2 * α) = 24 / 7 :=
sorry

end tan_double_angle_third_quadrant_l7_7074


namespace quadratic_solutions_l7_7157

theorem quadratic_solutions (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end quadratic_solutions_l7_7157


namespace snowboard_final_price_l7_7770

noncomputable def original_price : ℝ := 200
noncomputable def discount_friday : ℝ := 0.40
noncomputable def discount_monday : ℝ := 0.25

noncomputable def price_after_friday_discount (orig : ℝ) (discount : ℝ) : ℝ :=
  (1 - discount) * orig

noncomputable def final_price (price_friday : ℝ) (discount : ℝ) : ℝ :=
  (1 - discount) * price_friday

theorem snowboard_final_price :
  final_price (price_after_friday_discount original_price discount_friday) discount_monday = 90 := 
sorry

end snowboard_final_price_l7_7770


namespace sandwiches_provided_now_l7_7179

-- Define the initial number of sandwich kinds
def initialSandwichKinds : ℕ := 23

-- Define the number of sold out sandwich kinds
def soldOutSandwichKinds : ℕ := 14

-- Define the proof that the actual number of sandwich kinds provided now
theorem sandwiches_provided_now : initialSandwichKinds - soldOutSandwichKinds = 9 :=
by
  -- The proof goes here
  sorry

end sandwiches_provided_now_l7_7179


namespace inequality_solution_l7_7083

theorem inequality_solution (x : ℝ) : 3 * x ^ 2 + x - 2 < 0 ↔ -1 < x ∧ x < 2 / 3 :=
by
  -- The proof should factor the quadratic expression and apply the rule for solving strict inequalities
  sorry

end inequality_solution_l7_7083


namespace perp_bisector_eq_l7_7273

noncomputable def C1 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.1 - 7 = 0 }
noncomputable def C2 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.2 - 27 = 0 }

theorem perp_bisector_eq :
  ∃ x y, ( (x, y) ∈ C1 ∧ (x, y) ∈ C2 ) -> ( x - y = 0 ) :=
by
  sorry

end perp_bisector_eq_l7_7273


namespace ac_work_time_l7_7216

theorem ac_work_time (W : ℝ) (a_work_rate : ℝ) (b_work_rate : ℝ) (bc_work_rate : ℝ) (t : ℝ) : 
  (a_work_rate = W / 4) ∧ 
  (b_work_rate = W / 12) ∧ 
  (bc_work_rate = W / 3) → 
  t = 2 := 
by 
  sorry

end ac_work_time_l7_7216


namespace laila_utility_l7_7121

theorem laila_utility (u : ℝ) :
  (2 * u * (10 - 2 * u) = 2 * (4 - 2 * u) * (2 * u + 4)) → u = 4 := 
by 
  sorry

end laila_utility_l7_7121


namespace closest_ratio_l7_7006

noncomputable def ratio_closest_to_one (a c : ℕ) : ℚ :=
  if c = 0 then 0 else (a : ℚ) / (c : ℚ)

theorem closest_ratio :
  ∃ (a c : ℕ), 30 * a + 15 * c = 2400 ∧
              a > 0 ∧
              c > 0 ∧
              (ratio_closest_to_one a c = 27/26) :=
begin
  sorry
end

end closest_ratio_l7_7006


namespace solve_congruence_l7_7780

-- Define the initial condition of the problem
def condition (x : ℤ) : Prop := (15 * x + 3) % 21 = 9 % 21

-- The statement that we want to prove
theorem solve_congruence : ∃ (a m : ℤ), condition a ∧ a % m = 6 % 7 ∧ a < m ∧ a + m = 13 :=
by {
    sorry
}

end solve_congruence_l7_7780


namespace find_z_l7_7876

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l7_7876


namespace longest_tape_l7_7792

theorem longest_tape (r b y : ℚ) (h₀ : r = 11 / 6) (h₁ : b = 7 / 4) (h₂ : y = 13 / 8) : r > b ∧ r > y :=
by 
  sorry

end longest_tape_l7_7792


namespace bottle_caps_total_l7_7132

def initial_bottle_caps := 51.0
def given_bottle_caps := 36.0

theorem bottle_caps_total : initial_bottle_caps + given_bottle_caps = 87.0 := by
  sorry

end bottle_caps_total_l7_7132


namespace distance_between_trees_l7_7465

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ)
  (h_yard : yard_length = 400) (h_trees : num_trees = 26) : 
  (yard_length / (num_trees - 1)) = 16 :=
by
  sorry

end distance_between_trees_l7_7465


namespace weekly_income_l7_7362

-- Defining the daily catches
def blue_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 10
  | "Tuesday"   => 8
  | "Wednesday" => 12
  | "Thursday"  => 6
  | "Friday"    => 14
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

def red_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 14
  | "Tuesday"   => 16
  | "Wednesday" => 10
  | "Thursday"  => 18
  | "Friday"    => 12
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

-- Prices per crab
def price_per_blue_crab : ℕ := 6
def price_per_red_crab : ℕ := 4
def buckets : ℕ := 8

-- Daily income calculation
def daily_income (day : String) : ℕ :=
  let blue_income := (blue_crabs_per_bucket day) * buckets * price_per_blue_crab
  let red_income := (red_crabs_per_bucket day) * buckets * price_per_red_crab
  blue_income + red_income

-- Proving the weekly income is $6080
theorem weekly_income : 
  (daily_income "Monday" +
  daily_income "Tuesday" +
  daily_income "Wednesday" +
  daily_income "Thursday" +
  daily_income "Friday" +
  daily_income "Saturday" +
  daily_income "Sunday") = 6080 :=
by sorry

end weekly_income_l7_7362


namespace first_ball_red_given_second_black_l7_7372

open ProbabilityTheory

noncomputable def urn_A : Finset (Finset ℕ) := { {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 1, 2}, ... }
noncomputable def urn_B : Finset (Finset ℕ) := { {1, 1, 1, 2, 2, 2}, {1, 1, 2, 2, 2, 2}, ... }

noncomputable def prob_draw_red : ℕ := 7 / 15

theorem first_ball_red_given_second_black :
  (∑ A_Burn_selection in ({0, 1} : Finset ℕ), 
     ((∑ ball_draw from A_Burn_selection,
           if A_Burn_selection = 0 then (∑ red in urn_A, if red = 1 then 1 else 0) / 6 / 2
           else (∑ red in urn_B, if red = 1 then 1 else 0) / 6 / 2) *
     ((∑ second_urn_selection in ({0, 1} : Finset ℕ),
           if second_urn_selection = 0 and A_Burn_selection = 0 then 
              ∑ black in urn_A, if black = 1 then 1 else 0 / 6 / 2 
           else 
              ∑ black in urn_B, if black = 1 then 1 else 0 / 6 / 2))) = 7 / 15 :=
sorry

end first_ball_red_given_second_black_l7_7372


namespace students_behind_yoongi_l7_7807

theorem students_behind_yoongi (total_students : ℕ) (jungkook_position : ℕ) (yoongi_position : ℕ) (behind_students : ℕ)
  (h1 : total_students = 20)
  (h2 : jungkook_position = 3)
  (h3 : yoongi_position = jungkook_position + 1)
  (h4 : behind_students = total_students - yoongi_position) :
  behind_students = 16 :=
by
  sorry

end students_behind_yoongi_l7_7807


namespace find_P_nplus1_l7_7617

-- Conditions
def P (n : ℕ) (k : ℕ) : ℚ :=
  1 / Nat.choose n k

-- Lean 4 statement for the proof
theorem find_P_nplus1 (n : ℕ) : (if Even n then P n (n+1) = 1 else P n (n+1) = 0) := by
  sorry

end find_P_nplus1_l7_7617


namespace sequence_decreasing_l7_7971

theorem sequence_decreasing : 
  ∀ (n : ℕ), n ≥ 1 → (1 / 2^(n - 1)) > (1 / 2^n) := 
by {
  sorry
}

end sequence_decreasing_l7_7971


namespace decoration_sets_count_l7_7659

/-- 
Prove the number of different decoration sets that can be purchased for $120 dollars,
where each balloon costs $4, each ribbon costs $6, and the number of balloons must be even,
is exactly 2.
-/
theorem decoration_sets_count : 
  ∃ n : ℕ, n = 2 ∧ 
  (∃ (b r : ℕ), 
    4 * b + 6 * r = 120 ∧ 
    b % 2 = 0 ∧ 
    ∃ (i j : ℕ), 
      i ≠ j ∧ 
      (4 * i + 6 * (120 - 4 * i) / 6 = 120) ∧ 
      (4 * j + 6 * (120 - 4 * j) / 6 = 120) 
  )
:= sorry

end decoration_sets_count_l7_7659


namespace ferris_wheel_seats_l7_7340

def number_of_people_per_seat := 6
def total_number_of_people := 84

def number_of_seats := total_number_of_people / number_of_people_per_seat

theorem ferris_wheel_seats : number_of_seats = 14 := by
  sorry

end ferris_wheel_seats_l7_7340


namespace find_common_difference_l7_7309

section
variables (a1 a7 a8 a9 S5 S6 : ℚ) (d : ℚ)

/-- Given an arithmetic sequence with the sum of the first n terms S_n,
    if S_5 = a_8 + 5 and S_6 = a_7 + a_9 - 5, we need to find the common difference d. -/
theorem find_common_difference
  (h1 : S5 = a8 + 5)
  (h2 : S6 = a7 + a9 - 5)
  (h3 : S5 = 5 / 2 * (2 * a1 + 4 * d))
  (h4 : S6 = 6 / 2 * (2 * a1 + 5 * d))
  (h5 : a8 = a1 + 7 * d)
  (h6 : a7 = a1 + 6 * d)
  (h7 : a9 = a1 + 8 * d):
  d = -55 / 19 :=
by
  sorry
end

end find_common_difference_l7_7309


namespace tank_capacity_l7_7386

variable (C : ℝ)

theorem tank_capacity (h : (3/4) * C + 9 = (7/8) * C) : C = 72 :=
by
  sorry

end tank_capacity_l7_7386


namespace calculate_a_minus_b_l7_7101

theorem calculate_a_minus_b : 
  ∀ (a b : ℚ), (y = a * x + b) 
  ∧ (y = 4 ↔ x = 3) 
  ∧ (y = 22 ↔ x = 10) 
  → (a - b = 6 + 2 / 7)
:= sorry

end calculate_a_minus_b_l7_7101


namespace factor_x4_plus_16_l7_7427

theorem factor_x4_plus_16 (x : ℝ) : x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end factor_x4_plus_16_l7_7427


namespace solve_phi_l7_7433

theorem solve_phi (n : ℕ) : 
  (∃ (x y z : ℕ), 5 * x + 2 * y + z = 10 * n) → 
  (∃ (φ : ℕ), φ = 5 * n^2 + 4 * n + 1) :=
by 
  sorry

end solve_phi_l7_7433


namespace teammates_of_oliver_l7_7353

-- Define the player characteristics
structure Player :=
  (name   : String)
  (eyes   : String)
  (hair   : String)

-- Define the list of players with their given characteristics
def players : List Player := [
  {name := "Daniel", eyes := "Green", hair := "Red"},
  {name := "Oliver", eyes := "Gray", hair := "Brown"},
  {name := "Mia", eyes := "Gray", hair := "Red"},
  {name := "Ella", eyes := "Green", hair := "Brown"},
  {name := "Leo", eyes := "Green", hair := "Red"},
  {name := "Zoe", eyes := "Green", hair := "Brown"}
]

-- Define the condition for being on the same team
def same_team (p1 p2 : Player) : Bool :=
  (p1.eyes = p2.eyes && p1.hair ≠ p2.hair) || (p1.eyes ≠ p2.eyes && p1.hair = p2.hair)

-- Define the criterion to check if two players are on the same team as Oliver
def is_teammate_of_oliver (p : Player) : Bool :=
  let oliver := players[1] -- Oliver is the second player in the list
  same_team oliver p

-- Formal proof statement
theorem teammates_of_oliver : 
  is_teammate_of_oliver players[2] = true ∧ is_teammate_of_oliver players[3] = true :=
by
  -- Provide the intended proof here
  sorry

end teammates_of_oliver_l7_7353


namespace integer_solutions_of_quadratic_eq_l7_7613

theorem integer_solutions_of_quadratic_eq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ x1 x2 : ℤ, x1 * x2 = q^4 ∧ x1 + x2 = -p ∧ x1 = -1 ∧ x2 = - (q^4) ∧ p = 17 ∧ q = 2 := 
sorry

end integer_solutions_of_quadratic_eq_l7_7613


namespace equation_of_parametrized_curve_l7_7638

theorem equation_of_parametrized_curve :
  ∀ t : ℝ, let x := 3 * t + 6 
           let y := 5 * t - 8 
           ∃ (m b : ℝ), y = m * x + b ∧ m = 5 / 3 ∧ b = -18 :=
by
  sorry

end equation_of_parametrized_curve_l7_7638


namespace intersection_correct_l7_7081

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3, 4}

theorem intersection_correct : A ∩ B = {2, 3} := sorry

end intersection_correct_l7_7081


namespace temperature_difference_l7_7140

theorem temperature_difference (T_high T_low : ℝ) (h_high : T_high = 9) (h_low : T_low = -1) : 
  T_high - T_low = 10 :=
by
  rw [h_high, h_low]
  norm_num

end temperature_difference_l7_7140


namespace average_age_of_large_family_is_correct_l7_7397

def average_age_of_family 
  (num_grandparents : ℕ) (avg_age_grandparents : ℕ) 
  (num_parents : ℕ) (avg_age_parents : ℕ) 
  (num_children : ℕ) (avg_age_children : ℕ) 
  (num_siblings : ℕ) (avg_age_siblings : ℕ)
  (num_cousins : ℕ) (avg_age_cousins : ℕ)
  (num_aunts : ℕ) (avg_age_aunts : ℕ) : ℕ := 
  let total_age := num_grandparents * avg_age_grandparents + 
                   num_parents * avg_age_parents + 
                   num_children * avg_age_children + 
                   num_siblings * avg_age_siblings + 
                   num_cousins * avg_age_cousins + 
                   num_aunts * avg_age_aunts
  let total_family_members := num_grandparents + num_parents + num_children + num_siblings + num_cousins + num_aunts
  (total_age : ℕ) / total_family_members

theorem average_age_of_large_family_is_correct :
  average_age_of_family 4 67 3 41 5 8 2 35 3 22 2 45 = 35 := 
by 
  sorry

end average_age_of_large_family_is_correct_l7_7397


namespace line_bisects_circle_and_perpendicular_l7_7664

   def line_bisects_circle_and_is_perpendicular (x y : ℝ) : Prop :=
     (∃ (b : ℝ), ((2 * x - y + b = 0) ∧ (x^2 + y^2 - 2 * x - 4 * y = 0))) ∧
     ∀ b, (2 * 1 - 2 + b = 0) → b = 0 → (2 * x - y = 0)

   theorem line_bisects_circle_and_perpendicular :
     line_bisects_circle_and_is_perpendicular 1 2 :=
   by
     sorry
   
end line_bisects_circle_and_perpendicular_l7_7664


namespace expected_groups_l7_7174

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end expected_groups_l7_7174


namespace prime_factor_of_difference_l7_7688

theorem prime_factor_of_difference (A B C : ℕ) (hA : A ≠ 0) (hABC_digits : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (hA_range : 0 ≤ A ∧ A ≤ 9) (hB_range : 0 ≤ B ∧ B ≤ 9) (hC_range : 0 ≤ C ∧ C ≤ 9) :
  11 ∣ (100 * A + 10 * B + C) - (100 * C + 10 * B + A) :=
by
  sorry

end prime_factor_of_difference_l7_7688


namespace not_possible_one_lies_other_not_l7_7918

-- Variable definitions: Jean is lying (J), Pierre is lying (P)
variable (J P : Prop)

-- Conditions from the problem
def Jean_statement : Prop := P → J
def Pierre_statement : Prop := P → J

-- Theorem statement
theorem not_possible_one_lies_other_not (h1 : Jean_statement J P) (h2 : Pierre_statement J P) : ¬ ((J ∨ ¬ J) ∧ (P ∨ ¬ P) ∧ ((J ∧ ¬ P) ∨ (¬ J ∧ P))) :=
by
  sorry

end not_possible_one_lies_other_not_l7_7918


namespace polygon_sides_eq_2023_l7_7735

theorem polygon_sides_eq_2023 (n : ℕ) (h : n - 2 = 2021) : n = 2023 :=
sorry

end polygon_sides_eq_2023_l7_7735


namespace solve_for_x_l7_7445

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x : ∃ x, 2 * f x - 16 = f (x - 6) ∧ x = 1 := by
  exists 1
  sorry

end solve_for_x_l7_7445


namespace base8_satisfies_l7_7436

noncomputable def check_base (c : ℕ) : Prop := 
  ((2 * c ^ 2 + 4 * c + 3) + (1 * c ^ 2 + 5 * c + 6)) = (4 * c ^ 2 + 2 * c + 1)

theorem base8_satisfies : check_base 8 := 
by
  -- conditions: (243_c, 156_c, 421_c) translated as provided
  -- proof is skipped here as specified
  sorry

end base8_satisfies_l7_7436


namespace cell_division_relationship_l7_7525

noncomputable def number_of_cells_after_divisions (x : ℕ) : ℕ :=
  2^x

theorem cell_division_relationship (x : ℕ) : 
  number_of_cells_after_divisions x = 2^x := 
by 
  sorry

end cell_division_relationship_l7_7525


namespace sqrt_product_eq_l7_7830

theorem sqrt_product_eq : Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end sqrt_product_eq_l7_7830


namespace range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l7_7485

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < a + 1}
def setB : Set ℝ := {x : ℝ | x < -1 ∨ x > 2}

-- Question (1): Proof statement for A ∩ B = ∅ implying 0 ≤ a ≤ 1
theorem range_of_a_if_intersection_empty (a : ℝ) :
  (setA a ∩ setB = ∅) → (0 ≤ a ∧ a ≤ 1) := 
sorry

-- Question (2): Proof statement for A ∪ B = B implying a ≤ -2 or a ≥ 3
theorem range_of_a_if_union_equal_B (a : ℝ) :
  (setA a ∪ setB = setB) → (a ≤ -2 ∨ 3 ≤ a) := 
sorry

end range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l7_7485


namespace triangle_ABC_properties_l7_7261

noncomputable def is_arithmetic_sequence (α β γ : ℝ) : Prop :=
γ - β = β - α

theorem triangle_ABC_properties
  (A B C a c : ℝ)
  (b : ℝ := Real.sqrt 3)
  (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B) :
  is_arithmetic_sequence A B C ∧
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / 4 := by sorry

end triangle_ABC_properties_l7_7261


namespace number_of_descending_digit_numbers_l7_7849

theorem number_of_descending_digit_numbers : 
  (∑ k in Finset.range 8, Nat.choose 10 (k + 2)) + 1 = 1013 :=
by
  sorry

end number_of_descending_digit_numbers_l7_7849


namespace solve_for_x_l7_7065

theorem solve_for_x : ∀ x : ℝ, ( (x * x^(2:ℝ)) ^ (1/6) )^2 = 4 → x = 4 := by
  intro x
  sorry

end solve_for_x_l7_7065


namespace investment_amount_l7_7410

noncomputable def total_investment (A T : ℝ) : Prop :=
  (0.095 * T = 0.09 * A + 2750) ∧ (T = A + 25000)

theorem investment_amount :
  ∃ T, ∀ A, total_investment A T ∧ T = 100000 :=
by
  sorry

end investment_amount_l7_7410


namespace eq_system_correct_l7_7303

theorem eq_system_correct (x y : ℤ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
sorry

end eq_system_correct_l7_7303


namespace smallest_relatively_prime_210_l7_7713

theorem smallest_relatively_prime_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ (∀ y : ℕ, y > 1 → y < x → Nat.gcd y 210 ≠ 1) :=
sorry

end smallest_relatively_prime_210_l7_7713


namespace upper_bound_of_n_l7_7123

theorem upper_bound_of_n (m n : ℕ) (h_m : m ≥ 2)
  (h_div : ∀ a : ℕ, gcd a n = 1 → n ∣ a^m - 1) : 
  n ≤ 4 * m * (2^m - 1) := 
sorry

end upper_bound_of_n_l7_7123


namespace jones_elementary_school_students_l7_7987

theorem jones_elementary_school_students
  (X : ℕ)
  (boys_percent_total : ℚ)
  (num_students_represented : ℕ)
  (percent_of_boys : ℚ)
  (h1 : boys_percent_total = 0.60)
  (h2 : num_students_represented = 90)
  (h3 : percent_of_boys * (boys_percent_total * X) = 90)
  : X = 150 :=
by
  sorry

end jones_elementary_school_students_l7_7987


namespace allocation_first_grade_places_l7_7742

theorem allocation_first_grade_places (total_students : ℕ)
                                      (ratio_1 : ℕ)
                                      (ratio_2 : ℕ)
                                      (ratio_3 : ℕ)
                                      (total_places : ℕ) :
  total_students = 160 →
  ratio_1 = 6 →
  ratio_2 = 5 →
  ratio_3 = 5 →
  total_places = 160 →
  (total_places * ratio_1) / (ratio_1 + ratio_2 + ratio_3) = 60 :=
sorry

end allocation_first_grade_places_l7_7742


namespace first_machine_copies_per_minute_l7_7231

theorem first_machine_copies_per_minute
    (x : ℕ)
    (h1 : ∀ (x : ℕ), 30 * x + 30 * 55 = 2850) :
  x = 40 :=
by
  sorry

end first_machine_copies_per_minute_l7_7231


namespace solve_z_l7_7872

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l7_7872


namespace simplify_expression_l7_7452

theorem simplify_expression (p q r : ℝ) (hp : p ≠ 7) (hq : q ≠ 8) (hr : r ≠ 9) :
  ( ( (p - 7) / (9 - r) ) * ( (q - 8) / (7 - p) ) * ( (r - 9) / (8 - q) ) ) = -1 := 
by 
  sorry

end simplify_expression_l7_7452


namespace evaluate_fraction_l7_7843

theorem evaluate_fraction : (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = (8 / 21) :=
by
  sorry

end evaluate_fraction_l7_7843


namespace greatest_possible_integer_l7_7919

theorem greatest_possible_integer (n k l : ℕ) (h1 : n < 150) (h2 : n = 11 * k - 1) (h3 : n = 9 * l + 2) : n = 65 :=
by sorry

end greatest_possible_integer_l7_7919


namespace no_integer_roots_l7_7949

theorem no_integer_roots (x : ℤ) : ¬ (x^2 + 2^2018 * x + 2^2019 = 0) :=
sorry

end no_integer_roots_l7_7949


namespace sqrt5_times_sqrt6_minus_1_over_sqrt5_bound_l7_7693

theorem sqrt5_times_sqrt6_minus_1_over_sqrt5_bound :
  4 < (Real.sqrt 5) * ((Real.sqrt 6) - 1 / (Real.sqrt 5)) ∧ (Real.sqrt 5) * ((Real.sqrt 6) - 1 / (Real.sqrt 5)) < 5 :=
by
  sorry

end sqrt5_times_sqrt6_minus_1_over_sqrt5_bound_l7_7693


namespace line_x_intercept_l7_7540

theorem line_x_intercept {x1 y1 x2 y2 : ℝ} (h : (x1, y1) = (4, 6)) (h2 : (x2, y2) = (8, 2)) :
  ∃ x : ℝ, (y1 - y2) / (x1 - x2) * x + 6 - ((y1 - y2) / (x1 - x2)) * 4 = 0 ∧ x = 10 :=
by
  sorry

end line_x_intercept_l7_7540


namespace lattice_points_count_l7_7538

-- Definition of a lattice point
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Given endpoints of the line segment
def point1 : LatticePoint := ⟨5, 13⟩
def point2 : LatticePoint := ⟨38, 214⟩

-- Function to count lattice points on the line segment given the endpoints
def countLatticePoints (p1 p2 : LatticePoint) : ℕ := sorry

-- The proof statement
theorem lattice_points_count :
  countLatticePoints point1 point2 = 4 := sorry

end lattice_points_count_l7_7538


namespace determine_x_squared_plus_y_squared_l7_7563

theorem determine_x_squared_plus_y_squared :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (x * y + x + y = 119) ∧ ((x^2 * y + x * y^2) = 1680) ∧ (x^2 + y^2 = 1057) :=
begin
  sorry
end

end determine_x_squared_plus_y_squared_l7_7563


namespace complement_intersection_l7_7590

noncomputable def U : Set ℤ := {-1, 0, 2}
noncomputable def A : Set ℤ := {-1, 2}
noncomputable def B : Set ℤ := {0, 2}
noncomputable def C_U_A : Set ℤ := U \ A

theorem complement_intersection :
  (C_U_A ∩ B) = {0} :=
by {
  -- sorry to skip the proof part as per instruction
  sorry
}

end complement_intersection_l7_7590


namespace find_solutions_l7_7845

theorem find_solutions (x y z : ℝ) :
    (x^2 + y^2 - z * (x + y) = 2 ∧ y^2 + z^2 - x * (y + z) = 4 ∧ z^2 + x^2 - y * (z + x) = 8) ↔
    (x = 1 ∧ y = -1 ∧ z = 2) ∨ (x = -1 ∧ y = 1 ∧ z = -2) := sorry

end find_solutions_l7_7845


namespace solve_inequality_l7_7511

theorem solve_inequality : {x : ℝ | |x - 2| * (x - 1) < 2} = {x : ℝ | x < 3} :=
by
  sorry

end solve_inequality_l7_7511


namespace days_per_book_l7_7441

theorem days_per_book (total_books : ℕ) (total_days : ℕ)
  (h1 : total_books = 41)
  (h2 : total_days = 492) :
  total_days / total_books = 12 :=
by
  -- proof goes here
  sorry

end days_per_book_l7_7441


namespace problem_part1_problem_part2_l7_7883

-- Problem statements

theorem problem_part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  (a + b) * (a^5 + b^5) ≥ 4 := 
sorry

theorem problem_part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  a + b ≤ 2 := 
sorry

end problem_part1_problem_part2_l7_7883


namespace expected_groups_l7_7173

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end expected_groups_l7_7173


namespace maximal_k_value_l7_7308

noncomputable def max_edges (n : ℕ) : ℕ :=
  2 * n - 4
   
theorem maximal_k_value (k n : ℕ) (h1 : n = 2016) (h2 : k ≤ max_edges n) :
  k = 4028 :=
by sorry

end maximal_k_value_l7_7308


namespace right_triangle_leg_squared_l7_7503

variable (a b c : ℝ)

theorem right_triangle_leg_squared (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : b^2 = 4 * (a + 1) :=
by
  sorry

end right_triangle_leg_squared_l7_7503


namespace cube_dihedral_angle_is_60_degrees_l7_7744

-- Define the cube and related geometrical features
structure Point := (x y z : ℝ)
structure Cube :=
  (A B C D A₁ B₁ C₁ D₁ : Point)
  (is_cube : true) -- Placeholder for cube properties

-- Define the function to calculate dihedral angle measure
noncomputable def dihedral_angle_measure (cube: Cube) : ℝ := sorry

-- The theorem statement
theorem cube_dihedral_angle_is_60_degrees (cube : Cube) : dihedral_angle_measure cube = 60 :=
by sorry

end cube_dihedral_angle_is_60_degrees_l7_7744


namespace remainder_product_modulo_17_l7_7651

theorem remainder_product_modulo_17 :
  (1234 % 17) = 5 ∧ (1235 % 17) = 6 ∧ (1236 % 17) = 7 ∧ (1237 % 17) = 8 ∧ (1238 % 17) = 9 →
  ((1234 * 1235 * 1236 * 1237 * 1238) % 17) = 9 :=
by
  sorry

end remainder_product_modulo_17_l7_7651


namespace pentagon_position_3010_l7_7013

def rotate_72 (s : String) : String :=
match s with
| "ABCDE" => "EABCD"
| "EABCD" => "DCBAE"
| "DCBAE" => "EDABC"
| "EDABC" => "ABCDE"
| _ => s

def reflect_vertical (s : String) : String :=
match s with
| "EABCD" => "DCBAE"
| "DCBAE" => "EABCD"
| _ => s

def transform (s : String) (n : Nat) : String :=
match n % 5 with
| 0 => s
| 1 => reflect_vertical (rotate_72 s)
| 2 => rotate_72 (reflect_vertical (rotate_72 s))
| 3 => reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s)))
| 4 => rotate_72 (reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s))))
| _ => s

theorem pentagon_position_3010 :
  transform "ABCDE" 3010 = "ABCDE" :=
by 
  sorry

end pentagon_position_3010_l7_7013


namespace Jasmine_initial_percentage_is_5_l7_7988

noncomputable def initial_percentage_of_jasmine 
  (V_initial : ℕ := 90) 
  (V_added_jasmine : ℕ := 8) 
  (V_added_water : ℕ := 2) 
  (V_final : ℕ := 100) 
  (P_final : ℚ := 12.5 / 100) : ℚ := 
  (P_final * V_final - V_added_jasmine) / V_initial * 100

theorem Jasmine_initial_percentage_is_5 :
  initial_percentage_of_jasmine = 5 := 
by 
  sorry

end Jasmine_initial_percentage_is_5_l7_7988


namespace jordan_run_7_miles_in_112_div_3_minutes_l7_7750

noncomputable def time_for_steve (distance : ℝ) : ℝ := 36 / 4.5 * distance
noncomputable def jordan_initial_time (steve_time : ℝ) : ℝ := steve_time / 3
noncomputable def jordan_speed (distance time : ℝ) : ℝ := distance / time
noncomputable def adjusted_speed (speed : ℝ) : ℝ := speed * 0.9
noncomputable def running_time (distance speed : ℝ) : ℝ := distance / speed

theorem jordan_run_7_miles_in_112_div_3_minutes : running_time 7 ((jordan_speed 2.5 (jordan_initial_time (time_for_steve 4.5))) * 0.9) = 112 / 3 :=
by
  sorry

end jordan_run_7_miles_in_112_div_3_minutes_l7_7750


namespace num_square_tiles_l7_7989

theorem num_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 :=
  sorry

end num_square_tiles_l7_7989


namespace original_ratio_l7_7017

namespace OilBill

-- Definitions based on conditions
def JanuaryBill : ℝ := 179.99999999999991

def FebruaryBillWith30More (F : ℝ) : Prop := 
  3 * (F + 30) = 900

-- Statement of the problem proving the original ratio
theorem original_ratio (F : ℝ) (hF : FebruaryBillWith30More F) : 
  F / JanuaryBill = 3 / 2 :=
by
  -- This will contain the proof steps
  sorry

end OilBill

end original_ratio_l7_7017


namespace percentage_failed_in_hindi_l7_7108

theorem percentage_failed_in_hindi (P_E : ℝ) (P_H_and_E : ℝ) (P_P : ℝ) (H : ℝ) : 
  P_E = 0.5 ∧ P_H_and_E = 0.25 ∧ P_P = 0.5 → H = 0.25 :=
by
  sorry

end percentage_failed_in_hindi_l7_7108


namespace cube_less_than_three_times_l7_7377

theorem cube_less_than_three_times (x : ℤ) : x ^ 3 < 3 * x ↔ x = -3 ∨ x = -2 ∨ x = 1 :=
by
  sorry

end cube_less_than_three_times_l7_7377


namespace sqrt_of_expression_l7_7834

theorem sqrt_of_expression : Real.sqrt (5^2 * 7^6) = 1715 := 
by
  sorry

end sqrt_of_expression_l7_7834


namespace a_squared_plus_b_squared_a_plus_b_squared_l7_7256

variable (a b : ℚ)

-- Conditions
axiom h1 : a - b = 7
axiom h2 : a * b = 18

-- To Prove
theorem a_squared_plus_b_squared : a^2 + b^2 = 85 :=
by sorry

theorem a_plus_b_squared : (a + b)^2 = 121 :=
by sorry

end a_squared_plus_b_squared_a_plus_b_squared_l7_7256


namespace child_haircut_cost_l7_7159

/-
Problem Statement:
- Women's haircuts cost $48.
- Tayzia and her two daughters get haircuts.
- Tayzia wants to give a 20% tip to the hair stylist, which amounts to $24.
Question: How much does a child's haircut cost?
-/

noncomputable def cost_of_child_haircut (C : ℝ) : Prop :=
  let women's_haircut := 48
  let tip := 24
  let total_cost_before_tip := women's_haircut + 2 * C
  total_cost_before_tip * 0.20 = tip ∧ total_cost_before_tip = 120 ∧ C = 36

theorem child_haircut_cost (C : ℝ) (h1 : cost_of_child_haircut C) : C = 36 :=
  by sorry

end child_haircut_cost_l7_7159


namespace second_term_geometric_series_l7_7552

theorem second_term_geometric_series (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 48) (h3 : S = a / (1 - r)) :
  a * r = 9 :=
by
  -- Sorry is used to finalize the theorem without providing a proof here
  sorry

end second_term_geometric_series_l7_7552


namespace expression_equals_1390_l7_7558

theorem expression_equals_1390 :
  (25 + 15 + 8) ^ 2 - (25 ^ 2 + 15 ^ 2 + 8 ^ 2) = 1390 := 
by
  sorry

end expression_equals_1390_l7_7558


namespace number_of_pupils_l7_7527

-- Define the conditions.
variables (n : ℕ) -- Number of pupils in the class.

-- Axioms based on the problem statement.
axiom marks_difference : 67 - 45 = 22
axiom avg_increase : (1 / 2 : ℝ) * n = 22 

-- The theorem we need to prove.
theorem number_of_pupils : n = 44 := by
  -- Proof will go here.
  sorry

end number_of_pupils_l7_7527


namespace unique_solution_l7_7722

theorem unique_solution (a b c : ℝ) (hb : b ≠ 2) (hc : c ≠ 0) : 
  ∃! x : ℝ, 4 * x - 7 + a = 2 * b * x + c ∧ x = (c + 7 - a) / (4 - 2 * b) :=
by
  sorry

end unique_solution_l7_7722


namespace intersection_of_sets_l7_7271

theorem intersection_of_sets :
  let M := { x : ℝ | -3 < x ∧ x ≤ 5 }
  let N := { x : ℝ | -5 < x ∧ x < 5 }
  M ∩ N = { x : ℝ | -3 < x ∧ x < 5 } := 
by
  sorry

end intersection_of_sets_l7_7271


namespace boys_count_in_dance_class_l7_7300

theorem boys_count_in_dance_class
  (total_students : ℕ) 
  (ratio_girls_to_boys : ℕ) 
  (ratio_boys_to_girls: ℕ)
  (total_students_eq : total_students = 35)
  (ratio_eq : ratio_girls_to_boys = 3 ∧ ratio_boys_to_girls = 4) : 
  ∃ boys : ℕ, boys = 20 :=
by
  let k := total_students / (ratio_girls_to_boys + ratio_boys_to_girls)
  have girls := ratio_girls_to_boys * k
  have boys := ratio_boys_to_girls * k
  use boys
  sorry

end boys_count_in_dance_class_l7_7300


namespace roots_quadratic_expression_l7_7263

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) 
  (sum_roots : m + n = -2) (product_roots : m * n = -5) : m^2 + m * n + 3 * m + n = -2 :=
sorry

end roots_quadratic_expression_l7_7263


namespace second_term_of_geometric_series_l7_7412

theorem second_term_of_geometric_series 
  (a : ℝ) (r : ℝ) (S : ℝ) :
  r = 1 / 4 → S = 40 → S = a / (1 - r) → a * r = 7.5 :=
by
  intros hr hS hSum
  sorry

end second_term_of_geometric_series_l7_7412


namespace surface_area_of_second_cube_l7_7355

theorem surface_area_of_second_cube (V1 V2: ℝ) (a2: ℝ):
  (V1 = 16 ∧ V2 = 4 * V1 ∧ a2 = (V2)^(1/3)) → 6 * a2^2 = 96 :=
by intros h; sorry

end surface_area_of_second_cube_l7_7355


namespace least_possible_value_of_z_minus_x_l7_7740

theorem least_possible_value_of_z_minus_x 
  (x y z : ℤ) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) 
  (h4 : ∃ n : ℤ, x = 2 * n)
  (h5 : ∃ m : ℤ, y = 2 * m + 1) 
  (h6 : ∃ k : ℤ, z = 2 * k + 1) : 
  z - x = 9 := 
sorry

end least_possible_value_of_z_minus_x_l7_7740


namespace expected_min_leq_2_l7_7326

open ProbabilityTheory

variables (ξ η : ℝ → ℝ) -- ξ and η are random variables

-- Condition: expected value of ξ is 2
axiom E_ξ_eq_2 : ℝ
axiom E_ξ_is_2 : (∫ x in ⊤, ξ x) = 2

-- Goal: expected value of min(ξ, η) ≤ 2
theorem expected_min_leq_2 (h : ∀ x, min (ξ x) (η x) ≤ ξ x) : 
  (∫ x in ⊤, min (ξ x) (η x)) ≤ 2 := by
  -- use the provided axioms and conditions here
  sorry

end expected_min_leq_2_l7_7326


namespace frac_addition_l7_7523

theorem frac_addition :
  (3 / 5) + (2 / 15) = 11 / 15 :=
sorry

end frac_addition_l7_7523


namespace xiao_li_max_prob_interview_xiao_li_xiao_wang_prob_and_expected_value_l7_7536

noncomputable def xiao_li_passing_prob_B : ℝ := 2 / 3
noncomputable def xiao_li_passing_prob_C : ℝ := 1 / 3
noncomputable def xiao_li_passing_prob_D : ℝ := 1 / 2
noncomputable def xiao_wang_passing_prob : ℝ := 2 / 3

/-- Xiao Li should choose test locations B and D to maximize the probability of being eligible for an interview. -/
theorem xiao_li_max_prob_interview :
  max (xiao_li_passing_prob_B * xiao_li_passing_prob_C)
      (max (xiao_li_passing_prob_B * xiao_li_passing_prob_D)
           (xiao_li_passing_prob_C * xiao_li_passing_prob_D)) 
  = xiao_li_passing_prob_B * xiao_li_passing_prob_D := 
sorry

/-- Determine the probability distribution for the random variable ξ and its expected value. -/
theorem xiao_li_xiao_wang_prob_and_expected_value (B C D : Prop) [Indep of B C D] :
  let ξ := B.CD.in_front_of_xiao_wang_pass_events (B_intersects_xiao_li_C : bool) in
    ξ = 0 → P (ξ = 0) = 2 / 81
    ∧ ξ = 1 → P (ξ = 1) = 13 / 81
    ∧ ξ = 2 → P (ξ = 2) = 10 / 27
    ∧ ξ = 3 → P (ξ = 3) = 28 / 81
    ∧ ξ = 4 → P (ξ = 4) = 8 / 81
    ∧ E ξ = 7 / 3 := 
sorry

end xiao_li_max_prob_interview_xiao_li_xiao_wang_prob_and_expected_value_l7_7536


namespace add_water_to_solution_l7_7035

noncomputable def current_solution_volume : ℝ := 300
noncomputable def desired_water_percentage : ℝ := 0.70
noncomputable def current_water_volume : ℝ := 0.60 * current_solution_volume
noncomputable def current_acid_volume : ℝ := 0.40 * current_solution_volume

theorem add_water_to_solution (x : ℝ) : 
  (current_water_volume + x) / (current_solution_volume + x) = desired_water_percentage ↔ x = 100 :=
by
  sorry

end add_water_to_solution_l7_7035


namespace find_b_value_l7_7665

theorem find_b_value 
  (point1 : ℝ × ℝ) (point2 : ℝ × ℝ) (b : ℝ) 
  (h1 : point1 = (0, -2))
  (h2 : point2 = (1, 0))
  (h3 : (∃ m c, ∀ x y, y = m * x + c ↔ (x, y) = point1 ∨ (x, y) = point2))
  (h4 : ∀ x y, y = 2 * x - 2 → (x, y) = (7, b)) :
  b = 12 :=
sorry

end find_b_value_l7_7665


namespace conic_sections_l7_7014

theorem conic_sections (x y : ℝ) : 
  y^4 - 16*x^4 = 8*y^2 - 4 → 
  (y^2 - 4 * x^2 = 4 ∨ y^2 + 4 * x^2 = 4) :=
sorry

end conic_sections_l7_7014


namespace num_ways_distribute_balls_l7_7278

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end num_ways_distribute_balls_l7_7278


namespace salary_of_A_l7_7657

theorem salary_of_A (x y : ℝ) (h₁ : x + y = 4000) (h₂ : 0.05 * x = 0.15 * y) : x = 3000 :=
by {
    sorry
}

end salary_of_A_l7_7657


namespace quadratic_inequality_l7_7588

theorem quadratic_inequality (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 < 0) → a < 1 := 
by
  sorry

end quadratic_inequality_l7_7588


namespace num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l7_7964

def is_prime (n : ℕ) : Prop := Nat.Prime n
def ends_with_7 (n : ℕ) : Prop := n % 10 = 7

theorem num_prime_numbers_with_units_digit_7 (n : ℕ) (h1 : n < 100) (h2 : ends_with_7 n) : is_prime n :=
by sorry

theorem num_prime_numbers_less_than_100_with_units_digit_7 : 
  ∃ (l : List ℕ), (∀ x ∈ l, x < 100 ∧ ends_with_7 x ∧ is_prime x) ∧ l.length = 6 :=
by sorry

end num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l7_7964


namespace words_count_correct_l7_7898

def number_of_words (n : ℕ) : ℕ :=
if n % 2 = 0 then
  8 * 3^(n / 2 - 1)
else
  14 * 3^((n - 1) / 2)

theorem words_count_correct (n : ℕ) :
  number_of_words n = if n % 2 = 0 then 8 * 3^(n / 2 - 1) else 14 * 3^((n - 1) / 2) :=
by
  sorry

end words_count_correct_l7_7898


namespace compute_expression_l7_7825

theorem compute_expression :
  3 * 3^4 - 9^60 / 9^57 = -486 :=
by
  sorry

end compute_expression_l7_7825


namespace find_a_and_b_l7_7443

theorem find_a_and_b (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {2, 3}) 
  (hB : B = {x | x^2 + a * x + b = 0}) 
  (h_intersection : A ∩ B = {2}) 
  (h_union : A ∪ B = A) : 
  (a + b = 0) ∨ (a + b = 1) := 
sorry

end find_a_and_b_l7_7443


namespace certain_number_correct_l7_7734

theorem certain_number_correct (x : ℝ) (h1 : 213 * 16 = 3408) (h2 : 213 * x = 340.8) : x = 1.6 := by
  sorry

end certain_number_correct_l7_7734


namespace descending_digits_count_l7_7848

theorem descending_digits_count : 
  ∑ k in (finset.range 11).filter (λ k, 2 ≤ k), nat.choose 10 k = 1013 := 
sorry

end descending_digits_count_l7_7848


namespace smallest_rel_prime_210_l7_7698

theorem smallest_rel_prime_210 : ∃ x : ℕ, x > 1 ∧ gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 210 = 1 → x ≤ y := 
by
  sorry

end smallest_rel_prime_210_l7_7698


namespace determine_BD_l7_7747

variable (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB BC CD DA : ℝ)
variables (BD : ℝ)

-- Setting up the conditions:
axiom AB_eq_5 : AB = 5
axiom BC_eq_17 : BC = 17
axiom CD_eq_5 : CD = 5
axiom DA_eq_9 : DA = 9
axiom BD_is_integer : ∃ (n : ℤ), BD = n

theorem determine_BD : BD = 13 :=
by
  sorry

end determine_BD_l7_7747


namespace muffin_cost_relation_l7_7958

variable (m b : ℝ)

variable (S := 5 * m + 4 * b)
variable (C := 10 * m + 18 * b)

theorem muffin_cost_relation (h1 : C = 3 * S) : m = 1.2 * b :=
  sorry

end muffin_cost_relation_l7_7958


namespace polynomial_simplification_l7_7796

def A (x : ℝ) := 5 * x^2 + 4 * x - 1
def B (x : ℝ) := -x^2 - 3 * x + 3
def C (x : ℝ) := 8 - 7 * x - 6 * x^2

theorem polynomial_simplification (x : ℝ) : A x - B x + C x = 4 :=
by
  simp [A, B, C]
  sorry

end polynomial_simplification_l7_7796


namespace range_of_sum_of_zeros_l7_7457

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x else 1 - x / 2

noncomputable def F (x : ℝ) (m : ℝ) : ℝ :=
  f (f x + 1) + m

def has_zeros (F : ℝ → ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ F x₁ m = 0 ∧ F x₂ m = 0

theorem range_of_sum_of_zeros (m : ℝ) :
  has_zeros F m →
  ∃ (x₁ x₂ : ℝ), F x₁ m = 0 ∧ F x₂ m = 0 ∧ (x₁ + x₂) ≥ 4 - 2 * Real.log 2 := sorry

end range_of_sum_of_zeros_l7_7457


namespace shortest_distance_l7_7124

noncomputable def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 20
noncomputable def line (x : ℝ) : ℝ := x - 6

def distance_point_to_line (a : ℝ) : ℝ :=
  abs (a - (parabola a) - 6) / real.sqrt 2

theorem shortest_distance :
  ∃ a, ∀ b, distance_point_to_line a ≤ distance_point_to_line b :=
begin
  use 5/2,
  intro b,
  have h : distance_point_to_line (5/2) = 103 * real.sqrt 2 / 8,
  sorry, -- proof goes here
  rw h,
  refine le_of_eq _,
  sorry, -- proof goes here
end

end shortest_distance_l7_7124


namespace average_score_of_all_matches_is_36_l7_7618

noncomputable def average_score_of_all_matches
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) : ℝ :=
  (x + y + a + b + c) / 5

theorem average_score_of_all_matches_is_36
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) :
  average_score_of_all_matches x y a b c h1 h2 h3x h3y h3a h3b h3c h4 = 36 := 
  by 
  sorry

end average_score_of_all_matches_is_36_l7_7618


namespace one_in_set_A_l7_7269

theorem one_in_set_A : 1 ∈ {x | x ≥ -1} :=
sorry

end one_in_set_A_l7_7269


namespace sequence_an_eq_n_l7_7449

theorem sequence_an_eq_n (a : ℕ → ℝ) (S : ℕ → ℝ) (h₀ : ∀ n, n ≥ 1 → a n > 0) 
  (h₁ : ∀ n, n ≥ 1 → a n + 1 / 2 = Real.sqrt (2 * S n + 1 / 4)) : 
  ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end sequence_an_eq_n_l7_7449


namespace halfway_between_one_nine_and_one_eleven_l7_7431

theorem halfway_between_one_nine_and_one_eleven : 
  (1/9 + 1/11) / 2 = 10/99 :=
by sorry

end halfway_between_one_nine_and_one_eleven_l7_7431


namespace delacroix_band_max_members_l7_7636

theorem delacroix_band_max_members :
  ∃ n : ℕ, 30 * n % 28 = 6 ∧ 30 * n < 1200 ∧ 30 * n = 930 :=
by
  sorry

end delacroix_band_max_members_l7_7636


namespace carla_marbles_l7_7985

theorem carla_marbles (m : ℕ) : m + 134 = 187 ↔ m = 53 :=
by sorry

end carla_marbles_l7_7985


namespace julie_total_lettuce_pounds_l7_7118

theorem julie_total_lettuce_pounds :
  ∀ (cost_green cost_red cost_per_pound total_cost total_pounds : ℕ),
  cost_green = 8 →
  cost_red = 6 →
  cost_per_pound = 2 →
  total_cost = cost_green + cost_red →
  total_pounds = total_cost / cost_per_pound →
  total_pounds = 7 :=
by
  intros cost_green cost_red cost_per_pound total_cost total_pounds h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3] at h5
  norm_num at h4
  norm_num at h5
  exact h5

end julie_total_lettuce_pounds_l7_7118


namespace annual_increase_rate_l7_7425

theorem annual_increase_rate (PV FV : ℝ) (n : ℕ) (r : ℝ) :
  PV = 32000 ∧ FV = 40500 ∧ n = 2 ∧ FV = PV * (1 + r)^2 → r = 0.125 :=
by
  sorry

end annual_increase_rate_l7_7425


namespace expected_flashlight_lifetime_leq_two_l7_7316

theorem expected_flashlight_lifetime_leq_two
  (Ω : Type*) [MeasurableSpace Ω] [ProbabilitySpace Ω]
  (ξ η : Ω → ℝ)
  (h_min_leq_xi : ∀ ω, min (ξ ω) (η ω) ≤ ξ ω)
  (h_expectation_xi : expectation (ξ) = 2) :
  expectation (fun ω => min (ξ ω) (η ω)) ≤ 2 := 
sorry

end expected_flashlight_lifetime_leq_two_l7_7316


namespace range_of_expression_positive_range_of_expression_negative_l7_7463

theorem range_of_expression_positive (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 > 0) ↔ (x < -3/2 ∨ x > 4) :=
sorry

theorem range_of_expression_negative (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 < 0) ↔ ( -3/2 < x ∧ x < 4) :=
sorry

end range_of_expression_positive_range_of_expression_negative_l7_7463


namespace distinct_meals_l7_7416

def num_entrees : ℕ := 4
def num_drinks : ℕ := 2
def num_desserts : ℕ := 2

theorem distinct_meals : num_entrees * num_drinks * num_desserts = 16 := by
  sorry

end distinct_meals_l7_7416


namespace grandfather_grandson_ages_l7_7955

theorem grandfather_grandson_ages :
  ∃ (x y a b : ℕ), 
    70 < x ∧ 
    x < 80 ∧ 
    x - a = 10 * (y - a) ∧ 
    x + b = 8 * (y + b) ∧ 
    x = 71 ∧ 
    y = 8 :=
by
  sorry

end grandfather_grandson_ages_l7_7955


namespace gcd_between_30_40_l7_7788

-- Defining the number and its constraints
def num := {n : ℕ // 30 < n ∧ n < 40 ∧ Nat.gcd 15 n = 5}

-- Theorem statement
theorem gcd_between_30_40 : (n : num) → n = 35 :=
by
  -- This is where the proof would go
  sorry

end gcd_between_30_40_l7_7788


namespace three_gt_sqrt_seven_l7_7683

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := sorry

end three_gt_sqrt_seven_l7_7683


namespace sum_of_cubes_l7_7293

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 20) : x^3 + y^3 = 87.5 := 
by 
  sorry

end sum_of_cubes_l7_7293


namespace sum_pqrst_is_neg_15_over_2_l7_7482

variable (p q r s t x : ℝ)
variable (h1 : p + 2 = x)
variable (h2 : q + 3 = x)
variable (h3 : r + 4 = x)
variable (h4 : s + 5 = x)
variable (h5 : t + 6 = x)
variable (h6 : p + q + r + s + t + 10 = x)

theorem sum_pqrst_is_neg_15_over_2 : p + q + r + s + t = -15 / 2 := by
  sorry

end sum_pqrst_is_neg_15_over_2_l7_7482


namespace maximum_value_of_m_solve_inequality_l7_7852

theorem maximum_value_of_m (a b : ℝ) (h : a ≠ 0) : 
  ∃ m : ℝ, (∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ m * |a|) ∧ (m = 2) :=
by
  use 2
  sorry

theorem solve_inequality (x : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| ≤ 2 → (1/2 ≤ x ∧ x ≤ 5/2)) :=
by
  sorry

end maximum_value_of_m_solve_inequality_l7_7852


namespace train_problem_l7_7954

theorem train_problem (Sat M S C : ℕ) 
  (h_boarding_day : true)
  (h_arrival_day : true)
  (h_date_matches_car_on_monday : M = C)
  (h_seat_less_than_car : S < C)
  (h_sat_date_greater_than_car : Sat > C) :
  C = 2 ∧ S = 1 :=
by sorry

end train_problem_l7_7954


namespace polygon_num_sides_and_exterior_angle_l7_7644

theorem polygon_num_sides_and_exterior_angle 
  (n : ℕ) (x : ℕ) 
  (h : (n - 2) * 180 + x = 1350) 
  (hx : 0 < x ∧ x < 180) 
  : (n = 9) ∧ (x = 90) := 
by 
  sorry

end polygon_num_sides_and_exterior_angle_l7_7644


namespace tinas_extra_earnings_l7_7620

def price_per_candy_bar : ℕ := 2
def marvins_candy_bars_sold : ℕ := 35
def tinas_candy_bars_sold : ℕ := 3 * marvins_candy_bars_sold

def marvins_earnings : ℕ := marvins_candy_bars_sold * price_per_candy_bar
def tinas_earnings : ℕ := tinas_candy_bars_sold * price_per_candy_bar

theorem tinas_extra_earnings : tinas_earnings - marvins_earnings = 140 := by
  sorry

end tinas_extra_earnings_l7_7620


namespace simplify_expression_l7_7001

-- Defining each term in the sum
def term1  := 1 / ((1 / 3) ^ 1)
def term2  := 1 / ((1 / 3) ^ 2)
def term3  := 1 / ((1 / 3) ^ 3)

-- Sum of the terms
def terms_sum := term1 + term2 + term3

-- The simplification proof statement
theorem simplify_expression : 1 / terms_sum = 1 / 39 :=
by sorry

end simplify_expression_l7_7001


namespace probability_of_exactly_9_correct_matches_is_zero_l7_7202

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end probability_of_exactly_9_correct_matches_is_zero_l7_7202


namespace amount_left_for_gas_and_maintenance_l7_7478

def monthly_income : ℤ := 3200
def rent : ℤ := 1250
def utilities : ℤ := 150
def retirement_savings : ℤ := 400
def groceries_eating_out : ℤ := 300
def insurance : ℤ := 200
def miscellaneous : ℤ := 200
def car_payment : ℤ := 350

def total_expenses : ℤ :=
  rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment

theorem amount_left_for_gas_and_maintenance : monthly_income - total_expenses = 350 :=
by
  -- Proof is omitted
  sorry

end amount_left_for_gas_and_maintenance_l7_7478


namespace females_in_town_l7_7301

theorem females_in_town (population : ℕ) (ratio : ℕ × ℕ) (H : population = 480) (H_ratio : ratio = (3, 5)) : 
  let m := ratio.1
  let f := ratio.2
  f * (population / (m + f)) = 300 := by
  sorry

end females_in_town_l7_7301


namespace average_number_of_ducks_l7_7545

def average_ducks (A E K : ℕ) : ℕ :=
  (A + E + K) / 3

theorem average_number_of_ducks :
  ∀ (A E K : ℕ), A = 2 * E → E = K - 45 → A = 30 → average_ducks A E K = 35 :=
by 
  intros A E K h1 h2 h3
  sorry

end average_number_of_ducks_l7_7545


namespace price_of_ice_cream_bar_is_correct_l7_7990

noncomputable def price_ice_cream_bar (n_ice_cream_bars n_sundaes total_price price_of_sundae price_ice_cream_bar : ℝ) : Prop :=
  n_ice_cream_bars = 125 ∧
  n_sundaes = 125 ∧
  total_price = 225 ∧
  price_of_sundae = 1.2 →
  price_ice_cream_bar = 0.6

theorem price_of_ice_cream_bar_is_correct :
  price_ice_cream_bar 125 125 225 1.2 0.6 :=
by
  sorry

end price_of_ice_cream_bar_is_correct_l7_7990


namespace area_of_quadrilateral_l7_7462

theorem area_of_quadrilateral (θ : ℝ) (sin_θ : Real.sin θ = 4/5) (b1 b2 : ℝ) (h: ℝ) (base1 : b1 = 14) (base2 : b2 = 20) (height : h = 8) : 
  (1 / 2) * (b1 + b2) * h = 136 := by
  sorry

end area_of_quadrilateral_l7_7462


namespace man_fraction_ownership_l7_7398

theorem man_fraction_ownership :
  ∀ (F : ℚ), (3 / 5 * F = 15000) → (75000 = 75000) → (F / 75000 = 1 / 3) :=
by
  intros F h1 h2
  sorry

end man_fraction_ownership_l7_7398


namespace months_decreasing_l7_7489

noncomputable def stock_decrease (m : ℕ) : Prop :=
  2 * m + 2 * 8 = 18

theorem months_decreasing (m : ℕ) (h : stock_decrease m) : m = 1 :=
by
  exact sorry

end months_decreasing_l7_7489


namespace number_of_true_propositions_l7_7583

theorem number_of_true_propositions:
  (∀ x : ℝ, x^2 + 1 > 0) ∧
  (¬ ∀ x : ℕ, x^4 ≥ 1) ∧
  (∃ x : ℤ, x^3 < 1) ∧
  (∀ x : ℚ, x^2 ≠ 2) →
  3 = 3 :=
by
  intros h,
  sorry

end number_of_true_propositions_l7_7583


namespace f_f_4_eq_1_l7_7585

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x

theorem f_f_4_eq_1 : f (f 4) = 1 := by
  sorry

end f_f_4_eq_1_l7_7585


namespace marble_color_197_l7_7539

-- Define the types and properties of the marbles
inductive Color where
  | red | blue | green

-- Define a function to find the color of the nth marble in the cycle pattern
def colorOfMarble (n : Nat) : Color :=
  let cycleLength := 15
  let positionInCycle := n % cycleLength
  if positionInCycle < 6 then Color.red  -- first 6 marbles are red
  else if positionInCycle < 11 then Color.blue  -- next 5 marbles are blue
  else Color.green  -- last 4 marbles are green

-- The theorem asserting the color of the 197th marble
theorem marble_color_197 : colorOfMarble 197 = Color.red :=
sorry

end marble_color_197_l7_7539


namespace parabola_intersects_line_l7_7349

-- Define the conditions of the problem
variables {a k x1 x2 : ℝ}
variable (h_a_ne_0 : a ≠ 0)
variable (h_x1_x2 : x1 + x2 < 0)
variable (h_intersect : ∀ x, a * x^2 - a = k * x → x = x1 ∨ x = x2)

-- Define what needs to be proven
theorem parabola_intersects_line {h_a_ne_0 h_x1_x2 h_intersect} :
  ∃ k, (k < 0 ∧ k > 0 → (λ x, a * x + k).exists_first_and_fourth_quadrant) ∧ 
       (k > 0 ∧ k < 0 → (λ x, a * x + k).exists_first_and_fourth_quadrant) :=
sorry

end parabola_intersects_line_l7_7349


namespace grid_coloring_probability_sum_l7_7567

theorem grid_coloring_probability_sum : 
  ∃ m n : ℕ, 
    Nat.gcd m n = 1 ∧ 
    (m + n = 929) ∧ 
    (m / n = 417 / 512) :=
by
  sorry

end grid_coloring_probability_sum_l7_7567


namespace cubic_eq_has_real_roots_l7_7714

theorem cubic_eq_has_real_roots (K : ℝ) (hK : K ≠ 0) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by sorry

end cubic_eq_has_real_roots_l7_7714


namespace find_z_l7_7867

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l7_7867


namespace cube_root_of_neg_eight_squared_is_neg_four_l7_7500

-- Define the value of -8^2
def neg_eight_squared : ℤ := -8^2

-- Define what it means for a number to be the cube root of another number
def is_cube_root (a b : ℤ) : Prop := a^3 = b

-- The desired proof statement
theorem cube_root_of_neg_eight_squared_is_neg_four :
  neg_eight_squared = -64 → is_cube_root (-4) neg_eight_squared :=
by
  sorry

end cube_root_of_neg_eight_squared_is_neg_four_l7_7500


namespace find_r_l7_7689

-- Define vectors a and b
def vecA : ℝ^3 := ![3, 1, -2]
def vecB : ℝ^3 := ![1, 2, -1]
def target : ℝ^3 := ![5, 0, -5]

-- Define cross product function
def cross_prod (u v : ℝ^3) : ℝ^3 :=
  ![
    u[1]*v[2] - u[2]*v[1],
    u[2]*v[0] - u[0]*v[2],
    u[0]*v[1] - u[1]*v[0]
  ]

-- Cross product a x b
def vecAxB := cross_prod vecA vecB

-- Conditions derived from the solution
theorem find_r :
  target = λ p q r, p • vecA + q • vecB + r • vecAxB := sorry

end find_r_l7_7689


namespace weight_of_a_is_75_l7_7032

theorem weight_of_a_is_75 (a b c d e : ℕ) 
  (h1 : (a + b + c) / 3 = 84) 
  (h2 : (a + b + c + d) / 4 = 80) 
  (h3 : e = d + 3) 
  (h4 : (b + c + d + e) / 4 = 79) : 
  a = 75 :=
by
  -- Proof omitted
  sorry

end weight_of_a_is_75_l7_7032


namespace lcm_of_8_and_15_l7_7847

theorem lcm_of_8_and_15 : Nat.lcm 8 15 = 120 :=
by
  sorry

end lcm_of_8_and_15_l7_7847


namespace transform_uniform_random_l7_7648

theorem transform_uniform_random (a_1 : ℝ) (h : 0 ≤ a_1 ∧ a_1 ≤ 1) : -2 ≤ a_1 * 8 - 2 ∧ a_1 * 8 - 2 ≤ 6 :=
by sorry

end transform_uniform_random_l7_7648


namespace determine_slope_l7_7878

variables {a_n : ℕ → ℤ} {s_n : ℕ → ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ a₁ d, ∀ (n : ℕ), a_n n = a₁ + n * d

def sum_of_first_n_terms (a_n : ℕ → ℤ) (s_n : ℕ → ℤ) :=
  ∀ (n : ℕ), s_n n = (n * (a_n 0 + a_n (n-1))) / 2

noncomputable def find_slope {a_n : ℕ → ℤ} (n : ℕ) : ℤ :=
  (a_n (n+2) - a_n n) / (2 : ℕ)

theorem determine_slope (a_n s_n : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a_n)
  (h2 : sum_of_first_n_terms a_n s_n)
  (S2 : s_n 2 = 10) (S5 : s_n 5 = 55)
  (n : ℕ) (hn : n ≠ 0) : find_slope a_n n = 4 :=
begin
  sorry
end

end determine_slope_l7_7878


namespace cube_dihedral_angle_l7_7745

-- Define geometric entities in the cube
structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

def A := Point.mk 0 0 0
def B := Point.mk 1 0 0
def C := Point.mk 1 1 0
def D := Point.mk 0 1 0
def A1 := Point.mk 0 0 1
def B1 := Point.mk 1 0 1
def C1 := Point.mk 1 1 1
def D1 := Point.mk 0 1 1

-- Distance function to calculate lengths between points
def dist (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

-- Given the dihedral angle's planes definitions, we abstractly define the angle
def dihedral_angle (p1 p2 p3 : Point) : ℝ := sorry -- For now, we leave this abstract

theorem cube_dihedral_angle : dihedral_angle A (dist B D1) A1 = 60 :=
by
  sorry

end cube_dihedral_angle_l7_7745


namespace am_gm_inequality_l7_7612

theorem am_gm_inequality (a1 a2 a3 : ℝ) (h₀ : 0 < a1) (h₁ : 0 < a2) (h₂ : 0 < a3) (h₃ : a1 + a2 + a3 = 1) : 
  1 / a1 + 1 / a2 + 1 / a3 ≥ 9 :=
by
  sorry

end am_gm_inequality_l7_7612


namespace num_distinct_prime_factors_330_l7_7277

theorem num_distinct_prime_factors_330 : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, Nat.Prime x ∧ 330 % x = 0 := 
sorry

end num_distinct_prime_factors_330_l7_7277


namespace probability_of_exactly_9_correct_matches_is_zero_l7_7203

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end probability_of_exactly_9_correct_matches_is_zero_l7_7203


namespace initial_number_306_l7_7658

theorem initial_number_306 (x : ℝ) : 
  (x / 34) * 15 + 270 = 405 → x = 306 :=
by
  intro h
  sorry

end initial_number_306_l7_7658


namespace divide_one_meter_into_100_parts_l7_7565

theorem divide_one_meter_into_100_parts :
  (1 / 100 : ℝ) = 1 / 100 := 
by
  sorry

end divide_one_meter_into_100_parts_l7_7565


namespace indistinguishable_balls_boxes_l7_7286

open Finset

def partitions (n : ℕ) (k : ℕ) : ℕ :=
  (univ : Finset (Multiset ℕ)).filter (λ p, p.sum = n ∧ p.card ≤ k).card

theorem indistinguishable_balls_boxes : partitions 6 4 = 9 :=
sorry

end indistinguishable_balls_boxes_l7_7286


namespace lcm_14_18_20_l7_7378

theorem lcm_14_18_20 : Nat.lcm (Nat.lcm 14 18) 20 = 1260 :=
by
  -- Define the prime factorizations
  have fact_14 : 14 = 2 * 7 := by norm_num
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_20 : 20 = 2^2 * 5 := by norm_num
  
  -- Calculate the LCM based on the highest powers of each prime
  have lcm : Nat.lcm (Nat.lcm 14 18) 20 = 2^2 * 3^2 * 5 * 7 :=
    by
      sorry -- Proof details are not required

  -- Final verification that this calculation matches 1260
  exact lcm

end lcm_14_18_20_l7_7378


namespace intersection_A_B_union_A_B_diff_A_B_diff_B_A_l7_7727

def A : Set Real := {x | -1 < x ∧ x < 2}
def B : Set Real := {x | 0 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x < 2} :=
sorry

theorem union_A_B :
  A ∪ B = {x | -1 < x ∧ x < 4} :=
sorry

theorem diff_A_B :
  A \ B = {x | -1 < x ∧ x ≤ 0} :=
sorry

theorem diff_B_A :
  B \ A = {x | 2 ≤ x ∧ x < 4} :=
sorry

end intersection_A_B_union_A_B_diff_A_B_diff_B_A_l7_7727


namespace total_time_from_first_station_to_workplace_l7_7762

-- Pick-up time is defined as a constant for clarity in minutes from midnight (6 AM)
def pickup_time_in_minutes : ℕ := 6 * 60

-- Travel time to first station in minutes
def travel_time_to_station_in_minutes : ℕ := 40

-- Arrival time at work (9 AM) in minutes from midnight
def arrival_time_at_work_in_minutes : ℕ := 9 * 60

-- Definition to calculate arrival time at the first station
def arrival_time_at_first_station_in_minutes : ℕ := pickup_time_in_minutes + travel_time_to_station_in_minutes

-- Theorem to prove the total time taken from the first station to the workplace
theorem total_time_from_first_station_to_workplace :
  arrival_time_at_work_in_minutes - arrival_time_at_first_station_in_minutes = 140 :=
by
  -- Placeholder for the actual proof
  sorry

end total_time_from_first_station_to_workplace_l7_7762


namespace ratio_of_areas_l7_7947

noncomputable def area (A B C D : ℝ) : ℝ := 0  -- Placeholder, exact area definition will require geometrical formalism.

variables (A B C D P Q R S : ℝ)

-- Define the conditions
variables (h1 : AB = BP) (h2 : BC = CQ) (h3 : CD = DR) (h4 : DA = AS)

-- Lean 4 statement for the proof problem
theorem ratio_of_areas : area A B C D / area P Q R S = 1/5 :=
sorry

end ratio_of_areas_l7_7947


namespace max_snacks_l7_7772

-- Define the conditions and the main statement we want to prove

def single_snack_cost : ℕ := 2
def four_snack_pack_cost : ℕ := 6
def six_snack_pack_cost : ℕ := 8
def budget : ℕ := 20

def max_snacks_purchased : ℕ := 14

theorem max_snacks (h1 : single_snack_cost = 2) 
                   (h2 : four_snack_pack_cost = 6) 
                   (h3 : six_snack_pack_cost = 8) 
                   (h4 : budget = 20) : 
                   max_snacks_purchased = 14 := 
by {
  sorry
}

end max_snacks_l7_7772


namespace price_of_5_pound_bag_l7_7396

-- Definitions based on conditions
def price_10_pound_bag : ℝ := 20.42
def price_25_pound_bag : ℝ := 32.25
def min_pounds : ℝ := 65
def max_pounds : ℝ := 80
def total_min_cost : ℝ := 98.77

-- Define the sought price of the 5-pound bag in the hypothesis
variable {price_5_pound_bag : ℝ}

-- The theorem to prove based on the given conditions
theorem price_of_5_pound_bag
  (h₁ : price_10_pound_bag = 20.42)
  (h₂ : price_25_pound_bag = 32.25)
  (h₃ : min_pounds = 65)
  (h₄ : max_pounds = 80)
  (h₅ : total_min_cost = 98.77) :
  price_5_pound_bag = 2.02 :=
sorry

end price_of_5_pound_bag_l7_7396


namespace find_point_P_l7_7716

noncomputable def f (x : ℝ) : ℝ := x^2 - x

theorem find_point_P :
  (∃ x y : ℝ, f x = y ∧ (2 * x - 1 = 1) ∧ (y = x^2 - x)) ∧ (x = 1) ∧ (y = 0) :=
by
  sorry

end find_point_P_l7_7716


namespace surface_area_of_glued_cubes_l7_7976

noncomputable def calculate_surface_area (large_cube_edge_length : ℕ) : ℕ :=
sorry

theorem surface_area_of_glued_cubes :
  calculate_surface_area 4 = 136 :=
sorry

end surface_area_of_glued_cubes_l7_7976


namespace days_to_exceed_50000_l7_7933

noncomputable def A : ℕ → ℝ := sorry
def k : ℝ := sorry
def t := 10
def lg2 := 0.3
def lg (x : ℝ) : ℝ := sorry -- assuming the definition of logarithm base 10

axiom user_count : ∀ t, A t = 500 * Real.exp (k * t)
axiom condition_after_10_days : A 10 = 2000
axiom log_property : lg 2 = lg2

theorem days_to_exceed_50000 : ∃ t : ℕ, t ≥ 34 ∧ 500 * Real.exp (k * t) > 50000 := 
sorry

end days_to_exceed_50000_l7_7933


namespace find_x_l7_7598

/-
If two minus the reciprocal of (3 - x) equals the reciprocal of (2 + x), 
then x equals (1 + sqrt(15)) / 2 or (1 - sqrt(15)) / 2.
-/
theorem find_x (x : ℝ) :
  (2 - (1 / (3 - x)) = (1 / (2 + x))) → 
  (x = (1 + Real.sqrt 15) / 2 ∨ x = (1 - Real.sqrt 15) / 2) :=
by 
  sorry

end find_x_l7_7598


namespace ahmed_goats_correct_l7_7405

-- Definitions based on the conditions given in the problem.
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 5 + 2 * adam_goats
def ahmed_goats : ℕ := andrew_goats - 6

-- The theorem statement that needs to be proven.
theorem ahmed_goats_correct : ahmed_goats = 13 := by
    sorry

end ahmed_goats_correct_l7_7405


namespace derivative_at_0_l7_7238

noncomputable def f : ℝ → ℝ
| x => if x = 0 then 0 else x^2 * Real.exp (|x|) * Real.sin (1 / x^2)

theorem derivative_at_0 : deriv f 0 = 0 := by
  sorry

end derivative_at_0_l7_7238


namespace full_price_tickets_revenue_l7_7660

theorem full_price_tickets_revenue (f h p : ℕ) (h1 : f + h + 12 = 160) (h2 : f * p + h * (p / 2) + 12 * (2 * p) = 2514) :  f * p = 770 := 
sorry

end full_price_tickets_revenue_l7_7660


namespace trig_identity_proofs_l7_7576

theorem trig_identity_proofs (α : ℝ) 
  (h : Real.sin α + Real.cos α = 1 / 5) :
  (Real.sin α - Real.cos α = 7 / 5 ∨ Real.sin α - Real.cos α = -7 / 5) ∧
  (Real.sin α ^ 3 + Real.cos α ^ 3 = 37 / 125) :=
by
  sorry

end trig_identity_proofs_l7_7576


namespace prob_9_correct_matches_is_zero_l7_7198

noncomputable def probability_of_exactly_9_correct_matches : ℝ :=
  let n := 10 in
  -- Since choosing 9 correct implies the 10th is also correct, the probability is 0.
  0

theorem prob_9_correct_matches_is_zero : probability_of_exactly_9_correct_matches = 0 :=
by
  sorry

end prob_9_correct_matches_is_zero_l7_7198


namespace total_paper_clips_l7_7400

/-
Given:
- The number of cartons: c = 3
- The number of boxes: b = 4
- The number of bags: p = 2
- The number of paper clips in each carton: paper_clips_per_carton = 300
- The number of paper clips in each box: paper_clips_per_box = 550
- The number of paper clips in each bag: paper_clips_per_bag = 1200

Prove that the total number of paper clips is 5500.
-/

theorem total_paper_clips :
  let c := 3
  let paper_clips_per_carton := 300
  let b := 4
  let paper_clips_per_box := 550
  let p := 2
  let paper_clips_per_bag := 1200
  (c * paper_clips_per_carton + b * paper_clips_per_box + p * paper_clips_per_bag) = 5500 :=
by
  sorry

end total_paper_clips_l7_7400


namespace range_of_real_number_a_l7_7893

theorem range_of_real_number_a (a : ℝ) : (∀ (x : ℝ), 0 < x → a < x + 1/x) → a < 2 := 
by
  sorry

end range_of_real_number_a_l7_7893


namespace pima_initial_investment_l7_7630

/-- Pima's initial investment in Ethereum. The investment value gained 25% in the first week and 50% of its current value in the second week. The final investment value is $750. -/
theorem pima_initial_investment (I : ℝ) 
  (h1 : 1.25 * I * 1.5 = 750) : I = 400 :=
sorry

end pima_initial_investment_l7_7630


namespace balls_in_boxes_l7_7281

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end balls_in_boxes_l7_7281


namespace problem1_problem2_l7_7272

open Set

variable (a : Real)

-- Problem 1: Prove the intersection M ∩ (C_R N) equals the given set
theorem problem1 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | 3 ≤ x ∧ x ≤ 5 }
  let C_RN := { x : ℝ | x < 3 ∨ 5 < x }
  M ∩ C_RN = { x : ℝ | -2 ≤ x ∧ x < 3 } :=
by
  sorry

-- Problem 2: Prove the range of values for a such that M ∪ N = M
theorem problem2 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | a+1 ≤ x ∧ x ≤ 2*a+1 }
  (M ∪ N = M) → a ≤ 2 :=
by
  sorry

end problem1_problem2_l7_7272


namespace hyperbola_eccentricity_is_sqrt2_l7_7880

noncomputable def eccentricity_of_hyperbola {a b : ℝ} (h : a ≠ 0) (hb : b = a) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  (c / a)

theorem hyperbola_eccentricity_is_sqrt2 {a : ℝ} (h : a ≠ 0) :
  eccentricity_of_hyperbola h (rfl) = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_is_sqrt2_l7_7880


namespace neq_zero_necessary_not_sufficient_l7_7208

theorem neq_zero_necessary_not_sufficient (x : ℝ) (h : x ≠ 0) : 
  (¬ (x = 0) ↔ x > 0) ∧ ¬ (x > 0 → x ≠ 0) :=
by sorry

end neq_zero_necessary_not_sufficient_l7_7208


namespace largest_common_value_less_than_1000_l7_7251

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a < 1000 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 5 + 8 * m) ∧ 
            (∀ b : ℕ, (b < 1000 ∧ (∃ n : ℤ, b = 4 + 5 * n) ∧ (∃ m : ℤ, b = 5 + 8 * m)) → b ≤ a) :=
sorry

end largest_common_value_less_than_1000_l7_7251


namespace divisible_by_bn_l7_7168

variables {u v a b : ℤ} {n : ℕ}

theorem divisible_by_bn 
  (h1 : ∀ x : ℤ, x^2 + a*x + b = 0 → x = u ∨ x = v)
  (h2 : a^2 % b = 0) 
  (h3 : ∀ m : ℕ, m = 2 * n) : 
  ∀ n : ℕ, (u^m + v^m) % (b^n) = 0 := 
  sorry

end divisible_by_bn_l7_7168


namespace yesterday_tomorrow_is_friday_l7_7464

-- Defining the days of the week
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to go to the next day
def next_day : Day → Day
| Sunday    => Monday
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday

-- Function to go to the previous day
def previous_day : Day → Day
| Sunday    => Saturday
| Monday    => Sunday
| Tuesday   => Monday
| Wednesday => Tuesday
| Thursday  => Wednesday
| Friday    => Thursday
| Saturday  => Friday

-- Proving the statement
theorem yesterday_tomorrow_is_friday (T : Day) (H : next_day (previous_day T) = Thursday) : previous_day (next_day (next_day T)) = Friday :=
by
  sorry

end yesterday_tomorrow_is_friday_l7_7464


namespace total_time_per_week_l7_7360

noncomputable def meditating_time_per_day : ℝ := 1
noncomputable def reading_time_per_day : ℝ := 2 * meditating_time_per_day
noncomputable def exercising_time_per_day : ℝ := 0.5 * meditating_time_per_day
noncomputable def practicing_time_per_day : ℝ := (1/3) * reading_time_per_day

noncomputable def total_time_per_day : ℝ :=
  meditating_time_per_day + reading_time_per_day + exercising_time_per_day + practicing_time_per_day

theorem total_time_per_week :
  total_time_per_day * 7 = 29.17 := by
  sorry

end total_time_per_week_l7_7360


namespace sequence_term_1000_l7_7105

open Nat

theorem sequence_term_1000 :
  (∃ b : ℕ → ℤ,
    b 1 = 3010 ∧
    b 2 = 3011 ∧
    (∀ n, 1 ≤ n → b n + b (n + 1) + b (n + 2) = n + 4) ∧
    b 1000 = 3343) :=
sorry

end sequence_term_1000_l7_7105


namespace quotient_of_division_l7_7011

theorem quotient_of_division (S L Q : ℕ) (h1 : S = 270) (h2 : L - S = 1365) (h3 : L % S = 15) : Q = 6 :=
by
  sorry

end quotient_of_division_l7_7011


namespace parallel_planes_of_perpendicular_lines_l7_7923

-- Definitions of planes and lines
variable (Plane Line : Type)
variable (α β γ : Plane)
variable (m n : Line)

-- Relations between planes and lines
variable (perpendicular : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Conditions for the proof
variable (m_perp_α : perpendicular α m)
variable (n_perp_β : perpendicular β n)
variable (m_par_n : line_parallel m n)

-- Statement of the theorem
theorem parallel_planes_of_perpendicular_lines :
  parallel α β :=
sorry

end parallel_planes_of_perpendicular_lines_l7_7923


namespace meaningful_fraction_l7_7967

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by {
  sorry -- Proof goes here
}

end meaningful_fraction_l7_7967


namespace savings_after_four_weeks_l7_7631

noncomputable def hourly_wage (name : String) : ℝ :=
  match name with
  | "Robby" | "Jaylen" | "Miranda" => 10
  | "Alex" => 12
  | "Beth" => 15
  | "Chris" => 20
  | _ => 0

noncomputable def daily_hours (name : String) : ℝ :=
  match name with
  | "Robby" | "Miranda" => 10
  | "Jaylen" => 8
  | "Alex" => 6
  | "Beth" => 4
  | "Chris" => 3
  | _ => 0

noncomputable def saving_rate (name : String) : ℝ :=
  match name with
  | "Robby" => 2/5
  | "Jaylen" => 3/5
  | "Miranda" => 1/2
  | "Alex" => 1/3
  | "Beth" => 1/4
  | "Chris" => 3/4
  | _ => 0

noncomputable def weekly_earning (name : String) : ℝ :=
  hourly_wage name * daily_hours name * 5

noncomputable def weekly_saving (name : String) : ℝ :=
  weekly_earning name * saving_rate name

noncomputable def combined_savings : ℝ :=
  4 * (weekly_saving "Robby" + 
       weekly_saving "Jaylen" + 
       weekly_saving "Miranda" + 
       weekly_saving "Alex" + 
       weekly_saving "Beth" + 
       weekly_saving "Chris")

theorem savings_after_four_weeks :
  combined_savings = 4440 :=
by
  sorry

end savings_after_four_weeks_l7_7631


namespace fraction_irreducible_iff_l7_7854

-- Define the condition for natural number n
def is_natural (n : ℕ) : Prop :=
  True  -- All undergraduate natural numbers abide to True

-- Main theorem formalized in Lean 4
theorem fraction_irreducible_iff (n : ℕ) :
  (∃ (g : ℕ), g = 1 ∧ (∃ a b : ℕ, 2 * n * n + 11 * n - 18 = a * g ∧ n + 7 = b * g)) ↔ 
  (n % 3 = 0 ∨ n % 3 = 1) :=
by sorry

end fraction_irreducible_iff_l7_7854


namespace mary_cards_left_l7_7759

noncomputable def mary_initial_cards : ℝ := 18.0
noncomputable def cards_to_fred : ℝ := 26.0
noncomputable def cards_bought : ℝ := 40.0
noncomputable def mary_final_cards : ℝ := 32.0

theorem mary_cards_left :
  (mary_initial_cards + cards_bought) - cards_to_fred = mary_final_cards := 
by 
  sorry

end mary_cards_left_l7_7759


namespace option_c_correct_l7_7292

theorem option_c_correct (x y : ℝ) (h : x < y) : -x > -y := 
sorry

end option_c_correct_l7_7292


namespace expected_min_leq_2_l7_7325

open ProbabilityTheory

variables (ξ η : ℝ → ℝ) -- ξ and η are random variables

-- Condition: expected value of ξ is 2
axiom E_ξ_eq_2 : ℝ
axiom E_ξ_is_2 : (∫ x in ⊤, ξ x) = 2

-- Goal: expected value of min(ξ, η) ≤ 2
theorem expected_min_leq_2 (h : ∀ x, min (ξ x) (η x) ≤ ξ x) : 
  (∫ x in ⊤, min (ξ x) (η x)) ≤ 2 := by
  -- use the provided axioms and conditions here
  sorry

end expected_min_leq_2_l7_7325


namespace compute_expression_l7_7824

theorem compute_expression :
  3 * 3^4 - 9^60 / 9^57 = -486 :=
by
  sorry

end compute_expression_l7_7824


namespace avg_growth_rate_eq_l7_7743

variable (x : ℝ)

theorem avg_growth_rate_eq :
  (560 : ℝ) * (1 + x)^2 = 830 :=
sorry

end avg_growth_rate_eq_l7_7743


namespace solve_for_x_minus_y_l7_7196

theorem solve_for_x_minus_y (x y : ℝ) 
  (h1 : 3 * x - 5 * y = 5)
  (h2 : x / (x + y) = 5 / 7) : 
  x - y = 3 := 
by 
  -- Proof would go here
  sorry

end solve_for_x_minus_y_l7_7196


namespace simplify_expression_l7_7002

-- Defining each term in the sum
def term1  := 1 / ((1 / 3) ^ 1)
def term2  := 1 / ((1 / 3) ^ 2)
def term3  := 1 / ((1 / 3) ^ 3)

-- Sum of the terms
def terms_sum := term1 + term2 + term3

-- The simplification proof statement
theorem simplify_expression : 1 / terms_sum = 1 / 39 :=
by sorry

end simplify_expression_l7_7002


namespace exists_root_in_interval_l7_7726

noncomputable def f (x : ℝ) : ℝ := (6 / x) - Real.logBase 2 x

theorem exists_root_in_interval :
  (∃ c ∈ Ioo (2 : ℝ) 4, f c = 0) :=
by
  have f_cont : ContinuousOn f (Ioo (2 : ℝ) 4) := sorry,
  have f_2_pos : f 2 > 0 := sorry,
  have f_4_neg : f 4 < 0 := sorry,
  have := IntermediateValueTheorem,
  apply this,
  exact f_cont,
  split,
  { exact f_2_pos },
  { exact f_4_neg }

end exists_root_in_interval_l7_7726


namespace minimize_distance_on_ellipse_l7_7885

theorem minimize_distance_on_ellipse (a m n : ℝ) (hQ : 0 < a ∧ a ≠ Real.sqrt 3)
  (hP : m^2 / 3 + n^2 / 2 = 1) :
  |minimize_distance| = Real.sqrt 3 ∨ |minimize_distance| = 3 * a := sorry

end minimize_distance_on_ellipse_l7_7885


namespace carla_marbles_start_l7_7679

-- Conditions defined as constants
def marblesBought : ℝ := 489.0
def marblesTotalNow : ℝ := 2778.0

-- Theorem statement
theorem carla_marbles_start (marblesBought marblesTotalNow: ℝ) :
  marblesTotalNow - marblesBought = 2289.0 := by
  sorry

end carla_marbles_start_l7_7679


namespace complex_exp_identity_l7_7461

theorem complex_exp_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end complex_exp_identity_l7_7461


namespace smallest_clock_equivalent_number_l7_7771

theorem smallest_clock_equivalent_number :
  ∃ h : ℕ, h > 4 ∧ h^2 % 24 = h % 24 ∧ h = 12 := by
  sorry

end smallest_clock_equivalent_number_l7_7771


namespace smallest_possible_sector_angle_l7_7608

theorem smallest_possible_sector_angle : ∃ a₁ d : ℕ, 2 * a₁ + 9 * d = 72 ∧ a₁ = 9 :=
by
  sorry

end smallest_possible_sector_angle_l7_7608


namespace abs_sum_zero_l7_7294

theorem abs_sum_zero (a b : ℝ) (h : |a - 5| + |b + 8| = 0) : a + b = -3 := 
sorry

end abs_sum_zero_l7_7294


namespace find_num_female_workers_l7_7212

-- Defining the given constants and equations
def num_male_workers : Nat := 20
def num_child_workers : Nat := 5
def wage_male_worker : Nat := 35
def wage_female_worker : Nat := 20
def wage_child_worker : Nat := 8
def avg_wage_paid : Nat := 26

-- Defining the total number of workers and total daily wage
def total_workers (num_female_workers : Nat) : Nat := 
  num_male_workers + num_female_workers + num_child_workers

def total_wage (num_female_workers : Nat) : Nat :=
  (num_male_workers * wage_male_worker) + (num_female_workers * wage_female_worker) + (num_child_workers * wage_child_worker)

-- Proving the number of female workers given the average wage
theorem find_num_female_workers (F : Nat) 
  (h : avg_wage_paid * total_workers F = total_wage F) : 
  F = 15 :=
by
  sorry

end find_num_female_workers_l7_7212


namespace positive_difference_is_9107_03_l7_7060

noncomputable def Cedric_balance : ℝ :=
  15000 * (1 + 0.06) ^ 20

noncomputable def Daniel_balance : ℝ :=
  15000 * (1 + 20 * 0.08)

noncomputable def Elaine_balance : ℝ :=
  15000 * (1 + 0.055 / 2) ^ 40

-- Positive difference between highest and lowest balances.
noncomputable def positive_difference : ℝ :=
  let highest := max Cedric_balance (max Daniel_balance Elaine_balance)
  let lowest := min Cedric_balance (min Daniel_balance Elaine_balance)
  highest - lowest

theorem positive_difference_is_9107_03 :
  positive_difference = 9107.03 := by
  sorry

end positive_difference_is_9107_03_l7_7060


namespace solve_complex_equation_l7_7861

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l7_7861


namespace solve_equation_l7_7805

theorem solve_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := 
by
  sorry  -- Placeholder for the proof

end solve_equation_l7_7805


namespace closest_ratio_one_l7_7783

theorem closest_ratio_one (a c : ℕ) (h1 : 30 * a + 15 * c = 2700) (h2 : a ≥ 1) (h3 : c ≥ 1) :
  a = c :=
by sorry

end closest_ratio_one_l7_7783


namespace bug_visits_all_vertices_in_three_moves_l7_7534

-- Define the vertices of the tetrahedron
inductive Vertex
  | A | B | C | D

-- Define the move relation between vertices (adjacent vertices)
def move : Vertex -> Vertex -> Prop
| Vertex.A, Vertex.B => true
| Vertex.A, Vertex.C => true
| Vertex.A, Vertex.D => true
| Vertex.B, Vertex.A => true
| Vertex.B, Vertex.C => true
| Vertex.B, Vertex.D => true
| Vertex.C, Vertex.A => true
| Vertex.C, Vertex.B => true
| Vertex.C, Vertex.D => true
| Vertex.D, Vertex.A => true
| Vertex.D, Vertex.B => true
| Vertex.D, Vertex.C => true
| _, _ => false

-- Define a function for the probability that a bug starting at a vertex visits all vertices within 3 moves
noncomputable def probability_visits_all_vertices : ℚ :=
  -- Placeholder for the actual computation
  (2 / 9 : ℚ)

-- Proof statement
theorem bug_visits_all_vertices_in_three_moves :
  probability_visits_all_vertices = (2 / 9 : ℚ) :=
sorry

end bug_visits_all_vertices_in_three_moves_l7_7534


namespace max_discount_rate_l7_7220

-- Define the cost price and selling price.
def cp : ℝ := 4
def sp : ℝ := 5

-- Define the minimum profit margin.
def min_profit_margin : ℝ := 0.4

-- Define the discount rate d.
def discount_rate (d : ℝ) : ℝ := sp * (1 - d / 100) - cp

-- The theorem to prove the maximum discount rate.
theorem max_discount_rate (d : ℝ) (H : discount_rate d ≥ min_profit_margin) : d ≤ 12 :=
sorry

end max_discount_rate_l7_7220


namespace arithmetic_sequence_difference_l7_7187

theorem arithmetic_sequence_difference :
  ∀ (a d : ℤ), a = -2 → d = 7 →
  |(a + (3010 - 1) * d) - (a + (3000 - 1) * d)| = 70 :=
by
  intros a d a_def d_def
  rw [a_def, d_def]
  sorry

end arithmetic_sequence_difference_l7_7187


namespace last_two_digits_of_product_squared_l7_7531

def mod_100 (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_product_squared :
  mod_100 ((301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 2) = 76 := 
by
  sorry

end last_two_digits_of_product_squared_l7_7531


namespace expected_value_of_groups_l7_7172

noncomputable def expectedNumberOfGroups (k m : ℕ) : ℝ :=
  1 + (2 * k * m) / (k + m)

theorem expected_value_of_groups (k m : ℕ) :
  k > 0 → m > 0 → expectedNumberOfGroups k m = 1 + 2 * k * m / (k + m) :=
by
  intros
  unfold expectedNumberOfGroups
  sorry

end expected_value_of_groups_l7_7172


namespace arrangement_count_l7_7513

-- Definitions corresponding to the given problem conditions
def numMathBooks : Nat := 3
def numPhysicsBooks : Nat := 2
def numChemistryBooks : Nat := 1
def totalArrangements : Nat := 2592

-- Statement of the theorem
theorem arrangement_count :
  ∃ (numM numP numC : Nat), 
    numM = 3 ∧ 
    numP = 2 ∧ 
    numC = 1 ∧ 
    (numM + numP + numC = 6) ∧ 
    allMathBooksAdjacent ∧ 
    physicsBooksNonAdjacent → 
    totalArrangements = 2592 :=
by
  sorry

end arrangement_count_l7_7513


namespace total_cotton_yield_l7_7022

variables {m n a b : ℕ}

theorem total_cotton_yield (m n a b : ℕ) : 
  m * a + n * b = m * a + n * b := by
  sorry

end total_cotton_yield_l7_7022


namespace indistinguishable_balls_boxes_l7_7285

open Finset

def partitions (n : ℕ) (k : ℕ) : ℕ :=
  (univ : Finset (Multiset ℕ)).filter (λ p, p.sum = n ∧ p.card ≤ k).card

theorem indistinguishable_balls_boxes : partitions 6 4 = 9 :=
sorry

end indistinguishable_balls_boxes_l7_7285


namespace half_hour_half_circle_half_hour_statement_is_true_l7_7916

-- Definitions based on conditions
def half_circle_divisions : ℕ := 30
def small_divisions_per_minute : ℕ := 1
def total_small_divisions : ℕ := 60
def minutes_per_circle : ℕ := 60

-- Relation of small divisions and time taken
def time_taken_for_small_divisions (divs : ℕ) : ℕ := divs * small_divisions_per_minute

-- Theorem to prove the statement
theorem half_hour_half_circle : time_taken_for_small_divisions half_circle_divisions = 30 :=
by
  -- Given half circle covers 30 small divisions
  -- Each small division represents 1 minute
  -- Therefore, time taken for 30 divisions should be 30 minutes
  exact rfl

-- The final statement proving the truth of the condition
theorem half_hour_statement_is_true : 
  (time_taken_for_small_divisions half_circle_divisions = 30) → True :=
by
  intro h
  trivial

end half_hour_half_circle_half_hour_statement_is_true_l7_7916


namespace pencil_partition_l7_7900

theorem pencil_partition (total_length green_fraction green_length remaining_length white_fraction half_remaining white_length gold_length : ℝ)
  (h1 : green_fraction = 7 / 10)
  (h2 : total_length = 2)
  (h3 : green_length = green_fraction * total_length)
  (h4 : remaining_length = total_length - green_length)
  (h5 : white_fraction = 1 / 2)
  (h6 : white_length = white_fraction * remaining_length)
  (h7 : gold_length = remaining_length - white_length) :
  (gold_length / remaining_length) = 1 / 2 :=
sorry

end pencil_partition_l7_7900


namespace Aarti_work_days_l7_7404

theorem Aarti_work_days (x : ℕ) : (3 * x = 24) → x = 8 := by
  intro h
  linarith

end Aarti_work_days_l7_7404


namespace smallest_square_side_length_l7_7071

theorem smallest_square_side_length :
  ∃ (n s : ℕ),  14 * n = s^2 ∧ s = 14 := 
by
  existsi 14, 14
  sorry

end smallest_square_side_length_l7_7071


namespace inequality_proof_l7_7877

noncomputable def a := (1 / 4) * Real.logb 2 3
noncomputable def b := 1 / 2
noncomputable def c := (1 / 2) * Real.logb 5 3

theorem inequality_proof : c < a ∧ a < b :=
by
  sorry

end inequality_proof_l7_7877


namespace range_tan_squared_plus_tan_plus_one_l7_7851

theorem range_tan_squared_plus_tan_plus_one :
  (∀ y, ∃ x : ℝ, x ≠ (k : ℤ) * Real.pi + Real.pi / 2 → y = Real.tan x ^ 2 + Real.tan x + 1) ↔ 
  ∀ y, y ∈ Set.Ici (3 / 4) :=
sorry

end range_tan_squared_plus_tan_plus_one_l7_7851


namespace range_of_a_l7_7266

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end range_of_a_l7_7266


namespace sasha_made_50_muffins_l7_7777

/-- 
Sasha made some chocolate muffins for her school bake sale fundraiser. Melissa made 4 times as many 
muffins as Sasha, and Tiffany made half of Sasha and Melissa's total number of muffins. They 
contributed $900 to the fundraiser by selling muffins at $4 each. Prove that Sasha made 50 muffins.
-/
theorem sasha_made_50_muffins 
  (S : ℕ)
  (Melissa_made : ℕ := 4 * S)
  (Tiffany_made : ℕ := (1 / 2) * (S + Melissa_made))
  (Total_muffins : ℕ := S + Melissa_made + Tiffany_made)
  (total_income : ℕ := 900)
  (price_per_muffin : ℕ := 4)
  (muffins_sold : ℕ := total_income / price_per_muffin)
  (eq_muffins_sold : Total_muffins = muffins_sold) : 
  S = 50 := 
by sorry

end sasha_made_50_muffins_l7_7777


namespace triangle_perimeter_l7_7581

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 3) 
  (h2 : b = 5) 
  (hc : c ^ 2 - 3 * c = c - 3) 
  (h3 : 3 + 3 > 5) 
  (h4 : 3 + 5 > 3) 
  (h5 : 5 + 3 > 3) : 
  a + b + c = 11 :=
by
  sorry

end triangle_perimeter_l7_7581


namespace remaining_walking_time_is_30_l7_7143

-- Define all the given conditions
def total_distance_to_store : ℝ := 2.5
def distance_already_walked : ℝ := 1.0
def time_per_mile : ℝ := 20.0

-- Define the target remaining walking time
def remaining_distance : ℝ := total_distance_to_store - distance_already_walked
def remaining_time : ℝ := remaining_distance * time_per_mile

-- Prove the remaining walking time is 30 minutes
theorem remaining_walking_time_is_30 : remaining_time = 30 :=
by
  -- Formal proof would go here using corresponding Lean tactics
  sorry

end remaining_walking_time_is_30_l7_7143


namespace pages_copied_l7_7296

theorem pages_copied (cost_per_page : ℕ) (amount_in_dollars : ℕ)
    (cents_per_dollar : ℕ) (total_cents : ℕ) 
    (pages : ℕ)
    (h1 : cost_per_page = 3)
    (h2 : amount_in_dollars = 25)
    (h3 : cents_per_dollar = 100)
    (h4 : total_cents = amount_in_dollars * cents_per_dollar)
    (h5 : total_cents = 2500)
    (h6 : pages = total_cents / cost_per_page) :
  pages = 833 := 
sorry

end pages_copied_l7_7296


namespace collinear_points_sum_l7_7419

theorem collinear_points_sum (p q : ℝ) 
  (h1 : p = 2) (h2 : q = 4) 
  (collinear : ∃ (s : ℝ), 
     (2, p, q) = (2, s*p, s*q) ∧ 
     (p, 3, q) = (s*p, 3, s*q) ∧ 
     (p, q, 4) = (s*p, s*q, 4)): 
  p + q = 6 := by
  sorry

end collinear_points_sum_l7_7419


namespace option_one_correct_l7_7237

theorem option_one_correct (x : ℝ) : 
  (x ≠ 0 → x + |x| > 0) ∧ ¬((x + |x| > 0) → x ≠ 0) := 
by
  sorry

end option_one_correct_l7_7237


namespace gross_revenue_is_47_l7_7526

def total_net_profit : ℤ := 44
def babysitting_profit : ℤ := 31
def lemonade_stand_expense : ℤ := 34

def gross_revenue_from_lemonade_stand (P_t P_b E : ℤ) : ℤ :=
  P_t - P_b + E

theorem gross_revenue_is_47 :
  gross_revenue_from_lemonade_stand total_net_profit babysitting_profit lemonade_stand_expense = 47 :=
by
  sorry

end gross_revenue_is_47_l7_7526


namespace baby_turtles_on_sand_l7_7669

theorem baby_turtles_on_sand (total_swept : ℕ) (total_hatched : ℕ) (h1 : total_hatched = 42) (h2 : total_swept = total_hatched / 3) :
  total_hatched - total_swept = 28 := by
  sorry

end baby_turtles_on_sand_l7_7669


namespace sum_of_common_divisors_60_18_l7_7070

def positive_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n+1))

def common_divisors (m n : ℕ) : List ℕ :=
  List.filter (λ d, d ∈ positive_divisors m) (positive_divisors n)

theorem sum_of_common_divisors_60_18 : 
  List.sum (common_divisors 60 18) = 12 := by
  sorry

end sum_of_common_divisors_60_18_l7_7070


namespace chi_square_confidence_l7_7904

theorem chi_square_confidence (chi_square : ℝ) (df : ℕ) (critical_value : ℝ) :
  chi_square = 6.825 ∧ df = 1 ∧ critical_value = 6.635 → confidence_level = 0.99 := 
by
  sorry

end chi_square_confidence_l7_7904


namespace flashlight_lifetime_expectation_leq_two_l7_7315

noncomputable def min_lifetime_expectation (ξ η : ℝ) (E_ξ : ℝ) : Prop :=
  E_ξ = 2 → E(min ξ η) ≤ 2

-- Assume ξ and η are random variables and E denotes the expectation.
axiom E : (ℝ → ℝ) → ℝ

theorem flashlight_lifetime_expectation_leq_two (ξ η : ℝ) (E_ξ : ℝ) (hE_ξ : E_ξ = 2) : E(min ξ η) ≤ 2 :=
  by
    sorry

end flashlight_lifetime_expectation_leq_two_l7_7315


namespace greatest_of_consecutive_integers_with_sum_39_l7_7519

theorem greatest_of_consecutive_integers_with_sum_39 :
  ∃ x : ℤ, x + (x + 1) + (x + 2) = 39 ∧ max (max x (x + 1)) (x + 2) = 14 :=
by
  sorry

end greatest_of_consecutive_integers_with_sum_39_l7_7519


namespace total_wheels_l7_7057

def regular_bikes := 7
def children_bikes := 11
def tandem_bikes_4_wheels := 5
def tandem_bikes_6_wheels := 3
def unicycles := 4
def tricycles := 6
def bikes_with_training_wheels := 8

def wheels_regular := 2
def wheels_children := 4
def wheels_tandem_4 := 4
def wheels_tandem_6 := 6
def wheel_unicycle := 1
def wheels_tricycle := 3
def wheels_training := 4

theorem total_wheels : 
  (regular_bikes * wheels_regular) +
  (children_bikes * wheels_children) + 
  (tandem_bikes_4_wheels * wheels_tandem_4) + 
  (tandem_bikes_6_wheels * wheels_tandem_6) + 
  (unicycles * wheel_unicycle) + 
  (tricycles * wheels_tricycle) + 
  (bikes_with_training_wheels * wheels_training) 
  = 150 := by
  sorry

end total_wheels_l7_7057


namespace simplify_expression_l7_7000

-- Defining each term in the sum
def term1  := 1 / ((1 / 3) ^ 1)
def term2  := 1 / ((1 / 3) ^ 2)
def term3  := 1 / ((1 / 3) ^ 3)

-- Sum of the terms
def terms_sum := term1 + term2 + term3

-- The simplification proof statement
theorem simplify_expression : 1 / terms_sum = 1 / 39 :=
by sorry

end simplify_expression_l7_7000


namespace probability_60_or_more_points_l7_7036

theorem probability_60_or_more_points :
  let five_choose k := Nat.choose 5 k
  let prob_correct (k : Nat) := (five_choose k) * (1 / 2)^5
  let prob_at_least_3_correct := prob_correct 3 + prob_correct 4 + prob_correct 5
  prob_at_least_3_correct = 1 / 2 := 
sorry

end probability_60_or_more_points_l7_7036


namespace minimum_bag_count_l7_7234

theorem minimum_bag_count (n a b : ℕ) (h1 : 7 * a + 11 * b = 77) (h2 : a + b = n) : n = 17 :=
by
  sorry

end minimum_bag_count_l7_7234


namespace number_of_real_roots_l7_7746

def equation (x : ℝ) : Prop := 2 * Real.sqrt (x - 3) + 6 = x

theorem number_of_real_roots : ∃! x : ℝ, equation x := by
  -- Proof goes here
  sorry

end number_of_real_roots_l7_7746


namespace min_sum_of_factors_of_72_l7_7126

theorem min_sum_of_factors_of_72 (a b: ℤ) (h: a * b = 72) : a + b = -73 :=
sorry

end min_sum_of_factors_of_72_l7_7126


namespace exists_positive_m_f99_divisible_1997_l7_7754

def f (x : ℕ) : ℕ := 3 * x + 2

noncomputable
def higher_order_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => sorry  -- placeholder since f^0 isn't defined in this context
  | 1 => f x
  | k + 1 => f (higher_order_f k x)

theorem exists_positive_m_f99_divisible_1997 :
  ∃ m : ℕ, m > 0 ∧ higher_order_f 99 m % 1997 = 0 :=
sorry

end exists_positive_m_f99_divisible_1997_l7_7754


namespace sum_and_product_of_conjugates_l7_7907

theorem sum_and_product_of_conjugates (c d : ℚ) 
  (h1 : 2 * c = 6)
  (h2 : c^2 - 4 * d = 4) :
  c + d = 17 / 4 :=
by
  sorry

end sum_and_product_of_conjugates_l7_7907


namespace wide_flags_made_l7_7562

theorem wide_flags_made
  (initial_fabric : ℕ) (square_flag_side : ℕ) (wide_flag_width : ℕ) (wide_flag_height : ℕ)
  (tall_flag_width : ℕ) (tall_flag_height : ℕ) (made_square_flags : ℕ) (made_tall_flags : ℕ)
  (remaining_fabric : ℕ) (used_fabric_for_small_flags : ℕ) (used_fabric_for_tall_flags : ℕ)
  (used_fabric_for_wide_flags : ℕ) (wide_flag_area : ℕ) :
    initial_fabric = 1000 →
    square_flag_side = 4 →
    wide_flag_width = 5 →
    wide_flag_height = 3 →
    tall_flag_width = 3 →
    tall_flag_height = 5 →
    made_square_flags = 16 →
    made_tall_flags = 10 →
    remaining_fabric = 294 →
    used_fabric_for_small_flags = 256 →
    used_fabric_for_tall_flags = 150 →
    used_fabric_for_wide_flags = initial_fabric - remaining_fabric - (used_fabric_for_small_flags + used_fabric_for_tall_flags) →
    wide_flag_area = wide_flag_width * wide_flag_height →
    (used_fabric_for_wide_flags / wide_flag_area) = 20 :=
by
  intros; 
  sorry

end wide_flags_made_l7_7562


namespace flashlight_lifetime_expectation_leq_two_l7_7313

noncomputable def min_lifetime_expectation (ξ η : ℝ) (E_ξ : ℝ) : Prop :=
  E_ξ = 2 → E(min ξ η) ≤ 2

-- Assume ξ and η are random variables and E denotes the expectation.
axiom E : (ℝ → ℝ) → ℝ

theorem flashlight_lifetime_expectation_leq_two (ξ η : ℝ) (E_ξ : ℝ) (hE_ξ : E_ξ = 2) : E(min ξ η) ≤ 2 :=
  by
    sorry

end flashlight_lifetime_expectation_leq_two_l7_7313


namespace inequality_transitive_l7_7444

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) (h3 : c ≠ 0) (h4 : d ≠ 0) :
  a + c > b + d :=
by {
  sorry
}

end inequality_transitive_l7_7444


namespace pond_87_5_percent_algae_free_on_day_17_l7_7784

/-- The algae in a local pond doubles every day. -/
def algae_doubles_every_day (coverage : ℕ → ℝ) : Prop :=
  ∀ n, coverage (n + 1) = 2 * coverage n

/-- The pond is completely covered in algae on day 20. -/
def pond_completely_covered_on_day_20 (coverage : ℕ → ℝ) : Prop :=
  coverage 20 = 1

/-- Determine the day on which the pond was 87.5% algae-free. -/
theorem pond_87_5_percent_algae_free_on_day_17 (coverage : ℕ → ℝ)
  (h1 : algae_doubles_every_day coverage)
  (h2 : pond_completely_covered_on_day_20 coverage) :
  coverage 17 = 0.125 :=
sorry

end pond_87_5_percent_algae_free_on_day_17_l7_7784


namespace debby_weekly_jog_distance_l7_7139

theorem debby_weekly_jog_distance :
  let monday_distance := 3.0
  let tuesday_distance := 5.5
  let wednesday_distance := 9.7
  let thursday_distance := 10.8
  let friday_distance_miles := 2.0
  let miles_to_km := 1.60934
  let friday_distance := friday_distance_miles * miles_to_km
  let total_distance := monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance
  total_distance = 32.21868 :=
by
  sorry

end debby_weekly_jog_distance_l7_7139


namespace percentage_increase_selling_price_l7_7103

-- Defining the conditions
def original_price : ℝ := 6
def increased_price : ℝ := 8.64
def total_sales_per_hour : ℝ := 216
def max_price : ℝ := 10

-- Statement for Part 1
theorem percentage_increase (x : ℝ) : 6 * (1 + x)^2 = 8.64 → x = 0.2 :=
by
  sorry

-- Statement for Part 2
theorem selling_price (a : ℝ) : (6 + a) * (30 - 2 * a) = 216 → 6 + a ≤ 10 → 6 + a = 9 :=
by
  sorry

end percentage_increase_selling_price_l7_7103


namespace problem1_problem2_problem3_l7_7587

noncomputable def f (x a : ℝ) : ℝ := abs x * (x - a)

-- 1. Prove a = 0 if f(x) is odd
theorem problem1 (h: ∀ x : ℝ, f (-x) a = -f x a) : a = 0 :=
sorry

-- 2. Prove a ≤ 0 if f(x) is increasing on the interval [0, 2]
theorem problem2 (h: ∀ x y : ℝ, 0 ≤ x → x ≤ y → y ≤ 2 → f x a ≤ f y a) : a ≤ 0 :=
sorry

-- 3. Prove there exists an a < 0 such that the maximum value of f(x) on [-1, 1/2] is 2, and find a = -3
theorem problem3 (h: ∃ a : ℝ, a < 0 ∧ ∀ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a ≤ 2 ∧ ∃ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a = 2) : a = -3 :=
sorry

end problem1_problem2_problem3_l7_7587


namespace solve_equation_l7_7633

open Real

theorem solve_equation (t : ℝ) :
  ¬cos t = 0 ∧ ¬cos (2 * t) = 0 → 
  (tan (2 * t) / (cos t)^2 - tan t / (cos (2 * t))^2 = 0 ↔ 
    (∃ k : ℤ, t = π * ↑k) ∨ (∃ n : ℤ, t = π * ↑n + π / 6) ∨ (∃ n : ℤ, t = π * ↑n - π / 6)) :=
by
  intros h
  sorry

end solve_equation_l7_7633


namespace students_play_both_football_and_tennis_l7_7599

theorem students_play_both_football_and_tennis 
  (T : ℕ) (F : ℕ) (L : ℕ) (N : ℕ) (B : ℕ)
  (hT : T = 38) (hF : F = 26) (hL : L = 20) (hN : N = 9) :
  B = F + L - (T - N) → B = 17 :=
by 
  intros h
  rw [hT, hF, hL, hN] at h
  exact h

end students_play_both_football_and_tennis_l7_7599


namespace prove_divisibility_l7_7328

-- Definitions for natural numbers m, n, k
variables (m n k : ℕ)

-- Conditions stating divisibility
def div1 := m^n ∣ n^m
def div2 := n^k ∣ k^n

-- The final theorem to prove
theorem prove_divisibility (hmn : div1 m n) (hnk : div2 n k) : m^k ∣ k^m :=
sorry

end prove_divisibility_l7_7328


namespace henry_finishes_on_thursday_l7_7896

theorem henry_finishes_on_thursday :
  let total_days := 210
  let start_day := 4  -- Assume Thursday is 4th day of the week in 0-indexed (0=Sunday, 1=Monday, ..., 6=Saturday)
  (start_day + total_days) % 7 = start_day :=
by
  sorry

end henry_finishes_on_thursday_l7_7896


namespace xiaoxiao_types_faster_l7_7192

-- Defining the characters typed and time taken by both individuals
def characters_typed_taoqi : ℕ := 200
def time_taken_taoqi : ℕ := 5
def characters_typed_xiaoxiao : ℕ := 132
def time_taken_xiaoxiao : ℕ := 3

-- Calculating typing speeds
def speed_taoqi : ℕ := characters_typed_taoqi / time_taken_taoqi
def speed_xiaoxiao : ℕ := characters_typed_xiaoxiao / time_taken_xiaoxiao

-- Proving that 笑笑 types faster
theorem xiaoxiao_types_faster : speed_xiaoxiao > speed_taoqi := by
  -- Given calculations:
  -- speed_taoqi = 40
  -- speed_xiaoxiao = 44
  sorry

end xiaoxiao_types_faster_l7_7192


namespace union_M_N_is_U_l7_7619

-- Defining the universal set as the set of real numbers
def U : Set ℝ := Set.univ

-- Defining the set M
def M : Set ℝ := {x | x > 0}

-- Defining the set N
def N : Set ℝ := {x | x^2 >= x}

-- Stating the theorem that M ∪ N = U
theorem union_M_N_is_U : M ∪ N = U :=
  sorry

end union_M_N_is_U_l7_7619


namespace set_difference_M_N_l7_7889

def setM : Set ℝ := { x | -1 < x ∧ x < 1 }
def setN : Set ℝ := { x | x / (x - 1) ≤ 0 }

theorem set_difference_M_N :
  setM \ setN = { x | -1 < x ∧ x < 0 } := sorry

end set_difference_M_N_l7_7889


namespace keith_apples_correct_l7_7487

def mike_apples : ℕ := 7
def nancy_apples : ℕ := 3
def total_apples : ℕ := 16
def keith_apples : ℕ := total_apples - (mike_apples + nancy_apples)

theorem keith_apples_correct : keith_apples = 6 := by
  -- the actual proof would go here
  sorry

end keith_apples_correct_l7_7487


namespace silver_coin_worth_l7_7996

theorem silver_coin_worth :
  ∀ (g : ℕ) (S : ℕ) (n_gold n_silver cash : ℕ), 
  g = 50 →
  n_gold = 3 →
  n_silver = 5 →
  cash = 30 →
  n_gold * g + n_silver * S + cash = 305 →
  S = 25 :=
by
  intros g S n_gold n_silver cash
  intros hg hng hnsi hcash htotal
  sorry

end silver_coin_worth_l7_7996


namespace exists_infinite_irregular_set_l7_7185

def is_irregular (A : Set ℤ) :=
  ∀ ⦃x y : ℤ⦄, x ∈ A → y ∈ A → x ≠ y → ∀ ⦃k : ℤ⦄, x + k * (y - x) ≠ x ∧ x + k * (y - x) ≠ y

theorem exists_infinite_irregular_set : ∃ A : Set ℤ, Set.Infinite A ∧ is_irregular A :=
sorry

end exists_infinite_irregular_set_l7_7185


namespace max_distance_complex_l7_7615

theorem max_distance_complex (z : ℂ) (hz : complex.abs z = 3) : 
  ∃ d, d = complex.abs ((5 + 2 * complex.I) * z^3 - z^5) ∧ d ≤ 99 :=
by sorry

end max_distance_complex_l7_7615


namespace distance_is_660_km_l7_7365

def distance_between_cities (x y : ℝ) : ℝ :=
  3.3 * (x + y)

def train_A_dep_earlier (x y : ℝ) : Prop :=
  3.4 * (x + y) = 3.3 * (x + y) + 14

def train_B_dep_earlier (x y : ℝ) : Prop :=
  3.6 * (x + y) = 3.3 * (x + y) + 9

theorem distance_is_660_km (x y : ℝ) (hx : train_A_dep_earlier x y) (hy : train_B_dep_earlier x y) :
    distance_between_cities x y = 660 :=
sorry

end distance_is_660_km_l7_7365


namespace wastewater_volume_2013_l7_7603

variable (x_2013 x_2014 : ℝ)
variable (condition1 : x_2014 = 38000)
variable (condition2 : x_2014 = 1.6 * x_2013)

theorem wastewater_volume_2013 : x_2013 = 23750 := by
  sorry

end wastewater_volume_2013_l7_7603


namespace average_ducks_l7_7544

theorem average_ducks (a e k : ℕ) 
  (h1 : a = 2 * e) 
  (h2 : e = k - 45) 
  (h3 : a = 30) :
  (a + e + k) / 3 = 35 :=
by
  sorry

end average_ducks_l7_7544


namespace solve_for_z_l7_7863

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l7_7863


namespace books_left_in_library_l7_7015

theorem books_left_in_library (initial_books : ℕ) (borrowed_books : ℕ) (left_books : ℕ) 
  (h1 : initial_books = 75) (h2 : borrowed_books = 18) : left_books = 57 :=
by
  sorry

end books_left_in_library_l7_7015


namespace remaining_walking_time_is_30_l7_7141

-- Define all the given conditions
def total_distance_to_store : ℝ := 2.5
def distance_already_walked : ℝ := 1.0
def time_per_mile : ℝ := 20.0

-- Define the target remaining walking time
def remaining_distance : ℝ := total_distance_to_store - distance_already_walked
def remaining_time : ℝ := remaining_distance * time_per_mile

-- Prove the remaining walking time is 30 minutes
theorem remaining_walking_time_is_30 : remaining_time = 30 :=
by
  -- Formal proof would go here using corresponding Lean tactics
  sorry

end remaining_walking_time_is_30_l7_7141


namespace initial_speed_100kmph_l7_7399

theorem initial_speed_100kmph (v x : ℝ) (h1 : 0 < v) (h2 : 100 - x = v / 2) 
  (h3 : (80 - x) / (v - 10) - 20 / (v - 20) = 1 / 12) : v = 100 :=
by 
  sorry

end initial_speed_100kmph_l7_7399


namespace quadratic_eq_roots_are_coeffs_l7_7384

theorem quadratic_eq_roots_are_coeffs :
  ∃ (a b : ℝ), (a = r_1) → (b = r_2) →
  (r_1 + r_2 = -a) → (r_1 * r_2 = b) →
  r_1 = 1 ∧ r_2 = -2 ∧ (x^2 + x - 2 = 0):=
by
  sorry

end quadratic_eq_roots_are_coeffs_l7_7384


namespace original_number_is_seven_l7_7654

theorem original_number_is_seven (x : ℕ) (h : 3 * x - 5 = 16) : x = 7 := by
sorry

end original_number_is_seven_l7_7654


namespace binomial_probability_l7_7718

noncomputable theory
open Probability

def ξ : ℕ → ℕ := sorry -- Define the binomial random variable appropriately

theorem binomial_probability (p : ℝ) (n k : ℕ) (h_p : p = 1/3) (h_n : n = 3) (h_k : k = 2) :
  P(ξ=k) = (3 choose 2) * (1/3)^2 * (2/3) :=
by
  have h1 : (3 choose 2) = 3 := by norm_num
  have h2 : (1/3)^2 = 1/9 := by norm_num
  have h3 : (1 - 1/3) = 2/3 := by norm_num
  have h4 : (2/3) = 2/3 := by norm_num
  calc
    P(ξ=2) = 3 * (1/9) * (2/3) := by norm_num
           = 2/9 := by norm_num
  sorry

end binomial_probability_l7_7718


namespace probability_red_first_given_black_second_l7_7374

open ProbabilityTheory MeasureTheory

-- Definitions for Urn A and Urn B ball quantities
def urnA := (white : 4, red : 2)
def urnB := (red : 3, black : 3)

-- Event of drawing a red ball first and a black ball second
def eventRedFirst := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "red") ∨ (urn = 2 ∧ ball = "red")
def eventBlackSecond := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "black") ∨ (urn = 2 ∧ ball = "black")

-- Probability function definition
noncomputable def P := sorry -- Probability function placeholder

-- Conditional Probability
theorem probability_red_first_given_black_second :
  P(eventRedFirst | eventBlackSecond) = 2 / 5 := sorry

end probability_red_first_given_black_second_l7_7374


namespace inscribed_square_in_right_triangle_l7_7952

theorem inscribed_square_in_right_triangle
  (DE EF DF : ℝ) (h1 : DE = 5) (h2 : EF = 12) (h3 : DF = 13)
  (t : ℝ) (h4 : t = 780 / 169) :
  (∃ PQRS : ℝ, PQRS = t) :=
begin
  use t,
  exact h4,
  sorry,
end

end inscribed_square_in_right_triangle_l7_7952


namespace expected_lifetime_flashlight_l7_7320

section
variables {Ω : Type} [ProbabilitySpace Ω]
variables (ξ η : Ω → ℝ)
variables (h_ξ_expect : E[ξ] = 2)

-- Define the minimum of ξ and η
def min_ξ_η (ω : Ω) : ℝ := min (ξ ω) (η ω)

theorem expected_lifetime_flashlight : E[min_ξ_η ξ η] ≤ 2 :=
by
  sorry
end

end expected_lifetime_flashlight_l7_7320


namespace min_value_of_x_plus_y_l7_7095

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x > 0) (h2 : y > 0) (h3 : y + 9 * x = x * y)

-- The statement of the problem
theorem min_value_of_x_plus_y : x + y ≥ 16 :=
sorry

end min_value_of_x_plus_y_l7_7095


namespace average_four_numbers_l7_7160

variable {x : ℝ}

theorem average_four_numbers (h : (15 + 25 + x + 30) / 4 = 23) : x = 22 :=
by
  sorry

end average_four_numbers_l7_7160


namespace distinct_prime_factors_330_l7_7274

def num_prime_factors (n : ℕ) : ℕ :=
  if n = 330 then 4 else 0

theorem distinct_prime_factors_330 : num_prime_factors 330 = 4 :=
sorry

end distinct_prime_factors_330_l7_7274


namespace gcf_450_144_l7_7977

theorem gcf_450_144 : Nat.gcd 450 144 = 18 := by
  sorry

end gcf_450_144_l7_7977


namespace find_integers_10_le_n_le_20_mod_7_l7_7695

theorem find_integers_10_le_n_le_20_mod_7 :
  ∃ n, (10 ≤ n ∧ n ≤ 20 ∧ n % 7 = 4) ∧
  (n = 11 ∨ n = 18) := by
  sorry

end find_integers_10_le_n_le_20_mod_7_l7_7695


namespace balls_in_boxes_l7_7283

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end balls_in_boxes_l7_7283


namespace smallest_rel_prime_210_l7_7705

theorem smallest_rel_prime_210 (x : ℕ) (hx : x > 1) (hrel_prime : Nat.gcd x 210 = 1) : x = 11 :=
sorry

end smallest_rel_prime_210_l7_7705


namespace merchant_markup_l7_7666

theorem merchant_markup (C : ℝ) (M : ℝ) (h1 : (1 + M / 100 - 0.40 * (1 + M / 100)) * C = 1.05 * C) : 
  M = 75 := sorry

end merchant_markup_l7_7666


namespace fifth_scroll_age_l7_7663

def scrolls_age (n : ℕ) : ℕ :=
  match n with
  | 0 => 4080
  | k+1 => (3 * scrolls_age k) / 2

theorem fifth_scroll_age : scrolls_age 4 = 20655 := sorry

end fifth_scroll_age_l7_7663


namespace total_beads_needed_l7_7767

-- Condition 1: Number of members in the crafts club
def members := 9

-- Condition 2: Number of necklaces each member makes
def necklaces_per_member := 2

-- Condition 3: Number of beads each necklace requires
def beads_per_necklace := 50

-- Total number of beads needed
theorem total_beads_needed :
  (members * (necklaces_per_member * beads_per_necklace)) = 900 := 
by
  sorry

end total_beads_needed_l7_7767


namespace sum_common_divisors_sixty_and_eighteen_l7_7069

theorem sum_common_divisors_sixty_and_eighteen : 
  ∑ d in ({d ∈ ({1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} : finset ℕ) | d ∈ ({1, 2, 3, 6, 9, 18} : finset ℕ)} : finset ℕ), d = 12 :=
by sorry

end sum_common_divisors_sixty_and_eighteen_l7_7069


namespace sqrt_equiv_1715_l7_7837

noncomputable def sqrt_five_squared_times_seven_sixth : ℕ := 
  Nat.sqrt (5^2 * 7^6)

theorem sqrt_equiv_1715 : sqrt_five_squared_times_seven_sixth = 1715 := by
  sorry

end sqrt_equiv_1715_l7_7837


namespace speed_of_stream_l7_7389

theorem speed_of_stream (v_s : ℝ) (D : ℝ) (h1 : D / (78 - v_s) = 2 * (D / (78 + v_s))) : v_s = 26 :=
by
  sorry

end speed_of_stream_l7_7389


namespace television_price_l7_7764

theorem television_price (SP : ℝ) (RP : ℕ) (discount : ℝ) (h1 : discount = 0.20) (h2 : SP = RP - discount * RP) (h3 : SP = 480) : RP = 600 :=
by
  sorry

end television_price_l7_7764


namespace bus_stops_per_hour_l7_7694

-- Define the constants and conditions given in the problem
noncomputable def speed_without_stoppages : ℝ := 54 -- km/hr
noncomputable def speed_with_stoppages : ℝ := 45 -- km/hr

-- Theorem statement to prove the number of minutes the bus stops per hour
theorem bus_stops_per_hour : (speed_without_stoppages - speed_with_stoppages) / (speed_without_stoppages / 60) = 10 :=
by
  sorry

end bus_stops_per_hour_l7_7694


namespace count_paths_to_form_2005_l7_7635

/-- Define the structure of a circle label. -/
inductive CircleLabel
| two
| zero
| five

open CircleLabel

/-- Define the number of possible moves from each circle. -/
def moves_from_two : Nat := 6
def moves_from_zero_to_zero : Nat := 2
def moves_from_zero_to_five : Nat := 3

/-- Define the total number of paths to form 2005. -/
def total_paths : Nat := moves_from_two * moves_from_zero_to_zero * moves_from_zero_to_five

/-- The proof statement: The total number of different paths to form the number 2005 is 36. -/
theorem count_paths_to_form_2005 : total_paths = 36 :=
by
  sorry

end count_paths_to_form_2005_l7_7635


namespace nat_representation_l7_7421

theorem nat_representation (k : ℕ) : ∃ n r : ℕ, (r = 0 ∨ r = 1 ∨ r = 2) ∧ k = 3 * n + r :=
by
  sorry

end nat_representation_l7_7421


namespace term_2_6_position_l7_7946

theorem term_2_6_position : 
  ∃ (seq : ℕ → ℚ), 
    (seq 23 = 2 / 6) ∧ 
    (∀ n, ∃ k, (n = (k * (k + 1)) / 2 ∧ k > 0 ∧ k <= n)) :=
by sorry

end term_2_6_position_l7_7946


namespace smallest_x_multiple_of_1024_l7_7522

theorem smallest_x_multiple_of_1024 (x : ℕ) (hx : 900 * x % 1024 = 0) : x = 256 :=
sorry

end smallest_x_multiple_of_1024_l7_7522


namespace real_root_in_interval_l7_7696

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem real_root_in_interval : ∃ α : ℝ, f α = 0 ∧ 1 < α ∧ α < 2 :=
sorry

end real_root_in_interval_l7_7696


namespace new_tax_rate_l7_7809

-- Condition definitions
def previous_tax_rate : ℝ := 0.20
def initial_income : ℝ := 1000000
def new_income : ℝ := 1500000
def additional_taxes_paid : ℝ := 250000

-- Theorem statement
theorem new_tax_rate : 
  ∃ T : ℝ, 
    (new_income * T = initial_income * previous_tax_rate + additional_taxes_paid) ∧ 
    T = 0.30 :=
by sorry

end new_tax_rate_l7_7809


namespace time_to_eliminate_mice_l7_7812

def total_work : ℝ := 1
def work_done_by_2_cats_in_5_days : ℝ := 0.5
def initial_2_cats : ℕ := 2
def additional_cats : ℕ := 3
def total_initial_days : ℝ := 5
def total_cats : ℕ := initial_2_cats + additional_cats

theorem time_to_eliminate_mice (h : total_initial_days * (work_done_by_2_cats_in_5_days / total_initial_days) = work_done_by_2_cats_in_5_days) : 
  total_initial_days + (total_work - work_done_by_2_cats_in_5_days) / (total_cats * (work_done_by_2_cats_in_5_days / total_initial_days / initial_2_cats)) = 7 := 
by
  sorry

end time_to_eliminate_mice_l7_7812


namespace correct_equation_l7_7601

-- Definitions of the conditions
def contributes_5_coins (x : ℕ) (P : ℕ) : Prop :=
  5 * x + 45 = P

def contributes_7_coins (x : ℕ) (P : ℕ) : Prop :=
  7 * x + 3 = P

-- Mathematical proof problem
theorem correct_equation 
(x : ℕ) (P : ℕ) (h1 : contributes_5_coins x P) (h2 : contributes_7_coins x P) : 
5 * x + 45 = 7 * x + 3 := 
by
  sorry

end correct_equation_l7_7601


namespace albums_total_l7_7944

noncomputable def miriam_albums (katrina_albums : ℕ) := 5 * katrina_albums
noncomputable def katrina_albums (bridget_albums : ℕ) := 6 * bridget_albums
noncomputable def bridget_albums (adele_albums : ℕ) := adele_albums - 15
noncomputable def total_albums (adele_albums : ℕ) (bridget_albums : ℕ) (katrina_albums : ℕ) (miriam_albums : ℕ) := 
  adele_albums + bridget_albums + katrina_albums + miriam_albums

theorem albums_total (adele_has_30 : adele_albums = 30) : 
  total_albums 30 (bridget_albums 30) (katrina_albums (bridget_albums 30)) (miriam_albums (katrina_albums (bridget_albums 30))) = 585 := 
by 
  -- In Lean, replace the term 'sorry' below with the necessary steps to finish the proof
  sorry

end albums_total_l7_7944


namespace spider_legs_total_l7_7243

def num_spiders : ℕ := 4
def legs_per_spider : ℕ := 8
def total_legs : ℕ := num_spiders * legs_per_spider

theorem spider_legs_total : total_legs = 32 := by
  sorry -- proof is skipped with 'sorry'

end spider_legs_total_l7_7243


namespace student_estimated_score_l7_7997

theorem student_estimated_score :
  (6 * 5 + 3 * 5 * (3 / 4) + 2 * 5 * (1 / 3) + 1 * 5 * (1 / 4)) = 41.25 :=
by
 sorry

end student_estimated_score_l7_7997


namespace cost_per_set_l7_7153

variable (C : ℝ)

theorem cost_per_set :
  let total_manufacturing_cost := 10000 + 500 * C
  let revenue := 500 * 50
  let profit := revenue - total_manufacturing_cost
  profit = 5000 → C = 20 := 
by
  sorry

end cost_per_set_l7_7153


namespace circle_center_l7_7846

theorem circle_center (x y : ℝ) : ∀ (h k : ℝ), (x^2 - 6*x + y^2 + 2*y = 9) → (x - h)^2 + (y - k)^2 = 19 → h = 3 ∧ k = -1 :=
by
  intros h k h_eq c_eq
  sorry

end circle_center_l7_7846


namespace unique_A_value_l7_7839

theorem unique_A_value (A : ℝ) (x1 x2 : ℂ) (hx1_ne : x1 ≠ x2) :
  (x1 * (x1 + 1) = A) ∧ (x2 * (x2 + 1) = A) ∧ (A * x1^4 + 3 * x1^3 + 5 * x1 = x2^4 + 3 * x2^3 + 5 * x2) 
  → A = -7 := by
  sorry

end unique_A_value_l7_7839


namespace emily_required_sixth_score_is_99_l7_7842

/-- Emily's quiz scores and the required mean score -/
def emily_scores : List ℝ := [85, 90, 88, 92, 98]
def required_mean_score : ℝ := 92

/-- The function to calculate the required sixth quiz score for Emily -/
def required_sixth_score (scores : List ℝ) (mean : ℝ) : ℝ :=
  let sum_current := scores.sum
  let total_required := mean * (scores.length + 1)
  total_required - sum_current

/-- Emily needs to score 99 on her sixth quiz for an average of 92 -/
theorem emily_required_sixth_score_is_99 : 
  required_sixth_score emily_scores required_mean_score = 99 :=
by
  sorry

end emily_required_sixth_score_is_99_l7_7842


namespace student_in_eighth_group_l7_7299

-- Defining the problem: total students and their assignment into groups
def total_students : ℕ := 50
def students_assigned_numbers (n : ℕ) : Prop := n > 0 ∧ n ≤ total_students

-- Grouping students: Each group has 5 students
def grouped_students (group_num student_num : ℕ) : Prop := 
  student_num > (group_num - 1) * 5 ∧ student_num ≤ group_num * 5

-- Condition: Student 12 is selected from the third group
def condition : Prop := grouped_students 3 12

-- Goal: the number of the student selected from the eighth group is 37
theorem student_in_eighth_group : condition → grouped_students 8 37 :=
by
  sorry

end student_in_eighth_group_l7_7299


namespace option_D_correct_l7_7189

variable (x : ℝ)

theorem option_D_correct : (2 * x^7) / x = 2 * x^6 := sorry

end option_D_correct_l7_7189


namespace select_terms_from_sequence_l7_7262

theorem select_terms_from_sequence (k : ℕ) (hk : k ≥ 3) :
  ∃ (terms : Fin k → ℚ), (∀ i j : Fin k, i < j → (terms j - terms i) = (j.val - i.val) / k!) ∧
  (∀ i : Fin k, terms i ∈ {x : ℚ | ∃ n : ℕ, x = 1 / (n : ℚ)}) :=
by
  sorry

end select_terms_from_sequence_l7_7262


namespace find_f_of_2_l7_7725

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = 4 * x - 1) : f 2 = 3 :=
by
  sorry

end find_f_of_2_l7_7725


namespace sum_of_interior_numbers_l7_7418

def sum_interior (n : ℕ) : ℕ := 2^(n-1) - 2

theorem sum_of_interior_numbers :
  sum_interior 8 + sum_interior 9 + sum_interior 10 = 890 :=
by
  sorry

end sum_of_interior_numbers_l7_7418


namespace fabric_needed_for_coats_l7_7442

variable (m d : ℝ)

def condition1 := 4 * m + 2 * d = 16
def condition2 := 2 * m + 6 * d = 18

theorem fabric_needed_for_coats (h1 : condition1 m d) (h2 : condition2 m d) :
  m = 3 ∧ d = 2 :=
by
  sorry

end fabric_needed_for_coats_l7_7442


namespace student_l7_7548

noncomputable def allowance_after_video_games (A : ℝ) : ℝ := (3 / 7) * A

noncomputable def allowance_after_comic_books (remaining_after_video_games : ℝ) : ℝ := (3 / 5) * remaining_after_video_games

noncomputable def allowance_after_trading_cards (remaining_after_comic_books : ℝ) : ℝ := (5 / 8) * remaining_after_comic_books

noncomputable def last_allowance (remaining_after_trading_cards : ℝ) : ℝ := remaining_after_trading_cards

theorem student's_monthly_allowance (A : ℝ) (h1 : last_allowance (allowance_after_trading_cards (allowance_after_comic_books (allowance_after_video_games A))) = 1.20) :
  A = 7.47 := 
sorry

end student_l7_7548


namespace domain_correct_l7_7345

def domain_of_function (x : ℝ) : Prop :=
  (x > 2) ∧ (x ≠ 5)

theorem domain_correct : {x : ℝ | domain_of_function x} = {x : ℝ | x > 2 ∧ x ≠ 5} :=
by
  sorry

end domain_correct_l7_7345


namespace derivative_of_y_l7_7250

noncomputable def y (x : ℝ) : ℝ :=
  (4 * x + 1) / (16 * x^2 + 8 * x + 3) + (1 / Real.sqrt 2) * Real.arctan ((4 * x + 1) / Real.sqrt 2)

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 16 / (16 * x^2 + 8 * x + 3)^2 :=
by 
  sorry

end derivative_of_y_l7_7250


namespace x_squared_eq_r_floor_x_has_2_or_3_solutions_l7_7151

theorem x_squared_eq_r_floor_x_has_2_or_3_solutions (r : ℝ) (hr : r > 2) : 
  ∃! (s : Finset ℝ), s.card = 2 ∨ s.card = 3 ∧ ∀ x ∈ s, x^2 = r * (⌊x⌋) :=
by
  sorry

end x_squared_eq_r_floor_x_has_2_or_3_solutions_l7_7151


namespace kyle_gas_and_maintenance_expense_l7_7476

def monthly_income : ℝ := 3200
def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous_expenses : ℝ := 200
def car_payment : ℝ := 350

def total_bills : ℝ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous_expenses

theorem kyle_gas_and_maintenance_expense :
  monthly_income - total_bills - car_payment = 350 :=
by
  sorry

end kyle_gas_and_maintenance_expense_l7_7476


namespace total_number_of_birds_l7_7778

variable (swallows : ℕ) (bluebirds : ℕ) (cardinals : ℕ)
variable (h1 : swallows = 2)
variable (h2 : bluebirds = 2 * swallows)
variable (h3 : cardinals = 3 * bluebirds)

theorem total_number_of_birds : 
  swallows + bluebirds + cardinals = 18 := by
  sorry

end total_number_of_birds_l7_7778


namespace solve_for_z_l7_7858

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l7_7858


namespace sum_product_le_four_l7_7926

theorem sum_product_le_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := 
sorry

end sum_product_le_four_l7_7926


namespace combination_15_3_l7_7686

theorem combination_15_3 :
  (Nat.choose 15 3 = 455) :=
by
  sorry

end combination_15_3_l7_7686


namespace measure_of_angle_A_l7_7906

theorem measure_of_angle_A {A B C : ℝ} (hC : C = 2 * B) (hB : B = 21) :
  A = 180 - B - C := 
by 
  sorry

end measure_of_angle_A_l7_7906


namespace alpine_school_math_students_l7_7554

theorem alpine_school_math_students (total_players : ℕ) (physics_players : ℕ) (both_players : ℕ) :
  total_players = 15 → physics_players = 9 → both_players = 4 → 
  ∃ math_players : ℕ, math_players = total_players - (physics_players - both_players) + both_players := by
  sorry

end alpine_school_math_students_l7_7554


namespace initial_number_of_men_l7_7960

theorem initial_number_of_men (x : ℕ) :
    (50 * x = 25 * (x + 20)) → x = 20 := 
by
  sorry

end initial_number_of_men_l7_7960


namespace gcd_ab_l7_7828

def a := 59^7 + 1
def b := 59^7 + 59^3 + 1

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end gcd_ab_l7_7828


namespace peter_remaining_walk_time_l7_7147

-- Define the parameters and conditions
def total_distance : ℝ := 2.5
def time_per_mile : ℝ := 20
def distance_walked : ℝ := 1

-- Define the remaining distance
def remaining_distance : ℝ := total_distance - distance_walked

-- Define the remaining time Peter needs to walk
def remaining_time_to_walk (d : ℝ) (t : ℝ) : ℝ := d * t

-- State the problem we want to prove
theorem peter_remaining_walk_time :
  remaining_time_to_walk remaining_distance time_per_mile = 30 :=
by
  -- Placeholder for the proof
  sorry

end peter_remaining_walk_time_l7_7147


namespace angle_ADE_l7_7166

-- Definitions and conditions
variable (x : ℝ)

def angle_ABC := 60
def angle_CAD := x
def angle_BAD := x
def angle_BCA := 120 - 2 * x
def angle_DCE := 180 - (120 - 2 * x)

-- Theorem statement
theorem angle_ADE (x : ℝ) : angle_CAD x = x → angle_BAD x = x → angle_ABC = 60 → 
                            angle_DCE x = 180 - angle_BCA x → 
                            120 - 3 * x = 120 - 3 * x := 
by
  intro h1 h2 h3 h4
  sorry

end angle_ADE_l7_7166


namespace ways_to_divide_week_l7_7819

-- Define the total number of seconds in a week
def total_seconds_in_week : ℕ := 604800

-- Define the math problem statement
theorem ways_to_divide_week (n m : ℕ) (h : n * m = total_seconds_in_week) (hn : 0 < n) (hm : 0 < m) : 
  (∃ (n_pairs : ℕ), n_pairs = 144) :=
sorry

end ways_to_divide_week_l7_7819


namespace root_interval_sum_l7_7596

theorem root_interval_sum (a b : Int) (h1 : b - a = 1) (h2 : ∃ x, a < x ∧ x < b ∧ (x^3 - x + 1) = 0) : a + b = -3 := 
sorry

end root_interval_sum_l7_7596


namespace total_time_from_first_station_to_workplace_l7_7763

-- Pick-up time is defined as a constant for clarity in minutes from midnight (6 AM)
def pickup_time_in_minutes : ℕ := 6 * 60

-- Travel time to first station in minutes
def travel_time_to_station_in_minutes : ℕ := 40

-- Arrival time at work (9 AM) in minutes from midnight
def arrival_time_at_work_in_minutes : ℕ := 9 * 60

-- Definition to calculate arrival time at the first station
def arrival_time_at_first_station_in_minutes : ℕ := pickup_time_in_minutes + travel_time_to_station_in_minutes

-- Theorem to prove the total time taken from the first station to the workplace
theorem total_time_from_first_station_to_workplace :
  arrival_time_at_work_in_minutes - arrival_time_at_first_station_in_minutes = 140 :=
by
  -- Placeholder for the actual proof
  sorry

end total_time_from_first_station_to_workplace_l7_7763


namespace Jolene_charge_per_car_l7_7117

theorem Jolene_charge_per_car (babysitting_families cars_washed : ℕ) (charge_per_family total_raised babysitting_earnings car_charge : ℕ) :
  babysitting_families = 4 →
  charge_per_family = 30 →
  cars_washed = 5 →
  total_raised = 180 →
  babysitting_earnings = babysitting_families * charge_per_family →
  car_charge = (total_raised - babysitting_earnings) / cars_washed →
  car_charge = 12 :=
by
  intros
  sorry

end Jolene_charge_per_car_l7_7117


namespace consecutive_squares_not_arithmetic_sequence_l7_7253

theorem consecutive_squares_not_arithmetic_sequence (x y z w : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
  (h_order: x < y ∧ y < z ∧ z < w) :
  ¬ (∃ d : ℕ, y^2 = x^2 + d ∧ z^2 = y^2 + d ∧ w^2 = z^2 + d) :=
sorry

end consecutive_squares_not_arithmetic_sequence_l7_7253


namespace sum_of_positive_integers_n_l7_7978

theorem sum_of_positive_integers_n
  (n : ℕ) (h1: n > 0)
  (h2 : Nat.lcm n 100 = Nat.gcd n 100 + 300) :
  n = 350 :=
sorry

end sum_of_positive_integers_n_l7_7978


namespace committee_probability_l7_7498

theorem committee_probability :
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  specific_committees / total_committees = 64 / 211 := 
by
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  have h_total_committees : total_committees = 593775 := by sorry
  have h_boys_choose : boys_choose = 816 := by sorry
  have h_girls_choose : girls_choose = 220 := by sorry
  have h_specific_committees : specific_committees = 179520 := by sorry
  have h_probability : specific_committees / total_committees = 64 / 211 := by sorry
  exact h_probability

end committee_probability_l7_7498


namespace solve_quadratic_l7_7004

theorem solve_quadratic : ∃ x : ℚ, 3 * x^2 + 11 * x - 20 = 0 ∧ x > 0 ∧ x = 4 / 3 :=
by
  sorry

end solve_quadratic_l7_7004


namespace probability_each_mailbox_has_at_least_one_letter_l7_7650

noncomputable def probability_mailbox (total_letters : ℕ) (mailboxes : ℕ) : ℚ := 
  let total_ways := mailboxes ^ total_letters
  let favorable_ways := Nat.choose total_letters (mailboxes - 1) * (mailboxes - 1).factorial
  favorable_ways / total_ways

theorem probability_each_mailbox_has_at_least_one_letter :
  probability_mailbox 3 2 = 3 / 4 := by
  sorry

end probability_each_mailbox_has_at_least_one_letter_l7_7650


namespace expected_lifetime_flashlight_l7_7319

section
variables {Ω : Type} [ProbabilitySpace Ω]
variables (ξ η : Ω → ℝ)
variables (h_ξ_expect : E[ξ] = 2)

-- Define the minimum of ξ and η
def min_ξ_η (ω : Ω) : ℝ := min (ξ ω) (η ω)

theorem expected_lifetime_flashlight : E[min_ξ_η ξ η] ≤ 2 :=
by
  sorry
end

end expected_lifetime_flashlight_l7_7319


namespace common_denominator_step1_error_in_step3_simplified_expression_l7_7177

theorem common_denominator_step1 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2):
  (3 * x / (x - 2) - x / (x + 2)) = (3 * x * (x + 2)) / ((x - 2) * (x + 2)) - (x * (x - 2)) / ((x - 2) * (x + 2)) :=
sorry

theorem error_in_step3 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2) :
  (3 * x^2 + 6 * x - (x^2 - 2 * x)) / ((x - 2) * (x + 2)) ≠ (3 * x^2 + 6 * x * (x^2 - 2 * x)) / ((x - 2) * (x + 2)) :=
sorry

theorem simplified_expression (x : ℝ) (h1: x ≠ 0) (h2: x ≠ 2) (h3: x ≠ -2) :
  ((3 * x / (x - 2) - x / (x + 2)) * (x^2 - 4) / x) = 2 * x + 8 :=
sorry

end common_denominator_step1_error_in_step3_simplified_expression_l7_7177


namespace best_fitting_model_l7_7307

theorem best_fitting_model :
  ∀ (R1 R2 R3 R4 : ℝ), R1 = 0.976 → R2 = 0.776 → R3 = 0.076 → R4 = 0.351 →
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  sorry

end best_fitting_model_l7_7307


namespace pencils_per_box_l7_7423

theorem pencils_per_box (boxes : ℕ) (total_pencils : ℕ) (h1 : boxes = 3) (h2 : total_pencils = 27) : (total_pencils / boxes) = 9 := 
by
  sorry

end pencils_per_box_l7_7423


namespace bus_stoppage_time_l7_7426

theorem bus_stoppage_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (reduction_in_speed : speed_excluding_stoppages - speed_including_stoppages = 8) :
  ∃ t : ℝ, t = 9.6 := 
sorry

end bus_stoppage_time_l7_7426


namespace expected_lifetime_flashlight_l7_7322

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l7_7322


namespace candy_problem_l7_7046

theorem candy_problem (
  a : ℤ
) : (a % 10 = 6) →
    (a % 15 = 11) →
    (200 ≤ a ∧ a ≤ 250) →
    (a = 206 ∨ a = 236) :=
sorry

end candy_problem_l7_7046


namespace valentine_problem_l7_7134

def initial_valentines : ℕ := 30
def given_valentines : ℕ := 8
def remaining_valentines : ℕ := 22

theorem valentine_problem : initial_valentines - given_valentines = remaining_valentines := by
  sorry

end valentine_problem_l7_7134


namespace ensure_nonempty_intersection_l7_7073

def M (x : ℝ) : Prop := x ≤ 1
def N (x : ℝ) (p : ℝ) : Prop := x > p

theorem ensure_nonempty_intersection (p : ℝ) : (∃ x : ℝ, M x ∧ N x p) ↔ p < 1 :=
by
  sorry

end ensure_nonempty_intersection_l7_7073


namespace rods_in_one_mile_l7_7882

-- Define the given conditions
def mile_to_chains : ℕ := 10
def chain_to_rods : ℕ := 4

-- Prove the number of rods in one mile
theorem rods_in_one_mile : (1 * mile_to_chains * chain_to_rods) = 40 := by
  sorry

end rods_in_one_mile_l7_7882


namespace star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l7_7455

def star (x y : ℤ) := (x + 2) * (y + 2) - 2

-- Statement A: commutativity
theorem star_comm : ∀ x y : ℤ, star x y = star y x := 
by sorry

-- Statement B: distributivity over addition
theorem star_distrib_over_add : ¬(∀ x y z : ℤ, star x (y + z) = star x y + star x z) :=
by sorry

-- Statement C: special case
theorem star_special_case : ¬(∀ x : ℤ, star (x - 2) (x + 2) = star x x - 2) :=
by sorry

-- Statement D: identity element
theorem star_no_identity : ¬(∃ e : ℤ, ∀ x : ℤ, star x e = x ∧ star e x = x) :=
by sorry

-- Statement E: associativity
theorem star_not_assoc : ¬(∀ x y z : ℤ, star (star x y) z = star x (star y z)) :=
by sorry

end star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l7_7455


namespace train_speed_in_kph_l7_7817

-- Define the given conditions
def length_of_train : ℝ := 200 -- meters
def time_crossing_pole : ℝ := 16 -- seconds

-- Define conversion factor
def mps_to_kph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Statement of the theorem
theorem train_speed_in_kph : mps_to_kph (length_of_train / time_crossing_pole) = 45 := 
sorry

end train_speed_in_kph_l7_7817


namespace problem_D_l7_7091

theorem problem_D (a b c : ℝ) (h : |a^2 + b + c| + |a + b^2 - c| ≤ 1) : a^2 + b^2 + c^2 < 100 := 
sorry

end problem_D_l7_7091


namespace grant_total_earnings_l7_7894

def earnings_first_month : ℕ := 350
def earnings_second_month : ℕ := 2 * earnings_first_month + 50
def earnings_third_month : ℕ := 4 * (earnings_first_month + earnings_second_month)
def total_earnings : ℕ := earnings_first_month + earnings_second_month + earnings_third_month

theorem grant_total_earnings : total_earnings = 5500 := by
  sorry

end grant_total_earnings_l7_7894


namespace socks_difference_l7_7921

-- Definitions of the conditions
def week1 : ℕ := 12
def week2 (S : ℕ) : ℕ := S
def week3 (S : ℕ) : ℕ := (12 + S) / 2
def week4 (S : ℕ) : ℕ := (12 + S) / 2 - 3
def total (S : ℕ) : ℕ := week1 + week2 S + week3 S + week4 S

-- Statement of the theorem
theorem socks_difference (S : ℕ) (h : total S = 57) : S - week1 = 1 :=
by 
  -- Proof is not required
  sorry

end socks_difference_l7_7921


namespace slope_of_given_line_l7_7458

def slope_of_line (l : String) : Real :=
  -- Assuming that we have a function to parse the line equation
  -- and extract its slope. Normally, this would be a complex parsing function.
  1 -- Placeholder, as the slope calculation logic is trivial here.

theorem slope_of_given_line : slope_of_line "x - y - 1 = 0" = 1 := by
  sorry

end slope_of_given_line_l7_7458


namespace abc_divisibility_l7_7986

theorem abc_divisibility (a b c : Nat) (h1 : a^3 ∣ b) (h2 : b^3 ∣ c) (h3 : c^3 ∣ a) :
  ∃ k : Nat, (a + b + c)^13 = k * a * b * c :=
by
  sorry

end abc_divisibility_l7_7986


namespace geometric_sequence_product_l7_7306

variable {α : Type*} [LinearOrderedField α]

theorem geometric_sequence_product :
  ∀ (a r : α), (a^3 * r^6 = 3) → (a^3 * r^15 = 24) → (a^3 * r^24 = 192) :=
by
  intros a r h1 h2
  sorry

end geometric_sequence_product_l7_7306


namespace max_discount_rate_l7_7221

-- Define the cost price and selling price.
def cp : ℝ := 4
def sp : ℝ := 5

-- Define the minimum profit margin.
def min_profit_margin : ℝ := 0.4

-- Define the discount rate d.
def discount_rate (d : ℝ) : ℝ := sp * (1 - d / 100) - cp

-- The theorem to prove the maximum discount rate.
theorem max_discount_rate (d : ℝ) (H : discount_rate d ≥ min_profit_margin) : d ≤ 12 :=
sorry

end max_discount_rate_l7_7221


namespace partial_fraction_sum_equals_251_l7_7311

theorem partial_fraction_sum_equals_251 (p q r A B C : ℝ) :
  (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ 
  (A ≠ 0) ∧ (B ≠ 0) ∧ (C ≠ 0) ∧
  (∀ s : ℝ, (s ≠ p) ∧ (s ≠ q) ∧ (s ≠ r) →
  1 / (s^3 - 24*s^2 + 151*s - 650) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (p + q + r = 24) →
  (p * q + p * r + q * r = 151) →
  (p * q * r = 650) →
  (1 / A + 1 / B + 1 / C = 251) :=
by
  sorry

end partial_fraction_sum_equals_251_l7_7311


namespace range_of_a_l7_7255

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := {x | x - a > 0}
def setB : Set ℝ := {x | x ≤ 0}

-- The main theorem asserting the condition
theorem range_of_a {a : ℝ} (h : setA a ∩ setB = ∅) : a ≥ 0 := by
  sorry

end range_of_a_l7_7255


namespace angle_BCM_in_pentagon_l7_7467

-- Definitions of the conditions
structure Pentagon (A B C D E : Type) :=
  (is_regular : ∀ (x y : Type), ∃ (angle : ℝ), angle = 108)

structure EquilateralTriangle (A B M : Type) :=
  (is_equilateral : ∀ (x y : Type), ∃ (angle : ℝ), angle = 60)

-- Problem statement
theorem angle_BCM_in_pentagon (A B C D E M : Type) (P : Pentagon A B C D E) (T : EquilateralTriangle A B M) :
  ∃ (angle : ℝ), angle = 66 :=
by
  sorry

end angle_BCM_in_pentagon_l7_7467


namespace how_many_halves_to_sum_one_and_one_half_l7_7731

theorem how_many_halves_to_sum_one_and_one_half : 
  (3 / 2) / (1 / 2) = 3 := 
by 
  sorry

end how_many_halves_to_sum_one_and_one_half_l7_7731


namespace pat_stickers_l7_7331

theorem pat_stickers (stickers_given_away stickers_left : ℝ) 
(h_given_away : stickers_given_away = 22.0)
(h_left : stickers_left = 17.0) : 
(stickers_given_away + stickers_left = 39) :=
by
  sorry

end pat_stickers_l7_7331


namespace select_monkey_l7_7338

theorem select_monkey (consumption : ℕ → ℕ) (n bananas minutes : ℕ)
  (h1 : consumption 1 = 1) (h2 : consumption 2 = 2) (h3 : consumption 3 = 3)
  (h4 : consumption 4 = 4) (h5 : consumption 5 = 5) (h6 : consumption 6 = 6)
  (h_total_minutes : minutes = 18) (h_total_bananas : bananas = 18) :
  consumption 1 * minutes = bananas :=
by
  sorry

end select_monkey_l7_7338


namespace last_two_digits_condition_l7_7504

-- Define the function to get last two digits of a number
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

-- Given numbers
def n1 := 122
def n2 := 123
def n3 := 125
def n4 := 129

-- The missing number
variable (x : ℕ)

theorem last_two_digits_condition : 
  last_two_digits (last_two_digits n1 * last_two_digits n2 * last_two_digits n3 * last_two_digits n4 * last_two_digits x) = 50 ↔ last_two_digits x = 1 :=
by 
  sorry

end last_two_digits_condition_l7_7504


namespace instantaneous_velocity_at_2_l7_7814

def s (t : ℝ) : ℝ := 3 * t^2 + t

theorem instantaneous_velocity_at_2 : (deriv s 2) = 13 :=
by
  sorry

end instantaneous_velocity_at_2_l7_7814


namespace no_grammatical_errors_in_B_l7_7671

-- Definitions for each option’s description (conditions)
def sentence_A := "The \"Criminal Law Amendment (IX)\", which was officially implemented on November 1, 2015, criminalizes exam cheating for the first time, showing the government's strong determination to combat exam cheating, and may become the \"magic weapon\" to govern the chaos of exams."
def sentence_B := "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region."
def sentence_C := "Since the implementation of the comprehensive two-child policy, many Chinese families have chosen not to have a second child. It is said that it's not because they don't want to, but because they can't afford it, as the cost of raising a child in China is too high."
def sentence_D := "Although it ended up being a futile effort, having fought for a dream, cried, and laughed, we are without regrets. For us, such experiences are treasures in themselves."

-- The statement that option B has no grammatical errors
theorem no_grammatical_errors_in_B : sentence_B = "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region." :=
by
  sorry

end no_grammatical_errors_in_B_l7_7671


namespace max_odd_integers_l7_7051

theorem max_odd_integers (chosen : Fin 5 → ℕ) (hpos : ∀ i, chosen i > 0) (heven : ∃ i, chosen i % 2 = 0) : 
  ∃ odd_count, odd_count = 4 ∧ (∀ i, i < 4 → chosen i % 2 = 1) := 
by 
  sorry

end max_odd_integers_l7_7051


namespace first_three_flips_HHT_l7_7383

-- Definitions based on the conditions and questions
def fair_coin : ProbabilityMassFunction Bool := 
  ProbabilityMassFunction.ofFintype uniform

theorem first_three_flips_HHT :
  let event_space := [true, true, false] in
  ProbabilityMassFunction.experiment_space fair_coin 3 = event_space →
  ProbabilityMassFunction.probability_of_event_space fair_coin event_space = 1 / 8 := 
sorry

end first_three_flips_HHT_l7_7383


namespace size_of_smaller_package_l7_7242

theorem size_of_smaller_package
  (total_coffee : ℕ)
  (n_ten_ounce_packages : ℕ)
  (extra_five_ounce_packages : ℕ)
  (size_smaller_package : ℕ)
  (h1 : total_coffee = 115)
  (h2 : size_smaller_package = 5)
  (h3 : n_ten_ounce_packages = 7)
  (h4 : extra_five_ounce_packages = 2)
  (h5 : total_coffee = n_ten_ounce_packages * 10 + (n_ten_ounce_packages + extra_five_ounce_packages) * size_smaller_package) :
  size_smaller_package = 5 :=
by 
  sorry

end size_of_smaller_package_l7_7242


namespace milk_students_l7_7605

theorem milk_students (T : ℕ) (h1 : (1 / 4) * T = 80) : (3 / 4) * T = 240 := by
  sorry

end milk_students_l7_7605


namespace find_roots_square_sum_and_min_y_l7_7724

-- Definitions from the conditions
def sum_roots (m : ℝ) :=
  -(m + 1)

def product_roots (m : ℝ) :=
  2 * m - 2

def roots_square_sum (m x₁ x₂ : ℝ) :=
  x₁^2 + x₂^2

def y (m : ℝ) :=
  (m - 1)^2 + 4

-- Proof statement
theorem find_roots_square_sum_and_min_y (m x₁ x₂ : ℝ) (h_sum : x₁ + x₂ = sum_roots m)
  (h_prod : x₁ * x₂ = product_roots m) :
  roots_square_sum m x₁ x₂ = (m - 1)^2 + 4 ∧ y m ≥ 4 :=
by
  sorry

end find_roots_square_sum_and_min_y_l7_7724


namespace intersection_of_A_and_B_l7_7806

def A : Set ℝ := { x | x^2 - x - 2 ≥ 0 }
def B : Set ℝ := { x | -2 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -2 ≤ x ∧ x ≤ -1 } := by
-- The proof would go here
sorry

end intersection_of_A_and_B_l7_7806


namespace find_rate_squares_sum_l7_7841

theorem find_rate_squares_sum {b j s : ℤ} 
(H1 : 3 * b + 2 * j + 2 * s = 112)
(H2 : 2 * b + 3 * j + 4 * s = 129) : b^2 + j^2 + s^2 = 1218 :=
by sorry

end find_rate_squares_sum_l7_7841


namespace ratio_red_to_black_l7_7111

theorem ratio_red_to_black (a b x : ℕ) (h1 : x + b = 3 * a) (h2 : x = 2 * b - 3 * a) :
  a / b = 1 / 2 := by
  sorry

end ratio_red_to_black_l7_7111


namespace min_colors_for_grid_coloring_l7_7474

theorem min_colors_for_grid_coloring : ∃c : ℕ, c = 4 ∧ (∀ (color : ℕ × ℕ → ℕ), 
  (∀ i j : ℕ, i < 5 ∧ j < 5 → 
     ((i < 4 → color (i, j) ≠ color (i+1, j+1)) ∧ 
      (j < 4 → color (i, j) ≠ color (i+1, j-1))) ∧ 
     ((i > 0 → color (i, j) ≠ color (i-1, j-1)) ∧ 
      (j > 0 → color (i, j) ≠ color (i-1, j+1)))) → 
  c = 4) :=
sorry

end min_colors_for_grid_coloring_l7_7474


namespace descending_order_numbers_count_l7_7850

theorem descending_order_numbers_count : 
  ∃ (n : ℕ), (n = 1013) ∧ 
  ∀ (x : ℕ), (∃ (xs : list ℕ), 
                (∀ i, i < xs.length - 1 → xs.nth_le i sorry > xs.nth_le (i+1) sorry) ∧ 
                nat_digits_desc xs ∧
                1 < xs.length) → 
             x ∈ nat_digits xs →
             ∃ (refs : list ℕ), n = refs.length ∧ 
             ∀ ref, ref ∈ refs → ref < x :=
sorry

end descending_order_numbers_count_l7_7850


namespace three_gt_sqrt_seven_l7_7680

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end three_gt_sqrt_seven_l7_7680


namespace find_z_l7_7875

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l7_7875


namespace inheritance_amount_l7_7610

theorem inheritance_amount (x : ℝ) 
  (h1 : x * 0.25 + (x - x * 0.25) * 0.12 = 13600) : x = 40000 :=
by
  -- This is where the proof would go
  sorry

end inheritance_amount_l7_7610


namespace average_speed_return_trip_l7_7387

def speed1 : ℝ := 12 -- Speed for the first part of the trip in miles per hour
def distance1 : ℝ := 18 -- Distance for the first part of the trip in miles
def speed2 : ℝ := 10 -- Speed for the second part of the trip in miles per hour
def distance2 : ℝ := 18 -- Distance for the second part of the trip in miles
def total_round_trip_time : ℝ := 7.3 -- Total time for the round trip in hours

theorem average_speed_return_trip :
  let time1 := distance1 / speed1 -- Time taken for the first part of the trip
  let time2 := distance2 / speed2 -- Time taken for the second part of the trip
  let total_time_to_destination := time1 + time2 -- Total time for the trip to the destination
  let time_return_trip := total_round_trip_time - total_time_to_destination -- Time for the return trip
  let return_trip_distance := distance1 + distance2 -- Distance for the return trip (same as to the destination)
  let avg_speed_return_trip := return_trip_distance / time_return_trip -- Average speed for the return trip
  avg_speed_return_trip = 9 := 
by
  sorry

end average_speed_return_trip_l7_7387


namespace nhai_highway_construction_l7_7765

/-- Problem definition -/
def total_man_hours (men1 men2 days1 days2 hours1 hours2 : Nat) : Nat := 
  (men1 * days1 * hours1) + (men2 * days2 * hours2)

theorem nhai_highway_construction :
  let men := 100
  let days1 := 25
  let days2 := 25
  let hours1 := 8
  let hours2 := 10
  let additional_men := 60
  let total_days := 50
  total_man_hours men (men + additional_men) total_days total_days hours1 hours2 = 
  2 * total_man_hours men men days1 days2 hours1 hours1 :=
  sorry

end nhai_highway_construction_l7_7765


namespace dawson_failed_by_36_l7_7245

-- Define the constants and conditions
def max_marks : ℕ := 220
def passing_percentage : ℝ := 0.3
def marks_obtained : ℕ := 30

-- Calculate the minimum passing marks
noncomputable def min_passing_marks : ℝ :=
  passing_percentage * max_marks

-- Calculate the marks Dawson failed by
noncomputable def marks_failed_by : ℝ :=
  min_passing_marks - marks_obtained

-- State the theorem
theorem dawson_failed_by_36 :
  marks_failed_by = 36 := by
  -- Proof is omitted
  sorry

end dawson_failed_by_36_l7_7245


namespace system_solution_l7_7030

theorem system_solution (x y : ℝ) (h1 : x + y = 1) (h2 : x - y = 3) : x = 2 ∧ y = -1 :=
by
  sorry

end system_solution_l7_7030


namespace total_texts_sent_l7_7627

def texts_sent_monday_allison : ℕ := 5
def texts_sent_monday_brittney : ℕ := 5
def texts_sent_tuesday_allison : ℕ := 15
def texts_sent_tuesday_brittney : ℕ := 15

theorem total_texts_sent : (texts_sent_monday_allison + texts_sent_monday_brittney) + 
                           (texts_sent_tuesday_allison + texts_sent_tuesday_brittney) = 40 :=
by
  sorry

end total_texts_sent_l7_7627


namespace sum_of_arith_seq_l7_7130

noncomputable def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arith_seq (a : ℕ → ℝ) (h_a : is_arith_seq a)
  (h_f_sum : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 21 :=
sorry

end sum_of_arith_seq_l7_7130


namespace product_of_sums_of_four_squares_is_sum_of_four_squares_l7_7775

theorem product_of_sums_of_four_squares_is_sum_of_four_squares (x1 x2 x3 x4 y1 y2 y3 y4 : ℤ) :
  let a := x1^2 + x2^2 + x3^2 + x4^2
  let b := y1^2 + y2^2 + y3^2 + y4^2
  let z1 := x1 * y1 + x2 * y2 + x3 * y3 + x4 * y4
  let z2 := x1 * y2 - x2 * y1 + x3 * y4 - x4 * y3
  let z3 := x1 * y3 - x3 * y1 + x4 * y2 - x2 * y4
  let z4 := x1 * y4 - x4 * y1 + x2 * y3 - x3 * y2
  a * b = z1^2 + z2^2 + z3^2 + z4^2 :=
by
  sorry

end product_of_sums_of_four_squares_is_sum_of_four_squares_l7_7775


namespace repeating_decimal_division_l7_7058

-- Definitions based on given conditions
def repeating_54_as_frac : ℚ := 54 / 99
def repeating_18_as_frac : ℚ := 18 / 99

-- Theorem stating the required proof
theorem repeating_decimal_division :
  (repeating_54_as_frac / repeating_18_as_frac = 3) :=
  sorry

end repeating_decimal_division_l7_7058


namespace maximum_distance_point_to_line_l7_7333

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m - 1) * x + m * y + 2 = 0

-- Statement of the problem to prove
theorem maximum_distance_point_to_line :
  ∀ (x y m : ℝ), circle_C x y → ∃ P : ℝ, line_l m x y → P = 6 :=
by 
  sorry

end maximum_distance_point_to_line_l7_7333


namespace johns_payment_ratio_is_one_half_l7_7609

-- Define the initial conditions
def num_members := 4
def join_fee_per_person := 4000
def monthly_cost_per_person := 1000
def johns_payment_per_year := 32000

-- Calculate total cost for joining
def total_join_fee := num_members * join_fee_per_person

-- Calculate total monthly cost for a year
def total_monthly_cost := num_members * monthly_cost_per_person * 12

-- Calculate total cost for the first year
def total_cost_for_year := total_join_fee + total_monthly_cost

-- The ratio of John's payment to the total cost
def johns_ratio := johns_payment_per_year / total_cost_for_year

-- The statement to be proved
theorem johns_payment_ratio_is_one_half : johns_ratio = (1 / 2) := by sorry

end johns_payment_ratio_is_one_half_l7_7609


namespace second_difference_is_quadratic_l7_7470

theorem second_difference_is_quadratic (f : ℕ → ℝ) 
  (h : ∀ n : ℕ, (f (n + 2) - 2 * f (n + 1) + f n) = 2) :
  ∃ (a b : ℝ), ∀ (n : ℕ), f n = n^2 + a * n + b :=
by
  sorry

end second_difference_is_quadratic_l7_7470


namespace sum_of_solutions_l7_7493

theorem sum_of_solutions : 
  (∃ x, 3^(x^2 + 6*x + 9) = 27^(x + 3)) → (∀ x₁ x₂, (3^(x₁^2 + 6*x₁ + 9) = 27^(x₁ + 3) ∧ 3^(x₂^2 + 6*x₂ + 9) = 27^(x₂ + 3)) → x₁ + x₂ = -3) :=
sorry

end sum_of_solutions_l7_7493


namespace alicia_tax_correct_l7_7409

theorem alicia_tax_correct :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let basic_tax_rate := 0.01
  let additional_tax_rate := 0.0075
  let basic_tax := basic_tax_rate * hourly_wage_cents
  let excess_amount_cents := (hourly_wage_dollars - 20) * 100
  let additional_tax := additional_tax_rate * excess_amount_cents
  basic_tax + additional_tax = 28.75 := 
by
  sorry

end alicia_tax_correct_l7_7409


namespace ernie_circles_l7_7549

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes ali_circles : ℕ)
  (h1: boxes_per_circle_ali = 8)
  (h2: boxes_per_circle_ernie = 10)
  (h3: total_boxes = 80)
  (h4: ali_circles = 5) : 
  (total_boxes - ali_circles * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l7_7549


namespace trigonometric_equation_solution_l7_7755

theorem trigonometric_equation_solution (n : ℕ) (h_pos : 0 < n) (x : ℝ) (hx1 : ∀ k : ℤ, x ≠ k * π / 2) :
  (1 / (Real.sin x)^(2 * n) + 1 / (Real.cos x)^(2 * n) = 2^(n + 1)) ↔ ∃ k : ℤ, x = (2 * k + 1) * π / 4 :=
by sorry

end trigonometric_equation_solution_l7_7755


namespace fraction_red_surface_area_eq_3_over_4_l7_7395

-- Define the larger cube made of smaller cubes.
structure Cube :=
  (side_length : ℕ)
  (num_cubes : ℕ)

-- Define the color distribution.
structure ColorDistribution :=
  (red_cubes : ℕ)
  (blue_cubes : ℕ)

-- Conditions
def larger_cube : Cube := ⟨4, 64⟩
def color_dist : ColorDistribution := ⟨32, 32⟩
def blue_per_face : ℕ := 4

-- Theorem statement
theorem fraction_red_surface_area_eq_3_over_4 :
  let total_surface_area := 6 * (larger_cube.side_length ^ 2)
  let blue_faces := blue_per_face * 6
  let red_faces := total_surface_area - blue_faces in
  (red_faces : ℚ) / (total_surface_area : ℚ) = 3 / 4 := by
  sorry

end fraction_red_surface_area_eq_3_over_4_l7_7395


namespace integer_solutions_of_quadratic_eq_l7_7614

theorem integer_solutions_of_quadratic_eq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ x1 x2 : ℤ, x1 * x2 = q^4 ∧ x1 + x2 = -p ∧ x1 = -1 ∧ x2 = - (q^4) ∧ p = 17 ∧ q = 2 := 
sorry

end integer_solutions_of_quadratic_eq_l7_7614


namespace flower_count_l7_7357

theorem flower_count (roses carnations : ℕ) (h₁ : roses = 5) (h₂ : carnations = 5) : roses + carnations = 10 :=
by
  sorry

end flower_count_l7_7357


namespace moles_CH3COOH_equiv_l7_7571

theorem moles_CH3COOH_equiv (moles_NaOH moles_NaCH3COO : ℕ)
    (h1 : moles_NaOH = 1)
    (h2 : moles_NaCH3COO = 1) :
    moles_NaOH = moles_NaCH3COO :=
by
  sorry

end moles_CH3COOH_equiv_l7_7571


namespace inverse_proportion_quadrants_l7_7100

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0))) ↔ k > 0 := by
  sorry

end inverse_proportion_quadrants_l7_7100


namespace range_of_F_l7_7572

theorem range_of_F (A B C : ℝ) (h1 : 0 < A) (h2 : A ≤ B) (h3 : B ≤ C) (h4 : C < π / 2) :
  1 + (Real.sqrt 2) / 2 < (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) ∧
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 :=
  sorry

end range_of_F_l7_7572


namespace average_ducks_l7_7543

theorem average_ducks (a e k : ℕ) 
  (h1 : a = 2 * e) 
  (h2 : e = k - 45) 
  (h3 : a = 30) :
  (a + e + k) / 3 = 35 :=
by
  sorry

end average_ducks_l7_7543


namespace cost_price_proof_l7_7821

noncomputable def cost_price (C : ℝ) : Prop :=
  let SP := 0.76 * C in
  let ISP := 1.18 * C in
  ISP = SP + 450 ∧ C = 1071.43

theorem cost_price_proof : ∃ C : ℝ, cost_price C :=
by
  use 1071.43
  unfold cost_price
  split
  . exact eq.symm (by norm_num : 1.18 * 1071.43 = 0.76 * 1071.43 + 450)
  . exact eq.refl 1071.43

end cost_price_proof_l7_7821


namespace age_difference_is_eight_l7_7480

theorem age_difference_is_eight (A B k : ℕ)
  (h1 : A = B + k)
  (h2 : A - 1 = 3 * (B - 1))
  (h3 : A = 2 * B + 3) :
  k = 8 :=
by sorry

end age_difference_is_eight_l7_7480


namespace probability_of_winning_pair_l7_7913

-- Define the cards and their properties
inductive Color
| Red
| Green

inductive Label
| A
| B
| C

structure Card :=
  (color : Color)
  (label : Label)

-- Define the deck of cards
def deck : Finset Card :=
  {⟨Color.Red, Label.A⟩, ⟨Color.Red, Label.B⟩, ⟨Color.Red, Label.C⟩,
   ⟨Color.Green, Label.A⟩, ⟨Color.Green, Label.B⟩, ⟨Color.Green, Label.C⟩}

-- Define what it means to draw a winning pair
def is_winning_pair (c1 c2 : Card) : Prop :=
  (c1.color = c2.color) ∨ (c1.label = c2.label)

-- Problem statement: prove the probability of drawing a winning pair
theorem probability_of_winning_pair : 
  (deck.cardinal.choose 2) = 15 →
  ∀ (winning_count : Nat), 
    (∃ (p : Nat), p = 9 ∧ p = winning_count) →
    ∃ p : ℚ, p = (9 : ℚ) / 15 :=
begin
  sorry
end

end probability_of_winning_pair_l7_7913


namespace sqrt_of_expression_l7_7833

theorem sqrt_of_expression : Real.sqrt (5^2 * 7^6) = 1715 := 
by
  sorry

end sqrt_of_expression_l7_7833


namespace least_value_of_sum_l7_7982

theorem least_value_of_sum (x y z : ℤ) 
  (h_cond : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z ≥ 56 :=
sorry

end least_value_of_sum_l7_7982


namespace total_texts_sent_l7_7626

def texts_sent_monday_allison : ℕ := 5
def texts_sent_monday_brittney : ℕ := 5
def texts_sent_tuesday_allison : ℕ := 15
def texts_sent_tuesday_brittney : ℕ := 15

theorem total_texts_sent : (texts_sent_monday_allison + texts_sent_monday_brittney) + 
                           (texts_sent_tuesday_allison + texts_sent_tuesday_brittney) = 40 :=
by
  sorry

end total_texts_sent_l7_7626


namespace total_shaded_area_is_2pi_l7_7008

theorem total_shaded_area_is_2pi (sm_radius large_radius : ℝ) 
  (h_sm_radius : sm_radius = 1) 
  (h_large_radius : large_radius = 2) 
  (sm_circle_area large_circle_area total_shaded_area : ℝ) 
  (h_sm_circle_area : sm_circle_area = π * sm_radius^2) 
  (h_large_circle_area : large_circle_area = π * large_radius^2) 
  (h_total_shaded_area : total_shaded_area = large_circle_area - 2 * sm_circle_area) :
  total_shaded_area = 2 * π :=
by
  -- Proof goes here
  sorry

end total_shaded_area_is_2pi_l7_7008


namespace trajectory_of_center_l7_7717

-- Define a structure for Point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the given point A
def A : Point := { x := -2, y := 0 }

-- Define a property for the circle being tangent to a line
def tangent_to_line (center : Point) (line_x : ℝ) : Prop :=
  center.x + line_x = 0

-- The main theorem to be proved
theorem trajectory_of_center :
  ∀ (C : Point), tangent_to_line C 2 → (C.y)^2 = -8 * C.x :=
sorry

end trajectory_of_center_l7_7717


namespace probability_red_first_given_black_second_l7_7373

open ProbabilityTheory MeasureTheory

-- Definitions for Urn A and Urn B ball quantities
def urnA := (white : 4, red : 2)
def urnB := (red : 3, black : 3)

-- Event of drawing a red ball first and a black ball second
def eventRedFirst := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "red") ∨ (urn = 2 ∧ ball = "red")
def eventBlackSecond := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "black") ∨ (urn = 2 ∧ ball = "black")

-- Probability function definition
noncomputable def P := sorry -- Probability function placeholder

-- Conditional Probability
theorem probability_red_first_given_black_second :
  P(eventRedFirst | eventBlackSecond) = 2 / 5 := sorry

end probability_red_first_given_black_second_l7_7373


namespace sum_of_common_divisors_60_18_l7_7068

theorem sum_of_common_divisors_60_18 : 
  let a := 60 
  let b := 18 
  let common_divisors := {n | n ∣ a ∧ n ∣ b ∧ n > 0 } 
  (∑ n in common_divisors, n) = 12 :=
by
  let a := 60
  let b := 18
  let common_divisors := { n | n ∣ a ∧ n ∣ b ∧ n > 0 }
  have : (∑ n in common_divisors, n) = 12 := sorry
  exact this

end sum_of_common_divisors_60_18_l7_7068


namespace mushrooms_count_l7_7797

theorem mushrooms_count:
  ∃ (n : ℕ) (m : ℕ) (x : ℕ),
  n ≤ 70 ∧ 
  m = (13 * n) / 25 ∧ 
  (n - 3 ≠ 0) ∧ 
  2 * (m - x) = n - 3 ∧ 
  n = 25 :=
by
  exists 25
  exists 13
  exists 2
  simp
  sorry

end mushrooms_count_l7_7797


namespace total_albums_l7_7936

theorem total_albums (Adele Bridget Katrina Miriam : ℕ) 
  (h₁ : Adele = 30)
  (h₂ : Bridget = Adele - 15)
  (h₃ : Katrina = 6 * Bridget)
  (h₄ : Miriam = 5 * Katrina) : Adele + Bridget + Katrina + Miriam = 585 := 
by
  sorry

end total_albums_l7_7936


namespace nicky_catchup_time_l7_7136

-- Definitions related to the problem
def head_start : ℕ := 12
def speed_cristina : ℕ := 5
def speed_nicky : ℕ := 3
def time_to_catchup : ℕ := 36
def nicky_runtime_before_catchup : ℕ := head_start + time_to_catchup

-- Theorem to prove the correct runtime for Nicky before Cristina catches up
theorem nicky_catchup_time : nicky_runtime_before_catchup = 48 := by
  sorry

end nicky_catchup_time_l7_7136


namespace bicycle_stock_decrease_l7_7138

-- Define the conditions and the problem
theorem bicycle_stock_decrease (m : ℕ) (jan_to_oct_decrease june_to_oct_decrease monthly_decrease : ℕ) 
  (h1: monthly_decrease = 4)
  (h2: jan_to_oct_decrease = 36)
  (h3: june_to_oct_decrease = 4 * monthly_decrease):
  m * monthly_decrease = jan_to_oct_decrease - june_to_oct_decrease → m = 5 := 
by
  sorry

end bicycle_stock_decrease_l7_7138


namespace sum_of_roots_of_quadratic_l7_7979

theorem sum_of_roots_of_quadratic :
  ∑ x in {x : ℝ | x^2 - 16 * x + 21 = 0}.to_finset = 16 :=
by
  sorry

end sum_of_roots_of_quadratic_l7_7979


namespace find_number_l7_7808

theorem find_number (x : ℝ) (h : 0.60 * x - 40 = 50) : x = 150 := 
by
  sorry

end find_number_l7_7808


namespace solve_for_z_l7_7860

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l7_7860


namespace B_catches_up_with_A_l7_7670

theorem B_catches_up_with_A :
  let d := 140
  let vA := 10
  let vB := 20
  let tA := d / vA
  let tB := d / vB
  tA - tB = 7 := 
by
  -- Definitions
  let d := 140
  let vA := 10
  let vB := 20
  let tA := d / vA
  let tB := d / vB
  -- Goal
  show tA - tB = 7
  sorry

end B_catches_up_with_A_l7_7670


namespace sum_of_squares_of_ages_eq_35_l7_7021

theorem sum_of_squares_of_ages_eq_35
  (d t h : ℕ)
  (h1 : 3 * d + 4 * t = 2 * h + 2)
  (h2 : 2 * d^2 + t^2 = 6 * h)
  (relatively_prime : Nat.gcd (Nat.gcd d t) h = 1) :
  d^2 + t^2 + h^2 = 35 := 
sorry

end sum_of_squares_of_ages_eq_35_l7_7021


namespace randy_blocks_left_l7_7152

-- Formalize the conditions
def initial_blocks : ℕ := 78
def blocks_used_first_tower : ℕ := 19
def blocks_used_second_tower : ℕ := 25

-- Formalize the result for verification
def blocks_left : ℕ := initial_blocks - blocks_used_first_tower - blocks_used_second_tower

-- State the theorem to be proven
theorem randy_blocks_left :
  blocks_left = 34 :=
by
  -- Not providing the proof as per instructions
  sorry

end randy_blocks_left_l7_7152


namespace inequality_proof_l7_7491

theorem inequality_proof 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 > 0) 
  (h2 : a2 > 0) 
  (h3 : a3 > 0)
  (h4 : a4 > 0):
  (a1 + a3) / (a1 + a2) + 
  (a2 + a4) / (a2 + a3) + 
  (a3 + a1) / (a3 + a4) + 
  (a4 + a2) / (a4 + a1) ≥ 4 :=
by
  sorry

end inequality_proof_l7_7491


namespace correct_operation_l7_7803

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ (2 * a^3 / a = 2 * a^2) ∧ ¬((a * b)^2 = a * b^2) ∧ ¬((-a^3)^3 = -a^6) :=
by
  sorry

end correct_operation_l7_7803


namespace arthur_money_left_l7_7823

theorem arthur_money_left {initial_amount spent_fraction : ℝ} (h_initial : initial_amount = 200) (h_fraction : spent_fraction = 4 / 5) : 
  (initial_amount - spent_fraction * initial_amount = 40) :=
by
  sorry

end arthur_money_left_l7_7823


namespace square_of_leg_l7_7914

theorem square_of_leg (a c b : ℝ) (h1 : c = 2 * a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = 3 * a^2 + 4 * a + 1 :=
by
  sorry

end square_of_leg_l7_7914


namespace ln_abs_a_even_iff_a_eq_zero_l7_7190

theorem ln_abs_a_even_iff_a_eq_zero (a : ℝ) :
  (∀ x : ℝ, Real.log (abs (x - a)) = Real.log (abs (-x - a))) ↔ (a = 0) :=
by
  sorry

end ln_abs_a_even_iff_a_eq_zero_l7_7190


namespace find_sum_of_numbers_l7_7170

theorem find_sum_of_numbers (x A B C : ℝ) (h1 : x > 0) (h2 : A = x) (h3 : B = 2 * x) (h4 : C = 3 * x) (h5 : A^2 + B^2 + C^2 = 2016) : A + B + C = 72 :=
sorry

end find_sum_of_numbers_l7_7170


namespace odot_computation_l7_7965

noncomputable def op (a b : ℚ) : ℚ := 
  (a + b) / (1 + a * b)

theorem odot_computation : op 2 (op 3 (op 4 5)) = 7 / 8 := 
  by 
  sorry

end odot_computation_l7_7965


namespace sqrt_equiv_1715_l7_7836

noncomputable def sqrt_five_squared_times_seven_sixth : ℕ := 
  Nat.sqrt (5^2 * 7^6)

theorem sqrt_equiv_1715 : sqrt_five_squared_times_seven_sixth = 1715 := by
  sorry

end sqrt_equiv_1715_l7_7836


namespace brad_has_9_green_balloons_l7_7673

theorem brad_has_9_green_balloons (total_balloons red_balloons : ℕ) (h_total : total_balloons = 17) (h_red : red_balloons = 8) : total_balloons - red_balloons = 9 :=
by {
  sorry
}

end brad_has_9_green_balloons_l7_7673


namespace equal_opposite_roots_eq_m_l7_7297

theorem equal_opposite_roots_eq_m (a b c : ℝ) (m : ℝ) (h : (∃ x : ℝ, (a * x - c ≠ 0) ∧ (((x^2 - b * x) / (a * x - c)) = ((m - 1) / (m + 1)))) ∧
(∀ x : ℝ, ((x^2 - b * x) = 0 → x = 0) ∧ (∃ t : ℝ, t > 0 ∧ ((x = t) ∨ (x = -t))))):
  m = (a - b) / (a + b) :=
by
  sorry

end equal_opposite_roots_eq_m_l7_7297


namespace leonardo_sleep_fraction_l7_7122

theorem leonardo_sleep_fraction (h : 60 ≠ 0) : (12 / 60 : ℚ) = (1 / 5 : ℚ) :=
by
  sorry

end leonardo_sleep_fraction_l7_7122


namespace monkey_climbing_distance_l7_7042

theorem monkey_climbing_distance
  (x : ℝ)
  (h1 : ∀ t : ℕ, t % 2 = 0 → t ≠ 0 → x - 3 > 0) -- condition (2,4)
  (h2 : ∀ t : ℕ, t % 2 = 1 → x > 0) -- condition (5)
  (h3 : 18 * (x - 3) + x = 60) -- condition (6)
  : x = 6 :=
sorry

end monkey_climbing_distance_l7_7042


namespace minimum_difference_l7_7910

def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem minimum_difference (x y z : ℤ) 
  (hx : even x) (hy : odd y) (hz : odd z)
  (hxy : x < y) (hyz : y < z) (hzx : z - x = 9) : y - x = 1 := 
sorry

end minimum_difference_l7_7910


namespace radius_of_roots_circle_l7_7550

theorem radius_of_roots_circle (z : ℂ) (hz : (z - 2)^6 = 64 * z^6) : ∃ r : ℝ, r = 2 / 3 :=
by
  sorry

end radius_of_roots_circle_l7_7550


namespace inconsistent_conditions_l7_7186

-- Definitions based on the given conditions
def B : Nat := 59
def C : Nat := 27
def D : Nat := 31
def A := B * C + D

theorem inconsistent_conditions (A_is_factor : ∃ k : Nat, 4701 = k * A) : false := by
  sorry

end inconsistent_conditions_l7_7186


namespace meals_per_day_l7_7181

-- Definitions based on given conditions
def number_of_people : Nat := 6
def total_plates_used : Nat := 144
def number_of_days : Nat := 4
def plates_per_meal : Nat := 2

-- Theorem to prove
theorem meals_per_day : (total_plates_used / number_of_days) / plates_per_meal / number_of_people = 3 :=
by
  sorry

end meals_per_day_l7_7181


namespace time_to_paint_one_room_l7_7813

variables (rooms_total rooms_painted : ℕ) (hours_to_paint_remaining : ℕ)

-- The conditions
def painter_conditions : Prop :=
  rooms_total = 10 ∧ rooms_painted = 8 ∧ hours_to_paint_remaining = 16

-- The goal is to find out the hours to paint one room
theorem time_to_paint_one_room (h : painter_conditions rooms_total rooms_painted hours_to_paint_remaining) : 
  let rooms_remaining := rooms_total - rooms_painted
  let hours_per_room := hours_to_paint_remaining / rooms_remaining
  hours_per_room = 8 :=
by sorry

end time_to_paint_one_room_l7_7813


namespace more_girls_than_boys_l7_7970

def ratio_boys_girls (B G : ℕ) : Prop := B = (3/5 : ℚ) * G

def total_students (B G : ℕ) : Prop := B + G = 16

theorem more_girls_than_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : G - B = 4 :=
by
  sorry

end more_girls_than_boys_l7_7970


namespace lucy_money_left_l7_7757

theorem lucy_money_left : 
  ∀ (initial_money : ℕ) 
    (one_third_loss : ℕ → ℕ) 
    (one_fourth_spend : ℕ → ℕ), 
    initial_money = 30 → 
    one_third_loss initial_money = initial_money / 3 → 
    one_fourth_spend (initial_money - one_third_loss initial_money) = (initial_money - one_third_loss initial_money) / 4 → 
  initial_money - one_third_loss initial_money - one_fourth_spend (initial_money - one_third_loss initial_money) = 15 :=
by
  intros initial_money one_third_loss one_fourth_spend
  intro h_initial_money
  intro h_one_third_loss
  intro h_one_fourth_spend
  sorry

end lucy_money_left_l7_7757


namespace fence_pole_count_l7_7038

-- Define the conditions
def path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the goal
def total_poles : ℕ := 286

-- The statement to prove
theorem fence_pole_count :
  let total_length_to_fence := (path_length - bridge_length)
  let poles_per_side := total_length_to_fence / pole_spacing
  let total_poles_needed := 2 * poles_per_side
  total_poles_needed = total_poles :=
by
  sorry

end fence_pole_count_l7_7038


namespace max_discount_rate_l7_7229

-- Define the conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1
def min_profit : ℝ := cost_price * min_profit_margin

-- Define the discount rate
def discount_rate (x : ℝ) : ℝ :=
  selling_price * (1 - x / 100)

-- Define the profit after discount
def profit_after_discount (x : ℝ) : ℝ :=
  discount_rate x - cost_price

-- The statement we want to prove
theorem max_discount_rate : ∃ x : ℝ, x = 8.8 ∧ profit_after_discount x ≥ min_profit := 
by
  sorry

end max_discount_rate_l7_7229


namespace prob_9_correct_matches_is_zero_l7_7199

noncomputable def probability_of_exactly_9_correct_matches : ℝ :=
  let n := 10 in
  -- Since choosing 9 correct implies the 10th is also correct, the probability is 0.
  0

theorem prob_9_correct_matches_is_zero : probability_of_exactly_9_correct_matches = 0 :=
by
  sorry

end prob_9_correct_matches_is_zero_l7_7199


namespace number_of_candies_picked_up_l7_7592

-- Definitions of the conditions
def num_sides_decagon := 10
def diagonals_from_one_vertex (n : Nat) : Nat := n - 3

-- The theorem stating the number of candies Hyeonsu picked up
theorem number_of_candies_picked_up : diagonals_from_one_vertex num_sides_decagon = 7 := by
  sorry

end number_of_candies_picked_up_l7_7592


namespace cos_negative_570_equals_negative_sqrt3_div_2_l7_7385

theorem cos_negative_570_equals_negative_sqrt3_div_2 : Real.cos (-570 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_negative_570_equals_negative_sqrt3_div_2_l7_7385


namespace smallest_n_for_multiples_of_7_l7_7158

theorem smallest_n_for_multiples_of_7 (x y : ℤ) (h1 : x ≡ 4 [ZMOD 7]) (h2 : y ≡ 5 [ZMOD 7]) :
  ∃ n : ℕ, 0 < n ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7]) ∧ ∀ m : ℕ, 0 < m ∧ (x^2 + x * y + y^2 + m ≡ 0 [ZMOD 7]) → n ≤ m :=
by
  sorry

end smallest_n_for_multiples_of_7_l7_7158


namespace chestnuts_distribution_l7_7406

theorem chestnuts_distribution:
  ∃ (chestnuts_Alya chestnuts_Valya chestnuts_Galya : ℕ),
    chestnuts_Alya + chestnuts_Valya + chestnuts_Galya = 70 ∧
    4 * chestnuts_Valya = 3 * chestnuts_Alya ∧
    6 * chestnuts_Galya = 7 * chestnuts_Alya ∧
    chestnuts_Alya = 24 ∧
    chestnuts_Valya = 18 ∧
    chestnuts_Galya = 28 :=
by {
  sorry
}

end chestnuts_distribution_l7_7406


namespace swimming_pool_length_correct_l7_7356

noncomputable def swimming_pool_length (V_removed: ℝ) (W: ℝ) (H: ℝ) (gal_to_cuft: ℝ): ℝ :=
  V_removed / (W * H / gal_to_cuft)

theorem swimming_pool_length_correct:
  swimming_pool_length 3750 25 0.5 7.48052 = 40.11 :=
by
  sorry

end swimming_pool_length_correct_l7_7356


namespace find_rate_percent_l7_7521

theorem find_rate_percent (SI P T : ℝ) (h : SI = (P * R * T) / 100) (H_SI : SI = 250) 
  (H_P : P = 1500) (H_T : T = 5) : R = 250 / 75 := by
  sorry

end find_rate_percent_l7_7521


namespace coefficient_of_a3b2_in_expansions_l7_7801

theorem coefficient_of_a3b2_in_expansions 
  (a b c : ℝ) :
  (1 : ℝ) * (a + b)^5 * (c + c⁻¹)^8 = 700 :=
by 
  sorry

end coefficient_of_a3b2_in_expansions_l7_7801


namespace range_of_a_l7_7099

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l7_7099


namespace prob_top_odd_correct_l7_7033

def total_dots : Nat := 78
def faces : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Probability calculation for odd dots after removal
def prob_odd_dot (n : Nat) : Rat :=
  if n % 2 = 1 then
    1 - (n : Rat) / total_dots
  else
    (n : Rat) / total_dots

-- Probability that the top face shows an odd number of dots
noncomputable def prob_top_odd : Rat :=
  (1 / (faces.length : Rat)) * (faces.map prob_odd_dot).sum

theorem prob_top_odd_correct :
  prob_top_odd = 523 / 936 :=
by
  sorry

end prob_top_odd_correct_l7_7033


namespace tom_tickets_l7_7056

theorem tom_tickets :
  (45 + 38 + 52) - (12 + 23) = 100 := by
sorry

end tom_tickets_l7_7056


namespace mean_proportional_c_l7_7298

theorem mean_proportional_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 27) (h3 : c^2 = a * b) : c = 9 := by
  sorry

end mean_proportional_c_l7_7298


namespace domain_log2_x_minus_1_l7_7161

theorem domain_log2_x_minus_1 (x : ℝ) : (1 < x) ↔ (∃ y : ℝ, y = Real.logb 2 (x - 1)) := by
  sorry

end domain_log2_x_minus_1_l7_7161


namespace total_albums_l7_7938

theorem total_albums (Adele Bridget Katrina Miriam : ℕ) 
  (h₁ : Adele = 30)
  (h₂ : Bridget = Adele - 15)
  (h₃ : Katrina = 6 * Bridget)
  (h₄ : Miriam = 5 * Katrina) : Adele + Bridget + Katrina + Miriam = 585 := 
by
  sorry

end total_albums_l7_7938


namespace polygon_sides_given_interior_angle_l7_7998

theorem polygon_sides_given_interior_angle
  (h : ∀ (n : ℕ), (n > 2) → ((n - 2) * 180 = n * 140)): n = 9 := by
  sorry

end polygon_sides_given_interior_angle_l7_7998


namespace probability_selecting_cooking_l7_7992

theorem probability_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let favorable_outcomes := 1
  let total_outcomes := courses.length
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 4 :=
by
  sorry

end probability_selecting_cooking_l7_7992


namespace stationery_difference_l7_7574

theorem stationery_difference :
  let georgia := 25
  let lorene := 3 * georgia
  lorene - georgia = 50 :=
by
  let georgia := 25
  let lorene := 3 * georgia
  show lorene - georgia = 50
  sorry

end stationery_difference_l7_7574


namespace line_passing_through_points_l7_7012

-- Definition of points
def point1 : ℝ × ℝ := (1, 0)
def point2 : ℝ × ℝ := (0, -2)

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Theorem statement
theorem line_passing_through_points : 
  line_eq point1.1 point1.2 ∧ line_eq point2.1 point2.2 :=
by
  sorry

end line_passing_through_points_l7_7012


namespace three_gt_sqrt_seven_l7_7682

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := sorry

end three_gt_sqrt_seven_l7_7682


namespace b_horses_pasture_l7_7528

theorem b_horses_pasture (H : ℕ) : (9 * H / (96 + 9 * H + 108)) * 870 = 360 → H = 6 :=
by
  -- Here we state the problem and skip the proof
  sorry

end b_horses_pasture_l7_7528


namespace arrangement_of_athletes_l7_7632

def num_arrangements (n : ℕ) (available_tracks_for_A : ℕ) (permutations_remaining : ℕ) : ℕ :=
  n * available_tracks_for_A * permutations_remaining

theorem arrangement_of_athletes :
  num_arrangements 2 3 24 = 144 :=
by
  sorry

end arrangement_of_athletes_l7_7632


namespace min_reciprocal_sum_l7_7087

theorem min_reciprocal_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : S 2019 = 4038) 
  (h_seq : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  ∃ m, m = 4 ∧ (∀ i, i = 9 → ∀ j, j = 2011 → 
  a i + a j = 4 ∧ m = min (1 / a i + 9 / a j) 4) :=
by sorry

end min_reciprocal_sum_l7_7087


namespace quotient_unchanged_l7_7098

-- Define the variables
variables (a b k : ℝ)

-- Condition: k ≠ 0
theorem quotient_unchanged (h : k ≠ 0) : (a * k) / (b * k) = a / b := by
  sorry

end quotient_unchanged_l7_7098


namespace min_value_perpendicular_vectors_l7_7591

theorem min_value_perpendicular_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (hperp : x + 3 * y = 1) : (1 / x + 1 / (3 * y)) = 4 :=
by sorry

end min_value_perpendicular_vectors_l7_7591


namespace max_discount_rate_l7_7228

-- Define the conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1
def min_profit : ℝ := cost_price * min_profit_margin

-- Define the discount rate
def discount_rate (x : ℝ) : ℝ :=
  selling_price * (1 - x / 100)

-- Define the profit after discount
def profit_after_discount (x : ℝ) : ℝ :=
  discount_rate x - cost_price

-- The statement we want to prove
theorem max_discount_rate : ∃ x : ℝ, x = 8.8 ∧ profit_after_discount x ≥ min_profit := 
by
  sorry

end max_discount_rate_l7_7228


namespace part_one_part_two_l7_7888

variable {x m : ℝ}

theorem part_one (h1 : ∀ x : ℝ, ¬(m * x^2 - (m + 1) * x + (m + 1) ≥ 0)) : m < -1 := sorry

theorem part_two (h2 : ∀ x : ℝ, 1 < x → m * x^2 - (m + 1) * x + (m + 1) ≥ 0) : m ≥ 1 / 3 := sorry

end part_one_part_two_l7_7888


namespace eccentricity_equilateral_l7_7244

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) :=
  let c := real.sqrt (a^2 - b^2) in
  let e := c / a in
  e

theorem eccentricity_equilateral (a b : ℝ) (h : a > b ∧ b > 0)
  (hc : c = real.sqrt (a^2 - b^2))
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (F1 := (-c, 0)) (F2 := (c, 0)) :
  let PQ := dist P Q in
  let F1P := dist F1 P in
  let F2P := dist F2 P in
  let F1Q := dist F1 Q in
  let F2Q := dist F2 Q in
  let e := c / a in
  PQ = PF1 ∧ PQ = PF2 ∧ PQ = QF1 ∧ PQ = QF2 ∧ PQ = PF2 ∧ PQ = QF2 → 
  2 * e = real.sqrt 3 - real.sqrt 3 * e^2 → 
  e = real.sqrt 3 / 3 :=
sorry

end eccentricity_equilateral_l7_7244


namespace sequence_equiv_l7_7577

theorem sequence_equiv (n : ℕ) (hn : n > 0) : ∃ (p : ℕ), p > 0 ∧ (4 * p + 5 = (3^n)^2) :=
by
  sorry

end sequence_equiv_l7_7577


namespace train_length_l7_7547

noncomputable def convert_speed (v_kmh : ℝ) : ℝ :=
  v_kmh * (5 / 18)

def length_of_train (speed_mps : ℝ) (time_sec : ℝ) : ℝ :=
  speed_mps * time_sec

theorem train_length (v_kmh : ℝ) (t_sec : ℝ) (length_m : ℝ) :
  v_kmh = 60 →
  t_sec = 45 →
  length_m = 750 →
  length_of_train (convert_speed v_kmh) t_sec = length_m :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end train_length_l7_7547


namespace max_points_of_intersection_l7_7993

open Set

def Point := ℝ × ℝ

structure Circle :=
(center : Point)
(radius : ℝ)

structure Line :=
(coeffs : ℝ × ℝ × ℝ) -- Assume line equation in the form Ax + By + C = 0

def max_intersection_points (circle : Circle) (lines : List Line) : ℕ :=
  let circle_line_intersect_count := 2
  let line_line_intersect_count := 1
  
  let number_of_lines := lines.length
  let pairwise_line_intersections := number_of_lines.choose 2
  
  let circle_and_lines_intersections := circle_line_intersect_count * number_of_lines
  let total_intersections := circle_and_lines_intersections + pairwise_line_intersections

  total_intersections

theorem max_points_of_intersection (c : Circle) (l1 l2 l3 : Line) :
  max_intersection_points c [l1, l2, l3] = 9 :=
by
  sorry

end max_points_of_intersection_l7_7993


namespace cost_equality_and_inequality_l7_7232

section CommunicationCost

-- Definitions based on given conditions
def y1 (x : ℝ) := 50 + 0.4 * x
def y2 (x : ℝ) := 0.6 * x

-- Theorem to prove given questions and conditions
theorem cost_equality_and_inequality (x : ℝ) :
  (y1 125 = y2 125) ∧ (∀ x > 125, y1 x < y2 x) :=
by
  sorry
  
end CommunicationCost

end cost_equality_and_inequality_l7_7232


namespace number_of_pipes_l7_7167

theorem number_of_pipes (h_same_height : forall (height : ℝ), height > 0)
  (diam_large : ℝ) (hl : diam_large = 6)
  (diam_small : ℝ) (hs : diam_small = 1) :
  (π * (diam_large / 2)^2) / (π * (diam_small / 2)^2) = 36 :=
by
  sorry

end number_of_pipes_l7_7167


namespace inclination_angle_l7_7454

noncomputable theory

-- Given point P
def P : ℝ × ℝ := (0, real.sqrt 3)

-- Line l with angle of inclination α passing through P
def parametric_line (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * real.cos α, real.sqrt 3 + t * real.sin α)

-- Polar equation of circle C translated to Cartesian form
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - real.sqrt 3)^2 = 5

-- Main theorem to prove the inclination angle α
theorem inclination_angle (α : ℝ) :
  (∀ (t₁ t₂ : ℝ),
    circle_C (parametric_line α t₁).1 (parametric_line α t₁).2 ∧
    circle_C (parametric_line α t₂).1 (parametric_line α t₂).2 →
    abs (t₁ - t₂) = real.sqrt 2) →
  (α = real.pi / 4 ∨ α = 3 * real.pi / 4) := by
  sorry

end inclination_angle_l7_7454


namespace locus_of_T_is_pair_of_tangents_l7_7078

noncomputable def locus_of_T (C : Circle) (L : Line) (O : Point) (hO : O ∈ L) : Set Point :=
  {T | ∃ (P : Point) (hP : P ∈ L), is_tangent (Circle.mk P (dist P O)) C T ∧ dist P T = dist P O}

theorem locus_of_T_is_pair_of_tangents (C : Circle) (L : Line) (O : Point) (hO : O ∈ L) :
  locus_of_T C L O hO =
  {T | ∃ (P1 P2 : Point), is_tangent P1 C T ∧ is_perpendicular L (Line.mk P1 O) ∧ is_tangent P2 C T ∧ is_perpendicular L (Line.mk P2 O)} :=
sorry

end locus_of_T_is_pair_of_tangents_l7_7078


namespace sqrt_product_eq_l7_7831

theorem sqrt_product_eq : Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end sqrt_product_eq_l7_7831


namespace sum_common_divisors_l7_7067

-- Define the sum of a set of numbers
def set_sum (s : Set ℕ) : ℕ :=
  s.fold (λ x acc => x + acc) 0

-- Define the divisors of a number
def divisors (n : ℕ) : Set ℕ :=
  { d | d > 0 ∧ n % d = 0 }

-- Definitions based on the given conditions
def divisors_of_60 : Set ℕ := divisors 60
def divisors_of_18 : Set ℕ := divisors 18
def common_divisors : Set ℕ := divisors_of_60 ∩ divisors_of_18

-- Declare the theorem to be proved
theorem sum_common_divisors : set_sum common_divisors = 12 :=
  sorry

end sum_common_divisors_l7_7067


namespace sum_gcd_lcm_75_4410_l7_7652

theorem sum_gcd_lcm_75_4410 :
  Nat.gcd 75 4410 + Nat.lcm 75 4410 = 22065 := by
  sorry

end sum_gcd_lcm_75_4410_l7_7652


namespace inverse_B_squared_l7_7481

-- Defining the inverse matrix B_inv
def B_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 0, 1]

-- Theorem to prove that the inverse of B^2 is a specific matrix
theorem inverse_B_squared :
  (B_inv * B_inv) = !![9, -6; 0, 1] :=
  by sorry


end inverse_B_squared_l7_7481


namespace triangle_properties_l7_7472

theorem triangle_properties (A B C a b c : ℝ) (h1 : a * Real.tan C = 2 * c * Real.sin A)
  (h2 : C > 0 ∧ C < Real.pi)
  (h3 : a / Real.sin A = c / Real.sin C) :
  C = Real.pi / 3 ∧ (1 / 2 < Real.sin (A + Real.pi / 6) ∧ Real.sin (A + Real.pi / 6) ≤ 1) →
  (Real.sqrt 3 / 2 < Real.sin A + Real.sin B ∧ Real.sin A + Real.sin B ≤ Real.sqrt 3) :=
by
  intro h4
  sorry

end triangle_properties_l7_7472


namespace angle_bisectors_l7_7675

open Real

noncomputable def r1 : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def r2 : ℝ × ℝ × ℝ := (0, 1, 1)

theorem angle_bisectors :
  ∃ (phi : ℝ), 0 ≤ phi ∧ phi ≤ π ∧ cos phi = 1 / 2 :=
sorry

end angle_bisectors_l7_7675


namespace shark_sightings_in_Daytona_Beach_l7_7063

def CM : ℕ := 7

def DB : ℕ := 3 * CM + 5

theorem shark_sightings_in_Daytona_Beach : DB = 26 := by
  sorry

end shark_sightings_in_Daytona_Beach_l7_7063


namespace smallest_number_among_neg2_neg1_0_pi_l7_7551

/-- The smallest number among -2, -1, 0, and π is -2. -/
theorem smallest_number_among_neg2_neg1_0_pi : min (min (min (-2 : ℝ) (-1)) 0) π = -2 := 
sorry

end smallest_number_among_neg2_neg1_0_pi_l7_7551


namespace red_suit_top_card_probability_l7_7233

theorem red_suit_top_card_probability :
  let num_cards := 104
  let num_red_suits := 4
  let cards_per_suit := 26
  let num_red_cards := num_red_suits * cards_per_suit
  let top_card_is_red_probability := num_red_cards / num_cards
  top_card_is_red_probability = 1 := by
  sorry

end red_suit_top_card_probability_l7_7233


namespace height_of_first_building_l7_7041

theorem height_of_first_building (h : ℕ) (h_condition : h + 2 * h + 9 * h = 7200) : h = 600 :=
by
  sorry

end height_of_first_building_l7_7041


namespace average_number_of_ducks_l7_7546

def average_ducks (A E K : ℕ) : ℕ :=
  (A + E + K) / 3

theorem average_number_of_ducks :
  ∀ (A E K : ℕ), A = 2 * E → E = K - 45 → A = 30 → average_ducks A E K = 35 :=
by 
  intros A E K h1 h2 h3
  sorry

end average_number_of_ducks_l7_7546


namespace f_is_odd_f_inequality_l7_7884

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 - 3^x else -1 + 3^(-x)

theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) :=
by
  intro x
  unfold f
  split_ifs;
  {
    sorry
  }

theorem f_inequality (a : ℝ) (h_a : a ≥ 6) : ∀ x : ℝ, (2 ≤ x ∧ x ≤ 8) → f ((Real.log x / Real.log 2) ^ 2) + f (5 - a * (Real.log x / Real.log 2)) ≥ 0 :=
by
  intros x hx
  have hlog1 : 1 ≤ Real.log x / Real.log 2 := sorry
  have hlog2 : Real.log x / Real.log 2 ≤ 3 := sorry
  let t := Real.log x / Real.log 2
  let g := λ t : ℝ, t^2 - a * t + 5
  have h_gmax : ∀ t : ℝ, (1 ≤ t ∧ t ≤ 3) → g t ≤ 0 := sorry
  exact h_gmax t ⟨hlog1, hlog2⟩

end f_is_odd_f_inequality_l7_7884


namespace albums_total_l7_7942

noncomputable def miriam_albums (katrina_albums : ℕ) := 5 * katrina_albums
noncomputable def katrina_albums (bridget_albums : ℕ) := 6 * bridget_albums
noncomputable def bridget_albums (adele_albums : ℕ) := adele_albums - 15
noncomputable def total_albums (adele_albums : ℕ) (bridget_albums : ℕ) (katrina_albums : ℕ) (miriam_albums : ℕ) := 
  adele_albums + bridget_albums + katrina_albums + miriam_albums

theorem albums_total (adele_has_30 : adele_albums = 30) : 
  total_albums 30 (bridget_albums 30) (katrina_albums (bridget_albums 30)) (miriam_albums (katrina_albums (bridget_albums 30))) = 585 := 
by 
  -- In Lean, replace the term 'sorry' below with the necessary steps to finish the proof
  sorry

end albums_total_l7_7942


namespace number_of_Cl_atoms_l7_7661

/-- 
Given a compound with 1 aluminum atom and a molecular weight of 132 g/mol,
prove that the number of chlorine atoms in the compound is 3.
--/
theorem number_of_Cl_atoms 
  (weight_Al : ℝ) 
  (weight_Cl : ℝ) 
  (molecular_weight : ℝ)
  (ha : weight_Al = 26.98)
  (hc : weight_Cl = 35.45)
  (hm : molecular_weight = 132) :
  ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_Cl_atoms_l7_7661


namespace three_gt_sqrt_seven_l7_7684

theorem three_gt_sqrt_seven : (3 : ℝ) > real.sqrt 7 := 
sorry

end three_gt_sqrt_seven_l7_7684


namespace compute_exponent_problem_l7_7827

noncomputable def exponent_problem : ℤ :=
  3 * (3^4) - (9^60) / (9^57)

theorem compute_exponent_problem : exponent_problem = -486 := by
  sorry

end compute_exponent_problem_l7_7827


namespace probability_of_rolling_2_4_6_on_8_sided_die_l7_7027

theorem probability_of_rolling_2_4_6_on_8_sided_die : 
  ∀ (ω : Fin 8), 
  (1 / 8) * (ite (ω = 1 ∨ ω = 3 ∨ ω = 5) 1 0) = 3 / 8 := 
by 
  sorry

end probability_of_rolling_2_4_6_on_8_sided_die_l7_7027


namespace smaller_number_is_180_l7_7197

theorem smaller_number_is_180 (a b : ℕ) (h1 : a = 3 * b) (h2 : a + 4 * b = 420) :
  a = 180 :=
sorry

end smaller_number_is_180_l7_7197


namespace truncatedPyramidVolume_l7_7542

noncomputable def volumeOfTruncatedPyramid (R : ℝ) : ℝ :=
  let h := R * Real.sqrt 3 / 2
  let S_lower := 3 * R^2 * Real.sqrt 3 / 2
  let S_upper := 3 * R^2 * Real.sqrt 3 / 8
  let sqrt_term := Real.sqrt (S_lower * S_upper)
  (1/3) * h * (S_lower + S_upper + sqrt_term)

theorem truncatedPyramidVolume (R : ℝ) (h := R * Real.sqrt 3 / 2)
  (S_lower := 3 * R^2 * Real.sqrt 3 / 2)
  (S_upper := 3 * R^2 * Real.sqrt 3 / 8)
  (V := (1/3) * h * (S_lower + S_upper + Real.sqrt (S_lower * S_upper))) :
  volumeOfTruncatedPyramid R = 21 * R^3 / 16 := by
  sorry

end truncatedPyramidVolume_l7_7542


namespace quadruple_nested_function_l7_7983

def a (k : ℕ) : ℕ := (k + 1) ^ 2

theorem quadruple_nested_function (k : ℕ) (h : k = 1) : a (a (a (a (k)))) = 458329 :=
by
  rw [h]
  sorry

end quadruple_nested_function_l7_7983


namespace polynomial_factorization_l7_7787

noncomputable def polyExpression (a b c : ℕ) : ℕ := a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4

theorem polynomial_factorization (a b c : ℕ) :
  ∃ q : ℕ → ℕ → ℕ → ℕ, q a b c = (a + b + c)^3 - 3 * a * b * c ∧
  polyExpression a b c = (a - b) * (b - c) * (c - a) * q a b c := by
  -- The proof goes here
  sorry

end polynomial_factorization_l7_7787


namespace contemporary_probability_l7_7182

noncomputable def lifespan_distribution := uniform 50 120

def probability_of_contemporary : ℝ :=
  let overlap_area := 640000 - 7245 in
  overlap_area / 640000

theorem contemporary_probability : 
  (∀ (L_Alice L_Bob : ℝ), (L_Alice ~ lifespan_distribution) ∧ (L_Bob ~ lifespan_distribution) →
  P((L_Alice and L_Bob are contemporaries)) = probability_of_contemporary :=
sorry

end contemporary_probability_l7_7182


namespace part1_part2_l7_7082

variable (x : ℝ)

def A : Set ℝ := { x | 2 * x + 1 < 5 }
def B : Set ℝ := { x | x^2 - x - 2 < 0 }

theorem part1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem part2 : A ∪ { x | x ≤ -1 ∨ x ≥ 2 } = Set.univ :=
sorry

end part1_part2_l7_7082


namespace equation_has_two_solutions_l7_7508

theorem equation_has_two_solutions (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^x1 = x1^2 - 2*x1 - a ∧ a^x2 = x2^2 - 2*x2 - a :=
sorry

end equation_has_two_solutions_l7_7508


namespace sum_common_divisors_60_18_l7_7066

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end sum_common_divisors_60_18_l7_7066


namespace range_of_a_l7_7270

open Set

variable {a : ℝ} 

def M (a : ℝ) : Set ℝ := {x : ℝ | -4 * x + 4 * a < 0 }

theorem range_of_a (hM : 2 ∉ M a) : a ≥ 2 :=
by
  sorry

end range_of_a_l7_7270


namespace smallest_rel_prime_210_l7_7702

theorem smallest_rel_prime_210 (x : ℕ) (hx : x > 1) (hrel_prime : Nat.gcd x 210 = 1) : x = 11 :=
sorry

end smallest_rel_prime_210_l7_7702


namespace evaluate_f_at_2_l7_7902

def f (x : ℝ) : ℝ := x^2 - x

theorem evaluate_f_at_2 : f 2 = 2 := by
  sorry

end evaluate_f_at_2_l7_7902


namespace LaurynCompanyEmployees_l7_7922

noncomputable def LaurynTotalEmployees (men women total : ℕ) : Prop :=
  men = 80 ∧ women = men + 20 ∧ total = men + women

theorem LaurynCompanyEmployees : ∃ total, ∀ men women, LaurynTotalEmployees men women total → total = 180 :=
by 
  sorry

end LaurynCompanyEmployees_l7_7922


namespace smallest_coprime_to_210_l7_7709

theorem smallest_coprime_to_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 210 = 1 → y ≥ x :=
by
  sorry

end smallest_coprime_to_210_l7_7709


namespace slope_negative_l7_7260

theorem slope_negative (k b m n : ℝ) (h₁ : k ≠ 0) (h₂ : m < n) 
  (ha : m = k * 1 + b) (hb : n = k * -1 + b) : k < 0 :=
by
  sorry

end slope_negative_l7_7260


namespace sum_of_square_roots_l7_7380

theorem sum_of_square_roots :
  (Real.sqrt 1 + Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4)) = 
  (1 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10) := 
sorry

end sum_of_square_roots_l7_7380


namespace compute_exponent_problem_l7_7826

noncomputable def exponent_problem : ℤ :=
  3 * (3^4) - (9^60) / (9^57)

theorem compute_exponent_problem : exponent_problem = -486 := by
  sorry

end compute_exponent_problem_l7_7826


namespace solve_four_tuple_l7_7417

-- Define the problem conditions
theorem solve_four_tuple (a b c d : ℝ) : 
    (ab + c + d = 3) → 
    (bc + d + a = 5) → 
    (cd + a + b = 2) → 
    (da + b + c = 6) → 
    (a = 2) ∧ (b = 0) ∧ (c = 0) ∧ (d = 3) :=
by
  intros h1 h2 h3 h4
  sorry

end solve_four_tuple_l7_7417


namespace value_of_a10_l7_7088

/-- Define arithmetic sequence and properties -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n) / 2)

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
axiom arith_seq : arithmetic_sequence a d
axiom sum_formula : sum_of_first_n_terms a 5 S
axiom sum_condition : S 5 = 60
axiom term_condition : a 1 + a 2 + a 3 = a 4 + a 5

theorem value_of_a10 : a 10 = 26 :=
sorry

end value_of_a10_l7_7088


namespace base_of_first_term_l7_7645

theorem base_of_first_term (e : ℕ) (b : ℝ) (h : e = 35) :
  b^e * (1/4)^18 = 1/(2 * 10^35) → b = 1/5 :=
by
  sorry

end base_of_first_term_l7_7645


namespace expected_value_of_groups_l7_7171

noncomputable def expectedNumberOfGroups (k m : ℕ) : ℝ :=
  1 + (2 * k * m) / (k + m)

theorem expected_value_of_groups (k m : ℕ) :
  k > 0 → m > 0 → expectedNumberOfGroups k m = 1 + 2 * k * m / (k + m) :=
by
  intros
  unfold expectedNumberOfGroups
  sorry

end expected_value_of_groups_l7_7171


namespace insphere_radius_l7_7468

theorem insphere_radius (V S : ℝ) (hV : V > 0) (hS : S > 0) : 
  ∃ r : ℝ, r = 3 * V / S := by
  sorry

end insphere_radius_l7_7468


namespace coefficient_of_a3b2_in_expansion_l7_7802

-- Define the binomial coefficient function.
def binom : ℕ → ℕ → ℕ
| n k := nat.choose n k

-- Define the coefficient of a^3b^2 in (a + b)^5
def coefficient_ab : ℕ :=
  binom 5 3

-- Define the constant term in (c + 1/c)^8
def constant_term : ℕ :=
  binom 8 4

-- Define the final coefficient of a^3b^2 in (a + b)^5 * (c + 1/c)^8
def final_coefficient : ℕ :=
  coefficient_ab * constant_term

-- The main statement to prove.
theorem coefficient_of_a3b2_in_expansion : final_coefficient = 700 :=
by
  sorry  -- Proof to be provided

end coefficient_of_a3b2_in_expansion_l7_7802


namespace initial_bacteria_count_l7_7499

theorem initial_bacteria_count (doubling_time : ℕ) (initial_time : ℕ) (initial_bacteria : ℕ) 
(final_bacteria : ℕ) (doubling_rate : initial_time / doubling_time = 8 ∧ final_bacteria = 524288) : 
  initial_bacteria = 2048 :=
by
  sorry

end initial_bacteria_count_l7_7499


namespace average_community_age_l7_7741

variable (num_women num_men : Nat)
variable (avg_age_women avg_age_men : Nat)

def ratio_women_men := num_women = 7 * num_men / 8
def average_age_women := avg_age_women = 30
def average_age_men := avg_age_men = 35

theorem average_community_age (k : Nat) 
  (h_ratio : ratio_women_men (7 * k) (8 * k)) 
  (h_avg_women : average_age_women 30)
  (h_avg_men : average_age_men 35) : 
  (30 * (7 * k) + 35 * (8 * k)) / (15 * k) = 32 + (2 / 3) := 
sorry

end average_community_age_l7_7741


namespace nolan_total_savings_l7_7329

-- Define the conditions given in the problem
def monthly_savings : ℕ := 3000
def number_of_months : ℕ := 12

-- State the equivalent proof problem in Lean 4
theorem nolan_total_savings : (monthly_savings * number_of_months) = 36000 := by
  -- Proof is omitted
  sorry

end nolan_total_savings_l7_7329


namespace max_discount_rate_l7_7222

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l7_7222


namespace sum_of_possible_x_values_l7_7492

theorem sum_of_possible_x_values (x : ℝ) : 
  (3 : ℝ)^(x^2 + 6*x + 9) = (27 : ℝ)^(x + 3) → x = 0 ∨ x = -3 → x = 0 ∨ x = -3 := 
sorry

end sum_of_possible_x_values_l7_7492


namespace max_discount_rate_l7_7223

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l7_7223


namespace gcd_factorial_sub_one_l7_7928

theorem gcd_factorial_sub_one (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p > q) :
  Nat.gcd (p.factorial - 1) (q.factorial - 1) ≤ p ^ (5/3 : ℚ) := 
sorry

end gcd_factorial_sub_one_l7_7928


namespace geom_inequality_l7_7582

noncomputable def geom_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geom_inequality (a1 q : ℝ) (h_q : q ≠ 0) :
  (a1 * (a1 * q^2)) > 0 :=
by
  sorry

end geom_inequality_l7_7582


namespace a_8_is_256_l7_7469

variable (a : ℕ → ℕ)

axiom a_1 : a 1 = 2

axiom a_pq : ∀ p q : ℕ, a (p + q) = a p * a q

theorem a_8_is_256 : a 8 = 256 := by
  sorry

end a_8_is_256_l7_7469


namespace points_opposite_sides_l7_7086

theorem points_opposite_sides (x y : ℝ) (h : (3 * x + 2 * y - 8) * (-1) < 0) : 3 * x + 2 * y > 8 := 
by
  sorry

end points_opposite_sides_l7_7086


namespace part1_part2_l7_7879

def setA (a : ℝ) := {x : ℝ | a - 1 ≤ x ∧ x ≤ 3 - 2 * a}
def setB := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}

theorem part1 (a : ℝ) : (setA a ∪ setB = setB) ↔ (-(1 / 2) ≤ a) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ∈ setB ↔ x ∈ setA a) ↔ (a ≤ -1) :=
sorry

end part1_part2_l7_7879


namespace line_equation_l7_7429

theorem line_equation {x y : ℝ} (h : (x = 1) ∧ (y = -3)) :
  ∃ c : ℝ, x - 2 * y + c = 0 ∧ c = 7 :=
by
  sorry

end line_equation_l7_7429


namespace final_percentage_is_46_l7_7995

def initial_volume : ℚ := 50
def initial_concentration : ℚ := 0.60
def drained_volume : ℚ := 35
def replacement_concentration : ℚ := 0.40

def initial_chemical_amount : ℚ := initial_volume * initial_concentration
def drained_chemical_amount : ℚ := drained_volume * initial_concentration
def remaining_chemical_amount : ℚ := initial_chemical_amount - drained_chemical_amount
def added_chemical_amount : ℚ := drained_volume * replacement_concentration
def final_chemical_amount : ℚ := remaining_chemical_amount + added_chemical_amount
def final_volume : ℚ := initial_volume

def final_percentage : ℚ := (final_chemical_amount / final_volume) * 100

theorem final_percentage_is_46 :
  final_percentage = 46 := by
  sorry

end final_percentage_is_46_l7_7995


namespace expectation_sum_ne_sum_expectation_l7_7856

open MeasureTheory

-- Define the indicator function
def indicator {α : Type*} (s : set α) [decidable_pred s] (x : α) : ℝ :=
if x ∈ s then 1 else 0

noncomputable def xi (n : ℕ) (U : ℝ) : ℝ :=
n * indicator {x : ℝ | n * x ≤ 1} U - (n - 1) * indicator {x : ℝ | (n - 1) * x ≤ 1} U

theorem expectation_sum_ne_sum_expectation (U : ℝ) (hU : U ∈ set.Icc 0.0 1.0) :
  ∃ (xi : ℕ → ℝ), (ℝ → ℝ) (∞) -> ¬(∑' n, (xi n)) = ∑' n, xi :=
begin
  sorry
end

end expectation_sum_ne_sum_expectation_l7_7856


namespace system_solution_l7_7728

theorem system_solution :
  ∀ (a1 b1 c1 a2 b2 c2 : ℝ),
  (a1 * 8 + b1 * 5 = c1) ∧ (a2 * 8 + b2 * 5 = c2) →
  ∃ (x y : ℝ), (4 * a1 * x - 5 * b1 * y = 3 * c1) ∧ (4 * a2 * x - 5 * b2 * y = 3 * c2) ∧ 
               (x = 6) ∧ (y = -3) :=
by
  sorry

end system_solution_l7_7728


namespace find_b_l7_7642

-- Define the curve and the line equations
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x + 1
def line (k : ℝ) (b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the conditions in the problem
def passes_through_point (a : ℝ) : Prop := curve a 2 = 3
def is_tangent_at_point (a k b : ℝ) : Prop :=
  ∀ x : ℝ, curve a x = 3 → line k b 2 = 3

-- Main theorem statement
theorem find_b (a k b : ℝ) (h1 : passes_through_point a) (h2 : is_tangent_at_point a k b) : b = -15 :=
by sorry

end find_b_l7_7642


namespace exists_arithmetic_progression_with_sum_zero_l7_7241

theorem exists_arithmetic_progression_with_sum_zero : 
  ∃ (a d : Int) (n : Int), n > 0 ∧ (n * (2 * a + (n - 1) * d)) = 0 :=
by 
  sorry

end exists_arithmetic_progression_with_sum_zero_l7_7241


namespace range_of_fx_over_x_l7_7163

variable (f : ℝ → ℝ)

noncomputable def is_odd (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

theorem range_of_fx_over_x (odd_f : is_odd f)
                           (increasing_f_pos : is_increasing_on f {x : ℝ | x > 0})
                           (hf1 : f (-1) = 0) :
  {x | f x / x < 0} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end range_of_fx_over_x_l7_7163


namespace car_fuel_efficiency_l7_7392

theorem car_fuel_efficiency 
  (H C T : ℤ)
  (h₁ : 900 = H * T)
  (h₂ : 600 = C * T)
  (h₃ : C = H - 5) :
  C = 10 := by
  sorry

end car_fuel_efficiency_l7_7392


namespace hannah_age_l7_7691

-- Define the constants and conditions
variables (E F G H : ℕ)
axiom h₁ : E = F - 4
axiom h₂ : F = G + 6
axiom h₃ : H = G + 2
axiom h₄ : E = 15

-- Prove that Hannah is 15 years old
theorem hannah_age : H = 15 :=
by sorry

end hannah_age_l7_7691


namespace max_discount_rate_l7_7227

theorem max_discount_rate 
  (cost_price : ℝ) (selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 → selling_price = 5 → min_profit_margin = 0.1 →
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 8.8 ∧ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_rate_l7_7227


namespace range_of_a_l7_7927

noncomputable def condition_p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
noncomputable def condition_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬(∀ x, condition_p x)) → (¬(∀ x, condition_q x a)) → 
  (∀ x, condition_p x ↔ condition_q x a) → (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l7_7927


namespace log2_real_coeff_sum_l7_7930

open Complex Nat

theorem log2_real_coeff_sum 
    (T : ℝ)
    (hT : T = (∑ k in Finset.range (2012), if even k then (↑((choose 2011 k : ℕ) * (I ^ k)) * (1 ^ k)).re else 0)) :
    log 2 T = 1005 :=
sorry

end log2_real_coeff_sum_l7_7930


namespace increasing_interval_of_f_l7_7164

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * Real.pi / 3 - 2 * x)

theorem increasing_interval_of_f :
  ∃ a b : ℝ, f x = 3 * Real.sin (2 * Real.pi / 3 - 2 * x) ∧ (a = 7 * Real.pi / 12) ∧ (b = 13 * Real.pi / 12) ∧ ∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2 := 
sorry

end increasing_interval_of_f_l7_7164


namespace min_f_abs_l7_7804

def f (x y : ℤ) : ℤ := 5 * x^2 + 11 * x * y - 5 * y^2

theorem min_f_abs (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : (∃ m, ∀ x y : ℤ, (x ≠ 0 ∨ y ≠ 0) → |f x y| ≥ m) ∧ 5 = 5 :=
by
  sorry -- proof goes here

end min_f_abs_l7_7804


namespace grassy_pathway_area_correct_l7_7541

-- Define the dimensions of the plot and the pathway width
def length_plot : ℝ := 15
def width_plot : ℝ := 10
def width_pathway : ℝ := 2

-- Define the required areas
def total_area : ℝ := (length_plot + 2 * width_pathway) * (width_plot + 2 * width_pathway)
def plot_area : ℝ := length_plot * width_plot
def grassy_pathway_area : ℝ := total_area - plot_area

-- Prove that the area of the grassy pathway is 116 m²
theorem grassy_pathway_area_correct : grassy_pathway_area = 116 := by
  sorry

end grassy_pathway_area_correct_l7_7541


namespace force_required_l7_7347

theorem force_required 
  (F : ℕ → ℕ)
  (h_inv : ∀ L L' : ℕ, F L * L = F L' * L')
  (h1 : F 12 = 300) :
  F 18 = 200 :=
by
  sorry

end force_required_l7_7347


namespace cost_of_one_of_the_shirts_l7_7438

theorem cost_of_one_of_the_shirts
    (total_cost : ℕ) 
    (cost_two_shirts : ℕ) 
    (num_equal_shirts : ℕ) 
    (cost_of_shirt : ℕ) :
    total_cost = 85 → 
    cost_two_shirts = 20 → 
    num_equal_shirts = 3 → 
    cost_of_shirt = (total_cost - 2 * cost_two_shirts) / num_equal_shirts → 
    cost_of_shirt = 15 :=
by
  intros
  sorry

end cost_of_one_of_the_shirts_l7_7438


namespace min_sum_of_squares_l7_7931

theorem min_sum_of_squares 
  (x_1 x_2 x_3 : ℝ)
  (h1: x_1 + 3 * x_2 + 4 * x_3 = 72)
  (h2: x_1 = 3 * x_2)
  (h3: 0 < x_1)
  (h4: 0 < x_2)
  (h5: 0 < x_3) : 
  x_1^2 + x_2^2 + x_3^2 = 347.04 := 
sorry

end min_sum_of_squares_l7_7931


namespace check_true_propositions_l7_7090

open Set

theorem check_true_propositions : 
  ∀ (Prop1 Prop2 Prop3 : Prop),
    (Prop1 ↔ (∀ x : ℝ, x^2 > 0)) →
    (Prop2 ↔ ∃ x : ℝ, x^2 ≤ x) →
    (Prop3 ↔ ∀ (M N : Set ℝ) (x : ℝ), x ∈ (M ∩ N) → x ∈ M ∧ x ∈ N) →
    (¬Prop1 ∧ Prop2 ∧ Prop3) →
    (2 = 2) := sorry

end check_true_propositions_l7_7090


namespace num_valid_seat_permutations_l7_7422

/-- 
  The number of ways eight people can switch their seats in a circular 
  arrangement such that no one sits in the same, adjacent, or directly 
  opposite chair they originally occupied is 5.
-/
theorem num_valid_seat_permutations : 
  ∃ (σ : Equiv.Perm (Fin 8)), 
  (∀ i : Fin 8, σ i ≠ i) ∧ 
  (∀ i : Fin 8, σ i ≠ if i.val < 7 then i + 1 else 0) ∧ 
  (∀ i : Fin 8, σ i ≠ if i.val < 8 / 2 then (i + 8 / 2) % 8 else (i - 8 / 2) % 8) :=
  sorry

end num_valid_seat_permutations_l7_7422


namespace shortest_distance_to_line_l7_7150

open Classical

variables {P A B C : Type} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (PA PB PC : ℝ)
variables (l : ℕ) -- l represents the line

-- Given conditions
def PA_dist : ℝ := 4
def PB_dist : ℝ := 5
def PC_dist : ℝ := 2

theorem shortest_distance_to_line (hPA : PA = PA_dist) (hPB : PB = PB_dist) (hPC : PC = PC_dist) :
  ∃ d, d ≤ 2 := 
sorry

end shortest_distance_to_line_l7_7150


namespace max_positive_n_l7_7456

def a (n : ℕ) : ℤ := 19 - 2 * n

theorem max_positive_n (n : ℕ) (h : a n > 0) : n ≤ 9 :=
by
  sorry

end max_positive_n_l7_7456


namespace solve_z_l7_7865

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l7_7865


namespace a_values_condition_l7_7738

def is_subset (A B : Set ℝ) : Prop := ∀ x, x ∈ A → x ∈ B

theorem a_values_condition (a : ℝ) : 
  (2 * a + 1 ≤ 3 ∧ 3 * a - 5 ≤ 22 ∧ 2 * a + 1 ≤ 3 * a - 5) 
  ↔ (6 ≤ a ∧ a ≤ 9) :=
by 
  sorry

end a_values_condition_l7_7738


namespace xyz_value_l7_7791

theorem xyz_value (x y z : ℕ) (h1 : x + 2 * y = z) (h2 : x^2 - 4 * y^2 + z^2 = 310) :
  xyz = 4030 ∨ xyz = 23870 :=
by
  -- placeholder for proof steps
  sorry

end xyz_value_l7_7791


namespace at_least_one_fraction_less_than_two_l7_7076

theorem at_least_one_fraction_less_than_two {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
by
  sorry

end at_least_one_fraction_less_than_two_l7_7076


namespace jury_selection_duration_is_two_l7_7475

variable (jury_selection_days : ℕ) (trial_days : ℕ) (deliberation_days : ℕ)

axiom trial_lasts_four_times_jury_selection : trial_days = 4 * jury_selection_days
axiom deliberation_is_six_full_days : deliberation_days = (6 * 24) / 16
axiom john_spends_nineteen_days : jury_selection_days + trial_days + deliberation_days = 19

theorem jury_selection_duration_is_two : jury_selection_days = 2 :=
by
  sorry

end jury_selection_duration_is_two_l7_7475


namespace remainder_8_pow_2023_mod_5_l7_7024

theorem remainder_8_pow_2023_mod_5 :
  8 ^ 2023 % 5 = 2 :=
by
  sorry

end remainder_8_pow_2023_mod_5_l7_7024


namespace compute_abs_difference_l7_7756

theorem compute_abs_difference (x y : ℝ) 
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 3.6)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 4.5) : 
  |x - y| = 1.1 :=
by 
  sorry

end compute_abs_difference_l7_7756


namespace base8_to_base10_4513_l7_7061

theorem base8_to_base10_4513 : (4 * 8^3 + 5 * 8^2 + 1 * 8^1 + 3 * 8^0 = 2379) :=
by
  sorry

end base8_to_base10_4513_l7_7061


namespace smallest_coprime_to_210_l7_7707

theorem smallest_coprime_to_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 210 = 1 → y ≥ x :=
by
  sorry

end smallest_coprime_to_210_l7_7707


namespace solve_for_z_l7_7870

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l7_7870


namespace sum_first_six_terms_arithmetic_seq_l7_7501

theorem sum_first_six_terms_arithmetic_seq :
  ∃ a_1 d : ℤ, (a_1 + 3 * d = 7) ∧ (a_1 + 4 * d = 12) ∧ (a_1 + 5 * d = 17) ∧ 
  (6 * (2 * a_1 + 5 * d) / 2 = 27) :=
by
  sorry

end sum_first_six_terms_arithmetic_seq_l7_7501


namespace overall_average_score_l7_7394

theorem overall_average_score (first_6_avg last_4_avg : ℝ) (n_first n_last n_total : ℕ) 
    (h_matches : n_first + n_last = n_total)
    (h_first_avg : first_6_avg = 41)
    (h_last_avg : last_4_avg = 35.75)
    (h_n_first : n_first = 6)
    (h_n_last : n_last = 4)
    (h_n_total : n_total = 10) :
    ((first_6_avg * n_first + last_4_avg * n_last) / n_total) = 38.9 := by
  sorry

end overall_average_score_l7_7394


namespace distance_points_lt_2_over_3_r_l7_7915

theorem distance_points_lt_2_over_3_r (r : ℝ) (h_pos_r : 0 < r) (points : Fin 17 → ℝ × ℝ)
  (h_points_in_circle : ∀ i, (points i).1 ^ 2 + (points i).2 ^ 2 < r ^ 2) :
  ∃ i j : Fin 17, i ≠ j ∧ (dist (points i) (points j) < 2 * r / 3) :=
by
  sorry

end distance_points_lt_2_over_3_r_l7_7915


namespace tyler_saltwater_animals_l7_7366

/-- Tyler had 56 aquariums for saltwater animals and each aquarium has 39 animals in it. 
    We need to prove that the total number of saltwater animals Tyler has is 2184. --/
theorem tyler_saltwater_animals : (56 * 39) = 2184 := by
  sorry

end tyler_saltwater_animals_l7_7366


namespace min_radius_circle_line_intersection_l7_7886

theorem min_radius_circle_line_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) (r : ℝ) (hr : r > 0)
    (intersect : ∃ (x y : ℝ), (x - Real.cos θ)^2 + (y - Real.sin θ)^2 = r^2 ∧ 2 * x - y - 10 = 0) :
    r ≥ 2 * Real.sqrt 5 - 1 :=
  sorry

end min_radius_circle_line_intersection_l7_7886


namespace solve_complex_equation_l7_7862

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l7_7862


namespace triangle_equilateral_l7_7593

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ A = B ∧ B = C

theorem triangle_equilateral 
  (a b c A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * a * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) : 
  is_equilateral_triangle a b c A B C :=
sorry

end triangle_equilateral_l7_7593


namespace inequality_generalization_l7_7451

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : n > 0) (hx : x > 0) 
  (h1 : x + 1 / x ≥ 2) (h2 : x + 4 / (x ^ 2) = (x / 2) + (x / 2) + 4 / (x ^ 2) ∧ (x / 2) + (x / 2) + 4 / (x ^ 2) ≥ 3) : 
  x + n^n / x^n ≥ n + 1 := 
sorry

end inequality_generalization_l7_7451


namespace hancho_milk_consumption_l7_7646

theorem hancho_milk_consumption :
  ∀ (initial_yeseul_consumption gayoung_bonus liters_left initial_milk consumption_yeseul consumption_gayoung consumption_total), 
  initial_yeseul_consumption = 0.1 →
  gayoung_bonus = 0.2 →
  liters_left = 0.3 →
  initial_milk = 1 →
  consumption_yeseul = initial_yeseul_consumption →
  consumption_gayoung = initial_yeseul_consumption + gayoung_bonus →
  consumption_total = consumption_yeseul + consumption_gayoung →
  (initial_milk - (consumption_total + liters_left)) = 0.3 :=
by sorry

end hancho_milk_consumption_l7_7646


namespace expected_lifetime_flashlight_l7_7323

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l7_7323


namespace evaluate_expression_l7_7424

theorem evaluate_expression : (3 : ℚ) / (1 - (2 : ℚ) / 5) = 5 := sorry

end evaluate_expression_l7_7424


namespace expected_flashlight_lifetime_leq_two_l7_7318

theorem expected_flashlight_lifetime_leq_two
  (Ω : Type*) [MeasurableSpace Ω] [ProbabilitySpace Ω]
  (ξ η : Ω → ℝ)
  (h_min_leq_xi : ∀ ω, min (ξ ω) (η ω) ≤ ξ ω)
  (h_expectation_xi : expectation (ξ) = 2) :
  expectation (fun ω => min (ξ ω) (η ω)) ≤ 2 := 
sorry

end expected_flashlight_lifetime_leq_two_l7_7318


namespace fence_pole_count_l7_7037

-- Define the conditions
def path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the goal
def total_poles : ℕ := 286

-- The statement to prove
theorem fence_pole_count :
  let total_length_to_fence := (path_length - bridge_length)
  let poles_per_side := total_length_to_fence / pole_spacing
  let total_poles_needed := 2 * poles_per_side
  total_poles_needed = total_poles :=
by
  sorry

end fence_pole_count_l7_7037


namespace least_number_to_addition_l7_7381

-- Given conditions
def n : ℤ := 2496

-- The least number to be added to n to make it divisible by 5
def least_number_to_add (n : ℤ) : ℤ :=
  if (n % 5 = 0) then 0 else (5 - (n % 5))

-- Prove that adding 4 to 2496 makes it divisible by 5
theorem least_number_to_addition : (least_number_to_add n) = 4 :=
  by
    sorry

end least_number_to_addition_l7_7381


namespace number_of_rows_of_desks_is_8_l7_7466

-- Definitions for the conditions
def first_row_desks : ℕ := 10
def desks_increment : ℕ := 2
def total_desks : ℕ := 136

-- Definition for the sum of an arithmetic series
def arithmetic_series_sum (n a1 d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- The proof problem statement
theorem number_of_rows_of_desks_is_8 :
  ∃ n : ℕ, arithmetic_series_sum n first_row_desks desks_increment = total_desks ∧ n = 8 :=
by
  sorry

end number_of_rows_of_desks_is_8_l7_7466


namespace reciprocal_of_2023_l7_7510

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem reciprocal_of_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l7_7510


namespace expression_pos_intervals_l7_7690

theorem expression_pos_intervals :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ∨ (x > 3) ↔ (x + 1) * (x - 1) * (x - 3) > 0 := by
  sorry

end expression_pos_intervals_l7_7690


namespace gcd_binom_is_integer_l7_7753

theorem gcd_binom_is_integer 
  (m n : ℤ) 
  (hm : m ≥ 1) 
  (hn : n ≥ m)
  (gcd_mn : ℤ := Int.gcd m n)
  (binom_nm : ℤ := Nat.choose n.toNat m.toNat) :
  (gcd_mn * binom_nm) % n.toNat = 0 := by
  sorry

end gcd_binom_is_integer_l7_7753


namespace probability_exactly_nine_matches_l7_7205

theorem probability_exactly_nine_matches (n : ℕ) (h : n = 10) : 
  (∃ p : ℕ, p = 9 ∧ probability_of_exact_matches n p = 0) :=
by {
  sorry
}

end probability_exactly_nine_matches_l7_7205


namespace total_weight_of_plastic_rings_l7_7920

-- Conditions
def orange_ring_weight : ℝ := 0.08
def purple_ring_weight : ℝ := 0.33
def white_ring_weight : ℝ := 0.42

-- Proof Statement
theorem total_weight_of_plastic_rings :
  orange_ring_weight + purple_ring_weight + white_ring_weight = 0.83 := by
  sorry

end total_weight_of_plastic_rings_l7_7920


namespace mike_spent_total_l7_7133

-- Define the prices of the items
def price_trumpet : ℝ := 145.16
def price_song_book : ℝ := 5.84

-- Define the total amount spent
def total_spent : ℝ := price_trumpet + price_song_book

-- The theorem to be proved
theorem mike_spent_total :
  total_spent = 151.00 :=
sorry

end mike_spent_total_l7_7133


namespace students_walk_fraction_l7_7555

theorem students_walk_fraction :
  (1 - (1/3 + 1/5 + 1/10 + 1/15)) = 3/10 :=
by sorry

end students_walk_fraction_l7_7555


namespace min_diff_between_y_and_x_l7_7911

theorem min_diff_between_y_and_x (x y z : ℤ)
    (h1 : x < y)
    (h2 : y < z)
    (h3 : Even x)
    (h4 : Odd y)
    (h5 : Odd z)
    (h6 : z - x = 9) :
    y - x = 1 := 
  by sorry

end min_diff_between_y_and_x_l7_7911


namespace number_of_rolls_in_case_l7_7218

-- Definitions: Cost of a case, cost per roll individually, percent savings per roll
def cost_of_case : ℝ := 9
def cost_per_roll_individual : ℝ := 1
def percent_savings_per_roll : ℝ := 0.25

-- Theorem: Proving the number of rolls in the case is 12
theorem number_of_rolls_in_case (n : ℕ) (h1 : cost_of_case = 9)
    (h2 : cost_per_roll_individual = 1)
    (h3 : percent_savings_per_roll = 0.25) : n = 12 := 
  sorry

end number_of_rolls_in_case_l7_7218


namespace total_miles_traveled_l7_7112

noncomputable def distance_to_first_museum : ℕ := 5
noncomputable def distance_to_second_museum : ℕ := 15
noncomputable def distance_to_cultural_center : ℕ := 10
noncomputable def extra_detour : ℕ := 3

theorem total_miles_traveled : 
  (2 * (distance_to_first_museum + extra_detour) + 2 * distance_to_second_museum + 2 * distance_to_cultural_center) = 66 :=
  by
  sorry

end total_miles_traveled_l7_7112


namespace correct_value_of_wrongly_read_number_l7_7959

theorem correct_value_of_wrongly_read_number 
  (avg_wrong : ℝ) (n : ℕ) (wrong_value : ℝ) (avg_correct : ℝ) :
  avg_wrong = 5 →
  n = 10 →
  wrong_value = 26 →
  avg_correct = 6 →
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  correct_value = 36 :=
by
  intros h_avg_wrong h_n h_wrong_value h_avg_correct
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  sorry

end correct_value_of_wrongly_read_number_l7_7959


namespace find_divisor_l7_7490

theorem find_divisor (x : ℕ) (h : 172 = 10 * x + 2) : x = 17 :=
sorry

end find_divisor_l7_7490


namespace max_odd_integers_l7_7052

theorem max_odd_integers (a b c d e : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) (h_even : a * b * c * d * e % 2 = 0) :
  ∃ m : ℕ, m = 4 ∧ ∀ o1 o2 o3 o4 : ℕ, (o1 % 2 = 1 ∧ o2 % 2 = 1 ∧ o3 % 2 = 1 ∧ o4 % 2 = 1) ∧
    (list.perm [a, b, c, d, e] [o1, o2, o3, o4, (2 * (a * b * c * d * e / 2 / o1 / o2 / o3 / o4))]) :=
sorry

end max_odd_integers_l7_7052


namespace six_x_plus_four_eq_twenty_two_l7_7732

theorem six_x_plus_four_eq_twenty_two (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 := 
by
  sorry

end six_x_plus_four_eq_twenty_two_l7_7732


namespace p_n_divisible_by_5_l7_7259

noncomputable def p_n (n : ℕ) : ℕ := 1^n + 2^n + 3^n + 4^n

theorem p_n_divisible_by_5 (n : ℕ) (h : n ≠ 0) : p_n n % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end p_n_divisible_by_5_l7_7259


namespace expected_number_of_groups_l7_7175

-- Define the conditions
variables (k m : ℕ) (h : 0 < k ∧ 0 < m)

-- Expected value of groups in the sequence
theorem expected_number_of_groups : 
  ∀ k m, (0 < k) → (0 < m) → 
  let total_groups := 1 + (2 * k * m) / (k + m) in total_groups = 1 + (2 * k * m) / (k + m) :=
by
  intros k m hk hm
  let total_groups := 1 + (2 * k * m) / (k + m)
  exact (rfl : total_groups = 1 + (2 * k * m) / (k + m))

end expected_number_of_groups_l7_7175


namespace angle_b_is_acute_l7_7334

-- Definitions for angles being right, acute, and sum of angles in a triangle
def is_right_angle (θ : ℝ) : Prop := θ = 90
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_sum_to_180 (α β γ : ℝ) : Prop := α + β + γ = 180

-- Main theorem statement
theorem angle_b_is_acute {α β γ : ℝ} (hC : is_right_angle γ) (hSum : angles_sum_to_180 α β γ) : is_acute_angle β :=
by
  sorry

end angle_b_is_acute_l7_7334


namespace peter_walks_more_time_l7_7146

-- Define the total distance Peter has to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def distance_walked : ℝ := 1.0

-- Define Peter's walking pace in minutes per mile
def walking_pace : ℝ := 20.0

-- Prove that Peter has to walk 30 more minutes to reach the grocery store
theorem peter_walks_more_time : walking_pace * (total_distance - distance_walked) = 30 :=
by
  sorry

end peter_walks_more_time_l7_7146


namespace smallest_rel_prime_210_l7_7700

theorem smallest_rel_prime_210 : ∃ x : ℕ, x > 1 ∧ gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 210 = 1 → x ≤ y := 
by
  sorry

end smallest_rel_prime_210_l7_7700


namespace ratio_of_geometric_sequence_sum_l7_7079

theorem ratio_of_geometric_sequence_sum (a : ℕ → ℕ) 
    (q : ℕ) (h_q_pos : 0 < q) (h_q_ne_one : q ≠ 1)
    (h_geo_seq : ∀ n : ℕ, a (n + 1) = a n * q)
    (h_arith_seq : 2 * a (3 + 2) = a 3 - a (3 + 1)) :
  (a 4 * (1 - q ^ 4) / (1 - q)) / (a 4 * (1 - q ^ 2) / (1 - q)) = 5 / 4 := 
  sorry

end ratio_of_geometric_sequence_sum_l7_7079


namespace smallest_rel_prime_210_l7_7699

theorem smallest_rel_prime_210 : ∃ x : ℕ, x > 1 ∧ gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 210 = 1 → x ≤ y := 
by
  sorry

end smallest_rel_prime_210_l7_7699


namespace find_a_l7_7715

theorem find_a (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) 
  (h_max : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a^(2*x) + 2 * a^x - 1 ≤ 7) 
  (h_eq : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^(2*x) + 2 * a^x - 1 = 7) : 
  a = 2 ∨ a = 1 / 2 :=
by
  sorry

end find_a_l7_7715


namespace find_a_in_triangle_l7_7748

theorem find_a_in_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : b^2 - c^2 + 2 * a = 0) 
  (h2 : Real.tan C / Real.tan B = 3) 
  : a = 4 :=
  sorry

end find_a_in_triangle_l7_7748


namespace length_of_pond_l7_7962

-- Define the problem conditions
variables (W L S : ℝ)
variables (h1 : L = 2 * W) (h2 : L = 24) 
variables (A_field A_pond : ℝ)
variables (h3 : A_pond = 1 / 8 * A_field)

-- State the theorem
theorem length_of_pond :
  A_field = L * W ∧ A_pond = S^2 ∧ A_pond = 1 / 8 * A_field ∧ L = 24 ∧ L = 2 * W → 
  S = 6 :=
by
  sorry

end length_of_pond_l7_7962


namespace tan_of_theta_minus_pi_div_4_l7_7578

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ := (Real.sin θ, 2)
noncomputable def vector_b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1)

def collinear (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem tan_of_theta_minus_pi_div_4 (θ : ℝ) 
  (h₁ : ∃ (k : ℝ), vector_a θ = (k * vector_b θ).fst ∧ 
                    (2 : ℝ) = k * (1 : ℝ)) : 
  Real.tan (θ - Real.pi / 4) = 1 / 3 := 
sorry

end tan_of_theta_minus_pi_div_4_l7_7578


namespace george_total_coins_l7_7072

-- We'll state the problem as proving the total number of coins George has.
variable (num_nickels num_dimes : ℕ)
variable (value_of_coins : ℝ := 2.60)
variable (value_of_nickels : ℝ := 0.05 * num_nickels)
variable (value_of_dimes : ℝ := 0.10 * num_dimes)

theorem george_total_coins :
  num_nickels = 4 → 
  value_of_coins = value_of_nickels + value_of_dimes → 
  num_nickels + num_dimes = 28 := 
by
  sorry

end george_total_coins_l7_7072


namespace satisfies_equation_l7_7156

noncomputable def y (b x : ℝ) : ℝ := (b + x) / (1 + b * x)

theorem satisfies_equation (b x : ℝ) :
  let y_val := y b x
  let y_prime := (1 - b^2) / (1 + b * x)^2
  y_val - x * y_prime = b * (1 + x^2 * y_prime) :=
by
  sorry

end satisfies_equation_l7_7156


namespace donuts_selection_l7_7629

def number_of_selections (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem donuts_selection : number_of_selections 6 4 = 84 := by
  sorry

end donuts_selection_l7_7629


namespace length_of_green_caterpillar_l7_7766

def length_of_orange_caterpillar : ℝ := 1.17
def difference_in_length_between_caterpillars : ℝ := 1.83

theorem length_of_green_caterpillar :
  (length_of_orange_caterpillar + difference_in_length_between_caterpillars) = 3.00 :=
by
  sorry

end length_of_green_caterpillar_l7_7766


namespace solve_for_z_l7_7857

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l7_7857


namespace original_number_of_coins_in_first_pile_l7_7514

noncomputable def originalCoinsInFirstPile (x y z : ℕ) : ℕ :=
  if h : (2 * (x - y) = 16) ∧ (2 * y - z = 16) ∧ (2 * z - (x + y) = 16) then x else 0

theorem original_number_of_coins_in_first_pile (x y z : ℕ) (h1 : 2 * (x - y) = 16) 
                                              (h2 : 2 * y - z = 16) 
                                              (h3 : 2 * z - (x + y) = 16) : x = 22 :=
by sorry

end original_number_of_coins_in_first_pile_l7_7514


namespace greatest_number_of_police_officers_needed_l7_7622

-- Define the conditions within Math City
def number_of_streets : ℕ := 10
def number_of_tunnels : ℕ := 2
def intersections_without_tunnels : ℕ := (number_of_streets * (number_of_streets - 1)) / 2
def intersections_bypassed_by_tunnels : ℕ := number_of_tunnels

-- Define the number of police officers required (which is the same as the number of intersections not bypassed)
def police_officers_needed : ℕ := intersections_without_tunnels - intersections_bypassed_by_tunnels

-- The main theorem: Given the conditions, the greatest number of police officers needed is 43.
theorem greatest_number_of_police_officers_needed : police_officers_needed = 43 := 
by {
  -- Proof would go here, but we'll use sorry to indicate it's not provided.
  sorry
}

end greatest_number_of_police_officers_needed_l7_7622


namespace box_contents_l7_7178

-- Definitions for the boxes and balls
inductive Ball
| Black | White | Green

-- Define the labels on each box
def label_box1 := "white"
def label_box2 := "black"
def label_box3 := "white or green"

-- Conditions based on the problem
def box1_label := label_box1
def box2_label := label_box2
def box3_label := label_box3

-- Statement of the problem
theorem box_contents (b1 b2 b3 : Ball) 
  (h1 : b1 ≠ Ball.White) 
  (h2 : b2 ≠ Ball.Black) 
  (h3 : b3 = Ball.Black) 
  (h4 : ∀ (x y z : Ball), x ≠ y ∧ y ≠ z ∧ z ≠ x → 
        (x = b1 ∨ y = b1 ∨ z = b1) ∧
        (x = b2 ∨ y = b2 ∨ z = b2) ∧
        (x = b3 ∨ y = b3 ∨ z = b3)) : 
  b1 = Ball.Green ∧ b2 = Ball.White ∧ b3 = Ball.Black :=
sorry

end box_contents_l7_7178


namespace translate_line_down_l7_7363

theorem translate_line_down (k : ℝ) (b : ℝ) : 
  (∀ x : ℝ, b = 0 → (y = k * x - 3) = (y = k * x - 3)) :=
by
  sorry

end translate_line_down_l7_7363


namespace find_r_and_k_l7_7506

-- Define the line equation
def line (x : ℝ) : ℝ := 5 * x - 7

-- Define the parameterization
def param (t r k : ℝ) : ℝ × ℝ := 
  (r + 3 * t, 2 + k * t)

-- Theorem stating that (r, k) = (9/5, 15) satisfies the given conditions
theorem find_r_and_k 
  (r k : ℝ)
  (H1 : param 0 r k = (r, 2))
  (H2 : line r = 2)
  (H3 : param 1 r k = (r + 3, 2 + k))
  (H4 : line (r + 3) = 2 + k)
  : (r, k) = (9/5, 15) :=
sorry

end find_r_and_k_l7_7506


namespace choir_arrangement_l7_7535

/-- There are 4 possible row-lengths for arranging 90 choir members such that each row has the same
number of individuals and the number of members per row is between 6 and 15. -/
theorem choir_arrangement (x : ℕ) (h : 6 ≤ x ∧ x ≤ 15 ∧ 90 % x = 0) :
  x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15 :=
by
  sorry

end choir_arrangement_l7_7535


namespace max_value_f_l7_7507

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + x + 1)

theorem max_value_f : ∀ x : ℝ, f x ≤ 4 / 3 :=
sorry

end max_value_f_l7_7507


namespace sofia_total_cost_l7_7779

def shirt_cost : ℕ := 7
def shoes_cost : ℕ := shirt_cost + 3
def two_shirts_cost : ℕ := 2 * shirt_cost
def total_clothes_cost : ℕ := two_shirts_cost + shoes_cost
def bag_cost : ℕ := total_clothes_cost / 2
def total_cost : ℕ := two_shirts_cost + shoes_cost + bag_cost

theorem sofia_total_cost : total_cost = 36 := by
  sorry

end sofia_total_cost_l7_7779


namespace suzanna_textbooks_page_total_l7_7339

theorem suzanna_textbooks_page_total :
  let H := 160
  let G := H + 70
  let M := (H + G) / 2
  let S := 2 * H
  let L := (H + G) - 30
  let E := M + L + 25
  H + G + M + S + L + E = 1845 := by
  sorry

end suzanna_textbooks_page_total_l7_7339


namespace find_number_l7_7901

theorem find_number (x : ℝ) : 
  (72 = 0.70 * x + 30) -> x = 60 :=
by
  sorry

end find_number_l7_7901


namespace constant_k_independent_of_b_l7_7739

noncomputable def algebraic_expression (a b k : ℝ) : ℝ :=
  a * b * (5 * k * a - 3 * b) - (k * a - b) * (3 * a * b - 4 * a^2)

theorem constant_k_independent_of_b (a : ℝ) : (algebraic_expression a b 2) = (algebraic_expression a 1 2) :=
by
  sorry

end constant_k_independent_of_b_l7_7739


namespace verify_squaring_method_l7_7495

theorem verify_squaring_method (x : ℝ) :
  ((x + 1)^3 - (x - 1)^3 - 2) / 6 = x^2 :=
by
  sorry

end verify_squaring_method_l7_7495


namespace fraction_books_left_l7_7165

theorem fraction_books_left (initial_books sold_books remaining_books : ℕ)
  (h1 : initial_books = 9900) (h2 : sold_books = 3300) (h3 : remaining_books = initial_books - sold_books) :
  (remaining_books : ℚ) / initial_books = 2 / 3 :=
by
  sorry

end fraction_books_left_l7_7165


namespace sum_of_reciprocals_of_squares_l7_7388

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 19) : 1 / (a * a : ℚ) + 1 / (b * b : ℚ) = 362 / 361 := 
by
  sorry

end sum_of_reciprocals_of_squares_l7_7388


namespace probability_red_given_black_l7_7368

noncomputable def urn_A := {white := 4, red := 2}
noncomputable def urn_B := {red := 3, black := 3}

-- Define the probabilities as required in the conditions
def prob_urn_A := 1 / 2
def prob_urn_B := 1 / 2

def draw_red_from_A := 2 / 6
def draw_black_from_B := 3 / 6
def draw_red_from_B := 3 / 6
def draw_black_from_B_after_red := 3 / 5
def draw_black_from_B_after_black := 2 / 5

def probability_first_red_second_black :=
  (prob_urn_A * draw_red_from_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_black)

def probability_second_black :=
  (prob_urn_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_black_from_B * prob_urn_B * draw_black_from_B_after_black)

theorem probability_red_given_black :
  probability_first_red_second_black / probability_second_black = 7 / 15 :=
sorry

end probability_red_given_black_l7_7368


namespace find_m_l7_7290

theorem find_m (a0 a1 a2 a3 a4 a5 a6 : ℝ) (m : ℝ)
  (h1 : (1 + m) * x ^ 6 = a0 + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4 + a5 * x ^ 5 + a6 * x ^ 6)
  (h2 : a1 - a2 + a3 - a4 + a5 - a6 = -63)
  (h3 : a0 = 1) :
  m = 3 ∨ m = -1 :=
by
  sorry

end find_m_l7_7290


namespace flashlight_lifetime_expectation_leq_two_l7_7314

noncomputable def min_lifetime_expectation (ξ η : ℝ) (E_ξ : ℝ) : Prop :=
  E_ξ = 2 → E(min ξ η) ≤ 2

-- Assume ξ and η are random variables and E denotes the expectation.
axiom E : (ℝ → ℝ) → ℝ

theorem flashlight_lifetime_expectation_leq_two (ξ η : ℝ) (E_ξ : ℝ) (hE_ξ : E_ξ = 2) : E(min ξ η) ≤ 2 :=
  by
    sorry

end flashlight_lifetime_expectation_leq_two_l7_7314


namespace fence_poles_count_l7_7039

def length_path : ℕ := 900
def length_bridge : ℕ := 42
def distance_between_poles : ℕ := 6

theorem fence_poles_count :
  2 * (length_path - length_bridge) / distance_between_poles = 286 :=
by
  sorry

end fence_poles_count_l7_7039


namespace min_value_of_M_l7_7450

noncomputable def min_val (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :=
  max (1/(a*c) + b) (max (1/a + b*c) (a/b + c))

theorem min_value_of_M (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (min_val a b c h1 h2 h3) >= 2 :=
sorry

end min_value_of_M_l7_7450


namespace Spot_dog_reachable_area_l7_7496

noncomputable def Spot_reachable_area (side_length tether_length : ℝ) : ℝ := 
  -- Note here we compute using the areas described in the problem
  6 * Real.pi * (tether_length^2) / 3 - Real.pi * (side_length^2)

theorem Spot_dog_reachable_area (side_length tether_length : ℝ)
  (H1 : side_length = 2) (H2 : tether_length = 3) :
    Spot_reachable_area side_length tether_length = (22 * Real.pi) / 3 := by
  sorry

end Spot_dog_reachable_area_l7_7496


namespace fran_speed_l7_7114

theorem fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
    (h_joann : joann_speed = 15) (h_joann_time : joann_time = 4) (h_fran_time : fran_time = 5) : 
    (joann_speed * joann_time) / fran_time = 12 :=
by
  rw [h_joann, h_joann_time, h_fran_time]
  norm_num
  sorry

end fran_speed_l7_7114


namespace total_hours_worked_l7_7484

variable (A B C D E T : ℝ)

theorem total_hours_worked (hA : A = 12)
  (hB : B = 1 / 3 * A)
  (hC : C = 2 * B)
  (hD : D = 1 / 2 * E)
  (hE : E = A + 3)
  (hT : T = A + B + C + D + E) : T = 46.5 :=
by
  sorry

end total_hours_worked_l7_7484


namespace problems_completed_l7_7093

theorem problems_completed (p t : ℕ) (hp : p > 10) (eqn : p * t = (2 * p - 2) * (t - 1)) :
  p * t = 48 := 
sorry

end problems_completed_l7_7093


namespace lychee_ratio_l7_7945

theorem lychee_ratio (total_lychees : ℕ) (sold_lychees : ℕ) (remaining_home : ℕ) (remaining_after_eat : ℕ) 
    (h1: total_lychees = 500) 
    (h2: sold_lychees = total_lychees / 2) 
    (h3: remaining_home = total_lychees - sold_lychees) 
    (h4: remaining_after_eat = 100)
    (h5: remaining_after_eat + (remaining_home - remaining_after_eat) = remaining_home) : 
    (remaining_home - remaining_after_eat) / remaining_home = 3 / 5 :=
by
    -- Proof is omitted
    sorry

end lychee_ratio_l7_7945


namespace smallest_relatively_prime_210_l7_7712

theorem smallest_relatively_prime_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ (∀ y : ℕ, y > 1 → y < x → Nat.gcd y 210 ≠ 1) :=
sorry

end smallest_relatively_prime_210_l7_7712


namespace larger_number_is_20_l7_7794

theorem larger_number_is_20 (a b : ℕ) (h1 : a + b = 9 * (a - b)) (h2 : a + b = 36) (h3 : a > b) : a = 20 :=
by
  sorry

end larger_number_is_20_l7_7794


namespace longest_side_length_l7_7963

-- Define the sides of the triangle
def side_a : ℕ := 9
def side_b (x : ℕ) : ℕ := 2 * x + 3
def side_c (x : ℕ) : ℕ := 3 * x - 2

-- Define the perimeter condition
def perimeter_condition (x : ℕ) : Prop := side_a + side_b x + side_c x = 45

-- Main theorem statement: Length of the longest side is 19
theorem longest_side_length (x : ℕ) (h : perimeter_condition x) : side_b x = 19 ∨ side_c x = 19 :=
sorry

end longest_side_length_l7_7963


namespace license_plate_count_l7_7730

/-- Number of vowels available for the license plate -/
def num_vowels := 6

/-- Number of consonants available for the license plate -/
def num_consonants := 20

/-- Number of possible digits for the license plate -/
def num_digits := 10

/-- Number of special characters available for the license plate -/
def num_special_chars := 2

/-- Calculate the total number of possible license plates -/
def total_license_plates : Nat :=
  num_vowels * num_consonants * num_digits * num_consonants * num_special_chars

/- Prove that the total number of possible license plates is 48000 -/
theorem license_plate_count : total_license_plates = 48000 :=
  by
    unfold total_license_plates
    sorry

end license_plate_count_l7_7730


namespace percent_decrease_first_year_l7_7211

theorem percent_decrease_first_year (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) 
  (h_second_year : 0.9 * (100 - x) = 54) : x = 40 :=
by sorry

end percent_decrease_first_year_l7_7211


namespace range_of_k_l7_7075

theorem range_of_k (x : ℝ) (h1 : 0 < x) (h2 : x < 2) (h3 : x / Real.exp x < 1 / (k + 2 * x - x^2)) :
    0 ≤ k ∧ k < Real.exp 1 - 1 :=
sorry

end range_of_k_l7_7075


namespace field_length_proof_l7_7505

noncomputable def field_width (w : ℝ) : Prop := w > 0

def pond_side_length : ℝ := 7

def pond_area : ℝ := pond_side_length * pond_side_length

def field_length (w l : ℝ) : Prop := l = 2 * w

def field_area (w l : ℝ) : ℝ := l * w

def pond_area_condition (w l : ℝ) : Prop :=
  pond_area = (1 / 8) * field_area w l

theorem field_length_proof {w l : ℝ} (hw : field_width w)
                           (hl : field_length w l)
                           (hpond : pond_area_condition w l) :
  l = 28 := by
  sorry

end field_length_proof_l7_7505


namespace smallest_rel_prime_210_l7_7704

theorem smallest_rel_prime_210 (x : ℕ) (hx : x > 1) (hrel_prime : Nat.gcd x 210 = 1) : x = 11 :=
sorry

end smallest_rel_prime_210_l7_7704


namespace find_people_who_own_only_cats_l7_7973

variable (C : ℕ)

theorem find_people_who_own_only_cats
  (ownsOnlyDogs : ℕ)
  (ownsCatsAndDogs : ℕ)
  (ownsCatsDogsSnakes : ℕ)
  (totalPetOwners : ℕ)
  (h1 : ownsOnlyDogs = 15)
  (h2 : ownsCatsAndDogs = 5)
  (h3 : ownsCatsDogsSnakes = 3)
  (h4 : totalPetOwners = 59) :
  C = 36 :=
by
  sorry

end find_people_who_own_only_cats_l7_7973


namespace find_prices_max_basketballs_l7_7109

-- Definition of given conditions
def conditions1 (x y : ℝ) : Prop := 
  (x - y = 50) ∧ (6 * x + 8 * y = 1700)

-- Definitions of questions:
-- Question 1: Find the price of one basketball and one soccer ball
theorem find_prices (x y : ℝ) (h: conditions1 x y) : x = 150 ∧ y = 100 := sorry

-- Definition of given conditions for Question 2
def conditions2 (x y : ℝ) (a : ℕ) : Prop :=
  (x = 150) ∧ (y = 100) ∧ 
  (0.9 * x * a + 0.85 * y * (10 - a) ≤ 1150)

-- Question 2: The school plans to purchase 10 items with given discounts
theorem max_basketballs (x y : ℝ) (a : ℕ) (h1: x = 150) (h2: y = 100) (h3: a ≤ 10) (h4: conditions2 x y a) : a ≤ 6 := sorry

end find_prices_max_basketballs_l7_7109


namespace ac_work_time_l7_7215

theorem ac_work_time (W : ℝ) (a_work_rate : ℝ) (b_work_rate : ℝ) (bc_work_rate : ℝ) (t : ℝ) : 
  (a_work_rate = W / 4) ∧ 
  (b_work_rate = W / 12) ∧ 
  (bc_work_rate = W / 3) → 
  t = 2 := 
by 
  sorry

end ac_work_time_l7_7215


namespace max_discount_rate_l7_7219

-- Define the cost price and selling price.
def cp : ℝ := 4
def sp : ℝ := 5

-- Define the minimum profit margin.
def min_profit_margin : ℝ := 0.4

-- Define the discount rate d.
def discount_rate (d : ℝ) : ℝ := sp * (1 - d / 100) - cp

-- The theorem to prove the maximum discount rate.
theorem max_discount_rate (d : ℝ) (H : discount_rate d ≥ min_profit_margin) : d ≤ 12 :=
sorry

end max_discount_rate_l7_7219


namespace abs_sum_zero_l7_7295

theorem abs_sum_zero (a b : ℝ) (h : |a - 5| + |b + 8| = 0) : a + b = -3 := 
sorry

end abs_sum_zero_l7_7295


namespace find_number_l7_7188

theorem find_number (n : ℕ) (h1 : 45 = 11 * n + 1) : n = 4 :=
  sorry

end find_number_l7_7188


namespace number_of_true_propositions_l7_7584

open Classical

-- Define each proposition as a term or lemma in Lean
def prop1 : Prop := ∀ x : ℝ, x^2 + 1 > 0
def prop2 : Prop := ∀ x : ℕ, x^4 ≥ 1
def prop3 : Prop := ∃ x : ℤ, x^3 < 1
def prop4 : Prop := ∀ x : ℚ, x^2 ≠ 2

-- The main theorem statement that the number of true propositions is 3 given the conditions
theorem number_of_true_propositions : (prop1 ∧ prop3 ∧ prop4) ∧ ¬prop2 → 3 = 3 := by
  sorry

end number_of_true_propositions_l7_7584


namespace min_value_l7_7446

theorem min_value (a b c x y z : ℝ) (h1 : a + b + c = 1) (h2 : x + y + z = 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  ∃ val : ℝ, val = -1 / 4 ∧ ∀ a b c x y z : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ x → 0 ≤ y → 0 ≤ z → a + b + c = 1 → x + y + z = 1 → (a - x^2) * (b - y^2) * (c - z^2) ≥ val :=
sorry

end min_value_l7_7446


namespace lunks_for_apples_l7_7459

theorem lunks_for_apples : 
  (∀ (a : ℕ) (b : ℕ) (k : ℕ), 3 * b * k = 5 * a → 15 * k = 9 * a ∧ 2 * a * 9 = 4 * b * 9 → 15 * 2 * a / 4 = 18) :=
by
  intro a b k h1 h2
  sorry

end lunks_for_apples_l7_7459


namespace commute_time_l7_7760

theorem commute_time (start_time : ℕ) (first_station_time : ℕ) (work_time : ℕ) 
  (h1 : start_time = 6 * 60) 
  (h2 : first_station_time = 40) 
  (h3 : work_time = 9 * 60) : 
  work_time - (start_time + first_station_time) = 140 :=
by
  sorry

end commute_time_l7_7760


namespace ratio_of_part_to_whole_l7_7628

theorem ratio_of_part_to_whole (N : ℝ) :
  (2/15) * N = 14 ∧ 0.40 * N = 168 → (14 / ((1/3) * (2/5) * N)) = 1 :=
by
  -- We assume the conditions given in the problem and need to prove the ratio
  intro h
  obtain ⟨h1, h2⟩ := h
  -- Establish equality through calculations
  sorry

end ratio_of_part_to_whole_l7_7628


namespace lcm_210_913_eq_2310_l7_7961

theorem lcm_210_913_eq_2310 : Nat.lcm 210 913 = 2310 := by
  sorry

end lcm_210_913_eq_2310_l7_7961


namespace emily_necklaces_l7_7692

theorem emily_necklaces (n beads_per_necklace total_beads : ℕ) (h1 : beads_per_necklace = 8) (h2 : total_beads = 16) : n = total_beads / beads_per_necklace → n = 2 :=
by sorry

end emily_necklaces_l7_7692


namespace three_gt_sqrt_seven_l7_7685

theorem three_gt_sqrt_seven : (3 : ℝ) > real.sqrt 7 := 
sorry

end three_gt_sqrt_seven_l7_7685


namespace crescents_area_eq_rectangle_area_l7_7162

noncomputable def rectangle_area (a b : ℝ) : ℝ := 4 * a * b

noncomputable def semicircle_area (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2

noncomputable def circumscribed_circle_area (a b : ℝ) : ℝ :=
  Real.pi * (a^2 + b^2)

noncomputable def combined_area (a b : ℝ) : ℝ :=
  rectangle_area a b + 2 * (semicircle_area a) + 2 * (semicircle_area b)

theorem crescents_area_eq_rectangle_area (a b : ℝ) : 
  combined_area a b - circumscribed_circle_area a b = rectangle_area a b :=
by
  unfold combined_area
  unfold circumscribed_circle_area
  unfold rectangle_area
  unfold semicircle_area
  sorry

end crescents_area_eq_rectangle_area_l7_7162


namespace polar_to_cartesian_l7_7169

theorem polar_to_cartesian (p θ : ℝ) (x y : ℝ) (hp : p = 8 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 + y^2 = 8 * x := 
sorry

end polar_to_cartesian_l7_7169


namespace peter_walks_more_time_l7_7145

-- Define the total distance Peter has to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def distance_walked : ℝ := 1.0

-- Define Peter's walking pace in minutes per mile
def walking_pace : ℝ := 20.0

-- Prove that Peter has to walk 30 more minutes to reach the grocery store
theorem peter_walks_more_time : walking_pace * (total_distance - distance_walked) = 30 :=
by
  sorry

end peter_walks_more_time_l7_7145


namespace difference_square_consecutive_l7_7351

theorem difference_square_consecutive (x : ℕ) (h : x * (x + 1) = 812) : (x + 1)^2 - x = 813 :=
sorry

end difference_square_consecutive_l7_7351


namespace find_value_of_y_l7_7594

theorem find_value_of_y (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := 
by {
  sorry
}

end find_value_of_y_l7_7594


namespace train_speed_l7_7050

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 350) (h_time : time = 7) : 
  length / time = 50 :=
by
  rw [h_length, h_time]
  norm_num

end train_speed_l7_7050


namespace prop1_prop2_prop3_prop4_final_l7_7254

variables (a b c : ℝ) (h_a : a ≠ 0)

-- Proposition ①
theorem prop1 (h1 : a + b + c = 0) : b^2 - 4 * a * c ≥ 0 := 
sorry

-- Proposition ②
theorem prop2 (h2 : ∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) : 2 * a + c = 0 := 
sorry

-- Proposition ③
theorem prop3 (h3 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + c = 0 ∧ a * x2^2 + c = 0) : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
sorry

-- Proposition ④
theorem prop4 (h4 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ ∃! x : ℝ, a * x^2 + b * x + c = 0) : ¬ (∃ x : ℝ, a * x^2 + b * x + c = 1 ∧ a * x^2 + b * x + 1 = 0) :=
sorry

-- Collectively checking that ①, ②, and ③ are true, and ④ is false
theorem final (h1 : a + b + c = 0)
              (h2 : ∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)
              (h3 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + c = 0 ∧ a * x2^2 + c = 0)
              (h4 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ ∃! x : ℝ, a * x^2 + b * x + c = 0) : 
  (b^2 - 4 * a * c ≥ 0 ∧ 2 * a + c = 0 ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧ 
  ¬ (∃ x : ℝ, a * x^2 + b * x + c = 1 ∧ a * x^2 + b * x + 1 = 0)) :=
sorry

end prop1_prop2_prop3_prop4_final_l7_7254


namespace f_2015_l7_7264

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 2) = -f x

axiom f_interval : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2 ^ x

theorem f_2015 : f 2015 = 1 / 2 :=
sorry

end f_2015_l7_7264


namespace soldier_score_9_points_l7_7566

-- Define the conditions and expected result in Lean 4
theorem soldier_score_9_points (shots : List ℕ) :
  shots.length = 10 ∧
  (∀ shot ∈ shots, shot = 7 ∨ shot = 8 ∨ shot = 9 ∨ shot = 10) ∧
  shots.count 10 = 4 ∧
  shots.sum = 90 →
  shots.count 9 = 3 :=
by 
  sorry

end soldier_score_9_points_l7_7566


namespace height_of_smaller_cone_l7_7537

theorem height_of_smaller_cone (h_frustum : ℝ) (area_lower_base area_upper_base : ℝ) 
  (h_frustum_eq : h_frustum = 18) 
  (area_lower_base_eq : area_lower_base = 144 * Real.pi) 
  (area_upper_base_eq : area_upper_base = 16 * Real.pi) : 
  ∃ (x : ℝ), x = 9 :=
by
  -- Definitions and assumptions go here
  sorry

end height_of_smaller_cone_l7_7537


namespace max_discount_rate_l7_7225

theorem max_discount_rate 
  (cost_price : ℝ) (selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 → selling_price = 5 → min_profit_margin = 0.1 →
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 8.8 ∧ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_rate_l7_7225


namespace minimum_difference_l7_7909

def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem minimum_difference (x y z : ℤ) 
  (hx : even x) (hy : odd y) (hz : odd z)
  (hxy : x < y) (hyz : y < z) (hzx : z - x = 9) : y - x = 1 := 
sorry

end minimum_difference_l7_7909


namespace partitions_of_6_into_4_indistinguishable_boxes_l7_7288

theorem partitions_of_6_into_4_indistinguishable_boxes : 
  ∃ (X : Finset (Multiset ℕ)), X.card = 9 ∧ 
  ∀ p ∈ X, p.sum = 6 ∧ p.card ≤ 4 := 
sorry

end partitions_of_6_into_4_indistinguishable_boxes_l7_7288


namespace arcsin_double_angle_identity_l7_7929

open Real

theorem arcsin_double_angle_identity (x θ : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (h₃ : arcsin x = θ) (h₄ : -π / 2 ≤ θ) (h₅ : θ ≤ -π / 4) :
    arcsin (2 * x * sqrt (1 - x^2)) = -(π + 2 * θ) := by
  sorry

end arcsin_double_angle_identity_l7_7929


namespace smallest_rel_prime_210_l7_7701

theorem smallest_rel_prime_210 : ∃ x : ℕ, x > 1 ∧ gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 210 = 1 → x ≤ y := 
by
  sorry

end smallest_rel_prime_210_l7_7701


namespace first_ball_red_given_second_black_l7_7371

open ProbabilityTheory

noncomputable def urn_A : Finset (Finset ℕ) := { {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 1, 2}, ... }
noncomputable def urn_B : Finset (Finset ℕ) := { {1, 1, 1, 2, 2, 2}, {1, 1, 2, 2, 2, 2}, ... }

noncomputable def prob_draw_red : ℕ := 7 / 15

theorem first_ball_red_given_second_black :
  (∑ A_Burn_selection in ({0, 1} : Finset ℕ), 
     ((∑ ball_draw from A_Burn_selection,
           if A_Burn_selection = 0 then (∑ red in urn_A, if red = 1 then 1 else 0) / 6 / 2
           else (∑ red in urn_B, if red = 1 then 1 else 0) / 6 / 2) *
     ((∑ second_urn_selection in ({0, 1} : Finset ℕ),
           if second_urn_selection = 0 and A_Burn_selection = 0 then 
              ∑ black in urn_A, if black = 1 then 1 else 0 / 6 / 2 
           else 
              ∑ black in urn_B, if black = 1 then 1 else 0 / 6 / 2))) = 7 / 15 :=
sorry

end first_ball_red_given_second_black_l7_7371


namespace distinct_prime_factors_330_l7_7275

def num_prime_factors (n : ℕ) : ℕ :=
  if n = 330 then 4 else 0

theorem distinct_prime_factors_330 : num_prime_factors 330 = 4 :=
sorry

end distinct_prime_factors_330_l7_7275


namespace texts_sent_total_l7_7624

def texts_sent_on_monday_to_allison_and_brittney : Nat := 5 + 5
def texts_sent_on_tuesday_to_allison_and_brittney : Nat := 15 + 15

def total_texts_sent (texts_monday : Nat) (texts_tuesday : Nat) : Nat := texts_monday + texts_tuesday

theorem texts_sent_total :
  total_texts_sent texts_sent_on_monday_to_allison_and_brittney texts_sent_on_tuesday_to_allison_and_brittney = 40 :=
by
  sorry

end texts_sent_total_l7_7624


namespace largest_n_unique_k_l7_7520

theorem largest_n_unique_k : ∃ n : ℕ, (∀ k : ℤ, (8 / 15 : ℚ) < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < (7 / 13 : ℚ) → k = unique_k) ∧ n = 112 :=
sorry

end largest_n_unique_k_l7_7520


namespace abs_diff_60th_terms_arithmetic_sequences_l7_7518

theorem abs_diff_60th_terms_arithmetic_sequences :
  let C : (ℕ → ℤ) := λ n => 25 + 15 * (n - 1)
  let D : (ℕ → ℤ) := λ n => 40 - 15 * (n - 1)
  |C 60 - D 60| = 1755 :=
by
  sorry

end abs_diff_60th_terms_arithmetic_sequences_l7_7518


namespace g_h_of_2_eq_2340_l7_7097

def g (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem g_h_of_2_eq_2340 : g (h 2) = 2340 := 
  sorry

end g_h_of_2_eq_2340_l7_7097


namespace mary_change_in_dollars_l7_7621

theorem mary_change_in_dollars :
  let cost_berries_euros := 7.94
  let cost_peaches_dollars := 6.83
  let exchange_rate := 1.2
  let money_handed_euros := 20
  let money_handed_dollars := 10
  let cost_berries_dollars := cost_berries_euros * exchange_rate
  let total_cost_dollars := cost_berries_dollars + cost_peaches_dollars
  let total_handed_dollars := (money_handed_euros * exchange_rate) + money_handed_dollars
  total_handed_dollars - total_cost_dollars = 17.642 :=
by
  intros
  sorry

end mary_change_in_dollars_l7_7621


namespace solve_for_b_l7_7502

theorem solve_for_b (b : ℝ) : 
  (∀ x y, 3 * y - 2 * x + 6 = 0 ↔ y = (2 / 3) * x - 2) → 
  (∀ x y, 4 * y + b * x + 3 = 0 ↔ y = -(b / 4) * x - 3 / 4) → 
  (∀ m1 m2, (m1 = (2 / 3)) → (m2 = -(b / 4)) → m1 * m2 = -1) → 
  b = 6 :=
sorry

end solve_for_b_l7_7502


namespace max_discount_rate_l7_7226

theorem max_discount_rate 
  (cost_price : ℝ) (selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 → selling_price = 5 → min_profit_margin = 0.1 →
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 8.8 ∧ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_rate_l7_7226


namespace find_value_of_y_l7_7595

theorem find_value_of_y (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := 
by {
  sorry
}

end find_value_of_y_l7_7595


namespace decrease_percent_in_revenue_l7_7530

theorem decrease_percent_in_revenue
  (T C : ℝ)
  (h_pos_T : 0 < T)
  (h_pos_C : 0 < C)
  (h_new_tax : T_new = 0.80 * T)
  (h_new_consumption : C_new = 1.20 * C) :
  let original_revenue := T * C
  let new_revenue := 0.80 * T * 1.20 * C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 4 := by
sorry

end decrease_percent_in_revenue_l7_7530


namespace num_values_of_a_l7_7092

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {1, a^2 - 2 * a}

theorem num_values_of_a : ∃v : Finset ℝ, (∀ a ∈ v, B a ⊆ A) ∧ v.card = 3 :=
by
  sorry

end num_values_of_a_l7_7092


namespace negation_example_l7_7348

theorem negation_example :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0 :=
by
  sorry

end negation_example_l7_7348


namespace percentage_failing_both_l7_7107

-- Define the conditions as constants
def percentage_failing_hindi : ℝ := 0.25
def percentage_failing_english : ℝ := 0.48
def percentage_passing_both : ℝ := 0.54

-- Define the percentage of students who failed in at least one subject
def percentage_failing_at_least_one : ℝ := 1 - percentage_passing_both

-- The main theorem statement we want to prove
theorem percentage_failing_both :
  percentage_failing_at_least_one = percentage_failing_hindi + percentage_failing_english - 0.27 := by
sorry

end percentage_failing_both_l7_7107


namespace max_distance_equals_2_sqrt_5_l7_7790

noncomputable def max_distance_from_point_to_line : Real :=
  let P : Real × Real := (2, -1)
  let Q : Real × Real := (-2, 1)
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_distance_equals_2_sqrt_5 : max_distance_from_point_to_line = 2 * Real.sqrt 5 := by
  sorry

end max_distance_equals_2_sqrt_5_l7_7790


namespace max_discount_rate_l7_7230

-- Define the conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1
def min_profit : ℝ := cost_price * min_profit_margin

-- Define the discount rate
def discount_rate (x : ℝ) : ℝ :=
  selling_price * (1 - x / 100)

-- Define the profit after discount
def profit_after_discount (x : ℝ) : ℝ :=
  discount_rate x - cost_price

-- The statement we want to prove
theorem max_discount_rate : ∃ x : ℝ, x = 8.8 ∧ profit_after_discount x ≥ min_profit := 
by
  sorry

end max_discount_rate_l7_7230


namespace calculation_result_l7_7532

theorem calculation_result:
  5 * 301 + 4 * 301 + 3 * 301 + 300 = 3912 :=
by
  sorry

end calculation_result_l7_7532


namespace intersection_complement_N_l7_7453

def is_universal_set (R : Set ℝ) : Prop := ∀ x : ℝ, x ∈ R

def is_complement (U S C : Set ℝ) : Prop := 
  ∀ x : ℝ, x ∈ C ↔ x ∈ U ∧ x ∉ S

theorem intersection_complement_N 
  (U M N C : Set ℝ)
  (h_universal : is_universal_set U)
  (hM : M = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (hN : N = {x : ℝ | x < 1})
  (h_compl : is_complement U M C) :
  (C ∩ N) = {x : ℝ | x < -2} := 
by 
  sorry

end intersection_complement_N_l7_7453


namespace total_cubes_in_stack_l7_7720

theorem total_cubes_in_stack :
  let bottom_layer := 4
  let middle_layer := 2
  let top_layer := 1
  bottom_layer + middle_layer + top_layer = 7 :=
by
  sorry

end total_cubes_in_stack_l7_7720


namespace total_albums_l7_7940

-- Definitions based on given conditions
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- The final statement to be proved
theorem total_albums : adele_albums + bridget_albums + katrina_albums + miriam_albums = 585 :=
by
  sorry

end total_albums_l7_7940


namespace three_gt_sqrt_seven_l7_7681

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end three_gt_sqrt_seven_l7_7681


namespace max_area_circle_eq_l7_7639

theorem max_area_circle_eq (m : ℝ) :
  (x y : ℝ) → (x - 1) ^ 2 + (y + m) ^ 2 = -(m - 3) ^ 2 + 1 → 
  (∃ (r : ℝ), (r = (1 : ℝ)) ∧ (m = 3) ∧ ((x - 1) ^ 2 + (y + 3) ^ 2 = 1)) :=
by
  sorry

end max_area_circle_eq_l7_7639


namespace tangent_line_at_1_f_geq_x_minus_1_min_value_a_l7_7586

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- 1. Proof that the equation of the tangent line at the point (1, f(1)) is y = x - 1
theorem tangent_line_at_1 :
  ∃ k b, (k = 1 ∧ b = -1 ∧ (∀ x, (f x - k * x - b) = 0)) :=
sorry

-- 2. Proof that f(x) ≥ x - 1 for all x in (0, +∞)
theorem f_geq_x_minus_1 :
  ∀ x, 0 < x → f x ≥ x - 1 :=
sorry

-- 3. Proof that the minimum value of a such that f(x) ≥ ax² + 2/a for all x in (0, +∞) is -e³
theorem min_value_a :
  ∃ a, (∀ x, 0 < x → f x ≥ a * x^2 + 2 / a) ∧ (a = -Real.exp 3) :=
sorry

end tangent_line_at_1_f_geq_x_minus_1_min_value_a_l7_7586


namespace smallest_rel_prime_210_l7_7703

theorem smallest_rel_prime_210 (x : ℕ) (hx : x > 1) (hrel_prime : Nat.gcd x 210 = 1) : x = 11 :=
sorry

end smallest_rel_prime_210_l7_7703


namespace sum_of_smallest_two_consecutive_numbers_l7_7643

theorem sum_of_smallest_two_consecutive_numbers (n : ℕ) (h : n * (n + 1) * (n + 2) = 210) : n + (n + 1) = 11 :=
sorry

end sum_of_smallest_two_consecutive_numbers_l7_7643


namespace closest_integer_power_of_eight_l7_7497

theorem closest_integer_power_of_eight : 
  ∃ n : ℤ, 2^(3 * n / 5) ≈ 100 ∧ n = 11 :=
by
  sorry

end closest_integer_power_of_eight_l7_7497


namespace marching_band_max_l7_7344

-- Define the conditions
variables (m k n : ℕ)

-- Lean statement of the problem
theorem marching_band_max (H1 : m = k^2 + 9) (H2 : m = n * (n + 5)) : m = 234 :=
sorry

end marching_band_max_l7_7344


namespace instantaneous_velocity_at_3_l7_7393

-- Definitions based on the conditions.
def displacement (t : ℝ) := 2 * t ^ 3

-- The statement to prove.
theorem instantaneous_velocity_at_3 : (deriv displacement 3) = 54 := by
  sorry

end instantaneous_velocity_at_3_l7_7393


namespace min_amount_for_free_shipping_l7_7687

def book1 : ℝ := 13.00
def book2 : ℝ := 15.00
def book3 : ℝ := 10.00
def book4 : ℝ := 10.00
def discount_rate : ℝ := 0.25
def shipping_threshold : ℝ := 9.00

def total_cost_before_discount : ℝ := book1 + book2 + book3 + book4
def discount_amount : ℝ := book1 * discount_rate + book2 * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

theorem min_amount_for_free_shipping : total_cost_after_discount + shipping_threshold = 50.00 :=
by
  sorry

end min_amount_for_free_shipping_l7_7687


namespace dog_food_consumption_per_meal_l7_7556

theorem dog_food_consumption_per_meal
  (dogs : ℕ) (meals_per_day : ℕ) (total_food_kg : ℕ) (days : ℕ)
  (h_dogs : dogs = 4) (h_meals_per_day : meals_per_day = 2)
  (h_total_food_kg : total_food_kg = 100) (h_days : days = 50) :
  (total_food_kg * 1000 / days / meals_per_day / dogs) = 250 :=
by
  sorry

end dog_food_consumption_per_meal_l7_7556


namespace value_at_2007_l7_7265

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom symmetric_property (x : ℝ) : f (2 + x) = f (2 - x)
axiom specific_value : f (-3) = -2

theorem value_at_2007 : f 2007 = -2 :=
sorry

end value_at_2007_l7_7265


namespace expected_lifetime_flashlight_l7_7324

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l7_7324


namespace lcm_48_75_l7_7570

theorem lcm_48_75 : Nat.lcm 48 75 = 1200 := by
  sorry

end lcm_48_75_l7_7570


namespace score_of_tenth_game_must_be_at_least_l7_7471

variable (score_5 average_9 average_10 score_10 : ℤ)
variable (H1 : average_9 > score_5 / 5)
variable (H2 : average_10 > 18)
variable (score_6 score_7 score_8 score_9 : ℤ)
variable (H3 : score_6 = 23)
variable (H4 : score_7 = 14)
variable (H5 : score_8 = 11)
variable (H6 : score_9 = 20)
variable (H7 : average_9 = (score_5 + score_6 + score_7 + score_8 + score_9) / 9)
variable (H8 : average_10 = (score_5 + score_6 + score_7 + score_8 + score_9 + score_10) / 10)

theorem score_of_tenth_game_must_be_at_least :
  score_10 ≥ 29 :=
by
  sorry

end score_of_tenth_game_must_be_at_least_l7_7471


namespace solve_z_l7_7871

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l7_7871


namespace action_figure_total_l7_7113

variable (initial_figures : ℕ) (added_figures : ℕ)

theorem action_figure_total (h₁ : initial_figures = 8) (h₂ : added_figures = 2) : (initial_figures + added_figures) = 10 := by
  sorry

end action_figure_total_l7_7113


namespace expected_min_leq_2_l7_7327

open ProbabilityTheory

variables (ξ η : ℝ → ℝ) -- ξ and η are random variables

-- Condition: expected value of ξ is 2
axiom E_ξ_eq_2 : ℝ
axiom E_ξ_is_2 : (∫ x in ⊤, ξ x) = 2

-- Goal: expected value of min(ξ, η) ≤ 2
theorem expected_min_leq_2 (h : ∀ x, min (ξ x) (η x) ≤ ξ x) : 
  (∫ x in ⊤, min (ξ x) (η x)) ≤ 2 := by
  -- use the provided axioms and conditions here
  sorry

end expected_min_leq_2_l7_7327


namespace find_first_number_l7_7991

theorem find_first_number : ∃ x : ℕ, x + 7314 = 3362 + 13500 ∧ x = 9548 :=
by
  -- This is where the proof would go
  sorry

end find_first_number_l7_7991


namespace trigonometric_identity_l7_7853

noncomputable def special_operation (a b : ℝ) : ℝ := a^2 - a * b - b^2

theorem trigonometric_identity :
  special_operation (Real.sin (Real.pi / 12)) (Real.cos (Real.pi / 12))
  = - (1 + 2 * Real.sqrt 3) / 4 :=
by
  sorry

end trigonometric_identity_l7_7853


namespace peter_remaining_walk_time_l7_7148

-- Define the parameters and conditions
def total_distance : ℝ := 2.5
def time_per_mile : ℝ := 20
def distance_walked : ℝ := 1

-- Define the remaining distance
def remaining_distance : ℝ := total_distance - distance_walked

-- Define the remaining time Peter needs to walk
def remaining_time_to_walk (d : ℝ) (t : ℝ) : ℝ := d * t

-- State the problem we want to prove
theorem peter_remaining_walk_time :
  remaining_time_to_walk remaining_distance time_per_mile = 30 :=
by
  -- Placeholder for the proof
  sorry

end peter_remaining_walk_time_l7_7148


namespace commute_time_l7_7761

theorem commute_time (start_time : ℕ) (first_station_time : ℕ) (work_time : ℕ) 
  (h1 : start_time = 6 * 60) 
  (h2 : first_station_time = 40) 
  (h3 : work_time = 9 * 60) : 
  work_time - (start_time + first_station_time) = 140 :=
by
  sorry

end commute_time_l7_7761


namespace systemOfEquationsUniqueSolution_l7_7359

def largeBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  5 * x + y = 3

def smallBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  x + 5 * y = 2

theorem systemOfEquationsUniqueSolution (x y : ℝ) :
  (largeBarrelHolds x y) ∧ (smallBarrelHolds x y) ↔ 
  (5 * x + y = 3 ∧ x + 5 * y = 2) :=
by
  sorry

end systemOfEquationsUniqueSolution_l7_7359


namespace Alex_sandwich_count_l7_7407

theorem Alex_sandwich_count :
  let meats := 10
  let cheeses := 9
  let sandwiches := meats * (cheeses.choose 2)
  sandwiches = 360 :=
by
  -- Here start your proof
  sorry

end Alex_sandwich_count_l7_7407


namespace heal_time_l7_7917

theorem heal_time (x : ℝ) (hx_pos : 0 < x) (h_total : 2.5 * x = 10) : x = 4 := 
by {
  -- Lean proof will be here
  sorry
}

end heal_time_l7_7917


namespace tina_brownies_per_meal_l7_7799

-- Define the given conditions
def total_brownies : ℕ := 24
def days : ℕ := 5
def meals_per_day : ℕ := 2
def brownies_by_husband_per_day : ℕ := 1
def total_brownies_shared_with_guests : ℕ := 4
def total_brownies_left : ℕ := 5

-- Conjecture: How many brownies did Tina have with each meal
theorem tina_brownies_per_meal :
  (total_brownies 
  - (brownies_by_husband_per_day * days) 
  - total_brownies_shared_with_guests 
  - total_brownies_left)
  / (days * meals_per_day) = 1 :=
by
  sorry

end tina_brownies_per_meal_l7_7799


namespace train_speed_correct_l7_7236

-- Define the length of the train
def train_length : ℝ := 200

-- Define the time taken to cross the telegraph post
def cross_time : ℝ := 8

-- Define the expected speed of the train
def expected_speed : ℝ := 25

-- Prove that the speed of the train is as expected
theorem train_speed_correct (length time : ℝ) (h_length : length = train_length) (h_time : time = cross_time) : 
  (length / time = expected_speed) :=
by
  rw [h_length, h_time]
  sorry

end train_speed_correct_l7_7236


namespace range_of_theta_l7_7899

theorem range_of_theta (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (h_ineq : 3 * (Real.sin θ ^ 5 + Real.cos (2 * θ) ^ 5) > 5 * (Real.sin θ ^ 3 + Real.cos (2 * θ) ^ 3)) :
    θ ∈ Set.Ico (7 * Real.pi / 6) (11 * Real.pi / 6) :=
sorry

end range_of_theta_l7_7899


namespace median_eq_altitude_eq_perp_bisector_eq_l7_7447

open Real

def point := ℝ × ℝ

def A : point := (1, 3)
def B : point := (3, 1)
def C : point := (-1, 0)

-- Median on BC
theorem median_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x, y) = ((1 + (-1))/2, (1 + 0)/2) → x = 1 :=
by
  intros x y h
  sorry

-- Altitude on BC
theorem altitude_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x - 1) / (y - 3) = -4 → 4*x + y - 7 = 0 :=
by
  intros x y h
  sorry

-- Perpendicular bisector of BC
theorem perp_bisector_eq : ∀ (x y : ℝ), (x = 1 ∧ y = 1/2) ∨ (x - 1) / (y - 1/2) = -4 
                          → 8*x + 2*y - 9 = 0 :=
by
  intros x y h
  sorry

end median_eq_altitude_eq_perp_bisector_eq_l7_7447


namespace min_m_value_inequality_x2y2z_l7_7580

theorem min_m_value (a b : ℝ) (h1 : a * b > 0) (h2 : a^2 * b = 2) : 
  ∃ (m : ℝ), m = a * b + a^2 ∧ m = 3 :=
sorry

theorem inequality_x2y2z 
  (t : ℝ) (ht : t = 3) (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x^2 + y^2 + z^2 = t / 3) : 
  |x + 2 * y + 2 * z| ≤ 3 :=
sorry

end min_m_value_inequality_x2y2z_l7_7580


namespace system1_solution_system2_solution_l7_7337

theorem system1_solution (x y : ℤ) : 
  (x - y = 3) ∧ (x = 3 * y - 1) → (x = 5) ∧ (y = 2) :=
by
  sorry

theorem system2_solution (x y : ℤ) : 
  (2 * x + 3 * y = -1) ∧ (3 * x - 2 * y = 18) → (x = 4) ∧ (y = -3) :=
by
  sorry

end system1_solution_system2_solution_l7_7337


namespace range_of_t_l7_7312

-- Define set A and set B as conditions
def setA := { x : ℝ | -3 < x ∧ x < 7 }
def setB (t : ℝ) := { x : ℝ | t + 1 < x ∧ x < 2 * t - 1 }

-- Lean statement to prove the range of t
theorem range_of_t (t : ℝ) : setB t ⊆ setA → t ≤ 4 :=
by
  -- sorry acts as a placeholder for the proof
  sorry

end range_of_t_l7_7312


namespace ones_digit_seven_consecutive_integers_l7_7509

theorem ones_digit_seven_consecutive_integers (k : ℕ) (hk : k % 5 = 1) :
  (k * (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6)) % 10 = 0 :=
by
  sorry

end ones_digit_seven_consecutive_integers_l7_7509


namespace track_length_l7_7975

theorem track_length (x : ℝ) (tom_dist1 jerry_dist1 : ℝ) (tom_dist2 jerry_dist2 : ℝ) (deg_gap : ℝ) :
  deg_gap = 120 ∧ 
  tom_dist1 = 120 ∧ 
  (tom_dist1 + jerry_dist1 = x * deg_gap / 360) ∧ 
  (jerry_dist1 + jerry_dist2 = x * deg_gap / 360 + 180) →
  x = 630 :=
by
  sorry

end track_length_l7_7975


namespace min_value_expression_l7_7887

open Real

/-- 
  Given that the function y = log_a(2x+3) - 4 passes through a fixed point P and the fixed point P lies on the line l: ax + by + 7 = 0,
  prove the minimum value of 1/(a+2) + 1/(4b) is 4/9, where a > 0, a ≠ 1, and b > 0.
-/
theorem min_value_expression (a b : ℝ) (h_a : 0 < a) (h_a_ne_1 : a ≠ 1) (h_b : 0 < b)
  (h_eqn : (a * -1 + b * -4 + 7 = 0) → (a + 2 + 4 * b = 9)):
  (1 / (a + 2) + 1 / (4 * b)) = 4 / 9 :=
by
  sorry

end min_value_expression_l7_7887


namespace quadratic_solution_l7_7494

theorem quadratic_solution (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
sorry

end quadratic_solution_l7_7494


namespace num_people_for_new_avg_l7_7343

def avg_salary := 430
def old_supervisor_salary := 870
def new_supervisor_salary := 870
def num_workers := 8
def total_people_before := num_workers + 1
def total_salary_before := total_people_before * avg_salary
def workers_salary := total_salary_before - old_supervisor_salary
def total_salary_after := workers_salary + new_supervisor_salary

theorem num_people_for_new_avg :
    ∃ (x : ℕ), x * avg_salary = total_salary_after ∧ x = 9 :=
by
  use 9
  field_simp
  sorry

end num_people_for_new_avg_l7_7343


namespace megawheel_seat_capacity_l7_7341

theorem megawheel_seat_capacity (seats people : ℕ) (h1 : seats = 15) (h2 : people = 75) : people / seats = 5 := by
  sorry

end megawheel_seat_capacity_l7_7341


namespace total_albums_l7_7941

-- Definitions based on given conditions
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- The final statement to be proved
theorem total_albums : adele_albums + bridget_albums + katrina_albums + miriam_albums = 585 :=
by
  sorry

end total_albums_l7_7941


namespace divisibility_of_special_number_l7_7611

theorem divisibility_of_special_number (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
    ∃ d : ℕ, 100100 * a + 10010 * b + 1001 * c = 11 * d := 
sorry

end divisibility_of_special_number_l7_7611


namespace proper_fraction_cubed_numerator_triples_denominator_add_three_l7_7434

theorem proper_fraction_cubed_numerator_triples_denominator_add_three
  (a b : ℕ)
  (h1 : a < b)
  (h2 : (a^3 : ℚ) / (b + 3) = 3 * (a : ℚ) / b) : 
  a = 2 ∧ b = 9 :=
by
  sorry

end proper_fraction_cubed_numerator_triples_denominator_add_three_l7_7434


namespace x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l7_7209

theorem x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0 :
  (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l7_7209


namespace tigers_losses_l7_7637

theorem tigers_losses (L T : ℕ) (h1 : 56 = 38 + L + T) (h2 : T = L / 2) : L = 12 :=
by sorry

end tigers_losses_l7_7637


namespace three_digit_number_l7_7890

theorem three_digit_number (a b c : ℕ) (h1 : a * (b + c) = 33) (h2 : b * (a + c) = 40) : 
  100 * a + 10 * b + c = 347 :=
by
  sorry

end three_digit_number_l7_7890


namespace Kolya_can_form_triangles_l7_7974

theorem Kolya_can_form_triangles :
  ∃ (K1a K1b K1c K3a K3b K3c V1 V2 V3 : ℝ), 
  (K1a + K1b + K1c = 1) ∧
  (K3a + K3b + K3c = 1) ∧
  (V1 + V2 + V3 = 1) ∧
  (K1a = 0.5) ∧ (K1b = 0.25) ∧ (K1c = 0.25) ∧
  (K3a = 0.5) ∧ (K3b = 0.25) ∧ (K3c = 0.25) ∧
  (∀ (V1 V2 V3 : ℝ), V1 + V2 + V3 = 1 → 
  (
    (K1a + V1 > K3b ∧ K1a + K3b > V1 ∧ V1 + K3b > K1a) ∧ 
    (K1b + V2 > K3a ∧ K1b + K3a > V2 ∧ V2 + K3a > K1b) ∧ 
    (K1c + V3 > K3c ∧ K1c + K3c > V3 ∧ V3 + K3c > K1c)
  )) :=
sorry

end Kolya_can_form_triangles_l7_7974


namespace coin_count_l7_7512

theorem coin_count (x y : ℕ) 
  (h1 : x + y = 12) 
  (h2 : 5 * x + 10 * y = 90) :
  x = 6 ∧ y = 6 := 
sorry

end coin_count_l7_7512


namespace solve_z_l7_7866

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l7_7866


namespace find_number_l7_7031

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end find_number_l7_7031


namespace second_person_days_l7_7390

theorem second_person_days (h1 : 2 * (1 : ℝ) / 8 = 1) 
                           (h2 : 1 / 24 + x / 24 = 1 / 8) : x = 1 / 12 :=
sorry

end second_person_days_l7_7390


namespace remainder_form_l7_7751

open Polynomial Int

-- Define the conditions
variable (f : Polynomial ℤ)
variable (h1 : ∀ n : ℤ, 3 ∣ eval n f)

-- Define the proof problem statement
theorem remainder_form (h1 : ∀ n : ℤ, 3 ∣ eval n f) :
  ∃ (M r : Polynomial ℤ), f = (X^3 - X) * M + C 3 * r :=
sorry

end remainder_form_l7_7751


namespace katherine_has_5_bananas_l7_7119

theorem katherine_has_5_bananas
  (apples : ℕ) (pears : ℕ) (bananas : ℕ) (total_fruits : ℕ)
  (h1 : apples = 4)
  (h2 : pears = 3 * apples)
  (h3 : total_fruits = apples + pears + bananas)
  (h4 : total_fruits = 21) :
  bananas = 5 :=
by
  sorry

end katherine_has_5_bananas_l7_7119


namespace collinear_points_sum_l7_7420

theorem collinear_points_sum (p q : ℝ) (h1 : 2 = p) (h2 : q = 4) : p + q = 6 :=
by 
  rw [h1, h2]
  sorry

end collinear_points_sum_l7_7420


namespace toys_produced_each_day_l7_7195

-- Given conditions
def total_weekly_production := 5500
def days_worked_per_week := 4

-- Define daily production calculation
def daily_production := total_weekly_production / days_worked_per_week

-- Proof that daily production is 1375 toys
theorem toys_produced_each_day :
  daily_production = 1375 := by
  sorry

end toys_produced_each_day_l7_7195


namespace range_of_m_l7_7077

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x-1)^2 < m^2 → |1 - (x-1)/3| < 2) → (abs m ≤ 3) :=
by
  sorry

end range_of_m_l7_7077


namespace derivative_at_x_equals_1_l7_7010

variable (x : ℝ)
def y : ℝ := (x + 1) * (x - 1)

theorem derivative_at_x_equals_1 : deriv y 1 = 2 :=
by
  sorry

end derivative_at_x_equals_1_l7_7010


namespace length_of_second_train_l7_7023

/-- 
  Given:
  * Speed of train 1 is 60 km/hr.
  * Speed of train 2 is 40 km/hr.
  * Length of train 1 is 500 meters.
  * Time to cross each other is 44.99640028797697 seconds.

  Then the length of train 2 is 750 meters.
-/
theorem length_of_second_train (v1 v2 t : ℝ) (d1 L : ℝ) : 
  v1 = 60 ∧
  v2 = 40 ∧
  t = 44.99640028797697 ∧
  d1 = 500 ∧
  L = ((v1 + v2) * (1000 / 3600) * t - d1) →
  L = 750 :=
by sorry

end length_of_second_train_l7_7023


namespace problem_solution_l7_7838

theorem problem_solution (y : Fin 8 → ℝ)
  (h1 : y 0 + 4 * y 1 + 9 * y 2 + 16 * y 3 + 25 * y 4 + 36 * y 5 + 49 * y 6 + 64 * y 7 = 2)
  (h2 : 4 * y 0 + 9 * y 1 + 16 * y 2 + 25 * y 3 + 36 * y 4 + 49 * y 5 + 64 * y 6 + 81 * y 7 = 15)
  (h3 : 9 * y 0 + 16 * y 1 + 25 * y 2 + 36 * y 3 + 49 * y 4 + 64 * y 5 + 81 * y 6 + 100 * y 7 = 156)
  (h4 : 16 * y 0 + 25 * y 1 + 36 * y 2 + 49 * y 3 + 64 * y 4 + 81 * y 5 + 100 * y 6 + 121 * y 7 = 1305) :
  25 * y 0 + 36 * y 1 + 49 * y 2 + 64 * y 3 + 81 * y 4 + 100 * y 5 + 121 * y 6 + 144 * y 7 = 4360 :=
sorry

end problem_solution_l7_7838


namespace probability_red_given_black_l7_7369

noncomputable def urn_A := {white := 4, red := 2}
noncomputable def urn_B := {red := 3, black := 3}

-- Define the probabilities as required in the conditions
def prob_urn_A := 1 / 2
def prob_urn_B := 1 / 2

def draw_red_from_A := 2 / 6
def draw_black_from_B := 3 / 6
def draw_red_from_B := 3 / 6
def draw_black_from_B_after_red := 3 / 5
def draw_black_from_B_after_black := 2 / 5

def probability_first_red_second_black :=
  (prob_urn_A * draw_red_from_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_black)

def probability_second_black :=
  (prob_urn_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_black_from_B * prob_urn_B * draw_black_from_B_after_black)

theorem probability_red_given_black :
  probability_first_red_second_black / probability_second_black = 7 / 15 :=
sorry

end probability_red_given_black_l7_7369


namespace expected_flashlight_lifetime_leq_two_l7_7317

theorem expected_flashlight_lifetime_leq_two
  (Ω : Type*) [MeasurableSpace Ω] [ProbabilitySpace Ω]
  (ξ η : Ω → ℝ)
  (h_min_leq_xi : ∀ ω, min (ξ ω) (η ω) ≤ ξ ω)
  (h_expectation_xi : expectation (ξ) = 2) :
  expectation (fun ω => min (ξ ω) (η ω)) ≤ 2 := 
sorry

end expected_flashlight_lifetime_leq_two_l7_7317


namespace candy_problem_l7_7047

theorem candy_problem (
  a : ℤ
) : (a % 10 = 6) →
    (a % 15 = 11) →
    (200 ≤ a ∧ a ≤ 250) →
    (a = 206 ∨ a = 236) :=
sorry

end candy_problem_l7_7047


namespace smallest_relatively_prime_210_l7_7711

theorem smallest_relatively_prime_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ (∀ y : ℕ, y > 1 → y < x → Nat.gcd y 210 ≠ 1) :=
sorry

end smallest_relatively_prime_210_l7_7711


namespace arithmetic_sequence_properties_sum_of_sequence_b_n_l7_7721

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h₁ : a 2 = 3) 
  (h₂ : S 5 + a 3 = 30) 
  (h₃ : ∀ n, S n = (n * (a 1 + (n-1) * ((a 2) - (a 1)))) / 2 
                     ∧ a n = a 1 + (n-1) * ((a 2) - (a 1))) : 
  (∀ n, a n = 2 * n - 1 ∧ S n = n^2) := 
sorry

theorem sum_of_sequence_b_n (b : ℕ → ℝ) 
  (T : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h₁ : ∀ n, b n = (a (n+1)) / (S n * S (n+1))) 
  (h₂ : ∀ n, a n = 2 * n - 1 ∧ S n = n^2) : 
  (∀ n, T n = (1 - 1 / (n+1)^2)) := 
sorry

end arithmetic_sequence_properties_sum_of_sequence_b_n_l7_7721


namespace triangle_division_congruent_l7_7600

theorem triangle_division_congruent 
  (A B C D K N P M : Point)
  (h_parallel : Parallel (Line.mk A B) (Line.mk C D))
  (h_acute_ABC : AcuteAngle (Angle.mk A B C))
  (h_acute_BAD : AcuteAngle (Angle.mk B A D))
  (h_K_intersection : Intersection K (Line.mk A C) (Line.mk B D))
  (h_NP_parallel : Parallel (Line.mk N P) (Line.mk A B))
  (h_NKP_collinear : Collinear N K P)
  (h_M_midpoint : Midpoint M A B) :
  ∃ X₁ X₂ X₃ X₄ Y₁ Y₂ Y₃ Y₄ : Triangle,
    DivideInto (Triangle.mk A B C) [X₁, X₂, X₃, X₄] ∧
    DivideInto (Triangle.mk A B D) [Y₁, Y₂, Y₃, Y₄] ∧
    ∀ i, i ∈ {0, 1, 2, 3} → Congruent (X i) (Y i) :=
sorry

end triangle_division_congruent_l7_7600


namespace broken_seashells_count_l7_7647

def total_seashells : ℕ := 7
def unbroken_seashells : ℕ := 3

theorem broken_seashells_count : (total_seashells - unbroken_seashells) = 4 := by
  sorry

end broken_seashells_count_l7_7647


namespace user_count_exceed_50000_l7_7934

noncomputable def A (t : ℝ) (k : ℝ) := 500 * Real.exp (k * t)

theorem user_count_exceed_50000 :
  (∃ k : ℝ, A 10 k = 2000) →
  (∀ t : ℝ, A t k > 50000) →
  ∃ t : ℝ, t >= 34 :=
by
  sorry

end user_count_exceed_50000_l7_7934


namespace toy_cost_l7_7773

-- Conditions
def initial_amount : ℕ := 3
def allowance : ℕ := 7
def total_amount : ℕ := initial_amount + allowance
def number_of_toys : ℕ := 2

-- Question and Proof
theorem toy_cost :
  total_amount / number_of_toys = 5 :=
by
  sorry

end toy_cost_l7_7773


namespace total_hours_watching_tv_and_playing_games_l7_7332

-- Defining the conditions provided in the problem
def hours_watching_tv_saturday : ℕ := 6
def hours_watching_tv_sunday : ℕ := 3
def hours_watching_tv_tuesday : ℕ := 2
def hours_watching_tv_thursday : ℕ := 4

def hours_playing_games_monday : ℕ := 3
def hours_playing_games_wednesday : ℕ := 5
def hours_playing_games_friday : ℕ := 1

-- The proof statement
theorem total_hours_watching_tv_and_playing_games :
  hours_watching_tv_saturday + hours_watching_tv_sunday + hours_watching_tv_tuesday + hours_watching_tv_thursday
  + hours_playing_games_monday + hours_playing_games_wednesday + hours_playing_games_friday = 24 := 
by
  sorry

end total_hours_watching_tv_and_playing_games_l7_7332


namespace max_discount_rate_l7_7224

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l7_7224


namespace solved_just_B_is_six_l7_7774

variables (a b c d e f g : ℕ)

-- Conditions given
axiom total_competitors : a + b + c + d + e + f + g = 25
axiom twice_as_many_solved_B : b + d = 2 * (c + d)
axiom only_A_one_more : a = 1 + (e + f + g)
axiom A_equals_B_plus_C : a = b + c

-- Prove that the number of competitors solving just problem B is 6.
theorem solved_just_B_is_six : b = 6 :=
by
  sorry

end solved_just_B_is_six_l7_7774


namespace number_of_students_l7_7007

theorem number_of_students (n : ℕ) (A : ℕ) 
  (h1 : A = 10 * n)
  (h2 : (A - 11 + 41) / n = 11) :
  n = 30 := 
sorry

end number_of_students_l7_7007


namespace amount_left_for_gas_and_maintenance_l7_7479

def monthly_income : ℤ := 3200
def rent : ℤ := 1250
def utilities : ℤ := 150
def retirement_savings : ℤ := 400
def groceries_eating_out : ℤ := 300
def insurance : ℤ := 200
def miscellaneous : ℤ := 200
def car_payment : ℤ := 350

def total_expenses : ℤ :=
  rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment

theorem amount_left_for_gas_and_maintenance : monthly_income - total_expenses = 350 :=
by
  -- Proof is omitted
  sorry

end amount_left_for_gas_and_maintenance_l7_7479


namespace find_the_number_l7_7677

theorem find_the_number (x : ℕ) (h : x * 9999 = 4691110842) : x = 469211 := by
    sorry

end find_the_number_l7_7677


namespace true_proposition_l7_7736

-- Define the propositions p and q
def p : Prop := 2 % 2 = 0
def q : Prop := 5 % 2 = 0

-- Define the problem statement
theorem true_proposition (hp : p) (hq : ¬ q) : p ∨ q :=
by
  sorry

end true_proposition_l7_7736


namespace increasing_intervals_decreasing_intervals_max_value_min_value_l7_7267

noncomputable def func (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

theorem increasing_intervals : 
  ∀ x ∈ (Set.Icc 0 (Real.pi / 8) ∪ Set.Icc (5 * Real.pi / 8) Real.pi), 
  0 < Real.cos (2 * x + Real.pi / 4) := 
sorry

theorem decreasing_intervals : 
  ∀ x ∈ Set.Icc (Real.pi / 8) (5 * Real.pi / 8), 
  Real.cos (2 * x + Real.pi / 4) < 0 := 
sorry

theorem max_value : func (Real.pi / 8) = 3 :=
sorry

theorem min_value : func (5 * Real.pi / 8) = -3 :=
sorry

end increasing_intervals_decreasing_intervals_max_value_min_value_l7_7267


namespace eq_zero_or_one_if_square_eq_self_l7_7291

theorem eq_zero_or_one_if_square_eq_self (a : ℝ) (h : a^2 = a) : a = 0 ∨ a = 1 :=
sorry

end eq_zero_or_one_if_square_eq_self_l7_7291


namespace slope_of_line_l7_7564

theorem slope_of_line (x y : ℝ) (h : x / 4 + y / 3 = 1) : ∀ m : ℝ, (y = m * x + 3) → m = -3/4 :=
by
  sorry

end slope_of_line_l7_7564


namespace correct_calculation_l7_7981

variable (n : ℕ)
variable (h1 : 63 + n = 70)

theorem correct_calculation : 36 * n = 252 :=
by
  -- Here we will need the Lean proof, which we skip using sorry
  sorry

end correct_calculation_l7_7981


namespace greater_num_792_l7_7193

theorem greater_num_792 (x y : ℕ) (h1 : x + y = 1443) (h2 : x - y = 141) : x = 792 :=
by
  sorry

end greater_num_792_l7_7193


namespace max_ab_bc_cd_l7_7129

-- Definitions of nonnegative numbers and their sum condition
variables (a b c d : ℕ) 
variables (h_sum : a + b + c + d = 120)

-- The goal to prove
theorem max_ab_bc_cd : ab + bc + cd <= 3600 :=
sorry

end max_ab_bc_cd_l7_7129


namespace relationship_between_p_and_q_l7_7257

variable {a b : ℝ}

theorem relationship_between_p_and_q 
  (h_a : a > 2) 
  (h_p : p = a + 1 / (a - 2)) 
  (h_q : q = -b^2 - 2 * b + 3) : 
  p ≥ q := 
sorry

end relationship_between_p_and_q_l7_7257


namespace num_ways_distribute_balls_l7_7279

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end num_ways_distribute_balls_l7_7279


namespace katie_added_new_songs_l7_7120

-- Definitions for the conditions
def initial_songs := 11
def deleted_songs := 7
def current_songs := 28

-- Definition of the expected answer
def new_songs_added := current_songs - (initial_songs - deleted_songs)

-- Statement of the problem in Lean
theorem katie_added_new_songs : new_songs_added = 24 :=
by
  sorry

end katie_added_new_songs_l7_7120


namespace remaining_walking_time_is_30_l7_7142

-- Define all the given conditions
def total_distance_to_store : ℝ := 2.5
def distance_already_walked : ℝ := 1.0
def time_per_mile : ℝ := 20.0

-- Define the target remaining walking time
def remaining_distance : ℝ := total_distance_to_store - distance_already_walked
def remaining_time : ℝ := remaining_distance * time_per_mile

-- Prove the remaining walking time is 30 minutes
theorem remaining_walking_time_is_30 : remaining_time = 30 :=
by
  -- Formal proof would go here using corresponding Lean tactics
  sorry

end remaining_walking_time_is_30_l7_7142


namespace smallest_coprime_to_210_l7_7708

theorem smallest_coprime_to_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 210 = 1 → y ≥ x :=
by
  sorry

end smallest_coprime_to_210_l7_7708


namespace integral_x2_plus_sin_l7_7557

theorem integral_x2_plus_sin (f : ℝ → ℝ) (a b : ℝ) :
  ∫ x in -1..1, (x^2 + sin x) = 2 / 3 :=
by
  have H₁ : ∫ x in -1..1, x^2 = 2 / 3, sorry
  have H₂ : ∫ x in -1..1, sin x = 0, sorry
  calc ∫ x in -1..1, (x^2 + sin x)
      = ∫ x in -1..1, x^2 + ∫ x in -1..1, sin x : by sorry
  ... = 2 / 3 + 0 : by sorry
  ... = 2 / 3 : by sorry

end integral_x2_plus_sin_l7_7557


namespace find_m_l7_7891

variables (a b : ℝ × ℝ) (m : ℝ)

def vectors := (a = (3, 4)) ∧ (b = (2, -1))

def perpendicular (a b : ℝ × ℝ) : Prop :=
a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (h1 : vectors a b) (h2 : perpendicular (a.1 + m * b.1, a.2 + m * b.2) (a.1 - b.1, a.2 - b.2)) :
  m = 23 / 3 :=
sorry

end find_m_l7_7891


namespace find_a_l7_7239

noncomputable def csc (x : ℝ) : ℝ := 1 / (Real.sin x)

theorem find_a (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : a * csc (b * (Real.pi / 6) + c) = 3) : a = 3 := 
sorry

end find_a_l7_7239


namespace tournament_committees_count_l7_7110

-- Definitions corresponding to the conditions
def num_teams : ℕ := 4
def team_size : ℕ := 8
def members_selected_by_winning_team : ℕ := 3
def members_selected_by_other_teams : ℕ := 2

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Counting the number of possible committees
def total_committees : ℕ :=
  let num_ways_winning_team := binom team_size members_selected_by_winning_team
  let num_ways_other_teams := binom team_size members_selected_by_other_teams
  num_teams * num_ways_winning_team * (num_ways_other_teams ^ (num_teams - 1))

-- The statement to be proved
theorem tournament_committees_count : total_committees = 4917248 := by
  sorry

end tournament_committees_count_l7_7110


namespace quadratic_solution_l7_7956

theorem quadratic_solution (x : ℝ) : x^2 - 5 * x - 6 = 0 ↔ (x = 6 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l7_7956


namespace calculate_total_cost_l7_7980

def cost_of_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def cost_of_non_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def total_cost (p_l1 p_l2 np_l1 np_l2 ppf np_pf : ℕ) : ℕ :=
  cost_of_parallel_sides p_l1 p_l2 ppf + cost_of_non_parallel_sides np_l1 np_l2 np_pf

theorem calculate_total_cost :
  total_cost 25 37 20 24 48 60 = 5616 :=
by
  -- Assuming the conditions are correctly applied, the statement aims to validate that the calculated
  -- sum of the costs for the specified fence sides equal Rs 5616.
  sorry

end calculate_total_cost_l7_7980


namespace fraction_combination_l7_7733

theorem fraction_combination (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 :=
by
  -- Proof steps will be inserted here (for now using sorry)
  sorry

end fraction_combination_l7_7733


namespace valid_paths_in_grid_l7_7623

theorem valid_paths_in_grid : 
  let total_paths := Nat.choose 15 4;
  let paths_through_EF := (Nat.choose 7 2) * (Nat.choose 7 2);
  let valid_paths := total_paths - 2 * paths_through_EF;
  grid_size == (11, 4) ∧
  blocked_segments == [((5, 2), (5, 3)), ((6, 2), (6, 3))] 
  → valid_paths = 483 :=
by
  sorry

end valid_paths_in_grid_l7_7623


namespace compute_xy_l7_7183

theorem compute_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^3 + y^3 = 351) : x * y = 14 :=
by
  sorry

end compute_xy_l7_7183


namespace solve_for_z_l7_7864

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l7_7864


namespace value_of_g_at_13_l7_7094

-- Define the function g
def g (n : ℕ) : ℕ := n^2 + n + 23

-- The theorem to prove
theorem value_of_g_at_13 : g 13 = 205 := by
  -- Rewrite using the definition of g
  unfold g
  -- Perform the arithmetic
  sorry

end value_of_g_at_13_l7_7094


namespace sqrt_conjecture_l7_7488

theorem sqrt_conjecture (n : ℕ) (h : n ≥ 1) : 
  (Real.sqrt (n + (1 / (n + 2)))) = ((n + 1) * Real.sqrt (1 / (n + 2))) :=
sorry

end sqrt_conjecture_l7_7488


namespace minimum_possible_sum_of_4x4x4_cube_l7_7034

theorem minimum_possible_sum_of_4x4x4_cube: 
  (∀ die: ℕ, (1 ≤ die) ∧ (die ≤ 6) ∧ (∃ opposite, die + opposite = 7)) → 
  (∃ sum, sum = 304) :=
by
  sorry

end minimum_possible_sum_of_4x4x4_cube_l7_7034


namespace rojas_speed_l7_7335

theorem rojas_speed (P R : ℝ) (h1 : P = 3) (h2 : 4 * (R + P) = 28) : R = 4 :=
by
  sorry

end rojas_speed_l7_7335


namespace total_people_at_fair_l7_7782

theorem total_people_at_fair (num_children : ℕ) (num_adults : ℕ) 
  (children_attended : num_children = 700) 
  (adults_attended : num_adults = 1500) : 
  num_children + num_adults = 2200 := by
  sorry

end total_people_at_fair_l7_7782


namespace relationship_between_A_and_B_l7_7575

theorem relationship_between_A_and_B (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let A := a^2
  let B := 2 * a - 1
  A > B :=
by
  let A := a^2
  let B := 2 * a - 1
  sorry

end relationship_between_A_and_B_l7_7575


namespace max_min_values_l7_7697

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

theorem max_min_values :
  let max_val := 2
  let min_val := -25
  ∃ x_max x_min, 
    0 ≤ x_max ∧ x_max ≤ 4 ∧ f x_max = max_val ∧ 
    0 ≤ x_min ∧ x_min ≤ 4 ∧ f x_min = min_val :=
sorry

end max_min_values_l7_7697


namespace calculate_stripes_l7_7137

theorem calculate_stripes :
  let olga_stripes_per_shoe := 3
  let rick_stripes_per_shoe := olga_stripes_per_shoe - 1
  let hortense_stripes_per_shoe := olga_stripes_per_shoe * 2
  let ethan_stripes_per_shoe := hortense_stripes_per_shoe + 2
  (olga_stripes_per_shoe * 2 + rick_stripes_per_shoe * 2 + hortense_stripes_per_shoe * 2 + ethan_stripes_per_shoe * 2) / 2 = 19 := 
by
  sorry

end calculate_stripes_l7_7137


namespace meaningful_fraction_l7_7966

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by {
  sorry -- Proof goes here
}

end meaningful_fraction_l7_7966


namespace full_price_tickets_count_l7_7529

def num_tickets_reduced := 5400
def total_tickets := 25200
def num_tickets_full := 5 * num_tickets_reduced

theorem full_price_tickets_count :
  num_tickets_reduced + num_tickets_full = total_tickets → num_tickets_full = 27000 :=
by
  sorry

end full_price_tickets_count_l7_7529


namespace total_boxes_sold_l7_7769

-- Define the variables for each day's sales
def friday_sales : ℕ := 30
def saturday_sales : ℕ := 2 * friday_sales
def sunday_sales : ℕ := saturday_sales - 15
def total_sales : ℕ := friday_sales + saturday_sales + sunday_sales

-- State the theorem to prove the total sales over three days
theorem total_boxes_sold : total_sales = 135 :=
by 
  -- Here we would normally put the proof steps, but since we're asked only for the statement,
  -- we skip the proof with sorry
  sorry

end total_boxes_sold_l7_7769


namespace texts_sent_total_l7_7625

def texts_sent_on_monday_to_allison_and_brittney : Nat := 5 + 5
def texts_sent_on_tuesday_to_allison_and_brittney : Nat := 15 + 15

def total_texts_sent (texts_monday : Nat) (texts_tuesday : Nat) : Nat := texts_monday + texts_tuesday

theorem texts_sent_total :
  total_texts_sent texts_sent_on_monday_to_allison_and_brittney texts_sent_on_tuesday_to_allison_and_brittney = 40 :=
by
  sorry

end texts_sent_total_l7_7625


namespace euclid_1976_part_a_problem_4_l7_7957

theorem euclid_1976_part_a_problem_4
  (p q y1 y2 : ℝ)
  (h1 : y1 = p * 1^2 + q * 1 + 5)
  (h2 : y2 = p * (-1)^2 + q * (-1) + 5)
  (h3 : y1 + y2 = 14) :
  p = 2 :=
by
  sorry

end euclid_1976_part_a_problem_4_l7_7957


namespace total_ticket_cost_l7_7003

theorem total_ticket_cost :
  ∀ (A : ℝ), 
  -- Conditions
  (6 : ℝ) * (5 : ℝ) + (2 : ℝ) * A = 50 :=
by
  sorry

end total_ticket_cost_l7_7003


namespace rain_at_house_l7_7191

/-- Define the amounts of rain on the three days Greg was camping. -/
def rain_day1 : ℕ := 3
def rain_day2 : ℕ := 6
def rain_day3 : ℕ := 5

/-- Define the total rain experienced by Greg while camping. -/
def total_rain_camping := rain_day1 + rain_day2 + rain_day3

/-- Define the difference in the rain experienced by Greg while camping and at his house. -/
def rain_difference : ℕ := 12

/-- Define the total amount of rain at Greg's house. -/
def total_rain_house := total_rain_camping + rain_difference

/-- Prove that the total rain at Greg's house is 26 mm. -/
theorem rain_at_house : total_rain_house = 26 := by
  /- We know that total_rain_camping = 14 mm and rain_difference = 12 mm -/
  /- Therefore, total_rain_house = 14 mm + 12 mm = 26 mm -/
  sorry

end rain_at_house_l7_7191


namespace probability_exactly_nine_matches_l7_7204

theorem probability_exactly_nine_matches (n : ℕ) (h : n = 10) : 
  (∃ p : ℕ, p = 9 ∧ probability_of_exact_matches n p = 0) :=
by {
  sorry
}

end probability_exactly_nine_matches_l7_7204


namespace nalani_fraction_sold_is_3_over_8_l7_7135

-- Definitions of conditions
def num_dogs : ℕ := 2
def puppies_per_dog : ℕ := 10
def total_amount_received : ℕ := 3000
def price_per_puppy : ℕ := 200

-- Calculation of total puppies and sold puppies
def total_puppies : ℕ := num_dogs * puppies_per_dog
def puppies_sold : ℕ := total_amount_received / price_per_puppy

-- Fraction of puppies sold
def fraction_sold : ℚ := puppies_sold / total_puppies

theorem nalani_fraction_sold_is_3_over_8 :
  fraction_sold = 3 / 8 :=
sorry

end nalani_fraction_sold_is_3_over_8_l7_7135


namespace total_candies_darrel_took_l7_7180

theorem total_candies_darrel_took (r b x : ℕ) (h1 : r = 3 * b)
  (h2 : r - x = 4 * (b - x))
  (h3 : r - x - 12 = 5 * (b - x - 12)) : 2 * x = 48 := sorry

end total_candies_darrel_took_l7_7180


namespace largest_number_of_pangs_largest_number_of_pangs_possible_l7_7672

theorem largest_number_of_pangs (x y z : ℕ) 
  (hx : x ≥ 2) 
  (hy : y ≥ 3) 
  (hcost : 3 * x + 4 * y + 9 * z = 100) : 
  z ≤ 9 :=
by sorry

theorem largest_number_of_pangs_possible (x y z : ℕ) 
  (hx : x = 2) 
  (hy : y = 3) 
  (hcost : 3 * x + 4 * y + 9 * z = 100) : 
  z = 9 :=
by sorry

end largest_number_of_pangs_largest_number_of_pangs_possible_l7_7672


namespace fraction_meaningful_l7_7968

-- Define the condition for the fraction being meaningful
def denominator_not_zero (x : ℝ) : Prop := x + 1 ≠ 0

-- Define the statement to be proved
theorem fraction_meaningful (x : ℝ) : denominator_not_zero x ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_l7_7968


namespace sum_of_areas_of_disks_l7_7844

theorem sum_of_areas_of_disks (r : ℝ) (a b c : ℕ) (h : a + b + c = 123) :
  ∃ (r : ℝ), (15 * Real.pi * r^2 = Real.pi * ((105 / 4) - 15 * Real.sqrt 3) ∧ r = 1 - (Real.sqrt 3) / 2) := 
by
  sorry

end sum_of_areas_of_disks_l7_7844


namespace find_Matrix_M_l7_7430

open Matrix

noncomputable def M : Matrix (Fin 3) (Fin 3) ℚ :=
  !![[-8, -3, 15],
     [3, -1, 5],
     [0, 0, 1]]

def matrix_A : Matrix (Fin 3) (Fin 3) ℚ :=
  !![[-2, 3, 0],
     [6, -8, 5],
     [0, 0, 1]]

def I : Matrix (Fin 3) (Fin 3) ℚ :=
  !![[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

theorem find_Matrix_M : M * matrix_A = I := 
  by
  sorry

end find_Matrix_M_l7_7430


namespace num_subsets_with_even_is_24_l7_7589

def A : Set ℕ := {1, 2, 3, 4, 5}
def odd_subsets_count : ℕ := 2^3

theorem num_subsets_with_even_is_24 : 
  let total_subsets := 2^5
  total_subsets - odd_subsets_count = 24 := by
  sorry

end num_subsets_with_even_is_24_l7_7589


namespace find_c_l7_7460

variable (a b c : ℕ)

theorem find_c (h1 : a = 9) (h2 : b = 2) (h3 : Odd c) (h4 : a + b > c) (h5 : a - b < c) (h6 : b + c > a) (h7 : b - c < a) : c = 9 :=
sorry

end find_c_l7_7460


namespace rogers_coaches_l7_7154

-- Define the structure for the problem conditions
structure snacks_problem :=
  (team_members : ℕ)
  (helpers : ℕ)
  (packs_purchased : ℕ)
  (pouches_per_pack : ℕ)

-- Create an instance of the problem with given conditions
def rogers_problem : snacks_problem :=
  { team_members := 13,
    helpers := 2,
    packs_purchased := 3,
    pouches_per_pack := 6 }

-- Define the theorem to state that given the conditions, the number of coaches is 3
theorem rogers_coaches (p : snacks_problem) : p.packs_purchased * p.pouches_per_pack - p.team_members - p.helpers = 3 :=
by
  sorry

end rogers_coaches_l7_7154


namespace log_base_2_y_l7_7903

theorem log_base_2_y (y : ℝ) (h : y = (Real.log 3 / Real.log 9) ^ Real.log 27 / Real.log 3) : 
  Real.log y = -3 :=
by
  sorry

end log_base_2_y_l7_7903


namespace moles_of_ca_oh_2_l7_7432

-- Define the chemical reaction
def ca_o := 1
def h_2_o := 1
def ca_oh_2 := ca_o + h_2_o

-- Prove the result of the reaction
theorem moles_of_ca_oh_2 :
  ca_oh_2 = 1 := by sorry

end moles_of_ca_oh_2_l7_7432


namespace geometric_seq_a9_l7_7604

theorem geometric_seq_a9 
  (a : ℕ → ℤ)  -- The sequence definition
  (h_geometric : ∀ n : ℕ, a (n+1) = a 1 * (a 2 ^ n) / a 1 ^ n)  -- Geometric sequence property
  (h_a1 : a 1 = 2)  -- Given a₁ = 2
  (h_a5 : a 5 = 18)  -- Given a₅ = 18
: a 9 = 162 := sorry

end geometric_seq_a9_l7_7604


namespace intersection_M_N_l7_7932

def M : Set ℝ := { x | x < 2017 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } := 
by 
  sorry

end intersection_M_N_l7_7932


namespace monotone_f_range_l7_7597

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_f_range (a : ℝ) :
  (∀ x : ℝ, (1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x) ≥ 0) ↔ (-1 / 3 ≤ a ∧ a ≤ 1 / 3) := 
sorry

end monotone_f_range_l7_7597


namespace find_p_l7_7602

/-- Given the points Q(0, 15), A(3, 15), B(15, 0), O(0, 0), and C(0, p).
The area of triangle ABC is given as 45.
We need to prove that p = 11.25. -/
theorem find_p (ABC_area : ℝ) (p : ℝ) (h : ABC_area = 45) :
  p = 11.25 :=
by
  sorry

end find_p_l7_7602


namespace find_number_exists_l7_7020

theorem find_number_exists (n : ℤ) : (50 < n ∧ n < 70) ∧
    (n % 5 = 3) ∧
    (n % 7 = 2) ∧
    (n % 8 = 2) → n = 58 := 
sorry

end find_number_exists_l7_7020


namespace find_z_l7_7873

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l7_7873


namespace find_multiple_l7_7473

-- Definitions based on the conditions provided
def mike_chocolate_squares : ℕ := 20
def jenny_chocolate_squares : ℕ := 65
def extra_squares : ℕ := 5

-- The theorem to prove the multiple
theorem find_multiple : ∃ (multiple : ℕ), jenny_chocolate_squares = mike_chocolate_squares * multiple + extra_squares ∧ multiple = 3 := by
  sorry

end find_multiple_l7_7473


namespace total_albums_l7_7937

theorem total_albums (Adele Bridget Katrina Miriam : ℕ) 
  (h₁ : Adele = 30)
  (h₂ : Bridget = Adele - 15)
  (h₃ : Katrina = 6 * Bridget)
  (h₄ : Miriam = 5 * Katrina) : Adele + Bridget + Katrina + Miriam = 585 := 
by
  sorry

end total_albums_l7_7937


namespace john_frank_age_ratio_l7_7573

theorem john_frank_age_ratio
  (F J : ℕ)
  (h1 : F + 4 = 16)
  (h2 : J - F = 15)
  (h3 : ∃ k : ℕ, J + 3 = k * (F + 3)) :
  (J + 3) / (F + 3) = 2 :=
by
  sorry

end john_frank_age_ratio_l7_7573


namespace problem_293_l7_7999

theorem problem_293 (s : ℝ) (R' : ℝ) (rectangle1 : ℝ) (circle1 : ℝ) 
  (condition1 : s = 4) 
  (condition2 : rectangle1 = 2 * 4) 
  (condition3 : circle1 = Real.pi * 1^2) 
  (condition4 : R' = s^2 - (rectangle1 + circle1)) 
  (fraction_form : ∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n) : 
  (∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n ∧ m + n = 293) := 
sorry

end problem_293_l7_7999


namespace candy_problem_l7_7045

theorem candy_problem (a : ℕ) (h₁ : a % 10 = 6) (h₂ : a % 15 = 11) (h₃ : 200 ≤ a) (h₄ : a ≤ 250) :
  a = 206 ∨ a = 236 :=
sorry

end candy_problem_l7_7045


namespace total_candies_l7_7953

variable (Adam James Rubert : Nat)
variable (Adam_has_candies : Adam = 6)
variable (James_has_candies : James = 3 * Adam)
variable (Rubert_has_candies : Rubert = 4 * James)

theorem total_candies : Adam + James + Rubert = 96 :=
by
  sorry

end total_candies_l7_7953


namespace balls_in_boxes_l7_7282

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end balls_in_boxes_l7_7282


namespace gcd_fifteen_x_five_l7_7789

theorem gcd_fifteen_x_five (n : ℕ) (h1 : 30 ≤ n) (h2 : n ≤ 40) (h3 : Nat.gcd 15 n = 5) : n = 35 ∨ n = 40 := 
sorry

end gcd_fifteen_x_five_l7_7789


namespace probability_of_exactly_nine_correct_matches_is_zero_l7_7207

theorem probability_of_exactly_nine_correct_matches_is_zero :
  let n := 10 in
  let match_probability (correct: Fin n → Fin n) (guess: Fin n → Fin n) (right_count: Nat) :=
    (Finset.univ.filter (λ i => correct i = guess i)).card = right_count in
  ∀ (correct_guessing: Fin n → Fin n), 
    ∀ (random_guessing: Fin n → Fin n),
      match_probability correct_guessing random_guessing 9 → 
        match_probability correct_guessing random_guessing 10 :=
begin
  sorry -- This skips the proof part
end

end probability_of_exactly_nine_correct_matches_is_zero_l7_7207


namespace original_number_unique_l7_7026

theorem original_number_unique (N : ℤ) (h : (N - 31) % 87 = 0) : N = 118 :=
by
  sorry

end original_number_unique_l7_7026


namespace num_ways_distribute_balls_l7_7280

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end num_ways_distribute_balls_l7_7280


namespace necessary_but_not_sufficient_l7_7125

def lines_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (a * x + 2 * y = 0) ↔ (x + (a + 1) * y + 4 = 0)

theorem necessary_but_not_sufficient (a : ℝ) :
  (a = 1 → lines_parallel a) ∧ ¬(lines_parallel a → a = 1) :=
by
  sorry

end necessary_but_not_sufficient_l7_7125


namespace arthur_amount_left_l7_7822

def initial_amount : ℝ := 200
def fraction_spent : ℝ := 4 / 5

def spent (initial : ℝ) (fraction : ℝ) : ℝ := fraction * initial

def amount_left (initial : ℝ) (spent_amount : ℝ) : ℝ := initial - spent_amount

theorem arthur_amount_left : amount_left initial_amount (spent initial_amount fraction_spent) = 40 := 
by
  sorry

end arthur_amount_left_l7_7822


namespace inverse_function_coeff_ratio_l7_7641

noncomputable def f_inv_coeff_ratio : ℝ :=
  let f (x : ℝ) := (2 * x - 1) / (x + 5)
  let a := 5
  let b := 1
  let c := -1
  let d := 2
  a / c

theorem inverse_function_coeff_ratio :
  f_inv_coeff_ratio = -5 := 
by
  sorry

end inverse_function_coeff_ratio_l7_7641


namespace total_albums_l7_7939

-- Definitions based on given conditions
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- The final statement to be proved
theorem total_albums : adele_albums + bridget_albums + katrina_albums + miriam_albums = 585 :=
by
  sorry

end total_albums_l7_7939


namespace B_can_finish_work_in_6_days_l7_7217

theorem B_can_finish_work_in_6_days :
  (A_work_alone : ℕ) → (A_work_before_B : ℕ) → (A_B_together : ℕ) → (B_days_alone : ℕ) → 
  (A_work_alone = 12) → (A_work_before_B = 3) → (A_B_together = 3) → B_days_alone = 6 :=
by
  intros A_work_alone A_work_before_B A_B_together B_days_alone
  intros h1 h2 h3
  sorry

end B_can_finish_work_in_6_days_l7_7217


namespace total_ladybugs_eq_11676_l7_7019

def Number_of_leaves : ℕ := 84
def Ladybugs_per_leaf : ℕ := 139

theorem total_ladybugs_eq_11676 : Number_of_leaves * Ladybugs_per_leaf = 11676 := by
  sorry

end total_ladybugs_eq_11676_l7_7019
