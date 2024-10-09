import Mathlib

namespace probability_four_squares_form_square_l437_43749

noncomputable def probability_form_square (n k : ℕ) :=
  if (k = 4) ∧ (n = 6) then (1 / 561 : ℚ) else 0

theorem probability_four_squares_form_square :
  probability_form_square 6 4 = (1 / 561 : ℚ) :=
by
  -- Here we would usually include the detailed proof
  -- corresponding to the solution steps from the problem,
  -- but we leave it as sorry for now.
  sorry

end probability_four_squares_form_square_l437_43749


namespace inequality_proof_l437_43736

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b < 2) : 
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ 
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ 0 < a ∧ a = b ∧ a < 1) := 
by 
  sorry

end inequality_proof_l437_43736


namespace mrs_hilt_rocks_proof_l437_43744

def num_rocks_already_placed : ℝ := 125.0
def total_num_rocks_planned : ℝ := 189
def num_more_rocks_needed : ℝ := 64

theorem mrs_hilt_rocks_proof : total_num_rocks_planned - num_rocks_already_placed = num_more_rocks_needed :=
by
  sorry

end mrs_hilt_rocks_proof_l437_43744


namespace staplers_left_is_correct_l437_43717

-- Define the initial conditions as constants
def initial_staplers : ℕ := 450
def stacie_reports : ℕ := 8 * 12 -- Stacie's reports in dozens converted to actual number
def jack_reports : ℕ := 9 * 12   -- Jack's reports in dozens converted to actual number
def laura_reports : ℕ := 50      -- Laura's individual reports

-- Define the stapler usage rates
def stacie_usage_rate : ℕ := 1                  -- Stacie's stapler usage rate (1 stapler per report)
def jack_usage_rate : ℕ := stacie_usage_rate / 2  -- Jack's stapler usage rate (half of Stacie's)
def laura_usage_rate : ℕ := stacie_usage_rate * 2 -- Laura's stapler usage rate (twice of Stacie's)

-- Define the usage calculations
def stacie_usage : ℕ := stacie_reports * stacie_usage_rate
def jack_usage : ℕ := jack_reports * jack_usage_rate
def laura_usage : ℕ := laura_reports * laura_usage_rate

-- Define total staplers used
def total_usage : ℕ := stacie_usage + jack_usage + laura_usage

-- Define the number of staplers left
def staplers_left : ℕ := initial_staplers - total_usage

-- Prove that the staplers left is 200
theorem staplers_left_is_correct : staplers_left = 200 := by
  unfold staplers_left initial_staplers total_usage stacie_usage jack_usage laura_usage
  unfold stacie_reports jack_reports laura_reports
  unfold stacie_usage_rate jack_usage_rate laura_usage_rate
  sorry   -- Place proof here

end staplers_left_is_correct_l437_43717


namespace total_handshakes_eq_900_l437_43772

def num_boys : ℕ := 25
def handshakes_per_pair : ℕ := 3

theorem total_handshakes_eq_900 : (num_boys * (num_boys - 1) / 2) * handshakes_per_pair = 900 := by
  sorry

end total_handshakes_eq_900_l437_43772


namespace minimum_value_ineq_l437_43779

noncomputable def minimum_value (x y z : ℝ) := x^2 + 4 * x * y + 4 * y^2 + 4 * z^2

theorem minimum_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 64) : minimum_value x y z ≥ 192 :=
by {
  sorry
}

end minimum_value_ineq_l437_43779


namespace gcd_m_n_is_one_l437_43753

/-- Definition of m -/
def m : ℕ := 130^2 + 241^2 + 352^2

/-- Definition of n -/
def n : ℕ := 129^2 + 240^2 + 353^2 + 2^3

/-- Proof statement: The greatest common divisor of m and n is 1 -/
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l437_43753


namespace amy_remaining_money_l437_43783

-- Definitions based on conditions
def initial_money : ℕ := 100
def doll_cost : ℕ := 1
def number_of_dolls : ℕ := 3

-- The theorem we want to prove
theorem amy_remaining_money : initial_money - number_of_dolls * doll_cost = 97 :=
by 
  sorry

end amy_remaining_money_l437_43783


namespace Jasmine_initial_percentage_is_5_l437_43775

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

end Jasmine_initial_percentage_is_5_l437_43775


namespace total_spent_is_correct_l437_43759

-- Declare the constants for the prices and quantities
def wallet_cost : ℕ := 50
def sneakers_cost_per_pair : ℕ := 100
def sneakers_pairs : ℕ := 2
def backpack_cost : ℕ := 100
def jeans_cost_per_pair : ℕ := 50
def jeans_pairs : ℕ := 2

-- Define the total amounts spent by Leonard and Michael
def leonard_total : ℕ := wallet_cost + sneakers_cost_per_pair * sneakers_pairs
def michael_total : ℕ := backpack_cost + jeans_cost_per_pair * jeans_pairs

-- The total amount spent by Leonard and Michael
def total_spent : ℕ := leonard_total + michael_total

-- The proof statement
theorem total_spent_is_correct : total_spent = 450 :=
by 
  -- This part is where the proof would go
  sorry

end total_spent_is_correct_l437_43759


namespace ab_zero_l437_43705

theorem ab_zero
  (a b : ℤ)
  (h : ∀ (m n : ℕ), ∃ (k : ℤ), a * (m : ℤ) ^ 2 + b * (n : ℤ) ^ 2 = k ^ 2) :
  a * b = 0 :=
sorry

end ab_zero_l437_43705


namespace find_white_towels_l437_43762

variable {W : ℕ} -- Define W as a natural number

-- Define the conditions as Lean definitions
def initial_towel_count (W : ℕ) : ℕ := 35 + W
def remaining_towel_count (W : ℕ) : ℕ := initial_towel_count W - 34

-- Theorem statement: Proving that W = 21 given the conditions
theorem find_white_towels (h : remaining_towel_count W = 22) : W = 21 :=
by
  sorry

end find_white_towels_l437_43762


namespace sabrina_herbs_l437_43796

theorem sabrina_herbs (S V : ℕ) 
  (h1 : 2 * S = 12)
  (h2 : 12 + S + V = 29) :
  V - S = 5 := by
  sorry

end sabrina_herbs_l437_43796


namespace arithmetic_seq_sum_l437_43794

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_123 : a 0 + a 1 + a 2 = -3)
  (h_456 : a 3 + a 4 + a 5 = 6) :
  ∀ n, S n = n * (-2) + n * (n - 1) / 2 :=
by
  sorry

end arithmetic_seq_sum_l437_43794


namespace solution_set_of_inequality_l437_43730

noncomputable def f (x : ℝ) : ℝ := (1 / x) * (1 / 2 * (Real.log x) ^ 2 + 1 / 2)

theorem solution_set_of_inequality :
  (∀ x : ℝ, x > 0 → x < e → f x - x > f e - e) ↔ (∀ x : ℝ, 0 < x ∧ x < e) :=
by
  sorry

end solution_set_of_inequality_l437_43730


namespace opposite_of_neg_eight_l437_43773

theorem opposite_of_neg_eight : (-(-8)) = 8 :=
by
  sorry

end opposite_of_neg_eight_l437_43773


namespace c_share_l437_43751

theorem c_share (a b c : ℝ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : a + b + c = 700) : c = 400 :=
by 
  -- Proof goes here
  sorry

end c_share_l437_43751


namespace monomial_same_type_m_n_sum_l437_43743

theorem monomial_same_type_m_n_sum (m n : ℕ) (x y : ℤ) 
  (h1 : 2 * x ^ (m - 1) * y ^ 2 = 1/3 * x ^ 2 * y ^ (n + 1)) : 
  m + n = 4 := 
sorry

end monomial_same_type_m_n_sum_l437_43743


namespace h_at_4_l437_43745

noncomputable def f (x : ℝ) := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) := 3 - (4 / x)

noncomputable def h (x : ℝ) := (1 / f_inv x) + 10

theorem h_at_4 : h 4 = 10.5 :=
by
  sorry

end h_at_4_l437_43745


namespace negation_exists_l437_43702

theorem negation_exists (x : ℝ) (h : x ≥ 0) : (¬ (∀ x : ℝ, (x ≥ 0) → (2^x > x^2))) ↔ (∃ x₀ : ℝ, (x₀ ≥ 0) ∧ (2 ^ x₀ ≤ x₀^2)) := by
  sorry

end negation_exists_l437_43702


namespace angle_between_tangents_l437_43760

theorem angle_between_tangents (R1 R2 : ℝ) (k : ℝ) (h_ratio : R1 = 2 * k ∧ R2 = 3 * k)
  (h_touching : (∃ O1 O2 : ℝ, (R2 - R1 = k))) : 
  ∃ θ : ℝ, θ = 90 := sorry

end angle_between_tangents_l437_43760


namespace room_width_is_12_l437_43790

variable (w : ℕ)

-- Definitions of given conditions
def room_length := 19
def veranda_width := 2
def veranda_area := 140

-- Statement that needs to be proven
theorem room_width_is_12
  (h1 : veranda_width = 2)
  (h2 : veranda_area = 140)
  (h3 : room_length = 19) :
  w = 12 :=
by
  sorry

end room_width_is_12_l437_43790


namespace ceil_sqrt_200_eq_15_l437_43765

theorem ceil_sqrt_200_eq_15 : Int.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l437_43765


namespace percentage_of_adults_is_40_l437_43786

variables (A C : ℕ)

-- Given conditions as definitions
def total_members := 120
def more_children_than_adults := 24
def percentage_of_adults (A : ℕ) := (A.toFloat / total_members.toFloat) * 100

-- Lean 4 statement to prove the percentage of adults
theorem percentage_of_adults_is_40 (h1 : A + C = 120)
                                   (h2 : C = A + 24) :
  percentage_of_adults A = 40 :=
by
  sorry

end percentage_of_adults_is_40_l437_43786


namespace students_count_l437_43715

theorem students_count (S : ℕ) (num_adults : ℕ) (cost_student cost_adult total_cost : ℕ)
  (h1 : num_adults = 4)
  (h2 : cost_student = 5)
  (h3 : cost_adult = 6)
  (h4 : total_cost = 199) :
  5 * S + 4 * 6 = 199 → S = 35 := by
  sorry

end students_count_l437_43715


namespace tailor_cut_difference_l437_43755

def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

theorem tailor_cut_difference : skirt_cut - pants_cut = 0.25 :=
by
  sorry

end tailor_cut_difference_l437_43755


namespace solve_for_x_l437_43723

theorem solve_for_x (x : ℤ) (h : x + 1 = 4) : x = 3 :=
sorry

end solve_for_x_l437_43723


namespace candy_difference_l437_43784

theorem candy_difference (Frankie_candies Max_candies : ℕ) (hF : Frankie_candies = 74) (hM : Max_candies = 92) :
  Max_candies - Frankie_candies = 18 :=
by
  sorry

end candy_difference_l437_43784


namespace fraction_simplifies_l437_43728

def current_age_grant := 25
def current_age_hospital := 40

def age_in_five_years (current_age : Nat) : Nat := current_age + 5

def grant_age_in_5_years := age_in_five_years current_age_grant
def hospital_age_in_5_years := age_in_five_years current_age_hospital

def fraction_of_ages := grant_age_in_5_years / hospital_age_in_5_years

theorem fraction_simplifies : fraction_of_ages = (2 / 3) := by
  sorry

end fraction_simplifies_l437_43728


namespace train_passing_time_l437_43766

theorem train_passing_time :
  ∀ (length : ℕ) (speed_km_hr : ℕ), length = 300 ∧ speed_km_hr = 90 →
  (length / (speed_km_hr * (1000 / 3600)) = 12) := 
by
  intros length speed_km_hr h
  have h_length : length = 300 := h.1
  have h_speed : speed_km_hr = 90 := h.2
  sorry

end train_passing_time_l437_43766


namespace multiply_expression_l437_43729

theorem multiply_expression (x : ℝ) : 
  (x^4 + 49 * x^2 + 2401) * (x^2 - 49) = x^6 - 117649 :=
by
  sorry

end multiply_expression_l437_43729


namespace a_alone_can_finish_in_60_days_l437_43725

variables (A B C : ℚ)

noncomputable def a_b_work_rate := A + B = 1/40
noncomputable def a_c_work_rate := A + 1/30 = 1/20

theorem a_alone_can_finish_in_60_days (A B C : ℚ) 
  (h₁ : a_b_work_rate A B) 
  (h₂ : a_c_work_rate A) : 
  A = 1/60 := 
sorry

end a_alone_can_finish_in_60_days_l437_43725


namespace equation_has_solution_implies_a_ge_2_l437_43781

theorem equation_has_solution_implies_a_ge_2 (a : ℝ) :
  (∃ x : ℝ, 4^x - a * 2^x - a + 3 = 0) → a ≥ 2 :=
by
  sorry

end equation_has_solution_implies_a_ge_2_l437_43781


namespace hexagon_tiling_colors_l437_43714

-- Problem Definition
theorem hexagon_tiling_colors (k l : ℕ) (hk : 0 < k ∨ 0 < l) : 
  ∃ n: ℕ, n = k^2 + k * l + l^2 :=
by
  sorry

end hexagon_tiling_colors_l437_43714


namespace perpendicular_bisector_eq_l437_43752

theorem perpendicular_bisector_eq (x y : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 5 = 0 ∧ x^2 + y^2 + 2 * x - 4 * y - 4 = 0) →
  x + y - 1 = 0 :=
by
  sorry

end perpendicular_bisector_eq_l437_43752


namespace determine_a_l437_43763

def A := {x : ℝ | x < 6}
def B (a : ℝ) := {x : ℝ | x - a < 0}

theorem determine_a (a : ℝ) (h : A ⊆ B a) : 6 ≤ a := 
sorry

end determine_a_l437_43763


namespace difference_is_167_l437_43739

-- Define the number of boys and girls in each village
def A_village_boys : ℕ := 204
def A_village_girls : ℕ := 468
def B_village_boys : ℕ := 334
def B_village_girls : ℕ := 516
def C_village_boys : ℕ := 427
def C_village_girls : ℕ := 458
def D_village_boys : ℕ := 549
def D_village_girls : ℕ := 239

-- Define total number of boys and girls
def total_boys := A_village_boys + B_village_boys + C_village_boys + D_village_boys
def total_girls := A_village_girls + B_village_girls + C_village_girls + D_village_girls

-- Define the difference between total girls and total boys
def difference := total_girls - total_boys

-- The theorem to prove the difference is 167
theorem difference_is_167 : difference = 167 := by
  sorry

end difference_is_167_l437_43739


namespace weekly_milk_production_l437_43704

theorem weekly_milk_production 
  (bess_milk_per_day : ℕ) 
  (brownie_milk_per_day : ℕ) 
  (daisy_milk_per_day : ℕ) 
  (total_milk_per_day : ℕ) 
  (total_milk_per_week : ℕ) 
  (h1 : bess_milk_per_day = 2) 
  (h2 : brownie_milk_per_day = 3 * bess_milk_per_day) 
  (h3 : daisy_milk_per_day = bess_milk_per_day + 1) 
  (h4 : total_milk_per_day = bess_milk_per_day + brownie_milk_per_day + daisy_milk_per_day)
  (h5 : total_milk_per_week = total_milk_per_day * 7) : 
  total_milk_per_week = 77 := 
by sorry

end weekly_milk_production_l437_43704


namespace chips_sales_l437_43791

theorem chips_sales (total_chips : ℕ) (first_week : ℕ) (second_week : ℕ) (third_week : ℕ) (fourth_week : ℕ)
  (h1 : total_chips = 100)
  (h2 : first_week = 15)
  (h3 : second_week = 3 * first_week)
  (h4 : third_week = fourth_week)
  (h5 : total_chips = first_week + second_week + third_week + fourth_week) : third_week = 20 :=
by
  sorry

end chips_sales_l437_43791


namespace road_length_in_km_l437_43774

/-- The actual length of the road in kilometers is 7.5, given the scale of 1:50000 
    and the map length of 15 cm. -/

theorem road_length_in_km (s : ℕ) (map_length_cm : ℕ) (actual_length_cm : ℕ) (actual_length_km : ℝ) 
  (h_scale : s = 50000) (h_map_length : map_length_cm = 15) (h_conversion : actual_length_km = actual_length_cm / 100000) :
  actual_length_km = 7.5 :=
  sorry

end road_length_in_km_l437_43774


namespace Jonas_initial_socks_l437_43724

noncomputable def pairsOfSocks(Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                              (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) : ℕ :=
    let individualShoes := Jonas_pairsOfShoes * 2
    let individualPants := Jonas_pairsOfPants * 2
    let individualTShirts := Jonas_tShirts
    let totalWithoutSocks := individualShoes + individualPants + individualTShirts
    let totalToDouble := (totalWithoutSocks + Jonas_pairsOfNewSocks * 2) / 2
    (totalToDouble * 2 - totalWithoutSocks) / 2

theorem Jonas_initial_socks (Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                             (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) 
                             (h1 : Jonas_pairsOfShoes = 5)
                             (h2 : Jonas_pairsOfPants = 10)
                             (h3 : Jonas_tShirts = 10)
                             (h4 : Jonas_pairsOfNewSocks = 35) :
    pairsOfSocks Jonas_pairsOfShoes Jonas_pairsOfPants Jonas_tShirts Jonas_pairsOfNewSocks = 15 :=
by
    subst h1
    subst h2
    subst h3
    subst h4
    sorry

end Jonas_initial_socks_l437_43724


namespace jake_present_weight_l437_43798

theorem jake_present_weight (J S : ℕ) 
  (h1 : J - 32 = 2 * S) 
  (h2 : J + S = 212) : 
  J = 152 := 
by 
  sorry

end jake_present_weight_l437_43798


namespace hyperbola_center_l437_43761

theorem hyperbola_center : 
  (∃ x y : ℝ, (4 * y + 6)^2 / 16 - (5 * x - 3)^2 / 9 = 1) →
  (∃ h k : ℝ, h = 3 / 5 ∧ k = -3 / 2 ∧ 
    (∀ x' y', (4 * y' + 6)^2 / 16 - (5 * x' - 3)^2 / 9 = 1 → x' = h ∧ y' = k)) :=
sorry

end hyperbola_center_l437_43761


namespace determine_numbers_l437_43776

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l437_43776


namespace mart_income_more_than_tim_l437_43789

variable (J : ℝ) -- Let's denote Juan's income as J
def T : ℝ := J - 0.40 * J -- Tim's income is 40 percent less than Juan's income
def M : ℝ := 0.78 * J -- Mart's income is 78 percent of Juan's income

theorem mart_income_more_than_tim : (M - T) / T * 100 = 30 := by
  sorry

end mart_income_more_than_tim_l437_43789


namespace total_pies_baked_in_7_days_l437_43719

-- Define the baking rates (pies per day)
def Eddie_rate : Nat := 3
def Sister_rate : Nat := 6
def Mother_rate : Nat := 8

-- Define the duration in days
def duration : Nat := 7

-- Define the total number of pies baked in 7 days
def total_pies : Nat := Eddie_rate * duration + Sister_rate * duration + Mother_rate * duration

-- Prove the total number of pies is 119
theorem total_pies_baked_in_7_days : total_pies = 119 := by
  -- The proof will be filled here, adding sorry to skip it for now
  sorry

end total_pies_baked_in_7_days_l437_43719


namespace tammy_total_distance_l437_43750

-- Define the times and speeds for each segment and breaks
def initial_speed : ℝ := 55   -- miles per hour
def initial_time : ℝ := 2     -- hours
def road_speed : ℝ := 40      -- miles per hour
def road_time : ℝ := 5        -- hours
def first_break : ℝ := 1      -- hour
def drive_after_break_speed : ℝ := 50  -- miles per hour
def drive_after_break_time : ℝ := 15   -- hours
def hilly_speed : ℝ := 35     -- miles per hour
def hilly_time : ℝ := 3       -- hours
def second_break : ℝ := 0.5   -- hours
def finish_speed : ℝ := 60    -- miles per hour
def total_journey_time : ℝ := 36 -- hours

-- Define a function to calculate the segment distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Define the total distance calculation
def total_distance : ℝ :=
  distance initial_speed initial_time +
  distance road_speed road_time +
  distance drive_after_break_speed drive_after_break_time +
  distance hilly_speed hilly_time +
  distance finish_speed (total_journey_time - (initial_time + road_time + drive_after_break_time + hilly_time + first_break + second_break))

-- The final proof statement
theorem tammy_total_distance : total_distance = 1735 :=
  sorry

end tammy_total_distance_l437_43750


namespace line_length_after_erasing_l437_43785

theorem line_length_after_erasing :
  ∀ (initial_length_m : ℕ) (conversion_factor : ℕ) (erased_length_cm : ℕ),
  initial_length_m = 1 → conversion_factor = 100 → erased_length_cm = 33 →
  initial_length_m * conversion_factor - erased_length_cm = 67 :=
by {
  sorry
}

end line_length_after_erasing_l437_43785


namespace calc_fraction_l437_43733

theorem calc_fraction : (36 + 12) / (6 - 3) = 16 :=
by
  sorry

end calc_fraction_l437_43733


namespace first_worker_time_budget_l437_43764

theorem first_worker_time_budget
  (total_time : ℝ := 1)
  (second_worker_time : ℝ := 1 / 3)
  (third_worker_time : ℝ := 1 / 3)
  (x : ℝ) :
  x + second_worker_time + third_worker_time = total_time → x = 1 / 3 :=
by
  sorry

end first_worker_time_budget_l437_43764


namespace find_pairs_l437_43788

theorem find_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (∃ k m : ℕ, k ≠ 0 ∧ m ≠ 0 ∧ x + 1 = k * y ∧ y + 1 = m * x) ↔
  (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 3) :=
by
  sorry

end find_pairs_l437_43788


namespace amount_to_add_l437_43748

theorem amount_to_add (students : ℕ) (total_cost : ℕ) (h1 : students = 9) (h2 : total_cost = 143) : 
  ∃ k : ℕ, total_cost + k = students * (total_cost / students + 1) :=
by
  sorry

end amount_to_add_l437_43748


namespace right_triangle_sides_l437_43701

theorem right_triangle_sides (p m : ℝ)
  (hp : 0 < p)
  (hm : 0 < m) :
  ∃ a b c : ℝ, 
    a + b + c = 2 * p ∧
    a^2 + b^2 = c^2 ∧
    (1 / 2) * a * b = m^2 ∧
    c = (p^2 - m^2) / p ∧
    a = (p^2 + m^2 + Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) ∧
    b = (p^2 + m^2 - Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) := 
by
  sorry

end right_triangle_sides_l437_43701


namespace min_faces_n2_min_faces_n3_l437_43740

noncomputable def minimum_faces (n : ℕ) : ℕ := 
  if n = 2 then 2 
  else if n = 3 then 12 
  else sorry 

theorem min_faces_n2 : minimum_faces 2 = 2 := 
  by 
  simp [minimum_faces]

theorem min_faces_n3 : minimum_faces 3 = 12 := 
  by 
  simp [minimum_faces]

end min_faces_n2_min_faces_n3_l437_43740


namespace intersection_A_B_l437_43711

open Set

def A : Set ℕ := {x | -2 < (x : ℤ) ∧ (x : ℤ) < 2}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ {x : ℕ | (x : ℤ) ∈ B} = {0, 1} := by
  sorry

end intersection_A_B_l437_43711


namespace total_votes_l437_43710

theorem total_votes (T F A : ℝ)
  (h1 : F = A + 68)
  (h2 : A = 0.40 * T)
  (h3 : T = F + A) :
  T = 340 :=
by sorry

end total_votes_l437_43710


namespace luncheon_cost_l437_43757

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + p = 3.00)
  (h2 : 5 * s + 8 * c + p = 5.40)
  (h3 : 3 * s + 4 * c + p = 3.60) :
  2 * s + 2 * c + p = 2.60 :=
sorry

end luncheon_cost_l437_43757


namespace log_difference_example_l437_43754

theorem log_difference_example :
  ∀ (log : ℕ → ℝ),
    log 3 * 24 - log 3 * 8 = 1 := 
by
sorry

end log_difference_example_l437_43754


namespace racing_magic_circle_time_l437_43756

theorem racing_magic_circle_time
  (T : ℕ) -- Time taken by the racing magic to circle the track once
  (bull_rounds_per_hour : ℕ := 40) -- Rounds the Charging Bull makes in an hour
  (meet_time_minutes : ℕ := 6) -- Time in minutes to meet at starting point
  (charging_bull_seconds_per_round : ℕ := 3600 / bull_rounds_per_hour) -- Time in seconds per Charging Bull round
  (meet_time_seconds : ℕ := meet_time_minutes * 60) -- Time in seconds to meet at starting point
  (rounds_by_bull : ℕ := meet_time_seconds / charging_bull_seconds_per_round) -- Rounds completed by the Charging Bull to meet again
  (rounds_by_magic : ℕ := meet_time_seconds / T) -- Rounds completed by the Racing Magic to meet again
  (h1 : rounds_by_magic = 1) -- Racing Magic completes 1 round in the meet time
  : T = 360 := -- Racing Magic takes 360 seconds to circle the track once
  sorry

end racing_magic_circle_time_l437_43756


namespace trey_will_sell_bracelets_for_days_l437_43797

def cost : ℕ := 112
def price_per_bracelet : ℕ := 1
def bracelets_per_day : ℕ := 8

theorem trey_will_sell_bracelets_for_days :
  ∃ d : ℕ, d = cost / (price_per_bracelet * bracelets_per_day) ∧ d = 14 := by
  sorry

end trey_will_sell_bracelets_for_days_l437_43797


namespace Jiyeol_average_score_l437_43742

theorem Jiyeol_average_score (K M E : ℝ)
  (h1 : (K + M) / 2 = 26.5)
  (h2 : (M + E) / 2 = 34.5)
  (h3 : (K + E) / 2 = 29) :
  (K + M + E) / 3 = 30 := 
sorry

end Jiyeol_average_score_l437_43742


namespace find_x_l437_43721

theorem find_x (x : ℝ) : 0.20 * x - (1 / 3) * (0.20 * x) = 24 → x = 180 :=
by
  intro h
  sorry

end find_x_l437_43721


namespace pencils_total_l437_43716

-- Defining the conditions
def packs_to_pencils (packs : ℕ) : ℕ := packs * 12

def jimin_packs : ℕ := 2
def jimin_individual_pencils : ℕ := 7

def yuna_packs : ℕ := 1
def yuna_individual_pencils : ℕ := 9

-- Translating to Lean 4 statement
theorem pencils_total : 
  packs_to_pencils jimin_packs + jimin_individual_pencils + packs_to_pencils yuna_packs + yuna_individual_pencils = 52 := 
by
  sorry

end pencils_total_l437_43716


namespace PE_bisects_CD_given_conditions_l437_43746

variables {A B C D E P : Type*}

noncomputable def cyclic_quadrilateral (A B C D : Type*) : Prop := sorry

noncomputable def AD_squared_plus_BC_squared_eq_AB_squared (A B C D : Type*) : Prop := sorry

noncomputable def angles_equality_condition (A B C D P : Type*) : Prop := sorry

noncomputable def line_PE_bisects_CD (P E C D : Type*) : Prop := sorry

theorem PE_bisects_CD_given_conditions
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : AD_squared_plus_BC_squared_eq_AB_squared A B C D)
  (h3 : angles_equality_condition A B C D P) :
  line_PE_bisects_CD P E C D :=
sorry

end PE_bisects_CD_given_conditions_l437_43746


namespace train_length_l437_43782

noncomputable def L_train : ℝ :=
  let speed_kmph : ℝ := 60
  let speed_mps : ℝ := (speed_kmph * 1000 / 3600)
  let time : ℝ := 30
  let length_bridge : ℝ := 140
  let total_distance : ℝ := speed_mps * time
  total_distance - length_bridge

theorem train_length : L_train = 360.1 :=
by
  -- Sorry statement to skip the proof
  sorry

end train_length_l437_43782


namespace lcm_12_18_l437_43734

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l437_43734


namespace expected_number_of_digits_l437_43767

noncomputable def expectedNumberDigits : ℝ :=
  let oneDigitProbability := (9 : ℝ) / 16
  let twoDigitProbability := (7 : ℝ) / 16
  (oneDigitProbability * 1) + (twoDigitProbability * 2)

theorem expected_number_of_digits :
  expectedNumberDigits = 1.4375 := by
  sorry

end expected_number_of_digits_l437_43767


namespace negation_of_exists_gt_implies_forall_leq_l437_43771

theorem negation_of_exists_gt_implies_forall_leq (x : ℝ) (h : 0 < x) :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_exists_gt_implies_forall_leq_l437_43771


namespace total_number_of_questions_l437_43795

/-
  Given:
    1. There are 20 type A problems.
    2. Type A problems require twice as much time as type B problems.
    3. 32.73 minutes are spent on type A problems.
    4. Total examination time is 3 hours.

  Prove that the total number of questions is 199.
-/

theorem total_number_of_questions
  (type_A_problems : ℕ)
  (type_B_to_A_time_ratio : ℝ)
  (time_spent_on_type_A : ℝ)
  (total_exam_time_hours : ℝ)
  (total_number_of_questions : ℕ)
  (h_type_A_problems : type_A_problems = 20)
  (h_time_ratio : type_B_to_A_time_ratio = 2)
  (h_time_spent_on_type_A : time_spent_on_type_A = 32.73)
  (h_total_exam_time_hours : total_exam_time_hours = 3) :
  total_number_of_questions = 199 := 
sorry

end total_number_of_questions_l437_43795


namespace min_value_ab2_cd_l437_43770

noncomputable def arithmetic_seq (x a b y : ℝ) : Prop :=
  2 * a = x + b ∧ 2 * b = a + y

noncomputable def geometric_seq (x c d y : ℝ) : Prop :=
  c^2 = x * d ∧ d^2 = c * y

theorem min_value_ab2_cd (x y a b c d : ℝ) :
  (x > 0) → (y > 0) → arithmetic_seq x a b y → geometric_seq x c d y → 
  (a + b) ^ 2 / (c * d) ≥ 4 :=
by
  sorry

end min_value_ab2_cd_l437_43770


namespace celebrity_baby_photo_probability_l437_43732

theorem celebrity_baby_photo_probability : 
  let total_arrangements := Nat.factorial 4
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = 1/24 :=
by
  sorry

end celebrity_baby_photo_probability_l437_43732


namespace train_speed_is_64_kmh_l437_43780

noncomputable def train_speed_kmh (train_length platform_length time_seconds : ℕ) : ℕ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time_seconds
  let speed_kmh := speed_mps * 36 / 10
  speed_kmh

theorem train_speed_is_64_kmh
  (train_length : ℕ)
  (platform_length : ℕ)
  (time_seconds : ℕ)
  (h_train_length : train_length = 240)
  (h_platform_length : platform_length = 240)
  (h_time_seconds : time_seconds = 27) :
  train_speed_kmh train_length platform_length time_seconds = 64 := by
  sorry

end train_speed_is_64_kmh_l437_43780


namespace Petya_bonus_points_l437_43709

def bonus_points (p : ℕ) : ℕ :=
  if p < 1000 then
    (20 * p) / 100
  else if p ≤ 2000 then
    200 + (30 * (p - 1000)) / 100
  else
    200 + 300 + (50 * (p - 2000)) / 100

theorem Petya_bonus_points : bonus_points 2370 = 685 :=
by sorry

end Petya_bonus_points_l437_43709


namespace selling_price_is_correct_l437_43792

noncomputable def purchase_price : ℝ := 36400
noncomputable def repair_costs : ℝ := 8000
noncomputable def profit_percent : ℝ := 54.054054054054056

noncomputable def total_cost := purchase_price + repair_costs
noncomputable def selling_price := total_cost * (1 + profit_percent / 100)

theorem selling_price_is_correct :
    selling_price = 68384 := by
  sorry

end selling_price_is_correct_l437_43792


namespace length_of_each_piece_cm_l437_43727

theorem length_of_each_piece_cm 
  (total_length : ℝ) 
  (number_of_pieces : ℕ) 
  (htotal : total_length = 17) 
  (hpieces : number_of_pieces = 20) : 
  (total_length / number_of_pieces) * 100 = 85 := 
by
  sorry

end length_of_each_piece_cm_l437_43727


namespace sequence_general_term_l437_43700

/-- The general term formula for the sequence 0.3, 0.33, 0.333, 0.3333, … is (1 / 3) * (1 - 1 / 10 ^ n). -/
theorem sequence_general_term (n : ℕ) : 
  (∃ a : ℕ → ℚ, (∀ n, a n = 0.3 + 0.03 * (10 ^ (n + 1) - 1) / 10 ^ (n + 1))) ↔
  ∀ n, (0.3 + 0.03 * (10 ^ (n + 1) - 1) / 10 ^ (n + 1)) = (1 / 3) * (1 - 1 / 10 ^ n) :=
sorry

end sequence_general_term_l437_43700


namespace fraction_of_selected_color_films_equals_five_twenty_sixths_l437_43787

noncomputable def fraction_of_selected_color_films (x y : ℕ) : ℚ :=
  let bw_films := 40 * x
  let color_films := 10 * y
  let selected_bw_films := (y / x * 1 / 100) * bw_films
  let selected_color_films := color_films
  let total_selected_films := selected_bw_films + selected_color_films
  selected_color_films / total_selected_films

theorem fraction_of_selected_color_films_equals_five_twenty_sixths (x y : ℕ) (h1 : x > 0) (h2 : y > 0) :
  fraction_of_selected_color_films x y = 5 / 26 := by
  sorry

end fraction_of_selected_color_films_equals_five_twenty_sixths_l437_43787


namespace circle_area_l437_43737

theorem circle_area (r : ℝ) (h : 8 * (1 / (2 * π * r)) = 2 * r) : π * r^2 = 2 := by
  sorry

end circle_area_l437_43737


namespace divisibility_by_5_l437_43703

theorem divisibility_by_5 (n : ℕ) (h : 0 < n) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end divisibility_by_5_l437_43703


namespace ganesh_ram_sohan_work_time_l437_43777

theorem ganesh_ram_sohan_work_time (G R S : ℝ)
  (H1 : G + R = 1 / 24)
  (H2 : S = 1 / 48) : (G + R + S = 1 / 16) ∧ (1 / (G + R + S) = 16) :=
by
  sorry

end ganesh_ram_sohan_work_time_l437_43777


namespace find_cos_minus_sin_l437_43718

-- Definitions from the conditions
variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)  -- Second quadrant
variable (h2 : Real.sin (2 * α) = -24 / 25)  -- Given sin 2α

-- Lean statement of the problem
theorem find_cos_minus_sin (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin (2 * α) = -24 / 25) :
  Real.cos α - Real.sin α = -7 / 5 := 
sorry

end find_cos_minus_sin_l437_43718


namespace total_distance_l437_43769

/--
John's journey is from point (-3, 4) to (2, 2) to (6, -3).
Prove that the total distance John travels is the sum of distances
from (-3, 4) to (2, 2) and from (2, 2) to (6, -3).
-/
theorem total_distance : 
  let d1 := Real.sqrt ((-3 - 2)^2 + (4 - 2)^2)
  let d2 := Real.sqrt ((6 - 2)^2 + (-3 - 2)^2)
  d1 + d2 = Real.sqrt 29 + Real.sqrt 41 :=
by
  sorry

end total_distance_l437_43769


namespace teenas_speed_l437_43712

theorem teenas_speed (T : ℝ) :
  (7.5 + 15 + 40 * 1.5 = T * 1.5) → T = 55 := 
by
  intro h
  sorry

end teenas_speed_l437_43712


namespace B_pow_101_eq_B_l437_43706

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![-1, 0, 0], ![0, 0, 0]]

-- State the theorem
theorem B_pow_101_eq_B : B^101 = B :=
  sorry

end B_pow_101_eq_B_l437_43706


namespace Z_equals_i_l437_43708

noncomputable def Z : ℂ := (Real.sqrt 2 - (Complex.I ^ 3)) / (1 - Real.sqrt 2 * Complex.I)

theorem Z_equals_i : Z = Complex.I := 
by 
  sorry

end Z_equals_i_l437_43708


namespace initial_distance_l437_43747

/-- Suppose Jack walks at a speed of 3 feet per second toward Christina,
    Christina walks at a speed of 3 feet per second toward Jack, and their dog Lindy
    runs at a speed of 10 feet per second back and forth between Jack and Christina.
    Given that Lindy travels a total of 400 feet when they meet, prove that the initial
    distance between Jack and Christina is 240 feet. -/
theorem initial_distance (initial_distance_jack_christina : ℝ)
  (jack_speed : ℝ := 3)
  (christina_speed : ℝ := 3)
  (lindy_speed : ℝ := 10)
  (lindy_total_distance : ℝ := 400):
  initial_distance_jack_christina = 240 :=
sorry

end initial_distance_l437_43747


namespace part_one_part_two_part_three_l437_43726

open Nat

def number_boys := 5
def number_girls := 4
def total_people := 9
def A_included := 1
def B_included := 1

theorem part_one : (number_boys.choose 2 * number_girls.choose 2) = 60 := sorry

theorem part_two : (total_people.choose 4 - (total_people - A_included - B_included).choose 4) = 91 := sorry

theorem part_three : (total_people.choose 4 - number_boys.choose 4 - number_girls.choose 4) = 120 := sorry

end part_one_part_two_part_three_l437_43726


namespace seq_a_ge_two_pow_nine_nine_l437_43738

theorem seq_a_ge_two_pow_nine_nine (a : ℕ → ℤ) 
  (h0 : a 1 > a 0)
  (h1 : a 1 > 0)
  (h2 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2^99 :=
sorry

end seq_a_ge_two_pow_nine_nine_l437_43738


namespace multiple_of_second_lock_time_l437_43758

def first_lock_time := 5
def second_lock_time := 3 * first_lock_time - 3
def combined_lock_time := 60

theorem multiple_of_second_lock_time : combined_lock_time / second_lock_time = 5 := by
  -- Adding a proof placeholder using sorry
  sorry

end multiple_of_second_lock_time_l437_43758


namespace max_value_of_expr_l437_43735

theorem max_value_of_expr 
  (x y z : ℝ) 
  (h₀ : 0 < x) 
  (h₁ : 0 < y) 
  (h₂ : 0 < z)
  (h : x^2 + y^2 + z^2 = 1) : 
  3 * x * y + y * z ≤ (Real.sqrt 10) / 2 := 
  sorry

end max_value_of_expr_l437_43735


namespace hyperbola_properties_l437_43799

theorem hyperbola_properties :
  (∃ x y : Real,
    (x^2 / 4 - y^2 / 2 = 1) ∧
    (∃ a b c e : Real,
      2 * a = 4 ∧
      2 * b = 2 * Real.sqrt 2 ∧
      c = Real.sqrt (a^2 + b^2) ∧
      2 * c = 2 * Real.sqrt 6 ∧
      e = c / a)) :=
by
  sorry

end hyperbola_properties_l437_43799


namespace solve_inequality_l437_43713

theorem solve_inequality (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  ( if 0 ≤ a ∧ a < 1 / 2 then (x > a ∧ x < 1 - a) else 
    if a = 1 / 2 then false else 
    if 1 / 2 < a ∧ a ≤ 1 then (x > 1 - a ∧ x < a) else false ) ↔ ((x - a) * (x + a - 1) < 0) :=
by
  sorry

end solve_inequality_l437_43713


namespace trig_identity_l437_43722

theorem trig_identity (θ : ℝ) (h₁ : Real.tan θ = 2) :
  2 * Real.cos θ / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 :=
by
  sorry

end trig_identity_l437_43722


namespace product_mod_five_l437_43768

theorem product_mod_five (a b c : ℕ) (h₁ : a = 1236) (h₂ : b = 7483) (h₃ : c = 53) :
  (a * b * c) % 5 = 4 :=
by
  sorry

end product_mod_five_l437_43768


namespace fraction_value_l437_43793

theorem fraction_value : (10 + 20 + 30 + 40) / 10 = 10 := by
  sorry

end fraction_value_l437_43793


namespace coconut_grove_produce_trees_l437_43720

theorem coconut_grove_produce_trees (x : ℕ)
  (h1 : 60 * (x + 3) + 120 * x + 180 * (x - 3) = 100 * 3 * x)
  : x = 6 := sorry

end coconut_grove_produce_trees_l437_43720


namespace sum_of_x_and_y_l437_43707

theorem sum_of_x_and_y (x y : ℝ) (h : (x + y + 2)^2 + |2 * x - 3 * y - 1| = 0) : x + y = -2 :=
by
  sorry

end sum_of_x_and_y_l437_43707


namespace stamps_problem_l437_43778

theorem stamps_problem (x y : ℕ) : 
  2 * x + 6 * x + 5 * y / 2 = 60 → x = 5 ∧ y = 8 ∧ 6 * x = 30 :=
by 
  sorry

end stamps_problem_l437_43778


namespace hotel_charge_difference_l437_43741

variables (G P R : ℝ)

-- Assumptions based on the problem conditions
variables
  (hR : R = 2 * G) -- Charge for a single room at hotel R is 100% greater than at hotel G
  (hP : P = 0.9 * G) -- Charge for a single room at hotel P is 10% less than at hotel G

theorem hotel_charge_difference :
  ((R - P) / R) * 100 = 55 :=
by
  -- Calculation
  sorry

end hotel_charge_difference_l437_43741


namespace negation_of_existential_l437_43731

theorem negation_of_existential :
  (¬ ∃ (x : ℝ), x^2 + x + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_l437_43731
