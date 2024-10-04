import Mathlib

namespace smallest_marble_count_l217_217542

theorem smallest_marble_count (N : ℕ) (a b c : ℕ) (h1 : N > 1)
  (h2 : N ≡ 2 [MOD 5])
  (h3 : N ≡ 2 [MOD 7])
  (h4 : N ≡ 2 [MOD 9]) : N = 317 :=
sorry

end smallest_marble_count_l217_217542


namespace prob_A_given_at_least_one_hit_l217_217234

theorem prob_A_given_at_least_one_hit (P_A P_B : ℝ) (hA : P_A = 0.6) (hB : P_B = 0.5) :
  let P_at_least_one_hit := P_A * (1 - P_B) + (1 - P_A) * P_B + P_A * P_B in
  let P_A_and_at_least_one_hit := P_A * (1 - P_B) + P_A * P_B in
  P_A_and_at_least_one_hit / P_at_least_one_hit = 3 / 4 :=
by {
  sorry
}

end prob_A_given_at_least_one_hit_l217_217234


namespace complex_point_in_fourth_quadrant_l217_217372

theorem complex_point_in_fourth_quadrant (z : ℂ) (h : z = 1 / (1 + I)) :
  z.re > 0 ∧ z.im < 0 :=
by
  -- Here we would provide the proof, but it is omitted as per the instructions.
  sorry

end complex_point_in_fourth_quadrant_l217_217372


namespace factor_quadratic_l217_217711

theorem factor_quadratic (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := 
by 
  sorry

end factor_quadratic_l217_217711


namespace vector_properties_l217_217726

noncomputable theory

open_locale real_inner_product_space

variables (a b c : ℝ × ℝ) (t : ℝ)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2)
variables (hab : ‖a - b‖ = real.sqrt 7)
variables (h_perp : inner_product_space.has_inner.smul_right_inner_product a c = 0)

-- Define the main theorem
theorem vector_properties :
  ∃ θ : ℝ, ∃ t : ℝ, (inner_product_space.cosine a b = -1 / 2) ∧ 
  (θ = 2 * real.pi / 3) ∧ (t = 1) ∧ (‖t • a + b‖ = real.sqrt 3) :=
by {
  sorry -- proof is omitted
}

end vector_properties_l217_217726


namespace original_players_count_l217_217230

theorem original_players_count (n : ℕ) (W : ℕ) :
  (W = n * 103) →
  ((W + 110 + 60) = (n + 2) * 99) →
  n = 7 :=
by sorry

end original_players_count_l217_217230


namespace line_equation_l217_217868

theorem line_equation (θ : Real) (b : Real) (h1 : θ = 45) (h2 : b = 2) : (y = x + b) :=
by
  -- Assume θ = 45°. The corresponding slope is k = tan(θ) = 1.
  -- Since the y-intercept b = 2, the equation of the line y = mx + b = x + 2.
  sorry

end line_equation_l217_217868


namespace no_solutions_to_equation_l217_217490

theorem no_solutions_to_equation :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ^ 2 - 2 * y ^ 2 = 5 := by
  sorry

end no_solutions_to_equation_l217_217490


namespace n_divides_2n_plus_1_implies_multiple_of_3_l217_217481

theorem n_divides_2n_plus_1_implies_multiple_of_3 {n : ℕ} (h₁ : n ≥ 2) (h₂ : n ∣ (2^n + 1)) : 3 ∣ n :=
sorry

end n_divides_2n_plus_1_implies_multiple_of_3_l217_217481


namespace apples_shared_equally_l217_217354

-- Definitions of the given conditions
def num_apples : ℕ := 9
def num_friends : ℕ := 3

-- Statement of the problem
theorem apples_shared_equally : num_apples / num_friends = 3 := by
  sorry

end apples_shared_equally_l217_217354


namespace sam_bikes_speed_l217_217849

noncomputable def EugeneSpeed : ℝ := 5
noncomputable def ClaraSpeed : ℝ := (3/4) * EugeneSpeed
noncomputable def SamSpeed : ℝ := (4/3) * ClaraSpeed

theorem sam_bikes_speed :
  SamSpeed = 5 :=
by
  -- Proof will be filled here.
  sorry

end sam_bikes_speed_l217_217849


namespace olga_fish_count_at_least_l217_217920

def number_of_fish (yellow blue green : ℕ) : ℕ :=
  yellow + blue + green

theorem olga_fish_count_at_least :
  ∃ (fish_count : ℕ), 
  (∃ (yellow blue green : ℕ), 
       yellow = 12 ∧ blue = yellow / 2 ∧ green = yellow * 2 ∧ fish_count = number_of_fish yellow blue green) ∧
  fish_count = 42 :=
by
  let yellow := 12
  let blue := yellow / 2
  let green := yellow * 2
  let fish_count := number_of_fish yellow blue green
  have h : fish_count = 42 := sorry
  use fish_count, yellow, blue, green
  repeat {constructor}
  assumption
  assumption
  assumption
  assumption
  assumption
  assumption

end olga_fish_count_at_least_l217_217920


namespace initial_population_l217_217070

theorem initial_population (rate_decrease : ℝ) (population_after_2_years : ℝ) (P : ℝ) : 
  rate_decrease = 0.1 → 
  population_after_2_years = 8100 → 
  ((1 - rate_decrease) ^ 2) * P = population_after_2_years → 
  P = 10000 :=
by
  intros h1 h2 h3
  sorry

end initial_population_l217_217070


namespace sum_of_first_nine_terms_l217_217956

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def a_n (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem sum_of_first_nine_terms (a₁ d : ℕ) (h : a_n a₁ d 2 + a_n a₁ d 6 + a_n a₁ d 7 = 18) :
  arithmetic_sequence_sum a₁ d 9 = 54 :=
sorry

end sum_of_first_nine_terms_l217_217956


namespace Joan_spent_on_shirt_l217_217040

/-- Joan spent $15 on shorts, $14.82 on a jacket, and a total of $42.33 on clothing.
    Prove that Joan spent $12.51 on the shirt. -/
theorem Joan_spent_on_shirt (shorts jacket total: ℝ) 
                            (h1: shorts = 15)
                            (h2: jacket = 14.82)
                            (h3: total = 42.33) :
  total - (shorts + jacket) = 12.51 :=
by
  sorry

end Joan_spent_on_shirt_l217_217040


namespace find_extra_digit_l217_217245

theorem find_extra_digit (x y a : ℕ) (hx : x + y = 23456) (h10x : 10 * x + a + y = 55555) (ha : 0 ≤ a ∧ a ≤ 9) : a = 5 :=
by
  sorry

end find_extra_digit_l217_217245


namespace master_craftsman_quota_l217_217892

theorem master_craftsman_quota (N : ℕ) (initial_rate increased_rate : ℕ) (additional_hours extra_hours : ℝ) :
  initial_rate = 35 →
  increased_rate = initial_rate + 15 →
  additional_hours = 0.5 →
  extra_hours = 1 →
  N / initial_rate - N / increased_rate = additional_hours + extra_hours →
  N = 175 →
  (initial_rate + N) = 210 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  rw h6,
  exact rfl,
}

end master_craftsman_quota_l217_217892


namespace freezer_temperature_l217_217331

theorem freezer_temperature 
  (refrigeration_temp : ℝ)
  (freezer_temp_diff : ℝ)
  (h1 : refrigeration_temp = 4)
  (h2 : freezer_temp_diff = 22)
  : (refrigeration_temp - freezer_temp_diff) = -18 :=
by 
  sorry

end freezer_temperature_l217_217331


namespace range_of_m_l217_217568

noncomputable def condition_p (x : ℝ) : Prop := -2 < x ∧ x < 10
noncomputable def condition_q (x m : ℝ) : Prop := (x - 1)^2 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, condition_p x → condition_q x m) ∧ (∃ x, ¬ condition_p x ∧ condition_q x m) ↔ 9 ≤ m := sorry

end range_of_m_l217_217568


namespace find_whole_number_N_l217_217404

theorem find_whole_number_N (N : ℕ) (h1 : 6.75 < (N / 4 : ℝ)) (h2 : (N / 4 : ℝ) < 7.25) : N = 28 := 
by 
  sorry

end find_whole_number_N_l217_217404


namespace rectangle_area_l217_217812

variable (L B : ℕ)

theorem rectangle_area :
  (L - B = 23) ∧ (2 * L + 2 * B = 166) → (L * B = 1590) :=
by
  sorry

end rectangle_area_l217_217812


namespace minimum_chocolates_l217_217621

theorem minimum_chocolates (x : ℤ) (h1 : x ≥ 150) (h2 : x % 15 = 7) : x = 157 :=
sorry

end minimum_chocolates_l217_217621


namespace student_ticket_cost_l217_217307

theorem student_ticket_cost 
  (total_tickets_sold : ℕ) 
  (total_revenue : ℕ) 
  (nonstudent_ticket_cost : ℕ) 
  (student_tickets_sold : ℕ) 
  (cost_per_student_ticket : ℕ) 
  (nonstudent_tickets_sold : ℕ) 
  (H1 : total_tickets_sold = 821) 
  (H2 : total_revenue = 1933)
  (H3 : nonstudent_ticket_cost = 3)
  (H4 : student_tickets_sold = 530) 
  (H5 : nonstudent_tickets_sold = total_tickets_sold - student_tickets_sold)
  (H6 : 530 * cost_per_student_ticket + nonstudent_tickets_sold * 3 = 1933) : 
  cost_per_student_ticket = 2 := 
by
  sorry

end student_ticket_cost_l217_217307


namespace Jake_peaches_l217_217904

variables (Jake Steven Jill : ℕ)

def peaches_relation : Prop :=
  (Jake = Steven - 6) ∧
  (Steven = Jill + 18) ∧
  (Jill = 5)

theorem Jake_peaches : peaches_relation Jake Steven Jill → Jake = 17 := by
  sorry

end Jake_peaches_l217_217904


namespace remainder_expression_div_10_l217_217326

theorem remainder_expression_div_10 (p t : ℕ) (hp : p > t) (ht : t > 1) :
  (92^p * 5^p + t + 11^t * 6^(p * t)) % 10 = 1 :=
by
  sorry

end remainder_expression_div_10_l217_217326


namespace quadratic_root_a_value_l217_217741

theorem quadratic_root_a_value (a : ℝ) (h : 2^2 - 2 * a + 6 = 0) : a = 5 :=
sorry

end quadratic_root_a_value_l217_217741


namespace triangle_circumradius_sqrt3_triangle_area_l217_217003

variables {a b c : ℝ} {A B C : ℝ} {R : ℝ}
variables (triangle_ABC : a = 2 * R * sin A ∧ c = 2 * R * sin C ∧ b = 2 * R * sin B)

theorem triangle_circumradius_sqrt3 (ha : a * sin C + sqrt 3 * c * cos A = 0) (hR : R = sqrt 3)
  (hsinC_nonzero : sin C ≠ 0) : a = 3 :=
begin
  sorry
end

theorem triangle_area (ha : a = 3) (hb : b + c = sqrt 11) (hA : A = 2 * real.pi / 3) : 
  (1/2)*b*c*sin A = sqrt 3 / 2 :=
begin
  sorry
end

end triangle_circumradius_sqrt3_triangle_area_l217_217003


namespace master_craftsman_quota_l217_217893

theorem master_craftsman_quota (N : ℕ) (initial_rate increased_rate : ℕ) (additional_hours extra_hours : ℝ) :
  initial_rate = 35 →
  increased_rate = initial_rate + 15 →
  additional_hours = 0.5 →
  extra_hours = 1 →
  N / initial_rate - N / increased_rate = additional_hours + extra_hours →
  N = 175 →
  (initial_rate + N) = 210 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  rw h6,
  exact rfl,
}

end master_craftsman_quota_l217_217893


namespace function_translation_l217_217798

def translateLeft (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x + a)
def translateUp (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x => (f x) + b

theorem function_translation :
  (translateUp (translateLeft (λ x => 2 * x^2) 1) 3) = λ x => 2 * (x + 1)^2 + 3 :=
by
  sorry

end function_translation_l217_217798


namespace abs_sum_bound_l217_217875

theorem abs_sum_bound (x : ℝ) (a : ℝ) (h : |x - 4| + |x - 3| < a) (ha : 0 < a) : 1 < a :=
by
  sorry

end abs_sum_bound_l217_217875


namespace radius_of_third_circle_l217_217642

theorem radius_of_third_circle (r₁ r₂ : ℝ) (r₁_val : r₁ = 23) (r₂_val : r₂ = 37) : 
  ∃ r : ℝ, r = 2 * Real.sqrt 210 :=
by
  sorry

end radius_of_third_circle_l217_217642


namespace bananas_to_oranges_equivalence_l217_217408

noncomputable def bananas_to_apples (bananas apples : ℕ) : Prop :=
  4 * apples = 3 * bananas

noncomputable def apples_to_oranges (apples oranges : ℕ) : Prop :=
  5 * oranges = 2 * apples

theorem bananas_to_oranges_equivalence (x y : ℕ) (hx : bananas_to_apples 24 x) (hy : apples_to_oranges x y) :
  y = 72 / 10 := by
  sorry

end bananas_to_oranges_equivalence_l217_217408


namespace total_stamps_l217_217050

def c : ℕ := 578833
def bw : ℕ := 523776
def total : ℕ := 1102609

theorem total_stamps : c + bw = total := 
by 
  sorry

end total_stamps_l217_217050


namespace negation_of_universal_proposition_l217_217213

theorem negation_of_universal_proposition (x : ℝ) :
  (¬ (∀ x : ℝ, |x| < 0)) ↔ (∃ x_0 : ℝ, |x_0| ≥ 0) := by
  sorry

end negation_of_universal_proposition_l217_217213


namespace lcm_of_20_45_75_l217_217652

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l217_217652


namespace problem_statement_l217_217721

variable {x y z : ℝ}

theorem problem_statement (h : x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0)
  (hne : ¬(x = y ∧ y = z)) (hpos : x > 0 ∧ y > 0 ∧ z > 0) :
  (x + y + z = 3) ∧ (x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6) :=
sorry

end problem_statement_l217_217721


namespace pen_tip_movement_l217_217613

-- Definition of movements
def move_left (x : Int) : Int := -x
def move_right (x : Int) : Int := x

theorem pen_tip_movement :
  move_left 6 + move_right 3 = -3 :=
by
  sorry

end pen_tip_movement_l217_217613


namespace distance_from_center_to_chord_l217_217993

theorem distance_from_center_to_chord (a b : ℝ) : 
  ∃ d : ℝ, d = (1/4) * |a - b| := 
sorry

end distance_from_center_to_chord_l217_217993


namespace lotion_cost_l217_217463

variable (shampoo_conditioner_cost lotion_total_spend: ℝ)
variable (num_lotions num_lotions_cost_target: ℕ)
variable (free_shipping_threshold additional_spend_needed: ℝ)

noncomputable def cost_of_each_lotion := lotion_total_spend / num_lotions

theorem lotion_cost
    (h1 : shampoo_conditioner_cost = 10)
    (h2 : num_lotions = 3)
    (h3 : additional_spend_needed = 12)
    (h4 : free_shipping_threshold = 50)
    (h5 : (shampoo_conditioner_cost * 2) + additional_spend_needed + lotion_total_spend = free_shipping_threshold) :
    cost_of_each_lotion = 10 :=
by
  sorry

end lotion_cost_l217_217463


namespace truthful_dwarfs_count_l217_217553

def dwarf (n : ℕ) := n < 10
def vanilla_ice_cream (n : ℕ) := dwarf n ∧ (∀ m, dwarf m)
def chocolate_ice_cream (n : ℕ) := dwarf n ∧ m % 2 = 0
def fruit_ice_cream (n : ℕ) := dwarf n ∧ m % 9 = 0

theorem truthful_dwarfs_count :
  ∃ T L : ℕ, T + L = 10 ∧ T + 2 * L = 16 ∧ T = 4 :=
by
  sorry

end truthful_dwarfs_count_l217_217553


namespace range_of_f_gt_f_2x_l217_217569

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * abs (x - 1)

theorem range_of_f_gt_f_2x :
  {x : ℝ | f x > f (2 * x)} = set.Ioo 0 (2 / 3) :=
by
  sorry

end range_of_f_gt_f_2x_l217_217569


namespace probability_heads_exactly_10_out_of_12_flips_l217_217083

theorem probability_heads_exactly_10_out_of_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := Nat.choose 12 10
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = (66 : ℝ) / 4096 :=
by
  let total_outcomes := 2^12
  let favorable_outcomes := Nat.choose 12 10
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  have total_outcomes_val : total_outcomes = 4096 := by norm_num
  have favorable_outcomes_val : favorable_outcomes = 66 := by norm_num
  have probability_val : probability = (66 : ℝ) / 4096 := by rw [favorable_outcomes_val, total_outcomes_val]
  exact probability_val

end probability_heads_exactly_10_out_of_12_flips_l217_217083


namespace yellow_pill_cost_22_5_l217_217410

-- Definitions based on conditions
def number_of_days := 3 * 7
def total_cost := 903
def daily_cost := total_cost / number_of_days
def blue_pill_cost (yellow_pill_cost : ℝ) := yellow_pill_cost - 2

-- Prove that the cost of one yellow pill is 22.5 dollars
theorem yellow_pill_cost_22_5 : 
  ∃ (yellow_pill_cost : ℝ), 
    number_of_days = 21 ∧
    total_cost = 903 ∧ 
    (∀ yellow_pill_cost, daily_cost = yellow_pill_cost + blue_pill_cost yellow_pill_cost → yellow_pill_cost = 22.5) :=
by 
  sorry

end yellow_pill_cost_22_5_l217_217410


namespace solve_congruence_l217_217941

theorem solve_congruence : ∃ n : ℕ, 0 ≤ n ∧ n < 43 ∧ 11 * n % 43 = 7 :=
by
  sorry

end solve_congruence_l217_217941


namespace option_B_not_well_defined_l217_217269

-- Definitions based on given conditions 
def is_well_defined_set (description : String) : Prop :=
  match description with
  | "All positive numbers" => True
  | "All elderly people" => False
  | "All real numbers that are not equal to 0" => True
  | "The four great inventions of ancient China" => True
  | _ => False

-- Theorem stating option B "All elderly people" is not a well-defined set
theorem option_B_not_well_defined : ¬ is_well_defined_set "All elderly people" :=
  by sorry

end option_B_not_well_defined_l217_217269


namespace kim_time_away_from_home_l217_217907

noncomputable def time_away_from_home (distance_to_friend : ℕ) (detour_percent : ℕ) (stay_time : ℕ) (speed_mph : ℕ) : ℕ :=
  let return_distance := distance_to_friend * (1 + detour_percent / 100)
  let total_distance := distance_to_friend + return_distance
  let driving_time := total_distance / speed_mph
  let driving_time_minutes := driving_time * 60
  driving_time_minutes + stay_time

theorem kim_time_away_from_home : 
  time_away_from_home 30 20 30 44 = 120 := 
by
  -- We will handle the proof here
  sorry

end kim_time_away_from_home_l217_217907


namespace shaded_region_area_l217_217887

open Real

noncomputable def semicircle_area (d : ℝ) : ℝ := (1 / 8) * π * d^2

theorem shaded_region_area :
  let UV := 3
  let VW := 5
  let WX := 4
  let XY := 6
  let YZ := 7
  let UZ := UV + VW + WX + XY + YZ
  let area_UZ := semicircle_area UZ
  let area_UV := semicircle_area UV
  let area_VW := semicircle_area VW
  let area_WX := semicircle_area WX
  let area_XY := semicircle_area XY
  let area_YZ := semicircle_area YZ
  area_UZ - (area_UV + area_VW + area_WX + area_XY + area_YZ) = (247/4) * π :=
sorry

end shaded_region_area_l217_217887


namespace determinant_scaled_matrix_l217_217426

example (x y z w : ℝ) (h : |Matrix![[x, y], [z, w]]| = 7) :
  |Matrix![[3 * x, 3 * y], [3 * z, 3 * w]]| = 9 * |Matrix![[x, y], [z, w]]| := by
  sorry

theorem determinant_scaled_matrix (x y z w : ℝ) (h : |Matrix![[x, y], [z, w]]| = 7) :
  |Matrix![[3 * x, 3 * y], [3 * z, 3 * w]]| = 63 := by
  rw [Matrix.det_smul, h]
  norm_num
  sorry

end determinant_scaled_matrix_l217_217426


namespace anna_stamp_count_correct_l217_217533

-- Defining the initial counts of stamps
def anna_initial := 37
def alison_initial := 28
def jeff_initial := 31

-- Defining the operations
def alison_gives_half_to_anna := alison_initial / 2
def anna_after_receiving_from_alison := anna_initial + alison_gives_half_to_anna
def anna_final := anna_after_receiving_from_alison - 2 + 1

-- Formalizing the proof problem
theorem anna_stamp_count_correct : anna_final = 50 := by
  -- proof omitted
  sorry

end anna_stamp_count_correct_l217_217533


namespace find_y_l217_217323

theorem find_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := 
sorry

end find_y_l217_217323


namespace fractional_equation_solution_l217_217787

theorem fractional_equation_solution (x : ℝ) (h₁ : x ≠ 0) : (1 / x = 2 / (x + 3)) → x = 3 := by
  sorry

end fractional_equation_solution_l217_217787


namespace exists_group_of_four_l217_217594

-- Assuming 21 students, and any three have done homework together exactly once in either mathematics or Russian.
-- We aim to prove there exists a group of four students such that any three of them have done homework together in the same subject.
noncomputable def students : Type := Fin 21

-- Define a predicate to show that three students have done homework together.
-- We use "math" and "russian" to denote the subjects.
inductive Subject
| math
| russian

-- Define a relation expressing that any three students have done exactly one subject homework together.
axiom homework_done (s1 s2 s3 : students) : Subject 

theorem exists_group_of_four :
  ∃ (a b c d : students), 
    (homework_done a b c = homework_done a b d) ∧
    (homework_done a b c = homework_done a c d) ∧
    (homework_done a b c = homework_done b c d) ∧
    (homework_done a b d = homework_done a c d) ∧
    (homework_done a b d = homework_done b c d) ∧
    (homework_done a c d = homework_done b c d) :=
sorry

end exists_group_of_four_l217_217594


namespace probability_diff_colors_l217_217839

theorem probability_diff_colors (total_balls red_balls white_balls selected_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_red : red_balls = 2)
  (h_white : white_balls = 2)
  (h_selected : selected_balls = 2) :
  (∃ P : ℚ, P = (red_balls.choose (selected_balls / 2) * white_balls.choose (selected_balls / 2)) / total_balls.choose selected_balls ∧ P = 2 / 3) :=
by 
  sorry

end probability_diff_colors_l217_217839


namespace pig_farm_fence_l217_217246

theorem pig_farm_fence (fenced_side : ℝ) (area : ℝ) 
  (h1 : fenced_side * 2 * fenced_side = area) 
  (h2 : area = 1250) :
  4 * fenced_side = 100 :=
by {
  sorry
}

end pig_farm_fence_l217_217246


namespace simplify_expression_l217_217484

theorem simplify_expression (x : ℝ) :
  4 * x - 8 * x ^ 2 + 10 - (5 - 4 * x + 8 * x ^ 2) = -16 * x ^ 2 + 8 * x + 5 :=
by
  sorry

end simplify_expression_l217_217484


namespace final_tree_count_l217_217228

noncomputable def current_trees : ℕ := 39
noncomputable def trees_planted_today : ℕ := 41
noncomputable def trees_planted_tomorrow : ℕ := 20

theorem final_tree_count : current_trees + trees_planted_today + trees_planted_tomorrow = 100 := by
  sorry

end final_tree_count_l217_217228


namespace part_a_part_b_l217_217211

/-- Part (a) statement: -/
theorem part_a (x : Fin 100 → ℕ) :
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) →
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) :=
by
  sorry

/-- Part (b) statement: -/
theorem part_b (x : Fin 100 → ℕ) :
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) →
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) :=
by
  sorry

end part_a_part_b_l217_217211


namespace largest_divisor_of_five_even_numbers_l217_217645

theorem largest_divisor_of_five_even_numbers (n : ℕ) (h₁ : n % 2 = 1) : 
  ∃ d, (∀ n, n % 2 = 1 → d ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) ∧ 
       (∀ d', (∀ n, n % 2 = 1 → d' ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) → d' ≤ d) ∧ 
       d = 480 := sorry

end largest_divisor_of_five_even_numbers_l217_217645


namespace lcm_of_20_45_75_l217_217656

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l217_217656


namespace volume_ratio_of_cones_l217_217265

theorem volume_ratio_of_cones (R : ℝ) (hR : 0 < R) :
  let circumference := 2 * Real.pi * R
  let sector1_circumference := (2 / 3) * circumference
  let sector2_circumference := (1 / 3) * circumference
  let r1 := sector1_circumference / (2 * Real.pi)
  let r2 := sector2_circumference / (2 * Real.pi)
  let s := R
  let h1 := Real.sqrt (R^2 - r1^2)
  let h2 := Real.sqrt (R^2 - r2^2)
  let V1 := (Real.pi * r1^2 * h1) / 3
  let V2 := (Real.pi * r2^2 * h2) / 3
  V1 / V2 = Real.sqrt 10 := 
by
  sorry

end volume_ratio_of_cones_l217_217265


namespace integer_solutions_to_equation_l217_217557

theorem integer_solutions_to_equation :
  ∃ (x y : ℤ), 2 * x^2 + 8 * y^2 = 17 * x * y - 423 ∧
               ((x = 11 ∧ y = 19) ∨ (x = -11 ∧ y = -19)) :=
by
  sorry

end integer_solutions_to_equation_l217_217557


namespace bees_second_day_l217_217769

-- Define the number of bees on the first day
def bees_on_first_day : ℕ := 144 

-- Define the multiplier for the second day
def multiplier : ℕ := 3

-- Define the number of bees on the second day
def bees_on_second_day : ℕ := bees_on_first_day * multiplier

-- Theorem stating the number of bees seen on the second day
theorem bees_second_day : bees_on_second_day = 432 := by
  -- Proof is pending.
  sorry

end bees_second_day_l217_217769


namespace kim_total_time_away_l217_217906

noncomputable def total_time_away (d : ℝ) (detour_percentage : ℝ) (time_at_friends : ℝ) (speed : ℝ) : ℝ :=
  let detour_distance := d * detour_percentage
  let total_return_distance := d + detour_distance
  let total_distance := d + total_return_distance
  let driving_time := total_distance / speed
  driving_time + time_at_friends

theorem kim_total_time_away :
  total_time_away 30 0.2 (30 / 60) 44 = 2 :=
by
  delta total_time_away -- unfold the definition of total_time_away
  simp only [div_eq_mul_inv]
  norm_num
  sorry

end kim_total_time_away_l217_217906


namespace base_eight_to_base_ten_l217_217240

theorem base_eight_to_base_ten (n : ℕ) (h : n = 4 * 8^1 + 7 * 8^0) : n = 39 := by
  sorry

end base_eight_to_base_ten_l217_217240


namespace sum_possible_values_k_l217_217035

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l217_217035


namespace final_purchase_price_correct_l217_217768

-- Definitions
def initial_house_value : ℝ := 100000
def profit_percentage_Mr_Brown : ℝ := 0.10
def renovation_percentage : ℝ := 0.05
def profit_percentage_Mr_Green : ℝ := 0.07
def loss_percentage_Mr_Brown : ℝ := 0.10

-- Calculations
def purchase_price_mr_brown : ℝ := initial_house_value * (1 + profit_percentage_Mr_Brown)
def total_cost_mr_brown : ℝ := purchase_price_mr_brown * (1 + renovation_percentage)
def purchase_price_mr_green : ℝ := total_cost_mr_brown * (1 + profit_percentage_Mr_Green)
def final_purchase_price_mr_brown : ℝ := purchase_price_mr_green * (1 - loss_percentage_Mr_Brown)

-- Statement to prove
theorem final_purchase_price_correct : 
  final_purchase_price_mr_brown = 111226.50 :=
by
  sorry -- Proof is omitted

end final_purchase_price_correct_l217_217768


namespace complex_number_solution_l217_217448

theorem complex_number_solution (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * ⟨1, -a⟩ = (2 : ℂ)) : a = 1 :=
sorry

end complex_number_solution_l217_217448


namespace p_at_zero_l217_217043

-- We state the conditions: p is a polynomial of degree 6, and p(3^n) = 1/(3^n) for n = 0 to 6
def p : Polynomial ℝ := sorry

axiom p_degree : p.degree = 6
axiom p_values : ∀ (n : ℕ), n ≤ 6 → p.eval (3^n) = 1 / (3^n)

-- We want to prove that p(0) = 29523 / 2187
theorem p_at_zero : p.eval 0 = 29523 / 2187 := by sorry

end p_at_zero_l217_217043


namespace measurable_masses_l217_217519

theorem measurable_masses (k : ℤ) (h : -121 ≤ k ∧ k ≤ 121) : 
  ∃ (a b c d e : ℤ), k = a * 1 + b * 3 + c * 9 + d * 27 + e * 81 ∧ 
  (a = -1 ∨ a = 0 ∨ a = 1) ∧
  (b = -1 ∨ b = 0 ∨ b = 1) ∧
  (c = -1 ∨ c = 0 ∨ c = 1) ∧
  (d = -1 ∨ d = 0 ∨ d = 1) ∧
  (e = -1 ∨ e = 0 ∨ e = 1) :=
sorry

end measurable_masses_l217_217519


namespace fish_added_l217_217735

theorem fish_added (T C : ℕ) (h1 : T + C = 20) (h2 : C = T - 4) : C = 8 :=
by
  sorry

end fish_added_l217_217735


namespace binomial_expansion_integer_exponents_terms_l217_217949

theorem binomial_expansion_integer_exponents_terms :
  let n := 8 in
  let terms_with_integer_exponents := 
    {r : ℕ | ∃ k : ℤ, n - 2 * r = 2 * k} in
  (terms_with_integer_exponents.card = 3) :=
by sorry

end binomial_expansion_integer_exponents_terms_l217_217949


namespace rose_bushes_planted_l217_217377

-- Define the conditions as variables
variable (current_bushes planted_bushes total_bushes : Nat)
variable (h1 : current_bushes = 2) (h2 : total_bushes = 6)
variable (h3 : total_bushes = current_bushes + planted_bushes)

theorem rose_bushes_planted : planted_bushes = 4 := by
  sorry

end rose_bushes_planted_l217_217377


namespace unique_solution_a_exists_l217_217185

open Real

noncomputable def equation (a x : ℝ) :=
  4 * a^2 + 3 * x * log x + 3 * (log x)^2 = 13 * a * log x + a * x

theorem unique_solution_a_exists : 
  ∃! a : ℝ, ∃ x : ℝ, 0 < x ∧ equation a x :=
sorry

end unique_solution_a_exists_l217_217185


namespace solve_fractional_equation_l217_217790

theorem solve_fractional_equation : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -3 → (1 / x = 2 / (x + 3) ↔ x = 3) :=
by
  intros x hx
  sorry

end solve_fractional_equation_l217_217790


namespace sum_of_three_numbers_l217_217248

theorem sum_of_three_numbers : 3.15 + 0.014 + 0.458 = 3.622 :=
by sorry

end sum_of_three_numbers_l217_217248


namespace find_largest_angle_l217_217882

noncomputable def largest_angle_in_convex_pentagon (x : ℝ) : Prop :=
  let angle1 := 2 * x + 2
  let angle2 := 3 * x - 3
  let angle3 := 4 * x + 4
  let angle4 := 6 * x - 6
  let angle5 := x + 5
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧
  max (max angle1 (max angle2 (max angle3 angle4))) angle5 = angle4 ∧
  angle4 = 195.75

theorem find_largest_angle (x : ℝ) : largest_angle_in_convex_pentagon x := by
  sorry

end find_largest_angle_l217_217882


namespace algebraic_expression_value_l217_217244

noncomputable def a := Real.sqrt 2 + 1
noncomputable def b := Real.sqrt 2 - 1

theorem algebraic_expression_value : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) = Real.sqrt 2 / 2 := by
  sorry

end algebraic_expression_value_l217_217244


namespace greatest_possible_sum_of_digits_l217_217857

theorem greatest_possible_sum_of_digits 
  (n : ℕ) (a b d : ℕ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_d : d ≠ 0)
  (h1 : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (d * ((10 ^ (3 * n1) - 1) / 9) - b * ((10 ^ n1 - 1) / 9) = a^3 * ((10^n1 - 1) / 9)^3) 
                      ∧ (d * ((10 ^ (3 * n2) - 1) / 9) - b * ((10 ^ n2 - 1) / 9) = a^3 * ((10^n2 - 1) / 9)^3)) : 
  a + b + d = 12 := 
sorry

end greatest_possible_sum_of_digits_l217_217857


namespace meeting_probability_l217_217690

noncomputable def probability_meeting_occurs : ℝ :=
  let x y w z : ℝ := sorry
  if h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 0 ≤ w ∧ w ≤ 3 ∧ 0 ≤ z ∧ z ≤ 3 ∧ z > x ∧ z > y ∧ z > w ∧ |x - y| ≤ 0.5 ∧ |x - w| ≤ 0.5 ∧ |y - w| ≤ 0.5
  then 1/6
  else 0

theorem meeting_probability : probability_meeting_occurs = 1/6 := 
  sorry

end meeting_probability_l217_217690


namespace sequence_bound_100_l217_217006

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = a (n - 1) + 1 / a (n - 1)

theorem sequence_bound_100 (a : ℕ → ℝ) (h : seq a) : 
  14 < a 100 ∧ a 100 < 18 := 
sorry

end sequence_bound_100_l217_217006


namespace find_a6_l217_217886

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem find_a6 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h2 : a 2 = 4) (h4 : a 4 = 2) : a 6 = 0 :=
by sorry

end find_a6_l217_217886


namespace area_of_triangle_AOB_l217_217749

def S (OA OB : ℝ) (angleAOB : ℝ) : ℝ := (1 / 2) * OA * OB * Real.sin angleAOB

theorem area_of_triangle_AOB :
  let OA := 2
  let OB := 4
  let angleAOB := π / 6
  S OA OB angleAOB = 2 :=
by
  let OA := 2
  let OB := 4
  let angleAOB := π / 6
  have h1 : S OA OB angleAOB = 1/2 * OA * OB * Real.sin angleAOB :=
    by rfl
  have h2 : S OA OB angleAOB = 1/2 * 2 * 4 * Real.sin (π / 6) :=
    by rw [h1, Real.sin_pi_div_six]
  have h3 : 1/2 * 2 * 4 * 1/2 = 2 :=
    by norm_num
  exact eq.trans h2 h3

end area_of_triangle_AOB_l217_217749


namespace combinations_of_letters_l217_217611

-- Definitions based on the conditions in the problem statement.
def word : List Char := ['B', 'I', 'O', 'L', 'O', 'G', 'Y']
def vowels : List Char := ['I', 'O', 'O']
def consonants : List Char := ['B', 'L', 'G', 'G']

-- The main theorem to prove.
theorem combinations_of_letters : 
  ∃ n : ℕ, n = 12 ∧ (∃ (vowel_combinations consonant_combinations : List (Finset Char)),
  vowel_combinations.length = 3 ∧ consonant_combinations.length = 4 
  ∧
  (vowel_combinations.product consonant_combinations).length = n) :=
sorry

end combinations_of_letters_l217_217611


namespace man_speed_is_correct_l217_217120

noncomputable def train_length : ℝ := 165
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def time_seconds : ℝ := 9

-- Function to convert speed from kmph to m/s
noncomputable def kmph_to_mps (speed_kmph: ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Function to convert speed from m/s to kmph
noncomputable def mps_to_kmph (speed_mps: ℝ) : ℝ :=
  speed_mps * 3600 / 1000

-- The speed of the train in m/s
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- The relative speed of the train with respect to the man in m/s
noncomputable def relative_speed_mps : ℝ := train_length / time_seconds

-- The speed of the man in m/s
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps

-- The speed of the man in kmph
noncomputable def man_speed_kmph : ℝ := mps_to_kmph man_speed_mps

-- The statement to be proved
theorem man_speed_is_correct : man_speed_kmph = 5.976 := 
sorry

end man_speed_is_correct_l217_217120


namespace sum_of_first_10_terms_of_arithmetic_sequence_l217_217866

theorem sum_of_first_10_terms_of_arithmetic_sequence :
  ∀ (a n : ℕ) (a₁ : ℤ) (d : ℤ),
  (d = -2) →
  (a₇ : ℤ := a₁ + 6 * d) →
  (a₃ : ℤ := a₁ + 2 * d) →
  (a₁₀ : ℤ := a₁ + 9 * d) →
  (a₇ * a₇ = a₃ * a₁₀) →
  (S₁₀ : ℤ := 10 * a₁ + 45 * d) →
  S₁₀ = 270 :=
by
  intros a n a₁ d hd ha₇ ha₃ ha₁₀ hgm hS₁₀
  sorry

end sum_of_first_10_terms_of_arithmetic_sequence_l217_217866


namespace fractions_with_smallest_difference_l217_217306

theorem fractions_with_smallest_difference 
    (x y : ℤ) 
    (f1 : ℚ := (x : ℚ) / 8) 
    (f2 : ℚ := (y : ℚ) / 13) 
    (h : abs (13 * x - 8 * y) = 1): 
    (f1 ≠ f2) ∧ abs ((x : ℚ) / 8 - (y : ℚ) / 13) = 1 / 104 :=
by
  sorry

end fractions_with_smallest_difference_l217_217306


namespace max_snowmen_l217_217685

theorem max_snowmen (snowballs : Finset ℕ) (h_mass_range : ∀ m ∈ snowballs, 1 ≤ m ∧ m ≤ 99)
  (h_size : snowballs.card = 99)
  (h_stackable : ∀ a b c ∈ snowballs, a ≥ b / 2 ∧ b ≥ c / 2 → a + b + c = a + b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ max_snowmen, max_snowmen = 24 :=
by
  sorry

end max_snowmen_l217_217685


namespace trigonometric_identity_l217_217859

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := 
by 
  sorry

end trigonometric_identity_l217_217859


namespace roots_geometric_progression_two_complex_conjugates_l217_217713

theorem roots_geometric_progression_two_complex_conjugates (a : ℝ) :
  (∃ b k : ℝ, b ≠ 0 ∧ k ≠ 0 ∧ (k + 1/ k = 2) ∧ 
    (b * (1 + k + 1/k) = 9) ∧ (b^2 * (k + 1 + 1/k) = 27) ∧ (b^3 = -a)) →
  a = -27 :=
by sorry

end roots_geometric_progression_two_complex_conjugates_l217_217713


namespace decrease_percent_revenue_l217_217225

theorem decrease_percent_revenue 
  (T C : ℝ) 
  (hT : T > 0) 
  (hC : C > 0) 
  (new_tax : ℝ := 0.65 * T) 
  (new_consumption : ℝ := 1.15 * C) 
  (original_revenue : ℝ := T * C) 
  (new_revenue : ℝ := new_tax * new_consumption) :
  100 * (original_revenue - new_revenue) / original_revenue = 25.25 :=
sorry

end decrease_percent_revenue_l217_217225


namespace area_of_rhombus_l217_217329

-- Given values for the diagonals of a rhombus.
def d1 : ℝ := 14
def d2 : ℝ := 24

-- The target statement we want to prove.
theorem area_of_rhombus : (d1 * d2) / 2 = 168 := by
  sorry

end area_of_rhombus_l217_217329


namespace LCM_20_45_75_is_900_l217_217659

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l217_217659


namespace train_probability_correct_l217_217837

/-- Define the necessary parameters and conditions --/
noncomputable def train_arrival_prob (train_start train_wait max_time_Alex max_time_train : ℝ) : ℝ :=
  let total_possible_area := max_time_Alex * max_time_train
  let overlap_area := (max_time_train - train_wait) * train_wait + (train_wait) * max_time_train / 2
  overlap_area / total_possible_area

/-- Main theorem stating that the probability is 3/10 --/
theorem train_probability_correct :
  train_arrival_prob 0 15 75 60 = 3 / 10 :=
by sorry

end train_probability_correct_l217_217837


namespace fractional_equation_solution_l217_217795

theorem fractional_equation_solution (x : ℝ) (hx : x ≠ -3) (h : 1/x = 2/(x+3)) : x = 3 :=
sorry

end fractional_equation_solution_l217_217795


namespace determine_set_B_l217_217205
open Set

/-- Given problem conditions and goal in Lean 4 -/
theorem determine_set_B (U A B : Set ℕ) (hU : U = { x | x < 10 } )
  (hA_inter_compl_B : A ∩ (U \ B) = {1, 3, 5, 7, 9} ) :
  B = {2, 4, 6, 8} :=
by
  sorry

end determine_set_B_l217_217205


namespace sum_of_first_nine_terms_l217_217318

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a n = a 1 + d * (n - 1)

variables (a : ℕ → ℝ) (h_seq : arithmetic_sequence a)

-- Given condition: a₂ + a₃ + a₇ + a₈ = 20
def condition : Prop := a 2 + a 3 + a 7 + a 8 = 20

-- Statement: Prove that the sum of the first 9 terms is 45
theorem sum_of_first_nine_terms (h : condition a) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 45 :=
by sorry

end sum_of_first_nine_terms_l217_217318


namespace simplify_expression_l217_217358

theorem simplify_expression :
  (Real.sqrt 15 + Real.sqrt 45 - (Real.sqrt (4/3) - Real.sqrt 108)) = 
  (Real.sqrt 15 + 3 * Real.sqrt 5 + 16 * Real.sqrt 3 / 3) :=
by
  sorry

end simplify_expression_l217_217358


namespace simplify_nested_sqrt_l217_217935

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l217_217935


namespace sum_of_possible_ks_l217_217026

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l217_217026


namespace race_distance_l217_217452

theorem race_distance (T_A T_B : ℝ) (D : ℝ) (V_A V_B : ℝ)
  (h1 : T_A = 23)
  (h2 : T_B = 30)
  (h3 : V_A = D / 23)
  (h4 : V_B = (D - 56) / 30)
  (h5 : D = (D - 56) * (23 / 30) + 56) :
  D = 56 :=
by
  sorry

end race_distance_l217_217452


namespace probability_of_choosing_A_on_second_day_l217_217842

-- Definitions of the probabilities given in the problem conditions.
def p_first_day_A := 0.5
def p_first_day_B := 0.5
def p_second_day_A_given_first_day_A := 0.6
def p_second_day_A_given_first_day_B := 0.5

-- Define the problem to be proved in Lean 4
theorem probability_of_choosing_A_on_second_day :
  (p_first_day_A * p_second_day_A_given_first_day_A) +
  (p_first_day_B * p_second_day_A_given_first_day_B) = 0.55 :=
by
  sorry

end probability_of_choosing_A_on_second_day_l217_217842


namespace ellipse_hyperbola_tangent_n_value_l217_217144

theorem ellipse_hyperbola_tangent_n_value :
  (∃ n : ℝ, (∀ x y : ℝ, 4 * x^2 + y^2 = 4 ∧ x^2 - n * (y - 1)^2 = 1) ↔ n = 3 / 2) :=
by
  sorry

end ellipse_hyperbola_tangent_n_value_l217_217144


namespace max_area_ABC_l217_217000

-- Definitions of the given conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 2)
def C : {p : ℝ × ℝ // p.1^2 + p.2^2 - 2 * p.1 = 0}

-- The theorem statement
theorem max_area_ABC : 
  ∃ C : {p : ℝ × ℝ // p.1^2 + p.2^2 - 2 * p.1 = 0}, 
  (area_triangle A B C) = 3 + real.sqrt 2 :=
sorry

end max_area_ABC_l217_217000


namespace unbounded_n_satisfies_modified_triangle_property_l217_217520

theorem unbounded_n_satisfies_modified_triangle_property
  (n : ℕ) (T : finset ℕ) (hT : ∀ (y ∈ T), y ≥ 5 ∧ y ≤ n ∧ y ≠ 5 ∧ y ≠ 6 ∧ y ≠ 7)
  (ht : T.card = 10) :
  ∃ (a b c : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a^2 + b^2 = c^2 :=
begin
  sorry
end

end unbounded_n_satisfies_modified_triangle_property_l217_217520


namespace value_of_n_l217_217588

theorem value_of_n (n : ℤ) :
  (∀ x : ℤ, (x + n) * (x + 2) = x^2 + 2 * x + n * x + 2 * n → 2 + n = 0) → n = -2 := 
by
  intro h
  have h1 := h 0
  sorry

end value_of_n_l217_217588


namespace percentage_for_overnight_stays_l217_217758

noncomputable def total_bill : ℝ := 5000
noncomputable def medication_percentage : ℝ := 0.50
noncomputable def food_cost : ℝ := 175
noncomputable def ambulance_cost : ℝ := 1700

theorem percentage_for_overnight_stays :
  let medication_cost := medication_percentage * total_bill
  let remaining_bill := total_bill - medication_cost
  let cost_for_overnight_stays := remaining_bill - food_cost - ambulance_cost
  (cost_for_overnight_stays / remaining_bill) * 100 = 25 :=
by
  sorry

end percentage_for_overnight_stays_l217_217758


namespace smallest_prime_with_digit_sum_23_l217_217088

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Prime p ∧ digit_sum p = 23 ∧ ∀ q : ℕ, Prime q ∧ digit_sum q = 23 → p ≤ q :=
by
  sorry

end smallest_prime_with_digit_sum_23_l217_217088


namespace unique_combination_of_segments_l217_217746

theorem unique_combination_of_segments :
  ∃! (x y : ℤ), 7 * x + 12 * y = 100 := sorry

end unique_combination_of_segments_l217_217746


namespace sufficient_but_not_necessary_l217_217431

-- Definitions for lines and planes
def line : Type := ℝ × ℝ × ℝ
def plane : Type := ℝ × ℝ × ℝ × ℝ

-- Predicate for perpendicularity of a line to a plane
def perp_to_plane (l : line) (α : plane) : Prop := sorry

-- Predicate for parallelism of two planes
def parallel_planes (α β : plane) : Prop := sorry

-- Predicate for perpendicularity of two lines
def perp_lines (l m : line) : Prop := sorry

-- Predicate for a line being parallel to a plane
def parallel_to_plane (m : line) (β : plane) : Prop := sorry

-- Given conditions
variable (l : line)
variable (m : line)
variable (alpha : plane)
variable (beta : plane)
variable (H1 : perp_to_plane l alpha) -- l ⊥ α
variable (H2 : parallel_to_plane m beta) -- m ∥ β

-- Theorem statement
theorem sufficient_but_not_necessary :
  (parallel_planes alpha beta → perp_lines l m) ∧ ¬(perp_lines l m → parallel_planes alpha beta) :=
sorry

end sufficient_but_not_necessary_l217_217431


namespace conditions_not_sufficient_nor_necessary_l217_217867

theorem conditions_not_sufficient_nor_necessary (a : ℝ) (b : ℝ) :
  (a ≠ 5) ∧ (b ≠ -5) ↔ ¬((a ≠ 5) ∨ (b ≠ -5)) ∧ (a + b ≠ 0) := 
sorry

end conditions_not_sufficient_nor_necessary_l217_217867


namespace boy_present_age_l217_217247

-- Define the boy's present age
variable (x : ℤ)

-- Conditions from the problem statement
def condition_one : Prop :=
  x + 4 = 2 * (x - 6)

-- Prove that the boy's present age is 16
theorem boy_present_age (h : condition_one x) : x = 16 := 
sorry

end boy_present_age_l217_217247


namespace stratified_sampling_second_grade_l217_217785

theorem stratified_sampling_second_grade (r1 r2 r3 : ℕ) (total_sample : ℕ) (total_ratio : ℕ):
  r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ total_sample = 50 ∧ total_ratio = r1 + r2 + r3 →
  (r2 * total_sample) / total_ratio = 15 :=
by
  sorry

end stratified_sampling_second_grade_l217_217785


namespace determine_OP_l217_217772

variable (a b c d : ℝ)
variable (O A B C D P : ℝ)
variable (p : ℝ)

def OnLine (O A B C D P : ℝ) : Prop := O < A ∧ A < B ∧ B < C ∧ C < D ∧ B < P ∧ P < C

theorem determine_OP (h : OnLine O A B C D P) 
(hAP : P - A = p - a) 
(hPD : D - P = d - p) 
(hBP : P - B = p - b) 
(hPC : C - P = c - p) 
(hAP_PD_BP_PC : (p - a) / (d - p) = (p - b) / (c - p)) :
  p = (a * c - b * d) / (a - b + c - d) :=
sorry

end determine_OP_l217_217772


namespace triangle_right_angle_l217_217121

theorem triangle_right_angle {A B C : ℝ} 
  (h1 : A + B + C = 180)
  (h2 : A = B)
  (h3 : A = (1/2) * C) :
  C = 90 :=
by 
  sorry

end triangle_right_angle_l217_217121


namespace minimum_value_expression_l217_217914

theorem minimum_value_expression (x y z : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2) ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2)) :=
sorry

end minimum_value_expression_l217_217914


namespace trigonometric_identity_proof_l217_217427

theorem trigonometric_identity_proof (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = - (Real.sqrt 3 + 2) / 3 :=
by
  sorry

end trigonometric_identity_proof_l217_217427


namespace contradiction_even_odd_l217_217076

theorem contradiction_even_odd (a b c : ℕ) (h1 : (a % 2 = 1 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ c % 2 = 1) ∨ (b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1)) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  -- proof by contradiction
  sorry

end contradiction_even_odd_l217_217076


namespace quarters_needed_to_buy_items_l217_217524

-- Define the costs of each item in cents
def cost_candy_bar : ℕ := 25
def cost_chocolate : ℕ := 75
def cost_juice : ℕ := 50

-- Define the quantities of each item
def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Define the value of a quarter in cents
def value_of_quarter : ℕ := 25

-- Define the total cost of the items
def total_cost : ℕ := (num_candy_bars * cost_candy_bar) + (num_chocolates * cost_chocolate) + (num_juice_packs * cost_juice)

-- Calculate the number of quarters needed
def num_quarters_needed : ℕ := total_cost / value_of_quarter

-- The theorem to prove that the number of quarters needed is 11
theorem quarters_needed_to_buy_items : num_quarters_needed = 11 := by
  -- Proof omitted
  sorry

end quarters_needed_to_buy_items_l217_217524


namespace fraction_relationships_l217_217737

variable (p r s u : ℚ)

theorem fraction_relationships (h1 : p / r = 8) (h2 : s / r = 5) (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 :=
sorry

end fraction_relationships_l217_217737


namespace master_craftsman_quota_l217_217888

theorem master_craftsman_quota (parts_first_hour : ℕ)
  (extra_hour_needed : ℕ)
  (increased_speed : ℕ)
  (time_diff : ℕ)
  (total_parts : ℕ) :
  parts_first_hour = 35 →
  extra_hour_needed = 1 →
  increased_speed = 15 →
  time_diff = 1.5 →
  total_parts = parts_first_hour + (175 : ℕ) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3]
  norm_num
  rw [add_comm]
  exact sorry

end master_craftsman_quota_l217_217888


namespace cheryl_material_left_l217_217133

theorem cheryl_material_left (
  h₁ : (5 : ℚ) / 11 + 2 / 3 = 37 / 33) 
  (h₂ : 0.6666666666666665 = 2 / 3 : ℚ)
  (total_used : 2 / 3 : ℚ)
  (h₃ : (37 / 33) - total_used = 5 / 11) : 
  ((5 : ℚ) / 11 + 2 / 3 - 2 / 3 = 5 / 11) :=
by {
  subst h₂,
  rw h₁,
  exact h₃,
}

end cheryl_material_left_l217_217133


namespace shortest_path_correct_l217_217779

noncomputable def shortest_path_length (length width height : ℕ) : ℝ :=
  let diagonal := Real.sqrt ((length + height)^2 + width^2)
  Real.sqrt 145

theorem shortest_path_correct :
  ∀ (length width height : ℕ),
    length = 4 → width = 5 → height = 4 →
    shortest_path_length length width height = Real.sqrt 145 :=
by
  intros length width height h1 h2 h3
  rw [h1, h2, h3]
  sorry

end shortest_path_correct_l217_217779


namespace ivan_income_tax_l217_217752

theorem ivan_income_tax :
  let salary_probation := 20000
  let probation_months := 2
  let salary_after_probation := 25000
  let after_probation_months := 8
  let bonus := 10000
  let tax_rate := 0.13
  let total_income := salary_probation * probation_months +
                      salary_after_probation * after_probation_months + bonus
  total_income * tax_rate = 32500 := sorry

end ivan_income_tax_l217_217752


namespace partition_sum_le_152_l217_217001

theorem partition_sum_le_152 {S : ℕ} (l : List ℕ) 
  (h1 : ∀ n ∈ l, 1 ≤ n ∧ n ≤ 10) 
  (h2 : l.sum = S) : 
  (∃ l1 l2 : List ℕ, l1.sum ≤ 80 ∧ l2.sum ≤ 80 ∧ l1 ++ l2 = l) ↔ S ≤ 152 := 
by
  sorry

end partition_sum_le_152_l217_217001


namespace total_toothpicks_correct_l217_217080

def number_of_horizontal_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height + 1) * width

def number_of_vertical_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height) * (width + 1)

def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
number_of_horizontal_toothpicks height width + number_of_vertical_toothpicks height width

theorem total_toothpicks_correct:
  total_toothpicks 30 15 = 945 :=
by
  sorry

end total_toothpicks_correct_l217_217080


namespace cos_angle_identity_l217_217164

theorem cos_angle_identity (a : ℝ) (h : Real.sin (π / 6 - a) - Real.cos a = 1 / 3) :
  Real.cos (2 * a + π / 3) = 7 / 9 :=
by
  sorry

end cos_angle_identity_l217_217164


namespace unique_solution_value_k_l217_217413

theorem unique_solution_value_k (k : ℚ) :
  (∀ x : ℚ, (x + 3) / (k * x - 2) = x → x = -2) ↔ k = -3 / 4 :=
by
  sorry

end unique_solution_value_k_l217_217413


namespace right_triangle_perimeter_l217_217117

theorem right_triangle_perimeter (area leg1 : ℕ) (h_area : area = 180) (h_leg1 : leg1 = 30) :
  ∃ leg2 hypotenuse perimeter, 
    (2 * area = leg1 * leg2) ∧ 
    (hypotenuse^2 = leg1^2 + leg2^2) ∧ 
    (perimeter = leg1 + leg2 + hypotenuse) ∧ 
    (perimeter = 42 + 2 * Real.sqrt 261) :=
by
  sorry

end right_triangle_perimeter_l217_217117


namespace chicken_bucket_feeds_l217_217681

theorem chicken_bucket_feeds :
  ∀ (cost_per_bucket : ℝ) (total_cost : ℝ) (total_people : ℕ),
  cost_per_bucket = 12 →
  total_cost = 72 →
  total_people = 36 →
  (total_people / (total_cost / cost_per_bucket)) = 6 :=
by
  intros cost_per_bucket total_cost total_people h1 h2 h3
  sorry

end chicken_bucket_feeds_l217_217681


namespace a_lt_sqrt3b_l217_217863

open Int

theorem a_lt_sqrt3b (a b : ℤ) (h1 : a > b) (h2 : b > 1) 
    (h3 : a + b ∣ a * b + 1) (h4 : a - b ∣ a * b - 1) : a < sqrt 3 * b :=
  sorry

end a_lt_sqrt3b_l217_217863


namespace simplify_sqrt_expression_l217_217926

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l217_217926


namespace sled_distance_in_40_seconds_l217_217266

noncomputable def sled_distance (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n / 2) * (2 * a + (n - 1) * d)

theorem sled_distance_in_40_seconds :
  sled_distance 8 10 40 = 8120 :=
by
  -- sum of the first 40 terms of the sequence: 8, 18, 28, ..., up to the 40th term.
  sorry

end sled_distance_in_40_seconds_l217_217266


namespace four_numbers_sum_divisible_by_2016_l217_217510

theorem four_numbers_sum_divisible_by_2016 {x : Fin 65 → ℕ} (h_distinct: Function.Injective x) (h_range: ∀ i, x i ≤ 2016) :
  ∃ a b c d : Fin 65, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (x a + x b - x c - x d) % 2016 = 0 :=
by
  -- Proof omitted
  sorry

end four_numbers_sum_divisible_by_2016_l217_217510


namespace molecular_weight_proof_l217_217803

noncomputable def molecular_weight_C7H6O2 := 
  (7 * 12.01) + (6 * 1.008) + (2 * 16.00) -- molecular weight of one mole of C7H6O2

noncomputable def total_molecular_weight_9_moles := 
  9 * molecular_weight_C7H6O2 -- total molecular weight of 9 moles of C7H6O2

theorem molecular_weight_proof : 
  total_molecular_weight_9_moles = 1099.062 := 
by
  sorry

end molecular_weight_proof_l217_217803


namespace xiao_wang_parts_processed_l217_217389

-- Definitions for the processing rates and conditions
def xiao_wang_rate := 15 -- parts per hour
def xiao_wang_max_continuous_hours := 2
def xiao_wang_break_hours := 1

def xiao_li_rate := 12 -- parts per hour

-- Constants for the problem setup
def xiao_wang_process_time := 4 -- hours including breaks after first cycle
def xiao_li_process_time := 5 -- hours including no breaks

-- Total parts processed by both when they finish simultaneously
def parts_processed_when_finished_simultaneously := 60

theorem xiao_wang_parts_processed :
  (xiao_wang_rate * xiao_wang_max_continuous_hours) * (xiao_wang_process_time / 
  (xiao_wang_max_continuous_hours + xiao_wang_break_hours)) =
  parts_processed_when_finished_simultaneously :=
sorry

end xiao_wang_parts_processed_l217_217389


namespace base_eight_to_base_ten_l217_217236

theorem base_eight_to_base_ten : ∃ n : ℕ, 47 = 4 * 8 + 7 ∧ n = 39 :=
by
  sorry

end base_eight_to_base_ten_l217_217236


namespace total_cost_of_fencing_l217_217419

def diameter : ℝ := 28
def cost_per_meter : ℝ := 1.50
def pi_approx : ℝ := 3.14159

noncomputable def circumference : ℝ := pi_approx * diameter
noncomputable def total_cost : ℝ := circumference * cost_per_meter

theorem total_cost_of_fencing : total_cost = 131.94 :=
by
  sorry

end total_cost_of_fencing_l217_217419


namespace lcm_of_20_45_75_l217_217654

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l217_217654


namespace series_2023_power_of_3_squared_20_equals_653_l217_217253

def series (A : ℕ → ℕ) : Prop :=
  A 0 = 1 ∧ 
  ∀ n > 0, 
  A n = A (n / 2023) + A (n / 2023^2) + A (n / 2023^3)

theorem series_2023_power_of_3_squared_20_equals_653 (A : ℕ → ℕ) (h : series A) : A (2023 ^ (3^2) + 20) = 653 :=
by
  -- placeholder for proof
  sorry

end series_2023_power_of_3_squared_20_equals_653_l217_217253


namespace algebraic_sum_of_coefficients_l217_217544

open Nat

theorem algebraic_sum_of_coefficients
  (u : ℕ → ℤ)
  (h1 : u 1 = 5)
  (hrec : ∀ n : ℕ, n > 0 → u (n + 1) - u n = 3 + 4 * (n - 1)) :
  (∃ P : ℕ → ℤ, (∀ n, u n = P n) ∧ (P 1 + P 0 = 5)) :=
sorry

end algebraic_sum_of_coefficients_l217_217544


namespace largest_stamps_per_page_l217_217340

theorem largest_stamps_per_page (a b c : ℕ) (h1 : a = 924) (h2 : b = 1260) (h3 : c = 1386) : 
  Nat.gcd (Nat.gcd a b) c = 42 := by
  sorry

end largest_stamps_per_page_l217_217340


namespace average_sale_six_months_l217_217105

-- Define the sales for the first five months
def sale_month1 : ℕ := 6335
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562

-- Define the required sale for the sixth month
def sale_month6 : ℕ := 5091

-- Proof that the desired average sale for the six months is 6500
theorem average_sale_six_months : 
  (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6500 :=
by
  sorry

end average_sale_six_months_l217_217105


namespace no_sum_of_two_squares_l217_217980

theorem no_sum_of_two_squares (n : ℤ) (h : n % 4 = 3) : ¬∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end no_sum_of_two_squares_l217_217980


namespace fraction_to_decimal_l217_217320

theorem fraction_to_decimal : (5 : ℚ) / 8 = 0.625 := by
  -- Prove that the fraction 5/8 equals the decimal 0.625
  sorry

end fraction_to_decimal_l217_217320


namespace triangle_inequality_l217_217402

theorem triangle_inequality
  (a b c : ℝ)
  (habc : ¬(a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a)) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := 
by {
  sorry
}

end triangle_inequality_l217_217402


namespace tv_weight_difference_l217_217843

noncomputable def BillTV_width : ℝ := 48
noncomputable def BillTV_height : ℝ := 100
noncomputable def BobTV_width : ℝ := 70
noncomputable def BobTV_height : ℝ := 60
noncomputable def weight_per_sq_inch : ℝ := 4
noncomputable def ounces_per_pound : ℝ := 16

theorem tv_weight_difference :
  let BillTV_area := BillTV_width * BillTV_height,
      BillTV_weight_oz := BillTV_area * weight_per_sq_inch,
      BillTV_weight_lb := BillTV_weight_oz / ounces_per_pound,
      BobTV_area := BobTV_width * BobTV_height,
      BobTV_weight_oz := BobTV_area * weight_per_sq_inch,
      BobTV_weight_lb := BobTV_weight_oz / ounces_per_pound
  in BillTV_weight_lb - BobTV_weight_lb = 150 :=
by
  sorry

end tv_weight_difference_l217_217843


namespace parallel_condition_l217_217728

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x^2, 4 * x)

-- Define the condition for parallelism for two-dimensional vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- Define the theorem to prove
theorem parallel_condition (x : ℝ) :
  parallel (vector_a x) (vector_b x) ↔ |x| = 2 :=
by {
  sorry
}

end parallel_condition_l217_217728


namespace right_triangle_side_product_l217_217964

theorem right_triangle_side_product :
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  (hypotenuse * 2 * Real.sqrt 7).round = 53 := -- using 53 to consider rounding to the nearest tenth

by
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  have h1 : hypotenuse = 10 := by sorry
  have h2 : other_leg = 2 * Real.sqrt 7 := by sorry
  have h_prod : (hypotenuse * 2 * Real.sqrt 7).round = 53 := by sorry
  exact h_prod

end right_triangle_side_product_l217_217964


namespace find_value_of_t_l217_217550

variable (a b v d t r : ℕ)

-- All variables are non-zero digits (1-9)
axiom non_zero_a : 0 < a ∧ a < 10
axiom non_zero_b : 0 < b ∧ b < 10
axiom non_zero_v : 0 < v ∧ v < 10
axiom non_zero_d : 0 < d ∧ d < 10
axiom non_zero_t : 0 < t ∧ t < 10
axiom non_zero_r : 0 < r ∧ r < 10

-- Given conditions
axiom condition1 : a + b = v
axiom condition2 : v + d = t
axiom condition3 : t + a = r
axiom condition4 : b + d + r = 18

theorem find_value_of_t : t = 9 :=
by sorry

end find_value_of_t_l217_217550


namespace find_T_l217_217959

theorem find_T (T : ℝ) (h : (3/4) * (1/8) * T = (1/2) * (1/6) * 72) : T = 64 :=
by {
  -- proof goes here
  sorry
}

end find_T_l217_217959


namespace perfect_square_expression_l217_217099

theorem perfect_square_expression (x y z : ℤ) :
    9 * (x^2 + y^2 + z^2)^2 - 8 * (x + y + z) * (x^3 + y^3 + z^3 - 3 * x * y * z) =
      ((x + y + z)^2 - 6 * (x * y + y * z + z * x))^2 := 
by 
  sorry

end perfect_square_expression_l217_217099


namespace Jean_average_speed_correct_l217_217543

noncomputable def Jean_avg_speed_until_meet
    (total_distance : ℕ)
    (chantal_flat_distance : ℕ)
    (chantal_flat_speed : ℕ)
    (chantal_steep_distance : ℕ)
    (chantal_steep_ascend_speed : ℕ)
    (chantal_steep_descend_distance : ℕ)
    (chantal_steep_descend_speed : ℕ)
    (jean_meet_position_ratio : ℚ) : ℚ :=
  let chantal_flat_time := (chantal_flat_distance : ℚ) / chantal_flat_speed
  let chantal_steep_ascend_time := (chantal_steep_distance : ℚ) / chantal_steep_ascend_speed
  let chantal_steep_descend_time := (chantal_steep_descend_distance : ℚ) / chantal_steep_descend_speed
  let total_time_until_meet := chantal_flat_time + chantal_steep_ascend_time + chantal_steep_descend_time
  let jean_distance_until_meet := (jean_meet_position_ratio * chantal_steep_distance : ℚ) + chantal_flat_distance
  jean_distance_until_meet / total_time_until_meet

theorem Jean_average_speed_correct :
  Jean_avg_speed_until_meet 6 3 5 3 3 1 4 (1 / 3) = 80 / 37 :=
by
  sorry

end Jean_average_speed_correct_l217_217543


namespace customer_wants_score_of_eggs_l217_217102

def Score := 20
def Dozen := 12

def options (n : Nat) : Prop :=
  n = Score ∨ n = 2 * Score ∨ n = 2 * Dozen ∨ n = 3 * Score

theorem customer_wants_score_of_eggs : 
  ∃ n, options n ∧ n = Score := 
by
  exists Score
  constructor
  apply Or.inl
  rfl
  rfl

end customer_wants_score_of_eggs_l217_217102


namespace equal_real_roots_possible_values_l217_217015

theorem equal_real_roots_possible_values (a : ℝ): 
  (∀ x : ℝ, x^2 + a * x + 1 = 0) → (a = 2 ∨ a = -2) :=
by
  sorry

end equal_real_roots_possible_values_l217_217015


namespace base_eight_to_base_ten_l217_217238

theorem base_eight_to_base_ten : ∃ n : ℕ, 47 = 4 * 8 + 7 ∧ n = 39 :=
by
  sorry

end base_eight_to_base_ten_l217_217238


namespace more_people_this_week_l217_217459

-- Define the conditions
variables (second_game first_game third_game : ℕ)
variables (total_last_week total_this_week : ℕ)

-- Conditions
def condition1 : Prop := second_game = 80
def condition2 : Prop := first_game = second_game - 20
def condition3 : Prop := third_game = second_game + 15
def condition4 : Prop := total_last_week = 200
def condition5 : Prop := total_this_week = second_game + first_game + third_game

-- Theorem statement
theorem more_people_this_week (h1 : condition1)
                             (h2 : condition2)
                             (h3 : condition3)
                             (h4 : condition4)
                             (h5 : condition5) : total_this_week - total_last_week = 35 :=
sorry

end more_people_this_week_l217_217459


namespace find_m_l217_217451

theorem find_m (x y m : ℝ) 
  (h1 : x + y = 8)
  (h2 : y - m * x = 7)
  (h3 : y - x = 7.5) : m = 3 := 
  sorry

end find_m_l217_217451


namespace sum_of_squares_of_distances_l217_217194

-- Definitions based on the conditions provided:
variables (A B C D X : Point)
variable (a : ℝ)
variable (h1 h2 h3 h4 : ℝ)

-- Conditions:
axiom square_side_length : a = 5
axiom area_ratios : (1/2 * a * h1) / (1/2 * a * h2) = 1 / 5 ∧ 
                    (1/2 * a * h2) / (1/2 * a * h3) = 5 / 9

-- Problem Statement to Prove:
theorem sum_of_squares_of_distances :
  h1^2 + h2^2 + h3^2 + h4^2 = 33 :=
sorry

end sum_of_squares_of_distances_l217_217194


namespace range_of_m_l217_217316

theorem range_of_m (m x : ℝ) : (m-1 < x ∧ x < m+1) → (1/3 < x ∧ x < 1/2) → (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  intros h1 h2
  have h3 : 1/3 < m + 1 := by sorry
  have h4 : m - 1 < 1/2 := by sorry
  have h5 : -1/2 ≤ m := by sorry
  have h6 : m ≤ 4/3 := by sorry
  exact ⟨h5, h6⟩

end range_of_m_l217_217316


namespace clock_chime_time_l217_217231

theorem clock_chime_time (t : ℕ) (h : t = 12) (k : 4 * (t / (4 - 1)) = 12) :
  12 * (t / (4 - 1)) - (12 - 1) * (t / (4 - 1)) = 44 :=
by {
  sorry
}

end clock_chime_time_l217_217231


namespace molecular_weight_CaSO4_2H2O_l217_217547

def Ca := 40.08
def S := 32.07
def O := 16.00
def H := 1.008

def Ca_weight := 1 * Ca
def S_weight := 1 * S
def O_in_sulfate_weight := 4 * O
def O_in_water_weight := 4 * O
def H_in_water_weight := 4 * H

def total_weight := Ca_weight + S_weight + O_in_sulfate_weight + O_in_water_weight + H_in_water_weight

theorem molecular_weight_CaSO4_2H2O : total_weight = 204.182 := 
by {
  sorry
}

end molecular_weight_CaSO4_2H2O_l217_217547


namespace joy_can_choose_17_rods_for_quadrilateral_l217_217759

theorem joy_can_choose_17_rods_for_quadrilateral :
  ∃ (possible_rods : Finset ℕ), 
    possible_rods.card = 17 ∧
    ∀ rod ∈ possible_rods, 
      rod > 0 ∧ rod <= 30 ∧
      (rod ≠ 3 ∧ rod ≠ 7 ∧ rod ≠ 15) ∧
      (rod > 15 - (3 + 7)) ∧
      (rod < 3 + 7 + 15) :=
by
  sorry

end joy_can_choose_17_rods_for_quadrilateral_l217_217759


namespace age_ratio_l217_217709

theorem age_ratio (darcie_age : ℕ) (father_age : ℕ) (mother_ratio : ℚ) (mother_fraction : ℚ)
  (h1 : darcie_age = 4)
  (h2 : father_age = 30)
  (h3 : mother_ratio = 4/5)
  (h4 : mother_fraction = mother_ratio * father_age)
  (h5 : mother_fraction = 24) :
  (darcie_age : ℚ) / mother_fraction = 1 / 6 :=
by
  sorry

end age_ratio_l217_217709


namespace proposition_form_l217_217780

-- Definitions based on the conditions
def p : Prop := (12 % 4 = 0)
def q : Prop := (12 % 3 = 0)

-- Problem statement to prove
theorem proposition_form : p ∧ q :=
by
  sorry

end proposition_form_l217_217780


namespace broken_seashells_l217_217380

-- Define the total number of seashells Tom found
def total_seashells : ℕ := 7

-- Define the number of unbroken seashells
def unbroken_seashells : ℕ := 3

-- Prove that the number of broken seashells equals 4
theorem broken_seashells : total_seashells - unbroken_seashells = 4 := by
  sorry

end broken_seashells_l217_217380


namespace N_subset_M_values_l217_217727

def M : Set ℝ := { x | 2 * x^2 - 3 * x - 2 = 0 }
def N (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem N_subset_M_values (a : ℝ) (h : N a ⊆ M) : a = 0 ∨ a = -2 ∨ a = 1/2 := 
by
  sorry

end N_subset_M_values_l217_217727


namespace correct_option_D_l217_217806

theorem correct_option_D (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by sorry

end correct_option_D_l217_217806


namespace lisa_punch_l217_217765

theorem lisa_punch (x : ℝ) (H : x = 0.125) :
  (0.3 + x) / (2 + x) = 0.20 :=
by
  sorry

end lisa_punch_l217_217765


namespace elixir_concentration_l217_217438

theorem elixir_concentration (x a : ℝ) 
  (h1 : (x * 100) / (100 + a) = 9) 
  (h2 : (x * 100 + a * 100) / (100 + 2 * a) = 23) : 
  x = 11 :=
by 
  sorry

end elixir_concentration_l217_217438


namespace expression_value_l217_217344

theorem expression_value (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x :=
by
  -- Insert proof here
  sorry

end expression_value_l217_217344


namespace product_of_base_9_digits_of_9876_l217_217667

def base9_digits (n : ℕ) : List ℕ := 
  let rec digits_aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else digits_aux (n / 9) ((n % 9) :: acc)
  digits_aux n []

def product (lst : List ℕ) : ℕ := lst.foldl (· * ·) 1

theorem product_of_base_9_digits_of_9876 :
  product (base9_digits 9876) = 192 :=
by 
  sorry

end product_of_base_9_digits_of_9876_l217_217667


namespace sin_alpha_plus_beta_alpha_plus_two_beta_l217_217865

variables {α β : ℝ} (hα_acute : 0 < α ∧ α < π / 2) (hβ_acute : 0 < β ∧ β < π / 2)
          (h_tan_α : Real.tan α = 1 / 7) (h_sin_β : Real.sin β = Real.sqrt 10 / 10)

theorem sin_alpha_plus_beta : 
    Real.sin (α + β) = Real.sqrt 5 / 5 :=
by
  sorry

theorem alpha_plus_two_beta : 
    α + 2 * β = π / 4 :=
by
  sorry

end sin_alpha_plus_beta_alpha_plus_two_beta_l217_217865


namespace find_number_l217_217011

theorem find_number : ∀ (x : ℝ), (0.15 * 0.30 * 0.50 * x = 99) → (x = 4400) :=
by
  intro x
  intro h
  sorry

end find_number_l217_217011


namespace area_between_curves_l217_217822

-- Function definitions:
def quartic (a b c d e x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e
def line (p q x : ℝ) : ℝ := p * x + q

-- Conditions:
variables (a b c d e p q α β : ℝ)
variable (a_ne_zero : a ≠ 0)
variable (α_lt_β : α < β)
variable (touch_at_α : quartic a b c d e α = line p q α ∧ deriv (quartic a b c d e) α = p)
variable (touch_at_β : quartic a b c d e β = line p q β ∧ deriv (quartic a b c d e) β = p)

-- Theorem:
theorem area_between_curves :
  ∫ x in α..β, |quartic a b c d e x - line p q x| = (a * (β - α)^5) / 30 :=
by sorry

end area_between_curves_l217_217822


namespace infinite_geometric_series_sum_l217_217301

-- First term of the geometric series
def a : ℚ := 5/3

-- Common ratio of the geometric series
def r : ℚ := -1/4

-- The sum of the infinite geometric series
def S : ℚ := a / (1 - r)

-- Prove that the sum of the series is equal to 4/3
theorem infinite_geometric_series_sum : S = 4/3 := by
  sorry

end infinite_geometric_series_sum_l217_217301


namespace solve_x_l217_217736

theorem solve_x : ∀ (x y : ℝ), (3 * x - y = 7) ∧ (x + 3 * y = 6) → x = 27 / 10 :=
by
  intros x y h
  sorry

end solve_x_l217_217736


namespace evaluate_product_l217_217850

theorem evaluate_product (m : ℕ) (h : m = 3) : (m - 2) * (m - 1) * m * (m + 1) * (m + 2) * (m + 3) = 720 :=
by {
  sorry
}

end evaluate_product_l217_217850


namespace correct_proposition_l217_217910

variable (a b : ℝ)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)
variable (a_gt_b : a > b)

theorem correct_proposition : 1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end correct_proposition_l217_217910


namespace fraction_option_C_l217_217094

def is_fraction (expr : String) : Prop := 
  expr = "fraction"

def option_C_fraction (x : ℝ) : Prop :=
  ∃ (numerator : ℝ), ∃ (denominator : ℝ), 
  numerator = 2 ∧ denominator = x + 3

theorem fraction_option_C (x : ℝ) (h : x ≠ -3) :
  is_fraction "fraction" ↔ option_C_fraction x :=
by 
  sorry

end fraction_option_C_l217_217094


namespace smallest_b_undefined_inverse_l217_217243

theorem smallest_b_undefined_inverse (b : ℕ) (h1 : Nat.gcd b 84 > 1) (h2 : Nat.gcd b 90 > 1) : b = 6 :=
sorry

end smallest_b_undefined_inverse_l217_217243


namespace total_doughnuts_l217_217210

-- Definitions used in the conditions
def boxes : ℕ := 4
def doughnuts_per_box : ℕ := 12

theorem total_doughnuts : boxes * doughnuts_per_box = 48 :=
by
  sorry

end total_doughnuts_l217_217210


namespace c_is_younger_l217_217509

variables (a b c d : ℕ) -- assuming ages as natural numbers

-- Conditions
axiom cond1 : a + b = b + c + 12
axiom cond2 : b + d = c + d + 8
axiom cond3 : d = a + 5

-- Question
theorem c_is_younger : c = a - 12 :=
sorry

end c_is_younger_l217_217509


namespace Dawn_has_10_CDs_l217_217197

-- Lean definition of the problem conditions
def Kristine_more_CDs (D K : ℕ) : Prop :=
  K = D + 7

def Total_CDs (D K : ℕ) : Prop :=
  D + K = 27

-- Lean statement of the proof
theorem Dawn_has_10_CDs (D K : ℕ) (h1 : Kristine_more_CDs D K) (h2 : Total_CDs D K) : D = 10 :=
by
  sorry

end Dawn_has_10_CDs_l217_217197


namespace hannah_strawberries_l217_217579

theorem hannah_strawberries (days give_away stolen remaining_strawberries x : ℕ) 
  (h1 : days = 30) 
  (h2 : give_away = 20) 
  (h3 : stolen = 30) 
  (h4 : remaining_strawberries = 100) 
  (hx : x = (remaining_strawberries + give_away + stolen) / days) : 
  x = 5 := 
by 
  -- The proof will go here
  sorry

end hannah_strawberries_l217_217579


namespace overall_average_length_of_ropes_l217_217229

theorem overall_average_length_of_ropes :
  let ropes := 6
  let third_part := ropes / 3
  let average1 := 70
  let average2 := 85
  let length1 := third_part * average1
  let length2 := (ropes - third_part) * average2
  let total_length := length1 + length2
  let overall_average := total_length / ropes
  overall_average = 80 := by
sorry

end overall_average_length_of_ropes_l217_217229


namespace large_bucket_capacity_l217_217396

variable (S L : ℕ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
sorry

end large_bucket_capacity_l217_217396


namespace number_of_principals_in_oxford_high_school_l217_217921

-- Define the conditions
def numberOfTeachers : ℕ := 48
def numberOfClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def totalStudents : ℕ := numberOfClasses * studentsPerClass
def totalPeople : ℕ := 349
def numberOfPrincipals : ℕ := totalPeople - (numberOfTeachers + totalStudents)

-- Proposition: Prove the number of principals in Oxford High School
theorem number_of_principals_in_oxford_high_school :
  numberOfPrincipals = 1 := by sorry

end number_of_principals_in_oxford_high_school_l217_217921


namespace fish_added_l217_217734

theorem fish_added (T C : ℕ) (h1 : T + C = 20) (h2 : C = T - 4) : C = 8 :=
by
  sorry

end fish_added_l217_217734


namespace solve_rational_numbers_l217_217418

theorem solve_rational_numbers:
  ∃ (a b c d : ℚ),
    8 * a^2 - 3 * b^2 + 5 * c^2 + 16 * d^2 - 10 * a * b + 42 * c * d + 18 * a + 22 * b - 2 * c - 54 * d = 42 ∧
    15 * a^2 - 3 * b^2 + 21 * c^2 - 5 * d^2 + 4 * a * b + 32 * c * d - 28 * a + 14 * b - 54 * c - 52 * d = -22 ∧
    a = 4 / 7 ∧ b = 19 / 7 ∧ c = 29 / 19 ∧ d = -6 / 19 :=
  sorry

end solve_rational_numbers_l217_217418


namespace scientific_notation_316000000_l217_217232

theorem scientific_notation_316000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 316000000 = a * 10 ^ n ∧ a = 3.16 ∧ n = 8 :=
by
  -- Proof would be here
  sorry

end scientific_notation_316000000_l217_217232


namespace initial_items_in_cart_l217_217529

theorem initial_items_in_cart (deleted_items : ℕ) (items_left : ℕ) (initial_items : ℕ) 
  (h1 : deleted_items = 10) (h2 : items_left = 8) : initial_items = 18 :=
by 
  -- Proof goes here
  sorry

end initial_items_in_cart_l217_217529


namespace proposition_1_proposition_2_proposition_3_proposition_4_l217_217292

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l217_217292


namespace quarters_needed_to_buy_items_l217_217523

-- Define the costs of each item in cents
def cost_candy_bar : ℕ := 25
def cost_chocolate : ℕ := 75
def cost_juice : ℕ := 50

-- Define the quantities of each item
def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Define the value of a quarter in cents
def value_of_quarter : ℕ := 25

-- Define the total cost of the items
def total_cost : ℕ := (num_candy_bars * cost_candy_bar) + (num_chocolates * cost_chocolate) + (num_juice_packs * cost_juice)

-- Calculate the number of quarters needed
def num_quarters_needed : ℕ := total_cost / value_of_quarter

-- The theorem to prove that the number of quarters needed is 11
theorem quarters_needed_to_buy_items : num_quarters_needed = 11 := by
  -- Proof omitted
  sorry

end quarters_needed_to_buy_items_l217_217523


namespace perfect_square_digits_l217_217784

theorem perfect_square_digits (x y : ℕ) (h_ne_zero : x ≠ 0) (h_perfect_square : ∀ n: ℕ, n ≥ 1 → ∃ k: ℕ, (10^(n + 2) * x + 10^(n + 1) * 6 + 10 * y + 4) = k^2) :
  (x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0) :=
sorry

end perfect_square_digits_l217_217784


namespace max_value_theta_argz_l217_217567

open Complex

theorem max_value_theta_argz (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) :
  (let z := (⟨3 * Real.cos θ, 2 * Real.sin θ⟩ : ℂ),
       y := θ - Complex.arg z in
   θ = Real.arctan (Real.sqrt 3 / 2)) :=
by 
  let z := (⟨3 * Real.cos θ, 2 * Real.sin θ⟩ : ℂ),
      y := θ - Complex.arg z
  sorry

end max_value_theta_argz_l217_217567


namespace batsman_sixes_l217_217989

theorem batsman_sixes (total_runs : ℕ) (boundaries : ℕ) (running_percentage : ℝ) (score_per_boundary : ℕ) (score_per_six : ℕ)
  (h1 : total_runs = 150)
  (h2 : boundaries = 5)
  (h3 : running_percentage = 66.67)
  (h4 : score_per_boundary = 4)
  (h5 : score_per_six = 6) :
  ∃ (sixes : ℕ), sixes = 5 :=
by
  -- Calculations omitted
  existsi 5
  sorry

end batsman_sixes_l217_217989


namespace sixty_three_times_fifty_seven_l217_217142

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 :=
by
  let a := 60
  let b := 3
  have h : (a + b) * (a - b) = a^2 - b^2 := by sorry
  have h1 : 63 = a + b := by rfl
  have h2 : 57 = a - b := by rfl
  calc
    63 * 57 = (a + b) * (a - b) : by rw [h1, h2]
    ... = a^2 - b^2 : by rw h
    ... = 60^2 - 3^2 : by rfl
    ... = 3600 - 9 : by sorry
    ... = 3591 : by norm_num

end sixty_three_times_fifty_seven_l217_217142


namespace sum_f_alpha_beta_gamma_neg_l217_217870

theorem sum_f_alpha_beta_gamma_neg (f : ℝ → ℝ)
  (h_f : ∀ x, f x = -x - x^3)
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 := 
sorry

end sum_f_alpha_beta_gamma_neg_l217_217870


namespace find_balanced_grid_pairs_l217_217069

-- Define a balanced grid condition
def is_balanced_grid (m n : ℕ) (grid : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, i < m → j < n →
    (∀ k, k < m → grid i k = grid i j) ∧ (∀ l, l < n → grid l j = grid i j)

-- Main theorem statement
theorem find_balanced_grid_pairs (m n : ℕ) :
  (∃ grid, is_balanced_grid m n grid) ↔ (m = n ∨ m = n / 2 ∨ n = 2 * m) :=
by
  sorry

end find_balanced_grid_pairs_l217_217069


namespace radius_of_circumcircle_l217_217816

-- Definitions of sides of a triangle and its area
variables {a b c t : ℝ}

-- Condition that t is the area of a triangle with sides a, b, and c
def is_triangle_area (a b c t : ℝ) : Prop := -- Placeholder condition stating these values form a triangle
sorry

-- Statement to prove the given radius formula for the circumscribed circle
theorem radius_of_circumcircle (h : is_triangle_area a b c t) : 
  ∃ r : ℝ, r = abc / (4 * t) :=
sorry

end radius_of_circumcircle_l217_217816


namespace base_eight_to_base_ten_l217_217241

theorem base_eight_to_base_ten (n : ℕ) (h : n = 4 * 8^1 + 7 * 8^0) : n = 39 := by
  sorry

end base_eight_to_base_ten_l217_217241


namespace min_chemistry_teachers_l217_217264

/--
A school has 7 maths teachers, 6 physics teachers, and some chemistry teachers.
Each teacher can teach a maximum of 3 subjects.
The minimum number of teachers required is 6.
Prove that the minimum number of chemistry teachers required is 1.
-/
theorem min_chemistry_teachers (C : ℕ) (math_teachers : ℕ := 7) (physics_teachers : ℕ := 6) 
  (max_subjects_per_teacher : ℕ := 3) (min_teachers_required : ℕ := 6) :
  7 + 6 + C ≤ 6 * 3 → C = 1 := 
by
  sorry

end min_chemistry_teachers_l217_217264


namespace factor_quadratic_l217_217712

theorem factor_quadratic (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := 
by 
  sorry

end factor_quadratic_l217_217712


namespace sale_in_second_month_l217_217828

def sale_first_month : ℕ := 6435
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6191
def average_sale : ℕ := 6700

theorem sale_in_second_month : 
  ∀ (sale_second_month : ℕ), 
    (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month = 6700 * 6) → 
    sale_second_month = 6927 :=
by
  intro sale_second_month h
  sorry

end sale_in_second_month_l217_217828


namespace ivan_income_tax_l217_217751

noncomputable def personalIncomeTax (monthly_salary: ℕ → ℕ) (bonus: ℕ) (tax_rate: ℚ) : ℕ :=
  let taxable_income := (monthly_salary 3 + monthly_salary 4) +
                       (List.sum (List.map monthly_salary [5, 6, 7, 8, 9, 10, 11, 12])) +
                       bonus
  in taxable_income * tax_rate

theorem ivan_income_tax :
  personalIncomeTax
    (λ m, if m ∈ [3, 4] then 20000 else if m ∈ [5, 6, 7, 8, 9, 10, 11, 12] then 25000 else 0)
    10000 0.13 = 32500 :=
  sorry

end ivan_income_tax_l217_217751


namespace simplify_sqrt_sum_l217_217937

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l217_217937


namespace minimal_n_is_40_l217_217005

def sequence_minimal_n (p : ℝ) (a : ℕ → ℝ) : Prop :=
  a 1 = p ∧
  a 2 = p + 1 ∧
  (∀ n, n ≥ 1 → a (n + 2) - 2 * a (n + 1) + a n = n - 20) ∧
  (∀ n, a n ≥ p) -- Since minimal \(a_n\) implies non-negative with given \(a_1, a_2\)

theorem minimal_n_is_40 (p : ℝ) (a : ℕ → ℝ) (h : sequence_minimal_n p a) : ∃ n, n = 40 ∧ (∀ m, n ≠ m → a n ≤ a m) :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end minimal_n_is_40_l217_217005


namespace bill_buys_125_bouquets_to_make_1000_l217_217278

-- Defining the conditions
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_difference : ℕ := 1000

-- To be demonstrated: number of bouquets Bill needs to buy to get a profit difference of $1000
theorem bill_buys_125_bouquets_to_make_1000 : 
  let total_cost_to_buy (n : ℕ) := n * cost_per_bouquet
  let total_roses (n : ℕ) := n * roses_per_bouquet_buy
  let bouquets_sell_from_roses (roses : ℕ) := roses / roses_per_bouquet_sell
  let total_revenue (bouquets : ℕ) := bouquets * cost_per_bouquet
  let profit (n : ℕ) := total_revenue (bouquets_sell_from_roses (total_roses n)) - total_cost_to_buy n
  profit (125 * 5) = target_difference := 
sorry

end bill_buys_125_bouquets_to_make_1000_l217_217278


namespace circle_touching_y_axis_radius_5_k_value_l217_217160

theorem circle_touching_y_axis_radius_5_k_value :
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) →
    (∃ r : ℝ, r = 5 ∧ (∀ c : ℝ × ℝ, (c.1 + 4)^2 + (c.2 + 2)^2 = r^2) ∧
      (∃ x : ℝ, x + 4 = 0)) :=
by
  sorry

end circle_touching_y_axis_radius_5_k_value_l217_217160


namespace acute_triangle_conditions_l217_217974

-- Definitions exclusively from the conditions provided.
def condition_A (AB AC : ℝ) : Prop :=
  AB * AC > 0

def condition_B (sinA sinB sinC : ℝ) : Prop :=
  sinA / sinB = 4 / 5 ∧ sinA / sinC = 4 / 6 ∧ sinB / sinC = 5 / 6

def condition_C (cosA cosB cosC : ℝ) : Prop :=
  cosA * cosB * cosC > 0

def condition_D (tanA tanB : ℝ) : Prop :=
  tanA * tanB = 2

-- Prove which conditions guarantee that triangle ABC is acute.
theorem acute_triangle_conditions (AB AC sinA sinB sinC cosA cosB cosC tanA tanB : ℝ) :
  (condition_B sinA sinB sinC ∨ condition_C cosA cosB cosC ∨ condition_D tanA tanB) →
  (∀ (A B C : ℝ), A < π / 2 ∧ B < π / 2 ∧ C < π / 2) :=
sorry

end acute_triangle_conditions_l217_217974


namespace inequality_proof_l217_217010

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : (1 / x) < (1 / y) :=
by
  sorry

end inequality_proof_l217_217010


namespace parts_manufactured_l217_217894

variable (initial_parts : ℕ) (initial_rate : ℕ) (increased_speed : ℕ) (time_diff : ℝ)
variable (N : ℕ)

-- initial conditions
def initial_parts := 35
def initial_rate := 35
def increased_speed := 15
def time_diff := 1.5

-- additional parts to be manufactured
noncomputable def additional_parts := N

-- equation representing the time differences
noncomputable def equation := (N / initial_rate) - (N / (initial_rate + increased_speed)) = time_diff

-- state the proof problem
theorem parts_manufactured : initial_parts + additional_parts = 210 :=
by
  -- Use the given conditions to solve the problem
  sorry

end parts_manufactured_l217_217894


namespace rice_containers_l217_217775

theorem rice_containers (pound_to_ounce : ℕ) (total_rice_lb : ℚ) (container_oz : ℕ) : 
  pound_to_ounce = 16 → 
  total_rice_lb = 33 / 4 → 
  container_oz = 33 → 
  (total_rice_lb * pound_to_ounce) / container_oz = 4 :=
by sorry

end rice_containers_l217_217775


namespace rabbits_ate_27_watermelons_l217_217604

theorem rabbits_ate_27_watermelons
  (original_watermelons : ℕ)
  (watermelons_left : ℕ)
  (watermelons_eaten : ℕ)
  (h1 : original_watermelons = 35)
  (h2 : watermelons_left = 8)
  (h3 : original_watermelons - watermelons_left = watermelons_eaten) :
  watermelons_eaten = 27 :=
by {
  -- Proof skipped
  sorry
}

end rabbits_ate_27_watermelons_l217_217604


namespace line_passes_second_and_third_quadrants_l217_217564

theorem line_passes_second_and_third_quadrants 
  (a b c p : ℝ)
  (h1 : a * b * c ≠ 0)
  (h2 : (a + b) / c = p)
  (h3 : (b + c) / a = p)
  (h4 : (c + a) / b = p) :
  ∀ (x y : ℝ), y = p * x + p → 
  ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
sorry

end line_passes_second_and_third_quadrants_l217_217564


namespace find_roots_of_polynomial_l217_217716

noncomputable def polynomial_roots : Set ℝ :=
  {x | (6 * x^4 + 25 * x^3 - 59 * x^2 + 28 * x) = 0 }

theorem find_roots_of_polynomial :
  polynomial_roots = {0, 1, (-31 + Real.sqrt 1633) / 12, (-31 - Real.sqrt 1633) / 12} :=
by
  sorry

end find_roots_of_polynomial_l217_217716


namespace power_of_two_sequence_invariant_l217_217199

theorem power_of_two_sequence_invariant
  (n : ℕ)
  (a b : ℕ → ℕ)
  (h₀ : a 0 = 1)
  (h₁ : b 0 = n)
  (hi : ∀ i : ℕ, a i < b i → a (i + 1) = 2 * a i + 1 ∧ b (i + 1) = b i - a i - 1)
  (hj : ∀ i : ℕ, a i > b i → a (i + 1) = a i - b i - 1 ∧ b (i + 1) = 2 * b i + 1)
  (hk : ∀ i : ℕ, a i = b i → a (i + 1) = a i ∧ b (i + 1) = b i)
  (k : ℕ)
  (h : a k = b k) :
  ∃ m : ℕ, n + 3 = 2 ^ m :=
by
  sorry

end power_of_two_sequence_invariant_l217_217199


namespace binomial_coefficient_7_5_permutation_7_5_l217_217845

-- Define function for binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define function for permutation calculation
def permutation (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

theorem binomial_coefficient_7_5 : binomial_coefficient 7 5 = 21 :=
by
  sorry

theorem permutation_7_5 : permutation 7 5 = 2520 :=
by
  sorry

end binomial_coefficient_7_5_permutation_7_5_l217_217845


namespace power_function_value_l217_217224

theorem power_function_value (f : ℝ → ℝ) (h : ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a) (h₁ : f 4 = 1 / 2) :
  f (1 / 16) = 4 :=
sorry

end power_function_value_l217_217224


namespace simplify_sqrt_sum_l217_217928

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l217_217928


namespace batsman_average_after_17th_inning_l217_217978

theorem batsman_average_after_17th_inning 
    (A : ℕ)  -- assuming A (the average before the 17th inning) is a natural number
    (h₁ : 16 * A + 85 = 17 * (A + 3)) : 
    A + 3 = 37 := by
  sorry

end batsman_average_after_17th_inning_l217_217978


namespace total_number_of_people_l217_217796

theorem total_number_of_people (c a : ℕ) (h1 : c = 2 * a) (h2 : c = 28) : c + a = 42 :=
by
  sorry

end total_number_of_people_l217_217796


namespace round_2741836_to_nearest_integer_l217_217922

theorem round_2741836_to_nearest_integer :
  (2741836.4928375).round = 2741836 := 
by
  -- Explanation that 0.4928375 < 0.5 leading to rounding down
  sorry

end round_2741836_to_nearest_integer_l217_217922


namespace evaluate_expression_l217_217417

theorem evaluate_expression (a b c : ℤ)
  (h1 : c = b - 12)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3 : ℚ) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 7 / 3 := 
sorry

end evaluate_expression_l217_217417


namespace min_value_a_plus_b_l217_217042

theorem min_value_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Real.sqrt (3^a * 3^b) = 3^((a + b) / 2)) : a + b = 4 := by
  sorry

end min_value_a_plus_b_l217_217042


namespace ratio_QP_l217_217628

theorem ratio_QP {P Q : ℚ} 
  (h : ∀ x : ℝ, x ≠ 0 → x ≠ 4 → x ≠ -4 → 
    P / (x^2 - 5 * x) + Q / (x + 4) = (x^2 - 3 * x + 8) / (x^3 - 5 * x^2 + 4 * x)) : 
  Q / P = 7 / 2 := 
sorry

end ratio_QP_l217_217628


namespace find_x_if_vectors_are_parallel_l217_217319

noncomputable def vector_a : ℝ × ℝ := (1, 1) 
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

noncomputable def vector_sum (x : ℝ) : ℝ × ℝ := (vector_a.1 + (vector_b x).1, vector_a.2 + (vector_b x).2)
noncomputable def vector_diff (x : ℝ) : ℝ × ℝ := (vector_a.1 - (vector_b x).1, vector_a.2 - (vector_b x).2)

theorem find_x_if_vectors_are_parallel (x : ℝ) : 
  vector_sum x = (3, x + 1) → 
  vector_diff x = (-1, 1 - x) → 
  vector_sum x.1 * vector_diff x.2 - vector_sum x.2 * vector_diff x.1 = 0 → 
  x = 2 := 
by 
  sorry

end find_x_if_vectors_are_parallel_l217_217319


namespace angle_B_is_arcsin_l217_217902

-- Define the triangle and its conditions
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  ∀ (A B C : ℝ), 
    a = 8 ∧ b = Real.sqrt 3 ∧ 
    (2 * Real.cos (A - B) / 2 ^ 2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)

-- Prove that the measure of ∠B is arcsin(√3 / 10)
theorem angle_B_is_arcsin (A B C : ℝ) (a b c : ℝ) (h : triangle_ABC A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
sorry

end angle_B_is_arcsin_l217_217902


namespace num_arrangements_teachers_students_l217_217535

theorem num_arrangements_teachers_students : 
  (∃ (teachers students : Type) (teacherGroup1 teacherGroup2 : teachers) (studentGroupA studentGroupB studentGroupC studentGroupD : students), 
    (∀ (locationA_teachers locationB_teachers : set teachers)
       (locationA_students locationB_students : set students),
      (locationA_teachers ∪ locationB_teachers = {teacherGroup1, teacherGroup2}) ∧
      (locationA_teachers ∩ locationB_teachers = ∅) ∧
      (locationA_students ∪ locationB_students = {studentGroupA, studentGroupB, studentGroupC, studentGroupD}) ∧
      (locationA_students ∩ locationB_students = ∅) ∧
      locationA_teachers.card = 1 ∧
      locationB_teachers.card = 1 ∧
      locationA_students.card = 2 ∧
      locationB_students.card = 2)
  ) → choose 2 1 * choose 4 2 = 12 := 
by sorry

end num_arrangements_teachers_students_l217_217535


namespace log_cut_piece_weight_l217_217467

-- Defining the conditions

def log_length : ℕ := 20
def half_log_length : ℕ := log_length / 2
def weight_per_foot : ℕ := 150

-- The main theorem stating the problem
theorem log_cut_piece_weight : (half_log_length * weight_per_foot) = 1500 := 
by 
  sorry

end log_cut_piece_weight_l217_217467


namespace fraction_equality_l217_217913

theorem fraction_equality
  (a b : ℝ)
  (x : ℝ)
  (h1 : x = (a^2) / (b^2))
  (h2 : a ≠ b)
  (h3 : b ≠ 0) :
  (a^2 + b^2) / (a^2 - b^2) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_equality_l217_217913


namespace simplify_sqrt_expression_is_correct_l217_217933

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l217_217933


namespace unique_solution_value_k_l217_217412

theorem unique_solution_value_k (k : ℚ) :
  (∀ x : ℚ, (x + 3) / (k * x - 2) = x → x = -2) ↔ k = -3 / 4 :=
by
  sorry

end unique_solution_value_k_l217_217412


namespace water_breaks_frequency_l217_217464

theorem water_breaks_frequency :
  ∃ W : ℕ, (240 / 120 + 10) = 240 / W :=
by
  existsi (20 : ℕ)
  sorry

end water_breaks_frequency_l217_217464


namespace shaded_area_l217_217192

-- Define the radii of the circles
def R : ℝ := 9        -- radius of the larger circle
def r : ℝ := R / 2    -- radius of each smaller circle (half the radius of the larger circle)

-- Define the areas of the circles
def area_large_circle : ℝ := Real.pi * R^2
def area_small_circle : ℝ := Real.pi * r^2
def total_area_small_circles : ℝ := 3 * area_small_circle

-- Prove the area of the shaded region
theorem shaded_area : area_large_circle - total_area_small_circles = 20.25 * Real.pi := by
  sorry

end shaded_area_l217_217192


namespace transform_cos_to_base_form_l217_217075

theorem transform_cos_to_base_form :
  let f (x : ℝ) := Real.cos (2 * x + (Real.pi / 3))
  let g (x : ℝ) := Real.cos (2 * x)
  ∃ (shift : ℝ), shift = Real.pi / 6 ∧
    (∀ x : ℝ, f (x - shift) = g x) :=
by
  let f := λ x : ℝ => Real.cos (2 * x + (Real.pi / 3))
  let g := λ x : ℝ => Real.cos (2 * x)
  use Real.pi / 6
  sorry

end transform_cos_to_base_form_l217_217075


namespace general_term_a_sum_of_bn_l217_217202

-- Define sequences a_n and b_n
noncomputable def a (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

-- Conditions
lemma condition_1 (n : ℕ) : a n > 0 := by sorry
lemma condition_2 (n : ℕ) : (a n)^2 + 2 * (a n) = 4 * (n * (n + 1)) + 3 := 
  by sorry

-- Theorem for question 1
theorem general_term_a (n : ℕ) : a n = 2 * n + 1 := by sorry

-- Theorem for question 2
theorem sum_of_bn (n : ℕ) : 
  (Finset.range n).sum b = (n : ℚ) / (6 * n + 9) := by sorry

end general_term_a_sum_of_bn_l217_217202


namespace sum_of_reciprocals_of_roots_l217_217171

noncomputable def polynomial_has_roots_cyclotomic_circle 
  (a b c d : ℝ) : Prop :=
  ∃ z1 z2 z3 z4 : ℂ, 
    (z1^4 + a * z1^3 + b * z1^2 + c * z1 + d = 0) ∧ 
    (z2^4 + a * z2^3 + b * z2^2 + c * z2 + d = 0) ∧ 
    (z3^4 + a * z3^3 + b * z3^2 + c * z3 + d = 0) ∧ 
    (z4^4 + a * z4^3 + b * z4^2 + c * z4 + d = 0) ∧ 
    |z1| = 1 ∧ |z2| = 1 ∧ |z3| = 1 ∧ |z4| = 1

theorem sum_of_reciprocals_of_roots 
    (a b c d : ℝ) 
    (h : polynomial_has_roots_cyclotomic_circle a b c d)
  : ∑ z : ℂ in {z1, z2, z3, z4}, z⁻¹ = -a := 
sorry

end sum_of_reciprocals_of_roots_l217_217171


namespace model_tower_height_l217_217609

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_real : real_height = 60) (v_real : real_volume = 200000) (v_model : model_volume = 0.2) :
  real_height / (real_volume / model_volume)^(1/3) = 0.6 :=
by
  rw [h_real, v_real, v_model]
  norm_num
  sorry

end model_tower_height_l217_217609


namespace heather_total_oranges_l217_217181

-- Define the initial conditions
def initial_oranges : ℝ := 60.0
def additional_oranges : ℝ := 35.0

-- Define the total number of oranges
def total_oranges : ℝ := initial_oranges + additional_oranges

-- State the theorem that needs to be proven
theorem heather_total_oranges : total_oranges = 95.0 := 
by
  sorry

end heather_total_oranges_l217_217181


namespace four_kids_wash_three_whiteboards_in_20_minutes_l217_217560

-- Condition: It takes one kid 160 minutes to wash six whiteboards
def time_per_whiteboard_for_one_kid : ℚ := 160 / 6

-- Calculation involving four kids
def time_per_whiteboard_for_four_kids : ℚ := time_per_whiteboard_for_one_kid / 4

-- The total time it takes for four kids to wash three whiteboards together
def total_time_for_four_kids_washing_three_whiteboards : ℚ := time_per_whiteboard_for_four_kids * 3

-- Statement to prove
theorem four_kids_wash_three_whiteboards_in_20_minutes : 
  total_time_for_four_kids_washing_three_whiteboards = 20 :=
by
  sorry

end four_kids_wash_three_whiteboards_in_20_minutes_l217_217560


namespace phone_plan_cost_equal_at_2500_l217_217691

-- We define the costs C1 and C2 as described in the problem conditions.
def C1 (x : ℕ) : ℝ :=
  if x <= 500 then 50 else 50 + 0.35 * (x - 500)

def C2 (x : ℕ) : ℝ :=
  if x <= 1000 then 75 else 75 + 0.45 * (x - 1000)

-- We need to prove that the costs are equal when x = 2500.
theorem phone_plan_cost_equal_at_2500 : C1 2500 = C2 2500 := by
  sorry

end phone_plan_cost_equal_at_2500_l217_217691


namespace triangle_perimeter_l217_217864

theorem triangle_perimeter (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) (angle_A : ℝ)
  (h1 : AB = 4) (h2 : AC = 4) (h3 : angle_A = 60) : 
  AB + AC + AB = 12 :=
by {
  sorry
}

end triangle_perimeter_l217_217864


namespace sum_of_arithmetic_sequence_l217_217317

variable {α : Type*} [LinearOrderedField α]

noncomputable def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
n * (a 1 + a n) / 2

theorem sum_of_arithmetic_sequence {a : ℕ → α} {d : α}
  (h3 : a 3 * a 7 = -16)
  (h4 : a 4 + a 6 = 0)
  (ha : is_arithmetic_sequence a d) :
  ∃ (s : α), s = n * (n - 9) ∨ s = -n * (n - 9) :=
sorry

end sum_of_arithmetic_sequence_l217_217317


namespace substitution_result_l217_217090

-- Conditions
def eq1 (x y : ℝ) : Prop := y = 2 * x - 3
def eq2 (x y : ℝ) : Prop := x - 2 * y = 6

-- The statement to be proven
theorem substitution_result (x y : ℝ) (h1 : eq1 x y) : (x - 4 * x + 6 = 6) :=
by sorry

end substitution_result_l217_217090


namespace sum_of_possible_k_l217_217022

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l217_217022


namespace sequence_an_square_l217_217899

theorem sequence_an_square (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) > a n) 
  (h3 : ∀ n : ℕ, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) :
  ∀ n : ℕ, a n = n^2 :=
by
  sorry

end sequence_an_square_l217_217899


namespace fraction_of_Charlie_circumference_l217_217903

/-- Definitions for the problem conditions -/
def Jack_head_circumference : ℕ := 12
def Charlie_head_circumference : ℕ := 9 + Jack_head_circumference / 2
def Bill_head_circumference : ℕ := 10

/-- Statement of the theorem to be proved -/
theorem fraction_of_Charlie_circumference :
  Bill_head_circumference / Charlie_head_circumference = 2 / 3 :=
sorry

end fraction_of_Charlie_circumference_l217_217903


namespace part1_part2_l217_217251

theorem part1 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ a b : students, a ≠ b ∧
  (∀ c : students, c ≠ a → d a c > d a b) ∧ 
  (∀ c : students, c ≠ b → d b c > d b a) :=
sorry

theorem part2 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ c : students, ∀ a : students, ¬ (∀ b : students, b ≠ a → d b a < d b c ∧ d a c < d a b) :=
sorry

end part1_part2_l217_217251


namespace sum_345_consecutive_sequences_l217_217457

theorem sum_345_consecutive_sequences :
  ∃ (n : ℕ), n = 7 ∧ (∀ (k : ℕ), n ≥ 2 →
    (n * (2 * k + n - 1) = 690 → 2 * k + n - 1 > n)) :=
sorry

end sum_345_consecutive_sequences_l217_217457


namespace max_min_values_of_x_l217_217764

theorem max_min_values_of_x (x y z : ℝ) (h1 : x + y + z = 0) (h2 : (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 2) :
  -2/3 ≤ x ∧ x ≤ 2/3 :=
sorry

end max_min_values_of_x_l217_217764


namespace number_of_bouquets_to_earn_1000_dollars_l217_217276

def cost_of_buying (n : ℕ) : ℕ :=
  n * 20

def revenue_from_selling (m : ℕ) : ℕ :=
  m * 20

def profit_per_operation : ℤ :=
  revenue_from_selling 7 - cost_of_buying 5

theorem number_of_bouquets_to_earn_1000_dollars :
  ∀ bouquets_needed : ℕ, bouquets_needed = 5 * (1000 / profit_per_operation.nat_abs) :=
sorry

end number_of_bouquets_to_earn_1000_dollars_l217_217276


namespace ellipse_with_foci_on_y_axis_l217_217945

theorem ellipse_with_foci_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1) ↔ (m > n ∧ n > 0) := 
sorry

end ellipse_with_foci_on_y_axis_l217_217945


namespace monotonic_decreasing_interval_l217_217370

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem monotonic_decreasing_interval :
  {x : ℝ | 0 < x ∧ x ≤ 1} = {x : ℝ | ∃ ε > 0, ∀ y, y < x → f y > f x ∧ y > 0} :=
sorry

end monotonic_decreasing_interval_l217_217370


namespace elena_allowance_fraction_l217_217300

variable {A m s : ℝ}

theorem elena_allowance_fraction {A : ℝ} (h1 : m = 0.25 * (A - s)) (h2 : s = 0.10 * (A - m)) : m + s = (4 / 13) * A :=
by
  sorry

end elena_allowance_fraction_l217_217300


namespace grandfather_age_l217_217517

variable (F S G : ℕ)

theorem grandfather_age (h1 : F = 58) (h2 : F - S = S) (h3 : S - 5 = (1 / 2) * G) : G = 48 := by
  sorry

end grandfather_age_l217_217517


namespace min_x2_y2_l217_217013

theorem min_x2_y2 (x y : ℝ) (h : x * y - x - y = 1) : x^2 + y^2 ≥ 6 - 4 * Real.sqrt 2 :=
by
  sorry

end min_x2_y2_l217_217013


namespace intersection_with_xz_plane_l217_217715

-- Initial points on the line
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def point1 : Point3D := ⟨2, -1, 3⟩
def point2 : Point3D := ⟨6, -4, 7⟩

-- Definition of the line parametrization
def param_line (t : ℝ) : Point3D :=
  ⟨ point1.x + t * (point2.x - point1.x)
  , point1.y + t * (point2.y - point1.y)
  , point1.z + t * (point2.z - point1.z) ⟩

-- Prove that the line intersects the xz-plane at the expected point
theorem intersection_with_xz_plane :
  ∃ t : ℝ, param_line t = ⟨ 2/3, 0, 5/3 ⟩ :=
sorry

end intersection_with_xz_plane_l217_217715


namespace james_spends_on_pistachios_per_week_l217_217755

theorem james_spends_on_pistachios_per_week :
  let cost_per_can := 10
  let ounces_per_can := 5
  let total_ounces_per_5_days := 30
  let days_per_week := 7
  let cost_per_ounce := cost_per_can / ounces_per_can
  let daily_ounces := total_ounces_per_5_days / 5
  let daily_cost := daily_ounces * cost_per_ounce
  daily_cost * days_per_week = 84 :=
by
  sorry

end james_spends_on_pistachios_per_week_l217_217755


namespace total_selling_price_l217_217673

def original_price : ℝ := 120
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.15

def sale_price (original_price discount_percent : ℝ) : ℝ :=
  original_price * (1 - discount_percent)

def final_price (sale_price tax_percent : ℝ) : ℝ :=
  sale_price * (1 + tax_percent)

theorem total_selling_price :
  final_price (sale_price original_price discount_percent) tax_percent = 96.6 :=
sorry

end total_selling_price_l217_217673


namespace range_of_x_satisfying_inequality_l217_217572

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * |x - 1|

theorem range_of_x_satisfying_inequality :
  {x : ℝ | f x > f (2 * x)} = {x : ℝ | 0 < x ∧ x < (2 : ℝ) / 3} :=
by
  sorry

end range_of_x_satisfying_inequality_l217_217572


namespace pears_value_equivalence_l217_217622

-- Condition: $\frac{3}{4}$ of $16$ apples are worth $12$ pears
def apples_to_pears (a p : ℕ) : Prop :=
  (3 * 16 / 4 * a = 12 * p)

-- Question: How many pears (p) are equivalent in value to $\frac{2}{3}$ of $9$ apples?
def pears_equivalent_to_apples (p : ℕ) : Prop :=
  (2 * 9 / 3 * p = 6)

theorem pears_value_equivalence (p : ℕ) (a : ℕ) (h1 : apples_to_pears a p) (h2 : pears_equivalent_to_apples p) : 
  p = 6 :=
sorry

end pears_value_equivalence_l217_217622


namespace right_triangle_third_side_product_l217_217963

theorem right_triangle_third_side_product :
  let a := 6
  let b := 8
  let c1 := Real.sqrt (a^2 + b^2)     -- Hypotenuse when a and b are legs
  let c2 := Real.sqrt (b^2 - a^2)     -- Other side when b is the hypotenuse
  20 * Real.sqrt 7 ≈ 52.7 := 
by
  sorry

end right_triangle_third_side_product_l217_217963


namespace map_distance_l217_217508

theorem map_distance
  (s d_m : ℝ) (d_r : ℝ)
  (h1 : s = 0.4)
  (h2 : d_r = 5.3)
  (h3 : d_m = 64) :
  (d_m * d_r / s) = 848 := by
  sorry

end map_distance_l217_217508


namespace determine_f_function_l217_217565

variable (f : ℝ → ℝ)

theorem determine_f_function (x : ℝ) (h : f (1 - x) = 1 + x) : f x = 2 - x := 
sorry

end determine_f_function_l217_217565


namespace percentage_increase_correct_l217_217778

def highest_price : ℕ := 24
def lowest_price : ℕ := 16

theorem percentage_increase_correct :
  ((highest_price - lowest_price) * 100 / lowest_price) = 50 :=
by
  sorry

end percentage_increase_correct_l217_217778


namespace func_symmetry_monotonicity_range_of_m_l217_217065

open Real

theorem func_symmetry_monotonicity (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (1 - x))
  (h2 : ∀ x1 x2, 2 < x1 → 2 < x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∀ x1 x2, (x1 > 2 ∧ x2 > 2 → f x1 < f x2 → x1 < x2) ∧
            (x2 > 2 ∧ x1 > x2 → f x2 < f x1 → x2 < x1)) := 
sorry

theorem range_of_m (f : ℝ → ℝ)
  (h : ∀ θ : ℝ, f (cos θ ^ 2 + 2 * (m : ℝ) ^ 2 + 2) < f (sin θ + m ^ 2 - 3 * m - 2)) :
  ∀ m, (3 - sqrt 42) / 6 < m ∧ m < (3 + sqrt 42) / 6 :=
sorry

end func_symmetry_monotonicity_range_of_m_l217_217065


namespace count_triples_l217_217731

open Set

theorem count_triples 
  (A B C : Set ℕ) 
  (h_union : A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  (h_inter : A ∩ B ∩ C = ∅) :
  (∃ n : ℕ, n = 60466176) :=
by
  -- Proof can be filled in here
  sorry

end count_triples_l217_217731


namespace total_worth_is_correct_l217_217052

-- Define the conditions
def rows : ℕ := 4
def gold_bars_per_row : ℕ := 20
def worth_per_gold_bar : ℕ := 20000

-- Define the calculated values
def total_gold_bars : ℕ := rows * gold_bars_per_row
def total_worth_of_gold_bars : ℕ := total_gold_bars * worth_per_gold_bar

-- Theorem statement to prove the correct total worth
theorem total_worth_is_correct : total_worth_of_gold_bars = 1600000 := by
  sorry

end total_worth_is_correct_l217_217052


namespace necessary_but_not_sufficient_condition_l217_217375

theorem necessary_but_not_sufficient_condition
    {a b : ℕ} :
    (¬ (a = 1) ∨ ¬ (b = 2)) ↔ (a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2) :=
by
    sorry

end necessary_but_not_sufficient_condition_l217_217375


namespace last_four_digits_5_2011_l217_217351

theorem last_four_digits_5_2011 :
  (5^2011 % 10000) = 8125 := by
  sorry

end last_four_digits_5_2011_l217_217351


namespace real_seq_proof_l217_217198

noncomputable def real_seq_ineq (a : ℕ → ℝ) : Prop :=
  ∀ k m : ℕ, k > 0 → m > 0 → |a (k + m) - a k - a m| ≤ 1

theorem real_seq_proof (a : ℕ → ℝ) (h : real_seq_ineq a) :
  ∀ k m : ℕ, k > 0 → m > 0 → |a k / k - a m / m| < 1 / k + 1 / m :=
by
  sorry

end real_seq_proof_l217_217198


namespace range_of_a_l217_217589

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2 * t - a < 0) ↔ a ≤ -1 :=
by sorry

end range_of_a_l217_217589


namespace sum_possible_k_l217_217037

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l217_217037


namespace max_value_of_exp_sum_l217_217762

theorem max_value_of_exp_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_pos : 0 < a * b) :
    ∃ θ : ℝ, a * Real.exp θ + b * Real.exp (-θ) = 2 * Real.sqrt (a * b) :=
by
  sorry

end max_value_of_exp_sum_l217_217762


namespace Maya_takes_longer_l217_217810

-- Define the constants according to the conditions
def Xavier_reading_speed : ℕ := 120
def Maya_reading_speed : ℕ := 60
def novel_pages : ℕ := 360
def minutes_per_hour : ℕ := 60

-- Define the times it takes for Xavier and Maya to read the novel
def Xavier_time : ℕ := novel_pages / Xavier_reading_speed
def Maya_time : ℕ := novel_pages / Maya_reading_speed

-- Define the time difference in hours and then in minutes
def time_difference_hours : ℕ := Maya_time - Xavier_time
def time_difference_minutes : ℕ := time_difference_hours * minutes_per_hour

-- The statement to prove
theorem Maya_takes_longer :
  time_difference_minutes = 180 :=
by
  sorry

end Maya_takes_longer_l217_217810


namespace coin_flip_probability_l217_217943

def total_outcomes := 2^6
def favorable_outcomes := 2^3
def probability := favorable_outcomes / total_outcomes

theorem coin_flip_probability :
  probability = 1 / 8 :=
by
  unfold probability total_outcomes favorable_outcomes
  sorry

end coin_flip_probability_l217_217943


namespace fraction_of_foreign_males_l217_217409

theorem fraction_of_foreign_males
  (total_students : ℕ)
  (female_ratio : ℚ)
  (non_foreign_males : ℕ)
  (foreign_male_fraction : ℚ)
  (h1 : total_students = 300)
  (h2 : female_ratio = 2/3)
  (h3 : non_foreign_males = 90) :
  foreign_male_fraction = 1/10 :=
by
  sorry

end fraction_of_foreign_males_l217_217409


namespace positive_integer_condition_l217_217012

theorem positive_integer_condition (p : ℕ) (hp : 0 < p) : 
  (∃ k : ℤ, k > 0 ∧ 4 * p + 17 = k * (3 * p - 8)) ↔ p = 3 :=
by {
  sorry
}

end positive_integer_condition_l217_217012


namespace boxes_needed_l217_217984

def initial_games : ℕ := 76
def games_sold : ℕ := 46
def games_per_box : ℕ := 5

theorem boxes_needed : (initial_games - games_sold) / games_per_box = 6 := by
  sorry

end boxes_needed_l217_217984


namespace geometric_sequence_sum_eq_80_243_l217_217376

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_eq_80_243 {n : ℕ} :
  let a := (1 / 3 : ℝ)
  let r := (1 / 3 : ℝ)
  geometric_sum a r n = 80 / 243 ↔ n = 3 :=
by
  intros a r
  sorry

end geometric_sequence_sum_eq_80_243_l217_217376


namespace summer_camp_skills_l217_217019

theorem summer_camp_skills
  (x y z a b c : ℕ)
  (h1 : x + y + z + a + b + c = 100)
  (h2 : y + z + c = 42)
  (h3 : z + x + b = 65)
  (h4 : x + y + a = 29) :
  a + b + c = 64 :=
by sorry

end summer_camp_skills_l217_217019


namespace common_sale_days_in_july_l217_217256

def BookstoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (d % 4 = 0)

def ShoeStoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (∃ k : ℕ, d = 2 + k * 7)

theorem common_sale_days_in_july : ∃! d, (BookstoreSaleDays d) ∧ (ShoeStoreSaleDays d) :=
by {
  sorry
}

end common_sale_days_in_july_l217_217256


namespace addition_correctness_l217_217384

theorem addition_correctness : 1.25 + 47.863 = 49.113 :=
by 
  sorry

end addition_correctness_l217_217384


namespace emily_annual_holidays_l217_217154

theorem emily_annual_holidays 
    (holidays_per_month : ℕ) 
    (months_in_year : ℕ) 
    (h1: holidays_per_month = 2)
    (h2: months_in_year = 12)
    : holidays_per_month * months_in_year = 24 := 
by
  sorry

end emily_annual_holidays_l217_217154


namespace abc_product_l217_217561

theorem abc_product (A B C D : ℕ) 
  (h1 : A + B + C + D = 64)
  (h2 : A + 3 = B - 3)
  (h3 : A + 3 = C * 3)
  (h4 : A + 3 = D / 3) :
  A * B * C * D = 19440 := 
by
  sorry

end abc_product_l217_217561


namespace smallest_number_among_l217_217530

theorem smallest_number_among
  (π : ℝ) (Hπ_pos : π > 0) :
  ∀ (a b c d : ℝ), 
    (a = 0) → 
    (b = -1) → 
    (c = -1.5) → 
    (d = π) → 
    (∀ (x y : ℝ), (x > 0) → (y > 0) → (x > y) ↔ x - y > 0) → 
    (∀ (x : ℝ), x < 0 → x < 0) → 
    (∀ (x y : ℝ), (x > 0) → (y < 0) → x > y) → 
    (∀ (x y : ℝ), (x < 0) → (y < 0) → (|x| > |y|) → x < y) → 
  c = -1.5 := 
by
  intros a b c d Ha Hb Hc Hd Hpos Hneg HposNeg Habs
  sorry

end smallest_number_among_l217_217530


namespace find_N_l217_217957

theorem find_N : 
  ∀ (a b c N : ℝ), 
  a + b + c = 80 → 
  2 * a = N → 
  b - 10 = N → 
  3 * c = N → 
  N = 38 := 
by sorry

end find_N_l217_217957


namespace minimum_k_l217_217434

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem minimum_k (e : ℝ) (h : e = Real.exp 1) :
  (∀ m : ℝ, m ∈ Set.Icc (-2 : ℝ) 4 → f (-2 * m^2 + 2 * m - 1) + f (8 * m + e^4) > 0) → 4 = 4 := 
sorry

end minimum_k_l217_217434


namespace percentage_of_ore_contains_alloy_l217_217125

def ore_contains_alloy_iron (weight_ore weight_iron : ℝ) (P : ℝ) : Prop :=
  (P / 100 * weight_ore) * 0.9 = weight_iron

theorem percentage_of_ore_contains_alloy (w_ore : ℝ) (w_iron : ℝ) (P : ℝ) 
    (h_w_ore : w_ore = 266.6666666666667) (h_w_iron : w_iron = 60) 
    (h_ore_contains : ore_contains_alloy_iron w_ore w_iron P) 
    : P = 25 :=
by
  rw [h_w_ore, h_w_iron] at h_ore_contains
  sorry

end percentage_of_ore_contains_alloy_l217_217125


namespace simplify_sqrt_sum_l217_217929

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l217_217929


namespace max_total_time_l217_217918

theorem max_total_time :
  ∀ (time_mowing time_fertilizing total_time : ℕ), 
    time_mowing = 40 ∧ time_fertilizing = 2 * time_mowing ∧ total_time = time_mowing + time_fertilizing → 
    total_time = 120 :=
by
  intros time_mowing time_fertilizing total_time h
  have h1: time_mowing = 40 := h.1
  have h2: time_fertilizing = 2 * time_mowing := h.2.1
  have h3: total_time = time_mowing + time_fertilizing := h.2.2
  rw[h1] at h2
  rw[h1, h2] at h3
  simp at h3
  exact h3.symm

end max_total_time_l217_217918


namespace cistern_fill_time_l217_217994

theorem cistern_fill_time (hA : ℝ) (hB : ℝ) (hC : ℝ) : hA = 12 → hB = 18 → hC = 15 → 
  1 / ((1 / hA) + (1 / hB) - (1 / hC)) = 180 / 13 :=
by
  intros hA_eq hB_eq hC_eq
  rw [hA_eq, hB_eq, hC_eq]
  sorry

end cistern_fill_time_l217_217994


namespace anna_final_stamp_count_l217_217531

theorem anna_final_stamp_count (anna_initial : ℕ) (alison_initial : ℕ) (jeff_initial : ℕ)
  (anna_receive_from_alison : ℕ) (anna_give_jeff : ℕ) (anna_receive_jeff : ℕ) :
  anna_initial = 37 →
  alison_initial = 28 →
  jeff_initial = 31 →
  anna_receive_from_alison = alison_initial / 2 →
  anna_give_jeff = 2 →
  anna_receive_jeff = 1 →
  ∃ result : ℕ, result = 50 :=
by
  intros
  sorry

end anna_final_stamp_count_l217_217531


namespace bouquets_needed_to_earn_1000_l217_217280

theorem bouquets_needed_to_earn_1000 :
  ∀ (cost_per_bouquet sell_price_bouquet: ℕ) (roses_per_bouquet_bought roses_per_bouquet_sold target_profit: ℕ),
    cost_per_bouquet = 20 →
    sell_price_bouquet = 20 →
    roses_per_bouquet_bought = 7 →
    roses_per_bouquet_sold = 5 →
    target_profit = 1000 →
    (target_profit / (sell_price_bouquet * roses_per_bouquet_sold / roses_per_bouquet_bought - cost_per_bouquet) * roses_per_bouquet_bought = 125) :=
by
  intros cost_per_bouquet sell_price_bouquet roses_per_bouquet_bought roses_per_bouquet_sold target_profit 
    h_cost_per_bouquet h_sell_price_bouquet h_roses_per_bouquet_bought h_roses_per_bouquet_sold h_target_profit
  sorry

end bouquets_needed_to_earn_1000_l217_217280


namespace sixty_three_times_fifty_seven_l217_217140

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 := by
  sorry

end sixty_three_times_fifty_seven_l217_217140


namespace bill_buys_125_bouquets_to_make_1000_l217_217277

-- Defining the conditions
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_difference : ℕ := 1000

-- To be demonstrated: number of bouquets Bill needs to buy to get a profit difference of $1000
theorem bill_buys_125_bouquets_to_make_1000 : 
  let total_cost_to_buy (n : ℕ) := n * cost_per_bouquet
  let total_roses (n : ℕ) := n * roses_per_bouquet_buy
  let bouquets_sell_from_roses (roses : ℕ) := roses / roses_per_bouquet_sell
  let total_revenue (bouquets : ℕ) := bouquets * cost_per_bouquet
  let profit (n : ℕ) := total_revenue (bouquets_sell_from_roses (total_roses n)) - total_cost_to_buy n
  profit (125 * 5) = target_difference := 
sorry

end bill_buys_125_bouquets_to_make_1000_l217_217277


namespace opposite_of_six_is_neg_six_l217_217584

-- Define the condition that \( a \) is the opposite of \( 6 \)
def is_opposite_of_six (a : Int) : Prop := a = -6

-- Prove that \( a = -6 \) given that \( a \) is the opposite of \( 6 \)
theorem opposite_of_six_is_neg_six (a : Int) (h : is_opposite_of_six a) : a = -6 :=
by
  sorry

end opposite_of_six_is_neg_six_l217_217584


namespace find_total_stock_worth_l217_217097

noncomputable def total_stock_worth (X : ℝ) : Prop :=
  let profit := 0.10 * (0.20 * X)
  let loss := 0.05 * (0.80 * X)
  loss - profit = 450

theorem find_total_stock_worth (X : ℝ) (h : total_stock_worth X) : X = 22500 :=
by
  sorry

end find_total_stock_worth_l217_217097


namespace partial_fraction_product_l217_217177

theorem partial_fraction_product (A B C : ℚ)
  (h_eq : ∀ x, (x^2 - 13) / ((x-2) * (x+2) * (x-3)) = A / (x-2) + B / (x+2) + C / (x-3))
  (h_A : A = 9 / 4)
  (h_B : B = -9 / 20)
  (h_C : C = -4 / 5) :
  A * B * C = 81 / 100 := 
by
  sorry

end partial_fraction_product_l217_217177


namespace value_of_frac_sum_l217_217739

theorem value_of_frac_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : (x + y) / 3 = 11 / 9 :=
by
  sorry

end value_of_frac_sum_l217_217739


namespace bouquets_needed_to_earn_1000_l217_217279

theorem bouquets_needed_to_earn_1000 :
  ∀ (cost_per_bouquet sell_price_bouquet: ℕ) (roses_per_bouquet_bought roses_per_bouquet_sold target_profit: ℕ),
    cost_per_bouquet = 20 →
    sell_price_bouquet = 20 →
    roses_per_bouquet_bought = 7 →
    roses_per_bouquet_sold = 5 →
    target_profit = 1000 →
    (target_profit / (sell_price_bouquet * roses_per_bouquet_sold / roses_per_bouquet_bought - cost_per_bouquet) * roses_per_bouquet_bought = 125) :=
by
  intros cost_per_bouquet sell_price_bouquet roses_per_bouquet_bought roses_per_bouquet_sold target_profit 
    h_cost_per_bouquet h_sell_price_bouquet h_roses_per_bouquet_bought h_roses_per_bouquet_sold h_target_profit
  sorry

end bouquets_needed_to_earn_1000_l217_217279


namespace kaleb_initial_games_l217_217196

-- Let n be the number of games Kaleb started out with
def initial_games (n : ℕ) : Prop :=
  let sold_games := 46
  let boxes := 6
  let games_per_box := 5
  n = sold_games + boxes * games_per_box

-- Now we state the theorem
theorem kaleb_initial_games : ∃ n, initial_games n ∧ n = 76 :=
  by sorry

end kaleb_initial_games_l217_217196


namespace carB_highest_avg_speed_l217_217540

-- Define the distances and times for each car
def distanceA : ℕ := 715
def timeA : ℕ := 11
def distanceB : ℕ := 820
def timeB : ℕ := 12
def distanceC : ℕ := 950
def timeC : ℕ := 14

-- Define the average speeds
def avgSpeedA : ℚ := distanceA / timeA
def avgSpeedB : ℚ := distanceB / timeB
def avgSpeedC : ℚ := distanceC / timeC

theorem carB_highest_avg_speed : avgSpeedB > avgSpeedA ∧ avgSpeedB > avgSpeedC :=
by
  -- Proof will be filled in here
  sorry

end carB_highest_avg_speed_l217_217540


namespace outlets_per_room_l217_217826

theorem outlets_per_room
  (rooms : ℕ)
  (total_outlets : ℕ)
  (h1 : rooms = 7)
  (h2 : total_outlets = 42) :
  total_outlets / rooms = 6 :=
by sorry

end outlets_per_room_l217_217826


namespace inequality_xy_l217_217915

-- Defining the constants and conditions
variables {x y : ℝ}

-- Main theorem to prove the inequality and find pairs for equality
theorem inequality_xy (h : (x + 1) * (y + 2) = 8) :
  (xy - 10)^2 ≥ 64 ∧ ((xy - 10)^2 = 64 → (x, y) = (1, 2) ∨ (x, y) = (-3, -6)) :=
sorry

end inequality_xy_l217_217915


namespace sum_of_possible_k_l217_217024

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l217_217024


namespace closest_to_one_tenth_l217_217322

noncomputable def p (n : ℕ) : ℚ :=
  1 / (n * (n + 2)) + 1 / ((n + 2) * (n + 4)) + 1 / ((n + 4) * (n + 6)) +
  1 / ((n + 6) * (n + 8)) + 1 / ((n + 8) * (n + 10))

theorem closest_to_one_tenth {n : ℕ} (h₀ : 4 ≤ n ∧ n ≤ 7) : 
  |(5 : ℚ) / (n * (n + 10)) - 1 / 10| ≤ 
  |(5 : ℚ) / (4 * (4 + 10)) - 1 / 10| ∧ n = 4 := 
sorry

end closest_to_one_tenth_l217_217322


namespace fractional_equation_solution_l217_217793

theorem fractional_equation_solution (x : ℝ) (hx : x ≠ -3) (h : 1/x = 2/(x+3)) : x = 3 :=
sorry

end fractional_equation_solution_l217_217793


namespace oranges_weigh_4_ounces_each_l217_217767

def apple_weight : ℕ := 4
def max_bag_capacity : ℕ := 49
def num_bags : ℕ := 3
def total_weight : ℕ := num_bags * max_bag_capacity
def total_apple_weight : ℕ := 84
def num_apples : ℕ := total_apple_weight / apple_weight
def num_oranges : ℕ := num_apples
def total_orange_weight : ℕ := total_apple_weight
def weight_per_orange : ℕ := total_orange_weight / num_oranges

theorem oranges_weigh_4_ounces_each :
  weight_per_orange = 4 := by
  sorry

end oranges_weigh_4_ounces_each_l217_217767


namespace jennys_wedding_guests_l217_217603

noncomputable def total_guests (C S : ℕ) : ℕ := C + S

theorem jennys_wedding_guests :
  ∃ (C S : ℕ), (S = 3 * C) ∧
               (18 * C + 25 * S = 1860) ∧
               (total_guests C S = 80) :=
sorry

end jennys_wedding_guests_l217_217603


namespace curve_is_circle_l217_217420

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b r : ℝ), (r > 0) ∧ ((x + a)^2 + (y + b)^2 = r^2) :=
by
  sorry

end curve_is_circle_l217_217420


namespace selected_student_in_eighteenth_group_l217_217644

def systematic_sampling (first_number common_difference nth_term : ℕ) : ℕ :=
  first_number + (nth_term - 1) * common_difference

theorem selected_student_in_eighteenth_group :
  systematic_sampling 22 50 18 = 872 :=
by
  sorry

end selected_student_in_eighteenth_group_l217_217644


namespace annual_interest_rate_is_correct_l217_217841

-- Define conditions
def principal : ℝ := 900
def finalAmount : ℝ := 992.25
def compoundingPeriods : ℕ := 2
def timeYears : ℕ := 1

-- Compound interest formula
def compound_interest (P A r : ℝ) (n t : ℕ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

-- Statement to prove
theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, compound_interest principal finalAmount r compoundingPeriods timeYears ∧ r = 0.10 :=
by 
  sorry

end annual_interest_rate_is_correct_l217_217841


namespace remainder_of_3_pow_19_mod_5_l217_217504

theorem remainder_of_3_pow_19_mod_5 : (3 ^ 19) % 5 = 2 := by
  have h : 3 ^ 4 % 5 = 1 := by sorry
  sorry

end remainder_of_3_pow_19_mod_5_l217_217504


namespace cyclist_waits_15_minutes_l217_217678

-- Definitions
def hiker_rate := 7 -- miles per hour
def cyclist_rate := 28 -- miles per hour
def wait_time := 15 / 60 -- hours, as the cyclist waits 15 minutes, converted to hours

-- The statement to be proven
theorem cyclist_waits_15_minutes :
  ∃ t : ℝ, t = 15 / 60 ∧
  (∀ d : ℝ, d = (hiker_rate * wait_time) →
            d = (cyclist_rate * t - hiker_rate * t)) :=
by
  sorry

end cyclist_waits_15_minutes_l217_217678


namespace son_work_rate_l217_217110

noncomputable def man_work_rate := 1/10
noncomputable def combined_work_rate := 1/5

theorem son_work_rate :
  ∃ S : ℝ, man_work_rate + S = combined_work_rate ∧ S = 1/10 := sorry

end son_work_rate_l217_217110


namespace bucket_B_more_than_C_l217_217073

-- Define the number of pieces of fruit in bucket B as a constant
def B := 12

-- Define the number of pieces of fruit in bucket C as a constant
def C := 9

-- Define the number of pieces of fruit in bucket A based on B
def A := B + 4

-- Define the total number of pieces of fruit in all three buckets
def total_fruit := A + B + C

-- Prove that bucket B has 3 more pieces of fruit than bucket C
theorem bucket_B_more_than_C : B - C = 3 := by
  -- sorry is used to skip the proof
  sorry

end bucket_B_more_than_C_l217_217073


namespace value_of_trig_expression_l217_217167

theorem value_of_trig_expression (α : Real) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -3 :=
by 
  sorry

end value_of_trig_expression_l217_217167


namespace jennifer_dogs_l217_217600

theorem jennifer_dogs (D : ℕ) (groom_time_per_dog : ℕ) (groom_days : ℕ) (total_groom_time : ℕ) :
  groom_time_per_dog = 20 →
  groom_days = 30 →
  total_groom_time = 1200 →
  groom_days * (groom_time_per_dog * D) = total_groom_time →
  D = 2 :=
by
  intro h1 h2 h3 h4
  sorry

end jennifer_dogs_l217_217600


namespace remainder_problem_l217_217443

theorem remainder_problem (d r : ℤ) (h1 : 1237 % d = r)
    (h2 : 1694 % d = r) (h3 : 2791 % d = r) (hd : d > 1) :
    d - r = 134 := sorry

end remainder_problem_l217_217443


namespace correct_calculation_l217_217387

theorem correct_calculation (a b : ℕ) : a^3 * b^3 = (a * b)^3 :=
sorry

end correct_calculation_l217_217387


namespace find_initial_oranges_l217_217046

variable (O : ℕ)
variable (reserved_fraction : ℚ := 1 / 4)
variable (sold_fraction : ℚ := 3 / 7)
variable (rotten_oranges : ℕ := 4)
variable (good_oranges_today : ℕ := 32)

-- Define the total oranges before finding the rotten oranges
def oranges_before_rotten := good_oranges_today + rotten_oranges

-- Define the remaining fraction of oranges after reserving for friends and selling some
def remaining_fraction := (1 - reserved_fraction) * (1 - sold_fraction)

-- State the theorem to be proven
theorem find_initial_oranges (h : remaining_fraction * O = oranges_before_rotten) : O = 84 :=
sorry

end find_initial_oranges_l217_217046


namespace union_correct_l217_217430

variable (x : ℝ)
def A := {x | -2 < x ∧ x < 1}
def B := {x | 0 < x ∧ x < 3}
def unionSet := {x | -2 < x ∧ x < 3}

theorem union_correct : ( {x | -2 < x ∧ x < 1} ∪ {x | 0 < x ∧ x < 3} ) = {x | -2 < x ∧ x < 3} := by
  sorry

end union_correct_l217_217430


namespace number_of_truthful_dwarfs_l217_217555

def dwarf_condition := 
  ∀ (dwarfs : ℕ) (truthful_dwarfs : ℕ) (lying_dwarfs : ℕ),
    dwarfs = 10 ∧ 
    (∀ n, n ∈ {truthful_dwarfs, lying_dwarfs} -> n ≥ 0) ∧ 
    truthful_dwarfs + lying_dwarfs = dwarfs ∧
    truthful_dwarfs + 2 * lying_dwarfs = 16

theorem number_of_truthful_dwarfs : ∃ (truthful_dwarfs : ℕ), (dwarf_condition ∧ truthful_dwarfs = 4) :=
by {
  let dwarfs := 10,
  let lying_dwarfs := 6,
  let truthful_dwarfs := dwarfs - lying_dwarfs,
  have h: truthful_dwarfs = 4,
  { calc
    truthful_dwarfs = dwarfs - lying_dwarfs : by rfl
    ... = 10 - 6 : by rfl
    ... = 4 : by rfl },
  existsi (4 : ℕ),
  refine ⟨_, ⟨dwarfs, truthful_dwarfs, lying_dwarfs, rfl, _, _, _⟩⟩,
  -- Now we can provide the additional details for lean to understand the conditions hold
  {
    intros n hn,
    simp,
    exact hn
  },
  {
    exact add_comm 6 4
  },
  {
    dsimp,
    ring,
  },
  {
    exact h,
  }
  -- Skip the actual proof with sorry
  sorry
}

end number_of_truthful_dwarfs_l217_217555


namespace gray_region_area_l217_217146

theorem gray_region_area (r : ℝ) : 
  let inner_circle_radius := r
  let outer_circle_radius := r + 3
  let inner_circle_area := Real.pi * (r ^ 2)
  let outer_circle_area := Real.pi * ((r + 3) ^ 2)
  let gray_region_area := outer_circle_area - inner_circle_area
  gray_region_area = 6 * Real.pi * r + 9 * Real.pi := 
by
  sorry

end gray_region_area_l217_217146


namespace problem_statement_l217_217288

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l217_217288


namespace decreasing_on_interval_l217_217724

variable {x m n : ℝ}

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := |x^2 - 2 * m * x + n|

theorem decreasing_on_interval
  (h : ∀ x, f x m n = |x^2 - 2 * m * x + n|)
  (h_cond : m^2 - n ≤ 0) :
  ∀ x y, x ≤ y → y ≤ m → f y m n ≤ f x m n :=
sorry

end decreasing_on_interval_l217_217724


namespace power_function_increasing_l217_217268

theorem power_function_increasing {α : ℝ} (hα : α = 1 ∨ α = 3 ∨ α = 1 / 2) :
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → x ^ α ≤ y ^ α := 
sorry

end power_function_increasing_l217_217268


namespace ellipse_standard_equation_parabola_standard_equation_l217_217252

theorem ellipse_standard_equation (x y : ℝ) (a b : ℝ) (h₁ : a > b ∧ b > 0)
  (h₂ : 2 * a = Real.sqrt ((3 + 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2) 
      + Real.sqrt ((3 - 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2))
  (h₃ : b^2 = a^2 - 4) 
  : (x^2 / 36 + y^2 / 32 = 1) :=
by sorry

theorem parabola_standard_equation (y : ℝ) (p : ℝ) (h₁ : p > 0)
  (h₂ : -p / 2 = -1 / 2) 
  : (y^2 = 2 * p * 1) :=
by sorry

end ellipse_standard_equation_parabola_standard_equation_l217_217252


namespace sum_lent_is_1050_l217_217676

-- Define the variables for the problem
variable (P : ℝ) -- Sum lent
variable (r : ℝ) -- Interest rate
variable (t : ℝ) -- Time period
variable (I : ℝ) -- Interest

-- Define the conditions
def conditions := 
  r = 0.06 ∧ 
  t = 6 ∧ 
  I = P - 672 ∧ 
  I = P * (r * t)

-- Define the main theorem
theorem sum_lent_is_1050 (P r t I : ℝ) (h : conditions P r t I) : P = 1050 :=
  sorry

end sum_lent_is_1050_l217_217676


namespace line_intersects_x_axis_at_neg3_l217_217127

theorem line_intersects_x_axis_at_neg3 :
  ∃ (x y : ℝ), (5 * y - 7 * x = 21 ∧ y = 0) ↔ (x = -3 ∧ y = 0) :=
by
  sorry

end line_intersects_x_axis_at_neg3_l217_217127


namespace find_percentage_decrease_in_fourth_month_l217_217591

theorem find_percentage_decrease_in_fourth_month
  (P0 : ℝ) (P1 : ℝ) (P2 : ℝ) (P3 : ℝ) (x : ℝ) :
  (P0 = 100) →
  (P1 = P0 + 0.30 * P0) →
  (P2 = P1 - 0.15 * P1) →
  (P3 = P2 + 0.10 * P2) →
  (P0 = P3 - x / 100 * P3) →
  x = 18 :=
by
  sorry

end find_percentage_decrease_in_fourth_month_l217_217591


namespace greatest_integer_difference_l217_217098

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) : 
  ∃ d, d = y - x ∧ d = 2 := 
by
  sorry

end greatest_integer_difference_l217_217098


namespace milk_for_9_cookies_l217_217379

def quarts_to_pints (q : ℕ) : ℕ := q * 2

def milk_for_cookies (cookies : ℕ) (milk_in_quarts : ℕ) : ℕ :=
  quarts_to_pints milk_in_quarts * cookies / 18

theorem milk_for_9_cookies :
  milk_for_cookies 9 3 = 3 :=
by
  -- We define the conversion and proportional conditions explicitly here.
  unfold milk_for_cookies
  unfold quarts_to_pints
  sorry

end milk_for_9_cookies_l217_217379


namespace sin_cos_difference_theorem_tan_theorem_l217_217162

open Real

noncomputable def sin_cos_difference (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5)

theorem sin_cos_difference_theorem (x : ℝ) (h : sin_cos_difference x) : 
  sin x - cos x = - 7 / 5 := by
  sorry

noncomputable def sin_cos_ratio (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5) ∧ (tan x = -3 / 4)

theorem tan_theorem (x : ℝ) (h : sin_cos_ratio x) :
  tan x = -3 / 4 := by
  sorry

end sin_cos_difference_theorem_tan_theorem_l217_217162


namespace part1_part2_l217_217020

-- Definitions corresponding to the conditions
def angle_A := 35
def angle_B1 := 40
def three_times_angle_triangle (A B C : ℕ) : Prop :=
  A + B + C = 180 ∧ (A = 3 * B ∨ B = 3 * A ∨ C = 3 * A ∨ A = 3 * C ∨ B = 3 * C ∨ C = 3 * B)

-- Part 1: Checking if triangle ABC is a "three times angle triangle".
theorem part1 : three_times_angle_triangle angle_A angle_B1 (180 - angle_A - angle_B1) :=
  sorry

-- Definitions corresponding to the new conditions
def angle_B2 := 60

-- Part 2: Finding the smallest interior angle in triangle ABC.
theorem part2 (angle_A angle_C : ℕ) :
  three_times_angle_triangle angle_A angle_B2 angle_C → (angle_A = 20 ∨ angle_A = 30 ∨ angle_C = 20 ∨ angle_C = 30) :=
  sorry

end part1_part2_l217_217020


namespace smallest_d_l217_217856

noncomputable def smallestPositiveD : ℝ := 1

theorem smallest_d (d : ℝ) : 
  (0 < d) →
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → 
    (Real.sqrt (x * y) + d * (x^2 - y^2)^2 ≥ x + y)) →
  d ≥ smallestPositiveD :=
by
  intros h1 h2
  sorry

end smallest_d_l217_217856


namespace intersection_complement_eq_l217_217872

def A : Set ℝ := { x | 1 ≤ x ∧ x < 3 }

def B : Set ℝ := { x | x^2 ≥ 4 }

def complementB : Set ℝ := { x | -2 < x ∧ x < 2 }

def intersection (A : Set ℝ) (B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_complement_eq : 
  intersection A complementB = { x | 1 ≤ x ∧ x < 2 } := 
sorry

end intersection_complement_eq_l217_217872


namespace part1_part2_l217_217987

variable {x : ℝ}

/-- Prove that the range of the function f(x) = (sqrt(1+x) + sqrt(1-x) + 2) * (sqrt(1-x^2) + 1) for 0 ≤ x ≤ 1 is (0, 8]. -/
theorem part1 (hx : 0 ≤ x ∧ x ≤ 1) :
  0 < ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ∧ 
  ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ≤ 8 :=
sorry

/-- Prove that for 0 ≤ x ≤ 1, there exists a positive number β such that sqrt(1+x) + sqrt(1-x) ≤ 2 - x^2 / β, with the minimal β = 4. -/
theorem part2 (hx : 0 ≤ x ∧ x ≤ 1) :
  ∃ β : ℝ, β > 0 ∧ β = 4 ∧ (Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) :=
sorry

end part1_part2_l217_217987


namespace sum_of_squares_of_rates_l217_217960

theorem sum_of_squares_of_rates :
  ∃ (b j s : ℕ), 3 * b + j + 5 * s = 89 ∧ 4 * b + 3 * j + 2 * s = 106 ∧ b^2 + j^2 + s^2 = 821 := 
by
  sorry

end sum_of_squares_of_rates_l217_217960


namespace simplify_sqrt_sum_l217_217930

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l217_217930


namespace time_to_reach_julia_via_lee_l217_217468

theorem time_to_reach_julia_via_lee (d1 d2 d3 : ℕ) (t1 t2 : ℕ) :
  d1 = 2 → 
  t1 = 6 → 
  d3 = 3 → 
  (∀ v, v = d1 / t1) → 
  t2 = d3 / v → 
  t2 = 9 :=
by
  intros h1 h2 h3 hv ht2
  sorry

end time_to_reach_julia_via_lee_l217_217468


namespace fourth_term_is_fifteen_l217_217881

-- Define the problem parameters
variables (a d : ℕ)

-- Define the conditions
def sum_first_third_term : Prop := (a + (a + 2 * d) = 10)
def fourth_term_def : ℕ := a + 3 * d

-- Declare the theorem to be proved
theorem fourth_term_is_fifteen (h1 : sum_first_third_term a d) : fourth_term_def a d = 15 :=
sorry

end fourth_term_is_fifteen_l217_217881


namespace solution_l217_217203

noncomputable def problem (a b c x y z : ℝ) :=
  11 * x + b * y + c * z = 0 ∧
  a * x + 19 * y + c * z = 0 ∧
  a * x + b * y + 37 * z = 0 ∧
  a ≠ 11 ∧
  x ≠ 0

theorem solution (a b c x y z : ℝ) (h : problem a b c x y z) :
  (a / (a - 11)) + (b / (b - 19)) + (c / (c - 37)) = 1 :=
sorry

end solution_l217_217203


namespace necessary_but_not_sufficient_l217_217820

theorem necessary_but_not_sufficient (a b : ℝ) : (a > b) → (a + 1 > b - 2) :=
by sorry

end necessary_but_not_sufficient_l217_217820


namespace total_number_of_questions_l217_217212

theorem total_number_of_questions (N : ℕ)
  (hp : 0.8 * N = (4 / 5 : ℝ) * N)
  (hv : 35 = 35)
  (hb : (N / 2 : ℕ) = 1 * (N.div 2))
  (ha : N - 7 = N - 7) : N = 60 :=
by
  sorry

end total_number_of_questions_l217_217212


namespace problem_1_split_terms_problem_2_split_terms_l217_217345

-- Problem 1 Lean statement
theorem problem_1_split_terms :
  (28 + 5/7) + (-25 - 1/7) = 3 + 4/7 := 
  sorry
  
-- Problem 2 Lean statement
theorem problem_2_split_terms :
  (-2022 - 2/7) + (-2023 - 4/7) + 4046 - 1/7 = 0 := 
  sorry

end problem_1_split_terms_problem_2_split_terms_l217_217345


namespace algebraic_fraction_l217_217091

theorem algebraic_fraction (x : ℝ) (h1 : 1 / 3 = 1 / 3) 
(h2 : x / Real.pi = x / Real.pi) 
(h3 : 2 / (x + 3) = 2 / (x + 3))
(h4 : (x + 2) / 3 = (x + 2) / 3) 
: 
2 / (x + 3) = 2 / (x + 3) := sorry

end algebraic_fraction_l217_217091


namespace projectile_height_time_l217_217366

theorem projectile_height_time (h : ∀ t : ℝ, -16 * t^2 + 100 * t = 64 → t = 1) : (∃ t : ℝ, -16 * t^2 + 100 * t = 64 ∧ t = 1) :=
by sorry

end projectile_height_time_l217_217366


namespace neon_signs_blink_together_l217_217074

-- Define the time intervals for the blinks
def blink_interval1 : ℕ := 7
def blink_interval2 : ℕ := 11
def blink_interval3 : ℕ := 13

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- State the theorem
theorem neon_signs_blink_together : Nat.lcm (Nat.lcm blink_interval1 blink_interval2) blink_interval3 = 1001 := by
  sorry

end neon_signs_blink_together_l217_217074


namespace perfect_square_for_x_l217_217814

def expr (x : ℝ) : ℝ := 11.98 * 11.98 + 11.98 * x + 0.02 * 0.02

theorem perfect_square_for_x : expr 0.04 = (11.98 + 0.02) ^ 2 :=
by
  sorry

end perfect_square_for_x_l217_217814


namespace smallest_p_condition_l217_217981

theorem smallest_p_condition (n p : ℕ) (hn1 : n % 2 = 1) (hn2 : n % 7 = 5) (hp : (n + p) % 10 = 0) : p = 1 := by
  sorry

end smallest_p_condition_l217_217981


namespace value_of_k_l217_217707

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 6
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k * x - 8

theorem value_of_k:
  (f 5) - (g 5 k) = 20 → k = -10.8 :=
by
  sorry

end value_of_k_l217_217707


namespace perimeter_of_regular_polygon_l217_217695

def regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : Prop :=
  exterior_angle = 360 / n ∧ n * side_length > 0

theorem perimeter_of_regular_polygon :
  ∀ (n : ℕ) (side_length : ℝ), regular_polygon n side_length 45 → side_length = 7 → 8 = n → n * side_length = 56 :=
by
  intros n side_length h1 h2 h3
  rw [h2, h3]
  sorry

end perimeter_of_regular_polygon_l217_217695


namespace right_triangle_perimeter_l217_217118

theorem right_triangle_perimeter (area leg1 : ℕ) (h_area : area = 180) (h_leg1 : leg1 = 30) :
  ∃ leg2 hypotenuse perimeter, 
    (2 * area = leg1 * leg2) ∧ 
    (hypotenuse^2 = leg1^2 + leg2^2) ∧ 
    (perimeter = leg1 + leg2 + hypotenuse) ∧ 
    (perimeter = 42 + 2 * Real.sqrt 261) :=
by
  sorry

end right_triangle_perimeter_l217_217118


namespace fish_count_l217_217884

variables
  (x g s r : ℕ)
  (h1 : x - g = (2 / 3 : ℚ) * x - 1)
  (h2 : x - r = (2 / 3 : ℚ) * x + 4)
  (h3 : x = g + s + r)

theorem fish_count :
  s - g = 2 :=
by
  sorry

end fish_count_l217_217884


namespace find_m_n_l217_217942

noncomputable def m_n_sum (x : ℝ) (m n : ℤ) : Prop :=
  ∃ (m n : ℤ), (sec x + tan x = 3) ∧ (csc x + cot x = (m / n)) ∧ (gcd m n = 1) 

theorem find_m_n : ∀ (x : ℝ) (m n : ℤ), m_n_sum x m n → m + n = 3 :=
  begin
    sorry
  end

end find_m_n_l217_217942


namespace tom_seashells_l217_217381

theorem tom_seashells (days : ℕ) (seashells_per_day : ℕ) (h1 : days = 5) (h2 : seashells_per_day = 7) : 
  seashells_per_day * days = 35 := 
by
  sorry

end tom_seashells_l217_217381


namespace books_read_in_common_l217_217962

theorem books_read_in_common (T D B total X : ℕ) 
  (hT : T = 23) 
  (hD : D = 12) 
  (hB : B = 17) 
  (htotal : total = 47)
  (h_eq : (T - X) + (D - X) + B + 1 = total) : 
  X = 3 :=
by
  -- Here would go the proof details.
  sorry

end books_read_in_common_l217_217962


namespace tan_alpha_two_implies_fraction_eq_three_fourths_l217_217188

variable {α : ℝ}

theorem tan_alpha_two_implies_fraction_eq_three_fourths (h1 : Real.tan α = 2) (h2 : Real.cos α ≠ 0) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

end tan_alpha_two_implies_fraction_eq_three_fourths_l217_217188


namespace initial_investment_l217_217055

theorem initial_investment (A P : ℝ) (r : ℝ) (n t : ℕ) 
  (hA : A = 16537.5)
  (hr : r = 0.10)
  (hn : n = 2)
  (ht : t = 1)
  (hA_calc : A = P * (1 + r / n) ^ (n * t)) :
  P = 15000 :=
by {
  sorry
}

end initial_investment_l217_217055


namespace pencil_distribution_l217_217309

-- Formalize the problem in Lean
theorem pencil_distribution (x1 x2 x3 x4 : ℕ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 5) (hx2 : 1 ≤ x2 ∧ x2 ≤ 5) (hx3 : 1 ≤ x3 ∧ x3 ≤ 5) (hx4 : 1 ≤ x4 ∧ x4 ≤ 5) :
  x1 + x2 + x3 + x4 = 10 → 64 = 64 :=
by {
  sorry
}

end pencil_distribution_l217_217309


namespace LCM_20_45_75_is_900_l217_217660

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l217_217660


namespace no_prime_ratio_circle_l217_217084

theorem no_prime_ratio_circle (A : Fin 2007 → ℕ) :
  ¬ (∀ i : Fin 2007, (∃ p : ℕ, Nat.Prime p ∧ (p = A i / A ((i + 1) % 2007) ∨ p = A ((i + 1) % 2007) / A i))) := by
  sorry

end no_prime_ratio_circle_l217_217084


namespace percentage_increase_in_area_l217_217817

variable (L W : Real)

theorem percentage_increase_in_area (hL : L > 0) (hW : W > 0) :
  ((1 + 0.25) * L * (1 + 0.25) * W - L * W) / (L * W) * 100 = 56.25 := by
  sorry

end percentage_increase_in_area_l217_217817


namespace max_value_of_f_l217_217781

open Real

noncomputable def f (x : ℝ) : ℝ := -x - 9 / x + 18

theorem max_value_of_f : ∀ x > 0, f x ≤ 12 :=
by
  sorry

end max_value_of_f_l217_217781


namespace min_deliveries_l217_217605

theorem min_deliveries (cost_per_delivery_income: ℕ) (cost_per_delivery_gas: ℕ) (van_cost: ℕ) (d: ℕ) : 
  (d * (cost_per_delivery_income - cost_per_delivery_gas) ≥ van_cost) ↔ (d ≥ van_cost / (cost_per_delivery_income - cost_per_delivery_gas)) :=
by
  sorry

def john_deliveries : ℕ := 7500 / (15 - 5)

example : john_deliveries = 750 :=
by
  sorry

end min_deliveries_l217_217605


namespace probability_heads_10_out_of_12_l217_217081

theorem probability_heads_10_out_of_12 :
  let total_outcomes := (2^12 : ℕ)
  let favorable_outcomes := nat.choose 12 10
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 66 / 4096 :=
by 
  sorry

end probability_heads_10_out_of_12_l217_217081


namespace staircase_tile_cover_possible_l217_217503
-- Import the necessary Lean Lean libraries

-- We use natural numbers here
open Nat

-- Declare the problem as a theorem in Lean
theorem staircase_tile_cover_possible (m n : ℕ) (h_m : 6 ≤ m) (h_n : 6 ≤ n) :
  (∃ a b, m = 12 * a ∧ n = b ∧ a ≥ 1 ∧ b ≥ 6) ∨ 
  (∃ c d, m = 3 * c ∧ n = 4 * d ∧ c ≥ 2 ∧ d ≥ 3) :=
sorry

end staircase_tile_cover_possible_l217_217503


namespace exp_inequality_solution_l217_217186

theorem exp_inequality_solution (x : ℝ) (h : 1 < Real.exp x ∧ Real.exp x < 2) : 0 < x ∧ x < Real.log 2 :=
by
  sorry

end exp_inequality_solution_l217_217186


namespace total_sum_of_subsets_eq_512_l217_217178

open Finset

def M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem total_sum_of_subsets_eq_512 :
  let sum_sums := ∑ S in M.powerset \ {∅}, ∑ k in S, (-1)^k * k
  in sum_sums = 512 :=
by
  simp only [M]
  have sum_formula : ∑ k in M, (-1)^k * k = -2 + 4 - 6 + 8 - 10 + 12 - 14 + 16 - 18 := sorry -- the calculation step here
  have total_subsets : 2^7 = 128 := by norm_num
  exact 128 * sum_formula = 512 := sorry -- total sum calculation
  sorry

end total_sum_of_subsets_eq_512_l217_217178


namespace triangle_third_side_lengths_product_l217_217966

def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

def leg (c b : ℝ) : ℝ :=
  real.sqrt (c^2 - b^2)

theorem triangle_third_side_lengths_product :
  let a := 6
  let b := 8
  let hyp := hypotenuse a b
  let leg := leg b a
  real.round (hyp * leg * 10) / 10 = 52.9 :=
by {
  -- Definitions and calculations have been provided in the problem statement
  sorry
}

end triangle_third_side_lengths_product_l217_217966


namespace complete_square_l217_217620

theorem complete_square 
  (x : ℝ) : 
  (2 * x^2 - 3 * x - 1 = 0) → 
  ((x - (3/4))^2 = (17/16)) :=
sorry

end complete_square_l217_217620


namespace machines_needed_l217_217976

theorem machines_needed (x Y : ℝ) (R : ℝ) :
  (4 * R * 6 = x) → (M * R * 6 = Y) → M = 4 * Y / x :=
by
  intros h1 h2
  sorry

end machines_needed_l217_217976


namespace angle_at_630_is_15_degrees_l217_217085

-- Definitions for positions of hour and minute hands at 6:30 p.m.
def angle_per_hour : ℝ := 30
def minute_hand_position_630 : ℝ := 180
def hour_hand_position_630 : ℝ := 195

-- The angle between the hour hand and minute hand at 6:30 p.m.
def angle_between_hands_630 : ℝ := |hour_hand_position_630 - minute_hand_position_630|

-- Statement to prove
theorem angle_at_630_is_15_degrees :
  angle_between_hands_630 = 15 := by
  sorry

end angle_at_630_is_15_degrees_l217_217085


namespace carB_highest_avg_speed_l217_217541

-- Define the distances and times for each car
def distanceA : ℕ := 715
def timeA : ℕ := 11
def distanceB : ℕ := 820
def timeB : ℕ := 12
def distanceC : ℕ := 950
def timeC : ℕ := 14

-- Define the average speeds
def avgSpeedA : ℚ := distanceA / timeA
def avgSpeedB : ℚ := distanceB / timeB
def avgSpeedC : ℚ := distanceC / timeC

theorem carB_highest_avg_speed : avgSpeedB > avgSpeedA ∧ avgSpeedB > avgSpeedC :=
by
  -- Proof will be filled in here
  sorry

end carB_highest_avg_speed_l217_217541


namespace sum_of_all_possible_k_values_l217_217031

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l217_217031


namespace find_width_of_lot_l217_217745

noncomputable def volume_of_rectangular_prism (l w h : ℝ) : ℝ := l * w * h

theorem find_width_of_lot
  (l h v : ℝ)
  (h_len : l = 40)
  (h_height : h = 2)
  (h_volume : v = 1600)
  : ∃ w : ℝ, volume_of_rectangular_prism l w h = v ∧ w = 20 := by
  use 20
  simp [volume_of_rectangular_prism, h_len, h_height, h_volume]
  sorry

end find_width_of_lot_l217_217745


namespace sum_of_all_possible_k_values_l217_217032

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l217_217032


namespace negation_of_universal_statement_l217_217558

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^3 - 3 * x > 0) ↔ ∃ x : ℝ, x^3 - 3 * x ≤ 0 :=
by
  sorry

end negation_of_universal_statement_l217_217558


namespace original_digit_sum_six_and_product_is_1008_l217_217122

theorem original_digit_sum_six_and_product_is_1008 (x : ℕ) :
  (2 ∣ x / 10) → (4 ∣ x / 10) → 
  (x % 10 + (x / 10) = 6) →
  ((x % 10) * 10 + (x / 10)) * ((x / 10) * 10 + (x % 10)) = 1008 →
  x = 42 ∨ x = 24 :=
by
  intro h1 h2 h3 h4
  sorry


end original_digit_sum_six_and_product_is_1008_l217_217122


namespace smallest_d_l217_217398

theorem smallest_d (d : ℝ) : 
  (∃ d, d > 0 ∧ (4 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d - 2)^2))) → d = 2 :=
sorry

end smallest_d_l217_217398


namespace visitors_answered_questionnaire_l217_217018

theorem visitors_answered_questionnaire (V : ℕ) (h : (3 / 4 : ℝ) * V = (V : ℝ) - 110) : V = 440 :=
sorry

end visitors_answered_questionnaire_l217_217018


namespace sum_possible_k_l217_217039

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l217_217039


namespace karting_routes_10_min_l217_217947

-- Define the recursive function for M_{n, A}
def num_routes : ℕ → ℕ
| 0 => 1   -- Starting point at A for 0 minutes (0 routes)
| 1 => 0   -- Impossible to end at A in just 1 move
| 2 => 1   -- Only one way to go A -> B -> A in 2 minutes
| n + 1 =>
  if n = 1 then 0 -- Additional base case for n=2 as defined
  else if n = 2 then 1
  else num_routes (n - 1) + num_routes (n - 2)

theorem karting_routes_10_min : num_routes 10 = 34 := by
  -- Proof steps go here
  sorry

end karting_routes_10_min_l217_217947


namespace find_polynomials_g_l217_217059

-- Define functions f and proof target is g
def f (x : ℝ) : ℝ := x ^ 2

-- g is defined as an unknown polynomial with some constraints
variable (g : ℝ → ℝ)

-- The proof problem stating that if f(g(x)) = 9x^2 + 12x + 4, 
-- then g(x) = 3x + 2 or g(x) = -3x - 2
theorem find_polynomials_g (h : ∀ x : ℝ, f (g x) = 9 * x ^ 2 + 12 * x + 4) :
  (∀ x : ℝ, g x = 3 * x + 2) ∨ (∀ x : ℝ, g x = -3 * x - 2) := 
by
  sorry

end find_polynomials_g_l217_217059


namespace circles_tangent_area_l217_217282

noncomputable def triangle_area (r1 r2 r3 : ℝ) := 
  let d1 := r1 + r2
  let d2 := r2 + r3
  let d3 := r1 + r3
  let s := (d1 + d2 + d3) / 2
  (s * (s - d1) * (s - d2) * (s - d3)).sqrt

theorem circles_tangent_area :
  let r1 := 5
  let r2 := 12
  let r3 := 13
  let area := triangle_area r1 r2 r3 / (4 * (r1 + r2 + r3)).sqrt
  area = 120 / 25 := 
by 
  sorry

end circles_tangent_area_l217_217282


namespace mean_equivalence_l217_217067

theorem mean_equivalence :
  (20 + 30 + 40) / 3 = (23 + 30 + 37) / 3 :=
by sorry

end mean_equivalence_l217_217067


namespace proposition_1_proposition_2_proposition_3_proposition_4_l217_217294

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l217_217294


namespace length_de_l217_217811

theorem length_de (a b c d e : ℝ) (ab bc cd de ac ae : ℝ)
  (H1 : ab = 5)
  (H2 : bc = 2 * cd)
  (H3 : ac = ab + bc)
  (H4 : ac = 11)
  (H5 : ae = ab + bc + cd + de)
  (H6 : ae = 18) :
  de = 4 :=
by {
  sorry
}

-- Explanation:
-- a, b, c, d, e are points on a straight line
-- ab, bc, cd, de, ac, ae are lengths of segments between these points
-- H1: ab = 5
-- H2: bc = 2 * cd
-- H3: ac = ab + bc
-- H4: ac = 11
-- H5: ae = ab + bc + cd + de
-- H6: ae = 18
-- Prove that de = 4

end length_de_l217_217811


namespace sum_345_consecutive_sequences_l217_217456

theorem sum_345_consecutive_sequences :
  ∃ (n : ℕ), n = 7 ∧ (∀ (k : ℕ), n ≥ 2 →
    (n * (2 * k + n - 1) = 690 → 2 * k + n - 1 > n)) :=
sorry

end sum_345_consecutive_sequences_l217_217456


namespace sum_of_possible_ks_l217_217027

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l217_217027


namespace fraction_division_l217_217183

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := by
  sorry

end fraction_division_l217_217183


namespace augmented_matrix_solution_l217_217328

theorem augmented_matrix_solution (c₁ c₂ : ℝ) (x y : ℝ) 
  (h1 : 2 * x + 3 * y = c₁) (h2 : 3 * x + 2 * y = c₂)
  (hx : x = 2) (hy : y = 1) : c₁ - c₂ = -1 := 
by
  sorry

end augmented_matrix_solution_l217_217328


namespace log_expression_value_l217_217537

theorem log_expression_value : 
  (Real.logb 10 (Real.sqrt 2) + Real.logb 10 (Real.sqrt 5) + 2 ^ 0 + (5 ^ (1 / 3)) ^ 2 * Real.sqrt 5 = 13 / 2) := 
by 
  -- The proof is omitted as per the instructions
  sorry

end log_expression_value_l217_217537


namespace max_snowmen_l217_217683

def snowball_mass := Finset.range 100 \ {0}

def can_place_on (m1 m2 : ℕ) : Prop := 
  m1 ≥ 2 * m2

def is_snowman (s1 s2 s3 : ℕ) : Prop := 
  snowball_mass s1 ∧ snowball_mass s2 ∧ snowball_mass s3 ∧ 
  can_place_on s1 s2 ∧ can_place_on s2 s3

theorem max_snowmen : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ s ∈ S, ∃ s1 s2 s3, is_snowman s1 s2 s3 ∧ s = {s1, s2, s3}) ∧ 
    S.card = 24 :=
sorry

end max_snowmen_l217_217683


namespace car_with_highest_avg_speed_l217_217539

-- Conditions
def distance_A : ℕ := 715
def time_A : ℕ := 11
def distance_B : ℕ := 820
def time_B : ℕ := 12
def distance_C : ℕ := 950
def time_C : ℕ := 14

-- Average Speeds
def avg_speed_A : ℚ := distance_A / time_A
def avg_speed_B : ℚ := distance_B / time_B
def avg_speed_C : ℚ := distance_C / time_C

-- Theorem
theorem car_with_highest_avg_speed : avg_speed_B > avg_speed_A ∧ avg_speed_B > avg_speed_C :=
by
  sorry

end car_with_highest_avg_speed_l217_217539


namespace max_trig_expression_l217_217157

theorem max_trig_expression (A : ℝ) : (2 * Real.sin (A / 2) + Real.cos (A / 2) ≤ Real.sqrt 3) :=
sorry

end max_trig_expression_l217_217157


namespace domain_of_sqrt_cos_minus_half_correct_l217_217148

noncomputable def domain_of_sqrt_cos_minus_half (x : ℝ) : Prop :=
  ∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3

theorem domain_of_sqrt_cos_minus_half_correct :
  ∀ x, (∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3) ↔
    ∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3 :=
by sorry

end domain_of_sqrt_cos_minus_half_correct_l217_217148


namespace g_minus_1001_l217_217221

def g (x : ℝ) : ℝ := sorry

theorem g_minus_1001 :
  (∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x) →
  g 1 = 3 →
  g (-1001) = 1005 :=
by
  intros h1 h2
  sorry

end g_minus_1001_l217_217221


namespace anna_stamp_count_correct_l217_217534

-- Defining the initial counts of stamps
def anna_initial := 37
def alison_initial := 28
def jeff_initial := 31

-- Defining the operations
def alison_gives_half_to_anna := alison_initial / 2
def anna_after_receiving_from_alison := anna_initial + alison_gives_half_to_anna
def anna_final := anna_after_receiving_from_alison - 2 + 1

-- Formalizing the proof problem
theorem anna_stamp_count_correct : anna_final = 50 := by
  -- proof omitted
  sorry

end anna_stamp_count_correct_l217_217534


namespace sum_of_arithmetic_sequence_l217_217472

theorem sum_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 2 * a 4 * a 6 * a 8 = 120)
  (h2 : 1 / (a 4 * a 6 * a 8) + 1 / (a 2 * a 6 * a 8) + 1 / (a 2 * a 4 * a 8) + 1 / (a 2 * a 4 * a 6) = 7/60) :
  S 9 = 63/2 :=
by
  sorry

end sum_of_arithmetic_sequence_l217_217472


namespace number_of_insects_l217_217701

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 48) (h2 : legs_per_insect = 6) : (total_legs / legs_per_insect) = 8 := by
  sorry

end number_of_insects_l217_217701


namespace system_of_inequalities_l217_217776

theorem system_of_inequalities :
  ∃ (a b : ℤ), 
  (11 > 2 * a - b) ∧ 
  (25 > 2 * b - a) ∧ 
  (42 < 3 * b - a) ∧ 
  (46 < 2 * a + b) ∧ 
  (a = 14) ∧ 
  (b = 19) := 
sorry

end system_of_inequalities_l217_217776


namespace abs_value_equation_l217_217763

-- Define the main proof problem
theorem abs_value_equation (a b c d : ℝ)
  (h : ∀ x : ℝ, |2 * x + 4| + |a * x + b| = |c * x + d|) :
  d = 2 * c :=
sorry -- Proof skipped for this exercise

end abs_value_equation_l217_217763


namespace sin_x_eq_2ab_div_a2_plus_b2_l217_217860

theorem sin_x_eq_2ab_div_a2_plus_b2
  (a b : ℝ) (x : ℝ)
  (h_tan : Real.tan x = 2 * a * b / (a^2 - b^2))
  (h_pos : 0 < b) (h_lt : b < a) (h_x : 0 < x ∧ x < Real.pi / 2) :
  Real.sin x = 2 * a * b / (a^2 + b^2) :=
by sorry

end sin_x_eq_2ab_div_a2_plus_b2_l217_217860


namespace contradiction_in_triangle_l217_217480

theorem contradiction_in_triangle (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (sum_angles : A + B + C = 180) : false :=
by
  sorry

end contradiction_in_triangle_l217_217480


namespace xiao_ming_total_score_l217_217977

-- Definitions for the given conditions
def score_regular : ℝ := 70
def score_midterm : ℝ := 80
def score_final : ℝ := 85

def weight_regular : ℝ := 0.3
def weight_midterm : ℝ := 0.3
def weight_final : ℝ := 0.4

-- The statement that we need to prove
theorem xiao_ming_total_score : 
  (score_regular * weight_regular) + (score_midterm * weight_midterm) + (score_final * weight_final) = 79 := 
by
  sorry

end xiao_ming_total_score_l217_217977


namespace number_of_hydrogen_atoms_l217_217257

theorem number_of_hydrogen_atoms (C_atoms : ℕ) (O_atoms : ℕ) (molecular_weight : ℕ) 
    (C_weight : ℕ) (O_weight : ℕ) (H_weight : ℕ) : C_atoms = 3 → O_atoms = 1 → 
    molecular_weight = 58 → C_weight = 12 → O_weight = 16 → H_weight = 1 → 
    (molecular_weight - (C_atoms * C_weight + O_atoms * O_weight)) / H_weight = 6 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end number_of_hydrogen_atoms_l217_217257


namespace either_d_or_2d_is_perfect_square_l217_217630

theorem either_d_or_2d_is_perfect_square
  (a c d : ℕ) (hrel_prime : Nat.gcd a c = 1) (hd : ∃ D : ℝ, D = d ∧ (D:ℝ) > 0)
  (hdiam : d^2 = 2 * a^2 + c^2) :
  ∃ m : ℕ, m^2 = d ∨ m^2 = 2 * d :=
by
  sorry

end either_d_or_2d_is_perfect_square_l217_217630


namespace sum_base6_l217_217702

theorem sum_base6 (a b c : ℕ) 
  (ha : a = 1 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 1 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hc : c = 1 * 6^1 + 5 * 6^0) :
  a + b + c = 2 * 6^3 + 2 * 6^2 + 0 * 6^1 + 3 * 6^0 :=
by 
  sorry

end sum_base6_l217_217702


namespace bart_firewood_burning_period_l217_217302

-- We'll state the conditions as definitions.
def pieces_per_tree := 75
def trees_cut_down := 8
def logs_burned_per_day := 5

-- The theorem to prove the period Bart burns the logs.
theorem bart_firewood_burning_period :
  (trees_cut_down * pieces_per_tree) / logs_burned_per_day = 120 :=
by
  sorry

end bart_firewood_burning_period_l217_217302


namespace sum_tan_square_eq_sqrt_two_l217_217908

open Real

theorem sum_tan_square_eq_sqrt_two :
  let S := {x : ℝ | 0 < x ∧ x < π / 2 ∧
                (∃ a b c : ℝ, {sin x, cos x, tan x} = {a, b, c} ∧ a^2 + b^2 = c^2)} in
  ∑ x in S, tan x * tan x = sqrt 2 :=
by sorry

end sum_tan_square_eq_sqrt_two_l217_217908


namespace find_range_of_m_l217_217170

theorem find_range_of_m:
  (∀ x: ℝ, ¬ ∃ x: ℝ, x^2 + (m - 3) * x + 1 = 0) →
  (∀ y: ℝ, ¬ ∀ y: ℝ, x^2 + y^2 / (m - 1) = 1) → 
  1 < m ∧ m ≤ 2 :=
by
  sorry

end find_range_of_m_l217_217170


namespace ratio_payment_shared_side_l217_217134

variable (length_side length_back : ℕ) (cost_per_foot cole_payment : ℕ)
variables (neighbor_back_contrib neighbor_left_contrib total_cost_fence : ℕ)
variables (total_cost_shared_side : ℕ)

theorem ratio_payment_shared_side
  (h1 : length_side = 9)
  (h2 : length_back = 18)
  (h3 : cost_per_foot = 3)
  (h4 : cole_payment = 72)
  (h5 : neighbor_back_contrib = (length_back / 2) * cost_per_foot)
  (h6 : total_cost_fence = (2* length_side + length_back) * cost_per_foot)
  (h7 : total_cost_shared_side = length_side * cost_per_foot)
  (h8 : cole_left_total_payment = cole_payment + neighbor_back_contrib)
  (h9 : neighbor_left_contrib = cole_left_total_payment - cole_payment):
  neighbor_left_contrib / total_cost_shared_side = 1 := 
sorry

end ratio_payment_shared_side_l217_217134


namespace factorization1_factorization2_factorization3_factorization4_l217_217853

-- Question 1
theorem factorization1 (a b : ℝ) :
  4 * a^2 * b - 6 * a * b^2 = 2 * a * b * (2 * a - 3 * b) :=
by 
  sorry

-- Question 2
theorem factorization2 (x y : ℝ) :
  25 * x^2 - 9 * y^2 = (5 * x + 3 * y) * (5 * x - 3 * y) :=
by 
  sorry

-- Question 3
theorem factorization3 (a b : ℝ) :
  2 * a^2 * b - 8 * a * b^2 + 8 * b^3 = 2 * b * (a - 2 * b)^2 :=
by 
  sorry

-- Question 4
theorem factorization4 (x : ℝ) :
  (x + 2) * (x - 8) + 25 = (x - 3)^2 :=
by 
  sorry

end factorization1_factorization2_factorization3_factorization4_l217_217853


namespace exists_triangle_with_sin_angles_l217_217049

theorem exists_triangle_with_sin_angles (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a^4 + b^4 + c^4 + 4*a^2*b^2*c^2 = 2 * (a^2*b^2 + a^2*c^2 + b^2*c^2)) : 
    ∃ (α β γ : ℝ), α + β + γ = Real.pi ∧ Real.sin α = a ∧ Real.sin β = b ∧ Real.sin γ = c :=
by
  sorry

end exists_triangle_with_sin_angles_l217_217049


namespace cost_of_mixture_verify_cost_of_mixture_l217_217337

variables {C1 C2 Cm : ℝ}

def ratio := 5 / 12

axiom cost_of_rice_1 : C1 = 4.5
axiom cost_of_rice_2 : C2 = 8.75
axiom mix_ratio : ratio = 5 / 12

theorem cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = (8.75 * 5 + 4.5 * 12) / 17 :=
by sorry

-- Prove that the cost of the mixture Cm is indeed 5.75
theorem verify_cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = 5.75 :=
by sorry

end cost_of_mixture_verify_cost_of_mixture_l217_217337


namespace millicent_fraction_books_l217_217009

variable (M H : ℝ)
variable (F : ℝ)

-- Conditions
def harold_has_half_books (M H : ℝ) : Prop := H = (1 / 2) * M
def harold_brings_one_third_books (M H : ℝ) : Prop := (1 / 3) * H = (1 / 6) * M
def new_library_capacity (M F : ℝ) : Prop := (1 / 6) * M + F * M = (5 / 6) * M

-- Target Proof Statement
theorem millicent_fraction_books (M H F : ℝ) 
    (h1 : harold_has_half_books M H) 
    (h2 : harold_brings_one_third_books M H) 
    (h3 : new_library_capacity M F) : 
    F = 2 / 3 :=
sorry

end millicent_fraction_books_l217_217009


namespace trailing_zeros_in_15_factorial_base_15_are_3_l217_217143

/--
Compute the number of trailing zeros in \( 15! \) when expressed in base 15.
-/
def compute_trailing_zeros_in_factorial_base_15 : ℕ :=
  let num_factors_3 := (15 / 3) + (15 / 9)
  let num_factors_5 := (15 / 5)
  min num_factors_3 num_factors_5

theorem trailing_zeros_in_15_factorial_base_15_are_3 :
  compute_trailing_zeros_in_factorial_base_15 = 3 :=
sorry

end trailing_zeros_in_15_factorial_base_15_are_3_l217_217143


namespace seq_an_identity_l217_217900

theorem seq_an_identity (n : ℕ) (a : ℕ → ℕ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) > a n)
  (h₃ : ∀ n, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) 
  : a n = n^2 := sorry

end seq_an_identity_l217_217900


namespace log_piece_weight_l217_217466

variable (length_of_log : ℕ) (weight_per_foot : ℕ) (number_of_pieces : ℕ)
variable (original_length : length_of_log = 20)
variable (weight_per_linear_foot : weight_per_foot = 150)
variable (cuts_in_half : number_of_pieces = 2)

theorem log_piece_weight : (length_of_log / number_of_pieces) * weight_per_foot = 1500 := by
  have length_of_piece : length_of_log / number_of_pieces = 10 := by
    rw [original_length, cuts_in_half]
    norm_num
  rw [length_of_piece, weight_per_linear_foot]
  norm_num
  -- Proof complete

#print log_piece_weight

end log_piece_weight_l217_217466


namespace prove_a4_plus_1_div_a4_l217_217058

theorem prove_a4_plus_1_div_a4 (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/(a^4) = 7 :=
by
  sorry

end prove_a4_plus_1_div_a4_l217_217058


namespace sum_possible_k_l217_217038

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l217_217038


namespace cashback_discount_percentage_l217_217847

noncomputable def iphoneOriginalPrice : ℝ := 800
noncomputable def iwatchOriginalPrice : ℝ := 300
noncomputable def iphoneDiscountRate : ℝ := 0.15
noncomputable def iwatchDiscountRate : ℝ := 0.10
noncomputable def finalPrice : ℝ := 931

noncomputable def iphoneDiscountedPrice : ℝ := iphoneOriginalPrice * (1 - iphoneDiscountRate)
noncomputable def iwatchDiscountedPrice : ℝ := iwatchOriginalPrice * (1 - iwatchDiscountRate)
noncomputable def totalDiscountedPrice : ℝ := iphoneDiscountedPrice + iwatchDiscountedPrice
noncomputable def cashbackAmount : ℝ := totalDiscountedPrice - finalPrice
noncomputable def cashbackRate : ℝ := (cashbackAmount / totalDiscountedPrice) * 100

theorem cashback_discount_percentage : cashbackRate = 2 := by
  sorry

end cashback_discount_percentage_l217_217847


namespace lcm_20_45_75_l217_217663

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l217_217663


namespace height_of_wooden_box_l217_217267

theorem height_of_wooden_box 
  (height : ℝ)
  (h₁ : ∀ (length width : ℝ), length = 8 ∧ width = 10)
  (h₂ : ∀ (small_length small_width small_height : ℕ), small_length = 4 ∧ small_width = 5 ∧ small_height = 6)
  (h₃ : ∀ (num_boxes : ℕ), num_boxes = 4000000) :
  height = 6 := 
sorry

end height_of_wooden_box_l217_217267


namespace rectangle_and_square_problems_l217_217848

theorem rectangle_and_square_problems :
  ∃ (length width : ℝ), 
    (length / width = 2) ∧ 
    (length * width = 50) ∧ 
    (length = 10) ∧
    (width = 5) ∧
    ∃ (side_length : ℝ), 
      (side_length ^ 2 = 50) ∧ 
      (side_length - width = 5 * (Real.sqrt 2 - 1)) := 
by
  sorry

end rectangle_and_square_problems_l217_217848


namespace bert_made_1_dollar_l217_217129

def bert_earnings (selling_price tax_rate markup : ℝ) : ℝ :=
  selling_price - (tax_rate * selling_price) - (selling_price - markup)

theorem bert_made_1_dollar :
  bert_earnings 90 0.1 10 = 1 :=
by 
  sorry

end bert_made_1_dollar_l217_217129


namespace circle_diameter_percentage_l217_217363

theorem circle_diameter_percentage (d_R d_S : ℝ) 
    (h : π * (d_R / 2)^2 = 0.04 * π * (d_S / 2)^2) : 
    d_R = 0.4 * d_S :=
by
    sorry

end circle_diameter_percentage_l217_217363


namespace visual_range_increase_l217_217824

def percent_increase (original new : ℕ) : ℕ :=
  ((new - original) * 100) / original

theorem visual_range_increase :
  percent_increase 50 150 = 200 := 
by
  -- the proof would go here
  sorry

end visual_range_increase_l217_217824


namespace ivan_income_tax_l217_217753

-- Define the salary schedule
def first_two_months_salary: ℕ := 20000
def post_probation_salary: ℕ := 25000
def bonus_in_december: ℕ := 10000
def income_tax_rate: ℝ := 0.13

-- Define the total taxable income
def total_taxable_income: ℕ :=
  (first_two_months_salary * 2) + (post_probation_salary * 8) + bonus_in_december

-- Define the expected tax amount
def expected_tax: ℕ := 32500

-- Define the personal income tax calculation function
def calculate_tax (income: ℕ) (rate: ℝ): ℕ :=
  (income * rate).toInt

-- The statement which shows that the calculated tax is equal to the expected tax
theorem ivan_income_tax: calculate_tax total_taxable_income income_tax_rate = expected_tax := by
  -- Skip the actual proof
  sorry

end ivan_income_tax_l217_217753


namespace sum_not_equals_any_l217_217072

-- Define the nine special natural numbers a1 to a9
def a1 (k : ℕ) : ℕ := (10^k - 1) / 9
def a2 (m : ℕ) : ℕ := 2 * (10^m - 1) / 9
def a3 (p : ℕ) : ℕ := 3 * (10^p - 1) / 9
def a4 (q : ℕ) : ℕ := 4 * (10^q - 1) / 9
def a5 (r : ℕ) : ℕ := 5 * (10^r - 1) / 9
def a6 (s : ℕ) : ℕ := 6 * (10^s - 1) / 9
def a7 (t : ℕ) : ℕ := 7 * (10^t - 1) / 9
def a8 (u : ℕ) : ℕ := 8 * (10^u - 1) / 9
def a9 (v : ℕ) : ℕ := 9 * (10^v - 1) / 9

-- Statement of the problem
theorem sum_not_equals_any (k m p q r s t u v : ℕ) :
  ¬ (a1 k = a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a2 m = a1 k + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a3 p = a1 k + a2 m + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a4 q = a1 k + a2 m + a3 p + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a5 r = a1 k + a2 m + a3 p + a4 q + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a6 s = a1 k + a2 m + a3 p + a4 q + a5 r + a7 t + a8 u + a9 v) ∧
  ¬ (a7 t = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a8 u + a9 v) ∧
  ¬ (a8 u = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a9 v) ∧
  ¬ (a9 v = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u) :=
  sorry

end sum_not_equals_any_l217_217072


namespace sum_of_possible_ks_l217_217030

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l217_217030


namespace average_salary_all_workers_l217_217487

-- Definitions based on the conditions
def num_technicians : ℕ := 7
def num_other_workers : ℕ := 7
def avg_salary_technicians : ℕ := 12000
def avg_salary_other_workers : ℕ := 8000
def total_workers : ℕ := 14

-- Total salary calculations based on the conditions
def total_salary_technicians : ℕ := num_technicians * avg_salary_technicians
def total_salary_other_workers : ℕ := num_other_workers * avg_salary_other_workers
def total_salary_all_workers : ℕ := total_salary_technicians + total_salary_other_workers

-- The statement to be proved
theorem average_salary_all_workers : total_salary_all_workers / total_workers = 10000 :=
by
  -- proof will be added here
  sorry

end average_salary_all_workers_l217_217487


namespace problem_statement_l217_217215

-- Define the sides of the original triangle
def side_5 := 5
def side_12 := 12
def side_13 := 13

-- Define the perimeters of the isosceles triangles
def P := 3 * side_5
def Q := 3 * side_12
def R := 3 * side_13

-- Statement we want to prove
theorem problem_statement : P + R = (3 / 2) * Q := by
  sorry

end problem_statement_l217_217215


namespace find_initial_money_l217_217411
 
theorem find_initial_money (x : ℕ) (gift_grandma gift_aunt_uncle gift_parents total_money : ℕ) 
  (h1 : gift_grandma = 25) 
  (h2 : gift_aunt_uncle = 20) 
  (h3 : gift_parents = 75) 
  (h4 : total_money = 279) 
  (h : x + (gift_grandma + gift_aunt_uncle + gift_parents) = total_money) : 
  x = 159 :=
by
  sorry

end find_initial_money_l217_217411


namespace max_c_for_range_l217_217156

theorem max_c_for_range (c : ℝ) :
  (∃ x : ℝ, (x^2 - 7*x + c = 2)) → c ≤ 57 / 4 :=
by
  sorry

end max_c_for_range_l217_217156


namespace back_wheel_revolutions_calculation_l217_217047

noncomputable def front_diameter : ℝ := 3 -- Diameter of the front wheel in feet
noncomputable def back_diameter : ℝ := 0.5 -- Diameter of the back wheel in feet
noncomputable def no_slippage : Prop := true -- No slippage condition
noncomputable def front_revolutions : ℕ := 150 -- Number of front wheel revolutions

theorem back_wheel_revolutions_calculation 
  (d_f : ℝ) (d_b : ℝ) (slippage : Prop) (n_f : ℕ) : 
  slippage → d_f = front_diameter → d_b = back_diameter → 
  n_f = front_revolutions → 
  ∃ n_b : ℕ, n_b = 900 := 
by
  sorry

end back_wheel_revolutions_calculation_l217_217047


namespace prime_quadratic_root_range_l217_217607

theorem prime_quadratic_root_range (p : ℕ) (hprime : Prime p) 
  (hroots : ∃ x1 x2 : ℤ, x1 * x2 = -580 * p ∧ x1 + x2 = p) : 20 < p ∧ p < 30 :=
by
  sorry

end prime_quadratic_root_range_l217_217607


namespace max_number_of_snowmen_l217_217686

theorem max_number_of_snowmen :
  let snowballs := (list.range' 1 99.succ).map (λ x, (x : ℕ))
  ∃ (k : ℕ), k = 24 ∧ 
  ∀ (s : list (list ℕ)), 
    (∀ t ∈ s, t.length = 3 ∧ 
              (t.nth 0).get_or_else 0 >= ((t.nth 1).get_or_else 0 / 2) ∧ 
              (t.nth 1).get_or_else 0 >= ((t.nth 2).get_or_else 0 / 2)) →
    (∀ e ∈ s.join, e ∈ snowballs) →
    s.length ≤ k :=
by
  sorry

end max_number_of_snowmen_l217_217686


namespace sum_possible_values_k_l217_217036

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l217_217036


namespace shoe_store_sale_l217_217549

theorem shoe_store_sale (total_sneakers : ℕ) (total_sandals : ℕ) (total_shoes : ℕ) (total_boots : ℕ) 
  (h1 : total_sneakers = 2) 
  (h2 : total_sandals = 4) 
  (h3 : total_shoes = 17) 
  (h4 : total_boots = total_shoes - (total_sneakers + total_sandals)) : 
  total_boots = 11 :=
by
  rw [h1, h2, h3] at h4
  exact h4
-- sorry

end shoe_store_sale_l217_217549


namespace master_craftsman_parts_l217_217897

/-- 
Given:
  (1) the master craftsman produces 35 parts in the first hour,
  (2) at the rate of 35 parts/hr, he would be one hour late to meet the quota,
  (3) by increasing his speed by 15 parts/hr, he finishes the quota 0.5 hours early,
Prove that the total number of parts manufactured during the shift is 210.
-/
theorem master_craftsman_parts (N : ℕ) (quota : ℕ) 
  (initial_rate : ℕ := 35)
  (increased_rate_diff : ℕ := 15)
  (extra_time_slow : ℕ := 1)
  (time_saved_fast : ℕ := 1/2) :
  (quota = initial_rate * (extra_time_slow + 1) + N ∧
   increased_rate_diff = 15 ∧
   increased_rate_diff = λ (x : ℕ), initial_rate + x ∧
   time_saved_fast = 1/2 ∧
   N = 35) →
  quota = 210 := 
by
  sorry

end master_craftsman_parts_l217_217897


namespace select_twins_in_grid_l217_217797

theorem select_twins_in_grid (persons : Fin 8 × Fin 8 → Fin 2) :
  ∃ (selection : Fin 8 × Fin 8 → Bool), 
    (∀ i : Fin 8, ∃ j : Fin 8, selection (i, j) = true) ∧ 
    (∀ j : Fin 8, ∃ i : Fin 8, selection (i, j) = true) :=
sorry

end select_twins_in_grid_l217_217797


namespace base_six_to_base_ten_equivalent_l217_217968

theorem base_six_to_base_ten_equivalent :
  let n := 12345
  (5 * 6^0 + 4 * 6^1 + 3 * 6^2 + 2 * 6^3 + 1 * 6^4) = 1865 :=
by
  sorry

end base_six_to_base_ten_equivalent_l217_217968


namespace sequence_an_square_l217_217898

theorem sequence_an_square (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) > a n) 
  (h3 : ∀ n : ℕ, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) :
  ∀ n : ℕ, a n = n^2 :=
by
  sorry

end sequence_an_square_l217_217898


namespace fraction_option_C_l217_217093

def is_fraction (expr : String) : Prop := 
  expr = "fraction"

def option_C_fraction (x : ℝ) : Prop :=
  ∃ (numerator : ℝ), ∃ (denominator : ℝ), 
  numerator = 2 ∧ denominator = x + 3

theorem fraction_option_C (x : ℝ) (h : x ≠ -3) :
  is_fraction "fraction" ↔ option_C_fraction x :=
by 
  sorry

end fraction_option_C_l217_217093


namespace remainder_when_divided_by_13_l217_217671

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) : (N = 39 * k + 17) → (N % 13 = 4) := by
  sorry

end remainder_when_divided_by_13_l217_217671


namespace count_multiples_4_or_5_not_20_l217_217729

-- We define the necessary ranges and conditions
def is_multiple_of (n k : ℕ) := n % k = 0

def count_multiples (n k : ℕ) := (n / k)

def not_multiple_of (n k : ℕ) := ¬ is_multiple_of n k

def count_multiples_excluding (n k l : ℕ) :=
  count_multiples n k + count_multiples n l - count_multiples n (Nat.lcm k l)

theorem count_multiples_4_or_5_not_20 : count_multiples_excluding 3010 4 5 = 1204 := 
by
  sorry

end count_multiples_4_or_5_not_20_l217_217729


namespace child_B_share_l217_217261

theorem child_B_share (total_money : ℕ) (ratio_A ratio_B ratio_C ratio_D ratio_E total_parts : ℕ) 
  (h1 : total_money = 12000)
  (h2 : ratio_A = 2)
  (h3 : ratio_B = 3)
  (h4 : ratio_C = 4)
  (h5 : ratio_D = 5)
  (h6 : ratio_E = 6)
  (h_total_parts : total_parts = ratio_A + ratio_B + ratio_C + ratio_D + ratio_E) :
  (total_money / total_parts) * ratio_B = 1800 :=
by
  sorry

end child_B_share_l217_217261


namespace remainder_when_squared_l217_217815

theorem remainder_when_squared (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := by
  sorry

end remainder_when_squared_l217_217815


namespace vacation_cost_split_l217_217348

theorem vacation_cost_split 
  (airbnb_cost : ℕ)
  (car_rental_cost : ℕ)
  (people : ℕ)
  (split_equally : Prop)
  (h1 : airbnb_cost = 3200)
  (h2 : car_rental_cost = 800)
  (h3 : people = 8)
  (h4 : split_equally)
  : (airbnb_cost + car_rental_cost) / people = 500 :=
by
  sorry

end vacation_cost_split_l217_217348


namespace expressions_equal_l217_217150

theorem expressions_equal {x y z : ℤ} : (x + 2 * y * z = (x + y) * (x + 2 * z)) ↔ (x + y + 2 * z = 1) :=
by
  sorry

end expressions_equal_l217_217150


namespace slope_of_line_through_focus_of_parabola_l217_217259

theorem slope_of_line_through_focus_of_parabola
  (C : (x y : ℝ) → y^2 = 4 * x)
  (F : (ℝ × ℝ) := (1, 0))
  (A B : (ℝ × ℝ))
  (l : ℝ → ℝ)
  (intersects : (x : ℝ) → (l x) ^ 2 = 4 * x)
  (passes_through_focus : l 1 = 0)
  (distance_condition : ∀ (d1 d2 : ℝ), d1 = 4 * d2 → dist F A = d1 ∧ dist F B = d2) :
  ∃ k : ℝ, (∀ (x : ℝ), l x = k * (x - 1)) ∧ (k = 4 / 3 ∨ k = -4 / 3) :=
by
  sorry

end slope_of_line_through_focus_of_parabola_l217_217259


namespace recycling_money_l217_217717

theorem recycling_money (cans_per_unit : ℕ) (payment_per_unit_cans : ℝ) 
  (newspapers_per_unit : ℕ) (payment_per_unit_newspapers : ℝ) 
  (total_cans : ℕ) (total_newspapers : ℕ) : 
  cans_per_unit = 12 → payment_per_unit_cans = 0.50 → 
  newspapers_per_unit = 5 → payment_per_unit_newspapers = 1.50 → 
  total_cans = 144 → total_newspapers = 20 → 
  (total_cans / cans_per_unit) * payment_per_unit_cans + 
  (total_newspapers / newspapers_per_unit) * payment_per_unit_newspapers = 12 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end recycling_money_l217_217717


namespace binomial_20_5_l217_217284

theorem binomial_20_5 : Nat.choose 20 5 = 15504 := by
  sorry

end binomial_20_5_l217_217284


namespace compute_mod_expression_l217_217281

theorem compute_mod_expression :
  (3 * (1 / 7) + 9 * (1 / 13)) % 72 = 18 := sorry

end compute_mod_expression_l217_217281


namespace james_weekly_pistachio_cost_l217_217757

def cost_per_can : ℕ := 10
def ounces_per_can : ℕ := 5
def consumption_per_5_days : ℕ := 30
def days_per_week : ℕ := 7

theorem james_weekly_pistachio_cost : (days_per_week / 5 * consumption_per_5_days) / ounces_per_can * cost_per_can = 90 := 
by
  sorry

end james_weekly_pistachio_cost_l217_217757


namespace range_of_x_satisfying_inequality_l217_217571

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * |x - 1|

theorem range_of_x_satisfying_inequality :
  {x : ℝ | f x > f (2 * x)} = {x : ℝ | 0 < x ∧ x < (2 : ℝ) / 3} :=
by
  sorry

end range_of_x_satisfying_inequality_l217_217571


namespace find_y_l217_217060

theorem find_y (t : ℝ) (x : ℝ := 3 - 2 * t) (y : ℝ := 5 * t + 6) (h : x = 1) : y = 11 :=
by
  sorry

end find_y_l217_217060


namespace math_problem_l217_217163

theorem math_problem (x y : ℝ) :
  let A := x^3 + 3*x^2*y + y^3 - 3*x*y^2
  let B := x^2*y - x*y^2
  A - 3*B = x^3 + y^3 := by
  sorry

end math_problem_l217_217163


namespace find_x_l217_217512

theorem find_x (x : ℝ) (h : 0.75 * x = (1 / 3) * x + 110) : x = 264 :=
sorry

end find_x_l217_217512


namespace predict_monthly_savings_l217_217114

noncomputable def sum_x_i := 80
noncomputable def sum_y_i := 20
noncomputable def sum_x_i_y_i := 184
noncomputable def sum_x_i_sq := 720
noncomputable def n := 10
noncomputable def x_bar := sum_x_i / n
noncomputable def y_bar := sum_y_i / n
noncomputable def b := (sum_x_i_y_i - n * x_bar * y_bar) / (sum_x_i_sq - n * x_bar^2)
noncomputable def a := y_bar - b * x_bar
noncomputable def regression_eqn(x: ℝ) := b * x + a

theorem predict_monthly_savings :
  regression_eqn 7 = 1.7 :=
by
  sorry

end predict_monthly_savings_l217_217114


namespace jewelry_store_total_cost_l217_217258

theorem jewelry_store_total_cost :
  let necklaces_needed := 7
  let rings_needed := 12
  let bracelets_needed := 7
  let necklace_price := 4
  let ring_price := 10
  let bracelet_price := 5
  let necklace_discount := if necklaces_needed >= 6 then 0.15 else if necklaces_needed >= 4 then 0.10 else 0
  let ring_discount := if rings_needed >= 20 then 0.10 else if rings_needed >= 10 then 0.05 else 0
  let bracelet_discount := if bracelets_needed >= 10 then 0.12 else if bracelets_needed >= 7 then 0.08 else 0
  let necklace_cost := necklaces_needed * (necklace_price * (1 - necklace_discount))
  let ring_cost := rings_needed * (ring_price * (1 - ring_discount))
  let bracelet_cost := bracelets_needed * (bracelet_price * (1 - bracelet_discount))
  let total_cost := necklace_cost + ring_cost + bracelet_cost
  total_cost = 170 := by
  -- calculation details omitted
  sorry

end jewelry_store_total_cost_l217_217258


namespace find_a_range_of_a_l217_217435

noncomputable def f (x a : ℝ) := x + a * Real.log x

-- Proof problem 1: Prove that a = 2 given f' (1) = 3 for f (x) = x + a log x
theorem find_a (a : ℝ) : 
  (1 + a = 3) → (a = 2) := sorry

-- Proof problem 2: Prove that the range of a such that f(x) ≥ a always holds is [-e^2, 0]
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ a) → (-Real.exp 2 ≤ a ∧ a ≤ 0) := sorry

end find_a_range_of_a_l217_217435


namespace quarters_needed_l217_217522

-- Define the cost of items in cents and declare the number of items to purchase.
def quarter_value : ℕ := 25
def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50

def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Theorem stating the number of quarters needed to buy the given items.
theorem quarters_needed : 
  (num_candy_bars * candy_bar_cost + num_chocolates * chocolate_cost + num_juice_packs * juice_cost) / quarter_value = 11 := 
sorry

end quarters_needed_l217_217522


namespace tan_double_angle_l217_217873

theorem tan_double_angle (α : ℝ) (h : 3 * Real.cos α + Real.sin α = 0) : 
    Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_double_angle_l217_217873


namespace anna_final_stamp_count_l217_217532

theorem anna_final_stamp_count (anna_initial : ℕ) (alison_initial : ℕ) (jeff_initial : ℕ)
  (anna_receive_from_alison : ℕ) (anna_give_jeff : ℕ) (anna_receive_jeff : ℕ) :
  anna_initial = 37 →
  alison_initial = 28 →
  jeff_initial = 31 →
  anna_receive_from_alison = alison_initial / 2 →
  anna_give_jeff = 2 →
  anna_receive_jeff = 1 →
  ∃ result : ℕ, result = 50 :=
by
  intros
  sorry

end anna_final_stamp_count_l217_217532


namespace inequality_sum_l217_217917

open Real
open BigOperators

theorem inequality_sum 
  (n : ℕ) 
  (h : n > 1) 
  (x : Fin n → ℝ)
  (hx1 : ∀ i, 0 < x i) 
  (hx2 : ∑ i, x i = 1) :
  ∑ i, x i / sqrt (1 - x i) ≥ (∑ i, sqrt (x i)) / sqrt (n - 1) :=
sorry

end inequality_sum_l217_217917


namespace expression_divisible_by_3_l217_217321

theorem expression_divisible_by_3 (k : ℤ) : ∃ m : ℤ, (2 * k + 3)^2 - 4 * k^2 = 3 * m :=
by
  sorry

end expression_divisible_by_3_l217_217321


namespace base_eight_to_base_ten_l217_217237

theorem base_eight_to_base_ten : ∃ n : ℕ, 47 = 4 * 8 + 7 ∧ n = 39 :=
by
  sorry

end base_eight_to_base_ten_l217_217237


namespace edge_ratio_of_cubes_l217_217813

theorem edge_ratio_of_cubes (a b : ℝ) (h : (a^3) / (b^3) = 64) : a / b = 4 :=
sorry

end edge_ratio_of_cubes_l217_217813


namespace number_of_truthful_dwarfs_l217_217552

-- Given conditions
variables (D : Type) [Fintype D] [DecidableEq D] [Card D = 10]
variables (IceCream : Type) [DecidableEq IceCream] (vanilla chocolate fruit : IceCream)
-- Assuming each dwarf likes exactly one type of ice cream
variable (Likes : D → IceCream)
-- Functions indicating if a dwarf raised their hand for each type of ice cream
variables (raisedHandForVanilla raisedHandForChocolate raisedHandForFruit : D → Prop)

-- Given conditions translated to Lean
axiom all_dwarfs_raised_for_vanilla : ∀ d, raisedHandForVanilla d
axiom half_dwarfs_raised_for_chocolate : Fintype.card {d // raisedHandForChocolate d} = 5
axiom one_dwarf_raised_for_fruit : Fintype.card {d // raisedHandForFruit d} = 1

-- Define that a dwarf either always tells the truth or always lies
inductive TruthStatus
| truthful : TruthStatus
| liar : TruthStatus

variable (Status : D → TruthStatus)

-- Definitions related to hand-raising based on dwarf's status and ice cream they like
def raisedHandCorrectly (d : D) : Prop :=
  match Status d with
  | TruthStatus.truthful => 
      raisedHandForVanilla d ↔ Likes d = vanilla ∧
      raisedHandForChocolate d ↔ Likes d = chocolate ∧
      raisedHandForFruit d ↔ Likes d = fruit
  | TruthStatus.liar =>
      raisedHandForVanilla d ↔ Likes d ≠ vanilla ∧
      raisedHandForChocolate d ↔ Likes d ≠ chocolate ∧
      raisedHandForFruit d ↔ Likes d ≠ fruit

-- Goal to prove
theorem number_of_truthful_dwarfs : Fintype.card {d // Status d = TruthStatus.truthful} = 4 :=
by sorry

end number_of_truthful_dwarfs_l217_217552


namespace find_f_at_one_l217_217433

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ := m * x^3 + n * x + 1

theorem find_f_at_one (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : f (-1) m n = 5) : f (1) m n = 7 :=
by
  -- proof goes here
  sorry

end find_f_at_one_l217_217433


namespace neg_exponent_reciprocal_l217_217383

theorem neg_exponent_reciprocal : (2 : ℝ) ^ (-1 : ℤ) = 1 / 2 := by
  -- Insert your proof here
  sorry

end neg_exponent_reciprocal_l217_217383


namespace cat_food_per_day_l217_217770

theorem cat_food_per_day
  (bowl_empty_weight : ℕ)
  (bowl_weight_after_eating : ℕ)
  (food_eaten : ℕ)
  (days_per_fill : ℕ)
  (daily_food : ℕ) :
  (bowl_empty_weight = 420) →
  (bowl_weight_after_eating = 586) →
  (food_eaten = 14) →
  (days_per_fill = 3) →
  (bowl_weight_after_eating - bowl_empty_weight + food_eaten = days_per_fill * daily_food) →
  daily_food = 60 :=
by
  sorry

end cat_food_per_day_l217_217770


namespace simplify_and_evaluate_l217_217618

variable (a : ℝ)
axiom a_cond : a = -1 / 3

theorem simplify_and_evaluate : (3 * a - 1) ^ 2 + 3 * a * (3 * a + 2) = 3 :=
by
  have ha : a = -1 / 3 := a_cond
  sorry

end simplify_and_evaluate_l217_217618


namespace divisible_2n_minus_3_l217_217774

theorem divisible_2n_minus_3 (n : ℕ) : (2^n - 1)^n - 3 ≡ 0 [MOD 2^n - 3] :=
by
  sorry

end divisible_2n_minus_3_l217_217774


namespace simplify_cosine_tangent_product_of_cosines_l217_217483

-- Problem 1
theorem simplify_cosine_tangent :
  Real.cos 40 * (1 + Real.sqrt 3 * Real.tan 10) = 1 :=
sorry

-- Problem 2
theorem product_of_cosines :
  (Real.cos (2 * Real.pi / 7)) * (Real.cos (4 * Real.pi / 7)) * (Real.cos (6 * Real.pi / 7)) = 1 / 8 :=
sorry

end simplify_cosine_tangent_product_of_cosines_l217_217483


namespace bottles_per_case_l217_217995

theorem bottles_per_case (total_bottles_per_day : ℕ) (cases_required : ℕ) (bottles_per_case : ℕ)
  (h1 : total_bottles_per_day = 65000)
  (h2 : cases_required = 5000) :
  bottles_per_case = total_bottles_per_day / cases_required :=
by
  sorry

end bottles_per_case_l217_217995


namespace multiply_63_57_l217_217138

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l217_217138


namespace find_m_n_value_l217_217576

theorem find_m_n_value (x m n : ℝ) 
  (h1 : x - 3 * m < 0) 
  (h2 : n - 2 * x < 0) 
  (h3 : -1 < x)
  (h4 : x < 3) 
  : (m + n) ^ 2023 = -1 :=
sorry

end find_m_n_value_l217_217576


namespace cleaner_used_after_30_minutes_l217_217692

-- Define function to calculate the total amount of cleaner used
def total_cleaner_used (time: ℕ) (rate1: ℕ) (time1: ℕ) (rate2: ℕ) (time2: ℕ) (rate3: ℕ) (time3: ℕ) : ℕ :=
  (rate1 * time1) + (rate2 * time2) + (rate3 * time3)

-- The main theorem statement
theorem cleaner_used_after_30_minutes : total_cleaner_used 30 2 15 3 10 4 5 = 80 := by
  -- insert proof here
  sorry

end cleaner_used_after_30_minutes_l217_217692


namespace simplify_sqrt_sum_l217_217939

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l217_217939


namespace cartesian_equation_of_parametric_l217_217855

variable (t : ℝ) (x y : ℝ)

open Real

theorem cartesian_equation_of_parametric 
  (h1 : x = sqrt t)
  (h2 : y = 2 * sqrt (1 - t))
  (h3 : 0 ≤ t ∧ t ≤ 1) :
  (x^2 / 1) + (y^2 / 4) = 1 := by 
  sorry

end cartesian_equation_of_parametric_l217_217855


namespace male_students_in_grade_l217_217526

-- Define the total number of students and the number of students in the sample
def total_students : ℕ := 1200
def sample_students : ℕ := 30

-- Define the number of female students in the sample
def female_students_sample : ℕ := 14

-- Calculate the number of male students in the sample
def male_students_sample := sample_students - female_students_sample

-- State the main theorem
theorem male_students_in_grade :
  (male_students_sample : ℕ) * total_students / sample_students = 640 :=
by
  -- placeholder for calculations based on provided conditions
  sorry

end male_students_in_grade_l217_217526


namespace num_digits_divisible_l217_217308

theorem num_digits_divisible (h : Nat) :
  (∃ n : Fin 10, (10 * 24 + n) % n = 0) -> h = 7 :=
by sorry

end num_digits_divisible_l217_217308


namespace f_10_l217_217450

noncomputable def f : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * f n

theorem f_10 : f 10 = 2^10 :=
by
  -- This would be filled in with the necessary proof steps to show f(10) = 2^10
  sorry

end f_10_l217_217450


namespace cost_price_per_meter_l217_217507

-- We define the given conditions
def meters_sold : ℕ := 60
def selling_price : ℕ := 8400
def profit_per_meter : ℕ := 12

-- We need to prove that the cost price per meter is Rs. 128
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 128 :=
by
  sorry

end cost_price_per_meter_l217_217507


namespace fuel_for_first_third_l217_217206

def total_fuel : ℕ := 60
def fuel_second_third : ℕ := total_fuel / 3
def fuel_final_third : ℕ := fuel_second_third / 2
def fuel_first_third : ℕ := total_fuel - fuel_second_third - fuel_final_third

theorem fuel_for_first_third (total_fuel : ℕ) : 
  (total_fuel = 60) → 
  (fuel_first_third = total_fuel - (total_fuel / 3) - (total_fuel / 6)) →
  fuel_first_third = 30 := 
by 
  intros h1 h2
  rw h1 at h2
  norm_num at h2
  exact h2

end fuel_for_first_third_l217_217206


namespace airplane_speeds_l217_217078

theorem airplane_speeds (v : ℝ) 
  (h1 : 2.5 * v + 2.5 * 250 = 1625) : 
  v = 400 := 
sorry

end airplane_speeds_l217_217078


namespace general_formula_sum_of_first_10_terms_l217_217885

variable (a : ℕ → ℝ) (d : ℝ) (S_10 : ℝ)
variable (h1 : a 5 = 11) (h2 : a 8 = 5)

theorem general_formula (n : ℕ) : a n = -2 * n + 21 :=
sorry

theorem sum_of_first_10_terms : S_10 = 100 :=
sorry

end general_formula_sum_of_first_10_terms_l217_217885


namespace fractional_equation_solution_l217_217788

theorem fractional_equation_solution (x : ℝ) (h₁ : x ≠ 0) : (1 / x = 2 / (x + 3)) → x = 3 := by
  sorry

end fractional_equation_solution_l217_217788


namespace james_spends_on_pistachios_per_week_l217_217754

theorem james_spends_on_pistachios_per_week :
  let cost_per_can := 10
  let ounces_per_can := 5
  let total_ounces_per_5_days := 30
  let days_per_week := 7
  let cost_per_ounce := cost_per_can / ounces_per_can
  let daily_ounces := total_ounces_per_5_days / 5
  let daily_cost := daily_ounces * cost_per_ounce
  daily_cost * days_per_week = 84 :=
by
  sorry

end james_spends_on_pistachios_per_week_l217_217754


namespace not_proportional_eqn_exists_l217_217416

theorem not_proportional_eqn_exists :
  ∀ (x y : ℝ), (4 * x + 2 * y = 8) → ¬ ((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) :=
by
  intros x y h
  sorry

end not_proportional_eqn_exists_l217_217416


namespace proof_problem_l217_217297

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l217_217297


namespace tournament_committees_l217_217883

-- Assuming each team has 7 members
def team_members : Nat := 7

-- There are 5 teams
def total_teams : Nat := 5

-- The host team selects 3 members including at least one woman
def select_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 3
  let all_men_combinations := Nat.choose (team_members - 1) 3
  total_combinations - all_men_combinations

-- Each non-host team selects 2 members including at least one woman
def select_non_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 2
  let all_men_combinations := Nat.choose (team_members - 1) 2
  total_combinations - all_men_combinations

-- Total number of committees when one team is the host
def one_team_host_total_combinations (w m : Nat) : ℕ :=
  select_host_team_members w m * (select_non_host_team_members w m) ^ (total_teams - 1)

-- Total number of possible 11-member tournament committees
def total_committees (w m : Nat) : ℕ :=
  one_team_host_total_combinations w m * total_teams

theorem tournament_committees (w m : Nat) (hw : w ≥ 1) (hm : m ≤ 6) :
  total_committees w m = 97200 :=
by
  sorry

end tournament_committees_l217_217883


namespace find_larger_number_l217_217948

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2415) (h2 : L = 21 * S + 15) : L = 2535 := 
by
  sorry

end find_larger_number_l217_217948


namespace reciprocal_of_3_div_2_l217_217972

def reciprocal (a : ℚ) : ℚ := a⁻¹

theorem reciprocal_of_3_div_2 : reciprocal (3 / 2) = 2 / 3 :=
by
  -- proof would go here
  sorry

end reciprocal_of_3_div_2_l217_217972


namespace simplify_nested_sqrt_l217_217934

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l217_217934


namespace find_positive_number_l217_217250

noncomputable def solve_number (x : ℝ) : Prop :=
  (2/3 * x = 64/216 * (1/x)) ∧ (x > 0)

theorem find_positive_number (x : ℝ) : solve_number x → x = (2/9) * Real.sqrt 3 :=
  by
  sorry

end find_positive_number_l217_217250


namespace percent_games_lost_l217_217632

def games_ratio (won lost : ℕ) : Prop :=
  won * 3 = lost * 7

def total_games (won lost : ℕ) : Prop :=
  won + lost = 50

def percentage_lost (lost total : ℕ) : ℕ :=
  lost * 100 / total

theorem percent_games_lost (won lost : ℕ) (h1 : games_ratio won lost) (h2 : total_games won lost) : 
  percentage_lost lost 50 = 30 := 
by
  sorry

end percent_games_lost_l217_217632


namespace correct_simplification_l217_217668

theorem correct_simplification (x y : ℝ) (hy : y ≠ 0):
  3 * x^4 * y / (x^2 * y) = 3 * x^2 :=
by
  sorry

end correct_simplification_l217_217668


namespace geric_initial_bills_l217_217720

theorem geric_initial_bills :
  ∀ (bills_jessa : ℕ) (bills_kylan: ℕ) (bills_geric : ℕ),
  bills_jessa = 10 →
  bills_kylan = bills_jessa - 2 →
  bills_geric = 2 * bills_kylan →
  bills_geric = 16 :=
by
  intros bills_jessa bills_kylan bills_geric h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  rw [h3]
  sorry

end geric_initial_bills_l217_217720


namespace new_class_mean_score_l217_217742

theorem new_class_mean_score : 
  let s1 := 68
  let n1 := 50
  let s2 := 75
  let n2 := 8
  let s3 := 82
  let n3 := 2
  (n1 * s1 + n2 * s2 + n3 * s3) / (n1 + n2 + n3) = 69.4 := by
  sorry

end new_class_mean_score_l217_217742


namespace sum_of_fractions_l217_217703

-- Definition of the fractions given as conditions
def frac1 := 2 / 10
def frac2 := 4 / 40
def frac3 := 6 / 60
def frac4 := 8 / 30

-- Statement of the theorem to prove
theorem sum_of_fractions : frac1 + frac2 + frac3 + frac4 = 2 / 3 := by
  sorry

end sum_of_fractions_l217_217703


namespace probability_sum_less_than_product_l217_217800

def set_of_even_integers : Set ℕ := {2, 4, 6, 8, 10}

def sum_less_than_product (a b : ℕ) : Prop :=
  a + b < a * b

theorem probability_sum_less_than_product :
  let total_combinations := 25
  let valid_combinations := 16
  (valid_combinations / total_combinations : ℚ) = 16 / 25 :=
by
  sorry

end probability_sum_less_than_product_l217_217800


namespace simplify_sqrt_expression_is_correct_l217_217932

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l217_217932


namespace tetrahedron_planes_count_l217_217149

def tetrahedron_planes : ℕ :=
  let vertices := 4
  let midpoints := 6
  -- The total number of planes calculated by considering different combinations
  4      -- planes formed by three vertices
  + 6    -- planes formed by two vertices and one midpoint
  + 12   -- planes formed by one vertex and two midpoints
  + 7    -- planes formed by three midpoints

theorem tetrahedron_planes_count :
  tetrahedron_planes = 29 :=
by
  sorry

end tetrahedron_planes_count_l217_217149


namespace truck_gasoline_rate_l217_217131

theorem truck_gasoline_rate (gas_initial gas_final : ℕ) (dist_supermarket dist_farm_turn dist_farm_final : ℕ) 
    (total_miles gas_used : ℕ) : 
  gas_initial = 12 →
  gas_final = 2 →
  dist_supermarket = 10 →
  dist_farm_turn = 4 →
  dist_farm_final = 6 →
  total_miles = dist_supermarket + dist_farm_turn + dist_farm_final →
  gas_used = gas_initial - gas_final →
  total_miles / gas_used = 2 :=
by sorry

end truck_gasoline_rate_l217_217131


namespace probability_meeting_proof_l217_217773

noncomputable def probability_meeting (arrival_time_paul arrival_time_caroline : ℝ) : Prop :=
  arrival_time_paul ≤ arrival_time_caroline + 1 / 4 ∧ arrival_time_paul ≥ arrival_time_caroline - 1 / 4

theorem probability_meeting_proof :
  ∀ (arrival_time_paul arrival_time_caroline : ℝ)
    (h_paul_range : 0 ≤ arrival_time_paul ∧ arrival_time_paul ≤ 1)
    (h_caroline_range: 0 ≤ arrival_time_caroline ∧ arrival_time_caroline ≤ 1),
  (probability_meeting arrival_time_paul arrival_time_caroline) → 
  ∃ p, p = 7/16 :=
by
  sorry

end probability_meeting_proof_l217_217773


namespace math_problem_l217_217583

theorem math_problem (a b : ℝ) (h : Real.sqrt (a + 2) + |b - 1| = 0) : (a + b) ^ 2023 = -1 := 
by
  sorry

end math_problem_l217_217583


namespace length_of_MN_l217_217566

theorem length_of_MN (A B C D K L M N : Type) 
  (h1 : A → B → C → D → Prop) -- Condition for rectangle ABCD
  (h2 : K → L → Prop) -- Condition for circle intersecting AB at K and L
  (h3 : M → N → Prop) -- Condition for circle intersecting CD at M and N
  (AK KL DN : ℝ)
  (h4 : AK = 10)
  (h5 : KL = 17)
  (h6 : DN = 7) :
  ∃ MN : ℝ, MN = 23 := 
sorry

end length_of_MN_l217_217566


namespace proof_problem_l217_217295

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l217_217295


namespace simplify_sqrt_expression_l217_217927

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l217_217927


namespace herring_invariant_l217_217825

/--
A circle is divided into six sectors. Each sector contains one herring. 
In one move, you can move any two herrings in adjacent sectors moving them in opposite directions.
Prove that it is impossible to gather all herrings into one sector using these operations.
-/
theorem herring_invariant (herring : Fin 6 → Bool) :
  ¬ ∃ i : Fin 6, ∀ j : Fin 6, herring j = herring i := 
sorry

end herring_invariant_l217_217825


namespace fractional_equation_solution_l217_217794

theorem fractional_equation_solution (x : ℝ) (hx : x ≠ -3) (h : 1/x = 2/(x+3)) : x = 3 :=
sorry

end fractional_equation_solution_l217_217794


namespace intersection_sets_l217_217187

def setA : Set ℝ := { x | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB : Set ℝ := { x | (x - 3) / (2 * x) ≤ 0 }

theorem intersection_sets (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 0 < x ∧ x ≤ 1 := by
  sorry

end intersection_sets_l217_217187


namespace tessa_still_owes_greg_l217_217180

def initial_debt : ℝ := 40
def first_repayment : ℝ := 0.25 * initial_debt
def debt_after_first_repayment : ℝ := initial_debt - first_repayment
def second_borrowing : ℝ := 25
def debt_after_second_borrowing : ℝ := debt_after_first_repayment + second_borrowing
def second_repayment : ℝ := 0.5 * debt_after_second_borrowing
def debt_after_second_repayment : ℝ := debt_after_second_borrowing - second_repayment
def third_borrowing : ℝ := 30
def debt_after_third_borrowing : ℝ := debt_after_second_repayment + third_borrowing
def third_repayment : ℝ := 0.1 * debt_after_third_borrowing
def final_debt : ℝ := debt_after_third_borrowing - third_repayment

theorem tessa_still_owes_greg : final_debt = 51.75 := by
  sorry

end tessa_still_owes_greg_l217_217180


namespace problem_statement_l217_217286

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l217_217286


namespace fraction_of_surface_area_is_red_l217_217516

structure Cube :=
  (edge_length : ℕ)
  (small_cubes : ℕ)
  (num_red_cubes : ℕ)
  (num_blue_cubes : ℕ)
  (blue_cube_edge_length : ℕ)
  (red_outer_layer : ℕ)

def surface_area (c : Cube) : ℕ := 6 * (c.edge_length * c.edge_length)

theorem fraction_of_surface_area_is_red (c : Cube) 
  (h_edge_length : c.edge_length = 4)
  (h_small_cubes : c.small_cubes = 64)
  (h_num_red_cubes : c.num_red_cubes = 40)
  (h_num_blue_cubes : c.num_blue_cubes = 24)
  (h_blue_cube_edge_length : c.blue_cube_edge_length = 2)
  (h_red_outer_layer : c.red_outer_layer = 1)
  : (surface_area c) / (surface_area c) = 1 := 
by
  sorry

end fraction_of_surface_area_is_red_l217_217516


namespace figure_square_count_l217_217285

theorem figure_square_count (f : ℕ → ℕ)
  (h0 : f 0 = 2)
  (h1 : f 1 = 8)
  (h2 : f 2 = 18)
  (h3 : f 3 = 32) :
  f 100 = 20402 :=
sorry

end figure_square_count_l217_217285


namespace exists_digit_sum_divisible_by_27_not_number_l217_217330

-- Definitions
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- Theorem statement
theorem exists_digit_sum_divisible_by_27_not_number (n : ℕ) :
  divisible_by (sum_of_digits n) 27 ∧ ¬ divisible_by n 27 :=
  sorry

end exists_digit_sum_divisible_by_27_not_number_l217_217330


namespace remainder_division_lemma_l217_217718

theorem remainder_division_lemma (j : ℕ) (hj : 0 < j) (hmod : 132 % (j^2) = 12) : 250 % j = 0 :=
sorry

end remainder_division_lemma_l217_217718


namespace total_worth_is_correct_l217_217051

-- Define the conditions
def rows : ℕ := 4
def gold_bars_per_row : ℕ := 20
def worth_per_gold_bar : ℕ := 20000

-- Define the calculated values
def total_gold_bars : ℕ := rows * gold_bars_per_row
def total_worth_of_gold_bars : ℕ := total_gold_bars * worth_per_gold_bar

-- Theorem statement to prove the correct total worth
theorem total_worth_is_correct : total_worth_of_gold_bars = 1600000 := by
  sorry

end total_worth_is_correct_l217_217051


namespace max_length_cos_theta_l217_217706

def domain (x y : ℝ) : Prop := (x^2 + (y - 1)^2 ≤ 1 ∧ x ≥ (Real.sqrt 2 / 3))

theorem max_length_cos_theta :
  (∃ x y : ℝ, domain x y ∧ ∀ θ : ℝ, (0 < θ ∧ θ < (Real.pi / 2)) → θ = Real.arctan (Real.sqrt 2) → 
  (Real.cos θ = Real.sqrt 3 / 3)) := sorry

end max_length_cos_theta_l217_217706


namespace cost_of_superman_game_l217_217077

-- Define the costs as constants
def cost_batman_game : ℝ := 13.60
def total_amount_spent : ℝ := 18.66

-- Define the theorem to prove the cost of the Superman game
theorem cost_of_superman_game : total_amount_spent - cost_batman_game = 5.06 :=
by
  sorry

end cost_of_superman_game_l217_217077


namespace regular_tetrahedron_triangles_l217_217439

theorem regular_tetrahedron_triangles :
  let vertices := 4
  ∃ triangles : ℕ, (triangles = Nat.choose vertices 3) ∧ (triangles = 4) :=
by {
  let vertices := 4,
  use Nat.choose vertices 3,
  split,
  { 
    refl,
  },
  {
    norm_num,
  }
}

end regular_tetrahedron_triangles_l217_217439


namespace least_common_multiple_of_20_45_75_l217_217649

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l217_217649


namespace frisbee_sales_total_receipts_l217_217119

theorem frisbee_sales_total_receipts 
  (total_frisbees : ℕ) 
  (price_3_frisbee : ℕ) 
  (price_4_frisbee : ℕ) 
  (sold_3 : ℕ) 
  (sold_4 : ℕ) 
  (total_receipts : ℕ) 
  (h1 : total_frisbees = 60) 
  (h2 : price_3_frisbee = 3)
  (h3 : price_4_frisbee = 4) 
  (h4 : sold_3 + sold_4 = total_frisbees) 
  (h5 : sold_4 ≥ 24)
  (h6 : total_receipts = sold_3 * price_3_frisbee + sold_4 * price_4_frisbee) :
  total_receipts = 204 :=
sorry

end frisbee_sales_total_receipts_l217_217119


namespace unique_solution_exists_l217_217415

theorem unique_solution_exists (k : ℚ) (h : k ≠ 0) : 
  (∀ x : ℚ, (x + 3) / (kx - 2) = x → x = -2) ↔ k = -3 / 4 := 
by
  sorry

end unique_solution_exists_l217_217415


namespace parabola_sum_l217_217422

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

end parabola_sum_l217_217422


namespace correct_option_D_l217_217805

theorem correct_option_D (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by sorry

end correct_option_D_l217_217805


namespace solution_l217_217217

-- Define the problem.
def problem (CD : ℝ) (hexagon_side : ℝ) (CY : ℝ) (BY : ℝ) : Prop :=
  CD = 2 ∧ hexagon_side = 2 ∧ CY = 4 * CD ∧ BY = 9 * Real.sqrt 2 → BY = 9 * Real.sqrt 2

theorem solution : problem 2 2 8 (9 * Real.sqrt 2) :=
by
  -- Contextualize the given conditions and directly link to the desired proof.
  intro h
  sorry

end solution_l217_217217


namespace tetrahedron_triangle_count_l217_217441

theorem tetrahedron_triangle_count : 
  let vertices := 4 in
  let choose_three := Nat.choose vertices 3 in
  choose_three = 4 :=
by
  have vertices : Nat := 4
  have choose_three := Nat.choose vertices 3
  show choose_three = 4
  sorry

end tetrahedron_triangle_count_l217_217441


namespace zoo_recovery_time_l217_217100

theorem zoo_recovery_time (lions rhinos recover_time : ℕ) (total_animals : ℕ) (total_time : ℕ)
    (h_lions : lions = 3) (h_rhinos : rhinos = 2) (h_recover_time : recover_time = 2)
    (h_total_animals : total_animals = lions + rhinos) (h_total_time : total_time = total_animals * recover_time) :
    total_time = 10 :=
by
  rw [h_lions, h_rhinos] at h_total_animals
  rw [h_total_animals, h_recover_time] at h_total_time
  exact h_total_time

end zoo_recovery_time_l217_217100


namespace A_number_is_35_l217_217101

theorem A_number_is_35 (A B : ℕ) 
  (h_sum_digits : A + B = 8) 
  (h_diff_numbers : 10 * B + A = 10 * A + B + 18) :
  10 * A + B = 35 :=
by {
  sorry
}

end A_number_is_35_l217_217101


namespace number_of_chocolates_l217_217639

-- Define the dimensions of the box
def W_box := 30
def L_box := 20
def H_box := 5

-- Define the dimensions of one chocolate
def W_chocolate := 6
def L_chocolate := 4
def H_chocolate := 1

-- Calculate the volume of the box
def V_box := W_box * L_box * H_box

-- Calculate the volume of one chocolate
def V_chocolate := W_chocolate * L_chocolate * H_chocolate

-- Lean theorem statement for the proof problem
theorem number_of_chocolates : V_box / V_chocolate = 125 := 
by
  sorry

end number_of_chocolates_l217_217639


namespace enhanced_inequality_l217_217606

theorem enhanced_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a^2 / (b + c) + 2 * b^2 / (c + a) + 2 * c^2 / (a + b) ≥ a + b + c + (2 * a - b - c)^2 / (a + b + c)) :=
sorry

end enhanced_inequality_l217_217606


namespace students_in_A_and_D_combined_l217_217283

theorem students_in_A_and_D_combined (AB BC CD : ℕ) (hAB : AB = 83) (hBC : BC = 86) (hCD : CD = 88) : (AB + CD - BC = 85) :=
by
  sorry

end students_in_A_and_D_combined_l217_217283


namespace final_fraction_of_water_is_243_over_1024_l217_217254

theorem final_fraction_of_water_is_243_over_1024 :
  let initial_volume := 20
  let replaced_volume := 5
  let cycles := 5
  let initial_fraction_of_water := 1
  let final_fraction_of_water :=
        (initial_fraction_of_water * (initial_volume - replaced_volume) / initial_volume) ^ cycles
  final_fraction_of_water = 243 / 1024 :=
by
  sorry

end final_fraction_of_water_is_243_over_1024_l217_217254


namespace crop_planting_ways_l217_217997

-- Definitions of the sections S1, S2, S3, S4 representing the grid
inductive Section
| S1 | S2 | S3 | S4

-- Definitions of the crops
inductive Crop
| Orange | Apple | Pear | Cherry

-- Predicate to determine adjacency
def adjacent : Section → Section → Prop
| Section.S1 Section.S2 := true
| Section.S1 Section.S3 := true
| Section.S2 Section.S1 := true
| Section.S2 Section.S4 := true
| Section.S3 Section.S1 := true
| Section.S3 Section.S4 := true
| Section.S4 Section.S2 := true
| Section.S4 Section.S3 := true
| _ _ := false

-- Predicate to determine diagonal relationships
def diagonal : Section → Section → Prop
| Section.S1 Section.S4 := true
| Section.S4 Section.S1 := true
| Section.S2 Section.S3 := true
| Section.S3 Section.S2 := true
| _ _ := false

-- Main theorem statement
theorem crop_planting_ways :
  ∃ (ways : ℕ), ways = 12 ∧
  ∀ (f : Section → Crop),
    (adjacent Section.S1 Section.S2 → (f Section.S1 = Crop.Orange ∨ f Section.S1 = Crop.Pear) → (f Section.S2 ≠ Crop.Orange ∧ f Section.S2 ≠ Crop.Pear)) ∧
    (adjacent Section.S1 Section.S3 → (f Section.S1 = Crop.Orange ∨ f Section.S1 = Crop.Pear) → (f Section.S3 ≠ Crop.Orange ∧ f Section.S3 ≠ Crop.Pear)) ∧
    (adjacent Section.S2 Section.S1 → (f Section.S2 = Crop.Orange ∨ f Section.S2 = Crop.Pear) → (f Section.S1 ≠ Crop.Orange ∧ f Section.S1 ≠ Crop.Pear)) ∧
    (adjacent Section.S2 Section.S4 → (f Section.S2 = Crop.Orange ∨ f Section.S2 = Crop.Pear) → (f Section.S4 ≠ Crop.Orange ∧ f Section.S4 ≠ Crop.Pear)) ∧
    (adjacent Section.S3 Section.S1 → (f Section.S3 = Crop.Orange ∨ f Section.S3 = Crop.Pear) → (f Section.S1 ≠ Crop.Orange ∧ f Section.S1 ≠ Crop.Pear)) ∧
    (adjacent Section.S3 Section.S4 → (f Section.S3 = Crop.Orange ∨ f Section.S3 = Crop.Pear) → (f Section.S4 ≠ Crop.Orange ∧ f Section.S4 ≠ Crop.Pear)) ∧
    (adjacent Section.S4 Section.S2 → (f Section.S4 = Crop.Orange ∨ f Section.S4 = Crop.Pear) → (f Section.S2 ≠ Crop.Orange ∧ f Section.S2 ≠ Crop.Pear)) ∧
    (adjacent Section.S4 Section.S3 → (f Section.S4 = Crop.Orange ∨ f Section.S4 = Crop.Pear) → (f Section.S3 ≠ Crop.Orange ∧ f Section.S3 ≠ Crop.Pear)) ∧
    (diagonal Section.S1 Section.S4 → (f Section.S1 = Crop.Apple ∨ f Section.S1 = Crop.Cherry) → (f Section.S4 ≠ Crop.Apple ∧ f Section.S4 ≠ Crop.Cherry)) ∧
    (diagonal Section.S2 Section.S3 → (f Section.S2 = Crop.Apple ∨ f Section.S2 = Crop.Cherry) → (f Section.S3 ≠ Crop.Apple ∧ f Section.S3 ≠ Crop.Cherry)) := sorry

end crop_planting_ways_l217_217997


namespace smallest_n_candy_price_l217_217744

theorem smallest_n_candy_price :
  ∃ n : ℕ, 25 * n = Nat.lcm (Nat.lcm 20 18) 24 ∧ ∀ k : ℕ, k > 0 ∧ 25 * k = Nat.lcm (Nat.lcm 20 18) 24 → n ≤ k :=
sorry

end smallest_n_candy_price_l217_217744


namespace second_discount_percentage_l217_217374

-- Define the initial conditions.
def listed_price : ℝ := 200
def first_discount_rate : ℝ := 0.20
def final_sale_price : ℝ := 144

-- Calculate the price after the first discount.
def first_discount_amount := first_discount_rate * listed_price
def price_after_first_discount := listed_price - first_discount_amount

-- Define the second discount amount.
def second_discount_amount := price_after_first_discount - final_sale_price

-- Define the theorem to prove the second discount rate.
theorem second_discount_percentage : 
  (second_discount_amount / price_after_first_discount) * 100 = 10 :=
by 
  sorry -- Proof placeholder

end second_discount_percentage_l217_217374


namespace cost_price_of_each_clock_l217_217979

theorem cost_price_of_each_clock
  (C : ℝ)
  (h1 : 40 * C * 1.1 + 50 * C * 1.2 - 90 * C * 1.15 = 40) :
  C = 80 :=
sorry

end cost_price_of_each_clock_l217_217979


namespace updated_mean_l217_217249

theorem updated_mean (n : ℕ) (observation_mean decrement : ℕ) 
  (h1 : n = 50) (h2 : observation_mean = 200) (h3 : decrement = 15) : 
  ((observation_mean * n - decrement * n) / n = 185) :=
by
  sorry

end updated_mean_l217_217249


namespace probability_non_first_class_product_l217_217617

theorem probability_non_first_class_product (P_A P_B P_C : ℝ) (hA : P_A = 0.65) (hB : P_B = 0.2) (hC : P_C = 0.1) : 1 - P_A = 0.35 :=
by
  sorry

end probability_non_first_class_product_l217_217617


namespace unique_solution_arith_prog_system_l217_217640

theorem unique_solution_arith_prog_system (x y : ℝ) : 
  (6 * x + 9 * y = 12) ∧ (15 * x + 18 * y = 21) ↔ (x = -1) ∧ (y = 2) :=
by sorry

end unique_solution_arith_prog_system_l217_217640


namespace algebraic_fraction_l217_217092

theorem algebraic_fraction (x : ℝ) (h1 : 1 / 3 = 1 / 3) 
(h2 : x / Real.pi = x / Real.pi) 
(h3 : 2 / (x + 3) = 2 / (x + 3))
(h4 : (x + 2) / 3 = (x + 2) / 3) 
: 
2 / (x + 3) = 2 / (x + 3) := sorry

end algebraic_fraction_l217_217092


namespace percentage_increase_in_average_visibility_l217_217235

theorem percentage_increase_in_average_visibility :
  let avg_visibility_without_telescope := (100 + 110) / 2
  let avg_visibility_with_telescope := (150 + 165) / 2
  let increase_in_avg_visibility := avg_visibility_with_telescope - avg_visibility_without_telescope
  let percentage_increase := (increase_in_avg_visibility / avg_visibility_without_telescope) * 100
  percentage_increase = 50 := by
  -- calculations are omitted; proof goes here
  sorry

end percentage_increase_in_average_visibility_l217_217235


namespace batsman_average_after_12th_inning_l217_217988

theorem batsman_average_after_12th_inning (average_initial : ℕ) (score_12th : ℕ) (average_increase : ℕ) (total_innings : ℕ) 
    (h_avg_init : average_initial = 29) (h_score_12th : score_12th = 65) (h_avg_inc : average_increase = 3) 
    (h_total_innings : total_innings = 12) : 
    (average_initial + average_increase = 32) := 
by
  sorry

end batsman_average_after_12th_inning_l217_217988


namespace watch_cost_l217_217699

variables (w s : ℝ)

theorem watch_cost (h1 : w + s = 120) (h2 : w = 100 + s) : w = 110 :=
by
  sorry

end watch_cost_l217_217699


namespace n_four_plus_n_squared_plus_one_not_prime_l217_217471

theorem n_four_plus_n_squared_plus_one_not_prime (n : ℤ) (h : n ≥ 2) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end n_four_plus_n_squared_plus_one_not_prime_l217_217471


namespace number_of_truthful_dwarfs_l217_217551

/-- 
Each of the 10 dwarfs either always tells the truth or always lies. 
Each dwarf likes exactly one type of ice cream: vanilla, chocolate, or fruit.
When asked, every dwarf raised their hand for liking vanilla ice cream.
When asked, 5 dwarfs raised their hand for liking chocolate ice cream.
When asked, only 1 dwarf raised their hand for liking fruit ice cream.
Prove that the number of truthful dwarfs is 4.
-/
theorem number_of_truthful_dwarfs (T L : ℕ) 
  (h1 : T + L = 10) 
  (h2 : T + 2 * L = 16) : 
  T = 4 := 
by
  -- Proof omitted
  sorry

end number_of_truthful_dwarfs_l217_217551


namespace split_cube_l217_217159

theorem split_cube (m : ℕ) (hm : m > 1) (h : ∃ k, ∃ l, l > 0 ∧ (3 + 2 * (k - 1)) = 59 ∧ (k + l = (m * (m - 1)) / 2)) : m = 8 :=
sorry

end split_cube_l217_217159


namespace largest_divisor_of_expression_l217_217444

theorem largest_divisor_of_expression (x : ℤ) (hx : x % 2 = 1) :
  864 ∣ (12 * x + 2) * (12 * x + 6) * (12 * x + 10) * (6 * x + 3) :=
sorry

end largest_divisor_of_expression_l217_217444


namespace incorrect_conclusion_l217_217388

noncomputable def data_set : List ℕ := [4, 1, 6, 2, 9, 5, 8]
def mean_x : ℝ := 2
def mean_y : ℝ := 20
def regression_eq (x : ℝ) : ℝ := 9.1 * x + 1.8
def chi_squared_value : ℝ := 9.632
def alpha : ℝ := 0.001
def critical_value : ℝ := 10.828

theorem incorrect_conclusion : ¬(chi_squared_value ≥ critical_value) := by
  -- Insert proof here
  sorry

end incorrect_conclusion_l217_217388


namespace recurring_sum_l217_217851

noncomputable def recurring_to_fraction (a b : ℕ) : ℚ := a / (10 ^ b - 1)

def r1 := recurring_to_fraction 12 2
def r2 := recurring_to_fraction 34 3
def r3 := recurring_to_fraction 567 5

theorem recurring_sum : r1 + r2 + r3 = 16133 / 99999 := by
  sorry

end recurring_sum_l217_217851


namespace even_four_digit_strict_inc_count_l217_217581

theorem even_four_digit_strict_inc_count :
  let numbers := { (a, b, c, d) | 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ (d = 4 ∨ d = 6 ∨ d = 8) ∧ d % 2 = 0 } in
  ∑ d in {4, 6, 8}, (if d = 4 then 1 else if d = 6 then (Nat.choose 5 3) else (Nat.choose 7 3)) = 46 :=
by
  sorry

end even_four_digit_strict_inc_count_l217_217581


namespace find_second_number_l217_217494

theorem find_second_number (x y z : ℚ) (h_sum : x + y + z = 120)
  (h_ratio1 : x = (3 / 4) * y) (h_ratio2 : z = (7 / 4) * y) :
  y = 240 / 7 :=
by {
  -- Definitions provided from conditions
  sorry  -- Proof omitted
}

end find_second_number_l217_217494


namespace danica_planes_l217_217708

def smallestAdditionalPlanes (n k : ℕ) : ℕ :=
  let m := k * (n / k + 1)
  m - n

theorem danica_planes : smallestAdditionalPlanes 17 7 = 4 :=
by
  -- Proof would go here
  sorry

end danica_planes_l217_217708


namespace exists_group_of_four_l217_217592

-- Define the given conditions
variables (students : Finset ℕ) (h_size : students.card = 21)
variables (done_homework : Finset ℕ → Prop)
variables (hw_unique : ∀ (s : Finset ℕ), s.card = 3 → done_homework s)

-- Define the theorem with the assertion to be proved
theorem exists_group_of_four (students : Finset ℕ) (h_size : students.card = 21)
  (done_homework : Finset ℕ → Prop)
  (hw_unique : ∀ s, s.card = 3 → done_homework s) :
  ∃ (grp : Finset ℕ), grp.card = 4 ∧ 
    (∀ (s : Finset ℕ), s ⊆ grp ∧ s.card = 3 → done_homework s) :=
sorry

end exists_group_of_four_l217_217592


namespace part1_solve_inequality_part2_range_of_a_l217_217608

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1_solve_inequality (x : ℝ) (h : -2 < x ∧ x < -2/3) :
    f x 1 > 1 :=
by
  sorry

theorem part2_range_of_a (h : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x (a : ℝ) > 0) :
    -5/2 < a ∧ a < -2 :=
by
  sorry

end part1_solve_inequality_part2_range_of_a_l217_217608


namespace sum_of_possible_ks_l217_217028

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l217_217028


namespace combined_salaries_l217_217954

variable {A B C E : ℝ}
variable (D : ℝ := 7000)
variable (average_salary : ℝ := 8400)
variable (n : ℕ := 5)

theorem combined_salaries (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ E) 
  (h4 : B ≠ C) (h5 : B ≠ E) (h6 : C ≠ E)
  (h7 : average_salary = (A + B + C + D + E) / n) :
  A + B + C + E = 35000 :=
by
  sorry

end combined_salaries_l217_217954


namespace modular_units_l217_217263

theorem modular_units (U N S : ℕ) 
  (h1 : N = S / 4)
  (h2 : (S : ℚ) / (S + U * N) = 0.14285714285714285) : 
  U = 24 :=
by
  sorry

end modular_units_l217_217263


namespace solve_fractional_equation_l217_217792

theorem solve_fractional_equation : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -3 → (1 / x = 2 / (x + 3) ↔ x = 3) :=
by
  intros x hx
  sorry

end solve_fractional_equation_l217_217792


namespace right_triangle_perimeter_l217_217116

theorem right_triangle_perimeter 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_area : 1/2 * 30 * b = 180)
  (h_pythagorean : c^2 = 30^2 + b^2)
  : a + b + c = 42 + 2 * Real.sqrt 261 :=
sorry

end right_triangle_perimeter_l217_217116


namespace solution_set_of_inequality_l217_217574

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2
  else if x > 0 then x - 2
  else 0

theorem solution_set_of_inequality :
  {x : ℝ | 2 * f x - 1 < 0} = {x | x < -3 / 2 ∨ (0 ≤ x ∧ x < 5 / 2)} :=
by
  sorry

end solution_set_of_inequality_l217_217574


namespace rhombus_side_length_l217_217946

theorem rhombus_side_length (d1 d2 : ℕ) (h1 : d1 = 24) (h2 : d2 = 70) : 
  ∃ (a : ℕ), a^2 = (d1 / 2)^2 + (d2 / 2)^2 ∧ a = 37 :=
by
  sorry

end rhombus_side_length_l217_217946


namespace min_value_expr_l217_217424

theorem min_value_expr : ∀ (x : ℝ), 0 < x ∧ x < 4 → ∃ y : ℝ, y = (1 / (4 - x) + 2 / x) ∧ y = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end min_value_expr_l217_217424


namespace hoseok_multiplied_number_l217_217580

theorem hoseok_multiplied_number (n : ℕ) (h : 11 * n = 99) : n = 9 := 
sorry

end hoseok_multiplied_number_l217_217580


namespace no_positive_integer_exists_l217_217643

theorem no_positive_integer_exists
  (P1 P2 : ℤ → ℤ)
  (a : ℤ)
  (h_a_neg : a < 0)
  (h_common_root : P1 a = 0 ∧ P2 a = 0) :
  ¬ ∃ b : ℤ, b > 0 ∧ P1 b = 2007 ∧ P2 b = 2008 :=
sorry

end no_positive_integer_exists_l217_217643


namespace lcm_20_45_75_l217_217666

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l217_217666


namespace initial_beavers_l217_217271

theorem initial_beavers (B C : ℕ) (h1 : C = 40) (h2 : B + C + 2 * B + (C - 10) = 130) : B = 20 :=
by
  sorry

end initial_beavers_l217_217271


namespace even_function_analytic_expression_l217_217314

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then Real.log (x^2 - 2 * x + 2) 
else Real.log (x^2 + 2 * x + 2)

theorem even_function_analytic_expression (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_nonneg : ∀ x : ℝ, 0 ≤ x → f x = Real.log (x^2 - 2 * x + 2)) :
  ∀ x : ℝ, x < 0 → f x = Real.log (x^2 + 2 * x + 2) :=
by
  sorry

end even_function_analytic_expression_l217_217314


namespace problem_statement_l217_217287

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l217_217287


namespace bags_of_white_flour_l217_217095

theorem bags_of_white_flour (total_flour wheat_flour : ℝ) (h1 : total_flour = 0.3) (h2 : wheat_flour = 0.2) : 
  total_flour - wheat_flour = 0.1 :=
by
  sorry

end bags_of_white_flour_l217_217095


namespace g_function_expression_l217_217722

theorem g_function_expression (f g : ℝ → ℝ) (a : ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, g (-x) = g x) (h3 : ∀ x : ℝ, f x + g x = x^2 + a * x + 2 * a - 1) (h4 : f 1 = 2) :
  ∀ t : ℝ, g t = t^2 + 4 * t - 1 :=
by
  sorry

end g_function_expression_l217_217722


namespace tip_percentage_l217_217996

theorem tip_percentage
  (total_amount_paid : ℝ)
  (price_of_food : ℝ)
  (sales_tax_rate : ℝ)
  (total_amount : ℝ)
  (tip_percentage : ℝ)
  (h1 : total_amount_paid = 184.80)
  (h2 : price_of_food = 140)
  (h3 : sales_tax_rate = 0.10)
  (h4 : total_amount = price_of_food + (price_of_food * sales_tax_rate))
  (h5 : tip_percentage = ((total_amount_paid - total_amount) / total_amount) * 100) :
  tip_percentage = 20 := sorry

end tip_percentage_l217_217996


namespace ones_digit_of_9_pow_47_l217_217971

theorem ones_digit_of_9_pow_47 : (9 ^ 47) % 10 = 9 := 
by
  sorry

end ones_digit_of_9_pow_47_l217_217971


namespace max_snowmen_l217_217688

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l217_217688


namespace max_value_char_l217_217578

theorem max_value_char (m x a b : ℕ) (h_sum : 28 * m + x + a + 2 * b = 368)
  (h1 : x ≤ 23) (h2 : x > a) (h3 : a > b) (h4 : b ≥ 0) :
  m + x ≤ 35 := 
sorry

end max_value_char_l217_217578


namespace true_compound_propositions_l217_217289

-- Define the propositions
def p1 : Prop := 
  ∀ (l₁ l₂ l₃ : Line), 
    (l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
    (∃ (p₁ p₂ p₃ : Point),
      l₁.contains p₁ ∧ l₁.contains p₂ ∧
      l₂.contains p₂ ∧ l₂.contains p₃ ∧
      l₃.contains p₃ ∧ l₃.contains p₁)
    → ∃ (α : Plane), 
      α.contains l₁ ∧ α.contains l₂ ∧ α.contains l₃

def p2 : Prop :=
  ∀ (a b c : Point),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    → ∃! (α : Plane), 
      α.contains a ∧ α.contains b ∧ α.contains c

def p3 : Prop :=
  ∀ (l₁ l₂ : Line),
    (¬ ∃ (p : Point), l₁.contains p ∧ l₂.contains p)
    → l₁.parallel l₂

def p4 : Prop :=
  ∀ (α : Plane) (l m : Line),
    (α.contains l ∧ ¬ α.contains m ∧ m.perpendicular α) 
    → m.perpendicular l

-- Main theorem that identifies the true compound propositions
theorem true_compound_propositions :
  (p1 ∧ p4) ∧ ¬ (p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) :=
by
  -- Proof might involve defining lines, points, and planes and their relationships
  sorry

end true_compound_propositions_l217_217289


namespace total_number_of_coins_l217_217991

theorem total_number_of_coins (x : ℕ) :
  5 * x + 10 * x + 25 * x = 120 → 3 * x = 9 :=
by
  intro h
  sorry

end total_number_of_coins_l217_217991


namespace range_of_a_l217_217951

theorem range_of_a (a : ℝ) : 
  ((-1 + a) ^ 2 + (-1 - a) ^ 2 < 4) ↔ (-1 < a ∧ a < 1) := 
by
  sorry

end range_of_a_l217_217951


namespace pentagon_area_ratio_l217_217200

theorem pentagon_area_ratio (ABCDE PQRST : Set Point)
  (is_regular_pentagon_ABCDE : ∀ (A B C D E : Point),
    ABCDE = {A, B, C, D, E} ∧
    (segment A B).is_regular_pentagon ABCDE)
  (is_equilateral_outside_triangle : ∀ (AB BC CD DE EA : Segment) (A B C D E P Q R S T : Point),
    ABCDE = {A, B, C, D, E} ∧
    (P, Q, R, S, T are_centers_of_equilateral_triangles ABCDE))
  : 1 = 1 :=  -- Formula for the area ratio, which is based on simplified calculations
sorry

end pentagon_area_ratio_l217_217200


namespace sequence_arithmetic_l217_217871

theorem sequence_arithmetic (a : ℕ → Real)
    (h₁ : a 3 = 2)
    (h₂ : a 7 = 1)
    (h₃ : ∃ d, ∀ n, 1 / (1 + a (n + 1)) = 1 / (1 + a n) + d):
    a 11 = 1 / 2 := by
  sorry

end sequence_arithmetic_l217_217871


namespace andrew_correct_answer_l217_217270

variable {x : ℕ}

theorem andrew_correct_answer (h : (x - 8) / 7 = 15) : (x - 5) / 11 = 10 :=
by
  sorry

end andrew_correct_answer_l217_217270


namespace y_squared_range_l217_217189

theorem y_squared_range (y : ℝ) 
  (h : Real.sqrt (Real.sqrt (y + 16)) - Real.sqrt (Real.sqrt (y - 16)) = 2) : 
  9200 ≤ y^2 ∧ y^2 ≤ 9400 := 
sorry

end y_squared_range_l217_217189


namespace find_C_coordinates_l217_217696

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -3, y := 5 }
def B : Point := { x := 9, y := -1 }
def C : Point := { x := 15, y := -4 }

noncomputable def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

theorem find_C_coordinates :
  let AB := vector A B
  let BC := { x := AB.x / 2, y := AB.y / 2 }
  let C_actual := { x := B.x + BC.x, y := B.y + BC.y }
  C = C_actual :=
by
  let AB := vector A B
  let BC := { x := AB.x / 2, y := AB.y / 2 }
  let C_actual := { x := B.x + BC.x, y := B.y + BC.y }
  show C = C_actual
  rfl

end find_C_coordinates_l217_217696


namespace minimum_value_ge_100_minimum_value_eq_100_l217_217041

noncomputable def minimum_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 10)^2

theorem minimum_value_ge_100 (α β : ℝ) : minimum_value_expression α β ≥ 100 :=
  sorry

theorem minimum_value_eq_100 (α β : ℝ)
  (hα : 3 * Real.cos α + 4 * Real.sin β = 7)
  (hβ : 3 * Real.sin α + 4 * Real.cos β = 10) :
  minimum_value_expression α β = 100 :=
  sorry

end minimum_value_ge_100_minimum_value_eq_100_l217_217041


namespace moe_share_of_pie_l217_217536

-- Definitions based on conditions
def leftover_pie : ℚ := 8 / 9
def num_people : ℚ := 3

-- Theorem to prove the amount of pie Moe took home
theorem moe_share_of_pie : (leftover_pie / num_people) = 8 / 27 := by
  sorry

end moe_share_of_pie_l217_217536


namespace vacation_cost_split_l217_217349

theorem vacation_cost_split 
  (airbnb_cost : ℕ)
  (car_rental_cost : ℕ)
  (people : ℕ)
  (split_equally : Prop)
  (h1 : airbnb_cost = 3200)
  (h2 : car_rental_cost = 800)
  (h3 : people = 8)
  (h4 : split_equally)
  : (airbnb_cost + car_rental_cost) / people = 500 :=
by
  sorry

end vacation_cost_split_l217_217349


namespace system_of_equations_is_B_l217_217597

-- Define the given conditions and correct answer
def condition1 (x y : ℝ) : Prop := 5 * x + y = 3
def condition2 (x y : ℝ) : Prop := x + 5 * y = 2
def correctAnswer (x y : ℝ) : Prop := 5 * x + y = 3 ∧ x + 5 * y = 2

theorem system_of_equations_is_B (x y : ℝ) : condition1 x y ∧ condition2 x y ↔ correctAnswer x y := by
  -- Proof goes here
  sorry

end system_of_equations_is_B_l217_217597


namespace three_digit_integer_one_more_than_LCM_l217_217804

theorem three_digit_integer_one_more_than_LCM:
  ∃ (n : ℕ), (n > 99 ∧ n < 1000) ∧ (∃ (k : ℕ), n = k + 1 ∧ (∃ m, k = 3 * 4 * 5 * 7 * 2^m)) :=
  sorry

end three_digit_integer_one_more_than_LCM_l217_217804


namespace modulo_11_residue_l217_217132

theorem modulo_11_residue : 
  (341 + 6 * 50 + 4 * 156 + 3 * 12^2) % 11 = 4 := 
by
  sorry

end modulo_11_residue_l217_217132


namespace fred_bought_books_l217_217423

theorem fred_bought_books (initial_money : ℕ) (remaining_money : ℕ) (book_cost : ℕ)
  (h1 : initial_money = 236)
  (h2 : remaining_money = 14)
  (h3 : book_cost = 37) :
  (initial_money - remaining_money) / book_cost = 6 :=
by {
  sorry
}

end fred_bought_books_l217_217423


namespace calculate_sum_l217_217634

theorem calculate_sum (P r : ℝ) (h1 : 2 * P * r = 10200) (h2 : P * ((1 + r) ^ 2 - 1) = 11730) : P = 17000 :=
sorry

end calculate_sum_l217_217634


namespace geometric_sequence_third_term_l217_217104

theorem geometric_sequence_third_term (r : ℕ) (h_r : 5 * r ^ 4 = 1620) : 5 * r ^ 2 = 180 := by
  sorry

end geometric_sequence_third_term_l217_217104


namespace least_common_multiple_of_20_45_75_l217_217651

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l217_217651


namespace part1_part2_l217_217174

theorem part1 (x : ℝ) : |x + 3| - 2 * x - 1 < 0 → 2 < x :=
by sorry

theorem part2 (m : ℝ) : (m > 0) →
  (∃ x : ℝ, |x - m| + |x + 1/m| = 2) → m = 1 :=
by sorry

end part1_part2_l217_217174


namespace joe_money_left_l217_217465

theorem joe_money_left
  (joe_savings : ℕ := 6000)
  (flight_cost : ℕ := 1200)
  (hotel_cost : ℕ := 800)
  (food_cost : ℕ := 3000) :
  joe_savings - (flight_cost + hotel_cost + food_cost) = 1000 :=
by
  sorry

end joe_money_left_l217_217465


namespace desired_depth_l217_217511

-- Define the given conditions
def men_hours_30m (d : ℕ) : ℕ := 18 * 8 * d
def men_hours_Dm (d1 : ℕ) (D : ℕ) : ℕ := 40 * 6 * d1

-- Define the proportion
def proportion (d d1 : ℕ) (D : ℕ) : Prop :=
  (men_hours_30m d) / 30 = (men_hours_Dm d1 D) / D

-- The main theorem to prove the desired depth
theorem desired_depth (d d1 : ℕ) (H : proportion d d1 50) : 50 = 50 :=
by sorry

end desired_depth_l217_217511


namespace weight_of_8_moles_of_AlI3_l217_217086

noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_I : ℝ := 126.90
noncomputable def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I

theorem weight_of_8_moles_of_AlI3 : 
  (8 * molecular_weight_AlI3) = 3261.44 := by
sorry

end weight_of_8_moles_of_AlI3_l217_217086


namespace diane_harvest_increase_l217_217710

-- Define the conditions
def last_year_harvest : ℕ := 2479
def this_year_harvest : ℕ := 8564

-- Definition of the increase in honey harvest
def increase_in_harvest : ℕ := this_year_harvest - last_year_harvest

-- The theorem statement we need to prove
theorem diane_harvest_increase : increase_in_harvest = 6085 := 
by
  -- skip the proof for now
  sorry

end diane_harvest_increase_l217_217710


namespace base_eight_to_base_ten_l217_217239

theorem base_eight_to_base_ten (n : ℕ) (h : n = 4 * 8^1 + 7 * 8^0) : n = 39 := by
  sorry

end base_eight_to_base_ten_l217_217239


namespace number_of_truthful_dwarfs_is_4_l217_217554

def dwarf := {x : ℕ // 1 ≤ x ≤ 10}
def likes_vanilla (d : dwarf) : Prop := sorry
def likes_chocolate (d : dwarf) : Prop := sorry
def likes_fruit (d : dwarf) : Prop := sorry
def tells_truth (d : dwarf) : Prop := sorry
def tells_lie (d : dwarf) : Prop := sorry

noncomputable def number_of_truthful_dwarfs : ℕ :=
  let total_dwarfs := 10 in
  let vanilla_raises := 10 in
  let chocolate_raises := 5 in
  let fruit_raises := 1 in
  -- T + L = total_dwarfs
  -- T + 2L = vanilla_raises + chocolate_raises + fruit_raises
  let T := total_dwarfs - 2 * (vanilla_raises + chocolate_raises + fruit_raises - total_dwarfs) in
  T

theorem number_of_truthful_dwarfs_is_4 : number_of_truthful_dwarfs = 4 := 
  by
    sorry

end number_of_truthful_dwarfs_is_4_l217_217554


namespace find_n_and_d_l217_217470

theorem find_n_and_d (n d : ℕ) (hn_pos : 0 < n) (hd_digit : d < 10)
    (h1 : 3 * n^2 + 2 * n + d = 263)
    (h2 : 3 * n^2 + 2 * n + 4 = 1 * 8^3 + 1 * 8^2 + d * 8 + 1) :
    n + d = 12 := 
sorry

end find_n_and_d_l217_217470


namespace metric_regression_equation_l217_217378

noncomputable def predicted_weight_imperial (height : ℝ) : ℝ :=
  4 * height - 130

def inch_to_cm (inch : ℝ) : ℝ := 2.54 * inch
def pound_to_kg (pound : ℝ) : ℝ := 0.45 * pound

theorem metric_regression_equation (height_cm : ℝ) :
  (0.72 * height_cm - 58.5) = 
  (pound_to_kg (predicted_weight_imperial (height_cm / 2.54))) :=
by
  sorry

end metric_regression_equation_l217_217378


namespace positive_slope_of_asymptote_l217_217369

-- Define the conditions
def is_hyperbola (x y : ℝ) : Prop :=
  abs (Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 5) ^ 2 + (y + 2) ^ 2)) = 3

-- Prove the positive slope of the asymptote of the given hyperbola
theorem positive_slope_of_asymptote :
  (∀ x y : ℝ, is_hyperbola x y) → abs (Real.sqrt 7 / 3) = Real.sqrt 7 / 3 :=
by
  intros h
  -- Proof to be provided (proof steps from the provided solution would be used here usually)
  sorry

end positive_slope_of_asymptote_l217_217369


namespace geric_initial_bills_l217_217719

theorem geric_initial_bills (G K J : ℕ) 
  (h1: G = 2 * K)
  (h2: K = J - 2)
  (h3: J - 3 = 7) : G = 16 := 
  by 
  sorry

end geric_initial_bills_l217_217719


namespace largest_n_l217_217909

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

axiom a1_gt_zero : a 1 > 0
axiom a2011_a2012_sum_gt_zero : a 2011 + a 2012 > 0
axiom a2011_a2012_prod_lt_zero : a 2011 * a 2012 < 0

-- Sum of first n terms of an arithmetic sequence
def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Problem statement to prove
theorem largest_n (H : is_arithmetic_sequence a) :
  ∀ n, (sequence_sum a 4022 > 0) ∧ (sequence_sum a 4023 < 0) → n = 4022 := by
  sorry

end largest_n_l217_217909


namespace diagonal_less_than_half_perimeter_l217_217983

theorem diagonal_less_than_half_perimeter (a b c d x : ℝ) 
  (h1 : x < a + b) (h2 : x < c + d) : x < (a + b + c + d) / 2 := 
by
  sorry

end diagonal_less_than_half_perimeter_l217_217983


namespace intersection_points_l217_217499

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 15
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 10

noncomputable def x1 : ℝ := (3 + Real.sqrt 209) / 4
noncomputable def x2 : ℝ := (3 - Real.sqrt 209) / 4

noncomputable def y1 : ℝ := parabola1 x1
noncomputable def y2 : ℝ := parabola1 x2

theorem intersection_points :
  (parabola1 x1 = parabola2 x1) ∧ (parabola1 x2 = parabola2 x2) :=
by
  sorry

end intersection_points_l217_217499


namespace find_n_l217_217343

theorem find_n (n : ℕ) (b : Fin (n + 1) → ℝ) (h0 : b 0 = 45) (h1 : b 1 = 81) (hn : b n = 0) (rec : ∀ (k : ℕ), 1 ≤ k → k < n → b (k+1) = b (k-1) - 5 / b k) : 
  n = 730 :=
sorry

end find_n_l217_217343


namespace age_of_oldest_child_l217_217063

theorem age_of_oldest_child (a1 a2 a3 x : ℕ) (h1 : a1 = 5) (h2 : a2 = 7) (h3 : a3 = 10) (h_avg : (a1 + a2 + a3 + x) / 4 = 8) : x = 10 :=
by
  sorry

end age_of_oldest_child_l217_217063


namespace negation_of_proposition_l217_217782

theorem negation_of_proposition :
  (¬∃ x₀ ∈ Set.Ioo 0 (π/2), Real.cos x₀ > Real.sin x₀) ↔ ∀ x ∈ Set.Ioo 0 (π / 2), Real.cos x ≤ Real.sin x :=
by
  sorry

end negation_of_proposition_l217_217782


namespace trig_identity_l217_217312

theorem trig_identity (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = 1 / 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 := 
sorry

end trig_identity_l217_217312


namespace ganesh_average_speed_l217_217982

variable (D : ℝ) -- distance between the two towns in kilometers
variable (V : ℝ) -- average speed from x to y in km/hr

-- Conditions
variable (h1 : V > 0) -- Speed must be positive
variable (h2 : 30 > 0) -- Speed must be positive
variable (h3 : 40 = (2 * D) / ((D / V) + (D / 30))) -- Average speed formula

theorem ganesh_average_speed : V = 60 :=
by {
  sorry
}

end ganesh_average_speed_l217_217982


namespace ineq_condition_l217_217819

theorem ineq_condition (a b : ℝ) : (a + 1 > b - 2) ↔ (a > b - 3 ∧ ¬(a > b)) :=
by
  sorry

end ineq_condition_l217_217819


namespace complex_pure_imaginary_l217_217014

theorem complex_pure_imaginary (a : ℝ) 
  (h1 : a^2 + 2*a - 3 = 0) 
  (h2 : a + 3 ≠ 0) : 
  a = 1 := 
by
  sorry

end complex_pure_imaginary_l217_217014


namespace correct_relationship_5_25_l217_217669

theorem correct_relationship_5_25 : 5^2 = 25 :=
by
  sorry

end correct_relationship_5_25_l217_217669


namespace average_speed_round_trip_l217_217223

noncomputable def distance_AB : ℝ := 120
noncomputable def speed_AB : ℝ := 30
noncomputable def speed_BA : ℝ := 40

theorem average_speed_round_trip :
  (2 * distance_AB * speed_AB * speed_BA) / (distance_AB * (speed_AB + speed_BA)) = 34 := 
  by 
    sorry

end average_speed_round_trip_l217_217223


namespace no_such_function_exists_l217_217152

theorem no_such_function_exists (f : ℝ → ℝ) (Hf : ∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) : False :=
by
  sorry

end no_such_function_exists_l217_217152


namespace temperature_difference_l217_217771

def highest_temperature : ℝ := 8
def lowest_temperature : ℝ := -1

theorem temperature_difference : highest_temperature - lowest_temperature = 9 := by
  sorry

end temperature_difference_l217_217771


namespace cos_17pi_over_4_eq_sqrt2_over_2_l217_217155

theorem cos_17pi_over_4_eq_sqrt2_over_2 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_17pi_over_4_eq_sqrt2_over_2_l217_217155


namespace Andy_collects_16_balls_l217_217406

-- Define the number of balls collected by Andy, Roger, and Maria.
variables (x : ℝ) (r : ℝ) (m : ℝ)

-- Define the conditions
def Andy_twice_as_many_as_Roger : Prop := r = x / 2
def Andy_five_more_than_Maria : Prop := m = x - 5
def Total_balls : Prop := x + r + m = 35

-- Define the main theorem to prove Andy's number of balls
theorem Andy_collects_16_balls (h1 : Andy_twice_as_many_as_Roger x r) 
                               (h2 : Andy_five_more_than_Maria x m) 
                               (h3 : Total_balls x r m) : 
                               x = 16 := 
by 
  sorry

end Andy_collects_16_balls_l217_217406


namespace bird_watcher_total_l217_217514

theorem bird_watcher_total
  (M : ℕ) (T : ℕ) (W : ℕ)
  (h1 : M = 70)
  (h2 : T = M / 2)
  (h3 : W = T + 8) :
  M + T + W = 148 :=
by
  -- proof omitted
  sorry

end bird_watcher_total_l217_217514


namespace area_enclosed_y_eq_kx_y_eq_xsq_l217_217723
-- Additional imports as required by the Lean system to work seamlessly

noncomputable def k_slope : ℝ :=
  deriv (λ x, real.exp (2 * x)) 0

theorem area_enclosed_y_eq_kx_y_eq_xsq :
  (k_slope = 2) → ∫ x in 0..2, (2 * x - x ^ 2) = 4 / 3 :=
begin
  -- Proof not included, only statement as per the requirements
  sorry
end

end area_enclosed_y_eq_kx_y_eq_xsq_l217_217723


namespace mike_spent_on_speakers_l217_217610

-- Definitions of the conditions:
def total_car_parts_cost : ℝ := 224.87
def new_tires_cost : ℝ := 106.33

-- Statement of the proof problem:
theorem mike_spent_on_speakers : total_car_parts_cost - new_tires_cost = 118.54 :=
by
  sorry

end mike_spent_on_speakers_l217_217610


namespace students_gold_award_freshmen_l217_217299

theorem students_gold_award_freshmen 
    (total_students total_award_winners : ℕ)
    (students_selected exchange_meeting : ℕ)
    (freshmen_selected gold_award_selected : ℕ)
    (prop1 : total_award_winners = 120)
    (prop2 : exchange_meeting = 24)
    (prop3 : freshmen_selected = 6)
    (prop4 : gold_award_selected = 4) :
    ∃ (gold_award_students : ℕ), gold_award_students = 4 ∧ gold_award_students ≤ freshmen_selected :=
by
  sorry

end students_gold_award_freshmen_l217_217299


namespace master_craftsman_parts_l217_217896

/-- 
Given:
  (1) the master craftsman produces 35 parts in the first hour,
  (2) at the rate of 35 parts/hr, he would be one hour late to meet the quota,
  (3) by increasing his speed by 15 parts/hr, he finishes the quota 0.5 hours early,
Prove that the total number of parts manufactured during the shift is 210.
-/
theorem master_craftsman_parts (N : ℕ) (quota : ℕ) 
  (initial_rate : ℕ := 35)
  (increased_rate_diff : ℕ := 15)
  (extra_time_slow : ℕ := 1)
  (time_saved_fast : ℕ := 1/2) :
  (quota = initial_rate * (extra_time_slow + 1) + N ∧
   increased_rate_diff = 15 ∧
   increased_rate_diff = λ (x : ℕ), initial_rate + x ∧
   time_saved_fast = 1/2 ∧
   N = 35) →
  quota = 210 := 
by
  sorry

end master_craftsman_parts_l217_217896


namespace time_saved_l217_217111

theorem time_saved (speed_with_tide distance1 time1 distance2 time2: ℝ) 
  (h1: speed_with_tide = 5) 
  (h2: distance1 = 5) 
  (h3: time1 = 1) 
  (h4: distance2 = 40) 
  (h5: time2 = 10) : 
  time2 - (distance2 / speed_with_tide) = 2 := 
sorry

end time_saved_l217_217111


namespace find_m_l217_217437

-- Define the hyperbola equation
def hyperbola1 (x y : ℝ) (m : ℝ) : Prop := (x^3 / m) - (y^2 / 3) = 1
def hyperbola2 (x y : ℝ) : Prop := (x^3 / 8) - (y^2 / 4) = 1

-- Define the condition for eccentricity equivalence
def same_eccentricity (m : ℝ) : Prop :=
  let e1_sq := 1 + (4 / 2^2)
  let e2_sq := 1 + (3 / m)
  e1_sq = e2_sq

-- The main theorem statement
theorem find_m (m : ℝ) : hyperbola1 x y m → hyperbola2 x y → same_eccentricity m → m = 6 :=
by
  -- Proof can be skipped with sorry to satisfy the statement-only requirement
  sorry

end find_m_l217_217437


namespace Mike_changed_2_sets_of_tires_l217_217339

theorem Mike_changed_2_sets_of_tires
  (wash_time_per_car : ℕ := 10)
  (oil_change_time_per_car : ℕ := 15)
  (tire_change_time_per_set : ℕ := 30)
  (num_washed_cars : ℕ := 9)
  (num_oil_changes : ℕ := 6)
  (total_work_time_minutes : ℕ := 4 * 60) :
  ((total_work_time_minutes - (num_washed_cars * wash_time_per_car + num_oil_changes * oil_change_time_per_car)) / tire_change_time_per_set) = 2 :=
by
  sorry

end Mike_changed_2_sets_of_tires_l217_217339


namespace max_stamps_l217_217879

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h1 : price_per_stamp = 45) (h2 : total_money = 5000) : ∃ n : ℕ, n = 111 ∧ 45 * n ≤ 5000 ∧ ∀ m : ℕ, (45 * m ≤ 5000) → m ≤ n := 
by
  sorry

end max_stamps_l217_217879


namespace exists_group_of_four_l217_217593

-- Define the given conditions
variables (students : Finset ℕ) (h_size : students.card = 21)
variables (done_homework : Finset ℕ → Prop)
variables (hw_unique : ∀ (s : Finset ℕ), s.card = 3 → done_homework s)

-- Define the theorem with the assertion to be proved
theorem exists_group_of_four (students : Finset ℕ) (h_size : students.card = 21)
  (done_homework : Finset ℕ → Prop)
  (hw_unique : ∀ s, s.card = 3 → done_homework s) :
  ∃ (grp : Finset ℕ), grp.card = 4 ∧ 
    (∀ (s : Finset ℕ), s ⊆ grp ∧ s.card = 3 → done_homework s) :=
sorry

end exists_group_of_four_l217_217593


namespace fish_added_l217_217732

theorem fish_added (x : ℕ) (hx : x + (x - 4) = 20) : x - 4 = 8 := by
  sorry

end fish_added_l217_217732


namespace largest_divisor_of_consecutive_even_product_l217_217646

theorem largest_divisor_of_consecutive_even_product (n : ℕ) (h : n % 2 = 1) :
  ∃ d, (∀ n, n % 2 = 1 → d ∣ (n+2) * (n+4) * (n+6) * (n+8) * (n+10)) ∧ d = 8 :=
begin
  existsi 8,
  split,
  { intros n hn,
    repeat { sorry },
  },
  { refl }
end

end largest_divisor_of_consecutive_even_product_l217_217646


namespace slope_of_line_I_l217_217829

-- Line I intersects y = 1 at point P
def intersects_y_eq_one (I P : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, P (x, 1) ↔ I (x, y) ∧ y = 1

-- Line I intersects x - y - 7 = 0 at point Q
def intersects_x_minus_y_eq_seven (I Q : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, Q (x, y) ↔ I (x, y) ∧ x - y - 7 = 0

-- The coordinates of the midpoint of segment PQ are (1, -1)
def midpoint_eq (P Q : ℝ × ℝ) : Prop :=
∃ x1 y1 x2 y2 : ℝ,
  P = (x1, y1) ∧ Q = (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)

-- We need to show that the slope of line I is -2/3
def slope_of_I (I : ℝ × ℝ → Prop) (k : ℝ) : Prop :=
∀ x y : ℝ, I (x, y) → y + 1 = k * (x - 1)

theorem slope_of_line_I :
  ∃ I P Q : (ℝ × ℝ → Prop),
    intersects_y_eq_one I P ∧
    intersects_x_minus_y_eq_seven I Q ∧
    (∃ x1 y1 x2 y2 : ℝ, P (x1, y1) ∧ Q (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)) →
    slope_of_I I (-2/3) :=
by
  sorry

end slope_of_line_I_l217_217829


namespace temperature_on_April_15_and_19_l217_217096

/-
We define the daily temperatures as functions of the temperature on April 15 (T_15) with the given increment of 1.5 degrees each day. 
T_15 represents the temperature on April 15.
-/
theorem temperature_on_April_15_and_19 (T : ℕ → ℝ) (T_avg : ℝ) (inc : ℝ) 
  (h1 : inc = 1.5)
  (h2 : T_avg = 17.5)
  (h3 : ∀ n, T (15 + n) = T 15 + inc * n)
  (h4 : (T 15 + T 16 + T 17 + T 18 + T 19) / 5 = T_avg) :
  T 15 = 14.5 ∧ T 19 = 20.5 :=
by
  sorry

end temperature_on_April_15_and_19_l217_217096


namespace difference_between_percent_and_fraction_l217_217057

-- Define the number
def num : ℕ := 140

-- Define the percentage and fraction calculations
def percent_65 (n : ℕ) : ℕ := (65 * n) / 100
def fraction_4_5 (n : ℕ) : ℕ := (4 * n) / 5

-- Define the problem's conditions and the required proof
theorem difference_between_percent_and_fraction : 
  percent_65 num ≤ fraction_4_5 num ∧ (fraction_4_5 num - percent_65 num = 21) :=
by
  sorry

end difference_between_percent_and_fraction_l217_217057


namespace no_nonzero_integer_solution_l217_217616

theorem no_nonzero_integer_solution (m n p : ℤ) :
  (m + n * Real.sqrt 2 + p * Real.sqrt 3 = 0) → (m = 0 ∧ n = 0 ∧ p = 0) :=
by sorry

end no_nonzero_integer_solution_l217_217616


namespace master_craftsman_total_parts_l217_217890

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l217_217890


namespace range_of_f_gt_f_2x_l217_217570

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * abs (x - 1)

theorem range_of_f_gt_f_2x :
  {x : ℝ | f x > f (2 * x)} = set.Ioo 0 (2 / 3) :=
by
  sorry

end range_of_f_gt_f_2x_l217_217570


namespace goods_train_speed_l217_217998

theorem goods_train_speed (train_length platform_length : ℝ) (time_sec : ℝ) : 
  train_length = 270.0416 ∧ platform_length = 250 ∧ time_sec = 26 → 
  (train_length + platform_length) / time_sec * 3.6 = 72.00576 :=
by
  sorry

end goods_train_speed_l217_217998


namespace simplify_sqrt_sum_l217_217938

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l217_217938


namespace find_all_pairs_l217_217303

def is_solution (m n : ℕ) : Prop := 200 * m + 6 * n = 2006

def valid_pairs : List (ℕ × ℕ) := [(1, 301), (4, 201), (7, 101), (10, 1)]

theorem find_all_pairs :
  ∀ (m n : ℕ), is_solution m n ↔ (m, n) ∈ valid_pairs := by sorry

end find_all_pairs_l217_217303


namespace max_snowmen_l217_217684

def can_stack (a b : ℕ) : Prop := a >= 2 * b

def mass_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

theorem max_snowmen :
  ∃ n, n = 24 ∧ 
      (∀ snowmen : list (ℕ × ℕ × ℕ), 
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → mass_range s.1 ∧ mass_range s.2 ∧ mass_range s.3) ∧
        (∀ (s : ℕ × ℕ × ℕ), s ∈ snowmen → can_stack s.1 s.2 ∧ can_stack s.2 s.3) ∧
        snowmen.length = 24
      ) := sorry

end max_snowmen_l217_217684


namespace coats_collected_in_total_l217_217360

def high_school_coats : Nat := 6922
def elementary_school_coats : Nat := 2515
def total_coats : Nat := 9437

theorem coats_collected_in_total : 
  high_school_coats + elementary_school_coats = total_coats := 
  by
  sorry

end coats_collected_in_total_l217_217360


namespace num_valid_10_digit_sequences_l217_217182

theorem num_valid_10_digit_sequences : 
  ∃ (n : ℕ), n = 64 ∧ 
  (∀ (seq : Fin 10 → Fin 3), 
    (∀ i : Fin 9, abs (seq i.succ - seq i) = 1) → 
    (∀ i : Fin 10, seq i < 3) →
    ∃ k : Nat, k = 10 ∧ seq 0 < 10 ∧ seq 1 < 10 ∧ seq 2 < 10 ∧ seq 3 < 10 ∧ 
      seq 4 < 10 ∧ seq 5 < 10 ∧ seq 6 < 10 ∧ seq 7 < 10 ∧ 
      seq 8 < 10 ∧ seq 9 < 10 ∧ k = 10 → n = 64) :=
sorry

end num_valid_10_digit_sequences_l217_217182


namespace correct_equation_l217_217807

theorem correct_equation (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by
  sorry

end correct_equation_l217_217807


namespace relationship_of_y_values_l217_217352

theorem relationship_of_y_values (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  (y₁ = (k^2 + 3) / (-3)) ∧ (y₂ = (k^2 + 3) / (-1)) ∧ (y₃ = (k^2 + 3) / 2) →
  y₂ < y₁ ∧ y₁ < y₃ :=
by
  intro h
  have h₁ : y₁ = (k^2 + 3) / (-3) := h.1
  have h₂ : y₂ = (k^2 + 3) / (-1) := h.2.1
  have h₃ : y₃ = (k^2 + 3) / 2 := h.2.2
  sorry

end relationship_of_y_values_l217_217352


namespace selling_price_correct_l217_217827

-- Define the conditions
def cost_price : ℝ := 900
def gain_percentage : ℝ := 0.2222222222222222

-- Define the selling price calculation
def profit := cost_price * gain_percentage
def selling_price := cost_price + profit

-- The problem statement in Lean 4
theorem selling_price_correct : selling_price = 1100 := 
by
  -- Proof to be filled in later
  sorry

end selling_price_correct_l217_217827


namespace face_value_shares_l217_217109

theorem face_value_shares (market_value : ℝ) (dividend_rate desired_rate : ℝ) (FV : ℝ) 
  (h1 : dividend_rate = 0.09)
  (h2 : desired_rate = 0.12)
  (h3 : market_value = 36.00000000000001)
  (h4 : (dividend_rate * FV) = (desired_rate * market_value)) :
  FV = 48.00000000000001 :=
by
  sorry

end face_value_shares_l217_217109


namespace finite_tasty_integers_l217_217124

def is_terminating_decimal (a b : ℕ) : Prop :=
  ∃ (c : ℕ), (b = c * 2^a * 5^a)

def is_tasty (n : ℕ) : Prop :=
  n > 2 ∧ ∀ (a b : ℕ), a + b = n → (is_terminating_decimal a b ∨ is_terminating_decimal b a)

theorem finite_tasty_integers : 
  ∃ (N : ℕ), ∀ (n : ℕ), n > N → ¬ is_tasty n :=
sorry

end finite_tasty_integers_l217_217124


namespace triangle_side_and_altitude_sum_l217_217527

theorem triangle_side_and_altitude_sum 
(x y : ℕ) (h1 : x < 75) (h2 : y < 28)
(h3 : x * 60 = 75 * 28) (h4 : 100 * y = 75 * 28) : 
x + y = 56 := 
sorry

end triangle_side_and_altitude_sum_l217_217527


namespace function_values_at_mean_l217_217173

noncomputable def f (x : ℝ) : ℝ := x^2 - 10 * x + 16

theorem function_values_at_mean (x₁ x₂ : ℝ) (h₁ : x₁ = 8) (h₂ : x₂ = 2) :
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  f x' = -9 ∧ f x'' = -8 := by
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  have hx' : x' = 5 := sorry
  have hx'' : x'' = 4 := sorry
  have hf_x' : f x' = -9 := sorry
  have hf_x'' : f x'' = -8 := sorry
  exact ⟨hf_x', hf_x''⟩

end function_values_at_mean_l217_217173


namespace polygon_not_hexagon_if_quadrilateral_after_cut_off_l217_217585

-- Definition of polygonal shape and quadrilateral condition
def is_quadrilateral (sides : Nat) : Prop := sides = 4

-- Definition of polygonal shape with general condition of cutting off one angle
def after_cut_off (original_sides : Nat) (remaining_sides : Nat) : Prop :=
  original_sides > remaining_sides ∧ remaining_sides + 1 = original_sides

-- Problem statement: If a polygon's one angle cut-off results in a quadrilateral, then it is not a hexagon
theorem polygon_not_hexagon_if_quadrilateral_after_cut_off
  (original_sides : Nat) (remaining_sides : Nat) :
  after_cut_off original_sides remaining_sides → is_quadrilateral remaining_sides → original_sides ≠ 6 :=
by
  sorry

end polygon_not_hexagon_if_quadrilateral_after_cut_off_l217_217585


namespace find_negative_a_l217_217204

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 0 then -x else 3 * x - 22

theorem find_negative_a (a : ℝ) (ha : a < 0) :
  g (g (g 7)) = g (g (g a)) ↔ a = -23 / 3 :=
by
  sorry

end find_negative_a_l217_217204


namespace sufficient_but_not_necessary_l217_217392

theorem sufficient_but_not_necessary (x : ℝ) : ((0 < x) → (|x-1| - |x| ≤ 1)) ∧ ((|x-1| - |x| ≤ 1) → True) ∧ ¬((|x-1| - |x| ≤ 1) → (0 < x)) := sorry

end sufficient_but_not_necessary_l217_217392


namespace share_of_a_l217_217985

variables (A B C : ℝ)

def conditions :=
  A = (2 / 3) * (B + C) ∧
  B = (2 / 3) * (A + C) ∧
  A + B + C = 700

theorem share_of_a (h : conditions A B C) : A = 280 :=
by { sorry }

end share_of_a_l217_217985


namespace Jon_regular_bottle_size_is_16oz_l217_217342

noncomputable def Jon_bottle_size (x : ℝ) : Prop :=
  let daily_intake := 4 * x + 2 * 1.25 * x
  let weekly_intake := 7 * daily_intake
  weekly_intake = 728

theorem Jon_regular_bottle_size_is_16oz : ∃ x : ℝ, Jon_bottle_size x ∧ x = 16 :=
by
  use 16
  sorry

end Jon_regular_bottle_size_is_16oz_l217_217342


namespace sufficient_condition_l217_217123

theorem sufficient_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 1) : ab > 1 :=
sorry

end sufficient_condition_l217_217123


namespace stock_decrease_required_l217_217834

theorem stock_decrease_required (x : ℝ) (h : x > 0) : 
  (∃ (p : ℝ), (1 - p) * 1.40 * x = x ∧ p * 100 = 28.57) :=
sorry

end stock_decrease_required_l217_217834


namespace number_of_solutions_l217_217730

theorem number_of_solutions (x y: ℕ) (hx : 0 < x) (hy : 0 < y) :
    (1 / (x + 1) + 1 / y + 1 / ((x + 1) * y) = 1 / 1991) →
    ∃! (n : ℕ), n = 64 :=
by
  sorry

end number_of_solutions_l217_217730


namespace find_a_l217_217064

variable (a : ℤ) -- We assume a is an integer for simplicity

def point_on_x_axis (P : Nat × ℤ) : Prop :=
  P.snd = 0

theorem find_a (h : point_on_x_axis (4, 2 * a + 6)) : a = -3 :=
by
  sorry

end find_a_l217_217064


namespace trajectory_of_moving_circle_l217_217861

-- Define the conditions
def passes_through (M : ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  M = A

def tangent_to_line (M : ℝ × ℝ) (l : ℝ) : Prop :=
  M.1 = -l

noncomputable def equation_of_trajectory (M : ℝ × ℝ) : Prop :=
  M.2 ^ 2 = 12 * M.1

theorem trajectory_of_moving_circle 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (l : ℝ)
  (h1 : passes_through M (3, 0))
  (h2 : tangent_to_line M 3)
  : equation_of_trajectory M := 
sorry

end trajectory_of_moving_circle_l217_217861


namespace simplify_sqrt_expression_is_correct_l217_217931

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l217_217931


namespace find_y_l217_217324

variable {x y : ℤ}

-- Definition 1: The first condition x - y = 20
def condition1 : Prop := x - y = 20

-- Definition 2: The second condition x + y = 10
def condition2 : Prop := x + y = 10

-- The main theorem to prove that y = -5 given the above conditions
theorem find_y (h1 : condition1) (h2 : condition2) : y = -5 :=
  sorry

end find_y_l217_217324


namespace apples_left_l217_217601

-- Define the initial number of apples and the conditions
def initial_apples := 150
def percent_sold_to_jill := 20 / 100
def percent_sold_to_june := 30 / 100
def apples_given_to_teacher := 2

-- Formulate the problem statement in Lean
theorem apples_left (initial_apples percent_sold_to_jill percent_sold_to_june apples_given_to_teacher : ℕ) :
  let sold_to_jill := percent_sold_to_jill * initial_apples
  let remaining_after_jill := initial_apples - sold_to_jill
  let sold_to_june := percent_sold_to_june * remaining_after_jill
  let remaining_after_june := remaining_after_jill - sold_to_june
  let final_apples := remaining_after_june - apples_given_to_teacher
  final_apples = 82 := 
by 
  sorry

end apples_left_l217_217601


namespace determine_values_l217_217298

-- Definitions of the conditions
variables {x y z : ℝ}
axiom condition_1 : |x - y^2| = z * x + y^2
axiom condition_2 : z * x + y^2 ≥ 0

-- Target conclusion
theorem determine_values (h1 : condition_1) (h2 : condition_2) :
  (x = 0 ∧ y = 0) ∨ (x = 2 * y^2 / (1 - z) ∧ z ≠ 1 ∧ z > -1) :=
sorry

end determine_values_l217_217298


namespace total_sample_size_l217_217403

theorem total_sample_size
    (undergrad_count : ℕ) (masters_count : ℕ) (doctoral_count : ℕ)
    (total_students : ℕ) (sample_size_doctoral : ℕ) (proportion_sample : ℕ)
    (n : ℕ)
    (H1 : undergrad_count = 12000)
    (H2 : masters_count = 1000)
    (H3 : doctoral_count = 200)
    (H4 : total_students = undergrad_count + masters_count + doctoral_count)
    (H5 : sample_size_doctoral = 20)
    (H6 : proportion_sample = sample_size_doctoral / doctoral_count)
    (H7 : n = proportion_sample * total_students) :
  n = 1320 := 
sorry

end total_sample_size_l217_217403


namespace one_person_remains_dry_l217_217477

theorem one_person_remains_dry (n : ℕ) :
  ∃ (person_dry : ℕ -> Bool), (∀ i : ℕ, i < 2 * n + 1 -> person_dry i = tt) := 
sorry

end one_person_remains_dry_l217_217477


namespace max_piece_length_l217_217147

theorem max_piece_length (a b c : ℕ) (h1 : a = 60) (h2 : b = 75) (h3 : c = 90) :
  Nat.gcd (Nat.gcd a b) c = 15 :=
by 
  sorry

end max_piece_length_l217_217147


namespace sum_of_all_possible_k_values_l217_217033

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l217_217033


namespace coin_count_l217_217255

theorem coin_count (x : ℝ) (h₁ : x + 0.50 * x + 0.25 * x = 35) : x = 20 :=
by
  sorry

end coin_count_l217_217255


namespace king_total_payment_l217_217679

theorem king_total_payment
  (crown_cost : ℕ)
  (architect_cost : ℕ)
  (chef_cost : ℕ)
  (crown_tip_percent : ℕ)
  (architect_tip_percent : ℕ)
  (chef_tip_percent : ℕ)
  (crown_tip : ℕ)
  (architect_tip : ℕ)
  (chef_tip : ℕ)
  (total_crown_cost : ℕ)
  (total_architect_cost : ℕ)
  (total_chef_cost : ℕ)
  (total_paid : ℕ) :
  crown_cost = 20000 →
  architect_cost = 50000 →
  chef_cost = 10000 →
  crown_tip_percent = 10 →
  architect_tip_percent = 5 →
  chef_tip_percent = 15 →
  crown_tip = crown_cost * crown_tip_percent / 100 →
  architect_tip = architect_cost * architect_tip_percent / 100 →
  chef_tip = chef_cost * chef_tip_percent / 100 →
  total_crown_cost = crown_cost + crown_tip →
  total_architect_cost = architect_cost + architect_tip →
  total_chef_cost = chef_cost + chef_tip →
  total_paid = total_crown_cost + total_architect_cost + total_chef_cost →
  total_paid = 86000 := by
  sorry

end king_total_payment_l217_217679


namespace gcd_3060_561_l217_217969

theorem gcd_3060_561 : Nat.gcd 3060 561 = 51 :=
by
  sorry

end gcd_3060_561_l217_217969


namespace value_of_composition_l217_217911

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x - 1

theorem value_of_composition : g (f (1 + 2 * g 3)) = 12 := by
  sorry

end value_of_composition_l217_217911


namespace packs_of_chocolate_l217_217766

theorem packs_of_chocolate (t c k x : ℕ) (ht : t = 42) (hc : c = 4) (hk : k = 22) (hx : x = t - (c + k)) : x = 16 :=
by
  rw [ht, hc, hk] at hx
  simp at hx
  exact hx

end packs_of_chocolate_l217_217766


namespace gold_bars_total_worth_l217_217054

theorem gold_bars_total_worth :
  let rows := 4
  let bars_per_row := 20
  let worth_per_bar : ℕ := 20000
  let total_bars := rows * bars_per_row
  let total_worth := total_bars * worth_per_bar
  total_worth = 1600000 :=
by
  sorry

end gold_bars_total_worth_l217_217054


namespace cannot_fit_480_pictures_l217_217061

theorem cannot_fit_480_pictures 
  (A_capacity : ℕ) (B_capacity : ℕ) (C_capacity : ℕ) 
  (n_A : ℕ) (n_B : ℕ) (n_C : ℕ) 
  (total_pictures : ℕ) : 
  A_capacity = 12 → B_capacity = 18 → C_capacity = 24 → 
  n_A = 6 → n_B = 4 → n_C = 3 → 
  total_pictures = 480 → 
  A_capacity * n_A + B_capacity * n_B + C_capacity * n_C < total_pictures :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end cannot_fit_480_pictures_l217_217061


namespace intersection_in_quadrant_II_l217_217846

theorem intersection_in_quadrant_II (x y : ℝ) 
  (h1: y ≥ -2 * x + 3) 
  (h2: y ≤ 3 * x + 6) 
  (h_intersection: x = -3 / 5 ∧ y = 21 / 5) :
  x < 0 ∧ y > 0 := 
sorry

end intersection_in_quadrant_II_l217_217846


namespace compute_63_times_57_l217_217135

theorem compute_63_times_57 : 63 * 57 = 3591 := 
by {
   have h : (60 + 3) * (60 - 3) = 60^2 - 3^2, from
     by simp [mul_add, add_mul, add_assoc, sub_mul, mul_sub, sub_add, sub_sub, add_sub, mul_self_sub],
   have h1 : 60^2 = 3600, from rfl,
   have h2 : 3^2 = 9, from rfl,
   have h3 : 60^2 - 3^2 = 3600 - 9, by rw [h1, h2],
   rw h at h3,
   exact h3,
}

end compute_63_times_57_l217_217135


namespace fraction_of_students_participated_l217_217395

theorem fraction_of_students_participated (total_students : ℕ) (did_not_participate : ℕ)
  (h_total : total_students = 39) (h_did_not_participate : did_not_participate = 26) :
  (total_students - did_not_participate) / total_students = 1 / 3 :=
by
  sorry

end fraction_of_students_participated_l217_217395


namespace apples_per_pie_l217_217700

-- Conditions
def initial_apples : ℕ := 50
def apples_per_teacher_per_child : ℕ := 3
def number_of_teachers : ℕ := 2
def number_of_children : ℕ := 2
def remaining_apples : ℕ := 24

-- Proof goal: the number of apples Jill uses per pie
theorem apples_per_pie : 
  initial_apples 
  - (apples_per_teacher_per_child * number_of_teachers * number_of_children)  - remaining_apples = 14 -> 14 / 2 = 7 := 
by
  sorry

end apples_per_pie_l217_217700


namespace solve_equations_l217_217219

theorem solve_equations (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) : a + b = 82 / 7 := by
  sorry

end solve_equations_l217_217219


namespace vertical_complementary_perpendicular_l217_217586

theorem vertical_complementary_perpendicular (α β : ℝ) (l1 l2 : ℝ) :
  (α = β ∧ α + β = 90) ∧ l1 = l2 -> l1 + l2 = 90 := by
  sorry

end vertical_complementary_perpendicular_l217_217586


namespace gardener_cabbages_this_year_l217_217390

-- Definitions for the conditions
def side_length_last_year (x : ℕ) := true
def area_last_year (x : ℕ) := x * x
def increase_in_output := 197

-- Proposition to prove the number of cabbages this year
theorem gardener_cabbages_this_year (x : ℕ) (hx : side_length_last_year x) : 
  (area_last_year x + increase_in_output) = 9801 :=
by 
  sorry

end gardener_cabbages_this_year_l217_217390


namespace LCM_20_45_75_is_900_l217_217658

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l217_217658


namespace triangle_area_is_4_l217_217021

variable {PQ RS : ℝ} -- lengths of PQ and RS respectively
variable {area_PQRS area_PQS : ℝ} -- areas of the trapezoid and triangle respectively

-- Given conditions
@[simp]
def trapezoid_area_is_12 (area_PQRS : ℝ) : Prop :=
  area_PQRS = 12

@[simp]
def RS_is_twice_PQ (PQ RS : ℝ) : Prop :=
  RS = 2 * PQ

-- To prove: the area of triangle PQS is 4 given the conditions
theorem triangle_area_is_4 (h1 : trapezoid_area_is_12 area_PQRS)
                          (h2 : RS_is_twice_PQ PQ RS)
                          (h3 : area_PQRS = 3 * area_PQS) : area_PQS = 4 :=
by
  sorry

end triangle_area_is_4_l217_217021


namespace master_craftsman_total_parts_l217_217891

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l217_217891


namespace value_of_expression_l217_217071

theorem value_of_expression : 1 + 2 + 3 - 4 + 5 + 6 + 7 - 8 + 9 + 10 + 11 - 12 = 30 :=
by
  sorry

end value_of_expression_l217_217071


namespace solve_fractional_equation_l217_217791

theorem solve_fractional_equation : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -3 → (1 / x = 2 / (x + 3) ↔ x = 3) :=
by
  intros x hx
  sorry

end solve_fractional_equation_l217_217791


namespace gun_fan_image_equivalence_l217_217677

def gunPiercingImage : String := "point moving to form a line"
def foldingFanImage : String := "line moving to form a surface"

theorem gun_fan_image_equivalence :
  (gunPiercingImage = "point moving to form a line") ∧ 
  (foldingFanImage = "line moving to form a surface") := by
  -- Proof goes here
  sorry

end gun_fan_image_equivalence_l217_217677


namespace total_cars_l217_217698

theorem total_cars (yesterday today : ℕ) (h_yesterday : yesterday = 60) (h_today : today = 2 * yesterday) : yesterday + today = 180 := 
sorry

end total_cars_l217_217698


namespace tan_sin_cos_log_expression_simplification_l217_217986

-- Proof Problem 1 Statement in Lean 4
theorem tan_sin_cos (α : ℝ) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by
  sorry

-- Proof Problem 2 Statement in Lean 4
theorem log_expression_simplification : 
  Real.logb 3 (Real.sqrt 27) + Real.logb 10 25 + Real.logb 10 4 + 
  (7 : ℝ) ^ Real.logb 7 2 + (-9.8) ^ 0 = 13 / 2 :=
by
  sorry

end tan_sin_cos_log_expression_simplification_l217_217986


namespace confectioner_pastry_l217_217515

theorem confectioner_pastry (P : ℕ) (h : P / 28 - 6 = P / 49) : P = 378 :=
sorry

end confectioner_pastry_l217_217515


namespace profit_percentage_l217_217990

def cost_price : ℝ := 60
def selling_price : ℝ := 78

theorem profit_percentage : ((selling_price - cost_price) / cost_price) * 100 = 30 := 
by
  sorry

end profit_percentage_l217_217990


namespace g_five_eq_one_l217_217368

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x z : ℝ) : g (x * z) = g x * g z
axiom g_one_ne_zero : g (1) ≠ 0

theorem g_five_eq_one : g (5) = 1 := 
by
  sorry

end g_five_eq_one_l217_217368


namespace sixty_three_times_fifty_seven_l217_217139

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 := by
  sorry

end sixty_three_times_fifty_seven_l217_217139


namespace coronavirus_case_ratio_l217_217950

theorem coronavirus_case_ratio (n_first_wave_cases : ℕ) (total_second_wave_cases : ℕ) (n_days : ℕ) 
  (h1 : n_first_wave_cases = 300) (h2 : total_second_wave_cases = 21000) (h3 : n_days = 14) :
  (total_second_wave_cases / n_days) / n_first_wave_cases = 5 :=
by sorry

end coronavirus_case_ratio_l217_217950


namespace largest_positive_integer_l217_217740

def binary_operation (n : Int) : Int := n - (n * 5)

theorem largest_positive_integer (n : Int) : (∀ m : Int, m > 0 → n - (n * 5) < -19 → m ≤ n) 
  ↔ n = 5 := 
by
  sorry

end largest_positive_integer_l217_217740


namespace compare_abc_l217_217563

noncomputable def a : ℝ :=
  (1/2) * Real.cos 16 - (Real.sqrt 3 / 2) * Real.sin 16

noncomputable def b : ℝ :=
  2 * Real.tan 14 / (1 + (Real.tan 14) ^ 2)

noncomputable def c : ℝ :=
  Real.sqrt ((1 - Real.cos 50) / 2)

theorem compare_abc : b > c ∧ c > a :=
  by sorry

end compare_abc_l217_217563


namespace lamps_remain_lit_after_toggling_l217_217227

theorem lamps_remain_lit_after_toggling :
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  1997 - pulled_three_times - pulled_once = 999 := by
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  have h : 1997 - pulled_three_times - (pulled_once) = 999 := sorry
  exact h

end lamps_remain_lit_after_toggling_l217_217227


namespace sum_of_box_dimensions_l217_217833

theorem sum_of_box_dimensions (X Y Z : ℝ) (h1 : X * Y = 32) (h2 : X * Z = 50) (h3 : Y * Z = 80) :
  X + Y + Z = 25.5 * Real.sqrt 2 :=
by sorry

end sum_of_box_dimensions_l217_217833


namespace arrangements_ABC_together_l217_217226

noncomputable def permutation_count_ABC_together (n : Nat) (unit_size : Nat) (remaining : Nat) : Nat :=
  (Nat.factorial unit_size) * (Nat.factorial (remaining + 1))

theorem arrangements_ABC_together : permutation_count_ABC_together 6 3 3 = 144 :=
by
  sorry

end arrangements_ABC_together_l217_217226


namespace nth_group_sum_correct_l217_217631

-- Define the function that computes the sum of the numbers in the nth group
def nth_group_sum (n : ℕ) : ℕ :=
  n * (n^2 + 1) / 2

-- The theorem statement
theorem nth_group_sum_correct (n : ℕ) : 
  nth_group_sum n = n * (n^2 + 1) / 2 := by
  sorry

end nth_group_sum_correct_l217_217631


namespace true_compound_propositions_l217_217291

-- Define the propositions
def p1 : Prop := 
  ∀ (l₁ l₂ l₃ : Line), 
    (l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
    (∃ (p₁ p₂ p₃ : Point),
      l₁.contains p₁ ∧ l₁.contains p₂ ∧
      l₂.contains p₂ ∧ l₂.contains p₃ ∧
      l₃.contains p₃ ∧ l₃.contains p₁)
    → ∃ (α : Plane), 
      α.contains l₁ ∧ α.contains l₂ ∧ α.contains l₃

def p2 : Prop :=
  ∀ (a b c : Point),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    → ∃! (α : Plane), 
      α.contains a ∧ α.contains b ∧ α.contains c

def p3 : Prop :=
  ∀ (l₁ l₂ : Line),
    (¬ ∃ (p : Point), l₁.contains p ∧ l₂.contains p)
    → l₁.parallel l₂

def p4 : Prop :=
  ∀ (α : Plane) (l m : Line),
    (α.contains l ∧ ¬ α.contains m ∧ m.perpendicular α) 
    → m.perpendicular l

-- Main theorem that identifies the true compound propositions
theorem true_compound_propositions :
  (p1 ∧ p4) ∧ ¬ (p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) :=
by
  -- Proof might involve defining lines, points, and planes and their relationships
  sorry

end true_compound_propositions_l217_217291


namespace lights_on_bottom_layer_l217_217747

theorem lights_on_bottom_layer
  (a₁ : ℕ)
  (q : ℕ := 3)
  (S₅ : ℕ := 242)
  (n : ℕ := 5)
  (sum_formula : S₅ = (a₁ * (q^n - 1)) / (q - 1)) :
  (a₁ * q^(n-1) = 162) :=
by
  sorry

end lights_on_bottom_layer_l217_217747


namespace lines_intersect_l217_217108

def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 4 * v, 9 - v)

theorem lines_intersect :
  ∃ s v : ℚ, (line1 s) = (line2 v) ∧ (line1 s) = (-17/5, 53/5) := 
sorry

end lines_intersect_l217_217108


namespace parabola_behavior_l217_217460

-- Definitions for the conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- The proof statement
theorem parabola_behavior (a : ℝ) (x : ℝ) (ha : 0 < a) : 
  (0 < a ∧ a < 1 → parabola a x < x^2) ∧
  (a > 1 → parabola a x > x^2) ∧
  (∀ ε > 0, ∃ δ > 0, δ ≤ a → |parabola a x - 0| < ε) := 
sorry

end parabola_behavior_l217_217460


namespace incenter_correct_l217_217336

variable (P Q R : Type) [AddCommGroup P] [Module ℝ P]
variable (p q r : ℝ)
variable (P_vec Q_vec R_vec : P)

noncomputable def incenter_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (p / (p + q + r), q / (p + q + r), r / (p + q + r))

theorem incenter_correct : 
  incenter_coordinates 8 10 6 = (1/3, 5/12, 1/4) := by
  sorry

end incenter_correct_l217_217336


namespace possible_values_of_k_l217_217361

noncomputable def has_roots (p q r s t k : ℂ) : Prop :=
  (p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0) ∧ 
  (p * k^4 + q * k^3 + r * k^2 + s * k + t = 0) ∧
  (q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)

theorem possible_values_of_k (p q r s t k : ℂ) (hk : has_roots p q r s t k) : 
  k^5 = 1 :=
  sorry

end possible_values_of_k_l217_217361


namespace find_first_number_l217_217222

/-- The Least Common Multiple (LCM) of two numbers A and B is 2310,
    and their Highest Common Factor (HCF) is 30.
    Given one of the numbers B is 180, find the other number A. -/
theorem find_first_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 30) (h3 : B = 180) (h4 : A * B = LCM * HCF) :
  A = 385 :=
by sorry

end find_first_number_l217_217222


namespace minimum_product_OP_OQ_l217_217004

theorem minimum_product_OP_OQ (a b : ℝ) (P Q : ℝ × ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : P ≠ Q) (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1) (h5 : Q.1 ^ 2 / a ^ 2 + Q.2 ^ 2 / b ^ 2 = 1)
  (h6 : P.1 * Q.1 + P.2 * Q.2 = 0) :
  (P.1 ^ 2 + P.2 ^ 2) * (Q.1 ^ 2 + Q.2 ^ 2) ≥ (2 * a ^ 2 * b ^ 2 / (a ^ 2 + b ^ 2)) :=
by sorry

end minimum_product_OP_OQ_l217_217004


namespace car_with_highest_avg_speed_l217_217538

-- Conditions
def distance_A : ℕ := 715
def time_A : ℕ := 11
def distance_B : ℕ := 820
def time_B : ℕ := 12
def distance_C : ℕ := 950
def time_C : ℕ := 14

-- Average Speeds
def avg_speed_A : ℚ := distance_A / time_A
def avg_speed_B : ℚ := distance_B / time_B
def avg_speed_C : ℚ := distance_C / time_C

-- Theorem
theorem car_with_highest_avg_speed : avg_speed_B > avg_speed_A ∧ avg_speed_B > avg_speed_C :=
by
  sorry

end car_with_highest_avg_speed_l217_217538


namespace repeating_decimal_to_fraction_l217_217852

noncomputable def repeating_decimal_sum (x y z : ℚ) : ℚ := x + y + z

theorem repeating_decimal_to_fraction :
  let x := 4 / 33
  let y := 34 / 999
  let z := 567 / 99999
  repeating_decimal_sum x y z = 134255 / 32929667 := by
  -- proofs are omitted
  sorry

end repeating_decimal_to_fraction_l217_217852


namespace regular_polygon_perimeter_l217_217694

theorem regular_polygon_perimeter (s : ℝ) (exterior_angle : ℝ) 
  (h1 : s = 7) (h2 : exterior_angle = 45) : 
  8 * s = 56 :=
by
  sorry

end regular_polygon_perimeter_l217_217694


namespace f_sum_neg_l217_217169

def f : ℝ → ℝ := sorry

theorem f_sum_neg (x₁ x₂ : ℝ)
  (h1 : ∀ x, f (4 - x) = - f x)
  (h2 : ∀ x, x < 2 → ∀ y, y < x → f y < f x)
  (h3 : x₁ + x₂ > 4)
  (h4 : (x₁ - 2) * (x₂ - 2) < 0)
  : f x₁ + f x₂ < 0 := 
sorry

end f_sum_neg_l217_217169


namespace line_through_origin_and_intersection_of_lines_l217_217854

theorem line_through_origin_and_intersection_of_lines 
  (x y : ℝ)
  (h1 : x - 3 * y + 4 = 0)
  (h2 : 2 * x + y + 5 = 0) :
  3 * x + 19 * y = 0 :=
sorry

end line_through_origin_and_intersection_of_lines_l217_217854


namespace min_coins_needed_l217_217801

-- Definitions for coins
def coins (pennies nickels dimes quarters : Nat) : Nat :=
  pennies + nickels + dimes + quarters

-- Condition: minimum number of coins to pay any amount less than a dollar
def can_pay_any_amount (pennies nickels dimes quarters : Nat) : Prop :=
  ∀ (amount : Nat), 1 ≤ amount ∧ amount < 100 →
  ∃ (p n d q : Nat), p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧
  p + 5 * n + 10 * d + 25 * q = amount

-- The main Lean 4 statement
theorem min_coins_needed :
  ∃ (pennies nickels dimes quarters : Nat),
    coins pennies nickels dimes quarters = 11 ∧
    can_pay_any_amount pennies nickels dimes quarters :=
sorry

end min_coins_needed_l217_217801


namespace fish_added_l217_217733

theorem fish_added (x : ℕ) (hx : x + (x - 4) = 20) : x - 4 = 8 := by
  sorry

end fish_added_l217_217733


namespace modified_cube_edges_l217_217525

/--
A solid cube with a side length of 4 has different-sized solid cubes removed from three of its corners:
- one corner loses a cube of side length 1,
- another corner loses a cube of side length 2,
- and a third corner loses a cube of side length 1.

The total number of edges of the modified solid is 22.
-/
theorem modified_cube_edges :
  let original_edges := 12
  let edges_removed_1x1 := 6
  let edges_added_2x2 := 16
  original_edges - 2 * edges_removed_1x1 + edges_added_2x2 = 22 := by
  sorry

end modified_cube_edges_l217_217525


namespace trapezoid_QR_length_l217_217624

variable (PQ RS Area Alt QR : ℝ)
variable (h1 : Area = 216)
variable (h2 : Alt = 9)
variable (h3 : PQ = 12)
variable (h4 : RS = 20)
variable (h5 : QR = 11)

theorem trapezoid_QR_length : 
  (∃ (PQ RS Area Alt QR : ℝ), 
    Area = 216 ∧
    Alt = 9 ∧
    PQ = 12 ∧
    RS = 20) → QR = 11 :=
by
  sorry

end trapezoid_QR_length_l217_217624


namespace roots_quad_sum_abs_gt_four_sqrt_three_l217_217577

theorem roots_quad_sum_abs_gt_four_sqrt_three
  (p r1 r2 : ℝ)
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 12)
  (h3 : p^2 > 48) : 
  |r1 + r2| > 4 * Real.sqrt 3 := 
by 
  sorry

end roots_quad_sum_abs_gt_four_sqrt_three_l217_217577


namespace smallest_a_exists_l217_217242

theorem smallest_a_exists : ∃ a b c : ℕ, 
                          (∀ α β : ℝ, 
                          (α > 0 ∧ α ≤ 1 / 1000) ∧ 
                          (β > 0 ∧ β ≤ 1 / 1000) ∧ 
                          (α + β = -b / a) ∧ 
                          (α * β = c / a) ∧ 
                          (b * b - 4 * a * c > 0)) ∧ 
                          (a = 1001000) := sorry

end smallest_a_exists_l217_217242


namespace find_value_of_a_l217_217382

theorem find_value_of_a (b : ℤ) (q : ℚ) (a : ℤ) (h₁ : b = 2120) (h₂ : q = 0.5) (h₃ : (a : ℚ) / b = q) : a = 1060 :=
sorry

end find_value_of_a_l217_217382


namespace salary_of_E_l217_217633

theorem salary_of_E (A B C D E : ℕ) (avg_salary : ℕ) 
  (hA : A = 8000) 
  (hB : B = 5000) 
  (hC : C = 11000) 
  (hD : D = 7000) 
  (h_avg : avg_salary = 8000) 
  (h_total_avg : avg_salary * 5 = A + B + C + D + E) : 
  E = 9000 :=
by {
  sorry
}

end salary_of_E_l217_217633


namespace triangle_area_l217_217371

-- Define the given conditions
def perimeter : ℝ := 60
def inradius : ℝ := 2.5

-- Prove the area of the triangle using the given inradius and perimeter
theorem triangle_area (p : ℝ) (r : ℝ) (h1 : p = 60) (h2 : r = 2.5) :
  (r * (p / 2)) = 75 := 
by
  rw [h1, h2]
  sorry

end triangle_area_l217_217371


namespace prime_divides_sum_l217_217614

theorem prime_divides_sum 
  (a b c : ℕ) 
  (h1 : a^3 + 4 * b + c = a * b * c)
  (h2 : a ≥ c)
  (h3 : Prime (a^2 + 2 * a + 2)) : 
  (a^2 + 2 * a + 2) ∣ (a + 2 * b + 2) := 
sorry

end prime_divides_sum_l217_217614


namespace calculate_exponent_product_l217_217704

theorem calculate_exponent_product : (2^2021) * (-1/2)^2022 = (1/2) :=
by
  sorry

end calculate_exponent_product_l217_217704


namespace necessary_but_not_sufficient_l217_217821

theorem necessary_but_not_sufficient (a b : ℝ) : (a > b) → (a + 1 > b - 2) :=
by sorry

end necessary_but_not_sufficient_l217_217821


namespace P_n_limit_l217_217201

-- The problem statement and conditions:
noncomputable def equilateral_triangle_side (a : ℝ) := a

noncomputable def point_on_AB (a : ℝ) := λ (n : ℕ), 0 ≤ n

noncomputable def BP_n_distance (a : ℝ) (n : ℕ) : ℝ :=
if n = 0 then a else let P_n_prev := BP_n_distance a (n - 1) in (3 / 4) * a - (1 / 8) * P_n_prev 

-- The final statement that needs to be proven:
theorem P_n_limit (a : ℝ) (n : ℕ) (h : 0 < a) : 
  ∃ L, L = (2 / 3) * a ∧ 
  ∀ ε > 0, ∃ N, ∀ m ≥ N, |BP_n_distance a m - L| < ε :=
begin
  sorry
end

end P_n_limit_l217_217201


namespace exists_group_of_four_l217_217595

-- Assuming 21 students, and any three have done homework together exactly once in either mathematics or Russian.
-- We aim to prove there exists a group of four students such that any three of them have done homework together in the same subject.
noncomputable def students : Type := Fin 21

-- Define a predicate to show that three students have done homework together.
-- We use "math" and "russian" to denote the subjects.
inductive Subject
| math
| russian

-- Define a relation expressing that any three students have done exactly one subject homework together.
axiom homework_done (s1 s2 s3 : students) : Subject 

theorem exists_group_of_four :
  ∃ (a b c d : students), 
    (homework_done a b c = homework_done a b d) ∧
    (homework_done a b c = homework_done a c d) ∧
    (homework_done a b c = homework_done b c d) ∧
    (homework_done a b d = homework_done a c d) ∧
    (homework_done a b d = homework_done b c d) ∧
    (homework_done a c d = homework_done b c d) :=
sorry

end exists_group_of_four_l217_217595


namespace min_sum_xyz_l217_217304

theorem min_sum_xyz (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) 
  (hxyz : x^2 + y^2 + z^2 ≥ 90) : 
  x + y + z ≥ 16 := 
sorry

end min_sum_xyz_l217_217304


namespace op_five_two_is_twentyfour_l217_217068

def op (x y : Int) : Int :=
  (x + y + 1) * (x - y)

theorem op_five_two_is_twentyfour : op 5 2 = 24 := by
  unfold op
  sorry

end op_five_two_is_twentyfour_l217_217068


namespace james_weekly_pistachio_cost_l217_217756

def cost_per_can : ℕ := 10
def ounces_per_can : ℕ := 5
def consumption_per_5_days : ℕ := 30
def days_per_week : ℕ := 7

theorem james_weekly_pistachio_cost : (days_per_week / 5 * consumption_per_5_days) / ounces_per_can * cost_per_can = 90 := 
by
  sorry

end james_weekly_pistachio_cost_l217_217756


namespace normal_price_of_article_l217_217087

theorem normal_price_of_article (P : ℝ) (sale_price : ℝ) (discount1 discount2 : ℝ) :
  discount1 = 0.10 → discount2 = 0.20 → sale_price = 108 →
  P * (1 - discount1) * (1 - discount2) = sale_price → P = 150 :=
by
  intro hd1 hd2 hsp hdiscount
  -- skipping the proof for now
  sorry

end normal_price_of_article_l217_217087


namespace brick_wall_completion_time_l217_217272

def rate (hours : ℚ) : ℚ := 1 / hours

/-- Avery can build a brick wall in 3 hours. -/
def avery_rate : ℚ := rate 3
/-- Tom can build a brick wall in 2.5 hours. -/
def tom_rate : ℚ := rate 2.5
/-- Catherine can build a brick wall in 4 hours. -/
def catherine_rate : ℚ := rate 4
/-- Derek can build a brick wall in 5 hours. -/
def derek_rate : ℚ := rate 5

/-- Combined rate for Avery, Tom, and Catherine working together. -/
def combined_rate_1 : ℚ := avery_rate + tom_rate + catherine_rate
/-- Combined rate for Tom and Catherine working together. -/
def combined_rate_2 : ℚ := tom_rate + catherine_rate
/-- Combined rate for Tom, Catherine, and Derek working together. -/
def combined_rate_3 : ℚ := tom_rate + catherine_rate + derek_rate

/-- Total time taken to complete the wall. -/
def total_time (t : ℚ) : Prop :=
  t = 2

theorem brick_wall_completion_time (t : ℚ) : total_time t :=
by
  sorry

end brick_wall_completion_time_l217_217272


namespace lcm_of_20_45_75_l217_217653

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l217_217653


namespace pedestrian_wait_probability_l217_217260

-- Define the duration of the red light
def red_light_duration := 45

-- Define the favorable time window for the pedestrian to wait at least 20 seconds
def favorable_window := 25

-- The probability that the pedestrian has to wait at least 20 seconds
def probability_wait_at_least_20 : ℚ := favorable_window / red_light_duration

theorem pedestrian_wait_probability : probability_wait_at_least_20 = 5 / 9 := by
  sorry

end pedestrian_wait_probability_l217_217260


namespace right_triangle_third_side_product_l217_217965

theorem right_triangle_third_side_product :
  ∀ (a b : ℝ), (a = 6 ∧ b = 8 ∧ (a^2 + b^2 = c^2 ∨ a^2 = b^2 - c^2)) →
  (a * b = 53.0) :=
by
  intros a b h
  sorry

end right_triangle_third_side_product_l217_217965


namespace new_supervisor_salary_l217_217486

-- Definitions
def average_salary_old (W : ℕ) : Prop :=
  (W + 870) / 9 = 430

def average_salary_new (W : ℕ) (S_new : ℕ) : Prop :=
  (W + S_new) / 9 = 430

-- Problem statement
theorem new_supervisor_salary (W : ℕ) (S_new : ℕ) :
  average_salary_old W →
  average_salary_new W S_new →
  S_new = 870 :=
by
  sorry

end new_supervisor_salary_l217_217486


namespace bakery_combinations_l217_217513

theorem bakery_combinations (h : ∀ (a b c : ℕ), a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ count : ℕ, count = 25 := 
sorry

end bakery_combinations_l217_217513


namespace sufficient_and_necessary_condition_l217_217912

def isMonotonicallyIncreasing {R : Type _} [LinearOrderedField R] (f : R → R) :=
  ∀ x y, x < y → f x < f y

def fx {R : Type _} [LinearOrderedField R] (x m : R) :=
  x^3 + 2*x^2 + m*x + 1

theorem sufficient_and_necessary_condition (m : ℝ) :
  (isMonotonicallyIncreasing (λ x => fx x m) ↔ m ≥ 4/3) :=
  sorry

end sufficient_and_necessary_condition_l217_217912


namespace bunch_of_bananas_cost_l217_217407

def cost_of_bananas (A : ℝ) : ℝ := 5 - A

theorem bunch_of_bananas_cost (A B T : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = T) : B = cost_of_bananas A :=
by
  sorry

end bunch_of_bananas_cost_l217_217407


namespace time_for_A_and_D_together_l217_217809

theorem time_for_A_and_D_together (A_rate D_rate combined_rate : ℝ)
  (hA : A_rate = 1 / 10) (hD : D_rate = 1 / 10) 
  (h_combined : combined_rate = A_rate + D_rate) :
  1 / combined_rate = 5 :=
by
  sorry

end time_for_A_and_D_together_l217_217809


namespace range_of_m_l217_217575

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x - m)/2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2)) →
  ∃ x : ℝ, x = 2 ∧ -3 < m ∧ m ≤ -2 :=
by
  sorry

end range_of_m_l217_217575


namespace master_craftsman_quota_l217_217889

theorem master_craftsman_quota (parts_first_hour : ℕ)
  (extra_hour_needed : ℕ)
  (increased_speed : ℕ)
  (time_diff : ℕ)
  (total_parts : ℕ) :
  parts_first_hour = 35 →
  extra_hour_needed = 1 →
  increased_speed = 15 →
  time_diff = 1.5 →
  total_parts = parts_first_hour + (175 : ℕ) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3]
  norm_num
  rw [add_comm]
  exact sorry

end master_craftsman_quota_l217_217889


namespace find_numbers_l217_217364

theorem find_numbers (a b : ℝ) (h₁ : a - b = 157) (h₂ : a / b = 2) : a = 314 ∧ b = 157 :=
sorry

end find_numbers_l217_217364


namespace compute_f_sum_l217_217777

noncomputable def f : ℝ → ℝ := sorry -- placeholder for f(x)

variables (x : ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom f_definition : ∀ x, 0 < x ∧ x < 1 → f x = x^2

-- Prove the main statement
theorem compute_f_sum : f (-3 / 2) + f 1 = 3 / 4 :=
by
  sorry

end compute_f_sum_l217_217777


namespace father_dig_time_l217_217209

-- Definitions based on the conditions
variable (T : ℕ) -- Time taken by the father to dig the hole in hours
variable (D : ℕ) -- Depth of the hole dug by the father in feet
variable (M : ℕ) -- Depth of the hole dug by Michael in feet

-- Conditions
def father_hole_depth : Prop := D = 4 * T
def michael_hole_depth : Prop := M = 2 * D - 400
def michael_dig_time : Prop := M = 4 * 700

-- The proof statement, proving T = 400 given the conditions
theorem father_dig_time (T D M : ℕ)
  (h1 : father_hole_depth T D)
  (h2 : michael_hole_depth D M)
  (h3 : michael_dig_time M) : T = 400 := 
by
  sorry

end father_dig_time_l217_217209


namespace cider_production_l217_217008

theorem cider_production (gd_pint : ℕ) (pl_pint : ℕ) (gs_pint : ℕ) (farmhands : ℕ) (gd_rate : ℕ) (pl_rate : ℕ) (gs_rate : ℕ) (work_hours : ℕ) 
  (gd_total : ℕ) (pl_total : ℕ) (gs_total : ℕ) (gd_ratio : ℕ) (pl_ratio : ℕ) (gs_ratio : ℕ) 
  (gd_pint_val : gd_pint = 20) (pl_pint_val : pl_pint = 40) (gs_pint_val : gs_pint = 30)
  (farmhands_val : farmhands = 6) (gd_rate_val : gd_rate = 120) (pl_rate_val : pl_rate = 240) (gs_rate_val : gs_rate = 180) 
  (work_hours_val : work_hours = 5) 
  (gd_total_val : gd_total = farmhands * work_hours * gd_rate) 
  (pl_total_val : pl_total = farmhands * work_hours * pl_rate) 
  (gs_total_val : gs_total = farmhands * work_hours * gs_rate) 
  (gd_ratio_val : gd_ratio = 1) (pl_ratio_val : pl_ratio = 2) (gs_ratio_val : gs_ratio = 3/2) 
  (ratio_condition : gd_total / gd_ratio = pl_total / pl_ratio ∧ pl_total / pl_ratio = gs_total / gs_ratio) : 
  (gd_total / gd_pint) = 180 := 
sorry

end cider_production_l217_217008


namespace cos_double_angle_l217_217165

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi + α) = 1 / 3) : Real.cos (2 * α) = 7 / 9 := 
by 
  sorry

end cos_double_angle_l217_217165


namespace log_abs_is_even_l217_217599

open Real

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

noncomputable def f (x : ℝ) : ℝ := log (abs x)

theorem log_abs_is_even : is_even_function f :=
by
  sorry

end log_abs_is_even_l217_217599


namespace parts_manufactured_l217_217895

variable (initial_parts : ℕ) (initial_rate : ℕ) (increased_speed : ℕ) (time_diff : ℝ)
variable (N : ℕ)

-- initial conditions
def initial_parts := 35
def initial_rate := 35
def increased_speed := 15
def time_diff := 1.5

-- additional parts to be manufactured
noncomputable def additional_parts := N

-- equation representing the time differences
noncomputable def equation := (N / initial_rate) - (N / (initial_rate + increased_speed)) = time_diff

-- state the proof problem
theorem parts_manufactured : initial_parts + additional_parts = 210 :=
by
  -- Use the given conditions to solve the problem
  sorry

end parts_manufactured_l217_217895


namespace proof_problem_l217_217296

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l217_217296


namespace maria_first_stop_distance_is_280_l217_217153

noncomputable def maria_travel_distance : ℝ := 560
noncomputable def first_stop_distance (x : ℝ) : ℝ := x
noncomputable def distance_after_first_stop (x : ℝ) : ℝ := maria_travel_distance - first_stop_distance x
noncomputable def second_stop_distance (x : ℝ) : ℝ := (1 / 4) * distance_after_first_stop x
noncomputable def remaining_distance : ℝ := 210

theorem maria_first_stop_distance_is_280 :
  ∃ x, first_stop_distance x = 280 ∧ second_stop_distance x + remaining_distance = distance_after_first_stop x := sorry

end maria_first_stop_distance_is_280_l217_217153


namespace number_of_new_terms_l217_217079

theorem number_of_new_terms (n : ℕ) (h : n > 1) :
  (2^(n+1) - 1) - (2^n - 1) + 1 = 2^n := by
sorry

end number_of_new_terms_l217_217079


namespace unique_k_largest_n_l217_217385

theorem unique_k_largest_n :
  ∃! k : ℤ, ∃ n : ℕ, (n > 0) ∧ (5 / 18 < n / (n + k) ∧ n / (n + k) < 9 / 17) ∧ (n = 1) :=
by
  sorry

end unique_k_largest_n_l217_217385


namespace trigonometric_simplification_l217_217482

theorem trigonometric_simplification (α : ℝ) :
  (2 * Real.cos α ^ 2 - 1) /
  (2 * Real.tan (π / 4 - α) * Real.sin (π / 4 + α) ^ 2) = 1 :=
sorry

end trigonometric_simplification_l217_217482


namespace remainder_when_587421_divided_by_6_l217_217386

theorem remainder_when_587421_divided_by_6 :
  ¬ (587421 % 2 = 0) → (587421 % 3 = 0) → 587421 % 6 = 3 :=
by sorry

end remainder_when_587421_divided_by_6_l217_217386


namespace milk_tea_sales_l217_217373

-- Definitions
def relationship (x y : ℕ) : Prop := y = 10 * x + 2

-- Theorem statement
theorem milk_tea_sales (x y : ℕ) :
  relationship x y → (y = 822 → x = 82) :=
by
  intros h_rel h_y
  sorry

end milk_tea_sales_l217_217373


namespace meeting_occurs_with_prob_1_div_4_l217_217518

noncomputable def meeting_probability : ℝ := 
  let a := 2^3      -- Total possible volume in hours^3
  let b := 2        -- Volume for the favorable conditions
  b / a

theorem meeting_occurs_with_prob_1_div_4 :
  meeting_probability = 1 / 4 :=
by
  -- Probability calculation as shown in the problem solution
  sorry

end meeting_occurs_with_prob_1_div_4_l217_217518


namespace max_snowmen_constructed_l217_217682

def can_stack (a b : ℝ) : Prop := b >= 2 * a

def max_snowmen (masses : Finset ℝ) : ℝ :=
  let snowballs_high := masses.filter (λ x, x ≥ 50)
  let snowballs_low := masses.filter (λ x, x < 50)
  let n_high := snowballs_high.card
  let n_low := snowballs_low.card
  if n_high <= n_low / 2 then n_high else n_low / 2

theorem max_snowmen_constructed : 
  (masses : Finset ℝ) (h : ∀ n ∈ masses, n ∈ (Finset.range 100).image (λ n, (n + 1 : ℝ)))
  : max_snowmen masses = 24 := 
by
  sorry

end max_snowmen_constructed_l217_217682


namespace factorization_problem_l217_217627

theorem factorization_problem :
  ∃ (a b : ℤ), (25 * x^2 - 130 * x - 120 = (5 * x + a) * (5 * x + b)) ∧ (a + 3 * b = -86) := by
  sorry

end factorization_problem_l217_217627


namespace mean_combined_scores_l217_217350

theorem mean_combined_scores (M A : ℝ) (m a : ℕ) 
  (hM : M = 88) 
  (hA : A = 72) 
  (hm : (m:ℝ) / (a:ℝ) = 2 / 3) :
  (88 * m + 72 * a) / (m + a) = 78 :=
by
  sorry

end mean_combined_scores_l217_217350


namespace pilot_weeks_l217_217334

-- Given conditions
def milesTuesday : ℕ := 1134
def milesThursday : ℕ := 1475
def totalMiles : ℕ := 7827

-- Calculate total miles flown in one week
def milesPerWeek : ℕ := milesTuesday + milesThursday

-- Define the proof problem statement
theorem pilot_weeks (w : ℕ) (h : w * milesPerWeek = totalMiles) : w = 3 :=
by
  -- Here we would provide the proof, but we leave it with a placeholder
  sorry

end pilot_weeks_l217_217334


namespace difference_in_circumferences_l217_217641

theorem difference_in_circumferences (r_inner r_outer : ℝ) (h1 : r_inner = 15) (h2 : r_outer = r_inner + 8) : 
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 16 * Real.pi :=
by
  rw [h1, h2]
  sorry

end difference_in_circumferences_l217_217641


namespace base8_to_base10_l217_217961

theorem base8_to_base10 (n : ℕ) : n = 4 * 8^3 + 3 * 8^2 + 7 * 8^1 + 2 * 8^0 → n = 2298 :=
by 
  sorry

end base8_to_base10_l217_217961


namespace verify_quadratic_eq_l217_217975

def is_quadratic (eq : String) : Prop :=
  eq = "ax^2 + bx + c = 0"

theorem verify_quadratic_eq :
  is_quadratic "x^2 - 1 = 0" :=
by
  -- Auxiliary functions or steps can be introduced if necessary, but proof is omitted here.
  sorry

end verify_quadratic_eq_l217_217975


namespace craftsman_jars_l217_217674

theorem craftsman_jars (J P : ℕ) 
  (h1 : J = 2 * P)
  (h2 : 5 * J + 15 * P = 200) : 
  J = 16 := by
  sorry

end craftsman_jars_l217_217674


namespace lcm_20_45_75_l217_217664

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l217_217664


namespace number_of_bouquets_to_earn_1000_dollars_l217_217275

def cost_of_buying (n : ℕ) : ℕ :=
  n * 20

def revenue_from_selling (m : ℕ) : ℕ :=
  m * 20

def profit_per_operation : ℤ :=
  revenue_from_selling 7 - cost_of_buying 5

theorem number_of_bouquets_to_earn_1000_dollars :
  ∀ bouquets_needed : ℕ, bouquets_needed = 5 * (1000 / profit_per_operation.nat_abs) :=
sorry

end number_of_bouquets_to_earn_1000_dollars_l217_217275


namespace already_installed_windows_l217_217112

-- Definitions based on given conditions
def total_windows : ℕ := 9
def hours_per_window : ℕ := 6
def remaining_hours : ℕ := 18

-- Main statement to prove
theorem already_installed_windows : (total_windows - remaining_hours / hours_per_window) = 6 :=
by
  -- To prove: total_windows - (remaining_hours / hours_per_window) = 6
  -- This step is intentionally left incomplete (proof to be filled in by the user)
  sorry

end already_installed_windows_l217_217112


namespace product_of_solutions_abs_eq_four_l217_217748

theorem product_of_solutions_abs_eq_four :
  (∀ x : ℝ, (|x - 5| - 4 = 0) → (x = 9 ∨ x = 1)) →
  (9 * 1 = 9) :=
by
  intros h
  sorry

end product_of_solutions_abs_eq_four_l217_217748


namespace trajectory_is_plane_l217_217172

/--
Given that the vertical coordinate of a moving point P is always 2, 
prove that the trajectory of the moving point P forms a plane in a 
three-dimensional Cartesian coordinate system.
-/
theorem trajectory_is_plane (P : ℝ × ℝ × ℝ) (hP : ∀ t : ℝ, ∃ x y, P = (x, y, 2)) :
  ∃ a b c d, a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧ (∀ x y, ∃ z, (a * x + b * y + c * z + d = 0) ∧ z = 2) :=
by
  -- This proof should show that there exist constants a, b, c, and d such that 
  -- the given equation represents a plane and the z-coordinate is always 2.
  sorry

end trajectory_is_plane_l217_217172


namespace dan_age_l217_217545

theorem dan_age (D : ℕ) (h : D + 20 = 7 * (D - 4)) : D = 8 :=
by
  sorry

end dan_age_l217_217545


namespace irene_total_income_l217_217598

noncomputable def irene_income (weekly_hours : ℕ) (base_pay : ℕ) (overtime_pay : ℕ) (hours_worked : ℕ) : ℕ :=
  base_pay + (if hours_worked > weekly_hours then (hours_worked - weekly_hours) * overtime_pay else 0)

theorem irene_total_income :
  irene_income 40 500 20 50 = 700 :=
by
  sorry

end irene_total_income_l217_217598


namespace find_least_skilled_painter_l217_217333

-- Define the genders
inductive Gender
| Male
| Female

-- Define the family members
inductive Member
| Grandmother
| Niece
| Nephew
| Granddaughter

-- Define a structure to hold the properties of each family member
structure Properties where
  gender : Gender
  age : Nat
  isTwin : Bool

-- Assume the properties of each family member as given
def grandmother : Properties := { gender := Gender.Female, age := 70, isTwin := false }
def niece : Properties := { gender := Gender.Female, age := 20, isTwin := false }
def nephew : Properties := { gender := Gender.Male, age := 20, isTwin := true }
def granddaughter : Properties := { gender := Gender.Female, age := 20, isTwin := true }

-- Define the best painter
def bestPainter := niece

-- Conditions based on the problem (rephrased to match formalization)
def conditions (least_skilled : Member) : Prop :=
  (bestPainter.gender ≠ (match least_skilled with
                          | Member.Grandmother => grandmother
                          | Member.Niece => niece
                          | Member.Nephew => nephew
                          | Member.Granddaughter => granddaughter ).gender) ∧
  ((match least_skilled with
    | Member.Grandmother => grandmother
    | Member.Niece => niece
    | Member.Nephew => nephew
    | Member.Granddaughter => granddaughter ).isTwin) ∧
  (bestPainter.age = (match least_skilled with
                      | Member.Grandmother => grandmother
                      | Member.Niece => niece
                      | Member.Nephew => nephew
                      | Member.Granddaughter => granddaughter ).age)

-- Statement of the problem
theorem find_least_skilled_painter : ∃ m : Member, conditions m ∧ m = Member.Granddaughter :=
by
  sorry

end find_least_skilled_painter_l217_217333


namespace sum_of_possible_k_l217_217023

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l217_217023


namespace value_of_expression_l217_217168

theorem value_of_expression (m : ℝ) (α : ℝ) (h : m < 0) (h_M : M = (3 * m, -m)) :
  let sin_alpha := -m / (Real.sqrt 10 * -m)
  let cos_alpha := 3 * m / (Real.sqrt 10 * -m)
  (1 / (2 * sin_alpha * cos_alpha + cos_alpha^2) = 10 / 3) :=
by
  sorry

end value_of_expression_l217_217168


namespace extended_fishing_rod_length_l217_217705

def original_length : ℝ := 48
def increase_factor : ℝ := 1.33
def extended_length (orig_len : ℝ) (factor : ℝ) : ℝ := orig_len * factor

theorem extended_fishing_rod_length : extended_length original_length increase_factor = 63.84 :=
  by
    -- proof goes here
    sorry

end extended_fishing_rod_length_l217_217705


namespace correct_weight_of_misread_boy_l217_217944

variable (num_boys : ℕ) (avg_weight_incorrect : ℝ) (misread_weight : ℝ) (avg_weight_correct : ℝ)

theorem correct_weight_of_misread_boy
  (h1 : num_boys = 20)
  (h2 : avg_weight_incorrect = 58.4)
  (h3 : misread_weight = 56)
  (h4 : avg_weight_correct = 58.6) : 
  misread_weight + (num_boys * avg_weight_correct - num_boys * avg_weight_incorrect) / num_boys = 60 := 
by 
  -- skipping proof
  sorry

end correct_weight_of_misread_boy_l217_217944


namespace digit_sum_10_pow_93_minus_937_l217_217176

-- Define a function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem digit_sum_10_pow_93_minus_937 :
  sum_of_digits (10^93 - 937) = 819 :=
by
  sorry

end digit_sum_10_pow_93_minus_937_l217_217176


namespace well_depth_l217_217831

def daily_climb_up : ℕ := 4
def daily_slip_down : ℕ := 3
def total_days : ℕ := 27

theorem well_depth : (daily_climb_up * (total_days - 1) - daily_slip_down * (total_days - 1)) + daily_climb_up = 30 := by
  -- conditions
  let net_daily_progress := daily_climb_up - daily_slip_down
  let net_26_days_progress := net_daily_progress * (total_days - 1)

  -- proof to be completed
  sorry

end well_depth_l217_217831


namespace determinant_scaled_l217_217425

variables (x y z w : ℝ)
variables (det : ℝ)

-- Given condition: determinant of the 2x2 matrix is 7.
axiom det_given : det = x * w - y * z
axiom det_value : det = 7

-- The target to be proven: the determinant of the scaled matrix is 63.
theorem determinant_scaled (x y z w : ℝ) (det : ℝ) (h_det : det = x * w - y * z) (det_value : det = 7) : 
  3 * 3 * (x * w - y * z) = 63 :=
by
  sorry

end determinant_scaled_l217_217425


namespace sixty_three_times_fifty_seven_l217_217141

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 :=
by
  let a := 60
  let b := 3
  have h : (a + b) * (a - b) = a^2 - b^2 := by sorry
  have h1 : 63 = a + b := by rfl
  have h2 : 57 = a - b := by rfl
  calc
    63 * 57 = (a + b) * (a - b) : by rw [h1, h2]
    ... = a^2 - b^2 : by rw h
    ... = 60^2 - 3^2 : by rfl
    ... = 3600 - 9 : by sorry
    ... = 3591 : by norm_num

end sixty_three_times_fifty_seven_l217_217141


namespace no_solution_A_eq_B_l217_217007

theorem no_solution_A_eq_B (a : ℝ) (h1 : a = 2 * a) (h2 : a ≠ 2) : false := by
  sorry

end no_solution_A_eq_B_l217_217007


namespace ruby_candies_l217_217923

theorem ruby_candies (number_of_friends : ℕ) (candies_per_friend : ℕ) (total_candies : ℕ)
  (h1 : number_of_friends = 9)
  (h2 : candies_per_friend = 4)
  (h3 : total_candies = number_of_friends * candies_per_friend) :
  total_candies = 36 :=
by {
  sorry
}

end ruby_candies_l217_217923


namespace cos_540_eq_neg1_l217_217670

theorem cos_540_eq_neg1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end cos_540_eq_neg1_l217_217670


namespace ineq_condition_l217_217818

theorem ineq_condition (a b : ℝ) : (a + 1 > b - 2) ↔ (a > b - 3 ∧ ¬(a > b)) :=
by
  sorry

end ineq_condition_l217_217818


namespace molecular_weight_of_compound_l217_217970

-- Definitions of the atomic weights.
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

-- Proof statement of the molecular weight of the compound.
theorem molecular_weight_of_compound :
  (1 * atomic_weight_K) + (1 * atomic_weight_Br) + (3 * atomic_weight_O) = 167.00 :=
  by
    sorry

end molecular_weight_of_compound_l217_217970


namespace number_of_plans_for_participation_l217_217357

open Finset

/-- Proof that there are 18 different plans for participation given the conditions. -/
theorem number_of_plans_for_participation :
  let students := {"A", "B", "C", "D"} in
  let must_participate := "A" in
  let remaining_students := erase students must_participate in
  (card (choose 2 remaining_students) * factorial 3) = 18 := sorry

end number_of_plans_for_participation_l217_217357


namespace find_a_l217_217449

open Complex

theorem find_a (a : ℝ) (h : (⟨a, 1⟩ * ⟨1, -a⟩ = 2)) : a = 1 :=
sorry

end find_a_l217_217449


namespace number_of_workers_is_25_l217_217992

noncomputable def original_workers (W : ℕ) :=
  W * 35 = (W + 10) * 25

theorem number_of_workers_is_25 : ∃ W, original_workers W ∧ W = 25 :=
by
  use 25
  unfold original_workers
  sorry

end number_of_workers_is_25_l217_217992


namespace true_compound_propositions_l217_217290

-- Define the propositions
def p1 : Prop := 
  ∀ (l₁ l₂ l₃ : Line), 
    (l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
    (∃ (p₁ p₂ p₃ : Point),
      l₁.contains p₁ ∧ l₁.contains p₂ ∧
      l₂.contains p₂ ∧ l₂.contains p₃ ∧
      l₃.contains p₃ ∧ l₃.contains p₁)
    → ∃ (α : Plane), 
      α.contains l₁ ∧ α.contains l₂ ∧ α.contains l₃

def p2 : Prop :=
  ∀ (a b c : Point),
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    → ∃! (α : Plane), 
      α.contains a ∧ α.contains b ∧ α.contains c

def p3 : Prop :=
  ∀ (l₁ l₂ : Line),
    (¬ ∃ (p : Point), l₁.contains p ∧ l₂.contains p)
    → l₁.parallel l₂

def p4 : Prop :=
  ∀ (α : Plane) (l m : Line),
    (α.contains l ∧ ¬ α.contains m ∧ m.perpendicular α) 
    → m.perpendicular l

-- Main theorem that identifies the true compound propositions
theorem true_compound_propositions :
  (p1 ∧ p4) ∧ ¬ (p1 ∧ p2) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) :=
by
  -- Proof might involve defining lines, points, and planes and their relationships
  sorry

end true_compound_propositions_l217_217290


namespace least_common_multiple_of_20_45_75_l217_217648

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l217_217648


namespace perpendicular_vectors_m_val_l217_217179

theorem perpendicular_vectors_m_val (m : ℝ) 
  (a : ℝ × ℝ := (-1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  m = 2 := 
by 
  sorry

end perpendicular_vectors_m_val_l217_217179


namespace valentines_given_l217_217475

theorem valentines_given (x y : ℕ) (h : x * y = x + y + 40) : x * y = 84 :=
by
  -- solving for x, y based on the factors of 41
  sorry

end valentines_given_l217_217475


namespace seven_k_plus_four_l217_217461

theorem seven_k_plus_four (k m n : ℕ) (h1 : 4 * k + 5 = m^2) (h2 : 9 * k + 4 = n^2) (hk : k = 5) : 
  7 * k + 4 = 39 :=
by 
  -- assume conditions
  have h1' := h1
  have h2' := h2
  have hk' := hk
  sorry

end seven_k_plus_four_l217_217461


namespace triangle_property_l217_217002

theorem triangle_property
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (hR : R = Real.sqrt 3)
  (h1 : a * Real.sin C + Real.sqrt 3 * c * Real.cos A = 0)
  (h2 : b + c = Real.sqrt 11)
  (htri : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R):
  a = 3 ∧ (1 / 2 * b * c * Real.sin A = Real.sqrt 3 / 2) := 
sorry

end triangle_property_l217_217002


namespace find_number_of_children_l217_217528

def admission_cost_adult : ℝ := 30
def admission_cost_child : ℝ := 15
def total_people : ℕ := 10
def soda_cost : ℝ := 5
def discount_rate : ℝ := 0.8
def total_paid : ℝ := 197

def total_cost_with_discount (adults children : ℕ) : ℝ :=
  discount_rate * (adults * admission_cost_adult + children * admission_cost_child)

theorem find_number_of_children (A C : ℕ) 
  (h1 : A + C = total_people)
  (h2 : total_cost_with_discount A C + soda_cost = total_paid) :
  C = 4 :=
sorry

end find_number_of_children_l217_217528


namespace solve_x_eq_10000_l217_217359

theorem solve_x_eq_10000 (x : ℝ) (h : 5 * x^(1/4 : ℝ) - 3 * (x / x^(3/4 : ℝ)) = 10 + x^(1/4 : ℝ)) : x = 10000 :=
by
  sorry

end solve_x_eq_10000_l217_217359


namespace min_ab_l217_217310

theorem min_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b - a * b + 3 = 0) : 
  9 ≤ a * b :=
sorry

end min_ab_l217_217310


namespace bob_can_order_199_sandwiches_l217_217362

-- Define the types of bread, meat, and cheese
def number_of_bread : ℕ := 5
def number_of_meat : ℕ := 7
def number_of_cheese : ℕ := 6

-- Define the forbidden combinations
def forbidden_turkey_swiss : ℕ := number_of_bread -- 5
def forbidden_rye_roastbeef : ℕ := number_of_cheese -- 6

-- Calculate the total sandwiches and subtract forbidden combinations
def total_sandwiches : ℕ := number_of_bread * number_of_meat * number_of_cheese
def forbidden_sandwiches : ℕ := forbidden_turkey_swiss + forbidden_rye_roastbeef

def sandwiches_bob_can_order : ℕ := total_sandwiches - forbidden_sandwiches

theorem bob_can_order_199_sandwiches :
  sandwiches_bob_can_order = 199 :=
by
  -- The calculation steps are encapsulated in definitions and are considered done
  sorry

end bob_can_order_199_sandwiches_l217_217362


namespace max_distance_from_origin_to_line_l217_217175

variable (k : ℝ)

def line (x y : ℝ) : Prop := k * x + y + 1 = 0

theorem max_distance_from_origin_to_line :
  ∃ k : ℝ, ∀ x y : ℝ, line k x y -> dist (0, 0) (x, y) ≤ 1 := 
sorry

end max_distance_from_origin_to_line_l217_217175


namespace LCM_20_45_75_is_900_l217_217657

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l217_217657


namespace range_of_a_l217_217877

noncomputable def inequality_always_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (a : ℝ) : inequality_always_holds a ↔ 0 ≤ a ∧ a < 1 := 
by
  sorry

end range_of_a_l217_217877


namespace divisible_by_1989_l217_217214

theorem divisible_by_1989 (n : ℕ) : 
  1989 ∣ (13 * (-50)^n + 17 * 40^n - 30) :=
by
  sorry

end divisible_by_1989_l217_217214


namespace number_of_possible_b2_values_l217_217400

open Nat

theorem number_of_possible_b2_values (b : ℕ → ℕ) 
  (h_seq : ∀ n, b (n + 2) = abs (b (n + 1) - b n))
  (h_b1 : b 1 = 1024)
  (h_b2_lt_1024 : b 2 < 1024)
  (h_b1004 : b 1004 = 1) : 
  { k : ℕ | k < 1024 ∧ gcd 1024 k = 1 ∧ odd k }.card = 512 := sorry

end number_of_possible_b2_values_l217_217400


namespace find_point_C_l217_217697

theorem find_point_C :
  ∃ C : ℝ × ℝ, let A : ℝ × ℝ := (-3, 5) in
                 let B : ℝ × ℝ := (9, -1) in
                 let AB := (B.1 - A.1, B.2 - A.2) in
                 C = (B.1 + 0.5 * AB.1, B.2 + 0.5 * AB.2) ∧ 
                 C = (15, -4) :=
by
  sorry

end find_point_C_l217_217697


namespace balance_three_diamonds_l217_217559

-- Define the problem conditions
variables (a b c : ℕ)

-- Four Δ's and two ♦'s will balance twelve ●'s
def condition1 : Prop :=
  4 * a + 2 * b = 12 * c

-- One Δ will balance a ♦ and two ●'s
def condition2 : Prop :=
  a = b + 2 * c

-- Theorem to prove how many ●'s will balance three ♦'s
theorem balance_three_diamonds (h1 : condition1 a b c) (h2 : condition2 a b c) : 3 * b = 2 * c :=
by sorry

end balance_three_diamonds_l217_217559


namespace solution_set_inequality_l217_217955

   theorem solution_set_inequality (a : ℝ) : (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) :=
   sorry
   
end solution_set_inequality_l217_217955


namespace arithmetic_sequence_properties_l217_217315

theorem arithmetic_sequence_properties (a : ℕ → ℤ) (T : ℕ → ℤ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) (h2 : a 4 = a 2 + 4) (h3 : a 3 = 6) :
  (∀ n, a n = 2 * n) ∧ (∀ n, T n = (4 / 3 * (4^n - 1))) :=
by
  sorry

end arithmetic_sequence_properties_l217_217315


namespace stability_of_scores_requires_variance_l217_217495

-- Define the conditions
variable (scores : List ℝ)

-- Define the main theorem
theorem stability_of_scores_requires_variance : True :=
  sorry

end stability_of_scores_requires_variance_l217_217495


namespace triangle_AC_range_l217_217750

noncomputable def length_AB : ℝ := 12
noncomputable def length_CD : ℝ := 6

def is_valid_AC (AC : ℝ) : Prop :=
  AC > 6 ∧ AC < 24

theorem triangle_AC_range :
  ∃ m n : ℝ, 
    (6 < m ∧ m < 24) ∧ (6 < n ∧ n < 24) ∧
    m + n = 30 ∧
    ∀ AC : ℝ, is_valid_AC AC →
      6 < AC ∧ AC < 24 :=
by
  use 6
  use 24
  simp
  sorry

end triangle_AC_range_l217_217750


namespace shaded_area_is_10_l217_217498

-- Definitions based on conditions:
def rectangle_area : ℕ := 12
def unshaded_triangle_area : ℕ := 2

-- Proof statement without the actual proof.
theorem shaded_area_is_10 : rectangle_area - unshaded_triangle_area = 10 := by
  sorry

end shaded_area_is_10_l217_217498


namespace proof_problem_l217_217506

-- Definitions based on the conditions from the problem
def optionA (A : Set α) : Prop := ∅ ∩ A = ∅

def optionC : Prop := { y | ∃ x, y = 1 / x } = { z | ∃ t, z = 1 / t }

-- The main theorem statement
theorem proof_problem (A : Set α) : optionA A ∧ optionC := by
  -- Placeholder for the proof
  sorry

end proof_problem_l217_217506


namespace unique_solution_count_l217_217184

theorem unique_solution_count : 
  ∃! (a : ℝ), ∀ {x : ℝ}, (0 < x) → 
    4 * a ^ 2 + 3 * x * log x + 3 * (log x)^2 = 13 * a * log x + a * x :=
sorry

end unique_solution_count_l217_217184


namespace point_same_side_of_line_l217_217548

def same_side (p₁ p₂ : ℝ × ℝ) (a b c : ℝ) : Prop :=
  (a * p₁.1 + b * p₁.2 + c > 0) ↔ (a * p₂.1 + b * p₂.2 + c > 0)

theorem point_same_side_of_line :
  same_side (1, 2) (1, 0) 2 (-1) 1 :=
by
  unfold same_side
  sorry

end point_same_side_of_line_l217_217548


namespace number_of_friends_l217_217356

def has14_pokemon_cards (x : String) : Prop :=
  x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith"

theorem number_of_friends :
  ∃ n, n = 4 ∧
        ∀ x, has14_pokemon_cards x ↔ x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith" :=
by
  sorry

end number_of_friends_l217_217356


namespace sum_of_consecutive_integers_345_l217_217455

-- Definition of the conditions
def is_consecutive_sum (n : ℕ) (k : ℕ) (s : ℕ) : Prop :=
  s = k * n + k * (k - 1) / 2

-- Problem statement
theorem sum_of_consecutive_integers_345 :
  ∃ k_set : Finset ℕ, (∀ k ∈ k_set, k ≥ 2 ∧ ∃ n : ℕ, is_consecutive_sum n k 345) ∧ k_set.card = 6 :=
sorry

end sum_of_consecutive_integers_345_l217_217455


namespace maximum_snowmen_count_l217_217687

-- Define the mass range
def masses : List ℕ := List.range' 1 99

-- Define the condition function for mass stacking
def can_stack (a b : ℕ) : Prop := a ≥ 2 * b

-- Define the main theorem that we need to prove
theorem maximum_snowmen_count (snowballs : List ℕ) (h1 : snowballs = masses) :
  ∃ m, m = 24 ∧ (∀ snowmen, (∀ s ∈ snowmen, s.length = 3 ∧
                                 (∀ i j, i < j → can_stack (s.get? i).get_or_else 0 (s.get? j).get_or_else 0)) →
                                snowmen.length ≤ m) :=
sorry

end maximum_snowmen_count_l217_217687


namespace number_of_students_l217_217017

variables (T S n : ℕ)

-- 1. The teacher's age is 24 years more than the average age of the students.
def condition1 : Prop := T = S / n + 24

-- 2. The teacher's age is 20 years more than the average age of everyone present.
def condition2 : Prop := T = (T + S) / (n + 1) + 20

-- Proving that the number of students in the classroom is 5 given the conditions.
theorem number_of_students (h1 : condition1 T S n) (h2 : condition2 T S n) : n = 5 :=
by sorry

end number_of_students_l217_217017


namespace lcm_20_45_75_l217_217665

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l217_217665


namespace jar_initial_water_fraction_l217_217602

theorem jar_initial_water_fraction (C W : ℝ) (hC : C > 0) (hW : W + C / 4 = 0.75 * C) : W / C = 0.5 :=
by
  -- necessary parameters and sorry for the proof 
  sorry

end jar_initial_water_fraction_l217_217602


namespace total_raining_time_correct_l217_217462

-- Define individual durations based on given conditions
def duration_day1 : ℕ := 10        -- 17:00 - 07:00 = 10 hours
def duration_day2 : ℕ := duration_day1 + 2    -- Second day: 10 hours + 2 hours = 12 hours
def duration_day3 : ℕ := duration_day2 * 2    -- Third day: 12 hours * 2 = 24 hours

-- Define the total raining time over three days
def total_raining_time : ℕ := duration_day1 + duration_day2 + duration_day3

-- Formally state the theorem to prove the total rain time is 46 hours
theorem total_raining_time_correct : total_raining_time = 46 := by
  sorry

end total_raining_time_correct_l217_217462


namespace sum_possible_values_k_l217_217034

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l217_217034


namespace probability_of_10_heads_in_12_flips_l217_217082

open_locale big_operators

noncomputable def calculate_probability : ℕ → ℕ → ℚ := 
  λ n k, (nat.choose n k : ℚ) / (2 ^ n)

theorem probability_of_10_heads_in_12_flips :
  calculate_probability 12 10 = 66 / 4096 :=
by
  sorry

end probability_of_10_heads_in_12_flips_l217_217082


namespace solve_for_y_l217_217940

theorem solve_for_y (y : ℚ) : y - 1 / 2 = 1 / 6 - 2 / 3 + 1 / 4 → y = 1 / 4 := by
  intro h
  sorry

end solve_for_y_l217_217940


namespace gold_bars_total_worth_l217_217053

theorem gold_bars_total_worth :
  let rows := 4
  let bars_per_row := 20
  let worth_per_bar : ℕ := 20000
  let total_bars := rows * bars_per_row
  let total_worth := total_bars * worth_per_bar
  total_worth = 1600000 :=
by
  sorry

end gold_bars_total_worth_l217_217053


namespace no_solution_exists_l217_217743

theorem no_solution_exists :
  ¬ ∃ m n : ℕ, 
    m + n = 2009 ∧ 
    (m * (m - 1) + n * (n - 1) = 2009 * 2008 / 2) := by
  sorry

end no_solution_exists_l217_217743


namespace wire_length_l217_217835

theorem wire_length (S L : ℝ) (h1 : S = 10) (h2 : S = (2 / 5) * L) : S + L = 35 :=
by
  sorry

end wire_length_l217_217835


namespace area_new_rectangle_greater_than_square_l217_217399

theorem area_new_rectangle_greater_than_square (a b : ℝ) (h : a > b) : 
  (2 * (a + b) * (2 * b + a) / 3) > ((a + b) * (a + b)) := 
sorry

end area_new_rectangle_greater_than_square_l217_217399


namespace percentage_drop_l217_217103

theorem percentage_drop (P N P' N' : ℝ) (h1 : N' = 1.60 * N) (h2 : P' * N' = 1.2800000000000003 * (P * N)) :
  P' = 0.80 * P :=
by
  sorry

end percentage_drop_l217_217103


namespace no_integer_polynomial_exists_l217_217479

theorem no_integer_polynomial_exists 
    (a b c d : ℤ) (h : a ≠ 0) (P : ℤ → ℤ) 
    (h1 : ∀ x, P x = a * x ^ 3 + b * x ^ 2 + c * x + d)
    (h2 : P 4 = 1) (h3 : P 7 = 2) : 
    false := 
by
    sorry

end no_integer_polynomial_exists_l217_217479


namespace problem_I_problem_II_l217_217436

-- Problem (I)
theorem problem_I (a b : ℝ) (h1 : a = 1) (h2 : b = 1) :
  { x : ℝ | |2*x + a| + |2*x - 2*b| + 3 > 8 } = 
  { x : ℝ | x < -1 ∨ x > 1.5 } := by
  sorry

-- Problem (II)
theorem problem_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ∀ x : ℝ, |2*x + a| + |2*x - 2*b| + 3 ≥ 5) :
  (1 / a + 1 / b) = (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end problem_I_problem_II_l217_217436


namespace bill_needs_125_bouquets_to_earn_1000_l217_217274

-- Define the constants for the problem
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_profit : ℕ := 1000

-- Define the problem in terms of a theorem
theorem bill_needs_125_bouquets_to_earn_1000 :
  ∃ n : ℕ, (35 / roses_per_bouquet_sell) * cost_per_bouquet - (5 * cost_per_bouquet) = 40 → (5 * n) = 125 :=
begin
  sorry
end

end bill_needs_125_bouquets_to_earn_1000_l217_217274


namespace total_area_of_squares_l217_217799

-- Condition 1: Definition of the side length
def side_length (s : ℝ) : Prop := s = 12

-- Condition 2: Definition of the center of one square coinciding with the vertex of another
-- Here, we assume the positions are fixed so this condition is given
def coincide_center_vertex (s₁ s₂ : ℝ) : Prop := s₁ = s₂ 

-- The main theorem statement
theorem total_area_of_squares
  (s₁ s₂ : ℝ) 
  (h₁ : side_length s₁)
  (h₂ : side_length s₂)
  (h₃ : coincide_center_vertex s₁ s₂) :
  (2 * s₁^2) - (s₁^2 / 4) = 252 :=
by
  sorry

end total_area_of_squares_l217_217799


namespace least_common_multiple_of_20_45_75_l217_217647

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l217_217647


namespace simplify_sqrt_expression_l217_217925

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l217_217925


namespace find_xyz_l217_217045

theorem find_xyz
  (a b c x y z : ℂ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0)
  (h7 : a = (b + c) / (x - 3))
  (h8 : b = (a + c) / (y - 3))
  (h9 : c = (a + b) / (z - 3))
  (h10 : x * y + x * z + y * z = 10)
  (h11 : x + y + z = 6) :
  x * y * z = 10 :=
sorry

end find_xyz_l217_217045


namespace rectangle_perimeter_divided_into_six_congruent_l217_217262

theorem rectangle_perimeter_divided_into_six_congruent (l w : ℕ) (h1 : 2 * (w + l / 6) = 40) (h2 : l = 120 - 6 * w) : 
  2 * (l + w) = 280 :=
by
  sorry

end rectangle_perimeter_divided_into_six_congruent_l217_217262


namespace geometric_and_arithmetic_sequence_solution_l217_217453

theorem geometric_and_arithmetic_sequence_solution:
  ∃ a b : ℝ, 
    (a > 0) ∧                  -- a is positive
    (∃ r : ℝ, 10 * r = a ∧ a * r = 1 / 2) ∧   -- geometric sequence condition
    (∃ d : ℝ, a + d = 5 ∧ 5 + d = b) ∧        -- arithmetic sequence condition
    a = Real.sqrt 5 ∧
    b = 10 - Real.sqrt 5 := 
by 
  sorry

end geometric_and_arithmetic_sequence_solution_l217_217453


namespace x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l217_217446

theorem x_plus_inv_x_eq_two_implies_x_pow_six_eq_one
  (x : ℝ) (h : x + 1/x = 2) : x^6 = 1 :=
sorry

end x_plus_inv_x_eq_two_implies_x_pow_six_eq_one_l217_217446


namespace product_of_possible_lengths_approx_l217_217967

noncomputable def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a * a + b * b)

noncomputable def other_leg (hypotenuse a : ℝ) : ℝ :=
  real.sqrt (hypotenuse * hypotenuse - a * a)

noncomputable def product_of_possible_lengths (a b : ℝ) : ℝ :=
  hypotenuse a b * other_leg (max a b) (min a b)

theorem product_of_possible_lengths_approx (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  Float.round (product_of_possible_lengths a b) 1 = 52.7 :=
by
  sorry

end product_of_possible_lengths_approx_l217_217967


namespace time_to_cross_bridge_l217_217401

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 150
def speed_in_kmhr : ℕ := 72
def speed_in_ms : ℕ := (speed_in_kmhr * 1000) / 3600

theorem time_to_cross_bridge : 
  (length_of_train + length_of_bridge) / speed_in_ms = 20 :=
by
  have total_distance := length_of_train + length_of_bridge
  have speed := speed_in_ms
  sorry

end time_to_cross_bridge_l217_217401


namespace total_time_proof_l217_217919

variable (mow_time : ℕ) (fertilize_time : ℕ) (total_time : ℕ)

-- Based on the problem conditions.
axiom mow_time_def : mow_time = 40
axiom fertilize_time_def : fertilize_time = 2 * mow_time
axiom total_time_def : total_time = mow_time + fertilize_time

-- The proof goal
theorem total_time_proof : total_time = 120 := by
  sorry

end total_time_proof_l217_217919


namespace square_area_ratio_l217_217629

theorem square_area_ratio (a b : ℕ) (h : 4 * a = 4 * (4 * b)) : (a^2) = 16 * (b^2) := 
by sorry

end square_area_ratio_l217_217629


namespace Victor_bought_6_decks_l217_217501

theorem Victor_bought_6_decks (V : ℕ) (h1 : 2 * 8 + 8 * V = 64) : V = 6 := by
  sorry

end Victor_bought_6_decks_l217_217501


namespace solution_set_l217_217635

theorem solution_set (x : ℝ) : (2 : ℝ) ^ (|x-2| + |x-4|) > 2^6 ↔ x < 0 ∨ x > 6 :=
by
  sorry

end solution_set_l217_217635


namespace no_polynomial_transform_l217_217338

theorem no_polynomial_transform (a b c : ℚ) :
  ¬ (∀ (x y : ℚ),
      ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 10) ∨ (x = 7 ∧ y = 7)) →
      a * x^2 + b * x + c = y) :=
by
  sorry

end no_polynomial_transform_l217_217338


namespace triangle_side_b_l217_217332

theorem triangle_side_b (A B C a b c : ℝ)
  (hA : A = 135)
  (hc : c = 1)
  (hSinB_SinC : Real.sin B * Real.sin C = Real.sqrt 2 / 10) :
  b = Real.sqrt 2 ∨ b = Real.sqrt 2 / 2 :=
by
  sorry

end triangle_side_b_l217_217332


namespace cost_of_ticket_when_Matty_was_born_l217_217505

theorem cost_of_ticket_when_Matty_was_born 
    (cost : ℕ → ℕ) 
    (h_halved : ∀ t : ℕ, cost (t + 10) = cost t / 2) 
    (h_age_30 : cost 30 = 125000) : 
    cost 0 = 1000000 := 
by 
  sorry

end cost_of_ticket_when_Matty_was_born_l217_217505


namespace distance_between_x_intercepts_l217_217830

theorem distance_between_x_intercepts :
  ∀ (x1 x2 : ℝ),
  (∀ x, x1 = 8 → x2 = 20 → 20 = 4 * (x - 8)) → 
  (∀ x, x1 = 8 → x2 = 20 → 20 = 7 * (x - 8)) → 
  abs ((3 : ℝ) - (36 / 7)) = (15 / 7) :=
by
  intros x1 x2 h1 h2
  sorry

end distance_between_x_intercepts_l217_217830


namespace cubical_tank_water_volume_l217_217675

theorem cubical_tank_water_volume 
    (s : ℝ) -- side length of the cube in feet
    (h_fill : 1 / 4 * s = 1) -- tank is filled to 0.25 of its capacity, water level is 1 foot
    (h_volume_water : 0.25 * (s ^ 3) = 16) -- 0.25 of the tank's total volume is the volume of water
    : s ^ 3 = 64 := 
by
  sorry

end cubical_tank_water_volume_l217_217675


namespace num_of_friends_l217_217355

theorem num_of_friends :
  ∃ friends : Finset String, 
  friends = {"Sam", "Dan", "Tom", "Keith"} ∧
  friends.card = 4 :=
begin
  sorry
end

end num_of_friends_l217_217355


namespace least_value_a_plus_b_l217_217325

theorem least_value_a_plus_b (a b : ℕ) (h : 20 / 19 = 1 + 1 / (1 + a / b)) : a + b = 19 :=
sorry

end least_value_a_plus_b_l217_217325


namespace sum_of_squares_of_roots_l217_217953

theorem sum_of_squares_of_roots :
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂) →
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + x₂^2 = 79 / 25) :=
by
  sorry

end sum_of_squares_of_roots_l217_217953


namespace coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l217_217546

noncomputable def coefficient_of_x4_expansion : ℕ :=
  let r := 2;
  let n := 5;
  let general_term_coefficient := Nat.choose n r * 2^(n-r);
  general_term_coefficient

theorem coefficient_of_x4_in_expansion_of_2x_plus_sqrtx :
  coefficient_of_x4_expansion = 80 :=
by
  -- We can bypass the actual proving steps by
  -- acknowledging that the necessary proof mechanism
  -- will properly verify the calculation:
  sorry

end coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l217_217546


namespace minimize_base_side_length_l217_217680

theorem minimize_base_side_length (V : ℝ) (a h : ℝ) 
  (volume_eq : V = a ^ 2 * h) (V_given : V = 256) (h_eq : h = 256 / (a ^ 2)) :
  a = 8 :=
by
  -- Recognize that for a given volume, making it a cube minimizes the surface area.
  -- As the volume of the cube a^3 = 256, solving for a gives 8.
  -- a := (256:ℝ) ^ (1/3:ℝ)
  sorry

end minimize_base_side_length_l217_217680


namespace projectile_reaches_64_first_time_l217_217365

theorem projectile_reaches_64_first_time :
  ∃ t : ℝ, t > 0 ∧ t ≈ 0.7 ∧ (-16 * t^2 + 100 * t = 64) :=
sorry

end projectile_reaches_64_first_time_l217_217365


namespace proof_negation_l217_217783

-- Definitions of rational and real numbers
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- Proposition stating the existence of an irrational number that is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational x

-- Negation of the original proposition
def negated_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬ is_rational x

theorem proof_negation : ¬ original_proposition = negated_proposition := 
sorry

end proof_negation_l217_217783


namespace sam_memorized_digits_l217_217216

theorem sam_memorized_digits (c s m : ℕ) 
  (h1 : s = c + 6) 
  (h2 : m = 6 * c)
  (h3 : m = 24) : 
  s = 10 :=
by
  sorry

end sam_memorized_digits_l217_217216


namespace largest_even_number_l217_217191

theorem largest_even_number (x : ℤ) (h1 : 3 * x + 6 = (x + (x + 2) + (x + 4)) / 3 + 44) : 
  x + 4 = 24 := 
by 
  sorry

end largest_even_number_l217_217191


namespace average_of_all_digits_l217_217958

theorem average_of_all_digits (d : List ℕ) (h_len : d.length = 9)
  (h1 : (d.take 4).sum = 32)
  (h2 : (d.drop 4).sum = 130) : 
  (d.sum / d.length : ℚ) = 18 := 
by
  sorry

end average_of_all_digits_l217_217958


namespace square_side_length_l217_217327

/-- If the area of a square is 9m^2 + 24mn + 16n^2, then the length of the side of the square is |3m + 4n|. -/
theorem square_side_length (m n : ℝ) (a : ℝ) (h : a^2 = 9 * m^2 + 24 * m * n + 16 * n^2) : a = |3 * m + 4 * n| :=
sorry

end square_side_length_l217_217327


namespace new_weight_is_77_l217_217488

theorem new_weight_is_77 (weight_increase_per_person : ℝ) (number_of_persons : ℕ) (old_weight : ℝ) 
  (total_weight_increase : ℝ) (new_weight : ℝ) 
  (h1 : weight_increase_per_person = 1.5)
  (h2 : number_of_persons = 8)
  (h3 : old_weight = 65)
  (h4 : total_weight_increase = number_of_persons * weight_increase_per_person)
  (h5 : new_weight = old_weight + total_weight_increase) :
  new_weight = 77 :=
sorry

end new_weight_is_77_l217_217488


namespace lcm_20_45_75_l217_217662

def lcm (a b : ℕ) : ℕ := nat.lcm a b

theorem lcm_20_45_75 : lcm (lcm 20 45) 75 = 900 :=
by
  sorry

end lcm_20_45_75_l217_217662


namespace bie_l217_217335

noncomputable def surface_area_of_sphere (PA AB AC : ℝ) (hPA_AB : PA = AB) (hPA : PA = 2) (hAC : AC = 4) (r : ℝ) : ℝ :=
  let PC := Real.sqrt (PA ^ 2 + AC ^ 2)
  let radius := PC / 2
  4 * Real.pi * radius ^ 2

theorem bie'zhi_tetrahedron_surface_area
  (PA AB AC : ℝ)
  (hPA_AB : PA = AB)
  (hPA : PA = 2)
  (hAC : AC = 4)
  (PC : ℝ := Real.sqrt (PA ^ 2 + AC ^ 2))
  (r : ℝ := PC / 2)
  (surface_area : ℝ := 4 * Real.pi * r ^ 2)
  :
  surface_area = 20 * Real.pi := 
sorry

end bie_l217_217335


namespace find_b_l217_217491

-- Variables representing the terms in the equations
variables (a b t : ℝ)

-- Conditions given in the problem
def cond1 : Prop := a - (t / 6) * b = 20
def cond2 : Prop := a - (t / 5) * b = -10
def t_value : Prop := t = 60

-- The theorem we need to prove
theorem find_b (H1 : cond1 a b t) (H2 : cond2 a b t) (H3 : t_value t) : b = 15 :=
by {
  -- Assuming the conditions are true
  sorry
}

end find_b_l217_217491


namespace vacation_cost_per_person_l217_217347

theorem vacation_cost_per_person (airbnb_cost car_cost : ℝ) (num_people : ℝ) 
  (h1 : airbnb_cost = 3200) (h2 : car_cost = 800) (h3 : num_people = 8) : 
  (airbnb_cost + car_cost) / num_people = 500 := 
by 
  sorry

end vacation_cost_per_person_l217_217347


namespace find_b_l217_217637

theorem find_b (b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 + x2 = -2) ∧
    ((x1 + 1)^3 + x1 / (x1 + 1) = -x1 + b) ∧
    ((x2 + 1)^3 + x2 / (x2 + 1) = -x2 + b)) →
  b = 0 :=
by
  sorry

end find_b_l217_217637


namespace no_integer_solution_mx2_minus_sy2_eq_3_l217_217353

theorem no_integer_solution_mx2_minus_sy2_eq_3 (m s : ℤ) (x y : ℤ) (h : m * s = 2000 ^ 2001) :
  ¬ (m * x ^ 2 - s * y ^ 2 = 3) :=
sorry

end no_integer_solution_mx2_minus_sy2_eq_3_l217_217353


namespace fraction_of_water_l217_217823

theorem fraction_of_water (total_weight sand_ratio water_weight gravel_weight : ℝ)
  (htotal : total_weight = 49.99999999999999)
  (hsand_ratio : sand_ratio = 1/2)
  (hwater : water_weight = total_weight - total_weight * sand_ratio - gravel_weight)
  (hgravel : gravel_weight = 15)
  : (water_weight / total_weight) = 1/5 :=
by
  sorry

end fraction_of_water_l217_217823


namespace proposition_1_proposition_2_proposition_3_proposition_4_l217_217293

axiom p1 : Prop
axiom p2 : Prop
axiom p3 : Prop
axiom p4 : Prop

axiom p1_true : p1 = true
axiom p2_false : p2 = false
axiom p3_false : p3 = false
axiom p4_true : p4 = true

theorem proposition_1 : (p1 ∧ p4) = true := by sorry
theorem proposition_2 : (p1 ∧ p2) = false := by sorry
theorem proposition_3 : (¬p2 ∨ p3) = true := by sorry
theorem proposition_4 : (¬p3 ∨ ¬p4) = true := by sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l217_217293


namespace expand_polynomial_l217_217556

open Polynomial

noncomputable def expand_p1 : Polynomial ℚ := Polynomial.monomial 3 4 - Polynomial.monomial 2 3 + Polynomial.monomial 1 2 + Polynomial.C 7

noncomputable def expand_p2 : Polynomial ℚ := Polynomial.monomial 4 5 + Polynomial.monomial 3 1 - Polynomial.monomial 1 3 + Polynomial.C 9

theorem expand_polynomial :
  (expand_p1 * expand_p2) = Polynomial.monomial 7 20 - Polynomial.monomial 5 27 + Polynomial.monomial 4 8 + Polynomial.monomial 3 45 - Polynomial.monomial 2 4 + Polynomial.monomial 1 51 + Polynomial.C 196 :=
by {
  sorry
}

end expand_polynomial_l217_217556


namespace max_value_of_x_plus_y_l217_217428

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop := 
  x > 0 ∧ y > 0 ∧ x + y + (1/x) + (1/y) = 5

-- Theorem statement
theorem max_value_of_x_plus_y (x y : ℝ) (h : conditions x y) : x + y ≤ 4 := 
sorry

end max_value_of_x_plus_y_l217_217428


namespace bound_riemann_sum_difference_l217_217313

theorem bound_riemann_sum_difference (f : ℝ → ℝ) (k : ℝ) (n : ℕ) (hn : 0 < n) 
  (hf : ∀ x ∈ Ioo (0 : ℝ) 1, differentiable_at ℝ f x) 
  (h_bound : ∀ x ∈ Ioo (0 : ℝ) 1, abs (deriv f x) ≤ k) :
  abs (∫ x in 0..1, f x - (∑ i in finset.range n, (f (i / n : ℝ) / n))) ≤ k / n :=
sorry

end bound_riemann_sum_difference_l217_217313


namespace patrick_purchased_pencils_l217_217612

theorem patrick_purchased_pencils 
  (S : ℝ) -- selling price of one pencil
  (C : ℝ) -- cost price of one pencil
  (P : ℕ) -- number of pencils purchased
  (h1 : C = 1.3333333333333333 * S) -- condition 1: cost of pencils is 1.3333333 times the selling price
  (h2 : (P : ℝ) * C - (P : ℝ) * S = 20 * S) -- condition 2: loss equals selling price of 20 pencils
  : P = 60 := 
sorry

end patrick_purchased_pencils_l217_217612


namespace algebraic_expression_value_l217_217738

theorem algebraic_expression_value (m : ℝ) (h : m^2 - m = 1) : 
  (m - 1)^2 + (m + 1) * (m - 1) + 2022 = 2024 :=
by
  sorry

end algebraic_expression_value_l217_217738


namespace emily_garden_larger_l217_217195

-- Define the dimensions and conditions given in the problem
def john_length : ℕ := 30
def john_width : ℕ := 60
def emily_length : ℕ := 35
def emily_width : ℕ := 55

-- Define the effective area for John’s garden given the double space requirement
def john_usable_area : ℕ := (john_length * john_width) / 2

-- Define the total area for Emily’s garden
def emily_usable_area : ℕ := emily_length * emily_width

-- State the theorem to be proved
theorem emily_garden_larger : emily_usable_area - john_usable_area = 1025 :=
by
  sorry

end emily_garden_larger_l217_217195


namespace total_songs_l217_217858

open Nat

/-- Define the overall context and setup for the problem --/
def girls : List String := ["Mary", "Alina", "Tina", "Hanna"]

def hanna_songs : ℕ := 7
def mary_songs : ℕ := 4

def alina_songs (a : ℕ) : Prop := a > mary_songs ∧ a < hanna_songs
def tina_songs (t : ℕ) : Prop := t > mary_songs ∧ t < hanna_songs

theorem total_songs (a t : ℕ) (h_alina : alina_songs a) (h_tina : tina_songs t) : 
  (11 + a + t) % 3 = 0 → (7 + 4 + a + t) / 3 = 7 := by
  sorry

end total_songs_l217_217858


namespace rowing_upstream_distance_l217_217689

theorem rowing_upstream_distance 
  (b s t d1 d2 : ℝ)
  (h1 : s = 7)
  (h2 : d1 = 72)
  (h3 : t = 3)
  (h4 : d1 = (b + s) * t) :
  d2 = (b - s) * t → d2 = 30 :=
by 
  intros h5
  sorry

end rowing_upstream_distance_l217_217689


namespace age_of_new_person_l217_217625

theorem age_of_new_person (n : ℕ) (T A : ℕ) (h₁ : n = 10) (h₂ : T = 15 * n)
    (h₃ : (T + A) / (n + 1) = 17) : A = 37 := by
  sorry

end age_of_new_person_l217_217625


namespace consecutive_probability_is_two_fifths_l217_217496

-- Conditions
def total_days : ℕ := 5
def select_days : ℕ := 2

-- Total number of basic events (number of ways to choose 2 days out of 5)
def total_events : ℕ := Nat.choose total_days select_days -- This is C(5, 2)

-- Number of basic events where 2 selected days are consecutive
def consecutive_events : ℕ := 4

-- Probability that the selected 2 days are consecutive
def consecutive_probability : ℚ := consecutive_events / total_events

-- Theorem to be proved
theorem consecutive_probability_is_two_fifths :
  consecutive_probability = 2 / 5 :=
by
  sorry

end consecutive_probability_is_two_fifths_l217_217496


namespace tangent_intersect_x_axis_l217_217672

-- Defining the conditions based on the given problem
def radius1 : ℝ := 3
def center1 : ℝ × ℝ := (0, 0)

def radius2 : ℝ := 5
def center2 : ℝ × ℝ := (12, 0)

-- Stating what needs to be proved
theorem tangent_intersect_x_axis : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (∀ (x1 x2 : ℝ), 
    (x1 = x) ∧ 
    (x2 = 12 - x) ∧ 
    (radius1 / (center2.1 - x) = radius2 / x2) → 
    (x = 9 / 2)) := 
sorry

end tangent_intersect_x_axis_l217_217672


namespace eventually_repeating_last_two_digits_l217_217397

theorem eventually_repeating_last_two_digits (K : ℕ) : ∃ N : ℕ, ∃ t : ℕ, 
    (∃ s : ℕ, t = s * 77 + N) ∨ (∃ u : ℕ, t = u * 54 + N) ∧ (t % 100) / 10 = (t % 100) % 10 :=
sorry

end eventually_repeating_last_two_digits_l217_217397


namespace simplify_nested_sqrt_l217_217936

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l217_217936


namespace ben_current_age_l217_217626

theorem ben_current_age (a b c : ℕ) 
  (h1 : a + b + c = 36) 
  (h2 : c = 2 * a - 4) 
  (h3 : b + 5 = 3 * (a + 5) / 4) : 
  b = 5 := 
by
  sorry

end ben_current_age_l217_217626


namespace distance_to_city_l217_217838

variable (d : ℝ)  -- Define d as a real number

theorem distance_to_city (h1 : ¬ (d ≥ 13)) (h2 : ¬ (d ≤ 10)) :
  10 < d ∧ d < 13 :=
by
  -- Here we will formalize the proof in Lean syntax
  sorry

end distance_to_city_l217_217838


namespace nonneg_sol_eq_l217_217714

theorem nonneg_sol_eq {a b c : ℝ} (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c) 
  (h1 : a * (a + b) = b * (b + c)) (h2 : b * (b + c) = c * (c + a)) : 
  a = b ∧ b = c := 
sorry

end nonneg_sol_eq_l217_217714


namespace no_fractions_satisfy_condition_l217_217145

theorem no_fractions_satisfy_condition :
  ∀ (x y : ℕ), 
    x > 0 → y > 0 → Nat.gcd x y = 1 →
    (1.2 : ℚ) * (x : ℚ) / (y : ℚ) = (x + 2 : ℚ) / (y + 2 : ℚ) →
    False :=
by
  intros x y hx hy hrel hcond
  sorry

end no_fractions_satisfy_condition_l217_217145


namespace _l217_217062

open EuclideanGeometry

noncomputable def orthocenter (A B C H : Point) : Prop :=
  is_perpendicular A H C B ∧ is_perpendicular B H A C ∧ is_perpendicular C H B A

noncomputable def midpoint (P Q M : Point) : Prop :=
  2 • M = P + Q

noncomputable theorem perpendicular_midpoints_to_altitudes {A B C H D E X Y : Point}
  (h_orthocenter : orthocenter A B C H)
  (h_altitude_AD : is_perpendicular A D B C)
  (h_altitude_BE : is_perpendicular B E A C)
  (h_midpoint_X : midpoint A B X)
  (h_midpoint_Y : midpoint C H Y) :
  is_perpendicular X Y D E :=
sorry

end _l217_217062


namespace fractional_equation_solution_l217_217789

theorem fractional_equation_solution (x : ℝ) (h₁ : x ≠ 0) : (1 / x = 2 / (x + 3)) → x = 3 := by
  sorry

end fractional_equation_solution_l217_217789


namespace anna_not_lose_l217_217126

theorem anna_not_lose :
  ∀ (cards : Fin 9 → ℕ),
    ∃ (A B C D : ℕ),
      (A + B ≥ C + D) :=
by
  sorry

end anna_not_lose_l217_217126


namespace trig_identity_l217_217311

theorem trig_identity (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) 
  : Real.cos (5 / 6 * π + α) + (Real.cos (4 * π / 3 + α))^2 = (2 - Real.sqrt 3) / 3 := 
sorry

end trig_identity_l217_217311


namespace students_in_class_l217_217485

theorem students_in_class (n S : ℕ) 
    (h1 : S = 15 * n)
    (h2 : (S + 56) / (n + 1) = 16) : n = 40 :=
by
  sorry

end students_in_class_l217_217485


namespace right_triangle_perimeter_l217_217115

theorem right_triangle_perimeter 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_area : 1/2 * 30 * b = 180)
  (h_pythagorean : c^2 = 30^2 + b^2)
  : a + b + c = 42 + 2 * Real.sqrt 261 :=
sorry

end right_triangle_perimeter_l217_217115


namespace lcm_of_20_45_75_l217_217655

-- Definitions for the given numbers and their prime factorizations
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 75

def factor1 : ℕ → Prop := λ n, n = 2 ^ 2 * 5
def factor2 : ℕ → Prop := λ n, n = 3 ^ 2 * 5
def factor3 : ℕ → Prop := λ n, n = 3 * 5 ^ 2

-- The condition of using the least common multiple function from mathlib
def lcm_def (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- The statement to prove
theorem lcm_of_20_45_75 : lcm_def num1 num2 num3 = 900 := by
  -- Factors condition (Note: These help ensure the numbers' factors are as stated)
  have h1 : factor1 num1 := by { unfold num1 factor1, exact rfl }
  have h2 : factor2 num2 := by { unfold num2 factor2, exact rfl }
  have h3 : factor3 num3 := by { unfold num3 factor3, exact rfl }
  sorry -- This is the place where the proof would go.

end lcm_of_20_45_75_l217_217655


namespace greatest_integer_radius_of_circle_l217_217587

theorem greatest_integer_radius_of_circle (r : ℕ) (A : ℝ) (hA : A < 80 * Real.pi) :
  r <= 8 ∧ r * r < 80 :=
sorry

end greatest_integer_radius_of_circle_l217_217587


namespace initial_bottle_count_l217_217469

variable (B: ℕ)

-- Conditions: Each bottle holds 15 stars, bought 3 more bottles, total 75 stars to fill
def bottle_capacity := 15
def additional_bottles := 3
def total_stars := 75

-- The main statement we want to prove
theorem initial_bottle_count (h : (B + additional_bottles) * bottle_capacity = total_stars) : 
    B = 2 :=
by sorry

end initial_bottle_count_l217_217469


namespace length_of_CD_l217_217478

theorem length_of_CD (x y : ℝ) (h1 : x = (1/5) * (4 + y))
  (h2 : (x + 4) / y = 2 / 3) (h3 : 4 = 4) : x + y + 4 = 17.143 :=
sorry

end length_of_CD_l217_217478


namespace tv_weight_difference_l217_217844

-- Definitions for the given conditions
def bill_tv_length : ℕ := 48
def bill_tv_width : ℕ := 100
def bob_tv_length : ℕ := 70
def bob_tv_width : ℕ := 60
def weight_per_square_inch : ℕ := 4
def ounces_per_pound : ℕ := 16

-- The statement to prove
theorem tv_weight_difference : (bill_tv_length * bill_tv_width * weight_per_square_inch)
                               - (bob_tv_length * bob_tv_width * weight_per_square_inch)
                               = 150 * ounces_per_pound := by
  sorry

end tv_weight_difference_l217_217844


namespace tens_digit_36_pow_12_l217_217973

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def tens_digit (n : ℕ) : ℕ :=
  (last_two_digits n) / 10

theorem tens_digit_36_pow_12 : tens_digit (36^12) = 3 :=
by
  sorry

end tens_digit_36_pow_12_l217_217973


namespace number_division_remainder_l217_217832

theorem number_division_remainder (N k m : ℤ) (h1 : N = 281 * k + 160) (h2 : N = D * m + 21) : D = 139 :=
by sorry

end number_division_remainder_l217_217832


namespace triangles_in_figure_l217_217442

-- Define the conditions of the problem.
def bottom_row_small := 4
def next_row_small := 3
def following_row_small := 2
def topmost_row_small := 1

def small_triangles := bottom_row_small + next_row_small + following_row_small + topmost_row_small

def medium_triangles := 3
def large_triangle := 1

def total_triangles := small_triangles + medium_triangles + large_triangle

-- Lean proof statement that the total number of triangles is 14
theorem triangles_in_figure : total_triangles = 14 :=
by
  unfold total_triangles
  unfold small_triangles
  unfold bottom_row_small next_row_small following_row_small topmost_row_small
  unfold medium_triangles large_triangle
  sorry

end triangles_in_figure_l217_217442


namespace distinct_triangles_from_tetrahedron_l217_217440

theorem distinct_triangles_from_tetrahedron (tetrahedron_vertices : Finset α)
  (h_tet : tetrahedron_vertices.card = 4) : 
  ∃ (triangles : Finset (Finset α)), triangles.card = 4 ∧ (∀ triangle ∈ triangles, triangle.card = 3 ∧ triangle ⊆ tetrahedron_vertices) :=
by
  -- Proof omitted
  sorry

end distinct_triangles_from_tetrahedron_l217_217440


namespace Trishul_invested_less_than_Raghu_l217_217502

-- Definitions based on conditions
def Raghu_investment : ℝ := 2500
def Total_investment : ℝ := 7225

def Vishal_invested_more_than_Trishul (T V : ℝ) : Prop :=
  V = 1.10 * T

noncomputable def percentage_decrease (original decrease : ℝ) : ℝ :=
  (decrease / original) * 100

theorem Trishul_invested_less_than_Raghu (T V : ℝ) 
  (h1 : Vishal_invested_more_than_Trishul T V)
  (h2 : T + V + Raghu_investment = Total_investment) :
  percentage_decrease Raghu_investment (Raghu_investment - T) = 10 := by
  sorry

end Trishul_invested_less_than_Raghu_l217_217502


namespace difference_in_spectators_l217_217458

-- Define the parameters given in the problem
def people_game_2 : ℕ := 80
def people_game_1 : ℕ := people_game_2 - 20
def people_game_3 : ℕ := people_game_2 + 15
def people_last_week : ℕ := 200

-- Total people who watched the games this week
def people_this_week : ℕ := people_game_1 + people_game_2 + people_game_3

-- Theorem statement: Prove the difference in people watching the games between this week and last week is 35.
theorem difference_in_spectators : people_this_week - people_last_week = 35 :=
  sorry

end difference_in_spectators_l217_217458


namespace rational_expression_nonnegative_l217_217874

theorem rational_expression_nonnegative (x : ℚ) : 2 * |x| + x ≥ 0 :=
  sorry

end rational_expression_nonnegative_l217_217874


namespace find_range_of_a_l217_217876

def quadratic_function (a x : ℝ) : ℝ :=
  x^2 + 2 * (a - 1) * x + 2

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≥ f y

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

def is_monotonic_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  is_decreasing_on f I ∨ is_increasing_on f I

theorem find_range_of_a (a : ℝ) :
  is_monotonic_on (quadratic_function a) (Set.Icc (-4) 4) ↔ (a ≤ -3 ∨ a ≥ 5) :=
sorry

end find_range_of_a_l217_217876


namespace mr_callen_total_loss_l217_217473

noncomputable def total_loss : ℤ :=
  let bought_paintings_price := 15 * 60
  let bought_wooden_toys_price := 12 * 25
  let bought_handmade_hats_price := 20 * 15
  let total_bought_price := bought_paintings_price + bought_wooden_toys_price + bought_handmade_hats_price
  let sold_paintings_price := 15 * (60 - (60 * 18 / 100))
  let sold_wooden_toys_price := 12 * (25 - (25 * 25 / 100))
  let sold_handmade_hats_price := 20 * (15 - (15 * 10 / 100))
  let total_sold_price := sold_paintings_price + sold_wooden_toys_price + sold_handmade_hats_price
  total_bought_price - total_sold_price

theorem mr_callen_total_loss : total_loss = 267 := by
  sorry

end mr_callen_total_loss_l217_217473


namespace max_stamps_purchase_l217_217880

theorem max_stamps_purchase (price_per_stamp : ℕ) (total_money : ℕ) (h_price : price_per_stamp = 45) (h_money : total_money = 5000) : 
  (total_money / price_per_stamp) = 111 := 
by 
  rw [h_price, h_money]
  rfl

sorry

end max_stamps_purchase_l217_217880


namespace height_of_platform_l217_217596

variables (l w h : ℕ)

theorem height_of_platform (hl1 : l + h - 2 * w = 36) (hl2 : w + h - l = 30) (hl3 : h = 2 * w) : h = 44 := 
sorry

end height_of_platform_l217_217596


namespace Frank_read_books_l217_217562

noncomputable def books_read (total_days : ℕ) (days_per_book : ℕ) : ℕ :=
total_days / days_per_book

theorem Frank_read_books : books_read 492 12 = 41 := by
  sorry

end Frank_read_books_l217_217562


namespace problem1_problem2_problem3_problem4_l217_217429

section

variables (x y : Real)

-- Given conditions
def x_def : x = 3 + 2 * Real.sqrt 2 := sorry
def y_def : y = 3 - 2 * Real.sqrt 2 := sorry

-- Problem 1: Prove x + y = 6
theorem problem1 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x + y = 6 := 
by sorry

-- Problem 2: Prove x - y = 4 * sqrt 2
theorem problem2 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x - y = 4 * Real.sqrt 2 :=
by sorry

-- Problem 3: Prove xy = 1
theorem problem3 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x * y = 1 := 
by sorry

-- Problem 4: Prove x^2 - 3xy + y^2 - x - y = 25
theorem problem4 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x^2 - 3 * x * y + y^2 - x - y = 25 :=
by sorry

end

end problem1_problem2_problem3_problem4_l217_217429


namespace least_common_multiple_of_20_45_75_l217_217650

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l217_217650


namespace inverse_function_correct_l217_217066

theorem inverse_function_correct :
  ( ∀ x : ℝ, (x > 1) → (∃ y : ℝ, y = 1 + Real.log (x - 1)) ↔ (∀ y : ℝ, y > 0 → (∃ x : ℝ, x = e^(y + 1) - 1))) :=
by
  sorry

end inverse_function_correct_l217_217066


namespace diane_stamp_combinations_l217_217151

/-- Define the types of stamps Diane has --/
def diane_stamps : List ℕ := [1, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8]

/-- Define the condition for the correct number of different arrangements to sum exactly to 12 cents -/
noncomputable def count_arrangements (stamps : List ℕ) (sum : ℕ) : ℕ :=
  -- Implementation of the counting function goes here
  sorry

/-- Prove that the number of distinct arrangements to make exactly 12 cents is 13 --/
theorem diane_stamp_combinations : count_arrangements diane_stamps 12 = 13 :=
  sorry

end diane_stamp_combinations_l217_217151


namespace problem_l217_217161

variable (f : ℝ → ℝ)

-- Given condition
axiom h : ∀ x : ℝ, f (1 / x) = 1 / (x + 1)

-- Prove that f(2) = 2/3
theorem problem : f 2 = 2 / 3 :=
sorry

end problem_l217_217161


namespace maximum_value_x2_add_3xy_add_y2_l217_217220

-- Define the conditions
variables {x y : ℝ}

-- State the theorem
theorem maximum_value_x2_add_3xy_add_y2 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : 3 * x^2 - 2 * x * y + 5 * y^2 = 12) :
  ∃ e f g h : ℕ,
    x^2 + 3 * x * y + y^2 = (1144 + 204 * Real.sqrt 15) / 91 ∧ e + f + g + h = 1454 :=
sorry

end maximum_value_x2_add_3xy_add_y2_l217_217220


namespace journey_distance_l217_217836

theorem journey_distance 
  (T : ℝ) 
  (s1 s2 s3 : ℝ) 
  (hT : T = 36) 
  (hs1 : s1 = 21)
  (hs2 : s2 = 45)
  (hs3 : s3 = 24) : ∃ (D : ℝ), D = 972 :=
  sorry

end journey_distance_l217_217836


namespace exists_t_for_f_inequality_l217_217862

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1) ^ 2

theorem exists_t_for_f_inequality :
  ∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → f (x + t) ≤ x := by
  sorry

end exists_t_for_f_inequality_l217_217862


namespace sum_reciprocal_squares_roots_l217_217044

-- Define the polynomial P(X) = X^3 - 3X - 1
noncomputable def P (X : ℂ) : ℂ := X^3 - 3 * X - 1

-- Define the roots of the polynomial
variables (r1 r2 r3 : ℂ)

-- State that r1, r2, and r3 are roots of the polynomial
variable (hroots : P r1 = 0 ∧ P r2 = 0 ∧ P r3 = 0)

-- Vieta's formulas conditions for the polynomial P
variable (hvieta : r1 + r2 + r3 = 0 ∧ r1 * r2 + r1 * r3 + r2 * r3 = -3 ∧ r1 * r2 * r3 = 1)

-- The sum of the reciprocals of the squares of the roots
theorem sum_reciprocal_squares_roots : (1 / r1^2) + (1 / r2^2) + (1 / r3^2) = 9 := 
sorry

end sum_reciprocal_squares_roots_l217_217044


namespace circles_radius_difference_l217_217493

variable (s : ℝ)

theorem circles_radius_difference (h : (π * (2*s)^2) / (π * s^2) = 4) : (2 * s - s) = s :=
by
  sorry

end circles_radius_difference_l217_217493


namespace compute_63_times_57_l217_217136

theorem compute_63_times_57 : 63 * 57 = 3591 := 
by {
   have h : (60 + 3) * (60 - 3) = 60^2 - 3^2, from
     by simp [mul_add, add_mul, add_assoc, sub_mul, mul_sub, sub_add, sub_sub, add_sub, mul_self_sub],
   have h1 : 60^2 = 3600, from rfl,
   have h2 : 3^2 = 9, from rfl,
   have h3 : 60^2 - 3^2 = 3600 - 9, by rw [h1, h2],
   rw h at h3,
   exact h3,
}

end compute_63_times_57_l217_217136


namespace sqrt6_special_op_l217_217590

-- Define the binary operation (¤) as given in the problem.
def special_op (x y : ℝ) : ℝ := (x + y) ^ 2 - (x - y) ^ 2

-- States that √6 ¤ √6 is equal to 24.
theorem sqrt6_special_op : special_op (Real.sqrt 6) (Real.sqrt 6) = 24 :=
by
  sorry

end sqrt6_special_op_l217_217590


namespace continuous_polynomial_continuous_cosecant_l217_217056

-- Prove that the function \( f(x) = 2x^2 - 1 \) is continuous on \(\mathbb{R}\)
theorem continuous_polynomial : Continuous (fun x : ℝ => 2 * x^2 - 1) :=
sorry

-- Prove that the function \( g(x) = (\sin x)^{-1} \) is continuous on \(\mathbb{R}\) \setminus \(\{ k\pi \mid k \in \mathbb{Z} \} \)
theorem continuous_cosecant : ∀ x : ℝ, x ∉ Set.range (fun k : ℤ => k * Real.pi) → ContinuousAt (fun x : ℝ => (Real.sin x)⁻¹) x :=
sorry

end continuous_polynomial_continuous_cosecant_l217_217056


namespace pencils_before_buying_l217_217840

theorem pencils_before_buying (x total bought : Nat) 
  (h1 : bought = 7) 
  (h2 : total = 10) 
  (h3 : total = x + bought) : x = 3 :=
by
  sorry

end pencils_before_buying_l217_217840


namespace unique_sums_count_l217_217128

open Set

-- Defining the sets of chips in bags C and D
def BagC : Set ℕ := {1, 3, 7, 9}
def BagD : Set ℕ := {4, 6, 8}

-- The proof problem: show there are 7 unique sums
theorem unique_sums_count : (BagC ×ˢ BagD).image (λ p => p.1 + p.2) = {5, 7, 9, 11, 13, 15, 17} :=
by
  -- Proof omitted; complete proof would go here
  sorry

end unique_sums_count_l217_217128


namespace cost_of_soccer_basketball_balls_max_basketballs_l217_217497

def cost_of_balls (x y : ℕ) : Prop :=
  (7 * x = 5 * y) ∧ (40 * x + 20 * y = 3400)

def cost_constraint (x y m : ℕ) : Prop :=
  (x = 50) ∧ (y = 70) ∧ (70 * m + 50 * (100 - m) ≤ 6300)

theorem cost_of_soccer_basketball_balls (x y : ℕ) (h : cost_of_balls x y) : x = 50 ∧ y = 70 :=
  by sorry

theorem max_basketballs (x y m : ℕ) (h : cost_constraint x y m) : m ≤ 65 :=
  by sorry

end cost_of_soccer_basketball_balls_max_basketballs_l217_217497


namespace bill_needs_125_bouquets_to_earn_1000_l217_217273

-- Define the constants for the problem
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_profit : ℕ := 1000

-- Define the problem in terms of a theorem
theorem bill_needs_125_bouquets_to_earn_1000 :
  ∃ n : ℕ, (35 / roses_per_bouquet_sell) * cost_per_bouquet - (5 * cost_per_bouquet) = 40 → (5 * n) = 125 :=
begin
  sorry
end

end bill_needs_125_bouquets_to_earn_1000_l217_217273


namespace valid_elixir_combinations_l217_217405

theorem valid_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let incompatible_herbs := 3
  let incompatible_crystals := 2
  let total_combinations := herbs * crystals
  let incompatible_combinations := incompatible_herbs * incompatible_crystals
  total_combinations - incompatible_combinations = 18 :=
by
  sorry

end valid_elixir_combinations_l217_217405


namespace fuel_first_third_l217_217207

-- Defining constants based on conditions
def total_fuel := 60
def fuel_second_third := total_fuel / 3
def fuel_final_third := fuel_second_third / 2

-- Defining what we need to prove
theorem fuel_first_third :
  total_fuel - (fuel_second_third + fuel_final_third) = 30 :=
by
  sorry

end fuel_first_third_l217_217207


namespace sally_received_quarters_l217_217924

theorem sally_received_quarters : 
  ∀ (original_quarters total_quarters received_quarters : ℕ), 
  original_quarters = 760 → 
  total_quarters = 1178 → 
  received_quarters = total_quarters - original_quarters → 
  received_quarters = 418 :=
by 
  intros original_quarters total_quarters received_quarters h_original h_total h_received
  rw [h_original, h_total] at h_received
  exact h_received

end sally_received_quarters_l217_217924


namespace basketball_game_first_half_points_l217_217107

theorem basketball_game_first_half_points (a b r d : ℕ) (H1 : a = b)
  (H2 : a * (1 + r + r^2 + r^3) = 4 * a + 6 * d + 1) 
  (H3 : 15 * a ≤ 100) (H4 : b + (b + d) + b + 2 * d + b + 3 * d < 100) : 
  (a + a * r + b + b + d) = 34 :=
by sorry

end basketball_game_first_half_points_l217_217107


namespace proof_height_difference_l217_217445

noncomputable def height_in_inches_between_ruby_and_xavier : Prop :=
  let janet_height_inches := 62.75
  let inch_to_cm := 2.54
  let janet_height_cm := janet_height_inches * inch_to_cm
  let charlene_height := 1.5 * janet_height_cm
  let pablo_height := charlene_height + 1.85 * 100
  let ruby_height := pablo_height - 0.5
  let xavier_height := charlene_height + 2.13 * 100 - 97.75
  let paul_height := ruby_height + 50
  let height_diff_cm := xavier_height - ruby_height
  let height_diff_inches := height_diff_cm / inch_to_cm
  height_diff_inches = -18.78

theorem proof_height_difference :
  height_in_inches_between_ruby_and_xavier :=
by
  sorry

end proof_height_difference_l217_217445


namespace subtraction_of_largest_three_digit_from_smallest_five_digit_l217_217391

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000

theorem subtraction_of_largest_three_digit_from_smallest_five_digit :
  smallest_five_digit_number - largest_three_digit_number = 9001 :=
by
  sorry

end subtraction_of_largest_three_digit_from_smallest_five_digit_l217_217391


namespace LCM_20_45_75_is_900_l217_217661

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l217_217661


namespace least_positive_integer_greater_than_100_l217_217802

theorem least_positive_integer_greater_than_100 : ∃ n : ℕ, n > 100 ∧ (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) ∧ n = 2521 :=
by
  sorry

end least_positive_integer_greater_than_100_l217_217802


namespace periodic_odd_function_value_l217_217367

theorem periodic_odd_function_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
    (h_periodic : ∀ x : ℝ, f (x + 2) = f x) (h_value : f 0.5 = -1) : f 7.5 = 1 :=
by
  -- Proof would go here.
  sorry

end periodic_odd_function_value_l217_217367


namespace moles_of_water_formed_l217_217421

-- Definitions (conditions)
def reaction : String := "NaOH + HCl → NaCl + H2O"

def initial_moles_NaOH : ℕ := 1
def initial_moles_HCl : ℕ := 1
def mole_ratio_NaOH_HCl : ℕ := 1
def mole_ratio_NaOH_H2O : ℕ := 1

-- The proof problem
theorem moles_of_water_formed :
  initial_moles_NaOH = mole_ratio_NaOH_HCl →
  initial_moles_HCl = mole_ratio_NaOH_HCl →
  mole_ratio_NaOH_H2O * initial_moles_NaOH = 1 :=
by
  intros h1 h2
  sorry

end moles_of_water_formed_l217_217421


namespace seq_an_identity_l217_217901

theorem seq_an_identity (n : ℕ) (a : ℕ → ℕ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) > a n)
  (h₃ : ∀ n, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) 
  : a n = n^2 := sorry

end seq_an_identity_l217_217901


namespace original_cost_price_l217_217693

theorem original_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1275) 
  (h2 : loss_percentage = 15) 
  (h3 : SP = (1 - loss_percentage / 100) * C) : 
  C = 1500 := 
by 
  sorry

end original_cost_price_l217_217693


namespace inequality_holds_l217_217048

theorem inequality_holds (x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4)
  (h5 : 0 < x5) (h6 : 0 < x6) (h7 : 0 < x7) (h8 : 0 < x8) 
  (h9 : 0 < x9) :
  (x1 - x3) / (x1 * x3 + 2 * x2 * x3 + x2^2) +
  (x2 - x4) / (x2 * x4 + 2 * x3 * x4 + x3^2) +
  (x3 - x5) / (x3 * x5 + 2 * x4 * x5 + x4^2) +
  (x4 - x6) / (x4 * x6 + 2 * x5 * x6 + x5^2) +
  (x5 - x7) / (x5 * x7 + 2 * x6 * x7 + x6^2) +
  (x6 - x8) / (x6 * x8 + 2 * x7 * x8 + x7^2) +
  (x7 - x9) / (x7 * x9 + 2 * x8 * x9 + x8^2) +
  (x8 - x1) / (x8 * x1 + 2 * x9 * x1 + x9^2) +
  (x9 - x2) / (x9 * x2 + 2 * x1 * x2 + x1^2) ≥ 0 := 
sorry

end inequality_holds_l217_217048


namespace inequality_ab_bc_ca_l217_217916

open Real

theorem inequality_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a) / (2 * (a + b + c))) := by
sorry

end inequality_ab_bc_ca_l217_217916


namespace correct_equation_l217_217808

theorem correct_equation (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by
  sorry

end correct_equation_l217_217808


namespace balls_in_third_pile_l217_217638

theorem balls_in_third_pile (a b c x : ℕ) (h1 : a + b + c = 2012) (h2 : b - x = 17) (h3 : a - x = 2 * (c - x)) : c = 665 := by
  sorry

end balls_in_third_pile_l217_217638


namespace vacation_cost_per_person_l217_217346

theorem vacation_cost_per_person (airbnb_cost car_cost : ℝ) (num_people : ℝ) 
  (h1 : airbnb_cost = 3200) (h2 : car_cost = 800) (h3 : num_people = 8) : 
  (airbnb_cost + car_cost) / num_people = 500 := 
by 
  sorry

end vacation_cost_per_person_l217_217346


namespace part_a_part_b_l217_217869

variable {f : ℝ → ℝ} 

-- Given conditions
axiom condition1 (x y : ℝ) : f (x + y) + 1 = f x + f y
axiom condition2 : f (1/2) = 0
axiom condition3 (x : ℝ) : x > 1/2 → f x < 0

-- Part (a)
theorem part_a (x : ℝ) : f x = 1/2 + 1/2 * f (2 * x) :=
sorry

-- Part (b)
theorem part_b (n : ℕ) (hn : n > 0) (x : ℝ) 
  (hx : 1 / 2^(n + 1) ≤ x ∧ x ≤ 1 / 2^n) : f x ≤ 1 - 1 / 2^n :=
sorry

end part_a_part_b_l217_217869


namespace part_a_part_b_l217_217394

-- Part (a): Proving at most one integer solution for general k
theorem part_a (k : ℤ) : 
  ∀ (x1 x2 : ℤ), (x1^3 - 24*x1 + k = 0 ∧ x2^3 - 24*x2 + k = 0) → x1 = x2 :=
sorry

-- Part (b): Proving exactly one integer solution for k = -2016
theorem part_b :
  ∃! (x : ℤ), x^3 + 24*x - 2016 = 0 :=
sorry

end part_a_part_b_l217_217394


namespace arithmetic_sequence_fifth_term_l217_217016

variable (a d : ℕ)

-- Conditions
def condition1 := (a + d) + (a + 3 * d) = 10
def condition2 := a + (a + 2 * d) = 8

-- Fifth term calculation
def fifth_term := a + 4 * d

theorem arithmetic_sequence_fifth_term (h1 : condition1 a d) (h2 : condition2 a d) : fifth_term a d = 7 :=
by
  sorry

end arithmetic_sequence_fifth_term_l217_217016


namespace sum_of_possible_ks_l217_217029

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l217_217029


namespace books_on_desk_none_useful_l217_217474

theorem books_on_desk_none_useful :
  ∃ (answer : String), answer = "none" ∧ 
  (answer = "nothing" ∨ answer = "no one" ∨ answer = "neither" ∨ answer = "none")
  → answer = "none"
:= by
  sorry

end books_on_desk_none_useful_l217_217474


namespace original_price_of_shoes_l217_217582

noncomputable def original_price (final_price : ℝ) (sales_tax : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  final_price / sales_tax / (discount1 * discount2)

theorem original_price_of_shoes :
  original_price 51 1.07 0.40 0.85 = 140.18 := by
    have h_pre_tax_price : 47.66 = 51 / 1.07 := sorry
    have h_price_relation : 47.66 = 0.85 * 0.40 * 140.18 := sorry
    sorry

end original_price_of_shoes_l217_217582


namespace trapezoid_ratio_l217_217760

structure Trapezoid (α : Type) [LinearOrderedField α] :=
  (AB CD : α)
  (areas : List α)
  (AB_gt_CD : AB > CD)
  (areas_eq : areas = [3, 5, 6, 8])

open Trapezoid

theorem trapezoid_ratio (α : Type) [LinearOrderedField α] (T : Trapezoid α) :
  ∃ ρ : α, T.AB / T.CD = ρ ∧ ρ = 8 / 3 :=
by
  sorry

end trapezoid_ratio_l217_217760


namespace gcd_18_30_is_6_gcd_18_30_is_even_l217_217305

def gcd_18_30 : ℕ := Nat.gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 := by
  sorry

theorem gcd_18_30_is_even : Even gcd_18_30 := by
  sorry

end gcd_18_30_is_6_gcd_18_30_is_even_l217_217305


namespace unique_solution_exists_l217_217414

theorem unique_solution_exists (k : ℚ) (h : k ≠ 0) : 
  (∀ x : ℚ, (x + 3) / (kx - 2) = x → x = -2) ↔ k = -3 / 4 := 
by
  sorry

end unique_solution_exists_l217_217414


namespace point_outside_circle_l217_217878

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) : a^2 + b^2 > 1 :=
by
  sorry

end point_outside_circle_l217_217878


namespace susan_remaining_money_l217_217623

theorem susan_remaining_money :
  let initial_amount := 90
  let food_spent := 20
  let game_spent := 3 * food_spent
  let total_spent := food_spent + game_spent
  initial_amount - total_spent = 10 :=
by 
  sorry

end susan_remaining_money_l217_217623


namespace min_distance_ellipse_line_l217_217489

theorem min_distance_ellipse_line :
  let ellipse (x y : ℝ) := (x ^ 2) / 16 + (y ^ 2) / 12 = 1
  let line (x y : ℝ) := x - 2 * y - 12 = 0
  ∃ (d : ℝ), d = 4 * Real.sqrt 5 / 5 ∧
             (∀ (x y : ℝ), ellipse x y → ∃ (d' : ℝ), line x y → d' ≥ d) :=
  sorry

end min_distance_ellipse_line_l217_217489


namespace quadratic_function_properties_l217_217573

noncomputable def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  (m + 2) * x^(m^2 + m - 4)

theorem quadratic_function_properties :
  (∀ m, (m^2 + m - 4 = 2) → (m = -3 ∨ m = 2))
  ∧ (m = -3 → quadratic_function m 0 = 0) 
  ∧ (m = -3 → ∀ x, x > 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0)
  ∧ (m = -3 → ∀ x, x < 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0) :=
by
  -- Proof will be supplied here.
  sorry

end quadratic_function_properties_l217_217573


namespace remaining_amount_to_be_paid_l217_217447

-- Define the conditions
def deposit_percentage : ℚ := 10 / 100
def deposit_amount : ℚ := 80

-- Define the total purchase price based on the conditions
def total_price : ℚ := deposit_amount / deposit_percentage

-- Define the remaining amount to be paid
def remaining_amount : ℚ := total_price - deposit_amount

-- State the theorem
theorem remaining_amount_to_be_paid : remaining_amount = 720 := by
  sorry

end remaining_amount_to_be_paid_l217_217447


namespace find_width_of_sheet_of_paper_l217_217113

def width_of_sheet_of_paper (W : ℝ) : Prop :=
  let margin := 1.5
  let length_of_paper := 10
  let area_covered := 38.5
  let width_of_picture := W - 2 * margin
  let length_of_picture := length_of_paper - 2 * margin
  width_of_picture * length_of_picture = area_covered

theorem find_width_of_sheet_of_paper : ∃ W : ℝ, width_of_sheet_of_paper W ∧ W = 8.5 :=
by
  -- Placeholder for the actual proof
  sorry

end find_width_of_sheet_of_paper_l217_217113


namespace certain_number_is_32_l217_217190

theorem certain_number_is_32 (k t : ℚ) (certain_number : ℚ) 
  (h1 : t = 5/9 * (k - certain_number))
  (h2 : t = 75) (h3 : k = 167) :
  certain_number = 32 :=
sorry

end certain_number_is_32_l217_217190


namespace multiply_63_57_l217_217137

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end multiply_63_57_l217_217137


namespace solution_set_inequality_l217_217786

theorem solution_set_inequality (x : ℝ) :
  (3 * x + 2 ≥ 1 ∧ (5 - x) / 2 < 0) ↔ (-1 / 3 ≤ x ∧ x < 5) :=
by
  sorry

end solution_set_inequality_l217_217786


namespace polynomial_value_at_3_l217_217500

def f (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

theorem polynomial_value_at_3 : f 3 = 1209.4 := 
by
  sorry

end polynomial_value_at_3_l217_217500


namespace parallelogram_area_l217_217476

theorem parallelogram_area {a b : ℝ} (h₁ : a = 9) (h₂ : b = 12) (angle : ℝ) (h₃ : angle = 150) : 
  ∃ (area : ℝ), area = 54 * Real.sqrt 3 :=
by
  sorry

end parallelogram_area_l217_217476


namespace complex_coordinates_l217_217193

theorem complex_coordinates : (⟨(-1:ℝ), (-1:ℝ)⟩ : ℂ) = (⟨0,1⟩ : ℂ) * (⟨-2,0⟩ : ℂ) / (⟨1,1⟩ : ℂ) :=
by
  sorry

end complex_coordinates_l217_217193


namespace solve_system_l217_217636

-- The system of equations as conditions in Lean
def system1 (x y : ℤ) : Prop := 5 * x + 2 * y = 25
def system2 (x y : ℤ) : Prop := 3 * x + 4 * y = 15

-- The statement that asserts the solution is (x = 5, y = 0)
theorem solve_system : ∃ x y : ℤ, system1 x y ∧ system2 x y ∧ x = 5 ∧ y = 0 :=
by
  sorry

end solve_system_l217_217636


namespace eccentricity_of_hyperbola_l217_217725

variable {a b c e : ℝ}
variable (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h_vertices : A1 = (-a, 0) ∧ A2 = (a, 0))
variable (h_imaginary_axis : B1 = (0, b) ∧ B2 = (0, -b))
variable (h_foci : F1 = (-c, 0) ∧ F2 = (c, 0))
variable (h_relation : a^2 + b^2 = c^2)
variable (h_tangent_circle : ∀ d, (d = 2*a) → (tangent (circle d) (rhombus F1 B1 F2 B2)))

theorem eccentricity_of_hyperbola : e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_hyperbola_l217_217725


namespace sum_g_h_k_l217_217952

def polynomial_product_constants (d g h k : ℤ) : Prop :=
  ((5 * d^2 + 4 * d + g) * (4 * d^2 + h * d - 5) = 20 * d^4 + 11 * d^3 - 9 * d^2 + k * d - 20)

theorem sum_g_h_k (d g h k : ℤ) (h1 : polynomial_product_constants d g h k) : g + h + k = -16 :=
by
  sorry

end sum_g_h_k_l217_217952


namespace sum_of_samples_is_six_l217_217999

-- Defining the conditions
def grains_varieties : ℕ := 40
def vegetable_oil_varieties : ℕ := 10
def animal_products_varieties : ℕ := 30
def fruits_and_vegetables_varieties : ℕ := 20
def sample_size : ℕ := 20
def total_varieties : ℕ := grains_varieties + vegetable_oil_varieties + animal_products_varieties + fruits_and_vegetables_varieties

def proportion_sample := (sample_size : ℚ) / total_varieties

-- Definitions for the problem
def vegetable_oil_sampled := (vegetable_oil_varieties : ℚ) * proportion_sample
def fruits_and_vegetables_sampled := (fruits_and_vegetables_varieties : ℚ) * proportion_sample

-- Lean 4 statement for the proof problem
theorem sum_of_samples_is_six :
  vegetable_oil_sampled + fruits_and_vegetables_sampled = 6 := by
  sorry

end sum_of_samples_is_six_l217_217999


namespace binom_np_n_mod_p2_l217_217615

   theorem binom_np_n_mod_p2 (p n : ℕ) (hp : Nat.Prime p) : (Nat.choose (n * p) n) % (p ^ 2) = n % (p ^ 2) :=
   by
     sorry
   
end binom_np_n_mod_p2_l217_217615


namespace bobby_truck_gasoline_consumption_rate_l217_217130

variable {initial_gasoline : ℝ}
variable {final_gasoline : ℝ}
variable {dist_to_supermarket : ℝ}
variable {dist_to_farm : ℝ}
variable {dist_into_farm_trip : ℝ}
variable {returned_dist : ℝ}
variable {total_miles_driven : ℝ}
variable {total_gasoline_used : ℝ}
variable {rate_of_consumption : ℝ}

-- Conditions given in the problem
axiom initial_gasoline_is_12 : initial_gasoline = 12
axiom final_gasoline_is_2 : final_gasoline = 2
axiom dist_home_to_supermarket : dist_to_supermarket = 5
axiom dist_home_to_farm : dist_to_farm = 6
axiom dist_home_to_turnaround : dist_into_farm_trip = 2
axiom returned_distance : returned_dist = dist_into_farm_trip * 2

-- Distance calculations based on problem description
def dist_to_supermarket_round_trip : ℝ := dist_to_supermarket * 2
def dist_home_to_turnaround_round_trip : ℝ := returned_dist
def full_farm_trip : ℝ := dist_to_farm

-- Total Distance Calculation
axiom total_distance_is_22 : total_miles_driven = 
  dist_to_supermarket_round_trip + dist_home_to_turnaround_round_trip + full_farm_trip
axiom total_gasoline_used_is_10 : total_gasoline_used = initial_gasoline - final_gasoline

-- Question: Prove the rate of consumption is 2.2 miles per gallon
def rate_of_consumption_calculation (total_miles : ℝ) (total_gas : ℝ) : ℝ :=
  total_miles / total_gas

theorem bobby_truck_gasoline_consumption_rate :
    rate_of_consumption_calculation total_miles_driven total_gasoline_used = 2.2 := 
  sorry

end bobby_truck_gasoline_consumption_rate_l217_217130


namespace tan_pi_minus_alpha_l217_217166

theorem tan_pi_minus_alpha (α : ℝ) (h : Real.tan (Real.pi - α) = -2) : 
  (1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5 / 2) :=
by
  sorry

end tan_pi_minus_alpha_l217_217166


namespace find_a_l217_217432

noncomputable def f (a x : ℝ) : ℝ := a * x * (x - 2)^2

theorem find_a (a : ℝ) (h1 : a ≠ 0)
  (h2 : ∃ x : ℝ, f a x = 32) :
  a = 27 :=
sorry

end find_a_l217_217432


namespace simplify_and_evaluate_l217_217619

-- Define the expression
def expr (a : ℚ) : ℚ := (3 * a - 1) ^ 2 + 3 * a * (3 * a + 2)

-- Given the condition
def a_value : ℚ := -1 / 3

-- State the theorem
theorem simplify_and_evaluate : expr a_value = 3 :=
by
  -- Proof will be added here
  sorry

end simplify_and_evaluate_l217_217619


namespace trevor_coin_difference_l217_217233

theorem trevor_coin_difference:
  ∀ (total_coins quarters: ℕ),
  total_coins = 77 →
  quarters = 29 →
  (total_coins - quarters = 48) := by
  intros total_coins quarters h1 h2
  sorry

end trevor_coin_difference_l217_217233


namespace distance_to_school_l217_217341

theorem distance_to_school : 
  ∀ (d v : ℝ), (d = v * (1 / 3)) → (d = (v + 20) * (1 / 4)) → d = 20 :=
by
  intros d v h1 h2
  sorry

end distance_to_school_l217_217341


namespace polynomial_no_in_interval_l217_217492

theorem polynomial_no_in_interval (P : Polynomial ℤ) (x₁ x₂ x₃ x₄ x₅ : ℤ) :
  (-- Conditions
  P.eval x₁ = 5 ∧ P.eval x₂ = 5 ∧ P.eval x₃ = 5 ∧ P.eval x₄ = 5 ∧ P.eval x₅ = 5 ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
  x₄ ≠ x₅)
  -- No x such that -6 <= P(x) <= 4 or 6 <= P(x) <= 16
  → (∀ x : ℤ, ¬(-6 ≤ P.eval x ∧ P.eval x ≤ 4) ∧ ¬(6 ≤ P.eval x ∧ P.eval x ≤ 16)) :=
by
  intro h
  sorry

end polynomial_no_in_interval_l217_217492


namespace sum_of_possible_ks_l217_217025

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l217_217025


namespace sum_of_roots_l217_217089

theorem sum_of_roots :
  let a := 1
  let b := 10
  let c := -25
  let sum_of_roots := -b / a
  (∀ x, 25 - 10 * x - x ^ 2 = 0 ↔ x ^ 2 + 10 * x - 25 = 0) →
  sum_of_roots = -10 :=
by
  intros
  sorry

end sum_of_roots_l217_217089


namespace sum_of_powers_l217_217761

-- Here is the statement in Lean 4
theorem sum_of_powers (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  (ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68) = (ω^2 - 1) / (ω^4 - 1) :=
sorry -- Proof is omitted as per instructions.

end sum_of_powers_l217_217761


namespace total_dollars_l217_217208

theorem total_dollars (mark_dollars : ℚ) (carolyn_dollars : ℚ) (mark_money : mark_dollars = 7 / 8) (carolyn_money : carolyn_dollars = 2 / 5) :
  mark_dollars + carolyn_dollars = 1.275 := sorry

end total_dollars_l217_217208


namespace evaluate_f_neg3_l217_217393

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_f_neg3 (a b c : ℝ) (h : f 3 a b c = 11) : f (-3) a b c = -9 := by
  sorry

end evaluate_f_neg3_l217_217393


namespace basketball_game_first_half_points_l217_217106

theorem basketball_game_first_half_points (a b r d : ℕ) (H1 : a = b)
  (H2 : a * (1 + r + r^2 + r^3) = 4 * a + 6 * d + 1) 
  (H3 : 15 * a ≤ 100) (H4 : b + (b + d) + b + 2 * d + b + 3 * d < 100) : 
  (a + a * r + b + b + d) = 34 :=
by sorry

end basketball_game_first_half_points_l217_217106


namespace quarters_needed_l217_217521

-- Define the cost of items in cents and declare the number of items to purchase.
def quarter_value : ℕ := 25
def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50

def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Theorem stating the number of quarters needed to buy the given items.
theorem quarters_needed : 
  (num_candy_bars * candy_bar_cost + num_chocolates * chocolate_cost + num_juice_packs * juice_cost) / quarter_value = 11 := 
sorry

end quarters_needed_l217_217521


namespace solve_for_x_l217_217218

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 14.7 -> x = 105 := by
  sorry

end solve_for_x_l217_217218


namespace units_digit_sum_2_pow_a_5_pow_b_l217_217158

theorem units_digit_sum_2_pow_a_5_pow_b (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 100)
  (h2 : 1 ≤ b ∧ b ≤ 100) :
  (2 ^ a + 5 ^ b) % 10 ≠ 8 :=
sorry

end units_digit_sum_2_pow_a_5_pow_b_l217_217158


namespace julien_contribution_l217_217905

def exchange_rate : ℝ := 1.5
def cost_of_pie : ℝ := 12
def lucas_cad : ℝ := 10

theorem julien_contribution : (cost_of_pie - lucas_cad / exchange_rate) = 16 / 3 := by
  sorry

end julien_contribution_l217_217905


namespace sum_of_consecutive_integers_345_l217_217454

-- Definition of the conditions
def is_consecutive_sum (n : ℕ) (k : ℕ) (s : ℕ) : Prop :=
  s = k * n + k * (k - 1) / 2

-- Problem statement
theorem sum_of_consecutive_integers_345 :
  ∃ k_set : Finset ℕ, (∀ k ∈ k_set, k ≥ 2 ∧ ∃ n : ℕ, is_consecutive_sum n k 345) ∧ k_set.card = 6 :=
sorry

end sum_of_consecutive_integers_345_l217_217454
