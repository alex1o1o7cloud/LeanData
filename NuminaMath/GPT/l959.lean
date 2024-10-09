import Mathlib

namespace length_BC_fraction_of_AD_l959_95995

-- Define variables and conditions
variables (x y : ℝ)
variable (h1 : 4 * x = 8 * y) -- given: length of AD from both sides
variable (h2 : 3 * x) -- AB = 3 * BD
variable (h3 : 7 * y) -- AC = 7 * CD

-- State the goal to prove
theorem length_BC_fraction_of_AD (x y : ℝ) (h1 : 4 * x = 8 * y) :
  (y / (4 * x)) = 1 / 8 := by
  sorry

end length_BC_fraction_of_AD_l959_95995


namespace points_satisfy_equation_l959_95931

theorem points_satisfy_equation (x y : ℝ) : 
  (2 * x^2 + 3 * x * y + y^2 + x = 1) ↔ (y = -x - 1) ∨ (y = -2 * x + 1) := by
  sorry

end points_satisfy_equation_l959_95931


namespace find_function_expression_l959_95953

theorem find_function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 - 1) = x^4 + 1) :
  ∀ x : ℝ, x ≥ -1 → f x = x^2 + 2*x + 2 :=
sorry

end find_function_expression_l959_95953


namespace ratio_of_container_volumes_l959_95951

-- Define the volumes of the first and second containers.
variables (A B : ℝ )

-- Hypotheses based on the problem conditions
-- First container is 4/5 full
variable (h1 : A * 4 / 5 = B * 2 / 3)

-- The statement to prove
theorem ratio_of_container_volumes : A / B = 5 / 6 :=
by
  sorry

end ratio_of_container_volumes_l959_95951


namespace cakes_served_at_lunch_today_l959_95933

variable (L : ℕ)
variable (dinnerCakes : ℕ) (yesterdayCakes : ℕ) (totalCakes : ℕ)

theorem cakes_served_at_lunch_today :
  (dinnerCakes = 6) → (yesterdayCakes = 3) → (totalCakes = 14) → (L + dinnerCakes + yesterdayCakes = totalCakes) → L = 5 :=
by
  intros h_dinner h_yesterday h_total h_eq
  sorry

end cakes_served_at_lunch_today_l959_95933


namespace profit_percentage_previous_year_l959_95942

-- Declaring variables
variables (R P : ℝ) -- revenues and profits in the previous year
variable (revenues_1999 := 0.8 * R) -- revenues in 1999
variable (profits_1999 := 0.14 * revenues_1999) -- profits in 1999

-- Given condition: profits in 1999 were 112.00000000000001 percent of the profits in the previous year
axiom profits_ratio : 0.112 * R = 1.1200000000000001 * P

-- Prove the profit as a percentage of revenues in the previous year was 10%
theorem profit_percentage_previous_year : (P / R) * 100 = 10 := by
  sorry

end profit_percentage_previous_year_l959_95942


namespace largest_binomial_coeff_and_rational_terms_l959_95996

theorem largest_binomial_coeff_and_rational_terms 
  (n : ℕ) 
  (h_sum_coeffs : 4^n - 2^n = 992) 
  (T : ℕ → ℝ → ℝ)
  (x : ℝ) :
  (∃ (r1 r2 : ℕ), T r1 x = 270 * x^(22/3) ∧ T r2 x = 90 * x^6)
  ∧
  (∃ (r3 r4 : ℕ), T r3 x = 243 * x^10 ∧ T r4 x = 90 * x^6)
:= 
  
sorry

end largest_binomial_coeff_and_rational_terms_l959_95996


namespace sum_is_24_l959_95911

-- Define the conditions
def A := 3
def B := 7 * A

-- Define the theorem to prove that the sum is 24
theorem sum_is_24 : A + B = 24 :=
by
  -- Adding sorry here since we're not required to provide the proof
  sorry

end sum_is_24_l959_95911


namespace inequality_solution_l959_95900

theorem inequality_solution (a x : ℝ) : 
  (x^2 - (a + 1) * x + a) ≤ 0 ↔ 
  (a > 1 → (1 ≤ x ∧ x ≤ a)) ∧ 
  (a = 1 → x = 1) ∧ 
  (a < 1 → (a ≤ x ∧ x ≤ 1)) :=
by 
  sorry

end inequality_solution_l959_95900


namespace maintain_order_time_l959_95993

theorem maintain_order_time :
  ∀ (x : ℕ), 
  (let ppl_per_min_norm := 9
   let ppl_per_min_cong := 3
   let total_people := 36 
   let teacher_time_saved := 6

   let time_without_order := total_people / ppl_per_min_cong
   let time_with_order := time_without_order - teacher_time_saved

   let ppl_passed_while_order := ppl_per_min_cong * x
   let ppl_passed_norm_order := ppl_per_min_norm * (time_with_order - x)

   ppl_passed_while_order + ppl_passed_norm_order = total_people) → 
  x = 3 :=
sorry

end maintain_order_time_l959_95993


namespace brenda_trays_l959_95914

-- Define main conditions
def cookies_per_tray : ℕ := 80
def cookies_per_box : ℕ := 60
def cost_per_box : ℕ := 350
def total_cost : ℕ := 1400  -- Using cents for calculation to avoid float numbers

-- State the problem
theorem brenda_trays :
  (total_cost / cost_per_box) * cookies_per_box / cookies_per_tray = 3 := 
by
  sorry

end brenda_trays_l959_95914


namespace f_monotonic_intervals_g_not_below_f_inequality_holds_l959_95971

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem f_monotonic_intervals :
  ∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < 1 / 2 → f x < f (x + 1)) ∧ 
    (1 / 2 < x ∧ x < 1 → f x > f (x + 1)) ∧ 
    (1 < x → f x < f (x + 1)) :=
sorry

theorem g_not_below_f :
  ∀ x : ℝ, 0 < x → f x < g x :=
sorry

theorem inequality_holds (n : ℕ) : (2 * n + 1)^2 > 4 * Real.log (Nat.factorial n) :=
sorry

end f_monotonic_intervals_g_not_below_f_inequality_holds_l959_95971


namespace smallest_odd_n_3_product_gt_5000_l959_95904

theorem smallest_odd_n_3_product_gt_5000 :
  ∃ n : ℕ, (∃ k : ℤ, n = 2 * k + 1 ∧ n > 0) ∧ (3 ^ ((n + 1)^2 / 8)) > 5000 ∧ n = 8 :=
by
  sorry

end smallest_odd_n_3_product_gt_5000_l959_95904


namespace parabola_coefficients_l959_95912

theorem parabola_coefficients
    (vertex : (ℝ × ℝ))
    (passes_through : (ℝ × ℝ))
    (vertical_axis_of_symmetry : Prop)
    (hv : vertex = (2, -3))
    (hp : passes_through = (0, 1))
    (has_vertical_axis : vertical_axis_of_symmetry) :
    ∃ a b c : ℝ, ∀ x : ℝ, (x = 0 → (a * x^2 + b * x + c = 1)) ∧ (x = 2 → (a * x^2 + b * x + c = -3)) := sorry

end parabola_coefficients_l959_95912


namespace double_acute_angle_l959_95958

theorem double_acute_angle (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end double_acute_angle_l959_95958


namespace trip_time_difference_l959_95906

theorem trip_time_difference (speed distance1 distance2 : ℕ) (h1 : speed > 0) (h2 : distance2 > distance1) 
  (h3 : speed = 60) (h4 : distance1 = 540) (h5 : distance2 = 570) : 
  (distance2 - distance1) / speed * 60 = 30 := 
by
  sorry

end trip_time_difference_l959_95906


namespace roots_of_eq_l959_95965

theorem roots_of_eq (x : ℝ) : (x - 1) * (x - 2) = 0 ↔ (x = 1 ∨ x = 2) := by
  sorry

end roots_of_eq_l959_95965


namespace find_common_difference_l959_95990

variable (a₁ d : ℝ)

theorem find_common_difference
  (h1 : a₁ + (a₁ + 6 * d) = 22)
  (h2 : (a₁ + 3 * d) + (a₁ + 9 * d) = 40) :
  d = 3 := by
  sorry

end find_common_difference_l959_95990


namespace softball_team_total_players_l959_95947

theorem softball_team_total_players 
  (M W : ℕ) 
  (h1 : W = M + 4)
  (h2 : (M : ℚ) / (W : ℚ) = 0.6666666666666666) :
  M + W = 20 :=
by sorry

end softball_team_total_players_l959_95947


namespace certain_number_N_l959_95936

theorem certain_number_N (G N : ℕ) (hG : G = 127)
  (h₁ : ∃ k : ℕ, N = G * k + 10)
  (h₂ : ∃ m : ℕ, 2045 = G * m + 13) :
  N = 2042 :=
sorry

end certain_number_N_l959_95936


namespace trace_ellipse_l959_95997

open Complex

theorem trace_ellipse (z : ℂ) (θ : ℝ) (h₁ : z = 3 * exp (θ * I))
  (h₂ : abs z = 3) : ∃ a b : ℝ, ∀ θ, z + 1/z = a * Real.cos θ + b * (I * Real.sin θ) :=
sorry

end trace_ellipse_l959_95997


namespace sample_size_is_150_l959_95937

theorem sample_size_is_150 
  (classes : ℕ) (students_per_class : ℕ) (selected_students : ℕ)
  (h1 : classes = 40) (h2 : students_per_class = 50) (h3 : selected_students = 150)
  : selected_students = 150 :=
sorry

end sample_size_is_150_l959_95937


namespace geometric_common_ratio_l959_95970

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_common_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : q > 0) 
  (h2 : geometric_seq a q) (h3 : a 3 * a 7 = 4 * (a 4)^2) : q = 2 := 
by 
  sorry

end geometric_common_ratio_l959_95970


namespace original_number_l959_95946

variable (n : ℝ)

theorem original_number :
  (2 * (n + 3)^2 - 3) / 2 = 49 → n = Real.sqrt (101 / 2) - 3 :=
by
  sorry

end original_number_l959_95946


namespace percentage_of_sikh_boys_l959_95984

-- Define the conditions
def total_boys : ℕ := 850
def percentage_muslim_boys : ℝ := 0.46
def percentage_hindu_boys : ℝ := 0.28
def boys_other_communities : ℕ := 136

-- Theorem to prove the percentage of Sikh boys is 10%
theorem percentage_of_sikh_boys : 
  (((total_boys - 
      (percentage_muslim_boys * total_boys + 
       percentage_hindu_boys * total_boys + 
       boys_other_communities))
    / total_boys) * 100 = 10) :=
by
  -- sorry prevents the need to provide proof here
  sorry

end percentage_of_sikh_boys_l959_95984


namespace simplify_and_evaluate_l959_95943

noncomputable def a := 2 * Real.sin (Real.pi / 4) + (1 / 2) ^ (-1 : ℤ)

theorem simplify_and_evaluate :
  (a^2 - 4) / a / ((4 * a - 4) / a - a) + 2 / (a - 2) = -1 - Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l959_95943


namespace find_k_l959_95949

theorem find_k {k : ℚ} (h : (3 : ℚ)^3 + 7 * (3 : ℚ)^2 + k * (3 : ℚ) + 23 = 0) : k = -113 / 3 :=
by
  sorry

end find_k_l959_95949


namespace total_green_peaches_l959_95961

-- Define the known conditions
def baskets : ℕ := 7
def green_peaches_per_basket : ℕ := 2

-- State the problem and the proof goal
theorem total_green_peaches : baskets * green_peaches_per_basket = 14 := by
  -- Provide a proof here
  sorry

end total_green_peaches_l959_95961


namespace gcd_280_2155_l959_95908

theorem gcd_280_2155 : Nat.gcd 280 2155 = 35 := 
sorry

end gcd_280_2155_l959_95908


namespace problem_1_problem_2_problem_3_l959_95940

-- Definition and proof state for problem 1
theorem problem_1 (a b m n : ℕ) (h₀ : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

-- Definition and proof state for problem 2
theorem problem_2 (a m n : ℕ) (h₀ : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = 13 ∨ a = 7 := by
  sorry

-- Definition and proof state for problem 3
theorem problem_3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
  sorry

end problem_1_problem_2_problem_3_l959_95940


namespace positive_irrational_less_than_one_l959_95975

theorem positive_irrational_less_than_one : 
  ∃! (x : ℝ), 
    (x = (Real.sqrt 6) / 3 ∧ Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = -(Real.sqrt 3) / 3 ∧ Irrational x ∧ x < 0) ∨ 
    (x = 1 / 3 ∧ ¬Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = Real.pi / 3 ∧ Irrational x ∧ x > 1) :=
by
  sorry

end positive_irrational_less_than_one_l959_95975


namespace novel_pages_l959_95919

theorem novel_pages (x : ℕ)
  (h1 : x - ((1 / 6 : ℝ) * x + 10) = (5 / 6 : ℝ) * x - 10)
  (h2 : (5 / 6 : ℝ) * x - 10 - ((1 / 5 : ℝ) * ((5 / 6 : ℝ) * x - 10) + 20) = (2 / 3 : ℝ) * x - 28)
  (h3 : (2 / 3 : ℝ) * x - 28 - ((1 / 4 : ℝ) * ((2 / 3 : ℝ) * x - 28) + 25) = (1 / 2 : ℝ) * x - 46) :
  (1 / 2 : ℝ) * x - 46 = 80 → x = 252 :=
by
  sorry

end novel_pages_l959_95919


namespace num_ways_first_to_fourth_floor_l959_95979

theorem num_ways_first_to_fourth_floor (floors : ℕ) (staircases_per_floor : ℕ) 
  (H_floors : floors = 4) (H_staircases : staircases_per_floor = 2) : 
  (staircases_per_floor) ^ (floors - 1) = 2^3 := 
by 
  sorry

end num_ways_first_to_fourth_floor_l959_95979


namespace smallest_solution_of_equation_l959_95994

theorem smallest_solution_of_equation : 
    ∃ x : ℝ, x*|x| = 3 * x - 2 ∧ 
            ∀ y : ℝ, y*|y| = 3 * y - 2 → x ≤ y :=
sorry

end smallest_solution_of_equation_l959_95994


namespace ab_cd_eq_neg_37_over_9_l959_95974

theorem ab_cd_eq_neg_37_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a + b + d = 2)
  (h3 : a + c + d = 3)
  (h4 : b + c + d = -3) :
  a * b + c * d = -37 / 9 := by
  sorry

end ab_cd_eq_neg_37_over_9_l959_95974


namespace length_of_bridge_correct_l959_95986

noncomputable def L_train : ℝ := 180
noncomputable def v_km_per_hr : ℝ := 60  -- speed in km/hr
noncomputable def t : ℝ := 25

-- Convert speed from km/hr to m/s
noncomputable def km_per_hr_to_m_per_s (v: ℝ) : ℝ := v * (1000 / 3600)
noncomputable def v : ℝ := km_per_hr_to_m_per_s v_km_per_hr

-- Distance covered by the train while crossing the bridge
noncomputable def d : ℝ := v * t

-- Length of the bridge
noncomputable def L_bridge : ℝ := d - L_train

theorem length_of_bridge_correct :
  L_bridge = 236.75 :=
  by
    sorry

end length_of_bridge_correct_l959_95986


namespace find_percentage_second_alloy_l959_95989

open Real

def percentage_copper_second_alloy (percentage_alloy1: ℝ) (ounces_alloy1: ℝ) (percentage_desired_alloy: ℝ) (total_ounces: ℝ) (percentage_second_alloy: ℝ) : Prop :=
  let copper_ounces_alloy1 := percentage_alloy1 * ounces_alloy1 / 100
  let desired_copper_ounces := percentage_desired_alloy * total_ounces / 100
  let needed_copper_ounces := desired_copper_ounces - copper_ounces_alloy1
  let ounces_alloy2 := total_ounces - ounces_alloy1
  (needed_copper_ounces / ounces_alloy2) * 100 = percentage_second_alloy

theorem find_percentage_second_alloy :
  percentage_copper_second_alloy 18 45 19.75 108 21 :=
by
  sorry

end find_percentage_second_alloy_l959_95989


namespace vasya_faster_than_petya_l959_95978

theorem vasya_faster_than_petya 
  (L : ℝ) (v : ℝ) (x : ℝ) (t : ℝ) 
  (meeting_condition : (v + x * v) * t = L)
  (petya_lap : v * t = L)
  (vasya_meet_petya_after_lap : x * v * t = 2 * L) :
  x = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end vasya_faster_than_petya_l959_95978


namespace anthony_more_shoes_than_jim_l959_95905

def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

theorem anthony_more_shoes_than_jim : (anthony_shoes - jim_shoes) = 2 :=
by
  sorry

end anthony_more_shoes_than_jim_l959_95905


namespace jenny_hours_left_l959_95945

theorem jenny_hours_left 
    (h_research : ℕ := 10)
    (h_proposal : ℕ := 2)
    (h_visual_aids : ℕ := 5)
    (h_editing : ℕ := 3)
    (h_total : ℕ := 25) :
    h_total - (h_research + h_proposal + h_visual_aids + h_editing) = 5 := by
  sorry

end jenny_hours_left_l959_95945


namespace intersection_range_of_b_l959_95985

theorem intersection_range_of_b (b : ℝ) :
  (∀ (m : ℝ), ∃ (x y : ℝ), x^2 + 2 * y^2 = 3 ∧ y = m * x + b) ↔ 
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := 
sorry

end intersection_range_of_b_l959_95985


namespace correct_division_result_l959_95920

theorem correct_division_result (x : ℝ) 
  (h : (x - 14) / 5 = 11) : (x - 5) / 7 = 64 / 7 :=
by
  sorry

end correct_division_result_l959_95920


namespace bah_rah_yah_equiv_l959_95969

-- We define the initial equivalences given in the problem statement.
theorem bah_rah_yah_equiv (bahs rahs yahs : ℕ) :
  (18 * bahs = 30 * rahs) ∧
  (12 * rahs = 20 * yahs) →
  (1200 * yahs = 432 * bahs) :=
by
  -- Placeholder for the actual proof
  sorry

end bah_rah_yah_equiv_l959_95969


namespace correct_operation_l959_95999

/-- Proving that among the given mathematical operations, only the second option is correct. -/
theorem correct_operation (m : ℝ) : ¬ (m^3 - m^2 = m) ∧ (3 * m^2 * 2 * m^3 = 6 * m^5) ∧ ¬ (3 * m^2 + 2 * m^3 = 5 * m^5) ∧ ¬ ((2 * m^2)^3 = 8 * m^5) :=
by
  -- These are the conditions, proof is omitted using sorry
  sorry

end correct_operation_l959_95999


namespace combination_seven_choose_three_l959_95948

-- Define the combination formula
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the problem-specific values
def n : ℕ := 7
def k : ℕ := 3

-- Problem statement: Prove that the number of combinations of 3 toppings from 7 is 35
theorem combination_seven_choose_three : combination 7 3 = 35 :=
  by
    sorry

end combination_seven_choose_three_l959_95948


namespace smallest_number_increased_by_seven_divisible_by_37_47_53_l959_95921

theorem smallest_number_increased_by_seven_divisible_by_37_47_53 : 
  ∃ n : ℕ, (n + 7) % 37 = 0 ∧ (n + 7) % 47 = 0 ∧ (n + 7) % 53 = 0 ∧ n = 92160 :=
by
  sorry

end smallest_number_increased_by_seven_divisible_by_37_47_53_l959_95921


namespace min_throws_to_same_sum_l959_95967

/-- Define the set of possible sums for four six-sided dice --/
def dice_sum_range := {s : ℕ | 4 ≤ s ∧ s ≤ 24}

/-- The total number of possible sums when rolling four six-sided dice --/
def num_possible_sums : ℕ := 24 - 4 + 1

/-- 
  The minimum number of throws required to ensure that the same sum appears at least twice 
  by the Pigeonhole principle.
--/
theorem min_throws_to_same_sum : num_possible_sums + 1 = 22 := by
  sorry

end min_throws_to_same_sum_l959_95967


namespace mean_books_read_l959_95932

theorem mean_books_read :
  let readers1 := 4
  let books1 := 3
  let readers2 := 5
  let books2 := 5
  let readers3 := 2
  let books3 := 7
  let readers4 := 1
  let books4 := 10
  let total_readers := readers1 + readers2 + readers3 + readers4
  let total_books := (readers1 * books1) + (readers2 * books2) + (readers3 * books3) + (readers4 * books4)
  let mean_books := total_books / total_readers
  mean_books = 5.0833 :=
by
  sorry

end mean_books_read_l959_95932


namespace ratio_of_money_l959_95972

-- Conditions
def amount_given := 14
def cost_of_gift := 28

-- Theorem statement to prove
theorem ratio_of_money (h1 : amount_given = 14) (h2 : cost_of_gift = 28) :
  amount_given / cost_of_gift = 1 / 2 := by
  sorry

end ratio_of_money_l959_95972


namespace trig_identity_l959_95903

theorem trig_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (α - 15 * Real.pi / 180) + Real.cos (105 * Real.pi / 180 - α) = -2 / 3 :=
sorry

end trig_identity_l959_95903


namespace carrey_fixed_amount_l959_95901

theorem carrey_fixed_amount :
  ∃ C : ℝ, 
    (C + 0.25 * 44.44444444444444 = 24 + 0.16 * 44.44444444444444) →
    C = 20 :=
by
  sorry

end carrey_fixed_amount_l959_95901


namespace alyssa_total_spent_l959_95952

theorem alyssa_total_spent :
  let grapes := 12.08
  let cherries := 9.85
  grapes + cherries = 21.93 := by
  sorry

end alyssa_total_spent_l959_95952


namespace cosine_squared_identity_l959_95909

theorem cosine_squared_identity (α : ℝ) (h : Real.sin (2 * α) = 1 / 3) : Real.cos (α - (π / 4)) ^ 2 = 2 / 3 :=
sorry

end cosine_squared_identity_l959_95909


namespace intersection_of_P_and_Q_l959_95922

theorem intersection_of_P_and_Q (P Q : Set ℕ) (h1 : P = {1, 3, 6, 9}) (h2 : Q = {1, 2, 4, 6, 8}) :
  P ∩ Q = {1, 6} :=
by
  sorry

end intersection_of_P_and_Q_l959_95922


namespace abs_sum_l959_95944

theorem abs_sum (a b c : ℚ) (h₁ : a = -1/4) (h₂ : b = -2) (h₃ : c = -11/4) :
  |a| + |b| - |c| = -1/2 :=
by {
  sorry
}

end abs_sum_l959_95944


namespace ladder_geometric_sequence_solution_l959_95966

-- A sequence {aₙ} is a 3rd-order ladder geometric sequence given by a_{n+3}^2 = a_n * a_{n+6} for any positive integer n
def ladder_geometric_3rd_order (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 3) ^ 2 = a n * a (n + 6)

-- Initial conditions
def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 4 = 2

-- Main theorem to be proven in Lean 4
theorem ladder_geometric_sequence_solution :
  ∃ a : ℕ → ℝ, ladder_geometric_3rd_order a ∧ initial_conditions a ∧ a 10 = 8 :=
by
  sorry

end ladder_geometric_sequence_solution_l959_95966


namespace tangents_product_is_constant_MN_passes_fixed_point_l959_95907

-- Define the parabola C and the tangency conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

variables {x1 y1 x2 y2 : ℝ}

-- Point G is on the axis of the parabola C (we choose the y-axis for part 2)
def point_G_on_axis (G : ℝ × ℝ) : Prop := G.2 = -1

-- Two tangent points from G to the parabola at A (x1, y1) and B (x2, y2)
def tangent_points (G : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂

-- Question 1 proof statement
theorem tangents_product_is_constant (G : ℝ × ℝ) (hG : point_G_on_axis G)
  (hT : tangent_points G x1 y1 x2 y2) : x1 * x2 + y1 * y2 = -3 := sorry

variables {M N : ℝ × ℝ}

-- Question 2 proof statement
theorem MN_passes_fixed_point {G : ℝ × ℝ} (hG : G.1 = 0) (xM yM xN yN : ℝ)
 (hMA : parabola M.1 M.2) (hMB : parabola N.1 N.2)
 (h_perpendicular : (M.1 - G.1) * (N.1 - G.1) + (M.2 - G.2) * (N.2 - G.2) = 0)
 : ∃ P, P = (2, 5) := sorry

end tangents_product_is_constant_MN_passes_fixed_point_l959_95907


namespace area_is_25_l959_95938

noncomputable def area_of_square (x : ℝ) : ℝ :=
  let side1 := 5 * x - 20
  let side2 := 25 - 4 * x
  if h : side1 = side2 then 
    side1 * side1
  else 
    0

theorem area_is_25 (x : ℝ) (h_eq : 5 * x - 20 = 25 - 4 * x) : area_of_square x = 25 :=
by
  sorry

end area_is_25_l959_95938


namespace percentage_less_l959_95930

theorem percentage_less (P T J : ℝ) (hT : T = 0.9375 * P) (hJ : J = 0.8 * T) : (P - J) / P * 100 = 25 :=
by
  sorry

end percentage_less_l959_95930


namespace bus_capacity_fraction_l959_95927

theorem bus_capacity_fraction
  (capacity : ℕ)
  (x : ℚ)
  (return_fraction : ℚ)
  (total_people : ℕ)
  (capacity_eq : capacity = 200)
  (return_fraction_eq : return_fraction = 4/5)
  (total_people_eq : total_people = 310)
  (people_first_trip_eq : 200 * x + 200 * 4/5 = 310) :
  x = 3/4 :=
by
  sorry

end bus_capacity_fraction_l959_95927


namespace tom_age_ratio_l959_95968

theorem tom_age_ratio (T N : ℕ)
  (sum_children : T = T) 
  (age_condition : T - N = 3 * (T - 4 * N)) :
  T / N = 11 / 2 := 
sorry

end tom_age_ratio_l959_95968


namespace granola_bars_relation_l959_95923

theorem granola_bars_relation (x y z : ℕ) (h1 : z = x / (3 * y)) : z = x / (3 * y) :=
by {
    sorry
}

end granola_bars_relation_l959_95923


namespace intersection_M_N_l959_95956

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 + 2*x - 3 ≤ 0}

theorem intersection_M_N :
  M ∩ N = {-2, -1, 0, 1} :=
by
  sorry

end intersection_M_N_l959_95956


namespace correct_operation_l959_95964

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b :=
by
  sorry

end correct_operation_l959_95964


namespace inequality_a_squared_plus_b_squared_l959_95916

variable (a b : ℝ)

theorem inequality_a_squared_plus_b_squared (h : a > b) : a^2 + b^2 > ab := 
sorry

end inequality_a_squared_plus_b_squared_l959_95916


namespace circle_equation_has_valid_k_l959_95926

theorem circle_equation_has_valid_k (k : ℝ) : (∃ a b r : ℝ, r > 0 ∧ ∀ x y : ℝ, (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) ↔ k < 5 / 4 := by
  sorry

end circle_equation_has_valid_k_l959_95926


namespace A_inter_complement_B_eq_01_l959_95950

open Set

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | x ≥ 1}
def complement_B : Set ℝ := U \ B

theorem A_inter_complement_B_eq_01 : A ∩ complement_B = (Set.Ioo 0 1) := 
by 
  sorry

end A_inter_complement_B_eq_01_l959_95950


namespace not_in_range_g_zero_l959_95957

noncomputable def g (x: ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else 0 -- g(x) is not defined at x = -3, this is a placeholder

theorem not_in_range_g_zero :
  ¬ (∃ x : ℝ, g x = 0) :=
sorry

end not_in_range_g_zero_l959_95957


namespace find_a7_a8_l959_95955

variable {R : Type*} [LinearOrderedField R]
variable {a : ℕ → R}

-- Conditions
def cond1 : a 1 + a 2 = 40 := sorry
def cond2 : a 3 + a 4 = 60 := sorry

-- Goal 
theorem find_a7_a8 : a 7 + a 8 = 135 := 
by 
  -- provide the actual proof here
  sorry

end find_a7_a8_l959_95955


namespace arithmetic_seq_a3_value_l959_95954

-- Given the arithmetic sequence {a_n}, where
-- a_1 + a_2 + a_3 + a_4 + a_5 = 20
def arithmetic_seq (a : ℕ → ℝ) := ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_seq_a3_value {a : ℕ → ℝ}
    (h_seq : arithmetic_seq a)
    (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
by
  sorry

end arithmetic_seq_a3_value_l959_95954


namespace sub_neg_four_l959_95913

theorem sub_neg_four : -3 - 1 = -4 :=
by
  sorry

end sub_neg_four_l959_95913


namespace max_equilateral_triangle_area_l959_95991

theorem max_equilateral_triangle_area (length width : ℝ) (h_len : length = 15) (h_width : width = 12) 
: ∃ (area : ℝ), area = 200.25 * Real.sqrt 3 - 450 := by
  sorry

end max_equilateral_triangle_area_l959_95991


namespace only_B_is_like_terms_l959_95962

def is_like_terms (terms : List (String × String)) : List Bool :=
  let like_term_checker := fun (term1 term2 : String) =>
    -- The function to check if two terms are like terms
    sorry
  terms.map (fun (term1, term2) => like_term_checker term1 term2)

theorem only_B_is_like_terms :
  is_like_terms [("−2x^3", "−3x^2"), ("−(1/4)ab", "18ba"), ("a^2b", "−ab^2"), ("4m", "6mn")] =
  [false, true, false, false] :=
by
  sorry

end only_B_is_like_terms_l959_95962


namespace lesser_solution_is_minus_15_l959_95917

noncomputable def lesser_solution : ℤ := -15

theorem lesser_solution_is_minus_15 :
  ∃ x y : ℤ, x^2 + 10 * x - 75 = 0 ∧ y^2 + 10 * y - 75 = 0 ∧ x < y ∧ x = lesser_solution :=
by 
  sorry

end lesser_solution_is_minus_15_l959_95917


namespace graph_equation_l959_95902

theorem graph_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end graph_equation_l959_95902


namespace necessary_but_not_sufficient_l959_95981

theorem necessary_but_not_sufficient (a : ℝ) : (a - 1 < 0 ↔ a < 1) ∧ (|a| < 1 → a < 1) ∧ ¬ (a < 1 → |a| < 1) := by
  sorry

end necessary_but_not_sufficient_l959_95981


namespace find_f_neg_one_l959_95915

theorem find_f_neg_one (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_symm : ∀ x, f (4 - x) = -f x)
  (h_f3 : f 3 = 3) :
  f (-1) = 3 := 
sorry

end find_f_neg_one_l959_95915


namespace find_n_divisibility_l959_95983

theorem find_n_divisibility :
  ∃ n : ℕ, n < 10 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 11 = 0 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 5 = 0 :=
by
  use 3
  sorry

end find_n_divisibility_l959_95983


namespace length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l959_95941

noncomputable def spring_length (x : ℝ) : ℝ :=
  2 * x + 18

-- Problem (1)
theorem length_at_4kg : (spring_length 4) = 26 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (2)
theorem length_increases_by_2 : ∀ (x y : ℝ), y = x + 1 → (spring_length y) = (spring_length x) + 2 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (3)
theorem relationship_linear : ∃ (k b : ℝ), (∀ x, spring_length x = k * x + b) ∧ k = 2 ∧ b = 18 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (4)
theorem length_at_12kg : (spring_length 12) = 42 :=
  by
    -- The complete proof is omitted.
    sorry

end length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l959_95941


namespace supplementary_angles_difference_l959_95982
-- Import necessary libraries

-- Define the conditions
def are_supplementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 180

def ratio_7_2 (θ₁ θ₂ : ℝ) : Prop := θ₁ / θ₂ = 7 / 2

-- State the theorem
theorem supplementary_angles_difference (θ₁ θ₂ : ℝ) 
  (h_supp : are_supplementary θ₁ θ₂) 
  (h_ratio : ratio_7_2 θ₁ θ₂) :
  |θ₁ - θ₂| = 100 :=
by
  sorry

end supplementary_angles_difference_l959_95982


namespace water_displaced_volume_square_l959_95998

-- Given conditions:
def radius : ℝ := 5
def height : ℝ := 10
def cube_side : ℝ := 6

-- Theorem statement for the problem
theorem water_displaced_volume_square (r h s : ℝ) (w : ℝ) 
  (hr : r = 5) 
  (hh : h = 10) 
  (hs : s = 6) : 
  (w * w) = 13141.855 :=
by 
  sorry

end water_displaced_volume_square_l959_95998


namespace hyperbola_equation_l959_95960

-- Definitions for a given hyperbola
variables {a b : ℝ}
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Definitions for the asymptote condition
axiom point_on_asymptote : (4 : ℝ) = (b / a) * 3

-- Definitions for the focal distance condition
axiom point_circle_intersect : (3 : ℝ)^2 + 4^2 = (a^2 + b^2)

-- The goal is to prove the hyperbola's specific equation
theorem hyperbola_equation : 
  (a^2 = 9 ∧ b^2 = 16) →
  (∃ a b : ℝ, (4 : ℝ)^2 + 3^2 = (a^2 + b^2) ∧ 
               (4 : ℝ) = (b / a) * 3 ∧ 
               ((a^2 = 9) ∧ (b^2 = 16)) ∧ (a > 0) ∧ (b > 0)) :=
sorry

end hyperbola_equation_l959_95960


namespace Amelia_wins_probability_correct_l959_95925

-- Define the probabilities
def probability_Amelia_heads := 1 / 3
def probability_Blaine_heads := 2 / 5

-- The infinite geometric series sum calculation for Amelia to win
def probability_Amelia_wins :=
  probability_Amelia_heads * (1 / (1 - (1 - probability_Amelia_heads) * (1 - probability_Blaine_heads)))

-- Given values p and q from the conditions
def p := 5
def q := 9

-- The correct answer $\frac{5}{9}$
def Amelia_wins_correct := 5 / 9

-- Prove that the probability calculation matches the given $\frac{5}{9}$, and find q - p
theorem Amelia_wins_probability_correct :
  probability_Amelia_wins = Amelia_wins_correct ∧ q - p = 4 := by
  sorry

end Amelia_wins_probability_correct_l959_95925


namespace max_sum_of_positives_l959_95934

theorem max_sum_of_positives (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + 1 / x + 1 / y = 5) : x + y ≤ 4 :=
sorry

end max_sum_of_positives_l959_95934


namespace width_of_wall_l959_95929

-- Define the dimensions of a single brick.
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of bricks.
def num_bricks : ℝ := 6800

-- Define the dimensions of the wall (length and height).
def wall_length : ℝ := 850
def wall_height : ℝ := 600

-- Prove that the width of the wall is 22.5 cm.
theorem width_of_wall : 
  (wall_length * wall_height * 22.5 = num_bricks * (brick_length * brick_width * brick_height)) :=
by
  sorry

end width_of_wall_l959_95929


namespace prove_problem_l959_95918

noncomputable def proof_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : Prop :=
  (1 + 1 / x) * (1 + 1 / y) ≥ 9

theorem prove_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : proof_problem x y hx hy h :=
  sorry

end prove_problem_l959_95918


namespace projection_inequality_l959_95935

-- Define the problem with given Cartesian coordinate system, finite set of points in space, and their orthogonal projections
variable (O_xyz : Type) -- Cartesian coordinate system
variable (S : Finset O_xyz) -- finite set of points in space
variable (S_x S_y S_z : Finset O_xyz) -- sets of orthogonal projections onto the planes

-- Define the orthogonal projections (left as a comment here since detailed implementation is not specified)
-- (In Lean, actual definitions of orthogonal projections would follow mathematical and geometric definitions)

-- State the theorem to be proved
theorem projection_inequality :
  (Finset.card S) ^ 2 ≤ (Finset.card S_x) * (Finset.card S_y) * (Finset.card S_z) := 
sorry

end projection_inequality_l959_95935


namespace puffy_muffy_total_weight_l959_95963

theorem puffy_muffy_total_weight (scruffy_weight muffy_weight puffy_weight : ℕ)
  (h1 : scruffy_weight = 12)
  (h2 : muffy_weight = scruffy_weight - 3)
  (h3 : puffy_weight = muffy_weight + 5) :
  puffy_weight + muffy_weight = 23 := by
  sorry

end puffy_muffy_total_weight_l959_95963


namespace parametrize_line_l959_95987

theorem parametrize_line (s h : ℝ) :
    s = -5/2 ∧ h = 20 → ∀ t : ℝ, ∃ x y : ℝ, 4 * x + 7 = y ∧ 
    (x = s + 5 * t ∧ y = -3 + h * t) :=
by
  sorry

end parametrize_line_l959_95987


namespace chosen_number_l959_95924

theorem chosen_number (x : ℤ) (h : 2 * x - 138 = 110) : x = 124 :=
sorry

end chosen_number_l959_95924


namespace question_statement_l959_95973

-- Definitions based on conditions
def all_cards : List ℕ := [8, 3, 6, 5, 0, 7]
def A : ℕ := 876  -- The largest number from the given cards.
def B : ℕ := 305  -- The smallest number from the given cards with non-zero hundreds place.

-- The proof problem statement
theorem question_statement :
  (A - B) * 6 = 3426 := by
  sorry

end question_statement_l959_95973


namespace abs_sum_sequence_l959_95939

def S (n : ℕ) : ℤ := n^2 - 4 * n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem abs_sum_sequence (h : ∀ n, S n = n^2 - 4 * n) :
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|) = 68 :=
by
  sorry

end abs_sum_sequence_l959_95939


namespace complement_complement_l959_95928

theorem complement_complement (alpha : ℝ) (h : alpha = 35) : (90 - (90 - alpha)) = 35 := by
  -- proof goes here, but we write sorry to skip it
  sorry

end complement_complement_l959_95928


namespace triangle_inequality_check_l959_95976

theorem triangle_inequality_check :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔
    ((a = 6 ∧ b = 9 ∧ c = 14) ∨ (a = 9 ∧ b = 6 ∧ c = 14) ∨ (a = 6 ∧ b = 14 ∧ c = 9) ∨
     (a = 14 ∧ b = 6 ∧ c = 9) ∨ (a = 9 ∧ b = 14 ∧ c = 6) ∨ (a = 14 ∧ b = 9 ∧ c = 6)) := sorry

end triangle_inequality_check_l959_95976


namespace male_red_ants_percentage_l959_95988

noncomputable def percentage_of_total_ant_population_that_are_red_females (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    (percent_red_are_females / 100) * red_population_pct

noncomputable def percentage_of_total_ant_population_that_are_red_males (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    red_population_pct - percentage_of_total_ant_population_that_are_red_females total_population_pct red_population_pct percent_red_are_females

theorem male_red_ants_percentage (total_population_pct red_population_pct percent_red_are_females male_red_ants_pct : ℝ) :
    red_population_pct = 85 → percent_red_are_females = 45 → male_red_ants_pct = 46.75 →
    percentage_of_total_ant_population_that_are_red_males total_population_pct red_population_pct percent_red_are_females = male_red_ants_pct :=
by
sorry

end male_red_ants_percentage_l959_95988


namespace total_germs_calculation_l959_95980

def number_of_dishes : ℕ := 10800
def germs_per_dish : ℕ := 500
def total_germs : ℕ := 5400000

theorem total_germs_calculation : germs_per_ddish * number_of_idshessh = total_germs :=
by sorry

end total_germs_calculation_l959_95980


namespace probability_neither_cake_nor_muffin_l959_95992

noncomputable def probability_of_neither (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) : ℚ :=
  (total - (cake + muffin - both)) / total

theorem probability_neither_cake_nor_muffin
  (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) (h_total : total = 100)
  (h_cake : cake = 50) (h_muffin : muffin = 40) (h_both : both = 18) :
  probability_of_neither total cake muffin both = 0.28 :=
by
  rw [h_total, h_cake, h_muffin, h_both]
  norm_num
  sorry

end probability_neither_cake_nor_muffin_l959_95992


namespace number_of_strawberries_stolen_l959_95977

-- Define the conditions
def daily_harvest := 5
def days_in_april := 30
def strawberries_given_away := 20
def strawberries_left_by_end := 100

-- Calculate total harvested strawberries
def total_harvest := daily_harvest * days_in_april
-- Calculate strawberries after giving away
def remaining_after_giveaway := total_harvest - strawberries_given_away

-- Prove the number of strawberries stolen
theorem number_of_strawberries_stolen : remaining_after_giveaway - strawberries_left_by_end = 30 := by
  sorry

end number_of_strawberries_stolen_l959_95977


namespace area_of_region_bounded_by_sec_and_csc_l959_95959

theorem area_of_region_bounded_by_sec_and_csc (x y : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ 0 ≤ x ∧ 0 ≤ y) → 
  (∃ (area : ℝ), area = 1) :=
by 
  sorry

end area_of_region_bounded_by_sec_and_csc_l959_95959


namespace total_people_l959_95910

-- Definitions of the given conditions
variable (I N B Ne T : ℕ)

-- These variables represent the given conditions
axiom h1 : I = 25
axiom h2 : N = 23
axiom h3 : B = 21
axiom h4 : Ne = 23

-- The theorem we want to prove
theorem total_people : T = 50 :=
by {
  sorry -- We denote the skipping of proof details.
}

end total_people_l959_95910
