import Mathlib

namespace total_animals_counted_l776_77687

theorem total_animals_counted :
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  show (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605
  sorry

end total_animals_counted_l776_77687


namespace central_angle_double_score_l776_77627

theorem central_angle_double_score 
  (prob: ℚ)
  (total_angle: ℚ)
  (num_regions: ℚ)
  (eq_regions: ℚ → Prop)
  (double_score_prob: prob = 1/8)
  (total_angle_eq: total_angle = 360)
  (num_regions_eq: num_regions = 6) 
  : ∃ x: ℚ, (prob = x / total_angle) → x = 45 :=
by
  sorry

end central_angle_double_score_l776_77627


namespace four_consecutive_product_divisible_by_12_l776_77602

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l776_77602


namespace rectangle_area_l776_77626

theorem rectangle_area (w L : ℝ) (h1 : L = w^2) (h2 : L + w = 25) : 
  L * w = (Real.sqrt 101 - 1)^3 / 8 := 
sorry

end rectangle_area_l776_77626


namespace greene_family_admission_cost_l776_77650

theorem greene_family_admission_cost (x : ℝ) (h1 : ∀ y : ℝ, y = x - 13) (h2 : ∀ z : ℝ, z = x + (x - 13)) :
  x = 45 :=
by
  sorry

end greene_family_admission_cost_l776_77650


namespace pentagon_perpendicular_sums_l776_77667

noncomputable def FO := 2
noncomputable def FQ := 2
noncomputable def FR := 2

theorem pentagon_perpendicular_sums :
  FO + FQ + FR = 6 :=
by
  sorry

end pentagon_perpendicular_sums_l776_77667


namespace suraj_average_increase_l776_77604

namespace SurajAverage

theorem suraj_average_increase (A : ℕ) (h : (16 * A + 112) / 17 = A + 6) : (A + 6) = 16 :=
  by
  sorry

end SurajAverage

end suraj_average_increase_l776_77604


namespace intersection_complement_l776_77684

def U : Set ℤ := Set.univ
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≠ x}
def C_U_B : Set ℤ := {x | x ≠ 0 ∧ x ≠ 1}

theorem intersection_complement :
  A ∩ C_U_B = {-1, 2} :=
by
  sorry

end intersection_complement_l776_77684


namespace simplify_expression_l776_77663

theorem simplify_expression (z : ℝ) : (3 - 5*z^2) - (4 + 3*z^2) = -1 - 8*z^2 :=
by
  sorry

end simplify_expression_l776_77663


namespace expression_evaluation_l776_77641

open Rat

theorem expression_evaluation :
  ∀ (a b c : ℚ),
  c = b - 4 →
  b = a + 4 →
  a = 3 →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 10) / (c + 7) = 117 / 40 :=
by
  intros a b c hc hb ha h1 h2 h3
  simp [hc, hb, ha]
  have h1 : 3 + 1 ≠ 0 := by norm_num
  have h2 : 7 - 3 ≠ 0 := by norm_num
  have h3 : 3 + 7 ≠ 0 := by norm_num
  -- Placeholder for the simplified expression computation
  sorry

end expression_evaluation_l776_77641


namespace randy_blocks_left_l776_77645

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

end randy_blocks_left_l776_77645


namespace find_y_l776_77693

theorem find_y 
  (h : (5 + 8 + 17) / 3 = (12 + y) / 2) : y = 8 :=
sorry

end find_y_l776_77693


namespace semicircle_radius_in_trapezoid_l776_77619

theorem semicircle_radius_in_trapezoid 
  (AB CD : ℝ) (AD BC : ℝ) (r : ℝ)
  (h1 : AB = 27) 
  (h2 : CD = 45) 
  (h3 : AD = 13) 
  (h4 : BC = 15) 
  (h5 : r = 13.5) :
  r = 13.5 :=
by
  sorry  -- Detailed proof steps will go here

end semicircle_radius_in_trapezoid_l776_77619


namespace distance_travelled_l776_77691

def actual_speed : ℝ := 50
def additional_speed : ℝ := 25
def time_difference : ℝ := 0.5

theorem distance_travelled (D : ℝ) : 0.5 = (D / actual_speed) - (D / (actual_speed + additional_speed)) → D = 75 :=
by sorry

end distance_travelled_l776_77691


namespace fourth_student_in_sample_l776_77689

def sample_interval (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  total_students / sample_size

def in_sample (student_number : ℕ) (start : ℕ) (interval : ℕ) (n : ℕ) : Prop :=
  student_number = start + n * interval

theorem fourth_student_in_sample :
  ∀ (total_students sample_size : ℕ) (s1 s2 s3 : ℕ),
    total_students = 52 →
    sample_size = 4 →
    s1 = 7 →
    s2 = 33 →
    s3 = 46 →
    ∃ s4, in_sample s4 s1 (sample_interval total_students sample_size) 1 ∧
           in_sample s2 s1 (sample_interval total_students sample_size) 2 ∧
           in_sample s3 s1 (sample_interval total_students sample_size) 3 ∧
           s4 = 20 := 
by
  sorry

end fourth_student_in_sample_l776_77689


namespace perpendicular_line_passing_point_l776_77656

theorem perpendicular_line_passing_point (x y : ℝ) (hx : 4 * x - 3 * y + 2 = 0) : 
  ∃ l : ℝ → ℝ → Prop, (∀ x y, l x y ↔ (3 * x + 4 * y + 1 = 0) → l 1 2) :=
sorry

end perpendicular_line_passing_point_l776_77656


namespace teams_in_league_l776_77610

def number_of_teams (n : ℕ) := n * (n - 1) / 2

theorem teams_in_league : ∃ n : ℕ, number_of_teams n = 36 ∧ n = 9 := by
  sorry

end teams_in_league_l776_77610


namespace cosine_squared_is_half_l776_77612

def sides_of_triangle (p q r : ℝ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q > r ∧ q + r > p ∧ r + p > q

noncomputable def cosine_squared (p q r : ℝ) : ℝ :=
  ((p^2 + q^2 - r^2) / (2 * p * q))^2

theorem cosine_squared_is_half (p q r : ℝ) (h : sides_of_triangle p q r) 
  (h_eq : p^4 + q^4 + r^4 = 2 * r^2 * (p^2 + q^2)) : cosine_squared p q r = 1 / 2 :=
by
  sorry

end cosine_squared_is_half_l776_77612


namespace complex_eq_z100_zReciprocal_l776_77671

theorem complex_eq_z100_zReciprocal
  (z : ℂ)
  (h : z + z⁻¹ = 2 * Real.cos (5 * Real.pi / 180)) :
  z^100 + z⁻¹^100 = -2 * Real.cos (40 * Real.pi / 180) :=
by
  sorry

end complex_eq_z100_zReciprocal_l776_77671


namespace a_range_l776_77649

noncomputable def f (a x : ℝ) : ℝ := x^3 + a*x^2 - 2*x + 5

noncomputable def f' (a x : ℝ) : ℝ := 3*x^2 + 2*a*x - 2

theorem a_range (a : ℝ) :
  (∃ x y : ℝ, (1/3 < x ∧ x < 1/2) ∧ (1/3 < y ∧ y < 1/2) ∧ f' a x = 0 ∧ f' a y = 0) ↔
  a ∈ Set.Ioo (5/4) (5/2) :=
by
  sorry

end a_range_l776_77649


namespace which_is_system_lin_eq_l776_77611

def option_A : Prop := ∀ (x : ℝ), x - 1 = 2 * x
def option_B : Prop := ∀ (x y : ℝ), x - 1/y = 1
def option_C : Prop := ∀ (x z : ℝ), x + z = 3
def option_D : Prop := ∀ (x y z : ℝ), x - y + z = 1

theorem which_is_system_lin_eq (hA : option_A) (hB : option_B) (hC : option_C) (hD : option_D) :
    (∀ (x z : ℝ), x + z = 3) :=
by
  sorry

end which_is_system_lin_eq_l776_77611


namespace daily_production_l776_77678

-- Definitions based on conditions
def weekly_production : ℕ := 3400
def working_days_in_week : ℕ := 5

-- Statement to prove the number of toys produced each day
theorem daily_production : (weekly_production / working_days_in_week) = 680 :=
by
  sorry

end daily_production_l776_77678


namespace amount_c_l776_77653

theorem amount_c (a b c d : ℝ) :
  a + c = 350 →
  b + d = 450 →
  a + d = 400 →
  c + d = 500 →
  a + b + c + d = 750 →
  c = 225 :=
by 
  intros h1 h2 h3 h4 h5
  -- Proof omitted.
  sorry

end amount_c_l776_77653


namespace units_digit_of_modifiedLucas_L20_eq_d_l776_77647

def modifiedLucas : ℕ → ℕ
| 0 => 3
| 1 => 2
| n + 2 => 2 * modifiedLucas (n + 1) + modifiedLucas n

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_modifiedLucas_L20_eq_d :
  ∃ d, units_digit (modifiedLucas (modifiedLucas 20)) = d :=
by
  sorry

end units_digit_of_modifiedLucas_L20_eq_d_l776_77647


namespace discriminant_quadratic_eq_l776_77688

theorem discriminant_quadratic_eq : 
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  Δ = 33 :=
by
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  exact sorry

end discriminant_quadratic_eq_l776_77688


namespace loss_percent_l776_77639

theorem loss_percent (CP SP : ℝ) (h_CP : CP = 600) (h_SP : SP = 550) :
  ((CP - SP) / CP) * 100 = 8.33 := by
  sorry

end loss_percent_l776_77639


namespace sunny_lead_l776_77675

-- Define the given conditions as hypotheses
variables (h d : ℝ) (s w : ℝ)
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h)

-- State the theorem we want to prove
theorem sunny_lead (h d : ℝ) (s w : ℝ) 
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h) :
    ∃ distance_ahead_Sunny : ℝ, distance_ahead_Sunny = (2 * d^2) / h :=
sorry

end sunny_lead_l776_77675


namespace find_other_number_l776_77664

noncomputable def calculateB (lcm hcf a : ℕ) : ℕ :=
  (lcm * hcf) / a

theorem find_other_number :
  ∃ B : ℕ, (calculateB 76176 116 8128) = 1087 :=
by
  use 1087
  sorry

end find_other_number_l776_77664


namespace proof_m_range_l776_77628

variable {x m : ℝ}

def A (m : ℝ) : Set ℝ := {x | x^2 + x + m + 2 = 0}
def B : Set ℝ := {x | x > 0}

theorem proof_m_range (h : A m ∩ B = ∅) : m ≤ -2 := 
sorry

end proof_m_range_l776_77628


namespace Cid_charges_5_for_car_wash_l776_77635

theorem Cid_charges_5_for_car_wash (x : ℝ) :
  5 * 20 + 10 * 30 + 15 * x = 475 → x = 5 :=
by
  intro h
  sorry

end Cid_charges_5_for_car_wash_l776_77635


namespace fraction_zero_l776_77614

theorem fraction_zero (x : ℝ) (h : x ≠ 3) : (2 * x^2 - 6 * x) / (x - 3) = 0 ↔ x = 0 := 
by
  sorry

end fraction_zero_l776_77614


namespace hexagons_cover_65_percent_l776_77637

noncomputable def hexagon_percent_coverage
    (a : ℝ)
    (square_area : ℝ := a^2) 
    (hexagon_area : ℝ := (3 * Real.sqrt 3 / 8 * a^2))
    (tile_pattern : ℝ := 3): Prop :=
    hexagon_area / square_area * tile_pattern = (65 / 100)

theorem hexagons_cover_65_percent (a : ℝ) : hexagon_percent_coverage a :=
by
    sorry

end hexagons_cover_65_percent_l776_77637


namespace p_at_zero_l776_77643

-- Define the quartic monic polynomial
noncomputable def p (x : ℝ) : ℝ := sorry

-- Conditions
axiom p_monic : true -- p is a monic polynomial, we represent it by an axiom here for simplicity
axiom p_neg2 : p (-2) = -4
axiom p_1 : p (1) = -1
axiom p_3 : p (3) = -9
axiom p_5 : p (5) = -25

-- The theorem to be proven
theorem p_at_zero : p 0 = -30 := by
  sorry

end p_at_zero_l776_77643


namespace Eric_return_time_l776_77672

theorem Eric_return_time (t1 t2 t_return : ℕ) 
  (h1 : t1 = 20) 
  (h2 : t2 = 10) 
  (h3 : t_return = 3 * (t1 + t2)) : 
  t_return = 90 := 
by 
  sorry

end Eric_return_time_l776_77672


namespace sequence_a_2017_l776_77692

theorem sequence_a_2017 :
  (∃ (a : ℕ → ℚ), (a 1 = 1) ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2016 * a n / (2014 * a n + 2016)) → a 2017 = 1008 / (1007 * 2017 + 1)) :=
by
  sorry

end sequence_a_2017_l776_77692


namespace paula_karl_age_sum_l776_77697

theorem paula_karl_age_sum :
  ∃ (P K : ℕ), (P - 5 = 3 * (K - 5)) ∧ (P + 6 = 2 * (K + 6)) ∧ (P + K = 54) :=
by
  sorry

end paula_karl_age_sum_l776_77697


namespace quotient_of_integers_l776_77646

theorem quotient_of_integers
  (a b : ℤ)
  (h : 1996 * a + b / 96 = a + b) :
  b / a = 2016 ∨ a / b = 2016 := 
sorry

end quotient_of_integers_l776_77646


namespace insects_ratio_l776_77662

theorem insects_ratio (total_insects : ℕ) (geckos : ℕ) (gecko_insects : ℕ) (lizards : ℕ)
  (H1 : geckos * gecko_insects + lizards * ((total_insects - geckos * gecko_insects) / lizards) = total_insects)
  (H2 : total_insects = 66)
  (H3 : geckos = 5)
  (H4 : gecko_insects = 6)
  (H5 : lizards = 3) :
  (total_insects - geckos * gecko_insects) / lizards / gecko_insects = 2 :=
by
  sorry

end insects_ratio_l776_77662


namespace largest_consecutive_odd_nat_divisible_by_3_sum_72_l776_77668

theorem largest_consecutive_odd_nat_divisible_by_3_sum_72
  (a : ℕ)
  (h₁ : a % 3 = 0)
  (h₂ : (a + 6) % 3 = 0)
  (h₃ : (a + 12) % 3 = 0)
  (h₄ : a % 2 = 1)
  (h₅ : (a + 6) % 2 = 1)
  (h₆ : (a + 12) % 2 = 1)
  (h₇ : a + (a + 6) + (a + 12) = 72) :
  a + 12 = 30 :=
by
  sorry

end largest_consecutive_odd_nat_divisible_by_3_sum_72_l776_77668


namespace prob_exactly_one_hits_prob_at_least_one_hits_l776_77606

noncomputable def prob_A_hits : ℝ := 1 / 2
noncomputable def prob_B_hits : ℝ := 1 / 3
noncomputable def prob_A_misses : ℝ := 1 - prob_A_hits
noncomputable def prob_B_misses : ℝ := 1 - prob_B_hits

theorem prob_exactly_one_hits :
  (prob_A_hits * prob_B_misses) + (prob_A_misses * prob_B_hits) = 1 / 2 :=
by sorry

theorem prob_at_least_one_hits :
  1 - (prob_A_misses * prob_B_misses) = 2 / 3 :=
by sorry

end prob_exactly_one_hits_prob_at_least_one_hits_l776_77606


namespace solve_for_x_l776_77655

theorem solve_for_x {x : ℤ} (h : 3 * x + 7 = -2) : x = -3 := 
by
  sorry

end solve_for_x_l776_77655


namespace descent_time_l776_77631

-- Definitions based on conditions
def time_to_top : ℝ := 4
def avg_speed_up : ℝ := 2.625
def avg_speed_total : ℝ := 3.5
def distance_to_top : ℝ := avg_speed_up * time_to_top -- 10.5 km
def total_distance : ℝ := 2 * distance_to_top       -- 21 km

-- Theorem statement: the time to descend (t_down) should be 2 hours
theorem descent_time (t_down : ℝ) : 
  avg_speed_total * (time_to_top + t_down) = total_distance →
  t_down = 2 := 
by 
  -- skip the proof
  sorry

end descent_time_l776_77631


namespace students_without_A_l776_77600

theorem students_without_A (total_students : ℕ) (students_english : ℕ) 
  (students_math : ℕ) (students_both : ℕ) (students_only_math : ℕ) :
  total_students = 30 → students_english = 6 → students_math = 15 → 
  students_both = 3 → students_only_math = 1 →
  (total_students - (students_math - students_only_math + 
                     students_english - students_both + 
                     students_both) = 12) :=
by sorry

end students_without_A_l776_77600


namespace atm_withdrawal_cost_l776_77695

theorem atm_withdrawal_cost (x y : ℝ)
  (h1 : 221 = x + 40000 * y)
  (h2 : 485 = x + 100000 * y) :
  (x + 85000 * y) = 419 := by
  sorry

end atm_withdrawal_cost_l776_77695


namespace correct_answer_l776_77694

def sum_squares_of_three_consecutive_even_integers (n : ℤ) : ℤ :=
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  a * a + b * b + c * c

def T : Set ℤ :=
  {t | ∃ n : ℤ, t = sum_squares_of_three_consecutive_even_integers n}

theorem correct_answer : (∀ t ∈ T, t % 4 = 0) ∧ (∀ t ∈ T, t % 7 ≠ 0) :=
sorry

end correct_answer_l776_77694


namespace compute_alpha_powers_l776_77677

variable (α1 α2 α3 : ℂ)

open Complex

-- Given conditions
def condition1 : Prop := α1 + α2 + α3 = 2
def condition2 : Prop := α1^2 + α2^2 + α3^2 = 6
def condition3 : Prop := α1^3 + α2^3 + α3^3 = 14

-- The required proof statement
theorem compute_alpha_powers (h1 : condition1 α1 α2 α3) (h2 : condition2 α1 α2 α3) (h3 : condition3 α1 α2 α3) :
  α1^7 + α2^7 + α3^7 = 46 := by
  sorry

end compute_alpha_powers_l776_77677


namespace roots_reciprocal_l776_77644

theorem roots_reciprocal (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 - 1 = 0) (h2 : x2^2 - 3 * x2 - 1 = 0) 
                         (h_sum : x1 + x2 = 3) (h_prod : x1 * x2 = -1) :
  (1 / x1) + (1 / x2) = -3 :=
by
  sorry

end roots_reciprocal_l776_77644


namespace math_problem_l776_77605

theorem math_problem (x y : ℝ) (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 :=
sorry

end math_problem_l776_77605


namespace length_of_BC_l776_77699

theorem length_of_BC (AB AC AM : ℝ) (hAB : AB = 5) (hAC : AC = 8) (hAM : AM = 4.5) : 
  ∃ BC, BC = Real.sqrt 97 :=
by
  sorry

end length_of_BC_l776_77699


namespace sequence_term_101_l776_77621

theorem sequence_term_101 :
  ∃ a : ℕ → ℚ, a 1 = 2 ∧ (∀ n : ℕ, 2 * a (n+1) - 2 * a n = 1) ∧ a 101 = 52 :=
by
  sorry

end sequence_term_101_l776_77621


namespace count_total_coins_l776_77607

theorem count_total_coins (quarters nickels : Nat) (h₁ : quarters = 4) (h₂ : nickels = 8) : quarters + nickels = 12 :=
by sorry

end count_total_coins_l776_77607


namespace simplification_correct_l776_77633

noncomputable def given_equation (x : ℚ) : Prop := 
  x / (2 * x - 1) - 3 = 2 / (1 - 2 * x)

theorem simplification_correct (x : ℚ) (h : given_equation x) : 
  x - 3 * (2 * x - 1) = -2 :=
sorry

end simplification_correct_l776_77633


namespace arccos_sin_1_5_eq_pi_over_2_minus_1_5_l776_77683

-- Define the problem statement in Lean 4.
theorem arccos_sin_1_5_eq_pi_over_2_minus_1_5 : 
  Real.arccos (Real.sin 1.5) = (Real.pi / 2) - 1.5 :=
by
  sorry

end arccos_sin_1_5_eq_pi_over_2_minus_1_5_l776_77683


namespace trapezoid_area_l776_77642

theorem trapezoid_area 
  (diagonals_perpendicular : ∀ A B C D : ℝ, (A ≠ B → C ≠ D → A * C + B * D = 0)) 
  (diagonal_length : ∀ B D : ℝ, B ≠ D → (B - D) = 17) 
  (height_of_trapezoid : ∀ (height : ℝ), height = 15) : 
  ∃ (area : ℝ), area = 4335 / 16 := 
sorry

end trapezoid_area_l776_77642


namespace green_or_blue_marble_probability_l776_77673

theorem green_or_blue_marble_probability :
  (4 + 3 : ℝ) / (4 + 3 + 8) = 0.4667 := by
  sorry

end green_or_blue_marble_probability_l776_77673


namespace cost_of_largest_pot_is_2_52_l776_77652

/-
Mark bought a set of 6 flower pots of different sizes at a total pre-tax cost.
Each pot cost 0.4 more than the next one below it in size.
The total cost, including a sales tax of 7.5%, was $9.80.
Prove that the cost of the largest pot before sales tax was $2.52.
-/

def cost_smallest_pot (x : ℝ) : Prop :=
  let total_cost := x + (x + 0.4) + (x + 0.8) + (x + 1.2) + (x + 1.6) + (x + 2.0)
  let pre_tax_cost := total_cost / 1.075
  let pre_tax_total_cost := (9.80 / 1.075)
  (total_cost = 6 * x + 6 ∧ total_cost = pre_tax_total_cost) →
  (x + 2.0 = 2.52)

theorem cost_of_largest_pot_is_2_52 :
  ∃ x : ℝ, cost_smallest_pot x :=
sorry

end cost_of_largest_pot_is_2_52_l776_77652


namespace product_is_48_l776_77665

-- Define the conditions and the target product
def problem (x y : ℝ) := 
  x ≠ y ∧ (x + y) / (x - y) = 7 ∧ (x * y) / (x - y) = 24

-- Prove that the product is 48 given the conditions
theorem product_is_48 (x y : ℝ) (h : problem x y) : x * y = 48 :=
sorry

end product_is_48_l776_77665


namespace total_votes_cast_l776_77679

theorem total_votes_cast (V : ℕ) (h1 : V > 0) (h2 : ∃ c r : ℕ, c = 40 * V / 100 ∧ r = 40 * V / 100 + 5000 ∧ c + r = V):
  V = 25000 :=
by
  sorry

end total_votes_cast_l776_77679


namespace carpenter_additional_logs_needed_l776_77632

theorem carpenter_additional_logs_needed 
  (total_woodblocks_needed : ℕ) 
  (logs_available : ℕ) 
  (woodblocks_per_log : ℕ) 
  (additional_logs_needed : ℕ)
  (h1 : total_woodblocks_needed = 80)
  (h2 : logs_available = 8)
  (h3 : woodblocks_per_log = 5)
  (h4 : additional_logs_needed = 8) : 
  (total_woodblocks_needed - (logs_available * woodblocks_per_log)) / woodblocks_per_log = additional_logs_needed :=
by
  sorry

end carpenter_additional_logs_needed_l776_77632


namespace expression_value_l776_77690

theorem expression_value : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by {
  sorry
}

end expression_value_l776_77690


namespace marcel_potatoes_eq_l776_77620

-- Define the given conditions
def marcel_corn := 10
def dale_corn := marcel_corn / 2
def dale_potatoes := 8
def total_vegetables := 27

-- Define the fact that they bought 27 vegetables in total
def total_corn := marcel_corn + dale_corn
def total_potatoes := total_vegetables - total_corn

-- State the theorem
theorem marcel_potatoes_eq :
  (total_potatoes - dale_potatoes) = 4 :=
by
  -- Lean proof would go here
  sorry

end marcel_potatoes_eq_l776_77620


namespace cost_of_each_scoop_l776_77634

theorem cost_of_each_scoop (x : ℝ) 
  (pierre_scoops : ℝ := 3)
  (mom_scoops : ℝ := 4)
  (total_bill : ℝ := 14) 
  (h : 7 * x = total_bill) :
  x = 2 :=
by 
  sorry

end cost_of_each_scoop_l776_77634


namespace find_difference_of_a_and_b_l776_77682

-- Define the conditions
variables (a b : ℝ)
axiom cond1 : 4 * a + 3 * b = 8
axiom cond2 : 3 * a + 4 * b = 6

-- Statement for the proof
theorem find_difference_of_a_and_b : a - b = 2 :=
by
  sorry

end find_difference_of_a_and_b_l776_77682


namespace chocolate_cost_is_75_l776_77629

def candy_bar_cost : ℕ := 25
def juice_pack_cost : ℕ := 50
def num_quarters : ℕ := 11
def total_cost_in_cents : ℕ := num_quarters * candy_bar_cost
def num_candy_bars : ℕ := 3
def num_pieces_of_chocolate : ℕ := 2

def chocolate_cost_in_cents (x : ℕ) : Prop :=
  (num_candy_bars * candy_bar_cost) + (num_pieces_of_chocolate * x) + juice_pack_cost = total_cost_in_cents

theorem chocolate_cost_is_75 : chocolate_cost_in_cents 75 :=
  sorry

end chocolate_cost_is_75_l776_77629


namespace Mason_fathers_age_indeterminate_l776_77640

theorem Mason_fathers_age_indeterminate
  (Mason_age : ℕ) (Sydney_age Mason_father_age D : ℕ)
  (hM : Mason_age = 20)
  (hS_M : Mason_age = Sydney_age / 3)
  (hS_F : Mason_father_age - D = Sydney_age) :
  ¬ ∃ F, Mason_father_age = F :=
by {
  sorry
}

end Mason_fathers_age_indeterminate_l776_77640


namespace point_on_graph_l776_77608

noncomputable def f (x : ℝ) : ℝ := abs (x^3 + 1) + abs (x^3 - 1)

theorem point_on_graph (a : ℝ) : ∃ (x y : ℝ), (x = a) ∧ (y = f (-a)) ∧ (y = f x) :=
by 
  sorry

end point_on_graph_l776_77608


namespace lcm_ac_is_420_l776_77681

theorem lcm_ac_is_420 (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 21) :
    Nat.lcm a c = 420 :=
sorry

end lcm_ac_is_420_l776_77681


namespace minimum_value_of_expression_l776_77618

theorem minimum_value_of_expression (p q r s t u : ℝ) 
  (hpqrsu_pos : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ 0 < t ∧ 0 < u) 
  (sum_eq : p + q + r + s + t + u = 8) : 
  98 ≤ (2 / p + 4 / q + 9 / r + 16 / s + 25 / t + 36 / u) :=
sorry

end minimum_value_of_expression_l776_77618


namespace pqrs_inequality_l776_77674

theorem pqrs_inequality (p q r : ℝ) (h_condition : ∀ x : ℝ, (x < -6 ∨ |x - 30| ≤ 2) ↔ ((x - p) * (x - q)) / (x - r) ≥ 0)
  (h_pq : p < q) : p = 28 ∧ q = 32 ∧ r = -6 ∧ p + 2 * q + 3 * r = 78 :=
by
  sorry

end pqrs_inequality_l776_77674


namespace selling_price_of_mixture_l776_77657

noncomputable def selling_price_per_pound (weight1 weight2 price1 price2 total_weight : ℝ) : ℝ :=
  (weight1 * price1 + weight2 * price2) / total_weight

theorem selling_price_of_mixture :
  selling_price_per_pound 20 10 2.95 3.10 30 = 3.00 :=
by
  -- Skipping the proof part
  sorry

end selling_price_of_mixture_l776_77657


namespace find_a5_l776_77624

-- Define the problem conditions within Lean
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions of the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, 0 < a n
def condition1 (a : ℕ → ℝ) := a 1 * a 3 = 4
def condition2 (a : ℕ → ℝ) := a 7 * a 9 = 25

-- Proposition to prove
theorem find_a5 :
  geometric_sequence a q →
  positive_terms a →
  condition1 a →
  condition2 a →
  a 5 = Real.sqrt 10 :=
by
  sorry

end find_a5_l776_77624


namespace smallest_base_b_l776_77601

theorem smallest_base_b (b : ℕ) (n : ℕ) : b > 3 ∧ 3 * b + 4 = n ^ 2 → b = 4 := 
by
  sorry

end smallest_base_b_l776_77601


namespace prime_power_divides_binomial_l776_77638

theorem prime_power_divides_binomial {p n k α : ℕ} (hp : Nat.Prime p) 
  (h : p^α ∣ Nat.choose n k) : p^α ≤ n := 
sorry

end prime_power_divides_binomial_l776_77638


namespace a_1000_value_l776_77669

open Nat

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), (a 1 = 1010) ∧ (a 2 = 1011) ∧ 
  (∀ n ≥ 1, a n + a (n+1) + a (n+2) = 2 * n) ∧ 
  (a 1000 = 1676) :=
sorry

end a_1000_value_l776_77669


namespace find_angle_D_l776_77658

noncomputable def measure.angle_A := 80
noncomputable def measure.angle_B := 30
noncomputable def measure.angle_C := 20

def sum_angles_pentagon (A B C : ℕ) := 540 - (A + B + C)

theorem find_angle_D
  (A B C E F : ℕ)
  (hA : A = measure.angle_A)
  (hB : B = measure.angle_B)
  (hC : C = measure.angle_C)
  (h_sum_pentagon : A + B + C + D + E + F = 540)
  (h_triangle : D + E + F = 180) :
  D = 130 :=
by
  sorry

end find_angle_D_l776_77658


namespace measure_angle_A_l776_77666

-- Angles A and B are supplementary
def supplementary (A B : ℝ) : Prop :=
  A + B = 180

-- Definition of the problem conditions
def problem_conditions (A B : ℝ) : Prop :=
  supplementary A B ∧ A = 4 * B

-- The measure of angle A
def measure_of_A := 144

-- The statement to prove
theorem measure_angle_A (A B : ℝ) :
  problem_conditions A B → A = measure_of_A := 
by
  sorry

end measure_angle_A_l776_77666


namespace inequality_solution_l776_77630

theorem inequality_solution (x : ℝ) : 
  (x + 1) * (2 - x) < 0 ↔ x < -1 ∨ x > 2 := 
sorry

end inequality_solution_l776_77630


namespace articleWords_l776_77609

-- Define the number of words per page for larger and smaller types
def wordsLargerType : Nat := 1800
def wordsSmallerType : Nat := 2400

-- Define the total number of pages and the number of pages in smaller type
def totalPages : Nat := 21
def smallerTypePages : Nat := 17

-- The number of pages in larger type
def largerTypePages : Nat := totalPages - smallerTypePages

-- Calculate the total number of words in the article
def totalWords : Nat := (largerTypePages * wordsLargerType) + (smallerTypePages * wordsSmallerType)

-- Prove that the total number of words in the article is 48,000
theorem articleWords : totalWords = 48000 := 
by
  sorry

end articleWords_l776_77609


namespace polynomial_remainder_theorem_l776_77698

open Polynomial

theorem polynomial_remainder_theorem (Q : Polynomial ℝ)
  (h1 : Q.eval 20 = 120)
  (h2 : Q.eval 100 = 40) :
  ∃ R : Polynomial ℝ, R.degree < 2 ∧ Q = (X - 20) * (X - 100) * R + (-X + 140) :=
by
  sorry

end polynomial_remainder_theorem_l776_77698


namespace Alan_shells_l776_77648

theorem Alan_shells (l b a : ℕ) (h1 : l = 36) (h2 : b = l / 3) (h3 : a = 4 * b) : a = 48 :=
by
sorry

end Alan_shells_l776_77648


namespace d_is_distance_function_l776_77636

noncomputable def d (x y : ℝ) : ℝ := |x - y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem d_is_distance_function : 
  (∀ x, d x x = 0) ∧ 
  (∀ x y, d x y = d y x) ∧ 
  (∀ x y z, d x y + d y z ≥ d x z) :=
by
  sorry

end d_is_distance_function_l776_77636


namespace value_of_p_l776_77676

theorem value_of_p (p q : ℝ) (h1 : q = (2 / 5) * p) (h2 : p * q = 90) : p = 15 :=
by
  sorry

end value_of_p_l776_77676


namespace union_of_A_B_l776_77616

def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }

def B : Set ℝ := { x | 1 < x ∧ x ≤ 3 }

theorem union_of_A_B : A ∪ B = { x | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end union_of_A_B_l776_77616


namespace red_light_cherries_cost_price_min_value_m_profit_l776_77625

-- Define the constants and cost conditions
def cost_price_red_light_cherries (x : ℝ) (y : ℝ) : Prop :=
  (6000 / (2 * x) - 100 = 1000 / x)

-- Define sales conditions and profit requirement
def min_value_m (m : ℝ) (profit : ℝ) : Prop :=
  (20 * 3 * m + 20 * (20 - 0.5 * m) + (28 - 20) * (50 - 3 * m - 20) >= profit)

-- Define the main proof goal statements
theorem red_light_cherries_cost_price :
  ∃ x, cost_price_red_light_cherries x 6000 ∧ 20 = x :=
sorry

theorem min_value_m_profit :
  ∃ m, min_value_m m 770 ∧ m >= 5 :=
sorry

end red_light_cherries_cost_price_min_value_m_profit_l776_77625


namespace shells_not_red_or_green_l776_77685

theorem shells_not_red_or_green (total_shells : ℕ) (red_shells : ℕ) (green_shells : ℕ) 
  (h_total : total_shells = 291) (h_red : red_shells = 76) (h_green : green_shells = 49) :
  total_shells - (red_shells + green_shells) = 166 :=
by
  sorry

end shells_not_red_or_green_l776_77685


namespace painted_faces_of_large_cube_l776_77680

theorem painted_faces_of_large_cube (n : ℕ) (unpainted_cubes : ℕ) :
  n = 9 ∧ unpainted_cubes = 343 → (painted_faces : ℕ) = 3 :=
by
  intros h
  let ⟨h_n, h_unpainted⟩ := h
  sorry

end painted_faces_of_large_cube_l776_77680


namespace contrapositive_quadratic_roots_l776_77670

theorem contrapositive_quadratic_roots (m : ℝ) (h_discriminant : 1 + 4 * m < 0) : m ≤ 0 :=
sorry

end contrapositive_quadratic_roots_l776_77670


namespace race_speed_ratio_l776_77659

theorem race_speed_ratio (L v_a v_b : ℝ) (h1 : v_a = v_b / 0.84375) :
  v_a / v_b = 32 / 27 :=
by sorry

end race_speed_ratio_l776_77659


namespace solve_equation_l776_77603

theorem solve_equation (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2/3) :=
by
  sorry

end solve_equation_l776_77603


namespace power_multiplication_l776_77686

theorem power_multiplication (a : ℝ) (b : ℝ) (m : ℕ) (n : ℕ) (h1 : a = 0.25) (h2 : b = 4) (h3 : m = 2023) (h4 : n = 2024) : 
  a^m * b^n = 4 := 
by 
  sorry

end power_multiplication_l776_77686


namespace twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l776_77661

theorem twelve_plus_four_times_five_minus_five_cubed_equals_twelve :
  12 + 4 * (5 - 10 / 2) ^ 3 = 12 := by
  sorry

end twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l776_77661


namespace negation_of_at_most_one_odd_l776_77623

variable (a b c : ℕ)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def at_most_one_odd (a b c : ℕ) : Prop :=
  (is_odd a ∧ ¬is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ ¬is_odd c)

theorem negation_of_at_most_one_odd :
  ¬ at_most_one_odd a b c ↔
  ∃ x y, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧ is_odd x ∧ is_odd y :=
sorry

end negation_of_at_most_one_odd_l776_77623


namespace greg_spent_on_shirt_l776_77615

-- Define the conditions in Lean
variables (S H : ℤ)
axiom condition1 : H = 2 * S + 9
axiom condition2 : S + H = 300

-- State the theorem to prove
theorem greg_spent_on_shirt : S = 97 :=
by
  sorry

end greg_spent_on_shirt_l776_77615


namespace max_apartment_size_l776_77696

/-- Define the rental rate and the maximum rent Michael can afford. -/
def rental_rate : ℝ := 1.20
def max_rent : ℝ := 720

/-- State the problem in Lean: Prove that the maximum apartment size Michael should consider is 600 square feet. -/
theorem max_apartment_size :
  ∃ s : ℝ, rental_rate * s = max_rent ∧ s = 600 := by
  sorry

end max_apartment_size_l776_77696


namespace arithmetic_result_l776_77613

theorem arithmetic_result :
  1325 + (572 / 52) - 225 + (2^3) = 1119 :=
by
  sorry

end arithmetic_result_l776_77613


namespace union_of_sets_l776_77654

def A : Set Int := {-1, 2, 3, 5}
def B : Set Int := {2, 4, 5}

theorem union_of_sets :
  A ∪ B = {-1, 2, 3, 4, 5} := by
  sorry

end union_of_sets_l776_77654


namespace cost_price_per_meter_l776_77622

def selling_price_for_85_meters : ℝ := 8925
def profit_per_meter : ℝ := 25
def number_of_meters : ℝ := 85

theorem cost_price_per_meter : (selling_price_for_85_meters - profit_per_meter * number_of_meters) / number_of_meters = 80 := by
  sorry

end cost_price_per_meter_l776_77622


namespace quadratic_equation_must_be_minus_2_l776_77617

-- Define the main problem statement
theorem quadratic_equation_must_be_minus_2 (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x ^ |m| - 3 * x - 7 = 0) →
  (∀ (h : |m| = 2), m - 2 ≠ 0) →
  m = -2 :=
sorry

end quadratic_equation_must_be_minus_2_l776_77617


namespace transformed_inequality_solution_l776_77651

variable {a b c d : ℝ}

theorem transformed_inequality_solution (H : ∀ x : ℝ, ((-1 < x ∧ x < -1/3) ∨ (1/2 < x ∧ x < 1)) → 
  (b / (x + a) + (x + d) / (x + c) < 0)) :
  ∀ x : ℝ, ((1 < x ∧ x < 3) ∨ (-2 < x ∧ x < -1)) ↔ (bx / (ax - 1) + (dx - 1) / (cx - 1) < 0) :=
sorry

end transformed_inequality_solution_l776_77651


namespace solve_inequality_l776_77660

theorem solve_inequality (x : ℝ) : 
  3*x^2 + 2*x - 3 > 10 - 2*x ↔ x < ( -2 - Real.sqrt 43 ) / 3 ∨ x > ( -2 + Real.sqrt 43 ) / 3 := 
by
  sorry

end solve_inequality_l776_77660
