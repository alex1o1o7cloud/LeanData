import Mathlib

namespace geometric_sequence_sum_ratio_l1026_102639

noncomputable def a (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * q^n

-- Sum of the first 'n' terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a1 q : ℝ) 
  (h : 8 * (a 11 a1 q) = (a 14 a1 q)) :
  (S 4 a1 q) / (S 2 a1 q) = 5 :=
by
  sorry

end geometric_sequence_sum_ratio_l1026_102639


namespace Ariella_total_amount_l1026_102624

-- We define the conditions
def Daniella_initial (daniella_amount : ℝ) := daniella_amount = 400
def Ariella_initial (daniella_amount : ℝ) (ariella_amount : ℝ) := ariella_amount = daniella_amount + 200
def simple_interest_rate : ℝ := 0.10
def investment_period : ℕ := 2

-- We state the goal to prove
theorem Ariella_total_amount (daniella_amount ariella_amount : ℝ) :
  Daniella_initial daniella_amount →
  Ariella_initial daniella_amount ariella_amount →
  ariella_amount + ariella_amount * simple_interest_rate * (investment_period : ℝ) = 720 :=
by
  sorry

end Ariella_total_amount_l1026_102624


namespace hyperbola_asymptote_l1026_102654

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 ∧ ∀ (x y : ℝ), (y = 3/5 * x ↔ y = 3 / 5 * x)) → a = 5 :=
by
  sorry

end hyperbola_asymptote_l1026_102654


namespace katie_total_earnings_l1026_102696

-- Define the conditions
def bead_necklaces := 4
def gem_necklaces := 3
def price_per_necklace := 3

-- The total money earned
def total_money_earned := bead_necklaces + gem_necklaces * price_per_necklace = 21

-- The statement to prove
theorem katie_total_earnings : total_money_earned :=
by
  sorry

end katie_total_earnings_l1026_102696


namespace find_value_l1026_102658

noncomputable def f : ℝ → ℝ := sorry

def tangent_line (x y : ℝ) : Prop := x + 2 * y + 1 = 0

def has_tangent_at (f : ℝ → ℝ) (x0 : ℝ) (L : ℝ → ℝ → Prop) : Prop :=
  L x0 (f x0)

theorem find_value (h : has_tangent_at f 2 tangent_line) :
  f 2 - 2 * (deriv f 2) = -1/2 :=
sorry

end find_value_l1026_102658


namespace volume_inhaled_per_breath_is_correct_l1026_102610

def breaths_per_minute : ℤ := 17
def volume_inhaled_24_hours : ℤ := 13600
def minutes_per_hour : ℤ := 60
def hours_per_day : ℤ := 24

def total_minutes_24_hours : ℤ := hours_per_day * minutes_per_hour
def total_breaths_24_hours : ℤ := total_minutes_24_hours * breaths_per_minute
def volume_per_breath := (volume_inhaled_24_hours : ℚ) / (total_breaths_24_hours : ℚ)

theorem volume_inhaled_per_breath_is_correct :
  volume_per_breath = 0.5556 := by
  sorry

end volume_inhaled_per_breath_is_correct_l1026_102610


namespace range_of_m_l1026_102633

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4 * m - 5) * x^2 - 4 * (m - 1) * x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by
  sorry

end range_of_m_l1026_102633


namespace product_of_fractions_l1026_102687

-- Define the fractions
def one_fourth : ℚ := 1 / 4
def one_half : ℚ := 1 / 2
def one_eighth : ℚ := 1 / 8

-- State the theorem we are proving
theorem product_of_fractions :
  one_fourth * one_half = one_eighth :=
by
  sorry

end product_of_fractions_l1026_102687


namespace junior_high_ten_total_games_l1026_102698

theorem junior_high_ten_total_games :
  let teams := 10
  let conference_games_per_team := 3
  let non_conference_games_per_team := 5
  let pairs_of_teams := Nat.choose teams 2
  let total_conference_games := pairs_of_teams * conference_games_per_team
  let total_non_conference_games := teams * non_conference_games_per_team
  let total_games := total_conference_games + total_non_conference_games
  total_games = 185 :=
by
  sorry

end junior_high_ten_total_games_l1026_102698


namespace min_sum_distinct_positive_integers_l1026_102647

theorem min_sum_distinct_positive_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (1 / a + 1 / b = k1 * (1 / c)) ∧ (1 / a + 1 / c = k2 * (1 / b)) ∧ (1 / b + 1 / c = k3 * (1 / a))) :
  a + b + c ≥ 11 :=
sorry

end min_sum_distinct_positive_integers_l1026_102647


namespace largest_cuts_9x9_l1026_102641

theorem largest_cuts_9x9 (k : ℕ) (V E F : ℕ) (hV : V = 81) (hE : E = 4 * k) (hF : F = 1 + 2 * k)
  (hEuler : V - E + F ≥ 2) : k ≤ 21 :=
by
  sorry

end largest_cuts_9x9_l1026_102641


namespace range_of_m_l1026_102685

theorem range_of_m (m : ℝ) : ((m + 3 > 0) ∧ (m - 1 < 0)) ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end range_of_m_l1026_102685


namespace math_problem_l1026_102690

theorem math_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + 2) * (b + 2) = 18) :
  (∀ x, (x = 3 / (a + 2) + 3 / (b + 2)) → x ≥ Real.sqrt 2) ∧
  ¬(∃ y, (y = a * b) ∧ y ≤ 11 - 6 * Real.sqrt 2) ∧
  (∀ z, (z = 2 * a + b) → z ≥ 6) ∧
  (∀ w, (w = (a + 1) * b) → w ≤ 8) :=
sorry

end math_problem_l1026_102690


namespace tan_add_pi_over_3_l1026_102608

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end tan_add_pi_over_3_l1026_102608


namespace abs_expression_equals_one_l1026_102631

theorem abs_expression_equals_one : 
  abs (abs (-(abs (2 - 3)) + 2) - 2) = 1 := 
  sorry

end abs_expression_equals_one_l1026_102631


namespace children_absent_l1026_102630

theorem children_absent (A : ℕ) (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ) :
  total_children = 660 →
  bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children * bananas_per_child) = 1320 →
  ((total_children - A) * (bananas_per_child + extra_bananas_per_child)) = 1320 →
  A = 330 :=
by
  intros
  sorry

end children_absent_l1026_102630


namespace angle_in_second_quadrant_l1026_102659

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : (2 * Real.tan (α / 2)) / (1 - (Real.tan (α / 2))^2) < 0) : 
  ∃ q, q = 2 ∧ α ∈ {α | 0 < α ∧ α < π} :=
by
  sorry

end angle_in_second_quadrant_l1026_102659


namespace compare_y1_y2_l1026_102686

def parabola (x : ℝ) (c : ℝ) : ℝ := -x^2 + 4 * x + c

theorem compare_y1_y2 (c y1 y2 : ℝ) :
  parabola (-1) c = y1 →
  parabola 1 c = y2 →
  y1 < y2 :=
by
  intro h1 h2
  sorry

end compare_y1_y2_l1026_102686


namespace sum_squares_inequality_l1026_102681

theorem sum_squares_inequality {a b c : ℝ} 
  (h1 : a > 0)
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
sorry

end sum_squares_inequality_l1026_102681


namespace sales_proof_valid_l1026_102634

variables (T: ℝ) (Teq: T = 30)
noncomputable def check_sales_proof : Prop :=
  (6.4 * T + 228 = 420)

theorem sales_proof_valid (T : ℝ) (Teq: T = 30) : check_sales_proof T :=
  by
    rw [Teq]
    norm_num
    sorry

end sales_proof_valid_l1026_102634


namespace total_wheels_correct_l1026_102667

def total_wheels (bicycles cars motorcycles tricycles quads : ℕ) 
(missing_bicycle_wheels broken_car_wheels missing_motorcycle_wheels : ℕ) : ℕ :=
  let bicycles_wheels := (bicycles - missing_bicycle_wheels) * 2 + missing_bicycle_wheels
  let cars_wheels := (cars - broken_car_wheels) * 4 + broken_car_wheels * 3
  let motorcycles_wheels := (motorcycles - missing_motorcycle_wheels) * 2
  let tricycles_wheels := tricycles * 3
  let quads_wheels := quads * 4
  bicycles_wheels + cars_wheels + motorcycles_wheels + tricycles_wheels + quads_wheels

theorem total_wheels_correct : total_wheels 25 15 8 3 2 5 2 1 = 134 := 
  by sorry

end total_wheels_correct_l1026_102667


namespace cube_root_inequality_l1026_102611

theorem cube_root_inequality {a b : ℝ} (h : a > b) : (a^(1/3)) > (b^(1/3)) :=
sorry

end cube_root_inequality_l1026_102611


namespace partnership_total_annual_gain_l1026_102605

theorem partnership_total_annual_gain 
  (x : ℝ) 
  (G : ℝ)
  (hA_investment : x * 12 = A_investment)
  (hB_investment : 2 * x * 6 = B_investment)
  (hC_investment : 3 * x * 4 = C_investment)
  (A_share : (A_investment / (A_investment + B_investment + C_investment)) * G = 6000) :
  G = 18000 := 
sorry

end partnership_total_annual_gain_l1026_102605


namespace clock_equiv_l1026_102626

theorem clock_equiv (h : ℕ) (h_gt_6 : h > 6) : h ≡ h^2 [MOD 12] ∧ h ≡ h^3 [MOD 12] → h = 9 :=
by
  sorry

end clock_equiv_l1026_102626


namespace ratio_of_segments_l1026_102653

theorem ratio_of_segments (a b x : ℝ) (h₁ : a = 9 * x) (h₂ : b = 99 * x) : b / a = 11 := by
  sorry

end ratio_of_segments_l1026_102653


namespace candy_remaining_l1026_102691

theorem candy_remaining
  (initial_candies : ℕ)
  (talitha_took : ℕ)
  (solomon_took : ℕ)
  (h_initial : initial_candies = 349)
  (h_talitha : talitha_took = 108)
  (h_solomon : solomon_took = 153) :
  initial_candies - (talitha_took + solomon_took) = 88 :=
by
  sorry

end candy_remaining_l1026_102691


namespace question_inequality_l1026_102674

theorem question_inequality (x y z : ℝ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x ≥ max (3/4 * (x - y)^2) (max (3/4 * (y - z)^2) (3/4 * (z - x)^2)) := 
sorry

end question_inequality_l1026_102674


namespace olivers_friend_gave_l1026_102618

variable (initial_amount saved_amount spent_frisbee spent_puzzle final_amount : ℕ) 

theorem olivers_friend_gave (h1 : initial_amount = 9) 
                           (h2 : saved_amount = 5) 
                           (h3 : spent_frisbee = 4) 
                           (h4 : spent_puzzle = 3) 
                           (h5 : final_amount = 15) : 
                           final_amount - (initial_amount + saved_amount - (spent_frisbee + spent_puzzle)) = 8 := 
by 
  sorry

end olivers_friend_gave_l1026_102618


namespace problem_l1026_102695

variable (a b : ℝ)

theorem problem (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end problem_l1026_102695


namespace max_min_diff_w_l1026_102670

theorem max_min_diff_w (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 4) :
  let w := a^2 + a*b + b^2
  let w1 := max (0^2 + 0*b + b^2) (4^2 + 4*b + b^2)
  let w2 := (2-2)^2 + 12
  w1 - w2 = 4 :=
by
  -- skip the proof
  sorry

end max_min_diff_w_l1026_102670


namespace quadratic_inequality_no_real_roots_l1026_102672

theorem quadratic_inequality_no_real_roots (a b c : ℝ) (h : a ≠ 0) (h_Δ : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) :=
sorry

end quadratic_inequality_no_real_roots_l1026_102672


namespace map_float_time_l1026_102649

theorem map_float_time
  (t₀ t₁ : Nat) -- times representing 12:00 PM and 12:21 PM in minutes since midnight
  (v_w v_b : ℝ) -- constant speed of water current and boat in still water
  (h₀ : t₀ = 12 * 60) -- t₀ is 12:00 PM
  (h₁ : t₁ = 12 * 60 + 21) -- t₁ is 12:21 PM
  : t₁ - t₀ = 21 := 
  sorry

end map_float_time_l1026_102649


namespace cos_sum_identity_l1026_102692

theorem cos_sum_identity :
  (Real.cos (75 * Real.pi / 180)) ^ 2 + (Real.cos (15 * Real.pi / 180)) ^ 2 + 
  (Real.cos (75 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 5 / 4 := 
by
  sorry

end cos_sum_identity_l1026_102692


namespace amanda_quizzes_l1026_102613

theorem amanda_quizzes (n : ℕ) (h1 : n > 0) (h2 : 92 * n + 97 = 93 * 5) : n = 4 :=
by
  sorry

end amanda_quizzes_l1026_102613


namespace sum_of_powers_modulo_l1026_102666

theorem sum_of_powers_modulo (R : Finset ℕ) (S : ℕ) :
  (∀ n < 100, ∃ r, r ∈ R ∧ r = 3^n % 500) →
  S = R.sum id →
  (S % 500) = 0 :=
by {
  -- Proof would go here
  sorry
}

end sum_of_powers_modulo_l1026_102666


namespace marble_158th_is_gray_l1026_102628

def marble_color (n : ℕ) : String :=
  if (n % 12 < 5) then "gray"
  else if (n % 12 < 9) then "white"
  else "black"

theorem marble_158th_is_gray : marble_color 157 = "gray" := 
by
  sorry

end marble_158th_is_gray_l1026_102628


namespace number_solution_l1026_102604

-- Statement based on identified conditions and answer
theorem number_solution (x : ℝ) (h : 0.10 * 0.30 * 0.50 * x = 90) : x = 6000 :=
by
  -- Skip the proof
  sorry

end number_solution_l1026_102604


namespace find_f_l1026_102661

-- Define the conditions
def g (x : ℝ) : ℝ := 2 * x + 3
def f (x : ℝ) : ℝ := g (x + 2)

-- State the theorem
theorem find_f :
  ∀ x : ℝ, f x = 2 * x + 7 :=
by
  sorry

end find_f_l1026_102661


namespace cos_sum_zero_l1026_102651

noncomputable def cos_sum : ℂ :=
  Real.cos (Real.pi / 15) + Real.cos (4 * Real.pi / 15) + Real.cos (7 * Real.pi / 15) + Real.cos (10 * Real.pi / 15)

theorem cos_sum_zero : cos_sum = 0 := by
  sorry

end cos_sum_zero_l1026_102651


namespace slope_angle_at_point_l1026_102689

def f (x : ℝ) : ℝ := 2 * x^3 - 7 * x + 2

theorem slope_angle_at_point :
  let deriv_f := fun x : ℝ => 6 * x^2 - 7
  let slope := deriv_f 1
  let angle := Real.arctan slope
  angle = (3 * Real.pi) / 4 :=
by
  sorry

end slope_angle_at_point_l1026_102689


namespace bonnets_difference_thursday_monday_l1026_102636

variable (Bm Bt Bf : ℕ)

-- Conditions
axiom monday_bonnets_made : Bm = 10
axiom tuesday_wednesday_bonnets_made : Bm + (2 * Bm) = 30
axiom bonnets_sent_to_orphanages : (Bm + Bt + (Bt - 5) + Bm + (2 * Bm)) / 5 = 11
axiom friday_bonnets_made : Bf = Bt - 5

theorem bonnets_difference_thursday_monday :
  Bt - Bm = 5 :=
sorry

end bonnets_difference_thursday_monday_l1026_102636


namespace foci_of_ellipse_l1026_102643

-- Define the ellipsis
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 25) = 1

-- Prove the coordinates of foci of the ellipse
theorem foci_of_ellipse :
  ∃ c : ℝ, c = 3 ∧ ((0, c) ∈ {p : ℝ × ℝ | ellipse p.1 p.2} ∧ (0, -c) ∈ {p : ℝ × ℝ | ellipse p.1 p.2}) :=
by
  sorry

end foci_of_ellipse_l1026_102643


namespace jack_initial_money_l1026_102627

-- Define the cost of one pair of socks
def cost_pair_socks : ℝ := 9.50

-- Define the cost of soccer shoes
def cost_soccer_shoes : ℝ := 92

-- Define the additional money Jack needs
def additional_money_needed : ℝ := 71

-- Define the total cost of two pairs of socks and one pair of soccer shoes
def total_cost : ℝ := 2 * cost_pair_socks + cost_soccer_shoes

-- Theorem to prove Jack's initial money
theorem jack_initial_money : ∃ m : ℝ, total_cost - additional_money_needed = 40 :=
by
  sorry

end jack_initial_money_l1026_102627


namespace sqrt_solution_range_l1026_102645

theorem sqrt_solution_range : 
  7 < (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) ∧ (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) < 8 := 
by
  sorry

end sqrt_solution_range_l1026_102645


namespace average_marks_l1026_102638

theorem average_marks (total_students : ℕ) (first_group : ℕ) (first_group_marks : ℕ)
                      (second_group : ℕ) (second_group_marks_diff : ℕ) (third_group_marks : ℕ)
                      (total_marks : ℕ) (class_average : ℕ) :
  total_students = 50 → 
  first_group = 10 → 
  first_group_marks = 90 → 
  second_group = 15 → 
  second_group_marks_diff = 10 → 
  third_group_marks = 60 →
  total_marks = (first_group * first_group_marks) + (second_group * (first_group_marks - second_group_marks_diff)) + ((total_students - (first_group + second_group)) * third_group_marks) →
  class_average = total_marks / total_students →
  class_average = 72 :=
by
  intros
  sorry

end average_marks_l1026_102638


namespace equation_of_line_l1026_102677

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  (dot_product / norm_sq * v.1, dot_product / norm_sq * v.2)

theorem equation_of_line (x y : ℝ) :
  projection (x, y) (7, 3) = (-7, -3) →
  y = -7/3 * x - 58/3 :=
by
  intro h
  sorry

end equation_of_line_l1026_102677


namespace circle_representation_l1026_102655

theorem circle_representation (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 + y^2 + x + 2*m*y + m = 0)) → m ≠ 1/2 :=
by
  sorry

end circle_representation_l1026_102655


namespace interest_rate_l1026_102612

theorem interest_rate (P1 P2 I T1 T2 total_amount : ℝ) (r : ℝ) :
  P1 = 10000 →
  P2 = 22000 →
  T1 = 2 →
  T2 = 3 →
  total_amount = 27160 →
  (I = P1 * r * T1 / 100 + P2 * r * T2 / 100) →
  P1 + P2 = 22000 →
  (P1 + I = total_amount) →
  r = 6 :=
by
  intros hP1 hP2 hT1 hT2 htotal_amount hI hP_total hP1_I_total
  -- Actual proof would go here
  sorry

end interest_rate_l1026_102612


namespace remaining_pie_after_carlos_and_maria_l1026_102660

theorem remaining_pie_after_carlos_and_maria (C M R : ℝ) (hC : C = 0.60) (hM : M = 0.25 * (1 - C)) : R = 1 - C - M → R = 0.30 :=
by
  intro hR
  simp only [hC, hM] at hR
  sorry

end remaining_pie_after_carlos_and_maria_l1026_102660


namespace find_x_parallel_vectors_l1026_102663

theorem find_x_parallel_vectors :
  ∀ x : ℝ, (∃ k : ℝ, (1, 2) = (k * (2 * x), k * (-3))) → x = -3 / 4 :=
by
  sorry

end find_x_parallel_vectors_l1026_102663


namespace gcd_840_1764_l1026_102607

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by 
  sorry

end gcd_840_1764_l1026_102607


namespace jenny_spent_fraction_l1026_102609

theorem jenny_spent_fraction
  (x : ℝ) -- The original amount of money Jenny had
  (h_half_x : 1/2 * x = 21) -- Half of the original amount is $21
  (h_left_money : x - 24 = 24) -- Jenny had $24 left after spending
  : (x - 24) / x = 3 / 7 := sorry

end jenny_spent_fraction_l1026_102609


namespace magnification_proof_l1026_102601

-- Define the conditions: actual diameter of the tissue and diameter of the magnified image
def actual_diameter := 0.0002
def magnified_diameter := 0.2

-- Define the magnification factor
def magnification_factor := magnified_diameter / actual_diameter

-- Prove that the magnification factor is 1000
theorem magnification_proof : magnification_factor = 1000 := by
  unfold magnification_factor
  unfold magnified_diameter
  unfold actual_diameter
  norm_num
  sorry

end magnification_proof_l1026_102601


namespace sum_is_eighteen_or_twentyseven_l1026_102656

theorem sum_is_eighteen_or_twentyseven :
  ∀ (A B C D E I J K L M : ℕ),
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ I ∧ A ≠ J ∧ A ≠ K ∧ A ≠ L ∧ A ≠ M ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ I ∧ B ≠ J ∧ B ≠ K ∧ B ≠ L ∧ B ≠ M ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ I ∧ C ≠ J ∧ C ≠ K ∧ C ≠ L ∧ C ≠ M ∧
  D ≠ E ∧ D ≠ I ∧ D ≠ J ∧ D ≠ K ∧ D ≠ L ∧ D ≠ M ∧
  E ≠ I ∧ E ≠ J ∧ E ≠ K ∧ E ≠ L ∧ E ≠ M ∧
  I ≠ J ∧ I ≠ K ∧ I ≠ L ∧ I ≠ M ∧
  J ≠ K ∧ J ≠ L ∧ J ≠ M ∧
  K ≠ L ∧ K ≠ M ∧
  L ≠ M ∧
  (0 < I) ∧ (0 < J) ∧ (0 < K) ∧ (0 < L) ∧ (0 < M) ∧
  A + B + C + D + E + I + J + K + L + M = 45 ∧
  (I + J + K + L + M) % 10 = 0 →
  A + B + C + D + E + (I + J + K + L + M) / 10 = 18 ∨
  A + B + C + D + E + (I + J + K + L + M) / 10 = 27 :=
by
  intros
  sorry

end sum_is_eighteen_or_twentyseven_l1026_102656


namespace simplify_fraction_120_1800_l1026_102693

theorem simplify_fraction_120_1800 :
  (120 : ℚ) / 1800 = (1 : ℚ) / 15 := by
  sorry

end simplify_fraction_120_1800_l1026_102693


namespace sum_arithmetic_seq_l1026_102646

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end sum_arithmetic_seq_l1026_102646


namespace average_first_n_numbers_eq_10_l1026_102664

theorem average_first_n_numbers_eq_10 (n : ℕ) 
  (h : (n * (n + 1)) / (2 * n) = 10) : n = 19 :=
  sorry

end average_first_n_numbers_eq_10_l1026_102664


namespace solve_fractions_l1026_102694

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end solve_fractions_l1026_102694


namespace missed_angle_l1026_102668

theorem missed_angle (sum_calculated : ℕ) (missed_angle_target : ℕ) 
  (h1 : sum_calculated = 2843) 
  (h2 : missed_angle_target = 37) : 
  ∃ n : ℕ, (sum_calculated + missed_angle_target = n * 180) :=
by {
  sorry
}

end missed_angle_l1026_102668


namespace food_sufficient_days_l1026_102676

theorem food_sufficient_days (D : ℕ) (h1 : 1000 * D - 10000 = 800 * D) : D = 50 :=
sorry

end food_sufficient_days_l1026_102676


namespace number_in_interval_l1026_102680

def number := 0.2012
def lower_bound := 0.2
def upper_bound := 0.25

theorem number_in_interval : lower_bound < number ∧ number < upper_bound :=
by
  sorry

end number_in_interval_l1026_102680


namespace maximum_value_expr_l1026_102603

theorem maximum_value_expr :
  ∀ (a b c d : ℝ), (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) ∧ (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1) →
  a + b + c + d - a * b - b * c - c * d - d * a ≤ 2 :=
by
  intros a b c d h
  sorry

end maximum_value_expr_l1026_102603


namespace predicted_yield_of_rice_l1026_102684

theorem predicted_yield_of_rice (x : ℝ) (h : x = 80) : 5 * x + 250 = 650 :=
by {
  sorry -- proof will be given later
}

end predicted_yield_of_rice_l1026_102684


namespace solution_set_for_inequality_l1026_102623

theorem solution_set_for_inequality : {x : ℝ | x ≠ 0 ∧ (x-1)/x ≤ 0} = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end solution_set_for_inequality_l1026_102623


namespace binary_arithmetic_l1026_102682

theorem binary_arithmetic :
  (110010:ℕ) * (1100:ℕ) / (100:ℕ) / (10:ℕ) = 100100 :=
by sorry

end binary_arithmetic_l1026_102682


namespace distinct_license_plates_l1026_102662

theorem distinct_license_plates :
  let num_digits := 10
  let num_letters := 26
  let num_digit_positions := 5
  let num_letter_pairs := num_letters * num_letters
  let num_letter_positions := num_digit_positions + 1
  num_digits^num_digit_positions * num_letter_pairs * num_letter_positions = 40560000 := by
  sorry

end distinct_license_plates_l1026_102662


namespace solution_l1026_102616

-- Define the conditions
variable (f : ℝ → ℝ)
variable (f_odd : ∀ x, f (-x) = -f x)
variable (f_periodic : ∀ x, f (x + 1) = f (1 - x))
variable (f_cubed : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x ^ 3)

-- Define the goal
theorem solution : f 2019 = -1 :=
by sorry

end solution_l1026_102616


namespace number_of_ways_to_choose_one_book_l1026_102621

theorem number_of_ways_to_choose_one_book:
  let chinese_books := 10
  let english_books := 7
  let mathematics_books := 5
  chinese_books + english_books + mathematics_books = 22 := by
    -- The actual proof should go here.
    sorry

end number_of_ways_to_choose_one_book_l1026_102621


namespace sam_wins_l1026_102629

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end sam_wins_l1026_102629


namespace midpoint_pentagon_inequality_l1026_102600

noncomputable def pentagon_area_midpoints (T : ℝ) : ℝ := sorry

theorem midpoint_pentagon_inequality {T t : ℝ} 
  (h1 : t = pentagon_area_midpoints T)
  (h2 : 0 < T) : 
  (3/4) * T > t ∧ t > (1/2) * T :=
  sorry

end midpoint_pentagon_inequality_l1026_102600


namespace fraction_equivalence_1_algebraic_identity_l1026_102602

/-- First Problem: Prove the equivalence of the fractions 171717/252525 and 17/25. -/
theorem fraction_equivalence_1 : 
  (171717 : ℚ) / 252525 = 17 / 25 := 
sorry

/-- Second Problem: Prove the equivalence of the algebraic expressions on both sides. -/
theorem algebraic_identity (a b : ℚ) : 
  2 * b^5 + (a^4 + a^3 * b + a^2 * b^2 + a * b^3 + b^4) * (a - b) = 
  (a^4 - a^3 * b + a^2 * b^2 - a * b^3 + b^4) * (a + b) := 
sorry

end fraction_equivalence_1_algebraic_identity_l1026_102602


namespace toy_cost_price_and_profit_l1026_102619

-- Define the cost price of type A toy
def cost_A (x : ℝ) : ℝ := x

-- Define the cost price of type B toy
def cost_B (x : ℝ) : ℝ := 1.5 * x

-- Spending conditions
def spending_A (x : ℝ) (num_A : ℝ) : Prop := num_A = 1200 / x
def spending_B (x : ℝ) (num_B : ℝ) : Prop := num_B = 1500 / (1.5 * x)

-- Quantity difference condition
def quantity_difference (num_A num_B : ℝ) : Prop := num_A - num_B = 20

-- Selling prices
def selling_price_A : ℝ := 12
def selling_price_B : ℝ := 20

-- Total toys purchased condition
def total_toys (num_A num_B : ℝ) : Prop := num_A + num_B = 75

-- Profit condition
def profit_condition (num_A num_B cost_A cost_B : ℝ) : Prop :=
  (selling_price_A - cost_A) * num_A + (selling_price_B - cost_B) * num_B ≥ 300

theorem toy_cost_price_and_profit :
  ∃ (x : ℝ), 
  cost_A x = 10 ∧
  cost_B x = 15 ∧
  ∀ (num_A num_B : ℝ),
  spending_A x num_A →
  spending_B x num_B →
  quantity_difference num_A num_B →
  total_toys num_A num_B →
  profit_condition num_A num_B (cost_A x) (cost_B x) →
  num_A ≤ 25 :=
by
  sorry

end toy_cost_price_and_profit_l1026_102619


namespace ratio_costs_equal_l1026_102675

noncomputable def cost_first_8_years : ℝ := 10000 * 8
noncomputable def john_share_first_8_years : ℝ := cost_first_8_years / 2
noncomputable def university_tuition : ℝ := 250000
noncomputable def john_share_university : ℝ := university_tuition / 2
noncomputable def total_paid_by_john : ℝ := 265000
noncomputable def cost_between_8_and_18 : ℝ := total_paid_by_john - john_share_first_8_years - john_share_university
noncomputable def cost_per_year_8_to_18 : ℝ := cost_between_8_and_18 / 10
noncomputable def cost_per_year_first_8_years : ℝ := 10000

theorem ratio_costs_equal : cost_per_year_8_to_18 / cost_per_year_first_8_years = 1 := by
  sorry

end ratio_costs_equal_l1026_102675


namespace time_to_finish_work_l1026_102665

theorem time_to_finish_work (a b c : ℕ) (h1 : 1/a + 1/9 + 1/18 = 1/4) : a = 12 :=
by
  sorry

end time_to_finish_work_l1026_102665


namespace oranges_weight_is_10_l1026_102620

def applesWeight (A : ℕ) : ℕ := A
def orangesWeight (A : ℕ) : ℕ := 5 * A
def totalWeight (A : ℕ) (O : ℕ) : ℕ := A + O
def totalCost (A : ℕ) (x : ℕ) (O : ℕ) (y : ℕ) : ℕ := A * x + O * y

theorem oranges_weight_is_10 (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := by
  sorry

end oranges_weight_is_10_l1026_102620


namespace juniors_more_than_seniors_l1026_102648

theorem juniors_more_than_seniors
  (j s : ℕ)
  (h1 : (1 / 3) * j = (2 / 3) * s)
  (h2 : j + s = 300) :
  j - s = 100 := 
sorry

end juniors_more_than_seniors_l1026_102648


namespace sufficient_condition_for_quadratic_l1026_102678

theorem sufficient_condition_for_quadratic (a : ℝ) : 
  (∃ (x : ℝ), (x > a) ∧ (x^2 - 5*x + 6 ≥ 0)) ∧ 
  (¬(∀ (x : ℝ), (x^2 - 5*x + 6 ≥ 0) → (x > a))) ↔ 
  a ≥ 3 :=
by
  sorry

end sufficient_condition_for_quadratic_l1026_102678


namespace min_n_constant_term_l1026_102652

theorem min_n_constant_term (x : ℕ) (hx : x > 0) : 
  ∃ n : ℕ, 
  (∀ r : ℕ, (2 * n = 5 * r) → n ≥ 5) ∧ 
  (∃ r : ℕ, (2 * n = 5 * r) ∧ n = 5) := by
  sorry

end min_n_constant_term_l1026_102652


namespace value_of_a_plus_b_l1026_102637

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 5) (h2 : |b| = 2) (h3 : a < b) : a + b = -3 := by
  -- Proof goes here
  sorry

end value_of_a_plus_b_l1026_102637


namespace X_is_N_l1026_102699

theorem X_is_N (X : Set ℕ) (h_nonempty : ∃ x, x ∈ X)
  (h_condition1 : ∀ x ∈ X, 4 * x ∈ X)
  (h_condition2 : ∀ x ∈ X, Nat.floor (Real.sqrt x) ∈ X) : 
  X = Set.univ := 
sorry

end X_is_N_l1026_102699


namespace clever_calculation_part1_clever_calculation_part2_clever_calculation_part3_l1026_102688

-- Prove that 46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 equals 56.056
theorem clever_calculation_part1 : 46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 = 56.056 :=
by
sorry

-- Prove that 101 * 92 - 92 equals 9200
theorem clever_calculation_part2 : 101 * 92 - 92 = 9200 :=
by
sorry

-- Prove that 36000 / 125 / 8 equals 36
theorem clever_calculation_part3 : 36000 / 125 / 8 = 36 :=
by
sorry

end clever_calculation_part1_clever_calculation_part2_clever_calculation_part3_l1026_102688


namespace arc_length_of_sector_l1026_102635

theorem arc_length_of_sector (r A l : ℝ) (h_r : r = 2) (h_A : A = π / 3) (h_area : A = 1 / 2 * r * l) : l = π / 3 :=
by
  rw [h_r, h_A] at h_area
  sorry

end arc_length_of_sector_l1026_102635


namespace original_price_of_shirt_l1026_102632

theorem original_price_of_shirt (P : ℝ) (h : 0.5625 * P = 18) : P = 32 := 
by 
sorry

end original_price_of_shirt_l1026_102632


namespace find_ratio_l1026_102683

noncomputable def complex_numbers_are_non_zero (x y z : ℂ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0

noncomputable def sum_is_30 (x y z : ℂ) : Prop :=
x + y + z = 30

noncomputable def expanded_equality (x y z : ℂ) : Prop :=
((x - y)^2 + (x - z)^2 + (y - z)^2) * (x + y + z) = x * y * z

theorem find_ratio (x y z : ℂ)
  (h1 : complex_numbers_are_non_zero x y z)
  (h2 : sum_is_30 x y z)
  (h3 : expanded_equality x y z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3.5 :=
sorry

end find_ratio_l1026_102683


namespace bottle_caps_total_l1026_102614

def initial_bottle_caps := 51.0
def given_bottle_caps := 36.0

theorem bottle_caps_total : initial_bottle_caps + given_bottle_caps = 87.0 := by
  sorry

end bottle_caps_total_l1026_102614


namespace coefficient_of_x2_in_expansion_l1026_102669

def binomial_coefficient (n k : Nat) : Nat := Nat.choose k n

def binomial_term (a x : ℕ) (n r : ℕ) : ℕ :=
  a^(n-r) * binomial_coefficient n r * x^r

theorem coefficient_of_x2_in_expansion : 
  binomial_term 2 1 5 2 = 80 := by sorry

end coefficient_of_x2_in_expansion_l1026_102669


namespace trapezium_area_l1026_102697

-- Definitions based on the problem conditions
def length_side_a : ℝ := 20
def length_side_b : ℝ := 18
def distance_between_sides : ℝ := 15

-- Statement of the proof problem
theorem trapezium_area :
  (1 / 2 * (length_side_a + length_side_b) * distance_between_sides) = 285 := by
  sorry

end trapezium_area_l1026_102697


namespace employed_population_percentage_l1026_102617

noncomputable def percent_population_employed (total_population employed_males employed_females : ℝ) : ℝ :=
  employed_males + employed_females

theorem employed_population_percentage (population employed_males_percentage employed_females_percentage : ℝ) 
  (h1 : employed_males_percentage = 0.36 * population)
  (h2 : employed_females_percentage = 0.36 * population)
  (h3 : employed_females_percentage + employed_males_percentage = 0.50 * total_population)
  : total_population = 0.72 * population :=
by 
  sorry

end employed_population_percentage_l1026_102617


namespace square_tile_area_l1026_102625

-- Definition and statement of the problem
theorem square_tile_area (side_length : ℝ) (h : side_length = 7) : 
  (side_length * side_length) = 49 :=
by
  sorry

end square_tile_area_l1026_102625


namespace calculate_g_inv_l1026_102622

noncomputable def g : ℤ → ℤ := sorry
noncomputable def g_inv : ℤ → ℤ := sorry

axiom g_inv_eq : ∀ x, g (g_inv x) = x

axiom cond1 : g (-1) = 2
axiom cond2 : g (0) = 3
axiom cond3 : g (1) = 6

theorem calculate_g_inv : 
  g_inv (g_inv 6 - g_inv 2) = -1 := 
by
  -- The proof goes here
  sorry

end calculate_g_inv_l1026_102622


namespace remaining_surface_area_correct_l1026_102679

open Real

-- Define the original cube and the corner cubes
def orig_cube : ℝ × ℝ × ℝ := (5, 5, 5)
def corner_cube : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define a function to compute the surface area of a cube given dimensions (a, b, c)
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)

-- Original surface area of the cube
def orig_surface_area : ℝ := surface_area 5 5 5

-- Total surface area of the remaining figure after removing 8 corner cubes
def remaining_surface_area : ℝ := 150  -- Calculated directly as 6 * 25

-- Theorem stating that the surface area of the remaining figure is 150 cm^2
theorem remaining_surface_area_correct :
  remaining_surface_area = 150 := sorry

end remaining_surface_area_correct_l1026_102679


namespace arithmetic_geometric_sequence_S30_l1026_102657

variable (S : ℕ → ℝ)

theorem arithmetic_geometric_sequence_S30 :
  S 10 = 10 →
  S 20 = 30 →
  S 30 = 70 := by
  intros h1 h2
  -- proof steps go here
  sorry

end arithmetic_geometric_sequence_S30_l1026_102657


namespace math_problem_l1026_102644

variables {a b : ℝ}
open Real

theorem math_problem (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (a - 1) * (b - 1) = 1 ∧ 
  (∀ b : ℝ, (a = 2 * b → a + 4 * b = 9)) ∧ 
  (∀ b : ℝ, (b = 3 → (1 / a^2 + 2 / b^2) = 2 / 3)) :=
by
  sorry

end math_problem_l1026_102644


namespace geometric_series_sum_l1026_102606

theorem geometric_series_sum : 
    ∑' n : ℕ, (1 : ℝ) * (-1 / 2) ^ n = 2 / 3 :=
by
    sorry

end geometric_series_sum_l1026_102606


namespace length_of_LO_l1026_102615

theorem length_of_LO (MN LO : ℝ) (alt_O_MN alt_N_LO : ℝ) (h_MN : MN = 15) 
  (h_alt_O_MN : alt_O_MN = 9) (h_alt_N_LO : alt_N_LO = 7) : 
  LO = 19 + 2 / 7 :=
by
  -- Sorry means to skip the proof.
  sorry

end length_of_LO_l1026_102615


namespace algebraic_expression_value_l1026_102673

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) : 
  x^2 - 4 * y^2 = -4 :=
by
  sorry

end algebraic_expression_value_l1026_102673


namespace number_that_divides_and_leaves_remainder_54_l1026_102640

theorem number_that_divides_and_leaves_remainder_54 :
  ∃ n : ℕ, n > 0 ∧ (55 ^ 55 + 55) % n = 54 ∧ n = 56 :=
by
  sorry

end number_that_divides_and_leaves_remainder_54_l1026_102640


namespace radius_of_surrounding_circles_is_correct_l1026_102650

noncomputable def r : Real := 1 + Real.sqrt 2

theorem radius_of_surrounding_circles_is_correct (r: ℝ)
  (h₁: ∃c : ℝ, c = 2) -- central circle radius is 2
  (h₂: ∃far: ℝ, far = (1 + (Real.sqrt 2))) -- r is the solution as calculated
: 2 * r = 1 + Real.sqrt 2 :=
by
  sorry

end radius_of_surrounding_circles_is_correct_l1026_102650


namespace sale_in_third_month_l1026_102642

theorem sale_in_third_month (
  f1 f2 f4 f5 f6 average : ℕ
) (h1 : f1 = 7435) 
  (h2 : f2 = 7927) 
  (h4 : f4 = 8230) 
  (h5 : f5 = 7562) 
  (h6 : f6 = 5991) 
  (havg : average = 7500) :
  ∃ f3, f3 = 7855 ∧ f1 + f2 + f3 + f4 + f5 + f6 = average * 6 :=
by {
  sorry
}

end sale_in_third_month_l1026_102642


namespace total_gas_cost_l1026_102671

def gas_price_station_1 : ℝ := 3
def gas_price_station_2 : ℝ := 3.5
def gas_price_station_3 : ℝ := 4
def gas_price_station_4 : ℝ := 4.5
def tank_capacity : ℝ := 12

theorem total_gas_cost :
  let cost_station_1 := tank_capacity * gas_price_station_1
  let cost_station_2 := tank_capacity * gas_price_station_2
  let cost_station_3 := tank_capacity * gas_price_station_3
  let cost_station_4 := tank_capacity * gas_price_station_4
  cost_station_1 + cost_station_2 + cost_station_3 + cost_station_4 = 180 :=
by
  -- Proof is skipped
  sorry

end total_gas_cost_l1026_102671
