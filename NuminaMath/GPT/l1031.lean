import Mathlib

namespace sqrt_of_n_is_integer_l1031_103163

theorem sqrt_of_n_is_integer (n : ℕ) (h : ∀ p, (0 ≤ p ∧ p < n) → ∃ m g, m + g = n ∧ (m - g) * (m - g) = n) :
  ∃ k : ℕ, k * k = n :=
by 
  sorry

end sqrt_of_n_is_integer_l1031_103163


namespace identify_1000g_weight_l1031_103132

-- Define the masses of the weights
def masses : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- The statement that needs to be proven
theorem identify_1000g_weight (masses : List ℕ) (h : masses = [1000, 1001, 1002, 1004, 1007]) :
  ∃ w, w ∈ masses ∧ w = 1000 ∧ by sorry :=
sorry

end identify_1000g_weight_l1031_103132


namespace positive_divisors_d17_l1031_103121

theorem positive_divisors_d17 (n : ℕ) (d : ℕ → ℕ) (k : ℕ) (h_order : d 1 = 1 ∧ ∀ i, 1 ≤ i → i ≤ k → d i < d (i + 1)) 
  (h_last : d k = n) (h_pythagorean : d 7 ^ 2 + d 15 ^ 2 = d 16 ^ 2) : 
  d 17 = 28 :=
sorry

end positive_divisors_d17_l1031_103121


namespace Mitch_hourly_rate_l1031_103143

theorem Mitch_hourly_rate :
  let weekday_hours := 5 * 5
  let weekend_hours := 3 * 2
  let equivalent_weekend_hours := weekend_hours * 2
  let total_hours := weekday_hours + equivalent_weekend_hours
  let weekly_earnings := 111
  weekly_earnings / total_hours = 3 :=
by
  let weekday_hours := 5 * 5
  let weekend_hours := 3 * 2
  let equivalent_weekend_hours := weekend_hours * 2
  let total_hours := weekday_hours + equivalent_weekend_hours
  let weekly_earnings := 111
  sorry

end Mitch_hourly_rate_l1031_103143


namespace range_of_a_l1031_103167

theorem range_of_a {x y a : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + y + 6 = 4 * x * y) : a ≤ 10 / 3 :=
  sorry

end range_of_a_l1031_103167


namespace twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l1031_103155

variable {m n : ℕ}

def P (m : ℕ) : ℕ := 2^m
def Q (n : ℕ) : ℕ := 3^n

theorem twelve_pow_mn_eq_P_pow_2n_Q_pow_m (m n : ℕ) : 12^(m * n) = (P m)^(2 * n) * (Q n)^m := 
sorry

end twelve_pow_mn_eq_P_pow_2n_Q_pow_m_l1031_103155


namespace imo_1989_q6_l1031_103124

-- Define the odd integer m greater than 2
def isOdd (m : ℕ) := ∃ k : ℤ, m = 2 * k + 1

-- Define the condition for divisibility
def smallest_n (m : ℕ) (k : ℕ) (p : ℕ) : ℕ :=
  if k ≤ 1989 then 2 ^ (1989 - k) else 1

theorem imo_1989_q6 
  (m : ℕ) (h_m_gt2 : m > 2) (h_m_odd : isOdd m) (k : ℕ) (p : ℕ) (h_m_form : m = 2^k * p - 1) (h_p_odd : isOdd p) (h_k_gt1 : k > 1) :
  ∃ n : ℕ, (2^1989 ∣ m^n - 1) ∧ n = smallest_n m k p :=
by
  sorry

end imo_1989_q6_l1031_103124


namespace cosine_greater_sine_cosine_cos_greater_sine_sin_l1031_103104

variable {f g : ℝ → ℝ}

-- Problem 1
theorem cosine_greater_sine (h : ∀ x, - (Real.pi / 2) < f x + g x ∧ f x + g x < Real.pi / 2
                            ∧ - (Real.pi / 2) < f x - g x ∧ f x - g x < Real.pi / 2) :
  ∀ x, Real.cos (f x) > Real.sin (g x) :=
sorry

-- Problem 2
theorem cosine_cos_greater_sine_sin (x : ℝ) :  Real.cos (Real.cos x) > Real.sin (Real.sin x) :=
sorry

end cosine_greater_sine_cosine_cos_greater_sine_sin_l1031_103104


namespace wage_ratio_l1031_103102

-- Define the conditions
variable (M W : ℝ) -- M stands for man's daily wage, W stands for woman's daily wage
variable (h1 : 40 * 10 * M = 14400) -- Condition 1: 40 men working for 10 days earn Rs. 14400
variable (h2 : 40 * 30 * W = 21600) -- Condition 2: 40 women working for 30 days earn Rs. 21600

-- The statement to prove
theorem wage_ratio (h1 : 40 * 10 * M = 14400) (h2 : 40 * 30 * W = 21600) : M / W = 2 := by
  sorry

end wage_ratio_l1031_103102


namespace watermelons_left_to_be_sold_tomorrow_l1031_103160

def initial_watermelons : ℕ := 10 * 12
def sold_yesterday : ℕ := initial_watermelons * 40 / 100
def remaining_after_yesterday : ℕ := initial_watermelons - sold_yesterday
def sold_today : ℕ := remaining_after_yesterday / 4
def remaining_after_today : ℕ := remaining_after_yesterday - sold_today

theorem watermelons_left_to_be_sold_tomorrow : remaining_after_today = 54 := 
by
  sorry

end watermelons_left_to_be_sold_tomorrow_l1031_103160


namespace envelope_width_l1031_103112

theorem envelope_width (Area Height Width : ℝ) (h_area : Area = 36) (h_height : Height = 6) (h_area_formula : Area = Width * Height) : Width = 6 :=
by
  sorry

end envelope_width_l1031_103112


namespace sum_of_first_n_terms_l1031_103195

variable (a_n : ℕ → ℝ) -- Sequence term
variable (S_n : ℕ → ℝ) -- Sum of first n terms

-- Conditions given in the problem
axiom sum_first_term : a_n 1 = 2
axiom sum_first_two_terms : a_n 1 + a_n 2 = 7
axiom sum_first_three_terms : a_n 1 + a_n 2 + a_n 3 = 18

-- Expected result to prove
theorem sum_of_first_n_terms 
  (h1 : S_n 1 = 2)
  (h2 : S_n 2 = 7)
  (h3 : S_n 3 = 18) :
  S_n n = (3/2) * ((n * (n + 1) * (2 * n + 1) / 6) - (n * (n + 1) / 2) + 2 * n) :=
sorry

end sum_of_first_n_terms_l1031_103195


namespace number_of_intersection_points_l1031_103116

-- Define the standard parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Define what it means for a line to be tangent to the parabola
def is_tangent (m : ℝ) (c : ℝ) : Prop :=
  ∃ x0 : ℝ, parabola x0 = m * x0 + c ∧ 2 * x0 = m

-- Define what it means for a line to intersect the parabola
def line_intersects_parabola (m : ℝ) (c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, parabola x1 = m * x1 + c ∧ parabola x2 = m * x2 + c

-- Main theorem statement
theorem number_of_intersection_points :
  (∃ m c : ℝ, is_tangent m c) → (∃ m' c' : ℝ, line_intersects_parabola m' c') →
  ∃ n : ℕ, n = 1 ∨ n = 2 ∨ n = 3 :=
sorry

end number_of_intersection_points_l1031_103116


namespace parabola_pass_through_fixed_point_l1031_103178

theorem parabola_pass_through_fixed_point
  (p : ℝ) (hp : p > 0)
  (xM yM : ℝ) (hM : (xM, yM) = (1, -2))
  (hMp : yM^2 = 2 * p * xM)
  (xA yA xC yC xB yB xD yD : ℝ)
  (hxA : xA = xC ∨ xA ≠ xC)
  (hxB : xB = xD ∨ xB ≠ xD)
  (x2 y0 : ℝ) (h : (x2, y0) = (2, 0))
  (m1 m2 : ℝ) (hm1m2 : m1 * m2 = -1)
  (l1_intersect_A : xA = m1 * yA + 2)
  (l1_intersect_C : xC = m1 * yC + 2)
  (l2_intersect_B : xB = m2 * yB + 2)
  (l2_intersect_D : xD = m2 * yD + 2)
  (hMidM : (2 * xA + 2 * xC = 4 * xM ∧ 2 * yA + 2 * yC = 4 * yM))
  (hMidN : (2 * xB + 2 * xD = 4 * xM ∧ 2 * yB + 2 * yD = 4 * yM)) :
  (yM^2 = 4 * xM) ∧ 
  (∃ k : ℝ, ∀ x : ℝ, y = k * x ↔ y = xM / (m1 + m2) ∧ y = m1) :=
sorry

end parabola_pass_through_fixed_point_l1031_103178


namespace total_soaking_time_l1031_103141

def stain_times (n_grass n_marinara n_coffee n_ink : Nat) (t_grass t_marinara t_coffee t_ink : Nat) : Nat :=
  n_grass * t_grass + n_marinara * t_marinara + n_coffee * t_coffee + n_ink * t_ink

theorem total_soaking_time :
  let shirt_grass_stains := 2
  let shirt_grass_time := 3
  let shirt_marinara_stains := 1
  let shirt_marinara_time := 7
  let pants_coffee_stains := 1
  let pants_coffee_time := 10
  let pants_ink_stains := 1
  let pants_ink_time := 5
  let socks_grass_stains := 1
  let socks_grass_time := 3
  let socks_marinara_stains := 2
  let socks_marinara_time := 7
  let socks_ink_stains := 1
  let socks_ink_time := 5
  let additional_ink_time := 2

  let shirt_time := stain_times shirt_grass_stains shirt_marinara_stains 0 0 shirt_grass_time shirt_marinara_time 0 0
  let pants_time := stain_times 0 0 pants_coffee_stains pants_ink_stains 0 0 pants_coffee_time pants_ink_time
  let socks_time := stain_times socks_grass_stains socks_marinara_stains 0 socks_ink_stains socks_grass_time socks_marinara_time 0 socks_ink_time
  let total_time := shirt_time + pants_time + socks_time
  let total_ink_stains := pants_ink_stains + socks_ink_stains
  let additional_ink_total_time := total_ink_stains * additional_ink_time
  let final_total_time := total_time + additional_ink_total_time

  final_total_time = 54 :=
by
  sorry

end total_soaking_time_l1031_103141


namespace clusters_of_oats_l1031_103110

-- Define conditions:
def clusters_per_spoonful : Nat := 4
def spoonfuls_per_bowl : Nat := 25
def bowls_per_box : Nat := 5

-- Define the question and correct answer:
def clusters_per_box : Nat :=
  clusters_per_spoonful * spoonfuls_per_bowl * bowls_per_box

-- Theorem statement for the proof problem:
theorem clusters_of_oats:
  clusters_per_box = 500 :=
by
  sorry

end clusters_of_oats_l1031_103110


namespace find_f_2010_l1031_103153

noncomputable def f : ℕ → ℤ := sorry

theorem find_f_2010 (f_prop : ∀ {a b n : ℕ}, a + b = 3 * 2^n → f a + f b = 2 * n^2) :
  f 2010 = 193 :=
sorry

end find_f_2010_l1031_103153


namespace green_peaches_per_basket_l1031_103196

/-- Define the conditions given in the problem. -/
def n_baskets : ℕ := 7
def n_red_each : ℕ := 10
def n_green_total : ℕ := 14

/-- Prove that there are 2 green peaches in each basket. -/
theorem green_peaches_per_basket : n_green_total / n_baskets = 2 := by
  sorry

end green_peaches_per_basket_l1031_103196


namespace largest_circle_at_A_l1031_103108

/--
Given a pentagon with side lengths AB = 16 cm, BC = 14 cm, CD = 17 cm, DE = 13 cm, and EA = 14 cm,
and given five circles with centers A, B, C, D, and E such that each pair of circles with centers at
the ends of a side of the pentagon touch on that side, the circle with center A
has the largest radius.
-/
theorem largest_circle_at_A
  (rA rB rC rD rE : ℝ) 
  (hAB : rA + rB = 16)
  (hBC : rB + rC = 14)
  (hCD : rC + rD = 17)
  (hDE : rD + rE = 13)
  (hEA : rE + rA = 14) :
  rA ≥ rB ∧ rA ≥ rC ∧ rA ≥ rD ∧ rA ≥ rE := 
sorry

end largest_circle_at_A_l1031_103108


namespace find_number_l1031_103192

theorem find_number (x : ℕ) : ((x * 12) / (180 / 3) + 70 = 71) → x = 5 :=
by
  sorry

end find_number_l1031_103192


namespace randy_trip_length_l1031_103150

-- Define the conditions
noncomputable def fraction_gravel := (1/4 : ℚ)
noncomputable def miles_pavement := (30 : ℚ)
noncomputable def fraction_dirt := (1/6 : ℚ)

-- The proof statement
theorem randy_trip_length :
  ∃ x : ℚ, (fraction_gravel + fraction_dirt + (miles_pavement / x) = 1) ∧ x = 360 / 7 := 
by
  sorry

end randy_trip_length_l1031_103150


namespace expression_divisible_by_16_l1031_103146

theorem expression_divisible_by_16 (m n : ℤ) : 
  ∃ k : ℤ, (5 * m + 3 * n + 1)^5 * (3 * m + n + 4)^4 = 16 * k :=
sorry

end expression_divisible_by_16_l1031_103146


namespace total_bathing_suits_l1031_103130

theorem total_bathing_suits (men_women_bathing_suits : Nat)
                            (men_bathing_suits : Nat := 14797)
                            (women_bathing_suits : Nat := 4969) :
    men_bathing_suits + women_bathing_suits = 19766 := by
  sorry

end total_bathing_suits_l1031_103130


namespace rachel_wrote_six_pages_l1031_103100

theorem rachel_wrote_six_pages
  (write_rate : ℕ)
  (research_time : ℕ)
  (editing_time : ℕ)
  (total_time : ℕ)
  (total_time_in_minutes : ℕ := total_time * 60)
  (actual_time_writing : ℕ := total_time_in_minutes - (research_time + editing_time))
  (pages_written : ℕ := actual_time_writing / write_rate) :
  write_rate = 30 →
  research_time = 45 →
  editing_time = 75 →
  total_time = 5 →
  pages_written = 6 :=
by
  intros h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  subst h4
  have h5 : total_time_in_minutes = 300 := by sorry
  have h6 : actual_time_writing = 180 := by sorry
  have h7 : pages_written = 6 := by sorry
  exact h7

end rachel_wrote_six_pages_l1031_103100


namespace blue_ball_higher_numbered_bin_l1031_103133

noncomputable def probability_higher_numbered_bin :
  ℝ := sorry

theorem blue_ball_higher_numbered_bin :
  probability_higher_numbered_bin = 7 / 16 :=
sorry

end blue_ball_higher_numbered_bin_l1031_103133


namespace sum_of_ratios_eq_four_l1031_103129

theorem sum_of_ratios_eq_four 
  (A B C D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E]
  (BD DC AE EB : ℝ)
  (h1 : BD = 2 * DC)
  (h2 : AE = 2 * EB) : 
  (BD / DC) + (AE / EB) = 4 :=
  sorry

end sum_of_ratios_eq_four_l1031_103129


namespace order_of_exponentials_l1031_103168

theorem order_of_exponentials :
  let a := 2^55
  let b := 3^44
  let c := 5^33
  let d := 6^22
  a < d ∧ d < b ∧ b < c :=
by
  let a := 2^55
  let b := 3^44
  let c := 5^33
  let d := 6^22
  sorry

end order_of_exponentials_l1031_103168


namespace find_fx_l1031_103165

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = 19 * x ^ 2 + 55 * x - 44) :
  ∀ x : ℝ, f x = 19 * x ^ 2 + 93 * x + 30 :=
by
  sorry

end find_fx_l1031_103165


namespace soccer_team_lineups_l1031_103106

-- Define the number of players in the team
def numPlayers : Nat := 16

-- Define the number of regular players to choose (excluding the goalie)
def numRegularPlayers : Nat := 10

-- Define the total number of starting lineups, considering the goalie and the combination of regular players
def totalStartingLineups : Nat :=
  numPlayers * Nat.choose (numPlayers - 1) numRegularPlayers

-- The theorem to prove
theorem soccer_team_lineups : totalStartingLineups = 48048 := by
  sorry

end soccer_team_lineups_l1031_103106


namespace determine_b_l1031_103109

theorem determine_b (a b c y1 y2 : ℝ) 
  (h1 : y1 = a * 2^2 + b * 2 + c)
  (h2 : y2 = a * (-2)^2 + b * (-2) + c)
  (h3 : y1 - y2 = -12) : 
  b = -3 := 
by
  sorry

end determine_b_l1031_103109


namespace find_ϕ_l1031_103135

noncomputable def f (ω ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem find_ϕ (ω ϕ : ℝ) (h1 : 0 < ω) (h2 : abs ϕ < Real.pi / 2) (h3 : ∀ x : ℝ, f ω ϕ (x + Real.pi / 6) = g ω x) 
  (h4 : 2 * Real.pi / ω = Real.pi) : ϕ = Real.pi / 3 :=
by sorry

end find_ϕ_l1031_103135


namespace total_present_ages_l1031_103171

variables (P Q : ℕ)

theorem total_present_ages :
  (P - 8 = (Q - 8) / 2) ∧ (P * 4 = Q * 3) → (P + Q = 28) :=
by
  sorry

end total_present_ages_l1031_103171


namespace rationalize_denominator_sum_A_B_C_D_l1031_103147

theorem rationalize_denominator :
  (1 / (5 : ℝ)^(1/3) - (2 : ℝ)^(1/3)) = 
  ((25 : ℝ)^(1/3) + (10 : ℝ)^(1/3) + (4 : ℝ)^(1/3)) / (3 : ℝ) := 
sorry

theorem sum_A_B_C_D : 25 + 10 + 4 + 3 = 42 := 
by norm_num

end rationalize_denominator_sum_A_B_C_D_l1031_103147


namespace statement_D_incorrect_l1031_103157

theorem statement_D_incorrect (a b c : ℝ) : a^2 > b^2 ∧ a * b > 0 → ¬(1 / a < 1 / b) :=
by sorry

end statement_D_incorrect_l1031_103157


namespace grandmother_dolls_l1031_103101

-- Define the conditions
variable (S G : ℕ)

-- Rene has three times as many dolls as her sister
def rene_dolls : ℕ := 3 * S

-- The sister has two more dolls than their grandmother
def sister_dolls_eq : Prop := S = G + 2

-- Together they have a total of 258 dolls
def total_dolls : Prop := (rene_dolls S) + S + G = 258

-- Prove that the grandmother has 50 dolls given the conditions
theorem grandmother_dolls : sister_dolls_eq S G → total_dolls S G → G = 50 :=
by
  intros h1 h2
  sorry

end grandmother_dolls_l1031_103101


namespace correct_operation_l1031_103140

variable (x y a : ℝ)

lemma correct_option_C :
  -4 * x^5 * y^3 / (2 * x^3 * y) = -2 * x^2 * y^2 :=
by sorry

lemma wrong_option_A :
  x * (2 * x + 3) ≠ 2 * x^2 + 3 :=
by sorry

lemma wrong_option_B :
  a^2 + a^3 ≠ a^5 :=
by sorry

lemma wrong_option_D :
  x^3 * x^2 ≠ x^6 :=
by sorry

theorem correct_operation :
  ((-4 * x^5 * y^3 / (2 * x^3 * y) = -2 * x^2 * y^2) ∧
   (x * (2 * x + 3) ≠ 2 * x^2 + 3) ∧
   (a^2 + a^3 ≠ a^5) ∧
   (x^3 * x^2 ≠ x^6)) :=
by
  exact ⟨correct_option_C x y, wrong_option_A x, wrong_option_B a, wrong_option_D x⟩

end correct_operation_l1031_103140


namespace number_times_half_squared_is_eight_l1031_103181

noncomputable def num : ℝ := 32

theorem number_times_half_squared_is_eight :
  (num * (1 / 2) ^ 2 = 2 ^ 3) :=
by
  sorry

end number_times_half_squared_is_eight_l1031_103181


namespace average_infection_l1031_103119

theorem average_infection (x : ℕ) (h : 1 + 2 * x + x^2 = 121) : x = 10 :=
by
  sorry -- Proof to be filled.

end average_infection_l1031_103119


namespace range_of_a_l1031_103161

theorem range_of_a {a : ℝ} : (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≤ -x2^2 + 4*a*x2)
  ∨ (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≥ -x2^2 + 4*a*x2) ↔ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l1031_103161


namespace proof_statement_d_is_proposition_l1031_103137

-- Define the conditions
def statement_a := "Do two points determine a line?"
def statement_b := "Take a point M on line AB"
def statement_c := "In the same plane, two lines do not intersect"
def statement_d := "The sum of two acute angles is greater than a right angle"

-- Define the property of being a proposition
def is_proposition (s : String) : Prop :=
  s ≠ "Do two points determine a line?" ∧
  s ≠ "Take a point M on line AB" ∧
  s ≠ "In the same plane, two lines do not intersect"

-- The equivalence proof that statement_d is the only proposition
theorem proof_statement_d_is_proposition :
  is_proposition statement_d ∧
  ¬is_proposition statement_a ∧
  ¬is_proposition statement_b ∧
  ¬is_proposition statement_c := by
  sorry

end proof_statement_d_is_proposition_l1031_103137


namespace division_addition_correct_l1031_103105

theorem division_addition_correct : 0.2 / 0.005 + 0.1 = 40.1 :=
by
  sorry

end division_addition_correct_l1031_103105


namespace liz_car_percentage_sale_l1031_103131

theorem liz_car_percentage_sale (P : ℝ) (h1 : 30000 = P - 2500) (h2 : 26000 = P * (80 / 100)) : 80 = 80 :=
by 
  sorry

end liz_car_percentage_sale_l1031_103131


namespace ratio_of_p_to_q_l1031_103114

theorem ratio_of_p_to_q (p q r : ℚ) (h1: p = r * q) (h2: 18 / 7 + (2 * q - p) / (2 * q + p) = 3) : r = 29 / 10 :=
by
  sorry

end ratio_of_p_to_q_l1031_103114


namespace problem_inequality_l1031_103198

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x - y + z) * (y - z + x) * (z - x + y) ≤ x * y * z := sorry

end problem_inequality_l1031_103198


namespace m_not_in_P_l1031_103180

noncomputable def m : ℝ := Real.sqrt 3
def P : Set ℝ := { x | x^2 - Real.sqrt 2 * x ≤ 0 }

theorem m_not_in_P : m ∉ P := by
  sorry

end m_not_in_P_l1031_103180


namespace max_ab_eq_one_quarter_l1031_103154

theorem max_ab_eq_one_quarter (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab_eq_one_quarter_l1031_103154


namespace arithmetic_sequence_common_difference_l1031_103136

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 13) 
  (h2 : (5 * (a 1 + a 5)) / 2 = 35) 
  (h_arithmetic_sequence : ∀ n, a (n+1) = a n + d) : 
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l1031_103136


namespace can_form_set_l1031_103199

-- Define each group of objects based on given conditions
def famous_movie_stars : Type := sorry
def small_rivers_in_our_country : Type := sorry
def students_2012_senior_class_Panzhihua : Type := sorry
def difficult_high_school_math_problems : Type := sorry

-- Define the property of having well-defined elements
def has_definite_elements (T : Type) : Prop := sorry

-- The groups in terms of propositions
def group_A : Prop := ¬ has_definite_elements famous_movie_stars
def group_B : Prop := ¬ has_definite_elements small_rivers_in_our_country
def group_C : Prop := has_definite_elements students_2012_senior_class_Panzhihua
def group_D : Prop := ¬ has_definite_elements difficult_high_school_math_problems

-- We need to prove that group C can form a set
theorem can_form_set : group_C :=
by
  sorry

end can_form_set_l1031_103199


namespace negation_example_l1031_103123

theorem negation_example : ¬ (∀ x : ℝ, x^2 ≥ Real.log 2) ↔ ∃ x : ℝ, x^2 < Real.log 2 :=
by
  sorry

end negation_example_l1031_103123


namespace percentage_alcohol_second_vessel_l1031_103115

theorem percentage_alcohol_second_vessel :
  (∀ (x : ℝ),
    (0.25 * 3 + (x / 100) * 5 = 0.275 * 10) -> x = 40) :=
by
  intro x h
  sorry

end percentage_alcohol_second_vessel_l1031_103115


namespace sword_length_difference_l1031_103127

def christopher_sword := 15.0
def jameson_sword := 2 * christopher_sword + 3
def june_sword := jameson_sword + 5
def average_length := (christopher_sword + jameson_sword + june_sword) / 3
def laura_sword := average_length - 0.1 * average_length
def difference := june_sword - laura_sword

theorem sword_length_difference :
  difference = 12.197 := 
sorry

end sword_length_difference_l1031_103127


namespace trig_identity_l1031_103122

theorem trig_identity :
  (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trig_identity_l1031_103122


namespace doughnut_machine_completion_time_l1031_103193

noncomputable def start_time : ℕ := 8 * 60 + 30  -- 8:30 AM in minutes
noncomputable def one_third_time : ℕ := 11 * 60 + 10  -- 11:10 AM in minutes
noncomputable def total_time_minutes : ℕ := 8 * 60  -- 8 hours in minutes
noncomputable def expected_completion_time : ℕ := 16 * 60 + 30  -- 4:30 PM in minutes

theorem doughnut_machine_completion_time :
  one_third_time - start_time = total_time_minutes / 3 →
  start_time + total_time_minutes = expected_completion_time :=
by
  intros h1
  sorry

end doughnut_machine_completion_time_l1031_103193


namespace larger_number_of_product_56_and_sum_15_l1031_103194

theorem larger_number_of_product_56_and_sum_15 (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 := 
by
  sorry

end larger_number_of_product_56_and_sum_15_l1031_103194


namespace airplane_cost_correct_l1031_103166

-- Define the conditions
def initial_amount : ℝ := 5.00
def change_received : ℝ := 0.72

-- Define the cost calculation
def airplane_cost (initial : ℝ) (change : ℝ) : ℝ := initial - change

-- Prove that the airplane cost is $4.28 given the conditions
theorem airplane_cost_correct : airplane_cost initial_amount change_received = 4.28 :=
by
  -- The actual proof goes here
  sorry

end airplane_cost_correct_l1031_103166


namespace find_n_from_binomial_expansion_l1031_103164

theorem find_n_from_binomial_expansion (x a : ℝ) (n : ℕ)
  (h4 : (Nat.choose n 3) * x^(n - 3) * a^3 = 210)
  (h5 : (Nat.choose n 4) * x^(n - 4) * a^4 = 420)
  (h6 : (Nat.choose n 5) * x^(n - 5) * a^5 = 630) :
  n = 19 :=
sorry

end find_n_from_binomial_expansion_l1031_103164


namespace yuri_total_puppies_l1031_103158

-- Conditions
def first_week_puppies := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies := first_week_puppies + 10

-- Total puppies
def total_puppies : ℕ := first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies

-- Theorem to prove
theorem yuri_total_puppies : total_puppies = 74 :=
by sorry

end yuri_total_puppies_l1031_103158


namespace determine_digit_X_l1031_103174

theorem determine_digit_X (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (h : 510 / X = 10 * 4 + X + 2 * X) : X = 8 :=
sorry

end determine_digit_X_l1031_103174


namespace geometric_seq_value_l1031_103191

theorem geometric_seq_value (a : ℕ → ℝ) (h : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geometric_seq_value_l1031_103191


namespace correct_calculation_l1031_103172

-- Define the conditions of the problem
variable (x : ℕ)
variable (h : x + 5 = 43)

-- The theorem we want to prove
theorem correct_calculation : 5 * x = 190 :=
by
  -- Since Lean requires a proof and we're skipping it, we use 'sorry'
  sorry

end correct_calculation_l1031_103172


namespace no_such_sequence_exists_l1031_103107

theorem no_such_sequence_exists (a : ℕ → ℝ) :
  (∀ i, 1 ≤ i ∧ i ≤ 13 → a i + a (i + 1) + a (i + 2) > 0) →
  (∀ i, 1 ≤ i ∧ i ≤ 12 → a i + a (i + 1) + a (i + 2) + a (i + 3) < 0) →
  False :=
by
  sorry

end no_such_sequence_exists_l1031_103107


namespace fg_diff_zero_l1031_103170

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 4 * x + 3

theorem fg_diff_zero (x : ℝ) : f (g x) - g (f x) = 0 :=
by
  sorry

end fg_diff_zero_l1031_103170


namespace simplify_expression_l1031_103177

theorem simplify_expression :
  (Real.sqrt (8^(1/3)) + Real.sqrt (17/4))^2 = (33 + 8 * Real.sqrt 17) / 4 :=
by
  sorry

end simplify_expression_l1031_103177


namespace possible_values_of_C_l1031_103188

variable {α : Type} [LinearOrderedField α]

-- Definitions of points A, B and C
def pointA (a : α) := a
def pointB (b : α) := b
def pointC (c : α) := c

-- Given condition
def given_condition (a b : α) : Prop := (a + 3) ^ 2 + |b - 1| = 0

-- Function to determine if the folding condition is met
def folding_number_line (A B C : α) : Prop :=
  (C = 2 * A - B ∨ C = 2 * B - A ∨ (A + B) / 2 = C)

-- Theorem to prove the possible values of C
theorem possible_values_of_C (a b : α) (h : given_condition a b) :
  ∃ C : α, folding_number_line (pointA a) (pointB b) (pointC C) ∧ (C = -7 ∨ C = 5 ∨ C = -1) :=
sorry

end possible_values_of_C_l1031_103188


namespace least_positive_integer_solution_l1031_103197

theorem least_positive_integer_solution : 
  ∃ x : ℕ, x + 3567 ≡ 1543 [MOD 14] ∧ x = 6 := 
by
  -- proof goes here
  sorry

end least_positive_integer_solution_l1031_103197


namespace ratio_a_c_l1031_103183

theorem ratio_a_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 1) 
  (h3 : d / b = 2 / 5) : 
  a / c = 25 / 32 := 
by sorry

end ratio_a_c_l1031_103183


namespace no_solution_iff_m_range_l1031_103185

theorem no_solution_iff_m_range (m : ℝ) : 
  ¬ ∃ x : ℝ, |x-1| + |x-m| < 2*m ↔ (0 < m ∧ m < 1/3) := sorry

end no_solution_iff_m_range_l1031_103185


namespace evaluate_g_at_neg2_l1031_103159

def g (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem evaluate_g_at_neg2 : g (-2) = 11 := by
  sorry

end evaluate_g_at_neg2_l1031_103159


namespace steve_height_after_growth_l1031_103126

/-- 
  Steve's height after growing 6 inches, given that he was initially 5 feet 6 inches tall.
-/
def steve_initial_height_feet : ℕ := 5
def steve_initial_height_inches : ℕ := 6
def inches_per_foot : ℕ := 12
def added_growth : ℕ := 6

theorem steve_height_after_growth (steve_initial_height_feet : ℕ) 
                                  (steve_initial_height_inches : ℕ) 
                                  (inches_per_foot : ℕ) 
                                  (added_growth : ℕ) : 
  steve_initial_height_feet * inches_per_foot + steve_initial_height_inches + added_growth = 72 :=
by
  sorry

end steve_height_after_growth_l1031_103126


namespace molecular_weight_of_4_moles_AlCl3_is_correct_l1031_103176

/-- The atomic weight of aluminum (Al) is 26.98 g/mol. -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of chlorine (Cl) is 35.45 g/mol. -/
def atomic_weight_Cl : ℝ := 35.45

/-- A molecule of AlCl3 consists of 1 atom of Al and 3 atoms of Cl. -/
def molecular_weight_AlCl3 := (1 * atomic_weight_Al) + (3 * atomic_weight_Cl)

/-- The total weight of 4 moles of AlCl3. -/
def total_weight_4_moles_AlCl3 := 4 * molecular_weight_AlCl3

/-- We prove that the total weight of 4 moles of AlCl3 is 533.32 g. -/
theorem molecular_weight_of_4_moles_AlCl3_is_correct :
  total_weight_4_moles_AlCl3 = 533.32 :=
sorry

end molecular_weight_of_4_moles_AlCl3_is_correct_l1031_103176


namespace initial_number_of_professors_l1031_103117

theorem initial_number_of_professors (p : ℕ) :
  (∃ p, (6480 / p : ℚ) < (11200 / (p + 3) : ℚ) ∧ 
   6480 % p = 0 ∧ 11200 % (p + 3) = 0 ∧ p > 4) → 
  p = 5 := 
sorry

end initial_number_of_professors_l1031_103117


namespace area_of_lawn_l1031_103142

theorem area_of_lawn 
  (park_length : ℝ) (park_width : ℝ) (road_width : ℝ) 
  (H1 : park_length = 60) (H2 : park_width = 40) (H3 : road_width = 3) : 
  (park_length * park_width - (park_length * road_width + park_width * road_width - road_width ^ 2)) = 2109 := 
by
  sorry

end area_of_lawn_l1031_103142


namespace translated_coordinates_of_B_l1031_103139

-- Define the initial coordinates of points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the translated coordinates of point A
def A' : ℝ × ℝ := (4, 0)

-- Define the expected coordinates of point B' after the translation
def B' : ℝ × ℝ := (1, -1)

-- Proof statement
theorem translated_coordinates_of_B (A A' B : ℝ × ℝ) (B' : ℝ × ℝ) :
  A = (1, 1) ∧ A' = (4, 0) ∧ B = (-2, 0) → B' = (1, -1) :=
by
  intros h
  sorry

end translated_coordinates_of_B_l1031_103139


namespace geometric_sequence_problem_l1031_103182

theorem geometric_sequence_problem
  (q : ℝ) (h_q : |q| ≠ 1) (m : ℕ)
  (a : ℕ → ℝ)
  (h_a1 : a 1 = -1)
  (h_am : a m = a 1 * a 2 * a 3 * a 4 * a 5) 
  (h_gseq : ∀ n, a (n + 1) = a n * q) :
  m = 11 :=
by
  sorry

end geometric_sequence_problem_l1031_103182


namespace tom_trip_cost_l1031_103187

-- Definitions of hourly rates
def rate_6AM_to_10AM := 10
def rate_10AM_to_2PM := 12
def rate_2PM_to_6PM := 15
def rate_6PM_to_10PM := 20

-- Definitions of trip start times and durations
def first_trip_start := 8
def second_trip_start := 14
def third_trip_start := 20

-- Function to calculate the cost for each trip segment
def cost (start_hour : Nat) (duration : Nat) : Nat :=
  if start_hour >= 6 ∧ start_hour < 10 then duration * rate_6AM_to_10AM
  else if start_hour >= 10 ∧ start_hour < 14 then duration * rate_10AM_to_2PM
  else if start_hour >= 14 ∧ start_hour < 18 then duration * rate_2PM_to_6PM
  else if start_hour >= 18 ∧ start_hour < 22 then duration * rate_6PM_to_10PM
  else 0

-- Function to calculate the total trip cost
def total_cost : Nat :=
  cost first_trip_start 2 + cost (first_trip_start + 2) 2 +
  cost second_trip_start 4 +
  cost third_trip_start 4

-- Proof statement
theorem tom_trip_cost : total_cost = 184 := by
  -- The detailed steps of the proof would go here. Replaced with 'sorry' presently to indicate incomplete proof.
  sorry

end tom_trip_cost_l1031_103187


namespace range_of_m_l1031_103128

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 4| + |x + 8| ≥ m) → m ≤ 4 :=
by
  sorry

end range_of_m_l1031_103128


namespace solve_first_equation_solve_second_equation_l1031_103186

theorem solve_first_equation (x : ℤ) : 4 * x + 3 = 5 * x - 1 → x = 4 :=
by
  intros h
  sorry

theorem solve_second_equation (x : ℤ) : 4 * (x - 1) = 1 - x → x = 1 :=
by
  intros h
  sorry

end solve_first_equation_solve_second_equation_l1031_103186


namespace minimum_degree_g_l1031_103184

open Polynomial

theorem minimum_degree_g (f g h : Polynomial ℝ) 
  (h_eq : 5 • f + 2 • g = h)
  (deg_f : f.degree = 11)
  (deg_h : h.degree = 12) : 
  ∃ d : ℕ, g.degree = d ∧ d >= 12 := 
sorry

end minimum_degree_g_l1031_103184


namespace conic_section_is_ellipse_l1031_103134

theorem conic_section_is_ellipse (x y : ℝ) : 
  (x - 3)^2 + 9 * (y + 2)^2 = 144 →
  (∃ h k a b : ℝ, a = 12 ∧ b = 4 ∧ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) :=
by
  intro h_eq
  use 3, -2, 12, 4
  constructor
  { sorry }
  constructor
  { sorry }
  sorry

end conic_section_is_ellipse_l1031_103134


namespace probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l1031_103103

noncomputable def probability_correct_answers : ℚ :=
  let pA := (1/5 : ℚ)
  let pB := (3/5 : ℚ)
  let pC := (1/5 : ℚ)
  ((pA * (3/9 : ℚ) * (2/3)^2 * (1/3)) + (pB * (6/9 : ℚ) * (2/3) * (1/3)^2) + (pC * (1/9 : ℚ) * (1/3)^3))

theorem probability_of_3_correct_answers_is_31_over_135 :
  probability_correct_answers = 31 / 135 := by
  sorry

noncomputable def expected_score : ℚ :=
  let E_m := (1/5 * 1 + 3/5 * 2 + 1/5 * 3 : ℚ)
  let E_n := (3 * (2/3 : ℚ))
  (15 * E_m + 10 * E_n)

theorem expected_value_of_total_score_is_50 :
  expected_score = 50 := by
  sorry

end probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l1031_103103


namespace range_of_a_l1031_103162

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 4 ≤ a := 
sorry

end range_of_a_l1031_103162


namespace wash_cycle_time_l1031_103148

-- Definitions for the conditions
def num_loads : Nat := 8
def dry_cycle_time_minutes : Nat := 60
def total_time_hours : Nat := 14
def total_time_minutes : Nat := total_time_hours * 60

-- The actual statement we need to prove
theorem wash_cycle_time (x : Nat) (h : num_loads * x + num_loads * dry_cycle_time_minutes = total_time_minutes) : x = 45 :=
by
  sorry

end wash_cycle_time_l1031_103148


namespace sin_triple_alpha_minus_beta_l1031_103138

open Real 

theorem sin_triple_alpha_minus_beta (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : π / 2 < β ∧ β < π)
  (h1 : cos (α - β) = 1 / 2)
  (h2 : sin (α + β) = 1 / 2) :
  sin (3 * α - β) = 1 / 2 :=
by
  sorry

end sin_triple_alpha_minus_beta_l1031_103138


namespace lcm_of_two_numbers_l1031_103152
-- Importing the math library

-- Define constants and variables
variables (A B LCM HCF : ℕ)

-- Given conditions
def product_condition : Prop := A * B = 17820
def hcf_condition : Prop := HCF = 12
def lcm_condition : Prop := LCM = Nat.lcm A B

-- Theorem to prove
theorem lcm_of_two_numbers : product_condition A B ∧ hcf_condition HCF →
                              lcm_condition A B LCM →
                              LCM = 1485 := 
by
  sorry

end lcm_of_two_numbers_l1031_103152


namespace find_a_b_l1031_103111

theorem find_a_b (a b : ℝ) (h1 : b - a = -7) (h2 : 64 * (a + b) = 20736) :
  a = 165.5 ∧ b = 158.5 :=
by
  sorry

end find_a_b_l1031_103111


namespace probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l1031_103125

theorem probability_one_piece_is_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) : 
  (if (piece_lengths.1 = 2 ∧ piece_lengths.2 ≠ 2) ∨ (piece_lengths.1 ≠ 2 ∧ piece_lengths.2 = 2) then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 2 / 5 :=
sorry

theorem probability_both_pieces_longer_than_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) :
  (if piece_lengths.1 > 2 ∧ piece_lengths.2 > 2 then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 1 / 3 :=
sorry

end probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l1031_103125


namespace debt_payments_l1031_103179

noncomputable def average_payment (total_amount : ℕ) (payments : ℕ) : ℕ := total_amount / payments

theorem debt_payments (x : ℕ) :
  8 * x + 44 * (x + 65) = 52 * 465 → x = 410 :=
by
  intros h
  sorry

end debt_payments_l1031_103179


namespace sqrt_of_4_l1031_103189

theorem sqrt_of_4 (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_of_4_l1031_103189


namespace min_value_my_function_l1031_103120

noncomputable def my_function (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x - 2) + 3 * abs (x - 3) + 4 * abs (x - 4)

theorem min_value_my_function :
  ∃ (x : ℝ), my_function x = 8 ∧ (∀ y : ℝ, my_function y ≥ 8) :=
sorry

end min_value_my_function_l1031_103120


namespace trajectory_passes_quadrants_l1031_103144

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 4

-- Define the condition for a point to belong to the first quadrant
def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Define the condition for a point to belong to the second quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- State the theorem that the trajectory of point P passes through the first and second quadrants
theorem trajectory_passes_quadrants :
  (∃ x y : ℝ, circle_equation x y ∧ in_first_quadrant x y) ∧
  (∃ x y : ℝ, circle_equation x y ∧ in_second_quadrant x y) :=
sorry

end trajectory_passes_quadrants_l1031_103144


namespace additional_people_needed_l1031_103173

theorem additional_people_needed (k m : ℕ) (h1 : 8 * 3 = k) (h2 : m * 2 = k) : (m - 8) = 4 :=
by
  sorry

end additional_people_needed_l1031_103173


namespace inequality_holds_for_all_x_l1031_103145

theorem inequality_holds_for_all_x : 
  ∀ (a : ℝ), (∀ (x : ℝ), |x| ≤ 1 → x^2 - (a + 1) * x + a + 1 > 0) ↔ a < -1 := 
sorry

end inequality_holds_for_all_x_l1031_103145


namespace range_a_l1031_103149

def f (x a : ℝ) := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

theorem range_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l1031_103149


namespace find_y_l1031_103151

theorem find_y (x y : ℤ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 :=
by
  subst h1
  have h : 3 * 4 + 2 * y = 30 := by rw [h2]
  linarith

end find_y_l1031_103151


namespace cosine_seventh_power_expansion_l1031_103169

theorem cosine_seventh_power_expansion :
  let b1 := (35 : ℝ) / 64
  let b2 := (0 : ℝ)
  let b3 := (21 : ℝ) / 64
  let b4 := (0 : ℝ)
  let b5 := (7 : ℝ) / 64
  let b6 := (0 : ℝ)
  let b7 := (1 : ℝ) / 64
  b1^2 + b2^2 + b3^2 + b4^2 + b5^2 + b6^2 + b7^2 = 1687 / 4096 := by
  sorry

end cosine_seventh_power_expansion_l1031_103169


namespace min_value_of_f_in_interval_l1031_103113

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) ∧ f x = -Real.sqrt 2 / 2 :=
by
  sorry

end min_value_of_f_in_interval_l1031_103113


namespace ratio_of_areas_l1031_103175

theorem ratio_of_areas (r : ℝ) (h : r > 0) :
  let R1 := r
  let R2 := 3 * r
  let S1 := 6 * R1
  let S2 := 6 * r
  let area_smaller_circle := π * R2 ^ 2
  let area_larger_square := S2 ^ 2
  (area_smaller_circle / area_larger_square) = π / 4 :=
by
  sorry

end ratio_of_areas_l1031_103175


namespace apprentice_daily_output_l1031_103190

namespace Production

variables (x y : ℝ)

theorem apprentice_daily_output
  (h1 : 4 * x + 7 * y = 765)
  (h2 : 6 * x + 2 * y = 765) :
  y = 45 :=
sorry

end Production

end apprentice_daily_output_l1031_103190


namespace num_true_propositions_l1031_103118

theorem num_true_propositions (x : ℝ) :
  (∀ x, x > -3 → x > -6) ∧
  (∀ x, x > -6 → x > -3 = false) ∧
  (∀ x, x ≤ -3 → x ≤ -6 = false) ∧
  (∀ x, x ≤ -6 → x ≤ -3) →
  2 = 2 :=
by
  sorry

end num_true_propositions_l1031_103118


namespace nathan_weeks_l1031_103156

-- Define the conditions as per the problem
def hours_per_day_nathan : ℕ := 3
def days_per_week : ℕ := 7
def hours_per_week_nathan : ℕ := hours_per_day_nathan * days_per_week
def hours_per_day_tobias : ℕ := 5
def hours_one_week_tobias : ℕ := hours_per_day_tobias * days_per_week
def total_hours : ℕ := 77

-- The number of weeks Nathan played
def weeks_nathan (w : ℕ) : Prop :=
  hours_per_week_nathan * w + hours_one_week_tobias = total_hours

-- Prove the number of weeks Nathan played is 2
theorem nathan_weeks : ∃ w : ℕ, weeks_nathan w ∧ w = 2 :=
by
  use 2
  sorry

end nathan_weeks_l1031_103156
