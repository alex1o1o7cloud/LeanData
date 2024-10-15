import Mathlib

namespace NUMINAMATH_GPT_diff_of_roots_l2244_224448

-- Define the quadratic equation and its coefficients
def quadratic_eq (z : ℝ) : ℝ := 2 * z^2 + 5 * z - 12

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the roots of the quadratic equation using the quadratic formula
noncomputable def larger_root (a b c : ℝ) : ℝ := (-b + Real.sqrt (discriminant a b c)) / (2 * a)
noncomputable def smaller_root (a b c : ℝ) : ℝ := (-b - Real.sqrt (discriminant a b c)) / (2 * a)

-- Define the proof statement
theorem diff_of_roots : 
  ∃ (a b c z1 z2 : ℝ), 
    a = 2 ∧ b = 5 ∧ c = -12 ∧
    quadratic_eq z1 = 0 ∧ quadratic_eq z2 = 0 ∧
    z1 = smaller_root a b c ∧ z2 = larger_root a b c ∧
    z2 - z1 = 5.5 := 
by 
  sorry

end NUMINAMATH_GPT_diff_of_roots_l2244_224448


namespace NUMINAMATH_GPT_find_p_q_of_divisibility_l2244_224415

theorem find_p_q_of_divisibility 
  (p q : ℤ) 
  (h1 : (x + 3) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  (h2 : (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  : p = -31 ∧ q = -71 :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_of_divisibility_l2244_224415


namespace NUMINAMATH_GPT_overall_average_score_l2244_224422

-- Definitions used from conditions
def male_students : Nat := 8
def male_avg_score : Real := 83
def female_students : Nat := 28
def female_avg_score : Real := 92

-- Theorem to prove the overall average score is 90
theorem overall_average_score : 
  (male_students * male_avg_score + female_students * female_avg_score) / (male_students + female_students) = 90 := 
by 
  sorry

end NUMINAMATH_GPT_overall_average_score_l2244_224422


namespace NUMINAMATH_GPT_tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l2244_224486

variable {α : Real}

theorem tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5 (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_2_implies_sin_2alpha_eq_4_over_5_l2244_224486


namespace NUMINAMATH_GPT_max_value_ineq_l2244_224431

theorem max_value_ineq (x y : ℝ) (hx1 : -5 ≤ x) (hx2 : x ≤ -3) (hy1 : 1 ≤ y) (hy2 : y ≤ 3) : 
  (x + y) / (x - 1) ≤ 2 / 3 := 
sorry

end NUMINAMATH_GPT_max_value_ineq_l2244_224431


namespace NUMINAMATH_GPT_extra_money_from_customer_l2244_224483

theorem extra_money_from_customer
  (price_per_craft : ℕ)
  (num_crafts_sold : ℕ)
  (deposit_amount : ℕ)
  (remaining_amount : ℕ)
  (total_amount_before_deposit : ℕ)
  (amount_made_from_crafts : ℕ)
  (extra_money : ℕ) :
  price_per_craft = 12 →
  num_crafts_sold = 3 →
  deposit_amount = 18 →
  remaining_amount = 25 →
  total_amount_before_deposit = deposit_amount + remaining_amount →
  amount_made_from_crafts = price_per_craft * num_crafts_sold →
  extra_money = total_amount_before_deposit - amount_made_from_crafts →
  extra_money = 7 :=
by
  intros; sorry

end NUMINAMATH_GPT_extra_money_from_customer_l2244_224483


namespace NUMINAMATH_GPT_launch_country_is_soviet_union_l2244_224411

-- Definitions of conditions
def launch_date : String := "October 4, 1957"
def satellite_launched_on (date : String) : Prop := date = "October 4, 1957"
def choices : List String := ["A. United States", "B. Soviet Union", "C. European Union", "D. Germany"]

-- Problem statement
theorem launch_country_is_soviet_union : 
  satellite_launched_on launch_date → 
  "B. Soviet Union" ∈ choices := 
by
  sorry

end NUMINAMATH_GPT_launch_country_is_soviet_union_l2244_224411


namespace NUMINAMATH_GPT_max_cos_half_sin_eq_1_l2244_224447

noncomputable def max_value_expression (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 - Real.sin θ)

theorem max_cos_half_sin_eq_1 : 
  ∀ θ : ℝ, 0 < θ ∧ θ < π → max_value_expression θ ≤ 1 :=
by
  intros θ h
  sorry

end NUMINAMATH_GPT_max_cos_half_sin_eq_1_l2244_224447


namespace NUMINAMATH_GPT_quadratic_distinct_positive_roots_l2244_224443

theorem quadratic_distinct_positive_roots (a : ℝ) : 
  9 * (a - 2) > 0 → 
  a > 0 → 
  a^2 - 9 * a + 18 > 0 → 
  a ≠ 11 → 
  (2 < a ∧ a < 3) ∨ (6 < a ∧ a < 11) ∨ (11 < a) := 
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_quadratic_distinct_positive_roots_l2244_224443


namespace NUMINAMATH_GPT_other_candidate_votes_l2244_224449

-- Define the constants according to the problem
variables (X Y Z : ℝ)
axiom h1 : X = Y + (1 / 2) * Y
axiom h2 : X = 22500
axiom h3 : Y = Z - (2 / 5) * Z

-- Define the goal
theorem other_candidate_votes : Z = 25000 :=
by
  sorry

end NUMINAMATH_GPT_other_candidate_votes_l2244_224449


namespace NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_l2244_224444

theorem value_of_x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_l2244_224444


namespace NUMINAMATH_GPT_chinese_character_equation_l2244_224485

noncomputable def units_digit (n: ℕ) : ℕ :=
  n % 10

noncomputable def tens_digit (n: ℕ) : ℕ :=
  (n / 10) % 10

noncomputable def hundreds_digit (n: ℕ) : ℕ :=
  (n / 100) % 10

def Math : ℕ := 25
def LoveMath : ℕ := 125
def ILoveMath : ℕ := 3125

theorem chinese_character_equation :
  Math * LoveMath = ILoveMath :=
by
  have h_units_math := units_digit Math
  have h_units_lovemath := units_digit LoveMath
  have h_units_ilovemath := units_digit ILoveMath
  
  have h_tens_math := tens_digit Math
  have h_tens_lovemath := tens_digit LoveMath
  have h_tens_ilovemath := tens_digit ILoveMath

  have h_hundreds_lovemath := hundreds_digit LoveMath
  have h_hundreds_ilovemath := hundreds_digit ILoveMath

  -- Check conditions:
  -- h_units_* should be 0, 1, 5 or 6
  -- h_tens_math == h_tens_lovemath == h_tens_ilovemath
  -- h_hundreds_lovemath == h_hundreds_ilovemath

  sorry -- Proof would go here

end NUMINAMATH_GPT_chinese_character_equation_l2244_224485


namespace NUMINAMATH_GPT_unique_real_solution_floor_eq_l2244_224427

theorem unique_real_solution_floor_eq (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k ≤ x ∧ x < k + 1 ∧ ⌊x⌋ * (x^2 + 1) = x^3 :=
sorry

end NUMINAMATH_GPT_unique_real_solution_floor_eq_l2244_224427


namespace NUMINAMATH_GPT_birds_in_house_l2244_224464

theorem birds_in_house (B : ℕ) :
  let dogs := 3
  let cats := 18
  let humans := 7
  let total_heads := B + dogs + cats + humans
  let total_feet := 2 * B + 4 * dogs + 4 * cats + 2 * humans
  total_feet = total_heads + 74 → B = 4 :=
by
  intros dogs cats humans total_heads total_feet condition
  -- We assume the condition and work towards the proof.
  sorry

end NUMINAMATH_GPT_birds_in_house_l2244_224464


namespace NUMINAMATH_GPT_find_n_l2244_224405

theorem find_n :
  ∃ (n : ℤ), (4 ≤ n ∧ n ≤ 8) ∧ (n % 5 = 2) ∧ (n = 7) :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2244_224405


namespace NUMINAMATH_GPT_find_n_l2244_224450

def factorial : ℕ → ℕ 
| 0 => 1
| (n + 1) => (n + 1) * factorial n

theorem find_n (n : ℕ) : 3 * n * factorial n + 2 * factorial n = 40320 → n = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2244_224450


namespace NUMINAMATH_GPT_discount_percentage_l2244_224453

theorem discount_percentage 
  (evening_ticket_cost : ℝ) (food_combo_cost : ℝ) (savings : ℝ) (discounted_food_combo_cost : ℝ) (discounted_total_cost : ℝ) 
  (h1 : evening_ticket_cost = 10) 
  (h2 : food_combo_cost = 10)
  (h3 : discounted_food_combo_cost = 10 * 0.5)
  (h4 : discounted_total_cost = evening_ticket_cost + food_combo_cost - savings)
  (h5 : savings = 7)
: (1 - discounted_total_cost / (evening_ticket_cost + food_combo_cost)) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l2244_224453


namespace NUMINAMATH_GPT_north_pond_ducks_l2244_224498

-- Definitions based on the conditions
def ducks_lake_michigan : ℕ := 100
def twice_ducks_lake_michigan : ℕ := 2 * ducks_lake_michigan
def additional_ducks : ℕ := 6
def ducks_north_pond : ℕ := twice_ducks_lake_michigan + additional_ducks

-- Theorem to prove the answer
theorem north_pond_ducks : ducks_north_pond = 206 :=
by
  sorry

end NUMINAMATH_GPT_north_pond_ducks_l2244_224498


namespace NUMINAMATH_GPT_exists_station_to_complete_loop_l2244_224478

structure CircularHighway where
  fuel_at_stations : List ℝ -- List of fuel amounts at each station
  travel_cost : List ℝ -- List of travel costs between consecutive stations

def total_fuel (hw : CircularHighway) : ℝ :=
  hw.fuel_at_stations.sum

def total_travel_cost (hw : CircularHighway) : ℝ :=
  hw.travel_cost.sum

def sufficient_fuel (hw : CircularHighway) : Prop :=
  total_fuel hw ≥ 2 * total_travel_cost hw

noncomputable def can_return_to_start (hw : CircularHighway) (start_station : ℕ) : Prop :=
  -- Function that checks if starting from a specific station allows for a return
  sorry

theorem exists_station_to_complete_loop (hw : CircularHighway) (h : sufficient_fuel hw) : ∃ start_station, can_return_to_start hw start_station :=
  sorry

end NUMINAMATH_GPT_exists_station_to_complete_loop_l2244_224478


namespace NUMINAMATH_GPT_given_cond_then_geq_eight_l2244_224402

theorem given_cond_then_geq_eight (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 1) : 
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 := 
  sorry

end NUMINAMATH_GPT_given_cond_then_geq_eight_l2244_224402


namespace NUMINAMATH_GPT_first_reduction_percentage_l2244_224460

theorem first_reduction_percentage 
  (P : ℝ)  -- original price
  (x : ℝ)  -- first day reduction percentage
  (h : P > 0) -- price assumption
  (h2 : 0 ≤ x ∧ x ≤ 100) -- percentage assumption
  (cond : P * (1 - x / 100) * 0.86 = 0.774 * P) : 
  x = 10 := 
sorry

end NUMINAMATH_GPT_first_reduction_percentage_l2244_224460


namespace NUMINAMATH_GPT_cos_of_theta_cos_double_of_theta_l2244_224495

noncomputable def theta : ℝ := sorry -- Placeholder for theta within the interval (0, π/2)
axiom theta_in_range : 0 < theta ∧ theta < Real.pi / 2
axiom sin_theta_eq : Real.sin theta = 1/3

theorem cos_of_theta : Real.cos theta = 2 * Real.sqrt 2 / 3 := by
  sorry

theorem cos_double_of_theta : Real.cos (2 * theta) = 7 / 9 := by
  sorry

end NUMINAMATH_GPT_cos_of_theta_cos_double_of_theta_l2244_224495


namespace NUMINAMATH_GPT_oxen_count_l2244_224470

theorem oxen_count (B C O : ℕ) (H1 : 3 * B = 4 * C) (H2 : 3 * B = 2 * O) (H3 : 15 * B + 24 * C + O * O = 33 * B + (3 / 2) * O * B) (H4 : 24 * B = 48) (H5 : 60 * C + 30 * B + 18 * (O * (3 / 2) * B) = 108 * B + (3 / 2) * O * B * 18)
: O = 8 :=
by 
  sorry

end NUMINAMATH_GPT_oxen_count_l2244_224470


namespace NUMINAMATH_GPT_find_y_when_x_is_7_l2244_224426

theorem find_y_when_x_is_7 (x y : ℝ) (h1 : x * y = 200) (h2 : x = 7) : y = 200 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_y_when_x_is_7_l2244_224426


namespace NUMINAMATH_GPT_roots_ratio_quadratic_eq_l2244_224467

theorem roots_ratio_quadratic_eq {k r s : ℝ} 
(h_eq : ∃ a b : ℝ, a * r = b * s) 
(ratio_3_2 : ∃ t : ℝ, r = 3 * t ∧ s = 2 * t) 
(eqn : r + s = -10 ∧ r * s = k) : 
k = 24 := 
sorry

end NUMINAMATH_GPT_roots_ratio_quadratic_eq_l2244_224467


namespace NUMINAMATH_GPT_C_gets_more_than_D_by_500_l2244_224419

-- Definitions based on conditions
def proportionA := 5
def proportionB := 2
def proportionC := 4
def proportionD := 3

def totalProportion := proportionA + proportionB + proportionC + proportionD

def A_share := 2500
def totalMoney := A_share * (totalProportion / proportionA)

def C_share := (proportionC / totalProportion) * totalMoney
def D_share := (proportionD / totalProportion) * totalMoney

-- The theorem stating the final question
theorem C_gets_more_than_D_by_500 : C_share - D_share = 500 := by
  sorry

end NUMINAMATH_GPT_C_gets_more_than_D_by_500_l2244_224419


namespace NUMINAMATH_GPT_find_f_three_l2244_224461

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def f_condition (f : ℝ → ℝ) := ∀ x : ℝ, x < 0 → f x = (1/2)^x

theorem find_f_three (f : ℝ → ℝ) (h₁ : odd_function f) (h₂ : f_condition f) : f 3 = -8 :=
sorry

end NUMINAMATH_GPT_find_f_three_l2244_224461


namespace NUMINAMATH_GPT_dacid_average_marks_is_75_l2244_224410

/-- Defining the marks obtained in each subject as constants -/
def english_marks : ℕ := 76
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

/-- Total marks calculation -/
def total_marks : ℕ :=
  english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks

/-- Number of subjects -/
def number_of_subjects : ℕ := 5

/-- Average marks calculation -/
def average_marks : ℕ :=
  total_marks / number_of_subjects

/-- Theorem proving that Dacid's average marks is 75 -/
theorem dacid_average_marks_is_75 : average_marks = 75 :=
  sorry

end NUMINAMATH_GPT_dacid_average_marks_is_75_l2244_224410


namespace NUMINAMATH_GPT_linear_function_no_third_quadrant_l2244_224481

theorem linear_function_no_third_quadrant :
  ∀ x y : ℝ, (y = -5 * x + 2023) → ¬ (x < 0 ∧ y < 0) := 
by
  intros x y h
  sorry

end NUMINAMATH_GPT_linear_function_no_third_quadrant_l2244_224481


namespace NUMINAMATH_GPT_simplify_expression_l2244_224465

theorem simplify_expression :
  (2^8 + 5^5) * (2^3 - (-2)^3)^7 = 9077567990336 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2244_224465


namespace NUMINAMATH_GPT_chord_length_l2244_224490

theorem chord_length
  (x y : ℝ)
  (h_circle : (x-1)^2 + (y-2)^2 = 2)
  (h_line : 3*x - 4*y = 0) :
  ∃ L : ℝ, L = 2 :=
sorry

end NUMINAMATH_GPT_chord_length_l2244_224490


namespace NUMINAMATH_GPT_negation_of_prop_p_is_correct_l2244_224459

-- Define the original proposition p
def prop_p (x y : ℝ) : Prop := x > 0 ∧ y > 0 → x * y > 0

-- Define the negation of the proposition p
def neg_prop_p (x y : ℝ) : Prop := x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0

-- The theorem we need to prove
theorem negation_of_prop_p_is_correct : ∀ x y : ℝ, neg_prop_p x y := 
sorry

end NUMINAMATH_GPT_negation_of_prop_p_is_correct_l2244_224459


namespace NUMINAMATH_GPT_whiskers_count_l2244_224432

variable (P C S : ℕ)

theorem whiskers_count :
  P = 14 →
  C = 2 * P - 6 →
  S = P + C + 8 →
  C = 22 ∧ S = 44 :=
by
  intros hP hC hS
  rw [hP] at hC
  rw [hP, hC] at hS
  exact ⟨hC, hS⟩

end NUMINAMATH_GPT_whiskers_count_l2244_224432


namespace NUMINAMATH_GPT_binomial_expansion_a0_a1_a3_a5_l2244_224435

theorem binomial_expansion_a0_a1_a3_a5 
    (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
    (h : (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_0 + a_1 + a_3 + a_5 = 123 :=
sorry

end NUMINAMATH_GPT_binomial_expansion_a0_a1_a3_a5_l2244_224435


namespace NUMINAMATH_GPT_helen_chocolate_chip_cookies_l2244_224401

def number_of_raisin_cookies := 231
def difference := 25

theorem helen_chocolate_chip_cookies :
  ∃ C, C = number_of_raisin_cookies + difference ∧ C = 256 :=
by
  sorry -- Skipping the proof

end NUMINAMATH_GPT_helen_chocolate_chip_cookies_l2244_224401


namespace NUMINAMATH_GPT_find_a_and_b_l2244_224482

theorem find_a_and_b (a b : ℝ) 
  (h_tangent_slope : (2 * a * 2 + b = 1)) 
  (h_point_on_parabola : (a * 4 + b * 2 + 9 = -1)) : 
  a = 3 ∧ b = -11 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l2244_224482


namespace NUMINAMATH_GPT_small_possible_value_l2244_224472

theorem small_possible_value (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : 2^12 * 3^3 = a^b) : a + b = 110593 := by
  sorry

end NUMINAMATH_GPT_small_possible_value_l2244_224472


namespace NUMINAMATH_GPT_area_of_triangle_ADC_l2244_224476

-- Define the constants for the problem
variable (BD DC : ℝ)
variable (abd_area adc_area : ℝ)

-- Given conditions
axiom ratio_condition : BD / DC = 5 / 2
axiom area_abd : abd_area = 35

-- Define the theorem to be proved
theorem area_of_triangle_ADC :
  ∃ adc_area, adc_area = 14 ∧ abd_area / adc_area = BD / DC := 
sorry

end NUMINAMATH_GPT_area_of_triangle_ADC_l2244_224476


namespace NUMINAMATH_GPT_AE_length_l2244_224400

theorem AE_length :
  ∀ (A B C D E : Type) 
    (AB CD AC BD AE EC : ℕ),
  AB = 12 → CD = 15 → AC = 18 → BD = 27 → 
  (AE + EC = AC) → 
  (AE * (18 - AE)) = (4 / 9 * 18 * 8) → 
  9 * AE = 72 → 
  AE = 8 := 
by
  intros A B C D E AB CD AC BD AE EC hAB hCD hAC hBD hSum hEqual hSolve
  sorry

end NUMINAMATH_GPT_AE_length_l2244_224400


namespace NUMINAMATH_GPT_Bella_age_l2244_224429

theorem Bella_age (B : ℕ) (h₁ : ∃ n : ℕ, n = B + 9) (h₂ : B + (B + 9) = 19) : B = 5 := 
by
  sorry

end NUMINAMATH_GPT_Bella_age_l2244_224429


namespace NUMINAMATH_GPT_pier_influence_duration_l2244_224441

noncomputable def distance_affected_by_typhoon (AB AC: ℝ) : ℝ :=
  let AD := 350
  let DC := (AD ^ 2 - AC ^ 2).sqrt
  2 * DC

noncomputable def duration_under_influence (distance speed: ℝ) : ℝ :=
  distance / speed

theorem pier_influence_duration :
  let AB := 400
  let AC := AB * (1 / 2)
  let speed := 40
  duration_under_influence (distance_affected_by_typhoon AB AC) speed = 2.5 :=
by
  -- Proof would go here, but since it's omitted
  sorry

end NUMINAMATH_GPT_pier_influence_duration_l2244_224441


namespace NUMINAMATH_GPT_farm_section_areas_l2244_224446

theorem farm_section_areas (n : ℕ) (total_area : ℕ) (sections : ℕ) 
  (hn : sections = 5) (ht : total_area = 300) : total_area / sections = 60 :=
by
  sorry

end NUMINAMATH_GPT_farm_section_areas_l2244_224446


namespace NUMINAMATH_GPT_polynomial_min_value_l2244_224421

theorem polynomial_min_value (x : ℝ) : x = -3 → x^2 + 6 * x + 10 = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_min_value_l2244_224421


namespace NUMINAMATH_GPT_expression_not_equal_l2244_224463

theorem expression_not_equal :
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  e2 ≠ product :=
by
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  sorry

end NUMINAMATH_GPT_expression_not_equal_l2244_224463


namespace NUMINAMATH_GPT_distance_between_points_l2244_224451

open Real

theorem distance_between_points : 
  let p1 := (2, 2)
  let p2 := (5, 9)
  dist (p1 : ℝ × ℝ) p2 = sqrt 58 :=
by
  let p1 := (2, 2)
  let p2 := (5, 9)
  have h1 : p1.1 = 2 := rfl
  have h2 : p1.2 = 2 := rfl
  have h3 : p2.1 = 5 := rfl
  have h4 : p2.2 = 9 := rfl
  sorry

end NUMINAMATH_GPT_distance_between_points_l2244_224451


namespace NUMINAMATH_GPT_polygon_sides_eq_eleven_l2244_224493

theorem polygon_sides_eq_eleven (n : ℕ) (D : ℕ)
(h1 : D = n + 33)
(h2 : D = n * (n - 3) / 2) :
  n = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_sides_eq_eleven_l2244_224493


namespace NUMINAMATH_GPT_intersection_A_B_l2244_224491

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | x^2 - 2 * x < 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by
  -- We are going to skip the proof for now
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2244_224491


namespace NUMINAMATH_GPT_find_x_l2244_224458

theorem find_x (x : ℕ) (h : x * 6000 = 480 * 10^5) : x = 8000 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l2244_224458


namespace NUMINAMATH_GPT_value_ab_plus_a_plus_b_l2244_224479

noncomputable def polynomial : Polynomial ℝ := Polynomial.C (-1) + Polynomial.X * Polynomial.C (-1) + Polynomial.X^2 * Polynomial.C (-4) + Polynomial.X^4

theorem value_ab_plus_a_plus_b {a b : ℝ} (h : polynomial.eval a = 0 ∧ polynomial.eval b = 0) : a * b + a + b = -1 / 2 :=
sorry

end NUMINAMATH_GPT_value_ab_plus_a_plus_b_l2244_224479


namespace NUMINAMATH_GPT_angles_of_triangle_l2244_224477

theorem angles_of_triangle (a b c m_a m_b : ℝ) (h1 : m_a ≥ a) (h2 : m_b ≥ b) : 
  ∃ (α β γ : ℝ), ∀ t, 
  (t = 90) ∧ (α = 45) ∧ (β = 45) := 
sorry

end NUMINAMATH_GPT_angles_of_triangle_l2244_224477


namespace NUMINAMATH_GPT_prime_sol_is_7_l2244_224492

theorem prime_sol_is_7 (p : ℕ) (x y : ℕ) (hp : Nat.Prime p) 
  (hx : p + 1 = 2 * x^2) (hy : p^2 + 1 = 2 * y^2) : 
  p = 7 := 
  sorry

end NUMINAMATH_GPT_prime_sol_is_7_l2244_224492


namespace NUMINAMATH_GPT_even_increasing_decreasing_l2244_224455

theorem even_increasing_decreasing (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = -x^2) :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, x < 0 → f x < f (x + 1)) ∧ (∀ x : ℝ, x > 0 → f x > f (x + 1)) :=
by
  sorry

end NUMINAMATH_GPT_even_increasing_decreasing_l2244_224455


namespace NUMINAMATH_GPT_large_box_chocolate_bars_l2244_224417

theorem large_box_chocolate_bars (num_small_boxes : ℕ) (chocolates_per_box : ℕ) 
  (h1 : num_small_boxes = 18) (h2 : chocolates_per_box = 28) : 
  num_small_boxes * chocolates_per_box = 504 := by
  sorry

end NUMINAMATH_GPT_large_box_chocolate_bars_l2244_224417


namespace NUMINAMATH_GPT_canonical_equations_of_line_l2244_224420

-- Conditions: Two planes given by their equations
def plane1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z + 8 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x + 5 * y - 4 * z + 4 = 0

-- Proving the canonical form of the line
theorem canonical_equations_of_line :
  ∃ x y z, plane1 x y z ∧ plane2 x y z ↔ 
  ∃ t, x = -1 + 5 * t ∧ y = 2 / 5 + 42 * t ∧ z = 60 * t :=
sorry

end NUMINAMATH_GPT_canonical_equations_of_line_l2244_224420


namespace NUMINAMATH_GPT_linear_function_does_not_pass_through_quadrant_3_l2244_224409

theorem linear_function_does_not_pass_through_quadrant_3
  (f : ℝ → ℝ) (h : ∀ x, f x = -3 * x + 5) :
  ¬ (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ f x = y) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_does_not_pass_through_quadrant_3_l2244_224409


namespace NUMINAMATH_GPT_cone_volume_l2244_224407

theorem cone_volume (r l h V: ℝ) (h1: 15 * Real.pi = Real.pi * r^2 + Real.pi * r * l)
  (h2: 2 * Real.pi * r = (1 / 3) * Real.pi * l) :
  (V = (1 / 3) * Real.pi * r^2 * h) → h = Real.sqrt (l^2 - r^2) → l = 6 * r → r = Real.sqrt (15 / 7) → 
  V = (25 * Real.sqrt 3 / 7) * Real.pi :=
sorry

end NUMINAMATH_GPT_cone_volume_l2244_224407


namespace NUMINAMATH_GPT_poly_div_simplification_l2244_224404

-- Assume a and b are real numbers.
variables (a b : ℝ)

-- Theorem to prove the equivalence
theorem poly_div_simplification (a b : ℝ) : (4 * a^2 - b^2) / (b - 2 * a) = -2 * a - b :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_poly_div_simplification_l2244_224404


namespace NUMINAMATH_GPT_opposite_of_neg_one_fifth_l2244_224434

theorem opposite_of_neg_one_fifth : -(- (1/5)) = (1/5) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_one_fifth_l2244_224434


namespace NUMINAMATH_GPT_negation_of_prop_original_l2244_224457

-- Definitions and conditions as per the problem
def prop_original : Prop :=
  ∃ x : ℝ, x^2 + x + 1 ≤ 0

def prop_negation : Prop :=
  ∀ x : ℝ, x^2 + x + 1 > 0

-- The theorem states the mathematical equivalence
theorem negation_of_prop_original : ¬ prop_original ↔ prop_negation := 
sorry

end NUMINAMATH_GPT_negation_of_prop_original_l2244_224457


namespace NUMINAMATH_GPT_range_of_a_l2244_224423

variable (x a : ℝ)

theorem range_of_a (h1 : ∀ x, x ≤ a → x < 2) (h2 : ∀ x, x < 2) : a ≥ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2244_224423


namespace NUMINAMATH_GPT_quadratic_inequality_l2244_224474

theorem quadratic_inequality (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 < 0) → a < 1 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l2244_224474


namespace NUMINAMATH_GPT_eval_expression_l2244_224488

theorem eval_expression : (8 / 4 - 3 * 2 + 9 - 3^2) = -4 := sorry

end NUMINAMATH_GPT_eval_expression_l2244_224488


namespace NUMINAMATH_GPT_min_chord_length_intercepted_line_eq_l2244_224436

theorem min_chord_length_intercepted_line_eq (m : ℝ)
  (hC : ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 16)
  (hL : ∀ (x y : ℝ), (2*m-1)*x + (m-1)*y - 3*m + 1 = 0)
  : ∃ x y : ℝ, x - 2*y - 4 = 0 := sorry

end NUMINAMATH_GPT_min_chord_length_intercepted_line_eq_l2244_224436


namespace NUMINAMATH_GPT_color_of_last_bead_is_white_l2244_224424

-- Defining the pattern of the beads
inductive BeadColor
| White
| Black
| Red

open BeadColor

-- Define the repeating pattern of the beads
def beadPattern : ℕ → BeadColor
| 0 => White
| 1 => Black
| 2 => Black
| 3 => Red
| 4 => Red
| 5 => Red
| (n + 6) => beadPattern n

-- Define the total number of beads
def totalBeads : ℕ := 85

-- Define the position of the last bead
def lastBead : ℕ := totalBeads - 1

-- Proving the color of the last bead
theorem color_of_last_bead_is_white : beadPattern lastBead = White :=
by
  sorry

end NUMINAMATH_GPT_color_of_last_bead_is_white_l2244_224424


namespace NUMINAMATH_GPT_paco_cookies_proof_l2244_224454

-- Define the initial conditions
def initial_cookies : Nat := 40
def cookies_eaten : Nat := 2
def cookies_bought : Nat := 37
def free_cookies_per_bought : Nat := 2

-- Define the total number of cookies after all operations
def total_cookies (initial_cookies cookies_eaten cookies_bought free_cookies_per_bought : Nat) : Nat :=
  let remaining_cookies := initial_cookies - cookies_eaten
  let free_cookies := cookies_bought * free_cookies_per_bought
  let cookies_from_bakery := cookies_bought + free_cookies
  remaining_cookies + cookies_from_bakery

-- The target statement that needs to be proved
theorem paco_cookies_proof : total_cookies initial_cookies cookies_eaten cookies_bought free_cookies_per_bought = 149 :=
by
  sorry

end NUMINAMATH_GPT_paco_cookies_proof_l2244_224454


namespace NUMINAMATH_GPT_sum_of_coefficients_eq_two_l2244_224462

theorem sum_of_coefficients_eq_two {a b c : ℤ} (h : ∀ x : ℤ, x * (x + 1) = a + b * x + c * x^2) : a + b + c = 2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_eq_two_l2244_224462


namespace NUMINAMATH_GPT_more_green_than_yellow_l2244_224456

-- Define constants
def red_peaches : ℕ := 2
def yellow_peaches : ℕ := 6
def green_peaches : ℕ := 14

-- Prove the statement
theorem more_green_than_yellow : green_peaches - yellow_peaches = 8 :=
by
  sorry

end NUMINAMATH_GPT_more_green_than_yellow_l2244_224456


namespace NUMINAMATH_GPT_find_a_and_b_l2244_224433

theorem find_a_and_b (a b m : ℝ) 
  (h1 : (3 * a - 5)^(1 / 3) = -2)
  (h2 : ∀ x, x^2 = b → x = m ∨ x = 1 - 5 * m) : 
  a = -1 ∧ b = 1 / 16 :=
by
  sorry  -- proof to be constructed

end NUMINAMATH_GPT_find_a_and_b_l2244_224433


namespace NUMINAMATH_GPT_find_sum_of_xyz_l2244_224403

theorem find_sum_of_xyz (x y z : ℕ) (h1 : 0 < x ∧ 0 < y ∧ 0 < z)
  (h2 : (x + y + z)^3 - x^3 - y^3 - z^3 = 300) : x + y + z = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_xyz_l2244_224403


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2244_224480

-- Condition definitions
def a : Int := 3
def d : Int := 2
def a_n : Int := 25
def n : Int := 12

-- Sum formula for an arithmetic sequence proof
theorem arithmetic_sequence_sum :
    let n := 12
    let S_n := (n * (a + a_n)) / 2
    S_n = 168 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2244_224480


namespace NUMINAMATH_GPT_tenth_term_of_arithmetic_sequence_l2244_224473

-- Define the initial conditions: first term 'a' and the common difference 'd'
def a : ℤ := 2
def d : ℤ := 1 - a

-- Define the n-th term of an arithmetic sequence formula
def nth_term (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Statement to prove
theorem tenth_term_of_arithmetic_sequence :
  nth_term a d 10 = -7 := 
by
  sorry

end NUMINAMATH_GPT_tenth_term_of_arithmetic_sequence_l2244_224473


namespace NUMINAMATH_GPT_sequence_b_n_l2244_224466

theorem sequence_b_n (b : ℕ → ℕ) (h₀ : b 1 = 3) (h₁ : ∀ n, b (n + 1) = b n + 3 * n + 1) :
  b 50 = 3727 :=
sorry

end NUMINAMATH_GPT_sequence_b_n_l2244_224466


namespace NUMINAMATH_GPT_probability_of_s_in_statistics_l2244_224416

theorem probability_of_s_in_statistics :
  let totalLetters := 10
  let count_s := 3
  (count_s / totalLetters : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_of_s_in_statistics_l2244_224416


namespace NUMINAMATH_GPT_problem_statement_l2244_224484

variable {R : Type} [LinearOrderedField R]
variable (f : R → R)

theorem problem_statement
  (hf1 : ∀ x y : R, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x < y → f x < f y)
  (hf2 : ∀ x : R, f (x + 2) = f (- (x + 2))) :
  f (7 / 2) < f 1 ∧ f 1 < f (5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2244_224484


namespace NUMINAMATH_GPT_average_side_lengths_l2244_224471

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end NUMINAMATH_GPT_average_side_lengths_l2244_224471


namespace NUMINAMATH_GPT_polynomial_integer_values_l2244_224418

theorem polynomial_integer_values (a b c d : ℤ) (h1 : ∃ (n : ℤ), n = (a * (-1)^3 + b * (-1)^2 - c * (-1) - d))
  (h2 : ∃ (n : ℤ), n = (a * 0^3 + b * 0^2 - c * 0 - d))
  (h3 : ∃ (n : ℤ), n = (a * 1^3 + b * 1^2 - c * 1 - d))
  (h4 : ∃ (n : ℤ), n = (a * 2^3 + b * 2^2 - c * 2 - d)) :
  ∀ x : ℤ, ∃ m : ℤ, m = a * x^3 + b * x^2 - c * x - d :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_polynomial_integer_values_l2244_224418


namespace NUMINAMATH_GPT_negate_universal_to_existential_l2244_224412

variable {f : ℝ → ℝ}

theorem negate_universal_to_existential :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔
  (∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0) :=
  sorry

end NUMINAMATH_GPT_negate_universal_to_existential_l2244_224412


namespace NUMINAMATH_GPT_triangle_inequality_l2244_224406

theorem triangle_inequality
  (a b c : ℝ)
  (habc : ¬(a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a)) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := 
by {
  sorry
}

end NUMINAMATH_GPT_triangle_inequality_l2244_224406


namespace NUMINAMATH_GPT_original_height_of_ball_l2244_224425

theorem original_height_of_ball (h : ℝ) : 
  (h + 2 * (0.5 * h) + 2 * ((0.5)^2 * h) = 200) -> 
  h = 800 / 9 := 
by
  sorry

end NUMINAMATH_GPT_original_height_of_ball_l2244_224425


namespace NUMINAMATH_GPT_number_of_real_b_l2244_224437

noncomputable def count_integer_roots_of_quadratic_eq_b : ℕ :=
  let pairs := [(1, 64), (2, 32), (4, 16), (8, 8), (-1, -64), (-2, -32), (-4, -16), (-8, -8)]
  pairs.length

theorem number_of_real_b : count_integer_roots_of_quadratic_eq_b = 8 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end NUMINAMATH_GPT_number_of_real_b_l2244_224437


namespace NUMINAMATH_GPT_option_D_is_div_by_9_l2244_224468

-- Define the parameters and expressions
def A (k : ℕ) : ℤ := 6 + 6 * 7^k
def B (k : ℕ) : ℤ := 2 + 7^(k - 1)
def C (k : ℕ) : ℤ := 2 * (2 + 7^(k + 1))
def D (k : ℕ) : ℤ := 3 * (2 + 7^k)

-- Define the main theorem to prove that D is divisible by 9
theorem option_D_is_div_by_9 (k : ℕ) (hk : k > 0) : D k % 9 = 0 :=
sorry

end NUMINAMATH_GPT_option_D_is_div_by_9_l2244_224468


namespace NUMINAMATH_GPT_gcd_g50_g52_l2244_224497

def g (x : ℕ) : ℕ := x^2 - 2 * x + 2021

theorem gcd_g50_g52 : Nat.gcd (g 50) (g 52) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_g50_g52_l2244_224497


namespace NUMINAMATH_GPT_average_salary_of_officers_l2244_224440

-- Define the given conditions
def avg_salary_total := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 480

-- Define the expected result
def avg_salary_officers := 440

-- Define the problem and statement to be proved in Lean
theorem average_salary_of_officers :
  (num_officers + num_non_officers) * avg_salary_total - num_non_officers * avg_salary_non_officers = num_officers * avg_salary_officers := 
by
  sorry

end NUMINAMATH_GPT_average_salary_of_officers_l2244_224440


namespace NUMINAMATH_GPT_ab_value_l2244_224487

theorem ab_value (a b : ℤ) (h : 48 * a * b = 65 * a * b) : a * b = 0 :=
  sorry

end NUMINAMATH_GPT_ab_value_l2244_224487


namespace NUMINAMATH_GPT_solve_for_asterisk_l2244_224438

theorem solve_for_asterisk (asterisk : ℝ) : 
  ((60 / 20) * (60 / asterisk) = 1) → asterisk = 180 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_asterisk_l2244_224438


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l2244_224452

theorem angle_in_third_quadrant (k : ℤ) (α : ℝ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  180 - α > -90 - k * 360 ∧ 180 - α < -k * 360 := 
by sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l2244_224452


namespace NUMINAMATH_GPT_ship_speed_l2244_224414

theorem ship_speed 
  (D : ℝ)
  (h1 : (D/2) - 200 = D/3)
  (S := (D / 2) / 20):
  S = 30 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_ship_speed_l2244_224414


namespace NUMINAMATH_GPT_find_d_l2244_224408

-- Define AP terms as S_n = a + (n-1)d, sum of first 10 terms, and difference expression
def arithmetic_progression (S : ℕ → ℕ) (a d : ℕ) : Prop :=
  ∀ n, S n = a + (n - 1) * d

def sum_first_ten (S : ℕ → ℕ) : Prop :=
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55

def difference_expression (S : ℕ → ℕ) (d : ℕ) : Prop :=
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = d

theorem find_d : ∃ (d : ℕ) (S : ℕ → ℕ) (a : ℕ), 
  (∀ n, S n = a + (n - 1) * d) ∧ 
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55 ∧
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = 16 :=
by
  sorry  -- proof is not required

end NUMINAMATH_GPT_find_d_l2244_224408


namespace NUMINAMATH_GPT_max_second_smallest_l2244_224413

noncomputable def f (M : ℕ) : ℕ :=
  (M - 1) * (90 - M) * (89 - M) * (88 - M)

theorem max_second_smallest (M : ℕ) (cond : 1 ≤ M ∧ M ≤ 89) : M = 23 ↔ (∀ N : ℕ, f M ≥ f N) :=
by
  sorry

end NUMINAMATH_GPT_max_second_smallest_l2244_224413


namespace NUMINAMATH_GPT_remainder_correct_l2244_224469

def dividend : ℝ := 13787
def divisor : ℝ := 154.75280898876406
def quotient : ℝ := 89
def remainder : ℝ := dividend - (divisor * quotient)

theorem remainder_correct: remainder = 14 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_correct_l2244_224469


namespace NUMINAMATH_GPT_medial_triangle_AB_AC_BC_l2244_224494

theorem medial_triangle_AB_AC_BC
  (l m n : ℝ)
  (A B C : Type)
  (midpoint_BC := (l, 0, 0))
  (midpoint_AC := (0, m, 0))
  (midpoint_AB := (0, 0, n)) :
  (AB^2 + AC^2 + BC^2) / (l^2 + m^2 + n^2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_medial_triangle_AB_AC_BC_l2244_224494


namespace NUMINAMATH_GPT_maximum_dn_l2244_224489

-- Definitions of a_n and d_n based on the problem statement
def a (n : ℕ) : ℕ := 150 + (n + 1)^2
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Statement of the theorem
theorem maximum_dn : ∃ M, M = 2 ∧ ∀ n, d n ≤ M :=
by
  -- proof should be written here
  sorry

end NUMINAMATH_GPT_maximum_dn_l2244_224489


namespace NUMINAMATH_GPT_r_squared_plus_s_squared_l2244_224439

theorem r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_r_squared_plus_s_squared_l2244_224439


namespace NUMINAMATH_GPT_parabola_focus_distance_l2244_224430

theorem parabola_focus_distance (p m : ℝ) (h1 : p > 0) (h2 : (2 - (-p/2)) = 4) : p = 4 := 
by
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l2244_224430


namespace NUMINAMATH_GPT_sum_of_inversion_counts_of_all_permutations_l2244_224475

noncomputable def sum_of_inversion_counts (n : ℕ) (fixed_val : ℕ) (fixed_pos : ℕ) : ℕ :=
  if n = 6 ∧ fixed_val = 4 ∧ fixed_pos = 3 then 120 else 0

theorem sum_of_inversion_counts_of_all_permutations :
  sum_of_inversion_counts 6 4 3 = 120 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_inversion_counts_of_all_permutations_l2244_224475


namespace NUMINAMATH_GPT_tree_planting_problem_l2244_224499

noncomputable def total_trees_needed (length width tree_distance : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let intervals := perimeter / tree_distance
  intervals

theorem tree_planting_problem : total_trees_needed 150 60 10 = 42 :=
by
  sorry

end NUMINAMATH_GPT_tree_planting_problem_l2244_224499


namespace NUMINAMATH_GPT_prime_triplets_l2244_224428

theorem prime_triplets (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p ^ q + q ^ p = r ↔ (p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17) := by
  sorry

end NUMINAMATH_GPT_prime_triplets_l2244_224428


namespace NUMINAMATH_GPT_area_of_sector_l2244_224442

theorem area_of_sector (L θ : ℝ) (hL : L = 4) (hθ : θ = 2) : 
  (1 / 2) * ((L / θ) ^ 2) * θ = 4 := by
  sorry

end NUMINAMATH_GPT_area_of_sector_l2244_224442


namespace NUMINAMATH_GPT_eval_derivative_at_one_and_neg_one_l2244_224445

def f (x : ℝ) : ℝ := x^4 + x - 1

theorem eval_derivative_at_one_and_neg_one : 
  (deriv f 1) + (deriv f (-1)) = 2 :=
by 
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_eval_derivative_at_one_and_neg_one_l2244_224445


namespace NUMINAMATH_GPT_ad_minus_bc_divisible_by_2017_l2244_224496

theorem ad_minus_bc_divisible_by_2017 
  (a b c d n : ℕ) 
  (h1 : (a * n + b) % 2017 = 0) 
  (h2 : (c * n + d) % 2017 = 0) : 
  (a * d - b * c) % 2017 = 0 :=
sorry

end NUMINAMATH_GPT_ad_minus_bc_divisible_by_2017_l2244_224496
