import Mathlib

namespace cos2theta_sin2theta_l2154_215494

theorem cos2theta_sin2theta (θ : ℝ) (h : 2 * Real.cos θ + Real.sin θ = 0) :
  Real.cos (2 * θ) + (1 / 2) * Real.sin (2 * θ) = -1 :=
sorry

end cos2theta_sin2theta_l2154_215494


namespace one_over_a_plus_one_over_b_eq_neg_one_l2154_215403

theorem one_over_a_plus_one_over_b_eq_neg_one
  (a b : ℝ) (h_distinct : a ≠ b)
  (h_eq : a / b + a = b / a + b) :
  1 / a + 1 / b = -1 :=
by
  sorry

end one_over_a_plus_one_over_b_eq_neg_one_l2154_215403


namespace star_five_three_l2154_215431

def star (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem star_five_three : star 5 3 = 14 := by
  sorry

end star_five_three_l2154_215431


namespace sum_of_functions_positive_l2154_215454

open Real

noncomputable def f (x : ℝ) : ℝ := (exp x - exp (-x)) / 2

theorem sum_of_functions_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 > 0) (h2 : x2 + x3 > 0) (h3 : x3 + x1 > 0) : f x1 + f x2 + f x3 > 0 := by
  sorry

end sum_of_functions_positive_l2154_215454


namespace distinct_integers_sum_l2154_215418

theorem distinct_integers_sum {p q r s t : ℤ} 
    (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t) 
    (h9 : r ≠ s) (h10 : r ≠ t) (h11 : s ≠ t) : 
  p + q + r + s + t = 35 := 
sorry

end distinct_integers_sum_l2154_215418


namespace profit_is_1500_l2154_215447

def cost_per_charm : ℕ := 15
def charms_per_necklace : ℕ := 10
def sell_price_per_necklace : ℕ := 200
def necklaces_sold : ℕ := 30

def cost_per_necklace : ℕ := cost_per_charm * charms_per_necklace
def profit_per_necklace : ℕ := sell_price_per_necklace - cost_per_necklace
def total_profit : ℕ := profit_per_necklace * necklaces_sold

theorem profit_is_1500 : total_profit = 1500 :=
by
  sorry

end profit_is_1500_l2154_215447


namespace f_is_odd_f_is_monotone_l2154_215400

noncomputable def f (k x : ℝ) : ℝ := x + k / x

-- Proving f(x) is odd
theorem f_is_odd (k : ℝ) (hk : k ≠ 0) : ∀ x : ℝ, f k (-x) = -f k x :=
by
  intro x
  sorry

-- Proving f(x) is monotonically increasing on [sqrt(k), +∞) for k > 0
theorem f_is_monotone (k : ℝ) (hk : k > 0) : ∀ x1 x2 : ℝ, 
  x1 ∈ Set.Ici (Real.sqrt k) → x2 ∈ Set.Ici (Real.sqrt k) → x1 < x2 → f k x1 < f k x2 :=
by
  intro x1 x2 hx1 hx2 hlt
  sorry

end f_is_odd_f_is_monotone_l2154_215400


namespace cube_volumes_total_l2154_215476

theorem cube_volumes_total :
  let v1 := 5^3
  let v2 := 6^3
  let v3 := 7^3
  v1 + v2 + v3 = 684 := by
  -- Here will be the proof using Lean's tactics
  sorry

end cube_volumes_total_l2154_215476


namespace sum_of_squares_ways_l2154_215498

theorem sum_of_squares_ways : 
  ∃ ways : ℕ, ways = 2 ∧
    (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = 100) ∧ 
    (∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x^2 + y^2 + z^2 + w^2 = 100) := 
sorry

end sum_of_squares_ways_l2154_215498


namespace isosceles_triangle_perimeter_l2154_215465

-- Definitions of the side lengths
def side1 : ℝ := 8
def side2 : ℝ := 4

-- Theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter (side1 side2 : ℝ) (h1 : side1 = 8 ∨ side2 = 8) (h2 : side1 = 4 ∨ side2 = 4) : ∃ p : ℝ, p = 20 :=
by
  -- We omit the proof using sorry
  sorry

end isosceles_triangle_perimeter_l2154_215465


namespace unit_digit_of_15_pow_l2154_215485

-- Define the conditions
def base_number : ℕ := 15
def base_unit_digit : ℕ := 5

-- State the question and objective in Lean 4
theorem unit_digit_of_15_pow (X : ℕ) (h : 0 < X) : (15^X) % 10 = 5 :=
sorry

end unit_digit_of_15_pow_l2154_215485


namespace total_machine_operation_time_l2154_215442

theorem total_machine_operation_time 
  (num_dolls : ℕ) 
  (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) 
  (time_per_doll time_per_accessory : ℕ)
  (num_shoes num_bags num_cosmetics num_hats num_accessories : ℕ) 
  (total_doll_time total_accessory_time total_time : ℕ) :
  num_dolls = 12000 →
  shoes_per_doll = 2 →
  bags_per_doll = 3 →
  cosmetics_per_doll = 1 →
  hats_per_doll = 5 →
  time_per_doll = 45 →
  time_per_accessory = 10 →
  num_shoes = num_dolls * shoes_per_doll →
  num_bags = num_dolls * bags_per_doll →
  num_cosmetics = num_dolls * cosmetics_per_doll →
  num_hats = num_dolls * hats_per_doll →
  num_accessories = num_shoes + num_bags + num_cosmetics + num_hats →
  total_doll_time = num_dolls * time_per_doll →
  total_accessory_time = num_accessories * time_per_accessory →
  total_time = total_doll_time + total_accessory_time →
  total_time = 1860000 := 
sorry

end total_machine_operation_time_l2154_215442


namespace river_depth_conditions_l2154_215495

noncomputable def depth_beginning_may : ℝ := 15
noncomputable def depth_increase_june : ℝ := 11.25

theorem river_depth_conditions (d k : ℝ)
  (h1 : ∃ d, d = depth_beginning_may) 
  (h2 : 1.5 * d + k = 45)
  (h3 : k = 0.75 * d) :
  d = depth_beginning_may ∧ k = depth_increase_june :=
by
  have H : d = 15 := sorry
  have K : k = 11.25 := sorry
  exact ⟨H, K⟩

end river_depth_conditions_l2154_215495


namespace binary_sum_l2154_215470

-- Define the binary representations in terms of their base 10 equivalent.
def binary_111111111 := 511
def binary_1111111 := 127

-- State the proof problem.
theorem binary_sum : binary_111111111 + binary_1111111 = 638 :=
by {
  -- placeholder for proof
  sorry
}

end binary_sum_l2154_215470


namespace range_of_a_l2154_215408

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l2154_215408


namespace cats_after_purchasing_l2154_215453

/-- Mrs. Sheridan's total number of cats after purchasing more -/
theorem cats_after_purchasing (a b : ℕ) (h₀ : a = 11) (h₁ : b = 43) : a + b = 54 := by
  sorry

end cats_after_purchasing_l2154_215453


namespace arithmetic_seq_condition_l2154_215402

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := 
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_seq_condition (a2 : ℕ) (S3 S9 : ℕ) :
  a2 = 1 → 
  (∃ d, (d > 4 ∧ S3 = 3 * a2 + (3 * (3 - 1) / 2) * d ∧ S9 = 9 * a2 + (9 * (9 - 1) / 2) * d) → (S3 + S9) > 93) ↔ 
  (∃ d, (S3 + S9 = sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9 ∧ (sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9) > 93 → d > 3 ∧ a2 + d > 5)) :=
by 
  sorry

end arithmetic_seq_condition_l2154_215402


namespace total_points_each_team_l2154_215419

def score_touchdown := 7
def score_field_goal := 3
def score_safety := 2

def team_hawks_first_match_score := 3 * score_touchdown + 2 * score_field_goal + score_safety
def team_eagles_first_match_score := 5 * score_touchdown + 4 * score_field_goal
def team_hawks_second_match_score := 4 * score_touchdown + 3 * score_field_goal
def team_falcons_second_match_score := 6 * score_touchdown + 2 * score_safety

def total_score_hawks := team_hawks_first_match_score + team_hawks_second_match_score
def total_score_eagles := team_eagles_first_match_score
def total_score_falcons := team_falcons_second_match_score

theorem total_points_each_team :
  total_score_hawks = 66 ∧ total_score_eagles = 47 ∧ total_score_falcons = 46 :=
by
  unfold total_score_hawks team_hawks_first_match_score team_hawks_second_match_score
           total_score_eagles team_eagles_first_match_score
           total_score_falcons team_falcons_second_match_score
           score_touchdown score_field_goal score_safety
  sorry

end total_points_each_team_l2154_215419


namespace multiply_polynomials_l2154_215436

variables {R : Type*} [CommRing R] -- Define R as a commutative ring
variable (x : R) -- Define variable x in R

theorem multiply_polynomials : (2 * x) * (5 * x^2) = 10 * x^3 := 
sorry -- Placeholder for the proof

end multiply_polynomials_l2154_215436


namespace infinitely_many_triples_no_triples_l2154_215407

theorem infinitely_many_triples :
  ∃ (m n p : ℕ), ∃ (k : ℕ), m > 0 ∧ n > 0 ∧ p > 0 ∧ 4 * m * n - m - n = p ^ 2 - 1 := 
sorry

theorem no_triples :
  ¬∃ (m n p : ℕ), m > 0 ∧ n > 0 ∧ p > 0 ∧ 4 * m * n - m - n = p ^ 2 := 
sorry

end infinitely_many_triples_no_triples_l2154_215407


namespace sqrt_neg_square_real_l2154_215499

theorem sqrt_neg_square_real : ∃! (x : ℝ), -(x + 2) ^ 2 = 0 := by
  sorry

end sqrt_neg_square_real_l2154_215499


namespace next_two_equations_l2154_215482

-- Definitions based on the conditions in the problem
def pattern1 (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Statement to prove the continuation of the pattern
theorem next_two_equations 
: pattern1 9 40 41 ∧ pattern1 11 60 61 :=
by
  sorry

end next_two_equations_l2154_215482


namespace measure_of_angle_x_l2154_215449

theorem measure_of_angle_x :
  ∀ (angle_ABC angle_BDE angle_DBE angle_ABD x : ℝ),
    angle_ABC = 132 ∧
    angle_BDE = 31 ∧
    angle_DBE = 30 ∧
    angle_ABD = 180 - 132 →
    x = 180 - (angle_BDE + angle_DBE) →
    x = 119 :=
by
  intros angle_ABC angle_BDE angle_DBE angle_ABD x h h2
  sorry

end measure_of_angle_x_l2154_215449


namespace inequality_xy_l2154_215432

-- Defining the constants and conditions
variables {x y : ℝ}

-- Main theorem to prove the inequality and find pairs for equality
theorem inequality_xy (h : (x + 1) * (y + 2) = 8) :
  (xy - 10)^2 ≥ 64 ∧ ((xy - 10)^2 = 64 → (x, y) = (1, 2) ∨ (x, y) = (-3, -6)) :=
sorry

end inequality_xy_l2154_215432


namespace find_value_of_2_minus_c_l2154_215441

theorem find_value_of_2_minus_c (c d : ℤ) (h1 : 5 + c = 6 - d) (h2 : 3 + d = 8 + c) : 2 - c = -1 := 
by
  sorry

end find_value_of_2_minus_c_l2154_215441


namespace max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l2154_215463

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a ≤ 2 :=
by
  -- Proof omitted
  sorry

theorem le_2_and_ge_neg_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : -2 ≤ a :=
by
  -- Proof omitted
  sorry

theorem max_a_is_2 (a : ℝ) (h3 : a ≤ 2) (h4 : -2 ≤ a) : a = 2 :=
by
  -- Proof omitted
  sorry

end max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l2154_215463


namespace sum_difference_4041_l2154_215412

def sum_of_first_n_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_difference_4041 :
  sum_of_first_n_integers 2021 - sum_of_first_n_integers 2019 = 4041 :=
by
  sorry

end sum_difference_4041_l2154_215412


namespace production_average_l2154_215435

theorem production_average (n : ℕ) (P : ℕ) (h1 : P / n = 50) (h2 : (P + 90) / (n + 1) = 54) : n = 9 :=
sorry

end production_average_l2154_215435


namespace problem_range_of_a_l2154_215492

theorem problem_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |3 + x| ≥ a^2 - 4 * a) ↔ -1 ≤ a ∧ a ≤ 5 :=
by
  sorry

end problem_range_of_a_l2154_215492


namespace f_evaluation_l2154_215440

def f (a b c : ℚ) : ℚ := a^2 + 2 * b * c

theorem f_evaluation :
  f 1 23 76 + f 23 76 1 + f 76 1 23 = 10000 := by
  sorry

end f_evaluation_l2154_215440


namespace general_admission_price_l2154_215409

theorem general_admission_price :
  ∃ x : ℝ,
    ∃ G V : ℕ,
      VIP_price = 45 ∧ Total_tickets_sold = 320 ∧ Total_revenue = 7500 ∧ VIP_tickets_less = 276 ∧
      G + V = Total_tickets_sold ∧ V = G - VIP_tickets_less ∧ 45 * V + x * G = Total_revenue ∧ x = 21.85 :=
sorry

end general_admission_price_l2154_215409


namespace marble_color_197th_l2154_215459

theorem marble_color_197th (n : ℕ) (total_marbles : ℕ) (marble_color : ℕ → ℕ)
                          (h_total : total_marbles = 240)
                          (h_pattern : ∀ k, marble_color (k + 15) = marble_color k)
                          (h_colors : ∀ i, (0 ≤ i ∧ i < 15) →
                                   (marble_color i = if i < 6 then 1
                                   else if i < 11 then 2
                                   else if i < 15 then 3
                                   else 0)) :
  marble_color 197 = 1 := sorry

end marble_color_197th_l2154_215459


namespace phone_number_fraction_l2154_215434

theorem phone_number_fraction : 
  let total_valid_numbers := 6 * (10^6)
  let valid_numbers_with_conditions := 10^5
  valid_numbers_with_conditions / total_valid_numbers = 1 / 60 :=
by sorry

end phone_number_fraction_l2154_215434


namespace jason_money_in_usd_l2154_215421

noncomputable def jasonTotalInUSD : ℝ :=
  let init_quarters_value := 49 * 0.25
  let init_dimes_value    := 32 * 0.10
  let init_nickels_value  := 18 * 0.05
  let init_euros_in_usd   := 22.50 * 1.20
  let total_initial       := init_quarters_value + init_dimes_value + init_nickels_value + init_euros_in_usd

  let dad_quarters_value  := 25 * 0.25
  let dad_dimes_value     := 15 * 0.10
  let dad_nickels_value   := 10 * 0.05
  let dad_euros_in_usd    := 12 * 1.20
  let total_additional    := dad_quarters_value + dad_dimes_value + dad_nickels_value + dad_euros_in_usd

  total_initial + total_additional

theorem jason_money_in_usd :
  jasonTotalInUSD = 66 := 
sorry

end jason_money_in_usd_l2154_215421


namespace shooter_hit_rate_l2154_215462

noncomputable def shooter_prob := 2 / 3

theorem shooter_hit_rate:
  ∀ (x : ℚ), (1 - x)^4 = 1 / 81 → x = shooter_prob :=
by
  intro x h
  -- Proof is omitted
  sorry

end shooter_hit_rate_l2154_215462


namespace trapezium_other_side_length_l2154_215437

theorem trapezium_other_side_length (a h Area : ℕ) (a_eq : a = 4) (h_eq : h = 6) (Area_eq : Area = 27) : 
  ∃ (b : ℕ), b = 5 := 
by
  sorry

end trapezium_other_side_length_l2154_215437


namespace coordinates_of_C_l2154_215458

structure Point :=
  (x : Int)
  (y : Int)

def reflect_over_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

def reflect_over_y_axis (p : Point) : Point :=
  {x := -p.x, y := p.y}

def C : Point := {x := 2, y := 2}

noncomputable def C'_reflected_x := reflect_over_x_axis C
noncomputable def C''_reflected_y := reflect_over_y_axis C'_reflected_x

theorem coordinates_of_C'' : C''_reflected_y = {x := -2, y := -2} :=
by
  sorry

end coordinates_of_C_l2154_215458


namespace total_songs_purchased_is_162_l2154_215401

variable (c_country : ℕ) (c_pop : ℕ) (c_jazz : ℕ) (c_rock : ℕ)
variable (s_country : ℕ) (s_pop : ℕ) (s_jazz : ℕ) (s_rock : ℕ)

-- Setting up the conditions
def num_country_albums := 6
def num_pop_albums := 2
def num_jazz_albums := 4
def num_rock_albums := 3

-- Number of songs per album
def country_album_songs := 9
def pop_album_songs := 9
def jazz_album_songs := 12
def rock_album_songs := 14

theorem total_songs_purchased_is_162 :
  num_country_albums * country_album_songs +
  num_pop_albums * pop_album_songs +
  num_jazz_albums * jazz_album_songs +
  num_rock_albums * rock_album_songs = 162 := by
  sorry

end total_songs_purchased_is_162_l2154_215401


namespace lisa_score_is_85_l2154_215491

def score_formula (c w : ℕ) : ℕ := 30 + 4 * c - w

theorem lisa_score_is_85 (c w : ℕ) 
  (score_equality : 85 = score_formula c w)
  (non_neg_w : w ≥ 0)
  (total_questions : c + w ≤ 30) :
  (c = 14 ∧ w = 1) :=
by
  sorry

end lisa_score_is_85_l2154_215491


namespace max_length_interval_l2154_215488

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := ((m ^ 2 + m) * x - 1) / (m ^ 2 * x)

theorem max_length_interval (a b m : ℝ) (h1 : m ≠ 0) (h2 : ∀ x, f m x = x → x ∈ Set.Icc a b) :
  |b - a| = (2 * Real.sqrt 3) / 3 := sorry

end max_length_interval_l2154_215488


namespace Rahul_savings_l2154_215483

variable (total_savings ppf_savings nsc_savings x : ℝ)

theorem Rahul_savings
  (h1 : total_savings = 180000)
  (h2 : ppf_savings = 72000)
  (h3 : nsc_savings = total_savings - ppf_savings)
  (h4 : x * nsc_savings = 0.5 * ppf_savings) :
  x = 1 / 3 :=
by
  -- Proof goes here
  sorry

end Rahul_savings_l2154_215483


namespace rectangular_field_perimeter_l2154_215416

-- Definitions for conditions
def width : ℕ := 75
def length : ℕ := (7 * width) / 5
def perimeter (L W : ℕ) : ℕ := 2 * (L + W)

-- Statement to prove
theorem rectangular_field_perimeter : perimeter length width = 360 := by
  sorry

end rectangular_field_perimeter_l2154_215416


namespace fourth_power_sum_l2154_215496

theorem fourth_power_sum
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 19.5 := 
sorry

end fourth_power_sum_l2154_215496


namespace ratio_a_b_l2154_215490

theorem ratio_a_b (a b c d : ℝ) 
  (h1 : b / c = 7 / 9) 
  (h2 : c / d = 5 / 7)
  (h3 : a / d = 5 / 12) : 
  a / b = 3 / 4 :=
  sorry

end ratio_a_b_l2154_215490


namespace aaron_total_amount_owed_l2154_215451

def total_cost (monthly_payment : ℤ) (months : ℤ) : ℤ :=
  monthly_payment * months

def interest_fee (amount : ℤ) (rate : ℤ) : ℤ :=
  amount * rate / 100

def total_amount_owed (monthly_payment : ℤ) (months : ℤ) (rate : ℤ) : ℤ :=
  let amount := total_cost monthly_payment months
  let fee := interest_fee amount rate
  amount + fee

theorem aaron_total_amount_owed :
  total_amount_owed 100 12 10 = 1320 :=
by
  sorry

end aaron_total_amount_owed_l2154_215451


namespace simplify_fraction_l2154_215429

variable (k : ℤ)

theorem simplify_fraction (a b : ℤ)
  (hk : a = 2)
  (hb : b = 4) :
  (6 * k + 12) / 3 = 2 * k + 4 ∧ (a : ℚ) / (b : ℚ) = 1 / 2 := 
by
  sorry

end simplify_fraction_l2154_215429


namespace evaTotalMarksCorrect_l2154_215460

-- Definition of marks scored by Eva in each subject across semesters
def evaMathsMarksSecondSemester : Nat := 80
def evaArtsMarksSecondSemester : Nat := 90
def evaScienceMarksSecondSemester : Nat := 90

def evaMathsMarksFirstSemester : Nat := evaMathsMarksSecondSemester + 10
def evaArtsMarksFirstSemester : Nat := evaArtsMarksSecondSemester - 15
def evaScienceMarksFirstSemester : Nat := evaScienceMarksSecondSemester - (evaScienceMarksSecondSemester / 3)

-- Total marks in each semester
def totalMarksFirstSemester : Nat := evaMathsMarksFirstSemester + evaArtsMarksFirstSemester + evaScienceMarksFirstSemester
def totalMarksSecondSemester : Nat := evaMathsMarksSecondSemester + evaArtsMarksSecondSemester + evaScienceMarksSecondSemester

-- Combined total
def evaTotalMarks : Nat := totalMarksFirstSemester + totalMarksSecondSemester

-- Statement to prove
theorem evaTotalMarksCorrect : evaTotalMarks = 485 := 
by
  -- This needs to be proved as per the conditions and calculations above
  sorry

end evaTotalMarksCorrect_l2154_215460


namespace correct_options_l2154_215479

-- Given conditions
def f : ℝ → ℝ := sorry -- We will assume there is some function f that satisfies the conditions

axiom xy_identity (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = x * f y + y * f x
axiom f_positive (x : ℝ) (hx : 1 < x) : 0 < f x

-- Proof of the required conclusion
theorem correct_options (h1 : f 1 = 0) (h2 : ∀ x y, f (x * y) ≠ f x * f y)
  (h3 : ∀ x, 1 < x → ∀ y, 1 < y → x < y → f x < f y)
  (h4 : ∀ x, 2 ≤ x → x * f (x - 3 / 2) ≥ (3 / 2 - x) * f x) : 
  f 1 = 0 ∧ (∀ x y, f (x * y) ≠ f x * f y) ∧ (∀ x, 1 < x → ∀ y, 1 < y → x < y → f x < f y) ∧ (∀ x, 2 ≤ x → x * f (x - 3 / 2) ≥ (3 / 2 - x) * f x) :=
sorry

end correct_options_l2154_215479


namespace train_start_time_l2154_215473

theorem train_start_time (D PQ : ℝ) (S₁ S₂ : ℝ) (T₁ T₂ meet : ℝ) :
  PQ = 110  -- Distance between stations P and Q
  ∧ S₁ = 20  -- Speed of the first train
  ∧ S₂ = 25  -- Speed of the second train
  ∧ T₂ = 8  -- Start time of the second train
  ∧ meet = 10 -- Meeting time
  ∧ T₁ + T₂ = meet → -- Meeting time condition
  T₁ = 7.5 := -- Answer: first train start time
by
sorry

end train_start_time_l2154_215473


namespace find_pairs_l2154_215471

theorem find_pairs (x y : ℕ) (h : x > 0 ∧ y > 0) (d : ℕ) (gcd_cond : d = Nat.gcd x y)
  (eqn_cond : x * y * d = x + y + d ^ 2) : (x, y) = (2, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (3, 2) :=
by {
  sorry
}

end find_pairs_l2154_215471


namespace triangle_inequality_l2154_215433

theorem triangle_inequality (a b c p q r : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum_zero : p + q + r = 0) : 
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
sorry

end triangle_inequality_l2154_215433


namespace how_many_more_cups_of_sugar_l2154_215417

def required_sugar : ℕ := 11
def required_flour : ℕ := 9
def added_flour : ℕ := 12
def added_sugar : ℕ := 10

theorem how_many_more_cups_of_sugar :
  required_sugar - added_sugar = 1 :=
by
  sorry

end how_many_more_cups_of_sugar_l2154_215417


namespace find_b_l2154_215414

variable (p q r b : ℤ)

-- Conditions
def condition1 : Prop := p - q = 2
def condition2 : Prop := p - r = 1

-- The main statement to prove
def problem_statement : Prop :=
  b = (r - q) * ((p - q)^2 + (p - q) * (p - r) + (p - r)^2) → b = 7

theorem find_b (h1 : condition1 p q) (h2 : condition2 p r) (h3 : problem_statement p q r b) : b = 7 :=
sorry

end find_b_l2154_215414


namespace range_of_b_plus_c_l2154_215422

noncomputable def func (b c x : ℝ) : ℝ := x^2 + b*x + c * 3^x

theorem range_of_b_plus_c {b c : ℝ} (h1 : ∃ x, func b c x = 0)
  (h2 : ∀ x, (func b c x = 0 ↔ func b c (func b c x) = 0)) : 
  0 ≤ b + c ∧ b + c < 4 :=
by
  sorry

end range_of_b_plus_c_l2154_215422


namespace inversely_proportional_example_l2154_215484

theorem inversely_proportional_example (x y k : ℝ) (h₁ : x * y = k) (h₂ : x = 30) (h₃ : y = 8) :
  y = 24 → x = 10 :=
by
  sorry

end inversely_proportional_example_l2154_215484


namespace xy_yz_zx_value_l2154_215480

namespace MathProof

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 147) 
  (h2 : y^2 + y * z + z^2 = 16) 
  (h3 : z^2 + x * z + x^2 = 163) :
  x * y + y * z + z * x = 56 := 
sorry      

end MathProof

end xy_yz_zx_value_l2154_215480


namespace trisha_total_distance_walked_l2154_215456

def d1 : ℝ := 0.1111111111111111
def d2 : ℝ := 0.1111111111111111
def d3 : ℝ := 0.6666666666666666

theorem trisha_total_distance_walked :
  d1 + d2 + d3 = 0.8888888888888888 := 
sorry

end trisha_total_distance_walked_l2154_215456


namespace quadratic_has_distinct_real_roots_l2154_215481

-- Definitions for the quadratic equation coefficients
def a : ℝ := 3
def b : ℝ := -4
def c : ℝ := 1

-- Definition of the discriminant
def Δ : ℝ := b^2 - 4 * a * c

-- Statement of the problem: Prove that the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots (hΔ : Δ = 4) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by
  sorry

end quadratic_has_distinct_real_roots_l2154_215481


namespace valentines_initial_l2154_215410

theorem valentines_initial (gave_away : ℕ) (left_over : ℕ) (initial : ℕ) : 
  gave_away = 8 → left_over = 22 → initial = gave_away + left_over → initial = 30 :=
by
  intros h1 h2 h3
  sorry

end valentines_initial_l2154_215410


namespace test_score_after_preparation_l2154_215405

-- Define the conditions in Lean 4
def score (k t : ℝ) : ℝ := k * t^2

theorem test_score_after_preparation (k t : ℝ)
    (h1 : score k 2 = 90) (h2 : k = 22.5) :
    score k 3 = 202.5 :=
by
  sorry

end test_score_after_preparation_l2154_215405


namespace nancy_kept_chips_l2154_215404

def nancy_initial_chips : ℕ := 22
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5

theorem nancy_kept_chips : nancy_initial_chips - (chips_given_to_brother + chips_given_to_sister) = 10 :=
by
  sorry

end nancy_kept_chips_l2154_215404


namespace find_a_l2154_215478

theorem find_a (a : ℚ) (A : Set ℚ) (h : 3 ∈ A) (hA : A = {a + 2, 2 * a^2 + a}) : a = 3 / 2 := 
by
  sorry

end find_a_l2154_215478


namespace find_C_l2154_215489

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def A : ℕ := sum_of_digits (4568 ^ 7777)
noncomputable def B : ℕ := sum_of_digits A
noncomputable def C : ℕ := sum_of_digits B

theorem find_C : C = 5 :=
by
  sorry

end find_C_l2154_215489


namespace min_denominator_of_sum_600_700_l2154_215445

def is_irreducible_fraction (a : ℕ) (b : ℕ) : Prop := 
  Nat.gcd a b = 1

def min_denominator_of_sum (d1 d2 : ℕ) (a b : ℕ) : ℕ :=
  let lcm := Nat.lcm d1 d2
  let sum_numerator := a * (lcm / d1) + b * (lcm / d2)
  Nat.gcd sum_numerator lcm

theorem min_denominator_of_sum_600_700 (a b : ℕ) (h1 : is_irreducible_fraction a 600) (h2 : is_irreducible_fraction b 700) :
  min_denominator_of_sum 600 700 a b = 168 := sorry

end min_denominator_of_sum_600_700_l2154_215445


namespace cristina_pace_is_4_l2154_215446

-- Definitions of the conditions
def head_start : ℝ := 36
def nicky_pace : ℝ := 3
def time : ℝ := 36

-- Definition of the distance Nicky runs
def distance_nicky_runs : ℝ := nicky_pace * time

-- Definition of the total distance Cristina ran to catch up
def distance_cristina_runs : ℝ := distance_nicky_runs + head_start

-- Lean 4 theorem statement to prove Cristina's pace
theorem cristina_pace_is_4 :
  (distance_cristina_runs / time) = 4 := 
by sorry

end cristina_pace_is_4_l2154_215446


namespace evaluated_expression_l2154_215475

noncomputable def evaluation_problem (x a y z c d : ℝ) : ℝ :=
  (2 * x^3 - 3 * a^4) / (y^2 + 4 * z^5) + c^4 - d^2

theorem evaluated_expression :
  evaluation_problem 0.66 0.1 0.66 0.1 0.066 0.1 = 1.309091916 :=
by
  sorry

end evaluated_expression_l2154_215475


namespace rectangle_width_l2154_215461

theorem rectangle_width (L W : ℝ) (h₁ : 2 * L + 2 * W = 54) (h₂ : W = L + 3) : W = 15 :=
sorry

end rectangle_width_l2154_215461


namespace gcd_gt_one_l2154_215426

-- Defining the given conditions and the statement to prove
theorem gcd_gt_one (a b x y : ℕ) (h : (a^2 + b^2) ∣ (a * x + b * y)) : 
  Nat.gcd (x^2 + y^2) (a^2 + b^2) > 1 := 
sorry

end gcd_gt_one_l2154_215426


namespace cheryl_material_left_l2154_215427

-- Conditions
def initial_material_type1 (m1 : ℚ) : Prop := m1 = 2/9
def initial_material_type2 (m2 : ℚ) : Prop := m2 = 1/8
def used_material (u : ℚ) : Prop := u = 0.125

-- Define the total material bought
def total_material (m1 m2 : ℚ) : ℚ := m1 + m2

-- Define the material left
def material_left (t u : ℚ) : ℚ := t - u

-- The target theorem
theorem cheryl_material_left (m1 m2 u : ℚ) 
  (h1 : initial_material_type1 m1)
  (h2 : initial_material_type2 m2)
  (h3 : used_material u) : 
  material_left (total_material m1 m2) u = 2/9 :=
by
  sorry

end cheryl_material_left_l2154_215427


namespace solve_for_q_l2154_215424

theorem solve_for_q (q : ℕ) : 16^4 = (8^3 / 2 : ℕ) * 2^(16 * q) → q = 1 / 2 :=
by
  sorry

end solve_for_q_l2154_215424


namespace average_of_numbers_between_6_and_36_divisible_by_7_l2154_215443

noncomputable def average_of_divisibles_by_seven : ℕ :=
  let numbers := [7, 14, 21, 28, 35]
  let sum := numbers.sum
  let count := numbers.length
  sum / count

theorem average_of_numbers_between_6_and_36_divisible_by_7 : average_of_divisibles_by_seven = 21 :=
by
  sorry

end average_of_numbers_between_6_and_36_divisible_by_7_l2154_215443


namespace contrapositive_example_l2154_215493

theorem contrapositive_example (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → (a^2 + b^2 ≠ 0) :=
by
  sorry

end contrapositive_example_l2154_215493


namespace problem1_problem2_l2154_215439

-- Proof problem 1
theorem problem1 : (-3)^2 / 3 + abs (-7) + 3 * (-1/3) = 3 :=
by
  sorry

-- Proof problem 2
theorem problem2 : (-1) ^ 2022 - ( (-1/4) - (-1/3) ) / (-1/12) = 2 :=
by
  sorry

end problem1_problem2_l2154_215439


namespace problem_statement_l2154_215411

theorem problem_statement 
  (a b c : ℤ)
  (h1 : (5 * a + 2) ^ (1/3) = 3)
  (h2 : (3 * a + b - 1) ^ (1/2) = 4)
  (h3 : c = Int.floor (Real.sqrt 13))
  : a = 5 ∧ b = 2 ∧ c = 3 ∧ Real.sqrt (3 * a - b + c) = 4 := 
by 
  sorry

end problem_statement_l2154_215411


namespace M_Mobile_cheaper_than_T_Mobile_l2154_215497

def T_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 50
  else 50 + (lines - 2) * 16

def M_Mobile_total_cost (lines : ℕ) : ℕ :=
  if lines <= 2 then 45
  else 45 + (lines - 2) * 14

theorem M_Mobile_cheaper_than_T_Mobile : 
  T_Mobile_total_cost 5 - M_Mobile_total_cost 5 = 11 :=
by
  sorry

end M_Mobile_cheaper_than_T_Mobile_l2154_215497


namespace sum_squares_mod_13_is_zero_l2154_215467

def sum_squares_mod_13 : ℕ :=
  (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2 + 11^2 + 12^2) % 13

theorem sum_squares_mod_13_is_zero : sum_squares_mod_13 = 0 := by
  sorry

end sum_squares_mod_13_is_zero_l2154_215467


namespace fifth_friend_paid_40_l2154_215472

variable (x1 x2 x3 x4 x5 : ℝ)

def conditions : Prop :=
  (x1 = 1/3 * (x2 + x3 + x4 + x5)) ∧
  (x2 = 1/4 * (x1 + x3 + x4 + x5)) ∧
  (x3 = 1/5 * (x1 + x2 + x4 + x5)) ∧
  (x4 = 1/6 * (x1 + x2 + x3 + x5)) ∧
  (x1 + x2 + x3 + x4 + x5 = 120)

theorem fifth_friend_paid_40 (h : conditions x1 x2 x3 x4 x5) : x5 = 40 := by
  sorry

end fifth_friend_paid_40_l2154_215472


namespace combined_mpg_correct_l2154_215468

def ray_mpg := 30
def tom_mpg := 15
def alice_mpg := 60
def distance_each := 120

-- Total gasoline consumption
def ray_gallons := distance_each / ray_mpg
def tom_gallons := distance_each / tom_mpg
def alice_gallons := distance_each / alice_mpg

def total_gallons := ray_gallons + tom_gallons + alice_gallons
def total_distance := 3 * distance_each

def combined_mpg := total_distance / total_gallons

theorem combined_mpg_correct :
  combined_mpg = 26 :=
by
  -- All the necessary calculations would go here.
  sorry

end combined_mpg_correct_l2154_215468


namespace plane_point_to_center_ratio_l2154_215444

variable (a b c p q r : ℝ)

theorem plane_point_to_center_ratio :
  (a / p) + (b / q) + (c / r) = 2 ↔ 
  (∀ (α β γ : ℝ), α = 2 * p ∧ β = 2 * q ∧ γ = 2 * r ∧ (α, 0, 0) = (a, b, c) → 
  (a / (2 * p)) + (b / (2 * q)) + (c / (2 * r)) = 1) :=
by {
  sorry
}

end plane_point_to_center_ratio_l2154_215444


namespace nine_b_equals_eighteen_l2154_215430

theorem nine_b_equals_eighteen (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 9 * b = 18 :=
  sorry

end nine_b_equals_eighteen_l2154_215430


namespace selling_price_per_sweater_correct_l2154_215425

-- Definitions based on the problem's conditions
def balls_of_yarn_per_sweater := 4
def cost_per_ball_of_yarn := 6
def number_of_sweaters := 28
def total_gain := 308

-- Defining the required selling price per sweater
def total_cost_of_yarn : Nat := balls_of_yarn_per_sweater * cost_per_ball_of_yarn * number_of_sweaters
def total_revenue : Nat := total_cost_of_yarn + total_gain
def selling_price_per_sweater : ℕ := total_revenue / number_of_sweaters

theorem selling_price_per_sweater_correct :
  selling_price_per_sweater = 35 :=
  by
  sorry

end selling_price_per_sweater_correct_l2154_215425


namespace boys_more_than_girls_l2154_215466

theorem boys_more_than_girls
  (x y a b : ℕ)
  (h1 : x > y)
  (h2 : x * a + y * b = x * b + y * a - 1) :
  x = y + 1 :=
sorry

end boys_more_than_girls_l2154_215466


namespace student_19_in_sample_l2154_215448

-- Definitions based on conditions
def total_students := 52
def sample_size := 4
def sampling_interval := 13

def selected_students := [6, 32, 45]

-- The theorem to prove
theorem student_19_in_sample : 19 ∈ selected_students ∨ ∃ k : ℕ, 13 * k + 6 = 19 :=
by
  sorry

end student_19_in_sample_l2154_215448


namespace paul_oil_change_rate_l2154_215428

theorem paul_oil_change_rate (P : ℕ) (h₁ : 8 * (P + 3) = 40) : P = 2 :=
by
  sorry

end paul_oil_change_rate_l2154_215428


namespace smaller_number_l2154_215487

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 :=
by
  sorry

end smaller_number_l2154_215487


namespace distinct_real_pairs_l2154_215423

theorem distinct_real_pairs (x y : ℝ) (h1 : x ≠ y) (h2 : x^100 - y^100 = 2^99 * (x - y)) (h3 : x^200 - y^200 = 2^199 * (x - y)) :
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
sorry

end distinct_real_pairs_l2154_215423


namespace max_value_of_expr_l2154_215457

open Classical
open Real

theorem max_value_of_expr 
  (x y : ℝ) 
  (h₁ : 0 < x) 
  (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  ∃ a b c d : ℝ, 
    (x^2 + 2 * x * y + 3 * y^2 = 20 + 10 * sqrt 3) ∧ 
    (a = 20) ∧ 
    (b = 10) ∧ 
    (c = 3) ∧ 
    (d = 2) := 
sorry

end max_value_of_expr_l2154_215457


namespace cos_B_of_triangle_l2154_215469

theorem cos_B_of_triangle (A B : ℝ) (a b : ℝ) (h1 : A = 2 * B) (h2 : a = 6) (h3 : b = 4) :
  Real.cos B = 3 / 4 :=
by
  sorry

end cos_B_of_triangle_l2154_215469


namespace sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l2154_215438

theorem sixty_percent_of_fifty_minus_thirty_percent_of_thirty : 
  (60 / 100 : ℝ) * 50 - (30 / 100 : ℝ) * 30 = 21 :=
by
  sorry

end sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l2154_215438


namespace arithmetic_geometric_inequality_l2154_215415

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) := 
sorry

end arithmetic_geometric_inequality_l2154_215415


namespace print_time_l2154_215406

-- Define the conditions
def pages : ℕ := 345
def rate : ℕ := 23
def expected_minutes : ℕ := 15

-- State the problem as a theorem
theorem print_time (pages rate : ℕ) : (pages / rate = 15) :=
by
  sorry

end print_time_l2154_215406


namespace bailey_rawhide_bones_l2154_215486

variable (dog_treats : ℕ) (chew_toys : ℕ) (total_items : ℕ)
variable (credit_cards : ℕ) (items_per_card : ℕ)

theorem bailey_rawhide_bones :
  (dog_treats = 8) →
  (chew_toys = 2) →
  (credit_cards = 4) →
  (items_per_card = 5) →
  (total_items = credit_cards * items_per_card) →
  (total_items - (dog_treats + chew_toys) = 10) :=
by
  intros
  sorry

end bailey_rawhide_bones_l2154_215486


namespace largest_integer_same_cost_l2154_215474

def sum_decimal_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sum_ternary_digits (n : ℕ) : ℕ :=
  n.digits 3 |>.sum

theorem largest_integer_same_cost :
  ∃ n : ℕ, n < 1000 ∧ sum_decimal_digits n = sum_ternary_digits n ∧ ∀ m : ℕ, m < 1000 ∧ sum_decimal_digits m = sum_ternary_digits m → m ≤ n := 
  sorry

end largest_integer_same_cost_l2154_215474


namespace Freddy_age_l2154_215455

noncomputable def M : ℕ := 11
noncomputable def R : ℕ := M - 2
noncomputable def F : ℕ := M + 4

theorem Freddy_age : F = 15 :=
  by
    sorry

end Freddy_age_l2154_215455


namespace additional_dividend_amount_l2154_215452

theorem additional_dividend_amount
  (E : ℝ) (Q : ℝ) (expected_extra_per_earnings : ℝ) (half_of_extra_per_earnings_to_dividend : ℝ) 
  (expected : E = 0.80) (quarterly_earnings : Q = 1.10)
  (extra_per_earnings : expected_extra_per_earnings = 0.30)
  (half_dividend : half_of_extra_per_earnings_to_dividend = 0.15):
  Q - E = expected_extra_per_earnings ∧ 
  expected_extra_per_earnings / 2 = half_of_extra_per_earnings_to_dividend :=
by sorry

end additional_dividend_amount_l2154_215452


namespace m_perp_beta_l2154_215464

variable {Point Line Plane : Type}
variable {belongs : Point → Line → Prop}
variable {perp : Line → Plane → Prop}
variable {intersect : Plane → Plane → Line}

variable (α β γ : Plane)
variable (m n l : Line)

-- Conditions for the problem
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- Proof goal: proving m is perpendicular to β
theorem m_perp_beta : perp m β :=
by
  sorry

end m_perp_beta_l2154_215464


namespace jake_present_weight_l2154_215477

variables (J S : ℕ)

theorem jake_present_weight :
  (J - 33 = 2 * S) ∧ (J + S = 153) → J = 113 :=
by
  sorry

end jake_present_weight_l2154_215477


namespace percentage_of_white_chips_l2154_215420

theorem percentage_of_white_chips (T : ℕ) (h1 : 3 = 10 * T / 100) (h2 : 12 = 12): (15 / T * 100) = 50 := by
  sorry

end percentage_of_white_chips_l2154_215420


namespace quadratic_coefficients_l2154_215413

theorem quadratic_coefficients (b c : ℝ) :
  (∀ x : ℝ, |x + 4| = 3 ↔ x^2 + bx + c = 0) → (b = 8 ∧ c = 7) :=
by
  sorry

end quadratic_coefficients_l2154_215413


namespace valid_n_value_l2154_215450

theorem valid_n_value (n : ℕ) (a : ℕ → ℕ)
    (h1 : ∀ k : ℕ, 1 ≤ k ∧ k < n → k ∣ a k)
    (h2 : ¬ n ∣ a n)
    (h3 : 2 ≤ n) :
    ∃ (p : ℕ) (α : ℕ), (Nat.Prime p) ∧ (n = p ^ α) ∧ (α ≥ 1) :=
by sorry

end valid_n_value_l2154_215450
