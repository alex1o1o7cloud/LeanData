import Mathlib

namespace time_to_save_for_downpayment_l179_179389

-- Definitions based on conditions
def annual_saving : ℝ := 0.10 * 150000
def downpayment : ℝ := 0.20 * 450000

-- Statement of the theorem to be proved
theorem time_to_save_for_downpayment (T : ℝ) (H1 : annual_saving = 15000) (H2 : downpayment = 90000) : 
  T = 6 :=
by
  -- Placeholder for the proof
  sorry

end time_to_save_for_downpayment_l179_179389


namespace painting_time_eq_l179_179030

theorem painting_time_eq (t : ℚ) : 
  (1/6 + 1/8 + 1/10) * (t - 2) = 1 := 
sorry

end painting_time_eq_l179_179030


namespace sum_divisible_by_seventeen_l179_179250

theorem sum_divisible_by_seventeen :
  (90 + 91 + 92 + 93 + 94 + 95 + 96 + 97) % 17 = 0 := 
by 
  sorry

end sum_divisible_by_seventeen_l179_179250


namespace find_c_l179_179810

open Real

def vector := (ℝ × ℝ)

def a : vector := (1, 2)
def b : vector := (2, -3)

def is_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_c (c : vector) : 
  (is_parallel (c.1 + a.1, c.2 + a.2) b) ∧ (is_perpendicular c (a.1 + b.1, a.2 + b.2)) → 
  c = (-7 / 9, -20 / 9) := 
by
  sorry

end find_c_l179_179810


namespace initial_average_age_l179_179329

theorem initial_average_age (A : ℕ) (h1 : ∀ x : ℕ, 10 * A = 10 * A)
  (h2 : 5 * 17 + 10 * A = 15 * (A + 1)) : A = 14 :=
by 
  sorry

end initial_average_age_l179_179329


namespace expand_polynomial_l179_179747

noncomputable def polynomial_expression (x : ℝ) : ℝ := -2 * (x - 3) * (x + 4) * (2 * x - 1)

theorem expand_polynomial (x : ℝ) :
  polynomial_expression x = -4 * x^3 - 2 * x^2 + 50 * x - 24 :=
sorry

end expand_polynomial_l179_179747


namespace set_intersection_eq_l179_179051

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- The proof statement
theorem set_intersection_eq :
  A ∩ B = A :=
sorry

end set_intersection_eq_l179_179051


namespace simplify_sum1_simplify_sum2_l179_179289

theorem simplify_sum1 : 296 + 297 + 298 + 299 + 1 + 2 + 3 + 4 = 1200 := by
  sorry

theorem simplify_sum2 : 457 + 458 + 459 + 460 + 461 + 462 + 463 = 3220 := by
  sorry

end simplify_sum1_simplify_sum2_l179_179289


namespace both_selected_probability_l179_179634

-- Define the probabilities of selection for X and Y
def P_X := 1 / 7
def P_Y := 2 / 9

-- Statement to prove that the probability of both being selected is 2 / 63
theorem both_selected_probability :
  (P_X * P_Y) = (2 / 63) :=
by
  -- Proof skipped
  sorry

end both_selected_probability_l179_179634


namespace number_of_music_files_l179_179427

-- The conditions given in the problem
variable {M : ℕ} -- M is a natural number representing the initial number of music files

-- Conditions: Initial state and changes
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23
def remaining_files : ℕ := 2

-- Statement of the theorem
theorem number_of_music_files (h : M + initial_video_files - files_deleted = remaining_files) : M = 4 :=
  by
  -- Proof goes here
  sorry

end number_of_music_files_l179_179427


namespace boat_travel_distance_upstream_l179_179084

noncomputable def upstream_distance (v : ℝ) : ℝ :=
  let d := 2.5191640969412834 * (v + 3)
  d

theorem boat_travel_distance_upstream :
  ∀ v : ℝ, 
  (∀ D : ℝ, D / (v + 3) = 2.5191640969412834 → D / (v - 3) = D / (v + 3) + 0.5) → 
  upstream_distance 33.2299691632954 = 91.25 :=
by
  sorry

end boat_travel_distance_upstream_l179_179084


namespace largest_among_options_l179_179046

theorem largest_among_options (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > (1/2) ∧ b > a^2 + b^2 ∧ b > 2*a*b := 
by
  sorry

end largest_among_options_l179_179046


namespace probability_point_in_square_l179_179088

theorem probability_point_in_square (r : ℝ) (hr : 0 < r) :
  (∃ p : ℝ, p = 2 / Real.pi) :=
by
  sorry

end probability_point_in_square_l179_179088


namespace investment_months_l179_179608

theorem investment_months (i_a i_b i_c a_gain total_gain : ℝ) (m : ℝ) :
  i_a = 1 ∧ i_b = 2 * i_a ∧ i_c = 3 * i_a ∧ a_gain = 6100 ∧ total_gain = 18300 ∧ m * i_b * (12 - m) + i_c * 3 * 4 = 12200 →
  a_gain / total_gain = i_a * 12 / (i_a * 12 + i_b * (12 - m) + i_c * 4) → m = 6 :=
by
  intros h1 h2
  obtain ⟨ha, hb, hc, hag, htg, h⟩ := h1
  -- proof omitted
  sorry

end investment_months_l179_179608


namespace parabolas_intersect_l179_179138

theorem parabolas_intersect :
  let eq1 (x : ℝ) := 3 * x^2 - 4 * x + 2
  let eq2 (x : ℝ) := -x^2 + 6 * x + 8
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = -0.5 ∧ y = 4.75) ∧
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = 3 ∧ y = 17) :=
by sorry

end parabolas_intersect_l179_179138


namespace flowers_sold_l179_179004

theorem flowers_sold (lilacs roses gardenias : ℕ) 
  (h1 : lilacs = 10)
  (h2 : roses = 3 * lilacs)
  (h3 : gardenias = lilacs / 2) : 
  lilacs + roses + gardenias = 45 :=
by
  sorry

end flowers_sold_l179_179004


namespace cannot_sum_85_with_five_coins_l179_179953

def coin_value (c : Nat) : Prop :=
  c = 1 ∨ c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50

theorem cannot_sum_85_with_five_coins : 
  ¬ ∃ (a b c d e : Nat), 
    coin_value a ∧ 
    coin_value b ∧ 
    coin_value c ∧ 
    coin_value d ∧ 
    coin_value e ∧ 
    a + b + c + d + e = 85 :=
by
  sorry

end cannot_sum_85_with_five_coins_l179_179953


namespace depth_of_ship_l179_179013

-- Condition definitions
def rate : ℝ := 80  -- feet per minute
def time : ℝ := 50  -- minutes

-- Problem Statement
theorem depth_of_ship : rate * time = 4000 :=
by
  sorry

end depth_of_ship_l179_179013


namespace find_xiao_li_compensation_l179_179296

-- Define the conditions
variable (total_days : ℕ) (extra_days : ℕ) (extra_compensation : ℕ)
variable (daily_work : ℕ) (daily_reward : ℕ) (xiao_li_days : ℕ)

-- Define the total compensation for Xiao Li
def xiao_li_compensation (xiao_li_days daily_reward : ℕ) : ℕ := xiao_li_days * daily_reward

-- The theorem statement asserting the final answer
theorem find_xiao_li_compensation
  (h1 : total_days = 12)
  (h2 : extra_days = 3)
  (h3 : extra_compensation = 2700)
  (h4 : daily_work = 1)
  (h5 : daily_reward = 225)
  (h6 : xiao_li_days = 2)
  (h7 : (total_days - extra_days) * daily_work = xiao_li_days * daily_work):
  xiao_li_compensation xiao_li_days daily_reward = 450 := 
sorry

end find_xiao_li_compensation_l179_179296


namespace gcd_204_85_l179_179513

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  have h1 : 204 = 2 * 85 + 34 := by rfl
  have h2 : 85 = 2 * 34 + 17 := by rfl
  have h3 : 34 = 2 * 17 := by rfl
  sorry

end gcd_204_85_l179_179513


namespace range_of_a_l179_179340

theorem range_of_a (x a : ℝ) (p : |x - 2| < 3) (q : 0 < x ∧ x < a) :
  (0 < a ∧ a ≤ 5) := 
sorry

end range_of_a_l179_179340


namespace cards_probability_l179_179812

-- Definitions based on conditions
def total_cards := 52
def suits := 4
def cards_per_suit := 13

-- Introducing probabilities for the conditions mentioned
def prob_first := 1
def prob_second := 39 / 52
def prob_third := 26 / 52
def prob_fourth := 13 / 52
def prob_fifth := 26 / 52

-- The problem statement
theorem cards_probability :
  (prob_first * prob_second * prob_third * prob_fourth * prob_fifth) = (3 / 64) :=
by
  sorry

end cards_probability_l179_179812


namespace percent_defective_units_shipped_for_sale_l179_179433

theorem percent_defective_units_shipped_for_sale 
  (P : ℝ) -- total number of units produced
  (h_defective : 0.06 * P = d) -- 6 percent of units are defective
  (h_shipped : 0.0024 * P = s) -- 0.24 percent of units are defective units shipped for sale
  : (s / d) * 100 = 4 :=
by
  sorry

end percent_defective_units_shipped_for_sale_l179_179433


namespace track_team_children_l179_179211

/-- There were initially 18 girls and 15 boys on the track team.
    7 more girls joined the team, and 4 boys quit the team.
    The proof shows that the total number of children on the track team after the changes is 36. -/
theorem track_team_children (initial_girls initial_boys girls_joined boys_quit : ℕ)
  (h_initial_girls : initial_girls = 18)
  (h_initial_boys : initial_boys = 15)
  (h_girls_joined : girls_joined = 7)
  (h_boys_quit : boys_quit = 4) :
  initial_girls + girls_joined - boys_quit + initial_boys = 36 :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end track_team_children_l179_179211


namespace part1_x1_part1_x0_part1_xneg2_general_inequality_l179_179904

-- Prove inequality for specific values of x
theorem part1_x1 : - (1/2 : ℝ) * (1: ℝ)^2 + 2 * (1: ℝ) < -(1: ℝ) + 5 := by
  sorry

theorem part1_x0 : - (1/2 : ℝ) * (0: ℝ)^2 + 2 * (0: ℝ) < -(0: ℝ) + 5 := by
  sorry

theorem part1_xneg2 : - (1/2 : ℝ) * (-2: ℝ)^2 + 2 * (-2: ℝ) < -(-2: ℝ) + 5 := by
  sorry

-- Prove general inequality for all real x
theorem general_inequality (x : ℝ) : - (1/2 : ℝ) * x^2 + 2 * x < -x + 5 := by
  sorry

end part1_x1_part1_x0_part1_xneg2_general_inequality_l179_179904


namespace triangle_area_l179_179257

theorem triangle_area (A B C : ℝ) (AB BC CA : ℝ) (sinA sinB sinC : ℝ)
    (h1 : sinA * sinB * sinC = 1 / 1000) 
    (h2 : AB * BC * CA = 1000) : 
    (AB * BC * CA / (4 * 50)) = 5 :=
by
  -- Proof is omitted
  sorry

end triangle_area_l179_179257


namespace shorten_other_side_area_l179_179823

-- Assuming initial dimensions and given conditions
variable (length1 length2 : ℕ)
variable (new_length : ℕ)
variable (area1 area2 : ℕ)

-- Initial dimensions of the index card
def initial_dimensions (length1 length2 : ℕ) : Prop :=
  length1 = 3 ∧ length2 = 7

-- Area when one side is shortened to a specific new length
def shortened_area (length1 length2 new_length : ℕ) : ℕ :=
  if new_length = length1 - 1 then new_length * length2 else length1 * (length2 - 1)

-- Condition that the area is 15 square inches when one side is shortened
def condition_area_15 (length1 length2 : ℕ) : Prop :=
  (shortened_area length1 length2 (length1 - 1) = 15 ∨
   shortened_area length1 length2 (length2 - 1) = 15)

-- Area when the other side is shortened by 1 inch
def new_area (length1 new_length : ℕ) : ℕ :=
  new_length * (length1 - 1)

-- Proving the final area when the other side is shortened
theorem shorten_other_side_area :
  initial_dimensions length1 length2 →
  condition_area_15 length1 length2 →
  new_area length2 (length2 - 1) = 10 :=
by
  intros hdim hc15
  have hlength1 : length1 = 3 := hdim.1
  have hlength2 : length2 = 7 := hdim.2
  sorry

end shorten_other_side_area_l179_179823


namespace quadratic_real_roots_opposite_signs_l179_179384

theorem quadratic_real_roots_opposite_signs (c : ℝ) : 
  (c < 0 → (∃ x1 x2 : ℝ, x1 * x2 = c ∧ x1 + x2 = -1 ∧ x1 ≠ x2 ∧ (x1 < 0 ∧ x2 > 0 ∨ x1 > 0 ∧ x2 < 0))) ∧ 
  (∃ x1 x2 : ℝ, x1 * x2 = c ∧ x1 + x2 = -1 ∧ x1 ≠ x2 ∧ (x1 < 0 ∧ x2 > 0 ∨ x1 > 0 ∧ x2 < 0) → c < 0) :=
by 
  sorry

end quadratic_real_roots_opposite_signs_l179_179384


namespace find_horses_l179_179986

theorem find_horses {x : ℕ} :
  (841 : ℝ) = 8 * (x : ℝ) + 16 * 9 + 18 * 6 → 348 = 16 * 9 →
  x = 73 :=
by
  intros h₁ h₂
  sorry

end find_horses_l179_179986


namespace expand_product_l179_179795

theorem expand_product (y : ℝ) (h : y ≠ 0) : 
  (3 / 7) * ((7 / y) - 14 * y^3 + 21) = (3 / y) - 6 * y^3 + 9 := 
by 
  sorry

end expand_product_l179_179795


namespace find_beta_l179_179648

variables {m n p : ℤ} -- defining variables m, n, p as integers
variables {α β : ℤ} -- defining roots α and β as integers

theorem find_beta (h1: α = 3)
  (h2: ∀ x, x^2 - (m+n)*x + (m*n - p) = 0) -- defining the quadratic equation
  (h3: α + β = m + n)
  (h4: α * β = m * n - p)
  (h5: m ≠ n) (h6: n ≠ p) (h7: m ≠ p) : -- ensuring m, n, and p are distinct
  β = m + n - 3 := sorry

end find_beta_l179_179648


namespace units_digit_p_plus_2_l179_179671

theorem units_digit_p_plus_2 {p : ℕ} 
  (h1 : p % 2 = 0) 
  (h2 : p % 10 ≠ 0) 
  (h3 : (p^3 % 10) = (p^2 % 10)) : 
  (p + 2) % 10 = 8 :=
sorry

end units_digit_p_plus_2_l179_179671


namespace intersection_A_B_l179_179628

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 8}
def B := {x : ℝ | x^2 - 3 * x - 4 < 0}
def expected := {x : ℝ | 2 ≤ x ∧ x < 4 }

theorem intersection_A_B : (A ∩ B) = expected := 
by 
  sorry

end intersection_A_B_l179_179628


namespace transformed_line_theorem_l179_179095

theorem transformed_line_theorem (k b : ℝ) (h₁ : k = 1) (h₂ : b = 1) (x : ℝ) :
  (k * x + b > 0) ↔ (x > -1) :=
by sorry

end transformed_line_theorem_l179_179095


namespace smallest_number_meeting_both_conditions_l179_179342

theorem smallest_number_meeting_both_conditions :
  ∃ n, (n = 2019) ∧
    (∃ a b c d e f : ℕ,
      n = a^4 + b^4 + c^4 + d^4 + e^4 ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
      c ≠ d ∧ c ≠ e ∧
      d ≠ e ∧
      a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
    (∃ x y z u v w : ℕ,
      y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
      n = x + y + z + u + v + w) ∧
    (¬ ∃ m, m < 2019 ∧
      (∃ a b c d e f : ℕ,
        m = a^4 + b^4 + c^4 + d^4 + e^4 ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
        c ≠ d ∧ c ≠ e ∧
        d ≠ e ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
      (∃ x y z u v w : ℕ,
        y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
        m = x + y + z + u + v + w)) :=
by
  sorry

end smallest_number_meeting_both_conditions_l179_179342


namespace most_balls_l179_179952

def soccerballs : ℕ := 50
def basketballs : ℕ := 26
def baseballs : ℕ := basketballs + 8

theorem most_balls :
  max (max soccerballs basketballs) baseballs = soccerballs := by
  sorry

end most_balls_l179_179952


namespace isosceles_triangle_sides_l179_179266

theorem isosceles_triangle_sides (a b c : ℝ) (hb : b = 3) (hc : a = 3 ∨ c = 3) (hperim : a + b + c = 7) :
  a = 2 ∨ a = 3 ∨ c = 2 ∨ c = 3 :=
by
  sorry

end isosceles_triangle_sides_l179_179266


namespace smallest_c_inv_l179_179632

def f (x : ℝ) : ℝ := (x + 3)^2 - 7

theorem smallest_c_inv (c : ℝ) : (∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) →
  c = -3 :=
sorry

end smallest_c_inv_l179_179632


namespace grasshopper_catched_in_finite_time_l179_179990

theorem grasshopper_catched_in_finite_time :
  ∀ (x0 y0 x1 y1 : ℤ),
  ∃ (T : ℕ), ∃ (x y : ℤ), 
  ((x = x0 + x1 * T) ∧ (y = y0 + y1 * T)) ∧ -- The hunter will catch the grasshopper at this point
  ((∀ t : ℕ, t ≤ T → (x ≠ x0 + x1 * t ∨ y ≠ y0 + y1 * t) → (x = x0 + x1 * t ∧ y = y0 + y1 * t))) :=
sorry

end grasshopper_catched_in_finite_time_l179_179990


namespace alice_sales_surplus_l179_179730

-- Define the constants
def adidas_cost : ℕ := 45
def nike_cost : ℕ := 60
def reebok_cost : ℕ := 35
def quota : ℕ := 1000

-- Define the quantities sold
def adidas_sold : ℕ := 6
def nike_sold : ℕ := 8
def reebok_sold : ℕ := 9

-- Calculate total sales
def total_sales : ℕ := adidas_sold * adidas_cost + nike_sold * nike_cost + reebok_sold * reebok_cost

-- Prove that Alice's total sales minus her quota is 65
theorem alice_sales_surplus : total_sales - quota = 65 := by
  -- Calculation is omitted here. Here is the mathematical fact to prove:
  sorry

end alice_sales_surplus_l179_179730


namespace dog_years_second_year_l179_179298

theorem dog_years_second_year (human_years : ℕ) :
  15 + human_years + 5 * 8 = 64 →
  human_years = 9 :=
by
  intro h
  sorry

end dog_years_second_year_l179_179298


namespace sufficient_and_necessary_condition_l179_179713

theorem sufficient_and_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, m * x ^ 2 + 2 * m * x - 1 < 0) ↔ (-1 < m ∧ m < -1 / 2) :=
by
  sorry

end sufficient_and_necessary_condition_l179_179713


namespace remainder_form_l179_179017

open Polynomial Int

-- Define the conditions
variable (f : Polynomial ℤ)
variable (h1 : ∀ n : ℤ, 3 ∣ eval n f)

-- Define the proof problem statement
theorem remainder_form (h1 : ∀ n : ℤ, 3 ∣ eval n f) :
  ∃ (M r : Polynomial ℤ), f = (X^3 - X) * M + C 3 * r :=
sorry

end remainder_form_l179_179017


namespace find_box_value_l179_179946

theorem find_box_value (r x : ℕ) 
  (h1 : x + r = 75)
  (h2 : (x + r) + 2 * r = 143) : 
  x = 41 := 
by
  sorry

end find_box_value_l179_179946


namespace prove_total_rent_of_field_l179_179361

def totalRentField (A_cows A_months B_cows B_months C_cows C_months 
                    D_cows D_months E_cows E_months F_cows F_months 
                    G_cows G_months A_rent : ℕ) : ℕ := 
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let E_cow_months := E_cows * E_months
  let F_cow_months := F_cows * F_months
  let G_cow_months := G_cows * G_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + 
                          D_cow_months + E_cow_months + F_cow_months + G_cow_months
  let rent_per_cow_month := A_rent / A_cow_months
  total_cow_months * rent_per_cow_month

theorem prove_total_rent_of_field : totalRentField 24 3 10 5 35 4 21 3 15 6 40 2 28 (7/2) 720 = 5930 :=
  by
  sorry

end prove_total_rent_of_field_l179_179361


namespace unique_four_digit_number_l179_179760

theorem unique_four_digit_number (a b c d : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) (hd : d ≤ 9)
  (h1 : a + b = c + d)
  (h2 : b + d = 2 * (a + c))
  (h3 : a + d = c)
  (h4 : b + c - a = 3 * d) :
  a = 1 ∧ b = 8 ∧ c = 5 ∧ d = 4 :=
by
  sorry

end unique_four_digit_number_l179_179760


namespace rectangle_equation_l179_179789

-- Given points in the problem, we define the coordinates
def A : ℝ × ℝ := (5, 5)
def B : ℝ × ℝ := (9, 2)
def C (a : ℝ) : ℝ × ℝ := (a, 13)
def D (b : ℝ) : ℝ × ℝ := (15, b)

-- We need to prove that a - b = 1 given the conditions
theorem rectangle_equation (a b : ℝ) (h1 : C a = (a, 13)) (h2 : D b = (15, b)) (h3 : 15 - a = 4) (h4 : 13 - b = 3) : 
     a - b = 1 := 
sorry

end rectangle_equation_l179_179789


namespace line_parabola_intersection_l179_179347

theorem line_parabola_intersection (k : ℝ) :
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → y = 1 ∧ x = 1 / 4) ∨
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → (k^2 * x^2 + (2 * k - 4) * x + 1 = 0) ∧ (4 * k * k - 16 * k + 16 - 4 * k * k = 0) → k = 1) :=
sorry

end line_parabola_intersection_l179_179347


namespace total_cost_of_replacing_floor_l179_179565

-- Dimensions of the first rectangular section
def length1 : ℕ := 8
def width1 : ℕ := 7

-- Dimensions of the second rectangular section
def length2 : ℕ := 6
def width2 : ℕ := 4

-- Cost to remove the old flooring
def cost_removal : ℕ := 50

-- Cost of new flooring per square foot
def cost_per_sqft : ℝ := 1.25

-- Total cost to replace the floor in both sections of the L-shaped room
theorem total_cost_of_replacing_floor 
  (A1 : ℕ := length1 * width1)
  (A2 : ℕ := length2 * width2)
  (total_area : ℕ := A1 + A2)
  (cost_flooring : ℝ := total_area * cost_per_sqft)
  : cost_removal + cost_flooring = 150 :=
sorry

end total_cost_of_replacing_floor_l179_179565


namespace mimi_shells_l179_179721

theorem mimi_shells (Kyle_shells Mimi_shells Leigh_shells : ℕ) 
  (h₀ : Kyle_shells = 2 * Mimi_shells) 
  (h₁ : Leigh_shells = Kyle_shells / 3) 
  (h₂ : Leigh_shells = 16) 
  : Mimi_shells = 24 := by 
  sorry

end mimi_shells_l179_179721


namespace probability_of_rolling_two_exactly_four_times_in_five_rolls_l179_179922

theorem probability_of_rolling_two_exactly_four_times_in_five_rolls :
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n-k)
  probability = (25 / 7776) :=
by
  let p := (1 / 6)
  let q := (5 / 6)
  let n := 5
  let k := 4
  let probability := (n.choose k) * p^k * q^(n - k)
  have h : probability = (25 / 7776) := sorry
  exact h

end probability_of_rolling_two_exactly_four_times_in_five_rolls_l179_179922


namespace parts_supplier_total_amount_received_l179_179576

noncomputable def total_amount_received (total_packages: ℕ) (price_per_package: ℚ) (discount_factor: ℚ)
  (X_percentage: ℚ) (Y_percentage: ℚ) : ℚ :=
  let X_packages := X_percentage * total_packages
  let Y_packages := Y_percentage * total_packages
  let Z_packages := total_packages - X_packages - Y_packages
  let discounted_price := discount_factor * price_per_package
  let cost_X := X_packages * price_per_package
  let cost_Y := Y_packages * price_per_package
  let cost_Z := 10 * price_per_package + (Z_packages - 10) * discounted_price
  cost_X + cost_Y + cost_Z

-- Given conditions
def total_packages : ℕ := 60
def price_per_package : ℚ := 20
def discount_factor : ℚ := 4 / 5
def X_percentage : ℚ := 0.20
def Y_percentage : ℚ := 0.15

theorem parts_supplier_total_amount_received :
  total_amount_received total_packages price_per_package discount_factor X_percentage Y_percentage = 1084 := 
by 
  -- Here we need the proof, but we put sorry to skip it as per instructions
  sorry

end parts_supplier_total_amount_received_l179_179576


namespace pencils_in_all_l179_179891

/-- Eugene's initial number of pencils -/
def initial_pencils : ℕ := 51

/-- Pencils Eugene gets from Joyce -/
def additional_pencils : ℕ := 6

/-- Total number of pencils Eugene has in all -/
def total_pencils : ℕ :=
  initial_pencils + additional_pencils

/-- Proof that Eugene has 57 pencils in all -/
theorem pencils_in_all : total_pencils = 57 := by
  sorry

end pencils_in_all_l179_179891


namespace no_valid_2011_matrix_l179_179818

def valid_matrix (A : ℕ → ℕ → ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 2011 →
    (∀ k, 1 ≤ k ∧ k ≤ 4021 →
      (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A i j = k) ∨ (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A j i = k))

theorem no_valid_2011_matrix :
  ¬ ∃ A : ℕ → ℕ → ℕ, (∀ i j, 1 ≤ i ∧ i ≤ 2011 ∧ 1 ≤ j ∧ j ≤ 2011 → 1 ≤ A i j ∧ A i j ≤ 4021) ∧ valid_matrix A :=
by
  sorry

end no_valid_2011_matrix_l179_179818


namespace calculate_expression_l179_179765

-- Define the expression x + x * (factorial x)^x
def expression (x : ℕ) : ℕ :=
  x + x * (Nat.factorial x) ^ x

-- Set the value of x
def x_value : ℕ := 3

-- State the proposition
theorem calculate_expression : expression x_value = 651 := 
by 
  -- By substitution and calculation, the proof follows.
  sorry

end calculate_expression_l179_179765


namespace opposite_number_113_is_114_l179_179790

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l179_179790


namespace calculate_arithmetic_expression_l179_179525

noncomputable def arithmetic_sum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem calculate_arithmetic_expression :
  3 * (arithmetic_sum 71 2 99) = 3825 :=
by
  sorry

end calculate_arithmetic_expression_l179_179525


namespace complex_pow_simplify_l179_179764

noncomputable def i : ℂ := Complex.I

theorem complex_pow_simplify :
  (1 + Real.sqrt 3 * Complex.I) ^ 3 * Complex.I = -8 * Complex.I :=
by
  sorry

end complex_pow_simplify_l179_179764


namespace part1_monotonicity_when_a_eq_1_part2_range_of_a_l179_179698

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * (Real.log (x - 2)) - a * (x - 3)

theorem part1_monotonicity_when_a_eq_1 :
  ∀ x, 2 < x → ∀ x1, (2 < x1 → f x 1 ≤ f x1 1) := by
  sorry

theorem part2_range_of_a :
  ∀ a, (∀ x, 3 < x → f x a > 0) → a ≤ 2 := by
  sorry

end part1_monotonicity_when_a_eq_1_part2_range_of_a_l179_179698


namespace tv_weight_calculations_l179_179526

theorem tv_weight_calculations
    (w1 h1 r1 : ℕ) -- Represents Bill's TV dimensions and weight ratio
    (w2 h2 r2 : ℕ) -- Represents Bob's TV dimensions and weight ratio
    (w3 h3 r3 : ℕ) -- Represents Steve's TV dimensions and weight ratio
    (ounce_to_pound: ℕ) -- Represents the conversion factor from ounces to pounds
    (bill_tv_weight bob_tv_weight steve_tv_weight : ℕ) -- Computed weights in pounds
    (weight_diff: ℕ):
  (w1 * h1 * r1) / ounce_to_pound = bill_tv_weight → -- Bill's TV weight calculation
  (w2 * h2 * r2) / ounce_to_pound = bob_tv_weight → -- Bob's TV weight calculation
  (w3 * h3 * r3) / ounce_to_pound = steve_tv_weight → -- Steve's TV weight calculation
  steve_tv_weight > (bill_tv_weight + bob_tv_weight) → -- Steve's TV is the heaviest
  steve_tv_weight - (bill_tv_weight + bob_tv_weight) = weight_diff → -- weight difference calculation
  True := sorry

end tv_weight_calculations_l179_179526


namespace Iggy_miles_on_Monday_l179_179410

theorem Iggy_miles_on_Monday 
  (tuesday_miles : ℕ)
  (wednesday_miles : ℕ)
  (thursday_miles : ℕ)
  (friday_miles : ℕ)
  (monday_minutes : ℕ)
  (pace : ℕ)
  (total_hours : ℕ)
  (total_minutes : ℕ)
  (total_tuesday_to_friday_miles : ℕ)
  (total_tuesday_to_friday_minutes : ℕ) :
  tuesday_miles = 4 →
  wednesday_miles = 6 →
  thursday_miles = 8 →
  friday_miles = 3 →
  pace = 10 →
  total_hours = 4 →
  total_minutes = total_hours * 60 →
  total_tuesday_to_friday_miles = tuesday_miles + wednesday_miles + thursday_miles + friday_miles →
  total_tuesday_to_friday_minutes = total_tuesday_to_friday_miles * pace →
  monday_minutes = total_minutes - total_tuesday_to_friday_minutes →
  (monday_minutes / pace) = 3 := sorry

end Iggy_miles_on_Monday_l179_179410


namespace smallest_positive_b_factors_l179_179165

theorem smallest_positive_b_factors (b : ℤ) : 
  (∃ p q : ℤ, x^2 + b * x + 2016 = (x + p) * (x + q) ∧ p + q = b ∧ p * q = 2016 ∧ p > 0 ∧ q > 0) → b = 95 := 
by {
  sorry
}

end smallest_positive_b_factors_l179_179165


namespace distance_between_lines_l179_179417

/-- The graph of the function y = x^2 + ax + b is drawn on a board.
Let the parabola intersect the horizontal lines y = s and y = t at points A, B and C, D respectively,
with A B = 5 and C D = 11. Then the distance between the lines y = s and y = t is 24. -/
theorem distance_between_lines 
  (a b s t : ℝ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + a * x1 + b = s) ∧ (x2^2 + a * x2 + b = s) ∧ |x1 - x2| = 5)
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ (x3^2 + a * x3 + b = t) ∧ (x4^2 + a * x4 + b = t) ∧ |x3 - x4| = 11) :
  |t - s| = 24 := 
by
  sorry

end distance_between_lines_l179_179417


namespace three_sleep_simultaneously_l179_179247

noncomputable def professors := Finset.range 5

def sleeping_times (p: professors) : Finset ℕ 
-- definition to be filled in, stating that p falls asleep twice.
:= sorry 

def moment_two_asleep (p q: professors) : ℕ 
-- definition to be filled in, stating that p and q are asleep together once.
:= sorry

theorem three_sleep_simultaneously :
  ∃ t : ℕ, ∃ p1 p2 p3 : professors, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
  (t ∈ sleeping_times p1) ∧
  (t ∈ sleeping_times p2) ∧
  (t ∈ sleeping_times p3) := by
  sorry

end three_sleep_simultaneously_l179_179247


namespace books_of_jason_l179_179393

theorem books_of_jason (M J : ℕ) (hM : M = 42) (hTotal : M + J = 60) : J = 18 :=
by
  sorry

end books_of_jason_l179_179393


namespace find_positive_real_numbers_l179_179688

open Real

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16

theorem find_positive_real_numbers (x : ℝ) (hx : x > 0) :
  satisfies_inequality x ↔ 15 * x^2 + 32 * x - 256 = 0 :=
sorry

end find_positive_real_numbers_l179_179688


namespace probability_of_matching_pair_l179_179360

noncomputable def num_socks := 22
noncomputable def red_socks := 12
noncomputable def blue_socks := 10

def ways_to_choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

noncomputable def probability_same_color : ℚ :=
  (ways_to_choose_two red_socks + ways_to_choose_two blue_socks : ℚ) / ways_to_choose_two num_socks

theorem probability_of_matching_pair :
  probability_same_color = 37 / 77 := 
by
  -- proof goes here
  sorry

end probability_of_matching_pair_l179_179360


namespace serving_calculation_correct_l179_179779

def prepared_orange_juice_servings (cans_of_concentrate : ℕ) 
                                  (oz_per_concentrate_can : ℕ) 
                                  (water_ratio : ℕ) 
                                  (oz_per_serving : ℕ) : ℕ :=
  let total_concentrate := cans_of_concentrate * oz_per_concentrate_can
  let total_water := cans_of_concentrate * water_ratio * oz_per_concentrate_can
  let total_juice := total_concentrate + total_water
  total_juice / oz_per_serving

theorem serving_calculation_correct :
  prepared_orange_juice_servings 60 5 3 6 = 200 := by
  sorry

end serving_calculation_correct_l179_179779


namespace cannot_determine_right_triangle_l179_179700

-- Definitions of conditions
def condition_A (A B C : ℝ) : Prop := A = B + C
def condition_B (a b c : ℝ) : Prop := a/b = 5/12 ∧ b/c = 12/13
def condition_C (a b c : ℝ) : Prop := a^2 = (b + c) * (b - c)
def condition_D (A B C : ℝ) : Prop := A/B = 3/4 ∧ B/C = 4/5

-- The proof problem
theorem cannot_determine_right_triangle (a b c A B C : ℝ)
  (hD : condition_D A B C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end cannot_determine_right_triangle_l179_179700


namespace no_first_or_fourth_quadrant_l179_179932

theorem no_first_or_fourth_quadrant (a b : ℝ) (h : a * b > 0) : 
  ¬ ((∃ x, a * x + b = 0 ∧ x > 0) ∧ (∃ x, b * x + a = 0 ∧ x > 0)) 
  ∧ ¬ ((∃ x, a * x + b = 0 ∧ x < 0) ∧ (∃ x, b * x + a = 0 ∧ x < 0)) := sorry

end no_first_or_fourth_quadrant_l179_179932


namespace min_value_S_l179_179973

theorem min_value_S (a b c : ℤ) (h1 : a + b + c = 2) (h2 : (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) > 200) :
  ∃ a b c : ℤ, a + b + c = 2 ∧ (2 * a + b * c) * (2 * b + c * a) * (2 * c + a * b) = 256 :=
sorry

end min_value_S_l179_179973


namespace annies_classmates_count_l179_179895

theorem annies_classmates_count (spent : ℝ) (cost_per_candy : ℝ) (candies_left : ℕ) (candies_per_classmate : ℕ) (expected_classmates : ℕ):
  spent = 8 ∧ cost_per_candy = 0.1 ∧ candies_left = 12 ∧ candies_per_classmate = 2 ∧ expected_classmates = 34 →
  (spent / cost_per_candy) - candies_left = (expected_classmates * candies_per_classmate) := 
by
  intros h
  sorry

end annies_classmates_count_l179_179895


namespace total_games_l179_179610

theorem total_games (N : ℕ) (p : ℕ)
  (hPetya : 2 ∣ N)
  (hKolya : 3 ∣ N)
  (hVasya : 5 ∣ N)
  (hGamesNotInvolving : 2 ≤ N - (N / 2 + N / 3 + N / 5)) :
  N = 30 :=
by
  sorry

end total_games_l179_179610


namespace find_grade_C_boxes_l179_179710

theorem find_grade_C_boxes (m n t : ℕ) (h : 2 * t = m + n) (total_boxes : ℕ) (h_total : total_boxes = 420) : t = 140 :=
by
  sorry

end find_grade_C_boxes_l179_179710


namespace solve_for_x_l179_179209

theorem solve_for_x (x : ℝ) : (5 * x - 2) / (6 * x - 6) = 3 / 4 ↔ x = -5 := by
  sorry

end solve_for_x_l179_179209


namespace find_number_l179_179214

theorem find_number (x : ℕ) (h : 3 * x = 2 * 51 - 3) : x = 33 :=
by
  sorry

end find_number_l179_179214


namespace problem_statement_l179_179768

theorem problem_statement (a b c : ℝ) (h : a^2 + b^2 - a * b = c^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a - c) * (b - c) ≤ 0 :=
by sorry

end problem_statement_l179_179768


namespace poly_coeff_sum_l179_179129

variable {a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ}

theorem poly_coeff_sum :
  (∀ x : ℝ, (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 + 6 * a_6 = 12 :=
by
  sorry

end poly_coeff_sum_l179_179129


namespace nat_pair_solution_l179_179075

theorem nat_pair_solution (x y : ℕ) : 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by
  sorry

end nat_pair_solution_l179_179075


namespace angle_A_is_pi_over_3_l179_179957

theorem angle_A_is_pi_over_3 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C)
  (h2 : a ^ 2 = b ^ 2 + c ^ 2 - bc * (2 * Real.cos A))
  (triangle_ABC : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ A + B + C = π) :
  A = π / 3 :=
by
  sorry

end angle_A_is_pi_over_3_l179_179957


namespace max_value_fourth_power_l179_179220

theorem max_value_fourth_power (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) : 
  a^4 + b^4 + c^4 + d^4 ≤ 4^(4/3) :=
sorry

end max_value_fourth_power_l179_179220


namespace swimming_championship_l179_179734

theorem swimming_championship (num_swimmers : ℕ) (lanes : ℕ) (advance : ℕ) (eliminated : ℕ) (total_races : ℕ) : 
  num_swimmers = 300 → 
  lanes = 8 → 
  advance = 2 → 
  eliminated = 6 → 
  total_races = 53 :=
by
  intros
  sorry

end swimming_championship_l179_179734


namespace part_I_part_II_l179_179890

noncomputable def f (x a : ℝ) : ℝ := |x - a| - |2 * x - 1|

theorem part_I (a : ℝ) (x : ℝ) (h : a = 2) :
    f x a + 3 ≥ 0 ↔ -4 ≤ x ∧ x ≤ 2 := 
by
    -- problem restatement
    sorry

theorem part_II (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 3) :
    -3 ≤ a ∧ a ≤ 5 := 
by
    -- problem restatement
    sorry

end part_I_part_II_l179_179890


namespace geom_seq_sum_l179_179278

theorem geom_seq_sum (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 3 + a 5 = 21)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 3 + a 5 + a 7 = 42 :=
sorry

end geom_seq_sum_l179_179278


namespace price_increase_solution_l179_179260

variable (x : ℕ)

def initial_profit := 10
def initial_sales := 500
def price_increase_effect := 20
def desired_profit := 6000

theorem price_increase_solution :
  ((initial_sales - price_increase_effect * x) * (initial_profit + x) = desired_profit) → (x = 5) :=
by
  sorry

end price_increase_solution_l179_179260


namespace bell_rings_count_l179_179633

-- Defining the conditions
def bell_rings_per_class : ℕ := 2
def total_classes_before_music : ℕ := 4
def bell_rings_during_music_start : ℕ := 1

-- The main proof statement
def total_bell_rings : ℕ :=
  total_classes_before_music * bell_rings_per_class + bell_rings_during_music_start

theorem bell_rings_count : total_bell_rings = 9 := by
  sorry

end bell_rings_count_l179_179633


namespace empty_set_negation_l179_179637

open Set

theorem empty_set_negation (α : Type) : ¬ (∀ s : Set α, ∅ ⊆ s) ↔ (∃ s : Set α, ¬(∅ ⊆ s)) :=
by
  sorry

end empty_set_negation_l179_179637


namespace qualifying_rate_l179_179086

theorem qualifying_rate (a b : ℝ) (h1 : 0 ≤ a ∧ a ≤ 1) (h2 : 0 ≤ b ∧ b ≤ 1) :
  (1 - a) * (1 - b) = 1 - a - b + a * b :=
by sorry

end qualifying_rate_l179_179086


namespace part1_tangent_line_at_x1_part2_a_range_l179_179386

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x

theorem part1_tangent_line_at_x1 (a : ℝ) (h1 : a = 1) : 
  let f' (x : ℝ) : ℝ := (x + 1) * Real.exp x - 1
  (2 * Real.exp 1 - 1) * 1 - (f 1 1) = Real.exp 1 :=
by 
  sorry

theorem part2_a_range (a : ℝ) (h2 : ∀ x > 0, f x a ≥ Real.log x - x + 1) : 
  0 < a ∧ a ≤ 2 :=
by 
  sorry

end part1_tangent_line_at_x1_part2_a_range_l179_179386


namespace triangle_area_l179_179543

noncomputable def area_of_triangle (a b c α β γ : ℝ) :=
  (1 / 2) * a * b * Real.sin γ

theorem triangle_area 
  (a b c A B C : ℝ)
  (h1 : b * Real.cos C = 3 * a * Real.cos B - c * Real.cos B)
  (h2 : (a * b * Real.cos C) / (a * b) = 2) :
  area_of_triangle a b c A B C = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_area_l179_179543


namespace point_on_x_axis_l179_179653

theorem point_on_x_axis (m : ℤ) (P : ℤ × ℤ) (hP : P = (m + 3, m + 1)) (h : P.2 = 0) : P = (2, 0) :=
by 
  sorry

end point_on_x_axis_l179_179653


namespace quadratic_eq_two_distinct_real_roots_l179_179593

theorem quadratic_eq_two_distinct_real_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 2 * x₁ + a = 0) ∧ (x₂^2 - 2 * x₂ + a = 0)) ↔ a < 1 :=
by
  sorry

end quadratic_eq_two_distinct_real_roots_l179_179593


namespace solve_problem_l179_179657

def num : ℕ := 1 * 3 * 5 * 7
def den : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7

theorem solve_problem : (num : ℚ) / den = 3.75 := 
by
  sorry

end solve_problem_l179_179657


namespace cost_price_correct_l179_179420

noncomputable def cost_price_per_meter (selling_price_per_meter : ℝ) (total_meters : ℝ) (loss_per_meter : ℝ) :=
  (selling_price_per_meter * total_meters + loss_per_meter * total_meters) / total_meters

theorem cost_price_correct :
  cost_price_per_meter 18000 500 5 = 41 :=
by 
  sorry

end cost_price_correct_l179_179420


namespace cos_comp_l179_179319

open Real

theorem cos_comp {a b c : ℝ} (h1 : a = cos (3 / 2)) (h2 : b = -cos (7 / 4)) (h3 : c = sin (1 / 10)) : 
  a < c ∧ c < b := 
by
  -- Assume the hypotheses
  sorry

end cos_comp_l179_179319


namespace remainder_div_13_l179_179405

theorem remainder_div_13 {N : ℕ} (k : ℕ) (h : N = 39 * k + 18) : N % 13 = 5 := sorry

end remainder_div_13_l179_179405


namespace triangle_inequality_l179_179785

-- Define the triangle angles, semiperimeter, and circumcircle radius
variables (α β γ s R : Real)

-- Define the sum of angles in a triangle
axiom angle_sum : α + β + γ = Real.pi

-- The inequality to prove
theorem triangle_inequality (h_sum : α + β + γ = Real.pi) :
  (α + β) * (β + γ) * (γ + α) ≤ 4 * (Real.pi / Real.sqrt 3)^3 * R / s := sorry

end triangle_inequality_l179_179785


namespace total_outfits_l179_179018

-- Define the quantities of each item.
def red_shirts : ℕ := 7
def green_shirts : ℕ := 8
def pants : ℕ := 10
def blue_hats : ℕ := 10
def red_hats : ℕ := 10
def scarves : ℕ := 5

-- The total number of outfits without having the same color of shirts and hats.
theorem total_outfits : 
  (red_shirts * pants * blue_hats * scarves) + (green_shirts * pants * red_hats * scarves) = 7500 := 
by sorry

end total_outfits_l179_179018


namespace relationship_between_M_and_P_l179_179000

def M := {y : ℝ | ∃ x : ℝ, y = x^2 - 4}
def P := {x : ℝ | 2 ≤ x ∧ x ≤ 4}

theorem relationship_between_M_and_P : ∀ y ∈ {y : ℝ | ∃ x ∈ P, y = x^2 - 4}, y ∈ M :=
by
  sorry

end relationship_between_M_and_P_l179_179000


namespace no_three_digit_number_such_that_sum_is_perfect_square_l179_179714

theorem no_three_digit_number_such_that_sum_is_perfect_square :
  ∀ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 →
  ¬ (∃ m : ℕ, m * m = 100 * a + 10 * b + c + 100 * b + 10 * c + a + 100 * c + 10 * a + b) := by
  sorry

end no_three_digit_number_such_that_sum_is_perfect_square_l179_179714


namespace sum_f_positive_l179_179419

noncomputable def f (x : ℝ) : ℝ := (x ^ 3) / (Real.cos x)

theorem sum_f_positive 
  (x1 x2 x3 : ℝ)
  (hdom1 : abs x1 < Real.pi / 2)
  (hdom2 : abs x2 < Real.pi / 2)
  (hdom3 : abs x3 < Real.pi / 2)
  (hx1x2 : x1 + x2 > 0)
  (hx2x3 : x2 + x3 > 0)
  (hx1x3 : x1 + x3 > 0) :
  f x1 + f x2 + f x3 > 0 :=
sorry

end sum_f_positive_l179_179419


namespace square_distance_from_B_to_center_l179_179283

noncomputable def distance_squared (a b : ℝ) : ℝ := a^2 + b^2

theorem square_distance_from_B_to_center :
  ∀ (a b : ℝ),
    (a^2 + (b + 8)^2 = 75) →
    ((a + 2)^2 + b^2 = 75) →
    distance_squared a b = 122 :=
by
  intros a b h1 h2
  sorry

end square_distance_from_B_to_center_l179_179283


namespace max_possible_value_of_y_l179_179044

theorem max_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = 4) : y ≤ 7 :=
sorry

end max_possible_value_of_y_l179_179044


namespace my_and_mothers_ages_l179_179825

-- Definitions based on conditions
noncomputable def my_age (x : ℕ) := x
noncomputable def mothers_age (x : ℕ) := 3 * x
noncomputable def sum_of_ages (x : ℕ) := my_age x + mothers_age x

-- Proposition that needs to be proved
theorem my_and_mothers_ages (x : ℕ) (h : sum_of_ages x = 40) :
  my_age x = 10 ∧ mothers_age x = 30 :=
by
  sorry

end my_and_mothers_ages_l179_179825


namespace area_of_smaller_part_l179_179536

noncomputable def average (a b : ℝ) : ℝ :=
  (a + b) / 2

theorem area_of_smaller_part:
  ∃ A B : ℝ, A + B = 900 ∧ (B - A) = (1 / 5) * average A B ∧ A = 405 :=
by
  sorry

end area_of_smaller_part_l179_179536


namespace find_correct_value_l179_179656

-- Definitions based on the problem's conditions
def incorrect_calculation (x : ℤ) : Prop := 7 * x = 126
def correct_value (x : ℤ) (y : ℤ) : Prop := x / 6 = y

theorem find_correct_value :
  ∃ (x y : ℤ), incorrect_calculation x ∧ correct_value x y ∧ y = 3 := by
  sorry

end find_correct_value_l179_179656


namespace ratio_m_over_n_l179_179913

theorem ratio_m_over_n : 
  ∀ (m n : ℕ) (a b : ℝ),
  let α := (3 : ℝ) / 4
  let β := (19 : ℝ) / 20
  (a = α * b) →
  (a = β * (a * m + b * n) / (m + n)) →
  (n ≠ 0) →
  m / n = 8 / 9 :=
by
  intros m n a b α β hα hβ hn
  sorry

end ratio_m_over_n_l179_179913


namespace GregPPO_reward_correct_l179_179867

-- Define the maximum ProcGen reward
def maxProcGenReward : ℕ := 240

-- Define the maximum CoinRun reward in the more challenging version
def maxCoinRunReward : ℕ := maxProcGenReward / 2

-- Define the percentage reward obtained by Greg's PPO algorithm
def percentageRewardObtained : ℝ := 0.9

-- Calculate the reward obtained by Greg's PPO algorithm
def rewardGregPPO : ℝ := percentageRewardObtained * maxCoinRunReward

-- The theorem to prove the correct answer
theorem GregPPO_reward_correct : rewardGregPPO = 108 := by
  sorry

end GregPPO_reward_correct_l179_179867


namespace distance_against_stream_l179_179169

variable (vs : ℝ) -- speed of the stream

-- condition: in one hour, the boat goes 9 km along the stream
def cond1 (vs : ℝ) := 7 + vs = 9

-- condition: the speed of the boat in still water (7 km/hr)
def speed_still_water := 7

-- theorem to prove: the distance the boat goes against the stream in one hour
theorem distance_against_stream (vs : ℝ) (h : cond1 vs) : 
  (speed_still_water - vs) * 1 = 5 :=
by
  rw [speed_still_water, mul_one]
  sorry

end distance_against_stream_l179_179169


namespace calculate_exponentiation_l179_179487

theorem calculate_exponentiation : (64^(0.375) * 64^(0.125) = 8) :=
by sorry

end calculate_exponentiation_l179_179487


namespace greatest_savings_by_choosing_boat_l179_179424

/-- Given the transportation costs:
     - plane cost: $600.00
     - boat cost: $254.00
     - helicopter cost: $850.00
    Prove that the greatest amount of money saved by choosing the boat over the other options is $596.00. -/
theorem greatest_savings_by_choosing_boat :
  let plane_cost := 600
  let boat_cost := 254
  let helicopter_cost := 850
  max (plane_cost - boat_cost) (helicopter_cost - boat_cost) = 596 :=
by
  sorry

end greatest_savings_by_choosing_boat_l179_179424


namespace solution_set_of_inequality_l179_179069

theorem solution_set_of_inequality (x : ℝ) :
  x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end solution_set_of_inequality_l179_179069


namespace ratio_equivalence_l179_179282

theorem ratio_equivalence (x : ℝ) :
  ((20 / 10) * 100 = (25 / x) * 100) → x = 12.5 :=
by
  intro h
  sorry

end ratio_equivalence_l179_179282


namespace equivalent_function_l179_179889

theorem equivalent_function :
  (∀ x : ℝ, (76 * x ^ 6) ^ 7 = |x|) :=
by
  sorry

end equivalent_function_l179_179889


namespace ratio_of_square_areas_l179_179174

noncomputable def ratio_of_areas (s : ℝ) : ℝ := s^2 / (4 * s^2)

theorem ratio_of_square_areas (s : ℝ) (h : s ≠ 0) : ratio_of_areas s = 1 / 4 := 
by
  sorry

end ratio_of_square_areas_l179_179174


namespace employee_payment_l179_179580

theorem employee_payment 
    (total_pay : ℕ)
    (pay_A : ℕ)
    (pay_B : ℕ)
    (h1 : total_pay = 560)
    (h2 : pay_A = 3 * pay_B / 2)
    (h3 : pay_A + pay_B = total_pay) :
    pay_B = 224 :=
sorry

end employee_payment_l179_179580


namespace find_length_of_brick_l179_179842

-- Definitions given in the problem
def w : ℕ := 4
def h : ℕ := 2
def SA : ℕ := 112
def surface_area (l w h : ℕ) : ℕ := 2 * l * w + 2 * l * h + 2 * w * h

-- Lean 4 statement for the proof problem
theorem find_length_of_brick (l : ℕ) (h w SA : ℕ) (h_w : w = 4) (h_h : h = 2) (h_SA : SA = 112) :
  surface_area l w h = SA → l = 8 := by
  intros H
  simp [surface_area, h_w, h_h, h_SA] at H
  sorry

end find_length_of_brick_l179_179842


namespace part1_l179_179310

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 3 * a - 10 ≤ x ∧ x < 2 * a + 1}
def Q : Set ℝ := {x | |2 * x - 3| ≤ 7}

-- Define the complement of Q in ℝ
def Q_complement : Set ℝ := {x | x < -2 ∨ x > 5}

-- Define the specific value of a
def a : ℝ := 2

-- Define the specific set P when a = 2
def P_a2 : Set ℝ := {x | -4 ≤ x ∧ x < 5}

-- Define the intersection
def intersection : Set ℝ := {x | -4 ≤ x ∧ x < -2}

theorem part1 : P a ∩ Q_complement = intersection := sorry

end part1_l179_179310


namespace power_six_rectangular_form_l179_179373

noncomputable def sin (x : ℂ) : ℂ := (Complex.exp (-Complex.I * x) - Complex.exp (Complex.I * x)) / (2 * Complex.I)
noncomputable def cos (x : ℂ) : ℂ := (Complex.exp (Complex.I * x) + Complex.exp (-Complex.I * x)) / 2

theorem power_six_rectangular_form :
  (2 * cos (20 * Real.pi / 180) + 2 * Complex.I * sin (20 * Real.pi / 180))^6 = -32 + 32 * Complex.I * Real.sqrt 3 := sorry

end power_six_rectangular_form_l179_179373


namespace find_original_number_l179_179738

theorem find_original_number (x : ℝ)
  (h : (((x + 3) * 3 - 3) / 3) = 10) : x = 8 :=
sorry

end find_original_number_l179_179738


namespace fraction_sum_in_simplest_form_l179_179446

theorem fraction_sum_in_simplest_form :
  ∃ a b : ℕ, a + b = 11407 ∧ 0.425875 = a / (b : ℝ) ∧ Nat.gcd a b = 1 :=
by
  sorry

end fraction_sum_in_simplest_form_l179_179446


namespace option_A_option_B_option_C_option_D_l179_179460

variables {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b)

-- A: Prove that \(a(6 - a) \leq 9\).
theorem option_A (h : 0 < a ∧ 0 < b) : a * (6 - a) ≤ 9 := sorry

-- B: Prove that if \(ab = a + b + 3\), then \(ab \geq 9\).
theorem option_B (h : ab = a + b + 3) : ab ≥ 9 := sorry

-- C: Prove that the minimum value of \(a^2 + \frac{4}{a^2 + 3}\) is not equal to 1.
theorem option_C : ∀ a > 0, (a^2 + 4 / (a^2 + 3) ≠ 1) := sorry

-- D: Prove that if \(a + b = 2\), then \(\frac{1}{a} + \frac{2}{b} \geq \frac{3}{2} + \sqrt{2}\).
theorem option_D (h : a + b = 2) : (1 / a + 2 / b) ≥ (3 / 2 + Real.sqrt 2) := sorry

end option_A_option_B_option_C_option_D_l179_179460


namespace geom_seq_product_equals_16_l179_179380

theorem geom_seq_product_equals_16
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : ∀ m n, a (m + 1) - a m = a (n + 1) - a n)
  (non_zero_diff : ∃ d, d ≠ 0 ∧ ∀ n, a (n + 1) - a n = d)
  (h_cond : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  (h_geom : ∀ m n, b (m + 1) / b m = b (n + 1) / b n)
  (h_b7 : b 7 = a 7):
  b 6 * b 8 = 16 := 
sorry

end geom_seq_product_equals_16_l179_179380


namespace exactly_one_greater_than_one_l179_179181

theorem exactly_one_greater_than_one (x1 x2 x3 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h4 : x1 * x2 * x3 = 1)
  (h5 : x1 + x2 + x3 > (1 / x1) + (1 / x2) + (1 / x3)) :
  (x1 > 1 ∧ x2 ≤ 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 > 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 ≤ 1 ∧ x3 > 1) :=
sorry

end exactly_one_greater_than_one_l179_179181


namespace new_average_age_l179_179374

theorem new_average_age (n : ℕ) (avg_old : ℕ) (new_person_age : ℕ) (new_avg_age : ℕ)
  (h1 : avg_old = 14)
  (h2 : n = 9)
  (h3 : new_person_age = 34)
  (h4 : new_avg_age = 16) :
  (n * avg_old + new_person_age) / (n + 1) = new_avg_age :=
sorry

end new_average_age_l179_179374


namespace range_of_x_l179_179596

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  1 - x > 0

theorem range_of_x (x : ℝ) : range_of_independent_variable x → x < 1 :=
by sorry

end range_of_x_l179_179596


namespace calculate_expression_l179_179690

variable (a : ℝ)

theorem calculate_expression (h : a ≠ 0) : (6 * a^2) / (a / 2) = 12 * a := by
  sorry

end calculate_expression_l179_179690


namespace slope_negative_l179_179912

theorem slope_negative (k b m n : ℝ) (h₁ : k ≠ 0) (h₂ : m < n) 
  (ha : m = k * 1 + b) (hb : n = k * -1 + b) : k < 0 :=
by
  sorry

end slope_negative_l179_179912


namespace bicycle_meets_light_vehicle_l179_179100

noncomputable def meeting_time (v_1 v_2 v_3 v_4 : ℚ) : ℚ :=
  let x := 2 * (v_1 + v_4)
  let y := 6 * (v_2 - v_4)
  (x + y) / (v_3 + v_4) + 12

theorem bicycle_meets_light_vehicle (v_1 v_2 v_3 v_4 : ℚ) (h1 : 2 * (v_1 + v_4) = x)
  (h2 : x + y = 4 * (v_1 + v_2))
  (h3 : x + y = 5 * (v_2 + v_3))
  (h4 : 6 * (v_2 - v_4) = y) :
  meeting_time v_1 v_2 v_3 v_4 = 15 + 1/3 :=
by
  sorry

end bicycle_meets_light_vehicle_l179_179100


namespace chadSavingsIsCorrect_l179_179273

noncomputable def chadSavingsAfterTaxAndConversion : ℝ :=
  let euroToUsd := 1.20
  let poundToUsd := 1.40
  let euroIncome := 600 * euroToUsd
  let poundIncome := 250 * poundToUsd
  let dollarIncome := 150 + 150
  let totalIncome := euroIncome + poundIncome + dollarIncome
  let taxRate := 0.10
  let taxedIncome := totalIncome * (1 - taxRate)
  let savingsRate := if taxedIncome ≤ 1000 then 0.20
                     else if taxedIncome ≤ 2000 then 0.30
                     else if taxedIncome ≤ 3000 then 0.40
                     else 0.50
  let savings := taxedIncome * savingsRate
  savings

theorem chadSavingsIsCorrect : chadSavingsAfterTaxAndConversion = 369.90 := by
  sorry

end chadSavingsIsCorrect_l179_179273


namespace negative_half_less_than_negative_third_l179_179235

theorem negative_half_less_than_negative_third : - (1 / 2 : ℝ) < - (1 / 3 : ℝ) :=
by
  sorry

end negative_half_less_than_negative_third_l179_179235


namespace balls_into_boxes_l179_179182

theorem balls_into_boxes : ∃ n : ℕ, n = 240 ∧ ∃ f : Fin 5 → Fin 4, ∀ i : Fin 4, ∃ j : Fin 5, f j = i := by
  sorry

end balls_into_boxes_l179_179182


namespace stampsLeftover_l179_179048

-- Define the number of stamps each person has
def oliviaStamps : ℕ := 52
def parkerStamps : ℕ := 66
def quinnStamps : ℕ := 23

-- Define the album's capacity in stamps
def albumCapacity : ℕ := 15

-- Define the total number of leftovers
def totalLeftover : ℕ := (oliviaStamps + parkerStamps + quinnStamps) % albumCapacity

-- Define the theorem we want to prove
theorem stampsLeftover : totalLeftover = 6 := by
  sorry

end stampsLeftover_l179_179048


namespace sufficient_but_not_necessary_l179_179215

theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : |a| > 0 → a > 0 ∨ a < 0) : 
  (a > 0 → |a| > 0) ∧ (¬(|a| > 0 → a > 0)) := 
by
  sorry

end sufficient_but_not_necessary_l179_179215


namespace product_of_numbers_l179_179466

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := 
sorry

end product_of_numbers_l179_179466


namespace find_constants_l179_179626

theorem find_constants (A B C : ℝ) (hA : A = 7) (hB : B = -9) (hC : C = 5) :
  (∀ (x : ℝ), x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → 
    ( -2 * x ^ 2 + 5 * x - 7) / (x ^ 3 - x) = A / x + (B * x + C) / (x ^ 2 - 1) ) :=
by
  intros x hx
  rw [hA, hB, hC]
  sorry

end find_constants_l179_179626


namespace inequality_am_gm_l179_179500

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
by
  sorry

end inequality_am_gm_l179_179500


namespace circle_radius_l179_179294

theorem circle_radius (r A C : Real) (h1 : A = π * r^2) (h2 : C = 2 * π * r) (h3 : A + (Real.cos (π / 3)) * C = 56 * π) : r = 7 := 
by 
  sorry

end circle_radius_l179_179294


namespace ratio_mn_eq_x_plus_one_over_two_x_plus_one_l179_179047

theorem ratio_mn_eq_x_plus_one_over_two_x_plus_one (x : ℝ) (m n : ℝ) 
  (hx : x > 0) 
  (hmn : m * n ≠ 0) 
  (hineq : m * x > n * x + n) : 
  m / (m + n) = (x + 1) / (2 * x + 1) := 
by 
  sorry

end ratio_mn_eq_x_plus_one_over_two_x_plus_one_l179_179047


namespace sufficient_and_not_necessary_condition_l179_179065

theorem sufficient_and_not_necessary_condition (a b : ℝ) (hb: a < 0 ∧ b < 0) : a + b < 0 :=
by
  sorry

end sufficient_and_not_necessary_condition_l179_179065


namespace plates_to_remove_l179_179534

-- Definitions based on the problem conditions
def number_of_plates : ℕ := 38
def weight_per_plate : ℕ := 10
def acceptable_weight : ℕ := 320

-- Theorem to prove
theorem plates_to_remove (initial_weight := number_of_plates * weight_per_plate) 
  (excess_weight := initial_weight - acceptable_weight) 
  (plates_to_remove := excess_weight / weight_per_plate) :
  plates_to_remove = 6 :=
by
  sorry

end plates_to_remove_l179_179534


namespace complex_identity_l179_179616

theorem complex_identity (a b : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 - 2 * i) * i = a + b * i) : a * b = 2 :=
by
  sorry

end complex_identity_l179_179616


namespace average_marks_of_class_l179_179591

theorem average_marks_of_class :
  (∀ (students total_students: ℕ) (marks95 marks0: ℕ) (avg_remaining: ℕ),
    total_students = 25 →
    students = 3 →
    marks95 = 95 →
    students = 5 →
    marks0 = 0 →
    (total_students - students - students) = 17 →
    avg_remaining = 45 →
    ((students * marks95 + students * marks0 + (total_students - students - students) * avg_remaining) / total_students) = 42)
:= sorry

end average_marks_of_class_l179_179591


namespace number_of_books_Ryan_l179_179574

structure LibraryProblem :=
  (Total_pages_Ryan : ℕ)
  (Total_days : ℕ)
  (Pages_per_book_brother : ℕ)
  (Extra_pages_Ryan : ℕ)

def calculate_books_received (p : LibraryProblem) : ℕ :=
  let Total_pages_brother := p.Pages_per_book_brother * p.Total_days
  let Ryan_daily_average := (Total_pages_brother / p.Total_days) + p.Extra_pages_Ryan
  p.Total_pages_Ryan / Ryan_daily_average

theorem number_of_books_Ryan (p : LibraryProblem) (h1 : p.Total_pages_Ryan = 2100)
  (h2 : p.Total_days = 7) (h3 : p.Pages_per_book_brother = 200) (h4 : p.Extra_pages_Ryan = 100) :
  calculate_books_received p = 7 := by
  sorry

end number_of_books_Ryan_l179_179574


namespace carson_gets_clawed_39_times_l179_179484

-- Conditions
def number_of_wombats : ℕ := 9
def claws_per_wombat : ℕ := 4
def number_of_rheas : ℕ := 3
def claws_per_rhea : ℕ := 1

-- Theorem statement
theorem carson_gets_clawed_39_times :
  (number_of_wombats * claws_per_wombat + number_of_rheas * claws_per_rhea) = 39 :=
by
  sorry

end carson_gets_clawed_39_times_l179_179484


namespace profit_difference_l179_179846

-- Define the initial capitals of A, B, and C
def capital_A := 8000
def capital_B := 10000
def capital_C := 12000

-- Define B's profit share
def profit_share_B := 3500

-- Define the total number of parts
def total_parts := 15

-- Define the number of parts for each person
def parts_A := 4
def parts_B := 5
def parts_C := 6

-- Define the total profit
noncomputable def total_profit := profit_share_B * (total_parts / parts_B)

-- Define the profit shares of A and C
noncomputable def profit_share_A := (parts_A / total_parts) * total_profit
noncomputable def profit_share_C := (parts_C / total_parts) * total_profit

-- Define the difference between the profit shares of A and C
noncomputable def profit_share_difference := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_difference :
  profit_share_difference = 1400 := by
  sorry

end profit_difference_l179_179846


namespace alex_score_correct_l179_179508

-- Conditions of the problem
def num_students := 20
def average_first_19 := 78
def new_average := 79

-- Alex's score calculation
def alex_score : ℕ :=
  let total_score_first_19 := 19 * average_first_19
  let total_score_all := num_students * new_average
  total_score_all - total_score_first_19

-- Problem statement: Prove Alex's score is 98
theorem alex_score_correct : alex_score = 98 := by
  sorry

end alex_score_correct_l179_179508


namespace waiting_probability_no_more_than_10_seconds_l179_179206

def total_cycle_time : ℕ := 30 + 10 + 40
def proceed_during_time : ℕ := 40 -- green time
def yellow_time : ℕ := 10

theorem waiting_probability_no_more_than_10_seconds :
  (proceed_during_time + yellow_time + yellow_time) / total_cycle_time = 3 / 4 := by
  sorry

end waiting_probability_no_more_than_10_seconds_l179_179206


namespace diane_postage_problem_l179_179923

-- Definition of stamps
def stamps : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]

-- Define a function to compute the number of arrangements that sums to a target value
def arrangements_sum_to (target : ℕ) (stamps : List (ℕ × ℕ)) : ℕ :=
  sorry -- Implementation detail is skipped

-- The main theorem to prove
theorem diane_postage_problem :
  arrangements_sum_to 15 stamps = 271 :=
by sorry

end diane_postage_problem_l179_179923


namespace domain_log_function_l179_179697

theorem domain_log_function :
  { x : ℝ | 12 + x - x^2 > 0 } = { x : ℝ | -3 < x ∧ x < 4 } :=
sorry

end domain_log_function_l179_179697


namespace angle_A_l179_179054

variable (a b c : ℝ) (A B C : ℝ)

-- Hypothesis: In triangle ABC, (a + c)(a - c) = b(b + c)
def condition (a b c : ℝ) : Prop := (a + c) * (a - c) = b * (b + c)

-- The goal is to show that under given conditions, ∠A = 2π/3
theorem angle_A (h : condition a b c) : A = 2 * π / 3 :=
sorry

end angle_A_l179_179054


namespace gcd_decomposition_l179_179441

open Polynomial

noncomputable def f : Polynomial ℚ := 4 * X ^ 4 - 2 * X ^ 3 - 16 * X ^ 2 + 5 * X + 9
noncomputable def g : Polynomial ℚ := 2 * X ^ 3 - X ^ 2 - 5 * X + 4

theorem gcd_decomposition :
  ∃ (u v : Polynomial ℚ), u * f + v * g = X - 1 :=
sorry

end gcd_decomposition_l179_179441


namespace area_covered_by_congruent_rectangles_l179_179272

-- Definitions of conditions
def length_AB : ℕ := 12
def width_AD : ℕ := 8
def area_rect (l w : ℕ) : ℕ := l * w

-- Center of the first rectangle
def center_ABCD : ℕ × ℕ := (length_AB / 2, width_AD / 2)

-- Proof statement
theorem area_covered_by_congruent_rectangles 
  (length_ABCD length_EFGH width_ABCD width_EFGH : ℕ)
  (congruent : length_ABCD = length_EFGH ∧ width_ABCD = width_EFGH)
  (center_E : ℕ × ℕ)
  (H_center_E : center_E = center_ABCD) :
  area_rect length_ABCD width_ABCD + area_rect length_EFGH width_EFGH - length_ABCD * width_ABCD / 2 = 168 := by
  sorry

end area_covered_by_congruent_rectangles_l179_179272


namespace joan_balloon_gain_l179_179083

theorem joan_balloon_gain
  (initial_balloons : ℕ)
  (final_balloons : ℕ)
  (h_initial : initial_balloons = 9)
  (h_final : final_balloons = 11) :
  final_balloons - initial_balloons = 2 :=
by {
  sorry
}

end joan_balloon_gain_l179_179083


namespace new_price_of_computer_l179_179070

theorem new_price_of_computer (d : ℝ) (h : 2 * d = 520) : d * 1.3 = 338 := 
sorry

end new_price_of_computer_l179_179070


namespace intersection_complement_l179_179394

open Set

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | 2 < x}
def R_complement_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement : M ∩ R_complement_N = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_complement_l179_179394


namespace cost_of_rusted_side_l179_179677

-- Define the conditions
def perimeter (s : ℕ) (l : ℕ) : ℕ :=
  2 * s + 2 * l

def long_side (s : ℕ) : ℕ :=
  3 * s

def cost_per_foot : ℕ :=
  5

-- Given these conditions, we prove the cost of replacing one short side.
theorem cost_of_rusted_side (s l : ℕ) (h1 : perimeter s l = 640) (h2 : l = long_side s) : 
  5 * s = 400 :=
by 
  sorry

end cost_of_rusted_side_l179_179677


namespace chess_club_girls_l179_179603

theorem chess_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 2 : ℝ) * G + B = 20) : G = 24 :=
by
  -- proof
  sorry

end chess_club_girls_l179_179603


namespace mango_production_l179_179838

-- Conditions
def num_papaya_trees := 2
def papayas_per_tree := 10
def num_mango_trees := 3
def total_fruits := 80

-- Definition to be proven
def mangos_per_mango_tree : Nat :=
  (total_fruits - num_papaya_trees * papayas_per_tree) / num_mango_trees

theorem mango_production :
  mangos_per_mango_tree = 20 := by
  sorry

end mango_production_l179_179838


namespace oil_drop_probability_l179_179443

theorem oil_drop_probability :
  let r_circle := 1 -- radius of the circle in cm
  let side_square := 0.5 -- side length of the square in cm
  let area_circle := π * r_circle^2
  let area_square := side_square * side_square
  (area_square / area_circle) = 1 / (4 * π) :=
by
  sorry

end oil_drop_probability_l179_179443


namespace minimum_bailing_rate_l179_179876

theorem minimum_bailing_rate
  (distance_from_shore : Real := 1.5)
  (rowing_speed : Real := 3)
  (water_intake_rate : Real := 12)
  (max_water : Real := 45) :
  (distance_from_shore / rowing_speed) * 60 * water_intake_rate - max_water / ((distance_from_shore / rowing_speed) * 60) >= 10.5 :=
by
  -- Provide the units are consistent and the calculations agree with the given numerical data
  sorry

end minimum_bailing_rate_l179_179876


namespace arithmetic_geometric_sum_l179_179387

def a (n : ℕ) : ℕ := 3 * n - 2
def b (n : ℕ) : ℕ := 3 ^ (n - 1)

theorem arithmetic_geometric_sum :
  a (b 1) + a (b 2) + a (b 3) = 33 := by
  sorry

end arithmetic_geometric_sum_l179_179387


namespace solve_adult_tickets_l179_179862

theorem solve_adult_tickets (A C : ℕ) (h1 : 8 * A + 5 * C = 236) (h2 : A + C = 34) : A = 22 :=
sorry

end solve_adult_tickets_l179_179862


namespace sufficient_not_necessary_condition_l179_179311

theorem sufficient_not_necessary_condition (x : ℝ) : (1 < x ∧ x < 2) → (x < 2) ∧ ((x < 2) → ¬(1 < x ∧ x < 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l179_179311


namespace gamma_donuts_received_l179_179076

theorem gamma_donuts_received (total_donuts delta_donuts gamma_donuts beta_donuts : ℕ) 
    (h1 : total_donuts = 40) 
    (h2 : delta_donuts = 8) 
    (h3 : beta_donuts = 3 * gamma_donuts) :
    delta_donuts + beta_donuts + gamma_donuts = total_donuts -> gamma_donuts = 8 :=
by 
  intro h4
  sorry

end gamma_donuts_received_l179_179076


namespace ratio_between_second_and_third_l179_179219

noncomputable def ratio_second_third : ℚ := sorry

theorem ratio_between_second_and_third (A B C : ℕ) (h₁ : A + B + C = 98) (h₂ : A * 3 = B * 2) (h₃ : B = 30) :
  ratio_second_third = 5 / 8 := sorry

end ratio_between_second_and_third_l179_179219


namespace expand_and_simplify_expression_l179_179217

variable {x y : ℝ} {i : ℂ}

-- Declare i as the imaginary unit satisfying i^2 = -1
axiom imaginary_unit : i^2 = -1

theorem expand_and_simplify_expression :
  (x + 3 + i * y) * (x + 3 - i * y) + (x - 2 + 2 * i * y) * (x - 2 - 2 * i * y)
  = 2 * x^2 + 2 * x + 13 - 5 * y^2 :=
by
  sorry

end expand_and_simplify_expression_l179_179217


namespace unique_divisors_form_l179_179689

theorem unique_divisors_form (n : ℕ) (h₁ : n > 1)
    (h₂ : ∀ d : ℕ, d ∣ n ∧ d > 1 → ∃ a r : ℕ, a > 1 ∧ r > 1 ∧ d = a^r + 1) :
    n = 10 := by
  sorry

end unique_divisors_form_l179_179689


namespace determine_y_l179_179943

theorem determine_y : 
  ∀ y : ℝ, 
    (2 * Real.arctan (1 / 5) + Real.arctan (1 / 25) + Real.arctan (1 / y) = Real.pi / 4) -> 
    y = -121 / 60 :=
by
  sorry

end determine_y_l179_179943


namespace tangent_line_at_x_2_range_of_m_for_three_roots_l179_179290

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

/-
Part 1: Proving the tangent line equation at x = 2
-/
theorem tangent_line_at_x_2 : ∃ k b, (k = 12) ∧ (b = -17) ∧ 
  (∀ x, 12 * x - f 2 - 17 = 0) :=
by
  sorry

/-
Part 2: Proving the range of m for three distinct real roots
-/
theorem range_of_m_for_three_roots (m : ℝ) :
  (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 + m = 0 ∧ f x2 + m = 0 ∧ f x3 + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
by
  sorry

end tangent_line_at_x_2_range_of_m_for_three_roots_l179_179290


namespace like_terms_constants_l179_179096

theorem like_terms_constants :
  ∀ (a b : ℚ), a = 1/2 → b = -1/3 → (a = 1/2 ∧ b = -1/3) → a + b = 1/2 + -1/3 :=
by
  intros a b ha hb h
  sorry

end like_terms_constants_l179_179096


namespace butterfly_count_l179_179426

theorem butterfly_count (total_butterflies : ℕ) (one_third_flew_away : ℕ) (initial_butterflies : total_butterflies = 9) (flew_away : one_third_flew_away = total_butterflies / 3) : 
(total_butterflies - one_third_flew_away) = 6 := by
  sorry

end butterfly_count_l179_179426


namespace johnny_selection_process_l179_179025

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem johnny_selection_process : 
  binomial_coefficient 10 4 * binomial_coefficient 4 2 = 1260 :=
by
  sorry

end johnny_selection_process_l179_179025


namespace increasing_interval_of_f_l179_179766

def f (x : ℝ) : ℝ := (x - 1) ^ 2 - 2

theorem increasing_interval_of_f : ∀ x, 1 < x → f x > f 1 := 
sorry

end increasing_interval_of_f_l179_179766


namespace mono_increasing_intervals_l179_179558

noncomputable def f : ℝ → ℝ :=
by sorry

theorem mono_increasing_intervals (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_sym : ∀ x, f x = f (-2 - x))
  (h_decr1 : ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ -1 → f y ≤ f x) :
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f x ≤ f y) ∧
  (∀ x y, 3 ≤ x ∧ x < y ∧ y ≤ 4 → f x ≤ f y) :=
sorry

end mono_increasing_intervals_l179_179558


namespace sum_series_eq_l179_179331

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end sum_series_eq_l179_179331


namespace isosceles_triangle_base_l179_179572

variable (a b : ℕ)

theorem isosceles_triangle_base 
  (h_isosceles : a = 7 ∧ b = 3)
  (triangle_inequality : 7 + 7 > 3) : b = 3 := by
-- Begin of the proof
sorry
-- End of the proof

end isosceles_triangle_base_l179_179572


namespace calc_one_calc_two_calc_three_l179_179577

theorem calc_one : (54 + 38) * 15 = 1380 := by
  sorry

theorem calc_two : 1500 - 32 * 45 = 60 := by
  sorry

theorem calc_three : 157 * (70 / 35) = 314 := by
  sorry

end calc_one_calc_two_calc_three_l179_179577


namespace hyperbola_asymptote_l179_179432

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y, 3 * x + 2 * y = 0 ∨ 3 * x - 2 * y = 0) →
  (∀ x y, y * y = 9 * (x * x / (a * a) - 1)) →
  a = 2 :=
by
  intros asymptote_constr hyp
  sorry

end hyperbola_asymptote_l179_179432


namespace actors_duration_l179_179317

-- Definition of conditions
def actors_at_a_time := 5
def total_actors := 20
def total_minutes := 60

-- Main statement to prove
theorem actors_duration : total_minutes / (total_actors / actors_at_a_time) = 15 := 
by
  sorry

end actors_duration_l179_179317


namespace walking_time_l179_179970

theorem walking_time (r s : ℕ) (h₁ : r + s = 50) (h₂ : 2 * s = 30) : 2 * r = 70 :=
by
  sorry

end walking_time_l179_179970


namespace evaluate_expression_l179_179544

theorem evaluate_expression : 
  3 * (-3)^4 + 3 * (-3)^3 + 3 * (-3)^2 + 3 * 3^2 + 3 * 3^3 + 3 * 3^4 = 540 := 
by 
  sorry

end evaluate_expression_l179_179544


namespace coprime_with_others_l179_179684

theorem coprime_with_others:
  ∀ (a b c d e : ℕ),
  a = 20172017 → 
  b = 20172018 → 
  c = 20172019 →
  d = 20172020 →
  e = 20172021 →
  (Nat.gcd c a = 1 ∧ 
   Nat.gcd c b = 1 ∧ 
   Nat.gcd c d = 1 ∧ 
   Nat.gcd c e = 1) :=
by
  sorry

end coprime_with_others_l179_179684


namespace number_of_possible_scenarios_l179_179045

-- Definitions based on conditions
def num_companies : Nat := 5
def reps_company_A : Nat := 2
def reps_other_companies : Nat := 1
def total_speakers : Nat := 3

-- Problem statement
theorem number_of_possible_scenarios : 
  ∃ (scenarios : Nat), scenarios = 16 ∧ 
  (scenarios = 
    (Nat.choose reps_company_A 1 * Nat.choose 4 2) + 
    Nat.choose 4 3) :=
by
  sorry

end number_of_possible_scenarios_l179_179045


namespace lemons_needed_for_3_dozen_is_9_l179_179227

-- Define the conditions
def lemon_tbs : ℕ := 4
def juice_needed_per_dozen : ℕ := 12
def dozens_needed : ℕ := 3
def total_juice_needed : ℕ := juice_needed_per_dozen * dozens_needed

-- The number of lemons needed to make 3 dozen cupcakes
def lemons_needed (total_juice : ℕ) (lemon_juice : ℕ) : ℕ :=
  total_juice / lemon_juice

-- Prove the number of lemons needed == 9
theorem lemons_needed_for_3_dozen_is_9 : lemons_needed total_juice_needed lemon_tbs = 9 :=
  by sorry

end lemons_needed_for_3_dozen_is_9_l179_179227


namespace bd_ad_ratio_l179_179718

noncomputable def mass_point_geometry_bd_ad : ℚ := 
  let AT_OVER_ET := 5
  let DT_OVER_CT := 2
  let mass_A := 1
  let mass_D := 3 * mass_A
  let mass_B := mass_A + mass_D
  mass_B / mass_D

theorem bd_ad_ratio (h1 : AT/ET = 5) (h2 : DT/CT = 2) : BD/AD = 4 / 3 :=
by
  have mass_A := 1
  have mass_D := 3
  have mass_B := 4
  have h := mass_B / mass_D
  sorry

end bd_ad_ratio_l179_179718


namespace smallest_prime_with_digits_sum_22_l179_179035

def digits_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem smallest_prime_with_digits_sum_22 : 
  ∃ p : ℕ, Prime p ∧ digits_sum p = 22 ∧ ∀ q : ℕ, Prime q ∧ digits_sum q = 22 → q ≥ p ∧ p = 499 :=
by sorry

end smallest_prime_with_digits_sum_22_l179_179035


namespace cos_neg_45_eq_one_over_sqrt_two_l179_179501

theorem cos_neg_45_eq_one_over_sqrt_two : Real.cos (-(45 : ℝ)) = 1 / Real.sqrt 2 := 
by
  sorry

end cos_neg_45_eq_one_over_sqrt_two_l179_179501


namespace ratio_of_paper_plates_l179_179836

theorem ratio_of_paper_plates (total_pallets : ℕ) (paper_towels : ℕ) (tissues : ℕ) (paper_cups : ℕ) :
  total_pallets = 20 →
  paper_towels = 20 / 2 →
  tissues = 20 / 4 →
  paper_cups = 1 →
  (total_pallets - (paper_towels + tissues + paper_cups)) / total_pallets = 1 / 5 :=
by
  intros h_total h_towels h_tissues h_cups
  sorry

end ratio_of_paper_plates_l179_179836


namespace rectangle_area_l179_179851

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 56) :
  l * b = 147 := by
  sorry

end rectangle_area_l179_179851


namespace sum_gcd_lcm_eq_4851_l179_179453

theorem sum_gcd_lcm_eq_4851 (a b : ℕ) (ha : a = 231) (hb : b = 4620) :
  Nat.gcd a b + Nat.lcm a b = 4851 :=
by
  rw [ha, hb]
  sorry

end sum_gcd_lcm_eq_4851_l179_179453


namespace non_integer_x_and_y_impossible_l179_179471

theorem non_integer_x_and_y_impossible 
  (x y : ℚ) (m n : ℤ) 
  (h1 : 5 * x + 7 * y = m)
  (h2 : 7 * x + 10 * y = n) : 
  ∃ (x y : ℤ), 5 * x + 7 * y = m ∧ 7 * x + 10 * y = n := 
sorry

end non_integer_x_and_y_impossible_l179_179471


namespace pyramid_height_l179_179948

noncomputable def height_pyramid (perimeter_base : ℝ) (distance_apex_vertex : ℝ) : ℝ :=
  let side_length := perimeter_base / 4
  let half_diagonal := (side_length * Real.sqrt 2) / 2
  Real.sqrt (distance_apex_vertex ^ 2 - half_diagonal ^ 2)

theorem pyramid_height
  (perimeter_base: ℝ)
  (h_perimeter : perimeter_base = 32)
  (distance_apex_vertex: ℝ)
  (h_distance : distance_apex_vertex = 10) :
  height_pyramid perimeter_base distance_apex_vertex = 2 * Real.sqrt 17 :=
by
  sorry

end pyramid_height_l179_179948


namespace find_certain_number_l179_179643

theorem find_certain_number (x : ℝ) (h : 0.80 * x = (4 / 5 * 20) + 16) : x = 40 :=
by sorry

end find_certain_number_l179_179643


namespace sixth_student_stickers_l179_179911

-- Define the given conditions.
def first_student_stickers := 29
def increment := 6

-- Define the number of stickers given to each subsequent student.
def stickers (n : ℕ) : ℕ :=
  first_student_stickers + n * increment

-- Theorem statement: the 6th student will receive 59 stickers.
theorem sixth_student_stickers : stickers 5 = 59 :=
by
  sorry

end sixth_student_stickers_l179_179911


namespace sum_of_remainders_l179_179448

theorem sum_of_remainders (n : ℤ) (h : n % 12 = 5) :
  (n % 4) + (n % 3) = 3 :=
by
  sorry

end sum_of_remainders_l179_179448


namespace values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l179_179210

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 - 3 * a * x^2 + 2 * b * x

theorem values_of_a_and_b (h : ∀ x, f x (1 / 3) (-1 / 2) ≤ f 1 (1 / 3) (-1 / 2)) :
  (∃ a b, a = 1 / 3 ∧ b = -1 / 2) :=
sorry

theorem intervals_of_monotonicity (a b : ℝ) (h : ∀ x, f x a b ≤ f 1 a b) :
  (∀ x, (f x a b ≥ 0 ↔ x ≤ -1 / 3 ∨ x ≥ 1) ∧ (f x a b ≤ 0 ↔ -1 / 3 ≤ x ∧ x ≤ 1)) :=
sorry

theorem range_of_a_for_three_roots :
  (∃ a, -1 < a ∧ a < 5 / 27) :=
sorry

end values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l179_179210


namespace range_of_a_l179_179151

theorem range_of_a (a : ℝ) (h : a < 1) : ∀ x : ℝ, |x - 4| + |x - 5| > a :=
by
  sorry

end range_of_a_l179_179151


namespace smallest_n_for_polygon_cutting_l179_179935

theorem smallest_n_for_polygon_cutting : 
  ∃ n : ℕ, (∃ k : ℕ, n - 2 = k * 31) ∧ (∃ k' : ℕ, n - 2 = k' * 65) ∧ n = 2017 :=
sorry

end smallest_n_for_polygon_cutting_l179_179935


namespace arithmetic_sequence_problem_l179_179944

variable (a_2 a_4 a_3 : ℤ)

theorem arithmetic_sequence_problem (h : a_2 + a_4 = 16) : a_3 = 8 :=
by
  -- The proof is not needed as per the instructions
  sorry

end arithmetic_sequence_problem_l179_179944


namespace sum_of_coefficients_l179_179412

theorem sum_of_coefficients (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℤ) :
  (5 * x - 2) ^ 6 = b_6 * x ^ 6 + b_5 * x ^ 5 + b_4 * x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 →
  b_6 + b_5 + b_4 + b_3 + b_2 + b_1 + b_0 = 729 :=
by
  sorry

end sum_of_coefficients_l179_179412


namespace quadratic_inequality_solution_l179_179002

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : (∀ x, ax^2 + bx + c = 0 ↔ x = 1 ∨ x = 3)) : 
  ∀ x, cx^2 + bx + a > 0 ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l179_179002


namespace parabola_y_intercepts_l179_179533

theorem parabola_y_intercepts : 
  (∀ y : ℝ, 3 * y^2 - 6 * y + 1 = 0) → (∃ y1 y2 : ℝ, y1 ≠ y2) :=
by sorry

end parabola_y_intercepts_l179_179533


namespace number_of_representatives_from_companyA_l179_179761

-- Define conditions
def companyA_representatives : ℕ := 120
def companyB_representatives : ℕ := 100
def total_selected : ℕ := 11

-- Define the theorem
theorem number_of_representatives_from_companyA : 120 * (11 / (120 + 100)) = 6 := by
  sorry

end number_of_representatives_from_companyA_l179_179761


namespace balls_in_boxes_l179_179105

theorem balls_in_boxes : 
  let balls := 4
  let boxes := 3
  (boxes^balls = 81) :=
by sorry

end balls_in_boxes_l179_179105


namespace time_difference_l179_179164

noncomputable def hour_angle (n : ℝ) : ℝ :=
  150 + (n / 2)

noncomputable def minute_angle (n : ℝ) : ℝ :=
  6 * n

theorem time_difference (n1 n2 : ℝ)
  (h1 : |(hour_angle n1) - (minute_angle n1)| = 120)
  (h2 : |(hour_angle n2) - (minute_angle n2)| = 120) :
  n2 - n1 = 43.64 := 
sorry

end time_difference_l179_179164


namespace take_home_pay_correct_l179_179784

def jonessa_pay : ℝ := 500
def tax_deduction_percent : ℝ := 0.10
def insurance_deduction_percent : ℝ := 0.05
def pension_plan_deduction_percent : ℝ := 0.03
def union_dues_deduction_percent : ℝ := 0.02

def total_deductions : ℝ :=
  jonessa_pay * tax_deduction_percent +
  jonessa_pay * insurance_deduction_percent +
  jonessa_pay * pension_plan_deduction_percent +
  jonessa_pay * union_dues_deduction_percent

def take_home_pay : ℝ := jonessa_pay - total_deductions

theorem take_home_pay_correct : take_home_pay = 400 :=
  by
  sorry

end take_home_pay_correct_l179_179784


namespace p_sufficient_not_necessary_for_q_l179_179089

variable (a : ℝ)

def p : Prop := a > 0
def q : Prop := a^2 + a ≥ 0

theorem p_sufficient_not_necessary_for_q : (p a → q a) ∧ ¬ (q a → p a) := by
  sorry

end p_sufficient_not_necessary_for_q_l179_179089


namespace guess_x_30_guess_y_127_l179_179053

theorem guess_x_30 : 120 = 4 * 30 := 
  sorry

theorem guess_y_127 : 87 = 127 - 40 := 
  sorry

end guess_x_30_guess_y_127_l179_179053


namespace find_a9_l179_179763

variable {a_n : ℕ → ℝ}

-- Definition of arithmetic progression
def is_arithmetic_progression (a : ℕ → ℝ) (a1 d : ℝ) := ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Conditions
variables (a1 d : ℝ)
variable (h1 : a1 + (a1 + d)^2 = -3)
variable (h2 : ((a1 + a1 + 4 * d) * 5 / 2) = 10)

-- Question, needing the final statement
theorem find_a9 (a : ℕ → ℝ) (ha : is_arithmetic_progression a a1 d) : a 9 = 20 :=
by
    -- Since the theorem requires solving the statements, we use sorry to skip the proof.
    sorry

end find_a9_l179_179763


namespace find_missing_number_l179_179555

theorem find_missing_number (x : ℕ) (h : 10111 - 10 * 2 * x = 10011) : x = 5 :=
sorry

end find_missing_number_l179_179555


namespace lines_parallel_lines_perpendicular_l179_179147

-- Definition of lines
def l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + a ^ 2 - 1 = 0

-- Parallel condition proof problem
theorem lines_parallel (a : ℝ) : (a = -1) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y →  
        (-(a / 2) = (1 / (1 - a))) ∧ (-3 ≠ -a - 1) :=
by
  intros
  sorry

-- Perpendicular condition proof problem
theorem lines_perpendicular (a : ℝ) : (a = 2 / 3) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y → 
        (- (a / 2) * (1 / (1 - a)) = -1) :=
by
  intros
  sorry

end lines_parallel_lines_perpendicular_l179_179147


namespace perimeter_remaining_shape_l179_179529

theorem perimeter_remaining_shape (length width square1 square2 : ℝ) 
  (H_len : length = 50) (H_width : width = 20) 
  (H_sq1 : square1 = 12) (H_sq2 : square2 = 4) : 
  2 * (length + width) + 4 * (square1 + square2) = 204 :=
by 
  rw [H_len, H_width, H_sq1, H_sq2]
  sorry

end perimeter_remaining_shape_l179_179529


namespace average_speed_bike_l179_179871

theorem average_speed_bike (t_goal : ℚ) (d_swim r_swim : ℚ) (d_run r_run : ℚ) (d_bike r_bike : ℚ) :
  t_goal = 1.75 →
  d_swim = 1 / 3 ∧ r_swim = 1.5 →
  d_run = 2.5 ∧ r_run = 8 →
  d_bike = 12 →
  r_bike = 1728 / 175 :=
by
  intros h_goal h_swim h_run h_bike
  sorry

end average_speed_bike_l179_179871


namespace max_seq_value_l179_179569

def is_arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + m) = a n + a m

variables (a : ℕ → ℤ)
variables (S : ℕ → ℤ)

axiom distinct_terms (h : is_arithmetic_seq a) : ∀ n m, n ≠ m → a n ≠ a m
axiom condition_1 : ∀ n, a (2 * n) = 2 * a n - 3
axiom condition_2 : a 6 * a 6 = a 1 * a 21
axiom sum_of_first_n_terms : ∀ n, S n = n * (n + 4)

noncomputable def seq (n : ℕ) : ℤ := S n / 2^(n - 1)

theorem max_seq_value : 
  (∀ n, seq n >= seq (n - 1) ∧ seq n >= seq (n + 1)) → 
  (∃ n, seq n = 6) :=
sorry

end max_seq_value_l179_179569


namespace correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l179_179216

theorem correct_inequality:
    (-21 : ℤ) > (-21 : ℤ) := by sorry

theorem incorrect_inequality1 :
    -abs (10 + 1 / 2) < (8 + 2 / 3) := by sorry

theorem incorrect_inequality2 :
    (-abs (7 + 2 / 3)) ≠ (- (- (7 + 2 / 3))) := by sorry

theorem correct_option_d :
    (-5 / 6 : ℚ) < (-4 / 5 : ℚ) := by sorry

end correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l179_179216


namespace part_1_part_2_l179_179333

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 4)

theorem part_1 (a : ℝ) (h : a = 3) :
  { x : ℝ | f x a ≥ 8 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | 1 ≤ x ∧ x ≤ 3 } ∪ { x : ℝ | x > 3 } := 
sorry

theorem part_2 (h : ∃ x : ℝ, f x a - abs (x + 2) ≤ 4) :
  -6 ≤ a ∧ a ≤ 2 :=
sorry

end part_1_part_2_l179_179333


namespace choose_officers_ways_l179_179567

theorem choose_officers_ways :
  let members := 12
  let vp_candidates := 4
  let remaining_after_president := members - 1
  let remaining_after_vice_president := remaining_after_president - 1
  let remaining_after_secretary := remaining_after_vice_president - 1
  let remaining_after_treasurer := remaining_after_secretary - 1
  (members * vp_candidates * (remaining_after_vice_president) *
   (remaining_after_secretary) * (remaining_after_treasurer)) = 34560 := by
  -- Calculation here
  sorry

end choose_officers_ways_l179_179567


namespace contradiction_assumption_l179_179556

theorem contradiction_assumption (a b c : ℕ) :
  (∃ k : ℕ, (k = a ∨ k = b ∨ k = c) ∧ ∃ n : ℕ, k = 2 * n + 1) →
  (∃ k1 k2 : ℕ, (k1 = a ∨ k1 = b ∨ k1 = c) ∧ (k2 = a ∨ k2 = b ∨ k2 = c) ∧ k1 ≠ k2 ∧ ∃ n1 n2 : ℕ, k1 = 2 * n1 ∧ k2 = 2 * n2) ∨
  (∀ k : ℕ, (k = a ∨ k = b ∨ k = c) → ∃ n : ℕ, k = 2 * n + 1) :=
sorry

end contradiction_assumption_l179_179556


namespace total_books_l179_179984

theorem total_books (hbooks : ℕ) (fbooks : ℕ) (gbooks : ℕ)
  (Harry_books : hbooks = 50)
  (Flora_books : fbooks = 2 * hbooks)
  (Gary_books : gbooks = hbooks / 2) :
  hbooks + fbooks + gbooks = 175 := by
  sorry

end total_books_l179_179984


namespace summer_camp_skills_l179_179399

theorem summer_camp_skills
  (x y z a b c : ℕ)
  (h1 : x + y + z + a + b + c = 100)
  (h2 : y + z + c = 42)
  (h3 : z + x + b = 65)
  (h4 : x + y + a = 29) :
  a + b + c = 64 :=
by sorry

end summer_camp_skills_l179_179399


namespace solve_inequality_l179_179723

theorem solve_inequality (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) → x ≥ -2 :=
by
  intros h
  sorry

end solve_inequality_l179_179723


namespace fish_bird_apple_fraction_l179_179094

theorem fish_bird_apple_fraction (M : ℝ) (hM : 0 < M) :
  let R_fish := 120
  let R_bird := 60
  let R_total := 180
  let T := M / R_total
  let fish_fraction := (R_fish * T) / M
  let bird_fraction := (R_bird * T) / M
  fish_fraction = 2/3 ∧ bird_fraction = 1/3 := by
  sorry

end fish_bird_apple_fraction_l179_179094


namespace range_of_g_l179_179098

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : 
  Set.range g = Set.Icc ((π / 2) - (π / 3)) ((π / 2) + (π / 3)) := by
  sorry

end range_of_g_l179_179098


namespace sum_of_z_values_l179_179914

def f (x : ℚ) : ℚ := x^2 + x + 1

theorem sum_of_z_values : ∃ z₁ z₂ : ℚ, f (4 * z₁) = 12 ∧ f (4 * z₂) = 12 ∧ (z₁ + z₂ = - 1 / 12) :=
by
  sorry

end sum_of_z_values_l179_179914


namespace find_e_m_l179_179316

noncomputable def B (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![4, 5], ![7, e]]
noncomputable def B_inv (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := (1 / (4 * e - 35)) • ![![e, -5], ![-7, 4]]

theorem find_e_m (e m : ℝ) (B_inv_eq_mB : B_inv e = m • B e) : e = -4 ∧ m = 1 / 51 :=
sorry

end find_e_m_l179_179316


namespace amount_spent_on_candy_l179_179540

-- Define the given conditions
def amount_from_mother := 80
def amount_from_father := 40
def amount_from_uncle := 70
def final_amount := 140 

-- Define the initial amount
def initial_amount := amount_from_mother + amount_from_father 

-- Prove the amount spent on candy
theorem amount_spent_on_candy : 
  initial_amount - (final_amount - amount_from_uncle) = 50 := 
by
  -- Placeholder for proof
  sorry

end amount_spent_on_candy_l179_179540


namespace angle_measure_l179_179754

theorem angle_measure (y : ℝ) (hyp : 45 + 3 * y + y = 180) : y = 33.75 :=
by
  sorry

end angle_measure_l179_179754


namespace union_M_N_eq_l179_179934

open Set

-- Define M according to the condition x^2 < 15 for x in ℕ
def M : Set ℕ := {x | x^2 < 15}

-- Define N according to the correct answer
def N : Set ℕ := {x | 0 < x ∧ x < 5}

-- Prove that M ∪ N = {x | 0 ≤ x ∧ x < 5}
theorem union_M_N_eq : M ∪ N = {x : ℕ | 0 ≤ x ∧ x < 5} :=
sorry

end union_M_N_eq_l179_179934


namespace unit_prices_min_chess_sets_l179_179879

-- Define the conditions and prove the unit prices.
theorem unit_prices (x y : ℝ) 
  (h1 : 6 * x + 5 * y = 190)
  (h2 : 8 * x + 10 * y = 320) : 
  x = 15 ∧ y = 20 :=
by
  sorry

-- Define the conditions for the budget and prove the minimum number of chess sets.
theorem min_chess_sets (x y : ℝ) (m : ℕ)
  (hx : x = 15)
  (hy : y = 20)
  (number_sets : m + (100 - m) = 100)
  (budget : 15 * ↑m + 20 * ↑(100 - m) ≤ 1800) :
  m ≥ 40 :=
by
  sorry

end unit_prices_min_chess_sets_l179_179879


namespace middle_income_sample_count_l179_179228

def total_households : ℕ := 600
def high_income_families : ℕ := 150
def middle_income_families : ℕ := 360
def low_income_families : ℕ := 90
def sample_size : ℕ := 80

theorem middle_income_sample_count : 
  (middle_income_families / total_households) * sample_size = 48 := 
by
  sorry

end middle_income_sample_count_l179_179228


namespace arithmetic_sequence_eleventh_term_l179_179131

theorem arithmetic_sequence_eleventh_term 
  (a d : ℚ)
  (h_sum_first_six : 6 * a + 15 * d = 30)
  (h_seventh_term : a + 6 * d = 10) : 
    a + 10 * d = 110 / 7 := 
by
  sorry

end arithmetic_sequence_eleventh_term_l179_179131


namespace base7_and_base13_addition_l179_179897

def base7_to_nat (a b c : ℕ) : ℕ := a * 49 + b * 7 + c

def base13_to_nat (a b c : ℕ) : ℕ := a * 169 + b * 13 + c

theorem base7_and_base13_addition (a b c d e f : ℕ) :
  a = 5 → b = 3 → c = 6 → d = 4 → e = 12 → f = 5 →
  base7_to_nat a b c + base13_to_nat d e f = 1109 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  unfold base7_to_nat base13_to_nat
  sorry

end base7_and_base13_addition_l179_179897


namespace transfer_people_correct_equation_l179_179382

theorem transfer_people_correct_equation (A B x : ℕ) (h1 : A = 28) (h2 : B = 20) : 
  A + x = 2 * (B - x) := 
by sorry

end transfer_people_correct_equation_l179_179382


namespace find_divisor_l179_179609

-- Definitions based on the conditions
def is_divisor (d : ℕ) (a b k : ℕ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ (b - a) / n = k ∧ k = d

-- Problem statement
theorem find_divisor (a b k : ℕ) (H : b = 43 ∧ a = 10 ∧ k = 11) : ∃ d, d = 3 :=
by
  sorry

end find_divisor_l179_179609


namespace solution_set_contains_0_and_2_l179_179947

theorem solution_set_contains_0_and_2 (k : ℝ) : 
  ∀ x, ((1 + k^2) * x ≤ k^4 + 4) → (x = 0 ∨ x = 2) :=
by {
  sorry -- Proof is omitted
}

end solution_set_contains_0_and_2_l179_179947


namespace additional_girls_needed_l179_179863

theorem additional_girls_needed (initial_girls initial_boys additional_girls : ℕ)
  (h_initial_girls : initial_girls = 2)
  (h_initial_boys : initial_boys = 6)
  (h_fraction_goal : (initial_girls + additional_girls) = (5 * (initial_girls + initial_boys + additional_girls)) / 8) :
  additional_girls = 8 :=
by
  -- A placeholder for the proof
  sorry

end additional_girls_needed_l179_179863


namespace marbles_problem_l179_179819

theorem marbles_problem (initial_marble_tyrone : ℕ) (initial_marble_eric : ℕ) (x : ℝ)
  (h1 : initial_marble_tyrone = 125)
  (h2 : initial_marble_eric = 25)
  (h3 : initial_marble_tyrone - x = 3 * (initial_marble_eric + x)) :
  x = 12.5 := 
sorry

end marbles_problem_l179_179819


namespace find_function_l179_179067

def satisfies_functional_eqn (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * f y) = f (x * y^2) - 2 * x^2 * f y - f x - 1

theorem find_function (f : ℝ → ℝ) :
  satisfies_functional_eqn f → (∀ y : ℝ, f y = y^2 - 1) :=
by
  intro h
  sorry

end find_function_l179_179067


namespace sum_of_first_six_primes_l179_179642

theorem sum_of_first_six_primes : (2 + 3 + 5 + 7 + 11 + 13) = 41 :=
by
  sorry

end sum_of_first_six_primes_l179_179642


namespace range_of_x_if_cos2_gt_sin2_l179_179356

theorem range_of_x_if_cos2_gt_sin2 (x : ℝ) (h1 : x ∈ Set.Icc 0 Real.pi) (h2 : Real.cos x ^ 2 > Real.sin x ^ 2) :
  x ∈ Set.Ico 0 (Real.pi / 4) ∪ Set.Ioc (3 * Real.pi / 4) Real.pi :=
by
  sorry

end range_of_x_if_cos2_gt_sin2_l179_179356


namespace g_expression_l179_179992

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := sorry

theorem g_expression :
  (∀ x : ℝ, g (x + 2) = f x) → ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end g_expression_l179_179992


namespace part_i_part_ii_l179_179583

noncomputable def f (x a : ℝ) := |x - a|

theorem part_i :
  (∀ (x : ℝ), (f x 1) ≥ (|x + 1| + 1) ↔ x ≤ -0.5) :=
sorry

theorem part_ii :
  (∀ (x a : ℝ), (f x a) + 3 * x ≤ 0 → { x | x ≤ -1 } ⊆ { x | (f x a) + 3 * x ≤ 0 }) →
  (∀ (a : ℝ), (0 ≤ a ∧ a ≤ 2) ∨ (-4 ≤ a ∧ a < 0)) :=
sorry

end part_i_part_ii_l179_179583


namespace min_sum_of_factors_of_72_l179_179245

theorem min_sum_of_factors_of_72 (a b: ℤ) (h: a * b = 72) : a + b = -73 :=
sorry

end min_sum_of_factors_of_72_l179_179245


namespace find_range_of_a_l179_179269

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

end find_range_of_a_l179_179269


namespace difference_area_octagon_shaded_l179_179001

-- Definitions based on the given conditions
def radius : ℝ := 10
def pi_value : ℝ := 3.14

-- Lean statement for the given proof problem
theorem difference_area_octagon_shaded :
  ∃ S_octagon S_shaded, 
    10^2 * pi_value = 314 ∧
    (20 / 2^0.5)^2 = 200 ∧
    S_octagon = 200 - 114 ∧ -- transposed to reverse engineering step
    S_shaded = 28 ∧ -- needs refinement
    S_octagon - S_shaded = 86 :=
sorry

end difference_area_octagon_shaded_l179_179001


namespace simplify_expression_l179_179492

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (18 * x^3) * (4 * x^2) * (1 / (2 * x)^3) = 9 * x^2 :=
by
  sorry

end simplify_expression_l179_179492


namespace billy_weight_l179_179875

variable (B Bd C D : ℝ)

theorem billy_weight
  (h1 : B = Bd + 9)
  (h2 : Bd = C + 5)
  (h3 : C = D - 8)
  (h4 : C = 145)
  (h5 : D = 2 * Bd) :
  B = 85.5 :=
by
  sorry

end billy_weight_l179_179875


namespace guests_not_eating_brownies_ala_mode_l179_179488

theorem guests_not_eating_brownies_ala_mode (total_brownies : ℕ) (eaten_brownies : ℕ) (eaten_scoops : ℕ)
    (scoops_per_serving : ℕ) (scoops_per_tub : ℕ) (tubs_eaten : ℕ) : 
    total_brownies = 32 → eaten_brownies = 28 → eaten_scoops = 48 → scoops_per_serving = 2 → scoops_per_tub = 8 → tubs_eaten = 6 → (eaten_scoops - eaten_brownies * scoops_per_serving) / scoops_per_serving = 4 :=
by
  intros
  sorry

end guests_not_eating_brownies_ala_mode_l179_179488


namespace radishes_difference_l179_179062

theorem radishes_difference 
    (total_radishes : ℕ)
    (groups : ℕ)
    (first_basket : ℕ)
    (second_basket : ℕ)
    (total_radishes_eq : total_radishes = 88)
    (groups_eq : groups = 4)
    (first_basket_eq : first_basket = 37)
    (second_basket_eq : second_basket = total_radishes - first_basket)
  : second_basket - first_basket = 14 :=
by
  sorry

end radishes_difference_l179_179062


namespace minute_hand_distance_l179_179864

noncomputable def distance_traveled (length_of_minute_hand : ℝ) (time_duration : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * length_of_minute_hand
  let revolutions := time_duration / 60
  circumference * revolutions

theorem minute_hand_distance :
  distance_traveled 8 45 = 12 * Real.pi :=
by
  sorry

end minute_hand_distance_l179_179864


namespace reducible_fraction_least_n_l179_179320

theorem reducible_fraction_least_n : ∃ n : ℕ, (0 < n) ∧ (n-15 > 0) ∧ (gcd (n-15) (3*n+4) > 1) ∧
  (∀ m : ℕ, (0 < m) ∧ (m-15 > 0) ∧ (gcd (m-15) (3*m+4) > 1) → n ≤ m) :=
by
  sorry

end reducible_fraction_least_n_l179_179320


namespace complex_projective_form_and_fixed_points_l179_179097

noncomputable def complex_projective_transformation (a b c d : ℂ) (z : ℂ) : ℂ :=
  (a * z + b) / (c * z + d)

theorem complex_projective_form_and_fixed_points (a b c d : ℂ) (h : d ≠ 0) :
  (∃ (f : ℂ → ℂ), ∀ z, f z = complex_projective_transformation a b c d z)
  ∧ ∃ (z₁ z₂ : ℂ), complex_projective_transformation a b c d z₁ = z₁ ∧ complex_projective_transformation a b c d z₂ = z₂ :=
by
  -- omitted proof, this is just the statement
  sorry

end complex_projective_form_and_fixed_points_l179_179097


namespace students_not_picked_l179_179956

theorem students_not_picked (total_students groups group_size : ℕ) (h1 : total_students = 64)
(h2 : groups = 4) (h3 : group_size = 7) :
total_students - groups * group_size = 36 :=
by
  sorry

end students_not_picked_l179_179956


namespace product_of_roots_l179_179476

theorem product_of_roots : ∃ (x : ℕ), x = 45 ∧ (∃ a b c : ℕ, a ^ 3 = 27 ∧ b ^ 4 = 81 ∧ c ^ 2 = 25 ∧ x = a * b * c) := 
sorry

end product_of_roots_l179_179476


namespace train_speed_l179_179545

theorem train_speed (distance time : ℝ) (h₀ : distance = 180) (h₁ : time = 9) : 
  ((distance / 1000) / (time / 3600)) = 72 :=
by 
  -- below statement will bring the remainder of the setup and will be proved without the steps
  sorry

end train_speed_l179_179545


namespace number_99_in_column_4_l179_179886

-- Definition of the arrangement rule
def column_of (num : ℕ) : ℕ :=
  ((num % 10) + 4) / 2 % 5 + 1

theorem number_99_in_column_4 : 
  column_of 99 = 4 :=
by
  sorry

end number_99_in_column_4_l179_179886


namespace distance_between_D_and_E_l179_179909

theorem distance_between_D_and_E 
  (A B C D E P : Type)
  (d_AB : ℕ) (d_BC : ℕ) (d_AC : ℕ) (d_PC : ℕ) 
  (AD_parallel_BC : Prop) (AB_parallel_CE : Prop) 
  (distance_DE : ℕ) :
  d_AB = 15 →
  d_BC = 18 → 
  d_AC = 21 → 
  d_PC = 7 → 
  AD_parallel_BC →
  AB_parallel_CE →
  distance_DE = 15 :=
by
  sorry

end distance_between_D_and_E_l179_179909


namespace cookie_radius_l179_179011

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 13 := 
sorry

end cookie_radius_l179_179011


namespace expand_polynomial_l179_179414

theorem expand_polynomial :
  (5 * x^2 + 3 * x - 4) * 3 * x^3 = 15 * x^5 + 9 * x^4 - 12 * x^3 := 
by
  sorry

end expand_polynomial_l179_179414


namespace digits_divisible_by_101_l179_179439

theorem digits_divisible_by_101 :
  ∃ x y : ℕ, x < 10 ∧ y < 10 ∧ (2013 * 100 + 10 * x + y) % 101 = 0 ∧ x = 9 ∧ y = 4 := by
  sorry

end digits_divisible_by_101_l179_179439


namespace average_hit_targets_value_average_hit_targets_ge_half_l179_179938

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n)^n)

theorem average_hit_targets_value (n : ℕ) :
  average_hit_targets n = n * (1 - (1 - 1 / n)^n) :=
by sorry

theorem average_hit_targets_ge_half (n : ℕ) :
  average_hit_targets n >= n / 2 :=
by sorry

end average_hit_targets_value_average_hit_targets_ge_half_l179_179938


namespace math_problem_l179_179332

theorem math_problem
  (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h1 : p * q + r = 47)
  (h2 : q * r + p = 47)
  (h3 : r * p + q = 47) :
  p + q + r = 48 :=
sorry

end math_problem_l179_179332


namespace staircase_steps_180_toothpicks_l179_179351

-- Condition definition: total number of toothpicks for \( n \) steps is \( n(n + 1) \)
def total_toothpicks (n : ℕ) : ℕ := n * (n + 1)

-- Theorem statement: for 180 toothpicks, the number of steps \( n \) is 12
theorem staircase_steps_180_toothpicks : ∃ n : ℕ, total_toothpicks n = 180 ∧ n = 12 :=
by sorry

end staircase_steps_180_toothpicks_l179_179351


namespace find_y_l179_179117

theorem find_y (y : ℝ) (h₁ : (y^2 - 7*y + 12) / (y - 3) + (3*y^2 + 5*y - 8) / (3*y - 1) = -8) : y = -6 :=
sorry

end find_y_l179_179117


namespace value_of_expression_l179_179635

theorem value_of_expression {a b : ℝ} (h1 : 2 * a^2 + 6 * a - 14 = 0) (h2 : 2 * b^2 + 6 * b - 14 = 0) :
  (2 * a - 3) * (4 * b - 6) = -2 :=
by
  sorry

end value_of_expression_l179_179635


namespace surface_area_invisible_block_l179_179937

-- Define the given areas of the seven blocks
def A1 := 148
def A2 := 46
def A3 := 72
def A4 := 28
def A5 := 88
def A6 := 126
def A7 := 58

-- Define total surface areas of the black and white blocks
def S_black := A1 + A2 + A3 + A4
def S_white := A5 + A6 + A7

-- Define the proof problem
theorem surface_area_invisible_block : S_black - S_white = 22 :=
by
  -- This sorry allows the Lean statement to build successfully
  sorry

end surface_area_invisible_block_l179_179937


namespace find_k_solution_l179_179803

theorem find_k_solution 
  (k : ℝ)
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) : 
  k = 2 :=
sorry

end find_k_solution_l179_179803


namespace marina_max_socks_l179_179391

theorem marina_max_socks (white black : ℕ) (hw : white = 8) (hb : black = 15) :
  ∃ n, n = 17 ∧ ∀ w b, w + b = n → 0 ≤ w ∧ 0 ≤ b ∧ w ≤ black ∧ b ≤ black ∧ w ≤ white ∧ b ≤ black → b > w :=
sorry

end marina_max_socks_l179_179391


namespace total_matches_played_l179_179212

theorem total_matches_played
  (avg_runs_first_20: ℕ) (num_first_20: ℕ) (avg_runs_next_10: ℕ) (num_next_10: ℕ) (overall_avg: ℕ) (total_matches: ℕ) :
  avg_runs_first_20 = 40 →
  num_first_20 = 20 →
  avg_runs_next_10 = 13 →
  num_next_10 = 10 →
  overall_avg = 31 →
  (num_first_20 + num_next_10 = total_matches) →
  total_matches = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_matches_played_l179_179212


namespace smallest_integer_consecutive_set_l179_179799

theorem smallest_integer_consecutive_set 
(n : ℤ) (h : 7 * n + 21 > 4 * n) : n > -7 :=
by
  sorry

end smallest_integer_consecutive_set_l179_179799


namespace seth_spent_more_on_ice_cream_l179_179276

-- Definitions based on the conditions
def cartons_ice_cream := 20
def cartons_yogurt := 2
def cost_per_carton_ice_cream := 6
def cost_per_carton_yogurt := 1

-- Theorem statement
theorem seth_spent_more_on_ice_cream :
  (cartons_ice_cream * cost_per_carton_ice_cream) - (cartons_yogurt * cost_per_carton_yogurt) = 118 :=
by
  sorry

end seth_spent_more_on_ice_cream_l179_179276


namespace best_fitting_model_l179_179605

-- Define the \(R^2\) values for each model
def R2_Model1 : ℝ := 0.75
def R2_Model2 : ℝ := 0.90
def R2_Model3 : ℝ := 0.25
def R2_Model4 : ℝ := 0.55

-- State that Model 2 is the best fitting model
theorem best_fitting_model : R2_Model2 = max (max R2_Model1 R2_Model2) (max R2_Model3 R2_Model4) :=
by -- Proof skipped
  sorry

end best_fitting_model_l179_179605


namespace sum_of_series_eq_5_over_16_l179_179692

theorem sum_of_series_eq_5_over_16 :
  ∑' n : ℕ, (n + 1 : ℝ) / (5 : ℝ)^(n + 1) = 5 / 16 := by
  sorry

end sum_of_series_eq_5_over_16_l179_179692


namespace lesser_fraction_exists_l179_179248

theorem lesser_fraction_exists (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : x = 1/4 ∨ y = 1/4 := by
  sorry

end lesser_fraction_exists_l179_179248


namespace remainder_4063_div_97_l179_179369

theorem remainder_4063_div_97 : 4063 % 97 = 86 := 
by sorry

end remainder_4063_div_97_l179_179369


namespace range_of_a_l179_179502

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then 2^(|x - a|) else x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ f 1 a) ↔ (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l179_179502


namespace pencils_initial_count_l179_179039

theorem pencils_initial_count (pencils_initially: ℕ) :
  (∀ n, n > 0 → n < 36 → 36 % n = 1) →
  pencils_initially + 30 = 36 → 
  pencils_initially = 6 :=
by
  intro h hn
  sorry

end pencils_initial_count_l179_179039


namespace sum_of_four_squares_l179_179447

theorem sum_of_four_squares (a b c : ℕ) 
    (h1 : 2 * a + b + c = 27)
    (h2 : 2 * b + a + c = 25)
    (h3 : 3 * c + a = 39) : 4 * c = 44 := 
  sorry

end sum_of_four_squares_l179_179447


namespace more_chickens_than_chicks_l179_179421

-- Let's define the given conditions
def total : Nat := 821
def chicks : Nat := 267

-- The statement we need to prove
theorem more_chickens_than_chicks : (total - chicks) - chicks = 287 :=
by
  -- This is needed for the proof and not part of conditions
  -- Add sorry as a placeholder for proof steps 
  sorry

end more_chickens_than_chicks_l179_179421


namespace ott_fraction_part_l179_179615

noncomputable def fractional_part_of_group_money (x : ℝ) (M L N P : ℝ) :=
  let total_initial := M + L + N + P + 2
  let money_received_by_ott := 4 * x
  let ott_final_money := 2 + money_received_by_ott
  let total_final := total_initial + money_received_by_ott
  (ott_final_money / total_final) = (3 / 14)

theorem ott_fraction_part (x : ℝ) (M L N P : ℝ)
    (hM : M = 6 * x) (hL : L = 5 * x) (hN : N = 4 * x) (hP : P = 7 * x) :
    fractional_part_of_group_money x M L N P :=
by
  sorry

end ott_fraction_part_l179_179615


namespace length_of_field_l179_179600

-- Define the problem conditions
variables (width length : ℕ)
  (pond_area field_area : ℕ)
  (h1 : length = 2 * width)
  (h2 : pond_area = 64)
  (h3 : pond_area = field_area / 8)

-- Define the proof problem
theorem length_of_field : length = 32 :=
by
  -- We'll provide the proof later
  sorry

end length_of_field_l179_179600


namespace necessary_and_sufficient_condition_l179_179665

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (|a + b| / (|a| + |b|) ≤ 1) ↔ (a^2 + b^2 ≠ 0) :=
sorry

end necessary_and_sufficient_condition_l179_179665


namespace minimum_value_of_expression_l179_179620

theorem minimum_value_of_expression :
  ∀ x y : ℝ, x^2 - x * y + y^2 ≥ 0 :=
by
  sorry

end minimum_value_of_expression_l179_179620


namespace no_solution_for_m_l179_179042

theorem no_solution_for_m (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (m : ℕ) (h3 : (ab)^2015 = (a^2 + b^2)^m) : false := 
sorry

end no_solution_for_m_l179_179042


namespace average_remaining_five_l179_179707

theorem average_remaining_five (S S4 S5 : ℕ) 
  (h1 : S = 18 * 9) 
  (h2 : S4 = 8 * 4) 
  (h3 : S5 = S - S4) 
  (h4 : S5 / 5 = 26) : 
  average_of_remaining_5 = 26 :=
by 
  sorry


end average_remaining_five_l179_179707


namespace part_I_part_II_l179_179639

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem part_I (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  ∃ x ∈ Set.Ioo m (m + 1), ∀ y ∈ Set.Ioo m (m + 1), f y ≤ f x := sorry

theorem part_II (x : ℝ) (h : 1 < x) :
  (x + 1) * (x + Real.exp (-x)) * f x > 2 * (1 + 1 / Real.exp 1) := sorry

end part_I_part_II_l179_179639


namespace eustace_age_in_3_years_l179_179241

variable (E M : ℕ)

theorem eustace_age_in_3_years
  (h1 : E = 2 * M)
  (h2 : M + 3 = 21) :
  E + 3 = 39 :=
sorry

end eustace_age_in_3_years_l179_179241


namespace prime_fraction_sum_l179_179016

theorem prime_fraction_sum (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
    (h : a + b + c + a * b * c = 99) :
    |(1 / a : ℚ) - (1 / b : ℚ)| + |(1 / b : ℚ) - (1 / c : ℚ)| + |(1 / c : ℚ) - (1 / a : ℚ)| = 9 / 11 := 
sorry

end prime_fraction_sum_l179_179016


namespace function_has_property_T_l179_179306

noncomputable def property_T (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ (f a ≠ 0) ∧ (f b ≠ 0) ∧ (f a * f b = -1)

theorem function_has_property_T : property_T (fun x => 1 + x * Real.log x) :=
sorry

end function_has_property_T_l179_179306


namespace men_required_l179_179334

variable (m w : ℝ) -- Work done by one man and one woman in one day respectively
variable (x : ℝ) -- Number of men

-- Conditions from the problem
def condition1 (m w : ℝ) (x : ℝ) : Prop :=
  x * m = 12 * w

def condition2 (m w : ℝ) : Prop :=
  (6 * m + 11 * w) * 12 = 1

-- Proving that the number of men required to do the work in 20 days is x
theorem men_required (m w : ℝ) (x : ℝ) (h1 : condition1 m w x) (h2 : condition2 m w) : 
  (∃ x, condition1 m w x ∧ condition2 m w) := 
sorry

end men_required_l179_179334


namespace cos_value_of_angle_l179_179624

theorem cos_value_of_angle (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
by
  sorry

end cos_value_of_angle_l179_179624


namespace range_of_quadratic_function_l179_179457

theorem range_of_quadratic_function :
  ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), -x^2 - 4 * x + 1 ∈ Set.Icc (-11) (5) :=
by
  sorry

end range_of_quadratic_function_l179_179457


namespace younger_person_age_l179_179114

theorem younger_person_age (e y : ℕ) 
  (h1: e = y + 20)
  (h2: e - 10 = 5 * (y - 10)) : 
  y = 15 := 
by
  sorry

end younger_person_age_l179_179114


namespace truncated_cone_volume_l179_179506

theorem truncated_cone_volume :
  let R := 10
  let r := 5
  let h_t := 10
  let V_large := (1/3:Real) * Real.pi * (R^2) * (20)
  let V_small := (1/3:Real) * Real.pi * (r^2) * (10)
  (V_large - V_small) = (1750/3) * Real.pi :=
by
  sorry

end truncated_cone_volume_l179_179506


namespace field_length_proof_l179_179511

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

end field_length_proof_l179_179511


namespace Frank_is_14_l179_179725

variable {d e f : ℕ}

theorem Frank_is_14
  (h1 : d + e + f = 30)
  (h2 : f - 5 = d)
  (h3 : e + 2 = 3 * (d + 2) / 4) :
  f = 14 :=
sorry

end Frank_is_14_l179_179725


namespace intersection_of_diagonals_l179_179968

-- Define the four lines based on the given conditions
def line1 (k b x : ℝ) : ℝ := k*x + b
def line2 (k b x : ℝ) : ℝ := k*x - b
def line3 (m b x : ℝ) : ℝ := m*x + b
def line4 (m b x : ℝ) : ℝ := m*x - b

-- Define a function to represent the problem
noncomputable def point_of_intersection_of_diagonals (k m b : ℝ) : ℝ × ℝ :=
(0, 0)

-- State the theorem to be proved
theorem intersection_of_diagonals (k m b : ℝ) :
  point_of_intersection_of_diagonals k m b = (0, 0) :=
sorry

end intersection_of_diagonals_l179_179968


namespace travis_total_cost_l179_179917

namespace TravelCost

def cost_first_leg : ℝ := 1500
def discount_first_leg : ℝ := 0.25
def fees_first_leg : ℝ := 100

def cost_second_leg : ℝ := 800
def discount_second_leg : ℝ := 0.20
def fees_second_leg : ℝ := 75

def cost_third_leg : ℝ := 1200
def discount_third_leg : ℝ := 0.35
def fees_third_leg : ℝ := 120

def discounted_cost (cost : ℝ) (discount : ℝ) : ℝ :=
  cost - (cost * discount)

def total_leg_cost (cost : ℝ) (discount : ℝ) (fees : ℝ) : ℝ :=
  (discounted_cost cost discount) + fees

def total_journey_cost : ℝ :=
  total_leg_cost cost_first_leg discount_first_leg fees_first_leg + 
  total_leg_cost cost_second_leg discount_second_leg fees_second_leg + 
  total_leg_cost cost_third_leg discount_third_leg fees_third_leg

theorem travis_total_cost : total_journey_cost = 2840 := by
  sorry

end TravelCost

end travis_total_cost_l179_179917


namespace initial_machines_count_l179_179497

theorem initial_machines_count (M : ℕ) (h1 : M * 8 = 8 * 1) (h2 : 72 * 6 = 12 * 2) : M = 64 :=
by
  sorry

end initial_machines_count_l179_179497


namespace range_of_a_l179_179885

def A (a : ℝ) := ({-1, 0, a} : Set ℝ)
def B := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B ≠ ∅) : 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l179_179885


namespace medium_supermarkets_in_sample_l179_179239

-- Definitions of the conditions
def total_supermarkets : ℕ := 200 + 400 + 1400
def prop_medium_supermarkets : ℚ := 400 / total_supermarkets
def sample_size : ℕ := 100

-- Problem: Prove that the number of medium-sized supermarkets in the sample is 20.
theorem medium_supermarkets_in_sample : 
  (sample_size * prop_medium_supermarkets) = 20 :=
by
  sorry

end medium_supermarkets_in_sample_l179_179239


namespace side_length_of_square_base_l179_179868

theorem side_length_of_square_base (area : ℝ) (slant_height : ℝ) (s : ℝ) (h : slant_height = 40) (a : area = 160) : s = 8 :=
by sorry

end side_length_of_square_base_l179_179868


namespace rhombus_has_perpendicular_diagonals_and_rectangle_not_l179_179859

-- Definitions based on conditions (a))
def rhombus (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_perpendicular : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_perpendicular

def rectangle (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_equal : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_equal

-- Theorem to prove (c))
theorem rhombus_has_perpendicular_diagonals_and_rectangle_not 
  (rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular : Prop)
  (rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal : Prop) :
  rhombus rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular → 
  rectangle rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal → 
  rhombus_diagonals_perpendicular ∧ ¬(rectangle (rectangle_sides_equal) (rectangle_diagonals_bisect) (rhombus_diagonals_perpendicular)) :=
sorry

end rhombus_has_perpendicular_diagonals_and_rectangle_not_l179_179859


namespace max_sum_cubes_l179_179452

theorem max_sum_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  a^3 + b^3 + c^3 + d^3 ≤ 8 :=
sorry

end max_sum_cubes_l179_179452


namespace modulo_power_l179_179553

theorem modulo_power (a n : ℕ) (p : ℕ) (hn_pos : 0 < n) (hp_odd : p % 2 = 1)
  (hp_prime : Nat.Prime p) (h : a^p ≡ 1 [MOD p^n]) : a ≡ 1 [MOD p^(n-1)] :=
by
  sorry

end modulo_power_l179_179553


namespace DollOutfit_l179_179515

variables (VeraDress OlyaCoat VeraCoat NinaCoat : Prop)
axiom FirstAnswer : (VeraDress ∧ ¬OlyaCoat) ∨ (¬VeraDress ∧ OlyaCoat)
axiom SecondAnswer : (VeraCoat ∧ ¬NinaCoat) ∨ (¬VeraCoat ∧ NinaCoat)
axiom OnlyOneTrueFirstAnswer : (VeraDress ∨ OlyaCoat) ∧ ¬(VeraDress ∧ OlyaCoat)
axiom OnlyOneTrueSecondAnswer : (VeraCoat ∨ NinaCoat) ∧ ¬(VeraCoat ∧ NinaCoat)

theorem DollOutfit :
  VeraDress ∧ NinaCoat ∧ ¬OlyaCoat ∧ ¬VeraCoat ∧ ¬NinaCoat :=
sorry

end DollOutfit_l179_179515


namespace covering_percentage_77_l179_179445

-- Definition section for conditions
def radius_of_circle (r a : ℝ) := 2 * r * Real.pi = 4 * a
def center_coincide (a b : ℝ) := a = b

-- Theorem to be proven
theorem covering_percentage_77
  (r a : ℝ)
  (h_radius: radius_of_circle r a)
  (h_center: center_coincide 0 0) : 
  (r^2 * Real.pi - 0.7248 * r^2) / (r^2 * Real.pi) * 100 = 77 := by
  sorry

end covering_percentage_77_l179_179445


namespace inequality_proof_l179_179363

open Real

theorem inequality_proof {x y : ℝ} (hx : x < 0) (hy : y < 0) : 
    (x ^ 4 / y ^ 4) + (y ^ 4 / x ^ 4) - (x ^ 2 / y ^ 2) - (y ^ 2 / x ^ 2) + (x / y) + (y / x) >= 2 := 
by
    sorry

end inequality_proof_l179_179363


namespace ratio_a_to_d_l179_179554

theorem ratio_a_to_d (a b c d : ℕ) 
  (h1 : a * 4 = b * 3) 
  (h2 : b * 9 = c * 7) 
  (h3 : c * 7 = d * 5) : 
  a * 3 = d := 
sorry

end ratio_a_to_d_l179_179554


namespace height_of_flagpole_l179_179285

theorem height_of_flagpole 
  (house_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ) 
  (flagpole_shadow : ℝ) (house_height : ℝ)
  (h1 : house_shadow = 70)
  (h2 : tree_height = 28)
  (h3 : tree_shadow = 40)
  (h4 : flagpole_shadow = 25)
  (h5 : house_height = (tree_height * house_shadow) / tree_shadow) :
  round ((house_height * flagpole_shadow / house_shadow) : ℝ) = 18 := 
by
  sorry

end height_of_flagpole_l179_179285


namespace meaningful_expression_l179_179783

theorem meaningful_expression (x : ℝ) : (1 / Real.sqrt (x + 2) > 0) → (x > -2) := 
sorry

end meaningful_expression_l179_179783


namespace circle_reflection_l179_179370

-- Definitions provided in conditions
def initial_center : ℝ × ℝ := (6, -5)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.snd, p.fst)
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.fst, p.snd)

-- The final statement we need to prove
theorem circle_reflection :
  reflect_y_axis (reflect_y_eq_x initial_center) = (5, 6) :=
by
  -- By reflecting the point (6, -5) over y = x and then over the y-axis, we should get (5, 6)
  sorry

end circle_reflection_l179_179370


namespace remove_one_piece_l179_179121

theorem remove_one_piece (pieces : Finset (Fin 8 × Fin 8)) (h_card : pieces.card = 15)
  (h_row : ∀ r : Fin 8, ∃ c, (r, c) ∈ pieces)
  (h_col : ∀ c : Fin 8, ∃ r, (r, c) ∈ pieces) :
  ∃ pieces' : Finset (Fin 8 × Fin 8), pieces'.card = 14 ∧ 
  (∀ r : Fin 8, ∃ c, (r, c) ∈ pieces') ∧ 
  (∀ c : Fin 8, ∃ r, (r, c) ∈ pieces') :=
sorry

end remove_one_piece_l179_179121


namespace find_x_l179_179901

theorem find_x (x : ℝ) (a b : ℝ) (h₀ : a * b = 4 * a - 2 * b)
  (h₁ : 3 * (6 * x) = -2) :
  x = 17 / 2 :=
by
  sorry

end find_x_l179_179901


namespace blue_hat_cost_l179_179550

variable (B : ℕ)
variable (totalHats : ℕ := 85)
variable (greenHatCost : ℕ := 7)
variable (greenHatsBought : ℕ := 38)
variable (totalCost : ℕ := 548)

theorem blue_hat_cost 
(h1 : greenHatsBought = 38) 
(h2 : totalHats = 85) 
(h3 : greenHatCost = 7)
(h4 : totalCost = 548) :
  let totalGreenHatCost := greenHatCost * greenHatsBought
  let totalBlueHatCost := totalCost - totalGreenHatCost
  let totalBlueHatsBought := totalHats - greenHatsBought
  B = totalBlueHatCost / totalBlueHatsBought := by
  sorry

end blue_hat_cost_l179_179550


namespace no_3_digit_numbers_sum_27_even_l179_179539

-- Define the conditions
def is_digit_sum_27 (n : ℕ) : Prop :=
  (n ≥ 100 ∧ n < 1000) ∧ ((n / 100) + (n / 10 % 10) + (n % 10) = 27)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Define the theorem
theorem no_3_digit_numbers_sum_27_even :
  ¬ ∃ n : ℕ, is_digit_sum_27 n ∧ is_even n :=
by
  sorry

end no_3_digit_numbers_sum_27_even_l179_179539


namespace intersection_complement_eq_l179_179695

variable (U : Set ℝ := Set.univ)
variable (A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 })
variable (B : Set ℝ := { x | x > -1 })

theorem intersection_complement_eq :
  A ∩ (U \ B) = { x | -2 ≤ x ∧ x ≤ -1 } :=
by {
  sorry
}

end intersection_complement_eq_l179_179695


namespace sum_of_all_numbers_after_n_steps_l179_179548

def initial_sum : ℕ := 2

def sum_after_step (n : ℕ) : ℕ :=
  2 * 3^n

theorem sum_of_all_numbers_after_n_steps (n : ℕ) : 
  sum_after_step n = 2 * 3^n :=
by sorry

end sum_of_all_numbers_after_n_steps_l179_179548


namespace problem_expression_value_l179_179870

theorem problem_expression_value :
  (100 - (3010 - 301)) + (3010 - (301 - 100)) = 200 :=
by
  sorry

end problem_expression_value_l179_179870


namespace fraction_comparisons_l179_179503

theorem fraction_comparisons :
  (1 / 8 : ℝ) * (3 / 7) < (1 / 8) ∧ 
  (9 / 8 : ℝ) * (1 / 5) > (9 / 8) * (1 / 8) ∧ 
  (2 / 3 : ℝ) < (2 / 3) / (6 / 11) := by
    sorry

end fraction_comparisons_l179_179503


namespace lcm_of_54_and_198_l179_179469

theorem lcm_of_54_and_198 : Nat.lcm 54 198 = 594 :=
by
  have fact1 : 54 = 2 ^ 1 * 3 ^ 3 := by norm_num
  have fact2 : 198 = 2 ^ 1 * 3 ^ 2 * 11 ^ 1 := by norm_num
  have lcm_prime : Nat.lcm 54 198 = 594 := by
    sorry -- Proof skipped
  exact lcm_prime

end lcm_of_54_and_198_l179_179469


namespace vector_subtraction_result_l179_179188

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_subtraction_result :
  2 • a - b = (7, -2) :=
by
  simp [a, b]
  sorry

end vector_subtraction_result_l179_179188


namespace install_time_per_window_l179_179242

/-- A new building needed 14 windows. The builder had already installed 8 windows.
    It will take the builder 48 hours to install the rest of the windows. -/
theorem install_time_per_window (total_windows installed_windows remaining_install_time : ℕ)
  (h_total : total_windows = 14)
  (h_installed : installed_windows = 8)
  (h_remaining_time : remaining_install_time = 48) :
  (remaining_install_time / (total_windows - installed_windows)) = 8 :=
by
  -- Insert usual proof steps here
  sorry

end install_time_per_window_l179_179242


namespace units_digit_G_1000_l179_179167

def modified_fermat_number (n : ℕ) : ℕ := 5^(5^n) + 6

theorem units_digit_G_1000 : (modified_fermat_number 1000) % 10 = 1 :=
by
  -- The proof goes here
  sorry

end units_digit_G_1000_l179_179167


namespace find_a12_l179_179965

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- The Lean statement for the problem
theorem find_a12 (h_seq : arithmetic_sequence a d)
  (h_cond1 : a 7 + a 9 = 16) (h_cond2 : a 4 = 1) : 
  a 12 = 15 :=
sorry

end find_a12_l179_179965


namespace inequality_proof_l179_179353

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
sorry

end inequality_proof_l179_179353


namespace bryden_payment_l179_179647

theorem bryden_payment :
  (let face_value := 0.25
   let quarters := 6
   let collector_multiplier := 16
   let discount := 0.10
   let initial_payment := collector_multiplier * (quarters * face_value)
   let final_payment := initial_payment - (initial_payment * discount)
   final_payment = 21.6) :=
by
  sorry

end bryden_payment_l179_179647


namespace mean_of_other_two_numbers_l179_179978

theorem mean_of_other_two_numbers (a b c d e f g h : ℕ)
  (h_tuple : a = 1871 ∧ b = 2011 ∧ c = 2059 ∧ d = 2084 ∧ e = 2113 ∧ f = 2167 ∧ g = 2198 ∧ h = 2210)
  (h_mean : (a + b + c + d + e + f) / 6 = 2100) :
  ((g + h) / 2 : ℚ) = 2056.5 :=
by
  sorry

end mean_of_other_two_numbers_l179_179978


namespace num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l179_179102

def num_valid_pairs : Nat := 34

def num_valid_first_digits : Nat := 6

def num_valid_last_digits : Nat := 10

theorem num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10 :
  (num_valid_first_digits * num_valid_pairs * num_valid_last_digits) = 2040 :=
by
  sorry

end num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l179_179102


namespace difference_of_two_numbers_l179_179894

theorem difference_of_two_numbers 
(x y : ℝ) 
(h1 : x + y = 20) 
(h2 : x^2 - y^2 = 160) : 
  x - y = 8 := 
by 
  sorry

end difference_of_two_numbers_l179_179894


namespace number_of_crosswalks_per_intersection_l179_179552

theorem number_of_crosswalks_per_intersection 
  (num_intersections : Nat) 
  (total_lines : Nat) 
  (lines_per_crosswalk : Nat) 
  (h1 : num_intersections = 5) 
  (h2 : total_lines = 400) 
  (h3 : lines_per_crosswalk = 20) :
  (total_lines / lines_per_crosswalk) / num_intersections = 4 :=
by
  -- Proof steps can be inserted here
  sorry

end number_of_crosswalks_per_intersection_l179_179552


namespace gardener_payment_l179_179034

theorem gardener_payment (total_cost : ℕ) (rect_area : ℕ) (rect_side1 : ℕ) (rect_side2 : ℕ)
                         (square1_area : ℕ) (square2_area : ℕ) (cost_per_are : ℕ) :
  total_cost = 570 →
  rect_area = 600 → rect_side1 = 20 → rect_side2 = 30 →
  square1_area = 400 → square2_area = 900 →
  cost_per_are * (rect_area + square1_area + square2_area) / 100 = total_cost →
  cost_per_are = 30 →
  ∃ (rect_payment : ℕ) (square1_payment : ℕ) (square2_payment : ℕ),
    rect_payment = 6 * cost_per_are ∧
    square1_payment = 4 * cost_per_are ∧
    square2_payment = 9 * cost_per_are ∧
    rect_payment + square1_payment + square2_payment = total_cost :=
by
  intros
  sorry

end gardener_payment_l179_179034


namespace steve_can_answer_38_questions_l179_179176

theorem steve_can_answer_38_questions (total_questions S : ℕ) 
  (h1 : total_questions = 45)
  (h2 : total_questions - S = 7) :
  S = 38 :=
by {
  -- The proof goes here
  sorry
}

end steve_can_answer_38_questions_l179_179176


namespace permutations_count_l179_179081

-- Define the conditions
variable (n : ℕ)
variable (a : Fin n → ℕ)

-- Define the main proposition
theorem permutations_count (hn : 2 ≤ n) (h_perm : ∀ k : Fin n, a k ≥ k.val - 2) :
  ∃! L, L = 2 * 3 ^ (n - 2) :=
by
  sorry

end permutations_count_l179_179081


namespace annual_percentage_increase_20_l179_179602

variable (P0 P1 : ℕ) (r : ℚ)

-- Population initial condition
def initial_population : Prop := P0 = 10000

-- Population after 1 year condition
def population_after_one_year : Prop := P1 = 12000

-- Define the annual percentage increase formula
def percentage_increase (P0 P1 : ℕ) : ℚ := ((P1 - P0 : ℚ) / P0) * 100

-- State the theorem
theorem annual_percentage_increase_20
  (h1 : initial_population P0)
  (h2 : population_after_one_year P1) :
  percentage_increase P0 P1 = 20 := by
  sorry

end annual_percentage_increase_20_l179_179602


namespace pyramid_volume_l179_179198

noncomputable def volume_of_pyramid (AB AD BD AE : ℝ) (p : AB = 9 ∧ AD = 10 ∧ BD = 11 ∧ AE = 10.5) : ℝ :=
  1 / 3 * (60 * (2 ^ (1 / 2))) * (5 * (2 ^ (1 / 2)))

theorem pyramid_volume (AB AD BD AE : ℝ) (h1 : AB = 9) (h2 : AD = 10) (h3 : BD = 11) (h4 : AE = 10.5)
  (V : ℝ) (hV : V = 200) : 
  volume_of_pyramid AB AD BD AE (⟨h1, ⟨h2, ⟨h3, h4⟩⟩⟩) = V :=
sorry

end pyramid_volume_l179_179198


namespace tan_alpha_two_implies_fraction_eq_three_fourths_l179_179154

variable {α : ℝ}

theorem tan_alpha_two_implies_fraction_eq_three_fourths (h1 : Real.tan α = 2) (h2 : Real.cos α ≠ 0) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

end tan_alpha_two_implies_fraction_eq_three_fourths_l179_179154


namespace magnitude_of_vec_sum_l179_179031

noncomputable def vec_a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def vec_b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))
noncomputable def vec_sum : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_vec_sum : magnitude vec_sum = Real.sqrt 7 := 
by 
  sorry

end magnitude_of_vec_sum_l179_179031


namespace math_problem_l179_179279

theorem math_problem : 
  ∃ (n m k : ℕ), 
    (∀ d : ℕ, d ∣ n → d > 0) ∧ 
    (n = m * 6^k) ∧
    (∀ d : ℕ, d ∣ m → 6 ∣ d → False) ∧
    (m + k = 60466182) ∧ 
    (n.factors.count 1 = 2023) :=
sorry

end math_problem_l179_179279


namespace parabola_properties_l179_179265

def parabola (a b x : ℝ) : ℝ :=
  a * x ^ 2 + b * x - 4

theorem parabola_properties :
  ∃ (a b : ℝ), (a = 2) ∧ (b = 2) ∧
  parabola a b (-2) = 0 ∧ 
  parabola a b (-1) = -4 ∧ 
  parabola a b 0 = -4 ∧ 
  parabola a b 1 = 0 ∧ 
  parabola a b 2 = 8 ∧ 
  parabola a b (-3) = 8 ∧ 
  (0, -4) ∈ {(x, y) | ∃ a b, y = parabola a b x} :=
sorry

end parabola_properties_l179_179265


namespace number_of_ways_to_fill_grid_l179_179772

noncomputable def totalWaysToFillGrid (S : Finset ℕ) : ℕ :=
  S.card.choose 5

theorem number_of_ways_to_fill_grid : totalWaysToFillGrid ({1, 2, 3, 4, 5, 6} : Finset ℕ) = 6 :=
by
  sorry

end number_of_ways_to_fill_grid_l179_179772


namespace rubber_duck_cost_l179_179406

theorem rubber_duck_cost 
  (price_large : ℕ)
  (num_regular : ℕ)
  (num_large : ℕ)
  (total_revenue : ℕ)
  (h1 : price_large = 5)
  (h2 : num_regular = 221)
  (h3 : num_large = 185)
  (h4 : total_revenue = 1588) :
  ∃ (cost_regular : ℕ), (num_regular * cost_regular + num_large * price_large = total_revenue) ∧ cost_regular = 3 :=
by
  exists 3
  sorry

end rubber_duck_cost_l179_179406


namespace problem_solution_exists_l179_179652

theorem problem_solution_exists (x : ℝ) (h : ∃ x, 2 * (3 * 5 - x) - x = -8) : x = 10 :=
sorry

end problem_solution_exists_l179_179652


namespace percentage_emails_moved_to_work_folder_l179_179124

def initialEmails : ℕ := 400
def trashedEmails : ℕ := initialEmails / 2
def remainingEmailsAfterTrash : ℕ := initialEmails - trashedEmails
def emailsLeftInInbox : ℕ := 120
def emailsMovedToWorkFolder : ℕ := remainingEmailsAfterTrash - emailsLeftInInbox

theorem percentage_emails_moved_to_work_folder :
  (emailsMovedToWorkFolder * 100 / remainingEmailsAfterTrash) = 40 := by
  sorry

end percentage_emails_moved_to_work_folder_l179_179124


namespace vacation_cost_split_l179_179312

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

end vacation_cost_split_l179_179312


namespace sum_num_den_252_l179_179485

theorem sum_num_den_252 (h : (252 : ℤ) / 100 = (63 : ℤ) / 25) : 63 + 25 = 88 :=
by
  sorry

end sum_num_den_252_l179_179485


namespace num_winners_is_4_l179_179908

variables (A B C D : Prop)

-- Conditions
axiom h1 : A → B
axiom h2 : B → (C ∨ ¬ A)
axiom h3 : ¬ D → (A ∧ ¬ C)
axiom h4 : D → A

-- Assumptions
axiom hA : A
axiom hD : D

-- Statement to prove
theorem num_winners_is_4 : A ∧ B ∧ C ∧ D :=
by {
  sorry
}

end num_winners_is_4_l179_179908


namespace part1_part2_l179_179874

-- Definitions of sets A and B
def A (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < a + 1 }
def B : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 ≥ 0 }

-- Proving the first condition
theorem part1 (a : ℝ) : (A a ∩ B = ∅) ∧ (A a ∪ B = Set.univ) ↔ a = 2 :=
by
  sorry

-- Proving the second condition
theorem part2 (a : ℝ) : (A a ⊆ B) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end part1_part2_l179_179874


namespace regular_tetrahedron_l179_179336

-- Define the types for points and tetrahedrons
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Tetrahedron :=
(A B C D : Point)
(insphere : Point)

-- Conditions
def sphere_touches_at_angle_bisectors (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_altitudes (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_medians (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

-- Main theorem statement
theorem regular_tetrahedron (T : Tetrahedron)
  (h1 : sphere_touches_at_angle_bisectors T)
  (h2 : sphere_touches_at_altitudes T)
  (h3 : sphere_touches_at_medians T) :
  T.A = T.B ∧ T.A = T.C ∧ T.A = T.D := 
sorry

end regular_tetrahedron_l179_179336


namespace two_digit_sum_reverse_l179_179479

theorem two_digit_sum_reverse (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
    (h₃ : 0 ≤ b) (h₄ : b ≤ 9)
    (h₅ : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
    (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end two_digit_sum_reverse_l179_179479


namespace cube_volume_in_pyramid_l179_179796

noncomputable def pyramid_base_side : ℝ := 2
noncomputable def equilateral_triangle_side : ℝ := 2 * Real.sqrt 2
noncomputable def equilateral_triangle_height : ℝ := Real.sqrt 6
noncomputable def cube_side : ℝ := Real.sqrt 6 / 2
noncomputable def cube_volume : ℝ := (Real.sqrt 6 / 2) ^ 3

theorem cube_volume_in_pyramid : cube_volume = 3 * Real.sqrt 6 / 4 :=
by
  sorry

end cube_volume_in_pyramid_l179_179796


namespace sum_of_exponents_l179_179679

-- Given product of integers from 1 to 15
def y := Nat.factorial 15

-- Prime exponent variables in the factorization of y
variables (i j k m n p q : ℕ)

-- Conditions
axiom h1 : y = 2^i * 3^j * 5^k * 7^m * 11^n * 13^p * 17^q 

-- Prove that the sum of the exponents equals 24
theorem sum_of_exponents :
  i + j + k + m + n + p + q = 24 := 
sorry

end sum_of_exponents_l179_179679


namespace gcd_lcm_product_360_distinct_gcd_values_l179_179135

/-- 
  Given two integers a and b, such that the product of their gcd and lcm is 360,
  we need to prove that the number of distinct possible values for their gcd is 9.
--/
theorem gcd_lcm_product_360_distinct_gcd_values :
  ∀ (a b : ℕ), gcd a b * lcm a b = 360 → 
  (∃ gcd_values : Finset ℕ, gcd_values.card = 9 ∧ ∀ g, g ∈ gcd_values ↔ g = gcd a b) :=
by
  sorry

end gcd_lcm_product_360_distinct_gcd_values_l179_179135


namespace compute_value_of_expression_l179_179472

theorem compute_value_of_expression (p q : ℝ) (h1 : 3 * p^2 - 7 * p + 1 = 0) (h2 : 3 * q^2 - 7 * q + 1 = 0) :
  (9 * p^3 - 9 * q^3) / (p - q) = 46 :=
sorry

end compute_value_of_expression_l179_179472


namespace lost_marble_count_l179_179008

def initial_marble_count : ℕ := 16
def remaining_marble_count : ℕ := 9

theorem lost_marble_count : initial_marble_count - remaining_marble_count = 7 := by
  -- Proof goes here
  sorry

end lost_marble_count_l179_179008


namespace domain_shift_l179_179808

noncomputable def domain := { x : ℝ | 1 ≤ x ∧ x ≤ 4 }
noncomputable def shifted_domain := { x : ℝ | 2 ≤ x ∧ x ≤ 5 }

theorem domain_shift (f : ℝ → ℝ) (h : ∀ x, x ∈ domain ↔ (1 ≤ x ∧ x ≤ 4)) :
  ∀ x, x ∈ shifted_domain ↔ ∃ y, (y = x - 1) ∧ y ∈ domain :=
by
  sorry

end domain_shift_l179_179808


namespace intersection_A_B_l179_179575

def A : Set ℝ := {x | 1 < x}
def B : Set ℝ := {y | y ≤ 2}
def expected_intersection : Set ℝ := {z | 1 < z ∧ z ≤ 2}

theorem intersection_A_B : (A ∩ B) = expected_intersection :=
by
  -- Proof to be completed
  sorry

end intersection_A_B_l179_179575


namespace shaded_area_equals_l179_179924

noncomputable def area_shaded_figure (R : ℝ) : ℝ :=
  let α := (60 : ℝ) * (Real.pi / 180)
  (2 * Real.pi * R^2) / 3

theorem shaded_area_equals : ∀ R : ℝ, area_shaded_figure R = (2 * Real.pi * R^2) / 3 := sorry

end shaded_area_equals_l179_179924


namespace chord_line_eq_l179_179756

theorem chord_line_eq (x y : ℝ) (h : x^2 + 4 * y^2 = 36) (midpoint : x = 4 ∧ y = 2) :
  x + 2 * y - 8 = 0 := 
sorry

end chord_line_eq_l179_179756


namespace sin_225_plus_alpha_l179_179152

theorem sin_225_plus_alpha (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = 5 / 13) :
    Real.sin (5 * Real.pi / 4 + α) = -5 / 13 :=
by
  sorry

end sin_225_plus_alpha_l179_179152


namespace find_number_l179_179954

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 := 
by
  sorry

end find_number_l179_179954


namespace smallest_rel_prime_210_l179_179292

theorem smallest_rel_prime_210 : ∃ (y : ℕ), y > 1 ∧ Nat.gcd y 210 = 1 ∧ (∀ z : ℕ, z > 1 ∧ Nat.gcd z 210 = 1 → y ≤ z) ∧ y = 11 :=
by {
  sorry -- proof to be filled in
}

end smallest_rel_prime_210_l179_179292


namespace abc_divisible_by_7_l179_179371

theorem abc_divisible_by_7 (a b c : ℤ) (h : 7 ∣ (a^3 + b^3 + c^3)) : 7 ∣ (a * b * c) :=
sorry

end abc_divisible_by_7_l179_179371


namespace figure_representation_l179_179999

theorem figure_representation (x y : ℝ) : 
  |x| + |y| ≤ 2 * Real.sqrt (x^2 + y^2) ∧ 2 * Real.sqrt (x^2 + y^2) ≤ 3 * max (|x|) (|y|) → 
  Figure2 :=
sorry

end figure_representation_l179_179999


namespace Oscar_height_correct_l179_179607

-- Definitions of the given conditions
def Tobias_height : ℕ := 184
def avg_height : ℕ := 178

def heights_valid (Victor Peter Oscar Tobias : ℕ) : Prop :=
  Tobias = 184 ∧ (Tobias + Victor + Peter + Oscar) / 4 = 178 ∧ 
  Victor = Tobias + (Tobias - Peter) ∧ 
  Oscar = Peter - (Tobias - Peter)

theorem Oscar_height_correct :
  ∃ (k : ℕ), ∀ (Victor Peter Oscar : ℕ), heights_valid Victor Peter Oscar Tobias_height →
  Oscar = 160 :=
by
  sorry

end Oscar_height_correct_l179_179607


namespace license_plate_calculation_l179_179826

def license_plate_count : ℕ :=
  let letter_choices := 26^3
  let first_digit_choices := 5
  let remaining_digit_combinations := 5 * 5
  letter_choices * first_digit_choices * remaining_digit_combinations

theorem license_plate_calculation :
  license_plate_count = 455625 :=
by
  sorry

end license_plate_calculation_l179_179826


namespace selection_methods_l179_179263

theorem selection_methods :
  ∃ (ways_with_girls : ℕ), ways_with_girls = Nat.choose 6 4 - Nat.choose 4 4 ∧ ways_with_girls = 14 := by
  sorry

end selection_methods_l179_179263


namespace positions_after_196_moves_l179_179578

def cat_position (n : ℕ) : ℕ :=
  n % 4

def mouse_position (n : ℕ) : ℕ :=
  n % 8

def cat_final_position : ℕ := 0 -- top left based on the reverse order cycle
def mouse_final_position : ℕ := 3 -- bottom middle based on the reverse order cycle

theorem positions_after_196_moves :
  cat_position 196 = cat_final_position ∧ mouse_position 196 = mouse_final_position :=
by
  sorry

end positions_after_196_moves_l179_179578


namespace water_added_l179_179395

theorem water_added (initial_volume : ℕ) (initial_sugar_percentage : ℝ) (final_sugar_percentage : ℝ) (V : ℝ) : 
  initial_volume = 3 →
  initial_sugar_percentage = 0.4 →
  final_sugar_percentage = 0.3 →
  V = 1 :=
by
  sorry

end water_added_l179_179395


namespace t_plus_inv_t_eq_three_l179_179748

theorem t_plus_inv_t_eq_three {t : ℝ} (h : t^2 - 3 * t + 1 = 0) (hnz : t ≠ 0) : t + 1 / t = 3 :=
sorry

end t_plus_inv_t_eq_three_l179_179748


namespace proposition_not_true_at_3_l179_179010

variable (P : ℕ → Prop)

theorem proposition_not_true_at_3
  (h1 : ∀ k : ℕ, P k → P (k + 1))
  (h2 : ¬ P 4) :
  ¬ P 3 :=
sorry

end proposition_not_true_at_3_l179_179010


namespace number_of_correct_conclusions_l179_179249

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def f (x : ℝ) : ℝ := x - floor x

theorem number_of_correct_conclusions : 
  ∃ n, n = 3 ∧ 
  (0 ≤ f 0) ∧ 
  (∀ x : ℝ, 0 ≤ f x) ∧ 
  (∀ x : ℝ, f x < 1) ∧ 
  (∀ x : ℝ, f (x + 1) = f x) ∧ 
  ¬ (∀ x : ℝ, f (-x) = f x) := 
sorry

end number_of_correct_conclusions_l179_179249


namespace ways_to_reach_5_5_l179_179726

def moves_to_destination : ℕ → ℕ → ℕ
| 0, 0     => 1
| 0, j+1   => moves_to_destination 0 j
| i+1, 0   => moves_to_destination i 0
| i+1, j+1 => moves_to_destination i (j+1) + moves_to_destination (i+1) j + moves_to_destination i j

theorem ways_to_reach_5_5 : moves_to_destination 5 5 = 1573 := by
  sorry

end ways_to_reach_5_5_l179_179726


namespace find_distance_between_stations_l179_179073

noncomputable def distance_between_stations (D T : ℝ) : Prop :=
  D = 100 * T ∧
  D = 50 * (T + 15 / 60) ∧
  D = 70 * (T + 7 / 60)

theorem find_distance_between_stations :
  ∃ D T : ℝ, distance_between_stations D T ∧ D = 25 :=
by
  sorry

end find_distance_between_stations_l179_179073


namespace polynomial_division_properties_l179_179287

open Polynomial

noncomputable def g : Polynomial ℝ := 3 * X^4 + 9 * X^3 - 7 * X^2 + 2 * X + 5
noncomputable def e : Polynomial ℝ := X^2 + 2 * X - 3

theorem polynomial_division_properties (s t : Polynomial ℝ) (h : g = s * e + t) (h_deg : t.degree < e.degree) :
  s.eval 1 + t.eval (-1) = -22 :=
sorry

end polynomial_division_properties_l179_179287


namespace bacterium_descendants_l179_179598

theorem bacterium_descendants (n a : ℕ) (h : a ≤ n / 2) :
  ∃ k, a ≤ k ∧ k ≤ 2 * a - 1 := 
sorry

end bacterium_descendants_l179_179598


namespace ratio_of_larger_to_smaller_l179_179291

variable {x y : ℝ}

-- Condition for x and y being positive and x > y
axiom x_pos : 0 < x
axiom y_pos : 0 < y
axiom x_gt_y : x > y

-- Condition for sum and difference relationship
axiom sum_diff_relation : x + y = 7 * (x - y)

-- Theorem: Ratio of the larger number to the smaller number is 2
theorem ratio_of_larger_to_smaller : x / y = 2 :=
by
  sorry

end ratio_of_larger_to_smaller_l179_179291


namespace toothpicks_in_20th_stage_l179_179080

theorem toothpicks_in_20th_stage :
  (3 + 3 * (20 - 1) = 60) :=
by
  sorry

end toothpicks_in_20th_stage_l179_179080


namespace Xiao_Ming_vertical_height_increase_l179_179781

noncomputable def vertical_height_increase (slope_ratio_v slope_ratio_h : ℝ) (distance : ℝ) : ℝ :=
  let x := distance / (Real.sqrt (1 + (slope_ratio_h / slope_ratio_v)^2))
  x

theorem Xiao_Ming_vertical_height_increase
  (slope_ratio_v slope_ratio_h distance : ℝ)
  (h_ratio : slope_ratio_v = 1)
  (h_ratio2 : slope_ratio_h = 2.4)
  (h_distance : distance = 130) :
  vertical_height_increase slope_ratio_v slope_ratio_h distance = 50 :=
by
  unfold vertical_height_increase
  rw [h_ratio, h_ratio2, h_distance]
  sorry

end Xiao_Ming_vertical_height_increase_l179_179781


namespace value_of_x_l179_179335

theorem value_of_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 :=
by
  sorry

end value_of_x_l179_179335


namespace least_cost_grass_seed_l179_179113

variable (cost_5_pound_bag : ℕ) [Fact (cost_5_pound_bag = 1380)]
variable (cost_10_pound_bag : ℕ) [Fact (cost_10_pound_bag = 2043)]
variable (cost_25_pound_bag : ℕ) [Fact (cost_25_pound_bag = 3225)]
variable (min_weight : ℕ) [Fact (min_weight = 65)]
variable (max_weight : ℕ) [Fact (max_weight = 80)]

theorem least_cost_grass_seed :
  ∃ (n5 n10 n25 : ℕ),
    n5 * 5 + n10 * 10 + n25 * 25 ≥ min_weight ∧
    n5 * 5 + n10 * 10 + n25 * 25 ≤ max_weight ∧
    n5 * cost_5_pound_bag + n10 * cost_10_pound_bag + n25 * cost_25_pound_bag = 9675 :=
  sorry

end least_cost_grass_seed_l179_179113


namespace workers_cut_down_correct_l179_179313

def initial_oak_trees : ℕ := 9
def remaining_oak_trees : ℕ := 7
def cut_down_oak_trees : ℕ := initial_oak_trees - remaining_oak_trees

theorem workers_cut_down_correct : cut_down_oak_trees = 2 := by
  sorry

end workers_cut_down_correct_l179_179313


namespace line_single_point_not_necessarily_tangent_l179_179318

-- Define a curve
def curve : Type := ℝ → ℝ

-- Define a line
def line (m b : ℝ) : curve := λ x => m * x + b

-- Define a point of intersection
def intersects_at (l : curve) (c : curve) (x : ℝ) : Prop :=
  l x = c x

-- Define the property of having exactly one common point
def has_single_intersection (l : curve) (c : curve) : Prop :=
  ∃ x, ∀ y ≠ x, l y ≠ c y

-- Define the tangent line property
def is_tangent (l : curve) (c : curve) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs ((c (x + h) - c x) / h - (l (x + h) - l x) / h) < ε

-- The proof statement: There exists a curve c and a line l such that l has exactly one intersection point with c, but l is not necessarily a tangent to c.
theorem line_single_point_not_necessarily_tangent :
  ∃ c : curve, ∃ l : curve, has_single_intersection l c ∧ ∃ x, ¬ is_tangent l c x :=
sorry

end line_single_point_not_necessarily_tangent_l179_179318


namespace peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l179_179680

-- Define the context of the problem
def total_people := 100
def men := 50
def women := 50

-- Define Peter Ivanovich being satisfied
def satisfies_peter_ivanovich := (women / (total_people - 1)) * ((women - 1) / (total_people - 2)) 

-- Define the probability that Peter Ivanovich is satisfied
theorem peter_ivanovich_satisfied_probability :
  satisfies_peter_ivanovich = 25 / 33 := 
sorry

-- Define the expected number of satisfied men
def expected_satisfied_men := men * (25 / 33)

-- Prove the expected number of satisfied men
theorem expected_satisfied_men_value :
  expected_satisfied_men = 1250 / 33 :=
sorry

end peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l179_179680


namespace alternating_sum_cubes_eval_l179_179660

noncomputable def alternating_sum_cubes : ℕ → ℤ
| 0 => 0
| n + 1 => alternating_sum_cubes n + (-1)^(n / 4) * (n + 1)^3

theorem alternating_sum_cubes_eval :
  alternating_sum_cubes 99 = S :=
by
  sorry

end alternating_sum_cubes_eval_l179_179660


namespace complex_multiplication_l179_179972

-- Definition of the imaginary unit
def is_imaginary_unit (i : ℂ) : Prop := i * i = -1

theorem complex_multiplication (i : ℂ) (h : is_imaginary_unit i) : (1 + i) * (1 - i) = 2 :=
by
  -- Given that i is the imaginary unit satisfying i^2 = -1
  -- We need to show that (1 + i) * (1 - i) = 2
  sorry

end complex_multiplication_l179_179972


namespace probability_same_spot_l179_179884

theorem probability_same_spot :
  let students := ["A", "B"]
  let spots := ["Spot 1", "Spot 2"]
  let total_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 1"), ("B", "Spot 2")),
                         (("A", "Spot 2"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]
  let favorable_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                             (("A", "Spot 2"), ("B", "Spot 2"))]
  ∀ (students : List String) (spots : List String)
    (total_outcomes favorable_outcomes : List ((String × String) × (String × String))),
  (students = ["A", "B"]) →
  (spots = ["Spot 1", "Spot 2"]) →
  (total_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                     (("A", "Spot 1"), ("B", "Spot 2")),
                     (("A", "Spot 2"), ("B", "Spot 1")),
                     (("A", "Spot 2"), ("B", "Spot 2"))]) →
  (favorable_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]) →
  favorable_outcomes.length / total_outcomes.length = 1 / 2 := 
by
  intros
  sorry

end probability_same_spot_l179_179884


namespace henley_initial_candies_l179_179344

variables (C : ℝ)
variables (h1 : 0.60 * C = 180)

theorem henley_initial_candies : C = 300 :=
by sorry

end henley_initial_candies_l179_179344


namespace expression_divisible_by_1961_l179_179103

theorem expression_divisible_by_1961 (n : ℕ) : 
  (5^(2*n) * 3^(4*n) - 2^(6*n)) % 1961 = 0 := by
  sorry

end expression_divisible_by_1961_l179_179103


namespace lunch_cost_before_tip_l179_179003

theorem lunch_cost_before_tip (C : ℝ) (h : C + 0.2 * C = 60.6) : C = 50.5 :=
sorry

end lunch_cost_before_tip_l179_179003


namespace sin_ninety_degrees_l179_179297

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l179_179297


namespace find_m_l179_179175

noncomputable def slope_at_one (m : ℝ) := 2 + m

noncomputable def tangent_line_eq (m : ℝ) (x : ℝ) := (slope_at_one m) * x - 2 * m

noncomputable def y_intercept (m : ℝ) := tangent_line_eq m 0

noncomputable def x_intercept (m : ℝ) := - (y_intercept m) / (slope_at_one m)

noncomputable def intercept_sum_eq (m : ℝ) := (x_intercept m) + (y_intercept m)

theorem find_m (m : ℝ) (h : m ≠ -2) (h2 : intercept_sum_eq m = 12) : m = -3 ∨ m = -4 := 
sorry

end find_m_l179_179175


namespace total_amount_after_interest_l179_179941

-- Define the constants
def principal : ℝ := 979.0209790209791
def rate : ℝ := 0.06
def time : ℝ := 2.4

-- Define the formula for interest calculation
def interest (P R T : ℝ) : ℝ := P * R * T

-- Define the formula for the total amount after interest is added
def total_amount (P I : ℝ) : ℝ := P + I

-- State the theorem
theorem total_amount_after_interest : 
    total_amount principal (interest principal rate time) = 1120.0649350649352 :=
by
    -- placeholder for the proof
    sorry

end total_amount_after_interest_l179_179941


namespace rectangle_diagonal_l179_179142

theorem rectangle_diagonal (P A: ℝ) (hP : P = 46) (hA : A = 120) : ∃ d : ℝ, d = 17 :=
by
  -- Sorry provides the placeholder for the actual proof.
  sorry

end rectangle_diagonal_l179_179142


namespace jill_spent_10_percent_on_food_l179_179455

theorem jill_spent_10_percent_on_food 
  (T : ℝ)                         
  (h1 : 0.60 * T = 0.60 * T)    -- 60% on clothing
  (h2 : 0.30 * T = 0.30 * T)    -- 30% on other items
  (h3 : 0.04 * (0.60 * T) = 0.024 * T)  -- 4% tax on clothing
  (h4 : 0.08 * (0.30 * T) = 0.024 * T)  -- 8% tax on other items
  (h5 : 0.048 * T = (0.024 * T + 0.024 * T)) -- total tax is 4.8%
  : 0.10 * T = (T - (0.60*T + 0.30*T)) :=
by
  -- Proof is omitted
  sorry

end jill_spent_10_percent_on_food_l179_179455


namespace length_of_BD_l179_179751

theorem length_of_BD (AB AC CB BD : ℝ) (h1 : AB = 10) (h2 : AC = 4 * CB) (h3 : AC = 4 * 2) (h4 : CB = 2) :
  BD = 3 :=
sorry

end length_of_BD_l179_179751


namespace intersection_eq_l179_179246

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof Problem
theorem intersection_eq : M ∩ N = {2, 3} := 
by
  sorry

end intersection_eq_l179_179246


namespace shoes_difference_l179_179243

theorem shoes_difference :
  let pairs_per_box := 20
  let boxes_A := 8
  let boxes_B := 5 * boxes_A
  let total_pairs_A := boxes_A * pairs_per_box
  let total_pairs_B := boxes_B * pairs_per_box
  total_pairs_B - total_pairs_A = 640 :=
by
  sorry

end shoes_difference_l179_179243


namespace parametric_hyperbola_l179_179187

theorem parametric_hyperbola (t : ℝ) (ht : t ≠ 0) : 
  let x := t + 1 / t
  let y := t - 1 / t
  x^2 - y^2 = 4 :=
by
  let x := t + 1 / t
  let y := t - 1 / t
  sorry

end parametric_hyperbola_l179_179187


namespace flags_left_l179_179495

theorem flags_left (interval circumference : ℕ) (total_flags : ℕ) (h1 : interval = 20) (h2 : circumference = 200) (h3 : total_flags = 12) : 
  total_flags - (circumference / interval) = 2 := 
by 
  -- Using the conditions h1, h2, h3
  sorry

end flags_left_l179_179495


namespace simplify_expression_l179_179541

variable (a b : ℝ)

theorem simplify_expression : -3 * a * (2 * a - 4 * b + 2) + 6 * a = -6 * a ^ 2 + 12 * a * b := by
  sorry

end simplify_expression_l179_179541


namespace inequality_proof_l179_179027

theorem inequality_proof (n : ℕ) (hn : n > 0) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end inequality_proof_l179_179027


namespace part1_part2_l179_179528

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ := Real.tan ((x / 2) - (Real.pi / 3))

-- Part (1)
theorem part1 : f (5 * Real.pi / 2) = Real.sqrt 3 - 2 :=
by
  sorry

-- Part (2)
theorem part2 (k : ℤ) : { x : ℝ | f x ≤ Real.sqrt 3 } = 
  {x | ∃ (k : ℤ), 2 * k * Real.pi - Real.pi / 3 < x ∧ x ≤ 2 * k * Real.pi + 4 * Real.pi / 3} :=
by
  sorry

end ProofProblem

end part1_part2_l179_179528


namespace valentines_left_l179_179489

theorem valentines_left (initial valentines_to_children valentines_to_neighbors valentines_to_coworkers : ℕ) (h_initial : initial = 30)
  (h1 : valentines_to_children = 8) (h2 : valentines_to_neighbors = 5) (h3 : valentines_to_coworkers = 3) : initial - (valentines_to_children + valentines_to_neighbors + valentines_to_coworkers) = 14 := by
  sorry

end valentines_left_l179_179489


namespace compute_y_series_l179_179560

theorem compute_y_series :
  (∑' n : ℕ, (1 / 3) ^ n) + (∑' n : ℕ, ((-1) ^ n) / (4 ^ n)) = ∑' n : ℕ, (1 / (23 / 13) ^ n) :=
by
  sorry

end compute_y_series_l179_179560


namespace smallest_even_integer_cube_mod_1000_l179_179052

theorem smallest_even_integer_cube_mod_1000 :
  ∃ n : ℕ, (n % 2 = 0) ∧ (n > 0) ∧ (n^3 % 1000 = 392) ∧ (∀ m : ℕ, (m % 2 = 0) ∧ (m > 0) ∧ (m^3 % 1000 = 392) → n ≤ m) ∧ n = 892 := 
sorry

end smallest_even_integer_cube_mod_1000_l179_179052


namespace derivative_f_l179_179837

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem derivative_f (x : ℝ) (hx : x ≠ 0) :
  deriv f x = 1 - (1 / (x ^ 2)) :=
by
  -- The proof goes here
  sorry

end derivative_f_l179_179837


namespace principal_amount_l179_179015

theorem principal_amount (SI P R T : ℝ) 
  (h1 : R = 12) (h2 : T = 3) (h3 : SI = 3600) : 
  SI = P * R * T / 100 → P = 10000 :=
by
  intros h
  sorry

end principal_amount_l179_179015


namespace find_diff_eq_l179_179644

noncomputable def general_solution (y : ℝ → ℝ) : Prop :=
∃ (C1 C2 : ℝ), ∀ x : ℝ, y x = C1 * x + C2

theorem find_diff_eq (y : ℝ → ℝ) (C1 C2 : ℝ) (h : ∀ x : ℝ, y x = C1 * x + C2) :
  ∀ x : ℝ, (deriv (deriv y)) x = 0 :=
by
  sorry

end find_diff_eq_l179_179644


namespace ceil_floor_arith_l179_179994

theorem ceil_floor_arith :
  (Int.ceil (((15: ℚ) / 8)^2 * (-34 / 4)) - Int.floor ((15 / 8) * Int.floor (-34 / 4))) = -12 :=
by sorry

end ceil_floor_arith_l179_179994


namespace roots_of_polynomial_inequality_l179_179625

theorem roots_of_polynomial_inequality :
  (∃ (p q r s : ℂ), (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) ∧
  (p * q * r * s = 3) ∧ (p*q + p*r + p*s + q*r + q*s + r*s = 11)) →
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3) :=
by
  sorry

end roots_of_polynomial_inequality_l179_179625


namespace action_movies_rented_l179_179163

-- Defining the conditions as hypotheses
theorem action_movies_rented (a M A D : ℝ) (h1 : 0.64 * M = 10 * a)
                             (h2 : D = 5 * A)
                             (h3 : D + A = 0.36 * M) :
    A = 0.9375 * a :=
sorry

end action_movies_rented_l179_179163


namespace exists_composite_l179_179223

theorem exists_composite (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 100) (hy : 2 ≤ y ∧ y ≤ 100) :
  ∃ n : ℕ, ∃ k : ℕ, x^(2^n) + y^(2^n) = k * (k + 1) :=
by {
  sorry -- proof goes here
}

end exists_composite_l179_179223


namespace max_omega_l179_179431

theorem max_omega (ω : ℕ) (T : ℝ) (h₁ : T = 2 * Real.pi / ω) (h₂ : 1 < T) (h₃ : T < 3) : ω = 6 :=
sorry

end max_omega_l179_179431


namespace geometric_sequence_sum_product_l179_179475

theorem geometric_sequence_sum_product {a b c : ℝ} : 
  a + b + c = 14 → 
  a * b * c = 64 → 
  (a = 8 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) :=
by
  sorry

end geometric_sequence_sum_product_l179_179475


namespace value_of_r_for_n_3_l179_179413

theorem value_of_r_for_n_3 :
  ∀ (r s : ℕ), 
  (r = 4^s + 3 * s) → 
  (s = 2^3 + 2) → 
  r = 1048606 :=
by
  intros r s h1 h2
  sorry

end value_of_r_for_n_3_l179_179413


namespace math_enthusiast_gender_relation_female_success_probability_l179_179828

-- Constants and probabilities
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 12
def d : ℕ := 28
def n : ℕ := 100
def P_male_success : ℚ := 3 / 4
def P_female_success : ℚ := 2 / 3
def K_threshold : ℚ := 6.635

-- Computation of K^2
def K_square : ℚ := n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- The first part of the proof comparing K^2 with threshold
theorem math_enthusiast_gender_relation : K_square < K_threshold := sorry

-- The second part calculating given conditions for probability calculation
def P_A : ℚ := (P_male_success ^ 2 * (1 - P_female_success)) + (2 * (1 - P_male_success) * P_male_success * P_female_success)
def P_AB : ℚ := 2 * (1 - P_male_success) * P_male_success * P_female_success
def P_B_given_A : ℚ := P_AB / P_A

theorem female_success_probability : P_B_given_A = 4 / 7 := sorry

end math_enthusiast_gender_relation_female_success_probability_l179_179828


namespace solve_quadratic_eq_l179_179594

theorem solve_quadratic_eq (x : ℝ) : (x^2 - 2*x + 1 = 9) → (x = 4 ∨ x = -2) :=
by
  intro h
  sorry

end solve_quadratic_eq_l179_179594


namespace remainder_of_sum_is_zero_l179_179474

-- Define the properties of m and n according to the conditions of the problem
def m : ℕ := 2 * 1004 ^ 2
def n : ℕ := 2007 * 1003

-- State the theorem that proves the remainder of (m + n) divided by 1004 is 0
theorem remainder_of_sum_is_zero : (m + n) % 1004 = 0 := by
  sorry

end remainder_of_sum_is_zero_l179_179474


namespace number_of_two_digit_primes_with_digit_sum_12_l179_179566

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_digit_sum_12 : 
  ∃! n, is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 12 :=
by
  sorry

end number_of_two_digit_primes_with_digit_sum_12_l179_179566


namespace correct_operations_result_l179_179349

/-
Pat intended to multiply a number by 8 but accidentally divided by 8.
Pat then meant to add 20 to the result but instead subtracted 20.
After these errors, the final outcome was 12.
Prove that if Pat had performed the correct operations, the final outcome would have been 2068.
-/

theorem correct_operations_result (n : ℕ) (h1 : n / 8 - 20 = 12) : 8 * n + 20 = 2068 :=
by
  sorry

end correct_operations_result_l179_179349


namespace repeating_decimal_sum_correct_l179_179429

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  x + y - z

theorem repeating_decimal_sum_correct :
  repeating_decimal_sum = 4 / 9 :=
by
  sorry

end repeating_decimal_sum_correct_l179_179429


namespace unique_intersection_l179_179032

theorem unique_intersection {m : ℝ} :
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25 / 3 :=
by
  sorry

end unique_intersection_l179_179032


namespace find_angle4_l179_179252

noncomputable def angle_1 := 70
noncomputable def angle_2 := 110
noncomputable def angle_3 := 35
noncomputable def angle_4 := 35

theorem find_angle4 (h1 : angle_1 + angle_2 = 180) (h2 : angle_3 = angle_4) :
  angle_4 = 35 :=
by
  have h3: angle_1 + 70 + 40 = 180 := by sorry
  have h4: angle_2 + angle_3 + angle_4 = 180 := by sorry
  sorry

end find_angle4_l179_179252


namespace train_pass_time_l179_179377

noncomputable def train_speed_kmh := 36  -- Speed in km/hr
noncomputable def train_speed_ms := 10   -- Speed in m/s (converted)
noncomputable def platform_length := 180 -- Length of the platform in meters
noncomputable def platform_pass_time := 30 -- Time in seconds to pass platform
noncomputable def train_length := 120    -- Train length derived from conditions

theorem train_pass_time 
  (speed_in_kmh : ℕ) (speed_in_ms : ℕ) (platform_len : ℕ) (pass_platform_time : ℕ) (train_len : ℕ)
  (h1 : speed_in_kmh = 36)
  (h2 : speed_in_ms = 10)
  (h3 : platform_len = 180)
  (h4 : pass_platform_time = 30)
  (h5 : train_len = 120) :
  (train_len / speed_in_ms) = 12 := by
  sorry

end train_pass_time_l179_179377


namespace sam_current_yellow_marbles_l179_179853

theorem sam_current_yellow_marbles (original_yellow : ℕ) (taken_yellow : ℕ) (current_yellow : ℕ) :
  original_yellow = 86 → 
  taken_yellow = 25 → 
  current_yellow = original_yellow - taken_yellow → 
  current_yellow = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sam_current_yellow_marbles_l179_179853


namespace sufficient_but_not_necessary_condition_l179_179661

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  ((x + 1) * (x - 3) < 0 → x > -1) ∧ ¬ (x > -1 → (x + 1) * (x - 3) < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l179_179661


namespace increasing_iff_range_a_three_distinct_real_roots_l179_179848

noncomputable def f (a x : ℝ) : ℝ :=
  if x >= 2 * a then x^2 + (2 - 2 * a) * x else - x^2 + (2 + 2 * a) * x

theorem increasing_iff_range_a (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem three_distinct_real_roots (a t : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2)
  (h_roots : ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧
                           f a x₁ = t * f a (2 * a) ∧
                           f a x₂ = t * f a (2 * a) ∧
                           f a x₃ = t * f a (2 * a)) :
  1 < t ∧ t < 9 / 8 :=
sorry

end increasing_iff_range_a_three_distinct_real_roots_l179_179848


namespace find_number_l179_179753

def initial_condition (x : ℝ) : Prop :=
  ((x + 7) * 3 - 12) / 6 = -8

theorem find_number (x : ℝ) (h : initial_condition x) : x = -19 := by
  sorry

end find_number_l179_179753


namespace seating_arrangement_l179_179145

-- Define the problem in Lean
theorem seating_arrangement :
  let n := 9   -- Total number of people
  let r := 7   -- Number of seats at the circular table
  let combinations := Nat.choose n 2  -- Ways to select 2 people not seated
  let factorial (k : ℕ) := Nat.recOn k 1 (λ k' acc => (k' + 1) * acc)
  let arrangements := factorial (r - 1)  -- Ways to seat 7 people around a circular table
  combinations * arrangements = 25920 :=
by
  -- In Lean, sorry is used to indicate that we skip the proof for now.
  sorry

end seating_arrangement_l179_179145


namespace find_x_l179_179293

theorem find_x (x : ℝ) (h : 3.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2800.0000000000005) : x = 0.3 :=
sorry

end find_x_l179_179293


namespace exists_increasing_or_decreasing_subsequence_l179_179258

theorem exists_increasing_or_decreasing_subsequence (n : ℕ) (a : Fin (n^2 + 1) → ℝ) :
  ∃ (b : Fin (n + 1) → ℝ), (StrictMono b ∨ StrictAnti b) :=
sorry

end exists_increasing_or_decreasing_subsequence_l179_179258


namespace bicycle_weight_l179_179251

theorem bicycle_weight (b s : ℝ) (h1 : 9 * b = 5 * s) (h2 : 4 * s = 160) : b = 200 / 9 :=
by
  sorry

end bicycle_weight_l179_179251


namespace sequence_inequality_l179_179396

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
    (h_subadd : ∀ m n : ℕ, a (n + m) ≤ a n + a m) :
  ∀ (n m : ℕ), m ≤ n → a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := 
by
  intros n m hnm
  sorry

end sequence_inequality_l179_179396


namespace even_function_implies_a_zero_l179_179636

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, (x^2 - |x + a|) = (x^2 - |x - a|)) → a = 0 :=
by
  sorry

end even_function_implies_a_zero_l179_179636


namespace negation_of_proposition_divisible_by_2_is_not_even_l179_179807

theorem negation_of_proposition_divisible_by_2_is_not_even :
  (¬ ∀ n : ℕ, n % 2 = 0 → (n % 2 = 0 → n % 2 = 0))
  ↔ ∃ n : ℕ, n % 2 = 0 ∧ n % 2 ≠ 0 := 
  by
    sorry

end negation_of_proposition_divisible_by_2_is_not_even_l179_179807


namespace sum_of_any_three_on_line_is_30_l179_179843

/-- Define the list of numbers from 1 to 19 -/
def numbers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- Define the specific sequence found in the solution -/
def arrangement :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 18,
   17, 16, 15, 14, 13, 12, 11]

/-- Define the function to compute the sum of any three numbers on a straight line -/
def sum_on_line (a b c : ℕ) := a + b + c

theorem sum_of_any_three_on_line_is_30 :
  ∀ i j k : ℕ, 
  i ∈ numbers ∧ j ∈ numbers ∧ k ∈ numbers ∧ (i = 10 ∨ j = 10 ∨ k = 10) →
  sum_on_line i j k = 30 :=
by
  sorry

end sum_of_any_three_on_line_is_30_l179_179843


namespace workout_total_correct_l179_179120

structure Band := 
  (A : ℕ) 
  (B : ℕ) 
  (C : ℕ)

structure Equipment := 
  (leg_weight_squat : ℕ) 
  (dumbbell : ℕ) 
  (leg_weight_lunge : ℕ) 
  (kettlebell : ℕ)

def total_weight (bands : Band) (equip : Equipment) : ℕ := 
  let squat_total := bands.A + bands.B + bands.C + (2 * equip.leg_weight_squat) + equip.dumbbell
  let lunge_total := bands.A + bands.C + (2 * equip.leg_weight_lunge) + equip.kettlebell
  squat_total + lunge_total

theorem workout_total_correct (bands : Band) (equip : Equipment) : 
  bands = ⟨7, 5, 3⟩ → 
  equip = ⟨10, 15, 8, 18⟩ → 
  total_weight bands equip = 94 :=
by 
  -- Insert your proof steps here
  sorry

end workout_total_correct_l179_179120


namespace packs_to_purchase_l179_179350

theorem packs_to_purchase {n m k : ℕ} (h : 8 * n + 15 * m + 30 * k = 135) : n + m + k = 5 :=
sorry

end packs_to_purchase_l179_179350


namespace isosceles_triangle_base_length_l179_179207

theorem isosceles_triangle_base_length (P Q : ℕ) (x y : ℕ) (hP : P = 15) (hQ : Q = 12) (hPerimeter : 2 * x + y = 27) 
      (hCondition : (y = P ∧ (1 / 2) * x + x = P) ∨ (y = Q ∧ (1 / 2) * x + x = Q)) : 
  y = 7 ∨ y = 11 :=
sorry

end isosceles_triangle_base_length_l179_179207


namespace fraction_of_menu_items_my_friend_can_eat_l179_179929

theorem fraction_of_menu_items_my_friend_can_eat {menu_size vegan_dishes nut_free_vegan_dishes : ℕ}
    (h1 : vegan_dishes = 6)
    (h2 : vegan_dishes = menu_size / 6)
    (h3 : nut_free_vegan_dishes = vegan_dishes - 5) :
    (nut_free_vegan_dishes : ℚ) / menu_size = 1 / 36 :=
by
  sorry

end fraction_of_menu_items_my_friend_can_eat_l179_179929


namespace arithmetic_sequence_a8_value_l179_179128

theorem arithmetic_sequence_a8_value
  (a : ℕ → ℤ) 
  (h1 : a 1 + 3 * a 8 + a 15 = 120)
  (h2 : a 1 + a 15 = 2 * a 8) :
  a 8 = 24 := 
sorry

end arithmetic_sequence_a8_value_l179_179128


namespace slope_of_line_l179_179193

theorem slope_of_line : ∀ (x y : ℝ), 2 * x - 4 * y + 7 = 0 → (y = (1/2) * x - 7 / 4) :=
by
  intro x y h
  -- This would typically involve rearranging the given equation to the slope-intercept form
  -- but as we are focusing on creating the statement, we insert sorry to skip the proof
  sorry

end slope_of_line_l179_179193


namespace garden_perimeter_l179_179668

theorem garden_perimeter (A : ℝ) (P : ℝ) : 
  (A = 97) → (P = 40) :=
by
  sorry

end garden_perimeter_l179_179668


namespace range_m_single_solution_l179_179559

-- Statement expressing the conditions and conclusion.
theorem range_m_single_solution :
  ∀ (m : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → x^3 - 3 * x + m = 0 → ∃! x, 0 ≤ x ∧ x ≤ 2) ↔ m ∈ (Set.Ico (-2 : ℝ) 0) ∪ {2} := 
sorry

end range_m_single_solution_l179_179559


namespace increasing_ω_l179_179444

noncomputable def f (ω x : ℝ) : ℝ := (1 / 2) * (Real.sin ((ω * x) / 2)) * (Real.cos ((ω * x) / 2))

theorem increasing_ω (ω : ℝ) (hω : 0 < ω) :
  (∀ x y, - (Real.pi / 3) ≤ x → x ≤ y → y ≤ (Real.pi / 4) → f ω x ≤ f ω y)
  ↔ 0 < ω ∧ ω ≤ (3 / 2) :=
sorry

end increasing_ω_l179_179444


namespace no_pairs_exist_l179_179588

theorem no_pairs_exist (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : (1/a + 1/b = 2/(a+b)) → False :=
by
  sorry

end no_pairs_exist_l179_179588


namespace gcd_lcm_mul_l179_179980

theorem gcd_lcm_mul (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := 
by
  sorry

end gcd_lcm_mul_l179_179980


namespace surface_area_of_circumscribed_sphere_l179_179551

theorem surface_area_of_circumscribed_sphere (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : 
  ∃ S : ℝ, S = 29 * Real.pi :=
by
  sorry

end surface_area_of_circumscribed_sphere_l179_179551


namespace sum_of_triangle_angles_sin_halves_leq_one_l179_179262

theorem sum_of_triangle_angles_sin_halves_leq_one (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC : A + B + C = Real.pi) : 
  8 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 := 
sorry 

end sum_of_triangle_angles_sin_halves_leq_one_l179_179262


namespace rate_of_interest_l179_179284

theorem rate_of_interest (P A T SI : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 2)
  (h4 : SI = A - P) (h5 : SI = (P * R * T) / 100) : R = 10 :=
by
  sorry

end rate_of_interest_l179_179284


namespace graph_of_f_does_not_pass_through_second_quadrant_l179_179026

def f (x : ℝ) : ℝ := x - 2

theorem graph_of_f_does_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = f x ∧ x < 0 ∧ y > 0 :=
sorry

end graph_of_f_does_not_pass_through_second_quadrant_l179_179026


namespace medium_ceiling_lights_count_l179_179976

theorem medium_ceiling_lights_count (S M L : ℕ) 
  (h1 : L = 2 * M) 
  (h2 : S = M + 10) 
  (h_bulbs : S + 2 * M + 3 * L = 118) : M = 12 :=
by
  -- Proof omitted
  sorry

end medium_ceiling_lights_count_l179_179976


namespace remainder_zero_l179_179586

theorem remainder_zero :
  ∀ (a b c d : ℕ),
  a % 53 = 47 →
  b % 53 = 4 →
  c % 53 = 10 →
  d % 53 = 14 →
  (((a * b * c) % 53) * d) % 47 = 0 := 
by 
  intros a b c d h1 h2 h3 h4
  sorry

end remainder_zero_l179_179586


namespace N2O3_weight_l179_179797

-- Definitions from the conditions
def molecularWeightN : Float := 14.01
def molecularWeightO : Float := 16.00
def molecularWeightN2O3 : Float := (2 * molecularWeightN) + (3 * molecularWeightO)
def moles : Float := 4

-- The main proof problem statement
theorem N2O3_weight (h1 : molecularWeightN = 14.01)
                    (h2 : molecularWeightO = 16.00)
                    (h3 : molecularWeightN2O3 = (2 * molecularWeightN) + (3 * molecularWeightO))
                    (h4 : moles = 4) :
                    (moles * molecularWeightN2O3) = 304.08 :=
by
  sorry

end N2O3_weight_l179_179797


namespace min_value_of_expression_is_6_l179_179229

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a

theorem min_value_of_expression_is_6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : min_value_of_expression a b c = 6 :=
by
  sorry

end min_value_of_expression_is_6_l179_179229


namespace isosceles_right_triangle_hypotenuse_l179_179712

noncomputable def hypotenuse_length : ℝ :=
  let a := Real.sqrt 363
  let c := Real.sqrt (2 * (a ^ 2))
  c

theorem isosceles_right_triangle_hypotenuse :
  ∀ (a : ℝ),
    (2 * (a ^ 2)) + (a ^ 2) = 1452 →
    hypotenuse_length = Real.sqrt 726 := by
  intro a h
  rw [hypotenuse_length]
  sorry

end isosceles_right_triangle_hypotenuse_l179_179712


namespace student_selection_problem_l179_179893

noncomputable def total_selections : ℕ :=
  let C := Nat.choose
  let A := Nat.factorial
  (C 3 1 * C 3 2 + C 3 2 * C 3 1 + C 3 3) * A 3

theorem student_selection_problem :
  total_selections = 114 :=
by
  sorry

end student_selection_problem_l179_179893


namespace Wayne_initially_collected_blocks_l179_179530

-- Let's denote the initial blocks collected by Wayne as 'w'.
-- According to the problem:
-- - Wayne's father gave him 6 more blocks.
-- - He now has 15 blocks in total.
--
-- We need to prove that the initial number of blocks Wayne collected (w) is 9.

theorem Wayne_initially_collected_blocks : 
  ∃ w : ℕ, (w + 6 = 15) ↔ (w = 9) := by
  sorry

end Wayne_initially_collected_blocks_l179_179530


namespace solve_for_ab_l179_179682

def f (a b : ℚ) (x : ℚ) : ℚ := a * x^3 - 4 * x^2 + b * x - 3

theorem solve_for_ab : 
  ∃ a b : ℚ, 
    f a b 1 = 3 ∧ 
    f a b (-2) = -47 ∧ 
    (a, b) = (4 / 3, 26 / 3) := 
by
  sorry

end solve_for_ab_l179_179682


namespace find_a_iff_l179_179408

def non_deg_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, 9 * (x^2) + (y^2) - 36 * x + 8 * y = k → 
  (∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0))

theorem find_a_iff (k : ℝ) : non_deg_ellipse k ↔ k > -52 := by
  sorry

end find_a_iff_l179_179408


namespace minimum_perimeter_is_728_l179_179409

noncomputable def minimum_common_perimeter (a b c : ℤ) (h1 : 2 * a + 18 * c = 2 * b + 20 * c)
  (h2 : 9 * c * Real.sqrt (a^2 - (9 * c)^2) = 10 * c * Real.sqrt (b^2 - (10 * c)^2)) 
  (h3 : a = b + c) : ℤ :=
2 * a + 18 * c

theorem minimum_perimeter_is_728 (a b c : ℤ) 
  (h1 : 2 * a + 18 * c = 2 * b + 20 * c) 
  (h2 : 9 * c * Real.sqrt (a^2 - (9 * c)^2) = 10 * c * Real.sqrt (b^2 - (10 * c)^2)) 
  (h3 : a = b + c) : 
  minimum_common_perimeter a b c h1 h2 h3 = 728 :=
sorry

end minimum_perimeter_is_728_l179_179409


namespace car_travel_time_l179_179520

noncomputable def travelTimes 
  (t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime : ℝ) : Prop :=
t_Ningi_Zipra = 0.80 * t_Ngapara_Zipra ∧
t_Ngapara_Zipra = 60 ∧
totalTravelTime = t_Ngapara_Zipra + t_Ningi_Zipra

theorem car_travel_time :
  ∃ t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime,
  travelTimes t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime ∧
  totalTravelTime = 108 :=
by
  sorry

end car_travel_time_l179_179520


namespace tripodasaurus_flock_l179_179989

theorem tripodasaurus_flock (num_tripodasauruses : ℕ) (total_head_legs : ℕ) 
  (H1 : ∀ T, total_head_legs = 4 * T)
  (H2 : total_head_legs = 20) :
  num_tripodasauruses = 5 :=
by
  sorry

end tripodasaurus_flock_l179_179989


namespace linear_function_incorrect_conclusion_C_l179_179742

theorem linear_function_incorrect_conclusion_C :
  ∀ (x y : ℝ), (y = -2 * x + 4) → ¬(∃ x, y = 0 ∧ (x = 0 ∧ y = 4)) := by
  sorry

end linear_function_incorrect_conclusion_C_l179_179742


namespace choir_members_minimum_l179_179491

theorem choir_members_minimum (n : Nat) (h9 : n % 9 = 0) (h10 : n % 10 = 0) (h11 : n % 11 = 0) (h14 : n % 14 = 0) : n = 6930 :=
sorry

end choir_members_minimum_l179_179491


namespace find_b_l179_179780

noncomputable def circle_center_radius : Prop :=
  let C := (2, 0) -- center
  let r := 2 -- radius
  C.1 = 2 ∧ C.2 = 0 ∧ r = 2

noncomputable def line (b : ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, M ≠ N ∧ 
  (M.2 = M.1 + b) ∧ (N.2 = N.1 + b) -- points on the line are M = (x1, x1 + b) and N = (x2, x2 + b)

noncomputable def perpendicular_condition (M N center: ℝ × ℝ) : Prop :=
  (M.1 - center.1) * (N.1 - center.1) + (M.2 - center.2) * (N.2 - center.2) = 0 -- CM ⟂ CN

theorem find_b (b : ℝ) : 
  circle_center_radius ∧
  (∃ M N, line b ∧ perpendicular_condition M N (2, 0)) →
  b = 0 ∨ b = -4 :=
by {
  -- Proof omitted
  sorry
}

end find_b_l179_179780


namespace option_d_is_pythagorean_triple_l179_179881

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem option_d_is_pythagorean_triple : is_pythagorean_triple 5 12 13 :=
by
  -- This will be the proof part, which is omitted as per the problem's instructions.
  sorry

end option_d_is_pythagorean_triple_l179_179881


namespace red_cards_taken_out_l179_179699

-- Definitions based on the conditions
def total_cards : ℕ := 52
def half_of_total_cards (n : ℕ) := n / 2
def initial_red_cards : ℕ := half_of_total_cards total_cards
def remaining_red_cards : ℕ := 16

-- The statement to prove
theorem red_cards_taken_out : initial_red_cards - remaining_red_cards = 10 := by
  sorry

end red_cards_taken_out_l179_179699


namespace fraction_meaningful_l179_179658

theorem fraction_meaningful (x : ℝ) : (x + 2 ≠ 0) ↔ x ≠ -2 := by
  sorry

end fraction_meaningful_l179_179658


namespace no_adjacent_same_roll_probability_l179_179099

-- We define probabilistic event on rolling a six-sided die and sitting around a circular table
noncomputable def probability_no_adjacent_same_roll : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6) * (4/6)

theorem no_adjacent_same_roll_probability :
  probability_no_adjacent_same_roll = 625/1944 :=
by
  sorry

end no_adjacent_same_roll_probability_l179_179099


namespace red_marbles_difference_l179_179118

theorem red_marbles_difference 
  (x y : ℕ) 
  (h1 : 7 * x + 3 * x = 140) 
  (h2 : 3 * y + 2 * y = 140)
  (h3 : 10 * x = 5 * y) : 
  7 * x - 3 * y = 20 := 
by 
  sorry

end red_marbles_difference_l179_179118


namespace find_x_l179_179755

theorem find_x
  (x : ℝ)
  (h : 0.20 * x = 0.40 * 140 + 80) :
  x = 680 := 
sorry

end find_x_l179_179755


namespace ratio_area_III_IV_l179_179963

theorem ratio_area_III_IV 
  (perimeter_I : ℤ)
  (perimeter_II : ℤ)
  (perimeter_IV : ℤ)
  (side_III_is_three_times_side_I : ℤ)
  (h1 : perimeter_I = 16)
  (h2 : perimeter_II = 20)
  (h3 : perimeter_IV = 32)
  (h4 : side_III_is_three_times_side_I = 3 * (perimeter_I / 4)) :
  (3 * (perimeter_I / 4))^2 / (perimeter_IV / 4)^2 = 9 / 4 :=
by
  sorry

end ratio_area_III_IV_l179_179963


namespace tangency_condition_and_point_l179_179664

variable (a b p q : ℝ)

/-- Condition for the line y = px + q to be tangent to the ellipse b^2 x^2 + a^2 y^2 = a^2 b^2. -/
theorem tangency_condition_and_point
  (h_cond : a^2 * p^2 + b^2 - q^2 = 0)
  : 
  ∃ (x₀ y₀ : ℝ), 
  x₀ = - (a^2 * p) / q ∧
  y₀ = b^2 / q ∧ 
  (b^2 * x₀^2 + a^2 * y₀^2 = a^2 * b^2 ∧ y₀ = p * x₀ + q) :=
sorry

end tangency_condition_and_point_l179_179664


namespace rain_at_least_one_day_probability_l179_179974

-- Definitions based on given conditions
def P_rain_Friday : ℝ := 0.30
def P_rain_Monday : ℝ := 0.20

-- Events probabilities based on independence
def P_no_rain_Friday := 1 - P_rain_Friday
def P_no_rain_Monday := 1 - P_rain_Monday
def P_no_rain_both := P_no_rain_Friday * P_no_rain_Monday

-- The probability of raining at least one day
def P_rain_at_least_one_day := 1 - P_no_rain_both

-- Expected probability
def expected_probability : ℝ := 0.44

theorem rain_at_least_one_day_probability : 
  P_rain_at_least_one_day = expected_probability := by
  sorry

end rain_at_least_one_day_probability_l179_179974


namespace marble_ratio_l179_179888

theorem marble_ratio (W L M : ℕ) (h1 : W = 16) (h2 : L = W + W / 4) (h3 : W + L + M = 60) :
  M / (W + L) = 2 / 3 := 
sorry

end marble_ratio_l179_179888


namespace sum_common_ratios_l179_179758

variable (k p r : ℝ)
variable (hp : p ≠ r)

theorem sum_common_ratios (h : k * p ^ 2 - k * r ^ 2 = 2 * (k * p - k * r)) : 
  p + r = 2 := by
  have hk : k ≠ 0 := sorry -- From the nonconstancy condition
  sorry

end sum_common_ratios_l179_179758


namespace sum_of_numbers_less_than_2_l179_179022

theorem sum_of_numbers_less_than_2:
  ∀ (a b c : ℝ), a = 0.8 → b = 1/2 → c = 0.9 → a < 2 ∧ b < 2 ∧ c < 2 → a + b + c = 2.2 :=
by
  -- We are stating that if a = 0.8, b = 1/2, and c = 0.9, and all are less than 2, then their sum is 2.2
  sorry

end sum_of_numbers_less_than_2_l179_179022


namespace sum_of_squares_of_rates_equals_536_l179_179589

-- Define the biking, jogging, and swimming rates as integers.
variables (b j s : ℤ)

-- Condition: Ed's total distance equation.
def ed_distance_eq : Prop := 3 * b + 2 * j + 4 * s = 80

-- Condition: Sue's total distance equation.
def sue_distance_eq : Prop := 4 * b + 3 * j + 2 * s = 98

-- The main statement to prove.
theorem sum_of_squares_of_rates_equals_536 (hb : b ≥ 0) (hj : j ≥ 0) (hs : s ≥ 0) 
  (h1 : ed_distance_eq b j s) (h2 : sue_distance_eq b j s) :
  b^2 + j^2 + s^2 = 536 :=
by sorry

end sum_of_squares_of_rates_equals_536_l179_179589


namespace compute_value_of_expression_l179_179362

theorem compute_value_of_expression (p q : ℝ) (hpq : 3 * p^2 - 5 * p - 8 = 0) (hq : 3 * q^2 - 5 * q - 8 = 0) (hneq : p ≠ q) :
  3 * (p^2 - q^2) / (p - q) = 5 :=
by
  have hpq_sum : p + q = 5 / 3 := sorry
  exact sorry

end compute_value_of_expression_l179_179362


namespace quad_equiv_proof_l179_179820

theorem quad_equiv_proof (a b : ℝ) (h : a ≠ 0) (hroot : a * 2019^2 + b * 2019 + 2 = 0) :
  ∃ x : ℝ, a * (x - 1)^2 + b * (x - 1) = -2 ∧ x = 2019 :=
sorry

end quad_equiv_proof_l179_179820


namespace cheryl_mms_l179_179056

/-- Cheryl's m&m problem -/
theorem cheryl_mms (c l g d : ℕ) (h1 : c = 25) (h2 : l = 7) (h3 : g = 13) :
  (c - l - g) = d → d = 5 :=
by
  sorry

end cheryl_mms_l179_179056


namespace function_monotone_increasing_l179_179450

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - log x

theorem function_monotone_increasing : ∀ x, 1 ≤ x → (0 < x) → (1 / 2) * x^2 - log x = f x → (∀ y, 1 ≤ y → (0 < y) → (f y ≤ f x)) :=
sorry

end function_monotone_increasing_l179_179450


namespace part1_monotonicity_part2_intersection_l179_179829

noncomputable def f (a x : ℝ) : ℝ := -x * Real.exp (a * x + 1)

theorem part1_monotonicity (a : ℝ) : 
  ∃ interval : Set ℝ, 
    (∀ x ∈ interval, ∃ interval' : Set ℝ, 
      (∀ x' ∈ interval', f a x' ≤ f a x) ∧ 
      (∀ x' ∈ Set.univ \ interval', f a x' > f a x)) :=
sorry

theorem part2_intersection (a b x_1 x_2 : ℝ) (h1 : a > 0) (h2 : b ≠ 0)
  (h3 : f a x_1 = -b * Real.exp 1) (h4 : f a x_2 = -b * Real.exp 1)
  (h5 : x_1 ≠ x_2) : 
  - (1 / Real.exp 1) < a * b ∧ a * b < 0 ∧ a * (x_1 + x_2) < -2 :=
sorry

end part1_monotonicity_part2_intersection_l179_179829


namespace correct_completion_of_sentence_l179_179850

def committee_discussing_problem : Prop := True -- Placeholder for the condition
def problem_expected_to_be_solved_next_week : Prop := True -- Placeholder for the condition

theorem correct_completion_of_sentence 
  (h1 : committee_discussing_problem) 
  (h2 : problem_expected_to_be_solved_next_week) 
  : "hopefully" = "hopefully" :=
by 
  sorry

end correct_completion_of_sentence_l179_179850


namespace hyperbola_asymptotes_l179_179101

variable (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)

theorem hyperbola_asymptotes (e : ℝ) (h_ecc : e = (Real.sqrt 5) / 2)
  (h_hyperbola : e = Real.sqrt (1 + (b^2 / a^2))) :
  (∀ x : ℝ, y = x * (b / a) ∨ y = -x * (b / a)) :=
by
  -- Here, the proof would follow logically from the given conditions.
  sorry

end hyperbola_asymptotes_l179_179101


namespace marble_distribution_l179_179802

theorem marble_distribution (x : ℚ) (total : ℚ) (boy1 : ℚ) (boy2 : ℚ) (boy3 : ℚ) :
  (4 * x + 2) + (2 * x + 1) + (3 * x) = total → total = 62 →
  boy1 = 4 * x + 2 → boy2 = 2 * x + 1 → boy3 = 3 * x →
  boy1 = 254 / 9 ∧ boy2 = 127 / 9 ∧ boy3 = 177 / 9 :=
by
  sorry

end marble_distribution_l179_179802


namespace zachary_pushups_l179_179458

theorem zachary_pushups (C P : ℕ) (h1 : C = 14) (h2 : P + C = 67) : P = 53 :=
by
  rw [h1] at h2
  linarith

end zachary_pushups_l179_179458


namespace tricycles_count_l179_179672

theorem tricycles_count (cars bicycles pickup_trucks tricycles : ℕ) (total_tires : ℕ) : 
  cars = 15 →
  bicycles = 3 →
  pickup_trucks = 8 →
  total_tires = 101 →
  4 * cars + 2 * bicycles + 4 * pickup_trucks + 3 * tricycles = total_tires →
  tricycles = 1 :=
by
  sorry

end tricycles_count_l179_179672


namespace N_even_for_all_permutations_l179_179587

noncomputable def N (a b : Fin 2013 → ℕ) : ℕ :=
  Finset.prod (Finset.univ : Finset (Fin 2013)) (λ i => a i - b i)

theorem N_even_for_all_permutations {a : Fin 2013 → ℕ}
  (h_distinct : Function.Injective a) :
  ∀ b : Fin 2013 → ℕ,
  (∀ i, b i ∈ Finset.univ.image a) →
  ∃ n, n = N a b ∧ Even n :=
by
  -- This is where the proof would go, using the given conditions.
  sorry

end N_even_for_all_permutations_l179_179587


namespace max_value_y_l179_179434

open Real

theorem max_value_y (x : ℝ) (h : -1 < x ∧ x < 1) : 
  ∃ y_max, y_max = 0 ∧ ∀ y, y = x / (x - 1) + x → y ≤ y_max :=
by
  have y : ℝ := x / (x - 1) + x
  use 0
  sorry

end max_value_y_l179_179434


namespace some_number_is_105_l179_179987

def find_some_number (a : ℕ) (num : ℕ) : Prop :=
  a ^ 3 = 21 * 25 * num * 7

theorem some_number_is_105 (a : ℕ) (num : ℕ) (h : a = 105) (h_eq : find_some_number a num) : num = 105 :=
by
  sorry

end some_number_is_105_l179_179987


namespace length_BC_fraction_AD_l179_179459

theorem length_BC_fraction_AD {A B C D : Type} {AB BD AC CD AD BC : ℕ} 
  (h1 : AB = 4 * BD) (h2 : AC = 9 * CD) (h3 : AD = AB + BD) (h4 : AD = AC + CD)
  (h5 : B ≠ A) (h6 : C ≠ A) (h7 : A ≠ D) : BC = AD / 10 :=
by
  sorry

end length_BC_fraction_AD_l179_179459


namespace areas_of_isosceles_triangles_l179_179224

theorem areas_of_isosceles_triangles (A B C : ℝ) (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13)
  (hA : A = 1/2 * a * a) (hB : B = 1/2 * b * b) (hC : C = 1/2 * c * c) :
  A + B = C :=
by
  sorry

end areas_of_isosceles_triangles_l179_179224


namespace minute_hand_40_min_angle_l179_179133

noncomputable def minute_hand_rotation_angle (minutes : ℕ): ℝ :=
  if minutes = 60 then -2 * Real.pi 
  else (minutes / 60) * -2 * Real.pi

theorem minute_hand_40_min_angle :
  minute_hand_rotation_angle 40 = - (4 / 3) * Real.pi :=
by
  sorry

end minute_hand_40_min_angle_l179_179133


namespace solution_set_of_abs_x_plus_one_gt_one_l179_179218

theorem solution_set_of_abs_x_plus_one_gt_one :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2 ∨ x > 0} :=
sorry

end solution_set_of_abs_x_plus_one_gt_one_l179_179218


namespace Trishul_invested_less_than_Raghu_l179_179122

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

end Trishul_invested_less_than_Raghu_l179_179122


namespace geom_seq_a3_value_l179_179900

theorem geom_seq_a3_value (a_n : ℕ → ℝ) (h1 : ∃ r : ℝ, ∀ n : ℕ, a_n (n+1) = a_n (1) * r^n) 
                          (h2 : a_n (2) * a_n (4) = 2 * a_n (3) - 1) :
  a_n (3) = 1 :=
sorry

end geom_seq_a3_value_l179_179900


namespace eval_ff_ff_3_l179_179892

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem eval_ff_ff_3 : f (f (f (f 3))) = 8 :=
  sorry

end eval_ff_ff_3_l179_179892


namespace intersection_of_medians_x_coord_l179_179915

def parabola (x : ℝ) : ℝ := x^2 - 4 * x - 1

theorem intersection_of_medians_x_coord (x_a x_b : ℝ) (y : ℝ) :
  (parabola x_a = y) ∧ (parabola x_b = y) ∧ (parabola 5 = parabola 5) → 
  (2 : ℝ) < ((5 + 4) / 3) :=
sorry

end intersection_of_medians_x_coord_l179_179915


namespace z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l179_179729

section
variable (m : ℝ)
def z : ℂ := (m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) * Complex.I

theorem z_is_real_iff_m_values :
  (z m).im = 0 ↔ m = -3 ∨ m = 5 :=
by sorry

theorem z_in_third_quadrant_iff_m_interval :
  (z m).re < 0 ∧ (z m).im < 0 ↔ m ∈ Set.Ioo (-3) (-2) :=
by sorry
end

end z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l179_179729


namespace find_a_plus_b_l179_179821

variable (a : ℝ) (b : ℝ)
def op (x y : ℝ) : ℝ := x + 2 * y + 3

theorem find_a_plus_b (a b : ℝ) (h1 : op (op (a^3) (a^2)) a = b)
    (h2 : op (a^3) (op (a^2) a) = b) : a + b = 21/8 :=
  sorry

end find_a_plus_b_l179_179821


namespace M_intersection_N_l179_179302

noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 16}
noncomputable def N := {x : ℝ | x ≥ 1 / 3}

theorem M_intersection_N :
  (M ∩ N) = {x : ℝ | 1 / 3 ≤ x ∧ x < 16} := by
sorry

end M_intersection_N_l179_179302


namespace worker_schedule_l179_179061

open Nat

theorem worker_schedule (x : ℕ) :
  24 * 3 + (15 - 3) * x > 408 :=
by
  sorry

end worker_schedule_l179_179061


namespace solution_set_l179_179321

theorem solution_set (x : ℝ) : (2 : ℝ) ^ (|x-2| + |x-4|) > 2^6 ↔ x < 0 ∨ x > 6 :=
by
  sorry

end solution_set_l179_179321


namespace arithmetic_progression_25th_term_l179_179355

theorem arithmetic_progression_25th_term (a1 d : ℤ) (n : ℕ) (h_a1 : a1 = 5) (h_d : d = 7) (h_n : n = 25) :
  a1 + (n - 1) * d = 173 :=
by
  sorry

end arithmetic_progression_25th_term_l179_179355


namespace cos_thirteen_pi_over_four_l179_179116

theorem cos_thirteen_pi_over_four : Real.cos (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_thirteen_pi_over_four_l179_179116


namespace sum_of_coefficients_of_y_terms_l179_179736

theorem sum_of_coefficients_of_y_terms :
  let p := (5 * x + 3 * y + 2) * (2 * x + 6 * y + 7)
  let expanded_p := 10 * x^2 + 36 * x * y + 39 * x + 18 * y^2 + 33 * y + 14
  (36 + 18 + 33) = 87 := by
  sorry

end sum_of_coefficients_of_y_terms_l179_179736


namespace sequence_all_integers_l179_179873

open Nat

def a : ℕ → ℤ
| 0 => 1
| 1 => 1
| n+2 => (a (n+1))^2 + 2 / a n

theorem sequence_all_integers :
  ∀ n : ℕ, ∃ k : ℤ, a n = k :=
by
  sorry

end sequence_all_integers_l179_179873


namespace simon_can_make_blueberry_pies_l179_179179

theorem simon_can_make_blueberry_pies (bush1 bush2 blueberries_per_pie : ℕ) (h1 : bush1 = 100) (h2 : bush2 = 200) (h3 : blueberries_per_pie = 100) : 
  (bush1 + bush2) / blueberries_per_pie = 3 :=
by
  -- Proof goes here
  sorry

end simon_can_make_blueberry_pies_l179_179179


namespace divisor_value_l179_179905

theorem divisor_value :
  ∃ D : ℕ, 
    (242 % D = 11) ∧
    (698 % D = 18) ∧
    (365 % D = 15) ∧
    (527 % D = 13) ∧
    ((242 + 698 + 365 + 527) % D = 9) ∧
    (D = 48) :=
sorry

end divisor_value_l179_179905


namespace row_time_14_24_l179_179571

variable (d c s r : ℝ)

-- Assumptions
def swim_with_current (d c s : ℝ) := s + c = d / 40
def swim_against_current (d c s : ℝ) := s - c = d / 45
def row_against_current (d c r : ℝ) := r - c = d / 15

-- Expected result
def time_to_row_harvard_mit (d c r : ℝ) := d / (r + c) = 14 + 24 / 60

theorem row_time_14_24 :
  swim_with_current d c s ∧
  swim_against_current d c s ∧
  row_against_current d c r →
  time_to_row_harvard_mit d c r :=
by
  sorry

end row_time_14_24_l179_179571


namespace number_difference_l179_179090

theorem number_difference (a b : ℕ) (h1 : a + b = 44) (h2 : 8 * a = 3 * b) : b - a = 20 := by
  sorry

end number_difference_l179_179090


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l179_179524

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l179_179524


namespace a5_a6_values_b_n_general_formula_minimum_value_T_n_l179_179221

section sequence_problems

def sequence_n (n : ℕ) : ℤ :=
if n = 0 then 1
else if n = 1 then 1
else sequence_n (n - 2) + 2 * (-1)^(n - 2)

def b_sequence (n : ℕ) : ℤ :=
sequence_n (2 * n)

def S_n (n : ℕ) : ℤ :=
(n + 1) * (sequence_n n)

def T_n (n : ℕ) : ℤ :=
(S_n (2 * n) - 18)

theorem a5_a6_values :
  sequence_n 4 = -3 ∧ sequence_n 5 = 5 := by
  sorry

theorem b_n_general_formula (n : ℕ) :
  b_sequence n = 2 * n - 1 := by
  sorry

theorem minimum_value_T_n :
  ∃ n, T_n n = -72 := by
  sorry

end sequence_problems

end a5_a6_values_b_n_general_formula_minimum_value_T_n_l179_179221


namespace g_symmetry_value_h_m_interval_l179_179307

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 12)) ^ 2

noncomputable def g (x : ℝ) : ℝ :=
  1 + 1 / 2 * Real.sin (2 * x)

noncomputable def h (x : ℝ) : ℝ :=
  f x + g x

theorem g_symmetry_value (k : ℤ) : 
  g (k * Real.pi / 2 - Real.pi / 12) = (3 + (-1) ^ k) / 4 :=
by
  sorry

theorem h_m_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc (- Real.pi / 12) (5 * Real.pi / 12), |h x - m| ≤ 1) ↔ (1 ≤ m ∧ m ≤ 9 / 4) :=
by
  sorry

end g_symmetry_value_h_m_interval_l179_179307


namespace solve_system_of_equations_solve_linear_inequality_l179_179183

-- Part 1: System of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 5 * x + 2 * y = 25) (h2 : 3 * x + 4 * y = 15) : 
  x = 5 ∧ y = 0 := sorry

-- Part 2: Linear inequality
theorem solve_linear_inequality (x : ℝ) (h : 2 * x - 6 < 3 * x) : 
  x > -6 := sorry

end solve_system_of_equations_solve_linear_inequality_l179_179183


namespace fuel_consumption_rate_l179_179404

theorem fuel_consumption_rate (fuel_left time_left r: ℝ) 
    (h_fuel: fuel_left = 6.3333) 
    (h_time: time_left = 0.6667) 
    (h_rate: r = fuel_left / time_left) : r = 9.5 := 
by
    sorry

end fuel_consumption_rate_l179_179404


namespace arithmetic_mean_of_integers_from_neg3_to_6_l179_179936

def integer_range := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

noncomputable def arithmetic_mean : ℚ :=
  (integer_range.sum : ℚ) / (integer_range.length : ℚ)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  arithmetic_mean = 1.5 := by
  sorry

end arithmetic_mean_of_integers_from_neg3_to_6_l179_179936


namespace y_intercept_of_line_l179_179709

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 :=
by
  sorry

end y_intercept_of_line_l179_179709


namespace kelseys_sisters_age_l179_179928

theorem kelseys_sisters_age :
  ∀ (current_year : ℕ) (kelsey_birth_year : ℕ)
    (kelsey_sister_birth_year : ℕ),
    kelsey_birth_year = 1999 - 25 →
    kelsey_sister_birth_year = kelsey_birth_year - 3 →
    current_year = 2021 →
    current_year - kelsey_sister_birth_year = 50 :=
by
  intros current_year kelsey_birth_year kelsey_sister_birth_year h1 h2 h3
  sorry

end kelseys_sisters_age_l179_179928


namespace probability_one_and_three_painted_faces_l179_179816

-- Define the conditions of the problem
def side_length := 5
def total_unit_cubes := side_length^3
def painted_faces := 2
def unit_cubes_one_painted_face := 26
def unit_cubes_three_painted_faces := 4

-- Define the probability statement in Lean
theorem probability_one_and_three_painted_faces :
  (unit_cubes_one_painted_face * unit_cubes_three_painted_faces : ℝ) / (total_unit_cubes * (total_unit_cubes - 1) / 2) = 52 / 3875 :=
by
  sorry

end probability_one_and_three_painted_faces_l179_179816


namespace roots_cubic_sum_of_cubes_l179_179372

theorem roots_cubic_sum_of_cubes (a b c : ℝ)
  (h1 : Polynomial.eval a (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h4 : a + b + c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 753 :=
by
  sorry

end roots_cubic_sum_of_cubes_l179_179372


namespace jenna_round_trip_pay_l179_179724

-- Definitions based on conditions
def rate : ℝ := 0.40
def one_way_distance : ℝ := 400
def round_trip_distance : ℝ := 2 * one_way_distance

-- Theorem based on the question and correct answer
theorem jenna_round_trip_pay : round_trip_distance * rate = 320 := by
  sorry

end jenna_round_trip_pay_l179_179724


namespace smallest_n_for_multiple_of_11_l179_179782

theorem smallest_n_for_multiple_of_11 
  (x y : ℤ) 
  (hx : x ≡ -2 [ZMOD 11]) 
  (hy : y ≡ 2 [ZMOD 11]) : 
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 11]) ∧ n = 7 :=
sorry

end smallest_n_for_multiple_of_11_l179_179782


namespace inequality_abc_l179_179832

theorem inequality_abc (a b c : ℝ) : a^2 + 4 * b^2 + 8 * c^2 ≥ 3 * a * b + 4 * b * c + 2 * c * a :=
by
  sorry

end inequality_abc_l179_179832


namespace a_can_complete_in_6_days_l179_179899

noncomputable def rate_b : ℚ := 1/8
noncomputable def rate_c : ℚ := 1/12
noncomputable def earnings_total : ℚ := 2340
noncomputable def earnings_b : ℚ := 780.0000000000001

theorem a_can_complete_in_6_days :
  ∃ (rate_a : ℚ), 
    (1 / rate_a) = 6 ∧
    rate_a + rate_b + rate_c = 3 * rate_b ∧
    earnings_b = (rate_b / (rate_a + rate_b + rate_c)) * earnings_total := sorry

end a_can_complete_in_6_days_l179_179899


namespace opposite_of_abs_frac_l179_179137

theorem opposite_of_abs_frac (h : 0 < (1 : ℝ) / 2023) : -|((1 : ℝ) / 2023)| = -(1 / 2023) := by
  sorry

end opposite_of_abs_frac_l179_179137


namespace donut_cubes_eaten_l179_179977

def cube_dimensions := 5

def total_cubes_in_cube : ℕ := cube_dimensions ^ 3

def even_neighbors (faces_sharing_cubes : ℕ) : Prop :=
  faces_sharing_cubes % 2 = 0

/-- A corner cube in a 5x5x5 cube has 3 neighbors. --/
def corner_cube_neighbors := 3

/-- An edge cube in a 5x5x5 cube (excluding corners) has 4 neighbors. --/
def edge_cube_neighbors := 4

/-- A face center cube in a 5x5x5 cube has 5 neighbors. --/
def face_center_cube_neighbors := 5

/-- An inner cube in a 5x5x5 cube has 6 neighbors. --/
def inner_cube_neighbors := 6

/-- Count of edge cubes that share 4 neighbors in a 5x5x5 cube. --/
def edge_cubes_count := 12 * (cube_dimensions - 2)

def inner_cubes_count := (cube_dimensions - 2) ^ 3

theorem donut_cubes_eaten :
  (edge_cubes_count + inner_cubes_count) = 63 := by
  sorry

end donut_cubes_eaten_l179_179977


namespace digit_68th_is_1_l179_179270

noncomputable def largest_n : ℕ :=
  (10^100 - 1) / 14

def digit_at_68th_place (n : ℕ) : ℕ :=
  (n / 10^(68 - 1)) % 10

theorem digit_68th_is_1 : digit_at_68th_place largest_n = 1 :=
sorry

end digit_68th_is_1_l179_179270


namespace negation_example_l179_179927

theorem negation_example (p : ∀ n : ℕ, n^2 < 2^n) : 
  ¬ (∀ n : ℕ, n^2 < 2^n) ↔ ∃ n : ℕ, n^2 ≥ 2^n :=
by sorry

end negation_example_l179_179927


namespace inequality_proof_l179_179144

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  (1 + 1/x) * (1 + 1/y) ≥ 4 :=
sorry

end inequality_proof_l179_179144


namespace subtracted_number_l179_179910

def least_sum_is (x y z : ℤ) (a : ℤ) : Prop :=
  (x - a) * (y - 5) * (z - 2) = 1000 ∧ x + y + z = 7

theorem subtracted_number (x y z a : ℤ) (h : least_sum_is x y z a) : a = 30 :=
sorry

end subtracted_number_l179_179910


namespace functional_equation_solution_l179_179792

theorem functional_equation_solution (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g (f (x + y)) = f x + 2 * (x + y) * g y) : 
  (∀ x : ℝ, f x = 0) ∧ (∀ x : ℝ, g x = 0) :=
sorry

end functional_equation_solution_l179_179792


namespace antecedent_is_50_l179_179402

theorem antecedent_is_50 (antecedent consequent : ℕ) (h_ratio : 4 * consequent = 6 * antecedent) (h_consequent : consequent = 75) : antecedent = 50 := by
  sorry

end antecedent_is_50_l179_179402


namespace how_many_roses_cut_l179_179074

theorem how_many_roses_cut :
  ∀ (r_i r_f r_c : ℕ), r_i = 6 → r_f = 16 → r_c = r_f - r_i → r_c = 10 :=
by
  intros r_i r_f r_c hri hrf heq
  rw [hri, hrf] at heq
  exact heq

end how_many_roses_cut_l179_179074


namespace paint_liters_needed_l179_179964

theorem paint_liters_needed :
  let cost_brushes : ℕ := 20
  let cost_canvas : ℕ := 3 * cost_brushes
  let cost_paint_per_liter : ℕ := 8
  let total_costs : ℕ := 120
  ∃ (liters_of_paint : ℕ), cost_brushes + cost_canvas + cost_paint_per_liter * liters_of_paint = total_costs ∧ liters_of_paint = 5 :=
by
  sorry

end paint_liters_needed_l179_179964


namespace total_earnings_l179_179997

def num_members : ℕ := 20
def candy_bars_per_member : ℕ := 8
def cost_per_candy_bar : ℝ := 0.5

theorem total_earnings :
  (num_members * candy_bars_per_member * cost_per_candy_bar) = 80 :=
by
  sorry

end total_earnings_l179_179997


namespace largest_n_unique_k_l179_179696

theorem largest_n_unique_k :
  ∃ (n : ℕ), ( ∃! (k : ℕ), (5 : ℚ) / 11 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 6 / 11 )
    ∧ n = 359 :=
sorry

end largest_n_unique_k_l179_179696


namespace friendly_sequences_exist_l179_179267

theorem friendly_sequences_exist :
  ∃ (a b : ℕ → ℕ), 
    (∀ n, a n = 2^(n-1)) ∧ 
    (∀ n, b n = 2*n - 1) ∧ 
    (∀ k : ℕ, ∃ (i j : ℕ), k = a i * b j) :=
by
  sorry

end friendly_sequences_exist_l179_179267


namespace six_times_six_l179_179811

-- Definitions based on the conditions
def pattern (n : ℕ) : ℕ := n * 6

-- Theorem statement to be proved
theorem six_times_six : pattern 6 = 36 :=
by {
  sorry
}

end six_times_six_l179_179811


namespace intersection_is_correct_l179_179767

def A : Set ℝ := {x | True}
def B : Set ℝ := {y | y ≥ 0}

theorem intersection_is_correct : A ∩ B = { x | x ≥ 0 } :=
by
  sorry

end intersection_is_correct_l179_179767


namespace minimum_pencils_needed_l179_179507

theorem minimum_pencils_needed (red_pencils blue_pencils : ℕ) (total_pencils : ℕ) 
  (h_red : red_pencils = 7) (h_blue : blue_pencils = 4) (h_total : total_pencils = red_pencils + blue_pencils) :
  (∃ n : ℕ, n = 8 ∧ n ≤ total_pencils ∧ (∀ m : ℕ, m < 8 → (m < red_pencils ∨ m < blue_pencils))) :=
by
  sorry

end minimum_pencils_needed_l179_179507


namespace negate_p_l179_179055

theorem negate_p (p : Prop) :
  (∃ x : ℝ, 0 < x ∧ 3^x < x^3) ↔ (¬ (∀ x : ℝ, 0 < x → 3^x ≥ x^3)) :=
by sorry

end negate_p_l179_179055


namespace initial_population_is_9250_l179_179786

noncomputable def initial_population : ℝ :=
  let final_population := 6514
  let factor := (1.08 * 0.85 * (1.02)^5 * 0.95 * 0.9)
  final_population / factor

theorem initial_population_is_9250 : initial_population = 9250 := by
  sorry

end initial_population_is_9250_l179_179786


namespace six_letter_words_count_l179_179775

def first_letter_possibilities := 26
def second_letter_possibilities := 26
def third_letter_possibilities := 26
def fourth_letter_possibilities := 26

def number_of_six_letter_words : Nat := 
  first_letter_possibilities * 
  second_letter_possibilities * 
  third_letter_possibilities * 
  fourth_letter_possibilities

theorem six_letter_words_count : number_of_six_letter_words = 456976 := by
  sorry

end six_letter_words_count_l179_179775


namespace train_speed_l179_179858

/--A train leaves Delhi at 9 a.m. at a speed of 30 kmph.
Another train leaves at 3 p.m. on the same day and in the same direction.
The two trains meet 720 km away from Delhi.
Prove that the speed of the second train is 120 kmph.-/
theorem train_speed
  (speed_first_train speed_first_kmph : 30 = 30)
  (leave_first_train : Nat)
  (leave_first_9am : 9 = 9)
  (leave_second_train : Nat)
  (leave_second_3pm : 3 = 3)
  (distance_meeting_km : Nat)
  (distance_meeting_720km : 720 = 720) :
  ∃ speed_second_train, speed_second_train = 120 := 
sorry

end train_speed_l179_179858


namespace fish_to_corn_value_l179_179232

/-- In an island kingdom, five fish can be traded for three jars of honey, 
    and a jar of honey can be traded for six cobs of corn. 
    Prove that one fish is worth 3.6 cobs of corn. -/

theorem fish_to_corn_value (f h c : ℕ) (h1 : 5 * f = 3 * h) (h2 : h = 6 * c) : f = 18 * c / 5 := by
  sorry

end fish_to_corn_value_l179_179232


namespace seventh_triangular_number_eq_28_l179_179847

noncomputable def triangular_number (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem seventh_triangular_number_eq_28 :
  triangular_number 7 = 28 :=
by
  sorry

end seventh_triangular_number_eq_28_l179_179847


namespace intersection_points_l179_179107

def equation1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9
def equation2 (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem intersection_points :
  ∃ (x1 y1 x2 y2 : ℝ),
    equation1 x1 y1 ∧ equation2 x1 y1 ∧
    equation1 x2 y2 ∧ equation2 x2 y2 ∧
    (x1, y1) ≠ (x2, y2) ∧
    ∀ (x y : ℝ), equation1 x y ∧ equation2 x y → (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) := sorry

end intersection_points_l179_179107


namespace work_days_difference_l179_179770

theorem work_days_difference (d_a d_b : ℕ) (H1 : d_b = 15) (H2 : d_a = d_b / 3) : 15 - d_a = 10 := by
  sorry

end work_days_difference_l179_179770


namespace number_system_base_l179_179675

theorem number_system_base (a : ℕ) (h : 2 * a^2 + 5 * a + 3 = 136) : a = 7 := 
sorry

end number_system_base_l179_179675


namespace num_four_digit_snappy_numbers_divisible_by_25_l179_179611

def is_snappy (n : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by_25 (n : ℕ) : Prop :=
  let last_two_digits := n % 100
  last_two_digits = 0 ∨ last_two_digits = 25 ∨ last_two_digits = 50 ∨ last_two_digits = 75

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem num_four_digit_snappy_numbers_divisible_by_25 : 
  ∃ n, n = 3 ∧ (∀ x, is_four_digit x ∧ is_snappy x ∧ is_divisible_by_25 x ↔ x = 5225 ∨ x = 0550 ∨ x = 5775)
:=
sorry

end num_four_digit_snappy_numbers_divisible_by_25_l179_179611


namespace cut_out_area_l179_179618

theorem cut_out_area (x : ℝ) (h1 : x * (x - 10) = 1575) : 10 * x - 10 * 10 = 450 := by
  -- Proof to be filled in here
  sorry

end cut_out_area_l179_179618


namespace david_marks_in_biology_l179_179878

theorem david_marks_in_biology (english: ℕ) (math: ℕ) (physics: ℕ) (chemistry: ℕ) (average: ℕ) (biology: ℕ) :
  english = 81 ∧ math = 65 ∧ physics = 82 ∧ chemistry = 67 ∧ average = 76 → (biology = 85) :=
by
  sorry

end david_marks_in_biology_l179_179878


namespace solve_equation_l179_179662

theorem solve_equation:
  ∀ x y z : ℝ, x^2 + 5 * y^2 + 5 * z^2 - 4 * x * z - 2 * y - 4 * y * z + 1 = 0 → 
    x = 4 ∧ y = 1 ∧ z = 2 :=
by
  intros x y z h
  sorry

end solve_equation_l179_179662


namespace pencil_weight_l179_179367

theorem pencil_weight (total_weight : ℝ) (empty_case_weight : ℝ) (num_pencils : ℕ)
  (h1 : total_weight = 11.14) 
  (h2 : empty_case_weight = 0.5) 
  (h3 : num_pencils = 14) :
  (total_weight - empty_case_weight) / num_pencils = 0.76 := by
  sorry

end pencil_weight_l179_179367


namespace tutors_all_work_together_after_360_days_l179_179733

theorem tutors_all_work_together_after_360_days :
  ∀ (n : ℕ), (n > 0) → 
    (∃ k, k > 0 ∧ k = Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 9 10)) ∧ 
     k % 7 = 3) := by
  sorry

end tutors_all_work_together_after_360_days_l179_179733


namespace tens_digit_of_7_pow_2011_l179_179831

-- Define the conditions for the problem
def seven_power := 7
def exponent := 2011
def modulo := 100

-- Define the target function to find the tens digit
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem formally
theorem tens_digit_of_7_pow_2011 : tens_digit (seven_power ^ exponent % modulo) = 4 := by
  sorry

end tens_digit_of_7_pow_2011_l179_179831


namespace at_least_one_f_nonnegative_l179_179514

theorem at_least_one_f_nonnegative 
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m * n > 1) : 
  (m^2 - m ≥ 0) ∨ (n^2 - n ≥ 0) :=
by sorry

end at_least_one_f_nonnegative_l179_179514


namespace calculate_revolutions_l179_179961

noncomputable def number_of_revolutions (diameter distance: ℝ) : ℝ :=
  distance / (Real.pi * diameter)

theorem calculate_revolutions :
  number_of_revolutions 10 5280 = 528 / Real.pi :=
by
  sorry

end calculate_revolutions_l179_179961


namespace proof_of_equivalence_l179_179496

variables (x y : ℝ)

def expression := 49 * x^2 - 36 * y^2
def optionD := (-6 * y + 7 * x) * (6 * y + 7 * x)

theorem proof_of_equivalence : expression x y = optionD x y := 
by sorry

end proof_of_equivalence_l179_179496


namespace find_c_value_l179_179581

variable {x: ℝ}

theorem find_c_value (d e c : ℝ) (h₁ : 6 * d = 18) (h₂ : -15 + 6 * e = -5)
(h₃ : (10 / 3) * c = 15) :
  c = 4.5 :=
by
  sorry

end find_c_value_l179_179581


namespace functional_equality_l179_179708

theorem functional_equality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f y + x ^ 2 + 1) + 2 * x = y + (f (x + 1)) ^ 2) →
  (∀ x : ℝ, f x = x) := 
by
  intro h
  sorry

end functional_equality_l179_179708


namespace geom_mean_between_2_and_8_l179_179376

theorem geom_mean_between_2_and_8 (b : ℝ) (h : b^2 = 16) : b = 4 ∨ b = -4 :=
by
  sorry

end geom_mean_between_2_and_8_l179_179376


namespace base6_to_base10_product_zero_l179_179352

theorem base6_to_base10_product_zero
  (c d e : ℕ)
  (h : (5 * 6^2 + 3 * 6^1 + 2 * 6^0) = (100 * c + 10 * d + e)) :
  (c * e) / 10 = 0 :=
by
  sorry

end base6_to_base10_product_zero_l179_179352


namespace remove_6_maximizes_probability_l179_179481

def original_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

-- Define what it means to maximize the probability of pairs summing to 12
def maximize_probability (l : List Int) : Prop :=
  ∀ x y, x ≠ y → x ∈ l → y ∈ l → x + y = 12

-- Prove that removing 6 maximizes the probability that the sum of the two chosen numbers is 12
theorem remove_6_maximizes_probability :
  maximize_probability (original_list.erase 6) :=
sorry

end remove_6_maximizes_probability_l179_179481


namespace number_of_girls_in_group_l179_179430

-- Define the given conditions
def total_students : ℕ := 20
def prob_of_selecting_girl : ℚ := 2/5

-- State the lean problem for the proof
theorem number_of_girls_in_group : (total_students : ℚ) * prob_of_selecting_girl = 8 := by
  sorry

end number_of_girls_in_group_l179_179430


namespace num_ways_to_queue_ABC_l179_179523

-- Definitions for the problem
def num_people : ℕ := 5
def fixed_order_positions : ℕ := 3

-- Lean statement to prove the problem
theorem num_ways_to_queue_ABC (h : num_people = 5) (h_fop : fixed_order_positions = 3) : 
  (Nat.factorial num_people / Nat.factorial (num_people - fixed_order_positions)) * 1 = 20 := 
by
  sorry

end num_ways_to_queue_ABC_l179_179523


namespace g_at_six_l179_179225

def g (x : ℝ) : ℝ := 2 * x^4 - 19 * x^3 + 30 * x^2 - 12 * x - 72

theorem g_at_six : g 6 = 288 :=
by
  sorry

end g_at_six_l179_179225


namespace problem1_problem2_problem3_l179_179148

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 1 ≤ x then x^2 - 2 * a * x + a
  else if 0 < x then 2 * x + a / x
  else 0 -- Undefined for x ≤ 0

theorem problem1 (a : ℝ) :
  (∀ x y : ℝ, (0 < x ∧ x < y) → f a x < f a y) ↔ (a ≤ -1 / 2) :=
sorry
  
theorem problem2 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ f a x1 = 1 ∧ f a x2 = 1 ∧ f a x3 = 1) ↔ (0 < a ∧ a < 1 / 8) :=
sorry

theorem problem3 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ x - 2 * a) ↔ (0 ≤ a ∧ a ≤ 1 + Real.sqrt 3 / 2) :=
sorry

end problem1_problem2_problem3_l179_179148


namespace paint_after_third_day_l179_179280

def initial_paint := 2
def paint_used_first_day (x : ℕ) := (1 / 2) * x
def remaining_after_first_day (x : ℕ) := x - paint_used_first_day x
def paint_used_second_day (y : ℕ) := (1 / 4) * y
def remaining_after_second_day (y : ℕ) := y - paint_used_second_day y
def paint_used_third_day (z : ℕ) := (1 / 3) * z
def remaining_after_third_day (z : ℕ) := z - paint_used_third_day z

theorem paint_after_third_day :
  remaining_after_third_day 
    (remaining_after_second_day 
      (remaining_after_first_day initial_paint)) = initial_paint / 2 := 
  by
  sorry

end paint_after_third_day_l179_179280


namespace arithmetic_geometric_inequality_l179_179687

variables {a b A1 A2 G1 G2 x y d q : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b)
variables (h₂ : a = x - 3 * d) (h₃ : A1 = x - d) (h₄ : A2 = x + d) (h₅ : b = x + 3 * d)
variables (h₆ : a = y / q^3) (h₇ : G1 = y / q) (h₈ : G2 = y * q) (h₉ : b = y * q^3)
variables (h₁₀ : x - 3 * d = y / q^3) (h₁₁ : x + 3 * d = y * q^3)

theorem arithmetic_geometric_inequality : A1 * A2 ≥ G1 * G2 :=
by {
  sorry
}

end arithmetic_geometric_inequality_l179_179687


namespace sqrt_of_sixteen_l179_179231

theorem sqrt_of_sixteen (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := 
sorry

end sqrt_of_sixteen_l179_179231


namespace sum_remainder_l179_179971

theorem sum_remainder (n : ℤ) : ((9 - n) + (n + 4)) % 9 = 4 := 
by 
  sorry

end sum_remainder_l179_179971


namespace smallest_positive_debt_resolvable_l179_179300

theorem smallest_positive_debt_resolvable :
  ∃ (p g : ℤ), 400 * p + 280 * g = 800 :=
sorry

end smallest_positive_debt_resolvable_l179_179300


namespace find_a_l179_179259

noncomputable section

def f (x a : ℝ) : ℝ := Real.sqrt (1 + a * 4^x)

theorem find_a (a : ℝ) : 
  (∀ (x : ℝ), x ≤ -1 → 1 + a * 4^x ≥ 0) → a = -4 :=
sorry

end find_a_l179_179259


namespace water_fee_20_water_fee_55_l179_179898

-- Define the water charge method as a function
def water_fee (a : ℕ) : ℝ :=
  if a ≤ 15 then 2 * a else 2.5 * a - 7.5

-- Prove the specific cases
theorem water_fee_20 :
  water_fee 20 = 42.5 :=
by sorry

theorem water_fee_55 :
  (∃ a : ℕ, water_fee a = 55) ↔ (a = 25) :=
by sorry

end water_fee_20_water_fee_55_l179_179898


namespace odd_function_symmetry_l179_179573

def f (x : ℝ) : ℝ := x^3 + x

-- Prove that f(-x) = -f(x)
theorem odd_function_symmetry : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end odd_function_symmetry_l179_179573


namespace parallelogram_perimeter_l179_179518

theorem parallelogram_perimeter 
  (EF FG EH : ℝ)
  (hEF : EF = 40) (hFG : FG = 30) (hEH : EH = 50) : 
  2 * (EF + FG) = 140 := 
by 
  rw [hEF, hFG]
  norm_num

end parallelogram_perimeter_l179_179518


namespace max_gcd_lcm_l179_179050

theorem max_gcd_lcm (a b c : ℕ) (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  ∃ x : ℕ, x = Nat.gcd (Nat.lcm a b) c ∧ ∀ y : ℕ, Nat.gcd (Nat.lcm a b) c ≤ 10 :=
sorry

end max_gcd_lcm_l179_179050


namespace camp_cedar_counselors_l179_179478

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) 
(counselors_for_boys : ℕ) (counselors_for_girls : ℕ) 
(total_counselors : ℕ) 
(h1 : boys = 80)
(h2 : girls = 6 * boys - 40)
(h3 : counselors_for_boys = boys / 5)
(h4 : counselors_for_girls = (girls + 11) / 12)  -- +11 to account for rounding up
(h5 : total_counselors = counselors_for_boys + counselors_for_girls) : 
total_counselors = 53 :=
by
  sorry

end camp_cedar_counselors_l179_179478


namespace arithmetic_geometric_mean_identity_l179_179561

theorem arithmetic_geometric_mean_identity (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 96) : x^2 + y^2 = 1408 :=
by
  sorry

end arithmetic_geometric_mean_identity_l179_179561


namespace show_revenue_l179_179640

variable (tickets_first_show : Nat) (tickets_cost : Nat) (multiplicator : Nat)
variable (tickets_second_show : Nat := multiplicator * tickets_first_show)
variable (total_tickets : Nat := tickets_first_show + tickets_second_show)
variable (total_revenue : Nat := total_tickets * tickets_cost)

theorem show_revenue :
    tickets_first_show = 200 ∧ tickets_cost = 25 ∧ multiplicator = 3 →
    total_revenue = 20000 := 
by
    intros h
    sorry

end show_revenue_l179_179640


namespace total_students_l179_179407

theorem total_students (S K : ℕ) (h1 : S = 4000) (h2 : K = 2 * S) :
  S + K = 12000 := by
  sorry

end total_students_l179_179407


namespace value_of_x_minus_y_l179_179670

theorem value_of_x_minus_y (x y : ℝ) (h1 : abs x = 4) (h2 : abs y = 7) (h3 : x + y > 0) :
  x - y = -3 ∨ x - y = -11 :=
sorry

end value_of_x_minus_y_l179_179670


namespace petya_vasya_same_result_l179_179728

theorem petya_vasya_same_result (a b : ℤ) 
  (h1 : b = a + 1)
  (h2 : (a - 1) / (b - 2) = (a + 1) / b) :
  (a / b) = 1 := 
by
  sorry

end petya_vasya_same_result_l179_179728


namespace series_converges_to_half_l179_179110

noncomputable def series_value : ℝ :=
  ∑' (n : ℕ), (n^4 + 3*n^3 + 10*n + 10) / (3^n * (n^4 + 4))

theorem series_converges_to_half : series_value = 1 / 2 :=
  sorry

end series_converges_to_half_l179_179110


namespace christina_speed_limit_l179_179861

theorem christina_speed_limit :
  ∀ (D total_distance friend_distance : ℝ), 
  total_distance = 210 → 
  friend_distance = 3 * 40 → 
  D = total_distance - friend_distance → 
  D / 3 = 30 :=
by
  intros D total_distance friend_distance 
  intros h1 h2 h3 
  sorry

end christina_speed_limit_l179_179861


namespace greg_ate_4_halves_l179_179299

def greg_ate_halves (total_cookies : ℕ) (brad_halves : ℕ) (left_halves : ℕ) : ℕ :=
  2 * total_cookies - (brad_halves + left_halves)

theorem greg_ate_4_halves : greg_ate_halves 14 6 18 = 4 := by
  sorry

end greg_ate_4_halves_l179_179299


namespace closest_to_fraction_l179_179415

theorem closest_to_fraction (options : List ℝ) (h1 : options = [2000, 1500, 200, 2500, 3000]) :
  ∃ closest : ℝ, closest ∈ options ∧ closest = 2000 :=
by
  sorry

end closest_to_fraction_l179_179415


namespace jim_travel_distance_l179_179203

theorem jim_travel_distance
  (john_distance : ℕ := 15)
  (jill_distance : ℕ := john_distance - 5)
  (jim_distance : ℕ := jill_distance * 20 / 100) :
  jim_distance = 2 := 
by
  sorry

end jim_travel_distance_l179_179203


namespace solve_inequality_l179_179149

theorem solve_inequality (x : ℝ) : 2 * (5 * x + 3) ≤ x - 3 * (1 - 2 * x) → x ≤ -3 :=
by
  sorry

end solve_inequality_l179_179149


namespace coupon1_best_discount_l179_179435

noncomputable def listed_prices : List ℝ := [159.95, 179.95, 199.95, 219.95, 239.95]

theorem coupon1_best_discount (x : ℝ) (h₁ : x ∈ listed_prices) (h₂ : x > 120) :
  0.15 * x > 25 ∧ 0.15 * x > 0.20 * (x - 120) ↔ 
  x = 179.95 ∨ x = 199.95 ∨ x = 219.95 ∨ x = 239.95 :=
sorry

end coupon1_best_discount_l179_179435


namespace sufficient_but_not_necessary_not_necessary_l179_179005

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : (|a| > 0) := by
  sorry

theorem not_necessary (a : ℝ) : |a| > 0 → ¬(a = 0) ∧ (a ≠ 0 → |a| > 0 ∧ (¬(a > 0) → (|a| > 0))) := by
  sorry

end sufficient_but_not_necessary_not_necessary_l179_179005


namespace range_of_a_l179_179896

theorem range_of_a (a : ℝ) :
  ((∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ (-4 < a ∧ a ≤ 0)) :=
sorry

end range_of_a_l179_179896


namespace total_leftover_tarts_l179_179009

def cherry_tarts := 0.08
def blueberry_tarts := 0.75
def peach_tarts := 0.08

theorem total_leftover_tarts : cherry_tarts + blueberry_tarts + peach_tarts = 0.91 := by
  sorry

end total_leftover_tarts_l179_179009


namespace solve_comb_eq_l179_179739

open Nat

def comb (n k : ℕ) : ℕ := (factorial n) / ((factorial k) * (factorial (n - k)))
def perm (n k : ℕ) : ℕ := (factorial n) / (factorial (n - k))

theorem solve_comb_eq (x : ℕ) :
  comb (x + 5) x = comb (x + 3) (x - 1) + comb (x + 3) (x - 2) + 3/4 * perm (x + 3) 3 ->
  x = 14 := 
by 
  sorry

end solve_comb_eq_l179_179739


namespace problem_l179_179111

theorem problem (x : ℝ) (h : x^2 + 5 * x - 990 = 0) : x^3 + 6 * x^2 - 985 * x + 1012 = 2002 :=
sorry

end problem_l179_179111


namespace sum_of_three_numbers_l179_179383

theorem sum_of_three_numbers (a b c : ℝ) (h₁ : a + b = 31) (h₂ : b + c = 48) (h₃ : c + a = 59) :
  a + b + c = 69 :=
by
  sorry

end sum_of_three_numbers_l179_179383


namespace error_in_area_l179_179619

theorem error_in_area (s : ℝ) (h : s > 0) :
  let s_measured := 1.02 * s
  let A_actual := s^2
  let A_measured := s_measured^2
  let error := (A_measured - A_actual) / A_actual * 100
  error = 4.04 := by
  sorry

end error_in_area_l179_179619


namespace problem_inequality_l179_179358

theorem problem_inequality (a b c d : ℝ) (h1 : d ≥ 0) (h2 : a + b = 2) (h3 : c + d = 2) :
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 :=
by sorry

end problem_inequality_l179_179358


namespace solve_x_l179_179676

-- Define the function f with the given properties
axiom f : ℝ → ℝ → ℝ
axiom f_assoc : ∀ (a b c : ℝ), f a (f b c) = f (f a b) c
axiom f_inv : ∀ (a : ℝ), f a a = 1

-- Define x and the equation to be solved
theorem solve_x : ∃ (x : ℝ), f x 36 = 216 :=
  sorry

end solve_x_l179_179676


namespace find_quaterns_l179_179860

theorem find_quaterns {
  x y z w : ℝ
} : 
  (x + y = z^2 + w^2 + 6 * z * w) → 
  (x + z = y^2 + w^2 + 6 * y * w) → 
  (x + w = y^2 + z^2 + 6 * y * z) → 
  (y + z = x^2 + w^2 + 6 * x * w) → 
  (y + w = x^2 + z^2 + 6 * x * z) → 
  (z + w = x^2 + y^2 + 6 * x * y) → 
  ( (x, y, z, w) = (0, 0, 0, 0) 
    ∨ (x, y, z, w) = (1/4, 1/4, 1/4, 1/4) 
    ∨ (x, y, z, w) = (-1/4, -1/4, 3/4, -1/4) 
    ∨ (x, y, z, w) = (-1/2, -1/2, 5/2, -1/2)
  ) :=
  sorry

end find_quaterns_l179_179860


namespace average_age_remains_l179_179268

theorem average_age_remains (total_age : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) (initial_people_avg : ℕ) 
                            (total_age_eq : total_age = initial_people_avg * 8) 
                            (new_total_age : ℕ := total_age - leaving_age)
                            (new_avg : ℝ := new_total_age / remaining_people) :
  (initial_people_avg = 25) ∧ (leaving_age = 20) ∧ (remaining_people = 7) → new_avg = 180 / 7 := 
by
  sorry

end average_age_remains_l179_179268


namespace number_of_distinct_sentences_l179_179981

noncomputable def count_distinct_sentences (phrase : String) : Nat :=
  let I_options := 3 -- absent, partially present, fully present
  let II_options := 2 -- absent, present
  let IV_options := 2 -- incomplete or absent
  let III_mandatory := 1 -- always present
  (III_mandatory * IV_options * I_options * II_options) - 1 -- subtract the original sentence

theorem number_of_distinct_sentences :
  count_distinct_sentences "ранним утром на рыбалку улыбающийся Игорь мчался босиком" = 23 :=
by
  sorry

end number_of_distinct_sentences_l179_179981


namespace four_bags_remainder_l179_179401

theorem four_bags_remainder (n : ℤ) (hn : n % 11 = 5) : (4 * n) % 11 = 9 := 
by
  sorry

end four_bags_remainder_l179_179401


namespace simplify_expression_l179_179436

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l179_179436


namespace faster_train_passes_slower_in_54_seconds_l179_179959

-- Definitions of the conditions.
def length_of_train := 75 -- Length of each train in meters.
def speed_faster_train := 46 * 1000 / 3600 -- Speed of the faster train in m/s.
def speed_slower_train := 36 * 1000 / 3600 -- Speed of the slower train in m/s.
def relative_speed := speed_faster_train - speed_slower_train -- Relative speed in m/s.
def total_distance := 2 * length_of_train -- Total distance to cover to pass the slower train.

-- The proof statement.
theorem faster_train_passes_slower_in_54_seconds : total_distance / relative_speed = 54 := by
  sorry

end faster_train_passes_slower_in_54_seconds_l179_179959


namespace proteges_57_l179_179109

def divisors (n : ℕ) : List ℕ := (List.range (n + 1)).filter (λ d => n % d = 0)

def units_digit (n : ℕ) : ℕ := n % 10

def proteges (n : ℕ) : List ℕ := (divisors n).map units_digit

theorem proteges_57 : proteges 57 = [1, 3, 9, 7] :=
sorry

end proteges_57_l179_179109


namespace cost_per_pizza_is_12_l179_179082

def numberOfPeople := 15
def peoplePerPizza := 3
def earningsPerNight := 4
def nightsBabysitting := 15

-- We aim to prove that the cost per pizza is $12
theorem cost_per_pizza_is_12 : 
  (earningsPerNight * nightsBabysitting) / (numberOfPeople / peoplePerPizza) = 12 := 
by 
  sorry

end cost_per_pizza_is_12_l179_179082


namespace triangle_inequality_l179_179777

variables (a b c : ℝ)

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) / (1 + a + b) > c / (1 + c) :=
sorry

end triangle_inequality_l179_179777


namespace smallest_x_division_remainder_l179_179470

theorem smallest_x_division_remainder :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ x = 167 := by
  sorry

end smallest_x_division_remainder_l179_179470


namespace temp_product_l179_179304

theorem temp_product (N : ℤ) (M D : ℤ)
  (h1 : M = D + N)
  (h2 : M - 8 = D + N - 8)
  (h3 : D + 5 = D + 5)
  (h4 : abs ((D + N - 8) - (D + 5)) = 3) :
  (N = 16 ∨ N = 10) →
  16 * 10 = 160 := 
by sorry

end temp_product_l179_179304


namespace greatest_possible_value_of_x_l179_179681

theorem greatest_possible_value_of_x (x : ℕ) (h₁ : x % 4 = 0) (h₂ : x > 0) (h₃ : x^3 < 8000) :
  x ≤ 16 := by
  apply sorry

end greatest_possible_value_of_x_l179_179681


namespace find_fathers_age_l179_179202

noncomputable def sebastian_age : ℕ := 40
noncomputable def age_difference : ℕ := 10
noncomputable def sum_ages_five_years_ago_ratio : ℚ := (3 : ℚ) / 4

theorem find_fathers_age 
  (sebastian_age : ℕ) 
  (age_difference : ℕ) 
  (sum_ages_five_years_ago_ratio : ℚ) 
  (h1 : sebastian_age = 40) 
  (h2 : age_difference = 10) 
  (h3 : sum_ages_five_years_ago_ratio = 3 / 4) 
: ∃ father_age : ℕ, father_age = 85 :=
sorry

end find_fathers_age_l179_179202


namespace not_all_ten_on_boundary_of_same_square_l179_179855

open Function

variable (points : Fin 10 → ℝ × ℝ)

def four_points_on_square (A B C D : ℝ × ℝ) : Prop :=
  -- Define your own predicate to check if 4 points A, B, C, D are on the boundary of some square
  sorry 

theorem not_all_ten_on_boundary_of_same_square :
  (∀ A B C D : Fin 10, four_points_on_square (points A) (points B) (points C) (points D)) →
  ¬ (∃ square : ℝ × ℝ → Prop, ∀ i : Fin 10, square (points i)) :=
by
  intro h
  sorry

end not_all_ten_on_boundary_of_same_square_l179_179855


namespace subsetneq_M_N_l179_179327

def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | (x < 0) ∨ (x > 1 / 2)}

theorem subsetneq_M_N : M ⊂ N :=
by
  sorry

end subsetneq_M_N_l179_179327


namespace total_markup_l179_179337

theorem total_markup (p : ℝ) (o : ℝ) (n : ℝ) (m : ℝ) : 
  p = 48 → o = 0.35 → n = 18 → m = o * p + n → m = 34.8 :=
by
  intro hp ho hn hm
  sorry

end total_markup_l179_179337


namespace common_divisors_greatest_l179_179630

theorem common_divisors_greatest (n : ℕ) (h₁ : ∀ d, d ∣ 120 ∧ d ∣ n ↔ d = 1 ∨ d = 3 ∨ d = 9) : 9 = Nat.gcd 120 n := by
  sorry

end common_divisors_greatest_l179_179630


namespace cube_sum_eq_one_l179_179143

theorem cube_sum_eq_one (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 2) (h3 : abc = 1) : a^3 + b^3 + c^3 = 1 :=
sorry

end cube_sum_eq_one_l179_179143


namespace perfect_square_form_l179_179982

theorem perfect_square_form (N : ℕ) (hN : 0 < N) : 
  ∃ x : ℤ, 2^N - 2 * (N : ℤ) = x^2 ↔ N = 1 ∨ N = 2 :=
by
  sorry

end perfect_square_form_l179_179982


namespace equal_real_roots_of_quadratic_l179_179686

theorem equal_real_roots_of_quadratic (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x + 4 = 0 ∧ (x-4)*(x-4) = 0) ↔ k = 4 ∨ k = -4 :=
by
  sorry

end equal_real_roots_of_quadratic_l179_179686


namespace sum_of_six_terms_l179_179505

variable {a : ℕ → ℝ} {q : ℝ}

/-- Given conditions:
* a is a decreasing geometric sequence with ratio q
-/
def is_decreasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem sum_of_six_terms
  (h_geo : is_decreasing_geometric_sequence a q)
  (h_decreasing : 0 < q ∧ q < 1)
  (h_a1 : 0 < a 1)
  (h_a1a3 : a 1 * a 3 = 1)
  (h_a2a4 : a 2 + a 4 = 5 / 4) :
  (a 1 * (1 - q^6) / (1 - q)) = 63 / 16 := by
  sorry

end sum_of_six_terms_l179_179505


namespace find_nat_number_l179_179813

theorem find_nat_number (N : ℕ) (d : ℕ) (hd : d < 10) (h : N = 5 * d + d) : N = 25 :=
by
  sorry

end find_nat_number_l179_179813


namespace cylindrical_tank_volume_increase_l179_179072

theorem cylindrical_tank_volume_increase (k : ℝ) (H R : ℝ) 
  (hR : R = 10) (hH : H = 5)
  (condition : (π * (10 * k)^2 * 5 - π * 10^2 * 5) = (π * 10^2 * (5 + k) - π * 10^2 * 5)) :
  k = (1 + Real.sqrt 101) / 10 :=
by
  sorry

end cylindrical_tank_volume_increase_l179_179072


namespace monotonic_increasing_interval_l179_179629

noncomputable def f (a : ℝ) (h : 0 < a ∧ a < 1) (x : ℝ) := a ^ (-x^2 + 3 * x + 2)

theorem monotonic_increasing_interval (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ x1 x2 : ℝ, (3 / 2 < x1 ∧ x1 < x2) → f a h x1 < f a h x2 :=
sorry

end monotonic_increasing_interval_l179_179629


namespace find_speed_of_man_in_still_water_l179_179762

noncomputable def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  (v_m + v_s) * 3 = 42 ∧ (v_m - v_s) * 3 = 18

theorem find_speed_of_man_in_still_water (v_s : ℝ) : ∃ v_m : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 10 :=
by
  sorry

end find_speed_of_man_in_still_water_l179_179762


namespace intersection_complement_eq_empty_l179_179468

open Set

variable {α : Type*} (M N U: Set α)

theorem intersection_complement_eq_empty (h : M ⊆ N) : M ∩ (compl N) = ∅ :=
sorry

end intersection_complement_eq_empty_l179_179468


namespace value_of_expression_l179_179023

theorem value_of_expression : 50^4 + 4 * 50^3 + 6 * 50^2 + 4 * 50 + 1 = 6765201 :=
by
  sorry

end value_of_expression_l179_179023


namespace nat_numbers_eq_floor_condition_l179_179840

theorem nat_numbers_eq_floor_condition (a b : ℕ):
  (⌊(a ^ 2 : ℚ) / b⌋₊ + ⌊(b ^ 2 : ℚ) / a⌋₊ = ⌊((a ^ 2 + b ^ 2) : ℚ) / (a * b)⌋₊ + a * b) →
  (b = a ^ 2 + 1) ∨ (a = b ^ 2 + 1) :=
by
  sorry

end nat_numbers_eq_floor_condition_l179_179840


namespace geom_seq_sum_5_terms_l179_179171

theorem geom_seq_sum_5_terms (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = 8 * a 1) (h2 : 2 * (a 2 + 1) = a 1 + a 3) (h_q : q = 2) :
    a 1 * (1 - q^5) / (1 - q) = 62 :=
by
    sorry

end geom_seq_sum_5_terms_l179_179171


namespace simplify_fraction_l179_179857

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l179_179857


namespace number_of_teams_in_BIG_N_l179_179295

theorem number_of_teams_in_BIG_N (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end number_of_teams_in_BIG_N_l179_179295


namespace alex_basketball_points_l179_179348

theorem alex_basketball_points (f t s : ℕ) 
  (h : f + t + s = 40) 
  (points_scored : ℝ := 0.8 * f + 0.3 * t + s) :
  points_scored = 28 :=
sorry

end alex_basketball_points_l179_179348


namespace a2_value_for_cubic_expansion_l179_179411

theorem a2_value_for_cubic_expansion (x a0 a1 a2 a3 : ℝ) : 
  (x ^ 3 = a0 + a1 * (x - 2) + a2 * (x - 2) ^ 2 + a3 * (x - 2) ^ 3) → a2 = 6 := by
  sorry

end a2_value_for_cubic_expansion_l179_179411


namespace quotient_of_fifths_l179_179749

theorem quotient_of_fifths : (2 / 5) / (1 / 5) = 2 := 
  by 
    sorry

end quotient_of_fifths_l179_179749


namespace stolen_bones_is_two_l179_179683

/-- Juniper's initial number of bones -/
def initial_bones : ℕ := 4

/-- Juniper's bones after receiving more bones -/
def doubled_bones : ℕ := initial_bones * 2

/-- Juniper's remaining number of bones after theft -/
def remaining_bones : ℕ := 6

/-- Number of bones stolen by the neighbor's dog -/
def stolen_bones : ℕ := doubled_bones - remaining_bones

theorem stolen_bones_is_two : stolen_bones = 2 := sorry

end stolen_bones_is_two_l179_179683


namespace find_x_l179_179233

-- Definitions based on conditions
def parabola_eq (y x p : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola (p : ℝ) : Prop := ∃ y x, parabola_eq y x p ∧ (x = 1) ∧ (y = 2)
def valid_p (p : ℝ) : Prop := p > 0
def dist_to_focus (x : ℝ) : ℝ := 1
def dist_to_line (x : ℝ) : ℝ := abs (x + 1)

-- Main statement to be proven
theorem find_x (p : ℝ) (h1 : point_on_parabola p) (h2 : valid_p p) :
  ∃ x, dist_to_focus x = dist_to_line x ∧ x = 1 :=
sorry

end find_x_l179_179233


namespace range_of_p_l179_179156

noncomputable def success_prob_4_engine (p : ℝ) : ℝ :=
  4 * p^3 * (1 - p) + p^4

noncomputable def success_prob_2_engine (p : ℝ) : ℝ :=
  p^2

theorem range_of_p (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  success_prob_4_engine p > success_prob_2_engine p ↔ (1/3 < p ∧ p < 1) :=
by
  sorry

end range_of_p_l179_179156


namespace find_function_l179_179801

/-- Any function f : ℝ → ℝ satisfying the two given conditions must be of the form f(x) = cx where |c| ≤ 1. -/
theorem find_function (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, x ≠ 0 → x * (f (x + 1) - f x) = f x)
  (h2 : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ c : ℝ, (∀ x : ℝ, f x = c * x) ∧ |c| ≤ 1 :=
by
  sorry

end find_function_l179_179801


namespace smallest_possible_stamps_l179_179623

theorem smallest_possible_stamps (M : ℕ) : 
  ((M % 5 = 2) ∧ (M % 7 = 2) ∧ (M % 9 = 2) ∧ (M > 2)) → M = 317 := 
by 
  sorry

end smallest_possible_stamps_l179_179623


namespace number_of_sequences_less_than_1969_l179_179521

theorem number_of_sequences_less_than_1969 :
  (∃ S : ℕ → ℕ, (∀ n : ℕ, S (n + 1) > (S n) * (S n)) ∧ S 1969 = 1969) →
  ∃ N : ℕ, N < 1969 :=
sorry

end number_of_sequences_less_than_1969_l179_179521


namespace quadratic_root_m_value_l179_179024

theorem quadratic_root_m_value (m : ℝ) (x : ℝ) (h : x = 1) (hx : x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end quadratic_root_m_value_l179_179024


namespace rightmost_three_digits_of_3_pow_2023_l179_179706

theorem rightmost_three_digits_of_3_pow_2023 :
  (3^2023) % 1000 = 787 := 
sorry

end rightmost_three_digits_of_3_pow_2023_l179_179706


namespace triangle_angle_sum_l179_179451

theorem triangle_angle_sum {x : ℝ} (h : 60 + 5 * x + 3 * x = 180) : x = 15 :=
by
  sorry

end triangle_angle_sum_l179_179451


namespace goldfish_graph_discrete_points_l179_179727

theorem goldfish_graph_discrete_points : 
  ∀ n : ℤ, 1 ≤ n ∧ n ≤ 10 → ∃ C : ℤ, C = 20 * n + 10 ∧ ∀ m : ℤ, (1 ≤ m ∧ m ≤ 10 ∧ m ≠ n) → C ≠ (20 * m + 10) :=
by
  sorry

end goldfish_graph_discrete_points_l179_179727


namespace fraction_raised_to_zero_l179_179931

theorem fraction_raised_to_zero:
  (↑(-4305835) / ↑1092370457 : ℚ)^0 = 1 := 
by
  sorry

end fraction_raised_to_zero_l179_179931


namespace annual_interest_rate_equivalent_l179_179621

noncomputable def quarterly_compound_rate : ℝ := 1 + 0.02
noncomputable def annual_compound_amount : ℝ := quarterly_compound_rate ^ 4

theorem annual_interest_rate_equivalent : 
  (annual_compound_amount - 1) * 100 = 8.24 := 
by
  sorry

end annual_interest_rate_equivalent_l179_179621


namespace smallest_triangle_perimeter_l179_179449

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end smallest_triangle_perimeter_l179_179449


namespace grasshopper_flea_adjacency_l179_179191

-- We assume that grid cells are indexed by pairs of integers (i.e., positions in ℤ × ℤ)
-- Red cells and white cells are represented as sets of these positions
variable (red_cells : Set (ℤ × ℤ))
variable (white_cells : Set (ℤ × ℤ))

-- We define that the grasshopper can only jump between red cells
def grasshopper_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ red_cells ∧ new_pos ∈ red_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- We define that the flea can only jump between white cells
def flea_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ white_cells ∧ new_pos ∈ white_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- Main theorem to be proved
theorem grasshopper_flea_adjacency (g_start : ℤ × ℤ) (f_start : ℤ × ℤ) :
    g_start ∈ red_cells → f_start ∈ white_cells →
    ∃ g1 g2 g3 f1 f2 f3 : ℤ × ℤ,
    (
      grasshopper_jump red_cells g_start g1 ∧
      grasshopper_jump red_cells g1 g2 ∧
      grasshopper_jump red_cells g2 g3
    ) ∧ (
      flea_jump white_cells f_start f1 ∧
      flea_jump white_cells f1 f2 ∧
      flea_jump white_cells f2 f3
    ) ∧
    (abs (g3.1 - f3.1) + abs (g3.2 - f3.2) = 1) :=
  sorry

end grasshopper_flea_adjacency_l179_179191


namespace range_for_m_l179_179717

def A := { x : ℝ | x^2 - 3 * x - 10 < 0 }
def B (m : ℝ) := { x : ℝ | m + 1 < x ∧ x < 1 - 3 * m }

theorem range_for_m (m : ℝ) (h : ∀ x, x ∈ A ∪ B m ↔ x ∈ B m) : m ≤ -3 := sorry

end range_for_m_l179_179717


namespace select_k_numbers_l179_179186

theorem select_k_numbers (a : ℕ → ℝ) (k : ℕ) (h1 : ∀ n, 0 < a n) 
  (h2 : ∀ n m, n < m → a n ≥ a m) (h3 : a 1 = 1 / (2 * k)) 
  (h4 : ∑' n, a n = 1) :
  ∃ (f : ℕ → ℕ) (hf : ∀ i j, i ≠ j → f i ≠ f j), 
    (∀ i, i < k → a (f i) > 1/2 * a (f 0)) :=
by
  sorry

end select_k_numbers_l179_179186


namespace star_neg5_4_star_neg3_neg6_l179_179037

-- Definition of the new operation
def star (a b : ℤ) : ℤ := 2 * a * b - b / 2

-- The first proof problem
theorem star_neg5_4 : star (-5) 4 = -42 := by sorry

-- The second proof problem
theorem star_neg3_neg6 : star (-3) (-6) = 39 := by sorry

end star_neg5_4_star_neg3_neg6_l179_179037


namespace max_sum_consecutive_integers_less_360_l179_179532

theorem max_sum_consecutive_integers_less_360 :
  ∃ n : ℤ, n * (n + 1) < 360 ∧ (n + (n + 1)) = 37 :=
by
  sorry

end max_sum_consecutive_integers_less_360_l179_179532


namespace debts_equal_in_25_days_l179_179918

-- Define the initial debts and the interest rates
def Darren_initial_debt : ℝ := 200
def Darren_interest_rate : ℝ := 0.08
def Fergie_initial_debt : ℝ := 300
def Fergie_interest_rate : ℝ := 0.04

-- Define the debts as a function of days passed t
def Darren_debt (t : ℝ) : ℝ := Darren_initial_debt * (1 + Darren_interest_rate * t)
def Fergie_debt (t : ℝ) : ℝ := Fergie_initial_debt * (1 + Fergie_interest_rate * t)

-- Prove that Darren and Fergie will owe the same amount in 25 days
theorem debts_equal_in_25_days : ∃ t, Darren_debt t = Fergie_debt t ∧ t = 25 := by
  sorry

end debts_equal_in_25_days_l179_179918


namespace cubes_divisible_by_9_l179_179014

theorem cubes_divisible_by_9 (n: ℕ) (h: n > 0) : 9 ∣ n^3 + (n + 1)^3 + (n + 2)^3 :=
by 
  sorry

end cubes_divisible_by_9_l179_179014


namespace solution_intervals_l179_179388

noncomputable def cubic_inequality (x : ℝ) : Prop :=
  x^3 - 3 * x^2 - 4 * x - 12 ≤ 0

noncomputable def linear_inequality (x : ℝ) : Prop :=
  2 * x + 6 > 0

theorem solution_intervals :
  { x : ℝ | cubic_inequality x ∧ linear_inequality x } = { x | -2 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end solution_intervals_l179_179388


namespace total_money_left_l179_179809

theorem total_money_left (david_start john_start emily_start : ℝ) 
  (david_percent_left john_percent_spent emily_percent_spent : ℝ) : 
  (david_start = 3200) → 
  (david_percent_left = 0.65) → 
  (john_start = 2500) → 
  (john_percent_spent = 0.60) → 
  (emily_start = 4000) → 
  (emily_percent_spent = 0.45) → 
  let david_spent := david_start / (1 + david_percent_left)
  let david_remaining := david_start - david_spent
  let john_remaining := john_start * (1 - john_percent_spent)
  let emily_remaining := emily_start * (1 - emily_percent_spent)
  david_remaining + john_remaining + emily_remaining = 4460.61 :=
by
  sorry

end total_money_left_l179_179809


namespace problem1_problem2_problem3_problem4_l179_179969

-- Question 1
theorem problem1 (a b : ℝ) (h : 5 * a + 3 * b = -4) : 2 * (a + b) + 4 * (2 * a + b) = -8 :=
by
  sorry

-- Question 2
theorem problem2 (a : ℝ) (h : a^2 + a = 3) : 2 * a^2 + 2 * a + 2023 = 2029 :=
by
  sorry

-- Question 3
theorem problem3 (a b : ℝ) (h : a - 2 * b = -3) : 3 * (a - b) - 7 * a + 11 * b + 2 = 14 :=
by
  sorry

-- Question 4
theorem problem4 (a b : ℝ) 
  (h1 : a^2 + 2 * a * b = -5) 
  (h2 : a * b - 2 * b^2 = -3) : a^2 + a * b + 2 * b^2 = -2 :=
by
  sorry

end problem1_problem2_problem3_problem4_l179_179969


namespace reflection_transformation_l179_179480

structure Point (α : Type) :=
(x : α)
(y : α)

def reflect_x_axis (p : Point ℝ) : Point ℝ :=
  {x := p.x, y := -p.y}

def reflect_x_eq_3 (p : Point ℝ) : Point ℝ :=
  {x := 6 - p.x, y := p.y}

def D : Point ℝ := {x := 4, y := 1}

def D' := reflect_x_axis D

def D'' := reflect_x_eq_3 D'

theorem reflection_transformation :
  D'' = {x := 2, y := -1} :=
by
  -- We skip the proof here
  sorry

end reflection_transformation_l179_179480


namespace sequence_fifth_term_l179_179271

theorem sequence_fifth_term (a : ℤ) (d : ℤ) (n : ℕ) (a_n : ℤ) :
  a_n = 89 ∧ d = 11 ∧ n = 5 → a + (n-1) * -d = 45 := 
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  exact sorry

end sequence_fifth_term_l179_179271


namespace factorize_x2_minus_2x_plus_1_l179_179907

theorem factorize_x2_minus_2x_plus_1 :
  ∀ (x : ℝ), x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  intro x
  linarith

end factorize_x2_minus_2x_plus_1_l179_179907


namespace solve_ineq_l179_179844

noncomputable def inequality (x : ℝ) : Prop :=
  (x^2 / (x+1)) ≥ (3 / (x+1) + 3)

theorem solve_ineq :
  { x : ℝ | inequality x } = { x : ℝ | x ≤ -6 ∨ (-1 < x ∧ x ≤ 3) } := sorry

end solve_ineq_l179_179844


namespace fraction_product_l179_179673

theorem fraction_product : (2 * (-4)) / (9 * 5) = -8 / 45 :=
  by sorry

end fraction_product_l179_179673


namespace line_intersects_x_axis_between_A_and_B_l179_179817

theorem line_intersects_x_axis_between_A_and_B (a : ℝ) :
  (∀ x, (x = 1 ∨ x = 3) → (2 * x + (3 - a) = 0)) ↔ 5 ≤ a ∧ a ≤ 9 :=
by
  sorry

end line_intersects_x_axis_between_A_and_B_l179_179817


namespace cubic_three_real_roots_l179_179701

theorem cubic_three_real_roots (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧
   x₁ ^ 3 - 3 * x₁ - a = 0 ∧
   x₂ ^ 3 - 3 * x₂ - a = 0 ∧
   x₃ ^ 3 - 3 * x₃ - a = 0) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end cubic_three_real_roots_l179_179701


namespace recruits_total_l179_179975

theorem recruits_total (x y z : ℕ) (total_people : ℕ) :
  (x = total_people - 51) ∧
  (y = total_people - 101) ∧
  (z = total_people - 171) ∧
  (x = 4 * y ∨ y = 4 * z ∨ x = 4 * z) ∧
  (∃ total_people, total_people = 211) :=
sorry

end recruits_total_l179_179975


namespace Carrie_has_50_dollars_left_l179_179330

/-
Conditions:
1. initial_amount = 91
2. sweater_cost = 24
3. tshirt_cost = 6
4. shoes_cost = 11
-/
def initial_amount : ℕ := 91
def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11

/-
Question:
How much money does Carrie have left?
-/
def total_spent : ℕ := sweater_cost + tshirt_cost + shoes_cost
def money_left : ℕ := initial_amount - total_spent

def proof_statement : Prop := money_left = 50

theorem Carrie_has_50_dollars_left : proof_statement :=
by
  sorry

end Carrie_has_50_dollars_left_l179_179330


namespace find_value_of_M_l179_179993

theorem find_value_of_M (M : ℝ) (h : 0.2 * M = 0.6 * 1230) : M = 3690 :=
by {
  sorry
}

end find_value_of_M_l179_179993


namespace transform_equation_to_square_form_l179_179617

theorem transform_equation_to_square_form : 
  ∀ x : ℝ, (x^2 - 6 * x = 0) → ∃ m n : ℝ, (x + m) ^ 2 = n ∧ m = -3 ∧ n = 9 := 
sorry

end transform_equation_to_square_form_l179_179617


namespace person_age_l179_179168

variable (x : ℕ) -- Define the variable for age

-- State the condition as a hypothesis
def condition (x : ℕ) : Prop :=
  3 * (x + 3) - 3 * (x - 3) = x

-- State the theorem to be proved
theorem person_age (x : ℕ) (h : condition x) : x = 18 := 
sorry

end person_age_l179_179168


namespace smallest_n_for_abc_factorials_l179_179942

theorem smallest_n_for_abc_factorials (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b + c = 2006) :
  ∃ m n : ℕ, (¬ ∃ k : ℕ, m = 10 * k) ∧ a.factorial * b.factorial * c.factorial = m * 10^n ∧ n = 492 :=
sorry

end smallest_n_for_abc_factorials_l179_179942


namespace new_ratio_is_three_half_l179_179939

theorem new_ratio_is_three_half (F J : ℕ) (h1 : F * 4 = J * 5) (h2 : J = 120) :
  ((F + 30) : ℚ) / J = 3 / 2 :=
by
  sorry

end new_ratio_is_three_half_l179_179939


namespace smallest_integer_coprime_with_462_l179_179222

theorem smallest_integer_coprime_with_462 :
  ∃ n, n > 1 ∧ Nat.gcd n 462 = 1 ∧ ∀ m, m > 1 ∧ Nat.gcd m 462 = 1 → n ≤ m → n = 13 := by
  sorry

end smallest_integer_coprime_with_462_l179_179222


namespace boa_constrictor_length_l179_179691

theorem boa_constrictor_length (garden_snake_length : ℕ) (boa_multiplier : ℕ) (boa_length : ℕ) 
    (h1 : garden_snake_length = 10) (h2 : boa_multiplier = 7) (h3 : boa_length = garden_snake_length * boa_multiplier) : 
    boa_length = 70 := 
sorry

end boa_constrictor_length_l179_179691


namespace arithmetic_progression_of_squares_l179_179033

theorem arithmetic_progression_of_squares 
  (a b c : ℝ)
  (h : 1 / (a + b) - 1 / (a + c) = 1 / (b + c) - 1 / (a + c)) :
  2 * b^2 = a^2 + c^2 :=
by
  sorry

end arithmetic_progression_of_squares_l179_179033


namespace KellyGamesLeft_l179_179230

def initialGames : ℕ := 121
def gamesGivenAway : ℕ := 99

theorem KellyGamesLeft : initialGames - gamesGivenAway = 22 := by
  sorry

end KellyGamesLeft_l179_179230


namespace matrix_multiplication_correct_l179_179641

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, 3, -1], ![1, -2, 5], ![0, 6, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 0, 4], ![3, 2, -1], ![0, 4, -2]]

def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![11, 2, 7], ![-5, 16, -4], ![18, 16, -8]]

theorem matrix_multiplication_correct :
  A * B = C :=
by
  sorry

end matrix_multiplication_correct_l179_179641


namespace expand_expression_l179_179106

variable {R : Type} [CommRing R]
variable (a b x : R)

theorem expand_expression (a b x : R) :
  (a * x^2 + b) * (5 * x^3) = 35 * x^5 + (-15) * x^3 :=
by
  -- The proof goes here
  sorry

end expand_expression_l179_179106


namespace correct_division_l179_179354

theorem correct_division (a : ℝ) : a^8 / a^2 = a^6 := by 
  sorry

end correct_division_l179_179354


namespace garden_length_l179_179309

theorem garden_length (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 240) : l = 80 :=
by
  sorry

end garden_length_l179_179309


namespace difference_approx_l179_179440

-- Let L be the larger number and S be the smaller number
variables (L S : ℝ)

-- Conditions given:
-- 1. L is approximately 1542.857
def approx_L : Prop := abs (L - 1542.857) < 1

-- 2. When L is divided by S, quotient is 8 and remainder is 15
def division_condition : Prop := L = 8 * S + 15

-- The theorem stating the difference L - S is approximately 1351.874
theorem difference_approx (hL : approx_L L) (hdiv : division_condition L S) :
  abs ((L - S) - 1351.874) < 1 :=
sorry

#check difference_approx

end difference_approx_l179_179440


namespace problem1_solution_problem2_solution_l179_179906

noncomputable def problem1 (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) : Real := 
  A

noncomputable def problem2 (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) : Real :=
  a

theorem problem1_solution (a b : ℝ) (A B : ℝ) (h1 : b * Real.cos A - a * Real.sin B = 0) :
  problem1 a b A B h1 = Real.pi / 4 :=
sorry

theorem problem2_solution (a b c : ℝ) (A : ℝ) (area : ℝ) (h1 : b = Real.sqrt 2) (h2 : A = Real.pi / 4) (h3 : area = 1) :
  problem2 a b c A area h1 h2 h3 = Real.sqrt 2 :=
sorry

end problem1_solution_problem2_solution_l179_179906


namespace kelly_can_buy_ten_pounds_of_mangoes_l179_179264

theorem kelly_can_buy_ten_pounds_of_mangoes (h : 0.5 * 1.2 = 0.60) : 12 / (2 * 0.60) = 10 :=
  by
    sorry

end kelly_can_buy_ten_pounds_of_mangoes_l179_179264


namespace gre_exam_month_l179_179041

def months_of_year := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def start_month := "June"
def preparation_duration := 5

theorem gre_exam_month :
  months_of_year[(months_of_year.indexOf start_month + preparation_duration) % 12] = "November" := by
  sorry

end gre_exam_month_l179_179041


namespace min_value_expr_l179_179390

noncomputable def expr (θ : Real) : Real :=
  3 * (Real.cos θ) + 2 / (Real.sin θ) + 2 * Real.sqrt 2 * (Real.tan θ)

theorem min_value_expr :
  ∃ (θ : Real), 0 < θ ∧ θ < Real.pi / 2 ∧ expr θ = (7 * Real.sqrt 2) / 2 := 
by
  sorry

end min_value_expr_l179_179390


namespace triangle_ABC_area_l179_179663

-- We define the basic structure of a triangle and its properties
structure Triangle :=
(base : ℝ)
(height : ℝ)
(right_angled_at : ℝ)

-- Define the specific triangle ABC with given properties
def triangle_ABC : Triangle := {
  base := 12,
  height := 15,
  right_angled_at := 90 -- since right-angled at C
}

-- Given conditions, we need to prove the area is 90 square cm
theorem triangle_ABC_area : 1/2 * triangle_ABC.base * triangle_ABC.height = 90 := 
by 
  sorry

end triangle_ABC_area_l179_179663


namespace volume_difference_l179_179669

theorem volume_difference (x1 x2 x3 Vmin Vmax : ℝ)
  (hx1 : 0.5 < x1 ∧ x1 < 1.5)
  (hx2 : 0.5 < x2 ∧ x2 < 1.5)
  (hx3 : 2016.5 < x3 ∧ x3 < 2017.5)
  (rV : 2017 = Nat.floor (x1 * x2 * x3))
  : abs (Vmax - Vmin) = 4035 := 
sorry

end volume_difference_l179_179669


namespace range_of_m_l179_179461

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) ↔ m < -1 :=
by
  sorry

end range_of_m_l179_179461


namespace exponential_fixed_point_l179_179104

variable (a : ℝ)

noncomputable def f (x : ℝ) := a^(x - 1) + 3

theorem exponential_fixed_point (ha1 : a > 0) (ha2 : a ≠ 1) : f a 1 = 4 :=
by
  sorry

end exponential_fixed_point_l179_179104


namespace polynomial_factorization_l179_179769

theorem polynomial_factorization (x y : ℝ) : -(2 * x - y) * (2 * x + y) = -4 * x ^ 2 + y ^ 2 :=
by sorry

end polynomial_factorization_l179_179769


namespace modulo_17_residue_l179_179737

theorem modulo_17_residue : (3^4 + 6 * 49 + 8 * 137 + 7 * 34) % 17 = 5 := 
by
  sorry

end modulo_17_residue_l179_179737


namespace min_colored_cells_65x65_l179_179322

def grid_size : ℕ := 65
def total_cells : ℕ := grid_size * grid_size

-- Define a function that calculates the minimum number of colored cells needed
noncomputable def min_colored_cells_needed (N: ℕ) : ℕ := (N * N) / 3

-- The main theorem stating the proof problem
theorem min_colored_cells_65x65 (H: grid_size = 65) : 
  min_colored_cells_needed grid_size = 1408 :=
by {
  sorry
}

end min_colored_cells_65x65_l179_179322


namespace power_function_evaluation_l179_179467

theorem power_function_evaluation (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 4 = 2) : f 16 = 4 :=
by
  sorry

end power_function_evaluation_l179_179467


namespace triangle_area_x_value_l179_179213

theorem triangle_area_x_value :
  ∃ x : ℝ, x > 0 ∧ 100 = (1 / 2) * x * (3 * x) ∧ x = 10 * Real.sqrt 6 / 3 :=
sorry

end triangle_area_x_value_l179_179213


namespace common_ratio_of_series_l179_179130

theorem common_ratio_of_series (a1 a2 : ℚ) (h1 : a1 = 5/6) (h2 : a2 = -4/9) :
  (a2 / a1) = -8/15 :=
by
  sorry

end common_ratio_of_series_l179_179130


namespace sum_coordinates_D_l179_179341

theorem sum_coordinates_D
    (M : (ℝ × ℝ))
    (C : (ℝ × ℝ))
    (D : (ℝ × ℝ))
    (H_M_midpoint : M = (5, 9))
    (H_C_coords : C = (11, 5))
    (H_M_def : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
    (D.1 + D.2) = 12 := 
by
  sorry
 
end sum_coordinates_D_l179_179341


namespace alexander_spends_total_amount_l179_179510

theorem alexander_spends_total_amount :
  (5 * 1) + (2 * 2) = 9 :=
by
  sorry

end alexander_spends_total_amount_l179_179510


namespace evaluate_fraction_l179_179057

-- Let's restate the problem in Lean
theorem evaluate_fraction :
  (∃ q, (2024 / 2023 - 2023 / 2024) = 4047 / q) :=
by
  -- Substitute a = 2023
  let a := 2023
  -- Provide the value we expect for q to hold in the reduced fraction.
  use (a * (a + 1)) -- The expected denominator
  -- The proof for the theorem is omitted here
  sorry

end evaluate_fraction_l179_179057


namespace sum_of_exponents_l179_179568

def power_sum_2021 (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ) : Prop :=
  (∀ k, 1 ≤ k ∧ k ≤ r → (a k = 1 ∨ a k = -1)) ∧
  (a 1 * 3 ^ n 1 + a 2 * 3 ^ n 2 + a 3 * 3 ^ n 3 + a 4 * 3 ^ n 4 + a 5 * 3 ^ n 5 + a 6 * 3 ^ n 6 = 2021) ∧
  (n 1 = 7 ∧ n 2 = 5 ∧ n 3 = 4 ∧ n 4 = 2 ∧ n 5 = 1 ∧ n 6 = 0) ∧
  (a 1 = 1 ∧ a 2 = -1 ∧ a 3 = 1 ∧ a 4 = -1 ∧ a 5 = 1 ∧ a 6 = -1)

theorem sum_of_exponents : ∃ (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ), power_sum_2021 a n r ∧ (n 1 + n 2 + n 3 + n 4 + n 5 + n 6 = 19) :=
by {
  sorry
}

end sum_of_exponents_l179_179568


namespace golden_section_search_third_point_l179_179735

noncomputable def golden_ratio : ℝ := 0.618

theorem golden_section_search_third_point :
  let L₀ := 1000
  let U₀ := 2000
  let d₀ := U₀ - L₀
  let x₁ := U₀ - golden_ratio * d₀
  let x₂ := L₀ + golden_ratio * d₀
  let d₁ := U₀ - x₁
  let x₃ := x₁ + golden_ratio * d₁
  x₃ = 1764 :=
by
  sorry

end golden_section_search_third_point_l179_179735


namespace h_at_3_l179_179705

theorem h_at_3 :
  ∃ h : ℤ → ℤ,
    (∀ x, (x^7 - 1) * h x = (x+1) * (x^2 + 1) * (x^4 + 1) - (x-1)) →
    h 3 = 3 := 
sorry

end h_at_3_l179_179705


namespace waste_in_scientific_notation_l179_179814

def water_waste_per_person : ℝ := 0.32
def number_of_people : ℝ := 10^6

def total_daily_waste : ℝ := water_waste_per_person * number_of_people

def scientific_notation (x : ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

theorem waste_in_scientific_notation :
  scientific_notation total_daily_waste ∧ total_daily_waste = 3.2 * 10^5 :=
by
  sorry

end waste_in_scientific_notation_l179_179814


namespace part1_part2_l179_179967

open Real

def f (x a : ℝ) := abs (x + 2 * a) + abs (x - 1)

section part1

variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

end part1

section part2

noncomputable def g (a : ℝ) := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part2 {a : ℝ} (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end part2

end part1_part2_l179_179967


namespace trash_can_ratio_l179_179139

theorem trash_can_ratio (streets_trash_cans total_trash_cans : ℕ) 
(h_streets : streets_trash_cans = 14) 
(h_total : total_trash_cans = 42) : 
(total_trash_cans - streets_trash_cans) / streets_trash_cans = 2 :=
by {
  sorry
}

end trash_can_ratio_l179_179139


namespace geometric_sequence_a5_value_l179_179841

theorem geometric_sequence_a5_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n m : ℕ, a n = a 0 * r ^ n)
  (h_condition : a 3 * a 7 = 8) :
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_a5_value_l179_179841


namespace range_of_3t_plus_s_l179_179343

noncomputable def f : ℝ → ℝ := sorry

def is_increasing (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

def symmetric_about (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x, f (x - a) = b - f (a - x)

def satisfies_inequality (s t : ℝ) (f : ℝ → ℝ) := 
  f (s^2 - 2*s) ≥ -f (2*t - t^2)

def in_interval (s : ℝ) := 1 ≤ s ∧ s ≤ 4

theorem range_of_3t_plus_s (f : ℝ → ℝ) :
  is_increasing f ∧ symmetric_about f 3 0 →
  (∀ s t, satisfies_inequality s t f → in_interval s → -2 ≤ 3 * t + s ∧ 3 * t + s ≤ 16) :=
sorry

end range_of_3t_plus_s_l179_179343


namespace product_of_x_y_l179_179645

-- Assume the given conditions
variables (EF GH FG HE : ℝ)
variables (x y : ℝ)
variable (EFGH : Type)

-- Conditions given
axiom h1 : EF = 58
axiom h2 : GH = 3 * x + 1
axiom h3 : FG = 2 * y^2
axiom h4 : HE = 36
-- It is given that EFGH forms a parallelogram
axiom h5 : EF = GH
axiom h6 : FG = HE

-- The product of x and y is determined by the conditions
theorem product_of_x_y : x * y = 57 * Real.sqrt 2 :=
by
  sorry

end product_of_x_y_l179_179645


namespace dima_story_retelling_count_l179_179071

theorem dima_story_retelling_count :
  ∃ n, (26 * (2 ^ 5) * (3 ^ 4)) = 33696 ∧ n = 9 :=
by
  sorry

end dima_story_retelling_count_l179_179071


namespace arithmetic_sequence_a₄_l179_179903

open Int

noncomputable def S (a₁ d n : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_a₄ {a₁ d : ℤ}
  (h₁ : S a₁ d 5 = 15) (h₂ : S a₁ d 9 = 63) :
  a₁ + 3 * d = 5 :=
  sorry

end arithmetic_sequence_a₄_l179_179903


namespace new_bucket_capacity_l179_179158

theorem new_bucket_capacity (init_buckets : ℕ) (init_capacity : ℕ) (new_buckets : ℕ) (total_volume : ℕ) :
  init_buckets * init_capacity = total_volume →
  new_buckets * 9 = total_volume →
  9 = total_volume / new_buckets :=
by
  intros h₁ h₂
  sorry

end new_bucket_capacity_l179_179158


namespace unique_real_root_eq_l179_179029

theorem unique_real_root_eq (x : ℝ) : (∃! x, x = Real.sin x + 1993) :=
sorry

end unique_real_root_eq_l179_179029


namespace correct_operation_l179_179579

theorem correct_operation (a b : ℝ) : (a * b) - 2 * (a * b) = - (a * b) :=
sorry

end correct_operation_l179_179579


namespace truck_capacity_cost_function_minimum_cost_l179_179542

theorem truck_capacity :
  ∃ (m n : ℕ),
    3 * m + 4 * n = 27 ∧ 
    4 * m + 5 * n = 35 ∧
    m = 5 ∧ 
    n = 3 :=
by {
  sorry
}

theorem cost_function (a : ℕ) (h : a ≤ 5) :
  ∃ (w : ℕ),
    w = 50 * a + 2250 :=
by {
  sorry
}

theorem minimum_cost :
  ∃ (w : ℕ),
    w = 2250 ∧ 
    ∀ (a : ℕ), a ≤ 5 → (50 * a + 2250) ≥ 2250 :=
by {
  sorry
}

end truck_capacity_cost_function_minimum_cost_l179_179542


namespace monotone_increasing_function_range_l179_179582

theorem monotone_increasing_function_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2 : ℝ) (3 : ℝ), (1 / x + 2 * a * x - 3) ≥ 0) ↔ a ≥ 9 / 8 := 
by 
  sorry

end monotone_increasing_function_range_l179_179582


namespace a₁₀_greater_than_500_l179_179940

variables (a : ℕ → ℕ) (b : ℕ → ℕ)

-- Conditions
def strictly_increasing (a : ℕ → ℕ) : Prop := ∀ n, a n < a (n + 1)

def largest_divisor (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ n, b n < a n ∧ ∃ d > 1, d ∣ a n ∧ b n = a n / d

def greater_sequence (b : ℕ → ℕ) : Prop := ∀ n, b n > b (n + 1)

-- Statement to prove
theorem a₁₀_greater_than_500
  (h1 : strictly_increasing a)
  (h2 : largest_divisor a b)
  (h3 : greater_sequence b) :
  a 10 > 500 :=
sorry

end a₁₀_greater_than_500_l179_179940


namespace overlap_coordinates_l179_179531

theorem overlap_coordinates :
  ∃ m n : ℝ, 
    (m + n = 6.8) ∧ 
    ((2 * (7 + m) / 2 - 3) = (3 + n) / 2) ∧ 
    ((2 * (7 + m) / 2 - 3) = - (m - 7) / 2) :=
by
  sorry

end overlap_coordinates_l179_179531


namespace exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l179_179238

open Real EuclideanGeometry

def is_isosceles_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def is_isosceles_triangle_3D (A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ((dist A B = dist B C) ∨ (dist A B = dist A C) ∨ (dist B C = dist A C))

def five_points_isosceles (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 5, is_isosceles_triangle (pts i) (pts j) (pts k)

def six_points_isosceles (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∀ i j k : Fin 6, is_isosceles_triangle (pts i) (pts j) (pts k)

def seven_points_isosceles_3D (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∀ i j k : Fin 7, is_isosceles_triangle_3D (pts i) (pts j) (pts k)

theorem exists_five_points_isosceles : ∃ (pts : Fin 5 → EuclideanSpace ℝ (Fin 2)), five_points_isosceles pts :=
sorry

theorem exists_six_points_isosceles : ∃ (pts : Fin 6 → EuclideanSpace ℝ (Fin 2)), six_points_isosceles pts :=
sorry

theorem exists_seven_points_isosceles_3D : ∃ (pts : Fin 7 → EuclideanSpace ℝ (Fin 3)), seven_points_isosceles_3D pts :=
sorry

end exists_five_points_isosceles_exists_six_points_isosceles_exists_seven_points_isosceles_3D_l179_179238


namespace distance_origin_to_point_l179_179345

theorem distance_origin_to_point : 
  let origin := (0, 0)
  let point := (8, 15)
  dist origin point = 17 :=
by
  let dist (p1 p2 : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  sorry

end distance_origin_to_point_l179_179345


namespace contradiction_prop_l179_179049

theorem contradiction_prop (p : Prop) : 
  (∃ x : ℝ, x < -1 ∧ x^2 - x + 1 < 0) → (∀ x : ℝ, x < -1 → x^2 - x + 1 ≥ 0) :=
sorry

end contradiction_prop_l179_179049


namespace proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l179_179185

noncomputable def problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : Prop :=
  y ≤ 4.5

noncomputable def problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : Prop :=
  y ≥ -8

noncomputable def problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : Prop :=
  -1 ≤ y ∧ y ≤ 1

noncomputable def problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : Prop :=
  y ≤ 1/3

-- Proving that the properties hold:
theorem proof_of_problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : problem1 x y h :=
  sorry

theorem proof_of_problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : problem2 x y h :=
  sorry

theorem proof_of_problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : problem3 x y h :=
  sorry

theorem proof_of_problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : problem4 x y h :=
  sorry

end proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l179_179185


namespace first_sales_amount_l179_179190

-- Conditions from the problem
def first_sales_royalty : ℝ := 8 -- million dollars
def second_sales_royalty : ℝ := 9 -- million dollars
def second_sales_amount : ℝ := 108 -- million dollars
def decrease_percentage : ℝ := 0.7916666666666667

-- The goal is to determine the first sales amount, S, meeting the conditions.
theorem first_sales_amount :
  ∃ S : ℝ,
    (first_sales_royalty / S - second_sales_royalty / second_sales_amount = decrease_percentage * (first_sales_royalty / S)) ∧
    S = 20 :=
sorry

end first_sales_amount_l179_179190


namespace binary_addition_is_correct_l179_179162

-- Definitions for the binary numbers
def bin1 := "10101"
def bin2 := "11"
def bin3 := "1010"
def bin4 := "11100"
def bin5 := "1101"

-- Function to convert binary string to nat (using built-in functionality)
def binStringToNat (s : String) : Nat :=
  String.foldl (fun n c => 2 * n + if c = '1' then 1 else 0) 0 s

-- Binary numbers converted to nat
def n1 := binStringToNat bin1
def n2 := binStringToNat bin2
def n3 := binStringToNat bin3
def n4 := binStringToNat bin4
def n5 := binStringToNat bin5

-- The expected result in nat
def expectedSum := binStringToNat "11101101"

-- Proof statement
theorem binary_addition_is_correct : n1 + n2 + n3 + n4 + n5 = expectedSum :=
  sorry

end binary_addition_is_correct_l179_179162


namespace not_washed_shirts_l179_179720

-- Definitions based on given conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def washed_shirts : ℕ := 29

-- Theorem to prove the number of shirts not washed
theorem not_washed_shirts : (short_sleeve_shirts + long_sleeve_shirts) - washed_shirts = 1 := by
  sorry

end not_washed_shirts_l179_179720


namespace m₁_m₂_relationship_l179_179464

-- Defining the conditions
variables {Point Line : Type}
variables (intersect : Line → Line → Prop)
variables (coplanar : Line → Line → Prop)

-- Assumption that lines l₁ and l₂ are non-coplanar.
variables {l₁ l₂ : Line} (h_non_coplanar : ¬ coplanar l₁ l₂)

-- Assuming m₁ and m₂ both intersect with l₁ and l₂.
variables {m₁ m₂ : Line}
variables (h_intersect_m₁_l₁ : intersect m₁ l₁)
variables (h_intersect_m₁_l₂ : intersect m₁ l₂)
variables (h_intersect_m₂_l₁ : intersect m₂ l₁)
variables (h_intersect_m₂_l₂ : intersect m₂ l₂)

-- Statement to prove that m₁ and m₂ are either intersecting or non-coplanar.
theorem m₁_m₂_relationship :
  (¬ coplanar m₁ m₂) ∨ (∃ p : Point, (intersect m₁ m₂ ∧ intersect m₂ m₁)) :=
sorry

end m₁_m₂_relationship_l179_179464


namespace weighted_average_is_correct_l179_179824

def bag1_pop_kernels := 60
def bag1_total_kernels := 75
def bag2_pop_kernels := 42
def bag2_total_kernels := 50
def bag3_pop_kernels := 25
def bag3_total_kernels := 100
def bag4_pop_kernels := 77
def bag4_total_kernels := 120
def bag5_pop_kernels := 106
def bag5_total_kernels := 150

noncomputable def weighted_average_percentage : ℚ :=
  ((bag1_pop_kernels / bag1_total_kernels * 100 * bag1_total_kernels) +
   (bag2_pop_kernels / bag2_total_kernels * 100 * bag2_total_kernels) +
   (bag3_pop_kernels / bag3_total_kernels * 100 * bag3_total_kernels) +
   (bag4_pop_kernels / bag4_total_kernels * 100 * bag4_total_kernels) +
   (bag5_pop_kernels / bag5_total_kernels * 100 * bag5_total_kernels)) /
  (bag1_total_kernels + bag2_total_kernels + bag3_total_kernels + bag4_total_kernels + bag5_total_kernels)

theorem weighted_average_is_correct : weighted_average_percentage = 60.61 := 
by
  sorry

end weighted_average_is_correct_l179_179824


namespace inequality_holds_l179_179482

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_holds (h_cont : Continuous f) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, 2 * f x - (deriv f x) > 0) : 
  f 1 > (f 2) / (Real.exp 2) :=
sorry

end inequality_holds_l179_179482


namespace quadratic_no_real_roots_l179_179178

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Conditions of the problem
def a : ℝ := 3
def b : ℝ := -6
def c : ℝ := 4

-- The proof statement
theorem quadratic_no_real_roots : discriminant a b c < 0 :=
by
  -- Calculate the discriminant to show it's negative
  let Δ := discriminant a b c
  show Δ < 0
  sorry

end quadratic_no_real_roots_l179_179178


namespace ratio_of_abc_l179_179125

theorem ratio_of_abc (a b c : ℝ) (h1 : a ≠ 0) (h2 : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : a / b = 1 / 2 ∧ a / c = 1 / 3 := 
sorry

end ratio_of_abc_l179_179125


namespace regular_polygon_sides_l179_179020

theorem regular_polygon_sides (n : ℕ) (h : 108 = 180 * (n - 2) / n) : n = 5 := 
sorry

end regular_polygon_sides_l179_179020


namespace subset_B_of_A_l179_179477

def A : Set ℕ := {2, 0, 3}
def B : Set ℕ := {2, 3}

theorem subset_B_of_A : B ⊆ A :=
by
  sorry

end subset_B_of_A_l179_179477


namespace sequence_increasing_l179_179339

noncomputable def a (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

theorem sequence_increasing (n : ℕ) (hn : 0 < n) : a n < a (n + 1) :=
by
  -- Proof to be provided
  sorry

end sequence_increasing_l179_179339


namespace area_square_l179_179160

-- Define the conditions
variables (l r s : ℝ)
variable (breadth : ℝ := 10)
variable (area_rect : ℝ := 180)

-- Given conditions
def length_is_two_fifths_radius : Prop := l = (2/5) * r
def radius_is_side_square : Prop := r = s
def area_of_rectangle : Prop := area_rect = l * breadth

-- The theorem statement
theorem area_square (h1 : length_is_two_fifths_radius l r)
                    (h2 : radius_is_side_square r s)
                    (h3 : area_of_rectangle l breadth area_rect) :
  s^2 = 2025 :=
by
  sorry

end area_square_l179_179160


namespace solve_equation_l179_179666

theorem solve_equation (x : ℝ) : 
  (x + 1) / 6 = 4 / 3 - x ↔ x = 1 :=
sorry

end solve_equation_l179_179666


namespace tens_digit_of_9_pow_1801_l179_179091

theorem tens_digit_of_9_pow_1801 : 
  ∀ n : ℕ, (9 ^ (1801) % 100) / 10 % 10 = 0 :=
by
  sorry

end tens_digit_of_9_pow_1801_l179_179091


namespace fraction_condition_l179_179622

theorem fraction_condition (x : ℚ) :
  (3 + 2 * x) / (4 + 3 * x) = 5 / 9 ↔ x = -7 / 3 :=
by
  sorry

end fraction_condition_l179_179622


namespace choir_average_age_l179_179077

-- Each condition as a definition in Lean 4
def avg_age_females := 28
def num_females := 12
def avg_age_males := 32
def num_males := 18
def total_people := num_females + num_males

-- The total sum of ages calculated from the given conditions
def sum_ages_females := avg_age_females * num_females
def sum_ages_males := avg_age_males * num_males
def total_sum_ages := sum_ages_females + sum_ages_males

-- The final proof statement to be proved
theorem choir_average_age : 
  (total_sum_ages : ℝ) / (total_people : ℝ) = 30.4 := by
  sorry

end choir_average_age_l179_179077


namespace segment_length_is_ten_l179_179170

-- Definition of the cube root function and the absolute value
def cube_root (x : ℝ) : ℝ := x^(1/3)

def absolute (x : ℝ) : ℝ := abs x

-- The prerequisites as conditions for the endpoints
def endpoints_satisfy (x : ℝ) : Prop := absolute (x - cube_root 27) = 5

-- Length of the segment determined by the endpoints
def segment_length (x1 x2 : ℝ) : ℝ := absolute (x2 - x1)

-- Theorem statement
theorem segment_length_is_ten : (∀ x, endpoints_satisfy x) → segment_length (-2) 8 = 10 :=
by
  intro h
  sorry

end segment_length_is_ten_l179_179170


namespace solve_arrangement_equation_l179_179253

def arrangement_numeral (x : ℕ) : ℕ :=
  x * (x - 1) * (x - 2)

theorem solve_arrangement_equation (x : ℕ) (h : 3 * (arrangement_numeral x)^3 = 2 * (arrangement_numeral (x + 1))^2 + 6 * (arrangement_numeral x)^2) : x = 5 := 
sorry

end solve_arrangement_equation_l179_179253


namespace remainder_when_divided_by_22_l179_179649

theorem remainder_when_divided_by_22 
    (y : ℤ) 
    (h : y % 264 = 42) :
    y % 22 = 20 :=
by
  sorry

end remainder_when_divided_by_22_l179_179649


namespace product_of_x1_to_x13_is_zero_l179_179423

theorem product_of_x1_to_x13_is_zero
  (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ)
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 :=
sorry

end product_of_x1_to_x13_is_zero_l179_179423


namespace num_factors_of_2_pow_20_minus_1_l179_179988

/-- 
Prove that the number of positive two-digit integers 
that are factors of \(2^{20} - 1\) is 5.
-/
theorem num_factors_of_2_pow_20_minus_1 :
  ∃ (n : ℕ), n = 5 ∧ (∀ (k : ℕ), k ∣ (2^20 - 1) → 10 ≤ k ∧ k < 100 → k = 33 ∨ k = 15 ∨ k = 27 ∨ k = 41 ∨ k = 45) 
  :=
sorry

end num_factors_of_2_pow_20_minus_1_l179_179988


namespace octagon_area_l179_179638

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l179_179638


namespace total_marbles_l179_179585

-- Define the number of marbles Mary and Joan have respectively
def mary_marbles := 9
def joan_marbles := 3

-- Prove that the total number of marbles is 12
theorem total_marbles : mary_marbles + joan_marbles = 12 := by
  sorry

end total_marbles_l179_179585


namespace price_reduction_correct_l179_179925

theorem price_reduction_correct :
  ∃ x : ℝ, (0.3 - x) * (500 + 4000 * x) = 180 ∧ x = 0.1 :=
by
  sorry

end price_reduction_correct_l179_179925


namespace find_m_interval_l179_179021

def seq (x : ℕ → ℚ) : Prop :=
  (x 0 = 7) ∧ (∀ n : ℕ, x (n + 1) = (x n ^ 2 + 8 * x n + 9) / (x n + 7))

def m_spec (x : ℕ → ℚ) (m : ℕ) : Prop :=
  (x m ≤ 5 + 1 / 2^15)

theorem find_m_interval :
  ∃ (x : ℕ → ℚ) (m : ℕ), seq x ∧ m_spec x m ∧ 81 ≤ m ∧ m ≤ 242 :=
sorry

end find_m_interval_l179_179021


namespace remainder_eq_six_l179_179983

theorem remainder_eq_six
  (Dividend : ℕ) (Divisor : ℕ) (Quotient : ℕ) (Remainder : ℕ)
  (h1 : Dividend = 139)
  (h2 : Divisor = 19)
  (h3 : Quotient = 7)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Remainder = 6 :=
by
  sorry

end remainder_eq_six_l179_179983


namespace distance_light_travels_500_years_l179_179236

def distance_light_travels_one_year : ℝ := 5.87e12
def years : ℕ := 500

theorem distance_light_travels_500_years :
  distance_light_travels_one_year * years = 2.935e15 := 
sorry

end distance_light_travels_500_years_l179_179236


namespace part_a_part_b_part_c_l179_179277

-- Part a
def can_ratings_increase_after_first_migration (QA_before : ℚ) (QB_before : ℚ) (QA_after : ℚ) (QB_after : ℚ) : Prop :=
  QA_before < QA_after ∧ QB_before < QB_after

-- Part b
def can_ratings_increase_after_second_migration (QA_after_first : ℚ) (QB_after_first : ℚ) (QA_after_second : ℚ) (QB_after_second : ℚ) : Prop :=
  QA_after_second ≤ QA_after_first ∨ QB_after_second ≤ QB_after_first

-- Part c
def can_all_ratings_increase_after_reversed_migration (QA_before : ℚ) (QB_before : ℚ) (QC_before : ℚ) (QA_after_first : ℚ) (QB_after_first : ℚ) (QC_after_first : ℚ)
  (QA_after_second : ℚ) (QB_after_second : ℚ) (QC_after_second : ℚ) : Prop :=
  QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧
  QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second ∧ QC_after_first <= QC_after_second


-- Lean statements
theorem part_a (QA_before QA_after QB_before QB_after : ℚ) (Q_moved : ℚ) 
  (h : QA_before < QA_after ∧ QA_after < Q_moved ∧ QB_before < QB_after ∧ QB_after < Q_moved) : 
  can_ratings_increase_after_first_migration QA_before QB_before QA_after QB_after := 
by sorry

theorem part_b (QA_after_first QB_after_first QA_after_second QB_after_second : ℚ):
  ¬ can_ratings_increase_after_second_migration QA_after_first QB_after_first QA_after_second QB_after_second := 
by sorry

theorem part_c (QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first
  QA_after_second QB_after_second QC_after_second: ℚ)
  (h: QA_before < QA_after_first ∧ QB_before < QB_after_first ∧ QC_before < QC_after_first ∧ 
      QA_after_first < QA_after_second ∧ QB_after_first < QB_after_second) :
   can_all_ratings_increase_after_reversed_migration QA_before QB_before QC_before QA_after_first QB_after_first QC_after_first QA_after_second QB_after_second QC_after_second :=
by sorry

end part_a_part_b_part_c_l179_179277


namespace broken_seashells_l179_179244

-- Define the total number of seashells Tom found
def total_seashells : ℕ := 7

-- Define the number of unbroken seashells
def unbroken_seashells : ℕ := 3

-- Prove that the number of broken seashells equals 4
theorem broken_seashells : total_seashells - unbroken_seashells = 4 := by
  sorry

end broken_seashells_l179_179244


namespace sum_of_coordinates_of_A_l179_179750

noncomputable def point := (ℝ × ℝ)
def B : point := (2, 6)
def C : point := (4, 12)
def AC (A C : point) : ℝ := (A.1 - C.1)^2 + (A.2 - C.2)^2
def AB (A B : point) : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2
def BC (B C : point) : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2

theorem sum_of_coordinates_of_A :
  ∃ A : point, AC A C / AB A B = (1/3) ∧ BC B C / AB A B = (1/3) ∧ A.1 + A.2 = 24 :=
by
  sorry

end sum_of_coordinates_of_A_l179_179750


namespace calculate_a3_b3_l179_179614

theorem calculate_a3_b3 (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by
  sorry

end calculate_a3_b3_l179_179614


namespace find_d_minus_r_l179_179416

theorem find_d_minus_r :
  ∃ d r : ℕ, 1 < d ∧ 1223 % d = r ∧ 1625 % d = r ∧ 2513 % d = r ∧ d - r = 1 :=
by
  sorry

end find_d_minus_r_l179_179416


namespace distinct_real_roots_l179_179112

-- Define the polynomial equation as a Lean function
def polynomial (a x : ℝ) : ℝ :=
  (a + 1) * (x ^ 2 + 1) ^ 2 - (2 * a + 3) * (x ^ 2 + 1) * x + (a + 2) * x ^ 2

-- The theorem we need to prove
theorem distinct_real_roots (a : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ polynomial a x = 0 ∧ polynomial a y = 0) ↔ a ≠ -1 :=
by
  sorry

end distinct_real_roots_l179_179112


namespace intersection_complement_l179_179743

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := { y | y ≥ 0 }
noncomputable def B : Set ℝ := { y | y ≥ 1 }

theorem intersection_complement :
  A ∩ (U \ B) = Ico 0 1 :=
by
  sorry

end intersection_complement_l179_179743


namespace number_of_students_in_class_l179_179880

theorem number_of_students_in_class :
  ∃ n : ℕ, n > 0 ∧ (∀ avg_age teacher_age total_avg_age, avg_age = 26 ∧ teacher_age = 52 ∧ total_avg_age = 27 →
    (∃ total_student_age total_age_with_teacher, 
      total_student_age = n * avg_age ∧ 
      total_age_with_teacher = total_student_age + teacher_age ∧ 
      (total_age_with_teacher / (n + 1) = total_avg_age) → n = 25)) :=
sorry

end number_of_students_in_class_l179_179880


namespace cabbage_price_is_4_02_l179_179601

noncomputable def price_of_cabbage (broccoli_price_per_pound: ℝ) (broccoli_pounds: ℝ) 
                                    (orange_price_each: ℝ) (oranges: ℝ) 
                                    (bacon_price_per_pound: ℝ) (bacon_pounds: ℝ) 
                                    (chicken_price_per_pound: ℝ) (chicken_pounds: ℝ) 
                                    (budget_percentage_for_meat: ℝ) 
                                    (meat_price: ℝ) : ℝ := 
  let broccoli_total := broccoli_pounds * broccoli_price_per_pound
  let oranges_total := oranges * orange_price_each
  let bacon_total := bacon_pounds * bacon_price_per_pound
  let chicken_total := chicken_pounds * chicken_price_per_pound
  let subtotal := broccoli_total + oranges_total + bacon_total + chicken_total
  let total_budget := meat_price / budget_percentage_for_meat
  total_budget - subtotal

theorem cabbage_price_is_4_02 : 
  price_of_cabbage 4 3 0.75 3 3 1 3 2 0.33 9 = 4.02 := 
by 
  sorry

end cabbage_price_is_4_02_l179_179601


namespace largest_d_for_range_l179_179490

theorem largest_d_for_range (d : ℝ) : (∃ x : ℝ, x^2 - 6*x + d = 2) ↔ d ≤ 11 := 
by
  sorry

end largest_d_for_range_l179_179490


namespace percentage_of_y_l179_179141

theorem percentage_of_y (x y P : ℝ) (h1 : 0.10 * x = (P/100) * y) (h2 : x / y = 2) : P = 20 :=
sorry

end percentage_of_y_l179_179141


namespace M_diff_N_eq_l179_179254

noncomputable def A_diff_B (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

noncomputable def M : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }

noncomputable def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem M_diff_N_eq : A_diff_B M N = { x | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end M_diff_N_eq_l179_179254


namespace luke_good_games_l179_179595

-- Definitions
def bought_from_friend : ℕ := 2
def bought_from_garage_sale : ℕ := 2
def defective_games : ℕ := 2

-- The theorem we want to prove
theorem luke_good_games :
  bought_from_friend + bought_from_garage_sale - defective_games = 2 := 
by 
  sorry

end luke_good_games_l179_179595


namespace combinatorial_identity_l179_179945

theorem combinatorial_identity :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 :=
sorry

end combinatorial_identity_l179_179945


namespace fish_caught_by_twentieth_fisherman_l179_179126

theorem fish_caught_by_twentieth_fisherman :
  ∀ (total_fishermen total_fish fish_per_fisherman nineten_fishermen : ℕ),
  total_fishermen = 20 →
  total_fish = 10000 →
  fish_per_fisherman = 400 →
  nineten_fishermen = 19 →
  (total_fishermen * fish_per_fisherman) - (nineten_fishermen * fish_per_fisherman) = 2400 :=
by
  intros
  sorry

end fish_caught_by_twentieth_fisherman_l179_179126


namespace max_sum_of_arithmetic_sequence_l179_179678

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (S_seq : ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)) 
  (S16_pos : S 16 > 0) (S17_neg : S 17 < 0) : 
  ∃ m, ∀ n, S n ≤ S m ∧ m = 8 := 
sorry

end max_sum_of_arithmetic_sequence_l179_179678


namespace inequality_proof_l179_179087

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  abc ≥ (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ∧
  (a + b + c) / ((1 / a^2) + (1 / b^2) + (1 / c^2)) ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end inequality_proof_l179_179087


namespace convert_decimal_to_fraction_l179_179064

theorem convert_decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by
  sorry

end convert_decimal_to_fraction_l179_179064


namespace run_faster_l179_179538

theorem run_faster (v_B k : ℝ) (h1 : ∀ (t : ℝ), 96 / (k * v_B) = t → 24 / v_B = t) : k = 4 :=
by {
  sorry
}

end run_faster_l179_179538


namespace bear_pies_l179_179537

-- Lean definitions model:

variables (v_M v_B u_M u_B : ℝ)
variables (M_raspberries B_raspberries : ℝ)
variables (P_M P_B : ℝ)

-- Given conditions
axiom v_B_eq_6v_M : v_B = 6 * v_M
axiom u_B_eq_3u_M : u_B = 3 * u_M
axiom B_raspberries_eq_2M_raspberries : B_raspberries = 2 * M_raspberries
axiom P_sum : P_B + P_M = 60
axiom P_B_eq_9P_M : P_B = 9 * P_M

-- The theorem to prove
theorem bear_pies : P_B = 54 :=
sorry

end bear_pies_l179_179537


namespace M_eq_N_l179_179741

def M : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k + 1) * Real.pi}
def N : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k - 1) * Real.pi}

theorem M_eq_N : M = N := by
  sorry

end M_eq_N_l179_179741


namespace overall_cost_for_all_projects_l179_179744

-- Define the daily salaries including 10% taxes and insurance.
def daily_salary_entry_level_worker : ℕ := 100 + 10
def daily_salary_experienced_worker : ℕ := 130 + 13
def daily_salary_electrician : ℕ := 2 * 100 + 20
def daily_salary_plumber : ℕ := 250 + 25
def daily_salary_architect : ℕ := (35/10) * 100 + 35

-- Define the total cost for each project.
def project1_cost : ℕ :=
  daily_salary_entry_level_worker +
  daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project2_cost : ℕ :=
  2 * daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project3_cost : ℕ :=
  2 * daily_salary_entry_level_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

-- Define the overall cost for all three projects.
def total_cost : ℕ :=
  project1_cost + project2_cost + project3_cost

theorem overall_cost_for_all_projects :
  total_cost = 3399 :=
by
  sorry

end overall_cost_for_all_projects_l179_179744


namespace vectors_are_coplanar_l179_179757

-- Definitions of the vectors a, b, and c.
def a (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -2)
def b : ℝ × ℝ × ℝ := (0, 1, 2)
def c : ℝ × ℝ × ℝ := (1, 0, 0)

-- The proof statement 
theorem vectors_are_coplanar (x : ℝ) 
  (h : ∃ m n : ℝ, a x = (n, m, 2 * m)) : 
  x = -1 :=
sorry

end vectors_are_coplanar_l179_179757


namespace typing_difference_l179_179509

theorem typing_difference (m : ℕ) (h1 : 10 * m - 8 * m = 10) : m = 5 :=
by
  sorry

end typing_difference_l179_179509


namespace geometric_sequence_product_l179_179562

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = a 1 * (a 2 / a 1) ^ n)
  (h1 : a 1 * a 4 = -3) : a 2 * a 3 = -3 :=
by
  -- sorry is placed here to indicate the proof is not provided.
  sorry

end geometric_sequence_product_l179_179562


namespace johns_burritos_l179_179493

-- Definitions based on conditions:
def initial_burritos : Nat := 3 * 20
def burritos_given_away : Nat := initial_burritos / 3
def burritos_after_giving_away : Nat := initial_burritos - burritos_given_away
def burritos_eaten : Nat := 3 * 10
def burritos_left : Nat := burritos_after_giving_away - burritos_eaten

-- The theorem we need to prove:
theorem johns_burritos : burritos_left = 10 := by
  sorry

end johns_burritos_l179_179493


namespace right_triangle_hypotenuse_l179_179085

theorem right_triangle_hypotenuse (a b : ℝ) (m_a m_b : ℝ)
    (h1 : m_a = Real.sqrt (b^2 + (a / 2)^2))
    (h2 : m_b = Real.sqrt (a^2 + (b / 2)^2))
    (h3 : m_a = Real.sqrt 30)
    (h4 : m_b = 6) :
  Real.sqrt (4 * (a^2 + b^2)) = 2 * Real.sqrt 52.8 :=
by
  sorry

end right_triangle_hypotenuse_l179_179085


namespace infinite_possible_matrices_A_squared_l179_179281

theorem infinite_possible_matrices_A_squared (A : Matrix (Fin 3) (Fin 3) ℝ) (hA : A^4 = 0) :
  ∃ (S : Set (Matrix (Fin 3) (Fin 3) ℝ)), (∀ B ∈ S, B = A^2) ∧ S.Infinite :=
sorry

end infinite_possible_matrices_A_squared_l179_179281


namespace geometric_series_sum_l179_179261

/-- The first term of the geometric series. -/
def a : ℚ := 3

/-- The common ratio of the geometric series. -/
def r : ℚ := -3 / 4

/-- The sum of the geometric series is equal to 12/7. -/
theorem geometric_series_sum : (∑' n : ℕ, a * r^n) = 12 / 7 := 
by
  /- The Sum function and its properties for the geometric series will be used here. -/
  sorry

end geometric_series_sum_l179_179261


namespace bananas_left_l179_179527

-- Definitions based on conditions
def original_bananas : ℕ := 46
def bananas_removed : ℕ := 5

-- Statement of the problem using the definitions
theorem bananas_left : original_bananas - bananas_removed = 41 :=
by sorry

end bananas_left_l179_179527


namespace salad_cucumbers_l179_179746

theorem salad_cucumbers (c t : ℕ) 
  (h1 : c + t = 280)
  (h2 : t = 3 * c) : c = 70 :=
sorry

end salad_cucumbers_l179_179746


namespace least_value_xy_l179_179613

theorem least_value_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/9) : x*y = 108 :=
sorry

end least_value_xy_l179_179613


namespace quadratic_solution_condition_sufficient_but_not_necessary_l179_179955

theorem quadratic_solution_condition_sufficient_but_not_necessary (m : ℝ) :
  (m < -2) → (∃ x : ℝ, x^2 + m * x + 1 = 0) ∧ ¬(∀ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0 → m < -2) :=
by 
  sorry

end quadratic_solution_condition_sufficient_but_not_necessary_l179_179955


namespace solve_for_x_l179_179438

theorem solve_for_x (x : ℝ) :
  (2 * x - 30) / 3 = (5 - 3 * x) / 4 + 1 → x = 147 / 17 := 
by
  intro h
  sorry

end solve_for_x_l179_179438


namespace S_not_eq_T_l179_179872

def S := {x : ℤ | ∃ n : ℤ, x = 2 * n}
def T := {x : ℤ | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}

theorem S_not_eq_T : S ≠ T := by
  sorry

end S_not_eq_T_l179_179872


namespace towers_per_castle_jeff_is_5_l179_179205

-- Define the number of sandcastles on Mark's beach
def num_castles_mark : ℕ := 20

-- Define the number of towers per sandcastle on Mark's beach
def towers_per_castle_mark : ℕ := 10

-- Calculate the total number of towers on Mark's beach
def total_towers_mark : ℕ := num_castles_mark * towers_per_castle_mark

-- Define the number of sandcastles on Jeff's beach (3 times that of Mark's)
def num_castles_jeff : ℕ := 3 * num_castles_mark

-- Define the total number of sandcastles on both beaches
def total_sandcastles : ℕ := num_castles_mark + num_castles_jeff
  
-- Define the combined total number of sandcastles and towers on both beaches
def combined_total : ℕ := 580

-- Define the number of towers per sandcastle on Jeff's beach
def towers_per_castle_jeff : ℕ := sorry

-- Define the total number of towers on Jeff's beach
def total_towers_jeff (T : ℕ) : ℕ := num_castles_jeff * T

-- Prove that the number of towers per sandcastle on Jeff's beach is 5
theorem towers_per_castle_jeff_is_5 : 
    200 + total_sandcastles + total_towers_jeff towers_per_castle_jeff = combined_total → 
    towers_per_castle_jeff = 5
:= by
    sorry

end towers_per_castle_jeff_is_5_l179_179205


namespace problem1_proof_l179_179093

-- Define the mathematical conditions and problems
def problem1_expression (x y : ℝ) : ℝ := y * (4 * x - 3 * y) + (x - 2 * y) ^ 2

-- State the theorem with the simplified form as the conclusion
theorem problem1_proof (x y : ℝ) : problem1_expression x y = x^2 + y^2 :=
by
  sorry

end problem1_proof_l179_179093


namespace largest_number_A_l179_179791

theorem largest_number_A (A B C : ℕ) (h1: A = 7 * B + C) (h2: B = C) 
  : A ≤ 48 :=
sorry

end largest_number_A_l179_179791


namespace max_vec_diff_magnitude_l179_179966

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ := (1, sin θ)
noncomputable def vec_b (θ : ℝ) : ℝ × ℝ := (1, cos θ)

noncomputable def vec_diff_magnitude (θ : ℝ) : ℝ :=
  let a := vec_a θ
  let b := vec_b θ
  abs ((a.1 - b.1)^2 + (a.2 - b.2)^2)^(1/2)

theorem max_vec_diff_magnitude : ∀ θ : ℝ, vec_diff_magnitude θ ≤ sqrt 2 :=
by
  intro θ
  sorry

end max_vec_diff_magnitude_l179_179966


namespace find_value_of_p_l179_179159

theorem find_value_of_p (p : ℝ) :
  (∀ x y, (x = 0 ∧ y = -2) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 1/2 ∧ y = 0) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 2 ∧ y = 0) → y = p*x^2 + 5*x + p) →
  p = -2 :=
by
  sorry

end find_value_of_p_l179_179159


namespace range_of_a_l179_179776

variable (x a : ℝ)
def inequality_sys := x < a ∧ x < 3
def solution_set := x < a

theorem range_of_a (h : ∀ x, inequality_sys x a → solution_set x a) : a ≤ 3 := by
  sorry

end range_of_a_l179_179776


namespace three_digit_cubes_divisible_by_16_l179_179745

theorem three_digit_cubes_divisible_by_16 (n : ℤ) (x : ℤ) 
  (h_cube : x = n^3)
  (h_div : 16 ∣ x) 
  (h_3digit : 100 ≤ x ∧ x ≤ 999) : 
  x = 512 := 
by {
  sorry
}

end three_digit_cubes_divisible_by_16_l179_179745


namespace rice_and_grain_separation_l179_179303

theorem rice_and_grain_separation (total_weight : ℕ) (sample_size : ℕ) (non_rice_sample : ℕ) (non_rice_in_batch : ℕ) :
  total_weight = 1524 →
  sample_size = 254 →
  non_rice_sample = 28 →
  non_rice_in_batch = total_weight * non_rice_sample / sample_size →
  non_rice_in_batch = 168 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end rice_and_grain_separation_l179_179303


namespace expression_equals_500_l179_179301

theorem expression_equals_500 :
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  C = 500 :=
by
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  sorry

end expression_equals_500_l179_179301


namespace wall_height_correct_l179_179338

noncomputable def brick_volume : ℝ := 25 * 11.25 * 6

noncomputable def wall_total_volume (num_bricks : ℕ) (brick_vol : ℝ) : ℝ := num_bricks * brick_vol

noncomputable def wall_height (total_volume : ℝ) (length : ℝ) (thickness : ℝ) : ℝ :=
  total_volume / (length * thickness)

theorem wall_height_correct :
  wall_height (wall_total_volume 7200 brick_volume) 900 22.5 = 600 := by
  sorry

end wall_height_correct_l179_179338


namespace monkeys_and_apples_l179_179136

theorem monkeys_and_apples
  {x a : ℕ}
  (h1 : a = 3 * x + 6)
  (h2 : 0 < a - 4 * (x - 1) ∧ a - 4 * (x - 1) < 4)
  : (x = 7 ∧ a = 27) ∨ (x = 8 ∧ a = 30) ∨ (x = 9 ∧ a = 33) :=
sorry

end monkeys_and_apples_l179_179136


namespace relationship_between_lines_l179_179359

-- Define the type for a line and a plane
structure Line where
  -- some properties (to be defined as needed, omitted for brevity)

structure Plane where
  -- some properties (to be defined as needed, omitted for brevity)

-- Define parallelism between a line and a plane
def parallel_line_plane (m : Line) (α : Plane) : Prop := sorry

-- Define line within a plane
def line_within_plane (n : Line) (α : Plane) : Prop := sorry

-- Define parallelism between two lines
def parallel_lines (m n : Line) : Prop := sorry

-- Define skewness between two lines
def skew_lines (m n : Line) : Prop := sorry

-- The mathematically equivalent proof problem
theorem relationship_between_lines (m n : Line) (α : Plane)
  (h1 : parallel_line_plane m α)
  (h2 : line_within_plane n α) :
  parallel_lines m n ∨ skew_lines m n := 
sorry

end relationship_between_lines_l179_179359


namespace loss_percentage_is_five_l179_179498

/-- Definitions -/
def original_price : ℝ := 490
def sold_price : ℝ := 465.50
def loss_amount : ℝ := original_price - sold_price

/-- Theorem -/
theorem loss_percentage_is_five :
  (loss_amount / original_price) * 100 = 5 :=
by
  sorry

end loss_percentage_is_five_l179_179498


namespace squirrels_in_tree_l179_179199

theorem squirrels_in_tree (N S : ℕ) (h₁ : N = 2) (h₂ : S - N = 2) : S = 4 :=
by
  sorry

end squirrels_in_tree_l179_179199


namespace certain_number_of_tenths_l179_179949

theorem certain_number_of_tenths (n : ℝ) (h : n = 375 * (1/10)) : n = 37.5 :=
by
  sorry

end certain_number_of_tenths_l179_179949


namespace fraction_sum_eq_one_l179_179650

theorem fraction_sum_eq_one (m n : ℝ) (h : m ≠ n) : (m / (m - n) + n / (n - m) = 1) :=
by
  sorry

end fraction_sum_eq_one_l179_179650


namespace distance_between_locations_A_and_B_l179_179078

-- Define the conditions
variables {x y s t : ℝ}

-- Conditions specified in the problem
axiom bus_a_meets_bus_b_after_85_km : 85 / x = (s - 85) / y 
axiom buses_meet_again_after_turnaround : (s - 85 + 65) / x + 1 / 2 = (85 + (s - 65)) / y + 1 / 2

-- The theorem to be proved
theorem distance_between_locations_A_and_B : s = 190 :=
by
  sorry

end distance_between_locations_A_and_B_l179_179078


namespace find_m_l179_179563

theorem find_m (m : ℝ) : (m - 2) * (0 : ℝ)^2 + 4 * (0 : ℝ) + 2 - |m| = 0 → m = -2 :=
by
  intros h
  sorry

end find_m_l179_179563


namespace find_first_number_l179_179240

theorem find_first_number (x : ℝ) : (x + 16 + 8 + 22) / 4 = 13 ↔ x = 6 :=
by 
  sorry

end find_first_number_l179_179240


namespace ring_rotation_count_l179_179719

-- Define the constants and parameters from the conditions
variables (R ω μ g : ℝ) -- radius, angular velocity, coefficient of friction, and gravity constant
-- Additional constraints on these variables
variable (m : ℝ) -- mass of the ring

theorem ring_rotation_count :
  ∃ n : ℝ, n = (ω^2 * R * (1 + μ^2)) / (4 * π * g * μ * (1 + μ)) :=
sorry

end ring_rotation_count_l179_179719


namespace find_smaller_number_l179_179800

theorem find_smaller_number (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h_ratio : u / v = 3 / 5) (h_sum : u + v = 16) : u = 6 :=
by
  sorry

end find_smaller_number_l179_179800


namespace initial_population_l179_179557

theorem initial_population (P : ℝ) (h1 : 1.20 * P = P_1) (h2 : 0.96 * P = P_2) (h3 : P_2 = 9600) : P = 10000 :=
by
  sorry

end initial_population_l179_179557


namespace domain_proof_l179_179655

def domain_of_function : Set ℝ := {x : ℝ | x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x}

theorem domain_proof :
  (∀ x : ℝ, (x ≠ 7) → (x^2 - 16 ≥ 0) → (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x)) ∧
  (∀ x : ℝ, (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x) → (x ≠ 7) ∧ (x^2 - 16 ≥ 0)) :=
by
  sorry

end domain_proof_l179_179655


namespace rita_total_hours_l179_179659

def h_backstroke : ℕ := 50
def h_breaststroke : ℕ := 9
def h_butterfly : ℕ := 121
def h_freestyle_sidestroke_per_month : ℕ := 220
def months : ℕ := 6

def h_total : ℕ := h_backstroke + h_breaststroke + h_butterfly + (h_freestyle_sidestroke_per_month * months)

theorem rita_total_hours :
  h_total = 1500 :=
by
  sorry

end rita_total_hours_l179_179659


namespace triangle_lattice_points_l179_179127

-- Given lengths of the legs of the right triangle
def DE : Nat := 15
def EF : Nat := 20

-- Calculate the hypotenuse using the Pythagorean theorem
def DF : Nat := Nat.sqrt (DE ^ 2 + EF ^ 2)

-- Calculate the area of the triangle
def Area : Nat := (DE * EF) / 2

-- Calculate the number of boundary points
def B : Nat :=
  let points_DE := DE + 1
  let points_EF := EF + 1
  let points_DF := DF + 1
  points_DE + points_EF + points_DF - 3

-- Calculate the number of interior points using Pick's Theorem
def I : Int := Area - (B / 2 - 1)

-- Calculate the total number of lattice points
def total_lattice_points : Int := I + Int.ofNat B

-- The theorem statement
theorem triangle_lattice_points : total_lattice_points = 181 := by
  -- The actual proof goes here
  sorry

end triangle_lattice_points_l179_179127


namespace gcd_a2_13a_36_a_6_eq_6_l179_179155

namespace GCDProblem

variable (a : ℕ)
variable (h : ∃ k, a = 1632 * k)

theorem gcd_a2_13a_36_a_6_eq_6 (ha : ∃ k : ℕ, a = 1632 * k) : 
  Int.gcd (a^2 + 13 * a + 36 : Int) (a + 6 : Int) = 6 := by
  sorry

end GCDProblem

end gcd_a2_13a_36_a_6_eq_6_l179_179155


namespace return_trip_speed_l179_179778

theorem return_trip_speed (d xy_dist : ℝ) (s xy_speed : ℝ) (avg_speed : ℝ) (r return_speed : ℝ) :
  xy_dist = 150 →
  xy_speed = 75 →
  avg_speed = 50 →
  2 * xy_dist / ((xy_dist / xy_speed) + (xy_dist / return_speed)) = avg_speed →
  return_speed = 37.5 :=
by
  intros hxy_dist hxy_speed h_avg_speed h_avg_speed_eq
  sorry

end return_trip_speed_l179_179778


namespace difference_of_digits_is_six_l179_179930

theorem difference_of_digits_is_six (a b : ℕ) (h_sum : a + b = 10) (h_number : 10 * a + b = 82) : a - b = 6 :=
sorry

end difference_of_digits_is_six_l179_179930


namespace eval_expression_l179_179201

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l179_179201


namespace remainder_of_number_of_minimally_intersecting_triples_l179_179066

noncomputable def number_of_minimally_intersecting_triples : Nat :=
  let n := (8 * 7 * 6) * (4 ^ 5)
  n % 1000

theorem remainder_of_number_of_minimally_intersecting_triples :
  number_of_minimally_intersecting_triples = 64 := by
  sorry

end remainder_of_number_of_minimally_intersecting_triples_l179_179066


namespace value_of_f_neg_one_l179_179368

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_neg_one (f_def : ∀ x, f (Real.tan x) = Real.sin (2 * x)) : f (-1) = -1 := 
by
sorry

end value_of_f_neg_one_l179_179368


namespace bc_product_l179_179305

theorem bc_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 2 = 0 → r^4 - b * r - c = 0) → b * c = 30 :=
by
  sorry

end bc_product_l179_179305


namespace gain_percentage_l179_179694

-- Define the conditions as a Lean problem
theorem gain_percentage (C G : ℝ) (hC : (9 / 10) * C = 1) (hSP : (10 / 6) = (1 + G / 100) * C) : 
  G = 50 :=
by
-- Here, you would generally have the proof steps, but we add sorry to skip the proof for now.
sorry

end gain_percentage_l179_179694


namespace f_properties_l179_179153

theorem f_properties (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^2) - f (y^2) ≤ (f x + y) * (x - f y)) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end f_properties_l179_179153


namespace complement_of_A_l179_179019

def U : Set ℤ := {-1, 2, 4}
def A : Set ℤ := {-1, 4}

theorem complement_of_A : U \ A = {2} := by
  sorry

end complement_of_A_l179_179019


namespace minimum_value_function_l179_179494

theorem minimum_value_function (x : ℝ) (h : x > -1) : 
  (∃ y, y = (x^2 + 7 * x + 10) / (x + 1) ∧ y ≥ 9) :=
sorry

end minimum_value_function_l179_179494


namespace sue_library_inventory_l179_179328

theorem sue_library_inventory :
  let initial_books := 15
  let initial_movies := 6
  let returned_books := 8
  let returned_movies := initial_movies / 3
  let borrowed_more_books := 9
  let current_books := initial_books - returned_books + borrowed_more_books
  let current_movies := initial_movies - returned_movies
  current_books + current_movies = 20 :=
by
  -- no implementation provided
  sorry

end sue_library_inventory_l179_179328


namespace initial_markup_percentage_l179_179921

theorem initial_markup_percentage
  (cost_price : ℝ := 100)
  (profit_percentage : ℝ := 14)
  (discount_percentage : ℝ := 5)
  (selling_price : ℝ := cost_price * (1 + profit_percentage / 100))
  (x : ℝ := 20) :
  (cost_price + cost_price * x / 100) * (1 - discount_percentage / 100) = selling_price := by
  sorry

end initial_markup_percentage_l179_179921


namespace op_example_l179_179200

variables {α β : ℚ}

def op (α β : ℚ) := α * β + 1

theorem op_example : op 2 (-3) = -5 :=
by
  -- The proof is omitted as requested
  sorry

end op_example_l179_179200


namespace gas_price_increase_l179_179920

theorem gas_price_increase (P C : ℝ) (x : ℝ) 
  (h1 : P * C = P * (1 + x) * 1.10 * C * (1 - 0.27272727272727)) :
  x = 0.25 :=
by
  -- The proof will be filled here
  sorry

end gas_price_increase_l179_179920


namespace products_selling_less_than_1000_l179_179325

theorem products_selling_less_than_1000 (N: ℕ) 
  (total_products: ℕ := 25) 
  (average_price: ℤ := 1200) 
  (min_price: ℤ := 400) 
  (max_price: ℤ := 12000) 
  (total_revenue := total_products * average_price) 
  (revenue_from_expensive: ℤ := max_price):
  12000 + (24 - N) * 1000 + N * 400 = 30000 ↔ N = 10 :=
by
  sorry

end products_selling_less_than_1000_l179_179325


namespace paco_more_cookies_l179_179442

def paco_cookies_difference
  (initial_cookies : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_given : ℕ) : ℕ :=
  cookies_eaten - cookies_given

theorem paco_more_cookies 
  (initial_cookies : ℕ)
  (cookies_eaten : ℕ)
  (cookies_given : ℕ)
  (h1 : initial_cookies = 17)
  (h2 : cookies_eaten = 14)
  (h3 : cookies_given = 13) :
  paco_cookies_difference initial_cookies cookies_eaten cookies_given = 1 :=
by
  rw [h2, h3]
  exact rfl

end paco_more_cookies_l179_179442


namespace polygon_sides_l179_179060

theorem polygon_sides {S n : ℕ} (h : S = 2160) (hs : S = 180 * (n - 2)) : n = 14 := 
by
  sorry

end polygon_sides_l179_179060


namespace faster_speed_l179_179204

theorem faster_speed (v : ℝ) (h1 : ∀ (t : ℝ), t = 50 / 10) (h2 : ∀ (d : ℝ), d = 50 + 20) (h3 : ∀ (t : ℝ), t = 70 / v) : v = 14 :=
by
  sorry

end faster_speed_l179_179204


namespace range_of_m_l179_179038

theorem range_of_m {m : ℝ} (h1 : ∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1)))
                   (h2 : ∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))
                   (h3 : ¬(∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1))) ∧
                           (∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))) :
  m > 1 :=
by
  sorry

end range_of_m_l179_179038


namespace largest_divisor_of_even_n_cube_difference_l179_179123

theorem largest_divisor_of_even_n_cube_difference (n : ℤ) (h : Even n) : 6 ∣ (n^3 - n) := by
  sorry

end largest_divisor_of_even_n_cube_difference_l179_179123


namespace triangle_third_side_length_l179_179516

theorem triangle_third_side_length 
  (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 3) :
  (4 < c ∧ c < 18) → c ≠ 3 :=
by
  sorry

end triangle_third_side_length_l179_179516


namespace hyperbola_asymptote_b_l179_179437

theorem hyperbola_asymptote_b {b : ℝ} (hb : b > 0) :
  (∀ x y : ℝ, x^2 - (y^2 / b^2) = 1 → (y = 2 * x)) → b = 2 := by
  sorry

end hyperbola_asymptote_b_l179_179437


namespace number_of_valid_pairings_l179_179483

-- Definition for the problem
def validPairingCount (n : ℕ) (k: ℕ) : ℕ :=
  sorry -- Calculating the valid number of pairings is deferred

-- The problem statement to be proven:
theorem number_of_valid_pairings : validPairingCount 12 3 = 14 :=
sorry

end number_of_valid_pairings_l179_179483


namespace decagon_perimeter_l179_179547

theorem decagon_perimeter (num_sides : ℕ) (side_length : ℝ) (h_num_sides : num_sides = 10) (h_side_length : side_length = 3) : 
  (num_sides * side_length = 30) :=
by
  sorry

end decagon_perimeter_l179_179547


namespace multiplication_factor_average_l179_179827

theorem multiplication_factor_average (a : ℕ) (b : ℕ) (c : ℕ) (F : ℝ) 
  (h1 : a = 7) 
  (h2 : b = 26) 
  (h3 : (c : ℝ) = 130) 
  (h4 : (a * b * F : ℝ) = a * c) :
  F = 5 := 
by 
  sorry

end multiplication_factor_average_l179_179827


namespace sum_of_coordinates_of_D_l179_179564

theorem sum_of_coordinates_of_D (P C D : ℝ × ℝ)
  (hP : P = (4, 9))
  (hC : C = (10, 5))
  (h_mid : P = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 11 :=
sorry

end sum_of_coordinates_of_D_l179_179564


namespace number_of_cubes_with_three_faces_painted_l179_179465

-- Definitions of conditions
def large_cube_side_length : ℕ := 4
def total_smaller_cubes := large_cube_side_length ^ 3

-- Prove the number of smaller cubes with at least 3 faces painted is 8
theorem number_of_cubes_with_three_faces_painted :
  (∃ (n : ℕ), n = 8) :=
by
  -- Conditions recall
  have side_length := large_cube_side_length
  have total_cubes := total_smaller_cubes
  
  -- Recall that the cube is composed by smaller cubes with painted faces.
  have painted_faces_condition : (∀ (cube : ℕ), cube = 8) := sorry
  
  exact ⟨8, painted_faces_condition 8⟩

end number_of_cubes_with_three_faces_painted_l179_179465


namespace cos_seven_theta_l179_179962

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l179_179962


namespace find_sum_of_a_b_l179_179398

def star (a b : ℕ) : ℕ := a^b - a * b

theorem find_sum_of_a_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 2) : a + b = 5 := 
by
  sorry

end find_sum_of_a_b_l179_179398


namespace overall_gain_is_correct_l179_179274

noncomputable def overall_gain_percentage : ℝ :=
  let CP_A := 100
  let SP_A := 120 / (1 - 0.20)
  let gain_A := SP_A - CP_A

  let CP_B := 200
  let SP_B := 240 / (1 + 0.10)
  let gain_B := SP_B - CP_B

  let CP_C := 150
  let SP_C := (165 / (1 + 0.05)) / (1 - 0.10)
  let gain_C := SP_C - CP_C

  let CP_D := 300
  let SP_D := (345 / (1 - 0.05)) / (1 + 0.15)
  let gain_D := SP_D - CP_D

  let total_gain := gain_A + gain_B + gain_C + gain_D
  let total_CP := CP_A + CP_B + CP_C + CP_D
  (total_gain / total_CP) * 100

theorem overall_gain_is_correct : abs (overall_gain_percentage - 14.48) < 0.01 := by
  sorry

end overall_gain_is_correct_l179_179274


namespace minimum_value_of_a_plus_5b_l179_179991

theorem minimum_value_of_a_plus_5b :
  ∀ (a b : ℝ), a > 0 → b > 0 → (1 / a + 5 / b = 1) → a + 5 * b ≥ 36 :=
by
  sorry

end minimum_value_of_a_plus_5b_l179_179991


namespace candle_cost_correct_l179_179606

-- Variables and conditions
def candles_per_cake : Nat := 8
def num_cakes : Nat := 3
def candles_needed : Nat := candles_per_cake * num_cakes

def candles_per_box : Nat := 12
def boxes_needed : Nat := candles_needed / candles_per_box

def cost_per_box : ℝ := 2.5
def total_cost : ℝ := boxes_needed * cost_per_box

-- Proof statement
theorem candle_cost_correct :
  total_cost = 5 := by
  sorry

end candle_cost_correct_l179_179606


namespace sum_of_100th_group_is_1010100_l179_179592

theorem sum_of_100th_group_is_1010100 : (100 + 100^2 + 100^3) = 1010100 :=
by
  sorry

end sum_of_100th_group_is_1010100_l179_179592


namespace eval_expression_l179_179132

-- Definitions for the problem conditions
def reciprocal (a : ℕ) : ℚ := 1 / a

-- The theorem statement
theorem eval_expression : (reciprocal 9 - reciprocal 6)⁻¹ = -18 := by
  sorry

end eval_expression_l179_179132


namespace profit_percentage_is_23_16_l179_179150

   noncomputable def cost_price (mp : ℝ) : ℝ := 95 * mp
   noncomputable def selling_price (mp : ℝ) : ℝ := 120 * (mp - (0.025 * mp))
   noncomputable def profit_percent (cp sp : ℝ) : ℝ := ((sp - cp) / cp) * 100

   theorem profit_percentage_is_23_16 
     (mp : ℝ) (h_mp_gt_zero : mp > 0) : 
       profit_percent (cost_price mp) (selling_price mp) = 23.16 :=
   by 
     sorry
   
end profit_percentage_is_23_16_l179_179150


namespace average_score_all_test_takers_l179_179115

def avg (scores : List ℕ) : ℕ := scores.sum / scores.length

theorem average_score_all_test_takers (s_avg u_avg n : ℕ) 
  (H1 : s_avg = 42) (H2 : u_avg = 38) (H3 : n = 20) : avg ([s_avg * n, u_avg * n]) / (2 * n) = 40 := 
by sorry

end average_score_all_test_takers_l179_179115


namespace find_smallest_x_l179_179704

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem find_smallest_x (x: ℕ) (h1: 2 * x = 144) (h2: 3 * x = 216) : x = 72 :=
by
  sorry

end find_smallest_x_l179_179704


namespace smallest_integer_of_lcm_gcd_l179_179184

theorem smallest_integer_of_lcm_gcd (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 60 m / Nat.gcd 60 m = 44) : m = 165 :=
sorry

end smallest_integer_of_lcm_gcd_l179_179184


namespace john_extra_hours_l179_179916

theorem john_extra_hours (daily_earnings : ℕ) (hours_worked : ℕ) (bonus : ℕ) (hourly_wage : ℕ) (total_earnings_with_bonus : ℕ) (total_hours_with_bonus : ℕ) : 
  daily_earnings = 80 ∧ 
  hours_worked = 8 ∧ 
  bonus = 20 ∧ 
  hourly_wage = 10 ∧ 
  total_earnings_with_bonus = daily_earnings + bonus ∧
  total_hours_with_bonus = total_earnings_with_bonus / hourly_wage → 
  total_hours_with_bonus - hours_worked = 2 := 
by 
  sorry

end john_extra_hours_l179_179916


namespace no_such_abc_exists_l179_179161

-- Define the conditions for the leading coefficients and constant terms
def leading_coeff_conditions (a b c : ℝ) : Prop :=
  ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0))

def constant_term_conditions (a b c : ℝ) : Prop :=
  ((c > 0 ∧ a < 0 ∧ b < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0) ∨ (b > 0 ∧ c < 0 ∧ a < 0))

-- The final statement that encapsulates the contradiction
theorem no_such_abc_exists : ¬ ∃ a b c : ℝ, leading_coeff_conditions a b c ∧ constant_term_conditions a b c :=
by
  sorry

end no_such_abc_exists_l179_179161


namespace simplify_expression_l179_179286

variables (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b)

theorem simplify_expression :
  (a^4 - a^2 * b^2) / (a - b)^2 / (a * (a + b) / b^2) * (b^2 / a) = b^4 / (a - b) :=
by
  sorry

end simplify_expression_l179_179286


namespace zoo_visitors_per_hour_l179_179288

theorem zoo_visitors_per_hour 
    (h1 : ∃ V, 0.80 * V = 320)
    (h2 : ∃ H : Nat, H = 8)
    : ∃ N : Nat, N = 50 :=
by
  sorry

end zoo_visitors_per_hour_l179_179288


namespace find_angle_B_l179_179590

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l179_179590


namespace sum_of_first_13_terms_is_39_l179_179852

-- Definition of arithmetic sequence and the given condition
def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

-- Given condition
axiom given_condition {a : ℕ → ℤ} (h : arithmetic_sequence a) : a 5 + a 6 + a 7 = 9

-- The main theorem
theorem sum_of_first_13_terms_is_39 {a : ℕ → ℤ} (h : arithmetic_sequence a) (h9 : a 5 + a 6 + a 7 = 9) : sum_of_first_n_terms a 12 = 39 :=
sorry

end sum_of_first_13_terms_is_39_l179_179852


namespace expression_value_l179_179428

theorem expression_value : (15 + 7)^2 - (15^2 + 7^2) = 210 := by
  sorry

end expression_value_l179_179428


namespace cost_per_sq_meter_l179_179381

def tank_dimensions : ℝ × ℝ × ℝ := (25, 12, 6)
def total_plastering_cost : ℝ := 186
def total_plastering_area : ℝ :=
  let (length, width, height) := tank_dimensions
  let area_bottom := length * width
  let area_longer_walls := length * height * 2
  let area_shorter_walls := width * height * 2
  area_bottom + area_longer_walls + area_shorter_walls

theorem cost_per_sq_meter : total_plastering_cost / total_plastering_area = 0.25 := by
  sorry

end cost_per_sq_meter_l179_179381


namespace eccentricity_hyperbola_l179_179068

variables (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variables (H : c = Real.sqrt (a^2 + b^2))
variables (L1 : ∀ x y : ℝ, x = c → (x^2/a^2 - y^2/b^2 = 1))
variables (L2 : ∀ (B C : ℝ × ℝ), (B.1 = c ∧ C.1 = c) ∧ (B.2 = -C.2) ∧ (B.2 = b^2/a))

theorem eccentricity_hyperbola : ∃ e, e = 2 :=
sorry

end eccentricity_hyperbola_l179_179068


namespace simplify_expression_l179_179849

variable (x y : ℝ)

theorem simplify_expression (A B : ℝ) (hA : A = x^2) (hB : B = y^2) :
  (A + B) / (A - B) + (A - B) / (A + B) = 2 * (x^4 + y^4) / (x^4 - y^4) :=
by {
  sorry
}

end simplify_expression_l179_179849


namespace sufficient_but_not_necessary_condition_l179_179486

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 0 → |x| > 0) ∧ (¬ (|x| > 0 → x > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l179_179486


namespace students_recess_time_l179_179711

def initial_recess : ℕ := 20

def extra_minutes_as (as : ℕ) : ℕ := 4 * as
def extra_minutes_bs (bs : ℕ) : ℕ := 3 * bs
def extra_minutes_cs (cs : ℕ) : ℕ := 2 * cs
def extra_minutes_ds (ds : ℕ) : ℕ := ds
def extra_minutes_es (es : ℕ) : ℤ := - es
def extra_minutes_fs (fs : ℕ) : ℤ := -2 * fs

def total_recess (as bs cs ds es fs : ℕ) : ℤ :=
  initial_recess + 
  (extra_minutes_as as + extra_minutes_bs bs +
  extra_minutes_cs cs + extra_minutes_ds ds +
  extra_minutes_es es + extra_minutes_fs fs : ℤ)

theorem students_recess_time :
  total_recess 10 12 14 5 3 2 = 122 := by sorry

end students_recess_time_l179_179711


namespace find_b_l179_179998

theorem find_b (b : ℝ) (h_floor : b + ⌊b⌋ = 22.6) : b = 11.6 :=
sorry

end find_b_l179_179998


namespace largest_gcd_sum780_l179_179180

theorem largest_gcd_sum780 (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 780) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 390 ∧ (∀ (d' : ℕ), d' = Nat.gcd a b → d' ≤ 390) :=
sorry

end largest_gcd_sum780_l179_179180


namespace number_minus_45_l179_179877

theorem number_minus_45 (x : ℕ) (h1 : (x / 2) / 2 = 85 + 45) : x - 45 = 475 := by
  sorry

end number_minus_45_l179_179877


namespace smallest_term_4_in_c_seq_l179_179839

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

noncomputable def b_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n * (n - 1) + 15

noncomputable def c_seq (n : ℕ) : ℚ :=
  if n = 0 then 0 else (b_seq n) / (a_seq n)

theorem smallest_term_4_in_c_seq : 
  ∀ n : ℕ, n > 0 → c_seq 4 ≤ c_seq n :=
sorry

end smallest_term_4_in_c_seq_l179_179839


namespace circle_through_points_line_perpendicular_and_tangent_to_circle_max_area_triangle_l179_179570

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 10

theorem circle_through_points (A B : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (6, 0)) (h_center : ∃ C: ℝ × ℝ, C.1 - C.2 - 4 = 0 ∧ (circle_eq C.1 C.2)) : ∀ x y, circle_eq x y ↔ (x - 3) ^ 2 + (y + 1) ^ 2 = 10 := 
by sorry

theorem line_perpendicular_and_tangent_to_circle (line_slope : ℝ) (tangent : ∀ x y, circle_eq x y → (x + 3*y + 10 = 0 ∨ x + 3*y - 10 = 0)) : ∀ x, x + 3*y + 10 = 0 ∨ x + 3*y - 10 = 0 :=
by sorry

theorem max_area_triangle (A B P : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (6, 0)) (hP : circle_eq P.1 P.2) : ∃ area : ℝ, area = 5 + 5 * Real.sqrt 2
:= 
by sorry

end circle_through_points_line_perpendicular_and_tangent_to_circle_max_area_triangle_l179_179570


namespace f_monotonic_f_odd_find_a_k_range_l179_179058
open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a

-- (1) Prove the monotonicity of the function f
theorem f_monotonic (a : ℝ) : ∀ {x y : ℝ}, x < y → f a x < f a y := sorry

-- (2) If f is an odd function, find the value of the real number a
theorem f_odd_find_a : ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -f a x) → a = -1/2 := sorry

-- (3) Under the condition in (2), if the inequality holds for all x ∈ ℝ, find the range of values for k
theorem k_range (k : ℝ) :
  (∀ x : ℝ, f (-1/2) (x^2 - 2*x) + f (-1/2) (2*x^2 - k) > 0) → k < -1/3 := sorry

end f_monotonic_f_odd_find_a_k_range_l179_179058


namespace non_zero_x_satisfies_equation_l179_179422

theorem non_zero_x_satisfies_equation :
  ∃ (x : ℝ), (x ≠ 0) ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16 / 7 :=
by {
  sorry
}

end non_zero_x_satisfies_equation_l179_179422


namespace find_range_of_m_l179_179845

noncomputable def p (m : ℝ) : Prop := 1 - Real.sqrt 2 < m ∧ m < 1 + Real.sqrt 2
noncomputable def q (m : ℝ) : Prop := 0 < m ∧ m < 4

theorem find_range_of_m (m : ℝ) (hpq : p m ∨ q m) (hnp : ¬ p m) : 1 + Real.sqrt 2 ≤ m ∧ m < 4 :=
sorry

end find_range_of_m_l179_179845


namespace total_red_papers_l179_179732

-- Defining the number of red papers in one box and the number of boxes Hoseok has
def red_papers_per_box : ℕ := 2
def number_of_boxes : ℕ := 2

-- Statement to prove
theorem total_red_papers : (red_papers_per_box * number_of_boxes) = 4 := by
  sorry

end total_red_papers_l179_179732


namespace part1_average_decrease_rate_part2_unit_price_reduction_l179_179197

-- Part 1: Prove the average decrease rate is 10%
theorem part1_average_decrease_rate (p0 p2 : ℝ) (x : ℝ) 
    (h1 : p0 = 200) 
    (h2 : p2 = 162) 
    (hx : (1 - x)^2 = p2 / p0) : x = 0.1 :=
by {
    sorry
}

-- Part 2: Prove the unit price reduction should be 15 yuan
theorem part2_unit_price_reduction (p_sell p_factory profit : ℝ) (n_initial dn m : ℝ)
    (h3 : p_sell = 200)
    (h4 : p_factory = 162)
    (h5 : n_initial = 20)
    (h6 : dn = 10)
    (h7 : profit = 1150)
    (hx : (38 - m) * (n_initial + 2 * m) = profit) : m = 15 :=
by {
    sorry
}

end part1_average_decrease_rate_part2_unit_price_reduction_l179_179197


namespace P_inter_Q_eq_l179_179364

def P (x : ℝ) : Prop := -1 < x ∧ x < 3
def Q (x : ℝ) : Prop := -2 < x ∧ x < 1

theorem P_inter_Q_eq : {x | P x} ∩ {x | Q x} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end P_inter_Q_eq_l179_179364


namespace jerry_age_l179_179759

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J - 8) (h2 : M = 24) : J = 8 :=
by
  sorry

end jerry_age_l179_179759


namespace first_point_x_coord_l179_179612

variables (m n : ℝ)

theorem first_point_x_coord (h1 : m = 2 * n + 5) (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m = 2 * n + 5 :=
by 
  sorry

end first_point_x_coord_l179_179612


namespace tenth_day_of_month_is_monday_l179_179865

theorem tenth_day_of_month_is_monday (Sundays_on_even_dates : ℕ → Prop)
  (h1: Sundays_on_even_dates 2)
  (h2: Sundays_on_even_dates 16)
  (h3: Sundays_on_even_dates 30) :
  ∃ k : ℕ, 10 = k + 2 + 7 * 1 ∧ k.succ.succ.succ.succ.succ.succ.succ.succ.succ.succ = 1 :=
by sorry

end tenth_day_of_month_is_monday_l179_179865


namespace quadratic_complete_square_l179_179834

theorem quadratic_complete_square (x p q : ℤ) 
  (h_eq : x^2 - 6 * x + 3 = 0) 
  (h_pq_form : x^2 - 6 * x + (p - x)^2 = q) 
  (h_int : ∀ t, t = p + q) : p + q = 3 := sorry

end quadratic_complete_square_l179_179834


namespace fraction_not_on_time_l179_179685

theorem fraction_not_on_time (total_attendees : ℕ) (male_fraction female_fraction male_on_time_fraction female_on_time_fraction : ℝ)
  (H1 : male_fraction = 3/5)
  (H2 : male_on_time_fraction = 7/8)
  (H3 : female_on_time_fraction = 4/5)
  : ((1 - (male_fraction * male_on_time_fraction + (1 - male_fraction) * female_on_time_fraction)) = 3/20) :=
sorry

end fraction_not_on_time_l179_179685


namespace airplane_seat_count_l179_179473

theorem airplane_seat_count (s : ℝ) 
  (h1 : 30 + 0.2 * s + 0.75 * s = s) : 
  s = 600 :=
sorry

end airplane_seat_count_l179_179473


namespace find_image_point_l179_179255

noncomputable def lens_equation (t f k : ℝ) : Prop :=
  (1 / k) + (1 / t) = (1 / f)

theorem find_image_point
  (O F T T_star K_star K : ℝ)
  (OT OTw OTw_star FK : ℝ)
  (OT_eq : OT = OTw)
  (OTw_star_eq : OTw_star = OT)
  (similarity_condition : ∀ (CTw_star OF : ℝ), CTw_star / OF = (CTw_star + OK) / OK)
  : lens_equation OTw FK K :=
sorry

end find_image_point_l179_179255


namespace point_p_final_position_l179_179456

theorem point_p_final_position :
  let P_start := -2
  let P_right := P_start + 5
  let P_final := P_right - 4
  P_final = -1 :=
by
  sorry

end point_p_final_position_l179_179456


namespace euler_totient_divisibility_l179_179079

theorem euler_totient_divisibility (n : ℕ) (h : n > 0) : n ∣ Nat.totient (2^n - 1) := by
  sorry

end euler_totient_divisibility_l179_179079


namespace quadratic_function_solution_l179_179752

theorem quadratic_function_solution :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, g (x + 1) - g x = 2 * x + 3 ∧ g 2 - g 6 = -40) :=
sorry

end quadratic_function_solution_l179_179752


namespace tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l179_179195

variable (α : ℝ)
variable (π : ℝ) [Fact (π > 0)]

-- Assume condition
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Goal (1): Prove that tan(α + π/4) = -3
theorem tan_sum_pi_div_4 : Real.tan (α + π / 4) = -3 :=
by
  sorry

-- Goal (2): Prove that (sin(2α) / (sin^2(α) + sin(α) * cos(α) - cos(2α) - 1)) = 1
theorem sin_fraction_simplifies_to_1 :
  (Real.sin (2 * α)) / (Real.sin (α)^2 + Real.sin (α) * Real.cos (α) - Real.cos (2 * α) - 1) = 1 :=
by
  sorry

end tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l179_179195


namespace kids_played_on_monday_l179_179651

theorem kids_played_on_monday (total : ℕ) (tuesday : ℕ) (monday : ℕ) (h_total : total = 16) (h_tuesday : tuesday = 14) :
  monday = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end kids_played_on_monday_l179_179651


namespace new_machine_rate_l179_179960

def old_machine_rate : ℕ := 100
def total_bolts : ℕ := 500
def time_hours : ℕ := 2

theorem new_machine_rate (R : ℕ) : 
  (old_machine_rate * time_hours + R * time_hours = total_bolts) → 
  R = 150 := 
by
  sorry

end new_machine_rate_l179_179960


namespace find_equation_of_tangent_line_perpendicular_l179_179462

noncomputable def tangent_line_perpendicular_to_curve (a b : ℝ) : Prop :=
  (∃ (P : ℝ × ℝ), P = (-1, -3) ∧ 2 * P.1 - 6 * P.2 + 1 = 0 ∧ P.2 = P.1^3 + 5 * P.1^2 - 5) ∧
  (-3) = 3 * (-1)^2 + 6 * (-1)

theorem find_equation_of_tangent_line_perpendicular :
  tangent_line_perpendicular_to_curve (-1) (-3) →
  ∀ x y : ℝ, 3 * x + y + 6 = 0 :=
by
  sorry

end find_equation_of_tangent_line_perpendicular_l179_179462


namespace concyclic_iff_l179_179194

variables {A B C H O' N D : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace H]
variables [MetricSpace O'] [MetricSpace N] [MetricSpace D]
variables (a b c R : ℝ)

-- Conditions from the problem
def is_orthocenter (H : Type*) (A B C : Type*) : Prop :=
  -- definition of orthocenter using suitable predicates (omitted for brevity) 
  sorry

def is_circumcenter (O' : Type*) (B H C : Type*) : Prop :=
  -- definition of circumcenter using suitable predicates (omitted for brevity) 
  sorry

def is_midpoint (N : Type*) (A O' : Type*) : Prop :=
  -- definition of midpoint using suitable predicates (omitted for brevity) 
  sorry

def is_reflection (N D : Type*) (B C : Type*) : Prop :=
  -- definition of reflection about the side BC (omitted for brevity) 
  sorry

-- Definition that points A, B, C, D are concyclic
def are_concyclic (A B C D : Type*) : Prop :=
  -- definition using suitable predicates (omitted for brevity)
  sorry

-- Main theorem statement
theorem concyclic_iff (h1 : is_orthocenter H A B C) (h2 : is_circumcenter O' B H C) 
                      (h3 : is_midpoint N A O') (h4 : is_reflection N D B C)
                      (ha : a = 1) (hb : b = 1) (hc : c = 1) (hR : R = 1) :
  are_concyclic A B C D ↔ b^2 + c^2 - a^2 = 3 * R^2 := 
sorry

end concyclic_iff_l179_179194


namespace equal_distribution_l179_179028

namespace MoneyDistribution

def Ann_initial := 777
def Bill_initial := 1111
def Charlie_initial := 1555
def target_amount := 1148
def Bill_to_Ann := 371
def Charlie_to_Bill := 408

theorem equal_distribution :
  (Bill_initial - Bill_to_Ann + Charlie_to_Bill = target_amount) ∧
  (Ann_initial + Bill_to_Ann = target_amount) ∧
  (Charlie_initial - Charlie_to_Bill = target_amount) :=
by
  sorry

end MoneyDistribution

end equal_distribution_l179_179028


namespace toys_per_rabbit_l179_179854

theorem toys_per_rabbit 
  (rabbits toys_mon toys_wed toys_fri toys_sat : ℕ) 
  (hrabbits : rabbits = 16) 
  (htoys_mon : toys_mon = 6)
  (htoys_wed : toys_wed = 2 * toys_mon)
  (htoys_fri : toys_fri = 4 * toys_mon)
  (htoys_sat : toys_sat = toys_wed / 2) :
  (toys_mon + toys_wed + toys_fri + toys_sat) / rabbits = 3 :=
by 
  sorry

end toys_per_rabbit_l179_179854


namespace jessica_older_than_claire_l179_179157

-- Define the current age of Claire
def claire_current_age := 20 - 2

-- Define the current age of Jessica
def jessica_current_age := 24

-- Prove that Jessica is 6 years older than Claire
theorem jessica_older_than_claire : jessica_current_age - claire_current_age = 6 :=
by
  -- Definitions of the ages
  let claire_current_age := 18
  let jessica_current_age := 24

  -- Prove the age difference
  sorry

end jessica_older_than_claire_l179_179157


namespace probability_5_consecutive_heads_in_8_flips_l179_179693

noncomputable def probability_at_least_5_consecutive_heads (n : ℕ) : ℚ :=
  if n = 8 then 5 / 128 else 0  -- Using conditional given the specificity to n = 8

theorem probability_5_consecutive_heads_in_8_flips : 
  probability_at_least_5_consecutive_heads 8 = 5 / 128 := 
by
  -- Proof to be provided here
  sorry

end probability_5_consecutive_heads_in_8_flips_l179_179693


namespace min_cuts_to_one_meter_pieces_l179_179454

theorem min_cuts_to_one_meter_pieces (x y : ℕ) (hx : x + y = 30) (hl : 3 * x + 4 * y = 100) : (2 * x + 3 * y) = 70 := 
by sorry

end min_cuts_to_one_meter_pieces_l179_179454


namespace total_pieces_of_mail_l179_179036

-- Definitions based on given conditions
def pieces_each_friend_delivers : ℕ := 41
def pieces_johann_delivers : ℕ := 98
def number_of_friends : ℕ := 2

-- Theorem statement to prove the total number of pieces of mail delivered
theorem total_pieces_of_mail :
  (number_of_friends * pieces_each_friend_delivers) + pieces_johann_delivers = 180 := 
by
  -- proof would go here
  sorry

end total_pieces_of_mail_l179_179036


namespace find_q_l179_179793

noncomputable def Sn (n : ℕ) (d : ℚ) : ℚ :=
  d^2 * (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def Tn (n : ℕ) (d : ℚ) (q : ℚ) : ℚ :=
  d^2 * (1 - q^n) / (1 - q)

theorem find_q (d : ℚ) (q : ℚ) (hd : d ≠ 0) (hq : 0 < q ∧ q < 1) :
  Sn 3 d / Tn 3 d q = 14 → q = 1 / 2 :=
by
  sorry

end find_q_l179_179793


namespace volume_of_cylinder_in_pyramid_l179_179833

theorem volume_of_cylinder_in_pyramid
  (a α : ℝ)
  (sin_alpha : ℝ := Real.sin α)
  (tan_alpha : ℝ := Real.tan α)
  (sin_pi_four_alpha : ℝ := Real.sin (Real.pi / 4 + α))
  (sqrt_two : ℝ := Real.sqrt 2) :
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3) / (128 * sin_pi_four_alpha^3) =
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3 / (128 * sin_pi_four_alpha^3)) :=
by
  sorry

end volume_of_cylinder_in_pyramid_l179_179833


namespace negation_of_existence_implies_universal_l179_179722

theorem negation_of_existence_implies_universal :
  ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_of_existence_implies_universal_l179_179722


namespace find_n_l179_179237

def Point : Type := ℝ × ℝ

def A : Point := (5, -8)
def B : Point := (9, -30)
def C (n : ℝ) : Point := (n, n)

def collinear (p1 p2 p3 : Point) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_n (n : ℝ) (h : collinear A B (C n)) : n = 3 := 
by
  sorry

end find_n_l179_179237


namespace max_annual_profit_l179_179400

noncomputable def annual_sales_volume (x : ℝ) : ℝ := - (1 / 3) * x^2 + 2 * x + 21

noncomputable def annual_sales_profit (x : ℝ) : ℝ := (- (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126)

theorem max_annual_profit :
  ∀ x : ℝ, (x > 6) →
  (annual_sales_volume x) = - (1 / 3) * x^2 + 2 * x + 21 →
  (annual_sales_volume 10 = 23 / 3) →
  (21 - annual_sales_volume x = (1 / 3) * (x^2 - 6 * x)) →
    (annual_sales_profit x = - (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126) ∧
    ∃ x_max : ℝ, 
      (annual_sales_profit x_max = 36) ∧
      x_max = 9 :=
by
  sorry

end max_annual_profit_l179_179400


namespace find_rate_of_interest_l179_179996

noncomputable def rate_of_interest (P : ℝ) (r : ℝ) : Prop :=
  let CI2 := P * (1 + r)^2 - P
  let CI3 := P * (1 + r)^3 - P
  CI2 = 1200 ∧ CI3 = 1272 → r = 0.06

theorem find_rate_of_interest (P : ℝ) (r : ℝ) : rate_of_interest P r :=
by sorry

end find_rate_of_interest_l179_179996


namespace f_plus_2012_odd_l179_179006

def f : ℝ → ℝ → ℝ := sorry

lemma f_property (α β : ℝ) : f α β = 2012 := sorry

theorem f_plus_2012_odd : ∀ x : ℝ, f (-x) + 2012 = -(f x + 2012) :=
by
  sorry

end f_plus_2012_odd_l179_179006


namespace Jeff_has_20_trucks_l179_179627

theorem Jeff_has_20_trucks
  (T C : ℕ)
  (h1 : C = 2 * T)
  (h2 : T + C = 60) :
  T = 20 :=
sorry

end Jeff_has_20_trucks_l179_179627


namespace solve_arithmetic_sequence_l179_179646

theorem solve_arithmetic_sequence (y : ℝ) (h1 : y ^ 2 = (4 + 25) / 2) (h2 : y > 0) :
  y = Real.sqrt 14.5 :=
sorry

end solve_arithmetic_sequence_l179_179646


namespace range_of_m_l179_179597

noncomputable def range_m (a b : ℝ) (m : ℝ) : Prop :=
  (3 * a + 4 / b = 1) ∧ a > 0 ∧ b > 0 → (1 / a + 3 * b > m)

theorem range_of_m (m : ℝ) : (∀ a b : ℝ, (range_m a b m)) ↔ m < 27 :=
by
  sorry

end range_of_m_l179_179597


namespace find_smallest_w_l179_179346

theorem find_smallest_w (w : ℕ) (h : 0 < w) : 
  (∀ k, k = 2^5 ∨ k = 3^3 ∨ k = 12^2 → (k ∣ (936 * w))) ↔ w = 36 := by 
  sorry

end find_smallest_w_l179_179346


namespace lattice_points_on_segment_l179_179403

theorem lattice_points_on_segment : 
  let x1 := 5 
  let y1 := 23 
  let x2 := 47 
  let y2 := 297 
  ∃ n, n = 3 ∧ ∀ p : ℕ × ℕ, (p = (x1, y1) ∨ p = (x2, y2) ∨ ∃ t : ℕ, p = (x1 + t * (x2 - x1) / 2, y1 + t * (y2 - y1) / 2)) := 
sorry

end lattice_points_on_segment_l179_179403


namespace evaluate_magnitude_of_product_l179_179425

theorem evaluate_magnitude_of_product :
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  Complex.abs (z1 * z2) = 4 * Real.sqrt 43 := by
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  suffices Complex.abs z1 * Complex.abs z2 = 4 * Real.sqrt 43 by sorry
  sorry

end evaluate_magnitude_of_product_l179_179425


namespace mark_eggs_supply_l179_179985

theorem mark_eggs_supply 
  (dozens_per_day : ℕ)
  (eggs_per_dozen : ℕ)
  (extra_eggs : ℕ)
  (days_in_week : ℕ)
  (H1 : dozens_per_day = 5)
  (H2 : eggs_per_dozen = 12)
  (H3 : extra_eggs = 30)
  (H4 : days_in_week = 7) :
  (dozens_per_day * eggs_per_dozen + extra_eggs) * days_in_week = 630 :=
by
  sorry

end mark_eggs_supply_l179_179985


namespace value_of_X_l179_179805

def M : ℕ := 2024 / 4
def N : ℕ := M / 2
def X : ℕ := M + N

theorem value_of_X : X = 759 := by
  sorry

end value_of_X_l179_179805


namespace log_properties_l179_179189

theorem log_properties :
  (Real.log 5) ^ 2 + (Real.log 2) * (Real.log 50) = 1 :=
by sorry

end log_properties_l179_179189


namespace min_value_PA_d_l179_179584

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_value_PA_d :
  let A : ℝ × ℝ := (3, 4)
  let parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1
  let distance_to_line (P : ℝ × ℝ) (line_x : ℝ) : ℝ := abs (P.1 - line_x)
  let d : ℝ := distance_to_line P (-1)
  ∀ P : ℝ × ℝ, parabola P → (distance P A + d) ≥ 2 * Real.sqrt 5 :=
by
  sorry

end min_value_PA_d_l179_179584


namespace ben_examined_7_trays_l179_179866

open Int

def trays_of_eggs (total_eggs : ℕ) (eggs_per_tray : ℕ) : ℕ := total_eggs / eggs_per_tray

theorem ben_examined_7_trays : trays_of_eggs 70 10 = 7 :=
by
  sorry

end ben_examined_7_trays_l179_179866


namespace shopkeeper_discount_l179_179375

theorem shopkeeper_discount
  (CP LP SP : ℝ)
  (H_CP : CP = 100)
  (H_LP : LP = CP + 0.4 * CP)
  (H_SP : SP = CP + 0.33 * CP)
  (discount_percent : ℝ) :
  discount_percent = ((LP - SP) / LP) * 100 → discount_percent = 5 :=
by
  sorry

end shopkeeper_discount_l179_179375


namespace solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l179_179731

open Real

theorem solve_diff_eq_for_k_ne_zero (k : ℝ) (h : k ≠ 0) (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x * (f x + g x) ^ k)
  (hg : ∀ x, deriv g x = f x * (f x + g x) ^ k)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) + (1 - k * x) ^ (1 / k)) ∧ g x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) - (1 - k * x) ^ (1 / k))) :=
sorry

theorem solve_diff_eq_for_k_eq_zero (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x)
  (hg : ∀ x, deriv g x = f x)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = cosh x ∧ g x = sinh x) :=
sorry

end solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l179_179731


namespace ratio_of_hypotenuse_segments_l179_179951

theorem ratio_of_hypotenuse_segments (a b c d : ℝ) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : b = (3/4) * a)
  (h3 : d^2 = (c - d)^2 + b^2) :
  (d / (c - d)) = (4 / 3) :=
sorry

end ratio_of_hypotenuse_segments_l179_179951


namespace calculate_lower_profit_percentage_l179_179869

theorem calculate_lower_profit_percentage 
  (CP : ℕ) 
  (profitAt18Percent : ℕ) 
  (additionalProfit : ℕ)
  (hCP : CP = 800) 
  (hProfitAt18Percent : profitAt18Percent = 144) 
  (hAdditionalProfit : additionalProfit = 72) 
  (hProfitRelation : profitAt18Percent = additionalProfit + ((9 * CP) / 100)) :
  9 = ((9 * CP) / 100) :=
by
  sorry

end calculate_lower_profit_percentage_l179_179869


namespace correct_conclusion_l179_179815

theorem correct_conclusion :
  ¬ (-(-3)^2 = 9) ∧
  ¬ (-6 / 6 * (1 / 6) = -6) ∧
  ((-3)^2 * abs (-1/3) = 3) ∧
  ¬ (3^2 / 2 = 9 / 4) :=
by
  sorry

end correct_conclusion_l179_179815


namespace complex_number_value_l179_179324

theorem complex_number_value (i : ℂ) (h : i^2 = -1) : i^13 * (1 + i) = -1 + i :=
by
  sorry

end complex_number_value_l179_179324


namespace train_length_calculation_l179_179314

theorem train_length_calculation 
  (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) 
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 25) 
  (h_train_speed_kmph : train_speed_kmph = 57.6) : 
  ∃ train_length, train_length = 250 :=
by
  sorry

end train_length_calculation_l179_179314


namespace probability_A_C_winning_l179_179654

-- Definitions based on the conditions given
def students := ["A", "B", "C", "D"]

def isDistictPositions (x y : String) : Prop :=
  x ≠ y

-- Lean statement for the mathematical problem
theorem probability_A_C_winning :
  ∃ (P : ℚ), P = 1/6 :=
by
  sorry

end probability_A_C_winning_l179_179654


namespace josette_additional_cost_l179_179773

def small_bottle_cost_eur : ℝ := 1.50
def large_bottle_cost_eur : ℝ := 2.40
def exchange_rate : ℝ := 1.20
def discount_10_percent : ℝ := 0.10
def discount_15_percent : ℝ := 0.15

def initial_small_bottles : ℕ := 3
def initial_large_bottles : ℕ := 2

def initial_total_cost_eur : ℝ :=
  (small_bottle_cost_eur * initial_small_bottles) +
  (large_bottle_cost_eur * initial_large_bottles)

def discounted_cost_eur_10 : ℝ :=
  initial_total_cost_eur * (1 - discount_10_percent)

def additional_bottle_cost_eur : ℝ := small_bottle_cost_eur

def new_total_cost_eur : ℝ :=
  initial_total_cost_eur + additional_bottle_cost_eur

def discounted_cost_eur_15 : ℝ :=
  new_total_cost_eur * (1 - discount_15_percent)

def cost_usd (eur_amount : ℝ) : ℝ :=
  eur_amount * exchange_rate

def discounted_cost_usd_10 : ℝ := cost_usd discounted_cost_eur_10
def discounted_cost_usd_15 : ℝ := cost_usd discounted_cost_eur_15

def additional_cost_usd : ℝ :=
  discounted_cost_usd_15 - discounted_cost_usd_10

theorem josette_additional_cost :
  additional_cost_usd = 0.972 :=
by 
  sorry

end josette_additional_cost_l179_179773


namespace seohyun_initial_marbles_l179_179308

variable (M : ℤ)

theorem seohyun_initial_marbles (h1 : (2 / 3) * M = 12) (h2 : (1 / 2) * M + 12 = M) : M = 36 :=
sorry

end seohyun_initial_marbles_l179_179308


namespace problem_statement_l179_179546

open Real

variables {f : ℝ → ℝ} {a b c : ℝ}

-- f is twice differentiable on ℝ
axiom hf : ∀ x : ℝ, Differentiable ℝ f
axiom hf' : ∀ x : ℝ, Differentiable ℝ (deriv f)

-- ∃ c ∈ ℝ, such that (f(b) - f(a)) / (b - a) ≠ f'(c) for all a ≠ b
axiom hc : ∃ c : ℝ, ∀ a b : ℝ, a ≠ b → (f b - f a) / (b - a) ≠ deriv f c

-- Prove that f''(c) = 0
theorem problem_statement : ∃ c : ℝ, (∀ a b : ℝ, a ≠ b → (f b - f a) / (b - a) ≠ deriv f c) → deriv (deriv f) c = 0 := sorry

end problem_statement_l179_179546


namespace subset_singleton_zero_A_l179_179504

def A : Set ℝ := {x | x > -3}

theorem subset_singleton_zero_A : {0} ⊆ A := 
by
  sorry  -- Proof is not required

end subset_singleton_zero_A_l179_179504


namespace find_f_x_l179_179275

theorem find_f_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x - 1) : 
  ∀ x : ℤ, f x = 2 * x - 3 :=
sorry

end find_f_x_l179_179275


namespace graph_inverse_point_sum_l179_179365

theorem graph_inverse_point_sum 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h1 : ∀ x, f_inv (f x) = x) 
  (h2 : ∀ x, f (f_inv x) = x) 
  (h3 : f 2 = 6) 
  (h4 : (2, 3) ∈ {p : ℝ × ℝ | p.snd = f p.fst / 2}) :
  (6, 1) ∈ {p : ℝ × ℝ | p.snd = f_inv p.fst / 2} ∧ (6 + 1 = 7) :=
by
  sorry

end graph_inverse_point_sum_l179_179365


namespace neighborhood_has_exactly_one_item_l179_179234

noncomputable def neighborhood_conditions : Prop :=
  let total_households := 120
  let households_no_items := 15
  let households_car_and_bike := 28
  let households_car := 52
  let households_bike := 32
  let households_scooter := 18
  let households_skateboard := 8
  let households_at_least_one_item := total_households - households_no_items
  let households_car_only := households_car - households_car_and_bike
  let households_bike_only := households_bike - households_car_and_bike
  let households_exactly_one_item := households_car_only + households_bike_only + households_scooter + households_skateboard
  households_at_least_one_item = 105 ∧ households_exactly_one_item = 54

theorem neighborhood_has_exactly_one_item :
  neighborhood_conditions :=
by
  -- Proof goes here
  sorry

end neighborhood_has_exactly_one_item_l179_179234


namespace annual_interest_rate_l179_179771

/-- Suppose you invested $10000, part at a certain annual interest rate and the rest at 9% annual interest.
After one year, you received $684 in interest. You invested $7200 at this rate and the rest at 9%.
What is the annual interest rate of the first investment? -/
theorem annual_interest_rate (r : ℝ) 
  (h : 7200 * r + 2800 * 0.09 = 684) : r = 0.06 :=
by
  sorry

end annual_interest_rate_l179_179771


namespace beth_score_l179_179326

-- Conditions
variables (B : ℕ)  -- Beth's points are some number.
def jan_points := 10 -- Jan scored 10 points.
def judy_points := 8 -- Judy scored 8 points.
def angel_points := 11 -- Angel scored 11 points.

-- First team has 3 more points than the second team
def first_team_points := B + jan_points
def second_team_points := judy_points + angel_points
def first_team_more_than_second := first_team_points = second_team_points + 3

-- Statement: Prove that B = 12
theorem beth_score : first_team_more_than_second → B = 12 :=
by
  -- Proof will be provided here
  sorry

end beth_score_l179_179326


namespace polar_coordinates_of_point_l179_179715

theorem polar_coordinates_of_point :
  ∃ (r θ : ℝ), r = 2 ∧ θ = (2 * Real.pi) / 3 ∧
  (r > 0) ∧ (0 ≤ θ) ∧ (θ < 2 * Real.pi) ∧
  (-1, Real.sqrt 3) = (r * Real.cos θ, r * Real.sin θ) :=
by 
  sorry

end polar_coordinates_of_point_l179_179715


namespace gcd_pow_sub_l179_179140

theorem gcd_pow_sub (a b : ℕ) (ha : a = 2000) (hb : b = 1990) :
  Nat.gcd (2^a - 1) (2^b - 1) = 1023 :=
sorry

end gcd_pow_sub_l179_179140


namespace nicholas_crackers_l179_179357

theorem nicholas_crackers (marcus_crackers mona_crackers nicholas_crackers : ℕ) 
  (h1 : marcus_crackers = 3 * mona_crackers)
  (h2 : nicholas_crackers = mona_crackers + 6)
  (h3 : marcus_crackers = 27) : nicholas_crackers = 15 := by
  sorry

end nicholas_crackers_l179_179357


namespace car_second_half_speed_l179_179119

theorem car_second_half_speed (D : ℝ) (V : ℝ) :
  let average_speed := 60  -- km/hr
  let first_half_speed := 75 -- km/hr
  average_speed = D / ((D / 2) / first_half_speed + (D / 2) / V) ->
  V = 150 :=
by
  sorry

end car_second_half_speed_l179_179119


namespace song_book_cost_correct_l179_179599

noncomputable def cost_of_trumpet : ℝ := 145.16
noncomputable def total_spent : ℝ := 151.00
noncomputable def cost_of_song_book : ℝ := total_spent - cost_of_trumpet

theorem song_book_cost_correct : cost_of_song_book = 5.84 :=
  by
    sorry

end song_book_cost_correct_l179_179599


namespace today_is_thursday_l179_179499

-- Define the days of the week as an enumerated type
inductive DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek
| Sunday : DayOfWeek

open DayOfWeek

-- Define the conditions for the lion and the unicorn
def lion_truth (d: DayOfWeek) : Bool :=
match d with
| Monday | Tuesday | Wednesday => false
| _ => true

def unicorn_truth (d: DayOfWeek) : Bool :=
match d with
| Thursday | Friday | Saturday => false
| _ => true

-- The statement made by the lion and the unicorn
def lion_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => lion_truth Sunday
| Tuesday => lion_truth Monday
| Wednesday => lion_truth Tuesday
| Thursday => lion_truth Wednesday
| Friday => lion_truth Thursday
| Saturday => lion_truth Friday
| Sunday => lion_truth Saturday

def unicorn_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => unicorn_truth Sunday
| Tuesday => unicorn_truth Monday
| Wednesday => unicorn_truth Tuesday
| Thursday => unicorn_truth Wednesday
| Friday => unicorn_truth Thursday
| Saturday => unicorn_truth Friday
| Sunday => unicorn_truth Saturday

-- Main theorem to prove the current day
theorem today_is_thursday (d: DayOfWeek) (lion_said: lion_statement d = false) (unicorn_said: unicorn_statement d = false) : d = Thursday :=
by
  -- Placeholder for actual proof
  sorry

end today_is_thursday_l179_179499


namespace find_original_number_l179_179315

theorem find_original_number (x : ℝ) (h : 0.5 * x = 30) : x = 60 :=
sorry

end find_original_number_l179_179315


namespace part_a_part_b_l179_179887

-- Define the predicate ensuring that among any three consecutive symbols, there is at least one zero
def valid_sequence (s : List Char) : Prop :=
  ∀ (i : Nat), i + 2 < s.length → (s.get! i = '0' ∨ s.get! (i + 1) = '0' ∨ s.get! (i + 2) = '0')

-- Count the valid sequences given the number of 'X's and 'O's
noncomputable def count_valid_sequences (n_zeros n_crosses : Nat) : Nat :=
  sorry -- Implementation of the combinatorial counting

-- Part (a): n = 29
theorem part_a : count_valid_sequences 14 29 = 15 := by
  sorry

-- Part (b): n = 28
theorem part_b : count_valid_sequences 14 28 = 120 := by
  sorry

end part_a_part_b_l179_179887


namespace distinct_possible_lunches_l179_179667

def main_dishes := 3
def beverages := 3
def snacks := 3

theorem distinct_possible_lunches : main_dishes * beverages * snacks = 27 := by
  sorry

end distinct_possible_lunches_l179_179667


namespace find_k_l179_179208

theorem find_k (k : ℝ) (h : ∀ x y : ℝ, (x, y) = (-2, -1) → y = k * x + 2) : k = 3 / 2 :=
sorry

end find_k_l179_179208


namespace paint_fraction_l179_179926

variable (T C : ℕ) (h : T = 60) (t : ℕ) (partial_t : ℚ)

theorem paint_fraction (hT : T = 60) (ht : t = 12) : partial_t = t / T := by
  rw [ht, hT]
  norm_num
  sorry

end paint_fraction_l179_179926


namespace hermia_elected_probability_l179_179192

-- Define the problem statement and conditions in Lean 4
noncomputable def probability_hermia_elected (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : ℚ :=
  if n = 1 then 1 else (2^n - 1) / (n * 2^(n-1))

-- Lean theorem statement
theorem hermia_elected_probability (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : 
  probability_hermia_elected n h_odd h_pos = (2^n - 1) / (n * 2^(n-1)) :=
by
  sorry

end hermia_elected_probability_l179_179192


namespace find_xyz_l179_179172

variable (x y z : ℝ)
variable (h1 : x = 80 + 0.11 * 80)
variable (h2 : y = 120 - 0.15 * 120)
variable (h3 : z = 0.20 * (0.40 * (x + y)) + 0.40 * (x + y))

theorem find_xyz (hx : x = 88.8) (hy : y = 102) (hz : z = 91.584) : 
  x = 88.8 ∧ y = 102 ∧ z = 91.584 := by
  sorry

end find_xyz_l179_179172


namespace clown_blew_more_balloons_l179_179256

theorem clown_blew_more_balloons :
  ∀ (initial_balloons final_balloons additional_balloons : ℕ),
    initial_balloons = 47 →
    final_balloons = 60 →
    additional_balloons = final_balloons - initial_balloons →
    additional_balloons = 13 :=
by
  intros initial_balloons final_balloons additional_balloons h1 h2 h3
  sorry

end clown_blew_more_balloons_l179_179256


namespace simplify_evaluate_expression_l179_179418

theorem simplify_evaluate_expression (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (2 / (x + 1) + 1 / (x - 2)) / (x - 1) / (x - 2) = Real.sqrt 3 := by
  sorry

end simplify_evaluate_expression_l179_179418


namespace condition_of_A_with_respect_to_D_l179_179958

variables {A B C D : Prop}

theorem condition_of_A_with_respect_to_D (h1 : A → B) (h2 : ¬ (B → A)) (h3 : B ↔ C) (h4 : C → D) (h5 : ¬ (D → C)) :
  (D → A) ∧ ¬ (A → D) :=
by
  sorry

end condition_of_A_with_respect_to_D_l179_179958


namespace base_salary_is_1600_l179_179703

theorem base_salary_is_1600 (B : ℝ) (C : ℝ) (sales : ℝ) (fixed_salary : ℝ) :
  C = 0.04 ∧ sales = 5000 ∧ fixed_salary = 1800 ∧ (B + C * sales = fixed_salary) → B = 1600 :=
by sorry

end base_salary_is_1600_l179_179703


namespace sum_of_fourth_powers_eq_82_l179_179385

theorem sum_of_fourth_powers_eq_82 (x y : ℝ) (hx : x + y = -2) (hy : x * y = -3) :
  x^4 + y^4 = 82 :=
by
  sorry

end sum_of_fourth_powers_eq_82_l179_179385


namespace Ariana_running_time_l179_179806

theorem Ariana_running_time
  (time_Sadie : ℝ)
  (speed_Sadie : ℝ)
  (speed_Ariana : ℝ)
  (speed_Sarah : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_Sadie := speed_Sadie * time_Sadie)
  (time_Ariana_Sarah := total_time - time_Sadie)
  (distance_Ariana_Sarah := total_distance - distance_Sadie) :
  (6 * (time_Ariana_Sarah - (11 - 6 * (time_Ariana_Sarah / (speed_Ariana + (4 / speed_Sarah)))))
  = (0.5 : ℝ)) :=
by
  sorry

end Ariana_running_time_l179_179806


namespace solve_for_x_l179_179177

theorem solve_for_x : ∀ x : ℤ, 5 - x = 8 → x = -3 :=
by
  intros x h
  sorry

end solve_for_x_l179_179177


namespace parallel_line_plane_l179_179392

-- Define vectors
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Dot product definition
def dotProduct (u v : Vector3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

-- Options given
def optionA : Vector3D × Vector3D := (⟨1, 0, 0⟩, ⟨-2, 0, 0⟩)
def optionB : Vector3D × Vector3D := (⟨1, 3, 5⟩, ⟨1, 0, 1⟩)
def optionC : Vector3D × Vector3D := (⟨0, 2, 1⟩, ⟨-1, 0, -1⟩)
def optionD : Vector3D × Vector3D := (⟨1, -1, 3⟩, ⟨0, 3, 1⟩)

-- Main theorem
theorem parallel_line_plane :
  (dotProduct (optionA.fst) (optionA.snd) ≠ 0) ∧
  (dotProduct (optionB.fst) (optionB.snd) ≠ 0) ∧
  (dotProduct (optionC.fst) (optionC.snd) ≠ 0) ∧
  (dotProduct (optionD.fst) (optionD.snd) = 0) :=
by
  -- Using sorry to skip the proof
  sorry

end parallel_line_plane_l179_179392


namespace fraction_of_b_equals_4_15_of_a_is_0_4_l179_179883

variable (A B : ℤ)
variable (X : ℚ)

def a_and_b_together_have_1210 : Prop := A + B = 1210
def b_has_484 : Prop := B = 484
def fraction_of_b_equals_4_15_of_a : Prop := (4 / 15 : ℚ) * A = X * B

theorem fraction_of_b_equals_4_15_of_a_is_0_4
  (h1 : a_and_b_together_have_1210 A B)
  (h2 : b_has_484 B)
  (h3 : fraction_of_b_equals_4_15_of_a A B X) :
  X = 0.4 := sorry

end fraction_of_b_equals_4_15_of_a_is_0_4_l179_179883


namespace combined_weight_of_daughter_and_child_l179_179933

variables (M D C : ℝ)
axiom mother_daughter_grandchild_weight : M + D + C = 120
axiom daughter_weight : D = 48
axiom child_weight_fraction_of_grandmother : C = (1 / 5) * M

theorem combined_weight_of_daughter_and_child : D + C = 60 :=
  sorry

end combined_weight_of_daughter_and_child_l179_179933


namespace range_of_f_l179_179173

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end range_of_f_l179_179173


namespace fraction_exponent_evaluation_l179_179631

theorem fraction_exponent_evaluation : 
  (3 ^ 10 + 3 ^ 8) / (3 ^ 10 - 3 ^ 8) = 5 / 4 :=
by sorry

end fraction_exponent_evaluation_l179_179631


namespace ratio_sheep_horses_eq_six_seven_l179_179012

noncomputable def total_food_per_day : ℕ := 12880
noncomputable def food_per_horse_per_day : ℕ := 230
noncomputable def num_sheep : ℕ := 48
noncomputable def num_horses : ℕ := total_food_per_day / food_per_horse_per_day
noncomputable def ratio_sheep_to_horses := num_sheep / num_horses

theorem ratio_sheep_horses_eq_six_seven :
  ratio_sheep_to_horses = 6 / 7 :=
by
  sorry

end ratio_sheep_horses_eq_six_seven_l179_179012


namespace candy_total_l179_179902

theorem candy_total (r b : ℕ) (hr : r = 145) (hb : b = 3264) : r + b = 3409 := by
  -- We can use Lean's rewrite tactic to handle the equalities, but since proof is skipped,
  -- it's not necessary to write out detailed tactics here.
  sorry

end candy_total_l179_179902


namespace geometric_sequence_div_sum_l179_179774

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable def S (n : ℕ) : ℝ := sorry

theorem geometric_sequence_div_sum 
  (h₁ : S 3 = (1 - (2 : ℝ) ^ 3) / (1 - (2 : ℝ) ^ 2) * a 1)
  (h₂ : S 2 = (1 - (2 : ℝ) ^ 2) / (1 - 2) * a 1)
  (h₃ : 8 * a 2 = a 5) : 
  S 3 / S 2 = 7 / 3 := 
by
  sorry

end geometric_sequence_div_sum_l179_179774


namespace problem_statement_l179_179674

theorem problem_statement : 100 * 29.98 * 2.998 * 1000 = (2998)^2 :=
by
  sorry

end problem_statement_l179_179674


namespace compute_abc_l179_179788

theorem compute_abc (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30) 
  (h2 : (1 / a + 1 / b + 1 / c + 420 / (a * b * c) = 1)) : 
  a * b * c = 450 := 
sorry

end compute_abc_l179_179788


namespace rental_cost_equation_l179_179995

theorem rental_cost_equation (x : ℕ) (h : x > 0) :
  180 / x - 180 / (x + 2) = 3 :=
sorry

end rental_cost_equation_l179_179995


namespace find_fruit_cost_l179_179919

-- Define the conditions
def muffin_cost : ℝ := 2
def francis_muffin_count : ℕ := 2
def francis_fruit_count : ℕ := 2
def kiera_muffin_count : ℕ := 2
def kiera_fruit_count : ℕ := 1
def total_cost : ℝ := 17

-- Define the cost of each fruit cup
variable (F : ℝ)

-- The statement to be proved
theorem find_fruit_cost (h : francis_muffin_count * muffin_cost 
                + francis_fruit_count * F 
                + kiera_muffin_count * muffin_cost 
                + kiera_fruit_count * F = total_cost) : 
                F = 1.80 :=
by {
  sorry
}

end find_fruit_cost_l179_179919


namespace fraction_addition_l179_179702

theorem fraction_addition : (3 / 5) + (2 / 15) = 11 / 15 := sorry

end fraction_addition_l179_179702


namespace find_side_length_of_left_square_l179_179379

theorem find_side_length_of_left_square (x : ℕ) 
  (h1 : x + (x + 17) + (x + 11) = 52) : 
  x = 8 :=
by
  -- The proof will go here
  sorry

end find_side_length_of_left_square_l179_179379


namespace fruits_calculation_l179_179535

structure FruitStatus :=
  (initial_picked  : ℝ)
  (initial_eaten  : ℝ)

def apples_status : FruitStatus :=
  { initial_picked := 7.0 + 3.0 + 5.0, initial_eaten := 6.0 + 2.0 }

def pears_status : FruitStatus :=
  { initial_picked := 0, initial_eaten := 4.0 + 3.0 }  -- number of pears picked is unknown, hence 0

def oranges_status : FruitStatus :=
  { initial_picked := 8.0, initial_eaten := 8.0 }

def cherries_status : FruitStatus :=
  { initial_picked := 4.0, initial_eaten := 4.0 }

theorem fruits_calculation :
  (apples_status.initial_picked - apples_status.initial_eaten = 7.0) ∧
  (pears_status.initial_picked - pears_status.initial_eaten = 0) ∧  -- cannot be determined in the problem statement
  (oranges_status.initial_picked - oranges_status.initial_eaten = 0) ∧
  (cherries_status.initial_picked - cherries_status.initial_eaten = 0) :=
by {
  sorry
}

end fruits_calculation_l179_179535


namespace pencils_needed_l179_179604

theorem pencils_needed (pencilsA : ℕ) (pencilsB : ℕ) (classroomsA : ℕ) (classroomsB : ℕ) (total_shortage : ℕ)
  (hA : pencilsA = 480)
  (hB : pencilsB = 735)
  (hClassA : classroomsA = 6)
  (hClassB : classroomsB = 9)
  (hShortage : total_shortage = 85) 
  : 90 = 6 + 5 * ((total_shortage / (classroomsA + classroomsB)) + 1) * classroomsB :=
by {
  sorry
}

end pencils_needed_l179_179604


namespace square_field_area_l179_179043

def square_area (side_length : ℝ) : ℝ :=
  side_length * side_length

theorem square_field_area :
  square_area 20 = 400 := by
  sorry

end square_field_area_l179_179043


namespace hexagon_monochromatic_triangles_l179_179092

theorem hexagon_monochromatic_triangles :
  let hexagon_edges := 15 -- $\binom{6}{2}$
  let monochromatic_tri_prob := (1 / 3) -- Prob of one triangle being monochromatic
  let combinations := 20 -- $\binom{6}{3}$, total number of triangles in K_6
  let exactly_two_monochromatic := (combinations.choose 2) * (monochromatic_tri_prob ^ 2) * ((2 / 3) ^ 18)
  (exactly_two_monochromatic = 49807360 / 3486784401) := sorry

end hexagon_monochromatic_triangles_l179_179092


namespace odd_function_sin_cos_product_l179_179822

-- Prove that if the function f(x) = sin(x + α) - 2cos(x - α) is an odd function, then sin(α) * cos(α) = 2/5
theorem odd_function_sin_cos_product (α : ℝ)
  (hf : ∀ x, Real.sin (x + α) - 2 * Real.cos (x - α) = -(Real.sin (-x + α) - 2 * Real.cos (-x - α))) :
  Real.sin α * Real.cos α = 2 / 5 :=
  sorry

end odd_function_sin_cos_product_l179_179822


namespace boys_in_parkway_l179_179108

theorem boys_in_parkway (total_students : ℕ) (students_playing_soccer : ℕ) (percentage_boys_playing_soccer : ℝ)
                        (girls_not_playing_soccer : ℕ) :
                        total_students = 420 ∧ students_playing_soccer = 250 ∧ percentage_boys_playing_soccer = 0.86 
                        ∧ girls_not_playing_soccer = 73 → 
                        ∃ total_boys : ℕ, total_boys = 312 :=
by
  -- Proof omitted
  sorry

end boys_in_parkway_l179_179108


namespace complex_number_coordinates_l179_179882

-- Define i as the imaginary unit
def i := Complex.I

-- State the theorem
theorem complex_number_coordinates : (i * (1 - i)).re = 1 ∧ (i * (1 - i)).im = 1 :=
by
  -- Proof would go here
  sorry

end complex_number_coordinates_l179_179882


namespace swimming_speed_in_still_water_l179_179798

variable (v : ℝ) -- the person's swimming speed in still water

-- Conditions
variable (water_speed : ℝ := 4) -- speed of the water
variable (time : ℝ := 2) -- time taken to swim 12 km against the current
variable (distance : ℝ := 12) -- distance swam against the current

theorem swimming_speed_in_still_water :
  (v - water_speed) = distance / time → v = 10 :=
by
  sorry

end swimming_speed_in_still_water_l179_179798


namespace speed_conversion_l179_179950

theorem speed_conversion (speed_mps : ℝ) (conversion_factor : ℝ) (speed_kmph_expected : ℝ) :
  speed_mps = 35.0028 →
  conversion_factor = 3.6 →
  speed_kmph_expected = 126.01008 →
  speed_mps * conversion_factor = speed_kmph_expected :=
by
  intros h_mps h_cf h_kmph
  rw [h_mps, h_cf, h_kmph]
  sorry

end speed_conversion_l179_179950


namespace sufficient_condition_x_gt_2_l179_179856

theorem sufficient_condition_x_gt_2 (x : ℝ) (h : x > 2) : x^2 - 2 * x > 0 := by
  sorry

end sufficient_condition_x_gt_2_l179_179856


namespace proof_sets_l179_179226

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 3, 6}
def complement (s : Set ℕ) : Set ℕ := {x | x ∈ I ∧ x ∉ s}

theorem proof_sets :
  M ∩ (complement N) = {4, 5} ∧ {2, 7, 8} = complement (M ∪ N) :=
by
  sorry

end proof_sets_l179_179226


namespace range_of_a_l179_179549

theorem range_of_a (x y a : ℝ): 
  (x + 3 * y = 3 - a) ∧ (2 * x + y = 1 + 3 * a) ∧ (x + y > 3 * a + 4) ↔ (a < -3 / 2) :=
sorry

end range_of_a_l179_179549


namespace time_to_cover_length_l179_179378

/-- Define the conditions for the problem -/
def angle_deg : ℝ := 30
def escalator_speed : ℝ := 12
def length_along_incline : ℝ := 160
def person_speed : ℝ := 8

/-- Define the combined speed as the sum of the escalator speed and the person speed -/
def combined_speed : ℝ := escalator_speed + person_speed

/-- Theorem stating the time taken to cover the length of the escalator is 8 seconds -/
theorem time_to_cover_length : (length_along_incline / combined_speed) = 8 := by
  sorry

end time_to_cover_length_l179_179378


namespace hyperbola_foci_l179_179196

/-- The coordinates of the foci of the hyperbola y^2 / 3 - x^2 = 1 are (0, ±2). -/
theorem hyperbola_foci (x y : ℝ) :
  x^2 - (y^2 / 3) = -1 → (0 = x ∧ (y = 2 ∨ y = -2)) :=
sorry

end hyperbola_foci_l179_179196


namespace six_times_number_eq_132_l179_179522

theorem six_times_number_eq_132 (x : ℕ) (h : x / 11 = 2) : 6 * x = 132 :=
sorry

end six_times_number_eq_132_l179_179522


namespace inequality_correct_l179_179059

theorem inequality_correct (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1) : (1 - a) ^ a > (1 - b) ^ b :=
sorry

end inequality_correct_l179_179059


namespace net_effect_on_sale_l179_179040

variable (P Q : ℝ) -- Price and Quantity

theorem net_effect_on_sale :
  let reduced_price := 0.40 * P
  let increased_quantity := 2.50 * Q
  let price_after_tax := 0.44 * P
  let price_after_discount := 0.418 * P
  let final_revenue := price_after_discount * increased_quantity 
  let original_revenue := P * Q
  final_revenue / original_revenue = 1.045 :=
by
  sorry

end net_effect_on_sale_l179_179040


namespace remainder_div_modulo_l179_179007

theorem remainder_div_modulo (N : ℕ) (h1 : N % 19 = 7) : N % 20 = 6 :=
by
  sorry

end remainder_div_modulo_l179_179007


namespace temperature_on_friday_is_35_l179_179166

variables (M T W Th F : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem temperature_on_friday_is_35
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 43)
  (h4 : is_odd M)
  (h5 : is_odd T)
  (h6 : is_odd W)
  (h7 : is_odd Th)
  (h8 : is_odd F) : 
  F = 35 :=
sorry

end temperature_on_friday_is_35_l179_179166


namespace find_k_l179_179519

theorem find_k (k l : ℝ) (C : ℝ × ℝ) (OC : ℝ) (A B D : ℝ × ℝ)
  (hC_coords : C = (0, 3))
  (hl_val : l = 3)
  (line_eqn : ∀ x, y = k * x + l)
  (intersect_eqn : ∀ x, y = 1 / x)
  (hA_coords : A = (1 / 6, 6))
  (hD_coords : D = (1 / 6, 6))
  (dist_ABC : dist A B = dist B C)
  (dist_BCD : dist B C = dist C D)
  (OC_val : OC = 3) :
  k = 18 := 
sorry

end find_k_l179_179519


namespace sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l179_179146

theorem sqrt_three_is_irrational_and_infinite_non_repeating_decimal :
    ∀ r : ℝ, r = Real.sqrt 3 → ¬ ∃ (m n : ℤ), n ≠ 0 ∧ r = m / n := by
    sorry

end sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l179_179146


namespace tourists_number_l179_179063

theorem tourists_number (m : ℕ) (k l : ℤ) (n : ℕ) (hn : n = 23) (hm1 : 2 * m ≡ 1 [MOD n]) (hm2 : 3 * m ≡ 13 [MOD n]) (hn_gt_13 : n > 13) : n = 23 := 
by
  sorry

end tourists_number_l179_179063


namespace factorization_correct_l179_179716

theorem factorization_correct (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l179_179716


namespace max_value_bx_plus_a_l179_179787

variable (a b : ℝ)

theorem max_value_bx_plus_a (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ |b * x + a| = 2 :=
by
  -- Proof goes here
  sorry

end max_value_bx_plus_a_l179_179787


namespace simplify_expression_l179_179830

theorem simplify_expression :
  let a := (1/2)^2
  let b := (1/2)^3
  let c := (1/2)^4
  let d := (1/2)^5
  1 / (1/a + 1/b + 1/c + 1/d) = 1/60 :=
by
  sorry

end simplify_expression_l179_179830


namespace find_x_for_parallel_vectors_l179_179134

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (4, x)
def b : ℝ × ℝ := (-4, 4)

-- Define parallelism condition for two 2D vectors
def are_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Define the main theorem statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : are_parallel (a x) b) : x = -4 :=
by sorry

end find_x_for_parallel_vectors_l179_179134


namespace point_outside_circle_l179_179794

theorem point_outside_circle
  (radius : ℝ) (distance : ℝ) (h_radius : radius = 8) (h_distance : distance = 10) :
  distance > radius :=
by sorry

end point_outside_circle_l179_179794


namespace tap_filling_time_l179_179979

theorem tap_filling_time
  (T : ℝ)
  (H1 : 10 > 0) -- Second tap can empty the cistern in 10 hours
  (H2 : T > 0)  -- First tap's time must be positive
  (H3 : (1 / T) - (1 / 10) = (3 / 20))  -- Both taps together fill the cistern in 6.666... hours
  : T = 4 := sorry

end tap_filling_time_l179_179979


namespace sons_ages_l179_179323

theorem sons_ages (m n : ℕ) (h : m * n + m + n = 34) : 
  (m = 4 ∧ n = 6) ∨ (m = 6 ∧ n = 4) :=
sorry

end sons_ages_l179_179323


namespace coordinates_P_wrt_origin_l179_179804

/-- Define a point P with coordinates we are given. -/
def P : ℝ × ℝ := (-1, 2)

/-- State that the coordinates of P with respect to the origin O are (-1, 2). -/
theorem coordinates_P_wrt_origin : P = (-1, 2) :=
by
  -- Proof would go here
  sorry

end coordinates_P_wrt_origin_l179_179804


namespace dog_count_l179_179740

theorem dog_count 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (long_furred_brown : ℕ) 
  (total : ℕ) 
  (h1 : long_furred = 29) 
  (h2 : brown = 17) 
  (h3 : neither = 8) 
  (h4 : long_furred_brown = 9)
  (h5 : total = long_furred + brown - long_furred_brown + neither) : 
  total = 45 :=
by 
  sorry

end dog_count_l179_179740


namespace determine_d_l179_179512

-- Given conditions
def equation (d x : ℝ) : Prop := 3 * (5 + d * x) = 15 * x + 15

-- Proof statement
theorem determine_d (d : ℝ) : (∀ x : ℝ, equation d x) ↔ d = 5 :=
by
  sorry

end determine_d_l179_179512


namespace provisions_last_60_days_l179_179463

/-
A garrison of 1000 men has provisions for a certain number of days.
At the end of 15 days, a reinforcement of 1250 arrives, and it is now found that the provisions will last only for 20 days more.
Prove that the provisions were supposed to last initially for 60 days.
-/

def initial_provisions (D : ℕ) : Prop :=
  let initial_garrison := 1000
  let reinforcement_garrison := 1250
  let days_spent := 15
  let remaining_days := 20
  initial_garrison * (D - days_spent) = (initial_garrison + reinforcement_garrison) * remaining_days

theorem provisions_last_60_days (D : ℕ) : initial_provisions D → D = 60 := by
  sorry

end provisions_last_60_days_l179_179463


namespace find_n_l179_179835

-- Defining the conditions.
def condition_one : Prop :=
  ∀ (c d : ℕ), 
  (80 * 2 * c = 320) ∧ (80 * 2 * d = 160)

def condition_two : Prop :=
  ∀ (c d : ℕ), 
  (100 * 3 * c = 450) ∧ (100 * 3 * d = 300)

def condition_three (n : ℕ) : Prop :=
  ∀ (c d : ℕ), 
  (40 * 4 * c = n) ∧ (40 * 4 * d = 160)

-- Statement of the proof problem using the conditions.
theorem find_n : 
  condition_one ∧ condition_two ∧ condition_three 160 :=
by
  sorry

end find_n_l179_179835


namespace roots_of_equation_l179_179517

theorem roots_of_equation : ∀ x : ℝ, x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by sorry

end roots_of_equation_l179_179517


namespace panthers_second_half_points_l179_179397

theorem panthers_second_half_points (C1 P1 C2 P2 : ℕ) 
  (h1 : C1 + P1 = 38) 
  (h2 : C1 = P1 + 16) 
  (h3 : C1 + C2 + P1 + P2 = 58) 
  (h4 : C1 + C2 = P1 + P2 + 22) : 
  P2 = 7 :=
by 
  -- Definitions and substitutions are skipped here
  sorry

end panthers_second_half_points_l179_179397


namespace max_min_product_l179_179366

theorem max_min_product (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h_sum : p + q + r = 13) (h_prod_sum : p * q + q * r + r * p = 30) :
  ∃ n, n = min (p * q) (min (q * r) (r * p)) ∧ n = 10 :=
by
  sorry

end max_min_product_l179_179366
