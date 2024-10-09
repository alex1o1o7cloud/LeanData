import Mathlib

namespace janet_total_pockets_l439_43968

theorem janet_total_pockets
  (total_dresses : ℕ)
  (dresses_with_pockets : ℕ)
  (dresses_with_2_pockets : ℕ)
  (dresses_with_3_pockets : ℕ)
  (pockets_from_2 : ℕ)
  (pockets_from_3 : ℕ)
  (total_pockets : ℕ)
  (h1 : total_dresses = 24)
  (h2 : dresses_with_pockets = total_dresses / 2)
  (h3 : dresses_with_2_pockets = dresses_with_pockets / 3)
  (h4 : dresses_with_3_pockets = dresses_with_pockets - dresses_with_2_pockets)
  (h5 : pockets_from_2 = 2 * dresses_with_2_pockets)
  (h6 : pockets_from_3 = 3 * dresses_with_3_pockets)
  (h7 : total_pockets = pockets_from_2 + pockets_from_3)
  : total_pockets = 32 := 
by
  sorry

end janet_total_pockets_l439_43968


namespace complement_correct_l439_43950

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}
def complement_U (M: Set ℕ) (U: Set ℕ) := {x ∈ U | x ∉ M}

theorem complement_correct : complement_U M U = {3, 4, 6} :=
by 
  sorry

end complement_correct_l439_43950


namespace find_z_l439_43938

theorem find_z (z : ℂ) (h : (Complex.I * z = 4 + 3 * Complex.I)) : z = 3 - 4 * Complex.I :=
by
  sorry

end find_z_l439_43938


namespace compute_diff_squares_l439_43990

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l439_43990


namespace smallest_discount_n_l439_43922

noncomputable def effective_discount_1 (x : ℝ) : ℝ := 0.64 * x
noncomputable def effective_discount_2 (x : ℝ) : ℝ := 0.614125 * x
noncomputable def effective_discount_3 (x : ℝ) : ℝ := 0.63 * x 

theorem smallest_discount_n (x : ℝ) (n : ℕ) (hx : x > 0) :
  (1 - n / 100 : ℝ) * x < effective_discount_1 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_2 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_3 x ↔ n = 39 := 
sorry

end smallest_discount_n_l439_43922


namespace sin_330_l439_43917

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Outline the proof here without providing it
  -- sorry to delay the proof
  sorry

end sin_330_l439_43917


namespace find_15th_term_l439_43958

-- Define the initial terms and the sequence properties
def first_term := 4
def second_term := 13
def third_term := 22

-- Define the common difference
def common_difference := second_term - first_term

-- Define the nth term formula for arithmetic sequence
def nth_term (a d : ℕ) (n : ℕ) := a + (n - 1) * d

-- State the theorem
theorem find_15th_term : nth_term first_term common_difference 15 = 130 := by
  -- The proof will come here
  sorry

end find_15th_term_l439_43958


namespace quadratic_sum_constants_l439_43971

-- Define the quadratic expression
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 27 * x + 135

-- Define the representation of the quadratic in the form a(x + b)^2 + c
def quadratic_rewritten (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum_constants :
  ∃ a b c, (∀ x, quadratic x = quadratic_rewritten a b c x) ∧ a + b + c = 197.75 :=
by
  sorry

end quadratic_sum_constants_l439_43971


namespace total_chocolate_bars_in_large_box_l439_43943

def large_box_contains_18_small_boxes : ℕ := 18
def small_box_contains_28_chocolate_bars : ℕ := 28

theorem total_chocolate_bars_in_large_box :
  (large_box_contains_18_small_boxes * small_box_contains_28_chocolate_bars) = 504 := 
by
  sorry

end total_chocolate_bars_in_large_box_l439_43943


namespace solve_y_l439_43906

theorem solve_y (y : ℝ) (h : (y ^ (7 / 8)) = 4) : y = 2 ^ (16 / 7) :=
sorry

end solve_y_l439_43906


namespace blood_flow_scientific_notation_l439_43940

theorem blood_flow_scientific_notation (blood_flow : ℝ) (h : blood_flow = 4900) : 
  4900 = 4.9 * (10 ^ 3) :=
by
  sorry

end blood_flow_scientific_notation_l439_43940


namespace find_x_l439_43935

theorem find_x : ∃ x : ℤ, (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 4 ∧ x = 28 := 
by sorry

end find_x_l439_43935


namespace cone_height_l439_43915

theorem cone_height (slant_height r : ℝ) (lateral_area : ℝ) (h : ℝ) 
  (h_slant : slant_height = 13) 
  (h_area : lateral_area = 65 * Real.pi) 
  (h_lateral_area : lateral_area = Real.pi * r * slant_height) 
  (h_radius : r = 5) :
  h = Real.sqrt (slant_height ^ 2 - r ^ 2) :=
by
  -- Definitions and conditions
  have h_slant_height : slant_height = 13 := h_slant
  have h_lateral_area_value : lateral_area = 65 * Real.pi := h_area
  have h_lateral_surface_area : lateral_area = Real.pi * r * slant_height := h_lateral_area
  have h_radius_5 : r = 5 := h_radius
  sorry -- Proof is omitted

end cone_height_l439_43915


namespace mrs_oaklyn_profit_is_correct_l439_43960

def cost_of_buying_rugs (n : ℕ) (cost_per_rug : ℕ) : ℕ :=
  n * cost_per_rug

def transportation_fee (n : ℕ) (fee_per_rug : ℕ) : ℕ :=
  n * fee_per_rug

def selling_price_before_tax (n : ℕ) (price_per_rug : ℕ) : ℕ :=
  n * price_per_rug

def total_tax (price_before_tax : ℕ) (tax_rate : ℕ) : ℕ :=
  price_before_tax * tax_rate / 100

def total_selling_price_after_tax (price_before_tax : ℕ) (tax_amount : ℕ) : ℕ :=
  price_before_tax + tax_amount

def profit (selling_price_after_tax : ℕ) (cost_of_buying : ℕ) (transport_fee : ℕ) : ℕ :=
  selling_price_after_tax - (cost_of_buying + transport_fee)

def rugs := 20
def cost_per_rug := 40
def transport_fee_per_rug := 5
def price_per_rug := 60
def tax_rate := 10

theorem mrs_oaklyn_profit_is_correct : 
  profit 
    (total_selling_price_after_tax 
      (selling_price_before_tax rugs price_per_rug) 
      (total_tax (selling_price_before_tax rugs price_per_rug) tax_rate)
    )
    (cost_of_buying_rugs rugs cost_per_rug) 
    (transportation_fee rugs transport_fee_per_rug) 
  = 420 :=
by sorry

end mrs_oaklyn_profit_is_correct_l439_43960


namespace Danica_additional_cars_l439_43910

theorem Danica_additional_cars (num_cars : ℕ) (cars_per_row : ℕ) (current_cars : ℕ) 
  (h_cars_per_row : cars_per_row = 8) (h_current_cars : current_cars = 35) :
  ∃ n, num_cars = 5 ∧ n = 40 ∧ n - current_cars = num_cars := 
by
  sorry

end Danica_additional_cars_l439_43910


namespace frog_arrangement_l439_43957

def arrangementCount (total_frogs green_frogs red_frogs blue_frog : ℕ) : ℕ :=
  if (green_frogs + red_frogs + blue_frog = total_frogs ∧ 
      green_frogs = 3 ∧ red_frogs = 4 ∧ blue_frog = 1) then 40320 else 0

theorem frog_arrangement :
  arrangementCount 8 3 4 1 = 40320 :=
by {
  -- Proof omitted
  sorry
}

end frog_arrangement_l439_43957


namespace common_tangent_lines_count_l439_43913

-- Define the first circle
def C1 (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9

-- Define the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y - 9 = 0

-- Definition for the number of common tangent lines between two circles
def number_of_common_tangent_lines (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The theorem stating the number of common tangent lines between the given circles
theorem common_tangent_lines_count : number_of_common_tangent_lines C1 C2 = 2 := by
  sorry

end common_tangent_lines_count_l439_43913


namespace sum_reciprocal_squares_l439_43953

open Real

theorem sum_reciprocal_squares (a : ℝ) (A B C D E F : ℝ)
    (square_ABCD : A = 0 ∧ B = a ∧ D = a ∧ C = a)
    (line_intersects : A = 0 ∧ E ≥ 0 ∧ E ≤ a ∧ F ≥ 0 ∧ F ≤ a) 
    (phi : ℝ) : 
    (cos phi * (a/cos phi))^2 + (sin phi * (a/sin phi))^2 = (1/a^2) := 
sorry 

end sum_reciprocal_squares_l439_43953


namespace tom_tickets_l439_43926

theorem tom_tickets :
  (45 + 38 + 52) - (12 + 23) = 100 := by
sorry

end tom_tickets_l439_43926


namespace slope_of_line_l439_43974

theorem slope_of_line (x y : ℝ) (h : 4 * x - 7 * y = 28) : (∃ m b : ℝ, y = m * x + b ∧ m = 4 / 7) :=
by
  -- Proof omitted
  sorry

end slope_of_line_l439_43974


namespace square_and_product_l439_43939

theorem square_and_product (x : ℤ) (h : x^2 = 1764) : (x = 42) ∧ ((x + 2) * (x - 2) = 1760) :=
by
  sorry

end square_and_product_l439_43939


namespace find_g_values_l439_43933

theorem find_g_values
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x * y) = x * g y)
  (h2 : g 1 = 30) :
  g 50 = 1500 ∧ g 0.5 = 15 :=
by
  sorry

end find_g_values_l439_43933


namespace leak_drains_in_34_hours_l439_43944

-- Define the conditions
def pump_rate := 1 / 2 -- rate at which the pump fills the tank (tanks per hour)
def time_with_leak := 17 / 8 -- time to fill the tank with the pump and the leak (hours)

-- Define the combined rate of pump and leak
def combined_rate := 1 / time_with_leak -- tanks per hour

-- Define the leak rate
def leak_rate := pump_rate - combined_rate -- solve for leak rate

-- Define the proof statement
theorem leak_drains_in_34_hours : (1 / leak_rate) = 34 := by
    sorry

end leak_drains_in_34_hours_l439_43944


namespace max_f_max_g_pow_f_l439_43903

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4 * x + 3) / (x^2 + 7 * x + 14)
noncomputable def g (x : ℝ) : ℝ := (x^2 - 5 * x + 10) / (x^2 + 5 * x + 20)

theorem max_f : ∀ x : ℝ, f x ≤ 2 := by
  intro x
  sorry

theorem max_g_pow_f : ∀ x : ℝ, g x ^ f x ≤ 9 := by
  intro x
  sorry

end max_f_max_g_pow_f_l439_43903


namespace sum_of_coefficients_sum_even_odd_coefficients_l439_43966

noncomputable def P (x : ℝ) : ℝ := (2 * x^2 - 2 * x + 1)^17 * (3 * x^2 - 3 * x + 1)^17

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

theorem sum_even_odd_coefficients :
  (P 1 + P (-1)) / 2 = (1 + 35^17) / 2 ∧ (P 1 - P (-1)) / 2 = (1 - 35^17) / 2 := by
  sorry

end sum_of_coefficients_sum_even_odd_coefficients_l439_43966


namespace parallelogram_sides_l439_43945

theorem parallelogram_sides (x y : ℕ) 
  (h₁ : 2 * x + 3 = 9) 
  (h₂ : 8 * y - 1 = 7) : 
  x + y = 4 :=
by
  sorry

end parallelogram_sides_l439_43945


namespace peanuts_added_l439_43919

theorem peanuts_added (initial final added : ℕ) (h1 : initial = 4) (h2 : final = 8) (h3 : final = initial + added) : added = 4 :=
by
  rw [h1] at h3
  rw [h2] at h3
  sorry

end peanuts_added_l439_43919


namespace argument_friends_count_l439_43931

-- Define the conditions
def original_friends: ℕ := 20
def current_friends: ℕ := 19
def new_friend: ℕ := 1

-- Define the statement that needs to be proved
theorem argument_friends_count : 
  (original_friends + new_friend - current_friends = 1) :=
by
  -- Placeholder for the proof
  sorry

end argument_friends_count_l439_43931


namespace statement1_statement2_l439_43918

def is_pow_of_two (a : ℕ) : Prop := ∃ n : ℕ, a = 2^(n + 1)
def in_A (a : ℕ) : Prop := is_pow_of_two a
def not_in_A (a : ℕ) : Prop := ¬ in_A a ∧ a ≠ 1

theorem statement1 : 
  ∀ (a : ℕ), in_A a → ∀ (b : ℕ), b < 2 * a - 1 → ¬ (2 * a ∣ b * (b + 1)) := 
by {
  sorry
}

theorem statement2 :
  ∀ (a : ℕ), not_in_A a → ∃ (b : ℕ), b < 2 * a - 1 ∧ (2 * a ∣ b * (b + 1)) :=
by {
  sorry
}

end statement1_statement2_l439_43918


namespace simplify_expression_l439_43902

theorem simplify_expression : 2 - 2 / (2 + Real.sqrt 5) + 2 / (2 - Real.sqrt 5) = 2 + 4 * Real.sqrt 5 :=
by sorry

end simplify_expression_l439_43902


namespace total_cards_after_giveaway_l439_43969

def ben_basketball_boxes := 8
def cards_per_basketball_box := 20
def ben_baseball_boxes := 10
def cards_per_baseball_box := 15
def ben_football_boxes := 12
def cards_per_football_box := 12

def alex_hockey_boxes := 6
def cards_per_hockey_box := 15
def alex_soccer_boxes := 9
def cards_per_soccer_box := 18

def cards_given_away := 175

def total_cards_for_ben := 
  (ben_basketball_boxes * cards_per_basketball_box) + 
  (ben_baseball_boxes * cards_per_baseball_box) + 
  (ben_football_boxes * cards_per_football_box)

def total_cards_for_alex := 
  (alex_hockey_boxes * cards_per_hockey_box) + 
  (alex_soccer_boxes * cards_per_soccer_box)

def total_cards_before_exchange := total_cards_for_ben + total_cards_for_alex

def ben_gives_to_alex := 
  (ben_basketball_boxes * (cards_per_basketball_box / 2)) + 
  (ben_baseball_boxes * (cards_per_baseball_box / 2))

def total_cards_remaining := total_cards_before_exchange - cards_given_away

theorem total_cards_after_giveaway :
  total_cards_before_exchange - cards_given_away = 531 := by
  sorry

end total_cards_after_giveaway_l439_43969


namespace soap_bubble_radius_l439_43989

/-- Given a spherical soap bubble that divides into two equal hemispheres, 
    each having a radius of 6 * (2 ^ (1 / 3)) cm, 
    show that the radius of the original bubble is also 6 * (2 ^ (1 / 3)) cm. -/
theorem soap_bubble_radius (r : ℝ) (R : ℝ) (π : ℝ) 
  (h_r : r = 6 * (2 ^ (1 / 3)))
  (h_volume_eq : (4 / 3) * π * R^3 = (4 / 3) * π * r^3) : 
  R = 6 * (2 ^ (1 / 3)) :=
by
  sorry

end soap_bubble_radius_l439_43989


namespace division_quotient_proof_l439_43979

theorem division_quotient_proof :
  (300324 / 29 = 10356) →
  (100007892 / 333 = 300324) :=
by
  intros h1
  sorry

end division_quotient_proof_l439_43979


namespace problem1_problem2_l439_43954

-- Define the conditions and the target proofs based on identified questions and answers

-- Problem 1
theorem problem1 (x : ℚ) : 
  9 * (x - 2)^2 ≤ 25 ↔ x = 11 / 3 ∨ x = 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (x y : ℚ) :
  (x + 1) / 3 = 2 * y ∧ 2 * (x + 1) - y = 11 ↔ x = 5 ∧ y = 1 :=
sorry

end problem1_problem2_l439_43954


namespace calculate_expression_l439_43994

theorem calculate_expression : 
  -1^4 - (1 - 0.5) * (2 - (-3)^2) = 5 / 2 :=
by
  sorry

end calculate_expression_l439_43994


namespace geometric_sequence_150th_term_l439_43998

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem geometric_sequence_150th_term :
  geometric_sequence 8 (-1 / 2) 150 = -8 * (1 / 2) ^ 149 :=
by
  -- This is the proof placeholder
  sorry

end geometric_sequence_150th_term_l439_43998


namespace polynomial_remainder_l439_43916

theorem polynomial_remainder (z : ℂ) :
  let dividend := 4*z^3 - 5*z^2 - 17*z + 4
  let divisor := 4*z + 6
  let quotient := z^2 - 4*z + (1/4 : ℝ)
  let remainder := 5*z^2 + 6*z + (5/2 : ℝ)
  dividend = divisor * quotient + remainder := sorry

end polynomial_remainder_l439_43916


namespace g_at_3_l439_43923

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 : (∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2 + 1) → 
  g 3 = 130 / 21 := 
by 
  sorry

end g_at_3_l439_43923


namespace line_forms_equivalence_l439_43936

noncomputable def points (P Q : ℝ × ℝ) : Prop := 
  ∃ m c, ∃ b d, P = (b, m * b + c) ∧ Q = (d, m * d + c)

theorem line_forms_equivalence :
  points (-2, 3) (4, -1) →
  (∀ x y : ℝ, (y + 1) / (3 + 1) = (x - 4) / (-2 - 4)) ∧
  (∀ x y : ℝ, y + 1 = - (2 / 3) * (x - 4)) ∧
  (∀ x y : ℝ, y = - (2 / 3) * x + 5 / 3) ∧
  (∀ x y : ℝ, x / (5 / 2) + y / (5 / 3) = 1) :=
  sorry

end line_forms_equivalence_l439_43936


namespace smallest_positive_integer_x_l439_43991

theorem smallest_positive_integer_x (x : ℕ) (h : 725 * x ≡ 1165 * x [MOD 35]) : x = 7 :=
sorry

end smallest_positive_integer_x_l439_43991


namespace financier_invariant_l439_43927

theorem financier_invariant (D A : ℤ) (hD : D = 1 ∨ D = 10 * (A - 1) + D ∨ D = D - 1 + 10 * A)
  (hA : A = 0 ∨ A = A + 10 * (1 - D) ∨ A = A - 1):
  (D - A) % 11 = 1 := 
sorry

end financier_invariant_l439_43927


namespace rate_of_interest_l439_43929

theorem rate_of_interest (P T SI CI : ℝ) (hP : P = 4000) (hT : T = 2) (hSI : SI = 400) (hCI : CI = 410) :
  ∃ r : ℝ, SI = (P * r * T) / 100 ∧ CI = P * ((1 + r / 100) ^ T - 1) ∧ r = 5 :=
by
  sorry

end rate_of_interest_l439_43929


namespace total_amount_paid_after_discount_l439_43999

-- Define the given conditions
def marked_price_per_article : ℝ := 10
def discount_percentage : ℝ := 0.60
def number_of_articles : ℕ := 2

-- Proving the total amount paid
theorem total_amount_paid_after_discount : 
  (marked_price_per_article * number_of_articles) * (1 - discount_percentage) = 8 := by
  sorry

end total_amount_paid_after_discount_l439_43999


namespace kendra_total_earnings_l439_43959

-- Definitions of the conditions based on the problem statement
def kendra_earnings_2014 : ℕ := 30000 - 8000
def laurel_earnings_2014 : ℕ := 30000
def kendra_earnings_2015 : ℕ := laurel_earnings_2014 + (laurel_earnings_2014 / 5)

-- The statement to be proved
theorem kendra_total_earnings : kendra_earnings_2014 + kendra_earnings_2015 = 58000 :=
by
  -- Using Lean tactics for the proof
  sorry

end kendra_total_earnings_l439_43959


namespace probability_picasso_consecutive_l439_43988

-- Given Conditions
def total_pieces : Nat := 12
def picasso_paintings : Nat := 4

-- Desired probability calculation
theorem probability_picasso_consecutive :
  (Nat.factorial (total_pieces - picasso_paintings + 1) * Nat.factorial picasso_paintings) / 
  Nat.factorial total_pieces = 1 / 55 :=
by
  sorry

end probability_picasso_consecutive_l439_43988


namespace polynomial_perfect_square_l439_43996

theorem polynomial_perfect_square (k : ℤ) : (∃ b : ℤ, (x + b)^2 = x^2 + 8 * x + k) -> k = 16 := by
  sorry

end polynomial_perfect_square_l439_43996


namespace parallelogram_area_l439_43952

theorem parallelogram_area (b h : ℕ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end parallelogram_area_l439_43952


namespace necessary_but_not_sufficient_l439_43986

variable (x : ℝ)

theorem necessary_but_not_sufficient (h : x > 2) : x > 1 ∧ ¬ (x > 1 → x > 2) :=
by
  sorry

end necessary_but_not_sufficient_l439_43986


namespace number_of_shoes_lost_l439_43907

-- Definitions for the problem conditions
def original_pairs : ℕ := 20
def pairs_left : ℕ := 15
def shoes_per_pair : ℕ := 2

-- Translating the conditions to individual shoe counts
def original_shoes : ℕ := original_pairs * shoes_per_pair
def remaining_shoes : ℕ := pairs_left * shoes_per_pair

-- Statement of the proof problem
theorem number_of_shoes_lost : original_shoes - remaining_shoes = 10 := by
  sorry

end number_of_shoes_lost_l439_43907


namespace incorrect_conclusions_l439_43964

variables (a b : ℝ)

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem incorrect_conclusions :
  a > 0 → b > 0 → a ≠ 1 → b ≠ 1 → log_base a b > 1 →
  (a < 1 ∧ b > a ∨ (¬ (b < 1 ∧ b < a) ∧ ¬ (a < 1 ∧ a < b))) :=
by intros ha hb ha_ne1 hb_ne1 hlog; sorry

end incorrect_conclusions_l439_43964


namespace B_can_complete_work_in_6_days_l439_43921

theorem B_can_complete_work_in_6_days (A B : ℝ) (h1 : (A + B) = 1 / 4) (h2 : A = 1 / 12) : B = 1 / 6 := 
by
  sorry

end B_can_complete_work_in_6_days_l439_43921


namespace locus_of_point_M_l439_43992

open Real

def distance (x y: ℝ × ℝ): ℝ :=
  ((x.1 - y.1)^2 + (x.2 - y.2)^2)^(1/2)

theorem locus_of_point_M :
  (∀ (M : ℝ × ℝ), 
     distance M (2, 0) + 1 = abs (M.1 + 3)) 
  → ∀ (M : ℝ × ℝ), M.2^2 = 8 * M.1 :=
sorry

end locus_of_point_M_l439_43992


namespace more_blue_count_l439_43937

-- Definitions based on the conditions given in the problem
def total_people : ℕ := 150
def more_green : ℕ := 95
def both_green_blue : ℕ := 35
def neither_green_blue : ℕ := 25

-- The Lean statement to prove the number of people who believe turquoise is "more blue"
theorem more_blue_count : 
  (total_people - neither_green_blue) - (more_green - both_green_blue) = 65 :=
by 
  sorry

end more_blue_count_l439_43937


namespace weight_of_new_person_l439_43914

-- Define the given conditions
variables (avg_increase : ℝ) (num_people : ℕ) (replaced_weight : ℝ)
variable (new_weight : ℝ)

-- These are the conditions directly from the problem
axiom avg_weight_increase : avg_increase = 4.5
axiom number_of_people : num_people = 6
axiom person_to_replace_weight : replaced_weight = 75

-- Mathematical equivalent of the proof problem
theorem weight_of_new_person :
  new_weight = replaced_weight + avg_increase * num_people := 
sorry

end weight_of_new_person_l439_43914


namespace volume_truncated_cone_l439_43975

-- Define the geometric constants
def large_base_radius : ℝ := 10
def small_base_radius : ℝ := 5
def height_truncated_cone : ℝ := 8

-- The statement to prove the volume of the truncated cone
theorem volume_truncated_cone :
  let V_large := (1/3) * Real.pi * (large_base_radius^2) * (height_truncated_cone + height_truncated_cone)
  let V_small := (1/3) * Real.pi * (small_base_radius^2) * height_truncated_cone
  V_large - V_small = (1400/3) * Real.pi :=
by
  sorry

end volume_truncated_cone_l439_43975


namespace a_received_share_l439_43946

variables (I_a I_b I_c b_share total_investment total_profit a_share : ℕ)
  (h1 : I_a = 11000)
  (h2 : I_b = 15000)
  (h3 : I_c = 23000)
  (h4 : b_share = 3315)
  (h5 : total_investment = I_a + I_b + I_c)
  (h6 : total_profit = b_share * total_investment / I_b)
  (h7 : a_share = I_a * total_profit / total_investment)

theorem a_received_share : a_share = 2662 := by
  sorry

end a_received_share_l439_43946


namespace book_pages_total_l439_43985

theorem book_pages_total
  (days_in_week : ℕ)
  (daily_read_times : ℕ)
  (pages_per_time : ℕ)
  (additional_pages_per_day : ℕ)
  (num_days : days_in_week = 7)
  (times_per_day : daily_read_times = 3)
  (pages_each_time : pages_per_time = 6)
  (extra_pages : additional_pages_per_day = 2) :
  daily_read_times * pages_per_time + additional_pages_per_day * days_in_week = 140 := 
sorry

end book_pages_total_l439_43985


namespace find_y_l439_43993

variable (t : ℝ)
variable (x : ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := x = 3 - t
def condition2 : Prop := y = 2 * t + 11
def condition3 : Prop := x = 1

theorem find_y (h1 : condition1 x t) (h2 : condition2 t y) (h3 : condition3 x) : y = 15 := by
  sorry

end find_y_l439_43993


namespace frac_mul_eq_l439_43948

theorem frac_mul_eq : (2/3) * (3/8) = 1/4 := 
by 
  sorry

end frac_mul_eq_l439_43948


namespace basil_plants_yielded_l439_43909

def initial_investment (seed_cost soil_cost : ℕ) : ℕ :=
  seed_cost + soil_cost

def total_revenue (net_profit initial_investment : ℕ) : ℕ :=
  net_profit + initial_investment

def basil_plants (total_revenue price_per_plant : ℕ) : ℕ :=
  total_revenue / price_per_plant

theorem basil_plants_yielded
  (seed_cost soil_cost net_profit price_per_plant expected_plants : ℕ)
  (h_seed_cost : seed_cost = 2)
  (h_soil_cost : soil_cost = 8)
  (h_net_profit : net_profit = 90)
  (h_price_per_plant : price_per_plant = 5)
  (h_expected_plants : expected_plants = 20) :
  basil_plants (total_revenue net_profit (initial_investment seed_cost soil_cost)) price_per_plant = expected_plants :=
by
  -- Proof steps will be here
  sorry

end basil_plants_yielded_l439_43909


namespace part1_part2_a_part2_b_part2_c_l439_43980

noncomputable def f (x a : ℝ) := Real.exp x - x - a

theorem part1 (x : ℝ) : f x 0 > x := 
by 
  -- here would be the proof
  sorry

theorem part2_a (a : ℝ) : a > 1 → ∃ z₁ z₂ : ℝ, f z₁ a = 0 ∧ f z₂ a = 0 ∧ z₁ ≠ z₂ := 
by 
  -- here would be the proof
  sorry

theorem part2_b (a : ℝ) : a < 1 → ¬ (∃ z : ℝ, f z a = 0) := 
by 
  -- here would be the proof
  sorry

theorem part2_c : f 0 1 = 0 := 
by 
  -- here would be the proof
  sorry

end part1_part2_a_part2_b_part2_c_l439_43980


namespace car_production_l439_43951

theorem car_production (mp : ℕ) (h1 : 1800 = (mp + 50) * 12) : mp = 100 :=
by
  sorry

end car_production_l439_43951


namespace cos_alpha_plus_pi_six_l439_43928

theorem cos_alpha_plus_pi_six (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 4 / 5) : 
  Real.cos (α + Real.pi / 6) = - (4 / 5) := 
by 
  sorry

end cos_alpha_plus_pi_six_l439_43928


namespace minimize_PA_PB_l439_43973

theorem minimize_PA_PB 
  (A B : ℝ × ℝ) 
  (hA : A = (1, 3)) 
  (hB : B = (5, 1)) : 
  ∃ P : ℝ × ℝ, P = (4, 0) ∧ 
  ∀ P' : ℝ × ℝ, P'.snd = 0 → (dist P A + dist P B) ≤ (dist P' A + dist P' B) :=
sorry

end minimize_PA_PB_l439_43973


namespace Trisha_walked_total_distance_l439_43908

theorem Trisha_walked_total_distance 
  (d1 d2 d3 : ℝ) (h_d1 : d1 = 0.11) (h_d2 : d2 = 0.11) (h_d3 : d3 = 0.67) :
  d1 + d2 + d3 = 0.89 :=
by sorry

end Trisha_walked_total_distance_l439_43908


namespace point_B_in_third_quadrant_l439_43961

theorem point_B_in_third_quadrant (x y : ℝ) (hx : x < 0) (hy : y < 1) :
    (y - 1 < 0) ∧ (x < 0) :=
by
  sorry  -- proof to be filled

end point_B_in_third_quadrant_l439_43961


namespace shiny_pennies_probability_l439_43982

theorem shiny_pennies_probability :
  ∃ (a b : ℕ), gcd a b = 1 ∧ a / b = 5 / 11 ∧ a + b = 16 :=
sorry

end shiny_pennies_probability_l439_43982


namespace largest_square_test_plots_l439_43997

/-- 
  A fenced, rectangular field measures 30 meters by 45 meters. 
  An agricultural researcher has 1500 meters of fence that can be used for internal fencing to partition 
  the field into congruent, square test plots. 
  The entire field must be partitioned, and the sides of the squares must be parallel to the edges of the field. 
  What is the largest number of square test plots into which the field can be partitioned using all or some of the 1500 meters of fence?
 -/
theorem largest_square_test_plots
  (field_length : ℕ := 30)
  (field_width : ℕ := 45)
  (total_fence_length : ℕ := 1500):
  ∃ (n : ℕ), n = 576 := 
sorry

end largest_square_test_plots_l439_43997


namespace ab_bc_ca_abc_inequality_l439_43976

open Real

theorem ab_bc_ca_abc_inequality :
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 + a * b * c = 4 →
    0 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 2 :=
by
  intro a b c
  intro h
  sorry

end ab_bc_ca_abc_inequality_l439_43976


namespace doug_money_l439_43981

def money_problem (J D B: ℝ) : Prop :=
  J + D + B = 68 ∧
  J = 2 * B ∧
  J = (3 / 4) * D

theorem doug_money (J D B: ℝ) (h: money_problem J D B): D = 36.27 :=
by sorry

end doug_money_l439_43981


namespace harry_did_not_get_an_A_l439_43962

theorem harry_did_not_get_an_A
  (emily_Imp_frank : Prop)
  (frank_Imp_gina : Prop)
  (gina_Imp_harry : Prop)
  (exactly_one_did_not_get_an_A : ¬ (emily_Imp_frank ∧ frank_Imp_gina ∧ gina_Imp_harry)) :
  ¬ harry_Imp_gina :=
  sorry

end harry_did_not_get_an_A_l439_43962


namespace probability_of_region_C_l439_43955

theorem probability_of_region_C (P_A P_B P_C : ℚ) (hA : P_A = 1/3) (hB : P_B = 1/2) (hTotal : P_A + P_B + P_C = 1) : P_C = 1/6 := 
by
  sorry

end probability_of_region_C_l439_43955


namespace complement_of_A_in_U_l439_43956

theorem complement_of_A_in_U :
    ∀ (U A : Set ℕ),
    U = {1, 2, 3, 4} →
    A = {1, 3} →
    (U \ A) = {2, 4} :=
by
  intros U A hU hA
  rw [hU, hA]
  sorry

end complement_of_A_in_U_l439_43956


namespace branches_on_fourth_tree_l439_43977

theorem branches_on_fourth_tree :
  ∀ (height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot : ℕ),
    height_1 = 50 →
    branches_1 = 200 →
    height_2 = 40 →
    branches_2 = 180 →
    height_3 = 60 →
    branches_3 = 180 →
    height_4 = 34 →
    avg_branches_per_foot = 4 →
    (height_4 * avg_branches_per_foot = 136) :=
by
  intros height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot
  intros h1_eq_50 b1_eq_200 h2_eq_40 b2_eq_180 h3_eq_60 b3_eq_180 h4_eq_34 avg_eq_4
  -- We assume the conditions of the problem are correct, so add them to the context
  have height1 := h1_eq_50
  have branches1 := b1_eq_200
  have height2 := h2_eq_40
  have branches2 := b2_eq_180
  have height3 := h3_eq_60
  have branches3 := b3_eq_180
  have height4 := h4_eq_34
  have avg_branches := avg_eq_4
  -- Now prove the desired result
  sorry

end branches_on_fourth_tree_l439_43977


namespace white_line_longer_l439_43941

-- Define the lengths of the white and blue lines
def white_line_length : ℝ := 7.678934
def blue_line_length : ℝ := 3.33457689

-- State the main theorem
theorem white_line_longer :
  white_line_length - blue_line_length = 4.34435711 :=
by
  sorry

end white_line_longer_l439_43941


namespace eval_expression_l439_43900

def base8_to_base10 (n : Nat) : Nat :=
  2 * 8^2 + 4 * 8^1 + 5 * 8^0

def base4_to_base10 (n : Nat) : Nat :=
  1 * 4^1 + 5 * 4^0

def base5_to_base10 (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 2 * 5^0

def base6_to_base10 (n : Nat) : Nat :=
  3 * 6^1 + 2 * 6^0

theorem eval_expression : 
  base8_to_base10 245 / base4_to_base10 15 - base5_to_base10 232 / base6_to_base10 32 = 15 :=
by sorry

end eval_expression_l439_43900


namespace bathroom_area_is_eight_l439_43995

def bathroomArea (length width : ℕ) : ℕ :=
  length * width

theorem bathroom_area_is_eight : bathroomArea 4 2 = 8 := 
by
  -- Proof omitted.
  sorry

end bathroom_area_is_eight_l439_43995


namespace am_gm_inequality_l439_43987

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem am_gm_inequality : (a / b) + (b / c) + (c / a) ≥ 3 := by
  sorry

end am_gm_inequality_l439_43987


namespace laptop_final_price_l439_43984

theorem laptop_final_price (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  initial_price = 500 → first_discount = 10 → second_discount = 20 →
  (initial_price * (1 - first_discount / 100) * (1 - second_discount / 100)) = initial_price * 0.72 :=
by
  sorry

end laptop_final_price_l439_43984


namespace amelia_wins_probability_l439_43930

theorem amelia_wins_probability :
  let pA := 1 / 4
  let pB := 1 / 3
  let pC := 1 / 2
  let cycle_probability := (1 - pA) * (1 - pB) * (1 - pC)
  let infinite_series_sum := 1 / (1 - cycle_probability)
  let total_probability := pA * infinite_series_sum
  total_probability = 1 / 3 :=
by
  sorry

end amelia_wins_probability_l439_43930


namespace shelves_fit_l439_43965

-- Define the total space of the room for the library
def totalSpace : ℕ := 400

-- Define the space each bookshelf takes up
def spacePerBookshelf : ℕ := 80

-- Define the reserved space for desk and walking area
def reservedSpace : ℕ := 160

-- Define the space available for bookshelves
def availableSpace : ℕ := totalSpace - reservedSpace

-- Define the number of bookshelves that can fit in the available space
def numberOfBookshelves : ℕ := availableSpace / spacePerBookshelf

-- The theorem stating the number of bookshelves Jonas can fit in the room
theorem shelves_fit : numberOfBookshelves = 3 := by
  -- We can defer the proof as we only need the statement for now
  sorry

end shelves_fit_l439_43965


namespace arithmetic_sequence_sum_six_terms_l439_43963

noncomputable def sum_of_first_six_terms (a : ℤ) (d : ℤ) : ℤ :=
  let a1 := a
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let a5 := a4 + d
  let a6 := a5 + d
  a1 + a2 + a3 + a4 + a5 + a6

theorem arithmetic_sequence_sum_six_terms
  (a3 a4 a5 : ℤ)
  (h3 : a3 = 8)
  (h4 : a4 = 13)
  (h5 : a5 = 18)
  (d : ℤ) (a : ℤ)
  (h_d : d = a4 - a3)
  (h_a : a + 2 * d = 8) :
  sum_of_first_six_terms a d = 63 :=
by
  sorry

end arithmetic_sequence_sum_six_terms_l439_43963


namespace range_of_a_l439_43901

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ :=
  (m * x + n) / (x ^ 2 + 1)

example (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1) : 
  m = 2 ∧ n = 0 :=
sorry

theorem range_of_a (m n : ℝ) (h_odd : ∀ x, f x m n = -f (-x) m n) (h_f1 : f 1 m n = 1)
  (h_m : m = 2) (h_n : n = 0) {a : ℝ} : f (a-1) m n + f (a^2-1) m n < 0 ↔ 0 ≤ a ∧ a < 1 :=
sorry

end range_of_a_l439_43901


namespace max_value_of_f_on_interval_exists_x_eq_min_1_l439_43983

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 6*x + 13)

theorem max_value_of_f_on_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → f x ≤ 1 / 4 := sorry

theorem exists_x_eq_min_1 : 
  ∃ x, -2 ≤ x ∧ x ≤ 2 ∧ f x = 1 / 4 := sorry

end max_value_of_f_on_interval_exists_x_eq_min_1_l439_43983


namespace no_nat_fourfold_digit_move_l439_43925

theorem no_nat_fourfold_digit_move :
  ¬ ∃ (N : ℕ), ∃ (a : ℕ), ∃ (n : ℕ), ∃ (x : ℕ),
    (1 ≤ a ∧ a ≤ 9) ∧ 
    (N = a * 10^n + x) ∧ 
    (4 * N = 10 * x + a) :=
by
  sorry

end no_nat_fourfold_digit_move_l439_43925


namespace brenda_initial_peaches_l439_43932

variable (P : ℕ)

def brenda_conditions (P : ℕ) : Prop :=
  let fresh_peaches := P - 15
  (P > 15) ∧ (fresh_peaches * 60 = 100 * 150)

theorem brenda_initial_peaches : ∃ (P : ℕ), brenda_conditions P ∧ P = 250 :=
by
  sorry

end brenda_initial_peaches_l439_43932


namespace total_travel_time_is_19_hours_l439_43904

-- Define the distances and speeds as constants
def distance_WA_ID := 640
def speed_WA_ID := 80
def distance_ID_NV := 550
def speed_ID_NV := 50

-- Define the times based on the given distances and speeds
def time_WA_ID := distance_WA_ID / speed_WA_ID
def time_ID_NV := distance_ID_NV / speed_ID_NV

-- Define the total time
def total_time := time_WA_ID + time_ID_NV

-- Prove that the total travel time is 19 hours
theorem total_travel_time_is_19_hours : total_time = 19 := by
  sorry

end total_travel_time_is_19_hours_l439_43904


namespace min_S_n_condition_l439_43912

noncomputable def a_n (n : ℕ) : ℤ := -28 + 4 * (n - 1)

noncomputable def S_n (n : ℕ) : ℤ := n * (a_n 1 + a_n n) / 2

theorem min_S_n_condition : S_n 7 = S_n 8 ∧ (∀ m < 7, S_n m > S_n 7) ∧ (∀ m < 8, S_n m > S_n 8) := 
by
  sorry

end min_S_n_condition_l439_43912


namespace unique_fraction_condition_l439_43949

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end unique_fraction_condition_l439_43949


namespace first_player_winning_strategy_l439_43934

-- Definitions based on conditions
def initial_position (m n : ℕ) : ℕ × ℕ := (m - 1, n - 1)

-- Main theorem statement
theorem first_player_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (initial_position m n).fst ≠ (initial_position m n).snd ↔ m ≠ n :=
by
  sorry

end first_player_winning_strategy_l439_43934


namespace jogging_track_circumference_l439_43911

theorem jogging_track_circumference
  (Deepak_speed : ℝ)
  (Wife_speed : ℝ)
  (meet_time_minutes : ℝ)
  (H_deepak_speed : Deepak_speed = 4.5)
  (H_wife_speed : Wife_speed = 3.75)
  (H_meet_time_minutes : meet_time_minutes = 3.84) :
  let meet_time_hours := meet_time_minutes / 60
  let distance_deepak := Deepak_speed * meet_time_hours
  let distance_wife := Wife_speed * meet_time_hours
  let total_distance := distance_deepak + distance_wife
  let circumference := 2 * total_distance
  circumference = 1.056 :=
by
  sorry

end jogging_track_circumference_l439_43911


namespace simple_interest_sum_l439_43905

variable {P R : ℝ}

theorem simple_interest_sum :
  P * (R + 6) = P * R + 3000 → P = 500 :=
by
  intro h
  sorry

end simple_interest_sum_l439_43905


namespace person_A_number_is_35_l439_43972

theorem person_A_number_is_35
    (A B : ℕ)
    (h1 : A + B = 8)
    (h2 : 10 * B + A - (10 * A + B) = 18) :
    10 * A + B = 35 :=
by
    sorry

end person_A_number_is_35_l439_43972


namespace amusement_park_ticket_price_l439_43942

-- Conditions as definitions in Lean
def weekday_adult_ticket_cost : ℕ := 22
def weekday_children_ticket_cost : ℕ := 7
def weekend_adult_ticket_cost : ℕ := 25
def weekend_children_ticket_cost : ℕ := 10
def adult_discount_rate : ℕ := 20
def sales_tax_rate : ℕ := 10
def num_of_adults : ℕ := 2
def num_of_children : ℕ := 2

-- Correct Answer to be proved equivalent:
def expected_total_price := 66

-- Statement translating the problem to Lean proof obligation
theorem amusement_park_ticket_price :
  let cost_before_discount := (num_of_adults * weekend_adult_ticket_cost) + (num_of_children * weekend_children_ticket_cost)
  let discount := (num_of_adults * weekend_adult_ticket_cost) * adult_discount_rate / 100
  let subtotal := cost_before_discount - discount
  let sales_tax := subtotal * sales_tax_rate / 100
  let total_cost := subtotal + sales_tax
  total_cost = expected_total_price :=
by
  sorry

end amusement_park_ticket_price_l439_43942


namespace scientific_notation_of_169200000000_l439_43970

theorem scientific_notation_of_169200000000 : 169200000000 = 1.692 * 10^11 :=
by sorry

end scientific_notation_of_169200000000_l439_43970


namespace angles_congruence_mod_360_l439_43947

theorem angles_congruence_mod_360 (a b c d : ℤ) : 
  (a = 30) → (b = -30) → (c = 630) → (d = -630) →
  (b % 360 = 330 % 360) ∧ 
  (a % 360 ≠ 330 % 360) ∧ (c % 360 ≠ 330 % 360) ∧ (d % 360 ≠ 330 % 360) :=
by
  intros
  sorry

end angles_congruence_mod_360_l439_43947


namespace equal_real_roots_quadratic_l439_43967

theorem equal_real_roots_quadratic (k : ℝ) : (∀ x : ℝ, (x^2 + 2*x + k = 0)) → k = 1 :=
by
sorry

end equal_real_roots_quadratic_l439_43967


namespace no_solution_in_natural_numbers_l439_43920

theorem no_solution_in_natural_numbers (x y z : ℕ) : 
  (x / y : ℚ) + (y / z : ℚ) + (z / x : ℚ) ≠ 1 := 
by sorry

end no_solution_in_natural_numbers_l439_43920


namespace min_value_fraction_l439_43978

theorem min_value_fraction : ∃ (x : ℝ), (∀ y : ℝ, (y^2 + 9) / (Real.sqrt (y^2 + 5)) ≥ (9 * Real.sqrt 5) / 5)
  := sorry

end min_value_fraction_l439_43978


namespace sequence_difference_l439_43924

theorem sequence_difference (a : ℕ → ℤ) (h_rec : ∀ n : ℕ, a (n + 1) + a n = n) (h_a1 : a 1 = 2) :
  a 4 - a 2 = 1 :=
sorry

end sequence_difference_l439_43924
