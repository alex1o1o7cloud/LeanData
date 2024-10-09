import Mathlib

namespace frog_climbs_out_l459_45976

theorem frog_climbs_out (d climb slip : ℕ) (h : d = 20) (h_climb : climb = 3) (h_slip : slip = 2) :
  ∃ n : ℕ, n = 20 ∧ d ≤ n * (climb - slip) + climb :=
sorry

end frog_climbs_out_l459_45976


namespace total_spent_is_140_l459_45979

-- Define the original prices and discounts
def original_price_shoes : ℕ := 50
def original_price_dress : ℕ := 100
def discount_shoes : ℕ := 40
def discount_dress : ℕ := 20

-- Define the number of items purchased
def number_of_shoes : ℕ := 2
def number_of_dresses : ℕ := 1

-- Define the calculation of discounted prices
def discounted_price_shoes (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

def discounted_price_dress (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

-- Define the total cost calculation
def total_cost : ℕ :=
  discounted_price_shoes original_price_shoes discount_shoes number_of_shoes +
  discounted_price_dress original_price_dress discount_dress number_of_dresses

-- The theorem to prove
theorem total_spent_is_140 : total_cost = 140 := by
  sorry

end total_spent_is_140_l459_45979


namespace base_area_of_cylinder_l459_45987

variables (S : ℝ) (cylinder : Type)
variables (square_cross_section : cylinder → Prop) (area_square : cylinder → ℝ)
variables (base_area : cylinder → ℝ)

-- Assume that the cylinder has a square cross-section with a given area
axiom cross_section_square : ∀ c : cylinder, square_cross_section c → area_square c = 4 * S

-- Theorem stating the area of the base of the cylinder
theorem base_area_of_cylinder (c : cylinder) (h : square_cross_section c) : base_area c = π * S :=
by
  -- Proof omitted
  sorry

end base_area_of_cylinder_l459_45987


namespace wolf_nobel_laureates_l459_45959

theorem wolf_nobel_laureates (W N total W_prize N_prize N_noW N_W : ℕ)
  (h1 : W_prize = 31)
  (h2 : total = 50)
  (h3 : N_prize = 27)
  (h4 : N_noW + N_W = total - W_prize)
  (h5 : N_W = N_noW + 3)
  (h6 : N_prize = W + N_W) :
  W = 16 :=
by {
  sorry
}

end wolf_nobel_laureates_l459_45959


namespace no_such_function_exists_l459_45973

theorem no_such_function_exists (f : ℤ → ℤ) (h : ∀ m n : ℤ, f (m + f n) = f m - n) : false :=
sorry

end no_such_function_exists_l459_45973


namespace projection_problem_l459_45911

noncomputable def vector_proj (w v : ℝ × ℝ) : ℝ × ℝ := sorry -- assume this definition

variables (v w : ℝ × ℝ)

-- Given condition
axiom proj_v : vector_proj w v = ⟨4, 3⟩

-- Proof Statement
theorem projection_problem :
  vector_proj w (7 • v + 2 • w) = ⟨28, 21⟩ + 2 • w :=
sorry

end projection_problem_l459_45911


namespace janet_earned_1390_in_interest_l459_45941

def janets_total_interest (total_investment investment_at_10_rate investment_at_10_interest investment_at_1_rate remaining_investment remaining_investment_interest : ℝ) : ℝ :=
    investment_at_10_interest + remaining_investment_interest

theorem janet_earned_1390_in_interest :
  janets_total_interest 31000 12000 0.10 (12000 * 0.10) 0.01 (19000 * 0.01) = 1390 :=
by
  sorry

end janet_earned_1390_in_interest_l459_45941


namespace triangle_inequality_l459_45938

-- Define the side lengths of a triangle
variables {a b c : ℝ}

-- State the main theorem
theorem triangle_inequality :
  (a + b) * (b + c) * (c + a) ≥ 8 * (a + b - c) * (b + c - a) * (c + a - b) :=
sorry

end triangle_inequality_l459_45938


namespace simplify_and_evaluate_l459_45935

theorem simplify_and_evaluate :
  ∀ (a : ℚ), a = 3 → ((a - 1) / (a + 2) / ((a ^ 2 - 2 * a) / (a ^ 2 - 4)) - (a + 1) / a) = -2 / 3 :=
by
  intros a ha
  have : a = 3 := ha
  sorry

end simplify_and_evaluate_l459_45935


namespace geometric_fraction_l459_45971

noncomputable def a_n : ℕ → ℝ := sorry
axiom a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5
axiom geometric_sequence : ∀ n, a_n (n + 1) = a_n n * a_n (n + 1) / a_n (n - 1) 

theorem geometric_fraction (a_n : ℕ → ℝ) (a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5) :
  (a_n 13) / (a_n 9) = 9 :=
sorry

end geometric_fraction_l459_45971


namespace peanut_butter_candy_pieces_l459_45914

theorem peanut_butter_candy_pieces :
  ∀ (pb_candy grape_candy banana_candy : ℕ),
  pb_candy = 4 * grape_candy →
  grape_candy = banana_candy + 5 →
  banana_candy = 43 →
  pb_candy = 192 :=
by
  sorry

end peanut_butter_candy_pieces_l459_45914


namespace determine_phi_l459_45991

theorem determine_phi (phi : ℝ) (h : 0 < phi ∧ phi < π) :
  (∃ k : ℤ, phi = 2*k*π + (3*π/4)) :=
by
  sorry

end determine_phi_l459_45991


namespace chocolates_difference_l459_45980

-- Conditions
def Robert_chocolates : Nat := 13
def Nickel_chocolates : Nat := 4

-- Statement
theorem chocolates_difference : (Robert_chocolates - Nickel_chocolates) = 9 := by
  sorry

end chocolates_difference_l459_45980


namespace problem1_problem2_l459_45931

-- Problem 1: Prove that (x + y + z)² - (x + y - z)² = 4z(x + y) for x, y, z ∈ ℝ
theorem problem1 (x y z : ℝ) : (x + y + z)^2 - (x + y - z)^2 = 4 * z * (x + y) := 
sorry

-- Problem 2: Prove that (a + 2b)² - 2(a + 2b)(a - 2b) + (a - 2b)² = 16b² for a, b ∈ ℝ
theorem problem2 (a b : ℝ) : (a + 2 * b)^2 - 2 * (a + 2 * b) * (a - 2 * b) + (a - 2 * b)^2 = 16 * b^2 := 
sorry

end problem1_problem2_l459_45931


namespace sum_a_b_l459_45910

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 7 * b = 12) (h2 : 9 * a + 2 * b = 23) : a + b = 176 / 57 :=
by
  sorry

end sum_a_b_l459_45910


namespace tickets_spent_dunk_a_clown_booth_l459_45939

/-
The conditions given:
1. Tom bought 40 tickets.
2. Tom went on 3 rides.
3. Each ride costs 4 tickets.
-/
def total_tickets : ℕ := 40
def rides_count : ℕ := 3
def tickets_per_ride : ℕ := 4

/-
We aim to prove that Tom spent 28 tickets at the 'dunk a clown' booth.
-/
theorem tickets_spent_dunk_a_clown_booth :
  (total_tickets - rides_count * tickets_per_ride) = 28 :=
by
  sorry

end tickets_spent_dunk_a_clown_booth_l459_45939


namespace other_x_intercept_vertex_symmetric_l459_45969

theorem other_x_intercept_vertex_symmetric (a b c : ℝ)
  (h_vertex : ∀ x y : ℝ, (4, 10) = (x, y) → y = a * x^2 + b * x + c)
  (h_intercept : ∀ x : ℝ, (-1, 0) = (x, 0) → a * x^2 + b * x + c = 0) :
  a * 9^2 + b * 9 + c = 0 :=
sorry

end other_x_intercept_vertex_symmetric_l459_45969


namespace equal_numbers_in_sequence_l459_45997

theorem equal_numbers_in_sequence (a : ℕ → ℚ)
  (h : ∀ m n : ℕ, a m + a n = a (m * n)) : 
  ∃ i j : ℕ, i ≠ j ∧ a i = a j :=
sorry

end equal_numbers_in_sequence_l459_45997


namespace sum_factors_of_18_l459_45945

theorem sum_factors_of_18 : (1 + 18 + 2 + 9 + 3 + 6) = 39 := by
  sorry

end sum_factors_of_18_l459_45945


namespace lcm_180_616_l459_45912

theorem lcm_180_616 : Nat.lcm 180 616 = 27720 := 
by
  sorry

end lcm_180_616_l459_45912


namespace circle_equation_l459_45930

theorem circle_equation 
  (circle_eq : ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = (x - 3)^2 + (y - 2)^2) 
  (tangent_to_line : ∀ (x y : ℝ), (2*x - y + 5) = 0 → 
    (x = -2 ∧ y = 1))
  (passes_through_N : ∀ (x y : ℝ), (x = 3 ∧ y = 2)) :
  ∀ (x y : ℝ), x^2 + y^2 - 9*x + (9/2)*y - (55/2) = 0 := 
sorry

end circle_equation_l459_45930


namespace divisor_greater_than_2_l459_45906

theorem divisor_greater_than_2 (w n d : ℕ) (h1 : ∃ q1 : ℕ, w = d * q1 + 2)
                                       (h2 : n % 8 = 5)
                                       (h3 : n < 180) : 2 < d :=
sorry

end divisor_greater_than_2_l459_45906


namespace maximum_negative_roots_l459_45933

theorem maximum_negative_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
    (discriminant1 : b^2 - 4 * a * c ≥ 0)
    (discriminant2 : c^2 - 4 * b * a ≥ 0)
    (discriminant3 : a^2 - 4 * c * b ≥ 0) :
    ∃ n : ℕ, n ≤ 2 ∧ ∀ x ∈ {x | a * x^2 + b * x + c = 0 ∨ b * x^2 + c * x + a = 0 ∨ c * x^2 + a * x + b = 0}, x < 0 ↔ n = 2 := 
sorry

end maximum_negative_roots_l459_45933


namespace zhen_zhen_test_score_l459_45990

theorem zhen_zhen_test_score
  (avg1 avg2 : ℝ) (n m : ℝ)
  (h1 : avg1 = 88)
  (h2 : avg2 = 90)
  (h3 : n = 4)
  (h4 : m = 5) :
  avg2 * m - avg1 * n = 98 :=
by
  -- Given the hypotheses h1, h2, h3, and h4,
  -- we need to show that avg2 * m - avg1 * n = 98.
  sorry

end zhen_zhen_test_score_l459_45990


namespace total_coins_Zain_l459_45937

variable (quartersEmerie dimesEmerie nickelsEmerie : Nat)
variable (additionalCoins : Nat)

theorem total_coins_Zain (h_q : quartersEmerie = 6)
                         (h_d : dimesEmerie = 7)
                         (h_n : nickelsEmerie = 5)
                         (h_add : additionalCoins = 10) :
    let quartersZain := quartersEmerie + additionalCoins
    let dimesZain := dimesEmerie + additionalCoins
    let nickelsZain := nickelsEmerie + additionalCoins
    quartersZain + dimesZain + nickelsZain = 48 := by
  sorry

end total_coins_Zain_l459_45937


namespace son_l459_45905

theorem son's_age (F S : ℕ) (h1 : F + S = 75) (h2 : F = 8 * (S - (F - S))) : S = 27 :=
sorry

end son_l459_45905


namespace stella_annual_income_after_tax_l459_45966

-- Definitions of the conditions
def base_salary_per_month : ℝ := 3500
def bonuses : List ℝ := [1200, 600, 1500, 900, 1200]
def months_paid : ℝ := 10
def tax_rate : ℝ := 0.05

-- Calculations derived from the conditions
def total_base_salary : ℝ := base_salary_per_month * months_paid
def total_bonuses : ℝ := bonuses.sum
def total_income_before_tax : ℝ := total_base_salary + total_bonuses
def tax_deduction : ℝ := total_income_before_tax * tax_rate
def annual_income_after_tax : ℝ := total_income_before_tax - tax_deduction

-- The theorem to prove
theorem stella_annual_income_after_tax :
  annual_income_after_tax = 38380 := by
  sorry

end stella_annual_income_after_tax_l459_45966


namespace k_value_l459_45924

theorem k_value {x y k : ℝ} (h : ∃ c : ℝ, (x ^ 2 + k * x * y + 49 * y ^ 2) = c ^ 2) : k = 14 ∨ k = -14 :=
by sorry

end k_value_l459_45924


namespace mixed_tea_sale_price_l459_45917

noncomputable def sale_price_of_mixed_tea (weight1 weight2 weight3 price1 price2 price3 profit1 profit2 profit3 : ℝ) : ℝ :=
  let total_cost1 := weight1 * price1
  let total_cost2 := weight2 * price2
  let total_cost3 := weight3 * price3
  let total_profit1 := profit1 * total_cost1
  let total_profit2 := profit2 * total_cost2
  let total_profit3 := profit3 * total_cost3
  let selling_price1 := total_cost1 + total_profit1
  let selling_price2 := total_cost2 + total_profit2
  let selling_price3 := total_cost3 + total_profit3
  let total_selling_price := selling_price1 + selling_price2 + selling_price3
  let total_weight := weight1 + weight2 + weight3
  total_selling_price / total_weight

theorem mixed_tea_sale_price :
  sale_price_of_mixed_tea 120 45 35 30 40 60 0.50 0.30 0.25 = 51.825 :=
by
  sorry

end mixed_tea_sale_price_l459_45917


namespace measure_of_angle_R_l459_45900

variable (S T A R : ℝ) -- Represent the angles as real numbers.

-- The conditions given in the problem.
axiom angles_congruent : S = T ∧ T = A ∧ A = R
axiom angle_A_equals_angle_S : A = S

-- Statement: Prove that the measure of angle R is 108 degrees.
theorem measure_of_angle_R : R = 108 :=
by
  sorry

end measure_of_angle_R_l459_45900


namespace middle_number_in_consecutive_nat_sum_squares_equals_2030_l459_45903

theorem middle_number_in_consecutive_nat_sum_squares_equals_2030 
  (n : ℕ)
  (h1 : (n - 1)^2 + n^2 + (n + 1)^2 = 2030)
  (h2 : (n^3 - n^2) % 7 = 0)
  : n = 26 := 
sorry

end middle_number_in_consecutive_nat_sum_squares_equals_2030_l459_45903


namespace sandwiches_lunch_monday_l459_45921

-- Define the conditions
variables (L : ℕ) 
variables (sandwiches_monday sandwiches_tuesday : ℕ)
variables (h1 : sandwiches_monday = L + 2 * L)
variables (h2 : sandwiches_tuesday = 1)

-- Define the fact that he ate 8 more sandwiches on Monday compared to Tuesday.
variables (h3 : sandwiches_monday = sandwiches_tuesday + 8)

theorem sandwiches_lunch_monday : L = 3 := 
by
  -- We need to prove L = 3 given the conditions (h1, h2, h3)
  -- Here is where the necessary proof would be constructed
  -- This placeholder indicates a proof needs to be inserted here
  sorry

end sandwiches_lunch_monday_l459_45921


namespace TripleApplicationOfF_l459_45962

def f (N : ℝ) : ℝ := 0.7 * N + 2

theorem TripleApplicationOfF :
  f (f (f 40)) = 18.1 :=
  sorry

end TripleApplicationOfF_l459_45962


namespace magnitude_BC_eq_sqrt29_l459_45957

noncomputable def A : (ℝ × ℝ) := (2, -1)
noncomputable def C : (ℝ × ℝ) := (0, 2)
noncomputable def AB : (ℝ × ℝ) := (3, 5)

theorem magnitude_BC_eq_sqrt29
    (A : ℝ × ℝ := (2, -1))
    (C : ℝ × ℝ := (0, 2))
    (AB : ℝ × ℝ := (3, 5)) :
    ∃ B : ℝ × ℝ, (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = 29 := 
by
  sorry

end magnitude_BC_eq_sqrt29_l459_45957


namespace event_B_more_likely_l459_45928

theorem event_B_more_likely (A B : Set (ℕ → ℕ)) 
  (hA : ∀ ω, ω ∈ A ↔ ∃ i j, i ≠ j ∧ ω i = ω j)
  (hB : ∀ ω, ω ∈ B ↔ ∀ i j, i ≠ j → ω i ≠ ω j) :
  ∃ prob_A prob_B : ℚ, prob_A = 4 / 9 ∧ prob_B = 5 / 9 ∧ prob_B > prob_A :=
by
  sorry

end event_B_more_likely_l459_45928


namespace new_average_age_after_person_leaves_l459_45964

theorem new_average_age_after_person_leaves (avg_age : ℕ) (n : ℕ) (leaving_age : ℕ) (remaining_count : ℕ) :
  ((n * avg_age - leaving_age) / remaining_count) = 33 :=
by
  -- Given conditions
  let avg_age := 30
  let n := 5
  let leaving_age := 18
  let remaining_count := n - 1
  -- Conclusion
  sorry

end new_average_age_after_person_leaves_l459_45964


namespace least_number_to_make_divisible_l459_45999

def least_common_multiple (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem least_number_to_make_divisible (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : 
  least_common_multiple a b = 77 → 
  (n % least_common_multiple a b) = 40 →
  c = (least_common_multiple a b - (n % least_common_multiple a b)) →
  c = 37 :=
by
sorry

end least_number_to_make_divisible_l459_45999


namespace rectangle_other_side_l459_45993

theorem rectangle_other_side
  (a b : ℝ)
  (Area : ℝ := 12 * a ^ 2 - 6 * a * b)
  (side1 : ℝ := 3 * a)
  (side2 : ℝ := Area / side1) :
  side2 = 4 * a - 2 * b :=
by
  sorry

end rectangle_other_side_l459_45993


namespace triangle_angle_and_area_l459_45913

theorem triangle_angle_and_area (a b c A B C : ℝ)
  (h₁ : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A))
  (h₂ : 0 < C ∧ C < Real.pi)
  (h₃ : c = 2 * Real.sqrt 3) :
  C = Real.pi / 3 ∧ 0 ≤ (1 / 2) * a * b * Real.sin C ∧ (1 / 2) * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
by
  sorry

end triangle_angle_and_area_l459_45913


namespace total_cards_needed_l459_45955

def red_card_credits := 3
def blue_card_credits := 5
def total_credits := 84
def red_cards := 8

theorem total_cards_needed :
  red_card_credits * red_cards + blue_card_credits * (total_credits - red_card_credits * red_cards) / blue_card_credits = 20 := by
  sorry

end total_cards_needed_l459_45955


namespace remi_water_consumption_proof_l459_45952

-- Definitions for the conditions
def daily_consumption (bottle_volume : ℕ) (refills_per_day : ℕ) : ℕ :=
  bottle_volume * refills_per_day

def total_spillage (spill1 : ℕ) (spill2 : ℕ) : ℕ :=
  spill1 + spill2

def total_consumption (daily : ℕ) (days : ℕ) (spill : ℕ) : ℕ :=
  (daily * days) - spill

-- Theorem proving the number of days d
theorem remi_water_consumption_proof (bottle_volume : ℕ) (refills_per_day : ℕ)
  (spill1 spill2 total_water : ℕ) (d : ℕ)
  (h1 : bottle_volume = 20) (h2 : refills_per_day = 3)
  (h3 : spill1 = 5) (h4 : spill2 = 8)
  (h5 : total_water = 407) :
  total_consumption (daily_consumption bottle_volume refills_per_day) d
    (total_spillage spill1 spill2) = total_water → d = 7 := 
by
  -- Assuming the hypotheses to show the equality
  intro h
  have daily := h1 ▸ h2 ▸ 20 * 3 -- ⇒ daily = 60
  have spillage := h3 ▸ h4 ▸ 5 + 8 -- ⇒ spillage = 13
  rw [daily_consumption, total_spillage, h5] at h
  rw [h1, h2, h3, h4] at h -- Substitute conditions in the hypothesis
  sorry -- place a placeholder for the actual proof

end remi_water_consumption_proof_l459_45952


namespace find_xyz_sum_l459_45982

theorem find_xyz_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 12)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 37) :
  x * y + y * z + z * x = 20 :=
sorry

end find_xyz_sum_l459_45982


namespace Cally_colored_shirts_l459_45983

theorem Cally_colored_shirts (C : ℕ) (hcally : 10 + 7 + 6 = 23) (hdanny : 6 + 8 + 10 + 6 = 30) (htotal : 23 + 30 + C = 58) : 
  C = 5 := 
by
  sorry

end Cally_colored_shirts_l459_45983


namespace r_daily_earnings_l459_45932

def earnings_problem (P Q R : ℝ) : Prop :=
  (9 * (P + Q + R) = 1890) ∧ 
  (5 * (P + R) = 600) ∧ 
  (7 * (Q + R) = 910)

theorem r_daily_earnings :
  ∃ P Q R : ℝ, earnings_problem P Q R ∧ R = 40 := sorry

end r_daily_earnings_l459_45932


namespace number_of_valid_arrangements_l459_45923

def total_permutations (n : ℕ) : ℕ := n.factorial

def valid_permutations (total : ℕ) (block : ℕ) (specific_restriction : ℕ) : ℕ :=
  total - specific_restriction

theorem number_of_valid_arrangements : valid_permutations (total_permutations 5) 48 24 = 96 :=
by
  sorry

end number_of_valid_arrangements_l459_45923


namespace weight_of_3_moles_HBrO3_is_386_73_l459_45972

noncomputable def H_weight : ℝ := 1.01
noncomputable def Br_weight : ℝ := 79.90
noncomputable def O_weight : ℝ := 16.00
noncomputable def HBrO3_weight : ℝ := H_weight + Br_weight + 3 * O_weight
noncomputable def weight_of_3_moles_of_HBrO3 : ℝ := 3 * HBrO3_weight

theorem weight_of_3_moles_HBrO3_is_386_73 : weight_of_3_moles_of_HBrO3 = 386.73 := by
  sorry

end weight_of_3_moles_HBrO3_is_386_73_l459_45972


namespace range_of_h_l459_45943

def f (x : ℝ) : ℝ := 4 * x - 3
def h (x : ℝ) : ℝ := f (f (f x))

theorem range_of_h : 
  (∀ x, -1 ≤ x ∧ x ≤ 3 → -127 ≤ h x ∧ h x ≤ 129) :=
by
  sorry

end range_of_h_l459_45943


namespace number_of_women_l459_45995

-- Definitions for the given conditions
variables (m w : ℝ)
variable (x : ℝ)

-- Conditions
def cond1 : Prop := 3 * m + 8 * w = 6 * m + 2 * w
def cond2 : Prop := 4 * m + x * w = 0.9285714285714286 * (3 * m + 8 * w)

-- Theorem to prove the number of women in the third group (x)
theorem number_of_women (h1 : cond1 m w) (h2 : cond2 m w x) : x = 5 :=
sorry

end number_of_women_l459_45995


namespace age_difference_l459_45918

theorem age_difference (A B : ℕ) (h1 : B = 38) (h2 : A + 10 = 2 * (B - 10)) : A - B = 8 :=
by
  sorry

end age_difference_l459_45918


namespace grapes_purchased_l459_45968

variable (G : ℕ)
variable (rate_grapes : ℕ) (qty_mangoes : ℕ) (rate_mangoes : ℕ) (total_paid : ℕ)

theorem grapes_purchased (h1 : rate_grapes = 70)
                        (h2 : qty_mangoes = 9)
                        (h3 : rate_mangoes = 55)
                        (h4 : total_paid = 1055) :
                        70 * G + 9 * 55 = 1055 → G = 8 :=
by
  sorry

end grapes_purchased_l459_45968


namespace solution_set_f_x_minus_1_lt_0_l459_45958

noncomputable def f (x : ℝ) : ℝ :=
if h : x ≥ 0 then x - 1 else -x - 1

theorem solution_set_f_x_minus_1_lt_0 :
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_f_x_minus_1_lt_0_l459_45958


namespace smallest_m_for_integral_solutions_l459_45919

theorem smallest_m_for_integral_solutions :
  ∃ (m : ℕ), (∀ (x : ℤ), (12 * x^2 - m * x + 504 = 0 → ∃ (p q : ℤ), p + q = m / 12 ∧ p * q = 42)) ∧
  m = 156 := by
sorry

end smallest_m_for_integral_solutions_l459_45919


namespace valentino_farm_birds_total_l459_45907

theorem valentino_farm_birds_total :
  let chickens := 200
  let ducks := 2 * chickens
  let turkeys := 3 * ducks
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof here
  sorry

end valentino_farm_birds_total_l459_45907


namespace last_number_written_on_sheet_l459_45970

/-- The given problem is to find the last number written on a sheet with specific rules. 
Given:
- The sheet has dimensions of 100 characters in width and 100 characters in height.
- Numbers are written successively with a space between each number.
- If the end of a line is reached, the next number continues at the beginning of the next line.

We need to prove that the last number written on the sheet is 2220.
-/
theorem last_number_written_on_sheet :
  ∃ (n : ℕ), n = 2220 ∧ 
    let width := 100
    let height := 100
    let sheet_size := width * height
    let write_number size occupied_space := occupied_space + size + 1 
    ∃ (numbers : ℕ → ℕ) (space_per_number : ℕ → ℕ),
      ( ∀ i, space_per_number i = if numbers i < 10 then 2 else if numbers i < 100 then 3 else if numbers i < 1000 then 4 else 5 ) ∧
      ∃ (current_space : ℕ), 
        (current_space ≤ sheet_size) ∧
        (∀ i, current_space = write_number (space_per_number i) current_space ) :=
sorry

end last_number_written_on_sheet_l459_45970


namespace trains_cross_time_l459_45998

noncomputable def time_to_cross_trains : ℝ :=
  let l1 := 220 -- length of the first train in meters
  let s1 := 120 * (5 / 18) -- speed of the first train in meters per second
  let l2 := 280.04 -- length of the second train in meters
  let s2 := 80 * (5 / 18) -- speed of the second train in meters per second
  let relative_speed := s1 + s2 -- relative speed in meters per second
  let total_length := l1 + l2 -- total length to be crossed in meters
  total_length / relative_speed -- time in seconds

theorem trains_cross_time :
  abs (time_to_cross_trains - 9) < 0.01 := -- Allowing a small error to account for approximation
by
  sorry

end trains_cross_time_l459_45998


namespace problem_I_problem_II_problem_III_l459_45953

-- Problem (I)
noncomputable def f (x a : ℝ) := Real.log x - a * (x - 1)
noncomputable def tangent_line (x a : ℝ) := (1 - a) * (x - 1)

theorem problem_I (a : ℝ) :
  ∃ y, tangent_line y a = f 1 a / (1 : ℝ) :=
sorry

-- Problem (II)
theorem problem_II (a : ℝ) (h : a ≥ 1 / 2) :
  ∀ x ≥ 1, f x a ≤ Real.log x / (x + 1) :=
sorry

-- Problem (III)
theorem problem_III (a : ℝ) :
  ∀ x ≥ 1, Real.exp (x - 1) - a * (x ^ 2 - x) ≥ x * f x a + 1 :=
sorry

end problem_I_problem_II_problem_III_l459_45953


namespace inequality_correct_l459_45984

theorem inequality_correct (a b : ℝ) (h : a - |b| > 0) : a + b > 0 :=
sorry

end inequality_correct_l459_45984


namespace slip_3_5_in_F_l459_45902

def slips := [1.5, 2, 2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5]

def cup_sum (x : List ℝ) := List.sum x

def slips_dist (A B C D E F : List ℝ) : Prop :=
  cup_sum A + cup_sum B + cup_sum C + cup_sum D + cup_sum E + cup_sum F = 50 ∧ 
  cup_sum A = 6 ∧ cup_sum B = 8 ∧ cup_sum C = 10 ∧ cup_sum D = 12 ∧ cup_sum E = 14 ∧ cup_sum F = 16 ∧
  2.5 ∈ B ∧ 2.5 ∈ D ∧ 4 ∈ C

def contains_slip (c : List ℝ) (v : ℝ) : Prop := v ∈ c

theorem slip_3_5_in_F (A B C D E F : List ℝ) (h : slips_dist A B C D E F) : 
  contains_slip F 3.5 :=
sorry

end slip_3_5_in_F_l459_45902


namespace amy_owes_thirty_l459_45974

variable (A D : ℝ)

theorem amy_owes_thirty
  (total_pledged remaining_owed sally_carl_owe derek_half_amys_owes : ℝ)
  (h1 : total_pledged = 285)
  (h2 : remaining_owed = 400 - total_pledged)
  (h3 : sally_carl_owe = 35 + 35)
  (h4 : derek_half_amys_owes = A / 2)
  (h5 : remaining_owed - sally_carl_owe = 45)
  (h6 : 45 = A + (A / 2)) :
  A = 30 :=
by
  -- Proof steps skipped
  sorry

end amy_owes_thirty_l459_45974


namespace find_a_l459_45978

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x >= 0 then a^x else a^(-x)

theorem find_a (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a)
(h_ge : ∀ x : ℝ, x >= 0 → f x a = a ^ x)
(h_a_gt_1 : a > 1)
(h_sol : ∀ x : ℝ, f x a ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2) :
a = 2 :=
sorry

end find_a_l459_45978


namespace distance_covered_by_wheel_l459_45947

noncomputable def pi_num : ℝ := 3.14159

noncomputable def wheel_diameter : ℝ := 14

noncomputable def number_of_revolutions : ℝ := 33.03002729754322

noncomputable def circumference : ℝ := pi_num * wheel_diameter

noncomputable def calculated_distance : ℝ := circumference * number_of_revolutions

theorem distance_covered_by_wheel : 
  calculated_distance = 1452.996 :=
sorry

end distance_covered_by_wheel_l459_45947


namespace x_power_2023_zero_or_neg_two_l459_45985

variable {x : ℂ} -- Assuming x is a complex number to handle general roots of unity.

theorem x_power_2023_zero_or_neg_two 
  (h1 : (x - 1) * (x + 1) = x^2 - 1)
  (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
  (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
  (pattern : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) :
  x^2023 - 1 = 0 ∨ x^2023 - 1 = -2 :=
by
  sorry

end x_power_2023_zero_or_neg_two_l459_45985


namespace benny_missed_games_l459_45996

theorem benny_missed_games (total_games attended_games missed_games : ℕ)
  (H1 : total_games = 39)
  (H2 : attended_games = 14)
  (H3 : missed_games = total_games - attended_games) :
  missed_games = 25 :=
by
  sorry

end benny_missed_games_l459_45996


namespace determine_K_class_comparison_l459_45967

variables (a b : ℕ) -- number of students in classes A and B respectively
variable (K : ℕ) -- amount that each A student would pay if they covered all cost

-- Conditions from the problem statement
def first_event_total (a b : ℕ) := 5 * a + 3 * b
def second_event_total (a b : ℕ) := 4 * a + 6 * b
def total_balance (a b K : ℕ) := 9 * (a + b) = K * (a + b)

-- Questions to be answered
theorem determine_K : total_balance a b K → K = 9 :=
by
  sorry

theorem class_comparison (a b : ℕ) : 5 * a + 3 * b = 4 * a + 6 * b → b > a :=
by
  sorry

end determine_K_class_comparison_l459_45967


namespace jihye_wallet_total_l459_45948

-- Declare the amounts
def notes_amount : Nat := 2 * 1000
def coins_amount : Nat := 560

-- Theorem statement asserting the total amount
theorem jihye_wallet_total : notes_amount + coins_amount = 2560 := by
  sorry

end jihye_wallet_total_l459_45948


namespace find_missing_number_l459_45909

theorem find_missing_number (x : ℤ) (h : 10010 - 12 * x * 2 = 9938) : x = 3 :=
by
  sorry

end find_missing_number_l459_45909


namespace total_annual_donation_l459_45946

-- Defining the conditions provided in the problem
def monthly_donation : ℕ := 1707
def months_in_year : ℕ := 12

-- Stating the theorem that answers the question
theorem total_annual_donation : monthly_donation * months_in_year = 20484 := 
by
  -- The proof is omitted for brevity
  sorry

end total_annual_donation_l459_45946


namespace second_closest_location_l459_45940
-- Import all necessary modules from the math library

-- Define the given distances (conditions)
def distance_library : ℝ := 1.912 * 1000  -- distance in meters
def distance_park : ℝ := 876              -- distance in meters
def distance_clothing_store : ℝ := 1.054 * 1000  -- distance in meters

-- State the proof problem
theorem second_closest_location :
  (distance_library = 1912) →
  (distance_park = 876) →
  (distance_clothing_store = 1054) →
  (distance_clothing_store = 1054) :=
by
  intros h1 h2 h3
  -- sorry to skip the proof
  sorry

end second_closest_location_l459_45940


namespace find_fraction_l459_45920

theorem find_fraction (N : ℕ) (hN : N = 90) (f : ℚ)
  (h : 3 + (1/2) * f * (1/5) * N = (1/15) * N) :
  f = 1/3 :=
by {
  sorry
}

end find_fraction_l459_45920


namespace train_length_l459_45934

theorem train_length (L : ℝ) 
    (cross_bridge : ∀ (t_bridge : ℝ), t_bridge = 10 → L + 200 = t_bridge * (L / 5))
    (cross_lamp_post : ∀ (t_lamp_post : ℝ), t_lamp_post = 5 → L = t_lamp_post * (L / 5)) :
  L = 200 := 
by 
  -- sorry is used to skip the proof part
  sorry

end train_length_l459_45934


namespace two_digit_integer_divides_491_remainder_59_l459_45954

theorem two_digit_integer_divides_491_remainder_59 :
  ∃ n Q : ℕ, (n = 10 * x + y) ∧ (0 < x) ∧ (x ≤ 9) ∧ (0 ≤ y) ∧ (y ≤ 9) ∧ (491 = n * Q + 59) ∧ (n = 72) :=
by
  sorry

end two_digit_integer_divides_491_remainder_59_l459_45954


namespace average_value_continuous_l459_45927

noncomputable def average_value (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (1 / (b - a)) * ∫ x in a..b, f x

theorem average_value_continuous (f : ℝ → ℝ) (a b : ℝ) (h : ContinuousOn f (Set.Icc a b)) :
  (average_value f a b) = (1 / (b - a)) * (∫ x in a..b, f x) :=
by
  sorry

end average_value_continuous_l459_45927


namespace find_B_sin_squared_sum_range_l459_45994

-- Define the angles and vectors
variables {A B C : ℝ}
variables (m n : ℝ × ℝ)
variables (α : ℝ)

-- Basic triangle angle sum condition
axiom angle_sum : A + B + C = Real.pi

-- Define vectors as per the problem statement
axiom vector_m : m = (Real.sin B, 1 - Real.cos B)
axiom vector_n : n = (2, 0)

-- The angle between vectors m and n is π/3
axiom angle_between_vectors : α = Real.pi / 3
axiom angle_condition : Real.cos α = (2 * Real.sin B + 0 * (1 - Real.cos B)) / 
                                     (Real.sqrt (Real.sin B ^ 2 + (1 - Real.cos B) ^ 2) * 2)

theorem find_B : B = 2 * Real.pi / 3 := 
sorry

-- Conditions for range of sin^2 A + sin^2 C
axiom range_condition : (0 < A ∧ A < Real.pi / 3) 
                     ∧ (0 < C ∧ C < Real.pi / 3)
                     ∧ (A + C = Real.pi / 3)

theorem sin_squared_sum_range : (Real.sin A) ^ 2 + (Real.sin C) ^ 2 ∈ Set.Ico (1 / 2) 1 := 
sorry

end find_B_sin_squared_sum_range_l459_45994


namespace find_square_sum_l459_45922

theorem find_square_sum :
  ∃ a b c : ℕ, a = 2494651 ∧ b = 1385287 ∧ c = 9406087 ∧ (a + b + c = 3645^2) :=
by
  have h1 : 2494651 + 1385287 + 9406087 = 13286025 := by norm_num
  have h2 : 3645^2 = 13286025 := by norm_num
  exact ⟨2494651, 1385287, 9406087, rfl, rfl, rfl, h2⟩

end find_square_sum_l459_45922


namespace mean_age_is_10_l459_45942

def ages : List ℤ := [7, 7, 7, 14, 15]

theorem mean_age_is_10 : (List.sum ages : ℤ) / (ages.length : ℤ) = 10 := by
-- sorry placeholder for the actual proof
sorry

end mean_age_is_10_l459_45942


namespace seventh_oblong_is_56_l459_45961

def oblong (n : ℕ) : ℕ := n * (n + 1)

theorem seventh_oblong_is_56 : oblong 7 = 56 := by
  sorry

end seventh_oblong_is_56_l459_45961


namespace least_possible_value_of_b_l459_45949

theorem least_possible_value_of_b (a b : ℕ) 
  (ha : ∃ p, (∀ q, p ∣ q ↔ q = 1 ∨ q = p ∨ q = p*p ∨ q = a))
  (hb : ∃ k, (∀ l, k ∣ l ↔ (l = 1 ∨ l = b)))
  (hdiv : a ∣ b) : 
  b = 12 :=
sorry

end least_possible_value_of_b_l459_45949


namespace common_ratio_arith_geo_sequence_l459_45965

theorem common_ratio_arith_geo_sequence (a : ℕ → ℝ) (d : ℝ) (q : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_geo : (a 1 + 2) * q = a 5 + 5) 
  (h_geo' : (a 5 + 5) * q = a 9 + 8) :
  q = 1 :=
by
  sorry

end common_ratio_arith_geo_sequence_l459_45965


namespace range_of_b_l459_45951

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem range_of_b (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (f x1 = b) ∧ (f x2 = b) ∧ (f x3 = b))
  ↔ (-4 / 3 < b ∧ b < 28 / 3) :=
by
  sorry

end range_of_b_l459_45951


namespace find_prices_max_basketballs_l459_45992

-- Define price of basketballs and soccer balls
def basketball_price : ℕ := 80
def soccer_ball_price : ℕ := 50

-- Define the equations given in the problem
theorem find_prices (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 310)
  (h2 : 5 * x + 2 * y = 500) : 
  x = basketball_price ∧ y = soccer_ball_price :=
sorry

-- Define the maximum number of basketballs given the cost constraints
theorem max_basketballs (m : ℕ)
  (htotal : m + (60 - m) = 60)
  (hcost : 80 * m + 50 * (60 - m) ≤ 4000) : 
  m ≤ 33 :=
sorry

end find_prices_max_basketballs_l459_45992


namespace initial_momentum_eq_2Fx_div_v_l459_45960

variable (m v F x t : ℝ)
variable (H_initial_conditions : v ≠ 0)
variable (H_force : F > 0)
variable (H_distance : x > 0)
variable (H_time : t > 0)
variable (H_stopping_distance : x = (m * v^2) / (2 * F))
variable (H_stopping_time : t = (m * v) / F)

theorem initial_momentum_eq_2Fx_div_v :
  m * v = (2 * F * x) / v :=
sorry

end initial_momentum_eq_2Fx_div_v_l459_45960


namespace line_through_point_parallel_l459_45981

theorem line_through_point_parallel (x y : ℝ) (h₁ : 2 * 2 + 4 * 3 + x = 0) (h₂ : x = -16) (h₃ : y = 8) :
  2 * x + 4 * y - 3 = 0 → x + 2 * y - 8 = 0 :=
by
  intro h₄
  sorry

end line_through_point_parallel_l459_45981


namespace original_savings_l459_45988

/-- Linda spent 3/4 of her savings on furniture and the rest on a TV costing $210. 
    What were her original savings? -/
theorem original_savings (S : ℝ) (h1 : S * (1/4) = 210) : S = 840 :=
by
  sorry

end original_savings_l459_45988


namespace conic_section_is_hyperbola_l459_45916

theorem conic_section_is_hyperbola :
  ∀ (x y : ℝ), x^2 - 16 * y^2 - 8 * x + 16 * y + 32 = 0 → 
               (∃ h k a b : ℝ, h = 4 ∧ k = 0.5 ∧ a = b ∧ a^2 = 2 ∧ b^2 = 2) :=
by
  sorry

end conic_section_is_hyperbola_l459_45916


namespace connie_tickets_l459_45908

variable (T : ℕ)

theorem connie_tickets (h : T = T / 2 + 10 + 15) : T = 50 :=
by 
sorry

end connie_tickets_l459_45908


namespace abs_a_gt_neg_b_l459_45944

variable {a b : ℝ}

theorem abs_a_gt_neg_b (h : a < b ∧ b < 0) : |a| > -b :=
by
  sorry

end abs_a_gt_neg_b_l459_45944


namespace gallons_10_percent_milk_needed_l459_45925

-- Definitions based on conditions
def amount_of_butterfat (x : ℝ) : ℝ := 0.10 * x
def total_butterfat_in_existing_milk : ℝ := 4
def final_butterfat (x : ℝ) : ℝ := amount_of_butterfat x + total_butterfat_in_existing_milk
def total_milk (x : ℝ) : ℝ := x + 8
def desired_butterfat (x : ℝ) : ℝ := 0.20 * total_milk x

-- Lean proof statement
theorem gallons_10_percent_milk_needed (x : ℝ) : final_butterfat x = desired_butterfat x → x = 24 :=
by
  intros h
  sorry

end gallons_10_percent_milk_needed_l459_45925


namespace domain_of_function_l459_45963

theorem domain_of_function :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x ≠ 0} = {x : ℝ | -2 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_function_l459_45963


namespace unit_prices_purchasing_schemes_maximize_profit_l459_45989

-- Define the conditions and variables
def purchase_price_system (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 240) ∧ (3 * x + 4 * y = 340)

def possible_schemes (a b : ℕ) : Prop :=
  (a + b = 200) ∧ (60 * a + 40 * b ≤ 10440) ∧ (a ≥ 3 * b / 2)

def max_profit (x y : ℝ) (a b : ℕ) : ℝ :=
  (x - 60) * a + (y - 40) * b

-- Prove the unit prices are $60 and $40
theorem unit_prices : ∃ x y, purchase_price_system x y ∧ x = 60 ∧ y = 40 :=
by
  sorry

-- Prove the possible purchasing schemes
theorem purchasing_schemes : ∀ a b, possible_schemes a b → 
  (a = 120 ∧ b = 80 ∨ a = 121 ∧ b = 79 ∨ a = 122 ∧ b = 78) :=
by
  sorry

-- Prove the maximum profit is 3610 with the purchase amounts (122, 78)
theorem maximize_profit :
  ∃ (a b : ℕ), max_profit 80 55 a b = 3610 ∧ purchase_price_system 60 40 ∧ possible_schemes a b ∧ a = 122 ∧ b = 78 :=
by
  sorry

end unit_prices_purchasing_schemes_maximize_profit_l459_45989


namespace people_happy_correct_l459_45986

-- Define the size and happiness percentage of an institution.
variables (size : ℕ) (happiness_percentage : ℚ)

-- Assume the size is between 100 and 200.
axiom size_range : 100 ≤ size ∧ size ≤ 200

-- Assume the happiness percentage is between 0.6 and 0.95.
axiom happiness_percentage_range : 0.6 ≤ happiness_percentage ∧ happiness_percentage ≤ 0.95

-- Define the number of people made happy at an institution.
def people_made_happy (size : ℕ) (happiness_percentage : ℚ) : ℚ := 
  size * happiness_percentage

-- Theorem stating that the number of people made happy is as expected.
theorem people_happy_correct : 
  ∀ (size : ℕ) (happiness_percentage : ℚ), 
  100 ≤ size → size ≤ 200 → 
  0.6 ≤ happiness_percentage → happiness_percentage ≤ 0.95 → 
  people_made_happy size happiness_percentage = size * happiness_percentage := 
by 
  intros size happiness_percentage hsize1 hsize2 hperc1 hperc2
  unfold people_made_happy
  sorry

end people_happy_correct_l459_45986


namespace second_percentage_increase_l459_45936

theorem second_percentage_increase (P : ℝ) (x : ℝ) :
  1.25 * P * (1 + x / 100) = 1.625 * P ↔ x = 30 :=
by
  sorry

end second_percentage_increase_l459_45936


namespace solve_equation_l459_45956

-- Define the equation as a function of y
def equation (y : ℝ) : ℝ :=
  y^4 - 20 * y + 1

-- State the theorem that y = -1 satisfies the equation.
theorem solve_equation : equation (-1) = 22 := 
  sorry

end solve_equation_l459_45956


namespace other_root_l459_45915

theorem other_root (m : ℝ) (x : ℝ) (hx : 3 * x ^ 2 + m * x - 7 = 0) (root1 : x = 1) :
  ∃ y : ℝ, 3 * y ^ 2 + m * y - 7 = 0 ∧ y = -7 / 3 :=
by
  sorry

end other_root_l459_45915


namespace evaluate_special_operation_l459_45929

-- Define the operation @
def special_operation (a b : ℕ) : ℚ := (a * b) / (a - b)

-- State the theorem
theorem evaluate_special_operation : special_operation 6 3 = 6 := by
  sorry

end evaluate_special_operation_l459_45929


namespace greatest_k_inequality_l459_45901

theorem greatest_k_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ( ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
    (a / b + b / c + c / a - 3) ≥ k * (a / (b + c) + b / (c + a) + c / (a + b) - 3 / 2) ) ↔ k = 1 := 
sorry

end greatest_k_inequality_l459_45901


namespace find_fraction_l459_45950

theorem find_fraction
  (F : ℚ) (m : ℕ) 
  (h1 : F^m * (1 / 4)^2 = 1 / 10^4)
  (h2 : m = 4) : 
  F = 1 / 5 :=
by
  sorry

end find_fraction_l459_45950


namespace helga_shoe_pairs_l459_45926

theorem helga_shoe_pairs
  (first_store_pairs: ℕ) 
  (second_store_pairs: ℕ) 
  (third_store_pairs: ℕ)
  (fourth_store_pairs: ℕ)
  (h1: first_store_pairs = 7)
  (h2: second_store_pairs = first_store_pairs + 2)
  (h3: third_store_pairs = 0)
  (h4: fourth_store_pairs = 2 * (first_store_pairs + second_store_pairs + third_store_pairs))
  : first_store_pairs + second_store_pairs + third_store_pairs + fourth_store_pairs = 48 := 
by
  sorry

end helga_shoe_pairs_l459_45926


namespace ratio_right_to_left_l459_45904

theorem ratio_right_to_left (L C R : ℕ) (hL : L = 12) (hC : C = L + 2) (hTotal : L + C + R = 50) :
  R / L = 2 :=
by
  sorry

end ratio_right_to_left_l459_45904


namespace floor_ceil_sum_l459_45977

theorem floor_ceil_sum (x : ℝ) (h : Int.floor x + Int.ceil x = 7) : x ∈ { x : ℝ | 3 < x ∧ x < 4 } ∪ {3.5} :=
sorry

end floor_ceil_sum_l459_45977


namespace distinct_arrangements_on_3x3_grid_l459_45975

def is_valid_position (pos : ℤ × ℤ) : Prop :=
  0 ≤ pos.1 ∧ pos.1 < 3 ∧ 0 ≤ pos.2 ∧ pos.2 < 3

def rotations_equiv (pos1 pos2 : ℤ × ℤ) : Prop :=
  pos1 = pos2 ∨ pos1 = (2 - pos2.2, pos2.1) ∨ pos1 = (2 - pos2.1, 2 - pos2.2) ∨ pos1 = (pos2.2, 2 - pos2.1)

def distinct_positions_count (grid_size : ℕ) : ℕ :=
  10  -- given from the problem solution

theorem distinct_arrangements_on_3x3_grid : distinct_positions_count 3 = 10 := sorry

end distinct_arrangements_on_3x3_grid_l459_45975
