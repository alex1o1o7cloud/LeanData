import Mathlib

namespace proof_problem_l1112_111242

noncomputable def log2 (n : ℝ) : ℝ := Real.log n / Real.log 2

theorem proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/2 * log2 x + 1/3 * log2 y = 1) : x^3 * y^2 = 64 := 
sorry 

end proof_problem_l1112_111242


namespace factor_and_divisor_statements_l1112_111255

theorem factor_and_divisor_statements :
  (∃ n : ℕ, 25 = 5 * n) ∧
  ((∃ n : ℕ, 209 = 19 * n) ∧ ¬ (∃ n : ℕ, 63 = 19 * n)) ∧
  (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end factor_and_divisor_statements_l1112_111255


namespace will_can_buy_correct_amount_of_toys_l1112_111207

-- Define the initial conditions as constants
def initial_amount : Int := 57
def amount_spent : Int := 27
def cost_per_toy : Int := 6

-- Lemma stating the problem to prove.
theorem will_can_buy_correct_amount_of_toys : (initial_amount - amount_spent) / cost_per_toy = 5 :=
by
  sorry

end will_can_buy_correct_amount_of_toys_l1112_111207


namespace total_enemies_l1112_111272

theorem total_enemies (E : ℕ) (h : 8 * (E - 2) = 40) : E = 7 := sorry

end total_enemies_l1112_111272


namespace sqrt_mixed_number_simplification_l1112_111289

theorem sqrt_mixed_number_simplification :
  Real.sqrt (7 + 9 / 16) = 11 / 4 :=
by
  sorry

end sqrt_mixed_number_simplification_l1112_111289


namespace find_pairs_l1112_111265

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l1112_111265


namespace combined_cost_price_correct_l1112_111293

def face_value_A : ℝ := 100
def discount_A : ℝ := 0.02
def face_value_B : ℝ := 100
def premium_B : ℝ := 0.015
def brokerage : ℝ := 0.002

def purchase_price_A := face_value_A * (1 - discount_A)
def brokerage_fee_A := purchase_price_A * brokerage
def total_cost_price_A := purchase_price_A + brokerage_fee_A

def purchase_price_B := face_value_B * (1 + premium_B)
def brokerage_fee_B := purchase_price_B * brokerage
def total_cost_price_B := purchase_price_B + brokerage_fee_B

def combined_cost_price := total_cost_price_A + total_cost_price_B

theorem combined_cost_price_correct :
  combined_cost_price = 199.899 :=
by
  sorry

end combined_cost_price_correct_l1112_111293


namespace binomial_12_10_eq_66_l1112_111296

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_eq_66_l1112_111296


namespace triangle_third_side_length_l1112_111297

theorem triangle_third_side_length
  (AC BC : ℝ)
  (h_a h_b h_c : ℝ)
  (half_sum_heights_eq : (h_a + h_b) / 2 = h_c) :
  AC = 6 → BC = 3 → AB = 4 :=
by
  sorry

end triangle_third_side_length_l1112_111297


namespace angle_sum_around_point_l1112_111231

theorem angle_sum_around_point (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) : 
    x + y + 130 = 360 → x + y = 230 := by
  sorry

end angle_sum_around_point_l1112_111231


namespace k_squared_minus_3k_minus_4_l1112_111232

theorem k_squared_minus_3k_minus_4 (a b c d k : ℚ)
  (h₁ : (2 * a) / (b + c + d) = k)
  (h₂ : (2 * b) / (a + c + d) = k)
  (h₃ : (2 * c) / (a + b + d) = k)
  (h₄ : (2 * d) / (a + b + c) = k) :
  k^2 - 3 * k - 4 = -50 / 9 ∨ k^2 - 3 * k - 4 = 6 :=
  sorry

end k_squared_minus_3k_minus_4_l1112_111232


namespace construction_costs_correct_l1112_111214

structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPerThousand : ℕ
  tileCostPerTile : ℕ
  landRequired : ℕ
  bricksRequired : ℕ
  tilesRequired : ℕ

noncomputable def totalConstructionCost (cc : ConstructionCosts) : ℕ :=
  let landCost := cc.landRequired * cc.landCostPerSqMeter
  let brickCost := (cc.bricksRequired / 1000) * cc.brickCostPerThousand
  let tileCost := cc.tilesRequired * cc.tileCostPerTile
  landCost + brickCost + tileCost

theorem construction_costs_correct (cc : ConstructionCosts)
  (h1 : cc.landCostPerSqMeter = 50)
  (h2 : cc.brickCostPerThousand = 100)
  (h3 : cc.tileCostPerTile = 10)
  (h4 : cc.landRequired = 2000)
  (h5 : cc.bricksRequired = 10000)
  (h6 : cc.tilesRequired = 500) :
  totalConstructionCost cc = 106000 := 
  by 
    sorry

end construction_costs_correct_l1112_111214


namespace Martha_reading_challenge_l1112_111246

theorem Martha_reading_challenge :
  ∀ x : ℕ,
  (12 + 18 + 14 + 20 + 11 + 13 + 19 + 15 + 17 + x) / 10 = 15 ↔ x = 11 :=
by sorry

end Martha_reading_challenge_l1112_111246


namespace people_in_line_l1112_111273

theorem people_in_line (initially_in_line : ℕ) (left_line : ℕ) (after_joined_line : ℕ) 
  (h1 : initially_in_line = 12) (h2 : left_line = 10) (h3 : after_joined_line = 17) : 
  initially_in_line - left_line + 15 = after_joined_line := by
  sorry

end people_in_line_l1112_111273


namespace area_difference_quarter_circles_l1112_111292

theorem area_difference_quarter_circles :
  let r1 := 28
  let r2 := 14
  let pi := (22 / 7)
  let quarter_area_big := (1 / 4) * pi * r1^2
  let quarter_area_small := (1 / 4) * pi * r2^2
  let rectangle_area := r1 * r2
  (quarter_area_big - (quarter_area_small + rectangle_area)) = 70 := by
  -- Placeholder for the proof
  sorry

end area_difference_quarter_circles_l1112_111292


namespace piggy_bank_after_8_weeks_l1112_111258

-- Define initial amount in the piggy bank
def initial_amount : ℝ := 43

-- Define weekly allowance amount
def weekly_allowance : ℝ := 10

-- Define fraction of allowance Jack saves
def saving_fraction : ℝ := 0.5

-- Define number of weeks
def number_of_weeks : ℕ := 8

-- Define weekly savings amount
def weekly_savings : ℝ := saving_fraction * weekly_allowance

-- Define total savings after a given number of weeks
def total_savings (weeks : ℕ) : ℝ := weeks * weekly_savings

-- Define the final amount in the piggy bank after a given number of weeks
def final_amount (weeks : ℕ) : ℝ := initial_amount + total_savings weeks

-- Theorem: Prove that final amount in piggy bank after 8 weeks is $83
theorem piggy_bank_after_8_weeks : final_amount number_of_weeks = 83 := by
  sorry

end piggy_bank_after_8_weeks_l1112_111258


namespace waiter_total_customers_l1112_111245

def numCustomers (T : ℕ) (totalTips : ℕ) (tipPerCustomer : ℕ) (numNoTipCustomers : ℕ) : ℕ :=
  T + numNoTipCustomers

theorem waiter_total_customers
  (T : ℕ)
  (h1 : 3 * T = 6)
  (numNoTipCustomers : ℕ := 5)
  (total := numCustomers T 6 3 numNoTipCustomers) :
  total = 7 := by
  sorry

end waiter_total_customers_l1112_111245


namespace option_B_correct_l1112_111253

theorem option_B_correct (x m : ℕ) : (x^3)^m / (x^m)^2 = x^m := sorry

end option_B_correct_l1112_111253


namespace difference_in_ages_27_l1112_111244

def conditions (a b : ℕ) : Prop :=
  10 * b + a = (1 / 2) * (10 * a + b) + 6 ∧
  10 * a + b + 2 = 5 * (10 * b + a - 4)

theorem difference_in_ages_27 {a b : ℕ} (h : conditions a b) :
  (10 * a + b) - (10 * b + a) = 27 :=
sorry

end difference_in_ages_27_l1112_111244


namespace neznaika_mistake_l1112_111269

-- Let's define the conditions
variables {X A Y M E O U : ℕ} -- Represents distinct digits

-- Ascending order of the numbers
variables (XA AY AX OY EM EY MU : ℕ)
  (h1 : XA < AY)
  (h2 : AY < AX)
  (h3 : AX < OY)
  (h4 : OY < EM)
  (h5 : EM < EY)
  (h6 : EY < MU)

-- Identical digits replaced with the same letters
variables (h7 : XA = 10 * X + A)
  (h8 : AY = 10 * A + Y)
  (h9 : AX = 10 * A + X)
  (h10 : OY = 10 * O + Y)
  (h11 : EM = 10 * E + M)
  (h12 : EY = 10 * E + Y)
  (h13 : MU = 10 * M + U)

-- Each letter represents a different digit
variables (h_distinct : X ≠ A ∧ X ≠ Y ∧ X ≠ M ∧ X ≠ E ∧ X ≠ O ∧ X ≠ U ∧
                       A ≠ Y ∧ A ≠ M ∧ A ≠ E ∧ A ≠ O ∧ A ≠ U ∧
                       Y ≠ M ∧ Y ≠ E ∧ Y ≠ O ∧ Y ≠ U ∧
                       M ≠ E ∧ M ≠ O ∧ M ≠ U ∧
                       E ≠ O ∧ E ≠ U ∧
                       O ≠ U)

-- Prove Neznaika made a mistake
theorem neznaika_mistake : false :=
by
  -- Here we'll reach a contradiction, proving false.
  sorry

end neznaika_mistake_l1112_111269


namespace snakes_hiding_l1112_111250

/-- The statement that given the total number of snakes and the number of snakes not hiding,
we can determine the number of snakes hiding. -/
theorem snakes_hiding (total_snakes : ℕ) (snakes_not_hiding : ℕ) (h1 : total_snakes = 95) (h2 : snakes_not_hiding = 31) :
  total_snakes - snakes_not_hiding = 64 :=
by {
  sorry
}

end snakes_hiding_l1112_111250


namespace area_of_rectangular_plot_l1112_111276

-- Defining the breadth
def breadth : ℕ := 26

-- Defining the length as thrice the breadth
def length : ℕ := 3 * breadth

-- Defining the area as the product of length and breadth
def area : ℕ := length * breadth

-- The theorem stating the problem to prove
theorem area_of_rectangular_plot : area = 2028 := by
  -- Initial proof step skipped
  sorry

end area_of_rectangular_plot_l1112_111276


namespace total_selling_price_l1112_111237

theorem total_selling_price
  (meters_cloth : ℕ)
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter)
  (total_selling_price : ℕ := selling_price_per_meter * meters_cloth)
  (h_mc : meters_cloth = 75)
  (h_ppm : profit_per_meter = 15)
  (h_cppm : cost_price_per_meter = 51)
  (h_spm : selling_price_per_meter = 66)
  (h_tsp : total_selling_price = 4950) : 
  total_selling_price = 4950 := 
  by
  -- Skipping the actual proof
  trivial

end total_selling_price_l1112_111237


namespace james_older_brother_age_l1112_111212

def johnAge : ℕ := 39

def ageCondition (johnAge : ℕ) (jamesAgeIn6 : ℕ) : Prop :=
  johnAge - 3 = 2 * jamesAgeIn6

def jamesOlderBrother (james : ℕ) : ℕ :=
  james + 4

theorem james_older_brother_age (johnAge jamesOlderBrotherAge : ℕ) (james : ℕ) :
  johnAge = 39 →
  (johnAge - 3 = 2 * (james + 6)) →
  jamesOlderBrotherAge = jamesOlderBrother james →
  jamesOlderBrotherAge = 16 :=
by
  sorry

end james_older_brother_age_l1112_111212


namespace xy_range_l1112_111213

theorem xy_range (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
    (h_eqn : x + 3 * y + 2 / x + 4 / y = 10) :
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
  sorry

end xy_range_l1112_111213


namespace length_GH_of_tetrahedron_l1112_111200

noncomputable def tetrahedron_edge_length : ℕ := 24

theorem length_GH_of_tetrahedron
  (a b c d e f : ℕ)
  (h1 : a = 8) 
  (h2 : b = 16) 
  (h3 : c = 24) 
  (h4 : d = 35) 
  (h5 : e = 45) 
  (h6 : f = 55)
  (hEF : f = 55)
  (hEGF : e + b > f)
  (hEHG: e + c > a ∧ e + c > d) 
  (hFHG : b + c > a ∧ b + f > c ∧ c + a > b):
   tetrahedron_edge_length = c := 
sorry

end length_GH_of_tetrahedron_l1112_111200


namespace num_children_l1112_111224

-- Defining the conditions
def num_adults : Nat := 10
def price_adult_ticket : Nat := 8
def total_bill : Nat := 124
def price_child_ticket : Nat := 4

-- Statement to prove: Number of children
theorem num_children (num_adults : Nat) (price_adult_ticket : Nat) (total_bill : Nat) (price_child_ticket : Nat) : Nat :=
  let cost_adults := num_adults * price_adult_ticket
  let cost_child := total_bill - cost_adults
  cost_child / price_child_ticket

example : num_children 10 8 124 4 = 11 := sorry

end num_children_l1112_111224


namespace sum_of_medians_powers_l1112_111259

noncomputable def median_length_squared (a b c : ℝ) : ℝ :=
  (a^2 + b^2 - c^2) / 4

noncomputable def sum_of_fourth_powers_of_medians (a b c : ℝ) : ℝ :=
  let mAD := (median_length_squared a b c)^2
  let mBE := (median_length_squared b c a)^2
  let mCF := (median_length_squared c a b)^2
  mAD^2 + mBE^2 + mCF^2

theorem sum_of_medians_powers :
  sum_of_fourth_powers_of_medians 13 14 15 = 7644.25 :=
by
  sorry

end sum_of_medians_powers_l1112_111259


namespace price_reduction_equation_l1112_111280

variable (x : ℝ)

theorem price_reduction_equation 
    (original_price : ℝ)
    (final_price : ℝ)
    (two_reductions : original_price * (1 - x) ^ 2 = final_price) :
    100 * (1 - x) ^ 2 = 81 :=
by
  sorry

end price_reduction_equation_l1112_111280


namespace perfect_square_m_value_l1112_111223

theorem perfect_square_m_value (y m : ℤ) (h : ∃ k : ℤ, y^2 - 8 * y + m = (y - k)^2) : m = 16 :=
sorry

end perfect_square_m_value_l1112_111223


namespace true_statements_count_is_two_l1112_111236

def original_proposition (a : ℝ) : Prop :=
  a < 0 → ∃ x : ℝ, x^2 + x + a = 0

def contrapositive (a : ℝ) : Prop :=
  ¬ (∃ x : ℝ, x^2 + x + a = 0) → a ≥ 0

def converse (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 + x + a = 0) → a < 0

def negation (a : ℝ) : Prop :=
  a < 0 → ¬ ∃ x : ℝ, x^2 + x + a = 0

-- Prove that there are exactly 2 true statements among the four propositions: 
-- original_proposition, contrapositive, converse, and negation.

theorem true_statements_count_is_two : 
  ∀ (a : ℝ), original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a) → 
  (original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a)) ↔ (2 = 2) := 
by
  sorry

end true_statements_count_is_two_l1112_111236


namespace larry_channel_reduction_l1112_111294

theorem larry_channel_reduction
  (initial_channels new_channels final_channels sports_package supreme_sports_package channels_at_end : ℕ)
  (h_initial : initial_channels = 150)
  (h_adjustment : new_channels = initial_channels - 20 + 12)
  (h_sports : sports_package = 8)
  (h_supreme_sports : supreme_sports_package = 7)
  (h_channels_at_end : channels_at_end = 147)
  (h_final : final_channels = channels_at_end - sports_package - supreme_sports_package) :
  initial_channels - 20 + 12 - final_channels = 10 := 
sorry

end larry_channel_reduction_l1112_111294


namespace probability_no_neighbouring_same_color_l1112_111215

-- Given conditions
def red_beads : ℕ := 4
def white_beads : ℕ := 2
def blue_beads : ℕ := 2
def total_beads : ℕ := red_beads + white_beads + blue_beads

-- Total permutations
def total_orderings : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- Probability calculation proof
theorem probability_no_neighbouring_same_color : (30 / 420 : ℚ) = (1 / 14 : ℚ) :=
by
  -- proof steps
  sorry

end probability_no_neighbouring_same_color_l1112_111215


namespace no_sphinx_tiling_l1112_111227

def equilateral_triangle_tiling_problem (side_length : ℕ) (pointing_up : ℕ) (pointing_down : ℕ) : Prop :=
  let total_triangles := side_length * side_length
  pointing_up + pointing_down = total_triangles ∧ 
  total_triangles = 36 ∧
  pointing_down = 1 + 2 + 3 + 4 + 5 ∧
  pointing_up = 1 + 2 + 3 + 4 + 5 + 6 ∧
  (pointing_up % 2 = 1) ∧
  (pointing_down % 2 = 1) ∧
  (2 * pointing_up + 4 * pointing_down ≠ total_triangles ∧ 4 * pointing_up + 2 * pointing_down ≠ total_triangles)

theorem no_sphinx_tiling : ¬equilateral_triangle_tiling_problem 6 21 15 :=
by
  sorry

end no_sphinx_tiling_l1112_111227


namespace largest_of_three_consecutive_integers_l1112_111288

theorem largest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : x + 2 = 7 := 
sorry

end largest_of_three_consecutive_integers_l1112_111288


namespace original_price_l1112_111205

theorem original_price (P : ℝ) (h : P * 0.5 = 1200) : P = 2400 := 
by
  sorry

end original_price_l1112_111205


namespace largest_prime_factor_of_4752_l1112_111216

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬ (m ∣ n)

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p)

def pf_4752 : ℕ := 4752

theorem largest_prime_factor_of_4752 : largest_prime_factor pf_4752 11 :=
  by
  sorry

end largest_prime_factor_of_4752_l1112_111216


namespace cars_meeting_time_l1112_111240

def problem_statement (V_A V_B V_C V_D : ℝ) :=
  (V_A ≠ V_B) ∧ (V_A ≠ V_C) ∧ (V_A ≠ V_D) ∧
  (V_B ≠ V_C) ∧ (V_B ≠ V_D) ∧ (V_C ≠ V_D) ∧
  (V_A + V_C = V_B + V_D) ∧
  (53 * (V_A - V_B) / 46 = 7) ∧
  (53 * (V_D - V_C) / 46 = 7)

theorem cars_meeting_time (V_A V_B V_C V_D : ℝ) (h : problem_statement V_A V_B V_C V_D) : 
  ∃ t : ℝ, t = 53 := 
sorry

end cars_meeting_time_l1112_111240


namespace option_D_not_necessarily_true_l1112_111298

variable {a b c : ℝ}

theorem option_D_not_necessarily_true 
  (h1 : c < b)
  (h2 : b < a)
  (h3 : a * c < 0) : ¬((c * b^2 < a * b^2) ↔ (b ≠ 0 ∨ b = 0 ∧ (c * b^2 < a * b^2))) := 
sorry

end option_D_not_necessarily_true_l1112_111298


namespace range_of_x_l1112_111283

section
  variable (f : ℝ → ℝ)

  -- Conditions:
  -- 1. f is an even function
  def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

  -- 2. f is monotonically increasing on [0, +∞)
  def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

  -- Range of x
  def in_range (x : ℝ) : Prop := (1 : ℝ) / 3 < x ∧ x < (2 : ℝ) / 3

  -- Main statement
  theorem range_of_x (f_is_even : is_even f) (f_is_mono : mono_increasing_on_nonneg f) :
    ∀ x, f (2 * x - 1) < f ((1 : ℝ) / 3) ↔ in_range x := 
  by
    sorry
end

end range_of_x_l1112_111283


namespace combined_age_l1112_111274

variable (m y o : ℕ)

noncomputable def younger_brother_age := 5

noncomputable def older_brother_age_based_on_younger := 3 * younger_brother_age

noncomputable def older_brother_age_based_on_michael (m : ℕ) := 1 + 2 * (m - 1)

theorem combined_age (m y o : ℕ) (h1 : y = younger_brother_age) (h2 : o = older_brother_age_based_on_younger) (h3 : o = older_brother_age_based_on_michael m) :
  y + o + m = 28 := by
  sorry

end combined_age_l1112_111274


namespace trigonometric_identity_l1112_111286

theorem trigonometric_identity
  (θ : ℝ) 
  (h_tan : Real.tan θ = 3) :
  (1 - Real.cos θ) / (Real.sin θ) - (Real.sin θ) / (1 + (Real.cos θ)^2) = (11 * Real.sqrt 10 - 101) / 33 := 
by
  sorry

end trigonometric_identity_l1112_111286


namespace number_of_a_values_l1112_111228

theorem number_of_a_values (a : ℝ) :
  (∃ x : ℝ, y = x + 2*a ∧ y = x^3 - 3*a*x + a^3) → a = 0 :=
by
  sorry

end number_of_a_values_l1112_111228


namespace math_problem_l1112_111206

variable {f : ℝ → ℝ}
variable {g : ℝ → ℝ}

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def odd_function (g : ℝ → ℝ) := ∀ x : ℝ, g x = -g (-x)

theorem math_problem
  (hf_even : even_function f)
  (hf_0 : f 0 = 1)
  (hg_odd : odd_function g)
  (hgf : ∀ x : ℝ, g x = f (x - 1)) :
  f 2011 + f 2012 + f 2013 = 1 := sorry

end math_problem_l1112_111206


namespace calculate_speed_l1112_111218

theorem calculate_speed :
  ∀ (distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph : ℚ),
  distance_ft = 200 →
  time_sec = 2 →
  miles_per_ft = 1 / 5280 →
  hours_per_sec = 1 / 3600 →
  approx_speed_mph = 68.1818181818 →
  (distance_ft * miles_per_ft) / (time_sec * hours_per_sec) = approx_speed_mph :=
by
  intros distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph
  intro h_distance_eq h_time_eq h_miles_eq h_hours_eq h_speed_eq
  sorry

end calculate_speed_l1112_111218


namespace chandra_pairings_l1112_111230

variable (bowls : ℕ) (glasses : ℕ)

theorem chandra_pairings : 
  bowls = 5 → 
  glasses = 4 → 
  bowls * glasses = 20 :=
by intros; 
    sorry

end chandra_pairings_l1112_111230


namespace range_of_a_l1112_111287

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x < 3) → (4 * a * x + 4 * (a - 3)) ≤ 0) ↔ (0 ≤ a ∧ a ≤ 3 / 4) :=
by
  sorry

end range_of_a_l1112_111287


namespace x1x2_lt_one_l1112_111225

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  |exp x - exp 1| + exp x + a * x

theorem x1x2_lt_one (a x1 x2 : ℝ) 
  (ha : a < -exp 1) 
  (hzero1 : f a x1 = 0) 
  (hzero2 : f a x2 = 0) 
  (h_order : x1 < x2) : x1 * x2 < 1 := 
sorry

end x1x2_lt_one_l1112_111225


namespace polygon_sides_eq_seven_l1112_111201

theorem polygon_sides_eq_seven (n : ℕ) (h : 2 * n - (n * (n - 3)) / 2 = 0) : n = 7 :=
by sorry

end polygon_sides_eq_seven_l1112_111201


namespace max_alpha_for_2_alpha_divides_3n_plus_1_l1112_111248

theorem max_alpha_for_2_alpha_divides_3n_plus_1 (n : ℕ) (hn : n > 0) : ∃ α : ℕ, (2 ^ α ∣ (3 ^ n + 1)) ∧ ¬ (2 ^ (α + 1) ∣ (3 ^ n + 1)) ∧ α = 1 :=
by
  sorry

end max_alpha_for_2_alpha_divides_3n_plus_1_l1112_111248


namespace rowing_distance_l1112_111295

theorem rowing_distance (D : ℝ) : 
  (D / 14 + D / 2 = 120) → D = 210 := by
  sorry

end rowing_distance_l1112_111295


namespace solve_system_of_equations_l1112_111257

theorem solve_system_of_equations : ∃ x y : ℤ, 3 * x - 2 * y = 6 ∧ 2 * x + 3 * y = 17 ∧ x = 4 ∧ y = 3 :=
by
  sorry

end solve_system_of_equations_l1112_111257


namespace counting_unit_difference_l1112_111219

-- Definitions based on conditions
def magnitude_equality : Prop := 75 = 75.0
def counting_unit_75 : Nat := 1
def counting_unit_75_0 : Nat := 1 / 10

-- Proof problem stating that 75 and 75.0 do not have the same counting units.
theorem counting_unit_difference : 
  ¬ (counting_unit_75 = counting_unit_75_0) :=
by sorry

end counting_unit_difference_l1112_111219


namespace proof_of_diagonal_length_l1112_111267

noncomputable def length_of_diagonal (d : ℝ) : Prop :=
  d^2 = 325 ∧ 17^2 + 36 = 325

theorem proof_of_diagonal_length (d : ℝ) : length_of_diagonal d → d = 5 * Real.sqrt 13 :=
by
  intro h
  sorry

end proof_of_diagonal_length_l1112_111267


namespace sqrt_one_sixty_four_l1112_111221

theorem sqrt_one_sixty_four : Real.sqrt (1 / 64) = 1 / 8 :=
sorry

end sqrt_one_sixty_four_l1112_111221


namespace complement_union_l1112_111241

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 4, 5}

theorem complement_union :
  (U \ A) ∪ (U \ B) = {1, 2, 3, 6} := 
by 
  sorry

end complement_union_l1112_111241


namespace intersection_is_correct_complement_is_correct_l1112_111290

open Set

variable {U : Set ℝ} (A B : Set ℝ)

-- Define the universal set U
def U_def : Set ℝ := { x | 1 < x ∧ x < 7 }

-- Define set A
def A_def : Set ℝ := { x | 2 ≤ x ∧ x < 5 }

-- Define set B using the simplified condition from the inequality
def B_def : Set ℝ := { x | x ≥ 3 }

-- Proof statement that A ∩ B is as specified
theorem intersection_is_correct :
  (A_def ∩ B_def) = { x : ℝ | 3 ≤ x ∧ x < 5 } := by
  sorry

-- Proof statement for the complement of A relative to U
theorem complement_is_correct :
  (U_def \ A_def) = { x : ℝ | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7) } := by
  sorry

end intersection_is_correct_complement_is_correct_l1112_111290


namespace value_of_expression_l1112_111263

theorem value_of_expression : 30 - 5^2 = 5 := by
  sorry

end value_of_expression_l1112_111263


namespace jake_watching_hours_l1112_111279

theorem jake_watching_hours
    (monday_hours : ℕ := 12) -- Half of 24 hours in a day is 12 hours for Monday
    (wednesday_hours : ℕ := 6) -- A quarter of 24 hours in a day is 6 hours for Wednesday
    (friday_hours : ℕ := 19) -- Jake watched 19 hours on Friday
    (total_hours : ℕ := 52) -- The entire show is 52 hours long
    (T : ℕ) -- To find the total number of hours on Tuesday
    (h : monday_hours + T + wednesday_hours + (monday_hours + T + wednesday_hours) / 2 + friday_hours = total_hours) :
    T = 4 := sorry

end jake_watching_hours_l1112_111279


namespace allan_plums_l1112_111271

theorem allan_plums (A : ℕ) (h1 : 7 - A = 3) : A = 4 :=
sorry

end allan_plums_l1112_111271


namespace probability_multiple_of_100_is_zero_l1112_111208

def singleDigitMultiplesOf5 : Set ℕ := {5}
def primeNumbersLessThan50 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
def isMultipleOf100 (n : ℕ) : Prop := 100 ∣ n

theorem probability_multiple_of_100_is_zero :
  (∀ m ∈ singleDigitMultiplesOf5, ∀ p ∈ primeNumbersLessThan50, ¬ isMultipleOf100 (m * p)) →
  r = 0 :=
sorry

end probability_multiple_of_100_is_zero_l1112_111208


namespace product_of_two_equal_numbers_l1112_111270

theorem product_of_two_equal_numbers :
  ∃ (x : ℕ), (5 * 20 = 12 + 22 + 16 + 2 * x) ∧ (x * x = 625) :=
by
  sorry

end product_of_two_equal_numbers_l1112_111270


namespace evaluate_fraction_l1112_111284

variable (a b x : ℝ)
variable (h1 : a ≠ b)
variable (h2 : b ≠ 0)
variable (h3 : x = a / b)

theorem evaluate_fraction :
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end evaluate_fraction_l1112_111284


namespace gcd_9240_12240_33720_l1112_111210

theorem gcd_9240_12240_33720 : Nat.gcd (Nat.gcd 9240 12240) 33720 = 240 := by
  sorry

end gcd_9240_12240_33720_l1112_111210


namespace reinforcement_correct_l1112_111262

-- Conditions
def initial_men : ℕ := 2000
def initial_days : ℕ := 54
def days_before_reinforcement : ℕ := 18
def days_after_reinforcement : ℕ := 20

-- Define the remaining provisions after 18 days
def provisions_left : ℕ := initial_men * (initial_days - days_before_reinforcement)

-- Define reinforcement
def reinforcement : ℕ := 
  sorry -- placeholder for the definition

-- Theorem to prove
theorem reinforcement_correct :
  reinforcement = 1600 :=
by
  -- Use the given conditions to derive the reinforcement value
  let total_provision := initial_men * initial_days
  let remaining_provision := provisions_left
  let men_after_reinforcement := initial_men + reinforcement
  have h := remaining_provision = men_after_reinforcement * days_after_reinforcement
  sorry -- placeholder for the proof

end reinforcement_correct_l1112_111262


namespace jill_water_stored_l1112_111291

theorem jill_water_stored (n : ℕ) (h : n = 24) : 
  8 * (1 / 4 : ℝ) + 8 * (1 / 2 : ℝ) + 8 * 1 = 14 :=
by
  sorry

end jill_water_stored_l1112_111291


namespace min_value_of_expression_l1112_111277

theorem min_value_of_expression (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (hpar : ∀ x y : ℝ, 2 * x + (n - 1) * y - 2 = 0 → ∃ c : ℝ, mx + ny + c = 0) :
  2 * m + n = 9 :=
by
  sorry

end min_value_of_expression_l1112_111277


namespace calculate_pens_l1112_111260

theorem calculate_pens (P : ℕ) (Students : ℕ) (Pencils : ℕ) (h1 : Students = 40) (h2 : Pencils = 920) (h3 : ∃ k : ℕ, Pencils = Students * k) 
(h4 : ∃ m : ℕ, P = Students * m) : ∃ k : ℕ, P = 40 * k := by
  sorry

end calculate_pens_l1112_111260


namespace part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l1112_111203

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem part1_smallest_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := 
sorry

theorem part1_monotonic_interval :
  ∀ k : ℤ, ∀ x, (k * Real.pi - Real.pi / 3) ≤ x ∧ x ≤ (k * Real.pi + Real.pi / 6) →
  ∃ (b a c : ℝ) (A : ℝ), b + c = 2 * a ∧ 2 * A = A + Real.pi / 3 ∧ 
  f A = 1 / 2 ∧ a = 3 * Real.sqrt 2 := 
sorry

theorem part2_value_of_a :
  ∀ (A b c : ℝ), 
  (∃ (a : ℝ), 2 * a = b + c ∧ 
  f A = 1 / 2 ∧ 
  b * c = 18 ∧ 
  Real.cos A = 1 / 2) → 
  ∃ a, a = 3 * Real.sqrt 2 := 
sorry

end part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l1112_111203


namespace digits_base8_2015_l1112_111275

theorem digits_base8_2015 : ∃ n : Nat, (8^n ≤ 2015 ∧ 2015 < 8^(n+1)) ∧ n + 1 = 4 := 
by 
  sorry

end digits_base8_2015_l1112_111275


namespace crystal_final_segment_distance_l1112_111217

theorem crystal_final_segment_distance :
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2 -- as nx, ny
  let southwest_component := southwest_distance / Real.sqrt 2 -- as sx, sy
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  Real.sqrt (net_north^2 + net_west^2) = 2 * Real.sqrt 3 :=
by
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2
  let southwest_component := southwest_distance / Real.sqrt 2
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  exact sorry

end crystal_final_segment_distance_l1112_111217


namespace solve_for_x_l1112_111268

-- Let us state and prove that x = 495 / 13 is a solution to the equation 3x + 5 = 500 - (4x + 6x)
theorem solve_for_x (x : ℝ) : 3 * x + 5 = 500 - (4 * x + 6 * x) → x = 495 / 13 :=
by
  sorry

end solve_for_x_l1112_111268


namespace degrees_subtraction_l1112_111235

theorem degrees_subtraction :
  (108 * 3600 + 18 * 60 + 25) - (56 * 3600 + 23 * 60 + 32) = (51 * 3600 + 54 * 60 + 53) :=
by sorry

end degrees_subtraction_l1112_111235


namespace evaluate_expression_l1112_111254

variable (a : ℕ)

theorem evaluate_expression (h : a = 2) : a^3 * a^4 = 128 :=
by
  sorry

end evaluate_expression_l1112_111254


namespace discount_on_item_l1112_111220

noncomputable def discount_percentage : ℝ := 20
variable (total_cart_value original_price final_amount : ℝ)
variable (coupon_discount : ℝ)

axiom cart_value : total_cart_value = 54
axiom item_price : original_price = 20
axiom coupon : coupon_discount = 0.10
axiom final_price : final_amount = 45

theorem discount_on_item :
  ∃ x : ℝ, (total_cart_value - (x / 100) * original_price) * (1 - coupon_discount) = final_amount ∧ x = discount_percentage :=
by
  have eq1 := cart_value
  have eq2 := item_price
  have eq3 := coupon
  have eq4 := final_price
  sorry

end discount_on_item_l1112_111220


namespace gcd_g_y_l1112_111202

def g (y : ℕ) : ℕ := (3*y + 4) * (8*y + 3) * (14*y + 9) * (y + 17)

theorem gcd_g_y (y : ℕ) (h : y % 42522 = 0) : Nat.gcd (g y) y = 102 := by
  sorry

end gcd_g_y_l1112_111202


namespace typeA_selling_price_maximize_profit_l1112_111299

theorem typeA_selling_price (sales_last_year : ℝ) (sales_increase_rate : ℝ) (price_increase : ℝ) 
                            (cars_sold_last_year : ℝ) : 
                            (sales_last_year = 32000) ∧ (sales_increase_rate = 1.25) ∧ 
                            (price_increase = 400) ∧ 
                            (sales_last_year / cars_sold_last_year = (sales_last_year * sales_increase_rate) / (cars_sold_last_year + price_increase)) → 
                            (cars_sold_last_year = 1600) :=
by
  sorry

theorem maximize_profit (typeA_price : ℝ) (typeB_price : ℝ) (typeA_cost : ℝ) (typeB_cost : ℝ) 
                        (total_cars : ℕ) :
                        (typeA_price = 2000) ∧ (typeB_price = 2400) ∧ 
                        (typeA_cost = 1100) ∧ (typeB_cost = 1400) ∧ 
                        (total_cars = 50) ∧ 
                        (∀ m : ℕ, m ≤ 50 / 3) → 
                        ∃ m : ℕ, (m = 17) ∧ (50 - m * 2 ≤ 33) :=
by
  sorry

end typeA_selling_price_maximize_profit_l1112_111299


namespace terminal_side_third_quadrant_l1112_111281

noncomputable def angle_alpha : ℝ := (7 * Real.pi) / 5

def is_in_third_quadrant (angle : ℝ) : Prop :=
  ∃ k : ℤ, (3 * Real.pi) / 2 < angle + 2 * k * Real.pi ∧ angle + 2 * k * Real.pi < 2 * Real.pi

theorem terminal_side_third_quadrant : is_in_third_quadrant angle_alpha :=
sorry

end terminal_side_third_quadrant_l1112_111281


namespace find_number_l1112_111239

theorem find_number (x : ℕ) (h : 8 * x = 64) : x = 8 :=
sorry

end find_number_l1112_111239


namespace fraction_equality_l1112_111226

theorem fraction_equality (x y z : ℝ) (k : ℝ) (hx : x = 3 * k) (hy : y = 5 * k) (hz : z = 7 * k) :
  (x - y + z) / (x + y - z) = 5 := 
  sorry

end fraction_equality_l1112_111226


namespace cost_of_gas_used_l1112_111256

theorem cost_of_gas_used (initial_odometer final_odometer fuel_efficiency cost_per_gallon : ℝ)
  (h₀ : initial_odometer = 82300)
  (h₁ : final_odometer = 82335)
  (h₂ : fuel_efficiency = 22)
  (h₃ : cost_per_gallon = 3.80) :
  (final_odometer - initial_odometer) / fuel_efficiency * cost_per_gallon = 6.04 :=
by
  sorry

end cost_of_gas_used_l1112_111256


namespace hyperbola_eccentricity_l1112_111247

theorem hyperbola_eccentricity (a b c : ℚ) (h1 : (c : ℚ) = 5)
  (h2 : (b / a) = 3 / 4) (h3 : c^2 = a^2 + b^2) :
  (c / a : ℚ) = 5 / 4 :=
by
  sorry

end hyperbola_eccentricity_l1112_111247


namespace relationship_among_a_b_c_l1112_111285

noncomputable def a : ℝ := Real.logb 0.5 0.2
noncomputable def b : ℝ := Real.logb 2 0.2
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 2)

theorem relationship_among_a_b_c : b < c ∧ c < a :=
by
  sorry

end relationship_among_a_b_c_l1112_111285


namespace find_number_l1112_111251

theorem find_number (N p q : ℝ) (h₁ : N / p = 8) (h₂ : N / q = 18) (h₃ : p - q = 0.2777777777777778) : N = 4 :=
sorry

end find_number_l1112_111251


namespace average_speed_additional_hours_l1112_111234

theorem average_speed_additional_hours
  (time_first_part : ℝ) (speed_first_part : ℝ) (total_time : ℝ) (avg_speed_total : ℝ)
  (additional_hours : ℝ) (speed_additional_hours : ℝ) :
  time_first_part = 4 → speed_first_part = 35 → total_time = 24 → avg_speed_total = 50 →
  additional_hours = total_time - time_first_part →
  (time_first_part * speed_first_part + additional_hours * speed_additional_hours) / total_time = avg_speed_total →
  speed_additional_hours = 53 :=
by intros; sorry

end average_speed_additional_hours_l1112_111234


namespace logic_problem_l1112_111264

variables (p q : Prop)

theorem logic_problem (hnp : ¬ p) (hpq : ¬ (p ∧ q)) : ¬ (p ∨ q) ∨ (p ∨ q) :=
by 
  sorry

end logic_problem_l1112_111264


namespace fried_busy_frog_l1112_111282

open ProbabilityTheory

def initial_position : (ℤ × ℤ) := (0, 0)

def possible_moves : List (ℤ × ℤ) := [(0, 0), (1, 0), (0, 1)]

def p (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = initial_position then 1 else 0

noncomputable def transition (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = (0, 0) then 1/3 * p n (0, 0)
  else if pos = (0, 1) then 1/3 * p n (0, 0) + 1/3 * p n (0, 1)
  else if pos = (1, 0) then 1/3 * p n (0, 0) + 1/3 * p n (1, 0)
  else 0

noncomputable def p_1 (pos : ℤ × ℤ) : ℚ := transition 0 pos

noncomputable def p_2 (pos : ℤ × ℤ) : ℚ := transition 1 pos

noncomputable def p_3 (pos : ℤ × ℤ) : ℚ := transition 2 pos

theorem fried_busy_frog :
  p_3 (0, 0) = 1/27 :=
by
  sorry

end fried_busy_frog_l1112_111282


namespace tangents_from_point_to_circle_l1112_111261

theorem tangents_from_point_to_circle (x y k : ℝ) (
    P : ℝ × ℝ)
    (h₁ : P = (1, -1))
    (circle_eq : x^2 + y^2 + 2*x + 2*y + k = 0)
    (h₂ : P = (1, -1))
    (has_two_tangents : 1^2 + (-1)^2 - k / 2 > 0):
  -2 < k ∧ k < 2 :=
by 
    sorry

end tangents_from_point_to_circle_l1112_111261


namespace incorrect_statement_C_l1112_111222

noncomputable def f (a x : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem incorrect_statement_C :
  ¬ (∀ a : ℝ, a > 0 → ∀ x : ℝ, x > 0 → f a x ≥ 0) := sorry

end incorrect_statement_C_l1112_111222


namespace smaller_square_area_percentage_l1112_111233

noncomputable def percent_area_of_smaller_square (side_length_larger_square : ℝ) : ℝ :=
  let diagonal_larger_square := side_length_larger_square * Real.sqrt 2
  let radius_circle := diagonal_larger_square / 2
  let x := (2 + 4 * (side_length_larger_square / 2)) / ((side_length_larger_square / 2) * 2) -- Simplified quadratic solution
  let side_length_smaller_square := side_length_larger_square * x
  let area_smaller_square := side_length_smaller_square ^ 2
  let area_larger_square := side_length_larger_square ^ 2
  (area_smaller_square / area_larger_square) * 100

-- Statement to show that under given conditions, the area of the smaller square is 4% of the larger square's area
theorem smaller_square_area_percentage :
  percent_area_of_smaller_square 4 = 4 := 
sorry

end smaller_square_area_percentage_l1112_111233


namespace gcd_polynomial_l1112_111266

open Nat

theorem gcd_polynomial (b : ℤ) (hb : 1632 ∣ b) : gcd (b^2 + 11 * b + 30) (b + 6) = 6 := by
  sorry

end gcd_polynomial_l1112_111266


namespace solution_set_inequality_l1112_111211

noncomputable def f (x : ℝ) : ℝ := x * (1 - 3 * x)

theorem solution_set_inequality : {x : ℝ | f x > 0} = { x | (0 < x) ∧ (x < 1/3) } := by
  sorry

end solution_set_inequality_l1112_111211


namespace geometric_sequence_arithmetic_progression_l1112_111209

open Nat

/--
Given a geometric sequence \( \{a_n\} \) where \( a_1 = 1 \) and the sequence terms
\( 4a_1 \), \( 2a_2 \), \( a_3 \) form an arithmetic progression, prove that
the common ratio \( q = 2 \) and the sum of the first four terms \( S_4 = 15 \).
-/
theorem geometric_sequence_arithmetic_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h₀ : a 1 = 1)
    (h₁ : ∀ n, S n = (1 - q^n) / (1 - q)) 
    (h₂ : ∀ k n, a (k + n) = a k * q ^ n) 
    (h₃ : 4 * a 1 + a 3 = 4 * a 2) :
  q = 2 ∧ S 4 = 15 := 
sorry

end geometric_sequence_arithmetic_progression_l1112_111209


namespace reservoir_shortage_l1112_111229

noncomputable def reservoir_information := 
  let current_level := 14 -- million gallons
  let normal_level_due_to_yield := current_level / 2
  let percentage_of_capacity := 0.70
  let evaporation_factor := 0.90
  let total_capacity := current_level / percentage_of_capacity
  let normal_level_after_evaporation := normal_level_due_to_yield * evaporation_factor
  let shortage := total_capacity - normal_level_after_evaporation
  shortage

theorem reservoir_shortage :
  reservoir_information = 13.7 := 
by
  sorry

end reservoir_shortage_l1112_111229


namespace evaluate_nested_radical_l1112_111249

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l1112_111249


namespace no_duplicate_among_expressions_l1112_111204

theorem no_duplicate_among_expressions
  (N a1 a2 b1 b2 c1 c2 d1 d2 : ℕ)
  (ha : a1 = x^2)
  (hb : b1 = y^3)
  (hc : c1 = z^5)
  (hd : d1 = w^7)
  (ha2 : a2 = m^2)
  (hb2 : b2 = n^3)
  (hc2 : c2 = p^5)
  (hd2 : d2 = q^7)
  (h1 : N = a1 - a2)
  (h2 : N = b1 - b2)
  (h3 : N = c1 - c2)
  (h4 : N = d1 - d2) :
  ¬ (a1 = b1 ∨ a1 = c1 ∨ a1 = d1 ∨ b1 = c1 ∨ b1 = d1 ∨ c1 = d1) :=
by
  -- Begin proof here
  sorry

end no_duplicate_among_expressions_l1112_111204


namespace find_pairs_l1112_111243

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

theorem find_pairs {a b : ℝ} :
  (0 < b) → (b ≤ 1) → (0 < a) → (a < 1) → (2 * a + b ≤ 2) →
  (∀ x y : ℝ, f a b (x * y) + f a b (x + y) ≥ f a b x * f a b y) :=
by
  intros h_b_gt_zero h_b_le_one h_a_gt_zero h_a_lt_one h_2a_b_le_2
  sorry

end find_pairs_l1112_111243


namespace gunther_cleaning_free_time_l1112_111278

theorem gunther_cleaning_free_time :
  let vacuum := 45
  let dusting := 60
  let mopping := 30
  let bathroom := 40
  let windows := 15
  let brushing_per_cat := 5
  let cats := 4

  let free_time_hours := 4
  let free_time_minutes := 25

  let cleaning_time := vacuum + dusting + mopping + bathroom + windows + (brushing_per_cat * cats)
  let free_time_total := (free_time_hours * 60) + free_time_minutes

  free_time_total - cleaning_time = 55 :=
by
  sorry

end gunther_cleaning_free_time_l1112_111278


namespace statement_B_statement_C_l1112_111238

variable (a b c : ℝ)

-- Condition: a > b
def condition1 := a > b

-- Condition: a / c^2 > b / c^2
def condition2 := a / c^2 > b / c^2

-- Statement B: If a > b, then a - 1 > b - 2
theorem statement_B (ha_gt_b : condition1 a b) : a - 1 > b - 2 :=
by sorry

-- Statement C: If a / c^2 > b / c^2, then a > b
theorem statement_C (ha_div_csqr_gt_hb_div_csqr : condition2 a b c) : a > b :=
by sorry

end statement_B_statement_C_l1112_111238


namespace difference_in_dimes_l1112_111252

variables (q : ℝ)

def samantha_quarters : ℝ := 3 * q + 2
def bob_quarters : ℝ := 2 * q + 8
def quarter_to_dimes : ℝ := 2.5

theorem difference_in_dimes :
  quarter_to_dimes * (samantha_quarters q - bob_quarters q) = 2.5 * q - 15 :=
by sorry

end difference_in_dimes_l1112_111252
