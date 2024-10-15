import Mathlib

namespace NUMINAMATH_GPT_find_divisor_l358_35869

theorem find_divisor (q r : ℤ) : ∃ d : ℤ, 151 = d * q + r ∧ q = 11 ∧ r = -4 → d = 14 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_divisor_l358_35869


namespace NUMINAMATH_GPT_john_spends_40_dollars_l358_35844

-- Definitions based on conditions
def cost_per_loot_box : ℝ := 5
def average_value_per_loot_box : ℝ := 3.5
def average_loss : ℝ := 12

-- Prove the amount spent on loot boxes is $40
theorem john_spends_40_dollars :
  ∃ S : ℝ, (S * (cost_per_loot_box - average_value_per_loot_box) / cost_per_loot_box = average_loss) ∧ S = 40 :=
by
  sorry

end NUMINAMATH_GPT_john_spends_40_dollars_l358_35844


namespace NUMINAMATH_GPT_trigonometric_identity_l358_35877

theorem trigonometric_identity (x : ℝ) : 
  x = Real.pi / 4 → (1 + Real.sin (x + Real.pi / 4) - Real.cos (x + Real.pi / 4)) / 
                          (1 + Real.sin (x + Real.pi / 4) + Real.cos (x + Real.pi / 4)) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l358_35877


namespace NUMINAMATH_GPT_circle_intersection_area_l358_35881

theorem circle_intersection_area
  (r : ℝ)
  (θ : ℝ)
  (a b c : ℝ)
  (h_r : r = 5)
  (h_θ : θ = π / 2)
  (h_expr : a * Real.sqrt b + c * π = 5 * 5 * π / 2 - 5 * 5 * Real.sqrt 3 / 2 ) :
  a + b + c = -9.5 :=
by
  sorry

end NUMINAMATH_GPT_circle_intersection_area_l358_35881


namespace NUMINAMATH_GPT_problem1_l358_35850

theorem problem1 (a b : ℝ) (ha : a > 2) (hb : b > 2) :
  (a - 2) * (b - 2) = 2 :=
sorry

end NUMINAMATH_GPT_problem1_l358_35850


namespace NUMINAMATH_GPT_betty_initial_marbles_l358_35883

theorem betty_initial_marbles (B : ℝ) (h1 : 0.40 * B = 24) : B = 60 :=
by
  sorry

end NUMINAMATH_GPT_betty_initial_marbles_l358_35883


namespace NUMINAMATH_GPT_number_of_nickels_l358_35884

variable (n : Nat) -- number of nickels

def value_of_nickels := n * 5 -- value of nickels n in cents
def total_value :=
    2 * 100 +   -- 2 one-dollar bills
    1 * 500 +   -- 1 five-dollar bill
    13 * 25 +   -- 13 quarters
    20 * 10 +   -- 20 dimes
    35 * 1 +    -- 35 pennies
    value_of_nickels n

theorem number_of_nickels :
    total_value n = 1300 ↔ n = 8 :=
by sorry

end NUMINAMATH_GPT_number_of_nickels_l358_35884


namespace NUMINAMATH_GPT_max_n_is_4024_l358_35805

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) : ℕ :=
  4024

theorem max_n_is_4024 (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) :
  max_n_for_positive_sum a d h1 h2 h3 = 4024 :=
by
  sorry

end NUMINAMATH_GPT_max_n_is_4024_l358_35805


namespace NUMINAMATH_GPT_myrtle_eggs_after_collection_l358_35875

def henA_eggs_per_day : ℕ := 3
def henB_eggs_per_day : ℕ := 4
def henC_eggs_per_day : ℕ := 2
def henD_eggs_per_day : ℕ := 5
def henE_eggs_per_day : ℕ := 3

def days_gone : ℕ := 12
def eggs_taken_by_neighbor : ℕ := 32

def eggs_dropped_day1 : ℕ := 3
def eggs_dropped_day2 : ℕ := 5
def eggs_dropped_day3 : ℕ := 2

theorem myrtle_eggs_after_collection :
  let total_eggs :=
    (henA_eggs_per_day * days_gone) +
    (henB_eggs_per_day * days_gone) +
    (henC_eggs_per_day * days_gone) +
    (henD_eggs_per_day * days_gone) +
    (henE_eggs_per_day * days_gone)
  let remaining_eggs_after_neighbor := total_eggs - eggs_taken_by_neighbor
  let total_dropped_eggs := eggs_dropped_day1 + eggs_dropped_day2 + eggs_dropped_day3
  let eggs_after_drops := remaining_eggs_after_neighbor - total_dropped_eggs
  eggs_after_drops = 162 := 
by 
  sorry

end NUMINAMATH_GPT_myrtle_eggs_after_collection_l358_35875


namespace NUMINAMATH_GPT_four_p_minus_three_is_perfect_square_l358_35815

theorem four_p_minus_three_is_perfect_square 
  {n p : ℕ} (hn : 1 < n) (hp : 1 < p) (hp_prime : Prime p) 
  (h1 : n ∣ (p - 1)) (h2 : p ∣ (n^3 - 1)) :
  ∃ k : ℕ, 4 * p - 3 = k ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_four_p_minus_three_is_perfect_square_l358_35815


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l358_35833

noncomputable def sum_first_ten_terms (a d : ℕ) : ℕ :=
  (10 / 2) * (2 * a + (10 - 1) * d)

theorem arithmetic_sequence_sum 
  (a d : ℕ) 
  (h1 : a + 2 * d = 8) 
  (h2 : a + 5 * d = 14) :
  sum_first_ten_terms a d = 130 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l358_35833


namespace NUMINAMATH_GPT_range_of_AB_l358_35860

variable (AB BC AC : ℝ)
variable (θ : ℝ)
variable (B : ℝ)

-- Conditions
axiom angle_condition : θ = 150
axiom length_condition : AC = 2

-- Theorem to prove
theorem range_of_AB (h_θ : θ = 150) (h_AC : AC = 2) : (0 < AB) ∧ (AB ≤ 4) :=
sorry

end NUMINAMATH_GPT_range_of_AB_l358_35860


namespace NUMINAMATH_GPT_sarah_annual_income_l358_35871

theorem sarah_annual_income (q : ℝ) (I T : ℝ)
    (h1 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)) 
    (h2 : T = 0.01 * (q + 0.5) * I) : 
    I = 36000 := by
  sorry

end NUMINAMATH_GPT_sarah_annual_income_l358_35871


namespace NUMINAMATH_GPT_ratio_A_to_B_l358_35800

theorem ratio_A_to_B (A B C : ℕ) (h1 : A + B + C = 406) (h2 : C = 232) (h3 : B = C / 2) : A / gcd A B = 1 ∧ B / gcd A B = 2 := 
by sorry

end NUMINAMATH_GPT_ratio_A_to_B_l358_35800


namespace NUMINAMATH_GPT_min_gloves_proof_l358_35851

-- Let n represent the number of participants
def n : Nat := 63

-- Let g represent the number of gloves per participant
def g : Nat := 2

-- The minimum number of gloves required
def min_gloves : Nat := n * g

theorem min_gloves_proof : min_gloves = 126 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_min_gloves_proof_l358_35851


namespace NUMINAMATH_GPT_decompose_96_l358_35812

theorem decompose_96 (x y : ℤ) (h1 : x * y = 96) (h2 : x^2 + y^2 = 208) :
  (x = 8 ∧ y = 12) ∨ (x = 12 ∧ y = 8) ∨ (x = -8 ∧ y = -12) ∨ (x = -12 ∧ y = -8) := by
  sorry

end NUMINAMATH_GPT_decompose_96_l358_35812


namespace NUMINAMATH_GPT_polynomial_value_at_two_l358_35807

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem polynomial_value_at_two : f 2 = 243 := by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_polynomial_value_at_two_l358_35807


namespace NUMINAMATH_GPT_doughnut_price_l358_35836

theorem doughnut_price
  (K C B : ℕ)
  (h1: K = 4 * C + 5)
  (h2: K = 5 * C - 6)
  (h3: K = 2 * C + 3 * B) :
  B = 9 := 
sorry

end NUMINAMATH_GPT_doughnut_price_l358_35836


namespace NUMINAMATH_GPT_total_marbles_l358_35847

theorem total_marbles (p y u : ℕ) :
  y + u = 10 →
  p + u = 12 →
  p + y = 6 →
  p + y + u = 14 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_marbles_l358_35847


namespace NUMINAMATH_GPT_birds_percentage_not_hawks_paddyfield_warblers_kingfishers_l358_35859

theorem birds_percentage_not_hawks_paddyfield_warblers_kingfishers
  (total_birds : ℕ)
  (hawks_percentage : ℝ := 0.3)
  (paddyfield_warblers_percentage : ℝ := 0.4)
  (kingfishers_ratio : ℝ := 0.25) :
  (35 : ℝ) = 100 * ( total_birds - (hawks_percentage * total_birds) 
                     - (paddyfield_warblers_percentage * (total_birds - (hawks_percentage * total_birds))) 
                     - (kingfishers_ratio * paddyfield_warblers_percentage * (total_birds - (hawks_percentage * total_birds))) )
                / total_birds :=
by
  sorry

end NUMINAMATH_GPT_birds_percentage_not_hawks_paddyfield_warblers_kingfishers_l358_35859


namespace NUMINAMATH_GPT_units_digit_of_3_pow_4_l358_35852

theorem units_digit_of_3_pow_4 : (3^4 % 10) = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_3_pow_4_l358_35852


namespace NUMINAMATH_GPT_exp_pos_for_all_x_l358_35879

theorem exp_pos_for_all_x (h : ¬ ∃ x_0 : ℝ, Real.exp x_0 ≤ 0) : ∀ x : ℝ, Real.exp x > 0 :=
by
  sorry

end NUMINAMATH_GPT_exp_pos_for_all_x_l358_35879


namespace NUMINAMATH_GPT_num_six_year_olds_l358_35839

theorem num_six_year_olds (x : ℕ) 
  (h3 : 13 = 13) 
  (h4 : 20 = 20) 
  (h5 : 15 = 15) 
  (h_sum1 : 13 + 20 = 33) 
  (h_sum2 : 15 + x = 15 + x) 
  (h_avg : 2 * 35 = 70) 
  (h_total : 33 + (15 + x) = 70) : 
  x = 22 :=
by
  sorry

end NUMINAMATH_GPT_num_six_year_olds_l358_35839


namespace NUMINAMATH_GPT_length_of_MN_l358_35810

theorem length_of_MN (A B C D K L M N : Type) 
  (h1 : A → B → C → D → Prop) -- Condition for rectangle ABCD
  (h2 : K → L → Prop) -- Condition for circle intersecting AB at K and L
  (h3 : M → N → Prop) -- Condition for circle intersecting CD at M and N
  (AK KL DN : ℝ)
  (h4 : AK = 10)
  (h5 : KL = 17)
  (h6 : DN = 7) :
  ∃ MN : ℝ, MN = 23 := 
sorry

end NUMINAMATH_GPT_length_of_MN_l358_35810


namespace NUMINAMATH_GPT_sequence_term_l358_35808

theorem sequence_term (S : ℕ → ℕ) (h : ∀ (n : ℕ), S n = 5 * n + 2 * n^2) (r : ℕ) : 
  (S r - S (r - 1) = 4 * r + 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_term_l358_35808


namespace NUMINAMATH_GPT_lucas_mod_prime_zero_l358_35892

-- Define the Lucas sequence
def lucas : ℕ → ℕ
| 0 => 1       -- Note that in the mathematical problem L_1 is given as 1. Therefore we adjust for 0-based index in programming.
| 1 => 3
| (n + 2) => lucas n + lucas (n + 1)

-- Main theorem statement
theorem lucas_mod_prime_zero (p : ℕ) (hp : Nat.Prime p) : (lucas p - 1) % p = 0 := by
  sorry

end NUMINAMATH_GPT_lucas_mod_prime_zero_l358_35892


namespace NUMINAMATH_GPT_area_three_layers_is_nine_l358_35842

-- Define the areas as natural numbers
variable (P Q R S T U V : ℕ)

-- Define the combined area of the rugs
def combined_area_rugs := P + Q + R + 2 * (S + T + U) + 3 * V = 90

-- Define the total area covered by the floor
def total_area_floor := P + Q + R + S + T + U + V = 60

-- Define the area covered by exactly two layers of rug
def area_two_layers := S + T + U = 12

-- Define the area covered by exactly three layers of rug
def area_three_layers := V

-- Prove the area covered by exactly three layers of rug is 9
theorem area_three_layers_is_nine
  (h1 : combined_area_rugs P Q R S T U V)
  (h2 : total_area_floor P Q R S T U V)
  (h3 : area_two_layers S T U) :
  area_three_layers V = 9 := by
  sorry

end NUMINAMATH_GPT_area_three_layers_is_nine_l358_35842


namespace NUMINAMATH_GPT_charge_y1_charge_y2_cost_effective_range_call_duration_difference_l358_35858

def y1 (x : ℕ) : ℝ :=
  if x ≤ 600 then 30 else 0.1 * x - 30

def y2 (x : ℕ) : ℝ :=
  if x ≤ 1200 then 50 else 0.1 * x - 70

theorem charge_y1 (x : ℕ) :
  (x ≤ 600 → y1 x = 30) ∧ (x > 600 → y1 x = 0.1 * x - 30) :=
by sorry

theorem charge_y2 (x : ℕ) :
  (x ≤ 1200 → y2 x = 50) ∧ (x > 1200 → y2 x = 0.1 * x - 70) :=
by sorry

theorem cost_effective_range (x : ℕ) :
  (0 ≤ x) ∧ (x < 800) → y1 x < y2 x :=
by sorry

noncomputable def call_time_xiaoming : ℕ := 1300
noncomputable def call_time_xiaohua : ℕ := 900

theorem call_duration_difference :
  call_time_xiaoming = call_time_xiaohua + 400 :=
by sorry

end NUMINAMATH_GPT_charge_y1_charge_y2_cost_effective_range_call_duration_difference_l358_35858


namespace NUMINAMATH_GPT_not_divisible_by_121_l358_35856

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 2 * n + 12)) :=
sorry

end NUMINAMATH_GPT_not_divisible_by_121_l358_35856


namespace NUMINAMATH_GPT_value_of_a_plus_one_l358_35828

theorem value_of_a_plus_one (a : ℤ) (h : |a| = 3) : a + 1 = 4 ∨ a + 1 = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_one_l358_35828


namespace NUMINAMATH_GPT_T_is_x_plus_3_to_the_4_l358_35866

variable (x : ℝ)

def T : ℝ := (x + 2)^4 + 4 * (x + 2)^3 + 6 * (x + 2)^2 + 4 * (x + 2) + 1

theorem T_is_x_plus_3_to_the_4 : T x = (x + 3)^4 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_T_is_x_plus_3_to_the_4_l358_35866


namespace NUMINAMATH_GPT_initial_integer_l358_35898

theorem initial_integer (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end NUMINAMATH_GPT_initial_integer_l358_35898


namespace NUMINAMATH_GPT_number_of_ways_to_choose_chairs_l358_35811

def choose_chairs_equivalent (chairs : Nat) (students : Nat) (professors : Nat) : Nat :=
  let positions := (chairs - 2)  -- exclude first and last chair
  Nat.choose positions professors * Nat.factorial professors

theorem number_of_ways_to_choose_chairs : choose_chairs_equivalent 10 5 4 = 1680 :=
by
  -- The positions for professors are available from chairs 2 through 9 which are 8 positions.
  /- Calculation for choosing 4 positions out of these 8:
     C(8,4) * 4! = 70 * 24 = 1680 -/
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_chairs_l358_35811


namespace NUMINAMATH_GPT_fifth_equation_l358_35878

-- Define the conditions
def condition1 : Prop := 2^1 * 1 = 2
def condition2 : Prop := 2^2 * 1 * 3 = 3 * 4
def condition3 : Prop := 2^3 * 1 * 3 * 5 = 4 * 5 * 6

-- The statement to prove
theorem fifth_equation (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 :=
sorry

end NUMINAMATH_GPT_fifth_equation_l358_35878


namespace NUMINAMATH_GPT_real_number_c_l358_35857

theorem real_number_c (x1 x2 c : ℝ) (h_eqn : x1 + x2 = -1) (h_prod : x1 * x2 = c) (h_cond : x1^2 * x2 + x2^2 * x1 = 3) : c = -3 :=
by sorry

end NUMINAMATH_GPT_real_number_c_l358_35857


namespace NUMINAMATH_GPT_jake_has_fewer_peaches_than_steven_l358_35801

theorem jake_has_fewer_peaches_than_steven :
  ∀ (jillPeaches jakePeaches stevenPeaches : ℕ),
    jillPeaches = 12 →
    jakePeaches = jillPeaches - 1 →
    stevenPeaches = jillPeaches + 15 →
    stevenPeaches - jakePeaches = 16 :=
  by
    intros jillPeaches jakePeaches stevenPeaches
    intro h_jill
    intro h_jake
    intro h_steven
    sorry

end NUMINAMATH_GPT_jake_has_fewer_peaches_than_steven_l358_35801


namespace NUMINAMATH_GPT_possible_values_of_a1_l358_35837

def sequence_satisfies_conditions (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 1, a n ≤ a (n + 1) ∧ a (n + 1) ≤ a n + 5) ∧
  (∀ n ≥ 1, n ∣ a n)

theorem possible_values_of_a1 (a : ℕ → ℕ) :
  sequence_satisfies_conditions a → ∃ k ≤ 26, a 1 = k :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a1_l358_35837


namespace NUMINAMATH_GPT_tilde_tilde_tilde_47_l358_35846

def tilde (N : ℝ) : ℝ := 0.4 * N + 2

theorem tilde_tilde_tilde_47 : tilde (tilde (tilde 47)) = 6.128 := 
by
  sorry

end NUMINAMATH_GPT_tilde_tilde_tilde_47_l358_35846


namespace NUMINAMATH_GPT_triangle_exterior_angle_bisectors_l358_35868

theorem triangle_exterior_angle_bisectors 
  (α β γ α1 β1 γ1 : ℝ) 
  (h₁ : α = (β / 2 + γ / 2)) 
  (h₂ : β = (γ / 2 + α / 2)) 
  (h₃ : γ = (α / 2 + β / 2)) :
  α = 180 - 2 * α1 ∧
  β = 180 - 2 * β1 ∧
  γ = 180 - 2 * γ1 := by
  sorry

end NUMINAMATH_GPT_triangle_exterior_angle_bisectors_l358_35868


namespace NUMINAMATH_GPT_find_k_value_l358_35829

theorem find_k_value (k : ℝ) : 
  (-x ^ 2 - (k + 11) * x - 8 = -( (x - 2) * (x - 4) ) ) → k = -17 := 
by 
  sorry

end NUMINAMATH_GPT_find_k_value_l358_35829


namespace NUMINAMATH_GPT_expression_value_zero_l358_35863

theorem expression_value_zero (a b c : ℝ) (h1 : a^2 + b = b^2 + c) (h2 : b^2 + c = c^2 + a) (h3 : c^2 + a = a^2 + b) :
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_zero_l358_35863


namespace NUMINAMATH_GPT_largest_of_five_consecutive_integers_with_product_15120_is_9_l358_35826

theorem largest_of_five_consecutive_integers_with_product_15120_is_9 :
  ∃ (a b c d e : ℤ), a * b * c * d * e = 15120 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e = 9 :=
sorry

end NUMINAMATH_GPT_largest_of_five_consecutive_integers_with_product_15120_is_9_l358_35826


namespace NUMINAMATH_GPT_cone_diameter_base_l358_35827

theorem cone_diameter_base 
  (r l : ℝ) 
  (h_semicircle : l = 2 * r) 
  (h_surface_area : π * r ^ 2 + π * r * l = 3 * π) 
  : 2 * r = 2 :=
by
  sorry

end NUMINAMATH_GPT_cone_diameter_base_l358_35827


namespace NUMINAMATH_GPT_A_more_than_B_l358_35855

noncomputable def proportion := (5, 3, 2, 3)
def C_share := 1000
def parts := 2
noncomputable def part_value := C_share / parts
noncomputable def A_share := part_value * 5
noncomputable def B_share := part_value * 3

theorem A_more_than_B : A_share - B_share = 1000 := by
  sorry

end NUMINAMATH_GPT_A_more_than_B_l358_35855


namespace NUMINAMATH_GPT_sin_60_equiv_l358_35876

theorem sin_60_equiv : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_60_equiv_l358_35876


namespace NUMINAMATH_GPT_range_of_a_in_fourth_quadrant_l358_35880

-- Define the fourth quadrant condition
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Define the point P(a+1, a-1) and state the theorem
theorem range_of_a_in_fourth_quadrant (a : ℝ) :
  in_fourth_quadrant (a + 1) (a - 1) → -1 < a ∧ a < 1 :=
by
  intro h
  have h1 : a + 1 > 0 := h.1
  have h2 : a - 1 < 0 := h.2
  have h3 : a > -1 := by linarith
  have h4 : a < 1 := by linarith
  exact ⟨h3, h4⟩

end NUMINAMATH_GPT_range_of_a_in_fourth_quadrant_l358_35880


namespace NUMINAMATH_GPT_not_all_ten_segments_form_triangle_l358_35882

theorem not_all_ten_segments_form_triangle :
  ∃ (segments : Fin 10 → ℕ), ∀ i j k : Fin 10, i < j → j < k → segments i + segments j ≤ segments k := 
sorry

end NUMINAMATH_GPT_not_all_ten_segments_form_triangle_l358_35882


namespace NUMINAMATH_GPT_equivalent_proof_problem_l358_35895

noncomputable def perimeter_inner_polygon (pentagon_perimeter : ℕ) : ℕ :=
  let side_length := pentagon_perimeter / 5
  let inner_polygon_sides := 10
  inner_polygon_sides * side_length

theorem equivalent_proof_problem :
  perimeter_inner_polygon 65 = 130 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l358_35895


namespace NUMINAMATH_GPT_number_of_people_in_each_van_l358_35865

theorem number_of_people_in_each_van (x : ℕ) 
  (h1 : 6 * x + 8 * 18 = 180) : x = 6 :=
by sorry

end NUMINAMATH_GPT_number_of_people_in_each_van_l358_35865


namespace NUMINAMATH_GPT_verify_A_l358_35888

def matrix_A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![62 / 7, -9 / 7], ![2 / 7, 17 / 7]]

theorem verify_A :
  matrix_A.mulVec ![1, 3] = ![5, 7] ∧
  matrix_A.mulVec ![-2, 1] = ![-19, 3] :=
by
  sorry

end NUMINAMATH_GPT_verify_A_l358_35888


namespace NUMINAMATH_GPT_monotonicity_of_f_inequality_of_f_l358_35874

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a * Real.log x

theorem monotonicity_of_f {a : ℝ}:
(a ≥ 0 → ∀ x y : ℝ, 0 < x ∧ x < y → f x a ≤ f y a) ∧
(a < 0 → ∀ x y : ℝ, 0 < x ∧ x < y ∧ x ≥ -1 + Real.sqrt (1 - 2 * a) → f x a ≤ f y a 
∨ 0 < x ∧ x < -1 + Real.sqrt (1 - 2 * a) → f x a ≥ f y a) := sorry

theorem inequality_of_f {a : ℝ} (h : t ≥ 1) :
(f (2*t-1) a ≥ 2 * f t a - 3) ↔ (a ≤ 2) := sorry

end NUMINAMATH_GPT_monotonicity_of_f_inequality_of_f_l358_35874


namespace NUMINAMATH_GPT_remainder_of_sum_l358_35873

theorem remainder_of_sum : 
  let a := 21160
  let b := 21162
  let c := 21164
  let d := 21166
  let e := 21168
  let f := 21170
  (a + b + c + d + e + f) % 12 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l358_35873


namespace NUMINAMATH_GPT_water_depth_in_cylindrical_tub_l358_35821

theorem water_depth_in_cylindrical_tub
  (tub_diameter : ℝ) (tub_depth : ℝ) (pail_angle : ℝ)
  (h_diam : tub_diameter = 40)
  (h_depth : tub_depth = 50)
  (h_angle : pail_angle = 45) :
  ∃ water_depth : ℝ, water_depth = 30 :=
by
  sorry

end NUMINAMATH_GPT_water_depth_in_cylindrical_tub_l358_35821


namespace NUMINAMATH_GPT_restore_temperature_time_l358_35891

theorem restore_temperature_time :
  let rate_increase := 8 -- degrees per hour
  let duration_increase := 3 -- hours
  let rate_decrease := 4 -- degrees per hour
  let total_increase := rate_increase * duration_increase
  let time := total_increase / rate_decrease
  time = 6 := 
by
  sorry

end NUMINAMATH_GPT_restore_temperature_time_l358_35891


namespace NUMINAMATH_GPT_range_of_k_l358_35896

theorem range_of_k 
  (h : ∀ x : ℝ, x^2 + 2 * k * x - (k - 2) > 0) : -2 < k ∧ k < 1 := 
sorry

end NUMINAMATH_GPT_range_of_k_l358_35896


namespace NUMINAMATH_GPT_shaded_area_correct_l358_35841

-- Conditions
def side_length_square := 40
def triangle1_base := 15
def triangle1_height := 15
def triangle2_base := 15
def triangle2_height := 15

-- Calculation
def square_area := side_length_square * side_length_square
def triangle1_area := 1 / 2 * triangle1_base * triangle1_height
def triangle2_area := 1 / 2 * triangle2_base * triangle2_height
def total_triangle_area := triangle1_area + triangle2_area
def shaded_region_area := square_area - total_triangle_area

-- Theorem to prove
theorem shaded_area_correct : shaded_region_area = 1375 := by
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l358_35841


namespace NUMINAMATH_GPT_motorcyclist_speed_before_delay_l358_35845

/-- Given conditions and question:
1. The motorcyclist was delayed by 0.4 hours.
2. After the delay, the motorcyclist increased his speed by 10 km/h.
3. The motorcyclist made up for the lost time over a stretch of 80 km.
-/
theorem motorcyclist_speed_before_delay :
  ∃ x : ℝ, (80 / x - 0.4 = 80 / (x + 10)) ∧ x = 40 :=
sorry

end NUMINAMATH_GPT_motorcyclist_speed_before_delay_l358_35845


namespace NUMINAMATH_GPT_root_sum_of_reciprocals_l358_35861

theorem root_sum_of_reciprocals {m : ℝ} :
  (∃ (a b : ℝ), a ≠ b ∧ (a + b) = 2 * (m + 1) ∧ (a * b) = m^2 + 2 ∧ (1/a + 1/b) = 1) →
  m = 2 :=
by sorry

end NUMINAMATH_GPT_root_sum_of_reciprocals_l358_35861


namespace NUMINAMATH_GPT_picture_area_l358_35803

theorem picture_area (x y : ℕ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (3*x + 3) * (y + 2) = 110) : x * y = 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_picture_area_l358_35803


namespace NUMINAMATH_GPT_low_income_households_sampled_l358_35831

def total_households := 500
def high_income_households := 125
def middle_income_households := 280
def low_income_households := 95
def sampled_high_income_households := 25

theorem low_income_households_sampled :
  (sampled_high_income_households / high_income_households) * low_income_households = 19 := by
  sorry

end NUMINAMATH_GPT_low_income_households_sampled_l358_35831


namespace NUMINAMATH_GPT_positive_integer_triples_l358_35832

theorem positive_integer_triples (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b ∣ (a + 1) ∧ c ∣ (b + 1) ∧ a ∣ (c + 1)) ↔ (a = 1 ∧ b = 1 ∧ c = 1 ∨
  a = 3 ∧ b = 4 ∧ c = 5 ∨ a = 4 ∧ b = 5 ∧ c = 3 ∨ a = 5 ∧ b = 3 ∧ c = 4) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_triples_l358_35832


namespace NUMINAMATH_GPT_no_solution_abs_val_l358_35885

theorem no_solution_abs_val (x : ℝ) : ¬(∃ x : ℝ, |5 * x| + 7 = 0) :=
sorry

end NUMINAMATH_GPT_no_solution_abs_val_l358_35885


namespace NUMINAMATH_GPT_sequence_10th_term_l358_35840

theorem sequence_10th_term (a : ℕ → ℝ) 
  (h_initial : a 1 = 1) 
  (h_recursive : ∀ n, a (n + 1) = 2 * a n / (a n + 2)) : 
  a 10 = 2 / 11 :=
sorry

end NUMINAMATH_GPT_sequence_10th_term_l358_35840


namespace NUMINAMATH_GPT_rationalize_denominator_l358_35887

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℤ), 
  (1 / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 11)) = 
    (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
  A + B + C + D + E + F = 136 := 
sorry

end NUMINAMATH_GPT_rationalize_denominator_l358_35887


namespace NUMINAMATH_GPT_complement_U_P_l358_35872

theorem complement_U_P :
  let U := {y : ℝ | y ≠ 0 }
  let P := {y : ℝ | 0 < y ∧ y < 1/2}
  let complement_U_P := {y : ℝ | y ∈ U ∧ y ∉ P}
  (complement_U_P = {y : ℝ | y < 0} ∪ {y : ℝ | y > 1/2}) :=
by
  sorry

end NUMINAMATH_GPT_complement_U_P_l358_35872


namespace NUMINAMATH_GPT_price_of_baseball_bat_l358_35862

theorem price_of_baseball_bat 
  (price_A : ℕ) (price_B : ℕ) (price_bat : ℕ) 
  (hA : price_A = 10 * 29)
  (hB : price_B = 14 * (25 / 10))
  (h0 : price_A = price_B + price_bat + 237) :
  price_bat = 18 :=
by
  sorry

end NUMINAMATH_GPT_price_of_baseball_bat_l358_35862


namespace NUMINAMATH_GPT_total_weight_correct_l358_35889

-- Conditions for the weights of different types of candies
def frank_chocolate_weight : ℝ := 3
def gwen_chocolate_weight : ℝ := 2
def frank_gummy_bears_weight : ℝ := 2
def gwen_gummy_bears_weight : ℝ := 2.5
def frank_caramels_weight : ℝ := 1
def gwen_caramels_weight : ℝ := 1
def frank_hard_candy_weight : ℝ := 4
def gwen_hard_candy_weight : ℝ := 1.5

-- Combined weights of each type of candy
def chocolate_weight : ℝ := frank_chocolate_weight + gwen_chocolate_weight
def gummy_bears_weight : ℝ := frank_gummy_bears_weight + gwen_gummy_bears_weight
def caramels_weight : ℝ := frank_caramels_weight + gwen_caramels_weight
def hard_candy_weight : ℝ := frank_hard_candy_weight + gwen_hard_candy_weight

-- Total weight of the Halloween candy haul
def total_halloween_weight : ℝ := 
  chocolate_weight +
  gummy_bears_weight +
  caramels_weight +
  hard_candy_weight

-- Theorem to prove the total weight is 17 pounds
theorem total_weight_correct : total_halloween_weight = 17 := by
  sorry

end NUMINAMATH_GPT_total_weight_correct_l358_35889


namespace NUMINAMATH_GPT_fish_caught_in_second_catch_l358_35822

theorem fish_caught_in_second_catch
  (tagged_fish_released : Int)
  (tagged_fish_in_second_catch : Int)
  (total_fish_in_pond : Int)
  (C : Int)
  (h_tagged_fish_count : tagged_fish_released = 60)
  (h_tagged_in_second_catch : tagged_fish_in_second_catch = 2)
  (h_total_fish : total_fish_in_pond = 1800) :
  C = 60 :=
by
  sorry

end NUMINAMATH_GPT_fish_caught_in_second_catch_l358_35822


namespace NUMINAMATH_GPT_temperature_conversion_l358_35849

theorem temperature_conversion (F : ℝ) (C : ℝ) : 
  F = 95 → 
  C = (F - 32) * 5 / 9 → 
  C = 35 := by
  intro hF hC
  sorry

end NUMINAMATH_GPT_temperature_conversion_l358_35849


namespace NUMINAMATH_GPT_at_least_two_equal_elements_l358_35824

open Function

theorem at_least_two_equal_elements :
  ∀ (k : Fin 10 → Fin 10),
    (∀ i j : Fin 10, i ≠ j → k i ≠ k j) → False :=
by
  intros k h
  sorry

end NUMINAMATH_GPT_at_least_two_equal_elements_l358_35824


namespace NUMINAMATH_GPT_circle_chord_area_l358_35820

noncomputable def part_circle_area_between_chords (R : ℝ) : ℝ :=
  (R^2 * (Real.pi + Real.sqrt 3)) / 2

theorem circle_chord_area (R : ℝ) :
  ∀ (a₃ a₆ : ℝ),
    a₃ = Real.sqrt 3 * R →
    a₆ = R →
    part_circle_area_between_chords R = (R^2 * (Real.pi + Real.sqrt 3)) / 2 :=
by
  intros a₃ a₆ h₁ h₂
  sorry

end NUMINAMATH_GPT_circle_chord_area_l358_35820


namespace NUMINAMATH_GPT_jane_journey_duration_l358_35870

noncomputable def hours_to_seconds (h : ℕ) : ℕ := h * 3600 + 30

theorem jane_journey_duration :
  ∃ (start_time end_time : ℕ), 
    (start_time > 10 * 3600) ∧ (start_time < 11 * 3600) ∧
    (end_time > 17 * 3600) ∧ (end_time < 18 * 3600) ∧
    end_time - start_time = hours_to_seconds 7 :=
by sorry

end NUMINAMATH_GPT_jane_journey_duration_l358_35870


namespace NUMINAMATH_GPT_rectangular_coordinates_from_polar_l358_35886

theorem rectangular_coordinates_from_polar (x y r θ : ℝ) (h1 : r * Real.cos θ = x) (h2 : r * Real.sin θ = y) :
    r = 10 ∧ θ = Real.arctan (6 / 8) ∧ (2 * r, 3 * θ) = (20, 3 * Real.arctan (6 / 8)) →
    (20 * Real.cos (3 * Real.arctan (6 / 8)), 20 * Real.sin (3 * Real.arctan (6 / 8))) = (-7.04, 18.72) :=
by
  intros
  -- We need to prove that the statement holds
  sorry

end NUMINAMATH_GPT_rectangular_coordinates_from_polar_l358_35886


namespace NUMINAMATH_GPT_slower_speed_l358_35848

theorem slower_speed (x : ℝ) :
  (5 * (24 / x) = 24 + 6) → x = 4 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_slower_speed_l358_35848


namespace NUMINAMATH_GPT_jill_arrives_earlier_by_30_minutes_l358_35823

theorem jill_arrives_earlier_by_30_minutes :
  ∀ (d : ℕ) (v_jill v_jack : ℕ),
  d = 2 →
  v_jill = 12 →
  v_jack = 3 →
  ((d / v_jack) * 60 - (d / v_jill) * 60) = 30 :=
by
  intros d v_jill v_jack hd hvjill hvjack
  sorry

end NUMINAMATH_GPT_jill_arrives_earlier_by_30_minutes_l358_35823


namespace NUMINAMATH_GPT_number_of_sheep_total_number_of_animals_l358_35890

theorem number_of_sheep (ratio_sh_horse : 5 / 7 * horses = sheep) 
    (horse_food_per_day : horses * 230 = 12880) :
    sheep = 40 :=
by
  -- These are all the given conditions
  sorry

theorem total_number_of_animals (sheep : ℕ) (horses : ℕ)
    (H1 : sheep = 40) (H2 : horses = 56) :
    sheep + horses = 96 :=
by
  -- Given conditions for the total number of animals on the farm
  sorry

end NUMINAMATH_GPT_number_of_sheep_total_number_of_animals_l358_35890


namespace NUMINAMATH_GPT_add_to_1_eq_62_l358_35819

theorem add_to_1_eq_62 :
  let y := 5 * 12 / (180 / 3)
  ∃ x, y + x = 62 ∧ x = 61 :=
by
  sorry

end NUMINAMATH_GPT_add_to_1_eq_62_l358_35819


namespace NUMINAMATH_GPT_mult_xy_eq_200_over_3_l358_35838

def hash_op (a b : ℚ) : ℚ := a + a / b

def x : ℚ := hash_op 8 3

def y : ℚ := hash_op 5 4

theorem mult_xy_eq_200_over_3 : x * y = 200 / 3 := 
by 
  -- lean uses real division operator, and hash_op must remain rational
  sorry

end NUMINAMATH_GPT_mult_xy_eq_200_over_3_l358_35838


namespace NUMINAMATH_GPT_remainder_of_number_mod_1000_l358_35853

-- Definitions according to the conditions
def num_increasing_8_digit_numbers_with_zero : ℕ := Nat.choose 17 8

-- The main statement to be proved
theorem remainder_of_number_mod_1000 : 
  (num_increasing_8_digit_numbers_with_zero % 1000) = 310 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_number_mod_1000_l358_35853


namespace NUMINAMATH_GPT_geometric_sequence_increasing_neither_sufficient_nor_necessary_l358_35825

-- Definitions based on the conditions
def is_geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop := ∀ n, a (n + 1) = a n * q
def is_increasing_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) > a n

-- Define the main theorem according to the problem statement
theorem geometric_sequence_increasing_neither_sufficient_nor_necessary (a : ℕ → ℝ) (a1 q : ℝ) 
  (h_geom : is_geometric_sequence a a1 q) :
  ¬ ( ( (∀ (h : a1 * q > 0), is_increasing_sequence a) ∨ 
        (∀ (h : is_increasing_sequence a), a1 * q > 0) ) ) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_increasing_neither_sufficient_nor_necessary_l358_35825


namespace NUMINAMATH_GPT_range_of_k_l358_35817

theorem range_of_k (k : ℝ) :
  (∃ a b c : ℝ, (a = 1) ∧ (b = -1) ∧ (c = -k) ∧ (b^2 - 4 * a * c > 0)) ↔ k > -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l358_35817


namespace NUMINAMATH_GPT_service_center_milepost_l358_35834

theorem service_center_milepost :
  let mp4 := 50
  let mp12 := 190
  let service_center := mp4 + (mp12 - mp4) / 2
  service_center = 120 :=
by
  let mp4 := 50
  let mp12 := 190
  let service_center := mp4 + (mp12 - mp4) / 2
  sorry

end NUMINAMATH_GPT_service_center_milepost_l358_35834


namespace NUMINAMATH_GPT_missed_questions_l358_35893

-- Define variables
variables (a b c T : ℕ) (X Y Z : ℝ)
variables (h1 : a + b + c = T) 
          (h2 : 0 ≤ X ∧ X ≤ 100) 
          (h3 : 0 ≤ Y ∧ Y ≤ 100) 
          (h4 : 0 ≤ Z ∧ Z ≤ 100) 
          (h5 : 6 * (a * (100 - X) / 500 + 2 * b * (100 - Y) / 500 + 3 * c * (100 - Z) / 500) = 216)

-- Define the theorem
theorem missed_questions : 5 * (a * (100 - X) / 500 + b * (100 - Y) / 500 + c * (100 - Z) / 500) = 180 :=
by sorry

end NUMINAMATH_GPT_missed_questions_l358_35893


namespace NUMINAMATH_GPT_total_handshakes_l358_35854

theorem total_handshakes (n : ℕ) (h : n = 10) : ∃ k, k = (n * (n - 1)) / 2 ∧ k = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_handshakes_l358_35854


namespace NUMINAMATH_GPT_cube_fraction_inequality_l358_35864

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end NUMINAMATH_GPT_cube_fraction_inequality_l358_35864


namespace NUMINAMATH_GPT_sum_of_intercepts_eq_16_l358_35899

noncomputable def line_eq (x y : ℝ) : Prop :=
  y + 3 = -3 * (x - 5)

def x_intercept : ℝ := 4
def y_intercept : ℝ := 12

theorem sum_of_intercepts_eq_16 : 
  (line_eq x_intercept 0) ∧ (line_eq 0 y_intercept) → x_intercept + y_intercept = 16 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sum_of_intercepts_eq_16_l358_35899


namespace NUMINAMATH_GPT_Johnson_potatoes_left_l358_35809

noncomputable def Gina_potatoes : ℝ := 93.5
noncomputable def Tom_potatoes : ℝ := 3.2 * Gina_potatoes
noncomputable def Anne_potatoes : ℝ := (2/3) * Tom_potatoes
noncomputable def Jack_potatoes : ℝ := (1/7) * (Gina_potatoes + Anne_potatoes)
noncomputable def Total_given_away : ℝ := Gina_potatoes + Tom_potatoes + Anne_potatoes + Jack_potatoes
noncomputable def Initial_potatoes : ℝ := 1250
noncomputable def Potatoes_left : ℝ := Initial_potatoes - Total_given_away

theorem Johnson_potatoes_left : Potatoes_left = 615.98 := 
  by
    sorry

end NUMINAMATH_GPT_Johnson_potatoes_left_l358_35809


namespace NUMINAMATH_GPT_sum_of_cubes_of_roots_eq_1_l358_35897

theorem sum_of_cubes_of_roots_eq_1 (a : ℝ) (x1 x2 : ℝ) :
  (x1^2 + a * x1 + a + 1 = 0) → 
  (x2^2 + a * x2 + a + 1 = 0) → 
  (x1 + x2 = -a) → 
  (x1 * x2 = a + 1) → 
  (x1^3 + x2^3 = 1) → 
  a = -1 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_of_roots_eq_1_l358_35897


namespace NUMINAMATH_GPT_triangle_side_ratios_l358_35804

theorem triangle_side_ratios
    (A B C : ℝ) (a b c : ℝ)
    (h1 : 2 * b * Real.sin (2 * A) = a * Real.sin B)
    (h2 : c = 2 * b) :
    a / b = 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_ratios_l358_35804


namespace NUMINAMATH_GPT_well_depth_l358_35830

theorem well_depth :
  (∃ t₁ t₂ : ℝ, t₁ + t₂ = 9.5 ∧ 20 * t₁ ^ 2 = d ∧ t₂ = d / 1000 ∧ d = 1332.25) :=
by
  sorry

end NUMINAMATH_GPT_well_depth_l358_35830


namespace NUMINAMATH_GPT_smallest_integer_l358_35816

theorem smallest_integer (k : ℕ) (n : ℕ) (h936 : 936 = 2^3 * 3^1 * 13^1)
  (h2 : 2^5 ∣ 936 * k)
  (h3 : 3^3 ∣ 936 * k)
  (h4 : 12^2 ∣ 936 * k) : 
  k = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_integer_l358_35816


namespace NUMINAMATH_GPT_isosceles_triangle_has_perimeter_22_l358_35867

noncomputable def isosceles_triangle_perimeter (a b : ℕ) : ℕ :=
if a + a > b ∧ a + b > a ∧ b + b > a then a + a + b else 0

theorem isosceles_triangle_has_perimeter_22 :
  isosceles_triangle_perimeter 9 4 = 22 :=
by 
  -- Add a note for clarity; this will be completed via 'sorry'
  -- Prove that with side lengths 9 and 4 (with 9 being the equal sides),
  -- they form a valid triangle and its perimeter is 22
  sorry

end NUMINAMATH_GPT_isosceles_triangle_has_perimeter_22_l358_35867


namespace NUMINAMATH_GPT_product_of_solutions_abs_eq_l358_35814

theorem product_of_solutions_abs_eq (x : ℝ) (h : |x - 5| + 4 = 7) : x * (if x = 8 then 2 else 8) = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_solutions_abs_eq_l358_35814


namespace NUMINAMATH_GPT_complement_of_M_with_respect_to_U_l358_35843

namespace Complements

open Set

def U : Set Int := {1, -2, 3, -4, 5, -6}
def M : Set Int := {1, -2, 3, -4}

theorem complement_of_M_with_respect_to_U :
  U \ M = {5, -6} :=
by
  sorry

end Complements

end NUMINAMATH_GPT_complement_of_M_with_respect_to_U_l358_35843


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_is_right_l358_35835

noncomputable def arithmetic_sequence_sum : ℤ :=
  let a1 := 1
  let d := -2
  let a2 := a1 + d
  let a3 := a1 + 2 * d
  let a6 := a1 + 5 * d
  let S6 := 6 * a1 + (6 * (6-1)) / 2 * d
  S6

theorem arithmetic_sequence_sum_is_right {d : ℤ} (h₀ : d ≠ 0) 
(h₁ : (a1 + 2 * d) ^ 2 = (a1 + d) * (a1 + 5 * d)) :
  arithmetic_sequence_sum = -24 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_is_right_l358_35835


namespace NUMINAMATH_GPT_binary_addition_l358_35806

theorem binary_addition :
  let num1 := 0b111111111
  let num2 := 0b101010101
  num1 + num2 = 852 := by
  sorry

end NUMINAMATH_GPT_binary_addition_l358_35806


namespace NUMINAMATH_GPT_angle_C_eq_pi_over_3_l358_35802

theorem angle_C_eq_pi_over_3 (a b c A B C : ℝ)
  (h : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)) :
  C = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_angle_C_eq_pi_over_3_l358_35802


namespace NUMINAMATH_GPT_octal_to_decimal_conversion_l358_35818

theorem octal_to_decimal_conversion : 
  let d8 := 8
  let f := fun (x: Nat) (y: Nat) => x * d8 ^ y
  7 * d8^0 + 6 * d8^1 + 3 * d8^2 = 247 := 
by
  let d8 := 8
  let f := fun (x: Nat) (y: Nat) => x * d8 ^ y
  sorry

end NUMINAMATH_GPT_octal_to_decimal_conversion_l358_35818


namespace NUMINAMATH_GPT_geometric_progression_l358_35813

theorem geometric_progression :
  ∃ (b1 q : ℚ), 
    (b1 * q * (q^2 - 1) = -45/32) ∧ 
    (b1 * q^3 * (q^2 - 1) = -45/512) ∧ 
    ((b1 = 6 ∧ q = 1/4) ∨ (b1 = -6 ∧ q = -1/4)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_l358_35813


namespace NUMINAMATH_GPT_find_b_l358_35894

noncomputable def curve (x : ℝ) : ℝ := x^3 - 3 * x^2
noncomputable def tangent_line (x b : ℝ) : ℝ := -3 * x + b

theorem find_b
  (b : ℝ)
  (h : ∃ x : ℝ, curve x = tangent_line x b ∧ deriv curve x = -3) :
  b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l358_35894
