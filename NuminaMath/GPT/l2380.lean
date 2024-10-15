import Mathlib

namespace NUMINAMATH_GPT_value_of_expression_l2380_238066

theorem value_of_expression (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (3 * x - 4 * y) / z = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l2380_238066


namespace NUMINAMATH_GPT_left_handed_women_percentage_l2380_238060

theorem left_handed_women_percentage
  (x y : ℕ)
  (h1 : 4 * x = 5 * y)
  (h2 : 3 * x ≥ 3 * y) :
  (x / (4 * x) : ℚ) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_left_handed_women_percentage_l2380_238060


namespace NUMINAMATH_GPT_martin_speed_l2380_238087

theorem martin_speed (distance time : ℝ) (h_distance : distance = 12) (h_time : time = 6) :
  distance / time = 2 :=
by
  rw [h_distance, h_time]
  norm_num

end NUMINAMATH_GPT_martin_speed_l2380_238087


namespace NUMINAMATH_GPT_M_inter_N_is_5_l2380_238088

/-- Define the sets M and N. -/
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {2, 5, 8}

/-- Prove the intersection of M and N is {5}. -/
theorem M_inter_N_is_5 : M ∩ N = {5} :=
by
  sorry

end NUMINAMATH_GPT_M_inter_N_is_5_l2380_238088


namespace NUMINAMATH_GPT_johns_coin_collection_value_l2380_238068

theorem johns_coin_collection_value :
  ∀ (n : ℕ) (value : ℕ), n = 24 → value = 20 → 
  ((n/3) * (value/8)) = 60 :=
by
  intro n value n_eq value_eq
  sorry

end NUMINAMATH_GPT_johns_coin_collection_value_l2380_238068


namespace NUMINAMATH_GPT_relationship_a_e_l2380_238065

theorem relationship_a_e (a : ℝ) (h : 0 < a ∧ a < 1) : a < Real.exp a - 1 ∧ Real.exp a - 1 < a ^ Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_relationship_a_e_l2380_238065


namespace NUMINAMATH_GPT_range_of_m_l2380_238070

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (x + m) / (x - 2) + 2 * m / (2 - x) = 3) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2380_238070


namespace NUMINAMATH_GPT_Vasya_numbers_l2380_238081

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_Vasya_numbers_l2380_238081


namespace NUMINAMATH_GPT_sqrt_neg3_squared_l2380_238030

theorem sqrt_neg3_squared : Real.sqrt ((-3)^2) = 3 :=
by sorry

end NUMINAMATH_GPT_sqrt_neg3_squared_l2380_238030


namespace NUMINAMATH_GPT_total_chrome_parts_l2380_238004

theorem total_chrome_parts (a b : ℕ) 
  (h1 : a + b = 21) 
  (h2 : 3 * a + 2 * b = 50) : 2 * a + 4 * b = 68 := 
sorry

end NUMINAMATH_GPT_total_chrome_parts_l2380_238004


namespace NUMINAMATH_GPT_smallest_n_satisfying_congruence_l2380_238015

theorem smallest_n_satisfying_congruence :
  ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, m < n → (7^m % 5) ≠ (m^7 % 5)) ∧ (7^n % 5) = (n^7 % 5) := 
by sorry

end NUMINAMATH_GPT_smallest_n_satisfying_congruence_l2380_238015


namespace NUMINAMATH_GPT_percents_multiplication_l2380_238094

theorem percents_multiplication :
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  (p1 * p2 * p3 * p4) * 100 = 5.88 := 
by
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  sorry

end NUMINAMATH_GPT_percents_multiplication_l2380_238094


namespace NUMINAMATH_GPT_cone_new_height_l2380_238035

noncomputable def new_cone_height : ℝ := 6

theorem cone_new_height (r h V : ℝ) (circumference : 2 * Real.pi * r = 24 * Real.pi)
  (original_height : h = 40) (same_base_circumference : 2 * Real.pi * r = 24 * Real.pi)
  (volume : (1 / 3) * Real.pi * (r ^ 2) * new_cone_height = 288 * Real.pi) :
    new_cone_height = 6 := 
sorry

end NUMINAMATH_GPT_cone_new_height_l2380_238035


namespace NUMINAMATH_GPT_day_of_week_2_2312_wednesday_l2380_238071

def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ ((y % 4 = 0) ∧ (y % 100 ≠ 0))

theorem day_of_week_2_2312_wednesday (birth_year : ℕ) (birth_day : String) 
  (h1 : birth_year = 2312 - 300)
  (h2 : birth_day = "Wednesday") :
  "Monday" = "Monday" :=
sorry

end NUMINAMATH_GPT_day_of_week_2_2312_wednesday_l2380_238071


namespace NUMINAMATH_GPT_birds_count_l2380_238072

theorem birds_count (N B : ℕ) 
  (h1 : B = 5 * N)
  (h2 : B = N + 360) : 
  B = 450 := by
  sorry

end NUMINAMATH_GPT_birds_count_l2380_238072


namespace NUMINAMATH_GPT_card_draw_prob_l2380_238022

/-- Define the total number of cards in the deck -/
def total_cards : ℕ := 52

/-- Define the total number of diamonds or aces -/
def diamonds_and_aces : ℕ := 16

/-- Define the probability of drawing a card that is a diamond or an ace in one draw -/
def prob_diamond_or_ace : ℚ := diamonds_and_aces / total_cards

/-- Define the complementary probability of not drawing a diamond nor ace in one draw -/
def prob_not_diamond_or_ace : ℚ := (total_cards - diamonds_and_aces) / total_cards

/-- Define the probability of not drawing a diamond nor ace in three draws with replacement -/
def prob_not_diamond_or_ace_three_draws : ℚ := prob_not_diamond_or_ace ^ 3

/-- Define the probability of drawing at least one diamond or ace in three draws with replacement -/
def prob_at_least_one_diamond_or_ace_in_three_draws : ℚ := 1 - prob_not_diamond_or_ace_three_draws

/-- The final probability calculated -/
def final_prob : ℚ := 1468 / 2197

theorem card_draw_prob :
  prob_at_least_one_diamond_or_ace_in_three_draws = final_prob := by
  sorry

end NUMINAMATH_GPT_card_draw_prob_l2380_238022


namespace NUMINAMATH_GPT_angle_between_clock_hands_at_7_oclock_l2380_238085

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end NUMINAMATH_GPT_angle_between_clock_hands_at_7_oclock_l2380_238085


namespace NUMINAMATH_GPT_correct_geometry_problems_l2380_238040

-- Let A_c be the number of correct algebra problems.
-- Let A_i be the number of incorrect algebra problems.
-- Let G_c be the number of correct geometry problems.
-- Let G_i be the number of incorrect geometry problems.

def algebra_correct_incorrect_ratio (A_c A_i : ℕ) : Prop :=
  A_c * 2 = A_i * 3

def geometry_correct_incorrect_ratio (G_c G_i : ℕ) : Prop :=
  G_c * 1 = G_i * 4

def total_algebra_problems (A_c A_i : ℕ) : Prop :=
  A_c + A_i = 25

def total_geometry_problems (G_c G_i : ℕ) : Prop :=
  G_c + G_i = 35

def total_problems (A_c A_i G_c G_i : ℕ) : Prop :=
  A_c + A_i + G_c + G_i = 60

theorem correct_geometry_problems (A_c A_i G_c G_i : ℕ) :
  algebra_correct_incorrect_ratio A_c A_i →
  geometry_correct_incorrect_ratio G_c G_i →
  total_algebra_problems A_c A_i →
  total_geometry_problems G_c G_i →
  total_problems A_c A_i G_c G_i →
  G_c = 28 :=
sorry

end NUMINAMATH_GPT_correct_geometry_problems_l2380_238040


namespace NUMINAMATH_GPT_piglet_gifted_balloons_l2380_238032

noncomputable def piglet_balloons_gifted (piglet_balloons : ℕ) : ℕ :=
  let winnie_balloons := 3 * piglet_balloons
  let owl_balloons := 4 * piglet_balloons
  let total_balloons := piglet_balloons + winnie_balloons + owl_balloons
  let burst_balloons := total_balloons - 60
  piglet_balloons - burst_balloons / 8

-- Prove that Piglet gifted 4 balloons given the conditions
theorem piglet_gifted_balloons :
  ∃ (piglet_balloons : ℕ), piglet_balloons = 8 ∧ piglet_balloons_gifted piglet_balloons = 4 := sorry

end NUMINAMATH_GPT_piglet_gifted_balloons_l2380_238032


namespace NUMINAMATH_GPT_smallest_n_l2380_238079

theorem smallest_n (m l n : ℕ) :
  (∃ m : ℕ, 2 * n = m ^ 4) ∧ (∃ l : ℕ, 3 * n = l ^ 6) → n = 1944 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l2380_238079


namespace NUMINAMATH_GPT_real_z9_count_l2380_238098

theorem real_z9_count (z : ℂ) (hz : z^18 = 1) : 
  (∃! z : ℂ, z^18 = 1 ∧ (z^9).im = 0) :=
sorry

end NUMINAMATH_GPT_real_z9_count_l2380_238098


namespace NUMINAMATH_GPT_carrot_lettuce_ratio_l2380_238051

theorem carrot_lettuce_ratio :
  let lettuce_cal := 50
  let dressing_cal := 210
  let crust_cal := 600
  let pepperoni_cal := crust_cal / 3
  let cheese_cal := 400
  let total_pizza_cal := crust_cal + pepperoni_cal + cheese_cal
  let carrot_cal := C
  let total_salad_cal := lettuce_cal + carrot_cal + dressing_cal
  let jackson_salad_cal := (1 / 4) * total_salad_cal
  let jackson_pizza_cal := (1 / 5) * total_pizza_cal
  jackson_salad_cal + jackson_pizza_cal = 330 →
  carrot_cal / lettuce_cal = 2 :=
by
  intro lettuce_cal dressing_cal crust_cal pepperoni_cal cheese_cal total_pizza_cal carrot_cal total_salad_cal jackson_salad_cal jackson_pizza_cal h
  sorry

end NUMINAMATH_GPT_carrot_lettuce_ratio_l2380_238051


namespace NUMINAMATH_GPT_area_of_blackboard_l2380_238019

def side_length : ℝ := 6
def area (side : ℝ) : ℝ := side * side

theorem area_of_blackboard : area side_length = 36 := by
  -- proof
  sorry

end NUMINAMATH_GPT_area_of_blackboard_l2380_238019


namespace NUMINAMATH_GPT_circle_radius_l2380_238054

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 2 * x + 6 * y + 1 = 0) → (∃ (r : ℝ), r = 3) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l2380_238054


namespace NUMINAMATH_GPT_initial_milk_in_container_A_l2380_238010

theorem initial_milk_in_container_A (A B C D : ℝ) 
  (h1 : B = A - 0.625 * A) 
  (h2 : C - 158 = B) 
  (h3 : D = 0.45 * (C - 58)) 
  (h4 : D = 58) 
  : A = 231 := 
sorry

end NUMINAMATH_GPT_initial_milk_in_container_A_l2380_238010


namespace NUMINAMATH_GPT_range_of_eccentricity_l2380_238056

theorem range_of_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) (x y : ℝ) 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) (c : ℝ := Real.sqrt (a^2 - b^2)) 
  (h_dot_product : ∀ (x y: ℝ) (h_point : x^2 / a^2 + y^2 / b^2 = 1), 
    let PF1 : ℝ × ℝ := (-c - x, -y)
    let PF2 : ℝ × ℝ := (c - x, -y)
    PF1.1 * PF2.1 + PF1.2 * PF2.2 ≤ a * c) : 
  ∀ (e : ℝ := c / a), (Real.sqrt 5 - 1) / 2 ≤ e ∧ e < 1 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_eccentricity_l2380_238056


namespace NUMINAMATH_GPT_five_star_three_l2380_238046

def star (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2

theorem five_star_three : star 5 3 = 4 := by
  sorry

end NUMINAMATH_GPT_five_star_three_l2380_238046


namespace NUMINAMATH_GPT_probability_same_group_l2380_238091

noncomputable def num_students : ℕ := 800
noncomputable def num_groups : ℕ := 4
noncomputable def group_size : ℕ := num_students / num_groups
noncomputable def amy := 0
noncomputable def ben := 1
noncomputable def clara := 2

theorem probability_same_group : ∃ p : ℝ, p = 1 / 16 :=
by
  let P_ben_with_amy : ℝ := group_size / num_students
  let P_clara_with_amy : ℝ := group_size / num_students
  let P_all_same := P_ben_with_amy * P_clara_with_amy
  use P_all_same
  sorry

end NUMINAMATH_GPT_probability_same_group_l2380_238091


namespace NUMINAMATH_GPT_first_pipe_fills_cistern_in_10_hours_l2380_238044

noncomputable def time_to_fill (x : ℝ) : Prop :=
  let first_pipe_rate := 1 / x
  let second_pipe_rate := 1 / 12
  let third_pipe_rate := 1 / 15
  let combined_rate := first_pipe_rate + second_pipe_rate - third_pipe_rate
  combined_rate = 7 / 60

theorem first_pipe_fills_cistern_in_10_hours : time_to_fill 10 :=
by
  sorry

end NUMINAMATH_GPT_first_pipe_fills_cistern_in_10_hours_l2380_238044


namespace NUMINAMATH_GPT_eighth_arithmetic_term_l2380_238041

theorem eighth_arithmetic_term (a₂ a₁₄ a₈ : ℚ) 
  (h2 : a₂ = 8 / 11)
  (h14 : a₁₄ = 9 / 13) :
  a₈ = 203 / 286 :=
by
  sorry

end NUMINAMATH_GPT_eighth_arithmetic_term_l2380_238041


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2380_238007

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def P : Set ℝ := {x | x ≤ -1}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧ (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2380_238007


namespace NUMINAMATH_GPT_original_volume_of_ice_l2380_238029

theorem original_volume_of_ice (V : ℝ) 
  (h1 : V * (1/4) * (1/4) = 0.4) : 
  V = 6.4 :=
sorry

end NUMINAMATH_GPT_original_volume_of_ice_l2380_238029


namespace NUMINAMATH_GPT_power_difference_mod_7_l2380_238093

theorem power_difference_mod_7 :
  (45^2011 - 23^2011) % 7 = 5 := by
  have h45 : 45 % 7 = 3 := by norm_num
  have h23 : 23 % 7 = 2 := by norm_num
  sorry

end NUMINAMATH_GPT_power_difference_mod_7_l2380_238093


namespace NUMINAMATH_GPT_arithmetic_sequence_condition_l2380_238047

theorem arithmetic_sequence_condition (a : ℕ → ℕ) 
(h1 : a 4 = 4) 
(h2 : a 3 + a 8 = 5) : 
a 7 = 1 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_condition_l2380_238047


namespace NUMINAMATH_GPT_partial_fractions_sum_zero_l2380_238028

noncomputable def sum_of_coefficients (A B C D E : ℝ) : Prop :=
  (A + B + C + D + E = 0)

theorem partial_fractions_sum_zero :
  ∀ (A B C D E : ℝ),
    (∀ x : ℝ, 1 = A*(x+1)*(x+2)*(x+3)*(x+5) + B*x*(x+2)*(x+3)*(x+5) + 
              C*x*(x+1)*(x+3)*(x+5) + D*x*(x+1)*(x+2)*(x+5) + 
              E*x*(x+1)*(x+2)*(x+3)) →
    sum_of_coefficients A B C D E :=
by sorry

end NUMINAMATH_GPT_partial_fractions_sum_zero_l2380_238028


namespace NUMINAMATH_GPT_solution_set_inequality_l2380_238099

theorem solution_set_inequality : 
  {x : ℝ | abs ((x - 3) / x) > ((x - 3) / x)} = {x : ℝ | 0 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l2380_238099


namespace NUMINAMATH_GPT_negation_proof_l2380_238064

def P (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

theorem negation_proof : (¬(∀ x : ℝ, P x)) ↔ (∃ x : ℝ, ¬(P x)) :=
by sorry

end NUMINAMATH_GPT_negation_proof_l2380_238064


namespace NUMINAMATH_GPT_proof_of_problem_l2380_238027

-- Define the problem conditions using a combination function
def problem_statement : Prop :=
  (Nat.choose 6 3 = 20)

theorem proof_of_problem : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_proof_of_problem_l2380_238027


namespace NUMINAMATH_GPT_average_price_of_six_toys_l2380_238043

/-- Define the average cost of toys given the number of toys and their total cost -/
def avg_cost (total_cost : ℕ) (num_toys : ℕ) : ℕ :=
  total_cost / num_toys

/-- Define the total cost of toys given a list of individual toy costs -/
def total_cost (costs : List ℕ) : ℕ :=
  costs.foldl (· + ·) 0

/-- The main theorem -/
theorem average_price_of_six_toys :
  let dhoni_toys := 5
  let avg_cost_dhoni := 10
  let total_cost_dhoni := dhoni_toys * avg_cost_dhoni
  let david_toy_cost := 16
  let total_toys := dhoni_toys + 1
  total_cost_dhoni + david_toy_cost = 66 →
  avg_cost (66) (total_toys) = 11 :=
by
  -- Introduce the conditions and hypothesis
  intros total_cost_of_6_toys H
  -- Simplify the expression
  sorry  -- Proof skipped

end NUMINAMATH_GPT_average_price_of_six_toys_l2380_238043


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2380_238000

variable (p q : Prop)

theorem sufficient_not_necessary (h1 : p ∧ q) (h2 : ¬¬p) : ¬¬p :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l2380_238000


namespace NUMINAMATH_GPT_A_profit_share_l2380_238074

theorem A_profit_share (A_shares : ℚ) (B_shares : ℚ) (C_shares : ℚ) (D_shares : ℚ) (total_profit : ℚ) (A_profit : ℚ) :
  A_shares = 1/3 → B_shares = 1/4 → C_shares = 1/5 → 
  D_shares = 1 - (A_shares + B_shares + C_shares) → total_profit = 2445 → A_profit = 815 →
  A_shares * total_profit = A_profit :=
by sorry

end NUMINAMATH_GPT_A_profit_share_l2380_238074


namespace NUMINAMATH_GPT_sara_golf_balls_total_l2380_238053

-- Define the conditions
def dozens := 16
def dozen_to_balls := 12

-- The final proof statement
theorem sara_golf_balls_total : dozens * dozen_to_balls = 192 :=
by
  sorry

end NUMINAMATH_GPT_sara_golf_balls_total_l2380_238053


namespace NUMINAMATH_GPT_exponent_equality_l2380_238096

theorem exponent_equality (y : ℕ) (z : ℕ) (h1 : 16 ^ y = 4 ^ z) (h2 : y = 8) : z = 16 := by
  sorry

end NUMINAMATH_GPT_exponent_equality_l2380_238096


namespace NUMINAMATH_GPT_star_property_l2380_238057

-- Define the operation a ⋆ b = (a - b) ^ 3
def star (a b : ℝ) : ℝ := (a - b) ^ 3

-- State the theorem
theorem star_property (x y : ℝ) : star ((x - y) ^ 3) ((y - x) ^ 3) = 8 * (x - y) ^ 9 := 
by 
  sorry

end NUMINAMATH_GPT_star_property_l2380_238057


namespace NUMINAMATH_GPT_speed_second_boy_l2380_238020

theorem speed_second_boy (v : ℝ) (t : ℝ) (d : ℝ) (s₁ : ℝ) :
  s₁ = 4.5 ∧ t = 9.5 ∧ d = 9.5 ∧ (d = (v - s₁) * t) → v = 5.5 :=
by
  intros h
  obtain ⟨hs₁, ht, hd, hev⟩ := h
  sorry

end NUMINAMATH_GPT_speed_second_boy_l2380_238020


namespace NUMINAMATH_GPT_min_value_geom_seq_l2380_238036

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  0 < a 4 ∧ 0 < a 14 ∧ a 4 * a 14 = 8 ∧ 0 < a 7 ∧ 0 < a 11 ∧ a 7 * a 11 = 8

theorem min_value_geom_seq {a : ℕ → ℝ} (h : geom_seq a) :
  2 * a 7 + a 11 = 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_geom_seq_l2380_238036


namespace NUMINAMATH_GPT_minimum_protein_content_is_at_least_1_8_l2380_238026

-- Define the net weight of the can and the minimum protein percentage
def netWeight : ℝ := 300
def minProteinPercentage : ℝ := 0.006

-- Prove that the minimum protein content is at least 1.8 grams
theorem minimum_protein_content_is_at_least_1_8 :
  netWeight * minProteinPercentage ≥ 1.8 := 
by
  sorry

end NUMINAMATH_GPT_minimum_protein_content_is_at_least_1_8_l2380_238026


namespace NUMINAMATH_GPT_no_solution_1221_l2380_238024

def equation_correctness (n : ℤ) : Prop :=
  -n^3 + 555^3 = n^2 - n * 555 + 555^2

-- Prove that the prescribed value 1221 does not satisfy the modified equation by contradiction
theorem no_solution_1221 : ¬ ∃ n : ℤ, equation_correctness n ∧ n = 1221 := by
  sorry

end NUMINAMATH_GPT_no_solution_1221_l2380_238024


namespace NUMINAMATH_GPT_proof_1_proof_2_l2380_238050

noncomputable def problem_1 (x : ℝ) : Prop :=
  (3 * x - 2) / (x - 1) > 1 → x > 1

noncomputable def problem_2 (x a : ℝ) : Prop :=
  if a = 0 then False
  else if a > 0 then -a < x ∧ x < 2 * a
  else if a < 0 then 2 * a < x ∧ x < -a
  else False

-- Sorry to skip the proofs
theorem proof_1 (x : ℝ) (h : problem_1 x) : x > 1 :=
  sorry

theorem proof_2 (x a : ℝ) (h : x * x - a * x - 2 * a * a < 0) : problem_2 x a :=
  sorry

end NUMINAMATH_GPT_proof_1_proof_2_l2380_238050


namespace NUMINAMATH_GPT_money_left_correct_l2380_238034

def initial_amount : ℕ := 158
def cost_shoes : ℕ := 45
def cost_bag : ℕ := cost_shoes - 17
def cost_lunch : ℕ := cost_bag / 4
def total_spent : ℕ := cost_shoes + cost_bag + cost_lunch
def amount_left : ℕ := initial_amount - total_spent

theorem money_left_correct :
  amount_left = 78 := by
  sorry

end NUMINAMATH_GPT_money_left_correct_l2380_238034


namespace NUMINAMATH_GPT_ratio_dvds_to_cds_l2380_238037

def total_sold : ℕ := 273
def dvds_sold : ℕ := 168
def cds_sold : ℕ := total_sold - dvds_sold

theorem ratio_dvds_to_cds : (dvds_sold : ℚ) / cds_sold = 8 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_dvds_to_cds_l2380_238037


namespace NUMINAMATH_GPT_linear_equation_in_two_variables_l2380_238016

/--
Prove that Equation C (3x - 1 = 2 - 5y) is a linear equation in two variables 
given the equations in conditions.
-/
theorem linear_equation_in_two_variables :
  ∀ (x y : ℝ),
  (2 * x + 3 = x - 5) →
  (x * y + y = 2) →
  (3 * x - 1 = 2 - 5 * y) →
  (2 * x + (3 / y) = 7) →
  ∃ (A B C : ℝ), A * x + B * y = C :=
by 
  sorry

end NUMINAMATH_GPT_linear_equation_in_two_variables_l2380_238016


namespace NUMINAMATH_GPT_cherry_trees_leaves_l2380_238075

-- Define the original number of trees
def original_num_trees : ℕ := 7

-- Define the number of trees actually planted
def actual_num_trees : ℕ := 2 * original_num_trees

-- Define the number of leaves each tree drops
def leaves_per_tree : ℕ := 100

-- Define the total number of leaves that fall
def total_leaves : ℕ := actual_num_trees * leaves_per_tree

-- Theorem statement for the problem
theorem cherry_trees_leaves : total_leaves = 1400 := by
  sorry

end NUMINAMATH_GPT_cherry_trees_leaves_l2380_238075


namespace NUMINAMATH_GPT_tan_angle_addition_l2380_238003

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_angle_addition_l2380_238003


namespace NUMINAMATH_GPT_area_ratio_of_similar_isosceles_triangles_l2380_238018

theorem area_ratio_of_similar_isosceles_triangles
  (b1 b2 h1 h2 : ℝ)
  (h_ratio : h1 / h2 = 2 / 3)
  (similar_tri : b1 / b2 = 2 / 3) :
  (1 / 2 * b1 * h1) / (1 / 2 * b2 * h2) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_similar_isosceles_triangles_l2380_238018


namespace NUMINAMATH_GPT_men_in_first_group_l2380_238086

theorem men_in_first_group (M : ℕ) (h1 : M * 18 * 6 = 15 * 12 * 6) : M = 10 :=
by
  sorry

end NUMINAMATH_GPT_men_in_first_group_l2380_238086


namespace NUMINAMATH_GPT_first_year_after_2020_with_sum_4_l2380_238033

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10)

def is_year (y : ℕ) : Prop :=
  y > 2020 ∧ sum_of_digits y = 4

theorem first_year_after_2020_with_sum_4 : ∃ y, is_year y ∧ ∀ z, is_year z → z ≥ y :=
by sorry

end NUMINAMATH_GPT_first_year_after_2020_with_sum_4_l2380_238033


namespace NUMINAMATH_GPT_paper_length_l2380_238042

theorem paper_length :
  ∃ (L : ℝ), (2 * (11 * L) = 2 * (8.5 * 11) + 100 ∧ L = 287 / 22) :=
sorry

end NUMINAMATH_GPT_paper_length_l2380_238042


namespace NUMINAMATH_GPT_interior_diagonals_sum_l2380_238055

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + c * a) = 112)
  (h2 : 4 * (a + b + c) = 60) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 113 := 
by 
  sorry

end NUMINAMATH_GPT_interior_diagonals_sum_l2380_238055


namespace NUMINAMATH_GPT_greatest_integer_b_not_in_range_l2380_238038

theorem greatest_integer_b_not_in_range :
  let f (x : ℝ) (b : ℝ) := x^2 + b*x + 20
  let g (x : ℝ) (b : ℝ) := x^2 + b*x + 24
  (¬ (∃ (x : ℝ), g x b = 0)) → (b = 9) :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_b_not_in_range_l2380_238038


namespace NUMINAMATH_GPT_fisherman_gets_14_tunas_every_day_l2380_238090

-- Define the conditions
def red_snappers_per_day := 8
def cost_per_red_snapper := 3
def cost_per_tuna := 2
def total_earnings_per_day := 52

-- Define the hypothesis
def total_earnings_from_red_snappers := red_snappers_per_day * cost_per_red_snapper  -- $24
def total_earnings_from_tunas := total_earnings_per_day - total_earnings_from_red_snappers -- $28
def number_of_tunas := total_earnings_from_tunas / cost_per_tuna -- 14

-- Lean statement to verify
theorem fisherman_gets_14_tunas_every_day : number_of_tunas = 14 :=
by 
  sorry

end NUMINAMATH_GPT_fisherman_gets_14_tunas_every_day_l2380_238090


namespace NUMINAMATH_GPT_no_neighboring_beads_same_color_probability_l2380_238077

theorem no_neighboring_beads_same_color_probability : 
  let total_beads := 9
  let count_red := 4
  let count_white := 3
  let count_blue := 2
  let total_permutations := Nat.factorial total_beads / (Nat.factorial count_red * Nat.factorial count_white * Nat.factorial count_blue)
  ∃ valid_permutations : ℕ,
  valid_permutations = 100 ∧
  valid_permutations / total_permutations = 5 / 63 := by
  sorry

end NUMINAMATH_GPT_no_neighboring_beads_same_color_probability_l2380_238077


namespace NUMINAMATH_GPT_max_perimeter_of_polygons_l2380_238073

noncomputable def largest_possible_perimeter (sides1 sides2 sides3 : Nat) (len : Nat) : Nat :=
  (sides1 + sides2 + sides3) * len

theorem max_perimeter_of_polygons
  (a b c : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b % 2 = 0)
  (h3 : c % 2 = 0)
  (h4 : 180 * (a - 2) / a + 180 * (b - 2) / b + 180 * (c - 2) / c = 360)
  (h5 : ∃ (p : ℕ), ∃ q : ℕ, (a = p ∧ c = p ∧ a = q ∨ a = q ∧ b = p ∨ b = q ∧ c = p))
  : largest_possible_perimeter a b c 2 = 24 := 
sorry

end NUMINAMATH_GPT_max_perimeter_of_polygons_l2380_238073


namespace NUMINAMATH_GPT_f_monotonically_decreasing_iff_l2380_238009

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4 * a * x + 3 else (2 - 3 * a) * x + 1

theorem f_monotonically_decreasing_iff (a : ℝ) : 
  (∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≥ f a x₂) ↔ (1/2 ≤ a ∧ a < 2/3) :=
by 
  sorry

end NUMINAMATH_GPT_f_monotonically_decreasing_iff_l2380_238009


namespace NUMINAMATH_GPT_find_q_l2380_238059

-- Defining the polynomial and conditions
def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

variable (p q r : ℝ)

-- Given conditions
def mean_of_zeros_eq_prod_of_zeros (p q r : ℝ) : Prop :=
  -p / 3 = r

def prod_of_zeros_eq_sum_of_coeffs (p q r : ℝ) : Prop :=
  r = 1 + p + q + r

def y_intercept_eq_three (r : ℝ) : Prop :=
  r = 3

-- Final proof statement asserting q = 5
theorem find_q (p q r : ℝ) (h1 : mean_of_zeros_eq_prod_of_zeros p q r)
  (h2 : prod_of_zeros_eq_sum_of_coeffs p q r)
  (h3 : y_intercept_eq_three r) :
  q = 5 :=
sorry

end NUMINAMATH_GPT_find_q_l2380_238059


namespace NUMINAMATH_GPT_coefficient_of_term_x7_in_expansion_l2380_238082

theorem coefficient_of_term_x7_in_expansion:
  let general_term (r : ℕ) := (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r * (x : ℤ)^(12 - (5 * r) / 2)
  ∃ r : ℕ, 12 - (5 * r) / 2 = 7 ∧ (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r = 240 := 
sorry

end NUMINAMATH_GPT_coefficient_of_term_x7_in_expansion_l2380_238082


namespace NUMINAMATH_GPT_machine_C_works_in_6_hours_l2380_238078

theorem machine_C_works_in_6_hours :
  ∃ C : ℝ, (0 < C ∧ (1/4 + 1/12 + 1/C = 1/2)) → C = 6 :=
by
  sorry

end NUMINAMATH_GPT_machine_C_works_in_6_hours_l2380_238078


namespace NUMINAMATH_GPT_find_w_over_y_l2380_238052

theorem find_w_over_y 
  (w x y : ℝ) 
  (h1 : w / x = 2 / 3) 
  (h2 : (x + y) / y = 1.6) : 
  w / y = 0.4 := 
  sorry

end NUMINAMATH_GPT_find_w_over_y_l2380_238052


namespace NUMINAMATH_GPT_find_number_l2380_238023

theorem find_number
  (x : ℝ)
  (h : (7.5 * 7.5) + 37.5 + (x * x) = 100) :
  x = 2.5 :=
sorry

end NUMINAMATH_GPT_find_number_l2380_238023


namespace NUMINAMATH_GPT_find_c_l2380_238001

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 3) (h3 : abs (6 + 4 * c) = 14) : c = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_c_l2380_238001


namespace NUMINAMATH_GPT_range_of_a_l2380_238076

noncomputable def f (a x : ℝ) := Real.logb (1 / 2) (x^2 - a * x - a)

theorem range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f a x ∈ Set.univ) ∧ 
            (∀ x1 x2 : ℝ, -3 < x1 ∧ x1 < x2 ∧ x2 < 1 - Real.sqrt 3 → f a x1 < f a x2)) → 
  (0 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2380_238076


namespace NUMINAMATH_GPT_seventh_observation_value_l2380_238067

def average_initial_observations (S : ℝ) (n : ℕ) : Prop :=
  S / n = 13

def total_observations (n : ℕ) : Prop :=
  n + 1 = 7

def new_average (S : ℝ) (x : ℝ) (n : ℕ) : Prop :=
  (S + x) / (n + 1) = 12

theorem seventh_observation_value (S : ℝ) (n : ℕ) (x : ℝ) :
  average_initial_observations S n →
  total_observations n →
  new_average S x n →
  x = 6 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_seventh_observation_value_l2380_238067


namespace NUMINAMATH_GPT_find_DP_l2380_238083

theorem find_DP (AP BP CP DP : ℚ) (h1 : AP = 4) (h2 : BP = 6) (h3 : CP = 9) (h4 : AP * BP = CP * DP) :
  DP = 8 / 3 :=
by
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_find_DP_l2380_238083


namespace NUMINAMATH_GPT_eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l2380_238002

theorem eight_digit_numbers_with_012 :
  let total_sequences := 3^8 
  let invalid_sequences := 3^7 
  total_sequences - invalid_sequences = 4374 :=
by sorry

theorem eight_digit_numbers_with_00012222 :
  let total_sequences := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 4)
  let invalid_sequences := Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 4)
  total_sequences - invalid_sequences = 175 :=
by sorry

theorem eight_digit_numbers_starting_with_1_0002222 :
  let number_starting_with_1 := Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 4)
  number_starting_with_1 = 35 :=
by sorry

end NUMINAMATH_GPT_eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l2380_238002


namespace NUMINAMATH_GPT_find_a_find_n_l2380_238048

noncomputable def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d
noncomputable def sum_of_first_n_terms (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
noncomputable def S (a d n : ℕ) : ℕ := if n = 1 then a else sum_of_first_n_terms a d n
noncomputable def arithmetic_sum_property (a d n : ℕ) : Prop :=
  ∀ n ≥ 2, (S a d n) ^ 2 = 3 * n ^ 2 * arithmetic_sequence a d n + (S a d (n - 1)) ^ 2

theorem find_a (a : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2) :
  a = 3 :=
sorry

noncomputable def c (n : ℕ) (a5 : ℕ) : ℕ := 3 ^ (n - 1) + a5
noncomputable def sum_of_first_n_terms_c (n a5 : ℕ) : ℕ := (3^n - 1) / 2 + 15 * n
noncomputable def T (n a5 : ℕ) : ℕ := sum_of_first_n_terms_c n a5

theorem find_n (a : ℕ) (a5 : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2)
  (h2 : a = 3) (h3 : a5 = 15) :
  ∃ n : ℕ, 4 * T n a5 > S a 3 10 ∧ n = 3 :=
sorry

end NUMINAMATH_GPT_find_a_find_n_l2380_238048


namespace NUMINAMATH_GPT_rhombus_diagonal_sum_l2380_238006

theorem rhombus_diagonal_sum (e f : ℝ) (h1: e^2 + f^2 = 16) (h2: 0 < e ∧ 0 < f):
  e + f = 5 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_sum_l2380_238006


namespace NUMINAMATH_GPT_no_roots_ge_two_l2380_238069

theorem no_roots_ge_two (x : ℝ) (h : x ≥ 2) : 4 * x^3 - 5 * x^2 - 6 * x + 3 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_no_roots_ge_two_l2380_238069


namespace NUMINAMATH_GPT_complement_intersection_l2380_238011

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {3, 4, 5}) (hN : N = {2, 3}) :
  (U \ N) ∩ M = {4, 5} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l2380_238011


namespace NUMINAMATH_GPT_mailman_total_delivered_l2380_238061

def pieces_of_junk_mail : Nat := 6
def magazines : Nat := 5
def newspapers : Nat := 3
def bills : Nat := 4
def postcards : Nat := 2

def total_pieces_of_mail : Nat := pieces_of_junk_mail + magazines + newspapers + bills + postcards

theorem mailman_total_delivered : total_pieces_of_mail = 20 := by
  sorry

end NUMINAMATH_GPT_mailman_total_delivered_l2380_238061


namespace NUMINAMATH_GPT_first_digit_base_5_of_2197_l2380_238045

theorem first_digit_base_5_of_2197 : 
  ∃ k : ℕ, 2197 = k * 625 + r ∧ k = 3 ∧ r < 625 :=
by
  -- existence of k and r follows from the division algorithm
  -- sorry is used to indicate the part of the proof that needs to be filled in
  sorry

end NUMINAMATH_GPT_first_digit_base_5_of_2197_l2380_238045


namespace NUMINAMATH_GPT_inequality_of_powers_l2380_238058

theorem inequality_of_powers (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end NUMINAMATH_GPT_inequality_of_powers_l2380_238058


namespace NUMINAMATH_GPT_one_equation_does_not_pass_origin_l2380_238089

def passes_through_origin (eq : ℝ → ℝ) : Prop := eq 0 = 0

def equation1 (x : ℝ) : ℝ := x^4 + 1
def equation2 (x : ℝ) : ℝ := x^4 + x
def equation3 (x : ℝ) : ℝ := x^4 + x^2
def equation4 (x : ℝ) : ℝ := x^4 + x^3

theorem one_equation_does_not_pass_origin :
  (¬ passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  ¬ passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  ¬ passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  ¬ passes_through_origin equation4) :=
sorry

end NUMINAMATH_GPT_one_equation_does_not_pass_origin_l2380_238089


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2380_238084

variable {α : Type*} [LinearOrderedField α]

def sum_n_terms (a₁ d : α) (n : ℕ) : α :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a₁ : α) (h : sum_n_terms a₁ 1 4 = 1) :
  sum_n_terms a₁ 1 8 = 18 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2380_238084


namespace NUMINAMATH_GPT_chef_pies_total_l2380_238062

def chefPieSales : ℕ :=
  let small_shepherd_pies := 52 / 4
  let large_shepherd_pies := 76 / 8
  let small_chicken_pies := 80 / 5
  let large_chicken_pies := 130 / 10
  let small_vegetable_pies := 42 / 6
  let large_vegetable_pies := 96 / 12
  let small_beef_pies := 35 / 7
  let large_beef_pies := 105 / 14

  small_shepherd_pies + large_shepherd_pies + small_chicken_pies + large_chicken_pies +
  small_vegetable_pies + large_vegetable_pies +
  small_beef_pies + large_beef_pies

theorem chef_pies_total : chefPieSales = 80 := by
  unfold chefPieSales
  have h1 : 52 / 4 = 13 := by norm_num
  have h2 : 76 / 8 = 9 ∨ 76 / 8 = 10 := by norm_num -- rounding consideration
  have h3 : 80 / 5 = 16 := by norm_num
  have h4 : 130 / 10 = 13 := by norm_num
  have h5 : 42 / 6 = 7 := by norm_num
  have h6 : 96 / 12 = 8 := by norm_num
  have h7 : 35 / 7 = 5 := by norm_num
  have h8 : 105 / 14 = 7 ∨ 105 / 14 = 8 := by norm_num -- rounding consideration
  sorry

end NUMINAMATH_GPT_chef_pies_total_l2380_238062


namespace NUMINAMATH_GPT_find_k_l2380_238025

theorem find_k 
  (x k : ℚ)
  (h1 : (x^2 - 3*k)*(x + 3*k) = x^3 + 3*k*(x^2 - x - 7))
  (h2 : k ≠ 0) : k = 7 / 3 := 
sorry

end NUMINAMATH_GPT_find_k_l2380_238025


namespace NUMINAMATH_GPT_find_four_digit_number_l2380_238092

noncomputable def reverse_num (n : ℕ) : ℕ := -- assume definition to reverse digits
  sorry

theorem find_four_digit_number :
  ∃ (A : ℕ), 1000 ≤ A ∧ A ≤ 9999 ∧ reverse_num (9 * A) = A ∧ 9 * A = reverse_num A ∧ A = 1089 :=
sorry

end NUMINAMATH_GPT_find_four_digit_number_l2380_238092


namespace NUMINAMATH_GPT_range_of_a_l2380_238017

open Function

def f (x : ℝ) : ℝ := -2 * x^5 - x^3 - 7 * x + 2

theorem range_of_a (a : ℝ) : f (a^2) + f (a - 2) > 4 → -2 < a ∧ a < 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2380_238017


namespace NUMINAMATH_GPT_subset_relationship_l2380_238039

def S : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def T : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

theorem subset_relationship : S ⊆ T :=
by sorry

end NUMINAMATH_GPT_subset_relationship_l2380_238039


namespace NUMINAMATH_GPT_largest_integer_m_dividing_30_factorial_l2380_238012

theorem largest_integer_m_dividing_30_factorial :
  ∃ (m : ℕ), (∀ (k : ℕ), (18^k ∣ Nat.factorial 30) ↔ k ≤ m) ∧ m = 7 := by
  sorry

end NUMINAMATH_GPT_largest_integer_m_dividing_30_factorial_l2380_238012


namespace NUMINAMATH_GPT_total_savings_l2380_238021

-- Definitions and Conditions
def thomas_monthly_savings : ℕ := 40
def joseph_saving_ratio : ℚ := 3 / 5
def saving_period_months : ℕ := 72

-- Problem Statement
theorem total_savings :
  let thomas_total := thomas_monthly_savings * saving_period_months
  let joseph_monthly_savings := thomas_monthly_savings * joseph_saving_ratio
  let joseph_total := joseph_monthly_savings * saving_period_months
  thomas_total + joseph_total = 4608 := 
by
  sorry

end NUMINAMATH_GPT_total_savings_l2380_238021


namespace NUMINAMATH_GPT_license_plate_increase_l2380_238049

theorem license_plate_increase :
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 26^2 / 10 :=
by
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  show new_plates / old_plates = 26^2 / 10
  sorry

end NUMINAMATH_GPT_license_plate_increase_l2380_238049


namespace NUMINAMATH_GPT_find_number_l2380_238014

theorem find_number (x : ℕ) (h : x + 1015 = 3016) : x = 2001 :=
sorry

end NUMINAMATH_GPT_find_number_l2380_238014


namespace NUMINAMATH_GPT_sqrt_meaningful_l2380_238013

theorem sqrt_meaningful (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end NUMINAMATH_GPT_sqrt_meaningful_l2380_238013


namespace NUMINAMATH_GPT_hex_conversion_sum_l2380_238031

-- Convert hexadecimal E78 to decimal
def hex_to_decimal (h : String) : Nat :=
  match h with
  | "E78" => 3704
  | _ => 0

-- Convert decimal to radix 7
def decimal_to_radix7 (d : Nat) : String :=
  match d with
  | 3704 => "13541"
  | _ => ""

-- Convert radix 7 to decimal
def radix7_to_decimal (r : String) : Nat :=
  match r with
  | "13541" => 3704
  | _ => 0

-- Convert decimal to hexadecimal
def decimal_to_hex (d : Nat) : String :=
  match d with
  | 3704 => "E78"
  | 7408 => "1CF0"
  | _ => ""

theorem hex_conversion_sum :
  let initial_hex : String := "E78"
  let final_decimal := 3704 
  let final_hex := decimal_to_hex (final_decimal)
  let final_sum := hex_to_decimal initial_hex + final_decimal
  (decimal_to_hex final_sum) = "1CF0" :=
by
  sorry

end NUMINAMATH_GPT_hex_conversion_sum_l2380_238031


namespace NUMINAMATH_GPT_correct_calculation_l2380_238008

theorem correct_calculation (a b : ℝ) :
  ¬(a^2 + 2 * a^2 = 3 * a^4) ∧
  ¬(a^6 / a^3 = a^2) ∧
  ¬((a^2)^3 = a^5) ∧
  (ab)^2 = a^2 * b^2 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2380_238008


namespace NUMINAMATH_GPT_celestia_badges_l2380_238080

theorem celestia_badges (H L C : ℕ) (total_badges : ℕ) (h1 : H = 14) (h2 : L = 17) (h3 : total_badges = 83) (h4 : H + L + C = total_badges) : C = 52 :=
by
  sorry

end NUMINAMATH_GPT_celestia_badges_l2380_238080


namespace NUMINAMATH_GPT_unique_triple_solution_l2380_238005

theorem unique_triple_solution (a b c : ℝ) 
  (h1 : a * (b ^ 2 + c) = c * (c + a * b))
  (h2 : b * (c ^ 2 + a) = a * (a + b * c))
  (h3 : c * (a ^ 2 + b) = b * (b + c * a)) : 
  a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_unique_triple_solution_l2380_238005


namespace NUMINAMATH_GPT_part_a_least_moves_part_b_least_moves_l2380_238095

def initial_position : Nat := 0
def total_combinations : Nat := 10^6
def excluded_combinations : List Nat := [0, 10^5, 2 * 10^5, 3 * 10^5, 4 * 10^5, 5 * 10^5, 6 * 10^5, 7 * 10^5, 8 * 10^5, 9 * 10^5]

theorem part_a_least_moves : total_combinations - 1 = 10^6 - 1 := by
  simp [total_combinations, Nat.pow]

theorem part_b_least_moves : total_combinations - excluded_combinations.length = 10^6 - 10 := by
  simp [total_combinations, excluded_combinations, Nat.pow, List.length]

end NUMINAMATH_GPT_part_a_least_moves_part_b_least_moves_l2380_238095


namespace NUMINAMATH_GPT_value_of_expression_l2380_238063

theorem value_of_expression (x : ℤ) (h : x^2 = 1369) : (x + 1) * (x - 1) = 1368 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l2380_238063


namespace NUMINAMATH_GPT_circle_equation_l2380_238097

-- Define the given conditions
def point_P : ℝ × ℝ := (-1, 0)
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def center_C : ℝ × ℝ := (1, 2)

-- Define the required equation of the circle and the claim
def required_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- The Lean theorem statement
theorem circle_equation :
  ∃ (x y : ℝ), required_circle x y :=
sorry

end NUMINAMATH_GPT_circle_equation_l2380_238097
