import Mathlib

namespace minimum_value_of_f_on_interval_l667_66779

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + Real.log x

theorem minimum_value_of_f_on_interval :
  (∀ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x ≥ f (Real.exp 1)) ∧
  ∃ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x = f (Real.exp 1) := 
by
  sorry

end minimum_value_of_f_on_interval_l667_66779


namespace temperature_at_noon_l667_66755

-- Definitions of the given conditions.
def morning_temperature : ℝ := 4
def temperature_drop : ℝ := 10

-- The theorem statement that needs to be proven.
theorem temperature_at_noon : morning_temperature - temperature_drop = -6 :=
by
  -- The proof can be filled in by solving the stated theorem.
  sorry

end temperature_at_noon_l667_66755


namespace verify_toothpick_count_l667_66704

def toothpick_problem : Prop :=
  let L := 45
  let W := 25
  let Mv := 8
  let Mh := 5
  -- Calculate the total number of vertical toothpicks
  let verticalToothpicks := (L + 1 - Mv) * W
  -- Calculate the total number of horizontal toothpicks
  let horizontalToothpicks := (W + 1 - Mh) * L
  -- Calculate the total number of toothpicks
  let totalToothpicks := verticalToothpicks + horizontalToothpicks
  -- Ensure the total matches the expected result
  totalToothpicks = 1895

theorem verify_toothpick_count : toothpick_problem :=
by
  sorry

end verify_toothpick_count_l667_66704


namespace andrew_total_payment_l667_66719

-- Given conditions
def quantity_of_grapes := 14
def rate_per_kg_grapes := 54
def quantity_of_mangoes := 10
def rate_per_kg_mangoes := 62

-- Calculations
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Theorem to prove
theorem andrew_total_payment : total_amount_paid = 1376 := by
  sorry

end andrew_total_payment_l667_66719


namespace problem_statement_l667_66740

theorem problem_statement (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + 1 / x^2 = 7 :=
sorry

end problem_statement_l667_66740


namespace not_enough_funds_to_buy_two_books_l667_66798

def storybook_cost : ℝ := 25.5
def sufficient_funds (amount : ℝ) : Prop := amount >= 50

theorem not_enough_funds_to_buy_two_books : ¬ sufficient_funds (2 * storybook_cost) :=
by
  sorry

end not_enough_funds_to_buy_two_books_l667_66798


namespace fruit_basket_count_l667_66789

-- Define the number of apples and oranges
def apples := 7
def oranges := 12

-- Condition: A fruit basket must contain at least two pieces of fruit
def min_pieces_of_fruit := 2

-- Problem: Prove that there are 101 different fruit baskets containing at least two pieces of fruit
theorem fruit_basket_count (n_apples n_oranges n_min_pieces : Nat) (h_apples : n_apples = apples) (h_oranges : n_oranges = oranges) (h_min_pieces : n_min_pieces = min_pieces_of_fruit) :
  (n_apples = 7) ∧ (n_oranges = 12) ∧ (n_min_pieces = 2) → (104 - 3 = 101) :=
by
  sorry

end fruit_basket_count_l667_66789


namespace price_of_olives_l667_66782

theorem price_of_olives 
  (cherries_price : ℝ)
  (total_cost_with_discount : ℝ)
  (num_bags : ℕ)
  (discount : ℝ)
  (olives_price : ℝ) :
  cherries_price = 5 →
  total_cost_with_discount = 540 →
  num_bags = 50 →
  discount = 0.10 →
  (0.9 * (num_bags * cherries_price + num_bags * olives_price) = total_cost_with_discount) →
  olives_price = 7 :=
by
  intros h_cherries_price h_total_cost h_num_bags h_discount h_equation
  sorry

end price_of_olives_l667_66782


namespace number_of_football_players_l667_66712

theorem number_of_football_players
  (cricket_players : ℕ)
  (hockey_players : ℕ)
  (softball_players : ℕ)
  (total_players : ℕ) :
  cricket_players = 22 →
  hockey_players = 15 →
  softball_players = 19 →
  total_players = 77 →
  total_players - (cricket_players + hockey_players + softball_players) = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_football_players_l667_66712


namespace inequality_proofs_l667_66745

def sinSumInequality (A B C ε : ℝ) : Prop :=
  ε * (Real.sin A + Real.sin B + Real.sin C) ≤ Real.sin A * Real.sin B * Real.sin C + 1 + ε^3

def sinProductInequality (A B C ε : ℝ) : Prop :=
  (1 + ε + Real.sin A) * (1 + ε + Real.sin B) * (1 + ε + Real.sin C) ≥ 9 * ε * (Real.sin A + Real.sin B + Real.sin C)

theorem inequality_proofs (A B C ε : ℝ) (hA : 0 ≤ A ∧ A ≤ Real.pi) (hB : 0 ≤ B ∧ B ≤ Real.pi) 
  (hC : 0 ≤ C ∧ C ≤ Real.pi) (hε : ε ≥ 1) :
  sinSumInequality A B C ε ∧ sinProductInequality A B C ε :=
by
  sorry

end inequality_proofs_l667_66745


namespace range_of_m_for_common_point_l667_66763

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ :=
  -x^2 - 2 * x + m

-- Define the condition for a common point with the x-axis (i.e., it has real roots)
def has_common_point_with_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_function x m = 0

-- The theorem statement
theorem range_of_m_for_common_point : ∀ m : ℝ, has_common_point_with_x_axis m ↔ m ≥ -1 := 
sorry

end range_of_m_for_common_point_l667_66763


namespace cuboid_volume_is_correct_l667_66785

-- Definition of cuboid edges and volume calculation
def cuboid_volume (a b c : ℕ) : ℕ := a * b * c

-- Given conditions
def edge1 : ℕ := 2
def edge2 : ℕ := 5
def edge3 : ℕ := 3

-- Theorem statement
theorem cuboid_volume_is_correct : cuboid_volume edge1 edge2 edge3 = 30 := 
by sorry

end cuboid_volume_is_correct_l667_66785


namespace train_pass_time_l667_66747

theorem train_pass_time (train_length : ℕ) (platform_length : ℕ) (speed : ℕ) (h1 : train_length = 50) (h2 : platform_length = 100) (h3 : speed = 15) : 
  (train_length + platform_length) / speed = 10 :=
by
  sorry

end train_pass_time_l667_66747


namespace smallest_number_four_solutions_sum_four_squares_l667_66781

def is_sum_of_four_squares (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a^2 + b^2 + c^2 + d^2

theorem smallest_number_four_solutions_sum_four_squares :
  ∃ n : ℕ,
    is_sum_of_four_squares n ∧
    (∃ (a1 b1 c1 d1 a2 b2 c2 d2 a3 b3 c3 d3 a4 b4 c4 d4 : ℕ),
      n = a1^2 + b1^2 + c1^2 + d1^2 ∧
      n = a2^2 + b2^2 + c2^2 + d2^2 ∧
      n = a3^2 + b3^2 + c3^2 + d3^2 ∧
      n = a4^2 + b4^2 + c4^2 + d4^2 ∧
      (a1, b1, c1, d1) ≠ (a2, b2, c2, d2) ∧
      (a1, b1, c1, d1) ≠ (a3, b3, c3, d3) ∧
      (a1, b1, c1, d1) ≠ (a4, b4, c4, d4) ∧
      (a2, b2, c2, d2) ≠ (a3, b3, c3, d3) ∧
      (a2, b2, c2, d2) ≠ (a4, b4, c4, d4) ∧
      (a3, b3, c3, d3) ≠ (a4, b4, c4, d4)) ∧
    (∀ m : ℕ,
      m < 635318657 →
      ¬ (∃ (a5 b5 c5 d5 a6 b6 c6 d6 a7 b7 c7 d7 a8 b8 c8 d8 : ℕ),
        m = a5^2 + b5^2 + c5^2 + d5^2 ∧
        m = a6^2 + b6^2 + c6^2 + d6^2 ∧
        m = a7^2 + b7^2 + c7^2 + d7^2 ∧
        m = a8^2 + b8^2 + c8^2 + d8^2 ∧
        (a5, b5, c5, d5) ≠ (a6, b6, c6, d6) ∧
        (a5, b5, c5, d5) ≠ (a7, b7, c7, d7) ∧
        (a5, b5, c5, d5) ≠ (a8, b8, c8, d8) ∧
        (a6, b6, c6, d6) ≠ (a7, b7, c7, d7) ∧
        (a6, b6, c6, d6) ≠ (a8, b8, c8, d8) ∧
        (a7, b7, c7, d7) ≠ (a8, b8, c8, d8))) :=
  sorry

end smallest_number_four_solutions_sum_four_squares_l667_66781


namespace line_equation_l667_66776

theorem line_equation {x y : ℝ} (m b : ℝ) (h1 : m = 2) (h2 : b = -3) :
    (∃ (f : ℝ → ℝ), (∀ x, f x = m * x + b) ∧ (∀ x, 2 * x - f x - 3 = 0)) :=
by
  sorry

end line_equation_l667_66776


namespace number_of_machines_in_first_scenario_l667_66766

noncomputable def machine_work_rate (R : ℝ) (hours_per_job : ℝ) : Prop :=
  (6 * R * 8 = 1)

noncomputable def machines_first_scenario (M : ℝ) (R : ℝ) (hours_per_job_first : ℝ) : Prop :=
  (M * R * hours_per_job_first = 1)

theorem number_of_machines_in_first_scenario (M : ℝ) (R : ℝ) :
  machine_work_rate R 8 ∧ machines_first_scenario M R 6 -> M = 8 :=
sorry

end number_of_machines_in_first_scenario_l667_66766


namespace card_arrangement_bound_l667_66754

theorem card_arrangement_bound : 
  ∀ (cards : ℕ) (cells : ℕ), cards = 1000 → cells = 1994 → 
  ∃ arrangements : ℕ, arrangements = cells - cards + 1 ∧ arrangements < 500000 :=
by {
  sorry
}

end card_arrangement_bound_l667_66754


namespace find_multiplier_n_l667_66792

variable (x y n : ℝ)

theorem find_multiplier_n (h1 : 5 * x = n * y) 
  (h2 : x * y ≠ 0) 
  (h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998) : 
  n = 6 := 
by
  sorry

end find_multiplier_n_l667_66792


namespace find_A_range_sinB_sinC_l667_66736

-- Given conditions in a triangle
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h_cos_eq : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)

-- Angle A verification
theorem find_A (h_sum_angles : A + B + C = Real.pi) : A = Real.pi / 3 :=
  sorry

-- Range of sin B + sin C
theorem range_sinB_sinC (h_sum_angles : A + B + C = Real.pi) :
  (0 < B ∧ B < 2 * Real.pi / 3) →
  Real.sin B + Real.sin C ∈ Set.Ioo (Real.sqrt 3 / 2) (Real.sqrt 3) :=
  sorry

end find_A_range_sinB_sinC_l667_66736


namespace two_pow_58_plus_one_factored_l667_66703

theorem two_pow_58_plus_one_factored :
  ∃ (a b c : ℕ), 2 < a ∧ 2 < b ∧ 2 < c ∧ 2 ^ 58 + 1 = a * b * c :=
sorry

end two_pow_58_plus_one_factored_l667_66703


namespace domain_range_g_l667_66751

variable (f : ℝ → ℝ) 

noncomputable def g (x : ℝ) := 2 - f (x + 1)

theorem domain_range_g :
  (∀ x, 0 ≤ x → x ≤ 3 → 0 ≤ f x → f x ≤ 1) →
  (∀ x, -1 ≤ x → x ≤ 2) ∧ (∀ y, 1 ≤ y → y ≤ 2) :=
sorry

end domain_range_g_l667_66751


namespace total_weight_of_family_l667_66731

theorem total_weight_of_family (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 40) : M + D + C = 160 :=
sorry

end total_weight_of_family_l667_66731


namespace count_three_digit_odd_increasing_order_l667_66744

theorem count_three_digit_odd_increasing_order : 
  ∃ n : ℕ, n = 10 ∧
  ∀ a b c : ℕ, (100 * a + 10 * b + c) % 2 = 1 ∧ a < b ∧ b < c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 → 
    (100 * a + 10 * b + c) % 2 = 1 := 
sorry

end count_three_digit_odd_increasing_order_l667_66744


namespace x_coordinate_l667_66725

theorem x_coordinate (x : ℝ) (y : ℝ) :
  (∃ m : ℝ, m = (0 + 6) / (4 + 8) ∧
            y + 6 = m * (x + 8) ∧
            y = 3) →
  x = 10 :=
by
  sorry

end x_coordinate_l667_66725


namespace average_speed_l667_66796

theorem average_speed (D : ℝ) :
  let time_by_bus := D / 80
  let time_walking := D / 16
  let time_cycling := D / 120
  let total_time := time_by_bus + time_walking + time_cycling
  let total_distance := 2 * D
  total_distance / total_time = 24 := by
  sorry

end average_speed_l667_66796


namespace remainder_equality_l667_66705

theorem remainder_equality
  (P P' K D R R' r r' : ℕ)
  (h1 : P > P')
  (h2 : P % K = 0)
  (h3 : P' % K = 0)
  (h4 : P % D = R)
  (h5 : P' % D = R')
  (h6 : (P * K - P') % D = r)
  (h7 : (R * K - R') % D = r') :
  r = r' :=
sorry

end remainder_equality_l667_66705


namespace Adam_final_amount_l667_66700

def initial_amount : ℝ := 5.25
def spent_on_game : ℝ := 2.30
def spent_on_snacks : ℝ := 1.75
def found_dollar : ℝ := 1.00
def allowance : ℝ := 5.50

theorem Adam_final_amount :
  (initial_amount - spent_on_game - spent_on_snacks + found_dollar + allowance) = 7.70 :=
by
  sorry

end Adam_final_amount_l667_66700


namespace minimum_valid_N_exists_l667_66739

theorem minimum_valid_N_exists (N : ℝ) (a : ℕ → ℕ) :
  (∀ n : ℕ, a n > 0) →
  (∀ n : ℕ, a n < a (n+1)) →
  (∀ n : ℕ, (a (2*n - 1) + a (2*n)) / a n = N) →
  N ≥ 4 :=
by
  sorry

end minimum_valid_N_exists_l667_66739


namespace curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l667_66773

-- Define the curve G as a set of points (x, y) satisfying the equation x^3 + y^3 - 6xy = 0
def curveG (x y : ℝ) : Prop :=
  x^3 + y^3 - 6 * x * y = 0

-- Prove symmetry of curveG with respect to the line y = x
theorem curveG_symmetric (x y : ℝ) (h : curveG x y) : curveG y x :=
  sorry

-- Prove unique common point with the line x + y - 6 = 0
theorem curveG_unique_common_point : ∃! p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 + p.2 = 6 :=
  sorry

-- Prove curveG has at least one common point with the line x - y + 1 = 0
theorem curveG_common_points_x_y : ∃ p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 - p.2 + 1 = 0 :=
  sorry

-- Prove the maximum distance from any point on the curveG to the origin is 3√2
theorem curveG_max_distance : ∀ p : ℝ × ℝ, curveG p.1 p.2 → p.1 > 0 → p.2 > 0 → (p.1^2 + p.2^2 ≤ 18) :=
  sorry

end curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l667_66773


namespace binom_2000_3_eq_l667_66774

theorem binom_2000_3_eq : Nat.choose 2000 3 = 1331000333 := by
  sorry

end binom_2000_3_eq_l667_66774


namespace find_q_r_s_l667_66711

noncomputable def is_valid_geometry 
  (AD : ℝ) (AL : ℝ) (AM : ℝ) (AN : ℝ) (q : ℕ) (r : ℕ) (s : ℕ) : Prop :=
  AD = 10 ∧ AL = 3 ∧ AM = 3 ∧ AN = 3 ∧ ¬(∃ p : ℕ, p^2 ∣ s)

theorem find_q_r_s : ∃ (q r s : ℕ), is_valid_geometry 10 3 3 3 q r s ∧ q + r + s = 711 :=
by
  sorry

end find_q_r_s_l667_66711


namespace composition_of_homotheties_l667_66787

-- Define points A1 and A2 and the coefficients k1 and k2
variables (A1 A2 : ℂ) (k1 k2 : ℂ)

-- Definition of homothety
def homothety (A : ℂ) (k : ℂ) (z : ℂ) : ℂ := k * (z - A) + A

-- Translation vector in case 1
noncomputable def translation_vector (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 = 1 then (1 - k1) * A1 + (k1 - 1) * A2 else 0 

-- Center A in case 2
noncomputable def center (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 ≠ 1 then (k2 * (1 - k1) * A1 + (1 - k2) * A2) / (k1 * k2 - 1) else 0

-- The final composition of two homotheties
noncomputable def composition (A1 A2 : ℂ) (k1 k2 : ℂ) (z : ℂ) : ℂ :=
  if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
  else homothety (center A1 A2 k1 k2) (k1 * k2) z

-- The theorem to prove
theorem composition_of_homotheties 
  (A1 A2 : ℂ) (k1 k2 : ℂ) : ∀ z : ℂ,
  composition A1 A2 k1 k2 z = if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
                              else homothety (center A1 A2 k1 k2) (k1 * k2) z := 
by sorry

end composition_of_homotheties_l667_66787


namespace diorama_time_subtraction_l667_66726

theorem diorama_time_subtraction (P B X : ℕ) (h1 : B = 3 * P - X) (h2 : B = 49) (h3 : P + B = 67) : X = 5 :=
by
  sorry

end diorama_time_subtraction_l667_66726


namespace find_vector_p_l667_66771

noncomputable def vector_proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := v.1 * u.1 + v.2 * u.2
  let dot_u := u.1 * u.1 + u.2 * u.2
  let scale := dot_uv / dot_u
  (scale * u.1, scale * u.2)

theorem find_vector_p :
  ∃ p : ℝ × ℝ,
    vector_proj (5, -2) p = p ∧
    vector_proj (2, 6) p = p ∧
    p = (14 / 73, 214 / 73) :=
by
  sorry

end find_vector_p_l667_66771


namespace trig_expression_evaluation_l667_66721

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) :
  Real.sin θ ^ 2 + (Real.sin θ * Real.cos θ) - 2 * (Real.cos θ ^ 2) = 4 / 5 := 
by
  sorry

end trig_expression_evaluation_l667_66721


namespace medium_size_shoes_initially_stocked_l667_66758

variable {M : ℕ}  -- The number of medium-size shoes initially stocked

noncomputable def initial_pairs_eq (M : ℕ) := 22 + M + 24
noncomputable def shoes_sold (M : ℕ) := initial_pairs_eq M - 13

theorem medium_size_shoes_initially_stocked :
  shoes_sold M = 83 → M = 26 :=
by
  sorry

end medium_size_shoes_initially_stocked_l667_66758


namespace gain_percentage_l667_66777

variables (C S : ℝ) (hC : C > 0)
variables (hS : S > 0)

def cost_price := 25 * C
def selling_price := 25 * S
def gain := 10 * S 

theorem gain_percentage (h_eq : 25 * S = 25 * C + 10 * S):
  (S = C) → 
  ((gain / cost_price) * 100 = 40) :=
by
  sorry

end gain_percentage_l667_66777


namespace total_weight_of_three_packages_l667_66762

theorem total_weight_of_three_packages (a b c d : ℝ)
  (h1 : a + b = 162)
  (h2 : b + c = 164)
  (h3 : c + a = 168) :
  a + b + c = 247 :=
sorry

end total_weight_of_three_packages_l667_66762


namespace large_circle_radius_l667_66748

theorem large_circle_radius (s : ℝ) (r : ℝ) (R : ℝ)
  (side_length : s = 6)
  (coverage : ∀ (x y : ℝ), (x - y)^2 + (x - y)^2 = (2 * R)^2) :
  R = 3 * Real.sqrt 2 :=
by
  sorry

end large_circle_radius_l667_66748


namespace gcd_g_values_l667_66750

def g (x : ℤ) : ℤ := x^2 - 2 * x + 2023

theorem gcd_g_values : gcd (g 102) (g 103) = 1 := by
  sorry

end gcd_g_values_l667_66750


namespace arithmetic_sequence_sum_l667_66799

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ),
    (∀ (n : ℕ), a_n n = 1 + (n - 1) * d) →  -- first condition
    d ≠ 0 →  -- second condition
    (∀ (n : ℕ), S_n n = n / 2 * (2 * 1 + (n - 1) * d)) →  -- third condition
    (1 * (1 + 4 * d) = (1 + d) ^ 2) →  -- fourth condition
    S_n 8 = 64 :=  -- conclusion
by {
  sorry
}

end arithmetic_sequence_sum_l667_66799


namespace percentage_of_women_lawyers_l667_66768

theorem percentage_of_women_lawyers
  (T : ℝ) 
  (h1 : 0.70 * T = W) 
  (h2 : 0.28 * T = WL) : 
  ((WL / W) * 100 = 40) :=
by
  sorry

end percentage_of_women_lawyers_l667_66768


namespace smallest_mult_to_cube_l667_66715

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_mult_to_cube (n : ℕ) (h : ∃ n, ∃ k, n * y = k^3) : n = 4500 := 
  sorry

end smallest_mult_to_cube_l667_66715


namespace Wayne_blocks_l667_66788

theorem Wayne_blocks (initial_blocks : ℕ) (additional_blocks : ℕ) (total_blocks : ℕ) 
  (h1 : initial_blocks = 9) (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 :=
by {
  -- h1: initial_blocks = 9
  -- h2: additional_blocks = 6
  -- h3: total_blocks = initial_blocks + additional_blocks
  sorry
}

end Wayne_blocks_l667_66788


namespace triangle_third_side_length_l667_66775

theorem triangle_third_side_length {x : ℝ}
    (h1 : 3 > 0)
    (h2 : 7 > 0)
    (h3 : 3 + 7 > x)
    (h4 : x + 3 > 7)
    (h5 : x + 7 > 3) :
    4 < x ∧ x < 10 := by
  sorry

end triangle_third_side_length_l667_66775


namespace number_of_segments_l667_66790

theorem number_of_segments (tangent_chords : ℕ) (angle_ABC : ℝ) (h : angle_ABC = 80) :
  tangent_chords = 18 :=
sorry

end number_of_segments_l667_66790


namespace find_t_max_value_of_xyz_l667_66741

-- Problem (1)
theorem find_t (t : ℝ) (x : ℝ) (h1 : |2 * x + t| - t ≤ 8) (sol_set : -5 ≤ x ∧ x ≤ 4) : t = 1 :=
sorry

-- Problem (2)
theorem max_value_of_xyz (x y z : ℝ) (h2 : x^2 + (1/4) * y^2 + (1/9) * z^2 = 2) : x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end find_t_max_value_of_xyz_l667_66741


namespace min_b_factors_l667_66795

theorem min_b_factors (x r s b : ℕ) (h : r * s = 1998) (fact : (x + r) * (x + s) = x^2 + b * x + 1998) : b = 91 :=
sorry

end min_b_factors_l667_66795


namespace hyperbola_solution_l667_66729

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 - x^2 / 3 = 1

theorem hyperbola_solution :
  ∃ x y : ℝ,
    (∃ c : ℝ, c = 2) ∧
    (∃ a : ℝ, a = 1) ∧
    (∃ n : ℝ, n = 1) ∧
    (∃ b : ℝ, b^2 = 3) ∧
    (∃ m : ℝ, m = -3) ∧
    hyperbola_eq x y := sorry

end hyperbola_solution_l667_66729


namespace seq_a2020_l667_66783

def seq (a : ℕ → ℕ) : Prop :=
(∀ n : ℕ, (a n + a (n+1) ≠ a (n+2) + a (n+3))) ∧
(∀ n : ℕ, (a n + a (n+1) + a (n+2) ≠ a (n+3) + a (n+4) + a (n+5))) ∧
(a 1 = 0)

theorem seq_a2020 (a : ℕ → ℕ) (h : seq a) : a 2020 = 1 :=
sorry

end seq_a2020_l667_66783


namespace range_of_m_l667_66767

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 1) : 
  ∀ m : ℝ, (x + 2 * y > m) ↔ (m < 8) :=
by 
  sorry

end range_of_m_l667_66767


namespace solution_of_inequality_l667_66708

theorem solution_of_inequality (x : ℝ) : 3 * x > 2 * x + 4 ↔ x > 4 := 
sorry

end solution_of_inequality_l667_66708


namespace y1_y2_positive_l667_66765

theorem y1_y2_positive 
  (x1 x2 x3 : ℝ)
  (y1 y2 y3 : ℝ)
  (h_line1 : y1 = -2 * x1 + 3)
  (h_line2 : y2 = -2 * x2 + 3)
  (h_line3 : y3 = -2 * x3 + 3)
  (h_order : x1 < x2 ∧ x2 < x3)
  (h_product_neg : x2 * x3 < 0) :
  y1 * y2 > 0 :=
by
  sorry

end y1_y2_positive_l667_66765


namespace midpoint_product_l667_66732

theorem midpoint_product (x y z : ℤ) 
  (h1 : (2 + x) / 2 = 4) 
  (h2 : (10 + y) / 2 = 6) 
  (h3 : (5 + z) / 2 = 3) : 
  x * y * z = 12 := 
by
  sorry

end midpoint_product_l667_66732


namespace isosceles_right_triangle_area_l667_66701

/--
Given an isosceles right triangle with a hypotenuse of 6√2 units, prove that the area
of this triangle is 18 square units.
-/
theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (hyp : h = 6 * Real.sqrt 2) 
  (isosceles : h = l * Real.sqrt 2) : 
  (1/2) * l^2 = 18 :=
by
  sorry

end isosceles_right_triangle_area_l667_66701


namespace cos_alpha_plus_beta_l667_66734

theorem cos_alpha_plus_beta (α β : ℝ) (hα : Complex.exp (Complex.I * α) = 4 / 5 + Complex.I * 3 / 5)
  (hβ : Complex.exp (Complex.I * β) = -5 / 13 + Complex.I * 12 / 13) : 
  Real.cos (α + β) = -7 / 13 :=
  sorry

end cos_alpha_plus_beta_l667_66734


namespace angus_total_investment_l667_66757

variable (x T : ℝ)

theorem angus_total_investment (h1 : 0.03 * x + 0.05 * 6000 = 660) (h2 : T = x + 6000) : T = 18000 :=
by
  sorry

end angus_total_investment_l667_66757


namespace adam_change_is_correct_l667_66793

-- Define the conditions
def adam_money : ℝ := 5.00
def airplane_cost : ℝ := 4.28
def change : ℝ := adam_money - airplane_cost

-- State the theorem
theorem adam_change_is_correct : change = 0.72 := 
by {
  -- Proof can be added later
  sorry
}

end adam_change_is_correct_l667_66793


namespace jason_current_cards_l667_66780

-- Define Jason's initial number of Pokemon cards
def jason_initial_cards : ℕ := 1342

-- Define the number of Pokemon cards Alyssa bought
def alyssa_bought_cards : ℕ := 536

-- Define the number of Pokemon cards Jason has now
def jason_final_cards (initial_cards bought_cards : ℕ) : ℕ :=
  initial_cards - bought_cards

-- Theorem statement verifying the final number of Pokemon cards Jason has
theorem jason_current_cards : jason_final_cards jason_initial_cards alyssa_bought_cards = 806 :=
by
  -- Proof goes here
  sorry

end jason_current_cards_l667_66780


namespace Henry_has_four_Skittles_l667_66720

-- Defining the initial amount of Skittles Bridget has
def Bridget_initial := 4

-- Defining the final amount of Skittles Bridget has after receiving all of Henry's Skittles
def Bridget_final := 8

-- Defining the amount of Skittles Henry has
def Henry_Skittles := Bridget_final - Bridget_initial

-- The proof statement to be proven
theorem Henry_has_four_Skittles : Henry_Skittles = 4 := by
  sorry

end Henry_has_four_Skittles_l667_66720


namespace units_digit_35_87_plus_93_49_l667_66772

theorem units_digit_35_87_plus_93_49 : (35^87 + 93^49) % 10 = 8 := by
  sorry

end units_digit_35_87_plus_93_49_l667_66772


namespace area_of_AFCH_l667_66769

-- Define the lengths of the sides of the rectangles
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the problem statement
theorem area_of_AFCH :
  let intersection_area := min BC FG * min EF AB
  let total_area := AB * FG
  let outer_ring_area := total_area - intersection_area
  intersection_area + outer_ring_area / 2 = 52.5 :=
by
  -- Use the values of AB, BC, EF, and FG to compute
  sorry

end area_of_AFCH_l667_66769


namespace money_difference_l667_66722

def share_ratio (w x y z : ℝ) (k : ℝ) : Prop :=
  w = k ∧ x = 6 * k ∧ y = 2 * k ∧ z = 4 * k

theorem money_difference (k : ℝ) (h : k = 375) : 
  ∀ w x y z : ℝ, share_ratio w x y z k → (x - y) = 1500 := 
by
  intros w x y z h_ratio
  rw [share_ratio] at h_ratio
  have h_w : w = k := h_ratio.1
  have h_x : x = 6 * k := h_ratio.2.1
  have h_y : y = 2 * k := h_ratio.2.2.1
  rw [h_x, h_y]
  rw [h] at h_x h_y
  sorry

end money_difference_l667_66722


namespace cos_225_eq_neg_sqrt2_div2_l667_66727

theorem cos_225_eq_neg_sqrt2_div2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end cos_225_eq_neg_sqrt2_div2_l667_66727


namespace choir_singers_joined_final_verse_l667_66718

theorem choir_singers_joined_final_verse (total_singers : ℕ) (first_verse_fraction : ℚ)
  (second_verse_fraction : ℚ) (initial_remaining : ℕ) (second_verse_joined : ℕ) : 
  total_singers = 30 → 
  first_verse_fraction = 1 / 2 → 
  second_verse_fraction = 1 / 3 → 
  initial_remaining = total_singers / 2 → 
  second_verse_joined = initial_remaining / 3 → 
  (total_singers - (initial_remaining + second_verse_joined)) = 10 := 
by
  intros
  sorry

end choir_singers_joined_final_verse_l667_66718


namespace base_conversion_problem_l667_66717

def base_to_dec (base : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ x acc => x + base * acc) 0

theorem base_conversion_problem : 
  (base_to_dec 8 [2, 5, 3] : ℝ) / (base_to_dec 4 [1, 3] : ℝ) + 
  (base_to_dec 5 [1, 3, 2] : ℝ) / (base_to_dec 3 [2, 3] : ℝ) = 28.67 := by
  sorry

end base_conversion_problem_l667_66717


namespace salary_recovery_l667_66764

theorem salary_recovery (S : ℝ) : 
  (0.80 * S) + (0.25 * (0.80 * S)) = S :=
by
  sorry

end salary_recovery_l667_66764


namespace parabola_focus_l667_66759

theorem parabola_focus (x y : ℝ) (h : y = 4 * x^2) : (0, 1) = (0, 1) :=
by 
  -- key steps would go here
  sorry

end parabola_focus_l667_66759


namespace positive_integer_solutions_l667_66743

theorem positive_integer_solutions : 
  (∀ x : ℤ, ((1 + 2 * (x:ℝ)) / 4 - (1 - 3 * (x:ℝ)) / 10 > -1 / 5) ∧ (3 * (x:ℝ) - 1 < 2 * ((x:ℝ) + 1)) → (x = 1 ∨ x = 2)) :=
by 
  sorry

end positive_integer_solutions_l667_66743


namespace max_distance_of_MN_l667_66749

noncomputable def curve_C_polar (θ : ℝ) : ℝ := 2 * Real.cos θ

noncomputable def curve_C_cartesian (x y : ℝ) := x^2 + y^2 - 2 * x

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  ( -1 + (Real.sqrt 5 / 5) * t, (2 * Real.sqrt 5 / 5) * t)

def point_M : ℝ × ℝ := (0, 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def center_C : ℝ × ℝ := (1, 0)

theorem max_distance_of_MN :
  ∃ N : ℝ × ℝ, 
  ∀ (θ : ℝ), N = (curve_C_polar θ * Real.cos θ, curve_C_polar θ * Real.sin θ) →
  distance point_M N ≤ Real.sqrt 5 + 1 :=
sorry

end max_distance_of_MN_l667_66749


namespace neg_root_sufficient_not_necessary_l667_66791

theorem neg_root_sufficient_not_necessary (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (a < 0) :=
sorry

end neg_root_sufficient_not_necessary_l667_66791


namespace hari_contribution_l667_66737

theorem hari_contribution (c_p: ℕ) (m_p: ℕ) (ratio_p: ℕ) 
                          (m_h: ℕ) (ratio_h: ℕ) (profit_ratio_p: ℕ) (profit_ratio_h: ℕ) 
                          (c_h: ℕ) : 
  (c_p = 3780) → 
  (m_p = 12) → 
  (ratio_p = 2) → 
  (m_h = 7) → 
  (ratio_h = 3) → 
  (profit_ratio_p = 2) →
  (profit_ratio_h = 3) →
  (c_p * m_p * profit_ratio_h) = (c_h * m_h * profit_ratio_p) → 
  c_h = 9720 :=
by
  intros
  sorry

end hari_contribution_l667_66737


namespace number_of_integers_satisfying_inequalities_l667_66707

theorem number_of_integers_satisfying_inequalities :
  ∃ (count : ℕ), count = 3 ∧
    (∀ x : ℤ, -4 * x ≥ x + 10 → -3 * x ≤ 15 → -5 * x ≥ 3 * x + 24 → 2 * x ≤ 18 →
      x = -5 ∨ x = -4 ∨ x = -3) :=
sorry

end number_of_integers_satisfying_inequalities_l667_66707


namespace average_effective_increase_correct_l667_66733

noncomputable def effective_increase (initial_price: ℕ) (price_increase_percent: ℕ) (discount_percent: ℕ) : ℕ :=
let increased_price := initial_price + (initial_price * price_increase_percent / 100)
let final_price := increased_price - (increased_price * discount_percent / 100)
(final_price - initial_price) * 100 / initial_price

noncomputable def average_effective_increase : ℕ :=
let increase1 := effective_increase 300 10 5
let increase2 := effective_increase 450 15 7
let increase3 := effective_increase 600 20 10
(increase1 + increase2 + increase3) / 3

theorem average_effective_increase_correct :
  average_effective_increase = 6483 / 100 :=
by
  sorry

end average_effective_increase_correct_l667_66733


namespace trader_profit_l667_66786

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def discount_price (P : ℝ) : ℝ := 0.95 * P
noncomputable def selling_price (P : ℝ) : ℝ := 1.52 * P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def percent_profit (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) (hP : 0 < P) : percent_profit P = 52 := by 
  sorry

end trader_profit_l667_66786


namespace length_of_rectangle_l667_66753

-- Define the conditions as given in the problem
variables (width : ℝ) (perimeter : ℝ) (length : ℝ)

-- The conditions provided
def conditions : Prop :=
  width = 15 ∧ perimeter = 70 ∧ perimeter = 2 * (length + width)

-- The statement to prove: the length of the rectangle is 20 feet
theorem length_of_rectangle {width perimeter length : ℝ} (h : conditions width perimeter length) : length = 20 :=
by 
  -- This is where the proof steps would go
  sorry

end length_of_rectangle_l667_66753


namespace sum_problem3_equals_50_l667_66714

-- Assume problem3_condition is a placeholder for the actual conditions described in problem 3
-- and sum_problem3 is a placeholder for the sum of elements described in problem 3.

axiom problem3_condition : Prop
axiom sum_problem3 : ℕ

theorem sum_problem3_equals_50 (h : problem3_condition) : sum_problem3 = 50 :=
sorry

end sum_problem3_equals_50_l667_66714


namespace find_vertical_shift_l667_66730

theorem find_vertical_shift (A B C D : ℝ) (h1 : ∀ x, -3 ≤ A * Real.cos (B * x + C) + D ∧ A * Real.cos (B * x + C) + D ≤ 5) :
  D = 1 :=
by
  -- Here's where the proof would go
  sorry

end find_vertical_shift_l667_66730


namespace lee_charge_per_action_figure_l667_66709

def cost_of_sneakers : ℕ := 90
def amount_saved : ℕ := 15
def action_figures_sold : ℕ := 10
def amount_left_after_purchase : ℕ := 25
def amount_charged_per_action_figure : ℕ := 10

theorem lee_charge_per_action_figure :
  (cost_of_sneakers - amount_saved + amount_left_after_purchase = 
  action_figures_sold * amount_charged_per_action_figure) :=
by
  -- The proof steps will go here, but they are not required in the statement.
  sorry

end lee_charge_per_action_figure_l667_66709


namespace evaluate_expression_l667_66761

def a : ℚ := 7/3

theorem evaluate_expression :
  (4 * a^2 - 11 * a + 7) * (2 * a - 3) = 140 / 27 :=
by
  sorry

end evaluate_expression_l667_66761


namespace find_abc_sum_l667_66716

theorem find_abc_sum (a b c : ℤ) (h1 : a - 2 * b = 4) (h2 : a * b + c^2 - 1 = 0) :
  a + b + c = 5 ∨ a + b + c = 3 ∨ a + b + c = -1 ∨ a + b + c = -3 :=
  sorry

end find_abc_sum_l667_66716


namespace part_a_part_b_l667_66760

def square_side_length : ℝ := 10
def square_area (side_length : ℝ) : ℝ := side_length * side_length
def triangle_area (base : ℝ) (height : ℝ) : ℝ := 0.5 * base * height

-- Part (a)
theorem part_a :
  let side_length := square_side_length
  let square := square_area side_length
  let triangle := triangle_area side_length side_length
  square - triangle = 50 := by
  sorry

-- Part (b)
theorem part_b :
  let side_length := square_side_length
  let square := square_area side_length
  let small_triangle_area := square / 8
  2 * small_triangle_area = 25 := by
  sorry

end part_a_part_b_l667_66760


namespace parallelogram_side_length_l667_66756

theorem parallelogram_side_length 
  (s : ℝ) 
  (A : ℝ)
  (angle : ℝ)
  (adj1 adj2 : ℝ) 
  (h : adj1 = s) 
  (h1 : adj2 = 2 * s) 
  (h2 : angle = 30)
  (h3 : A = 8 * Real.sqrt 3): 
  s = 2 * Real.sqrt 2 :=
by
  -- sorry to skip proofs
  sorry

end parallelogram_side_length_l667_66756


namespace range_of_a_l667_66784

open Set

variable {a : ℝ}
def M (a : ℝ) : Set ℝ := { x : ℝ | (2 * a - 1) < x ∧ x < (4 * a) }
def N : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

theorem range_of_a (h : N ⊆ M a) : 1 / 2 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l667_66784


namespace smaller_angle_formed_by_hands_at_3_15_l667_66752

def degrees_per_hour : ℝ := 30
def degrees_per_minute : ℝ := 6
def hour_hand_degrees_per_minute : ℝ := 0.5

def minute_position (minute : ℕ) : ℝ :=
  minute * degrees_per_minute

def hour_position (hour : ℕ) (minute : ℕ) : ℝ :=
  hour * degrees_per_hour + minute * hour_hand_degrees_per_minute

theorem smaller_angle_formed_by_hands_at_3_15 : 
  minute_position 15 = 90 ∧ 
  hour_position 3 15 = 97.5 →
  abs (hour_position 3 15 - minute_position 15) = 7.5 :=
by
  intros h
  sorry

end smaller_angle_formed_by_hands_at_3_15_l667_66752


namespace intersection_of_M_and_N_l667_66728

def M : Set ℤ := {0, 1}
def N : Set ℤ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end intersection_of_M_and_N_l667_66728


namespace three_pow_255_mod_7_l667_66706

theorem three_pow_255_mod_7 : 3^255 % 7 = 6 :=
by 
  have h1 : 3^1 % 7 = 3 := by norm_num
  have h2 : 3^2 % 7 = 2 := by norm_num
  have h3 : 3^3 % 7 = 6 := by norm_num
  have h4 : 3^4 % 7 = 4 := by norm_num
  have h5 : 3^5 % 7 = 5 := by norm_num
  have h6 : 3^6 % 7 = 1 := by norm_num
  sorry

end three_pow_255_mod_7_l667_66706


namespace interest_rate_b_to_c_l667_66794

open Real

noncomputable def calculate_rate_b_to_c (P : ℝ) (r1 : ℝ) (t : ℝ) (G : ℝ) : ℝ :=
  let I_a_b := P * (r1 / 100) * t
  let I_b_c := I_a_b + G
  (100 * I_b_c) / (P * t)

theorem interest_rate_b_to_c :
  calculate_rate_b_to_c 3200 12 5 400 = 14.5 := by
  sorry

end interest_rate_b_to_c_l667_66794


namespace third_cyclist_speed_l667_66710

theorem third_cyclist_speed (s1 s3 : ℝ) :
  (∃ s1 s3 : ℝ,
    (∀ t : ℝ, t > 0 → (s1 > s3) ∧ (20 = abs (10 * t - s1 * t)) ∧ (5 = abs (s1 * t - s3 * t)) ∧ (s1 ≥ 10))) →
  (s3 = 25 ∨ s3 = 5) :=
by sorry

end third_cyclist_speed_l667_66710


namespace zoo_animal_count_l667_66742

def tiger_enclosures : ℕ := 4
def zebra_enclosures_per_tiger_enclosures : ℕ := 2
def zebra_enclosures : ℕ := tiger_enclosures * zebra_enclosures_per_tiger_enclosures
def giraffe_enclosures_per_zebra_enclosures : ℕ := 3
def giraffe_enclosures : ℕ := zebra_enclosures * giraffe_enclosures_per_zebra_enclosures
def tigers_per_enclosure : ℕ := 4
def zebras_per_enclosure : ℕ := 10
def giraffes_per_enclosure : ℕ := 2

def total_animals_in_zoo : ℕ := 
    (tiger_enclosures * tigers_per_enclosure) + 
    (zebra_enclosures * zebras_per_enclosure) + 
    (giraffe_enclosures * giraffes_per_enclosure)

theorem zoo_animal_count : total_animals_in_zoo = 144 := 
by
  -- proof would go here
  sorry

end zoo_animal_count_l667_66742


namespace calculate_v_sum_l667_66770

def v (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem calculate_v_sum :
  v (2) + v (-2) + v (1) + v (-1) = 4 :=
by
  sorry

end calculate_v_sum_l667_66770


namespace temp_drop_of_8_deg_is_neg_8_l667_66778

theorem temp_drop_of_8_deg_is_neg_8 (rise_3_deg : ℤ) (h : rise_3_deg = 3) : ∀ drop_8_deg, drop_8_deg = -8 :=
by
  intros
  sorry

end temp_drop_of_8_deg_is_neg_8_l667_66778


namespace number_of_games_in_division_l667_66738

theorem number_of_games_in_division (P Q : ℕ) (h1 : P > 2 * Q) (h2 : Q > 6) (schedule_eq : 4 * P + 5 * Q = 82) : 4 * P = 52 :=
by sorry

end number_of_games_in_division_l667_66738


namespace length_AE_l667_66746

/-- Given points A, B, C, D, and E on a plane with distances:
  - CA = 12,
  - AB = 8,
  - BC = 4,
  - CD = 5,
  - DB = 3,
  - BE = 6,
  - ED = 3.
  Prove that AE = sqrt 113.
--/
theorem length_AE (A B C D E : ℝ × ℝ)
  (h1 : dist C A = 12)
  (h2 : dist A B = 8)
  (h3 : dist B C = 4)
  (h4 : dist C D = 5)
  (h5 : dist D B = 3)
  (h6 : dist B E = 6)
  (h7 : dist E D = 3) : 
  dist A E = Real.sqrt 113 := 
  by 
    sorry

end length_AE_l667_66746


namespace min_product_of_three_numbers_l667_66713

def SetOfNumbers : Set ℤ := {-9, -5, -1, 1, 3, 5, 8}

theorem min_product_of_three_numbers : 
  ∃ (a b c : ℤ), a ∈ SetOfNumbers ∧ b ∈ SetOfNumbers ∧ c ∈ SetOfNumbers ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = -360 :=
by {
  sorry
}

end min_product_of_three_numbers_l667_66713


namespace cash_price_eq_8000_l667_66702

noncomputable def cash_price (d m s : ℕ) : ℕ :=
  d + 30 * m - s

theorem cash_price_eq_8000 :
  cash_price 3000 300 4000 = 8000 :=
by
  -- Proof omitted.
  sorry

end cash_price_eq_8000_l667_66702


namespace rectangle_width_l667_66724

theorem rectangle_width (w l A : ℕ) 
  (h1 : l = 3 * w)
  (h2 : A = l * w)
  (h3 : A = 108) : 
  w = 6 := 
sorry

end rectangle_width_l667_66724


namespace pythagorean_triplet_l667_66723

theorem pythagorean_triplet (k : ℕ) :
  let a := k
  let b := 2 * k - 2
  let c := 2 * k - 1
  (a * b) ^ 2 + c ^ 2 = (2 * k ^ 2 - 2 * k + 1) ^ 2 :=
by
  sorry

end pythagorean_triplet_l667_66723


namespace average_height_correct_l667_66735

noncomputable def initially_calculated_average_height 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) 
  (A : ℝ) : Prop :=
  let incorrect_sum := num_students * A
  let height_difference := incorrect_height - correct_height
  let actual_sum := num_students * actual_average
  incorrect_sum = actual_sum + height_difference

theorem average_height_correct 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) :
  initially_calculated_average_height num_students incorrect_height correct_height actual_average 175 :=
by {
  sorry
}

end average_height_correct_l667_66735


namespace find_right_triangle_sides_l667_66797

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_condition (a b c : ℕ) : Prop :=
  a * b = 3 * (a + b + c)

theorem find_right_triangle_sides :
  ∃ (a b c : ℕ),
    is_right_triangle a b c ∧ area_condition a b c ∧
    ((a = 7 ∧ b = 24 ∧ c = 25) ∨
     (a = 8 ∧ b = 15 ∧ c = 17) ∨
     (a = 9 ∧ b = 12 ∧ c = 15)) :=
sorry

end find_right_triangle_sides_l667_66797
