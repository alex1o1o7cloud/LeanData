import Mathlib

namespace find_h_neg_one_l139_139886

theorem find_h_neg_one (h : ℝ → ℝ) (H : ∀ x, (x^7 - 1) * h x = (x + 1) * (x^2 + 1) * (x^4 + 1) + 1) : 
  h (-1) = 1 := 
by 
  sorry

end find_h_neg_one_l139_139886


namespace solve_system_l139_139258

-- Define the system of equations
def eq1 (x y : ℚ) : Prop := 4 * x - 3 * y = -10
def eq2 (x y : ℚ) : Prop := 6 * x + 5 * y = -13

-- Define the solution
def solution (x y : ℚ) : Prop := x = -89 / 38 ∧ y = 0.21053

-- Prove that the given solution satisfies both equations
theorem solve_system : ∃ x y : ℚ, eq1 x y ∧ eq2 x y ∧ solution x y :=
by
  sorry

end solve_system_l139_139258


namespace last_digit_3_pow_1991_plus_1991_pow_3_l139_139098

theorem last_digit_3_pow_1991_plus_1991_pow_3 :
  (3 ^ 1991 + 1991 ^ 3) % 10 = 8 :=
  sorry

end last_digit_3_pow_1991_plus_1991_pow_3_l139_139098


namespace circle_center_l139_139165

theorem circle_center (x y : ℝ) : (x - 2)^2 + (y + 1)^2 = 3 → (2, -1) = (2, -1) :=
by
  intro h
  -- Proof omitted
  sorry

end circle_center_l139_139165


namespace problem_l139_139873

noncomputable def f (x a b : ℝ) := x^2 + a*x + b
noncomputable def g (x c d : ℝ) := x^2 + c*x + d

theorem problem (a b c d : ℝ) (h_min_f : f (-a/2) a b = -25) (h_min_g : g (-c/2) c d = -25)
  (h_intersection_f : f 50 a b = -50) (h_intersection_g : g 50 c d = -50)
  (h_root_f_of_g : g (-a/2) c d = 0) (h_root_g_of_f : f (-c/2) a b = 0) :
  a + c = -200 := by
  sorry

end problem_l139_139873


namespace arithmetic_sequence_l139_139584

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h_n : n > 0) 
  (h_Sn : S (2 * n) - S (2 * n - 1) + a 2 = 424) : 
  a (n + 1) = 212 :=
sorry

end arithmetic_sequence_l139_139584


namespace find_g_seven_l139_139122

noncomputable def g : ℝ → ℝ :=
  sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_six : g 6 = 7

theorem find_g_seven : g 7 = 49 / 6 :=
by
  -- Proof omitted here
  sorry

end find_g_seven_l139_139122


namespace fraction_inequality_l139_139171

theorem fraction_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (1 / a) + (1 / b) ≥ (4 / (a + b)) :=
by 
-- Skipping the proof using 'sorry'
sorry

end fraction_inequality_l139_139171


namespace value_of_2a_plus_b_l139_139875

theorem value_of_2a_plus_b : ∀ (a b : ℝ), (∀ x : ℝ, x^2 - 4*x + 7 = 19 → (x = a ∨ x = b)) → a ≥ b → 2 * a + b = 10 :=
by
  intros a b h_sol h_order
  sorry

end value_of_2a_plus_b_l139_139875


namespace no_infinite_prime_sequence_l139_139650

theorem no_infinite_prime_sequence (p : ℕ) (h_prime : Nat.Prime p) :
  ¬(∃ (p_seq : ℕ → ℕ), (∀ n, Nat.Prime (p_seq n)) ∧ (∀ n, p_seq (n + 1) = 2 * p_seq n + 1)) :=
by
  sorry

end no_infinite_prime_sequence_l139_139650


namespace value_of_ratios_l139_139884

variable (x y z : ℝ)

-- Conditions
def geometric_sequence : Prop :=
  4 * y / (3 * x) = 5 * z / (4 * y)

def arithmetic_sequence : Prop :=
  2 / y = 1 / x + 1 / z

-- Theorem/Proof Statement
theorem value_of_ratios (h1 : geometric_sequence x y z) (h2 : arithmetic_sequence x y z) :
  (x / z) + (z / x) = 34 / 15 :=
by
  sorry

end value_of_ratios_l139_139884


namespace triangle_one_interior_angle_61_degrees_l139_139267

theorem triangle_one_interior_angle_61_degrees
  (x : ℝ) : 
  (x + 75 + 2 * x + 25 + 3 * x - 22 = 360) → 
  (1 / 2 * (2 * x + 25) = 61 ∨ 
   1 / 2 * (3 * x - 22) = 61 ∨ 
   1 / 2 * (x + 75) = 61) :=
by
  intros h_sum
  sorry

end triangle_one_interior_angle_61_degrees_l139_139267


namespace solve_for_x_l139_139531

theorem solve_for_x (x : ℝ) : x^2 + 6 * x + 8 = -(x + 4) * (x + 6) ↔ x = -4 :=
by {
  sorry
}

end solve_for_x_l139_139531


namespace rabbit_can_escape_l139_139847

def RabbitEscapeExists
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : Prop :=
 ∃ (rabbit_escapes : Bool), rabbit_escapes = true

theorem rabbit_can_escape
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : RabbitEscapeExists center_x center_y wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 wolf_speed rabbit_speed condition1 condition2 condition3 := 
sorry

end rabbit_can_escape_l139_139847


namespace find_matches_in_second_set_l139_139740

-- Conditions defined as Lean variables
variables (x : ℕ)
variables (avg_first_20 : ℚ := 40)
variables (avg_second_x : ℚ := 20)
variables (avg_all_30 : ℚ := 100 / 3)
variables (total_first_20 : ℚ := 20 * avg_first_20)
variables (total_all_30 : ℚ := 30 * avg_all_30)

-- Proof statement (question) along with conditions
theorem find_matches_in_second_set (x_value : x = 10) :
  avg_first_20 = 40 ∧ avg_second_x = 20 ∧ avg_all_30 = 100 / 3 →
  20 * avg_first_20 + x * avg_second_x = 30 * avg_all_30 → x = 10 := 
sorry

end find_matches_in_second_set_l139_139740


namespace final_movie_length_l139_139659

-- Definitions based on conditions
def original_movie_length : ℕ := 60
def cut_scene_length : ℕ := 3

-- Theorem statement proving the final length of the movie
theorem final_movie_length : original_movie_length - cut_scene_length = 57 :=
by
  -- The proof will go here
  sorry

end final_movie_length_l139_139659


namespace cryptarithm_problem_l139_139415

theorem cryptarithm_problem (F E D : ℤ) (h1 : F - E = D - 1) (h2 : D + E + F = 16) (h3 : F - E = D) : 
    F - E = 5 :=
by sorry

end cryptarithm_problem_l139_139415


namespace average_percentage_decrease_l139_139423

theorem average_percentage_decrease :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 100 * (1 - x)^2 = 81 ∧ x = 0.1 :=
by
  sorry

end average_percentage_decrease_l139_139423


namespace find_a_extreme_values_l139_139804

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + 4 * Real.log (x + 1)
noncomputable def f' (x a : ℝ) : ℝ := 2 * (x - a) + 4 / (x + 1)

-- Given conditions
theorem find_a (a : ℝ) :
  f' 1 a = 0 ↔ a = 2 :=
by
  sorry

theorem extreme_values :
  ∃ x : ℝ, -1 < x ∧ f (0 : ℝ) 2 = 4 ∨ f (1 : ℝ) 2 = 1 + 4 * Real.log 2 :=
by
  sorry

end find_a_extreme_values_l139_139804


namespace simplify_expression_l139_139367

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l139_139367


namespace wuzhen_conference_arrangements_l139_139927

theorem wuzhen_conference_arrangements 
  (countries : Finset ℕ)
  (hotels : Finset ℕ)
  (h_countries_count : countries.card = 5)
  (h_hotels_count : hotels.card = 3) :
  ∃ f : ℕ → ℕ,
  (∀ c ∈ countries, f c ∈ hotels) ∧
  (∀ h ∈ hotels, ∃ c ∈ countries, f c = h) ∧
  (Finset.card (Set.toFinset (f '' countries)) = 3) ∧
  ∃ n : ℕ,
  n = 150 := 
sorry

end wuzhen_conference_arrangements_l139_139927


namespace solve_x_in_equation_l139_139691

theorem solve_x_in_equation (x : ℕ) (h : x + (x + 1) + (x + 2) + (x + 3) = 18) : x = 3 :=
by
  sorry

end solve_x_in_equation_l139_139691


namespace percentage_supports_policy_l139_139556

theorem percentage_supports_policy
    (men_support_percentage : ℝ)
    (women_support_percentage : ℝ)
    (num_men : ℕ)
    (num_women : ℕ)
    (total_surveyed : ℕ)
    (total_supporters : ℕ)
    (overall_percentage : ℝ) :
    (men_support_percentage = 0.70) →
    (women_support_percentage = 0.75) →
    (num_men = 200) →
    (num_women = 800) →
    (total_surveyed = num_men + num_women) →
    (total_supporters = (men_support_percentage * num_men) + (women_support_percentage * num_women)) →
    (overall_percentage = (total_supporters / total_surveyed) * 100) →
    overall_percentage = 74 :=
by
  intros
  sorry

end percentage_supports_policy_l139_139556


namespace pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l139_139639

noncomputable def pit_a_no_replant_prob : ℝ := 0.875
noncomputable def one_pit_no_replant_prob : ℝ := 0.713
noncomputable def at_least_one_pit_replant_prob : ℝ := 0.330

theorem pit_A_no_replant (p : ℝ) (h1 : p = 0.5) : pit_a_no_replant_prob = 1 - (1 - p)^3 := by
  sorry

theorem exactly_one_pit_no_replant (p : ℝ) (h1 : p = 0.5) : one_pit_no_replant_prob = 1 - 3 * (1 - p)^3 * (p^3)^(2) := by
  sorry

theorem at_least_one_replant (p : ℝ) (h1 : p = 0.5) : at_least_one_pit_replant_prob = 1 - (1 - (1 - p)^3)^3 := by
  sorry

end pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l139_139639


namespace find_x_plus_y_l139_139291

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2023) 
                           (h2 : x + 2023 * Real.sin y = 2022) 
                           (h3 : (Real.pi / 2) ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2023 + Real.pi / 2 :=
sorry

end find_x_plus_y_l139_139291


namespace vertex_at_fixed_point_l139_139791

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 + 1

theorem vertex_at_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 2 :=
by
  sorry

end vertex_at_fixed_point_l139_139791


namespace find_m_l139_139241

def g (n : Int) : Int :=
  if n % 2 ≠ 0 then n + 5 else 
  if n % 3 = 0 then n / 3 else n

theorem find_m (m : Int) 
  (h_odd : m % 2 ≠ 0) 
  (h_ggg : g (g (g m)) = 35) : 
  m = 85 := 
by
  sorry

end find_m_l139_139241


namespace sales_tax_percentage_l139_139293

theorem sales_tax_percentage (total_amount : ℝ) (tip_percentage : ℝ) (food_price : ℝ) (tax_percentage : ℝ) : 
  total_amount = 158.40 ∧ tip_percentage = 0.20 ∧ food_price = 120 → tax_percentage = 0.10 :=
by
  intros h
  sorry

end sales_tax_percentage_l139_139293


namespace committee_count_l139_139534

theorem committee_count (total_students : ℕ) (include_students : ℕ) (choose_students : ℕ) :
  total_students = 8 → include_students = 2 → choose_students = 3 →
  Nat.choose (total_students - include_students) choose_students = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end committee_count_l139_139534


namespace annual_growth_rate_l139_139184

theorem annual_growth_rate (x : ℝ) (h : 2000 * (1 + x) ^ 2 = 2880) : x = 0.2 :=
by sorry

end annual_growth_rate_l139_139184


namespace no_equal_numbers_from_19_and_98_l139_139211

theorem no_equal_numbers_from_19_and_98 :
  ¬ (∃ s : ℕ, ∃ (a b : ℕ → ℕ), 
       (a 0 = 19) ∧ (b 0 = 98) ∧
       (∀ k, a (k + 1) = a k * a k ∨ a (k + 1) = a k + 1) ∧
       (∀ k, b (k + 1) = b k * b k ∨ b (k + 1) = b k + 1) ∧
       a s = b s) :=
sorry

end no_equal_numbers_from_19_and_98_l139_139211


namespace initial_percentage_increase_l139_139375

theorem initial_percentage_increase (W R : ℝ) (P : ℝ) 
  (h1 : R = W * (1 + P / 100)) 
  (h2 : R * 0.75 = W * 1.3500000000000001) : P = 80 := 
by
  sorry

end initial_percentage_increase_l139_139375


namespace simplify_expression_l139_139785

theorem simplify_expression (x y : ℝ) (P Q : ℝ) (hP : P = 2 * x + 3 * y) (hQ : Q = 3 * x + 2 * y) :
  ((P + Q) / (P - Q)) - ((P - Q) / (P + Q)) = (24 * x ^ 2 + 52 * x * y + 24 * y ^ 2) / (5 * x * y - 5 * y ^ 2) :=
by
  sorry

end simplify_expression_l139_139785


namespace seating_arrangements_count_is_134_l139_139150

theorem seating_arrangements_count_is_134 (front_row_seats : ℕ) (back_row_seats : ℕ) (valid_arrangements_with_no_next_to_each_other : ℕ) : 
  front_row_seats = 6 → back_row_seats = 7 → valid_arrangements_with_no_next_to_each_other = 134 :=
by
  intros h1 h2
  sorry

end seating_arrangements_count_is_134_l139_139150


namespace tony_drive_time_l139_139289

noncomputable def time_to_first_friend (d₁ d₂ t₂ : ℝ) : ℝ :=
  let v := d₂ / t₂
  d₁ / v

theorem tony_drive_time (d₁ d₂ t₂ : ℝ) (h_d₁ : d₁ = 120) (h_d₂ : d₂ = 200) (h_t₂ : t₂ = 5) : 
    time_to_first_friend d₁ d₂ t₂ = 3 := by
  rw [h_d₁, h_d₂, h_t₂]
  -- Further simplification would follow here based on the proof steps, which we are omitting
  sorry

end tony_drive_time_l139_139289


namespace parabola_focus_coordinates_l139_139525

-- Define the given conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def passes_through (a : ℝ) (p : ℝ × ℝ) : Prop := p.snd = parabola a p.fst

-- Main theorem: Prove the coordinates of the focus
theorem parabola_focus_coordinates (a : ℝ) (h : passes_through a (1, 4)) (ha : a = 4) : (0, 1 / 16) = (0, 1 / (4 * a)) :=
by
  rw [ha] -- substitute the value of a
  simp -- simplify the expression
  sorry

end parabola_focus_coordinates_l139_139525


namespace boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l139_139495

-- Problem 1: Specific case
theorem boat_and_current_speed (x y : ℝ) 
  (h1 : 3 * (x + y) = 75) 
  (h2 : 5 * (x - y) = 75) : 
  x = 20 ∧ y = 5 := 
sorry

-- Problem 2: General case
theorem boat_and_current_speed_general (x y : ℝ) (a b S : ℝ) 
  (h1 : a * (x + y) = S) 
  (h2 : b * (x - y) = S) : 
  x = (a + b) * S / (2 * a * b) ∧ 
  y = (b - a) * S / (2 * a * b) := 
sorry

theorem log_drift_time (y S a b : ℝ)
  (h_y : y = (b - a) * S / (2 * a * b)) : 
  S / y = 2 * a * b / (b - a) := 
sorry

end boat_and_current_speed_boat_and_current_speed_general_log_drift_time_l139_139495


namespace acute_triangle_l139_139491

theorem acute_triangle (a b c : ℝ) (n : ℕ) (h_n : 2 < n) (h_eq : a^n + b^n = c^n) : a^2 + b^2 > c^2 :=
sorry

end acute_triangle_l139_139491


namespace multiply_powers_zero_exponent_distribute_term_divide_powers_l139_139704

-- 1. Prove a^{2} \cdot a^{3} = a^{5}
theorem multiply_powers (a : ℝ) : a^2 * a^3 = a^5 := 
sorry

-- 2. Prove (3.142 - π)^{0} = 1
theorem zero_exponent : (3.142 - Real.pi)^0 = 1 := 
sorry

-- 3. Prove 2a(a^{2} - 1) = 2a^{3} - 2a
theorem distribute_term (a : ℝ) : 2 * a * (a^2 - 1) = 2 * a^3 - 2 * a := 
sorry

-- 4. Prove (-m^{3})^{2} \div m^{4} = m^{2}
theorem divide_powers (m : ℝ) : ((-m^3)^2) / (m^4) = m^2 := 
sorry

end multiply_powers_zero_exponent_distribute_term_divide_powers_l139_139704


namespace min_overlap_l139_139535

variable (P : Set ℕ → ℝ)
variable (B M : Set ℕ)

-- Conditions
def P_B_def : P B = 0.95 := sorry
def P_M_def : P M = 0.85 := sorry

-- To Prove
theorem min_overlap : P (B ∩ M) = 0.80 := sorry

end min_overlap_l139_139535


namespace cone_plane_distance_l139_139827

theorem cone_plane_distance (H α : ℝ) : 
  (x = 2 * H * (Real.sin (α / 4)) ^ 2) :=
sorry

end cone_plane_distance_l139_139827


namespace right_triangle_exists_l139_139657

-- Define the setup: equilateral triangle ABC, point P, and angle condition
def Point (α : Type*) := α 
def inside {α : Type*} (p : Point α) (A B C : Point α) : Prop := sorry
def angle_at {α : Type*} (p q r : Point α) (θ : ℝ) : Prop := sorry
noncomputable def PA {α : Type*} (P A : Point α) : ℝ := sorry
noncomputable def PB {α : Type*} (P B : Point α) : ℝ := sorry
noncomputable def PC {α : Type*} (P C : Point α) : ℝ := sorry

-- Theorem we need to prove
theorem right_triangle_exists {α : Type*} 
  (A B C P : Point α)
  (h1 : inside P A B C)
  (h2 : angle_at P A B 150) :
  ∃ (Q : Point α), angle_at P Q B 90 :=
sorry

end right_triangle_exists_l139_139657


namespace trisha_total_distance_l139_139027

theorem trisha_total_distance :
  let d1 := 0.1111111111111111
  let d2 := 0.1111111111111111
  let d3 := 0.6666666666666666
  d1 + d2 + d3 = 0.8888888888888888 := 
by
  sorry

end trisha_total_distance_l139_139027


namespace division_expression_result_l139_139174

theorem division_expression_result :
  -1 / (-5) / (-1 / 5) = -1 :=
by sorry

end division_expression_result_l139_139174


namespace triangle_ABC_is_acute_l139_139032

noncomputable def arithmeticSeqTerm (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def geometricSeqTerm (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r^(n - 1)

def tanA_condition (a1 d : ℝ) :=
  arithmeticSeqTerm a1 d 3 = -4 ∧ arithmeticSeqTerm a1 d 7 = 4

def tanB_condition (a1 r : ℝ) :=
  geometricSeqTerm a1 r 3 = 1/3 ∧ geometricSeqTerm a1 r 6 = 9

theorem triangle_ABC_is_acute {A B : ℝ} (a1a da a1b rb : ℝ) 
  (hA : tanA_condition a1a da) 
  (hB : tanB_condition a1b rb) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ (A + B) < π :=
  sorry

end triangle_ABC_is_acute_l139_139032


namespace Tim_paid_amount_l139_139217

theorem Tim_paid_amount (original_price : ℝ) (discount_percentage : ℝ) (discounted_price : ℝ) 
    (h1 : original_price = 1200) (h2 : discount_percentage = 0.15) 
    (discount_amount : ℝ) (h3 : discount_amount = original_price * discount_percentage) 
    (h4 : discounted_price = original_price - discount_amount) : discounted_price = 1020 := 
    by {
        sorry
    }

end Tim_paid_amount_l139_139217


namespace sum_of_ages_is_12_l139_139557

-- Let Y be the age of the youngest child
def Y : ℝ := 1.5

-- Let the ages of the other children
def age2 : ℝ := Y + 1
def age3 : ℝ := Y + 2
def age4 : ℝ := Y + 3

-- Define the sum of the ages
def sum_of_ages : ℝ := Y + age2 + age3 + age4

-- The theorem to prove the sum of the ages is 12 years
theorem sum_of_ages_is_12 : sum_of_ages = 12 :=
by
  -- The detailed proof is to be filled in later, currently skipped.
  sorry

end sum_of_ages_is_12_l139_139557


namespace largest_divisor_of_even_square_difference_l139_139224

theorem largest_divisor_of_even_square_difference (m n : ℕ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) :
  ∃ (k : ℕ), k = 8 ∧ ∀ m n : ℕ, m % 2 = 0 → n % 2 = 0 → n < m → k ∣ (m^2 - n^2) := by
  sorry

end largest_divisor_of_even_square_difference_l139_139224


namespace greatest_missed_problems_l139_139287

theorem greatest_missed_problems (total_problems : ℕ) (passing_percentage : ℝ) (missed_problems : ℕ) : 
  total_problems = 50 ∧ passing_percentage = 0.85 → missed_problems = 7 :=
by
  sorry

end greatest_missed_problems_l139_139287


namespace intersection_A_B_range_m_l139_139062

-- Definitions for Sets A, B, and C
def SetA : Set ℝ := { x | -2 ≤ x ∧ x < 5 }
def SetB : Set ℝ := { x | 3 * x - 5 ≥ x - 1 }
def SetC (m : ℝ) : Set ℝ := { x | -x + m > 0 }

-- Problem 1: Prove \( A \cap B = \{ x \mid 2 \leq x < 5 \} \)
theorem intersection_A_B : SetA ∩ SetB = { x : ℝ | 2 ≤ x ∧ x < 5 } :=
by
  sorry

-- Problem 2: Prove \( m \in [5, +\infty) \) given \( A \cup C = C \)
theorem range_m (m : ℝ) : (SetA ∪ SetC m = SetC m) → m ∈ Set.Ici 5 :=
by
  sorry

end intersection_A_B_range_m_l139_139062


namespace solve_inequality_solve_system_of_inequalities_l139_139097

-- Inequality proof problem
theorem solve_inequality (x : ℝ) (h : (2*x - 3)/3 > (3*x + 1)/6 - 1) : x > 1 := by
  sorry

-- System of inequalities proof problem
theorem solve_system_of_inequalities (x : ℝ) (h1 : x ≤ 3*x - 6) (h2 : 3*x + 1 > 2*(x - 1)) : x ≥ 3 := by
  sorry

end solve_inequality_solve_system_of_inequalities_l139_139097


namespace literature_club_students_neither_english_nor_french_l139_139418

theorem literature_club_students_neither_english_nor_french
  (total_students english_students french_students both_students : ℕ)
  (h1 : total_students = 120)
  (h2 : english_students = 72)
  (h3 : french_students = 52)
  (h4 : both_students = 12) :
  (total_students - ((english_students - both_students) + (french_students - both_students) + both_students) = 8) :=
by
  sorry

end literature_club_students_neither_english_nor_french_l139_139418


namespace LineDoesNotIntersectParabola_sum_r_s_l139_139118

noncomputable def r : ℝ := -0.6
noncomputable def s : ℝ := 40.6
def Q : ℝ × ℝ := (10, -6)
def line_through_Q_with_slope (m : ℝ) (p : ℝ × ℝ) : ℝ := m * p.1 - 10 * m - 6
def parabola (x : ℝ) : ℝ := 2 * x^2

theorem LineDoesNotIntersectParabola (m : ℝ) :
  r < m ∧ m < s ↔ (m^2 - 4 * 2 * (10 * m + 6) < 0) :=
by sorry

theorem sum_r_s : r + s = 40 :=
by sorry

end LineDoesNotIntersectParabola_sum_r_s_l139_139118


namespace factorization_of_1386_l139_139548

-- We start by defining the number and the requirements.
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def factors_mult (a b : ℕ) : Prop := a * b = 1386
def factorization_count (count : ℕ) : Prop :=
  ∃ (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors_mult a b ∧ 
  (∀ c d, is_two_digit c ∧ is_two_digit d ∧ factors_mult c d → 
  (c = a ∧ d = b ∨ c = b ∧ d = a) → c = a ∧ d = b ∨ c = b ∧ d = a) ∧
  count = 4

-- Now, we state the theorem.
theorem factorization_of_1386 : factorization_count 4 :=
sorry

end factorization_of_1386_l139_139548


namespace bike_cost_l139_139717

theorem bike_cost (days_in_two_weeks : ℕ) 
  (bracelets_per_day : ℕ)
  (price_per_bracelet : ℕ)
  (total_bracelets : ℕ)
  (total_money : ℕ) 
  (h1 : days_in_two_weeks = 2 * 7)
  (h2 : bracelets_per_day = 8)
  (h3 : price_per_bracelet = 1)
  (h4 : total_bracelets = days_in_two_weeks * bracelets_per_day)
  (h5 : total_money = total_bracelets * price_per_bracelet) :
  total_money = 112 :=
sorry

end bike_cost_l139_139717


namespace pauline_total_spending_l139_139809

theorem pauline_total_spending
  (total_before_tax : ℝ)
  (sales_tax_rate : ℝ)
  (h₁ : total_before_tax = 150)
  (h₂ : sales_tax_rate = 0.08) :
  total_before_tax + total_before_tax * sales_tax_rate = 162 :=
by {
  -- Proof here
  sorry
}

end pauline_total_spending_l139_139809


namespace ceil_sqrt_fraction_eq_neg2_l139_139330

theorem ceil_sqrt_fraction_eq_neg2 :
  (Int.ceil (-Real.sqrt (36 / 9))) = -2 :=
by
  sorry

end ceil_sqrt_fraction_eq_neg2_l139_139330


namespace complement_of_A_in_U_l139_139637

def U : Set ℝ := {x | x ≤ 1}
def A : Set ℝ := {x | x < 0}

theorem complement_of_A_in_U : (U \ A) = {x | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end complement_of_A_in_U_l139_139637


namespace min_ties_to_ensure_pairs_l139_139729

variable (red blue green yellow : Nat)
variable (total_ties : Nat)
variable (pairs_needed : Nat)

-- Define the conditions
def conditions : Prop :=
  red = 120 ∧
  blue = 90 ∧
  green = 70 ∧
  yellow = 50 ∧
  total_ties = 27 ∧
  pairs_needed = 12

-- Define the statement to be proven
theorem min_ties_to_ensure_pairs : conditions red blue green yellow total_ties pairs_needed → total_ties = 27 :=
sorry

end min_ties_to_ensure_pairs_l139_139729


namespace max_sum_first_n_terms_is_S_5_l139_139837

open Nat

-- Define the arithmetic sequence and the conditions.
variable {a : ℕ → ℝ} -- The arithmetic sequence {a_n}
variable {d : ℝ} -- The common difference of the arithmetic sequence
variable {S : ℕ → ℝ} -- The sum of the first n terms of the sequence a

-- Hypotheses corresponding to the conditions in the problem
lemma a_5_positive : a 5 > 0 := sorry
lemma a_4_plus_a_7_negative : a 4 + a 7 < 0 := sorry

-- Statement to prove that the maximum value of the sum of the first n terms is S_5 given the conditions
theorem max_sum_first_n_terms_is_S_5 :
  (∀ (n : ℕ), S n ≤ S 5) :=
sorry

end max_sum_first_n_terms_is_S_5_l139_139837


namespace music_library_avg_disk_space_per_hour_l139_139324

theorem music_library_avg_disk_space_per_hour 
  (days_of_music: ℕ) (total_space_MB: ℕ) (hours_in_day: ℕ) 
  (h1: days_of_music = 15) 
  (h2: total_space_MB = 18000) 
  (h3: hours_in_day = 24) : 
  (total_space_MB / (days_of_music * hours_in_day)) = 50 := 
by
  sorry

end music_library_avg_disk_space_per_hour_l139_139324


namespace simplify_and_evaluate_l139_139502

theorem simplify_and_evaluate :
  let x := (-1 : ℚ) / 2
  3 * x^2 - (5 * x - 3 * (2 * x - 1) + 7 * x^2) = -9 / 2 :=
by
  let x : ℚ := (-1 : ℚ) / 2
  sorry

end simplify_and_evaluate_l139_139502


namespace four_digit_number_sum_eq_4983_l139_139395

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  1000 * d4 + 100 * d3 + 10 * d2 + d1

theorem four_digit_number_sum_eq_4983 (n : ℕ) :
  n + reverse_number n = 4983 ↔ n = 1992 ∨ n = 2991 :=
by sorry

end four_digit_number_sum_eq_4983_l139_139395


namespace product_of_roots_of_cubic_polynomial_l139_139281

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l139_139281


namespace min_ν_of_cubic_eq_has_3_positive_real_roots_l139_139407

open Real

noncomputable def cubic_eq (x θ : ℝ) : ℝ :=
  x^3 * sin θ - (sin θ + 2) * x^2 + 6 * x - 4

noncomputable def ν (θ : ℝ) : ℝ :=
  (9 * sin θ ^ 2 - 4 * sin θ + 3) / 
  ((1 - cos θ) * (2 * cos θ - 6 * sin θ - 3 * sin (2 * θ) + 2))

theorem min_ν_of_cubic_eq_has_3_positive_real_roots :
  (∀ x:ℝ, cubic_eq x θ = 0 → 0 < x) →
  ν θ = 621 / 8 :=
sorry

end min_ν_of_cubic_eq_has_3_positive_real_roots_l139_139407


namespace zoo_pandas_l139_139067

-- Defining the conditions
variable (total_couples : ℕ)
variable (pregnant_couples : ℕ)
variable (baby_pandas : ℕ)
variable (total_pandas : ℕ)

-- Given conditions
def paired_mates : Prop := ∃ c : ℕ, c = total_couples

def pregnant_condition : Prop := pregnant_couples = (total_couples * 25) / 100

def babies_condition : Prop := baby_pandas = 2

def total_condition : Prop := total_pandas = total_couples * 2 + baby_pandas

-- The theorem to be proven
theorem zoo_pandas (h1 : paired_mates total_couples)
                   (h2 : pregnant_condition total_couples pregnant_couples)
                   (h3 : babies_condition baby_pandas)
                   (h4 : pregnant_couples = 2) :
                   total_condition total_couples baby_pandas total_pandas :=
by sorry

end zoo_pandas_l139_139067


namespace quadratic_root_relation_l139_139003

theorem quadratic_root_relation (x₁ x₂ : ℝ) (h₁ : x₁ ^ 2 - 3 * x₁ + 2 = 0) (h₂ : x₂ ^ 2 - 3 * x₂ + 2 = 0) :
  x₁ + x₂ - x₁ * x₂ = 1 := by
sorry

end quadratic_root_relation_l139_139003


namespace find_a_l139_139177

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a / ((Real.exp (2 * x)) - 1)

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, f a x = -f a (-x)) → a = 2 :=
by
  sorry

end find_a_l139_139177


namespace max_sum_of_factors_l139_139162

theorem max_sum_of_factors (x y : ℕ) (h1 : x * y = 48) (h2 : x ≠ y) : x + y ≤ 49 :=
by
  sorry

end max_sum_of_factors_l139_139162


namespace evaluate_expression_l139_139136

theorem evaluate_expression : -1 ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 = -2 :=
by
  -- sorry is added as a placeholder for the proof steps
  sorry

end evaluate_expression_l139_139136


namespace decreasing_interval_ln_quadratic_l139_139009

theorem decreasing_interval_ln_quadratic :
  ∀ x : ℝ, (x < 1 ∨ x > 3) → (∀ a b : ℝ, (a ≤ b) → (a < 1 ∨ a > 3) → (b < 1 ∨ b > 3) → (a ≤ x ∧ x ≤ b → (x^2 - 4 * x + 3) ≥ (b^2 - 4 * b + 3))) :=
by
  sorry

end decreasing_interval_ln_quadratic_l139_139009


namespace hammers_ordered_in_october_l139_139864

theorem hammers_ordered_in_october
  (ordered_in_june : Nat)
  (ordered_in_july : Nat)
  (ordered_in_august : Nat)
  (ordered_in_september : Nat)
  (pattern_increase : ∀ n : Nat, ordered_in_june + n = ordered_in_july ∧ ordered_in_july + (n + 1) = ordered_in_august ∧ ordered_in_august + (n + 2) = ordered_in_september) :
  ordered_in_september + 4 = 13 :=
by
  -- Proof omitted
  sorry

end hammers_ordered_in_october_l139_139864


namespace best_choice_to_calculate_89_8_sq_l139_139159

theorem best_choice_to_calculate_89_8_sq 
  (a b c d : ℚ) 
  (h1 : (89 + 0.8)^2 = a) 
  (h2 : (80 + 9.8)^2 = b) 
  (h3 : (90 - 0.2)^2 = c) 
  (h4 : (100 - 10.2)^2 = d) : 
  c = 89.8^2 := by
  sorry

end best_choice_to_calculate_89_8_sq_l139_139159


namespace min_value_expr_l139_139617

theorem min_value_expr (x : ℝ) (h : x > 1) : ∃ m, m = 5 ∧ ∀ y, y = x + 4 / (x - 1) → y ≥ m :=
by
  sorry

end min_value_expr_l139_139617


namespace sequence_formula_min_value_Sn_min_value_Sn_completion_l139_139803

-- Define the sequence sum Sn
def Sn (n : ℕ) : ℤ := n^2 - 48 * n

-- General term of the sequence
def an (n : ℕ) : ℤ :=
  match n with
  | 0     => 0 -- Conventionally, sequences start from 1 in these problems
  | (n+1) => 2 * (n + 1) - 49

-- Prove that the general term of the sequence produces the correct sum
theorem sequence_formula (n : ℕ) (h : 0 < n) : an n = 2 * n - 49 := by
  sorry

-- Prove that the minimum value of Sn is -576 and occurs at n = 24
theorem min_value_Sn : ∃ n : ℕ, Sn n = -576 ∧ ∀ m : ℕ, Sn m ≥ -576 := by
  use 24
  sorry

-- Alternative form of the theorem using the square completion form 
theorem min_value_Sn_completion (n : ℕ) : Sn n = (n - 24)^2 - 576 := by
  sorry

end sequence_formula_min_value_Sn_min_value_Sn_completion_l139_139803


namespace reasoning_is_invalid_l139_139825

-- Definitions based on conditions
variables {Line Plane : Type} (is_parallel_to : Line → Plane → Prop) (is_parallel_to' : Line → Line → Prop) (is_contained_in : Line → Plane → Prop)

-- Conditions
axiom major_premise (b : Line) (α : Plane) : is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a
axiom minor_premise1 (b : Line) (α : Plane) : is_parallel_to b α
axiom minor_premise2 (a : Line) (α : Plane) : is_contained_in a α

-- Conclusion
theorem reasoning_is_invalid : ∃ (a : Line) (b : Line) (α : Plane), ¬ (is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a) :=
sorry

end reasoning_is_invalid_l139_139825


namespace sum_of_digits_base10_representation_l139_139820

def digit_sum (n : ℕ) : ℕ := sorry  -- Define a function to calculate the sum of digits

noncomputable def a : ℕ := 7 * (10 ^ 1234 - 1) / 9
noncomputable def b : ℕ := 2 * (10 ^ 1234 - 1) / 9
noncomputable def product : ℕ := 7 * a * b

theorem sum_of_digits_base10_representation : digit_sum product = 11100 := 
by sorry

end sum_of_digits_base10_representation_l139_139820


namespace number_of_white_tshirts_in_one_pack_l139_139304

namespace TShirts

variable (W : ℕ)

noncomputable def total_white_tshirts := 2 * W
noncomputable def total_blue_tshirts := 4 * 3
noncomputable def cost_per_tshirt := 3
noncomputable def total_cost := 66

theorem number_of_white_tshirts_in_one_pack :
  2 * W * cost_per_tshirt + total_blue_tshirts * cost_per_tshirt = total_cost → W = 5 :=
by
  sorry

end TShirts

end number_of_white_tshirts_in_one_pack_l139_139304


namespace triangle_perimeter_is_720_l139_139072

-- Definitions corresponding to conditions
variables (x : ℕ)
noncomputable def shortest_side := 5 * x
noncomputable def middle_side := 6 * x
noncomputable def longest_side := 7 * x

-- Given the length of the longest side is 280 cm
axiom longest_side_eq : longest_side x = 280

-- Prove that the perimeter of the triangle is 720 cm
theorem triangle_perimeter_is_720 : 
  shortest_side x + middle_side x + longest_side x = 720 :=
by
  sorry

end triangle_perimeter_is_720_l139_139072


namespace sufficient_not_necessary_condition_l139_139024

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≤ 0)) → (a ≥ 5) :=
sorry

end sufficient_not_necessary_condition_l139_139024


namespace friends_prove_l139_139759

theorem friends_prove (a b c d : ℕ) (h1 : 3^a * 7^b = 3^c * 7^d) (h2 : 3^a * 7^b = 21) :
  (a - 1) * (d - 1) = (b - 1) * (c - 1) :=
by {
  sorry
}

end friends_prove_l139_139759


namespace simplified_expression_value_l139_139963

noncomputable def a : ℝ := Real.sqrt 3 + 1
noncomputable def b : ℝ := Real.sqrt 3 - 1

theorem simplified_expression_value :
  ( (a ^ 2 / (a - b) - (2 * a * b - b ^ 2) / (a - b)) / (a - b) * a * b ) = 2 := by
  sorry

end simplified_expression_value_l139_139963


namespace length_of_goods_train_l139_139843

-- Define the given conditions
def speed_kmph := 72
def platform_length := 260
def crossing_time := 26

-- Convert speed to m/s
def speed_mps := (speed_kmph * 5) / 18

-- Calculate distance covered
def distance_covered := speed_mps * crossing_time

-- Define the length of the train
def train_length := distance_covered - platform_length

theorem length_of_goods_train : train_length = 260 := by
  sorry

end length_of_goods_train_l139_139843


namespace equation_satisfying_solution_l139_139942

theorem equation_satisfying_solution (x y : ℤ) :
  (x = 1 ∧ y = 4 → x + 3 * y ≠ 7) ∧
  (x = 2 ∧ y = 1 → x + 3 * y ≠ 7) ∧
  (x = -2 ∧ y = 3 → x + 3 * y = 7) ∧
  (x = 4 ∧ y = 2 → x + 3 * y ≠ 7) :=
by
  sorry

end equation_satisfying_solution_l139_139942


namespace roots_quadratic_l139_139320

theorem roots_quadratic (a b : ℝ) (h₁ : a + b = 6) (h₂ : a * b = 8) :
  a^2 + a^5 * b^3 + a^3 * b^5 + b^2 = 10260 :=
by
  sorry

end roots_quadratic_l139_139320


namespace calculate_f_of_g_l139_139106

def g (x : ℝ) := 4 * x + 6
def f (x : ℝ) := 6 * x - 10

theorem calculate_f_of_g :
  f (g 10) = 266 := by
  sorry

end calculate_f_of_g_l139_139106


namespace maximum_k_l139_139240

theorem maximum_k (m k : ℝ) (h0 : 0 < m) (h1 : m < 1/2) (h2 : (1/m + 2/(1-2*m)) ≥ k): k ≤ 8 :=
sorry

end maximum_k_l139_139240


namespace painted_cubes_on_two_faces_l139_139675

theorem painted_cubes_on_two_faces (n : ℕ) (painted_faces_all : Prop) (equal_smaller_cubes : n = 27) : ∃ k : ℕ, k = 12 :=
by
  -- We only need the statement, not the proof
  sorry

end painted_cubes_on_two_faces_l139_139675


namespace intersection_complement_l139_139690

open Set

-- Defining sets A, B and universal set U
def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {x | 1 < x ∧ x ≤ 6}
def U : Set ℕ := A ∪ B

-- Statement of the proof problem
theorem intersection_complement :
  A ∩ (U \ B) = {1, 7} :=
by
  sorry

end intersection_complement_l139_139690


namespace percentage_increase_in_freelance_l139_139878

open Real

def initial_part_time_earnings := 65
def new_part_time_earnings := 72
def initial_freelance_earnings := 45
def new_freelance_earnings := 72

theorem percentage_increase_in_freelance :
  (new_freelance_earnings - initial_freelance_earnings) / initial_freelance_earnings * 100 = 60 :=
by
  -- Proof will go here
  sorry

end percentage_increase_in_freelance_l139_139878


namespace article_cost_price_l139_139907

theorem article_cost_price :
  ∃ C : ℝ, 
  (1.05 * C) - 2 = (1.045 * C) ∧ 
  ∃ C_new : ℝ, C_new = (0.95 * C) ∧ ((1.045 * C) = (C_new + 0.1 * C_new)) ∧ C = 400 := 
sorry

end article_cost_price_l139_139907


namespace john_avg_speed_l139_139366

/-- John's average speed problem -/
theorem john_avg_speed (d : ℕ) (total_time : ℕ) (time1 : ℕ) (speed1 : ℕ) 
  (time2 : ℕ) (speed2 : ℕ) (time3 : ℕ) (x : ℕ) :
  d = 144 ∧ total_time = 120 ∧ time1 = 40 ∧ speed1 = 64 
  ∧ time2 = 40 ∧ speed2 = 70 ∧ time3 = 40 
  → (d = time1 * speed1 + time2 * speed2 + time3 * x / 60)
  → x = 82 := 
by
  intros h1 h2
  sorry

end john_avg_speed_l139_139366


namespace germination_estimate_l139_139332

theorem germination_estimate (germination_rate : ℝ) (total_pounds : ℝ) 
  (hrate_nonneg : 0 ≤ germination_rate) (hrate_le_one : germination_rate ≤ 1) 
  (h_germination_value : germination_rate = 0.971) 
  (h_total_pounds_value : total_pounds = 1000) : 
  total_pounds * (1 - germination_rate) = 29 := 
by 
  sorry

end germination_estimate_l139_139332


namespace find_quadratic_expression_l139_139402

-- Define the quadratic function
def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

-- Define conditions
def intersects_x_axis_at_A (a b c : ℝ) : Prop :=
  quadratic a b c (-2) = 0

def intersects_x_axis_at_B (a b c : ℝ) : Prop :=
  quadratic a b c (1) = 0

def has_maximum_value (a : ℝ) : Prop :=
  a < 0

-- Define the target function
def f_expr (x : ℝ) : ℝ := -x^2 - x + 2

-- The theorem to be proved
theorem find_quadratic_expression :
  ∃ a b c, 
    intersects_x_axis_at_A a b c ∧
    intersects_x_axis_at_B a b c ∧
    has_maximum_value a ∧
    ∀ x, quadratic a b c x = f_expr x :=
sorry

end find_quadratic_expression_l139_139402


namespace jill_arrives_before_jack_l139_139506

theorem jill_arrives_before_jack
  (distance : ℝ)
  (jill_speed : ℝ)
  (jack_speed : ℝ)
  (jill_time_minutes : ℝ)
  (jack_time_minutes : ℝ) :
  distance = 2 →
  jill_speed = 15 →
  jack_speed = 6 →
  jill_time_minutes = (distance / jill_speed) * 60 →
  jack_time_minutes = (distance / jack_speed) * 60 →
  jack_time_minutes - jill_time_minutes = 12 :=
by
  sorry

end jill_arrives_before_jack_l139_139506


namespace find_num_candies_bought_l139_139440

-- Conditions
def cost_per_candy := 80
def sell_price_per_candy := 100
def num_sold := 48
def profit := 800

-- Question equivalence
theorem find_num_candies_bought (x : ℕ) 
  (hc : cost_per_candy = 80)
  (hs : sell_price_per_candy = 100)
  (hn : num_sold = 48)
  (hp : profit = 800) :
  48 * 100 - 80 * x = 800 → x = 50 :=
  by
  sorry

end find_num_candies_bought_l139_139440


namespace odd_function_zero_unique_l139_139930

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = - f (- x)

def functional_eq (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x + y) * f (x - y) = f x ^ 2 * f y ^ 2

theorem odd_function_zero_unique
  (h_odd : odd_function f)
  (h_func_eq : functional_eq f) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end odd_function_zero_unique_l139_139930


namespace probability_of_convex_quadrilateral_l139_139926

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_convex_quadrilateral :
  let num_points := 8
  let total_chords := binomial num_points 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
by
  -- definitions
  let num_points := 8
  let total_chords := binomial 8 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  
  -- assertion of result
  have h : (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
    sorry
  exact h

end probability_of_convex_quadrilateral_l139_139926


namespace gcd_triangular_number_l139_139389

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem gcd_triangular_number (n : ℕ) (h : n > 2) :
  ∃ k, n = 12 * k + 2 → gcd (6 * triangular_number n) (n - 2) = 12 :=
  sorry

end gcd_triangular_number_l139_139389


namespace jeff_cat_shelter_l139_139572

theorem jeff_cat_shelter :
  let initial_cats := 20
  let monday_cats := 2
  let tuesday_cats := 1
  let people_adopted := 3
  let cats_per_person := 2
  let total_cats := initial_cats + monday_cats + tuesday_cats
  let adopted_cats := people_adopted * cats_per_person
  total_cats - adopted_cats = 17 := 
by
  sorry

end jeff_cat_shelter_l139_139572


namespace solve_real_equation_l139_139018

theorem solve_real_equation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -3) :
  (x ^ 3 + 3 * x ^ 2 - x) / (x ^ 2 + 4 * x + 3) + x = -7 ↔ x = -5 / 2 ∨ x = -4 := 
by
  sorry

end solve_real_equation_l139_139018


namespace b11_eq_4_l139_139718

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d r : ℤ} {a1 : ℤ}

-- Define non-zero arithmetic sequence {a_n} with common difference d
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define geometric sequence {b_n} with common ratio r
def is_geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ n, b (n + 1) = b n * r

-- The given conditions
axiom a1_minus_a7_sq_plus_a13_eq_zero : a 1 - (a 7) ^ 2 + a 13 = 0
axiom b7_eq_a7 : b 7 = a 7

-- The problem statement to prove: b 11 = 4
theorem b11_eq_4
  (arith_seq : is_arithmetic_sequence a d)
  (geom_seq : is_geometric_sequence b r)
  (a1_non_zero : a1 ≠ 0) :
  b 11 = 4 :=
sorry

end b11_eq_4_l139_139718


namespace units_digit_p_l139_139305

theorem units_digit_p (p : ℕ) (h1 : p % 2 = 0) (h2 : ((p ^ 3 % 10) - (p ^ 2 % 10)) % 10 = 0) 
(h3 : (p + 4) % 10 = 0) : p % 10 = 6 :=
sorry

end units_digit_p_l139_139305


namespace possible_third_side_l139_139608

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l139_139608


namespace average_honey_per_bee_per_day_l139_139680

-- Definitions based on conditions
def num_honey_bees : ℕ := 50
def honey_bee_days : ℕ := 35
def total_honey_produced : ℕ := 75
def expected_avg_honey_per_bee_per_day : ℝ := 2.14

-- Statement of the proof problem
theorem average_honey_per_bee_per_day :
  ((total_honey_produced : ℝ) / (num_honey_bees * honey_bee_days)) = expected_avg_honey_per_bee_per_day := by
  sorry

end average_honey_per_bee_per_day_l139_139680


namespace total_lives_l139_139085

-- Definitions of given conditions
def original_friends : Nat := 2
def lives_per_player : Nat := 6
def additional_players : Nat := 2

-- Proof statement to show the total number of lives
theorem total_lives :
  (original_friends * lives_per_player) + (additional_players * lives_per_player) = 24 := by
  sorry

end total_lives_l139_139085


namespace find_range_of_m_l139_139585

noncomputable def quadratic_equation := 
  ∀ (m : ℝ), 
  ∃ x y : ℝ, 
  (m + 3) * x^2 - 4 * m * x + (2 * m - 1) = 0 ∧ 
  (m + 3) * y^2 - 4 * m * y + (2 * m - 1) = 0 ∧ 
  x * y < 0 ∧ 
  |x| > |y| ∧ 
  m ∈ Set.Ioo (-3:ℝ) (0:ℝ)

theorem find_range_of_m : quadratic_equation := 
by
  sorry

end find_range_of_m_l139_139585


namespace sum_coordinates_of_k_l139_139147

theorem sum_coordinates_of_k :
  ∀ (f k : ℕ → ℕ), (f 4 = 8) → (∀ x, k x = (f x) ^ 3) → (4 + k 4) = 516 :=
by
  intros f k h1 h2
  sorry

end sum_coordinates_of_k_l139_139147


namespace luncheon_cost_l139_139985

theorem luncheon_cost
  (s c p : ℝ)
  (h1 : 3 * s + 7 * c + p = 3.15)
  (h2 : 4 * s + 10 * c + p = 4.20) :
  s + c + p = 1.05 :=
by sorry

end luncheon_cost_l139_139985


namespace scenery_photos_correct_l139_139482

-- Define the problem conditions
def animal_photos := 10
def flower_photos := 3 * animal_photos
def photos_total := 45
def scenery_photos := flower_photos - 10

-- State the theorem
theorem scenery_photos_correct : scenery_photos = 20 ∧ animal_photos + flower_photos + scenery_photos = photos_total := by
  sorry

end scenery_photos_correct_l139_139482


namespace quadratic_inequality_solution_set_l139_139658

theorem quadratic_inequality_solution_set (p q : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) →
  p = 5 ∧ q = -6 ∧
  (∀ x : ℝ, - (1 : ℝ) / 2 < x ∧ x < - (1 : ℝ) / 3 → 6 * x^2 + 5 * x + 1 < 0) :=
by
  sorry

end quadratic_inequality_solution_set_l139_139658


namespace work_completion_by_C_l139_139306

theorem work_completion_by_C
  (A_work_rate : ℝ)
  (B_work_rate : ℝ)
  (C_work_rate : ℝ)
  (A_days_worked : ℝ)
  (B_days_worked : ℝ)
  (C_days_worked : ℝ)
  (A_total_days : ℝ)
  (B_total_days : ℝ)
  (C_completion_partial_work : ℝ)
  (H1 : A_work_rate = 1 / 40)
  (H2 : B_work_rate = 1 / 40)
  (H3 : A_days_worked = 10)
  (H4 : B_days_worked = 10)
  (H5 : C_days_worked = 10)
  (H6 : C_completion_partial_work = 1/2) :
  C_work_rate = 1 / 20 :=
by
  sorry

end work_completion_by_C_l139_139306


namespace octadecagon_identity_l139_139117

theorem octadecagon_identity (a r : ℝ) (h : a = 2 * r * Real.sin (π / 18)) :
  a^3 + r^3 = 3 * r^2 * a :=
sorry

end octadecagon_identity_l139_139117


namespace tesla_ratio_l139_139017

variables (s c e : ℕ)
variables (h1 : e = s + 10) (h2 : c = 6) (h3 : e = 13)

theorem tesla_ratio : s / c = 1 / 2 :=
by
  sorry

end tesla_ratio_l139_139017


namespace drew_got_wrong_19_l139_139672

theorem drew_got_wrong_19 :
  ∃ (D_wrong C_wrong : ℕ), 
    (20 + D_wrong = 52) ∧
    (14 + C_wrong = 52) ∧
    (C_wrong = 2 * D_wrong) ∧
    D_wrong = 19 :=
by
  sorry

end drew_got_wrong_19_l139_139672


namespace angle_BMC_not_obtuse_angle_BAC_is_120_l139_139232

theorem angle_BMC_not_obtuse (α β γ : ℝ) (h : α + β + γ = 180) :
  0 < 90 - α / 2 ∧ 90 - α / 2 < 90 :=
sorry

theorem angle_BAC_is_120 (α β γ : ℝ) (h1 : α + β + γ = 180)
  (h2 : 90 - α / 2 = α / 2) : α = 120 :=
sorry

end angle_BMC_not_obtuse_angle_BAC_is_120_l139_139232


namespace paper_folding_holes_l139_139309

def folded_paper_holes (folds: Nat) (holes: Nat) : Nat :=
  match folds with
  | 0 => holes
  | n+1 => 2 * folded_paper_holes n holes

theorem paper_folding_holes : folded_paper_holes 3 1 = 8 :=
by
  -- sorry to skip the proof
  sorry

end paper_folding_holes_l139_139309


namespace max_abs_sum_on_ellipse_l139_139467

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), (x^2 / 4) + (y^2 / 9) = 1 → |x| + |y| ≤ 5 :=
by sorry

end max_abs_sum_on_ellipse_l139_139467


namespace correct_cd_value_l139_139595

noncomputable def repeating_decimal (c d : ℕ) : ℝ :=
  1 + c / 10.0 + d / 100.0 + (c * 10 + d) / 990.0

theorem correct_cd_value (c d : ℕ) (h : (c = 9) ∧ (d = 9)) : 90 * (repeating_decimal 9 9 - (1 + 9 / 10.0 + 9 / 100.0)) = 0.9 :=
by
  sorry

end correct_cd_value_l139_139595


namespace A_plus_2B_plus_4_is_perfect_square_l139_139496

theorem A_plus_2B_plus_4_is_perfect_square (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  ∃ k : ℚ, (A + 2 * B + 4) = k^2 :=
by
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  use ((2/3) * (10^n + 2))
  sorry

end A_plus_2B_plus_4_is_perfect_square_l139_139496


namespace sum_of_edges_proof_l139_139002

noncomputable def sum_of_edges (a r : ℝ) : ℝ :=
  let l1 := a / r
  let l2 := a
  let l3 := a * r
  4 * (l1 + l2 + l3)

theorem sum_of_edges_proof : 
  ∀ (a r : ℝ), 
  (a > 0 ∧ r > 0 ∧ (a / r) * a * (a * r) = 512 ∧ 2 * ((a^2 / r) + a^2 + a^2 * r) = 384) → sum_of_edges a r = 96 :=
by
  intros a r h
  -- We skip the proof here with sorry
  sorry

end sum_of_edges_proof_l139_139002


namespace paint_houses_l139_139406

theorem paint_houses (time_per_house : ℕ) (hour_to_minute : ℕ) (hours_available : ℕ) 
  (h1 : time_per_house = 20) (h2 : hour_to_minute = 60) (h3 : hours_available = 3) :
  (hours_available * hour_to_minute) / time_per_house = 9 :=
by
  sorry

end paint_houses_l139_139406


namespace coffee_decaf_percentage_l139_139734

variable (initial_stock : ℝ) (initial_decaf_percent : ℝ)
variable (new_stock : ℝ) (new_decaf_percent : ℝ)

noncomputable def decaf_coffee_percentage : ℝ :=
  let initial_decaf : ℝ := initial_stock * (initial_decaf_percent / 100)
  let new_decaf : ℝ := new_stock * (new_decaf_percent / 100)
  let total_decaf : ℝ := initial_decaf + new_decaf
  let total_stock : ℝ := initial_stock + new_stock
  (total_decaf / total_stock) * 100

theorem coffee_decaf_percentage :
  initial_stock = 400 →
  initial_decaf_percent = 20 →
  new_stock = 100 →
  new_decaf_percent = 50 →
  decaf_coffee_percentage initial_stock initial_decaf_percent new_stock new_decaf_percent = 26 :=
by
  intros
  sorry

end coffee_decaf_percentage_l139_139734


namespace functional_equation_initial_condition_unique_f3_l139_139979

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (x y : ℝ) : f (f x + y) = f (x ^ 2 - y) + 2 * f x * y := sorry

theorem initial_condition : f 1 = 1 := sorry

theorem unique_f3 : f 3 = 9 := sorry

end functional_equation_initial_condition_unique_f3_l139_139979


namespace time_spent_per_bone_l139_139511

theorem time_spent_per_bone
  (total_hours : ℤ) (number_of_bones : ℤ) 
  (h1 : total_hours = 206) 
  (h2 : number_of_bones = 206) :
  (total_hours / number_of_bones = 1) := 
by {
  -- proof would go here
  sorry
}

end time_spent_per_bone_l139_139511


namespace speed_with_stream_l139_139749

-- Define the given conditions
def V_m : ℝ := 7 -- Man's speed in still water (7 km/h)
def V_as : ℝ := 10 -- Man's speed against the stream (10 km/h)

-- Define the stream's speed as the difference
def V_s : ℝ := V_m - V_as

-- Define man's speed with the stream
def V_ws : ℝ := V_m + V_s

-- (Correct Answer): Prove the man's speed with the stream is 10 km/h
theorem speed_with_stream :
  V_ws = 10 := by
  -- Sorry for no proof required in this task
  sorry

end speed_with_stream_l139_139749


namespace calc_remainder_l139_139337

theorem calc_remainder : 
  (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 +
   90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 -
   90^7 * Nat.choose 10 7 + 90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 +
   90^10 * Nat.choose 10 10) % 88 = 1 := 
by sorry

end calc_remainder_l139_139337


namespace total_toys_l139_139730

theorem total_toys (n : ℕ) (h1 : 3 * (n / 4) = 18) : n = 24 :=
by
  sorry

end total_toys_l139_139730


namespace desired_gain_percentage_l139_139488

theorem desired_gain_percentage (cp16 sp16 cp12881355932203391 sp12881355932203391 : ℝ) :
  sp16 = 1 →
  sp16 = 0.95 * cp16 →
  sp12881355932203391 = 1 →
  cp12881355932203391 = (12.881355932203391 / 16) * cp16 →
  (sp12881355932203391 - cp12881355932203391) / cp12881355932203391 * 100 = 18.75 :=
by sorry

end desired_gain_percentage_l139_139488


namespace range_of_x_for_valid_sqrt_l139_139126

theorem range_of_x_for_valid_sqrt (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
by
  sorry

end range_of_x_for_valid_sqrt_l139_139126


namespace limit_of_nested_radical_l139_139454

theorem limit_of_nested_radical :
  ∃ F : ℝ, F = 43 ∧ F = Real.sqrt (86 + 41 * F) :=
sorry

end limit_of_nested_radical_l139_139454


namespace smallest_perimeter_even_integer_triangl_l139_139382

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_l139_139382


namespace trader_sold_40_meters_l139_139539

noncomputable def meters_of_cloth_sold (profit_per_meter total_profit : ℕ) : ℕ :=
  total_profit / profit_per_meter

theorem trader_sold_40_meters (profit_per_meter total_profit : ℕ) (h1 : profit_per_meter = 35) (h2 : total_profit = 1400) :
  meters_of_cloth_sold profit_per_meter total_profit = 40 :=
by
  sorry

end trader_sold_40_meters_l139_139539


namespace joan_picked_apples_l139_139283

theorem joan_picked_apples (a b c : ℕ) (h1 : b = 27) (h2 : c = 70) (h3 : c = a + b) : a = 43 :=
by
  sorry

end joan_picked_apples_l139_139283


namespace Jenine_pencil_count_l139_139769

theorem Jenine_pencil_count
  (sharpenings_per_pencil : ℕ)
  (hours_per_sharpening : ℝ)
  (total_hours_needed : ℝ)
  (cost_per_pencil : ℝ)
  (budget : ℝ)
  (already_has_pencils : ℕ) :
  sharpenings_per_pencil = 5 →
  hours_per_sharpening = 1.5 →
  total_hours_needed = 105 →
  cost_per_pencil = 2 →
  budget = 8 →
  already_has_pencils = 10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Jenine_pencil_count_l139_139769


namespace product_as_difference_of_squares_l139_139788

theorem product_as_difference_of_squares (a b : ℝ) : 
  a * b = ( (a + b) / 2 )^2 - ( (a - b) / 2 )^2 :=
by
  sorry

end product_as_difference_of_squares_l139_139788


namespace no_eight_roots_for_nested_quadratics_l139_139226

theorem no_eight_roots_for_nested_quadratics
  (f g h : ℝ → ℝ)
  (hf : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e k : ℝ, ∀ x, g x = d * x^2 + e * x + k)
  (hh : ∃ p q r : ℝ, ∀ x, h x = p * x^2 + q * x + r)
  (hroots : ∀ x, f (g (h x)) = 0 → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8)) :
  false :=
by
  sorry

end no_eight_roots_for_nested_quadratics_l139_139226


namespace deployment_plans_l139_139260

/-- Given 6 volunteers and needing to select 4 to fill different positions of 
  translator, tour guide, shopping guide, and cleaner, and knowing that neither 
  supporters A nor B can work as the translator, the total number of deployment plans is 240. -/
theorem deployment_plans (volunteers : Fin 6) (A B : Fin 6) : 
  ∀ {translator tour_guide shopping_guide cleaner : Fin 6},
  A ≠ translator ∧ B ≠ translator → 
  ∃ plans : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6), plans.card = 240 :=
by 
sorry

end deployment_plans_l139_139260


namespace sin_45_degree_l139_139074

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l139_139074


namespace ways_to_score_at_least_7_points_l139_139937

-- Definitions based on the given conditions
def red_balls : Nat := 4
def white_balls : Nat := 6
def points_red : Nat := 2
def points_white : Nat := 1

-- Function to count the number of combinations for choosing k elements from n elements
def choose (n : Nat) (k : Nat) : Nat :=
  if h : k ≤ n then
    Nat.descFactorial n k / Nat.factorial k
  else
    0

-- The main theorem to prove the number of ways to get at least 7 points by choosing 5 balls out
theorem ways_to_score_at_least_7_points : 
  (choose red_balls 4 * choose white_balls 1) +
  (choose red_balls 3 * choose white_balls 2) +
  (choose red_balls 2 * choose white_balls 3) = 186 := 
sorry

end ways_to_score_at_least_7_points_l139_139937


namespace expression_A_expression_B_expression_C_expression_D_l139_139629

theorem expression_A :
  (Real.sin (7 * Real.pi / 180) * Real.cos (23 * Real.pi / 180) + 
   Real.sin (83 * Real.pi / 180) * Real.cos (67 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_B :
  (2 * Real.cos (75 * Real.pi / 180) * Real.sin (75 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_C :
  (Real.sqrt 3 * Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) / 
   Real.sin (50 * Real.pi / 180) ≠ 1 / 2 :=
sorry

theorem expression_D :
  (1 / ((1 + Real.tan (27 * Real.pi / 180)) * (1 + Real.tan (18 * Real.pi / 180)))) = 1 / 2 :=
sorry

end expression_A_expression_B_expression_C_expression_D_l139_139629


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l139_139250

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l139_139250


namespace inequality_for_a_l139_139590

noncomputable def f (x : ℝ) : ℝ :=
  2^x + (Real.log x) / (Real.log 2)

theorem inequality_for_a (n : ℕ) (a : ℝ) (h₁ : 2 < n) (h₂ : 0 < a) (h₃ : 2^a + Real.log a / Real.log 2 = n^2) :
  2 * Real.log n / Real.log 2 > a ∧ a > 2 * Real.log n / Real.log 2 - 1 / n :=
by
  sorry

end inequality_for_a_l139_139590


namespace unique_solution_for_a_eq_1_l139_139302

def equation (a x : ℝ) : Prop :=
  5^(x^2 - 6 * a * x + 9 * a^2) = a * x^2 - 6 * a^2 * x + 9 * a^3 + a^2 - 6 * a + 6

theorem unique_solution_for_a_eq_1 :
  (∃! x : ℝ, equation 1 x) ∧ 
  (∀ a : ℝ, (∃! x : ℝ, equation a x) → a = 1) :=
sorry

end unique_solution_for_a_eq_1_l139_139302


namespace condition_for_ellipse_l139_139673

-- Definition of the problem conditions
def is_ellipse (m : ℝ) : Prop :=
  (m - 2 > 0) ∧ (5 - m > 0) ∧ (m - 2 ≠ 5 - m)

noncomputable def necessary_not_sufficient_condition (m : ℝ) : Prop :=
  (2 < m) ∧ (m < 5)

-- The theorem to be proved
theorem condition_for_ellipse (m : ℝ) : 
  (necessary_not_sufficient_condition m) → (is_ellipse m) :=
by
  -- proof to be written here
  sorry

end condition_for_ellipse_l139_139673


namespace total_time_per_week_l139_139774

noncomputable def meditating_time_per_day : ℝ := 1
noncomputable def reading_time_per_day : ℝ := 2 * meditating_time_per_day
noncomputable def exercising_time_per_day : ℝ := 0.5 * meditating_time_per_day
noncomputable def practicing_time_per_day : ℝ := (1/3) * reading_time_per_day

noncomputable def total_time_per_day : ℝ :=
  meditating_time_per_day + reading_time_per_day + exercising_time_per_day + practicing_time_per_day

theorem total_time_per_week :
  total_time_per_day * 7 = 29.17 := by
  sorry

end total_time_per_week_l139_139774


namespace reading_hours_l139_139369

theorem reading_hours (h : ℕ) (lizaRate suzieRate : ℕ) (lizaPages suziePages : ℕ) 
  (hliza : lizaRate = 20) (hsuzie : suzieRate = 15) 
  (hlizaPages : lizaPages = lizaRate * h) (hsuziePages : suziePages = suzieRate * h) 
  (h_diff : lizaPages = suziePages + 15) : h = 3 :=
by {
  sorry
}

end reading_hours_l139_139369


namespace correlation_comparison_l139_139741

-- Definitions of the datasets
def data_XY : List (ℝ × ℝ) := [(10,1), (11.3,2), (11.8,3), (12.5,4), (13,5)]
def data_UV : List (ℝ × ℝ) := [(10,5), (11.3,4), (11.8,3), (12.5,2), (13,1)]

-- Definitions of the linear correlation coefficients
noncomputable def r1 : ℝ := sorry -- Calculation of correlation coefficient between X and Y
noncomputable def r2 : ℝ := sorry -- Calculation of correlation coefficient between U and V

-- The proof statement
theorem correlation_comparison :
  r2 < 0 ∧ 0 < r1 :=
sorry

end correlation_comparison_l139_139741


namespace multiplication_of_powers_l139_139589

theorem multiplication_of_powers :
  2^4 * 3^2 * 5^2 * 11 = 39600 := by
  sorry

end multiplication_of_powers_l139_139589


namespace running_speed_l139_139558

theorem running_speed (walk_speed total_distance walk_time total_time run_distance : ℝ) 
  (h_walk_speed : walk_speed = 4)
  (h_total_distance : total_distance = 4)
  (h_walk_time : walk_time = 0.5)
  (h_total_time : total_time = 0.75)
  (h_run_distance : run_distance = total_distance / 2) :
  (2 / ((total_time - walk_time) - 2 / walk_speed)) = 8 := 
by
  -- To be proven
  sorry

end running_speed_l139_139558


namespace prove_q_ge_bd_and_p_eq_ac_l139_139588

-- Definitions for the problem
variables (a b c d p q : ℕ)

-- Conditions given in the problem
axiom h1: a * d - b * c = 1
axiom h2: (a : ℚ) / b > (p : ℚ) / q
axiom h3: (p : ℚ) / q > (c : ℚ) / d

-- The theorem to be proved
theorem prove_q_ge_bd_and_p_eq_ac (a b c d p q : ℕ) (h1 : a * d - b * c = 1) 
  (h2 : (a : ℚ) / b > (p : ℚ) / q) (h3 : (p : ℚ) / q > (c : ℚ) / d) :
  q ≥ b + d ∧ (q = b + d → p = a + c) :=
by
  sorry

end prove_q_ge_bd_and_p_eq_ac_l139_139588


namespace profit_A_after_upgrade_profit_B_constrained_l139_139541

-- Part Ⅰ
theorem profit_A_after_upgrade (x : ℝ) (h : x^2 - 300 * x ≤ 0) : 0 < x ∧ x ≤ 300 := sorry

-- Part Ⅱ
theorem profit_B_constrained (a x : ℝ) (h1 : a ≤ (x/125 + 500/x + 3/2)) (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := sorry

end profit_A_after_upgrade_profit_B_constrained_l139_139541


namespace parallel_vectors_l139_139153

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 1)) (h₂ : b = (1, m))
  (h₃ : ∃ k : ℝ, b = k • a) : m = 1 / 2 :=
by 
  sorry

end parallel_vectors_l139_139153


namespace frac_inequality_l139_139520

theorem frac_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : d > c) (h4 : c > 0) : (a/c) > (b/d) := 
sorry

end frac_inequality_l139_139520


namespace slices_left_for_lunch_tomorrow_l139_139806

def pizza_slices : ℕ := 12
def lunch_slices : ℕ := pizza_slices / 2
def remaining_after_lunch : ℕ := pizza_slices - lunch_slices
def dinner_slices : ℕ := remaining_after_lunch * 1/3
def slices_left : ℕ := remaining_after_lunch - dinner_slices

theorem slices_left_for_lunch_tomorrow : slices_left = 4 :=
by
  sorry

end slices_left_for_lunch_tomorrow_l139_139806


namespace isolating_and_counting_bacteria_process_l139_139274

theorem isolating_and_counting_bacteria_process
  (soil_sampling : Prop)
  (spreading_dilution_on_culture_medium : Prop)
  (decompose_urea : Prop) :
  (soil_sampling ∧ spreading_dilution_on_culture_medium ∧ decompose_urea) →
  (Sample_dilution ∧ Selecting_colonies_that_can_grow ∧ Identification) :=
sorry

end isolating_and_counting_bacteria_process_l139_139274


namespace profit_percentage_is_correct_l139_139127

noncomputable def CP : ℝ := 47.50
noncomputable def SP : ℝ := 74.21875
noncomputable def MP : ℝ := SP / 0.8
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercentage : ℝ := (Profit / CP) * 100

theorem profit_percentage_is_correct : ProfitPercentage = 56.25 := by
  -- Proof steps to be filled in
  sorry

end profit_percentage_is_correct_l139_139127


namespace smallest_prime_fifth_term_of_arithmetic_sequence_l139_139166

theorem smallest_prime_fifth_term_of_arithmetic_sequence :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), 
    (∀ n, seq n = a + n * d) ∧ 
    (∀ n < 5, Prime (seq n)) ∧ 
    d = 6 ∧ 
    a = 5 ∧ 
    seq 4 = 29 := by
  sorry

end smallest_prime_fifth_term_of_arithmetic_sequence_l139_139166


namespace solution_exists_l139_139796

-- Defining the variables x and y
variables (x y : ℝ)

-- Defining the conditions
def condition_1 : Prop :=
  3 * x ≥ 2 * y + 16

def condition_2 : Prop :=
  x^4 + 2 * (x^2) * (y^2) + y^4 + 25 - 26 * (x^2) - 26 * (y^2) = 72 * x * y

-- Stating the theorem that (6, 1) satisfies the conditions
theorem solution_exists : condition_1 6 1 ∧ condition_2 6 1 :=
by
  -- Convert conditions into expressions
  have h1 : condition_1 6 1 := by sorry
  have h2 : condition_2 6 1 := by sorry
  -- Conjunction of both conditions is satisfied
  exact ⟨h1, h2⟩

end solution_exists_l139_139796


namespace vec_op_l139_139855

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (2, -2)
def two_a : ℝ × ℝ := (2 * 2, 2 * 1)
def result : ℝ × ℝ := (two_a.1 - b.1, two_a.2 - b.2)

theorem vec_op : (2 * a.1 - b.1, 2 * a.2 - b.2) = (2, 4) := by
  sorry

end vec_op_l139_139855


namespace tree_age_when_23_feet_l139_139175

theorem tree_age_when_23_feet (initial_age initial_height growth_rate final_height : ℕ) 
(h_initial_age : initial_age = 1)
(h_initial_height : initial_height = 5) 
(h_growth_rate : growth_rate = 3) 
(h_final_height : final_height = 23) : 
initial_age + (final_height - initial_height) / growth_rate = 7 := 
by sorry

end tree_age_when_23_feet_l139_139175


namespace min_value_x_y_xy_l139_139549

theorem min_value_x_y_xy (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  x + y + x * y ≥ -9 / 8 :=
sorry

end min_value_x_y_xy_l139_139549


namespace evaluate_sum_l139_139409

theorem evaluate_sum (a b c : ℝ) 
  (h : (a / (36 - a) + b / (49 - b) + c / (81 - c) = 9)) :
  (6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 5.047) :=
by
  sorry

end evaluate_sum_l139_139409


namespace Petya_can_determine_weight_l139_139358

theorem Petya_can_determine_weight (n : ℕ) (distinct_weights : Fin n → ℕ) 
  (device : (Fin 10 → Fin n) → ℕ) (ten_thousand_weights : n = 10000)
  (no_two_same : (∀ i j : Fin n, i ≠ j → distinct_weights i ≠ distinct_weights j)) :
  ∃ i : Fin n, ∃ w : ℕ, distinct_weights i = w :=
by
  sorry

end Petya_can_determine_weight_l139_139358


namespace middle_integer_is_five_l139_139961

-- Define the conditions of the problem
def consecutive_one_digit_positive_odd_integers (a b c : ℤ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧
  a + 2 = b ∧ b + 2 = c ∨ a + 2 = c ∧ c + 2 = b

def sum_is_one_seventh_of_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 7

-- Define the theorem to prove
theorem middle_integer_is_five :
  ∃ (b : ℤ), consecutive_one_digit_positive_odd_integers (b - 2) b (b + 2) ∧
             sum_is_one_seventh_of_product (b - 2) b (b + 2) ∧
             b = 5 :=
sorry

end middle_integer_is_five_l139_139961


namespace milk_needed_for_cookies_l139_139385

-- Define the given conditions
def liters_to_cups (liters : ℕ) : ℕ := liters * 4

def milk_per_cookies (cups cookies : ℕ) : ℚ := cups / cookies

-- Define the problem statement
theorem milk_needed_for_cookies (h1 : milk_per_cookies 20 30 = milk_per_cookies x 12) : x = 8 :=
sorry

end milk_needed_for_cookies_l139_139385


namespace convert_base8_to_base7_l139_139499

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * 8^2 + 3 * 8^1 + 1 * 8^0

def base10_to_base7 (n : ℕ) : ℕ :=
  1002  -- Directly providing the result from conditions given.

theorem convert_base8_to_base7 :
  base10_to_base7 (base8_to_base10 531) = 1002 := by
  sorry

end convert_base8_to_base7_l139_139499


namespace average_sale_l139_139396

-- Defining the monthly sales as constants
def sale_month1 : ℝ := 6435
def sale_month2 : ℝ := 6927
def sale_month3 : ℝ := 6855
def sale_month4 : ℝ := 7230
def sale_month5 : ℝ := 6562
def sale_month6 : ℝ := 7391

-- The final theorem statement to prove the average sale
theorem average_sale : (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6900 := 
by 
  sorry

end average_sale_l139_139396


namespace find_lower_rate_l139_139494

-- Definitions
def total_investment : ℝ := 20000
def total_interest : ℝ := 1440
def higher_rate : ℝ := 0.09
def fraction_higher : ℝ := 0.55

-- The amount invested at the higher rate
def x := fraction_higher * total_investment
-- The amount invested at the lower rate
def y := total_investment - x

-- The interest contributions
def interest_higher := x * higher_rate
def interest_lower (r : ℝ) := y * r

-- The equation we need to solve to find the lower interest rate
theorem find_lower_rate (r : ℝ) : interest_higher + interest_lower r = total_interest → r = 0.05 :=
by
  sorry

end find_lower_rate_l139_139494


namespace third_even_number_sequence_l139_139562

theorem third_even_number_sequence (x : ℕ) (h_even : x % 2 = 0) (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) = 180) : x + 4 = 30 :=
by
  sorry

end third_even_number_sequence_l139_139562


namespace Charlie_age_when_Jenny_twice_as_Bobby_l139_139222

theorem Charlie_age_when_Jenny_twice_as_Bobby (B C J : ℕ) 
  (h₁ : J = C + 5)
  (h₂ : C = B + 3)
  (h₃ : J = 2 * B) : 
  C = 11 :=
by
  sorry

end Charlie_age_when_Jenny_twice_as_Bobby_l139_139222


namespace unknown_number_value_l139_139254

theorem unknown_number_value (a x : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * x * 35 * 63) : x = 25 := by
  sorry

end unknown_number_value_l139_139254


namespace total_kids_at_camp_l139_139089

-- Definition of the conditions
def kids_from_lawrence_camp : ℕ := 34044
def kids_from_outside_camp : ℕ := 424944

-- The proof statement
theorem total_kids_at_camp : kids_from_lawrence_camp + kids_from_outside_camp = 459988 := by
  sorry

end total_kids_at_camp_l139_139089


namespace num_of_ordered_pairs_l139_139866

theorem num_of_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b > a)
(h4 : (a-2)*(b-2) = (ab / 2)) : (a, b) = (5, 12) ∨ (a, b) = (6, 8) :=
by
  sorry

end num_of_ordered_pairs_l139_139866


namespace find_x_l139_139661

variables {x : ℝ}
def vector_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x
  (h1 : (6, 1) = (6, 1))
  (h2 : (x, -3) = (x, -3))
  (h3 : vector_parallel (6, 1) (x, -3)) :
  x = -18 := by
  sorry

end find_x_l139_139661


namespace isabella_hourly_rate_l139_139530

def isabella_hours_per_day : ℕ := 5
def isabella_days_per_week : ℕ := 6
def isabella_weeks : ℕ := 7
def isabella_total_earnings : ℕ := 1050

theorem isabella_hourly_rate :
  (isabella_hours_per_day * isabella_days_per_week * isabella_weeks) * x = isabella_total_earnings → x = 5 := by
  sorry

end isabella_hourly_rate_l139_139530


namespace latus_rectum_of_parabola_l139_139763

theorem latus_rectum_of_parabola :
  (∃ p : ℝ, ∀ x y : ℝ, y = - (1 / 6) * x^2 → y = p ∧ p = 3 / 2) :=
sorry

end latus_rectum_of_parabola_l139_139763


namespace transformed_parabola_l139_139339

theorem transformed_parabola (x : ℝ) : 
  (λ x => -x^2 + 1) (x - 2) - 2 = - (x - 2)^2 - 1 := 
by 
  sorry 

end transformed_parabola_l139_139339


namespace foci_and_directrices_of_ellipse_l139_139204

noncomputable def parametricEllipse
    (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ + 1, 4 * Real.sin θ)

theorem foci_and_directrices_of_ellipse :
  (∀ θ : ℝ, parametricEllipse θ = (x, y)) →
  (∃ (f1 f2 : ℝ × ℝ) (d1 d2 : ℝ → Prop),
    f1 = (1, Real.sqrt 7) ∧
    f2 = (1, -Real.sqrt 7) ∧
    d1 = fun x => x = 1 + 9 / Real.sqrt 7 ∧
    d2 = fun x => x = 1 - 9 / Real.sqrt 7) := sorry

end foci_and_directrices_of_ellipse_l139_139204


namespace ratio_of_volumes_total_surface_area_smaller_cube_l139_139776

-- Definitions using the conditions in (a)
def edge_length_smaller_cube := 4 -- in inches
def edge_length_larger_cube := 24 -- in inches (2 feet converted to inches)

-- Propositions based on the correct answers in (b)
theorem ratio_of_volumes : 
  (edge_length_smaller_cube ^ 3) / (edge_length_larger_cube ^ 3) = 1 / 216 := by
  sorry

theorem total_surface_area_smaller_cube : 
  6 * (edge_length_smaller_cube ^ 2) = 96 := by
  sorry

end ratio_of_volumes_total_surface_area_smaller_cube_l139_139776


namespace ludvik_favorite_number_l139_139235

variable (a b : ℕ)
variable (ℓ : ℝ)

theorem ludvik_favorite_number (h1 : 2 * a = (b + 12) * ℓ)
(h2 : a - 42 = (b / 2) * ℓ) : ℓ = 7 :=
sorry

end ludvik_favorite_number_l139_139235


namespace sum_of_reciprocals_of_squares_l139_139179

open BigOperators

theorem sum_of_reciprocals_of_squares (n : ℕ) (h : n ≥ 2) :
   (∑ k in Finset.range n, 1 / (k + 1)^2) < (2 * n - 1) / n :=
sorry

end sum_of_reciprocals_of_squares_l139_139179


namespace find_D_c_l139_139635

-- Define the given conditions
def daily_wage_ratio (W_a W_b W_c : ℝ) : Prop :=
  W_a / W_b = 3 / 4 ∧ W_a / W_c = 3 / 5 ∧ W_b / W_c = 4 / 5

def total_earnings (W_a W_b W_c : ℝ) (D_a D_b D_c : ℕ) : ℝ :=
  W_a * D_a + W_b * D_b + W_c * D_c

variables {W_a W_b W_c : ℝ} 
variables {D_a D_b D_c : ℕ} 

-- Given values according to the problem
def W_c_value : ℝ := 110
def D_a_value : ℕ := 6
def D_b_value : ℕ := 9
def total_earnings_value : ℝ := 1628

-- The target proof statement
theorem find_D_c 
  (h_ratio : daily_wage_ratio W_a W_b W_c)
  (h_Wc : W_c = W_c_value)
  (h_earnings : total_earnings W_a W_b W_c D_a_value D_b_value D_c = total_earnings_value) 
  : D_c = 4 := 
sorry

end find_D_c_l139_139635


namespace odd_nat_composite_iff_exists_a_l139_139128

theorem odd_nat_composite_iff_exists_a (c : ℕ) (h_odd : c % 2 = 1) :
  (∃ a : ℕ, a ≤ c / 3 - 1 ∧ ∃ k : ℕ, (2*a - 1)^2 + 8*c = k^2) ↔
  ∃ d : ℕ, ∃ e : ℕ, d > 1 ∧ e > 1 ∧ d * e = c := 
sorry

end odd_nat_composite_iff_exists_a_l139_139128


namespace dakotas_medical_bill_l139_139336

theorem dakotas_medical_bill :
  let days_in_hospital := 3
  let hospital_bed_cost_per_day := 900
  let specialists_rate_per_hour := 250
  let specialist_minutes_per_day := 15
  let num_specialists := 2
  let ambulance_cost := 1800

  let hospital_bed_cost := hospital_bed_cost_per_day * days_in_hospital
  let specialists_total_minutes := specialist_minutes_per_day * num_specialists
  let specialists_hours := specialists_total_minutes / 60.0
  let specialists_cost := specialists_hours * specialists_rate_per_hour

  let total_medical_bill := hospital_bed_cost + specialists_cost + ambulance_cost

  total_medical_bill = 4625 := 
by
  sorry

end dakotas_medical_bill_l139_139336


namespace inequality_for_pos_reals_l139_139252

open Real Nat

theorem inequality_for_pos_reals
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 1/a + 1/b = 1)
  (n : ℕ) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) :=
by 
  sorry

end inequality_for_pos_reals_l139_139252


namespace exists_prime_divisor_in_sequence_l139_139317

theorem exists_prime_divisor_in_sequence
  (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d)
  (a : ℕ → ℕ)
  (h0 : a 1 = c)
  (hs : ∀ n, a (n+1) = a n ^ d + c) :
  ∀ (n : ℕ), 2 ≤ n →
  ∃ (p : ℕ), Prime p ∧ p ∣ a n ∧ ∀ i, 1 ≤ i ∧ i < n → ¬ p ∣ a i := sorry

end exists_prime_divisor_in_sequence_l139_139317


namespace suitable_chart_for_air_composition_l139_139119

/-- Given that air is a mixture of various gases, prove that the most suitable
    type of statistical chart to depict this data, while introducing it
    succinctly and effectively, is a pie chart. -/
theorem suitable_chart_for_air_composition :
  ∀ (air_composition : String) (suitable_for_introduction : String → Prop),
  (air_composition = "mixture of various gases") →
  (suitable_for_introduction "pie chart") →
  suitable_for_introduction "pie chart" :=
by
  intros air_composition suitable_for_introduction h_air_composition h_pie_chart
  sorry

end suitable_chart_for_air_composition_l139_139119


namespace find_number_l139_139050

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end find_number_l139_139050


namespace sum_of_coefficients_l139_139334

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, (x^3 - 1) * (x + 1)^7 = a_0 + a_1 * (x + 3) + 
           a_2 * (x + 3)^2 + a_3 * (x + 3)^3 + a_4 * (x + 3)^4 + 
           a_5 * (x + 3)^5 + a_6 * (x + 3)^6 + a_7 * (x + 3)^7 + 
           a_8 * (x + 3)^8 + a_9 * (x + 3)^9 + a_10 * (x + 3)^10) →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 9 := 
by
  -- proof steps skipped
  sorry

end sum_of_coefficients_l139_139334


namespace eval_expression_l139_139771

theorem eval_expression : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end eval_expression_l139_139771


namespace range_tan_squared_plus_tan_plus_one_l139_139238

theorem range_tan_squared_plus_tan_plus_one :
  (∀ y, ∃ x : ℝ, x ≠ (k : ℤ) * Real.pi + Real.pi / 2 → y = Real.tan x ^ 2 + Real.tan x + 1) ↔ 
  ∀ y, y ∈ Set.Ici (3 / 4) :=
sorry

end range_tan_squared_plus_tan_plus_one_l139_139238


namespace div_by_9_digit_B_l139_139586

theorem div_by_9_digit_B (B : ℕ) (h : (4 + B + B + 2) % 9 = 0) : B = 6 :=
by sorry

end div_by_9_digit_B_l139_139586


namespace jessie_interest_l139_139472

noncomputable def compoundInterest 
  (P : ℝ) -- Principal
  (r : ℝ) -- annual interest rate
  (n : ℕ) -- number of times interest applied per time period
  (t : ℝ) -- time periods elapsed
  : ℝ :=
  P * (1 + r / n)^(n * t)

theorem jessie_interest :
  let P := 1200
  let annual_rate := 0.08
  let periods_per_year := 2
  let years := 5
  let A := compoundInterest P annual_rate periods_per_year years
  let interest := A - P
  interest = 576.29 :=
by
  sorry

end jessie_interest_l139_139472


namespace find_line_through_midpoint_of_hyperbola_l139_139925

theorem find_line_through_midpoint_of_hyperbola
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ := (4, 1))
  (A : ℝ × ℝ := (x1, y1))
  (B : ℝ × ℝ := (x2, y2))
  (H_midpoint : P = ((x1 + x2) / 2, (y1 + y2) / 2))
  (H_hyperbola_A : (x1^2 / 4 - y1^2 = 1))
  (H_hyperbola_B : (x2^2 / 4 - y2^2 = 1)) :
  ∃ m b : ℝ, (m = 1) ∧ (b = 3) ∧ (∀ x y : ℝ, y = m * x + b → x - y - 3 = 0) := by
  sorry

end find_line_through_midpoint_of_hyperbola_l139_139925


namespace negation_of_p_l139_139790

-- Define the proposition p: ∀ x ∈ ℝ, sin x ≤ 1
def proposition_p : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- The statement to prove the negation of proposition p
theorem negation_of_p : ¬proposition_p ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end negation_of_p_l139_139790


namespace factor_polynomial_l139_139285

theorem factor_polynomial :
  ∃ (a b c d e f : ℤ), a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a = -8 ∧ b = 1 ∧ c = -3 ∧ d = 8 ∧ e = 1 ∧ f = -3) := by
  sorry

end factor_polynomial_l139_139285


namespace total_cost_of_horse_and_saddle_l139_139444

noncomputable def saddle_cost : ℝ := 1000
noncomputable def horse_cost : ℝ := 4 * saddle_cost
noncomputable def total_cost : ℝ := saddle_cost + horse_cost

theorem total_cost_of_horse_and_saddle :
    total_cost = 5000 := by
  sorry

end total_cost_of_horse_and_saddle_l139_139444


namespace locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l139_139087

noncomputable def locus_of_C (a x0 y0 ξ η : ℝ) : Prop :=
  (x0 - ξ) * η^2 - 2 * ξ * y0 * η + ξ^3 - 3 * x0 * ξ^2 - a^2 * ξ + 3 * a^2 * x0 = 0

noncomputable def special_case (a ξ η : ℝ) : Prop :=
  ξ = 0 ∨ ξ^2 + η^2 = a^2

theorem locus_of_C_general_case_eq_cubic (a x0 y0 ξ η : ℝ) (hs: locus_of_C a x0 y0 ξ η) : 
  locus_of_C a x0 y0 ξ η := 
  sorry

theorem locus_of_C_special_case_eq_y_axis_or_circle (a ξ η : ℝ) : 
  special_case a ξ η := 
  sorry

end locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l139_139087


namespace student_correct_ans_l139_139792

theorem student_correct_ans (c w : ℕ) (h1 : c + w = 80) (h2 : 4 * c - w = 120) : c = 40 :=
by
  sorry

end student_correct_ans_l139_139792


namespace travel_ways_A_to_C_l139_139923

-- We define the number of ways to travel from A to B
def ways_A_to_B : ℕ := 3

-- We define the number of ways to travel from B to C
def ways_B_to_C : ℕ := 2

-- We state the problem as a theorem
theorem travel_ways_A_to_C : ways_A_to_B * ways_B_to_C = 6 :=
by
  sorry

end travel_ways_A_to_C_l139_139923


namespace find_principal_sum_l139_139261

noncomputable def principal_sum (P R : ℝ) : ℝ := P * (R + 6) / 100 - P * R / 100

theorem find_principal_sum (P R : ℝ) (h : P * (R + 6) / 100 - P * R / 100 = 30) : P = 500 :=
by sorry

end find_principal_sum_l139_139261


namespace option_A_option_B_option_C_option_D_l139_139579

namespace Inequalities

theorem option_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  a + (1/a) > b + (1/b) :=
sorry

theorem option_B (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  (m + 1) / (n + 1) < m / n :=
sorry

theorem option_C (c a b : ℝ) (hc : c > 0) (ha : a > 0) (hb : b > 0) (hca : c > a) (hab : a > b) :
  a / (c - a) > b / (c - b) :=
sorry

theorem option_D (a b : ℝ) (ha : a > -1) (hb : b > -1) (hab : a ≥ b) :
  a / (a + 1) ≥ b / (b + 1) :=
sorry

end Inequalities

end option_A_option_B_option_C_option_D_l139_139579


namespace time_to_wash_car_l139_139000

theorem time_to_wash_car (W : ℕ) 
    (t_oil : ℕ := 15) 
    (t_tires : ℕ := 30) 
    (n_wash : ℕ := 9) 
    (n_oil : ℕ := 6) 
    (n_tires : ℕ := 2) 
    (total_time : ℕ := 240) 
    (h : n_wash * W + n_oil * t_oil + n_tires * t_tires = total_time) 
    : W = 10 := by
  sorry

end time_to_wash_car_l139_139000


namespace bake_sale_cookies_l139_139775

theorem bake_sale_cookies (raisin_cookies : ℕ) (oatmeal_cookies : ℕ) 
  (h1 : raisin_cookies = 42) 
  (h2 : raisin_cookies / oatmeal_cookies = 6) :
  raisin_cookies + oatmeal_cookies = 49 :=
sorry

end bake_sale_cookies_l139_139775


namespace polygon_sides_l139_139133

theorem polygon_sides (n : ℕ) (c : ℕ) 
  (h₁ : c = n * (n - 3) / 2)
  (h₂ : c = 2 * n) : n = 7 :=
sorry

end polygon_sides_l139_139133


namespace fraction_subtraction_l139_139725

theorem fraction_subtraction :
  (8 / 23) - (5 / 46) = 11 / 46 := by
  sorry

end fraction_subtraction_l139_139725


namespace unique_students_total_l139_139070

variables (euclid_students raman_students pythagoras_students overlap_3 : ℕ)

def total_students (E R P O : ℕ) : ℕ := E + R + P - O

theorem unique_students_total (hE : euclid_students = 12) 
                              (hR : raman_students = 10) 
                              (hP : pythagoras_students = 15) 
                              (hO : overlap_3 = 3) : 
    total_students euclid_students raman_students pythagoras_students overlap_3 = 34 :=
by
    sorry

end unique_students_total_l139_139070


namespace find_PB_l139_139063

variables (P A B C D : Point) (PA PD PC PB : ℝ)
-- Assume P is interior to rectangle ABCD
-- Conditions
axiom hPA : PA = 3
axiom hPD : PD = 4
axiom hPC : PC = 5

-- The main statement to prove
theorem find_PB (P A B C D : Point) (PA PD PC PB : ℝ)
  (hPA : PA = 3) (hPD : PD = 4) (hPC : PC = 5) : PB = 3 * Real.sqrt 2 :=
by
  sorry

end find_PB_l139_139063


namespace tom_blue_marbles_l139_139693

-- Definitions based on conditions
def jason_blue_marbles : Nat := 44
def total_blue_marbles : Nat := 68

-- The problem statement to prove
theorem tom_blue_marbles : (total_blue_marbles - jason_blue_marbles) = 24 :=
by
  sorry

end tom_blue_marbles_l139_139693


namespace total_cost_jello_l139_139120

def total_cost_james_spent : Real := 259.20

theorem total_cost_jello 
  (pounds_per_cubic_foot : ℝ := 8)
  (gallons_per_cubic_foot : ℝ := 7.5)
  (tablespoons_per_pound : ℝ := 1.5)
  (cost_red_jello : ℝ := 0.50)
  (cost_blue_jello : ℝ := 0.40)
  (cost_green_jello : ℝ := 0.60)
  (percentage_red_jello : ℝ := 0.60)
  (percentage_blue_jello : ℝ := 0.30)
  (percentage_green_jello : ℝ := 0.10)
  (volume_cubic_feet : ℝ := 6) :
  (volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_red_jello * cost_red_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_blue_jello * cost_blue_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_green_jello * cost_green_jello) = total_cost_james_spent :=
by
  sorry

end total_cost_jello_l139_139120


namespace problem_solution_l139_139124

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f x + 2 * Real.cos x ^ 2

theorem problem_solution :
  (∀ x, (∃ ω > 0, ∃ φ, |φ| < Real.pi / 2 ∧ Real.sin (ω * x - φ) = 0 ∧ 2 * ω = Real.pi)) →
  (∀ x, f x = Real.sin (2 * x - Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (g x ≤ 2 ∧ g x ≥ 1 / 2)) :=
by
  sorry

end problem_solution_l139_139124


namespace price_of_adult_ticket_l139_139092

/--
Given:
1. The price of a child's ticket is half the price of an adult's ticket.
2. Janet buys tickets for 10 people, 4 of whom are children.
3. Janet buys a soda for $5.
4. With the soda, Janet gets a 20% discount on the total admission price.
5. Janet paid $197 in total for everything.

Prove that the price of an adult admission ticket is $30.
-/
theorem price_of_adult_ticket : 
  ∃ (A : ℝ), 
  (∀ (childPrice adultPrice total : ℝ),
    adultPrice = A →
    childPrice = A / 2 →
    total = adultPrice * 6 + childPrice * 4 →
    totalPriceWithDiscount = 192 →
    total / 0.8 = total + 5 →
    A = 30) :=
sorry

end price_of_adult_ticket_l139_139092


namespace mari_buttons_l139_139671

/-- 
Given that:
1. Sue made 6 buttons
2. Sue made half as many buttons as Kendra.
3. Mari made 4 more than five times as many buttons as Kendra.

We are to prove that Mari made 64 buttons.
-/
theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : sue_buttons = kendra_buttons / 2)
  (h3 : mari_buttons = 5 * kendra_buttons + 4) :
  mari_buttons = 64 :=
  sorry

end mari_buttons_l139_139671


namespace solve_for_x_l139_139312

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 :=
by
  -- reduce the problem to its final steps
  sorry

end solve_for_x_l139_139312


namespace no_solutions_interval_length_l139_139507

theorem no_solutions_interval_length : 
  (∀ x a : ℝ, |x| ≠ ax - 2) → ([-1, 1].length = 2) :=
by {
  sorry
}

end no_solutions_interval_length_l139_139507


namespace temperature_representation_l139_139449

def represents_zero_degrees_celsius (t₁ : ℝ) : Prop := t₁ = 10

theorem temperature_representation (t₁ t₂ : ℝ) (h₀ : represents_zero_degrees_celsius t₁) 
    (h₁ : t₂ > t₁):
    t₂ = 17 :=
by
  -- Proof is omitted here
  sorry

end temperature_representation_l139_139449


namespace percent_employed_l139_139299

theorem percent_employed (E : ℝ) : 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30 -- 1 - percent_females
  (percent_males * E = employed_males) → E = 70 := 
by 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30
  intro h
  sorry

end percent_employed_l139_139299


namespace johns_donation_l139_139958

theorem johns_donation (A J : ℝ) 
  (h1 : (75 / 1.5) = A) 
  (h2 : A * 2 = 100)
  (h3 : (100 + J) / 3 = 75) : 
  J = 125 :=
by 
  sorry

end johns_donation_l139_139958


namespace average_math_score_first_year_students_l139_139457

theorem average_math_score_first_year_students 
  (total_male_students : ℕ) (total_female_students : ℕ)
  (sample_size : ℕ) (avg_score_male : ℕ) (avg_score_female : ℕ)
  (male_sample_size female_sample_size : ℕ)
  (weighted_avg : ℚ) :
  total_male_students = 300 → 
  total_female_students = 200 →
  sample_size = 60 → 
  avg_score_male = 110 →
  avg_score_female = 100 →
  male_sample_size = (3 * sample_size) / 5 →
  female_sample_size = (2 * sample_size) / 5 →
  weighted_avg = (male_sample_size * avg_score_male + female_sample_size * avg_score_female : ℕ) / sample_size → 
  weighted_avg = 106 := 
by
  sorry

end average_math_score_first_year_students_l139_139457


namespace necessary_but_not_sufficient_condition_l139_139043

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 3 * x < 0) → (0 < x ∧ x < 4) :=
sorry

end necessary_but_not_sufficient_condition_l139_139043


namespace f_at_neg_2_l139_139841

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + x^2 + b * x + 2

-- Given the condition
def f_at_2_eq_3 (a b : ℝ) : Prop := f 2 a b = 3

-- Prove the value of f(-2)
theorem f_at_neg_2 (a b : ℝ) (h : f_at_2_eq_3 a b) : f (-2) a b = 1 :=
sorry

end f_at_neg_2_l139_139841


namespace flattest_ellipse_is_B_l139_139145

-- Definitions for the given ellipses
def ellipseA : Prop := ∀ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1)
def ellipseB : Prop := ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1)
def ellipseC : Prop := ∀ (x y : ℝ), (x^2 / 6 + y^2 / 3 = 1)
def ellipseD : Prop := ∀ (x y : ℝ), (x^2 / 9 + y^2 / 8 = 1)

-- The proof to show that ellipseB is the flattest
theorem flattest_ellipse_is_B : ellipseB := by
  sorry

end flattest_ellipse_is_B_l139_139145


namespace javier_average_hits_l139_139599

-- Define the total number of games Javier plays and the first set number of games
def total_games := 30
def first_set_games := 20

-- Define the hit averages for the first set of games and the desired season average
def average_hits_first_set := 2
def desired_season_average := 3

-- Define the total hits Javier needs to achieve the desired average by the end of the season
def total_hits_needed : ℕ := total_games * desired_season_average

-- Define the hits Javier made in the first set of games
def hits_made_first_set : ℕ := first_set_games * average_hits_first_set

-- Define the remaining games and the hits Javier needs to achieve in these games to meet his target
def remaining_games := total_games - first_set_games
def hits_needed_remaining_games : ℕ := total_hits_needed - hits_made_first_set

-- Define the average hits Javier needs in the remaining games to meet his target
def average_needed_remaining_games (remaining_games hits_needed_remaining_games : ℕ) : ℕ :=
  hits_needed_remaining_games / remaining_games

theorem javier_average_hits : 
  average_needed_remaining_games remaining_games hits_needed_remaining_games = 5 := 
by
  -- The proof is omitted.
  sorry

end javier_average_hits_l139_139599


namespace one_fifth_of_5_times_7_l139_139568

theorem one_fifth_of_5_times_7 : (1 / 5) * (5 * 7) = 7 := by
  sorry

end one_fifth_of_5_times_7_l139_139568


namespace ratio_of_times_l139_139902

-- Given conditions as definitions
def distance : ℕ := 630 -- distance in km
def previous_time : ℕ := 6 -- time in hours
def new_speed : ℕ := 70 -- speed in km/h

-- Calculation of times
def previous_speed : ℕ := distance / previous_time

def new_time : ℕ := distance / new_speed

-- Main theorem statement
theorem ratio_of_times :
  (new_time : ℚ) / (previous_time : ℚ) = 3 / 2 :=
  sorry

end ratio_of_times_l139_139902


namespace fractions_of_120_equals_2_halves_l139_139178

theorem fractions_of_120_equals_2_halves :
  (1 / 6) * (1 / 4) * (1 / 5) * 120 = 2 / 2 := 
by
  sorry

end fractions_of_120_equals_2_halves_l139_139178


namespace hex_to_decimal_B4E_l139_139780

def hex_B := 11
def hex_4 := 4
def hex_E := 14
def base := 16
def hex_value := hex_B * base^2 + hex_4 * base^1 + hex_E * base^0

theorem hex_to_decimal_B4E : hex_value = 2894 :=
by
  -- here we would write the proof steps, this is skipped with "sorry"
  sorry

end hex_to_decimal_B4E_l139_139780


namespace equivalent_proof_problem_l139_139960

def math_problem (x y : ℚ) : ℚ :=
((x + y) * (3 * x - y) + y^2) / (-x)

theorem equivalent_proof_problem (hx : x = 4) (hy : y = -(1/4)) :
  math_problem x y = -23 / 2 :=
by
  sorry

end equivalent_proof_problem_l139_139960


namespace total_price_purchase_l139_139724

variable (S T : ℝ)

theorem total_price_purchase (h1 : 2 * S + T = 2600) (h2 : 900 = 1200 * 0.75) : 2600 + 900 = 3500 := by
  sorry

end total_price_purchase_l139_139724


namespace probability_of_picking_letter_from_MATHEMATICS_l139_139079

theorem probability_of_picking_letter_from_MATHEMATICS : 
  (8 : ℤ) / 26 = (4 : ℤ) / 13 :=
by
  norm_num

end probability_of_picking_letter_from_MATHEMATICS_l139_139079


namespace molly_age_is_63_l139_139417

variable (Sandy_age Molly_age : ℕ)

theorem molly_age_is_63 (h1 : Sandy_age = 49) (h2 : Sandy_age / Molly_age = 7 / 9) : Molly_age = 63 :=
by
  sorry

end molly_age_is_63_l139_139417


namespace combined_average_mark_l139_139028

theorem combined_average_mark 
  (n_A n_B n_C n_D n_E : ℕ) 
  (avg_A avg_B avg_C avg_D avg_E : ℕ)
  (students_A : n_A = 22) (students_B : n_B = 28)
  (students_C : n_C = 15) (students_D : n_D = 35)
  (students_E : n_E = 25)
  (avg_marks_A : avg_A = 40) (avg_marks_B : avg_B = 60)
  (avg_marks_C : avg_C = 55) (avg_marks_D : avg_D = 75)
  (avg_marks_E : avg_E = 50) : 
  (22 * 40 + 28 * 60 + 15 * 55 + 35 * 75 + 25 * 50) / (22 + 28 + 15 + 35 + 25) = 58.08 := 
  by 
    sorry

end combined_average_mark_l139_139028


namespace no_solution_if_and_only_if_zero_l139_139628

theorem no_solution_if_and_only_if_zero (n : ℝ) :
  ¬(∃ (x y z : ℝ), 2 * n * x + y = 2 ∧ 3 * n * y + z = 3 ∧ x + 2 * n * z = 2) ↔ n = 0 := 
  by
  sorry

end no_solution_if_and_only_if_zero_l139_139628


namespace product_gcd_lcm_24_60_l139_139941

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l139_139941


namespace student_count_l139_139129

theorem student_count 
  (initial_avg_height : ℚ)
  (incorrect_height : ℚ)
  (actual_height : ℚ)
  (actual_avg_height : ℚ)
  (n : ℕ)
  (h1 : initial_avg_height = 175)
  (h2 : incorrect_height = 151)
  (h3 : actual_height = 136)
  (h4 : actual_avg_height = 174.5)
  (h5 : n > 0) : n = 30 :=
by
  sorry

end student_count_l139_139129


namespace zack_initial_marbles_l139_139029

noncomputable def total_initial_marbles (x : ℕ) : ℕ :=
  81 * x + 27

theorem zack_initial_marbles :
  ∃ x : ℕ, total_initial_marbles x = 270 :=
by
  use 3
  sorry

end zack_initial_marbles_l139_139029


namespace total_plates_l139_139095

-- Define the initial conditions
def flower_plates_initial : ℕ := 4
def checked_plates : ℕ := 8
def polka_dotted_plates := 2 * checked_plates
def flower_plates_remaining := flower_plates_initial - 1

-- Prove the total number of plates Jack has left
theorem total_plates : flower_plates_remaining + polka_dotted_plates + checked_plates = 27 :=
by
  sorry

end total_plates_l139_139095


namespace A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l139_139799

def A : Set ℝ := { x | x^2 + x - 2 < 0 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

theorem A_union_B_when_m_neg_half : A ∪ B (-1/2) = { x | -2 < x ∧ x < 3/2 } :=
by
  sorry

theorem B_subset_A_implies_m_geq_zero (m : ℝ) : B m ⊆ A → 0 ≤ m :=
by
  sorry

end A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l139_139799


namespace distance_between_house_and_school_l139_139955

variable (T D : ℝ)

axiom cond1 : 9 * (T + 20 / 60) = D
axiom cond2 : 12 * (T - 20 / 60) = D
axiom cond3 : 15 * (T - 40 / 60) = D

theorem distance_between_house_and_school : D = 24 := 
by
  sorry

end distance_between_house_and_school_l139_139955


namespace scout_troop_profit_calc_l139_139390

theorem scout_troop_profit_calc
  (candy_bars : ℕ := 1200)
  (purchase_rate : ℚ := 3/6)
  (sell_rate : ℚ := 2/3) :
  (candy_bars * sell_rate - candy_bars * purchase_rate) = 200 :=
by
  sorry

end scout_troop_profit_calc_l139_139390


namespace probability_two_people_between_l139_139347

theorem probability_two_people_between (total_people : ℕ) (favorable_arrangements : ℕ) (total_arrangements : ℕ) :
  total_people = 6 ∧ favorable_arrangements = 144 ∧ total_arrangements = 720 →
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 5 :=
by
  intros h
  -- We substitute the given conditions
  have ht : total_people = 6 := h.1
  have hf : favorable_arrangements = 144 := h.2.1
  have ha : total_arrangements = 720 := h.2.2
  -- We need to calculate the probability considering the favorable and total arrangements
  sorry

end probability_two_people_between_l139_139347


namespace points_on_inverse_proportion_l139_139949

theorem points_on_inverse_proportion (y_1 y_2 : ℝ) :
  (2:ℝ) = 5 / y_1 → (3:ℝ) = 5 / y_2 → y_1 > y_2 :=
by
  intros h1 h2
  sorry

end points_on_inverse_proportion_l139_139949


namespace solve_for_a_l139_139885

theorem solve_for_a (x a : ℝ) (h : x = 3) (eqn : 2 * (x - 1) - a = 0) : a = 4 := 
by 
  sorry

end solve_for_a_l139_139885


namespace age_sum_is_47_l139_139574

theorem age_sum_is_47 (a b c : ℕ) (b_def : b = 18) 
  (a_def : a = b + 2) (c_def : c = b / 2) : a + b + c = 47 :=
by
  sorry

end age_sum_is_47_l139_139574


namespace correlations_are_1_3_4_l139_139034

def relation1 : Prop := ∃ (age wealth : ℝ), true
def relation2 : Prop := ∀ (point : ℝ × ℝ), ∃ (coords : ℝ × ℝ), coords = point
def relation3 : Prop := ∃ (yield : ℝ) (climate : ℝ), true
def relation4 : Prop := ∃ (diameter height : ℝ), true
def relation5 : Prop := ∃ (student : Type) (school : Type), true

theorem correlations_are_1_3_4 :
  (relation1 ∨ relation3 ∨ relation4) ∧ ¬ (relation2 ∨ relation5) :=
sorry

end correlations_are_1_3_4_l139_139034


namespace rod_total_length_l139_139703

theorem rod_total_length (n : ℕ) (piece_length : ℝ) (total_length : ℝ) 
  (h1 : n = 50) 
  (h2 : piece_length = 0.85) 
  (h3 : total_length = n * piece_length) : 
  total_length = 42.5 :=
by
  -- Proof steps will go here
  sorry

end rod_total_length_l139_139703


namespace puppy_weight_l139_139368

variable (a b c : ℝ)

theorem puppy_weight :
  (a + b + c = 30) →
  (a + c = 3 * b) →
  (a + b = c) →
  a = 7.5 := by
  intros h1 h2 h3
  sorry

end puppy_weight_l139_139368


namespace selling_price_is_correct_l139_139215

-- Definitions of the given conditions

def cost_of_string_per_bracelet := 1
def cost_of_beads_per_bracelet := 3
def number_of_bracelets_sold := 25
def total_profit := 50

def cost_of_bracelet := cost_of_string_per_bracelet + cost_of_beads_per_bracelet
def total_cost := cost_of_bracelet * number_of_bracelets_sold
def total_revenue := total_profit + total_cost
def selling_price_per_bracelet := total_revenue / number_of_bracelets_sold

-- Target theorem
theorem selling_price_is_correct : selling_price_per_bracelet = 6 :=
  by
  sorry

end selling_price_is_correct_l139_139215


namespace sum_a_c_eq_13_l139_139993

noncomputable def conditions (a b c d k : ℤ) :=
  d = a * b * c ∧
  1 < a ∧ a < b ∧ b < c ∧
  233 = d * k + 79

theorem sum_a_c_eq_13 (a b c d k : ℤ) (h : conditions a b c d k) : a + c = 13 := by
  sorry

end sum_a_c_eq_13_l139_139993


namespace original_price_of_lens_is_correct_l139_139623

-- Definitions based on conditions
def current_camera_price : ℝ := 4000
def new_camera_price : ℝ := current_camera_price + 0.30 * current_camera_price
def combined_price_paid : ℝ := 5400
def lens_discount : ℝ := 200
def combined_price_before_discount : ℝ := combined_price_paid + lens_discount

-- Calculated original price of the lens
def lens_original_price : ℝ := combined_price_before_discount - new_camera_price

-- The Lean theorem statement to prove the price is correct
theorem original_price_of_lens_is_correct : lens_original_price = 400 := by
  -- You do not need to provide the actual proof steps
  sorry

end original_price_of_lens_is_correct_l139_139623


namespace part1_part2_l139_139206

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l139_139206


namespace flowers_per_day_l139_139991

-- Definitions for conditions
def total_flowers := 360
def days := 6

-- Proof that the number of flowers Miriam can take care of in one day is 60
theorem flowers_per_day : total_flowers / days = 60 := by
  sorry

end flowers_per_day_l139_139991


namespace julia_tuesday_kids_l139_139816

-- Definitions based on conditions
def kids_on_monday : ℕ := 11
def tuesday_more_than_monday : ℕ := 1

-- The main statement to be proved
theorem julia_tuesday_kids : (kids_on_monday + tuesday_more_than_monday) = 12 := by
  sorry

end julia_tuesday_kids_l139_139816


namespace hyperbola_y_relation_l139_139298

theorem hyperbola_y_relation {k y₁ y₂ : ℝ} 
  (A_on_hyperbola : y₁ = k / 2) 
  (B_on_hyperbola : y₂ = k / 3) 
  (k_positive : 0 < k) : 
  y₁ > y₂ := 
sorry

end hyperbola_y_relation_l139_139298


namespace time_ratio_upstream_downstream_l139_139527

theorem time_ratio_upstream_downstream (S_boat S_stream D : ℝ) (h1 : S_boat = 72) (h2 : S_stream = 24) :
  let time_upstream := D / (S_boat - S_stream)
  let time_downstream := D / (S_boat + S_stream)
  (time_upstream / time_downstream) = 2 :=
by
  sorry

end time_ratio_upstream_downstream_l139_139527


namespace problem_l139_139854

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (3 * x - Real.pi / 3)

theorem problem 
  (x₁ x₂ : ℝ)
  (hx₁x₂ : |f x₁ - f x₂| = 4)
  (x : ℝ)
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 6)
  (m : ℝ) : m ≥ 1 / 3 :=
sorry

end problem_l139_139854


namespace min_value_g_squared_plus_f_l139_139950

def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_g_squared_plus_f (a b c : ℝ) (h : a ≠ 0) 
  (min_f_squared_plus_g : ∀ x : ℝ, (f a b x)^2 + g a c x ≥ 4)
  (exists_x_min : ∃ x : ℝ, (f a b x)^2 + g a c x = 4) :
  ∃ x : ℝ, (g a c x)^2 + f a b x = -9 / 2 :=
sorry

end min_value_g_squared_plus_f_l139_139950


namespace total_drink_volume_l139_139487

theorem total_drink_volume (coke_parts sprite_parts mtndew_parts : ℕ) (coke_volume : ℕ) :
  coke_parts = 2 → sprite_parts = 1 → mtndew_parts = 3 → coke_volume = 6 →
  (coke_volume / coke_parts) * (coke_parts + sprite_parts + mtndew_parts) = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end total_drink_volume_l139_139487


namespace box_depth_is_10_l139_139920

variable (depth : ℕ)

theorem box_depth_is_10 
  (length width : ℕ)
  (cubes : ℕ)
  (h1 : length = 35)
  (h2 : width = 20)
  (h3 : cubes = 56)
  (h4 : ∃ (cube_size : ℕ), ∀ (c : ℕ), c = cube_size → (length % cube_size = 0 ∧ width % cube_size = 0 ∧ 56 * cube_size^3 = length * width * depth)) :
  depth = 10 :=
by
  sorry

end box_depth_is_10_l139_139920


namespace max_value_expr_bound_l139_139789

noncomputable def max_value_expr (x : ℝ) : ℝ := 
  x^6 / (x^10 + x^8 - 6 * x^6 + 27 * x^4 + 64)

theorem max_value_expr_bound : 
  ∃ x : ℝ, max_value_expr x ≤ 1 / 8.38 := sorry

end max_value_expr_bound_l139_139789


namespace probability_third_smallest_is_five_l139_139152

theorem probability_third_smallest_is_five :
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  probability = Rat.ofInt 35 / 132 :=
by
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  show probability = Rat.ofInt 35 / 132
  sorry

end probability_third_smallest_is_five_l139_139152


namespace eval_expression_l139_139013

theorem eval_expression : (Real.pi + 2023)^0 + 2 * Real.sin (45 * Real.pi / 180) - (1 / 2)^(-1 : ℤ) + abs (Real.sqrt 2 - 2) = 1 :=
by
  sorry

end eval_expression_l139_139013


namespace grace_putting_down_mulch_hours_l139_139280

/-- Grace's earnings conditions and hours calculation in September. -/
theorem grace_putting_down_mulch_hours :
  ∃ h : ℕ, 
    6 * 63 + 11 * 9 + 9 * h = 567 ∧
    h = 10 :=
by
  sorry

end grace_putting_down_mulch_hours_l139_139280


namespace single_fraction_l139_139743

theorem single_fraction (c : ℕ) : (6 + 5 * c) / 5 + 3 = (21 + 5 * c) / 5 :=
by sorry

end single_fraction_l139_139743


namespace solve_equation_l139_139519

theorem solve_equation : 
  ∀ x : ℝ,
    (x + 5 ≠ 0) → 
    (x^2 + 3 * x + 4) / (x + 5) = x + 6 → 
    x = -13 / 4 :=
by 
  intro x
  intro hx
  intro h
  sorry

end solve_equation_l139_139519


namespace sam_distance_walked_l139_139615

variable (d : ℝ := 40) -- initial distance between Fred and Sam
variable (v_f : ℝ := 4) -- Fred's constant speed in miles per hour
variable (v_s : ℝ := 4) -- Sam's constant speed in miles per hour

theorem sam_distance_walked :
  (d / (v_f + v_s)) * v_s = 20 :=
by
  sorry

end sam_distance_walked_l139_139615


namespace scientific_notation_101_49_billion_l139_139644

-- Define the term "one hundred and one point four nine billion"
def billion (n : ℝ) := n * 10^9

-- Axiomatization of the specific number in question
def hundredOnePointFourNineBillion := billion 101.49

-- Theorem stating that the scientific notation for 101.49 billion is 1.0149 × 10^10
theorem scientific_notation_101_49_billion : hundredOnePointFourNineBillion = 1.0149 * 10^10 :=
by
  sorry

end scientific_notation_101_49_billion_l139_139644


namespace jane_not_finish_probability_l139_139480

theorem jane_not_finish_probability :
  (1 : ℚ) - (5 / 8) = (3 / 8) := by
  sorry

end jane_not_finish_probability_l139_139480


namespace find_a_l139_139116

theorem find_a (a : ℝ) (A B : ℝ × ℝ × ℝ) (hA : A = (-1, 1, -a)) (hB : B = (-a, 3, -1)) (hAB : dist A B = 2) : a = -1 := by
  sorry

end find_a_l139_139116


namespace part_a_sequence_l139_139641

def circle_sequence (n m : ℕ) : List ℕ :=
  List.replicate m 1 -- Placeholder: Define the sequence computation properly

theorem part_a_sequence :
  circle_sequence 5 12 = [1, 6, 11, 4, 9, 2, 7, 12, 5, 10, 3, 8, 1] := 
sorry

end part_a_sequence_l139_139641


namespace range_of_a_l139_139996

open Set

variable {a : ℝ} 

def M (a : ℝ) : Set ℝ := {x : ℝ | -4 * x + 4 * a < 0 }

theorem range_of_a (hM : 2 ∉ M a) : a ≥ 2 :=
by
  sorry

end range_of_a_l139_139996


namespace amount_saved_by_Dalton_l139_139861

-- Defining the costs of each item and the given conditions
def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def uncle_gift : ℕ := 13
def additional_needed : ℕ := 4

-- Calculated values based on the conditions
def total_cost : ℕ := jump_rope_cost + board_game_cost + playground_ball_cost
def total_money_needed : ℕ := uncle_gift + additional_needed

-- The theorem that needs to be proved
theorem amount_saved_by_Dalton : total_cost - total_money_needed = 6 :=
by
  sorry -- Proof to be filled in

end amount_saved_by_Dalton_l139_139861


namespace tom_calories_l139_139239

theorem tom_calories :
  let carrot_pounds := 1
  let broccoli_pounds := 2 * carrot_pounds
  let carrot_calories_per_pound := 51
  let broccoli_calories_per_pound := carrot_calories_per_pound / 3
  let total_carrot_calories := carrot_pounds * carrot_calories_per_pound
  let total_broccoli_calories := broccoli_pounds * broccoli_calories_per_pound
  let total_calories := total_carrot_calories + total_broccoli_calories
  total_calories = 85 :=
by
  sorry

end tom_calories_l139_139239


namespace region_ratio_l139_139040

theorem region_ratio (side_length : ℝ) (s r : ℝ) 
  (h1 : side_length = 2)
  (h2 : s = (1 / 2) * (1 : ℝ) * (1 : ℝ))
  (h3 : r = (1 / 2) * (Real.sqrt 2) * (Real.sqrt 2)) :
  r / s = 2 :=
by
  sorry

end region_ratio_l139_139040


namespace no_natural_number_divides_Q_by_x_squared_minus_one_l139_139570

def Q (n : ℕ) (x : ℝ) : ℝ := 1 + 5*x^2 + x^4 - (n - 1) * x^(n - 1) + (n - 8) * x^n

theorem no_natural_number_divides_Q_by_x_squared_minus_one :
  ∀ (n : ℕ), n > 0 → ¬ (x^2 - 1 ∣ Q n x) :=
by
  intros n h
  sorry

end no_natural_number_divides_Q_by_x_squared_minus_one_l139_139570


namespace selling_price_750_max_daily_profit_l139_139197

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 10) * (-10 * x + 300)

theorem selling_price_750 (x : ℝ) : profit x = 750 ↔ (x = 15 ∨ x = 25) :=
by sorry

theorem max_daily_profit : (∀ x : ℝ, profit x ≤ 1000) ∧ (profit 20 = 1000) :=
by sorry

end selling_price_750_max_daily_profit_l139_139197


namespace compare_sqrt_l139_139300

theorem compare_sqrt : 3 * Real.sqrt 2 > Real.sqrt 17 := by
  sorry

end compare_sqrt_l139_139300


namespace village_population_l139_139898

theorem village_population (P : ℕ) (h : 80 * P = 32000 * 100) : P = 40000 :=
sorry

end village_population_l139_139898


namespace king_then_ten_prob_l139_139091

def num_kings : ℕ := 4
def num_tens : ℕ := 4
def deck_size : ℕ := 52
def first_card_draw_prob := (num_kings : ℚ) / (deck_size : ℚ)
def second_card_draw_prob := (num_tens : ℚ) / (deck_size - 1 : ℚ)

theorem king_then_ten_prob : 
  first_card_draw_prob * second_card_draw_prob = 4 / 663 := by
  sorry

end king_then_ten_prob_l139_139091


namespace grazing_months_b_l139_139573

theorem grazing_months_b (a_oxen a_months b_oxen c_oxen c_months total_rent c_share : ℕ) (x : ℕ) 
  (h_a : a_oxen = 10) (h_am : a_months = 7) (h_b : b_oxen = 12) 
  (h_c : c_oxen = 15) (h_cm : c_months = 3) (h_tr : total_rent = 105) 
  (h_cs : c_share = 27) : 
  45 * 105 = 27 * (70 + 12 * x + 45) → x = 5 :=
by
  sorry

end grazing_months_b_l139_139573


namespace adam_remaining_loads_l139_139638

-- Define the initial conditions
def total_loads : ℕ := 25
def washed_loads : ℕ := 6

-- Define the remaining loads as the total loads minus the washed loads
def remaining_loads (total_loads washed_loads : ℕ) : ℕ := total_loads - washed_loads

-- State the theorem to be proved
theorem adam_remaining_loads : remaining_loads total_loads washed_loads = 19 := by
  sorry

end adam_remaining_loads_l139_139638


namespace rectangle_solution_l139_139824

-- Define the given conditions
variables (x y : ℚ)

-- Given equations
def condition1 := (Real.sqrt (x - y) = 2 / 5)
def condition2 := (Real.sqrt (x + y) = 2)

-- Solution
theorem rectangle_solution (x y : ℚ) (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 52 / 25 ∧ y = 48 / 25 ∧ (Real.sqrt ((52 / 25) * (48 / 25)) = 8 / 25) :=
by
  sorry

end rectangle_solution_l139_139824


namespace hexagon_area_ratio_l139_139055

open Real

theorem hexagon_area_ratio (r s : ℝ) (h_eq_diam : s = r * sqrt 3) :
    (let a1 := (3 * sqrt 3 / 2) * ((3 * r / 4) ^ 2)
     let a2 := (3 * sqrt 3 / 2) * r^2
     a1 / a2 = 9 / 16) :=
by
  sorry

end hexagon_area_ratio_l139_139055


namespace coastal_city_spending_l139_139819

def beginning_of_may_spending : ℝ := 1.2
def end_of_september_spending : ℝ := 4.5

theorem coastal_city_spending :
  (end_of_september_spending - beginning_of_may_spending) = 3.3 :=
by
  -- Proof can be filled in here
  sorry

end coastal_city_spending_l139_139819


namespace smallest_number_is_28_l139_139722

theorem smallest_number_is_28 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : (a + b + c) / 3 = 30) (h4 : b = 29) (h5 : c = b + 4) : a = 28 :=
by
  sorry

end smallest_number_is_28_l139_139722


namespace geometric_series_sum_l139_139282

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (last_term : ℝ) 
  (h_a : a = 1) (h_r : r = -3) 
  (h_last_term : last_term = 6561) 
  (h_last_term_eq : a * r^n = last_term) : 
  a * (r^n - 1) / (r - 1) = 4921.25 :=
by
  -- Proof goes here
  sorry

end geometric_series_sum_l139_139282


namespace odd_f_l139_139012

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2^x else if x < 0 then -x^2 + 2^(-x) else 0

theorem odd_f (x : ℝ) : (f (-x) = -f x) :=
by
  sorry

end odd_f_l139_139012


namespace expression_evaluation_l139_139456

theorem expression_evaluation : (3 * 15) + 47 - 27 * (2^3) / 4 = 38 := by
  sorry

end expression_evaluation_l139_139456


namespace bobArrivesBefore845Prob_l139_139321

noncomputable def probabilityBobBefore845 (totalTime: ℕ) (cutoffTime: ℕ) : ℚ :=
  let totalArea := (totalTime * totalTime) / 2
  let areaOfInterest := (cutoffTime * cutoffTime) / 2
  (areaOfInterest : ℚ) / totalArea

theorem bobArrivesBefore845Prob (totalTime: ℕ) (cutoffTime: ℕ) (ht: totalTime = 60) (hc: cutoffTime = 45) :
  probabilityBobBefore845 totalTime cutoffTime = 9 / 16 := by
  sorry

end bobArrivesBefore845Prob_l139_139321


namespace martha_weight_l139_139064

theorem martha_weight :
  ∀ (Bridget_weight : ℕ) (difference : ℕ) (Martha_weight : ℕ),
  Bridget_weight = 39 → difference = 37 →
  Bridget_weight = Martha_weight + difference →
  Martha_weight = 2 :=
by
  intros Bridget_weight difference Martha_weight hBridget hDifference hRelation
  sorry

end martha_weight_l139_139064


namespace theater_seats_l139_139207

theorem theater_seats (x y t : ℕ) (h1 : x = 532) (h2 : y = 218) (h3 : t = x + y) : t = 750 := 
by 
  rw [h1, h2] at h3
  exact h3

end theater_seats_l139_139207


namespace multiple_rate_is_correct_l139_139374

-- Define Lloyd's standard working hours per day
def regular_hours_per_day : ℝ := 7.5

-- Define Lloyd's standard hourly rate
def regular_rate : ℝ := 3.5

-- Define the total hours worked on a specific day
def total_hours_worked : ℝ := 10.5

-- Define the total earnings for that specific day
def total_earnings : ℝ := 42.0

-- Define the function to calculate the multiple of the regular rate for excess hours
noncomputable def multiple_of_regular_rate (r_hours : ℝ) (r_rate : ℝ) (t_hours : ℝ) (t_earnings : ℝ) : ℝ :=
  let regular_earnings := r_hours * r_rate
  let excess_hours := t_hours - r_hours
  let excess_earnings := t_earnings - regular_earnings
  (excess_earnings / excess_hours) / r_rate

-- The statement to be proved
theorem multiple_rate_is_correct : 
  multiple_of_regular_rate regular_hours_per_day regular_rate total_hours_worked total_earnings = 1.5 :=
by
  sorry

end multiple_rate_is_correct_l139_139374


namespace find_value_of_expression_l139_139812

theorem find_value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 4 * y + 13) : (12 * y - 5)^2 = 161 :=
sorry

end find_value_of_expression_l139_139812


namespace curve1_line_and_circle_curve2_two_points_l139_139038

-- Define the first condition: x(x^2 + y^2 - 4) = 0
def curve1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0

-- Define the second condition: x^2 + (x^2 + y^2 - 4)^2 = 0
def curve2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- The corresponding theorem statements
theorem curve1_line_and_circle : ∀ x y : ℝ, curve1 x y ↔ (x = 0 ∨ (x^2 + y^2 = 4)) := 
sorry 

theorem curve2_two_points : ∀ x y : ℝ, curve2 x y ↔ (x = 0 ∧ (y = 2 ∨ y = -2)) := 
sorry 

end curve1_line_and_circle_curve2_two_points_l139_139038


namespace A_minus_B_l139_139829

theorem A_minus_B (A B : ℚ) (n : ℕ) :
  (A : ℚ) = 1 / 6 →
  (B : ℚ) = -1 / 12 →
  A - B = 1 / 4 :=
by
  intro hA hB
  rw [hA, hB]
  norm_num

end A_minus_B_l139_139829


namespace trigonometric_inequality_l139_139463

theorem trigonometric_inequality (a b x : ℝ) :
  (Real.sin x + a * Real.cos x) * (Real.sin x + b * Real.cos x) ≤ 1 + ( (a + b) / 2 )^2 :=
by
  sorry

end trigonometric_inequality_l139_139463


namespace distribution_ways_l139_139753

def number_of_ways_to_distribute_problems : ℕ :=
  let friends := 10
  let problems := 7
  let max_receivers := 3
  let ways_to_choose_friends := Nat.choose friends max_receivers
  let ways_to_distribute_problems := max_receivers ^ problems
  ways_to_choose_friends * ways_to_distribute_problems

theorem distribution_ways :
  number_of_ways_to_distribute_problems = 262440 :=
by
  -- Proof is omitted
  sorry

end distribution_ways_l139_139753


namespace A_times_B_correct_l139_139647

noncomputable def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {y | y > 1}
noncomputable def A_times_B : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem A_times_B_correct : A_times_B = {x | (0 ≤ x ∧ x ≤ 1) ∨ x > 2} := 
sorry

end A_times_B_correct_l139_139647


namespace bananas_first_day_l139_139435

theorem bananas_first_day (x : ℕ) (h : x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 100) : x = 8 := by
  sorry

end bananas_first_day_l139_139435


namespace parallelogram_vector_sum_l139_139970

theorem parallelogram_vector_sum (A B C D : ℝ × ℝ) (parallelogram : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D ∧ (C - A = D - B) ∧ (B - D = A - C)) :
  (B - A) + (C - B) = C - A :=
by
  sorry

end parallelogram_vector_sum_l139_139970


namespace range_of_x_l139_139102

noncomputable def a (x : ℝ) : ℝ := x
def b : ℝ := 2
def B : ℝ := 60

-- State the problem: Prove the range of x given the conditions
theorem range_of_x (x : ℝ) (A : ℝ) (C : ℝ) (h1 : a x = b / (Real.sin (B * Real.pi / 180)) * (Real.sin (A * Real.pi / 180)))
  (h2 : A + C = 180 - 60) (two_solutions : (60 < A ∧ A < 120)) :
  2 < x ∧ x < 4 * Real.sqrt 3 / 3 :=
sorry

end range_of_x_l139_139102


namespace third_term_of_arithmetic_sequence_is_negative_22_l139_139193

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem third_term_of_arithmetic_sequence_is_negative_22
  (a d : ℤ)
  (H1 : arithmetic_sequence a d 14 = 14)
  (H2 : arithmetic_sequence a d 15 = 17) :
  arithmetic_sequence a d 2 = -22 :=
sorry

end third_term_of_arithmetic_sequence_is_negative_22_l139_139193


namespace glove_probability_correct_l139_139727

noncomputable def glove_probability : ℚ :=
  let red_pair := ("r1", "r2") -- pair of red gloves
  let black_pair := ("b1", "b2") -- pair of black gloves
  let white_pair := ("w1", "w2") -- pair of white gloves
  let all_pairs := [
    (red_pair.1, red_pair.2), 
    (black_pair.1, black_pair.2), 
    (white_pair.1, white_pair.2),
    (red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
    (red_pair.2, black_pair.1), (red_pair.2, white_pair.1),
    (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)
  ]
  let valid_pairs := [(red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
                      (red_pair.2, black_pair.1), (red_pair.2, white_pair.1), 
                      (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)]
  (valid_pairs.length : ℚ) / (all_pairs.length : ℚ)

theorem glove_probability_correct :
  glove_probability = 2 / 5 := 
by
  sorry

end glove_probability_correct_l139_139727


namespace divisible_by_primes_l139_139086

theorem divisible_by_primes (x y z : ℕ) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  (100100 * x + 10010 * y + 1001 * z) % 7 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 11 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 13 = 0 := 
by
  sorry

end divisible_by_primes_l139_139086


namespace crayons_initial_total_l139_139992

theorem crayons_initial_total 
  (lost_given : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : lost_given = 70) (h2 : left = 183) : 
  initial = lost_given + left := 
by
  sorry

end crayons_initial_total_l139_139992


namespace find_a_l139_139212

theorem find_a (a : ℚ) (h : a + a / 3 + a / 4 = 4) : a = 48 / 19 := by
  sorry

end find_a_l139_139212


namespace probability_sum_10_l139_139797

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end probability_sum_10_l139_139797


namespace find_L_for_perfect_square_W_l139_139596

theorem find_L_for_perfect_square_W :
  ∃ L W : ℕ, 1000 < W ∧ W < 2000 ∧ L > 1 ∧ W = 2 * L^3 ∧ ∃ m : ℕ, W = m^2 ∧ L = 8 :=
by sorry

end find_L_for_perfect_square_W_l139_139596


namespace problem_C_plus_D_l139_139493

theorem problem_C_plus_D (C D : ℚ)
  (h : ∀ x, (D * x - 17) / (x^2 - 8 * x + 15) = C / (x - 3) + 5 / (x - 5)) :
  C + D = 5.8 :=
sorry

end problem_C_plus_D_l139_139493


namespace stuffed_animals_count_l139_139528

theorem stuffed_animals_count
  (total_prizes : ℕ)
  (frisbees : ℕ)
  (yoyos : ℕ)
  (h1 : total_prizes = 50)
  (h2 : frisbees = 18)
  (h3 : yoyos = 18) :
  (total_prizes - (frisbees + yoyos) = 14) :=
by
  sorry

end stuffed_animals_count_l139_139528


namespace log_base_problem_l139_139408

noncomputable def log_of_base (base value : ℝ) : ℝ := Real.log value / Real.log base

theorem log_base_problem (x : ℝ) (h : log_of_base 16 (x - 3) = 1 / 4) : 1 / log_of_base (x - 3) 2 = 1 := 
by
  sorry

end log_base_problem_l139_139408


namespace evaluate_expression_l139_139284

theorem evaluate_expression : 8^3 + 3 * 8^2 + 3 * 8 + 1 = 729 := by
  sorry

end evaluate_expression_l139_139284


namespace determine_exponent_l139_139290

-- Declare variables
variables {x y : ℝ}
variable {n : ℕ}

-- Use condition that the terms are like terms
theorem determine_exponent (h : - x ^ 2 * y ^ n = 3 * y * x ^ 2) : n = 1 :=
sorry

end determine_exponent_l139_139290


namespace train_speed_in_kmh_l139_139850

def train_length : ℝ := 250 -- Length of the train in meters
def station_length : ℝ := 200 -- Length of the station in meters
def time_to_pass : ℝ := 45 -- Time to pass the station in seconds

theorem train_speed_in_kmh :
  (train_length + station_length) / time_to_pass * 3.6 = 36 :=
  sorry -- Proof is skipped

end train_speed_in_kmh_l139_139850


namespace son_work_rate_l139_139666

theorem son_work_rate (M S : ℝ) (hM : M = 1 / 5) (hMS : M + S = 1 / 4) : 1 / S = 20 :=
by
  sorry

end son_work_rate_l139_139666


namespace grasshoppers_after_transformations_l139_139767

-- Define initial conditions and transformation rules
def initial_crickets : ℕ := 30
def initial_grasshoppers : ℕ := 30

-- Define the transformations
def red_haired_transforms (g : ℕ) (c : ℕ) : ℕ × ℕ :=
  (g - 4, c + 1)

def green_haired_transforms (c : ℕ) (g : ℕ) : ℕ × ℕ :=
  (c - 5, g + 2)

-- Define the total number of transformations and the resulting condition
def total_transformations : ℕ := 18
def final_crickets : ℕ := 0

-- The proof goal
theorem grasshoppers_after_transformations : 
  initial_grasshoppers = 30 → 
  initial_crickets = 30 → 
  (∀ t, t = total_transformations → 
          ∀ g c, 
          (g, c) = (0, 6) → 
          (∃ m n, (m + n = t ∧ final_crickets = c))) →
  final_grasshoppers = 6 :=
by
  sorry

end grasshoppers_after_transformations_l139_139767


namespace factorize_x4_plus_81_l139_139765

noncomputable def factorize_poly (x : ℝ) : (ℝ × ℝ) :=
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  (p, q)

theorem factorize_x4_plus_81 : ∀ x : ℝ, (x^4 + 81) = (factorize_poly x).fst * (factorize_poly x).snd := by
  intro x
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  have h : x^4 + 81 = p * q
  { sorry }
  exact h

end factorize_x4_plus_81_l139_139765


namespace problem1_problem2_problem3_l139_139068

-- Problem (1)
theorem problem1 : -36 * (5 / 4 - 5 / 6 - 11 / 12) = 18 := by
  sorry

-- Problem (2)
theorem problem2 : (-2) ^ 2 - 3 * (-1) ^ 3 + 0 * (-2) ^ 3 = 7 := by
  sorry

-- Problem (3)
theorem problem3 (x : ℚ) (y : ℚ) (h1 : x = -2) (h2 : y = 1 / 2) : 
    (3 / 2) * x^2 * y + x * y^2 = 5 / 2 := by
  sorry

end problem1_problem2_problem3_l139_139068


namespace math_olympiad_scores_l139_139273

theorem math_olympiad_scores (a : Fin 20 → ℕ) 
  (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → j ≠ k → i ≠ k → a i < a j + a k) :
  ∀ i : Fin 20, a i > 18 := 
sorry

end math_olympiad_scores_l139_139273


namespace value_of_x_l139_139976

variable (x y z a b c : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h1 : x * y / (x + y) = a)
variable (h2 : x * z / (x + z) = b)
variable (h3 : y * z / (y + z) = c)

theorem value_of_x : x = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end value_of_x_l139_139976


namespace max_consecutive_interesting_numbers_l139_139508

def is_interesting (n : ℕ) : Prop :=
  (n / 100 % 3 = 0) ∨ (n / 10 % 10 % 3 = 0) ∨ (n % 10 % 3 = 0)

theorem max_consecutive_interesting_numbers :
  ∃ l r, 100 ≤ l ∧ r ≤ 999 ∧ r - l + 1 = 122 ∧ (∀ n, l ≤ n ∧ n ≤ r → is_interesting n) ∧ 
  ∀ l' r', 100 ≤ l' ∧ r' ≤ 999 ∧ r' - l' + 1 > 122 → ∃ n, l' ≤ n ∧ n ≤ r' ∧ ¬ is_interesting n := 
sorry

end max_consecutive_interesting_numbers_l139_139508


namespace specified_time_is_30_total_constuction_cost_is_180000_l139_139319

noncomputable def specified_time (x : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  (teamA_rate + teamB_rate) * 15 + 5 * teamA_rate = 1

theorem specified_time_is_30 : specified_time 30 :=
  by 
    sorry

noncomputable def total_constuction_cost (x : ℕ) (costA : ℕ) (costB : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  let total_time := 1 / (teamA_rate + teamB_rate)
  total_time * (costA + costB)

theorem total_constuction_cost_is_180000 : total_constuction_cost 30 6500 3500 = 180000 :=
  by 
    sorry

end specified_time_is_30_total_constuction_cost_is_180000_l139_139319


namespace quadratic_root_l139_139149

/-- If one root of the quadratic equation x^2 - 2x + n = 0 is 3, then n is -3. -/
theorem quadratic_root (n : ℝ) (h : (3 : ℝ)^2 - 2 * 3 + n = 0) : n = -3 :=
sorry

end quadratic_root_l139_139149


namespace Ivanka_more_months_l139_139429

variable (I : ℕ) (W : ℕ)

theorem Ivanka_more_months (hW : W = 18) (hI_W : I + W = 39) : I - W = 3 :=
by
  sorry

end Ivanka_more_months_l139_139429


namespace roots_inequality_l139_139818

noncomputable def a : ℝ := Real.sqrt 2020

theorem roots_inequality (x1 x2 x3 : ℝ) (h_roots : ∀ x, (a * x^3 - 4040 * x^2 + 4 = 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3))
  (h_inequality: x1 < x2 ∧ x2 < x3) : x2 * (x1 + x3) = 2 :=
sorry

end roots_inequality_l139_139818


namespace sequence_bound_l139_139474

open Real

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ)
  (h₀ : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h₁ : ∀ (i j : ℕ), 0 < i → 0 < j → i ≠ j → abs (a i - a j) ≥ 1 / (i + j)) :
  c ≥ 1 :=
by {
  sorry
}

end sequence_bound_l139_139474


namespace Xiaohong_wins_5_times_l139_139042

theorem Xiaohong_wins_5_times :
  ∃ W L : ℕ, (3 * W - 2 * L = 1) ∧ (W + L = 12) ∧ W = 5 :=
by
  sorry

end Xiaohong_wins_5_times_l139_139042


namespace five_letter_sequences_l139_139185

-- Define the quantities of each vowel.
def quantity_vowel_A : Nat := 3
def quantity_vowel_E : Nat := 4
def quantity_vowel_I : Nat := 5
def quantity_vowel_O : Nat := 6
def quantity_vowel_U : Nat := 7

-- Define the number of choices for each letter in a five-letter sequence.
def choices_per_letter : Nat := 5

-- Define the total number of five-letter sequences.
noncomputable def total_sequences : Nat := choices_per_letter ^ 5

-- Prove that the number of five-letter sequences is 3125.
theorem five_letter_sequences : total_sequences = 3125 :=
by sorry

end five_letter_sequences_l139_139185


namespace total_seeds_in_watermelon_l139_139593

theorem total_seeds_in_watermelon :
  let slices := 40
  let black_seeds_per_slice := 20
  let white_seeds_per_slice := 20
  let total_black_seeds := black_seeds_per_slice * slices
  let total_white_seeds := white_seeds_per_slice * slices
  total_black_seeds + total_white_seeds = 1600 := by
  sorry

end total_seeds_in_watermelon_l139_139593


namespace completing_the_square_x_squared_plus_4x_plus_3_eq_0_l139_139037

theorem completing_the_square_x_squared_plus_4x_plus_3_eq_0 :
  (x : ℝ) → x^2 + 4 * x + 3 = 0 → (x + 2)^2 = 1 :=
by
  intros x h
  -- The actual proof will be provided here
  sorry

end completing_the_square_x_squared_plus_4x_plus_3_eq_0_l139_139037


namespace correct_inequality_l139_139356

theorem correct_inequality (x : ℝ) : (1 / (x^2 + 1)) > (1 / (x^2 + 2)) :=
by {
  -- Lean proof steps would be here, but we will use 'sorry' instead to indicate the proof is omitted.
  sorry
}

end correct_inequality_l139_139356


namespace total_passengers_l139_139477

theorem total_passengers (P : ℕ) 
  (h1 : P = (1/12 : ℚ) * P + (1/4 : ℚ) * P + (1/9 : ℚ) * P + (1/6 : ℚ) * P + 42) :
  P = 108 :=
sorry

end total_passengers_l139_139477


namespace sum_and_divide_repeating_decimals_l139_139889

noncomputable def repeating_decimal_83 : ℚ := 83 / 99
noncomputable def repeating_decimal_18 : ℚ := 18 / 99

theorem sum_and_divide_repeating_decimals :
  (repeating_decimal_83 + repeating_decimal_18) / (1 / 5) = 505 / 99 :=
by
  sorry

end sum_and_divide_repeating_decimals_l139_139889


namespace find_n_l139_139662

theorem find_n (a b c : ℤ) (m n p : ℕ)
  (h1 : a = 3)
  (h2 : b = -7)
  (h3 : c = -6)
  (h4 : m > 0)
  (h5 : n > 0)
  (h6 : p > 0)
  (h7 : Nat.gcd m p = 1)
  (h8 : Nat.gcd m n = 1)
  (h9 : Nat.gcd n p = 1)
  (h10 : ∃ x1 x2 : ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p)
  : n = 121 :=
sorry

end find_n_l139_139662


namespace unique_polynomial_solution_l139_139509

def polynomial_homogeneous_of_degree_n (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (t x y : ℝ), P (t * x) (t * y) = t^n * P x y

def polynomial_symmetric_condition (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y z : ℝ), P (y + z) x + P (z + x) y + P (x + y) z = 0

def polynomial_value_at_point (P : ℝ → ℝ → ℝ) : Prop :=
  P 1 0 = 1

theorem unique_polynomial_solution (P : ℝ → ℝ → ℝ) (n : ℕ) :
  polynomial_homogeneous_of_degree_n P n →
  polynomial_symmetric_condition P →
  polynomial_value_at_point P →
  ∀ x y : ℝ, P x y = (x + y)^n * (x - 2 * y) := 
by
  intros h_deg h_symm h_value x y
  sorry

end unique_polynomial_solution_l139_139509


namespace expected_value_of_boy_girl_pairs_l139_139331

noncomputable def expected_value_of_T (boys girls : ℕ) : ℚ :=
  24 * ((boys / 24) * (girls / 23) + (girls / 24) * (boys / 23))

theorem expected_value_of_boy_girl_pairs (boys girls : ℕ) (h_boys : boys = 10) (h_girls : girls = 14) :
  expected_value_of_T boys girls = 12 :=
by
  rw [h_boys, h_girls]
  norm_num
  sorry

end expected_value_of_boy_girl_pairs_l139_139331


namespace square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l139_139565

theorem square_roots_of_four_ninths : {x : ℚ | x ^ 2 = 4 / 9} = {2 / 3, -2 / 3} :=
by
  sorry

theorem cube_root_of_neg_sixty_four : {y : ℚ | y ^ 3 = -64} = {-4} :=
by
  sorry

end square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l139_139565


namespace find_a_and_b_l139_139835

theorem find_a_and_b (a b : ℝ) :
  {-1, 3} = {x : ℝ | x^2 + a * x + b = 0} ↔ a = -2 ∧ b = -3 :=
by 
  sorry

end find_a_and_b_l139_139835


namespace find_first_discount_l139_139199

-- Definitions for the given conditions
def list_price : ℝ := 150
def final_price : ℝ := 105
def second_discount : ℝ := 12.5

-- Statement representing the mathematical proof problem
theorem find_first_discount (x : ℝ) : 
  list_price * ((100 - x) / 100) * ((100 - second_discount) / 100) = final_price → x = 20 :=
by
  sorry

end find_first_discount_l139_139199


namespace total_capacity_of_two_tanks_l139_139859

-- Conditions
def tank_A_initial_fullness : ℚ := 3 / 4
def tank_A_final_fullness : ℚ := 7 / 8
def tank_A_added_volume : ℚ := 5

def tank_B_initial_fullness : ℚ := 2 / 3
def tank_B_final_fullness : ℚ := 5 / 6
def tank_B_added_volume : ℚ := 3

-- Proof statement
theorem total_capacity_of_two_tanks :
  let tank_A_total_capacity := tank_A_added_volume / (tank_A_final_fullness - tank_A_initial_fullness)
  let tank_B_total_capacity := tank_B_added_volume / (tank_B_final_fullness - tank_B_initial_fullness)
  tank_A_total_capacity + tank_B_total_capacity = 58 := 
sorry

end total_capacity_of_two_tanks_l139_139859


namespace choir_members_count_l139_139965

theorem choir_members_count (n : ℕ) 
  (h1 : 150 < n) 
  (h2 : n < 300) 
  (h3 : n % 6 = 1) 
  (h4 : n % 8 = 3) 
  (h5 : n % 9 = 2) : 
  n = 163 :=
sorry

end choir_members_count_l139_139965


namespace a_7_is_4_l139_139551

-- Define the geometric sequence and its properties
variable {a : ℕ → ℝ}

-- Given conditions
axiom pos_seq : ∀ n, a n > 0
axiom geom_seq : ∀ n m, a (n + m) = a n * a m
axiom specific_condition : a 3 * a 11 = 16

theorem a_7_is_4 : a 7 = 4 :=
by
  sorry

end a_7_is_4_l139_139551


namespace sin_value_l139_139794

theorem sin_value (theta : ℝ) (h : Real.cos (3 * Real.pi / 14 - theta) = 1 / 3) : 
  Real.sin (2 * Real.pi / 7 + theta) = 1 / 3 :=
by
  -- Sorry replaces the actual proof which is not required for this task
  sorry

end sin_value_l139_139794


namespace juan_faster_than_peter_l139_139313

theorem juan_faster_than_peter (J : ℝ) :
  (Peter_speed : ℝ) = 5.0 →
  (time : ℝ) = 1.5 →
  (distance_apart : ℝ) = 19.5 →
  (J + 5.0) * time = distance_apart →
  J - 5.0 = 3 := 
by
  intros Peter_speed_eq time_eq distance_apart_eq relative_speed_eq
  sorry

end juan_faster_than_peter_l139_139313


namespace gcd_lcm_product_360_l139_139911

theorem gcd_lcm_product_360 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
    {d : ℕ | d = Nat.gcd a b } =
    {1, 2, 4, 8, 3, 6, 12, 24} := 
by
  sorry

end gcd_lcm_product_360_l139_139911


namespace ratio_of_heights_l139_139779

-- Define the height of the first rocket.
def H1 : ℝ := 500

-- Define the combined height of the two rockets.
def combined_height : ℝ := 1500

-- Define the height of the second rocket.
def H2 : ℝ := combined_height - H1

-- The statement to be proven.
theorem ratio_of_heights : H2 / H1 = 2 := by
  -- Proof goes here
  sorry

end ratio_of_heights_l139_139779


namespace stripes_distance_l139_139915

theorem stripes_distance (d : ℝ) (L : ℝ) (c : ℝ) (y : ℝ) 
  (hd : d = 40) (hL : L = 50) (hc : c = 15)
  (h_ratio : y / d = c / L) : y = 12 :=
by
  rw [hd, hL, hc] at h_ratio
  sorry

end stripes_distance_l139_139915


namespace solve_for_x_l139_139001

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 2) : x = 23 / 10 :=
by
  -- Proof is omitted
  sorry

end solve_for_x_l139_139001


namespace proof_subset_l139_139082

def set_A := {x : ℝ | x ≥ 0}

theorem proof_subset (B : Set ℝ) (h : set_A ∪ B = B) : set_A ⊆ B := 
by
  sorry

end proof_subset_l139_139082


namespace sum_of_slopes_correct_l139_139906

noncomputable def sum_of_slopes : ℚ :=
  let Γ1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}
  let Γ2 := {p : ℝ × ℝ | (p.1 - 10)^2 + (p.2 - 11)^2 = 1}
  let l := {k : ℝ | ∃ p1 ∈ Γ1, ∃ p2 ∈ Γ1, ∃ p3 ∈ Γ2, ∃ p4 ∈ Γ2, p1 ≠ p2 ∧ p3 ≠ p4 ∧ p1.2 = k * p1.1 ∧ p3.2 = k * p3.1}
  let valid_slopes := {k | k ∈ l ∧ (k = 11/10 ∨ k = 1 ∨ k = 5/4)}
  (11 / 10) + 1 + (5 / 4)

theorem sum_of_slopes_correct : sum_of_slopes = 67 / 20 := 
  by sorry

end sum_of_slopes_correct_l139_139906


namespace certain_number_105_l139_139349

theorem certain_number_105 (a x : ℕ) (h0 : a = 105) (h1 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end certain_number_105_l139_139349


namespace John_finishes_at_610PM_l139_139670

def TaskTime : Nat := 55
def StartTime : Nat := 14 * 60 + 30 -- 2:30 PM in minutes
def EndSecondTask : Nat := 16 * 60 + 20 -- 4:20 PM in minutes

theorem John_finishes_at_610PM (h1 : TaskTime * 2 = EndSecondTask - StartTime) : 
  (EndSecondTask + TaskTime * 2) = (18 * 60 + 10) :=
by
  sorry

end John_finishes_at_610PM_l139_139670


namespace rectangle_area_l139_139428

theorem rectangle_area (a b : ℝ) (x : ℝ) 
  (h1 : x^2 + (x / 2)^2 = (a + b)^2) 
  (h2 : x > 0) : 
  x * (x / 2) = (2 * (a + b)^2) / 5 := 
by 
  sorry

end rectangle_area_l139_139428


namespace probability_red_bean_l139_139989

section ProbabilityRedBean

-- Initially, there are 5 red beans and 9 black beans in a bag.
def initial_red_beans : ℕ := 5
def initial_black_beans : ℕ := 9
def initial_total_beans : ℕ := initial_red_beans + initial_black_beans

-- Then, 3 red beans and 3 black beans are added to the bag.
def added_red_beans : ℕ := 3
def added_black_beans : ℕ := 3
def final_red_beans : ℕ := initial_red_beans + added_red_beans
def final_black_beans : ℕ := initial_black_beans + added_black_beans
def final_total_beans : ℕ := final_red_beans + final_black_beans

-- The probability of drawing a red bean should be 2/5
theorem probability_red_bean :
  (final_red_beans : ℚ) / final_total_beans = 2 / 5 := by
  sorry

end ProbabilityRedBean

end probability_red_bean_l139_139989


namespace incorrect_proposition_l139_139308

-- Variables and conditions
variable (p q : Prop)
variable (m x a b c : ℝ)
variable (hreal : 1 + 4 * m ≥ 0)

-- Theorem statement
theorem incorrect_proposition :
  ¬ (∀ m > 0, (∃ x : ℝ, x^2 + x - m = 0) → m > 0) :=
sorry

end incorrect_proposition_l139_139308


namespace geometric_sequence_first_term_l139_139538

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r = 5) 
  (h2 : a * r^3 = 45) : 
  a = 5 / (3^(2/3)) := 
by
  -- proof steps to be filled here
  sorry

end geometric_sequence_first_term_l139_139538


namespace work_completion_time_l139_139582

theorem work_completion_time (A_rate B_rate : ℝ) (hA : A_rate = 1/60) (hB : B_rate = 1/20) :
  1 / (A_rate + B_rate) = 15 :=
by
  sorry

end work_completion_time_l139_139582


namespace problem1_problem2_l139_139750

theorem problem1 : 24 - (-16) + (-25) - 15 = 0 :=
by
  sorry

theorem problem2 : (-81) + 2 * (1 / 4) * (4 / 9) / (-16) = -81 - (1 / 16) :=
by
  sorry

end problem1_problem2_l139_139750


namespace max_range_walk_min_range_walk_count_max_range_sequences_l139_139182

variable {a b : ℕ}

-- Condition: a > b
def valid_walk (a b : ℕ) : Prop := a > b

-- Proof that the maximum possible range of the walk is a
theorem max_range_walk (h : valid_walk a b) : 
  (a + b) = a + b := sorry

-- Proof that the minimum possible range of the walk is a - b
theorem min_range_walk (h : valid_walk a b) : 
  (a - b) = a - b := sorry

-- Proof that the number of different sequences with the maximum possible range is b + 1
theorem count_max_range_sequences (h : valid_walk a b) : 
  b + 1 = b + 1 := sorry

end max_range_walk_min_range_walk_count_max_range_sequences_l139_139182


namespace choose_three_cards_of_different_suits_l139_139220

/-- The number of ways to choose 3 cards from a standard deck of 52 cards,
if all three cards must be of different suits -/
theorem choose_three_cards_of_different_suits :
  let n := 4
  let r := 3
  let suits_combinations := Nat.choose n r
  let cards_per_suit := 13
  let total_ways := suits_combinations * (cards_per_suit ^ r)
  total_ways = 8788 :=
by
  sorry

end choose_three_cards_of_different_suits_l139_139220


namespace train_length_l139_139426

theorem train_length
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (distance_covered : ℝ)
  (train_length : ℝ) :
  speed_kmph = 72 →
  platform_length = 240 →
  crossing_time = 26 →
  conversion_factor = 5 / 18 →
  speed_mps = speed_kmph * conversion_factor →
  distance_covered = speed_mps * crossing_time →
  train_length = distance_covered - platform_length →
  train_length = 280 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end train_length_l139_139426


namespace find_circle_center_value_x_plus_y_l139_139544

theorem find_circle_center_value_x_plus_y : 
  ∀ (x y : ℝ), (x^2 + y^2 = 4 * x - 6 * y + 9) → 
    x + y = -1 :=
by
  intros x y h
  sorry

end find_circle_center_value_x_plus_y_l139_139544


namespace units_digit_7_pow_5_l139_139256

theorem units_digit_7_pow_5 : (7^5) % 10 = 7 := 
by
  sorry

end units_digit_7_pow_5_l139_139256


namespace marion_paperclips_correct_l139_139371

def yun_initial_paperclips := 30
def yun_remaining_paperclips (x : ℕ) : ℕ := (2 * x) / 5
def marion_paperclips (x y : ℕ) : ℕ := (4 * (yun_remaining_paperclips x)) / 3 + y
def y := 7

theorem marion_paperclips_correct : marion_paperclips yun_initial_paperclips y = 23 := by
  sorry

end marion_paperclips_correct_l139_139371


namespace proportion_difference_l139_139397

theorem proportion_difference : (0.80 * 40) - ((4 / 5) * 20) = 16 := 
by 
  sorry

end proportion_difference_l139_139397


namespace find_b_l139_139196

-- Definitions for the conditions
variables (a b c d : ℝ)
def four_segments_proportional := a / b = c / d

theorem find_b (h1: a = 3) (h2: d = 4) (h3: c = 6) (h4: four_segments_proportional a b c d) : b = 2 :=
by
  sorry

end find_b_l139_139196


namespace parabola_directrix_l139_139952

theorem parabola_directrix (x y : ℝ) (h : x^2 = 2 * y) : y = -1 / 2 := 
  sorry

end parabola_directrix_l139_139952


namespace inradius_length_l139_139705

noncomputable def inradius (BC AB AC IC : ℝ) (r : ℝ) : Prop :=
  ∀ (r : ℝ), ((BC = 40) ∧ (AB = AC) ∧ (IC = 24)) →
    r = 4 * Real.sqrt 11

theorem inradius_length (BC AB AC IC : ℝ) (r : ℝ) :
  (BC = 40) ∧ (AB = AC) ∧ (IC = 24) →
  r = 4 * Real.sqrt 11 := 
by
  sorry

end inradius_length_l139_139705


namespace map_length_to_reality_l139_139008

def scale : ℝ := 500
def length_map : ℝ := 7.2
def length_actual : ℝ := 3600

theorem map_length_to_reality : length_actual = length_map * scale :=
by
  sorry

end map_length_to_reality_l139_139008


namespace minimum_sum_l139_139713

theorem minimum_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) + ((a^2 * b) / (18 * b * c)) ≥ 4 / 9 :=
sorry

end minimum_sum_l139_139713


namespace original_number_of_people_l139_139005

theorem original_number_of_people (x : ℕ) (h1 : x - x / 3 + (x / 3) * 3/4 = x * 1/4 + 15) : x = 30 :=
sorry

end original_number_of_people_l139_139005


namespace candy_bars_per_bag_l139_139414

def total_candy_bars : ℕ := 15
def number_of_bags : ℕ := 5

theorem candy_bars_per_bag : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end candy_bars_per_bag_l139_139414


namespace general_solution_of_diff_eq_l139_139536

theorem general_solution_of_diff_eq
  (f : ℝ → ℝ → ℝ)
  (D : Set (ℝ × ℝ))
  (hf : ∀ x y, f x y = x)
  (hD : D = Set.univ) :
  ∃ C : ℝ, ∀ x : ℝ, ∃ y : ℝ, y = (x^2) / 2 + C :=
by
  sorry

end general_solution_of_diff_eq_l139_139536


namespace trajectory_of_center_l139_139606

-- Define a structure for Point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the given point A
def A : Point := { x := -2, y := 0 }

-- Define a property for the circle being tangent to a line
def tangent_to_line (center : Point) (line_x : ℝ) : Prop :=
  center.x + line_x = 0

-- The main theorem to be proved
theorem trajectory_of_center :
  ∀ (C : Point), tangent_to_line C 2 → (C.y)^2 = -8 * C.x :=
sorry

end trajectory_of_center_l139_139606


namespace square_of_1024_l139_139295

theorem square_of_1024 : 1024^2 = 1048576 :=
by
  sorry

end square_of_1024_l139_139295


namespace circle_equation_bisected_and_tangent_l139_139130

theorem circle_equation_bisected_and_tangent :
  (∃ x0 y0 r : ℝ, x0 = y0 ∧ (x0 + y0 - 2 * r) = 0 ∧ (∀ x y : ℝ, (x - x0)^2 + (y - y0)^2 = r^2 → (x - 1)^2 + (y - 1)^2 = 2)) := sorry

end circle_equation_bisected_and_tangent_l139_139130


namespace problem1_problem2_l139_139157

-- Step 1
theorem problem1 (a b c A B C : ℝ) (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a^2 = b^2 + c^2 := sorry

-- Step 2
theorem problem2 (a b c : ℝ) (h_a : a = 5) (h_cosA : Real.cos A = 25 / 31) 
  (h_conditions : 2 * a^2 = b^2 + c^2 ∧ 2 * b * c = a^2 / Real.cos A) :
  a + b + c = 14 := sorry

end problem1_problem2_l139_139157


namespace cakes_in_november_l139_139547

-- Define the function modeling the number of cakes baked each month
def num_of_cakes (initial: ℕ) (n: ℕ) := initial + 2 * n

-- Given conditions
def cakes_in_october := 19
def cakes_in_december := 23
def cakes_in_january := 25
def cakes_in_february := 27
def monthly_increase := 2

-- Prove that the number of cakes baked in November is 21
theorem cakes_in_november : num_of_cakes cakes_in_october 1 = 21 :=
by
  sorry

end cakes_in_november_l139_139547


namespace ratio_of_length_to_height_l139_139732

theorem ratio_of_length_to_height
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (vol_eq : 129024 = w * h * l)
  (w_eq : w = 8) :
  l / h = 7 := 
sorry

end ratio_of_length_to_height_l139_139732


namespace simplify_expression_l139_139143

theorem simplify_expression :
  (∃ (a b c d e f : ℝ), 
    a = (7)^(1/4) ∧ 
    b = (3)^(1/3) ∧ 
    c = (7)^(1/2) ∧ 
    d = (3)^(1/6) ∧ 
    e = (a / b) / (c / d) ∧ 
    f = ((1 / 7)^(1/4)) * ((1 / 3)^(1/6))
    → e = f) :=
by {
  sorry
}

end simplify_expression_l139_139143


namespace total_blocks_in_pyramid_l139_139865

-- Define the number of blocks in each layer
def blocks_in_layer (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => 3 * blocks_in_layer n

-- Prove the total number of blocks in the four-layer pyramid
theorem total_blocks_in_pyramid : 
  (blocks_in_layer 0) + (blocks_in_layer 1) + (blocks_in_layer 2) + (blocks_in_layer 3) = 40 :=
by
  sorry

end total_blocks_in_pyramid_l139_139865


namespace g_10_44_l139_139132

def g (x y : ℕ) : ℕ := sorry

axiom g_cond1 (x : ℕ) : g x x = x ^ 2
axiom g_cond2 (x y : ℕ) : g x y = g y x
axiom g_cond3 (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_10_44 : g 10 44 = 440 := sorry

end g_10_44_l139_139132


namespace radius_of_circumcircle_l139_139292

-- Definitions of sides of a triangle and its area
variables {a b c t : ℝ}

-- Condition that t is the area of a triangle with sides a, b, and c
def is_triangle_area (a b c t : ℝ) : Prop := -- Placeholder condition stating these values form a triangle
sorry

-- Statement to prove the given radius formula for the circumscribed circle
theorem radius_of_circumcircle (h : is_triangle_area a b c t) : 
  ∃ r : ℝ, r = abc / (4 * t) :=
sorry

end radius_of_circumcircle_l139_139292


namespace find_m_plus_n_l139_139636

-- Define the sets and variables
def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}
def K (n : ℝ) : Set ℝ := {x | 3 < x ∧ x < n}

theorem find_m_plus_n (m n : ℝ) 
  (hM: M = {x | 0 < x ∧ x < 4})
  (hK_true: K n = M ∩ N m) :
  m + n = 7 := 
  sorry

end find_m_plus_n_l139_139636


namespace simplify_fraction_l139_139314

theorem simplify_fraction : 1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 :=
by
sorry

end simplify_fraction_l139_139314


namespace magician_can_always_determine_hidden_pair_l139_139501

-- Define the cards as an enumeration
inductive Card
| one | two | three | four | five

-- Define a pair of cards
structure CardPair where
  first : Card
  second : Card

-- Define the function the magician uses to decode the hidden pair 
-- based on the two cards the assistant points out, encoded as a pentagon
noncomputable def magician_decodes (assistant_cards spectator_announced: CardPair) : CardPair := sorry

-- Theorem statement: given the conditions, the magician can always determine the hidden pair.
theorem magician_can_always_determine_hidden_pair 
  (hidden_cards assistant_cards spectator_announced : CardPair)
  (assistant_strategy : CardPair → CardPair)
  (h : assistant_strategy assistant_cards = spectator_announced)
  : magician_decodes assistant_cards spectator_announced = hidden_cards := sorry

end magician_can_always_determine_hidden_pair_l139_139501


namespace gcd_mn_eq_one_l139_139424

def m : ℤ := 123^2 + 235^2 - 347^2
def n : ℤ := 122^2 + 234^2 - 348^2

theorem gcd_mn_eq_one : Int.gcd m n = 1 := 
by
  sorry

end gcd_mn_eq_one_l139_139424


namespace triangle_iff_inequality_l139_139939

variable {a b c : ℝ}

theorem triangle_iff_inequality :
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) := sorry

end triangle_iff_inequality_l139_139939


namespace expand_expression_l139_139533

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := 
by 
  sorry

end expand_expression_l139_139533


namespace find_number_of_breeding_rabbits_l139_139964

def breeding_rabbits_condition (B : ℕ) : Prop :=
  ∃ (kittens_first_spring remaining_kittens_first_spring kittens_second_spring remaining_kittens_second_spring : ℕ),
    kittens_first_spring = 10 * B ∧
    remaining_kittens_first_spring = 5 * B + 5 ∧
    kittens_second_spring = 60 ∧
    remaining_kittens_second_spring = kittens_second_spring - 4 ∧
    B + remaining_kittens_first_spring + remaining_kittens_second_spring = 121

theorem find_number_of_breeding_rabbits (B : ℕ) : breeding_rabbits_condition B → B = 10 :=
by
  sorry

end find_number_of_breeding_rabbits_l139_139964


namespace find_term_number_l139_139432

variable {α : ℝ} (b : ℕ → ℝ) (q : ℝ)

namespace GeometricProgression

noncomputable def geometric_progression (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ (n : ℕ), b (n + 1) = b n * q

noncomputable def satisfies_conditions (α : ℝ) (b : ℕ → ℝ) : Prop :=
  b 25 = 2 * Real.tan α ∧ b 31 = 2 * Real.sin α

theorem find_term_number (α : ℝ) (b : ℕ → ℝ) (q : ℝ) (hb : geometric_progression b q) (hc : satisfies_conditions α b) :
  ∃ n, b n = Real.sin (2 * α) ∧ n = 37 :=
sorry

end GeometricProgression

end find_term_number_l139_139432


namespace sequence_4951_l139_139836

theorem sequence_4951 :
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = a n + n) ∧ a 100 = 4951) :=
sorry

end sequence_4951_l139_139836


namespace evaluate_expression_l139_139715

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = 2) : 
  (x^3 * y^4 * z)^2 = 1 / 104976 :=
by 
  sorry

end evaluate_expression_l139_139715


namespace ratio_of_average_speeds_l139_139233

theorem ratio_of_average_speeds
    (time_eddy : ℝ) (distance_eddy : ℝ)
    (time_freddy : ℝ) (distance_freddy : ℝ) :
  time_eddy = 3 ∧ distance_eddy = 600 ∧ time_freddy = 4 ∧ distance_freddy = 460 →
  (distance_eddy / time_eddy) / (distance_freddy / time_freddy) = 200 / 115 :=
by
  sorry

end ratio_of_average_speeds_l139_139233


namespace amount_spent_on_marbles_l139_139059

/-- A theorem to determine the amount Mike spent on marbles. -/
theorem amount_spent_on_marbles 
  (total_amount : ℝ) 
  (cost_football : ℝ) 
  (cost_baseball : ℝ) 
  (total_amount_eq : total_amount = 20.52)
  (cost_football_eq : cost_football = 4.95)
  (cost_baseball_eq : cost_baseball = 6.52) :
  ∃ (cost_marbles : ℝ), cost_marbles = total_amount - (cost_football + cost_baseball) 
  ∧ cost_marbles = 9.05 := 
by
  sorry

end amount_spent_on_marbles_l139_139059


namespace y_coords_diff_of_ellipse_incircle_area_l139_139223

theorem y_coords_diff_of_ellipse_incircle_area
  (x1 y1 x2 y2 : ℝ)
  (F1 F2 : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : F1 = (-4, 0))
  (h4 : F2 = (4, 0))
  (h5 : 4 * (|y1 - y2|) = 20)
  (h6 : ∃ (x : ℝ), (x / 25)^2 + (y1 / 9)^2 = 1 ∧ (x / 25)^2 + (y2 / 9)^2 = 1) :
  |y1 - y2| = 5 :=
sorry

end y_coords_diff_of_ellipse_incircle_area_l139_139223


namespace find_lunch_break_duration_l139_139035

def lunch_break_duration : ℝ → ℝ → ℝ → ℝ
  | s, a, L => L

theorem find_lunch_break_duration (s a L : ℝ) :
  (8 - L) * (s + a) = 0.6 ∧ (6.4 - L) * a = 0.28 ∧ (9.6 - L) * s = 0.12 →
  lunch_break_duration s a L = 1 :=
  by
    sorry

end find_lunch_break_duration_l139_139035


namespace complex_expression_equals_neg3_l139_139632

noncomputable def nonreal_root_of_x4_eq_1 : Type :=
{ζ : ℂ // ζ^4 = 1 ∧ ζ.im ≠ 0}

theorem complex_expression_equals_neg3 (ζ : nonreal_root_of_x4_eq_1) :
  (1 - ζ.val + ζ.val^3)^4 + (1 + ζ.val^2 - ζ.val^3)^4 = -3 :=
sorry

end complex_expression_equals_neg3_l139_139632


namespace compute_expression_l139_139640

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 :=
by
  sorry

end compute_expression_l139_139640


namespace cylinder_surface_area_l139_139144

/-- A right cylinder with radius 3 inches and height twice the radius has a total surface area of 54π square inches. -/
theorem cylinder_surface_area (r : ℝ) (h : ℝ) (A_total : ℝ) (π : ℝ) : r = 3 → h = 2 * r → π = Real.pi → A_total = 54 * π :=
by
  sorry

end cylinder_surface_area_l139_139144


namespace fraction_paint_remaining_l139_139383

theorem fraction_paint_remaining :
  let original_paint := 1
  let first_day_usage := original_paint / 4
  let paint_remaining_after_first_day := original_paint - first_day_usage
  let second_day_usage := paint_remaining_after_first_day / 2
  let paint_remaining_after_second_day := paint_remaining_after_first_day - second_day_usage
  let third_day_usage := paint_remaining_after_second_day / 3
  let paint_remaining_after_third_day := paint_remaining_after_second_day - third_day_usage
  paint_remaining_after_third_day = original_paint / 4 := 
by
  sorry

end fraction_paint_remaining_l139_139383


namespace equation_has_at_most_one_real_root_l139_139445

def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x

theorem equation_has_at_most_one_real_root (f : ℝ → ℝ) (a : ℝ) (h : has_inverse f) :
  ∀ x1 x2 : ℝ, f x1 = a ∧ f x2 = a → x1 = x2 :=
by sorry

end equation_has_at_most_one_real_root_l139_139445


namespace original_price_color_TV_l139_139612

theorem original_price_color_TV (x : ℝ) 
  (h : 1.12 * x - x = 144) : 
  x = 1200 :=
sorry

end original_price_color_TV_l139_139612


namespace fixed_point_inequality_l139_139083

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * a^((x + 1) / 2) - 4

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = -1 :=
sorry

theorem inequality (a : ℝ) (x : ℝ) (h : a > 1) :
  f a (x - 3 / 4) ≥ 3 / (a^(x^2 / 2)) - 4 :=
sorry

end fixed_point_inequality_l139_139083


namespace total_expenditure_is_3000_l139_139335

/-- Define the Hall dimensions -/
def length : ℝ := 20
def width : ℝ := 15
def cost_per_square_meter : ℝ := 10

/-- Statement to prove --/
theorem total_expenditure_is_3000 
  (h_length : length = 20)
  (h_width : width = 15)
  (h_cost : cost_per_square_meter = 10) : 
  length * width * cost_per_square_meter = 3000 :=
sorry

end total_expenditure_is_3000_l139_139335


namespace find_numbers_l139_139325

theorem find_numbers (A B: ℕ) (h1: A + B = 581) (h2: (Nat.lcm A B) / (Nat.gcd A B) = 240) : 
  (A = 560 ∧ B = 21) ∨ (A = 21 ∧ B = 560) :=
by
  sorry

end find_numbers_l139_139325


namespace largest_k_divides_2_pow_3_pow_m_add_1_l139_139146

theorem largest_k_divides_2_pow_3_pow_m_add_1 (m : ℕ) : 9 ∣ 2^(3^m) + 1 := sorry

end largest_k_divides_2_pow_3_pow_m_add_1_l139_139146


namespace jenny_ran_further_l139_139578

-- Define the distances Jenny ran and walked
def ran_distance : ℝ := 0.6
def walked_distance : ℝ := 0.4

-- Define the difference between the distances Jenny ran and walked
def difference : ℝ := ran_distance - walked_distance

-- The proof statement
theorem jenny_ran_further : difference = 0.2 := by
  sorry

end jenny_ran_further_l139_139578


namespace roots_of_poly_l139_139338

theorem roots_of_poly (a b c : ℂ) :
  ∀ x, x = a ∨ x = b ∨ x = c → x^4 - a*x^3 - b*x + c = 0 :=
sorry

end roots_of_poly_l139_139338


namespace arithmetic_sequence_common_difference_l139_139761

variable {α : Type*} [AddGroup α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n, a (n + 1) = a n + (a 2 - a 1)

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 2)
  (h_a3 : a 3 = -4) :
  a 3 - a 2 = -6 := 
sorry

end arithmetic_sequence_common_difference_l139_139761


namespace max_value_of_z_l139_139500

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y ≤ 2) (h2 : x + y ≥ 0) (h3 : x ≤ 4) : 
  ∃ (z : ℝ), z = 2 * x + y ∧ z ≤ 11 :=
by
  sorry

end max_value_of_z_l139_139500


namespace evaluate_expression_l139_139598

theorem evaluate_expression : 
  60 + 120 / 15 + 25 * 16 - 220 - 420 / 7 + 3 ^ 2 = 197 :=
by
  sorry

end evaluate_expression_l139_139598


namespace simplify_expression_l139_139237

theorem simplify_expression (x y : ℤ) (h₁ : x = 2) (h₂ : y = -3) :
  ((2 * x - y) ^ 2 - (x - y) * (x + y) - 2 * y ^ 2) / x = 18 :=
by
  sorry

end simplify_expression_l139_139237


namespace mutually_exclusive_event_l139_139633

-- Define the events
def hits_first_shot : Prop := sorry  -- Placeholder for "hits the target on the first shot"
def hits_second_shot : Prop := sorry  -- Placeholder for "hits the target on the second shot"
def misses_first_shot : Prop := ¬ hits_first_shot
def misses_second_shot : Prop := ¬ hits_second_shot

-- Define the main events in the problem
def hitting_at_least_once : Prop := hits_first_shot ∨ hits_second_shot
def missing_both_times : Prop := misses_first_shot ∧ misses_second_shot

-- Statement of the theorem
theorem mutually_exclusive_event :
  missing_both_times ↔ ¬ hitting_at_least_once :=
by sorry

end mutually_exclusive_event_l139_139633


namespace length_of_second_edge_l139_139887

-- Define the edge lengths and volume
def edge1 : ℕ := 6
def edge3 : ℕ := 6
def volume : ℕ := 180

-- The theorem to state the length of the second edge
theorem length_of_second_edge (edge2 : ℕ) (h : edge1 * edge2 * edge3 = volume) :
  edge2 = 5 :=
by
  -- Skipping the proof
  sorry

end length_of_second_edge_l139_139887


namespace trapezoid_area_is_correct_l139_139268

def square_side_lengths : List ℕ := [1, 3, 5, 7]
def total_base_length : ℕ := square_side_lengths.sum
def tallest_square_height : ℕ := 7

noncomputable def trapezoid_area_between_segment_and_base : ℚ :=
  let height_at_x (x : ℚ) : ℚ := x * (7/16)
  let base_1 := 4
  let base_2 := 9
  let height_1 := height_at_x base_1
  let height_2 := height_at_x base_2
  ((height_1 + height_2) * (base_2 - base_1) / 2)

theorem trapezoid_area_is_correct :
  trapezoid_area_between_segment_and_base = 14.21875 :=
sorry

end trapezoid_area_is_correct_l139_139268


namespace set_list_method_l139_139057

theorem set_list_method : 
  {x : ℝ | x^2 - 2 * x + 1 = 0} = {1} :=
sorry

end set_list_method_l139_139057


namespace cash_sales_is_48_l139_139436

variable (total_sales : ℝ) (credit_fraction : ℝ) (cash_sales : ℝ)

-- Conditions: Total sales were $80, 2/5 of the total sales were credit sales
def problem_conditions := total_sales = 80 ∧ credit_fraction = 2/5 ∧ cash_sales = (1 - credit_fraction) * total_sales

-- Question: Prove that the amount of cash sales Mr. Brandon made is $48.
theorem cash_sales_is_48 (h : problem_conditions total_sales credit_fraction cash_sales) : 
  cash_sales = 48 :=
by
  sorry

end cash_sales_is_48_l139_139436


namespace smallest_number_l139_139883

theorem smallest_number (a b c d e : ℕ) (h₁ : a = 12) (h₂ : b = 16) (h₃ : c = 18) (h₄ : d = 21) (h₅ : e = 28) : 
    ∃ n : ℕ, (n - 4) % Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 0 ∧ n = 1012 :=
by
    sorry

end smallest_number_l139_139883


namespace smallest_n_mult_y_perfect_cube_l139_139607

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9

theorem smallest_n_mult_y_perfect_cube : ∃ n : ℕ, (∀ m : ℕ, y * n = m^3 → n = 1500) :=
sorry

end smallest_n_mult_y_perfect_cube_l139_139607


namespace todd_money_left_l139_139218

def candy_bar_cost : ℝ := 2.50
def chewing_gum_cost : ℝ := 1.50
def soda_cost : ℝ := 3
def discount : ℝ := 0.20
def initial_money : ℝ := 50
def number_of_candy_bars : ℕ := 7
def number_of_chewing_gum : ℕ := 5
def number_of_soda : ℕ := 3

noncomputable def total_candy_bar_cost : ℝ := number_of_candy_bars * candy_bar_cost
noncomputable def total_chewing_gum_cost : ℝ := number_of_chewing_gum * chewing_gum_cost
noncomputable def total_soda_cost : ℝ := number_of_soda * soda_cost
noncomputable def discount_amount : ℝ := total_soda_cost * discount
noncomputable def discounted_soda_cost : ℝ := total_soda_cost - discount_amount
noncomputable def total_cost : ℝ := total_candy_bar_cost + total_chewing_gum_cost + discounted_soda_cost
noncomputable def money_left : ℝ := initial_money - total_cost

theorem todd_money_left : money_left = 17.80 :=
by sorry

end todd_money_left_l139_139218


namespace edge_ratio_of_cubes_l139_139881

theorem edge_ratio_of_cubes (a b : ℝ) (h : a^3 / b^3 = 27 / 8) : a / b = 3 / 2 :=
by
  sorry

end edge_ratio_of_cubes_l139_139881


namespace vector_calculation_l139_139901

namespace VectorProof

variables (a b : ℝ × ℝ) (m : ℝ)

def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k • v2)

theorem vector_calculation
  (h₁ : a = (1, -2))
  (h₂ : b = (m, 4))
  (h₃ : parallel a b) :
  2 • a - b = (4, -8) :=
sorry

end VectorProof

end vector_calculation_l139_139901


namespace custom_operation_difference_correct_l139_139786

def custom_operation (x y : ℕ) : ℕ := x * y + 2 * x

theorem custom_operation_difference_correct :
  custom_operation 5 3 - custom_operation 3 5 = 4 :=
by
  sorry

end custom_operation_difference_correct_l139_139786


namespace find_a_l139_139686

theorem find_a (a x y : ℝ) (h1 : a^(3*x - 1) * 3^(4*y - 3) = 49^x * 27^y) (h2 : x + y = 4) : a = 7 := by
  sorry

end find_a_l139_139686


namespace minimum_distinct_lines_l139_139473

theorem minimum_distinct_lines (n : ℕ) (h : n = 31) : 
  ∃ (k : ℕ), k = 9 :=
by
  sorry

end minimum_distinct_lines_l139_139473


namespace find_b_l139_139172

theorem find_b (a b : ℝ) (h_inv_var : a^2 * Real.sqrt b = k) (h_ab : a * b = 72) (ha3 : a = 3) (hb64 : b = 64) : b = 18 :=
sorry

end find_b_l139_139172


namespace chord_length_l139_139711

-- Define the key components.
structure Circle := 
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the initial conditions.
def circle1 : Circle := { center := (0, 0), radius := 5 }
def circle2 : Circle := { center := (2, 0), radius := 3 }

-- Define the chord and tangency condition.
def touches_internally (C1 C2 : Circle) : Prop :=
  C1.radius > C2.radius ∧ dist C1.center C2.center = C1.radius - C2.radius

def chord_divided_ratio (AB_length : ℝ) (r1 r2 : ℝ) : Prop :=
  ∃ (x : ℝ), AB_length = 4 * x ∧ r1 = x ∧ r2 = 3 * x

-- The theorem to prove the length of the chord AB.
theorem chord_length (h1 : touches_internally circle1 circle2)
                     (h2 : chord_divided_ratio 8 2 (6)) : ∃ (AB_length : ℝ), AB_length = 8 :=
by
  sorry

end chord_length_l139_139711


namespace units_digit_47_power_47_l139_139605

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end units_digit_47_power_47_l139_139605


namespace possible_values_of_Q_l139_139826

theorem possible_values_of_Q (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ∃ Q : ℝ, Q = 8 ∨ Q = -1 := 
sorry

end possible_values_of_Q_l139_139826


namespace binom_20_19_eq_20_l139_139782

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_20_19_eq_20 : binom 20 19 = 20 := by
  sorry

end binom_20_19_eq_20_l139_139782


namespace largest_sum_valid_set_l139_139736

-- Define the conditions for the set S
def valid_set (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, 0 < x ∧ x ≤ 15) ∧
  ∀ (A B : Finset ℕ), A ⊆ S → B ⊆ S → A ≠ B → A ∩ B = ∅ → A.sum id ≠ B.sum id

-- The theorem stating the largest sum of such a set
theorem largest_sum_valid_set : ∃ (S : Finset ℕ), valid_set S ∧ S.sum id = 61 :=
sorry

end largest_sum_valid_set_l139_139736


namespace no_real_solutions_for_inequality_l139_139625

theorem no_real_solutions_for_inequality (a : ℝ) :
  ¬∃ x : ℝ, ∀ y : ℝ, |(x^2 + a*x + 2*a)| ≤ 5 → y = x :=
sorry

end no_real_solutions_for_inequality_l139_139625


namespace power_greater_than_any_l139_139815

theorem power_greater_than_any {p M : ℝ} (hp : p > 0) (hM : M > 0) : ∃ n : ℕ, (1 + p)^n > M :=
by
  sorry

end power_greater_than_any_l139_139815


namespace circle_through_and_tangent_l139_139913

noncomputable def circle_eq (a b r : ℝ) (x y : ℝ) : ℝ :=
  (x - a) ^ 2 + (y - b) ^ 2 - r ^ 2

theorem circle_through_and_tangent
(h1 : circle_eq 1 2 2 1 0 = 0)
(h2 : ∀ x y, circle_eq 1 2 2 x y = 0 → (x = 1 → y = 2 ∨ y = -2))
: ∀ x y, circle_eq 1 2 2 x y = 0 → (x - 1) ^ 2 + (y - 2) ^ 2 = 4 :=
by
  sorry

end circle_through_and_tangent_l139_139913


namespace negation_of_universal_proposition_l139_139452

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by sorry

end negation_of_universal_proposition_l139_139452


namespace find_three_digit_number_l139_139552

theorem find_three_digit_number (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c)
  (h_sum : 122 * a + 212 * b + 221 * c = 2003) :
  100 * a + 10 * b + c = 345 :=
by
  sorry

end find_three_digit_number_l139_139552


namespace shepherd_boys_equation_l139_139459

theorem shepherd_boys_equation (x : ℕ) :
  6 * x + 14 = 8 * x - 2 :=
by sorry

end shepherd_boys_equation_l139_139459


namespace sqrt_25_eq_pm_five_l139_139272

theorem sqrt_25_eq_pm_five (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end sqrt_25_eq_pm_five_l139_139272


namespace greatest_matching_pairs_left_l139_139115

-- Define the initial number of pairs and lost individual shoes
def initial_pairs : ℕ := 26
def lost_ind_shoes : ℕ := 9

-- The statement to be proved
theorem greatest_matching_pairs_left : 
  (initial_pairs * 2 - lost_ind_shoes) / 2 + (initial_pairs - (initial_pairs * 2 - lost_ind_shoes) / 2) / 1 = 17 := 
by 
  sorry

end greatest_matching_pairs_left_l139_139115


namespace problem_solution_l139_139186

theorem problem_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2):
    (x ∈ Set.Iio (-2) ∪ Set.Ioo (-2) ((1 - Real.sqrt 129)/8) ∪ Set.Ioo 2 3 ∪ Set.Ioi ((1 + (Real.sqrt 129))/8)) ↔
    (2 * x^2 / (x + 2) ≥ 3 / (x - 2) + 6 / 4) :=
by
  sorry

end problem_solution_l139_139186


namespace fraction_equals_seven_twentyfive_l139_139227

theorem fraction_equals_seven_twentyfive :
  (1722^2 - 1715^2) / (1731^2 - 1706^2) = (7 / 25) :=
by
  sorry

end fraction_equals_seven_twentyfive_l139_139227


namespace min_workers_to_make_profit_l139_139158

theorem min_workers_to_make_profit :
  ∃ n : ℕ, 500 + 8 * 15 * n < 124 * n ∧ n = 126 :=
by
  sorry

end min_workers_to_make_profit_l139_139158


namespace children_more_than_adults_l139_139663

-- Definitions based on given conditions
def price_per_child : ℚ := 4.50
def price_per_adult : ℚ := 6.75
def total_receipts : ℚ := 405
def number_of_children : ℕ := 48

-- Goal: Prove the number of children is 20 more than the number of adults.
theorem children_more_than_adults :
  ∃ (A : ℕ), (number_of_children - A) = 20 ∧ (price_per_child * number_of_children) + (price_per_adult * A) = total_receipts := by
  sorry

end children_more_than_adults_l139_139663


namespace frustum_volume_correct_l139_139694

-- Definitions of pyramids and their properties
structure Pyramid :=
  (base_edge : ℕ)
  (altitude : ℕ)
  (volume : ℚ)

-- Definition of the original pyramid and smaller pyramid
def original_pyramid : Pyramid := {
  base_edge := 20,
  altitude := 10,
  volume := (1 / 3 : ℚ) * (20 ^ 2) * 10
}

def smaller_pyramid : Pyramid := {
  base_edge := 8,
  altitude := 5,
  volume := (1 / 3 : ℚ) * (8 ^ 2) * 5
}

-- Definition and calculation of the volume of the frustum 
def volume_frustum (p1 p2 : Pyramid) : ℚ :=
  p1.volume - p2.volume

-- Main theorem to be proved
theorem frustum_volume_correct :
  volume_frustum original_pyramid smaller_pyramid = 992 := by
  sorry

end frustum_volume_correct_l139_139694


namespace combined_loss_percentage_l139_139262

theorem combined_loss_percentage
  (cost_price_radio : ℕ := 8000)
  (quantity_radio : ℕ := 5)
  (discount_radio : ℚ := 0.1)
  (tax_radio : ℚ := 0.06)
  (sale_price_radio : ℕ := 7200)
  (cost_price_tv : ℕ := 20000)
  (quantity_tv : ℕ := 3)
  (discount_tv : ℚ := 0.15)
  (tax_tv : ℚ := 0.07)
  (sale_price_tv : ℕ := 18000)
  (cost_price_phone : ℕ := 15000)
  (quantity_phone : ℕ := 4)
  (discount_phone : ℚ := 0.08)
  (tax_phone : ℚ := 0.05)
  (sale_price_phone : ℕ := 14500) :
  let total_cost_price := (quantity_radio * cost_price_radio) + (quantity_tv * cost_price_tv) + (quantity_phone * cost_price_phone)
  let total_sale_price := (quantity_radio * sale_price_radio) + (quantity_tv * sale_price_tv) + (quantity_phone * sale_price_phone)
  let total_loss := total_cost_price - total_sale_price
  let loss_percentage := (total_loss * 100 : ℚ) / total_cost_price
  loss_percentage = 7.5 :=
by
  sorry

end combined_loss_percentage_l139_139262


namespace provisions_last_for_more_days_l139_139259

def initial_men : ℕ := 2000
def initial_days : ℕ := 65
def additional_men : ℕ := 3000
def days_used : ℕ := 15
def remaining_provisions :=
  initial_men * initial_days - initial_men * days_used
def total_men_after_reinforcement := initial_men + additional_men
def remaining_days := remaining_provisions / total_men_after_reinforcement

theorem provisions_last_for_more_days :
  remaining_days = 20 := by
  sorry

end provisions_last_for_more_days_l139_139259


namespace workshop_total_number_of_workers_l139_139800

theorem workshop_total_number_of_workers
  (average_salary_all : ℝ)
  (average_salary_technicians : ℝ)
  (average_salary_non_technicians : ℝ)
  (num_technicians : ℕ)
  (total_salary_all : ℝ -> ℝ)
  (total_salary_technicians : ℕ -> ℝ)
  (total_salary_non_technicians : ℕ -> ℝ -> ℝ)
  (h1 : average_salary_all = 9000)
  (h2 : average_salary_technicians = 12000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : ∀ W, total_salary_all W = average_salary_all * W )
  (h6 : ∀ n, total_salary_technicians n = n * average_salary_technicians )
  (h7 : ∀ n W, total_salary_non_technicians n W = (W - n) * average_salary_non_technicians)
  (h8 : ∀ W, total_salary_all W = total_salary_technicians num_technicians + total_salary_non_technicians num_technicians W) :
  ∃ W, W = 14 :=
by
  sorry

end workshop_total_number_of_workers_l139_139800


namespace coordinates_of_point_P_l139_139977

theorem coordinates_of_point_P 
  (P : ℝ × ℝ)
  (h1 : P.1 < 0 ∧ P.2 < 0) 
  (h2 : abs P.2 = 3)
  (h3 : abs P.1 = 5) :
  P = (-5, -3) :=
sorry

end coordinates_of_point_P_l139_139977


namespace total_spending_march_to_july_l139_139777

-- Define the conditions
def beginning_of_march_spending : ℝ := 1.2
def end_of_july_spending : ℝ := 4.8

-- State the theorem to prove
theorem total_spending_march_to_july : 
  end_of_july_spending - beginning_of_march_spending = 3.6 :=
sorry

end total_spending_march_to_july_l139_139777


namespace no_real_roots_ff_eq_x_l139_139516

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_real_roots_ff_eq_x (a b c : ℝ)
  (h : a ≠ 0)
  (discriminant_condition : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x := 
by 
  sorry

end no_real_roots_ff_eq_x_l139_139516


namespace number_of_operations_to_equal_l139_139294

theorem number_of_operations_to_equal (a b : ℤ) (da db : ℤ) (initial_diff change_per_operation : ℤ) (n : ℤ) 
(h1 : a = 365) 
(h2 : b = 24) 
(h3 : da = 19) 
(h4 : db = 12) 
(h5 : initial_diff = a - b) 
(h6 : change_per_operation = da + db) 
(h7 : initial_diff = 341) 
(h8 : change_per_operation = 31) 
(h9 : initial_diff = change_per_operation * n) :
n = 11 := 
by
  sorry

end number_of_operations_to_equal_l139_139294


namespace find_x_l139_139194

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
sorry

end find_x_l139_139194


namespace compare_logs_l139_139340

theorem compare_logs (a b c : ℝ) (h1 : a = Real.log 6 / Real.log 3)
                              (h2 : b = Real.log 8 / Real.log 4)
                              (h3 : c = Real.log 10 / Real.log 5) : 
                              a > b ∧ b > c :=
by
  sorry

end compare_logs_l139_139340


namespace distance_from_left_focal_to_line_l139_139521

noncomputable def ellipse_eq_line_dist : Prop :=
  let a := 2
  let b := Real.sqrt 3
  let c := 1
  let x₀ := -1
  let y₀ := 0
  let x₁ := 0
  let y₁ := Real.sqrt 3
  let x₂ := 1
  let y₂ := 0
  
  -- Equation of the line derived from the upper vertex and right focal point
  let m := -(y₁ - y₂) / (x₁ - x₂)
  let line_eq (x y : ℝ) := (Real.sqrt 3 * x + y - Real.sqrt 3 = 0)
  
  -- Distance formula from point to line
  let d := abs (Real.sqrt 3 * x₀ + y₀ - Real.sqrt 3) / Real.sqrt ((Real.sqrt 3)^2 + 1^2)

  -- The assertion that the distance is √3
  d = Real.sqrt 3

theorem distance_from_left_focal_to_line : ellipse_eq_line_dist := 
  sorry  -- Proof is omitted as per the instruction

end distance_from_left_focal_to_line_l139_139521


namespace ratio_of_a_b_to_b_c_l139_139857

theorem ratio_of_a_b_to_b_c (a b c : ℝ) (h₁ : b / a = 3) (h₂ : c / b = 2) : 
  (a + b) / (b + c) = 4 / 9 := by
  sorry

end ratio_of_a_b_to_b_c_l139_139857


namespace least_number_divisor_l139_139566

theorem least_number_divisor (d : ℕ) (n m : ℕ) 
  (h1 : d = 1081)
  (h2 : m = 1077)
  (h3 : n = 4)
  (h4 : ∃ k, m + n = k * d) :
  d = 1081 :=
by
  sorry

end least_number_divisor_l139_139566


namespace probability_of_first_four_cards_each_suit_l139_139010

noncomputable def probability_first_four_different_suits : ℚ := 3 / 32

theorem probability_of_first_four_cards_each_suit :
  let n := 52
  let k := 5
  let suits := 4
  (probability_first_four_different_suits = (3 / 32)) :=
by
  sorry

end probability_of_first_four_cards_each_suit_l139_139010


namespace number_of_solutions_l139_139664

theorem number_of_solutions :
  ∃ (s : Finset (ℤ × ℤ)), (∀ (a : ℤ × ℤ), a ∈ s ↔ (a.1^4 + a.2^4 = 4 * a.2)) ∧ s.card = 3 :=
by
  sorry

end number_of_solutions_l139_139664


namespace shuttle_speed_in_kph_l139_139988

def sec_per_min := 60
def min_per_hour := 60
def sec_per_hour := sec_per_min * min_per_hour
def speed_in_kps := 12
def speed_in_kph := speed_in_kps * sec_per_hour

theorem shuttle_speed_in_kph :
  speed_in_kph = 43200 :=
by
  -- No proof needed
  sorry

end shuttle_speed_in_kph_l139_139988


namespace avg_speed_l139_139343

variable (d1 d2 t1 t2 : ℕ)

-- Conditions
def distance_first_hour : ℕ := 80
def distance_second_hour : ℕ := 40
def time_first_hour : ℕ := 1
def time_second_hour : ℕ := 1

-- Ensure that total distance and total time are defined correctly from conditions
def total_distance : ℕ := distance_first_hour + distance_second_hour
def total_time : ℕ := time_first_hour + time_second_hour

-- Theorem to prove the average speed
theorem avg_speed : total_distance / total_time = 60 := by
  sorry

end avg_speed_l139_139343


namespace total_rabbits_correct_l139_139512

def initial_breeding_rabbits : ℕ := 10
def kittens_first_spring : ℕ := initial_breeding_rabbits * 10
def adopted_first_spring : ℕ := kittens_first_spring / 2
def returned_adopted_first_spring : ℕ := 5
def total_rabbits_after_first_spring : ℕ :=
  initial_breeding_rabbits + (kittens_first_spring - adopted_first_spring + returned_adopted_first_spring)

def kittens_second_spring : ℕ := 60
def adopted_second_spring : ℕ := kittens_second_spring * 40 / 100
def returned_adopted_second_spring : ℕ := 10
def total_rabbits_after_second_spring : ℕ :=
  total_rabbits_after_first_spring + (kittens_second_spring - adopted_second_spring + returned_adopted_second_spring)

def breeding_rabbits_third_spring : ℕ := 12
def kittens_third_spring : ℕ := breeding_rabbits_third_spring * 8
def adopted_third_spring : ℕ := kittens_third_spring * 30 / 100
def returned_adopted_third_spring : ℕ := 3
def total_rabbits_after_third_spring : ℕ :=
  total_rabbits_after_second_spring + (kittens_third_spring - adopted_third_spring + returned_adopted_third_spring)

def kittens_fourth_spring : ℕ := breeding_rabbits_third_spring * 6
def adopted_fourth_spring : ℕ := kittens_fourth_spring * 20 / 100
def returned_adopted_fourth_spring : ℕ := 2
def total_rabbits_after_fourth_spring : ℕ :=
  total_rabbits_after_third_spring + (kittens_fourth_spring - adopted_fourth_spring + returned_adopted_fourth_spring)

theorem total_rabbits_correct : total_rabbits_after_fourth_spring = 242 := by
  sorry

end total_rabbits_correct_l139_139512


namespace find_first_number_l139_139610

theorem find_first_number (y x : ℤ) (h1 : (y + 76 + x) / 3 = 5) (h2 : x = -63) : y = 2 :=
by
  -- To be filled in with the proof steps
  sorry

end find_first_number_l139_139610


namespace weight_of_one_liter_ghee_brand_b_l139_139900

theorem weight_of_one_liter_ghee_brand_b (wa w_mix : ℕ) (vol_a vol_b : ℕ) (w_mix_total : ℕ) (wb : ℕ) :
  wa = 900 ∧ vol_a = 3 ∧ vol_b = 2 ∧ w_mix = 3360 →
  (vol_a * wa + vol_b * wb = w_mix →
  wb = 330) :=
by
  intros h_eq h_eq2
  obtain ⟨h_wa, h_vol_a, h_vol_b, h_w_mix⟩ := h_eq
  rw [h_wa, h_vol_a, h_vol_b, h_w_mix] at h_eq2
  sorry

end weight_of_one_liter_ghee_brand_b_l139_139900


namespace correct_factorization_l139_139105

theorem correct_factorization :
  (∀ (x y : ℝ), x^2 + y^2 ≠ (x + y)^2) ∧
  (∀ (x y : ℝ), x^2 + 2*x*y + y^2 ≠ (x - y)^2) ∧
  (∀ (x : ℝ), x^2 + x ≠ x * (x - 1)) ∧
  (∀ (x y : ℝ), x^2 - y^2 = (x + y) * (x - y)) :=
by 
  sorry

end correct_factorization_l139_139105


namespace trapezoid_area_no_solutions_l139_139733

noncomputable def no_solutions_to_trapezoid_problem : Prop :=
  ∀ (b1 b2 : ℕ), 
    (∃ (m n : ℕ), b1 = 10 * m ∧ b2 = 10 * n) →
    (b1 + b2 = 72) → false

theorem trapezoid_area_no_solutions : no_solutions_to_trapezoid_problem :=
by
  sorry

end trapezoid_area_no_solutions_l139_139733


namespace find_k_l139_139522

theorem find_k (k : ℤ) :
  (∃ a b c : ℤ, a = 49 + k ∧ b = 441 + k ∧ c = 961 + k ∧
  (∃ r : ℚ, b = r * a ∧ c = r * r * a)) ↔ k = 1152 := by
  sorry

end find_k_l139_139522


namespace probability_at_least_one_l139_139994

theorem probability_at_least_one (p1 p2 : ℝ) (hp1 : 0 ≤ p1) (hp2 : 0 ≤ p2) (hp1p2 : p1 ≤ 1) (hp2p2 : p2 ≤ 1)
  (h0 : 0 ≤ 1 - p1) (h1 : 0 ≤ 1 - p2) (h2 : 1 - (1 - p1) ≥ 0) (h3 : 1 - (1 - p2) ≥ 0) :
  1 - (1 - p1) * (1 - p2) = 1 - (1 - p1) * (1 - p2) := by
  sorry

end probability_at_least_one_l139_139994


namespace gcd_of_three_numbers_l139_139542

theorem gcd_of_three_numbers (a b c : ℕ) (h1: a = 4557) (h2: b = 1953) (h3: c = 5115) : 
    Nat.gcd a (Nat.gcd b c) = 93 :=
by
  rw [h1, h2, h3]
  -- Proof goes here
  sorry

end gcd_of_three_numbers_l139_139542


namespace max_value_of_quadratic_function_l139_139044

noncomputable def quadratic_function (x : ℝ) : ℝ := -5*x^2 + 25*x - 15

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 750 :=
by
-- maximum value
sorry

end max_value_of_quadratic_function_l139_139044


namespace ratio_of_x_to_y_l139_139425

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 1 / 2) : x / y = 7 / 4 :=
sorry

end ratio_of_x_to_y_l139_139425


namespace algebraic_expression_evaluation_l139_139543

theorem algebraic_expression_evaluation (a b c : ℝ) 
  (h1 : a^2 + b * c = 14) 
  (h2 : b^2 - 2 * b * c = -6) : 
  3 * a^2 + 4 * b^2 - 5 * b * c = 18 :=
by 
  sorry

end algebraic_expression_evaluation_l139_139543


namespace solve_7_at_8_l139_139872

theorem solve_7_at_8 : (7 * 8) / (7 + 8 + 3) = 28 / 9 := by
  sorry

end solve_7_at_8_l139_139872


namespace triangle_enlargement_invariant_l139_139903

theorem triangle_enlargement_invariant (α β γ : ℝ) (h_sum : α + β + γ = 180) (f : ℝ) :
  (α * f ≠ α) ∧ (β * f ≠ β) ∧ (γ * f ≠ γ) → (α * f + β * f + γ * f = 180 * f) → α + β + γ = 180 :=
by
  sorry

end triangle_enlargement_invariant_l139_139903


namespace quadratic_expression_sum_l139_139778

theorem quadratic_expression_sum :
  ∃ a h k : ℝ, (∀ x, 4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) ∧ (a + h + k = 2) :=
sorry

end quadratic_expression_sum_l139_139778


namespace program_output_l139_139014

theorem program_output (a : ℕ) (h : a = 3) : (if a < 10 then 2 * a else a * a) = 6 :=
by
  rw [h]
  norm_num

end program_output_l139_139014


namespace sequence_formula_correct_l139_139075

noncomputable def S (n : ℕ) : ℕ := 2^n - 3

def a (n : ℕ) : ℤ :=
  if n = 1 then -1
  else 2^(n-1)

theorem sequence_formula_correct (n : ℕ) :
  a n = (if n = 1 then -1 else 2^(n-1)) :=
by
  sorry

end sequence_formula_correct_l139_139075


namespace stratified_sampling_correct_l139_139656

def total_employees : ℕ := 150
def senior_titles : ℕ := 15
def intermediate_titles : ℕ := 45
def general_staff : ℕ := 90
def sample_size : ℕ := 30

def stratified_sampling (total_employees senior_titles intermediate_titles general_staff sample_size : ℕ) : (ℕ × ℕ × ℕ) :=
  (senior_titles * sample_size / total_employees, 
   intermediate_titles * sample_size / total_employees, 
   general_staff * sample_size / total_employees)

theorem stratified_sampling_correct :
  stratified_sampling total_employees senior_titles intermediate_titles general_staff sample_size = (3, 9, 18) :=
  by sorry

end stratified_sampling_correct_l139_139656


namespace inequality_and_equality_condition_l139_139370

theorem inequality_and_equality_condition (a b : ℝ) (h : a < b) :
  a^3 - 3 * a ≤ b^3 - 3 * b + 4 ∧ (a = -1 ∧ b = 1 → a^3 - 3 * a = b^3 - 3 * b + 4) :=
sorry

end inequality_and_equality_condition_l139_139370


namespace joyce_apples_l139_139036

theorem joyce_apples (initial_apples given_apples remaining_apples : ℕ) (h1 : initial_apples = 75) (h2 : given_apples = 52) (h3 : remaining_apples = initial_apples - given_apples) : remaining_apples = 23 :=
by
  rw [h1, h2] at h3
  exact h3

end joyce_apples_l139_139036


namespace rational_inequality_solution_l139_139210

variable (x : ℝ)

def inequality_conditions : Prop := (2 * x - 1) / (x + 1) > 1

def inequality_solution : Prop := x < -1 ∨ x > 2

theorem rational_inequality_solution : inequality_conditions x → inequality_solution x :=
by
  sorry

end rational_inequality_solution_l139_139210


namespace quadratic_neq_l139_139328

theorem quadratic_neq (m : ℝ) : (m-2) ≠ 0 ↔ m ≠ 2 :=
sorry

end quadratic_neq_l139_139328


namespace integer_solutions_positive_product_l139_139621

theorem integer_solutions_positive_product :
  {a : ℤ | (5 + a) * (3 - a) > 0} = {-4, -3, -2, -1, 0, 1, 2} :=
by
  sorry

end integer_solutions_positive_product_l139_139621


namespace inequality_solution_l139_139015

theorem inequality_solution :
  {x : ℝ | (3 * x - 8) * (x - 4) / (x - 1) ≥ 0 } = { x : ℝ | x < 1 } ∪ { x : ℝ | x ≥ 4 } :=
by {
  sorry
}

end inequality_solution_l139_139015


namespace find_A_and_B_l139_139802

theorem find_A_and_B (A : ℕ) (B : ℕ) (x y : ℕ) 
  (h1 : 1000 ≤ A ∧ A ≤ 9999) 
  (h2 : B = 10^5 * x + 10 * A + y) 
  (h3 : B = 21 * A)
  (h4 : x < 10) 
  (h5 : y < 10) : 
  A = 9091 ∧ B = 190911 :=
sorry

end find_A_and_B_l139_139802


namespace eccentricity_of_ellipse_l139_139784

theorem eccentricity_of_ellipse :
  (∃ θ : Real, (x = 3 * Real.cos θ) ∧ (y = 4 * Real.sin θ))
  → (∃ e : Real, e = Real.sqrt 7 / 4) := 
sorry

end eccentricity_of_ellipse_l139_139784


namespace sum_F_G_H_l139_139894

theorem sum_F_G_H : 
  ∀ (F G H : ℕ), 
    (F < 10 ∧ G < 10 ∧ H < 10) ∧ 
    ∃ k : ℤ, 
      (F - 8 + 6 - 1 + G - 2 - H - 11 * k = 0) → 
        F + G + H = 23 :=
by sorry

end sum_F_G_H_l139_139894


namespace simplify_expression_l139_139709

-- Define the variables and conditions
variables {a b x y : ℝ}
variable (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
variable (h2 : x ≠ -(a * y) / b)
variable (h3 : x ≠ (b * y) / a)

-- The Theorem to prove
theorem simplify_expression
  (a b x y : ℝ)
  (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
  (h2 : x ≠ -(a * y) / b)
  (h3 : x ≠ (b * y) / a) :
  (a * b * (x^2 + y^2) + x * y * (a^2 + b^2)) *
  ((a * x + b * y)^2 - 4 * a * b * x * y) /
  (a * b * (x^2 - y^2) + x * y * (a^2 - b^2)) = 
  a^2 * x^2 - b^2 * y^2 :=
sorry

end simplify_expression_l139_139709


namespace number_of_pieces_l139_139380

def length_piece : ℝ := 0.40
def total_length : ℝ := 47.5

theorem number_of_pieces : ⌊total_length / length_piece⌋ = 118 := by
  sorry

end number_of_pieces_l139_139380


namespace train_passing_time_l139_139998

noncomputable def speed_in_m_per_s : ℝ := (60 * 1000) / 3600

variable (L : ℝ) (S : ℝ)
variable (train_length : L = 500)
variable (train_speed : S = speed_in_m_per_s)

theorem train_passing_time : L / S = 30 := by
  sorry

end train_passing_time_l139_139998


namespace min_value_x_add_y_l139_139249

variable {x y : ℝ}
variable (hx : 0 < x) (hy : 0 < y)
variable (h : 2 * x + 8 * y - x * y = 0)

theorem min_value_x_add_y : x + y ≥ 18 :=
by
  /- Proof goes here -/
  sorry

end min_value_x_add_y_l139_139249


namespace race_track_width_l139_139434

noncomputable def width_of_race_track (C_inner : ℝ) (r_outer : ℝ) : ℝ :=
  let r_inner := C_inner / (2 * Real.pi)
  r_outer - r_inner

theorem race_track_width : 
  width_of_race_track 880 165.0563499208679 = 25.0492072460867 :=
by
  sorry

end race_track_width_l139_139434


namespace riding_owners_ratio_l139_139433

theorem riding_owners_ratio :
  ∃ (R W : ℕ), (R + W = 16) ∧ (4 * R + 6 * W = 80) ∧ (R : ℚ) / 16 = 1/2 :=
by
  sorry

end riding_owners_ratio_l139_139433


namespace no_x_squared_term_l139_139649

theorem no_x_squared_term {m : ℚ} (h : (x+1) * (x^2 + 5*m*x + 3) = x^3 + (5*m + 1)*x^2 + (3 + 5*m)*x + 3) : 
  5*m + 1 = 0 → m = -1/5 := by sorry

end no_x_squared_term_l139_139649


namespace initial_number_of_persons_l139_139288

noncomputable def avg_weight_change : ℝ := 5.5
noncomputable def old_person_weight : ℝ := 68
noncomputable def new_person_weight : ℝ := 95.5
noncomputable def weight_diff : ℝ := new_person_weight - old_person_weight

theorem initial_number_of_persons (N : ℝ) 
  (h1 : avg_weight_change * N = weight_diff) : N = 5 :=
  by
  sorry

end initial_number_of_persons_l139_139288


namespace total_weight_cashew_nuts_and_peanuts_l139_139895

theorem total_weight_cashew_nuts_and_peanuts (weight_cashew_nuts weight_peanuts : ℕ) (h1 : weight_cashew_nuts = 3) (h2 : weight_peanuts = 2) : 
  weight_cashew_nuts + weight_peanuts = 5 := 
by
  sorry

end total_weight_cashew_nuts_and_peanuts_l139_139895


namespace mean_equals_sum_of_squares_l139_139940

noncomputable def arithmetic_mean (x y z : ℝ) := (x + y + z) / 3
noncomputable def geometric_mean (x y z : ℝ) := (x * y * z) ^ (1 / 3)
noncomputable def harmonic_mean (x y z : ℝ) := 3 / ((1 / x) + (1 / y) + (1 / z))

theorem mean_equals_sum_of_squares (x y z : ℝ) (h1 : arithmetic_mean x y z = 10)
  (h2 : geometric_mean x y z = 6) (h3 : harmonic_mean x y z = 4) :
  x^2 + y^2 + z^2 = 576 :=
  sorry

end mean_equals_sum_of_squares_l139_139940


namespace dress_shirt_cost_l139_139510

theorem dress_shirt_cost (x : ℝ) :
  let total_cost_before_discounts := 4 * x + 2 * 40 + 150 + 2 * 30
  let total_cost_after_store_discount := total_cost_before_discounts * 0.8
  let total_cost_after_coupon := total_cost_after_store_discount * 0.9
  total_cost_after_coupon = 252 → x = 15 :=
by
  let total_cost_before_discounts := 4 * x + 2 * 40 + 150 + 2 * 30
  let total_cost_after_store_discount := total_cost_before_discounts * 0.8
  let total_cost_after_coupon := total_cost_after_store_discount * 0.9
  intro h
  sorry

end dress_shirt_cost_l139_139510


namespace area_of_segment_solution_max_sector_angle_solution_l139_139744
open Real

noncomputable def area_of_segment (α R : ℝ) : ℝ :=
  let l := (R * α)
  let sector := 0.5 * R * l
  let triangle := 0.5 * R^2 * sin α
  sector - triangle

theorem area_of_segment_solution : area_of_segment (π / 3) 10 = 50 * ((π / 3) - (sqrt 3 / 2)) :=
by sorry

noncomputable def max_sector_angle (c : ℝ) (hc : c > 0) : ℝ :=
  2

theorem max_sector_angle_solution (c : ℝ) (hc : c > 0) : max_sector_angle c hc = 2 :=
by sorry

end area_of_segment_solution_max_sector_angle_solution_l139_139744


namespace number_of_slices_with_both_l139_139200

def total_slices : ℕ := 20
def slices_with_pepperoni : ℕ := 12
def slices_with_mushrooms : ℕ := 14
def slices_with_both_toppings (n : ℕ) : Prop :=
  n + (slices_with_pepperoni - n) + (slices_with_mushrooms - n) = total_slices

theorem number_of_slices_with_both (n : ℕ) (h : slices_with_both_toppings n) : n = 6 :=
sorry

end number_of_slices_with_both_l139_139200


namespace part1_part2_l139_139613
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem part1 (x : ℝ) : (f x 1) ≤ 5 ↔ (-1/2 : ℝ) ≤ x ∧ x ≤ 3/4 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, ∀ y : ℝ, f x a ≥ f y a) ↔ (-3 : ℝ) ≤ a ∧ a ≤ 3 := by
  sorry

end part1_part2_l139_139613


namespace find_c_k_l139_139748

noncomputable def a_n (n d : ℕ) := 1 + (n - 1) * d
noncomputable def b_n (n r : ℕ) := r ^ (n - 1)
noncomputable def c_n (n d r : ℕ) := a_n n d + b_n n r

theorem find_c_k (d r k : ℕ) (hd1 : c_n (k - 1) d r = 200) (hd2 : c_n (k + 1) d r = 2000) :
  c_n k d r = 423 :=
sorry

end find_c_k_l139_139748


namespace sum_of_digits_of_B_is_7_l139_139464

theorem sum_of_digits_of_B_is_7 : 
  let A := 16 ^ 16
  let sum_digits (n : ℕ) : ℕ := n.digits 10 |>.sum
  let S := sum_digits
  let B := S (S A)
  sum_digits B = 7 :=
sorry

end sum_of_digits_of_B_is_7_l139_139464


namespace center_of_circle_l139_139386

-- Definition of the main condition: the given circle equation
def circle_equation (x y : ℝ) := x^2 + y^2 = 10 * x - 4 * y + 14

-- Statement to prove: that x + y = 3 when (x, y) is the center of the circle described by circle_equation
theorem center_of_circle {x y : ℝ} (h : circle_equation x y) : x + y = 3 := 
by 
  sorry

end center_of_circle_l139_139386


namespace minimum_value_of_a_plus_4b_l139_139924

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hgeo : Real.sqrt (a * b) = 2)

theorem minimum_value_of_a_plus_4b : a + 4 * b = 8 := by
  sorry

end minimum_value_of_a_plus_4b_l139_139924


namespace distinct_arrangements_apple_l139_139041

theorem distinct_arrangements_apple : 
  let n := 5
  let freq_p := 2
  let freq_a := 1
  let freq_l := 1
  let freq_e := 1
  (Nat.factorial n) / (Nat.factorial freq_p * Nat.factorial freq_a * Nat.factorial freq_l * Nat.factorial freq_e) = 60 :=
by
  sorry

end distinct_arrangements_apple_l139_139041


namespace triangles_side_product_relation_l139_139755

-- Define the two triangles with their respective angles and side lengths
variables (A B C A1 B1 C1 : Type) 
          (angle_A angle_A1 angle_B angle_B1 : ℝ) 
          (a b c a1 b1 c1 : ℝ)

-- Given conditions
def angles_sum_to_180 (angle_A angle_A1 : ℝ) : Prop :=
  angle_A + angle_A1 = 180

def angles_equal (angle_B angle_B1 : ℝ) : Prop :=
  angle_B = angle_B1

-- The main theorem to be proven
theorem triangles_side_product_relation 
  (h1 : angles_sum_to_180 angle_A angle_A1)
  (h2 : angles_equal angle_B angle_B1) :
  a * a1 = b * b1 + c * c1 :=
sorry

end triangles_side_product_relation_l139_139755


namespace triangle_angles_30_60_90_l139_139559

-- Definition of the angles based on the given ratio
def angles_ratio (A B C : ℝ) : Prop :=
  A / B = 1 / 2 ∧ B / C = 2 / 3

-- The main statement to be proved
theorem triangle_angles_30_60_90
  (A B C : ℝ)
  (h1 : angles_ratio A B C)
  (h2 : A + B + C = 180) :
  A = 30 ∧ B = 60 ∧ C = 90 := 
sorry

end triangle_angles_30_60_90_l139_139559


namespace part_a_part_b_l139_139969

theorem part_a (p : ℕ) (hp : Nat.Prime p) (a b : ℤ) (h : a ≡ b [ZMOD p]) : a ^ p ≡ b ^ p [ZMOD p^2] :=
  sorry

theorem part_b (p : ℕ) (hp : Nat.Prime p) : 
  Nat.card { n | n ∈ Finset.range (p^2) ∧ ∃ x, x ^ p ≡ n [ZMOD p^2] } = p :=
  sorry

end part_a_part_b_l139_139969


namespace abc_inequality_l139_139447

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + b * c + a * c)^2 ≥ 3 * a * b * c * (a + b + c) :=
by sorry

end abc_inequality_l139_139447


namespace average_breadth_of_plot_l139_139470

theorem average_breadth_of_plot :
  ∃ B L : ℝ, (L - B = 10) ∧ (23 * B = (1/2) * (L + B) * B) ∧ (B = 18) :=
by
  sorry

end average_breadth_of_plot_l139_139470


namespace sticker_ratio_l139_139660

theorem sticker_ratio (gold : ℕ) (silver : ℕ) (bronze : ℕ)
  (students : ℕ) (stickers_per_student : ℕ)
  (h1 : gold = 50)
  (h2 : bronze = silver - 20)
  (h3 : students = 5)
  (h4 : stickers_per_student = 46)
  (h5 : gold + silver + bronze = students * stickers_per_student) :
  silver / gold = 2 / 1 :=
by
  sorry

end sticker_ratio_l139_139660


namespace inequality_solution_l139_139721

theorem inequality_solution :
  {x : ℝ | (x - 3) * (x + 2) ≠ 0 ∧ (x^2 + 1) / ((x - 3) * (x + 2)) ≥ 0} = 
  {x : ℝ | x ≤ -2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end inequality_solution_l139_139721


namespace pure_alcohol_addition_l139_139922

variable (x : ℝ)

def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.25
def final_concentration : ℝ := 0.50

theorem pure_alcohol_addition :
  (1.5 + x) / (initial_volume + x) = final_concentration → x = 3 :=
by
  sorry

end pure_alcohol_addition_l139_139922


namespace yara_total_earnings_l139_139831

-- Lean code to represent the conditions and the proof statement

theorem yara_total_earnings
  (x : ℕ)  -- Yara's hourly wage
  (third_week_hours : ℕ := 18)
  (previous_week_hours : ℕ := 12)
  (extra_earnings : ℕ := 36)
  (third_week_earning : ℕ := third_week_hours * x)
  (previous_week_earning : ℕ := previous_week_hours * x)
  (total_earning : ℕ := third_week_earning + previous_week_earning) :
  third_week_earning = previous_week_earning + extra_earnings → 
  total_earning = 180 := 
by
  -- Proof here
  sorry

end yara_total_earnings_l139_139831


namespace minimize_value_l139_139870

noncomputable def minimize_y (a b x : ℝ) : ℝ := (x - a) ^ 3 + (x - b) ^ 3

theorem minimize_value (a b : ℝ) : ∃ x : ℝ, minimize_y a b x = minimize_y a b a ∨ minimize_y a b x = minimize_y a b b :=
sorry

end minimize_value_l139_139870


namespace stratified_sampling_grade12_l139_139391

theorem stratified_sampling_grade12 (total_students grade12_students sample_size : ℕ) 
  (h_total : total_students = 2000) 
  (h_grade12 : grade12_students = 700) 
  (h_sample : sample_size = 400) : 
  (sample_size * grade12_students) / total_students = 140 := 
by 
  sorry

end stratified_sampling_grade12_l139_139391


namespace speed_in_still_water_l139_139468

theorem speed_in_still_water (v_m v_s : ℝ)
  (downstream : 48 = (v_m + v_s) * 3)
  (upstream : 34 = (v_m - v_s) * 4) :
  v_m = 12.25 :=
by
  sorry

end speed_in_still_water_l139_139468


namespace volume_of_prism_l139_139401

variable (l w h : ℝ)

def area1 (l w : ℝ) : ℝ := l * w
def area2 (w h : ℝ) : ℝ := w * h
def area3 (l h : ℝ) : ℝ := l * h
def volume (l w h : ℝ) : ℝ := l * w * h

axiom cond1 : area1 l w = 15
axiom cond2 : area2 w h = 20
axiom cond3 : area3 l h = 30

theorem volume_of_prism : volume l w h = 30 * Real.sqrt 10 :=
by
  sorry

end volume_of_prism_l139_139401


namespace square_side_length_l139_139569

theorem square_side_length (radius : ℝ) (s1 s2 : ℝ) (h1 : s1 = s2) (h2 : radius = 2 - Real.sqrt 2):
  s1 = 1 :=
  sorry

end square_side_length_l139_139569


namespace mean_value_of_quadrilateral_interior_angles_l139_139357

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l139_139357


namespace arc_length_l139_139973

theorem arc_length (circumference : ℝ) (angle : ℝ) (h1 : circumference = 72) (h2 : angle = 45) :
  ∃ length : ℝ, length = 9 :=
by
  sorry

end arc_length_l139_139973


namespace find_a_l139_139602

def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := a * b^3 + c

theorem find_a (a : ℚ) : F a 2 3 = F a 3 8 → a = -5 / 19 :=
by
  sorry

end find_a_l139_139602


namespace avg_korean_language_score_l139_139486

theorem avg_korean_language_score (male_avg : ℝ) (female_avg : ℝ) (male_students : ℕ) (female_students : ℕ) 
    (male_avg_given : male_avg = 83.1) (female_avg_given : female_avg = 84) (male_students_given : male_students = 10) (female_students_given : female_students = 8) :
    (male_avg * male_students + female_avg * female_students) / (male_students + female_students) = 83.5 :=
by sorry

end avg_korean_language_score_l139_139486


namespace length_of_shorter_leg_l139_139756

variable (h x : ℝ)

theorem length_of_shorter_leg 
  (h_med : h / 2 = 5 * Real.sqrt 3) 
  (hypotenuse_relation : h = 2 * x) 
  (median_relation : h / 2 = h / 2) :
  x = 5 := by sorry

end length_of_shorter_leg_l139_139756


namespace increase_by_1_or_prime_l139_139583

theorem increase_by_1_or_prime (a : ℕ → ℕ) :
  a 0 = 6 →
  (∀ n, a (n + 1) = a n + Nat.gcd (a n) (n + 1)) →
  ∀ n, n < 1000000 → (∃ p, p = 1 ∨ Nat.Prime p ∧ a (n + 1) = a n + p) :=
by
  intro ha0 ha_step
  -- Proof omitted
  sorry

end increase_by_1_or_prime_l139_139583


namespace amy_bike_total_l139_139908

-- Define the miles Amy biked yesterday
def y : ℕ := 12

-- Define the miles Amy biked today
def t : ℕ := 2 * y - 3

-- Define the total miles Amy biked in two days
def total : ℕ := y + t

-- The theorem stating the total distance biked equals 33 miles
theorem amy_bike_total : total = 33 := by
  sorry

end amy_bike_total_l139_139908


namespace function_y_increases_when_x_gt_1_l139_139333

theorem function_y_increases_when_x_gt_1 :
  ∀ (x : ℝ), (x > 1 → 2*x^2 > 2*(x-1)^2) :=
by
  sorry

end function_y_increases_when_x_gt_1_l139_139333


namespace matrix_power_50_l139_139377

def P : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 2],
  ![-4, -3]
]

theorem matrix_power_50 :
  P ^ 50 = ![
    ![1, 0],
    ![0, 1]
  ] :=
sorry

end matrix_power_50_l139_139377


namespace find_k_l139_139148

theorem find_k (x y z k : ℝ) (h1 : 8 / (x + y + 1) = k / (x + z + 2)) (h2 : k / (x + z + 2) = 12 / (z - y + 3)) : k = 20 := by
  sorry

end find_k_l139_139148


namespace trig_identity_l139_139723

theorem trig_identity (x : ℝ) (h : Real.sin (x + π / 3) = 1 / 3) : 
  Real.sin ((5 * π) / 3 - x) - Real.cos (2 * x - π / 3) = 4 / 9 := 
by 
  sorry

end trig_identity_l139_139723


namespace nancy_soap_bars_l139_139155

def packs : ℕ := 6
def bars_per_pack : ℕ := 5

theorem nancy_soap_bars : packs * bars_per_pack = 30 := by
  sorry

end nancy_soap_bars_l139_139155


namespace exists_n_lt_p_minus_1_not_div_p2_l139_139738

theorem exists_n_lt_p_minus_1_not_div_p2 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) :
  ∃ (n : ℕ), n < p - 1 ∧ ¬(p^2 ∣ (n^((p - 1)) - 1)) ∧ ¬(p^2 ∣ ((n + 1)^((p - 1)) - 1)) := 
sorry

end exists_n_lt_p_minus_1_not_div_p2_l139_139738


namespace sum_of_fractions_decimal_equivalence_l139_139039

theorem sum_of_fractions :
  (2 / 15 : ℚ) + (4 / 20) + (5 / 45) = 4 / 9 := 
sorry

theorem decimal_equivalence :
  (4 / 9 : ℚ) = 0.444 := 
sorry

end sum_of_fractions_decimal_equivalence_l139_139039


namespace divisible_by_bn_l139_139277

variables {u v a b : ℤ} {n : ℕ}

theorem divisible_by_bn 
  (h1 : ∀ x : ℤ, x^2 + a*x + b = 0 → x = u ∨ x = v)
  (h2 : a^2 % b = 0) 
  (h3 : ∀ m : ℕ, m = 2 * n) : 
  ∀ n : ℕ, (u^m + v^m) % (b^n) = 0 := 
  sorry

end divisible_by_bn_l139_139277


namespace find_pairs_l139_139359

theorem find_pairs (a b q r : ℕ) (h1 : a * b = q * (a + b) + r)
  (h2 : q^2 + r = 2011) (h3 : 0 ≤ r ∧ r < a + b) : 
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 45 ∧ (a = t ∧ b = t + 2012 ∨ a = t + 2012 ∧ b = t)) :=
by
  sorry

end find_pairs_l139_139359


namespace angle_in_third_quadrant_l139_139980

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end angle_in_third_quadrant_l139_139980


namespace percentage_second_question_correct_l139_139833

theorem percentage_second_question_correct (a b c : ℝ) 
  (h1 : a = 0.75) (h2 : b = 0.20) (h3 : c = 0.50) :
  (1 - b) - (a - c) + c = 0.55 :=
by
  sorry

end percentage_second_question_correct_l139_139833


namespace racing_championship_guarantee_l139_139107

/-- 
In a racing championship consisting of five races, the points awarded are as follows: 
6 points for first place, 4 points for second place, and 2 points for third place, with no ties possible. 
What is the smallest number of points a racer must accumulate in these five races to be guaranteed of having more points than any other racer? 
-/
theorem racing_championship_guarantee :
  ∀ (points_1st : ℕ) (points_2nd : ℕ) (points_3rd : ℕ) (races : ℕ),
  points_1st = 6 → points_2nd = 4 → points_3rd = 2 → 
  races = 5 →
  (∃ min_points : ℕ, min_points = 26 ∧ 
    ∀ (possible_points : ℕ), possible_points ≠ min_points → 
    (possible_points < min_points)) :=
by
  sorry

end racing_championship_guarantee_l139_139107


namespace number_of_children_l139_139689

variables (n : ℕ) (y : ℕ) (d : ℕ)

def sum_of_ages (n : ℕ) (y : ℕ) (d : ℕ) : ℕ :=
  n * y + d * (n * (n - 1) / 2)

theorem number_of_children (H1 : sum_of_ages n 6 3 = 60) : n = 6 :=
by {
  sorry
}

end number_of_children_l139_139689


namespace sin_of_angle_l139_139821

theorem sin_of_angle (α : ℝ) (h : Real.cos (π + α) = -(1/3)) : Real.sin ((3 * π / 2) - α) = -(1/3) := 
by
  sorry

end sin_of_angle_l139_139821


namespace tourists_count_l139_139553

theorem tourists_count (n k : ℤ) (h1 : 2 * k % n = 1) (h2 : 3 * k % n = 13) : n = 23 := 
by
-- Proof is omitted
sorry

end tourists_count_l139_139553


namespace inverse_of_p_l139_139807

variables {p q r : Prop}

theorem inverse_of_p (m n : Prop) (hp : p = (m → n)) (hq : q = (¬m → ¬n)) (hr : r = (n → m)) : r = p ∧ r = (n → m) :=
by
  sorry

end inverse_of_p_l139_139807


namespace sum_of_values_not_satisfying_eq_l139_139682

variable {A B C x : ℝ}

theorem sum_of_values_not_satisfying_eq (h : (∀ x, ∃ C, ∃ B, A = 3 ∧ ((x + B) * (A * x + 36) = 3 * (x + C) * (x + 9)) ∧ (x ≠ -9))):
  ∃ y, y = -9 := sorry

end sum_of_values_not_satisfying_eq_l139_139682


namespace C0E_hex_to_dec_l139_139563

theorem C0E_hex_to_dec : 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  result = 3086 :=
by 
  let C := 12
  let E := 14 
  let result := C * 16^2 + 0 * 16^1 + E * 16^0
  sorry

end C0E_hex_to_dec_l139_139563


namespace Kim_morning_routine_time_l139_139720

def total_employees : ℕ := 9
def senior_employees : ℕ := 3
def overtime_employees : ℕ := 4
def regular_employees : ℕ := total_employees - senior_employees
def non_overtime_employees : ℕ := total_employees - overtime_employees

def coffee_time : ℕ := 5
def status_update_time (regular senior : ℕ) : ℕ := (regular * 2) + (senior * 3)
def payroll_update_time (overtime non_overtime : ℕ) : ℕ := (overtime * 3) + (non_overtime * 1)
def email_time : ℕ := 10
def task_allocation_time : ℕ := 7

def total_morning_routine_time : ℕ :=
  coffee_time +
  status_update_time regular_employees senior_employees +
  payroll_update_time overtime_employees non_overtime_employees +
  email_time +
  task_allocation_time

theorem Kim_morning_routine_time : total_morning_routine_time = 60 := by
  sorry

end Kim_morning_routine_time_l139_139720


namespace willie_cream_from_farm_l139_139431

variable (total_needed amount_to_buy amount_from_farm : ℕ)

theorem willie_cream_from_farm :
  total_needed = 300 → amount_to_buy = 151 → amount_from_farm = total_needed - amount_to_buy → amount_from_farm = 149 := by
  intros
  sorry

end willie_cream_from_farm_l139_139431


namespace smallest_n_l139_139874

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n % 8 = 5) (h4 : n > 20) : n = 136 := by
  sorry

end smallest_n_l139_139874


namespace product_of_first_nine_terms_l139_139990

-- Declare the geometric sequence and given condition
variable {α : Type*} [Field α]
variable {a : ℕ → α}
variable (r : α) (a1 : α)

-- Define that the sequence is geometric
def is_geometric_sequence (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  ∀ n : ℕ, a n = a1 * r ^ n

-- Given a_5 = -2 in the sequence
def geometric_sequence_with_a5 (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  is_geometric_sequence a r a1 ∧ a 5 = -2

-- Prove that the product of the first 9 terms is -512
theorem product_of_first_nine_terms 
  (a : ℕ → α) 
  (r : α) 
  (a₁ : α) 
  (h : geometric_sequence_with_a5 a r a₁) : 
  (a 0) * (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = -512 := 
by
  sorry

end product_of_first_nine_terms_l139_139990


namespace rationalize_denominator_correct_l139_139310

noncomputable def rationalize_denominator : Prop :=
  (1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2)

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l139_139310


namespace area_of_walkways_l139_139016

-- Define the dimensions of the individual flower bed
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3

-- Define the number of rows and columns of flower beds
def rows_of_beds : ℕ := 4
def cols_of_beds : ℕ := 3

-- Define the width of the walkways
def walkway_width : ℕ := 2

-- Calculate the total width and height of the garden including walkways
def total_width : ℕ := (cols_of_beds * flower_bed_width) + (cols_of_beds + 1) * walkway_width
def total_height : ℕ := (rows_of_beds * flower_bed_height) + (rows_of_beds + 1) * walkway_width

-- Calculate the area of the garden including walkways
def total_area : ℕ := total_width * total_height

-- Calculate the total area of all the flower beds
def total_beds_area : ℕ := (rows_of_beds * cols_of_beds) * (flower_bed_width * flower_bed_height)

-- Prove the area of walkways
theorem area_of_walkways : total_area - total_beds_area = 416 := by
  sorry

end area_of_walkways_l139_139016


namespace find_x_of_arithmetic_mean_l139_139634

theorem find_x_of_arithmetic_mean (x : ℝ) (h : (6 + 13 + 18 + 4 + x) / 5 = 10) : x = 9 :=
by
  sorry

end find_x_of_arithmetic_mean_l139_139634


namespace find_x_l139_139601

variable (x : ℝ)

def length := 4 * x
def width := x + 3

def area := length x * width x
def perimeter := 2 * length x + 2 * width x

theorem find_x (h : area x = 3 * perimeter x) : x = 5.342 := by
  sorry

end find_x_l139_139601


namespace product_divisible_by_8_probability_l139_139697

noncomputable def probability_product_divisible_by_8 (dice_rolls : Fin 6 → Fin 8) : ℚ :=
  -- Function to calculate the probability that the product of numbers is divisible by 8
  sorry

theorem product_divisible_by_8_probability :
  ∀ (dice_rolls : Fin 6 → Fin 8),
  probability_product_divisible_by_8 dice_rolls = 177 / 256 :=
sorry

end product_divisible_by_8_probability_l139_139697


namespace slope_of_line_l139_139978

theorem slope_of_line (x₁ y₁ x₂ y₂ : ℝ) (h₁ : x₁ = 1) (h₂ : y₁ = 3) (h₃ : x₂ = 4) (h₄ : y₂ = -6) : 
  (y₂ - y₁) / (x₂ - x₁) = -3 := by
  sorry

end slope_of_line_l139_139978


namespace Thabo_books_ratio_l139_139503

variable (P_f P_nf H_nf : ℕ)

theorem Thabo_books_ratio :
  P_f + P_nf + H_nf = 220 →
  H_nf = 40 →
  P_nf = H_nf + 20 →
  P_f / P_nf = 2 :=
by sorry

end Thabo_books_ratio_l139_139503


namespace average_of_hidden_primes_l139_139047

theorem average_of_hidden_primes (p₁ p₂ : ℕ) (h₁ : Nat.Prime p₁) (h₂ : Nat.Prime p₂) (h₃ : p₁ + 37 = p₂ + 53) : 
  (p₁ + p₂) / 2 = 11 := 
by
  sorry

end average_of_hidden_primes_l139_139047


namespace probability_correct_l139_139362

noncomputable def probability_one_white_one_black
    (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (draw_balls : ℕ) :=
if (total_balls = 4) ∧ (white_balls = 2) ∧ (black_balls = 2) ∧ (draw_balls = 2) then
  (2 * 2) / (Nat.choose total_balls draw_balls : ℚ)
else
  0

theorem probability_correct:
  probability_one_white_one_black 4 2 2 2 = 2 / 3 :=
by
  sorry

end probability_correct_l139_139362


namespace fraction_available_on_third_day_l139_139446

noncomputable def liters_used_on_first_day (initial_amount : ℕ) : ℕ :=
  (initial_amount / 2)

noncomputable def liters_added_on_second_day : ℕ :=
  1

noncomputable def original_solution : ℕ :=
  4

noncomputable def remaining_solution_after_first_day : ℕ :=
  original_solution - liters_used_on_first_day original_solution

noncomputable def remaining_solution_after_second_day : ℕ :=
  remaining_solution_after_first_day + liters_added_on_second_day

noncomputable def fraction_of_original_solution : ℚ :=
  remaining_solution_after_second_day / original_solution

theorem fraction_available_on_third_day : fraction_of_original_solution = 3 / 4 :=
by
  sorry

end fraction_available_on_third_day_l139_139446


namespace sum_of_consecutive_2022_l139_139817

theorem sum_of_consecutive_2022 (m n : ℕ) (h : m ≤ n - 1) (sum_eq : (n - m + 1) * (m + n) = 4044) :
  (m = 163 ∧ n = 174) ∨ (m = 504 ∧ n = 507) ∨ (m = 673 ∧ n = 675) :=
sorry

end sum_of_consecutive_2022_l139_139817


namespace green_sequins_per_row_correct_l139_139986

def total_blue_sequins : ℕ := 6 * 8
def total_purple_sequins : ℕ := 5 * 12
def total_green_sequins : ℕ := 162 - (total_blue_sequins + total_purple_sequins)
def green_sequins_per_row : ℕ := total_green_sequins / 9

theorem green_sequins_per_row_correct : green_sequins_per_row = 6 := 
by 
  sorry

end green_sequins_per_row_correct_l139_139986


namespace least_k_inequality_l139_139208

theorem least_k_inequality :
  ∃ k : ℝ, (∀ a b c : ℝ, 
    ((2 * a / (a - b)) ^ 2 + (2 * b / (b - c)) ^ 2 + (2 * c / (c - a)) ^ 2 + k 
    ≥ 4 * (2 * a / (a - b) + 2 * b / (b - c) + 2 * c / (c - a)))) ∧ k = 8 :=
by
  sorry  -- proof is omitted

end least_k_inequality_l139_139208


namespace cubic_difference_l139_139384

theorem cubic_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a^3 - b^3 = 108 :=
sorry

end cubic_difference_l139_139384


namespace geometric_to_arithmetic_l139_139838

theorem geometric_to_arithmetic (a_1 a_2 a_3 b_1 b_2 b_3: ℝ) (ha: a_1 > 0 ∧ a_2 > 0 ∧ a_3 > 0 ∧ b_1 > 0 ∧ b_2 > 0 ∧ b_3 > 0)
  (h_geometric_a : ∃ q : ℝ, a_2 = a_1 * q ∧ a_3 = a_1 * q^2)
  (h_geometric_b : ∃ q₁ : ℝ, b_2 = b_1 * q₁ ∧ b_3 = b_1 * q₁^2)
  (h_sum : a_1 + a_2 + a_3 = b_1 + b_2 + b_3)
  (h_arithmetic : 2 * a_2 * b_2 = a_1 * b_1 + a_3 * b_3) : 
  a_2 = b_2 :=
by
  sorry

end geometric_to_arithmetic_l139_139838


namespace range_of_g_l139_139707

noncomputable def g (a x : ℝ) : ℝ :=
  a * (Real.cos x)^4 - 2 * (Real.sin x) * (Real.cos x) + (Real.sin x)^4

theorem range_of_g (a : ℝ) (h : a > 0) :
  Set.range (g a) = Set.Icc (a - (3 - a) / 2) (a + (a + 1) / 2) :=
sorry

end range_of_g_l139_139707


namespace example_equation_l139_139842

-- Define what it means to be an equation in terms of containing an unknown and being an equality
def is_equation (expr : Prop) (contains_unknown : Prop) : Prop :=
  (contains_unknown ∧ expr)

-- Prove that 4x + 2 = 10 is an equation
theorem example_equation : is_equation (4 * x + 2 = 10) (∃ x : ℝ, true) :=
  by sorry

end example_equation_l139_139842


namespace exists_n_divisible_by_5_l139_139271

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h_div : a * m ^ 3 + b * m ^ 2 + c * m + d ≡ 0 [ZMOD 5]) 
  (h_d_nonzero : d ≠ 0) : 
  ∃ n : ℤ, d * n ^ 3 + c * n ^ 2 + b * n + a ≡ 0 [ZMOD 5] :=
sorry

end exists_n_divisible_by_5_l139_139271


namespace cube_of_sum_l139_139948

theorem cube_of_sum :
  (100 + 2) ^ 3 = 1061208 :=
by
  sorry

end cube_of_sum_l139_139948


namespace compare_powers_l139_139891

theorem compare_powers (a b c d : ℝ) (h1 : a + b = 0) (h2 : c + d = 0) : a^5 + d^6 = c^6 - b^5 :=
by
  sorry

end compare_powers_l139_139891


namespace charge_per_mile_l139_139189

def rental_fee : ℝ := 20.99
def total_amount_paid : ℝ := 95.74
def miles_driven : ℝ := 299

theorem charge_per_mile :
  (total_amount_paid - rental_fee) / miles_driven = 0.25 := 
sorry

end charge_per_mile_l139_139189


namespace determine_number_of_shelves_l139_139019

-- Define the total distance Karen bikes round trip
def total_distance : ℕ := 3200

-- Define the number of books per shelf
def books_per_shelf : ℕ := 400

-- Calculate the one-way distance from Karen's home to the library
def one_way_distance (total_distance : ℕ) : ℕ := total_distance / 2

-- Define the total number of books, which is the same as the one-way distance
def total_books (one_way_distance : ℕ) : ℕ := one_way_distance

-- Calculate the number of shelves
def number_of_shelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

theorem determine_number_of_shelves :
  number_of_shelves (total_books (one_way_distance total_distance)) books_per_shelf = 4 :=
by 
  -- the proof would go here
  sorry

end determine_number_of_shelves_l139_139019


namespace evelyn_total_marbles_l139_139236

def initial_marbles := 95
def marbles_from_henry := 9
def marbles_from_grace := 12
def number_of_cards := 6
def marbles_per_card := 4

theorem evelyn_total_marbles :
  initial_marbles + marbles_from_henry + marbles_from_grace + number_of_cards * marbles_per_card = 140 := 
by 
  sorry

end evelyn_total_marbles_l139_139236


namespace quadratic_min_value_l139_139221

theorem quadratic_min_value (p q : ℝ) (h : ∀ x : ℝ, 3 * x^2 + p * x + q ≥ 4) : q = p^2 / 12 + 4 :=
sorry

end quadratic_min_value_l139_139221


namespace batter_sugar_is_one_l139_139931

-- Definitions based on the conditions given
def initial_sugar : ℕ := 3
def sugar_per_bag : ℕ := 6
def num_bags : ℕ := 2
def frosting_sugar_per_dozen : ℕ := 2
def total_dozen_cupcakes : ℕ := 5

-- Total sugar Lillian has
def total_sugar : ℕ := initial_sugar + num_bags * sugar_per_bag

-- Sugar needed for frosting
def frosting_sugar_needed : ℕ := frosting_sugar_per_dozen * total_dozen_cupcakes

-- Sugar used for the batter
def batter_sugar_total : ℕ := total_sugar - frosting_sugar_needed

-- Question asked in the problem
def batter_sugar_per_dozen : ℕ := batter_sugar_total / total_dozen_cupcakes

theorem batter_sugar_is_one :
  batter_sugar_per_dozen = 1 :=
by
  sorry -- Proof is not required here

end batter_sugar_is_one_l139_139931


namespace find_number_l139_139909

theorem find_number (f : ℝ → ℝ) (x : ℝ)
  (h : f (x * 0.004) / 0.03 = 9.237333333333334)
  (h_linear : ∀ a, f a = a) :
  x = 69.3 :=
by
  -- Proof goes here
  sorry

end find_number_l139_139909


namespace product_of_slopes_l139_139181

theorem product_of_slopes (m n : ℝ) (φ₁ φ₂ : ℝ) 
  (h1 : ∀ x, y = m * x)
  (h2 : ∀ x, y = n * x)
  (h3 : φ₁ = 2 * φ₂) 
  (h4 : m = 3 * n)
  (h5 : m ≠ 0 ∧ n ≠ 0)
  : m * n = 3 / 5 :=
sorry

end product_of_slopes_l139_139181


namespace black_ants_employed_l139_139839

theorem black_ants_employed (total_ants : ℕ) (red_ants : ℕ) 
  (h1 : total_ants = 900) (h2 : red_ants = 413) :
    total_ants - red_ants = 487 :=
by
  -- The proof is given below.
  sorry

end black_ants_employed_l139_139839


namespace hexagonal_pyramid_volume_l139_139264

theorem hexagonal_pyramid_volume (a : ℝ) (h : a > 0) (lateral_surface_area : ℝ) (base_area : ℝ)
  (H_base_area : base_area = (3 * Real.sqrt 3 / 2) * a^2)
  (H_lateral_surface_area : lateral_surface_area = 10 * base_area) :
  (1 / 3) * base_area * (a * Real.sqrt 3 / 2) * 3 * Real.sqrt 11 = (9 * a^3 * Real.sqrt 11) / 4 :=
by sorry

end hexagonal_pyramid_volume_l139_139264


namespace probability_change_needed_l139_139928

noncomputable def toy_prices : List ℝ := List.range' 1 11 |>.map (λ n => n * 0.25)

def favorite_toy_price : ℝ := 2.25

def total_quarters : ℕ := 12

def total_toy_count : ℕ := 10

def total_orders : ℕ := Nat.factorial total_toy_count

def ways_to_buy_without_change : ℕ :=
  (Nat.factorial (total_toy_count - 1)) + 2 * (Nat.factorial (total_toy_count - 2))

def probability_without_change : ℚ :=
  ↑ways_to_buy_without_change / ↑total_orders

def probability_with_change : ℚ :=
  1 - probability_without_change

theorem probability_change_needed : probability_with_change = 79 / 90 :=
  sorry

end probability_change_needed_l139_139928


namespace sue_votes_correct_l139_139011

def total_votes : ℕ := 1000
def percentage_others : ℝ := 0.65
def sue_votes : ℕ := 350

theorem sue_votes_correct :
  sue_votes = (total_votes : ℝ) * (1 - percentage_others) :=
by
  sorry

end sue_votes_correct_l139_139011


namespace goldfish_in_first_tank_l139_139945

-- Definitions of conditions
def num_fish_third_tank : Nat := 10
def num_fish_second_tank := 3 * num_fish_third_tank
def num_fish_first_tank := num_fish_second_tank / 2
def goldfish_and_beta_sum (G : Nat) : Prop := G + 8 = num_fish_first_tank

-- Theorem to prove the number of goldfish in the first fish tank
theorem goldfish_in_first_tank (G : Nat) (h : goldfish_and_beta_sum G) : G = 7 :=
by
  sorry

end goldfish_in_first_tank_l139_139945


namespace find_pairs_l139_139388

theorem find_pairs (a b : ℕ) :
  (1111 * a) % (11 * b) = 11 * (a - b) →
  140 ≤ (1111 * a) / (11 * b) ∧ (1111 * a) / (11 * b) ≤ 160 →
  (a, b) = (3, 2) ∨ (a, b) = (6, 4) ∨ (a, b) = (7, 5) ∨ (a, b) = (9, 6) :=
by
  sorry

end find_pairs_l139_139388


namespace largest_three_digit_number_l139_139995

theorem largest_three_digit_number :
  ∃ (n : ℕ), (n < 1000) ∧ (n % 7 = 1) ∧ (n % 8 = 4) ∧ (∀ (m : ℕ), (m < 1000) ∧ (m % 7 = 1) ∧ (m % 8 = 4) → m ≤ n) :=
sorry

end largest_three_digit_number_l139_139995


namespace total_kayaks_built_by_april_l139_139699

def kayaks_built_february : ℕ := 5
def kayaks_built_next_month (n : ℕ) : ℕ := 3 * n
def kayaks_built_march : ℕ := kayaks_built_next_month kayaks_built_february
def kayaks_built_april : ℕ := kayaks_built_next_month kayaks_built_march

theorem total_kayaks_built_by_april : 
  kayaks_built_february + kayaks_built_march + kayaks_built_april = 65 :=
by
  -- proof goes here
  sorry

end total_kayaks_built_by_april_l139_139699


namespace table_covered_area_l139_139618

-- Definitions based on conditions
def length := 12
def width := 1
def number_of_strips := 4
def overlapping_strips := 3

-- Calculating the area of one strip
def area_of_one_strip := length * width

-- Calculating total area assuming no overlaps
def total_area_no_overlap := number_of_strips * area_of_one_strip

-- Calculating the total overlap area
def overlap_area := overlapping_strips * (width * width)

-- Final area after subtracting overlaps
def final_covered_area := total_area_no_overlap - overlap_area

-- Theorem stating the proof problem
theorem table_covered_area : final_covered_area = 45 :=
by
  sorry

end table_covered_area_l139_139618


namespace car_highway_miles_per_tankful_l139_139191

-- Defining conditions as per given problem
def city_miles_per_tank : ℕ := 336
def city_miles_per_gallon : ℕ := 8
def difference_miles_per_gallon : ℕ := 3
def highway_miles_per_gallon := city_miles_per_gallon + difference_miles_per_gallon
def tank_size := city_miles_per_tank / city_miles_per_gallon
def highway_miles_per_tank := highway_miles_per_gallon * tank_size

-- Theorem statement to prove
theorem car_highway_miles_per_tankful :
  highway_miles_per_tank = 462 :=
sorry

end car_highway_miles_per_tankful_l139_139191


namespace width_of_road_correct_l139_139609

-- Define the given conditions
def sum_of_circumferences (r R : ℝ) : Prop := 2 * Real.pi * r + 2 * Real.pi * R = 88
def radius_relation (r R : ℝ) : Prop := r = (1/3) * R
def width_of_road (R r : ℝ) := R - r

-- State the main theorem
theorem width_of_road_correct (R r : ℝ) (h1 : sum_of_circumferences r R) (h2 : radius_relation r R) :
    width_of_road R r = 22 / Real.pi := by
  sorry

end width_of_road_correct_l139_139609


namespace gcd_104_156_l139_139540

theorem gcd_104_156 : Nat.gcd 104 156 = 52 :=
by
  -- the proof steps will go here, but we can use sorry to skip it
  sorry

end gcd_104_156_l139_139540


namespace combined_work_rate_l139_139234

def work_done_in_one_day (A B : ℕ) (work_to_days : ℕ -> ℕ) : ℚ :=
  (work_to_days A + work_to_days B)

theorem combined_work_rate (A : ℕ) (B : ℕ) (work_to_days : ℕ -> ℕ) :
  work_to_days A = 1/18 ∧ work_to_days B = 1/9 → work_done_in_one_day A B (work_to_days) = 1/6 :=
by
  sorry

end combined_work_rate_l139_139234


namespace smallest_integer_remainder_conditions_l139_139594

theorem smallest_integer_remainder_conditions :
  ∃ b : ℕ, (b % 3 = 0) ∧ (b % 4 = 2) ∧ (b % 5 = 3) ∧ (∀ n : ℕ, (n % 3 = 0) ∧ (n % 4 = 2) ∧ (n % 5 = 3) → b ≤ n) :=
sorry

end smallest_integer_remainder_conditions_l139_139594


namespace four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l139_139248

noncomputable def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

noncomputable def sum_of_digits_is_11 (n : ℕ) : Prop := 
  let d1 := n / 1000
  let r1 := n % 1000
  let d2 := r1 / 100
  let r2 := r1 % 100
  let d3 := r2 / 10
  let d4 := r2 % 10
  d1 + d2 + d3 + d4 = 11

theorem four_digit_numbers_divisible_by_11_with_sum_of_digits_11
  (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : is_divisible_by_11 n)
  (h3 : sum_of_digits_is_11 n) : 
  n = 2090 ∨ n = 3080 ∨ n = 4070 ∨ n = 5060 ∨ n = 6050 ∨ n = 7040 ∨ n = 8030 ∨ n = 9020 :=
sorry

end four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l139_139248


namespace total_amount_paid_l139_139135

theorem total_amount_paid :
  let chapati_cost := 6
  let rice_cost := 45
  let mixed_vegetable_cost := 70
  let ice_cream_cost := 40
  let chapati_quantity := 16
  let rice_quantity := 5
  let mixed_vegetable_quantity := 7
  let ice_cream_quantity := 6
  let total_cost := chapati_quantity * chapati_cost +
                    rice_quantity * rice_cost +
                    mixed_vegetable_quantity * mixed_vegetable_cost +
                    ice_cream_quantity * ice_cream_cost
  total_cost = 1051 := by
  sorry

end total_amount_paid_l139_139135


namespace chord_line_equation_l139_139099

theorem chord_line_equation (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), y1^2 = -8 * x1 ∧ y2^2 = -8 * x2 ∧ (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1 ∧ y - 1 = -4 * (x + 1)) →
  4 * x + y + 3 = 0 :=
by
  sorry

end chord_line_equation_l139_139099


namespace propositions_correctness_l139_139183

variable {a b c d : ℝ}

theorem propositions_correctness (h0 : a > b) (h1 : c > d) (h2 : c > 0) :
  (a > b ∧ c > d → a + c > b + d) ∧ 
  (a > b ∧ c > d → ¬(a - c > b - d)) ∧ 
  (a > b ∧ c > d → ¬(a * c > b * d)) ∧ 
  (a > b ∧ c > 0 → a * c > b * c) :=
by
  sorry

end propositions_correctness_l139_139183


namespace original_number_solution_l139_139698

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l139_139698


namespace students_called_back_l139_139052

theorem students_called_back (g b d t c : ℕ) (h1 : g = 9) (h2 : b = 14) (h3 : d = 21) (h4 : t = g + b) (h5 : c = t - d) : c = 2 := by 
  sorry

end students_called_back_l139_139052


namespace intersection_union_complement_union_l139_139060

open Set

variable (U : Set ℝ) (A B : Set ℝ)
variable [Inhabited (Set ℝ)]

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 2) > 1 }
noncomputable def setB : Set ℝ := { x : ℝ | x ≥ 0 }

theorem intersection (U : Set ℝ) : 
  (setA ∩ setB) = { x : ℝ | (0 < x ∧ x < 1) ∨ x > 3 } := 
  sorry

theorem union (U : Set ℝ) : 
  (setA ∪ setB) = univ := 
  sorry

theorem complement_union (U : Set ℝ) : 
  ((U \ setA) ∪ setB) = { x : ℝ | x ≥ 0 } := 
  sorry

end intersection_union_complement_union_l139_139060


namespace ratio_of_investments_l139_139421

theorem ratio_of_investments {A B C : ℝ} (x y z k : ℝ)
  (h1 : B - A = 100)
  (h2 : A + B + C = 2900)
  (h3 : A = 6 * k)
  (h4 : B = 5 * k)
  (h5 : C = 4 * k) : 
  (x / y = 6 / 5) ∧ (y / z = 5 / 4) ∧ (x / z = 6 / 4) :=
by
  sorry

end ratio_of_investments_l139_139421


namespace billy_picked_36_dandelions_initially_l139_139342

namespace Dandelions

/-- The number of dandelions Billy picked initially. -/
def billy_initial (B : ℕ) : ℕ := B

/-- The number of dandelions George picked initially. -/
def george_initial (B : ℕ) : ℕ := B / 3

/-- The additional dandelions picked by Billy and George respectively. -/
def additional_dandelions : ℕ := 10

/-- The total dandelions picked by Billy and George initially and additionally. -/
def total_dandelions (B : ℕ) : ℕ :=
  billy_initial B + additional_dandelions + george_initial B + additional_dandelions

/-- The average number of dandelions picked by both Billy and George, given as 34. -/
def average_dandelions (total : ℕ) : Prop := total / 2 = 34

/-- The main theorem stating that Billy picked 36 dandelions initially. -/
theorem billy_picked_36_dandelions_initially :
  ∀ B : ℕ, average_dandelions (total_dandelions B) ↔ B = 36 :=
by
  intro B
  sorry

end Dandelions

end billy_picked_36_dandelions_initially_l139_139342


namespace segment_shadow_ratio_l139_139653

theorem segment_shadow_ratio (a b a' b' : ℝ) (h : a / b = a' / b') : a / a' = b / b' :=
sorry

end segment_shadow_ratio_l139_139653


namespace opposite_of_neg_2_l139_139808

theorem opposite_of_neg_2 : ∃ y : ℝ, -2 + y = 0 ∧ y = 2 := by
  sorry

end opposite_of_neg_2_l139_139808


namespace fraction_boxes_loaded_by_day_crew_l139_139921

variables {D W_d : ℝ}

theorem fraction_boxes_loaded_by_day_crew
  (h1 : ∀ (D W_d: ℝ), D > 0 → W_d > 0 → ∃ (D' W_n : ℝ), (D' = 0.5 * D) ∧ (W_n = 0.8 * W_d))
  (h2 : ∃ (D W_d : ℝ), ∀ (D' W_n : ℝ), (D' = 0.5 * D) → (W_n = 0.8 * W_d) → 
        (D * W_d / (D * W_d + D' * W_n)) = (5 / 7)) :
  (∃ (D W_d : ℝ), D > 0 → W_d > 0 → (D * W_d)/(D * W_d + 0.5 * D * 0.8 * W_d) = (5/7)) := 
  sorry 

end fraction_boxes_loaded_by_day_crew_l139_139921


namespace min_value_of_x_plus_y_l139_139242

open Real

theorem min_value_of_x_plus_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0)
  (a : ℝ × ℝ := (1 - x, 4)) (b : ℝ × ℝ := (x, -y))
  (h₃ : ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)) :
  x + y = 9 :=
by
  sorry

end min_value_of_x_plus_y_l139_139242


namespace probability_abs_x_le_one_l139_139176

noncomputable def geometric_probability (a b c d : ℝ) : ℝ := (b - a) / (d - c)

theorem probability_abs_x_le_one : 
  ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 3 →  
  geometric_probability (-1) 1 (-1) 3 = 1 / 2 := 
by
  sorry

end probability_abs_x_le_one_l139_139176


namespace solve_proof_problem_l139_139505

noncomputable def proof_problem (f g : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x + g y) = 2 * x + y → g (x + f y) = x / 2 + y

theorem solve_proof_problem (f g : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + g y) = 2 * x + y) :
  ∀ x y : ℝ, g (x + f y) = x / 2 + y :=
sorry

end solve_proof_problem_l139_139505


namespace unique_solution_condition_l139_139054

-- Define p and q as real numbers
variables (p q : ℝ)

-- The Lean statement to prove a unique solution when q ≠ 4
theorem unique_solution_condition : (∀ x : ℝ, (4 * x - 7 + p = q * x + 2) ↔ (q ≠ 4)) :=
by
  sorry

end unique_solution_condition_l139_139054


namespace abs_neg_one_ninth_l139_139526

theorem abs_neg_one_ninth : abs (- (1 / 9)) = 1 / 9 := by
  sorry

end abs_neg_one_ninth_l139_139526


namespace triangle_smallest_angle_l139_139360

theorem triangle_smallest_angle (a b c : ℝ) (h1 : a + b + c = 180) (h2 : a = 5 * c) (h3 : b = 3 * c) : c = 20 :=
by
  sorry

end triangle_smallest_angle_l139_139360


namespace Petya_has_24_chips_l139_139932

noncomputable def PetyaChips (x y : ℕ) : ℕ := 3 * x - 3

theorem Petya_has_24_chips (x y : ℕ) (h1 : y = x - 2) (h2 : 3 * x - 3 = 4 * y - 4) : PetyaChips x y = 24 :=
by
  sorry

end Petya_has_24_chips_l139_139932


namespace smallest_exponentiated_number_l139_139687

theorem smallest_exponentiated_number :
  127^8 < 63^10 ∧ 63^10 < 33^12 := 
by 
  -- Proof omitted
  sorry

end smallest_exponentiated_number_l139_139687


namespace exist_triangle_l139_139905

-- Definitions of points and properties required in the conditions
structure Point :=
(x : ℝ) (y : ℝ)

def orthocenter (M : Point) := M 
def centroid (S : Point) := S 
def vertex (C : Point) := C 

-- The problem statement that needs to be proven
theorem exist_triangle (M S C : Point) 
    (h_orthocenter : orthocenter M = M)
    (h_centroid : centroid S = S)
    (h_vertex : vertex C = C) : 
    ∃ (A B : Point), 
        -- A, B, and C form a triangle ABC
        -- S is the centroid of this triangle
        -- M is the orthocenter of this triangle
        -- C is one of the vertices
        true := 
sorry

end exist_triangle_l139_139905


namespace gemstones_needed_l139_139154

noncomputable def magnets_per_earring : ℕ := 2

noncomputable def buttons_per_earring : ℕ := magnets_per_earring / 2

noncomputable def gemstones_per_earring : ℕ := 3 * buttons_per_earring

noncomputable def sets_of_earrings : ℕ := 4

noncomputable def earrings_per_set : ℕ := 2

noncomputable def total_gemstones : ℕ := sets_of_earrings * earrings_per_set * gemstones_per_earring

theorem gemstones_needed :
  total_gemstones = 24 :=
  by
    sorry

end gemstones_needed_l139_139154


namespace students_without_an_A_l139_139890

theorem students_without_an_A :
  ∀ (total_students : ℕ) (history_A : ℕ) (math_A : ℕ) (computing_A : ℕ)
    (math_and_history_A : ℕ) (history_and_computing_A : ℕ)
    (math_and_computing_A : ℕ) (all_three_A : ℕ),
  total_students = 40 →
  history_A = 10 →
  math_A = 18 →
  computing_A = 9 →
  math_and_history_A = 5 →
  history_and_computing_A = 3 →
  math_and_computing_A = 4 →
  all_three_A = 2 →
  total_students - (history_A + math_A + computing_A - math_and_history_A - history_and_computing_A - math_and_computing_A + all_three_A) = 13 :=
by
  intros total_students history_A math_A computing_A math_and_history_A history_and_computing_A math_and_computing_A all_three_A 
         ht_total_students ht_history_A ht_math_A ht_computing_A ht_math_and_history_A ht_history_and_computing_A ht_math_and_computing_A ht_all_three_A
  sorry

end students_without_an_A_l139_139890


namespace distinct_p_q_r_s_t_sum_l139_139498

theorem distinct_p_q_r_s_t_sum (p q r s t : ℤ) (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t)
    (h9 : r ≠ s) (h10 : r ≠ t)
    (h11 : s ≠ t) : p + q + r + s + t = 25 := by
  sorry

end distinct_p_q_r_s_t_sum_l139_139498


namespace monkey_swinging_speed_l139_139555

namespace LamplighterMonkey

def running_speed : ℝ := 15
def running_time : ℝ := 5
def swinging_time : ℝ := 10
def total_distance : ℝ := 175

theorem monkey_swinging_speed : 
  (total_distance = running_speed * running_time + (running_speed / swinging_time) * swinging_time) → 
  (running_speed / swinging_time = 10) := 
by 
  intros h
  sorry

end LamplighterMonkey

end monkey_swinging_speed_l139_139555


namespace oranges_in_bag_l139_139475

variables (O : ℕ)

def initial_oranges (O : ℕ) := O
def initial_tangerines := 17
def oranges_left_after_taking_away := O - 2
def tangerines_left_after_taking_away := 7
def tangerines_and_oranges_condition (O : ℕ) := 7 = (O - 2) + 4

theorem oranges_in_bag (O : ℕ) (h₀ : tangerines_and_oranges_condition O) : O = 5 :=
by
  sorry

end oranges_in_bag_l139_139475


namespace fraction_simplified_form_l139_139772

variables (a b c : ℝ)

noncomputable def fraction : ℝ := (a^2 - b^2 + c^2 + 2 * b * c) / (a^2 - c^2 + b^2 + 2 * a * b)

theorem fraction_simplified_form (h : a^2 - c^2 + b^2 + 2 * a * b ≠ 0) :
  fraction a b c = (a^2 - b^2 + c^2 + 2 * b * c) / (a^2 - c^2 + b^2 + 2 * a * b) :=
by sorry

end fraction_simplified_form_l139_139772


namespace min_number_of_candy_kinds_l139_139096

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l139_139096


namespace smallest_natural_number_l139_139479

theorem smallest_natural_number :
  ∃ N : ℕ, ∃ f : ℕ → ℕ → ℕ, 
  f (f (f 9 8 - f 7 6) 5 + 4 - f 3 2) 1 = N ∧
  N = 1 := 
by sorry

end smallest_natural_number_l139_139479


namespace solve_xyz_l139_139051

theorem solve_xyz (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  (x, y, z) = ( (b + c) / 2, (c + a) / 2, (a + b) / 2 ) :=
sorry

end solve_xyz_l139_139051


namespace right_angled_triangle_only_B_l139_139764

def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_only_B :
  forms_right_angled_triangle 1 (Real.sqrt 3) 2 ∧
  ¬forms_right_angled_triangle 1 2 2 ∧
  ¬forms_right_angled_triangle 4 5 6 ∧
  ¬forms_right_angled_triangle 1 1 (Real.sqrt 3) :=
by
  sorry

end right_angled_triangle_only_B_l139_139764


namespace floor_neg_seven_thirds_l139_139959

theorem floor_neg_seven_thirds : ⌊-7 / 3⌋ = -3 :=
sorry

end floor_neg_seven_thirds_l139_139959


namespace part1_part2_l139_139113

noncomputable def f (m x : ℝ) : ℝ := m - |x - 1| - |x + 1|

theorem part1 (x : ℝ) : -3 / 2 < x ∧ x < 3 / 2 ↔ f 5 x > 2 := by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ y : ℝ, x^2 + 2 * x + 3 = f m y) ↔ 4 ≤ m := by
  sorry

end part1_part2_l139_139113


namespace primes_satisfying_condition_l139_139742

theorem primes_satisfying_condition :
    {p : ℕ | p.Prime ∧ ∀ q : ℕ, q.Prime ∧ q < p → ¬ ∃ n : ℕ, n^2 ∣ (p - (p / q) * q)} =
    {2, 3, 5, 7, 13} :=
by sorry

end primes_satisfying_condition_l139_139742


namespace gcf_360_180_l139_139323

theorem gcf_360_180 : Nat.gcd 360 180 = 180 :=
by
  sorry

end gcf_360_180_l139_139323


namespace dog_food_weight_l139_139701

/-- 
 Mike has 2 dogs, each dog eats 6 cups of dog food twice a day.
 Mike buys 9 bags of 20-pound dog food a month.
 Prove that a cup of dog food weighs 0.25 pounds.
-/
theorem dog_food_weight :
  let dogs := 2
  let cups_per_meal := 6
  let meals_per_day := 2
  let bags_per_month := 9
  let weight_per_bag := 20
  let days_per_month := 30
  let total_cups_per_day := cups_per_meal * meals_per_day * dogs
  let total_cups_per_month := total_cups_per_day * days_per_month
  let total_weight_per_month := bags_per_month * weight_per_bag
  (total_weight_per_month / total_cups_per_month : ℝ) = 0.25 :=
by
  sorry

end dog_food_weight_l139_139701


namespace find_p_l139_139896

theorem find_p 
  (h : {x | x^2 - 5 * x + p ≥ 0} = {x | x ≤ -1 ∨ x ≥ 6}) : p = -6 :=
by
  sorry

end find_p_l139_139896


namespace matching_pairs_less_than_21_in_at_least_61_positions_l139_139813

theorem matching_pairs_less_than_21_in_at_least_61_positions :
  ∀ (disks : ℕ) (total_sectors : ℕ) (red_sectors : ℕ) (max_overlap : ℕ) (rotations : ℕ),
  disks = 2 →
  total_sectors = 1965 →
  red_sectors = 200 →
  max_overlap = 20 →
  rotations = total_sectors →
  (∃ positions, positions = total_sectors - (red_sectors * red_sectors / (max_overlap + 1)) ∧ positions ≤ rotations) →
  positions = 61 :=
by {
  -- Placeholder to provide the structure of the theorem.
  sorry
}

end matching_pairs_less_than_21_in_at_least_61_positions_l139_139813


namespace sin_cos_value_l139_139893

theorem sin_cos_value (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (3 * Real.sin x + Real.cos x = 0) ∨ (3 * Real.sin x + Real.cos x = -4) :=
sorry

end sin_cos_value_l139_139893


namespace inequality_proof_l139_139327

theorem inequality_proof
  (a b c d e f : ℝ)
  (h1 : 1 ≤ a)
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d)
  (h5 : d ≤ e)
  (h6 : e ≤ f) :
  (a * f + b * e + c * d) * (a * f + b * d + c * e) ≤ (a + b^2 + c^3) * (d + e^2 + f^3) := 
by 
  sorry

end inequality_proof_l139_139327


namespace problem_l139_139665

noncomputable def f (x : ℝ) : ℝ := 3 / (x + 1)

theorem problem (x : ℝ) (h1 : 3 ≤ x) (h2 : x ≤ 5) :
  (∀ x₁ x₂, 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ > f x₁) ∧
  f 3 = 3/4 ∧
  f 5 = 1/2 :=
by sorry

end problem_l139_139665


namespace general_formula_l139_139123

-- Define the sequence term a_n
def sequence_term (n : ℕ) : ℚ :=
  if h : n = 0 then 1
  else (2 * n - 1 : ℚ) / (n * n)

-- State the theorem for the general formula of the nth term
theorem general_formula (n : ℕ) (hn : n ≠ 0) : 
  sequence_term n = (2 * n - 1 : ℚ) / (n * n) :=
by sorry

end general_formula_l139_139123


namespace jimmy_yellow_marbles_correct_l139_139768

def lorin_black_marbles : ℕ := 4
def alex_black_marbles : ℕ := 2 * lorin_black_marbles
def alex_total_marbles : ℕ := 19
def alex_yellow_marbles : ℕ := alex_total_marbles - alex_black_marbles
def jimmy_yellow_marbles : ℕ := 2 * alex_yellow_marbles

theorem jimmy_yellow_marbles_correct : jimmy_yellow_marbles = 22 := by
  sorry

end jimmy_yellow_marbles_correct_l139_139768


namespace counties_no_rain_l139_139192

theorem counties_no_rain 
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ) :
  P_A = 0.7 → P_B = 0.5 → P_A_and_B = 0.4 →
  (1 - (P_A + P_B - P_A_and_B) = 0.2) :=
by intros h1 h2 h3; sorry

end counties_no_rain_l139_139192


namespace jenni_age_l139_139897

theorem jenni_age (B J : ℕ) (h1 : B + J = 70) (h2 : B - J = 32) : J = 19 :=
by
  sorry

end jenni_age_l139_139897


namespace find_y_given_x_eq_neg6_l139_139620

theorem find_y_given_x_eq_neg6 :
  ∀ (y : ℤ), (∃ (x : ℤ), x = -6 ∧ x^2 - x + 6 = y - 6) → y = 54 :=
by
  intros y h
  obtain ⟨x, hx1, hx2⟩ := h
  rw [hx1] at hx2
  simp at hx2
  linarith

end find_y_given_x_eq_neg6_l139_139620


namespace oplus_calculation_l139_139301

def my_oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem oplus_calculation : my_oplus 2 3 = 23 := 
by
    sorry

end oplus_calculation_l139_139301


namespace binomial_expansion_judgments_l139_139706

theorem binomial_expansion_judgments :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r) ∧
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r + 3) :=
by
  sorry

end binomial_expansion_judgments_l139_139706


namespace sum_sum_sum_sum_eq_one_l139_139935

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Mathematical problem statement
theorem sum_sum_sum_sum_eq_one :
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits (2017^2017)))) = 1 := 
sorry

end sum_sum_sum_sum_eq_one_l139_139935


namespace player_matches_l139_139798

theorem player_matches (n : ℕ) :
  (34 * n + 78 = 38 * (n + 1)) → n = 10 :=
by
  intro h
  have h1 : 34 * n + 78 = 38 * n + 38 := by sorry
  have h2 : 78 = 4 * n + 38 := by sorry
  have h3 : 40 = 4 * n := by sorry
  have h4 : n = 10 := by sorry
  exact h4

end player_matches_l139_139798


namespace ned_total_mows_l139_139938

def ned_mowed_front (spring summer fall : Nat) : Nat :=
  spring + summer + fall

def ned_mowed_backyard (spring summer fall : Nat) : Nat :=
  spring + summer + fall

theorem ned_total_mows :
  let front_spring := 6
  let front_summer := 5
  let front_fall := 4
  let backyard_spring := 5
  let backyard_summer := 7
  let backyard_fall := 3
  ned_mowed_front front_spring front_summer front_fall +
  ned_mowed_backyard backyard_spring backyard_summer backyard_fall = 30 := by
  sorry

end ned_total_mows_l139_139938


namespace average_rounds_rounded_eq_4_l139_139065

def rounds_distribution : List (Nat × Nat) := [(1, 4), (2, 3), (4, 4), (5, 2), (6, 6)]

def total_rounds : Nat := rounds_distribution.foldl (λ acc (rounds, golfers) => acc + rounds * golfers) 0

def total_golfers : Nat := rounds_distribution.foldl (λ acc (_, golfers) => acc + golfers) 0

def average_rounds : Float := total_rounds.toFloat / total_golfers.toFloat

theorem average_rounds_rounded_eq_4 : Float.round average_rounds = 4 := by
  sorry

end average_rounds_rounded_eq_4_l139_139065


namespace original_price_of_computer_l139_139419

theorem original_price_of_computer (P : ℝ) (h1 : 1.20 * P = 351) (h2 : 2 * P = 585) : P = 292.5 :=
by
  sorry

end original_price_of_computer_l139_139419


namespace complex_point_second_quadrant_l139_139416

theorem complex_point_second_quadrant (i : ℂ) (h1 : i^4 = 1) :
  ∃ (z : ℂ), z = ((i^(2014))/(1 + i) * i) ∧ z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_second_quadrant_l139_139416


namespace players_in_physics_class_l139_139719

theorem players_in_physics_class (total players_math players_both : ℕ)
    (h1 : total = 15)
    (h2 : players_math = 9)
    (h3 : players_both = 4) :
    (players_math - players_both) + (total - (players_math - players_both + players_both)) + players_both = 10 :=
by {
  sorry
}

end players_in_physics_class_l139_139719


namespace transmission_time_is_128_l139_139392

def total_time (blocks chunks_per_block rate : ℕ) : ℕ :=
  (blocks * chunks_per_block) / rate

theorem transmission_time_is_128 :
  total_time 80 256 160 = 128 :=
  by
  sorry

end transmission_time_is_128_l139_139392


namespace mary_more_candy_initially_l139_139695

-- Definitions of the conditions
def Megan_initial_candy : ℕ := 5
def Mary_candy_after_addition : ℕ := 25
def additional_candy_Mary_adds : ℕ := 10

-- The proof problem statement
theorem mary_more_candy_initially :
  (Mary_candy_after_addition - additional_candy_Mary_adds) / Megan_initial_candy = 3 :=
by
  sorry

end mary_more_candy_initially_l139_139695


namespace avg_age_of_team_is_23_l139_139090

-- Conditions
def captain_age := 24
def wicket_keeper_age := captain_age + 7

def remaining_players_avg_age (team_avg_age : ℝ) := team_avg_age - 1
def total_team_age (team_avg_age : ℝ) := 11 * team_avg_age
def total_remaining_players_age (team_avg_age : ℝ) := 9 * remaining_players_avg_age team_avg_age

-- Proof statement
theorem avg_age_of_team_is_23 (team_avg_age : ℝ) :
  total_team_age team_avg_age = captain_age + wicket_keeper_age + total_remaining_players_age team_avg_age → 
  team_avg_age = 23 :=
by
  sorry

end avg_age_of_team_is_23_l139_139090


namespace eventually_periodic_sequence_l139_139270

theorem eventually_periodic_sequence
  (a : ℕ → ℕ) (h_pos : ∀ n, 0 < a n)
  (h_div : ∀ n m, (a (n + 2 * m)) ∣ (a n + a (n + m))) :
  ∃ N d, 0 < N ∧ 0 < d ∧ ∀ n, N < n → a n = a (n + d) :=
by
  sorry

end eventually_periodic_sequence_l139_139270


namespace number_of_puppies_l139_139078

def total_portions : Nat := 105
def feeding_days : Nat := 5
def feedings_per_day : Nat := 3

theorem number_of_puppies (total_portions feeding_days feedings_per_day : Nat) : 
  (total_portions / feeding_days / feedings_per_day = 7) := 
by 
  sorry

end number_of_puppies_l139_139078


namespace sum_of_series_l139_139830

theorem sum_of_series : 
  (1 / (1 * 2) + 1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 6 / 7 := 
 by sorry

end sum_of_series_l139_139830


namespace chess_tournament_total_players_l139_139481

-- Define the conditions

def total_points_calculation (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 132

def games_played (n : ℕ) : ℕ :=
  ((n + 12) * (n + 11)) / 2

theorem chess_tournament_total_players :
  ∃ n, total_points_calculation n = games_played n ∧ n + 12 = 34 :=
by {
  -- Assume n is found such that all conditions are satisfied
  use 22,
  -- Provide the necessary equations and conditions
  sorry
}

end chess_tournament_total_players_l139_139481


namespace least_sum_of_bases_l139_139944

theorem least_sum_of_bases (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : 4 * a + 7 = 7 * b + 4) (h4 : 4 * a + 3 % 7 = 0) :
  a + b = 24 :=
sorry

end least_sum_of_bases_l139_139944


namespace unit_vector_same_direction_l139_139810

-- Define the coordinates of points A and B as given in the conditions
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

-- Define the vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of vector AB
noncomputable def magnitudeAB : ℝ := Real.sqrt (vectorAB.1^2 + vectorAB.2^2)

-- Define the unit vector in the direction of AB
noncomputable def unitVectorAB : ℝ × ℝ := (vectorAB.1 / magnitudeAB, vectorAB.2 / magnitudeAB)

-- The theorem we want to prove
theorem unit_vector_same_direction :
  unitVectorAB = (3 / 5, -4 / 5) :=
sorry

end unit_vector_same_direction_l139_139810


namespace leak_empty_time_l139_139219

theorem leak_empty_time
  (R : ℝ) (L : ℝ)
  (hR : R = 1 / 8)
  (hRL : R - L = 1 / 10) :
  1 / L = 40 :=
by
  sorry

end leak_empty_time_l139_139219


namespace domain_tan_2x_plus_pi_over_3_l139_139951

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 2}

noncomputable def domain_tan_transformed : Set ℝ :=
  {x | ∃ k : ℤ, x = k * (Real.pi / 2) + Real.pi / 12}

theorem domain_tan_2x_plus_pi_over_3 :
  (∀ x, ¬ (x ∈ domain_tan)) ↔ (∀ x, ¬ (x ∈ domain_tan_transformed)) :=
by
  sorry

end domain_tan_2x_plus_pi_over_3_l139_139951


namespace decreasing_function_implies_inequality_l139_139020

theorem decreasing_function_implies_inequality (k b : ℝ) (h : ∀ x : ℝ, (2 * k + 1) * x + b = (2 * k + 1) * x + b) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k + 1) * x1 + b > (2 * k + 1) * x2 + b) → k < -1/2 :=
by sorry

end decreasing_function_implies_inequality_l139_139020


namespace sandwiches_left_l139_139929

theorem sandwiches_left 
    (initial_sandwiches : ℕ)
    (first_coworker : ℕ)
    (second_coworker : ℕ)
    (third_coworker : ℕ)
    (kept_sandwiches : ℕ) :
    initial_sandwiches = 50 →
    first_coworker = 4 →
    second_coworker = 3 →
    third_coworker = 2 * first_coworker →
    kept_sandwiches = 3 * second_coworker →
    initial_sandwiches - (first_coworker + second_coworker + third_coworker + kept_sandwiches) = 26 :=
by
  intros h_initial h_first h_second h_third h_kept
  rw [h_initial, h_first, h_second, h_third, h_kept]
  simp
  norm_num
  sorry

end sandwiches_left_l139_139929


namespace persons_in_office_l139_139100

theorem persons_in_office
  (P : ℕ)
  (h1 : (P - (1/7 : ℚ)*P) = (6/7 : ℚ)*P)
  (h2 : (16.66666666666667/100 : ℚ) = 1/6) :
  P = 35 :=
sorry

end persons_in_office_l139_139100


namespace correct_statement_of_abs_l139_139919

theorem correct_statement_of_abs (r : ℚ) :
  ¬ (∀ r : ℚ, abs r > 0) ∧
  ¬ (∀ a b : ℚ, a ≠ b → abs a ≠ abs b) ∧
  (∀ r : ℚ, abs r ≥ 0) ∧
  ¬ (∀ r : ℚ, r < 0 → abs r = -r ∧ abs r < 0 → abs r ≠ -r) :=
by
  sorry

end correct_statement_of_abs_l139_139919


namespace quadrant_of_points_l139_139169

theorem quadrant_of_points (x y : ℝ) (h : |3 * x + 2| + |2 * y - 1| = 0) : 
  ((x < 0) ∧ (y > 0) ∧ (x + 1 > 0) ∧ (y - 2 < 0)) :=
by
  sorry

end quadrant_of_points_l139_139169


namespace Mark_time_spent_l139_139341

theorem Mark_time_spent :
  let parking_time := 5
  let walking_time := 3
  let long_wait_time := 30
  let short_wait_time := 10
  let long_wait_days := 2
  let short_wait_days := 3
  let work_days := 5
  (parking_time + walking_time) * work_days + 
    long_wait_time * long_wait_days + 
    short_wait_time * short_wait_days = 130 :=
by
  sorry

end Mark_time_spent_l139_139341


namespace pipe_pumping_rate_l139_139255

theorem pipe_pumping_rate (R : ℕ) (h : 5 * R + 5 * 192 = 1200) : R = 48 := by
  sorry

end pipe_pumping_rate_l139_139255


namespace bobby_candy_total_l139_139077

-- Definitions for the conditions
def initial_candy : Nat := 20
def first_candy_eaten : Nat := 34
def second_candy_eaten : Nat := 18

-- Theorem to prove the total pieces of candy Bobby ate
theorem bobby_candy_total : first_candy_eaten + second_candy_eaten = 52 := by
  sorry

end bobby_candy_total_l139_139077


namespace can_transform_1220_to_2012_cannot_transform_1220_to_2021_l139_139073

def can_transform (abcd : ℕ) (wxyz : ℕ) : Prop :=
  ∀ a b c d w x y z, 
  abcd = a*1000 + b*100 + c*10 + d ∧ 
  wxyz = w*1000 + x*100 + y*10 + z →
  (∃ (k : ℕ) (m : ℕ), 
    (k = a ∧ a ≠ d  ∧ m = c  ∧ c ≠ w ∧ 
     w = b + (k - b) ∧ x = c + (m - c)) ∨
    (k = w ∧ w ≠ x  ∧ m = y  ∧ y ≠ z ∧ 
     z = a + (k - a) ∧ x = d + (m - d)))
          
theorem can_transform_1220_to_2012 : can_transform 1220 2012 :=
sorry

theorem cannot_transform_1220_to_2021 : ¬ can_transform 1220 2021 :=
sorry

end can_transform_1220_to_2012_cannot_transform_1220_to_2021_l139_139073


namespace intersection_complement_l139_139225

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x^2 < 1 }
def B : Set ℝ := { x | x^2 - 2 * x > 0 }

theorem intersection_complement (A B : Set ℝ) : 
  (A ∩ (U \ B)) = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_complement_l139_139225


namespace find_c_l139_139112

variable (x y c : ℝ)

def condition1 : Prop := 2 * x + 5 * y = 3
def condition2 : Prop := c = Real.sqrt (4^(x + 1/2) * 32^y)

theorem find_c (h1 : condition1 x y) (h2 : condition2 x y c) : c = 4 := by
  sorry

end find_c_l139_139112


namespace cos_double_angle_l139_139348

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = 1 / 3) : 
  Real.cos (2 * α) = -7 / 9 := 
by 
  sorry

end cos_double_angle_l139_139348


namespace total_grapes_l139_139823

theorem total_grapes (r a n : ℕ) (h1 : r = 25) (h2 : a = r + 2) (h3 : n = a + 4) : r + a + n = 83 := by
  sorry

end total_grapes_l139_139823


namespace work_problem_solution_l139_139943

theorem work_problem_solution :
  (∃ C: ℝ, 
    B_work_days = 8 ∧ 
    (1 / A_work_rate + 1 / B_work_days + C = 1 / 3) ∧ 
    C = 1 / 8
  ) → 
  A_work_days = 12 :=
by
  sorry

end work_problem_solution_l139_139943


namespace number_of_scooters_l139_139492

theorem number_of_scooters (b t s : ℕ) (h1 : b + t + s = 10) (h2 : 2 * b + 3 * t + 2 * s = 26) : s = 2 := 
by sorry

end number_of_scooters_l139_139492


namespace coach_recommendation_l139_139762

def shots_A : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def shots_B : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) (mean : ℚ) : ℚ :=
  (l.map (λ x => (x - mean) ^ 2)).sum / l.length

noncomputable def recommendation (shots_A shots_B : List ℕ) : String :=
  let avg_A := average shots_A
  let avg_B := average shots_B
  let var_A := variance shots_A avg_A
  let var_B := variance shots_B avg_B
  if avg_A = avg_B ∧ var_A > var_B then "player B" else "player A"

theorem coach_recommendation : recommendation shots_A shots_B = "player B" :=
  by
  sorry

end coach_recommendation_l139_139762


namespace integer_values_of_a_l139_139880

theorem integer_values_of_a (x : ℤ) (a : ℤ)
  (h : x^3 + 3*x^2 + a*x + 11 = 0) :
  a = -155 ∨ a = -15 ∨ a = 13 ∨ a = 87 :=
sorry

end integer_values_of_a_l139_139880


namespace equation_B_is_quadratic_l139_139448

theorem equation_B_is_quadratic : ∀ y : ℝ, ∃ A B C : ℝ, (5 * y ^ 2 - 5 * y = 0) ∧ A ≠ 0 :=
by
  sorry

end equation_B_is_quadratic_l139_139448


namespace trajectory_of_midpoint_l139_139346

theorem trajectory_of_midpoint (A B P : ℝ × ℝ)
  (hA : A = (2, 4))
  (hB : ∃ m n : ℝ, B = (m, n) ∧ n^2 = 2 * m)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.2 - 2)^2 = P.1 - 1 :=
sorry

end trajectory_of_midpoint_l139_139346


namespace find_multiple_l139_139469

theorem find_multiple (x y : ℕ) (h1 : x = 11) (h2 : x + y = 55) (h3 : ∃ k m : ℕ, y = k * x + m) :
  ∃ k : ℕ, y = k * x ∧ k = 4 :=
by
  sorry

end find_multiple_l139_139469


namespace lifespan_difference_l139_139619

variable (H : ℕ)

theorem lifespan_difference (H : ℕ) (bat_lifespan : ℕ) (frog_lifespan : ℕ) (total_lifespan : ℕ) 
    (hb : bat_lifespan = 10)
    (hf : frog_lifespan = 4 * H)
    (ht : H + bat_lifespan + frog_lifespan = total_lifespan)
    (t30 : total_lifespan = 30) :
    bat_lifespan - H = 6 :=
by
  -- here would be the proof
  sorry

end lifespan_difference_l139_139619


namespace interior_angle_of_regular_hexagon_l139_139190

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l139_139190


namespace evaluate_g_l139_139766

def g (a b c d : ℤ) : ℚ := (d * (c + 2 * a)) / (c + b)

theorem evaluate_g : g 4 (-1) (-8) 2 = 0 := 
by 
  sorry

end evaluate_g_l139_139766


namespace problem_statement_l139_139685
noncomputable def f (M : ℝ) (x : ℝ) : ℝ := M * Real.sin (2 * x + Real.pi / 6)
def is_symmetric (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def is_center_of_symmetry (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop := ∀ x, f (2 * c.1 - x) = 2 * c.2 - f x

theorem problem_statement (M : ℝ) (hM : M ≠ 0) : 
    is_symmetric (f M) (2 * Real.pi / 3) ∧ 
    is_periodic (f M) Real.pi ∧ 
    is_center_of_symmetry (f M) (5 * Real.pi / 12, 0) :=
by
  sorry

end problem_statement_l139_139685


namespace at_most_one_perfect_square_l139_139198

theorem at_most_one_perfect_square (a : ℕ → ℕ) :
  (∀ n, a (n + 1) = a n ^ 3 + 103) →
  (∃ n1, ∃ n2, a n1 = k1^2 ∧ a n2 = k2^2) → n1 = n2 
    ∨ (∀ n, a n ≠ k1^2) 
    ∨ (∀ n, a n ≠ k2^2) :=
sorry

end at_most_one_perfect_square_l139_139198


namespace second_discount_is_5_percent_l139_139053

noncomputable def salePriceSecondDiscount (initialPrice finalPrice priceAfterFirstDiscount: ℝ) : ℝ :=
  (initialPrice - priceAfterFirstDiscount) + (priceAfterFirstDiscount - finalPrice)

noncomputable def secondDiscountPercentage (initialPrice finalPrice priceAfterFirstDiscount: ℝ) : ℝ :=
  (priceAfterFirstDiscount - finalPrice) / priceAfterFirstDiscount * 100

theorem second_discount_is_5_percent :
  ∀ (initialPrice finalPrice priceAfterFirstDiscount: ℝ),
    initialPrice = 600 ∧
    finalPrice = 456 ∧
    priceAfterFirstDiscount = initialPrice * 0.80 →
    secondDiscountPercentage initialPrice finalPrice priceAfterFirstDiscount = 5 :=
by
  intros
  sorry

end second_discount_is_5_percent_l139_139053


namespace complete_square_equation_l139_139668

theorem complete_square_equation (b c : ℤ) (h : (x : ℝ) → x^2 - 6 * x + 5 = (x + b)^2 - c) : b + c = 1 :=
by
  sorry  -- This is where the proof would go

end complete_square_equation_l139_139668


namespace country_math_l139_139882

theorem country_math (h : (1 / 3 : ℝ) * 4 = 6) : 
  ∃ x : ℝ, (1 / 6 : ℝ) * x = 15 ∧ x = 405 :=
by
  sorry

end country_math_l139_139882


namespace find_larger_integer_l139_139033

-- Defining the problem statement with the given conditions
theorem find_larger_integer (x : ℕ) (h : (x + 6) * 2 = 4 * x) : 4 * x = 24 :=
sorry

end find_larger_integer_l139_139033


namespace triangle_ABC_right_angle_l139_139049

def point := (ℝ × ℝ)
def line (P: point) := P.1 = 5 ∨ ∃ a: ℝ, P.1 - 5 = a * (P.2 + 2)
def parabola (P: point) := P.2 ^ 2 = 4 * P.1
def perpendicular_slopes (k1 k2: ℝ) := k1 * k2 = -1

theorem triangle_ABC_right_angle (A B C: point) (P: point) 
  (hA: A = (1, 2))
  (hP: P = (5, -2))
  (h_line: line B ∧ line C)
  (h_parabola: parabola B ∧ parabola C):
  (∃ k_AB k_AC: ℝ, perpendicular_slopes k_AB k_AC) →
  ∃k_AB k_AC: ℝ, k_AB * k_AC = -1 :=
by sorry

end triangle_ABC_right_angle_l139_139049


namespace find_number_l139_139205

theorem find_number (x : ℝ) (h : 0.30 * x = 108.0) : x = 360 := 
sorry

end find_number_l139_139205


namespace sum_of_acutes_tan_eq_pi_over_4_l139_139560

theorem sum_of_acutes_tan_eq_pi_over_4 {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
    (h : (1 + Real.tan α) * (1 + Real.tan β) = 2) : α + β = π / 4 :=
sorry

end sum_of_acutes_tan_eq_pi_over_4_l139_139560


namespace quadratic_complex_roots_condition_l139_139125

theorem quadratic_complex_roots_condition (a : ℝ) :
  (∀ a, -2 ≤ a ∧ a ≤ 2 → (a^2 < 4)) ∧ 
  ¬(∀ a, (a^2 < 4) → -2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end quadratic_complex_roots_condition_l139_139125


namespace woodworker_tables_count_l139_139688

/-- A woodworker made a total of 40 furniture legs and has built 6 chairs.
    Each chair requires 4 legs. Prove that the number of tables made is 4,
    assuming each table also requires 4 legs. -/
theorem woodworker_tables_count (total_legs chairs tables : ℕ)
  (legs_per_chair legs_per_table : ℕ)
  (H1 : total_legs = 40)
  (H2 : chairs = 6)
  (H3 : legs_per_chair = 4)
  (H4 : legs_per_table = 4)
  (H5 : total_legs = chairs * legs_per_chair + tables * legs_per_table) :
  tables = 4 := 
  sorry

end woodworker_tables_count_l139_139688


namespace tiffany_lives_after_bonus_stage_l139_139581

theorem tiffany_lives_after_bonus_stage :
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  final_lives = 768 :=
by
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  exact sorry

end tiffany_lives_after_bonus_stage_l139_139581


namespace combined_weight_l139_139954

theorem combined_weight (a b c : ℕ) (h1 : a + b = 122) (h2 : b + c = 125) (h3 : c + a = 127) : 
  a + b + c = 187 :=
by
  sorry

end combined_weight_l139_139954


namespace solve_system_l139_139046

noncomputable def system_solution (x y : ℝ) :=
  x + y = 20 ∧ x * y = 36

theorem solve_system :
  (system_solution 18 2) ∧ (system_solution 2 18) :=
  sorry

end solve_system_l139_139046


namespace find_beta_l139_139576

variable (α β : ℝ)

theorem find_beta 
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) : β = Real.pi / 3 := sorry

end find_beta_l139_139576


namespace inequality_correct_l139_139676

-- Theorem: For all real numbers x and y, if x ≥ y, then x² + y² ≥ 2xy.
theorem inequality_correct (x y : ℝ) (h : x ≥ y) : x^2 + y^2 ≥ 2 * x * y := 
by {
  -- Placeholder for the proof
  sorry
}

end inequality_correct_l139_139676


namespace cyclist_speed_ratio_l139_139517

theorem cyclist_speed_ratio (v_1 v_2 : ℝ)
  (h1 : v_1 = 2 * v_2)
  (h2 : v_1 + v_2 = 6)
  (h3 : v_1 - v_2 = 2) :
  v_1 / v_2 = 2 := 
sorry

end cyclist_speed_ratio_l139_139517


namespace equation_solutions_l139_139513

theorem equation_solutions : 
  ∀ x : ℝ, (2 * x - 1) - x * (1 - 2 * x) = 0 ↔ (x = 1 / 2 ∨ x = -1) :=
by
  intro x
  sorry

end equation_solutions_l139_139513


namespace cylinder_height_l139_139604

theorem cylinder_height (r h : ℝ) (SA : ℝ) 
  (hSA : SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h) 
  (hr : r = 3) (hSA_val : SA = 36 * Real.pi) : 
  h = 3 :=
by
  sorry

end cylinder_height_l139_139604


namespace parrot_age_is_24_l139_139975

variable (cat_age : ℝ) (rabbit_age : ℝ) (dog_age : ℝ) (parrot_age : ℝ)

def ages (cat_age rabbit_age dog_age parrot_age : ℝ) : Prop :=
  cat_age = 8 ∧
  rabbit_age = cat_age / 2 ∧
  dog_age = rabbit_age * 3 ∧
  parrot_age = cat_age + rabbit_age + dog_age

theorem parrot_age_is_24 (cat_age rabbit_age dog_age parrot_age : ℝ) :
  ages cat_age rabbit_age dog_age parrot_age → parrot_age = 24 :=
by
  intro h
  sorry

end parrot_age_is_24_l139_139975


namespace total_cost_of_gas_l139_139770

theorem total_cost_of_gas :
  ∃ x : ℚ, (4 * (x / 4) - 4 * (x / 7) = 40) ∧ x = 280 / 3 :=
by
  sorry

end total_cost_of_gas_l139_139770


namespace solve_percentage_of_X_in_B_l139_139646

variable (P : ℝ)

def liquid_X_in_A_percentage : ℝ := 0.008
def mass_of_A : ℝ := 200
def mass_of_B : ℝ := 700
def mixed_solution_percentage_of_X : ℝ := 0.0142
def target_percentage_of_P_in_B : ℝ := 0.01597

theorem solve_percentage_of_X_in_B (P : ℝ) 
  (h1 : mass_of_A * liquid_X_in_A_percentage + mass_of_B * P = (mass_of_A + mass_of_B) * mixed_solution_percentage_of_X) :
  P = target_percentage_of_P_in_B :=
sorry

end solve_percentage_of_X_in_B_l139_139646


namespace find_k_l139_139999

def S (n : ℕ) : ℤ := n^2 - 9 * n

def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem find_k (k : ℕ) (h1 : 5 < a k) (h2 : a k < 8) : k = 8 := by
  sorry

end find_k_l139_139999


namespace elena_pens_l139_139863

theorem elena_pens (X Y : ℝ) 
  (h1 : X + Y = 12) 
  (h2 : 4 * X + 2.80 * Y = 40) :
  X = 5 :=
by
  sorry

end elena_pens_l139_139863


namespace paige_folders_l139_139899

def initial_files : Nat := 135
def deleted_files : Nat := 27
def files_per_folder : Rat := 8.5
def folders_rounded_up (files_left : Nat) (per_folder : Rat) : Nat :=
  (Rat.ceil (Rat.ofInt files_left / per_folder)).toNat

theorem paige_folders :
  folders_rounded_up (initial_files - deleted_files) files_per_folder = 13 :=
by
  sorry

end paige_folders_l139_139899


namespace incorrect_average_calculated_initially_l139_139257

theorem incorrect_average_calculated_initially 
    (S : ℕ) 
    (h1 : (S + 75) / 10 = 51) 
    (h2 : (S + 25) = a) 
    : a / 10 = 46 :=
by
  sorry

end incorrect_average_calculated_initially_l139_139257


namespace find_m_l139_139318

theorem find_m (x y m : ℝ) 
  (h1 : x + y = 8)
  (h2 : y - m * x = 7)
  (h3 : y - x = 7.5) : m = 3 := 
  sorry

end find_m_l139_139318


namespace vector_simplification_l139_139007

-- Define vectors AB, CD, AC, and BD
variables {V : Type*} [AddCommGroup V]

-- Given vectors
variables (AB CD AC BD : V)

-- Theorem to be proven
theorem vector_simplification :
  (AB - CD) - (AC - BD) = (0 : V) :=
sorry

end vector_simplification_l139_139007


namespace cos_alpha_plus_beta_l139_139684

variable {α β : ℝ}
variable (sin_alpha : Real.sin α = 3/5)
variable (cos_beta : Real.cos β = 4/5)
variable (α_interval : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (β_interval : β ∈ Set.Ioo 0 (Real.pi / 2))

theorem cos_alpha_plus_beta: Real.cos (α + β) = -1 :=
by
  sorry

end cos_alpha_plus_beta_l139_139684


namespace normal_price_of_article_l139_139652

theorem normal_price_of_article (P : ℝ) (h : 0.90 * 0.80 * P = 36) : P = 50 :=
by {
  sorry
}

end normal_price_of_article_l139_139652


namespace distinct_colorings_l139_139643

def sections : ℕ := 6
def red_count : ℕ := 3
def blue_count : ℕ := 1
def green_count : ℕ := 1
def yellow_count : ℕ := 1

def permutations_without_rotation : ℕ := Nat.factorial sections / 
  (Nat.factorial red_count * Nat.factorial blue_count * Nat.factorial green_count * Nat.factorial yellow_count)

def rotational_symmetry : ℕ := permutations_without_rotation / sections

theorem distinct_colorings (rotational_symmetry) : rotational_symmetry = 20 :=
  sorry

end distinct_colorings_l139_139643


namespace f_prime_neg_one_l139_139286

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f x = f (-x)
axiom h2 : ∀ x : ℝ, f (x + 1) - f (1 - x) = 2 * x

theorem f_prime_neg_one : f' (-1) = -1 := by
  -- The proof is omitted
  sorry

end f_prime_neg_one_l139_139286


namespace foil_covered_prism_width_l139_139462

theorem foil_covered_prism_width
    (l w h : ℕ)
    (inner_volume : l * w * h = 128)
    (width_length_relation : w = 2 * l)
    (width_height_relation : w = 2 * h) :
    (w + 2) = 10 := 
sorry

end foil_covered_prism_width_l139_139462


namespace tracy_michelle_distance_ratio_l139_139081

theorem tracy_michelle_distance_ratio :
  ∀ (T M K : ℕ), 
  (M = 294) → 
  (M = 3 * K) → 
  (T + M + K = 1000) →
  ∃ x : ℕ, (T = x * M + 20) ∧ x = 2 :=
by
  intro T M K
  intro hM hMK hDistance
  use 2
  sorry

end tracy_michelle_distance_ratio_l139_139081


namespace sufficient_condition_for_sets_l139_139912

theorem sufficient_condition_for_sets (A B : Set ℝ) (m : ℝ) :
    (∀ x, x ∈ A → x ∈ B) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
    have A_def : A = {y | ∃ x, y = x^2 - (3 / 2) * x + 1 ∧ (1 / 4) ≤ x ∧ x ≤ 2} := sorry
    have B_def : B = {x | x ≥ 1 - m^2} := sorry
    sorry

end sufficient_condition_for_sets_l139_139912


namespace sum_first_9_terms_arithmetic_sequence_l139_139160

noncomputable def sum_of_first_n_terms (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

def arithmetic_sequence_term (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

theorem sum_first_9_terms_arithmetic_sequence :
  ∃ a_1 d : ℤ, (a_1 + arithmetic_sequence_term a_1 d 4 + arithmetic_sequence_term a_1 d 7 = 39) ∧
               (arithmetic_sequence_term a_1 d 3 + arithmetic_sequence_term a_1 d 6 + arithmetic_sequence_term a_1 d 9 = 27) ∧
               (sum_of_first_n_terms a_1 d 9 = 99) :=
by
  sorry

end sum_first_9_terms_arithmetic_sequence_l139_139160


namespace empty_bidon_weight_l139_139681

theorem empty_bidon_weight (B M : ℝ) 
  (h1 : B + M = 34) 
  (h2 : B + M / 2 = 17.5) : 
  B = 1 := 
by {
  -- The proof steps would go here, but we just add sorry
  sorry
}

end empty_bidon_weight_l139_139681


namespace find_x_l139_139858

theorem find_x (x : ℕ) (hx : x > 0) : 1^(x + 3) + 2^(x + 2) + 3^x + 4^(x + 1) = 1958 → x = 4 :=
sorry

end find_x_l139_139858


namespace simplify_sqrt_expression_l139_139655

theorem simplify_sqrt_expression :
  (Real.sqrt 726 / Real.sqrt 242) + (Real.sqrt 484 / Real.sqrt 121) = Real.sqrt 3 + 2 :=
by
  -- Proof goes here
  sorry

end simplify_sqrt_expression_l139_139655


namespace avg_score_calculation_l139_139916

-- Definitions based on the conditions
def directly_proportional (a b : ℝ) : Prop := ∃ k, a = k * b

variables (score_math : ℝ) (score_science : ℝ)
variables (hours_math : ℝ := 4) (hours_science : ℝ := 5)
variables (next_hours_math_science : ℝ := 5)
variables (expected_avg_score : ℝ := 97.5)

axiom h1 : directly_proportional 80 4
axiom h2 : directly_proportional 95 5

-- Define the goal: Expected average score given the study hours next time
theorem avg_score_calculation :
  (score_math / hours_math = score_science / hours_science) →
  (score_math = 100 ∧ score_science = 95) →
  ((next_hours_math_science * score_math / hours_math + next_hours_math_science * score_science / hours_science) / 2 = expected_avg_score) :=
by sorry

end avg_score_calculation_l139_139916


namespace arithmetic_sequence_15th_term_l139_139080

theorem arithmetic_sequence_15th_term :
  ∀ (a d n : ℕ), a = 3 → d = 13 - a → n = 15 → 
  a + (n - 1) * d = 143 :=
by
  intros a d n ha hd hn
  rw [ha, hd, hn]
  sorry

end arithmetic_sequence_15th_term_l139_139080


namespace add_base8_l139_139523

-- Define the base 8 numbers 5_8 and 16_8
def five_base8 : ℕ := 5
def sixteen_base8 : ℕ := 1 * 8 + 6

-- Convert the result to base 8 from the sum in base 10
def sum_base8 (a b : ℕ) : ℕ :=
  let sum_base10 := a + b
  let d1 := sum_base10 / 8
  let d0 := sum_base10 % 8
  d1 * 10 + d0 

theorem add_base8 (x y : ℕ) (hx : x = five_base8) (hy : y = sixteen_base8) :
  sum_base8 x y = 23 :=
by
  sorry

end add_base8_l139_139523


namespace right_triangle_perimeter_l139_139373

theorem right_triangle_perimeter (n : ℕ) (hn : Nat.Prime n) (x y : ℕ) 
  (h1 : y^2 = x^2 + n^2) : n + x + y = n + n^2 := by
  sorry

end right_triangle_perimeter_l139_139373


namespace maximum_quadratic_expr_l139_139834

noncomputable def quadratic_expr (x : ℝ) : ℝ :=
  -5 * x^2 + 25 * x - 7

theorem maximum_quadratic_expr : ∃ x : ℝ, quadratic_expr x = 53 / 4 :=
by
  sorry

end maximum_quadratic_expr_l139_139834


namespace determine_k_coplanar_l139_139497

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable {A B C D : V}
variable (k : ℝ)

theorem determine_k_coplanar (h : 4 • A - 3 • B + 6 • C + k • D = 0) : k = -13 :=
sorry

end determine_k_coplanar_l139_139497


namespace number_of_coaches_l139_139591

theorem number_of_coaches (r : ℕ) (v : ℕ) (c : ℕ) (h1 : r = 60) (h2 : v = 3) (h3 : c * 5 = 60 * 3) : c = 36 :=
by
  -- We skip the proof as per instructions
  sorry

end number_of_coaches_l139_139591


namespace squares_triangles_product_l139_139244

theorem squares_triangles_product :
  let S := 7
  let T := 10
  S * T = 70 :=
by
  let S := 7
  let T := 10
  show (S * T = 70)
  sorry

end squares_triangles_product_l139_139244


namespace find_ordered_triple_l139_139069

theorem find_ordered_triple (a b c : ℝ) (h1 : a > 2) (h2 : b > 2) (h3 : c > 2)
  (h4 : (a + 1)^2 / (b + c - 1) + (b + 3)^2 / (c + a - 3) + (c + 5)^2 / (a + b - 5) = 27) :
  (a, b, c) = (9, 7, 2) :=
by sorry

end find_ordered_triple_l139_139069


namespace clock_shows_l139_139216

-- Definitions for the hands and their positions
variables {A B C : ℕ} -- Representing hands A, B, and C as natural numbers for simplicity

-- Conditions based on the problem description:
-- 1. Hands A and B point exactly at the hour markers.
-- 2. Hand C is slightly off from an hour marker.
axiom hand_A_hour_marker : A % 12 = A
axiom hand_B_hour_marker : B % 12 = B
axiom hand_C_slightly_off : C % 12 ≠ C

-- Theorem stating that given these conditions, the clock shows the time 4:50
theorem clock_shows (h1: A % 12 = A) (h2: B % 12 = B) (h3: C % 12 ≠ C) : A = 50 ∧ B = 12 ∧ C = 4 :=
sorry

end clock_shows_l139_139216


namespace total_amount_paid_l139_139478

-- Define the given conditions
def q_g : ℕ := 9        -- Quantity of grapes
def r_g : ℕ := 70       -- Rate per kg of grapes
def q_m : ℕ := 9        -- Quantity of mangoes
def r_m : ℕ := 55       -- Rate per kg of mangoes

-- Define the total amount paid calculation and prove it equals 1125
theorem total_amount_paid : (q_g * r_g + q_m * r_m) = 1125 :=
by
  -- Proof will be provided here. Currently using 'sorry' to skip it.
  sorry

end total_amount_paid_l139_139478


namespace workers_allocation_l139_139972

-- Definitions based on conditions
def num_workers := 90
def bolt_per_worker := 15
def nut_per_worker := 24
def bolt_matching_requirement := 2

-- Statement of the proof problem
theorem workers_allocation (x y : ℕ) :
  x + y = num_workers ∧
  bolt_matching_requirement * bolt_per_worker * x = nut_per_worker * y →
  x = 40 ∧ y = 50 :=
by
  sorry

end workers_allocation_l139_139972


namespace find_x_l139_139451

-- Define the conditions as hypotheses
def problem_statement (x : ℤ) : Prop :=
  (3 * x > 30) ∧ (x ≥ 10) ∧ (x > 5) ∧ 
  (x = 9)

-- Define the theorem statement
theorem find_x : ∃ x : ℤ, problem_statement x :=
by
  -- Sorry to skip proof as instructed
  sorry

end find_x_l139_139451


namespace arithmetic_sequence_common_difference_l139_139110

theorem arithmetic_sequence_common_difference (a_1 a_5 d : ℝ) 
  (h1 : a_5 = a_1 + 4 * d) 
  (h2 : a_1 + (a_1 + d) + (a_1 + 2 * d) = 6) : 
  d = 2 := 
  sorry

end arithmetic_sequence_common_difference_l139_139110


namespace determine_g_function_l139_139550

theorem determine_g_function (t x : ℝ) (g : ℝ → ℝ) 
  (line_eq : ∀ x y : ℝ, y = 2 * x - 40) 
  (param_eq : ∀ t : ℝ, (x, 20 * t - 14) = (g t, 20 * t - 14)) :
  g t = 10 * t + 13 :=
by 
  sorry

end determine_g_function_l139_139550


namespace ticket_price_l139_139683

theorem ticket_price (regular_price : ℕ) (discount_percent : ℕ) (paid_price : ℕ) 
  (h1 : regular_price = 15) 
  (h2 : discount_percent = 40) 
  (h3 : paid_price = regular_price - (regular_price * discount_percent / 100)) : 
  paid_price = 9 :=
by
  sorry

end ticket_price_l139_139683


namespace setC_not_pythagorean_l139_139352

/-- Defining sets of numbers as options -/
def SetA := (3, 4, 5)
def SetB := (5, 12, 13)
def SetC := (7, 25, 26)
def SetD := (6, 8, 10)

/-- Function to check if a set is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem stating set C is not a Pythagorean triple -/
theorem setC_not_pythagorean :
  ¬isPythagoreanTriple 7 25 26 :=
by {
  -- This slot will be filled with the concrete proof steps in Lean.
  sorry
}

end setC_not_pythagorean_l139_139352


namespace sum_xyz_l139_139851

theorem sum_xyz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + 2 * (y - 1) * (z - 1) = 85)
  (h2 : y^2 + 2 * (z - 1) * (x - 1) = 84)
  (h3 : z^2 + 2 * (x - 1) * (y - 1) = 89) :
  x + y + z = 18 := 
by
  sorry

end sum_xyz_l139_139851


namespace minimize_fencing_l139_139575

def area_requirement (w : ℝ) : Prop :=
  2 * (w * w) ≥ 800

def length_twice_width (l w : ℝ) : Prop :=
  l = 2 * w

def perimeter (w l : ℝ) : ℝ :=
  2 * l + 2 * w

theorem minimize_fencing (w l : ℝ) (h1 : area_requirement w) (h2 : length_twice_width l w) :
  w = 20 ∧ l = 40 :=
by
  sorry

end minimize_fencing_l139_139575


namespace total_tissues_l139_139571

-- define the number of students in each group
def g1 : Nat := 9
def g2 : Nat := 10
def g3 : Nat := 11

-- define the number of tissues per mini tissue box
def t : Nat := 40

-- state the main theorem
theorem total_tissues : (g1 + g2 + g3) * t = 1200 := by
  sorry

end total_tissues_l139_139571


namespace evaluate_expression_l139_139269

theorem evaluate_expression (a x : ℤ) (h : x = a + 7) : x - a + 3 = 10 := by
  sorry

end evaluate_expression_l139_139269


namespace apples_difference_l139_139230

-- Definitions for initial and remaining apples
def initial_apples : ℕ := 46
def remaining_apples : ℕ := 14

-- The theorem to prove the difference between initial and remaining apples is 32
theorem apples_difference : initial_apples - remaining_apples = 32 := by
  -- proof is omitted
  sorry

end apples_difference_l139_139230


namespace find_x_tan_identity_l139_139627

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l139_139627


namespace jane_spent_more_on_ice_cream_l139_139758

-- Definitions based on the conditions
def ice_cream_cone_cost : ℕ := 5
def pudding_cup_cost : ℕ := 2
def ice_cream_cones_bought : ℕ := 15
def pudding_cups_bought : ℕ := 5

-- The mathematically equivalent proof statement
theorem jane_spent_more_on_ice_cream : 
  (ice_cream_cones_bought * ice_cream_cone_cost - pudding_cups_bought * pudding_cup_cost) = 65 := 
by
  sorry

end jane_spent_more_on_ice_cream_l139_139758


namespace expand_expression_l139_139438

theorem expand_expression : ∀ (x : ℝ), (20 * x - 25) * 3 * x = 60 * x^2 - 75 * x := 
by
  intro x
  sorry

end expand_expression_l139_139438


namespace cauchy_schwarz_inequality_l139_139455

theorem cauchy_schwarz_inequality
  (x1 y1 z1 x2 y2 z2 : ℝ) :
  (x1 * x2 + y1 * y2 + z1 * z2) ^ 2 ≤ (x1 ^ 2 + y1 ^ 2 + z1 ^ 2) * (x2 ^ 2 + y2 ^ 2 + z2 ^ 2) := 
sorry

end cauchy_schwarz_inequality_l139_139455


namespace sum_of_first_four_terms_of_arithmetic_sequence_l139_139504

theorem sum_of_first_four_terms_of_arithmetic_sequence
  (a d : ℤ)
  (h1 : a + 4 * d = 10)  -- Condition for the fifth term
  (h2 : a + 5 * d = 14)  -- Condition for the sixth term
  (h3 : a + 6 * d = 18)  -- Condition for the seventh term
  : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 0 :=  -- Prove the sum of the first four terms is 0
by
  sorry

end sum_of_first_four_terms_of_arithmetic_sequence_l139_139504


namespace floor_sqrt_50_l139_139483

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l139_139483


namespace central_projection_intersect_l139_139247

def central_projection (lines : Set (Set Point)) : Prop :=
  ∃ point : Point, ∀ line ∈ lines, line (point)

theorem central_projection_intersect :
  ∀ lines : Set (Set Point), central_projection lines → ∃ point : Point, ∀ line ∈ lines, line (point) :=
by
  sorry

end central_projection_intersect_l139_139247


namespace average_speed_of_trip_l139_139514

noncomputable def total_distance (d1 d2 : ℝ) : ℝ :=
  d1 + d2

noncomputable def travel_time (distance speed : ℝ) : ℝ :=
  distance / speed

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem average_speed_of_trip :
  let d1 := 60
  let s1 := 20
  let d2 := 120
  let s2 := 60
  let total_d := total_distance d1 d2
  let time1 := travel_time d1 s1
  let time2 := travel_time d2 s2
  let total_t := time1 + time2
  average_speed total_d total_t = 36 :=
by
  sorry

end average_speed_of_trip_l139_139514


namespace empty_tank_time_l139_139045

-- Definitions based on problem conditions
def tank_full_fraction := 1 / 5
def pipeA_fill_time := 15
def pipeB_empty_time := 6

-- Derived definitions
def rate_of_pipeA := 1 / pipeA_fill_time
def rate_of_pipeB := 1 / pipeB_empty_time
def combined_rate := rate_of_pipeA - rate_of_pipeB 

-- The time to empty the tank when both pipes are open
def time_to_empty (initial_fraction : ℚ) (combined_rate : ℚ) : ℚ :=
  initial_fraction / -combined_rate

-- The main theorem to prove
theorem empty_tank_time
  (initial_fraction : ℚ := tank_full_fraction)
  (combined_rate : ℚ := combined_rate)
  (time : ℚ := time_to_empty initial_fraction combined_rate) :
  time = 2 :=
by
  sorry

end empty_tank_time_l139_139045


namespace reflections_composition_rotation_l139_139275

variable {α : ℝ} -- defining the angle α
variable {O : ℝ × ℝ} -- defining the point O, assuming the plane is represented as ℝ × ℝ

-- Define the lines that form the sides of the angle
variable (L1 L2 : ℝ × ℝ → Prop)

-- Assume α is the angle between L1 and L2 with O as the vertex
variable (hL1 : (L1 O))
variable (hL2 : (L2 O))

-- Assume reflections across L1 and L2
def reflect (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

theorem reflections_composition_rotation :
  ∀ A : ℝ × ℝ, (reflect (reflect A L1) L2) = sorry := 
sorry

end reflections_composition_rotation_l139_139275


namespace peaches_in_each_basket_l139_139404

variable (R : ℕ)

theorem peaches_in_each_basket (h : 6 * R = 96) : R = 16 :=
by
  sorry

end peaches_in_each_basket_l139_139404


namespace unique_solution_l139_139398

noncomputable def valid_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + (2 - a) * x + 1 = 0 ∧ -1 < x ∧ x ≤ 3 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2

theorem unique_solution (a : ℝ) :
  (valid_solutions a) ↔ (a = 4.5 ∨ (a < 0) ∨ (a > 16 / 3)) := 
sorry

end unique_solution_l139_139398


namespace parabola_distance_to_y_axis_l139_139489

theorem parabola_distance_to_y_axis :
  ∀ (M : ℝ × ℝ), (M.2 ^ 2 = 4 * M.1) → 
  dist (M, (1, 0)) = 10 →
  abs (M.1) = 9 :=
by
  intros M hParabola hDist
  sorry

end parabola_distance_to_y_axis_l139_139489


namespace proof_prob_at_least_one_die_3_or_5_l139_139877

def probability_at_least_one_die_3_or_5 (total_outcomes : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

theorem proof_prob_at_least_one_die_3_or_5 :
  let total_outcomes := 36
  let favorable_outcomes := 20
  probability_at_least_one_die_3_or_5 total_outcomes favorable_outcomes = 5 / 9 := 
by 
  sorry

end proof_prob_at_least_one_die_3_or_5_l139_139877


namespace greatest_value_l139_139188

theorem greatest_value (y : ℝ) (h : 4 * y^2 + 4 * y + 3 = 1) : (y + 1)^2 = 1/4 :=
sorry

end greatest_value_l139_139188


namespace exponent_property_l139_139814

theorem exponent_property (a : ℝ) (m n : ℝ) (h₁ : a^m = 4) (h₂ : a^n = 8) : a^(m + n) = 32 := 
by 
  sorry

end exponent_property_l139_139814


namespace sum_of_values_of_m_l139_139109

-- Define the inequality conditions
def condition1 (x m : ℝ) : Prop := (x - m) / 2 ≥ 0
def condition2 (x : ℝ) : Prop := x + 3 < 3 * (x - 1)

-- Define the equation constraint for y
def fractional_equation (y m : ℝ) : Prop := (3 - y) / (2 - y) + m / (y - 2) = 3

-- Sum function for the values of m
def sum_of_m (m1 m2 m3 : ℝ) : ℝ := m1 + m2 + m3

-- Main theorem
theorem sum_of_values_of_m : sum_of_m 3 (-3) (-1) = -1 := 
by { sorry }

end sum_of_values_of_m_l139_139109


namespace cubic_expression_l139_139187

theorem cubic_expression (x : ℝ) (hx : x + 1/x = -7) : x^3 + 1/x^3 = -322 :=
by sorry

end cubic_expression_l139_139187


namespace part1_part2_l139_139597

theorem part1 (x y : ℕ) (h1 : 25 * x + 30 * y = 1500) (h2 : x = 2 * y - 4) : x = 36 ∧ y = 20 :=
by
  sorry

theorem part2 (x y : ℕ) (h1 : x + y = 60) (h2 : x ≥ 2 * y)
  (h_profit : ∃ p, p = 7 * x + 10 * y) : 
  ∃ x y profit, x = 40 ∧ y = 20 ∧ profit = 480 :=
by
  sorry

end part1_part2_l139_139597


namespace second_term_of_geometric_series_l139_139532

theorem second_term_of_geometric_series (a r S: ℝ) (h_r : r = 1/4) (h_S : S = 40) (h_geom_sum : S = a / (1 - r)) : a * r = 7.5 :=
by
  sorry

end second_term_of_geometric_series_l139_139532


namespace max_g_eq_25_l139_139266

-- Define the function g on positive integers.
def g : ℕ → ℤ
| n => if n < 12 then n + 14 else g (n - 7)

-- Prove that the maximum value of g is 25.
theorem max_g_eq_25 : ∀ n : ℕ, 1 ≤ n → g n ≤ 25 ∧ (∃ n : ℕ, 1 ≤ n ∧ g n = 25) := by
  sorry

end max_g_eq_25_l139_139266


namespace worker_new_wage_after_increase_l139_139651

theorem worker_new_wage_after_increase (initial_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) 
  (h1 : initial_wage = 34) (h2 : increase_percentage = 0.50) 
  (h3 : new_wage = initial_wage + (increase_percentage * initial_wage)) : new_wage = 51 := 
by
  sorry

end worker_new_wage_after_increase_l139_139651


namespace find_s_l139_139022

theorem find_s (x y : Real -> Real) : 
  (x 2 = 2 ∧ y 2 = 5) ∧ 
  (x 6 = 6 ∧ y 6 = 17) ∧ 
  (x 10 = 10 ∧ y 10 = 29) ∧ 
  (∀ x, y x = 3 * x - 1) -> 
  (y 34 = 101) := 
by 
  sorry

end find_s_l139_139022


namespace lines_parallel_if_perpendicular_to_same_plane_l139_139296

-- Define a plane as a placeholder for other properties
axiom Plane : Type
-- Define Line as a placeholder for other properties
axiom Line : Type

-- Definition of what it means for a line to be perpendicular to a plane
axiom perpendicular_to_plane (l : Line) (π : Plane) : Prop

-- Definition of parallel lines
axiom parallel_lines (l1 l2 : Line) : Prop

-- Define the proof problem in Lean 4
theorem lines_parallel_if_perpendicular_to_same_plane
    (π : Plane) (l1 l2 : Line)
    (h1 : perpendicular_to_plane l1 π)
    (h2 : perpendicular_to_plane l2 π) :
    parallel_lines l1 l2 :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l139_139296


namespace repeated_root_condition_l139_139278

theorem repeated_root_condition (m : ℝ) : m = 10 → ∃ x, (5 * x) / (x - 2) + 1 = m / (x - 2) ∧ x = 2 :=
by
  sorry

end repeated_root_condition_l139_139278


namespace factorize_expr_l139_139138

theorem factorize_expr (x y : ℝ) : 3 * x^2 + 6 * x * y + 3 * y^2 = 3 * (x + y)^2 := 
  sorry

end factorize_expr_l139_139138


namespace vitamin_D_scientific_notation_l139_139350

def scientific_notation (x : ℝ) (m : ℝ) (n : ℤ) : Prop :=
  x = m * 10^n

theorem vitamin_D_scientific_notation :
  scientific_notation 0.0000046 4.6 (-6) :=
by {
  sorry
}

end vitamin_D_scientific_notation_l139_139350


namespace range_of_numbers_l139_139023

theorem range_of_numbers (a b c : ℕ) (h_mean : (a + b + c) / 3 = 4) (h_median : b = 4) (h_smallest : a = 1) :
  c - a = 6 :=
sorry

end range_of_numbers_l139_139023


namespace bob_total_profit_l139_139867

def initial_cost (num_dogs : ℕ) (cost_per_dog : ℕ) : ℕ := num_dogs * cost_per_dog

def revenue (num_puppies : ℕ) (price_per_puppy : ℕ) : ℕ := num_puppies * price_per_puppy

def total_profit (initial_cost : ℕ) (revenue : ℕ) : ℕ := revenue - initial_cost

theorem bob_total_profit (c1 : initial_cost 2 250 = 500)
                        (c2 : revenue 6 350 = 2100)
                        (c3 : total_profit 500 2100 = 1600) :
  total_profit (initial_cost 2 250) (revenue 6 350) = 1600 := by
  sorry

end bob_total_profit_l139_139867


namespace thor_fraction_correct_l139_139845

-- Define the initial conditions
def moes_money : ℕ := 12
def lokis_money : ℕ := 10
def nicks_money : ℕ := 8
def otts_money : ℕ := 6

def thor_received_from_each : ℕ := 2

-- Calculate total money each time
def total_initial_money : ℕ := moes_money + lokis_money + nicks_money + otts_money
def thor_total_received : ℕ := 4 * thor_received_from_each
def thor_fraction_of_total : ℚ := thor_total_received / total_initial_money

-- The theorem to prove
theorem thor_fraction_correct : thor_fraction_of_total = 2/9 :=
by
  sorry

end thor_fraction_correct_l139_139845


namespace sum_of_areas_l139_139441

theorem sum_of_areas :
  (∑' n : ℕ, Real.pi * (1 / 9 ^ n)) = (9 * Real.pi) / 8 :=
by
  sorry

end sum_of_areas_l139_139441


namespace union_of_S_and_T_l139_139546

-- Definitions of the sets S and T
def S : Set ℝ := { y | ∃ x : ℝ, y = Real.exp x - 2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

-- Lean proof problem statement
theorem union_of_S_and_T : (S ∪ T) = { y | -4 ≤ y } :=
by
  sorry

end union_of_S_and_T_l139_139546


namespace find_a_if_f_is_odd_function_l139_139947

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (a * 2^x - 2^(-x))

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_a_if_f_is_odd_function : 
  ∀ a : ℝ, is_odd_function (f a) → a = 1 :=
by
  sorry

end find_a_if_f_is_odd_function_l139_139947


namespace quadratic_inequality_empty_solution_set_l139_139137

theorem quadratic_inequality_empty_solution_set (a b c : ℝ) (hₐ : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0 → False) ↔ (a > 0 ∧ (b^2 - 4 * a * c) ≤ 0) :=
by 
  sorry

end quadratic_inequality_empty_solution_set_l139_139137


namespace min_B_minus_A_l139_139849

noncomputable def S_n (n : ℕ) : ℚ :=
  let a1 : ℚ := 2
  let r : ℚ := -1 / 3
  a1 * (1 - r ^ n) / (1 - r)

theorem min_B_minus_A :
  ∃ A B : ℚ, 
    (∀ n : ℕ, 1 ≤ n → A ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B) ∧
    ∀ A' B' : ℚ, 
      (∀ n : ℕ, 1 ≤ n → A' ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B') → 
      B' - A' ≥ 9 / 4 ∧ B - A = 9 / 4 :=
sorry

end min_B_minus_A_l139_139849


namespace dodecahedron_interior_diagonals_l139_139587

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l139_139587


namespace complete_the_square_l139_139058

theorem complete_the_square (z : ℤ) : 
    z^2 - 6*z + 17 = (z - 3)^2 + 8 :=
sorry

end complete_the_square_l139_139058


namespace train_speed_is_60_kmph_l139_139322

-- Define the conditions
def time_to_cross_pole_seconds : ℚ := 36
def length_of_train_meters : ℚ := 600

-- Define the conversion factors
def seconds_per_hour : ℚ := 3600
def meters_per_kilometer : ℚ := 1000

-- Convert the conditions to appropriate units
def time_to_cross_pole_hours : ℚ := time_to_cross_pole_seconds / seconds_per_hour
def length_of_train_kilometers : ℚ := length_of_train_meters / meters_per_kilometer

-- Prove that the speed of the train in km/hr is 60
theorem train_speed_is_60_kmph : 
  (length_of_train_kilometers / time_to_cross_pole_hours) = 60 := 
by
  sorry

end train_speed_is_60_kmph_l139_139322


namespace mark_hours_per_week_l139_139892

theorem mark_hours_per_week (w_historical : ℕ) (w_spring : ℕ) (h_spring : ℕ) (e_spring : ℕ) (e_goal : ℕ) (w_goal : ℕ) (h_goal : ℚ) :
  (e_spring : ℚ) / (w_historical * w_spring) = h_spring / w_spring →
  e_goal = 21000 →
  w_goal = 50 →
  h_spring = 35 →
  w_spring = 15 →
  e_spring = 4200 →
  (h_goal : ℚ) = 2625 / w_goal →
  h_goal = 52.5 :=
sorry

end mark_hours_per_week_l139_139892


namespace volume_of_mixture_l139_139631

theorem volume_of_mixture
    (weight_a : ℝ) (weight_b : ℝ) (ratio_a_b : ℝ) (total_weight : ℝ)
    (h1 : weight_a = 900) (h2 : weight_b = 700)
    (h3 : ratio_a_b = 3/2) (h4 : total_weight = 3280) :
    ∃ Va Vb : ℝ, (Va / Vb = ratio_a_b) ∧ (weight_a * Va + weight_b * Vb = total_weight) ∧ (Va + Vb = 4) := 
by
  sorry

end volume_of_mixture_l139_139631


namespace lcm_1404_972_l139_139957

def num1 := 1404
def num2 := 972

theorem lcm_1404_972 : Nat.lcm num1 num2 = 88452 := 
by 
  sorry

end lcm_1404_972_l139_139957


namespace count_prime_boring_lt_10000_l139_139801

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_boring (n : ℕ) : Prop := 
  let digits := n.digits 10
  match digits with
  | [] => false
  | (d::ds) => ds.all (fun x => x = d)

theorem count_prime_boring_lt_10000 : 
  ∃! n, is_prime n ∧ is_boring n ∧ n < 10000 := 
by 
  sorry

end count_prime_boring_lt_10000_l139_139801


namespace caterer_min_people_l139_139201

theorem caterer_min_people (x : ℕ) : 150 + 18 * x > 250 + 15 * x → x ≥ 34 :=
by
  intro h
  sorry

end caterer_min_people_l139_139201


namespace range_of_k_l139_139754

noncomputable def e := Real.exp 1

theorem range_of_k (k : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ e ^ (x1 - 1) = |k * x1| ∧ e ^ (x2 - 1) = |k * x2| ∧ e ^ (x3 - 1) = |k * x3|) : k^2 > 1 := sorry

end range_of_k_l139_139754


namespace solve_for_a_l139_139616

theorem solve_for_a (a y x : ℝ)
  (h1 : y = 5 * a)
  (h2 : x = 2 * a - 2)
  (h3 : y + 3 = x) :
  a = -5 / 3 :=
by
  sorry

end solve_for_a_l139_139616


namespace sufficient_not_necessary_l139_139156

theorem sufficient_not_necessary (x : ℝ) : (x - 1 > 0) → (x^2 - 1 > 0) ∧ ¬((x^2 - 1 > 0) → (x - 1 > 0)) :=
by
  sorry

end sufficient_not_necessary_l139_139156


namespace xiaoyangs_scores_l139_139844

theorem xiaoyangs_scores (average : ℕ) (diff : ℕ) (h_average : average = 96) (h_diff : diff = 8) :
  ∃ chinese_score math_score : ℕ, chinese_score = 92 ∧ math_score = 100 :=
by
  sorry

end xiaoyangs_scores_l139_139844


namespace probability_of_picking_same_color_shoes_l139_139622

theorem probability_of_picking_same_color_shoes
  (n_pairs_black : ℕ) (n_pairs_brown : ℕ) (n_pairs_gray : ℕ)
  (h_black_pairs : n_pairs_black = 8)
  (h_brown_pairs : n_pairs_brown = 4)
  (h_gray_pairs : n_pairs_gray = 3)
  (total_shoes : ℕ := 2 * (n_pairs_black + n_pairs_brown + n_pairs_gray)) :
  (16 / total_shoes * 8 / (total_shoes - 1) + 
   8 / total_shoes * 4 / (total_shoes - 1) + 
   6 / total_shoes * 3 / (total_shoes - 1)) = 89 / 435 :=
by
  sorry

end probability_of_picking_same_color_shoes_l139_139622


namespace macy_miles_left_to_run_l139_139066

-- Define the given conditions
def goal : ℕ := 24
def miles_per_day : ℕ := 3
def days : ℕ := 6

-- Define the statement to be proven
theorem macy_miles_left_to_run :
  goal - (miles_per_day * days) = 6 :=
by
  sorry

end macy_miles_left_to_run_l139_139066


namespace tangent_ellipse_hyperbola_l139_139253

theorem tangent_ellipse_hyperbola (m : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y + 3)^2 = 4) → m = 5 / 9 :=
by
  sorry

end tangent_ellipse_hyperbola_l139_139253


namespace largest_even_number_l139_139326

theorem largest_even_number (x : ℤ) 
  (h : x + (x + 2) + (x + 4) = x + 18) : x + 4 = 10 :=
by
  sorry

end largest_even_number_l139_139326


namespace daily_sale_correct_l139_139121

-- Define the original and additional amounts in kilograms
def original_rice := 4 * 1000 -- 4 tons converted to kilograms
def additional_rice := 4000 -- kilograms
def total_rice := original_rice + additional_rice -- total amount of rice in kilograms
def days := 4 -- days to sell all the rice

-- Statement to prove: The amount to be sold each day
def daily_sale_amount := 2000 -- kilograms per day

theorem daily_sale_correct : total_rice / days = daily_sale_amount :=
by 
  -- This is a placeholder for the proof
  sorry

end daily_sale_correct_l139_139121


namespace shirt_cost_l139_139214

-- Definitions and conditions
def num_ten_bills : ℕ := 2
def num_twenty_bills : ℕ := num_ten_bills + 1

def ten_bill_value : ℕ := 10
def twenty_bill_value : ℕ := 20

-- Statement to prove
theorem shirt_cost :
  (num_ten_bills * ten_bill_value) + (num_twenty_bills * twenty_bill_value) = 80 :=
by
  sorry

end shirt_cost_l139_139214


namespace hypotenuse_length_l139_139966

theorem hypotenuse_length (x a b: ℝ) (h1: a = 7) (h2: b = x - 1) (h3: a^2 + b^2 = x^2) : x = 25 :=
by {
  -- Condition h1 states that one leg 'a' is 7 cm.
  -- Condition h2 states that the other leg 'b' is 1 cm shorter than the hypotenuse 'x', i.e., b = x - 1.
  -- Condition h3 is derived from the Pythagorean theorem, i.e., a^2 + b^2 = x^2.
  -- We need to prove that x = 25 cm.
  sorry
}

end hypotenuse_length_l139_139966


namespace polar_to_rectangular_coordinates_l139_139209

theorem polar_to_rectangular_coordinates (r θ : ℝ) (hr : r = 5) (hθ : θ = (3 * Real.pi) / 2) :
    (r * Real.cos θ, r * Real.sin θ) = (0, -5) :=
by
  rw [hr, hθ]
  simp [Real.cos, Real.sin]
  sorry

end polar_to_rectangular_coordinates_l139_139209


namespace no_stew_left_l139_139093

theorem no_stew_left (company : Type) (stew : ℝ)
    (one_third_stayed : ℝ)
    (two_thirds_went : ℝ)
    (camp_consumption : ℝ)
    (range_consumption_per_portion : ℝ)
    (range_portion_multiplier : ℝ)
    (total_stew : ℝ) : 
    one_third_stayed = 1 / 3 →
    two_thirds_went = 2 / 3 →
    camp_consumption = 1 / 4 →
    range_portion_multiplier = 1.5 →
    total_stew = camp_consumption + (range_portion_multiplier * (two_thirds_went * (camp_consumption / one_third_stayed))) →
    total_stew = 1 →
    stew = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- here would be the proof steps
  sorry

end no_stew_left_l139_139093


namespace factorization_of_x_squared_minus_nine_l139_139088

theorem factorization_of_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by
  sorry

end factorization_of_x_squared_minus_nine_l139_139088


namespace find_m_collinear_l139_139642

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isCollinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

theorem find_m_collinear :
  ∀ (m : ℝ),
  let A := Point.mk (-2) 3
  let B := Point.mk 3 (-2)
  let C := Point.mk (1 / 2) m
  isCollinear A B C → m = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end find_m_collinear_l139_139642


namespace cos_of_angle_l139_139365

theorem cos_of_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.cos (3 * Real.pi / 2 + 2 * θ) = 3 / 5 := 
by
  sorry

end cos_of_angle_l139_139365


namespace max_value_f1_on_interval_range_of_a_g_increasing_l139_139458

noncomputable def f1 (x : ℝ) : ℝ := 2 * x^2 + x + 2

theorem max_value_f1_on_interval : 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → f1 x ≤ 5) ∧ 
  (∃ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) ∧ f1 x = 5) :=
sorry

noncomputable def f2 (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Set.Icc (1 : ℝ) (2 : ℝ) → f2 a x / x ≥ 2) → a ≥ 1 :=
sorry

noncomputable def g (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a + (1 - (a-1) * x^2) / x

theorem g_increasing (a : ℝ) : 
  (∀ x1 x2, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → g a x1 < g a x2) → a ≥ 1 / 16 :=
sorry

end max_value_f1_on_interval_range_of_a_g_increasing_l139_139458


namespace proof_problem_l139_139363

-- Definitions of parallel and perpendicular relationships for lines and planes
def parallel (α β : Type) : Prop := sorry
def perpendicular (α β : Type) : Prop := sorry
def contained_in (m : Type) (α : Type) : Prop := sorry

-- Variables representing lines and planes
variables (l m n : Type) (α β : Type)

-- Assumptions from the conditions in step a)
variables 
  (h1 : m ≠ l)
  (h2 : α ≠ β)
  (h3 : parallel m n)
  (h4 : perpendicular m α)
  (h5 : perpendicular n β)

-- The goal is to prove that the planes α and β are parallel under the given conditions
theorem proof_problem : parallel α β :=
sorry

end proof_problem_l139_139363


namespace tan_neg_five_pi_div_four_l139_139400

theorem tan_neg_five_pi_div_four : Real.tan (- (5 * Real.pi / 4)) = -1 := 
sorry

end tan_neg_five_pi_div_four_l139_139400


namespace min_value_expression_l139_139822

theorem min_value_expression (x y : ℝ) : ∃ (a b : ℝ), x = a ∧ y = b ∧ (x^2 + y^2 - 8*x - 6*y + 30 = 5) :=
by
  sorry

end min_value_expression_l139_139822


namespace greatest_prime_factor_of_15_l139_139378

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l139_139378


namespace range_of_a_l139_139678

variable (a : ℝ)

def a_n (n : ℕ) : ℝ :=
if n = 1 then a else 4 * ↑n + (-1 : ℝ) ^ n * (8 - 2 * a)

theorem range_of_a (h : ∀ n : ℕ, n > 0 → a_n a n < a_n a (n + 1)) : 3 < a ∧ a < 5 :=
by
  sorry

end range_of_a_l139_139678


namespace fill_time_with_leak_is_correct_l139_139752

-- Define the conditions
def time_to_fill_without_leak := 8
def time_to_empty_with_leak := 24

-- Define the rates
def fill_rate := 1 / time_to_fill_without_leak
def leak_rate := 1 / time_to_empty_with_leak
def effective_fill_rate := fill_rate - leak_rate

-- Prove the time to fill with leak
def time_to_fill_with_leak := 1 / effective_fill_rate

-- The theorem to prove that the time is 12 hours
theorem fill_time_with_leak_is_correct :
  time_to_fill_with_leak = 12 := by
  simp [time_to_fill_without_leak, time_to_empty_with_leak, fill_rate, leak_rate, effective_fill_rate, time_to_fill_with_leak]
  sorry

end fill_time_with_leak_is_correct_l139_139752


namespace function_value_at_minus_two_l139_139731

theorem function_value_at_minus_two {f : ℝ → ℝ} (h : ∀ x : ℝ, x ≠ 0 → f (1/x) + (1/x) * f (-x) = 2 * x) : f (-2) = 7 / 2 :=
sorry

end function_value_at_minus_two_l139_139731


namespace largest_three_digit_perfect_square_and_cube_l139_139956

theorem largest_three_digit_perfect_square_and_cube :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a : ℕ), n = a^6) ∧ ∀ (m : ℕ), ((100 ≤ m ∧ m ≤ 999) ∧ (∃ (b : ℕ), m = b^6)) → m ≤ n := 
by 
  sorry

end largest_three_digit_perfect_square_and_cube_l139_139956


namespace solve_system_of_equations_l139_139006

theorem solve_system_of_equations :
  ∀ x y z : ℝ,
  (3 * x * y - 5 * y * z - x * z = 3 * y) →
  (x * y + y * z = -y) →
  (-5 * x * y + 4 * y * z + x * z = -4 * y) →
  (x = 2 ∧ y = -1 / 3 ∧ z = -3) ∨ 
  (y = 0 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 0) :=
by
  sorry

end solve_system_of_equations_l139_139006


namespace complex_expression_l139_139315

theorem complex_expression (z : ℂ) (i : ℂ) (h1 : z^2 + 1 = 0) (h2 : i^2 = -1) : 
  (z^4 + i) * (z^4 - i) = 0 :=
sorry

end complex_expression_l139_139315


namespace number_of_fish_bought_each_year_l139_139871

-- Define the conditions
def initial_fish : ℕ := 2
def net_gain_each_year (x : ℕ) : ℕ := x - 1
def years : ℕ := 5
def final_fish : ℕ := 7

-- Define the problem statement as a Lean theorem
theorem number_of_fish_bought_each_year (x : ℕ) : 
  initial_fish + years * net_gain_each_year x = final_fish → x = 2 := 
sorry

end number_of_fish_bought_each_year_l139_139871


namespace no_two_adj_or_opposite_same_num_l139_139344

theorem no_two_adj_or_opposite_same_num :
  ∃ (prob : ℚ), prob = 25 / 648 ∧ 
  ∀ (A B C D E F : ℕ), 
    (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A) ∧
    (A ≠ D ∧ B ≠ E ∧ C ≠ F) ∧ 
    (1 ≤ A ∧ A ≤ 6) ∧ (1 ≤ B ∧ B ≤ 6) ∧ (1 ≤ C ∧ C ≤ 6) ∧ 
    (1 ≤ D ∧ D ≤ 6) ∧ (1 ≤ E ∧ E ≤ 6) ∧ (1 ≤ F ∧ F ≤ 6) →
    prob = (6 * 5 * 4 * 5 * 3 * 3) / (6^6) := 
sorry

end no_two_adj_or_opposite_same_num_l139_139344


namespace investment_time_l139_139515

theorem investment_time
  (p_investment_ratio : ℚ) (q_investment_ratio : ℚ)
  (profit_ratio_p : ℚ) (profit_ratio_q : ℚ)
  (q_investment_time : ℕ)
  (h1 : p_investment_ratio / q_investment_ratio = 7 / 5)
  (h2 : profit_ratio_p / profit_ratio_q = 7 / 10)
  (h3 : q_investment_time = 40) :
  ∃ t : ℚ, t = 28 :=
by
  sorry

end investment_time_l139_139515


namespace tablet_value_is_2100_compensation_for_m_days_l139_139471

-- Define the given conditions
def monthly_compensation: ℕ := 30
def monthly_tablet_value (x: ℕ) (cash: ℕ): ℕ := x + cash

def daily_compensation (days: ℕ) (x: ℕ) (cash: ℕ): ℕ :=
  days * (x / monthly_compensation + cash / monthly_compensation)

def received_compensation (tablet_value: ℕ) (cash: ℕ): ℕ :=
  tablet_value + cash

-- The proofs we need:
-- Proof that the tablet value is 2100 yuan
theorem tablet_value_is_2100:
  ∀ (x: ℕ) (cash_1 cash_2: ℕ), 
  ((20 * (x / monthly_compensation + 1500 / monthly_compensation)) = (x + 300)) → 
  x = 2100 := sorry

-- Proof that compensation for m days is 120m yuan
theorem compensation_for_m_days (m: ℕ):
  ∀ (x: ℕ), 
  ((x + 1500) / monthly_compensation) = 120 → 
  x = 2100 → 
  m * 120 = 120 * m := sorry

end tablet_value_is_2100_compensation_for_m_days_l139_139471


namespace correct_system_of_equations_l139_139781

theorem correct_system_of_equations (x y : ℝ) :
  (y - x = 4.5) ∧ (x - y / 2 = 1) ↔
  ((y - x = 4.5) ∧ (x - y / 2 = 1)) :=
by sorry

end correct_system_of_equations_l139_139781


namespace find_k_intersection_l139_139061

theorem find_k_intersection :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), y = 2 * x + 3 → y = k * x + 1 → (x = 1 ∧ y = 5) → k = 4) :=
sorry

end find_k_intersection_l139_139061


namespace B_investment_amount_l139_139904

-- Define given conditions in Lean 4

def A_investment := 400
def total_months := 12
def B_investment_months := 6
def total_profit := 100
def A_share := 80
def B_share := total_profit - A_share

-- The problem statement in Lean 4 that needs to be proven:
theorem B_investment_amount (A_investment B_investment_months total_profit A_share B_share: ℕ)
  (hA_investment : A_investment = 400)
  (htotal_months : total_months = 12)
  (hB_investment_months : B_investment_months = 6)
  (htotal_profit : total_profit = 100)
  (hA_share : A_share = 80)
  (hB_share : B_share = total_profit - A_share) 
  : (∃ (B: ℕ), 
       (5 * (A_investment * total_months) = 4 * (400 * total_months + B * B_investment_months)) 
       ∧ B = 200) :=
sorry

end B_investment_amount_l139_139904


namespace find_a_l139_139026

theorem find_a (a : ℝ) : 
  (∀ (i : ℂ), i^2 = -1 → (a * i / (2 - i) + 1 = 2 * i)) → a = 5 :=
by
  intro h
  sorry

end find_a_l139_139026


namespace regular_polygon_sides_l139_139103

theorem regular_polygon_sides (n : ℕ) (h₁ : n ≥ 3) (h₂ : 120 = 180 * (n - 2) / n) : n = 6 :=
by
  sorry

end regular_polygon_sides_l139_139103


namespace find_n_l139_139747

noncomputable def angles_periodic_mod_eq (n : ℤ) : Prop :=
  -100 < n ∧ n < 100 ∧ Real.tan (n * Real.pi / 180) = Real.tan (216 * Real.pi / 180)

theorem find_n (n : ℤ) (h : angles_periodic_mod_eq n) : n = 36 :=
  sorry

end find_n_l139_139747


namespace distinct_real_roots_find_p_l139_139131

theorem distinct_real_roots (p : ℝ) : 
  let f := (fun x => (x - 3) * (x - 2) - p^2)
  let Δ := 1 + 4 * p ^ 2 
  0 < Δ :=
by sorry

theorem find_p (x1 x2 p : ℝ) : 
  (x1 + x2 = 5) → 
  (x1 * x2 = 6 - p^2) → 
  (x1^2 + x2^2 = 3 * x1 * x2) → 
  (p = 1 ∨ p = -1) :=
by sorry

end distinct_real_roots_find_p_l139_139131


namespace probability_neither_red_nor_purple_l139_139413

theorem probability_neither_red_nor_purple :
  (100 - (47 + 3)) / 100 = 0.5 :=
by sorry

end probability_neither_red_nor_purple_l139_139413


namespace sufficient_but_not_necessary_l139_139476

theorem sufficient_but_not_necessary (x y : ℝ) (h : ⌊x⌋ = ⌊y⌋) : 
  |x - y| < 1 ∧ ∃ x y : ℝ, |x - y| < 1 ∧ ⌊x⌋ ≠ ⌊y⌋ :=
by 
  sorry

end sufficient_but_not_necessary_l139_139476


namespace gcd_of_three_digit_palindromes_l139_139213

def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 101 * a + 10 * b

theorem gcd_of_three_digit_palindromes :
  ∀ n, is_palindrome n → Nat.gcd n 1 = 1 := by
  sorry

end gcd_of_three_digit_palindromes_l139_139213


namespace max_min_sum_l139_139716

variable {α : Type*} [LinearOrderedField α]

def is_odd_function (g : α → α) : Prop :=
∀ x, g (-x) = - g x

def has_max_min (f : α → α) (M N : α) : Prop :=
  (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ (∀ x, N ≤ f x) ∧ (∃ x₁, f x₁ = N)

theorem max_min_sum (g f : α → α) (M N : α)
  (h_odd : is_odd_function g)
  (h_def : ∀ x, f x = g (x - 2) + 1)
  (h_max_min : has_max_min f M N) :
  M + N = 2 :=
sorry

end max_min_sum_l139_139716


namespace rectangle_length_l139_139231

variable (w l : ℝ)

def perimeter (w l : ℝ) : ℝ := 2 * w + 2 * l

theorem rectangle_length (h1 : l = w + 2) (h2 : perimeter w l = 20) : l = 6 :=
by sorry

end rectangle_length_l139_139231


namespace solution_exists_l139_139997

variable (x y : ℝ)

noncomputable def condition (x y : ℝ) : Prop :=
  (3 + 5 * x = -4 + 6 * y) ∧ (2 + (-6) * x = 6 + 8 * y)

theorem solution_exists : ∃ (x y : ℝ), condition x y ∧ x = -20 / 19 ∧ y = 11 / 38 := 
  by
  sorry

end solution_exists_l139_139997


namespace james_vegetable_intake_l139_139537

theorem james_vegetable_intake :
  let daily_asparagus := 0.25
  let daily_broccoli := 0.25
  let daily_intake := daily_asparagus + daily_broccoli
  let doubled_daily_intake := daily_intake * 2
  let weekly_intake_asparagus_broccoli := doubled_daily_intake * 7
  let weekly_kale := 3
  let total_weekly_intake := weekly_intake_asparagus_broccoli + weekly_kale
  total_weekly_intake = 10 := 
by
  sorry

end james_vegetable_intake_l139_139537


namespace instantaneous_velocity_at_2_l139_139245

def displacement (t : ℝ) : ℝ := 14 * t - t^2 

def velocity (t : ℝ) : ℝ :=
  sorry -- The velocity function which is the derivative of displacement

theorem instantaneous_velocity_at_2 :
  velocity 2 = 10 := 
  sorry

end instantaneous_velocity_at_2_l139_139245


namespace calculate_sequence_sum_l139_139161

noncomputable def sum_arithmetic_sequence (a l d: Int) : Int :=
  let n := ((l - a) / d) + 1
  (n * (a + l)) / 2

theorem calculate_sequence_sum :
  3 * (sum_arithmetic_sequence 45 93 2) + 2 * (sum_arithmetic_sequence (-4) 38 2) = 5923 := by
  sorry

end calculate_sequence_sum_l139_139161


namespace gcd_g105_g106_l139_139524

def g (x : ℕ) : ℕ := x^2 - x + 2502

theorem gcd_g105_g106 : gcd (g 105) (g 106) = 2 := by
  sorry

end gcd_g105_g106_l139_139524


namespace initial_oranges_l139_139846

theorem initial_oranges (X : ℕ) (h1 : X - 37 + 7 = 10) : X = 40 :=
by
  sorry

end initial_oranges_l139_139846


namespace length_of_AB_l139_139645

theorem length_of_AB
  (P Q : ℝ) (AB : ℝ)
  (hP : P = 3 / 7 * AB)
  (hQ : Q = 4 / 9 * AB)
  (hPQ : abs (Q - P) = 3) :
  AB = 189 :=
by
  sorry

end length_of_AB_l139_139645


namespace exists_int_squares_l139_139971

theorem exists_int_squares (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  ∃ x y : ℤ, (a^2 + b^2)^n = x^2 + y^2 :=
by
  sorry

end exists_int_squares_l139_139971


namespace set_subset_condition_l139_139983

theorem set_subset_condition (a : ℝ) :
  (∀ x, (1 < a * x ∧ a * x < 2) → (-1 < x ∧ x < 1)) → (|a| ≥ 2 ∨ a = 0) :=
by
  intro h
  sorry

end set_subset_condition_l139_139983


namespace initial_bushes_l139_139805

theorem initial_bushes (b : ℕ) (h1 : b + 4 = 6) : b = 2 :=
by {
  sorry
}

end initial_bushes_l139_139805


namespace tram_speed_l139_139567

/-- 
Given:
1. The pedestrian's speed is 1 km per 10 minutes, which converts to 6 km/h.
2. The speed of the trams is V km/h.
3. The relative speed of oncoming trams is V + 6 km/h.
4. The relative speed of overtaking trams is V - 6 km/h.
5. The ratio of the number of oncoming trams to overtaking trams is 700/300.
Prove:
The speed of the trams V is 15 km/h.
-/
theorem tram_speed (V : ℝ) (h1 : (V + 6) / (V - 6) = 700 / 300) : V = 15 :=
by
  sorry

end tram_speed_l139_139567


namespace number_of_blue_tiles_is_16_l139_139056

def length_of_floor : ℕ := 20
def breadth_of_floor : ℕ := 10
def tile_length : ℕ := 2

def total_tiles : ℕ := (length_of_floor / tile_length) * (breadth_of_floor / tile_length)

def black_tiles : ℕ :=
  let rows_length := 2 * (length_of_floor / tile_length)
  let rows_breadth := 2 * (breadth_of_floor / tile_length)
  (rows_length + rows_breadth) - 4

def remaining_tiles : ℕ := total_tiles - black_tiles
def white_tiles : ℕ := remaining_tiles / 3
def blue_tiles : ℕ := remaining_tiles - white_tiles

theorem number_of_blue_tiles_is_16 :
  blue_tiles = 16 :=
by
  sorry

end number_of_blue_tiles_is_16_l139_139056


namespace y_axis_symmetry_l139_139025

theorem y_axis_symmetry (x y : ℝ) (P : ℝ × ℝ) (hx : P = (-5, 3)) : 
  (P.1 = -5 ∧ P.2 = 3) → (P.1 * -1, P.2) = (5, 3) :=
by
  intro h
  rw [hx]
  simp [Neg.neg, h]
  sorry

end y_axis_symmetry_l139_139025


namespace mixture_replacement_l139_139246

theorem mixture_replacement (A B T x : ℝ)
  (h1 : A / (A + B) = 7 / 12)
  (h2 : A = 21)
  (h3 : (A / (B + x)) = 7 / 9) :
  x = 12 :=
by
  sorry

end mixture_replacement_l139_139246


namespace monthly_growth_rate_l139_139372

theorem monthly_growth_rate (x : ℝ)
  (turnover_may : ℝ := 1)
  (turnover_july : ℝ := 1.21)
  (growth_rate_condition : (1 + x) ^ 2 = 1.21) :
  x = 0.1 :=
sorry

end monthly_growth_rate_l139_139372


namespace slices_served_today_l139_139228

-- Definitions based on conditions from part a)
def slices_lunch_today : ℕ := 7
def slices_dinner_today : ℕ := 5

-- Proof statement based on part c)
theorem slices_served_today : slices_lunch_today + slices_dinner_today = 12 := 
by
  sorry

end slices_served_today_l139_139228


namespace construct_circle_feasible_l139_139674

theorem construct_circle_feasible (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : b^2 > (a^2 + c^2) / 2) :
  ∃ x y d : ℝ, 
  d > 0 ∧ 
  (d / 2)^2 = y^2 + (a / 2)^2 ∧ 
  (d / 2)^2 = (y - x)^2 + (b / 2)^2 ∧ 
  (d / 2)^2 = (y - 2 * x)^2 + (c / 2)^2 :=
sorry

end construct_circle_feasible_l139_139674


namespace fred_final_cards_l139_139702

def initial_cards : ℕ := 40
def keith_bought : ℕ := 22
def linda_bought : ℕ := 15

theorem fred_final_cards : initial_cards - keith_bought - linda_bought = 3 :=
by sorry

end fred_final_cards_l139_139702


namespace remaining_pie_l139_139142

theorem remaining_pie (carlos_take: ℝ) (sophia_share : ℝ) (final_remaining : ℝ) :
  carlos_take = 0.6 ∧ sophia_share = (1 - carlos_take) / 4 ∧ final_remaining = (1 - carlos_take) - sophia_share →
  final_remaining = 0.3 :=
by
  intros h
  sorry

end remaining_pie_l139_139142


namespace length_of_first_train_l139_139751

theorem length_of_first_train
  (speed_first : ℕ)
  (speed_second : ℕ)
  (length_second : ℕ)
  (distance_between : ℕ)
  (time_to_cross : ℕ)
  (h1 : speed_first = 10)
  (h2 : speed_second = 15)
  (h3 : length_second = 150)
  (h4 : distance_between = 50)
  (h5 : time_to_cross = 60) :
  ∃ L : ℕ, L = 100 :=
by
  sorry

end length_of_first_train_l139_139751


namespace birthday_count_l139_139490

theorem birthday_count (N : ℕ) (P : ℝ) (days : ℕ) (hN : N = 1200) (hP1 : P = 1 / 365 ∨ P = 1 / 366) 
  (hdays : days = 365 ∨ days = 366) : 
  N * P = 4 :=
by
  sorry

end birthday_count_l139_139490


namespace repeating_decimal_addition_l139_139876

def repeating_decimal_45 := (45 / 99 : ℚ)
def repeating_decimal_36 := (36 / 99 : ℚ)

theorem repeating_decimal_addition :
  repeating_decimal_45 + repeating_decimal_36 = 9 / 11 :=
by
  sorry

end repeating_decimal_addition_l139_139876


namespace cistern_length_l139_139450

theorem cistern_length (w d A : ℝ) (h : d = 1.25 ∧ w = 4 ∧ A = 68.5) :
  ∃ L : ℝ, (L * w) + (2 * L * d) + (2 * w * d) = A ∧ L = 9 :=
by
  obtain ⟨h_d, h_w, h_A⟩ := h
  use 9
  simp [h_d, h_w, h_A]
  norm_num
  sorry

end cistern_length_l139_139450


namespace good_permutations_count_l139_139529

-- Define the main problem and the conditions
theorem good_permutations_count (n : ℕ) (hn : n > 0) : 
  ∃ P : ℕ → ℕ, 
  (P n = (1 / Real.sqrt 5) * (((1 + Real.sqrt 5) / 2) ^ (n + 1) - ((1 - Real.sqrt 5) / 2) ^ (n + 1))) := 
sorry

end good_permutations_count_l139_139529


namespace find_A_l139_139376

theorem find_A :
  ∃ A B : ℕ, A < 10 ∧ B < 10 ∧ 5 * 100 + A * 10 + 8 - (B * 100 + 1 * 10 + 4) = 364 ∧ A = 7 :=
by
  sorry

end find_A_l139_139376


namespace club_committee_probability_l139_139437

noncomputable def probability_at_least_two_boys_and_two_girls (total_members boys girls committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_fewer_than_two_boys := (Nat.choose girls committee_size) + (boys * Nat.choose girls (committee_size - 1))
  let ways_fewer_than_two_girls := (Nat.choose boys committee_size) + (girls * Nat.choose boys (committee_size - 1))
  let ways_invalid := ways_fewer_than_two_boys + ways_fewer_than_two_girls
  (total_ways - ways_invalid) / total_ways

theorem club_committee_probability :
  probability_at_least_two_boys_and_two_girls 30 12 18 6 = 457215 / 593775 :=
by
  sorry

end club_committee_probability_l139_139437


namespace complement_of_67_is_23_l139_139828

-- Define complement function
def complement (x : ℝ) : ℝ := 90 - x

-- State the theorem
theorem complement_of_67_is_23 : complement 67 = 23 := 
by
  sorry

end complement_of_67_is_23_l139_139828


namespace remainder_division_l139_139139

theorem remainder_division (a b : ℕ) (h1 : a > b) (h2 : (a - b) % 6 = 5) : a % 6 = 5 :=
sorry

end remainder_division_l139_139139


namespace find_fraction_l139_139297

theorem find_fraction (x y : ℕ) (h₁ : x / (y + 1) = 1 / 2) (h₂ : (x + 1) / y = 1) : x = 2 ∧ y = 3 := by
  sorry

end find_fraction_l139_139297


namespace number_of_freshmen_l139_139345

theorem number_of_freshmen (n : ℕ) : n < 450 ∧ n % 19 = 18 ∧ n % 17 = 10 → n = 265 := by
  sorry

end number_of_freshmen_l139_139345


namespace exists_invertible_int_matrix_l139_139726

theorem exists_invertible_int_matrix (m : ℕ) (k : Fin m → ℤ) : 
  ∃ A : Matrix (Fin m) (Fin m) ℤ,
    (∀ j, IsUnit (A + k j • (1 : Matrix (Fin m) (Fin m) ℤ))) :=
sorry

end exists_invertible_int_matrix_l139_139726


namespace arithmetic_mean_l139_139692

theorem arithmetic_mean (a b c : ℚ) (h₁ : a = 8 / 12) (h₂ : b = 10 / 12) (h₃ : c = 9 / 12) :
  c = (a + b) / 2 :=
by
  sorry

end arithmetic_mean_l139_139692


namespace team_not_losing_probability_l139_139888

theorem team_not_losing_probability
  (p_center_forward : ℝ) (p_winger : ℝ) (p_attacking_midfielder : ℝ)
  (rate_center_forward : ℝ) (rate_winger : ℝ) (rate_attacking_midfielder : ℝ)
  (h_center_forward : p_center_forward = 0.2) (h_winger : p_winger = 0.5) (h_attacking_midfielder : p_attacking_midfielder = 0.3)
  (h_rate_center_forward : rate_center_forward = 0.4) (h_rate_winger : rate_winger = 0.2) (h_rate_attacking_midfielder : rate_attacking_midfielder = 0.2) :
  (p_center_forward * (1 - rate_center_forward) + p_winger * (1 - rate_winger) + p_attacking_midfielder * (1 - rate_attacking_midfielder)) = 0.76 :=
by
  sorry

end team_not_losing_probability_l139_139888


namespace difference_of_squares_l139_139648

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 20) : a^2 - b^2 = 1200 := 
sorry

end difference_of_squares_l139_139648


namespace part1_part2_l139_139243

noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (1 + x)

theorem part1 (x : ℝ) (h : 0 ≤ x) (h1 : x ≤ 1) : f x ≥ 1 - x + x^2 := sorry

theorem part2 (x : ℝ) (h : 0 ≤ x) (h1 : x ≤ 1) : 3 / 4 < f x ∧ f x ≤ 3 / 2 := sorry

end part1_part2_l139_139243


namespace complete_square_form_l139_139164

theorem complete_square_form :
  ∀ x : ℝ, (3 * x^2 - 6 * x + 2 = 0) → (x - 1)^2 = (1 / 3) :=
by
  intro x h
  sorry

end complete_square_form_l139_139164


namespace total_money_amount_l139_139982

-- Define the conditions
def num_bills : ℕ := 3
def value_per_bill : ℕ := 20
def initial_amount : ℕ := 75

-- Define the statement about the total amount of money James has
theorem total_money_amount : num_bills * value_per_bill + initial_amount = 135 := 
by 
  -- Since the proof is not required, we use 'sorry' to skip it
  sorry

end total_money_amount_l139_139982


namespace oblong_perimeter_182_l139_139848

variables (l w : ℕ) (x : ℤ)

def is_oblong (l w : ℕ) : Prop :=
l * w = 4624 ∧ l = 4 * x ∧ w = 3 * x

theorem oblong_perimeter_182 (l w x : ℕ) (hlw : is_oblong l w x) : 
  2 * l + 2 * w = 182 :=
by
  sorry

end oblong_perimeter_182_l139_139848


namespace length_of_bridge_l139_139667

noncomputable def L_train : ℝ := 110
noncomputable def v_train : ℝ := 72 * (1000 / 3600)
noncomputable def t : ℝ := 12.099

theorem length_of_bridge : (v_train * t - L_train) = 131.98 :=
by
  -- The proof should come here
  sorry

end length_of_bridge_l139_139667


namespace sin_double_theta_eq_three_fourths_l139_139862

theorem sin_double_theta_eq_three_fourths (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2)
  (h1 : Real.sin (π * Real.cos θ) = Real.cos (π * Real.sin θ)) :
  Real.sin (2 * θ) = 3 / 4 :=
  sorry

end sin_double_theta_eq_three_fourths_l139_139862


namespace unique_number_outside_range_f_l139_139987

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_outside_range_f (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : f a b c d 19 = 19) (h6 : f a b c d 97 = 97)
  (h7 : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) : 
  ∀ y : ℝ, y ≠ 58 → ∃ x : ℝ, f a b c d x ≠ y :=
sorry

end unique_number_outside_range_f_l139_139987


namespace fruit_seller_stock_l139_139700

-- Define the given conditions
def remaining_oranges : ℝ := 675
def remaining_percentage : ℝ := 0.25

-- Define the problem function
def original_stock (O : ℝ) : Prop :=
  remaining_percentage * O = remaining_oranges

-- Prove the original stock of oranges was 2700 kg
theorem fruit_seller_stock : original_stock 2700 :=
by
  sorry

end fruit_seller_stock_l139_139700


namespace henry_age_is_20_l139_139561

open Nat

def sum_ages (H J : ℕ) : Prop := H + J = 33
def age_relation (H J : ℕ) : Prop := H - 6 = 2 * (J - 6)

theorem henry_age_is_20 (H J : ℕ) (h1 : sum_ages H J) (h2 : age_relation H J) : H = 20 :=
by
  -- Proof goes here
  sorry

end henry_age_is_20_l139_139561


namespace intersection_point_of_lines_l139_139439

theorem intersection_point_of_lines :
  (∃ x y : ℝ, y = x ∧ y = -x + 2 ∧ (x = 1 ∧ y = 1)) :=
sorry

end intersection_point_of_lines_l139_139439


namespace four_nonzero_complex_numbers_form_square_l139_139104

open Complex

theorem four_nonzero_complex_numbers_form_square :
  ∃ (S : Finset ℂ), S.card = 4 ∧ (∀ z ∈ S, z ≠ 0) ∧ (∀ z ∈ S, ∃ (θ : ℝ), z = exp (θ * I) ∧ (exp (4 * θ * I) - z).re = 0 ∧ (exp (4 * θ * I) - z).im = cos (π / 2)) := 
sorry

end four_nonzero_complex_numbers_form_square_l139_139104


namespace board_cut_ratio_l139_139101

theorem board_cut_ratio (L S : ℝ) (h1 : S + L = 20) (h2 : S = L + 4) (h3 : S = 8.0) : S / L = 1 := by
  sorry

end board_cut_ratio_l139_139101


namespace quadratic_real_roots_and_a_value_l139_139856

-- Define the quadratic equation (a-5)x^2 - 4x - 1 = 0
def quadratic_eq (a : ℝ) (x : ℝ) := (a - 5) * x^2 - 4 * x - 1

-- Define the discriminant for the quadratic equation
def discriminant (a : ℝ) := 4 - 4 * (a - 5) * (-1)

-- Main theorem statement
theorem quadratic_real_roots_and_a_value
    (a : ℝ) (x1 x2 : ℝ) 
    (h_roots : (a - 5) ≠ 0)
    (h_eq : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0)
    (h_sum_product : x1 + x2 + x1 * x2 = 3) :
    (a ≥ 1) ∧ (a = 6) :=
  sorry

end quadratic_real_roots_and_a_value_l139_139856


namespace reduction_when_fifth_runner_twice_as_fast_l139_139714

theorem reduction_when_fifth_runner_twice_as_fast (T T1 T2 T3 T4 T5 : ℝ)
  (h1 : T = T1 + T2 + T3 + T4 + T5)
  (h_T1 : (T1 / 2 + T2 + T3 + T4 + T5) = 0.95 * T)
  (h_T2 : (T1 + T2 / 2 + T3 + T4 + T5) = 0.90 * T)
  (h_T3 : (T1 + T2 + T3 / 2 + T4 + T5) = 0.88 * T)
  (h_T4 : (T1 + T2 + T3 + T4 / 2 + T5) = 0.85 * T)
  : (T1 + T2 + T3 + T4 + T5 / 2) = 0.92 * T := 
sorry

end reduction_when_fifth_runner_twice_as_fast_l139_139714


namespace range_of_a_plus_b_l139_139953

theorem range_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b)
    (h3 : |2 - a^2| = |2 - b^2|) : 2 < a + b ∧ a + b < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_a_plus_b_l139_139953


namespace lines_parallel_condition_l139_139170

theorem lines_parallel_condition (a : ℝ) : 
  (a = 1) ↔ (∀ x y : ℝ, (a * x + 2 * y - 1 = 0 → x + (a + 1) * y + 4 = 0)) :=
sorry

end lines_parallel_condition_l139_139170


namespace graph_is_hyperbola_l139_139654

theorem graph_is_hyperbola : ∀ x y : ℝ, (x + y) ^ 2 = x ^ 2 + y ^ 2 + 2 * x + 2 * y ↔ (x - 1) * (y - 1) = 1 := 
by {
  sorry
}

end graph_is_hyperbola_l139_139654


namespace beth_finishes_first_l139_139545

open Real

noncomputable def andy_lawn_area : ℝ := sorry
noncomputable def beth_lawn_area : ℝ := andy_lawn_area / 3
noncomputable def carlos_lawn_area : ℝ := andy_lawn_area / 4

noncomputable def andy_mowing_rate : ℝ := sorry
noncomputable def beth_mowing_rate : ℝ := andy_mowing_rate
noncomputable def carlos_mowing_rate : ℝ := andy_mowing_rate / 2

noncomputable def carlos_break : ℝ := 10

noncomputable def andy_mowing_time := andy_lawn_area / andy_mowing_rate
noncomputable def beth_mowing_time := beth_lawn_area / beth_mowing_rate
noncomputable def carlos_mowing_time := (carlos_lawn_area / carlos_mowing_rate) + carlos_break

theorem beth_finishes_first :
  beth_mowing_time < andy_mowing_time ∧ beth_mowing_time < carlos_mowing_time := by
  sorry

end beth_finishes_first_l139_139545


namespace fruit_display_total_l139_139412

-- Define the number of bananas
def bananas : ℕ := 5

-- Define the number of oranges, twice the number of bananas
def oranges : ℕ := 2 * bananas

-- Define the number of apples, twice the number of oranges
def apples : ℕ := 2 * oranges

-- Define the total number of fruits
def total_fruits : ℕ := bananas + oranges + apples

-- Theorem statement: Prove that the total number of fruits is 35
theorem fruit_display_total : total_fruits = 35 := 
by
  -- proof will be skipped, but all necessary conditions and definitions are included to ensure the statement is valid
  sorry

end fruit_display_total_l139_139412


namespace simplify_expression_l139_139679

theorem simplify_expression : (8 * (15 / 9) * (-45 / 40) = -(1 / 15)) :=
by
  sorry

end simplify_expression_l139_139679


namespace product_of_fractions_l139_139387

-- Definitions from the conditions
def a : ℚ := 2 / 3 
def b : ℚ := 3 / 5
def c : ℚ := 4 / 7
def d : ℚ := 5 / 9

-- Statement of the proof problem
theorem product_of_fractions : a * b * c * d = 8 / 63 := 
by
  sorry

end product_of_fractions_l139_139387


namespace time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l139_139427

noncomputable def time_to_count_envelopes (num_envelopes : ℕ) : ℕ :=
(num_envelopes / 10) * 10

theorem time_to_count_60_envelopes_is_40 :
  time_to_count_envelopes 60 = 40 := 
sorry

theorem time_to_count_90_envelopes_is_10 :
  time_to_count_envelopes 90 = 10 := 
sorry

end time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l139_139427


namespace integer_solutions_l139_139163

theorem integer_solutions (x y z : ℤ) : 
  x + y + z = 3 ∧ x^3 + y^3 + z^3 = 3 ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 4 ∧ y = 4 ∧ z = -5) ∨
  (x = 4 ∧ y = -5 ∧ z = 4) ∨
  (x = -5 ∧ y = 4 ∧ z = 4) := 
sorry

end integer_solutions_l139_139163


namespace total_brown_mms_3rd_4th_bags_l139_139614

def brown_mms_in_bags := (9 : ℕ) + (12 : ℕ) + (3 : ℕ)

def total_bags := 5

def average_mms_per_bag := 8

theorem total_brown_mms_3rd_4th_bags (x y : ℕ) 
  (h1 : brown_mms_in_bags + x + y = average_mms_per_bag * total_bags) : 
  x + y = 16 :=
by
  have h2 : brown_mms_in_bags + x + y = 40 := by sorry
  sorry

end total_brown_mms_3rd_4th_bags_l139_139614


namespace find_A_l139_139712

theorem find_A (A B : ℕ) (h1: 3 + 6 * (100 + 10 * A + B) = 691) (h2 : 100 ≤ 6 * (100 + 10 * A + B) ∧ 6 * (100 + 10 * A + B) < 1000) : 
A = 8 :=
sorry

end find_A_l139_139712


namespace range_of_f_l139_139879

open Set

noncomputable def f (x : ℝ) : ℝ := 3^x + 5

theorem range_of_f :
  range f = Ioi 5 :=
sorry

end range_of_f_l139_139879


namespace calculate_value_l139_139111

theorem calculate_value (x y d : ℕ) (hx : x = 2024) (hy : y = 1935) (hd : d = 225) : 
  (x - y)^2 / d = 35 := by
  sorry

end calculate_value_l139_139111


namespace solution_exists_for_any_y_l139_139868

theorem solution_exists_for_any_y (z : ℝ) : (∀ y : ℝ, ∃ x : ℝ, x^2 + y^2 + 4*z^2 + 2*x*y*z - 9 = 0) ↔ |z| ≤ 3 / 2 := 
sorry

end solution_exists_for_any_y_l139_139868


namespace smallest_digit_not_in_units_place_of_odd_l139_139394

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l139_139394


namespace unique_games_count_l139_139832

noncomputable def total_games_played (n : ℕ) (m : ℕ) : ℕ :=
  (n * m) / 2

theorem unique_games_count (students : ℕ) (games_per_student : ℕ) (h1 : students = 9) (h2 : games_per_student = 6) :
  total_games_played students games_per_student = 27 :=
by
  rw [h1, h2]
  -- This partially evaluates total_games_played using the values from h1 and h2.
  -- Performing actual proof steps is not necessary, so we'll use sorry.
  sorry

end unique_games_count_l139_139832


namespace olivia_money_left_l139_139981

-- Defining hourly wages
def wage_monday : ℕ := 10
def wage_wednesday : ℕ := 12
def wage_friday : ℕ := 14
def wage_saturday : ℕ := 20

-- Defining hours worked each day
def hours_monday : ℕ := 5
def hours_wednesday : ℕ := 4
def hours_friday : ℕ := 3
def hours_saturday : ℕ := 2

-- Defining business-related expenses and tax rate
def expenses : ℕ := 50
def tax_rate : ℝ := 0.15

-- Calculate total earnings
def total_earnings : ℕ :=
  (hours_monday * wage_monday) +
  (hours_wednesday * wage_wednesday) +
  (hours_friday * wage_friday) +
  (hours_saturday * wage_saturday)

-- Earnings after expenses
def earnings_after_expenses : ℕ :=
  total_earnings - expenses

-- Calculate tax amount
def tax_amount : ℝ :=
  tax_rate * (total_earnings : ℝ)

-- Final amount Olivia has left
def remaining_amount : ℝ :=
  (earnings_after_expenses : ℝ) - tax_amount

theorem olivia_money_left : remaining_amount = 103 := by
  sorry

end olivia_money_left_l139_139981


namespace max_C_trees_l139_139180

theorem max_C_trees 
  (price_A : ℕ) (price_B : ℕ) (price_C : ℕ) (total_price : ℕ) (total_trees : ℕ)
  (h_price_ratio : 2 * price_B = 2 * price_A ∧ 3 * price_A = 2 * price_C)
  (h_price_A : price_A = 200)
  (h_total_price : total_price = 220120)
  (h_total_trees : total_trees = 1000) :
  ∃ (num_C : ℕ), num_C = 201 ∧ ∀ num_C', num_C' > num_C → 
  total_price < price_A * (total_trees - num_C') + price_C * num_C' :=
by
  sorry

end max_C_trees_l139_139180


namespace eval_expression_l139_139460

theorem eval_expression : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 2023) / (2023 * 2024) = -4044 :=
by 
  sorry

end eval_expression_l139_139460


namespace advertising_department_employees_l139_139307

theorem advertising_department_employees (N S A_s x : ℕ) (hN : N = 1000) (hS : S = 80) (hA_s : A_s = 4) 
(h_stratified : x / N = A_s / S) : x = 50 :=
sorry

end advertising_department_employees_l139_139307


namespace find_angle_AOD_l139_139141

noncomputable def angleAOD (x : ℝ) : ℝ :=
4 * x

theorem find_angle_AOD (x : ℝ) (h1 : 4 * x = 180) : angleAOD x = 135 :=
by
  -- x = 45
  have h2 : x = 45 := by linarith

  -- angleAOD 45 = 4 * 45 = 135
  rw [angleAOD, h2]
  norm_num
  sorry

end find_angle_AOD_l139_139141


namespace resulting_polygon_sides_l139_139422

theorem resulting_polygon_sides 
    (triangle_sides : ℕ := 3) 
    (square_sides : ℕ := 4) 
    (pentagon_sides : ℕ := 5) 
    (heptagon_sides : ℕ := 7) 
    (hexagon_sides : ℕ := 6) 
    (octagon_sides : ℕ := 8) 
    (shared_sides : ℕ := 1) :
    (2 * shared_sides + 4 * (shared_sides + 1)) = 16 := by 
  sorry

end resulting_polygon_sides_l139_139422


namespace aiden_nap_is_15_minutes_l139_139410

def aiden_nap_duration_in_minutes (nap_in_hours : ℚ) (minutes_per_hour : ℕ) : ℚ :=
  nap_in_hours * minutes_per_hour

theorem aiden_nap_is_15_minutes :
  aiden_nap_duration_in_minutes (1/4) 60 = 15 := by
  sorry

end aiden_nap_is_15_minutes_l139_139410


namespace roof_length_width_difference_l139_139934

theorem roof_length_width_difference (w l : ℝ) 
  (h1 : l = 5 * w) 
  (h2 : l * w = 720) : l - w = 48 := 
sorry

end roof_length_width_difference_l139_139934


namespace hiking_time_l139_139554

-- Define the conditions
def Distance : ℕ := 12
def Pace_up : ℕ := 4
def Pace_down : ℕ := 6

-- Statement to be proved
theorem hiking_time (d : ℕ) (pu : ℕ) (pd : ℕ) (h₁ : d = Distance) (h₂ : pu = Pace_up) (h₃ : pd = Pace_down) :
  d / pu + d / pd = 5 :=
by sorry

end hiking_time_l139_139554


namespace union_when_m_is_one_range_of_m_condition_1_range_of_m_condition_2_l139_139465

open Set

noncomputable def A := {x : ℝ | -2 < x ∧ x < 2}
noncomputable def B (m : ℝ) := {x : ℝ | (m - 2) ≤ x ∧ x ≤ (2 * m + 1)}

-- Part (1):
theorem union_when_m_is_one :
  A ∪ B 1 = {x : ℝ | -2 < x ∧ x ≤ 3} := sorry

-- Part (2):
theorem range_of_m_condition_1 :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ∈ Iic (-3/2) ∪ Ici 4 := sorry

theorem range_of_m_condition_2 :
  ∀ m : ℝ, A ∪ B m = A ↔ m ∈ Iio (-3) ∪ Ioo 0 (1/2) := sorry

end union_when_m_is_one_range_of_m_condition_1_range_of_m_condition_2_l139_139465


namespace P_at_2007_l139_139485

noncomputable def P (x : ℝ) : ℝ :=
x^15 - 2008 * x^14 + 2008 * x^13 - 2008 * x^12 + 2008 * x^11
- 2008 * x^10 + 2008 * x^9 - 2008 * x^8 + 2008 * x^7
- 2008 * x^6 + 2008 * x^5 - 2008 * x^4 + 2008 * x^3
- 2008 * x^2 + 2008 * x

-- Statement to show that P(2007) = 2007
theorem P_at_2007 : P 2007 = 2007 :=
  sorry

end P_at_2007_l139_139485


namespace pascal_fifth_element_row_20_l139_139443

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end pascal_fifth_element_row_20_l139_139443


namespace complex_simplify_l139_139195

theorem complex_simplify :
  10.25 * Real.sqrt 6 * Complex.exp (Complex.I * 160 * Real.pi / 180)
  / (Real.sqrt 3 * Complex.exp (Complex.I * 40 * Real.pi / 180))
  = (-Real.sqrt 2 / 2) + Complex.I * (Real.sqrt 6 / 2) := by
  sorry

end complex_simplify_l139_139195


namespace dimes_given_l139_139917

theorem dimes_given (initial_dimes final_dimes dimes_dad_gave : ℕ)
  (h1 : initial_dimes = 9)
  (h2 : final_dimes = 16)
  (h3 : final_dimes = initial_dimes + dimes_dad_gave) :
  dimes_dad_gave = 7 :=
by
  rw [h1, h2] at h3
  linarith

end dimes_given_l139_139917


namespace find_ad_l139_139263

-- Defining the two-digit and three-digit numbers
def two_digit (a b : ℕ) : ℕ := 10 * a + b
def three_digit (a b : ℕ) : ℕ := 100 + two_digit a b

def two_digit' (c d : ℕ) : ℕ := 10 * c + d
def three_digit' (c d : ℕ) : ℕ := 100 * c + 10 * d + 1

-- The main problem
theorem find_ad (a b c d : ℕ) (h1 : three_digit a b = three_digit' c d + 15) (h2 : two_digit a b = two_digit' c d + 24) :
    two_digit a d = 32 := by
  sorry

end find_ad_l139_139263


namespace change_in_surface_area_zero_l139_139303

-- Original rectangular solid dimensions
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

-- Smaller prism dimensions
structure SmallerPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Conditions
def originalSolid : RectangularSolid := { length := 4, width := 3, height := 2 }
def removedPrism : SmallerPrism := { length := 1, width := 1, height := 2 }

-- Surface area calculation function
def surface_area (solid : RectangularSolid) : ℝ := 
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

-- Calculate the change in surface area
theorem change_in_surface_area_zero :
  let original_surface_area := surface_area originalSolid
  let removed_surface_area := (removedPrism.length * removedPrism.height)
  let new_exposed_area := (removedPrism.length * removedPrism.height)
  (original_surface_area - removed_surface_area + new_exposed_area) = original_surface_area :=
by
  sorry

end change_in_surface_area_zero_l139_139303


namespace find_multiple_of_sons_age_l139_139710

theorem find_multiple_of_sons_age (F S k : ℕ) 
  (h1 : F = 33)
  (h2 : F = k * S + 3)
  (h3 : F + 3 = 2 * (S + 3) + 10) : 
  k = 3 :=
by
  sorry

end find_multiple_of_sons_age_l139_139710


namespace geometric_sequence_common_ratio_l139_139420

theorem geometric_sequence_common_ratio 
  (a1 q : ℝ) 
  (h : (a1 * (1 - q^3) / (1 - q)) + 3 * (a1 * (1 - q^2) / (1 - q)) = 0) : 
  q = -1 :=
sorry

end geometric_sequence_common_ratio_l139_139420


namespace remainder_of_12345678910_div_101_l139_139351

theorem remainder_of_12345678910_div_101 :
  12345678910 % 101 = 31 :=
sorry

end remainder_of_12345678910_div_101_l139_139351


namespace Darius_scored_10_points_l139_139084

theorem Darius_scored_10_points
  (D Marius Matt : ℕ)
  (h1 : Marius = D + 3)
  (h2 : Matt = D + 5)
  (h3 : D + Marius + Matt = 38) : 
  D = 10 :=
by
  sorry

end Darius_scored_10_points_l139_139084


namespace value_of_item_l139_139403

theorem value_of_item (a b m p : ℕ) (h : a ≠ b) (eq_capitals : a * x + m = b * x + p) : 
  x = (p - m) / (a - b) :=
by
  sorry

end value_of_item_l139_139403


namespace value_of_x_plus_y_l139_139630

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem value_of_x_plus_y
  (x y : ℝ)
  (h1 : x ≥ 1)
  (h2 : y ≥ 1)
  (h3 : x * y = 10)
  (h4 : x^(lg x) * y^(lg y) ≥ 10) :
  x + y = 11 :=
  sorry

end value_of_x_plus_y_l139_139630


namespace tensor_identity_l139_139757

namespace tensor_problem

def otimes (x y : ℝ) : ℝ := x^2 + y

theorem tensor_identity (a : ℝ) : otimes a (otimes a a) = 2 * a^2 + a :=
by sorry

end tensor_problem

end tensor_identity_l139_139757


namespace annual_population_change_l139_139484

theorem annual_population_change (initial_population : Int) (moved_in : Int) (moved_out : Int) (final_population : Int) (years : Int) : 
  initial_population = 780 → 
  moved_in = 100 →
  moved_out = 400 →
  final_population = 60 →
  years = 4 →
  (initial_population + moved_in - moved_out - final_population) / years = 105 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end annual_population_change_l139_139484


namespace least_possible_value_z_minus_x_l139_139592

theorem least_possible_value_z_minus_x
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  z - x = 9 :=
  sorry

end least_possible_value_z_minus_x_l139_139592


namespace optimal_chalk_length_l139_139783

theorem optimal_chalk_length (l : ℝ) (h₁: 10 ≤ l) (h₂: l ≤ 15) (h₃: l = 12) : l = 12 :=
by
  sorry

end optimal_chalk_length_l139_139783


namespace odd_function_f_neg_9_l139_139076

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then x^(1/2) 
else -((-x)^(1/2))

theorem odd_function_f_neg_9 : f (-9) = -3 := by
  sorry

end odd_function_f_neg_9_l139_139076


namespace find_geometric_sequence_values_l139_139229

theorem find_geometric_sequence_values :
  ∃ (a b c : ℤ), (∃ q : ℤ, q ≠ 0 ∧ 2 * q ^ 4 = 32 ∧ a = 2 * q ∧ b = 2 * q ^ 2 ∧ c = 2 * q ^ 3)
                 ↔ ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = -4 ∧ b = 8 ∧ c = -16)) := by
  sorry

end find_geometric_sequence_values_l139_139229


namespace average_speed_triathlon_l139_139461

theorem average_speed_triathlon :
  let swimming_distance := 1.5
  let biking_distance := 3
  let running_distance := 2
  let swimming_speed := 2
  let biking_speed := 25
  let running_speed := 8

  let t_s := swimming_distance / swimming_speed
  let t_b := biking_distance / biking_speed
  let t_r := running_distance / running_speed
  let total_time := t_s + t_b + t_r

  let total_distance := swimming_distance + biking_distance + running_distance
  let average_speed := total_distance / total_time

  average_speed = 5.8 :=
  by
    sorry

end average_speed_triathlon_l139_139461


namespace parallel_lines_minimum_distance_l139_139708

theorem parallel_lines_minimum_distance :
  ∀ (m n : ℝ) (k : ℝ), 
  k = 2 ∧ ∀ (L1 L2 : ℝ → ℝ), -- we define L1 and L2 as functions
  (L1 = λ y => 2 * y + 3) ∧ (L2 = λ y => k * y - 1) ∧ 
  ((L1 n = m) ∧ (L2 (n + k) = m + 2)) → 
  dist (m, n) (m + 2, n + 2) = 2 * Real.sqrt 2 := 
sorry

end parallel_lines_minimum_distance_l139_139708


namespace find_last_two_digits_l139_139860

noncomputable def tenth_digit (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ) : ℕ :=
d7 + d8

noncomputable def ninth_digit (d1 d2 d3 d4 d5 d6 d7 : ℕ) : ℕ :=
d6 + d7

theorem find_last_two_digits :
  ∃ d9 d10 : ℕ, d9 = ninth_digit 1 1 2 3 5 8 13 ∧ d10 = tenth_digit 1 1 2 3 5 8 13 21 :=
by
  sorry

end find_last_two_digits_l139_139860


namespace Yuna_place_l139_139773

theorem Yuna_place (Eunji_place : ℕ) (distance : ℕ) (Yuna_place : ℕ) 
  (h1 : Eunji_place = 100) 
  (h2 : distance = 11) 
  (h3 : Yuna_place = Eunji_place + distance) : 
  Yuna_place = 111 := 
sorry

end Yuna_place_l139_139773


namespace oliver_spent_amount_l139_139353

theorem oliver_spent_amount :
  ∀ (S : ℕ), (33 - S + 32 = 61) → S = 4 :=
by
  sorry

end oliver_spent_amount_l139_139353


namespace arithmetic_expression_evaluation_l139_139354

theorem arithmetic_expression_evaluation :
  (1 / 6 * -6 / (-1 / 6) * 6) = 36 :=
by {
  sorry
}

end arithmetic_expression_evaluation_l139_139354


namespace units_digit_n_l139_139031

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31 ^ 6) (h2 : m % 10 = 9) : n % 10 = 2 := 
sorry

end units_digit_n_l139_139031


namespace determine_x_l139_139430

/-
  Determine \( x \) when \( y = 19 \)
  given the ratio of \( 5x - 3 \) to \( y + 10 \) is constant,
  and when \( x = 3 \), \( y = 4 \).
-/

theorem determine_x (x y k : ℚ) (h1 : ∀ x y, (5 * x - 3) / (y + 10) = k)
  (h2 : 5 * 3 - 3 / (4 + 10) = k) : x = 39 / 7 :=
sorry

end determine_x_l139_139430


namespace extremum_of_cubic_function_l139_139745

noncomputable def cubic_function (x : ℝ) : ℝ := 2 - x^2 - x^3

theorem extremum_of_cubic_function : 
  ∃ x_max x_min : ℝ, 
    cubic_function x_max = x_max_value ∧ 
    cubic_function x_min = x_min_value ∧ 
    ∀ x : ℝ, cubic_function x ≤ cubic_function x_max ∧ cubic_function x_min ≤ cubic_function x :=
sorry

end extremum_of_cubic_function_l139_139745


namespace cone_prism_volume_ratio_correct_l139_139910

noncomputable def cone_prism_volume_ratio (π : ℝ) : ℝ :=
  let r := 1.5
  let h := 5
  let V_cone := (1 / 3) * π * r^2 * h
  let V_prism := 3 * 4 * h
  V_cone / V_prism

theorem cone_prism_volume_ratio_correct (π : ℝ) : 
  cone_prism_volume_ratio π = π / 4.8 :=
sorry

end cone_prism_volume_ratio_correct_l139_139910


namespace probability_second_third_different_colors_l139_139399

def probability_different_colors (blue_chips : ℕ) (red_chips : ℕ) (yellow_chips : ℕ) : ℚ :=
  let total_chips := blue_chips + red_chips + yellow_chips
  let prob_diff :=
    ((blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips)) +
    ((red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips)) +
    ((yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips))
  prob_diff

theorem probability_second_third_different_colors :
  probability_different_colors 7 6 5 = 107 / 162 :=
by
  sorry

end probability_second_third_different_colors_l139_139399


namespace problem_statement_l139_139021

noncomputable def f : ℝ → ℝ := sorry  -- Define f as a noncomputable function to accommodate the problem constraints

variables (a : ℝ)

theorem problem_statement (periodic_f : ∀ x, f (x + 3) = f x)
    (odd_f : ∀ x, f (-x) = -f x)
    (ineq_f1 : f 1 < 1)
    (eq_f2 : f 2 = (2*a-1)/(a+1)) :
    a < -1 ∨ 0 < a :=
by
  sorry

end problem_statement_l139_139021


namespace proof_a_plus_2b_equal_7_l139_139453

theorem proof_a_plus_2b_equal_7 (a b : ℕ) (h1 : 82 * 1000 + a * 10 + 7 + 6 * b = 190) (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 1 ≤ b) (h5 : b < 10) : 
  a + 2 * b = 7 :=
by sorry

end proof_a_plus_2b_equal_7_l139_139453


namespace find_m_for_parallel_lines_l139_139577

noncomputable def parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 (m : ℝ) : Prop :=
  let l1_slope := -(1 + m) / 1
  let l2_slope := -m / 2
  l1_slope = l2_slope

theorem find_m_for_parallel_lines :
  parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 m →
  m = 1 :=
by
  intro h_parallel
  -- Here we would present the proof steps to show that m = 1 under the given conditions.
  sorry

end find_m_for_parallel_lines_l139_139577


namespace closest_point_on_line_l139_139202

structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def line (s : ℚ) : Point ℚ :=
⟨3 + s, 2 - 3 * s, 4 * s⟩

def distance (p1 p2 : Point ℚ) : ℚ :=
(p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

def closestPoint : Point ℚ := ⟨37/17, 74/17, -56/17⟩

def givenPoint : Point ℚ := ⟨1, 4, -2⟩

theorem closest_point_on_line :
  ∃ s : ℚ, line s = closestPoint ∧ 
           ∀ t : ℚ, distance closestPoint givenPoint ≤ distance (line t) givenPoint :=
by
  sorry

end closest_point_on_line_l139_139202


namespace actual_time_l139_139048

variables (m_pos : ℕ) (h_pos : ℕ)

-- The mirrored positions
def minute_hand_in_mirror : ℕ := 10
def hour_hand_in_mirror : ℕ := 5

theorem actual_time (m_pos h_pos : ℕ) 
  (hm : m_pos = 2) 
  (hh : h_pos < 7 ∧ h_pos ≥ 6) : 
  m_pos = 10 ∧ h_pos < 7 ∧ h_pos ≥ 6 :=
sorry

end actual_time_l139_139048


namespace run_time_is_48_minutes_l139_139203

noncomputable def cycling_speed : ℚ := 5 / 2
noncomputable def running_speed : ℚ := cycling_speed * 0.5
noncomputable def walking_speed : ℚ := running_speed * 0.5

theorem run_time_is_48_minutes (d : ℚ) (h : (d / cycling_speed) + (d / walking_speed) = 2) : 
  (60 * d / running_speed) = 48 :=
by
  sorry

end run_time_is_48_minutes_l139_139203


namespace flower_problem_solution_l139_139108

/-
Given the problem conditions:
1. There are 88 flowers.
2. Each flower was visited by at least one bee.
3. Each bee visited exactly 54 flowers.

Prove that bitter flowers exceed sweet flowers by 14.
-/

noncomputable def flower_problem : Prop :=
  ∃ (s g : ℕ), 
    -- Condition: The total number of flowers
    s + g + (88 - s - g) = 88 ∧ 
    -- Condition: Total number of visits by bees
    3 * 54 = 162 ∧ 
    -- Proof goal: Bitter flowers exceed sweet flowers by 14
    g - s = 14

theorem flower_problem_solution : flower_problem :=
by
  sorry

end flower_problem_solution_l139_139108


namespace P_in_first_quadrant_l139_139393

def point_in_first_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 > 0

theorem P_in_first_quadrant (k : ℝ) (h : k > 0) : point_in_first_quadrant (3, k) :=
by
  sorry

end P_in_first_quadrant_l139_139393


namespace find_number_l139_139329

-- Given conditions:
def sum_and_square (n : ℕ) : Prop := n^2 + n = 252
def is_factor (n d : ℕ) : Prop := d % n = 0

-- Equivalent proof problem statement
theorem find_number : ∃ n : ℕ, sum_and_square n ∧ is_factor n 180 ∧ n > 0 ∧ n = 14 :=
by
  sorry

end find_number_l139_139329


namespace part1_part2_l139_139852

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (x - 1) + a
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + Real.log x

theorem part1 (x : ℝ) (hx : 0 < x) :
  f x 0 ≥ g x 0 + 1 := sorry

theorem part2 {x0 : ℝ} (hx0 : ∃ y0 : ℝ, f x0 0 = g x0 0 ∧ ∀ x ≠ x0, f x 0 ≠ g x 0) :
  x0 < 2 := sorry

end part1_part2_l139_139852


namespace henri_total_time_l139_139167

variable (m1 m2 : ℝ) (r w : ℝ)

theorem henri_total_time (H1 : m1 = 3.5) (H2 : m2 = 1.5) (H3 : r = 10) (H4 : w = 1800) :
    m1 + m2 + w / r / 60 = 8 := by
  sorry

end henri_total_time_l139_139167


namespace distance_from_point_to_focus_l139_139379

theorem distance_from_point_to_focus (x0 : ℝ) (h1 : (2 * Real.sqrt 3)^2 = 4 * x0) :
    x0 + 1 = 4 := by
  sorry

end distance_from_point_to_focus_l139_139379


namespace quadratic_equal_roots_iff_a_eq_4_l139_139603

theorem quadratic_equal_roots_iff_a_eq_4 (a : ℝ) (h : ∃ x : ℝ, (a * x^2 - 4 * x + 1 = 0) ∧ (a * x^2 - 4 * x + 1 = 0)) :
  a = 4 :=
by
  sorry

end quadratic_equal_roots_iff_a_eq_4_l139_139603


namespace arithmetic_seq_problem_l139_139004

theorem arithmetic_seq_problem
  (a : ℕ → ℤ)  -- sequence a_n is an arithmetic sequence
  (h0 : ∃ (a1 d : ℤ), ∀ (n : ℕ), a n = a1 + n * d)  -- exists a1 and d such that a_n = a1 + n * d
  (h1 : a 0 + 3 * a 7 + a 14 = 120) :                -- given a1 + 3a8 + a15 = 120
  3 * a 8 - a 10 = 48 :=                             -- prove 3a9 - a11 = 48
sorry

end arithmetic_seq_problem_l139_139004


namespace tallest_stack_is_b_l139_139251

def number_of_pieces_a : ℕ := 8
def number_of_pieces_b : ℕ := 11
def number_of_pieces_c : ℕ := 6

def height_per_piece_a : ℝ := 2
def height_per_piece_b : ℝ := 1.5
def height_per_piece_c : ℝ := 2.5

def total_height_a : ℝ := number_of_pieces_a * height_per_piece_a
def total_height_b : ℝ := number_of_pieces_b * height_per_piece_b
def total_height_c : ℝ := number_of_pieces_c * height_per_piece_c

theorem tallest_stack_is_b : (total_height_b = 16.5) ∧ (total_height_b > total_height_a) ∧ (total_height_b > total_height_c) := 
by
  sorry

end tallest_stack_is_b_l139_139251


namespace min_cut_length_no_triangle_l139_139669

theorem min_cut_length_no_triangle (a b c x : ℝ) 
  (h_y : a = 7) 
  (h_z : b = 24) 
  (h_w : c = 25) 
  (h1 : a - x > 0)
  (h2 : b - x > 0)
  (h3 : c - x > 0)
  (h4 : (a - x) + (b - x) ≤ (c - x)) :
  x = 6 :=
by
  sorry

end min_cut_length_no_triangle_l139_139669


namespace mod_2_pow_1000_by_13_l139_139151

theorem mod_2_pow_1000_by_13 :
  (2 ^ 1000) % 13 = 3 := by
  sorry

end mod_2_pow_1000_by_13_l139_139151


namespace rectangle_diagonal_opposite_vertex_l139_139626

theorem rectangle_diagonal_opposite_vertex :
  ∀ (x y : ℝ),
    (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
      (x1, y1) = (5, 10) ∧ (x2, y2) = (15, -6) ∧ (x3, y3) = (11, 2) ∧
      (∃ (mx my : ℝ), mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2 ∧
        mx = (x + x3) / 2 ∧ my = (y + y3) / 2) ∧
      x = 9 ∧ y = 2) :=
by
  sorry

end rectangle_diagonal_opposite_vertex_l139_139626


namespace cos_monotonic_increasing_interval_l139_139405

open Real

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6}

theorem cos_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ,
    (∃ y, y = cos (π / 3 - 2 * x)) →
    (monotonic_increasing_interval k x) :=
by
  sorry

end cos_monotonic_increasing_interval_l139_139405


namespace combined_weight_l139_139173

theorem combined_weight (y z : ℝ) 
  (h_avg : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y + z) / 6) :
  y + z = 62 :=
by
  sorry

end combined_weight_l139_139173


namespace mulberry_sales_l139_139853

theorem mulberry_sales (x : ℝ) (p : ℝ) (h1 : 3000 = x * p)
    (h2 : 150 * (p * 1.4) + (x - 150) * (p * 0.8) - 3000 = 750) :
    x = 200 := by sorry

end mulberry_sales_l139_139853


namespace percentage_reduction_is_20_l139_139580

noncomputable def reduction_in_length (L W : ℝ) (x : ℝ) := 
  (L * (1 - x / 100)) * (W * 1.25) = L * W

theorem percentage_reduction_is_20 (L W : ℝ) : 
  reduction_in_length L W 20 := 
by 
  unfold reduction_in_length
  sorry

end percentage_reduction_is_20_l139_139580


namespace unique_diff_of_cubes_l139_139933

theorem unique_diff_of_cubes (n k : ℕ) (h : 61 = n^3 - k^3) : n = 5 ∧ k = 4 :=
sorry

end unique_diff_of_cubes_l139_139933


namespace height_of_first_building_l139_139518

theorem height_of_first_building (h : ℕ) (h_condition : h + 2 * h + 9 * h = 7200) : h = 600 :=
by
  sorry

end height_of_first_building_l139_139518


namespace fraction_inequality_solution_set_l139_139974

theorem fraction_inequality_solution_set : 
  {x : ℝ | (2 - x) / (x + 4) > 0} = {x : ℝ | -4 < x ∧ x < 2} :=
by sorry

end fraction_inequality_solution_set_l139_139974


namespace unique_real_root_of_quadratic_l139_139918

theorem unique_real_root_of_quadratic (k : ℝ) :
  (∃ a : ℝ, ∀ b : ℝ, ((k^2 - 9) * b^2 - 2 * (k + 1) * b + 1 = 0 → b = a)) ↔ (k = 3 ∨ k = -3 ∨ k = -5) :=
by
  sorry

end unique_real_root_of_quadratic_l139_139918


namespace arithmetic_sequence_a9_l139_139355

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Assume arithmetic sequence: a(n) = a1 + (n-1)d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ := a 1 + (n - 1) * d

-- Given conditions
axiom condition1 : arithmetic_sequence a d 5 + arithmetic_sequence a d 7 = 16
axiom condition2 : arithmetic_sequence a d 3 = 1

-- Prove that a₉ = 15
theorem arithmetic_sequence_a9 : arithmetic_sequence a d 9 = 15 := by
  sorry

end arithmetic_sequence_a9_l139_139355


namespace min_value_l139_139793

theorem min_value (a : ℝ) (h : a > 0) : a + 4 / a ≥ 4 :=
by sorry

end min_value_l139_139793


namespace x_eq_one_l139_139946

theorem x_eq_one (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (div_cond : ∀ n : ℕ, 0 < n → (2^n * y + 1) ∣ (x^(2^n) - 1)) : x = 1 := by
  sorry

end x_eq_one_l139_139946


namespace no_intersection_of_ellipses_l139_139984

theorem no_intersection_of_ellipses :
  (∀ (x y : ℝ), (9*x^2 + y^2 = 9) ∧ (x^2 + 16*y^2 = 16) → false) :=
sorry

end no_intersection_of_ellipses_l139_139984


namespace count_integers_l139_139739

def satisfies_conditions (n : ℤ) (r : ℤ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5

theorem count_integers (n : ℤ) (r : ℤ) :
  (satisfies_conditions n r) → ∃! n, 200 < n ∧ n < 300 ∧ ∃ r, n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5 :=
by
  sorry

end count_integers_l139_139739


namespace find_integer_b_l139_139760

theorem find_integer_b (z : ℝ) : ∃ b : ℝ, (z^2 - 6*z + 17 = (z - 3)^2 + b) ∧ b = 8 :=
by
  -- The proof would go here
  sorry

end find_integer_b_l139_139760


namespace area_triangle_ABC_l139_139624

theorem area_triangle_ABC (x y : ℝ) (h : x * y ≠ 0) (hAOB : 1 / 2 * |x * y| = 4) : 
  1 / 2 * |(x * (-2 * y) + x * (2 * y) + (-x) * (2 * y))| = 8 :=
by
  sorry

end area_triangle_ABC_l139_139624


namespace wang_hao_not_last_l139_139811

theorem wang_hao_not_last (total_players : ℕ) (players_to_choose : ℕ) 
  (wang_hao : ℕ) (ways_to_choose_if_not_last : ℕ) : 
  total_players = 6 ∧ players_to_choose = 3 → 
  ways_to_choose_if_not_last = 100 := 
by
  sorry

end wang_hao_not_last_l139_139811


namespace intersection_of_asymptotes_l139_139564

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6 * x + 8) / (x^2 - 6 * x + 9)

theorem intersection_of_asymptotes :
  ∃ (p : ℝ × ℝ), p = (3, 1) ∧
    (∀ (x : ℝ), x ≠ 3 → f x ≠ 1) ∧
    ((∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 3| ∧ |x - 3| < δ → |f x - 1| < ε) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - 1| ∧ |y - 1| < δ → |f (3 + y) - 1| < ε)) :=
by
  sorry

end intersection_of_asymptotes_l139_139564


namespace triangle_inequality_l139_139361

variables {a b c x y z : ℝ}

theorem triangle_inequality 
  (h1 : ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  (h2 : x + y + z = 0) :
  a^2 * y * z + b^2 * z * x + c^2 * x * y ≤ 0 :=
sorry

end triangle_inequality_l139_139361


namespace infinite_gcd_one_l139_139737

theorem infinite_gcd_one : ∃ᶠ n in at_top, Int.gcd n ⌊Real.sqrt 2 * n⌋ = 1 := sorry

end infinite_gcd_one_l139_139737


namespace domain_lg_function_l139_139600

theorem domain_lg_function (x : ℝ) : (1 + x > 0 ∧ x - 1 > 0) ↔ (1 < x) :=
by
  sorry

end domain_lg_function_l139_139600


namespace percentage_increase_of_gross_sales_l139_139735

theorem percentage_increase_of_gross_sales 
  (P R : ℝ) 
  (orig_gross new_price new_qty new_gross : ℝ)
  (h1 : new_price = 0.8 * P)
  (h2 : new_qty = 1.8 * R)
  (h3 : orig_gross = P * R)
  (h4 : new_gross = new_price * new_qty) :
  ((new_gross - orig_gross) / orig_gross) * 100 = 44 :=
by sorry

end percentage_increase_of_gross_sales_l139_139735


namespace initial_caps_correct_l139_139914

variable (bought : ℕ)
variable (total : ℕ)

def initial_bottle_caps (bought : ℕ) (total : ℕ) : ℕ :=
  total - bought

-- Given conditions
def bought_caps : ℕ := 7
def total_caps : ℕ := 47

theorem initial_caps_correct : initial_bottle_caps bought_caps total_caps = 40 :=
by
  -- proof here
  sorry

end initial_caps_correct_l139_139914


namespace find_x_l139_139364

variable (m k x Km2 mk : ℚ)

def valid_conditions (m k : ℚ) : Prop :=
  m > 2 * k ∧ k > 0

def initial_acid (m : ℚ) : ℚ :=
  (m*m)/100

def diluted_acid (m k x : ℚ) : ℚ :=
  ((2*m) - k) * (m + x) / 100

theorem find_x (m k : ℚ) (h : valid_conditions m k):
  ∃ x : ℚ, (m^2 = diluted_acid m k x) ∧ x = (k * m - m^2) / (2 * m - k) :=
sorry

end find_x_l139_139364


namespace problem1_solution_correct_problem2_solution_correct_l139_139611

def problem1 (x : ℤ) : Prop := (x - 1) ∣ (x + 3)
def problem2 (x : ℤ) : Prop := (x + 2) ∣ (x^2 + 2)
def solution1 (x : ℤ) : Prop := x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5
def solution2 (x : ℤ) : Prop := x = -8 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 4

theorem problem1_solution_correct : ∀ x: ℤ, problem1 x ↔ solution1 x := by
  sorry

theorem problem2_solution_correct : ∀ x: ℤ, problem2 x ↔ solution2 x := by
  sorry

end problem1_solution_correct_problem2_solution_correct_l139_139611


namespace train_ride_cost_difference_l139_139746

-- Definitions based on the conditions
def bus_ride_cost : ℝ := 1.40
def total_cost : ℝ := 9.65

-- Lemma to prove the mathematical question
theorem train_ride_cost_difference :
  ∃ T : ℝ, T + bus_ride_cost = total_cost ∧ (T - bus_ride_cost) = 6.85 :=
by
  sorry

end train_ride_cost_difference_l139_139746


namespace product_three_numbers_l139_139442

theorem product_three_numbers 
  (a b c : ℝ)
  (h1 : a + b + c = 30)
  (h2 : a = 3 * (b + c))
  (h3 : b = 5 * c) : 
  a * b * c = 176 := 
by
  sorry

end product_three_numbers_l139_139442


namespace isosceles_trapezoid_height_l139_139381

/-- Given an isosceles trapezoid with area 100 and diagonals that are mutually perpendicular,
    we want to prove that the height of the trapezoid is 10. -/
theorem isosceles_trapezoid_height (BC AD h : ℝ) 
    (area_eq_100 : 100 = (1 / 2) * (BC + AD) * h)
    (height_eq_half_sum : h = (1 / 2) * (BC + AD)) :
    h = 10 :=
by
  sorry

end isosceles_trapezoid_height_l139_139381


namespace inverse_proportional_fraction_l139_139936

theorem inverse_proportional_fraction (N : ℝ) (d f : ℝ) (h : N ≠ 0):
  d * f = N :=
sorry

end inverse_proportional_fraction_l139_139936


namespace simplify_fraction_l139_139311

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (hx2 : x^2 - (1 / y) ≠ 0) (hy2 : y^2 - (1 / x) ≠ 0) :
  (x^2 - 1 / y) / (y^2 - 1 / x) = (x * (x^2 * y - 1)) / (y * (y^2 * x - 1)) :=
sorry

end simplify_fraction_l139_139311


namespace even_function_b_eq_zero_l139_139967

theorem even_function_b_eq_zero (b : ℝ) :
  (∀ x : ℝ, (x^2 + b * x) = (x^2 - b * x)) → b = 0 :=
by sorry

end even_function_b_eq_zero_l139_139967


namespace minimize_x_2y_l139_139030

noncomputable def minimum_value_x_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 3 / (x + 2) + 3 / (y + 2) = 1) : ℝ :=
  x + 2 * y

theorem minimize_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / (x + 2) + 3 / (y + 2) = 1) :
  minimum_value_x_2y x y hx hy h = 3 + 6 * Real.sqrt 2 :=
sorry

end minimize_x_2y_l139_139030


namespace maia_daily_client_requests_l139_139094

theorem maia_daily_client_requests (daily_requests : ℕ) (remaining_requests : ℕ) (days : ℕ) 
  (received_requests : ℕ) (total_requests : ℕ) (worked_requests : ℕ) :
  (daily_requests = 6) →
  (remaining_requests = 10) →
  (days = 5) →
  (received_requests = daily_requests * days) →
  (total_requests = received_requests - remaining_requests) →
  (worked_requests = total_requests / days) →
  worked_requests = 4 :=
by
  sorry

end maia_daily_client_requests_l139_139094


namespace equation_completing_square_l139_139728

theorem equation_completing_square :
  ∃ (a b c : ℤ), 64 * x^2 + 80 * x - 81 = 0 → 
  (a > 0) ∧ (2 * a * b = 80) ∧ (a^2 = 64) ∧ (a + b + c = 119) :=
sorry

end equation_completing_square_l139_139728


namespace enclosed_area_eq_32_over_3_l139_139276

def line (x : ℝ) : ℝ := 2 * x + 3
def parabola (x : ℝ) : ℝ := x^2

theorem enclosed_area_eq_32_over_3 :
  ∫ x in (-(1:ℝ))..(3:ℝ), (line x - parabola x) = 32 / 3 :=
by
  sorry

end enclosed_area_eq_32_over_3_l139_139276


namespace maximum_area_right_triangle_hypotenuse_8_l139_139265

theorem maximum_area_right_triangle_hypotenuse_8 :
  ∃ a b : ℝ, (a^2 + b^2 = 64) ∧ (a * b) / 2 = 16 :=
by
  sorry

end maximum_area_right_triangle_hypotenuse_8_l139_139265


namespace find_C_and_D_l139_139962

theorem find_C_and_D (C D : ℚ) :
  (∀ x : ℚ, ((6 * x - 8) / (2 * x^2 + 5 * x - 3) = (C / (x - 1)) + (D / (2 * x + 3)))) →
  (2*x^2 + 5*x - 3 = (2*x - 1)*(x + 3)) →
  (∀ x : ℚ, ((C*(2*x + 3) + D*(x - 1)) / ((2*x - 1)*(x + 3))) = ((6*x - 8) / ((2*x - 1)*(x + 3)))) →
  (∀ x : ℚ, C*(2*x + 3) + D*(x - 1) = 6*x - 8) →
  C = -2/5 ∧ D = 34/5 := 
by 
  sorry

end find_C_and_D_l139_139962


namespace determine_x_l139_139134

theorem determine_x (x : ℝ) (h : (1 / (Real.log x / Real.log 3) + 1 / (Real.log x / Real.log 5) + 1 / (Real.log x / Real.log 6) = 1)) : 
    x = 90 := 
by 
  sorry

end determine_x_l139_139134


namespace illegally_parked_percentage_l139_139968

theorem illegally_parked_percentage (total_cars : ℕ) (towed_cars : ℕ)
  (ht : towed_cars = 2 * total_cars / 100) (not_towed_percentage : ℕ)
  (hp : not_towed_percentage = 80) : 
  (100 * (5 * towed_cars) / total_cars) = 10 :=
by
  sorry

end illegally_parked_percentage_l139_139968


namespace square_remainder_is_square_l139_139140

theorem square_remainder_is_square (a : ℤ) : ∃ b : ℕ, (a^2 % 16 = b) ∧ (∃ c : ℕ, b = c^2) :=
by
  sorry

end square_remainder_is_square_l139_139140


namespace shelly_thread_length_l139_139168

theorem shelly_thread_length 
  (threads_per_keychain : ℕ := 12) 
  (friends_in_class : ℕ := 6) 
  (friends_from_clubs := friends_in_class / 2)
  (total_friends := friends_in_class + friends_from_clubs) 
  (total_threads_needed := total_friends * threads_per_keychain) : 
  total_threads_needed = 108 := 
by 
  -- proof skipped
  sorry

end shelly_thread_length_l139_139168


namespace parabola_focus_l139_139411

theorem parabola_focus (x y p : ℝ) (h_eq : y = 2 * x^2) (h_standard_form : x^2 = (1 / 2) * y) (h_p : p = 1 / 4) : 
    (0, p / 2) = (0, 1 / 8) := by
    sorry

end parabola_focus_l139_139411


namespace math_problem_l139_139114

open Real

variable (x : ℝ)
variable (h : x + 1 / x = sqrt 3)

theorem math_problem : x^7 - 3 * x^5 + x^2 = -5 * x + 4 * sqrt 3 :=
by sorry

end math_problem_l139_139114


namespace central_angle_of_section_l139_139279

theorem central_angle_of_section (A : ℝ) (hA : 0 < A) (prob : ℝ) (hprob : prob = 1 / 4) :
  ∃ θ : ℝ, (θ / 360) = prob :=
by
  use 90
  sorry

end central_angle_of_section_l139_139279


namespace weight_of_white_ring_l139_139677

def weight_orange := 0.08333333333333333
def weight_purple := 0.3333333333333333
def total_weight := 0.8333333333

def weight_white := 0.41666666663333337

theorem weight_of_white_ring :
  weight_white + weight_orange + weight_purple = total_weight :=
by
  sorry

end weight_of_white_ring_l139_139677


namespace speaker_is_tweedledee_l139_139840

-- Definitions
variable (Speaks : Prop) (is_tweedledum : Prop) (has_black_card : Prop)

-- Condition: If the speaker is Tweedledum, then the card in the speaker's pocket is not a black suit.
axiom A1 : is_tweedledum → ¬ has_black_card

-- Goal: Prove that the speaker is Tweedledee.
theorem speaker_is_tweedledee (h1 : Speaks) : ¬ is_tweedledum :=
by
  sorry

end speaker_is_tweedledee_l139_139840


namespace angle_difference_l139_139316

-- Define the conditions
variables (A B : ℝ) 

def is_parallelogram := A + B = 180
def smaller_angle := A = 70
def larger_angle := B = 180 - 70

-- State the theorem to be proved
theorem angle_difference (A B : ℝ) (h1 : is_parallelogram A B) (h2 : smaller_angle A) : B - A = 40 := by
  sorry

end angle_difference_l139_139316


namespace ellipse_product_l139_139795

/-- Given conditions:
1. OG = 8
2. The diameter of the inscribed circle of triangle ODG is 4
3. O is the center of an ellipse with major axis AB and minor axis CD
4. Point G is one focus of the ellipse
--/
theorem ellipse_product :
  ∀ (O G D : Point) (a b : ℝ),
    OG = 8 → 
    (a^2 - b^2 = 64) →
    (a - b = 4) →
    (AB = 2*a) →
    (CD = 2*b) →
    (AB * CD = 240) :=
by
  intros O G D a b hOG h1 h2 h3 h4
  sorry

end ellipse_product_l139_139795


namespace sum_of_reciprocal_AP_l139_139466

theorem sum_of_reciprocal_AP (a1 a2 a3 : ℝ) (d : ℝ)
  (h1 : a1 + a2 + a3 = 11/18)
  (h2 : 1/a1 + 1/a2 + 1/a3 = 18)
  (h3 : 1/a2 = 1/a1 + d)
  (h4 : 1/a3 = 1/a1 + 2*d) :
  (a1 = 1/9 ∧ a2 = 1/6 ∧ a3 = 1/3) ∨ (a1 = 1/3 ∧ a2 = 1/6 ∧ a3 = 1/9) :=
sorry

end sum_of_reciprocal_AP_l139_139466


namespace ratio_of_spinsters_to_cats_l139_139869

-- Defining the problem in Lean 4
theorem ratio_of_spinsters_to_cats (S C : ℕ) (h₁ : S = 22) (h₂ : C = S + 55) : S / gcd S C = 2 ∧ C / gcd S C = 7 :=
by
  sorry

end ratio_of_spinsters_to_cats_l139_139869


namespace closest_point_l139_139696

noncomputable def point_on_line_closest_to (x y : ℝ) : ℝ × ℝ :=
( -11 / 5, 7 / 5 )

theorem closest_point (x y : ℝ) (h_line : y = 2 * x + 3) (h_point : (x, y) = (3, -4)) :
  point_on_line_closest_to x y = ( -11 / 5, 7 / 5 ) :=
sorry

end closest_point_l139_139696


namespace max_star_player_salary_l139_139787

-- Define the constants given in the problem
def num_players : Nat := 12
def min_salary : Nat := 20000
def total_salary_cap : Nat := 1000000

-- Define the statement we want to prove
theorem max_star_player_salary :
  (∃ star_player_salary : Nat, 
    star_player_salary ≤ total_salary_cap - (num_players - 1) * min_salary ∧
    star_player_salary = 780000) :=
sorry

end max_star_player_salary_l139_139787


namespace melinda_probability_correct_l139_139071

def probability_two_digit_between_20_and_30 : ℚ :=
  11 / 36

theorem melinda_probability_correct :
  probability_two_digit_between_20_and_30 = 11 / 36 :=
by
  sorry

end melinda_probability_correct_l139_139071
