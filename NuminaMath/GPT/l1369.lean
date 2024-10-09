import Mathlib

namespace solve_equation_l1369_136915

theorem solve_equation :
  ∀ x : ℝ, (x * (2 * x + 4) = 10 + 5 * x) ↔ (x = -2 ∨ x = 2.5) :=
by
  sorry

end solve_equation_l1369_136915


namespace infinite_n_perfect_squares_l1369_136921

-- Define the condition that k is a positive natural number and k >= 2
variable (k : ℕ) (hk : 2 ≤ k) 

-- Define the statement asserting the existence of infinitely many n such that both kn + 1 and (k+1)n + 1 are perfect squares
theorem infinite_n_perfect_squares : ∀ k : ℕ, (2 ≤ k) → ∃ n : ℕ, ∀ m : ℕ, (2 ≤ k) → k * n + 1 = m * m ∧ (k + 1) * n + 1 = (m + k) * (m + k) := 
by
  sorry

end infinite_n_perfect_squares_l1369_136921


namespace minimum_value_of_function_l1369_136936

noncomputable def function_y (x : ℝ) : ℝ := 1 / (Real.sqrt (x - x^2))

theorem minimum_value_of_function : (∀ x : ℝ, 0 < x ∧ x < 1 → function_y x ≥ 2) ∧ (∃ x : ℝ, 0 < x ∧ x < 1 ∧ function_y x = 2) :=
by
  sorry

end minimum_value_of_function_l1369_136936


namespace license_plate_count_l1369_136953

theorem license_plate_count : (26^3 * 5 * 5 * 4) = 1757600 := 
by 
  sorry

end license_plate_count_l1369_136953


namespace find_n_l1369_136981

theorem find_n (x : ℝ) (n : ℝ) (G : ℝ) (hG : G = (7*x^2 + 21*x + 5*n) / 7) :
  (∃ c d : ℝ, c^2 * x^2 + 2*c*d*x + d^2 = G) ↔ n = 63 / 20 :=
by
  sorry

end find_n_l1369_136981


namespace max_value_k_l1369_136941

noncomputable def sqrt_minus (x : ℝ) : ℝ := Real.sqrt (x - 3)
noncomputable def sqrt_six_minus (x : ℝ) : ℝ := Real.sqrt (6 - x)

theorem max_value_k (k : ℝ) : (∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ sqrt_minus x + sqrt_six_minus x ≥ k) ↔ k ≤ Real.sqrt 12 := by
  sorry

end max_value_k_l1369_136941


namespace max_n_satisfying_inequality_l1369_136970

theorem max_n_satisfying_inequality : 
  ∃ (n : ℤ), 303 * n^3 ≤ 380000 ∧ ∀ m : ℤ, m > n → 303 * m^3 > 380000 := sorry

end max_n_satisfying_inequality_l1369_136970


namespace angle_alpha_not_2pi_over_9_l1369_136996

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos (2 * x)) * (Real.cos (4 * x))

theorem angle_alpha_not_2pi_over_9 (α : ℝ) (h : f α = 1 / 8) : α ≠ 2 * π / 9 :=
sorry

end angle_alpha_not_2pi_over_9_l1369_136996


namespace complete_residue_system_mod_l1369_136951

open Nat

theorem complete_residue_system_mod (m : ℕ) (x : Fin m → ℕ)
  (h : ∀ i j : Fin m, i ≠ j → ¬ ((x i) % m = (x j) % m)) :
  (Finset.image (λ i => x i % m) (Finset.univ : Finset (Fin m))) = Finset.range m :=
by
  -- Skipping the proof steps.
  sorry

end complete_residue_system_mod_l1369_136951


namespace percentage_slump_in_business_l1369_136908

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.05 * Y = 0.04 * X) : (X > 0) → (Y > 0) → (X - Y) / X * 100 = 20 := 
by
  sorry

end percentage_slump_in_business_l1369_136908


namespace any_nat_as_fraction_form_l1369_136945

theorem any_nat_as_fraction_form (n : ℕ) : ∃ (x y : ℕ), x = n^3 ∧ y = n^2 ∧ (x^3 / y^4 : ℝ) = n :=
by
  sorry

end any_nat_as_fraction_form_l1369_136945


namespace find_positive_integers_l1369_136964

theorem find_positive_integers (n : ℕ) : 
  (∀ a : ℕ, a.gcd n = 1 → 2 * n * n ∣ a ^ n - 1) ↔ (n = 2 ∨ n = 6 ∨ n = 42 ∨ n = 1806) :=
sorry

end find_positive_integers_l1369_136964


namespace aspirin_mass_percentages_l1369_136963

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_aspirin : ℝ := (9 * atomic_mass_C) + (8 * atomic_mass_H) + (4 * atomic_mass_O)

theorem aspirin_mass_percentages :
  let mass_percent_H := ((8 * atomic_mass_H) / molar_mass_aspirin) * 100
  let mass_percent_C := ((9 * atomic_mass_C) / molar_mass_aspirin) * 100
  let mass_percent_O := ((4 * atomic_mass_O) / molar_mass_aspirin) * 100
  mass_percent_H = 4.48 ∧ mass_percent_C = 60.00 ∧ mass_percent_O = 35.52 :=
by
  -- Placeholder for the proof
  sorry

end aspirin_mass_percentages_l1369_136963


namespace plan_b_cheaper_than_plan_a_l1369_136934

theorem plan_b_cheaper_than_plan_a (x : ℕ) (h : 401 ≤ x) :
  2000 + 5 * x < 10 * x :=
by
  sorry

end plan_b_cheaper_than_plan_a_l1369_136934


namespace intersection_of_A_and_B_l1369_136905

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B_l1369_136905


namespace part_one_part_two_l1369_136903

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem part_one (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, (Real.log x + 1/x + 1 - a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

theorem part_two (a : ℝ) (h_pos : 0 < a) :
  (∀ x, (x - 1) * (f x a) ≥ 0) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end part_one_part_two_l1369_136903


namespace num_pairs_equals_one_l1369_136943

noncomputable def fractional_part (x : ℚ) : ℚ := x - x.floor

open BigOperators

theorem num_pairs_equals_one :
  ∃! (n : ℕ) (q : ℚ), 
    (0 < q ∧ q < 2000) ∧ 
    ¬ q.isInt ∧ 
    fractional_part (q^2) = fractional_part (n.choose 2000)
:= sorry

end num_pairs_equals_one_l1369_136943


namespace smallest_group_size_l1369_136973

theorem smallest_group_size (n : ℕ) (k : ℕ) (hk : k > 2) (h1 : n % 2 = 0) (h2 : n % k = 0) :
  n = 6 :=
sorry

end smallest_group_size_l1369_136973


namespace tom_pie_portion_l1369_136954

theorem tom_pie_portion :
  let pie_left := 5 / 8
  let friends := 4
  let portion_per_person := pie_left / friends
  portion_per_person = 5 / 32 := by
  sorry

end tom_pie_portion_l1369_136954


namespace max_competitors_l1369_136987

theorem max_competitors (P1 P2 P3 : ℕ → ℕ → ℕ)
(hP1 : ∀ i, 0 ≤ P1 i ∧ P1 i ≤ 7)
(hP2 : ∀ i, 0 ≤ P2 i ∧ P2 i ≤ 7)
(hP3 : ∀ i, 0 ≤ P3 i ∧ P3 i ≤ 7)
(hDistinct : ∀ i j, i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :
  ∃ n, n ≤ 64 ∧ ∀ k, k < n → (∀ i j, i < k → j < k → i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :=
sorry

end max_competitors_l1369_136987


namespace equal_divided_value_l1369_136940

def n : ℕ := 8^2022

theorem equal_divided_value : n / 4 = 4^3032 := 
by {
  -- We state the equivalence and details used in the proof.
  sorry
}

end equal_divided_value_l1369_136940


namespace min_value_f_l1369_136978

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin x + Real.sin (Real.pi / 2 + x)

theorem min_value_f : ∃ x : ℝ, f x = -2 := by
  sorry

end min_value_f_l1369_136978


namespace frames_sharing_point_with_line_e_l1369_136960

def frame_shares_common_point_with_line (n : ℕ) : Prop := 
  n = 0 ∨ n = 1 ∨ n = 9 ∨ n = 17 ∨ n = 25 ∨ n = 33 ∨ n = 41 ∨ n = 49 ∨
  n = 6 ∨ n = 14 ∨ n = 22 ∨ n = 30 ∨ n = 38 ∨ n = 46

theorem frames_sharing_point_with_line_e :
  ∀ (i : ℕ), i < 50 → frame_shares_common_point_with_line i = 
  (i = 0 ∨ i = 1 ∨ i = 9 ∨ i = 17 ∨ i = 25 ∨ i = 33 ∨ i = 41 ∨ i = 49 ∨
   i = 6 ∨ i = 14 ∨ i = 22 ∨ i = 30 ∨ i = 38 ∨ i = 46) := 
by 
  sorry

end frames_sharing_point_with_line_e_l1369_136960


namespace binomial_square_constant_l1369_136992

theorem binomial_square_constant :
  ∃ c : ℝ, (∀ x : ℝ, 9*x^2 - 21*x + c = (3*x + -3.5)^2) → c = 12.25 :=
by
  sorry

end binomial_square_constant_l1369_136992


namespace wheels_in_garage_l1369_136974

theorem wheels_in_garage :
  let bicycles := 9
  let cars := 16
  let single_axle_trailers := 5
  let double_axle_trailers := 3
  let wheels_per_bicycle := 2
  let wheels_per_car := 4
  let wheels_per_single_axle_trailer := 2
  let wheels_per_double_axle_trailer := 4
  let total_wheels := bicycles * wheels_per_bicycle + cars * wheels_per_car + single_axle_trailers * wheels_per_single_axle_trailer + double_axle_trailers * wheels_per_double_axle_trailer
  total_wheels = 104 := by
  sorry

end wheels_in_garage_l1369_136974


namespace intersection_A_B_complement_l1369_136995

def universal_set : Set ℝ := {x : ℝ | True}
def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def B_complement : Set ℝ := {x : ℝ | x ≤ 1}

theorem intersection_A_B_complement :
  (A ∩ B_complement) = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_B_complement_l1369_136995


namespace sale_in_third_month_l1369_136939

theorem sale_in_third_month
  (sale1 sale2 sale4 sale5 sale6 avg : ℝ)
  (n : ℕ)
  (h_sale1 : sale1 = 6235)
  (h_sale2 : sale2 = 6927)
  (h_sale4 : sale4 = 7230)
  (h_sale5 : sale5 = 6562)
  (h_sale6 : sale6 = 5191)
  (h_avg : avg = 6500)
  (h_n : n = 6) :
  ∃ sale3 : ℝ, sale3 = 6855 := by
  sorry

end sale_in_third_month_l1369_136939


namespace max_handshakes_25_people_l1369_136971

theorem max_handshakes_25_people : 
  (∃ n : ℕ, n = 25) → 
  (∀ p : ℕ, p ≤ 24) → 
  ∃ m : ℕ, m = 300 :=
by sorry

end max_handshakes_25_people_l1369_136971


namespace sticker_price_of_smartphone_l1369_136925

theorem sticker_price_of_smartphone (p : ℝ)
  (h1 : 0.90 * p - 100 = 0.80 * p - 20) : p = 800 :=
sorry

end sticker_price_of_smartphone_l1369_136925


namespace volleyball_club_members_l1369_136907

variables (B G : ℝ)

theorem volleyball_club_members (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 18) : B = 12 := by
  -- Mathematical steps and transformations done here to show B = 12
  sorry

end volleyball_club_members_l1369_136907


namespace football_team_starting_lineup_count_l1369_136957

theorem football_team_starting_lineup_count :
  let total_members := 12
  let offensive_lineman_choices := 4
  let quarterback_choices := 2
  let remaining_after_ol := total_members - 1 -- after choosing one offensive lineman
  let remaining_after_qb := remaining_after_ol - 1 -- after choosing one quarterback
  let running_back_choices := remaining_after_ol
  let wide_receiver_choices := remaining_after_qb - 1
  let tight_end_choices := remaining_after_qb - 2
  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 5760 := 
by
  sorry

end football_team_starting_lineup_count_l1369_136957


namespace problem_l1369_136962

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem problem : f (g 2) + g (f 2) = 38 / 5 :=
by
  sorry

end problem_l1369_136962


namespace andrew_grapes_purchase_l1369_136913

theorem andrew_grapes_purchase (G : ℕ) (rate_grape rate_mango total_paid total_mango_cost : ℕ)
  (h1 : rate_grape = 54)
  (h2 : rate_mango = 62)
  (h3 : total_paid = 1376)
  (h4 : total_mango_cost = 10 * rate_mango)
  (h5 : total_paid = rate_grape * G + total_mango_cost) : G = 14 := by
  sorry

end andrew_grapes_purchase_l1369_136913


namespace find_number_l1369_136990

theorem find_number (x : ℝ) (h : 0.60 * 50 = 0.45 * x + 16.5) : x = 30 :=
by
  sorry

end find_number_l1369_136990


namespace y1_lt_y2_l1369_136997

-- Definitions of conditions
def linear_function (x : ℝ) : ℝ := 2 * x + 1

def y1 : ℝ := linear_function (-3)
def y2 : ℝ := linear_function 4

-- Proof statement
theorem y1_lt_y2 : y1 < y2 :=
by
  -- The proof step is omitted
  sorry

end y1_lt_y2_l1369_136997


namespace cone_in_sphere_less_half_volume_l1369_136989

theorem cone_in_sphere_less_half_volume
  (R r m : ℝ)
  (h1 : m < 2 * R)
  (h2 : r <= R) :
  (1 / 3 * Real.pi * r^2 * m < 1 / 2 * 4 / 3 * Real.pi * R^3) :=
by
  sorry

end cone_in_sphere_less_half_volume_l1369_136989


namespace problem1_problem2_l1369_136998

-- Statement for Problem 1
theorem problem1 (x y : ℝ) : (x - y) ^ 2 + x * (x + 2 * y) = 2 * x ^ 2 + y ^ 2 :=
by sorry

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  ((-3 * x + 4) / (x - 1) + x) / ((x - 2) / (x ^ 2 - x)) = x ^ 2 - 2 * x :=
by sorry

end problem1_problem2_l1369_136998


namespace boats_meeting_distance_l1369_136923

theorem boats_meeting_distance (X : ℝ) 
  (H1 : ∃ (X : ℝ), (1200 - X) + 900 = X + 1200 + 300) 
  (H2 : X + 1200 + 300 = 2100 + X): 
  X = 300 :=
by
  sorry

end boats_meeting_distance_l1369_136923


namespace cost_prices_l1369_136935

variable {C1 C2 : ℝ}

theorem cost_prices (h1 : 0.30 * C1 - 0.15 * C1 = 120) (h2 : 0.25 * C2 - 0.10 * C2 = 150) :
  C1 = 800 ∧ C2 = 1000 := 
by
  sorry

end cost_prices_l1369_136935


namespace parabola_hyperbola_focus_l1369_136988

theorem parabola_hyperbola_focus {p : ℝ} :
  let focus_parabola := (p / 2, 0)
  let focus_hyperbola := (2, 0)
  focus_parabola = focus_hyperbola -> p = 4 :=
by
  intro h
  sorry

end parabola_hyperbola_focus_l1369_136988


namespace find_value_of_b_l1369_136948

theorem find_value_of_b (a b : ℕ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
sorry

end find_value_of_b_l1369_136948


namespace evaluate_expression_at_x_neg3_l1369_136930

theorem evaluate_expression_at_x_neg3 :
  (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 :=
by
  sorry

end evaluate_expression_at_x_neg3_l1369_136930


namespace max_sum_of_factors_of_1764_l1369_136994

theorem max_sum_of_factors_of_1764 :
  ∃ (a b : ℕ), a * b = 1764 ∧ a + b = 884 :=
by
  sorry

end max_sum_of_factors_of_1764_l1369_136994


namespace integer_squared_equals_product_l1369_136977

theorem integer_squared_equals_product : 
  3^8 * 3^12 * 2^5 * 2^10 = 1889568^2 :=
by
  sorry

end integer_squared_equals_product_l1369_136977


namespace distance_from_desk_to_fountain_l1369_136901

-- Problem definitions with given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Formulate the proof problem as a Lean theorem statement
theorem distance_from_desk_to_fountain :
  total_distance / trips = 30 :=
by
  sorry

end distance_from_desk_to_fountain_l1369_136901


namespace equivalent_proof_problem_l1369_136933

-- Define the conditions as Lean 4 definitions
variable (x₁ x₂ : ℝ)

-- The conditions given in the problem
def condition1 : Prop := x₁ * Real.logb 2 x₁ = 1008
def condition2 : Prop := x₂ * 2^x₂ = 1008

-- The problem to be proved
theorem equivalent_proof_problem (hx₁ : condition1 x₁) (hx₂ : condition2 x₂) : 
  x₁ * x₂ = 1008 := 
sorry

end equivalent_proof_problem_l1369_136933


namespace problem_statement_l1369_136983

def has_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - m * x - 1 = 0

def p : Prop := ∀ m : ℝ, has_solutions m

def q : Prop := ∃ x_0 : ℕ, x_0^2 - 2 * x_0 - 1 ≤ 0

theorem problem_statement : ¬ (p ∧ ¬ q) := 
sorry

end problem_statement_l1369_136983


namespace find_number_l1369_136906

theorem find_number (x : ℝ) (h : x / 3 = x - 4) : x = 6 := 
by 
  sorry

end find_number_l1369_136906


namespace intersecting_lines_l1369_136984

theorem intersecting_lines (a b c d : ℝ) (h₁ : a ≠ b) (h₂ : ∃ x y : ℝ, y = a*x + a ∧ y = b*x + b ∧ y = c*x + d) : c = d :=
sorry

end intersecting_lines_l1369_136984


namespace parabola_equation_l1369_136958

def equation_of_parabola (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
              (∃ a : ℝ, y = a * (x - 3)^2 + 5) ∧
              y = (if x = 0 then 2 else y)

theorem parabola_equation :
  equation_of_parabola (-1 / 3) 2 2 :=
by
  -- First, show that the vertex form (x-3)^2 + 5 meets the conditions
  sorry

end parabola_equation_l1369_136958


namespace ellas_coins_worth_l1369_136942

theorem ellas_coins_worth :
  ∀ (n d : ℕ), n + d = 18 → n = d + 2 → 5 * n + 10 * d = 130 := by
  intros n d h1 h2
  sorry

end ellas_coins_worth_l1369_136942


namespace problem_l1369_136932

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem problem : nabla (nabla 1 3) 2 = 67 :=
by
  sorry

end problem_l1369_136932


namespace find_rate_of_stream_l1369_136926

noncomputable def rate_of_stream (v : ℝ) : Prop :=
  let rowing_speed := 36
  let downstream_speed := rowing_speed + v
  let upstream_speed := rowing_speed - v
  (1 / upstream_speed) = 3 * (1 / downstream_speed)

theorem find_rate_of_stream : ∃ v : ℝ, rate_of_stream v ∧ v = 18 :=
by
  use 18
  unfold rate_of_stream
  sorry

end find_rate_of_stream_l1369_136926


namespace numerator_is_12_l1369_136904

theorem numerator_is_12 (x : ℕ) (h1 : (x : ℤ) / (2 * x + 4 : ℤ) = 3 / 7) : x = 12 := 
sorry

end numerator_is_12_l1369_136904


namespace zou_mei_competition_l1369_136928

theorem zou_mei_competition (n : ℕ) (h1 : 271 = n^2 + 15) (h2 : n^2 + 33 = (n + 1)^2) : 
  ∃ n, 271 = n^2 + 15 ∧ n^2 + 33 = (n + 1)^2 :=
by
  existsi n
  exact ⟨h1, h2⟩

end zou_mei_competition_l1369_136928


namespace cos_30_eq_sqrt3_div_2_l1369_136991

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_30_eq_sqrt3_div_2_l1369_136991


namespace probability_P_plus_S_is_two_less_than_multiple_of_7_l1369_136965

def is_distinct (a b : ℕ) : Prop :=
  a ≠ b

def in_range (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100

def mod_condition (a b : ℕ) : Prop :=
  (a * b + a + b) % 7 = 5

noncomputable def probability_p_s (p q : ℕ) : ℚ :=
  p / q

theorem probability_P_plus_S_is_two_less_than_multiple_of_7 :
  probability_p_s (1295) (4950) = 259 / 990 := 
sorry

end probability_P_plus_S_is_two_less_than_multiple_of_7_l1369_136965


namespace smaller_factor_of_4851_l1369_136920

-- Define the condition
def product_lim (m n : ℕ) : Prop := m * n = 4851 ∧ 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100

-- The lean theorem statement
theorem smaller_factor_of_4851 : ∃ m n : ℕ, product_lim m n ∧ m = 49 := 
by {
    sorry
}

end smaller_factor_of_4851_l1369_136920


namespace highest_elevation_l1369_136922

   noncomputable def elevation (t : ℝ) : ℝ := 240 * t - 24 * t^2

   theorem highest_elevation : ∃ t : ℝ, elevation t = 600 ∧ ∀ x : ℝ, elevation x ≤ 600 := 
   sorry
   
end highest_elevation_l1369_136922


namespace mean_age_euler_family_l1369_136969

theorem mean_age_euler_family :
  let ages := [6, 6, 9, 11, 13, 16]
  let total_children := 6
  let total_sum := 61
  (total_sum / total_children : ℝ) = (61 / 6 : ℝ) :=
by
  sorry

end mean_age_euler_family_l1369_136969


namespace prove_pattern_example_l1369_136975

noncomputable def pattern_example : Prop :=
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  (123456 * 9 + 7 = 1111111)

theorem prove_pattern_example : pattern_example := by
  sorry

end prove_pattern_example_l1369_136975


namespace selection_schemes_count_l1369_136944

theorem selection_schemes_count :
  let total_teachers := 9
  let select_from_total := Nat.choose 9 3
  let select_all_male := Nat.choose 5 3
  let select_all_female := Nat.choose 4 3
  select_from_total - (select_all_male + select_all_female) = 420 := by
    sorry

end selection_schemes_count_l1369_136944


namespace employed_male_percent_problem_l1369_136950

noncomputable def employed_percent_population (total_population_employed_percent : ℝ) (employed_females_percent : ℝ) : ℝ :=
  let employed_males_percent := (1 - employed_females_percent) * total_population_employed_percent
  employed_males_percent

theorem employed_male_percent_problem :
  employed_percent_population 0.72 0.50 = 0.36 := by
  sorry

end employed_male_percent_problem_l1369_136950


namespace product_of_repeating_decimals_l1369_136956

noncomputable def repeating_decimal_038 : ℚ := 38 / 999
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem product_of_repeating_decimals :
  repeating_decimal_038 * repeating_decimal_4 = 152 / 8991 :=
by
  sorry

end product_of_repeating_decimals_l1369_136956


namespace find_m_and_other_root_l1369_136911

theorem find_m_and_other_root (m : ℝ) (r : ℝ) :
    (∃ x : ℝ, x^2 + m*x - 2 = 0) ∧ (x = -1) → (m = -1 ∧ r = 2) :=
by
  sorry

end find_m_and_other_root_l1369_136911


namespace number_of_girls_in_club_l1369_136967

theorem number_of_girls_in_club (total : ℕ) (C1 : total = 36) 
    (C2 : ∀ (S : Finset ℕ), S.card = 33 → ∃ g b : ℕ, g + b = 33 ∧ g > b) 
    (C3 : ∃ (S : Finset ℕ), S.card = 31 ∧ ∃ g b : ℕ, g + b = 31 ∧ b > g) : 
    ∃ G : ℕ, G = 20 :=
by
  sorry

end number_of_girls_in_club_l1369_136967


namespace sequence_solution_l1369_136985

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → a (n + 1) = a n * ((n + 2) / n)

theorem sequence_solution (a : ℕ → ℝ) (h1 : seq a) (h2 : a 1 = 1) :
  ∀ n : ℕ, 0 < n → a n = (n * (n + 1)) / 2 :=
by
  sorry

end sequence_solution_l1369_136985


namespace tripod_max_height_l1369_136927

noncomputable def tripod_new_height (original_height : ℝ) (original_leg_length : ℝ) (broken_leg_length : ℝ) : ℝ :=
  (broken_leg_length / original_leg_length) * original_height

theorem tripod_max_height :
  let original_height := 5
  let original_leg_length := 6
  let broken_leg_length := 4
  let h := tripod_new_height original_height original_leg_length broken_leg_length
  h = (10 / 3) :=
by
  sorry

end tripod_max_height_l1369_136927


namespace Sarah_pool_depth_l1369_136946

theorem Sarah_pool_depth (S J : ℝ) (h1 : J = 2 * S + 5) (h2 : J = 15) : S = 5 := by
  sorry

end Sarah_pool_depth_l1369_136946


namespace evaluate_g_f_l1369_136917

def f (a b : ℤ) : ℤ × ℤ := (-a, b)

def g (m n : ℤ) : ℤ × ℤ := (m, -n)

theorem evaluate_g_f : g (f 2 (-3)).1 (f 2 (-3)).2 = (-2, 3) := by
  sorry

end evaluate_g_f_l1369_136917


namespace common_ratio_q_l1369_136909

variable {α : Type*} [LinearOrderedField α]

def geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => geom_seq a q n * q

def sum_geom_seq (a q : α) : ℕ → α
| 0 => a
| n+1 => sum_geom_seq a q n + geom_seq a q (n + 1)

theorem common_ratio_q (a q : α) (hq : 0 < q) (h_inc : ∀ n, geom_seq a q n < geom_seq a q (n + 1))
  (h1 : geom_seq a q 1 = 2)
  (h2 : sum_geom_seq a q 2 = 7) :
  q = 2 :=
sorry

end common_ratio_q_l1369_136909


namespace unique_line_equation_l1369_136961

theorem unique_line_equation
  (k : ℝ)
  (m b : ℝ)
  (h1 : |(k^2 + 4*k + 3) - (m*k + b)| = 4)
  (h2 : 2*m + b = 8)
  (h3 : b ≠ 0) :
  (m = 6 ∧ b = -4) :=
by
  sorry

end unique_line_equation_l1369_136961


namespace min_value_1abc_l1369_136986

theorem min_value_1abc (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : c = 0) 
    (h₄ : (1000 + 100 * a + 10 * b + c) % 2 = 0) 
    (h₅ : (1000 + 100 * a + 10 * b + c) % 3 = 0) 
    (h₆ : (1000 + 100 * a + 10 * b + c) % 5 = 0)
  : 1000 + 100 * a + 10 * b + c = 1020 :=
by
  sorry

end min_value_1abc_l1369_136986


namespace fraction_of_students_with_partner_l1369_136966

theorem fraction_of_students_with_partner (s t : ℕ) 
  (h : t = (4 * s) / 3) :
  (t / 4 + s / 3) / (t + s) = 2 / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_students_with_partner_l1369_136966


namespace minimum_sum_at_nine_l1369_136982

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem minimum_sum_at_nine {a1 d : ℤ} (h_a1_neg : a1 < 0) 
    (h_sum_equal : sum_of_arithmetic_sequence a1 d 12 = sum_of_arithmetic_sequence a1 d 6) :
  ∀ n : ℕ, (n = 9) → sum_of_arithmetic_sequence a1 d n ≤ sum_of_arithmetic_sequence a1 d m :=
sorry

end minimum_sum_at_nine_l1369_136982


namespace grandmother_age_l1369_136976

theorem grandmother_age 
  (avg_age : ℝ)
  (age1 age2 age3 grandma_age : ℝ)
  (h_avg_age : avg_age = 20)
  (h_ages : age1 = 5)
  (h_ages2 : age2 = 10)
  (h_ages3 : age3 = 13)
  (h_eq : (age1 + age2 + age3 + grandma_age) / 4 = avg_age) : 
  grandma_age = 52 := 
by
  sorry

end grandmother_age_l1369_136976


namespace cricket_players_count_l1369_136938

theorem cricket_players_count (Hockey Football Softball Total Cricket : ℕ) 
    (hHockey : Hockey = 12)
    (hFootball : Football = 18)
    (hSoftball : Softball = 13)
    (hTotal : Total = 59)
    (hTotalCalculation : Total = Hockey + Football + Softball + Cricket) : 
    Cricket = 16 := by
  sorry

end cricket_players_count_l1369_136938


namespace transformation_composition_l1369_136912

def f (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def g (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 - p.2)

theorem transformation_composition :
  f (g (-1, 2)) = (1, -3) :=
by {
  sorry
}

end transformation_composition_l1369_136912


namespace investment_in_business_l1369_136968

theorem investment_in_business (Q : ℕ) (P : ℕ) 
  (h1 : Q = 65000) 
  (h2 : 4 * Q = 5 * P) : 
  P = 52000 :=
by
  rw [h1] at h2
  linarith

end investment_in_business_l1369_136968


namespace min_ab_l1369_136979

theorem min_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b - a * b + 3 = 0) : 
  9 ≤ a * b :=
sorry

end min_ab_l1369_136979


namespace Mrs_Early_speed_l1369_136918

noncomputable def speed_to_reach_on_time (distance : ℝ) (ideal_time : ℝ) : ℝ := distance / ideal_time

theorem Mrs_Early_speed:
  ∃ (d t : ℝ), 
    (d = 50 * (t + 5/60)) ∧ 
    (d = 80 * (t - 7/60)) ∧ 
    (speed_to_reach_on_time d t = 59) := sorry

end Mrs_Early_speed_l1369_136918


namespace solve_inequalities_l1369_136902

theorem solve_inequalities (x : ℝ) (h₁ : (x - 1) / 2 < 2 * x + 1) (h₂ : -3 * (1 - x) ≥ -4) : x ≥ -1 / 3 :=
by
  sorry

end solve_inequalities_l1369_136902


namespace range_of_m_l1369_136931

noncomputable def f (x : ℝ) : ℝ := 1 + Real.sin (2 * x)

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + m

theorem range_of_m (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x₀ ≥ g x₀ m) → m ≤ Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_m_l1369_136931


namespace units_digit_35_pow_7_plus_93_pow_45_l1369_136947

-- Definitions of units digit calculations for the specific values
def units_digit (n : ℕ) : ℕ := n % 10

def units_digit_35_pow_7 : ℕ := units_digit (35 ^ 7)
def units_digit_93_pow_45 : ℕ := units_digit (93 ^ 45)

-- Statement to prove that the sum of the units digits is 8
theorem units_digit_35_pow_7_plus_93_pow_45 : 
  units_digit (35 ^ 7) + units_digit (93 ^ 45) = 8 :=
by 
  sorry -- proof omitted

end units_digit_35_pow_7_plus_93_pow_45_l1369_136947


namespace number_of_men_l1369_136914

theorem number_of_men (M W : ℕ) (h1 : W = 2) (h2 : ∃k, k = 4) : M = 4 :=
by
  sorry

end number_of_men_l1369_136914


namespace variance_transformation_l1369_136924

theorem variance_transformation (a_1 a_2 a_3 : ℝ) (h : (1 / 3) * ((a_1 - ((a_1 + a_2 + a_3) / 3))^2 + (a_2 - ((a_1 + a_2 + a_3) / 3))^2 + (a_3 - ((a_1 + a_2 + a_3) / 3))^2) = 1) :
  (1 / 3) * ((3 * a_1 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_2 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2 + (3 * a_3 + 2 - (3 * (a_1 + a_2 + a_3) / 3 + 2))^2) = 9 := by 
  sorry

end variance_transformation_l1369_136924


namespace interval_of_monotonic_increase_l1369_136910

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def y' (x : ℝ) : ℝ := 2 * x * Real.exp x + x^2 * Real.exp x

theorem interval_of_monotonic_increase :
  ∀ x : ℝ, (y' x ≥ 0 ↔ (x ∈ Set.Ici 0 ∨ x ∈ Set.Iic (-2))) :=
by
  sorry

end interval_of_monotonic_increase_l1369_136910


namespace smallest_n_divisible_by_100_million_l1369_136919

noncomputable def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

noncomputable def nth_term (a1 r : ℚ) (n : ℕ) : ℚ := a1 * r^(n - 1)

theorem smallest_n_divisible_by_100_million :
  ∀ (a1 a2 : ℚ), a1 = 5/6 → a2 = 25 → 
  ∃ n : ℕ, nth_term a1 (common_ratio a1 a2) n % 100000000 = 0 ∧ n = 9 :=
by
  intros a1 a2 h1 h2
  have r := common_ratio a1 a2
  have a9 := nth_term a1 r 9
  sorry

end smallest_n_divisible_by_100_million_l1369_136919


namespace option_C_correct_l1369_136980

theorem option_C_correct : (Real.sqrt 2) * (Real.sqrt 6) = 2 * (Real.sqrt 3) :=
by sorry

end option_C_correct_l1369_136980


namespace determine_xyz_l1369_136929

-- Define the conditions for the variables x, y, and z
variables (x y z : ℝ)

-- State the problem as a theorem
theorem determine_xyz :
  (x + y + z) * (x * y + x * z + y * z) = 24 ∧
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 8 →
  x * y * z = 16 / 3 :=
by
  intros h
  sorry

end determine_xyz_l1369_136929


namespace distance_MC_l1369_136959

theorem distance_MC (MA MB MC : ℝ) (hMA : MA = 2) (hMB : MB = 3) (hABC : ∀ x y z : ℝ, x + y > z ∧ y + z > x ∧ z + x > y) :
  1 ≤ MC ∧ MC ≤ 5 := 
by 
  sorry

end distance_MC_l1369_136959


namespace problem_solution_l1369_136916

noncomputable def arithmetic_sequences
    (a : ℕ → ℚ) (b : ℕ → ℚ)
    (Sn : ℕ → ℚ) (Tn : ℕ → ℚ) : Prop :=
  (∀ n : ℕ, Sn n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) ∧
  (∀ n : ℕ, Tn n = n / 2 * (2 * b 1 + (n - 1) * (b 2 - b 1))) ∧
  (∀ n : ℕ, Sn n / Tn n = (2 * n - 3) / (4 * n - 3))

theorem problem_solution
    (a : ℕ → ℚ) (b : ℕ → ℚ) (Sn : ℕ → ℚ) (Tn : ℕ → ℚ)
    (h_arith : arithmetic_sequences a b Sn Tn) :
    (a 9 / (b 5 + b 7)) + (a 3 / (b 8 + b 4)) = 19 / 41 :=
by
  sorry

end problem_solution_l1369_136916


namespace second_friend_shells_l1369_136937

theorem second_friend_shells (initial_shells : ℕ) (first_friend_shells : ℕ) (total_shells : ℕ) (second_friend_shells : ℕ) :
  initial_shells = 5 → first_friend_shells = 15 → total_shells = 37 → initial_shells + first_friend_shells + second_friend_shells = total_shells → second_friend_shells = 17 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end second_friend_shells_l1369_136937


namespace range_of_dot_product_l1369_136955

theorem range_of_dot_product
  (x y : ℝ)
  (on_ellipse : x^2 / 2 + y^2 = 1) :
  ∃ m n : ℝ, (m = 0) ∧ (n = 1) ∧ m ≤ x^2 / 2 ∧ x^2 / 2 ≤ n :=
sorry

end range_of_dot_product_l1369_136955


namespace scientific_notation_of_508_billion_yuan_l1369_136952

-- Definition for a billion in the international system.
def billion : ℝ := 10^9

-- The amount of money given in the problem.
def amount_in_billion (n : ℝ) : ℝ := n * billion

-- The Lean theorem statement to prove.
theorem scientific_notation_of_508_billion_yuan :
  amount_in_billion 508 = 5.08 * 10^11 :=
by
  sorry

end scientific_notation_of_508_billion_yuan_l1369_136952


namespace problem1_problem2_l1369_136949

-- Define the function f(x)
def f (m x : ℝ) : ℝ := |m * x + 1| + |2 * x - 3|

-- Problem 1: Prove the range of x for f(x) = 4 when m = 2
theorem problem1 (x : ℝ) : f 2 x = 4 ↔ -1 / 2 ≤ x ∧ x ≤ 3 / 2 :=
by
  sorry

-- Problem 2: Prove the range of m given f(1) ≤ (2a^2 + 8) / a for any positive a
theorem problem2 (m : ℝ) (h : ∀ a : ℝ, a > 0 → f m 1 ≤ (2 * a^2 + 8) / a) : -8 ≤ m ∧ m ≤ 6 :=
by
  sorry

end problem1_problem2_l1369_136949


namespace average_weight_bc_is_43_l1369_136993

variable (a b c : ℝ)

-- Definitions of the conditions
def average_weight_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_weight_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def weight_b (b : ℝ) : Prop := b = 31

-- The theorem to prove
theorem average_weight_bc_is_43 :
  ∀ (a b c : ℝ), average_weight_abc a b c → average_weight_ab a b → weight_b b → (b + c) / 2 = 43 :=
by
  intros a b c h_average_weight_abc h_average_weight_ab h_weight_b
  sorry

end average_weight_bc_is_43_l1369_136993


namespace parallelogram_slope_l1369_136900

theorem parallelogram_slope (a b c d : ℚ) :
    a = 35 + c ∧ b = 125 - c ∧ 875 - 25 * c = 280 + 8 * c ∧ (a, 8) = (b, 25)
    → ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ (∃ h : 8 * 33 * a + 595 = 2350, (m, n) = (25, 4)) :=
by
  sorry

end parallelogram_slope_l1369_136900


namespace find_sum_of_squares_l1369_136972

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- Given conditions
def condition1 : Prop := x + y = 12
def condition2 : Prop := x * y = 50

-- The statement we need to prove
theorem find_sum_of_squares (h1 : condition1 x y) (h2 : condition2 x y) : x^2 + y^2 = 44 := by
  sorry

end find_sum_of_squares_l1369_136972


namespace intersection_of_sets_l1369_136999

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def set_B : Set ℝ := {x | (x + 1) * (x - 4) > 0}

theorem intersection_of_sets :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ {x | (x + 1) * (x - 4) > 0} = {x | -2 ≤ x ∧ x < -1} :=
by
  sorry

end intersection_of_sets_l1369_136999
