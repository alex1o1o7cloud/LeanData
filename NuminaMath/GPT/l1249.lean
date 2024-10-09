import Mathlib

namespace carrie_total_spend_l1249_124913

def cost_per_tshirt : ℝ := 9.15
def number_of_tshirts : ℝ := 22

theorem carrie_total_spend : (cost_per_tshirt * number_of_tshirts) = 201.30 := by 
  sorry

end carrie_total_spend_l1249_124913


namespace coin_stack_height_l1249_124952

def alpha_thickness : ℝ := 1.25
def beta_thickness : ℝ := 2.00
def gamma_thickness : ℝ := 0.90
def delta_thickness : ℝ := 1.60
def stack_height : ℝ := 18.00

theorem coin_stack_height :
  (∃ n : ℕ, stack_height = n * beta_thickness) ∨ (∃ n : ℕ, stack_height = n * gamma_thickness) :=
sorry

end coin_stack_height_l1249_124952


namespace no_values_of_b_l1249_124908

def f (b x : ℝ) := x^2 + b * x - 1

theorem no_values_of_b : ∀ b : ℝ, ∃ x : ℝ, f b x = 3 :=
by
  intro b
  use 0  -- example, needs actual computation
  sorry

end no_values_of_b_l1249_124908


namespace range_of_m_l1249_124936

theorem range_of_m (m : ℝ) :
  (1 - 2 * m > 0) ∧ (m + 1 > 0) → -1 < m ∧ m < 1/2 :=
by
  sorry

end range_of_m_l1249_124936


namespace bars_cannot_form_triangle_l1249_124904

theorem bars_cannot_form_triangle 
  (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 10) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) :=
by 
  rw [h1, h2, h3]
  sorry

end bars_cannot_form_triangle_l1249_124904


namespace factorize_xcube_minus_x_l1249_124983

theorem factorize_xcube_minus_x (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by 
  sorry

end factorize_xcube_minus_x_l1249_124983


namespace matt_total_points_l1249_124912

variable (n2_successful_shots : Nat) (n3_successful_shots : Nat)

def total_points (n2 : Nat) (n3 : Nat) : Nat :=
  2 * n2 + 3 * n3

theorem matt_total_points :
  total_points 4 2 = 14 :=
by
  sorry

end matt_total_points_l1249_124912


namespace calculate_lives_lost_l1249_124958

-- Define the initial number of lives
def initial_lives : ℕ := 98

-- Define the remaining number of lives
def remaining_lives : ℕ := 73

-- Define the number of lives lost
def lives_lost : ℕ := initial_lives - remaining_lives

-- Prove that Kaleb lost 25 lives
theorem calculate_lives_lost : lives_lost = 25 := 
by {
  -- The proof would go here, but we'll skip it
  sorry
}

end calculate_lives_lost_l1249_124958


namespace total_charge_for_2_hours_l1249_124974

theorem total_charge_for_2_hours (F A : ℕ) 
  (h1 : F = A + 40) 
  (h2 : F + 4 * A = 375) : 
  F + A = 174 :=
by 
  sorry

end total_charge_for_2_hours_l1249_124974


namespace certain_fraction_ratio_l1249_124990

theorem certain_fraction_ratio :
  ∃ x : ℚ,
    (2 / 5 : ℚ) / x = (0.46666666666666673 : ℚ) / (1 / 2) ∧ x = 3 / 7 :=
by sorry

end certain_fraction_ratio_l1249_124990


namespace distance_from_P_to_y_axis_l1249_124948

theorem distance_from_P_to_y_axis (P : ℝ × ℝ) :
  (P.2 ^ 2 = -12 * P.1) → (dist P (-3, 0) = 9) → abs P.1 = 6 :=
by
  sorry

end distance_from_P_to_y_axis_l1249_124948


namespace martha_savings_l1249_124989

-- Definitions based on conditions
def weekly_latte_spending : ℝ := 4.00 * 5
def weekly_iced_coffee_spending : ℝ := 2.00 * 3
def total_weekly_coffee_spending : ℝ := weekly_latte_spending + weekly_iced_coffee_spending
def annual_coffee_spending : ℝ := total_weekly_coffee_spending * 52
def savings_percentage : ℝ := 0.25

-- The theorem to be proven
theorem martha_savings : annual_coffee_spending * savings_percentage = 338.00 := by
  sorry

end martha_savings_l1249_124989


namespace book_pages_l1249_124929

theorem book_pages (P : ℝ) (h1 : P / 2 + 0.15 * (P / 2) + 210 = P) : P = 600 := 
sorry

end book_pages_l1249_124929


namespace ConfuciusBirthYear_l1249_124938

-- Definitions based on the conditions provided
def birthYearAD (year : Int) : Int := year

def birthYearBC (year : Int) : Int := -year

theorem ConfuciusBirthYear :
  birthYearBC 551 = -551 :=
by
  sorry

end ConfuciusBirthYear_l1249_124938


namespace solve_congruence_l1249_124997

theorem solve_congruence (n : ℕ) (h₀ : 0 ≤ n ∧ n < 47) (h₁ : 13 * n ≡ 5 [MOD 47]) :
  n = 4 :=
sorry

end solve_congruence_l1249_124997


namespace stickers_per_student_l1249_124926

theorem stickers_per_student (G S B N: ℕ) (hG: G = 50) (hS: S = 2 * G) (hB: B = S - 20) (hN: N = 5) : 
  (G + S + B) / N = 46 := by
  sorry

end stickers_per_student_l1249_124926


namespace minimum_glue_drops_to_prevent_37_gram_subset_l1249_124943

def stones : List ℕ := List.range' 1 36  -- List of stones with masses from 1 to 36 grams

def glue_drop_combination_invalid (stones : List ℕ) : Prop :=
  ¬ (∃ (subset : List ℕ), subset.sum = 37 ∧ (∀ s ∈ subset, s ∈ stones))

def min_glue_drops (stones : List ℕ) : ℕ := 
  9 -- as per the solution

theorem minimum_glue_drops_to_prevent_37_gram_subset :
  ∀ (s : List ℕ), s = stones → glue_drop_combination_invalid s → min_glue_drops s = 9 :=
by intros; sorry

end minimum_glue_drops_to_prevent_37_gram_subset_l1249_124943


namespace probability_two_heads_one_tail_in_three_tosses_l1249_124957

theorem probability_two_heads_one_tail_in_three_tosses
(P : ℕ → Prop) (pr : ℤ) : 
  (∀ n, P n → pr = 1 / 2) -> 
  P 3 → pr = 3 / 8 :=
by
  sorry

end probability_two_heads_one_tail_in_three_tosses_l1249_124957


namespace disjoint_subsets_exist_l1249_124969

theorem disjoint_subsets_exist (n : ℕ) (h : 0 < n) 
  (A : Fin (n + 1) → Set (Fin n)) (hA : ∀ i : Fin (n + 1), A i ≠ ∅) :
  ∃ (I J : Finset (Fin (n + 1))), I ≠ ∅ ∧ J ≠ ∅ ∧ Disjoint I J ∧ 
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) :=
sorry

end disjoint_subsets_exist_l1249_124969


namespace negative_movement_south_l1249_124979

noncomputable def movement_interpretation (x : ℤ) : String :=
if x > 0 then 
  "moving " ++ toString x ++ "m north"
else 
  "moving " ++ toString (-x) ++ "m south"

theorem negative_movement_south : movement_interpretation (-50) = "moving 50m south" := 
by 
  sorry

end negative_movement_south_l1249_124979


namespace radius_of_tangent_circle_l1249_124935

theorem radius_of_tangent_circle (side_length : ℝ) (num_semicircles : ℕ)
  (r_s : ℝ) (r : ℝ)
  (h1 : side_length = 4)
  (h2 : num_semicircles = 16)
  (h3 : r_s = side_length / 4 / 2)
  (h4 : r = (9 : ℝ) / (2 * Real.sqrt 5)) :
  r = (9 * Real.sqrt 5) / 10 :=
by
  rw [h4]
  sorry

end radius_of_tangent_circle_l1249_124935


namespace hexagon_circle_radius_l1249_124920

noncomputable def hexagon_radius (sides : List ℝ) (probability : ℝ) : ℝ :=
  let total_angle := 360.0
  let visible_angle := probability * total_angle
  let side_length_average := (sides.sum / sides.length : ℝ)
  let theta := (visible_angle / 6 : ℝ) -- assuming θ approximately splits equally among 6 gaps
  side_length_average / Real.sin (theta / 2 * Real.pi / 180.0)

theorem hexagon_circle_radius :
  hexagon_radius [3, 2, 4, 3, 2, 4] (1 / 3) = 17.28 :=
by
  sorry

end hexagon_circle_radius_l1249_124920


namespace estimate_sqrt_diff_l1249_124922

-- Defining approximate values for square roots
def approx_sqrt_90 : ℝ := 9.5
def approx_sqrt_88 : ℝ := 9.4

-- Main statement
theorem estimate_sqrt_diff : |(approx_sqrt_90 - approx_sqrt_88) - 0.10| < 0.01 := by
  sorry

end estimate_sqrt_diff_l1249_124922


namespace maria_average_speed_l1249_124921

theorem maria_average_speed:
  let distance1 := 180
  let time1 := 4.5
  let distance2 := 270
  let time2 := 5.25
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time = 46.15 := by
  -- Sorry to skip the proof
  sorry

end maria_average_speed_l1249_124921


namespace greatest_t_solution_l1249_124981

theorem greatest_t_solution :
  ∀ t : ℝ, t ≠ 8 ∧ t ≠ -5 →
  (t^2 - t - 40) / (t - 8) = 5 / (t + 5) →
  t ≤ -2 :=
by
  sorry

end greatest_t_solution_l1249_124981


namespace cows_in_herd_l1249_124917

theorem cows_in_herd (n : ℕ) (h1 : n / 3 + n / 6 + n / 7 < n) (h2 : 15 = n * 5 / 14) : n = 42 :=
sorry

end cows_in_herd_l1249_124917


namespace final_milk_concentration_l1249_124907

theorem final_milk_concentration
  (initial_mixture_volume : ℝ)
  (initial_milk_volume : ℝ)
  (replacement_volume : ℝ)
  (replacements_count : ℕ)
  (final_milk_volume : ℝ) :
  initial_mixture_volume = 100 → 
  initial_milk_volume = 36 → 
  replacement_volume = 50 →
  replacements_count = 2 →
  final_milk_volume = 9 →
  (final_milk_volume / initial_mixture_volume * 100) = 9 :=
by
  sorry

end final_milk_concentration_l1249_124907


namespace will_remaining_balance_l1249_124996

theorem will_remaining_balance :
  ∀ (initial_money conversion_fee : ℝ) 
    (exchange_rate : ℝ)
    (sweater_cost tshirt_cost shoes_cost hat_cost socks_cost : ℝ)
    (shoes_refund_percentage : ℝ)
    (discount_percentage sales_tax_percentage : ℝ),
  initial_money = 74 →
  conversion_fee = 2 →
  exchange_rate = 1.5 →
  sweater_cost = 13.5 →
  tshirt_cost = 16.5 →
  shoes_cost = 45 →
  hat_cost = 7.5 →
  socks_cost = 6 →
  shoes_refund_percentage = 0.85 →
  discount_percentage = 0.10 →
  sales_tax_percentage = 0.05 →
  (initial_money - conversion_fee) * exchange_rate -
  ((sweater_cost + tshirt_cost + shoes_cost + hat_cost + socks_cost - shoes_cost * shoes_refund_percentage) *
   (1 - discount_percentage) * (1 + sales_tax_percentage)) /
  exchange_rate = 39.87 :=
by
  intros initial_money conversion_fee exchange_rate
        sweater_cost tshirt_cost shoes_cost hat_cost socks_cost
        shoes_refund_percentage discount_percentage sales_tax_percentage
        h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end will_remaining_balance_l1249_124996


namespace proof_by_contradiction_example_l1249_124911

theorem proof_by_contradiction_example (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end proof_by_contradiction_example_l1249_124911


namespace max_sum_first_n_terms_formula_sum_terms_abs_l1249_124937

theorem max_sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  ∃ (n : ℕ), n = 15 ∧ S 15 = 225 := by
  sorry

theorem formula_sum_terms_abs (a : ℕ → ℤ) (S T : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  (∀ n, n ≤ 15 → T n = 30 * n - n * n) ∧
  (∀ n, n ≥ 16 → T n = n * n - 30 * n + 450) := by
  sorry

end max_sum_first_n_terms_formula_sum_terms_abs_l1249_124937


namespace range_of_a_l1249_124966

open Real

/-- Proposition p: x^2 + 2*a*x + 4 > 0 for all x in ℝ -/
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: the exponential function (3 - 2*a)^x is increasing -/
def q (a : ℝ) : Prop :=
  3 - 2*a > 1

/-- Given p ∧ q, prove that -2 < a < 1 -/
theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : -2 < a ∧ a < 1 :=
sorry

end range_of_a_l1249_124966


namespace equation_of_motion_l1249_124918

section MotionLaw

variable (t s : ℝ)
variable (v : ℝ → ℝ)
variable (C : ℝ)

-- Velocity function
def velocity (t : ℝ) : ℝ := 6 * t^2 + 1

-- Displacement function (indefinite integral of velocity)
def displacement (t : ℝ) (C : ℝ) : ℝ := 2 * t^3 + t + C

-- Given condition: displacement at t = 3 is 60
axiom displacement_at_3 : displacement 3 C = 60

-- Prove that the equation of motion is s = 2t^3 + t + 3
theorem equation_of_motion :
  ∃ C, displacement t C = 2 * t^3 + t + 3 :=
by
  use 3
  sorry

end MotionLaw

end equation_of_motion_l1249_124918


namespace tan_neg4095_eq_one_l1249_124977

theorem tan_neg4095_eq_one : Real.tan (Real.pi / 180 * -4095) = 1 := by
  sorry

end tan_neg4095_eq_one_l1249_124977


namespace largest_is_D_l1249_124945

-- Definitions based on conditions
def A : ℕ := 27
def B : ℕ := A + 7
def C : ℕ := B - 9
def D : ℕ := 2 * C

-- Theorem stating D is the largest
theorem largest_is_D : D = max (max A B) (max C D) :=
by
  -- Inserting sorry because the proof is not required.
  sorry

end largest_is_D_l1249_124945


namespace lead_amount_in_mixture_l1249_124987

theorem lead_amount_in_mixture 
  (W : ℝ) 
  (h_copper : 0.60 * W = 12) 
  (h_mixture_composition : (0.15 * W = 0.15 * W) ∧ (0.25 * W = 0.25 * W) ∧ (0.60 * W = 0.60 * W)) :
  (0.25 * W = 5) :=
by
  sorry

end lead_amount_in_mixture_l1249_124987


namespace find_geometric_progression_l1249_124950

theorem find_geometric_progression (a b c : ℚ)
  (h1 : a * c = b * b)
  (h2 : a + c = 2 * (b + 8))
  (h3 : a * (c + 64) = (b + 8) * (b + 8)) :
  (a = 4/9 ∧ b = -20/9 ∧ c = 100/9) ∨ (a = 4 ∧ b = 12 ∧ c = 36) :=
sorry

end find_geometric_progression_l1249_124950


namespace find_k_l1249_124960

-- Defining the vectors
def a (k : ℝ) : ℝ × ℝ := (k, -2)
def b : ℝ × ℝ := (2, 2)

-- Condition 1: a + b is not the zero vector
def non_zero_sum (k : ℝ) := (a k).1 + b.1 ≠ 0 ∨ (a k).2 + b.2 ≠ 0

-- Condition 2: a is perpendicular to a + b
def perpendicular (k : ℝ) := (a k).1 * ((a k).1 + b.1) + (a k).2 * ((a k).2 + b.2) = 0

-- The theorem to prove
theorem find_k (k : ℝ) (cond1 : non_zero_sum k) (cond2 : perpendicular k) : k = 0 := 
sorry

end find_k_l1249_124960


namespace sum_equals_one_l1249_124963

noncomputable def sum_proof (x y z : ℝ) (h : x * y * z = 1) : ℝ :=
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x))

theorem sum_equals_one (x y z : ℝ) (h : x * y * z = 1) : 
  sum_proof x y z h = 1 := sorry

end sum_equals_one_l1249_124963


namespace train_speed_l1249_124971

theorem train_speed (v : ℝ) (h1 : 60 * 6.5 + v * 6.5 = 910) : v = 80 := 
sorry

end train_speed_l1249_124971


namespace ratio_third_to_first_second_l1249_124970

-- Define the times spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_total : ℕ := 90
def time_third_step : ℕ := time_total - (time_first_step + time_second_step)

-- Define the combined time for the first two steps
def time_combined_first_second : ℕ := time_first_step + time_second_step

-- The goal is to prove that the ratio of the time spent on the third step to the combined time spent on the first and second steps is 1:1
theorem ratio_third_to_first_second : time_third_step = time_combined_first_second :=
by
  -- Proof goes here
  sorry

end ratio_third_to_first_second_l1249_124970


namespace cost_per_book_l1249_124934

-- Definitions and conditions
def number_of_books : ℕ := 8
def amount_tommy_has : ℕ := 13
def amount_tommy_needs_to_save : ℕ := 27

-- Total money Tommy needs to buy the books
def total_amount_needed : ℕ := amount_tommy_has + amount_tommy_needs_to_save

-- Proven statement
theorem cost_per_book : (total_amount_needed / number_of_books) = 5 := by
  -- Skip proof
  sorry

end cost_per_book_l1249_124934


namespace total_people_count_l1249_124988

-- Definitions based on given conditions
def Cannoneers : ℕ := 63
def Women : ℕ := 2 * Cannoneers
def Men : ℕ := 2 * Women
def TotalPeople : ℕ := Women + Men

-- Lean statement to prove
theorem total_people_count : TotalPeople = 378 := by
  -- placeholders for proof steps
  sorry

end total_people_count_l1249_124988


namespace gcd_of_90_and_405_l1249_124967

def gcd_90_405 : ℕ := Nat.gcd 90 405

theorem gcd_of_90_and_405 : gcd_90_405 = 45 :=
by
  -- proof goes here
  sorry

end gcd_of_90_and_405_l1249_124967


namespace john_weekly_earnings_before_raise_l1249_124903

theorem john_weekly_earnings_before_raise :
  ∀(x : ℝ), (70 = 1.0769 * x) → x = 64.99 :=
by
  intros x h
  sorry

end john_weekly_earnings_before_raise_l1249_124903


namespace prove_function_domain_l1249_124984

def function_domain := {x : ℝ | (x + 4 ≥ 0 ∧ x ≠ 0)}

theorem prove_function_domain :
  function_domain = {x : ℝ | x ∈ (Set.Icc (-4:ℝ) 0).diff ({0}:Set ℝ) ∪ (Set.Ioi 0)} :=
by
  sorry

end prove_function_domain_l1249_124984


namespace find_y_is_90_l1249_124959

-- Definitions for given conditions
def angle_ABC : ℝ := 120
def angle_ABD : ℝ := 180 - angle_ABC
def angle_BDA : ℝ := 30

-- The theorem to prove y = 90 degrees
theorem find_y_is_90 :
  ∃ y : ℝ, angle_ABD = 60 ∧ angle_BDA = 30 ∧ (30 + 60 + y = 180) → y = 90 :=
by
  sorry

end find_y_is_90_l1249_124959


namespace age_proof_l1249_124999

-- Let's define the conditions first
variable (s f : ℕ) -- s: age of the son, f: age of the father

-- Conditions derived from the problem statement
def son_age_condition : Prop := s = 8 - 1
def father_age_condition : Prop := f = 5 * s

-- The goal is to prove that the father's age is 35
theorem age_proof (s f : ℕ) (h₁ : son_age_condition s) (h₂ : father_age_condition s f) : f = 35 :=
by sorry

end age_proof_l1249_124999


namespace alligators_hiding_correct_l1249_124992

def total_alligators := 75
def not_hiding_alligators := 56

def hiding_alligators (total not_hiding : Nat) : Nat :=
  total - not_hiding

theorem alligators_hiding_correct : hiding_alligators total_alligators not_hiding_alligators = 19 := 
by
  sorry

end alligators_hiding_correct_l1249_124992


namespace instantaneous_rate_of_change_at_0_l1249_124976

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp (Real.sin x)

theorem instantaneous_rate_of_change_at_0 : (deriv f 0) = 2 :=
  by
  sorry

end instantaneous_rate_of_change_at_0_l1249_124976


namespace pencil_eraser_cost_l1249_124978

variable (p e : ℕ)

theorem pencil_eraser_cost
  (h1 : 15 * p + 5 * e = 125)
  (h2 : p > e)
  (h3 : p > 0)
  (h4 : e > 0) :
  p + e = 11 :=
sorry

end pencil_eraser_cost_l1249_124978


namespace cos_2alpha_minus_pi_over_6_l1249_124962

theorem cos_2alpha_minus_pi_over_6 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hSin : Real.sin (α + π / 6) = 3 / 5) :
  Real.cos (2 * α - π / 6) = 24 / 25 :=
sorry

end cos_2alpha_minus_pi_over_6_l1249_124962


namespace basketball_player_possible_scores_l1249_124941

-- Define the conditions
def isValidBasketCount (n : Nat) : Prop := n = 7
def isValidBasketValue (v : Nat) : Prop := v = 1 ∨ v = 2 ∨ v = 3

-- Define the theorem statement
theorem basketball_player_possible_scores :
  ∃ (s : Finset ℕ), s = {n | ∃ n1 n2 n3 : Nat, 
                                n1 + n2 + n3 = 7 ∧ 
                                n = 1 * n1 + 2 * n2 + 3 * n3 ∧ 
                                n1 + n2 + n3 = 7 ∧ 
                                n >= 7 ∧ n <= 21} ∧
                                s.card = 15 :=
by
  sorry

end basketball_player_possible_scores_l1249_124941


namespace maximum_value_at_vertex_l1249_124961

-- Defining the parabola as a function
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Defining the vertex condition
def vertex_condition (a b c : ℝ) := ∀ x : ℝ, parabola a b c x = a * x^2 + b * x + c

-- Defining the condition that the parabola opens downward
def opens_downward (a : ℝ) := a < 0

-- Defining the vertex coordinates condition
def vertex_coordinates (a b c : ℝ) := 
  ∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = -3 ∧ parabola a b c x₀ = y₀

-- The main theorem statement
theorem maximum_value_at_vertex (a b c : ℝ) (h1 : opens_downward a) (h2 : vertex_coordinates a b c) : ∃ y₀, y₀ = -3 ∧ ∀ x : ℝ, parabola a b c x ≤ y₀ :=
by
  sorry

end maximum_value_at_vertex_l1249_124961


namespace sum_of_coordinates_l1249_124985

-- Definitions of points and their coordinates
def pointC (x : ℝ) : ℝ × ℝ := (x, 8)
def pointD (x : ℝ) : ℝ × ℝ := (x, -8)

-- The goal is to prove that the sum of the four coordinate values of points C and D is 2x
theorem sum_of_coordinates (x : ℝ) :
  (pointC x).1 + (pointC x).2 + (pointD x).1 + (pointD x).2 = 2 * x :=
by
  sorry

end sum_of_coordinates_l1249_124985


namespace triangle_area_inequality_l1249_124915

variables {a b c S x y z T : ℝ}

-- Definitions based on the given conditions
def side_lengths_of_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def area_of_triangle (a b c S : ℝ) : Prop :=
  16 * S * S = (a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c)

def new_side_lengths (a b c : ℝ) (x y z : ℝ) : Prop :=
  x = a + b / 2 ∧ y = b + c / 2 ∧ z = c + a / 2

def area_condition (S T : ℝ) : Prop :=
  T ≥ 9 / 4 * S

-- Main theorem statement
theorem triangle_area_inequality
  (h_triangle: side_lengths_of_triangle a b c)
  (h_area: area_of_triangle a b c S)
  (h_new_sides: new_side_lengths a b c x y z) :
  ∃ T : ℝ, side_lengths_of_triangle x y z ∧ area_condition S T :=
sorry

end triangle_area_inequality_l1249_124915


namespace smallest_n_divides_999_l1249_124909

/-- 
Given \( 1 \leq n < 1000 \), \( n \) divides 999, and \( n+6 \) divides 99,
prove that the smallest possible value of \( n \) is 27.
 -/
theorem smallest_n_divides_999 (n : ℕ) 
  (h1 : 1 ≤ n) 
  (h2 : n < 1000) 
  (h3 : n ∣ 999) 
  (h4 : n + 6 ∣ 99) : 
  n = 27 :=
  sorry

end smallest_n_divides_999_l1249_124909


namespace range_of_a_l1249_124946

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Lean statement asserting the requirement
theorem range_of_a (a : ℝ) (h : A ⊆ B a ∧ A ≠ B a) : 2 < a := by
  sorry

end range_of_a_l1249_124946


namespace product_of_solutions_eq_zero_l1249_124956

theorem product_of_solutions_eq_zero : 
  (∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4)) → 
  ∃ (x1 x2 : ℝ), (x1 = 0 ∨ x1 = 5) ∧ (x2 = 0 ∨ x2 = 5) ∧ x1 * x2 = 0 :=
by
  sorry

end product_of_solutions_eq_zero_l1249_124956


namespace ratio_x_to_w_as_percentage_l1249_124910

theorem ratio_x_to_w_as_percentage (x y z w : ℝ) 
    (h1 : x = 1.20 * y) 
    (h2 : y = 0.30 * z) 
    (h3 : z = 1.35 * w) : 
    (x / w) * 100 = 48.6 := 
by sorry

end ratio_x_to_w_as_percentage_l1249_124910


namespace brother_age_in_5_years_l1249_124947

theorem brother_age_in_5_years
  (nick_age : ℕ)
  (sister_age : ℕ)
  (brother_age : ℕ)
  (h_nick : nick_age = 13)
  (h_sister : sister_age = nick_age + 6)
  (h_brother : brother_age = (nick_age + sister_age) / 2) :
  brother_age + 5 = 21 := 
by 
  sorry

end brother_age_in_5_years_l1249_124947


namespace log_ab_a2_plus_log_ab_b2_eq_2_l1249_124902

theorem log_ab_a2_plus_log_ab_b2_eq_2 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h_distinct : a ≠ b) (h_a_gt_2 : a > 2) (h_b_gt_2 : b > 2) :
  Real.log (a^2) / Real.log (a * b) + Real.log (b^2) / Real.log (a * b) = 2 :=
by
  sorry

end log_ab_a2_plus_log_ab_b2_eq_2_l1249_124902


namespace goals_even_more_likely_l1249_124939

theorem goals_even_more_likely (p_1 : ℝ) (q_1 : ℝ) (h1 : p_1 + q_1 = 1) :
  let p := p_1^2 + q_1^2 
  let q := 2 * p_1 * q_1
  p ≥ q := by
    sorry

end goals_even_more_likely_l1249_124939


namespace scramble_language_words_count_l1249_124931

theorem scramble_language_words_count :
  let total_words (n : ℕ) := 25 ^ n
  let words_without_B (n : ℕ) := 24 ^ n
  let words_with_B (n : ℕ) := total_words n - words_without_B n
  words_with_B 1 + words_with_B 2 + words_with_B 3 + words_with_B 4 + words_with_B 5 = 1863701 :=
by
  sorry

end scramble_language_words_count_l1249_124931


namespace chloe_paid_per_dozen_l1249_124905

-- Definitions based on conditions
def half_dozen_sale_price : ℕ := 30
def profit : ℕ := 500
def dozens_sold : ℕ := 50
def full_dozen_sale_price := 2 * half_dozen_sale_price
def total_revenue := dozens_sold * full_dozen_sale_price
def total_cost := total_revenue - profit

-- Proof problem
theorem chloe_paid_per_dozen : (total_cost / dozens_sold) = 50 :=
by
  sorry

end chloe_paid_per_dozen_l1249_124905


namespace arithmetic_progression_rth_term_l1249_124919

open Nat

theorem arithmetic_progression_rth_term (n r : ℕ) (Sn : ℕ → ℕ) 
  (h : ∀ n, Sn n = 5 * n + 4 * n^2) : Sn r - Sn (r - 1) = 8 * r + 1 :=
by
  sorry

end arithmetic_progression_rth_term_l1249_124919


namespace distance_from_Q_to_EF_is_24_div_5_l1249_124991

-- Define the configuration of the square and points
def E := (0, 8)
def F := (8, 8)
def G := (8, 0)
def H := (0, 0)
def N := (4, 0) -- Midpoint of GH
def r1 := 4 -- Radius of the circle centered at N
def r2 := 8 -- Radius of the circle centered at E

-- Definition of the first circle centered at N with radius r1
def circle1 (x y : ℝ) := (x - 4)^2 + y^2 = r1^2

-- Definition of the second circle centered at E with radius r2
def circle2 (x y : ℝ) := x^2 + (y - 8)^2 = r2^2

-- Define the intersection point Q, other than H
def Q := (32 / 5, 16 / 5) -- Found as an intersection point between circle1 and circle2

-- Define the distance from point Q to the line EF
def dist_to_EF := 8 - (Q.2) -- (Q.2 is the y-coordinate of Q)

-- The main statement to prove
theorem distance_from_Q_to_EF_is_24_div_5 : dist_to_EF = 24 / 5 := by
  sorry

end distance_from_Q_to_EF_is_24_div_5_l1249_124991


namespace problem_statement_l1249_124933

theorem problem_statement (n : ℕ) : 2 ^ n ∣ (1 + ⌊(3 + Real.sqrt 5) ^ n⌋) :=
by
  sorry

end problem_statement_l1249_124933


namespace solve_for_x_l1249_124973

theorem solve_for_x (x : ℝ) (h : 4 * x - 5 = 3) : x = 2 :=
by sorry

end solve_for_x_l1249_124973


namespace calculate_gain_percentage_l1249_124954

theorem calculate_gain_percentage (CP SP : ℝ) (h1 : 0.9 * CP = 450) (h2 : SP = 550) : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end calculate_gain_percentage_l1249_124954


namespace remainder_when_dividing_by_2x_minus_4_l1249_124998

def f (x : ℝ) := 4 * x^3 - 9 * x^2 + 12 * x - 14
def g (x : ℝ) := 2 * x - 4

theorem remainder_when_dividing_by_2x_minus_4 : f 2 = 6 := by
  sorry

end remainder_when_dividing_by_2x_minus_4_l1249_124998


namespace stockholm_to_uppsala_distance_l1249_124916

theorem stockholm_to_uppsala_distance :
  let map_distance_cm : ℝ := 45
  let map_scale_cm_to_km : ℝ := 10
  (map_distance_cm * map_scale_cm_to_km = 450) :=
by
  sorry

end stockholm_to_uppsala_distance_l1249_124916


namespace congruence_from_overlap_l1249_124940

-- Definitions used in the conditions
def figure := Type
def equal_area (f1 f2 : figure) : Prop := sorry
def equal_perimeter (f1 f2 : figure) : Prop := sorry
def equilateral_triangle (f : figure) : Prop := sorry
def can_completely_overlap (f1 f2 : figure) : Prop := sorry

-- Theorem that should be proven
theorem congruence_from_overlap (f1 f2 : figure) (h: can_completely_overlap f1 f2) : f1 = f2 := sorry

end congruence_from_overlap_l1249_124940


namespace sin_105_mul_sin_15_eq_one_fourth_l1249_124968

noncomputable def sin_105_deg := Real.sin (105 * Real.pi / 180)
noncomputable def sin_15_deg := Real.sin (15 * Real.pi / 180)

theorem sin_105_mul_sin_15_eq_one_fourth :
  sin_105_deg * sin_15_deg = 1 / 4 :=
by
  sorry

end sin_105_mul_sin_15_eq_one_fourth_l1249_124968


namespace overtime_rate_is_correct_l1249_124928

/-
Define the parameters:
ordinary_rate: Rate per hour for ordinary time in dollars
total_hours: Total hours worked in a week
overtime_hours: Overtime hours worked in a week
total_earnings: Total earnings for the week in dollars
-/

def ordinary_rate : ℝ := 0.60
def total_hours : ℝ := 50
def overtime_hours : ℝ := 8
def total_earnings : ℝ := 32.40

noncomputable def overtime_rate : ℝ :=
(total_earnings - ordinary_rate * (total_hours - overtime_hours)) / overtime_hours

theorem overtime_rate_is_correct :
  overtime_rate = 0.90 :=
by
  sorry

end overtime_rate_is_correct_l1249_124928


namespace distinct_valid_c_values_l1249_124965

theorem distinct_valid_c_values : 
  let is_solution (c : ℤ) (x : ℚ) := (5 * ⌊x⌋₊ + 3 * ⌈x⌉₊ = c) 
  ∃ s : Finset ℤ, (∀ c ∈ s, (∃ x : ℚ, is_solution c x)) ∧ s.card = 500 :=
by sorry

end distinct_valid_c_values_l1249_124965


namespace sufficient_but_not_necessary_condition_l1249_124986

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 4 → a^2 > 16) ∧ (∃ a, (a < -4) ∧ (a^2 > 16)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1249_124986


namespace original_acid_percentage_l1249_124924

variables (a w : ℝ)

-- Conditions from the problem
def cond1 : Prop := a / (a + w + 2) = 0.18
def cond2 : Prop := (a + 2) / (a + w + 4) = 0.36

-- The Lean statement to prove
theorem original_acid_percentage (hc1 : cond1 a w) (hc2 : cond2 a w) : (a / (a + w)) * 100 = 19 :=
sorry

end original_acid_percentage_l1249_124924


namespace dart_prob_center_square_l1249_124953

noncomputable def hexagon_prob (s : ℝ) : ℝ :=
  let square_area := s^2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  square_area / hexagon_area

theorem dart_prob_center_square (s : ℝ) : hexagon_prob s = 2 * Real.sqrt 3 / 9 :=
by
  -- Proof omitted
  sorry

end dart_prob_center_square_l1249_124953


namespace find_d_l1249_124944

theorem find_d {x d : ℤ} (h : (x + (x + 2) + (x + 4) + (x + 7) + (x + d)) / 5 = (x + 4) + 6) : d = 37 :=
sorry

end find_d_l1249_124944


namespace big_al_bananas_l1249_124995

-- Define conditions for the arithmetic sequence and total consumption
theorem big_al_bananas (a : ℕ) : 
  (a + (a + 6) + (a + 12) + (a + 18) + (a + 24) = 100) → 
  (a + 24 = 32) :=
by
  sorry

end big_al_bananas_l1249_124995


namespace positive_difference_is_9107_03_l1249_124932

noncomputable def Cedric_balance : ℝ :=
  15000 * (1 + 0.06) ^ 20

noncomputable def Daniel_balance : ℝ :=
  15000 * (1 + 20 * 0.08)

noncomputable def Elaine_balance : ℝ :=
  15000 * (1 + 0.055 / 2) ^ 40

-- Positive difference between highest and lowest balances.
noncomputable def positive_difference : ℝ :=
  let highest := max Cedric_balance (max Daniel_balance Elaine_balance)
  let lowest := min Cedric_balance (min Daniel_balance Elaine_balance)
  highest - lowest

theorem positive_difference_is_9107_03 :
  positive_difference = 9107.03 := by
  sorry

end positive_difference_is_9107_03_l1249_124932


namespace min_value_xyz_l1249_124906

theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 8) : 
  x + 2 * y + 4 * z ≥ 12 := sorry

end min_value_xyz_l1249_124906


namespace sleeping_bag_selling_price_l1249_124964

def wholesale_cost : ℝ := 24.56
def gross_profit_percentage : ℝ := 0.14

def gross_profit (x : ℝ) : ℝ := gross_profit_percentage * x

def selling_price (x y : ℝ) : ℝ := x + y

theorem sleeping_bag_selling_price :
  selling_price wholesale_cost (gross_profit wholesale_cost) = 28 := by
  sorry

end sleeping_bag_selling_price_l1249_124964


namespace quadratic_func_condition_l1249_124923

noncomputable def f (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_func_condition (b c : ℝ) (h : f (-3) b c = f 1 b c) :
  f 1 b c > c ∧ c > f (-1) b c :=
by
  sorry

end quadratic_func_condition_l1249_124923


namespace yogurt_calories_per_ounce_l1249_124949

variable (calories_strawberries_per_unit : ℕ)
variable (calories_yogurt_total : ℕ)
variable (calories_total : ℕ)
variable (strawberries_count : ℕ)
variable (yogurt_ounces_count : ℕ)

theorem yogurt_calories_per_ounce (h1: strawberries_count = 12)
                                   (h2: yogurt_ounces_count = 6)
                                   (h3: calories_strawberries_per_unit = 4)
                                   (h4: calories_total = 150)
                                   (h5: calories_yogurt_total = calories_total - strawberries_count * calories_strawberries_per_unit):
                                   calories_yogurt_total / yogurt_ounces_count = 17 :=
by
  -- We conjecture that this is correct based on given conditions.
  sorry

end yogurt_calories_per_ounce_l1249_124949


namespace max_third_side_length_l1249_124982

theorem max_third_side_length (x : ℕ) (h1 : 28 + x > 47) (h2 : 47 + x > 28) (h3 : 28 + 47 > x) :
  x = 74 :=
sorry

end max_third_side_length_l1249_124982


namespace range_of_x_l1249_124975

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x > -(1 / 2) :=
sorry

end range_of_x_l1249_124975


namespace common_difference_ne_3_l1249_124942

theorem common_difference_ne_3 
  (d : ℕ) (hd_pos : d > 0) 
  (exists_n : ∃ n : ℕ, 81 = 1 + (n - 1) * d) : 
  d ≠ 3 :=
by sorry

end common_difference_ne_3_l1249_124942


namespace set_intersection_complement_l1249_124927

def U : Set ℝ := Set.univ
def A : Set ℝ := { y | ∃ x, x > 0 ∧ y = 4 / x }
def B : Set ℝ := { y | ∃ x, x < 1 ∧ y = 2^x }
def comp_B : Set ℝ := { y | y ≤ 0 } ∪ { y | y ≥ 2 }
def intersection : Set ℝ := { y | y ≥ 2 }

theorem set_intersection_complement :
  A ∩ comp_B = intersection :=
by
  sorry

end set_intersection_complement_l1249_124927


namespace number_of_integers_covered_l1249_124951

-- Define the number line and the length condition
def unit_length_cm (p : ℝ) := p = 1
def length_AB_cm (length : ℝ) := length = 2009

-- Statement of the proof problem in Lean
theorem number_of_integers_covered (ab_length : ℝ) (unit_length : ℝ) 
    (h1 : unit_length_cm unit_length) (h2 : length_AB_cm ab_length) :
    ∃ n : ℕ, n = 2009 ∨ n = 2010 :=
by
  sorry

end number_of_integers_covered_l1249_124951


namespace find_value_l1249_124972

variable (a b : ℝ)

def quadratic_equation_roots : Prop :=
  a^2 - 4 * a - 1 = 0 ∧ b^2 - 4 * b - 1 = 0

def sum_of_roots : Prop :=
  a + b = 4

def product_of_roots : Prop :=
  a * b = -1

theorem find_value (ha : quadratic_equation_roots a b) (hs : sum_of_roots a b) (hp : product_of_roots a b) :
  2 * a^2 + 3 / b + 5 * b = 22 :=
sorry

end find_value_l1249_124972


namespace smaller_circle_y_coordinate_l1249_124930

theorem smaller_circle_y_coordinate 
  (center : ℝ × ℝ) 
  (P : ℝ × ℝ)
  (S : ℝ × ℝ) 
  (QR : ℝ)
  (r_large : ℝ):
    center = (0, 0) → P = (5, 12) → QR = 2 → S.1 = 0 → S.2 = k → r_large = 13 → k = 11 := 
by
  intros h_center hP hQR hSx hSy hr_large
  sorry

end smaller_circle_y_coordinate_l1249_124930


namespace remainder_of_3_pow_100_plus_5_mod_8_l1249_124980

theorem remainder_of_3_pow_100_plus_5_mod_8 :
  (3^100 + 5) % 8 = 6 := by
sorry

end remainder_of_3_pow_100_plus_5_mod_8_l1249_124980


namespace count_consecutive_integers_l1249_124925

theorem count_consecutive_integers : 
  ∃ n : ℕ, (∀ x : ℕ, (1 < x ∧ x < 111) → (x - 1) + x + (x + 1) < 333) ∧ n = 109 := 
  by
    sorry

end count_consecutive_integers_l1249_124925


namespace correct_system_of_equations_l1249_124900

theorem correct_system_of_equations (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  (y = 7 * x + 7) ∧ (y = 9 * (x - 1)) :=
by
  sorry

end correct_system_of_equations_l1249_124900


namespace part1_part2_l1249_124994

noncomputable section

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + 2 * a)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 4 ↔ f x a < 4 - 2 * a) →
  a = 0 := 
sorry

theorem part2 (m : ℝ) :
  (∀ x : ℝ, f x 1 - f (-2 * x) 1 ≤ x + m) →
  2 ≤ m :=
sorry

end part1_part2_l1249_124994


namespace find_tan_theta_l1249_124914

theorem find_tan_theta
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h2 : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 :=
sorry

end find_tan_theta_l1249_124914


namespace proof_problem_l1249_124955

variable (x y : ℕ) -- define x and y as natural numbers

-- Define the problem-specific variables m and n
variable (m n : ℕ)

-- Assume the conditions given in the problem
axiom H1 : 2 = m
axiom H2 : n = 3

-- The goal is to prove that -m^n equals -8 given the conditions H1 and H2
theorem proof_problem : - (m^n : ℤ) = -8 :=
by
  sorry

end proof_problem_l1249_124955


namespace number_of_boxes_l1249_124901

theorem number_of_boxes (eggs_per_box : ℕ) (total_eggs : ℕ) (h1 : eggs_per_box = 3) (h2 : total_eggs = 6) : total_eggs / eggs_per_box = 2 := by
  sorry

end number_of_boxes_l1249_124901


namespace b_finishes_remaining_work_in_5_days_l1249_124993

theorem b_finishes_remaining_work_in_5_days :
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  days_b_to_finish = 5 :=
by
  let A_work_rate := 1 / 4
  let B_work_rate := 1 / 14
  let combined_work_rate := A_work_rate + B_work_rate
  let work_completed_together := 2 * combined_work_rate
  let work_remaining := 1 - work_completed_together
  let days_b_to_finish := work_remaining / B_work_rate
  show days_b_to_finish = 5
  sorry

end b_finishes_remaining_work_in_5_days_l1249_124993
