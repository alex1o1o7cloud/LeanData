import Mathlib

namespace product_divisible_by_60_l2007_200773

open Nat

theorem product_divisible_by_60 (S : Finset ℕ) (h_card : S.card = 10) (h_sum : S.sum id = 62) :
  60 ∣ S.prod id :=
  sorry

end product_divisible_by_60_l2007_200773


namespace min_value_of_squares_l2007_200744

theorem min_value_of_squares (a b t : ℝ) (h : a + b = t) : (a^2 + b^2) ≥ t^2 / 2 := 
by
  sorry

end min_value_of_squares_l2007_200744


namespace original_price_of_sweater_l2007_200782

theorem original_price_of_sweater (sold_price : ℝ) (discount : ℝ) (original_price : ℝ) 
    (h1 : sold_price = 120) (h2 : discount = 0.40) (h3: (1 - discount) * original_price = sold_price) : 
    original_price = 200 := by 
  sorry

end original_price_of_sweater_l2007_200782


namespace profit_in_december_l2007_200719

variable (a : ℝ)

theorem profit_in_december (h_a: a > 0):
  (1 - 0.06) * (1 + 0.10) * a = (1 - 0.06) * (1 + 0.10) * a :=
by
  sorry

end profit_in_december_l2007_200719


namespace inequality_proof_l2007_200752

theorem inequality_proof (a b : ℝ) (h1 : b < 0) (h2 : 0 < a) : a - b > a + b :=
sorry

end inequality_proof_l2007_200752


namespace no_real_solutions_for_equation_l2007_200792

theorem no_real_solutions_for_equation : ¬ (∃ x : ℝ, x + Real.sqrt (2 * x - 6) = 5) :=
sorry

end no_real_solutions_for_equation_l2007_200792


namespace trigonometric_identity_l2007_200754

-- Definition for the given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 2

-- The proof goal
theorem trigonometric_identity (α : ℝ) (h : tan_alpha α) : 
  Real.cos (π + α) * Real.cos (π / 2 + α) = 2 / 5 :=
by
  sorry

end trigonometric_identity_l2007_200754


namespace evaluate_at_2_l2007_200732

-- Define the polynomial function using Lean
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1

-- State the theorem that f(2) evaluates to 35 using Horner's method
theorem evaluate_at_2 : f 2 = 35 := by
  sorry

end evaluate_at_2_l2007_200732


namespace prob_students_both_days_l2007_200710

def num_scenarios (students : ℕ) (choices : ℕ) : ℕ :=
  choices ^ students

def scenarios_sat_sun (total_scenarios : ℕ) (both_days_empty : ℕ) : ℕ :=
  total_scenarios - both_days_empty

theorem prob_students_both_days :
  let students := 3
  let choices := 2
  let total_scenarios := num_scenarios students choices
  let both_days_empty := 2 -- When all choose Saturday or all choose Sunday
  let scenarios_both := scenarios_sat_sun total_scenarios both_days_empty
  let probability := scenarios_both / total_scenarios
  probability = 3 / 4 :=
by
  sorry

end prob_students_both_days_l2007_200710


namespace minimize_expression_l2007_200793

theorem minimize_expression :
  ∀ n : ℕ, 0 < n → (n = 6 ↔ ∀ m : ℕ, 0 < m → (n ≤ (2 * (m + 9))/(m))) := 
by
  sorry

end minimize_expression_l2007_200793


namespace mod_remainder_l2007_200730

theorem mod_remainder (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by
  sorry

end mod_remainder_l2007_200730


namespace correct_operation_c_l2007_200783

theorem correct_operation_c (a b : ℝ) :
  ¬ (a^2 + a^2 = 2 * a^4)
  ∧ ¬ ((-3 * a * b^2)^2 = -6 * a^2 * b^4)
  ∧ a^6 / (-a)^2 = a^4
  ∧ ¬ ((a - b)^2 = a^2 - b^2) :=
by
  sorry

end correct_operation_c_l2007_200783


namespace feathers_before_crossing_road_l2007_200716

theorem feathers_before_crossing_road : 
  ∀ (F : ℕ), 
  (F - (2 * 23) = 5217) → 
  F = 5263 :=
by
  intros F h
  sorry

end feathers_before_crossing_road_l2007_200716


namespace area_decreases_by_28_l2007_200721

def decrease_in_area (s h : ℤ) (h_eq : h = s + 3) : ℤ :=
  let new_area := (s - 4) * (s + 7)
  let original_area := s * h
  new_area - original_area

theorem area_decreases_by_28 (s h : ℤ) (h_eq : h = s + 3) : decrease_in_area s h h_eq = -28 :=
sorry

end area_decreases_by_28_l2007_200721


namespace arith_seq_sum_signs_l2007_200700

variable {α : Type*} [LinearOrderedField α]
variable {a : ℕ → α} {S : ℕ → α} {d : α}

noncomputable def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  n * (a 1 + a n) / 2

-- Given conditions
variable (a_8_neg : a 8 < 0)
variable (a_9_pos : a 9 > 0)
variable (a_9_greater_abs_a_8 : a 9 > abs (a 8))

-- The theorem to prove
theorem arith_seq_sum_signs (h : is_arith_seq a) :
  (∀ n, n ≤ 15 → sum_first_n_terms a n < 0) ∧ (∀ n, n ≥ 16 → sum_first_n_terms a n > 0) :=
sorry

end arith_seq_sum_signs_l2007_200700


namespace solve_inequality_l2007_200789

open Real

theorem solve_inequality (a : ℝ) :
  ((a < 0 ∨ a > 1) → (∀ x, a < x ∧ x < a^2 ↔ (x - a) * (x - a^2) < 0)) ∧
  ((0 < a ∧ a < 1) → (∀ x, a^2 < x ∧ x < a ↔ (x - a) * (x - a^2) < 0)) ∧
  ((a = 0 ∨ a = 1) → (∀ x, ¬((x - a) * (x - a^2) < 0))) :=
by
  sorry

end solve_inequality_l2007_200789


namespace find_f_of_2_l2007_200768

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = 4 * x - 1) : f 2 = 3 :=
by
  sorry

end find_f_of_2_l2007_200768


namespace avg_annual_growth_rate_optimal_selling_price_l2007_200733

theorem avg_annual_growth_rate (v2022 v2024 : ℕ) (x : ℝ) 
  (h1 : v2022 = 200000) 
  (h2 : v2024 = 288000)
  (h3: v2024 = v2022 * (1 + x)^2) :
  x = 0.2 :=
by
  sorry

theorem optimal_selling_price (cost : ℝ) (initial_price : ℝ) (initial_cups : ℕ) 
  (price_drop_effect : ℝ) (initial_profit : ℝ) (daily_profit : ℕ) (y : ℝ)
  (h1 : cost = 6)
  (h2 : initial_price = 25) 
  (h3 : initial_cups = 300)
  (h4 : price_drop_effect = 1)
  (h5 : initial_profit = 6300)
  (h6 : (y - cost) * (initial_cups + 30 * (initial_price - y)) = daily_profit) :
  y = 20 :=
by
  sorry

end avg_annual_growth_rate_optimal_selling_price_l2007_200733


namespace no_absolute_winner_prob_l2007_200791

def P_A_beats_B : ℝ := 0.6
def P_B_beats_V : ℝ := 0.4
def P_V_beats_A : ℝ := 1

theorem no_absolute_winner_prob :
  P_A_beats_B * P_B_beats_V * P_V_beats_A + 
  P_A_beats_B * (1 - P_B_beats_V) * (1 - P_V_beats_A) = 0.36 :=
by
  sorry

end no_absolute_winner_prob_l2007_200791


namespace sum_C_D_eq_one_fifth_l2007_200747

theorem sum_C_D_eq_one_fifth (D C : ℚ) :
  (∀ x : ℚ, (Dx - 13) / (x^2 - 9 * x + 20) = C / (x - 4) + 5 / (x - 5)) →
  (C + D) = 1/5 :=
by
  sorry

end sum_C_D_eq_one_fifth_l2007_200747


namespace arithmetic_sequence_general_formula_l2007_200781

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom a2 : a 2 = 6
axiom S5 : S 5 = 40

-- Prove the general formulas
theorem arithmetic_sequence_general_formula (n : ℕ)
  (h1 : ∃ d a1, ∀ n, a n = a1 + (n - 1) * d)
  (h2 : ∃ d a1, ∀ n, S n = n * ((2 * a1) + (n - 1) * d) / 2) :
  (a n = 2 * n + 2) ∧ (S n = n * (n + 3)) := by
  sorry

end arithmetic_sequence_general_formula_l2007_200781


namespace no_value_of_b_valid_l2007_200706

theorem no_value_of_b_valid (b n : ℤ) : b^2 + 3 * b + 1 ≠ n^2 := by
  sorry

end no_value_of_b_valid_l2007_200706


namespace linda_fraction_savings_l2007_200796

theorem linda_fraction_savings (savings tv_cost : ℝ) (f : ℝ) 
  (h1 : savings = 800) 
  (h2 : tv_cost = 200) 
  (h3 : f * savings + tv_cost = savings) : 
  f = 3 / 4 := 
sorry

end linda_fraction_savings_l2007_200796


namespace sum_is_1716_l2007_200797

-- Given conditions:
variables (a b c d : ℤ)
variable (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
variable (h_roots1 : ∀ t, t * t - 12 * a * t - 13 * b = 0 ↔ t = c ∨ t = d)
variable (h_roots2 : ∀ t, t * t - 12 * c * t - 13 * d = 0 ↔ t = a ∨ t = b)

-- Prove the desired sum of the constants:
theorem sum_is_1716 : a + b + c + d = 1716 :=
by
  sorry

end sum_is_1716_l2007_200797


namespace books_per_continent_l2007_200709

-- Definition of the given conditions
def total_books := 488
def continents_visited := 4

-- The theorem we need to prove
theorem books_per_continent : total_books / continents_visited = 122 :=
sorry

end books_per_continent_l2007_200709


namespace eq_satisfied_in_entire_space_l2007_200757

theorem eq_satisfied_in_entire_space (x y z : ℝ) : 
  (x + y + z)^2 = x^2 + y^2 + z^2 ↔ xy + xz + yz = 0 :=
by
  sorry

end eq_satisfied_in_entire_space_l2007_200757


namespace common_difference_of_arithmetic_sequence_l2007_200763

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a n = a 0 + n * d) (h2 : a 5 = 10) (h3 : a 10 = -5) : d = -3 := 
by 
  sorry

end common_difference_of_arithmetic_sequence_l2007_200763


namespace length_PQ_l2007_200746

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 + (P.z - Q.z) ^ 2)

def P : Point3D := { x := 3, y := 4, z := 5 }

def Q : Point3D := { x := 3, y := 4, z := 0 }

theorem length_PQ : distance P Q = 5 :=
by
  sorry

end length_PQ_l2007_200746


namespace neither_A_B_C_prob_correct_l2007_200742

noncomputable def P (A B C : Prop) : Prop :=
  let P_A := 0.25
  let P_B := 0.35
  let P_C := 0.40
  let P_A_and_B := 0.10
  let P_A_and_C := 0.15
  let P_B_and_C := 0.20
  let P_A_and_B_and_C := 0.05
  
  let P_A_or_B_or_C := 
    P_A + P_B + P_C - P_A_and_B - P_A_and_C - P_B_and_C + P_A_and_B_and_C
  
  let P_neither_A_nor_B_nor_C := 1 - P_A_or_B_or_C
    
  P_neither_A_nor_B_nor_C = 0.45

theorem neither_A_B_C_prob_correct :
  P A B C := by
  sorry

end neither_A_B_C_prob_correct_l2007_200742


namespace simplify_expression_l2007_200741

theorem simplify_expression (x : ℝ) : 
  (3*x - 4)*(2*x + 9) - (x + 6)*(3*x + 2) = 3*x^2 - x - 48 :=
by
  sorry

end simplify_expression_l2007_200741


namespace operation_ab_equals_nine_l2007_200755

variable (a b : ℝ)

def operation (x y : ℝ) : ℝ := a * x + b * y - 1

theorem operation_ab_equals_nine
  (h1 : operation a b 1 2 = 4)
  (h2 : operation a b (-2) 3 = 10)
  : a * b = 9 :=
by
  sorry

end operation_ab_equals_nine_l2007_200755


namespace jack_more_emails_morning_than_afternoon_l2007_200745

def emails_afternoon := 3
def emails_morning := 5

theorem jack_more_emails_morning_than_afternoon :
  emails_morning - emails_afternoon = 2 :=
by
  sorry

end jack_more_emails_morning_than_afternoon_l2007_200745


namespace fractions_product_l2007_200775

theorem fractions_product :
  (8 / 4) * (10 / 25) * (20 / 10) * (15 / 45) * (40 / 20) * (24 / 8) * (30 / 15) * (35 / 7) = 64 := by
  sorry

end fractions_product_l2007_200775


namespace minimum_discount_l2007_200784

variable (C P : ℝ) (r x : ℝ)

def microwave_conditions := 
  C = 1000 ∧ 
  P = 1500 ∧ 
  r = 0.02 ∧ 
  P * (x / 10) ≥ C * (1 + r)

theorem minimum_discount : ∃ x, microwave_conditions C P r x ∧ x ≥ 6.8 :=
by 
  sorry

end minimum_discount_l2007_200784


namespace point_bisector_second_quadrant_l2007_200760

theorem point_bisector_second_quadrant (a : ℝ) : 
  (a < 0 ∧ 2 > 0) ∧ (2 = -a) → a = -2 :=
by sorry

end point_bisector_second_quadrant_l2007_200760


namespace distance_to_school_l2007_200722

variable (T D : ℕ)

/-- Given the conditions, prove the distance from the child's home to the school is 630 meters --/
theorem distance_to_school :
  (5 * (T + 6) = D) →
  (7 * (T - 30) = D) →
  D = 630 :=
by
  intros h1 h2
  sorry

end distance_to_school_l2007_200722


namespace biking_distance_l2007_200704

/-- Mathematical equivalent proof problem for the distance biked -/
theorem biking_distance
  (x t d : ℕ)
  (h1 : d = (x + 1) * (3 * t / 4))
  (h2 : d = (x - 1) * (t + 3)) :
  d = 36 :=
by
  -- The proof goes here
  sorry

end biking_distance_l2007_200704


namespace island_knight_majority_villages_l2007_200769

def NumVillages := 1000
def NumInhabitants := 99
def TotalKnights := 54054
def AnswersPerVillage : ℕ := 66 -- Number of villagers who answered "more knights"
def RemainingAnswersPerVillage : ℕ := 33 -- Number of villagers who answered "more liars"

theorem island_knight_majority_villages : 
  ∃ n : ℕ, n = 638 ∧ (66 * n + 33 * (NumVillages - n) = TotalKnights) :=
by -- Begin the proof
  sorry -- Proof to be filled in later

end island_knight_majority_villages_l2007_200769


namespace median_production_l2007_200762

def production_data : List ℕ := [5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10]

def median (l : List ℕ) : ℕ :=
  if l.length % 2 = 1 then
    l.nthLe (l.length / 2) sorry
  else
    let m := l.length / 2
    (l.nthLe (m - 1) sorry + l.nthLe m sorry) / 2

theorem median_production :
  median (production_data) = 8 :=
by
  sorry

end median_production_l2007_200762


namespace smallest_two_digit_prime_with_conditions_l2007_200778

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem smallest_two_digit_prime_with_conditions :
  ∃ p : ℕ, is_prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 = 3) ∧ is_composite (((p % 10) * 10) + (p / 10) + 5) ∧ p = 31 :=
by
  sorry

end smallest_two_digit_prime_with_conditions_l2007_200778


namespace scheduled_conference_games_total_l2007_200749

def number_of_teams_in_A := 7
def number_of_teams_in_B := 5
def games_within_division (n : Nat) : Nat := n * (n - 1)
def interdivision_games := 7 * 5
def rivalry_games := 7

theorem scheduled_conference_games_total : 
  let games_A := games_within_division number_of_teams_in_A
  let games_B := games_within_division number_of_teams_in_B
  let total_games := games_A + games_B + interdivision_games + rivalry_games
  total_games = 104 :=
by
  sorry

end scheduled_conference_games_total_l2007_200749


namespace third_rectangle_area_l2007_200714

-- Definitions for dimensions of the first two rectangles
def rect1_length := 3
def rect1_width := 8

def rect2_length := 2
def rect2_width := 5

-- Total area of the first two rectangles
def total_area := (rect1_length * rect1_width) + (rect2_length * rect2_width)

-- Declaration of the theorem to be proven
theorem third_rectangle_area :
  ∃ a b : ℝ, a * b = 4 ∧ total_area + a * b = total_area + 4 :=
by
  sorry

end third_rectangle_area_l2007_200714


namespace sequence_23rd_term_is_45_l2007_200707

def sequence_game (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * n - 1 else 2 * n + 1

theorem sequence_23rd_term_is_45 :
  sequence_game 23 = 45 :=
by
  -- Proving the 23rd term in the sequence as given by the game rules
  sorry

end sequence_23rd_term_is_45_l2007_200707


namespace proof_b_greater_a_greater_c_l2007_200738

def a : ℤ := -2 * 3^2
def b : ℤ := (-2 * 3)^2
def c : ℤ := - (2 * 3)^2

theorem proof_b_greater_a_greater_c (ha : a = -18) (hb : b = 36) (hc : c = -36) : b > a ∧ a > c := 
by
  rw [ha, hb, hc]
  exact And.intro (by norm_num) (by norm_num)

end proof_b_greater_a_greater_c_l2007_200738


namespace running_time_around_pentagon_l2007_200756

theorem running_time_around_pentagon :
  let l₁ := 40
  let l₂ := 50
  let l₃ := 60
  let l₄ := 45
  let l₅ := 55
  let v₁ := 9 * 1000 / 60
  let v₂ := 8 * 1000 / 60
  let v₃ := 7 * 1000 / 60
  let v₄ := 6 * 1000 / 60
  let v₅ := 5 * 1000 / 60
  let t₁ := l₁ / v₁
  let t₂ := l₂ / v₂
  let t₃ := l₃ / v₃
  let t₄ := l₄ / v₄
  let t₅ := l₅ / v₅
  t₁ + t₂ + t₃ + t₄ + t₅ = 2.266 := by
    sorry

end running_time_around_pentagon_l2007_200756


namespace length_side_AB_is_4_l2007_200794

-- Defining a triangle ABC with area 6
variables {A B C K L Q : Type*}
variables {side_AB : Float} {ratio_K : Float} {ratio_L : Float} {dist_Q : Float}
variables (area_ABC : ℝ := 6) (ratio_AK_BK : ℝ := 2 / 3) (ratio_AL_LC : ℝ := 5 / 3)
variables (dist_Q_to_AB : ℝ := 1.5)

theorem length_side_AB_is_4 : 
  side_AB = 4 → 
  (area_ABC = 6 ∧ ratio_AK_BK = 2 / 3 ∧ ratio_AL_LC = 5 / 3 ∧ dist_Q_to_AB = 1.5) :=
by
  sorry

end length_side_AB_is_4_l2007_200794


namespace correct_option_is_d_l2007_200786

theorem correct_option_is_d (x : ℚ) : -x^3 = (-x)^3 :=
sorry

end correct_option_is_d_l2007_200786


namespace quadratic_function_symmetry_l2007_200774

theorem quadratic_function_symmetry
  (p : ℝ → ℝ)
  (h_sym : ∀ x, p (5.5 - x) = p (5.5 + x))
  (h_0 : p 0 = -4) :
  p 11 = -4 :=
by sorry

end quadratic_function_symmetry_l2007_200774


namespace sum_of_acute_angles_l2007_200734

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (hcosα : Real.cos α = 1 / Real.sqrt 10)
variable (hcosβ : Real.cos β = 1 / Real.sqrt 5)

theorem sum_of_acute_angles :
  α + β = 3 * Real.pi / 4 := by
  sorry

end sum_of_acute_angles_l2007_200734


namespace polygon_interior_exterior_relation_l2007_200743

theorem polygon_interior_exterior_relation :
  ∃ (n : ℕ), (n > 2) ∧ ((n - 2) * 180 = 4 * 360) ∧ n = 10 :=
by
  sorry

end polygon_interior_exterior_relation_l2007_200743


namespace trig_identity_l2007_200718

theorem trig_identity (θ : ℝ) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
  sorry

end trig_identity_l2007_200718


namespace oil_spending_l2007_200767

-- Define the original price per kg of oil
def original_price (P : ℝ) := P

-- Define the reduced price per kg of oil
def reduced_price (P : ℝ) := 0.75 * P

-- Define the reduced price as Rs. 60
def reduced_price_fixed := 60

-- State the condition that reduced price enables 5 kgs more oil
def extra_kg := 5

-- The amount of money spent by housewife at reduced price which is to be proven as Rs. 1200
def amount_spent (M : ℝ) := M

-- Define the problem to prove in Lean 4
theorem oil_spending (P X : ℝ) (h1 : reduced_price P = reduced_price_fixed) (h2 : X * original_price P = (X + extra_kg) * reduced_price_fixed) : amount_spent ((X + extra_kg) * reduced_price_fixed) = 1200 :=
  sorry

end oil_spending_l2007_200767


namespace sugar_fill_count_l2007_200728

noncomputable def sugar_needed_for_one_batch : ℚ := 3 + 1/2
noncomputable def total_batches : ℕ := 2
noncomputable def cup_capacity : ℚ := 1/3
noncomputable def total_sugar_needed : ℚ := total_batches * sugar_needed_for_one_batch

theorem sugar_fill_count : (total_sugar_needed / cup_capacity) = 21 :=
by
  -- Assuming necessary preliminary steps already defined, we just check the equality directly
  sorry

end sugar_fill_count_l2007_200728


namespace inequality_solution_l2007_200790

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) ↔
  (1 < x ∧ x < 2 ∨ 3 < x ∧ x < 6) :=
by
  sorry

end inequality_solution_l2007_200790


namespace isosceles_triangle_angle_B_l2007_200777

theorem isosceles_triangle_angle_B (A B C : ℝ)
  (h_triangle : (A + B + C = 180))
  (h_exterior_A : 180 - A = 110)
  (h_sum_angles : A + B + C = 180) :
  B = 70 ∨ B = 55 ∨ B = 40 :=
by
  sorry

end isosceles_triangle_angle_B_l2007_200777


namespace cube_fit_count_cube_volume_percentage_l2007_200753

-- Definitions based on the conditions in the problem.
def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_side : ℕ := 4

-- Definitions for the calculated values.
def num_cubes_length : ℕ := box_length / cube_side
def num_cubes_width : ℕ := box_width / cube_side
def num_cubes_height : ℕ := box_height / cube_side

def total_cubes : ℕ := num_cubes_length * num_cubes_width * num_cubes_height

def volume_cube : ℕ := cube_side^3
def volume_cubes_total : ℕ := total_cubes * volume_cube
def volume_box : ℕ := box_length * box_width * box_height

def percentage_volume : ℕ := (volume_cubes_total * 100) / volume_box

-- The proof statements.
theorem cube_fit_count : total_cubes = 6 := by
  sorry

theorem cube_volume_percentage : percentage_volume = 100 := by
  sorry

end cube_fit_count_cube_volume_percentage_l2007_200753


namespace percentage_runs_by_running_l2007_200711

theorem percentage_runs_by_running 
  (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) 
  (runs_per_boundary : ℕ) (runs_per_six : ℕ)
  (H_total_runs : total_runs = 120)
  (H_boundaries : boundaries = 3)
  (H_sixes : sixes = 8)
  (H_runs_per_boundary : runs_per_boundary = 4)
  (H_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs : ℚ) * 100 = 50 := 
by
  sorry

end percentage_runs_by_running_l2007_200711


namespace cannot_be_sum_of_consecutive_nat_iff_power_of_two_l2007_200750

theorem cannot_be_sum_of_consecutive_nat_iff_power_of_two (n : ℕ) : 
  (∀ a b : ℕ, n ≠ (b - a + 1) * (a + b) / 2) ↔ (∃ k : ℕ, n = 2 ^ k) := by
  sorry

end cannot_be_sum_of_consecutive_nat_iff_power_of_two_l2007_200750


namespace gathering_gift_exchange_l2007_200726

def number_of_guests (x : ℕ) : Prop :=
  x * (x - 1) = 56

theorem gathering_gift_exchange :
  ∃ x : ℕ, number_of_guests x :=
sorry

end gathering_gift_exchange_l2007_200726


namespace general_formula_a_S_n_no_arithmetic_sequence_in_b_l2007_200715

def sequence_a (a : ℕ → ℚ) :=
  (a 1 = 1 / 4) ∧ (∀ n : ℕ, n > 0 → 3 * a (n + 1) - 2 * a n = 1)

def sequence_b (b : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n : ℕ, n > 0 → b n = a (n + 1) - a n

theorem general_formula_a_S_n (a : ℕ → ℚ) (S : ℕ → ℚ) :
  sequence_a a →
  (∀ n : ℕ, n > 0 → a n = 1 - (3 / 4) * (2 / 3)^(n - 1)) →
  (∀ n : ℕ, n > 0 → S n = (2 / 3)^(n - 2) + n - 9 / 4) →
  True := sorry

theorem no_arithmetic_sequence_in_b (b : ℕ → ℚ) (a : ℕ → ℚ) :
  sequence_b b a →
  (∀ n : ℕ, n > 0 → b n = (1 / 4) * (2 / 3)^(n - 1)) →
  (∀ r s t : ℕ, r < s ∧ s < t → ¬ (b s - b r = b t - b s)) :=
  sorry

end general_formula_a_S_n_no_arithmetic_sequence_in_b_l2007_200715


namespace original_ticket_price_l2007_200764

open Real

theorem original_ticket_price 
  (P : ℝ)
  (total_revenue : ℝ)
  (revenue_equation : total_revenue = 10 * 0.60 * P + 20 * 0.85 * P + 15 * P) 
  (total_revenue_val : total_revenue = 760) : 
  P = 20 := 
by
  sorry

end original_ticket_price_l2007_200764


namespace complement_of_A_in_U_l2007_200724

-- Define the universal set U
def U : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 1 / x ≥ 1}

-- Define the complement of A within U
def complement_U_A : Set ℝ := {x | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_of_A_in_U : complement_U_A = {x | -1 < x ∧ x ≤ 0} :=
  sorry

end complement_of_A_in_U_l2007_200724


namespace characterization_of_M_l2007_200758

noncomputable def M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem characterization_of_M : M = {z : ℂ | ∃ r : ℝ, z = r} :=
by
  sorry

end characterization_of_M_l2007_200758


namespace children_more_than_adults_l2007_200779

-- Conditions
def total_members : ℕ := 120
def adult_percentage : ℝ := 0.40
def child_percentage : ℝ := 1 - adult_percentage

-- Proof problem statement
theorem children_more_than_adults : 
  let number_of_adults := adult_percentage * total_members
  let number_of_children := child_percentage * total_members
  let difference := number_of_children - number_of_adults
  difference = 24 :=
by
  sorry

end children_more_than_adults_l2007_200779


namespace circle_equation_through_points_l2007_200703

-- Definitions of the points A, B, and C
def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (-1, 1)

-- Prove that the equation of the circle passing through A, B, and C is (x - 1)^2 + y^2 = 5
theorem circle_equation_through_points :
  ∃ (D E F : ℝ), (∀ x y : ℝ, 
  x^2 + y^2 + D * x + E * y + F = 0 ↔
  x = -1 ∧ y = -1 ∨ 
  x = 2 ∧ y = 2 ∨ 
  x = -1 ∧ y = 1) ∧ 
  ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x - 1)^2 + y^2 = 5 :=
by
  sorry

end circle_equation_through_points_l2007_200703


namespace regular_hexagon_interior_angle_l2007_200701

theorem regular_hexagon_interior_angle : ∀ (n : ℕ), n = 6 → ∀ (angle_sum : ℕ), angle_sum = (n - 2) * 180 → (∀ (angle : ℕ), angle = angle_sum / n → angle = 120) :=
by sorry

end regular_hexagon_interior_angle_l2007_200701


namespace sum_of_Ns_l2007_200765

theorem sum_of_Ns (N R : ℝ) (hN_nonzero : N ≠ 0) (h_eq : N - 3 * N^2 = R) : 
  ∃ N1 N2 : ℝ, N1 ≠ 0 ∧ N2 ≠ 0 ∧ 3 * N1^2 - N1 + R = 0 ∧ 3 * N2^2 - N2 + R = 0 ∧ (N1 + N2) = 1 / 3 :=
sorry

end sum_of_Ns_l2007_200765


namespace difference_in_profit_l2007_200713

def records := 300
def price_sammy := 4
def price_bryan_two_thirds := 6
def price_bryan_one_third := 1
def price_christine_thirty := 10
def price_christine_remaining := 3

def profit_sammy := records * price_sammy
def profit_bryan := ((records * 2 / 3) * price_bryan_two_thirds) + ((records * 1 / 3) * price_bryan_one_third)
def profit_christine := (30 * price_christine_thirty) + ((records - 30) * price_christine_remaining)

theorem difference_in_profit : 
  max profit_sammy (max profit_bryan profit_christine) - min profit_sammy (min profit_bryan profit_christine) = 190 :=
by
  sorry

end difference_in_profit_l2007_200713


namespace depth_of_channel_l2007_200739

theorem depth_of_channel (h : ℝ) 
  (top_width : ℝ := 12) (bottom_width : ℝ := 6) (area : ℝ := 630) :
  1 / 2 * (top_width + bottom_width) * h = area → h = 70 :=
sorry

end depth_of_channel_l2007_200739


namespace longest_segment_CD_l2007_200761

variables (A B C D : Type)
variables (angle_ABD angle_ADB angle_BDC angle_CBD : ℝ)

axiom angle_ABD_eq : angle_ABD = 30
axiom angle_ADB_eq : angle_ADB = 65
axiom angle_BDC_eq : angle_BDC = 60
axiom angle_CBD_eq : angle_CBD = 80

theorem longest_segment_CD
  (h_ABD : angle_ABD = 30)
  (h_ADB : angle_ADB = 65)
  (h_BDC : angle_BDC = 60)
  (h_CBD : angle_CBD = 80) : false :=
sorry

end longest_segment_CD_l2007_200761


namespace coin_ratio_l2007_200705

theorem coin_ratio (n₁ n₅ n₂₅ : ℕ) (total_value : ℕ) 
  (h₁ : n₁ = 40) 
  (h₅ : n₅ = 40) 
  (h₂₅ : n₂₅ = 40) 
  (hv : total_value = 70) 
  (hv_calc : n₁ * 1 + n₅ * (50 / 100) + n₂₅ * (25 / 100) = total_value) : 
  n₁ = n₅ ∧ n₁ = n₂₅ :=
by
  sorry

end coin_ratio_l2007_200705


namespace percentage_taxed_l2007_200785

theorem percentage_taxed (T : ℝ) (H1 : 3840 = T * (P : ℝ)) (H2 : 480 = 0.25 * T * (P : ℝ)) : P = 0.5 := 
by
  sorry

end percentage_taxed_l2007_200785


namespace Megan_pictures_left_l2007_200770

theorem Megan_pictures_left (zoo_pictures museum_pictures deleted_pictures : ℕ) 
  (h1 : zoo_pictures = 15) 
  (h2 : museum_pictures = 18) 
  (h3 : deleted_pictures = 31) : 
  zoo_pictures + museum_pictures - deleted_pictures = 2 := 
by
  sorry

end Megan_pictures_left_l2007_200770


namespace triangle_sum_l2007_200748

-- Define the triangle operation
def triangle (a b c : ℕ) : ℕ := a + b + c

-- State the theorem
theorem triangle_sum :
  triangle 2 4 3 + triangle 1 6 5 = 21 :=
by
  sorry

end triangle_sum_l2007_200748


namespace rectangle_area_increase_l2007_200799

variable (L B : ℝ)

theorem rectangle_area_increase :
  let L_new := 1.30 * L
  let B_new := 1.45 * B
  let A_original := L * B
  let A_new := L_new * B_new
  let A_increase := A_new - A_original
  let percentage_increase := (A_increase / A_original) * 100
  percentage_increase = 88.5 := by
    sorry

end rectangle_area_increase_l2007_200799


namespace CE_squared_plus_DE_squared_proof_l2007_200731

noncomputable def CE_squared_plus_DE_squared (radius : ℝ) (diameter : ℝ) (BE : ℝ) (angle_AEC : ℝ) : ℝ :=
  if radius = 10 ∧ diameter = 20 ∧ BE = 4 ∧ angle_AEC = 30 then 200 else sorry

theorem CE_squared_plus_DE_squared_proof : CE_squared_plus_DE_squared 10 20 4 30 = 200 := by
  sorry

end CE_squared_plus_DE_squared_proof_l2007_200731


namespace number_of_single_windows_upstairs_l2007_200737

theorem number_of_single_windows_upstairs :
  ∀ (num_double_windows_downstairs : ℕ)
    (glass_panels_per_double_window : ℕ)
    (num_single_windows_upstairs : ℕ)
    (glass_panels_per_single_window : ℕ)
    (total_glass_panels : ℕ),
  num_double_windows_downstairs = 6 →
  glass_panels_per_double_window = 4 →
  glass_panels_per_single_window = 4 →
  total_glass_panels = 80 →
  num_single_windows_upstairs = (total_glass_panels - (num_double_windows_downstairs * glass_panels_per_double_window)) / glass_panels_per_single_window →
  num_single_windows_upstairs = 14 :=
by
  intros
  sorry

end number_of_single_windows_upstairs_l2007_200737


namespace quadratic_vertex_form_l2007_200723

theorem quadratic_vertex_form (a h k x: ℝ) (h_a : a = 3) (hx : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by {
  sorry
}

end quadratic_vertex_form_l2007_200723


namespace maximize_profit_l2007_200780

/-- A car sales company purchased a total of 130 vehicles of models A and B, 
with x vehicles of model A purchased. The profit y is defined by selling 
prices and factory prices of both models. -/
def total_profit (x : ℕ) : ℝ := -2 * x + 520

theorem maximize_profit :
  ∃ x : ℕ, (130 - x ≤ 2 * x) ∧ (total_profit x = 432) ∧ (∀ y : ℕ, (130 - y ≤ 2 * y) → (total_profit y ≤ 432)) :=
by {
  sorry
}

end maximize_profit_l2007_200780


namespace veenapaniville_private_independent_district_A_l2007_200729

theorem veenapaniville_private_independent_district_A :
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_B_private := 2
  let remaining_schools := total_schools - district_A_schools - district_B_schools
  let each_kind_in_C := remaining_schools / 3
  let district_C_private := each_kind_in_C
  let district_A_private := private_schools - district_B_private - district_C_private
  district_A_private = 2 := by
  sorry

end veenapaniville_private_independent_district_A_l2007_200729


namespace totalOwlsOnFence_l2007_200759

-- Define the conditions given in the problem
def initialOwls : Nat := 3
def joinedOwls : Nat := 2

-- Define the total number of owls
def totalOwls : Nat := initialOwls + joinedOwls

-- State the theorem we want to prove
theorem totalOwlsOnFence : totalOwls = 5 := by
  sorry

end totalOwlsOnFence_l2007_200759


namespace find_CD_l2007_200787

theorem find_CD (C D : ℚ) :
  (∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 → (7 * x - 4) / (x ^ 2 - 9 * x - 36) = C / (x - 12) + D / (x + 3))
  → C = 16 / 3 ∧ D = 5 / 3 :=
by
  sorry

end find_CD_l2007_200787


namespace fruits_left_l2007_200751

theorem fruits_left (plums guavas apples given : ℕ) (h1 : plums = 16) (h2 : guavas = 18) (h3 : apples = 21) (h4 : given = 40) : 
  (plums + guavas + apples - given = 15) :=
by
  sorry

end fruits_left_l2007_200751


namespace time_to_cross_first_platform_l2007_200766

variable (length_first_platform : ℝ)
variable (length_second_platform : ℝ)
variable (time_to_cross_second_platform : ℝ)
variable (length_of_train : ℝ)

theorem time_to_cross_first_platform :
  length_first_platform = 160 →
  length_second_platform = 250 →
  time_to_cross_second_platform = 20 →
  length_of_train = 110 →
  (270 / (360 / 20) = 15) := 
by
  intro h1 h2 h3 h4
  sorry

end time_to_cross_first_platform_l2007_200766


namespace geometric_sequence_first_term_l2007_200740

variable (a y z : ℕ)
variable (r : ℕ)
variable (h₁ : 16 = a * r^2)
variable (h₂ : 128 = a * r^4)

theorem geometric_sequence_first_term 
  (h₃ : r = 2) : a = 4 :=
by
  sorry

end geometric_sequence_first_term_l2007_200740


namespace product_eq_sum_l2007_200795

variables {x y : ℝ}

theorem product_eq_sum (h : x * y = x + y) (h_ne : y ≠ 1) : x = y / (y - 1) :=
sorry

end product_eq_sum_l2007_200795


namespace fifteen_power_ab_l2007_200702

theorem fifteen_power_ab (a b : ℤ) (R S : ℝ) 
  (hR : R = 3^a) 
  (hS : S = 5^b) : 
  15^(a * b) = R^b * S^a :=
by sorry

end fifteen_power_ab_l2007_200702


namespace sum_x_coordinates_common_points_l2007_200798

-- Definition of the equivalence relation modulo 9
def equiv_mod (a b n : ℤ) : Prop := ∃ k : ℤ, a = b + n * k

-- Definitions of the given conditions
def graph1 (x y : ℤ) : Prop := equiv_mod y (3 * x + 6) 9
def graph2 (x y : ℤ) : Prop := equiv_mod y (7 * x + 3) 9

-- Definition of when two graphs intersect
def points_in_common (x y : ℤ) : Prop := graph1 x y ∧ graph2 x y

-- Proof that the sum of the x-coordinates of the points in common is 3
theorem sum_x_coordinates_common_points : 
  ∃ x y, points_in_common x y ∧ (x = 3) := 
sorry

end sum_x_coordinates_common_points_l2007_200798


namespace false_statement_D_l2007_200772

theorem false_statement_D :
  ¬ (∀ {α β : ℝ}, α = β → (true → true → true → α = β ↔ α = β)) :=
by
  sorry

end false_statement_D_l2007_200772


namespace warehouse_can_release_100kg_l2007_200776

theorem warehouse_can_release_100kg (a b c d : ℕ) : 
  24 * a + 23 * b + 17 * c + 16 * d = 100 → True :=
by
  sorry

end warehouse_can_release_100kg_l2007_200776


namespace sum_of_three_numbers_is_98_l2007_200771

variable (A B C : ℕ) (h_ratio1 : A = 2 * (B / 3)) (h_ratio2 : B = 30) (h_ratio3 : B = 5 * (C / 8))

theorem sum_of_three_numbers_is_98 : A + B + C = 98 := by
  sorry

end sum_of_three_numbers_is_98_l2007_200771


namespace note_relationship_l2007_200727

theorem note_relationship
  (x y z : ℕ) 
  (h1 : x + 5 * y + 10 * z = 480)
  (h2 : x + y + z = 90)
  (h3 : y = 2 * x)
  (h4 : z = 3 * x) : 
  x = 15 ∧ y = 30 ∧ z = 45 :=
by 
  sorry

end note_relationship_l2007_200727


namespace ratio_of_houses_second_to_first_day_l2007_200712

theorem ratio_of_houses_second_to_first_day 
    (houses_day1 : ℕ)
    (houses_day2 : ℕ)
    (sales_per_house : ℕ)
    (sold_pct_day2 : ℝ) 
    (total_sales_day1 : ℕ)
    (total_sales_day2 : ℝ) :
    houses_day1 = 20 →
    sales_per_house = 2 →
    sold_pct_day2 = 0.8 →
    total_sales_day1 = houses_day1 * sales_per_house →
    total_sales_day2 = sold_pct_day2 * houses_day2 * sales_per_house →
    total_sales_day1 = total_sales_day2 →
    (houses_day2 : ℝ) / houses_day1 = 5 / 4 :=
by
    intro h1 h2 h3 h4 h5 h6
    sorry

end ratio_of_houses_second_to_first_day_l2007_200712


namespace systems_on_second_street_l2007_200735

-- Definitions based on the conditions
def commission_per_system : ℕ := 25
def total_commission : ℕ := 175
def systems_on_first_street (S : ℕ) := S / 2
def systems_on_third_street : ℕ := 0
def systems_on_fourth_street : ℕ := 1

-- Question: How many security systems did Rodney sell on the second street?
theorem systems_on_second_street (S : ℕ) :
  S / 2 + S + 0 + 1 = total_commission / commission_per_system → S = 4 :=
by
  intros h
  sorry

end systems_on_second_street_l2007_200735


namespace minimum_prime_factorization_sum_l2007_200720

theorem minimum_prime_factorization_sum (x y a b c d : ℕ) (hx : x > 0) (hy : y > 0)
  (h : 5 * x^7 = 13 * y^17) (h_pf: x = a ^ c * b ^ d) :
  a + b + c + d = 33 :=
sorry

end minimum_prime_factorization_sum_l2007_200720


namespace water_added_l2007_200736

theorem water_added (capacity : ℝ) (percentage_initial : ℝ) (percentage_final : ℝ) :
  capacity = 120 →
  percentage_initial = 0.30 →
  percentage_final = 0.75 →
  ((percentage_final * capacity) - (percentage_initial * capacity)) = 54 :=
by intros
   sorry

end water_added_l2007_200736


namespace k_zero_only_solution_l2007_200788

noncomputable def polynomial_factorable (k : ℤ) : Prop :=
  ∃ (A B C D E F : ℤ), (A * D = 1) ∧ (B * E = 4) ∧ (A * E + B * D = k) ∧ (A * F + C * D = 1) ∧ (C * F = -k)

theorem k_zero_only_solution : ∀ k : ℤ, polynomial_factorable k ↔ k = 0 :=
by 
  sorry

end k_zero_only_solution_l2007_200788


namespace bridge_length_l2007_200717

theorem bridge_length
  (train_length : ℝ) (train_speed : ℝ) (time_taken : ℝ)
  (h_train_length : train_length = 280)
  (h_train_speed : train_speed = 18)
  (h_time_taken : time_taken = 20) : ∃ L : ℝ, L = 80 :=
by
  let distance_covered := train_speed * time_taken
  have h_distance_covered : distance_covered = 360 := by sorry
  let bridge_length := distance_covered - train_length
  have h_bridge_length : bridge_length = 80 := by sorry
  existsi bridge_length
  exact h_bridge_length

end bridge_length_l2007_200717


namespace cos_double_angle_l2007_200725

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 1 / 2) : Real.cos (2 * α) = 3 / 5 :=
by sorry

end cos_double_angle_l2007_200725


namespace arithmetic_seq_common_difference_l2007_200708

theorem arithmetic_seq_common_difference (a1 d : ℝ) (h1 : a1 + 2 * d = 10) (h2 : 4 * a1 + 6 * d = 36) : d = 2 :=
by
  sorry

end arithmetic_seq_common_difference_l2007_200708
