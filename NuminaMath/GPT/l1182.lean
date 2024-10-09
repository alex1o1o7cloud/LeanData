import Mathlib

namespace walk_fraction_correct_l1182_118212

def bus_fraction := 1/3
def automobile_fraction := 1/5
def bicycle_fraction := 1/8
def metro_fraction := 1/15

def total_transport_fraction := bus_fraction + automobile_fraction + bicycle_fraction + metro_fraction

def walk_fraction := 1 - total_transport_fraction

theorem walk_fraction_correct : walk_fraction = 11/40 := by
  sorry

end walk_fraction_correct_l1182_118212


namespace more_than_10_weights_missing_l1182_118254

/-- 
Given weights of 5, 24, and 43 grams with an equal number of each type
and that the total remaining mass is 606060...60 grams,
prove that more than 10 weights are missing.
-/
theorem more_than_10_weights_missing (total_mass : ℕ) (n : ℕ) (k : ℕ) 
  (total_mass_eq : total_mass = k * (5 + 24 + 43))
  (total_mass_mod : total_mass % 72 ≠ 0) :
  k < n - 10 :=
sorry

end more_than_10_weights_missing_l1182_118254


namespace sum_of_reciprocals_l1182_118221

variable (x y : ℝ)

theorem sum_of_reciprocals (h1 : x + y = 10) (h2 : x * y = 20) : 1 / x + 1 / y = 1 / 2 :=
by
  sorry

end sum_of_reciprocals_l1182_118221


namespace building_height_l1182_118252

theorem building_height (h : ℕ) 
  (shadow_building : ℕ) 
  (shadow_pole : ℕ) 
  (height_pole : ℕ) 
  (ratio_proportional : shadow_building * height_pole = shadow_pole * h) 
  (shadow_building_val : shadow_building = 63) 
  (shadow_pole_val : shadow_pole = 32) 
  (height_pole_val : height_pole = 28) : 
  h = 55 := 
by 
  sorry

end building_height_l1182_118252


namespace hoseok_more_than_minyoung_l1182_118240

-- Define the initial amounts and additional earnings
def initial_amount : ℕ := 1500000
def additional_min : ℕ := 320000
def additional_hos : ℕ := 490000

-- Define the new amounts
def new_amount_min : ℕ := initial_amount + additional_min
def new_amount_hos : ℕ := initial_amount + additional_hos

-- Define the proof problem: Hoseok's new amount - Minyoung's new amount = 170000
theorem hoseok_more_than_minyoung : (new_amount_hos - new_amount_min) = 170000 :=
by
  -- The proof is skipped.
  sorry

end hoseok_more_than_minyoung_l1182_118240


namespace euler_family_mean_age_l1182_118219

theorem euler_family_mean_age : 
  let girls_ages := [5, 5, 10, 15]
  let boys_ages := [8, 12, 16]
  let children_ages := girls_ages ++ boys_ages
  let total_sum := List.sum children_ages
  let number_of_children := List.length children_ages
  (total_sum : ℚ) / number_of_children = 10.14 := 
by
  sorry

end euler_family_mean_age_l1182_118219


namespace min_value_fraction_l1182_118267

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) = 2) : 
  ∃ x : ℝ, x = (8 * a + b) / (a * b) ∧ x = 9 :=
by
  sorry

end min_value_fraction_l1182_118267


namespace range_of_m_l1182_118213

noncomputable def p (m : ℝ) : Prop := ∀ x : ℝ, -m * x ^ 2 + 2 * x - m > 0
noncomputable def q (m : ℝ) : Prop := ∀ x > 0, (4 / x + x - m + 1) > 2

theorem range_of_m : 
  (∃ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m)) → (∃ (m : ℝ), -1 ≤ m ∧ m < 3) :=
by
  intros h
  sorry

end range_of_m_l1182_118213


namespace probability_same_color_l1182_118255

theorem probability_same_color :
  let red_marble_prob := (5 / 21) * (4 / 20) * (3 / 19)
  let white_marble_prob := (6 / 21) * (5 / 20) * (4 / 19)
  let blue_marble_prob := (7 / 21) * (6 / 20) * (5 / 19)
  let green_marble_prob := (3 / 21) * (2 / 20) * (1 / 19)
  red_marble_prob + white_marble_prob + blue_marble_prob + green_marble_prob = 66 / 1330 := by
  sorry

end probability_same_color_l1182_118255


namespace power_equivalence_l1182_118233

theorem power_equivalence (m : ℕ) : 16^6 = 4^m → m = 12 :=
by
  sorry

end power_equivalence_l1182_118233


namespace ball_hits_ground_time_l1182_118217

noncomputable def find_time_when_ball_hits_ground (a b c : ℝ) : ℝ :=
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)

theorem ball_hits_ground_time :
  find_time_when_ball_hits_ground (-16) 40 50 = (5 + 5 * Real.sqrt 3) / 4 :=
by
  sorry

end ball_hits_ground_time_l1182_118217


namespace peter_reads_one_book_18_hours_l1182_118250

-- Definitions of conditions given in the problem
variables (P : ℕ)

-- Condition: Peter can read three times as fast as Kristin
def reads_three_times_as_fast (P : ℕ) : Prop :=
  ∀ (K : ℕ), K = 3 * P

-- Condition: Kristin reads half of her 20 books in 540 hours
def half_books_in_540_hours (K : ℕ) : Prop :=
  K = 54

-- Theorem stating the main proof problem: proving P equals 18 hours
theorem peter_reads_one_book_18_hours
  (H1 : reads_three_times_as_fast P)
  (H2 : half_books_in_540_hours (3 * P)) :
  P = 18 :=
sorry

end peter_reads_one_book_18_hours_l1182_118250


namespace farm_problem_l1182_118292

theorem farm_problem
    (initial_cows : ℕ := 12)
    (initial_pigs : ℕ := 34)
    (remaining_animals : ℕ := 30)
    (C : ℕ)
    (P : ℕ)
    (h1 : P = 3 * C)
    (h2 : initial_cows - C + (initial_pigs - P) = remaining_animals) :
    C = 4 :=
by
  sorry

end farm_problem_l1182_118292


namespace B_work_rate_l1182_118294

theorem B_work_rate :
  let A := (1 : ℝ) / 8
  let C := (1 : ℝ) / 4.8
  (A + B + C = 1 / 2) → (B = 1 / 6) :=
by
  intro h
  let A : ℝ := 1 / 8
  let C : ℝ := 1 / 4.8
  let B : ℝ := 1 / 6
  sorry

end B_work_rate_l1182_118294


namespace dividend_is_217_l1182_118259

-- Given conditions
def r : ℕ := 1
def q : ℕ := 54
def d : ℕ := 4

-- Define the problem as a theorem in Lean 4
theorem dividend_is_217 : (d * q) + r = 217 := by
  -- proof is omitted
  sorry

end dividend_is_217_l1182_118259


namespace transformation_g_from_f_l1182_118271

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (8 * x + 3 * Real.pi / 2)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem transformation_g_from_f :
  (∀ x, g x = f (x + Real.pi / 4) * 2) ∨ (∀ x, g x = f (x - Real.pi / 4) * 2) := 
by
  sorry

end transformation_g_from_f_l1182_118271


namespace tom_paid_correct_amount_l1182_118298

def quantity_of_apples : ℕ := 8
def rate_per_kg_apples : ℕ := 70
def quantity_of_mangoes : ℕ := 9
def rate_per_kg_mangoes : ℕ := 45

def cost_of_apples : ℕ := quantity_of_apples * rate_per_kg_apples
def cost_of_mangoes : ℕ := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid : ℕ := cost_of_apples + cost_of_mangoes

theorem tom_paid_correct_amount :
  total_amount_paid = 965 :=
sorry

end tom_paid_correct_amount_l1182_118298


namespace min_distance_origin_to_line_l1182_118230

noncomputable def distance_from_origin_to_line(A B C : ℝ) : ℝ :=
  let d := |A * 0 + B * 0 + C| / (Real.sqrt (A^2 + B^2))
  d

theorem min_distance_origin_to_line : distance_from_origin_to_line 1 1 (-4) = 2 * Real.sqrt 2 := by 
  sorry

end min_distance_origin_to_line_l1182_118230


namespace value_of_a_plus_b_minus_c_l1182_118289

def a : ℤ := 1 -- smallest positive integer
def b : ℤ := 0 -- number with the smallest absolute value
def c : ℤ := -1 -- largest negative integer

theorem value_of_a_plus_b_minus_c : a + b - c = 2 := by
  -- skipping the proof
  sorry

end value_of_a_plus_b_minus_c_l1182_118289


namespace melanie_dimes_final_l1182_118244

-- Define a type representing the initial state of Melanie's dimes
variable {initial_dimes : ℕ} (h_initial : initial_dimes = 7)

-- Define a function representing the result after attempting to give away dimes
def remaining_dimes_after_giving (initial_dimes : ℕ) (given_dimes : ℕ) : ℕ :=
  if given_dimes <= initial_dimes then initial_dimes - given_dimes else initial_dimes

-- State the problem
theorem melanie_dimes_final (h_initial : initial_dimes = 7) (given_dimes_dad : ℕ) (h_given_dad : given_dimes_dad = 8) (received_dimes_mom : ℕ) (h_received_mom : received_dimes_mom = 4) :
  remaining_dimes_after_giving initial_dimes given_dimes_dad + received_dimes_mom = 11 :=
by
  sorry

end melanie_dimes_final_l1182_118244


namespace small_cone_altitude_l1182_118237

noncomputable def frustum_height : ℝ := 18
noncomputable def lower_base_area : ℝ := 400 * Real.pi
noncomputable def upper_base_area : ℝ := 100 * Real.pi

theorem small_cone_altitude (h_frustum : frustum_height = 18) 
    (A_lower : lower_base_area = 400 * Real.pi) 
    (A_upper : upper_base_area = 100 * Real.pi) : 
    ∃ (h_small_cone : ℝ), h_small_cone = 18 := 
by
  sorry

end small_cone_altitude_l1182_118237


namespace perfect_square_trinomial_m_l1182_118204

theorem perfect_square_trinomial_m (m : ℤ) :
  ∀ y : ℤ, ∃ a : ℤ, (y^2 - m * y + 1 = (y + a) ^ 2) ∨ (y^2 - m * y + 1 = (y - a) ^ 2) → (m = 2 ∨ m = -2) :=
by 
  sorry

end perfect_square_trinomial_m_l1182_118204


namespace neg_q_is_true_l1182_118284

variable (p q : Prop)

theorem neg_q_is_true (hp : p) (hq : ¬ q) : ¬ q :=
by
  exact hq

end neg_q_is_true_l1182_118284


namespace problem_statement_l1182_118295

open Set

noncomputable def U := ℝ

def A : Set ℝ := { x | 0 < 2 * x + 4 ∧ 2 * x + 4 < 10 }
def B : Set ℝ := { x | x < -4 ∨ x > 2 }
def C (a : ℝ) (h : a < 0) : Set ℝ := { x | x^2 - 4 * a * x + 3 * a^2 < 0 }

theorem problem_statement (a : ℝ) (ha : a < 0) :
    A ∪ B = { x | x < -4 ∨ x > -2 } ∧
    compl (A ∪ B) ⊆ C a ha → -2 < a ∧ a < -4 / 3 :=
sorry

end problem_statement_l1182_118295


namespace comparison_abc_l1182_118290

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l1182_118290


namespace accounting_majors_count_l1182_118299

theorem accounting_majors_count (p q r s t u : ℕ) 
  (h_eq : p * q * r * s * t * u = 51030)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) : 
  p = 2 :=
sorry

end accounting_majors_count_l1182_118299


namespace football_team_total_players_l1182_118274

theorem football_team_total_players (P : ℕ) (throwers : ℕ) (left_handed : ℕ) (right_handed : ℕ) :
  throwers = 49 →
  right_handed = 63 →
  left_handed = (1/3) * (P - 49) →
  (P - 49) - left_handed = (2/3) * (P - 49) →
  70 = P :=
by
  intros h_throwers h_right_handed h_left_handed h_remaining
  sorry

end football_team_total_players_l1182_118274


namespace sin_double_angle_l1182_118286

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = -3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end sin_double_angle_l1182_118286


namespace rational_root_of_factors_l1182_118272

theorem rational_root_of_factors (p : ℕ) (a : ℚ) (hprime : Nat.Prime p) 
  (f : Polynomial ℚ) (hf : f = Polynomial.X ^ p - Polynomial.C a)
  (hfactors : ∃ g h : Polynomial ℚ, f = g * h ∧ 1 ≤ g.degree ∧ 1 ≤ h.degree) : 
  ∃ r : ℚ, Polynomial.eval r f = 0 :=
sorry

end rational_root_of_factors_l1182_118272


namespace systematic_sampling_methods_l1182_118249

-- Definitions for sampling methods ①, ②, ④
def sampling_method_1 : Prop :=
  ∀ (l : ℕ), (l ≤ 15 ∧ l + 5 ≤ 15 ∧ l + 10 ≤ 15 ∨
              l ≤ 15 ∧ l + 5 ≤ 20 ∧ l + 10 ≤ 20) → True

def sampling_method_2 : Prop :=
  ∀ (t : ℕ), (t % 5 = 0) → True

def sampling_method_3 : Prop :=
  ∀ (n : ℕ), (n > 0) → True

def sampling_method_4 : Prop :=
  ∀ (row : ℕ) (seat : ℕ), (seat = 12) → True

-- Equivalence Proof Statement
theorem systematic_sampling_methods :
  sampling_method_1 ∧ sampling_method_2 ∧ sampling_method_4 :=
by sorry

end systematic_sampling_methods_l1182_118249


namespace polygonal_chain_segments_l1182_118291

theorem polygonal_chain_segments (n : ℕ) :
  (∃ (S : Type) (chain : S → Prop), (∃ (closed_non_self_intersecting : S → Prop), 
  (∀ s : S, chain s → closed_non_self_intersecting s) ∧
  ∀ line_segment : S, chain line_segment → 
  (∃ other_segment : S, chain other_segment ∧ line_segment ≠ other_segment))) ↔ 
  (∃ k : ℕ, (n = 2 * k ∧ 5 ≤ k) ∨ (n = 2 * k + 1 ∧ 7 ≤ k)) :=
by sorry

end polygonal_chain_segments_l1182_118291


namespace a3_eq_5_l1182_118205

-- Define the geometric sequence and its properties
variables {a : ℕ → ℝ} {q : ℝ}

-- Assumptions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a 1 * (q ^ n)
axiom a1_pos : a 1 > 0
axiom a2a4_eq_25 : a 2 * a 4 = 25
axiom geom : geom_seq a q

-- Statement to prove
theorem a3_eq_5 : a 3 = 5 :=
by sorry

end a3_eq_5_l1182_118205


namespace profitability_when_x_gt_94_daily_profit_when_x_le_94_max_profit_occurs_at_84_l1182_118207

theorem profitability_when_x_gt_94 (A : ℕ) (x : ℕ) (hx : x > 94) : 
  1/3 * x * A - (2/3 * x * (A / 2)) = 0 := 
sorry

theorem daily_profit_when_x_le_94 (A : ℕ) (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 94) : 
  ∃ T : ℕ, T = (x - 3 * x / (2 * (96 - x))) * A := 
sorry

theorem max_profit_occurs_at_84 (A : ℕ) : 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 94 ∧ 
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 94 → 
    (y - 3 * y / (2 * (96 - y))) * A ≤ (84 - 3 * 84 / (2 * (96 - 84))) * A) := 
sorry

end profitability_when_x_gt_94_daily_profit_when_x_le_94_max_profit_occurs_at_84_l1182_118207


namespace smallest_n_exists_l1182_118232

theorem smallest_n_exists :
  ∃ n : ℕ, n > 0 ∧ 3^(3^(n + 1)) ≥ 3001 :=
by
  sorry

end smallest_n_exists_l1182_118232


namespace both_pipes_opened_together_for_2_minutes_l1182_118234

noncomputable def fill_time (t : ℝ) : Prop :=
  let rate_p := 1 / 12
  let rate_q := 1 / 15
  let combined_rate := rate_p + rate_q
  let work_done_by_p_q := combined_rate * t
  let work_done_by_q := rate_q * 10.5
  work_done_by_p_q + work_done_by_q = 1

theorem both_pipes_opened_together_for_2_minutes : ∃ t : ℝ, fill_time t ∧ t = 2 :=
by
  use 2
  unfold fill_time
  sorry

end both_pipes_opened_together_for_2_minutes_l1182_118234


namespace largest_integral_x_l1182_118269

theorem largest_integral_x (x : ℤ) (h1 : 1/4 < (x:ℝ)/6) (h2 : (x:ℝ)/6 < 7/9) : x ≤ 4 :=
by
  -- This is where the proof would go
  sorry

end largest_integral_x_l1182_118269


namespace find_f_minus_2_l1182_118265

namespace MathProof

def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 5

theorem find_f_minus_2 (a b c : ℝ) (h : f a b c 2 = 3) : f a b c (-2) = -13 := 
by
  sorry

end MathProof

end find_f_minus_2_l1182_118265


namespace team_formation_l1182_118275

def nat1 : ℕ := 7  -- Number of natives who know mathematics and physics
def nat2 : ℕ := 6  -- Number of natives who know physics and chemistry
def nat3 : ℕ := 3  -- Number of natives who know chemistry and mathematics
def nat4 : ℕ := 4  -- Number of natives who know physics and biology

def totalWaysToFormTeam (n1 n2 n3 n4 : ℕ) : ℕ := (n1 + n2 + n3 + n4).choose 3
def waysFromSameGroup (n : ℕ) : ℕ := n.choose 3

def waysFromAllGroups (n1 n2 n3 n4 : ℕ) : ℕ := (waysFromSameGroup n1) + (waysFromSameGroup n2) + (waysFromSameGroup n3) + (waysFromSameGroup n4)

theorem team_formation : totalWaysToFormTeam nat1 nat2 nat3 nat4 - waysFromAllGroups nat1 nat2 nat3 nat4 = 1080 := 
by
    sorry

end team_formation_l1182_118275


namespace line_parabola_intersection_l1182_118238

theorem line_parabola_intersection (k : ℝ) : 
    (∀ l p: ℝ → ℝ, l = (fun x => k * x + 1) ∧ p = (fun x => 4 * x ^ 2) → 
        (∃ x, l x = p x) ∧ (∀ x1 x2, l x1 = p x1 ∧ l x2 = p x2 → x1 = x2) 
    ↔ k = 0 ∨ k = 1) :=
sorry

end line_parabola_intersection_l1182_118238


namespace inequality_a3_b3_c3_l1182_118279

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := 
by 
  sorry

end inequality_a3_b3_c3_l1182_118279


namespace f_periodic_analytic_expression_f_distinct_real_roots_l1182_118202

noncomputable def f (x : ℝ) (k : ℤ) : ℝ := (x - 2 * k)^2

def I_k (k : ℤ) : Set ℝ := { x | 2 * k - 1 < x ∧ x ≤ 2 * k + 1 }

def M_k (k : ℕ) : Set ℝ := { a | 0 < a ∧ a ≤ 1 / (2 * ↑k + 1) }

theorem f_periodic (x : ℝ) (k : ℤ) : f x k = f (x - 2 * k) 0 := by
  sorry

theorem analytic_expression_f (x : ℝ) (k : ℤ) (hx : x ∈ I_k k) : f x k = (x - 2 * k)^2 := by
  sorry

theorem distinct_real_roots (k : ℕ) (a : ℝ) (h : a ∈ M_k k) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ I_k k ∧ x2 ∈ I_k k ∧ f x1 k = a * x1 ∧ f x2 k = a * x2 := by
  sorry

end f_periodic_analytic_expression_f_distinct_real_roots_l1182_118202


namespace brown_eyed_brunettes_count_l1182_118228

/--
There are 50 girls in a group. Each girl is either blonde or brunette and either blue-eyed or brown-eyed.
14 girls are blue-eyed blondes. 31 girls are brunettes. 18 girls are brown-eyed.
Prove that the number of brown-eyed brunettes is equal to 13.
-/
theorem brown_eyed_brunettes_count
  (total_girls : ℕ)
  (blue_eyed_blondes : ℕ)
  (total_brunettes : ℕ)
  (total_brown_eyed : ℕ)
  (total_girls_eq : total_girls = 50)
  (blue_eyed_blondes_eq : blue_eyed_blondes = 14)
  (total_brunettes_eq : total_brunettes = 31)
  (total_brown_eyed_eq : total_brown_eyed = 18) :
  ∃ (brown_eyed_brunettes : ℕ), brown_eyed_brunettes = 13 :=
by sorry

end brown_eyed_brunettes_count_l1182_118228


namespace sequence_formula_l1182_118216

open Nat

def a : ℕ → ℤ
| 0     => 0  -- Defining a(0) though not used
| 1     => 1
| (n+2) => 3 * a (n+1) + 2^(n+2)

theorem sequence_formula (n : ℕ) (hn : n ≥ 1) :
  a n = 5 * 3^(n-1) - 2^(n+1) :=
by
  sorry

end sequence_formula_l1182_118216


namespace total_pennies_l1182_118210

variable (C J : ℕ)

def cassandra_pennies : ℕ := 5000
def james_pennies (C : ℕ) : ℕ := C - 276

theorem total_pennies (hC : C = cassandra_pennies) (hJ : J = james_pennies C) :
  C + J = 9724 :=
by
  sorry

end total_pennies_l1182_118210


namespace geom_sequence_arith_ratio_l1182_118260

variable (a : ℕ → ℝ) (q : ℝ)
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_arith : 3 * a 0 + 2 * a 1 = 2 * (1/2) * a 2)

theorem geom_sequence_arith_ratio (ha : 3 * a 0 + 2 * a 1 = a 2) :
    (a 8 + a 9) / (a 6 + a 7) = 9 := sorry

end geom_sequence_arith_ratio_l1182_118260


namespace john_spent_half_on_fruits_and_vegetables_l1182_118200

theorem john_spent_half_on_fruits_and_vegetables (M : ℝ) (F : ℝ) 
  (spent_on_meat : ℝ) (spent_on_bakery : ℝ) (spent_on_candy : ℝ) :
  (M = 120) → 
  (spent_on_meat = (1 / 3) * M) → 
  (spent_on_bakery = (1 / 10) * M) → 
  (spent_on_candy = 8) → 
  (F * M + spent_on_meat + spent_on_bakery + spent_on_candy = M) → 
  (F = 1 / 2) := 
  by 
    sorry

end john_spent_half_on_fruits_and_vegetables_l1182_118200


namespace deadlift_weight_loss_is_200_l1182_118214

def initial_squat : ℕ := 700
def initial_bench : ℕ := 400
def initial_deadlift : ℕ := 800
def lost_squat_percent : ℕ := 30
def new_total : ℕ := 1490

theorem deadlift_weight_loss_is_200 : initial_deadlift - (new_total - ((initial_squat * (100 - lost_squat_percent)) / 100 + initial_bench)) = 200 :=
by
  sorry

end deadlift_weight_loss_is_200_l1182_118214


namespace sum_of_consecutive_integers_l1182_118220

theorem sum_of_consecutive_integers (x y : ℕ) (h1 : y = x + 1) (h2 : x * y = 812) : x + y = 57 :=
by
  -- proof skipped
  sorry

end sum_of_consecutive_integers_l1182_118220


namespace find_dividend_l1182_118261

theorem find_dividend (partial_product : ℕ) (remainder : ℕ) (divisor quotient : ℕ) :
  partial_product = 2015 → 
  remainder = 0 →
  divisor = 105 → 
  quotient = 197 → 
  divisor * quotient + remainder = partial_product → 
  partial_product * 10 = 20685 :=
by {
  -- Proof skipped
  sorry
}

end find_dividend_l1182_118261


namespace valid_license_plates_l1182_118247

-- Define the number of vowels and the total alphabet letters.
def num_vowels : ℕ := 5
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the total number of valid license plates in Eldoria.
theorem valid_license_plates : num_vowels * num_letters * num_digits^3 = 130000 := by
  sorry

end valid_license_plates_l1182_118247


namespace computation_l1182_118206

theorem computation :
  ( ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * 
    ( (7^3 - 1) / (7^3 + 1) ) * ( (8^3 - 1) / (8^3 + 1) ) 
  ) = (73 / 312) :=
by
  sorry

end computation_l1182_118206


namespace monotone_increasing_interval_for_shifted_function_l1182_118236

variable (f : ℝ → ℝ)

-- Given definition: f(x+1) is an even function
def even_function : Prop :=
  ∀ x, f (x+1) = f (-(x+1))

-- Given condition: f(x+1) is monotonically decreasing on [0, +∞)
def monotone_decreasing_on_nonneg : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f (x+1) ≥ f (y+1)

-- Theorem to prove: the interval on which f(x-1) is monotonically increasing is (-∞, 2]
theorem monotone_increasing_interval_for_shifted_function
  (h_even : even_function f)
  (h_mono_dec : monotone_decreasing_on_nonneg f) :
  ∀ x y, x ≤ 2 → y ≤ 2 → x ≤ y → f (x-1) ≤ f (y-1) :=
by
  sorry

end monotone_increasing_interval_for_shifted_function_l1182_118236


namespace max_abs_ax_plus_b_l1182_118246

theorem max_abs_ax_plus_b (a b c : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, |x| ≤ 1 → |a * x + b| ≤ 2 :=
by
  sorry

end max_abs_ax_plus_b_l1182_118246


namespace band_total_l1182_118256

theorem band_total (flutes_total clarinets_total trumpets_total pianists_total : ℕ)
                   (flutes_pct clarinets_pct trumpets_pct pianists_pct : ℚ)
                   (h_flutes : flutes_total = 20)
                   (h_clarinets : clarinets_total = 30)
                   (h_trumpets : trumpets_total = 60)
                   (h_pianists : pianists_total = 20)
                   (h_flutes_pct : flutes_pct = 0.8)
                   (h_clarinets_pct : clarinets_pct = 0.5)
                   (h_trumpets_pct : trumpets_pct = 1/3)
                   (h_pianists_pct : pianists_pct = 1/10) :
  flutes_total * flutes_pct + clarinets_total * clarinets_pct + 
  trumpets_total * trumpets_pct + pianists_total * pianists_pct = 53 := by
  sorry

end band_total_l1182_118256


namespace equilateral_triangle_lines_l1182_118208

-- Define the properties of an equilateral triangle
structure EquilateralTriangle :=
(sides_length : ℝ) -- All sides are of equal length
(angle : ℝ := 60)  -- All internal angles are 60 degrees

-- Define the concept that altitudes, medians, and angle bisectors coincide
structure CoincidingLines (T : EquilateralTriangle) :=
(altitude : T.angle = 60)
(median : T.angle = 60)
(angle_bisector : T.angle = 60)

-- Define a statement that proves the number of distinct lines in the equilateral triangle
theorem equilateral_triangle_lines (T : EquilateralTriangle) (L : CoincidingLines T) :  
  -- The total number of distinct lines consisting of altitudes, medians, and angle bisectors
  (3 = 3) :=
by
  sorry

end equilateral_triangle_lines_l1182_118208


namespace mango_price_reduction_l1182_118226

theorem mango_price_reduction (P R : ℝ) (M : ℕ)
  (hP_orig : 110 * P = 366.67)
  (hM : M * P = 360)
  (hR_red : (M + 12) * R = 360) :
  ((P - R) / P) * 100 = 10 :=
by sorry

end mango_price_reduction_l1182_118226


namespace incorrect_expression_l1182_118264

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (1 - 3*x > 1 - 3*y) :=
sorry

end incorrect_expression_l1182_118264


namespace smallest_positive_a_l1182_118263

/-- Define a function f satisfying the given conditions. -/
noncomputable def f : ℝ → ℝ :=
  sorry -- we'll define it later according to the problem

axiom condition1 : ∀ x > 0, f (2 * x) = 2 * f x

axiom condition2 : ∀ x, 1 < x ∧ x < 2 → f x = 2 - x

theorem smallest_positive_a :
  (∃ a > 0, f a = f 2020) ∧ ∀ b > 0, (f b = f 2020 → b ≥ 36) :=
  sorry

end smallest_positive_a_l1182_118263


namespace sequence_gcd_is_index_l1182_118282

theorem sequence_gcd_is_index (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ i : ℕ, a i = i :=
by
  sorry

end sequence_gcd_is_index_l1182_118282


namespace find_x_l1182_118225

theorem find_x (x : ℝ) (A1 A2 : ℝ) (P1 P2 : ℝ)
    (hA1 : A1 = x^2 + 4*x + 4)
    (hA2 : A2 = 4*x^2 - 12*x + 9)
    (hP : P1 + P2 = 32)
    (hP1 : P1 = 4 * (x + 2))
    (hP2 : P2 = 4 * (2*x - 3)) :
    x = 3 :=
by
  sorry

end find_x_l1182_118225


namespace euler_line_of_isosceles_triangle_l1182_118231

theorem euler_line_of_isosceles_triangle (A B : ℝ × ℝ) (hA : A = (2,0)) (hB : B = (0,4)) (C : ℝ × ℝ) (hC1 : dist A C = dist B C) :
  ∃ a b c : ℝ, a * (C.1 - 2) + b * (C.2 - 0) + c = 0 ∧ x - 2 * y + 3 = 0 :=
by
  sorry

end euler_line_of_isosceles_triangle_l1182_118231


namespace inequality_solution_l1182_118273

theorem inequality_solution (x : ℝ) : 
  -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 3 ≤ x ∧ x < 4 → 
  (x + 6 ≥ 0) ∧ (x + 1 > 0) ∧ (5 - x > 0) ∧ (x ≠ 0) ∧ (x ≠ 1) ∧ (x ≠ 4) ∧
  ( (x - 3) / ((x - 1) * (4 - x)) ≥ 0 ) :=
sorry

end inequality_solution_l1182_118273


namespace no_valid_arithmetic_operation_l1182_118209

-- Definition for arithmetic operations
inductive Operation
| div : Operation
| mul : Operation
| add : Operation
| sub : Operation

open Operation

-- Given conditions
def equation (op : Operation) : Prop :=
  match op with
  | div => (8 / 2) + 5 - (3 - 2) = 12
  | mul => (8 * 2) + 5 - (3 - 2) = 12
  | add => (8 + 2) + 5 - (3 - 2) = 12
  | sub => (8 - 2) + 5 - (3 - 2) = 12

-- Statement to prove
theorem no_valid_arithmetic_operation : ∀ op : Operation, ¬ equation op := by
  sorry

end no_valid_arithmetic_operation_l1182_118209


namespace shift_right_graph_l1182_118242

theorem shift_right_graph (x : ℝ) :
  (3 : ℝ)^(x+1) = (3 : ℝ)^((x+1) - 1) :=
by 
  -- Here we prove that shifting the graph of y = 3^(x+1) to right by 1 unit 
  -- gives the graph of y = 3^x
  sorry

end shift_right_graph_l1182_118242


namespace addends_are_negative_l1182_118288

theorem addends_are_negative (a b : ℤ) (h1 : a + b < a) (h2 : a + b < b) : a < 0 ∧ b < 0 := 
sorry

end addends_are_negative_l1182_118288


namespace guy_has_sixty_cents_l1182_118262

-- Definitions for the problem conditions
def lance_has (lance_cents : ℕ) : Prop := lance_cents = 70
def margaret_has (margaret_cents : ℕ) : Prop := margaret_cents = 75
def bill_has (bill_cents : ℕ) : Prop := bill_cents = 60
def total_has (total_cents : ℕ) : Prop := total_cents = 265

-- Problem Statement in Lean format
theorem guy_has_sixty_cents (lance_cents margaret_cents bill_cents total_cents guy_cents : ℕ) 
    (h_lance : lance_has lance_cents)
    (h_margaret : margaret_has margaret_cents)
    (h_bill : bill_has bill_cents)
    (h_total : total_has total_cents) :
    guy_cents = total_cents - (lance_cents + margaret_cents + bill_cents) → guy_cents = 60 :=
by
  intros h
  simp [lance_has, margaret_has, bill_has, total_has] at *
  rw [h_lance, h_margaret, h_bill, h_total] at h
  exact h

end guy_has_sixty_cents_l1182_118262


namespace hillary_activities_l1182_118248

-- Define the conditions
def swims_every : ℕ := 6
def runs_every : ℕ := 4
def cycles_every : ℕ := 16

-- Define the theorem to prove
theorem hillary_activities : Nat.lcm (Nat.lcm swims_every runs_every) cycles_every = 48 :=
by
  -- Provide a placeholder for the proof
  sorry

end hillary_activities_l1182_118248


namespace shortest_distance_between_circles_is_zero_l1182_118287

open Real

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop :=
  x^2 - 12 * x + y^2 - 8 * y - 12 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop :=
  x^2 + 10 * x + y^2 - 10 * y + 34 = 0

-- Statement of the proof problem: 
-- Prove the shortest distance between the two circles defined by circle1 and circle2 is 0.
theorem shortest_distance_between_circles_is_zero :
    ∀ (x1 y1 x2 y2 : ℝ),
      circle1 x1 y1 →
      circle2 x2 y2 →
      0 = 0 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end shortest_distance_between_circles_is_zero_l1182_118287


namespace negative_comparison_l1182_118243

theorem negative_comparison : -2023 > -2024 :=
sorry

end negative_comparison_l1182_118243


namespace mr_lee_gain_l1182_118218

noncomputable def cost_price_1 (revenue : ℝ) (profit_percentage : ℝ) : ℝ :=
  revenue / (1 + profit_percentage)

noncomputable def cost_price_2 (revenue : ℝ) (loss_percentage : ℝ) : ℝ :=
  revenue / (1 - loss_percentage)

theorem mr_lee_gain
    (revenue : ℝ)
    (profit_percentage : ℝ)
    (loss_percentage : ℝ)
    (revenue_1 : ℝ := 1.44)
    (revenue_2 : ℝ := 1.44)
    (profit_percent : ℝ := 0.20)
    (loss_percent : ℝ := 0.10):
  let cost_1 := cost_price_1 revenue_1 profit_percent
  let cost_2 := cost_price_2 revenue_2 loss_percent
  let total_cost := cost_1 + cost_2
  let total_revenue := revenue_1 + revenue_2
  total_revenue - total_cost = 0.08 :=
by
  sorry

end mr_lee_gain_l1182_118218


namespace least_number_of_stamps_l1182_118251

theorem least_number_of_stamps (p q : ℕ) (h : 5 * p + 4 * q = 50) : p + q = 11 :=
sorry

end least_number_of_stamps_l1182_118251


namespace min_value_of_expression_l1182_118277

variable (a b : ℝ)

theorem min_value_of_expression (h : b ≠ 0) : 
  ∃ (a b : ℝ), (a^2 + b^2 + a / b + 1 / b^2) = Real.sqrt 3 :=
sorry

end min_value_of_expression_l1182_118277


namespace min_value_xy_l1182_118278

theorem min_value_xy {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : (2 / x) + (8 / y) = 1) : x * y ≥ 64 :=
sorry

end min_value_xy_l1182_118278


namespace find_k_l1182_118229

theorem find_k 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (hSn : ∀ n, S n = -2 + 2 * (1 / 3) ^ n) 
  (h_geom : ∀ n, a (n + 1) = a n * a 2 / a 1) :
  k = -2 :=
sorry

end find_k_l1182_118229


namespace dogs_in_pet_shop_l1182_118296

variable (D C B : ℕ) (x : ℕ)

theorem dogs_in_pet_shop
  (h1 : D = 3 * x)
  (h2 : C = 7 * x)
  (h3 : B = 12 * x)
  (h4 : D + B = 375) :
  D = 75 :=
by
  sorry

end dogs_in_pet_shop_l1182_118296


namespace total_capsules_in_july_l1182_118281

theorem total_capsules_in_july : 
  let mondays := 4
  let tuesdays := 5
  let wednesdays := 5
  let thursdays := 4
  let fridays := 4
  let saturdays := 4
  let sundays := 5

  let capsules_monday := mondays * 2
  let capsules_tuesday := tuesdays * 3
  let capsules_wednesday := wednesdays * 2
  let capsules_thursday := thursdays * 3
  let capsules_friday := fridays * 2
  let capsules_saturday := saturdays * 4
  let capsules_sunday := sundays * 4

  let total_capsules := capsules_monday + capsules_tuesday + capsules_wednesday + capsules_thursday + capsules_friday + capsules_saturday + capsules_sunday

  let missed_capsules_tuesday := 3
  let missed_capsules_sunday := 4

  let total_missed_capsules := missed_capsules_tuesday + missed_capsules_sunday

  let total_consumed_capsules := total_capsules - total_missed_capsules
  total_consumed_capsules = 82 := 
by
  -- Details omitted, proof goes here
  sorry

end total_capsules_in_july_l1182_118281


namespace sum_of_real_numbers_l1182_118224

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l1182_118224


namespace least_possible_value_of_z_minus_x_l1182_118215

theorem least_possible_value_of_z_minus_x 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) : 
  z - x = 9 :=
sorry

end least_possible_value_of_z_minus_x_l1182_118215


namespace cycle_final_selling_price_l1182_118283

-- Lean 4 statement capturing the problem definition and final selling price
theorem cycle_final_selling_price (original_price : ℝ) (initial_discount_rate : ℝ) 
  (loss_rate : ℝ) (exchange_discount_rate : ℝ) (final_price : ℝ) :
  original_price = 1400 →
  initial_discount_rate = 0.05 →
  loss_rate = 0.25 →
  exchange_discount_rate = 0.10 →
  final_price = 
    (original_price * (1 - initial_discount_rate) * (1 - loss_rate) * (1 - exchange_discount_rate)) →
  final_price = 897.75 :=
by
  sorry

end cycle_final_selling_price_l1182_118283


namespace chameleons_cannot_all_turn_to_single_color_l1182_118211

theorem chameleons_cannot_all_turn_to_single_color
  (W : ℕ) (B : ℕ)
  (hW : W = 20)
  (hB : B = 25)
  (h_interaction: ∀ t : ℕ, ∃ W' B' : ℕ,
    W' + B' = W + B ∧
    (W - B) % 3 = (W' - B') % 3) :
  ∀ t : ℕ, (W - B) % 3 ≠ 0 :=
by
  sorry

end chameleons_cannot_all_turn_to_single_color_l1182_118211


namespace jelly_bean_probability_l1182_118257

theorem jelly_bean_probability :
  ∀ (P_red P_orange P_green P_yellow : ℝ),
  P_red = 0.1 →
  P_orange = 0.4 →
  P_green = 0.2 →
  P_red + P_orange + P_green + P_yellow = 1 →
  P_yellow = 0.3 :=
by
  intros P_red P_orange P_green P_yellow h_red h_orange h_green h_sum
  sorry

end jelly_bean_probability_l1182_118257


namespace Valleyball_Soccer_League_members_l1182_118227

theorem Valleyball_Soccer_League_members (cost_socks cost_tshirt total_expenditure cost_per_member: ℕ) (h1 : cost_socks = 6) (h2 : cost_tshirt = cost_socks + 8) (h3 : total_expenditure = 3740) (h4 : cost_per_member = cost_socks + 2 * cost_tshirt) : 
  total_expenditure = 3740 → cost_per_member = 34 → total_expenditure / cost_per_member = 110 :=
sorry

end Valleyball_Soccer_League_members_l1182_118227


namespace rectangle_length_fraction_of_circle_radius_l1182_118297

noncomputable def square_side (area : ℕ) : ℕ :=
  Nat.sqrt area

noncomputable def rectangle_length (breadth area : ℕ) : ℕ :=
  area / breadth

theorem rectangle_length_fraction_of_circle_radius
  (square_area : ℕ)
  (rectangle_breadth : ℕ)
  (rectangle_area : ℕ)
  (side := square_side square_area)
  (radius := side)
  (length := rectangle_length rectangle_breadth rectangle_area) :
  square_area = 4761 →
  rectangle_breadth = 13 →
  rectangle_area = 598 →
  length / radius = 2 / 3 :=
by
  -- Proof steps go here
  sorry

end rectangle_length_fraction_of_circle_radius_l1182_118297


namespace school_club_profit_l1182_118285

theorem school_club_profit :
  let pencils := 1200
  let buy_rate := 4 / 3 -- pencils per dollar
  let sell_rate := 5 / 4 -- pencils per dollar
  let cost_per_pencil := 3 / 4 -- dollars per pencil
  let sell_per_pencil := 4 / 5 -- dollars per pencil
  let cost := pencils * cost_per_pencil
  let revenue := pencils * sell_per_pencil
  let profit := revenue - cost
  profit = 60 := 
by
  sorry

end school_club_profit_l1182_118285


namespace find_m_and_p_l1182_118239

-- Definition of a point being on the parabola y^2 = 2px
def on_parabola (m : ℝ) (p : ℝ) : Prop :=
  (-3)^2 = 2 * p * m

-- Definition of the distance from the point (m, -3) to the focus being 5
def distance_to_focus (m : ℝ) (p : ℝ) : Prop :=
  m + p / 2 = 5

theorem find_m_and_p (m p : ℝ) (hp : 0 < p) : 
  (on_parabola m p) ∧ (distance_to_focus m p) → 
  (m = 1 / 2 ∧ p = 9) ∨ (m = 9 / 2 ∧ p = 1) :=
by
  sorry

end find_m_and_p_l1182_118239


namespace circumference_circle_l1182_118235

theorem circumference_circle {d r : ℝ} (h1 : ∀ (d r : ℝ), d = 2 * r) : 
  ∃ C : ℝ, C = π * d ∨ C = 2 * π * r :=
by {
  sorry
}

end circumference_circle_l1182_118235


namespace ages_of_Xs_sons_l1182_118201

def ages_problem (x y : ℕ) : Prop :=
x ≠ y ∧ x ≤ 10 ∧ y ≤ 10 ∧
∀ u v : ℕ, u * v = x * y → u ≤ 10 ∧ v ≤ 10 → (u, v) = (x, y) ∨ (u, v) = (y, x) ∨
(∀ z w : ℕ, z / w = x / y → z = x ∧ w = y ∨ z = y ∧ w = x → u ≠ z ∧ v ≠ w) →
(∀ a b : ℕ, a - b = (x - y) ∨ b - a = (y - x) → (x, y) = (a, b) ∨ (x, y) = (b, a))

theorem ages_of_Xs_sons : ages_problem 8 2 := 
by {
  sorry
}


end ages_of_Xs_sons_l1182_118201


namespace max_consecutive_sum_l1182_118222

theorem max_consecutive_sum (a N : ℤ) (h₀ : N > 0) (h₁ : N * (2 * a + N - 1) = 90) : N = 90 :=
by
  -- Proof to be provided
  sorry

end max_consecutive_sum_l1182_118222


namespace differentiable_implies_continuous_l1182_118276

-- Theorem: If a function f is differentiable at x0, then it is continuous at x0.
theorem differentiable_implies_continuous {f : ℝ → ℝ} {x₀ : ℝ} (h : DifferentiableAt ℝ f x₀) : 
  ContinuousAt f x₀ :=
sorry

end differentiable_implies_continuous_l1182_118276


namespace number_is_seven_point_five_l1182_118223

theorem number_is_seven_point_five (x : ℝ) (h : x^2 + 100 = (x - 20)^2) : x = 7.5 :=
by
  sorry

end number_is_seven_point_five_l1182_118223


namespace agent_007_encryption_l1182_118203

theorem agent_007_encryption : ∃ (m n : ℕ), (0.07 : ℝ) = (1 / m : ℝ) + (1 / n : ℝ) := 
sorry

end agent_007_encryption_l1182_118203


namespace calculate_jessie_points_l1182_118270

theorem calculate_jessie_points (total_points : ℕ) (some_players_points : ℕ) (players : ℕ) :
  total_points = 311 →
  some_players_points = 188 →
  players = 3 →
  (total_points - some_players_points) / players = 41 :=
by
  intros
  sorry

end calculate_jessie_points_l1182_118270


namespace congruent_triangles_have_equal_perimeters_and_areas_l1182_118241

-- Definitions based on the conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (A B C : ℝ) -- angles of the triangle

def congruent_triangles (Δ1 Δ2 : Triangle) : Prop :=
  Δ1.a = Δ2.a ∧ Δ1.b = Δ2.b ∧ Δ1.c = Δ2.c ∧
  Δ1.A = Δ2.A ∧ Δ1.B = Δ2.B ∧ Δ1.C = Δ2.C

-- perimeters and areas (assuming some function calc_perimeter and calc_area for simplicity)
def perimeter (Δ : Triangle) : ℝ := Δ.a + Δ.b + Δ.c
def area (Δ : Triangle) : ℝ := sorry -- implement area calculation, e.g., using Heron's formula

-- Statement to be proved
theorem congruent_triangles_have_equal_perimeters_and_areas (Δ1 Δ2 : Triangle) :
  congruent_triangles Δ1 Δ2 →
  perimeter Δ1 = perimeter Δ2 ∧ area Δ1 = area Δ2 :=
sorry

end congruent_triangles_have_equal_perimeters_and_areas_l1182_118241


namespace smallest_positive_integer_square_begins_with_1989_l1182_118266

theorem smallest_positive_integer_square_begins_with_1989 :
  ∃ (A : ℕ), (1989 * 10^0 ≤ A^2 ∧ A^2 < 1990 * 10^0) 
  ∨ (1989 * 10^1 ≤ A^2 ∧ A^2 < 1990 * 10^1) 
  ∨ (1989 * 10^2 ≤ A^2 ∧ A^2 < 1990 * 10^2)
  ∧ A = 446 :=
sorry

end smallest_positive_integer_square_begins_with_1989_l1182_118266


namespace abhay_speed_l1182_118253

theorem abhay_speed
    (A S : ℝ)
    (h1 : 30 / A = 30 / S + 2)
    (h2 : 30 / (2 * A) = 30 / S - 1) :
    A = 5 * Real.sqrt 6 :=
by
  sorry

end abhay_speed_l1182_118253


namespace student_marks_equals_125_l1182_118245

-- Define the maximum marks
def max_marks : ℕ := 500

-- Define the percentage required to pass
def pass_percentage : ℚ := 33 / 100

-- Define the marks required to pass
def pass_marks : ℚ := pass_percentage * max_marks

-- Define the marks by which the student failed
def fail_by_marks : ℕ := 40

-- Define the obtained marks by the student
def obtained_marks : ℚ := pass_marks - fail_by_marks

-- Prove that the obtained marks are 125
theorem student_marks_equals_125 : obtained_marks = 125 := by
  sorry

end student_marks_equals_125_l1182_118245


namespace Xiaoxi_has_largest_final_answer_l1182_118293

def Laura_final : ℕ := 8 - 2 * 3 + 3
def Navin_final : ℕ := (8 * 3) - 2 + 3
def Xiaoxi_final : ℕ := (8 - 2 + 3) * 3

theorem Xiaoxi_has_largest_final_answer : 
  Xiaoxi_final > Laura_final ∧ Xiaoxi_final > Navin_final :=
by
  unfold Laura_final Navin_final Xiaoxi_final
  -- Proof steps would go here, but we skip them as per instructions
  sorry

end Xiaoxi_has_largest_final_answer_l1182_118293


namespace muffins_in_morning_l1182_118280

variable (M : ℕ)

-- Conditions
def goal : ℕ := 20
def afternoon_sales : ℕ := 4
def additional_needed : ℕ := 4
def morning_sales (M : ℕ) : ℕ := M

-- Proof statement (no need to prove here, just state it)
theorem muffins_in_morning :
  morning_sales M + afternoon_sales + additional_needed = goal → M = 12 :=
sorry

end muffins_in_morning_l1182_118280


namespace initial_books_in_bin_l1182_118268

variable (X : ℕ)

theorem initial_books_in_bin (h1 : X - 3 + 10 = 11) : X = 4 :=
by
  sorry

end initial_books_in_bin_l1182_118268


namespace digit_place_value_ratio_l1182_118258

theorem digit_place_value_ratio : 
  let num := 43597.2468
  let digit5_place_value := 10    -- tens place
  let digit2_place_value := 0.1   -- tenths place
  digit5_place_value / digit2_place_value = 100 := 
by 
  sorry

end digit_place_value_ratio_l1182_118258
