import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Analysis.Calc.Interval
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.TriangleCyclic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Use

namespace average_of_distinct_s_values_l633_633666

theorem average_of_distinct_s_values : 
  (1 + 5 + 2 + 4 + 3 + 3 + 4 + 2 + 5 + 1) / 3 = 7.33 :=
by
  sorry

end average_of_distinct_s_values_l633_633666


namespace arithmetic_mean_of_q_and_r_l633_633439

theorem arithmetic_mean_of_q_and_r (p q r : ℝ) 
  (h₁: (p + q) / 2 = 10) 
  (h₂: r - p = 20) : 
  (q + r) / 2 = 20 :=
sorry

end arithmetic_mean_of_q_and_r_l633_633439


namespace draw_balls_count_l633_633512

theorem draw_balls_count : 
  let balls := list.range' 1 20 in 
  ∃ (draws : list ℕ), 
    draws.length = 4 ∧ 
    draws.head + draws.getLast! (by simp only [draws]) = 21 ∧ 
    ∀ i j, i ≠ j → draws.nth i ≠ draws.nth j →
  (list.permutations balls).count (λ perm, 
    let draws := perm.take 4 in 
    draws.head + draws.getLast! (by simp only [draws]) = 21
  ) = 3060 := 
begin
  sorry
end

end draw_balls_count_l633_633512


namespace sum_of_angles_of_solutions_to_z6_eq_neg32_l633_633207

theorem sum_of_angles_of_solutions_to_z6_eq_neg32 :
  let θk (k : ℕ) := (180 + 360 * k) / 6 in
  (θk 0 + θk 1 + θk 2 + θk 3 + θk 4 + θk 5 = 1080) := 
  sorry  

end sum_of_angles_of_solutions_to_z6_eq_neg32_l633_633207


namespace max_height_of_ball_l633_633515

-- Define the height equation for the ball
def height (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 10

-- Define the maximum height
def maxHeight := 30

-- Lean statement to prove the maximum height
theorem max_height_of_ball : (∀ t : ℝ, height 1 = maxHeight) := by
  -- skipping the actual proof
  sorry

end max_height_of_ball_l633_633515


namespace engineers_crimson_meet_in_tournament_l633_633913

noncomputable def probability_engineers_crimson_meet : ℝ := 
  1 - Real.exp (-1)

theorem engineers_crimson_meet_in_tournament :
  (∃ (n : ℕ), n = 128) → 
  (∀ (i : ℕ), i < 128 → (∀ (j : ℕ), j < 128 → i ≠ j → ∃ (p : ℝ), p = probability_engineers_crimson_meet)) :=
sorry

end engineers_crimson_meet_in_tournament_l633_633913


namespace distance_AB_eq_2_l633_633732

-- Define the polar points A and B
def A := (2 : ℝ, Real.pi / 6)
def B := (2 : ℝ, -Real.pi / 6)

-- Define the theorem to prove the distance between A and B is 2
theorem distance_AB_eq_2 : real.sqrt ((A.1 - B.1)^2 + ((A.2 - B.2)^2)) = 2 := by
  sorry

end distance_AB_eq_2_l633_633732


namespace sum_of_integers_l633_633811

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
by
  sorry

end sum_of_integers_l633_633811


namespace turnip_bag_weighs_l633_633090

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l633_633090


namespace mary_income_percent_juan_l633_633772

-- Assume Juan's income is J
variable (J : ℝ)

-- Condition 1: Tim's income is 40 percent less than Juan's income
def T : ℝ := 0.60 * J

-- Condition 2: Mary's income is 70 percent more than Tim's income
def M : ℝ := 1.70 * T

-- The proof goal: Mary's income is 102 percent of Juan's income.
theorem mary_income_percent_juan (J : ℝ) : M = 1.02 * J := 
sorry

end mary_income_percent_juan_l633_633772


namespace count_5_in_range_1_to_700_l633_633283

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  n.digits 10 |>.contains d

def count_numbers_with_digit (d : ℕ) (m : ℕ) : ℕ :=
  (List.range' 1 m) |>.filter (contains_digit d) |>.length

theorem count_5_in_range_1_to_700 : count_numbers_with_digit 5 700 = 214 := by
  sorry

end count_5_in_range_1_to_700_l633_633283


namespace cash_price_of_tablet_l633_633397

-- Define the conditions
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 4 * 40
def next_4_months_payment : ℕ := 4 * 35
def last_4_months_payment : ℕ := 4 * 30
def savings : ℕ := 70

-- Define the total installment payments
def total_installment_payments : ℕ := down_payment + first_4_months_payment + next_4_months_payment + last_4_months_payment

-- The statement to prove
theorem cash_price_of_tablet : total_installment_payments - savings = 450 := by
  -- proof goes here
  sorry

end cash_price_of_tablet_l633_633397


namespace prime_remainder_composite_l633_633943

open Nat

theorem prime_remainder_composite (p : ℕ) (q r : ℕ) (h_prime : Prime p) (h_eq : p = 21 * q + r) (h_r_range : 0 < r ∧ r < 21) :
  r = 4 ∨ r = 8 ∨ r = 10 ∨ r = 16 ∨ r = 20 :=
by
  sorry

end prime_remainder_composite_l633_633943


namespace count_invitations_l633_633558

theorem count_invitations (teachers : Finset ℕ) (A B : ℕ) (hA : A ∈ teachers) (hB : B ∈ teachers) (h_size : teachers.card = 10):
  ∃ (ways : ℕ), ways = 140 ∧ ∀ (S : Finset ℕ), S.card = 6 → ((A ∈ S ∧ B ∉ S) ∨ (A ∉ S ∧ B ∈ S) ∨ (A ∉ S ∧ B ∉ S)) ↔ ways = 140 := 
sorry

end count_invitations_l633_633558


namespace calc_pow_product_l633_633164

theorem calc_pow_product : (0.25 ^ 2023) * (4 ^ 2023) = 1 := 
  by 
  sorry

end calc_pow_product_l633_633164


namespace calc_g_x_plus_3_l633_633382

def g (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem calc_g_x_plus_3 (x : ℝ) : g (x + 3) = x^2 + 3*x + 2 :=
by
  sorry

end calc_g_x_plus_3_l633_633382


namespace bisector_planes_intersect_in_single_line_l633_633915

variables {S A B C : Type} {a b c : Vector} (unit_SA : unit_vec a) (unit_SB : unit_vec b) (unit_SC : unit_vec c)

/-- Prove that the planes passing through the bisectors of the faces of a trihedral angle S A B C
  and perpendicular to the planes of these faces intersect in a single line described by the 
  vector [a, b] + [b, c] + [c, a] -/
theorem bisector_planes_intersect_in_single_line :
  ∃ L : Line, {P : Plane | P.passing_through_bisectors_of_faces SA SB SC} ∩ {Q : Plane | Q.perpendicular_to_faces SA SB SC} = {L} ∧
    L.vector = cross_product(a, b) + cross_product(b, c) + cross_product(c, a) :=
sorry

end bisector_planes_intersect_in_single_line_l633_633915


namespace area_of_section_parallel_to_base_l633_633721

-- Define the conditions and the final expression that needs to be proven
theorem area_of_section_parallel_to_base 
  (a α : ℝ) 
  (h1 : a > 0)
  (h2 : 0 < α ∧ α < π/2) :
  ∃ S : ℝ, S = a^2 * sin((π / 3) - α)^2 / (2 * sin((π / 3) + α)^2) :=
begin
  -- Proof is not required, hence we just use sorry.
  sorry
end

end area_of_section_parallel_to_base_l633_633721


namespace line_through_two_points_l633_633816

theorem line_through_two_points :
  (∃ m b : ℚ, ∀ x y : ℚ, (x, y) = (1, 3) ∨ (x, y) = (4, -2) → y = m * x + b) →
  let m := (-5 / 3 : ℚ)
  let b := (4 / 3 : ℚ)
  m + b = -1 / 3 :=
begin
  sorry
end

end line_through_two_points_l633_633816


namespace f_decreasing_on_interval_l633_633893

def f (x : ℝ) : ℝ := x + 3 / x

theorem f_decreasing_on_interval : ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 ∧ x2 < real.sqrt 3 → f x1 > f x2 :=
by
  sorry

end f_decreasing_on_interval_l633_633893


namespace min_ratio_ax_l633_633751

theorem min_ratio_ax (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) 
: y^2 - 1 = a^2 * (x^2 - 1) → ∃ (k : ℕ), k = 2 ∧ (a = k * x) := 
sorry

end min_ratio_ax_l633_633751


namespace find_fx_l633_633798

def shifted_cosine_function_eq (x : ℝ) : Prop :=
  cos (2 * (x - π / 4)) = sin (2 * x)

def required_function_eq (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x * sin x = sin (2 * x)

theorem find_fx :
  ∃ f : ℝ → ℝ, ∀ x : ℝ,
    shifted_cosine_function_eq x →
    required_function_eq f x → 
    f x = 2 * cos x :=
sorry

end find_fx_l633_633798


namespace g_at_neg2_eq_8_l633_633390

-- Define the functions f and g
def f (x : ℤ) : ℤ := 4 * x - 6
def g (y : ℤ) : ℤ := 3 * (y + 6/4)^2 + 4 * (y + 6/4) + 1

-- Statement of the math proof problem:
theorem g_at_neg2_eq_8 : g (-2) = 8 := 
by 
  sorry

end g_at_neg2_eq_8_l633_633390


namespace ratio_comparison_l633_633735

-- Define the ratios in the standard and sport formulations
def ratio_flavor_corn_standard : ℚ := 1 / 12
def ratio_flavor_water_standard : ℚ := 1 / 30
def ratio_flavor_water_sport : ℚ := 1 / 60

-- Define the amounts of corn syrup and water in the sport formulation
def corn_syrup_sport : ℚ := 2
def water_sport : ℚ := 30

-- Calculate the amount of flavoring in the sport formulation
def flavoring_sport : ℚ := water_sport / 60

-- Calculate the ratio of flavoring to corn syrup in the sport formulation
def ratio_flavor_corn_sport : ℚ := flavoring_sport / corn_syrup_sport

-- Define the theorem to prove the ratio comparison
theorem ratio_comparison :
  (ratio_flavor_corn_sport / ratio_flavor_corn_standard) = 3 :=
by
  -- Using the given conditions and definitions, prove the theorem
  sorry

end ratio_comparison_l633_633735


namespace turnip_bag_weight_l633_633062

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l633_633062


namespace trigonometric_identity_proof_l633_633659

theorem trigonometric_identity_proof (α : ℝ) (h1 : α ∈ (Real.pi / 2, Real.pi)) (h2 : Real.cos α = -5 / 13) : 
  Real.tan (α + Real.pi / 2) / Real.cos (α + Real.pi) = 13 / 12 :=
sorry

end trigonometric_identity_proof_l633_633659


namespace num_valid_a1s_le_2008_l633_633761

noncomputable def sequence (a : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := if even (sequence n) then (sequence n) / 2 else 3 * (sequence n) + 1

theorem num_valid_a1s_le_2008 : 
  (∀ a : ℕ, a ≤ 2008 → (sequence a 1) > a ∧ (sequence a 2) > a ∧ (sequence a 3) > a) ↔ 
  finset.card {a : ℕ | a ≤ 2008 ∧ (sequence a 1) > a ∧ (sequence a 2) > a ∧ (sequence a 3) > a} = 502 :=
sorry

end num_valid_a1s_le_2008_l633_633761


namespace no_snow_five_days_l633_633836

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633836


namespace turnip_bag_weights_l633_633074

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l633_633074


namespace true_propositions_count_l633_633962

-- Conditions
def prop1 := ∀ (p : Prop), p ∨ ¬ p
def prop2 := 2 + Real.sqrt 5 > Real.sqrt 3 + Real.sqrt 6
def prop3 := ∀ n : ℕ, (5 - 2 * n) > (5 - 2 * (n + 1))
def prop4 := ∀ x : ℝ, x < 0 → (2 * x + 1 / x) ≥ -2 * Real.sqrt 2

-- Theorem
theorem true_propositions_count : 
  let count_true := (if prop1 then 1 else 0) +
                    (if prop2 then 1 else 0) +
                    (if prop3 then 1 else 0) +
                    (if prop4 then 1 else 0)
  in count_true = 3 :=
by
  sorry

end true_propositions_count_l633_633962


namespace museum_pictures_l633_633563

theorem museum_pictures (P : ℕ) (h1 : ¬ (∃ k, P = 2 * k)) (h2 : ∃ k, P + 1 = 2 * k) : P = 3 := 
by 
  sorry

end museum_pictures_l633_633563


namespace xy_continuous_l633_633799

theorem xy_continuous {x y : ℝ} : Continuous (λ p : ℝ × ℝ, p.1 * p.2) :=
sorry

end xy_continuous_l633_633799


namespace least_number_to_add_l633_633494

theorem least_number_to_add (n : ℕ) :
  (exists n, 1202 + n % 4 = 0 ∧ (∀ m, (1202 + m) % 4 = 0 → n ≤ m)) → n = 2 :=
by
  sorry

end least_number_to_add_l633_633494


namespace values_of_2n_plus_m_l633_633827

theorem values_of_2n_plus_m (n m : ℤ) (h1 : 3 * n - m ≤ 4) (h2 : n + m ≥ 27) (h3 : 3 * m - 2 * n ≤ 45) 
  (h4 : n = 8) (h5 : m = 20) : 2 * n + m = 36 := by
  sorry

end values_of_2n_plus_m_l633_633827


namespace charity_tickets_l633_633525

theorem charity_tickets (f h p : ℕ) (H1 : f + h = 140) (H2 : f * p + h * (p / 2) = 2001) : f * p = 782 := 
sorry

end charity_tickets_l633_633525


namespace medals_awarded_satisfy_condition_l633_633476

-- Define the total number of sprinters
def total_sprinters : Nat := 10

-- Define the number of American sprinters
def american_sprinters : Nat := 4

-- Define the number of medals
def medals : Nat := 3

-- Define a function to compute the permutations of selecting k items from n items
def permutations (n k : Nat) : Nat := ∏ i in Finset.range k, n - i

-- Define the number of ways to award the medals such that at most one American gets a medal
def ways_to_award_medals (total_sprinters american_sprinters medals : Nat) : Nat :=
  let non_american_sprinters := total_sprinters - american_sprinters
  -- Case 1: No Americans get a medal
  let case1 := permutations non_american_sprinters medals
  -- Case 2: Exactly one American gets a medal
  let case2 := american_sprinters * medals * permutations non_american_sprinters (medals - 1)
  case1 + case2

-- The final theorem to be proved
theorem medals_awarded_satisfy_condition :
  ways_to_award_medals total_sprinters american_sprinters medals = 480 :=
by
  -- Placeholder for the proof
  sorry

end medals_awarded_satisfy_condition_l633_633476


namespace probability_factor_120_less_9_l633_633490

theorem probability_factor_120_less_9 : 
  ∀ n : ℕ, n = 120 → (∃ p : ℚ, p = 7 / 16 ∧ (∃ factors_less_9 : ℕ, factors_less_9 < 16 ∧ factors_less_9 = 7)) := 
by 
  sorry

end probability_factor_120_less_9_l633_633490


namespace area_triangle_AOB_l633_633043

theorem area_triangle_AOB (F A B O : ℝ × ℝ)
  (hF : F = (1, 0))
  (h_parabola_A : A.1 * A.1 = 4 * A.2)
  (h_parabola_B : B.1 * B.1 = 4 * B.2)
  (h_line : ∃ θ : ℝ, 0 < θ ∧ θ < real.pi ∧ A ≠ B ∧ (A.2 - F.2) = tan θ * (A.1 - F.1) ∧ (B.2 - F.2) = tan θ * (B.1 - F.1))
  (h_AF : dist A F = 3)
  (hO : O = (0, 0)) :
  let area := (1 / 2) * real.abs (O.1 * (A.2 - B.2) + A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2))
  in
  area = 3 * real.sqrt 2 / 2 := 
sorry

end area_triangle_AOB_l633_633043


namespace dvds_on_second_rack_l633_633499

theorem dvds_on_second_rack :
  ∃ (r : ℕ), r = 4 ∧
             (∀ (n : ℕ), (n = 1 → r = 2 * 1) ∧
                           (n = 2 → r = 2 * 2) ∧
                           (n = 3 → r = 2 * 4) ∧
                           (n = 4 → r = 2 * 8) ∧
                           (n = 5 → r = 2 * 16) ∧
                           (n = 6 → r = 2 * 32)) where 
sorry

end dvds_on_second_rack_l633_633499


namespace solve_for_x_l633_633424

theorem solve_for_x (x : ℝ) (h : (x - 75) / 3 = (8 - 3 * x) / 4) : 
  x = 324 / 13 :=
sorry

end solve_for_x_l633_633424


namespace known_number_is_24_l633_633870

noncomputable def HCF (a b : ℕ) : ℕ := sorry
noncomputable def LCM (a b : ℕ) : ℕ := sorry

theorem known_number_is_24 (A B : ℕ) (h1 : B = 182)
  (h2 : HCF A B = 14)
  (h3 : LCM A B = 312) : A = 24 := by
  sorry

end known_number_is_24_l633_633870


namespace tv_price_change_l633_633917

noncomputable def net_price_change (P : ℝ) : ℝ :=
  let new_price := (P * 0.80) in
  let final_price := (new_price * 1.30) in
  final_price - P

theorem tv_price_change :
  ∀ (P : ℝ), net_price_change P = 0.04 * P :=
by
  intro P
  rw [net_price_change]
  calc
    let new_price := P * 0.80
    let final_price := new_price * 1.30
    final_price - P
        = (P * 0.80 * 1.30) - P : by sorry  -- filling in the steps omitted
    ... = 1.04 * P - P : by sorry  -- intermediate calculation step
    ... = 0.04 * P : by sorry  -- final step to prove the theorem

end tv_price_change_l633_633917


namespace race_head_start_l633_633914

/-- A's speed is 22/19 times that of B. If A and B run a race, A should give B a head start of (3 / 22) of the race length so the race ends in a dead heat. -/
theorem race_head_start {Va Vb L H : ℝ} (hVa : Va = (22 / 19) * Vb) (hL_Va : L / Va = (L - H) / Vb) : 
  H = (3 / 22) * L :=
by
  sorry

end race_head_start_l633_633914


namespace turnip_bag_weighs_l633_633086

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l633_633086


namespace average_marathon_time_is_7_hours_l633_633979

-- Definitions of the problem conditions
def Casey_time : ℕ := 6
def Zendaya_additional_time (C : ℕ) : ℕ := (1 / 3 : ℚ) * C

@[simp]
theorem average_marathon_time_is_7_hours :
  let Zendaya_time := Casey_time + Zendaya_additional_time Casey_time,
      combined_time := Casey_time + Zendaya_time,
      average_time := combined_time / 2
  in average_time = 7 :=
by
  sorry

end average_marathon_time_is_7_hours_l633_633979


namespace rectangle_width_l633_633437

noncomputable def width_of_rectangle (area length width : ℕ) : Prop :=
  area = length * width ∧ width = length - 2

theorem rectangle_width : ∃ (width : ℕ), width_of_rectangle 63 9 7 :=
by
  exists 7
  unfold width_of_rectangle
  simp
  split
  · exact rfl
  · exact rfl

end rectangle_width_l633_633437


namespace total_distinct_plants_l633_633510

-- Declare the sets and their cardinalities
variables (A B C : Set ℕ)
variables (hA : |A| = 800) (hB : |B| = 700) (hC : |C| = 600)
variables (hAB : |A ∩ B| = 120) (hAC : |A ∩ C| = 200) (hBC : |B ∩ C| = 150)
variables (hABC : |A ∩ B ∩ C| = 75)

-- The statement we need to prove
theorem total_distinct_plants : |A ∪ B ∪ C| = 1705 :=
by
  -- Sorry indicates the proof is omitted; this ensures the statement compiles
  sorry

end total_distinct_plants_l633_633510


namespace product_of_inradius_and_circumradius_l633_633806

noncomputable def inradius_circumradius_product (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := sqrt (s * (s - a) * (s - b) * (s - c)) -- using Heron's formula
  let r := area / s
  let R := (a * b * c) / (4 * area)
  r * R

theorem product_of_inradius_and_circumradius :
  (∃ (a b c : ℝ), (a + b + c = 27) ∧ (a * b * c = 540) ∧ inradius_circumradius_product a b c = 10) :=
begin
  sorry
end

end product_of_inradius_and_circumradius_l633_633806


namespace probability_face_cards_l633_633888

theorem probability_face_cards :
  let first_card_hearts_face := 3 / 52
  let second_card_clubs_face_after_hearts := 3 / 51
  let combined_probability := first_card_hearts_face * second_card_clubs_face_after_hearts
  combined_probability = 1 / 294 :=
by 
  sorry

end probability_face_cards_l633_633888


namespace distinct_roots_on_circle_l633_633387

noncomputable def circle_condition (a : ℝ) : Prop :=
  (a > -1 ∧ a < 1) ∨ a = -3

theorem distinct_roots_on_circle
  (a : ℝ)
  (quadratic1 : z^2 - 2 * z + 5 = 0)
  (quadratic2 : z^2 + 2 * a * z + 1 = 0) :
  (∀ (z1 z2 z3 z4 : ℂ), (quadratic1 z1 = 0) → (quadratic1 z2 = 0) → (quadratic2 z3 = 0) → (quadratic2 z4 = 0) → z1 ≠ z2 ∧ z3 ≠ z4 ∧ ¬ z1 = z3 ∧ ¬ z1 = z4 ∧ ¬ z2 = z3 ∧ ¬ z2 = z4 ∧ ∃ (c : ℂ) (r : ℝ), r > 0 ∧ dist c z1 = r ∧ dist c z2 = r ∧ dist c z3 = r ∧ dist c z4 = r) ↔ circle_condition a := 
begin
  sorry
end

end distinct_roots_on_circle_l633_633387


namespace turnips_bag_l633_633047

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l633_633047


namespace midpoint_of_chord_intersection_l633_633968

theorem midpoint_of_chord_intersection
  (circle : Type)
  [incircle : MetricSpace circle]
  (O : circle) -- Center of the circle
  (radius : ℝ) -- Radius of the circle
  (A B : circle) -- Points on the circle such that AB is a chord
  (M : circle) (M_mid : midpoint A B M) -- M is the midpoint of the chord AB
  (C D E F : circle) -- Arbitrary points on the circle
  (P Q : circle) -- Intersection points on AB
  (CF ED : Prop) (CF_intersect : CF.intersects P) (ED_intersect : ED.intersects Q)
  (PM MQ : ℝ) -- Lengths PM and MQ
  (PM_eq_MQ : PM = MQ) : PM = MQ := 
sorry

end midpoint_of_chord_intersection_l633_633968


namespace find_m_value_l633_633679

noncomputable def parabola_and_lines (m : ℝ) : Prop :=
  let f := (λ x : ℝ, x^2 + 2 * x - 3)
  let l1 := (λ x : ℝ, -x + m)
  let x := (λ x : ℝ, x * ((1 : ℝ) / 2)) -- axis of symmetry x = -1
  let A := (x₁, f x₁)
  let C := (x₂, f x₂)
  let B := (x₃, f x₃)
  let D := (x₄, f x₄) in
  (x₁ + x₂ = -3) ∧ (x₁ * x₂ = -(3 + m)) ∧ (AC := (x₁ - x₂)^2 = 13) ∧ AC * BD = 26 → m = -2

theorem find_m_value (m : ℝ) : parabola_and_lines m :=
  sorry

end find_m_value_l633_633679


namespace period_of_tan_cot_sin_l633_633489

theorem period_of_tan_cot_sin (x : ℝ) :
  (∀ x, tan x + cot x = (1:ℝ) / (sin x * cos x)) →
  (∀ x, tendsto tan x (nhds x+π) (nhds (tan (x+π))) ∧ tendsto cot x (nhds x+π) (nhds (cot (x+π)))) →
  (∀ x, tendsto (1 / (sin x * cos x)) (nhds x+π) (nhds (1 / (sin (x+π) * cos (x+π))))) →
  (∀ x, tendsto sin x (nhds x+2*π) (nhds (sin (x+2*π)))) →
  (∃ p,  p > 0 ∧ ∀ x, tan x + cot x + sin x = tan (x + p) + cot (x + p) + sin (x + p)) :=
by
  sorry

end period_of_tan_cot_sin_l633_633489


namespace trapezoid_larger_angle_l633_633034

theorem trapezoid_larger_angle (n : ℕ) (total_degrees : ℝ) (angle_ratio : ℝ) (isosceles : Prop) :
  n = 12 → 
  total_degrees = 360 → 
  angle_ratio = 2 → 
  isosceles → 
  ∃ θ, θ = 97.5 :=
by
  intros h1 h2 h3 h4,
  sorry

end trapezoid_larger_angle_l633_633034


namespace smallest_c_inequality_l633_633493

theorem smallest_c_inequality (m n : ℤ) : ∃ c, c ≥ 9 ∧ (27 ^ c) * (2 ^ (24 - n)) > (3 ^ (24 + m)) * (5 ^ n) :=
sorry

end smallest_c_inequality_l633_633493


namespace lennon_reimbursement_rate_l633_633371

theorem lennon_reimbursement_rate :
  let miles_monday := 18
  let miles_tuesday := 26
  let miles_wednesday := 20
  let miles_thursday := 20
  let miles_friday := 16
  let total_miles := miles_monday + miles_tuesday + miles_wednesday + miles_thursday + miles_friday
  let total_reimbursement : ℝ := 36
  (total_reimbursement / total_miles) = 0.36 :=
by
  let miles_monday := 18
  let miles_tuesday := 26
  let miles_wednesday := 20
  let miles_thursday := 20
  let miles_friday := 16
  let total_miles := miles_monday + miles_tuesday + miles_wednesday + miles_thursday + miles_friday
  let total_reimbursement : ℝ := 36
  have h_total_miles : total_miles = 18 + 26 + 20 + 20 + 16 := rfl
  have h_total_reimbursement : total_miles = 100 := by norm_num
  have h_rate : (total_reimbursement / total_miles = 36 / 100) := rfl
  norm_num
  exact h_rate

end lennon_reimbursement_rate_l633_633371


namespace sandy_carrots_l633_633794

-- Definitions and conditions
def total_carrots : ℕ := 14
def mary_carrots : ℕ := 6

-- Proof statement
theorem sandy_carrots : (total_carrots - mary_carrots) = 8 :=
by
  -- sorry is used to bypass the actual proof steps
  sorry

end sandy_carrots_l633_633794


namespace zero_in_interval_l633_633875

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval (h_mono : ∀ x y, 0 < x → x < y → f x < f y) (h_f2 : f 2 < 0) (h_f3 : 0 < f 3) :
  ∃ x₀ ∈ (Set.Ioo 2 3), f x₀ = 0 :=
by
  sorry

end zero_in_interval_l633_633875


namespace compute_expression_l633_633592

theorem compute_expression :
  -9 * 5 - (-(7 * -2)) + (-(11 * -6)) = 7 :=
by
  sorry

end compute_expression_l633_633592


namespace sue_dogs_walked_l633_633591

theorem sue_dogs_walked :
  ∀ (christian_saved sue_saved perfume_cost cost_per_yard num_yards cost_per_dog remaining_amount christian_earnings total_savings amount_needed sue_earnings sue_needed_dogs : ℕ),
    christian_saved = 5 →
    sue_saved = 7 →
    perfume_cost = 50 →
    cost_per_yard = 5 →
    num_yards = 4 →
    cost_per_dog = 2 →
    remaining_amount = 6 →
    christian_earnings = num_yards * cost_per_yard →
    total_savings = christian_saved + sue_saved + christian_earnings →
    amount_needed = perfume_cost - total_savings →
    sue_earnings = amount_needed - remaining_amount →
    sue_needed_dogs = sue_earnings / cost_per_dog →
    sue_needed_dogs = 6 :=
by
  intros christian_saved sue_saved perfume_cost cost_per_yard num_yards cost_per_dog remaining_amount christian_earnings total_savings amount_needed sue_earnings sue_needed_dogs 
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12
  have H1 : christian_saved = 5 := h1
  have H2 : sue_saved = 7 := h2
  have H3 : perfume_cost = 50 := h3
  have H4 : cost_per_yard = 5 := h4
  have H5 : num_yards = 4 := h5
  have H6 : cost_per_dog = 2 := h6
  have H7 : remaining_amount = 6 := h7
  have H8 : christian_earnings = 4 * 5 := h8
  have H9 : total_savings = 5 + 7 + (4 * 5) := h9
  have H10 : amount_needed = 50 - (5 + 7 + (4 * 5)) := h10
  have H11 : sue_earnings = (50 - (5 + 7 + (4 * 5))) - 6 := h11
  have H12 : sue_needed_dogs = ((50 - (5 + 7 + (4 * 5))) - 6) / 2 := h12
  exact sorry

end sue_dogs_walked_l633_633591


namespace parallelogram_diagonal_property_l633_633764

theorem parallelogram_diagonal_property
  (A B C D E F: Point)
  (h_parallelogram: parallelogram A B C D)
  (h_CE_perp_AB: ∃ G : Point, collinear C E G ∧ perpendicular G A B)
  (h_CF_perp_AD: ∃ H : Point, collinear C F H ∧ perpendicular H A D)
  :
  let AB := distance A B
  let AD := distance A D
  let AC := distance A C
  let AE := distance A E
  let AF := distance A F
  in AB * AE + AD * AF = AC ^ 2 := sorry


end parallelogram_diagonal_property_l633_633764


namespace turnip_bag_weight_l633_633060

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l633_633060


namespace correct_statements_l633_633264

def line : Type := sorry     -- Placeholder for line definition
def is_perpendicular (a b : line) : Prop := sorry   -- a ⊥ b
def is_parallel (a b : line) : Prop := sorry        -- a ∥ b
def forms_angle (a : line) (P : Type) (θ : float) : Prop := sorry   -- Line forms angle θ with line a through point P
def are_skew_lines (a b : line) : Prop := sorry    -- a and b are skew lines

theorem correct_statements {a b c : line} :
  (is_parallel a b ∧ is_parallel b c → is_parallel a c) ∧
  (∃ infinite_lines : set line, ∀ l ∈ infinite_lines, is_perpendicular l a ∧ is_perpendicular l b) := 
by 
  sorry

end correct_statements_l633_633264


namespace fill_time_l633_633780

-- Definition of the conditions
def faster_pipe_rate (t : ℕ) := 1 / t
def slower_pipe_rate (t : ℕ) := 1 / (4 * t)
def combined_rate (t : ℕ) := faster_pipe_rate t + slower_pipe_rate t
def time_to_fill_tank (t : ℕ) := 1 / combined_rate t

-- Given t = 50, prove the combined fill time is 40 minutes which is equal to the target time to fill the tank.
theorem fill_time (t : ℕ) (h : 4 * t = 200) : t = 50 → time_to_fill_tank t = 40 :=
by
  intros ht
  rw [ht]
  sorry

end fill_time_l633_633780


namespace number_of_teams_l633_633320

-- Define the statement representing the problem and conditions
theorem number_of_teams (n : ℕ) (h : 2 * n * (n - 1) = 9800) : n = 50 :=
sorry

end number_of_teams_l633_633320


namespace total_days_2010_to_2013_l633_633696

theorem total_days_2010_to_2013 :
  let year2010_days := 365
  let year2011_days := 365
  let year2012_days := 366
  let year2013_days := 365
  year2010_days + year2011_days + year2012_days + year2013_days = 1461 := by
  sorry

end total_days_2010_to_2013_l633_633696


namespace Apollonius_circle_symmetry_l633_633348

theorem Apollonius_circle_symmetry (a : ℝ) (h : a > 1): 
  let F1 := (-1, 0)
  let F2 := (1, 0)
  let locus_C := {P : ℝ × ℝ | ∃ x y, P = (x, y) ∧ (Real.sqrt ((x + 1)^2 + y^2) = a * Real.sqrt ((x - 1)^2 + y^2))}
  let symmetric_y := ∀ (P : ℝ × ℝ), P ∈ locus_C → (P.1, -P.2) ∈ locus_C
  symmetric_y := sorry

end Apollonius_circle_symmetry_l633_633348


namespace arithmetic_sequence_50th_term_l633_633487

-- Definitions based on the conditions stated
def first_term := 3
def common_difference := 5
def n := 50

-- Function to calculate the n-th term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- The theorem that needs to be proven
theorem arithmetic_sequence_50th_term : nth_term first_term common_difference n = 248 := 
by
  sorry

end arithmetic_sequence_50th_term_l633_633487


namespace max_closed_companies_for_connected_graph_l633_633876

theorem max_closed_companies_for_connected_graph (n : ℕ) : 
  ∃ k, k = n ∧ 
    let towns := 2 ^ (2 * n + 1)
    let companies := 2 * n + 1
    ∀ closed_companies, closed_companies ≤ n → 
      ∀ k : ℕ, k ∈ closed_companies → 
        is_connected (remove_edges (complete_graph towns) k) :=
sorry

end max_closed_companies_for_connected_graph_l633_633876


namespace cost_per_kg_blend_correct_l633_633113

noncomputable def cost_per_kg_blend : ℝ :=
  let cost_mozzarella := 19 * 504.35
  let cost_romano := 18.999999999999986 * 887.75
  let total_cost := cost_mozzarella + cost_romano
  let total_weight := 19 + 18.999999999999986
  total_cost / total_weight

theorem cost_per_kg_blend_correct : cost_per_kg_blend ≈ 695.89 := by
  sorry

end cost_per_kg_blend_correct_l633_633113


namespace polar_to_rectangular_inequality_range_l633_633959

-- Part A: Transforming a polar coordinate equation to a rectangular coordinate equation
theorem polar_to_rectangular (ρ θ : ℝ) : 
  (ρ^2 * Real.cos θ - ρ = 0) ↔ ((ρ = 0 ∧ 0 = 1) ∨ (ρ ≠ 0 ∧ Real.cos θ = 1 / ρ)) := 
sorry

-- Part B: Determining range for an inequality
theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2-x| + |x+1| ≤ a) ↔ (a ≥ 9) := 
sorry

end polar_to_rectangular_inequality_range_l633_633959


namespace camera_film_remaining_photos_l633_633889

variables (m n : ℕ)
variables (a b p : ℕ)

-- Conditions from the problem
def delegation1 : ℕ := 35 * m + 15
def delegation2 : ℕ := 35 * n + 20
def total_photos : ℕ := delegation1 * delegation2

-- Theorem to prove that after last photo we can take 15 more photos
theorem camera_film_remaining_photos (m n : ℕ) : (35 * m + 15) * (35 * n + 20) % 35 = 20 :=
by {
  let a := 35 * m + 15,
  let b := 35 * n + 20,
  exact nat.mul_mod_right a b
}

end camera_film_remaining_photos_l633_633889


namespace turnip_bag_weights_l633_633075

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l633_633075


namespace sum_of_divisors_of_250_mod_eq_6_is_375_l633_633763

theorem sum_of_divisors_of_250_mod_eq_6_is_375 :
  (∑ d in (finset.filter (λ d : ℕ, d < 1000 ∧ d > 99) 
    (finset.filter (λ d : ℕ, 256 % d = 6) (finset.divisors 250))), d) = 375 :=
by
  sorry

end sum_of_divisors_of_250_mod_eq_6_is_375_l633_633763


namespace largest_k_l633_633793

def S := { x : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ | let (a, b, c, d, e, f) := x in a^2 + b^2 + c^2 + d^2 + e^2 = f^2 }

theorem largest_k : ∀ x ∈ S, 24 ∣ (x.1 * x.2 * x.3 * x.4 * x.5 * x.6) := by
  intro x hx
  let (a, b, c, d, e, f) := x
  rw [Set.mem_setOf_eq] at hx
  sorry

end largest_k_l633_633793


namespace turnip_bag_weight_l633_633059

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l633_633059


namespace heptagon_diagonal_relation_l633_633224

theorem heptagon_diagonal_relation (A : Type*) [MetricSpace A] 
  (heptagon : List A) (h : heptagon.length = 7) (regular : ∀ i j, dist (heptagon.nth_le i (by sorry)) (heptagon.nth_le j (by sorry)) = dist (heptagon.nth_le ((i + 1) % 7) (by sorry)) (heptagon.nth_le ((j + 1) % 7) (by sorry))) :
  1 / dist (heptagon.nth_le 0 (by sorry)) (heptagon.nth_le 1 (by sorry)) = 
  1 / dist (heptagon.nth_le 0 (by sorry)) (heptagon.nth_le 2 (by sorry)) +
  1 / dist (heptagon.nth_le 0 (by sorry)) (heptagon.nth_le 3 (by sorry)) :=
by sorry

end heptagon_diagonal_relation_l633_633224


namespace turnips_bag_l633_633049

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l633_633049


namespace turnip_bag_weight_l633_633054

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l633_633054


namespace math_problem_l633_633637

noncomputable def A_n := sorry
noncomputable def C_n := sorry
noncomputable def polynomial (x : ℝ) (n : ℕ) := ∑ i in (finset.range (n+1)), sorry

theorem math_problem :
  (A_n 4 = 24 * C_n 6) →
  (∀n, (2 * x - 3)^n = polynomial x n) →
  n = 10 ∧ (∑ i in (finset.range n).filter (λ i, i ≠ 0 ),  sorry)= 0 :=
by {
  sorry
}

end math_problem_l633_633637


namespace sufficient_condition_inequality_l633_633221

theorem sufficient_condition_inequality (x m : ℝ) (h_p : (x - 1) / x ≤ 0) : 4^x + 2^x - m ≤ 0 → 6 ≤ m :=
by
  -- conditions for p and q
  have p := (0 < x ∧ x ≤ 1),
  have q := (4^x + 2^x - m ≤ 0),
  sorry

end sufficient_condition_inequality_l633_633221


namespace combined_total_time_l633_633970

def jerry_time : ℕ := 3
def elaine_time : ℕ := 2 * jerry_time
def george_time : ℕ := elaine_time / 3
def kramer_time : ℕ := 0
def total_time : ℕ := jerry_time + elaine_time + george_time + kramer_time

theorem combined_total_time : total_time = 11 := by
  unfold total_time jerry_time elaine_time george_time kramer_time
  rfl

end combined_total_time_l633_633970


namespace quadratic_inequality_l633_633715

noncomputable def a : ℤ := -12
noncomputable def b : ℤ := -2

theorem quadratic_inequality (a b : ℤ) 
  (h1 : ∀ x : ℝ, -0.5 < x ∧ x < 1/3 → a * x^2 + b * x + 2 > 0) 
  (h2 : ∀ x : ℝ, ~( -0.5 < x ∧ x < 1/3 ) → a * x^2 + b * x + 2 ≤ 0) :
  a - b = -10 :=
sorry

end quadratic_inequality_l633_633715


namespace max_min_values_l633_633197

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem max_min_values : 
  (∃ a b, a ∈ set.interval (-1 : ℝ) 1 ∧ b ∈ set.interval (-1 : ℝ) 1 ∧ 
    (∀ x ∈ set.interval (-1 : ℝ) 1, f x ≤ f b) ∧ 
    (∀ x ∈ set.interval (-1 : ℝ) 1, f a ≤ f x) ∧ 
    f a = 1 ∧ f b = 5) :=
sorry

end max_min_values_l633_633197


namespace projection_of_b_on_a_l633_633268

variables (a b : ℝ^3)
variables (ha : ∥a∥ = 2) (hb : ∥b∥ = √2)
variables (h_perp : a ⬝ (a + 2 • b) = 0)

theorem projection_of_b_on_a :
  (a ⬝ b) / ∥a∥ = -1 :=
sorry

end projection_of_b_on_a_l633_633268


namespace scientific_notation_conversion_l633_633966

theorem scientific_notation_conversion :
  (6.1 * 10^9 = (6.1 : ℝ) * 10^8) :=
sorry

end scientific_notation_conversion_l633_633966


namespace not_snowing_next_five_days_l633_633862

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l633_633862


namespace sum_mod_7_l633_633631

/-- Define the six numbers involved. -/
def a := 102345
def b := 102346
def c := 102347
def d := 102348
def e := 102349
def f := 102350

/-- State the theorem to prove the remainder of their sum when divided by 7. -/
theorem sum_mod_7 : 
  (a + b + c + d + e + f) % 7 = 5 := 
by sorry

end sum_mod_7_l633_633631


namespace seq_geom_and_sum_l633_633638

theorem seq_geom_and_sum (S : ℕ → ℝ) (a : ℕ → ℝ)  (n : ℕ) (h : n > 0) 
  (hSn : ∀ n : ℕ, S n = 2 * a n + n^2 - 3 * n - 2) :
  (∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n - 2 * n + 2) ∧ 
  (∀ b : ℕ → ℝ, (∀ n : ℕ, b n = a n * (if n % 2 = 0 then 1 else -1)) → 
    (∀ P : ℕ → ℝ, (P n = ∑ i in Finset.range n, b (i + 1)) →
      (∀ n : ℕ, n > 0 →
        (P n = if n % 2 = 1 then - (2^(n + 1) / 3) - n - 5/3 else (2/3) * (2^n - 1) + n)))) := 
sorry

end seq_geom_and_sum_l633_633638


namespace baseball_card_value_change_l633_633928

theorem baseball_card_value_change (v0 : ℝ) :
  let v1 := v0 * 0.85,
      v2 := v1 * 1.10,
      v3 := v2 * 0.80,
      v4 := v3 * 0.75,
      percent_change := ((v4 - v0) / v0) * 100 in
  percent_change = -43.9 :=
by
  sorry

end baseball_card_value_change_l633_633928


namespace john_naps_60_hours_in_days_l633_633369

theorem john_naps_60_hours_in_days :
  (∀ (naps_per_week nap_duration total_hours : ℕ),
     (naps_per_week = 3) →
     (nap_duration = 2) →
     (total_hours = 60) →
     (total_hours / nap_duration / naps_per_week * 7 = 70)) :=
begin
  intros naps_per_week nap_duration total_hours,
  assume h1 h2 h3,
  sorry
end

end john_naps_60_hours_in_days_l633_633369


namespace correct_total_cost_correct_remaining_donuts_l633_633432

-- Conditions
def budget : ℝ := 50
def cost_per_box : ℝ := 12
def discount_percentage : ℝ := 0.10
def number_of_boxes_bought : ℕ := 4
def donuts_per_box : ℕ := 12
def boxes_given_away : ℕ := 1
def additional_donuts_given_away : ℕ := 6

-- Calculations based on conditions
def total_cost_before_discount : ℝ := number_of_boxes_bought * cost_per_box
def discount_amount : ℝ := discount_percentage * total_cost_before_discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

def total_donuts : ℕ := number_of_boxes_bought * donuts_per_box
def total_donuts_given_away : ℕ := (boxes_given_away * donuts_per_box) + additional_donuts_given_away
def remaining_donuts : ℕ := total_donuts - total_donuts_given_away

-- Theorems to prove
theorem correct_total_cost : total_cost_after_discount = 43.20 := by
  -- proof here
  sorry

theorem correct_remaining_donuts : remaining_donuts = 30 := by
  -- proof here
  sorry

end correct_total_cost_correct_remaining_donuts_l633_633432


namespace problem_statement_l633_633223

noncomputable def f (x : ℝ) : ℝ := sorry -- definition of f, to be provided later

variables {x1 x2 x3 : ℝ}

-- Define the conditions as hypotheses in Lean
hypothesis h_monotonic : ∀ x y : ℝ, x < y → f(x) < f(y)
hypothesis h_odd_function : ∀ x : ℝ, f(-x) + f(x) = 0
hypothesis h_condition1 : x1 + x2 > 0
hypothesis h_condition2 : x2 + x3 > 0
hypothesis h_condition3 : x3 + x1 > 0

-- The theorem we want to prove
theorem problem_statement : f(x1) + f(x2) + f(x3) > 0 :=
sorry

end problem_statement_l633_633223


namespace bookmarks_sold_l633_633607

-- Definitions pertaining to the problem
def total_books_sold : ℕ := 72
def books_ratio : ℕ := 9
def bookmarks_ratio : ℕ := 2

-- Statement of the theorem
theorem bookmarks_sold :
  (total_books_sold / books_ratio) * bookmarks_ratio = 16 :=
by
  sorry

end bookmarks_sold_l633_633607


namespace regular_2022gon_area_is_3_l633_633946

noncomputable def area_of_regular_2022gon (P : ℝ) (hP : P = 6.28) : ℕ :=
  let n := 2022
  let side_length := P / n
  let R := side_length / (2 * real.sin (real.pi / n))
  let area := real.pi * R^2
  nat.floor area
  
theorem regular_2022gon_area_is_3 :
  area_of_regular_2022gon 6.28 (by norm_num) = 3 := 
sorry

end regular_2022gon_area_is_3_l633_633946


namespace student_scores_l633_633122

def weighted_average (math history science geography : ℝ) : ℝ :=
  (math * 0.30) + (history * 0.30) + (science * 0.20) + (geography * 0.20)

theorem student_scores :
  ∀ (math history science geography : ℝ),
    math = 74 →
    history = 81 →
    science = geography + 5 →
    science ≥ 75 →
    weighted_average math history science geography = 80 →
    science = 86.25 ∧ geography = 81.25 :=
by
  intros math history science geography h_math h_history h_science h_min_sci h_avg
  sorry

end student_scores_l633_633122


namespace true_proposition_among_three_l633_633595

theorem true_proposition_among_three : 
  (∀ (x : ℝ) (a : ℝ), a > 0 ∧ a ≠ 1 → (deriv (λ x, log a x) x = 1 / (x * log a (exp 1)))) ∧
  (∀ (x : ℝ), deriv (λ x, cos x) x = -sin x) ∧
  (∀ (u v : ℝ → ℝ) (x : ℝ), deriv (λ x, u x / v x) x = (u x * deriv v x - v x * deriv u x) / (v x ^ 2)) → 
  true :=
sorry

end true_proposition_among_three_l633_633595


namespace number_of_terms_in_expansion_l633_633182

theorem number_of_terms_in_expansion (a b : ℂ) : 
  (number_of_distinct_terms (expand ([(a+2*b)^3 * (a-2*b)^3]^3)) = 10) :=
sorry

end number_of_terms_in_expansion_l633_633182


namespace sum_of_xyz_l633_633604

theorem sum_of_xyz :
  (∃ x y z, log 2 (log 5 (log 3 x)) = 1 ∧ 
            log 5 (log 3 (log 2 y)) = 0 ∧ 
            log 3 (log 2 (log 5 z)) = 1 ∧ 
            x + y + z = 3^25 + 8 + 5^8) := 
by sorry

end sum_of_xyz_l633_633604


namespace total_books_l633_633586

theorem total_books (b1 b2 b3 b4 b5 b6 b7 b8 b9 : ℕ) :
  b1 = 56 →
  b2 = b1 + 2 →
  b3 = b2 + 2 →
  b4 = b3 + 2 →
  b5 = b4 + 2 →
  b6 = b5 + 2 →
  b7 = b6 - 4 →
  b8 = b7 - 4 →
  b9 = b8 - 4 →
  b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 = 490 :=
by
  sorry

end total_books_l633_633586


namespace circle_properties_l633_633755

-- Definition of the given circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4 * y - 15 = - y^2 + 12 * x + 27

-- Definitions for the center (p, q) and radius
def p : ℝ := 6
def q : ℝ := 2
def s : ℝ := Real.sqrt 82

-- Statement of the proof problem
theorem circle_properties : (∃ x y : ℝ, circle_equation x y) ∧ (p = 6 ∧ q = 2 ∧ s = Real.sqrt 82 ∧ p + q + s = 8 + Real.sqrt 82) :=
by
  sorry

end circle_properties_l633_633755


namespace man_speed_with_stream_l633_633942

-- Define the man's rate in still water
def man_rate_in_still_water : ℝ := 6

-- Define the man's rate against the stream
def man_rate_against_stream (stream_speed : ℝ) : ℝ :=
  man_rate_in_still_water - stream_speed

-- The given condition that the man's rate against the stream is 10 km/h
def man_rate_against_condition : Prop := ∃ (stream_speed : ℝ), man_rate_against_stream stream_speed = 10

-- We aim to prove that the man's speed with the stream is 10 km/h
theorem man_speed_with_stream (stream_speed : ℝ) (h : man_rate_against_stream stream_speed = 10) :
  man_rate_in_still_water + stream_speed = 10 := by
  sorry

end man_speed_with_stream_l633_633942


namespace prime_squares_mod_180_l633_633576

theorem prime_squares_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  ∃ r ∈ {1, 49}, ∀ q ∈ {1, 49}, (q ≡ p^2 [MOD 180]) → q = r :=
by
  sorry

end prime_squares_mod_180_l633_633576


namespace sqrt_eighth_equation_l633_633997

theorem sqrt_eighth_equation :
    ( ∑ k in finset.range(9), ( nat.choose 8 k) * 100^k ) = 549755289601 →
    real.rpow 549755289601 ((1:ℝ) / 8) = 101 := by
  sorry

end sqrt_eighth_equation_l633_633997


namespace vector_calculation_l633_633618

open Matrix

-- Define the vectors and scalar multiplication
def vec1 : Fin 2 → ℤ := ![4, -2]
def vec2 : Fin 2 → ℤ := ![-3, 5]
def scalar : ℤ := 3
def scaled_vec : Fin 2 → ℤ := λ i, scalar * vec1 i

-- Define the vector addition
def result_vec : Fin 2 → ℤ := λ i, scaled_vec i + vec2 i

theorem vector_calculation :
  result_vec = ![9, -1] :=
by 
  -- Proof will be provided here
  sorry

end vector_calculation_l633_633618


namespace proof_equivalence_l633_633240

variables {a b c d e f : Prop}

theorem proof_equivalence (h₁ : (a ≥ b) → (c > d)) 
                        (h₂ : (c > d) → (a ≥ b)) 
                        (h₃ : (a < b) ↔ (e ≤ f)) :
  (c ≤ d) ↔ (e ≤ f) :=
sorry

end proof_equivalence_l633_633240


namespace count_numbers_with_digit_five_l633_633277

theorem count_numbers_with_digit_five : 
  (finset.filter (λ n : ℕ, ∃ d : ℕ, d ∈ digits 10 n ∧ d = 5) (finset.range 701)).card = 133 := 
by 
  sorry

end count_numbers_with_digit_five_l633_633277


namespace jimmy_shared_expenses_with_2_friends_l633_633609

variable (charge_hostel_per_night charge_cabin_per_night total_expense nights_hostel nights_cabin friends_share : ℕ)

-- Defining the variables based on conditions
def charge_hostel_per_night := 15
def charge_cabin_per_night := 45
def total_expense := 75
def nights_hostel := 3
def nights_cabin := 2
def friends_share := 2

-- Calculation of total cost at hostel and cabin, checking if Jimmy shared expenses with 2 friends
theorem jimmy_shared_expenses_with_2_friends :
  let cost_hostel := nights_hostel * charge_hostel_per_night;
      total_cabin_cost := nights_cabin * charge_cabin_per_night;
      jimmy_share := total_cabin_cost / friends_share;
      cost_cabin := total_expense - cost_hostel;
      num_friends := total_cabin_cost / jimmy_share - 1
  in num_friends = 2 := 
by 
  sorry

end jimmy_shared_expenses_with_2_friends_l633_633609


namespace binomial_expansion_l633_633713

-- We define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- You define necessary variables and constants
variables (a : ℝ)

-- Main statement of the problem
theorem binomial_expansion (h : binomial 6 3 * a^(-3) = 5/2) : a = 2 :=
by 
  sorry

end binomial_expansion_l633_633713


namespace b_share_l633_633539

theorem b_share (total_money : ℕ) (ratio_a ratio_b ratio_c : ℕ) (money_parts : ℕ) (value_one_part : ℕ) (b_share : ℕ) :
  total_money = 5400 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  money_parts = ratio_a + ratio_b + ratio_c →
  value_one_part = total_money / money_parts →
  b_share = value_one_part * ratio_b →
  b_share = 1800 :=
begin
  intros h_total_money h_ratio_a h_ratio_b h_ratio_c h_money_parts h_value_one_part h_b_share,
  simp [h_total_money, h_ratio_a, h_ratio_b, h_ratio_c] at h_money_parts,
  simp [h_total_money, h_money_parts] at h_value_one_part,
  simp [h_value_one_part, h_ratio_b] at h_b_share,
  simp [h_b_share],
end

end b_share_l633_633539


namespace probability_two_red_two_blue_l633_633025

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose 20 4
  let choose_2_red := Nat.choose 12 2
  let choose_2_blue := Nat.choose 8 2
  (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690 := by
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose total_marbles 4
  let choose_2_red := Nat.choose total_red 2
  let choose_2_blue := Nat.choose total_blue 2
  show (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690
  sorry

end probability_two_red_two_blue_l633_633025


namespace find_marks_in_math_l633_633599

/-
David obtained 81 marks in English, some marks in Mathematics, 82 in Physics, 67 in Chemistry, and 85 in Biology. His average marks are 76. What are his marks in Mathematics?
-/

/- Define the given conditions as Lean definitions -/
def marks_in_english : ℕ := 81
def marks_in_physics : ℕ := 82
def marks_in_chemistry : ℕ := 67
def marks_in_biology : ℕ := 85
def average_marks : ℕ := 76
def number_of_subjects : ℕ := 5

/- Define the expected result -/
def marks_in_math : ℕ := 65

/- Prove that David's marks in Mathematics are indeed 65 given the conditions -/
theorem find_marks_in_math : 
  let total_marks := average_marks * number_of_subjects in
  let total_known_marks := marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology in
  total_marks - total_known_marks = marks_in_math := by
  sorry

end find_marks_in_math_l633_633599


namespace multiple_of_3_converges_l633_633013

noncomputable def cube_sum_of_digits (n : Nat) : Nat :=
  (n.digits 10).map (λ d => d^3).foldl (· + ·) 0

theorem multiple_of_3_converges :
  ∀ (n : Nat), n % 3 = 0 → ∃ k : Nat, (∀ m, m >= k → cube_sum_of_digits^[m] n = 153)
:= by
  -- Proof goes here
  sorry

end multiple_of_3_converges_l633_633013


namespace odd_numbers_square_division_l633_633686

theorem odd_numbers_square_division (m n : ℤ) (hm : Odd m) (hn : Odd n) (h : m^2 - n^2 + 1 ∣ n^2 - 1) : ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := 
sorry

end odd_numbers_square_division_l633_633686


namespace unique_zero_of_f_l633_633256

def f (a b : ℝ) (x : ℝ) : ℝ := a ^ x + x - b

theorem unique_zero_of_f :
  (∃ a b : ℝ, 2^a = 3 ∧ 3^b = 2) →
  (∃! x : ℝ, f (log 2 3) (log 3 2) x = 0) :=
by
  sorry

end unique_zero_of_f_l633_633256


namespace treaty_day_is_wednesday_l633_633473

def war_start : ℕ := 2000_01_15
def war_end : ℕ := 2003_03_25
def start_day_of_week : ℕ := 0 -- 0 represents Sunday

/--
If a war started on January 15, 2000, which was a Sunday,
and ended on March 25, 2003, then the day of the week on
March 25, 2003, is a Wednesday.
-/
theorem treaty_day_is_wednesday : 
  (days_since_start war_start war_end) % 7 = 3 :=
sorry

/--
Number of days between two given dates.
-/
def days_since_start (start end : ℕ) : ℕ := 
  -- Calculation for days between dates here
  sorry

end treaty_day_is_wednesday_l633_633473


namespace not_snowing_next_five_days_l633_633861

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l633_633861


namespace probability_point_closer_to_origin_l633_633540

noncomputable def is_point_closer_to_origin (P : ℝ × ℝ) : Prop :=
  (P.1^2 + P.2^2) < ((P.1 - 4)^2 + (P.2 - 2)^2)

theorem probability_point_closer_to_origin :
  let rectangle := {P : ℝ × ℝ | 0 ≤ P.1 ∧ P.1 ≤ 3 ∧ 0 ≤ P.2 ∧ P.2 ≤ 2} in
  (volume (measure_theory.outer_measure.of_function
    (λ P, if is_point_closer_to_origin P then 1 else 0) 0 rectangle)) / (volume rectangle) = 1/6 :=
by sorry

end probability_point_closer_to_origin_l633_633540


namespace probability_no_snow_l633_633839

theorem probability_no_snow {
  let p_snow : ℚ := 2/3,     -- Probability of snow on any one day
  let p_no_snow : ℚ := 1 - p_snow,  -- Probability of no snow on any one day
  let p_no_snow_5_days : ℚ := p_no_snow ^ 5  -- Probability of no snow on any of the five days
} : p_no_snow_5_days = 1 / 243 := by
  sorry

end probability_no_snow_l633_633839


namespace no_snow_five_days_l633_633854

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633854


namespace probability_denis_oleg_play_l633_633134

theorem probability_denis_oleg_play (n : ℕ) (h_n : n = 26) :
  (1 : ℚ) / 13 = 
  let total_matches : ℕ := n - 1 in
  let num_pairs := n * (n - 1) / 2 in
  (total_matches : ℚ) / num_pairs :=
by
  -- You can provide a proof here if necessary
  sorry

end probability_denis_oleg_play_l633_633134


namespace triangle_formation_probability_l633_633185

theorem triangle_formation_probability :
  let sticks := {1, 2, 4, 6, 8, 10, 12, 14}
  let valid_combinations := 
    {(4, 6, 8), (4, 8, 10), (4, 10, 12), (6, 8, 10), (6, 10, 12), 
     (6, 10, 14), (8, 10, 12), (8, 10, 14), (8, 12, 14), (10, 12, 14)}
  in
  ((finset.card valid_combinations : ℚ) / (nat.choose 8 3 : ℚ)) = (5 / 28) :=
sorry

end triangle_formation_probability_l633_633185


namespace letter_lock_unsuccessful_attempts_l633_633042

theorem letter_lock_unsuccessful_attempts :
  let total_combinations := 10^5 in
  total_combinations - 1 = 99999 := by
  sorry

end letter_lock_unsuccessful_attempts_l633_633042


namespace range_of_m_l633_633242

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4^x + m * 2^x + m^2 - 1 = 0) ↔ - (2 * Real.sqrt 3) / 3 ≤ m ∧ m < 1 :=
sorry

end range_of_m_l633_633242


namespace max_sum_of_quadrilateral_areas_l633_633765

noncomputable def max_sum_of_areas (n : ℕ) : ℝ :=
  1 / 3 * n^2 * (2 * n + 1) * (2 * n - 1)

theorem max_sum_of_quadrilateral_areas (n : ℕ) (h : n > 0)
  (S : set (ℕ × ℕ)) (hS : S = {p | p.1 < 2 * n ∧ p.2 < 2 * n })
  (F : finset (finset (ℕ × ℕ))) (hF : ∀ p ∈ S, ∃! Q ∈ F, p ∈ Q ∧ finset.card Q = 4) :
  Σ (Q ∈ F), finset.area Q = max_sum_of_areas n :=
sorry

end max_sum_of_quadrilateral_areas_l633_633765


namespace younger_person_age_l633_633441

/-- Let E be the present age of the elder person and Y be the present age of the younger person.
Given the conditions :
1) E - Y = 20
2) E - 15 = 2 * (Y - 15)
Prove that Y = 35. -/
theorem younger_person_age (E Y : ℕ) 
  (h1 : E - Y = 20) 
  (h2 : E - 15 = 2 * (Y - 15)) : 
  Y = 35 :=
sorry

end younger_person_age_l633_633441


namespace sub_vector_correct_l633_633691

/-- Define the vectors a and b -/
def a : ℝ × ℝ × ℝ := (-5, 1, 3)

def b : ℝ × ℝ × ℝ := (3, -2, 0)

/-- Define the scalar multiplication of a vector -/
def smul (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

/-- Define vector subtraction -/
def vsub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

/-- The theorem statement -/
theorem sub_vector_correct :
  vsub a (smul 4 b) = (-17, 9, 3) :=
  sorry

end sub_vector_correct_l633_633691


namespace numbers_with_digit_5_count_numbers_with_digit_5_l633_633274

theorem numbers_with_digit_5 (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 700) : 
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ d = 5) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m : ℕ, (n = m * 100 + d ∧ d = 5)) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m2 m1 : ℕ, (n = m2 * 1000 + m1 * 100 + d ∧ d = 5)) :=
sorry

theorem count_numbers_with_digit_5 : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 700 ∧ (numbers_with_digit_5 n sorry) }.to_finset.card = 214 := 
sorry

end numbers_with_digit_5_count_numbers_with_digit_5_l633_633274


namespace probability_Denis_Oleg_play_l633_633143

theorem probability_Denis_Oleg_play (n : ℕ) (h : n = 26) :
  let C := λ (n : ℕ), n * (n - 1) / 2 in
  (n - 1 : ℚ) / C n = 1 / 13 :=
by 
  sorry

end probability_Denis_Oleg_play_l633_633143


namespace eval_abc_l633_633186

theorem eval_abc (a b c : ℚ) (h1 : a = 1 / 2) (h2 : b = 3 / 4) (h3 : c = 8) :
  a^3 * b^2 * c = 9 / 16 :=
by
  sorry

end eval_abc_l633_633186


namespace turnip_bag_weights_l633_633073

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l633_633073


namespace prove_p_eq_m_cubed_sub_3mn_l633_633312

theorem prove_p_eq_m_cubed_sub_3mn (p q m n : ℝ) (α β : ℝ) 
  (hroots_mn : α + β = -m ∧ α * β = n)
  (hroots_pq : α^3 + β^3 = -p ∧ α^3 * β^3 = q) : 
  p = m^3 - 3 * m * n :=
by
  have h1 : α^3 + β^3 = (α + β) * (α^2 - α * β + β^2),
  {
    sorry
  },
  have h2 : α^2 + β^2 = (-m)^2 - 2 * n,
  {
    sorry
  },
  have h3 : α^2 - α * β + β^2 = m^2 - n,
  {
    sorry
  },
  have h4 : α^3 + β^3 = -m * (m^2 - n),
  {
    sorry
  },
  have h5 : -p = -m^3 + 3 * m * n,
  {
    sorry
  },
  exact sorry

end prove_p_eq_m_cubed_sub_3mn_l633_633312


namespace megan_final_balance_same_as_starting_balance_l633_633401

theorem megan_final_balance_same_as_starting_balance :
  let starting_balance : ℝ := 125
  let increased_balance := starting_balance * (1 + 0.25)
  let final_balance := increased_balance * (1 - 0.20)
  final_balance = starting_balance :=
by
  sorry

end megan_final_balance_same_as_starting_balance_l633_633401


namespace marble_probability_l633_633016

theorem marble_probability :
  let total_marbles : ℕ := 20
  let red_marbles : ℕ := 12
  let blue_marbles : ℕ := 8
  let total_probability : ℚ := (6 * ((12 * 11 * 8 * 7) / (20 * 19 * 18 * 17))) in
  total_probability = (1232 / 4845) :=
by
  sorry

end marble_probability_l633_633016


namespace painting_cost_conversion_l633_633779

def paintingCostInCNY (paintingCostNAD : ℕ) (usd_to_nad : ℕ) (usd_to_cny : ℕ) : ℕ :=
  paintingCostNAD * (1 / usd_to_nad) * usd_to_cny

theorem painting_cost_conversion :
  (paintingCostInCNY 105 7 6 = 90) :=
by
  sorry

end painting_cost_conversion_l633_633779


namespace not_snow_probability_l633_633863

theorem not_snow_probability :
  let p_snow : ℚ := 2 / 3,
      p_not_snow : ℚ := 1 - p_snow in
  (p_not_snow ^ 5) = 1 / 243 := by
  sorry

end not_snow_probability_l633_633863


namespace sufficiency_not_necessity_condition_l633_633006

theorem sufficiency_not_necessity_condition (a : ℝ) (h : a > 1) : (a^2 > 1) ∧ ¬(∀ x : ℝ, x^2 > 1 → x > 1) :=
by
  sorry

end sufficiency_not_necessity_condition_l633_633006


namespace isosceles_trapezoid_area_l633_633002

theorem isosceles_trapezoid_area (a b c : ℝ) (h : sqrt (c^2 - (b - a)^2 / 4) = √(c^2 - 9)) : (1 / 2) * (a + b) * h = 36 :=
by
  have h : h = 4 := by sorry
  rw h
  linarith

end isosceles_trapezoid_area_l633_633002


namespace calculate_n_l633_633949

def regular_polygon_interior_angle (m : ℕ) : ℝ :=
  (m - 2) * 180 / m

def regular_polygon_exterior_angle (m : ℕ) : ℝ :=
  180 - regular_polygon_interior_angle m

def condition (n : ℕ) : Prop :=
  3 * (120 / n) = 30

theorem calculate_n : ∃ n : ℕ, condition n ∧ n = 12 :=
by {
  use 12,
  split,
  { sorry },
  { refl }
}

end calculate_n_l633_633949


namespace repeating_decimal_sum_l633_633617

theorem repeating_decimal_sum :
  (0.\overline{2} : ℝ) + (0.\overline{02} : ℝ) + (0.\overline{0002} : ℝ) = 2426/9999 :=
begin
  sorry
end

end repeating_decimal_sum_l633_633617


namespace more_than_half_remains_l633_633937

def cubic_block := { n : ℕ // n > 0 }

noncomputable def total_cubes (b : cubic_block) : ℕ := b.val ^ 3

noncomputable def outer_layer_cubes (b : cubic_block) : ℕ := 6 * (b.val ^ 2) - 12 * b.val + 8

noncomputable def remaining_cubes (b : cubic_block) : ℕ := total_cubes b - outer_layer_cubes b

theorem more_than_half_remains (b : cubic_block) (h : b.val = 10) : remaining_cubes b > total_cubes b / 2 :=
by
  sorry

end more_than_half_remains_l633_633937


namespace distance_AF_l633_633750

theorem distance_AF (A B C D E F : ℝ×ℝ)
  (h1 : A = (0, 0))
  (h2 : B = (5, 0))
  (h3 : C = (5, 5))
  (h4 : D = (0, 5))
  (h5 : E = (2.5, 5))
  (h6 : ∃ k : ℝ, F = (k, 2 * k) ∧ dist F C = 5) :
  dist A F = Real.sqrt 5 :=
by
  sorry

end distance_AF_l633_633750


namespace equal_angles_in_trapezoid_l633_633736

-- Geometry entities and basic definitions
universe u

variable {Point : Type u} [HilbertPlane Point]

open Triangle

-- Given conditions in flower
variables {A B C D O S : Point} 
  (h_trapezoid : Trapezoid A B C D)
  (h_perp : IsPerpendicular CD (Line.mk A B))
  (h_diag_intersect : inter_of_diagonals h_trapezoid O)
  (h_S_diametric_opposite : IsDiametricallyOpposite O S (circumcircle O C D))

-- Prove that ∠BSC = ∠ASD
theorem equal_angles_in_trapezoid 
  (h_trapezoid : Trapezoid A B C D)
  (h_perp : IsPerpendicular CD (Line.mk A B))
  (h_diag_intersect : inter_of_diagonals h_trapezoid O)
  (h_S_diametric_opposite : IsDiametricallyOpposite O S (circumcircle O C D)) :
  angle B S C = angle A S D := sorry

end equal_angles_in_trapezoid_l633_633736


namespace florist_first_picking_l633_633038

theorem florist_first_picking (x : ℝ) (h1 : 37.0 + x + 19.0 = 72.0) : x = 16.0 :=
by
  sorry

end florist_first_picking_l633_633038


namespace river_depth_conditions_l633_633969

noncomputable def depth_beginning_may : ℝ := 15
noncomputable def depth_increase_june : ℝ := 11.25

theorem river_depth_conditions (d k : ℝ)
  (h1 : ∃ d, d = depth_beginning_may) 
  (h2 : 1.5 * d + k = 45)
  (h3 : k = 0.75 * d) :
  d = depth_beginning_may ∧ k = depth_increase_june :=
by
  have H : d = 15 := sorry
  have K : k = 11.25 := sorry
  exact ⟨H, K⟩

end river_depth_conditions_l633_633969


namespace circle_tangent_l633_633641

theorem circle_tangent {m : ℝ} (h : ∃ (x y : ℝ), (x - 3)^2 + (y - 4)^2 = 25 - m ∧ x^2 + y^2 = 1) :
  m = 9 :=
sorry

end circle_tangent_l633_633641


namespace no_snow_five_days_l633_633834

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633834


namespace length_CD_l633_633172

-- Defining the setting of a quadrilateral inscribed in a circle.
variable (O A B C D : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
noncomputable def radius := 300
noncomputable def sideLength := 300

-- Conditions
axiom circle : ∀ X ∈ {A, B, C, D}, dist O X = radius
axiom eq_AB : dist A B = sideLength
axiom eq_BC : dist B C = sideLength
axiom eq_AD : dist A D = sideLength

-- Prove CD = sideLength (300 meters)
theorem length_CD : dist C D = sideLength := by
  sorry

end length_CD_l633_633172


namespace primes_squared_mod_180_l633_633570

theorem primes_squared_mod_180 (p : ℕ) (hp_prime : Prime p) (hp_gt_5 : p > 5) :
  {r | ∃ k : ℕ, p^2 = 180 * k + r}.card = 2 :=
by
  sorry

end primes_squared_mod_180_l633_633570


namespace product_of_numbers_eq_120_l633_633465

theorem product_of_numbers_eq_120 (x y P : ℝ) (h1 : x + y = 23) (h2 : x^2 + y^2 = 289) (h3 : x * y = P) : P = 120 := 
sorry

end product_of_numbers_eq_120_l633_633465


namespace least_integer_k_l633_633467

def sequence (b : ℕ → ℝ) : Prop := 
  b 1 = 2 ∧ ∀ n ≥ 1, 3^(b (n+1) - b n) - 1 = 1 / (n + 1/4 : ℝ)

theorem least_integer_k (b : ℕ → ℝ) (k : ℕ) (h : sequence b) : k > 1 ∧ b k ∈ ℤ ↔ k = 11 :=
by
  sorry

end least_integer_k_l633_633467


namespace cardboard_box_height_l633_633395

theorem cardboard_box_height :
  ∃ (x : ℕ), x ≥ 0 ∧ 10 * x^2 + 4 * x ≥ 130 ∧ (2 * x + 1) = 9 :=
sorry

end cardboard_box_height_l633_633395


namespace inverse_variation_l633_633509

theorem inverse_variation (x y : ℝ) (h1 : 7 * y = 1400 / x^3) (h2 : x = 4) : y = 25 / 8 :=
  by
  sorry

end inverse_variation_l633_633509


namespace line_MN_tangent_fixed_circle_l633_633832

open EuclideanGeometry

variables {A B C M N O : Point}
variables {ABC : Triangle}
variables [h_ABC : AcuteAngledTriangle ABC]
variables (h_perp_bis : PerpendicularBisectorsIntersect AC BC M N)
variables (h_C_moves : C moves_along Circumcircle(ABC))
variables (h_positions : A B Fixed)

theorem line_MN_tangent_fixed_circle (h_conditions : 
    is_perpendicular_bisector M AC ∧ 
    is_perpendicular_bisector N BC ∧
    point_C_moves_along_circumcircle ABC C ∧ 
    in_one_half_plane C A B ∧
    points_A_B_fixed A B) : 
    exists (ω' : Circle), line_MN_tangent ω' :=
sorry

end line_MN_tangent_fixed_circle_l633_633832


namespace not_snowing_next_five_days_l633_633858

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l633_633858


namespace ones_digit_of_3_pow_52_l633_633901

theorem ones_digit_of_3_pow_52 : (3 ^ 52 % 10) = 1 := 
by sorry

end ones_digit_of_3_pow_52_l633_633901


namespace dusty_change_l633_633184

noncomputable def single_layer_cost := 4
noncomputable def single_layer_tax := 0.05
noncomputable def single_layer_quantity := 7

noncomputable def double_layer_cost := 7
noncomputable def double_layer_tax := 0.1
noncomputable def double_layer_quantity := 5

noncomputable def fruit_tart_cost := 5
noncomputable def fruit_tart_tax := 0.08
noncomputable def fruit_tart_quantity := 3

noncomputable def exchange_rate := 0.85
noncomputable def dollars_paid := 200

noncomputable def single_layer_total := (single_layer_cost + single_layer_cost * single_layer_tax) * single_layer_quantity
noncomputable def double_layer_total := (double_layer_cost + double_layer_cost * double_layer_tax) * double_layer_quantity
noncomputable def fruit_tart_total := (fruit_tart_cost + fruit_tart_cost * fruit_tart_tax) * fruit_tart_quantity

noncomputable def total_cost_euros := single_layer_total + double_layer_total + fruit_tart_total
noncomputable def total_dollars_to_euros := dollars_paid * exchange_rate
noncomputable def change_euros := total_dollars_to_euros - total_cost_euros
noncomputable def change_dollars := change_euros / exchange_rate

theorem dusty_change :
  change_dollars = 101.06 :=
sorry

end dusty_change_l633_633184


namespace cone_cylinder_volume_ratio_l633_633938

noncomputable def vol_ratio (α β : ℝ) : ℝ :=
  (cos α)^3 * (cos β)^3 / (3 * sin α * sin β * (cos (α + β))^2)

theorem cone_cylinder_volume_ratio (α β : ℝ) : 
  ∀ V1 V2 : ℝ,
  V1 = (1/3) * real.pi * (1 : ℝ)^2 * (real.cot β) →
  V2 = real.pi * ( (real.cos (α + β) / (real.cos α * real.cos β))^2) * real.tan α →
  (V1 / V2) = vol_ratio α β :=
by
  intro V1 V2 hV1 hV2
  rw [hV1, hV2]
  sorry

end cone_cylinder_volume_ratio_l633_633938


namespace total_stones_is_60_l633_633796

-- Define the conditions as variables and equations
variables {x x1 x2 x3 x4 x5 : ℕ}
variables (h1 : x3 = x)
variables (h2 : x5 = 6 * x3)
variables (h3 : x2 = 2 * (x3 + x5))
variables (h4 : x1 = x5 / 3)
variables (h5 : x1 = x4 - 10)
variables (h6 : x4 = x2 / 2)

-- Define the total number of stones
def total_stones : ℕ := x1 + x2 + x3 + x4 + x5

-- The proof problem statement
theorem total_stones_is_60 : total_stones = 60 :=
by
  sorry

end total_stones_is_60_l633_633796


namespace correct_option_of_polynomial_operations_l633_633497

-- Definitions and conditions
theorem correct_option_of_polynomial_operations (x : ℝ) :
  (2 * x^2 - x^2 = x^2) ∧ ¬(x^2 + x^3 = x^5) ∧ ¬(x^2 * x^3 = x^6) ∧ ¬((x^2)^3 = x^5) :=
by {
  sorry,
}

end correct_option_of_polynomial_operations_l633_633497


namespace solve_for_t_requires_numerical_methods_l633_633709

variables {g V0 a t S V : ℝ}

-- Definitions for the conditions:
def velocity_eq (V t: ℝ) (g V0: ℝ) : Prop := V = g * t + V0
def displacement_eq (S t: ℝ) (g V0 a: ℝ) : Prop := S = (1 / 2) * g * t^2 + V0 * t + (1 / 3) * a * t^3

-- Goal: Prove that solving for t generally requires numerical methods.
theorem solve_for_t_requires_numerical_methods 
  (h1: velocity_eq V t g V0)
  (h2: displacement_eq S t g V0 a) :
  ¬ ∃ (t_expression : ℝ → ℝ → ℝ → ℝ), ∀ (S g V0 a : ℝ), t = t_expression S g V0 a := by
  sorry

end solve_for_t_requires_numerical_methods_l633_633709


namespace count_5_in_range_1_to_700_l633_633285

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  n.digits 10 |>.contains d

def count_numbers_with_digit (d : ℕ) (m : ℕ) : ℕ :=
  (List.range' 1 m) |>.filter (contains_digit d) |>.length

theorem count_5_in_range_1_to_700 : count_numbers_with_digit 5 700 = 214 := by
  sorry

end count_5_in_range_1_to_700_l633_633285


namespace locus_equation_rectangle_perimeter_greater_l633_633332

open Real

theorem locus_equation (P : ℝ × ℝ) : 
  (abs P.2 = sqrt (P.1 ^ 2 + (P.2 - 1 / 2) ^ 2)) → (P.2 = P.1 ^ 2 + 1 / 4) :=
by
  intro h
  sorry

theorem rectangle_perimeter_greater (A B C D : ℝ × ℝ) :
  (A.2 = A.1 ^ 2 + 1 / 4) ∧ 
  (B.2 = B.1 ^ 2 + 1 / 4) ∧ 
  (C.2 = C.1 ^ 2 + 1 / 4) ∧ 
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) → 
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
by
  intro h
  sorry

end locus_equation_rectangle_perimeter_greater_l633_633332


namespace olympic_medals_distribution_l633_633478

theorem olympic_medals_distribution :
  ∀ (sprinters : Finset ℕ) (americans : Finset ℕ),
    sprinters.card = 10 ∧ americans.card = 4 ∧ americans ⊆ sprinters →
    let non_americans := sprinters \ americans in
    (multiset.card (non_americans.val.powerset.length_eq 3) * 6! / 3! / 3!) +
    (americans.card * 3 * multiset.card (non_americans.val.powerset.length_eq 2) * 2! / 2!) = 480 :=
begin
  sorry
end

end olympic_medals_distribution_l633_633478


namespace smallest_sum_of_two_3digit_numbers_l633_633492
open Nat

theorem smallest_sum_of_two_3digit_numbers : 
  ∃ (a b : ℕ), 
  a < 1000 ∧ b < 1000 ∧ 
  (∀ x y, x ∈ {1, 2, 3, 7, 8, 9} → y ∈ {1, 2, 3, 7, 8, 9} → x ≠ y → 
  x ∈ digits 10 a ∧ y ∈ digits 10 b) ∧ 
  a + b = 417 := 
begin
  -- proof goes here
  sorry
end

end smallest_sum_of_two_3digit_numbers_l633_633492


namespace employee_percentage_six_years_or_more_l633_633033

theorem employee_percentage_six_years_or_more
  (x : ℕ)
  (total_employees : ℕ := 36 * x)
  (employees_6_or_more : ℕ := 8 * x) :
  (employees_6_or_more : ℚ) / (total_employees : ℚ) * 100 = 22.22 := 
sorry

end employee_percentage_six_years_or_more_l633_633033


namespace fractional_exponent_calculation_l633_633157

variables (a b : ℝ) -- Define a and b as real numbers
variable (ha : a > 0) -- Condition a > 0
variable (hb : b > 0) -- Condition b > 0

theorem fractional_exponent_calculation :
  (a^(2 * b^(1/4)) / (a * b^(1/2))^(1/2)) = a^(1/2) :=
by
  sorry -- Proof is not required, skip with sorry

end fractional_exponent_calculation_l633_633157


namespace prime_quadruple_solution_l633_633194

-- Define the problem statement in Lean
theorem prime_quadruple_solution :
  ∀ (p q r : ℕ) (n : ℕ),
    Prime p → Prime q → Prime r → n > 0 →
    p^2 = q^2 + r^n →
    (p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4) :=
by
  sorry -- Proof omitted

end prime_quadruple_solution_l633_633194


namespace number_of_minutes_away_l633_633958

noncomputable def time_away (n : ℚ) :=
  |(150 : ℚ) - (11 * n / 2)| = 120

theorem number_of_minutes_away 
  (n₁ n₂ : ℚ) 
  (h₁ : time_away n₁) 
  (h₂ : time_away n₂) 
  (h_neq : n₁ ≠ n₂):
  n₂ - n₁ = 480 / 11 := 
sorry

end number_of_minutes_away_l633_633958


namespace original_denominator_l633_633551

theorem original_denominator (d : ℤ) : 
  (∀ n : ℤ, n = 3 → (n + 8) / (d + 8) = 1 / 3) → d = 25 :=
by
  intro h
  specialize h 3 rfl
  sorry

end original_denominator_l633_633551


namespace slope_of_line_intersecting_circle_l633_633253

theorem slope_of_line_intersecting_circle (k : ℝ) :
  (∃ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A.2 = k * A.1 + 1) ∧ (B.2 = k * B.1 + 1) ∧ (∠ complex.arg (complex.of_real A.1 + A.2 * I) (complex.of_real B.1 + B.2 * I) = real.pi / 3)) → 
  (k = √3/3 ∨ k = -√3/3) :=
by
  intros
  sorry

end slope_of_line_intersecting_circle_l633_633253


namespace find_m_l633_633154

noncomputable def distance_ellipse_foci (m : ℝ) : ℝ :=
  real.sqrt (4 - m^2)

noncomputable def distance_hyperbola_foci (m : ℝ) : ℝ :=
  real.sqrt (m + 2)

theorem find_m (m : ℝ) :
  distance_ellipse_foci m = distance_hyperbola_foci m → m = 1 :=
by
  sorry

end find_m_l633_633154


namespace reflect_translation_l633_633874

variable {P P1 P12 P123 P1234 V1 V2 V3 V4 : Point}

-- Assume P is an arbitrary point
-- V1, V2, V3, V4 are vertices of a regular tetrahedron

-- Define sequential reflections
def reflect (P : Point) (V : Point) : Point := sorry  -- some reflection function

-- Define the specific reflections
def P1 : Point := reflect P V1
def P12 : Point := reflect P1 V2
def P123 : Point := reflect P12 V3
def P1234 : Point := reflect P123 V4

-- Define the translation
def translation (P : Point) (V1 V2 V3 V4 : Point) : Point :=
  P + 2 * (overrightarrow V1 V2 + overrightarrow V3 V4)

-- The theorem we want to prove
theorem reflect_translation :
  P1234 = translation P V1 V2 V3 V4 :=
sorry

end reflect_translation_l633_633874


namespace denis_and_oleg_probability_l633_633127

noncomputable def probability_denisolga_play_each_other (n : ℕ) (i j : ℕ) (h1 : n = 26) (h2 : i ≠ j) : ℚ :=
  let number_of_pairs := (n * (n - 1)) / 2
  in (n - 1) / number_of_pairs

theorem denis_and_oleg_probability :
  probability_denisolga_play_each_other 26 1 2 rfl dec_trivial = 1 / 13 :=
sorry

end denis_and_oleg_probability_l633_633127


namespace power_calc_l633_633306

noncomputable def n := 2 ^ 0.3
noncomputable def b := 13.333333333333332

theorem power_calc : n ^ b = 16 := by
  sorry

end power_calc_l633_633306


namespace reflection_point_l633_633543

theorem reflection_point 
  (A B : ℝ × ℝ)
  (hA : A = (-2, 2))
  (hB : B = (0, 1)) :
  ∃ C : ℝ × ℝ, C = (-2/3, 0) ∧ 
  (let a := (C.1 : ℝ) in 
  (2 - 0)/(-2 - a) = -((1 - 0) / (0 - a))) :=
by
  -- Here we need to find the coordinates of point C on the x-axis.
  sorry

end reflection_point_l633_633543


namespace average_speed_is_correct_l633_633519

noncomputable def average_speed_for_220_km : ℝ := 
  let x := 40
  in x

theorem average_speed_is_correct :
  let total_distance := 250
  let total_time := 6
  let distance_at_60_kmph := 30
  ∃ (x : ℝ), 
    (220 / x) + (30 / 60) = 6 ∧
    x = average_speed_for_220_km :=
begin
  sorry
end

end average_speed_is_correct_l633_633519


namespace sector_area_correct_l633_633234

def central_angle : ℝ := (2 / 5) * Real.pi
def radius : ℝ := 5
def sector_area : ℝ := (1 / 2) * central_angle * radius^2

theorem sector_area_correct : sector_area = 5 * Real.pi :=
by
  sorry

end sector_area_correct_l633_633234


namespace sum_series_l633_633162

theorem sum_series (n : ℕ) : 
  let S := (range n).sum (λ k, (k+1) * (k+2) * (k+3)) in 
  S = (1 / 4) * n * (n + 1) * (n + 2) * (n + 3) := 
begin
  sorry
end

end sum_series_l633_633162


namespace number_of_subjects_l633_633552

variable (P C M : ℝ)

-- Given conditions
def conditions (P C M : ℝ) : Prop :=
  (P + C + M) / 3 = 75 ∧
  (P + M) / 2 = 90 ∧
  (P + C) / 2 = 70 ∧
  P = 95

-- Proposition with given conditions and the conclusion
theorem number_of_subjects (P C M : ℝ) (h : conditions P C M) : 
  (∃ n : ℕ, n = 3) :=
by
  sorry

end number_of_subjects_l633_633552


namespace part1_part2_l633_633303

-- Part 1
def is_R2_sequence (a : ℕ → ℝ) : Prop := 
  (∀ n : ℕ, a (n + 1) ≥ a n) ∧ 
  (∀ n : ℕ, n > 2 → a (n - 2) + a (n + 2) = 2 * a n)

def a_n (n : ℕ) : ℝ := if odd n then 2 * n - 1 else 2 * n

theorem part1 : is_R2_sequence a_n :=
sorry

-- Part 2
def is_R3_sequence (b : ℕ → ℝ) : Prop := 
  (∀ n : ℕ, b (n + 1) ≥ b n) ∧ 
  (∀ n : ℕ, n > 3 → b (n - 3) + b (n + 3) = 2 * b n)

def forms_arithmetic_sequence (b : ℕ → ℝ) (k : ℕ) : Prop :=
  ∃ p > 1, b (3 * p - 3) + b (3 * p + 3) = 2 * (b (3 * p - 1) + b (3 * p + 1))

theorem part2 (b : ℕ → ℝ) 
  (h1 : is_R3_sequence b) 
  (h2 : forms_arithmetic_sequence b 3) : 
  ∀ m n : ℕ, b m + b n = 2 * b ((m + n) / 2) := 
sorry

end part1_part2_l633_633303


namespace total_profit_is_correct_l633_633145

-- Definitions based on conditions
def investment_A : ℝ := 27000
def investment_B : ℝ := 72000
def investment_C : ℝ := 81000
def investment_D : ℝ := 63000
def investment_E : ℝ := 45000

def ratio_A : ℝ := 3
def ratio_B : ℝ := 5
def ratio_C : ℝ := 6
def ratio_D : ℝ := 4
def ratio_E : ℝ := 3

def share_C : ℝ := 60000

-- Definition of the value of each ratio unit based on C's share
def value_per_unit : ℝ := share_C / ratio_C

-- Total ratio units
def total_ratio_units : ℝ := ratio_A + ratio_B + ratio_C + ratio_D + ratio_E

-- Total profit calculation
def total_profit : ℝ := total_ratio_units * value_per_unit

theorem total_profit_is_correct : total_profit = 210000 := by
  sorry

end total_profit_is_correct_l633_633145


namespace find_abs_3h_minus_4k_l633_633313

theorem find_abs_3h_minus_4k
  (h k : ℤ)
  (factor1_eq_zero : 3 * (-3)^3 - h * (-3) - 3 * k = 0)
  (factor2_eq_zero : 3 * 2^3 - h * 2 - 3 * k = 0) :
  |3 * h - 4 * k| = 615 :=
by
  sorry

end find_abs_3h_minus_4k_l633_633313


namespace intersection_eq_l633_633262

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 3}
def B : Set ℝ := {x | Real.log 2 x > 0}
def CUB : Set ℝ := {x | x <= 1}

theorem intersection_eq :
  A ∩ CUB = {x | x <= 1} :=
by
  sorry

end intersection_eq_l633_633262


namespace DollOutfit_l633_633485

variables (VeraDress OlyaCoat VeraCoat NinaCoat : Prop)
axiom FirstAnswer : (VeraDress ∧ ¬OlyaCoat) ∨ (¬VeraDress ∧ OlyaCoat)
axiom SecondAnswer : (VeraCoat ∧ ¬NinaCoat) ∨ (¬VeraCoat ∧ NinaCoat)
axiom OnlyOneTrueFirstAnswer : (VeraDress ∨ OlyaCoat) ∧ ¬(VeraDress ∧ OlyaCoat)
axiom OnlyOneTrueSecondAnswer : (VeraCoat ∨ NinaCoat) ∧ ¬(VeraCoat ∧ NinaCoat)

theorem DollOutfit :
  VeraDress ∧ NinaCoat ∧ ¬OlyaCoat ∧ ¬VeraCoat ∧ ¬NinaCoat :=
sorry

end DollOutfit_l633_633485


namespace swim_speed_current_l633_633123

theorem swim_speed_current (swim_still_water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (swim_still_water_speed_eq : swim_still_water_speed = 3) 
  (distance_eq : distance = 8) 
  (time_eq : time = 5) : 
  let effective_speed := distance / time in
  let current_speed := swim_still_water_speed - effective_speed in
  current_speed = 1.4 :=
by 
  -- Define the effective speed against the current
  let effective_speed := distance / time
  -- Define the speed of the river's current
  let current_speed := swim_still_water_speed - effective_speed
  have effective_speed_eq : effective_speed = 8 / 5 := by rw [distance_eq, time_eq]
  have current_speed_eq : current_speed = swim_still_water_speed - (8 / 5) := by rw effective_speed_eq
  rw [swim_still_water_speed_eq] at current_speed_eq
  have current_speed_val : current_speed = 3 - 1.6 := by norm_num [current_speed_eq]
  show current_speed = 1.4, by norm_num [current_speed_val]

end swim_speed_current_l633_633123


namespace time_difference_l633_633694

-- Definitions for the conditions
def blocks_to_office : Nat := 12
def walk_time_per_block : Nat := 1 -- time in minutes
def bike_time_per_block : Nat := 20 / 60 -- time in minutes, converted from seconds

-- Definitions for the total times
def walk_time : Nat := blocks_to_office * walk_time_per_block
def bike_time : Nat := blocks_to_office * bike_time_per_block

-- Theorem statement
theorem time_difference : walk_time - bike_time = 8 := by
  -- Proof omitted
  sorry

end time_difference_l633_633694


namespace right_triangle_of_sin_cos_eq_l633_633739

variable {A B C : ℝ}  
variable {α β γ : Angle}

axiom triangle_angle_sum (A B C : ℝ) : A + B + C = π
axiom sin_eq_1 (α : Angle) : sin α = 1 → α = (π/2 : ℝ)

/-- If sin A cos B = 1 - cos A sin B in triangle ABC, then it is a right triangle. -/
theorem right_triangle_of_sin_cos_eq (h : sin A * cos B = 1 - cos A * sin B) (hsum : A + B + C = π) : C = π / 2 := 
by
  have h1 : sin (A + B) = 1 := sorry -- sin(A + B) = 1
  exact sin_eq_1 (Angle.ofReal (A + B))
  have h2 : A + B = π / 2 := sorry -- angle A + B should be π / 2
  have h3 : A + B + C = π := hsum
  exact sub_eq_of_eq_add' h3
  exact add_sub_cancel'_right h3


end right_triangle_of_sin_cos_eq_l633_633739


namespace find_lambda_l633_633231

open Real

theorem find_lambda (λ p y1 y2 : ℝ) (hλ : λ ≥ 0) (hp : 0 < p) 
(a_def : ∃ (xa ya : ℝ), ya = xa^2 ∧ A = (xa, ya) ∧ xa = y1^2 / (2 * p))
(b_def : ∃ (xb yb : ℝ), yb = xb^2 ∧ B = (xb, yb) ∧ xb = y2^2 / (2 * p))
(min_dot_product_zero : 
  let EA := (y1^2 / (2 * p) + λ, y1),
      EB := (y2^2 / (2 * p) + λ, y2)
  in (EA.1 * EB.1 + EA.2 * EB.2) = 0 ) : λ = p / 2 :=
sorry

end find_lambda_l633_633231


namespace denis_and_oleg_probability_l633_633124

noncomputable def probability_denisolga_play_each_other (n : ℕ) (i j : ℕ) (h1 : n = 26) (h2 : i ≠ j) : ℚ :=
  let number_of_pairs := (n * (n - 1)) / 2
  in (n - 1) / number_of_pairs

theorem denis_and_oleg_probability :
  probability_denisolga_play_each_other 26 1 2 rfl dec_trivial = 1 / 13 :=
sorry

end denis_and_oleg_probability_l633_633124


namespace circle_center_sum_l633_633605

theorem circle_center_sum (x y : ℝ) (h : (x - 5)^2 + (y - 2)^2 = 38) : x + y = 7 := 
  sorry

end circle_center_sum_l633_633605


namespace locus_equation_perimeter_greater_l633_633339

-- Define the conditions under which the problem is stated
def distance_to_x_axis (P : ℝ × ℝ) : ℝ := abs P.2
def distance_to_point (P : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- P is a point on the locus W if the distance to the x-axis is equal to the distance to (0, 1/2)
def on_locus (P : ℝ × ℝ) : Prop := 
  distance_to_x_axis P = distance_to_point P (0, 1/2)

-- Prove that the equation of W is y = x^2 + 1/4 given the conditions
theorem locus_equation (P : ℝ × ℝ) (h : on_locus P) : 
  P.2 = P.1^2 + 1/4 := 
sorry

-- Assume rectangle ABCD with three points on W
def point_on_w (P : ℝ × ℝ) : Prop := 
  P.2 = P.1^2 + 1/4

def points_form_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 ≠ B.1 ∧ B.1 ≠ C.1 ∧ C.1 ≠ D.1 ∧ D.1 ≠ A.1 ∧
  A.2 ≠ B.2 ∧ B.2 ≠ C.2 ∧ C.2 ≠ D.2 ∧ D.2 ≠ A.2

-- P1, P2, and P3 are three points on the locus W
def points_on_locus (A B C : ℝ × ℝ) : Prop := 
  point_on_w A ∧ point_on_w B ∧ point_on_w C

-- Prove the perimeter of rectangle ABCD with three points on W is greater than 3sqrt(3)
theorem perimeter_greater (A B C D : ℝ × ℝ) 
  (h1 : points_on_locus A B C) 
  (h2 : points_form_rectangle A B C D) : 
  2 * (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
       real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) > 
  3 * real.sqrt 3 := 
sorry

end locus_equation_perimeter_greater_l633_633339


namespace count_digit_five_1_to_700_l633_633292

def contains_digit_five (n : ℕ) : Prop :=
  n.digits 10 ∈ [5]

def count_up_to (n : ℕ) (p : ℕ → Prop) : ℕ :=
  (finset.range n).count p

theorem count_digit_five_1_to_700 : count_up_to 701 contains_digit_five = 52 := sorry

end count_digit_five_1_to_700_l633_633292


namespace sum_reciprocal_S_n_l633_633745

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Condition definitions
axiom arithmetic_sequence (a : ℕ → ℝ) : Prop
axiom sum_of_sequence (S : ℕ → ℝ) : Prop
axiom sum_relation : ∀ n, S n = n * (n + 1)

-- Specific conditions
axiom condition_1 : a 9 = 0.5 * a 12 + 6
axiom condition_2 : a 2 = 4

-- Question: Sum of the first 10 terms of the sequence {1 / S_n}
theorem sum_reciprocal_S_n (h_seq : arithmetic_sequence a) (h_sum : sum_of_sequence S) 
                           (h_sum_rel : sum_relation) 
                           (h_cond1 : condition_1) (h_cond2 : condition_2) : 
  ∑ i in Finset.range 10, (1 / S (i + 1)) = 10 / 11 := 
sorry

end sum_reciprocal_S_n_l633_633745


namespace smallest_x_mod_7_one_sq_l633_633014

theorem smallest_x_mod_7_one_sq (x : ℕ) (h : 1 < x) (hx : (x * x) % 7 = 1) : x = 6 :=
  sorry

end smallest_x_mod_7_one_sq_l633_633014


namespace third_chapter_pages_l633_633028

theorem third_chapter_pages (x : ℕ) (h : 18 = x + 15) : x = 3 :=
by
  sorry

end third_chapter_pages_l633_633028


namespace begin_of_winter_conditions_l633_633960

/-- 
According to meteorological standards, if the daily average temperature is below 10°C 
for 5 consecutive days, it is considered the beginning of winter.
The correct samples definitely meeting this criteria are:
B: The mean is less than 4 and the range is less than or equal to 3
D: The mode is 5 and the range is less than or equal to 4
-/
theorem begin_of_winter_conditions {temps : ℕ → ℝ} (h1 : ∀ i, temps i < 10)
  (B : (temps.sum / 5 < 4) ∧ (temps.max - temps.min ≤ 3))
  (D : (∃ m, m = 5 ∧ ∀ t ∈ temps, t ≤ m + 4)) : 
  B ∨ D :=
sorry

end begin_of_winter_conditions_l633_633960


namespace range_of_m_l633_633245

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2 * x + 5

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Icc (-1 : ℝ) 2, f x < m) ↔ 7 < m :=
begin
  sorry
end

end range_of_m_l633_633245


namespace bus_driver_total_compensation_l633_633931

-- Definitions of conditions
def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def overtime_rate : ℝ := regular_rate * 1.75
def total_hours : ℝ := 65
def total_compensation : ℝ := (regular_rate * regular_hours) + (overtime_rate * (total_hours - regular_hours))

-- Theorem stating the total compensation
theorem bus_driver_total_compensation : total_compensation = 1340 :=
by
  sorry

end bus_driver_total_compensation_l633_633931


namespace precise_location_on_number_line_l633_633150

noncomputable def sqrt_of_sixteen : ℕ := 4
noncomputable def neg_sqrt_of_three : ℝ := -real.sqrt 3
noncomputable def cbrt_of_eight : ℕ := 2
noncomputable def seven_thirds : ℚ := 7 / 3

theorem precise_location_on_number_line :
  ∃ (a b c : ℝ), a = sqrt_of_sixteen ∧ b = cbrt_of_eight ∧ c = seven_thirds :=
by
  use [sqrt_of_sixteen, cbrt_of_eight, seven_thirds]
  sorry

end precise_location_on_number_line_l633_633150


namespace infinite_product_eq_cbrt_27_l633_633619

noncomputable def product_series : ℝ :=
  ∏' n : ℕ+ , (3 : ℝ)^((n : ℝ) / 3^(n : ℝ))

theorem infinite_product_eq_cbrt_27 : 
  product_series = real.exp ((3 : ℝ) / 4 * real.log 3) :=
by
  sorry

end infinite_product_eq_cbrt_27_l633_633619


namespace probability_no_snow_l633_633841

theorem probability_no_snow {
  let p_snow : ℚ := 2/3,     -- Probability of snow on any one day
  let p_no_snow : ℚ := 1 - p_snow,  -- Probability of no snow on any one day
  let p_no_snow_5_days : ℚ := p_no_snow ^ 5  -- Probability of no snow on any of the five days
} : p_no_snow_5_days = 1 / 243 := by
  sorry

end probability_no_snow_l633_633841


namespace turnip_weights_are_13_or_16_l633_633097

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l633_633097


namespace quadratic_function_min_value_l633_633744

noncomputable def f (a h k : ℝ) (x : ℝ) : ℝ :=
  a * (x - h) ^ 2 + k

theorem quadratic_function_min_value :
  ∀ (f : ℝ → ℝ) (n : ℕ),
  (f n = 13) ∧ (f (n + 1) = 13) ∧ (f (n + 2) = 35) →
  (∃ k, k = 2) :=
  sorry

end quadratic_function_min_value_l633_633744


namespace equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l633_633340

theorem equation_of_W (P : ℝ × ℝ) :
  let x := P.1 in let y := P.2 in
  |y| = real.sqrt (x^2 + (y - 1/2)^2) ↔ y = x^2 + 1/4 :=
by sorry

theorem rectangle_perimeter_greater_than_3sqrt3 {A B C D : ℝ × ℝ}
  (hA : A.2 = A.1^2 + 1/4) (hB : B.2 = B.1^2 + 1/4) (hC : C.2 = C.1^2 + 1/4)
  (hAB_perp_BC : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  2 * ((real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) + (real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2))) > 3 * real.sqrt 3 :=
by sorry

end equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l633_633340


namespace percentage_shoes_polished_l633_633583

theorem percentage_shoes_polished (total_pairs : ℕ) (shoes_to_polish : ℕ)
  (total_individual_shoes : ℕ := total_pairs * 2)
  (shoes_polished : ℕ := total_individual_shoes - shoes_to_polish)
  (percentage_polished : ℚ := (shoes_polished : ℚ) / total_individual_shoes * 100) :
  total_pairs = 10 → shoes_to_polish = 11 → percentage_polished = 45 :=
by
  intros hpairs hleft
  sorry

end percentage_shoes_polished_l633_633583


namespace probability_event_A_occurs_1400_times_2400_trials_l633_633630

noncomputable theory

-- Definitions for the problem
def num_trials : ℕ := 2400
def successes : ℕ := 1400
def prob_success : ℝ := 0.6
def prob_failure : ℝ := 1 - prob_success

-- Standard normal density function
def std_normal_density (x : ℝ) : ℝ :=
  (1 / (real.sqrt (2 * real.pi))) * real.exp (- (x ^ 2 / 2))

-- De Moivre-Laplace theorem formula for binomial distribution
def de_moivre_laplace (n k : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - p in
  let x := (k - n * p) / real.sqrt (n * p * q) in
  (1 / real.sqrt (n * p * q)) * std_normal_density x

theorem probability_event_A_occurs_1400_times_2400_trials :
  de_moivre_laplace num_trials successes prob_success = 0.0041 := by
  sorry

end probability_event_A_occurs_1400_times_2400_trials_l633_633630


namespace union_eq_self_l633_633260

open Set

variable (U : Type) [Fintype U]

def M : Set U := {a, b, c, d, e}
def N : Set U := {b, d, e}

theorem union_eq_self : M ∪ N = M := 
by
  sorry

end union_eq_self_l633_633260


namespace turnip_bag_weight_l633_633061

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l633_633061


namespace james_writing_daily_to_each_person_l633_633746

theorem james_writing_daily_to_each_person :
  (let pages_per_hour := 10 in
   let hours_per_week := 7 in
   let days_per_week := 7 in
   let people := 2 in
   (pages_per_hour * hours_per_week) / (days_per_week * people) = 5) := 
by
  sorry

end james_writing_daily_to_each_person_l633_633746


namespace gcd_polynomial_is_25_l633_633658

theorem gcd_polynomial_is_25 (b : ℕ) (h : ∃ k : ℕ, b = 2700 * k) :
  Nat.gcd (b^2 + 27 * b + 75) (b + 25) = 25 :=
by 
    sorry

end gcd_polynomial_is_25_l633_633658


namespace no_snow_probability_l633_633846

theorem no_snow_probability (p_snow : ℚ) (h : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  (p_no_snow ^ 5 = 1 / 243) :=
by
  sorry

end no_snow_probability_l633_633846


namespace sum_arith_series_correct_l633_633977

-- Define the arithmetic series with first term, last term, and common difference.
def arith_series (a1 an d : ℝ) (n : ℕ) : list ℝ :=
  (list.range n).map (λ i, a1 + i * d)

-- Define the formula for the sum of an arithmetic series.
def sum_arith_series (a1 an d : ℝ) (n : ℕ) : ℝ :=
  (n * (a1 + an)) / 2

-- Prove that the sum of the given arithmetic series is 1710.
theorem sum_arith_series_correct :
  ∃ n, a1 = 15 ∧ an = 30 ∧ d = 0.2 ∧ n = 76 ∧ sum_arith_series 15 30 0.2 76 = 1710 :=
by sorry

end sum_arith_series_correct_l633_633977


namespace interval_solution_inequality_l633_633428

theorem interval_solution_inequality (x : ℝ) : 
  (1/3) ^ (x^2 - 8) > 3 ^ (-2 * x) → -2 < x ∧ x < 4 :=
by
  sorry

end interval_solution_inequality_l633_633428


namespace linear_dependence_condition_l633_633468

theorem linear_dependence_condition (k : ℝ) :
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 1 + b * 4 = 0) ∧ (a * 2 + b * k = 0) ∧ (a * 1 + b * 2 = 0)) ↔ k = 8 := 
by sorry

end linear_dependence_condition_l633_633468


namespace max_proj_area_min_proj_area_l633_633645

structure Tetrahedron := 
  (vertices : Fin 4 → (ℝ × ℝ × ℝ))
  (regular : ∀ i j k l, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
     dist (vertices i) (vertices j) = dist (vertices i) (vertices k)
     ∧ dist (vertices i) (vertices j) = dist (vertices i) (vertices l)
     ∧ dist (vertices i) (vertices j) = dist (vertices j) (vertices k))

def maxProjectedAreaPlane (T : Tetrahedron) : (ℝ × ℝ × ℝ) :=
  -- Point M in the problem
  sorry -- Exact definition of M based on the tetrahedron structure

def minProjectedAreaPlane (T : Tetrahedron) : (ℝ × ℝ × ℝ) :=
  -- Point L in the problem
  sorry -- Exact definition of L based on the tetrahedron structure

theorem max_proj_area (T : Tetrahedron) : 
  ∀ plane, isPerpendicularPlane T plane → 
  plane = maxProjectedAreaPlane T → 
  projectedArea T plane = maxProjectedArea T :=
sorry

theorem min_proj_area (T : Tetrahedron) : 
  ∀ plane, isPerpendicularPlane T plane → 
  plane = minProjectedAreaPlane T → 
  projectedArea T plane = minProjectedArea T :=
sorry

end max_proj_area_min_proj_area_l633_633645


namespace first_discount_l633_633932

noncomputable theory

def final_price (P : ℝ) (D1 D2 D3 : ℝ) : ℝ :=
  P * (1 - D1 / 100) * (1 - D2 / 100) * (1 - D3 / 100)

theorem first_discount (P : ℝ) (D1 D2 D3 final : ℝ) (h1 : P = 9502.92) (h2 : D2 = 10) (h3 : D3 = 5) (h4 : final = 6500) :
  D1 = 20.03 :=
by
  sorry

end first_discount_l633_633932


namespace find_2n_plus_m_l633_633825

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 2 * n + m = 36 :=
sorry

end find_2n_plus_m_l633_633825


namespace count_three_digit_numbers_l633_633161

theorem count_three_digit_numbers : 
  (∃ digits : Finset ℕ, digits = {1, 2, 3, 4}) →
  (∃ count : ℕ, count = 64) := 
by
  intro h
  use 64
  sorry

end count_three_digit_numbers_l633_633161


namespace chapters_per_book_l633_633403

theorem chapters_per_book (books : ℕ) (total_chapters : ℕ) (books_eq : books = 4) (total_chapters_eq : total_chapters = 68) : 
  total_chapters / books = 17 :=
by
  rw [books_eq, total_chapters_eq]
  norm_num

end chapters_per_book_l633_633403


namespace find_constant_a_l633_633621

theorem find_constant_a :
  ∃ (a : ℂ), (∀ (s d : ℂ), (s - d) ≠ s + d ∧ (s - d) + s + (s + d) = 9 ∧
    (s - d) * s * (s + d) = -a ∧
    ((s - d) * s + (s - d) * (s + d) + s * (s + d) = 27) ∧
    (s - d, s, s + d).complex_nonreal) → a = 9 :=
begin
  sorry
end

end find_constant_a_l633_633621


namespace prime_squares_mod_180_l633_633577

theorem prime_squares_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  ∃ r ∈ {1, 49}, ∀ q ∈ {1, 49}, (q ≡ p^2 [MOD 180]) → q = r :=
by
  sorry

end prime_squares_mod_180_l633_633577


namespace arrange_chairs_and_stools_l633_633530

-- Definition of the mathematical entities based on the conditions
def num_ways_to_arrange (women men : ℕ) : ℕ :=
  let total := women + men
  (total.factorial) / (women.factorial * men.factorial)

-- Prove that the arrangement yields the correct number of ways
theorem arrange_chairs_and_stools :
  num_ways_to_arrange 7 3 = 120 := by
  -- The specific definitions and steps are not to be included in the Lean statement;
  -- hence, adding a placeholder for the proof.
  sorry

end arrange_chairs_and_stools_l633_633530


namespace transformation_impossible_l633_633685

theorem transformation_impossible :
  let initial_set := {5, 12, 18}
  let target_set := {3, 13, 20}
  let operation (a b : ℝ) : (ℝ × ℝ) := (√2 / 2 * (a + b), √2 / 2 * (a - b))
  ∀ seq : list (ℝ × ℝ),
    (∀ (p : ℝ × ℝ), p ∈ seq → ∃ a b : ℝ, (a, b) ∈ initial_set ∧ p = operation a b) →
      initial_set ≠ target_set := by
  /- proof goes here -/
  sorry

end transformation_impossible_l633_633685


namespace multiply_fractions_l633_633976

theorem multiply_fractions :
  (2/3) * (4/7) * (9/11) * (5/8) = 15/77 :=
by
  -- It is just a statement, no need for the proof steps here
  sorry

end multiply_fractions_l633_633976


namespace eval_expression_l633_633165

theorem eval_expression : 4 * Real.sin (45 * Real.pi / 180) + (Real.sqrt 2 - Real.pi)^0 - Real.sqrt 8 + (1/3)^-2 = 10 := 
by
  sorry

end eval_expression_l633_633165


namespace reciprocal_of_neg6_l633_633869

theorem reciprocal_of_neg6 : 1 / (-6 : ℝ) = -1 / 6 := 
sorry

end reciprocal_of_neg6_l633_633869


namespace sum_identity_l633_633414

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  if h : k ≤ n then nat.choose n k else 0

theorem sum_identity
  (p q : ℕ)
  (a b : ℚ) :
  ∑ i in finset.range (p + 1), (binomial_coefficient p i * binomial_coefficient p i * a^(p-i) * b^i) =
  ∑ i in finset.range (p + 1), (binomial_coefficient p i * binomial_coefficient (q + i) i * (a - b)^(p - i) * b^i) :=
by
  sorry

end sum_identity_l633_633414


namespace jackie_apples_l633_633146

variable (A J : ℕ)

-- Condition: Adam has 3 more apples than Jackie.
axiom h1 : A = J + 3

-- Condition: Adam has 9 apples.
axiom h2 : A = 9

-- Question: How many apples does Jackie have?
theorem jackie_apples : J = 6 :=
by
  -- We would normally the proof steps here, but we'll skip to the answer
  sorry

end jackie_apples_l633_633146


namespace verify_polynomial_relationship_l633_633263

theorem verify_polynomial_relationship :
  (∀ x : ℕ, f x = x^3 + 2 * x + 1) ∧
  f 1 = 4 ∧
  f 2 = 15 ∧
  f 3 = 40 ∧
  f 4 = 85 ∧
  f 5 = 156 :=
by
  let f := λ x : ℕ, x^3 + 2 * x + 1
  split; intros;
  { sorry }

end verify_polynomial_relationship_l633_633263


namespace turnips_bag_l633_633048

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l633_633048


namespace min_filtrations_l633_633526

-- Definitions of initial conditions and log values.
def initial_impurity := 0.10
def market_requirement := 0.005
def reduction_factor := 2 / 3
def log2 := 0.3010
def log3 := 0.4771

-- The main theorem stating that at least 8 filtrations are required.
theorem min_filtrations (x : ℕ) :
  (initial_impurity * (reduction_factor ^ x) ≤ market_requirement) -> x ≥ 8 :=
by
  sorry

end min_filtrations_l633_633526


namespace area_of_figure_l633_633692

theorem area_of_figure : 
  let S := { p : ℝ × ℝ | (|p.1| + p.1)^2 + (|p.2| - p.2)^2 ≤ 16 ∧ p.2 - 3 * p.1 ≤ 0 } in
  measure_theory.measure.inter_volume S = (20/3 + real.pi) :=
sorry

end area_of_figure_l633_633692


namespace spends_at_arcade_each_weekend_l633_633894

def vanessa_savings : ℕ := 20
def parents_weekly_allowance : ℕ := 30
def dress_cost : ℕ := 80
def weeks : ℕ := 3

theorem spends_at_arcade_each_weekend (arcade_weekend_expense : ℕ) :
  (vanessa_savings + weeks * parents_weekly_allowance - dress_cost = weeks * parents_weekly_allowance - arcade_weekend_expense * weeks) →
  arcade_weekend_expense = 30 :=
by
  intro h
  sorry

end spends_at_arcade_each_weekend_l633_633894


namespace standard_ellipse_equation_l633_633912

-- Definition of constants based on the conditions of the problem.
def a : ℝ := 6
def c : ℝ := 3 * Real.sqrt 3
def foci : List (ℝ × ℝ) := [(0, -2), (0, 2)]
def point_A : ℝ × ℝ := (3, 2)

-- Main theorem to prove the standard equation of the ellipse based on the given conditions.
theorem standard_ellipse_equation (h1 : a = 6) (h2 : c = 3 * Real.sqrt 3) (h3 : foci = [(0, -2), (0, 2)]) (h4 : point_A = (3, 2)) :
    ∃ x y : ℝ, (y^2 / 16) + (x^2 / 12) = 1 :=
by
    sorry

end standard_ellipse_equation_l633_633912


namespace arrange_chairs_and_stools_l633_633529

-- Definition of the mathematical entities based on the conditions
def num_ways_to_arrange (women men : ℕ) : ℕ :=
  let total := women + men
  (total.factorial) / (women.factorial * men.factorial)

-- Prove that the arrangement yields the correct number of ways
theorem arrange_chairs_and_stools :
  num_ways_to_arrange 7 3 = 120 := by
  -- The specific definitions and steps are not to be included in the Lean statement;
  -- hence, adding a placeholder for the proof.
  sorry

end arrange_chairs_and_stools_l633_633529


namespace shoe_price_friday_l633_633464

theorem shoe_price_friday :
  let price_on_wednesday := 50
  let thursday_increase := 15 / 100
  let friday_discount := 20 / 100
  let price_on_thursday := price_on_wednesday * (1 + thursday_increase)
  let price_on_friday := price_on_thursday * (1 - friday_discount)
  price_on_friday = 46 := by
  let price_on_wednesday := 50
  let thursday_increase := 15 / 100
  let friday_discount := 20 / 100
  let price_on_thursday := price_on_wednesday * (1 + thursday_increase)
  let price_on_friday := price_on_thursday * (1 - friday_discount)
  show price_on_friday = 46 from sorry

end shoe_price_friday_l633_633464


namespace find_m_value_l633_633680

noncomputable def parabola_and_lines (m : ℝ) : Prop :=
  let f := (λ x : ℝ, x^2 + 2 * x - 3)
  let l1 := (λ x : ℝ, -x + m)
  let x := (λ x : ℝ, x * ((1 : ℝ) / 2)) -- axis of symmetry x = -1
  let A := (x₁, f x₁)
  let C := (x₂, f x₂)
  let B := (x₃, f x₃)
  let D := (x₄, f x₄) in
  (x₁ + x₂ = -3) ∧ (x₁ * x₂ = -(3 + m)) ∧ (AC := (x₁ - x₂)^2 = 13) ∧ AC * BD = 26 → m = -2

theorem find_m_value (m : ℝ) : parabola_and_lines m :=
  sorry

end find_m_value_l633_633680


namespace max_min_values_l633_633198

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem max_min_values : 
  (∃ a b, a ∈ set.interval (-1 : ℝ) 1 ∧ b ∈ set.interval (-1 : ℝ) 1 ∧ 
    (∀ x ∈ set.interval (-1 : ℝ) 1, f x ≤ f b) ∧ 
    (∀ x ∈ set.interval (-1 : ℝ) 1, f a ≤ f x) ∧ 
    f a = 1 ∧ f b = 5) :=
sorry

end max_min_values_l633_633198


namespace turnip_weights_are_13_or_16_l633_633100

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l633_633100


namespace range_of_a_l633_633246

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0.5 then -0.5*x + 0.25 else x/(x+2)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  a * (Real.cos (π * x / 2)) + 5 - 2 * a

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∃ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ 1) ∧ (0 ≤ x2 ∧ x2 ≤ 1) ∧ f x1 = g a x2) ↔ 
  (7/3 ≤ a ∧ a ≤ 5) :=
by
  sorry

end range_of_a_l633_633246


namespace probability_two_red_two_blue_l633_633026

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose 20 4
  let choose_2_red := Nat.choose 12 2
  let choose_2_blue := Nat.choose 8 2
  (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690 := by
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose total_marbles 4
  let choose_2_red := Nat.choose total_red 2
  let choose_2_blue := Nat.choose total_blue 2
  show (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690
  sorry

end probability_two_red_two_blue_l633_633026


namespace no_snow_five_days_l633_633833

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633833


namespace not_increasing_in_interval_0_infty_l633_633998

/-- Define the functions we are considering -/
def f1 (x : ℝ) : ℝ := 2 ^ x
def f2 (x : ℝ) : ℝ := Real.log x
def f3 (x : ℝ) : ℝ := x ^ 3
def f4 (x : ℝ) : ℝ := 1 / x

/-- Statement of the problem -/
theorem not_increasing_in_interval_0_infty
  (x : ℝ) (h : 0 < x) :
  ¬ (∀ x > 0, f4 x > f4 (x - 1)) :=
sorry

end not_increasing_in_interval_0_infty_l633_633998


namespace linear_equation_l633_633697

noncomputable def is_linear (k : ℝ) : Prop :=
  2 * (|k|) = 1 ∧ k ≠ 1

theorem linear_equation (k : ℝ) : is_linear k ↔ k = -1 :=
by
  sorry

end linear_equation_l633_633697


namespace fraction_meaningful_l633_633310

theorem fraction_meaningful (x : ℝ) : (x-5) ≠ 0 ↔ (1 / (x - 5)) = (1 / (x - 5)) := 
by 
  sorry

end fraction_meaningful_l633_633310


namespace meryll_written_questions_ratio_l633_633774

theorem meryll_written_questions_ratio
  (total_mcqs : ℕ) (total_psqs : ℕ) 
  (written_psqs_fraction : ℚ) (remaining_questions : ℕ)
  (h1 : total_mcqs = 35)
  (h2 : total_psqs = 15)
  (h3 : written_psqs_fraction = 1 / 3)
  (h4 : remaining_questions = 31) :
  let written_psqs := total_psqs * written_psqs_fraction in
  let written_mcqs := total_mcqs - (remaining_questions - written_psqs) in
  written_mcqs / total_mcqs = 9 / 35 :=
by
  sorry

end meryll_written_questions_ratio_l633_633774


namespace find_angle_between_vectors_l633_633267

open Real EuclideanGeometry

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions
def norm_a : ℝ := Real.sqrt 3
def norm_b : ℝ := 2
def orthogonal : InnerProductSpace ℝ (Fin 2) := {
  dist := sorry
}

-- Problem Statement
theorem find_angle_between_vectors (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : ∥ a ∥ = norm_a)
  (h2 : ∥ b ∥ = norm_b)
  (h3 : inner (a - b) a = 0) :
  angle a b = π / 6 :=
sorry

end find_angle_between_vectors_l633_633267


namespace triangle_area_ratio_l633_633366

theorem triangle_area_ratio :
  let base_jihye := 3
  let height_jihye := 2
  let base_donggeon := 3
  let height_donggeon := 6.02
  let area_jihye := (base_jihye * height_jihye) / 2
  let area_donggeon := (base_donggeon * height_donggeon) / 2
  (area_donggeon / area_jihye) = 3.01 :=
by
  sorry

end triangle_area_ratio_l633_633366


namespace angle_XCY_less_than_60_degrees_l633_633323

open EuclideanGeometry

/-- In a triangle ABC, AB is the shortest side.
Points X and Y are given on the circumcircle of △ABC
such that CX = AX + BX and CY = AY + BY.
Prove that ∠XCY < 60° . -/
theorem angle_XCY_less_than_60_degrees
  {A B C X Y : Point}
  (hABC: Triangle ABC)
  (h_short: dist A B < dist A C ∧ dist A B < dist B C)
  (hX_circum: OnCircumcircle X A B C)
  (hY_circum: OnCircumcircle Y A B C)
  (hCX_eq: dist C X = dist A X + dist B X)
  (hCY_eq: dist C Y = dist A Y + dist B Y) :
  ∠ X C Y < 60 :=
sorry

end angle_XCY_less_than_60_degrees_l633_633323


namespace rectangle_dimensions_l633_633183

theorem rectangle_dimensions (a b : ℝ) 
  (h_area : a * b = 12) 
  (h_perimeter : 2 * (a + b) = 26) : 
  (a = 1 ∧ b = 12) ∨ (a = 12 ∧ b = 1) :=
sorry

end rectangle_dimensions_l633_633183


namespace diophantine_solution_count_l633_633435

theorem diophantine_solution_count :
  {p : ℤ × ℤ | (p.fst^2 + p.snd^2 = p.fst * p.snd + 2 * p.fst + 2 * p.snd)}.to_finset.card = 6 :=
by sorry

end diophantine_solution_count_l633_633435


namespace time_addition_sum_l633_633364

theorem time_addition_sum (A B C : ℕ) (h1 : A = 7) (h2 : B = 59) (h3 : C = 59) : A + B + C = 125 :=
sorry

end time_addition_sum_l633_633364


namespace primes_squared_mod_180_l633_633573

theorem primes_squared_mod_180 (p : ℕ) (hp_prime : Prime p) (hp_gt_5 : p > 5) :
  {r | ∃ k : ℕ, p^2 = 180 * k + r}.card = 2 :=
by
  sorry

end primes_squared_mod_180_l633_633573


namespace not_snow_probability_l633_633866

theorem not_snow_probability :
  let p_snow : ℚ := 2 / 3,
      p_not_snow : ℚ := 1 - p_snow in
  (p_not_snow ^ 5) = 1 / 243 := by
  sorry

end not_snow_probability_l633_633866


namespace original_game_start_player_wins_modified_game_start_player_wins_l633_633891

def divisor_game_condition (num : ℕ) := ∀ d : ℕ, d ∣ num → ∀ x : ℕ, x ∣ d → x = d ∨ x = 1
def modified_divisor_game_condition (num d_prev : ℕ) := ∀ d : ℕ, d ∣ num → d ≠ d_prev → ∃ k l : ℕ, d = k * l ∧ k ≠ 1 ∧ l ≠ 1 ∧ k ≤ l

/-- Prove that if the starting player plays wisely, they will always win the original game. -/
theorem original_game_start_player_wins : ∀ d : ℕ, divisor_game_condition 1000 → d = 100 → (∃ p : ℕ, p != 1000) := 
sorry

/-- What happens if the game is modified such that a divisor cannot be mentioned if it has fewer divisors than any previously mentioned number? -/
theorem modified_game_start_player_wins : ∀ d_prev : ℕ, modified_divisor_game_condition 1000 d_prev → d_prev = 100 → (∃ p : ℕ, p != 1000) := 
sorry

end original_game_start_player_wins_modified_game_start_player_wins_l633_633891


namespace accounting_class_exam_l633_633717

theorem accounting_class_exam
  (students : ℕ)
  (assigned_avg : ℕ)
  (makeup_avg : ℕ)
  (total_avg : ℕ)
  (h_students : students = 100)
  (h_assigned_avg : assigned_avg = 65)
  (h_makeup_avg : makeup_avg = 95)
  (h_total_avg : total_avg = 74)
  : 
  (x : ℝ) 
  (h_eq : 65 * x + 95 * (100 - x) = 74 * 100)
  : x = 70 :=
sorry

end accounting_class_exam_l633_633717


namespace room_length_l633_633453

theorem room_length (w : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (h : w = 4) (h1 : cost_rate = 800) (h2 : total_cost = 17600) : 
  let L := total_cost / (w * cost_rate)
  L = 5.5 :=
by
  sorry

end room_length_l633_633453


namespace combined_total_time_l633_633971

def jerry_time : ℕ := 3
def elaine_time : ℕ := 2 * jerry_time
def george_time : ℕ := elaine_time / 3
def kramer_time : ℕ := 0
def total_time : ℕ := jerry_time + elaine_time + george_time + kramer_time

theorem combined_total_time : total_time = 11 := by
  unfold total_time jerry_time elaine_time george_time kramer_time
  rfl

end combined_total_time_l633_633971


namespace determine_a_l633_633644

noncomputable def normal_distribution_a (ξ : ℝ → Prop) (σ : ℝ) (a : ℝ) : Prop :=
  (ξ ~ N(2, σ^2)) ∧ (P(ξ ≤ 4 - a) = P(ξ ≥ 2 + 3a))

theorem determine_a (ξ : ℝ → Prop) (σ : ℝ) :
  (∃ a, normal_distribution_a ξ σ a) → (a = -1) :=
by
  sorry

end determine_a_l633_633644


namespace arrangement_count_l633_633532

theorem arrangement_count (n m : ℕ) (h : n = 7 ∧ m = 3) :
  (n + m).choose m = 120 :=
by
  rw [h.1, h.2]
  unfold nat.choose
  sorry

end arrangement_count_l633_633532


namespace scientific_notation_of_209_6_billion_l633_633802

theorem scientific_notation_of_209_6_billion (billion : ℝ) (209_6_billion : ℝ) : 
  (billion = 10^9) → (209_6_billion = 209.6 * 10^9) →
  209_6_billion = 2.096 * 10^10 :=
by
  intros hb h2096
  rw [h2096, hb]
  sorry

end scientific_notation_of_209_6_billion_l633_633802


namespace like_apple_orange_mango_l633_633924

theorem like_apple_orange_mango (A B C: ℕ) 
  (h1: A = 40) 
  (h2: B = 7) 
  (h3: C = 10) 
  (total: ℕ) 
  (h_total: total = 47) 
: ∃ x: ℕ, 40 + (10 - x) + x = 47 ∧ x = 3 := 
by 
  sorry

end like_apple_orange_mango_l633_633924


namespace area_of_first_part_is_correct_l633_633967

def width_first_part : ℕ := 30

theorem area_of_first_part_is_correct :
  (width_first_part * ((1000 / (width_first_part + 10)))) = 750 :=
by
  -- The width of the first part is given as 30 meters
  have h1 : width_first_part = 30 := rfl
  -- The length of the parts based on the conditions
  have len_second_part_eq_len_third_part : (1000 / (width_first_part + 10)) = (650 / (width_first_part - 4)) :=
    by sorry
  -- Since the lengths must be equal, we solve for width_first_part
  calc
    width_first_part * (1000 / (width_first_part + 10))
    = 30 * 25 : by { rw [h1], sorry }
    ... = 750 : by rfl

-- Sorry used to indicate parts of the proof that need to be filled.

end area_of_first_part_is_correct_l633_633967


namespace prime_square_mod_180_l633_633569

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : 5 < p) :
  ∃ (s : Finset ℕ), s.card = 2 ∧ ∀ r ∈ s, ∃ n : ℕ, p^2 % 180 = r :=
sorry

end prime_square_mod_180_l633_633569


namespace octahedron_tetrahedron_volume_ratio_l633_633948

theorem octahedron_tetrahedron_volume_ratio (s : ℝ) :
  let V_T := (s^3 * Real.sqrt 2) / 12
  let a := s / 2
  let V_O := (a^3 * Real.sqrt 2) / 3
  V_O / V_T = 1 / 2 :=
by
  sorry

end octahedron_tetrahedron_volume_ratio_l633_633948


namespace charity_ticket_revenue_l633_633523

noncomputable def full_price_ticket_revenue
  (f h p : ℕ) -- number of full-price tickets, number of half-price tickets, price of a full-price ticket
  (tickets_sold : f + h = 140)
  (total_revenue : f * p + h * (p / 2) = 2001) : ℕ :=
  f * p

theorem charity_ticket_revenue :
  ∃ (f h p : ℕ), f + h = 140 ∧ f * p + h * (p / 2) = 2001 ∧ f * p = 782 :=
begin
  sorry
end

end charity_ticket_revenue_l633_633523


namespace central_angle_unit_circle_l633_633324

theorem central_angle_unit_circle :
  ∀ (θ : ℝ), (∃ (A : ℝ), A = 1 ∧ (A = 1 / 2 * θ)) → θ = 2 :=
by
  intro θ
  rintro ⟨A, hA1, hA2⟩
  sorry

end central_angle_unit_circle_l633_633324


namespace closest_term_l633_633770

variables {a_n b_n c_n : ℕ → ℝ} 
variables (b_sum : ∀ n, (finset.range n).sum (λ i, a_n i) = b_n n)
variables (c_prod : ∀ n, (finset.range n).prod (λ i, b_n i) = c_n n)
variables (b_c_rel : ∀ n, b_n n + c_n n = 1)

noncomputable def find_closest_term : ℕ := 
if h : ∃ n, abs (n * (n + 1) - 2002) = min (abs ((nat.floor (real.sqrt 8009 / 2)).succ * ((nat.floor (real.sqrt 8009 / 2)).succ + 1) - 2002))
                                                   (abs ((nat.floor (real.sqrt 8009 / 2)) * ((nat.floor (real.sqrt 8009 / 2)) + 1) - 2002))
then classical.some h else 0

theorem closest_term :
  ∃ n, find_closest_term = 1980 :=
begin
  sorry,
end

end closest_term_l633_633770


namespace prime_square_mod_180_l633_633566

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : 5 < p) :
  ∃ (s : Finset ℕ), s.card = 2 ∧ ∀ r ∈ s, ∃ n : ℕ, p^2 % 180 = r :=
sorry

end prime_square_mod_180_l633_633566


namespace no_integer_root_l633_633418

theorem no_integer_root (q : ℤ) : ¬ ∃ x : ℤ, x^2 + 7 * x - 14 * (q^2 + 1) = 0 := sorry

end no_integer_root_l633_633418


namespace probability_denis_oleg_play_l633_633136

theorem probability_denis_oleg_play (n : ℕ) (h_n : n = 26) :
  (1 : ℚ) / 13 = 
  let total_matches : ℕ := n - 1 in
  let num_pairs := n * (n - 1) / 2 in
  (total_matches : ℚ) / num_pairs :=
by
  -- You can provide a proof here if necessary
  sorry

end probability_denis_oleg_play_l633_633136


namespace product_of_integers_l633_633472

theorem product_of_integers (x y : ℕ) (h1 : x + y = 72) (h2 : x - y = 18) : x * y = 1215 := 
sorry

end product_of_integers_l633_633472


namespace proof_num_solutions_eq_l633_633626

noncomputable 
def num_solutions_eq : ℕ := number_of_solutions (λ x : ℝ, sqrt (8 - x) = x * sqrt (8 - x))

theorem proof_num_solutions_eq : num_solutions_eq = 2 :=
by
  sorry

end proof_num_solutions_eq_l633_633626


namespace find_acute_angle_of_rhombus_l633_633884

variables {α : ℝ}

theorem find_acute_angle_of_rhombus (α : ℝ) :
  ∃ (A : ℝ), A = 2 * Real.arccot (2 * Real.cos α) :=
begin
  use 2 * Real.arccot (2 * Real.cos α),
  sorry
end

end find_acute_angle_of_rhombus_l633_633884


namespace largest_base_b_digits_not_18_l633_633495

-- Definition of the problem:
-- Let n = 12^3 in base 10
def n : ℕ := 12 ^ 3

-- Definition of the conditions:
-- In base 8, 1728 (12^3 in base 10) has its digits sum to 17
def sum_of_digits_base_8 (x : ℕ) : ℕ :=
  let digits := x.digits (8)
  digits.sum

-- Proof statement
theorem largest_base_b_digits_not_18 : ∃ b : ℕ, (max b) = 8 ∧ sum_of_digits_base_8 n ≠ 18 := by
  sorry

end largest_base_b_digits_not_18_l633_633495


namespace top_card_is_club_probability_l633_633121

-- Definitions based on the conditions
def deck_size := 52
def suit_count := 4
def cards_per_suit := deck_size / suit_count

-- The question we want to prove
theorem top_card_is_club_probability :
  (13 : ℝ) / (52 : ℝ) = 1 / 4 :=
by 
  sorry

end top_card_is_club_probability_l633_633121


namespace numOf14DigitNumbersDivisibleBy792_l633_633301

-- Define a function to check if a number formed by the digits is divisible by 792
def isDivisibleBy792 (p q r s : ℕ) : Prop :=
  let n := 88663311000000 + p * 1000000 + q * 100000 + r * 10000 + s * 1000 + 48
  (n % 792 = 0)

-- Define the main theorem statement that requires proving
theorem numOf14DigitNumbersDivisibleBy792 : ∃ n : ℕ, n = 50 ∧
  n = (Finset.univ.filter (λ ⟨p, q, r, s⟩ : Finset (ℕ × ℕ × ℕ × ℕ), 
         isDivisibleBy792 p q r s)).card := 
sorry

end numOf14DigitNumbersDivisibleBy792_l633_633301


namespace sum_of_squares_of_roots_l633_633300

theorem sum_of_squares_of_roots :
  let f := Polynomial.C 2 + Polynomial.C 9 * Polynomial.X + Polynomial.C 12 * (Polynomial.X ^ 2) + Polynomial.C 6 * (Polynomial.X ^ 3) + Polynomial.X ^ 4
  (A : ℝ) := 1 + 4 + 7 → A = 12 :=
sorry

end sum_of_squares_of_roots_l633_633300


namespace ellipse_hyperbola_tangent_l633_633445

theorem ellipse_hyperbola_tangent : ∀ m : ℝ, 
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y + 3)^2 = 1 → false) → 
  m = 8 / 9 :=
begin
  sorry,
end

end ellipse_hyperbola_tangent_l633_633445


namespace solution_set_inequality_l633_633872

theorem solution_set_inequality
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → ax^2 + bx + c > 0) :
  ∃ s : Set ℝ, s = {x | (1/2) < x ∧ x < 1} ∧ ∀ x : ℝ, x ∈ s → cx^2 + bx + a > 0 := by
sorry

end solution_set_inequality_l633_633872


namespace percentage_of_consumption_l633_633116

-- Define variables and constants based on given conditions
def x : ℝ := 10 -- average wage per capita in thousand yuan
def y : ℝ := 7.5 -- average consumption per capita in thousand yuan

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 0.6 * x + 1.5

-- Define the percentage calculation
def percentage (y x : ℝ) : ℝ := (y / x) * 100

-- State the theorem
theorem percentage_of_consumption : regression_equation x = y → percentage y x = 75 :=
by
  sorry

end percentage_of_consumption_l633_633116


namespace fatima_cuts_count_l633_633190

-- Condition Definitions
def initial_size : ℕ := 100
def donated_total : ℕ := 75
def donate_each_cut (size: ℕ) (n: ℕ) : ℕ :=
  let cut_sizes := (List.iterate (λ s, s / 2) n size).drop 1
  List.sum (List.take n cut_sizes)

-- Problem Statement
theorem fatima_cuts_count : ∃ n : ℕ, donate_each_cut initial_size n = donated_total := by
  -- Skip the actual proof
  sorry

end fatima_cuts_count_l633_633190


namespace right_triangle_of_sine_cosine_identity_l633_633738

theorem right_triangle_of_sine_cosine_identity (A B C : ℝ) (h1 : sin A * cos B = 1 - cos A * sin B) (h2 : A + B + C = Real.pi) : C = Real.pi / 2 :=
by
  sorry

end right_triangle_of_sine_cosine_identity_l633_633738


namespace expected_urns_with_one_marble_is_correct_l633_633633

noncomputable def number_of_marbles := 5
noncomputable def number_of_urns := 7

def probability_of_exactly_one_marble_in_urn (m u : ℕ) : ℚ :=
  ((nat.choose m 1) * (u - 1)^(m - 1) : ℚ) / (u ^ m)

def expected_urns_with_exactly_one_marble (m u : ℕ) : ℚ :=
  u * probability_of_exactly_one_marble_in_urn m u

theorem expected_urns_with_one_marble_is_correct :
  expected_urns_with_exactly_one_marble number_of_marbles number_of_urns = 6480 / 2401 :=
by
  sorry

end expected_urns_with_one_marble_is_correct_l633_633633


namespace find_c_in_triangle_l633_633316

theorem find_c_in_triangle
  (angle_B : ℝ)
  (a : ℝ)
  (S : ℝ)
  (h1 : angle_B = 45)
  (h2 : a = 4)
  (h3 : S = 16 * Real.sqrt 2) :
  ∃ c : ℝ, c = 16 :=
by
  sorry

end find_c_in_triangle_l633_633316


namespace roller_coaster_mean_median_diff_zero_l633_633474

theorem roller_coaster_mean_median_diff_zero : 
  let heights := [180, 150, 210, 195, 170, 220] in
  let mean := (heights.sum : ℝ) / heights.length in
  let sorted_heights := heights.qsort (≤) in
  let median := ((sorted_heights.nthLe (sorted_heights.length / 2 - 1) sorry) +
                 (sorted_heights.nthLe (sorted_heights.length / 2) sorry)) / 2 in
  |mean - median| = 0 :=
by
  let heights := [180, 150, 210, 195, 170, 220]
  let mean := (heights.sum : ℝ) / heights.length
  let sorted_heights := heights.qsort (≤)
  let median := ((sorted_heights.nthLe (sorted_heights.length / 2 - 1) sorry) +
                 (sorted_heights.nthLe (sorted_heights.length / 2) sorry)) / 2
  have h : |mean - median| = 0
  -- Details of proof steps are omitted here
  sorry

end roller_coaster_mean_median_diff_zero_l633_633474


namespace right_triangle_of_sin_cos_eq_l633_633740

variable {A B C : ℝ}  
variable {α β γ : Angle}

axiom triangle_angle_sum (A B C : ℝ) : A + B + C = π
axiom sin_eq_1 (α : Angle) : sin α = 1 → α = (π/2 : ℝ)

/-- If sin A cos B = 1 - cos A sin B in triangle ABC, then it is a right triangle. -/
theorem right_triangle_of_sin_cos_eq (h : sin A * cos B = 1 - cos A * sin B) (hsum : A + B + C = π) : C = π / 2 := 
by
  have h1 : sin (A + B) = 1 := sorry -- sin(A + B) = 1
  exact sin_eq_1 (Angle.ofReal (A + B))
  have h2 : A + B = π / 2 := sorry -- angle A + B should be π / 2
  have h3 : A + B + C = π := hsum
  exact sub_eq_of_eq_add' h3
  exact add_sub_cancel'_right h3


end right_triangle_of_sin_cos_eq_l633_633740


namespace charity_ticket_revenue_l633_633522

noncomputable def full_price_ticket_revenue
  (f h p : ℕ) -- number of full-price tickets, number of half-price tickets, price of a full-price ticket
  (tickets_sold : f + h = 140)
  (total_revenue : f * p + h * (p / 2) = 2001) : ℕ :=
  f * p

theorem charity_ticket_revenue :
  ∃ (f h p : ℕ), f + h = 140 ∧ f * p + h * (p / 2) = 2001 ∧ f * p = 782 :=
begin
  sorry
end

end charity_ticket_revenue_l633_633522


namespace hyperbola_asymptotes_l633_633252

theorem hyperbola_asymptotes 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (sqrt 5) / 2 = (real.sqrt (a^2 + b^2)) / a) :
  ∀ x : ℝ, y = b / a * x ↔ y = ± (1/2) * x :=
sorry

end hyperbola_asymptotes_l633_633252


namespace train_speed_in_kmph_l633_633555

theorem train_speed_in_kmph
  (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ)
  (H1: train_length = 200) (H2: bridge_length = 150) (H3: time_seconds = 34.997200223982084) :
  train_length + bridge_length = 200 + 150 →
  (train_length + bridge_length) / time_seconds * 3.6 = 36 :=
sorry

end train_speed_in_kmph_l633_633555


namespace probability_of_point_in_smaller_sphere_l633_633545

theorem probability_of_point_in_smaller_sphere 
  (R r : ℝ) (P : EuclideanSpace ℝ (Fin 3)) (hcirc : Sphere (EuclideanSpace ℝ (Fin 3)) R) 
  (hin : Sphere (EuclideanSpace ℝ (Fin 3)) r)
  (hrel : R = 2 * r) 
  (h_spheres : ∀ i : Fin 5, Sphere (EuclideanSpace ℝ (Fin 3)) r) :
  5 * (\(Sphere.vol (Sphere (EuclideanSpace ℝ (Fin 3)) r)) / \(Sphere.vol (Sphere (EuclideanSpace ℝ (Fin 3)) R)) = 0.625 :=
by
  sorry

end probability_of_point_in_smaller_sphere_l633_633545


namespace evaluate_expression_l633_633613

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 4) : (a^b)^a - (b^a)^b = -16245775 := by
  rw [ha, hb]
  -- we start with the given equations and rewrite
  have h1 : (3^4)^3 = 81^3 := rfl
  have h2 : (4^3)^4 = 64^4 := rfl
  rw [h1, h2]
  -- final equality
  have h3 : 81^3 = 531441 := rfl
  have h4 : 64^4 = 16777216 := rfl
  rw [h3, h4]
  exact rfl

end evaluate_expression_l633_633613


namespace school_population_total_l633_633120

variables (B : ℕ) (deaf_students_initial blind_students_initial other_disabilities_initial : ℕ)

noncomputable def deaf_students_end := deaf_students_initial + (10 * deaf_students_initial / 100)
noncomputable def blind_students_end := blind_students_initial + (15 * blind_students_initial / 100)
noncomputable def other_disabilities_end := other_disabilities_initial + (12 * other_disabilities_initial / 100)

noncomputable def total_students_end := deaf_students_end + blind_students_end + other_disabilities_end

theorem school_population_total
  (h1 : deaf_students_initial = 3 * B)
  (h2 : other_disabilities_initial = 2 * B)
  (h3 : deaf_students_initial = 180) :
  total_students_end B deaf_students_initial blind_students_initial other_disabilities_initial = 401 :=
sorry

end school_population_total_l633_633120


namespace count_numbers_with_last_two_digits_96_l633_633963

def has_last_two_digits_96 (n : ℕ) : Prop :=
  n^2 % 100 = 96

theorem count_numbers_with_last_two_digits_96 :
  {k : ℕ | has_last_two_digits_96 k ∧ 1 ≤ k ∧ k ≤ 1996}.to_finset.card = 80 := 
sorry

end count_numbers_with_last_two_digits_96_l633_633963


namespace range_of_x_l633_633381

noncomputable def f : ℝ → ℝ := sorry
variable (x : ℝ)

-- f is an even function
axiom even_f (x : ℝ) : f(x) = f(-x)

-- f is monotonically increasing on [0, +∞)
axiom monotone_f : ∀ ⦃a b : ℝ⦄, 0 ≤ a → 0 ≤ b → a ≤ b → f(a) ≤ f(b)

-- Determine the range of x satisfying f(2x - 1) ≤ f(1)
theorem range_of_x {x : ℝ} (h : f(2x - 1) ≤ f(1)) : 0 ≤ x ∧ x ≤ 1 :=
sorry

end range_of_x_l633_633381


namespace number_of_valid_x_for_isosceles_triangle_area_l633_633206

theorem number_of_valid_x_for_isosceles_triangle_area :
  ∃ x : ℕ, ∃ k : ℕ, (k ≥ 1 ∧ k ≤ 2112) ∧
  let θ := k * arcsin(2 * k / 4225) in
  let A := (4225 / 2) * sin θ in
  A ∈ ℕ ∧ (2 * 2112 = 4224) :=
sorry

end number_of_valid_x_for_isosceles_triangle_area_l633_633206


namespace determine_natural_numbers_l633_633919

-- Definitions of s and d
def d(n : ℕ) : ℕ := (finset.range (n + 1)).filter (λ x, x > 0 ∧ n % x = 0).card
def s(n : ℕ) : ℕ := (finset.range (n + 1)).filter (λ x, x > 0 ∧ n % x = 0).sum id

theorem determine_natural_numbers (x : ℕ) : 
  s(x) * d(x) = 96 ↔ x = 47 ∨ x = 14 ∨ x = 15 :=
by
  sorry

end determine_natural_numbers_l633_633919


namespace slices_with_both_toppings_l633_633926

theorem slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices : ℕ)
    (all_have_topping : total_slices = 24)
    (pepperoni_cond: pepperoni_slices = 14)
    (mushroom_cond: mushroom_slices = 16)
    (at_least_one_topping : total_slices = pepperoni_slices + mushroom_slices - slices_with_both):
    slices_with_both = 6 := by
  sorry

end slices_with_both_toppings_l633_633926


namespace sum_of_sequence_l633_633118

def sequence_t (n : ℕ) : ℚ :=
  if n % 2 = 1 then 1 / 7^n else 2 / 7^n

theorem sum_of_sequence :
  (∑' n:ℕ, sequence_t (n + 1)) = 3 / 16 :=
by
  sorry

end sum_of_sequence_l633_633118


namespace a_2016_eq_neg1_l633_633734

noncomputable def a : ℕ → ℝ
| 0       := -1
| (n + 1) := (1 + a n) / (1 - a n)

theorem a_2016_eq_neg1 : a 2015 = -1 :=
by sorry

end a_2016_eq_neg1_l633_633734


namespace probability_of_two_red_two_blue_l633_633022

-- Definitions for the conditions
def total_red_marbles : ℕ := 12
def total_blue_marbles : ℕ := 8
def total_marbles : ℕ := total_red_marbles + total_blue_marbles
def num_selected_marbles : ℕ := 4
def num_red_selected : ℕ := 2
def num_blue_selected : ℕ := 2

-- Definition for binomial coefficient (combinations)
def C (n k : ℕ) : ℕ := n.choose k

-- Probability calculation
def probability_two_red_two_blue :=
  (C total_red_marbles num_red_selected * C total_blue_marbles num_blue_selected : ℚ) / C total_marbles num_selected_marbles

-- The theorem statement
theorem probability_of_two_red_two_blue :
  probability_two_red_two_blue = 1848 / 4845 :=
by
  sorry

end probability_of_two_red_two_blue_l633_633022


namespace least_positive_value_of_cubic_eq_l633_633769

theorem least_positive_value_of_cubic_eq (x y z w : ℕ) 
  (hx : Nat.Prime x) (hy : Nat.Prime y) 
  (hz : Nat.Prime z) (hw : Nat.Prime w) 
  (sum_lt_50 : x + y + z + w < 50) : 
  24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3 = 1464 :=
by
  sorry

end least_positive_value_of_cubic_eq_l633_633769


namespace turnip_weight_possible_l633_633085

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l633_633085


namespace probability_denis_oleg_play_l633_633137

theorem probability_denis_oleg_play (n : ℕ) (h_n : n = 26) :
  (1 : ℚ) / 13 = 
  let total_matches : ℕ := n - 1 in
  let num_pairs := n * (n - 1) / 2 in
  (total_matches : ℚ) / num_pairs :=
by
  -- You can provide a proof here if necessary
  sorry

end probability_denis_oleg_play_l633_633137


namespace perfect_number_divisible_by_nine_l633_633109

-- Define the sum of divisors function sigma(n)
def sigma (n : Nat) : Nat :=
  Nat.sumDivisors n

-- Define what it means to be a perfect number
def isPerfect (n : Nat) : Prop :=
  sigma n = 2 * n

-- Define the main statement to prove
theorem perfect_number_divisible_by_nine (n : Nat) (h1 : isPerfect n) (h2 : n > 6) (h3 : ∃ k, n = 3 * k) : ∃ m, n = 9 * m :=
by
  sorry

end perfect_number_divisible_by_nine_l633_633109


namespace parallel_condition_iff_l633_633830

noncomputable def condition_for_parallel (a : ℝ) : Prop :=
  let l1 := λ x y : ℝ, ax + y - a + 1 = 0
  let l2 := λ x y : ℝ, 4x + ay - 2 = 0
  a ≠ 0 → (∀ x y : ℝ, l1 x y = 0 → l2 x y ≠ 0) → (a = 2 ∨ a = -2)

theorem parallel_condition_iff (a : ℝ) : 
  (condition_for_parallel a) ↔ (a = 2 ∨ a = -2 ∧ (¬∀ x y : ℝ, (condition_for_parallel a))) := sorry

end parallel_condition_iff_l633_633830


namespace time_to_fill_tank_l633_633410

-- Conditions as definitions
def pipeA_fill_time : ℝ := 12
def pipeB_fill_time (A_fill_time : ℝ) : ℝ := A_fill_time / 3

-- Target proof statement
theorem time_to_fill_tank (A_fill_time : ℝ) (B_fill_time : ℝ) (h1 : A_fill_time = 12)
    (h2 : B_fill_time = A_fill_time / 3) : 
    1 / (1 / A_fill_time + 1 / B_fill_time) = 3 :=
by
  sorry

end time_to_fill_tank_l633_633410


namespace parabola_line_range_of_m_l633_633239

theorem parabola_line_range_of_m (k m : ℝ) (h₁ : k > 2) (h₂ : m > -3) :
  let M : ℝ × ℝ := ((k^2 + 2) / k^2, 2 / k) in
  (| 3 * ((k^2 + 2) / k^2) + 4 * (2 / k) + m | = 1) →
  (-3 < m ∧ m < -2) :=
by
  intro M
  sorry

end parabola_line_range_of_m_l633_633239


namespace find_M_l633_633482

theorem find_M (x y z M : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x - 10 = M) 
  (h3 : y + 10 = M) 
  (h4 : z / 10 = M) : 
  M = 10 := 
by
  sorry

end find_M_l633_633482


namespace num_positive_divisors_not_divisible_by_3_l633_633296

theorem num_positive_divisors_not_divisible_by_3 :
  let n := 252 in
  let divisors := {d : ℕ | d ∣ n} in
  let prime_factors := {2, 3, 7} in
  let valid_divisors := {d : ℕ | d ∈ divisors ∧ ¬ (3 ∣ d)} in
  valid_divisors.card = 6 :=
by
  sorry

end num_positive_divisors_not_divisible_by_3_l633_633296


namespace verify_fill_blanks_verify_estimate_teachers_at_B_or_above_verify_estimate_articles_read_per_year_l633_633886

noncomputable def teachers_learning_times : List ℕ := 
  [79, 85, 73, 80, 75, 76, 87, 70, 75, 94, 75, 79, 81, 71, 75, 80, 86, 69, 83, 77]

def number_of_teachers : ℕ := 300

def sample_size : ℕ := 20

def grade_distribution (xs : List ℕ) : (ℕ × ℕ × ℕ × ℕ) :=
let grade_A := xs.countp (λ x => 90 ≤ x ∧ x ≤ 100)
let grade_B := xs.countp (λ x => 80 ≤ x ∧ x ≤ 89)
let grade_C := xs.countp (λ x => 70 ≤ x ∧ x ≤ 79)
let grade_D := xs.countp (λ x => 60 ≤ x ∧ x ≤ 69)
in (grade_D, grade_C, grade_B, grade_A)

def statistic_data (xs : List ℕ) : (Float × ℕ × Float) :=
let total := xs.length
let mean = (xs.sum.toFloat) / (total.toFloat)
let sorted_xs := xs.qsort (≤)
let median := if total % 2 = 0 then
  ((sorted_xs.get! (total / 2 - 1)) + (sorted_xs.get! (total / 2))).toFloat / 2
  else sorted_xs.get! (total / 2).toFloat
let mode := xs.foldl (λ acc x => if (xs.count x) > (xs.count acc) then x else acc) (xs.head!)
in (mean, mode, median)

def fill_blanks : (ℕ × Float × Float) :=
let (grade_D, grade_C, grade_B, grade_A) := grade_distribution teachers_learning_times
let grade_C_count := grade_C
let (mean, mode, median) := statistic_data teachers_learning_times
(grade_C_count, mean, median)

def estimate_teachers_at_B_or_above : ℕ :=
let (grade_D, grade_C, grade_B, grade_A) := grade_distribution teachers_learning_times
(number_of_teachers * (grade_B + grade_A)) / sample_size

def estimate_articles_read_per_year : ℕ :=
let (mean, mode, median) := statistic_data teachers_learning_times
((((mean / 3) / 5) * 365).round).toNat

theorem verify_fill_blanks :
  fill_blanks = (11, 78.5, 78) :=
by
  unfold fill_blanks grade_distribution statistic_data
  sorry

theorem verify_estimate_teachers_at_B_or_above :
  estimate_teachers_at_B_or_above = 120 :=
by
  unfold estimate_teachers_at_B_or_above grade_distribution
  sorry

theorem verify_estimate_articles_read_per_year :
  estimate_articles_read_per_year = 1910 :=
by
  unfold estimate_articles_read_per_year statistic_data
  sorry

end verify_fill_blanks_verify_estimate_teachers_at_B_or_above_verify_estimate_articles_read_per_year_l633_633886


namespace area_of_triangle_ABC_l633_633992

theorem area_of_triangle_ABC (A B C : Point) 
  (h_rt : ∠B = 90) 
  (h_Aangle : ∠A = 45) 
  (h_Cangle : ∠C = 45) 
  (h_hypotenuse : dist A C = 20) 
: triangle_area A B C = 100 :=
sorry

end area_of_triangle_ABC_l633_633992


namespace find_f_neg_sqrt3_l633_633249

def f (a b : ℝ) (x : ℝ) : ℝ := a * (x^3) + b * (Real.sin x) + 1

theorem find_f_neg_sqrt3 (a b : ℝ) (h : f a b (Real.sqrt 3) = 2) : f a b (-Real.sqrt 3) = 0 :=
by
  sorry

end find_f_neg_sqrt3_l633_633249


namespace color_of_last_bead_l633_633518

-- Define the sequence and length of repeated pattern
def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "green", "blue"]
def pattern_length : Nat := bead_pattern.length

-- Define the total number of beads in the bracelet
def total_beads : Nat := 85

-- State the theorem to prove the color of the last bead
theorem color_of_last_bead : bead_pattern.get? ((total_beads - 1) % pattern_length) = some "yellow" :=
by
  sorry

end color_of_last_bead_l633_633518


namespace probability_denis_oleg_play_l633_633135

theorem probability_denis_oleg_play (n : ℕ) (h_n : n = 26) :
  (1 : ℚ) / 13 = 
  let total_matches : ℕ := n - 1 in
  let num_pairs := n * (n - 1) / 2 in
  (total_matches : ℚ) / num_pairs :=
by
  -- You can provide a proof here if necessary
  sorry

end probability_denis_oleg_play_l633_633135


namespace max_value_sin_function_l633_633456

theorem max_value_sin_function : 
  ∀ x, (-(π)/2 ≤ x ∧ x ≤ 0) → (3 * sin x + 2 ≤ 2) :=
by
  assume x h,
  sorry

end max_value_sin_function_l633_633456


namespace numbers_with_digit_5_count_numbers_with_digit_5_l633_633275

theorem numbers_with_digit_5 (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 700) : 
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ d = 5) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m : ℕ, (n = m * 100 + d ∧ d = 5)) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m2 m1 : ℕ, (n = m2 * 1000 + m1 * 100 + d ∧ d = 5)) :=
sorry

theorem count_numbers_with_digit_5 : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 700 ∧ (numbers_with_digit_5 n sorry) }.to_finset.card = 214 := 
sorry

end numbers_with_digit_5_count_numbers_with_digit_5_l633_633275


namespace abs_h_eq_1_div_2_l633_633470

theorem abs_h_eq_1_div_2 {h : ℝ} 
  (h_sum_sq_roots : ∀ (r s : ℝ), (r + s) = 4 * h ∧ (r * s) = -8 → (r ^ 2 + s ^ 2) = 20) : 
  |h| = 1 / 2 :=
sorry

end abs_h_eq_1_div_2_l633_633470


namespace edge_sum_impossible_l633_633548

theorem edge_sum_impossible (k : ℕ) : 
  let sum_per_triangle := 1 + 2 + 3 in
  let total_triangle_sum := k * sum_per_triangle in
  let proposed_edge_sum := 55 in
  let total_proposed_sum := 3 * proposed_edge_sum in
  total_proposed_sum ≠ total_triangle_sum :=
by
  let sum_per_triangle := 6
  let total_triangle_sum := k * sum_per_triangle
  let proposed_edge_sum := 55
  let total_proposed_sum := 3 * proposed_edge_sum
  sorry

end edge_sum_impossible_l633_633548


namespace new_oranges_added_l633_633550

-- Defining the initial conditions
def initial_oranges : Nat := 40
def thrown_away_oranges : Nat := 37
def total_oranges_now : Nat := 10
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges := total_oranges_now - remaining_oranges

-- The theorem we want to prove
theorem new_oranges_added : new_oranges = 7 := by
  sorry

end new_oranges_added_l633_633550


namespace solve_for_largest_x_l633_633426

noncomputable def largest_solution (a b c d e : ℝ) : ℝ :=
  (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)

theorem solve_for_largest_x (x : ℝ) (h : 5 * (9 * x^2 + 9 * x + 11) = x * (10 * x - 50)) :
  x = largest_solution 7 19 11 :=
sorry

end solve_for_largest_x_l633_633426


namespace length_of_OP_is_sqrt_200_div_3_l633_633325

open Real

def square (a : ℝ) := a * a

theorem length_of_OP_is_sqrt_200_div_3 (KL MO MP OP : ℝ) (h₁ : KL = 10)
  (h₂: MO = MP) (h₃: square (10) = 100)
  (h₄ : 1 / 6 * 100 = 1 / 2 * (MO * MP)) : OP = sqrt (200/3) :=
by
  sorry

end length_of_OP_is_sqrt_200_div_3_l633_633325


namespace turnip_weight_possible_l633_633079

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l633_633079


namespace p_is_necessary_but_not_sufficient_for_q_l633_633651

variable (x : ℝ)

def p := x > 4
def q := 4 < x ∧ x < 10

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end p_is_necessary_but_not_sufficient_for_q_l633_633651


namespace rectangle_width_decrease_percent_l633_633452

theorem rectangle_width_decrease_percent (L W : ℝ) (h : L * W = L * W) :
  let L_new := 1.3 * L
  let W_new := W / 1.3 
  let percent_decrease := (1 - (W_new / W)) * 100
  percent_decrease = 23.08 :=
sorry

end rectangle_width_decrease_percent_l633_633452


namespace correct_statements_l633_633596

theorem correct_statements :
  let quadrilateral : Type := sorry,
      parallel_sides_and_equal_angles_is_parallelogram : quadrilateral → Prop := sorry,
      equal_diagonals_is_rectangle : quadrilateral → Prop := sorry,
      midpoints_form_rhombus_implies_equal_diagonals : quadrilateral → Prop := sorry,
      all_sides_equal_is_square : quadrilateral → Prop := sorry in
  ((parallel_sides_and_equal_angles_is_parallelogram quadrilateral) ∧ 
  (midpoints_form_rhombus_implies_equal_diagonals quadrilateral)) ∧ 
  ¬((equal_diagonals_is_rectangle quadrilateral) ∨ 
  (all_sides_equal_is_square quadrilateral)) := sorry

end correct_statements_l633_633596


namespace no_snow_five_days_l633_633852

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633852


namespace value_of_f_log2_3_l633_633673

noncomputable def f : ℝ → ℝ 
| x := if x ≥ 4 then 2^(-x) else f (x + 1)

theorem value_of_f_log2_3 :
  f (Real.log 3 / Real.log 2) = 1 / 24 :=
by
  sorry

end value_of_f_log2_3_l633_633673


namespace count_numbers_with_digit_five_l633_633280

theorem count_numbers_with_digit_five : 
  (finset.filter (λ n : ℕ, ∃ d : ℕ, d ∈ digits 10 n ∧ d = 5) (finset.range 701)).card = 133 := 
by 
  sorry

end count_numbers_with_digit_five_l633_633280


namespace find_remainder_l633_633756

-- Define the conditions
def valid_digits (x : Int) := x ≥ 0 ∧ x ≤ 9999

def dist_digits (d : Nat) := 
  let digits : List Nat := [d / 1000 % 10, d / 100 % 10, d / 10 % 10, d % 10]
  digits.Nodup

def M_condition (M : Int) := 
  M % 8 = 0 ∧ M % 16 ≠ 0 ∧ valid_digits (M % 10000)

-- Define the operation result for the given digits
def operation_result (a b : Int) := 4 * a - 3 * b

-- Define the greatest integer multiple of 8 based on conditions given
def greatest_M : Int :=
  max (Set.toFinset { x | ∃ a b : Int, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ x = operation_result a b ∧ dist_digits x})
    (Set.toFinset {x | M_condition x})

-- Define the final formalized theorem statement
theorem find_remainder : 
  ∃ M : Int, M_condition M → (greatest_M % 1000 = 624) :=
by
  sorry

end find_remainder_l633_633756


namespace total_gift_money_l633_633590

-- Definitions based on the conditions given in the problem
def initialAmount : ℕ := 159
def giftFromGrandmother : ℕ := 25
def giftFromAuntAndUncle : ℕ := 20
def giftFromParents : ℕ := 75

-- Lean statement to prove the total amount of money Chris has after receiving his birthday gifts
theorem total_gift_money : 
    initialAmount + giftFromGrandmother + giftFromAuntAndUncle + giftFromParents = 279 := by
sorry

end total_gift_money_l633_633590


namespace sufficiency_not_necessity_condition_l633_633007

theorem sufficiency_not_necessity_condition (a : ℝ) (h : a > 1) : (a^2 > 1) ∧ ¬(∀ x : ℝ, x^2 > 1 → x > 1) :=
by
  sorry

end sufficiency_not_necessity_condition_l633_633007


namespace haar_orthogonal_haar_norm_integral_x_squared_finite_decompose_x_haar_l633_633600

-- Define Haar functions
def h (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then 1
  else if n = 1 then if 0 < x ∧ x < 1 / 2 then 1 else if 1 / 2 < x ∧ x < 1 then -1 else 0
  else let k := (log n / log 2).toInt in
       let u_n := 2 ^ (k / 2 : ℝ) in
       let α_n := (n - 1) / 2 ^ k : ℝ in
       let β_n := α_n + 1 / 2 ^ (k + 1 : ℝ) in
       let γ_n := α_n + 1 / 2 ^ k in
       if α_n < x ∧ x < β_n then u_n
       else if β_n < x ∧ x < γ_n then -u_n
       else 0

-- Orthogonality condition of Haar functions
theorem haar_orthogonal (m n : ℕ) (hne : m ≠ n) :
  ∫ x in 0..1, (h m x * h n x) = 0 := sorry

-- Integral of squared Haar function equals 1
theorem haar_norm (n : ℕ) : ∫ x in 0..1, (h n x) ^ 2 = 1 := sorry

-- Coefficient calculation using Euler-Fourier formula
def a (n : ℕ) : ℝ :=
  ∫ x in 0..1, (x * h n x) / ∫ x in 0..1, (h n x * h n x)

-- Finiteness of integral condition
theorem integral_x_squared_finite : ∫ x in 0..1, x ^ 2 < ∞ := sorry

theorem decompose_x_haar :
  ∀ x, 0 < x ∧ x < 1 → x =
    (1 / 2) - (1 / 4) * ∑' n, a n * h n x := sorry

end haar_orthogonal_haar_norm_integral_x_squared_finite_decompose_x_haar_l633_633600


namespace arithmetic_geometric_sequence_l633_633225

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_common_diff : d = 2) (h_geom : a 2 ^ 2 = a 1 * a 5) : 
  a 2 = 3 :=
by
  sorry

end arithmetic_geometric_sequence_l633_633225


namespace three_correct_operations_l633_633818

-- Define the four differentiation operations as hypotheses
def operation1 : Prop := (fun x : ℝ => x - 1 / x)' = (fun x : ℝ => 1 + x^2 / x^2)
def operation2 : Prop := (fun x : ℝ => log (2 * x - 1))' = (fun x : ℝ => 2 / (2 * x - 1))
def operation3 : Prop := (fun x : ℝ => x^2 * exp x)' = (fun x : ℝ => 2 * x * exp x)
def operation4 : Prop := (fun x : ℝ => logBase 2 x)' = (fun x : ℝ => 1 / (x * log 2))

theorem three_correct_operations : (operation1 ∧ operation2 ∧ operation4 ∧ ¬ operation3) ∨ 
                                   (operation1 ∧ operation2 ∧ ¬ operation4 ∧ operation3) ∨ 
                                   (operation1 ∧ ¬ operation2 ∧ operation4 ∧ operation3) ∨ 
                                   (¬ operation1 ∧ operation2 ∧ operation4 ∧ operation3) := by
  sorry

end three_correct_operations_l633_633818


namespace count_digit_five_1_to_700_l633_633293

def contains_digit_five (n : ℕ) : Prop :=
  n.digits 10 ∈ [5]

def count_up_to (n : ℕ) (p : ℕ → Prop) : ℕ :=
  (finset.range n).count p

theorem count_digit_five_1_to_700 : count_up_to 701 contains_digit_five = 52 := sorry

end count_digit_five_1_to_700_l633_633293


namespace num_ways_to_write_3050_l633_633757

theorem num_ways_to_write_3050 : 
  let M := (0 : ℕ) in
  let count := λ b3: ℕ, if b3 < 3 then 100 else if b3 = 3 then 6 else 0 in
  M + count 0 + count 1 + count 2 + count 3 = 306 :=
by
  sorry

end num_ways_to_write_3050_l633_633757


namespace students_liking_combination_is_correct_l633_633322

noncomputable def class_size : ℕ := 200
noncomputable def blue_percentage : ℝ := 0.20
noncomputable def remaining_students : ℕ := class_size - class_size * blue_percentage
noncomputable def yellow_percentage : ℝ := 0.20
noncomputable def green_percentage : ℝ := 0.25

def blue_students : ℕ := (blue_percentage * class_size).to_nat
def yellow_students : ℕ := (yellow_percentage * remaining_students).to_nat
def green_students : ℕ := (green_percentage * remaining_students).to_nat

def total_students_liking_yellow_green_blue : ℕ := yellow_students + green_students + blue_students

theorem students_liking_combination_is_correct : total_students_liking_yellow_green_blue = 112 :=
by
  -- Proof omitted
  sorry

end students_liking_combination_is_correct_l633_633322


namespace pi_minus_3_to_zero_eq_one_l633_633907

theorem pi_minus_3_to_zero_eq_one : (\pi - 3)^0 = 1 :=
by
  sorry

end pi_minus_3_to_zero_eq_one_l633_633907


namespace sin_product_l633_633358

-- Assume all the given conditions
variables (K L M P O Q R : Type) [triangle K L M] [Median K P L M] [Circumcenter O K L M] [Incenter Q K L M]
          (OR PR QR KR : ℝ) (α β : ℝ)

-- Given conditions
axiom angle_sum_property : α + β + π / 3 = π
axiom given_angle : angle L K M = π / 3
axiom given_relation : OR / PR = sqrt 14 * (QR / KR)

-- Definition of the target theorem
theorem sin_product (h1 : α + β = 2 * π / 3) 
                    (h2 : sin (α + β) = sin (2 * π / 3)) :
                    sin α * sin β = 5 / 8 := 
sorry

end sin_product_l633_633358


namespace turnips_bag_l633_633050

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l633_633050


namespace count_5_in_range_1_to_700_l633_633282

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  n.digits 10 |>.contains d

def count_numbers_with_digit (d : ℕ) (m : ℕ) : ℕ :=
  (List.range' 1 m) |>.filter (contains_digit d) |>.length

theorem count_5_in_range_1_to_700 : count_numbers_with_digit 5 700 = 214 := by
  sorry

end count_5_in_range_1_to_700_l633_633282


namespace area_in_sq_yds_l633_633952

-- Definitions based on conditions
def side_length_ft : ℕ := 9
def sq_ft_per_sq_yd : ℕ := 9

-- Statement to prove
theorem area_in_sq_yds : (side_length_ft * side_length_ft) / sq_ft_per_sq_yd = 9 :=
by
  sorry

end area_in_sq_yds_l633_633952


namespace smallest_n_for_negative_sum_l633_633704

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d, ∀ n, a (n + 1) = a n + d

theorem smallest_n_for_negative_sum (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 1 > 0) 
  (h3 : a 2022 + a 2023 > 0) 
  (h4 : a 2022 * a 2023 < 0) : 
  ∃ n : ℕ, (n ≥ 1) ∧ (sum (range n) a < 0) ∧ (∀ m : ℕ, (m < n → m ≥ 1 → sum (range m) a ≥ 0)) :=
begin
  use 4045,
  sorry
end

end smallest_n_for_negative_sum_l633_633704


namespace money_last_weeks_l633_633365

-- Conditions
def money_from_mowing : ℕ := 14
def money_from_weeding : ℕ := 31
def weekly_spending : ℕ := 5

-- Total money made
def total_money : ℕ := money_from_mowing + money_from_weeding

-- Expected result
def expected_weeks : ℕ := 9

-- Prove the number of weeks the money will last Jerry
theorem money_last_weeks : (total_money / weekly_spending) = expected_weeks :=
by
  sorry

end money_last_weeks_l633_633365


namespace mike_pens_l633_633502

-- Definitions based on the conditions
def initial_pens : ℕ := 25
def pens_after_mike (M : ℕ) : ℕ := initial_pens + M
def pens_after_cindy (M : ℕ) : ℕ := 2 * pens_after_mike M
def pens_after_sharon (M : ℕ) : ℕ := pens_after_cindy M - 19
def final_pens : ℕ := 75

-- The theorem we need to prove
theorem mike_pens (M : ℕ) (h : pens_after_sharon M = final_pens) : M = 22 := by
  have h1 : pens_after_sharon M = 2 * (25 + M) - 19 := rfl
  rw [h1] at h
  sorry

end mike_pens_l633_633502


namespace problem_intersection_points_l633_633168

noncomputable theory

def fractional_part (x : ℝ) : ℝ := x - floor x

theorem problem_intersection_points :
  let eq1 := (fractional_part x)^2 + y^2 = fractional_part x
  let eq2 := y = (3/5) * x
  12 = number_of_intersection_points eq1 eq2 :=
begin
  sorry
end

end problem_intersection_points_l633_633168


namespace round_trip_ticket_percentage_l633_633001

theorem round_trip_ticket_percentage (P R : ℝ) 
  (h1 : 0.20 * P = 0.50 * R) : R = 0.40 * P :=
by
  sorry

end round_trip_ticket_percentage_l633_633001


namespace range_OA_OB_l633_633326

noncomputable def curve_C1_polar_equation (θ : ℝ) : ℝ := 2 * (Real.cos θ)

noncomputable def curve_C2_cartesian_equation (x y : ℝ) : Prop :=
  x^2 = 2 * y

noncomputable def ray_intersection_A (k : ℝ) (h₁ : 1 ≤ k) (h₂ : k < Real.sqrt 3) (α : ℝ) : ℝ :=
  2 * (Real.cos α)

noncomputable def ray_intersection_B (k : ℝ) (h₁ : 1 ≤ k) (h₂ : k < Real.sqrt 3) (α : ℝ) : ℝ :=
  (2 * (Real.sin α)) / (Real.cos α)^2

theorem range_OA_OB (k : ℝ) (h₁ : 1 ≤ k) (h₂ : k < Real.sqrt 3) :
  ∃ α : ℝ, 4 * k = 4 * (Real.tan α) :=
by {
  sorry,
}

end range_OA_OB_l633_633326


namespace denis_oleg_probability_l633_633133

theorem denis_oleg_probability :
  let n := 26 in
  let players := {denis, oleg} in
  let total_matches := n - 1 in
  let total_pairs := n * (n - 1) / 2 in
  ∃ (P : ℚ), P = 1 / 13 ∧
  ∀ (i : ℕ), i ∈ fin total_matches →
  let match_pairs := (n - i) * (n - i - 1) / 2 in
  players ⊆ fin match_pairs → P = 1 / 13 := 
sorry

end denis_oleg_probability_l633_633133


namespace find_magnitude_of_b_l633_633688

variables (a b : ℝ × ℝ) -- Define the vectors in 2D
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2) -- Define magnitude of a vector
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2 -- Define dot product

-- Condition 1: |a| = 1
def condition1 : Prop := magnitude a = 1 

-- Condition 2: |a - 2b| = sqrt(21)
def condition2 : Prop := magnitude (a.1 - 2*b.1, a.2 - 2*b.2) = real.sqrt 21

-- Condition 3: The angle between a and b is 120º, which means a · b = |a||b|cos(120º) = -1/2 |b|
def condition3 : Prop := dot_product a b = -1/2 * magnitude b

-- The main proposition: Given the conditions, prove |b| = 2
theorem find_magnitude_of_b (h1 : condition1 a) (h2 : condition2 a b) (h3 : condition3 a b) : magnitude b = 2 :=
sorry

end find_magnitude_of_b_l633_633688


namespace turnip_bag_weights_l633_633070

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l633_633070


namespace book_arrangement_problem_l633_633785

-- Define the number of books in different languages
def arabic_books : ℕ := 2
def german_books_distinguishable : ℕ := 2
def german_books_indistinguishable : ℕ := 2
def spanish_books : ℕ := 4

-- The total number of books
def total_books : ℕ := arabic_books + german_books_distinguishable + german_books_indistinguishable + spanish_books

-- Calculate the total number of ways to arrange the books given the conditions
def total_arrangements (total_books : ℕ) (arabic_books : ℕ) (german_books_distinguishable : ℕ) (german_books_indistinguishable : ℕ) (spanish_books : ℕ) : ℕ := 
  let units := 4! / 2! in -- Arrange 4 units (Arabic, Spanish, German pair, German single) with 2 indistinguishable
  let arrange_arabic := 2! in -- Arrange Arabic books
  let arrange_spanish := 4! in -- Arrange Spanish books
  units * arrange_arabic * arrange_spanish

-- The assertion that needs to be proven
theorem book_arrangement_problem : total_arrangements total_books arabic_books german_books_distinguishable german_books_indistinguishable spanish_books = 576 := 
by
  sorry

end book_arrangement_problem_l633_633785


namespace x_is_perfect_square_l633_633412

theorem x_is_perfect_square (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (hdiv : 2 * x * y ∣ x^2 + y^2 - x) : ∃ (n : ℕ), x = n^2 :=
by
  sorry

end x_is_perfect_square_l633_633412


namespace nail_insertion_l633_633104

theorem nail_insertion (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  (4/7) + (4/7) * k + (4/7) * k^2 = 1 :=
by sorry

end nail_insertion_l633_633104


namespace market_value_calculation_l633_633514

variables (annual_dividend_per_share face_value yield market_value : ℝ)

axiom annual_dividend_definition : annual_dividend_per_share = 0.09 * face_value
axiom face_value_definition : face_value = 100
axiom yield_definition : yield = 0.25

theorem market_value_calculation (annual_dividend_per_share face_value yield market_value : ℝ) 
  (h1: annual_dividend_per_share = 0.09 * face_value)
  (h2: face_value = 100)
  (h3: yield = 0.25):
  market_value = annual_dividend_per_share / yield :=
sorry

end market_value_calculation_l633_633514


namespace triangle_abe_area_l633_633741

theorem triangle_abe_area
  (ABC : Type) [Triangle ABC]
  (A B C D E F : Point ABC)
  (area_ABC : Real)
  (h_area_ABC : area_ABC = 12)
  (AD : Segment A B)
  (DB : Segment A B)
  (h_AD : length AD = 3)
  (h_DB : length DB = 4)
  (h_A_BE_E_F : TriangleAreaEqual ABC A B E D B E F) :
  area (Triangle A B E) = 36 / 7 := 
by 
  sorry

end triangle_abe_area_l633_633741


namespace samantha_spends_36_dollars_l633_633442

def cost_per_toy : ℝ := 12.00
def discount_factor : ℝ := 0.5
def num_toys_bought : ℕ := 4

def total_spent (cost_per_toy : ℝ) (discount_factor : ℝ) (num_toys_bought : ℕ) : ℝ :=
  let pair_cost := cost_per_toy + (cost_per_toy * discount_factor)
  let num_pairs := num_toys_bought / 2
  num_pairs * pair_cost

theorem samantha_spends_36_dollars :
  total_spent cost_per_toy discount_factor num_toys_bought = 36.00 :=
sorry

end samantha_spends_36_dollars_l633_633442


namespace prove_a_value_sum_of_zeros_gt_two_l633_633676

open Real

-- Define the function f and its derivative
def f (a x : ℝ) := a * x - x * log x
def f' (a x : ℝ) := a - log x - 1

-- 1. Prove that if f has an extreme value at x = exp (-2), then a = -1
theorem prove_a_value (a : ℝ) (h : f' a (exp (-2)) = 0) : a = -1 := sorry

-- Define the function F
def F (x : ℝ) := x^2 - log x - x - 1

-- 2. Prove that if F has two distinct zeros x1 and x2, then x1 + x2 > 2
theorem sum_of_zeros_gt_two (x1 x2 : ℝ) (h1 : F x1 = 0) (h2 : F x2 = 0) (h3 : x1 ≠ x2) : x1 + x2 > 2 := sorry

end prove_a_value_sum_of_zeros_gt_two_l633_633676


namespace revenue_equation_l633_633321

theorem revenue_equation (x : ℝ) (r_j r_t : ℝ) (h1 : r_j = 90) (h2 : r_t = 144) :
  r_j + r_j * (1 + x) + r_j * (1 + x)^2 = r_t :=
by
  rw [h1, h2]
  sorry

end revenue_equation_l633_633321


namespace prime_squares_mod_180_l633_633575

theorem prime_squares_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  ∃ r ∈ {1, 49}, ∀ q ∈ {1, 49}, (q ≡ p^2 [MOD 180]) → q = r :=
by
  sorry

end prime_squares_mod_180_l633_633575


namespace train_platform_length_l633_633554

noncomputable def kmph_to_mps (v : ℕ) : ℕ := v * 1000 / 3600

theorem train_platform_length :
  ∀ (train_length speed_kmph time_sec : ℕ),
    speed_kmph = 36 →
    train_length = 175 →
    time_sec = 40 →
    let speed_mps := kmph_to_mps speed_kmph
    let total_distance := speed_mps * time_sec
    let platform_length := total_distance - train_length
    platform_length = 225 :=
by
  intros train_length speed_kmph time_sec h_speed h_train h_time
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_sec
  let platform_length := total_distance - train_length
  sorry

end train_platform_length_l633_633554


namespace toys_produced_each_day_l633_633504

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_worked_per_week : ℕ)
  (same_number_toys_each_day : Prop) : 
  total_weekly_production = 4340 → days_worked_per_week = 2 → 
  same_number_toys_each_day →
  (total_weekly_production / days_worked_per_week = 2170) :=
by
  intros h_production h_days h_same_toys
  -- proof skipped
  sorry

end toys_produced_each_day_l633_633504


namespace max_and_min_values_l633_633202

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

theorem max_and_min_values :
  (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x ≤ 5) ∧ (∃ c ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f c = 5) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), 1 ≤ f x) ∧ (∃ c ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f c = 1) :=
begin
  sorry
end

end max_and_min_values_l633_633202


namespace primes_squared_mod_180_l633_633572

theorem primes_squared_mod_180 (p : ℕ) (hp_prime : Prime p) (hp_gt_5 : p > 5) :
  {r | ∃ k : ℕ, p^2 = 180 * k + r}.card = 2 :=
by
  sorry

end primes_squared_mod_180_l633_633572


namespace turnip_bag_weighs_l633_633088

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l633_633088


namespace rate_percent_per_annum_l633_633304

-- Let P be the principal amount (initial sum of money)
def principal_amount (P : ℝ) : Prop := P > 0

-- Let R be the rate of interest per annum in percent
def rate_percent (R : ℝ) : Prop := R > 0

-- Let T be the time period in years, which is 20 years in this problem
def time_period (T : ℝ) : Prop := T = 20

-- Let the money double, so the final amount is 2P
def doubles_itself (P : ℝ) : Prop := (2 * P)

-- Simple interest formula
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- Given conditions:
axiom principal (P : ℝ) : principal_amount P
axiom rate (R : ℝ) : rate_percent R
axiom time (T : ℝ) : time_period T
axiom final_amount (P : ℝ) : (simple_interest P R T) + P = 2 * P

-- Lean 4 statement to prove
theorem rate_percent_per_annum (P R : ℝ) (T : ℝ) :
  principal_amount P →
  rate_percent R →
  time_period T →
  simple_interest P R T + P = 2 * P →
  R = 5 :=
by 
  intros hP hR hT hFA
  sorry

end rate_percent_per_annum_l633_633304


namespace min_value_of_3x_plus_4y_is_5_l633_633308

theorem min_value_of_3x_plus_4y_is_5 :
  ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → (∃ (b : ℝ), b = 3 * x + 4 * y ∧ ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → 3 * x + 4 * y ≥ b) :=
by
  intro x y x_pos y_pos h_eq
  let b := 5
  use b
  simp [b]
  sorry

end min_value_of_3x_plus_4y_is_5_l633_633308


namespace correct_propositions_l633_633669

/-
  Given the propositions:
  - Proposition 1: "Events A and B are mutually exclusive" is a necessary but not sufficient condition for "Events A and B are complementary."
  - Proposition 2: The inverse proposition of "All congruent triangles are similar triangles" is true.
  - Proposition 3: The negation of the proposition "The diagonals of a rectangle are equal" is false.
  - Proposition 4: In triangle ABC, "∠B=60°" is the necessary and sufficient condition for the angles ∠A, ∠B, ∠C to form an arithmetic sequence.

  Prove that propositions 1 and 3 are correct.
-/

def ev_mut_exclusive_nec_not_suff_complementary : Prop :=
  ∀ A B : Prop, (¬ (A ∧ B)) → A ≠ B

def inv_congruent_similar_triangles_false : Prop :=
  ¬ (∀ (T₁ T₂ : Triangle), T₁ ≅ T₂ → T₁ ∼ T₂)

def neg_diagonals_rectangle_equal_false : Prop :=
  ¬ (∀ (Q : Quadrilateral), (¬ is_rectangle Q) → (¬ (diagonals_equal Q)))

def angle_B_60_nec_suff_arithmetic_sequence : Prop :=
  ∀ (A B C : ℕ) [is_triangle A B C], (B = 60) ↔ (2 * B = A + C)

theorem correct_propositions (P1 P3 : Prop) :
  ev_mut_exclusive_nec_not_suff_complementary → inv_congruent_similar_triangles_false → 
  neg_diagonals_rectangle_equal_false → angle_B_60_nec_suff_arithmetic_sequence → 
  (P1 ∧ P3) :=
by
  intros h1 h2 h3 h4
  exact (h1, h3)

end correct_propositions_l633_633669


namespace turnip_weights_are_13_or_16_l633_633101

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l633_633101


namespace stamps_initial_count_l633_633420

theorem stamps_initial_count (total_stamps stamps_received initial_stamps : ℕ) 
  (h1 : total_stamps = 61)
  (h2 : stamps_received = 27)
  (h3 : initial_stamps = total_stamps - stamps_received) :
  initial_stamps = 34 :=
sorry

end stamps_initial_count_l633_633420


namespace probability_of_two_red_two_blue_l633_633021

-- Definitions for the conditions
def total_red_marbles : ℕ := 12
def total_blue_marbles : ℕ := 8
def total_marbles : ℕ := total_red_marbles + total_blue_marbles
def num_selected_marbles : ℕ := 4
def num_red_selected : ℕ := 2
def num_blue_selected : ℕ := 2

-- Definition for binomial coefficient (combinations)
def C (n k : ℕ) : ℕ := n.choose k

-- Probability calculation
def probability_two_red_two_blue :=
  (C total_red_marbles num_red_selected * C total_blue_marbles num_blue_selected : ℚ) / C total_marbles num_selected_marbles

-- The theorem statement
theorem probability_of_two_red_two_blue :
  probability_two_red_two_blue = 1848 / 4845 :=
by
  sorry

end probability_of_two_red_two_blue_l633_633021


namespace combined_length_is_correct_l633_633775

-- Definitions of the total heights
def height_Aisha : ℝ := 174
def height_Benjamin : ℝ := 190

-- Proportions of legs and arms
def legs_Aisha : ℝ := (1/3) * height_Aisha
def arms_Aisha : ℝ := (1/6) * height_Aisha

def legs_Benjamin : ℝ := (3/7) * height_Benjamin
def arms_Benjamin : ℝ := (1/4) * height_Benjamin

-- Combined lengths of legs and arms
def combined_legs : ℝ := legs_Aisha + legs_Benjamin
def combined_arms : ℝ := arms_Aisha + arms_Benjamin

-- Final combined length of legs and arms
def total_combined_length : ℝ := combined_legs + combined_arms

-- The main theorem to prove
theorem combined_length_is_correct : total_combined_length = 215.93 :=
by
  -- This sorry will be replaced by the actual proof
  sorry

end combined_length_is_correct_l633_633775


namespace turnip_bag_weight_l633_633058

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l633_633058


namespace probability_sequence_inequality_l633_633951

theorem probability_sequence_inequality (n : ℕ) (h : n > 3) :
  let s := {i | i ∈ permutation (finset.range n)} in
  let favorable := {i ∈ s | ∀ k ∈ finset.range n, i k ≥ k - 3} in
  (fintype.card favorable : ℚ) / (fintype.card s : ℚ) = (4^(n-3) * 6) / n! :=
by sorry

end probability_sequence_inequality_l633_633951


namespace probability_Denis_Oleg_play_l633_633139

theorem probability_Denis_Oleg_play (n : ℕ) (h : n = 26) :
  let C := λ (n : ℕ), n * (n - 1) / 2 in
  (n - 1 : ℚ) / C n = 1 / 13 :=
by 
  sorry

end probability_Denis_Oleg_play_l633_633139


namespace turnip_weight_possible_l633_633080

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l633_633080


namespace average_marathon_time_is_7_hours_l633_633980

-- Definitions of the problem conditions
def Casey_time : ℕ := 6
def Zendaya_additional_time (C : ℕ) : ℕ := (1 / 3 : ℚ) * C

@[simp]
theorem average_marathon_time_is_7_hours :
  let Zendaya_time := Casey_time + Zendaya_additional_time Casey_time,
      combined_time := Casey_time + Zendaya_time,
      average_time := combined_time / 2
  in average_time = 7 :=
by
  sorry

end average_marathon_time_is_7_hours_l633_633980


namespace sequence_a₄_l633_633350

noncomputable def arithmetic_sequence (a₁ d : ℤ) : ℕ → ℤ
| 0     := a₁
| (n+1) := arithmetic_sequence a₁ d n + d

theorem sequence_a₄ : arithmetic_sequence (-1) (-3) 3 = -10 := by
  sorry

end sequence_a₄_l633_633350


namespace domain_of_g_l633_633899

noncomputable def g (t : ℝ) : ℝ := 1 / ((t - 2)^2 + (t + 2)^2 + 1)

theorem domain_of_g : ∀ t : ℝ, (t - 2)^2 + (t + 2)^2 + 1 > 0 := 
by {
  intro t,
  calc
    (t - 2)^2 + (t + 2)^2 + 1
      = (t^2 - 4t + 4) + (t^2 + 4t + 4) + 1 : by ring
  ... = 2t^2 + 9 : by ring
  ... > 0 : by {
    apply lt_add_of_le_of_pos,
    apply mul_nonneg,
    apply zero_le_two,
    apply pow_two_nonneg,
    norm_num,
    },
}

end domain_of_g_l633_633899


namespace sequence_digits_count_l633_633500

/-- The total number of digits written in the sequence of consecutive natural odd numbers starting 
from 1 until the first occurrence of "2014" is 7850. -/
theorem sequence_digits_count : 
  let count_single_digit := 5 in
  let count_two_digits := (99 - 11) / 2 + 1 in
  let count_three_digits := (999 - 101) / 2 + 1 in
  let count_four_digits := (2001 - 1001) / 2 + 1 in
  let total_digits := 
    (count_single_digit * 1) + (count_two_digits * 2) + (count_three_digits * 3) + 
    (count_four_digits * 4) + 1 in
  total_digits = 7850 := 
by 
  let count_single_digit := 5
  let count_two_digits := (99 - 11) / 2 + 1
  let count_three_digits := (999 - 101) / 2 + 1
  let count_four_digits := (2001 - 1001) / 2 + 1
  let total_digits := 
    (count_single_digit * 1) + (count_two_digits * 2) + (count_three_digits * 3) + 
    (count_four_digits * 4) + 1
  show total_digits = 7850
  sorry

end sequence_digits_count_l633_633500


namespace exists_ints_a_b_l633_633212

theorem exists_ints_a_b (n : ℤ) (h : n % 4 ≠ 2) : ∃ a b : ℤ, n + a^2 = b^2 :=
by
  sorry

end exists_ints_a_b_l633_633212


namespace percentage_decrease_l633_633466

/-- The percentage decrease in production value of the year before last compared to last year -/
theorem percentage_decrease (P : ℝ) (x : ℝ) : 
  let y := x * (1 + P / 100) in
  (y - x) / y = P / (100 + P) :=
by
  let y := x * (1 + P / 100)
  sorry

end percentage_decrease_l633_633466


namespace range_k_for_symmetric_points_l633_633226

theorem range_k_for_symmetric_points (f g : ℝ → ℝ) (k : ℝ)
  (hx : ∀ x ∈ Icc (1 / Real.exp 1) (Real.exp 1), f x = k * x)
  (hy : ∀ x ∈ Icc (1 / Real.exp 1) (Real.exp 1), g x = (1 / Real.exp 1) ^ (x / 2))
  : ∃ M N : ℝ × ℝ, 
    (∃ x ∈ Icc (1 / Real.exp 1) (Real.exp 1), M = (x, f x)) ∧
    (∃ x ∈ Icc (1 / Real.exp 1) (Real.exp 1), N = (g x, x)) ∧ 
    M = (N.2, N.1) → -2 / Real.exp 1 ≤ k ∧ k ≤ 2 * Real.exp 1 := 
sorry

end range_k_for_symmetric_points_l633_633226


namespace convert_110110001_to_base4_l633_633175

def binary_to_base4_conversion (b : ℕ) : ℕ :=
  -- assuming b is the binary representation of the number to be converted
  1 * 4^4 + 3 * 4^3 + 2 * 4^2 + 0 * 4^1 + 1 * 4^0

theorem convert_110110001_to_base4 : binary_to_base4_conversion 110110001 = 13201 :=
  sorry

end convert_110110001_to_base4_l633_633175


namespace median_is_2_mode_is_0_l633_633900

noncomputable def moon_counts : List ℕ := [0, 0, 0, 1, 1, 2, 2, 5, 15, 18, 25]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get (l.length / 2)

def frequency (l : List ℕ) (n : ℕ) : ℕ := l.count n

def mode (l : List ℕ) : ℕ :=
  let freqs := l.toFinset.data.map (fun n => (n, frequency l n))
  let max_freq := freqs.maximumBy Prod.snd sorry  -- Finset lacks well-defined methods in Lean 4
  max_freq.fst

theorem median_is_2 : median moon_counts = 2 := by
  sorry

theorem mode_is_0 : mode moon_counts = 0 := by
  sorry

end median_is_2_mode_is_0_l633_633900


namespace problem_statement_l633_633243

noncomputable def find_sum (x y : ℝ) : ℝ := x + y

theorem problem_statement (x y : ℝ)
  (hx : |x| + x + y = 12)
  (hy : x + |y| - y = 14) :
  find_sum x y = 22 / 5 :=
sorry

end problem_statement_l633_633243


namespace dislike_tv_and_sports_count_l633_633781

def total_people := 1500
def fraction_dislike_tv := 0.4
def fraction_dislike_both := 0.15

theorem dislike_tv_and_sports_count : 
  let dislike_tv := total_people * fraction_dislike_tv in
  let dislike_both := dislike_tv * fraction_dislike_both in
  dislike_both = 90 := by
    sorry

end dislike_tv_and_sports_count_l633_633781


namespace final_student_count_l633_633585

def initial_students := 150
def students_joined := 30
def students_left := 15

theorem final_student_count : initial_students + students_joined - students_left = 165 := by
  sorry

end final_student_count_l633_633585


namespace probability_of_two_red_two_blue_l633_633020

-- Definitions for the conditions
def total_red_marbles : ℕ := 12
def total_blue_marbles : ℕ := 8
def total_marbles : ℕ := total_red_marbles + total_blue_marbles
def num_selected_marbles : ℕ := 4
def num_red_selected : ℕ := 2
def num_blue_selected : ℕ := 2

-- Definition for binomial coefficient (combinations)
def C (n k : ℕ) : ℕ := n.choose k

-- Probability calculation
def probability_two_red_two_blue :=
  (C total_red_marbles num_red_selected * C total_blue_marbles num_blue_selected : ℚ) / C total_marbles num_selected_marbles

-- The theorem statement
theorem probability_of_two_red_two_blue :
  probability_two_red_two_blue = 1848 / 4845 :=
by
  sorry

end probability_of_two_red_two_blue_l633_633020


namespace problem_part_1_problem_part_2_l633_633244

noncomputable def f (x m : ℝ) : ℝ := x^2 + m * x - 1

theorem problem_part_1 (m n : ℝ) :
  (∀ x, f x m < 0 ↔ -2 < x ∧ x < n) → m = 5 / 2 ∧ n = 1 / 2 :=
sorry

theorem problem_part_2 (m : ℝ) :
  (∀ x, m ≤ x ∧ x ≤ m + 1 → f x m < 0) → m ∈ Set.Ioo (-Real.sqrt (2) / 2) 0 :=
sorry

end problem_part_1_problem_part_2_l633_633244


namespace railway_plan_theorem_l633_633486

noncomputable def construct_railway_plan (A B C D E : Point) (AB BC AC : ℝ) (BD : ℝ) (scale: ℝ) (angle_105: ℝ) : Prop :=
  let AB_scaled := AB * scale in
  let BC_scaled := BC * scale in
  let AC_scaled := AC * scale in
  let BD_scaled := BD * scale in
  AB = 5 ∧ BC = 8.8 ∧ AC = 12 ∧ BD = 1 ∧
  scale = 1 / 100000 ∧
  angle_105 = 105 ∧ 
  -- Constructing geometry
  ∃ (triangle : Triangle), 
  triangle.sides = (AB_scaled, BC_scaled, AC_scaled) ∧
  ∃ (railway_plan : RailwayPlan),
  railway_plan.start = A ∧
  railway_plan.through = B ∧
  railway_plan.endpoint = D ∧
  angle (D, B, E) = angle_105

theorem railway_plan_theorem : 
  ∃ (railway_plan : RailwayPlan), 
  construct_railway_plan A B C D E 5 8.8 12 1 (1 / 100000) 105 = 
  true := sorry

end railway_plan_theorem_l633_633486


namespace sum_of_common_ratios_l633_633394

variable {k p r : ℝ}

-- Condition 1: geometric sequences with distinct common ratios
-- Condition 2: a_3 - b_3 = 3(a_2 - b_2)
def geometric_sequences (k p r : ℝ) : Prop :=
  (k ≠ 0) ∧ (p ≠ r) ∧ (k * p^2 - k * r^2 = 3 * (k * p - k * r))

theorem sum_of_common_ratios (k p r : ℝ) (h : geometric_sequences k p r) : p + r = 3 :=
by
  sorry

end sum_of_common_ratios_l633_633394


namespace arithmetic_sequence_example_l633_633727

theorem arithmetic_sequence_example (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h2 : a 2 = 2) (h14 : a 14 = 18) : a 8 = 10 :=
by
  sorry

end arithmetic_sequence_example_l633_633727


namespace tan_alpha_value_l633_633266

/-- Statement of the problem -/
theorem tan_alpha_value (α : ℝ)
  (h1 : ∃ k : ℝ, (sin α = k * 1 ∧ (cos α - 2 * sin α) = k * 2)) :
  tan α = 1 / 4 :=
sorry

end tan_alpha_value_l633_633266


namespace probability_Denis_Oleg_play_l633_633141

theorem probability_Denis_Oleg_play (n : ℕ) (h : n = 26) :
  let C := λ (n : ℕ), n * (n - 1) / 2 in
  (n - 1 : ℚ) / C n = 1 / 13 :=
by 
  sorry

end probability_Denis_Oleg_play_l633_633141


namespace inequality_square_l633_633706

theorem inequality_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 :=
sorry

end inequality_square_l633_633706


namespace turnips_bag_l633_633051

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l633_633051


namespace locus_equation_perimeter_greater_than_3sqrt3_l633_633328

-- The locus W and the conditions 
def locus_eq (P : ℝ × ℝ) : Prop :=
  P.snd = P.fst ^ 2 + 1/4

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

-- Prove part (1): The locus W is y = x^2 + 1/4
theorem locus_equation (x y : ℝ) : 
  point_on_x_axis (x, y) → locus_eq (x, y) :=
sorry

-- Prove part (2): Perimeter of rectangle ABCD is greater than 3sqrt(3) if three vertices are on W
def rectangle_on_w (A B C : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  locus_eq A ∧ locus_eq B ∧ locus_eq C ∧ locus_eq D ∧ 
  (∃x₁ x₂ x₃ x₄ : ℝ, A = (x₁, x₁ ^ 2 + 1/4) ∧ B = (x₂, x₂ ^ 2 + 1/4) ∧ 
  C = (x₃, x₃ ^ 2 + 1/4) ∧ D = (x₄, x₄ ^ 2 + 1/4))

theorem perimeter_greater_than_3sqrt3 (A B C D : ℝ × ℝ) : 
  rectangle_on_w A B C D → 
  2 * (abs (B.fst - A.fst) + abs (C.fst - B.fst)) > 3 * sqrt 3 :=
sorry

end locus_equation_perimeter_greater_than_3sqrt3_l633_633328


namespace range_of_a_l633_633004

def is_monotonically_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, (0 < x) → (x < y) → (f x ≤ f y)

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem range_of_a (a : ℝ) : 
  is_monotonically_increasing (f a) a → a ≤ 2 :=
sorry

end range_of_a_l633_633004


namespace turnip_bag_weight_l633_633063

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l633_633063


namespace regular_polygon_sides_l633_633712

theorem regular_polygon_sides (θ : ℝ) (h : θ = 72) : ∃ n : ℕ, ∀ (h : θ = 360 / n), n = 5 :=
by {
  use 5,
  intro h,
  rw h,
  exact (by norm_num : 360 / 5 = 72),
  exact h.symm,
  sorry
}

end regular_polygon_sides_l633_633712


namespace original_price_of_iPhone_X_l633_633549

noncomputable def smartphone_discount_problem :=
  let P : ℝ := 600
  (seller_discount_percentage : ℝ := 5 / 100)
  (number_of_phones : ℕ := 3)
  (savings_amount : ℝ := 90) in
  true → 
    P * number_of_phones - (P * number_of_phones * seller_discount_percentage) = 
    P * number_of_phones - savings_amount

theorem original_price_of_iPhone_X (P : ℝ) (discount_percentage : ℝ) (num_phones : ℕ) (saving : ℝ) 
  (h : saving = P * num_phones * discount_percentage) : P = 600 := by 
  sorry

end original_price_of_iPhone_X_l633_633549


namespace green_dots_second_row_l633_633405

theorem green_dots_second_row :
  ∀ n1 n3 n4 n5 n2 : ℕ, n1 = 3 → n3 = 9 → n4 = 12 → n5 = 15 → (n3 - n1 = n4 - n3) → 
  (n4 - n3 = n5 - n4) → n2 = n3 - (n4 - n3) → n2 = 6 :=
by {
  intros n1 n3 n4 n5 n2,
  intros h1 h3 h4 h5 h6 h7 h8,
  rw [h1, h3, h4, h5, h6, h7, h8],
  exact rfl,
}

end green_dots_second_row_l633_633405


namespace determine_worker_machines_l633_633881

def Worker := Type
constant Dan : Worker
constant Emma : Worker
constant Fiona : Worker

def Machine := Type
constant MachineA : Machine
constant MachineB : Machine
constant MachineC : Machine

def operates : Worker → Machine → Prop

axiom unique_true_statement :
  (operates Emma MachineA = false ∧ operates Dan MachineC = true ∧ operates Fiona MachineB = false ∨
  operates Emma MachineA = true ∧ operates Dan MachineC = false ∧ operates Fiona MachineB = false ∨
  operates Emma MachineA = false ∧ operates Dan MachineC = false ∧ operates Fiona MachineB = true) ∧
  ((operates Emma MachineA = false) ∨ (operates Dan MachineC = true) ∨ (operates Fiona MachineB = true)) ∧
  (¬ (operates Emma MachineA = false ∧ operates Dan MachineC = true ∧ operates Fiona MachineB = true))

theorem determine_worker_machines :
  operates Dan MachineC ∧ operates Emma MachineA ∧ operates Fiona MachineB :=
by
  -- The proof will go here, but we use sorry to indicate it's omitted.
  sorry

end determine_worker_machines_l633_633881


namespace sum_of_integers_l633_633810

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
by
  sorry

end sum_of_integers_l633_633810


namespace scores_median_unchanged_l633_633933

theorem scores_median_unchanged {S : List ℝ} (h_len : S.length = 9) :
  let S' := List.erase (List.erase S (List.maximum S)) (List.minimum S) in
  List.median S = List.median S' := by
sorry

end scores_median_unchanged_l633_633933


namespace part_I_part_II_part_III_l633_633250

section Problem

variable (f : ℝ → ℝ) (a : ℝ) (x : ℝ)

-- (I) Prove that when a = -2, f(x) is increasing on (1, +∞)
theorem part_I (a := -2) (h1 : ∀ x > 0, f x = x^2 + a * Real.log x) 
  : ∀ x > 1,  ∃ ε > 0, ∀ y ∈ Ioc x (x + ε), f y > f x := sorry

-- (II) Find the minimum value of f(x) on [1,e] and corresponding x
theorem part_II (h2 : ∀ x ∈ Icc 1 (Real.exp 1), f x = x^2 + a * Real.log x) :
  (∀ a ≥ -2, ∃ x ∈ Icc 1 (Real.exp 1), f x = 1) ∧ 
  (∀ a (ha : -2 * Real.exp 2 < a) (hb : a < -2), ∃ x ∈ Icc 1 (Real.exp 1), f x = a/2 * Real.log (-a/2) - a/2) ∧
  (∀ a ≤ -2 * Real.exp 2, ∃ x ∈ Icc 1 (Real.exp 1), f x = a + Real.exp 2) := sorry

-- (III) Prove that if f(x) ≤ (a + 2)x for x ∈ [1, e], the range of a is [-1, +∞)
theorem part_III (h3 : ∀ x ∈ Icc 1 (Real.exp 1), f x = x^2 + a * Real.log x) 
  (h4 : ∀ x ∈ Icc 1 (Real.exp 1), f x ≤ (a + 2) * x) : a ≥ -1 := sorry

end Problem

end part_I_part_II_part_III_l633_633250


namespace product_simplifies_to_one_over_35_l633_633159

-- Define sequence product
noncomputable def sequence_product : ℚ :=
  ∏ k in finset.range 64, (k + 1) / (k + 5)

-- Proof statement
theorem product_simplifies_to_one_over_35 :
  sequence_product = 1 / 35 :=
by
  sorry

end product_simplifies_to_one_over_35_l633_633159


namespace simplify_fifth_root_l633_633496

theorem simplify_fifth_root (c d : ℕ) (h₁ : c * nat.root 5 (2^11 * 3^5) = nat.root 5 (2^(11) * 3^(5))) : c + d = 518 :=
sorry

end simplify_fifth_root_l633_633496


namespace primes_squared_mod_180_l633_633571

theorem primes_squared_mod_180 (p : ℕ) (hp_prime : Prime p) (hp_gt_5 : p > 5) :
  {r | ∃ k : ℕ, p^2 = 180 * k + r}.card = 2 :=
by
  sorry

end primes_squared_mod_180_l633_633571


namespace binomial_expansion_sum_zero_l633_633517

open BigOperators

theorem binomial_expansion_sum_zero (a b m n : ℕ) (k : ℕ) (h1 : n ≥ 2) (h2 : ab ≠ 0) (h3 : a = m * b) (h4 : m = k + 2) (h5 : ∀ (i : ℕ), i = 3 ∨ i = 4 → ∑ i in range n, binomial n i * b^(n-i) * a^i = 0) : n = 2 * m + 3 := 
sorry

end binomial_expansion_sum_zero_l633_633517


namespace printer_Z_time_l633_633911

def time_for_X := 12
def time_for_Y := 10
def ratio := 1.8

def rate_X := 1 / time_for_X
def rate_Y := 1 / time_for_Y

theorem printer_Z_time (time_Z : ℝ) (h : 1 / ((1 / time_for_Y) + (1 / time_Z)) = time_for_X / ratio) : time_Z = 20 :=
by
  sorry

end printer_Z_time_l633_633911


namespace inequality_abc_l633_633661

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / (a * b * c) + 1 ≥ 3 * (1 / (a^2 + b^2 + c^2) + 1 / (a + b + c)) :=
by
  sorry

end inequality_abc_l633_633661


namespace turnip_bag_weight_l633_633065

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l633_633065


namespace no_snow_five_days_l633_633837

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633837


namespace cone_surface_area_eq_3pi_l633_633533

def base_radius := 1
def height := Real.sqrt 3

theorem cone_surface_area_eq_3pi :
  let r := base_radius
  let h := height
  let l := Real.sqrt (r^2 + h^2)
  let S := Real.pi * r^2 + Real.pi * r * l
  S = 3 * Real.pi :=
by
  let r := base_radius
  let h := height
  let l := Real.sqrt (r^2 + h^2)
  let S := Real.pi * r^2 + Real.pi * r * l
  sorry

end cone_surface_area_eq_3pi_l633_633533


namespace geometric_mean_condition_of_triangle_l633_633718

variable {A B C D : Type} [Real (BC a : Real)] [triangle : IsTriangle A B C]

theorem geometric_mean_condition_of_triangle :
  ∀ (AD BD DC : Real), BC = a ∧ 
  (AD ^ 2 = BD * DC) →
  (b + c) <= a * Real.sqrt 2 :=
by
sorrry

end geometric_mean_condition_of_triangle_l633_633718


namespace ArianaBoughtTulips_l633_633565

theorem ArianaBoughtTulips (total_flowers : ℕ) (fraction_roses : ℚ) (carnations : ℕ) 
    (h_total : total_flowers = 40) (h_fraction : fraction_roses = 2/5) (h_carnations : carnations = 14) : 
    total_flowers - (total_flowers * fraction_roses + carnations) = 10 := by
  sorry

end ArianaBoughtTulips_l633_633565


namespace number_of_perpendicular_points_on_ellipse_l633_633011

theorem number_of_perpendicular_points_on_ellipse :
  let e : Ellipse := ⟨8, 4, ⟨0, 0⟩⟩;
  let F₁ := ⟨2, 0⟩;
  let F₂ := ⟨-2, 0⟩;
  (∃ P : Point, e.includes P ∧ P.dist(F₁).perp P.dist(F₂)) = 2 :=
sorry

end number_of_perpendicular_points_on_ellipse_l633_633011


namespace handshake_max_l633_633720

theorem handshake_max (N : ℕ) (hN : N > 4) (pN pNm1 : ℕ) 
    (hpN : pN ≠ pNm1) (h1 : ∃ p1, pN ≠ p1) (h2 : ∃ p2, pNm1 ≠ p2) :
    ∀ (i : ℕ), i ≤ N - 2 → i ≤ N - 2 :=
sorry

end handshake_max_l633_633720


namespace find_M_l633_633698

theorem find_M :
  (∃ (M : ℕ), (5 + 7 + 9) / 3 = (4020 + 4021 + 4022) / M) → M = 1723 :=
  by
  sorry

end find_M_l633_633698


namespace total_length_XYZ_l633_633171

theorem total_length_XYZ :
  let straight_segments := 7
  let slanted_segments := 7 * Real.sqrt 2
  straight_segments + slanted_segments = 7 + 7 * Real.sqrt 2 :=
by
  sorry

end total_length_XYZ_l633_633171


namespace number_of_even_five_digit_numbers_l633_633535

theorem number_of_even_five_digit_numbers :
  let digits := {0, 1, 2, 3, 4, 5}
  ∃ n ∈ digits.permutations.filter (fun l => l.length = 5 ∧ l.head ≠ 5 ∧ l.tail.head ≠ 0 ∧ l.last ∈ {0, 2, 4} ∧ l.head < 5 ∧ list_to_num l < 50000), 
  list_to_num n = 240 :=
by {
  sorry
}

end number_of_even_five_digit_numbers_l633_633535


namespace sum_of_roots_l633_633448

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -f (-x)
else if 0 < x ∧ x <= 1 then 2^x
else if 1 < x then 1 / 2 * f (x - 1)
else 0 -- placeholder for f(0)

theorem sum_of_roots : 
  (∑ x in finset.filter (λ x, f x = 1 / x) (finset.Icc (-3 : ℝ) 5), x) = 4 := 
sorry

end sum_of_roots_l633_633448


namespace geometric_sequence_sum_inequality_l633_633230

theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_geom : ∀ k, a (k + 1) = a k * q)
  (h_pos : ∀ k ≤ 7, a k > 0)
  (h_q_ne_one : q ≠ 1) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end geometric_sequence_sum_inequality_l633_633230


namespace average_time_l633_633987

variable casey_time : ℝ
variable zendaya_multiplier : ℝ

theorem average_time (h1 : casey_time = 6) (h2 : zendaya_multiplier = 1/3) :
  (casey_time + (casey_time + zendaya_multiplier * casey_time)) / 2 = 7 := by
  sorry

end average_time_l633_633987


namespace max_cake_pieces_l633_633488

-- Define the dimensions of the cake and pieces.
def cake_side : ℕ := 50
def piece_4x4_side : ℕ := 4
def piece_6x6_side : ℕ := 6
def piece_8x8_side : ℕ := 8

-- Calculate the maximum number of pieces 
def max_number_of_pieces (cake_side piece_8x8_side piece_4x4_side : ℕ) : ℕ :=
let pieces_8x8 := (cake_side / piece_8x8_side) * (cake_side / piece_8x8_side) in
let remaining_len := cake_side - pieces_8x8 * piece_8x8_side in
let pieces_4x4 := 2 * (cake_side / piece_4x4_side) in
pieces_8x8 + pieces_4x4

-- Lean statement to prove
theorem max_cake_pieces : max_number_of_pieces cake_side piece_8x8_side piece_4x4_side = 60 :=
by sorry

end max_cake_pieces_l633_633488


namespace equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l633_633342

theorem equation_of_W (P : ℝ × ℝ) :
  let x := P.1 in let y := P.2 in
  |y| = real.sqrt (x^2 + (y - 1/2)^2) ↔ y = x^2 + 1/4 :=
by sorry

theorem rectangle_perimeter_greater_than_3sqrt3 {A B C D : ℝ × ℝ}
  (hA : A.2 = A.1^2 + 1/4) (hB : B.2 = B.1^2 + 1/4) (hC : C.2 = C.1^2 + 1/4)
  (hAB_perp_BC : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  2 * ((real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) + (real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2))) > 3 * real.sqrt 3 :=
by sorry

end equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l633_633342


namespace S_eq_T_l633_633687

noncomputable
def S (n : ℕ) : ℚ :=
if n = 0 then 0 else (1 - ((list.range n).sum (λ k, (-1) ^ k * (1 / (k + 1)))))

noncomputable
def T (n : ℕ) : ℚ :=
if n = 0 then 0 else ((list.range (n + 1)).sum (λ k, 1 / (k + n + 1)))

theorem S_eq_T (n : ℕ) (hn : n > 0) : S n = T n := by
  intro n hn
  sorry

end S_eq_T_l633_633687


namespace solve_system_l633_633429

theorem solve_system :
  { (x : ℝ) × (y : ℝ) // x^2 + y^2 ≤ 2 ∧ 81 * x^4 - 18 * x^2 * y^2 + y^4 - 360 * x^2 - 40 * y^2 + 400 = 0 }
    = {((x, y) : ℝ × ℝ) |
         (x = -3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
         (x = -3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
         (x = 3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
         (x = 3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) } :=
by
  sorry

end solve_system_l633_633429


namespace cost_of_each_card_is_2_l633_633367

-- Define the conditions
def christmas_cards : ℕ := 20
def birthday_cards : ℕ := 15
def total_spent : ℝ := 70

-- Define the total number of cards
def total_cards : ℕ := christmas_cards + birthday_cards

-- Define the cost per card
noncomputable def cost_per_card : ℝ := total_spent / total_cards

-- The theorem
theorem cost_of_each_card_is_2 : cost_per_card = 2 := by
  sorry

end cost_of_each_card_is_2_l633_633367


namespace notebook_pen_cost_l633_633564

theorem notebook_pen_cost :
  ∃ (n p : ℕ), 15 * n + 4 * p = 160 ∧ n > p ∧ n + p = 18 := 
sorry

end notebook_pen_cost_l633_633564


namespace triangle_inequality_l633_633388

variables {a b c S_triangle : ℝ}

-- Assume the conditions
variables (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (ha_non_neg : a > 0) (hb_non_neg : b > 0) (hc_non_neg : c > 0)
  (S_triangle_non_neg : S_triangle > 0)

-- State the theorem
theorem triangle_inequality :
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (S_triangle > 0) →
  (a^3 / ((a - b) * (a - c)) + b^3 / ((b - c) * (b - a)) + c^3 / ((c - a) * (c - b))
  > 2 * 3^(3/4) * S_triangle^(1/2)) :=
begin
  intro h,
  sorry
end

end triangle_inequality_l633_633388


namespace log_3_6561_eq_8_l633_633612

theorem log_3_6561_eq_8 : log 3 6561 = 8 := by
  have h1 : 6561 = 3 ^ 8 := by norm_num
  have h2 : log 3 (3 ^ 8) = 8 * log 3 3 := by rw log_pow (by norm_num : 3 > 1) h1
  have h3 : log 3 3 = 1 := log_self (ne_of_gt (by norm_num : 3 > 1))
  rw [h3, mul_one] at h2
  exact h2

end log_3_6561_eq_8_l633_633612


namespace denis_oleg_probability_l633_633130

theorem denis_oleg_probability :
  let n := 26 in
  let players := {denis, oleg} in
  let total_matches := n - 1 in
  let total_pairs := n * (n - 1) / 2 in
  ∃ (P : ℚ), P = 1 / 13 ∧
  ∀ (i : ℕ), i ∈ fin total_matches →
  let match_pairs := (n - i) * (n - i - 1) / 2 in
  players ⊆ fin match_pairs → P = 1 / 13 := 
sorry

end denis_oleg_probability_l633_633130


namespace incorrect_statement_is_A_l633_633999

theorem incorrect_statement_is_A :
  (∀ (w h : ℝ), w * (2 * h) ≠ 3 * (w * h)) ∧
  (∀ (s : ℝ), (2 * s) ^ 2 = 4 * (s ^ 2)) ∧
  (∀ (s : ℝ), (2 * s) ^ 3 = 8 * (s ^ 3)) ∧
  (∀ (w h : ℝ), (w / 2) * (3 * h) = (3 / 2) * (w * h)) ∧
  (∀ (l w : ℝ), (2 * l) * (3 * w) = 6 * (l * w)) →
  ∃ (incorrect_statement : String), incorrect_statement = "A" := 
by 
  sorry

end incorrect_statement_is_A_l633_633999


namespace turnip_weight_possible_l633_633084

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l633_633084


namespace measure_of_angle_C_l633_633216

variable {a b c S : ℝ}
variable {A B C : ℝ} [Cpos : 0 < C] [Cpi : C < Real.pi]

theorem measure_of_angle_C (hS : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2))
                           (hcos : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) 
                           (habc : a^2 + b^2 - c^2 = 2 * a * b * Real.cos C): 
                           C = Real.pi / 3 := 
by
  sorry

end measure_of_angle_C_l633_633216


namespace not_snowing_next_five_days_l633_633857

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l633_633857


namespace log_expression_simplification_l633_633010

open Real

theorem log_expression_simplification :
    -2 * log 5 10 - log 5 0.25 + 2 = 0 := 
sorry

end log_expression_simplification_l633_633010


namespace locus_equation_perimeter_greater_than_3sqrt3_l633_633330

-- The locus W and the conditions 
def locus_eq (P : ℝ × ℝ) : Prop :=
  P.snd = P.fst ^ 2 + 1/4

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

-- Prove part (1): The locus W is y = x^2 + 1/4
theorem locus_equation (x y : ℝ) : 
  point_on_x_axis (x, y) → locus_eq (x, y) :=
sorry

-- Prove part (2): Perimeter of rectangle ABCD is greater than 3sqrt(3) if three vertices are on W
def rectangle_on_w (A B C : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  locus_eq A ∧ locus_eq B ∧ locus_eq C ∧ locus_eq D ∧ 
  (∃x₁ x₂ x₃ x₄ : ℝ, A = (x₁, x₁ ^ 2 + 1/4) ∧ B = (x₂, x₂ ^ 2 + 1/4) ∧ 
  C = (x₃, x₃ ^ 2 + 1/4) ∧ D = (x₄, x₄ ^ 2 + 1/4))

theorem perimeter_greater_than_3sqrt3 (A B C D : ℝ × ℝ) : 
  rectangle_on_w A B C D → 
  2 * (abs (B.fst - A.fst) + abs (C.fst - B.fst)) > 3 * sqrt 3 :=
sorry

end locus_equation_perimeter_greater_than_3sqrt3_l633_633330


namespace turnip_weights_are_13_or_16_l633_633094

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l633_633094


namespace probability_Denis_Oleg_play_l633_633142

theorem probability_Denis_Oleg_play (n : ℕ) (h : n = 26) :
  let C := λ (n : ℕ), n * (n - 1) / 2 in
  (n - 1 : ℚ) / C n = 1 / 13 :=
by 
  sorry

end probability_Denis_Oleg_play_l633_633142


namespace point_on_x_axis_coordinates_l633_633307

-- Define the conditions
def lies_on_x_axis (M : ℝ × ℝ) : Prop := M.snd = 0

-- State the problem
theorem point_on_x_axis_coordinates (a : ℝ) :
  lies_on_x_axis (a + 3, a + 1) → (a = -1) ∧ ((a + 3, 0) = (2, 0)) :=
by
  intro h
  rw [lies_on_x_axis] at h
  sorry

end point_on_x_axis_coordinates_l633_633307


namespace math_problem_l633_633588

theorem math_problem :
  ( (16 / 81) ^ -(3 / 4) + log (3 / 7) + log 70 + sqrt ( (log 3) ^ 2 - log 9 + 1 ) = 43 / 8 ) :=
by
  sorry

end math_problem_l633_633588


namespace no_snow_five_days_l633_633851

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633851


namespace question1_question2_question3_l633_633248

noncomputable def f (k x : ℝ) := (sin (k * x) * (sin x) ^ k) + (cos (k * x) * (cos x) ^ k) - (cos (2 * x)) ^ k

noncomputable def g (f a : ℝ) := f / (a + f ^ 2)

theorem question1 (k : ℤ) : ∀ x : ℝ, k = 1 → f 1 x = 1 - cos (2 * x) :=
sorry

theorem question2 (a : ℝ) (h : a > 0) (x : ℝ) : x ∈ Icc 0 (π / 3) → 
  (∀ t ∈ Icc 0 (3 / 2), g (1 - cos (2 * t)) a ≤ max ((sqrt a) / (2 * a)) (6 / (4 * a + 9))) :=
sorry

theorem question3 : ∃ k : ℕ, f k 0 = 0 ∧ ∀ m : ℕ, k = 4 * m - 1 ∧ m > 0 → f k (π / 2) = 0 :=
sorry

end question1_question2_question3_l633_633248


namespace cube_fit_count_cube_volume_percentage_l633_633173

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

end cube_fit_count_cube_volume_percentage_l633_633173


namespace rectangle_to_cylinder_max_volume_ratio_l633_633994

/-- Given a rectangle with a perimeter of 12 and converting it into a cylinder 
with the height being the same as the width of the rectangle, prove that the 
ratio of the circumference of the cylinder's base to its height when the volume 
is maximized is 2:1. -/
theorem rectangle_to_cylinder_max_volume_ratio : 
  ∃ (x : ℝ), (2 * x + 2 * (6 - x)) = 12 → 2 * (6 - x) / x = 2 :=
sorry

end rectangle_to_cylinder_max_volume_ratio_l633_633994


namespace rhombus_area_l633_633447

noncomputable def polynomial := 
  λ z : ℂ, z^4 + 4 * (complex.I) * z^3 + (2 + 2 * (complex.I)) * z^2 + (4 - 4 * (complex.I)) * z - (1 + 4 * (complex.I))

theorem rhombus_area :
  ∃ a b c d : ℂ, (polynomial a = 0 ∧ polynomial b = 0 ∧ polynomial c = 0 ∧ polynomial d = 0) ∧ 
               (∃ O : ℂ, O = -complex.I ∧ (a + b + c + d = -4 * complex.I)) ∧
               (∃ p q : ℝ, (abs (a + complex.I) = p ∧ abs (b + complex.I) = q ∧ 2 * p * 2 * q = 2 * p * q * 2)) ∧ 
               ((p * q * 2) = (6 * complex.sqrt 2)) := 
sorry

end rhombus_area_l633_633447


namespace find_a_plus_b_l633_633378

def star (a b : ℕ) : ℕ := a^b + a + b

theorem find_a_plus_b (a b : ℕ) (h2a : 2 ≤ a) (h2b : 2 ≤ b) (h_ab : star a b = 20) :
  a + b = 6 :=
sorry

end find_a_plus_b_l633_633378


namespace correct_reflection_l633_633908

section
variable {Point : Type}
variables (PQ : Point → Point → Prop) (shaded_figure : Point → Prop)
variables (A B C D E : Point → Prop)

-- Condition: The line segment PQ is the axis of reflection.
-- Condition: The shaded figure is positioned above the line PQ and touches it at two points.
-- Define the reflection operation (assuming definitions for points and reflections are given).

def reflected (fig : Point → Prop) (axis : Point → Point → Prop) : Point → Prop := sorry  -- Define properly

-- The correct answer: The reflected figure should match figure (A).
theorem correct_reflection :
  reflected shaded_figure PQ = A :=
sorry
end

end correct_reflection_l633_633908


namespace pirate_treasure_probability_l633_633112

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem pirate_treasure_probability :
  let p_treasure := 1 / 5
  let p_traps := 1 / 10
  let p_neither := 7 / 10
  let num_islands := 8
  let num_treasure := 4
  binomial num_islands num_treasure * p_treasure^num_treasure * p_neither^(num_islands - num_treasure) = 673 / 25000 :=
by
  sorry

end pirate_treasure_probability_l633_633112


namespace complex_number_problem_l633_633222

theorem complex_number_problem (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i :=
by {
  -- provide proof here
  sorry
}

end complex_number_problem_l633_633222


namespace coordinates_C_l633_633117

theorem coordinates_C 
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) 
  (hA : A = (-1, 3)) 
  (hB : B = (11, 7))
  (hBC_AB : (C.1 - B.1, C.2 - B.2) = (2 / 3) • (B.1 - A.1, B.2 - A.2)) :
  C = (19, 29 / 3) :=
sorry

end coordinates_C_l633_633117


namespace locus_equation_perimeter_greater_than_3sqrt3_l633_633331

-- The locus W and the conditions 
def locus_eq (P : ℝ × ℝ) : Prop :=
  P.snd = P.fst ^ 2 + 1/4

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

-- Prove part (1): The locus W is y = x^2 + 1/4
theorem locus_equation (x y : ℝ) : 
  point_on_x_axis (x, y) → locus_eq (x, y) :=
sorry

-- Prove part (2): Perimeter of rectangle ABCD is greater than 3sqrt(3) if three vertices are on W
def rectangle_on_w (A B C : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  locus_eq A ∧ locus_eq B ∧ locus_eq C ∧ locus_eq D ∧ 
  (∃x₁ x₂ x₃ x₄ : ℝ, A = (x₁, x₁ ^ 2 + 1/4) ∧ B = (x₂, x₂ ^ 2 + 1/4) ∧ 
  C = (x₃, x₃ ^ 2 + 1/4) ∧ D = (x₄, x₄ ^ 2 + 1/4))

theorem perimeter_greater_than_3sqrt3 (A B C D : ℝ × ℝ) : 
  rectangle_on_w A B C D → 
  2 * (abs (B.fst - A.fst) + abs (C.fst - B.fst)) > 3 * sqrt 3 :=
sorry

end locus_equation_perimeter_greater_than_3sqrt3_l633_633331


namespace turnip_bag_weight_l633_633064

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l633_633064


namespace time_boarding_in_London_l633_633167

open Nat

def time_in_ET_to_London_time (time_et: ℕ) : ℕ :=
  (time_et + 5) % 24

def subtract_hours (time: ℕ) (hours: ℕ) : ℕ :=
  (time + 24 * (hours / 24) - (hours % 24)) % 24

theorem time_boarding_in_London :
  let cape_town_arrival_time_et := 10
  let flight_duration_ny_to_cape := 10
  let ny_departure_time := subtract_hours cape_town_arrival_time_et flight_duration_ny_to_cape
  let flight_duration_london_to_ny := 18
  let ny_arrival_time := subtract_hours ny_departure_time flight_duration_london_to_ny
  let london_time := time_in_ET_to_London_time ny_arrival_time
  let london_departure_time := subtract_hours london_time flight_duration_london_to_ny
  london_departure_time = 17 :=
by
  -- Proof omitted
  sorry

end time_boarding_in_London_l633_633167


namespace neg_exists_to_forall_l633_633768

theorem neg_exists_to_forall (n : ℝ) :
  (¬ ∃ a : ℝ, a ≥ -1 ∧ ln (exp n + 1) > 1 / 2) ↔ (∀ a : ℝ, a ≥ -1 → ln (exp n + 1) ≤ 1 / 2) :=
by
  sorry

end neg_exists_to_forall_l633_633768


namespace domain_of_f_l633_633443

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 2) / Real.log 2

theorem domain_of_f : { x : ℝ | x > 2 } = { x : ℝ | x ∈ (2, +∞) } :=
by
  sorry

end domain_of_f_l633_633443


namespace rectangle_width_decrease_l633_633449

theorem rectangle_width_decrease (L W : ℝ) (h1 : 0 < L) (h2 : 0 < W) 
(h3 : ∀ W' : ℝ, 0 < W' → (1.3 * L * W' = L * W) → W' = (100 - 23.077) / 100 * W) : 
  ∃ W' : ℝ, 0 < W' ∧ (1.3 * L * W' = L * W) ∧ ((W - W') / W = 23.077 / 100) :=
by
  sorry

end rectangle_width_decrease_l633_633449


namespace not_snow_probability_l633_633865

theorem not_snow_probability :
  let p_snow : ℚ := 2 / 3,
      p_not_snow : ℚ := 1 - p_snow in
  (p_not_snow ^ 5) = 1 / 243 := by
  sorry

end not_snow_probability_l633_633865


namespace part1_part2_l633_633393

open Set

variable (a : ℝ)

def I := univ real
def M := { x : ℝ | (x + 3)^2 ≤ 0 }
def N := { x : ℝ | x^2 + x - 6 = 0 }
def A := compl M ∩ N
def B := { x : ℝ | a - 1 ≤ x ∧ x ≤ 5 - a }

theorem part1 : compl M ∩ N = {2} := sorry

theorem part2 : (B ∪ A = A) → a = 3 := sorry

end part1_part2_l633_633393


namespace probability_no_snow_l633_633840

theorem probability_no_snow {
  let p_snow : ℚ := 2/3,     -- Probability of snow on any one day
  let p_no_snow : ℚ := 1 - p_snow,  -- Probability of no snow on any one day
  let p_no_snow_5_days : ℚ := p_no_snow ^ 5  -- Probability of no snow on any of the five days
} : p_no_snow_5_days = 1 / 243 := by
  sorry

end probability_no_snow_l633_633840


namespace angle_C_in_triangle_l633_633353

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 90) (h2 : A + B + C = 180) : C = 90 :=
sorry

end angle_C_in_triangle_l633_633353


namespace find_sin_C_find_area_of_triangle_l633_633716

noncomputable theory

variables {A B C : ℝ}
variables {BC : ℝ}
variables {sinC : ℝ}
variables {area : ℝ}

-- Conditions:
def cos_A : ℝ := -5/13
def cos_B : ℝ := 3/5
def BC_value : ℝ := 5

-- Question 1: Value of sin C
def sin_C_value : Prop := sinC = 16/65

-- Question 2: Area of triangle ABC
def area_of_triangle : Prop := area = 8/3

-- Proof Goals:
theorem find_sin_C (h1 : cos_A = -5/13) (h2 : cos_B = 3/5) : sinC = 16/65 := sorry

theorem find_area_of_triangle (h1 : cos_A = -5/13) (h2 : cos_B = 3/5) (h3 : BC = 5) : area = 8/3 := sorry

end find_sin_C_find_area_of_triangle_l633_633716


namespace profit_percentage_A_is_20_l633_633546

-- Definitions of conditions
def cost_price_A := 156 -- Cost price of the cricket bat for A
def selling_price_C := 234 -- Selling price of the cricket bat to C
def profit_percent_B := 25 / 100 -- Profit percentage for B

-- Calculations
def cost_price_B := selling_price_C / (1 + profit_percent_B) -- Cost price of the cricket bat for B
def selling_price_A := cost_price_B -- Selling price of the cricket bat for A

-- Profit and profit percentage calculations
def profit_A := selling_price_A - cost_price_A -- Profit for A
def profit_percent_A := profit_A / cost_price_A * 100 -- Profit percentage for A

-- Statement to prove
theorem profit_percentage_A_is_20 : profit_percent_A = 20 :=
by
  sorry

end profit_percentage_A_is_20_l633_633546


namespace derek_february_savings_l633_633178

theorem derek_february_savings :
  ∀ (savings : ℕ → ℕ),
  (savings 1 = 2) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 12 → savings (n + 1) = 2 * savings n) ∧
  (savings 12 = 4096) →
  savings 2 = 4 :=
by
  sorry

end derek_february_savings_l633_633178


namespace candidate_a_votes_l633_633608

theorem candidate_a_votes (x : ℕ) (h : 2 * x + x = 21) : 2 * x = 14 :=
by sorry

end candidate_a_votes_l633_633608


namespace problem1_problem2_l633_633922

-- Problem (1): Prove that 2 * sqrt 3 - |sqrt 2 - sqrt 3| = sqrt 3 + sqrt 2
theorem problem1 : 2 * Real.sqrt 3 - Real.abs (Real.sqrt 2 - Real.sqrt 3) = Real.sqrt 3 + Real.sqrt 2 := 
  sorry

-- Problem (2): Prove that sqrt (16 / 9) + cbrt (-8) + sqrt ((-2 / 3) ^ 2) = 0
theorem problem2 : Real.sqrt (16 / 9) + Real.cbrt (-8) + Real.sqrt ((-2 / 3) ^ 2) = 0 :=
  sorry

end problem1_problem2_l633_633922


namespace a_7_is_4_l633_633209

-- Define the geometric sequence and its properties
variable {a : ℕ → ℝ}

-- Given conditions
axiom pos_seq : ∀ n, a n > 0
axiom geom_seq : ∀ n m, a (n + m) = a n * a m
axiom specific_condition : a 3 * a 11 = 16

theorem a_7_is_4 : a 7 = 4 :=
by
  sorry

end a_7_is_4_l633_633209


namespace sum_sqrt_ineq_l633_633389

theorem sum_sqrt_ineq (n : ℕ) (a : ℕ → ℝ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) :
  ∑ i in finset.range (n + 1), (real.sqrt ((a i)^(n-1) / ((a i)^(n-1) + (n^2 - 1) * ∏ j in (finset.range (n+1)).filter (λ k, k ≠ i), a j))) ≥ 1 :=
sorry

end sum_sqrt_ineq_l633_633389


namespace number_and_its_square_root_l633_633903

theorem number_and_its_square_root (x : ℝ) (h : x + 10 * Real.sqrt x = 39) : x = 9 :=
sorry

end number_and_its_square_root_l633_633903


namespace deductive_reasoning_example_l633_633498

/-- 
 Given the problem of identifying which reasoning belongs to deductive reasoning among 
 the provided options, Option D (Gold, silver, and copper conduct electricity 
 because they are metals and all metals conduct electricity) is the correct one.
-/
theorem deductive_reasoning_example :
  let A := "Inferring the properties of a sphere from the properties of a circle"
  let B := "Generalizing that the sum of the interior angles of all triangles is \(180^{\circ}\) based on equilateral and right triangles"
  let C := "Inferring that Xiao Ming scored full marks in all other subjects based on his full marks in a math test"
  let D := "Gold, silver, and copper conduct electricity because they are metals and all metals conduct electricity"
  (D = "Deductive Reasoning") :=
sorry

end deductive_reasoning_example_l633_633498


namespace arrange_abc_l633_633760

noncomputable def a : ℝ := Real.log 0.8 / Real.log 0.6
noncomputable def b : ℝ := Real.log 0.9 / Real.log 1.1
noncomputable def c : ℝ := 1.1 ^ 0.8

theorem arrange_abc : b < a ∧ a < c := 
by 
  -- Definitions based on conditions
  let a := Real.log 0.8 / Real.log 0.6
  let b := Real.log 0.9 / Real.log 1.1
  let c := 1.1 ^ 0.8
  
  -- Proof of ordering
  have dec_log_0_6 := by sorry  -- Proof that log base 0.6 is decreasing
  have inc_log_1_1 := by sorry  -- Proof that log base 1.1 is increasing
  have inc_1_1_pow := by sorry  -- Proof that 1.1^x is increasing
  sorry -- To be filled with the proof

end arrange_abc_l633_633760


namespace stool_height_l633_633149

theorem stool_height (bulb_below_ceiling : ℕ) (ceiling_height : ℕ) 
  (alice_height : ℕ) (hat_height : ℕ) (reach_above_head : ℕ) 
  (floor_dip : ℕ) (effective_reach : ℕ) : 
  bulb_below_ceiling = 15 → 
  ceiling_height = 280 → 
  alice_height = 160 → 
  hat_height = 5 → 
  reach_above_head = 50 → 
  floor_dip = 3 → 
  effective_reach = 212 → 
  ∃ stool_height : ℕ, stool_height = (ceiling_height - bulb_below_ceiling) - effective_reach :=
begin
  -- Proof begins here
  sorry
end

end stool_height_l633_633149


namespace percentage_loss_is_correct_l633_633044

-- Define the cost price and selling price
def cost_price : ℕ := 2000
def selling_price : ℕ := 1800

-- Define the calculation of loss and percentage loss
def loss (cp sp : ℕ) := cp - sp
def percentage_loss (loss cp : ℕ) := (loss * 100) / cp

-- The goal is to prove that the percentage loss is 10%
theorem percentage_loss_is_correct : percentage_loss (loss cost_price selling_price) cost_price = 10 := by
  sorry

end percentage_loss_is_correct_l633_633044


namespace numbers_contain_digit_five_l633_633286

theorem numbers_contain_digit_five :
  (finset.range 701).filter (λ n, ∃ d ∈ (n.digits 10), d = 5).card = 214 := 
sorry

end numbers_contain_digit_five_l633_633286


namespace angle_is_pi_over_3_l633_633690

variables (a b : ℝ^3)

def norm (v : ℝ^3) : ℝ := real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

def angle_between_vectors (v1 v2 : ℝ^3) : ℝ :=
real.acos ((v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / (norm v1 * norm v2))

theorem angle_is_pi_over_3 (h1 : norm a = 3) 
                          (h2 : norm b = 2) 
                          (h3 : norm (2 * a + b) = 2 * real.sqrt 13) :
  angle_between_vectors a b = real.pi / 3 :=
sorry

end angle_is_pi_over_3_l633_633690


namespace max_product_with_digits_l633_633892

theorem max_product_with_digits :
  ∃ a b c d e : ℕ,
    {a, b, c, d, e} = {1, 3, 5, 8, 9} ∧
    (c % 2 = 0) ∧
    100 * a + 10 * b + c = 951 ∧
    let p := (100 * a + 10 * b + c) * (10 * d + e) in 
    ∀ a' b' c' d' e' : ℕ, 
      {a', b', c', d', e'} = {1, 3, 5, 8, 9} ∧
      (c' % 2 = 0) → 
      let p' := (100 * a' + 10 * b' + c') * (10 * d' + e') in 
      p' ≤ p :=
sorry

end max_product_with_digits_l633_633892


namespace sum_of_solutions_l633_633302

theorem sum_of_solutions (y : ℝ) (h : y^2 = 25) : ∃ (a b : ℝ), (a = 5 ∨ a = -5) ∧ (b = 5 ∨ b = -5) ∧ a + b = 0 :=
sorry

end sum_of_solutions_l633_633302


namespace number_of_restaurants_l633_633031

def units_in_building : ℕ := 300
def residential_units := units_in_building / 2
def remaining_units := units_in_building - residential_units
def restaurants := remaining_units / 2

theorem number_of_restaurants : restaurants = 75 :=
by
  sorry

end number_of_restaurants_l633_633031


namespace not_snowing_next_five_days_l633_633859

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l633_633859


namespace tile_5x7_rectangle_with_L_trominos_l633_633361

theorem tile_5x7_rectangle_with_L_trominos :
  ∀ k : ℕ, ¬ (∃ (tile : ℕ → ℕ → ℕ), (∀ i j, tile (i+1) (j+1) = tile (i+3) (j+3)) ∧
    ∀ i j, (i < 5 ∧ j < 7) → (tile i j = k)) :=
by sorry

end tile_5x7_rectangle_with_L_trominos_l633_633361


namespace exercise_l633_633652

variable (α β : ℝ)
variable (a : ℝ)

def prop_p : Prop := ∃ α β : ℝ, sin (α + β) = sin α + sin β

def prop_q : Prop := ∀ (a : ℝ), a > 2 ∧ a ≠ 1 → log a 2 + log 2 a ≥ 2

theorem exercise :
  (∃ α β : ℝ, sin (α + β) = sin α + sin β) ∨ (∀ (a : ℝ), a > 2 ∧ a ≠ 1 → log a 2 + log 2 a ≥ 2) :=
by 
  sorry

end exercise_l633_633652


namespace soda_cost_l633_633961

variable {b s f : ℕ}

theorem soda_cost :
    5 * b + 3 * s + 2 * f = 520 ∧
    3 * b + 2 * s + f = 340 →
    s = 80 :=
by
  sorry

end soda_cost_l633_633961


namespace area_of_hexagon_l633_633754

noncomputable def HexagonArea (J K L : Point) (areaJKL : ℝ) [metric_space Point] := 
  9 * areaJKL / 3

theorem area_of_hexagon (J K L : Point) (midpointJKL : ∀ {A B C D E F : Point}, is_midpoint A B J ∧ is_midpoint C D K ∧ is_midpoint E F L) (areaJKL100 : HexagonArea J K L 100) :
  True := sorry

end area_of_hexagon_l633_633754


namespace prime_square_mod_180_l633_633567

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : 5 < p) :
  ∃ (s : Finset ℕ), s.card = 2 ∧ ∀ r ∈ s, ∃ n : ℕ, p^2 % 180 = r :=
sorry

end prime_square_mod_180_l633_633567


namespace x_squared_convex_sqrt_x_concave_x_cubed_convex_concave_reciprocal_x_l633_633483

variables {x : ℝ} {q1 q2 x1 x2 : ℝ}

-- Condition that q1 + q2 = 1
axiom h : q1 + q2 = 1

-- Defining convexity and concavity conditions
def convex (f : ℝ → ℝ) := ∀ x1 x2 q1 q2, h → f(q1 * x1 + q2 * x2) ≤ q1 * f(x1) + q2 * f(x2)
def concave (f : ℝ → ℝ) := ∀ x1 x2 q1 q2, h → f(q1 * x1 + q2 * x2) ≥ q1 * f(x1) + q2 * f(x2)

-- Main conjectures for the four functions
theorem x_squared_convex : convex (λ x, x^2) := sorry
theorem sqrt_x_concave : concave (λ x, real.sqrt x) := sorry
theorem x_cubed_convex_concave : 
  (∀ x > 0, convex (λ x, x^3)) ∧
  (∀ x < 0, concave (λ x, x^3)) := sorry
theorem reciprocal_x :
  (∀ x > 0, convex (λ x, 1 / x)) ∧
  (∀ x < 0, concave (λ x, 1 / x)) := sorry

end x_squared_convex_sqrt_x_concave_x_cubed_convex_concave_reciprocal_x_l633_633483


namespace MitchWorks25Hours_l633_633402

noncomputable def MitchWorksHours : Prop :=
  let weekday_earnings_rate := 3
  let weekend_earnings_rate := 6
  let weekly_earnings := 111
  let weekend_hours := 6
  let weekday_hours (x : ℕ) := 5 * x
  let weekend_earnings := weekend_hours * weekend_earnings_rate
  let weekday_earnings (x : ℕ) := x * weekday_earnings_rate
  let total_weekday_earnings (x : ℕ) := weekly_earnings - weekend_earnings
  ∀ (x : ℕ), weekday_earnings x = total_weekday_earnings x → x = 25

theorem MitchWorks25Hours : MitchWorksHours := by
  sorry

end MitchWorks25Hours_l633_633402


namespace M_minimizes_BC_projection_l633_633372

noncomputable def minimal_BC_projection (A B C : Point) : Point := 
  footOfPerpendicular A B C

theorem M_minimizes_BC_projection (A B C M B_prime C_prime : Point) 
  (hM_on_BC : M ∈ lineSegment B C)
  (hB_prime_projection : B_prime = orthogonalProjection M AC)
  (hC_prime_projection : C_prime = orthogonalProjection M AB) :
  M = footOfPerpendicular A B C ↔
  minimal_length B_prime C_prime := 
sorry

end M_minimizes_BC_projection_l633_633372


namespace polynomial_evaluation_l633_633187

theorem polynomial_evaluation :
  ∀ x : ℤ, x = -2 → (x^3 + x^2 + x + 1 = -5) :=
by
  intros x hx
  rw [hx]
  norm_num

end polynomial_evaluation_l633_633187


namespace arrangement_count_l633_633531

theorem arrangement_count (n m : ℕ) (h : n = 7 ∧ m = 3) :
  (n + m).choose m = 120 :=
by
  rw [h.1, h.2]
  unfold nat.choose
  sorry

end arrangement_count_l633_633531


namespace simplify_expr_l633_633800

theorem simplify_expr (a : ℝ) (h : a > 1) : (1 - a) * (1 / (a - 1)).sqrt = -(a - 1).sqrt :=
sorry

end simplify_expr_l633_633800


namespace additional_pots_last_vs_first_l633_633103

-- Conditions
def MachineA_first_hour_rate := 6  -- minutes per pot
def MachineA_eighth_hour_rate := 5.2  -- minutes per pot
def MachineB_first_hour_rate := 5.5  -- minutes per pot
def MachineB_eighth_hour_rate := 5.1  -- minutes per pot

-- Calculate pots produced in the first hour
def MachineA_first_hour_pots := 60 / MachineA_first_hour_rate
def MachineB_first_hour_pots := 60 / MachineB_first_hour_rate

-- Calculate pots produced in the eighth hour
def MachineA_eighth_hour_pots := 60 / MachineA_eighth_hour_rate
def MachineB_eighth_hour_pots := 60 / MachineB_eighth_hour_rate

-- Calculate additional pots in the eighth hour compared to the first hour for each machine
def MachineA_additional_pots := MachineA_eighth_hour_pots - MachineA_first_hour_pots
def MachineB_additional_pots := MachineB_eighth_hour_pots - MachineB_first_hour_pots

-- Total additional pots produced in the last hour compared to the first
def total_additional_pots := MachineA_additional_pots + MachineB_additional_pots

-- Proof statement
theorem additional_pots_last_vs_first : total_additional_pots = 2 :=
by
  sorry

end additional_pots_last_vs_first_l633_633103


namespace dice_tower_even_n_l633_633795

/-- Given that n standard dice are stacked in a vertical tower,
and the total visible dots on each of the four vertical walls are all odd,
prove that n must be even.
-/
theorem dice_tower_even_n (n : ℕ)
  (h : ∀ (S T : ℕ), (S + T = 7 * n → (S % 2 = 1 ∧ T % 2 = 1))) : n % 2 = 0 :=
by sorry

end dice_tower_even_n_l633_633795


namespace numbers_not_expressible_as_difference_of_squares_l633_633346

theorem numbers_not_expressible_as_difference_of_squares :
  ∃ (s : Finset ℤ), s = {2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010} ∧
  (s.filter (λ x, ∀ n m : ℤ, n^2 - m^2 ≠ x)).card = 3 :=
by
  sorry

end numbers_not_expressible_as_difference_of_squares_l633_633346


namespace total_possible_ranking_sequences_l633_633723

def team : Type := Fin 6  -- We assume teams are indexed from 0 to 5, where 0 is A, 1 is B, ..., 5 is F.

def tournament_ranking_possible_sequences (teams : Finset team) : Nat :=
  if teams.card = 5 then 32 else 0

theorem total_possible_ranking_sequences :
  tournament_ranking_possible_sequences {0, 1, 2, 3, 4} = 32 :=
by
  sorry

end total_possible_ranking_sequences_l633_633723


namespace part_I_part_I_max_value_part_I_min_value_part_II_increasing_intervals_l633_633674

noncomputable def f (x : ℝ) := 2 * sin x * cos x + 2 * sqrt 3 * cos x ^ 2 - sqrt 3

-- The smallest positive period is π
theorem part_I:
  (∀ x : ℝ, f (x + π) = f x) := sorry

-- The maximum value over all x
theorem part_I_max_value:
  (∃ x : ℝ, ∀ y : ℝ, f x ≥ f y) ∧ (∃ x : ℝ, f x = 2) := sorry

-- The minimum value over all x
theorem part_I_min_value:
  (∃ x : ℝ, ∀ y : ℝ, f x ≤ f y) ∧ (∃ x : ℝ, f x = -2) := sorry

-- Intervals where f(x) is increasing
theorem part_II_increasing_intervals (k : ℤ) :
  ∀ x : ℝ, (k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 6) → (∀ y : ℝ, y ∈ Icc (k * π - π / 6) (k * π + π / 6) → f y' > f y) := sorry

end part_I_part_I_max_value_part_I_min_value_part_II_increasing_intervals_l633_633674


namespace product_of_sines_KLM_l633_633355

open Real

variables (K L M : Type) [NormedAddCommGroup K] [NormedAddCommGroup L] [NormedAddCommGroup M]
          (Klm : Triangle K L M)
          (KP P : Set K)
          (O Q R : Set K)
          [IsMedian KP Klm]
          [Circumcenter O Klm]
          [Incenter Q Klm]
          [Intersects R KP OQ Klm]
          (a b : Type) [NormedAddCommGroup a] [NormedAddCommGroup b]
          [Angle LKM = pi/3]

noncomputable def product_of_sines (α β : ℝ) (h1 : α + β = 2 * pi / 3) (h2 : sin α * sin β = 5 / 8) : Prop :=
  sin α * sin β = 5 / 8

theorem product_of_sines_KLM {K L M : Type} [NormedAddCommGroup K] [NormedAddCommGroup L] [NormedAddCommGroup M]
          (Klm : Triangle K L M)
          (KP P : Set K)
          (O Q R : Set K)
          [IsMedian KP Klm]
          [Circumcenter O Klm]
          [Incenter Q Klm]
          [Intersects R KP OQ Klm]
          (h : ∀ O R P Q : ℝ, OR / PR = √14 * (QR / KR))
          [Angle LKM = pi/3]
          {α β : ℝ} (h1 : α + β = 2 * pi / 3) : sin α * sin β = 5 / 8 := 
sorry

end product_of_sines_KLM_l633_633355


namespace locus_equation_rectangle_perimeter_greater_l633_633334

open Real

theorem locus_equation (P : ℝ × ℝ) : 
  (abs P.2 = sqrt (P.1 ^ 2 + (P.2 - 1 / 2) ^ 2)) → (P.2 = P.1 ^ 2 + 1 / 4) :=
by
  intro h
  sorry

theorem rectangle_perimeter_greater (A B C D : ℝ × ℝ) :
  (A.2 = A.1 ^ 2 + 1 / 4) ∧ 
  (B.2 = B.1 ^ 2 + 1 / 4) ∧ 
  (C.2 = C.1 ^ 2 + 1 / 4) ∧ 
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) → 
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
by
  intro h
  sorry

end locus_equation_rectangle_perimeter_greater_l633_633334


namespace total_selling_price_l633_633045

theorem total_selling_price (CP : ℕ) (num_toys : ℕ) (gain_toys : ℕ) (TSP : ℕ)
  (h1 : CP = 1300)
  (h2 : num_toys = 18)
  (h3 : gain_toys = 3) :
  TSP = 27300 := by
  sorry

end total_selling_price_l633_633045


namespace find_two_digit_number_with_cubic_ending_in_9_l633_633730

theorem find_two_digit_number_with_cubic_ending_in_9:
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n^3 % 10 = 9 ∧ n = 19 := 
by
  sorry

end find_two_digit_number_with_cubic_ending_in_9_l633_633730


namespace perfect_number_divisible_by_nine_l633_633108

-- Define the sum of divisors function sigma(n)
def sigma (n : Nat) : Nat :=
  Nat.sumDivisors n

-- Define what it means to be a perfect number
def isPerfect (n : Nat) : Prop :=
  sigma n = 2 * n

-- Define the main statement to prove
theorem perfect_number_divisible_by_nine (n : Nat) (h1 : isPerfect n) (h2 : n > 6) (h3 : ∃ k, n = 3 * k) : ∃ m, n = 9 * m :=
by
  sorry

end perfect_number_divisible_by_nine_l633_633108


namespace M_geq_N_l633_633215

variable (a b : ℝ)

def M : ℝ := a^2 + 12 * a - 4 * b
def N : ℝ := 4 * a - 20 - b^2

theorem M_geq_N : M a b ≥ N a b := by
  sorry

end M_geq_N_l633_633215


namespace award_medals_at_most_one_canadian_l633_633877

/-- Definition of conditions -/
def sprinter_count := 10 -- Total number of sprinters
def canadian_sprinter_count := 4 -- Number of Canadian sprinters
def medals := ["Gold", "Silver", "Bronze"] -- Types of medals

/-- Definition stating the requirement of the problem -/
def atMostOneCanadianMedal (total_sprinters : Nat) (canadian_sprinters : Nat) 
    (medal_types : List String) : Bool := 
  if total_sprinters = sprinter_count ∧ canadian_sprinters = canadian_sprinter_count ∧ medal_types = medals then
    true
  else
    false

/-- Statement to prove the number of ways to award the medals -/
theorem award_medals_at_most_one_canadian :
  (atMostOneCanadianMedal sprinter_count canadian_sprinter_count medals) →
  ∃ (ways : Nat), ways = 480 :=
by
  sorry

end award_medals_at_most_one_canadian_l633_633877


namespace hannahs_son_cuts_three_strands_per_minute_l633_633269

variable (x : ℕ)

theorem hannahs_son_cuts_three_strands_per_minute
  (total_strands : ℕ)
  (hannah_rate : ℕ)
  (total_time : ℕ)
  (total_strands_cut : ℕ := hannah_rate * total_time)
  (son_rate := (total_strands - total_strands_cut) / total_time)
  (hannah_rate := 8)
  (total_time := 2)
  (total_strands := 22) :
  son_rate = 3 := 
by
  sorry

end hannahs_son_cuts_three_strands_per_minute_l633_633269


namespace max_value_sin_function_l633_633457

theorem max_value_sin_function : 
  ∀ x, (-(π)/2 ≤ x ∧ x ≤ 0) → (3 * sin x + 2 ≤ 2) :=
by
  assume x h,
  sorry

end max_value_sin_function_l633_633457


namespace not_snow_probability_l633_633864

theorem not_snow_probability :
  let p_snow : ℚ := 2 / 3,
      p_not_snow : ℚ := 1 - p_snow in
  (p_not_snow ^ 5) = 1 / 243 := by
  sorry

end not_snow_probability_l633_633864


namespace vector_field_is_solenoidal_l633_633419

noncomputable def vector_field_r (r θ : ℝ) : ℝ := (2 * Real.cos θ) / (r^3)
noncomputable def vector_field_θ (r θ : ℝ) : ℝ := (Real.sin θ) / (r^3)

theorem vector_field_is_solenoidal :
  ∀ r θ : ℝ, (r ≠ 0) →
  let a_r := vector_field_r r θ
  let a_θ := vector_field_θ r θ in
  ((1 / r^2) * (∂/∂r (r^2 * a_r)) + (1 / (r * Real.sin θ)) * (∂/∂θ (Real.sin θ * a_θ)) = 0) :=
sorry

end vector_field_is_solenoidal_l633_633419


namespace smallest_bob_number_l633_633148

theorem smallest_bob_number (a b : ℕ) (ha : a = 90)
    (h : ∀ p : ℕ, p.prime → p ∣ a → p ∣ b) : b = 30 :=
by
  have prime_factors_90 : ∀ p : ℕ, p.prime → p ∣ 90 → p = 2 ∨ p = 3 ∨ p = 5 := by
    -- This step would normally involve proving the prime factorization of 90, 
    -- but we will assume it to simplify the proof demonstration
    intros p hp hpa
    rw [Nat.prime_dvd_prime_iff_eq] at hpa
    cases hpa with 
    | inr h' => exact (h : ∀ p : ℕ, p.prime → p ∣ a → p ∣ b) 
    | inl h' => exact h' (by norm_num1)

  have factor_2 : 2 ∣ b := by
    apply h 2 (by norm_num)

  have factor_3 : 3 ∣ b := by
    apply h 3 (by norm_num)

  have factor_5 : 5 ∣ b := by
    apply h 5 (by norm_num)

  have min_b : b = 30 := by
    -- This step would prove that 30 is the smallest number satisfying all conditions 
    sorry -- skipping the detailed proof steps
  
  exact min_b

end smallest_bob_number_l633_633148


namespace number_of_solutions_l633_633297

theorem number_of_solutions :
  let count_solutions (f : ℝ → Prop) (a b : ℝ) : ℕ :=
        (finset.Icc a b).filter (λ x, f x).card in
    count_solutions (λ x, (x > -25) ∧ (x < 120) ∧ (cos x)^2 + 3*(sin x)^2 = 1) ⌊-25/π⌋ ⌈120/π⌉ = 46 :=
sorry

end number_of_solutions_l633_633297


namespace non_participating_students_l633_633409

theorem non_participating_students (total_students : ℕ) (fraction_participating : ℚ) 
  (h_total_students : total_students = 89) (h_fraction_participating : fraction_participating = 3/5) :
  total_students - (fraction_participating * total_students).natAbs = 35 := 
by
  sorry

end non_participating_students_l633_633409


namespace maximum_smallest_triplet_sum_l633_633777

theorem maximum_smallest_triplet_sum (circle : Fin 10 → ℕ) (h : ∀ i : Fin 10, 1 ≤ circle i ∧ circle i ≤ 10 ∧ ∀ j k, j ≠ k → circle j ≠ circle k):
  ∃ (i : Fin 10), ∀ j ∈ ({i, i + 1, i + 2} : Finset (Fin 10)), circle i + circle (i + 1) + circle (i + 2) ≤ 15 :=
sorry

end maximum_smallest_triplet_sum_l633_633777


namespace sum_of_nums_divisible_by_7_from_balls_l633_633480

theorem sum_of_nums_divisible_by_7_from_balls :
  let balls := [1, 2, 3, 5]
  let two_digit_numbers := [12, 13, 15, 21, 23, 25, 31, 32, 35, 51, 52, 53]
  let divisible_by_7 := two_digit_numbers.filter (λ n, n % 7 = 0)
  let result := divisible_by_7.sum
  in result = 56 :=
by
  let balls := [1, 2, 3, 5]
  let two_digit_numbers := balls.product balls
                    |>.filter (λ (a, b), a ≠ b)
                    |>.map (λ (a, b), 10 * a + b)
  let divisible_by_7 := two_digit_numbers.filter (λ n, n % 7 = 0)
  let result := divisible_by_7.sum
  have h1 : two_digit_numbers = [12, 13, 15, 21, 23, 25, 31, 32, 35, 51, 52, 53] := sorry
  have h2 : divisible_by_7 = [21, 35] := sorry
  have h3 : result = 56 := sorry
  exact h3

end sum_of_nums_divisible_by_7_from_balls_l633_633480


namespace more_crayons_than_erasers_l633_633783

theorem more_crayons_than_erasers
  (E : ℕ) (C : ℕ) (C_left : ℕ) (E_left : ℕ)
  (hE : E = 457) (hC : C = 617) (hC_left : C_left = 523) (hE_left : E_left = E) :
  C_left - E_left = 66 := 
by
  sorry

end more_crayons_than_erasers_l633_633783


namespace average_time_l633_633986

variable casey_time : ℝ
variable zendaya_multiplier : ℝ

theorem average_time (h1 : casey_time = 6) (h2 : zendaya_multiplier = 1/3) :
  (casey_time + (casey_time + zendaya_multiplier * casey_time)) / 2 = 7 := by
  sorry

end average_time_l633_633986


namespace cricket_target_runs_l633_633936

theorem cricket_target_runs (run_rate_first: ℝ) (overs_first: ℝ) 
                           (run_rate_remaining: ℝ) (overs_remaining: ℝ) :
    run_rate_first = 3.2 →
    overs_first = 10 →
    run_rate_remaining = 5.75 →
    overs_remaining = 40 →
    let total_runs_first := run_rate_first * overs_first in
    let total_runs_remaining := run_rate_remaining * overs_remaining in
    (total_runs_first + total_runs_remaining) = 262 :=
by
  intros h₁ h₂ h₃ h₄
  have h₅ : total_runs_first = 3.2 * 10 := by rw [h₁, h₂]
  have h₆ : total_runs_remaining = 5.75 * 40 := by rw [h₃, h₄]
  rw [h₅, h₆]
  norm_num
  sorry

end cricket_target_runs_l633_633936


namespace mode_three_l633_633436

def data_set : List ℕ := [3, 1, 3, 0, 3, 2, 1, 2]

def mode (l : List ℕ) : ℕ :=
  l.foldr (λ x freqMap, freqMap.insert x (freqMap.findD x 0 + 1))
          (RBMap.empty ℕ (· < ·)).toList
          |>.foldl (λ res pair, if pair.2 > res.2 then pair else res).1

theorem mode_three : mode data_set = 3 := by
  sorry

end mode_three_l633_633436


namespace solve_system_l633_633430

theorem solve_system :
  { (x : ℝ) × (y : ℝ) // x^2 + y^2 ≤ 2 ∧ 81 * x^4 - 18 * x^2 * y^2 + y^4 - 360 * x^2 - 40 * y^2 + 400 = 0 }
    = {((x, y) : ℝ × ℝ) |
         (x = -3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
         (x = -3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
         (x = 3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
         (x = 3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) } :=
by
  sorry

end solve_system_l633_633430


namespace derek_added_water_l633_633930

variable (original_amount final_amount water_added : ℝ)

def initial_conditions :=
  original_amount = 3 ∧ final_amount = 9.8

theorem derek_added_water (h : initial_conditions original_amount final_amount) :
  water_added = final_amount - original_amount → water_added = 6.8 :=
by
  intro h_add
  rw [h_add, h.1, h.2]
  norm_num
  sorry

end derek_added_water_l633_633930


namespace fraction_to_decimal_l633_633598

theorem fraction_to_decimal : (3 : ℝ) / 50 = 0.06 := by
  sorry

end fraction_to_decimal_l633_633598


namespace min_remove_prod_free_l633_633792

def remaining_set_no_prod (s : set ℕ) : Prop :=
  ∀ a b c ∈ s, a * b ≠ c

noncomputable def min_elements_to_remove : ℕ := 43

theorem min_remove_prod_free :
  ∃ (s : set ℕ), (s ⊆ { i | 1 ≤ i ∧ i ≤ 1982 }) ∧ 
                 remaining_set_no_prod s ∧ 
                 (1982 - s.card = min_elements_to_remove) :=
sorry

end min_remove_prod_free_l633_633792


namespace min_value_trig_expr_l633_633625

theorem min_value_trig_expr (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 4) :
    3 * Real.cos θ + 1 / Real.sin θ + (Real.sqrt 3) * Real.tan θ + 2 * Real.sin θ ≥ 4 * Real.root 4 (3 * Real.sqrt 3) := by
  sorry

end min_value_trig_expr_l633_633625


namespace y_value_l633_633537

def reciprocal (y : ℝ) : ℝ := 1 / y
def additive_inverse (y : ℝ) : ℝ := -y

theorem y_value : ∃ (y : ℝ), y = 2 * (reciprocal y) * (additive_inverse y) - 4 ∧ y = -6 := by
  sorry

end y_value_l633_633537


namespace solve_system_of_equations_l633_633683

theorem solve_system_of_equations (x y m : ℝ) 
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = m - 3) 
  (h3 : x - y = 2) : m = 8 :=
by
  -- Proof part is replaced with sorry as mentioned
  sorry

end solve_system_of_equations_l633_633683


namespace cricket_average_increase_l633_633534

theorem cricket_average_increase (runs_mean : ℕ) (innings : ℕ) (runs : ℕ) (new_runs : ℕ) (x : ℕ) :
  runs_mean = 35 → innings = 10 → runs = 79 → (total_runs : ℕ) = runs_mean * innings → 
  (new_total : ℕ) = total_runs + runs → (new_mean : ℕ) = new_total / (innings + 1) ∧ new_mean = runs_mean + x → x = 4 :=
by
  sorry

end cricket_average_increase_l633_633534


namespace binary_conversion_subtraction_l633_633995

def binary_to_decimal (b : list ℕ) : ℕ :=
  b.foldl (λ acc bit, acc * 2 + bit) 0

theorem binary_conversion_subtraction (b : list ℕ)
  (h : b = [1, 1, 1, 0, 0, 1]) :
  binary_to_decimal b - 3 = 54 :=
by
  rw [h]
  simp [binary_to_decimal]
  -- detailed step-by-step proof here
  sorry

end binary_conversion_subtraction_l633_633995


namespace ab_product_is_two_l633_633974

theorem ab_product_is_two (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (h_period : ∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π / (2 * b))))
  (h_point1 : a * Real.tan (b * (π / 8)) = 1)
  (h_point2 : a * Real.tan (b * (3 * π / 8)) = -1) :
  a * b = 2 :=
by 
sor$route


end ab_product_is_two_l633_633974


namespace turnip_weight_possible_l633_633083

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l633_633083


namespace f_half_l633_633219

theorem f_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - 2 * x) = 1 / (x ^ 2)) :
  f (1 / 2) = 16 :=
sorry

end f_half_l633_633219


namespace max_value_of_f_on_interval_exists_x_eq_min_1_l633_633831

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 6*x + 13)

theorem max_value_of_f_on_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → f x ≤ 1 / 4 := sorry

theorem exists_x_eq_min_1 : 
  ∃ x, -2 ≤ x ∧ x ≤ 2 ∧ f x = 1 / 4 := sorry

end max_value_of_f_on_interval_exists_x_eq_min_1_l633_633831


namespace number_of_restaurants_l633_633029

theorem number_of_restaurants
  (total_units : ℕ)
  (residential_units : ℕ)
  (non_residential_units : ℕ)
  (restaurants : ℕ)
  (h1 : total_units = 300)
  (h2 : residential_units = total_units / 2)
  (h3 : non_residential_units = total_units - residential_units)
  (h4 : restaurants = non_residential_units / 2)
  : restaurants = 75 := 
by
  sorry

end number_of_restaurants_l633_633029


namespace total_price_correct_l633_633040

-- Definitions of given conditions
def original_price : Float := 120
def discount_rate : Float := 0.30
def tax_rate : Float := 0.08

-- Definition of the final selling price
def sale_price : Float := original_price * (1 - discount_rate)
def total_selling_price : Float := sale_price * (1 + tax_rate)

-- Lean 4 statement to prove the total selling price is 90.72
theorem total_price_correct : total_selling_price = 90.72 := by
  sorry

end total_price_correct_l633_633040


namespace intersection_A_B_l633_633654

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 4 < x ∧ x < 7} :=
by
  sorry

end intersection_A_B_l633_633654


namespace radius_of_base_circle_of_cone_l633_633036

theorem radius_of_base_circle_of_cone (θ : ℝ) (r_sector : ℝ) (L : ℝ) (C : ℝ) (r_base : ℝ) :
  θ = 120 ∧ r_sector = 6 ∧ L = (θ / 360) * 2 * Real.pi * r_sector ∧ C = L ∧ C = 2 * Real.pi * r_base → r_base = 2 := by
  sorry

end radius_of_base_circle_of_cone_l633_633036


namespace probability_no_snow_l633_633842

theorem probability_no_snow {
  let p_snow : ℚ := 2/3,     -- Probability of snow on any one day
  let p_no_snow : ℚ := 1 - p_snow,  -- Probability of no snow on any one day
  let p_no_snow_5_days : ℚ := p_no_snow ^ 5  -- Probability of no snow on any of the five days
} : p_no_snow_5_days = 1 / 243 := by
  sorry

end probability_no_snow_l633_633842


namespace collinear_circumcenter_l633_633345

theorem collinear_circumcenter (O A B C P D E F : Point) (circumcenter : Triangle → Point)
  (Gamma1 Gamma2 : Circle) :
  (O = circumcenter (triangle A B C)) →
  (on_circle A Gamma1) →
  (on_circle B Gamma1) →
  (tangent_to AC Gamma1) →
  (on_circle A Gamma2) →
  (on_circle C Gamma2) →
  (tangent_to AB Gamma2) →
  (intersect_at Gamma1 Gamma2 A P) →
  (D_on BC (line_segment B C D)) →
  (E_on CA (line_segment C A E)) →
  (F_on AB (line_segment A B F)) →
  parallel (line DE) (line AB) →
  parallel (line DF) (line AC) →
  (collinear O P E) ↔ (perpendicular (line DE) (line EF)) :=
begin
  sorry
end

end collinear_circumcenter_l633_633345


namespace maximum_score_after_n_minutes_l633_633904

-- Define the initial polynomial
def initial_poly : Polynomial ℝ := Polynomial.C 1

-- Define the two actions
def add_monomial (P : Polynomial ℝ) (n : ℕ) : Polynomial ℝ := P + Polynomial.monomial n 1

def replace_by_shift (P : Polynomial ℝ) : Polynomial ℝ := Polynomial.eval (x + 1) P

-- Define the function for sum of coefficients
def sum_of_coefficients (P : Polynomial ℝ) : ℝ := Polynomial.eval 1 P

-- State the theorem
theorem maximum_score_after_n_minutes (n : ℕ) (h : n = 9) : 
  ∃ P : Polynomial ℝ, sum_of_coefficients P = 64 := 
sorry

end maximum_score_after_n_minutes_l633_633904


namespace football_joins_l633_633819

theorem football_joins (num_pentagons : ℕ) (num_hexagons : ℕ)
    (edges_per_pentagon : ℕ) (edges_per_hexagon : ℕ)
    (total_edges := (num_pentagons * edges_per_pentagon) + (num_hexagons * edges_per_hexagon))
    (joins := total_edges / 2) :
    num_pentagons = 12 → num_hexagons = 20 → 
    edges_per_pentagon = 5 → edges_per_hexagon = 6 → joins = 90 :=
by
    intros h_p h_h h_pe h_he
    rw [h_p, h_h, h_pe, h_he]
    simp [total_edges]
    norm_num
    sorry

end football_joins_l633_633819


namespace g_is_even_l633_633665

variable {f : ℝ → ℝ}

def g (x : ℝ) : ℝ := f x + f (-x)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x :=
by
  intro x
  unfold g
  calc
    f (-x) + f x = g x : by rw [add_comm]
    sorry

end g_is_even_l633_633665


namespace linear_independence_implies_zero_sum_l633_633656

variables {α : Type*} [add_comm_group α] [module ℝ α]
variables (a b c : α) (x y z : ℝ)

theorem linear_independence_implies_zero_sum
  (h_basis : linear_independent ℝ ![a, b, c])
  (h_eq : x • a + y • b + z • c = 0) : x^2 + y^2 + z^2 = 0 := 
by {
  -- We should prove that the only solution is x = 0, y = 0, z = 0
  -- And hence x^2 + y^2 + z^2 = 0 should be 0^2 + 0^2 + 0^2 = 0
  -- however, that proof is skipped for now.
  sorry
}

end linear_independence_implies_zero_sum_l633_633656


namespace charity_tickets_l633_633524

theorem charity_tickets (f h p : ℕ) (H1 : f + h = 140) (H2 : f * p + h * (p / 2) = 2001) : f * p = 782 := 
sorry

end charity_tickets_l633_633524


namespace fibonacci_periodicity_l633_633505

-- Definitions for p-arithmetic and Fibonacci sequence
def is_prime (p : ℕ) := Nat.Prime p
def sqrt_5_extractable (p : ℕ) : Prop := ∃ k : ℕ, p = 5 * k + 1 ∨ p = 5 * k - 1

-- Definitions of Fibonacci sequences and properties
def fibonacci : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fibonacci n + fibonacci (n + 1)

-- Main theorem
theorem fibonacci_periodicity (p : ℕ) (r : ℕ) (h_prime : is_prime p) (h_not_2_or_5 : p ≠ 2 ∧ p ≠ 5)
    (h_period : r = (p+1) ∨ r = (p-1)) (h_div : (sqrt_5_extractable p → r ∣ (p - 1)) ∧ (¬ sqrt_5_extractable p → r ∣ (p + 1)))
    : (fibonacci (p+1) % p = 0 ∨ fibonacci (p-1) % p = 0) := by
          sorry

end fibonacci_periodicity_l633_633505


namespace count_digit_five_1_to_700_l633_633291

def contains_digit_five (n : ℕ) : Prop :=
  n.digits 10 ∈ [5]

def count_up_to (n : ℕ) (p : ℕ → Prop) : ℕ :=
  (finset.range n).count p

theorem count_digit_five_1_to_700 : count_up_to 701 contains_digit_five = 52 := sorry

end count_digit_five_1_to_700_l633_633291


namespace exists_infinite_n_with_digit_parity_sequence_l633_633650

theorem exists_infinite_n_with_digit_parity_sequence (m : ℕ) (hm : m > 1) :
  ∃ (a : ℕ → ℕ), (∀ i < m, a i < 10) ∧ (a 1 = 5) ∧ (∀ i < m, (a i).odd ↔ (a (i-1)).even) ∧ 
  ∃ infinitely_many (n : ℕ), ∀ i < m, (5^n % 10^m) % 10^(i+1) / 10^i = a i :=
by sorry

end exists_infinite_n_with_digit_parity_sequence_l633_633650


namespace no_snow_five_days_l633_633835

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633835


namespace area_ratio_correct_l633_633883

noncomputable def concentric_circles_area_ratio : ℕ :=
  let diameter_red : ℕ := 2
  let diameter_middle : ℕ := 4
  let diameter_large : ℕ := 6
  let radius_red := diameter_red / 2
  let radius_middle := diameter_middle / 2
  let radius_large := diameter_large / 2
  let area_red := Real.pi * radius_red^2
  let area_middle := Real.pi * radius_middle^2
  let area_large := Real.pi * radius_large^2
  let area_green := area_large - area_middle
  let ratio := area_green / area_red
  ratio

theorem area_ratio_correct : concentric_circles_area_ratio = 5 :=
by
  sorry

end area_ratio_correct_l633_633883


namespace impossible_tiling_l_tromino_l633_633363

theorem impossible_tiling_l_tromino (k : ℕ) : ¬ ∃ (tiling : (ℕ × ℕ) → ℕ), 
  (∀ x y, 0 ≤ x ∧ x < 5 → 0 ≤ y ∧ y < 7 → ∃ n, tiling (x, y) = k * n) ∧ 
  (∀ t, ttro t ∧ covers_tiling t tiling → 
    ∀ (x y), (x, y) ∈ cells_covered_by t → tiling (x, y) = k) :=
sorry

end impossible_tiling_l_tromino_l633_633363


namespace prime_square_remainders_l633_633578

theorem prime_square_remainders (p : ℕ) (hp : Nat.Prime p) (hgt : p > 5) : 
    {r | ∃ k : ℕ, p^2 = 180 * k + r}.finite ∧ 
    {r | ∃ k : ℕ, p^2 = 180 * k + r} = {1, 145} := 
by
  sorry

end prime_square_remainders_l633_633578


namespace solve_for_x_l633_633314

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) : x = 37 := by
  sorry

end solve_for_x_l633_633314


namespace turnip_weights_are_13_or_16_l633_633099

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l633_633099


namespace percentage_runs_by_running_l633_633929

theorem percentage_runs_by_running 
  (total_runs boundaries sixes threes twos singles : ℕ)
  (h_total: total_runs = 190)
  (h_boundaries: boundaries = 7 * 4)
  (h_sixes: sixes = 6 * 6)
  (h_threes: threes = 3 * 3)
  (h_twos: twos = 11 * 2)
  (h_runs_by_running: singles = total_runs - (boundaries + sixes) - (threes + twos))
  (h_total_runs_by_running: singles + threes + twos = 126) :
  (singles + threes + twos : ℚ) / total_runs * 100 ≈ 66.32 := 
  by
    sorry

end percentage_runs_by_running_l633_633929


namespace find_m_l633_633311

theorem find_m {x : ℝ} (m : ℝ) (h : ∀ x, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2 * x > m * x)) : m = 1 :=
sorry

end find_m_l633_633311


namespace smallest_n_sum_lt_zero_l633_633703

variables {a : ℕ → ℚ}
variables (h_arith_seq : ∃ d : ℚ, ∀ n : ℕ, a n = a 1 + (n - 1) * d)
variables (a1_pos : a 1 > 0)
variables (h_sum_pos: a 2022 + a 2023 > 0)
variables (h_prod_neg: a 2022 * a 2023 < 0)

theorem smallest_n_sum_lt_zero : ∃ n : ℕ, (n = 4045) ∧ (∑ i in finset.range n, a i) < 0 := sorry

end smallest_n_sum_lt_zero_l633_633703


namespace right_triangle_fab_eccentricity_l633_633444

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  (classical.some (quadratic_eq_exists (a * a - b * b) (-b * b) 1 rfl))

theorem right_triangle_fab_eccentricity :
  ∀ {a b : ℝ} (h : a > b ∧ b > 0),
  eccentricity_of_ellipse a b h = (-1 + Real.sqrt 5) / 2 := by
  sorry

end right_triangle_fab_eccentricity_l633_633444


namespace cameron_minimum_average_l633_633724

-- Definitions of the conditions
def required_average : ℝ := 85.0
def total_semesters : ℕ := 5
def first_three_scores : List ℝ := [84.0, 88.0, 80.0]

-- Define the function that calculates the total and average scores
def total_points_needed (required_average : ℝ) (total_semesters : ℕ) : ℝ :=
  required_average * total_semesters

def total_points_first_three (scores : List ℝ) : ℝ :=
  scores.sum

def points_needed (required_points : ℝ) (achieved_points : ℝ) : ℝ :=
  required_points - achieved_points

def average_needed (points : ℝ) (semesters : ℕ) : ℝ :=
  points / (semesters : ℝ)

-- The theorem that needs to be proved
theorem cameron_minimum_average :
  average_needed (points_needed (total_points_needed required_average total_semesters) (total_points_first_three first_three_scores)) 2 = 86.5 :=
by sorry

end cameron_minimum_average_l633_633724


namespace Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l633_633561

-- Defining the number of cookies each person had
def Alyssa_cookies : ℕ := 1523
def Aiyanna_cookies : ℕ := 3720
def Brady_cookies : ℕ := 2265

-- Proving the statements
theorem Aiyanna_more_than_Alyssa : Aiyanna_cookies - Alyssa_cookies = 2197 := by
  sorry

theorem Brady_fewer_than_Aiyanna : Aiyanna_cookies - Brady_cookies = 1455 := by
  sorry

theorem Brady_more_than_Alyssa : Brady_cookies - Alyssa_cookies = 742 := by
  sorry

end Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l633_633561


namespace recycled_cans_num_l633_633882

-- Define the initial number of cans and the recycling condition
def initial_cans : ℕ := 243
def recycle_ratio : ℕ := 3

-- Define the number of new cans eventually made as a function
def total_new_cans (n : ℕ) : ℕ :=
  let k := (n : ℚ) / recycle_ratio
  let terms := Nat.log recycle_ratio k
  let first_term := k / recycle_ratio + 1
  List.sum [first_term.pow(i) for i in List.range(terms)]

-- Proof problem statement: Prove that total_new_cans initial_cans equals 121
theorem recycled_cans_num : total_new_cans initial_cans = 121 := by
  sorry

end recycled_cans_num_l633_633882


namespace smallest_n_l633_633766

theorem smallest_n (n : ℕ) (x : fin n → ℝ) 
  (h1 : ∀ i, 0 ≤ x i) 
  (h2 : (finset.univ : finset (fin n)).sum x = 1) 
  (h3 : (finset.univ : finset (fin n)).sum (λ i => (x i) ^ 2) ≤ 1/50) : 
  50 ≤ n := 
sorry

end smallest_n_l633_633766


namespace radius_of_inscribed_sphere_l633_633557

theorem radius_of_inscribed_sphere (AC AD h p : ℝ)
  (H1 : 2 * AC = 9 * AD)
  (H2 : ∀ A M D C, is_angle_bisector A M D C)
  (H3 : ∃ F M N, is_median_intersection F M A D B ∧ plane_passing_through F M N DB = N)
  (H4 : CA / AD = DN / NB + 1)
  (H5 : area_ratio A D B ABCD = p)
  (H6 : perp_from_vertex C plane A B D = h)
  (H7 : plane_passing_through N is_parallel_to ACB ∧ intersects_edges N CD = K ∧ DA = L) :
  radius_of_inscribed_sphere D K L N = (h * p * (real.sqrt 193 - 11)) / (8 * (real.sqrt 193 + 7)) :=
sorry

end radius_of_inscribed_sphere_l633_633557


namespace eleven_consecutive_integers_sum_l633_633906

theorem eleven_consecutive_integers_sum :
  ∃ x : ℤ, (∀ n : ℤ, n ∈ finset.range 11 → (x + n) ∈ finset.range' 25 36) ∧ 
           ((finset.range 6).sum (λ n, x + n) = (finset.range 5).sum (λ n, x + n + 6)) :=
by
  -- Define x to be 25
  let x := 25

  -- The eleven consecutive integers from 25 to 35
  have h : ∀ n, n ∈ finset.range 11 → x + n ∈ finset.range' 25 36 := sorry

  -- Calculate the sum of the first six numbers
  have S1 : (finset.range 6).sum (λ n, x + n) = 165 := sorry
  
  -- Calculate the sum of the last five numbers
  have S2 : (finset.range 5).sum (λ n, x + n + 6) = 165 := sorry

  -- Sum equality conclusion
  use x
  exact ⟨h, by simp [S1, S2]⟩

end eleven_consecutive_integers_sum_l633_633906


namespace values_of_2n_plus_m_l633_633828

theorem values_of_2n_plus_m (n m : ℤ) (h1 : 3 * n - m ≤ 4) (h2 : n + m ≥ 27) (h3 : 3 * m - 2 * n ≤ 45) 
  (h4 : n = 8) (h5 : m = 20) : 2 * n + m = 36 := by
  sorry

end values_of_2n_plus_m_l633_633828


namespace base6_to_base10_l633_633434

theorem base6_to_base10 (c d : ℕ) (h1 : 524 = 2 * (10 * c + d)) (hc : c < 10) (hd : d < 10) :
  (c * d : ℚ) / 12 = 3 / 4 := by
  sorry

end base6_to_base10_l633_633434


namespace total_students_high_school_l633_633719

variable (students_music students_art students_both students_neither : ℕ)

theorem total_students_high_school :
  students_music = 30 →
  students_art = 10 →
  students_both = 10 →
  students_neither = 470 →
  (students_music - students_both) + (students_art - students_both) + students_both + students_neither = 500 :=
by
  intros h_music h_art h_both h_neither
  rw [h_music, h_art, h_both, h_neither]
  sorry

end total_students_high_school_l633_633719


namespace log_expr_eq_range_m_l633_633166

noncomputable def log_expr : ℝ :=
  log 3 (427 / 3) + log10 25 + log10 4 + 7^(log 7 2) + log 2 3 * log 3 4

theorem log_expr_eq : log_expr = 23 / 4 :=
by
  sorry

def setA (x : ℝ) : Prop := -5 ≤ x ∧ x ≤ 2

def setB (m : ℝ) (x : ℝ) : Prop := m - 1 < x ∧ x < 2 * m + 1

theorem range_m (m : ℝ) : (∀ x, setB m x → setA x) ↔ m ∈ (Iio (-2)) ∪ (Icc (-1) (1 / 2)) :=
by
  sorry

end log_expr_eq_range_m_l633_633166


namespace probability_no_snow_l633_633844

theorem probability_no_snow {
  let p_snow : ℚ := 2/3,     -- Probability of snow on any one day
  let p_no_snow : ℚ := 1 - p_snow,  -- Probability of no snow on any one day
  let p_no_snow_5_days : ℚ := p_no_snow ^ 5  -- Probability of no snow on any of the five days
} : p_no_snow_5_days = 1 / 243 := by
  sorry

end probability_no_snow_l633_633844


namespace log_expression_eq_l633_633896

theorem log_expression_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log (y^4)) * 
  (Real.log (y^3) / Real.log (x^6)) * 
  (Real.log (x^4) / Real.log (y^3)) * 
  (Real.log (y^4) / Real.log (x^2)) * 
  (Real.log (x^6) / Real.log y) = 
  16 * Real.log x / Real.log y := 
sorry

end log_expression_eq_l633_633896


namespace count_digit_five_1_to_700_l633_633295

def contains_digit_five (n : ℕ) : Prop :=
  n.digits 10 ∈ [5]

def count_up_to (n : ℕ) (p : ℕ → Prop) : ℕ :=
  (finset.range n).count p

theorem count_digit_five_1_to_700 : count_up_to 701 contains_digit_five = 52 := sorry

end count_digit_five_1_to_700_l633_633295


namespace rectangle_width_decrease_percent_l633_633451

theorem rectangle_width_decrease_percent (L W : ℝ) (h : L * W = L * W) :
  let L_new := 1.3 * L
  let W_new := W / 1.3 
  let percent_decrease := (1 - (W_new / W)) * 100
  percent_decrease = 23.08 :=
sorry

end rectangle_width_decrease_percent_l633_633451


namespace find_n_l633_633208

theorem find_n : ∃ (n : ℕ), (∑ k in finset.range (n - 3) + 4, 1 / (Real.sqrt k + Real.sqrt (k + 1))) = 10 ∧ n = 143 :=
begin
  sorry
end

end find_n_l633_633208


namespace january_revenue_fraction_l633_633521

variable {N D J : ℚ}

def condition1 : Prop := N = (2 / 5) * D
def condition2 : Prop := D = 3.75 * ((N + J) / 2)

theorem january_revenue_fraction (h1 : condition1) (h2 : condition2) : J / N = 1 / 3 := 
by
  sorry

end january_revenue_fraction_l633_633521


namespace sum_f_eq_2018_l633_633670

-- Define the function f
def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

-- We need to state the sum 
theorem sum_f_eq_2018 :
  ∑ i in (finset.range 2018).map nat.cast + ∑ i in (finset.range 2018).map (λ i, (1 / (nat.cast i.succ)): ℝ), f i = 2018 :=
by sorry

end sum_f_eq_2018_l633_633670


namespace min_value_expr_l633_633458

theorem min_value_expr (x : ℝ) (h : x > 0) : 2 * real.sqrt x + 3 / real.sqrt x ≥ 2 * real.sqrt 6 :=
begin
  sorry
end

end min_value_expr_l633_633458


namespace find_fourth_term_geometric_progression_l633_633597

theorem find_fourth_term_geometric_progression (x : ℝ) (a1 a2 a3 : ℝ) (r : ℝ)
  (h1 : a1 = x)
  (h2 : a2 = 3 * x + 6)
  (h3 : a3 = 7 * x + 21)
  (h4 : ∃ r, a2 / a1 = r ∧ a3 / a2 = r)
  (hx : x = 3 / 2) :
  7 * (7 * x + 21) = 220.5 :=
by
  sorry

end find_fourth_term_geometric_progression_l633_633597


namespace polar_equation_C1_cartesian_equation_C2_range_OA_OB_l633_633725

theorem polar_equation_C1 (α θ : ℝ) :
  (∃ (α : ℝ), ∀ x y : ℝ, 
    x = 1 + cos α ∧ y = sin α ∧ 
    (x - 1)^2 + y^2 = 1) →
  (∀ α : ℝ, ∃ θ: ℝ, ρ = 2 * cos θ) := 
sorry

theorem cartesian_equation_C2 (ρ θ : ℝ) :
  (∃ θ : ℝ, ρ * cos θ^2 = 2 * sin θ) →
  (∀ ρ : ℝ, ∃ x y : ℝ, x^2 = 2y) := 
sorry

theorem range_OA_OB (k : ℝ) :
  (∀ k : ℝ, 1 ≤ k ∧ k < sqrt 3 →
    (∃ t : ℝ, |OA| = 2 * cos α ∧ 
    |OB| = 2 * sin α / cos α^2 → 
    4 * tan α = 4k)) →
  ∃ k : ℝ, 4 ≤ 4 * k ∧ 4 * k < 4 * sqrt 3 :=
sorry

end polar_equation_C1_cartesian_equation_C2_range_OA_OB_l633_633725


namespace difference_is_correct_l633_633988

/-
Chris is trying to sell his car for $5200 and has gotten three price offers.
1. The first buyer offered to pay the full price if Chris would pay for the car maintenance inspection, which cost a tenth of Chris’s asking price.
2. The second buyer agreed to pay the price if Chris replaced the headlights for $80, the tires for three times the cost of the headlights, and the battery which costs twice as much as the tires.
3. The third buyer asked for a 15% discount on the selling price and all he wants is for Chris to do the car fresh paint job which costs one-fifth of the discounted price.
Prove that the difference between the amounts Chris will earn from the highest offer and the lowest offer is $1144.
-/

def car_price : ℕ := 5200
def maintenance_cost : ℕ := car_price / 10
def net_amount_first_buyer : ℕ := car_price - maintenance_cost

def headlight_cost : ℕ := 80
def tire_cost : ℕ := 3 * headlight_cost
def battery_cost : ℕ := 2 * tire_cost
def total_cost_second_buyer : ℕ := headlight_cost + tire_cost + battery_cost
def net_amount_second_buyer : ℕ := car_price - total_cost_second_buyer

def discount_rate : ℚ := 0.15
def discounted_price : ℕ := car_price - (car_price * discount_rate).to_nat
def paint_job_cost : ℕ := discounted_price / 5
def net_amount_third_buyer : ℕ := discounted_price - paint_job_cost

def highest_offer : ℕ := max (max net_amount_first_buyer net_amount_second_buyer) net_amount_third_buyer
def lowest_offer : ℕ := min (min net_amount_first_buyer net_amount_second_buyer) net_amount_third_buyer
def difference : ℕ := highest_offer - lowest_offer

theorem difference_is_correct : difference = 1144 :=
by
  -- conversion from problem to a lean statement
  sorry

end difference_is_correct_l633_633988


namespace geometric_seq_sum_l633_633229

theorem geometric_seq_sum :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    (∀ n, a (n + 1) = a n * q) ∧ 
    (a 4 + a 7 = 2) ∧ 
    (a 5 * a 6 = -8) → 
    a 1 + a 10 = -7 := 
by sorry

end geometric_seq_sum_l633_633229


namespace arrangements_of_students_l633_633513

theorem arrangements_of_students (s : Finset ℕ) (A : ℕ) (h : A ∈ s) (hs : s.card = 5) :
  (∃ L : List ℕ, (s.erase A).card = 4 ∧ L.perm s ∧ A ≠ L.head) → 
  4 * (s.erase A).card.factorial = 96 := by
sorry

end arrangements_of_students_l633_633513


namespace triangular_array_problem_l633_633556

theorem triangular_array_problem :
  ∃ (arrangements : fin 13 → ℕ), 
  (∀ i, arrangements i = 0 ∨ arrangements i = 1) ∧
  (∑ k in finset.range 13, binomial 12 k * arrangements k % 6 = 0) ∧
  (2 ^ 9 * (choose 4 0 + choose 4 3) = 2560) := 
sorry

end triangular_array_problem_l633_633556


namespace locus_equation_perimeter_greater_than_3sqrt3_l633_633329

-- The locus W and the conditions 
def locus_eq (P : ℝ × ℝ) : Prop :=
  P.snd = P.fst ^ 2 + 1/4

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  |P.snd| = sqrt (P.fst ^ 2 + (P.snd - 1/2) ^ 2)

-- Prove part (1): The locus W is y = x^2 + 1/4
theorem locus_equation (x y : ℝ) : 
  point_on_x_axis (x, y) → locus_eq (x, y) :=
sorry

-- Prove part (2): Perimeter of rectangle ABCD is greater than 3sqrt(3) if three vertices are on W
def rectangle_on_w (A B C : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  locus_eq A ∧ locus_eq B ∧ locus_eq C ∧ locus_eq D ∧ 
  (∃x₁ x₂ x₃ x₄ : ℝ, A = (x₁, x₁ ^ 2 + 1/4) ∧ B = (x₂, x₂ ^ 2 + 1/4) ∧ 
  C = (x₃, x₃ ^ 2 + 1/4) ∧ D = (x₄, x₄ ^ 2 + 1/4))

theorem perimeter_greater_than_3sqrt3 (A B C D : ℝ × ℝ) : 
  rectangle_on_w A B C D → 
  2 * (abs (B.fst - A.fst) + abs (C.fst - B.fst)) > 3 * sqrt 3 :=
sorry

end locus_equation_perimeter_greater_than_3sqrt3_l633_633329


namespace neither_people_is_42_l633_633511

def conference : Type := { total_people : ℕ // total_people = 100 } 

structure ConferenceCond (C : conference) :=
  (writers : ℕ)
  (editors : ℕ)
  (both_writers_and_editors : ℕ)
  (max_both_writers_and_editors : ℕ)
  (num_neither : ℕ)
  (writers_spec : writers = 40)
  (editors_spec : 38 < editors)
  (both_spec : both_writers_and_editors ≤ max_both_writers_and_editors)
  (max_both_spec : max_both_writers_and_editors = 21)
  (neither_spec : num_neither = C.total_people - (writers - both_writers_and_editors + editors - both_writers_and_editors + both_writers_and_editors))
  (neither_val : num_neither = 42)

theorem neither_people_is_42 (C : conference) (hC : ConferenceCond C) : hC.num_neither = 42 :=
  by
    rw [hC.neither_val]
    sorry

end neither_people_is_42_l633_633511


namespace marked_price_l633_633536

theorem marked_price (a b c d e f : ℝ) (discount purchase_price gain selling_price marked_price : ℝ) :
  a = 36 →
  b = 0.15 →
  discount = a * b →
  purchase_price = a - discount →
  c = 0.25 →
  gain = purchase_price * c →
  selling_price = purchase_price + gain →
  d = 0.10 →
  selling_price = marked_price * (1 - d) →
  marked_price = 42.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  have h_discount : discount = 36 * 0.15, from h1 ▸ h2 ▸ rfl,
  have h_purchase_price : purchase_price = 36 - 36 * 0.15, from h3 ▸ h4 ▸ h_discount ▸ rfl,
  have h_gain : gain = 30.6 * 0.25, from calc
    purchase_price = 30.6 : h_purchase_price
    ... * c = 30.6 * 0.25 : h5 ▸ rfl,
  have h_selling_price : selling_price = 30.6 + 7.65, from calc
    purchase_price = 30.6 : h_purchase_price
    ... + gain = 30.6 + 7.65 : h6 ▸ h_gain ▸ rfl,
  have h_price_eq : selling_price = 38.25, from h7 ▸ h_selling_price ▸ rfl,
  have h_final : 38.25 = marked_price * (1 - 0.10) ↔ marked_price = 42.5, from calc
    selling_price = 38.25 : h_price_eq
    ... = marked_price * 0.90 : h8,
  exact h9 ▸ h_final
  sorry

end marked_price_l633_633536


namespace rational_solution_l633_633635

theorem rational_solution (m n : ℤ) (h : a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2)) : 
  ∃ a : ℚ, a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2) :=
by {
  sorry
}

end rational_solution_l633_633635


namespace someone_made_a_mistake_l633_633484

theorem someone_made_a_mistake : 
  (∀ (vasya_sums petya_sums : list ℕ), 
    vasya_sums.length = 200 ∧ 
    petya_sums.length = 200 ∧ 
    (∃ (vasya_numbers petya_numbers : list ℕ), 
      (∀ i < 200, vasya_numbers.sum_digits = vasya_sums.nth_le i (by omega) ∧
                  petya_numbers.sum_digits = petya_sums.nth_le i (by omega)) ∧
      ∃ (tanya_numbers : list ℕ),
        (∀ i < 200, 
          tanya_numbers.nth_le i (by omega) = 
            (vasya_sums.nth_le i (by omega) * petya_sums.nth_le i (by omega)) ∧
          consecutive_naturals tanya_numbers) ->
        false).
Proof := sorry

end someone_made_a_mistake_l633_633484


namespace determine_a4_l633_633257

theorem determine_a4 :
  let a : ℕ → ℚ := λ n, match n with
                         | 0     => 1
                         | _ + 1 => (1 / 2) * (a n) + 1 / (2 ^ n)
                       end
  in a 4 = 1 / 2 :=
begin
  sorry
end

end determine_a4_l633_633257


namespace magnitude_of_linear_combination_l633_633689

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem magnitude_of_linear_combination (h₁ : inner_product a b = 0)
  (h₂ : ∥a∥ = 1) 
  (h₃ : ∥b∥ = 2) :
  ∥2 • a - b∥ = 2 :=
by sorry

end magnitude_of_linear_combination_l633_633689


namespace base7_digits_of_1200_l633_633695

theorem base7_digits_of_1200 : ∃ digits : ℕ, digits = 4 ∧ digits = Nat.digits 7 1200.length := by
  sorry

end base7_digits_of_1200_l633_633695


namespace unique_circumscribing_sphere_l633_633594

-- Define a hexahedron and conditions
structure Hexahedron where
  vertices : List (EuclideanSpace 3 ℝ)
  faces : List (List (EuclideanSpace 3 ℝ))
  hex_property : vertices.length = 8 ∧ faces.length = 6 ∧ 
    (∀ face ∈ faces, face.length = 4 ∧ ∃ circ: Circle, ∀ x ∈ face, x ∈ circ.carrier)

-- The theorem to show the existence and uniqueness of the circumscribing sphere
theorem unique_circumscribing_sphere (H : Hexahedron) : 
  ∃! sphere : Sphere (EuclideanSpace 3 ℝ), ∀ v ∈ H.vertices, v ∈ sphere.carrier :=
sorry

end unique_circumscribing_sphere_l633_633594


namespace max_value_of_f_on_interval_min_value_of_f_on_interval_l633_633204

-- Define the function f(x) = x^3 - 3x^2 + 5
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

-- Define the interval
def interval := set.Icc (-1:ℝ) 1

-- State that the maximum value on the interval is 5
theorem max_value_of_f_on_interval : 
  is_lub (set.image f interval) 5 := 
sorry

-- State that the minimum value on the interval is 1
theorem min_value_of_f_on_interval : 
  is_glb (set.image f interval) 1 := 
sorry

end max_value_of_f_on_interval_min_value_of_f_on_interval_l633_633204


namespace circumcenter_of_condition_C_l633_633731

variable {V : Type*} [inner_product_space ℝ V]

variables (A B C O : V)

-- Definition of the conditions related to statement C
def condition_C : Prop :=
  (inner (O + B - A) (B - A) = 0) ∧
  (inner (O + C - B) (C - B) = 0) ∧
  (inner (O + A - C) (A - C) = 0)

-- The theorem statement that if condition_C holds, then O is the circumcenter of triangle ABC.
theorem circumcenter_of_condition_C (hC : condition_C A B C O) : (circumcenter A B C = O) :=
sorry

end circumcenter_of_condition_C_l633_633731


namespace identify_opposite_pair_l633_633152

-- Define what it means for two numbers to be opposites
def are_opposite (a b : ℝ) : Prop :=
  a = -b

-- Define the given pairs in the problem
def pair_A : ℝ × ℝ := (-(-2), 2)
def pair_B : ℝ × ℝ := (+(-3), -(+3))
def pair_C : ℝ × ℝ := (1/2, -2)
def pair_D : ℝ × ℝ := (-(-5), -abs(+5))

-- State the theorem to be proved
theorem identify_opposite_pair : 
  ¬ are_opposite pair_A.1 pair_A.2 ∧
  ¬ are_opposite pair_B.1 pair_B.2 ∧
  ¬ are_opposite pair_C.1 pair_C.2 ∧
  are_opposite pair_D.1 pair_D.2 :=
sorry

end identify_opposite_pair_l633_633152


namespace ratio_of_larger_to_smaller_is_sqrt_six_l633_633471

def sum_of_squares_eq_seven_times_difference (a b : ℝ) : Prop := 
  a^2 + b^2 = 7 * (a - b)

theorem ratio_of_larger_to_smaller_is_sqrt_six {a b : ℝ} (h : sum_of_squares_eq_seven_times_difference a b) (h1 : a > b) : 
  a / b = Real.sqrt 6 :=
sorry

end ratio_of_larger_to_smaller_is_sqrt_six_l633_633471


namespace only_n1_makes_n4_plus4_prime_l633_633634

theorem only_n1_makes_n4_plus4_prime (n : ℕ) (h : n > 0) : (n = 1) ↔ Prime (n^4 + 4) :=
sorry

end only_n1_makes_n4_plus4_prime_l633_633634


namespace determine_monotonic_solve_inequality_l633_633664

variable {R : Type*} [OrderedRing R] [DecidableLinearOrder R]

noncomputable def monotonic_odd_function (f : R → R) :=
  (∀ x y : R, x < y → f x > f y) ∧ (∀ x : R, f (-x) = -f x)

theorem determine_monotonic (f : R → R) (h_mono_odd : monotonic_odd_function f) (h1 : f 1 = -2) :
  ∀ x1 x2 : R, x1 < x2 → f x1 > f x2 :=
by
  sorry

theorem solve_inequality (f : R → R) (h_mono_odd : monotonic_odd_function f) (h1 : f 1 = -2) :
  ∀ x : R, 1 < x ∧ x < 2 → f x + f (2 * x - x^2 - 2) < 0 :=
by
  sorry

end determine_monotonic_solve_inequality_l633_633664


namespace gcd_a_41_gcd_a_n_l633_633195

def gcd_set (n : ℕ) : ℕ :=
  if n = 0 then 0 else 
  let primes := { p : ℕ | p.prime ∧ (p - 1) ∣ (n - 1) } in
  if n.even then 2 * primes.prod else primes.prod

theorem gcd_a_41 (a : ℕ) (ha : a ≥ 2) : gcd_set 41 = 13530 := by
  sorry

theorem gcd_a_n (n : ℕ) : gcd_set n = 
  if n = 0 then 0 else 
  let primes := { p : ℕ | p.prime ∧ (p - 1) ∣ (n - 1) } in
  if n.even then 2 * primes.prod else primes.prod := by
  sorry

end gcd_a_41_gcd_a_n_l633_633195


namespace blueberries_count_l633_633925

theorem blueberries_count (total_berries raspberries blackberries blueberries : ℕ)
  (h1 : total_berries = 42)
  (h2 : raspberries = total_berries / 2)
  (h3 : blackberries = total_berries / 3)
  (h4 : blueberries = total_berries - raspberries - blackberries) :
  blueberries = 7 :=
sorry

end blueberries_count_l633_633925


namespace fraction_of_satisfactory_is_15_over_23_l633_633808

def num_students_with_grade_A : ℕ := 6
def num_students_with_grade_B : ℕ := 5
def num_students_with_grade_C : ℕ := 4
def num_students_with_grade_D : ℕ := 2
def num_students_with_grade_F : ℕ := 6

def num_satisfactory_students : ℕ := 
  num_students_with_grade_A + num_students_with_grade_B + num_students_with_grade_C

def total_students : ℕ := 
  num_satisfactory_students + num_students_with_grade_D + num_students_with_grade_F

def fraction_satisfactory : ℚ := 
  (num_satisfactory_students : ℚ) / (total_students : ℚ)

theorem fraction_of_satisfactory_is_15_over_23 : 
  fraction_satisfactory = 15/23 :=
by
  -- proof omitted
  sorry

end fraction_of_satisfactory_is_15_over_23_l633_633808


namespace not_snow_probability_l633_633867

theorem not_snow_probability :
  let p_snow : ℚ := 2 / 3,
      p_not_snow : ℚ := 1 - p_snow in
  (p_not_snow ^ 5) = 1 / 243 := by
  sorry

end not_snow_probability_l633_633867


namespace abs_pi_sub_abs_pi_sub_10_eq_l633_633593

theorem abs_pi_sub_abs_pi_sub_10_eq
  (pi : ℝ) (h : pi < 10) : |pi - |pi - 10|| = 10 - 2 * pi :=
by
  sorry

end abs_pi_sub_abs_pi_sub_10_eq_l633_633593


namespace number_of_chili_beans_ordered_l633_633176

-- Conditions
variables {T C : ℕ}
def ratio_condition := T = C / 2
def total_cans := T + C = 12

-- Proof statement
theorem number_of_chili_beans_ordered (T C : ℕ) (h1 : ratio_condition T C) (h2 : total_cans T C) : C = 8 :=
sorry

end number_of_chili_beans_ordered_l633_633176


namespace function_even_and_monotonically_increasing_l633_633562

-- Definition: Even Function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Definition: Monotonically Increasing on (0, ∞)
def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- Given Function
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem to prove
theorem function_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on_pos f := by
  sorry

end function_even_and_monotonically_increasing_l633_633562


namespace turnip_weights_are_13_or_16_l633_633096

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l633_633096


namespace problem_inequality_l633_633660

theorem problem_inequality (k m n : ℕ) (hk1 : 1 < k) (hkm : k ≤ m) (hmn : m < n) :
  (1 + m) ^ 2 > (1 + n) ^ m :=
  sorry

end problem_inequality_l633_633660


namespace relationship_inequality_l633_633603

noncomputable def a : ℝ := 2 ^ 0.3
noncomputable def b : ℝ := 0.3 ^ 2
noncomputable def c (x : ℝ) (hx : x > 1) : ℝ := log x (x ^ 2 + 0.3)

theorem relationship_inequality (x : ℝ) (hx : x > 1) : 
  let a := 2 ^ 0.3
  let b := 0.3 ^ 2
  let c := log x (x ^ 2 + 0.3)
  b < a ∧ a < c :=
sorry

end relationship_inequality_l633_633603


namespace infinite_ap_with_large_digit_sum_l633_633417

theorem infinite_ap_with_large_digit_sum (k : ℕ) (hk : k > 0) :
  ∃ (a d : ℕ), (∀ n : ℕ, ∃ b : ℕ, n = b → (nat.digits 10 (a + n * d)).sum > k) ∧ ¬ (∃ m : ℕ, d = 10 * m) :=
by sorry

end infinite_ap_with_large_digit_sum_l633_633417


namespace yearly_profit_l633_633368

variable (num_subletters : ℕ) (rent_per_subletter_per_month rent_per_month : ℕ)

theorem yearly_profit (h1 : num_subletters = 3)
                     (h2 : rent_per_subletter_per_month = 400)
                     (h3 : rent_per_month = 900) :
  12 * (num_subletters * rent_per_subletter_per_month - rent_per_month) = 3600 :=
by
  sorry

end yearly_profit_l633_633368


namespace polar_coordinate_of_line_x_eq_one_l633_633629

theorem polar_coordinate_of_line_x_eq_one (ρ θ : ℝ) : 
  (∀ x, x = ρ * Real.cos θ) → (ρ * Real.cos θ = 1) :=
begin
  -- Substitute x = 1
  intro h,
  have : 1 = ρ * Real.cos θ := h 1,
  exact this
end

end polar_coordinate_of_line_x_eq_one_l633_633629


namespace derivative_neither_odd_nor_even_l633_633671

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2) * x^2

theorem derivative_neither_odd_nor_even :
  let f' (x : ℝ) := (Real.log x - (1/2) * x^2)' in
  ∀ x : ℝ, f' (-x) ≠ f' x ∧ f' (-x) ≠ -f' x :=
by
  let f' (x : ℝ) := (1 / x) - x
  intros
  sorry

end derivative_neither_odd_nor_even_l633_633671


namespace prob_ge_neg2_l633_633241

noncomputable def normal_dist (mean variance : ℝ) : Type := sorry -- This is a placeholder for the normal distribution type.

variable (ξ : normal_dist 0 4)

theorem prob_ge_neg2 (h : P(ξ ≥ 2) = 0.3) : P(ξ ≥ -2) = 0.7 :=
begin
  sorry
end

end prob_ge_neg2_l633_633241


namespace TruckX_initial_distance_l633_633887

-- Definitions
def initial_distance (X_speed Y_speed : ℕ) (time_over : ℕ) (additional_dist : ℕ) (relative_speed : ℕ) : ℕ :=
  let closed_gap := time_over * relative_speed
  in closed_gap + additional_dist

-- Given conditions as definitions
def TruckX_speed := 47
def TruckY_speed := 53
def time_to_overtake := 3
def additional_distance := 5
def relative_speed := TruckY_speed - TruckX_speed

-- Theorem statement, proving the initial distance
theorem TruckX_initial_distance :
  initial_distance TruckX_speed TruckY_speed time_to_overtake additional_distance relative_speed = 23 :=
by
  sorry

end TruckX_initial_distance_l633_633887


namespace find_f_of_4_l633_633237

noncomputable def power_function (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_of_4 :
  (∃ α : ℝ, power_function 3 α = Real.sqrt 3) →
  power_function 4 (1/2) = 2 :=
by
  sorry

end find_f_of_4_l633_633237


namespace triangle_area_l633_633956

/-- Define the area of a triangle with one side of length 13, an opposite angle of 60 degrees, and side ratio 4:3. -/
theorem triangle_area (a b c : ℝ) (A : ℝ) (S : ℝ) 
  (h_a : a = 13)
  (h_A : A = Real.pi / 3)
  (h_bc_ratio : b / c = 4 / 3)
  (h_cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h_area : S = 1 / 2 * b * c * Real.sin A) :
  S = 39 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l633_633956


namespace turnip_weight_possible_l633_633078

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l633_633078


namespace circle_sum_condition_l633_633404

theorem circle_sum_condition (n : ℕ) (n_ge_1 : n ≥ 1)
  (x : Fin n → ℝ) (sum_x : (Finset.univ.sum x) = n - 1) :
  ∃ j : Fin n, ∀ k : ℕ, k ≥ 1 → k ≤ n → (Finset.range k).sum (fun i => x ⟨(j + i) % n, sorry⟩) ≥ k - 1 :=
sorry

end circle_sum_condition_l633_633404


namespace combined_time_in_pool_l633_633973

theorem combined_time_in_pool : 
    ∀ (Jerry_time Elaine_time George_time Kramer_time : ℕ), 
    Jerry_time = 3 →
    Elaine_time = 2 * Jerry_time →
    George_time = Elaine_time / 3 →
    Kramer_time = 0 →
    Jerry_time + Elaine_time + George_time + Kramer_time = 11 :=
by 
  intros Jerry_time Elaine_time George_time Kramer_time hJerry hElaine hGeorge hKramer
  sorry

end combined_time_in_pool_l633_633973


namespace probability_x_squared_in_interval_l633_633106

theorem probability_x_squared_in_interval:
  let x := ℝ in
  let interval := Set.Icc (-1 : ℝ) 1 in
  let favorable_interval := Set.Ioc (-1/2 : ℝ) 0 ∪ Set.Ioo 0 (1/2 : ℝ) in
  ∃ (P : ℝ), P = (favorable_interval.measure / interval.measure) ∧ P = (1/2 : ℝ)
:= sorry

end probability_x_squared_in_interval_l633_633106


namespace cardinality_A_inter_B_l633_633653

-- Definitions of the sets A and B
def A : Set (ℕ × ℕ) := {p | (∃ x y : ℕ, p = (x, y))}
def B : Set (ℕ × ℕ) := {p | ∃ x y : ℕ, p = (x, y) ∧ x^2 + y^2 = 25}

-- The intersection of sets A and B
def A_inter_B : Set (ℕ × ℕ) := A ∩ B

-- The fact we need to prove: the cardinality of the set A ∩ B is 4
theorem cardinality_A_inter_B : (A_inter_B.toFinset.card = 4) :=
by sorry

end cardinality_A_inter_B_l633_633653


namespace marble_probability_l633_633017

theorem marble_probability :
  let total_marbles : ℕ := 20
  let red_marbles : ℕ := 12
  let blue_marbles : ℕ := 8
  let total_probability : ℚ := (6 * ((12 * 11 * 8 * 7) / (20 * 19 * 18 * 17))) in
  total_probability = (1232 / 4845) :=
by
  sorry

end marble_probability_l633_633017


namespace intensity_solution_of_red_paint_added_l633_633431

/-- Define the intensities and the fraction -/
def original_intensity := 45
def new_intensity := 40
def fraction_replaced := 0.25

/-- Define the remaining fraction and the unknown intensity -/
def remaining_fraction := 1 - fraction_replaced
def unknown_intensity := (new_intensity - remaining_fraction * original_intensity) / fraction_replaced

/-- The goal is to prove that the unknown intensity is 25% -/
theorem intensity_solution_of_red_paint_added :
  unknown_intensity = 25 := by
  sorry

end intensity_solution_of_red_paint_added_l633_633431


namespace range_of_a_l633_633392

def A (a : ℝ) : set ℝ := {x | abs (x - a) < 1}
def B : set ℝ := {x | 1 < x ∧ x < 5}

theorem range_of_a (a : ℝ) : (A a ∩ B).nonempty → 0 < a ∧ a < 6 :=
sorry

end range_of_a_l633_633392


namespace angle_ATI_eq_angle_CTI_l633_633528

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def tangent_circle (A B C : Point) : Circle := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def tangent_point (cir1 cir2 : Circle) : Point := sorry
noncomputable def angle (A B C : Point) : Real := sorry

theorem angle_ATI_eq_angle_CTI (A B C T I : Point) 
  (hI : I = incenter A B C)
  (hT_circumcircle_tangent : tangent_circle A B C ∣ tangent_point (circumcircle A B C))
  (hT_tangent_sides : tangent_circle A B C ∣ tangent_point (line_through A B ∧ line_through B C))
  : angle A T I = angle C T I :=
sorry

end angle_ATI_eq_angle_CTI_l633_633528


namespace sequence_general_formula_l633_633547

def mean_reciprocal (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / (finset.range n).sum (λ k, a (k + 1))

theorem sequence_general_formula (a : ℕ → ℝ)
  (h : ∀ n, mean_reciprocal a n = 1 / (2 * n - 1)) :
  ∀ n, a n = 4 * n - 3 :=
begin
  sorry,
end

end sequence_general_formula_l633_633547


namespace cookies_in_each_bag_l633_633778

-- Definitions based on the conditions
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41
def baggies : ℕ := 6

-- Assertion of the correct answer
theorem cookies_in_each_bag : 
  (chocolate_chip_cookies + oatmeal_cookies) / baggies = 9 := by
  sorry

end cookies_in_each_bag_l633_633778


namespace least_multiple_of_121_l633_633762

def sequence (n : ℕ) : ℕ :=
  if n = 10 then 10
  else if n > 10 then 121 * sequence (n - 1) + 2 * n
  else 0 -- otherwise, default

theorem least_multiple_of_121 :
  (∃ n > 10, sequence n % 121 = 0) ∧ (∀ m > 10, sequence m % 121 = 0 → 21 ≤ m) :=
by
  sorry

end least_multiple_of_121_l633_633762


namespace turnip_bag_weights_l633_633072

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l633_633072


namespace cuberoot_eight_is_512_l633_633462

-- Define the condition on x
def cuberoot_is_eight (x : ℕ) : Prop := 
  x^(1 / 3) = 8

-- The statement to be proved
theorem cuberoot_eight_is_512 : ∃ x : ℕ, cuberoot_is_eight x ∧ x = 512 := 
by 
  -- Proof is omitted
  sorry

end cuberoot_eight_is_512_l633_633462


namespace neg_p_l633_633786

theorem neg_p : ∀ (m : ℝ), ∀ (x : ℝ), (x^2 + m*x + 1 ≠ 0) :=
by
  sorry

end neg_p_l633_633786


namespace juanita_loss_l633_633370

theorem juanita_loss
  (entry_fee : ℝ) (hit_threshold : ℕ) (drum_payment_per_hit : ℝ) (drums_hit : ℕ) :
  entry_fee = 10 →
  hit_threshold = 200 →
  drum_payment_per_hit = 0.025 →
  drums_hit = 300 →
  - (entry_fee - ((drums_hit - hit_threshold) * drum_payment_per_hit)) = 7.50 :=
by
  intros h1 h2 h3 h4
  sorry

end juanita_loss_l633_633370


namespace polynomial_evaluation_l633_633188

theorem polynomial_evaluation :
  ∀ x : ℤ, x = -2 → (x^3 + x^2 + x + 1 = -5) :=
by
  intros x hx
  rw [hx]
  norm_num

end polynomial_evaluation_l633_633188


namespace radio_selling_price_l633_633945

noncomputable def sellingPrice (costPrice : ℝ) (lossPercentage : ℝ) : ℝ :=
  costPrice - (lossPercentage / 100 * costPrice)

theorem radio_selling_price :
  sellingPrice 490 5 = 465.5 :=
by
  sorry

end radio_selling_price_l633_633945


namespace cargo_total_ship_l633_633119

-- Define the initial cargo and the additional cargo loaded
def initial_cargo := 5973
def additional_cargo := 8723

-- Define the total cargo the ship holds after loading additional cargo
def total_cargo := initial_cargo + additional_cargo

-- Statement of the problem
theorem cargo_total_ship (h1 : initial_cargo = 5973) (h2 : additional_cargo = 8723) : 
  total_cargo = 14696 := 
by
  sorry

end cargo_total_ship_l633_633119


namespace math_problem_l633_633163

theorem math_problem :
  (-1:ℤ) ^ 2023 - |(-3:ℤ)| + ((-1/3:ℚ) ^ (-2:ℤ)) + ((Real.pi - 3.14)^0) = 6 := 
by 
  sorry

end math_problem_l633_633163


namespace complex_vector_sum_eq_vector_ba_eq_neg_ab_distance_ab_eq_l633_633344

noncomputable def complex_vector_sum : ℂ := (-3 - complex.I) + (5 + complex.I)

noncomputable def vector_oa : ℂ := -3 - complex.I
noncomputable def vector_ob : ℂ := 5 + complex.I
noncomputable def vector_ab : ℂ := vector_ob - vector_oa
noncomputable def vector_ba : ℂ := -vector_ab

noncomputable def distance_a_b : ℝ := complex.abs vector_ab

theorem complex_vector_sum_eq : complex_vector_sum = 2 := by {
  sorry
}

theorem vector_ba_eq_neg_ab : vector_ba = -8 - 2*complex.I := by {
  sorry
}

theorem distance_ab_eq: distance_a_b = 2 * real.sqrt 17 := by {
  sorry
}

end complex_vector_sum_eq_vector_ba_eq_neg_ab_distance_ab_eq_l633_633344


namespace identify_opposite_pair_l633_633153

-- Define what it means for two numbers to be opposites
def are_opposite (a b : ℝ) : Prop :=
  a = -b

-- Define the given pairs in the problem
def pair_A : ℝ × ℝ := (-(-2), 2)
def pair_B : ℝ × ℝ := (+(-3), -(+3))
def pair_C : ℝ × ℝ := (1/2, -2)
def pair_D : ℝ × ℝ := (-(-5), -abs(+5))

-- State the theorem to be proved
theorem identify_opposite_pair : 
  ¬ are_opposite pair_A.1 pair_A.2 ∧
  ¬ are_opposite pair_B.1 pair_B.2 ∧
  ¬ are_opposite pair_C.1 pair_C.2 ∧
  are_opposite pair_D.1 pair_D.2 :=
sorry

end identify_opposite_pair_l633_633153


namespace smallest_number_divisible_by_conditions_l633_633491

theorem smallest_number_divisible_by_conditions (N : ℕ) (X : ℕ) (H1 : (N - 12) % 8 = 0) (H2 : (N - 12) % 12 = 0)
(H3 : (N - 12) % X = 0) (H4 : (N - 12) % 24 = 0) (H5 : (N - 12) / 24 = 276) : N = 6636 :=
by
  sorry

end smallest_number_divisible_by_conditions_l633_633491


namespace triangle_b_and_tan_A_l633_633315

theorem triangle_b_and_tan_A (a b c : ℝ) (B : ℝ) 
  (h1 : c = 2 * a) 
  (h2 : B = 120 * real.pi / 180) 
  (h3 : 0.5 * a * c * real.sin B = real.sqrt 3 / 2) :
  (b = real.sqrt 7) ∧ (real.tan (real.atan2 (real.sin A) (real.cos A)) = real.sqrt 3 / 5) :=
begin
  sorry
end

end triangle_b_and_tan_A_l633_633315


namespace clock_hand_integer_distances_l633_633935

def hour_hand_length : ℝ := 3
def minute_hand_length : ℝ := 4

theorem clock_hand_integer_distances :
  ∃ occurrences : ℕ,
    (occurrences = 132) ∧
    ∀ t : ℝ, (0 ≤ t ∧ t < 12) → 
      ∃ d : ℝ, 
        d ∈ {i | ∃ n : ℤ, (n : ℝ) = i} ∧ 
        d = real.sqrt (hour_hand_length^2 + minute_hand_length^2 - 2 * hour_hand_length * minute_hand_length * real.cos (t * (2 * real.pi / 12))) → 
        occurrences = 132
:= sorry

end clock_hand_integer_distances_l633_633935


namespace percentage_of_50_l633_633147

theorem percentage_of_50 (P : ℝ) :
  (0.10 * 30) + (P / 100 * 50) = 10.5 → P = 15 := by
  sorry

end percentage_of_50_l633_633147


namespace proof_x_plus_y_l633_633298

theorem proof_x_plus_y (x y : ℝ)
  (h1 : 8^x / 4^(x + y) = 16)
  (h2 : 16^(x + y) / 4^(7 * y) = 1024) :
  x + y = 13 :=
by 
  -- Proof omitted
  sorry

end proof_x_plus_y_l633_633298


namespace ratio_of_areas_l633_633035

theorem ratio_of_areas (r : ℝ) : 
  let original_area := Real.pi * r^2
  let new_radius := 3 * r
  let new_area := Real.pi * new_radius^2
  (original_area / new_area) = (1 / 9) :=
by
  let original_area := Real.pi * r^2
  let new_radius := 3 * r
  let new_area := Real.pi * new_radius^2
  rw [new_radius, new_area]
  sorry

end ratio_of_areas_l633_633035


namespace steel_bar_volume_is_2700_l633_633996

noncomputable def original_volume_of_steel_bar
  (h : ℝ) (r : ℝ) (cylinder_height : ℝ) (surface_area_increase : ℝ) : ℝ :=
  let r_squared := surface_area_increase / (2 * Real.pi)
  V := Real.pi * r_squared * cylinder_height
  V

theorem steel_bar_volume_is_2700 :
  original_volume_of_steel_bar 300 (Real.sqrt (18 / (2 * Real.pi))) 300 18 = 2700 := 
by
  sorry

end steel_bar_volume_is_2700_l633_633996


namespace medals_awarded_satisfy_condition_l633_633477

-- Define the total number of sprinters
def total_sprinters : Nat := 10

-- Define the number of American sprinters
def american_sprinters : Nat := 4

-- Define the number of medals
def medals : Nat := 3

-- Define a function to compute the permutations of selecting k items from n items
def permutations (n k : Nat) : Nat := ∏ i in Finset.range k, n - i

-- Define the number of ways to award the medals such that at most one American gets a medal
def ways_to_award_medals (total_sprinters american_sprinters medals : Nat) : Nat :=
  let non_american_sprinters := total_sprinters - american_sprinters
  -- Case 1: No Americans get a medal
  let case1 := permutations non_american_sprinters medals
  -- Case 2: Exactly one American gets a medal
  let case2 := american_sprinters * medals * permutations non_american_sprinters (medals - 1)
  case1 + case2

-- The final theorem to be proved
theorem medals_awarded_satisfy_condition :
  ways_to_award_medals total_sprinters american_sprinters medals = 480 :=
by
  -- Placeholder for the proof
  sorry

end medals_awarded_satisfy_condition_l633_633477


namespace minimum_shirts_to_save_money_l633_633805

-- Definitions for the costs
def EliteCost (n : ℕ) : ℕ := 30 + 8 * n
def OmegaCost (n : ℕ) : ℕ := 10 + 12 * n

-- Theorem to prove the given solution
theorem minimum_shirts_to_save_money : ∃ n : ℕ, 30 + 8 * n < 10 + 12 * n ∧ n = 6 :=
by {
  sorry
}

end minimum_shirts_to_save_money_l633_633805


namespace triangle_ratios_l633_633317

theorem triangle_ratios (A B C : Type) [metric_space A B C] 
  (a b c : ℝ) (angle_A angle_B angle_C : ℝ) 
  (h1 : angle_A + angle_B + angle_C = 180)
  (h2 : angle_A = 30) (h3 : angle_B = 60) (h4 : angle_C = 90) :
  a / b / c = 1 / real.sqrt 3 / 2 := 
sorry

end triangle_ratios_l633_633317


namespace voucher_code_count_l633_633144

def num_voucher_codes : Nat :=
  let first_char_choices := 3  -- 'V', 'X', 'P'
  let second_char_choices := 10 -- Digits 0 through 9
  let third_char_choices := 9 -- Must be different from the second digit
  first_char_choices * second_char_choices * third_char_choices

theorem voucher_code_count : num_voucher_codes = 270 := by
  rw [num_voucher_codes]
  sorry

end voucher_code_count_l633_633144


namespace sufficient_condition_a_gt_1_l633_633008

variable (a : ℝ)

theorem sufficient_condition_a_gt_1 (h : a > 1) : a^2 > 1 :=
by sorry

end sufficient_condition_a_gt_1_l633_633008


namespace dot_product_a_b_magnitude_a_plus_b_l633_633233

variables {α : Type*} [inner_product_space ℝ α]
variables (a b : α)
variables (angle_ab : real.angle) (alpha : ℝ)

noncomputable def angle_between_vectors (a b : α) : ℝ :=
real.angle.to_real (inner_product_geometry.angle a b)

-- Given conditions
axiom angle_ab_is_60_deg : angle_ab = real.pi / 3
axiom norm_a_is_2 : ∥a∥ = 2
axiom b_is_cos_sin : ∃ α : ℝ, b = ![(real.cos α), (real.sin α)]

-- Proving the dot product and magnitude of a + b
theorem dot_product_a_b : ⟪a, b⟫ = 1 :=
by sorry

theorem magnitude_a_plus_b : ∥a + b∥ = real.sqrt 7 :=
by sorry

end dot_product_a_b_magnitude_a_plus_b_l633_633233


namespace surprise_gift_combinations_l633_633155

theorem surprise_gift_combinations :
  let monday_choices := 2,
      tuesday_choices := 1,
      wednesday_choices := 1,
      thursday_choices := 4,
      friday_choices := 1
  in monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 8 :=
by
  -- here is where the proof will go
  sorry

end surprise_gift_combinations_l633_633155


namespace domain_of_g_l633_633898

noncomputable def g (t : ℝ) : ℝ := 1 / ((t - 2)^2 + (t + 2)^2 + 1)

theorem domain_of_g : ∀ t : ℝ, (t - 2)^2 + (t + 2)^2 + 1 > 0 := 
by {
  intro t,
  calc
    (t - 2)^2 + (t + 2)^2 + 1
      = (t^2 - 4t + 4) + (t^2 + 4t + 4) + 1 : by ring
  ... = 2t^2 + 9 : by ring
  ... > 0 : by {
    apply lt_add_of_le_of_pos,
    apply mul_nonneg,
    apply zero_le_two,
    apply pow_two_nonneg,
    norm_num,
    },
}

end domain_of_g_l633_633898


namespace find_fifth_month_sale_l633_633940

theorem find_fifth_month_sale (
  a1 a2 a3 a4 a6 : ℕ
) (avg_sales : ℕ)
  (h1 : a1 = 5420)
  (h2 : a2 = 5660)
  (h3 : a3 = 6200)
  (h4 : a4 = 6350)
  (h6 : a6 = 7070)
  (avg_condition : avg_sales = 6200)
  (total_condition : (a1 + a2 + a3 + a4 + a6 + (6500)) / 6 = avg_sales)
  : (∃ a5 : ℕ, a5 = 6500 ∧ (a1 + a2 + a3 + a4 + a5 + a6) / 6 = avg_sales) :=
by {
  sorry
}

end find_fifth_month_sale_l633_633940


namespace turnip_bag_weighs_l633_633087

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l633_633087


namespace find_number_l633_633003

theorem find_number : ∃ x : ℝ, (x / 6 * 12 = 10) ∧ x = 5 :=
by
 sorry

end find_number_l633_633003


namespace given_problem_l633_633299

theorem given_problem (x y : ℝ) (hx : x ≠ 0) (hx4 : x ≠ 4) (hy : y ≠ 0) (hy6 : y ≠ 6) :
  (2 / x + 3 / y = 1 / 2) ↔ (4 * y / (y - 6) = x) :=
sorry

end given_problem_l633_633299


namespace turnip_bag_weight_l633_633068

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l633_633068


namespace count_numbers_with_digit_five_l633_633278

theorem count_numbers_with_digit_five : 
  (finset.filter (λ n : ℕ, ∃ d : ℕ, d ∈ digits 10 n ∧ d = 5) (finset.range 701)).card = 133 := 
by 
  sorry

end count_numbers_with_digit_five_l633_633278


namespace circumradius_eq_l633_633759

noncomputable def circumradius (r : ℂ) (t1 t2 t3 : ℂ) : ℂ :=
  (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1))

theorem circumradius_eq (r t1 t2 t3 : ℂ) (h_pos_r : r ≠ 0) :
  circumradius r t1 t2 t3 = (2 * r ^ 4) / Complex.abs ((t1 + t2) * (t2 + t3) * (t3 + t1)) :=
  by sorry

end circumradius_eq_l633_633759


namespace arrange_numbers_l633_633965

noncomputable def a := (10^100)^10
noncomputable def b := 10^(10^10)
noncomputable def c := Nat.factorial 1000000
noncomputable def d := (Nat.factorial 100)^10

theorem arrange_numbers :
  a < d ∧ d < c ∧ c < b := 
sorry

end arrange_numbers_l633_633965


namespace probability_no_snow_l633_633843

theorem probability_no_snow {
  let p_snow : ℚ := 2/3,     -- Probability of snow on any one day
  let p_no_snow : ℚ := 1 - p_snow,  -- Probability of no snow on any one day
  let p_no_snow_5_days : ℚ := p_no_snow ^ 5  -- Probability of no snow on any of the five days
} : p_no_snow_5_days = 1 / 243 := by
  sorry

end probability_no_snow_l633_633843


namespace repeating_decimal_sum_l633_633616

theorem repeating_decimal_sum :
  (0.\overline{2} : ℝ) + (0.\overline{02} : ℝ) + (0.\overline{0002} : ℝ) = 2426/9999 :=
begin
  sorry
end

end repeating_decimal_sum_l633_633616


namespace propA_necessary_but_not_sufficient_l633_633662

variable {a : ℝ}

-- Proposition A: ∀ x ∈ ℝ, ax² + 2ax + 1 > 0
def propA (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0

-- Proposition B: 0 < a < 1
def propB (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem statement: Proposition A is necessary but not sufficient for Proposition B
theorem propA_necessary_but_not_sufficient (a : ℝ) :
  (propB a → propA a) ∧
  (propA a → propB a → False) :=
by
  sorry

end propA_necessary_but_not_sufficient_l633_633662


namespace impossible_tiling_l_tromino_l633_633362

theorem impossible_tiling_l_tromino (k : ℕ) : ¬ ∃ (tiling : (ℕ × ℕ) → ℕ), 
  (∀ x y, 0 ≤ x ∧ x < 5 → 0 ≤ y ∧ y < 7 → ∃ n, tiling (x, y) = k * n) ∧ 
  (∀ t, ttro t ∧ covers_tiling t tiling → 
    ∀ (x y), (x, y) ∈ cells_covered_by t → tiling (x, y) = k) :=
sorry

end impossible_tiling_l_tromino_l633_633362


namespace locus_equation_perimeter_greater_l633_633336

-- Define the conditions under which the problem is stated
def distance_to_x_axis (P : ℝ × ℝ) : ℝ := abs P.2
def distance_to_point (P : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- P is a point on the locus W if the distance to the x-axis is equal to the distance to (0, 1/2)
def on_locus (P : ℝ × ℝ) : Prop := 
  distance_to_x_axis P = distance_to_point P (0, 1/2)

-- Prove that the equation of W is y = x^2 + 1/4 given the conditions
theorem locus_equation (P : ℝ × ℝ) (h : on_locus P) : 
  P.2 = P.1^2 + 1/4 := 
sorry

-- Assume rectangle ABCD with three points on W
def point_on_w (P : ℝ × ℝ) : Prop := 
  P.2 = P.1^2 + 1/4

def points_form_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 ≠ B.1 ∧ B.1 ≠ C.1 ∧ C.1 ≠ D.1 ∧ D.1 ≠ A.1 ∧
  A.2 ≠ B.2 ∧ B.2 ≠ C.2 ∧ C.2 ≠ D.2 ∧ D.2 ≠ A.2

-- P1, P2, and P3 are three points on the locus W
def points_on_locus (A B C : ℝ × ℝ) : Prop := 
  point_on_w A ∧ point_on_w B ∧ point_on_w C

-- Prove the perimeter of rectangle ABCD with three points on W is greater than 3sqrt(3)
theorem perimeter_greater (A B C D : ℝ × ℝ) 
  (h1 : points_on_locus A B C) 
  (h2 : points_form_rectangle A B C D) : 
  2 * (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
       real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) > 
  3 * real.sqrt 3 := 
sorry

end locus_equation_perimeter_greater_l633_633336


namespace turnips_bag_l633_633046

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l633_633046


namespace max_k_for_good_numbers_sum_multiple_l633_633105

def isGoodNumber (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), digits ~ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
  digits.foldl (λ acc d => 10 * acc + d) 0 = n

theorem max_k_for_good_numbers_sum_multiple (S : ℕ) (k : ℕ) :
  (∀ n, n ∈ finset.range(999_999_999 + 1) → isGoodNumber n → ∃ m, 9 * m = n) →
  (∃ ns, (∀ n ∈ ns, isGoodNumber n) ∧ S = ns.sum ∧ ns.length = 9 ∧ S ≤ 8_888_888_889) →
  ∃ k_max, (∀ k, (9 * 10^k ≤ S) → k ≤ k_max) ∧ k_max = 8 :=
by
  sorry

end max_k_for_good_numbers_sum_multiple_l633_633105


namespace liters_per_bottle_l633_633803

-- Condition statements
def price_per_liter : ℕ := 1
def total_cost : ℕ := 12
def num_bottles : ℕ := 6

-- Desired result statement
theorem liters_per_bottle : (total_cost / price_per_liter) / num_bottles = 2 := by
  sorry

end liters_per_bottle_l633_633803


namespace max_and_min_values_l633_633201

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

theorem max_and_min_values :
  (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x ≤ 5) ∧ (∃ c ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f c = 5) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), 1 ≤ f x) ∧ (∃ c ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f c = 1) :=
begin
  sorry
end

end max_and_min_values_l633_633201


namespace repeating_decimal_sum_l633_633614

def repeating_decimal_to_fraction (d : ℕ) (n : ℕ) : ℚ := n / ((10^d) - 1)

theorem repeating_decimal_sum : 
  repeating_decimal_to_fraction 1 2 + repeating_decimal_to_fraction 2 2 + repeating_decimal_to_fraction 4 2 = 2474646 / 9999 := 
sorry

end repeating_decimal_sum_l633_633614


namespace planting_methods_l633_633039

-- Define the conditions
def n : ℕ := 6
def k : ℕ := 4

-- Define the chromatic polynomial for a cycle graph
def chromatic_poly (n k : ℕ) : ℕ := (k-1)^n + (-1)^n * (k-1)

-- State the proof problem
theorem planting_methods : chromatic_poly n k = 732 := by
  sorry

end planting_methods_l633_633039


namespace theta_range_l633_633639

theorem theta_range (θ : ℝ) (hθ : θ ∈ set.Icc 0 (2 * Real.pi)) :
  (∀ x : ℝ, x ∈ set.Icc 0 1 → 2 * x^2 * Real.sin θ - 4 * x * (1 - x) * Real.cos θ + 3 * (1 - x)^2 > 0) ↔
  θ ∈ set.Ioo (Real.pi / 6) Real.pi :=
sorry

end theta_range_l633_633639


namespace sum_of_integers_l633_633812

theorem sum_of_integers (x y : ℕ) (hxy_diff : x - y = 8) (hxy_prod : x * y = 240) (hx_gt_hy : x > y) : x + y = 32 := by
  sorry

end sum_of_integers_l633_633812


namespace symmetric_circle_equation_l633_633815

-- Define the given circle's equation
def givenCircle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 2016

-- Define the line of symmetry's equation
def symmetryLine (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric circle's equation
def symmetricCircle (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 2016

-- The proof statement
theorem symmetric_circle_equation : 
  ∀(x y : ℝ), 
    givenCircle x y → 
    (∃ x' y', symmetryLine x' y' ∧ symmetricCircle x' y') :=
by 
  intro x y hg,
  use [-1, -1],
  split;
  unfold symmetryLine symmetricCircle;
  try {linarith};
  sorry

end symmetric_circle_equation_l633_633815


namespace prime_square_remainders_l633_633580

theorem prime_square_remainders (p : ℕ) (hp : Nat.Prime p) (hgt : p > 5) : 
    {r | ∃ k : ℕ, p^2 = 180 * k + r}.finite ∧ 
    {r | ∃ k : ℕ, p^2 = 180 * k + r} = {1, 145} := 
by
  sorry

end prime_square_remainders_l633_633580


namespace donut_selection_count_l633_633782

theorem donut_selection_count : (∃ (g c p : ℕ), g + c + p = 5) ↔ finset.card (finset.filter (λ (k : fin 8), k.val ≤ 5) (finset.Ico 0 8)) = 21 := 
by
  sorry

end donut_selection_count_l633_633782


namespace find_f_2_l633_633677

def f (x : ℕ) : ℤ := sorry

axiom func_def : ∀ x : ℕ, f (x + 1) = x^2 - x

theorem find_f_2 : f 2 = 0 :=
by
  sorry

end find_f_2_l633_633677


namespace ordered_pair_and_sum_of_squares_l633_633627

theorem ordered_pair_and_sum_of_squares :
  ∃ x y : ℚ, 
    6 * x - 48 * y = 2 ∧ 
    3 * y - x = 4 ∧ 
    x ^ 2 + y ^ 2 = 442 / 25 :=
by
  sorry

end ordered_pair_and_sum_of_squares_l633_633627


namespace average_time_l633_633985

variable casey_time : ℝ
variable zendaya_multiplier : ℝ

theorem average_time (h1 : casey_time = 6) (h2 : zendaya_multiplier = 1/3) :
  (casey_time + (casey_time + zendaya_multiplier * casey_time)) / 2 = 7 := by
  sorry

end average_time_l633_633985


namespace marble_probability_l633_633019

theorem marble_probability :
  let total_marbles : ℕ := 20
  let red_marbles : ℕ := 12
  let blue_marbles : ℕ := 8
  let total_probability : ℚ := (6 * ((12 * 11 * 8 * 7) / (20 * 19 * 18 * 17))) in
  total_probability = (1232 / 4845) :=
by
  sorry

end marble_probability_l633_633019


namespace sum_of_integers_l633_633813

theorem sum_of_integers (x y : ℕ) (hxy_diff : x - y = 8) (hxy_prod : x * y = 240) (hx_gt_hy : x > y) : x + y = 32 := by
  sorry

end sum_of_integers_l633_633813


namespace area_triangle_ABC_l633_633170

variable {α d : ℝ}
variables {A B C D : Type} 

-- Conditions
variable [cyclic_quad (A B C D)]
variable (h_eq1 : AB = BC)
variable (h_eq2 : AB = AD + CD)
variable (h_alpha : ∠BAD = α)
variable (h_d : dist A C = d)

-- Statement
theorem area_triangle_ABC (A B C D : Type) [cyclic_quad (A B C D)]
  (h_eq1 : AB = BC) (h_eq2 : AB = AD + CD) (h_alpha : ∠BAD = α) (h_d : dist A C = d) :
  area ABC = 0.5 * d^2 * sin α :=
sorry

end area_triangle_ABC_l633_633170


namespace denis_and_oleg_probability_l633_633125

noncomputable def probability_denisolga_play_each_other (n : ℕ) (i j : ℕ) (h1 : n = 26) (h2 : i ≠ j) : ℚ :=
  let number_of_pairs := (n * (n - 1)) / 2
  in (n - 1) / number_of_pairs

theorem denis_and_oleg_probability :
  probability_denisolga_play_each_other 26 1 2 rfl dec_trivial = 1 / 13 :=
sorry

end denis_and_oleg_probability_l633_633125


namespace average_time_proof_l633_633984

-- Given conditions
def casey_time : ℝ := 6
def zendaya_additional_time_factor : ℝ := 1/3

-- Derived conditions from the problem
def zendaya_time : ℝ := casey_time + (zendaya_additional_time_factor * casey_time)
def combined_time : ℝ := casey_time + zendaya_time
def num_people : ℝ := 2

-- Statement to prove
theorem average_time_proof : (combined_time / num_people) = 7 := by
  sorry

end average_time_proof_l633_633984


namespace turnip_bag_weighs_l633_633089

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l633_633089


namespace smallest_n_for_negative_sum_l633_633705

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d, ∀ n, a (n + 1) = a n + d

theorem smallest_n_for_negative_sum (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 1 > 0) 
  (h3 : a 2022 + a 2023 > 0) 
  (h4 : a 2022 * a 2023 < 0) : 
  ∃ n : ℕ, (n ≥ 1) ∧ (sum (range n) a < 0) ∧ (∀ m : ℕ, (m < n → m ≥ 1 → sum (range m) a ≥ 0)) :=
begin
  use 4045,
  sorry
end

end smallest_n_for_negative_sum_l633_633705


namespace variance_of_numbers_l633_633440

theorem variance_of_numbers (a : ℝ) 
    (h_avg : (1 + 2 + 3 + 4 + a) / 5 = 3) : 
    variance [1, 2, 3, 4, a] = 2 := 
by 
  sorry

end variance_of_numbers_l633_633440


namespace pencils_in_drawer_l633_633481

theorem pencils_in_drawer (P : ℕ) 
  (h1 : 19 + 16 = 35)
  (h2 : P + 35 = 78) : 
  P = 43 := 
by
  sorry

end pencils_in_drawer_l633_633481


namespace hcf_36_84_l633_633622

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end hcf_36_84_l633_633622


namespace right_triangle_area_l633_633722

theorem right_triangle_area (A B C D : Type) [incidence_geometry A B C]
  (right_angle : ∠BAC = 90°)
  (median : is_midpoint D B C)
  (AD_length : AD = m)
  (angle_ratio : ∠BAD / ∠CAD = 1 / 2) :
  area (triangle A B C) = m^2 * √3 / 2 :=
sorry

end right_triangle_area_l633_633722


namespace construct_ngon_l633_633993

theorem construct_ngon (n : ℕ) (P : Fin n → Point) (α : Fin n → ℝ) :
  (∑ i in Finset.range n, α i) % 360 ≠ 0 →
  ∃ (A : Fin n → Point), 
    (∀ i, Triangle (A i) (A ((i + 1) % n)) (P i) ∧ 
      ∡ (A i) (P i) (A ((i + 1) % n)) = α i) :=
sorry

end construct_ngon_l633_633993


namespace car_actual_miles_traveled_l633_633520

def skips_digit (d : ℕ) (n : ℕ) : Prop :=
  let digits := n.digits 10
  ¬ digits.any (λ x => x = d)

def effective_odometer_reading (skip_d : ℕ) (displayed : ℕ) : ℕ :=
  (List.range (displayed + 1)).filter (skips_digit skip_d).length

theorem car_actual_miles_traveled (current_reading : ℕ) (result : ℕ) :
  current_reading = 3008 → 
  skips_digit 3 current_reading →
  result = effective_odometer_reading 3 current_reading →
  result = 2465 :=
  by
  intros h_reading h_skip h_result
  sorry

end car_actual_miles_traveled_l633_633520


namespace equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l633_633341

theorem equation_of_W (P : ℝ × ℝ) :
  let x := P.1 in let y := P.2 in
  |y| = real.sqrt (x^2 + (y - 1/2)^2) ↔ y = x^2 + 1/4 :=
by sorry

theorem rectangle_perimeter_greater_than_3sqrt3 {A B C D : ℝ × ℝ}
  (hA : A.2 = A.1^2 + 1/4) (hB : B.2 = B.1^2 + 1/4) (hC : C.2 = C.1^2 + 1/4)
  (hAB_perp_BC : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  2 * ((real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) + (real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2))) > 3 * real.sqrt 3 :=
by sorry

end equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l633_633341


namespace evaluate_expression_l633_633708

theorem evaluate_expression (x y : ℤ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^4 + 3 * x^2 - 2 * y + 2 * y^2) / 6 = 22 :=
by
  -- Conditions from the problem
  rw [h₁, h₂]
  -- Sorry is used to skip the proof
  sorry

end evaluate_expression_l633_633708


namespace tile_5x7_rectangle_with_L_trominos_l633_633360

theorem tile_5x7_rectangle_with_L_trominos :
  ∀ k : ℕ, ¬ (∃ (tile : ℕ → ℕ → ℕ), (∀ i j, tile (i+1) (j+1) = tile (i+3) (j+3)) ∧
    ∀ i j, (i < 5 ∧ j < 7) → (tile i j = k)) :=
by sorry

end tile_5x7_rectangle_with_L_trominos_l633_633360


namespace yanna_kept_apples_l633_633501

-- Define the given conditions
def initial_apples : ℕ := 60
def percentage_given_to_zenny : ℝ := 0.40
def percentage_given_to_andrea : ℝ := 0.25

-- Prove the main statement
theorem yanna_kept_apples : 
  let apples_given_to_zenny := (percentage_given_to_zenny * initial_apples)
  let apples_remaining_after_zenny := (initial_apples - apples_given_to_zenny)
  let apples_given_to_andrea := (percentage_given_to_andrea * apples_remaining_after_zenny)
  let apples_kept := (apples_remaining_after_zenny - apples_given_to_andrea)
  apples_kept = 27 :=
by
  sorry

end yanna_kept_apples_l633_633501


namespace angle_bisector_of_right_triangle_reflection_l633_633005

variables {α : Type*}
variables [has_ordered_sub α]

/-- Given a right-angled triangle ABC with ∠ABC = 90°, AB < BC, B' is the reflection of B over AC,
    and M is the midpoint of AC. Point D is chosen on the ray BM such that BD = AC.
    Prove that B'C is the angle bisector of ∠MB'D. -/
theorem angle_bisector_of_right_triangle_reflection
  (A B C B' D M : α)
  (hABC: ∠ABC = 90)
  (hAB_LT_BC : AB < BC)
  (hReflect : reflect_over AC B = B')
  (hMidpoint : midpoint A C = M)
  (hBD_AC : BD = AC) :
  angle_bisector B' C M D :=
sorry

end angle_bisector_of_right_triangle_reflection_l633_633005


namespace jovana_shells_l633_633749

def initial_weight : ℕ := 5
def added_weight : ℕ := 23
def total_weight : ℕ := 28

theorem jovana_shells :
  initial_weight + added_weight = total_weight :=
by
  sorry

end jovana_shells_l633_633749


namespace total_interest_received_l633_633041

variables (P_B P_C T_B T_C R : ℝ)
def SI (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_interest_received :
  P_B = 5000 → T_B = 2 → P_C = 3000 → T_C = 4 → R = 8 → 
  SI P_B R T_B + SI P_C R T_C = 1760 := by 
  sorry

end total_interest_received_l633_633041


namespace green_marbles_count_l633_633927

-- Conditions
def num_red_marbles : ℕ := 2

def probability_of_two_reds (G : ℕ) : ℝ :=
  (num_red_marbles / (num_red_marbles + G)) * ((num_red_marbles - 1) / (num_red_marbles + G - 1))

-- Problem Statement
theorem green_marbles_count (G : ℕ) (h : probability_of_two_reds G = 0.1) : G = 3 :=
sorry

end green_marbles_count_l633_633927


namespace complementary_union_is_certain_event_l633_633305

variables {Ω : Type*}
variable (A B : Set Ω)
def is_mutually_exclusive (A B : Set Ω) : Prop := A ∩ B = ∅
def is_certain_event (S : Set Ω) : Prop := S = set.univ
def complement (S : Set Ω) : Set Ω := set.univ \ S

-- The math proof statement
theorem complementary_union_is_certain_event
  (A B : Set Ω)
  (h1 : is_mutually_exclusive A B) :
  is_certain_event (complement A ∪ complement B) :=
begin
  -- Proof goes here
  sorry
end

end complementary_union_is_certain_event_l633_633305


namespace find_k_l633_633254

theorem find_k (k : ℝ) (h₀ : k ≠ 0) :
  (∃ A B : ℝ × ℝ, ((A.1 - 3)^2 + (A.2 - 2)^2 = 4) ∧ (A.2 = k * A.1 + 3) ∧
                   ((B.1 - 3)^2 + (B.2 - 2)^2 = 4) ∧ (B.2 = k * B.1 + 3) ∧
                   (dist A B = 2 * sqrt 3)) →
  (k = 0 ∨ k = -3 / 4) :=
sorry

end find_k_l633_633254


namespace find_point_B_l633_633114

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def plane (a b c d : ℝ) (p : Point3D) : Prop :=
  a * p.x + b * p.y + c * p.z = d

def collinear (A B C : Point3D) : Prop :=
  ∃ (λ : ℝ), (λ * (C.x - B.x) = B.x - A.x) ∧ (λ * (C.y - B.y) = B.y - A.y) ∧ (λ * (C.z - B.z) = B.z - A.z)

noncomputable def reflection_point (A : Point3D) (B : Point3D) : Point3D :=
  { x := 2 * B.x - A.x,
    y := 2 * B.y - A.y,
    z := 2 * B.z - A.z }

noncomputable def intersection_point (A : Point3D) (v : Point3D) (t : ℝ) := 
  { x := A.x + v.x * t,
    y := A.y + v.y * t,
    z := A.z + v.z * t }

axiom given_points_and_plane :
  let A := Point3D.mk (-2) 8 12 in
  let C := Point3D.mk 4 2 10 in
  let B := Point3D.mk (20 / 3) (-94 / 3) (-122 / 3) in
  let normal_vec := Point3D.mk 2 (-1) 1 in
  ∃ t : ℝ, 
  let P := intersection_point A normal_vec t in 
  P = Point3D.mk (2/3) (20/3) (40/3) ∧ 
  let D := reflection_point A P in
  collinear D B C ∧ 
  plane 2 (-1) 1 8 B

theorem find_point_B :
  let B := Point3D.mk (20 / 3) (-94 / 3) (-122 / 3) in
  ∃ A C : Point3D, 
    let A := Point3D.mk (-2) 8 12 in
    let C := Point3D.mk 4 2 10 in
    let P := Point3D.mk (2 / 3) (20 / 3) (40 / 3) in
    let D := reflection_point A P in
    collinear D B C ∧ 
    plane 2 (-1) 1 8 B := 
sorry

end find_point_B_l633_633114


namespace quadratic_discriminant_irrational_l633_633944

theorem quadratic_discriminant_irrational
  (a b c : ℝ)
  (h_b_rational : b ∈ ℚ)
  (h_no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (h_irrational_one_of_c_fc : (c ∈ ℚ ∧ (a * c^2 + b * c + c) ∉ ℚ) ∨ (c ∉ ℚ ∧ (a * c^2 + b * c + c) ∈ ℚ)) :
  (b^2 - 4 * a * c) ∉ ℚ :=
by
  sorry

end quadratic_discriminant_irrational_l633_633944


namespace find_m_eq_neg2_l633_633681

noncomputable def find_m (m : ℝ) : Prop :=
  let p : ℕ → ℝ := λ x, x^2 + 2*x - 3 in
  let l1 : ℕ → ℝ := λ x, -x + m in
  let A : (ℝ × ℝ) := (x1, p x1) in
  let C : (ℝ × ℝ) := (x2, p x2) in
  let l2 : ℕ → ℝ := λ x, -x - m in
  let B : (ℝ × ℝ) := (x3, p x3) in
  let D : (ℝ × ℝ) := (x4, p x4) in
  let AC : ℝ := dist A C in
  let BD : ℝ := dist B D in
  (AC * BD = 26) → (m = -2)

theorem find_m_eq_neg2 (m : ℝ) (h : find_m m) :
  m = -2 := 
  by
    sorry

end find_m_eq_neg2_l633_633681


namespace a_n_geometric_sequence_and_formula_b_n_min_value_l633_633647

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {T : ℕ → ℕ}

-- Conditions for the sequence a_n
axiom a_sequence_condition (n : ℕ) (hn : n > 0) : S n + n = 2 * a n

-- Definitions of sequences b_n and T_n
noncomputable def b (n : ℕ) := n * (a n) + n
noncomputable def T (n : ℕ) := (n - 1) * 2^(n+1) + 2

-- Statements to prove
theorem a_n_geometric_sequence_and_formula : ∀ n : ℕ, n > 0 → (a n + 1 = 2 ^ n) :=
sorry

theorem b_n_min_value (n : ℕ) : ( ∀ k : ℕ, k ≥ 11 → (T k - 2) / k ≤ 2018 ) ∧ 
  ( ∀ m : ℕ, m < 11 → (T m - 2) / m > 2018 ) :=
sorry

end a_n_geometric_sequence_and_formula_b_n_min_value_l633_633647


namespace common_ratio_l633_633701

-- Given conditions
variables {a : ℕ → ℝ} (q : ℝ)
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition (a : ℕ → ℝ) : Prop :=
  2 * a 4 = a 6 - a 5

-- Proof goal
theorem common_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → condition a → (q = -1 ∨ q = 2) :=
by
  intros h_sequence h_condition
  have h_q_eqn : q ^ 2 - q - 2 = 0, sorry
  have h_q_solutions : q = -1 ∨ q = 2, sorry
  exact h_q_solutions

end common_ratio_l633_633701


namespace simplify_sqrt_subtraction_l633_633422

theorem simplify_sqrt_subtraction : sqrt 20 - sqrt 5 = sqrt 5 :=
by
  sorry -- Proof will be provided here

end simplify_sqrt_subtraction_l633_633422


namespace max_value_of_f_on_interval_min_value_of_f_on_interval_l633_633203

-- Define the function f(x) = x^3 - 3x^2 + 5
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

-- Define the interval
def interval := set.Icc (-1:ℝ) 1

-- State that the maximum value on the interval is 5
theorem max_value_of_f_on_interval : 
  is_lub (set.image f interval) 5 := 
sorry

-- State that the minimum value on the interval is 1
theorem min_value_of_f_on_interval : 
  is_glb (set.image f interval) 1 := 
sorry

end max_value_of_f_on_interval_min_value_of_f_on_interval_l633_633203


namespace find_mass_of_elliptical_plate_l633_633196

noncomputable def mass_of_elliptical_plate (a b λ : ℝ) (h : a > b) : ℝ :=
  (4 / 3) * λ * a^2 * b

theorem find_mass_of_elliptical_plate (a b λ : ℝ) (h : a > b) :
  let δ := λ (x : ℝ), λ * |x|
  let m := mass_of_elliptical_plate a b λ h in
  m = (4 / 3) * λ * a^2 * b :=
by
  -- Proof steps are omitted
  sorry

end find_mass_of_elliptical_plate_l633_633196


namespace inverse_expression_l633_633978

theorem inverse_expression : (5⁻¹ - 4⁻¹)⁻¹ = -20 := 
sorry

end inverse_expression_l633_633978


namespace chef_pillsbury_requirements_l633_633589

section

variables {F : Type} [Field F]

/-- Given ratios:
    - 7 eggs for every 2 cups of flour.
    - 5 cups of milk for every 14 eggs.
    - 3 tablespoons of sugar for every 1 cup of milk.
    Prove that for 24 cups of flour, the requirements are:
    - 84 eggs
    - 30 cups of milk
    - 90 tablespoons of sugar. -/
theorem chef_pillsbury_requirements (x y z : ℕ) (flour: ℕ)
  (h1: 7 * flour / 2 = 84)
  (h2: 5 * 84 / 14 = 30)
  (h3: 3 * 30 = 90): 
  (x = 84) 
  ∧ (y = 30)
  ∧ (z = 90) :=
by {
  split,
  case left 
  {
    exact h1,
  },
  split,
  case left
  {
    exact h2,
  },
  case right 
  {
    exact h3,
  },
}

end

end chef_pillsbury_requirements_l633_633589


namespace cubic_root_expression_l633_633383

theorem cubic_root_expression (u v w : ℂ) (huvwx : u * v * w ≠ 0)
  (h1 : u^3 - 6 * u^2 + 11 * u - 6 = 0)
  (h2 : v^3 - 6 * v^2 + 11 * v - 6 = 0)
  (h3 : w^3 - 6 * w^2 + 11 * w - 6 = 0) :
  (u * v / w) + (v * w / u) + (w * u / v) = 49 / 6 :=
sorry

end cubic_root_expression_l633_633383


namespace turnip_bag_weight_l633_633056

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l633_633056


namespace denis_oleg_probability_l633_633131

theorem denis_oleg_probability :
  let n := 26 in
  let players := {denis, oleg} in
  let total_matches := n - 1 in
  let total_pairs := n * (n - 1) / 2 in
  ∃ (P : ℚ), P = 1 / 13 ∧
  ∀ (i : ℕ), i ∈ fin total_matches →
  let match_pairs := (n - i) * (n - i - 1) / 2 in
  players ⊆ fin match_pairs → P = 1 / 13 := 
sorry

end denis_oleg_probability_l633_633131


namespace minimize_f_l633_633180

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l633_633180


namespace find_z_squared_l633_633710

def z : ℂ := 2 + 5 * complex.I

theorem find_z_squared : z^2 = -21 + 20 * complex.I :=
by
  -- Proof to be filled
  sorry

end find_z_squared_l633_633710


namespace no_snow_probability_l633_633850

theorem no_snow_probability (p_snow : ℚ) (h : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  (p_no_snow ^ 5 = 1 / 243) :=
by
  sorry

end no_snow_probability_l633_633850


namespace coefficient_of_x2_term_in_expansion_l633_633602

noncomputable def coefficient_x2_term : ℤ :=
  let binomial_coeff := fun (n k : ℕ) => nat.choose n k
  let term1 := (2 : ℤ)
  let term2 := (-1 : ℤ) / (x : ℚ)
  let expansion_term (k : ℕ) :=
    binomial_coeff 4 k * (1 : ℤ)^(4 - k) * (-2 : ℤ)^(k)
  let coefficient_expansion_x2 := expansion_term 2 * 24
  let coefficient_expansion_x3 := expansion_term 3 * (-32)
  coefficient_expansion_x2 + coefficient_expansion_x3

theorem coefficient_of_x2_term_in_expansion :
  coefficient_x2_term = 80 := by
  sorry

end coefficient_of_x2_term_in_expansion_l633_633602


namespace percentage_difference_l633_633107

theorem percentage_difference (y : ℝ) (h : y ≠ 0) (x z : ℝ) (hx : x = 5 * y) (hz : z = 1.20 * y) :
  ((z - y) / x * 100) = 4 :=
by
  rw [hz, hx]
  simp
  sorry

end percentage_difference_l633_633107


namespace turnip_weights_are_13_or_16_l633_633098

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l633_633098


namespace parallel_AB_CH_l633_633820

-- Definitions of the geometric elements
variables {A B C D E F G H : Type*}

def incircle_touches (ω : Type*) (ABC : triangle) (D E F : Type*) : Prop :=
  touches ω [BC B C, CA C A, AB A B] [D E F]

def is_diameter (ω : Type*) (F G : Type*) : Prop :=
  diameter (ω) FG

def lines_intersect (E G F D H : Type*) : Prop :=
  intersect (line_through E G) (line_through F D) H

theorem parallel_AB_CH
  (ω : Type*)
  (ABC : triangle)
  (D E F G H : Type*)
  (h1 : incircle_touches ω ABC D E F)
  (h2 : is_diameter ω F G)
  (h3 : lines_intersect E G F D H) :
  parallel (line_through A B) (line_through C H) := sorry

end parallel_AB_CH_l633_633820


namespace player_B_secures_victory_l633_633460

theorem player_B_secures_victory :
  let initial_number := 2015 in
  ∀ k : ℕ, (k % 2 = 0 → ∃ d, d ∣ k ∧ k - d = 0) ∧
           (k % 2 = 1 → ∃ d, d ∣ k ∧ k - d ≠ 0) →
  (∃ B_turns : ℕ, B_turns % 2 = 0 ∧ initial_number - B_turns = 0) →
  (∃ A_turns : ℕ, A_turns % 2 = 1 ∧ initial_number - A_turns ≠ 0) →
  ∀ turns : ℕ, (turns % 2 = 1 → ∃ x, x % 2 = 1 ∧ x - turns = 0 ) ∧
              (turns % 2 = 0 → ∃ y, y % 2 = 0 ∧ y - turns ≠ 0) →
  Player_B_secures_victory
by
  sorry

end player_B_secures_victory_l633_633460


namespace find_xy_l633_633538

theorem find_xy (x y : ℝ) (h1 : x / 7 = 5 / 14) (h2 : x / 7 + y = 10) :
  x = 2.5 ∧ y ≈ 9.64 :=
by
  sorry

end find_xy_l633_633538


namespace turnip_weights_are_13_or_16_l633_633095

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight (l : List ℕ) : ℕ := l.sum

def valid_turnip_weights (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  (∑ x in bag_weights, x) = 106 ∧
  (106 - T) % 3 = 0

theorem turnip_weights_are_13_or_16 :
  ∀ (T : ℕ), valid_turnip_weights T → T = 13 ∨ T = 16 :=
by
  intros T h,
  sorry

end turnip_weights_are_13_or_16_l633_633095


namespace probability_Denis_Oleg_play_l633_633140

theorem probability_Denis_Oleg_play (n : ℕ) (h : n = 26) :
  let C := λ (n : ℕ), n * (n - 1) / 2 in
  (n - 1 : ℚ) / C n = 1 / 13 :=
by 
  sorry

end probability_Denis_Oleg_play_l633_633140


namespace digit_in_thousandths_place_l633_633897

theorem digit_in_thousandths_place : (3 / 16 : ℚ) = 0.1875 :=
by sorry

end digit_in_thousandths_place_l633_633897


namespace no_snow_five_days_l633_633856

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633856


namespace turnip_bag_weight_l633_633055

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l633_633055


namespace harmonic_mean_of_squares_l633_633438
noncomputable def harmonic_mean (a b c : ℕ) : ℝ :=
  3 / (1 / (Real.sqrt a) + 1 / (Real.sqrt b) + 1 / (Real.sqrt c))

theorem harmonic_mean_of_squares : 
  ∃ (H : ℝ), harmonic_mean 25 64 144 = H ∧ H ≈ 7.35 :=
by
  exists harmonic_mean 25 64 144
  split
  sorry

end harmonic_mean_of_squares_l633_633438


namespace sin_product_l633_633357

-- Assume all the given conditions
variables (K L M P O Q R : Type) [triangle K L M] [Median K P L M] [Circumcenter O K L M] [Incenter Q K L M]
          (OR PR QR KR : ℝ) (α β : ℝ)

-- Given conditions
axiom angle_sum_property : α + β + π / 3 = π
axiom given_angle : angle L K M = π / 3
axiom given_relation : OR / PR = sqrt 14 * (QR / KR)

-- Definition of the target theorem
theorem sin_product (h1 : α + β = 2 * π / 3) 
                    (h2 : sin (α + β) = sin (2 * π / 3)) :
                    sin α * sin β = 5 / 8 := 
sorry

end sin_product_l633_633357


namespace integral_sum_l633_633160

-- Define the integrands as piecewise functions.
def f (x : ℝ) : ℝ := sqrt (1 - x^2)
def g (x : ℝ) : ℝ := exp (abs x)

-- Calculate the definite integrals.
theorem integral_sum :
  ∫ x in -1..1, f x + g x = (π / 2) + 2 * exp 1 - 2 :=
by
  -- Proof of the theorem
  sorry  -- Proof placeholder

end integral_sum_l633_633160


namespace denis_oleg_probability_l633_633129

theorem denis_oleg_probability :
  let n := 26 in
  let players := {denis, oleg} in
  let total_matches := n - 1 in
  let total_pairs := n * (n - 1) / 2 in
  ∃ (P : ℚ), P = 1 / 13 ∧
  ∀ (i : ℕ), i ∈ fin total_matches →
  let match_pairs := (n - i) * (n - i - 1) / 2 in
  players ⊆ fin match_pairs → P = 1 / 13 := 
sorry

end denis_oleg_probability_l633_633129


namespace insurance_compensation_zero_l633_633823

noncomputable def insured_amount : ℝ := 500000
noncomputable def deductible : ℝ := 0.01
noncomputable def actual_damage : ℝ := 4000

theorem insurance_compensation_zero :
  actual_damage < insured_amount * deductible → 0 = 0 := by
sorry

end insurance_compensation_zero_l633_633823


namespace average_time_proof_l633_633982

-- Given conditions
def casey_time : ℝ := 6
def zendaya_additional_time_factor : ℝ := 1/3

-- Derived conditions from the problem
def zendaya_time : ℝ := casey_time + (zendaya_additional_time_factor * casey_time)
def combined_time : ℝ := casey_time + zendaya_time
def num_people : ℝ := 2

-- Statement to prove
theorem average_time_proof : (combined_time / num_people) = 7 := by
  sorry

end average_time_proof_l633_633982


namespace length_of_train_l633_633954

theorem length_of_train (V L : ℝ) (h1 : L = V * 18) (h2 : L + 250 = V * 33) : L = 300 :=
by
  sorry

end length_of_train_l633_633954


namespace shaded_fraction_is_one_third_l633_633115

def geometric_series_sum (a r : ℝ) (h : r < 1) : ℝ := 
a / (1 - r)

theorem shaded_fraction_is_one_third : 
let S := ∑' (n : ℕ), (1 / 4 : ℝ) ^ (n + 1) in
S = 1 / 3 :=
by
  have r_pos : (1 / 4 : ℝ) < 1 := by norm_num
  have a := (1 / 4 : ℝ)
  have S := geometric_series_sum a (1 / 4) r_pos
  have H : S = 1 / 3 := by norm_num
  exact H

end shaded_fraction_is_one_third_l633_633115


namespace measure_of_angle_A_length_of_side_c_l633_633318

variable {A B C : ℝ}
variable {a b c : ℝ} -- sides of the triangle, opposite to angles A, B, and C respectively

-- Given condition for part 1
axiom cond1 : c / 2 = b - a * Real.cos C

-- Given values and angle
axiom given_a : a = Real.sqrt 15
axiom given_b : b = 4
axiom given_A : Real.cos A = 1 / 2

-- Prove measure of angle A is 60 degrees
theorem measure_of_angle_A : degrees A = 60 := by
  sorry

-- Prove length of side c
theorem length_of_side_c : c = 2 + Real.sqrt 3 ∨ c = 2 - Real.sqrt 3 := by
  sorry

end measure_of_angle_A_length_of_side_c_l633_633318


namespace similar_triangle_PQR_ABC_l633_633934

open EuclideanGeometry

variables {A B C M N K P Q R : Point}

-- Defining the conditions given
variables (hM : PointOfTangencyIncircle M B C)
variables (hN : PointOfTangencyIncircle N C A)
variables (hK : PointOfTangencyIncircle K A B)

variables (hP : FootOfPerpendicular P K N)
variables (hQ : FootOfPerpendicular Q N M)
variables (hR : FootOfPerpendicular R M K)

theorem similar_triangle_PQR_ABC :
  Similar (Triangle.mk P Q R) (Triangle.mk A B C) :=
by
  sorry

end similar_triangle_PQR_ABC_l633_633934


namespace log_sequence_eval_l633_633646

open Real

variable {a : ℕ → ℝ}

-- Definitions
def sequence_condition (a : ℕ → ℝ) := ∀ n, log 3 (a n) + 1 = log 3 (a (n + 1))
def specific_terms_sum (a : ℕ → ℝ) := a 2 + a 4 + a 6 = 9

-- Theorem statement
theorem log_sequence_eval (a : ℕ → ℝ) 
  (h1 : sequence_condition a) 
  (h2 : specific_terms_sum a) : 
  log (1 / 3) (a 5 + a 7 + a 9) = -5 := by
    sorry

end log_sequence_eval_l633_633646


namespace probability_at_least_one_head_l633_633582

theorem probability_at_least_one_head :
  (1 - (1 / 4) ^ 5) = 1023 / 1024 := 
begin
  sorry
end

end probability_at_least_one_head_l633_633582


namespace max_value_of_function_l633_633454

noncomputable def max_value (x : ℝ) : ℝ := 3 * Real.sin x + 2

theorem max_value_of_function : 
  ∀ x : ℝ, (- (Real.pi / 2)) ≤ x ∧ x ≤ 0 → max_value x ≤ 2 :=
sorry

end max_value_of_function_l633_633454


namespace numbers_contain_digit_five_l633_633289

theorem numbers_contain_digit_five :
  (finset.range 701).filter (λ n, ∃ d ∈ (n.digits 10), d = 5).card = 214 := 
sorry

end numbers_contain_digit_five_l633_633289


namespace blood_expiry_date_l633_633957

noncomputable def sec_per_day : ℕ := 60 * 60 * 24
noncomputable def blood_expiry_seconds : ℕ := Nat.fact 8
noncomputable def donation_date : String := "January 1"

theorem blood_expiry_date : (blood_expiry_seconds / sec_per_day) < 1 → donation_date = "January 1" :=
by
  sorry

end blood_expiry_date_l633_633957


namespace jori_water_left_l633_633748

theorem jori_water_left (initial used : ℚ) (h1 : initial = 3) (h2 : used = 4 / 3) :
  initial - used = 5 / 3 :=
by
  sorry

end jori_water_left_l633_633748


namespace integer_square_mod_4_l633_633789

theorem integer_square_mod_4 (N : ℤ) : (N^2 % 4 = 0) ∨ (N^2 % 4 = 1) :=
by sorry

end integer_square_mod_4_l633_633789


namespace emily_page_production_difference_l633_633406

variables (p h : ℕ)

def first_day_pages (p h : ℕ) : ℕ := p * h
def second_day_pages (p h : ℕ) : ℕ := (p - 3) * (h + 3)
def page_difference (p h : ℕ) : ℕ := second_day_pages p h - first_day_pages p h

theorem emily_page_production_difference (h : ℕ) (p_eq_3h : p = 3 * h) :
  page_difference p h = 6 * h - 9 :=
by sorry

end emily_page_production_difference_l633_633406


namespace ratio_of_returned_to_borrowed_l633_633797

variables (B : ℝ) (returned_to_date : ℝ) (future_debt : ℝ)
 
-- Condition 1: Shara returned $10 per month for 6 months.
def monthly_return_rate := 10
def months_returned := 6
def initial_returns := months_returned * monthly_return_rate

-- Condition 2: She will owe $20, 4 months from now.
def future_months := 4
def future_debt := 20
def future_returns := future_months * monthly_return_rate

-- The total return amount including future returns:
def total_returns := initial_returns + future_returns

-- The total amount borrowed B
def total_borrowed := total_returns + future_debt

-- Ratio of returned amount to borrowed amount:
def returned_ratio := initial_returns / total_borrowed

theorem ratio_of_returned_to_borrowed : returned_ratio = 1 / 2 :=
by 
  sorry

end ratio_of_returned_to_borrowed_l633_633797


namespace probability_of_two_red_two_blue_l633_633023

-- Definitions for the conditions
def total_red_marbles : ℕ := 12
def total_blue_marbles : ℕ := 8
def total_marbles : ℕ := total_red_marbles + total_blue_marbles
def num_selected_marbles : ℕ := 4
def num_red_selected : ℕ := 2
def num_blue_selected : ℕ := 2

-- Definition for binomial coefficient (combinations)
def C (n k : ℕ) : ℕ := n.choose k

-- Probability calculation
def probability_two_red_two_blue :=
  (C total_red_marbles num_red_selected * C total_blue_marbles num_blue_selected : ℚ) / C total_marbles num_selected_marbles

-- The theorem statement
theorem probability_of_two_red_two_blue :
  probability_two_red_two_blue = 1848 / 4845 :=
by
  sorry

end probability_of_two_red_two_blue_l633_633023


namespace solve_for_y_l633_633425

theorem solve_for_y (y : ℝ) (h : (8 * y^2 + 50 * y + 3) / (4 * y + 21) = 2 * y + 1) : y = 4.5 :=
by
  -- Proof goes here
  sorry

end solve_for_y_l633_633425


namespace sum_of_numbers_l633_633507

/-- Given three numbers in the ratio 1:2:5, with the sum of their squares being 4320,
prove that the sum of the numbers is 96. -/

theorem sum_of_numbers (x : ℝ) (h1 : (x:ℝ) = x) (h2 : 2 * x = 2 * x) (h3 : 5 * x = 5 * x) 
  (h4 : x^2 + (2 * x)^2 + (5 * x)^2 = 4320) :
  x + 2 * x + 5 * x = 96 := 
sorry

end sum_of_numbers_l633_633507


namespace lower_right_is_3_l633_633885

-- Define the initial grid and constraints for the 5x5 Sudoku problem
structure Grid (n : Nat) :=
( cells : Fin n → Fin n → Option Nat ) -- A grid with possible empty cells

-- Given the initial layout and constraints
def initial_grid : Grid 5 := {
  cells := λ r c,
    match r, c with 
    | 0, 0 => some 1
    | 0, 1 => some 2
    | 1, 2 => some 3
    | 1, 4 => some 1
    | 3, 4 => some 2
    | _, _ => none
    end
}

-- Predicate stating that each number from 1 to n appears exactly once in each row and column
def valid_sudoku (g : Grid 5) : Prop :=
  (∀ i : Fin 5, ∀ x : Nat, x ∈ [1, 2, 3, 4, 5] → (∃! j : Fin 5, g.cells i j = some x)) ∧
  (∀ j : Fin 5, ∀ x : Nat, x ∈ [1, 2, 3, 4, 5] → (∃! i : Fin 5, g.cells i j = some x))
  
-- Conclude the number at the lower right-hand corner of the grid is 3
theorem lower_right_is_3 : (∀ g : Grid 5, valid_sudoku g → g.cells (Fin.mk 4 (by decide)) (Fin.mk 4 (by decide)) = some 3) :=
by
  sorry

end lower_right_is_3_l633_633885


namespace numbers_contain_digit_five_l633_633290

theorem numbers_contain_digit_five :
  (finset.range 701).filter (λ n, ∃ d ∈ (n.digits 10), d = 5).card = 214 := 
sorry

end numbers_contain_digit_five_l633_633290


namespace dogs_at_center_l633_633156

theorem dogs_at_center
  (fetch roll_over play_dead : Finset ℕ)
  (dogs_fetch := fetch.card = 55)
  (dogs_roll_over := roll_over.card = 32)
  (dogs_play_dead := play_dead.card = 40)
  (fr := (fetch ∩ roll_over).card = 20)
  (fp := (fetch ∩ play_dead).card = 18)
  (rp := (roll_over ∩ play_dead).card = 15)
  (frp := (fetch ∩ roll_over ∩ play_dead).card = 12)
  (none := ∀ (x : ℕ), x ∉ fetch ∧ x ∉ roll_over ∧ x ∉ play_dead → x = 14) :
  fetch.card + roll_over.card + play_dead.card - (fr + fp + rp) + frp + none = 100 :=
sorry

end dogs_at_center_l633_633156


namespace intersection_of_convex_is_convex_l633_633788

-- Define a convex set
def is_convex (S : set ℝ) : Prop :=
  ∀ {x y : ℝ} (hx : x ∈ S) (hy : y ∈ S) (t : ℝ), 0 ≤ t → t ≤ 1 → (t * x + (1 - t) * y) ∈ S

-- Define the intersection of sets
def intersection (sets : set (set ℝ)) : set ℝ :=
  {x | ∀ S ∈ sets, x ∈ S}

-- Prove that the intersection of convex sets is a convex set
theorem intersection_of_convex_is_convex (φ : set (set ℝ)) (h_convex : ∀ S ∈ φ, is_convex S) :
  is_convex (intersection φ) := sorry

end intersection_of_convex_is_convex_l633_633788


namespace denis_oleg_probability_l633_633132

theorem denis_oleg_probability :
  let n := 26 in
  let players := {denis, oleg} in
  let total_matches := n - 1 in
  let total_pairs := n * (n - 1) / 2 in
  ∃ (P : ℚ), P = 1 / 13 ∧
  ∀ (i : ℕ), i ∈ fin total_matches →
  let match_pairs := (n - i) * (n - i - 1) / 2 in
  players ⊆ fin match_pairs → P = 1 / 13 := 
sorry

end denis_oleg_probability_l633_633132


namespace min_dot_product_l633_633327

noncomputable def A : ℝ × ℝ := (1, 0)
noncomputable def B : ℝ × ℝ := (0, 1)
noncomputable def P (x : ℝ) : ℝ × ℝ := (x, Real.exp x)

def vector_op (x : ℝ) : ℝ × ℝ := P x
def vector_ab : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2

theorem min_dot_product : 
  ∃ x : ℝ, ∀ y : ℝ, dot_product (vector_op y) vector_ab ≥ dot_product (vector_op x) vector_ab ∧ dot_product (vector_op x) vector_ab = 1 :=
begin
  sorry
end

end min_dot_product_l633_633327


namespace prime_squares_mod_180_l633_633574

theorem prime_squares_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  ∃ r ∈ {1, 49}, ∀ q ∈ {1, 49}, (q ≡ p^2 [MOD 180]) → q = r :=
by
  sorry

end prime_squares_mod_180_l633_633574


namespace f_is_polynomial_with_integer_coefficients_f_cannot_be_factored_l633_633377

noncomputable def omega : ℂ := complex.cos (real.pi / 5) + complex.sin (real.pi / 5) * complex.I

def f (x : ℂ) : ℂ := 
  (x - omega) * (x - omega^3) * (x - omega^7) * (x - omega^9)

theorem f_is_polynomial_with_integer_coefficients : ∃ (p : polynomial ℤ), 
  ∀ x, f x = p.eval x :=
sorry 

theorem f_cannot_be_factored : ¬ ∃ (p q : polynomial ℤ), 
  degree p ≥ 1 ∧ degree q ≥ 1 ∧ f = p * q :=
sorry

end f_is_polynomial_with_integer_coefficients_f_cannot_be_factored_l633_633377


namespace locus_equation_rectangle_perimeter_greater_l633_633333

open Real

theorem locus_equation (P : ℝ × ℝ) : 
  (abs P.2 = sqrt (P.1 ^ 2 + (P.2 - 1 / 2) ^ 2)) → (P.2 = P.1 ^ 2 + 1 / 4) :=
by
  intro h
  sorry

theorem rectangle_perimeter_greater (A B C D : ℝ × ℝ) :
  (A.2 = A.1 ^ 2 + 1 / 4) ∧ 
  (B.2 = B.1 ^ 2 + 1 / 4) ∧ 
  (C.2 = C.1 ^ 2 + 1 / 4) ∧ 
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) → 
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
by
  intro h
  sorry

end locus_equation_rectangle_perimeter_greater_l633_633333


namespace arithmetic_sequence_geometric_subsequence_l633_633648

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℤ) (a1 a3 a4 : ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 2)
  (h3 : a1 = a 1)
  (h4 : a3 = a 3)
  (h5 : a4 = a 4)
  (h6 : a3^2 = a1 * a4) :
  a 6 = 2 := 
by
  sorry

end arithmetic_sequence_geometric_subsequence_l633_633648


namespace inverse_matrix_l633_633623

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 7], ![-1, -1]]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![-(1/3 : ℚ), -(7/3 : ℚ)], ![1/3, 4/3]]

theorem inverse_matrix : A.det ≠ 0 → A⁻¹ = A_inv := by
  sorry

end inverse_matrix_l633_633623


namespace prime_square_remainders_l633_633581

theorem prime_square_remainders (p : ℕ) (hp : Nat.Prime p) (hgt : p > 5) : 
    {r | ∃ k : ℕ, p^2 = 180 * k + r}.finite ∧ 
    {r | ∃ k : ℕ, p^2 = 180 * k + r} = {1, 145} := 
by
  sorry

end prime_square_remainders_l633_633581


namespace count_5_in_range_1_to_700_l633_633281

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  n.digits 10 |>.contains d

def count_numbers_with_digit (d : ℕ) (m : ℕ) : ℕ :=
  (List.range' 1 m) |>.filter (contains_digit d) |>.length

theorem count_5_in_range_1_to_700 : count_numbers_with_digit 5 700 = 214 := by
  sorry

end count_5_in_range_1_to_700_l633_633281


namespace length_AQ_l633_633611

noncomputable def side_length_XYZ := 8
noncomputable def XC := 2
noncomputable def CY := 3

theorem length_AQ : ∃ (AQ : ℝ), AQ = 6.5 := by
  let BC := XC + CY
  let AC := real.sqrt (XC^2 + CY^2)
  let AB := real.sqrt (AC^2 + BC^2)
  let AQ := 6.5
  have hBC : BC = 5 := by sorry
  have hAC : AC = real.sqrt 13 := by sorry
  have hAB : AB = real.sqrt 38 := by sorry
  have hHalfAngleCos : real.cos (π / 3) = 1/2 := by sorry
  -- Verify the computation of AQ through law of cosines
  have hAQ : 38 = AQ^2 + (8 - AQ)^2 - 2 * AQ * (8 - AQ) * real.cos (π / 3) := by sorry
  exact ⟨AQ, rfl⟩

end length_AQ_l633_633611


namespace omega_range_l633_633672

open Real

noncomputable def f (ω x : ℝ) : ℝ := (sin (ω * x / 2))^2 + (1/2) * sin (ω * x) - (1/2)

theorem omega_range (ω : ℝ) : 
  (ω > 0) ∧ (∃ x : ℝ, x ∈ Ioo pi (2 * pi) ∧ f ω x = 0) ↔ 
  ω ∈ (Ioo (1 / 8) (1 / 4) ∪ Ioi (5 / 8)) :=
by sorry

end omega_range_l633_633672


namespace turnip_bag_weighs_l633_633092

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l633_633092


namespace insuranceCompensationIsZero_l633_633821

-- Define the insurance problem parameters
def insuredAmount : ℕ := 500000
def deductibleRate : ℚ := 1 / 100
def actualDamage : ℕ := 4000

-- Define what the threshold value for the deductible is
def thresholdValue : ℚ := insuredAmount * deductibleRate

def insuranceCompensation (insuredAmount : ℕ) (deductibleRate : ℚ) (actualDamage : ℕ) : ℕ := 
  if actualDamage < thresholdValue.to_nat then 0 else sorry -- Use sorry to handle the else part.

-- Prove that given the conditions, the insurance compensation amount is 0 rubles.
theorem insuranceCompensationIsZero :
  insuranceCompensation insuredAmount deductibleRate actualDamage = 0 := by
  -- Since the actual damage (4000) is less than the threshold (5000), the deduction applies.
  have h_threshold : actualDamage < thresholdValue.to_nat := by 
    calc
      actualDamage = 4000 : rfl
      ... < 5000 : by norm_num
  -- Thus, the compensation amount should be zero.
  rw [insuranceCompensation]
  simp [h_threshold]
  exact rfl

end insuranceCompensationIsZero_l633_633821


namespace denis_and_oleg_probability_l633_633128

noncomputable def probability_denisolga_play_each_other (n : ℕ) (i j : ℕ) (h1 : n = 26) (h2 : i ≠ j) : ℚ :=
  let number_of_pairs := (n * (n - 1)) / 2
  in (n - 1) / number_of_pairs

theorem denis_and_oleg_probability :
  probability_denisolga_play_each_other 26 1 2 rfl dec_trivial = 1 / 13 :=
sorry

end denis_and_oleg_probability_l633_633128


namespace find_points_M_l633_633407

-- Definition of the function y = x^2/4
def parabola (x : ℝ) : ℝ := x^2 / 4

-- Condition 1: The line x = 1
def line_x_equals_1 (M : ℝ × ℝ) : Prop :=
  M.fst = 1

-- Condition 3: Two tangents to the graph form an angle of 45 degrees
def tangents_form_45_degrees (M : ℝ × ℝ) : Prop :=
  let y0 := M.snd
  let k1 := (1 + Real.sqrt (1 + 4 * y0)) / 2
  let k2 := (1 - Real.sqrt (1 + 4 * y0)) / 2
  Real.tan (Real.arctan k2 - Real.arctan k1) = 1

theorem find_points_M :
  ∃ M : ℝ × ℝ, line_x_equals_1 M ∧ tangents_form_45_degrees M ∧ (M = (1, 0) ∨ M = (1, -6)) :=
by
  sorry

end find_points_M_l633_633407


namespace locus_equation_rectangle_perimeter_greater_l633_633335

open Real

theorem locus_equation (P : ℝ × ℝ) : 
  (abs P.2 = sqrt (P.1 ^ 2 + (P.2 - 1 / 2) ^ 2)) → (P.2 = P.1 ^ 2 + 1 / 4) :=
by
  intro h
  sorry

theorem rectangle_perimeter_greater (A B C D : ℝ × ℝ) :
  (A.2 = A.1 ^ 2 + 1 / 4) ∧ 
  (B.2 = B.1 ^ 2 + 1 / 4) ∧ 
  (C.2 = C.1 ^ 2 + 1 / 4) ∧ 
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) → 
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
by
  intro h
  sorry

end locus_equation_rectangle_perimeter_greater_l633_633335


namespace puppy_weight_l633_633941

variable (p k d : ℕ)

theorem puppy_weight :
  p + k + d = 36 ∧ p + d = 3 * k ∧ p + k = d → p = 9 := 
by
  intro h
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end puppy_weight_l633_633941


namespace problem_l633_633379

noncomputable def f : ℝ → ℝ := sorry

theorem problem (hf : ∀ x y : ℝ, f (f x + y) = f (x + y) + 2 * x * f y - 2 * x * y - 2 * x + 2) :
  let n := 1
  let s := f 1 
  n * s = 3 := 
by
  have n_def : n = 1 := rfl
  have s_def : s = f 1 := rfl
  rw [n_def, s_def]
  sorry

end problem_l633_633379


namespace zero_vector_collinear_with_any_vector_l633_633910

theorem zero_vector_collinear_with_any_vector
  (v : ℝ^3)
  (h0: ∥0∥ = 0)
  (h_collinear: ∃ l : ℝ, v = l • (0 : ℝ^3))
  : v ≠ 0 → collinear v 0 := 
sorry

end zero_vector_collinear_with_any_vector_l633_633910


namespace similar_triangle_perimeter_l633_633964

theorem similar_triangle_perimeter :
  ∀ (a b c : ℝ), a = 7 ∧ b = 7 ∧ c = 12 →
  ∀ (d : ℝ), d = 30 →
  ∃ (p : ℝ), p = 65 ∧ 
  (∃ a' b' c' : ℝ, (a' = 17.5 ∧ b' = 17.5 ∧ c' = d) ∧ p = a' + b' + c') :=
by sorry

end similar_triangle_perimeter_l633_633964


namespace angle_ABM_is_75_degrees_l633_633354

open Real
open EuclideanGeometry

noncomputable def is_midpoint (M A B : Point) : Prop := dist(A, M) = dist(M, B)

-- Defining the given problem in Lean 4
theorem angle_ABM_is_75_degrees
  (A B C M N : Point)
  (hM : is_midpoint M A C)
  (hN : is_midpoint N B C)
  (hMAN : angle M A N = 15)
  (hBAN : angle B A N = 45) :
  angle A B M = 75 :=
begin
  sorry
end

end angle_ABM_is_75_degrees_l633_633354


namespace barium_oxide_amount_l633_633620

theorem barium_oxide_amount (BaO H2O BaOH₂ : ℕ) 
  (reaction : BaO + H2O = BaOH₂) 
  (molar_ratio : BaOH₂ = BaO) 
  (required_BaOH₂ : BaOH₂ = 2) :
  BaO = 2 :=
by 
  sorry

end barium_oxide_amount_l633_633620


namespace expression_bounds_l633_633373

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) (hw : 0 ≤ w) (hw1 : w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by
  sorry

end expression_bounds_l633_633373


namespace find_x_such_that_l633_633192

theorem find_x_such_that {x : ℝ} (h : ⌈x⌉ * x + 15 = 210) : x = 195 / 14 :=
by
  sorry

end find_x_such_that_l633_633192


namespace polygon_angle_sum_l633_633873

theorem polygon_angle_sum (n : ℕ) (angle : ℕ) 
    (h₀ : ∀ (m : ℕ), m = n - 1 → 180 * m + angle = 1800)
    (h₁ : ∀ (s : ℕ), s = n - 2 → 180 * s = 1620) : 
    n = 11 ∧ angle = 180 :=
begin
  have h₂ : 1800 = 180 * (n - 1) + angle,
  { apply h₀, refl, },
  have h₃ : 1620 = 180 * (n - 2),
  { apply h₁, refl, },
  
  -- Solve for n and the angle using the provided equalities
  sorry
end

end polygon_angle_sum_l633_633873


namespace smallest_hotdog_packages_l633_633399

theorem smallest_hotdog_packages (h : ℕ := 10) (b : ℕ := 12) : nat.lcm h b / h = 6 := by
  sorry

end smallest_hotdog_packages_l633_633399


namespace barycentric_coordinates_are_proportional_l633_633376

open_locale big_operators

variables {P A B C : Type*}
          {α β γ : ℝ}
          [affine_space ℝ (triangle ABC)]
          [has_scalar ℝ (triangle ABC)]

-- Assume P is inside the triangle ABC and define areas as functions
def in_triangle (P A B C : triangle ABC) : Prop := ∃ α β γ, α + β + γ = 1

def area (triangle : triangle ABC) : ℝ := sorry

-- Define the barycentric coordinates of P with respect to triangle ABC
def barycentric_coords (P A B C : triangle ABC) : ℝ × ℝ × ℝ :=
  let [BPC] := area ⟨B, P, C⟩,
      [PCA] := area ⟨P, C, A⟩,
      [PAB] := area ⟨P, A, B⟩ in
  ([BPC], [PCA], [PAB])

-- Prove that the barycentric coordinates of P are proportional to the areas of the sub-triangles
theorem barycentric_coordinates_are_proportional
  (P A B C : triangle ABC) (hP : in_triangle P A B C) :
  barycentric_coords P A B C = (area ⟨B, P, C⟩, area ⟨P, C, A⟩, area ⟨P, A, B⟩) :=
sorry

end barycentric_coordinates_are_proportional_l633_633376


namespace line_perp_plane_l633_633923

variables {Point : Type} {Line : Type} {Plane : Type}
variables (m n : Line) (α β : Plane)
variable (perpendicular : Point → Point → Prop)

def line_perpendicular_plane (l : Line) (p : Plane) : Prop :=
  ∀ a b : Point, a ∈ l → b ∈ p → perpendicular a b

axiom per_line_plane : line_perpendicular_plane m β
axiom per_line_line : line_perpendicular_plane n β
axiom per_plane_plane : line_perpendicular_plane n α

theorem line_perp_plane : line_perpendicular_plane m α :=
by sorry

end line_perp_plane_l633_633923


namespace count_digit_five_1_to_700_l633_633294

def contains_digit_five (n : ℕ) : Prop :=
  n.digits 10 ∈ [5]

def count_up_to (n : ℕ) (p : ℕ → Prop) : ℕ :=
  (finset.range n).count p

theorem count_digit_five_1_to_700 : count_up_to 701 contains_digit_five = 52 := sorry

end count_digit_five_1_to_700_l633_633294


namespace expression_always_positive_l633_633411

theorem expression_always_positive (x : ℝ) : x^2 + |x| + 1 > 0 :=
by 
  sorry

end expression_always_positive_l633_633411


namespace marble_probability_l633_633018

theorem marble_probability :
  let total_marbles : ℕ := 20
  let red_marbles : ℕ := 12
  let blue_marbles : ℕ := 8
  let total_probability : ℚ := (6 * ((12 * 11 * 8 * 7) / (20 * 19 * 18 * 17))) in
  total_probability = (1232 / 4845) :=
by
  sorry

end marble_probability_l633_633018


namespace denis_and_oleg_probability_l633_633126

noncomputable def probability_denisolga_play_each_other (n : ℕ) (i j : ℕ) (h1 : n = 26) (h2 : i ≠ j) : ℚ :=
  let number_of_pairs := (n * (n - 1)) / 2
  in (n - 1) / number_of_pairs

theorem denis_and_oleg_probability :
  probability_denisolga_play_each_other 26 1 2 rfl dec_trivial = 1 / 13 :=
sorry

end denis_and_oleg_probability_l633_633126


namespace max_value_of_f_on_interval_min_value_of_f_on_interval_l633_633205

-- Define the function f(x) = x^3 - 3x^2 + 5
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

-- Define the interval
def interval := set.Icc (-1:ℝ) 1

-- State that the maximum value on the interval is 5
theorem max_value_of_f_on_interval : 
  is_lub (set.image f interval) 5 := 
sorry

-- State that the minimum value on the interval is 1
theorem min_value_of_f_on_interval : 
  is_glb (set.image f interval) 1 := 
sorry

end max_value_of_f_on_interval_min_value_of_f_on_interval_l633_633205


namespace expr1_value_at_x_neg3_n_2_expr2_value_at_x_neg1_l633_633421

def expr1 (x n : ℕ) : ℕ :=
  x^n * (x^n + 9 * x - 12) - 3 * (3 * x^(n + 1) - 4 * x^n)

theorem expr1_value_at_x_neg3_n_2 : expr1 (-3) 2 = 81 :=
  sorry

def expr2 (x : ℕ) : ℕ :=
  (x^2 + 2*x + 2) * (x + 2) + (-x^2 + 1) * (x - 5)

theorem expr2_value_at_x_neg1 : expr2 (-1) = 1 :=
  sorry

end expr1_value_at_x_neg3_n_2_expr2_value_at_x_neg1_l633_633421


namespace number_of_restaurants_l633_633032

def units_in_building : ℕ := 300
def residential_units := units_in_building / 2
def remaining_units := units_in_building - residential_units
def restaurants := remaining_units / 2

theorem number_of_restaurants : restaurants = 75 :=
by
  sorry

end number_of_restaurants_l633_633032


namespace turnips_bag_l633_633053

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l633_633053


namespace insuranceCompensationIsZero_l633_633822

-- Define the insurance problem parameters
def insuredAmount : ℕ := 500000
def deductibleRate : ℚ := 1 / 100
def actualDamage : ℕ := 4000

-- Define what the threshold value for the deductible is
def thresholdValue : ℚ := insuredAmount * deductibleRate

def insuranceCompensation (insuredAmount : ℕ) (deductibleRate : ℚ) (actualDamage : ℕ) : ℕ := 
  if actualDamage < thresholdValue.to_nat then 0 else sorry -- Use sorry to handle the else part.

-- Prove that given the conditions, the insurance compensation amount is 0 rubles.
theorem insuranceCompensationIsZero :
  insuranceCompensation insuredAmount deductibleRate actualDamage = 0 := by
  -- Since the actual damage (4000) is less than the threshold (5000), the deduction applies.
  have h_threshold : actualDamage < thresholdValue.to_nat := by 
    calc
      actualDamage = 4000 : rfl
      ... < 5000 : by norm_num
  -- Thus, the compensation amount should be zero.
  rw [insuranceCompensation]
  simp [h_threshold]
  exact rfl

end insuranceCompensationIsZero_l633_633822


namespace perfect_number_divisibility_l633_633111

theorem perfect_number_divisibility (n : ℕ) (h1 : perfect_number n) (h2 : 6 < n) (h3 : 3 ∣ n) : 9 ∣ n :=
sorry

-- Definitions depending on what we need for the proof, these are placeholders.
def perfect_number (n : ℕ) : Prop :=
  ∑ m in (finset.range n).filter (λ d, d ∣ n), m = n

end perfect_number_divisibility_l633_633111


namespace no_snow_five_days_l633_633838

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633838


namespace neg_prop_l633_633459

-- Definition of the proposition to be negated
def prop (x : ℝ) : Prop := x^2 + 2 * x + 5 = 0

-- Negation of the proposition
theorem neg_prop : ¬ (∃ x : ℝ, prop x) ↔ ∀ x : ℝ, ¬ prop x :=
by
  sorry

end neg_prop_l633_633459


namespace curve_C2_param_line_l_eq_max_distance_l633_633349

noncomputable def curve_C1 : ℝ → ℝ × ℝ :=
λ θ, (cos θ, sin θ)

def line_l_rect : (ℝ × ℝ) → ℝ :=
λ p, 2 * p.1 - p.2 - 6

noncomputable def curve_C2 : ℝ → ℝ × ℝ :=
λ θ, (sqrt 3 * cos θ, 2 * sin θ)

theorem curve_C2_param {θ : ℝ} :
  curve_C2 θ = (sqrt 3 * cos θ, 2 * sin θ) :=
by simp [curve_C2]

theorem line_l_eq : ∀ p : ℝ × ℝ, line_l_rect p = 2 * p.1 - p.2 - 6 :=
by simp [line_l_rect]

theorem max_distance {θ : ℝ} :
  0 ≤ θ ∧ θ ≤ 2 * pi →
  ∃ θ : ℝ, (θ = (5 * pi / 6)) ∧ (abs (2 * sqrt 3 * cos θ - 2 * sin θ - 6) / sqrt 5) = 2 * sqrt 5 :=
sorry

end curve_C2_param_line_l_eq_max_distance_l633_633349


namespace Soyun_total_numbers_count_l633_633433

-- Definition for the set of cards
def Soyun_initial_cards := {3, 5, 8, 9}

-- Function that computes the number of three-digit numbers formed by the remaining three cards
def num_three_digit_numbers (cards : Finset ℕ) : ℕ :=
  if cards.card = 3 then
    cards.to_list.permutations.length
  else
    0

-- Prove that the total number of three-digit numbers is 24
theorem Soyun_total_numbers_count : 
  ∀ (cards : Finset ℕ), Soyun_initial_cards = {3, 5, 8, 9} → 
    (Finset.sum (Soyun_initial_cards.image (λ card, num_three_digit_numbers (Soyun_initial_cards.erase card))) id = 24) :=
by
  sorry

end Soyun_total_numbers_count_l633_633433


namespace closest_point_to_line_l633_633628

theorem closest_point_to_line {x y : ℝ} :
  (y = 2 * x - 7) → (∃ p : ℝ × ℝ, p.1 = 5 ∧ p.2 = 3 ∧ (p.1, p.2) ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} ∧ (∀ q : ℝ × ℝ, q ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} → dist ⟨x, y⟩ p ≤ dist ⟨x, y⟩ q)) :=
by
  -- proof goes here
  sorry

end closest_point_to_line_l633_633628


namespace expected_value_of_coin_flip_l633_633102

open ProbabilityTheory

noncomputable def coinFlipWinnings : pmf ℤ :=
  pmf.of_multiset { 5, -3 }

theorem expected_value_of_coin_flip :
  expected_value coinFlipWinnings = 1 := by
  sorry

end expected_value_of_coin_flip_l633_633102


namespace no_snow_probability_l633_633848

theorem no_snow_probability (p_snow : ℚ) (h : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  (p_no_snow ^ 5 = 1 / 243) :=
by
  sorry

end no_snow_probability_l633_633848


namespace averageRounds_is_3_l633_633461

-- Define the number of rounds of golf and the corresponding number of golfers
def rounds : List ℕ := [1, 2, 3, 4, 5]
def golfers : List ℕ := [5, 2, 2, 3, 5]

-- Calculate the total rounds played
def totalRounds : ℕ := (List.zip rounds golfers).map (fun (r, g) => r * g).sum

-- Calculate the total number of golfers
def totalGolfers : ℕ := golfers.sum

-- Calculate the average number of rounds
def averageRounds : ℚ := totalRounds / totalGolfers

-- Round the average to the nearest whole number
def roundedAverageRounds : ℤ := averageRounds.toInt -- .toInt rounds to the nearest integer

-- The proof problem statement
theorem averageRounds_is_3 :
  roundedAverageRounds = 3 :=
by
  have h1 : totalRounds = 52 := by sorry
  have h2 : totalGolfers = 17 := by sorry
  have h3 : averageRounds = 52 / 17 := by sorry
  have h4 : roundedAverageRounds = (52 / 17).toInt := by sorry
  have h5 : roundedAverageRounds = 3 := by sorry
  assumption

end averageRounds_is_3_l633_633461


namespace count_odd_digits_in_product_l633_633270

def count_odd_digits (n : ℕ) : ℕ :=
  n.digits.filter (λ d, d % 2 = 1).length

theorem count_odd_digits_in_product :
  count_odd_digits (11111111 * 99999999) = 8 :=
by
  sorry

end count_odd_digits_in_product_l633_633270


namespace rectangle_width_decrease_l633_633450

theorem rectangle_width_decrease (L W : ℝ) (h1 : 0 < L) (h2 : 0 < W) 
(h3 : ∀ W' : ℝ, 0 < W' → (1.3 * L * W' = L * W) → W' = (100 - 23.077) / 100 * W) : 
  ∃ W' : ℝ, 0 < W' ∧ (1.3 * L * W' = L * W) ∧ ((W - W') / W = 23.077 / 100) :=
by
  sorry

end rectangle_width_decrease_l633_633450


namespace numbers_contain_digit_five_l633_633287

theorem numbers_contain_digit_five :
  (finset.range 701).filter (λ n, ∃ d ∈ (n.digits 10), d = 5).card = 214 := 
sorry

end numbers_contain_digit_five_l633_633287


namespace sqrt_a_b_c_l633_633469

variable (a b c : ℤ)

theorem sqrt_a_b_c :
  (∀ a b c, (2 * a - 1 = 9) ∧ (3 * a + b - 1 = 16) ∧ (c = 4) → (Real.sqrt (a + b - c) = Real.sqrt 3) ∨ (Real.sqrt (a + b - c) = -Real.sqrt 3)) :=
by
  intros a b c
  intro h
  cases h
  rw [← h_left] at h_right
  sorry

end sqrt_a_b_c_l633_633469


namespace minimum_of_PF_plus_PA_l633_633817

noncomputable def minimize_distance : ℝ := sorry

theorem minimum_of_PF_plus_PA :
  let p := (x, y) ∈ setOf (x, y) | y^2 = 4 * x
  let F := (1, 0)
  let A := (3, 1)
  ∃ P : ℝ × ℝ, P ∈ p ∧ ∀ P', P' ∈ p → |(P.1 - F.1)^2 + (P.2 - F.2)^2| + |(P.1 - A.1)^2 + (P.2 - A.2)^2| ≥ minimize_distance := sorry

end minimum_of_PF_plus_PA_l633_633817


namespace initial_persons_count_l633_633804

theorem initial_persons_count (P : ℕ) (H1 : 18 * P = 1) (H2 : 6 * P = 1/3) (H3 : 9 * (P + 4) = 2/3) : P = 12 :=
by
  sorry

end initial_persons_count_l633_633804


namespace smallest_integer_in_set_l633_633871

theorem smallest_integer_in_set : ∃ (n : ℤ), ∀ (x : ℝ), |x - 2| ≤ 5 → n ≤ x ∧ (∀ m, m < n → (∃ (y : ℝ), |y - 2| ≤ 5 ∧ m ≤ y) → false) :=
by
  have : ∃ (n : ℤ), -3 ≤ n :=
    ⟨-3, le_refl (-3)⟩
  cases this with n hn
  use n
  intro x hx
  simp only [abs_le, sub_le, le_sub_iff_add_le, sub_le_iff_le_add] at hx
  cases hx with h1 h2
  split
  exact h1.trans (le_of_lt h2)
  intro m hm hmy
  exfalso
  cases hmy with y hy
  cases hy
  have : y ≤ -3 :=
    h2.trans hy.2
  linarith

end smallest_integer_in_set_l633_633871


namespace projection_of_a_on_b_l633_633232

noncomputable def vector_projection (a b : ℝ × ℝ) (θ : ℝ) : ℝ :=
  (real.sqrt (a.fst^2 + a.snd^2)) * (real.cos θ)

theorem projection_of_a_on_b :
  let a := (-2 : ℝ, -4 : ℝ)
  let b_magnitude := real.sqrt 5
  let θ := real.pi * 2 / 3  -- 120 degrees in radians
  vector_projection a ⟨b_magnitude, 0⟩ θ = - real.sqrt 5 :=
sorry

end projection_of_a_on_b_l633_633232


namespace trigonometric_expression_l633_633217

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 3) : 
  ((Real.cos (α - π / 2) + Real.cos (α + π)) / (2 * Real.sin α) = 1 / 3) :=
by
  sorry

end trigonometric_expression_l633_633217


namespace max_min_value_l633_633251

def f (x t : ℝ) : ℝ := x^2 - 2 * t * x + t

theorem max_min_value : 
  ∀ t : ℝ, (-1 ≤ t ∧ t ≤ 1) →
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t) →
  (∃ t : ℝ, (-1 ≤ t ∧ t ≤ 1) ∧ ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t ∧ -t^2 + t = 1/4) :=
sorry

end max_min_value_l633_633251


namespace ascorbic_acid_oxygen_mass_percentage_l633_633624

noncomputable def mass_percentage_oxygen_in_ascorbic_acid : Float := 54.49

theorem ascorbic_acid_oxygen_mass_percentage :
  let C_mass := 12.01
  let H_mass := 1.01
  let O_mass := 16.00
  let ascorbic_acid_formula := (6, 8, 6) -- (number of C, number of H, number of O)
  let total_mass := 6 * C_mass + 8 * H_mass + 6 * O_mass
  let O_mass_total := 6 * O_mass
  mass_percentage_oxygen_in_ascorbic_acid = (O_mass_total / total_mass) * 100 := by
  sorry

end ascorbic_acid_oxygen_mass_percentage_l633_633624


namespace factor_value_l633_633012

theorem factor_value 
  (m : ℝ) 
  (h : ∀ x : ℝ, x + 5 = 0 → (x^2 - m * x - 40) = 0) : 
  m = 3 := 
sorry

end factor_value_l633_633012


namespace no_snow_probability_l633_633849

theorem no_snow_probability (p_snow : ℚ) (h : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  (p_no_snow ^ 5 = 1 / 243) :=
by
  sorry

end no_snow_probability_l633_633849


namespace max_intersections_4725_l633_633191

/-- Given 15 points on the positive x-axis and 10 points on the positive y-axis,
    and 150 segments connecting each point on the x-axis to each point on the y-axis,
    the maximum number of intersections points of these segments in the interior
    of the first quadrant is 4725. -/
theorem max_intersections_4725 :
  ∃ (x_points y_points : Finset ℝ), x_points.card = 15 ∧ y_points.card = 10 ∧
  (∃ (segments : Finset (ℝ × ℝ)),
    segments.card = 150 ∧
    (∀ (p : ℝ × ℝ), p ∈ segments → p.1 > 0 ∧ p.2 > 0) ∧
    let intersections := (choose 15 2) * (choose 10 2) in
    intersections = 4725) :=
begin
  sorry
end

end max_intersections_4725_l633_633191


namespace alpha_parallel_to_beta_l633_633657

variables (a b : ℝ → ℝ → ℝ) (α β : ℝ → ℝ)

-- Definitions based on conditions
def are_distinct_lines : a ≠ b := sorry
def are_distinct_planes : α ≠ β := sorry

def line_parallel_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define parallel relation
def line_perpendicular_to_plane (l : ℝ → ℝ → ℝ) (p : ℝ → ℝ) : Prop := sorry -- Define perpendicular relation
def planes_parallel (p1 p2 : ℝ → ℝ) : Prop := sorry -- Define planes being parallel

-- Given as conditions
axiom a_perpendicular_to_alpha : line_perpendicular_to_plane a α
axiom b_perpendicular_to_beta : line_perpendicular_to_plane b β
axiom a_parallel_to_b : a = b

-- The proposition to prove
theorem alpha_parallel_to_beta : planes_parallel α β :=
by {
  -- Placeholder for the logic provided through the previous solution steps.
  sorry
}

end alpha_parallel_to_beta_l633_633657


namespace simplify_expression_l633_633801

theorem simplify_expression (a : ℝ) (h : a = Real.sqrt 3 - 3) : 
  (a^2 - 4 * a + 4) / (a^2 - 4) / ((a - 2) / (a^2 + 2 * a)) + 3 = Real.sqrt 3 :=
by
  sorry

end simplify_expression_l633_633801


namespace existence_of_integers_l633_633427

-- Definitions based on the problem conditions
variables {x y z a b c : ℝ}

-- The system of equations given as conditions
def eq1 := (x * (y + z)) / (x + y + z) = a
def eq2 := (y * (z + x)) / (x + y + z) = b
def eq3 := (z * (x + y)) / (x + y + z) = c

-- The proof that such integers x, y, z satisfying the equations exist.
theorem existence_of_integers (h1 : eq1) (h2 : eq2) (h3 : eq3) : ∃ (x y z : ℤ), eq1 ∧ eq2 ∧ eq3 :=
begin
  sorry
end

end existence_of_integers_l633_633427


namespace raisins_in_second_box_l633_633584

theorem raisins_in_second_box 
  (total_raisins : ℕ := 437)
  (num_boxes : ℕ := 5)
  (first_box_raisins : ℕ := 72)
  (three_boxes_raisins_each : ℕ := 97)
  (raisins_in_second_box : ℕ) :
  first_box_raisins + (3 * three_boxes_raisins_each) + raisins_in_second_box = total_raisins → 
  raisins_in_second_box = 74 := 
by {
  intro h,
  sorry
}

end raisins_in_second_box_l633_633584


namespace exists_abc_for_n_l633_633787

open Real

noncomputable def exists_abc_ratios (k : ℕ) := set.Ioo (k^2 : ℝ) (k^2 + k + 3 * real.sqrt 3)

theorem exists_abc_for_n (n : ℕ) : 
  ∃ a b c : ℝ, (∃ k₁ k₂ k₃ : ℕ, a ∈ exists_abc_ratios k₁ ∧ b ∈ exists_abc_ratios k₂ ∧ c ∈ exists_abc_ratios k₃) ∧ n = a * b / c :=
sorry

end exists_abc_for_n_l633_633787


namespace find_2n_plus_m_l633_633826

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 2 * n + m = 36 :=
sorry

end find_2n_plus_m_l633_633826


namespace regular_octagon_area_l633_633947

-- Definitions and conditions
def radius : ℝ := 2.5
def n : ℕ := 8
def central_angle : ℝ := 360 / n
def triangle_area : ℝ := 1 / 2 * radius^2 * Real.sin (Float.toRadians (central_angle / 2))
def octagon_area : ℝ := n * triangle_area

-- Theorem statement
theorem regular_octagon_area : octagon_area = 17.672 := by
  sorry

end regular_octagon_area_l633_633947


namespace sara_gave_dan_pears_l633_633415

theorem sara_gave_dan_pears :
  ∀ (original_pears left_pears given_to_dan : ℕ),
    original_pears = 35 →
    left_pears = 7 →
    given_to_dan = original_pears - left_pears →
    given_to_dan = 28 :=
by
  intros original_pears left_pears given_to_dan h_original h_left h_given
  rw [h_original, h_left] at h_given
  exact h_given

end sara_gave_dan_pears_l633_633415


namespace find_solutions_for_x_l633_633508

noncomputable def solutions (x : Real) : Prop :=
  sqrt (1 + (cos x / sin x)) = sin x + cos x ∧ sin x ≠ 0 ∧ cos x ≠ 0

theorem find_solutions_for_x (x : Real) : 
  solutions x → 
  ∃ n : Int, x = n * 2 * Real.pi + (π / 4) ∨ 
  x = n * 2 * Real.pi + (-π / 4) ∨ 
  x = n * 2 * Real.pi + (3 * π / 4) :=
by 
  sorry

end find_solutions_for_x_l633_633508


namespace find_standard_ellipse_equation_find_maximum_area_of_triangle_OAB_and_line_equation_l633_633649

noncomputable def ellipse := 
  ∃ a b: ℝ, (a > b ∧ b > 0) ∧ ∀ x y: ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def line_through_focus_with_slope (alpha: ℝ): Prop := 
  ∃ A B: ℝ × ℝ, 
  ∃ F: ℝ × ℝ, F = (-√3, 0) ∧ 
  ∀ (x y: ℝ), y = tan(alpha) * x - tan(alpha) * (F.1) ∧ 
  (x, y) on_ellipse

noncomputable def slope_alpha : Prop :=
  slope = (π / 6) ∧ ∀ y1 y2: ℝ, y1 = -7 * y2

theorem find_standard_ellipse_equation: 
  ellipse ∧ line_through_focus_with_slope (π / 6) ∧ slope_alpha →
  (∀ x y: ℝ, (x^2 / 4) + y^2 = 1) := 
by 
  sorry

theorem find_maximum_area_of_triangle_OAB_and_line_equation:
  ellipse ∧ line_through_focus_with_slope (π / 6) ∧ slope_alpha →
  (∃ S: ℝ, S = 1) ∧ 
  (∃ k1 k2: ℝ, 
  k1 = (√2)/2 ∨ k1 = - (√2)/2 ∧ 
  k2 = (√2)/2 ∨ k2 = - (√2)/2 ∧ 
  ∀ x y: ℝ, y = k1 * x + √3 ∨ y = k2 * x + √3) := 
by 
  sorry

end find_standard_ellipse_equation_find_maximum_area_of_triangle_OAB_and_line_equation_l633_633649


namespace turnip_weight_possible_l633_633082

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l633_633082


namespace seating_problem_l633_633560

theorem seating_problem (A B C D E : char)
  (h1 : A ≠ 'B')
  (h2 : ((E = 'E' ∧ C = 'C') ∨ (C = 'E' ∧ E = 'C')))
  (h3 : ((D = 'E' ∧ ('A' = 'C' ∨ 'B' = 'C')) ∨ (D = 'C' ∧ ('A' = 'E' ∨ 'B' = 'E')))) :
  B = 'B' :=
sorry

end seating_problem_l633_633560


namespace olympic_medals_distribution_l633_633479

theorem olympic_medals_distribution :
  ∀ (sprinters : Finset ℕ) (americans : Finset ℕ),
    sprinters.card = 10 ∧ americans.card = 4 ∧ americans ⊆ sprinters →
    let non_americans := sprinters \ americans in
    (multiset.card (non_americans.val.powerset.length_eq 3) * 6! / 3! / 3!) +
    (americans.card * 3 * multiset.card (non_americans.val.powerset.length_eq 2) * 2! / 2!) = 480 :=
begin
  sorry
end

end olympic_medals_distribution_l633_633479


namespace sufficient_condition_a_gt_1_l633_633009

variable (a : ℝ)

theorem sufficient_condition_a_gt_1 (h : a > 1) : a^2 > 1 :=
by sorry

end sufficient_condition_a_gt_1_l633_633009


namespace area_of_triangle_QCA_l633_633179

noncomputable def triangle_area (x p : ℝ) (hx : x > 0) (hp : p < 12) : ℝ :=
  1 / 2 * x * (12 - p)

theorem area_of_triangle_QCA (x p : ℝ) (hx : x > 0) (hp : p < 12) :
  triangle_area x p hx hp = x * (12 - p) / 2 := by
  sorry

end area_of_triangle_QCA_l633_633179


namespace vector_opposite_coordinates_l633_633668

theorem vector_opposite_coordinates
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (3, -4))
  (h2 : ∃ x : ℝ, x < 0 ∧ b = (3 * x, -4 * x)) 
  (h3 : ∥b∥ = 10) :
  b = (-6, 8) :=
sorry

end vector_opposite_coordinates_l633_633668


namespace cos_double_angle_l633_633228

open Real

theorem cos_double_angle (α β : ℝ) 
    (h1 : sin α = 2 * sin β) 
    (h2 : tan α = 3 * tan β) :
  cos (2 * α) = -1 / 4 ∨ cos (2 * α) = 1 := 
sorry

end cos_double_angle_l633_633228


namespace product_of_sines_KLM_l633_633356

open Real

variables (K L M : Type) [NormedAddCommGroup K] [NormedAddCommGroup L] [NormedAddCommGroup M]
          (Klm : Triangle K L M)
          (KP P : Set K)
          (O Q R : Set K)
          [IsMedian KP Klm]
          [Circumcenter O Klm]
          [Incenter Q Klm]
          [Intersects R KP OQ Klm]
          (a b : Type) [NormedAddCommGroup a] [NormedAddCommGroup b]
          [Angle LKM = pi/3]

noncomputable def product_of_sines (α β : ℝ) (h1 : α + β = 2 * pi / 3) (h2 : sin α * sin β = 5 / 8) : Prop :=
  sin α * sin β = 5 / 8

theorem product_of_sines_KLM {K L M : Type} [NormedAddCommGroup K] [NormedAddCommGroup L] [NormedAddCommGroup M]
          (Klm : Triangle K L M)
          (KP P : Set K)
          (O Q R : Set K)
          [IsMedian KP Klm]
          [Circumcenter O Klm]
          [Incenter Q Klm]
          [Intersects R KP OQ Klm]
          (h : ∀ O R P Q : ℝ, OR / PR = √14 * (QR / KR))
          [Angle LKM = pi/3]
          {α β : ℝ} (h1 : α + β = 2 * pi / 3) : sin α * sin β = 5 / 8 := 
sorry

end product_of_sines_KLM_l633_633356


namespace problem1_problem2_part1_problem2_part2_l633_633352

-- Define the problem in Lean
axiom angle_A : ℝ
axiom angle_B : ℝ
axiom angle_C : ℝ
axiom ω : ℝ
axiom k : ℤ

-- Conditions from the problem
def triangle_angles (A B C : ℝ) : Prop := 
  A + B + C = π

def angle_relation (A B : ℝ) : Prop := 
  B = 2 * A

def vectors_perpendicular (A B : ℝ) : Prop := 
  let m := (Real.cos A, -Real.sin B)
  let n := (Real.cos B, Real.sin A)
  m.1 * n.1 + m.2 * n.2 = 0

def function_f (x : ℝ) (ω A : ℝ) : ℝ :=
  Real.cos (ω * x - A / 2) + Real.sin (ω * x)

def smallest_period (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x : ℝ, f(x + period) = f(x)

def increasing_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f(x) < f(y)

noncomputable def max_value_on_interval (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  Real.sup (Set.image f (Set.Icc a b))

-- Lean statements
theorem problem1 :
  triangle_angles angle_A angle_B angle_C →
  angle_relation angle_A angle_B →
  vectors_perpendicular angle_A angle_B →
  angle_B = π / 3 :=
by 
  intros _ _ _ 
  sorry

theorem problem2_part1 :
  smallest_period (function_f ω angle_B) π →
  increasing_interval (function_f ω π/6) (-2*π/3 + k*π) (π/3 + k*π) :=
by 
  intros _ 
  sorry

theorem problem2_part2 :
  ∀ (x : ℝ), 0 ≤ x → x ≤ π/2 →
  max_value_on_interval (function_f ω (π/6)) 0 (π/2) = sqrt 3 :=
by 
  intros _ _ 
  sorry

end problem1_problem2_part1_problem2_part2_l633_633352


namespace sequence_ineq_l633_633391

theorem sequence_ineq (a : ℕ → ℝ) :
  (∀ k : ℕ, 0 ≤ a[k] ∧ a[k] - 2 * a[k+1] + a[k+2] ≥ 0) ∧
  (∀ k : ℕ, ∑ j in range k, a[j] ≤ 1) → 
  ∀ k : ℕ, 0 ≤ a[k] - a[k+1] ∧ a[k] - a[k+1] < 2 / (k:ℝ)^2 := 
sorry

end sequence_ineq_l633_633391


namespace grey_to_brown_profit_percentage_l633_633776

-- Define the conditions
def house_original_value : ℝ := 100000
def house_sold_at_loss : ℝ := 99000
def loss_percentage : ℝ := 0.10

-- Main theorem statement
theorem grey_to_brown_profit_percentage :
  let bought_price := house_sold_at_loss / (1 - loss_percentage) in
  let selling_price := bought_price in
  let profit_percentage := ((selling_price - house_original_value) / house_original_value) * 100 in
  profit_percentage = 10 :=
by
  -- Proof of the theorem (We skip this as per instruction)
  sorry

end grey_to_brown_profit_percentage_l633_633776


namespace income_of_sixth_member_l633_633880

def income_member1 : ℝ := 11000
def income_member2 : ℝ := 15000
def income_member3 : ℝ := 10000
def income_member4 : ℝ := 9000
def income_member5 : ℝ := 13000
def number_of_members : ℕ := 6
def average_income : ℝ := 12000
def total_income_of_five_members := income_member1 + income_member2 + income_member3 + income_member4 + income_member5

theorem income_of_sixth_member :
  6 * average_income - total_income_of_five_members = 14000 := by
  sorry

end income_of_sixth_member_l633_633880


namespace range_of_x_l633_633636

theorem range_of_x (x : ℝ) : (det !![(x + 3) x^2; 1 4] < 0) ↔ (x < -2 ∨ x > 6) :=
by {
  -- Proof step is not required, but it leads to: 
  -- The determinant of the matrix < 0 simplifies to (x + 2)(x - 6) > 0
  sorry
}

end range_of_x_l633_633636


namespace choose_lines_intersect_l633_633359

-- We need to define the proof problem
theorem choose_lines_intersect : 
  ∃ (lines : ℕ → ℝ × ℝ → ℝ), 
    (∀ i j, i < 100 ∧ j < 100 ∧ i ≠ j → (lines i = lines j) → ∃ (p : ℕ), p = 2022) :=
sorry

end choose_lines_intersect_l633_633359


namespace train_length_proof_l633_633955

-- Definitions and conditions
def train_speed_kmh : ℝ := 54
def bridge_length_m : ℝ := 320
def crossing_time_s : ℝ := 30

-- Conversion factor: kilometers per hour to meters per second
def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Problem statement
theorem train_length_proof : 
  let train_speed_mps := kmh_to_mps train_speed_kmh in
  let total_distance := train_speed_mps * crossing_time_s in
  let train_length := total_distance - bridge_length_m in
  train_length = 130 :=
by
  sorry

end train_length_proof_l633_633955


namespace numbers_with_digit_5_count_numbers_with_digit_5_l633_633272

theorem numbers_with_digit_5 (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 700) : 
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ d = 5) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m : ℕ, (n = m * 100 + d ∧ d = 5)) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m2 m1 : ℕ, (n = m2 * 1000 + m1 * 100 + d ∧ d = 5)) :=
sorry

theorem count_numbers_with_digit_5 : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 700 ∧ (numbers_with_digit_5 n sorry) }.to_finset.card = 214 := 
sorry

end numbers_with_digit_5_count_numbers_with_digit_5_l633_633272


namespace no_snow_probability_l633_633847

theorem no_snow_probability (p_snow : ℚ) (h : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  (p_no_snow ^ 5 = 1 / 243) :=
by
  sorry

end no_snow_probability_l633_633847


namespace probability_denis_oleg_play_l633_633138

theorem probability_denis_oleg_play (n : ℕ) (h_n : n = 26) :
  (1 : ℚ) / 13 = 
  let total_matches : ℕ := n - 1 in
  let num_pairs := n * (n - 1) / 2 in
  (total_matches : ℚ) / num_pairs :=
by
  -- You can provide a proof here if necessary
  sorry

end probability_denis_oleg_play_l633_633138


namespace part_1_part_2_l633_633667

noncomputable def f (x : ℝ) (m : ℤ) : ℝ := x ^ (-m^2 + 2 * m + 3)

theorem part_1 (h1 : (f x m = f (-x) m)) (h2 : 0 < x → f x 1 > f y) : m = 1 := 
sorry

noncomputable def g (x : ℝ) (c : ℝ) : ℝ := (f x 1).sqrt + 2 * x + c

theorem part_2 (h3 : ∀ x, g x c > 2) : c > 3 :=
sorry

end part_1_part_2_l633_633667


namespace tangent_line_at_pi_over_2_l633_633678

noncomputable def f (x : ℝ) : ℝ := x^2 * Float.sin x

def tangent_line_eq (x y : ℝ) : Prop := π * x - y - (π^2 / 4) = 0

theorem tangent_line_at_pi_over_2 : 
    tangent_line_eq (π/2) (f (π / 2)) :=
by
    sorry

end tangent_line_at_pi_over_2_l633_633678


namespace trajectory_of_P_range_of_area_l633_633663

-- Define the circle and the point F
def circle (x y : ℝ) : Prop := (x + real.sqrt 3)^2 + y^2 = 16
def point_F (x y : ℝ) : Prop := x = real.sqrt 3 ∧ y = 0

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- The first proof problem: trajectory of P (the intersection point of perpendicular bisector and EM)
theorem trajectory_of_P (M : ℝ × ℝ) (P : ℝ × ℝ) :
  circle M.1 M.2 →
  ∃ E : ℝ × ℝ, E.1 = -real.sqrt 3 ∧ E.2 = 0 ∧ -- E is the center of the circle, specifically ordered pair
  (∃ EF : ℝ, EF = 2 * real.sqrt 3) ∧
  (∃ C : ℝ → ℝ → Prop, C = ellipse) →
  C P.1 P.2 := 
sorry

-- The second proof problem: range of the rectangle's area S
theorem range_of_area (S : ℝ) :
  (∃ k1 k2 m n : ℝ, k1 * k2 = -1 ∧ 
    (∃ x y : ℝ, ellipse x y ∧ y = k1 * x + m ∧ |m| = real.sqrt (4 * k1^2 + 1) ∧
    (∃ x y : ℝ, ellipse x y ∧ y = k2 * x + n ∧ |n| = real.sqrt (4 * k2^2 + 1))) ∧
    S = 4 * real.sqrt (9 / (2 + (k1^2 + 1/k1^2)) + 4) ∧ 
    S ∈ set.Icc 8 10) →
  S ∈ set.Icc 8 10 :=
sorry

end trajectory_of_P_range_of_area_l633_633663


namespace chemist_salt_solution_l633_633527

theorem chemist_salt_solution (x : ℝ) :
  let total_volume := 1 + 2,
      resulting_salt_percentage := 20,
      resulting_salt_volume := (total_volume:ℝ) * resulting_salt_percentage / 100,
      input_salt_volume := (2:ℝ) * (x / 100) in
  input_salt_volume = resulting_salt_volume → x = 30 :=
by
  intros
  sorry

end chemist_salt_solution_l633_633527


namespace part1_part2_l633_633235

-- Given
variable (m : ℝ) (z ω : ℂ)
def purely_imaginary (z : ℂ) : Prop := z.re = 0

-- Part 1: Prove that z is purely imaginary implies m = 2
theorem part1 (hm : purely_imaginary (m * (m - 2) + complex.i * m)) : m = 2 := 
sorry

-- Part 2: Given omega satisfies |ω| = |z| and ω + conjugate(ω) = 2, find ω
theorem part2 (hz : z = 2 * complex.i) (hω : complex.abs ω = complex.abs z) (hω_conj : ω + ω.conj = 2) :
ω = 1 + complex.sqrt 3 * complex.i ∨ ω = 1 - complex.sqrt 3 * complex.i := 
sorry


end part1_part2_l633_633235


namespace angle_bisectors_form_rectangle_l633_633413

variables {A B C D M N K L : Type} [MetricSpace C]

structure Parallelogram (A B C D : C) : Prop :=
  (dist_AB_eq_CD : dist A B = dist C D)
  (dist_BC_eq_AD : dist B C = dist A D)
  (not_rhombus : dist A B ≠ dist A D)

structure Rectangle (M N K L : C) : Prop :=
  (dist_MK_eq_NL : dist M K = dist N L)
  (dist_MN_eq_KL : dist M N = dist K L)
  (right_angle : ∀ (Q : C), angle Q M N = π / 2)

-- The main theorem stating that the angle bisectors of the parallelogram form a rectangle
-- and that the diagonal of this rectangle is equal to the difference of the lengths of any two adjacent sides.

theorem angle_bisectors_form_rectangle
  (A B C D M N K L : C)
  (h_parallelogram : Parallelogram A B C D)
  (h_intersections : M = intersection_of_angle_bisectors B C ∧
                     N = intersection_of_angle_bisectors C D ∧
                     K = intersection_of_angle_bisectors A D ∧
                     L = intersection_of_angle_bisectors A B) :
  Rectangle M N K L ∧ dist M K = abs (dist A B - dist B C) :=
begin
  sorry
end

end angle_bisectors_form_rectangle_l633_633413


namespace calculate_selling_price_l633_633037

theorem calculate_selling_price (CP : ℝ) (gain_percent : ℝ) (gain_amount : ℝ) (SP : ℝ) : 
  CP = 900 → gain_percent = 31.11111111111111 → gain_amount = (gain_percent / 100) * CP → 
  SP = CP + gain_amount → SP = 1180 :=
by
  intros hCP hGP hGA hSP
  rw [hCP, hGP] at hGA hSP
  sorry

end calculate_selling_price_l633_633037


namespace inequality_a_b_c_l633_633640

theorem inequality_a_b_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
sorry

end inequality_a_b_c_l633_633640


namespace min_value_C_ratio_l633_633655

variable (x : ℝ) (m : ℕ) (h_x_pos : x > 0)

def C (x : ℝ) (m : ℕ) : ℝ :=
  x * (x - 1) * ... * (x - m + 1) / m.factorial

theorem min_value_C_ratio : (C x 3) / (C x 1)^2 ≥ (real.sqrt(2) / 3 - 1 / 2) :=
by
  sorry

end min_value_C_ratio_l633_633655


namespace center_of_circumscribed_circle_lies_on_BC_l633_633386

-- Define the points and lines involved
variables {A A1 B B1 C C1 H K L : Type*}
variables (ABC : Triangle A B C) (AA1 : Line A A1) (BB1 : Line B B1) (CC1 : Line C C1)
          (KLline : Line K L) (ACline : Line A C) (A1C1line : Line A1 C1)

-- Conditions
def is_altitude (line : Line) (triangle : Triangle) : Prop :=
  -- Define what it means for a line to be an altitude in this context
  sorry

def is_perpendicular_to (line1 line2 : Line) : Prop :=
  -- Define perpendicular lines
  sorry

def intersects_at (line1 line2 : Line) (point : Type*) : Prop :=
  -- Define line intersection at a given point
  sorry

def lies_on (point : Type*) (line : Line) : Prop :=
  -- Define a point lying on a line
  sorry

noncomputable def circumscribed_circle_center (triangle : Triangle) : Type* :=
  -- Define the center of the circumscribed circle of a given triangle
  sorry

-- Proof Problem
theorem center_of_circumscribed_circle_lies_on_BC :
  is_altitude AA1 ABC →
  is_altitude BB1 ABC →
  is_altitude CC1 ABC →
  is_perpendicular_to KLline (line_through A B) →
  intersects_at KLline ACline K →
  intersects_at KLline A1C1line L →
  lies_on (circumscribed_circle_center (Triangle.mk K L B1)) (line_through B C) :=
begin
  -- Proof to be filled in
  sorry
end

end center_of_circumscribed_circle_lies_on_BC_l633_633386


namespace problem_solution_l633_633733

noncomputable theory
open_locale classical

-- Define the sequence a_n
def a_n (n : ℕ) : ℝ :=
if n = 1 then 1/2 else
if n = 2 then 3/4 else
if n = 3 then 7/8 else sorry -- Define for other n as needed

-- Define b_n and c_n
def b_n (n : ℕ) : ℝ := a_n n - 1

def c_n (n : ℕ) : ℝ := b_n n * (n - n^2)

-- Main theorem to prove the required properties
theorem problem_solution (t : ℝ) (n : ℕ) (hn : n > 0) :
  (a_n 1 = 1/2) ∧ (a_n 2 = 3/4) ∧ (a_n 3 = 7/8) ∧
  (∀ n : ℕ, n > 0 → ∃ (r : ℝ), 
    (r = 1/2) ∧ (b_n (n + 1) = r * b_n n)) ∧
  (∀ n : ℕ, n > 0 → c_n n + 1/4 * t ≤ t^2 →
    t ≥ 1 ∨ t ≤ -3/4) :=
by
  split,
  -- Prove a1, a2, a3
  { show a_n 1 = 1/2, sorry },
  { split, 
    { show a_n 2 = 3/4, sorry },
    { split, 
      { show a_n 3 = 7/8, sorry },
      -- Prove the sequence {a_n - 1} is geometric
      { split, 
        { intros n hn,
          use 1/2,
          split, 
          { refl },
          { sorry } },
        -- Prove the range of t
        { intros n hn h,
          left,
          sorry } } } } }

end problem_solution_l633_633733


namespace first_discount_is_20_l633_633916

theorem first_discount_is_20
  (initial_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (x : ℝ) :
  initial_price = 150 →
  final_price = 105 →
  second_discount = 0.125 →
  (initial_price - (x / 100) * initial_price) * (1 - second_discount) = final_price →
  x = 20 :=
by
  intros h1 h2 h3 h4
  have eq1 : (150 - (x / 100) * 150) * (1 - 0.125) = 105, from h4,
  sorry

end first_discount_is_20_l633_633916


namespace prime_square_mod_180_l633_633568

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : 5 < p) :
  ∃ (s : Finset ℕ), s.card = 2 ∧ ∀ r ∈ s, ∃ n : ℕ, p^2 % 180 = r :=
sorry

end prime_square_mod_180_l633_633568


namespace polynomial_has_root_l633_633909

theorem polynomial_has_root :
  let a := Real.sin (10 * Real.pi / 180) in
  8 * a^3 - 6 * a + 1 = 0 :=
by
  let a := Real.sin (10 * Real.pi / 180)
  sorry

end polynomial_has_root_l633_633909


namespace no_snow_five_days_l633_633855

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633855


namespace number_of_valid_pairs_l633_633939

theorem number_of_valid_pairs :
  let s := {5, 8, 12, 15}
  let t := insert 12 (insert 29 s)
  let u := insert 29 (insert 12 s)
  ∃ (f : set ℕ), f = t ∧ (f.mean == f.median) ∨ f = u ∧ (f.mean == f.median) → 
  ∃! (m n : ℕ), 
      m ≠ n ∧ m ∉ s ∧ n ∉ s ∧ 
      let ns := s ∪ {m, n} in 
      ns.mean = ns.median :=
sorry

end number_of_valid_pairs_l633_633939


namespace tan_identity_proof_l633_633921

theorem tan_identity_proof :
  let a := Real.tan (20 * Real.pi / 180)
  let b := Real.tan (30 * Real.pi / 180)
  let c := Real.tan (40 * Real.pi / 180)
  tan (40 * Real.pi / 180 + 20 * Real.pi / 180) = Real.sqrt 3 →
  tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 →
  a * b + b * c + c * a = 1 := by
  intro h1 h2
  sorry

end tan_identity_proof_l633_633921


namespace no_blonde_girls_added_l633_633396

-- Initial number of girls
def total_girls : Nat := 80
def initial_blonde_girls : Nat := 30
def black_haired_girls : Nat := 50

-- Number of blonde girls added
def blonde_girls_added : Nat := total_girls - black_haired_girls - initial_blonde_girls

theorem no_blonde_girls_added : blonde_girls_added = 0 :=
by
  sorry

end no_blonde_girls_added_l633_633396


namespace no_snow_five_days_l633_633853

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l633_633853


namespace proof_problem_equivalent_expression_l633_633309

variable {a b : ℝ}

theorem proof_problem (h : |a + 2| + Real.sqrt (b - 4) = 0) : a = -2 ∧ b = 4 :=
begin
  sorry
end

theorem equivalent_expression (ha : a = -2) (hb : b = 4) : (a^2) / b = 1 :=
begin
  sorry
end

end proof_problem_equivalent_expression_l633_633309


namespace abe_budget_ratio_l633_633559

theorem abe_budget_ratio :
  ∀ (budget : ℕ) (supply_ratio : ℕ) (employee_wages : ℕ) (food_expense : ℕ),
    budget = 3000 →
    supply_ratio = 1/4 →
    employee_wages = 1250 →
    food_expense = budget - (budget * supply_ratio) - employee_wages →
    food_expense.to_rat / budget = 1 / 3 :=
by
  intros budget supply_ratio employee_wages food_expense
  intros h_budget h_supply_ratio h_employee_wages h_food_expense
  sorry

end abe_budget_ratio_l633_633559


namespace benny_missed_games_l633_633975

theorem benny_missed_games (total_games attended_games missed_games : ℕ)
  (H1 : total_games = 39)
  (H2 : attended_games = 14)
  (H3 : missed_games = total_games - attended_games) :
  missed_games = 25 :=
by
  sorry

end benny_missed_games_l633_633975


namespace basketball_scores_l633_633516

theorem basketball_scores (n : ℕ) (h : n = 7) : 
  ∃ (k : ℕ), k = 8 :=
by {
  sorry
}

end basketball_scores_l633_633516


namespace turnip_bag_weights_l633_633071

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l633_633071


namespace cost_green_tea_july_l633_633918

-- Definitions of variables and conditions
variables {cost_june : ℝ} -- cost per pound of green tea and coffee in June
variable (price_mix : ℝ) -- cost of 3 lbs mixture in July
axiom eq_cost : price_mix = 3.15
axiom k1 : price_mix = ((1.5 * 0.1 * cost_june) + (1.5 * 2 * cost_june))

theorem cost_green_tea_july (cost_june : ℝ) (price_mix : ℝ) (eq_cost : price_mix = 3.15) (k1 : price_mix = ((1.5 * 0.1 * cost_june) + (1.5 * 2 * cost_june))) :
    (0.1 * cost_june) = 0.10 :=
by
suffices cost_june = 1 by
calc
  0.1 * cost_june = 0.1 * 1 : by rw [←this]
  ... = 0.10 : by norm_num
sorry

end cost_green_tea_july_l633_633918


namespace not_snowing_next_five_days_l633_633860

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l633_633860


namespace correct_mark_l633_633541

theorem correct_mark (x : ℕ) (h1 : 73 - x = 10) : x = 63 :=
by
  sorry

end correct_mark_l633_633541


namespace average_marathon_time_is_7_hours_l633_633981

-- Definitions of the problem conditions
def Casey_time : ℕ := 6
def Zendaya_additional_time (C : ℕ) : ℕ := (1 / 3 : ℚ) * C

@[simp]
theorem average_marathon_time_is_7_hours :
  let Zendaya_time := Casey_time + Zendaya_additional_time Casey_time,
      combined_time := Casey_time + Zendaya_time,
      average_time := combined_time / 2
  in average_time = 7 :=
by
  sorry

end average_marathon_time_is_7_hours_l633_633981


namespace tangent_line_at_extremum_range_of_a_plus_b_l633_633675

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 4 * x ^ 2 + 1 / x - a
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := f x a + b

theorem tangent_line_at_extremum (a : ℝ) (h1 : 12 - a = 0) : 
  ∀ x, y = x * (f x a) → 
  (∃ y', (f'(1) = 7) → y' + 7 = 7 * (x - 1) → y = 7 * x - 14) := 
sorry

theorem range_of_a_plus_b (a b : ℝ) (h2 : ∀ x, f x a = 0 → f(g x a b) = 6) :
  ∃ a b, f(g(x) a b) < 2 → a + b < 2 := 
sorry

end tangent_line_at_extremum_range_of_a_plus_b_l633_633675


namespace magic_card_profit_l633_633771

theorem magic_card_profit (purchase_price : ℝ) (multiplier : ℝ) (selling_price : ℝ) (profit : ℝ) 
                          (h1 : purchase_price = 100) 
                          (h2 : multiplier = 3) 
                          (h3 : selling_price = purchase_price * multiplier) 
                          (h4 : profit = selling_price - purchase_price) : 
                          profit = 200 :=
by 
  -- Here, you can introduce intermediate steps if needed.
  sorry

end magic_card_profit_l633_633771


namespace lock_settings_count_l633_633953

theorem lock_settings_count : 
    (∃ k n : ℕ, k = 4 ∧ n = 10 ∧ nat.choose (k + n - 1) k = 715) :=
  sorry

end lock_settings_count_l633_633953


namespace locus_equation_perimeter_greater_l633_633337

-- Define the conditions under which the problem is stated
def distance_to_x_axis (P : ℝ × ℝ) : ℝ := abs P.2
def distance_to_point (P : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- P is a point on the locus W if the distance to the x-axis is equal to the distance to (0, 1/2)
def on_locus (P : ℝ × ℝ) : Prop := 
  distance_to_x_axis P = distance_to_point P (0, 1/2)

-- Prove that the equation of W is y = x^2 + 1/4 given the conditions
theorem locus_equation (P : ℝ × ℝ) (h : on_locus P) : 
  P.2 = P.1^2 + 1/4 := 
sorry

-- Assume rectangle ABCD with three points on W
def point_on_w (P : ℝ × ℝ) : Prop := 
  P.2 = P.1^2 + 1/4

def points_form_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 ≠ B.1 ∧ B.1 ≠ C.1 ∧ C.1 ≠ D.1 ∧ D.1 ≠ A.1 ∧
  A.2 ≠ B.2 ∧ B.2 ≠ C.2 ∧ C.2 ≠ D.2 ∧ D.2 ≠ A.2

-- P1, P2, and P3 are three points on the locus W
def points_on_locus (A B C : ℝ × ℝ) : Prop := 
  point_on_w A ∧ point_on_w B ∧ point_on_w C

-- Prove the perimeter of rectangle ABCD with three points on W is greater than 3sqrt(3)
theorem perimeter_greater (A B C D : ℝ × ℝ) 
  (h1 : points_on_locus A B C) 
  (h2 : points_form_rectangle A B C D) : 
  2 * (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
       real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) > 
  3 * real.sqrt 3 := 
sorry

end locus_equation_perimeter_greater_l633_633337


namespace tan_alpha_implication_l633_633218

theorem tan_alpha_implication (α : ℝ) (h : Real.tan α = 2) :
    (2 * Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 3 / 5 := 
by 
  sorry

end tan_alpha_implication_l633_633218


namespace locus_equation_perimeter_greater_l633_633338

-- Define the conditions under which the problem is stated
def distance_to_x_axis (P : ℝ × ℝ) : ℝ := abs P.2
def distance_to_point (P : ℝ × ℝ) (Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- P is a point on the locus W if the distance to the x-axis is equal to the distance to (0, 1/2)
def on_locus (P : ℝ × ℝ) : Prop := 
  distance_to_x_axis P = distance_to_point P (0, 1/2)

-- Prove that the equation of W is y = x^2 + 1/4 given the conditions
theorem locus_equation (P : ℝ × ℝ) (h : on_locus P) : 
  P.2 = P.1^2 + 1/4 := 
sorry

-- Assume rectangle ABCD with three points on W
def point_on_w (P : ℝ × ℝ) : Prop := 
  P.2 = P.1^2 + 1/4

def points_form_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 ≠ B.1 ∧ B.1 ≠ C.1 ∧ C.1 ≠ D.1 ∧ D.1 ≠ A.1 ∧
  A.2 ≠ B.2 ∧ B.2 ≠ C.2 ∧ C.2 ≠ D.2 ∧ D.2 ≠ A.2

-- P1, P2, and P3 are three points on the locus W
def points_on_locus (A B C : ℝ × ℝ) : Prop := 
  point_on_w A ∧ point_on_w B ∧ point_on_w C

-- Prove the perimeter of rectangle ABCD with three points on W is greater than 3sqrt(3)
theorem perimeter_greater (A B C D : ℝ × ℝ) 
  (h1 : points_on_locus A B C) 
  (h2 : points_form_rectangle A B C D) : 
  2 * (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
       real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) > 
  3 * real.sqrt 3 := 
sorry

end locus_equation_perimeter_greater_l633_633338


namespace value_of_a_cubed_l633_633699

noncomputable def integral_cos_pi_div_4 : ℝ := ∫ x in 0..(Real.pi / 4), Real.cos x
noncomputable def integral_x_sq_a (a : ℝ) : ℝ := ∫ x in 0..a, x^2

theorem value_of_a_cubed {a : ℝ} (h : integral_cos_pi_div_4 = integral_x_sq_a a) : a^3 = (3 * Real.sqrt 2) / 2 :=
by
  sorry

end value_of_a_cubed_l633_633699


namespace prob_bashers_win_at_least_4_out_of_5_l633_633807

-- Define the probability p that the Bashers win a single game.
def p := 4 / 5

-- Define the number of games n.
def n := 5

-- Define the random trial outcome space.
def trials : Type := Fin n → Bool

-- Define the number of wins (true means a win, false means a loss).
def wins (t : trials) : ℕ := (Finset.univ.filter (λ i => t i = true)).card

-- Define winning exactly k games.
def win_exactly (t : trials) (k : ℕ) : Prop := wins t = k

-- Define the probability of winning exactly k games.
noncomputable def prob_win_exactly (k : ℕ) : ℚ :=
  (Nat.descFactorial n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the event of winning at least 4 out of 5 games.
def event_win_at_least (t : trials) := (wins t ≥ 4)

-- Define the probability of winning at least k out of n games.
noncomputable def prob_win_at_least (k : ℕ) : ℚ :=
  prob_win_exactly k + prob_win_exactly (k + 1)

-- Theorem to prove: Probability of winning at least 4 out of 5 games is 3072/3125.
theorem prob_bashers_win_at_least_4_out_of_5 :
  prob_win_at_least 4 = 3072 / 3125 :=
by
  sorry

end prob_bashers_win_at_least_4_out_of_5_l633_633807


namespace xialiang_payment_correct_l633_633398

-- Conditions of the problem are expressed as definitions
def discount (x : ℝ) : ℝ :=
  if x < 200 then x
  else if x < 500 then 0.9 * x
  else 450 + 0.8 * (x - 500)

-- Example shopping amounts
def xiaoming_first_trip := 198
def xiaoming_second_trip := 554

-- Calculate the total price of items purchased in two trips
def total_goods_price :=
  if discount xiaoming_first_trip = 198 then 198 else 220 +
  (554 + 0.8 * (630 - 500))

-- Total amount needed if purchased in one go
def xialiang_payment := 
  if total_goods_price = 828 then 450 + (828 - 500) * 0.8 else 
  450 + (850 - 500) * 0.8

-- We are proving that the total payment equals the given answers
theorem xialiang_payment_correct : xialiang_payment = 712.4 ∨ xialiang_payment = 730 :=
by sorry

end xialiang_payment_correct_l633_633398


namespace product_rounded_result_l633_633587

theorem product_rounded_result :
  let sum : ℝ := 30.2 + 0.08;
  let product : ℝ := 2.1 * sum in
  Int.round product = 64 :=
by
  let sum : ℝ := 30.2 + 0.08
  let product : ℝ := 2.1 * sum
  -- skipping the exact steps to calculate and round it
  sorry

end product_rounded_result_l633_633587


namespace circle_area_is_162_pi_l633_633181

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

def R : ℝ × ℝ := (5, -2)
def S : ℝ × ℝ := (-4, 7)

theorem circle_area_is_162_pi :
  circle_area (distance R S) = 162 * Real.pi :=
by
  sorry

end circle_area_is_162_pi_l633_633181


namespace distance_from_Red_Deer_to_Calgary_l633_633610

theorem distance_from_Red_Deer_to_Calgary
    (distance_Edmonton_RedDeer : ℕ := 220)
    (speed : ℕ := 110)
    (time : ℕ := 3) :
    let total_distance : ℕ := speed * time in
    let distance_RedDeer_Calgary := total_distance - distance_Edmonton_RedDeer in
    distance_RedDeer_Calgary = 110 := 
by
  sorry

end distance_from_Red_Deer_to_Calgary_l633_633610


namespace max_and_min_values_l633_633200

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5

theorem max_and_min_values :
  (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x ≤ 5) ∧ (∃ c ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f c = 5) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), 1 ≤ f x) ∧ (∃ c ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f c = 1) :=
begin
  sorry
end

end max_and_min_values_l633_633200


namespace find_b_3_pow_100_l633_633767

noncomputable def seq : ℕ → ℕ
| 1       := 2
| (3 * n) := (n + 1) * seq n
| _       := 0  -- We only use the definition for 1 and multiples of 3

theorem find_b_3_pow_100 :
  seq (3^100) = 2 * ∏ k in finset.range 100, (3^k + 1) :=
sorry

end find_b_3_pow_100_l633_633767


namespace max_value_of_function_l633_633455

noncomputable def max_value (x : ℝ) : ℝ := 3 * Real.sin x + 2

theorem max_value_of_function : 
  ∀ x : ℝ, (- (Real.pi / 2)) ≤ x ∧ x ≤ 0 → max_value x ≤ 2 :=
sorry

end max_value_of_function_l633_633455


namespace population_proof_l633_633238

theorem population_proof
  (P0 : ℕ) (r1 r2 : ℝ) (I E : ℕ)
  (hP0 : P0 = 15000)
  (hr1 : r1 = 0.10)
  (hr2 : r2 = 0.08)
  (hI : I = 100)
  (hE : E = 50) :
  let P1 := P0 + (r1 * P0).toInt + (I - E)
  let P2 := P1 + (r2 * P1).toInt + (I - E) in
  P2 = 17924 := 
by
  -- Definitions and calculations go here
  -- sorry is added to skip the proof
  sorry

end population_proof_l633_633238


namespace sum_floor_div_eq_l633_633758

theorem sum_floor_div_eq : 
  (∑ n in Finset.range 2014, (Int.floor (n / 2) + Int.floor (n / 3) + Int.floor (n / 6))) = 2027091 := 
by
  sorry

end sum_floor_div_eq_l633_633758


namespace insurance_compensation_zero_l633_633824

noncomputable def insured_amount : ℝ := 500000
noncomputable def deductible : ℝ := 0.01
noncomputable def actual_damage : ℝ := 4000

theorem insurance_compensation_zero :
  actual_damage < insured_amount * deductible → 0 = 0 := by
sorry

end insurance_compensation_zero_l633_633824


namespace distance_between_foci_l633_633211

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 36 = 0

-- Define the distance between the foci of the ellipse
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 2 * Real.sqrt 14.28 = 2 * Real.sqrt 14.28 :=
by sorry

end distance_between_foci_l633_633211


namespace numbers_contain_digit_five_l633_633288

theorem numbers_contain_digit_five :
  (finset.range 701).filter (λ n, ∃ d ∈ (n.digits 10), d = 5).card = 214 := 
sorry

end numbers_contain_digit_five_l633_633288


namespace repeating_decimal_sum_l633_633615

def repeating_decimal_to_fraction (d : ℕ) (n : ℕ) : ℚ := n / ((10^d) - 1)

theorem repeating_decimal_sum : 
  repeating_decimal_to_fraction 1 2 + repeating_decimal_to_fraction 2 2 + repeating_decimal_to_fraction 4 2 = 2474646 / 9999 := 
sorry

end repeating_decimal_sum_l633_633615


namespace angle_between_vectors_is_pi_over_4_l633_633265

variables {α : Type*} [inner_product_space ℝ α]

theorem angle_between_vectors_is_pi_over_4
  (a b : α)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (h1 : ∥a + b∥ = ∥a - b∥)
  (h2 : ∥a + b∥ = real.sqrt 2 * ∥a∥) :
  real.angle a (a + b) = real.pi / 4 :=
sorry

end angle_between_vectors_is_pi_over_4_l633_633265


namespace right_triangle_five_congruent_l633_633416

theorem right_triangle_five_congruent {A B C D F E G : Point} (h : right_triangle A B C)
  (h2 : seg_len A C = 2 * seg_len B C) :
  ∃ D F E G, congruent (triangle A B C) (triangle D E F) ∧ congruent (triangle A B C) (triangle F G E) ∧
  congruent (triangle A B C) (triangle G D F) ∧ congruent (triangle A B C) (triangle E D F) ∧ congruent (triangle A B C) (triangle B C D) :=
sorry

end right_triangle_five_congruent_l633_633416


namespace find_f_at_two_l633_633752

theorem find_f_at_two :
  ∃ (f : ℝ → ℝ), (∀ x > 0, differentiable ℝ f)
  ∧ (∀ x > 0, f x - x * deriv f x = f x - 1)
  ∧ (f 1 = 0)
  ∧ (f 2 = Real.log 2) :=
by
  sorry

end find_f_at_two_l633_633752


namespace pqst_value_l633_633227

noncomputable def solve_pqst (P Q S T : ℝ) (h1 : log 10 (P * S) + log 10 (P * T) = 3) 
         (h2 : log 10 (S * T) + log 10 (S * Q) = 4) 
         (h3 : log 10 (Q * P) + log 10 (Q * T) = 5) : Prop :=
  PQST = 10 ^ 4

theorem pqst_value (P Q S T : ℝ) (h1 : log 10 (P * S) + log 10 (P * T) = 3) 
         (h2 : log 10 (S * T) + log 10 (S * Q) = 4) 
         (h3 : log 10 (Q * P) + log 10 (Q * T) = 5) :
  solve_pqst P Q S T :=
begin
  -- This is a placeholder. The actual proof should be here.
  sorry
end

end pqst_value_l633_633227


namespace ellipse_hyperbola_tangent_l633_633446

theorem ellipse_hyperbola_tangent : ∀ m : ℝ, 
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y + 3)^2 = 1 → false) → 
  m = 8 / 9 :=
begin
  sorry,
end

end ellipse_hyperbola_tangent_l633_633446


namespace count_numbers_with_digit_five_l633_633276

theorem count_numbers_with_digit_five : 
  (finset.filter (λ n : ℕ, ∃ d : ℕ, d ∈ digits 10 n ∧ d = 5) (finset.range 701)).card = 133 := 
by 
  sorry

end count_numbers_with_digit_five_l633_633276


namespace at_least_one_inequality_holds_l633_633384

theorem at_least_one_inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y > 2) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l633_633384


namespace turnip_bag_weight_l633_633067

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l633_633067


namespace length_of_BC_l633_633351

theorem length_of_BC (A B C : Type) [triangle A B C] {AB AC BC : ℝ} 
    (hAB : AB = 3) (hAC : AC = 4) (hArea : (1 / 2) * AB * AC * Real.sin (angle A B C) = 3 * Real.sqrt 3) :
    BC = sqrt 13 ∨ BC = sqrt 37 :=
by
  sorry

end length_of_BC_l633_633351


namespace tractors_in_first_crew_l633_633189

theorem tractors_in_first_crew 
  (total_acres : ℕ)
  (first_crew_days : ℕ)
  (second_crew_days : ℕ)
  (second_crew_tractors : ℕ)
  (acres_per_day_per_tractor : ℕ)
  (h1 : total_acres = 1700)
  (h2 : first_crew_days = 2)
  (h3 : second_crew_days = 3)
  (h4 : second_crew_tractors = 7)
  (h5 : acres_per_day_per_tractor = 68) :
  ∃ x : ℕ, (2 * x + 21) * 68 = 1700 :=
by {
  use 2,
  have h6 : (2 * 2 + 21) * 68 = 1700, by norm_num,
  exact h6,
}

end tractors_in_first_crew_l633_633189


namespace numbers_with_digit_5_count_numbers_with_digit_5_l633_633271

theorem numbers_with_digit_5 (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 700) : 
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ d = 5) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m : ℕ, (n = m * 100 + d ∧ d = 5)) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m2 m1 : ℕ, (n = m2 * 1000 + m1 * 100 + d ∧ d = 5)) :=
sorry

theorem count_numbers_with_digit_5 : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 700 ∧ (numbers_with_digit_5 n sorry) }.to_finset.card = 214 := 
sorry

end numbers_with_digit_5_count_numbers_with_digit_5_l633_633271


namespace robot_routes_with_integer_distance_l633_633950

theorem robot_routes_with_integer_distance (k : ℕ) : 
  ∃ (routes : ℕ → bool), 
  (∀i < 3 * k, routes i = tt ∨ routes i = ff) ∧ 
  (dist (Σ i in finset.range (3 * k), if routes i then 2 ^ i else 0) (Σ i in finset.range (3 * k), if ¬(routes i) then 2 ^ i else 0) : ℕ) :=
begin
  sorry
end

end robot_routes_with_integer_distance_l633_633950


namespace find_x_l633_633700

theorem find_x 
  (h1 : ∀ (x : ℝ), sin (π / 2 - x) = - (sqrt 3 / 2)) 
  (h2 : π < x)
  (h3 : x < 2 * π) : x = 7 * π / 6 :=
sorry

end find_x_l633_633700


namespace fake_coin_weighings_l633_633895

theorem fake_coin_weighings {n k : ℕ}
  (h_pos : n > 1)
  (h_weighings : ∀ (f : ℕ) (is_fake : f < n) (lighter : bool), ∃ (weighings : list (fin n × fin n)), weighings.length ≤ k ∧ correct_result weighings f lighter) :
  k ≥ ⌈log(2 * n)⌉₊ :=
sorry

end fake_coin_weighings_l633_633895


namespace sum_of_f_values_l633_633247

def f (x : ℝ) : ℝ := (2 / (2^x + 1)) + Real.sin x

theorem sum_of_f_values : 
    f (-2) + f (-1) + f 0 + f 1 + f 2 = 5 := by
    sorry

end sum_of_f_values_l633_633247


namespace prime_square_remainders_l633_633579

theorem prime_square_remainders (p : ℕ) (hp : Nat.Prime p) (hgt : p > 5) : 
    {r | ∃ k : ℕ, p^2 = 180 * k + r}.finite ∧ 
    {r | ∃ k : ℕ, p^2 = 180 * k + r} = {1, 145} := 
by
  sorry

end prime_square_remainders_l633_633579


namespace round_trip_ticket_percentage_l633_633000

variable (P : ℝ) -- Denotes total number of passengers
variable (R : ℝ) -- Denotes number of round-trip ticket holders

-- Condition 1: 15% of passengers held round-trip tickets and took their cars aboard
def condition1 : Prop := 0.15 * P = 0.40 * R

-- Prove that 37.5% of the ship's passengers held round-trip tickets.
theorem round_trip_ticket_percentage (h1 : condition1 P R) : R / P = 0.375 :=
by
  sorry

end round_trip_ticket_percentage_l633_633000


namespace compute_expression_l633_633169

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 :=
by
  sorry

end compute_expression_l633_633169


namespace problem_solution_l633_633711

def p : Prop := ∀ x : ℝ, x > 0 → x + 1 / 2 > 2

def q : Prop := ∃ x₀ : ℝ, 2 ^ x₀ < 0

theorem problem_solution : (¬ p) ∧ q :=
by
sorry

end problem_solution_l633_633711


namespace arithmetic_sequence_a5_l633_633726

variable (a : ℕ → ℝ) (h : a 1 + a 9 = 10)

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : 
  a 5 = 5 :=
by sorry

end arithmetic_sequence_a5_l633_633726


namespace max_min_values_l633_633199

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem max_min_values : 
  (∃ a b, a ∈ set.interval (-1 : ℝ) 1 ∧ b ∈ set.interval (-1 : ℝ) 1 ∧ 
    (∀ x ∈ set.interval (-1 : ℝ) 1, f x ≤ f b) ∧ 
    (∀ x ∈ set.interval (-1 : ℝ) 1, f a ≤ f x) ∧ 
    f a = 1 ∧ f b = 5) :=
sorry

end max_min_values_l633_633199


namespace turnip_bag_weighs_l633_633093

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l633_633093


namespace no_snow_probability_l633_633845

theorem no_snow_probability (p_snow : ℚ) (h : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  (p_no_snow ^ 5 = 1 / 243) :=
by
  sorry

end no_snow_probability_l633_633845


namespace largest_prime_factor_of_sum_is_37_l633_633991

-- Definitions based on the conditions
def is_cyclic (seq : List ℕ) : Prop :=
  seq.head = seq.ilast && ∀ i, seq.nth i.last 3 = seq.nth (i + 1) % seq.length.head 3

def sum_seq (seq : List ℕ) : ℕ :=
  seq.foldl (· + ·) 0

theorem largest_prime_factor_of_sum_is_37 (seq : List ℕ) (hcyclic : is_cyclic seq) :
  ∃ p : ℕ, Nat.Prime p ∧ p = 37 ∧ p ∣ sum_seq seq :=
sorry

end largest_prime_factor_of_sum_is_37_l633_633991


namespace solve_my_operation_eq_l633_633601

def my_operation (a b : ℝ) : ℝ :=
  if a ≥ b then a^2 + b^2 else a^2 - b^2

theorem solve_my_operation_eq : 
  (my_operation 2 2 = 12) ∨ (my_operation (-4) 2 = 12) :=
by {
  sorry
}

end solve_my_operation_eq_l633_633601


namespace points_within_distance_l633_633213

theorem points_within_distance {a b : ℝ} (h1 : a = 3) (h2 : b = 4) (h3 : ∃ (points : Fin 7 → (ℝ × ℝ)), 
  ∀ i, (0 ≤ (points i).1 ∧ (points i).1 ≤ a) ∧ (0 ≤ (points i).2 ∧ (points i).2 ≤ b)) : 
  ∃ (i j : Fin 7), i ≠ j ∧ (dist (points i) (points j)) ≤ sqrt 5 := 
begin
  sorry
end

end points_within_distance_l633_633213


namespace differentiable_inequality_l633_633380

theorem differentiable_inequality 
  {a b : ℝ} 
  {f g : ℝ → ℝ} 
  (hdiff_f : DifferentiableOn ℝ f (Set.Icc a b))
  (hdiff_g : DifferentiableOn ℝ g (Set.Icc a b))
  (hderiv_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x > deriv g x)) :
  ∀ x ∈ Set.Ioo a b, f x + g a > g x + f a :=
by 
  sorry

end differentiable_inequality_l633_633380


namespace equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l633_633343

theorem equation_of_W (P : ℝ × ℝ) :
  let x := P.1 in let y := P.2 in
  |y| = real.sqrt (x^2 + (y - 1/2)^2) ↔ y = x^2 + 1/4 :=
by sorry

theorem rectangle_perimeter_greater_than_3sqrt3 {A B C D : ℝ × ℝ}
  (hA : A.2 = A.1^2 + 1/4) (hB : B.2 = B.1^2 + 1/4) (hC : C.2 = C.1^2 + 1/4)
  (hAB_perp_BC : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) :
  2 * ((real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) + (real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2))) > 3 * real.sqrt 3 :=
by sorry

end equation_of_W_rectangle_perimeter_greater_than_3sqrt3_l633_633343


namespace rahim_average_price_per_book_l633_633790

noncomputable section

open BigOperators

def store_A_price_per_book : ℝ := 
  let original_total := 1600
  let discount := original_total * 0.15
  let discounted_total := original_total - discount
  let sales_tax := discounted_total * 0.05
  let final_total := discounted_total + sales_tax
  final_total / 25

def store_B_price_per_book : ℝ := 
  let original_total := 3200
  let effective_books_paid := 35 - (35 / 4)
  original_total / effective_books_paid

def store_C_price_per_book : ℝ := 
  let original_total := 3800
  let discount := 0.10 * (4 * (original_total / 40))
  let discounted_total := original_total - discount
  let service_charge := discounted_total * 0.07
  let final_total := discounted_total + service_charge
  final_total / 40

def store_D_price_per_book : ℝ := 
  let original_total := 2400
  let discount := 0.50 * (original_total / 30)
  let discounted_total := original_total - discount
  let sales_tax := discounted_total * 0.06
  let final_total := discounted_total + sales_tax
  final_total / 30

def store_E_price_per_book : ℝ := 
  let original_total := 1800
  let discount := original_total * 0.08
  let discounted_total := original_total - discount
  let additional_fee := discounted_total * 0.04
  let final_total := discounted_total + additional_fee
  final_total / 20

def total_books : ℝ := 25 + 35 + 40 + 30 + 20

def total_amount : ℝ := 
  store_A_price_per_book * 25 + 
  store_B_price_per_book * 35 + 
  store_C_price_per_book * 40 + 
  store_D_price_per_book * 30 + 
  store_E_price_per_book * 20

def average_price_per_book : ℝ := total_amount / total_books

theorem rahim_average_price_per_book : average_price_per_book = 85.85 :=
sorry

end rahim_average_price_per_book_l633_633790


namespace probability_same_group_l633_633878

theorem probability_same_group (cards : Finset ℕ) (n : ℕ) (a b : ℕ) : 
  (cards = {1, 2, ..., 20}) → 
  (n = 4) → 
  (a = 5) → 
  (b = 14) → 
  ∃ p : ℚ, p = 7 / 51 :=
by
  intros 
  sorry

end probability_same_group_l633_633878


namespace count_5_in_range_1_to_700_l633_633284

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  n.digits 10 |>.contains d

def count_numbers_with_digit (d : ℕ) (m : ℕ) : ℕ :=
  (List.range' 1 m) |>.filter (contains_digit d) |>.length

theorem count_5_in_range_1_to_700 : count_numbers_with_digit 5 700 = 214 := by
  sorry

end count_5_in_range_1_to_700_l633_633284


namespace no_mems_are_veens_l633_633261

variable (Mem En Veen : Type)
variable (mem : Mem → En)
variable (noveens : ∀ (e : En), ¬ Veen e)

theorem no_mems_are_veens (m : Mem) : ¬ Veen (mem m) := by
  exact noveens (mem m)

end no_mems_are_veens_l633_633261


namespace smallest_n_sum_lt_zero_l633_633702

variables {a : ℕ → ℚ}
variables (h_arith_seq : ∃ d : ℚ, ∀ n : ℕ, a n = a 1 + (n - 1) * d)
variables (a1_pos : a 1 > 0)
variables (h_sum_pos: a 2022 + a 2023 > 0)
variables (h_prod_neg: a 2022 * a 2023 < 0)

theorem smallest_n_sum_lt_zero : ∃ n : ℕ, (n = 4045) ∧ (∑ i in finset.range n, a i) < 0 := sorry

end smallest_n_sum_lt_zero_l633_633702


namespace wire_temperature_l633_633743

def σ : ℝ := 5.67e-8 -- Stefan-Boltzmann constant
def U : ℝ := 220    -- Applied Voltage in Volts
def I : ℝ := 5      -- Current in Amperes
def L : ℝ := 0.5    -- Length of the wire in meters
def D : ℝ := 2e-3   -- Diameter of the wire in meters

def S : ℝ := L * π * D       -- Surface area of the wire
def P : ℝ := U * I           -- Power dissipated by current
def P_thermal : ℝ := σ * (1576^4) * S  -- Power of thermal radiation at calculated temperature

theorem wire_temperature :
  P = P_thermal :=
by
  sorry

end wire_temperature_l633_633743


namespace turnips_bag_l633_633052

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l633_633052


namespace maxwell_distance_traveled_l633_633506

theorem maxwell_distance_traveled
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (meeting_time : ℕ)
  (h1 : distance_between_homes = 72)
  (h2 : maxwell_speed = 6)
  (h3 : brad_speed = 12)
  (h4 : meeting_time = distance_between_homes / (maxwell_speed + brad_speed)) :
  maxwell_speed * meeting_time = 24 :=
by
  sorry

end maxwell_distance_traveled_l633_633506


namespace geometric_sequence_S4_l633_633643

noncomputable def geometric_sequence_sum (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  let q := (a_n 2 / a_n 1) in
  a_n 1 * ((1 - q^n) / (1 - q))

theorem geometric_sequence_S4 
  (a_n : ℕ → ℝ) 
  (h_geom : ∀ n, a_n (n + 1) = a_n 0 * ((a_n 2 / a_n 1) ^ n)) 
  (h1 : (a_n 2) * (a_n 3) = 2 * (a_n 1)) 
  (h2 : (1 / 2 * a_n 4 + a_n 7) / 2 = 5 / 8) : 
  geometric_sequence_sum a_n 4 = 30 := 
by {
  sorry
}

end geometric_sequence_S4_l633_633643


namespace problem_proof_l633_633347

def curve_C1 (t : ℝ) : ℝ × ℝ :=
  (t + 1 / t, 2 * (t - 1 / t))

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Math.cos θ, ρ * Math.sin θ)

noncomputable def curve_C2_rectangular_eq : Prop :=
  ∀ (ρ θ : ℝ), ρ = 1 / (Math.sin θ - 3 * Math.cos θ) → (3 * (ρ * Math.cos θ) - (ρ * Math.sin θ) + 1 = 0)

noncomputable def min_distance_PQ (t : ℝ) : ℝ :=
  Real.abs ((t + 5 / t + 1) / Real.sqrt 10)

noncomputable def min_distance_value : ℝ :=
  (10 * Real.sqrt 2 - Real.sqrt 10) / 10

noncomputable def coordinates_of_P : ℝ × ℝ :=
  (-6 * Real.sqrt 5 / 5, -8 * Real.sqrt 5 / 5)

theorem problem_proof :
  (curve_C2_rectangular_eq) ∧ 
  (min_distance_PQ (-Real.sqrt 5) = min_distance_value) ∧ 
  (curve_C1 (-Real.sqrt 5) = coordinates_of_P) :=
by sorry

end problem_proof_l633_633347


namespace measure_of_PB_l633_633728

-- Definitions for given conditions
variables {C : Type*} [metric_space C] [normed_group C] [normed_space ℝ C]

variables {x : ℝ}
variables (M P A B C D : C) -- Points
variable  (AB AC CD : ℝ)    -- Chords
variables (AP MP PB : ℝ)    -- Segments

-- Given Conditions:
-- M is the midpoint of arc CABD
def is_midpoint_arc_CABD (M C A B D : C) : Prop := sorry

-- MP is perpendicular to AB at P
def is_perpendicular (MP AP PB : ℝ) : Prop := sorry

-- Chord AC measures 2x
def chord_AC (AC : ℝ) (x : ℝ) := AC = 2 * x

-- Segment AP measures 2x + 1
def segment_AP (AP : ℝ) (x : ℝ) := AP = 2 * x + 1

-- Chord CD measures x + 2
def chord_CD (CD : ℝ) (x : ℝ) := CD = x + 2

-- Proof Statement
theorem measure_of_PB 
  (M_is_midpoint : is_midpoint_arc_CABD M C A B D)
  (MP_perpendicular : is_perpendicular MP AP PB)
  (chord_AC_len : chord_AC AC x)
  (segment_AP_len : segment_AP AP x)
  (chord_CD_len : chord_CD CD x) :
  PB = 2 * x + 1 :=
sorry

end measure_of_PB_l633_633728


namespace range_of_f_l633_633714

noncomputable def f (x : ℝ) : ℝ := 1 + 2 / (x - 1)

theorem range_of_f : 
  (set.Ioo (5 / 3 : ℝ) 3) = set.range (λ x, f x) (set.Ico 2 4) := 
by
  sorry

end range_of_f_l633_633714


namespace find_x_value_l633_633809

theorem find_x_value 
(h1 : ∠BAD = 43)
(h2 : ∠DAC = 29)
(h3 : 90 + 90 + ∠BAD + ∠DAC + x = 360) :
x = 108 := by
  sorry

end find_x_value_l633_633809


namespace line_intersects_circle_but_not_center_l633_633210

theorem line_intersects_circle_but_not_center {k : ℝ} :
  ∀ k : ℝ, ∃ p : ℝ × ℝ, (p.1 ^ 2 + p.2 ^ 2 = 2) ∧ (p.2 = k * p.1 + 1) ∧ (p ≠ (0, 0)) :=
by
  intro k
  -- The proof is inevitable here as by assumption the line always intersects.
  have h₁ : ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1 := sorry
  use (classical.some h₁)
  have h₂ : classical.some h₁ ≠ (0, 0) := sorry
  tauto

end line_intersects_circle_but_not_center_l633_633210


namespace find_m_eq_neg2_l633_633682

noncomputable def find_m (m : ℝ) : Prop :=
  let p : ℕ → ℝ := λ x, x^2 + 2*x - 3 in
  let l1 : ℕ → ℝ := λ x, -x + m in
  let A : (ℝ × ℝ) := (x1, p x1) in
  let C : (ℝ × ℝ) := (x2, p x2) in
  let l2 : ℕ → ℝ := λ x, -x - m in
  let B : (ℝ × ℝ) := (x3, p x3) in
  let D : (ℝ × ℝ) := (x4, p x4) in
  let AC : ℝ := dist A C in
  let BD : ℝ := dist B D in
  (AC * BD = 26) → (m = -2)

theorem find_m_eq_neg2 (m : ℝ) (h : find_m m) :
  m = -2 := 
  by
    sorry

end find_m_eq_neg2_l633_633682


namespace average_time_proof_l633_633983

-- Given conditions
def casey_time : ℝ := 6
def zendaya_additional_time_factor : ℝ := 1/3

-- Derived conditions from the problem
def zendaya_time : ℝ := casey_time + (zendaya_additional_time_factor * casey_time)
def combined_time : ℝ := casey_time + zendaya_time
def num_people : ℝ := 2

-- Statement to prove
theorem average_time_proof : (combined_time / num_people) = 7 := by
  sorry

end average_time_proof_l633_633983


namespace smallest_non_factor_product_of_48_l633_633890

def factors_of (n : ℕ) : set ℕ := {d | d ∣ n}

theorem smallest_non_factor_product_of_48 :
  ∀ (a b : ℕ), a ≠ b ∧ a ∈ factors_of 48 ∧ b ∈ factors_of 48 ∧ ¬ (a * b) ∈ factors_of 48 → a * b = 18 :=
by
  intros a b h
  sorry

end smallest_non_factor_product_of_48_l633_633890


namespace max_value_sqrt_expr_l633_633385

theorem max_value_sqrt_expr
  (x y z : ℝ)
  (h_sum : x + y + z = 3)
  (h_x : x ≥ -1)
  (h_y : y ≥ -2)
  (h_z : z ≥ -4) :
  sqrt (4 * x + 4) + sqrt (4 * y + 8) + sqrt (4 * z + 16) ≤ 2 * sqrt 30 := 
sorry

end max_value_sqrt_expr_l633_633385


namespace original_volume_l633_633475

variable {π : Real} (r h : Real)

theorem original_volume (hπ : π ≠ 0) (hr : r ≠ 0) (hh : h ≠ 0) (condition : 3 * π * r^2 * h = 180) : π * r^2 * h = 60 := by
  sorry

end original_volume_l633_633475


namespace turnip_bag_weight_l633_633069

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l633_633069


namespace rate_per_kg_of_grapes_l633_633693

theorem rate_per_kg_of_grapes
  (G : ℕ)
  (Harkamal_paid : 1135)
  (grape_cost: 8 * G)
  (mango_cost: 9 * 55) 
  (total_cost : 8 * G + 9 * 55 = 1135) : 
  G = 80 :=
by
  sorry

end rate_per_kg_of_grapes_l633_633693


namespace pascal_triangle_tenth_number_l633_633902

theorem pascal_triangle_tenth_number :
  let n := 50 in
  nat.choose n 9 = 2586948580 := 
by
  let n := 50
  have h : nat.choose n 9 = 2586948580 := sorry
  exact h

end pascal_triangle_tenth_number_l633_633902


namespace count_numbers_with_digit_five_l633_633279

theorem count_numbers_with_digit_five : 
  (finset.filter (λ n : ℕ, ∃ d : ℕ, d ∈ digits 10 n ∧ d = 5) (finset.range 701)).card = 133 := 
by 
  sorry

end count_numbers_with_digit_five_l633_633279


namespace turnip_weight_possible_l633_633081

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l633_633081


namespace domain_log_function_l633_633814

open Real

def quadratic_term (x : ℝ) : ℝ := 4 - 3 * x - x^2

def valid_argument (x : ℝ) : Prop := quadratic_term x > 0

theorem domain_log_function : { x : ℝ | valid_argument x } = Set.Ioo (-4 : ℝ) (1 : ℝ) :=
by
  sorry

end domain_log_function_l633_633814


namespace value_of_e_l633_633503

theorem value_of_e
  (a b c d e : ℤ)
  (h1 : b = a + 2)
  (h2 : c = a + 4)
  (h3 : d = a + 6)
  (h4 : e = a + 8)
  (h5 : a + c = 146) :
  e = 79 :=
  by sorry

end value_of_e_l633_633503


namespace f_cos_x_l633_633707

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 2 - Real.cos x ^ 2) : f (Real.cos x) = 2 + Real.sin x ^ 2 := by
  sorry

end f_cos_x_l633_633707


namespace probability_not_A_l633_633791

noncomputable def P (event : Type) : ℝ := sorry

def A : Type := sorry
def B : Type := sorry
def C : Type := sorry

axiom P_A : P A = 0.65
axiom P_B : P B = 0.2
axiom P_C : P C = 0.1

theorem probability_not_A : P (λ (x : Type), x ≠ A) = 0.35 :=
by
  have complement_A : P (λ (x : Type), x ≠ A) = 1 - P A := sorry
  rw [complement_A, P_A]
  norm_num

end probability_not_A_l633_633791


namespace numbers_with_digit_5_count_numbers_with_digit_5_l633_633273

theorem numbers_with_digit_5 (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 700) : 
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ d = 5) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m : ℕ, (n = m * 100 + d ∧ d = 5)) ∨
  (∃ d : ℕ, (d ∈ {d | 0 ≤ d ∧ d < 10} ∧ ∃ k : ℕ, n = k * 10 + d) ∧ ∃ m2 m1 : ℕ, (n = m2 * 1000 + m1 * 100 + d ∧ d = 5)) :=
sorry

theorem count_numbers_with_digit_5 : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 700 ∧ (numbers_with_digit_5 n sorry) }.to_finset.card = 214 := 
sorry

end numbers_with_digit_5_count_numbers_with_digit_5_l633_633273


namespace points_with_equal_distances_form_sides_of_triangle_l633_633632

noncomputable def distances_form_triangle (A B C M : Point) : Prop :=
  let x := distance M (line_through B C)
  let y := distance M (line_through A C)
  let z := distance M (line_through A B)
  x < y + z ∧ y < z + x ∧ z < x + y

noncomputable def is_in_interior_of_angle_bisectors_triangle (A B C M : Point) : Prop :=
  let bisector_A := angle_bisector A B C
  let bisector_B := angle_bisector B A C
  let bisector_C := angle_bisector C A B
  let intersection_A := bisector_A.intersect_side (line_through B C)
  let intersection_B := bisector_B.intersect_side (line_through A C)
  let intersection_C := bisector_C.intersect_side (line_through A B)
  is_point_in_triangle M intersection_A intersection_B intersection_C

theorem points_with_equal_distances_form_sides_of_triangle {A B C M : Point} :
  ( ∃ M : Point, is_in_interior_of_angle_bisectors_triangle A B C M ) ↔ 
  distances_form_triangle A B C M :=
sorry

end points_with_equal_distances_form_sides_of_triangle_l633_633632


namespace correct_box_choice_l633_633920

-- Define the boxes and their inscriptions as propositions
def golden_incription := "The dagger is in this box."
def silver_incription := "This box is empty."
def lead_incription := "No more than one of the three inscriptions on the boxes is true."

-- Define the actual locations of the items for the suitor
def contains_dagger (box : String) : Prop :=
  match box with
  | "golden" => True
  | "silver" => False
  | "lead" => False
  | _ => False

def empty (box : String) : Prop :=
  match box with
  | "golden" => False
  | "silver" => True
  | "lead" => False
  | _ => False

theorem correct_box_choice : ∃ box : String, empty box ∧
  ((contains_dagger "golden" ∧ ¬contains_dagger "silver" ∧ ¬contains_dagger "lead") ↔ golden_incription ∧ ¬silver_incription ∧ ¬lead_incription) :=
by
  sorry

end correct_box_choice_l633_633920


namespace value_of_a_l633_633258

theorem value_of_a (M : Set ℝ) (N : Set ℝ) (a : ℝ) 
  (hM : M = {-1, 0, 1, 2}) (hN : N = {x | x^2 - a * x < 0}) 
  (hIntersect : M ∩ N = {1, 2}) : 
  a = 3 := 
sorry

end value_of_a_l633_633258


namespace symmetric_trapezoid_construction_possible_l633_633174

-- Define lengths of legs and distance from intersection point
variables (a b : ℝ)

-- Symmetric trapezoid feasibility condition
theorem symmetric_trapezoid_construction_possible : 3 * b > 2 * a := sorry

end symmetric_trapezoid_construction_possible_l633_633174


namespace multiplication_problem_l633_633158

theorem multiplication_problem :
  250 * 24.98 * 2.498 * 1250 = 19484012.5 := by
  sorry

end multiplication_problem_l633_633158


namespace pyramid_volume_l633_633542

-- Definitions based on the given conditions
def AB : ℝ := 15
def AD : ℝ := 8
def Area_Δ_ABE : ℝ := 120
def Area_Δ_CDE : ℝ := 64
def h : ℝ := 16
def Base_Area : ℝ := AB * AD

-- Statement to prove the volume of the pyramid is 640
theorem pyramid_volume : (1 / 3) * Base_Area * h = 640 :=
sorry

end pyramid_volume_l633_633542


namespace non_honda_red_percentage_l633_633319

-- Define the conditions
def total_cars : ℕ := 900
def honda_percentage_red : ℝ := 0.90
def total_percentage_red : ℝ := 0.60
def honda_cars : ℕ := 500

-- The statement to prove
theorem non_honda_red_percentage : 
  (0.60 * 900 - 0.90 * 500) / (900 - 500) * 100 = 22.5 := 
  by sorry

end non_honda_red_percentage_l633_633319


namespace polynomial_solution_l633_633193

variable (P : ℝ → ℝ → ℝ)

theorem polynomial_solution :
  (∀ x y : ℝ, P (x + y) (x - y) = 2 * P x y) →
  (∃ b c d : ℝ, ∀ x y : ℝ, P x y = b * x^2 + c * x * y + d * y^2) :=
by
  sorry

end polynomial_solution_l633_633193


namespace base7_digits_l633_633375

theorem base7_digits (D E F : ℕ) (h1 : D ≠ 0) (h2 : E ≠ 0) (h3 : F ≠ 0) (h4 : D < 7) (h5 : E < 7) (h6 : F < 7)
  (h_diff1 : D ≠ E) (h_diff2 : D ≠ F) (h_diff3 : E ≠ F)
  (h_eq : (49 * D + 7 * E + F) + (49 * E + 7 * F + D) + (49 * F + 7 * D + E) = 400 * D) :
  E + F = 6 :=
by
  sorry

end base7_digits_l633_633375


namespace perfect_number_divisibility_l633_633110

theorem perfect_number_divisibility (n : ℕ) (h1 : perfect_number n) (h2 : 6 < n) (h3 : 3 ∣ n) : 9 ∣ n :=
sorry

-- Definitions depending on what we need for the proof, these are placeholders.
def perfect_number (n : ℕ) : Prop :=
  ∑ m in (finset.range n).filter (λ d, d ∣ n), m = n

end perfect_number_divisibility_l633_633110


namespace turnip_bag_weights_l633_633076

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l633_633076


namespace number_of_restaurants_l633_633030

theorem number_of_restaurants
  (total_units : ℕ)
  (residential_units : ℕ)
  (non_residential_units : ℕ)
  (restaurants : ℕ)
  (h1 : total_units = 300)
  (h2 : residential_units = total_units / 2)
  (h3 : non_residential_units = total_units - residential_units)
  (h4 : restaurants = non_residential_units / 2)
  : restaurants = 75 := 
by
  sorry

end number_of_restaurants_l633_633030


namespace area_KMLN_eq_ADK_plus_BCL_l633_633408

variable (A B C D M N K L : Type)
variables [inst : add_comm_group K] [vector_space ℝ K]

-- Assume basic geometric collinear and coplanar constraints
#check (affine_space ℝ K)

-- Conditions of the problem
variable (AM_MB_eq_CN_ND : ratio (AM / MB) = ratio (CN / ND))

-- Collinear points and intersections
variable (collinear_ABC : collinear {A, B, C})
variable (intersect_AN_DM_K : intersection (AN) (DM) = {K})
variable (intersect_BN_CM_L : intersection (BN) (CM) = {L})

-- To prove
theorem area_KMLN_eq_ADK_plus_BCL :
  area (quadrilateral K M L N) = area (triangle A D K) + area (triangle B C L) := sorry

end area_KMLN_eq_ADK_plus_BCL_l633_633408


namespace scientists_nobel_greater_than_not_nobel_by_three_l633_633015

-- Definitions of the given conditions
def total_scientists := 50
def wolf_prize_laureates := 31
def nobel_prize_laureates := 25
def wolf_and_nobel_laureates := 14

-- Derived quantities
def no_wolf_prize := total_scientists - wolf_prize_laureates
def only_wolf_prize := wolf_prize_laureates - wolf_and_nobel_laureates
def only_nobel_prize := nobel_prize_laureates - wolf_and_nobel_laureates
def nobel_no_wolf := only_nobel_prize
def no_wolf_no_nobel := no_wolf_prize - nobel_no_wolf
def difference := nobel_no_wolf - no_wolf_no_nobel

-- The theorem to be proved
theorem scientists_nobel_greater_than_not_nobel_by_three :
  difference = 3 := 
sorry

end scientists_nobel_greater_than_not_nobel_by_three_l633_633015


namespace number_of_cutlery_pieces_added_l633_633879

-- Define the initial conditions
def forks_initial := 6
def knives_initial := forks_initial + 9
def spoons_initial := 2 * knives_initial
def teaspoons_initial := forks_initial / 2
def total_initial_cutlery := forks_initial + knives_initial + spoons_initial + teaspoons_initial
def total_final_cutlery := 62

-- Define the total number of cutlery pieces added
def cutlery_added := total_final_cutlery - total_initial_cutlery

-- Define the theorem to prove
theorem number_of_cutlery_pieces_added : cutlery_added = 8 := by
  sorry

end number_of_cutlery_pieces_added_l633_633879


namespace sum_preservation_impossible_product_preservation_possible_l633_633742

open Rational

-- Define the coloring type
inductive Color
| Red
| Blue

-- Assume rational numbers are colored
variable (color : ℚ → Color)

-- Sum is defined for positive rational numbers (assuming closure under addition)
noncomputable def isSumColorPreserved : Prop :=
  ∀ (a b : ℚ), a > 0 → b > 0 → color a = Color.Red → color b = Color.Red → color (a + b) = Color.Red

noncomputable def isSumDisjointColorPreserved : Prop :=
  ∀ (a b : ℚ), a > 0 → b > 0 → color a = Color.Blue → color b = Color.Blue → color (a + b) = Color.Blue

-- Product is defined for positive rational numbers (assuming closure under multiplication)
noncomputable def isProductColorPreserved : Prop :=
  ∀ (a b : ℚ), a > 0 → b > 0 → color a = Color.Red → color b = Color.Red → color (a * b) = Color.Red

noncomputable def isProductDisjointColorPreserved : Prop :=
  ∀ (a b : ℚ), a > 0 → b > 0 → color a = Color.Blue → color b = Color.Blue → color (a * b) = Color.Blue

-- Statement for part (a)
theorem sum_preservation_impossible :
  ¬ (∃ (color : ℚ → Color), (isSumColorPreserved color) ∧ (isSumDisjointColorPreserved color)) := 
  sorry

-- Statement for part (b)
theorem product_preservation_possible :
  ∃ (color : ℚ → Color), (isProductColorPreserved color) ∧ (isProductDisjointColorPreserved color) :=
  sorry

end sum_preservation_impossible_product_preservation_possible_l633_633742


namespace combined_time_in_pool_l633_633972

theorem combined_time_in_pool : 
    ∀ (Jerry_time Elaine_time George_time Kramer_time : ℕ), 
    Jerry_time = 3 →
    Elaine_time = 2 * Jerry_time →
    George_time = Elaine_time / 3 →
    Kramer_time = 0 →
    Jerry_time + Elaine_time + George_time + Kramer_time = 11 :=
by 
  intros Jerry_time Elaine_time George_time Kramer_time hJerry hElaine hGeorge hKramer
  sorry

end combined_time_in_pool_l633_633972


namespace identify_opposite_pair_l633_633151

-- Define what it means for two numbers to be opposites
def are_opposite (a b : ℝ) : Prop :=
  a = -b

-- Define the given pairs in the problem
def pair_A : ℝ × ℝ := (-(-2), 2)
def pair_B : ℝ × ℝ := (+(-3), -(+3))
def pair_C : ℝ × ℝ := (1/2, -2)
def pair_D : ℝ × ℝ := (-(-5), -abs(+5))

-- State the theorem to be proved
theorem identify_opposite_pair : 
  ¬ are_opposite pair_A.1 pair_A.2 ∧
  ¬ are_opposite pair_B.1 pair_B.2 ∧
  ¬ are_opposite pair_C.1 pair_C.2 ∧
  are_opposite pair_D.1 pair_D.2 :=
sorry

end identify_opposite_pair_l633_633151


namespace complex_multiplication_identity_l633_633989

theorem complex_multiplication_identity :
  (λ i : ℂ, (3 - 4 * i) * (-7 + 2 * i)) (complex.i) = -13 + 34 * complex.i :=
by
  have h₁ : complex.i^2 = -1 := by
    exact I_sq_eq_neg_one
  sorry

end complex_multiplication_identity_l633_633989


namespace total_oranges_picked_l633_633177

theorem total_oranges_picked :
  let del_daily_orange := 23
  let del_days := 2
  let juan_oranges := 61
  let del_total_oranges := del_daily_orange * del_days
  let total_oranges := del_total_oranges + juan_oranges
  total_oranges = 107 :=
by 
  let del_daily_orange := 23
  let del_days := 2
  let juan_oranges := 61
  let del_total_oranges := del_daily_orange * del_days
  let total_oranges := del_total_oranges + juan_oranges
  show total_oranges = 107
  from sorry

end total_oranges_picked_l633_633177


namespace ED_perp_EF_l633_633642

-- Define the necessary points and circles
variables {O K A B C D P E F : Type}

-- Define the circles and points involve with necessary constraints
structure Circle (center : Type) where
  radius : Type

structure Point (x y : Float)

constant circle_O : Circle O
constant point_K : K
constant point_A : A
constant point_B : B
constant point_C : C
constant point_D : D
constant point_P : P
constant point_E : E
constant point_F : F
constant line_segment_AB : LineSegment A B
constant line_segment_OC : LineSegment O C

-- Conditions
axiom tangent_KA : Tangent KA circle_O A
axiom tangent_KB : Tangent KB circle_O B
axiom circle_K : Circle K
axiom radius_KA : circle_K.radius = distance K A
axiom point_C_on_AB : OnSegment C line_segment_AB
axiom intersection_OC_D_P : Intersect OC circle_K D P
axiom P_on_circle_O : OnCircle P circle_O
axiom intersect_tangent_PC_E : Tangent PC circle_O E
axiom intersect_tangent_PD_F : Tangent PD circle_O F

-- The proof statement
theorem ED_perp_EF : Perpendicular (Line E D) (Line E F) :=
  sorry

end ED_perp_EF_l633_633642


namespace number_of_sides_of_regular_polygon_l633_633544

theorem number_of_sides_of_regular_polygon (P s n : ℕ) (hP : P = 150) (hs : s = 15) (hP_formula : P = n * s) : n = 10 :=
  by {
    -- proof goes here
    sorry
  }

end number_of_sides_of_regular_polygon_l633_633544


namespace diagonal_not_perpendicular_l633_633729

open Real

theorem diagonal_not_perpendicular (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_a_ne_b : a ≠ b) (h_c_ne_d : c ≠ d) (h_a_ne_c : a ≠ c) (h_b_ne_d : b ≠ d): 
  ¬ ((d - b) * (b - a) = - (c - a) * (d - c)) :=
by
  sorry

end diagonal_not_perpendicular_l633_633729


namespace area_of_circle_O_l633_633784

open Real EuclideanGeometry

noncomputable def area_of_circle (O : Circle) (D E F P : Point) (h1 : D ∈ O.circumference)
  (h2 : E ∈ O.circumference) (h3 : F ∈ O.circumference) (h4 : Tangent O D P)
  (h5 : OnRay P E F) (PD : Real) (PF : Real) (angle_FPD : Real) : Real :=
  let PD := 4
  let PF := 2
  let angle_FPD := π / 3 -- 60 degrees in radians
  12 * π

theorem area_of_circle_O {O : Circle} {D E F P : Point} 
  (h1 : D ∈ O.circumference)
  (h2 : E ∈ O.circumference)
  (h3 : F ∈ O.circumference)
  (h4 : Tangent O D P)
  (h5 : OnRay P E F)
  (PD := 4)
  (PF := 2)
  (angle_FPD := π / 3) : area_of_circle O D E F P h1 h2 h3 h4 h5 PD PF angle_FPD = 12 * π := by 
  sorry

end area_of_circle_O_l633_633784


namespace not_snow_probability_l633_633868

theorem not_snow_probability :
  let p_snow : ℚ := 2 / 3,
      p_not_snow : ℚ := 1 - p_snow in
  (p_not_snow ^ 5) = 1 / 243 := by
  sorry

end not_snow_probability_l633_633868


namespace simplify_sqrt_subtraction_l633_633423

theorem simplify_sqrt_subtraction : sqrt 20 - sqrt 5 = sqrt 5 :=
by
  sorry -- Proof will be provided here

end simplify_sqrt_subtraction_l633_633423


namespace abs_sum_coeffs_binom_l633_633753

theorem abs_sum_coeffs_binom (a : ℕ → ℤ) :
  (∀ n, (2 - x)^6 = ∑ i in finset.range 7, a i * x^i) →
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 665) :=
by
  sorry

end abs_sum_coeffs_binom_l633_633753


namespace same_side_probability_l633_633905

theorem same_side_probability :
  (∀ (n : ℕ), n = 4 → ∀ (p : ℚ), p = 1/2 → (p ^ n = 1/16)) :=
by
  sorry

end same_side_probability_l633_633905


namespace turnip_bag_weight_l633_633057

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l633_633057


namespace turnip_bag_weights_l633_633077

theorem turnip_bag_weights :
  ∃ (T : ℕ), (T = 13 ∨ T = 16) ∧ 
  (∀ T', T' ≠ T → (
    (2 * total_weight_of_has_weight_1.not_turnip T' O + O = 106 - T') → 
    let turnip_condition := 106 - T in
    turnip_condition % 3 = 0)) ∧
  ∀ (bag_weights : List ℕ) (other : bag_weights = [13, 15, 16, 17, 21, 24] ∧ 
                                          List.length bag_weights = 6),
  True := by sorry

end turnip_bag_weights_l633_633077


namespace joe_notebooks_l633_633747

-- Define the conditions
variable (N : ℝ) 
variable (cost_notebook cost_book money_given money_left : ℝ)
variable (books_bought : ℝ)

-- Set the conditions
def initial_conditions : Prop :=
  cost_notebook = 4 ∧ cost_book = 7 ∧
  money_given = 56 ∧ books_bought = 2 ∧
  money_left = 14

-- Define the proof problem
theorem joe_notebooks (h : initial_conditions) : N = 7 :=
by 
  -- Unfold the conditions for easier manipulation
  unfold initial_conditions at h
  -- Use the conditions to set up the equation and solve for N
  cases h with h₁ h_rest
  cases h₁ with h₀₁ h_rest
  cases h_rest with h₀₂ h_rest
  cases h_rest with h₀₃ h_rest
  cases h_rest with h₀₄ h₀₅
  have cost_total := h₀₁ * N + h₀₄ * h₀₂,  
  have money_spent := h₀₃ - h₀₅,
  have equation := cost_total = money_spent,
  simp only [h₀₁, h₀₂, h₀₃, h₀₄, h₀₅] at equation,
  linarith,

end joe_notebooks_l633_633747


namespace probability_two_red_two_blue_l633_633024

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose 20 4
  let choose_2_red := Nat.choose 12 2
  let choose_2_blue := Nat.choose 8 2
  (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690 := by
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose total_marbles 4
  let choose_2_red := Nat.choose total_red 2
  let choose_2_blue := Nat.choose total_blue 2
  show (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690
  sorry

end probability_two_red_two_blue_l633_633024


namespace minimum_value_of_f_l633_633220

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem minimum_value_of_f : ∃ x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end minimum_value_of_f_l633_633220


namespace range_of_m_l633_633236

-- Define the function g and its derivative
variable (g : ℝ → ℝ)
variable (g' : ℝ → ℝ)
variable (deriv_g : ∀ x : ℝ, deriv g x = g' x)

-- Conditions provided in the problem
variable (cond1 : ∀ x : ℝ, g(3 * x - 2) ≥ Real.exp(x - 3) * g(2 * x + 1))
variable (cond2 : ∀ x : ℝ, g(x) / g(-x) = exp(2 * x))
variable (cond3 : ∀ x ≥ 0, g' x > g x)

-- Statement of the problem
theorem range_of_m : {m : ℝ | g (3 * m - 2) ≥ Real.exp(m - 3) * g (2 * m + 1)} = {m : ℝ | m ≤ (1 / 5) ∨ 3 ≤ m} :=
by
  sorry

end range_of_m_l633_633236


namespace turnip_bag_weight_l633_633066

/-- Given six bags with weights 13, 15, 16, 17, 21, and 24 kg,
one bag contains turnips, and the others contain either onions or carrots.
The total weight of the carrots equals twice the total weight of the onions.
Prove that the bag containing turnips can weigh either 13 kg or 16 kg. -/
theorem turnip_bag_weight (ws : list ℕ) (T : ℕ) (O C : ℕ) (h_ws : ws = [13, 15, 16, 17, 21, 24])
  (h_sum : ws.sum = 106) (h_co : C = 2 * O) (h_weight : C + O = 106 - T) :
  T = 13 ∨ T = 16 :=
sorry

end turnip_bag_weight_l633_633066


namespace right_triangle_of_sine_cosine_identity_l633_633737

theorem right_triangle_of_sine_cosine_identity (A B C : ℝ) (h1 : sin A * cos B = 1 - cos A * sin B) (h2 : A + B + C = Real.pi) : C = Real.pi / 2 :=
by
  sorry

end right_triangle_of_sine_cosine_identity_l633_633737


namespace find_x_l633_633684

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b : ℝ × ℝ := (1, sqrt 3)
def projection (a b : ℝ × ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2) / (sqrt (b.1 ^ 2 + b.2 ^ 2))

theorem find_x (x : ℝ) (h : projection (vector_a x) vector_b = sqrt 3) : x = sqrt 3 := by
  sorry

end find_x_l633_633684


namespace masha_friends_inconsistent_l633_633400

theorem masha_friends_inconsistent :
  let A := 25   -- girls who study mathematics
  let B := 30   -- girls who have been to Moscow
  let C := 28   -- girls who traveled by train
  let AC := 18  -- girls who study mathematics and traveled by train
  let BC := 17  -- girls who have been to Moscow and traveled by train
  let AB := 16  -- girls who study mathematics and have been to Moscow
  let ABC := 15 -- girls who study mathematics, have been to Moscow, and traveled by train
  let U := 45   -- total number of girls in the ensemble
  A + B + C - AB - AC - BC + ABC ≠ U :=
by {
  let A := 25
  let B := 30
  let C := 28
  let AC := 18
  let BC := 17
  let AB := 16
  let ABC := 15
  let U := 45
  show A + B + C - AB - AC - BC + ABC ≠ U, from sorry
}

end masha_friends_inconsistent_l633_633400


namespace least_additional_shading_required_l633_633990

-- Define the grid dimensions and initial conditions
def grid_size : ℕ := 4
def initial_shaded_squares : fin 4 × fin 4 → bool := 
  λ pos, pos = (1, 3) ∨ pos = (3, 0)

-- Define what it means for the grid to have both horizontal and vertical symmetry
def is_symmetric (grid : fin 4 × fin 4 → bool) : Prop :=
  (∀ i j, grid (i, j) = grid (3 - i, j)) ∧ -- Vertical symmetry
  (∀ i j, grid (i, j) = grid (i, 3 - j))   -- Horizontal symmetry

-- The theorem to be proven
theorem least_additional_shading_required :
  ∃ s, (∀ pos, s pos → ¬ initial_shaded_squares pos) ∧
       (∀ pos, initial_shaded_squares pos ∨ s pos) ∧
       is_symmetric (λ pos, initial_shaded_squares pos ∨ s pos) ∧
       (finset.filter (λ pos, s pos) (finset.univ : finset (fin 4 × fin 4))).card = 2 :=
sorry

end least_additional_shading_required_l633_633990


namespace semi_circle_perimeter_l633_633463

theorem semi_circle_perimeter (r : ℝ) (r_val : r = 35.00860766835085) : 
    (P : ℝ) (P_val : P = r * (Real.pi + 2)) → P ≈ 180.08 :=
by
  sorry

end semi_circle_perimeter_l633_633463


namespace length_of_XY_l633_633374

theorem length_of_XY (A B C D P Q X Y : ℝ) (h₁ : A = B) (h₂ : C = D) 
  (h₃ : A + B = 13) (h₄ : C + D = 21) (h₅ : A + P = 7) 
  (h₆ : C + Q = 8) (h₇ : P ≠ Q) (h₈ : P + Q = 30) :
  ∃ k : ℝ, XY = 2 * k + 30 + 31 / 15 :=
by sorry

end length_of_XY_l633_633374


namespace melinda_textbook_problem_m_plus_n_l633_633773

noncomputable def probability_math_books_in_same_box : ℚ :=
  (C(n 11, 0) * C(11, 1) + C(11, 1) * C(10, 4) + C(11, 2) * C(9, 4)) / 
  (C(15, 4) * C(11, 5) * C(6, 6))

theorem melinda_textbook_problem : 
  (C(n 11, 0) * C(11, 1) + C(11, 1) * C(10, 4) + C(11, 2) * C(9, 4)) / 
  (C(15, 4) * C(11, 5) * C(6, 6)) = 2 / 129 :=
sorry

theorem m_plus_n : 
  ∃ m n : ℕ, m + n = 131 ∧ (2 / 129 = (m / n)) :=
sorry

end melinda_textbook_problem_m_plus_n_l633_633773


namespace paving_cost_l633_633829

def length : Real := 5.5
def width : Real := 3.75
def rate : Real := 700
def area : Real := length * width
def cost : Real := area * rate

theorem paving_cost :
  cost = 14437.50 :=
by
  -- Proof steps go here
  sorry

end paving_cost_l633_633829


namespace math_problem_proof_l633_633259

def set_P : set ℝ := {x : ℝ | x^2 - 4*x + 3 ≤ 0}
def set_Q : set ℝ := {x : ℝ | x^2 - 4 < 0}
def set_complement_Q : set ℝ := {x : ℝ | x ≤ -2 ∨ x ≥ 2}

theorem math_problem_proof : 
  set_P ∪ set_complement_Q = {x : ℝ | x ≤ -2 ∨ x ≥ 1} := 
by
  sorry

end math_problem_proof_l633_633259


namespace gcd_of_ten_digit_same_five_digit_l633_633553

def ten_digit_same_five_digit (n : ℕ) : Prop :=
  n > 9999 ∧ n < 100000 ∧ ∃ k : ℕ, k = n * (10^10 + 10^5 + 1)

theorem gcd_of_ten_digit_same_five_digit :
  (∀ n : ℕ, ten_digit_same_five_digit n → ∃ d : ℕ, d = 10000100001 ∧ ∀ m : ℕ, m ∣ d) := 
sorry

end gcd_of_ten_digit_same_five_digit_l633_633553


namespace turnip_bag_weighs_l633_633091

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end turnip_bag_weighs_l633_633091


namespace simplified_expression_l633_633606

theorem simplified_expression (b : ℝ) (hb : b ≠ 0): 
    ((1/25) * b^0 + (1/(25*b))^0 - 125^(-1/3) - (-125)^(-3/3)) = 231/125 :=
by
  sorry

end simplified_expression_l633_633606


namespace parabola_properties_l633_633255

theorem parabola_properties (p n : ℝ) (h1 : p > 0) (h2 : n ≠ 0)
    (F : ℝ × ℝ) (hF : F = (0, 2))
    (line1 : ℝ → ℝ) (h_line1 : line1 = λ x, x + 1)
    (line2 : ℝ → ℝ) (h_line2 : line2 = λ x, (1/2) * x + n)
    (H : ℝ → ℝ → Prop) (hH : ∀ (x y : ℝ), H x y ↔ x^2 = 2 * p * y)
    (A B : ℝ × ℝ) (h_AB_intersect : H A.1 A.2 ∧ H B.1 B.2 ∧ 
                                      line1 A.1 = A.2 ∧ line1 B.1 = B.2)
    (AB_distance : ℝ) (h_AB_distance : AB_distance = 8 * real.sqrt 3)
    (C D : ℝ × ℝ) (h_CD_intersect : H C.1 C.2 ∧ H D.1 D.2 ∧ 
                                      line2 C.1 = C.2 ∧ line2 D.1 = D.2)
    (circle_diameter : ℝ) (h_circle_passes_F : (F.1 - C.1) * (F.1 - D.1) + 
                                               (F.2 - C.2) * (F.2 - D.2) = 0) :
  (H x y ↔ x^2 = 8 * y) ∧ 
  (∃ r : ℝ, r = real.sqrt ((2 - 0)^2 + (13 - 2)^2) ∧ 
            real.pi * r^2 = 125 * real.pi) :=
by
  sorry

end parabola_properties_l633_633255


namespace probability_two_red_two_blue_l633_633027

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose 20 4
  let choose_2_red := Nat.choose 12 2
  let choose_2_blue := Nat.choose 8 2
  (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690 := by
  let total_marbles := 20
  let total_red := 12
  let total_blue := 8
  let choose_4 := Nat.choose total_marbles 4
  let choose_2_red := Nat.choose total_red 2
  let choose_2_blue := Nat.choose total_blue 2
  show (choose_2_red * choose_2_blue).toRational / choose_4.toRational = 3696 / 9690
  sorry

end probability_two_red_two_blue_l633_633027


namespace magnitude_proof_l633_633214

noncomputable def z : ℂ := (3 + complex.I) / (1 - complex.I) + complex.I
def z_conj : ℂ := conj z
def z_conj_plus_i : ℂ := z_conj + complex.I
def magnitude : ℝ := complex.abs z_conj_plus_i

theorem magnitude_proof : magnitude = real.sqrt 5 := by
  sorry

end magnitude_proof_l633_633214
