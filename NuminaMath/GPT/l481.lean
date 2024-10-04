import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Field
import Mathlib.Algebra.GcdDomain
import Mathlib.Algebra.Group.Prod
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Mod
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import data.nat.digits

namespace probability_of_rolling_number_less_than_5_is_correct_l481_481218

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l481_481218


namespace calculate_3_pow_5_mul_6_pow_5_l481_481297

theorem calculate_3_pow_5_mul_6_pow_5 :
  3^5 * 6^5 = 34012224 := 
by 
  sorry

end calculate_3_pow_5_mul_6_pow_5_l481_481297


namespace max_additional_viewers_l481_481979

theorem max_additional_viewers (x : ℕ) (h1 : 0 ≤ x ∧ x ≤ 10) (h2 : ∀ y : ℕ, 0 ≤ y ∧ y ≤ 10) :
  ∃ n : ℕ, n = 5 ∧ (∀ k : ℕ, k ≤ n → new_rating(n, x) = x - k) :=
by
  sorry

end max_additional_viewers_l481_481979


namespace students_not_enrolled_correct_l481_481395

def students_enrolled (total_students: ℕ) (enrollment_percentage: ℕ) : ℕ :=
  (total_students * enrollment_percentage) / 100

def students_not_enrolled (total_students: ℕ) (enrollment_percentage: ℕ) : ℕ :=
  total_students - students_enrolled total_students enrollment_percentage

theorem students_not_enrolled_correct (total_students: ℕ) (enrollment_percentage: ℕ) (h: total_students = 880) (h1: enrollment_percentage = 30) : students_not_enrolled total_students enrollment_percentage = 616 :=
by
  rw [h, h1]
  unfold students_not_enrolled students_enrolled
  norm_num
  sorry

end students_not_enrolled_correct_l481_481395


namespace women_in_room_l481_481438

theorem women_in_room :
  ∃ (x : ℤ), 
    let men_initial := 4 * x,
        women_initial := 5 * x,
        men_now := men_initial + 2,
        women_left := women_initial - 3,
        women_doubled := 2 * women_left,
        men_now = 14 in
    2 * (5 * x - 3) = 24 :=
by
  sorry

end women_in_room_l481_481438


namespace set_intersection_eq_l481_481496

noncomputable def M : Set ℝ := {x | -2 < x ∧ x < 3}
noncomputable def N : Set ℝ := {x | 2^(x + 1) ≤ 1}
noncomputable def complement_N : Set ℝ := {x | ¬ (2^(x + 1) ≤ 1)}

theorem set_intersection_eq :
  M ∩ complement_N = {x | -1 < x ∧ x < 3} := by 
sorry

end set_intersection_eq_l481_481496


namespace count_zeros_in_10000_power_50_l481_481585

theorem count_zeros_in_10000_power_50 :
  10000^50 = 10^200 :=
by
  have h1 : 10000 = 10^4 := by sorry
  have h2 : (10^4)^50 = 10^(4 * 50) := by sorry
  exact h2.trans (by norm_num)

end count_zeros_in_10000_power_50_l481_481585


namespace right_triangle_least_side_l481_481771

theorem right_triangle_least_side (a b : ℕ) (h₁ : a = 8) (h₂ : b = 15) :
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ a^2 = c^2 + b^2 ∨ b^2 = c^2 + a^2) ∧ c = Real.sqrt 161 := 
sorry

end right_triangle_least_side_l481_481771


namespace exponents_of_ten_problem_zeros_10000_pow_50_l481_481587

theorem exponents_of_ten (a : ℤ) (b : ℕ) (h : a = 10^4) : a^b = 10^(4 * b) := by
  rw [h, pow_mul]
  simp
  sorry

theorem problem_zeros_10000_pow_50 : 10000^50 = 10^200 :=
  exponents_of_ten 10000 50 rfl

end exponents_of_ten_problem_zeros_10000_pow_50_l481_481587


namespace box_triple_count_l481_481631

theorem box_triple_count (a b c : ℕ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a * b * c = 2 * (a * b + b * c + c * a)) :
  (a = 2 ∧ b = 8 ∧ c = 8) ∨ (a = 3 ∧ b = 6 ∧ c = 6) ∨ (a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = 5 ∧ b = 5 ∧ c = 5) ∨ (a = 6 ∧ b = 6 ∧ c = 6) :=
sorry

end box_triple_count_l481_481631


namespace monotonic_intervals_and_extrema_l481_481736

noncomputable def f (x : Real) : Real := Real.exp (-x) * Real.sin x

theorem monotonic_intervals_and_extrema :
  (∀ k : Int,
    ∀ x ∈ Set.interval (2 * Real.pi * k - 3 * Real.pi / 4) (2 * Real.pi * k + Real.pi / 4), 
    f' x > 0) ∧
  (∀ k : Int,
    ∀ x ∈ Set.interval (2 * Real.pi * k + Real.pi / 4) (2 * Real.pi * k + 5 * Real.pi / 4), 
    f' x < 0) ∧
  (∃ x_max ∈ Set.interval (-Real.pi) Real.pi, 
    x_max = Real.pi / 4 ∧ f x_max = sqrt(2) / 2 * Real.exp (-Real.pi / 4)) ∧
  (∃ x_min ∈ Set.interval (-Real.pi) Real.pi, 
    x_min = -3 * Real.pi / 4 ∧ f x_min = -sqrt(2) / 2 * Real.exp (3 * Real.pi / 4)) := 
  sorry

end monotonic_intervals_and_extrema_l481_481736


namespace geometric_sequence_common_ratio_l481_481781

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h₁ : a 1 = 1) (h₂ : a 1 * a 2 * a 3 = -8) :
  q = -2 :=
sorry

end geometric_sequence_common_ratio_l481_481781


namespace captives_strategy_guarantees_release_l481_481819

theorem captives_strategy_guarantees_release (n : ℕ) (hs : n ≥ 3)
  (figures : Fin n → ℕ)
  (h_unique : ∀ i j, figures i ≠ figures j → multiset.card (multiset.filter (= i) figures) ≠ multiset.card (multiset.filter (= j) figures))
  (prisoners_can_see_each_other : ∀ i j, i ≠ j → figures j) :
  ∃ (strategy : (Fin n → ℕ) → (Fin n → ℕ)), 
    (∀ perm : Fin n → ℕ, ∃ i : Fin n, strategy perm i = perm i) :=
sorry

end captives_strategy_guarantees_release_l481_481819


namespace range_of_a_l481_481400

theorem range_of_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * sqrt(a) * x1 + 2 * a - 1 = 0) ∧ (x2^2 - 2 * sqrt(a) * x2 + 2 * a - 1 = 0)) : 
0 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l481_481400


namespace sum_of_values_satisfying_equation_l481_481578

theorem sum_of_values_satisfying_equation : 
  ∃ (s : ℤ), (∀ (x : ℤ), (abs (x + 5) = 9) → (x = 4 ∨ x = -14) ∧ (s = 4 + (-14))) :=
begin
  sorry
end

end sum_of_values_satisfying_equation_l481_481578


namespace jane_loan_difference_l481_481805

def compound_interest (P r n t : ℝ) : ℝ := P * (1 + r / n) ^ (n * t)

def simple_interest (P r t : ℝ) : ℝ := P + P * r * t

theorem jane_loan_difference :
  let P := 15000
      r1 := 0.08
      n1 := 2
      t := 15 / 2
      compounded_after_7_5 := compound_interest P r1 n1 t
      half_payment := compounded_after_7_5 / 2
      remaining_balance := half_payment
      compounded_after_another_7_5 := compound_interest remaining_balance r1 n1 t
      total_compounded := half_payment + compounded_after_another_7_5
      r2 := 0.1
      simple_interest_total := simple_interest P r2 (15 : ℝ)
  in
  abs (simple_interest_total - total_compounded) = 9525 := sorry

end jane_loan_difference_l481_481805


namespace rate_of_second_wheat_purchase_l481_481648

theorem rate_of_second_wheat_purchase :
  let first_purchase_wheat := 30
  let first_purchase_rate := 11.50
  let second_purchase_wheat := 20
  let selling_price_per_kg := 13.86
  let total_wheat := first_purchase_wheat + second_purchase_wheat
  let total_cost_first := first_purchase_wheat * first_purchase_rate
  let profit_margin := 1.10
  let selling_price_total := total_wheat * selling_price_per_kg
  let total_cost := total_cost_first + second_purchase_wheat * x
  in
  (selling_price_total / profit_margin = total_cost) -> 
  x = 14.25 :=
by
  sorry

end rate_of_second_wheat_purchase_l481_481648


namespace monotonic_range_l481_481371

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2 - x

def is_monotonic (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  (∀ x y ∈ s, x ≤ y → f x ≤ f y) ∨ (∀ x y ∈ s, x ≤ y → f y ≤ f x)

theorem monotonic_range (a : ℝ) :
  is_monotonic (λ x, f x a) { x : ℝ | 0 < x } ↔ a ≥ 1 / Real.exp 1 :=
sorry

end monotonic_range_l481_481371


namespace no_constant_sum_adjacent_faces_l481_481767

-- Definitions and conditions
def faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
def n_faces := 12
def neighbors_per_face := 5
def sum_1_to_12 := 78

-- Statement of the problem
theorem no_constant_sum_adjacent_faces :
  ¬ ∃ S : ℕ, 12 * S = 5 * sum_1_to_12 :=
by sorry

end no_constant_sum_adjacent_faces_l481_481767


namespace vector_dot_product_AO_AB_eq_4_l481_481720

-- Define conditions and proof goal
theorem vector_dot_product_AO_AB_eq_4
    (O A B : EuclideanSpace ℝ (Fin 2))
    (hA : ‖A - O‖ = 2)
    (hB : ‖B - O‖ = 2)
    (h_eq : ‖(A - O) + (B - O)‖ = ‖(A - O) - (B - O)‖) :
    (A - O) ∙ (A - B) = 4 := by
  sorry

end vector_dot_product_AO_AB_eq_4_l481_481720


namespace soccer_team_lineup_l481_481269

theorem soccer_team_lineup (total_members injured_player : ℕ) (num_positions : ℕ) (injured_not_goalkeeper : total_members > 0 ∧ injured_player = 1) : 
  (total_members = 16 ∧ num_positions = 4 ∧ injured_not_goalkeeper) →
  ∃ num_ways : ℕ, num_ways = 42210 :=
begin
  sorry
end

end soccer_team_lineup_l481_481269


namespace distance_covered_at_40_kmph_l481_481233

theorem distance_covered_at_40_kmph
   (total_distance : ℝ)
   (speed1 : ℝ)
   (speed2 : ℝ)
   (total_time : ℝ)
   (part_distance1 : ℝ) :
   total_distance = 250 ∧
   speed1 = 40 ∧
   speed2 = 60 ∧
   total_time = 6 ∧
   (part_distance1 / speed1 + (total_distance - part_distance1) / speed2 = total_time) →
   part_distance1 = 220 :=
by sorry

end distance_covered_at_40_kmph_l481_481233


namespace incircle_area_sum_l481_481380

variables {a b c : ℝ}

def semiperimeter (a b c : ℝ) := (a + b + c) / 2

def inradius (A : ℝ) (p : ℝ) := A / p

noncomputable def area_triangle (a b c : ℝ) : ℝ := 
  Real.sqrt (semiperimeter a b c * (semiperimeter a b c - a) *
               (semiperimeter a b c - b) * (semiperimeter a b c - c))

noncomputable def sum_incircle_areas (a b c : ℝ) : ℝ :=
  let p := semiperimeter a b c in
  let A := area_triangle a b c in
  let r := inradius A p in
  let r1 := inradius A (p - a) in
  let r2 := inradius A (p - b) in
  let r3 := inradius A (p - c) in
  π * (r ^ 2 + r1 ^ 2 + r2 ^ 2 + r3 ^ 2)

theorem incircle_area_sum (a b c : ℝ) :
  sum_incircle_areas a b c =
  π * (a^2 + b^2 + c^2) * (b + c - a) * (c + a - b) * (a + b - c) / (a + b + c)^3 :=
by sorry

end incircle_area_sum_l481_481380


namespace ball_hits_ground_l481_481289

theorem ball_hits_ground :
  ∃ t : ℝ, -16 * t^2 + 20 * t + 100 = 0 ∧ t = (5 + Real.sqrt 425) / 8 :=
by
  sorry

end ball_hits_ground_l481_481289


namespace simplify_expression_l481_481522

theorem simplify_expression :
  (1 / (1 / ((1 / 3)^1) + 1 / ((1 / 3)^2) + 1 / ((1 / 3)^3))) = 1 / 39 :=
by
  sorry

end simplify_expression_l481_481522


namespace area_of_ellipse_l481_481298

-- Define the necessary parameters
variables {a b : ℝ}

-- State the conditions
def conditions : Prop := a > b ∧ b > 0

-- The statement to prove the area of the ellipse
theorem area_of_ellipse (h : conditions) : ∃ S : ℝ, S = π * a * b :=
sorry

end area_of_ellipse_l481_481298


namespace calculation_correct_l481_481598

noncomputable def problem_calculation : ℝ :=
  4 * Real.sin (Real.pi / 3) - abs (-1) + (Real.sqrt 3 - 1)^0 + Real.sqrt 48

theorem calculation_correct : problem_calculation = 6 * Real.sqrt 3 :=
by
  sorry

end calculation_correct_l481_481598


namespace log_equality_ineq_l481_481377

--let a = \log_{\sqrt{5x-1}}(4x+1)
--let b = \log_{4x+1}\left(\frac{x}{2} + 2\right)^2
--let c = \log_{\frac{x}{2} + 2}(5x-1)

noncomputable def a (x : ℝ) : ℝ := 
  Real.log (4 * x + 1) / Real.log (Real.sqrt (5 * x - 1))

noncomputable def b (x : ℝ) : ℝ := 
  2 * (Real.log ((x / 2) + 2) / Real.log (4 * x + 1))

noncomputable def c (x : ℝ) : ℝ := 
  Real.log (5 * x - 1) / Real.log ((x / 2) + 2)

theorem log_equality_ineq (x : ℝ) : 
  a x = b x ∧ c x = a x - 1 ↔ x = 2 := 
by
  sorry

end log_equality_ineq_l481_481377


namespace cost_of_children_ticket_l481_481259

theorem cost_of_children_ticket (total_cost : ℝ) (cost_adult_ticket : ℝ) (num_total_tickets : ℕ) (num_adult_tickets : ℕ) (cost_children_ticket : ℝ) :
  total_cost = 119 ∧ cost_adult_ticket = 21 ∧ num_total_tickets = 7 ∧ num_adult_tickets = 4 -> cost_children_ticket = 11.67 :=
by
  intros h
  sorry

end cost_of_children_ticket_l481_481259


namespace frog_jump_probability_is_one_fifth_l481_481618

noncomputable def frog_jump_probability : ℝ := sorry

theorem frog_jump_probability_is_one_fifth : frog_jump_probability = 1 / 5 := sorry

end frog_jump_probability_is_one_fifth_l481_481618


namespace prime_count_between_80_and_90_l481_481054

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (list.range' a (b - a + 1)).filter is_prime |>.length

theorem prime_count_between_80_and_90 :
  count_primes_in_range 80 90 = 2 :=
by
  sorry

end prime_count_between_80_and_90_l481_481054


namespace probability_of_rolling_number_less_than_5_is_correct_l481_481217

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l481_481217


namespace rachel_remaining_amount_l481_481870

-- Definitions of the initial earning, the fraction spent on lunch, and the fraction spent on the DVD.
def initial_amount : ℝ := 200
def fraction_lunch : ℝ := 1/4
def fraction_dvd : ℝ := 1/2

-- Calculation of the remaining amount Rachel has.
theorem rachel_remaining_amount :
  let spent_on_lunch := fraction_lunch * initial_amount in
  let spent_on_dvd := fraction_dvd * initial_amount in
  let remaining_amount := initial_amount - spent_on_lunch - spent_on_dvd in
  remaining_amount = 50 :=
by
  sorry

end rachel_remaining_amount_l481_481870


namespace partition_gcd_l481_481550

theorem partition_gcd (S : set ℕ) (n : ℕ) 
  (hS : S = { x | 1 ≤ x ∧ x ≤ 2022 })
  (hPartition : ∃ (S_i : ℕ → set ℕ) (hf : ∀ i, S_i i ⊆ S ∧ (∀ (i j : ℕ), i ≠ j → S_i i ∩ S_i j = ∅) ∧ (⋃ i, S_i i = S) ∧ (∀ i, (∀ x y ∈ S_i i, x ≠ y → gcd x y > 1) ∨ (∀ x y ∈ S_i i, x ≠ y → gcd x y = 1))))
: n = 15 := sorry

end partition_gcd_l481_481550


namespace percent_of_x_is_z_l481_481061

def condition1 (z y : ℝ) : Prop := 0.45 * z = 0.72 * y
def condition2 (y x : ℝ) : Prop := y = 0.75 * x
def condition3 (w z : ℝ) : Prop := w = 0.60 * z^2
def condition4 (z w : ℝ) : Prop := z = 0.30 * w^(1/3)

theorem percent_of_x_is_z (x y z w : ℝ) 
  (h1 : condition1 z y) 
  (h2 : condition2 y x)
  (h3 : condition3 w z)
  (h4 : condition4 z w) : 
  z / x = 1.2 :=
sorry

end percent_of_x_is_z_l481_481061


namespace longest_chord_length_l481_481568

theorem longest_chord_length (A : ℝ) (r R : ℝ) (h : R > r) (h_area : A = π * (R^2 - r^2)) : 
  2 * √(A / π) = 2 * √((R^2 - r^2)) := 
by 
  sorry

end longest_chord_length_l481_481568


namespace isosceles_triangle_height_decreases_base_l481_481663

theorem isosceles_triangle_height_decreases_base (a b : ℝ) (ha : a > 0) (hb : 0 < b ∧ b < a) :
  ∃ h : ℝ, h = real.sqrt (a^2 - b^2) ∧ (∀ (b' : ℝ), b' > b → real.sqrt (a^2 - b'^2) < h) :=
by
  sorry

end isosceles_triangle_height_decreases_base_l481_481663


namespace no_prime_in_range_l481_481691

theorem no_prime_in_range {n : ℤ} (h1 : n > 2) : 
  ∀ k, (2 ≤ k ∧ k ≤ n + 1) → ¬is_prime ((n+1)! + k) :=
by sorry

end no_prime_in_range_l481_481691


namespace collinear_points_P_Q_reflection_N_AC_l481_481069

noncomputable theory
open_locale classical

variables {A B C O N X Y P Q : Point}
variables {triangle : Triangle}
variables {circumcircle : Circle}

/- Define the specific geometric properties and relationships -/
def is_obtuse_angle (B : Point) (A B C : Triangle) : Prop :=
angle B > 90

def midpoint_arc (N : Point) (circumcircle : Circle) : Prop :=
∃ A B C : Point, is_circle circumcircle ∧ is_midpoint_of_arc N circumcircle A B C

def reflection_wrt_line (N : Point) (line_AC : Line) : Point := 
reflection N line_AC

/- The main theorem stating that given the conditions, points P, Q and the reflection of N are collinear -/
theorem collinear_points_P_Q_reflection_N_AC (
    obtuse_angle_B : is_obtuse_angle B A B C,
    AB_neq_BC : A ≠ C,
    circumcenter_O : circumcenter O triangle,
    midpoint_N : midpoint_arc N circumcircle,
    circumcircle_intersection : intersects_at (circumcircle_of_triangle BON) AC X Y,
    PX_intersects_circumcircle : intersects_at (line B X) circumcircle P B,
    QY_intersects_circumcircle : intersects_at (line B Y) circumcircle Q B) :
    collinear P Q (reflection_wrt_line N AC) := 
sorry

end collinear_points_P_Q_reflection_N_AC_l481_481069


namespace alice_can_prevent_bob_win_l481_481642

-- Define the game strategy for Alice and Bob where the digits are chosen in alternating turns
-- The condition is that each newly chosen digit must belong to a different residue class modulo 3 from the previous digit.
def can_alice_prevent_bob_win : Prop :=
  ∀ (digits : Fin 2018 → Fin 10),
    ∃ (alice_strategy : Fin 1009 → Fin 10),
      (∀ i : Fin 1009, (digits (2 * i).val + alice_strategy i).val % 3 ≠ (digits ((2 * i).val + 1)).val % 3) ∧
      ((digits ⟨2017, sorry⟩.val + digits ⟨2016, sorry⟩.val + ... + digits ⟨0, sorry⟩.val) % 3 ≠ 0)
       
-- The theorem corresponding to the translated proof problem
theorem alice_can_prevent_bob_win : can_alice_prevent_bob_win :=
sorry

end alice_can_prevent_bob_win_l481_481642


namespace minimum_draw_to_ensure_twelve_l481_481608

-- Define the number of balls for each color.
def red_balls : ℕ := 15
def green_balls : ℕ := 25
def yellow_balls : ℕ := 20
def blue_balls : ℕ := 22
def white_balls : ℕ := 18

-- Define the minimum number of balls needed to guarantee 12 balls of one color.
def minimum_balls_to_draw : ℕ := 56

-- The theorem to be proven: At least 12 balls of one color will be drawn if 56 balls are drawn.
theorem minimum_draw_to_ensure_twelve (n_drawn : ℕ) (h : n_drawn = 56) : 
  ∃ color ∈ {red_balls, green_balls, yellow_balls, blue_balls, white_balls}, color >= 12 := 
by
  sorry

end minimum_draw_to_ensure_twelve_l481_481608


namespace count_zeros_in_10000_power_50_l481_481584

theorem count_zeros_in_10000_power_50 :
  10000^50 = 10^200 :=
by
  have h1 : 10000 = 10^4 := by sorry
  have h2 : (10^4)^50 = 10^(4 * 50) := by sorry
  exact h2.trans (by norm_num)

end count_zeros_in_10000_power_50_l481_481584


namespace sum_of_valid_p_values_l481_481413

-- Definitions and variables based on problem conditions
variable (teams participants : ℕ)
variable (p t : ℕ)
def total_participants := participants = 360
def num_teams := t >= 12
def participants_per_team := p >= 20
def multiplication_constraint := t * p = 360

-- Theorem statement
theorem sum_of_valid_p_values (total_participants : participants = 360)
                             (num_teams : t ≥ 12)
                             (participants_per_team : p ≥ 20)
                             (multiplication_constraint : t * p = 360) :
    ∑ (x : ℕ) in {p | 20 ≤ p ∧ 360 % p = 0 ∧ 360 / p ≥ 12}, x = 54 :=
begin
  sorry  -- Proof can be constructed here
end

end sum_of_valid_p_values_l481_481413


namespace coal_last_days_l481_481258

def actual_days (planned_consumption : ℝ) (planned_days : ℕ) (reduction : ℝ) : ℝ :=
  let total_coal := planned_consumption * planned_days
  let actual_consumption := planned_consumption * (1 - reduction)
  total_coal / actual_consumption

theorem coal_last_days (planned_consumption : ℝ) (planned_days : ℕ) (reduction : ℝ) :
  planned_consumption = 0.25 → planned_days = 80 → reduction = 0.2 →
  actual_days planned_consumption planned_days reduction = 100 := by
  intros
  sorry

end coal_last_days_l481_481258


namespace perp_line_to_plane_implies_perp_lines_l481_481381

variables (l m : Line) (α : Plane)
constants (is_perpendicular : Line → Plane → Prop) 
          (is_parallel : Line → Plane → Prop)
          (is_perpendicular_lines : Line → Line → Prop)

theorem perp_line_to_plane_implies_perp_lines
  (h1 : is_perpendicular l α)
  (h2 : is_parallel m α) :
  is_perpendicular_lines l m :=
sorry

end perp_line_to_plane_implies_perp_lines_l481_481381


namespace nine_digit_prime_check_l481_481317

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def remove_digits (n : ℕ) (indices : List ℕ) : ℕ :=
  -- Function to remove digits at specified indices from the number
  sorry

def seven_digit_removals (n : ℕ) : List ℕ :=
  -- Function to generate all numbers by removing seven digits
  sorry

theorem nine_digit_prime_check : 
  ∃ N : ℕ, N = 391524680 ∧ 
  (nat.digits 10 N).length = 9 ∧ 
  (list.nodup (nat.digits 10 N)) ∧ 
  let results := seven_digit_removals N in
  (list.filter is_prime results).length ≤ 1 :=
by
  sorry

end nine_digit_prime_check_l481_481317


namespace quarters_initially_l481_481462

theorem quarters_initially (quarters_borrowed : ℕ) (quarters_now : ℕ) (initial_quarters : ℕ) 
   (h1 : quarters_borrowed = 3) (h2 : quarters_now = 5) :
   initial_quarters = quarters_now + quarters_borrowed :=
by
  -- Proof goes here
  sorry

end quarters_initially_l481_481462


namespace count_non_self_intersecting_paths_l481_481559

-- Define the number of points on the circle
def N : ℕ := 10

-- Function to count the number of non-self-intersecting 9-segment paths
def count_paths (n : ℕ) : ℕ :=
  -- Formula derived from the solution steps: 10 * 2^7
  n * 2^(n - 2)

-- Proof statement: Given 10 points on a circle, prove that the number of such paths is 1280
theorem count_non_self_intersecting_paths (points_on_circle : ℕ) (num_segments : ℕ) (h1 : points_on_circle = N) (h2 : num_segments = 9) :
  count_paths points_on_circle = 1280 :=
by
  -- Utilize given conditions to initialize the formula
  rw [h1, h2]
  -- Substitute values and confirm if it matches the expected result
  dsimp [count_paths]
  sorry

end count_non_self_intersecting_paths_l481_481559


namespace exists_polynomial_with_properties_l481_481139

theorem exists_polynomial_with_properties (n : ℤ) (h : n ≥ 4) :
  ∃ f : ℤ → ℤ, (∀ a : ℤ, is_positive_integer_coeff a ∧ degree f = n) ∧
  (∀ m k : ℤ, k ≥ 2 → ∀ (r : Fin k).distinct_positive_integers,
    f m ≠ Finset.prod (Finset.fin_range k) (λ i, f r i)) := sorry

def is_positive_integer_coeff (a : ℤ) : Prop :=
  a > 0

def degree (f : ℤ → ℤ) : ℤ :=
  -- logic to determine the degree of the polynomial
  sorry

def distinct_positive_integers (k : ℤ) : Prop :=
  -- logic to ensure that a set of integers are distinct positive integers
  sorry

end exists_polynomial_with_properties_l481_481139


namespace second_date_sum_eq_80_l481_481593

theorem second_date_sum_eq_80 (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 + a2 + a3 + a4 + a5 = 80)
  (h2 : a2 = a1 + 1) (h3 : a3 = a2 + 1) (h4 : a4 = a3 + 1) (h5 : a5 = a4 + 1): a2 = 15 :=
by
  sorry

end second_date_sum_eq_80_l481_481593


namespace min_k_for_intersection_l481_481700

theorem min_k_for_intersection (k : ℝ) : 
  (∀ (x y : ℝ), (x + 2)^2 + y^2 = 4 → k * x - y = 2 * k) → k ≥ - real.sqrt 3 / 3 := 
sorry

end min_k_for_intersection_l481_481700


namespace quadratic_has_two_real_roots_root_greater_than_three_l481_481379
noncomputable theory

-- Part 1: Prove that the quadratic equation always has two real roots.
theorem quadratic_has_two_real_roots (a : ℝ) : 
  let Δ := (a - 2)^2 in Δ ≥ 0 :=
sorry

-- Part 2: If the equation has one real root greater than 3, find the range of values for a.
theorem root_greater_than_three (a : ℝ) (h : ∃ x : ℝ, (x * x - a * x + a - 1 = 0 ∧ x > 3)) : 
  a > 4 :=
sorry

end quadratic_has_two_real_roots_root_greater_than_three_l481_481379


namespace disjoint_intervals_sum_l481_481879

def f (x : ℝ) : ℝ := ∑ (k : ℕ) in finset.range 71, (k : ℝ) / (x - k)

theorem disjoint_intervals_sum :
  (∑ (k : ℕ) in finset.range 71, (a k - k)) = 1988 :=
by 
  sorry

end disjoint_intervals_sum_l481_481879


namespace angle_bisector_proof_l481_481566

variable {α : Type*}

-- Assuming basic definitions for points, lines, and circles
variables (A B C D M X Y : α) (w y : set α) (l1 l2 : set α)

-- Assumptions to set up the problem
variables (is_triangle_ABC : ∀ P Q, P ∈ w → Q ∈ w → P ≠ Q → BC ≠ P ∧ BC ≠ Q)
          (circumscribed_ABQ : w = set_of (λ P, P ≠ A ∧ P ≠ B ∧ P ≠ C))
          (angle_bisector_l1 : ∃ l1, ∀ angle (X : α), (∠ AXC = ∠BXA))
          (line_intersections_D_M : D ∈ BC ∧ M ∈ w ∧ M ∈ l1)
          (circle_y_def : y = set_of (λ P : α, dist M P = dist M B))
          (line_l2_intersects_y : ∃ l2, ∀ P, P ∈ l2 ↔ ∃ P', P, P' ∈ y ∧ dist D P = dist D P')

theorem angle_bisector_proof :
  ∃ l1, ∀ angle (X Y : α), (∠ XAY = ∠BAX) :=
sorry

end angle_bisector_proof_l481_481566


namespace saddle_value_l481_481131

theorem saddle_value (S : ℝ) (H : ℝ) (h1 : S + H = 100) (h2 : H = 7 * S) : S = 12.50 :=
by
  sorry

end saddle_value_l481_481131


namespace find_tax_rate_l481_481300

def price_before_discount : ℝ := 40
def discount : ℝ := 8
def cody_paid : ℝ := 17
def total_paid : ℝ := cody_paid * 2
def price_after_discount : ℝ := total_paid + discount

def tax_rate (r : ℝ) : Prop :=
  price_before_discount * (1 + r) = price_after_discount

theorem find_tax_rate : tax_rate 0.05 :=
by {
  unfold tax_rate,
  unfold price_before_discount,
  unfold price_after_discount,
  unfold total_paid,
  norm_num,
  sorry
}

end find_tax_rate_l481_481300


namespace log_sum_geom_seq_l481_481796

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x^2 + 4 * x - 1

theorem log_sum_geom_seq :
  let geometric_sequence := λ (n : ℕ) (a₁ r : ℝ), (a₁ : ℝ) * (r ^ (n - 1))
  ∃ (a₁ a₂ a₁₅ : ℝ), 
    (is_extreme_point f a₁ ∧ is_extreme_point f a₁₅) ∧
    (geometric_sequence 1 = a₁) ∧
    (geometric_sequence 2015 = a₁₅) ∧
    (a₁ * a₁₅ = 4) →
  log_sum (geometric_sequence a₁ r 2015) = 2014 := 
sorry

end log_sum_geom_seq_l481_481796


namespace geometric_sequence_relation_l481_481711

-- Define the sums of the sequence's first n, 2n, and 3n terms
variables {A B C : ℝ}

-- Assuming A, B, C follow the specified relationship in the geometric sequence
axiom geometric_sequence_sums (n : ℕ) :
  (Sₙ S 0) = A ∧ (Sₙ S n) = B ∧ (Sₙ S 2n) = C

-- The theorem we need to prove
theorem geometric_sequence_relation (n : ℕ) (A B C : ℝ) :
  A^2 + B^2 = A * (B + C) :=
sorry

end geometric_sequence_relation_l481_481711


namespace least_number_to_subtract_from_5785_l481_481926

noncomputable def least_number_to_subtract (n : ℕ) : ℕ :=
  let lcm := Nat.lcm (Nat.lcm 11 13) (Nat.lcm 19 23)
  (n - (lcm - 12))

theorem least_number_to_subtract_from_5785 : least_number_to_subtract 5785 = 78 :=
begin
  sorry
end

end least_number_to_subtract_from_5785_l481_481926


namespace probability_of_rolling_less_than_5_l481_481220

theorem probability_of_rolling_less_than_5 :
  (let outcomes := 8 in
   let successes := 4 in
   let probability := (successes: ℚ) / outcomes in
   probability = 1 / 2) :=
by
  sorry

end probability_of_rolling_less_than_5_l481_481220


namespace common_tangents_proof_l481_481839

noncomputable def common_tangents (m : ℝ) : Prop :=
  (m ≠ 0) →
  (∀ x y : ℝ, ∃ t : Prop,
    x^2 + y^2 - 4*(m+1)*x - 2*m*y + 4*m^2 + 4*m + 1 = 0 → t ∧
    (t → ((4*x - 3*y - 4 = 0) ∨ (y = 0))))

theorem common_tangents_proof :
  ∀ m : ℝ, common_tangents m := 
begin
  sorry
end

end common_tangents_proof_l481_481839


namespace no_real_roots_of_quadratic_l481_481770

theorem no_real_roots_of_quadratic (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≠ 0) ↔ a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

end no_real_roots_of_quadratic_l481_481770


namespace expected_value_of_kelvins_score_l481_481816

theorem expected_value_of_kelvins_score :
  let T := ℕ in
  -- Definition of flip outcomes and probabilities
  let probability_end_in_first_minute := (1/2) * (1/2) in
  let probability_one_coin_left := 2 * (1/2) * (1/2) in
  let probability_both_coins_left := (1/2) * (1/2) in
  -- Expected value calculations
  let E1 := 2 in -- Derived expected value with one coin left
  let E := 2 + 1/4 * E in
  -- Final expected score, squared
  (E^2 = 64 / 9) := 
sorry

end expected_value_of_kelvins_score_l481_481816


namespace solve_exponential_equation_l481_481693

theorem solve_exponential_equation (x : ℝ) : 
    (10^x) * (1000^(2 * x)) = 10000^5 ↔ x = 20 / 7 := 
by
  sorry

end solve_exponential_equation_l481_481693


namespace monotonically_decreasing_interval_l481_481015

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}
variable {x : ℝ}

noncomputable def is_monotonically_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y ∈ s, x < y → f y ≤ f x

theorem monotonically_decreasing_interval : 
  (∀ x, f' x = (x - 3) * (x + 1) ^ 2) → is_monotonically_decreasing f {x | x ≤ 3} :=
by 
  intro h
  sorry

end monotonically_decreasing_interval_l481_481015


namespace negation_of_existence_l481_481169

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * a * x + a > 0 :=
by
  sorry

end negation_of_existence_l481_481169


namespace total_hours_over_two_weeks_l481_481995

-- Define the conditions of Bethany's riding schedule
def hours_per_week : ℕ :=
  1 * 3 + -- Monday, Wednesday, and Friday
  (30 / 60) * 2 + -- Tuesday and Thursday, converting minutes to hours
  2 -- Saturday

-- The theorem to prove the total hours over 2 weeks
theorem total_hours_over_two_weeks : hours_per_week * 2 = 12 := 
by
  -- Proof to be completed here
  sorry

end total_hours_over_two_weeks_l481_481995


namespace max_height_attained_l481_481250

noncomputable def v (t : ℝ) : ℝ := 29.4 - 9.8 * t

theorem max_height_attained :
  ∫ x in (0 : ℝ) .. 3, v x = 44.1 :=
by
  -- proof goes here
  sorry

end max_height_attained_l481_481250


namespace kyle_practice_time_l481_481466

-- Definitions for the conditions
def weightlifting_time : ℕ := 20  -- in minutes
def running_time : ℕ := 2 * weightlifting_time  -- twice the weightlifting time
def total_running_and_weightlifting_time : ℕ := weightlifting_time + running_time  -- total time for running and weightlifting
def shooting_time : ℕ := total_running_and_weightlifting_time  -- because it's half the practice time

-- Total daily practice time, in minutes
def total_practice_time_minutes : ℕ := shooting_time + total_running_and_weightlifting_time

-- Total daily practice time, in hours
def total_practice_time_hours : ℕ := total_practice_time_minutes / 60

-- Theorem stating that Kyle practices for 2 hours every day given the conditions
theorem kyle_practice_time : total_practice_time_hours = 2 := by
  sorry

end kyle_practice_time_l481_481466


namespace prob_A3_A10_l481_481078

def P (A : ℕ → Prop) : ℕ → ℝ 
| 1 => 1
| 2 => 0
| k + 1 => (1 - P A k) / 3

theorem prob_A3_A10 (A : ℕ → Prop) : P A 3 = 1 / 3 ∧ P A 10 ≈ 0.25 :=
by
  sorry

end prob_A3_A10_l481_481078


namespace clown_additional_balloons_l481_481892

theorem clown_additional_balloons (b_initial b_total b_additional : ℕ) 
  (h_initial : b_initial = 47) 
  (h_total : b_total = 60) :
  b_additional = b_total - b_initial := 
by
  rw [h_total, h_initial]
  exact b_additional = 13

end clown_additional_balloons_l481_481892


namespace number_of_possible_values_U_l481_481129
open Set

def possible_sum_U (C : Set ℕ) (hC : C ⊆ Icc 10 120) (hCard : C.card = 80) : ℕ := C.sum id

theorem number_of_possible_values_U :
  ∃ n : ℕ, (∀ (C : Finset ℕ), (C ⊆ Icc 10 120) ∧ (C.card = 80) → (possible_sum_U C = n)) :=
  2481 :=
sorry

end number_of_possible_values_U_l481_481129


namespace probability_of_less_than_5_is_one_half_l481_481228

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l481_481228


namespace probability_of_event_E_l481_481211

-- Define the event E as rolling a number less than 5
def event_E (n : ℕ) : Prop := n < 5

-- Define the sample space as the set {1, 2, ..., 8} for an 8-sided die
def sample_space : set ℕ := {n | n ≥ 1 ∧ n ≤ 8}

-- Define the probability function
def probability (E : set ℕ) (S : set ℕ) : ℚ :=
  if S.nonempty then
    E.to_finset.card.to_rat / S.to_finset.card.to_rat
  else 0

-- Prove that the probability of event E is 1/2 given the sample space
theorem probability_of_event_E :
  probability {n | event_E n} sample_space = 1 / 2 := by
  sorry

end probability_of_event_E_l481_481211


namespace find_common_ratio_l481_481030

-- Definitions
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q ∧ 1 < q

def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ (n + 1)) / (1 - q)

def a3 := 8
def S3 := ∫ x in 0..2, (4 * x + 3)

-- Theorem
theorem find_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom_seq: increasing_geometric_sequence a q)
  (h_a3: a 3 = 8)
  (h_S3: S3 = ∫ x in 0..2, (4 * x + 3))
  (h_sum: geometric_sum a 2 = S3) : q = 2 :=
by 
  sorry

end find_common_ratio_l481_481030


namespace max_min_g_l481_481340

noncomputable def f (x : ℝ) : ℝ := 3 + Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := f(x^2) - (f(x))^2

theorem max_min_g :
  ∀ x ∈ set.Icc (1 : ℝ) (4 : ℝ), 
  (-6 ≤ g x ∧ g x ≤ -11) ∨ (-11 ≤ g x ∧ g x ≤ -6) :=
sorry

end max_min_g_l481_481340


namespace time_difference_l481_481808

def joey_time : ℕ :=
  let uphill := 12 / 6 * 60
  let downhill := 10 / 25 * 60
  let flat := 20 / 15 * 60
  uphill + downhill + flat

def sue_time : ℕ :=
  let downhill := 10 / 35 * 60
  let uphill := 12 / 12 * 60
  let flat := 20 / 25 * 60
  downhill + uphill + flat

theorem time_difference : joey_time - sue_time = 99 := by
  -- calculation steps skipped
  sorry

end time_difference_l481_481808


namespace ff_one_eq_half_l481_481538

def f (x : ℝ) : ℝ := 
  if x ≤ 1 then 2^x else Real.log x / Real.log 4

theorem ff_one_eq_half : f (f 1) = 1 / 2 := 
  sorry

end ff_one_eq_half_l481_481538


namespace racetrack_initial_amount_l481_481135

theorem racetrack_initial_amount : 
    ∃ x : ℝ, 
    (x > 0) ∧ 
    (2 * (2 * (2 * (2 * x - 60) - 60) - 60) - 60 = 0) :=
begin
    -- The exact x we are looking for
    use 52.5,
    split,
    -- prove x > 0
    linarith,
    -- prove the series of bets leads to 0
    -- leading to calculate
    sorry
end

end racetrack_initial_amount_l481_481135


namespace m_over_n_add_one_l481_481392

theorem m_over_n_add_one (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : (m + n : ℚ) / n = 10 / 7 :=
by
  sorry

end m_over_n_add_one_l481_481392


namespace length_be_l481_481422

theorem length_be (A B C D F E : Type*) [AddGroup A] [HasIntCast A] [HasSmul ℕ A] [HasInner A]
  (ABC_is_square : A = C - B)
  (BC_is_side_length_1 : ∥ C - B ∥ = 1)
  (F_midpoint_BC : F = (B + C) / 2)
  (E_perpendicular_A_DF : ∥ A - E ∥ = ∥ proj_line DF (A - E) ∥) :
  ∥ B - E ∥ = 1 := 
sorry

end length_be_l481_481422


namespace max_digit_sum_base_six_l481_481573

theorem max_digit_sum_base_six (n : ℕ) (h : n < 1728) : 
  ∃ m < 1728, base_digit_sum 6 m = 20 :=
sorry

end max_digit_sum_base_six_l481_481573


namespace number_of_women_l481_481441

theorem number_of_women (x : ℕ) 
  (h1 : 4 * x + 2 = 14) : 2 * (5 * x - 3) = 24 :=
by 
  ext
  sorry

end number_of_women_l481_481441


namespace total_crayons_l481_481313

-- We're given the conditions
def crayons_per_child : ℕ := 6
def number_of_children : ℕ := 12

-- We need to prove the total number of crayons.
theorem total_crayons (c : ℕ := crayons_per_child) (n : ℕ := number_of_children) : (c * n) = 72 := by
  sorry

end total_crayons_l481_481313


namespace Mona_joined_groups_l481_481852

theorem Mona_joined_groups (G : ℕ) (h : G * 4 - 3 = 33) : G = 9 :=
by
  sorry

end Mona_joined_groups_l481_481852


namespace kite_perimeter_is_correct_l481_481925

/-- Define the kite ABCD with given conditions -/
def Kite (A B C D : Type) : Prop :=
  ∃ (d : ℝ) (a : ℝ) (b : ℝ), 
    A = d ∧ B = a ∧ C = b ∧ 
    d ^ 2 + a ^ 2 = b ^ 2

/-- Defining lengths of sides of the kite ABCD -/
variables (AB BD AD DC : ℝ)
variables (AB_perpendicular_BD : AB * BD = 0)
variables (AB_length : AB = 10)
variables (BD_length : BD = 15)
variables (AD_length : AD = 7)
variables (DC_length : DC = 7)

theorem kite_perimeter_is_correct (A B C D : ℝ) : 
  (∃ (AB_perpendicular_BD AB_length BD_length AD_length DC_length : ℝ), 
     AB_perpendicular_BD * AB + BD + AD + DC == 42.5) :=
  sorry

end kite_perimeter_is_correct_l481_481925


namespace t_minus_s_eq_l481_481960

-- Define the class sizes.
def class_sizes : List ℕ := [60, 30, 20, 5, 5]

-- Define the number of teachers.
def total_teachers : ℕ := 6

-- Define the number of teaching teachers.
def teaching_teachers : ℕ := 5

-- Define the number of students.
def total_students : ℕ := 120

-- Define the average number of students per class for a randomly chosen teacher who is actually teaching.
def t : ℚ :=
  class_sizes.sum / teaching_teachers

-- Define the average number of students in the class for a randomly chosen student.
def s : ℚ :=
  class_sizes.map (λ n => n * n).sum / total_students

-- Prove that the difference t - s is -17.25.
theorem t_minus_s_eq : t - s = -17.25 :=
by
  have h1 : t = (60 + 30 + 20 + 5 + 5) / 5 := by rw [t] -- 24
  have h2 : s = (60 * 60 + 30 * 30 + 20 * 20 + 5 * 5 + 5 * 5) / 120 := by rw [s] -- 41.25
  rw [h1, h2]
  sorry

end t_minus_s_eq_l481_481960


namespace hyperbola_eccentricity_l481_481740

noncomputable def eccentricity_of_hyperbola (a : ℝ) (h : a > 0) : ℝ :=
  let b := 2  in
  let c := Math.sqrt(a^2 + b^2) in
  c / a

theorem hyperbola_eccentricity
  (a : ℝ) (h : a > 0)
  (intersects_circle : (x y : ℝ), (2 * x - a * y = 0) → ((x - 3)^2 + y^2 = 8))
  (MN_length : ∀ (M N : ℝ × ℝ), (M.1 - 3)^2 + M.2^2 = 8 ∧ (N.1 - 3)^2 + N.2^2 = 8 →
               (2 * M.1 - a * M.2 = 0) ∧ (2 * N.1 - a * N.2 = 0) →
               |M - N| = 4) :
  eccentricity_of_hyperbola a h = (3 * Math.sqrt 5) / 5 :=
begin
  sorry
end

end hyperbola_eccentricity_l481_481740


namespace family_reunion_l481_481311

-- Definitions of variables
variables {n S a b : ℕ}

-- Given conditions
def avg_excl_oldest := (S - b) / (n - 1) = 18
def avg_excl_youngest := (S - a) / (n - 1) = 20
def age_diff := b - a = 40

-- Theorem statement
theorem family_reunion (h1 : avg_excl_oldest) (h2 : avg_excl_youngest) (h3 : age_diff) : n = 21 :=
sorry

end family_reunion_l481_481311


namespace wire_length_approx_l481_481248

-- Definitions and conditions
def volume_cm3 : ℝ := 66      -- Volume in cubic centimeters
def diameter_mm : ℝ := 1      -- Diameter in millimeters
def π : ℝ := Real.pi          -- Pi constant

-- Convert given values to consistent units (meters)
def volume_m3 : ℝ := volume_cm3 * 1e-6 -- Convert cubic cm to cubic meters
def radius_m : ℝ := (diameter_mm * 1e-3) / 2 -- Convert diameter to meters and find radius

-- The length of the wire in meters
def length_of_wire : ℝ := volume_m3 / (π * radius_m^2)

theorem wire_length_approx :
  abs (length_of_wire - 84.029) < 1e-3 :=
by
  -- Placeholder for the proof
  sorry

end wire_length_approx_l481_481248


namespace total_oranges_correct_l481_481312

-- Define the conditions
def oranges_per_child : Nat := 3
def number_of_children : Nat := 4

-- Define the total number of oranges and the statement to be proven
def total_oranges : Nat := oranges_per_child * number_of_children

theorem total_oranges_correct : total_oranges = 12 := by
  sorry

end total_oranges_correct_l481_481312


namespace candied_cherries_total_l481_481862

-- Define the set of jesters and heights
def jesters : Fin 100 := {n | 1 ≤ n ∧ n ≤ 100}

-- Number of jesters each day (heights at most n)
def jesters_at_most (n : Fin 100) : Finset (Fin 100) :=
  {j | j ≤ n}

-- Median height condition for receiving candied cherries
def receives_candies_on_day (n : Fin 101) :=
  if n = 101 then ∀ (G : Finset (Fin 100)), G.card = 6 ∧ G.median = 50.5 -> 1
  else ∀ (G : Finset (Fin 100)), G.card = 6 ∧ G.median = n - 50 -> 2

-- Total number of candied cherries received
def total_candies : ℕ :=
  ∑ n in jesters, receives_candies_on_day n

-- Theorem statement
theorem candied_cherries_total :
  total_candies = 384160000 :=
  sorry

end candied_cherries_total_l481_481862


namespace constant_term_is_neg42_l481_481161

noncomputable def constantTermExpansion (x : ℝ) : ℝ :=
  (x^2 + 1) * (1 / sqrt(x) - 2)^5

theorem constant_term_is_neg42 :
  ∃ x : ℝ, constantTermExpansion x = -42 :=
by
  sorry

end constant_term_is_neg42_l481_481161


namespace area_larger_sphere_red_is_83_point_25_l481_481969

-- Define the radii and known areas

def radius_smaller_sphere := 4 -- cm
def radius_larger_sphere := 6 -- cm
def area_smaller_sphere_red := 37 -- square cm

-- Prove the area of the region outlined in red on the larger sphere
theorem area_larger_sphere_red_is_83_point_25 :
  ∃ (area_larger_sphere_red : ℝ),
    area_larger_sphere_red = 83.25 ∧
    area_larger_sphere_red = area_smaller_sphere_red * (radius_larger_sphere ^ 2 / radius_smaller_sphere ^ 2) :=
by {
  sorry
}

end area_larger_sphere_red_is_83_point_25_l481_481969


namespace necessary_not_sufficient_condition_l481_481841

-- Definitions of conditions
variable (x : ℝ)

-- Statement of the problem in Lean 4
theorem necessary_not_sufficient_condition (h : |x - 1| ≤ 1) : 2 - x ≥ 0 := sorry

end necessary_not_sufficient_condition_l481_481841


namespace find_fourth_month_sale_l481_481620

theorem find_fourth_month_sale (s1 s2 s3 s4 s5 : ℕ) (avg_sale nL5 : ℕ)
  (h1 : s1 = 5420)
  (h2 : s2 = 5660)
  (h3 : s3 = 6200)
  (h5 : s5 = 6500)
  (havg : avg_sale = 6300)
  (hnL5 : nL5 = 5)
  (h_average : avg_sale * nL5 = s1 + s2 + s3 + s4 + s5) :
  s4 = 7720 := sorry

end find_fourth_month_sale_l481_481620


namespace investment_C_investment_l481_481654

noncomputable def investment_C (C D investment_D total_profit D_profit : ℝ) : ℝ :=
  let C_profit := total_profit - D_profit
  let ratio := C / investment_D
  if ratio = C_profit / D_profit then C else 0

theorem investment_C_investment :
  ∀ (total_profit D_profit investment_D : ℝ), 
  total_profit = 500 → D_profit = 100 → investment_D = 1500 → 
  investment_C x investment_D total_profit D_profit = 6000 :=
by {
  intros,
  sorry
}

end investment_C_investment_l481_481654


namespace part_I_part_II_max_val_part_II_mon_inc_intervals_l481_481735

-- Definitions and conditions
def f (x : Real) : Real := Real.sin x + Real.cos x
noncomputable def F (x : Real) : Real := f x * f (-x) + (f x) ^ 2

-- Statement for Part I
theorem part_I (x : Real) (h : f x = 2 * f (-x)) :
  (Real.cos x)^2 - Real.sin x * Real.cos x) / (1 + (Real.sin x)^2)) = 6 / 11 := sorry

-- Statement for Part II
-- Maximum value of F(x)
theorem part_II_max_val :
  ∃ x : Real, F x = sqrt 2 + 1 := sorry

-- Monotonically increasing interval
theorem part_II_mon_inc_intervals (x : Real) :
  -3 * Real.pi / 8 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 8 + k * Real.pi ↔ ∀ k : ℤ := sorry

end part_I_part_II_max_val_part_II_mon_inc_intervals_l481_481735


namespace rich_total_distance_l481_481516

-- Definitions of distances walked
def distance1 := 20
def distance2 := 200
def total_distance1 := distance1 + distance2
def distance3 := 2 * total_distance1
def distance4 := 1.5 * distance3
def distance5 := 300
def total_distance2 := total_distance1 + distance3 + distance4 + distance5
def distance6 := 3 * total_distance2
def total_distance3 := total_distance2 + distance6
def distance7 := 0.75 * total_distance3
def total_distance_before_return := total_distance3 + distance7
def total_distance := 2 * total_distance_before_return

-- Theorem stating the total distance walked
theorem rich_total_distance : total_distance = 22680 := by 
  sorry

end rich_total_distance_l481_481516


namespace train_length_is_correct_l481_481274

noncomputable def length_of_train (t : ℝ) (v_train : ℝ) (v_man : ℝ) : ℝ :=
  let relative_speed : ℝ := (v_train - v_man) * (5/18)
  relative_speed * t

theorem train_length_is_correct :
  length_of_train 23.998 63 3 = 400 :=
by
  -- Placeholder for the proof
  sorry

end train_length_is_correct_l481_481274


namespace women_in_room_l481_481439

theorem women_in_room :
  ∃ (x : ℤ), 
    let men_initial := 4 * x,
        women_initial := 5 * x,
        men_now := men_initial + 2,
        women_left := women_initial - 3,
        women_doubled := 2 * women_left,
        men_now = 14 in
    2 * (5 * x - 3) = 24 :=
by
  sorry

end women_in_room_l481_481439


namespace area_of_triangle_PQR_l481_481774

-- Definitions based on the given conditions
variables {P Q R M N S : Type} [euclidean_geometry P Q R M N S]
variables [midpoint M P Q] [ratio N P R 2 1] [ratio S P M 1 2]
axiom area_MNS : area M N S = 10

-- Statement: area of triangle PQR
theorem area_of_triangle_PQR (h1 : midpoint M P Q)
                             (h2 : ratio N P R 2 1)
                             (h3 : ratio S P M 1 2)
                             (h4 : area M N S = 10) :
  area P Q R = 45 :=
by sorry

end area_of_triangle_PQR_l481_481774


namespace influenza_virus_diameter_l481_481537

theorem influenza_virus_diameter (n : ℤ) (h: 0.000000203 = 2.03 * 10^n) : n = -7 := 
by 
  sorry

end influenza_virus_diameter_l481_481537


namespace find_m_probability_l481_481316

noncomputable def probability_of_stack_height :=
  let total_crates := 15
  let crate_dims := {dim1 := 4, dim2 := 5, dim3 := 7}
  let desired_height := 50
  let total_height_ways := 3 ^ total_crates
  let valid_configurations := 560
  valid_configurations / total_height_ways

theorem find_m_probability :
  let m := 560 in
  ∃ n, (m.gcd n = 1) ∧ (m = 560) ∧ (probability_of_stack_height = m / (3 ^ 15)) :=
sorry

end find_m_probability_l481_481316


namespace factory_earns_8100_per_day_l481_481410

-- Define the conditions
def working_hours_machines := 23
def working_hours_fourth_machine := 12
def production_per_hour := 2
def price_per_kg := 50
def number_of_machines := 3

-- Calculate earnings
def total_earnings : ℕ :=
  let total_runtime_machines := number_of_machines * working_hours_machines
  let production_machines := total_runtime_machines * production_per_hour
  let production_fourth_machine := working_hours_fourth_machine * production_per_hour
  let total_production := production_machines + production_fourth_machine
  total_production * price_per_kg

theorem factory_earns_8100_per_day : total_earnings = 8100 :=
by
  sorry

end factory_earns_8100_per_day_l481_481410


namespace acute_triangle_AK_length_l481_481417

noncomputable def length_AK (ABC : Triangle) (H K A B C E D F G : Point) (circle_DE : Circle) : Float :=
  let CE := altitude C AB
  let BD := altitude B AC
  let CE_inter_BD := CE ∧ BD
  let circle_d := circle_DE.diameter D E
  let FG := line F G
  let intersect_FG_AH := FG ∧ (line A H)
  let BC := 25
  let BD_value := 20
  let BE := 7

  if valid := valid_setup_intersections CE_inter_BD intersect_FG_AH B C A random_trivial_type then 32 else -1

theorem acute_triangle_AK_length (ABC : Triangle)
  (H K A B C E D F G : Point) (circle_DE : Circle)
  (CE : altitude C AB)
  (BD : altitude B AC)
  (CE_inter_BD : CE ∧ BD = H)
  (circle_d : circle_DE.diameter D E)
  (FG : line F G)
  (intersect_FG_AH : FG ∧ (line A H) = K)
  (BC : Float) (BD_value : Float) (BE : Float) :
  BC = 25 → BD_value = 20 → BE= 7 → length_AK ABC H K A B C E D F G circle_DE = 32 :=
sorry

end acute_triangle_AK_length_l481_481417


namespace area_of_triangle_l481_481321

/-
Theorem: The area of a triangle whose vertices are at points A(1,2,3), B(3,2,1), and C(1,0,1) is 2 * √3.
-/

-- Define points A, B, and C
def A : (ℝ × ℝ × ℝ) := (1, 2, 3)
def B : (ℝ × ℝ × ℝ) := (3, 2, 1)
def C : (ℝ × ℝ × ℝ) := (1, 0, 1)

-- Compute vectors AB and AC
def vector_sub (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)
def AB := vector_sub A B
def AC := vector_sub A C

-- Compute the cross product of two vectors in 3D
def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)
def AB_cross_AC := cross_product AB AC

-- Compute the magnitude of a vector in 3D
def magnitude (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)
def area_parallelogram := magnitude AB_cross_AC

-- The area of the triangle is half of the area of the parallelogram
def area_triangle := 1 / 2 * area_parallelogram

theorem area_of_triangle :
  area_triangle = 2 * real.sqrt 3 := sorry

end area_of_triangle_l481_481321


namespace women_current_in_room_l481_481453

-- Definitions
variable (m w : ℕ) -- number of men and women
variable (x : ℕ) -- positive integer representing the scaling factor for initial ratio

-- Conditions
def initial_ratio (x : ℕ) : Prop := m = 4*x ∧ w = 5*x
def after_changes (x : ℕ) : Prop := (m + 2) = 14 ∧ (2 * (w - 3)) = 24

-- Theorem statement
theorem women_current_in_room (x : ℕ) (m w : ℕ) (h1 : initial_ratio x) (h2 : after_changes x) : w = 24 :=
by
  sorry

end women_current_in_room_l481_481453


namespace probability_grid_fully_black_after_two_rotations_l481_481604

-- Defining the problem conditions
def initial_probability : ℝ := 1 / 2
def black_after_first_rotation (n : ℕ) : ℝ := (3 / 4) ^ n
def total_probability (n : ℕ) : ℝ := initial_probability * black_after_first_rotation n

-- Proving the required probability
theorem probability_grid_fully_black_after_two_rotations :
  total_probability 8 = 6561 / 131072 := 
sorry

end probability_grid_fully_black_after_two_rotations_l481_481604


namespace range_of_a_l481_481036

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f' x ≥ a) → (a ≤ 2) :=
by
  sorry

end range_of_a_l481_481036


namespace sum_of_solutions_l481_481079

theorem sum_of_solutions (x y : ℝ) (h1 : y = 9) (h2 : x^2 + y^2 = 225) : 2 * x = 0 :=
by
  sorry

end sum_of_solutions_l481_481079


namespace ratio_A_to_B_l481_481942

theorem ratio_A_to_B (A B C : ℕ) (h1 : A + B + C = 406) (h2 : C = 232) (h3 : B = C / 2) : A / gcd A B = 1 ∧ B / gcd A B = 2 := 
by sorry

end ratio_A_to_B_l481_481942


namespace max_n_l481_481239

-- Define the conditions in Lean
variables {S : Type*} (A : ℕ → set S) (hS : fintype S) (h_card : fintype.card S = 2019)

-- Condition: union of any three subsets equals S
def three_union (i j k : ℕ) : Prop := (A i ∪ A j ∪ A k = set.univ)

-- Condition: union of any two subsets does not equal S
def two_union (i j : ℕ) : Prop := (A i ∪ A j ≠ set.univ)

-- The statement that the maximum value of n satisfying the conditions is 64
theorem max_n {n : ℕ} (h1 : ∀ i j k, i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k → three_union A i j k)
  (h2 : ∀ i j, i < n → j < n → i ≠ j → two_union A i j) :
  n ≤ 64 :=
sorry

end max_n_l481_481239


namespace limit_of_sequence_l481_481937

open Real Topology

/--
Let f(n) = (∛(n^2 + 2) - 5n^2) / (n - √(n^4 - n + 1)). 
Show that lim (n → ∞) f(n) = 5.
-/
theorem limit_of_sequence :
  tendsto (λ n : ℝ, (n^(2/3) - 5 * n^2) / (n - (n^2))) at_top (𝓝 5) :=
sorry

end limit_of_sequence_l481_481937


namespace sum_of_two_numbers_l481_481906

theorem sum_of_two_numbers (a b : ℕ) (h1 : (a + b) * (a - b) = 1996) (h2 : (a + b) % 2 = (a - b) % 2) (h3 : a + b > a - b) : a + b = 998 := 
sorry

end sum_of_two_numbers_l481_481906


namespace fraction_to_decimal_l481_481315

theorem fraction_to_decimal : (58 / 125 : ℚ) = 0.464 := 
by {
  -- proof omitted
  sorry
}

end fraction_to_decimal_l481_481315


namespace range_of_y0_l481_481021

noncomputable def hyperbola_is_satisfied (x0 y0 : ℝ) : Prop :=
  (x0^2 / 2) - y0^2 = 1

noncomputable def foci_product_neg (x0 y0 : ℝ) : Prop :=
  let f1 := (-ℝ.sqrt 3, 0)
  let f2 := (ℝ.sqrt 3, 0)
  ((-ℝ.sqrt 3 - x0) * (ℝ.sqrt 3 - x0) + y0^2) < 0

theorem range_of_y0 (x0 y0 : ℝ) (h1 : hyperbola_is_satisfied x0 y0) (h2 : foci_product_neg x0 y0) :
  -ℝ.sqrt 3 / 3 < y0 ∧ y0 < ℝ.sqrt 3 / 3 :=
sorry

end range_of_y0_l481_481021


namespace initial_pennies_l481_481877

theorem initial_pennies (initial: ℕ) (h : initial + 93 = 191) : initial = 98 := by
  sorry

end initial_pennies_l481_481877


namespace area_of_triangle_PBC_probability_l481_481786

variable (ABC : Type) (S : ℝ) (P : ABC → ABC → ABC → ℝ) (probability : ℝ)

-- Assume ABC is a triangle with an area S, P a point on AB
variable [points_on_side_AB : ∀ (A B C : ABC), (0 ≤ P A B C) ∧ (P A B C ≤ 1)]

-- Theorem statement
theorem area_of_triangle_PBC_probability (A B C : ABC) (h : ℝ) 
  (H_area : ∀ (A B C : ABC), (area_of_triangle A B C = S) → True) 
  (H_height : height_of_triangle_A_to_BC A B C = h) : 
  probability = 3 / 4 :=
sorry

end area_of_triangle_PBC_probability_l481_481786


namespace imaginary_condition_l481_481766

theorem imaginary_condition (m : ℝ) : 
  let Z := (m + 2 * complex.I) / (1 + complex.I) in 
  Z.im ≠ 0 → Z.re = 0 → m = -2 :=
by
  sorry

end imaginary_condition_l481_481766


namespace probability_color_change_l481_481976

theorem probability_color_change (cycle_green : ℕ) (cycle_yellow : ℕ) (cycle_red : ℕ) (watch_interval : ℕ) :
  cycle_green = 50 → cycle_yellow = 5 → cycle_red = 40 → watch_interval = 5 →
  (∃ p : ℚ, p = 3 / 19) :=
by
  intros h_green h_yellow h_red h_interval
  have h_cycle : cycle_green + cycle_yellow + cycle_red = 95 := by rw [h_green, h_yellow, h_red]; refl
  have h_total : 15 / 95 = 3 / 19 := by norm_num
  use 3 / 19
  exact h_total

end probability_color_change_l481_481976


namespace range_of_a_range_of_lambda_l481_481376

variable (a : ℝ) (f : ℝ → ℝ) (x₁ x₂ λ : ℝ)

def f := λ x, a * Real.log x + 0.5 * x ^ 2 - a * x

axiom h1 : a > 4
axiom h2 : ∃ x₁ x₂ : ℝ, ∀ x : ℝ, x > 0 → f.1 x = 0 ∧ x₁ ≠ x₂
axiom h3 : f x₁ + f x₂ < λ * (x₁ + x₂)

theorem range_of_a : a > 4 := 
by sorry

theorem range_of_lambda : λ ≥ Real.log 4 - 3 := 
by sorry

end range_of_a_range_of_lambda_l481_481376


namespace find_three_digit_numbers_l481_481137

def is_ascending (n : ℕ) : Prop :=
  let digits := (to_digits 10 n).reverse
  digits = list.sort digits

def has_same_starting_letter_in_russian (n : ℕ) : Prop := sorry -- Placeholder for the Russian language condition

def has_different_starting_letters_in_russian (n : ℕ) : Prop := sorry -- Placeholder for the Russian language condition

theorem find_three_digit_numbers :
  (∃ n1 n2 : ℕ, (100 ≤ n1 ∧ n1 < 1000 ∧ is_ascending n1 ∧ has_same_starting_letter_in_russian n1) ∧
                 (100 ≤ n2 ∧ n2 < 1000 ∧ (n2 % 111 = 0) ∧ has_different_starting_letters_in_russian n2) ∧ n1 = 147 ∧ n2 = 111) :=
sorry

end find_three_digit_numbers_l481_481137


namespace average_is_correct_l481_481199

theorem average_is_correct : 
  let numbers := [12, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]
  let n := numbers.length
  let given_incorrect_average := 858.5454545454545
  (numbers.sum / n) ≠ given_incorrect_average ∧ (numbers.sum / n) = 125830.7 :=
by
  sorry

end average_is_correct_l481_481199


namespace binomial_coefficient_x2_l481_481792

theorem binomial_coefficient_x2 : 
  (polynomial.coeff (polynomial.expand ℝ 10 (x + 1/x)^10) 2) = 210 :=
by
  -- We state the problem    
  sorry

end binomial_coefficient_x2_l481_481792


namespace sum_exp_to_polar_form_l481_481656

theorem sum_exp_to_polar_form :
  let theta1 := 3 * Real.pi / 14
  let theta2 := 17 * Real.pi / 28
  let average := (theta1 + theta2) / 2
  12 * Complex.exp (Complex.I * theta1) + 10 * Complex.exp (Complex.I * theta2) = 
  24 * Real.cos ((theta2 - theta1) / 2) * Complex.exp (Complex.I * average) := 
by
  let theta1 := 3 * Real.pi / 14
  let theta2 := 17 * Real.pi / 28
  let average := (theta1 + theta2) / 2
  have h_sum : 12 * Complex.exp (Complex.I * theta1) + 10 * Complex.exp (Complex.I * theta2) =
    24 * Real.cos ((theta2 - theta1) / 2) * Complex.exp (Complex.I * average), sorry
  exact h_sum

end sum_exp_to_polar_form_l481_481656


namespace integer_pairs_satisfying_equation_l481_481678

theorem integer_pairs_satisfying_equation:
  ∀ (a b : ℕ), a ≥ 1 → b ≥ 1 → a^(b^2) = b^a ↔ (a, b) = (1, 1) ∨ (a, b) = (16, 2) ∨ (a, b) = (27, 3) :=
by
  sorry

end integer_pairs_satisfying_equation_l481_481678


namespace initial_coffee_stock_l481_481621

theorem initial_coffee_stock (x : ℝ) :
  let initial_decaf := 0.40 * x
  let additional_coffee := 100
  let additional_decaf := 0.60 * additional_coffee
  let total_coffee := x + additional_coffee
  let total_decaf := initial_decaf + additional_decaf
  let decaf_ratio := 0.44
  0.44 * total_coffee = total_decaf → x = 400 :=
by {
  intro h,
  suffices : x = 400,
  rw this,
  },
sorry

end initial_coffee_stock_l481_481621


namespace least_n_such_that_f_over_g_equals_4_over_7_l481_481474

def f (n : ℕ) : ℕ :=
  if nat.sqrt n * nat.sqrt n = n then nat.sqrt n
  else 1 + f (n + 1)

def g (n : ℕ) : ℕ :=
  if nat.sqrt n * nat.sqrt n = n then nat.sqrt n
  else 2 + g (n + 2)

theorem least_n_such_that_f_over_g_equals_4_over_7 : 
  ∃ n : ℕ, n > 0 ∧ f n = 4 * (g n / 7) ∧ 
  ∀ m : ℕ, m > 0 → f m = 4 * (g m / 7) → n <= m :=
begin
  let n := 258,
  existsi n,
  split,
  -- Verify n is a positive number
  { exact nat.succ_pos 257 },
  split,
  -- Verify f(n)/g(n) = 4/7
  { sorry },
  -- Verify n is the smallest such number
  { intros m hm1 hm2,
    sorry }
end

end least_n_such_that_f_over_g_equals_4_over_7_l481_481474


namespace quadratic_has_two_real_roots_root_greater_than_three_l481_481378
noncomputable theory

-- Part 1: Prove that the quadratic equation always has two real roots.
theorem quadratic_has_two_real_roots (a : ℝ) : 
  let Δ := (a - 2)^2 in Δ ≥ 0 :=
sorry

-- Part 2: If the equation has one real root greater than 3, find the range of values for a.
theorem root_greater_than_three (a : ℝ) (h : ∃ x : ℝ, (x * x - a * x + a - 1 = 0 ∧ x > 3)) : 
  a > 4 :=
sorry

end quadratic_has_two_real_roots_root_greater_than_three_l481_481378


namespace smallest_y_for_perfect_square_l481_481263

theorem smallest_y_for_perfect_square (x y: ℕ) (h : x = 5 * 32 * 45) (hY: y = 2) : 
  ∃ v: ℕ, (x * y = v ^ 2) :=
by
  use 2
  rw [h, hY]
  -- expand and simplify
  sorry

end smallest_y_for_perfect_square_l481_481263


namespace probability_of_less_than_5_is_one_half_l481_481226

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l481_481226


namespace solve_system_of_equations_l481_481193

theorem solve_system_of_equations :
  ∀ x y : ℝ,
  (y^2 + 2*x*y + x^2 - 6*y - 6*x + 5 = 0) ∧ (y - x + 1 = x^2 - 3*x) ∧ (x ≠ 0) ∧ (x ≠ 3) →
  (x, y) = (-1, 2) ∨ (x, y) = (2, -1) ∨ (x, y) = (-2, 7) :=
by
  sorry

end solve_system_of_equations_l481_481193


namespace complement_in_U_l481_481713

def A : Set ℝ := { x : ℝ | |x - 1| > 3 }
def U : Set ℝ := Set.univ

theorem complement_in_U :
  (U \ A) = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
by
  sorry

end complement_in_U_l481_481713


namespace total_borrowed_books_proof_l481_481821

def borrowed_books_on_monday : ℕ := 40

def borrowed_percentage_increase : ℝ := 0.05
def borrowed_percentage_friday : ℝ := 0.40

def books_on_tuesday : ℕ := (40 + (0.05 * 40)).to_nat
def books_on_wednesday : ℕ := (42 + (0.05 * 42)).to_nat
def books_on_thursday : ℕ := (44 + (0.05 * 44)).to_nat
def books_on_friday : ℕ := (46 + (0.40 * 46)).to_nat

def total_borrowed_books : ℕ := 40 + books_on_tuesday + books_on_wednesday + books_on_thursday + books_on_friday

theorem total_borrowed_books_proof : total_borrowed_books = 236 :=
by
  unfold total_borrowed_books
  unfold books_on_tuesday
  unfold books_on_wednesday
  unfold books_on_thursday
  unfold books_on_friday
  sorry

end total_borrowed_books_proof_l481_481821


namespace sqrt_43_between_6_and_7_l481_481280

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 :=
by
  sorry

end sqrt_43_between_6_and_7_l481_481280


namespace range_of_m_values_l481_481033

theorem range_of_m_values {P Q : ℝ × ℝ} (hP : P = (-1, 1)) (hQ : Q = (2, 2)) (m : ℝ) :
  -3 < m ∧ m < -2 / 3 → (∃ (l : ℝ → ℝ), ∀ x y, y = l x → x + m * y + m = 0) :=
sorry

end range_of_m_values_l481_481033


namespace trig_identity_proof_l481_481231

theorem trig_identity_proof (α : ℝ) :
  (sin (3 * π - 4 * α))^2 + 4 * cos ((3 / 2) * π - 2 * α)^2 - 4 = cot (2 * α)^4 :=
by
  sorry

end trig_identity_proof_l481_481231


namespace sequence_AMS_ends_in_14_l481_481503

def start := 3
def add_two (x : ℕ) := x + 2
def multiply_three (x : ℕ) := x * 3
def subtract_one (x : ℕ) := x - 1

theorem sequence_AMS_ends_in_14 : 
  subtract_one (multiply_three (add_two start)) = 14 :=
by
  -- The proof would go here if required.
  sorry

end sequence_AMS_ends_in_14_l481_481503


namespace find_real_solutions_l481_481318

noncomputable def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem find_real_solutions (x : ℝ) :
  isPureImaginary ((x + complex.i) * (x + 1 + complex.i) * (x + 2 + 2 * complex.i)) ↔
    (x = -1 ∨ x = 1 ∨ x = 3) := 
sorry

end find_real_solutions_l481_481318


namespace interval_monotonicity_l481_481690

theorem interval_monotonicity {f : ℝ → ℝ} (h_diff : ∀ x, differentiable ℝ f x) 
  (h_cond : ∀ x, (x^2 - 3*x + 2) * (deriv f x) ≤ 0) :
  ∀ x ∈ set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2 :=
begin
  sorry
end

end interval_monotonicity_l481_481690


namespace cylindrical_tube_volume_difference_l481_481645

def volume (r: ℝ) (h: ℝ) : ℝ := π * r^2 * h

def radius_A : ℝ := 5 / π
def radius_B : ℝ := real.sqrt (244) / (2 * π)

def volume_A : ℝ := volume radius_A 12
def volume_B : ℝ := volume radius_B 10

theorem cylindrical_tube_volume_difference :
  π * (volume_B - volume_A) = 310 := by
sory

end cylindrical_tube_volume_difference_l481_481645


namespace period_of_f_f_monotonically_decreasing_intervals_l481_481375

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + 2 * sin x * cos x + 3 * cos x ^ 2

theorem period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
by
  exists π
  sorry

theorem f_monotonically_decreasing_intervals : 
  ∀ k : ℤ, ∀ x : ℝ, k * π + π / 8 ≤ x ∧ x ≤ k * π + 5 * π / 8 → 
  ∀ y : ℝ, x ≤ y ∧ y ≤ k * π + 5 * π / 8 → f x ≥ f y :=
by
  sorry

end period_of_f_f_monotonically_decreasing_intervals_l481_481375


namespace allocation_schemes_12_l481_481178

def number_of_allocation_schemes (doctors nurses : ℕ) (conditions : doctors = 2 ∧ nurses = 4) : ℕ :=
sorry

theorem allocation_schemes_12 (h : number_of_allocation_schemes 2 4 (by simp [*])) : h = 12 :=
sorry

end allocation_schemes_12_l481_481178


namespace number_of_women_l481_481445

theorem number_of_women (x : ℕ) 
  (h1 : 4 * x + 2 = 14) : 2 * (5 * x - 3) = 24 :=
by 
  ext
  sorry

end number_of_women_l481_481445


namespace minimize_distance_l481_481721

variable (a b : ℝ) (xa xb ya yb : ℝ)

noncomputable def vec_a : ℝ × ℝ := (xa, ya)
noncomputable def vec_b : ℝ × ℝ := (xb, yb)

theorem minimize_distance:
  (xa = 2) → 
  (ya = 0) → 
  (xb = 1) → 
  (yb = 0) → 
  ((1 : ℝ) = 1) -- This helps the angle condition as cos(60°) = 1/2
  ∃ x : ℝ, x = 1 ↔ 
  (∃ x : ℝ, (λ expr := sqrt (vec_a.1^2 + vec_a.2^2 - 2 * expr * (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2) + expr^2 * (vec_b.1^2 + vec_b.2^2)) = sqrt 3) x)
:=
by 
  intros h1 h2 h3 h4 h5
  use 1
  rw [h5]
  sorry

end minimize_distance_l481_481721


namespace exterior_angle_of_DEF_is_correct_l481_481515

theorem exterior_angle_of_DEF_is_correct :
  ∃ (DEA FEA : ℝ), 
    (DEA = (180 * (5 - 2) / 5)) ∧ 
    (FEA = (180 * (7 - 2) / 7)) ∧ 
    (∠ DEF = 360 - DEA - FEA) → 
    ∠ DEF ≈ 123.43 :=
begin
  intro h,
  cases h with DEA h1,
  cases h1 with FEA h2,
  cases h2 with h3 h4,
  sorry
end

end exterior_angle_of_DEF_is_correct_l481_481515


namespace problem1_problem2_problem3_l481_481045

def f (x : ℝ) (k : ℝ) : ℝ := 8 * x^2 + 16 * x - k
def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x

-- (1)
theorem problem1 (k : ℝ) : (∀ x ∈ set.Icc (-3 : ℝ) 3, f x k ≤ g x) → k ≥ 45 :=
by sorry

-- (2)
theorem problem2 (k : ℝ) : (∃ x ∈ set.Icc (-3 : ℝ) 3, f x k ≤ g x) → k ≥ -7 :=
by sorry

-- (3)
theorem problem3 (k : ℝ) : (∀ x1 ∈ set.Icc (-3 : ℝ) 3, ∀ x2 ∈ set.Icc (-3 : ℝ) 3, f x1 k ≤ g x2) → k ≥ 141 :=
by sorry

end problem1_problem2_problem3_l481_481045


namespace rachel_remaining_money_l481_481874

def initial_earnings : ℝ := 200
def lunch_expense : ℝ := (1/4) * initial_earnings
def dvd_expense : ℝ := (1/2) * initial_earnings
def total_expenses : ℝ := lunch_expense + dvd_expense
def remaining_amount : ℝ := initial_earnings - total_expenses

theorem rachel_remaining_money :
  remaining_amount = 50 := 
by
  sorry

end rachel_remaining_money_l481_481874


namespace find_f_2014_l481_481028

section
  variables {α : Type*} [AddGroup α] [HasSmul ℚ α]

  -- Given conditions
  variable (f : ℝ → ℝ)
  variable (g : ℝ → ℝ)

  axiom h1 : ∀ x, f(-x) = f(x)           -- f is an even function
  axiom h2 : ∀ x, g(-x) = -g(x)          -- g is an odd function
  axiom h3 : ∀ x, g(x) = f(x - 1)
  axiom h4 : g(3) = 2013

  -- Target statement
  theorem find_f_2014 : f(2014) = 2013 :=
  sorry
end

end find_f_2014_l481_481028


namespace man_works_total_hours_l481_481935

-- Define conditions
def regular_days_per_week := 5
def working_hours_per_day := 8
def regular_rate := 2.40
def overtime_rate := 3.20
def total_earnings := 432.00
def weeks := 4

-- Calculate regular work hours and regular earnings
def regular_work_hours := regular_days_per_week * working_hours_per_day * weeks
def regular_earnings := regular_work_hours * regular_rate

-- Calculate overtime earnings
def overtime_earnings := total_earnings - regular_earnings

-- Number of overtime hours
def overtime_hours := overtime_earnings / overtime_rate

-- Total hours worked
def total_hours_worked := regular_work_hours + overtime_hours

theorem man_works_total_hours : total_hours_worked = 175 := by
  sorry

end man_works_total_hours_l481_481935


namespace modular_expression_evaluation_l481_481303

theorem modular_expression_evaluation :
  (7 * 9⁻¹ + 3 * 7⁻¹) % 35 = 8 :=
by
  have h1 : 9⁻¹ % 35 = 4 := sorry
  have h2 : 7⁻¹ % 35 = 5 := sorry
  sorry

end modular_expression_evaluation_l481_481303


namespace problem1_problem2_l481_481044

-- Define sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x : ℝ | x < -2 ∨ x > 6}

-- Define the two proof problems as Lean statements
theorem problem1 (a : ℝ) : setA a ∩ setB = ∅ ↔ -2 ≤ a ∧ a ≤ 3 := by
  sorry

theorem problem2 (a : ℝ) : setA a ⊆ setB ↔ (a < -5 ∨ a > 6) := by
  sorry

end problem1_problem2_l481_481044


namespace geometric_sequence_sum_identity_l481_481472

variables {a : ℕ → ℝ} {n : ℕ}
noncomputable def sum_first_n_terms (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, f i

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum_identity {a : ℕ → ℝ} 
  (hn : n > 0)
  (hgeo : is_geometric_sequence a) :
  let X := sum_first_n_terms a n in
  let Y := sum_first_n_terms a (2 * n) in
  let Z := sum_first_n_terms a (3 * n) in
  Y * (Y - X) = X * (Z - X) :=
by
  sorry

end geometric_sequence_sum_identity_l481_481472


namespace correct_propositions_l481_481035

-- Define the propositions as constants
constant P1 : Prop
constant P2 : Prop
constant P3 : Prop
constant P4 : Prop
constant P5 : Prop

-- Define the conditions that these propositions are true
axiom P1_true : P1
axiom P2_true : P2
axiom P3_false : ¬P3
axiom P4_true : P4
axiom P5_true : P5

-- Create a theorem statement equivalent to proving that the correct propositions are exactly ①, ②, ④, ⑤.
theorem correct_propositions : (P1 ∧ P2 ∧ ¬P3 ∧ P4 ∧ P5) ↔ (P1_true ∧ P2_true ∧ P3_false ∧ P4_true ∧ P5_true) := by
  sorry

end correct_propositions_l481_481035


namespace remainder_is_2x_l481_481310

namespace PolynomialRepro

open Polynomial

noncomputable def poly1 : ℤ[X] := X^5 - 2 * X^3 + X - 1
noncomputable def poly2 : ℤ[X] := X^3 - X + 1
noncomputable def divisor : ℤ[X] := X^2 + X + 1

theorem remainder_is_2x :
  (poly1 * poly2) % divisor = 2 * X :=
sorry

end PolynomialRepro

end remainder_is_2x_l481_481310


namespace translated_function_value_l481_481540

theorem translated_function_value : 
  let g := λ x : ℝ, 2 * (Real.sin (2 * x))
  in g (5 * Real.pi / 6) = -Real.sqrt 3 := 
by
  let g := λ x : ℝ, 2 * (Real.sin (2 * x))
  show g (5 * Real.pi / 6) = -Real.sqrt 3
  sorry

end translated_function_value_l481_481540


namespace find_angle_A_l481_481071

theorem find_angle_A (B C: ℝ) (h: 3 * cos B * cos C + 1 = 3 * sin B * sin C + cos (2 * (π / 3))) :
  cos (π / 3) = 1 / 2 :=
by {
sor...

end find_angle_A_l481_481071


namespace space_filled_with_rhombic_dodecahedra_l481_481519

/-
  Given: Space can be filled completely using cubic cells (cubic lattice).
  To Prove: Space can be filled completely using rhombic dodecahedron cells.
-/

theorem space_filled_with_rhombic_dodecahedra :
  (∀ (cubic_lattice : Type), (∃ fill_space_with_cubes : (cubic_lattice → Prop), 
    ∀ x : cubic_lattice, fill_space_with_cubes x)) →
  (∃ (rhombic_dodecahedra_lattice : Type), 
      (∀ fill_space_with_rhombic_dodecahedra : rhombic_dodecahedra_lattice → Prop, 
        ∀ y : rhombic_dodecahedra_lattice, fill_space_with_rhombic_dodecahedra y)) :=
by {
  sorry
}

end space_filled_with_rhombic_dodecahedra_l481_481519


namespace simplify_expression_l481_481520

theorem simplify_expression (x y : ℤ) (h1 : x = -2) (h2 : y = -1) :
  (2 * (x - 2 * y) * (2 * x + y) - (x + 2 * y)^2 + x * (8 * y - 3 * x)) / (6 * y) = 2 :=
by sorry

end simplify_expression_l481_481520


namespace fraction_simplifies_l481_481077

def current_age_grant := 25
def current_age_hospital := 40

def age_in_five_years (current_age : Nat) : Nat := current_age + 5

def grant_age_in_5_years := age_in_five_years current_age_grant
def hospital_age_in_5_years := age_in_five_years current_age_hospital

def fraction_of_ages := grant_age_in_5_years / hospital_age_in_5_years

theorem fraction_simplifies : fraction_of_ages = (2 / 3) := by
  sorry

end fraction_simplifies_l481_481077


namespace m_over_n_add_one_l481_481391

theorem m_over_n_add_one (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : (m + n : ℚ) / n = 10 / 7 :=
by
  sorry

end m_over_n_add_one_l481_481391


namespace women_count_l481_481431

/-- 
Initially, the men and women in a room were in the ratio of 4:5.
Then, 2 men entered the room and 3 women left the room.
The number of women then doubled.
There are now 14 men in the room.
Prove that the number of women currently in the room is 24.
-/
theorem women_count (x : ℕ) (h1 : 4 * x + 2 = 14) (h2 : 2 * (5 * x - 3) = n) : 
  n = 24 :=
by
  sorry

end women_count_l481_481431


namespace count_valid_subsets_l481_481483

def setS : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def valid_subsets (T : Set ℕ) : Prop :=
  ∀ x ∈ T, (2 * x ∈ setS) → (2 * x ∈ T)

theorem count_valid_subsets : 
  {T : Set ℕ | valid_subsets T}.toFinset.card = 180 := 
sorry

end count_valid_subsets_l481_481483


namespace sqrt_43_between_6_and_7_l481_481283

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 := sorry

end sqrt_43_between_6_and_7_l481_481283


namespace algorithm_statements_handle_large_problems_l481_481170

theorem algorithm_statements_handle_large_problems :
  (∀ (input_output assignment conditional loop : Prop), 
    (loop ∧ 
     (large_computational_problems_involve_repetitive_calculations → 
      handle_difficult_problems(input_output, assignment, conditional, loop)))) :=
by
  sorry

end algorithm_statements_handle_large_problems_l481_481170


namespace probability_two_painted_faces_is_three_eighths_l481_481952

-- Define our conditions
def is_cut_into_64_smaller_cubes : Prop := true
def each_edge_has_2_painted_face_cubes : Prop := true
def cube_has_12_edges : Prop := true

-- Define the total number of smaller cubes
def total_smaller_cubes : ℕ := 64

-- Define the number of smaller cubes with exactly two painted faces
def smaller_cubes_with_two_painted_faces (edges : ℕ) (cubes_per_edge : ℕ) : ℕ := edges * cubes_per_edge

-- Define the probability calculation
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Main theorem
theorem probability_two_painted_faces_is_three_eighths :
  is_cut_into_64_smaller_cubes → each_edge_has_2_painted_face_cubes → cube_has_12_edges →
  probability (smaller_cubes_with_two_painted_faces 12 2) total_smaller_cubes = 3 / 8 :=
by
  intro h1 h2 h3
  calc
    probability (smaller_cubes_with_two_painted_faces 12 2) total_smaller_cubes
        = (12 * 2) / 64 : by sorry
    ... = 24 / 64 : by sorry
    ... = 3 / 8 : by sorry

end probability_two_painted_faces_is_three_eighths_l481_481952


namespace number_of_women_is_24_l481_481448

-- Define the variables and conditions
variables (x : ℕ) (men_initial : ℕ) (women_initial : ℕ) (men_current : ℕ) (women_current : ℕ)

-- representing the initial ratio and the changes
def initial_conditions : Prop :=
  men_initial = 4 * x ∧ women_initial = 5 * x ∧
  men_current = men_initial + 2 ∧ women_current = 2 * (women_initial - 3)

-- representing the current number of men
def current_men_condition : Prop := men_current = 14

-- The proof we need to generate
theorem number_of_women_is_24 (x : ℕ) (men_initial women_initial men_current women_current : ℕ)
  (h1 : initial_conditions x men_initial women_initial men_current women_current)
  (h2 : current_men_condition men_current) : women_current = 24 :=
by
  -- proof steps here
  sorry

end number_of_women_is_24_l481_481448


namespace quadratic_inequality_solution_l481_481068

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
  sorry

end quadratic_inequality_solution_l481_481068


namespace sequence_2016th_number_l481_481732

def sequence (n : ℕ) : ℤ :=
  match n % 4 with
  | 0 => 1
  | 1 => 1
  | 2 => -1
  | 3 => 0
  | _ => 0 -- this case will never be reached

theorem sequence_2016th_number : sequence 2015 = 0 :=
by
  sorry

end sequence_2016th_number_l481_481732


namespace speed_ratio_l481_481640

theorem speed_ratio (v_A v_B : ℝ) (L t : ℝ) 
  (h1 : v_A * t = (1 - 0.11764705882352941) * L)
  (h2 : v_B * t = L) : 
  v_A / v_B = 1.11764705882352941 := 
by 
  sorry

end speed_ratio_l481_481640


namespace john_height_in_feet_l481_481810

theorem john_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (months : ℕ) (inches_per_foot : ℕ) :
  initial_height = 66 → growth_rate = 2 → months = 3 → inches_per_foot = 12 → 
  (initial_height + growth_rate * months) / inches_per_foot = 6 := by
  intros h1 h2 h3 h4
  sorry

end john_height_in_feet_l481_481810


namespace irreducible_f_roots_of_f_l481_481468

def f : Polynomial ℤ := Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 2 * Polynomial.X ^ 3 + Polynomial.C (-1) * Polynomial.X + Polynomial.C (-1)

-- Part (a)
theorem irreducible_f : ¬ ∃ p q : Polynomial ℤ, p.degree > 0 ∧ q.degree > 0 ∧ f = p * q := 
sorry

-- Part (b)
theorem roots_of_f : 
  (∃ x, Polynomial.eval x f = 0) ∧ 
  x = - (1 - (√5 : ℤ)/2) ∨ x = - (1 + (√5 : ℤ)/2)  :=
sorry

end irreducible_f_roots_of_f_l481_481468


namespace monotonic_intervals_range_of_m_l481_481038

noncomputable def f (m x : ℝ) : ℝ := (mx - 1) * real.exp x - x^2

-- Part 1
theorem monotonic_intervals (m : ℝ) (h_slope : (f' 1) = e - 2) :
  (f 1 = (1 - 1) * real.exp 1 - 1^2) →
  ∃ (m : ℝ), m = 1 ∧ 
    (∀ x, x < 0 → f' m x > 0) ∧ 
    (∀ x, x > real.log 2 → f' sm x > 0) ∧ 
      ∀ x, 0 < x ∧ x < real.log 2 → f' m x < 0 :=
by sorry

-- Part 2
theorem range_of_m (m : ℝ) (h_ineq : ∃! x : ℤ, f m x < -x^2 + m*x - m ) : 
  (f m 0 = real.exp 0 / (0 * real.exp 0 - 0 + 1)) ∧ 
  (f m 1 = real.exp 1 / (1 * real.exp 1 - 1 + 1)) ∧ 
  (∀ x₀ > 0 ∧ x₀ < 1, f m x₀ = 0) → 
  (∀ x, m < real.exp x / (x * real.exp x - x + 1)) ∧ 
  ∃! m, (real.exp 2 / (2 * real.exp 2 - 1) ≤ m) ∧ m < 1 :=
by sorry

end monotonic_intervals_range_of_m_l481_481038


namespace array_sum_fraction_is_one_over_thirty_six_relatively_prime_equal_sum_of_terms_find_m_plus_n_l481_481605

noncomputable def array_sum_fraction : ℚ := 
  let entry (r c : ℕ) := (1 / (6 * 7^r : ℚ)) * (1 / 7^c)
  ∑' r, ∑' c, entry r c

theorem array_sum_fraction_is_one_over_thirty_six : array_sum_fraction = 1 / 36 := 
  by
  sorry

theorem relatively_prime_equal_sum_of_terms (m n : ℕ) :
  coprime m n → m = 1 → n = 36 → m + n = 37 :=
  by
  intro h h₁ h₂
  rw [h₁, h₂]
  rfl

theorem find_m_plus_n : (∃ m n : ℕ, coprime m n ∧ array_sum_fraction = (m : ℚ) / n) → ∃ m n, m + n = 37 :=
  by
  intro h
  rcases h with ⟨m, n, hmn_coprime, h_eq⟩
  have h_mn := array_sum_fraction_is_one_over_thirty_six
  rw [h_eq] at h_mn
  norm_cast at h_mn
  exact ⟨1, 36, relatively_prime_equal_sum_of_terms 1 36 hmn_coprime rfl rfl⟩

end array_sum_fraction_is_one_over_thirty_six_relatively_prime_equal_sum_of_terms_find_m_plus_n_l481_481605


namespace largest_percentage_increase_l481_481285

theorem largest_percentage_increase :
  let students : ℕ → ℕ := λ y,
    match y with
    | 2002 => 70
    | 2003 => 77
    | 2004 => 85
    | 2005 => 89
    | 2006 => 95
    | 2007 => 104
    | 2008 => 112
    | _ => 0
    end,
  let percentage_increase (a b : ℕ) : ℚ :=
    ((students b - students a : ℚ) / students a) * 100,
  let p_inc_2003_2004 := percentage_increase 2003 2004,
  let p_inc_2002_2003 := percentage_increase 2002 2003,
  let p_inc_2004_2005 := percentage_increase 2004 2005,
  let p_inc_2005_2006 := percentage_increase 2005 2006,
  let p_inc_2006_2007 := percentage_increase 2006 2007,
  let p_inc_2007_2008 := percentage_increase 2007 2008
in
  p_inc_2003_2004 > p_inc_2002_2003 ∧
  p_inc_2003_2004 > p_inc_2004_2005 ∧
  p_inc_2003_2004 > p_inc_2005_2006 ∧
  p_inc_2003_2004 > p_inc_2006_2007 ∧
  p_inc_2003_2004 > p_inc_2007_2008 :=
sorry

end largest_percentage_increase_l481_481285


namespace min_value_eq_9_l481_481256

-- Defining the conditions
variable (a b : ℝ)
variable (ha : a > 0) (hb : b > 0)
variable (h_eq : a - 2 * b = 0)

-- The goal is to prove the minimum value of (1/a) + (4/b) is 9
theorem min_value_eq_9 (ha : a > 0) (hb : b > 0) (h_eq : a - 2 * b = 0) 
  : ∃ (m : ℝ), m = 9 ∧ (∀ x, x = 1/a + 4/b → x ≥ m) :=
sorry

end min_value_eq_9_l481_481256


namespace description_of_T_l481_481305

-- Define the set T
def T : Set (ℝ × ℝ) := 
  {p | (p.1 = 1 ∧ p.2 ≤ 9) ∨ (p.2 = 9 ∧ p.1 ≤ 1) ∨ (p.2 = p.1 + 8 ∧ p.1 ≥ 1)}

-- State the formal proof problem: T is three rays with a common point
theorem description_of_T :
  (∃ p : ℝ × ℝ, p = (1, 9) ∧ 
    ∀ q ∈ T, 
      (q.1 = 1 ∧ q.2 ≤ 9) ∨ 
      (q.2 = 9 ∧ q.1 ≤ 1) ∨ 
      (q.2 = q.1 + 8 ∧ q.1 ≥ 1)) :=
by
  sorry

end description_of_T_l481_481305


namespace expansion_coefficient_l481_481670

theorem expansion_coefficient :
  (∃ (A B : ℝ) (n : ℕ), 
    A = (λ x : ℝ, (sqrt x - (1 / x) + 1)^7) ∧ 
    B = (λ x : ℝ, 35 * x^2) ∧ 
    ∀ x, (function_to_find_coefficient A n) x = B x) :=
sorry

end expansion_coefficient_l481_481670


namespace women_current_in_room_l481_481451

-- Definitions
variable (m w : ℕ) -- number of men and women
variable (x : ℕ) -- positive integer representing the scaling factor for initial ratio

-- Conditions
def initial_ratio (x : ℕ) : Prop := m = 4*x ∧ w = 5*x
def after_changes (x : ℕ) : Prop := (m + 2) = 14 ∧ (2 * (w - 3)) = 24

-- Theorem statement
theorem women_current_in_room (x : ℕ) (m w : ℕ) (h1 : initial_ratio x) (h2 : after_changes x) : w = 24 :=
by
  sorry

end women_current_in_room_l481_481451


namespace ac_bc_ge_sqrt2_triangles_abc_gap_similar_l481_481824

variables {A B C D F G P : Type} [EuclideanGeometry A] [EuclideanGeometry B] 
  [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry F]
  [EuclideanGeometry G] [EuclideanGeometry P]

-- Given conditions
variables (ABC : Triangle A B C)
variables (G : Point)
variables (hGcen : is_centroid G ABC)
variables (hCobtuse : ∡BAC > 90°)
variables (AD_option : line A D := median ABC A B C)
variables (CF_option : line C F := median ABC C A B)
variables (hConcyclic : concyclic B D G F)
variables (P : Point)
variables (hAGCP_parallelogram : parallelogram A G C P)
variables (hP_on_BG : P ∈ line G B)

-- Proof 1: AC / BC >= sqrt(2)
theorem ac_bc_ge_sqrt2 (h_conditions : Conditions AD_option CF_option hConcyclic hCobtuse) :
  AC / BC >= sqrt(2) := sorry

-- Proof 2: Triangle ABC and GAP are similar
theorem triangles_abc_gap_similar (h_conditions : Conditions AD_option CF_option hConcyclic hAGCP_parallelogram hP_on_BG) :
  similar_triangles ABC GAP := sorry

end ac_bc_ge_sqrt2_triangles_abc_gap_similar_l481_481824


namespace sum_face_angles_of_inner_polyhedral_angle_lt_outer_l481_481136

theorem sum_face_angles_of_inner_polyhedral_angle_lt_outer
  (O : Point)
  (A : list Point)
  (B : list Point)
  [convex_polyhedral_angle O A]
  [convex_polyhedral_angle O B]
  (h_inner : ∀ i j, A[i] ∈ convex_hull (B[j]))
  : sum_face_angles (O, A) < sum_face_angles (O, B) :=
sorry

end sum_face_angles_of_inner_polyhedral_angle_lt_outer_l481_481136


namespace berengere_needs_to_contribute_l481_481652

-- Definition and conditions
def pie_cost : ℝ := 8
def lucas_cad : ℝ := 10
def euro_to_cad : ℝ := 1.5

-- Conversion from CAD to EUR
def lucas_eur := lucas_cad / euro_to_cad

-- Finding the amount Berengere needs to contribute
noncomputable def berengere_contribution := pie_cost - lucas_eur

-- The theorem we need to prove
theorem berengere_needs_to_contribute : berengere_contribution = 4 / 3 :=
sorry

end berengere_needs_to_contribute_l481_481652


namespace find_B_l481_481120

-- Define the infinite nested radical as a limit
noncomputable def nested_radical : ℝ :=
  Real.lim (λ (n : ℕ), (Real.sqrt ∘ (λ x, 10 + x))^[n] 0)

-- State the theorem
theorem find_B : ∃ B : ℤ, B = Int.floor (10 + nested_radical) ∧ B = 13 :=
by
  use 13
  sorry

end find_B_l481_481120


namespace intersection_square_distance_84_l481_481536

noncomputable def square_distance_between_intersections (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
let eq1 := (x - (C1.1))^2 + (y - (C1.2))^2 - r1^2
let eq2 := (x - (C2.1))^2 + (y - (C2.2))^2 - r2^2
let intersections := {p : ℝ × ℝ // eq1 = 0 ∧ eq2 = 0}
let distance := (C D : intersections), dist (C.1, C.2) (D.1, D.2)
distance^2

theorem intersection_square_distance_84 :
  square_distance_between_intersections (1,2) (1,8) 5 (sqrt 13) = 84 :=
by
  sorry

end intersection_square_distance_84_l481_481536


namespace cost_of_paving_floor_l481_481237

theorem cost_of_paving_floor :
  let L := 5.5
  let W := 3.75
  let R := 800
  let Area := L * W
  let Cost := Area * R
  Cost = 16500 := by
  let L := 5.5
  let W := 3.75
  let R := 800
  let Area := L * W
  let Cost := Area * R
  sorry

end cost_of_paving_floor_l481_481237


namespace value_of_expression_l481_481060

-- Given conditions
variable (n : ℤ)
def m : ℤ := 4 * n + 3

-- Main theorem statement
theorem value_of_expression (n : ℤ) : 
  (m n)^2 - 8 * (m n) * n + 16 * n^2 = 9 := 
  sorry

end value_of_expression_l481_481060


namespace problem_l481_481037

def f (x : ℝ) : ℝ :=
  1 / (2 * (Real.tan x)) + (Real.sin (x / 2) * Real.cos (x / 2)) / (2 * (Real.cos (x / 2))^2 - 1)

theorem problem (h : f (π / 8) = sqrt 2) : f (π / 8) = sqrt 2 := 
  sorry

end problem_l481_481037


namespace equidistant_point_l481_481679

theorem equidistant_point 
  (x y : ℝ) 
  (h1 : y = 2 * x ^ 2 - 1)
  (h2 : x = 4 * y ^ 2 - 2) : 
  (x, y) = (1/8, 1/4) :=
begin
  -- proof steps skipped
  sorry
end

end equidistant_point_l481_481679


namespace polygonal_line_even_segments_l481_481183

theorem polygonal_line_even_segments {n : ℕ} (h : ∀ (segments : Fin n → (ℝ × ℝ) × (ℝ × ℝ)), 
  (closed_self_intersecting_polygonal_line segments → (∀ i, ∃! j, segment_intersects segments i j))) : 
  Even n :=
by 
  sorry

end polygonal_line_even_segments_l481_481183


namespace min_rectilinear_distance_to_parabola_l481_481939

theorem min_rectilinear_distance_to_parabola :
  ∃ t : ℝ, ∀ t', (|t' + 1| + t'^2) ≥ (|t + 1| + t^2) ∧ (|t + 1| + t^2) = 3 / 4 := sorry

end min_rectilinear_distance_to_parabola_l481_481939


namespace find_a_n_l481_481707

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_sum : ∀ n : ℕ, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n : ℕ, a n = Real.sqrt ↑n - Real.sqrt (↑n - 1) :=
by
  sorry

end find_a_n_l481_481707


namespace hyperbola_eccentricity_l481_481828

variable {a b c e : ℝ}

-- Definitions according to the problem conditions
def hyperbola_eq := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 
def focus_eq := c = Real.sqrt (a^2 + b^2)
def slope_of_line_through_focus := ∀ (x y : ℝ), y = x + c -- y = x + c passing through (-c, 0)
def asymptotes := ∀ (x y : ℝ), y = (b/a) * x ∨ y = -(b/a) * x

-- Key condition given in the problem
def intersection_ratio := |F.1 - A.1| / |F.1 - B.1| = 1/2 -- where coordinates derived

-- The main theorem to prove.
theorem hyperbola_eccentricity (h1 : hyperbola_eq) (h2 : a > b) (h3 : b > 0) (h4 : focus_eq) (h5 : intersection_ratio) : 
  e = Real.sqrt 10 := 
sorry

end hyperbola_eccentricity_l481_481828


namespace angle_between_nonzero_vectors_l481_481470

namespace VectorAngleProof

variables {V : Type*} [inner_product_space ℝ V] [nontrivial V]

def min_dot_product_value (a b : V) : Prop :=
  let x : ℕ → V := λ n, if n % 2 = 0 then a else b
  let y : ℕ → V := λ n, if n % 2 = 0 then b else a
  (∑ i in finset.range 4, ⟪x i, y i⟫) = 4 * ∥a∥^2

def angle_between_vectors (a b : V) : Prop :=
  real.angle_between a b = real.pi / 3

theorem angle_between_nonzero_vectors (a b : V)
  (h1 : a ≠ 0 ∧ b ≠ 0)
  (h2 : ∥b∥ = 2 * ∥a∥)
  (h3 : min_dot_product_value a b) :
  angle_between_vectors a b :=
sorry

end VectorAngleProof

end angle_between_nonzero_vectors_l481_481470


namespace original_gain_percentage_is_5_l481_481956

def costPrice : ℝ := 200
def newCostPrice : ℝ := costPrice * 0.95
def desiredProfitRatio : ℝ := 0.10
def newSellingPrice : ℝ := newCostPrice * (1 + desiredProfitRatio)
def originalSellingPrice : ℝ := newSellingPrice + 1

theorem original_gain_percentage_is_5 :
  ((originalSellingPrice - costPrice) / costPrice) * 100 = 5 :=
by 
  sorry

end original_gain_percentage_is_5_l481_481956


namespace area_of_quadrilateral_FDBG_proof_l481_481567

noncomputable def area_of_quadrilateral_FDBG : ℝ :=
  let AB : ℝ := 60 in
  let AC : ℝ := 45 in
  let area_ABC : ℝ := 270 in
  let AD : ℝ := AB / 2 in
  let AE : ℝ := AC / 2 in
  let DE := 0.0 -- Placeholder since intersection via angle bisector would be needed
  let F := 0.0 -- Placeholder for F coordinate
  let G := 86.93 -- The problem reduces to showing area FDBG equals this value
  G

theorem area_of_quadrilateral_FDBG_proof :
  area_of_quadrilateral_FDBG = 86.93 :=
  by
  -- Proof goes here
  sorry

end area_of_quadrilateral_FDBG_proof_l481_481567


namespace find_B_l481_481119

-- Define the infinite nested radical as a limit
noncomputable def nested_radical : ℝ :=
  Real.lim (λ (n : ℕ), (Real.sqrt ∘ (λ x, 10 + x))^[n] 0)

-- State the theorem
theorem find_B : ∃ B : ℤ, B = Int.floor (10 + nested_radical) ∧ B = 13 :=
by
  use 13
  sorry

end find_B_l481_481119


namespace sarah_likes_digits_l481_481854

theorem sarah_likes_digits : ∀ n : ℕ, n % 8 = 0 → (n % 10 = 0 ∨ n % 10 = 4 ∨ n % 10 = 8) :=
by
  sorry

end sarah_likes_digits_l481_481854


namespace quadratic_root_equation_l481_481360

-- Define the conditions given in the problem
variables (a b x : ℝ)

-- Assertion for a ≠ 0
axiom a_ne_zero : a ≠ 0

-- Root assumption
axiom root_assumption : (x^2 + b * x + a = 0) → x = -a

-- Lean statement to prove that b - a = 1
theorem quadratic_root_equation (h : x^2 + b * x + a = 0) : b - a = 1 :=
sorry

end quadratic_root_equation_l481_481360


namespace chess_tournament_participants_l481_481760

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 := 
by sorry

end chess_tournament_participants_l481_481760


namespace series_simplify_l481_481151

-- Define the sum of the first k natural numbers
def sum_natural (k : ℕ) : ℕ :=
  (k * (k + 1)) / 2

-- Define the nth term in the series
def nth_term (k : ℕ) : ℚ :=
  1 / sum_natural k

-- Define the series
def series (n : ℕ) : ℚ :=
  1 + (Finset.range n).sum (λ k, nth_term (k + 1))

-- The main statement
theorem series_simplify (n : ℕ) : series n = 2 * n / (n + 1) :=
by
  sorry

end series_simplify_l481_481151


namespace fraction_evaluation_l481_481675

theorem fraction_evaluation :
  (18 / 42) - (2 / 9) + (1 / 14) = (5 / 18) :=
by
  -- Proof goes here
  sorry

end fraction_evaluation_l481_481675


namespace statement1_statement2_statement3_statement4_correctness_A_l481_481354

variables {a b : Line} {α β γ : Plane}

def perpendicular (a : Line) (α : Plane) : Prop := sorry
def parallel (a b : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- Statement ①: If a ⊥ α and b ⊥ α, then a ∥ b
theorem statement1 (h1 : perpendicular a α) (h2 : perpendicular b α) : parallel a b := sorry

-- Statement ②: If a ⊥ α, b ⊥ β, and a ∥ b, then α ∥ β
theorem statement2 (h1 : perpendicular a α) (h2 : perpendicular b β) (h3 : parallel a b) : parallel_planes α β := sorry

-- Statement ③: If γ ⊥ α and γ ⊥ β, then α ∥ β
theorem statement3 (h1 : perpendicular γ α) (h2 : perpendicular γ β) : parallel_planes α β := sorry

-- Statement ④: If a ⊥ α and α ⊥ β, then a ∥ β
theorem statement4 (h1 : perpendicular a α) (h2 : parallel_planes α β) : parallel a b := sorry

-- The correct choice is A: Statements ① and ② are correct
theorem correctness_A : statement1_correct ∧ statement2_correct := sorry

end statement1_statement2_statement3_statement4_correctness_A_l481_481354


namespace student_mentor_selection_l481_481255

theorem student_mentor_selection:
  ∃ (selection_schemes : ℕ), 
    (∃ mentors students : ℕ, mentors = 2 ∧ students = 5 ∧ students_choose_from_mentors students mentors 3) 
    ∧ selection_schemes = 20 :=
sorry

end student_mentor_selection_l481_481255


namespace dessert_menu_count_l481_481625

-- Define the list of desserts
inductive Dessert
| tart
| cookie
| sorbet
| flan

open Dessert

-- Define conditions
def noConsecutiveRepeats (menu : List Dessert) : Prop :=
  ∀ i, (i < menu.length - 1) → (menu[i] ≠ menu[i + 1])

def fixedDesserts (menu : List Dessert) : Prop :=
  menu.nth 0 = some flan ∧ menu.nth 2 = some tart

-- Prove the number of different menus satisfying the conditions
theorem dessert_menu_count :
  (∃ menus : List (List Dessert),
    (∀ menu ∈ menus, noConsecutiveRepeats menu ∧ fixedDesserts menu) ∧ 
    menus.length = 243) :=
sorry

end dessert_menu_count_l481_481625


namespace determine_a_l481_481494

theorem determine_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f(x) = |x - a| - 2) (h2 : ∀ x, |f x| < 1 ↔ x ∈ set.Ioo (-2 : ℝ) 0 ∨ x ∈ set.Ioo (2 : ℝ) 4) : a = 1 :=
sorry

end determine_a_l481_481494


namespace correct_statements_l481_481665

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem correct_statements :
  (∀ x : ℝ, 2 * Real.cos (2 * x - 2 * Real.pi / 3) = f x) ∧ 
  (∀ x : ℝ, f (-Real.pi / 6 + x) = f (-Real.pi / 6 - x)) :=
by
  sorry

end correct_statements_l481_481665


namespace RachelLeftoverMoney_l481_481867

def RachelEarnings : ℝ := 200
def SpentOnLunch : ℝ := (1/4) * RachelEarnings
def SpentOnDVD : ℝ := (1/2) * RachelEarnings
def TotalSpent : ℝ := SpentOnLunch + SpentOnDVD

theorem RachelLeftoverMoney : RachelEarnings - TotalSpent = 50 := by
  sorry

end RachelLeftoverMoney_l481_481867


namespace selection_methods_count_l481_481191

/-- Consider a school with 16 teachers, divided into four departments (First grade, Second grade, Third grade, and Administrative department), with 4 teachers each. 
We need to select 3 leaders such that not all leaders are from the same department and at least one leader is from the Administrative department. 
Prove that the number of different selection methods that satisfy these conditions is 336. -/
theorem selection_methods_count :
  let num_teachers := 16
  let teachers_per_department := 4
  ∃ (choose : ℕ → ℕ → ℕ), 
  choose num_teachers 3 = 336 :=
  sorry

end selection_methods_count_l481_481191


namespace primes_between_80_and_90_l481_481051

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

theorem primes_between_80_and_90 : 
  (set.count set_of (λ n, 80 ≤ n ∧ n ≤ 90 ∧ is_prime n)) = 2 :=
by
  sorry

end primes_between_80_and_90_l481_481051


namespace num_odd_digits_base7_528_l481_481325

theorem num_odd_digits_base7_528 : 
  let base7_digits (n : ℕ) : List ℕ := 
    if n = 0 then [0] else List.unfoldr (λ x, if x = 0 then none else some (x % 7, x / 7)) n in
  List.countp (λ d, d % 2 = 1) (base7_digits 528) = 4 := 
by 
  sorry

end num_odd_digits_base7_528_l481_481325


namespace sequence_general_term_l481_481017

theorem sequence_general_term (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = 2 ^ n - 1) 
  (h2 : ∀ n, a 1 = S 1 ∧ (∀ n ≥ 2, a n = S n - S (n - 1))) : 
  a n = 2 ^ (n - 1) :=
  sorry


end sequence_general_term_l481_481017


namespace women_in_room_l481_481440

theorem women_in_room :
  ∃ (x : ℤ), 
    let men_initial := 4 * x,
        women_initial := 5 * x,
        men_now := men_initial + 2,
        women_left := women_initial - 3,
        women_doubled := 2 * women_left,
        men_now = 14 in
    2 * (5 * x - 3) = 24 :=
by
  sorry

end women_in_room_l481_481440


namespace area_on_larger_sphere_l481_481964

-- Define the variables representing the radii and the given area on the smaller sphere
variable (r1 r2 : ℝ) (area1 : ℝ)

-- Given conditions
def conditions : Prop :=
  r1 = 4 ∧ r2 = 6 ∧ area1 = 37

-- Define the statement that we need to prove
theorem area_on_larger_sphere (h : conditions r1 r2 area1) : 
  let area2 := area1 * (r2^2 / r1^2) in
  area2 = 83.25 :=
by
  -- Insert the proof here
  sorry

end area_on_larger_sphere_l481_481964


namespace complex_number_quadrant_l481_481032

noncomputable def z : ℂ := (3 - I) / (2 + I)

def quadrant (z : ℂ) : string :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
  else "On axis"

theorem complex_number_quadrant :
  quadrant z = "Fourth quadrant" :=
by
  sorry

end complex_number_quadrant_l481_481032


namespace power_function_increasing_iff_l481_481769

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_increasing_iff (a : ℝ) : 
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → power_function a x1 < power_function a x2) ↔ a > 0 := 
by
  sorry

end power_function_increasing_iff_l481_481769


namespace equalize_costs_l481_481103

theorem equalize_costs (A B C : ℝ) (h : A < B ∧ B < C) :
  let total_expenses := A + B + C
      per_person_share := total_expenses / 3
  in (per_person_share - A) = (B + C - 2 * A) / 3 :=
sorry

end equalize_costs_l481_481103


namespace general_term_formula_sum_first_n_terms_l481_481018

-- General term formula for the sequence {a_n}
theorem general_term_formula {a : ℕ → ℤ} (a1 a2 : ℤ) (h1 : a 1 = a1) (h2 : a 2 = a2) 
  (geo : ∀ n : ℕ, a (n+1) - 1 = 2 * (a n - 1)) :
  ∀ n : ℕ, a n = 2^(n-1) + 1 :=
by
  sorry

-- Sum of the first n terms of the sequence {b_n}
theorem sum_first_n_terms {a : ℕ → ℤ} {b : ℕ → ℤ} (a1 a2 : ℤ) (h1 : a 1 = a1) (h2 : a 2 = a2) 
  (geo : ∀ n : ℕ, a (n+1) - 1 = 2 * (a n - 1)) (bn : ∀ n : ℕ, b n = n * a n) :
  ∀ n : ℕ, (∑ i in finset.range n, b (i + 1)) = (n - 1) * 2^(n+1) + (n^2 + n + 4) / 2 :=
by
  sorry

end general_term_formula_sum_first_n_terms_l481_481018


namespace smallest_positive_period_of_f_is_pi_maximum_value_of_f_analytical_expression_for_g_range_of_h_on_interval_l481_481384

section

variable (x : ℝ)

-- Given conditions
def a : ℝ × ℝ := (2 * sin x, -√3 * cos x)
def b : ℝ × ℝ := (cos x, 2 * cos x)
def f : ℝ → ℝ := λ x, (a.1 * b.1 + a.2 * b.2)
def g : ℝ → ℝ := λ x, 2 * sin (2 * x + π / 3 )
def h : ℝ → ℝ := λ x, (deriv g x)

-- Proof problem statements
theorem smallest_positive_period_of_f_is_pi : (∃ p > 0, ∀ x, f (x + p) = f x) := sorry
theorem maximum_value_of_f : (∃ x, f x = 2 - √3) := sorry
theorem analytical_expression_for_g : g = λ x, 2 * sin (2 * x + π / 3) := sorry
theorem range_of_h_on_interval : ( ∀ x, 0 ≤ x ∧ x ≤ π / 2 → -2 ≤ h x ∧ h x ≤ 2) := sorry

end

end smallest_positive_period_of_f_is_pi_maximum_value_of_f_analytical_expression_for_g_range_of_h_on_interval_l481_481384


namespace gharial_fish_requirement_l481_481314

-- Define the conditions of the problem
def frogs_needed (flies_per_day : ℕ) (frogs_per_fly : ℕ) : ℕ :=
  flies_per_day / frogs_per_fly

def fish_needed (frogs_per_day : ℕ) (fish_per_frog : ℕ) : ℕ :=
  frogs_per_day / fish_per_frog

def fish_per_gharial (total_fish_needed : ℕ) (num_gharials : ℕ) : ℕ :=
  total_fish_needed / num_gharials

-- Given conditions
constant flies_per_day : ℕ := 32400
constant frogs_per_fly : ℕ := 30
constant fish_per_frog : ℕ := 8
constant num_gharials : ℕ := 9

-- Calculate intermediate values based on given conditions
def total_frogs_needed := frogs_needed flies_per_day frogs_per_fly
def total_fish_needed := fish_needed total_frogs_needed fish_per_frog
def fish_per_gharial_needed := fish_per_gharial total_fish_needed num_gharials

-- The theorem stating the problem to solve
theorem gharial_fish_requirement : fish_per_gharial_needed = 15 :=
by
  sorry

end gharial_fish_requirement_l481_481314


namespace solve_equation_1_solve_equation_2_l481_481882

namespace Proofs

theorem solve_equation_1 (x : ℝ) :
  (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 6) = 6 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 :=
by
  sorry

end Proofs

end solve_equation_1_solve_equation_2_l481_481882


namespace find_cos_and_sin_sum_l481_481716

theorem find_cos_and_sin_sum (α : ℝ) (h₁ : sin α = 3 / 5) (h₂ : 0 < α) (h₃ : α < π / 2) :
  cos α = 4 / 5 ∧ sin (α + π / 4) = 7 * sqrt 2 / 10 := by
  sorry

end find_cos_and_sin_sum_l481_481716


namespace red_marked_area_on_larger_sphere_l481_481965

-- Define the conditions
def r1 : ℝ := 4 -- radius of the smaller sphere
def r2 : ℝ := 6 -- radius of the larger sphere
def A1 : ℝ := 37 -- area marked on the smaller sphere

-- State the proportional relationship as a Lean theorem
theorem red_marked_area_on_larger_sphere : 
  let A2 := A1 * (r2^2 / r1^2)
  A2 = 83.25 :=
by
  sorry

end red_marked_area_on_larger_sphere_l481_481965


namespace smallest_b_for_factorization_l481_481684

theorem smallest_b_for_factorization :
  ∃ b : ℕ, (∀ p q : ℤ, p + q = b ∧ p * q = 2016 → p = 42 ∧ q = 48) ∧ b = 90 :=
begin
  sorry
end

end smallest_b_for_factorization_l481_481684


namespace conical_pile_volume_l481_481951

noncomputable def volume_of_cone (d : ℝ) (h : ℝ) : ℝ :=
  (Real.pi * (d / 2) ^ 2 * h) / 3

theorem conical_pile_volume :
  let diameter := 10
  let height := 0.60 * diameter
  volume_of_cone diameter height = 50 * Real.pi :=
by
  sorry

end conical_pile_volume_l481_481951


namespace game_necessarily_ends_winning_strategy_l481_481301

-- Definitions and conditions based on problem:
def Card := Fin 2009

def isWhite (c : Fin 2009) : Prop := sorry -- Placeholder for actual white card predicate

def validMove (k : Fin 2009) : Prop := k.val < 1969 ∧ isWhite k

def applyMove (k : Fin 2009) (cards : Fin 2009 → Prop) : Fin 2009 → Prop :=
  fun c => if c.val ≥ k.val ∧ c.val < k.val + 41 then ¬isWhite c else isWhite c

-- Theorem statements to match proof problem:
theorem game_necessarily_ends : ∃ n, n = 2009 → (∀ (cards : Fin 2009 → Prop), (∃ k < 1969, validMove k) → (∀ k < 1969, ¬(validMove k))) :=
sorry

theorem winning_strategy (cards : Fin 2009 → Prop) : ∃ strategy : (Fin 2009 → Prop) → Fin 2009, ∀ s, (s = applyMove (strategy s) s) → strategy s = sorry :=
sorry

end game_necessarily_ends_winning_strategy_l481_481301


namespace boa_constrictors_in_park_l481_481179

theorem boa_constrictors_in_park :
  ∃ (B : ℕ), (∃ (p : ℕ), p = 3 * B) ∧ (B + 3 * B + 40 = 200) ∧ B = 40 :=
by
  sorry

end boa_constrictors_in_park_l481_481179


namespace inequality_solution_l481_481524

theorem inequality_solution 
  (a x : ℝ) : 
  (a = 2 ∨ a = -2 → x > 1 / 4) ∧ 
  (a > 2 → x > 1 / (a + 2) ∨ x < 1 / (2 - a)) ∧ 
  (a < -2 → x < 1 / (a + 2) ∨ x > 1 / (2 - a)) ∧ 
  (-2 < a ∧ a < 2 → 1 / (a + 2) < x ∧ x < 1 / (2 - a)) 
  :=
by
  sorry

end inequality_solution_l481_481524


namespace sum_of_coordinates_l481_481915

-- Define the conditions and theorem
theorem sum_of_coordinates :
  let points := {p : ℝ × ℝ | abs (p.2 - 15) = 7 ∧ (p.1 - 5)^2 + (p.2 - 15)^2 = 225} in
  (∑ p in points, p.1 + p.2) = 85 :=
by
  sorry

end sum_of_coordinates_l481_481915


namespace megatek_employee_percentage_difference_l481_481887

theorem megatek_employee_percentage_difference :
  let total_angle := 360
  let manufacturing_angle := 144
  let rnd_angle := 108
  let marketing_angle := 108
  let total_percent := 100
  let perc (angle : ℕ) := angle * total_percent / total_angle
  let manuf_pct := perc manufacturing_angle
  let rnd_mark_combined := perc (rnd_angle + marketing_angle)
  abs (manuf_pct - rnd_mark_combined) = 20 := 
by
  sorry

end megatek_employee_percentage_difference_l481_481887


namespace number_of_elements_of_S_l481_481105

def is_1000_digit_odd_diff_2 (n : ℕ) : Prop :=
  (digits_count n = 1000) ∧
  (∀ d ∈ digits n, odd d) ∧
  (∀ i ∈ finset.range (1000 - 1), (digits n).nth i - (digits n).nth (i + 1) = 2 ∨ (digits n).nth i - (digits n).nth (i + 1) = -2)

def cardinality_S : ℕ :=
  8 * 3 ^ 499

theorem number_of_elements_of_S : 
  {n : ℕ // is_1000_digit_odd_diff_2 n}.card = cardinality_S :=
sorry

end number_of_elements_of_S_l481_481105


namespace find_b_value_l481_481714

theorem find_b_value {b : ℚ} (h : -8 ^ 2 + b * -8 - 45 = 0) : b = 19 / 8 :=
sorry

end find_b_value_l481_481714


namespace prob_B_draws_given_A_draws_black_fairness_l481_481562

noncomputable def event_A1 : Prop := true  -- A draws the red ball
noncomputable def event_A2 : Prop := true  -- B draws the red ball
noncomputable def event_A3 : Prop := true  -- C draws the red ball

noncomputable def prob_A1 : ℝ := 1 / 3
noncomputable def prob_not_A1 : ℝ := 2 / 3
noncomputable def prob_A2_given_not_A1 : ℝ := 1 / 2

theorem prob_B_draws_given_A_draws_black : (prob_not_A1 * prob_A2_given_not_A1) / prob_not_A1 = 1 / 2 := by
  sorry

theorem fairness :
  let prob_A1 := 1 / 3
  let prob_A2 := prob_not_A1 * prob_A2_given_not_A1
  let prob_A3 := prob_not_A1 * prob_A2_given_not_A1 * 1
  prob_A1 = prob_A2 ∧ prob_A2 = prob_A3 := by
  sorry

end prob_B_draws_given_A_draws_black_fairness_l481_481562


namespace stamp_collection_value_l481_481461

theorem stamp_collection_value :
  ∀ (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℕ) (equal_value : Prop),
  total_stamps = 20 →
  subset_stamps = 4 →
  subset_value = 16 →
  (∀ t s v, equal_value ↔ t = s * v) →
  (total_stamps = 20 ∧ subset_stamps = 4 ∧ subset_value = 16 ∧ equal_value) →
  total_stamps * (subset_value / subset_stamps) = 80 :=
by
  intros total_stamps subset_stamps subset_value equal_value h1 h2 h3 h4 conditions
  sorry

end stamp_collection_value_l481_481461


namespace eighteen_consecutive_not_all_good_l481_481659

def is_good (n : ℕ) : Prop :=
  has_divisors_two_primes n

theorem eighteen_consecutive_not_all_good :
  ¬ ∃ a : ℕ, ∀ i : ℕ, (i < 18) → is_good (a + i) := 
sorry

end eighteen_consecutive_not_all_good_l481_481659


namespace no_solution_for_k_l481_481020

theorem no_solution_for_k 
  (a1 a2 a3 a4 : ℝ) 
  (h_pos1 : 0 < a1) (h_pos2 : a1 < a2) 
  (h_pos3 : a2 < a3) (h_pos4 : a3 < a4) 
  (x1 x2 x3 x4 k : ℝ) 
  (h1 : x1 + x2 + x3 + x4 = 1) 
  (h2 : a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 = k) 
  (h3 : a1^2 * x1 + a2^2 * x2 + a3^2 * x3 + a4^2 * x4 = k^2) 
  (hx1 : 0 ≤ x1) (hx2 : 0 ≤ x2) (hx3 : 0 ≤ x3) (hx4 : 0 ≤ x4) :
  false := 
sorry

end no_solution_for_k_l481_481020


namespace students_own_all_pets_l481_481407

theorem students_own_all_pets (s d c b n a_d a_c a_b x y w z : ℕ)
    (hs : s = 48) (hd : d = s / 2) (hc : c = (5 * s) / 16) (hb : b = 8)
    (hn : n = 7) (ha_d : a_d = 12) (ha_c : a_c = 2) (ha_b : a_b = 4)
    (h_total : s - n = 41) :
    (x + y + z = 12) → (x + w + z = 13) → (y + w + z = 4) → (x + y + w + z = 23) →
    z = 1 :=
by
  intros hs hd hc hb hn ha_d ha_c ha_b h_total
  intros h1 h2 h3 h4
  sorry

end students_own_all_pets_l481_481407


namespace math_problem_l481_481418

open Nat

def arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, S n = (n * (2 + n))

def geometric_sequence (b : ℕ → ℕ) (q : ℕ) : Prop :=
  b 1 = 2 ∧ 0 < q ∧ ∀ n, b (n + 1) = b n * q

def conditions (b : ℕ → ℕ) (S : ℕ → ℕ) (q : ℕ) : Prop :=
  b 2 + S 2 = 16 ∧ 4 * S 2 = q * b 2

def sequence_c (c : ℕ → ℕ → ℚ) (S : ℕ → ℕ) : Prop :=
  ∀ n, c n = 1 / S n

def T_sum (T : ℕ → ℕ → ℚ) (c : ℕ → ℕ → ℚ) : Prop :=
  ∀ n, T n = (0 < n) → (S (λ k, c k)) n 1 / S n

theorem math_problem (a b : ℕ → ℕ) (S : ℕ → ℕ) (c T : ℕ → ℕ → ℚ) (q : ℕ) :
  arithmetic_sequence a S ∧
  geometric_sequence b q ∧
  conditions b S q ∧
  sequence_c c S ∧
  T_sum T c →
  (∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 2 ^ (2 * n - 1)) ∧
  (∀ n, T n = 3/4 - (2 * n + 3) / (2 * (n + 1) * (n + 2))) :=
by
  sorry

end math_problem_l481_481418


namespace option_D_is_odd_l481_481591

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem option_D_is_odd : 
  let x1 := 6^2,
      x2 := 23 - 17,
      x3 := 9 * 24,
      x4 := 9 * 41,
      x5 := 96 / 8
  in is_odd x4 :=
by 
  sorry

end option_D_is_odd_l481_481591


namespace discount_rate_on_pony_jeans_l481_481334

theorem discount_rate_on_pony_jeans (F P : ℝ) 
  (h1 : F + P = 25)
  (h2 : 45 * F + 36 * P = 900) :
  P = 25 :=
by
  sorry

end discount_rate_on_pony_jeans_l481_481334


namespace simplest_square_root_l481_481589

-- Define the problem options
def optionA : Real := sqrt 24
def optionB : Real := sqrt 0.5
def optionC : Real := sqrt (a^2 + 4)
def optionD : Real := sqrt (a / b)

-- Define what it means to be 'simplest'
-- For this example, we simplify the problem into checking that
-- no equivalent simpler radical form exists for optionC compared to others

theorem simplest_square_root (a b : Real) (hb : b ≠ 0) : 
  (optionC = sqrt (a^2 + 4)) ∧ 
  (optionA ≠ sqrt (a^2 + 4)) ∧ 
  (optionB ≠ sqrt (a^2 + 4)) ∧ 
  (optionD ≠ sqrt (a^2 + 4)) := 
by
  sorry

end simplest_square_root_l481_481589


namespace area_larger_sphere_red_is_83_point_25_l481_481970

-- Define the radii and known areas

def radius_smaller_sphere := 4 -- cm
def radius_larger_sphere := 6 -- cm
def area_smaller_sphere_red := 37 -- square cm

-- Prove the area of the region outlined in red on the larger sphere
theorem area_larger_sphere_red_is_83_point_25 :
  ∃ (area_larger_sphere_red : ℝ),
    area_larger_sphere_red = 83.25 ∧
    area_larger_sphere_red = area_smaller_sphere_red * (radius_larger_sphere ^ 2 / radius_smaller_sphere ^ 2) :=
by {
  sorry
}

end area_larger_sphere_red_is_83_point_25_l481_481970


namespace tony_ken_ratio_l481_481853

-- Definitions of conditions
def kenAmount : ℕ := 1750
def totalAmount : ℕ := 5250

-- Tony's amount
def tonysAmount : ℕ := totalAmount - kenAmount

-- Lean 4 statement to prove Tony's ratio to Ken's ratio is 2:1
theorem tony_ken_ratio : tonysAmount / kenAmount = 2 :=
by 
  simp [tonysAmount, kenAmount, totalAmount, Nat.div]
  sorry  -- Skipping the proof

end tony_ken_ratio_l481_481853


namespace average_temperature_dc_l481_481509

theorem average_temperature_dc :
  let t := [90, 90, 90, 79, 71] in
  (t.sum / (t.length : ℝ)) = 84 := by
  sorry

end average_temperature_dc_l481_481509


namespace subset_M_union_N_l481_481830

theorem subset_M_union_N (M N P : Set ℝ) (f g : ℝ → ℝ)
  (hM : M = {x | f x = 0} ∧ M ≠ ∅)
  (hN : N = {x | g x = 0} ∧ N ≠ ∅)
  (hP : P = {x | f x * g x = 0} ∧ P ≠ ∅) :
  P ⊆ (M ∪ N) := 
sorry

end subset_M_union_N_l481_481830


namespace distance_from_start_after_walking_l481_481624

theorem distance_from_start_after_walking :
  let side_length : ℝ := 4
  let hexagon_perimeter_distance : ℝ := 10
  let start_point : ℝ × ℝ := (0, 0)
  let position_after_4_km : ℝ × ℝ := (4, 0)
  let position_after_8_km : ℝ × ℝ := (4 - 2, 2 * Real.sqrt 3)
  let final_position : ℝ × ℝ := (2 + 2 * Real.sqrt 3, 2 * Real.sqrt 3 - 2)
  let distance_from_start : ℝ := Real.sqrt ((2 + 2 * Real.sqrt 3)^2 + (2 * Real.sqrt 3 - 2)^2)
  let expected_distance : ℝ := 4 * Real.sqrt 2
  in distance_from_start = expected_distance :=
by
  sorry

end distance_from_start_after_walking_l481_481624


namespace domain_of_F_l481_481895

variable {α : Type*} [LinearOrder α]

theorem domain_of_F 
  (f : α → α) 
  (a b : α) 
  (h1 : ∀ x, a ≤ x → x ≤ b → x ∈ set.univ) 
  (h2 : b > 0) 
  (h3 : -a > 0) 
  (h4 : -a < b) 
  : (∀ x, a ≤ x → x ≤ -a → F x = f x - f (-x)) := 
sorry

end domain_of_F_l481_481895


namespace three_collinear_points_same_color_l481_481262

theorem three_collinear_points_same_color (line_colored_in_two_colors : ℕ → bool) :
  ∃ A B C : ℕ, (line_colored_in_two_colors A = line_colored_in_two_colors B ∧
                line_colored_in_two_colors B = line_colored_in_two_colors C ∧
                2 * B = A + C) :=
sorry

end three_collinear_points_same_color_l481_481262


namespace values_of_B_l481_481001

theorem values_of_B (B : ℕ) : ∃ n, n = 2 ∧ (∀ B, B ∣ 72 ∧ (B6 : ℕ := B * 10 + 6) % 4 = 0 → B = 4 ∨ B = 8) :=
by
  -- The proof will be inserted here
  sorry

end values_of_B_l481_481001


namespace John_height_in_feet_after_growth_spurt_l481_481813

def John_initial_height : ℕ := 66
def growth_rate_per_month : ℕ := 2
def number_of_months : ℕ := 3
def inches_per_foot : ℕ := 12

theorem John_height_in_feet_after_growth_spurt :
  (John_initial_height + growth_rate_per_month * number_of_months) / inches_per_foot = 6 := by
  sorry

end John_height_in_feet_after_growth_spurt_l481_481813


namespace k_times_a_plus_b_l481_481628

/-- Given a quadrilateral with vertices P(ka, kb), Q(kb, ka), R(-ka, -kb), and S(-kb, -ka),
where a and b are consecutive integers with a > b > 0, and k is an odd integer.
It is given that the area of PQRS is 50.
Prove that k(a + b) = 5. -/
theorem k_times_a_plus_b (a b k : ℤ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a = b + 1)
  (h4 : Odd k)
  (h5 : 2 * k^2 * (a - b) * (a + b) = 50) :
  k * (a + b) = 5 := by
  sorry

end k_times_a_plus_b_l481_481628


namespace three_identical_numbers_l481_481836

theorem three_identical_numbers (n : ℕ) (x : Fin n → ℝ)
  (h₁ : n ≥ 3)
  (h₂ : ∀ (f : ℝ → ℝ), (∀ x, ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ f (x i) = f (x j) ∧ f (x j) = f (x k))) :
  ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ x i = x j ∧ x j = x k :=
sorry

end three_identical_numbers_l481_481836


namespace line_plane_parallelism_l481_481718

-- Definitions of the lines and plane
variable (m n : Type) [line m] [line n] (α : Type) [plane α]

-- Conditions
variable (p1 : is_parallel m n) (p2 : is_parallel m α) (p3 : is_parallel n α)

-- The propositions to be proven
theorem line_plane_parallelism :
  (p1 ∧ p2 → p3) ∨ (p1 ∧ p3 → p2) :=
sorry

end line_plane_parallelism_l481_481718


namespace tetrahedron_volume_l481_481154

theorem tetrahedron_volume (R S1 S2 S3 S4 : ℝ) : 
    V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume_l481_481154


namespace bethany_total_hours_l481_481994

-- Define the hours Bethany rode on each set of days
def hours_mon_wed_fri : ℕ := 3  -- 1 hour each on Monday, Wednesday, and Friday
def hours_tue_thu : ℕ := 1  -- 30 min each on Tuesday and Thursday
def hours_sat : ℕ := 2  -- 2 hours on Saturday

-- Define the total hours per week
def total_hours_per_week : ℕ := hours_mon_wed_fri + hours_tue_thu + hours_sat

-- Define the total hours in 2 weeks
def total_hours_in_2_weeks : ℕ := total_hours_per_week * 2

-- Prove that the total hours in 2 weeks is 12
theorem bethany_total_hours : total_hours_in_2_weeks = 12 :=
by
  -- Replace the definitions with their values and check the equality
  rw [total_hours_in_2_weeks, total_hours_per_week, hours_mon_wed_fri, hours_tue_thu, hours_sat]
  simp
  norm_num
  sorry

end bethany_total_hours_l481_481994


namespace express_recurring_decimal_as_fraction_l481_481676

theorem express_recurring_decimal_as_fraction (h : 0.01 = (1 : ℚ) / 99) : 2.02 = (200 : ℚ) / 99 :=
by 
  sorry

end express_recurring_decimal_as_fraction_l481_481676


namespace area_equality_l481_481081

variables (A B C L K M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace L] [MetricSpace K] [MetricSpace M] [MetricSpace N]
variables (f : Triangle A B C) (g : Line A L) (h : Line L K) (i : Line L M)
variables (j : AngleBisector A B C) (k : PerpendicularDrop L A B) (l : PerpendicularDrop L A C) (m : Circumcircle A B C N)
variables (n : AcuteTriangle A B C)

def area_triang (T : Triangle A B C) : ℝ := sorry  -- Placeholder

def area_quad (Q : Quadrilateral A K N M) : ℝ := sorry  -- Placeholder

theorem area_equality : area_triang ⟨A, B, C⟩ = area_quad ⟨A, K, N, M⟩ := sorry

end area_equality_l481_481081


namespace concur_circumcircles_l481_481109

variables {A B C D E F S T : Type*}

-- Quadrilateral and segments definitions
variables (quadrilateral : quadrilateral A B C D)
variables (pointEonAD : E ∈ [A, D])
variables (pointFonBC : F ∈ [B, C])
variables (ratio_condition : AE / ED = BF / FC)

-- Definitions of intersections
variables (S_is_intersection : S = (EF ∩ AB))
variables (T_is_intersection : T = (EF ∩ CD))

-- The theorem statement
theorem concur_circumcircles 
    (h1 : pointEonAD) 
    (h2 : pointFonBC) 
    (h3 : ratio_condition) 
    (h4 : S_is_intersection) 
    (h5 : T_is_intersection) 
    : concurrent (circumcircle S A E) (circumcircle S B F) (circumcircle T C F) (circumcircle T D E) := sorry

end concur_circumcircles_l481_481109


namespace sum_of_numbers_of_large_cube_l481_481943

def sum_faces_of_die := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice := 125

def number_of_faces_per_die := 6

def total_exposed_faces (side_length: ℕ) : ℕ := 6 * (side_length * side_length)

theorem sum_of_numbers_of_large_cube (side_length : ℕ) (dice_count : ℕ) 
    (sum_per_die : ℕ) (opposite_face_sum : ℕ) :
    dice_count = 125 →
    total_exposed_faces side_length = 150 →
    sum_per_die = 21 →
    (∀ f1 f2, (f1 + f2 = opposite_face_sum)) →
    dice_count * sum_per_die = 2625 →
    (210 ≤ dice_count * sum_per_die ∧ dice_count * sum_per_die ≤ 840) :=
by 
  intro h_dice_count
  intro h_exposed_faces
  intro h_sum_per_die
  intro h_opposite_faces
  intro h_total_sum
  sorry

end sum_of_numbers_of_large_cube_l481_481943


namespace minimum_value_expr_l481_481508

theorem minimum_value_expr :
  ∃ (x y : ℝ) , (xy : ℝ) ^ 2 + (x + 7) ^ 2 + (2 * y + 7) ^ 2 = 45 :=
begin
  sorry
end

end minimum_value_expr_l481_481508


namespace find_a_l481_481847

noncomputable def consecutive_sum (k : ℕ) : ℕ :=
  (k + (k + 1) + ... + (k + 24))

noncomputable def distance_sum (a k : ℕ) : ℕ :=
  abs (25 * a - consecutive_sum k)

noncomputable def distance_sum_square (a k : ℕ) : ℕ :=
  abs (25 * a * a - consecutive_sum k)

theorem find_a (a : ℚ) (k : ℕ) (h1 : distance_sum a k = 1270) (h2 : distance_sum_square a k = 1234) :
  a = -4 / 5 := sorry

end find_a_l481_481847


namespace triangle_perimeter_of_folded_square_l481_481271

-- Definitions for the conditions
structure Square (α : Type) :=
  (A B C D : α × α)
  (side_length : α)
  (is_square : (B.1 - A.1) = side_length ∧ (B.2 - A.2) = 0 ∧
               (C.1 - B.1) = side_length ∧ (C.2 - B.2) = 0 ∧
               (D.1 - C.1) = side_length ∧ (D.2 - C.2) = 0 ∧
               (A.1 - D.1) = side_length ∧ (A.2 - D.2) = 0)

def point_on_line (p l1 l2 : ℚ × ℚ) : Prop :=
  l1.1 ≤ p.1 ∧ p.1 ≤ l2.1 ∧ l1.2 ≤ p.2 ∧ p.2 <= l2.2

noncomputable def is_perimeter_of_triangle (α : Type) (triangle_points : α × α × α) (perimeter : α) : Prop :=
  let (A, E, C') := triangle_points in
  let AE := (A.1 - E.1)^2 + (A.2 - E.2)^2;
  let EC' := (E.1 - C'.1)^2 + (E.2 - C'.2)^2;
  let AC' := (A.1 - C'.1)^2 + (A.2 - C'.2)^2;
  perimeter = AE^(1/2) + EC'^(1/2) + AC'^(1/2)

-- Problem Statement in Lean 4
theorem triangle_perimeter_of_folded_square :
  ∃ (A E C' : ℚ × ℚ), let s : Square ℚ := {A := (0, 2), B := (0, 0), C := (2, 0), D := (2, 2), 
                                             side_length := 2, 
                                             is_square := by sorry};
    let C' := (2, (4 / 3)) in
    point_on_line C' s.A s.D →
    let E := (3 / 2, 3 / 2) in
    let perimeter := (4 * real.sqrt 10) / 3 in
    is_perimeter_of_triangle ℚ (s.A, E, C') perimeter :=
sorry

end triangle_perimeter_of_folded_square_l481_481271


namespace b_plus_d_over_a_l481_481188

theorem b_plus_d_over_a (a b c d e : ℝ) (h : a ≠ 0) 
  (root1 : a * (5:ℝ)^4 + b * (5:ℝ)^3 + c * (5:ℝ)^2 + d * (5:ℝ) + e = 0)
  (root2 : a * (-3:ℝ)^4 + b * (-3:ℝ)^3 + c * (-3:ℝ)^2 + d * (-3:ℝ) + e = 0)
  (root3 : a * (2:ℝ)^4 + b * (2:ℝ)^3 + c * (2:ℝ)^2 + d * (2:ℝ) + e = 0) :
  (b + d) / a = - (12496 / 3173) :=
sorry

end b_plus_d_over_a_l481_481188


namespace rational_seq_integer_l481_481629

theorem rational_seq_integer (x : ℚ) : ∃ (x_n : ℕ → ℚ), x_n 0 = x ∧ 
  (∀ n ≥ 1, (x_n (n + 1) = 2 * x_n n ∨ x_n (n + 1) = 2 * x_n n + 1 / (n + 1))) ∧ 
  ∃ n, x_n n ∈ ℤ :=
by
  sorry

end rational_seq_integer_l481_481629


namespace problem_a_problem_b_problem_c_l481_481108

noncomputable def inequality_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (0 * y + 1)) + 1 / (y * (0 * z + 1)) + 1 / (z * (0 * x + 1))) ≥ 3

noncomputable def inequality_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (1 * y + 0)) + 1 / (y * (1 * z + 0)) + 1 / (z * (1 * x + 0))) ≥ 3

noncomputable def inequality_c (x y z : ℝ) (a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : Prop :=
  (1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b))) ≥ 3

theorem problem_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_a x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_b x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_c (x y z a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : inequality_c x y z a b h1 h2 h3 h4 h5 h6 h7 :=
  by sorry

end problem_a_problem_b_problem_c_l481_481108


namespace find_f_x_max_min_f_interval_l481_481717

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

-- Given condition
axiom f_condition : ∀ x : ℝ, f ((x - 1) / (x + 1)) = -x - 1

-- Problem 1: f(x)
theorem find_f_x (x : ℝ) (h : x ≠ 1) : f x = 2 / (x - 1) :=
by 
  rw [f]
  trivial

-- Problem 2: Monotonicity and extremum values in [2, 6]
theorem max_min_f_interval : 
  (∀ x ∈ set.Icc 2 6, ∃! y ∈ set.Icc 2 6, y = x) ∧ 
  (∀ (x1 x2 : ℝ) (h1 : x1 ∈ set.Icc 2 6) (h2 : x2 ∈ set.Icc 2 6), x1 < x2 → f x1 > f x2) ∧ 
  (∀ x ∈ set.Icc 2 6, f 2 = 2) ∧ 
  (∀ x ∈ set.Icc 2 6, f 6 = 2 / 5) := 
sorry

end find_f_x_max_min_f_interval_l481_481717


namespace jim_added_amount_l481_481463

theorem jim_added_amount {X : ℝ} :
  let initial_amount := 80
  let growth_rate1 := 0.15
  let growth_rate2 := 0.10
  let final_amount := 132 
  after_first_year := initial_amount * (1 + growth_rate1)
  after_second_year := (after_first_year + X) * (1 + growth_rate2)
  after_second_year = final_amount -> 
  X = 28 :=
by
  sorry

end jim_added_amount_l481_481463


namespace smallest_prime_p_l481_481326

noncomputable def N : ℕ := nat.factorial 3 * nat.factorial 11 * nat.factorial 61

def gcd(N : ℕ, p : ℕ) : ℕ := Nat.gcd N p

theorem smallest_prime_p (N : ℕ) : ∃ p, Prime p ∧ p > 61 ∧ gcd N p = 1 ∧ (∀ q, Prime q ∧ q > 61 ∧ gcd N q = 1 → p ≤ q) :=
begin
  let p := 67,
  use p,
  split,
  { exact prime_of_prime_67, },
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  { intros q Hprime Hgt61 Hgcd,
    exact dec_trivial, },
end

end smallest_prime_p_l481_481326


namespace right_triangle_area_and_perimeter_l481_481544

theorem right_triangle_area_and_perimeter (a b c : ℝ) (h : c = 13) (ha : a = 5) (hc : c^2 = a^2 + b^2) :
  (∃ b : ℝ, b = sqrt(13^2 - 5^2) ∧ (1 / 2) * a * b = 30 ∧ a + b + c = 30) :=
by 
  use sqrt (13^2 - 5^2)
  have hb : b = sqrt (13^2 - 5^2), from sorry,
  have area : (1 / 2) * 5 * sqrt (13^2 - 5^2) = 30, from sorry,
  have perimeter : 5 + sqrt (13^2 - 5^2) + 13 = 30, from sorry,
  exact ⟨hb, area, perimeter⟩
  sorry

end right_triangle_area_and_perimeter_l481_481544


namespace area_of_triangle_ADC_l481_481429

theorem area_of_triangle_ADC (ABC : Triangle) (A B C D : Point)
  (h_right_angle : ∠ ABC = 90)
  (h_angle_bisector : IsAngleBisector AD)
  (h_AB : AB = 50)
  (h_BC_y : BC = y)
  (h_AC : AC = 3 * y - 10) :
  area ADC = 412 :=
by
  sorry

end area_of_triangle_ADC_l481_481429


namespace trader_sold_meters_l481_481638

theorem trader_sold_meters (total_selling_price : ℝ)
                           (profit_per_meter : ℝ)
                           (cost_price_per_meter : ℝ)
                           (selling_price_per_meter : ℝ := 
                              cost_price_per_meter + profit_per_meter)
                           (meters_sold_approx : ℕ) : 
  meters_sold_approx ≈ Nat.floor (total_selling_price / selling_price_per_meter) :=
by
  let total_selling_price := 6788
  let profit_per_meter := 29
  let cost_price_per_meter := 58.02564102564102
  let meters_sold_approx := 78
  let selling_price_per_meter := cost_price_per_meter + profit_per_meter
  have calculation := total_selling_price / selling_price_per_meter
  have approx_eq := (Nat.floor calculation : ℕ)
  have result := meters_sold_approx = approx_eq
  sorry -- skipping the proof

end trader_sold_meters_l481_481638


namespace find_b_l481_481666

def h (x : ℝ) : ℝ := 5 * x + 6

theorem find_b : ∃ b : ℝ, h b = 0 ∧ b = -6 / 5 :=
by
  sorry

end find_b_l481_481666


namespace purely_imaginary_m_value_bisector_m_values_l481_481730

def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

def is_on_bisector (z : ℂ) : Prop :=
  z.im = -z.re

noncomputable def z (m : ℝ) : ℂ :=
  (2 + Complex.i) * m^2 - 6 * m / (1 - Complex.i) - 2 * (1 - Complex.i)

theorem purely_imaginary_m_value :
  is_purely_imaginary (z (-1/2)) :=
sorry

theorem bisector_m_values :
  (is_on_bisector (z 0)) ∧ (is_on_bisector (z 2)) :=
sorry

end purely_imaginary_m_value_bisector_m_values_l481_481730


namespace first_interest_rate_l481_481517

theorem first_interest_rate (r : ℝ) : 
  (70000:ℝ) = (60000:ℝ) + (10000:ℝ) →
  (8000:ℝ) = (60000 * r / 100) + (10000 * 20 / 100) →
  r = 10 :=
by
  intros h1 h2
  sorry

end first_interest_rate_l481_481517


namespace g_at_1_eq_binom_coeff_l481_481238

def g (k l : ℕ) (x : ℝ) : ℝ := (A : ℝ → ℝ) x / (1 - x) ^ k

theorem g_at_1_eq_binom_coeff (k l : ℕ) (A : ℝ → ℝ) :
    g k l (1 : ℝ) = Nat.choose (k + l) k :=
    sorry

end g_at_1_eq_binom_coeff_l481_481238


namespace imaginary_number_m_l481_481764

theorem imaginary_number_m (m : ℝ) : 
  (∀ Z, Z = (m + 2 * Complex.I) / (1 + Complex.I) → Z.im = 0 → Z.re = 0) → m = -2 :=
by
  sorry

end imaginary_number_m_l481_481764


namespace k_value_range_l481_481734

noncomputable def f (x : ℝ) : ℝ := x - 1 - Real.log x

theorem k_value_range {k : ℝ} (h : ∀ x : ℝ, 0 < x → f x ≥ k * x - 2) : 
  k ≤ 1 - 1 / Real.exp 2 := 
sorry

end k_value_range_l481_481734


namespace number_of_solutions_x2_plus_y2_l481_481697

theorem number_of_solutions_x2_plus_y2 (p : ℕ) [fact (nat.prime p)] (hp : p % 2 = 1) (a : ℤ) :
  (finset.card {pair : fin (p) × fin (p) // (pair.1 : ℤ)^2 + (pair.2 : ℤ)^2 % p = a % p}.1) = p + 1 :=
sorry

end number_of_solutions_x2_plus_y2_l481_481697


namespace circles_internally_tangent_l481_481660

def positional_relationship (C1 C2 : Set (ℝ × ℝ)) : Prop :=
  -- Positional relationship for circles C1 and C2
  ∃ (x1 y1 r1 x2 y2 r2 : ℝ),
    C1 = {p | (p.1 + x1)^2 + (p.2 + y1)^2 = r1^2} ∧
    C2 = {p | (p.1 + x2)^2 + (p.2 + y2)^2 = r2^2} ∧
    sqrt ((x2 - x1)^2 + (y2 - y1)^2) = abs (r1 - r2)

theorem circles_internally_tangent :
  positional_relationship {p : ℝ × ℝ | p.1^2 + p.2^2 + 4 * p.1 + 8 * p.2 - 5 = 0}
                          {p : ℝ × ℝ | p.1^2 + p.2^2 + 4 * p.1 + 4 * p.2 - 1 = 0} :=
by
  sorry

end circles_internally_tangent_l481_481660


namespace nth_equation_l481_481856

theorem nth_equation (n : ℕ) : 
  n ≥ 1 → (∃ k, k = n + 1 ∧ (k^2 - n^2 - 1) / 2 = n) :=
by
  intros h
  use n + 1
  sorry

end nth_equation_l481_481856


namespace find_length_AB_l481_481794

variables {A B C D E : Type} -- Define variables A, B, C, D, E as types, representing points

-- Define lengths of the segments AD and CD
def length_AD : ℝ := 2
def length_CD : ℝ := 2

-- Define the angles at vertices B, C, and D
def angle_B : ℝ := 30
def angle_C : ℝ := 90
def angle_D : ℝ := 120

-- The goal is to prove the length of segment AB
theorem find_length_AB : 
  (∃ (A B C D : Type) 
    (angle_B angle_C angle_D length_AD length_CD : ℝ), 
      angle_B = 30 ∧ 
      angle_C = 90 ∧ 
      angle_D = 120 ∧ 
      length_AD = 2 ∧ 
      length_CD = 2) → 
  (length_AB = 6) := by sorry

end find_length_AB_l481_481794


namespace complex_point_location_l481_481546

theorem complex_point_location :
  let z := (5 - 6 * Complex.I) + (-2 - Complex.I) - (3 + 4 * Complex.I)
  in z = -11 * Complex.I ∧ 
     (z.re = 0 ∧ z.im < 0) :=
by
  let z := (5 - 6 * Complex.I) + (-2 - Complex.I) - (3 + 4 * Complex.I)
  have h1 : z = -11 * Complex.I, by sorry
  have h2 : z.re = 0, by sorry
  have h3 : z.im < 0, by sorry
  exact ⟨h1, ⟨h2, h3⟩⟩

end complex_point_location_l481_481546


namespace perpendicular_bisector_eq_triangle_area_l481_481367

-- Definitions for the vertices of triangle ABC
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Problem 1: Equation of the perpendicular bisector of side BC
theorem perpendicular_bisector_eq : 
  ∃ g : ℝ × ℝ → Prop, 
  (g (1,2) ∧ (∀ p, g p → (p.2 - 2) = -2 * (p.1 - 1))) :=
  sorry

-- Problem 2: Area of triangle ABC
theorem triangle_area :
  let ab := sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)
  let bc := sqrt ((C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2)
  let ca := sqrt ((C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2)
  ∃ (S : ℝ), S = 7 ∧ S = 1/2 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)| :=
  sorry

end perpendicular_bisector_eq_triangle_area_l481_481367


namespace stickers_at_end_of_week_l481_481133

theorem stickers_at_end_of_week (initial_stickers earned_stickers total_stickers : Nat) :
  initial_stickers = 39 →
  earned_stickers = 22 →
  total_stickers = initial_stickers + earned_stickers →
  total_stickers = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end stickers_at_end_of_week_l481_481133


namespace total_amount_received_l481_481946

-- Definitions based on conditions
def days_A : Nat := 6
def days_B : Nat := 8
def days_ABC : Nat := 3

def share_A : Nat := 300
def share_B : Nat := 225
def share_C : Nat := 75

-- The theorem stating the total amount received for the work
theorem total_amount_received (dA dB dABC : Nat) (sA sB sC : Nat)
  (h1 : dA = days_A) (h2 : dB = days_B) (h3 : dABC = days_ABC)
  (h4 : sA = share_A) (h5 : sB = share_B) (h6 : sC = share_C) : 
  sA + sB + sC = 600 := by
  sorry

end total_amount_received_l481_481946


namespace multiply_scaled_values_l481_481236

theorem multiply_scaled_values (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by 
  sorry

end multiply_scaled_values_l481_481236


namespace purely_imaginary_complex_is_two_l481_481762

theorem purely_imaginary_complex_is_two
  (a : ℝ)
  (h_imag : (a^2 - 3 * a + 2) + (a - 1) * I = (a - 1) * I) :
  a = 2 := by
  sorry

end purely_imaginary_complex_is_two_l481_481762


namespace x_squared_y_minus_xy_squared_l481_481008

theorem x_squared_y_minus_xy_squared (x y : ℝ) (h1 : x - y = -2) (h2 : x * y = 3) : x^2 * y - x * y^2 = -6 := 
by 
  sorry

end x_squared_y_minus_xy_squared_l481_481008


namespace fourDigitNumbersCount_l481_481049

-- Define the conditions for the problem
def isFourDigitNumber (n : Nat) : Prop :=
  n >= 1000 ∧ n < 10000

def leadingDigitGreaterThanFive (n : Nat) : Prop :=
  n / 1000 >= 5

def validDigits (d : Nat) : Prop :=
  d >= 0 ∧ d <= 9

def onlyFourCanBeRepeated (digits : List Nat) : Prop :=
  ∀ d ∈ digits, (d = 4 ∨ count digits d = 1)

-- Problem statement
theorem fourDigitNumbersCount :
  ∃ count, count = 2645 ∧
  (count = 
    ∑ a in {n | isFourDigitNumber n ∧ leadingDigitGreaterThanFive n}.toFinset, 
    ∑ b in {d | validDigits d}.toFinset,
    ∑ c in {d | validDigits d}.toFinset,
    ∑ d in {d | validDigits d}.toFinset, 
    if (onlyFourCanBeRepeated [b, c, d]) then 1 else 0) :=
sorry

end fourDigitNumbersCount_l481_481049


namespace distance_from_point_to_line_in_polar_l481_481797

theorem distance_from_point_to_line_in_polar :
  let ρ := 1
  let θ := 0
  let distance (ρ θ : ℝ) : ℝ := 
    | ρ - 2 / (Real.cos θ + Real.sin θ) |
  in distance ρ θ = 1 :=
by
  sorry

end distance_from_point_to_line_in_polar_l481_481797


namespace train_speeds_l481_481912

-- Define the given conditions
variables (x : ℝ) (express_speed : ℝ)

-- Speed conditions
def bullet_train_speed := 2 * x
def high_speed_train_speed := 1.25 * bullet_train_speed

-- Given condition constraints
def average_high_and_regular := (x + high_speed_train_speed x) / 2 = express_speed + 15
def average_bullet_and_regular := (x + bullet_train_speed x) / 2 = express_speed - 10

-- Conclusion: Speed of regular train and high-speed train
theorem train_speeds (h1 : average_high_and_regular x express_speed)
                     (h2 : average_bullet_and_regular x express_speed) :
  x = 100 ∧ high_speed_train_speed x = 250 :=
by
  sorry

end train_speeds_l481_481912


namespace lines_parallel_l481_481905

def line1 (x y : ℝ) : Prop := x - y + 2 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem lines_parallel : 
  (∀ x y, line1 x y ↔ y = x + 2) ∧ 
  (∀ x y, line2 x y ↔ y = x + 1) ∧ 
  ∃ m₁ m₂ c₁ c₂, (∀ x y, (y = m₁ * x + c₁) ↔ line1 x y) ∧ (∀ x y, (y = m₂ * x + c₂) ↔ line2 x y) ∧ m₁ = m₂ ∧ c₁ ≠ c₂ :=
by
  sorry

end lines_parallel_l481_481905


namespace greatest_visible_unit_cubes_from_single_point_l481_481984

-- Define the size of the cube
def cube_size : ℕ := 9

-- The total number of unit cubes in the 9x9x9 cube
def total_unit_cubes (n : ℕ) : ℕ := n^3

-- The greatest number of unit cubes visible from a single point
def visible_unit_cubes (n : ℕ) : ℕ := 3 * n^2 - 3 * (n - 1) + 1

-- The given cube size is 9
def given_cube_size : ℕ := cube_size

-- The correct answer for the greatest number of visible unit cubes from a single point
def correct_visible_cubes : ℕ := 220

-- Theorem stating the visibility calculation for a 9x9x9 cube
theorem greatest_visible_unit_cubes_from_single_point :
  visible_unit_cubes cube_size = correct_visible_cubes := by
  sorry

end greatest_visible_unit_cubes_from_single_point_l481_481984


namespace triangle_cosine_rule_example_l481_481426

theorem triangle_cosine_rule_example (a b C : ℝ) (h_a : a = 1) (h_b : b = 2) (h_C : C = real.pi / 3) :
  ∃ (c : ℝ), c = real.sqrt 3 := 
by
  use real.sqrt 3
  sorry

end triangle_cosine_rule_example_l481_481426


namespace evaluate_expression_l481_481473

def M (x y : ℝ) := max x y
def m (x y : ℝ) := min x y

variables (a b c d e : ℝ)
variables (h1 : e < d) (h2 : d < c) (h3 : c < b) (h4 : b < a)
variables (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)

theorem evaluate_expression : M (M e (m d b)) (m c (m a e)) = d := by
  sorry

end evaluate_expression_l481_481473


namespace intersection_points_l481_481893

def curve1 (x : ℝ) : ℝ := sin x

def curve2 (x y r : ℝ) [hr : fact (0 < r)] : Prop := (x^2 + (y + r - 1/2)^2 = r^2)

theorem intersection_points (r : ℝ) [hr : fact (0 < r)] :
  ∃ (x y : ℝ), x^2 + (y + r - 1/2)^2 = r^2 ∧ y = sin x :=
sorry

end intersection_points_l481_481893


namespace pens_per_student_l481_481561

theorem pens_per_student (n : ℕ) (h1 : 0 < n) (h2 : n ≤ 50) (h3 : 100 % n = 0) (h4 : 50 % n = 0) : 100 / n = 2 :=
by
  -- proof goes here
  sorry

end pens_per_student_l481_481561


namespace cos_angle_between_a_b_l481_481023

noncomputable def cos_angle_between_vectors
  (e1 e2 : ℝ × ℝ × ℝ)  -- unit vectors in ℝ³
  (unit_e1 : ∥e1∥ = 1)
  (unit_e2 : ∥e2∥ = 1)
  (angle_60 : inner e1 e2 = 1 / 2)  -- cos 60° = 1/2
  : ℝ :=
let a := (2 : ℝ) • e1 + e2 in
let b := -(3 : ℝ) • e1 + (2 : ℝ) • e2 in
inner a b / (∥a∥ * ∥b∥)

theorem cos_angle_between_a_b
  (e1 e2 : ℝ × ℝ × ℝ)
  (unit_e1 : ∥e1∥ = 1)
  (unit_e2 : ∥e2∥ = 1)
  (angle_60 : inner e1 e2 = 1 / 2) :
  cos_angle_between_vectors e1 e2 unit_e1 unit_e2 angle_60 = -1 / 2 := sorry

end cos_angle_between_a_b_l481_481023


namespace find_a2_l481_481343

variable (a : ℕ → ℝ) (d : ℝ)

axiom arithmetic_seq (n : ℕ) : a (n + 1) = a n + d
axiom common_diff : d = 2
axiom geometric_mean : (a 4) ^ 2 = (a 5) * (a 2)

theorem find_a2 : a 2 = -8 := 
by 
  sorry

end find_a2_l481_481343


namespace line_AB_eq_and_AB_length_l481_481351

theorem line_AB_eq_and_AB_length :
  (∃ A B : ℝ × ℝ, 
    (∃ k, A = (k, 2*k)) ∧ 
    (∃ m, B = (m, -m / 2)) ∧
    let P := (0, 10 / 2 : ℝ) in
    A.1 + B.1 = 2 * P.1 ∧ (A.2 + B.2) / 2 = P.2) →
  (∃ l : ℝ, l = (3:ℤ)*x - (4:ℤ)*y + 20) ∧
  (∃ length_AB : ℝ, length_AB = 10) :=
begin
  sorry
end

end line_AB_eq_and_AB_length_l481_481351


namespace number_of_even_four_digit_numbers_l481_481421

noncomputable def count_even_four_digit_numbers_no_repetition : ℕ :=
  let digits := {0, 1, 2, 3, 4, 5, 6, 7}
  let units_digit_zero := 7 * 6 * 5
  let units_digit_246_no_zero := 3 * (6 * 5 * 4)
  let units_digit_246_with_zero := 3 * 3 * (5 * 4)
  in units_digit_zero + units_digit_246_no_zero + units_digit_246_with_zero

theorem number_of_even_four_digit_numbers :
  count_even_four_digit_numbers_no_repetition = 750 :=
by
  sorry

end number_of_even_four_digit_numbers_l481_481421


namespace problem_statement_l481_481930

def f (x : ℝ) : ℝ := (√3) * sin x * cos x + (cos x)^2 - (1 / 2)
def point_is_not_center_of_symmetry (x0 y0 : ℝ) (f : ℝ → ℝ) : Prop :=
  ¬ (∀ x : ℝ, f (2 * x0 - x) = 2 * y0 - f x)

theorem problem_statement : point_is_not_center_of_symmetry (π / 6) 0 f :=
by sorry

end problem_statement_l481_481930


namespace solution_correctness_l481_481174

noncomputable def solution_set : Set ℝ := { x : ℝ | (x + 1) * (x - 2) > 0 }

theorem solution_correctness (x : ℝ) :
  (x ∈ solution_set) ↔ (x < -1 ∨ x > 2) :=
by sorry

end solution_correctness_l481_481174


namespace tetrahedron_vertex_triangle_l481_481863

-- Define vertices and edges of the tetrahedron
variables {A B C D : Type}

-- Define lengths of the edges
variables (AB AC AD BC BD CD : ℝ)

-- Assume AB is the longest edge
hypothesis (h1 : AB ≥ AC ∧ AB ≥ AD ∧ AB ≥ BC ∧ AB ≥ BD ∧ AB ≥ CD)

-- Define the condition for forming a triangle
def triangle_condition (x y z : ℝ) : Prop := x < y + z

-- The main theorem statement
theorem tetrahedron_vertex_triangle :
  ∃ v, v ∈ {A, B, C, D} ∧
    (∃ x y z : ℝ, {x, y, z} = {AB, AC, AD} ∧ triangle_condition x y z) ∨
    (∃ x y z : ℝ, {x, y, z} = {AB, BC, BD} ∧ triangle_condition x y z) ∨
    (∃ x y z : ℝ, {x, y, z} = {AC, BC, CD} ∧ triangle_condition x y z) ∨
    (∃ x y z : ℝ, {x, y, z} = {AD, BD, CD} ∧ triangle_condition x y z) :=
sorry

end tetrahedron_vertex_triangle_l481_481863


namespace circle_area_radius_8_l481_481201

variable (r : ℝ) (π : ℝ)

theorem circle_area_radius_8 : r = 8 → (π * r^2) = 64 * π :=
by
  sorry

end circle_area_radius_8_l481_481201


namespace total_marbles_l481_481057

theorem total_marbles
  (R B Y : ℕ)  -- Red, Blue, and Yellow marbles as natural numbers
  (h_ratio : 2 * (R + B + Y) = 9 * Y)  -- The ratio condition translated
  (h_yellow : Y = 36)  -- The number of yellow marbles condition
  : R + B + Y = 81 :=  -- Statement that the total number of marbles is 81
sorry

end total_marbles_l481_481057


namespace sum_of_digits_of_B_l481_481838

theorem sum_of_digits_of_B : 
  let A := sum_of_digits (4444 ^ 4144)
  let B := sum_of_digits A
  B = 7 :=
by
  sorry

noncomputable def sum_of_digits (n: ℤ) : ℤ := 
  -- Definition to compute the sum of digits of a number
  sorry

end sum_of_digits_of_B_l481_481838


namespace mark_fewer_than_susan_l481_481745

variable (apples_total : ℕ) (greg_apples : ℕ) (susan_apples : ℕ) (mark_apples : ℕ) (mom_apples : ℕ)

def evenly_split (total : ℕ) : ℕ := total / 2

theorem mark_fewer_than_susan
    (h1 : apples_total = 18)
    (h2 : greg_apples = evenly_split apples_total)
    (h3 : susan_apples = 2 * greg_apples)
    (h4 : mom_apples = 40 + 9)
    (h5 : mark_apples = mom_apples - susan_apples) :
    susan_apples - mark_apples = 13 := 
sorry

end mark_fewer_than_susan_l481_481745


namespace domain_of_f2x_l481_481722

noncomputable def f : ℝ → ℝ := sorry

theorem domain_of_f2x (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, y = f (x + 1)) :
  set.Icc 0 2 ⊆ { x : ℝ | ∃ y : ℝ, y = f (2^x) } :=
by
  sorry

end domain_of_f2x_l481_481722


namespace simplify_and_evaluate_div_expr_l481_481150

variable (m : ℤ)

theorem simplify_and_evaluate_div_expr (h : m = 2) :
  ( (m^2 - 9) / (m^2 - 6 * m + 9) / (1 - 2 / (m - 3)) = -5 / 3) :=
by
  sorry

end simplify_and_evaluate_div_expr_l481_481150


namespace ratio_of_cone_to_sphere_l481_481683

theorem ratio_of_cone_to_sphere (r : ℝ) (h := 2 * r) : 
  (1 / 3 * π * r^2 * h) / ((4 / 3) * π * r^3) = 1 / 2 :=
by 
  sorry

end ratio_of_cone_to_sphere_l481_481683


namespace problem_statement_l481_481106

theorem problem_statement (m n : ℕ) (h : ∃ᶠ k in filter.at_top, ∃ t : ℤ, k^2 + 2*k*n + m^2 = t^2) : m = n :=
sorry

end problem_statement_l481_481106


namespace surface_area_of_solid_of_revolution_l481_481542

theorem surface_area_of_solid_of_revolution 
  (a α : ℝ) : 
  0 < a ∧ 0 < α ∧ α < π / 2 →
  let S := 8 * real.pi * a^2 * real.sin α * real.cos (α / 2) * real.cos ((real.pi / 6) + (α / 2)) * real.cos ((real.pi / 6) - (α / 2)) in
  S = 8 * real.pi * a^2 * real.sin α * real.cos (α / 2) * real.cos ((real.pi / 6) + (α / 2)) * real.cos ((real.pi / 6) - (α / 2)) :=
by
  intros h
  sorry

end surface_area_of_solid_of_revolution_l481_481542


namespace find_S_11_l481_481086

variables (a : ℕ → ℤ)
variables (d : ℤ) (n : ℕ)

def is_arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

noncomputable def a_3 := a 3
noncomputable def a_6 := a 6
noncomputable def a_9 := a 9

theorem find_S_11
  (h1 : is_arithmetic_sequence a d)
  (h2 : a_3 + a_9 = 18 - a_6) :
  sum_first_n_terms a 11 = 66 :=
sorry

end find_S_11_l481_481086


namespace comp_figure_perimeter_l481_481153

-- Given conditions
def side_length_square : ℕ := 2
def side_length_triangle : ℕ := 1
def number_of_squares : ℕ := 4
def number_of_triangles : ℕ := 3

-- Define the perimeter calculation
def perimeter_of_figure : ℕ :=
  let perimeter_squares := (2 * (number_of_squares - 2) + 2 * 2 + 2 * 1) * side_length_square
  let perimeter_triangles := number_of_triangles * side_length_triangle
  perimeter_squares + perimeter_triangles

-- Target theorem
theorem comp_figure_perimeter : perimeter_of_figure = 17 := by
  sorry

end comp_figure_perimeter_l481_481153


namespace math_majors_consecutive_probability_l481_481195

def twelve_people := 12
def math_majors := 5
def physics_majors := 4
def biology_majors := 3

def total_ways := Nat.choose twelve_people math_majors

-- Computes the probability that all five math majors sit in consecutive seats
theorem math_majors_consecutive_probability :
  (12 : ℕ) / (Nat.choose twelve_people math_majors) = 1 / 66 := by
  sorry

end math_majors_consecutive_probability_l481_481195


namespace number_of_women_is_24_l481_481447

-- Define the variables and conditions
variables (x : ℕ) (men_initial : ℕ) (women_initial : ℕ) (men_current : ℕ) (women_current : ℕ)

-- representing the initial ratio and the changes
def initial_conditions : Prop :=
  men_initial = 4 * x ∧ women_initial = 5 * x ∧
  men_current = men_initial + 2 ∧ women_current = 2 * (women_initial - 3)

-- representing the current number of men
def current_men_condition : Prop := men_current = 14

-- The proof we need to generate
theorem number_of_women_is_24 (x : ℕ) (men_initial women_initial men_current women_current : ℕ)
  (h1 : initial_conditions x men_initial women_initial men_current women_current)
  (h2 : current_men_condition men_current) : women_current = 24 :=
by
  -- proof steps here
  sorry

end number_of_women_is_24_l481_481447


namespace length_of_DE_l481_481785

variable (D E F : Type)
variable [linear_ordered_field D]
variable [linear_ordered_field F]
variable [linear_ordered_field E]

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem length_of_DE 
  (D E F : Type) 
  [linear_ordered_field D] 
  [linear_ordered_field E] 
  [linear_ordered_field F] 
  (DE DF EF : ℝ) 
  (h1 : right_triangle DE EF DF) 
  (h2 : DF = 12) 
  (h3 : sin F = 5 / 12) : 
  DE = 5 := 
  sorry

end length_of_DE_l481_481785


namespace solve_proof_problem_l481_481099

noncomputable def proof_problem : Prop :=
  let short_videos_per_day := 2
  let short_video_time := 2
  let longer_videos_per_day := 1
  let week_days := 7
  let total_weekly_video_time := 112
  let total_short_video_time_per_week := short_videos_per_day * short_video_time * week_days
  let total_longer_video_time_per_week := total_weekly_video_time - total_short_video_time_per_week
  let longer_video_multiple := total_longer_video_time_per_week / short_video_time
  longer_video_multiple = 42

theorem solve_proof_problem : proof_problem :=
by
  /- Proof goes here -/
  sorry

end solve_proof_problem_l481_481099


namespace star_angle_sum_l481_481920

-- Define variables and angles for Petya's and Vasya's stars.
variables {α β γ δ ε : ℝ}
variables {φ χ ψ ω : ℝ}
variables {a b c d e : ℝ}

-- Conditions
def all_acute (a b c d e : ℝ) : Prop := a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90 ∧ e < 90
def one_obtuse (a b c d e : ℝ) : Prop := (a > 90 ∨ b > 90 ∨ c > 90 ∨ d > 90 ∨ e > 90)

-- Question: Prove the sum of the angles at the vertices of both stars is equal
theorem star_angle_sum : all_acute α β γ δ ε → one_obtuse φ χ ψ ω α → 
  α + β + γ + δ + ε = φ + χ + ψ + ω + α := 
by sorry

end star_angle_sum_l481_481920


namespace solve_for_number_l481_481523

-- Definition of variables and conditions in Lean
variables (x y some_number : ℤ)

-- Given conditions
def equation1 := 19 * (x + y) + 17 = 19 * (-x + y) - some_number
def condition1 := x = 1

-- The goal to prove
theorem solve_for_number (h1: equation1) (h2: condition1) : some_number = -55 :=
sorry

end solve_for_number_l481_481523


namespace sum_and_reverse_multiple_of_81_l481_481299

def reverse_number (digits : List ℕ) : ℕ :=
  digits.reverse.enumFrom(0).foldl (λ acc ⟨i, a⟩ => acc + a * 10^i) 0

def sum_of_digits (digits : List ℕ) : ℕ :=
  digits.sum

theorem sum_and_reverse_multiple_of_81 
  (digits : List ℕ) (h1 : digits.headI ≠ 0) (h2 : digits.lastI ≠ 0) :
  (81 ∣ (digits.enumFrom(0).foldl (λ acc ⟨i, a⟩ => acc + a * 10^i) 0 + 
         reverse_number digits)) ↔ 
  (81 ∣ sum_of_digits digits) :=
sorry

end sum_and_reverse_multiple_of_81_l481_481299


namespace vector_parallel_m_eq_one_l481_481382

theorem vector_parallel_m_eq_one (m : ℝ) (a b : ℝ × ℝ × ℝ)
  (ha : a = (2, m, 5))
  (hb : b = (4, m + 1, 10))
  (h_parallel : ∃ k : ℝ, ∀ i : fin 3, b.1 * k = a.1) :
  m = 1 :=
sorry

end vector_parallel_m_eq_one_l481_481382


namespace finish_together_in_4_days_l481_481641

-- Definitions for the individual days taken by A, B, and C
def days_for_A := 12
def days_for_B := 24
def days_for_C := 8 -- C's approximated days

-- The rates are the reciprocals of the days
def rate_A := 1 / days_for_A
def rate_B := 1 / days_for_B
def rate_C := 1 / days_for_C

-- The combined rate of A, B, and C
def combined_rate := rate_A + rate_B + rate_C

-- The total days required to finish the work together
def total_days := 1 / combined_rate

-- Theorem stating that the total days required is 4
theorem finish_together_in_4_days : total_days = 4 := 
by 
-- proof omitted
sorry

end finish_together_in_4_days_l481_481641


namespace solve_for_x_l481_481393

theorem solve_for_x (x : ℚ) (h : (1 / 2 - 1 / 3) = 3 / x) : x = 18 :=
sorry

end solve_for_x_l481_481393


namespace number_of_solutions_cos2x_plus_3sin2x_eq_cot2x_l481_481056

theorem number_of_solutions_cos2x_plus_3sin2x_eq_cot2x : 
  ∃ n : ℕ, n = 37 ∧ ∃ x : ℝ, -17 < x ∧ x < 100 ∧ cos x ^ 2 + 3 * (sin x) ^ 2 = (cos x / sin x) ^ 2 :=
by
  sorry

end number_of_solutions_cos2x_plus_3sin2x_eq_cot2x_l481_481056


namespace S_19_eq_95_l481_481913

variable {a : ℕ → ℝ} -- Arithmetic sequence

-- Given conditions
def sum_n (n : ℕ) : ℕ → ℝ := λ n, if n = 0 then 0 else n/2 * (a 1 + a n)
def condition1 (n : ℕ) : ℝ := sum_n n
def condition2 : Prop := a 3 + a 17 = 10

-- Prove S_19 = 95
theorem S_19_eq_95 (h1 : condition1 19 = 95) (h2 : condition2) : sum_n 19 = 95 := 
by 
  sorry

end S_19_eq_95_l481_481913


namespace overall_average_score_l481_481534

-- Definitions based on given conditions
def n_m : ℕ := 8   -- number of male students
def avg_m : ℚ := 87  -- average score of male students
def n_f : ℕ := 12  -- number of female students
def avg_f : ℚ := 92  -- average score of female students

-- The target statement to prove
theorem overall_average_score (n_m : ℕ) (avg_m : ℚ) (n_f : ℕ) (avg_f : ℚ) (overall_avg : ℚ) :
  n_m = 8 ∧ avg_m = 87 ∧ n_f = 12 ∧ avg_f = 92 → overall_avg = 90 :=
by
  sorry

end overall_average_score_l481_481534


namespace aaron_week_earnings_l481_481385

namespace AaronPay

def monday_hours : ℝ := 2
def tuesday_hours : ℝ := 1 + 15/60
def wednesday_hours : ℝ := 2 + 50/60
def friday_hours : ℝ := 40/60
def pay_rate : ℝ := 5

def total_hours_worked : ℝ := monday_hours + tuesday_hours + wednesday_hours + friday_hours
def total_pay : ℝ := total_hours_worked * pay_rate

theorem aaron_week_earnings : total_pay = 38.75 := by
  calc 
    total_pay = (monday_hours + tuesday_hours + wednesday_hours + friday_hours) * pay_rate : rfl
    ... = (2 + (1 + 15/60) + (2 + 50/60) + 40/60) * 5 : rfl
    ... = (2 + 1.25 + 2.833 + 0.667) * 5 : by norm_num
    ... = 7.75 * 5 : by norm_num
    ... = 38.75 : by norm_num

end aaron_week_earnings_l481_481385


namespace coeff_x2_in_binomial_expansion_l481_481790

theorem coeff_x2_in_binomial_expansion :
  (∃ c : ℕ, (x + (1/x))^10 = (c * x^2 + ...) ∧ c = 210) :=
sorry

end coeff_x2_in_binomial_expansion_l481_481790


namespace surface_area_correct_l481_481089

-- Define the points in the space rectangular coordinate system
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def O := Point3D.mk 0 0 0
def A := Point3D.mk 2 0 0
def B := Point3D.mk 0 2 0
def C := Point3D.mk 0 0 2

-- Function to calculate the surface area of the tetrahedron given the vertices in the space rectangular coordinate system.
def surface_area_tetrahedron (O A B C : Point3D) : ℝ :=
  (0.5 * ((A.x - O.x) * (B.y - O.y))) +
  (0.5 * ((A.x - O.x) * (C.z - O.z))) +
  (0.5 * ((B.y - O.y) * (C.z - O.z))) +
  ((√3) / 4) * ((2 * √2) ^ 2)

-- Prove that the surface area of the tetrahedron is 6 + 2√3 using given points O, A, B, and C.
theorem surface_area_correct :
  surface_area_tetrahedron O A B C = 6 + 2 * √3 := 
  by
  sorry

end surface_area_correct_l481_481089


namespace probability_of_event_E_l481_481212

-- Define the event E as rolling a number less than 5
def event_E (n : ℕ) : Prop := n < 5

-- Define the sample space as the set {1, 2, ..., 8} for an 8-sided die
def sample_space : set ℕ := {n | n ≥ 1 ∧ n ≤ 8}

-- Define the probability function
def probability (E : set ℕ) (S : set ℕ) : ℚ :=
  if S.nonempty then
    E.to_finset.card.to_rat / S.to_finset.card.to_rat
  else 0

-- Prove that the probability of event E is 1/2 given the sample space
theorem probability_of_event_E :
  probability {n | event_E n} sample_space = 1 / 2 := by
  sorry

end probability_of_event_E_l481_481212


namespace no_integer_solution_for_150_l481_481356

theorem no_integer_solution_for_150 : ∀ (x : ℤ), x - Int.sqrt x ≠ 150 := 
sorry

end no_integer_solution_for_150_l481_481356


namespace bc_gt_hk_l481_481396

-- Define the positive reals and progressions
variables (a b c d h k : ℝ) (t r : ℝ)
-- Conditions of arithmetic progression
hypothesis (h1 : a = d + 3 * t) (h2 : b = d + 2 * t) (h3 : c = d + t) (h4 : d > 0)
-- Conditions of geometric progression
hypothesis (h5 : a = d * r^3) (h6 : h = d * r^2) (h7 : k = d * r)
-- Ordering conditions
hypothesis (h8 : a > b) (h9 : b > c) (h10 : c > d) (h11 : a > h) (h12 : h > k) (h13 : k > d)

-- Conclusion to be proved
theorem bc_gt_hk : (b * c) > (h * k) :=
by
  -- Proof goes here
  sorry

end bc_gt_hk_l481_481396


namespace negation_of_proposition_l481_481902

theorem negation_of_proposition (x : ℝ) : ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_of_proposition_l481_481902


namespace cevians_concurrent_l481_481708

-- Define the points and angles conditions
variables {A B C X Y Z : Type*}

-- Angles conditions
variables 
  (h1 : ∠ B A Z = ∠ C A Y) 
  (h2 : ∠ C B X = ∠ A B Z) 
  (h3 : ∠ A C Y = ∠ B C X)

-- Problem statement
theorem cevians_concurrent 
  (h1 : ∠ B A Z = ∠ C A Y) 
  (h2 : ∠ C B X = ∠ A B Z) 
  (h3 : ∠ A C Y = ∠ B C X) : 
  concurrent A X B Y C Z :=
sorry

end cevians_concurrent_l481_481708


namespace number_of_women_l481_481443

theorem number_of_women (x : ℕ) 
  (h1 : 4 * x + 2 = 14) : 2 * (5 * x - 3) = 24 :=
by 
  ext
  sorry

end number_of_women_l481_481443


namespace total_workers_l481_481257

open Nat

theorem total_workers (x y : ℕ) :
  (45 = 20 + 10 + 15) →
  ∀ m, (20 + 10 + 15 = 45) →
      (20 + 10 + 15) * m = 45 * 300 → 
      m = 900 :=
by
  intros h1 h2 h3
  have h4: 45 = 45 := rfl
  have h5: (20 + 10 + 15) = 45 := h2
  sorry

end total_workers_l481_481257


namespace probability_of_rolling_less_than_five_l481_481208

/-- Rolling an 8-sided die -/
def outcomes := { 1, 2, 3, 4, 5, 6, 7, 8 }

/-- Successful outcomes (numbers less than 5) -/
def successful_outcomes := { 1, 2, 3, 4 }

/-- Number of total outcomes -/
def total_outcomes := 8

/-- Number of successful outcomes -/
def num_successful_outcomes := 4

/-- Probability of rolling a number less than 5 on an 8-sided die -/
def probability : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_rolling_less_than_five : probability = 1 / 2 := by
  sorry

end probability_of_rolling_less_than_five_l481_481208


namespace plane_through_bisectors_perpendicular_l481_481709

-- Definitions for vectors and trihedral angle conditions
variables {V : Type} [inner_product_space ℝ V]
variables (a b c : V) (h_a : ∥a∥ = 1) (h_b : ∥b∥ = 1) (h_c : ∥c∥ = 1)

-- Condition: <a + b, b + c> = 0
def bisectors_perpendicular : Prop :=
  inner (a + b) (b + c) = 0

-- Theorem statement: Prove plane through bisectors is perpendicular to third plane 
theorem plane_through_bisectors_perpendicular (h : bisectors_perpendicular a b c) :
  ∃ n : V, ∥n∥ = 1 ∧ ∀ x ∈ submodule.span ℝ {a + b, b + c}, inner (a + b) n = 0 ∧ inner n (b + c) = 0 :=
sorry

end plane_through_bisectors_perpendicular_l481_481709


namespace RachelLeftoverMoney_l481_481868

def RachelEarnings : ℝ := 200
def SpentOnLunch : ℝ := (1/4) * RachelEarnings
def SpentOnDVD : ℝ := (1/2) * RachelEarnings
def TotalSpent : ℝ := SpentOnLunch + SpentOnDVD

theorem RachelLeftoverMoney : RachelEarnings - TotalSpent = 50 := by
  sorry

end RachelLeftoverMoney_l481_481868


namespace single_translation_possible_l481_481087

theorem single_translation_possible 
  (points : set (ℝ × ℝ))
  (h : ∀ p1 p2 p3 ∈ points, ∃ t : ℝ × ℝ, is_translation_of_rectangle (t + p1) (t + p2) (t + p3) (0,0) (1,0) (0,2) (1,2)) :
  ∃ t : ℝ × ℝ, ∀ p ∈ points, (t + p) ∈ rectangle (0,0) (1,0) (0,2) (1,2) :=
sorry

def is_translation_of_rectangle (a b c d pa pb pc pd : ℝ × ℝ) : Prop := 
  a + pa = b + pb ∧ 
  a + pa = c + pc ∧ 
  a + pa = d + pd ∧ 
  b + pb = c + pc ∧ 
  b + pb = d + pd ∧ 
  c + pc = d + pd
  
def rectangle (a b c d : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  p = a ∨ p = b ∨ p = c ∨ p = d

end single_translation_possible_l481_481087


namespace system_of_equations_n_eq_1_l481_481741

theorem system_of_equations_n_eq_1 {x y n : ℝ} 
  (h₁ : 5 * x - 4 * y = n) 
  (h₂ : 3 * x + 5 * y = 8)
  (h₃ : x = y) : 
  n = 1 := 
by
  sorry

end system_of_equations_n_eq_1_l481_481741


namespace shorts_and_jersey_different_colors_probability_l481_481815

def shorts_colors := {'black, 'gold, 'blue}
def jerseys_colors := {'black, 'white, 'gold, 'blue}

noncomputable def probability_different_colors : ℚ :=
let total_combinations := shorts_colors.card * jerseys_colors.card in
let mismatched_combinations :=
  shorts_colors.card * (jerseys_colors.card - 1) in
mismatched_combinations / total_combinations

theorem shorts_and_jersey_different_colors_probability :
  probability_different_colors = 3 / 4 :=
by
  sorry

end shorts_and_jersey_different_colors_probability_l481_481815


namespace find_discounts_l481_481492

variables (a b c : ℝ)
variables (x y z : ℝ)

theorem find_discounts (h1 : 1.1 * a - x * a = 0.99 * a)
                       (h2 : 1.12 * b - y * b = 0.99 * b)
                       (h3 : 1.15 * c - z * c = 0.99 * c) : 
x = 0.11 ∧ y = 0.13 ∧ z = 0.16 := 
sorry

end find_discounts_l481_481492


namespace polynomial_roots_sum_l481_481189

theorem polynomial_roots_sum (a b c d e : ℝ) (h₀ : a ≠ 0)
  (h1 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
  (h2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h3 : a * 2^4 + b * 2^3 + c * 2^2 + d * 2 + e = 0) :
  (b + d) / a = -2677 := 
sorry

end polynomial_roots_sum_l481_481189


namespace cross_ratio_invariance_l481_481104

variables (A B C A' B' C' X Y Z X' Y' Z' P : Point)
variables (BC CA AB YZ ZX XY : Line)

-- Conditions
axiom cond1 : A' ∈ BC
axiom cond2 : B' ∈ CA
axiom cond3 : C' ∈ AB
axiom cond4 : X' ∈ YZ
axiom cond5 : Y' ∈ ZX
axiom cond6 : Z' ∈ XY
axiom cond7 : P ∈ (Line.mk A X)
axiom cond8 : P ∈ (Line.mk B Y)
axiom cond9 : P ∈ (Line.mk C Z)
axiom cond10 : P ∈ (Line.mk A' X')
axiom cond11 : P ∈ (Line.mk B' Y')
axiom cond12 : P ∈ (Line.mk C' Z')

-- Theorem to be proved
theorem cross_ratio_invariance :
  (BA' / A'C) * (CB' / B'A) * (AC' / C'B) = (YX' / X'Z) * (ZY' / Y'X) * (XZ' / Z'Y) :=
sorry

end cross_ratio_invariance_l481_481104


namespace remy_used_25_gallons_l481_481875

noncomputable def RomanGallons : ℕ := 8

noncomputable def RemyGallons (R : ℕ) : ℕ := 3 * R + 1

theorem remy_used_25_gallons (R : ℕ) (h1 : RemyGallons R = 1 + 3 * R) (h2 : R + RemyGallons R = 33) : RemyGallons R = 25 := by
  sorry

end remy_used_25_gallons_l481_481875


namespace white_ball_probability_l481_481777

variable {P : Type} [Probability P]

def prob_white_ball (P_A P_B P_C : P) :=
  P_A + P_B = (5 : ℝ) / 12 ∧
  P_B + P_C = (5 : ℝ) / 12 ∧
  P_A + P_B + P_C = (2 : ℝ) / 3 →
  P_C = (1 : ℝ) / 4

theorem white_ball_probability (P_A P_B P_C : P) :
  P_A + P_B = (5 : ℝ) / 12 ∧
  P_B + P_C = (5 : ℝ) / 12 ∧
  P_A + P_B + P_C = (2 : ℝ) / 3 →
  P_C = (1 : ℝ) / 4 :=
by
  sorry

end white_ball_probability_l481_481777


namespace chapters_per_day_calc_l481_481003

theorem chapters_per_day_calc (total_pages : ℝ) (total_chapters : ℝ) (total_days : ℝ) :
  total_pages = 193 → total_chapters = 15 → total_days = 660 →
  (total_chapters / total_days) = 0.0227 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end chapters_per_day_calc_l481_481003


namespace tangency_distance_half_difference_l481_481425

/-- Given a triangle ABC with sides AB, AC, and BC, where BC = a and AC = b with a > b,
    M as the midpoint of AB, α and β the incircles of triangles ACM and BCM,
    and A' and B' the points of tangency of α and β on CM, respectively,
    we need to prove that A'B' = (a - b) / 2. -/
theorem tangency_distance_half_difference (a b : ℝ) (A B C M A' B' : Point)
  (h1 : a > b)
  (h2 : a = dist B C)
  (h3 : b = dist A C)
  (h4 : midpoint A B M)
  (h5 : incircle α ACM)
  (h6 : incircle β BCM)
  (h7 : tangent_point α A' CM)
  (h8 : tangent_point β B' CM):
  dist A' B' = (a - b) / 2 :=
sorry

end tangency_distance_half_difference_l481_481425


namespace determine_other_asymptote_l481_481507

def hyperbola_asymptote_equation (asymptote1 : ℝ → ℝ)
                                 (foci_x : ℝ)
                                 (other_asymptote : ℝ → ℝ) :=
  ∃ C : ℝ × ℝ, asymptote1 = λ x, 5 * x ∧ foci_x = 3 ∧
                            C = (3, 15) ∧
                            other_asymptote = λ x, -5 * x + 30

theorem determine_other_asymptote :
  hyperbola_asymptote_equation (λ x, 5 * x) 3 (λ x, -5 * x + 30) :=
by
  sorry

end determine_other_asymptote_l481_481507


namespace area_on_larger_sphere_l481_481963

-- Define the variables representing the radii and the given area on the smaller sphere
variable (r1 r2 : ℝ) (area1 : ℝ)

-- Given conditions
def conditions : Prop :=
  r1 = 4 ∧ r2 = 6 ∧ area1 = 37

-- Define the statement that we need to prove
theorem area_on_larger_sphere (h : conditions r1 r2 area1) : 
  let area2 := area1 * (r2^2 / r1^2) in
  area2 = 83.25 :=
by
  -- Insert the proof here
  sorry

end area_on_larger_sphere_l481_481963


namespace cos_double_angle_l481_481339

theorem cos_double_angle (α : ℝ) (h : sin α = 3 / 5) : cos (2 * α) = 7 / 25 :=
by
  sorry

end cos_double_angle_l481_481339


namespace f_2012_is_2013_l481_481761

-- Define the function f and the given conditions
axiom f : ℕ → ℕ
axiom f_comp_eq_2n_plus_3 : ∀ n, f(f(n)) + f(n) = 2 * n + 3
axiom f_0_is_1 : f 0 = 1

-- State the theorem to be proved
theorem f_2012_is_2013 : f 2012 = 2013 :=
by
  sorry

end f_2012_is_2013_l481_481761


namespace probability_red_blue_yellow_l481_481184

-- Define the probabilities for white, green, and black marbles
def p_white : ℚ := 1/4
def p_green : ℚ := 1/6
def p_black : ℚ := 1/8

-- Define the problem: calculating the probability of drawing a red, blue, or yellow marble
theorem probability_red_blue_yellow : 
  p_white = 1/4 → p_green = 1/6 → p_black = 1/8 →
  (1 - (p_white + p_green + p_black)) = 11/24 := 
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end probability_red_blue_yellow_l481_481184


namespace coeff_x_cubed_in_expansion_l481_481160

theorem coeff_x_cubed_in_expansion (x : ℝ) : 
  (expand (1 : ℝ - 2 * x) 5).coeff 3 = -80 := 
begin
  sorry
end

end coeff_x_cubed_in_expansion_l481_481160


namespace circle_center_distance_l481_481471

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem circle_center_distance
  {P Q R : Type}
  (dist_PQ : dist P Q = 17)
  (dist_PR : dist P R = 19)
  (dist_QR : dist Q R = 20) :
  let I := incenter P Q R,
      E := excenter P Q R
  in dist I E = sqrt 1360 - sqrt 157 :=
by
  sorry

end circle_center_distance_l481_481471


namespace length_of_CD_l481_481910

noncomputable def volume_of_cylinder_with_hemispheres (r h : ℝ) : ℝ :=
  let volume_of_hemisphere := (2 * (2 / 3) * π * r^3)
  let volume_of_cylinder := π * r^2 * h
  volume_of_cylinder + volume_of_hemisphere

theorem length_of_CD 
  (V : ℝ) 
  (r : ℝ) 
  (h : ℝ) 
  (volume_eq : volume_of_cylinder_with_hemispheres r h = V)
  (radius_eq : r = 4)
  (volume_given : V = 352 * π) :
  h = 16.67 := 
by
  sorry

end length_of_CD_l481_481910


namespace relationship_among_f_l481_481261

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f (x + 1) = f (x - 1)
axiom condition2 : ∀ x : ℝ, f (x + 1) = f (-x - 1)
axiom condition3 : ∀ x1 x2 : ℝ, x1 ∈ Icc (0 : ℝ) 1 → x2 ∈ Icc (0 : ℝ) 1 → (f x1 - f x2) * (x1 - x2) > 0

theorem relationship_among_f :
  f 3 > f (3 / 2) ∧ f (3 / 2) > f 2 := 
begin
  -- Proof goes here, but for now we'll use sorry
  sorry
end

end relationship_among_f_l481_481261


namespace correct_statements_l481_481002

-- Insert conditions as definitions in Lean 4
def a_condition (a : ℝ) : Prop := a > 0 ∧ a ≠ 1

-- Definitions of the statements
def statement1 (a : ℝ) : Prop := ∀ x : ℝ, a_condition a → (λ y, y = a^(x+2)) = (λ y, y = a^x) ∘ (λ z, z - 2)

def statement2 : Prop := ¬ ∀ x : ℝ, (λ y, y = 2^x) = (λ y, y = real.log y / real.log 2) ∘ (λ z, z^(-1))

def statement3 : Prop := ¬ (∀ x : ℝ, (log 5 (2 * x + 1) = log 5 (x^2 - 2)) ↔ x = -1 ∨ x = 3)

def statement4 : Prop := ∀ x : ℝ, x ∈ set.Ioo (-1 : ℝ) 1 → ln(1 + x) - ln(1 - x) = - (ln(1 - x) - ln(1 + x))

/-- Prove the correctness of the statements -/
theorem correct_statements (a : ℝ) (x : ℝ) : a_condition a →
  (statement1 a ∧ statement4 ∧ ¬ statement2 ∧ ¬ statement3) :=
by {
  sorry
}

end correct_statements_l481_481002


namespace arithmetic_sequence_a8_l481_481076

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) (d : ℤ) :
  a 2 = 4 → a 4 = 2 → a 8 = -2 :=
by intros ha2 ha4
   sorry

end arithmetic_sequence_a8_l481_481076


namespace probability_of_rolling_less_than_five_l481_481207

/-- Rolling an 8-sided die -/
def outcomes := { 1, 2, 3, 4, 5, 6, 7, 8 }

/-- Successful outcomes (numbers less than 5) -/
def successful_outcomes := { 1, 2, 3, 4 }

/-- Number of total outcomes -/
def total_outcomes := 8

/-- Number of successful outcomes -/
def num_successful_outcomes := 4

/-- Probability of rolling a number less than 5 on an 8-sided die -/
def probability : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_rolling_less_than_five : probability = 1 / 2 := by
  sorry

end probability_of_rolling_less_than_five_l481_481207


namespace reporters_cover_local_politics_l481_481610

theorem reporters_cover_local_politics :
  ∀ (total : ℕ) (p_cover : ℕ) (local_cover : ℕ), (total = 100) → 
  (0.5 * total = p_cover) → 
  (0.3 * p_cover = local_cover) → 
  (local_cover = 35) := 
by
  intros total p_cover local_cover total_eq total_p_cover_eq local_p_cover_eq
  sorry

end reporters_cover_local_politics_l481_481610


namespace tangent_line_at_2_l481_481029

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, 2 * f (4 - x) = f x + x^2 - 10 * x + 17

theorem tangent_line_at_2 :
  ∀ (f : ℝ → ℝ),
  (∀ x : ℝ, 2 * f (4 - x) = f x + x^2 - 10 * x + 17) →
  tangent_line f 2 (f 2) = (λ x : ℝ, 2 * x - 3) := sorry

-- Helper definition to represent the tangent line equation
def tangent_line (f : ℝ → ℝ) (a : ℝ) (fa : ℝ) : (ℝ → ℝ) :=
  let f' := (λ x, sorry);  -- Placeholder for the derivative of f
  λ x, fa + f' a * (x - a)

end tangent_line_at_2_l481_481029


namespace identify_minor_premise_l481_481423

-- Define the statements in Lean.
def major_premise : Prop :=
  ∀ (X : Type) (student : X → Prop) (good_deed : X → Prop),
    (∀ x, student x → good_deed x)

def minor_premise : Prop :=
  ∃ (ZhangSan : Type) (student : ZhangSan → Prop),
    student ZhangSan

def conclusion : Prop :=
  ∀ (ZhangSan : Type) (student : ZhangSan → Prop) (good_deed : ZhangSan → Prop),
    student ZhangSan → good_deed ZhangSan

-- The main theorem we need to prove.
theorem identify_minor_premise : minor_premise :=
  sorry

end identify_minor_premise_l481_481423


namespace fraction_of_students_liking_maths_l481_481775

variable (students total_liking_h_m : ℕ)
variable (m : ℚ)
variable (like_maths : fraction (students) = m)

theorem fraction_of_students_liking_maths (h1 : students = 25)
                                         (h2 : total_liking_h_m = 20)
                                         (h3 : fraction (students) = m)
: m = 2/5 := 
sorry

end fraction_of_students_liking_maths_l481_481775


namespace D_is_midpoint_AE_l481_481427

noncomputable theory -- Noncomputable declaration as geometry involves non-algorithmic constructs

open locale classical -- Opens classical logic for use (needed for non-constructive proof)

variables (A B C D E : Point)
variables (Γ1 Γ2 : Circle)
variables (ABC : Triangle)
variables (circumcircle_ABC : Circle)

-- Conditions
axiom Triangle_ABC : is_triangle ABC
axiom Gamma1_is_circle : passes_through Γ1 B ∧ is_tangent Γ1 CA A
axiom Gamma2_is_circle : passes_through Γ2 C ∧ is_tangent Γ2 AB A
axiom Intersect_Γ1_Γ2_A_D : intersects_at Γ1 Γ2 A ∧ intersects_at Γ1 Γ2 D
axiom Line_AD_intersects_circumcircle_ABC_E : intersects_at (line_through A D) circumcircle_ABC E

-- Theorem
theorem D_is_midpoint_AE :
  is_midpoint D A E :=
sorry

end D_is_midpoint_AE_l481_481427


namespace shoe_company_additional_revenue_l481_481635

noncomputable def additional_revenue (current_sales : ℕ) (sports_sales : ℕ) (casual_sales : ℕ) 
  (yearly_target : ℕ) (additional_revenue_factor : ℕ) : ℝ :=
by
  let current_yearly_sales := current_sales * 12
  let additional_needed := yearly_target - current_yearly_sales
  let additional_monthly_needed := additional_needed / 12
  let x := additional_monthly_needed / (1 + additional_revenue_factor : ℝ)
  let sports_additional := additional_revenue_factor * x
  exact (x, sports_additional)

theorem shoe_company_additional_revenue :
  additional_revenue 5500 3200 2300 72000 2 = (166.67, 333.34) :=
sorry

end shoe_company_additional_revenue_l481_481635


namespace total_cost_is_correct_l481_481098

-- Conditions
def cost_of_piano : ℕ := 500
def number_of_lessons : ℕ := 20
def cost_per_lesson : ℕ := 40
def discount_rate : ℝ := 0.25

-- Calculations
def original_cost_of_lessons : ℕ := number_of_lessons * cost_per_lesson
def discount_amount : ℝ := original_cost_of_lessons * discount_rate
def discounted_cost_of_lessons : ℕ := original_cost_of_lessons - discount_amount.to_nat
def total_cost : ℕ := cost_of_piano + discounted_cost_of_lessons

-- Proof statement
theorem total_cost_is_correct : total_cost = 1100 := by
  sorry

end total_cost_is_correct_l481_481098


namespace greatest_perimeter_l481_481888

theorem greatest_perimeter (w l : ℕ) (h1 : w * l = 12) : 
  ∃ (P : ℕ), P = 2 * (w + l) ∧ ∀ (w' l' : ℕ), w' * l' = 12 → 2 * (w' + l') ≤ P := 
sorry

end greatest_perimeter_l481_481888


namespace ellipse_equation_triangle_area_range_l481_481710

open Real

def ellipse : Type := ℝ × ℝ

def focus_parabola := (1:ℝ, 0:ℝ)

def triangle_area (a b c : ellipse) :=
  (1 / 2) * abs ((a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)))

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hb_le_ha : b < a)
  (h_focus : focus_parabola = (1, 0))
  (h_max_area : ∀ (M : ellipse), triangle_area M (1, 0) (-1, 0) ≤ 1) :
  (a = sqrt 2) ∧ (b = 1) → ellipse = (λ ⟨x, y⟩, x^2 / 2 + y^2 = 1) :=
sorry

theorem triangle_area_range (k m : ℝ)
  (hk_non_zero : k ≠ 0)
  (hm_non_zero : m ≠ 0)
  (hm_bounds : 0 < m^2 ∧ m^2 < 2 ∧ m^2 ≠ 1)
  (h_geometric_mean_slopes : k^2 = 1/2) :
  ∀ (A B : ellipse), triangle_area (0,0) A B ≤ sqrt 2 / 2 :=
sorry

end ellipse_equation_triangle_area_range_l481_481710


namespace flag_arrangement_remainder_l481_481182

theorem flag_arrangement_remainder :
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  M % div = 441 := 
by
  -- Definitions
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  -- Proof
  sorry

end flag_arrangement_remainder_l481_481182


namespace selected_sample_l481_481959

def random_number_table : list (list string) :=
  [["03", "47", "4373", "86", "36", "96", "47", "36", "61", "46", "98", "63", "71", "62", "33", "26", "16", "80"],
   ["45", "60", "11", "14", "10", "95", "97", "74", "24", "67", "62", "42", "81", "14", "57", "20", "42", "53"],
   ["32", "37", "32", "27", "07", "36", "07", "51", "24", "51", "79", "89", "73", "16", "76", "62", "27", "66"],
   ["56", "50", "26", "71", "07", "32", "90", "79", "78", "53", "13", "55", "38", "58", "59", "88", "97", "54"],
   ["14", "10", "12", "56", "85", "99", "26", "96", "96", "68", "27", "31", "05", "03", "72", "93", "15", "57"],
   ["12", "10", "14", "21", "88", "26", "49", "81", "76", "55", "59", "56", "35", "64", "38", "54", "82", "46"],
   ["22", "31", "62", "43", "09", "90", "06", "18", "44", "32", "53", "23", "83", "01", "30", "30"]]

def is_valid_student_number (s : string) : bool :=
  match s.toInt? with
  | some i => 1 ≤ i ∧ i ≤ 50
  | none   => false

def selected_student_numbers : list string :=
  list.filterMap (λ row, list.filter is_valid_student_number row).flatten (list.drop 2 random_number_table) -- Start reading from the 3rd row

theorem selected_sample : selected_student_numbers.take 5 = ["22", "02", "10", "29", "07"] :=
  by sorry

end selected_sample_l481_481959


namespace B_join_time_l481_481232

theorem B_join_time (x : ℕ) (hx : (45000 * 12) / (27000 * (12 - x)) = 2) : x = 2 :=
sorry

end B_join_time_l481_481232


namespace perimeter_of_square_from_quadratic_roots_l481_481342

theorem perimeter_of_square_from_quadratic_roots :
  let r1 := 1
  let r2 := 10
  let larger_root := if r1 > r2 then r1 else r2
  let area := larger_root * larger_root
  let side_length := Real.sqrt area
  4 * side_length = 40 := by
  let r1 := 1
  let r2 := 10
  let larger_root := if r1 > r2 then r1 else r2
  let area := larger_root * larger_root
  let side_length := Real.sqrt area
  sorry

end perimeter_of_square_from_quadratic_roots_l481_481342


namespace incorrect_statement_D_l481_481590

theorem incorrect_statement_D :
  (∃ x : ℝ, x ^ 3 = -64 ∧ x = -4) ∧
  (∃ y : ℝ, y ^ 2 = 49 ∧ y = 7) ∧
  (∃ z : ℝ, z ^ 3 = 1 / 27 ∧ z = 1 / 3) ∧
  (∀ w : ℝ, w ^ 2 = 1 / 16 → w = 1 / 4 ∨ w = -1 / 4)
  → ¬ (∀ w : ℝ, w ^ 2 = 1 / 16 → w = 1 / 4) :=
by
  sorry

end incorrect_statement_D_l481_481590


namespace logarithmic_function_satisfies_property_l481_481982

noncomputable def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y

theorem logarithmic_function_satisfies_property :
  satisfies_property (λ x, Real.log x) :=
by sorry

end logarithmic_function_satisfies_property_l481_481982


namespace table_cost_l481_481611

variable (T : ℝ) -- Cost of the table
variable (C : ℝ) -- Cost of a chair

-- Conditions
axiom h1 : C = T / 7
axiom h2 : T + 4 * C = 220

theorem table_cost : T = 140 :=
by
  sorry

end table_cost_l481_481611


namespace log_inverse_transform_l481_481058

theorem log_inverse_transform (x : ℝ) (h : log 16 (x - 1) = 1 / 2) : (1 / log x 2) = 1 / log 5 2 :=
sorry

end log_inverse_transform_l481_481058


namespace trig_expression_simplification_l481_481114

theorem trig_expression_simplification :
  let d := Real.pi / 7
  (\sin (2 * d) * cos (3 * d) * sin (4 * d)) / 
  (\sin d * sin (2 * d) * sin (3 * d) * cos (4 * d)) = 
  (\sin (2 * Real.pi / 7) * sin (4 * Real.pi / 7)) / 
  (sin (Real.pi / 7) * sin (3 * Real.pi / 7)) := 
by
  sorry

end trig_expression_simplification_l481_481114


namespace q_r_share_difference_l481_481985

theorem q_r_share_difference
  (T : ℝ) -- Total amount of money
  (x : ℝ) -- Common multiple of shares
  (p_share q_share r_share s_share : ℝ) -- Shares before tax
  (p_tax q_tax r_tax s_tax : ℝ) -- Tax percentages
  (h_ratio : p_share = 3 * x ∧ q_share = 7 * x ∧ r_share = 12 * x ∧ s_share = 5 * x) -- Ratio condition
  (h_tax : p_tax = 0.10 ∧ q_tax = 0.15 ∧ r_tax = 0.20 ∧ s_tax = 0.25) -- Tax condition
  (h_difference_pq : q_share * (1 - q_tax) - p_share * (1 - p_tax) = 2400) -- Difference between p and q after tax
  : (r_share * (1 - r_tax) - q_share * (1 - q_tax)) = 2695.38 := sorry

end q_r_share_difference_l481_481985


namespace speed_ratio_between_Q_and_R_l481_481293

-- Define the dimensions of the park and the division of the bottom edge.
def length_of_park : ℝ := 600
def width_of_park : ℝ := 400
def segment_length : ℝ := length_of_park / 6

-- Define the positions of the points Q and R.
def position_Q : ℝ := 4 * segment_length
def position_R : ℝ := 3 * segment_length

-- Define the distances Betty and Ann travel to reach points Q and R.
def distance_betty_to_Q : ℝ := length_of_park + width_of_park + position_Q
def distance_ann_to_Q : ℝ := width_of_park + position_Q

def distance_betty_to_R : ℝ := length_of_park + width_of_park + position_R
def distance_ann_to_R : ℝ := width_of_park + position_R

-- Define the speed ratios when meeting at points Q and R.
def speed_ratio_Q : ℝ := distance_betty_to_Q / distance_ann_to_Q
def speed_ratio_R : ℝ := distance_betty_to_R / distance_ann_to_R

-- Define the possible speed ratio to be checked
def possible_speed_ratio : ℝ := 9 / 4

theorem speed_ratio_between_Q_and_R :
  speed_ratio_Q < possible_speed_ratio ∧ possible_speed_ratio < speed_ratio_R :=
sorry

end speed_ratio_between_Q_and_R_l481_481293


namespace parallelepiped_diagonal_relationship_l481_481202

theorem parallelepiped_diagonal_relationship {a b c d e f g : ℝ} 
  (h1 : c = d) 
  (h2 : e = e) 
  (h3 : f = f) 
  (h4 : g = g) 
  : a^2 + b^2 + c^2 + g^2 = d^2 + e^2 + f^2 :=
by
  sorry

end parallelepiped_diagonal_relationship_l481_481202


namespace log_eq_two_or_half_l481_481005

def M : Set ℕ := {2, 4}
def N (a b : ℕ) : Set ℕ := {a, b}

theorem log_eq_two_or_half (a b : ℕ) (h : N a b = M) : 
  log a b = 2 ∨ log a b = 1 / 2 := by 
  sorry

end log_eq_two_or_half_l481_481005


namespace smallest_a_gt_1_g2_eq_ga2_l481_481488

def g (x : ℕ) : ℕ :=
  if x % 28 = 0 then x / 28
  else if x % 7 = 0 then 4 * x
  else if x % 4 = 0 then 7 * x
  else x + 4

def g_iter : ℕ → ℕ → ℕ
| 0, x     := x
| (n+1), x := g (g_iter n x)

theorem smallest_a_gt_1_g2_eq_ga2 : ∃ a, a > 1 ∧ g_iter a 2 = g 2 ∧ ∀ b, b > 1 ∧ b < a → g_iter b 2 ≠ g 2 :=
begin
  use 6,
  split,
  { linarith },
  split,
  { refl }, -- g_iter 6 2 = 6 which is g 2
  { intros b hb,
    rcases hb with ⟨hb1, hb2⟩,
    -- the previous values of b by the solution steps
    by_interval.search
  },
  sorry
end

end smallest_a_gt_1_g2_eq_ga2_l481_481488


namespace solve_quadratic_l481_481597

theorem solve_quadratic :
  ∀ x : ℝ, x^2 + 10*x + 9 = 0 ↔ x = -9 ∨ x = -1 := 
by
  intro x
  split
  sorry -- Proof goes here
  sorry -- Proof goes here

end solve_quadratic_l481_481597


namespace each_person_share_approx_l481_481553

noncomputable def dining_bill : ℝ := 211.00
noncomputable def tip_rate : ℝ := 0.15
noncomputable def number_of_people : ℕ := 10

noncomputable def total_tip : ℝ := tip_rate * dining_bill
noncomputable def final_bill : ℝ := dining_bill + total_tip
noncomputable def share_per_person : ℝ := final_bill / number_of_people

theorem each_person_share_approx : Real.roundDec 2 share_per_person = 24.27 := by
  let total_tip := 0.15 * 211.00
  let final_bill := 211.00 + total_tip
  let share_per_person := final_bill / 10
  have h : Real.roundDec 2 share_per_person = 24.27
  sorry

end each_person_share_approx_l481_481553


namespace moles_of_BaO_l481_481320

theorem moles_of_BaO (e : ℕ) : 
  ∀ (BaO H₂O Ba(OH)₂ : Type) 
  (reaction : BaO → H₂O → Ba(OH)₂) 
  (mole_ratio : ∀ (a b : BaO) (c d : H₂O) (x y : Ba(OH)₂), 
    reaction a b = x → reaction c d = y → a = c ∧ x = y ∧ 1 = 1), 
  ∀ (formed_baoh2_moles : ℕ), 
  formed_baoh2_moles = e → e = e :=
by
  intros BaO H₂O Ba(OH)₂ reaction mole_ratio formed_baoh2_moles formation_eq
  simp only [formation_eq]
  sorry

end moles_of_BaO_l481_481320


namespace average_salary_of_laborers_l481_481780

-- Define the main statement as a theorem
theorem average_salary_of_laborers 
  (total_workers : ℕ)
  (total_salary_all : ℕ)
  (supervisors : ℕ)
  (supervisor_salary : ℕ)
  (laborers : ℕ)
  (expected_laborer_salary : ℝ) :
  total_workers = 48 → 
  total_salary_all = 60000 →
  supervisors = 6 →
  supervisor_salary = 2450 →
  laborers = 42 →
  expected_laborer_salary = 1078.57 :=
sorry

end average_salary_of_laborers_l481_481780


namespace solve_hours_l481_481918

variable (x y : ℝ)

-- Conditions
def Condition1 : x > 0 := sorry
def Condition2 : y > 0 := sorry
def Condition3 : (2:ℝ) / 3 * y / x + (3 * x * y - 2 * y^2) / (3 * x) = x * y / (x + y) + 2 := sorry
def Condition4 : 2 * y / (x + y) = (3 * x - 2 * y) / (3 * x) := sorry

-- Question: How many hours would it take for A and B to complete the task alone?
theorem solve_hours : x = 6 ∧ y = 3 := 
by
  -- Use assumed conditions and variables to define the context
  have h1 := Condition1
  have h2 := Condition2
  have h3 := Condition3
  have h4 := Condition4
  -- Combine analytical relationship and solve for x and y 
  sorry

end solve_hours_l481_481918


namespace classify_tangents_through_point_l481_481661

-- Definitions for the Lean theorem statement
noncomputable def curve (x : ℝ) : ℝ :=
  x^3 - x

noncomputable def phi (t x₀ y₀ : ℝ) : ℝ :=
  2*t^3 - 3*x₀*t^2 + (x₀ + y₀)

theorem classify_tangents_through_point (x₀ y₀ : ℝ) :
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) = 
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) :=
  sorry

end classify_tangents_through_point_l481_481661


namespace no_nat_nums_gt_one_divisibility_conditions_l481_481674

theorem no_nat_nums_gt_one_divisibility_conditions :
  ¬ ∃ (a b c : ℕ), 
    1 < a ∧ 1 < b ∧ 1 < c ∧
    (c ∣ a^2 - 1) ∧ (b ∣ a^2 - 1) ∧ 
    (a ∣ c^2 - 1) ∧ (b ∣ c^2 - 1) :=
by 
  sorry

end no_nat_nums_gt_one_divisibility_conditions_l481_481674


namespace direct_proportional_functions_l481_481681

theorem direct_proportional_functions (a : ℝ) :
  (∀ x : ℝ, a^2 * x ≥ x - 3) → (∀ x : ℝ, f(x) = a * x) → (f(x) = -x ∨ f(x) = x) :=
begin
  sorry
end

end direct_proportional_functions_l481_481681


namespace find_x_l481_481420

theorem find_x (AB DC : ClassName) (ACE : is_straight_line ACE)
  (angle_ABC angle_ACD angle_ADC : ℝ)
  (h_par : parallel AB DC) 
  (h_straight : ACE) 
  (h_ABC : angle_ABC = 85) 
  (h_ACD : angle_ACD = 105) 
  (h_ADC : angle_ADC = 125) : 
  let ACB := 180 - angle_ACD
  let BAC := 180 - angle_ABC - ACB 
  let DAC := 180 - angle_ADC - BAC
  x = DAC :=
by 
  sorry

end find_x_l481_481420


namespace women_current_in_room_l481_481452

-- Definitions
variable (m w : ℕ) -- number of men and women
variable (x : ℕ) -- positive integer representing the scaling factor for initial ratio

-- Conditions
def initial_ratio (x : ℕ) : Prop := m = 4*x ∧ w = 5*x
def after_changes (x : ℕ) : Prop := (m + 2) = 14 ∧ (2 * (w - 3)) = 24

-- Theorem statement
theorem women_current_in_room (x : ℕ) (m w : ℕ) (h1 : initial_ratio x) (h2 : after_changes x) : w = 24 :=
by
  sorry

end women_current_in_room_l481_481452


namespace functions_satisfy_inverse_negative_l481_481331

def invert_neg_transform (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = -f x

def f1 (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

def f2 (x : ℝ) : ℝ := (1 - x^2) / (1 + x^2)

def f3 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x
  else if x = 1 then 0
  else -1/x

theorem functions_satisfy_inverse_negative :
  (invert_neg_transform f2) ∧ (invert_neg_transform f3) :=
by
  sorry

end functions_satisfy_inverse_negative_l481_481331


namespace train_length_correct_l481_481275

def length_of_train (time : ℝ) (speed_train_km_hr : ℝ) (speed_man_km_hr : ℝ) : ℝ :=
  let relative_speed_km_hr := speed_train_km_hr - speed_man_km_hr
  let relative_speed_m_s := relative_speed_km_hr * (5 / 18)
  relative_speed_m_s * time

theorem train_length_correct :
  length_of_train 23.998 63 3 = 1199.9 := 
by
  sorry

end train_length_correct_l481_481275


namespace probability_of_rolling_less_than_5_l481_481222

theorem probability_of_rolling_less_than_5 :
  (let outcomes := 8 in
   let successes := 4 in
   let probability := (successes: ℚ) / outcomes in
   probability = 1 / 2) :=
by
  sorry

end probability_of_rolling_less_than_5_l481_481222


namespace probability_of_rolling_less_than_five_l481_481204

/-- Rolling an 8-sided die -/
def outcomes := { 1, 2, 3, 4, 5, 6, 7, 8 }

/-- Successful outcomes (numbers less than 5) -/
def successful_outcomes := { 1, 2, 3, 4 }

/-- Number of total outcomes -/
def total_outcomes := 8

/-- Number of successful outcomes -/
def num_successful_outcomes := 4

/-- Probability of rolling a number less than 5 on an 8-sided die -/
def probability : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_rolling_less_than_five : probability = 1 / 2 := by
  sorry

end probability_of_rolling_less_than_five_l481_481204


namespace pages_per_book_proof_l481_481130

-- Definitions based on conditions
def last_week_books := 5
def pages_per_book := (P : ℕ) -- Unknown number of pages per book
def this_week_pages := 4500
def total_pages_last_week := last_week_books * pages_per_book
def total_pages_this_week := 2 * total_pages_last_week

-- Theorem stating the relationship asked in the problem
theorem pages_per_book_proof (P : ℕ) (h : total_pages_this_week = this_week_pages) : pages_per_book = 450 :=
by obta sorry

end pages_per_book_proof_l481_481130


namespace crow_eats_fifth_time_l481_481609

-- Defining the rates and fractions
def crow_eats_quarter (total_nuts : ℝ) : Real := 1/4
def crow_time_to_eat_quarter : Real := 5

-- translation to a mathematically equivalent statement
theorem crow_eats_fifth_time (total_nuts : ℝ) (crow_eats_quarter : total_nuts = 1/4) (crow_time_to_eat_quarter : Real := 5) : Real :=
  let rate := (1/4) / crow_time_to_eat_quarter
  let time := (1 / 5) / rate
  time = 4 := sorry

end crow_eats_fifth_time_l481_481609


namespace polynomial_divisibility_by_6_l481_481458

theorem polynomial_divisibility_by_6 (a b c : ℤ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 :=
sorry

end polynomial_divisibility_by_6_l481_481458


namespace ratio_of_areas_l481_481897

-- Define the conditions
variable (s : ℝ) (h_pos : s > 0)
-- The total perimeter of four small square pens is reused for one large square pen
def total_fencing_length := 16 * s
def large_square_side_length := 4 * s

-- Define the areas
def small_squares_total_area := 4 * s^2
def large_square_area := (4 * s)^2

-- The statement to prove
theorem ratio_of_areas : small_squares_total_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l481_481897


namespace monic_poly_7_r_8_l481_481487

theorem monic_poly_7_r_8 :
  ∃ (r : ℕ → ℕ), (r 1 = 1) ∧ (r 2 = 2) ∧ (r 3 = 3) ∧ (r 4 = 4) ∧ (r 5 = 5) ∧ (r 6 = 6) ∧ (r 7 = 7) ∧ (∀ (n : ℕ), 8 < n → r n = n) ∧ r 8 = 5048 :=
sorry

end monic_poly_7_r_8_l481_481487


namespace find_m_of_perpendicular_unit_vectors_l481_481497

variable {α : Type} [InnerProductSpace ℝ α]

theorem find_m_of_perpendicular_unit_vectors
  (a b : α)
  (h₁ : ∥a∥ = 1)
  (h₂ : ∥b∥ = 1)
  (h₃ : ⟪a, b⟫ = 0)  -- mutual perpendicularity
  (h₄ : ∥a + 3 • b∥ = m * ∥a - b∥) :
  m = Real.sqrt 5 :=
sorry

end find_m_of_perpendicular_unit_vectors_l481_481497


namespace smallest_set_with_arithmetic_mean_condition_l481_481634

def arithmetic_mean (s : Finset ℝ) : ℝ := (s.sum id) / (s.card : ℝ)

theorem smallest_set_with_arithmetic_mean_condition :
  ∃ (S : Finset ℝ), (S.card = 5) ∧
  (∀ s1 s2 s3 : Finset ℝ, 
    (s1 ⊆ S ∧ s1.card = 2 ∧ 
     s2 ⊆ S ∧ s2.card = 3 ∧ 
     s3 ⊆ S ∧ s3.card = 4 ∧ 
     arithmetic_mean s1 = arithmetic_mean s2 ∧ 
     arithmetic_mean s2 = arithmetic_mean s3)) :=
by {
  -- proof to be provided
  sorry
}

end smallest_set_with_arithmetic_mean_condition_l481_481634


namespace fraction_identity_l481_481390

theorem fraction_identity (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : ((m + n) : ℚ) / n = 10 / 7 := 
sorry

end fraction_identity_l481_481390


namespace product_of_ks_l481_481309

theorem product_of_ks :
  (∏ k in finset.Icc 1 19, k) = (19.factorial) :=
by sorry

end product_of_ks_l481_481309


namespace cone_height_l481_481345

open Real

-- Define the cone with given base radius and lateral surface area
def base_radius := 1
def lateral_surface_area := 2 * π

-- Statement: Given these conditions, the height of the cone is sqrt(3) cm.
theorem cone_height :
  (height : ℝ) (r : ℝ := base_radius) (lateral_area : ℝ := lateral_surface_area)
  (slant_height := lateral_area / (π * r)) :
  height = sqrt (slant_height ^ 2 - r ^ 2) -> height = sqrt 3 :=
by
  sorry

end cone_height_l481_481345


namespace horse_food_calculation_l481_481908

theorem horse_food_calculation
  (num_sheep : ℕ)
  (ratio_sheep_horses : ℕ)
  (total_horse_food : ℕ)
  (H : ℕ)
  (num_sheep_eq : num_sheep = 56)
  (ratio_eq : ratio_sheep_horses = 7)
  (total_food_eq : total_horse_food = 12880)
  (num_horses : H = num_sheep * 1 / ratio_sheep_horses)
  : num_sheep = ratio_sheep_horses → total_horse_food / H = 230 :=
by
  sorry

end horse_food_calculation_l481_481908


namespace find_n_l481_481404

variable (x n : ℝ)

-- Definitions
def positive (x : ℝ) : Prop := x > 0
def equation (x n : ℝ) : Prop := x / n + x / 25 = 0.06 * x

-- Theorem statement
theorem find_n (h1 : positive x) (h2 : equation x n) : n = 50 :=
sorry

end find_n_l481_481404


namespace towels_per_load_l481_481849

-- Defining the given conditions
def total_towels : ℕ := 42
def number_of_loads : ℕ := 6

-- Defining the problem statement: Prove the number of towels per load
theorem towels_per_load : total_towels / number_of_loads = 7 := by 
  sorry

end towels_per_load_l481_481849


namespace soccer_team_games_played_l481_481636

theorem soccer_team_games_played 
  (players : ℕ) (total_goals : ℕ) (third_players_goals_per_game : ℕ → ℕ) (other_players_goals : ℕ) (G : ℕ)
  (h1 : players = 24)
  (h2 : total_goals = 150)
  (h3 : ∃ n, n = players / 3 ∧ ∀ g, third_players_goals_per_game g = n * g)
  (h4 : other_players_goals = 30)
  (h5 : total_goals = third_players_goals_per_game G + other_players_goals) :
  G = 15 := by
  -- Proof would go here
  sorry

end soccer_team_games_played_l481_481636


namespace imaginary_condition_l481_481765

theorem imaginary_condition (m : ℝ) : 
  let Z := (m + 2 * complex.I) / (1 + complex.I) in 
  Z.im ≠ 0 → Z.re = 0 → m = -2 :=
by
  sorry

end imaginary_condition_l481_481765


namespace no_element_divisible_by_4_l481_481304

def sequence : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n+2) := sequence n * sequence (n+1) + 1

theorem no_element_divisible_by_4 :
  ∀ n : ℕ, sequence n % 4 ≠ 0 :=
by
  sorry

end no_element_divisible_by_4_l481_481304


namespace train_length_correct_l481_481276

def length_of_train (time : ℝ) (speed_train_km_hr : ℝ) (speed_man_km_hr : ℝ) : ℝ :=
  let relative_speed_km_hr := speed_train_km_hr - speed_man_km_hr
  let relative_speed_m_s := relative_speed_km_hr * (5 / 18)
  relative_speed_m_s * time

theorem train_length_correct :
  length_of_train 23.998 63 3 = 1199.9 := 
by
  sorry

end train_length_correct_l481_481276


namespace most_likely_outcomes_l481_481686

open BigOperators
open ProbabilityTheory

noncomputable def child_gender_probability : ℚ := (1 : ℚ) / 2

def probability_all_boys (n : ℕ) : ℚ := child_gender_probability ^ n

def probability_all_girls (n : ℕ) : ℚ := child_gender_probability ^ n

def probability_mixed (total girls boys : ℕ) : ℚ :=
  nat.choose total girls * (child_gender_probability ^ total)

theorem most_likely_outcomes 
  (n : ℕ) 
  (h : n = 5) : 
  max (probability_mixed n 3 2) (max (probability_mixed n 2 3) (probability_mixed n 4 1 + probability_mixed n 1 4)) > 
  max (probability_all_boys n) (probability_all_girls n) := by
  sorry

end most_likely_outcomes_l481_481686


namespace sequence_general_term_correct_sqrt_inequality_l481_481244

-- Part (I): Sequence Definition
def sequence (a : ℕ → ℚ) (n : ℕ) : Prop := 
  S n = 2 * n - a n

-- Conjectured General Term
def conjectured_term (a : ℕ → ℚ) (n : ℕ) : Prop := 
  a n = (2^n - 1) / 2^(n-1) 

-- The proof problem for part (I)
theorem sequence_general_term_correct (a : ℕ → ℚ) (n : ℕ) :
  (∀ (n : ℕ), sequence a n) → conjectured_term a n := 
sorry

-- Part (II): Inequality proof
theorem sqrt_inequality :
  sqrt 6 + sqrt 7 > 2 * sqrt 2 + sqrt 5 := 
sorry

end sequence_general_term_correct_sqrt_inequality_l481_481244


namespace intersect_sets_eq_3_l481_481843

variable (a : ℝ)

def A : Set ℝ := {-1, 1, 3}
def B : Set ℝ := {a + 2, a^2 + 4}

theorem intersect_sets_eq_3 (h : A ∩ B = {3}) : a = 1 := by
  -- conditions given
  have hA : A = {-1, 1, 3} := rfl
  have hB : B = {a + 2, a^2 + 4} := rfl
  sorry

end intersect_sets_eq_3_l481_481843


namespace distance_ran_by_Juan_l481_481101

-- Definitions based on the condition
def speed : ℝ := 10 -- in miles per hour
def time : ℝ := 8 -- in hours

-- Theorem statement
theorem distance_ran_by_Juan : speed * time = 80 := by
  sorry

end distance_ran_by_Juan_l481_481101


namespace wine_price_increase_l481_481252

-- Definitions translating the conditions
def wine_cost_today : ℝ := 20.0
def bottles_count : ℕ := 5
def tariff_rate : ℝ := 0.25

-- Statement to prove
theorem wine_price_increase (wine_cost_today : ℝ) (bottles_count : ℕ) (tariff_rate : ℝ) : 
  bottles_count * wine_cost_today * tariff_rate = 25.0 := 
by
  -- Proof is omitted
  sorry

end wine_price_increase_l481_481252


namespace value_of_a_plus_b_l481_481756

theorem value_of_a_plus_b (a b : ℝ) (h1 : abs a = 3) (h2 : b^2 = 25) (h3 : a * b < 0) :
  a + b = 2 ∨ a + b = -2 :=
by
  sorry

end value_of_a_plus_b_l481_481756


namespace kyle_weightlifting_time_l481_481465

theorem kyle_weightlifting_time :
  let total_practice_time := 120
  let time_running_weightlifting := total_practice_time / 2
  let time_weightlifting := 20
  let time_running := 2 * time_weightlifting
  in time_running + time_weightlifting = time_running_weightlifting :=
by
  let total_practice_time := 120
  let time_running_weightlifting := total_practice_time / 2
  let time_weightlifting := 20
  let time_running := 2 * time_weightlifting
  have h1 : time_running + time_weightlifting = time_running_weightlifting,
  { sorry },
  exact h1

end kyle_weightlifting_time_l481_481465


namespace brittany_first_test_grade_l481_481295

theorem brittany_first_test_grade (x : ℤ) (h1 : (x + 84) / 2 = 81) : x = 78 :=
by
  sorry

end brittany_first_test_grade_l481_481295


namespace probability_of_event_E_l481_481210

-- Define the event E as rolling a number less than 5
def event_E (n : ℕ) : Prop := n < 5

-- Define the sample space as the set {1, 2, ..., 8} for an 8-sided die
def sample_space : set ℕ := {n | n ≥ 1 ∧ n ≤ 8}

-- Define the probability function
def probability (E : set ℕ) (S : set ℕ) : ℚ :=
  if S.nonempty then
    E.to_finset.card.to_rat / S.to_finset.card.to_rat
  else 0

-- Prove that the probability of event E is 1/2 given the sample space
theorem probability_of_event_E :
  probability {n | event_E n} sample_space = 1 / 2 := by
  sorry

end probability_of_event_E_l481_481210


namespace point_P_through_graph_l481_481373

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x - 1)

theorem point_P_through_graph (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : 
  f a 1 = 5 :=
by
  unfold f
  sorry

end point_P_through_graph_l481_481373


namespace hyperbola_eccentricity_l481_481323

variables (a b c e : ℝ)
hypothesis (h_c : c^2 = a^2 + b^2)
hypothesis (h_distance : b = 0.5 * (a + c))

theorem hyperbola_eccentricity :
  e = c / a →
  3 * e^2 - 2 * e - 5 = 0 →
  e = 5 / 3 :=
by
  intros h_ecc h_equation
  sorry

end hyperbola_eccentricity_l481_481323


namespace probability_exactly_two_even_dice_l481_481651

theorem probability_exactly_two_even_dice :
  let p_even := 1 / 2
  let p_not_even := 1 / 2
  let number_of_ways := 3
  let probability_each_way := (p_even * p_even * p_not_even)
  3 * probability_each_way = 3 / 8 :=
by
  sorry

end probability_exactly_two_even_dice_l481_481651


namespace median_a_classA_xiaolu_in_classA_l481_481961

open List

def classA_scores : List ℝ := 
  [70, 70, 71, 72, 73, 74, 75, 75, 76, 76, 77, 78]
def total_classA_students : ℕ := 40
def total_classB_students : ℕ := 40

noncomputable def median (l : List ℝ) : ℝ := 
  let sorted_l := sort l
  let n := length sorted_l 
  if n % 2 = 0 then 
    (sorted_l.get (n / 2 - 1) + sorted_l.get (n / 2)) / 2
  else 
    sorted_l.get (n / 2)

def median_classA := (72 + 73) / 2

theorem median_a_classA : median_classA = 72.5 :=
by norm_num

def xiaolu_score : ℝ := 73
def median_classA_score : ℝ := 72.5
def median_classB_score : ℝ := 74

theorem xiaolu_in_classA : xiaolu_score > median_classA_score ∧ xiaolu_score < median_classB_score :=
by norm_num

#check median_a_classA
#check xiaolu_in_classA

end median_a_classA_xiaolu_in_classA_l481_481961


namespace birds_left_in_tree_l481_481247

-- Define the initial number of birds in the tree
def initialBirds : ℝ := 42.5

-- Define the number of birds that flew away
def birdsFlewAway : ℝ := 27.3

-- Theorem statement: Prove the number of birds left in the tree
theorem birds_left_in_tree : initialBirds - birdsFlewAway = 15.2 :=
by 
  sorry

end birds_left_in_tree_l481_481247


namespace geometric_mean_focus_minor_axis_l481_481837

variables (B1 B2 F1 : Point) (O : Point)
variables (c b : ℝ)

-- Conditions
def is_on_minor_axis (b1 b2 o : Point) : Prop :=
  ellipse.minor_axis_endpoints o b1 b2

def distance_from_center_to_focus (o f1 : Point) (c : ℝ) : Prop :=
  dist o f1 = c

def distance_minor_axis (b1 b2 : Point) (b : ℝ) : Prop :=
  dist b1 b2 = 2 * b

def distance_f1b2 (f1 b2 : Point) (d : ℝ) : Prop :=
  dist f1 b2 = d

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

-- Proof of the provided relationship
theorem geometric_mean_focus_minor_axis 
  (h_minor_axis : is_on_minor_axis B1 B2 O)
  (h_center_focus : distance_from_center_to_focus O F1 c)
  (h_minor_axis_length : distance_minor_axis B1 B2 b)
  (h_geometric_mean : ∃ d, d = geometric_mean c (2 * b)) :
  ∃ d, distance_f1b2 F1 B2 d :=
  sorry

end geometric_mean_focus_minor_axis_l481_481837


namespace cross_section_area_of_pyramid_l481_481535

-- Define the structures and conditions
structure Pyramid :=
  (a : ℝ)    -- Lengths of the edges
  (SA_perp_ABC : Prop) -- SA is perpendicular to plane ABC
  (AB_perp_BC : Prop)  -- AB perpendicular to BC
  (midpoint_M : Prop)  -- M is the midpoint of AB
  (cross_section_perpendicular : Prop) -- Cross-section is perpendicular to SC

-- Define the required properties
def conditions (p : Pyramid) : Prop := 
  p.SA_perp_ABC ∧ p.AB_perp_BC ∧ p.midpoint_M ∧ p.cross_section_perpendicular

-- The goal theorem
theorem cross_section_area_of_pyramid (p : Pyramid) (h : conditions p) : 
  (area_of_quadrilateral p = (a ^ 2 * sqrt 3) / 8) :=
sorry

end cross_section_area_of_pyramid_l481_481535


namespace find_k_and_m_l481_481369

theorem find_k_and_m (k m : ℝ) :
  (|k| - 3) = 0 ∧ (∀ x : ℝ, 3 * x = 4 - 5 * x ↔ (|k| - 3) * x^2 - (k - 3) * x + 2 * m + 1 = 0) →
  k = -3 ∧ m = -2 := by
  sorry

end find_k_and_m_l481_481369


namespace strips_cover_circle_proof_l481_481260

noncomputable def strips_cover_circle (strips : ℕ → ℝ) (circle_radius : ℝ) : Prop :=
  (∑ i, strips i = 100) ∧ (circle_radius = 1) → 
  ∃ translations : ℕ → ℝ × ℝ, 
    (∀ i, strips i > 0) ∧ ∀ point ∈ circle (0, 0) 1, ∃ i, point ∈ strip (translations i) (strips i)

theorem strips_cover_circle_proof (strips : ℕ → ℝ) (circle_radius : ℝ) :
  strips_cover_circle strips circle_radius :=
sorry

end strips_cover_circle_proof_l481_481260


namespace pie_machine_completion_time_l481_481627

def time := ℕ -- Simple representation of time in minutes from midnight.

def start_time : time := 9 * 60   -- 9:00 AM in minutes from midnight
def half_work_time : time := 3 * 60 + 30  -- 3 hours and 30 minutes in minutes
def break_time : time := 45  -- 45 minutes
def full_work_time : time := 2 * half_work_time -- full work time without break

def to_min (h : ℕ) (m : ℕ) : time := h * 60 + m

theorem pie_machine_completion_time :
  start_time + full_work_time + break_time = to_min 16 45 :=
by
  sorry

end pie_machine_completion_time_l481_481627


namespace cube_split_includes_2015_l481_481330

theorem cube_split_includes_2015 (m : ℕ) (h1 : m > 1) (h2 : ∃ (k : ℕ), 2 * k + 1 = 2015) : m = 45 :=
by
  sorry

end cube_split_includes_2015_l481_481330


namespace binomial_coefficient_x2_l481_481793

theorem binomial_coefficient_x2 : 
  (polynomial.coeff (polynomial.expand ℝ 10 (x + 1/x)^10) 2) = 210 :=
by
  -- We state the problem    
  sorry

end binomial_coefficient_x2_l481_481793


namespace probability_of_event_E_l481_481209

-- Define the event E as rolling a number less than 5
def event_E (n : ℕ) : Prop := n < 5

-- Define the sample space as the set {1, 2, ..., 8} for an 8-sided die
def sample_space : set ℕ := {n | n ≥ 1 ∧ n ≤ 8}

-- Define the probability function
def probability (E : set ℕ) (S : set ℕ) : ℚ :=
  if S.nonempty then
    E.to_finset.card.to_rat / S.to_finset.card.to_rat
  else 0

-- Prove that the probability of event E is 1/2 given the sample space
theorem probability_of_event_E :
  probability {n | event_E n} sample_space = 1 / 2 := by
  sorry

end probability_of_event_E_l481_481209


namespace distinguishable_balls_boxes_l481_481388

theorem distinguishable_balls_boxes : (3^6 = 729) :=
by {
  sorry
}

end distinguishable_balls_boxes_l481_481388


namespace cone_area_ratio_l481_481067

noncomputable def ratio_of_areas (r l : ℝ) (h_l : l = 2 * r) : ℝ :=
  let base_area := Mathlib.pi * r^2
  let lateral_surface_area := (1 / 2) * Mathlib.pi * l^2
  base_area / lateral_surface_area

theorem cone_area_ratio (r l : ℝ) (h_l : l = 2 * r) :
  ratio_of_areas r l h_l = 1 / 2 := by
  -- Given condition: l = 2 * r
  sorry

end cone_area_ratio_l481_481067


namespace tom_catches_up_in_60_minutes_l481_481848

-- Definitions of the speeds and initial distance
def lucy_speed : ℝ := 4  -- Lucy's speed in miles per hour
def tom_speed : ℝ := 6   -- Tom's speed in miles per hour
def initial_distance : ℝ := 2  -- Initial distance between Tom and Lucy in miles

-- Conclusion that needs to be proved
theorem tom_catches_up_in_60_minutes :
  (initial_distance / (tom_speed - lucy_speed)) * 60 = 60 :=
by
  sorry

end tom_catches_up_in_60_minutes_l481_481848


namespace solve_system_l481_481883

theorem solve_system :
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (14.996, 19.994)) ∨
  (∃ x y : ℝ, 4 * x - 3 * y = -3 ∧ 8 * x + 5 * y = 11 + x ^ 2 ∧ (x, y) = (0.421, 1.561)) :=
  sorry

end solve_system_l481_481883


namespace mikey_initial_leaves_l481_481502

axiom leaves_left : ℕ
axiom leaves_blew_away : ℕ

def initial_leaves := leaves_left + leaves_blew_away

axiom h_leaves_left : leaves_left = 112
axiom h_leaves_blew_away : leaves_blew_away = 244

theorem mikey_initial_leaves : initial_leaves = 356 :=
by
  rw [initial_leaves, h_leaves_left, h_leaves_blew_away]
  norm_num

end mikey_initial_leaves_l481_481502


namespace rate_of_discount_correct_l481_481606

def marked_price : ℤ := 125
def selling_price : ℤ := 120

def discount (mp sp : ℤ) : ℤ := mp - sp
def rate_of_discount (d mp : ℤ) : ℚ := (d.to_rat / mp.to_rat) * 100

theorem rate_of_discount_correct (mp sp : ℤ) (hmp : mp = 125) (hsp : sp = 120) :
  rate_of_discount (discount mp sp) mp = 4 := by
  sorry

end rate_of_discount_correct_l481_481606


namespace min_value_of_f_on_interval_l481_481372

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * (x - 1)

theorem min_value_of_f_on_interval (a : ℝ) :
  (a ≤ 1 → ∃ x ∈ Set.Icc 1 Real.exp 1, f a x = 0) ∧
  (1 < a ∧ a < 2 → ∃ x ∈ Set.Icc 1 Real.exp 1, f a x = a - Real.exp (a - 1)) ∧
  (a ≥ 2 → ∃ x ∈ Set.Icc 1 Real.exp 1, f a x = a + Real.exp 1 - a * Real.exp 1) :=
by
  sorry

end min_value_of_f_on_interval_l481_481372


namespace total_fires_l481_481986

theorem total_fires (Doug_fires Kai_fires Eli_fires : ℕ)
  (h1 : Doug_fires = 20)
  (h2 : Kai_fires = 3 * Doug_fires)
  (h3 : Eli_fires = Kai_fires / 2) :
  Doug_fires + Kai_fires + Eli_fires = 110 :=
by
  sorry

end total_fires_l481_481986


namespace divisor_of_4k2_minus_1_squared_iff_even_l481_481486

-- Define the conditions
variable (k : ℕ) (h_pos : 0 < k)

-- Define the theorem
theorem divisor_of_4k2_minus_1_squared_iff_even :
  ∃ n : ℕ, (8 * k * n - 1) ∣ (4 * k ^ 2 - 1) ^ 2 ↔ Even k :=
by { sorry }

end divisor_of_4k2_minus_1_squared_iff_even_l481_481486


namespace collinear_vectors_x_is_sqrt_2_l481_481743

noncomputable def collinear_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem collinear_vectors_x_is_sqrt_2 :
  let a := (1 : ℝ, real.sqrt (1 + real.sin (40 * real.pi / 180)))
  let b := (1 / real.sin (65 * real.pi / 180), x)
  collinear_vectors a b → x = real.sqrt 2 :=
by
  intro h
  -- Further steps would involve solving the proportional equations to show x = sqrt(2)
  sorry

end collinear_vectors_x_is_sqrt_2_l481_481743


namespace student_passes_test_l481_481414

noncomputable def probability_passing_test : ℝ :=
  let p := 0.6 in
  let q := 0.4 in
  let C_n_k (n k : ℕ) : ℕ := Nat.choose n k in
  (C_n_k 3 2 * p^2 * q) + (C_n_k 3 3 * p^3)

theorem student_passes_test :
  probability_passing_test = 81 / 125 :=
by 
  sorry

end student_passes_test_l481_481414


namespace circle_range_of_t_max_radius_t_value_l481_481368

open Real

theorem circle_range_of_t {t x y : ℝ} :
  (x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16*t^4 + 9 = 0) →
  (- (1:ℝ)/7 < t ∧ t < 1) :=
by
  sorry

theorem max_radius_t_value {t x y : ℝ} :
  (x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16*t^4 + 9 = 0) →
  (- (1:ℝ)/7 < t ∧ t < 1) →
  (∃ r, r^2 = -7*t^2 + 6*t + 1) →
  t = 3 / 7 :=
by
  sorry

end circle_range_of_t_max_radius_t_value_l481_481368


namespace sum_of_all_x_l481_481580

theorem sum_of_all_x (x1 x2 : ℝ) (h1 : (x1 + 5)^2 = 81) (h2 : (x2 + 5)^2 = 81) : x1 + x2 = -10 :=
by
  sorry

end sum_of_all_x_l481_481580


namespace a6_eq_11_l481_481088

variable (a : ℕ → ℤ)

def a1 := (a 1 = 1)
def arithmetic_seq := ∀ n : ℕ, a (n + 1) - a n = 2

theorem a6_eq_11 (h1 : a1) (h2 : arithmetic_seq) : a 6 = 11 := sorry

end a6_eq_11_l481_481088


namespace hyperbola_pf_product_l481_481361

theorem hyperbola_pf_product :
  ∃ M P F₁ F₂ : Point ℝ,
  (is_hyperbola M F₁ F₂) ∧
  (F₁.x = -4) ∧
  (F₁.y = 0) ∧
  (Foci_on_x_axis M) ∧
  (asymptote M (λ x y, (√(7:ℝ))*x + 3*y = 0)) ∧
  (on_hyperbola P M) ∧
  (dot_product (vector PF₁) (vector PF₂) = 0) ∧
  (directrix_through_focus (parabola y^2 = 16 * x) F₁) →
  (vector_length PF₁) * (vector_length PF₂) = 14 :=
by sorry

end hyperbola_pf_product_l481_481361


namespace sum_equivalence_l481_481010
open BigOperators

noncomputable def sum_formula (m n: ℕ) : ℕ :=
∑ k in finset.range n, (m + k) * (m + k + 1)

theorem sum_equivalence (m n: ℕ) : sum_formula m n
  = n * (3 * m^2 + 3 * m * n + n^2 - 1) / 3 := by
  sorry

end sum_equivalence_l481_481010


namespace f_sum_positive_l481_481006

variable {α : Type*} [LinearOrderedField α]

def f (x : α) : α := x ^ 3 + x

theorem f_sum_positive (a b c : α) (h1 : a + b > 0) (h2 : a + c > 0) (h3 : b + c > 0) : 
  f(a) + f(b) + f(c) > 0 := by
sorry

end f_sum_positive_l481_481006


namespace find_a_l481_481491

theorem find_a (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_b : b = 1)
    (h_ab_ccb : (10 * a + b)^2 = 100 * c + 10 * c + b) (h_ccb_gt_300 : 100 * c + 10 * c + b > 300) :
    a = 2 :=
sorry

end find_a_l481_481491


namespace sin_cot_identity_l481_481513

theorem sin_cot_identity (n : ℕ) (h₀ : n ≠ 0) (x : ℝ) (h₁ : ∀ k : ℕ, k ≤ n → x ≠ (n * Real.pi) / (2 * k)) :
  (∑ i in (Finset.range n).filter (λ k, k ≠ 0), (1 / Real.sin ((2^i : ℕ) * x))) = (Real.cot x) - (Real.cot (2^n * x)) :=
sorry

end sin_cot_identity_l481_481513


namespace mutually_exclusive_but_not_complementary_l481_481731

def event_M (s : Set Seed) : Prop := ∀ seed ∈ s, germinate seed
def event_N (s : Set Seed) : Prop := ∀ seed ∈ s, ¬ germinate seed

theorem mutually_exclusive_but_not_complementary (s : Set Seed) :
  (event_M s ∩ event_N s = ∅) ∧ (event_M s ∪ event_N s ≠ univ) := by
  sorry

end mutually_exclusive_but_not_complementary_l481_481731


namespace complex_numbers_count_with_nonzero_imaginary_part_l481_481337

theorem complex_numbers_count_with_nonzero_imaginary_part :
  let s := {0, 1, 2, 3, 4, 5}
  in (∑ b in s, ∑ a in (s.diff {b}), if b = 0 then 0 else 1) = 25 :=
by
  sorry

end complex_numbers_count_with_nonzero_imaginary_part_l481_481337


namespace correct_sequence_l481_481855

def step1 := "Collect the admission ticket"
def step2 := "Register"
def step3 := "Written and computer-based tests"
def step4 := "Photography"

theorem correct_sequence : [step2, step4, step1, step3] = ["Register", "Photography", "Collect the admission ticket", "Written and computer-based tests"] :=
by
  sorry

end correct_sequence_l481_481855


namespace twelfth_even_multiple_of_4_l481_481581

theorem twelfth_even_multiple_of_4 :
  ∃ n : ℕ, n > 0 ∧ (n % 2 = 0) ∧ (n % 4 = 0) ∧ (n = 4 * 12) :=
by
  use 48
  split
  · exact Nat.zero_lt_succ 47
  split
  · exact rfl
  split
  · exact rfl
  sorry

end twelfth_even_multiple_of_4_l481_481581


namespace part_one_part_two_l481_481695

theorem part_one (α : Real) 
  (h1 : (sin (π / 2 - α) + sin (-π - α)) / (3 * cos (2 * π + α) + cos (3 * π / 2 - α)) = 3) :
  (sin α - 3 * cos α) / (sin α + cos α) = -1 / 3 :=
  sorry

theorem part_two (a : Real) 
  (h2 : ∀ (y : Real), y = tan (α) * x → 0 = abs (2 * a) / sqrt (tan α * tan α + 1) - 2 * sqrt 5)
  (h3 :  height 8) : 
  ∃ (r : Real), ∃ (k : Real), (x - a)^2 + y^2 = 6 = 36 :=
  sorry

end part_one_part_two_l481_481695


namespace base_equivalence_l481_481903

theorem base_equivalence :
  let n_7 := 4 * 7 + 3  -- 43 in base 7 expressed in base 10.
  ∃ d : ℕ, (3 * d + 4 = n_7) → d = 9 :=
by
  let n_7 := 31
  sorry

end base_equivalence_l481_481903


namespace sara_oranges_l481_481464

-- Conditions
def joan_oranges : Nat := 37
def total_oranges : Nat := 47

-- Mathematically equivalent proof problem: Prove that the number of oranges picked by Sara is 10
theorem sara_oranges : total_oranges - joan_oranges = 10 :=
by
  sorry

end sara_oranges_l481_481464


namespace count_valid_subsets_l481_481484

def setS : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def valid_subsets (T : Set ℕ) : Prop :=
  ∀ x ∈ T, (2 * x ∈ setS) → (2 * x ∈ T)

theorem count_valid_subsets : 
  {T : Set ℕ | valid_subsets T}.toFinset.card = 180 := 
sorry

end count_valid_subsets_l481_481484


namespace find_x_l481_481533

theorem find_x (x : ℝ) :
  (1 / 3) * ((3 * x + 4) + (7 * x - 5) + (4 * x + 9)) = (5 * x - 3) → x = 17 :=
by
  sorry

end find_x_l481_481533


namespace sequence_linear_constant_l481_481246

open Nat

theorem sequence_linear_constant (a : ℕ → ℕ) 
  (h1 : ∀ n, 1 < a 1 ∧ a (n + 1) > a n)
  (h2 : ∀ n, a (n + a n) = 2 * a n) :
  ∃ c : ℕ, ∀ n, a n = n + c := 
sorry

end sequence_linear_constant_l481_481246


namespace chocolate_bar_count_l481_481613

theorem chocolate_bar_count (bar_weight : ℕ) (box_weight : ℕ) (H1 : bar_weight = 125) (H2 : box_weight = 2000) : box_weight / bar_weight = 16 :=
by
  sorry

end chocolate_bar_count_l481_481613


namespace prob_exactly_one_absent_correct_l481_481776

def prob_absent := 1 / 20
def prob_present := 1 - prob_absent

def prob_exactly_one_absent := 
  (prob_absent * prob_present) + 
  (prob_present * prob_absent)

theorem prob_exactly_one_absent_correct : 
  (prob_exactly_one_absent * 100) = 9.5 := 
  by
    sorry

end prob_exactly_one_absent_correct_l481_481776


namespace tan_pi_minus_alpha_l481_481059

/-- Given the conditions: tan(α + π / 4) = sin(2α) + cos^2(α) and α ∈ (π / 2, π),
    then prove tan(π - α) = 3 -/
theorem tan_pi_minus_alpha (α : ℝ) 
  (h1 : tan (α + π / 4) = sin (2 * α) + cos α ^ 2)
  (h2 : α ∈ (π / 2, π)) :
  tan (π - α) = 3 :=
sorry

end tan_pi_minus_alpha_l481_481059


namespace seven_b_value_l481_481752

theorem seven_b_value (a b : ℚ) (h₁ : 8 * a + 3 * b = 0) (h₂ : a = b - 3) :
  7 * b = 168 / 11 :=
sorry

end seven_b_value_l481_481752


namespace scientific_notation_of_0_00000012_l481_481528

theorem scientific_notation_of_0_00000012 :
  0.00000012 = 1.2 * 10 ^ (-7) :=
by
  sorry

end scientific_notation_of_0_00000012_l481_481528


namespace slope_of_horizontal_line_l481_481173

theorem slope_of_horizontal_line (c : ℝ) : slope (line_eqn_y c) = 0 :=
sorry

end slope_of_horizontal_line_l481_481173


namespace root_zero_implies_m_neg3_l481_481175

theorem root_zero_implies_m_neg3 (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + (3m - 1) * x + m^2 - 9 = 0) →
  (∃ x : ℝ, x = 0) →
  m = -3 :=
sorry

end root_zero_implies_m_neg3_l481_481175


namespace find_a_n_l481_481706

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_sum : ∀ n : ℕ, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n : ℕ, a n = Real.sqrt ↑n - Real.sqrt (↑n - 1) :=
by
  sorry

end find_a_n_l481_481706


namespace water_flow_speed_l481_481626

/-- A person rows a boat for 15 li. If he rows at his usual speed,
the time taken to row downstream is 5 hours less than rowing upstream.
If he rows at twice his usual speed, the time taken to row downstream
is only 1 hour less than rowing upstream. 
Prove that the speed of the water flow is 2 li/hour.
-/
theorem water_flow_speed (y x : ℝ)
  (h1 : 15 / (y - x) - 15 / (y + x) = 5)
  (h2 : 15 / (2 * y - x) - 15 / (2 * y + x) = 1) :
  x = 2 := 
sorry

end water_flow_speed_l481_481626


namespace cot_subtraction_l481_481329

open Real

theorem cot_subtraction (x : ℝ) (h : x ≠ 0 ∧ x ≠ π ∧ x ≠ 3 * π) :
  cot (x / 3) - cot x = (sin ((2 : ℝ) * x / 3)) / (sin (x / 3) * sin x) :=
by
  sorry

end cot_subtraction_l481_481329


namespace Joan_balloons_l481_481807

theorem Joan_balloons: (original_balloons lost_balloons remaining_balloons : ℕ) 
  (h1 : original_balloons = 8) (h2 : lost_balloons = 2) 
  (h3 : remaining_balloons = original_balloons - lost_balloons): remaining_balloons = 6 :=
by
  sorry

end Joan_balloons_l481_481807


namespace train_passes_jogger_in_36_seconds_l481_481934

def jogger_speed_kmph : ℝ := 9
def jogger_speed_mps : ℝ := jogger_speed_kmph * (1000 / 3600)

def train_speed_kmph : ℝ := 45
def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

def initial_distance : ℝ := 240
def train_length : ℝ := 120

def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
def total_distance_to_cover : ℝ := initial_distance + train_length

def time_to_pass_jogger : ℝ := total_distance_to_cover / relative_speed_mps

theorem train_passes_jogger_in_36_seconds : time_to_pass_jogger = 36 :=
by
  sorry

end train_passes_jogger_in_36_seconds_l481_481934


namespace books_arrangement_l481_481751

/-
  Theorem:
  If there are 4 distinct math books, 6 distinct English books, and 3 distinct science books,
  and each category of books must stay together, then the number of ways to arrange
  these books on a shelf is 622080.
-/

def num_math_books : ℕ := 4
def num_english_books : ℕ := 6
def num_science_books : ℕ := 3

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_arrangements :=
  factorial 3 * factorial num_math_books * factorial num_english_books * factorial num_science_books

theorem books_arrangement : num_arrangements = 622080 := by
  sorry

end books_arrangement_l481_481751


namespace exponent_simplification_l481_481241

theorem exponent_simplification (k : ℤ) :
  2^(-(2*k+1)) - 2^(-(2*k-1)) + 2^(-2*k) = -2^(-(2*k+1)) :=
by
  sorry

end exponent_simplification_l481_481241


namespace conjugate_of_complex_number_l481_481365

theorem conjugate_of_complex_number (i : ℂ) (h_i : i = Complex.I) :
  let z := 2 / (1 - i)
  Complex.conj z = 1 - i :=
by
  sorry

end conjugate_of_complex_number_l481_481365


namespace general_term_An_sum_terms_inv_b_prod_l481_481364

-- Definitions based on problem conditions
def S (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of the sequence {a_n}

def a : ℕ → ℝ
| 0 => 3 -- Generally sequences are 0-indexed in Lean
| (n+1) => 2 * S (n+1) + 3 / 2

-- Sequence {b_n} as per the given conditions
def b (n : ℕ) : ℝ := 3 * (2 * n - 1)

-- Sum of the first n terms of {b_n}
def T : ℕ → ℝ
| 0 => 3
| 2 => 2 * b 2  -- Assuming T is based on the sum of arithmetic sequence

-- Sequence {1/(b_n*b_{n+1})}
def inv_b_prod (n : ℕ) : ℝ := (1 : ℝ) / ((b n) * (b (n + 1)))

-- Sum of the first n terms of the sequence {1/(b_n*b_{n+1})}
def Q (n : ℕ) : ℝ := (1 / 18) * (1 - (1 / (2 * n + 1)))

-- Theorem to prove the general term formula of {a_n}
theorem general_term_An (n : ℕ) : a n = 3^(n+1) :=
by
  sorry

-- Theorem to prove the sum of the first n terms of {1/(b_n*b_{n+1})}
theorem sum_terms_inv_b_prod (n : ℕ) : 
  ∑ i in range n, inv_b_prod i = Q n :=
by
  sorry

end general_term_An_sum_terms_inv_b_prod_l481_481364


namespace a5_b3_c_divisible_by_6_l481_481460

theorem a5_b3_c_divisible_by_6 (a b c : ℤ) (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) :=
by
  sorry

end a5_b3_c_divisible_by_6_l481_481460


namespace no_trisecting_point_exists_l481_481802

theorem no_trisecting_point_exists (ABC : Triangle)
  (P : Point)
  (h1 : is_inside_triangle P ABC)
  (h2 : ∠ (P, A, B) = (1/3) * ∠ (A, B, C))
  (h3 : ∠ (P, B, C) = (1/3) * ∠ (B, C, A))
  (h4 : ∠ (P, C, A) = (1/3) * ∠ (C, A, B)) : 
  False :=
sorry

end no_trisecting_point_exists_l481_481802


namespace terminal_side_in_third_quadrant_l481_481753

-- Define the conditions
def sin_condition (α : Real) : Prop := Real.sin α < 0
def tan_condition (α : Real) : Prop := Real.tan α > 0

-- State the theorem
theorem terminal_side_in_third_quadrant (α : Real) (h1 : sin_condition α) (h2 : tan_condition α) : α ∈ Set.Ioo (π / 2) π :=
  sorry

end terminal_side_in_third_quadrant_l481_481753


namespace LCM_meeting_time_l481_481990

theorem LCM_meeting_time (b c d : ℕ) (start_time : ℕ) : 
  (b = 5) → (c = 8) → (d = 9) → (start_time = 7) → 
  let lcm_time := Nat.lcm b (Nat.lcm c d) in 
  lcm_time = 360 ∧ (start_time * 60 + lcm_time) / 60 = 13 := 
by
  sorry

end LCM_meeting_time_l481_481990


namespace find_a_square_plus_a_inverse_square_l481_481696

theorem find_a_square_plus_a_inverse_square (a : ℂ) (h : a + a⁻¹ = 3) : a^2 + a⁻² = 7 :=
sorry

end find_a_square_plus_a_inverse_square_l481_481696


namespace compare_xyz_l481_481719

open Real

theorem compare_xyz (x y z : ℝ) : x = Real.log π → y = log 2 / log 5 → z = exp (-1 / 2) → y < z ∧ z < x := by
  intros h_x h_y h_z
  sorry

end compare_xyz_l481_481719


namespace isosceles_triangles_l481_481864

theorem isosceles_triangles (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_triangle : ∀ n : ℕ, (a^n + b^n > c^n ∧ b^n + c^n > a^n ∧ c^n + a^n > b^n)) :
  b = c := 
sorry

end isosceles_triangles_l481_481864


namespace proof_ellipse_tangent_l481_481034

noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
                ∃ (f : ℝ × ℝ), f.1^2 + f.2^2 = a^2 - b^2 ∧ 
                2 * b * (a^2 - b^2) = 2 ∧ a^2 = b^2 + (a^2 - b^2)

theorem proof_ellipse_tangent :
    ∀ a b : ℝ, a > b → b > 0 → 
    (∃ x y : ℝ, (x^2 / 2 + y^2 = 1)) ∧ 
    (∀ P F A₁ A₂ B : ℝ × ℝ, 
        A₁ = (- real.sqrt 2, 0) → 
        A₂ = (real.sqrt 2, 0) → 
        F = (1, 0) → 
        B.1 = real.sqrt 2 → 
        B.2 = 2 * real.sqrt 2 → 
        let M := (real.sqrt 2, real.sqrt 2) 
            r := real.sqrt 2 in 
        d(P, F) = r) :=
  sorry

end proof_ellipse_tangent_l481_481034


namespace john_spent_on_books_l481_481814

variable (earnings_per_day : ℕ)
variable (total_days_april : ℕ)
variable (sundays_april : ℕ)
variable (money_left : ℕ)
variable (x : ℕ)

def total_days_worked : ℕ := total_days_april - sundays_april

def total_earnings : ℕ := total_days_worked * earnings_per_day

def spent_on_books : ℕ := x

def given_to_sister : ℕ := x

theorem john_spent_on_books :
  total_earnings - spent_on_books - given_to_sister = money_left → x = 50 :=
by
  intros h1
  let total_earnings := total_days_worked * earnings_per_day
  have h2 : total_earnings = 260 := by sorry -- Assumed correct calculation (26 * 10)
  have h3 : total_earnings - 2 * x = money_left := by sorry -- Derived from problem statement
  have h4 : total_earnings - 2 * x = 160 := by sorry
  have h5 : 2 * x = 100 := by sorry
  have h6 : x = 50 := by sorry
  exact h6

end john_spent_on_books_l481_481814


namespace area_ratio_l481_481614

variables {ABC : Type*} [triangle ABC] {r R : ℝ} 

noncomputable def area_of_triangle (A B C : ABC) : ℝ := sorry

noncomputable def inradius (A B C : ABC) : ℝ := r
noncomputable def circumradius (A B C : ABC) : ℝ := R

variables {T1 T2 T3 : ABC} (inscribed: ∀ A B C, is_tangent T1 T2 T3)

theorem area_ratio (h : inscribed (inradius ABC) (circumradius ABC)) :
  area_of_triangle T1 T2 T3 / area_of_triangle A B C = r / (2 * R) :=
sorry

end area_ratio_l481_481614


namespace exists_two_integers_with_difference_divisible_by_2022_l481_481296

theorem exists_two_integers_with_difference_divisible_by_2022 (a : Fin 2023 → ℤ) : 
  ∃ i j : Fin 2023, i ≠ j ∧ (a i - a j) % 2022 = 0 := by
  sorry

end exists_two_integers_with_difference_divisible_by_2022_l481_481296


namespace necessary_but_not_sufficient_l481_481476

def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) :=
by sorry

end necessary_but_not_sufficient_l481_481476


namespace factory_daily_earnings_l481_481409

def num_original_machines : ℕ := 3
def original_machine_hours : ℕ := 23
def num_new_machines : ℕ := 1
def new_machine_hours : ℕ := 12
def production_rate : ℕ := 2 -- kg per hour per machine
def price_per_kg : ℕ := 50 -- dollars per kg

theorem factory_daily_earnings :
  let daily_production_original := num_original_machines * original_machine_hours * production_rate,
      daily_production_new := num_new_machines * new_machine_hours * production_rate,
      total_daily_production := daily_production_original + daily_production_new,
      daily_earnings := total_daily_production * price_per_kg
  in
  daily_earnings = 8100 :=
by
  sorry

end factory_daily_earnings_l481_481409


namespace women_count_l481_481435

/-- 
Initially, the men and women in a room were in the ratio of 4:5.
Then, 2 men entered the room and 3 women left the room.
The number of women then doubled.
There are now 14 men in the room.
Prove that the number of women currently in the room is 24.
-/
theorem women_count (x : ℕ) (h1 : 4 * x + 2 = 14) (h2 : 2 * (5 * x - 3) = n) : 
  n = 24 :=
by
  sorry

end women_count_l481_481435


namespace water_leakage_relationship_l481_481230

theorem water_leakage_relationship (t : ℕ) (y : ℕ) :
  (∀ t, y = 5 * t + 2) ↔
  (y 1 = 7 ∧ y 2 = 12) :=
sorry

end water_leakage_relationship_l481_481230


namespace elevator_max_weight_l481_481148

theorem elevator_max_weight :
  let avg_weight_adult := 150
  let num_adults := 7
  let avg_weight_child := 70
  let num_children := 5
  let orig_max_weight := 1500
  let weight_adults := num_adults * avg_weight_adult
  let weight_children := num_children * avg_weight_child
  let current_weight := weight_adults + weight_children
  let upgrade_percentage := 0.10
  let new_max_weight := orig_max_weight * (1 + upgrade_percentage)
  new_max_weight - current_weight = 250 := 
  by
    sorry

end elevator_max_weight_l481_481148


namespace javier_exercise_duration_l481_481096

-- Define the variables and conditions
variables (d : ℕ)
def javier_minutes : ℕ := 50 * d
def sanda_minutes : ℕ := 90 * 3
def total_minutes : ℕ := javier_minutes d + sanda_minutes

-- The main statement
theorem javier_exercise_duration : total_minutes d = 620 → javier_minutes d = 350 :=
begin
  intro h,
  have hd : 50 * d = 350, {
    linarith,
  },
  rw hd,
  refl,
end

end javier_exercise_duration_l481_481096


namespace find_proj_b_l481_481469

variable (a b v : ℝ×ℝ)

def orthogonal (a b : ℝ×ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

def proj (u v : ℝ×ℝ) [noncomputable] : ℝ×ℝ := 
  let scalar := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  in (scalar * v.1, scalar * v.2)

theorem find_proj_b (h_orth : orthogonal a b)
  (h_proj_a : proj v a = (-3/5, -6/5)) :
  proj v b = (18/5, -9/5) :=
sorry

end find_proj_b_l481_481469


namespace perfect_cubes_in_range_l481_481750

theorem perfect_cubes_in_range :
  ∃ (n : ℕ), (∀ (k : ℕ), (50 < k^3 ∧ k^3 ≤ 1000) → (k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10)) ∧
    (∃ m, (m = 7)) :=
by
  sorry

end perfect_cubes_in_range_l481_481750


namespace circle_value_a_l481_481353

noncomputable def represents_circle (a : ℝ) (x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0

theorem circle_value_a {a : ℝ} (h : ∀ x y : ℝ, represents_circle a x y) :
  a = -1 :=
by
  sorry

end circle_value_a_l481_481353


namespace rachel_remaining_amount_l481_481871

-- Definitions of the initial earning, the fraction spent on lunch, and the fraction spent on the DVD.
def initial_amount : ℝ := 200
def fraction_lunch : ℝ := 1/4
def fraction_dvd : ℝ := 1/2

-- Calculation of the remaining amount Rachel has.
theorem rachel_remaining_amount :
  let spent_on_lunch := fraction_lunch * initial_amount in
  let spent_on_dvd := fraction_dvd * initial_amount in
  let remaining_amount := initial_amount - spent_on_lunch - spent_on_dvd in
  remaining_amount = 50 :=
by
  sorry

end rachel_remaining_amount_l481_481871


namespace closest_integer_to_sum_l481_481324

theorem closest_integer_to_sum : 
  let s := 500 * (Finset.sum (Finset.range (3001 - 5)).map (λ n, 1 / ((n + 5)^2 - 9))) in
  Int.closest (Real.toInt s) s = 97 := 
sorry

end closest_integer_to_sum_l481_481324


namespace pythagorean_numbers_set_d_l481_481983

theorem pythagorean_numbers_set_d :
  ∃ (a b c : ℕ), (a, b, c) = (9, 12, 15) ∧ a^2 + b^2 = c^2 :=
by
  use 9, 12, 15
  split
  · rfl
  · sorry

end pythagorean_numbers_set_d_l481_481983


namespace smallest_two_ks_l481_481558

theorem smallest_two_ks (k : ℕ) (h : ℕ → Prop) : 
  (∀ k, (k^2 + 36) % 180 = 0 → k = 12 ∨ k = 18) :=
by {
 sorry
}

end smallest_two_ks_l481_481558


namespace x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one_l481_481240

theorem x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one {x : ℝ} :
  (x > 1 → |x| > 1) ∧ (¬(|x| > 1 → x > 1)) :=
by
  sorry

end x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one_l481_481240


namespace triangle_front_view_solids_l481_481066

-- Define each solid as an inductive type
inductive Solid
  | triangular_pyramid
  | square_pyramid
  | triangular_prism
  | square_prism
  | cone
  | cylinder
  deriving DecidableEq

-- Define a function that checks if a solid can have a triangle as its front view
def has_triangle_front_view (s : Solid) : Prop :=
  match s with
  | Solid.triangular_pyramid => True
  | Solid.square_pyramid     => True
  | Solid.triangular_prism   => True
  | Solid.square_prism       => False
  | Solid.cone               => True
  | Solid.cylinder           => False
  end

-- Prove the specified solids have a triangle as their front view
theorem triangle_front_view_solids :
  ∀ s : Solid, has_triangle_front_view s ↔ s = Solid.triangular_pyramid ∨ s = Solid.square_pyramid ∨ s = Solid.triangular_prism ∨ s = Solid.cone := by
  intro s
  cases s <;> simp [has_triangle_front_view]
  repeat {split}; try {repeat {tauto}}

end triangle_front_view_solids_l481_481066


namespace women_count_l481_481433

/-- 
Initially, the men and women in a room were in the ratio of 4:5.
Then, 2 men entered the room and 3 women left the room.
The number of women then doubled.
There are now 14 men in the room.
Prove that the number of women currently in the room is 24.
-/
theorem women_count (x : ℕ) (h1 : 4 * x + 2 = 14) (h2 : 2 * (5 * x - 3) = n) : 
  n = 24 :=
by
  sorry

end women_count_l481_481433


namespace line_PQ_bisects_segment_AB_l481_481649

theorem line_PQ_bisects_segment_AB
  (A B C D P Q : Point) -- points
  (rA rB rC rD : ℝ) -- radii of circles
  (circleA : Circle A rA) (circleB : Circle B rB) 
  (circleC : Circle C rC) (circleD : Circle D rD) 
  (h_eq_large : circleA.radius = circleB.radius)
  (h_neq_small : circleC.radius ≠ circleD.radius)
  (h_int_large : intersects circleA circleB)
  (h_int_small : intersects_at circleC circleD P Q)
  (h_tangent_C_internal : tangent_internal circleC circleA)
  (h_tangent_C_external : tangent_external circleC circleB)
  (h_tangent_D_internal : tangent_internal circleD circleA)
  (h_tangent_D_external : tangent_external circleD circleB) :
  let midpoint := midpoint A B in
  on_line P Q midpoint :=
  sorry

end line_PQ_bisects_segment_AB_l481_481649


namespace find_a5_l481_481016

/-- Define the initial term of the sequence. -/
def a₁ : ℤ := 60

/-- Define the sequence difference as a geometric sequence -/
def seq_diff (n : ℕ) : ℤ :=
  (-4) * (2 ^ n)

/-- Define the sequence a following the given conditions -/
noncomputable def a (n : ℕ) : ℤ :=
  a₁ + ∑ i in Finset.range n, seq_diff i

/-- The main theorem stating a₅ is 0 -/
theorem find_a5 : a 5 = 0 := 
  by 
  sorry

end find_a5_l481_481016


namespace prime_count_between_80_and_90_l481_481053

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (list.range' a (b - a + 1)).filter is_prime |>.length

theorem prime_count_between_80_and_90 :
  count_primes_in_range 80 90 = 2 :=
by
  sorry

end prime_count_between_80_and_90_l481_481053


namespace find_y_l481_481698

theorem find_y
  (a b c x : ℝ)
  (p q r y : ℝ)
  (log_base : ℝ)
  (h1 : log_base ≠ 1)
  (h2 : log_base > 0)
  (h3 : log_base ≠ 0)
  (h4 : log a / p = log b / q)
  (h5 : log b / q = log c / r)
  (h6 : log c / r = log x)
  (h7 : log x ≠ 0)
  (h8 : b^3 / (a^2 * c) = x^y) :
  y = 3 * q - 2 * p - r := 
sorry

end find_y_l481_481698


namespace nested_radical_floor_l481_481121

theorem nested_radical_floor :
  let y := sqrt(10 + sqrt(10 + sqrt(10 + sqrt(10 + ⋯)))),
      B := ⌊10 + y⌋ in
  (y * y = 10 + y) ∧ (B = 13) :=
sorry

end nested_radical_floor_l481_481121


namespace reciprocal_opposites_l481_481402

theorem reciprocal_opposites (a b : ℝ) (h1 : 1 / a = -8) (h2 : 1 / -b = 8) : a = b :=
sorry

end reciprocal_opposites_l481_481402


namespace simplify_expression_l481_481521

theorem simplify_expression :
  (1 / (1 / ((1 / 3)^1) + 1 / ((1 / 3)^2) + 1 / ((1 / 3)^3))) = 1 / 39 :=
by
  sorry

end simplify_expression_l481_481521


namespace max_product_of_three_nums_l481_481671

theorem max_product_of_three_nums : 
  let S := {-5, -3, -2, 4, 6, 7}
  ∃ a b c ∈ S, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∀ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ x ≠ z → x * y * z ≤ a * b * c ∧ a * b * c = 105 := 
sorry

end max_product_of_three_nums_l481_481671


namespace general_equations_and_intersection_properties_l481_481526

def point_M : ℝ × ℝ := (1, 0)

def line_l_polar (ρ θ : ℝ) : Prop :=
  sqrt 2 * ρ * cos(θ + π/4) - 1 = 0

def curve_C_parametric (t : ℝ) : ℝ × ℝ :=
  (4 * t^2, 4 * t)

theorem general_equations_and_intersection_properties :
  ( ∀ (ρ θ : ℝ), line_l_polar ρ θ ↔ (∃ x y : ℝ, x = ρ * cos θ ∧ y = ρ * sin θ ∧ x - y - 1 = 0) ) ∧
  ( ∀ (t : ℝ), curve_C_parametric t ↔ (∃ x y : ℝ, x = 4 * t^2 ∧ y = 4 * t ∧ y^2 = 4 * x) ) ∧
  ( ∀ (A B : ℝ × ℝ), (∃ t₁ t₂ : ℝ, curve_C_parametric t₁ = A ∧ curve_C_parametric t₂ = B ∧ (∃ x y : ℝ, x = ρ * cos(atan(y/x)) ∧ y = ρ * sin(atan(y/x)) ∧ x - y - 1 = 0)) → (1 / (sqrt((1 - A.1)^2 + (0 - A.2)^2) + 1 / sqrt((1 - B.1)^2 + (0 - B.2)^2)) = 1 )
  :=
sorry

end general_equations_and_intersection_properties_l481_481526


namespace circle_properties_l481_481341

theorem circle_properties
    (radius : ℝ)
    (center : ℝ × ℝ)
    (on_ray : center.2 = -2 * center.1)
    (center_x_neg : center.1 < 0)
    (tangent_line : ∀ x y, (x, y) = center → x + y + 1 = 0)
    (P : ℝ × ℝ)
    (PO_eq_PM : dist P (0, 0) = dist P (center.1, center.2) - radius)
    :
    (∃ x y, (x + 1)^2 + (y - 2)^2 = 2) ∧
    (let PM_min_area := (3 * Real.sqrt 10) / 20 in
     P = (-3 / 10, 3 / 5) ∧
     ∃ M, PM_min_area = (1 / 2) * (dist P M) * Real.sqrt 2) := 
sorry

end circle_properties_l481_481341


namespace part_a_part_b_l481_481186

variables {a b c d e f g h x y z u v : ℝ}

-- Given areas based on the formula for the area of a triangle
def area_u (a b α : ℝ) : ℝ := (1/2) * a * b * Real.sin α
def area_v (c d α : ℝ) : ℝ := (1/2) * c * d * Real.sin α
def area_y (a c α : ℝ) : ℝ := (1/2) * a * c * Real.sin α
def area_z (b d α : ℝ) : ℝ := (1/2) * b * d * Real.sin α

-- Prove part (a)
theorem part_a (a b c d α : ℝ) (u v y z : ℝ)
  (hu : u = area_u a b α) (hv : v = area_v c d α)
  (hy : y = area_y a c α) (hz : z = area_z b d alpha) :
  u * v = y * z := 
sorry

-- Menelaus's theorem condition for part (b)
theorem part_b (a b c d e f g h S x y z : ℝ)
  (hu : u = area_u a b α) (hv : v = area_v c d α)
  (hy : y = area_y a c α) (hz : z = area_z b d α)
  (menelaus : (a / d) * (h / g) * ((e + f) / e) = 1) :
  S ≤ (x * z) / y := 
sorry

end part_a_part_b_l481_481186


namespace automaton_base3_even_correct_automaton_base10_multiple7_correct_l481_481936

-- Definition of automaton for even numbers in base 3
structure AutomatonBase3Even :=
  (states : fin 2)
  (initial: states) -- 0
  (accepting: states) -- 0
  (transition: states → nat → states)
  
-- Definition of automaton for multiples of 7 in base 10
structure AutomatonBase10Multiple7 :=
  (states : fin 7)
  (initial: states) -- 0
  (accepting: states) -- all states (x, 0) where x ranges through the cycle of {1, 3, 2, 6, 4, 5}
  (transition: states → nat → states)

-- Lean 4 statement to prove the correctness of the automaton for even numbers in base 3
theorem automaton_base3_even_correct 
  (A: AutomatonBase3Even)
  (valid_state: A.states ∈ fin 2)
  (correct_initial: A.initial = 0)
  (correct_accepting: A.accepting = 0)
  (correct_transitions:
    A.transition 0 0 = 0 ∧
    A.transition 0 1 = 1 ∧
    A.transition 0 2 = 0 ∧
    A.transition 1 0 = 1 ∧
    A.transition 1 1 = 0 ∧
    A.transition 1 2 = 1)
  (input_digits: list nat)
  (base3: nat)
  (valid_digits: ∀ d, d ∈ input_digits → d < 3) : 
  (∀ s, s ∈ fin 2 → 
    A.transition s nat.digits 3 base3
    nat.digits 3 base3 = input_digits → (nat.digits 3 base3 mod 2 = 0 → s = 0)) :=
sorry

-- Lean 4 statement to prove the correctness of the automaton for multiples of 7 in base 10
theorem automaton_base10_multiple7_correct 
  (A: AutomatonBase10Multiple7)
  (valid_state: A.states ∈ fin 7)
  (correct_initial: A.initial = 0)
  (correct_accepting: A.accepting = 0)
  (correct_transitions: ∀ i digit, i ∈ fin 7 → digit < 10 →
    A.transition i digit = (digit + 10 * i) % 7)
  (input_digits: list nat)
  (base10: nat)
  (valid_digits: ∀ d, d ∈ input_digits → d < 10) : 
  (∀ s, s ∈ fin 7 → 
    A.transition s nat.digits 10 base10
    nat.digits 10 base10 = input_digits → (nat.digits 10 base10 mod 7 = 0 → s = 0)) :=
sorry

end automaton_base3_even_correct_automaton_base10_multiple7_correct_l481_481936


namespace shooting_test_performance_l481_481333

theorem shooting_test_performance (m n : ℝ)
    (h1 : m > 9.7)
    (h2 : n < 0.25) :
    (m = 9.9 ∧ n = 0.2) :=
sorry

end shooting_test_performance_l481_481333


namespace total_cost_harkamal_l481_481047

-- Define all the given conditions as constants
def weight_grapes : ℝ := 10
def rate_grapes : ℝ := 70
def discount_grapes : ℝ := 0.1

def weight_mangoes : ℝ := 9
def rate_mangoes : ℝ := 55

def weight_apples : ℝ := 12
def rate_apples : ℝ := 80
def discount_apples : ℝ := 0.05

def weight_papayas : ℝ := 7
def rate_papayas : ℝ := 45
def discount_papayas : ℝ := 0.15

def weight_oranges : ℝ := 15
def rate_oranges : ℝ := 30

def weight_bananas : ℝ := 5
def rate_bananas : ℝ := 25

-- Prove that the total cost Harkamal paid equals $2879.75
theorem total_cost_harkamal : 
    let cost_grapes := weight_grapes * rate_grapes * (1 - discount_grapes),
        cost_mangoes := weight_mangoes * rate_mangoes,
        cost_apples := weight_apples * rate_apples * (1 - discount_apples),
        cost_papayas := weight_papayas * rate_papayas * (1 - discount_papayas),
        cost_oranges := weight_oranges * rate_oranges,
        cost_bananas := weight_bananas * rate_bananas,
        total_cost := cost_grapes + cost_mangoes + cost_apples + cost_papayas + cost_oranges + cost_bananas
    in total_cost = 2879.75 := 
by
  sorry

end total_cost_harkamal_l481_481047


namespace range_of_a_if_f_is_monotonically_increasing_l481_481401

theorem range_of_a_if_f_is_monotonically_increasing 
    (a : ℝ) 
    (f : ℝ → ℝ) 
    (Hf : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) :
    -1 ≤ a ∧ a ≤ 0 :=
by
  let f : ℝ → ℝ := λ x, x^2 + a * |x - (1/2)|
  have h1 : ∀ x ≥ (1/2), 0 ≤ 2 * x + a := sorry  -- Derivation for x ≥ 1/2
  have h2 : ∀ x < (1/2), 0 ≤ 2 * x - a := sorry  -- Derivation for 0 ≤ x < 1/2
  sorry  -- Conclusion combining h1 and h2
  

end range_of_a_if_f_is_monotonically_increasing_l481_481401


namespace xy_range_l481_481726

theorem xy_range (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
    (h_eqn : x + 3 * y + 2 / x + 4 / y = 10) :
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
  sorry

end xy_range_l481_481726


namespace find_k_l481_481759

theorem find_k (k : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 - 60 * x + k = (x + b)^2) → k = 900 :=
by 
  sorry

end find_k_l481_481759


namespace monotonic_intervals_and_extreme_values_range_of_a_l481_481664

noncomputable def f (x : ℝ) : ℝ := Math.sin x - Math.cos x + x + 1

theorem monotonic_intervals_and_extreme_values : 
  (∀ x, 0 < x /\ x < π → 0 < (1 + Real.sqrt 2 * Math.sin (x + π / 4))) ∧ 
  (∀ x, π < x /\ x < (3 * π / 2) → (1 + Real.sqrt 2 * Math.sin (x + π / 4)) < 0) ∧
  (∀ x, (3 * π / 2) < x /\ x < 2 * π → 0 < (1 + Real.sqrt2 * Math.sin (x + π / 4))) ∧ 
  (f π = π + 2) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x, 0 ≤ x /\ x ≤ π → 0 ≤ Math.cos x + Math.sin x + 1 - a) ↔ a ≤ 0 := sorry

end monotonic_intervals_and_extreme_values_range_of_a_l481_481664


namespace replaced_weight_l481_481891

theorem replaced_weight (avg_increase : ℝ) (new_weight : ℝ) : 
  avg_increase = 3.5 → new_weight = 68 → 
  let total_increase := 6 * avg_increase in
  let old_weight := new_weight - total_increase in
  old_weight = 47 :=
by
  intros h1 h2
  let total_increase := 6 * avg_increase
  have h3 : total_increase = 21 := by
    rw [h1]
    norm_num
  let old_weight := new_weight - total_increase
  have h4 : old_weight = 47 := by
    rw [h2, h3]
    norm_num
  exact h4

end replaced_weight_l481_481891


namespace polynomial_identity_l481_481941

theorem polynomial_identity
  (z1 z2 : ℂ)
  (h1 : z1 + z2 = -6)
  (h2 : z1 * z2 = 11)
  : (1 + z1^2 * z2) * (1 + z1 * z2^2) = 1266 := 
by 
  sorry

end polynomial_identity_l481_481941


namespace convex_quadrilateral_properties_l481_481094

open EuclideanGeometry

-- Define the convex quadrilateral and its properties
variables {A B C D E : Point}

theorem convex_quadrilateral_properties 
  (h_convex : ConvexQuadrilateral A B C D)
  (h_angle_sum : angle B + angle C < 180) 
  (h_intersection : intersectsAt A B C D E) :
  (CD * CE = AC^2 + AB * AE) ↔ angle B = angle D :=
sorry

end convex_quadrilateral_properties_l481_481094


namespace die_not_fair_if_10_consective_ones_l481_481092

open Probability

-- Define the condition: rolling a die 10 times consecutive and landing on 1 each time.
def roll_die_10_consecutive_with_1 : Prop :=
  (∀ i : Fin 10, roll_die i = 1)

-- Define the fairness of the die:
def fair_die : Prop :=
  ∀ (n : Nat), (n ≥ 1 ∧ n ≤ 6) → (probability (roll_die = n)) = 1 / 6

-- The main theorem statement expressing that if the die lands on 1 in 10 consecutive rolls, it is not fair:
theorem die_not_fair_if_10_consective_ones (h : roll_die_10_consecutive_with_1) : ¬fair_die :=
by
  sorry

end die_not_fair_if_10_consective_ones_l481_481092


namespace min_value_mn_l481_481022

theorem min_value_mn (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : log 3 m + log 3 n ≥ 4) : m + n ≥ 18 :=
sorry

end min_value_mn_l481_481022


namespace find_p_plus_q_l481_481155

theorem find_p_plus_q (M N : set ℝ) (p q : ℝ) 
  (hM: M = {x | x^2 - p * x + 8 = 0})
  (hN: N = {x | x^2 - q * x + p = 0})
  (h_intersect: M ∩ N = {1}) : p + q = 19 :=
by
  -- hint: proofs should be added here
  sorry

end find_p_plus_q_l481_481155


namespace area_on_larger_sphere_l481_481962

-- Define the variables representing the radii and the given area on the smaller sphere
variable (r1 r2 : ℝ) (area1 : ℝ)

-- Given conditions
def conditions : Prop :=
  r1 = 4 ∧ r2 = 6 ∧ area1 = 37

-- Define the statement that we need to prove
theorem area_on_larger_sphere (h : conditions r1 r2 area1) : 
  let area2 := area1 * (r2^2 / r1^2) in
  area2 = 83.25 :=
by
  -- Insert the proof here
  sorry

end area_on_larger_sphere_l481_481962


namespace Duke_broke_record_by_5_l481_481744

theorem Duke_broke_record_by_5 :
  let free_throws := 5
  let regular_baskets := 4
  let normal_three_pointers := 2
  let extra_three_pointers := 1
  let points_per_free_throw := 1
  let points_per_regular_basket := 2
  let points_per_three_pointer := 3
  let points_to_tie_record := 17

  let total_points_scored := (free_throws * points_per_free_throw) +
                             (regular_baskets * points_per_regular_basket) +
                             ((normal_three_pointers + extra_three_pointers) * points_per_three_pointer)
  total_points_scored = 22 →
  total_points_scored - points_to_tie_record = 5 :=

by
  intros
  sorry

end Duke_broke_record_by_5_l481_481744


namespace range_of_a_l481_481689

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x^2 + 2 * x - a > 0) → a < 3 :=
by
  sorry

end range_of_a_l481_481689


namespace sam_gave_fred_5_balloons_l481_481146

/-
Sam gives some of his 6 yellow balloons to Fred, and Mary has 7 yellow balloons.
Sam and Mary have 8 yellow balloons in total. Prove that Sam gave Fred 5 yellow balloons.
-/
theorem sam_gave_fred_5_balloons 
    (sam_initial : ℕ) 
    (mary : ℕ) 
    (total_together : ℕ)
    (sam_left : ℕ) 
    (sam_gave : ℕ) :
  sam_initial = 6 →
  mary = 7 →
  total_together = 8 →
  total_together = sam_left + mary →
  sam_gave = sam_initial - sam_left →
  sam_gave = 5 :=
by
  intros h_sam_initial h_mary h_total_together h_total_constraint h_sam_gave
  -- Assuming the given conditions
  have h1 : sam_initial = 6 := h_sam_initial
  have h2 : mary = 7 := h_mary
  have h3 : total_together = 8 := h_total_together
  have h4 : total_together = sam_left + mary := h_total_constraint
  have h5 : sam_gave = sam_initial - sam_left := h_sam_gave
  -- From h3 and h4, we deduce sam_left = 1
  have h_sam_left : sam_left = 1 := by {
    rw [h2] at h4
    rw [h3, add_comm, add_right_cancel_iff] at h4
    exact h4
  }
  -- Substituting sam_left = 1 in h_sam_gave should give us sam_gave = 5
  rw [h_sam_left] at h5
  rw [h1] at h5
  norm_num at h5
  exact h5

end sam_gave_fred_5_balloons_l481_481146


namespace sum_of_first_15_numbers_l481_481886

def sequence (n : ℕ) : ℕ :=
  if n % 2 = 1 then n + 1 else n - 1

theorem sum_of_first_15_numbers : (Finset.range 15).sum (λ n, sequence (n + 1)) = 121 :=
by
  -- Start by defining the sequences as given in the condition
  let evens := [2, 4, 6, 8, 10, 12, 14, 16] -- length 8
  let odds := [1, 3, 5, 7, 9, 11, 13]      -- length 7
  have h_even_sum : evens.sum = 72 := sorry
  have h_odd_sum : odds.sum = 49 := sorry
  -- Use the results to show the sum is 121
  rw [Finset.sum_range_succ, h_even_sum, h_odd_sum]
  norm_num
  done

end sum_of_first_15_numbers_l481_481886


namespace area_equality_l481_481082

variables (A B C L K M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace L] [MetricSpace K] [MetricSpace M] [MetricSpace N]
variables (f : Triangle A B C) (g : Line A L) (h : Line L K) (i : Line L M)
variables (j : AngleBisector A B C) (k : PerpendicularDrop L A B) (l : PerpendicularDrop L A C) (m : Circumcircle A B C N)
variables (n : AcuteTriangle A B C)

def area_triang (T : Triangle A B C) : ℝ := sorry  -- Placeholder

def area_quad (Q : Quadrilateral A K N M) : ℝ := sorry  -- Placeholder

theorem area_equality : area_triang ⟨A, B, C⟩ = area_quad ⟨A, K, N, M⟩ := sorry

end area_equality_l481_481082


namespace number_of_rabbits_l481_481778

theorem number_of_rabbits (x y : ℕ) (h1 : x + y = 28) (h2 : 4 * x = 6 * y + 12) : x = 18 :=
by
  sorry

end number_of_rabbits_l481_481778


namespace unique_solution_for_power_equation_l481_481319

theorem unique_solution_for_power_equation 
  (a p n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hp_pos : 0 < p) (hn : 0 < n) : 
  p ^ a - 1 = 2 ^ n * (p - 1) → a = 2 ∧ ∃ n, p = 2 ^ n - 1 ∧ Nat.Prime (2 ^ n - 1) :=
begin
  sorry
end

end unique_solution_for_power_equation_l481_481319


namespace women_in_room_l481_481437

theorem women_in_room :
  ∃ (x : ℤ), 
    let men_initial := 4 * x,
        women_initial := 5 * x,
        men_now := men_initial + 2,
        women_left := women_initial - 3,
        women_doubled := 2 * women_left,
        men_now = 14 in
    2 * (5 * x - 3) = 24 :=
by
  sorry

end women_in_room_l481_481437


namespace problem_alternating_sum_zero_l481_481835

theorem problem_alternating_sum_zero :
  let f (x : ℝ) := x^2 * (1 - x)^2 in
  ∑ k in finset.range 2020 | k + 1, (-1) ^ (k + 1) * f (k / 2021) = 0 := sorry

end problem_alternating_sum_zero_l481_481835


namespace probability_intersection_ab_probability_subtraction_union_l481_481042

def set_a (x : ℝ) : Prop := x^2 + 3 * x - 4 < 0
def set_b (x : ℝ) : Prop := (x + 2) / (x - 4) < 0
def interval : set ℝ := { x | -4 < x ∧ x < 5 }
def intersection_ab (x : ℝ) : Prop := set_a x ∧ set_b x

noncomputable def prob_intersection : ℝ := 1 / 3 -- Calculated probability of set A ∩ B in given interval
theorem probability_intersection_ab : 
  prob (λ x, intersection_ab x ∧ interval x) = prob_intersection :=
sorry

def set_int_a : set ℤ := {a | set_a a}
def set_int_b : set ℤ := {b | set_b b}
def union_ab (x : ℝ) : Prop := set_a x ∨ set_b x

noncomputable def prob_union_subtraction : ℝ := 7 / 10 -- Calculated probability of a - b ∈ A ∪ B for integers a ∈ A, b ∈ B
theorem probability_subtraction_union : 
  prob (λ p : ℤ × ℤ, union_ab (p.1 - p.2)) (set_int_a ×ˢ set_int_b) = prob_union_subtraction :=
sorry

end probability_intersection_ab_probability_subtraction_union_l481_481042


namespace area_A₀B₀C₀_eq_2_area_AC₁BA₁CB₁_area_A₀B₀C₀_ge_4_area_ABC_l481_481073

open EuclideanGeometry

-- Definitions and conditions
variables {A B C A₁ B₁ C₁ A₀ B₀ C₀ : Point}
variables {α β γ : ℝ}

-- Given conditions about the triangle
axioms
  (hA₁ : IsAngleBisector A B C A₁)
  (hB₁ : IsAngleBisector B C A B₁)
  (hC₁ : IsAngleBisector C A B C₁)
  (hAA₁ : IntersectsCircumcircle A A₁)
  (hB₀ : IsExternalAngleBisector A₁ A C B₀)
  (hC₀ : IsExternalAngleBisector A₁ A B C₀)

-- First proof goal: 
theorem area_A₀B₀C₀_eq_2_area_AC₁BA₁CB₁ :
  area (triangle A₀ B₀ C₀) = 2 * area (hexagon A C₁ B A₁ C B₁) :=
sorry

-- Second proof goal:
theorem area_A₀B₀C₀_ge_4_area_ABC :
  area (triangle A₀ B₀ C₀) ≥ 4 * area (triangle A B C) :=
sorry

end area_A₀B₀C₀_eq_2_area_AC₁BA₁CB₁_area_A₀B₀C₀_ge_4_area_ABC_l481_481073


namespace rachel_remaining_amount_l481_481869

-- Definitions of the initial earning, the fraction spent on lunch, and the fraction spent on the DVD.
def initial_amount : ℝ := 200
def fraction_lunch : ℝ := 1/4
def fraction_dvd : ℝ := 1/2

-- Calculation of the remaining amount Rachel has.
theorem rachel_remaining_amount :
  let spent_on_lunch := fraction_lunch * initial_amount in
  let spent_on_dvd := fraction_dvd * initial_amount in
  let remaining_amount := initial_amount - spent_on_lunch - spent_on_dvd in
  remaining_amount = 50 :=
by
  sorry

end rachel_remaining_amount_l481_481869


namespace find_constants_l481_481685

theorem find_constants (C D : ℝ) :
  (∀ x : ℝ, x ≠ 6 ∧ x ≠ -3 → 5 * x - 3 = C * (x + 3) + D * (x - 6)) →
  (C = 3 ∧ D = 2) :=
by
  intros h
  have hC : C = 3,
  { 
    specialize h 6,
    -- Simplifying h 6 with the condition x ≠ 6 ∧ x ≠ -3
    sorry,
  },
  have hD : D = 2,
  { 
    specialize h (-3),
    -- Simplifying h (-3) with the condition x ≠ 6 ∧ x ≠ -3
    sorry,
  },
  exact ⟨hC, hD⟩

end find_constants_l481_481685


namespace Vince_ride_longer_l481_481921

def Vince_ride_length : ℝ := 0.625
def Zachary_ride_length : ℝ := 0.5

theorem Vince_ride_longer : Vince_ride_length - Zachary_ride_length = 0.125 := by
  sorry

end Vince_ride_longer_l481_481921


namespace fraction_in_classroom_l481_481557

theorem fraction_in_classroom (total_students absent_fraction canteen_students present_students class_students : ℕ) 
  (h_total : total_students = 40)
  (h_absent_fraction : absent_fraction = 1 / 10)
  (h_canteen_students : canteen_students = 9)
  (h_absent_students : absent_fraction * total_students = 4)
  (h_present_students : present_students = total_students - absent_fraction * total_students)
  (h_class_students : class_students = present_students - canteen_students) :
  class_students / present_students = 3 / 4 := 
by {
  sorry
}

end fraction_in_classroom_l481_481557


namespace AC_total_l481_481978

theorem AC_total (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 450) (h3 : C = 100) : A + C = 250 := by
  sorry

end AC_total_l481_481978


namespace factorization_correct_l481_481171

theorem factorization_correct (a b : ℝ) : 
  a^2 + 2 * b - b^2 - 1 = (a - b + 1) * (a + b - 1) :=
by
  sorry

end factorization_correct_l481_481171


namespace max_value_acos_bsin_l481_481112

theorem max_value_acos_bsin 
  (a b : ℝ) : ∃ θ : ℝ, a * real.cos θ + b * real.sin θ = real.sqrt (a^2 + b^2) :=
sorry

end max_value_acos_bsin_l481_481112


namespace find_angle_FPD_l481_481194

-- Define the initial setup with points and angles
axiom exists_points (D E F P : Type) : Prop

axiom is_triangle (DEF : triangle D E F)

axiom is_isosceles (isosceles_DEF : isosceles DEF DF EF)

axiom angle_DFE (DFE : angle D F E) : DFE = 104

axiom angle_DPF (DPF : angle D P F) : DPF = 13

axiom angle_EPF (EPF : angle E P F) : EPF = 21

theorem find_angle_FPD :
  ∃ (FPD : angle F P D), FPD = 34 :=
by
  sorry

end find_angle_FPD_l481_481194


namespace sum_of_elements_in_T_eq_111111000₂_l481_481110

-- Define the set T as the set of all five-digit binary numbers
def T := { n : ℕ | ∃ k : Fin 16, n = 16 + k }

-- Proof that the sum of all elements in T is 111111000_2
theorem sum_of_elements_in_T_eq_111111000₂ :
  (∑ n in T, n) = 0b111111000 :=
by
  sorry -- Detailed proof is omitted

end sum_of_elements_in_T_eq_111111000₂_l481_481110


namespace gold_coins_percentage_l481_481646

def percentage_of_gold_coins (total_objects beads marbles coins silver_coins : ℝ) : ℝ :=
let percentage_coins := total_objects - beads - marbles in
let percentage_gold_coins := 1 - silver_coins in
percentage_coins * percentage_gold_coins

theorem gold_coins_percentage (total_objects beads marbles silver_coins : ℝ) :
  beads = 0.30 * total_objects → marbles = 0.10 * total_objects →
  silver_coins = 0.45 → 
  percentage_of_gold_coins 1 0.30 0.10 0.60 0.45 = 0.33 :=
by
  intro h_beads h_marbles h_silver_coins
  simp [percentage_of_gold_coins, h_beads, h_marbles, h_silver_coins]
  norm_num

end gold_coins_percentage_l481_481646


namespace average_monthly_growth_rate_proof_profit_in_may_proof_l481_481850

theorem average_monthly_growth_rate_proof :
  ∃ r : ℝ, 2400 * (1 + r)^2 = 3456 ∧ r = 0.2 := sorry

theorem profit_in_may_proof (r : ℝ) (h_r : r = 0.2) :
  3456 * (1 + r) = 4147.2 := sorry

end average_monthly_growth_rate_proof_profit_in_may_proof_l481_481850


namespace exist_lines_intersect_circle_l481_481243

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 4 = 0

noncomputable def line_eq1 (x y : ℝ) : Prop :=
  y = x + 1

noncomputable def line_eq2 (x y : ℝ) : Prop :=
  y = x - 4

theorem exist_lines_intersect_circle (x y : ℝ) :
  (∃ (x y : ℝ), circle_eq x y ∧ line_eq1 x y) ∨ 
  (∃ (x y : ℝ), circle_eq x y ∧ line_eq2 x y) :=
sorry

end exist_lines_intersect_circle_l481_481243


namespace find_f_of_f_l481_481733

def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 3 else 2 ^ x

theorem find_f_of_f ( : f (f (1 / 3)) = 1 / 2 :=
by
  sorry

end find_f_of_f_l481_481733


namespace existence_of_points_d_e_f_l481_481480

variables {A B C O H D E F : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace O] [MetricSpace H] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (triangle_abc : Triangle A B C)
variables (circumcenter_o : Circumcenter O triangle_abc)
variables (orthocenter_h : Orthocenter H triangle_abc)

theorem existence_of_points_d_e_f :
  exists (D E F : Point), 
    (D ∈ Segment B C) ∧ (E ∈ Segment C A) ∧ (F ∈ Segment A B) ∧
    (dist O D + dist D H = dist O E + dist E H) ∧ (dist O E + dist E H = dist O F + dist F H) ∧
    are_concurrent (Line A D) (Line B E) (Line C F) :=
sorry

end existence_of_points_d_e_f_l481_481480


namespace remainder_8547_div_9_l481_481575

theorem remainder_8547_div_9 : 8547 % 9 = 6 :=
by
  sorry

end remainder_8547_div_9_l481_481575


namespace derivative_lg_over_x_l481_481894

open Real

noncomputable def y (x : ℝ) : ℝ := (log x / log 10) / x

theorem derivative_lg_over_x (x : ℝ) (hx : 0 < x) :
    deriv (λ x, (log x / log 10) / x) x = (1 - log 10 * (log x / log 10)) / (x^2 * log 10) :=
by
  sorry

end derivative_lg_over_x_l481_481894


namespace probability_one_pair_of_same_color_l481_481746

noncomputable def countCombinations : ℕ :=
  Nat.choose 10 5

noncomputable def countFavCombinations : ℕ :=
  (Nat.choose 5 4) * 4 * (2 * 2 * 2)

theorem probability_one_pair_of_same_color :
  (countFavCombinations : ℚ) / (countCombinations : ℚ) = 40 / 63 := by
  sorry

end probability_one_pair_of_same_color_l481_481746


namespace h_sum_condition_b_range_condition_l481_481374

variables (a b c : ℝ) (f g h : ℝ → ℝ)

-- Define f(x), given a > 0, b, c ∈ ℝ
def f (x : ℝ) : ℝ := (1 / 3) * a * x^3 + (1 / 2) * b * x^2 + c * x

-- Define g(x) as the derivative of f(x)
def g (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define h(x) based on g(x)
def h (x : ℝ) : ℝ :=
  if x ≥ 1 then g (x - 1)
  else - g (x - 1)

-- Given conditions for the first problem
axiom a_pos : a > 0
axiom c_one : c = 1
axiom g_min_at_neg1 : g (-1) = 0

-- Objective for the first problem
theorem h_sum_condition : h 2 + h (-2) = 0 := sorry

-- Given conditions for the second problem
axiom a_one : a = 1
axiom c_zero : c = 0
axiom abs_g_le_one : ∀ x ∈ Ioo 0 2, |g x| ≤ 1

-- Objective for the second problem
theorem b_range_condition : -2 ≤ b ∧ b ≤ -3 / 2 := sorry

end h_sum_condition_b_range_condition_l481_481374


namespace bob_spending_in_usd_l481_481653

theorem bob_spending_in_usd :
  ∀ (yen_per_usd : ℝ) (coffee_cost_den : ℝ) (snack_cost_den : ℝ),
  yen_per_usd = 100 →
  coffee_cost_den = 250 →
  snack_cost_den = 150 →
  (coffee_cost_den + snack_cost_den) / yen_per_usd = 4 :=
by
  intros yen_per_usd coffee_cost_den snack_cost_den yen_conv coffee_cost snack_cost
  rw [yen_conv, coffee_cost, snack_cost]
  norm_num
  sorry

end bob_spending_in_usd_l481_481653


namespace RachelLeftoverMoney_l481_481866

def RachelEarnings : ℝ := 200
def SpentOnLunch : ℝ := (1/4) * RachelEarnings
def SpentOnDVD : ℝ := (1/2) * RachelEarnings
def TotalSpent : ℝ := SpentOnLunch + SpentOnDVD

theorem RachelLeftoverMoney : RachelEarnings - TotalSpent = 50 := by
  sorry

end RachelLeftoverMoney_l481_481866


namespace triangle_perimeter_l481_481272

/-
  A square piece of paper with side length 2 has vertices A, B, C, and D. 
  The paper is folded such that vertex A meets edge BC at point A', 
  and A'C = 1/2. Prove that the perimeter of triangle A'BD is (3 + sqrt(17))/2 + 2sqrt(2).
-/
theorem triangle_perimeter
  (A B C D A' : ℝ × ℝ)
  (side_length : ℝ)
  (BC_length : ℝ)
  (CA'_length : ℝ)
  (BA'_length : ℝ)
  (BD_length : ℝ)
  (DA'_length : ℝ)
  (perimeter_correct : ℝ) :
  side_length = 2 ∧
  BC_length = 2 ∧
  CA'_length = 1/2 ∧
  BA'_length = 3/2 ∧
  BD_length = 2 * Real.sqrt 2 ∧
  DA'_length = Real.sqrt 17 / 2 →
  perimeter_correct = (3 + Real.sqrt 17) / 2 + 2 * Real.sqrt 2 →
  (side_length ≠ 0 ∧ BC_length = side_length ∧ 
   CA'_length ≠ 0 ∧ BA'_length ≠ 0 ∧ 
   BD_length ≠ 0 ∧ DA'_length ≠ 0) →
  (BA'_length + BD_length + DA'_length = perimeter_correct) :=
  sorry

end triangle_perimeter_l481_481272


namespace coeff_x2_in_binomial_expansion_l481_481791

theorem coeff_x2_in_binomial_expansion :
  (∃ c : ℕ, (x + (1/x))^10 = (c * x^2 + ...) ∧ c = 210) :=
sorry

end coeff_x2_in_binomial_expansion_l481_481791


namespace outfit_count_l481_481405

theorem outfit_count 
  (S P T J : ℕ) 
  (hS : S = 8) 
  (hP : P = 5) 
  (hT : T = 4) 
  (hJ : J = 3) : 
  S * P * (T + 1) * (J + 1) = 800 := by 
  sorry

end outfit_count_l481_481405


namespace sequence_properties_l481_481803

theorem sequence_properties 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (T : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_sum : ∀ n, S n = ∑ i in range (n+1), a i)
  (h_a2an : ∀ n, a 2 * a n = S 2 + S n)
  (h_a1_pos : a 1 > 0)
  (h_bn : ∀ n, b n = log (10 * a 1 / a n))
  (h_T : ∀ n, T n = ∑ i in range (n+1), b i) :
  (a 1 = sqrt 2 + 1 ∧ a 2 = 2 + sqrt 2) ∨ (a 1 = 1 - sqrt 2 ∧ a 2 = 2 - sqrt 2) ∧
  (∀ n, b n = 1 - (n * log (sqrt 2))) ∧
  T 7 = 7 - (21 / 2 * log 2) ∧ ∀ n, T n ≤ T 7 :=
by
  sorry

end sequence_properties_l481_481803


namespace make_tea_time_efficiently_l481_481851

theorem make_tea_time_efficiently (minutes_kettle minutes_boil minutes_teapot minutes_teacups minutes_tea_leaves total_estimate total_time : ℕ)
  (h1 : minutes_kettle = 1)
  (h2 : minutes_boil = 15)
  (h3 : minutes_teapot = 1)
  (h4 : minutes_teacups = 1)
  (h5 : minutes_tea_leaves = 2)
  (h6 : total_estimate = 20)
  (h_total_time : total_time = minutes_kettle + minutes_boil) :
  total_time = 16 :=
by
  sorry

end make_tea_time_efficiently_l481_481851


namespace largest_k_triangle_inequality_l481_481682

theorem largest_k_triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) :
  (∃ k : ℝ, (∀ (a b c : ℝ), 
      a + b > c ∧ a + c > b ∧ b + c > a → 
      (bc / (b + c - a) + ac / (a + c - b) + ab / (a + b - c) >= k * (a + b + c))
      ) → k <= 1 :=
by
  sorry

end largest_k_triangle_inequality_l481_481682


namespace Bethany_total_riding_hours_l481_481999

-- Define daily riding hours
def Monday_hours : Nat := 1
def Wednesday_hours : Nat := 1
def Friday_hours : Nat := 1
def Tuesday_hours : Nat := 1 / 2
def Thursday_hours : Nat := 1 / 2
def Saturday_hours : Nat := 2

-- Define total weekly hours
def weekly_hours : Nat :=
  Monday_hours + Wednesday_hours + Friday_hours + (Tuesday_hours + Thursday_hours) + Saturday_hours

-- Definition to account for the 2-week period
def total_hours (weeks : Nat) : Nat := weeks * weekly_hours

-- Prove that Bethany rode 12 hours over 2 weeks
theorem Bethany_total_riding_hours : total_hours 2 = 12 := by
  sorry

end Bethany_total_riding_hours_l481_481999


namespace least_n_for_product_exceeds_2010_l481_481117

def p (k: ℕ) : ℝ := 1 + (1 / k) - (1 / k^2) - (1 / k^3)

def product_p_2_to_n (n: ℕ) : ℝ := (finset.range n).product (λ k, p (k + 2))

theorem least_n_for_product_exceeds_2010 :
  ∃ (n: ℕ), product_p_2_to_n n > 2010 ∧ ∀ (m: ℕ), m < n → product_p_2_to_n m ≤ 2010 :=
sorry

end least_n_for_product_exceeds_2010_l481_481117


namespace range_of_a_l481_481040

theorem range_of_a (a : ℝ) :
  let f (x : ℝ) := if x ≥ 0 then x^2 - 2*x + 2 else x + a/x + 3*a in
  (∀ y ∈ set.range f, ∃ x, f x = y) ↔ (a ∈ set.Iic 0 ∪ set.Ici 1) :=
by sorry

end range_of_a_l481_481040


namespace circle_O1_cartesian_circle_O2_cartesian_intersection_line_l481_481565

-- Definitions for circles in polar coordinates
def polar_circle_O1 := ∀ θ : ℝ, ρ = 4 * Real.cos θ
def polar_circle_O2 := ∀ θ : ℝ, ρ = - Real.sin θ

-- Theorem statements using Cartesian coordinates
theorem circle_O1_cartesian (θ : ℝ) (x y : ℝ) (h : ρ = 4 * Real.cos θ) : x^2 + y^2 - 4 * x = 0 := sorry
theorem circle_O2_cartesian (θ : ℝ) (x y : ℝ) (h : ρ = - Real.sin θ) : x^2 + y^2 + y = 0 := sorry

-- Theorem statement for the line passing through intersection points of the two circles
theorem intersection_line (x y : ℝ) 
  (h1 : x^2 + y^2 - 4 * x = 0) 
  (h2 : x^2 + y^2 + y = 0) : 4 * x + y = 0 := sorry

end circle_O1_cartesian_circle_O2_cartesian_intersection_line_l481_481565


namespace problem_statement_l481_481945

noncomputable def x : ℝ := 1260 / 1212

theorem problem_statement : 
  ∃ x : ℝ, (40 * 30 + (12 + 8) * 3) / x = 1212 ∧ x ≈ 1.04 :=
by
  use 1260 / 1212
  split
  { norm_num }
  { exact rfl }
  sorry

end problem_statement_l481_481945


namespace combined_savings_zero_l481_481973

theorem combined_savings_zero (price_per_window : ℝ) (window_price : price_per_window = 150) 
  (free_windows : ℕ → ℕ) (hw_free_windows : ∀ n, free_windows (n + 6) = free_windows n + 2) 
  (dave_needs : ℕ) (hdave : dave_needs = 9) (doug_needs : ℕ) (hdoug : doug_needs = 10) : 
  let cost_without_offer (n : ℕ) := n * price_per_window in
  let total_needed := dave_needs + doug_needs in
  let get_free_windows (n : ℕ) := n / 6 * 2 in
  let cost_with_offer (n : ℕ) := 
    price_per_window * (n - get_free_windows n) in
  cost_without_offer total_needed - cost_with_offer total_needed 
  = (cost_without_offer dave_needs - cost_with_offer dave_needs) +
    (cost_without_offer doug_needs - cost_with_offer doug_needs) := by 
  sorry

end combined_savings_zero_l481_481973


namespace grunters_win_all_6_games_l481_481527

noncomputable def prob_no_overtime_win : ℚ := 0.54
noncomputable def prob_overtime_win : ℚ := 0.05
noncomputable def prob_win_any_game : ℚ := prob_no_overtime_win + prob_overtime_win
noncomputable def prob_win_all_6_games : ℚ := prob_win_any_game ^ 6

theorem grunters_win_all_6_games :
  prob_win_all_6_games = (823543 / 10000000) :=
by sorry

end grunters_win_all_6_games_l481_481527


namespace people_per_car_l481_481758

theorem people_per_car (total_people cars : ℕ) (h1 : total_people = 63) (h2 : cars = 9) :
  total_people / cars = 7 :=
by
  sorry

end people_per_car_l481_481758


namespace triangle_minimum_area_l481_481242

theorem triangle_minimum_area :
  ∃ p q : ℤ, p ≠ 0 ∧ q ≠ 0 ∧ (1 / 2) * |30 * q - 18 * p| = 3 :=
sorry

end triangle_minimum_area_l481_481242


namespace polynomial_divisibility_by_6_l481_481457

theorem polynomial_divisibility_by_6 (a b c : ℤ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 :=
sorry

end polynomial_divisibility_by_6_l481_481457


namespace solve_system_of_equations_l481_481846

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : x + y = 55) 
  (h2 : x - y = 15) 
  (h3 : x > y) : 
  x = 35 ∧ y = 20 := 
sorry

end solve_system_of_equations_l481_481846


namespace pyramid_surface_area_leq_cube_face_l481_481958

theorem pyramid_surface_area_leq_cube_face (V E F : Type)
  [add_comm_group V] [module ℝ V]
  (cube : E)
  (sphere : V)
  (plane_tangent_to_sphere : V) 
  (intersecting_plane : E) 
  (triangular_pyramid : F) 
  (face_of_cube : F)
  (inscribed_sphere : sphere ∈ cube)
  (plane_tangent_sphere : plane_tangent_to_sphere ∈ sphere)
  (pyramid_formed_by_intersection : triangular_pyramid ∈ intersecting_plane ∩ cube)
  (face_area : ℝ)
  (pyramid_area : ℝ) :
  pyramid_area ≤ face_area := 
sorry

end pyramid_surface_area_leq_cube_face_l481_481958


namespace cos_neg_17pi_over_4_l481_481673

noncomputable def cos_value : ℝ := (Real.pi / 4).cos

theorem cos_neg_17pi_over_4 :
  (Real.cos (-17 * Real.pi / 4)) = cos_value :=
by
  -- Define even property of cosine and angle simplification
  sorry

end cos_neg_17pi_over_4_l481_481673


namespace total_dots_correct_l481_481857

/-- Define the initial conditions -/
def monday_ladybugs : ℕ := 8
def monday_dots_per_ladybug : ℕ := 6
def tuesday_ladybugs : ℕ := 5
def wednesday_ladybugs : ℕ := 4

/-- Define the derived conditions -/
def tuesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 1
def wednesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 2

/-- Calculate the total number of dots -/
def monday_total_dots : ℕ := monday_ladybugs * monday_dots_per_ladybug
def tuesday_total_dots : ℕ := tuesday_ladybugs * tuesday_dots_per_ladybug
def wednesday_total_dots : ℕ := wednesday_ladybugs * wednesday_dots_per_ladybug
def total_dots : ℕ := monday_total_dots + tuesday_total_dots + wednesday_total_dots

/-- Prove the total dots equal to 89 -/
theorem total_dots_correct : total_dots = 89 := by
  sorry

end total_dots_correct_l481_481857


namespace find_substring_with_counts_l481_481176

theorem find_substring_with_counts (n : ℕ) (h1 : 0 < n) (s : String) 
    (h2 : s.count 'A' = 3 * n) (h3 : s.count 'B' = 2 * n) : 
    ∃ t : String, t ∈ s.substr : (5.0 : ℕ) ∧ t.count 'A' = 3 ∧ t.count 'B' = 2 :=
by
  -- Here the proof would go, but we use sorry as per the instructions
  sorry

end find_substring_with_counts_l481_481176


namespace sum_abs_coeffs_l481_481004

theorem sum_abs_coeffs : 
  let f := (1 - 2 * x)^7 
  let a := fun (n : ℕ) => (1 - 2 * x)^7.coeff n 
  ∑ i in (Finset.range 7).erase 0, |a i| = 3^7 - 1 :=
by
  sorry

end sum_abs_coeffs_l481_481004


namespace John_height_in_feet_after_growth_spurt_l481_481812

def John_initial_height : ℕ := 66
def growth_rate_per_month : ℕ := 2
def number_of_months : ℕ := 3
def inches_per_foot : ℕ := 12

theorem John_height_in_feet_after_growth_spurt :
  (John_initial_height + growth_rate_per_month * number_of_months) / inches_per_foot = 6 := by
  sorry

end John_height_in_feet_after_growth_spurt_l481_481812


namespace angle_A_is_pi_over_3_area_of_triangle_l481_481070

-- Conditions and Definitions
variables {a b c A B C : ℝ} (ΔABC : a^2 + b^2 - 2 * b * c * (cos A) = a^2)
def a_value : ℝ := 2
def f (x : ℝ) : ℝ := sqrt 3 * sin (x / 2) * cos (x / 2) + cos^2 (x / 2)

-- First Proof Problem: Measure of angle A
theorem angle_A_is_pi_over_3 (h : b^2 + c^2 - a^2 = b * c) : 
  A = π / 3 :=
sorry

-- Second Proof Problem: Area of the triangle
theorem area_of_triangle (hA : A = π / 3) (hmax : ∀ x, f x ≤ f B) : 
  area = sqrt 3 :=
sorry

end angle_A_is_pi_over_3_area_of_triangle_l481_481070


namespace length_of_BD_l481_481692

theorem length_of_BD (A B C D E : Point)
  (h1 : dist A B = dist B D)
  (h2 : angle A B D = angle D B C)
  (h3 : angle B C D = 90)
  (E_on_BC : E ∈ line_segment B C)
  (h4 : dist A D = dist D E)
  (BE_length : dist B E = 7)
  (EC_length : dist E C = 5) :
  dist B D = 17 :=
by 
  sorry

end length_of_BD_l481_481692


namespace exists_five_primes_in_arithmetic_progression_exists_six_primes_in_arithmetic_progression_l481_481938

def is_prime (n : ℕ) : Prop := Nat.Prime n

def arithmetic_progression (a d : ℕ) (n : ℕ) : list ℕ :=
(list.range n).map (λ i, a + i * d)

theorem exists_five_primes_in_arithmetic_progression : ∃ (a d : ℕ), ∀ i < 5, is_prime (arithmetic_progression a d 5)[i] :=
by 
  sorry

theorem exists_six_primes_in_arithmetic_progression : ∃ (a d : ℕ), ∀ i < 6, is_prime (arithmetic_progression a d 6)[i] :=
by
  sorry

end exists_five_primes_in_arithmetic_progression_exists_six_primes_in_arithmetic_progression_l481_481938


namespace apples_bought_l481_481294

theorem apples_bought (x : ℕ) 
  (h1 : x ≠ 0)  -- x must be a positive integer
  (h2 : 2 * (x/3) = 2 * x / 3 + 2 - 6) : x = 24 := 
  by sorry

end apples_bought_l481_481294


namespace find_n_l481_481399

-- Define the conditions
variables (x n : ℤ)
variable (k m : ℤ)
hypothesis h1 : x % 62 = 7
hypothesis h2 : (x + n) % 31 = 18

-- Statement we need to prove
theorem find_n (h1 : x % 62 = 7) (h2 : (x + n) % 31 = 18) : n = 11 := 
sorry

end find_n_l481_481399


namespace simplify_and_evaluate_div_expr_l481_481149

variable (m : ℤ)

theorem simplify_and_evaluate_div_expr (h : m = 2) :
  ( (m^2 - 9) / (m^2 - 6 * m + 9) / (1 - 2 / (m - 3)) = -5 / 3) :=
by
  sorry

end simplify_and_evaluate_div_expr_l481_481149


namespace red_marked_area_on_larger_sphere_l481_481966

-- Define the conditions
def r1 : ℝ := 4 -- radius of the smaller sphere
def r2 : ℝ := 6 -- radius of the larger sphere
def A1 : ℝ := 37 -- area marked on the smaller sphere

-- State the proportional relationship as a Lean theorem
theorem red_marked_area_on_larger_sphere : 
  let A2 := A1 * (r2^2 / r1^2)
  A2 = 83.25 :=
by
  sorry

end red_marked_area_on_larger_sphere_l481_481966


namespace max_min_values_l481_481932

def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)
def interval := Set.Icc (0 : ℝ) (3 * Real.pi / 2)

theorem max_min_values :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ interval ∧ x₂ ∈ interval ∧
  (∀ x ∈ interval, f x ≤ f x₁) ∧ (∀ x ∈ interval, f x₂ ≤ f x) ∧
  f x₁ = 3 * Real.sqrt 3 / 2 ∧ f x₂ = -2 :=
begin
  sorry
end

end max_min_values_l481_481932


namespace dan_initial_money_l481_481306

theorem dan_initial_money (spent : ℕ) (left : ℕ) (initial_money : ℕ) :
  spent = 3 → left = 1 → initial_money = spent + left → initial_money = 4 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end dan_initial_money_l481_481306


namespace age_of_staff_l481_481889

theorem age_of_staff (n_students : ℕ) (avg_age_students avg_age_total : ℕ) (total_students total_age_student total_age_total : ℕ) (staff_age : ℕ) :
  n_students = 32 →
  avg_age_students = 16 →
  avg_age_total = 17 →
  total_students = n_students + 1 →
  total_age_student = n_students * avg_age_students →
  total_age_total = total_students * avg_age_total →
  total_age_student + staff_age = total_age_total →
  staff_age = 49 :=
by
  intros
  have h1 : n_students * avg_age_students = 32 * 16 := by simp [*]
  have h2 : total_students = 33 := by simp [*]
  have h3 : total_age_total = 33 * 17 := by simp [*]
  have h4 : total_age_student + staff_age = 561 := by simp [h1, h3]
  have h5 : total_age_student = 512 := by simp [h1]
  have h6 : staff_age = 561 - 512 := by simp [*]
  rw [h6]
  exact Nat.sub_self 49
  done

end age_of_staff_l481_481889


namespace probability_A_plus_complement_B_l481_481954

-- Define the probability of a single outcome on a fair six-sided die
def probability_single_outcome : ℚ := 1/6

-- Define the set of all possible outcomes of a six-sided die
def total_outcomes := {1, 2, 3, 4, 5, 6}

-- Define event A: an even number less than 5 appears
def event_A := {2, 4}

-- Define event B: a number less than 5 appears
def event_B := {1, 2, 3, 4}

-- Define the complement of event B: a number not less than 5 appears
def event_not_B := {5, 6}

-- Probabilities of events A and B
def P_A : ℚ := event_A.size / total_outcomes.size
def P_B : ℚ := event_B.size / total_outcomes.size

-- Probability of event not B
def P_not_B : ℚ := 1 - P_B

-- Proof statement for the combined probability of A and not B (using mutual exclusiveness)
theorem probability_A_plus_complement_B : P_A + P_not_B = 2/3 := by
  sorry

end probability_A_plus_complement_B_l481_481954


namespace fifth_largest_divisor_l481_481166

theorem fifth_largest_divisor (n : ℕ) (h : n = 5040000000) : 
  (nat.divisors n).nth_le (nat.divisors n).length.pred.pred.pred.pred 4 = 315000000 :=
  sorry

end fifth_largest_divisor_l481_481166


namespace women_in_room_l481_481436

theorem women_in_room :
  ∃ (x : ℤ), 
    let men_initial := 4 * x,
        women_initial := 5 * x,
        men_now := men_initial + 2,
        women_left := women_initial - 3,
        women_doubled := 2 * women_left,
        men_now = 14 in
    2 * (5 * x - 3) = 24 :=
by
  sorry

end women_in_room_l481_481436


namespace Kath_takes_3_friends_l481_481102

theorem Kath_takes_3_friends
  (total_paid: Int)
  (price_before_6: Int)
  (price_reduction: Int)
  (num_family_members: Int)
  (start_time: Int)
  (start_time_condition: start_time < 18)
  (total_payment_condition: total_paid = 30)
  (admission_cost_before_6: price_before_6 = 8 - price_reduction)
  (num_family_members_condition: num_family_members = 3):
  (total_paid / price_before_6 - num_family_members = 3) := 
by
  -- Since no proof is required, simply add sorry to skip the proof
  sorry

end Kath_takes_3_friends_l481_481102


namespace courses_combination_count_l481_481268

theorem courses_combination_count :
  let courses := {A, B, C, D, E, F, G} in
  ∀ courses_chosen : Finset (Fin 7),
    (courses_chosen.card = 3) ∧ (¬courses.chosen.contains A ∨ ¬courses_chosen.contains B ∨ ¬courses_chosen.contains C) 
    → courses_chosen.card = 22 :=
by
  sorry

end courses_combination_count_l481_481268


namespace complex_division_example_l481_481328

theorem complex_division_example :
  (√2 - complex.i) / (1 + √2 * complex.i) = -complex.i := 
by
  -- Define complex numbers
  let num := √2 - complex.i
  let den := 1 + √2 * complex.i

  -- Perform the division
  have h : num / den = -complex.i
  -- Accepting the proof for now
  sorry

#check complex_division_example

end complex_division_example_l481_481328


namespace seating_arrangements_count_l481_481530

-- Definitions based on the conditions
def chairs : ℕ := 15
def zetonians : ℕ := 5
def pultonians : ℕ := 5
def earthlings : ℕ := 5

def chair1 := 1
def chair15 := 15

-- Definitions for seating rules
def no_earthling_left_of_zetonian (seating : ℕ → char) :=
  ∀ i, seating i = 'Z' → seating (if i = 1 then 15 else i - 1) ≠ 'E'

def no_zetonian_left_of_pultonian (seating : ℕ → char) :=
  ∀ i, seating i = 'P' → seating (if i = 1 then 15 else i - 1) ≠ 'Z'

def no_pultonian_left_of_earthling (seating : ℕ → char) :=
  ∀ i, seating i = 'E' → seating (if i = 1 then 15 else i - 1) ≠ 'P'

-- Main theorem statement
theorem seating_arrangements_count (N : ℕ) :
  ∃ seating : (ℕ → char),
    seating chair1 = 'Z' ∧
    seating chair15 = 'P' ∧
    no_earthling_left_of_zetonian seating ∧
    no_zetonian_left_of_pultonian seating ∧
    no_pultonian_left_of_earthling seating ∧
    N * (fact 5) ^ 3 = 346 * (fact 5) ^ 3 :=
  sorry

end seating_arrangements_count_l481_481530


namespace symmetry_about_x_eq_1_l481_481672

noncomputable def ceiling (x : ℝ) : ℤ := ⌈x⌉₊

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋₊

def f (x : ℝ) : ℝ := abs (ceiling x) - abs (floor (2 - x))

theorem symmetry_about_x_eq_1 :
  ∀ x : ℝ, f x = f (2 - x) := by
  sorry

end symmetry_about_x_eq_1_l481_481672


namespace probability_of_rolling_number_less_than_5_is_correct_l481_481215

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l481_481215


namespace train_crossing_time_l481_481430

noncomputable def speed_km_per_hr_to_m_per_s (v : ℚ) : ℚ :=
  v * 1000 / 3600

noncomputable def time_to_cross_pole (length : ℚ) (speed_kph : ℚ) : ℚ :=
  let speed_mps := speed_km_per_hr_to_m_per_s speed_kph
  length / speed_mps

theorem train_crossing_time :
  time_to_cross_pole 255 317 ≈ 2.895 :=
by {
  -- For the purposes of this statement, we will rely on the given approximation.
  -- Here is the logic we would follow:
  -- speed_km_per_hr_to_m_per_s 317 ≈ 88.0556,
  -- thus time_to_cross_pole 255 317 ≈ 2.895
  sorry
}

end train_crossing_time_l481_481430


namespace train_length_is_correct_l481_481273

noncomputable def length_of_train (t : ℝ) (v_train : ℝ) (v_man : ℝ) : ℝ :=
  let relative_speed : ℝ := (v_train - v_man) * (5/18)
  relative_speed * t

theorem train_length_is_correct :
  length_of_train 23.998 63 3 = 400 :=
by
  -- Placeholder for the proof
  sorry

end train_length_is_correct_l481_481273


namespace sum_of_values_satisfying_equation_l481_481577

theorem sum_of_values_satisfying_equation : 
  ∃ (s : ℤ), (∀ (x : ℤ), (abs (x + 5) = 9) → (x = 4 ∨ x = -14) ∧ (s = 4 + (-14))) :=
begin
  sorry
end

end sum_of_values_satisfying_equation_l481_481577


namespace part_I_part_II_l481_481355

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x
noncomputable def g (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := b * x^2 + c

theorem part_I (t : ℝ) (ht : t ≠ 0) (P : ℝ × ℝ)
  (hP : P = (t, 0))
  (hf : f t (-t^2) = 0)
  (hg : g t t (-t^3) = 0)
  (htangent : ∀ t, deriv (λ x, f x (-t^2)) t = deriv (λ x, g x t (-t^3)) t) :
  (a = -t^2) ∧ (b = t) ∧ (c = -t^3) :=
sorry

theorem part_II (t : ℝ) (ht : t ≠ 0) 
  (h_decreasing : ∀ x ∈ Ioo (-1 : ℝ) 3, deriv (λ x, f x (-t^2) - g x t (-t^3)) x < 0) :
  t ∈ Set.Icc (-∞) (-9) ∪ Set.Icc 3 ∞ :=
sorry

end part_I_part_II_l481_481355


namespace subway_boarding_probability_l481_481358

theorem subway_boarding_probability :
  ∀ (total_interval boarding_interval : ℕ),
  total_interval = 10 →
  boarding_interval = 1 →
  (boarding_interval : ℚ) / total_interval = 1 / 10 := by
  intros total_interval boarding_interval ht hb
  rw [hb, ht]
  norm_num

end subway_boarding_probability_l481_481358


namespace optimal_rent_is_4050_l481_481267

section RentalCompany

def rent_costs (rent_increase : ℕ) : ℕ :=
  3000 + rent_increase * 50

def revenue (rent_increase : ℕ) : ℕ :=
  let cars_rented := 100 - rent_increase in
  let rent_per_car := rent_costs rent_increase in
  let maintenance_cost := cars_rented * 150 + rent_increase * 50 in
  cars_rented * rent_per_car - maintenance_cost

theorem optimal_rent_is_4050 :
  ∃ rent_increase, rent_costs rent_increase = 4050 ∧ revenue rent_increase = 307050 :=
begin
  use 21, -- 4050 = 3000 + 21 * 50
  split,
  { refl, },
  { simp [revenue, rent_costs],
    sorry, -- We would provide the computation steps here.
  }
end

end RentalCompany

end optimal_rent_is_4050_l481_481267


namespace consecutive_integers_sum_24_greatest_9_l481_481924

theorem consecutive_integers_sum_24_greatest_9 :
  ∃ n : ℕ, ∃ x : ℤ, (Σ i in finset.range n, x + i) = 24 ∧ (x + n - 1) = 9 ∧ n = 3 :=
by { sorry }

end consecutive_integers_sum_24_greatest_9_l481_481924


namespace factory_earns_8100_per_day_l481_481411

-- Define the conditions
def working_hours_machines := 23
def working_hours_fourth_machine := 12
def production_per_hour := 2
def price_per_kg := 50
def number_of_machines := 3

-- Calculate earnings
def total_earnings : ℕ :=
  let total_runtime_machines := number_of_machines * working_hours_machines
  let production_machines := total_runtime_machines * production_per_hour
  let production_fourth_machine := working_hours_fourth_machine * production_per_hour
  let total_production := production_machines + production_fourth_machine
  total_production * price_per_kg

theorem factory_earns_8100_per_day : total_earnings = 8100 :=
by
  sorry

end factory_earns_8100_per_day_l481_481411


namespace f_one_div_f_two_inequality_l481_481539

def f (x : ℝ) : ℝ := sorry

axiom f_domain : ∀ x, 0 < x → 0 < f x
axiom deriv_inequality : ∀ x, 0 < x → 2 * f x < x * (derivative f x) ∧ x * (derivative f x) < 3 * f x

theorem f_one_div_f_two_inequality :
  1 / 8 < f 1 / f 2 ∧ f 1 / f 2 < 1 / 4 :=
sorry

end f_one_div_f_two_inequality_l481_481539


namespace number_of_short_trees_to_plant_l481_481180

-- Definitions of the conditions
def current_short_trees : ℕ := 41
def current_tall_trees : ℕ := 44
def total_short_trees_after_planting : ℕ := 98

-- The statement to be proved
theorem number_of_short_trees_to_plant :
  total_short_trees_after_planting - current_short_trees = 57 :=
by
  -- Proof goes here
  sorry

end number_of_short_trees_to_plant_l481_481180


namespace rectangle_color_invariance_l481_481525

/-- A theorem stating that in any 3x7 rectangle with some cells colored black at random, there necessarily exist four cells of the same color, whose centers are the vertices of a rectangle with sides parallel to the sides of the original rectangle. -/
theorem rectangle_color_invariance :
  ∀ (color : Fin 3 × Fin 7 → Bool), 
  ∃ i1 i2 j1 j2 : Fin 3, i1 < i2 ∧ j1 < j2 ∧ 
  color ⟨i1, j1⟩ = color ⟨i1, j2⟩ ∧ 
  color ⟨i1, j1⟩ = color ⟨i2, j1⟩ ∧ 
  color ⟨i1, j1⟩ = color ⟨i2, j2⟩ :=
by
  -- The proof is omitted
  sorry

end rectangle_color_invariance_l481_481525


namespace problem1_problem2_l481_481406

-- Problem 1: Prove A = π / 3
theorem problem1 (A B C a b c : ℝ)
  (h1 : 0 < A ∧ A < pi)
  (h2 : a = B * tan(A))
  (tan_eq : tan A / tan B = (2 * c - b) / b) :
  A = pi / 3 :=
sorry

-- Problem 2: Prove b / c = sqrt(6) - 1 given additional conditions
theorem problem2 (A B C a b c : ℝ)
  (h1 : 0 < A ∧ A < pi)
  (h2 : a = B * tan(A))
  (tan_eq : tan A / tan B = (2 * c - b) / b)
  (sin_cos_eq : sin (B + C) = 6 * cos B * sin C)
  (A_eq : A = pi / 3) :
  b / c = sqrt(6) - 1 :=
sorry

end problem1_problem2_l481_481406


namespace enclosed_area_of_graphs_l481_481485

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - x^2)

theorem enclosed_area_of_graphs :
  let r := 1 in
  let circle_area := real.pi * (r^2) in
  let enclosed_area := circle_area / 2 in
  (∫ x in -r..r, f x) = enclosed_area :=
by
  sorry

end enclosed_area_of_graphs_l481_481485


namespace monotonic_iff_m_ge_one_third_l481_481900

-- Define the function f(x) = x^3 + x^2 + mx + 1
def f (x m : ℝ) : ℝ := x^3 + x^2 + m * x + 1

-- Define the derivative of the function f w.r.t x
def f' (x m : ℝ) : ℝ := 3 * x^2 + 2 * x + m

-- State the main theorem: f is monotonic on ℝ if and only if m ≥ 1/3
theorem monotonic_iff_m_ge_one_third (m : ℝ) :
  (∀ x y : ℝ, x < y → f x m ≤ f y m) ↔ (m ≥ 1 / 3) :=
sorry

end monotonic_iff_m_ge_one_third_l481_481900


namespace find_set_A_l481_481599

noncomputable def A_condition (a1 a2 a3 a4 : ℤ) : Prop :=
  let A := {a1, a2, a3, a4}
  let B := {-1, 3, 5, 8}
  (∃ (b₁ b₂ b₃ : ℤ), A ⊆ {b₁, b₂, b₃} ∧ (b₁ + b₂ + b₃ ∈ B)) ∧
  3 * (a1 + a2 + a3 + a4) = -1 + 3 + 5 + 8

theorem find_set_A (a1 a2 a3 a4 : ℤ) :
  A_condition a1 a2 a3 a4 → {a1, a2, a3, a4} = {-3, 0, 2, 6} :=
by
  intro h
  /- proof skipped -/
  sorry

end find_set_A_l481_481599


namespace watermelons_with_seeds_l481_481177

def ripe_watermelons : ℕ := 11
def unripe_watermelons : ℕ := 13
def seedless_watermelons : ℕ := 15
def total_watermelons := ripe_watermelons + unripe_watermelons

theorem watermelons_with_seeds :
  total_watermelons - seedless_watermelons = 9 :=
by
  sorry

end watermelons_with_seeds_l481_481177


namespace first_test_point_second_test_point_l481_481164

noncomputable def feed_rates : List ℝ := 
  [0.30, 0.33, 0.35, 0.40, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65, 0.71, 0.81, 0.91]

theorem first_test_point :
  (feed_rates.nth 6).get_or_else 0 = 0.50 :=
by {
  sorry
}

theorem second_test_point :
  ((feed_rates.nth 2).get_or_else 0 + (feed_rates.nth 3).get_or_else 0) / 2 = 0.375 :=
by {
  sorry
}

end first_test_point_second_test_point_l481_481164


namespace variance_changes_when_adding_3_l481_481633

def original_data_set : List ℕ := [1, 3, 3, 5]

def new_data_set : List ℕ := [1, 3, 3, 3, 5]

def mean (data : List ℕ) : ℚ := (data.foldl (· + ·) 0) / data.length

def median (data : List ℕ) : ℚ := 
  let sorted_data := data.qsort (· ≤ ·)
  if data.length % 2 = 0 then
    (sorted_data.nth! (data.length / 2 - 1) + sorted_data.nth! (data.length / 2)) / 2
  else 
    sorted_data.nth! (data.length / 2)

def mode (data : List ℕ) : ℕ := data.foldl (λ acc x -> if data.count x > data.count acc then x else acc) 0

def variance (data : List ℕ) : ℚ := 
  let μ := mean data
  (data.foldl (λ acc x -> acc + (x - μ)^2) 0) / data.length

theorem variance_changes_when_adding_3 :
  variance new_data_set ≠ variance original_data_set :=
sorry

end variance_changes_when_adding_3_l481_481633


namespace ratio_BD_BO_l481_481138

open EuclideanGeometry

namespace CircleProblem

theorem ratio_BD_BO (A B C O D : Point)
  (h1: Circle O contains A ∧ Circle O contains C) 
  (h2: Tangent B A (Circle O) ∧ Tangent B C (Circle O)) 
  (h3: IsIsoscelesTriangle A B C)
  (h4: ∡BAC = 80) 
  (h5: ∃ D, Line B O intersects (Circle O) at D) :
  ratio (Segment B D) (Segment B O) = 0.3572 := 
sorry

end CircleProblem

end ratio_BD_BO_l481_481138


namespace probability_cocaptains_l481_481914

theorem probability_cocaptains (team_sizes : List ℕ)
  (h_sizes : team_sizes = [5, 7, 8])
  (num_cocaptains : ∀ (n : ℕ), n ∈ team_sizes → n ≥ 2) :
  let probability_cocaptains (team_size : ℕ) : ℚ :=
    2 / (team_size * (team_size - 1))
  let total_probability : ℚ :=
    (1 / 3) * (probability_cocaptains 5 + probability_cocaptains 7 + probability_cocaptains 8)
  total_probability = 11 / 180 :=
by
  sorry

end probability_cocaptains_l481_481914


namespace original_triangle_area_quadrupled_l481_481162

theorem original_triangle_area_quadrupled {A : ℝ} (h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64)) : A = 4 :=
by
  have h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64) := by
    intro a ha
    sorry
  sorry

end original_triangle_area_quadrupled_l481_481162


namespace probability_of_event_E_l481_481213

-- Define the event E as rolling a number less than 5
def event_E (n : ℕ) : Prop := n < 5

-- Define the sample space as the set {1, 2, ..., 8} for an 8-sided die
def sample_space : set ℕ := {n | n ≥ 1 ∧ n ≤ 8}

-- Define the probability function
def probability (E : set ℕ) (S : set ℕ) : ℚ :=
  if S.nonempty then
    E.to_finset.card.to_rat / S.to_finset.card.to_rat
  else 0

-- Prove that the probability of event E is 1/2 given the sample space
theorem probability_of_event_E :
  probability {n | event_E n} sample_space = 1 / 2 := by
  sorry

end probability_of_event_E_l481_481213


namespace cost_of_video_game_console_l481_481878

-- Define the problem conditions
def earnings_Mar_to_Aug : ℕ := 460
def hours_Mar_to_Aug : ℕ := 23
def earnings_per_hour : ℕ := earnings_Mar_to_Aug / hours_Mar_to_Aug
def hours_Sep_to_Feb : ℕ := 8
def cost_car_fix : ℕ := 340
def additional_hours_needed : ℕ := 16

-- Proof that the cost of the video game console is $600
theorem cost_of_video_game_console :
  let initial_earnings := earnings_Mar_to_Aug
  let earnings_from_Sep_to_Feb := hours_Sep_to_Feb * earnings_per_hour
  let total_earnings_before_expenses := initial_earnings + earnings_from_Sep_to_Feb
  let current_savings := total_earnings_before_expenses - cost_car_fix
  let earnings_after_additional_work := additional_hours_needed * earnings_per_hour
  let total_savings := current_savings + earnings_after_additional_work
  total_savings = 600 :=
by
  sorry

end cost_of_video_game_console_l481_481878


namespace number_of_proper_subsets_of_A_l481_481338

def A : Set ℕ := {x | 0 ≤ x ∧ x < 3}

theorem number_of_proper_subsets_of_A : 
  (A = {0, 1, 2}) → (Fintype.card (Set.univ \ {S | S ∈ ({A} : Set (Set ℕ))}) = 7) :=
by
  sorry

end number_of_proper_subsets_of_A_l481_481338


namespace sum_of_all_x_l481_481579

theorem sum_of_all_x (x1 x2 : ℝ) (h1 : (x1 + 5)^2 = 81) (h2 : (x2 + 5)^2 = 81) : x1 + x2 = -10 :=
by
  sorry

end sum_of_all_x_l481_481579


namespace john_height_in_feet_l481_481811

theorem john_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (months : ℕ) (inches_per_foot : ℕ) :
  initial_height = 66 → growth_rate = 2 → months = 3 → inches_per_foot = 12 → 
  (initial_height + growth_rate * months) / inches_per_foot = 6 := by
  intros h1 h2 h3 h4
  sorry

end john_height_in_feet_l481_481811


namespace domain_of_sqrt_x_ln_2_minus_x_l481_481163

theorem domain_of_sqrt_x_ln_2_minus_x :
  {x : ℝ | x ≥ 0 ∧ 2 - x > 0} = set.Ico 0 2 :=
by
  sorry

end domain_of_sqrt_x_ln_2_minus_x_l481_481163


namespace min_value_n_minus_m_l481_481737

noncomputable def f (x : ℝ) : ℝ :=
  if 1 < x then Real.log x else (1 / 2) * x + (1 / 2)

theorem min_value_n_minus_m (m n : ℝ) (hmn : m < n) (hf_eq : f m = f n) : n - m = 3 - 2 * Real.log 2 :=
  sorry

end min_value_n_minus_m_l481_481737


namespace number_of_women_is_24_l481_481446

-- Define the variables and conditions
variables (x : ℕ) (men_initial : ℕ) (women_initial : ℕ) (men_current : ℕ) (women_current : ℕ)

-- representing the initial ratio and the changes
def initial_conditions : Prop :=
  men_initial = 4 * x ∧ women_initial = 5 * x ∧
  men_current = men_initial + 2 ∧ women_current = 2 * (women_initial - 3)

-- representing the current number of men
def current_men_condition : Prop := men_current = 14

-- The proof we need to generate
theorem number_of_women_is_24 (x : ℕ) (men_initial women_initial men_current women_current : ℕ)
  (h1 : initial_conditions x men_initial women_initial men_current women_current)
  (h2 : current_men_condition men_current) : women_current = 24 :=
by
  -- proof steps here
  sorry

end number_of_women_is_24_l481_481446


namespace greatest_air_conditioning_but_no_racing_stripes_l481_481594

variable (total_cars : ℕ) (no_air_conditioning_cars : ℕ) (at_least_racing_stripes_cars : ℕ)
variable (total_cars_eq : total_cars = 100)
variable (no_air_conditioning_cars_eq : no_air_conditioning_cars = 37)
variable (at_least_racing_stripes_cars_ge : at_least_racing_stripes_cars ≥ 51)

theorem greatest_air_conditioning_but_no_racing_stripes
  (total_cars_eq : total_cars = 100)
  (no_air_conditioning_cars_eq : no_air_conditioning_cars = 37)
  (at_least_racing_stripes_cars_ge : at_least_racing_stripes_cars ≥ 51) :
  ∃ max_air_conditioning_no_racing_stripes : ℕ, max_air_conditioning_no_racing_stripes = 12 :=
by
  sorry

end greatest_air_conditioning_but_no_racing_stripes_l481_481594


namespace total_cars_for_sale_l481_481288

-- Define the conditions given in the problem
def salespeople : Nat := 10
def cars_per_salesperson_per_month : Nat := 10
def months : Nat := 5

-- Statement to prove the total number of cars for sale
theorem total_cars_for_sale : (salespeople * cars_per_salesperson_per_month) * months = 500 := by
  -- Proof goes here
  sorry

end total_cars_for_sale_l481_481288


namespace number_of_even_factors_l481_481048

-- Define the integer n based on the given prime factorization
def n : ℕ := 2^3 * 3^2 * 5^1 * 7^3

-- Define the conditions for the factors
def is_factor_even (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 3

-- Define the set of even factors based on the conditions
def even_factors_set : set ℕ :=
  { k | ∃ a b c d, k = 2^a * 3^b * 5^c * 7^d ∧ is_factor_even a b c d }

-- The statement to be proved
theorem number_of_even_factors : even_factors_set.to_finset.card = 72 :=
  by
    sorry

end number_of_even_factors_l481_481048


namespace range_area_triangle_PAB_l481_481126

noncomputable def f : ℝ → ℝ
| x => if 0 < x ∧ x < 1 then -real.log x else if 1 < x then real.log x else 0

theorem range_area_triangle_PAB :
  ∀ (x1 x2 : ℝ), 
  0 < x1 ∧ x1 < 1 ∧ 1 < x2  ∧ x1 * x2 = 1 → 
  let l1 := -(1 / x1) in
  let l2 := (1 / x2) in
  let y1 := -real.log x1 in
  let y2 := real.log x2 in
  let A := (0, 1 - y1) in
  let B := (0, -1 + y2) in
  let P := (2 * x1 * x2 / (x1 + x2), 0) in
  let AB := abs (2 - (real.log x1 + real.log x2)) in
  let area := (1/2) * AB * (2 * x1 * x2 / (x1 + x2)) in
  0 < area ∧ area < 1 :=
by
  intros
  sorry

end range_area_triangle_PAB_l481_481126


namespace cone_base_circumference_l481_481615

theorem cone_base_circumference 
  (r : ℝ) 
  (θ : ℝ) 
  (h₁ : r = 5) 
  (h₂ : θ = 225) : 
  (θ / 360 * 2 * Real.pi * r) = (25 * Real.pi / 4) :=
by
  -- Proof skipped
  sorry

end cone_base_circumference_l481_481615


namespace minimum_value_of_expression_l481_481833

theorem minimum_value_of_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 3) : 
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 72 :=
sorry

end minimum_value_of_expression_l481_481833


namespace units_digit_of_large_power_l481_481582

theorem units_digit_of_large_power
  (units_147_1997_pow2999: ℕ) 
  (h1 : units_147_1997_pow2999 = (147 ^ 1997) % 10)
  (h2 : ∀ k, (7 ^ (k * 4 + 1)) % 10 = 7)
  (h3 : ∀ m, (7 ^ (m * 4 + 3)) % 10 = 3)
  : units_147_1997_pow2999 % 10 = 3 :=
sorry

end units_digit_of_large_power_l481_481582


namespace total_hours_over_two_weeks_l481_481997

-- Define the conditions of Bethany's riding schedule
def hours_per_week : ℕ :=
  1 * 3 + -- Monday, Wednesday, and Friday
  (30 / 60) * 2 + -- Tuesday and Thursday, converting minutes to hours
  2 -- Saturday

-- The theorem to prove the total hours over 2 weeks
theorem total_hours_over_two_weeks : hours_per_week * 2 = 12 := 
by
  -- Proof to be completed here
  sorry

end total_hours_over_two_weeks_l481_481997


namespace area_4g_shifted_l481_481909

-- Define the function g : ℝ → ℝ
variable (g : ℝ → ℝ)

-- Define the given condition about the integral of g
axiom area_g (h : ∀ x, g x >= 0) : ∫ x in -∞..∞, g x = 12

-- Define the statement we want to prove
theorem area_4g_shifted (h : ∀ x, g x >= 0) : 
  ∫ x in -∞..∞, 4 * g (x - 3) + 2 = 48 :=
by
  sorry

end area_4g_shifted_l481_481909


namespace fraction_identity_l481_481389

theorem fraction_identity (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : ((m + n) : ℚ) / n = 10 / 7 := 
sorry

end fraction_identity_l481_481389


namespace ratio_OM_PC_eq_one_half_l481_481091

variables {A B C M P O : Type}
variables {distance : A -> A -> ℝ}
variables [Euclidean_space A]

-- Conditions
variables {M_midpoint_AC : midpoint M A C}
variables {P_on_BC : on_line P B C}
variables {intersects : intersect AP BM O }
variables {BO_eq_BP : distance B O = distance B P }

-- Theorem to be proved
theorem ratio_OM_PC_eq_one_half : 
  ∀ (A B C M P O : A)
  (distance : A -> A -> ℝ)
  (M_midpoint_AC : midpoint M A C)
  (P_on_BC : on_line P B C)
  (intersects : intersect AP BM O)
  (BO_eq_BP : distance B O = distance B P), 
  distance O M / distance P C = 1 / 2 := 
by 
  sorry

end ratio_OM_PC_eq_one_half_l481_481091


namespace sufficient_condition_perpendicular_l481_481831

-- Definitions of perpendicularity and lines/planes intersections
variables {Plane : Type} {Line : Type}

variable (α β γ : Plane)
variable (m n l : Line)

-- Axioms representing the given conditions
axiom perp_planes (p₁ p₂ : Plane) : Prop -- p₁ is perpendicular to p₂
axiom perp_line_plane (line : Line) (plane : Plane) : Prop -- line is perpendicular to plane

-- Given conditions for the problem.
axiom n_perp_α : perp_line_plane n α
axiom n_perp_β : perp_line_plane n β
axiom m_perp_α : perp_line_plane m α

-- The proposition to be proved.
theorem sufficient_condition_perpendicular (h₁ : perp_line_plane n α)
                                           (h₂ : perp_line_plane n β)
                                           (h₃ : perp_line_plane m α) :
  perp_line_plane m β := sorry

end sufficient_condition_perpendicular_l481_481831


namespace point_with_largest_angle_on_perpendicular_leg_l481_481531

theorem point_with_largest_angle_on_perpendicular_leg 
  (a c b d : ℝ) 
  (h_area : (a + c) * d = b * d)
  (h_mid : ∃ (G : ℝ × ℝ), G = ((ad_x + 0) / 2, (d + 0) / 2)) :
  ∃ G : ℝ × ℝ, 
    (G = ((ad_x + 0) / 2, (d + 0) / 2) ∧ ∀ (P : ℝ × ℝ), P ≠ G → 
      ∃ θ : ℝ, θ = angle_between_legs P (ad_x, d) (bc_x, b) → θ < π / 2) :=
by
  sorry

end point_with_largest_angle_on_perpendicular_leg_l481_481531


namespace triangle_inequality_example_l481_481916

theorem triangle_inequality_example (a b c : ℝ) (ha : a = 30) (hb : b = 50) (hc : c = 30) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by {
  rw [ha, hb, hc],
  simp,
  apply and.intro,
  { linarith, },
  apply and.intro,
  { linarith, },
  { linarith, }
}

end triangle_inequality_example_l481_481916


namespace area_of_plane_region_l481_481359

-- Define the system of inequalities as a condition (dummy example as specifics are not provided)
def within_plane_region (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b < 2

-- Define the point N in terms of a and b
def point_N (a b : ℝ) : ℝ × ℝ :=
  (a + b, a - b)

-- Statement that asserts the area of the region where point N lies is 4
theorem area_of_plane_region (a b : ℝ) (h : within_plane_region a b) : 
  area (plane_region (point_N a b)) = 4 :=
sorry

end area_of_plane_region_l481_481359


namespace min_value_f_on_interval_l481_481168

-- Given conditions
def f (x : ℝ) : ℝ := 
  (real.sqrt 2) * real.sin x * (real.cos x + real.sin x) - (real.sqrt 2) / 2

-- Problem definition: Find the minimum value of f(x) over the interval [0, π/2]
theorem min_value_f_on_interval : 
  ∃ x ∈ set.Icc (0:ℝ) (real.pi/2), f x = - (real.sqrt 2) / 2 :=
by sorry

end min_value_f_on_interval_l481_481168


namespace continuous_paths_A_to_B_no_revisit_l481_481748

-- Definitions of the points
constant A B C D E F : Type

-- Definition of the segments
constant segment : Type
constant segments : segment → segment → Prop

-- The specified segments of the figure
constant segments_AB_CD_AC_AD_AE_AF_BF : Prop

-- Condition: Paths are continuous along the given segments and do not revisit points
constant is_continuous_no_revisit : List segment → Prop

-- The figure's specific segments
axiom figure_segments : segments AB AC ∧ segments AC AD ∧ segments AD AE ∧
                        segments AE AF ∧ segments AF BF

-- The main theorem to prove
theorem continuous_paths_A_to_B_no_revisit : 
  ∃ num_paths : ℕ, (num_paths = 10) ∧
  (∃ paths : List (List segment), 
   ∀ path ∈ paths, 
     is_continuous_no_revisit path →
     (path.head = A ∧ path.reverse.head = B) ∧
     List.length paths = num_paths) :=
sorry

end continuous_paths_A_to_B_no_revisit_l481_481748


namespace find_k_find_lambda_l481_481383

-- Definitions
def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (3, 4)

-- Question 1: Prove that k = -1/2 if 2a - b is parallel to a + kb
theorem find_k (k : ℝ) : 
  (let v1 := (2 * vec_a.1 - vec_b.1, 2 * vec_a.2 - vec_b.2) in
   let v2 := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2) in
   ∃ c : ℝ, (v1.1 = c * v2.1) ∧ (v1.2 = c * v2.2)) ↔ k = -1/2 :=
by
  sorry

-- Question 2: Prove that λ = 1 if the projection of λa onto b is 1
theorem find_lambda (λ : ℝ) :
  (let dot_product := λ * vec_a.1 * vec_b.1 + λ * vec_a.2 * vec_b.2 in
   dot_product / (vec_b.1^2 + vec_b.2^2) = 1) ↔ λ = 1 :=
by
  sorry

end find_k_find_lambda_l481_481383


namespace math_problem_l481_481899
-- Initial problem and solution analysis:

-- Problem:
-- (1) Determine whether (1+i)/(1-i) ∈ M, where M = {m | m = i^2, n ∈ N*}
-- (2) Determine whether p ∨ q is true. 
--       p: The graph of f(x) = a^x - 2 (a > 0, a ≠ 1) always passes through (0, -2)
--       q: The function f(x) = lg|x| (x ≠ 0) has two zeros
-- (3) Find the maximum slope of the tangent line to f(x) = e^(-x) - e^x
-- (4) Determine if there exists an irrational x_0 such that x_0^2 is irrational.

-- Solution:
-- (1) (1+i)/(1-i) = i, so it is not in M. False.
-- (2) p is true since f(0) = -2, and q is true since f(x) has two zeros at x=±1. True.
-- (3) The maximum slope f'(x) = -e^(-x) - e^x is -2, not 2. So false.
-- (4) Example: x_0 = √2, then x_0^2 = 2, which is irrational. True.
-- Correct propositions are (2) and (4). Answer: (2), (4).

-- Steps of conversion from problem and solution to Lean problem statement:

-- Step a) Identify all questions and conditions:
-- Questions:
-- (1) Whether (1+i)/(1-i) belongs to M?
-- (2) Whether p ∨ q is true?
-- (3) What is the maximum slope of the tangent line of f(x) = e^(-x) - e^x?
-- (4) Whether there exists an irrational x_0 such that x_0^2 is irrational?

-- Conditions:
-- M = {m | m = i^2, n ∈ N* }
-- f(x) = a^x - 2
-- a > 0, a ≠ 1
-- f(x) = lg|x|
-- x ≠ 0
-- f(x) = e^(-x) - e^x

-- Step b) Identify solution steps and correct answers:
-- Solution steps:
-- Simplified complex number to find if (1+i)/(1-i) ∈ M.
-- Verified properties of function graphs to determine truth of p ∨ q.
-- Calculated derivative to find maximum slope.
-- Provided an example of irrational x_0.
-- Correct answers: False, True, False, True.

-- Step c) Translate to mathematically equivalent proof problem:
-- Prove that propositions (2) and (4) are true while (1) and (3) are false.

-- Step d) Rewrite the problem statement in Lean:


open Complex Real

-- Define the set M
def M : Set ℂ := {m : ℂ | ∃ (n : ℕ) (hn : n > 0), m = Complex.i ^ 2}

-- Define the conditions p and q for (2)
def p (a : ℝ) (ha : a > 0 ∧ a ≠ 1) : Prop := ∀ x, x = 0 → a^x - 2 = -2
def q : Prop := (log 1 = 0) ∨ (log (-1) = 0)

-- Define the function for (3)
def f (x : ℝ) : ℝ := exp (-x) - exp x

-- Define the condition for (4)
def exists_irrational_square_is_irrational : Prop := ∃ (x : ℝ), ¬ (rational x) ∧ ¬ (rational (x^2))

theorem math_problem :
  (¬ ((1 + Complex.i) / (1 - Complex.i) ∈ M)) ∧
  (∃ a, (a > 0 ∧ a ≠ 1) ∧ p a (by sorry)) ∧
  q ∧
  (¬ ((∃ x, deriv (λ x, f x) x = -2))) ∧
  exists_irrational_square_is_irrational :=
by sorry

end math_problem_l481_481899


namespace number_of_women_l481_481442

theorem number_of_women (x : ℕ) 
  (h1 : 4 * x + 2 = 14) : 2 * (5 * x - 3) = 24 :=
by 
  ext
  sorry

end number_of_women_l481_481442


namespace range_of_a_l481_481019

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Proposition P: f(x) has a root in the interval [-1, 1]
def P (a : ℝ) : Prop := ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0

-- Proposition Q: There is only one real number x satisfying the inequality
def Q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

-- The theorem stating the range of a if either P or Q is false
theorem range_of_a (a : ℝ) : ¬(P a) ∨ ¬(Q a) → (a > -1 ∧ a < 0) ∨ (a > 0 ∧ a < 1) :=
sorry

end range_of_a_l481_481019


namespace golden_skew_line_pairs_in_cube_l481_481064

def isGoldenSkewLinePair (l1 l2 : ℝ -> ℝ^3) : Prop :=
  ∃ θ : ℝ, θ = 60 ∧ skew_line l1 ∧ skew_line l2 ∧ angle l1 l2 = θ

def number_of_golden_skew_line_pairs (edges : Finset (ℝ -> ℝ^3)) : Nat :=
  (edges.to_list.pairs.filter (λ (l1, l2), isGoldenSkewLinePair l1 l2)).length

theorem golden_skew_line_pairs_in_cube (s : ℝ) (cube_edges : Finset (ℝ -> ℝ^3)) 
    (h : cube_edges.card = 12) : 
  number_of_golden_skew_line_pairs cube_edges = 24 :=
by
  sorry

end golden_skew_line_pairs_in_cube_l481_481064


namespace monotonic_decreasing_interval_l481_481012

-- Condition: The slope of the tangent line at any point (x0, f(x0)) 
-- is given by k = (x0 - 3) * (x0 + 1)^2
def slope (x0 : ℝ) : ℝ := (x0 - 3) * (x0 + 1)^2

-- The function f(x) is monotonically decreasing in the interval (-∞, 3]
theorem monotonic_decreasing_interval :
  ∀ x0 : ℝ, slope x0 < 0 → x0 < 3 :=
sorry

end monotonic_decreasing_interval_l481_481012


namespace trigonometric_truths_l481_481928

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry
noncomputable def tan (x : ℝ) : ℝ := sorry

theorem trigonometric_truths :
  (sin(62 * (π / 180)) * cos(32 * (π / 180)) - cos(62 * (π / 180)) * sin(32 * (π / 180)) = 1 / 2) ∧
  ¬ (sin(75 * (π / 180)) * cos(75 * (π / 180)) = sqrt 3 / 4) ∧
  ¬ ((1 + tan(75 * (π / 180))) / (1 - tan(75 * (π / 180))) = sqrt 3) ∧
  (sin(50 * (π / 180)) * (sqrt 3 * sin(10 * (π / 180)) + cos(10 * (π / 180))) / 
  cos(10 * (π / 180)) = 1) := by
  -- Proof goes here
  sorry

end trigonometric_truths_l481_481928


namespace probability_of_rolling_less_than_5_l481_481219

theorem probability_of_rolling_less_than_5 :
  (let outcomes := 8 in
   let successes := 4 in
   let probability := (successes: ℚ) / outcomes in
   probability = 1 / 2) :=
by
  sorry

end probability_of_rolling_less_than_5_l481_481219


namespace prisoners_release_strategy_exists_l481_481817

theorem prisoners_release_strategy_exists :
  ∀ (n : ℕ) (figures : list ℕ),
    (3 ≤ n) ∧ (∀ f ∈ figures, 1 ≤ list.count figures f) ∧ 
    (∀ f g ∈ figures, f ≠ g → list.count figures f ≠ list.count figures g) →
    (∃ s : ℕ, s ∈ figures ∧ 
      ∀ p : fin n,
        let visible_figures := (list.erase figures (figures.get p)) in
        let most_frequent := list.maximum visible_figures in
        figures.get p = most_frequent) →
    ∃ (p : fin n), figures.get p ∈ figures :=
begin 
  sorry 
end

end prisoners_release_strategy_exists_l481_481817


namespace A_works_alone_45_days_l481_481253

open Nat

theorem A_works_alone_45_days (x : ℕ) :
  (∀ x : ℕ, (9 * (1 / x + 1 / 40) + 23 * (1 / 40) = 1) → (x = 45)) :=
sorry

end A_works_alone_45_days_l481_481253


namespace maximum_sum_Kostya_can_obtain_l481_481511

-- Define the initial setup and the constraints.
def frog_jumps_max_sum (table : ℕ → ℕ → ℕ) : Prop :=
  ∃ (frogs : Fin 5 → ℕ × ℕ)
  (initial_visible_sum : ℕ)
  (subsequent_sums : List ℕ),
    -- 1. The initial sum of visible numbers is 10
    initial_visible_sum = 10 ∧
    -- 2. The sums grow by a factor of 10 each jump up to the maximum sum
    subsequent_sums = [10^2, 10^3, 10^4, 10^5, 10^6] ∧
    -- 3. Frogs jump to adjacent squares
    ∀ k: ℕ, k < 5 → 
      let previous_sum := if k = 0 then initial_visible_sum else subsequent_sums.nth (k-1)
      let current_sum := subsequent_sums.nth k
      current_sum = (previous_sum * 10)

theorem maximum_sum_Kostya_can_obtain :
  ∀ table,
  frog_jumps_max_sum table →
  ∃ s, s = 10^6 :=
by 
  intro table h
  cases h with frogs initial_visible_sum subsequent_sums
  existsi 10^6
  sorry

end maximum_sum_Kostya_can_obtain_l481_481511


namespace isabel_earned_l481_481093

theorem isabel_earned :
  let bead_necklace_price := 4
  let gemstone_necklace_price := 8
  let bead_necklace_count := 3
  let gemstone_necklace_count := 3
  let sales_tax_rate := 0.05
  let discount_rate := 0.10

  let total_cost_before_tax := bead_necklace_count * bead_necklace_price + gemstone_necklace_count * gemstone_necklace_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  let discount := total_cost_after_tax * discount_rate
  let final_amount_earned := total_cost_after_tax - discount

  final_amount_earned = 34.02 :=
by {
  sorry
}

end isabel_earned_l481_481093


namespace eccentricity_of_given_hyperbola_l481_481738

open Real

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : ∀ c : ℝ, c = 4) (dot_product : (-a, -b).1 * (4, -b).1 + (-a, -b).2 * (4, -b).2 = 2 * a) (relation : 4 ^ 2 = a ^ 2 + b ^ 2) : ℝ :=
  c / a

theorem eccentricity_of_given_hyperbola :
  ∀ (a b : ℝ), a > 0 → b > 0 → (∀ c : ℝ, c = 4) → (-a, -b).1 * (4, -b).1 + (-a, -b).2 * (4, -b).2 = 2 * a → 4 ^ 2 = a ^ 2 + b ^ 2 → eccentricity_of_hyperbola a b := 
  sorry

end eccentricity_of_given_hyperbola_l481_481738


namespace log_inequality_l481_481754

noncomputable def m : ℝ := sorry
noncomputable def a : ℝ := log m
noncomputable def b : ℝ := log (m^2)
noncomputable def c : ℝ := log (m^3)

theorem log_inequality
  (h_m_range : m ∈ set.Ioo (1/10 : ℝ) 1) :
  b < a ∧ a < c := 
sorry

end log_inequality_l481_481754


namespace prob_at_least_one_multiple_of_4_correct_l481_481500

-- defining the set of integers from 1 to 60
def numbers := { n : ℕ | 1 ≤ n ∧ n ≤ 60 }

-- defining what it means for a number to be a multiple of 4
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

-- counting the multiples of 4 in the set
def multiples_of_4 := { n ∈ numbers | is_multiple_of_4 n }

-- probability of choosing a number that is not a multiple of 4
def prob_not_multiple_of_4 := (numbers.card - multiples_of_4.card) / numbers.card

-- probability that in two random choices, neither is a multiple of 4
def prob_neither_multiple_of_4 := prob_not_multiple_of_4 * prob_not_multiple_of_4

-- probability that at least one is a multiple of 4
def prob_at_least_one_multiple_of_4 := 1 - prob_neither_multiple_of_4

-- The final theorem statement
theorem prob_at_least_one_multiple_of_4_correct :
  prob_at_least_one_multiple_of_4 = 7 / 16 :=
sorry

end prob_at_least_one_multiple_of_4_correct_l481_481500


namespace circle_condition_l481_481027

variables {A B C D E F : ℝ}
variable (hD : D ≠ 0)

lemma necessary_but_not_sufficient_condition (h_eq: A = C ∧ B = 0) :
  (Ax : ℝ) + (Ay : ℝ) + (Cy : ℝ) + (Dx : ℝ) + (Ey : ℝ) + (F : ℝ) ≠ 0 :=
sorry

theorem circle_condition (h_circle : ∀ x y, Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0) :
  A = C ∧ B = 0 :=
sorry

end circle_condition_l481_481027


namespace complement_of_A_in_U_l481_481844

def U : Set ℕ := {1, 2, 3, 4}

def satisfies_inequality (x : ℕ) : Prop := x^2 - 5 * x + 4 < 0

def A : Set ℕ := {x | satisfies_inequality x}

theorem complement_of_A_in_U : U \ A = {1, 4} :=
by
  -- Proof omitted.
  sorry

end complement_of_A_in_U_l481_481844


namespace sum_of_x_and_y_l481_481397

theorem sum_of_x_and_y (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hprod : x * y = 555) : x + y = 52 :=
by
  sorry

end sum_of_x_and_y_l481_481397


namespace total_work_stations_l481_481612

theorem total_work_stations (total_students : ℕ) (stations_for_2 : ℕ) (stations_for_3 : ℕ)
  (h1 : total_students = 38)
  (h2 : stations_for_2 = 10)
  (h3 : 20 + 3 * stations_for_3 = total_students) :
  stations_for_2 + stations_for_3 = 16 :=
by
  sorry

end total_work_stations_l481_481612


namespace common_difference_of_arithmetic_sequence_l481_481415

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ) -- define the arithmetic sequence
  (h_arith : ∀ n : ℕ, a n = a 0 + n * 4) -- condition of arithmetic sequence
  (h_a5 : a 4 = 8) -- given a_5 = 8
  (h_a9 : a 8 = 24) -- given a_9 = 24
  : 4 = 4 := -- statement to be proven
by
  sorry

end common_difference_of_arithmetic_sequence_l481_481415


namespace sally_nickels_count_l481_481876

theorem sally_nickels_count (original_nickels dad_nickels mom_nickels : ℕ) 
    (h1: original_nickels = 7) 
    (h2: dad_nickels = 9) 
    (h3: mom_nickels = 2) 
    : original_nickels + dad_nickels + mom_nickels = 18 :=
by
  sorry

end sally_nickels_count_l481_481876


namespace shari_jogged_distance_l481_481518

theorem shari_jogged_distance (rate time : ℕ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 :=
by
  rw [h_rate, h_time]
  norm_num
  sorry

end shari_jogged_distance_l481_481518


namespace prisoners_release_strategy_exists_l481_481818

theorem prisoners_release_strategy_exists :
  ∀ (n : ℕ) (figures : list ℕ),
    (3 ≤ n) ∧ (∀ f ∈ figures, 1 ≤ list.count figures f) ∧ 
    (∀ f g ∈ figures, f ≠ g → list.count figures f ≠ list.count figures g) →
    (∃ s : ℕ, s ∈ figures ∧ 
      ∀ p : fin n,
        let visible_figures := (list.erase figures (figures.get p)) in
        let most_frequent := list.maximum visible_figures in
        figures.get p = most_frequent) →
    ∃ (p : fin n), figures.get p ∈ figures :=
begin 
  sorry 
end

end prisoners_release_strategy_exists_l481_481818


namespace total_fires_l481_481987

theorem total_fires (Doug_fires Kai_fires Eli_fires : ℕ)
  (h1 : Doug_fires = 20)
  (h2 : Kai_fires = 3 * Doug_fires)
  (h3 : Eli_fires = Kai_fires / 2) :
  Doug_fires + Kai_fires + Eli_fires = 110 :=
by
  sorry

end total_fires_l481_481987


namespace not_necessarily_parallel_planes_l481_481111

noncomputable theory

variables (α β : Type*) [plane α] [plane β]
variables (m n : Type*) [line m] [line n]

-- Definition of parallel and perpendicular for lines and planes
def parallel (p q : Type*) [relation p q] : Prop := sorry
def perpendicular (p q : Type*) [relation p q] : Prop := sorry

-- Given conditions based on the problem statement
axiom different_planes : α ≠ β
axiom different_lines : m ≠ n

axiom parallel_m_α : parallel m α
axiom parallel_n_α : parallel n α
axiom parallel_m_β : parallel m β
axiom parallel_n_β : parallel n β

-- Translation of the incorrect option D in Lean 4 statement
theorem not_necessarily_parallel_planes : ¬ parallel α β := sorry

end not_necessarily_parallel_planes_l481_481111


namespace jackson_earnings_l481_481095

def hourly_rate_usd : ℝ := 5
def hourly_rate_gbp : ℝ := 3
def hourly_rate_jpy : ℝ := 400

def hours_vacuuming : ℝ := 2
def sessions_vacuuming : ℝ := 2

def hours_washing_dishes : ℝ := 0.5
def hours_cleaning_bathroom := hours_washing_dishes * 3

def exchange_rate_gbp_to_usd : ℝ := 1.35
def exchange_rate_jpy_to_usd : ℝ := 0.009

def earnings_in_usd : ℝ := (hours_vacuuming * sessions_vacuuming * hourly_rate_usd)
def earnings_in_gbp : ℝ := (hours_washing_dishes * hourly_rate_gbp)
def earnings_in_jpy : ℝ := (hours_cleaning_bathroom * hourly_rate_jpy)

def converted_gbp_to_usd : ℝ := earnings_in_gbp * exchange_rate_gbp_to_usd
def converted_jpy_to_usd : ℝ := earnings_in_jpy * exchange_rate_jpy_to_usd

def total_earnings_usd : ℝ := earnings_in_usd + converted_gbp_to_usd + converted_jpy_to_usd

theorem jackson_earnings : total_earnings_usd = 27.425 := by
  sorry

end jackson_earnings_l481_481095


namespace squirrel_acorns_l481_481075

theorem squirrel_acorns :
  ∃ (c s r : ℕ), (4 * c = 5 * s) ∧ (3 * r = 4 * c) ∧ (r = s + 3) ∧ (5 * s = 40) :=
by
  sorry

end squirrel_acorns_l481_481075


namespace min_balls_to_ensure_30_same_color_l481_481560

namespace MathProof

variables (W B R : ℕ) (total_balls : ℕ := 100)

-- Condition: Total number of balls is 100
axiom total_balls_eq : W + B + R = total_balls 

-- Condition: Drawing 26 balls guarantees at least 10 of the same color
axiom draw_26_balls_guarantee : ∀ (drawn_w drawn_b drawn_r : ℕ), 
  drawn_w + drawn_b + drawn_r = 26 → 
  drawn_w ≥ 10 ∨ drawn_b ≥ 10 ∨ drawn_r ≥ 10

-- Proof problem: Minimum number of balls needed to ensure drawing at least 30 of the same color
theorem min_balls_to_ensure_30_same_color : 
  (∀ (drawn_w drawn_b drawn_r : ℕ), 
    drawn_w + drawn_b + drawn_r = 88 → 
    drawn_w ≥ 30 ∨ drawn_b ≥ 30 ∨ drawn_r ≥ 30)
sorry

end MathProof

end min_balls_to_ensure_30_same_color_l481_481560


namespace proof_geometric_set_is_sphere_l481_481576

noncomputable def geometric_set_property (S : set ℝ) : Prop :=
  ∃ (A B C D : ℝ), 
  A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ 
  ¬ (A = B ∧ B = C ∧  C = D) ∧ 
  (∀ (x y z : ℝ), x ∈ S ∧ y ∈ S ∧ z ∈ S → ∃ (φ : ℝ), x ∈ φ ∧ y ∈ φ ∧ z ∈ φ ∧ φ ⊆ S)

theorem proof_geometric_set_is_sphere (S : set ℝ) (h1 : geometric_set_property S) : 
  ∃ (sphere : set ℝ), (∀ (P : ℝ), P ∈ sphere ↔ P ∈ S) ∧ 
  ∃ (O R : ℝ), (∀ (P : ℝ), P ∈ sphere ↔ (P - O)^2 = R^2) :=
sorry

end proof_geometric_set_is_sphere_l481_481576


namespace division_addition_rational_eq_l481_481570

theorem division_addition_rational_eq :
  (3 / 7 / 4) + (1 / 2) = 17 / 28 :=
by
  sorry

end division_addition_rational_eq_l481_481570


namespace women_current_in_room_l481_481454

-- Definitions
variable (m w : ℕ) -- number of men and women
variable (x : ℕ) -- positive integer representing the scaling factor for initial ratio

-- Conditions
def initial_ratio (x : ℕ) : Prop := m = 4*x ∧ w = 5*x
def after_changes (x : ℕ) : Prop := (m + 2) = 14 ∧ (2 * (w - 3)) = 24

-- Theorem statement
theorem women_current_in_room (x : ℕ) (m w : ℕ) (h1 : initial_ratio x) (h2 : after_changes x) : w = 24 :=
by
  sorry

end women_current_in_room_l481_481454


namespace family_friends_eat_percentage_l481_481514

-- Definitions of the conditions
definition num_pies : ℕ := 2
definition slices_per_pie : ℕ := 8
definition slices_initially_consumed_by_rebecca : ℕ := 2
definition slices_consumed_by_rebecca_and_husband_on_sunday : ℕ := 2
definition slices_remaining : ℕ := 5

-- Statement to prove
theorem family_friends_eat_percentage :
    let total_slices := num_pies * slices_per_pie,
        slices_left_after_initial := total_slices - slices_initially_consumed_by_rebecca,
        slices_left_after_sunday := slices_left_after_initial - slices_consumed_by_rebecca_and_husband_on_sunday,
        slices_eaten_by_friends_and_family := slices_left_after_sunday - slices_remaining,
        percentage_eaten_by_friends_and_family := (slices_eaten_by_friends_and_family * 100) / slices_left_after_initial
    in percentage_eaten_by_friends_and_family = 50 := 
by 
  sorry

end family_friends_eat_percentage_l481_481514


namespace solve_s_l481_481307

def F (a b c : ℕ) : ℕ := a * b^(c + 1)

theorem solve_s : ∃ s : ℕ, F s s 2 = 1296 ∧ s = 6 := 
by {
  -- Conditions
  unfold F,
  -- Given F(s, s, 2) = 1296,
  -- Show that s = 6
  sorry
}

end solve_s_l481_481307


namespace saturday_ice_cream_l481_481278

variable (friday total : ℝ)
-- Conditions
axiom friday_eq : friday = 3.25
axiom total_eq : total = 3.5

-- Question and correct answer
theorem saturday_ice_cream (saturday : ℝ) (h : saturday = total - friday) : 
  saturday = 0.25 :=
by
  have h1 : friday = 3.25 := friday_eq
  have h2 : total = 3.5 := total_eq
  rw [h1, h2] at h
  exact h

end saturday_ice_cream_l481_481278


namespace intersection_point_k_value_l481_481499

theorem intersection_point_k_value :
  (∃ (k : ℝ), (∀ (x y : ℝ),
    ((y = 2 * x + 3 ∧ y = k * x + 2) → (x = 1 ∧ y = 5))) → k = 3) :=
sorry

end intersection_point_k_value_l481_481499


namespace cyclic_quadrilateral_iff_conditions_l481_481011

variable {A B C D P Q : Type}

-- Definitions and assumptions of conditions
variables [ConvexQuadrilateral A B C D]
variables [IntersectRay A B C D P] [IntersectRay B C A D Q]

-- Statement to prove
theorem cyclic_quadrilateral_iff_conditions :
  (Cyclic A B C D) ↔ (AB + CD = BC + AD ∨ AP + CQ = AQ + CP ∨ BP + BQ = DP + DQ) :=
sorry

end cyclic_quadrilateral_iff_conditions_l481_481011


namespace geometric_sequence_formula_l481_481729

variable (a_n : ℕ → ℝ)
variable (a_1 : ℝ) (a_2 : ℝ)
variable (S_6 : ℝ)
variable (q : ℝ)

axiom geometric_sum_condition : S_6 = 21
axiom arithmetic_sequence_condition :  2 * (3 / 2) * a_2 = 4 * a_1 + a_2
axiom geometric_sequence_a2 : a_2 = a_1 * q

theorem geometric_sequence_formula :
  (∃ (a_1 : ℝ) (q : ℝ), q = 2 ∧ S_6 = (1 - q^6) * a_1 / (1 - q) ∧ a_n = a_1 * q^(n - 1)) → 
  (a_n = (2:ℝ)^(n - 1) / 3) :=
by
  intro h
  cases h with a_1 h,
  cases h with q h,
  cases h with hq h,
  cases h with hS ha,
  subst hq,
  sorry

end geometric_sequence_formula_l481_481729


namespace perpendicular_OP_CD_l481_481147

variables {Point : Type}

-- Definitions of all the points involved
variables (A B C D P O : Point)
-- Definitions for distances / lengths
variables (dist : Point → Point → ℝ)
-- Definitions for relationships
variables (circumcenter : Point → Point → Point → Point)
variables (perpendicular : Point → Point → Point → Point → Prop)

-- Segment meet condition
variables (meet_at : Point → Point → Point → Prop)

-- Assuming the given conditions
theorem perpendicular_OP_CD 
  (meet : meet_at A C P)
  (meet' : meet_at B D P)
  (h1 : dist P A = dist P D)
  (h2 : dist P B = dist P C)
  (hO : circumcenter P A B = O) :
  perpendicular O P C D :=
sorry

end perpendicular_OP_CD_l481_481147


namespace probability_of_drawing_ball_2_l481_481949

theorem probability_of_drawing_ball_2 : 
  let total_balls := 3
  let favorable_ball := 1
  probability_of_drawing_ball_2 (total_balls = 3) (ball = 2) : 
    let total_outcomes := total_balls
    let favorable_outcomes := favorable_ball
    let probability := favorable_outcomes / total_outcomes
  probability = 1 / 3 :=
begin
  sorry,
end

end probability_of_drawing_ball_2_l481_481949


namespace choose_6_from_17_l481_481861

theorem choose_6_from_17 : nat.choose 17 6 = 12376 := 
by
  -- will provide a proof here
  sorry

end choose_6_from_17_l481_481861


namespace exponents_of_ten_problem_zeros_10000_pow_50_l481_481586

theorem exponents_of_ten (a : ℤ) (b : ℕ) (h : a = 10^4) : a^b = 10^(4 * b) := by
  rw [h, pow_mul]
  simp
  sorry

theorem problem_zeros_10000_pow_50 : 10000^50 = 10^200 :=
  exponents_of_ten 10000 50 rfl

end exponents_of_ten_problem_zeros_10000_pow_50_l481_481586


namespace incorrectProposition_l481_481370

-- Definitions of the propositions given as conditions.
def PropositionA : Prop := ∀ (p : Quadrilateral), p.isParallelogram ∧ (p.oppositeSidesParallel ∧ p.oppositeSidesEqual)
def PropositionC : Prop := ∀ (p : Quadrilateral), p.isParallelogram ∧ p.hasRightAngle → p.isRectangle
def PropositionD : Prop := ∀ (p : Quadrilateral), p.isRectangle ∧ p.adjacentSidesEqual → p.isSquare

-- Proposition B as a hypothesis.
def PropositionB : Prop := ∀ (p : Quadrilateral), p.diagonalsPerpendicular → p.isRhombus

-- Hypothesis that we need to prove that Proposition B is the false proposition.
theorem incorrectProposition : ¬ PropositionB := by
  sorry

end incorrectProposition_l481_481370


namespace intersection_M_N_l481_481043

def M (x : ℝ) : Prop := -2 < x ∧ x < 2
def N (x : ℝ) : Prop := |x - 1| ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
sorry

end intersection_M_N_l481_481043


namespace liu_dong_correct_answers_l481_481783

theorem liu_dong_correct_answers :
  ∃ (correct_answers : ℕ), correct_answers = 14 ∧ 
  (∃ (incorrect_answers : ℕ), incorrect_answers = 6 ∧ 
  (total_questions = 20 ∧ correct_score = 5 ∧ incorrect_score = 3 ∧ 
  actual_score = 52 ∧ 
  total_score = correct_answers * correct_score - incorrect_answers * incorrect_score ∧ 
  total_score = actual_score)) :=
begin
  let total_questions := 20,
  let correct_score := 5,
  let incorrect_score := -3,
  let actual_score := 52,
  let total_score := 100 - 48,
  use 14,
  split,
  {
    exact rfl,
  },
  {
    use 6,
    split,
    {
      exact rfl,
    },
    split,
    {
      exact and.intro rfl (and.intro rfl (and.intro rfl (and.intro rfl (and.intro rfl (and.intro rfl rfl)))))
    }
  }
end

end liu_dong_correct_answers_l481_481783


namespace age_of_b_l481_481125

/-- Variables representing ages of four people -/
variables (a b c d : ℕ)

/-- Condition: a is two years older than b -/
axiom h1 : a = b + 2

/-- Condition: b is twice as old as c -/
axiom h2 : b = 2 * c

/-- Condition: d is three times as old as a -/
axiom h3 : d = 3 * a

/-- Condition: the total of their ages is 72 -/
axiom h4 : a + b + c + d = 72

/-- From these conditions, we prove that b = 12 -/
theorem age_of_b : b = 12 :=
by
  sorry

end age_of_b_l481_481125


namespace smaller_tetrahedra_share_internal_point_l481_481971

theorem smaller_tetrahedra_share_internal_point (ABC D : Point3D)
    (sphere : Sphere3D)
    (parallel_planes : ∀ face, Plane3D) :
    (∀ P ∈ {A, B, C, D}, Plane3D.parallel (sphere.tangent_plane P) (tetrahedron.face P)) →
    (∀ face1 face2, area(tetrahedron.face face1) ≠ area(tetrahedron.face face2)) →
    ∃ p, ∀ tetrahedron_piece ∈ {tetrahedron_piece₁, tetrahedron_piece₂, tetrahedron_piece₃, tetrahedron_piece₄}, 
        (tetrahedron_piece.interior p) :=
begin
    sorry
end

end smaller_tetrahedra_share_internal_point_l481_481971


namespace packages_needed_l481_481541

-- Define the range of apartment numbers for each floor
def first_floor_apartments : list ℕ := list.range' 105 26
def second_floor_apartments : list ℕ := list.range' 205 26
def third_floor_apartments : list ℕ := list.range' 305 11

-- Combine all apartment numbers into one list
def all_apartments : list ℕ := first_floor_apartments ++ second_floor_apartments ++ third_floor_apartments

-- Calculate digit frequency for each digit from 0 to 9 in the list of all apartment numbers
def digit_frequency (d : ℕ) : ℕ :=
  all_apartments.foldl (λ acc n, acc + (n.digits 10).count d) 0

-- Find the maximum frequency among all digits
def maximum_frequency : ℕ :=
  list.foldl max 0 (list.map digit_frequency (list.range 10))

theorem packages_needed : maximum_frequency = 82 :=
  by
    -- Proof omitted; based on enumerating and counting digits in the list all_apartments
    sorry

end packages_needed_l481_481541


namespace area_of_triangles_in_E_l481_481822

-- Define a triangle in 3-dimensional space
structure Triangle :=
  (A : ℤ × ℤ × ℤ) (B : ℤ × ℤ × ℤ) (C : ℤ × ℤ × ℤ)

-- Define the set E of triangles
def E : Set Triangle := 
  {T | ∀ (x y z : ℤ), (x, y, z) = T.A ∨ (x, y, z) = T.B ∨ (x, y, z) = T.C}

-- Function f to calculate the area of a triangle
noncomputable def f (T : Triangle) : ℝ :=
  let (ax, ay, az) := T.A in
  let (bx, by, bz) := T.B in
  let (cx, cy, cz) := T.C in
  (1 / 2) * Real.sqrt (
    abs (
      ax * (by * 1 - bz * 1) -
      ay * (bx * 1 - bz * 1) +
      az * (bx * 1 - by * 1)
    )
  )

-- Lean statement for the proof
theorem area_of_triangles_in_E :
  ∀ r : ℝ, (0 < r) → 
  ∃ T ∈ E, f T = r :=
by
  sorry

end area_of_triangles_in_E_l481_481822


namespace polynomial_degree_l481_481572

def poly : ℝ[X] := 3 + 7 * X^2 + 200 + 5 * real.pi * X^5 + 3 * real.sqrt 2 * X^6 + 15

theorem polynomial_degree : nat_degree poly = 6 := 
by {
    sorry
}

end polynomial_degree_l481_481572


namespace quadratic_has_two_real_roots_one_non_negative_real_root_l481_481702

theorem quadratic_has_two_real_roots (m : ℝ) :
  let a := (1 : ℝ);
  let b := (4 - m);
  let c := (3 - m);
  let discriminant := b^2 - 4 * a * c
  in discriminant ≥ 0 := by
  -- Proof omitted
  sorry

theorem one_non_negative_real_root (m : ℝ) :
  let root1 := -1;
  let root2 := m - 3
  in (root2 ≥ 0) = (m ≥ 3) := by
  -- Proof omitted
  sorry

end quadratic_has_two_real_roots_one_non_negative_real_root_l481_481702


namespace cistern_length_l481_481616

theorem cistern_length
  (L W D A : ℝ)
  (hW : W = 4)
  (hD : D = 1.25)
  (hA : A = 49)
  (hWetSurface : A = L * W + 2 * L * D) :
  L = 7.54 := by
  sorry

end cistern_length_l481_481616


namespace find_length_of_sheet_l481_481623

noncomputable section

-- Axioms regarding the conditions
def width_of_sheet : ℝ := 36       -- The width of the metallic sheet is 36 meters
def side_of_square : ℝ := 7        -- The side length of the square cut off from each corner is 7 meters
def volume_of_box : ℝ := 5236      -- The volume of the resulting box is 5236 cubic meters

-- Define the length of the metallic sheet as L
def length_of_sheet (L : ℝ) : Prop :=
  let new_length := L - 2 * side_of_square
  let new_width := width_of_sheet - 2 * side_of_square
  let height := side_of_square
  volume_of_box = new_length * new_width * height

-- The condition to prove
theorem find_length_of_sheet : ∃ L : ℝ, length_of_sheet L ∧ L = 48 :=
by
  sorry

end find_length_of_sheet_l481_481623


namespace acute_triangles_at_most_three_quarters_l481_481858

theorem acute_triangles_at_most_three_quarters (n : ℕ) (h_n : 3 < n) 
    (h_collinear : ∀ a b c : point, ¬ collinear a b c) : 
    ∃ T : set (triangle), 
    ((count_acute T).toReal / (count_total T).toReal) ≤ 3 / 4 := 
sorry

end acute_triangles_at_most_three_quarters_l481_481858


namespace general_formula_no_geometric_subseq_t_n_leq_3_4_l481_481845
open Real -- Assuming sequences are real-valued

-- Define the geometric sequence and its sum
def geom_sum (a r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

-- Condition: a_{n+1} = 2 * S_n + 2
def a_recurrence (a S : ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) = 2 * S n + 2

-- General formula for the sequence {a_n}
def a_n (n : ℕ) : ℝ := 2 * 3 ^ (n - 1)

-- Definitions for d_n and T_n
def d_n (n : ℕ) : ℝ := 4 * 3 ^ (n - 1) / (n + 1)

def T_n (n : ℕ) : ℝ := ∑ k in range n, 1 / d_n k

-- Proof statements
theorem general_formula :
  ∀ (S : ℕ → ℝ) (a : ℕ → ℝ),
    (∀ n, S n = geom_sum (a 0) 3 n) →
    (∀ n, a_recurrence a S n) →
    a = a_n :=
sorry

theorem no_geometric_subseq (d : ℕ → ℝ) :
  (∀ n, d n = d_n n) →
  ∀ (m k p : ℕ), (m + p = 2 * k) → ¬(d k) ^ 2 = (d m) * (d p) :=
sorry

theorem t_n_leq_3_4 :
  {n : ℕ | T_n n ≤ 3 / 4} = {1, 2} :=
sorry

end general_formula_no_geometric_subseq_t_n_leq_3_4_l481_481845


namespace captives_strategy_guarantees_release_l481_481820

theorem captives_strategy_guarantees_release (n : ℕ) (hs : n ≥ 3)
  (figures : Fin n → ℕ)
  (h_unique : ∀ i j, figures i ≠ figures j → multiset.card (multiset.filter (= i) figures) ≠ multiset.card (multiset.filter (= j) figures))
  (prisoners_can_see_each_other : ∀ i j, i ≠ j → figures j) :
  ∃ (strategy : (Fin n → ℕ) → (Fin n → ℕ)), 
    (∀ perm : Fin n → ℕ, ∃ i : Fin n, strategy perm i = perm i) :=
sorry

end captives_strategy_guarantees_release_l481_481820


namespace a5_b3_c_divisible_by_6_l481_481459

theorem a5_b3_c_divisible_by_6 (a b c : ℤ) (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) :=
by
  sorry

end a5_b3_c_divisible_by_6_l481_481459


namespace find_other_integer_l481_481498

theorem find_other_integer (x y : ℤ) (h_sum : 3 * x + 2 * y = 115) (h_one_is_25 : x = 25 ∨ y = 25) : (x = 25 → y = 20) ∧ (y = 25 → x = 20) :=
by
  sorry

end find_other_integer_l481_481498


namespace hyperbola_eccentricity_l481_481554

noncomputable def hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a < 2) : Prop :=
  let e := Real.sqrt (1 + (b / a)^2) in
  1 < e ∧ e < Real.sqrt 5

theorem hyperbola_eccentricity {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : b / a < 2) :
  hyperbola_eccentricity_range a b h1 h2 h3 :=
by
  sorry

end hyperbola_eccentricity_l481_481554


namespace find_y_l481_481755

noncomputable def x (y : ℝ) : ℝ := 2 + 1 / y
noncomputable def y (x : ℝ) : ℝ := 1 + 1 / x

theorem find_y (y : ℝ) (h₁ : x y = 1 + sqrt 3) (h₂ : y (x y) = y) (hx : x y ≠ 0) (hy : y (x y) ≠ 0) : y = (1 + sqrt 3) / 2 :=
sorry

end find_y_l481_481755


namespace fourth_derivative_at_zero_l481_481467

variable (f : ℝ → ℝ)
hypothesis h1 : f 0 = 1
hypothesis h2 : deriv f 0 = 2
hypothesis h3 : ∀ t, deriv^[2] f t = 4 * deriv f t - 3 * f t + 1

theorem fourth_derivative_at_zero : deriv^[4] f 0 = 54 := 
by 
  sorry

end fourth_derivative_at_zero_l481_481467


namespace prime_divisors_of_390_l481_481655

theorem prime_divisors_of_390 : 
  (2 * 195 = 390) → 
  (3 * 65 = 195) → 
  (5 * 13 = 65) → 
  ∃ (S : Finset ℕ), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (S.card = 4) ∧ 
    (∀ d ∈ S, d ∣ 390) := 
by
  sorry

end prime_divisors_of_390_l481_481655


namespace count_distinct_convex_numbers_l481_481602

def is_convex_number (n : ℕ) : Prop :=
  let d1 := (n / 100) % 10
  let d2 := (n / 10) % 10
  let d3 := n % 10
  n > 99 ∧ n < 1000 ∧ d2 > d1 ∧ d2 > d3 ∧ d1 < d2 ∧ d3 < d2

theorem count_distinct_convex_numbers : 
  let digits := [1, 2, 3, 4, 5]
  let convex_numbers := (List.range 900).filter is_convex_number
  (convex_numbers.filter (λ n, ∀ d, n ∈ digits)).length = 20 :=
sorry

end count_distinct_convex_numbers_l481_481602


namespace count_triangles_l481_481556

def number_of_points : ℕ := 18

lemma no_collinear_sets_of_three (points : fin number_of_points → ℝ×ℝ) : 
  ∀ (A B C : fin number_of_points), A ≠ B ∧ B ≠ C ∧ A ≠ C → ¬ collinear {points A, points B, points C} :=
sorry

def red_or_blue_edge (points : fin number_of_points → ℝ×ℝ) (color : (ℝ×ℝ) → (ℝ×ℝ) → Prop) : Prop :=
  ∀ (A B : fin number_of_points), A ≠ B → (color (points A) (points B) ∨ color (points B) (points A)) ∧
  ¬ (color (points A) (points B) ∧ color (points B) (points A))

noncomputable def specific_point_A := 0

def odd_number_red_segments (red_edge : (ℝ×ℝ) → (ℝ×ℝ) → Prop ) (points : fin number_of_points → ℝ×ℝ) : Prop :=
  ∃ (point_idx : fin number_of_points), point_idx = specific_point_A ∧
  card {B | red_edge (points point_idx) (points B)} % 2 = 1

def distinct_red_segments (red_edge : (ℝ×ℝ) → (ℝ×ℝ) → Prop) (points : fin number_of_points → ℝ×ℝ) : Prop :=
  ∀ (i j : fin number_of_points), i ≠ j → 
  card {B | red_edge (points i) (points B)} ≠ card {B | red_edge (points j) (points B)}

theorem count_triangles (points : fin number_of_points → ℝ×ℝ)
    (red_edge blue_edge : (ℝ×ℝ) → (ℝ×ℝ) → Prop)
    (h1 : no_collinear_sets_of_three points)
    (h2 : red_or_blue_edge points red_edge ∧ red_or_blue_edge points blue_edge)
    (h3 : odd_number_red_segments red_edge points)
    (h4 : distinct_red_segments red_edge points):
    ∃ (all_red : ℕ) (two_red_one_blue : ℕ), all_red = 204 ∧ two_red_one_blue = 240 :=
sorry

end count_triangles_l481_481556


namespace find_a10_l481_481419

-- Define the arithmetic sequence using the general term formula
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Conditions given in the problem
variables (a3 a7 : ℤ) (d : ℤ)
hypothesis h1 : a3 = 5
hypothesis h2 : a7 = -7
hypothesis h3 : a7 = arithmetic_sequence a3 d 7

-- Prove a10 = -16
theorem find_a10 : ∃ (a10 : ℤ), a10 = -16 :=
  begin
    use a7 + 3*d,
    have h4 : a7 = 5 + 6 * d, -- deriving from h1, h2, and the formula for arithmetic_sequence
    from sorry, -- placeholder for derived steps
    linarith, -- this step automatically uses the hypotheses to get a10 = -16
  end

end find_a10_l481_481419


namespace probability_of_less_than_5_is_one_half_l481_481225

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l481_481225


namespace general_term_a_sum_T_formula_l481_481703

-- Define the sequence a_n and the sum S_n
def seq_a (n : ℕ) := 2 * n
def sum_S (n : ℕ) := (n + 1) * seq_a n - n * (n + 1)

-- Define the sequence b_n based on a_n
def seq_b (n : ℕ) := 2^(n - 1) * seq_a n

-- Define the sum T_n of the first n terms of b_n
def sum_T : ℕ → ℕ
| 0     := 0
| (n+1) := sum_T n + seq_b (n+1)

-- Proof statement for the general term of a_n
theorem general_term_a (n : ℕ) (h_pos : 0 < n) :
  seq_a n = 2 * n := by
  sorry

-- Proof statement for the sum of the first n terms of b_n
theorem sum_T_formula (n : ℕ) :
  sum_T n = (n - 1) * 2^(n + 1) + 2 := by
  sorry

end general_term_a_sum_T_formula_l481_481703


namespace same_terminal_side_l481_481158

theorem same_terminal_side (k : ℤ): ∃ k : ℤ, 1303 = k * 360 - 137 := by
  -- Proof left as an exercise.
  sorry

end same_terminal_side_l481_481158


namespace greatest_number_in_consecutive_multiples_l481_481595

theorem greatest_number_in_consecutive_multiples (s : Set ℕ)
  (h1 : ∃ n, s = {x | ∃ k, x = 305 + (k * 5) ∧ 0 ≤ k ∧ k < 150})
  (h2 : ∃ x ∈ s, ∀ y ∈ s, x ≥ y) :
  (∀ x ∈ s, x ≤ 1050) :=
by
  sorry

end greatest_number_in_consecutive_multiples_l481_481595


namespace train_passes_platform_in_39_2_seconds_l481_481977

def length_of_train : ℝ := 360
def speed_in_kmh : ℝ := 45
def length_of_platform : ℝ := 130

noncomputable def speed_in_mps : ℝ := speed_in_kmh * 1000 / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_in_mps

theorem train_passes_platform_in_39_2_seconds :
  time_to_pass_platform = 39.2 := by
  sorry

end train_passes_platform_in_39_2_seconds_l481_481977


namespace sum_of_coordinates_l481_481362

theorem sum_of_coordinates (g : ℝ → ℝ) (h₁ : g 3 = 10) : 
  let (x, y) := (1, (4 * g (3 * 1) - 2) / 5) in
  x + y = 8.6 := 
by
  have h₂ : x = 1 := rfl
  have h₃ : y = (4 * g (3 * 1) - 2) / 5 := rfl
  rw [h₃, h₁]
  have y_val : (4 * 10 - 2) / 5 = 7.6
  { calc
    (4 * 10 - 2) / 5 = (40 - 2) / 5 : by ring
                   ... = 38 / 5      : by ring
                   ... = 7.6         : by norm_num }
  rw [y_val]
  exact add_comm 1 7.6 ▸ rfl

end sum_of_coordinates_l481_481362


namespace total_candy_pieces_l481_481922

theorem total_candy_pieces : 
  (brother_candy = 6) → 
  (wendy_boxes = 2) → 
  (pieces_per_box = 3) → 
  (brother_candy + (wendy_boxes * pieces_per_box) = 12) 
  := 
  by 
    intros brother_candy wendy_boxes pieces_per_box 
    sorry

end total_candy_pieces_l481_481922


namespace apartment_building_floors_l481_481940

theorem apartment_building_floors (K E P : ℕ) (h1 : 1 < K) (h2 : K < E) (h3 : E < P) (h4 : K * E * P = 715) : 
  E = 11 :=
sorry

end apartment_building_floors_l481_481940


namespace not_third_l481_481152

section race

variables (A B C D E F : ℕ) -- Using natural numbers to represent the position of each runner

-- Conditions as inequalities
def conditions :=
  A < B ∧ A < D ∧ B < C ∧ B < F ∧ C < D ∧ E < F ∧ A < E

-- The main theorem to prove that A and F cannot be in the third position
theorem not_third (h : conditions A B C D E F) :
  (A ≠ 3 ∧ F ≠ 3) :=
begin
  sorry
end

end race

end not_third_l481_481152


namespace b_plus_d_over_a_l481_481187

theorem b_plus_d_over_a (a b c d e : ℝ) (h : a ≠ 0) 
  (root1 : a * (5:ℝ)^4 + b * (5:ℝ)^3 + c * (5:ℝ)^2 + d * (5:ℝ) + e = 0)
  (root2 : a * (-3:ℝ)^4 + b * (-3:ℝ)^3 + c * (-3:ℝ)^2 + d * (-3:ℝ) + e = 0)
  (root3 : a * (2:ℝ)^4 + b * (2:ℝ)^3 + c * (2:ℝ)^2 + d * (2:ℝ) + e = 0) :
  (b + d) / a = - (12496 / 3173) :=
sorry

end b_plus_d_over_a_l481_481187


namespace probability_of_rolling_number_less_than_5_is_correct_l481_481214

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l481_481214


namespace find_radius_of_third_circle_l481_481667

noncomputable def radius_of_third_circle : ℝ :=
  (7 / (2 * (Real.sqrt 2 + Real.sqrt 10))) ^ 2

theorem find_radius_of_third_circle
  (A B : Point)
  (radius_A radius_B : ℝ)
  (hA : radius_A = 2)
  (hB : radius_B = 5)
  (tangent : Circle A radius_A = Circle B radius_B) :
  radius_of_third_circle = (7 / (2 * (Real.sqrt 2 + Real.sqrt 10))) ^ 2 := by
  sorry

end find_radius_of_third_circle_l481_481667


namespace negative_coefficient_exists_l481_481302

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 + x^3 - 3 * x^2 + x + 2

-- Define the polynomial P(x)^k
def P_pow (x : ℝ) (k : ℕ) : ℝ := (P x) ^ k

-- Statement of the problem
theorem negative_coefficient_exists (k : ℕ) (hk : k > 0) :
  ∃ (n : ℤ), (coeff (P_pow x k) n) < 0 :=
sorry

end negative_coefficient_exists_l481_481302


namespace find_b_l481_481041

-- Definitions
def quadratic (x b c : ℝ) : ℝ := x^2 + b * x + c

theorem find_b (b c : ℝ) 
  (h_diff : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → (∀ y : ℝ, 1 ≤ y ∧ y ≤ 7 → quadratic x b c - quadratic y b c = 25)) :
  b = -4 ∨ b = -12 :=
by sorry

end find_b_l481_481041


namespace pie_eating_contest_l481_481569

theorem pie_eating_contest :
  (8 / 9 : ℚ) - (5 / 6 : ℚ) = 1 / 18 := 
by {
  sorry
}

end pie_eating_contest_l481_481569


namespace percentage_of_water_in_first_liquid_l481_481955

theorem percentage_of_water_in_first_liquid (x : ℝ) 
  (h1 : 0 < x ∧ x ≤ 1)
  (h2 : 0.35 = 0.35)
  (h3 : 10 = 10)
  (h4 : 4 = 4)
  (h5 : 0.24285714285714285 = 0.24285714285714285) :
  ((10 * x + 4 * 0.35) / (10 + 4) = 0.24285714285714285) → (x = 0.2) :=
sorry

end percentage_of_water_in_first_liquid_l481_481955


namespace fractional_shaded_area_infinite_series_l481_481637

theorem fractional_shaded_area_infinite_series :
  let shaded_fraction := (1/4 : ℝ)
  infinite_series_sum : ℝ := shaded_fraction / (1 - shaded_fraction)
  infinite_series_sum = (1 / 3 : ℝ) given
    -- Condition: a square is initially divided into sixteen equal smaller squares.
    -- Condition: the center four squares are then each divided into sixteen smaller squares of equal area,
    -- and this pattern continues indefinitely.
    -- Condition: in each set of sixteen smaller squares, the four corner squares are shaded.
  infinite_series_sum = (1 / 3 : ℝ) := 
sorry

end fractional_shaded_area_infinite_series_l481_481637


namespace hyperbola_eccentricity_l481_481357

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_angle : b / a = Real.sqrt 3 / 3) :
    let e := Real.sqrt (1 + (b / a)^2)
    e = 2 * Real.sqrt 3 / 3 := 
sorry

end hyperbola_eccentricity_l481_481357


namespace max_sum_of_squares_70_l481_481885

theorem max_sum_of_squares_70 :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a^2 + b^2 + c^2 + d^2 = 70 ∧ a + b + c + d = 16 :=
by
  sorry

end max_sum_of_squares_70_l481_481885


namespace decagon_intersection_points_l481_481981

theorem decagon_intersection_points : 
  let n := 10 in 
  ∑ i in (Fintype.elems (Finset.range (n+1)).image (λ k, (Finset.card (Finset.choose 4 (Finset.range n)) i)), 
    i = 210 := 
begin
  sorry
end

end decagon_intersection_points_l481_481981


namespace maximize_revenue_at_p_l481_481948

theorem maximize_revenue_at_p (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 37.5) :
  let R := p * (150 - 4 * p) in
  ∃ p_max : ℝ, p_max = 18.75 ∧ ∀ q : ℝ, 0 ≤ q → q ≤ 37.5 → R ≤ q * (150 - 4 * q) :=
by
  sorry

end maximize_revenue_at_p_l481_481948


namespace difference_in_speed_l481_481607

-- Conditions
def bike_distance : ℝ := 136
def bike_time : ℝ := 8
def truck_distance : ℝ := 112
def truck_time : ℝ := 8

-- Definitions based on conditions
def bike_speed : ℝ := bike_distance / bike_time
def truck_speed : ℝ := truck_distance / truck_time

-- Problem statement: Prove the difference in speeds is 3 mph
theorem difference_in_speed : (bike_speed - truck_speed) = 3 :=
by
  sorry

end difference_in_speed_l481_481607


namespace trajectory_equation_minimum_S_l481_481701

-- Problem Definitions
def point := ℝ × ℝ -- Definition for points in ℝ²

def E : point := (2, 0) -- Point E on x-axis
def F : point := (1, 0) -- Point F on x-axis

def is_on_trajectory (M : point) : Prop := M.2 ^ 2 = 4 * M.1 -- y^2 = 4x

def dot_product (A B : point) : ℝ := A.1 * B.1 + A.2 * B.2

def triangle_area (O A B : point) : ℝ := (1 / 2) * (O.1 * (A.2 - B.2) + A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2))

theorem trajectory_equation
  (M : point) (H_ME : (M.1 - 2)^2 + M.2^2 = M.1^2 + 4) :
  is_on_trajectory M :=
sorry

theorem minimum_S
  (A B : point) (H_A : is_on_trajectory A) (H_B : is_on_trajectory B)
  (H_dot : dot_product A B = -4) :
  ∃ y1, A.2 = y1 ∧ ∃ y2, B.2 = y2 ∧ 
  y1 > 0 ∧ y1 * y2 = -8 ∧ 
  let S := (1/2) * |F.1 * A.2| + |(1/2) * F.1 * (A.2 - B.2)| in
  S ≥ 4 * √3 ∧ S = 4 * √3 :=
sorry

end trajectory_equation_minimum_S_l481_481701


namespace find_max_lambda_l481_481712

theorem find_max_lambda (n : ℕ) (h1 : 2 ≤ n) (θ : Fin n → ℝ) 
  (hθ1 : ∀ i, 0 < θ i ∧ θ i < π / 2) :
  ∃ λ, λ = -n ∧ 
  ((∑ i, Real.tan (θ i)) * (∑ i, Real.cot (θ i)) ≥ 
  (∑ i, Real.sin (θ i))^2 + 
  (∑ i, Real.cos (θ i))^2 + 
  λ * (θ 0 - θ (n - 1))^2) :=
begin
  use -n,
  split,
  { refl, },  -- Prove that λ = -n.
  { sorry }   -- This part would be the actual proof of the inequality.
end

end find_max_lambda_l481_481712


namespace average_minutes_run_per_day_l481_481286

theorem average_minutes_run_per_day (total_students : ℕ) (sixth_avg seventh_avg eighth_avg : ℕ) 
  (ratio_sixth_to_seventh : ℕ) (ratio_seventh_to_eighth : ℕ) 
  (h_total_students : total_students = 210)
  (h_sixth_avg : sixth_avg = 10)
  (h_seventh_avg : seventh_avg = 12)
  (h_eighth_avg : eighth_avg = 14)
  (h_ratio_sixth : ratio_sixth_to_seventh = 3)
  (h_ratio_seventh : ratio_seventh_to_eighth = 4) :
  (let sixth := 3 * (4 * (210 / 17)), 
       seventh := 4 * (210 / 17), 
       eighth := (210 / 17),
       total_minutes := 10 * sixth + 12 * seventh + 14 * eighth
   in (total_minutes / total_students) = 420 / 39) := by
  sorry

end average_minutes_run_per_day_l481_481286


namespace sin_product_identity_l481_481000

-- Definitions
def c_n (n : ℕ) : ℝ := 2^(1-n)

-- Conjecture to prove
theorem sin_product_identity (n : ℕ) (x : ℝ) (hn : n > 1) : 
  (List.prod (List.map (λ k : ℕ, Real.sin (x + k * Real.pi / n)) (List.range n))) = c_n n * Real.sin (n * x) := 
by
  sorry

end sin_product_identity_l481_481000


namespace monotonic_decreasing_interval_l481_481013

-- Condition: The slope of the tangent line at any point (x0, f(x0)) 
-- is given by k = (x0 - 3) * (x0 + 1)^2
def slope (x0 : ℝ) : ℝ := (x0 - 3) * (x0 + 1)^2

-- The function f(x) is monotonically decreasing in the interval (-∞, 3]
theorem monotonic_decreasing_interval :
  ∀ x0 : ℝ, slope x0 < 0 → x0 < 3 :=
sorry

end monotonic_decreasing_interval_l481_481013


namespace expression_value_l481_481583

theorem expression_value (a b : ℤ) (ha : a = -4) (hb : b = 3) : 
  -2 * a - b ^ 3 + 2 * a * b = -43 := by
  rw [ha, hb]
  sorry

end expression_value_l481_481583


namespace third_cyclist_speed_l481_481917

theorem third_cyclist_speed (a b : ℝ) : 
    ∃ v : ℝ, 
        (∀ t1 t2 t3 x, 
            t1 = 1/6 + x ∧ 
            t2 = t1 + 1/3 ∧ 
            t3 = x ∧ 
            (t1*a = x*v) ∧ 
            (t2*b = (x + 1/3)*v)
        ) → 
        v = (1/4) * (a + 3 * b + real.sqrt (a^2 - 10*a*b + b^2)) := 
begin
    sorry
end

end third_cyclist_speed_l481_481917


namespace rachel_remaining_money_l481_481873

def initial_earnings : ℝ := 200
def lunch_expense : ℝ := (1/4) * initial_earnings
def dvd_expense : ℝ := (1/2) * initial_earnings
def total_expenses : ℝ := lunch_expense + dvd_expense
def remaining_amount : ℝ := initial_earnings - total_expenses

theorem rachel_remaining_money :
  remaining_amount = 50 := 
by
  sorry

end rachel_remaining_money_l481_481873


namespace hedgehogs_in_garden_l481_481489

theorem hedgehogs_in_garden (h : ∀ (s : ℝ), s = 1) : 
  ∑ i in (finset.range 3), (median_length i 1) = real.sqrt 3 := sorry

-- Definitions for median_length, triangle, etc., should be given but are omitted for the lean code skeleton.

end hedgehogs_in_garden_l481_481489


namespace find_a_of_my_function_odd_l481_481025

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def my_function (a : ℝ) (x : ℝ) : ℝ :=
  real.exp x + a * real.exp (-x)

theorem find_a_of_my_function_odd (a : ℝ) :
  is_odd_function (my_function a) → a = -1 :=
by
  intro h
  sorry

end find_a_of_my_function_odd_l481_481025


namespace solve_for_x_l481_481881

theorem solve_for_x (x : ℝ) (h : sqrt (2 / x + 3) = 4 / 3) : x = -18 / 11 :=
by
  sorry

end solve_for_x_l481_481881


namespace cube_surface_area_l481_481596

theorem cube_surface_area (v : ℝ) (h : v = 343) : ∃ (a : ℝ), a = 294 :=
by
  let side := Real.cbrt v
  let surface_area := 6 * side^2
  have hs : surface_area = 6 * (Real.cbrt 343)^2 := by sorry
  show ∃ (a : ℝ), a = 294, from ⟨surface_area, hs⟩

end cube_surface_area_l481_481596


namespace wine_price_increase_l481_481251

-- Definitions translating the conditions
def wine_cost_today : ℝ := 20.0
def bottles_count : ℕ := 5
def tariff_rate : ℝ := 0.25

-- Statement to prove
theorem wine_price_increase (wine_cost_today : ℝ) (bottles_count : ℕ) (tariff_rate : ℝ) : 
  bottles_count * wine_cost_today * tariff_rate = 25.0 := 
by
  -- Proof is omitted
  sorry

end wine_price_increase_l481_481251


namespace two_digit_number_representation_l481_481639

theorem two_digit_number_representation (x : ℕ) (h : x < 10) : 10 * x + 5 < 100 :=
by sorry

end two_digit_number_representation_l481_481639


namespace exists_inscribed_quadrilateral_example_l481_481141

noncomputable def inscribed_quadrilateral_example : Prop :=
  ∃ (a1 b1 c1 a2 b2 c2 : ℕ),
    a1 ≠ b1 ∧ a1 ≠ c1 ∧ a1 ≠ a2 ∧ a1 ≠ b2 ∧ a1 ≠ c2 ∧
    b1 ≠ c1 ∧ b1 ≠ a2 ∧ b1 ≠ b2 ∧ b1 ≠ c2 ∧
    c1 ≠ a2 ∧ c1 ≠ b2 ∧ c1 ≠ c2 ∧
    a2 ≠ b2 ∧ a2 ≠ c2 ∧
    b2 ≠ c2 ∧
    (a1^2 + b1^2 = c1^2) ∧ (a2^2 + b2^2 = c2^2) ∧
    ((c1^2 + c2^2) / 2).is_integer
  
theorem exists_inscribed_quadrilateral_example :
  inscribed_quadrilateral_example :=
sorry

end exists_inscribed_quadrilateral_example_l481_481141


namespace find_face_value_l481_481200

noncomputable def stock_face_value (CP : ℝ) (discount_rate : ℝ) (brokerage_rate : ℝ) : ℝ :=
  CP / ( (1 - discount_rate) * ( 1 + brokerage_rate * (1 - discount_rate) ) )

theorem find_face_value :
  let CP := 91.2
  let discount_rate := 0.09
  let brokerage_rate := 0.002
  stock_face_value CP discount_rate brokerage_rate ≈ 100 :=
by
  sorry

end find_face_value_l481_481200


namespace john_daily_visits_l481_481809

theorem john_daily_visits 
  (weekly_sales : ℕ := 5000) 
  (work_days : ℕ := 5) 
  (sales_percentage : ℚ := 0.20) 
  (average_sale : ℚ := 100) 
  : 
  (total_houses : ℕ) :=
  total_houses = weekly_sales / (work_days * sales_percentage * average_sale) := by
  sorry

end john_daily_visits_l481_481809


namespace find_expression_value_find_m_value_find_roots_and_theta_l481_481352

-- Define the conditions
variable (θ : ℝ) (m : ℝ)
variable (h1 : θ > 0) (h2 : θ < 2 * Real.pi)
variable (h3 : ∀ x, (2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0) → (x = Real.sin θ ∨ x = Real.cos θ))

-- Theorem 1: Find the value of a given expression
theorem find_expression_value :
  (Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + Real.cos θ / (1 - Real.tan θ) = (Real.sqrt 3 + 1) / 2 :=
  sorry

-- Theorem 2: Find the value of m
theorem find_m_value :
  m = Real.sqrt 3 / 2 :=
  sorry

-- Theorem 3: Find the roots of the equation and the value of θ
theorem find_roots_and_theta :
  (∀ x, (2 * x^2 - (Real.sqrt 3 + 1) * x + Real.sqrt 3 / 2 = 0) → (x = Real.sqrt 3 / 2 ∨ x = 1 / 2)) ∧
  (θ = Real.pi / 6 ∨ θ = Real.pi / 3) :=
  sorry

end find_expression_value_find_m_value_find_roots_and_theta_l481_481352


namespace initial_average_marks_is_90_l481_481159

def incorrect_average_marks (A : ℝ) : Prop :=
  let wrong_sum := 10 * A
  let correct_sum := 10 * 95
  wrong_sum + 50 = correct_sum

theorem initial_average_marks_is_90 : ∃ A : ℝ, incorrect_average_marks A ∧ A = 90 :=
by
  use 90
  unfold incorrect_average_marks
  simp
  sorry

end initial_average_marks_is_90_l481_481159


namespace towel_bleaching_decrease_area_l481_481235

theorem towel_bleaching_decrease_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    let A := L * B
    let A_new := 0.64 * L * B
    (A - A_new) / A * 100 = 36 := by
  let A := L * B
  let A_new := 0.64 * L * B
  have h1 : A - A_new = 0.36 * A := by sorry
  have h2 : (A - A_new) / A = 0.36 := by sorry
  have h3 : 0.36 * 100 = 36 := by sorry
  exact h3

end towel_bleaching_decrease_area_l481_481235


namespace number_of_graphic_novels_l481_481991

theorem number_of_graphic_novels (total_books novels_percent comics_percent : ℝ) 
  (h_total : total_books = 120) 
  (h_novels_percent : novels_percent = 0.65) 
  (h_comics_percent : comics_percent = 0.20) :
  total_books - (novels_percent * total_books + comics_percent * total_books) = 18 :=
by
  sorry

end number_of_graphic_novels_l481_481991


namespace probability_of_spade_or_king_in_two_draws_l481_481254

def total_cards : ℕ := 52
def spades_count : ℕ := 13
def kings_count : ℕ := 4
def king_of_spades_count : ℕ := 1
def spades_or_kings_count : ℕ := spades_count + kings_count - king_of_spades_count
def probability_not_spade_or_king : ℚ := (total_cards - spades_or_kings_count) / total_cards
def probability_both_not_spade_or_king : ℚ := probability_not_spade_or_king^2
def probability_at_least_one_spade_or_king : ℚ := 1 - probability_both_not_spade_or_king

theorem probability_of_spade_or_king_in_two_draws :
  probability_at_least_one_spade_or_king = 88 / 169 :=
sorry

end probability_of_spade_or_king_in_two_draws_l481_481254


namespace share_of_b_l481_481933

theorem share_of_b (x : ℝ) (h : 3300 / ((7/2) * x) = 2 / 7) :  
   let total_profit := 3300
   let B_share := (x / ((7/2) * x)) * total_profit
   B_share = 942.86 :=
by sorry

end share_of_b_l481_481933


namespace complex_quadrant_example_correct_l481_481366

noncomputable def complex_quadrant_example : Prop :=
  ∃ z : ℂ, z = (3 + complex.i) / (1 + complex.i) ∧ complex.re z > 0 ∧ complex.im z < 0

theorem complex_quadrant_example_correct : complex_quadrant_example :=
  sorry

end complex_quadrant_example_correct_l481_481366


namespace vector_combination_l481_481662

open Matrix

def vec1 : Fin 2 → ℤ := ![3, -9]
def vec2 : Fin 2 → ℤ := ![-1, 6]
def vec3 : Fin 2 → ℤ := ![0, -2]
def scal_mult (k : ℤ) (v : Fin 2 → ℤ) := λ i, k * v i

theorem vector_combination :
  scal_mult 4 vec1 + scal_mult 3 vec2 - scal_mult 5 vec3 = ![9, -8] :=
by sorry

end vector_combination_l481_481662


namespace probability_of_rolling_less_than_5_l481_481223

theorem probability_of_rolling_less_than_5 :
  (let outcomes := 8 in
   let successes := 4 in
   let probability := (successes: ℚ) / outcomes in
   probability = 1 / 2) :=
by
  sorry

end probability_of_rolling_less_than_5_l481_481223


namespace build_wall_time_l481_481804

theorem build_wall_time (r : ℝ): 
    (60 * r * 3 = 1) →
    (40 * r * t = 1) →
    t = 4.5 :=
by
  intros,
  sorry

end build_wall_time_l481_481804


namespace neznaika_correct_l481_481504

theorem neznaika_correct :
  ∀ (f : ℕ → ℕ → ℕ), 
  (∀ a b, f a b = a + b ∨ f a b = a * b) →
  (∃ (l : list ℕ) (g : list ℕ → ℕ),
    (∀ (x : ℕ) (xs : list ℕ), g (x :: xs) = f x (g xs)) ∧
    g l = 2014 ∧
    (∀ (f' : ℕ → ℕ → ℕ),
      (∀ a b, f' a b = a * b ∨ f' a b = a + b) →
      g' l = 2014)) :=
by sorry

end neznaika_correct_l481_481504


namespace area_of_triangle_ABC_l481_481842

noncomputable def point (α : Type*) :=  α × α × α

variables 
  (O A B C : point ℝ)
  (OA : ℝ)
  (BAC : ℝ)

open real

def origin : point ℝ := (0, 0, 0)
def A_pos_axis : point ℝ := (real.sqrt 50, 0, 0)
def B_pos_axis : point ℝ := (0, OA, 0)
def C_pos_axis : point ℝ := (0, 0, OA)

theorem area_of_triangle_ABC
  (hO : O = origin)
  (hA : A = A_pos_axis)
  (hB : B = B_pos_axis)
  (hC : C = C_pos_axis)
  (hOA : OA = real.sqrt 50)
  (hBAC : BAC = π / 4): 
  (1 / 2 * OA * OA * sin BAC) = 12.5 :=
sorry

end area_of_triangle_ABC_l481_481842


namespace cube_not_formed_by_tetrahedrons_l481_481669

theorem cube_not_formed_by_tetrahedrons (c : ℝ) (N k : ℕ) :
  let volume_tetrahedron := (√2 / 12) * c^3,
      area_face_tetrahedron := (√3 / 4) * c^2,
      volume_cube := 1,
      area_face_cube := 1
  in
  (N * volume_tetrahedron = volume_cube) → 
  (k * area_face_tetrahedron = area_face_cube) → 
  false :=
by sorry

end cube_not_formed_by_tetrahedrons_l481_481669


namespace red_marked_area_on_larger_sphere_l481_481967

-- Define the conditions
def r1 : ℝ := 4 -- radius of the smaller sphere
def r2 : ℝ := 6 -- radius of the larger sphere
def A1 : ℝ := 37 -- area marked on the smaller sphere

-- State the proportional relationship as a Lean theorem
theorem red_marked_area_on_larger_sphere : 
  let A2 := A1 * (r2^2 / r1^2)
  A2 = 83.25 :=
by
  sorry

end red_marked_area_on_larger_sphere_l481_481967


namespace correct_shelf_probability_l481_481144

theorem correct_shelf_probability:
  ∃ (arrangements : Finset (List ℕ)) (valid : Finset (List ℕ)), 
     arrangements.card = 420 ∧ 
     valid.card = 24 ∧ 
     ((valid.card : ℝ) / (arrangements.card : ℝ) = 2 / 35) := 
by 
  -- Let's define arrangement and valid conditions here 
  sorry

end correct_shelf_probability_l481_481144


namespace number_of_valid_subsets_l481_481481

open Finset

def is_valid_subset (S T : Finset ℕ) : Prop :=
  ∀ x, x ∈ T → 2 * x ∈ S → 2 * x ∈ T

theorem number_of_valid_subsets :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.to_finset in
  ( {T : Finset ℕ | T ⊆ S ∧ is_valid_subset S T}.to_finset.card = 180 ) :=
sorry

end number_of_valid_subsets_l481_481481


namespace area_larger_sphere_red_is_83_point_25_l481_481968

-- Define the radii and known areas

def radius_smaller_sphere := 4 -- cm
def radius_larger_sphere := 6 -- cm
def area_smaller_sphere_red := 37 -- square cm

-- Prove the area of the region outlined in red on the larger sphere
theorem area_larger_sphere_red_is_83_point_25 :
  ∃ (area_larger_sphere_red : ℝ),
    area_larger_sphere_red = 83.25 ∧
    area_larger_sphere_red = area_smaller_sphere_red * (radius_larger_sphere ^ 2 / radius_smaller_sphere ^ 2) :=
by {
  sorry
}

end area_larger_sphere_red_is_83_point_25_l481_481968


namespace production_increase_percentage_l481_481953

variable (T : ℝ) -- Initial production
variable (T1 T2 T5 : ℝ) -- Productions at different years
variable (x : ℝ) -- Unknown percentage increase for last three years

-- Conditions
def condition1 : Prop := T1 = T * 1.06
def condition2 : Prop := T2 = T1 * 1.08
def condition3 : Prop := T5 = T * (1.1 ^ 5)

-- Statement to prove
theorem production_increase_percentage :
  condition1 T T1 →
  condition2 T1 T2 →
  (T5 = T2 * (1 + x / 100) ^ 3) →
  x = 12.1 :=
by
  sorry

end production_increase_percentage_l481_481953


namespace angle_ACB_is_120_l481_481980

-- Define the points A and B with their respective geographical coordinates
def Point (latitude longitude : ℝ) : Type :=
  { lat : ℝ // lat = latitude } × { lon : ℝ // lon = longitude }

-- Ajay's position
def A : Point := (⟨0, rfl⟩, ⟨110, rfl⟩)

-- Billy's position
def B : Point := (⟨45, rfl⟩, ⟨-115, rfl⟩)

-- Earth as a perfect sphere and center at C, we need to prove the angle ACB is 120 degrees
theorem angle_ACB_is_120 (A B : Point) (C : Type) (earth_is_spherical: spherical C) : 
  angle A C B = 120 :=
sorry

end angle_ACB_is_120_l481_481980


namespace leftmost_digit_base9_l481_481788

theorem leftmost_digit_base9 (x : ℕ) (h : x = 3^19 + 2*3^18 + 1*3^17 + 1*3^16 + 2*3^15 + 2*3^14 + 1*3^13 + 1*3^12 + 1*3^11 + 2*3^10 + 2*3^9 + 2*3^8 + 1*3^7 + 1*3^6 + 1*3^5 + 1*3^4 + 2*3^3 + 2*3^2 + 2*3^1 + 2) : ℕ :=
by
  sorry

end leftmost_digit_base9_l481_481788


namespace sufficient_but_not_necessary_condition_l481_481974

theorem sufficient_but_not_necessary_condition (x : ℝ) : (0 < x ∧ x < 2) → (x^2 - 4*x < 0)  :=
begin
  intro h,
  cases h with h1 h2,
  apply lt_trans,
  { linarith },
  sorry
end

end sufficient_but_not_necessary_condition_l481_481974


namespace rival_awards_l481_481097

theorem rival_awards (scott_awards jessie_awards rival_awards : ℕ)
  (h1 : scott_awards = 4)
  (h2 : jessie_awards = 3 * scott_awards)
  (h3 : rival_awards = 2 * jessie_awards) :
  rival_awards = 24 :=
by sorry

end rival_awards_l481_481097


namespace probability_of_rolling_number_less_than_5_is_correct_l481_481216

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l481_481216


namespace chord_length_l481_481543

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 4 * y + 6 = 0

-- Define the line l that is the symmetry axis
def line_l (k x y : ℝ) : Prop :=
  k * x + y + 4 = 0

-- Define the line m passing through point A(0, k) with slope 1
def line_m (k x y : ℝ) : Prop :=
  y = x + k

theorem chord_length (k : ℝ) (h_symmetry : ∃ x y, circle_C x y ∧ line_l k x y) :
  ∃ d : ℝ, d = sqrt 6 ∧
  ∃ x1 y1 x2 y2, circle_C x1 y1 ∧ circle_C x2 y2 ∧ line_m k x1 y1 ∧ line_m k x2 y2 ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = d ^ 2 :=
begin
  -- The proof is not required, hence we add sorry to skip the proof.
  sorry
end

end chord_length_l481_481543


namespace q_minus_r_max_value_l481_481545

theorem q_minus_r_max_value :
  ∃ (q r : ℕ), 1073 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q - r = 31 :=
sorry

end q_minus_r_max_value_l481_481545


namespace room_width_l481_481165

theorem room_width (W : ℝ) (L : ℝ := 17) (veranda_width : ℝ := 2) (veranda_area : ℝ := 132) :
  (21 * (W + veranda_width) - L * W = veranda_area) → W = 12 :=
by
  -- setup of the problem
  have total_length := L + 2 * veranda_width
  have total_width := W + 2 * veranda_width
  have area_room_incl_veranda := total_length * total_width - (L * W)
  -- the statement is already provided in the form of the theorem to be proven
  sorry

end room_width_l481_481165


namespace angle_ABD_in_parallelogram_l481_481789

theorem angle_ABD_in_parallelogram (A B C D O₁ O₂ : Point) (hParallelogram : Parallelogram A B C D)
  (hAngleACD : ∠ C A D = 30) (hCircumCenters : OnLine O₁ O₂ (Line A C))
  (hCircumcircleABD : CircumcircleCenter O₁ A B D)
  (hCircumcircleBCD : CircumcircleCenter O₂ B C D) :
  ∠ A B D = 30 ∨ ∠ A B D = 60 := 
sorry

end angle_ABD_in_parallelogram_l481_481789


namespace imaginary_number_m_l481_481763

theorem imaginary_number_m (m : ℝ) : 
  (∀ Z, Z = (m + 2 * Complex.I) / (1 + Complex.I) → Z.im = 0 → Z.re = 0) → m = -2 :=
by
  sorry

end imaginary_number_m_l481_481763


namespace part_one_solution_set_part_two_range_of_m_l481_481039

-- Part I
theorem part_one_solution_set (x : ℝ) : (|x + 1| + |x - 2| - 5 > 0) ↔ (x > 3 ∨ x < -2) :=
sorry

-- Part II
theorem part_two_range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) ↔ (m ≤ 1) :=
sorry

end part_one_solution_set_part_two_range_of_m_l481_481039


namespace math_problem_proof_l481_481658

-- Define the problem statement
def problem_expr : ℕ :=
  28 * 7 * 25 + 12 * 7 * 25 + 7 * 11 * 3 + 44

-- Prove the problem statement equals to the correct answer
theorem math_problem_proof : problem_expr = 7275 := by
  sorry

end math_problem_proof_l481_481658


namespace false_statement_l481_481229

theorem false_statement :
  (∀ (rectangle : Type) (a b : rectangle)
    (ha : is_rectangle rectangle a b), 
    (equal_diagonals rectangle a b) ∧ (bisect_each_other rectangle a b)) ∧
  (∀ (rhombus : Type) (r : rhombus)
    (hr : is_rhombus rhombus r)
    (eq_diagonals : equal_diagonals rhombus r), 
    is_square rhombus r) ∧
  (∀ (line_segment : Type) (A B P : line_segment)
    (hg : golden_section_point A B P) 
    (hAP : AP > BP)
    (hBP : BP = 6), 
    AP = 3 * sqrt 5 + 3) ∧
  (∀ (triangle1 triangle2 : Type) 
    (isos1 : is_isosceles_triangle triangle1) 
    (isos2 : is_isosceles_triangle triangle2)
    (equal_angles : equal_base_angles triangle1 triangle2), 
    ¬ similar_triangles triangle1 triangle2) := by
  sorry

end false_statement_l481_481229


namespace length_of_LN_l481_481424

theorem length_of_LN 
  (LM LN : ℝ)
  (h1 : sin (real.pi / 2) = 3 / 5) 
  (h2 : LM = 15) : 
  LN = 25 :=
by
  sorry -- skip the proof

end length_of_LN_l481_481424


namespace sqrt_43_between_6_and_7_l481_481281

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 :=
by
  sorry

end sqrt_43_between_6_and_7_l481_481281


namespace find_angle_B_l481_481715

theorem find_angle_B 
  (A B : ℝ)
  (h1 : B + A = 90)
  (h2 : B = 4 * A) : 
  B = 144 :=
by
  sorry

end find_angle_B_l481_481715


namespace cos_C_max_ab_over_c_l481_481799

theorem cos_C_max_ab_over_c
  (a b c S : ℝ) (A B C : ℝ)
  (h1 : 6 * S = a^2 * Real.sin A + b^2 * Real.sin B)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : S = 0.5 * a * b * Real.sin C)
  : Real.cos C = 7 / 9 := 
sorry

end cos_C_max_ab_over_c_l481_481799


namespace sum_of_squares_distances_constant_l481_481972

theorem sum_of_squares_distances_constant
  (A B C D O P : Point)
  (R : ℝ)
  (tetrahedron : regular_tetrahedron A B C D)
  (sphere_circumscribed : sphere O R)
  (P_on_sphere : point_on_sphere P O R) :
  PA^2 + PB^2 + PC^2 + PD^2 = 8 * R^2 := 
sorry

end sum_of_squares_distances_constant_l481_481972


namespace tangent_product_l481_481512

noncomputable def tangent (x : ℝ) : ℝ := Real.tan x

theorem tangent_product : 
  tangent (20 * Real.pi / 180) * 
  tangent (40 * Real.pi / 180) * 
  tangent (60 * Real.pi / 180) * 
  tangent (80 * Real.pi / 180) = 3 :=
by
  -- Definitions and conditions
  have tg60 := Real.tan (60 * Real.pi / 180) = Real.sqrt 3
  
  -- Add tangent addition, subtraction, and triple angle formulas
  -- tangent addition formula
  have tg_add := ∀ x y : ℝ, tangent (x + y) = (tangent x + tangent y) / (1 - tangent x * tangent y)
  -- tangent subtraction formula
  have tg_sub := ∀ x y : ℝ, tangent (x - y) = (tangent x - tangent y) / (1 + tangent x * tangent y)
  -- tangent triple angle formula
  have tg_triple := ∀ α : ℝ, tangent (3 * α) = (3 * tangent α - tangent α^3) / (1 - 3 * tangent α^2)
  
  -- sorry to skip the proof
  sorry


end tangent_product_l481_481512


namespace rational_terms_expansion_polynomial_coefficient_value_l481_481363

-- Define the given function and assumptions.
def expansion_sum_condition (n : ℕ) : Prop :=
  2^(2*n) - 1 = 255

-- Translating the first part of the problem statement to Lean.
theorem rational_terms_expansion (n : ℕ): 
  expansion_sum_condition n → 
  n = 4 → 
  ∃ (a b c : ℚ), 
  (a = 1 ∧ b = 35/8 ∧ c = 1/256) := sorry

-- Define the polynomial coefficients sum.
def sum_of_coefficients (n : ℕ) (a : ℕ → ℚ) : Prop :=
  ∑ i in finset.range (n + 1), a i = 1

-- Define the alternating coefficients sum.
def alternating_sum_of_coefficients (n : ℕ) (a : ℕ → ℚ) : Prop :=
  ∑ i in finset.range (n + 1), a i * (-1)^i = 81

-- Translating the second part of the problem statement to Lean.
theorem polynomial_coefficient_value (n : ℕ) (a : ℕ → ℚ): 
  expansion_sum_condition n → 
  n = 4 → 
  sum_of_coefficients n a → 
  alternating_sum_of_coefficients n a →
  (a 0 + a 2 + a 4) ^ 2 - (a 1 + a 3) ^ 2 = 81 := sorry

end rational_terms_expansion_polynomial_coefficient_value_l481_481363


namespace ratio_of_perimeters_l481_481266

theorem ratio_of_perimeters
    (length : ℕ) (width : ℕ)
    (folded_length : ℕ := length / 2)
    (small_rectangle_perimeter : ℕ := 2 * (folded_length + width / 2))
    (large_rectangle_perimeter : ℕ := 2 * (folded_length + width))
    (length = 10) (width = 8) :
  (small_rectangle_perimeter : ℚ) / large_rectangle_perimeter = 9 / 13 :=
by sorry

end ratio_of_perimeters_l481_481266


namespace triangle_proof_1_l481_481428

theorem triangle_proof_1
  (A B C D K I : Point)
  (h1 : Triangle A B C)
  (h2 : IsAngleBisector C D)
  (h3 : CircleIntersection C K (CircumCircle (Triangle A B C)))
  (h4 : Incenter I (Triangle A B C)) :
  (1 / (distance I D) - 1 / (distance I K) = 1 / (distance I C)) ∧
  ((distance I C) / (distance I D) - (distance I D) / (distance D K) = 1) :=
sorry

end triangle_proof_1_l481_481428


namespace max_min_values_f_at_a_is_half_range_a_l481_481350

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
def g (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem max_min_values_f_at_a_is_half :
  let a := -1 / 2,
  let max_val := Real.log 2 - 1,
  let min_val := -1 / 2 in
  ∃ x_max x_min ∈ set.Icc 1 Real.exp 1,
  f a x_max = max_val ∧   f a x_min = min_val := sorry

theorem range_a :
  ∀ x1 ∈ set.Icc (-1 : ℝ) 2,
  ∃ x2 ∈ set.Ioi 0,
  ∀ a ∈ set.Ioi (- Real.exp (-6)),
  g x1 < f a x2 := sorry

end max_min_values_f_at_a_is_half_range_a_l481_481350


namespace max_product_843_l481_481919

-- Define the conditions
def is_valid_3_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n.digits ∈ permutations [1, 3, 4, 5, 7, 8].map (λ l, l.take 3)

def is_valid_2_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n.digits ∈ permutations [1, 3, 4, 5, 7, 8].map (λ l, l.drop 3.take 2)

noncomputable def max_product : ℕ :=
  max ( (100 * 8 + 10 * 4 + 3) * (10 * 7 + 5) )
      ( (100 * 7 + 10 * 4 + 3) * (10 * 8 + 5) )

theorem max_product_843 : ∃ n m : ℕ, is_valid_3_digit n ∧ is_valid_2_digit m ∧ n * m = max_product :=
sorry

end max_product_843_l481_481919


namespace probability_of_less_than_5_is_one_half_l481_481224

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l481_481224


namespace area_equality_l481_481083

open Real

namespace Geometry

variable {P Q R A B C L N K M : Type*} [RealInnerProductSpace R]

noncomputable def area_triangle (A B C : R) : ℝ :=
  1 / 2 * abs ((B - A) ⬝ (C - A))

noncomputable def area_quadrilateral (A K N M : R) : ℝ :=
  area_triangle A K N + area_triangle A N M

-- Given conditions
variables
  (ABC_acute : ∀ {A B C K M N L : R}, acute_angle A B C)
  (bisector_A : B.angle_bisector A C = L)
  (circumcircle_A : ∃ (K A B C), circle A B C)
  (LK_perp_AB : L ⊥ AB)
  (LM_perp_AC : L ⊥ AC)

-- Prove the area equivalence
theorem area_equality
  (A B C L N K M : R)
  (h_acute : ∀ {A B C K M N L : R}, acute_angle A B C)
  (h_bisector : bisector_A A B C L)
  (h_circumcircle : circumcircle_A A B C L N K)
  (h_LK_perp : LK_perp_AB L K)
  (h_LM_perp : LM_perp_AC L M) :
  (area_triangle A B C) = (area_quadrilateral A K N M) :=
begin
  sorry,
end

end Geometry

end area_equality_l481_481083


namespace positive_divisors_not_divisible_by_2_l481_481050

/-- The number of positive divisors of 180 that are not divisible by 2 is 6. -/
theorem positive_divisors_not_divisible_by_2 (n : ℕ) (h : n = 180) : 
  ∃ k, k = 6 ∧ (∀ d, d ∣ n → ¬ (2 ∣ d) → d ∈ {d | d ∣ n} ↔ d ∈ {d | 3^i * 5^j}) :=
begin
  sorry
end

end positive_divisors_not_divisible_by_2_l481_481050


namespace problem1_problem2_l481_481668

-- Define the function f with the given conditions
def f (x : ℝ) : ℝ := sorry

-- The given conditions for the function f
axiom f_equation (x y : ℝ) : f(x + y) + f(x - y) = 2 * f(x) * Real.cos y
axiom f_at_zero : f(0) = 1
axiom f_at_pi_div_2 : f(Real.pi / 2) = 2

-- Defining the function g
noncomputable def g (x : ℝ) : ℝ :=
  (4 * f(x) - 2 * (3 - Real.sqrt 3) * Real.sin x) /
  (Real.sin x + Real.sqrt (1 - Real.sin x))

-- Problem 1: Prove that f(-π/2) = -2
theorem problem1 : f(-Real.pi / 2) = -2 := sorry

-- Problem 2: Prove that the maximum value of g(x) for x ∈ [0, π/3] ∪ [5π/6, π] is 2(√3 + 1)
theorem problem2 : 
  ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 3) ∪ Set.Icc (5 * Real.pi / 6) Real.pi → 
  g(x) ≤ 2 * (Real.sqrt 3 + 1) := sorry

end problem1_problem2_l481_481668


namespace token_arrangement_impossible_l481_481801

theorem token_arrangement_impossible :
  ∀ (black white : Type) (circle : list (black ⊕ white)),
  (∀ n, (list.length circle = 2 * n) → n = 10) →
  (∀ i : ℕ, i < list.length circle → 
    (circle.nth i).is_some →
    (circle.nth ((i + n) % list.length circle)).is_some →
    (circle.nth i) ≠ (circle.nth ((i + n) % list.length circle))) →
  (∀ i : ℕ, i < list.length circle →
    (circle.nth i).is_some →
    (circle.nth (i + 1 % list.length circle)).is_none) →
  false :=
by
  sorry

end token_arrangement_impossible_l481_481801


namespace planted_fraction_is_correct_l481_481677

noncomputable def right_triangle_field_planted_fraction (a b : ℕ) (x : ℝ) : ℝ :=
  let hypotenuse := real.sqrt (a^2 + b^2)
  let total_area := (a * b) / 2
  let nearest_dist := 3
  if a = 5 ∧ b = 12 ∧ nearest_dist = 3 then
    let side_square := (7 - real.sqrt(13)) / 2
    let area_square := side_square^2
    let planted_area := total_area - area_square
    planted_area / total_area
  else
    0

-- Define the main theorem
theorem planted_fraction_is_correct :
  right_triangle_field_planted_fraction 5 12 ((7 - real.sqrt(13)) / 2) = 
  \(\frac{30}{30} - \frac{\left(\frac{7 - \sqrt{13}}{2}\right)^2}{30}\) :=
by
  sorry

end planted_fraction_is_correct_l481_481677


namespace sum_first_100_terms_is_l481_481911

open Nat

noncomputable def seq (a_n : ℕ → ℤ) : Prop :=
  a_n 2 = 2 ∧ ∀ n : ℕ, n > 0 → a_n (n + 2) + (-1)^(n + 1) * a_n n = 1 + (-1)^n

def sum_seq (f : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum f

theorem sum_first_100_terms_is :
  ∃ (a_n : ℕ → ℤ), seq a_n ∧ sum_seq a_n 100 = 2550 :=
by
  sorry

end sum_first_100_terms_is_l481_481911


namespace solve_distance_between_homes_l481_481650

noncomputable def home_distance (cinema_distance_xh : ℕ) (cinema_distance_xl : ℕ) : ℕ :=
  cinema_distance_xh + cinema_distance_xl

theorem solve_distance_between_homes :
  (52 : ℕ) * 18 + (70 : ℕ) * 18 = 2196 := by
  calc
    (52 : ℕ) * 18 + (70 : ℕ) * 18
        = 936 + 1260 : by sorry
    ... = 2196 : by sorry

end solve_distance_between_homes_l481_481650


namespace indeterminate_original_value_percentage_l481_481950

-- Lets define the problem as a structure with the given conditions
structure StockData where
  yield_percent : ℚ
  market_value : ℚ

-- We need to prove this condition
theorem indeterminate_original_value_percentage (d : StockData) :
  d.yield_percent = 8 ∧ d.market_value = 125 → false :=
by
  sorry

end indeterminate_original_value_percentage_l481_481950


namespace polynomial_roots_sum_l481_481190

theorem polynomial_roots_sum (a b c d e : ℝ) (h₀ : a ≠ 0)
  (h1 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
  (h2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h3 : a * 2^4 + b * 2^3 + c * 2^2 + d * 2 + e = 0) :
  (b + d) / a = -2677 := 
sorry

end polynomial_roots_sum_l481_481190


namespace partition_positive_integers_l481_481823

theorem partition_positive_integers (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ (A B C D : set ℕ), set.univ = A ∪ B ∪ C ∪ D ∧
  disjoint A B ∧ disjoint A C ∧ disjoint A D ∧ disjoint B C ∧ disjoint B D ∧ disjoint C D ∧
  (∀ a ∈ A, ∀ b ∈ A, abs (a - b) ≠ m ∧ abs (a - b) ≠ n ∧ abs (a - b) ≠ m + n) ∧
  (∀ a ∈ B, ∀ b ∈ B, abs (a - b) ≠ m ∧ abs (a - b) ≠ n ∧ abs (a - b) ≠ m + n) ∧
  (∀ a ∈ C, ∀ b ∈ C, abs (a - b) ≠ m ∧ abs (a - b) ≠ n ∧ abs (a - b) ≠ m + n) ∧
  (∀ a ∈ D, ∀ b ∈ D, abs (a - b) ≠ m ∧ abs (a - b) ≠ n ∧ abs (a - b) ≠ m + n) :=
sorry

end partition_positive_integers_l481_481823


namespace area_ratio_l481_481265

variables {A B C D E F : Point}  -- Define points A, B, C, D, E, F
variables {BA' BC' BD' : Point}  -- Define points BA', BC', BD' on edges of tetrahedron ABCD
variables (plane : Plane)  -- Define the plane
 
-- Hypothesis: The plane intersects edges BA, BC, BD in the ratio 2:1
axiom intersect_ratios : 
  (ratio (BA' : line_segment A B)) = 2 ∧ 
  (ratio (BC' : line_segment B C)) = 2 ∧ 
  (ratio (BD' : line_segment B D)) = 2

-- Hypothesis: The plane intersects lines CD and AC at points E and F
axiom plane_intersects : 
  Plane_intersects_line plane (line_segment C D) E ∧ 
  Plane_intersects_line plane (line_segment A C) F

-- Define the area function for triangles
noncomputable def area (P Q R : Point) : ℝ := sorry

-- Proposition: The ratio of the areas of triangles EFC and ACD is 1/9
theorem area_ratio (h1 : intersect_ratios) (h2 : plane_intersects) :
  (area E F C) / (area A C D) = 1 / 9 :=
sorry

end area_ratio_l481_481265


namespace find_radius_of_inscribed_sphere_l481_481630

variables (a b c s : ℝ)

theorem find_radius_of_inscribed_sphere
  (h1 : a + b + c = 18)
  (h2 : 2 * (a * b + b * c + c * a) = 216)
  (h3 : a^2 + b^2 + c^2 = 108) :
  s = 3 * Real.sqrt 3 :=
by
  sorry

end find_radius_of_inscribed_sphere_l481_481630


namespace commutative_star_associative_star_l481_481332

def star (x y : ℝ) : ℝ := (x * y) / (x + y)

theorem commutative_star (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  star x y = star y x :=
by
  sorry

theorem associative_star (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  star (star x y) z = star x (star y z) :=
by
  sorry

end commutative_star_associative_star_l481_481332


namespace correct_options_l481_481929

variables {A B C P O : Type} 
variables [InnerProductSpace ℝ E]
variables (a b : E)

def OptionACondition := (∥b∥ = 2 * ∥a∥ ∧ ∥a∥ = 1)
#check OptionACondition

def OptionA := (∥ (a - 2 • b) ∥ ≤ 5)

def OptionBCondition := (dist A C = 3 ∧ dist A B = 1) -- O is the circumcenter must be introduced
#check OptionBCondition

def OptionB := (dist B C * dist A O = 4)

def OptionCCondition := ∃ k : ℤ, ∀ x, tan(2 * x - π / 3) = 0 
#check OptionCCondition

def OptionC := False 

def OptionDCondition := PointInsideTriangle P A B C ∧ dot (P-A) (P-B) = dot (P-C) (P-B) ∧ dot (P-A) (P-C) = dot (P-B) (P-C)
#check OptionDCondition

def OptionD := IsOrthocenter P A B C   

theorem correct_options (hA : OptionACondition ∧ OptionA)
                        (hB : OptionBCondition ∧ OptionB)
                        (hC : ¬OptionCCondition ∧ OptionC)
                        (hD : OptionDCondition ∧ OptionD)
                        : (hA ∨ hB ∨ hD) := sorry

end correct_options_l481_481929


namespace probability_of_multiple_of_3_is_one_third_l481_481563

noncomputable def probability_multiple_of_3 : ℚ :=
  let total_tickets := finset.range 22
  let multiples_of_3 := total_tickets.filter (λ x, x % 3 = 0)
  (multiples_of_3.card : ℚ) / total_tickets.card

theorem probability_of_multiple_of_3_is_one_third :
  probability_multiple_of_3 = 1 / 3 :=
by
  sorry

end probability_of_multiple_of_3_is_one_third_l481_481563


namespace cong_triangles_eq_area_l481_481931

/-- Definition of congruent triangles: Two triangles are congruent if they have the same shape and size. -/
structure Triangle :=
  (a b c : ℝ)  -- side lengths of the triangle

def congruent (Δ1 Δ2 : Triangle) : Prop :=
  (Δ1.a = Δ2.a) ∧ (Δ1.b = Δ2.b) ∧ (Δ1.c = Δ2.c)

/-- Definition of the area of a triangle using Heron's formula. -/
def area (Δ : Triangle) : ℝ :=
  let s := (Δ.a + Δ.b + Δ.c) / 2
  in Math.sqrt (s * (s - Δ.a) * (s - Δ.b) * (s - Δ.c))

/-- Theorem: The area of congruent triangles is always equal. -/
theorem cong_triangles_eq_area (Δ1 Δ2 : Triangle) 
  (h : congruent Δ1 Δ2) : area Δ1 = area Δ2 := by
  sorry  -- Proof omitted

end cong_triangles_eq_area_l481_481931


namespace angle_ratio_1_2_4_l481_481832

-- Definitions and conditions from part a)
variables (A B C a b c : Real)
variables (triangle_angles : A + B + C = Real.pi)
variables (side_ineq : a < b ∧ b < c)
variables (eq1 : b / a = |b^2 + c^2 - a^2| / (b * c))
variables (eq2 : c / b = |c^2 + a^2 - b^2| / (c * a))
variables (eq3 : a / c = |a^2 + b^2 - c^2| / (a * b))

-- Target proof statement
theorem angle_ratio_1_2_4 : (A, B, C : Real),
  A + B + C = Real.pi → a < b ∧ b < c → 
  (b / a = |b^2 + c^2 - a^2| / (b * c)) →
  (c / b = |c^2 + a^2 - b^2| / (c * a)) →
  (a / c = |a^2 + b^2 - c^2| / (a * b)) →
  B = 2 * A ∧ C = 4 * A :=
begin
  sorry
end

end angle_ratio_1_2_4_l481_481832


namespace bethany_total_hours_l481_481992

-- Define the hours Bethany rode on each set of days
def hours_mon_wed_fri : ℕ := 3  -- 1 hour each on Monday, Wednesday, and Friday
def hours_tue_thu : ℕ := 1  -- 30 min each on Tuesday and Thursday
def hours_sat : ℕ := 2  -- 2 hours on Saturday

-- Define the total hours per week
def total_hours_per_week : ℕ := hours_mon_wed_fri + hours_tue_thu + hours_sat

-- Define the total hours in 2 weeks
def total_hours_in_2_weeks : ℕ := total_hours_per_week * 2

-- Prove that the total hours in 2 weeks is 12
theorem bethany_total_hours : total_hours_in_2_weeks = 12 :=
by
  -- Replace the definitions with their values and check the equality
  rw [total_hours_in_2_weeks, total_hours_per_week, hours_mon_wed_fri, hours_tue_thu, hours_sat]
  simp
  norm_num
  sorry

end bethany_total_hours_l481_481992


namespace forest_triangle_side_lengths_l481_481412

-- Definitions of the conditions
def speed_kmh := 4  -- speed in km/h
def time_Ivo_h := 15 / 60  -- Ivo's time in hours
def time_Petr_h := 12 / 60  -- Petr's time in hours

def distance_Ivo_km := speed_kmh * time_Ivo_h  -- Ivo's distance in km
def distance_Petr_km := speed_kmh * time_Petr_h  -- Petr's distance in km

-- Definitions of the side lengths as variables
variable {a b : ℝ}  -- in kilometers

-- Condition to check isosceles triangle
def is_isosceles_triangle (a b : ℝ) : Prop :=
  ∃ h, b + h = distance_Ivo_km ∧ a + h = distance_Petr_km  -- Assume h is the half-base distance

-- Main theorem to prove
theorem forest_triangle_side_lengths :
  is_isosceles_triangle a b →
  (a ≈ 467 / 1000 ∧ b ≈ 667 / 1000) ∨ (a ≈ 733 / 1000 ∧ b ≈ 533 / 1000) :=
sorry

end forest_triangle_side_lengths_l481_481412


namespace average_growth_rate_income_prediction_l481_481779

-- Define the given conditions
def income2018 : ℝ := 20000
def income2020 : ℝ := 24200
def growth_rate : ℝ := 0.1
def predicted_income2021 : ℝ := 26620

-- Lean 4 statement for the first part of the problem
theorem average_growth_rate :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  growth_rate = 0.1 :=
by
  intros h
  sorry

-- Lean 4 statement for the second part of the problem
theorem income_prediction :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  (growth_rate = 0.1) →
  (income2018 * (1 + growth_rate)^3 = predicted_income2021) :=
by
  intros h1 h2
  sorry

end average_growth_rate_income_prediction_l481_481779


namespace decrease_hours_worked_percentage_l481_481264

-- Definitions based on conditions
variable (W H : ℝ) -- Original hourly wage W and original hours worked H
variable (newW := 1.5 * W) -- New hourly wage after 50% increase
variable (newH := H / 1.5) -- New hours worked to maintain same income

-- The theorem statement
theorem decrease_hours_worked_percentage :
  (H - newH) / H * 100 ≈ 33.33 := by
sorry

end decrease_hours_worked_percentage_l481_481264


namespace centroid_of_quadrant_arc_l481_481322

def circle_equation (R : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = R^2
def density (ρ₀ x y : ℝ) : ℝ := ρ₀ * x * y

theorem centroid_of_quadrant_arc (R ρ₀ : ℝ) :
  (∃ x y, circle_equation R x y ∧ x ≥ 0 ∧ y ≥ 0) →
  ∃ x_c y_c, x_c = 2 * R / 3 ∧ y_c = 2 * R / 3 :=
sorry

end centroid_of_quadrant_arc_l481_481322


namespace fruit_days_l481_481747

/-
  Henry and his brother believe in the famous phrase, "An apple a day, keeps the doctor away." 
  Henry's sister, however, believes that "A banana a day makes the trouble fade away" 
  and their father thinks that "An orange a day will keep the weaknesses at bay." 
  A box of apples contains 14 apples, a box of bananas has 20 bananas, and a box of oranges contains 12 oranges. 

  If Henry and his brother eat 1 apple each a day, their sister consumes 2 bananas per day, 
  and their father eats 3 oranges per day, how many days can the family of four continue eating fruits 
  if they have 3 boxes of apples, 4 boxes of bananas, and 5 boxes of oranges? 

  However, due to seasonal changes, oranges are only available for the first 20 days. 
  Moreover, Henry's sister has decided to only eat bananas on days when the day of the month is an odd number. 
  Considering these constraints, determine the total number of days the family of four can continue eating their preferred fruits.
-/

def apples_per_box := 14
def bananas_per_box := 20
def oranges_per_box := 12

def apples_boxes := 3
def bananas_boxes := 4
def oranges_boxes := 5

def daily_apple_consumption := 2
def daily_banana_consumption := 2
def daily_orange_consumption := 3

def orange_availability_days := 20

def odd_days_in_month := 16

def total_number_of_days : ℕ :=
  let total_apples := apples_boxes * apples_per_box
  let total_bananas := bananas_boxes * bananas_per_box
  let total_oranges := oranges_boxes * oranges_per_box
  
  let days_with_apples := total_apples / daily_apple_consumption
  let days_with_bananas := (total_bananas / (odd_days_in_month * daily_banana_consumption)) * 30
  let days_with_oranges := if total_oranges / daily_orange_consumption > orange_availability_days then orange_availability_days else total_oranges / daily_orange_consumption
  min (min days_with_apples days_with_oranges) (days_with_bananas / 30 * 30)

theorem fruit_days : total_number_of_days = 20 := 
  sorry

end fruit_days_l481_481747


namespace num_selection_schemes_l481_481157

-- Given conditions and definitions needed for the problem
def num_total_managers : Nat := 6
def num_selected_managers : Nat := 4

def manager_not_in_wenzhou (a : Type) : Prop := ¬(a = 'A')
def manager_not_in_jinhua (b : Type) : Prop := ¬(b = 'B')

-- Statement of the problem
theorem num_selection_schemes (num_total_managers = 6) (num_selected_managers = 4) 
  (mgrs: Fin 6 → Char) :
  (∀ i, manager_not_in_wenzhou (mgrs i) ∧ manager_not_in_jinhua (mgrs i)) →
  num_selection_schemes mgrs = 252 :=
sorry

end num_selection_schemes_l481_481157


namespace arithmetic_sequence_properties_l481_481347

noncomputable def arithmetic_sequence (a_n : ℕ → ℤ) :=
  ∀ n, ∃ a₁ d, a_n n = a₁ + (n - 1) * d

def sum_of_terms (a_n : ℕ → ℤ) (n : ℕ) := ∑ i in finset.range(n + 1), a_n i

theorem arithmetic_sequence_properties (a_n : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith_seq : arithmetic_sequence a_n) (hS3 : S 3 = 0) (hS5 : S 5 = -5) :
  (∀ n, a_n n = 2 - n) ∧ (∀ n, (∑ i in finset.range(n + 1), a_n (3 * i + 1)) = (n + 1) * (2 - 3 * n) / 2) :=
by
  sorry

end arithmetic_sequence_properties_l481_481347


namespace expected_value_is_one_third_l481_481947

noncomputable def expected_value_of_winnings : ℚ :=
  let p1 := (1/6 : ℚ)
  let p2 := (1/6 : ℚ)
  let p3 := (1/6 : ℚ)
  let p4 := (1/6 : ℚ)
  let p5 := (1/6 : ℚ)
  let p6 := (1/6 : ℚ)
  let winnings1 := (5 : ℚ)
  let winnings2 := (5 : ℚ)
  let winnings3 := (0 : ℚ)
  let winnings4 := (0 : ℚ)
  let winnings5 := (-4 : ℚ)
  let winnings6 := (-4 : ℚ)
  (p1 * winnings1 + p2 * winnings2 + p3 * winnings3 + p4 * winnings4 + p5 * winnings5 + p6 * winnings6)

theorem expected_value_is_one_third : expected_value_of_winnings = 1 / 3 := by
  sorry

end expected_value_is_one_third_l481_481947


namespace proof_john_more_marbles_l481_481292

def initial_marbles := { Ben: 18, John: 17, Lisa: 12, Max: 9 }

def final_marbles (initial: Nat) (percent: Nat) : Nat :=
  initial - ((percent * initial) / 100)

def final_marbles_ben (initial: Nat) (percent_john: Nat) (percent_lisa: Nat) : Nat :=
  initial - (percent_john * initial / 100) - (percent_lisa * initial / 100)

def marbles_john_after_ben (initial_ben: Nat) (initial_john: Nat) (percent_ben_to_john: Nat) : Nat :=
  initial_john + (percent_ben_to_john * initial_ben / 100)

def marbles_john_final (initial_john: Nat) (percent_ben_to_john: Nat) (percent_john_to_max: Nat) (initial_ben: Nat)
                       (percent_lisa_to_john: Nat) (initial_lisa: Nat) : Nat :=
  marbles_john_after_ben initial_ben initial_john percent_ben_to_john - 
  (percent_john_to_max * (percent_ben_to_john * initial_ben / 100) / 100) + 
  (percent_lisa_to_john * initial_lisa / 100 + (percent_ben_to_john * initial_ben / 100))

def more_marbles_john_than_ben (initials: {Ben: Nat, John: Nat, Lisa: Nat, Max: Nat}) := 
  (marbles_john_final initials.John 50 65 initials.Ben 20 initials.Lisa) - 
  (final_marbles_ben initials.Ben 50 25) = 22.5

theorem proof_john_more_marbles (initials: {Ben: Nat, John: Nat, Lisa: Nat, Max: Nat}) :
  more_marbles_john_than_ben initials := 
by
  intros
  rw [initial_marbles_initials, final_marbles_ben, final_marbles, marbles_john_after_ben, marbles_john_final]
  -- input the conditions (a) and correct answer (b)
  exact sorry

end proof_john_more_marbles_l481_481292


namespace fractional_part_of_sqrt_l481_481116

theorem fractional_part_of_sqrt (n : ℕ) (h1 : 100 < n) : (Nat.floor ((sqrt((↑n ^ 2 + 3 * ↑n + 1 : ℝ)) - Nat.floor (sqrt (↑n ^ 2 + 3 * ↑n + 1 : ℝ)))) * 100) = 50 :=
sorry

end fractional_part_of_sqrt_l481_481116


namespace grasshopper_reach_3_after_2003_jumps_l481_481619

theorem grasshopper_reach_3_after_2003_jumps :
  (∀ n : ℤ, (∃ k : ℤ, n = k * 4) → False) → 
  (∃ k : ℤ, (∑ i in (finset.range 2003).attach, (if i.val % 2 = 0 then 1 else 5) * (-1) ^ (i.val / 2) = k)) → 
  False :=
by
  sorry

end grasshopper_reach_3_after_2003_jumps_l481_481619


namespace quadrilateral_area_ratio_correctness_l481_481142

noncomputable def problem (r : ℝ) (a b c : ℕ) : Prop :=
EG_is_diameter (EG : ℝ) : (EG = 2 * r) ∧
angle_FEG (FEG : ℝ) : (FEG = 25) ∧
angle_GEF (GEF : ℝ) : (GEF = 35) ∧
area_ratio_expression : (a + b + c = 5)

theorem quadrilateral_area_ratio_correctness (r : ℝ) (a b c : ℕ) :
  problem r a b c :=
by
  sorry

end quadrilateral_area_ratio_correctness_l481_481142


namespace sum_of_xyz_l481_481118

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : 1/x + y + z = 3) 
  (h2 : x + 1/y + z = 3) 
  (h3 : x + y + 1/z = 3) : 
  ∃ m n : ℕ, m = 9 ∧ n = 2 ∧ Nat.gcd m n = 1 ∧ 100 * m + n = 902 := 
sorry

end sum_of_xyz_l481_481118


namespace find_k_l481_481772

noncomputable def expr_to_complete_square (x : ℝ) : ℝ :=
  x^2 - 6 * x

theorem find_k (x : ℝ) : ∃ a h k, expr_to_complete_square x = a * (x - h)^2 + k ∧ k = -9 :=
by
  use 1, 3, -9
  -- detailed steps of the proof would go here
  sorry

end find_k_l481_481772


namespace song_duration_l481_481386

variable (x : ℕ)

theorem song_duration (h1 : 25 + 10 = 35) (h2 : 35 * x = 105) : x = 3 :=
by {
  have h3 : 35 = 25 + 10 := h1,
  sorry
}

end song_duration_l481_481386


namespace locus_midpoints_of_XY_l481_481617

noncomputable def cube_vertices : list (ℝ × ℝ × ℝ) :=
[
  (0, 0, 0),   -- A
  (1, 0, 0),   -- B
  (1, 1, 0),   -- C
  (0, 1, 0),   -- D
  (0, 0, 1),   -- A'
  (1, 0, 1),   -- B'
  (1, 1, 1),   -- C'
  (0, 1, 1)    -- D'
]

def point_X (t : ℝ) : ℝ × ℝ × ℝ :=
if t < 1 then (t, 0, 0)
else if t < 2 then (1, t - 1, 0)
else if t < 3 then (3 - t, 1, 0)
else (0, 4 - t, 0)

def point_Y (t : ℝ) : ℝ × ℝ × ℝ :=
if t < 1 then (1, t, 1)
else if t < 2 then (1, 1, 2 - t)
else if t < 3 then (1, 3 - t, 0)
else (1, 0, t - 3)

def midpoint (X Y : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((X.1 + Y.1) / 2, (X.2 + Y.2) / 2, (X.3 + Y.3) / 2)

-- The proof statement
theorem locus_midpoints_of_XY :
  ∀ (t : ℝ), 
  let X := point_X t,
      Y := point_Y t,
      M := midpoint X Y in
  (M.1, M.2, M.3) ∈ {p : ℝ × ℝ × ℝ | 
    (p.3 = 1/2 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) ∨ 
    (p.3 = 0 ∧ p.1 = 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) ∨ 
    (p.1 = 1/2 ∧ p.2 = 1/2 ∧ 0 ≤ p.3 ∧ p.3 ≤ 1) } :=
sorry

end locus_midpoints_of_XY_l481_481617


namespace min_value_expression_l481_481834

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  \frac{a + b}{c} + \frac{a + c}{b} + \frac{b + c}{a} >= 6 :=
sorry

end min_value_expression_l481_481834


namespace f_parity_and_monotonicity_l481_481495

theorem f_parity_and_monotonicity 
  (f : ℝ → ℝ) 
  (hf_def : ∀ x, f x = 2 * x^2 - 1 / x^2) :
  (∀ x, f (-x) = f x) ∧ (∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2) :=
by {
  sorry,
}

end f_parity_and_monotonicity_l481_481495


namespace find_m_l481_481723

theorem find_m (m : ℝ) 
  (g : ℝ → ℝ)
  (h1 : ∀ x, g x = m - exp (- x))
  (h2 : g 0 + g (-log 2) = 1) : m = 2 := 
sorry

end find_m_l481_481723


namespace second_group_persons_count_l481_481249

-- Declare the constants based on the conditions
constant n₁ : ℕ := 69      -- Number of persons in the first group
constant d₁ : ℕ := 12      -- Number of days the first group works
constant h₁ : ℕ := 5       -- Number of hours per day the first group works

constant d₂ : ℕ := 23      -- Number of days the second group works
constant h₂ : ℕ := 6       -- Number of hours per day the second group works

-- The total work done is equal for both groups
theorem second_group_persons_count : 
  ∃ n₂ : ℕ, n₁ * d₁ * h₁ = n₂ * d₂ * h₂ ∧ n₂ = 30 :=
by 
  use (30 : ℕ)
  rw [Nat.mul_eq, Nat.mul_eq, Nat.mul_eq]
  -- Proof will go here
  sorry

end second_group_persons_count_l481_481249


namespace general_formula_an_sum_Tn_l481_481031

/-
Given the following conditions:
1. The sequence {a_n} is an arithmetic sequence.
2. a_5 = 3 * a_2.
3. S_7 = 14 * a_2 + 7.
4. The sequence {a_n + b_n} is a geometric sequence with first term 1 and common ratio 2.
-/

variables (a : Nat → ℤ) (b : Nat → ℤ) (S : Nat → ℤ) (n : ℕ)

-- Condition 1: Arithmetic sequence definition
def arithmetic_seq := ∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d
  
-- Condition 2: a_5 = 3 * a_2
def condition_a2 := a 5 = 3 * a 2

-- Condition 3: S_7 = 14 * a_2 + 7
def condition_S7 := S 7 = 14 * a 2 + 7
  
-- Condition 4: {a_n + b_n} is a geometric sequence with first term 1 and common ratio 2
def geometric_seq := ∀ n : ℕ, a n + b n = 2 ^ n

-- To prove: General formula for {a_n} is a_n = 2n - 1
theorem general_formula_an (h1: arithmetic_seq a) (h2: condition_a2 a) (h3: condition_S7 a S) : 
  ∀ n : ℕ, a n = 2 * n - 1 :=
sorry

-- To prove: Sum T_n is given by the specified formula
theorem sum_Tn (h4: geometric_seq a b) : 
  ∀ n : ℕ, ∑ i in Finset.range n, (-1)^i * b i * (a i + b i) = 
    ( -1 + (-4)^n) / 5 - 1 / 9 - (6 * n - 1) * (-2)^n / 9 :=
sorry

end general_formula_an_sum_Tn_l481_481031


namespace sum_of_integer_solutions_eq_zero_l481_481327

theorem sum_of_integer_solutions_eq_zero :
  (∑ x in {x : Int | x^4 - 33 * x^2 + 272 = 0}.toFinset, x) = 0 :=
sorry

end sum_of_integer_solutions_eq_zero_l481_481327


namespace IK_eq_IA_l481_481072

open Real EuclideanGeometry

variables {A B C I T E F K : Point}

-- Definitions based on the conditions
def incenter (A B C : Point) : Point := sorry
def incircle (A B C : Point) : Circle := sorry
def circumcircle (A B C : Point) : Circle := sorry
def parallel (L M : Line) : Prop := sorry
def line_through (P Q : Point) : Line := sorry
def is_concyclic (P Q R S : Point) : Prop := sorry

-- Given
axiom incenter_def : incenter A B C = I
axiom AI_intersects_BC_at_T : Line_through A I ∩ Line_through B C = T
axiom incircle_touches_BC_at_E : incircle A B C ∩ Line_through B C = E
axiom line_through_A_parallel_to_BC_intersects_circumcircle_at_F : is_parallel (Line_through A E) (Line_through B C) ∧ Line_through A E ∩ circumcircle A B C = F
axiom AETK_concyclic : is_concyclic A E T K

-- Goal
theorem IK_eq_IA : dist I K = dist I A := sorry

end IK_eq_IA_l481_481072


namespace walter_zoo_visits_seal_time_l481_481198

theorem walter_zoo_visits_seal_time (S : ℕ) (h1 : 4 * 20 ≤ S * 25) 
  (total_time : S + 8 * S + 13 + S / 2 + 3 * S = 260) :
  (S = 20) ∧ ((S + S / 2) = 30) :=
begin
  -- Let S be the initial time Walter spends looking at the seals
  have h_equation : 12.5 * S = 260 - 13,
  {
    sorry
  },
  -- Solving for S,
  have h_S_approx : S ≈ 20,
  {
    sorry
  },
  -- Total time looking at seals
  have total_seal_time : S + S / 2 = 30,
  {
    sorry
  },
  -- Combining results
  apply_and_intro h_S_approx total_seal_time,
end

end walter_zoo_visits_seal_time_l481_481198


namespace percentage_of_women_do_not_speak_french_l481_481601

-- Defining conditions
def total_employees : ℕ := 100
def men_percentage : ℕ := 70
def men_speak_french_percentage : ℕ := 50
def total_speak_french_percentage : ℕ := 40

-- Definitions based on conditions
def total_men := (total_employees * men_percentage) / 100
def total_women := total_employees - total_men
def men_speak_french := (total_men * men_speak_french_percentage) / 100
def total_speak_french := (total_employees * total_speak_french_percentage) / 100
def women_speak_french := total_speak_french - men_speak_french
def women_do_not_speak_french := total_women - women_speak_french

-- The main statement
theorem percentage_of_women_do_not_speak_french : 
  (women_do_not_speak_french * 100 / total_women : ℚ) ≈ 83.33 := 
sorry

end percentage_of_women_do_not_speak_french_l481_481601


namespace odd_function_f_l481_481348

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then exp x else -exp (-x)

theorem odd_function_f (x : ℝ) (h : x < 0) : f x = -exp (-x) :=
by
  unfold f
  simp [h]
  sorry

end odd_function_f_l481_481348


namespace factory_daily_earnings_l481_481408

def num_original_machines : ℕ := 3
def original_machine_hours : ℕ := 23
def num_new_machines : ℕ := 1
def new_machine_hours : ℕ := 12
def production_rate : ℕ := 2 -- kg per hour per machine
def price_per_kg : ℕ := 50 -- dollars per kg

theorem factory_daily_earnings :
  let daily_production_original := num_original_machines * original_machine_hours * production_rate,
      daily_production_new := num_new_machines * new_machine_hours * production_rate,
      total_daily_production := daily_production_original + daily_production_new,
      daily_earnings := total_daily_production * price_per_kg
  in
  daily_earnings = 8100 :=
by
  sorry

end factory_daily_earnings_l481_481408


namespace eligible_to_retire_in_2007_l481_481234

noncomputable def rule_of_70 (current_year : ℕ) (hire_year : ℕ) (hire_age : ℕ) : Prop :=
  let age := current_year - hire_year + hire_age
  let years_of_employment := current_year - hire_year
  age + years_of_employment >= 70

theorem eligible_to_retire_in_2007 :
  ∀ (hire_year hire_age : ℕ), hire_year = 1988 → hire_age = 32 → rule_of_70 2007 hire_year hire_age := 
by 
  intros hire_year hire_age h1 h2
  dsimp [rule_of_70]
  rw [h1, h2]
  simp
  norm_num
  sorry

end eligible_to_retire_in_2007_l481_481234


namespace smaller_angle_at_3_45_l481_481571

/-
  The problem is to prove the degree measure of the smaller angle between the hour hand and the minute hand
  of a clock at 3:45 p.m. on a 12-hour analog clock is 157.5 degrees.
-/

noncomputable def minute_hand_angle (minutes : ℕ) : ℝ := minutes * 6
noncomputable def hour_hand_angle (hours : ℕ) (minutes : ℕ) : ℝ := (hours * 30) + (minutes * 0.5)

theorem smaller_angle_at_3_45 : 
  let minute_angle := minute_hand_angle 45 in
  let hour_angle := hour_hand_angle 3 45 in
  let angle_diff := abs (minute_angle - hour_angle) in
  min angle_diff (360 - angle_diff) = 157.5 :=
by
  sorry

end smaller_angle_at_3_45_l481_481571


namespace midpoint_line_intersect_l481_481479

-- Define the convex quadrilateral and its properties
variables {A B C D E F : Type}
variables [affine_space A D]
variables [affine_space B C]

-- Declare necessary conditions and entities
variables (h1 : is_convex_quad A B C D)
variables (h2 : ∞AD ≠ ∞BC) -- Sides AD and BC are not parallel
variables (h3 : meet (circle_diameter A B) (circle_diameter C D) = {E, F}) -- Circles meet at E and F

def omegaE := circle (foot (perpendicular E A B)) (foot (perpendicular E B C)) (foot (perpendicular E C D))
def omegaF := circle (foot (perpendicular F C D)) (foot (perpendicular F D A)) (foot (perpendicular F A B))

def midpoint (X Y : Type) := (X + Y) / 2

theorem midpoint_line_intersect 
  (hE : E ∈ omegaE)
  (hF : F ∈ omegaF)
  (hM : meet (circle_intersections omegaE omegaF)) (midpoint E F)) : 
  midpoint E F ∈ meet (circle_intersections omegaE omegaF) := 
sorry

end midpoint_line_intersect_l481_481479


namespace vector_sum_zero_l481_481336
-- We use a broad import to include necessary libraries.

-- Define the necessary conditions
variable (O : Point)  -- Point O inside the polyhedron

noncomputable def polyhedron : Type := {
  faces : List Face,
  convex : convexpolyhedron,
  inside_point : O ∈ interior polyhedron
}

-- Define the vector sum problem
theorem vector_sum_zero (polyhedron : Type) :
  let vectors := polyhedron.faces.map (λ F, F.area * F.normal) in
  vectors.sum = 0 := by sorry

end vector_sum_zero_l481_481336


namespace max_quadratic_equations_l481_481644

theorem max_quadratic_equations :
  let numbers := {n : ℕ | 300 ≤ n ∧ n < 400 ∨ 500 ≤ n ∧ n < 600 ∨ 700 ≤ n ∧ n < 800 ∨ 900 ≤ n ∧ n < 1000}
  ∃ (max_count : ℕ), (∀ (a b c ∈ numbers),
  (b^2 - 4 * a * c ≥ 0) → ∃ count, count ≤ max_count) ∧ max_count = 100 :=
by 
  sorry

end max_quadratic_equations_l481_481644


namespace same_function_B_same_function_D_different_function_A_different_function_C_l481_481284

-- Given conditions for the problem
def f_A (x : ℝ) := x + 1
def g_A (x : ℝ) := x + 2

def f_B (x : ℝ) := abs x
def g_B (x : ℝ) := real.sqrt (x^2)

def f_C (x : ℝ) := x^2
def g_C (x : ℝ) := if x = 0 then 0 else (x^3 / x)

def f_D (x : ℝ) := x^2
def g_D (t : ℝ) := t^2

-- Prove that options B and D represent the same functions
theorem same_function_B : ∀ (x : ℝ), f_B x = g_B x := by
  sorry

theorem same_function_D : ∀ (x t : ℝ), f_D x = g_D t := by
  sorry

-- Prove that options A and C do not represent the same functions
theorem different_function_A : ∃ (x : ℝ), f_A x ≠ g_A x := by
  sorry

theorem different_function_C : ∀ (x : ℝ), f_C x ≠ g_C x := by
  sorry

end same_function_B_same_function_D_different_function_A_different_function_C_l481_481284


namespace find_hyperbola_equation_l481_481739

noncomputable def hyperbola_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (b_nat : b ∈ Nat) :
  Exists (λ h : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →
    (| (x, y) | < 5) → 
    (Exists (λ P F1 F2 : ℝ, | P - F1 | ∈ { p | p ^ 2 = | P - F2 | ^ 2 * | F1 - F2 | ^ 2 }))) :=
begin
  sorry
end

theorem find_hyperbola_equation :
  ∃ (a b : ℝ), (a = 2) ∧ (b = 1) ∧ (a > 0) ∧ (b > 0) ∧ (b ∈ Nat) ∧ ∀ x y : ℝ, 
    (x^2 / a^2 - y^2 / b^2 = 1) ∧ 
    (y = (b / 2) * x) →
    clean_theorem :=
begin
  sorry
end

end find_hyperbola_equation_l481_481739


namespace class_average_correct_l481_481859

-- Define the conditions
def boys_average := 90
def girls_average := 96
def ratio_boys_to_girls := 0.5

-- Define the overall class average calculation as a theorem
theorem class_average_correct 
  (B G : ℕ)  -- number of boys and girls
  (h_ratio : (B : ℚ) / G = ratio_boys_to_girls)  -- ratio condition
  (h_boys_avg : boys_average = 90)
  (h_girls_avg : girls_average = 96) :
  (boys_average * B + girls_average * G) / (B + G) = 94 :=
by
  sorry

end class_average_correct_l481_481859


namespace f_neg_1_eq_zero_l481_481007

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x^2 - x else -(x^2 - x)

theorem f_neg_1_eq_zero : f (-1) = 0 := by
  -- f is defined to be an odd function
  have h1 : ∀ x, f (-x) = -f (x) := by
    intro x
    unfold f
    split_ifs
    . sorry
    . sorry
  -- f(x) = x^2 - x for x > 0
  have h2 : ∀ x, x > 0 → f (x) = x^2 - x := by
    intros x hx
    unfold f
    split_ifs
    . exact rfl
    . exfalso
      exact lt_irrefl _ h
  -- Using h1 and h2 to prove f(-1) = 0
  have h3 := h2 1 (by linarith)
  calc
  f (-1) = -f 1  := h1 1
        ... = -0 := by rw [h3]
        ... = 0  := neg_zero

end f_neg_1_eq_zero_l481_481007


namespace find_a_n_l481_481705

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : ∀ n, a n > 0)
  (h₂ : ∀ n, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
by
  sorry

end find_a_n_l481_481705


namespace correct_decimal_product_l481_481279

theorem correct_decimal_product : (0.125 * 3.2 = 4.0) :=
sorry

end correct_decimal_product_l481_481279


namespace fried_frog_probability_l481_481335

/--
Suppose Frieda begins her sequence of hops from the center square (2, 2) on a 4x4 grid.
We define the grid with positions and wrap-around movements. Frieda moves one square on each hop and
chooses a random direction: up, down, left, or right. If the direction of a hop takes Frieda off the grid,
she wraps around to the opposite edge. Frieda stops hopping if she lands on a corner square.
Additionally, Frieda can make at most five hops.

Prove that the probability of Frieda reaching any corner square within five hops is \(15/16\).
-/
theorem fried_frog_probability : 
  let g := grid 4 4,
  start_pos : (2, 2),
  hops : {h : ℕ // h ≤ 5} → (g.pos → g.pos), -- Hops defined as a function from one grid position to another
  let prob_corner : ℕ → ℚ := -- Define probability function
    λ n, /- Calculate probability that Frieda reaches a corner within n hops -/
    sorry,
  prob_corner 5 = 15 / 16 := 
sorry

end fried_frog_probability_l481_481335


namespace probability_of_less_than_5_is_one_half_l481_481227

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l481_481227


namespace probability_two_queens_or_at_least_one_ace_l481_481757

theorem probability_two_queens_or_at_least_one_ace :
  let total_cards := 52
  let num_aces := 4
  let num_queens := 4
  let prob_two_queens := (4 / total_cards) * (3 / (total_cards - 1))
  let prob_one_ace_FIRST := (4 / total_cards) * ((total_cards - num_aces) / (total_cards - 1))
  let prob_one_ace_SECOND := ((total_cards - num_aces) / total_cards) * (4 / (total_cards - 1))
  let prob_one_ace := prob_one_ace_FIRST + prob_one_ace_SECOND
  let prob_exactly_one_ace := prob_one_ace
  let prob_two_aces := (4 / total_cards) * (3 / (total_cards - 1))
  let prob_at_least_one_ace := prob_exactly_one_ace + prob_two_aces
  let prob_two_queens_or_at_least_one_ace := prob_two_queens + prob_at_least_one_ace
  (prob_two_queens_or_at_least_one_ace = 2 / 13) :=
by
  let total_cards := 52
  let num_aces := 4
  let num_queens := 4
  let prob_two_queens := (4 / total_cards) * (3 / (total_cards - 1))
  let prob_one_ace_FIRST := (4 / total_cards) * ((total_cards - num_aces) / (total_cards - 1))
  let prob_one_ace_SECOND := ((total_cards - num_aces) / total_cards) * (4 / (total_cards - 1))
  let prob_one_ace := prob_one_ace_FIRST + prob_one_ace_SECOND
  let prob_exactly_one_ace := prob_one_ace
  let prob_two_aces := (4 / total_cards) * (3 / (total_cards - 1))
  let prob_at_least_one_ace := prob_exactly_one_ace + prob_two_aces
  let prob_two_queens_or_at_least_one_ace := prob_two_queens + prob_at_least_one_ace
  have h : prob_two_queens_or_at_least_one_ace = (1 / 221) + (32 / 221 + 1 / 221) :=
    sorry -- This should be proved by simplification
  have h2 : (1 / 221) + (32 / 221 + 1 / 221) = 34 / 221 :=
    sorry -- This should be proved by simplification
  have h3 : 34 / 221 = 2 / 13 :=
    sorry -- This should be proved by simplification
  exact Eq.trans h (Eq.trans h2 h3)

end probability_two_queens_or_at_least_one_ace_l481_481757


namespace meetings_percentage_l481_481501

theorem meetings_percentage
  (workday_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (second_meeting_factor : ℕ)
  (third_meeting_factor : ℕ)
  (total_minutes : ℕ)
  (total_meeting_minutes : ℕ) :
  workday_hours = 9 →
  first_meeting_minutes = 30 →
  second_meeting_factor = 2 →
  third_meeting_factor = 3 →
  total_minutes = workday_hours * 60 →
  total_meeting_minutes = first_meeting_minutes + second_meeting_factor * first_meeting_minutes + third_meeting_factor * first_meeting_minutes →
  (total_meeting_minutes : ℚ) / (total_minutes : ℚ) * 100 = 33.33 :=
by
  sorry

end meetings_percentage_l481_481501


namespace inequality_solution_l481_481884

theorem inequality_solution (x : ℝ) (h : x ≠ 2) :
  (x^3 - 2*x^2 - 13*x + 10) / (x - 2) > 0 ↔ x ∈ Set.Ioo (-∞) (-5) ∪ Set.Ioo 1 2 ∪ Set.Ioo 2 ∞ :=
by
  sorry

end inequality_solution_l481_481884


namespace inequality_equivalence_l481_481687

theorem inequality_equivalence (x y : ℝ) (h : y = 2 * x + 4) :
  2 ≤ abs (x - 3) ∧ abs (x - 3) ≤ 8 ↔ y ∈ set.Icc (-6) 6 ∪ set.Icc 14 26 :=
by sorry

end inequality_equivalence_l481_481687


namespace distance_from_origin_to_line_PQ_l481_481416

open Real

theorem distance_from_origin_to_line_PQ :
  ∀ (P Q : ℝ × ℝ), 
  (sqrt (P.1^2 + 2 * sqrt 7 * P.1 + P.2^2 + 7) 
   + sqrt (P.1^2 - 2 * sqrt 7 * P.1 + P.2^2 + 7) = 8) ∧
  (sqrt (Q.1^2 + 2 * sqrt 7 * Q.1 + Q.2^2 + 7) 
   + sqrt (Q.1^2 - 2 * sqrt 7 * Q.1 + Q.2^2 + 7) = 8) ∧
  (P ≠ Q) →
  let diameter := dist P Q in
  let midpoint := (P.1 + Q.1) / 2, (P.2 + Q.2) / 2 in
  (sqrt (midpoint.1^2 + midpoint.2^2) = diameter / 2) →
  (let d := (abs (P.1 - Q.1) * P.2 + abs (P.2 - Q.2) * Q.2) / sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) in
  d = 12 / 5) :=
begin
  sorry
end

end distance_from_origin_to_line_PQ_l481_481416


namespace sum_c_n_less_than_3_4_l481_481728

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^n
noncomputable def b_n (n : ℕ) : ℝ := 1 / n
noncomputable def c_n (n : ℕ) : ℝ := n * (1/3)^n

theorem sum_c_n_less_than_3_4 (n : ℕ) : 
  (∑ i in Finset.range n.succ, c_n i) < 3 / 4 :=
sorry

end sum_c_n_less_than_3_4_l481_481728


namespace calculate_expression_l481_481657

theorem calculate_expression : 1 + (Real.sqrt 2 - Real.sqrt 3) + abs (Real.sqrt 2 - Real.sqrt 3) = 1 :=
by
  sorry

end calculate_expression_l481_481657


namespace minimum_value_expression_l481_481742

theorem minimum_value_expression 
  (a b c : ℝ) 
  (h1 : 3 * a + 2 * b + c = 5) 
  (h2 : 2 * a + b - 3 * c = 1) 
  (h3 : 0 ≤ a) 
  (h4 : 0 ≤ b) 
  (h5 : 0 ≤ c) : 
  ∃(c : ℝ), (c ≥ 3/7 ∧ c ≤ 7/11) ∧ (3 * a + b - 7 * c = -5/7) :=
sorry 

end minimum_value_expression_l481_481742


namespace nested_radical_floor_l481_481122

theorem nested_radical_floor :
  let y := sqrt(10 + sqrt(10 + sqrt(10 + sqrt(10 + ⋯)))),
      B := ⌊10 + y⌋ in
  (y * y = 10 + y) ∧ (B = 13) :=
sorry

end nested_radical_floor_l481_481122


namespace g_iterated_six_times_is_2_l481_481115

def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem g_iterated_six_times_is_2 : g (g (g (g (g (g 2))))) = 2 := 
by 
  sorry

end g_iterated_six_times_is_2_l481_481115


namespace find_value_l481_481688

variable {x y : ℝ}

theorem find_value (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y + x * y = 0) : y / x + x / y = -2 := 
sorry

end find_value_l481_481688


namespace count_more_blue_l481_481944

-- Definitions derived from the provided conditions
variables (total_people more_green both neither : ℕ)
variable (more_blue : ℕ)

-- Condition 1: There are 150 people in total
axiom total_people_def : total_people = 150

-- Condition 2: 90 people believe that teal is "more green"
axiom more_green_def : more_green = 90

-- Condition 3: 35 people believe it is both "more green" and "more blue"
axiom both_def : both = 35

-- Condition 4: 25 people think that teal is neither "more green" nor "more blue"
axiom neither_def : neither = 25


-- Theorem statement
theorem count_more_blue (total_people more_green both neither more_blue : ℕ) 
  (total_people_def : total_people = 150)
  (more_green_def : more_green = 90)
  (both_def : both = 35)
  (neither_def : neither = 25) :
  more_blue = 70 :=
by
  sorry

end count_more_blue_l481_481944


namespace number_of_women_l481_481444

theorem number_of_women (x : ℕ) 
  (h1 : 4 * x + 2 = 14) : 2 * (5 * x - 3) = 24 :=
by 
  ext
  sorry

end number_of_women_l481_481444


namespace complex_magnitude_sqrt_five_l481_481065

open Complex

theorem complex_magnitude_sqrt_five
  (z : ℂ)
  (h : 2 * z + conj z = 3 - 2 * Complex.I) :
  ∥z∥ = Real.sqrt 5 :=
sorry

end complex_magnitude_sqrt_five_l481_481065


namespace third_number_in_row_l481_481647

theorem third_number_in_row (n : ℕ) (h : n ≥ 3) : 
    let odd_number_in_row (r k: ℕ) := r * (r - 1) + 2 * k - 1 in
    odd_number_in_row n 3 = n^2 - n + 5 := 
by 
  sorry

end third_number_in_row_l481_481647


namespace current_age_ratio_arun_deepak_l481_481287

def arun_age_after_10_years := 30
def deepak_current_age := 50

theorem current_age_ratio_arun_deepak :
  let arun_current_age := arun_age_after_10_years - 10 in
  arun_current_age / deepak_current_age = 2 / 5 :=
by
  sorry

end current_age_ratio_arun_deepak_l481_481287


namespace range_of_a_l481_481907

open Real

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ , x^2 + a * x + 1 < 0) ↔ (-2 : ℝ) ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l481_481907


namespace rachel_remaining_money_l481_481872

def initial_earnings : ℝ := 200
def lunch_expense : ℝ := (1/4) * initial_earnings
def dvd_expense : ℝ := (1/2) * initial_earnings
def total_expenses : ℝ := lunch_expense + dvd_expense
def remaining_amount : ℝ := initial_earnings - total_expenses

theorem rachel_remaining_money :
  remaining_amount = 50 := 
by
  sorry

end rachel_remaining_money_l481_481872


namespace q_at_2_5_l481_481291

def q (x : ℝ) : ℝ :=
  Float.signum (2 * x - 3) * |2 * x - 3|^(1/3 : ℝ) +
  3 * Float.signum (2 * x - 3) * |2 * x - 3|^(1/2 : ℝ) +
  |2 * x - 3|^(1/7 : ℝ)

theorem q_at_2_5 : q 2.5 = 7 :=
by
  -- We state the theorem without the proof here.
  sorry

end q_at_2_5_l481_481291


namespace length_of_base_BC_l481_481860

-- Definitions reflecting the conditions
variables {a m BC : ℝ}
variables (AD PQ : ℝ)
variables (parallel_PD_AD : line PQ = parallel line AD)
variables (three_parts : ∃ n : ℝ, m = 3 * n)
variables (intersect_inside : ∃ E F : Point, E ∈ line AC ∧ F ∈ line BD ∧ (E ∈ quadrilateral BPCQ) ∧ (F ∈ quadrilateral BPCQ))

-- Statement of the problem
theorem length_of_base_BC (hAD : AD = a) (hPQ : PQ = m)
  (h_parallel : parallel_PD_AD) (h_three_parts : three_parts)
  (h_intersect_inside : intersect_inside) :
  BC = (a * m) / (3 * a - 2 * m) :=
sorry

end length_of_base_BC_l481_481860


namespace Bethany_total_riding_hours_l481_481998

-- Define daily riding hours
def Monday_hours : Nat := 1
def Wednesday_hours : Nat := 1
def Friday_hours : Nat := 1
def Tuesday_hours : Nat := 1 / 2
def Thursday_hours : Nat := 1 / 2
def Saturday_hours : Nat := 2

-- Define total weekly hours
def weekly_hours : Nat :=
  Monday_hours + Wednesday_hours + Friday_hours + (Tuesday_hours + Thursday_hours) + Saturday_hours

-- Definition to account for the 2-week period
def total_hours (weeks : Nat) : Nat := weeks * weekly_hours

-- Prove that Bethany rode 12 hours over 2 weeks
theorem Bethany_total_riding_hours : total_hours 2 = 12 := by
  sorry

end Bethany_total_riding_hours_l481_481998


namespace bethany_total_hours_l481_481993

-- Define the hours Bethany rode on each set of days
def hours_mon_wed_fri : ℕ := 3  -- 1 hour each on Monday, Wednesday, and Friday
def hours_tue_thu : ℕ := 1  -- 30 min each on Tuesday and Thursday
def hours_sat : ℕ := 2  -- 2 hours on Saturday

-- Define the total hours per week
def total_hours_per_week : ℕ := hours_mon_wed_fri + hours_tue_thu + hours_sat

-- Define the total hours in 2 weeks
def total_hours_in_2_weeks : ℕ := total_hours_per_week * 2

-- Prove that the total hours in 2 weeks is 12
theorem bethany_total_hours : total_hours_in_2_weeks = 12 :=
by
  -- Replace the definitions with their values and check the equality
  rw [total_hours_in_2_weeks, total_hours_per_week, hours_mon_wed_fri, hours_tue_thu, hours_sat]
  simp
  norm_num
  sorry

end bethany_total_hours_l481_481993


namespace alice_wins_iff_n_mod_4_eq_3_l481_481827

theorem alice_wins_iff_n_mod_4_eq_3 (n : ℕ) (h : n ≥ 2) :
  (∃ i1 i2 : ℕ, i1 < n ∧ i2 < n ∧ i1 ≠ i2) →
  (∀ k ≥ 2, ∃ I1 I2 : fin k, I1 ≠ I2 ∧ ((I1 = 1 ∨ I2 = 1) ∨ (I1 = 2 ∨ I2 = 2))) →
  ((n % 4 = 3) ↔ alice_wins h) :=
sorry

end alice_wins_iff_n_mod_4_eq_3_l481_481827


namespace distinct_real_solutions_count_l481_481749

theorem distinct_real_solutions_count : 
  ∀ (x : ℝ), (x^2 - 7)^2 = 49 → x = 0 ∨ x = √14 ∨ x = -√14 ∧ 
  (∀ (a b c : ℝ), a ≠ b → a ≠ c → b ≠ c → (a = x ∨ a ≠ x) → (b = x ∨ b ≠ x) → (c = x ∨ c ≠ x) → 3 =
  (if x = 0 ∨ x = √14 ∨ x = -√14 then 3 else 0)) :=
by
  sorry

end distinct_real_solutions_count_l481_481749


namespace pp_cotton_filter_min_layers_l481_481277

theorem pp_cotton_filter_min_layers 
  (x0 : ℝ) (xf : ℝ) (log2 : ℝ) (log3 : ℝ)
  (hx0 : x0 = 80) 
  (hxf : xf = 2)
  (hlog2 : log2 ≈ 0.30) 
  (hlog3 : log3 ≈ 0.48) :
  ∃ n : ℕ, 80 * (2 / 3)^n ≤ 2 ∧ n = 9 :=
by
  sorry

end pp_cotton_filter_min_layers_l481_481277


namespace inequality_of_positive_numbers_l481_481840

open BigOperators

theorem inequality_of_positive_numbers (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i) :
    (∑ i, x i) * (∑ i, (x i)⁻¹) ≥ n^2 :=
begin
    sorry
end

end inequality_of_positive_numbers_l481_481840


namespace number_of_women_is_24_l481_481449

-- Define the variables and conditions
variables (x : ℕ) (men_initial : ℕ) (women_initial : ℕ) (men_current : ℕ) (women_current : ℕ)

-- representing the initial ratio and the changes
def initial_conditions : Prop :=
  men_initial = 4 * x ∧ women_initial = 5 * x ∧
  men_current = men_initial + 2 ∧ women_current = 2 * (women_initial - 3)

-- representing the current number of men
def current_men_condition : Prop := men_current = 14

-- The proof we need to generate
theorem number_of_women_is_24 (x : ℕ) (men_initial women_initial men_current women_current : ℕ)
  (h1 : initial_conditions x men_initial women_initial men_current women_current)
  (h2 : current_men_condition men_current) : women_current = 24 :=
by
  -- proof steps here
  sorry

end number_of_women_is_24_l481_481449


namespace fibonacci_product_l481_481829

def fibonacci : ℕ → ℕ
| 1     := 1
| 2     := 1
| (n+1) := fibonacci n + fibonacci (n-1)

theorem fibonacci_product :
  ∏ k in (finset.range 99).map (nat.succ ∘ nat.succ ∘ nat.succ), 
      ((fibonacci k) / (fibonacci (k - 1)) - (fibonacci k) / (fibonacci (k + 1))) =
  (fibonacci 101) / (fibonacci 102) :=
by
  sorry

end fibonacci_product_l481_481829


namespace greatest_visible_unit_cubes_from_corner_l481_481603

def cube_size : ℕ := 12
def unit_cubes : ℕ := cube_size^3
def gap_size : ℕ := 1

theorem greatest_visible_unit_cubes_from_corner : 
  ∀ (cube_size unit_cubes gap_size : ℕ), 
  cube_size = 12 → 
  unit_cubes = 12^3 → 
  gap_size = 1 → 
  412 ≤ 3 * (cube_size * cube_size) - 3 * (cube_size - 1) + 12 + 1 ∧ 
  3 * (cube_size * cube_size) - 3 * (cube_size - 1) + 12 + 1 ≤ 412 :=
by
  intros cube_size unit_cubes gap_size h1 h2 h3
  have h4 : 3 * (cube_size * cube_size) - 3 * (cube_size - 1) + 12 + 1 = 412
  {
    sorry
  }
  exact ⟨le_of_eq h4, le_of_eq (eq.symm h4)⟩

end greatest_visible_unit_cubes_from_corner_l481_481603


namespace arithmetic_geom_seq_l481_481346

-- Define sequences and the given condition
def a (n : ℕ) : ℝ := a₀ + n * d

variable {d a₀ : ℝ}
variable (hne : d ≠ 0)
variable (hgeo : (a 1 + 2 * d) ^ 2 = (a 1) * (a 1 + 8 * d))

-- The theorem to prove
theorem arithmetic_geom_seq :
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5 / 8 :=
by {
  sorry -- proof omitted
}

end arithmetic_geom_seq_l481_481346


namespace find_T_minimize_S_l481_481600

variables (a b x y a_0 b_0 : ℝ) (a_pos b_pos a_0_pos b_0_pos : a > 0 ∧ b > 0 ∧ a_0 > 0 ∧ b_0 > 0)

-- Definition of the hyperbola and its tangent line
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def tangent_line (x_0 y_0 a b : ℝ) (x y : ℝ) : Prop := (x_0 * x / a^2) - (y_0 * y / b^2) = 1

-- Definition of the midpoint condition
def midpoint_condition (x_A x_B y_A y_B : ℝ) : Prop := 
  (x_A + x_B) / 2 = 2 ∧ (y_A + y_B) / 2 = 1

-- Definition of the ellipse and point conditions
def ellipse (a_0 b_0 x y : ℝ) : Prop := (x^2 / a_0^2) + (y^2 / b_0^2) = 1
def points (C D : ℝ × ℝ) : Prop := C = (0, b_0) ∧ D = (a_0, 0)

-- Definition of the area of the triangle
def area_triangle (T E F : ℝ × ℝ) : ℝ :=
  let (x_T, y_T) := T in
  let (x_E, y_E) := E in
  let (x_F, y_F) := F in
  abs ((x_E - x_T) * (y_F - y_T) - (x_F - x_T) * (y_E - y_T)) / 2

-- Statements to prove
theorem find_T (x T_x T_y : ℝ) (h_hyp : hyperbola a b x y)
  (h_tangent : ∀ x y, tangent_line T_x T_y a b x y)
  (h_midpoint : ∃ x_A x_B y_A y_B, midpoint_condition x_A x_B y_A y_B)
  : T_x = 2 ∧ T_y = 1 :=
sorry

theorem minimize_S (CD_mid : ℝ) (min_S : ℝ → ℝ)
  (h_ellipse : ellipse a_0 b_0 x y)
  (h_points : ∃ C D, points C D)
  (h_area : ∀ E F, area_triangle (2, 1) E F = S)
  : min_S CD_mid = 1 :=
sorry

end find_T_minimize_S_l481_481600


namespace range_of_a_l481_481768

noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x ≤ 1 / 2 then (1 / 2) ^ (x - 1 / 2) else log a x

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (sqrt 2 / 2 ≤ a ∧ a < 1) :=
sorry

end range_of_a_l481_481768


namespace hexagon_area_l481_481270

noncomputable def side_length_of_square (area : ℝ) : ℝ :=
  real.sqrt area

noncomputable def side_length_of_hexagon (s : ℝ) : ℝ :=
  (2 * s) / 3

noncomputable def area_of_square (s : ℝ) : ℝ :=
  s^2

noncomputable def area_of_hexagon (t : ℝ) : ℝ :=
  (3 * t^2 * real.sqrt 3) / 2

theorem hexagon_area (h1 : area_of_square s = 16) 
  (h2 : 4 * s = 6 * t) : 
  area_of_hexagon t = (32 * real.sqrt 3) / 3 :=
by
  sorry

end hexagon_area_l481_481270


namespace sum_distances_greater_than_last_distance_l481_481140

-- Definitions representing the problem conditions
structure IsoscelesTrapezoid (A B C D M : Type) [MetricSpace M] where
  ab_parallel_cd : ∀ (x : M), dist A x = dist B x ∧ dist C x = dist D x
  ab_equal_cd : dist A B = dist C D
  ad_equal_bc : dist A D = dist B C

-- The theorem we want to prove
theorem sum_distances_greater_than_last_distance 
  (A B C D M : Type) [MetricSpace M] (trapezoid : IsoscelesTrapezoid A B C D M)
  (P : M) :
  dist P A + dist P B + dist P C ≥ dist P D :=
by sorry

end sum_distances_greater_than_last_distance_l481_481140


namespace min_value_expression_l481_481123

theorem min_value_expression (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : (a - b) * (b - c) * (c - a) = -16) : 
  ∃ x : ℝ, x = (1 / (a - b)) + (1 / (b - c)) - (1 / (c - a)) ∧ x = 5 / 4 :=
by
  sorry

end min_value_expression_l481_481123


namespace b_is_minus_two_l481_481510

noncomputable def verify_b_value (b c d : ℝ) : Prop :=
  ∃ (m r : ℝ), 
    (∀ x : ℝ, (f : ℝ → ℝ) = (λ x, x^3 + b*x^2 + c*x + d)) ∧
    (f(-1) = 0) ∧ (f(1) = 0) ∧ (f(0) = 2) ∧ (b = -2)

theorem b_is_minus_two : ∀ (b c d : ℝ), verify_b_value b c d → b = -2 :=
  by
    intros b c d h
    sorry

end b_is_minus_two_l481_481510


namespace avg_sqrt_solution_l481_481398

theorem avg_sqrt_solution :
  (∃ x : ℝ, sqrt (3 * x^2 + 4) = sqrt 28) →
  avg (set_of (λ x : ℝ, sqrt (3 * x^2 + 4) = sqrt 28)) = 0 :=
by
  sorry

end avg_sqrt_solution_l481_481398


namespace sum_of_smallest_and_largest_prime_is_49_l481_481192

-- Define the set of prime numbers between 1 and 50.
def primesBetween1And50 : Set ℕ := {n | n ∈ Finset.range 51 ∧ Nat.Prime n}

-- Identify the smallest prime number in the set.
def smallestPrime : ℕ := Finset.min' (Finset.filter Nat.Prime (Finset.range 51)) begin
  -- Proof of non-empty set is omitted for brevity
  sorry
end

-- Identify the largest prime number in the set.
def largestPrime : ℕ := Finset.max' (Finset.filter Nat.Prime (Finset.range 51)) begin
  -- Proof of non-empty set is omitted for brevity
  sorry
end

-- Define the theorem to prove that the sum of smallest and largest prime is 49.
theorem sum_of_smallest_and_largest_prime_is_49 : smallestPrime + largestPrime = 49 := by
  sorry

end sum_of_smallest_and_largest_prime_is_49_l481_481192


namespace find_a_n_l481_481704

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : ∀ n, a n > 0)
  (h₂ : ∀ n, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
by
  sorry

end find_a_n_l481_481704


namespace probability_of_rolling_less_than_five_l481_481206

/-- Rolling an 8-sided die -/
def outcomes := { 1, 2, 3, 4, 5, 6, 7, 8 }

/-- Successful outcomes (numbers less than 5) -/
def successful_outcomes := { 1, 2, 3, 4 }

/-- Number of total outcomes -/
def total_outcomes := 8

/-- Number of successful outcomes -/
def num_successful_outcomes := 4

/-- Probability of rolling a number less than 5 on an 8-sided die -/
def probability : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_rolling_less_than_five : probability = 1 / 2 := by
  sorry

end probability_of_rolling_less_than_five_l481_481206


namespace rational_uniq_solution_l481_481826

-- Define the conditions
def isSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2
def evenDecimalDigitsPrimeFactors (n : ℕ) : Prop := 
  ∀ p : ℕ, nat.prime p ∧ nat.powMod 10 (nat.numDigits 10 p) ≡ 0 [MOD p] ∧ p ∣ n → 
  even (nat.numDigits 10 p)

-- Define the polynomial P
def P (x : ℚ) (n : ℕ) : ℚ := x^n - 1987 * x

-- The main theorem statement
theorem rational_uniq_solution (n : ℕ) (hn1 : isSquare n) (hn2 : evenDecimalDigitsPrimeFactors n) 
  (x y : ℚ) (h : P x n = P y n) : x = y := 
sorry

end rational_uniq_solution_l481_481826


namespace segments_of_opposite_square_centers_are_equal_and_perpendicular_l481_481134

theorem segments_of_opposite_square_centers_are_equal_and_perpendicular
  (quadrilateral : Type) [finite quadrilateral] 
  (squares : quadrilateral → Type) 
  (side_has_square : ∀ s : quadrilateral, squares s) 
  (center_of_square : ∀ s : quadrilateral, point) 
  (O₁ O₂ O₃ O₄ : point) 
  (h₁ : center_of_square side_has_square = O₁ ∨ center_of_square side_has_square = O₂ ∨ center_of_square side_has_square = O₃ ∨ center_of_square side_has_square = O₄) :
  (distance O₁ O₃ = distance O₂ O₄) ∧ (is_perpendicular O₁ O₃ O₂ O₄) :=
sorry

end segments_of_opposite_square_centers_are_equal_and_perpendicular_l481_481134


namespace plane_through_A_perpendicular_to_BC_l481_481592

-- Define the points A, B, and C.
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point := { x := 2, y := 1, z := 7 }
def B : Point := { x := 9, y := 0, z := 2 }
def C : Point := { x := 9, y := 2, z := 3 }

-- Define the vector BC.
def vectorBC (B C : Point) : Point :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

def N := vectorBC B C

-- Example target equation of plane in general form.
def plane_equation (p : Point) (N : Point) : (ℝ × ℝ × ℝ) :=
  (N.x * (p.x - A.x) + N.y * (p.y - A.y) + N.z * (p.z - A.z), 2 * p.y + p.z - 9 = 0)

-- Theorem to state that the plane equation is 2y + z - 9 = 0
theorem plane_through_A_perpendicular_to_BC :
  let eq : ℝ × ℝ × ℝ := plane_equation A N in 
    eq = (0, 2 * A.y + A.z - 9 = 0) :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l481_481592


namespace area_of_quadrilateral_l481_481551

noncomputable def quadrilateral_area
  (AB CD r : ℝ) (k : ℝ) 
  (h_perpendicular : AB * CD = 0)
  (h_equal_diameters : AB = 2 * r ∧ CD = 2 * r)
  (h_ratio : BC / AD = k) : ℝ := 
  (3 * r^2 * abs (1 - k^2)) / (1 + k^2)

theorem area_of_quadrilateral
  (AB CD r : ℝ) (k : ℝ)
  (h_perpendicular : AB * CD = 0)
  (h_equal_diameters : AB = 2 * r ∧ CD = 2 * r)
  (h_ratio : BC / AD = k) :
  quadrilateral_area AB CD r k h_perpendicular h_equal_diameters h_ratio = (3 * r^2 * abs (1 - k^2)) / (1 + k^2) :=
sorry

end area_of_quadrilateral_l481_481551


namespace greatest_possible_n_is_4_l481_481308

noncomputable def max_n : ℕ :=
  if h : ∃ n ≥ 3, ∃ (a : Fin n → ℕ), 
       (∀ i j, i ≠ j → gcd (a i) (a j) > 1) ∧
       (∀ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k → gcd (gcd (a i) (a j)) (a k) = 1) ∧
       (∀ i, a i < 5000) 
  then Classical.choose h else
  0 -- Fallback case if no such n exists
  
theorem greatest_possible_n_is_4 : max_n = 4 :=
by
  sorry

end greatest_possible_n_is_4_l481_481308


namespace monotonically_decreasing_interval_l481_481014

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}
variable {x : ℝ}

noncomputable def is_monotonically_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y ∈ s, x < y → f y ≤ f x

theorem monotonically_decreasing_interval : 
  (∀ x, f' x = (x - 3) * (x + 1) ^ 2) → is_monotonically_decreasing f {x | x ≤ 3} :=
by 
  intro h
  sorry

end monotonically_decreasing_interval_l481_481014


namespace total_fires_l481_481989

-- Conditions as definitions
def Doug_fires : Nat := 20
def Kai_fires : Nat := 3 * Doug_fires
def Eli_fires : Nat := Kai_fires / 2

-- Theorem to prove the total number of fires
theorem total_fires : Doug_fires + Kai_fires + Eli_fires = 110 := by
  sorry

end total_fires_l481_481989


namespace find_a_l481_481493

theorem find_a (a : ℝ) : (∀ x : ℝ, (deriv (λ x : ℝ, a * x^2)) = 2 * a * x) → 
  let tangent_slope := (deriv (λ x : ℝ, a * x^2)) 1 in
  let line_slope := 2 in
  tangent_slope = line_slope → a = 1 :=
by
  intros deriv_eq tangent_slope line_slope
  have h1 : tangent_slope = 2 * a := by
    exact deriv_eq 1
  have h2 : tangent_slope = 2 := line_slope
  rw [h1, h2]
  sorry

end find_a_l481_481493


namespace initial_number_of_red_balls_l481_481181

theorem initial_number_of_red_balls 
  (num_white_balls num_red_balls : ℕ)
  (h1 : num_red_balls = 4 * num_white_balls + 3)
  (num_actions : ℕ)
  (h2 : 4 + 5 * num_actions = num_white_balls)
  (h3 : 34 + 17 * num_actions = num_red_balls) : 
  num_red_balls = 119 := 
by
  sorry

end initial_number_of_red_balls_l481_481181


namespace sqrt_43_between_6_and_7_l481_481282

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 := sorry

end sqrt_43_between_6_and_7_l481_481282


namespace lcm_first_ten_sum_first_ten_l481_481574

theorem lcm_first_ten :
  ∃ n, (∀ m ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10} : Set ℕ), m ∣ n) ∧ n = 2520 :=
by
  sorry

theorem sum_first_ten :
  (Finset.sum (Finset.range 11)) = 55 :=
by
  sorry

end lcm_first_ten_sum_first_ten_l481_481574


namespace total_students_proof_l481_481904

variable (S : ℕ) -- Number of students who wish to go scavenger hunting
variable (Skiing : ℕ) -- Number of students who wish to go skiing
variable (Camping : ℕ) -- Number of students who wish to go camping
variable (Total_students : ℕ) -- Total number of students

-- Given conditions
axiom h1 : S = 4000
axiom h2 : Skiing = 2 * S
axiom h3 : Camping = Skiing + (15 * Skiing) / 100

-- Proof statement
theorem total_students_proof : Total_students = S + Skiing + Camping → Total_students = 21200 :=
by
  intros h
  rw [h1, h2] at h
  have : Skiing = 8000, from (by rw [h1, show 2 * 4000 = 8000 by norm_num])
  rw [this] at h
  have : Camping = 9200, from (by rw [this, show 2 * 4000 = 8000 by norm_num, show 15 * 8000 / 100 = 1200 by norm_num])
  rw [this] at h
  rw [h]
  sorry

end total_students_proof_l481_481904


namespace number_of_valid_subsets_l481_481482

open Finset

def is_valid_subset (S T : Finset ℕ) : Prop :=
  ∀ x, x ∈ T → 2 * x ∈ S → 2 * x ∈ T

theorem number_of_valid_subsets :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.to_finset in
  ( {T : Finset ℕ | T ⊆ S ∧ is_valid_subset S T}.to_finset.card = 180 ) :=
sorry

end number_of_valid_subsets_l481_481482


namespace area_equality_l481_481084

open Real

namespace Geometry

variable {P Q R A B C L N K M : Type*} [RealInnerProductSpace R]

noncomputable def area_triangle (A B C : R) : ℝ :=
  1 / 2 * abs ((B - A) ⬝ (C - A))

noncomputable def area_quadrilateral (A K N M : R) : ℝ :=
  area_triangle A K N + area_triangle A N M

-- Given conditions
variables
  (ABC_acute : ∀ {A B C K M N L : R}, acute_angle A B C)
  (bisector_A : B.angle_bisector A C = L)
  (circumcircle_A : ∃ (K A B C), circle A B C)
  (LK_perp_AB : L ⊥ AB)
  (LM_perp_AC : L ⊥ AC)

-- Prove the area equivalence
theorem area_equality
  (A B C L N K M : R)
  (h_acute : ∀ {A B C K M N L : R}, acute_angle A B C)
  (h_bisector : bisector_A A B C L)
  (h_circumcircle : circumcircle_A A B C L N K)
  (h_LK_perp : LK_perp_AB L K)
  (h_LM_perp : LM_perp_AC L M) :
  (area_triangle A B C) = (area_quadrilateral A K N M) :=
begin
  sorry,
end

end Geometry

end area_equality_l481_481084


namespace total_hours_over_two_weeks_l481_481996

-- Define the conditions of Bethany's riding schedule
def hours_per_week : ℕ :=
  1 * 3 + -- Monday, Wednesday, and Friday
  (30 / 60) * 2 + -- Tuesday and Thursday, converting minutes to hours
  2 -- Saturday

-- The theorem to prove the total hours over 2 weeks
theorem total_hours_over_two_weeks : hours_per_week * 2 = 12 := 
by
  -- Proof to be completed here
  sorry

end total_hours_over_two_weeks_l481_481996


namespace packages_needed_to_label_apartments_l481_481901

theorem packages_needed_to_label_apartments :
  let digits_needed (range_start range_end : ℕ) : ℕ :=
    let freq (d : ℕ) : ℕ := List.sum $ List.map (fun n => List.count d (n.digits 10)) (List.range' range_start (range_end - range_start + 1))
    in let max_freq := List.maximum [freq 0, freq 1, freq 2, freq 3, freq 4, freq 5, freq 6, freq 7, freq 8, freq 9]
    in max_freq
  digits_needed 107 132 + digits_needed 207 232 = 46 :=
by 
  sorry

end packages_needed_to_label_apartments_l481_481901


namespace construct_segment_l481_481046

variables {α : Type*} [linear_ordered_field α]

structure Point (α : Type*) :=
(x : α) (y : α)

def parallel (l1 l2 : set (Point α)) : Prop :=
-- Definition for two lines being parallel
sorry

def reflect (p : Point α) (l : set (Point α)) : Point α :=
-- Definition for reflection of point p across line l
sorry

def view_circle (A B : Point α) (angle : α) : set (Point α) :=
-- Definition for view circle passing through A and B with given angle
sorry

theorem construct_segment
  (A B : Point α)
  (e : set (Point α))
  (α : α)
  (h_parallel : parallel (λ (p : Point α), p.x * (B.y - A.y) = p.y * (B.x - A.x)) e) :
  ∃ (C D : Point α),
    C ∈ e ∧ D ∈ e ∧ ∠ CAD = α :=
begin
  sorry
end

end construct_segment_l481_481046


namespace tan_x_tan_y_eq_four_l481_481124

variables {x y : ℝ}

def condition1 := (sin x / cos y) - (sin y / cos x) = 2
def condition2 := (cos x / sin y) - (cos y / sin x) = 3

theorem tan_x_tan_y_eq_four 
  (h1 : condition1) 
  (h2 : condition2) : 
  (tan x / tan y) + (tan y / tan x) = 4 :=
sorry

end tan_x_tan_y_eq_four_l481_481124


namespace polygon_sides_sum_l481_481403

theorem polygon_sides_sum (n : ℕ) (x : ℝ) (hx : 0 < x ∧ x < 180) 
  (h_sum : 180 * (n - 2) - x = 2190) : n = 15 :=
sorry

end polygon_sides_sum_l481_481403


namespace number_of_even_divisors_l481_481127

noncomputable def count_even_divisors (n : ℕ) : ℕ :=
  (List.filter (λ d, d % 2 = 0) (List.range (n + 1))).length

theorem number_of_even_divisors (p : ℕ) (hp : p.prime) (h2p : 2 < p) :
  count_even_divisors (14 * p) = 4 :=
  sorry

end number_of_even_divisors_l481_481127


namespace tan_theta_eq_2_l481_481009

theorem tan_theta_eq_2 (θ : ℝ) (h : 2 * tan θ - tan (θ + π / 4) = 7) : tan θ = 2 :=
by
  sorry

end tan_theta_eq_2_l481_481009


namespace ellipse_equation_point_M_exists_l481_481725

-- Condition: Point (1, sqrt(2)/2) lies on the ellipse
def point_lies_on_ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (a_gt_b : a > b) : Prop :=
  (1, Real.sqrt 2 / 2).fst^2 / a^2 + (1, Real.sqrt 2 / 2).snd^2 / b^2 = 1

-- Condition: Eccentricity of the ellipse is sqrt(2)/2
def eccentricity_condition (a b : ℝ) (c : ℝ) : Prop :=
  c / a = Real.sqrt 2 / 2 ∧ a^2 = b^2 + c^2

-- Question (I): Equation of ellipse should be (x^2 / 2 + y^2 = 1)
theorem ellipse_equation (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (a_gt_b : a > b) (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : a = Real.sqrt 2 ∧ b = 1 := 
sorry

-- Question (II): There exists M such that MA · MB is constant
theorem point_M_exists (a b c x0 : ℝ)
    (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b) 
    (a_val : a = Real.sqrt 2) (b_val : b = 1) 
    (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : 
    ∃ (M : ℝ × ℝ), M.fst = 5 / 4 ∧ M.snd = 0 ∧ -7 / 16 = -7 / 16 := 
sorry

end ellipse_equation_point_M_exists_l481_481725


namespace rectangle_dimensions_l481_481552

theorem rectangle_dimensions (l w : ℝ) (h1 : l = 2 * w) (h2 : 2 * (l + w) = 3 * (l * w)) : 
  w = 1 ∧ l = 2 := by
  sorry

end rectangle_dimensions_l481_481552


namespace max_points_empty_1x1_square_l481_481107

open Set

variable (k n : ℕ)
variable (s : Set (ℝ × ℝ))

-- Define the square ABCD with side length 4
def ABCD : Set (ℝ × ℝ) := 
  {p | 0 < p.1 ∧ p.1 < 4 ∧ 0 < p.2 ∧ p.2 < 4}

-- Define a function to check if a 1x1 square is empty in its interior
def is_empty_interior (s : Set (ℝ × ℝ)) (x y : ℝ) : Prop :=
  ∀ p ∈ s, ¬((x < p.1 ∧ p.1 < x + 1) ∧ (y < p.2 ∧ p.2 < y + 1))

-- The statement to prove
theorem max_points_empty_1x1_square (k : ℕ) (k_le_15 : k ≤ 15) :
  ∀ s : Set (ℝ × ℝ), s ⊆ ABCD ∧ s.card = k →
  ∃ (x y : ℝ), 0 ≤ x ∧ x + 1 ≤ 4 ∧ 0 ≤ y ∧ y + 1 ≤ 4 ∧ is_empty_interior s x y :=
by
  sorry

end max_points_empty_1x1_square_l481_481107


namespace average_speed_last_segment_l481_481145

theorem average_speed_last_segment (D : ℝ) (T_mins : ℝ) (S1 S2 : ℝ) (t : ℝ) (S_last : ℝ) :
  D = 150 ∧ T_mins = 135 ∧ S1 = 50 ∧ S2 = 60 ∧ t = 45 →
  S_last = 90 :=
by
    sorry

end average_speed_last_segment_l481_481145


namespace length_MN_trapezoid_MN_value_l481_481798

variables {A B C D M N : ℝ}
variables {trapezoid_ABC: Trapezoid ABCD}
variables (midpoints : M = (A + B) / 2 ∧ N = (C + D) / 2)
variables (AD BC : ℝ)
variables (h1 : AD = 2) (h2 : BC = 4)

theorem length_MN :
  MN = (1 / 2) * (AD + BC) :=
by
  sorry

theorem trapezoid_MN_value :
  MN = 3 :=
by
  have := length_MN midpoints h1 h2,
  sorry

end length_MN_trapezoid_MN_value_l481_481798


namespace area_triangle_ABC_l481_481923

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ := 
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem area_triangle_ABC :
  let A := (-4 : ℝ,  3 : ℝ)
  let B := ( 0 : ℝ,  6 : ℝ)
  let C := ( 2 : ℝ, -2 : ℝ)
  area_of_triangle A B C = 19 :=
by
  sorry

end area_triangle_ABC_l481_481923


namespace find_x100_l481_481727

noncomputable def geomSeqTerm (n : ℕ) : ℕ :=
  2 ^ (n - 1)

noncomputable def arithSeqTerm (n : ℕ) : ℕ :=
  5 * n - 3

theorem find_x100 :
  let x100 := 2 ^ 397 in
  ∀ n : ℕ, geomSeqTerm (4 * n - 3) = arithSeqTerm n → n = 100 →
  geomSeqTerm (4 * n - 3) = x100 := 
by
  sorry

end find_x100_l481_481727


namespace complex_magnitude_l481_481026

noncomputable def z (i : ℂ) : ℂ := i
def equation_holds (z i : ℂ) : Prop := (1 - i) * z = 1 + i
def magnitude (z : ℂ) : ℝ := complex.abs z

theorem complex_magnitude (z i : ℂ) (h: equation_holds z i) : magnitude z = 1 := by
  sorry

end complex_magnitude_l481_481026


namespace triangle_area_and_angle_l481_481773

theorem triangle_area_and_angle (a b c A B C : ℝ) 
  (habc: A + B + C = Real.pi)
  (h1: (2*a + b)*Real.cos C + c*Real.cos B = 0)
  (h2: c = 2*Real.sqrt 6 / 3)
  (h3: Real.sin A * Real.cos B = (Real.sqrt 3 - 1)/4) :
  (C = 2*Real.pi / 3) ∧ (1/2 * b * c * Real.sin A = (6 - 2 * Real.sqrt 3)/9) :=
by
  sorry

end triangle_area_and_angle_l481_481773


namespace joneal_stops_at_A_l481_481506

def circumference := 75 -- Circumference of the track in feet
def distance_run := 4950 -- Total distance Joneal runs in feet

def laps (distance_run : ℕ) (circumference : ℕ) : ℕ :=
  distance_run / circumference

theorem joneal_stops_at_A :
  laps distance_run circumference * circumference = distance_run →
  distance_run % circumference = 0 →
  "A" :=
by
  intros h1 h2
  -- At this point, it is clear Joneal stops at the starting point 'S', corresponding to quarter 'A'.
  exact "A"

end joneal_stops_at_A_l481_481506


namespace probability_two_positives_one_negative_l481_481085

-- Define the arithmetic sequence
noncomputable def a_n (n : ℕ) : ℤ := 10 - 2 * n

-- Condition definitions
def a_4_eq_2 := a_n 4 = 2
def a_7_eq_neg4 := a_n 7 = -4

-- Prove target probability
theorem probability_two_positives_one_negative :
  (∃ a_n, (a_4_eq_2 ∧ a_7_eq_neg4)) →
  let positive_prob := (6 / 10 : ℚ)
  let negative_prob := (4 / 10 : ℚ)
  let target_prob := 3 * negative_prob * (positive_prob^2) 
  target_prob = (6 / 25 : ℚ) :=
by
  sorry

end probability_two_positives_one_negative_l481_481085


namespace ray_DY_bisects_angle_ZDB_l481_481143

theorem ray_DY_bisects_angle_ZDB
  (A B C D X Y Z : Type)
  [quads : quadrilateral A B C D]
  [circleΩ : circle Ω]
  (h1: inscribed ABCD Ω)
  (h2: BC = CD)
  (h3: diagonals_intersect A C B D X)
  (h4: AD < AB)
  (h5: circumcircle_of_triangle_BCX_intersects_AB_at_Y BCX A B (Y ≠ B))
  (h6: ray_CY_meets_Ω_again_at_Z (CY) Ω Z (Z ≠ C)) :
  (ray_DY_bisects_angle_ZDB D Y Z D B) := 
sorry

end ray_DY_bisects_angle_ZDB_l481_481143


namespace cross_product_solution_l481_481680

def vec1 := (4, -3, 5)
def vec2 := (2, -2, 7)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := u
  let (x, y, z) := v
  (b * z - c * y, c * x - a * z, a * y - b * x)

theorem cross_product_solution : cross_product vec1 vec2 = (-11, -18, -2) :=
by
  sorry

end cross_product_solution_l481_481680


namespace probability_of_rolling_less_than_5_l481_481221

theorem probability_of_rolling_less_than_5 :
  (let outcomes := 8 in
   let successes := 4 in
   let probability := (successes: ℚ) / outcomes in
   probability = 1 / 2) :=
by
  sorry

end probability_of_rolling_less_than_5_l481_481221


namespace original_number_is_two_l481_481588

theorem original_number_is_two (x : ℝ) (hx : 0 < x) (h : x^2 = 8 * (1 / x)) : x = 2 :=
  sorry

end original_number_is_two_l481_481588


namespace infinite_numbers_of_form_n_sq_plus_3_l481_481880

theorem infinite_numbers_of_form_n_sq_plus_3 :
  ∃ᶠ n : ℕ in at_top, ∀ p : ℕ, prime p → p ∣ n^2 + 3 → ∃ k : ℕ, k^2 + 3 ∣ n^2 + 3 ∧ k < n :=
sorry

end infinite_numbers_of_form_n_sq_plus_3_l481_481880


namespace line_equation_cartesian_circle_equation_cartesian_l481_481080

theorem line_equation_cartesian (t : ℝ) (x y : ℝ) : 
  (x = 3 - (Real.sqrt 2 / 2) * t ∧ y = Real.sqrt 5 + (Real.sqrt 2 / 2) * t) -> 
  y = -2 * x + 6 + Real.sqrt 5 :=
sorry

theorem circle_equation_cartesian (ρ θ x y : ℝ) : 
  (ρ = 2 * Real.sqrt 5 * Real.sin θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) -> 
  x^2 = 0 :=
sorry

end line_equation_cartesian_circle_equation_cartesian_l481_481080


namespace stella_desired_weather_l481_481529

noncomputable def probability_of_exactly_two_sunny_days (rain_prob sunny_prob : ℚ) (days holiday_length : ℕ) : ℚ :=
  ∑ s in finset.powersetLen days (finset.range holiday_length), (rain_prob ^ (holiday_length - days)) * (sunny_prob ^ days)

theorem stella_desired_weather :
  probability_of_exactly_two_sunny_days (3/5) (2/5) 2 5 = 4320 / 15625 := by
  sorry

end stella_desired_weather_l481_481529


namespace baker_sold_cakes_l481_481290

theorem baker_sold_cakes :
  ∀ (C : ℕ),  -- C is the number of cakes Baker sold
    (∃ (cakes pastries : ℕ), 
      cakes = 14 ∧ 
      pastries = 153 ∧ 
      (∃ (sold_pastries : ℕ), sold_pastries = 8 ∧ 
      C = 89 + sold_pastries)) 
  → C = 97 :=
by
  intros C h
  rcases h with ⟨cakes, pastries, cakes_eq, pastries_eq, ⟨sold_pastries, sold_pastries_eq, C_eq⟩⟩
  -- Fill in the proof details
  sorry

end baker_sold_cakes_l481_481290


namespace remainder_t_div_6_l481_481548

theorem remainder_t_div_6 (s t : ℕ) (h1 : s % 6 = 2) (h2 : s > t) (h3 : (s - t) % 6 = 5) : t % 6 = 3 :=
by
  sorry

end remainder_t_div_6_l481_481548


namespace termites_eaten_black_squares_l481_481156

def is_black_square (i j : ℕ) : Prop :=
  (i % 2 = 0 ∧ j % 2 = 0) ∨ (i % 2 = 1 ∧ j % 2 = 1)

noncomputable def count_black_squares (damaged : list (ℕ × ℕ)) : ℕ :=
  damaged.filter (λ pos, is_black_square pos.1 pos.2).length

theorem termites_eaten_black_squares (damaged: list (ℕ × ℕ)) (h_damaged: damaged = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (3, 1), (4, 0), (4, 2), (5, 1), (6, 0), (6, 2), (7, 1)]) :
  count_black_squares damaged = 12 :=
by sorry

end termites_eaten_black_squares_l481_481156


namespace isosceles_triangle_tangent_condition_l481_481456

theorem isosceles_triangle_tangent_condition {A B C : ℝ}
    (h : Real.tan (A - B) + Real.tan (B - C) + Real.tan (C - A) = 0) :
    ∃ ⦃a b c : ℝ⦄, (Real.tan A = a) ∧ (Real.tan B = b) ∧ (Real.tan C = c) ∧ (a = b ∨ b = c ∨ c = a) :=
sorry

end isosceles_triangle_tangent_condition_l481_481456


namespace rational_powers_implies_rational_l481_481063

theorem rational_powers_implies_rational (x : ℝ) (m n : ℤ) (hmn : Int.gcd m n = 1)
  (hm : x^m ∈ ℚ) (hn : x^n ∈ ℚ) : x ∈ ℚ := sorry

end rational_powers_implies_rational_l481_481063


namespace triangle_tan_inequality_l481_481800

theorem triangle_tan_inequality (A B C : ℝ) (hA : A + B + C = π) :
    (Real.tan A)^2 + (Real.tan B)^2 + (Real.tan C)^2 ≥ (Real.tan A) * (Real.tan B) + (Real.tan B) * (Real.tan C) + (Real.tan C) * (Real.tan A) :=
by
  sorry

end triangle_tan_inequality_l481_481800


namespace larger_inscribed_angle_corresponds_to_larger_chord_l481_481865

theorem larger_inscribed_angle_corresponds_to_larger_chord
  (R : ℝ) (α β : ℝ) (hα : α < 90) (hβ : β < 90) (h : α < β)
  (BC LM : ℝ) (hBC : BC = 2 * R * Real.sin α) (hLM : LM = 2 * R * Real.sin β) :
  BC < LM :=
sorry

end larger_inscribed_angle_corresponds_to_larger_chord_l481_481865


namespace sum_totient_mod_1000_l481_481825

open Nat

def euler_totient (n : ℕ) : ℕ :=
  ∑ i in range n.succ, if gcd i n = 1 then 1 else 0

theorem sum_totient_mod_1000 :
  (∑ d in divisors 2008, euler_totient d) % 1000 = 8 :=
by
  sorry

end sum_totient_mod_1000_l481_481825


namespace geometric_sequence_common_ratio_l481_481782

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h₁ : a 1 = 1) (h₂ : a 1 * a 2 * a 3 = -8) :
  q = -2 :=
sorry

end geometric_sequence_common_ratio_l481_481782


namespace maximum_sum_of_squares_of_roots_l481_481694

theorem maximum_sum_of_squares_of_roots (a : ℝ) :
  (-3 ≤ a ∧ a ≤ -1) →
  let p := (2a : ℝ) in
  let q := (2a^2 + 4a + 3 : ℝ) in
  let disc := p^2 - 4 * (q : ℝ) in
  disc ≥ 0 → 
  let sum_squares := -8*a - 6 in
  -3 = a → sum_squares = 18 := 
by
  intros h leq disc_pos a_eq
  sorry

end maximum_sum_of_squares_of_roots_l481_481694


namespace number_of_women_is_24_l481_481450

-- Define the variables and conditions
variables (x : ℕ) (men_initial : ℕ) (women_initial : ℕ) (men_current : ℕ) (women_current : ℕ)

-- representing the initial ratio and the changes
def initial_conditions : Prop :=
  men_initial = 4 * x ∧ women_initial = 5 * x ∧
  men_current = men_initial + 2 ∧ women_current = 2 * (women_initial - 3)

-- representing the current number of men
def current_men_condition : Prop := men_current = 14

-- The proof we need to generate
theorem number_of_women_is_24 (x : ℕ) (men_initial women_initial men_current women_current : ℕ)
  (h1 : initial_conditions x men_initial women_initial men_current women_current)
  (h2 : current_men_condition men_current) : women_current = 24 :=
by
  -- proof steps here
  sorry

end number_of_women_is_24_l481_481450


namespace nylon_needed_for_one_dog_collor_l481_481806

-- Define the conditions as given in the problem
def nylon_for_dog (x : ℝ) : ℝ := x
def nylon_for_cat : ℝ := 10
def total_nylon_used (x : ℝ) : ℝ := 9 * (nylon_for_dog x) + 3 * (nylon_for_cat)

-- Prove the required statement under the given conditions
theorem nylon_needed_for_one_dog_collor : total_nylon_used 18 = 192 :=
by
  -- adding the proof step using sorry as required
  sorry

end nylon_needed_for_one_dog_collor_l481_481806


namespace polynomial_divisibility_l481_481547

noncomputable def C_and_D_sum (C D : ℝ) : ℝ := C + D

theorem polynomial_divisibility (C D : ℝ) (h : ∃ (P : ℝ[X]), (X^103 + C • X + C = (X^2 - X + 1) * P)) : C_and_D_sum C D = -1 :=
by
  sorry

end polynomial_divisibility_l481_481547


namespace range_of_f_in_acute_triangle_maximum_area_of_triangle_l481_481074

theorem range_of_f_in_acute_triangle (A B C a b c : ℝ) 
  (h1 : A + C = 2 * π / 3) 
  (h2 : b = 1)
  (h3 : A + B + C = π) 
  (h4 : ∀ (x : ℝ), A = x → A > 0 → A < π / 2) :
  (sqrt(3) < 2 * sin (A + π / 6) ∧ 2 * sin (A + π / 6) ≤ 2) := 
sorry

theorem maximum_area_of_triangle (A B C a b c : ℝ) 
  (h1 : A + C = 2 * π / 3) 
  (h2 : b = 1)
  (h3 : A + B + C = π)
  (h4 : B = π / 3) :
  (1 / 2 * a * c * sin (π / 3) = sqrt(3) / 4) :=
sorry

end range_of_f_in_acute_triangle_maximum_area_of_triangle_l481_481074


namespace find_minimal_x_l481_481477

-- Conditions
variables (x y : ℕ)
variable (pos_x : x > 0)
variable (pos_y : y > 0)
variable (h : 3 * x^7 = 17 * y^11)

-- Proof Goal
theorem find_minimal_x : ∃ a b c d : ℕ, x = a^c * b^d ∧ a + b + c + d = 30 :=
by {
  sorry
}

end find_minimal_x_l481_481477


namespace noemi_lost_on_roulette_l481_481132

theorem noemi_lost_on_roulette (initial_purse := 1700) (final_purse := 800) (loss_on_blackjack := 500) :
  (initial_purse - final_purse) - loss_on_blackjack = 400 := by
  sorry

end noemi_lost_on_roulette_l481_481132


namespace num_prime_pairs_sum_52_l481_481055

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_pairs_sum_to (s : ℕ) : ℕ :=
  (finset.filter (λ p : ℕ × ℕ, p.1 + p.2 = s ∧ is_prime p.1 ∧ is_prime p.2 ∧ p.1 ≤ p.2)
                 (finset.product (finset.range s) (finset.range s))).card

theorem num_prime_pairs_sum_52 : prime_pairs_sum_to 52 = 3 := 
sorry

end num_prime_pairs_sum_52_l481_481055


namespace area_of_circle_outside_triangle_l481_481478

noncomputable theory
open Real

def right_triangle (A B C : ℝ × ℝ) : Prop :=
  let ⟨ax, ay⟩ := A in let ⟨bx, by⟩ := B in let ⟨cx, cy⟩ := C in
  ax ≠ bx ∧ ax ≠ cx ∧ ay ≠ by ∧ ay ≠ cy ∧
  (bx - ax) ^ 2 + (cy - ay) ^ 2 = (cx - ax) ^ 2 + (by - ay) ^ 2

def circle_tangent (A B C : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) : Prop :=
  let ⟨ax, ay⟩ := A in let ⟨bx, by⟩ := B in let ⟨cx, cy⟩ := C in
  let ⟨ox, oy⟩ := O in 
  (ox - ax)^2 + (oy - ay)^2 = r^2 ∧
  (ox - bx)^2 + (oy - by)^2 = r^2 ∧
  (ox - cx)^2 + (oy - cy)^2 = r^2

def diameter_points_on_bc (A B C : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) (X' Y' : ℝ × ℝ) : Prop :=
  let ⟨bx, by⟩ := B in let ⟨cx, cy⟩ := C in
  let ⟨x'x, x'y⟩ := X' in let ⟨y'x, y'y⟩ := Y' in
  let ⟨ox, oy⟩ := O in
  (ox - x'x)^2 + (oy - x'y)^2 = r^2 ∧
  (ox - y'x)^2 + (oy - y'y)^2 = r^2 ∧
  (bx ≤ x'x ∧ x'x ≤ cx ∧ by ≤ x'y ∧ x'y ≤ cy) ∧
  (bx ≤ y'x ∧ y'x ≤ cx ∧ by ≤ y'y ∧ y'y ≤ cy)

theorem area_of_circle_outside_triangle
  {A B C O X' Y' : ℝ × ℝ} {r : ℝ}
  (h1 : right_triangle A B C)
  (h2 : circle_tangent A B C O r)
  (h3 : diameter_points_on_bc A B C O r X' Y')
  (h4 : dist A B = 6) :
  π * r^2 / 4 - r^2 / 2 = π - 2 :=
sorry

end area_of_circle_outside_triangle_l481_481478


namespace three_collinear_points_l481_481505

theorem three_collinear_points (f : ℝ → Prop) (h_black_or_white : ∀ (x : ℝ), f x = true ∨ f x = false)
: ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b = (a + c) / 2) ∧ ((f a = f b) ∧ (f b = f c)) :=
sorry

end three_collinear_points_l481_481505


namespace tyler_saltwater_aquariums_l481_481197

def num_animals_per_aquarium : ℕ := 39
def total_saltwater_animals : ℕ := 2184

theorem tyler_saltwater_aquariums : 
  total_saltwater_animals / num_animals_per_aquarium = 56 := 
by
  sorry

end tyler_saltwater_aquariums_l481_481197


namespace amanda_pay_l481_481100

theorem amanda_pay (hourly_wage : ℝ) (overtime_rate : ℝ) (commission : ℝ) (total_hours : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) (penalty_rate : ℝ) :
  hourly_wage = 50 → commission = 150 → total_hours = 10 → regular_hours = 8 → overtime_hours = (total_hours - regular_hours) → 
  overtime_rate = 1.5 * hourly_wage → penalty_rate = 0.2 →
  let regular_pay := regular_hours * hourly_wage in
  let overtime_pay := overtime_hours * overtime_rate in
  let total_earnings := regular_pay + overtime_pay + commission in
  let penalty := penalty_rate * total_earnings in
  let earnings_with_penalty := total_earnings - penalty in
  earnings_with_penalty = 560 :=
by { intros, simp only [mul_add, mul_sub, add_assoc], rw [←hf, ←hx, ←hs, ←hp, ←ha], sorry }

end amanda_pay_l481_481100


namespace ratio_of_x_to_y_l481_481394

variable (x y : ℝ)

theorem ratio_of_x_to_y (h : 0.10 * x = 0.20 * y) : x / y = 2 :=
by sorry

end ratio_of_x_to_y_l481_481394


namespace rhombus_area_l481_481632

-- Define the given conditions: diagonals and side length
def d1 : ℕ := 40
def d2 : ℕ := 18
def s : ℕ := 25

-- Prove that the area of the rhombus is 360 square units given the conditions
theorem rhombus_area :
  (d1 * d2) / 2 = 360 :=
by
  sorry

end rhombus_area_l481_481632


namespace jian_wins_cases_l481_481203

inductive Move
| rock : Move
| paper : Move
| scissors : Move

def wins (jian shin : Move) : Prop :=
  (jian = Move.rock ∧ shin = Move.scissors) ∨
  (jian = Move.paper ∧ shin = Move.rock) ∨
  (jian = Move.scissors ∧ shin = Move.paper)

theorem jian_wins_cases : ∃ n : Nat, n = 3 ∧ (∀ jian shin, wins jian shin → n = 3) :=
by
  sorry

end jian_wins_cases_l481_481203


namespace circle_diameter_l481_481795

theorem circle_diameter
    (O : Point) (A B : Point) (r : ℝ)
    (hO_center : center_of O) 
    (hAB_chord : is_chord AB 5)
    (hAB_rectangle: forms_rectangle O A B C) :
    diameter O = 10 :=
by
    sorry

end circle_diameter_l481_481795


namespace primes_between_80_and_90_l481_481052

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

theorem primes_between_80_and_90 : 
  (set.count set_of (λ n, 80 ≤ n ∧ n ≤ 90 ∧ is_prime n)) = 2 :=
by
  sorry

end primes_between_80_and_90_l481_481052


namespace sum_of_fractional_g_values_l481_481113

def g (n : ℕ) : ℕ := ⌊(n:ℝ)^(1/3) + 0.5⌋

theorem sum_of_fractional_g_values : ∑ k in (Finset.range 1250 \Finset.singleton 0), (1 / g k : ℚ) = 1188 := 
by
  sorry

end sum_of_fractional_g_values_l481_481113


namespace probability_of_rolling_less_than_five_l481_481205

/-- Rolling an 8-sided die -/
def outcomes := { 1, 2, 3, 4, 5, 6, 7, 8 }

/-- Successful outcomes (numbers less than 5) -/
def successful_outcomes := { 1, 2, 3, 4 }

/-- Number of total outcomes -/
def total_outcomes := 8

/-- Number of successful outcomes -/
def num_successful_outcomes := 4

/-- Probability of rolling a number less than 5 on an 8-sided die -/
def probability : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_rolling_less_than_five : probability = 1 / 2 := by
  sorry

end probability_of_rolling_less_than_five_l481_481205


namespace technician_percentage_l481_481975

variable (D : ℝ) -- One-way distance to the service center
variable (P : ℝ) -- Percentage of the way back completed

-- Total distance of round trip is 2D
def round_trip_distance : ℝ := 2 * D

-- Technician has completed the drive to the center and some percent of the way back
def technician_distance : ℝ := D + (P / 100) * D

-- Technician has completed 70% of the round trip
axiom h : technician_distance D P = 0.7 * round_trip_distance D

-- The correct answer should be
def correct_answer : ℝ := 40

-- Prove that the percentage P is 40
theorem technician_percentage (D : ℝ) : P = correct_answer :=
by sorry

end technician_percentage_l481_481975


namespace solve_problem_l481_481172

-- Defining the poems
inductive Poem
| "Bringing_in_the_Wine"
| "Autumn_Dusk_at_the_Mountain_Residence"
| "Viewing_the_Mountain"
| "Sending_off_Du_Fu_to_Shu_State"
| Other1
| Other2

open Poem

noncomputable def numberOfArrangements : Nat :=
  let poems := [Bringing_in_the_Wine, Autumn_Dusk_at_the_Mountain_Residence, Viewing_the_Mountain, Sending_off_Du_Fu_to_Shu_State, Other1, Other2]
  let condition1 := ∀ i j, poems[i] = Bringing_in_the_Wine → poems[j] = Viewing_the_Mountain → i < j
  let condition2 := ∀ i, poems[i] = Autumn_Dusk_at_the_Mountain_Residence → i ≠ 5
  let condition3 := ∀ i, poems[i] = Sending_off_Du_Fu_to_Shu_State → i ≠ 5
  let condition4 := ∀ i j, poems[i] = Autumn_Dusk_at_the_Mountain_Residence → poems[j] = Sending_off_Du_Fu_to_Shu_State → |i - j| ≠ 1
  if (condition1 ∧ condition2 ∧ condition3 ∧ condition4) then 144 else 0

theorem solve_problem : numberOfArrangements = 144 := 
  sorry

end solve_problem_l481_481172


namespace range_of_k_equation_of_line_l_l481_481699

-- Definitions of the conditions for the Lean 4 statement.
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 20
def line_l (k x y : ℝ) : Prop := y = k * x
def intersects_at_two_distinct_points (k : ℝ) : Prop :=
  let d := abs (6 * k) / Real.sqrt (k^2 + 1) in
  d < Real.sqrt 20

-- Lean 4 statement for Part (I)
theorem range_of_k
  (h : ∀ k : ℝ, intersects_at_two_distinct_points k) :
  ∀ k : ℝ, -Real.sqrt 5 / 2 < k ∧ k < Real.sqrt 5 / 2 := sorry

-- Definitions for Part (II) conditions
def OB_eq_2_OA (x1 x2 : ℝ) : Prop :=
  x2 = 2 * x1

-- Lean 4 statement for Part (II)
theorem equation_of_line_l
  (h : ∀ k x : ℝ, circle_C x (k * x) → line_l k x (k * x)) 
  (h2 : ∀ k x1 x2 : ℝ, intersects_at_two_distinct_points k → OB_eq_2_OA x1 x2) :
  ∀ k : ℝ, y = ±k*x := sorry

end range_of_k_equation_of_line_l_l481_481699


namespace teacher_age_l481_481532

theorem teacher_age (students_avg_age : ℕ) (num_students : ℕ) (combined_avg_age : ℕ) :
  students_avg_age = 15 → num_students = 20 → combined_avg_age = 16 →
  (num_students + 1) * combined_avg_age - num_students * students_avg_age = 36 :=
by
  intros h_avg_students h_num_students h_avg_combined
  rw [h_avg_students, h_num_students, h_avg_combined]
  sorry

end teacher_age_l481_481532


namespace ordinate_of_P1_l481_481128

noncomputable def P2_x : ℝ := -3 / 5
noncomputable def alpha (h : 0 < α ∧ α < π / 2) : ℝ := α
noncomputable def P2_y (α : ℝ) : ℝ := sqrt (1 - P2_x ^ 2)
noncomputable def sin_alpha (α : ℝ) (h : 0 < α ∧ α < π / 2) : ℝ :=
  (sqrt 2 / 2) * (P2_y α + P2_x)

theorem ordinate_of_P1 (α : ℝ) (h : 0 < α ∧ α < π / 2) : sin_alpha α h = 7 * sqrt 2 / 10 :=
sorry

end ordinate_of_P1_l481_481128


namespace unique_mode_of_set_l481_481344

def is_mode (s : Finset ℕ) (m : ℕ) : Prop :=
  ∀ n ∈ s, s.count m ≥ s.count n ∧ s.count m > s.count n → m = n

theorem unique_mode_of_set 
  (x : ℕ) 
  (h : is_mode {2, 3, 5, x, 5, 3} 3) : x = 3 :=
sorry

end unique_mode_of_set_l481_481344


namespace women_count_l481_481432

/-- 
Initially, the men and women in a room were in the ratio of 4:5.
Then, 2 men entered the room and 3 women left the room.
The number of women then doubled.
There are now 14 men in the room.
Prove that the number of women currently in the room is 24.
-/
theorem women_count (x : ℕ) (h1 : 4 * x + 2 = 14) (h2 : 2 * (5 * x - 3) = n) : 
  n = 24 :=
by
  sorry

end women_count_l481_481432


namespace total_fires_l481_481988

-- Conditions as definitions
def Doug_fires : Nat := 20
def Kai_fires : Nat := 3 * Doug_fires
def Eli_fires : Nat := Kai_fires / 2

-- Theorem to prove the total number of fires
theorem total_fires : Doug_fires + Kai_fires + Eli_fires = 110 := by
  sorry

end total_fires_l481_481988


namespace women_current_in_room_l481_481455

-- Definitions
variable (m w : ℕ) -- number of men and women
variable (x : ℕ) -- positive integer representing the scaling factor for initial ratio

-- Conditions
def initial_ratio (x : ℕ) : Prop := m = 4*x ∧ w = 5*x
def after_changes (x : ℕ) : Prop := (m + 2) = 14 ∧ (2 * (w - 3)) = 24

-- Theorem statement
theorem women_current_in_room (x : ℕ) (m w : ℕ) (h1 : initial_ratio x) (h2 : after_changes x) : w = 24 :=
by
  sorry

end women_current_in_room_l481_481455


namespace minimal_mutually_visible_pairs_l481_481245

theorem minimal_mutually_visible_pairs (n : ℕ) (h_n : n = 155) 
  (P : Fin n → Point)
  (is_circle : ∀ i j : Fin n, P i ≠ P j → m(P i P j) > 10) : 
  ∃ (k m : ℕ) (h_km : 20 * 6 + 15 * 10 = 270) , n = k * 4 + m * 5 :=
by
  sorry

end minimal_mutually_visible_pairs_l481_481245


namespace salary_increase_percentage_l481_481549

theorem salary_increase_percentage (x : ℝ) (h₁ : ∀ (S : ℝ), S * (1 + x / 100) * 0.6 = 1.16 * S) :
  x ≈ 93.33 :=
by
  sorry

end salary_increase_percentage_l481_481549


namespace volume_circumsphere_tetrahedron_eq_l481_481090

open Real

/-- The volume of the circumscribed sphere of tetrahedron ABCD where
- all angles at D, namely ∠ADB, ∠BDC, and ∠CDA are 60 degrees,
- AD = BD = 3,
- and CD = 2,
is 4 * sqrt 3 * pi. -/
theorem volume_circumsphere_tetrahedron_eq :
  ∀ (A B C D : ℝ^3),
    ∠ A D B = 60 ∧ ∠ B D C = 60 ∧ ∠ C D A = 60 ∧
    dist A D = 3 ∧ dist B D = 3 ∧ dist C D = 2 →
    (4 / 3) * π * (sqrt 3) ^ 3 = 4 * sqrt 3 * π :=
by
  intros A B C D h
  sorry

end volume_circumsphere_tetrahedron_eq_l481_481090


namespace divisor_count_10_fac_l481_481387

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n+1) * factorial n

def prime_exponent (n p : ℕ) : ℕ :=
  if p = 0 then 0 else
  let rec aux (n m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc else (if m % p = 0 then aux n (m / p) (acc + 1) else acc)
  aux n n 0

def divisor_count (n : ℕ) : ℕ :=
  let primes := [2, 3, 5, 7]
  primes.foldl (λ acc p => acc * (prime_exponent n p + 1)) 1

theorem divisor_count_10_fac : divisor_count (factorial 10) = 270 := sorry

end divisor_count_10_fac_l481_481387


namespace sum_a_b_eq_neg_one_l481_481024

theorem sum_a_b_eq_neg_one (a b : ℝ) (i : ℂ) (h_i : i = complex.I) (h : (a + 2 * i) * i = b + i) : a + b = -1 :=
by
  sorry

end sum_a_b_eq_neg_one_l481_481024


namespace find_x_l481_481890

theorem find_x :
  ∃ x : ℚ, (1 / 3) * ((x + 8) + (8*x + 3) + (3*x + 9)) = 5*x - 9 ∧ x = 47 / 3 :=
by
  sorry

end find_x_l481_481890


namespace min_sum_of_squares_of_y_coords_l481_481622

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_point (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

theorem min_sum_of_squares_of_y_coords :
  ∃ (m : ℝ), ∀ (x1 y1 x2 y2 : ℝ),
  (line_through_point m x1 y1) →
  (parabola x1 y1) →
  (line_through_point m x2 y2) →
  (parabola x2 y2) →
  x1 ≠ x2 → 
  ((y1 + y2)^2 - 2 * y1 * y2) = 32 :=
sorry

end min_sum_of_squares_of_y_coords_l481_481622


namespace find_n_th_term_l481_481898

theorem find_n_th_term :
  ∃ n : ℕ, 
    ∀ (x : ℕ),
    let a1 := 3 * x - 5,
        a2 := 7 * x - 15,
        a3 := 4 * x + 3,
        d  := a2 - a1 in
    d = a3 - a2 ∧ 
    (∃ x', x' = 4 ∧ 4019 = a1 + (669 - 1) * d) ∧
    n = 669 := sorry

end find_n_th_term_l481_481898


namespace product_real_parts_roots_l481_481475

-- Definitions
def i := complex.I

-- Theorem statement
theorem product_real_parts_roots : 
  (λ (z : ℂ), z^2 + 3 * z + (7 - 5 * i) = 0) roots (z^2 + 3 * z = -7 + 5 * i) → real_part_z1 * real_part_z2 = 4 := 
sorry

end product_real_parts_roots_l481_481475


namespace ellipse_equation_area_triangle_MON_l481_481896

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
(ecc : b^2 + (a * (sqrt 3 / 2))^2 = a^2) : 
  a = 2 ∧ b = 1 ∧ (∀ x y, (x ^ 2 / 4 + y ^ 2 = 1) ↔ (a = 2 ∧ b = 1 ∧ x^2 / a^2 + y^2 / b^2 = 1)) :=
by 
  sorry

theorem area_triangle_MON (k1 k2 : ℝ) (h : k1 * k2 = -1/4) : 
  ∀ m : ℝ, 2 * sqrt ((m^2 / (4 * (k1^2 + k2^2) + 1)) * (1 - (m^2 / (4 * (k1^2 + k2^2) + 1)))) = 1 :=
by
  sorry

end ellipse_equation_area_triangle_MON_l481_481896


namespace area_of_triangle_aef_l481_481167

noncomputable def length_ab : ℝ := 10
noncomputable def width_ad : ℝ := 6
noncomputable def diagonal_ac : ℝ := Real.sqrt (length_ab^2 + width_ad^2)
noncomputable def segment_length_ac : ℝ := diagonal_ac / 4
noncomputable def area_aef : ℝ := (1/2) * segment_length_ac * ((60 * diagonal_ac) / diagonal_ac^2)

theorem area_of_triangle_aef : area_aef = 7.5 := by
  sorry

end area_of_triangle_aef_l481_481167


namespace exists_cycle_l481_481555

-- Define the problem conditions as a graph
structure Graph(V : Type u) :=
  (adj : V → V → Prop)

namespace Graph

-- Add necessary assumptions about the graph
variables {V : Type u} (G : Graph V)
variable [Fintype V]  -- Finite number of vertices
variable [DecidableRel G.adj]
variable (h_n_edges : Fintype.card {e // adj e.1 e.2} = Fintype.card V)  -- n edges
variable (h_one_edge_between : ∀ u v : V, ∃! e : G.adj u v, adj u v)  -- at most one edge between any two vertices

-- Main theorem statement
theorem exists_cycle : ∃ (v : V), ∃ (cycle : List V), 
  cycle.head = v ∧ cycle.last = some v ∧
  (∀ u v ∈ cycle, adj u v) ∧ (∀ u ∈ cycle, cycle.count u = 2) :=
sorry

end Graph

end exists_cycle_l481_481555


namespace sum_abs_values_of_factors_l481_481062

theorem sum_abs_values_of_factors (a w c d : ℤ)
  (h1 : 6 * (x : ℤ)^2 + x - 12 = (a * x + w) * (c * x + d)) :
  abs a + abs w + abs c + abs d = 22 :=
sorry

end sum_abs_values_of_factors_l481_481062


namespace perpendicular_slopes_l481_481724

theorem perpendicular_slopes {m : ℝ} (h : (1 : ℝ) * -m = -1) : m = 1 :=
by sorry

end perpendicular_slopes_l481_481724


namespace particle_speed_is_correct_l481_481957

def position (t : ℝ) : ℝ × ℝ := (2 * t + 7, 4 * t - 13)

def speed := (|2|^2 + |4|^2)^(1/2)

theorem particle_speed_is_correct : 
  (|2|^2 + |4|^2)^(1/2) = 2 * Real.sqrt 5 := 
by
  sorry -- proof omitted

end particle_speed_is_correct_l481_481957


namespace probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l481_481185

/-- 
There are 30 tiles in box C numbered from 1 to 30 and 30 tiles in box D numbered from 21 to 50. 
We want to prove that the probability of drawing a tile less than 20 from box C and a tile that 
is either odd or greater than 40 from box D is 19/45. 
-/
theorem probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40 :
  (19 / 30) * (2 / 3) = (19 / 45) :=
by sorry

end probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l481_481185


namespace women_count_l481_481434

/-- 
Initially, the men and women in a room were in the ratio of 4:5.
Then, 2 men entered the room and 3 women left the room.
The number of women then doubled.
There are now 14 men in the room.
Prove that the number of women currently in the room is 24.
-/
theorem women_count (x : ℕ) (h1 : 4 * x + 2 = 14) (h2 : 2 * (5 * x - 3) = n) : 
  n = 24 :=
by
  sorry

end women_count_l481_481434


namespace correct_calculation_l481_481927

theorem correct_calculation (a b : ℝ) : (5 * a^2 * b - 6 * a^2 * b = -a^2 * b) ∧ 
  ¬(3 * a - 4 * a = -1) ∧ 
  ¬(a^2 + a^2 = a^4) ∧ 
  ¬(3 * a^2 + 2 * a^3 = 5 * a^5) :=
begin
  sorry
end

end correct_calculation_l481_481927


namespace area_inside_octagon_outside_semicircles_l481_481784

theorem area_inside_octagon_outside_semicircles :
  let s := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area := (1/2) * Real.pi * (s / 2)^2
  let total_semicircle_area := 8 * semicircle_area
  octagon_area - total_semicircle_area = 54 + 24 * Real.sqrt 2 - 9 * Real.pi :=
sorry

end area_inside_octagon_outside_semicircles_l481_481784


namespace mile_time_sum_is_11_l481_481564

def mile_time_sum (Tina_time Tony_time Tom_time : ℕ) : ℕ :=
  Tina_time + Tony_time + Tom_time

theorem mile_time_sum_is_11 :
  ∃ (Tina_time Tony_time Tom_time : ℕ),
  (Tina_time = 6 ∧ Tony_time = Tina_time / 2 ∧ Tom_time = Tina_time / 3) →
  mile_time_sum Tina_time Tony_time Tom_time = 11 :=
by
  sorry

end mile_time_sum_is_11_l481_481564


namespace polynomial_B_value_l481_481643

theorem polynomial_B_value
  (A B C D : ℤ)
  (roots : Fin 6 → ℤ)
  (positive_integers : ∀ i, roots i > 0)
  (poly_eq : ∀ z, (∏ i, (z - roots i)) = z^6 - 12 * z^5 + A * z^4 + B * z^3 + C * z^2 + D * z + 144) :
  B = -126 :=
by sorry

end polynomial_B_value_l481_481643


namespace square_area_and_multiply_l481_481490

theorem square_area_and_multiply (s : ℝ) (area : ℝ) : 
    s = 10^0.2 ∧ area = s^2 → 
    10^(0.2 * 2) * 10^0.1 * 10^(-0.3) * 10^0.4 = 10^0.6 :=
by
    -- Assume conditions
    intro h,
    cases h with hs ha,
    -- Define assumptions in Lean
    rw [hs] at ha,
    -- Display assumptions without solving
    have ha : area = 10^(0.2 * 2) := ha,
    have hs0 : 10^(0.2 * 2) = 10^0.4 := by norm_num,
    rw [hs0] at ha,
    have hs1 : 10^(0.4) * 10^0.1 * 10^(-0.3) * 10^0.4 = 10^(0.4+0.1-0.3+0.4) := by norm_num,
    rw [hs1],
    have hs2 : 10^(0.4 + 0.1 - 0.3 + 0.4) = 10^0.6 := by norm_num,
    exact hs2
    -- Proof completed

end square_area_and_multiply_l481_481490


namespace smallest_positive_debt_resolved_l481_481196

theorem smallest_positive_debt_resolved : ∃ (D : ℤ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 250 * g) ∧ D = 50 :=
by
  sorry

end smallest_positive_debt_resolved_l481_481196


namespace acute_triangle_tangent_sum_range_l481_481787

theorem acute_triangle_tangent_sum_range
  (a b c : ℝ) (A B C : ℝ)
  (triangle_ABC_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (opposite_sides : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (side_relation : b^2 - a^2 = a * c)
  (angle_relation : A + B + C = π)
  (angles_in_radians : 0 < A ∧ A < π)
  (angles_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  1 < (1 / Real.tan A + 1 / Real.tan B) ∧ (1 / Real.tan A + 1 / Real.tan B) < 2 * Real.sqrt 3 / 3 :=
sorry 

end acute_triangle_tangent_sum_range_l481_481787


namespace neg_p_sufficient_not_necessary_for_neg_q_l481_481349

noncomputable def p (x : ℝ) : Prop := abs (x + 1) > 0
noncomputable def q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬ p x → ¬ q x) ∧ ¬ (∀ x, ¬ q x → ¬ p x) :=
by
  sorry

end neg_p_sufficient_not_necessary_for_neg_q_l481_481349
