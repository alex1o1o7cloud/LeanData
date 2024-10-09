import Mathlib

namespace distinct_pairs_l715_71597

-- Definitions of rational numbers and distinctness.
def is_distinct (x y : ℚ) : Prop := x ≠ y

-- Conditions
variables {a b r s : ℚ}

-- Main theorem: prove that there is only 1 distinct pair (a, b)
theorem distinct_pairs (h_ab_distinct : is_distinct a b)
  (h_rs_distinct : is_distinct r s)
  (h_eq : ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s)) : 
    ∃! (a b : ℚ), ∀ z : ℚ, (z - r) * (z - s) = (z - a * r) * (z - b * s) :=
  sorry

end distinct_pairs_l715_71597


namespace years_passed_l715_71589

-- Let PV be the present value of the machine, FV be the final value of the machine, r be the depletion rate, and t be the time in years.
def PV : ℝ := 900
def FV : ℝ := 729
def r : ℝ := 0.10

-- The formula for exponential decay is FV = PV * (1 - r)^t.
-- Given FV = 729, PV = 900, and r = 0.10, we want to prove that t = 2.

theorem years_passed (t : ℕ) : FV = PV * (1 - r)^t → t = 2 := 
by 
  intro h
  sorry

end years_passed_l715_71589


namespace possible_denominators_count_l715_71572

theorem possible_denominators_count :
  ∀ a b c : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ (a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9) →
  ∃ (D : Finset ℕ), D.card = 7 ∧ 
  ∀ num denom, (num = 100*a + 10*b + c) → (denom = 999) → (gcd num denom > 1) → 
  denom ∈ D := 
sorry

end possible_denominators_count_l715_71572


namespace max_non_intersecting_segments_l715_71576

theorem max_non_intersecting_segments (n m : ℕ) (hn: 1 < n) (hm: m ≥ 3): 
  ∃ L, L = 3 * n - m - 3 :=
by
  sorry

end max_non_intersecting_segments_l715_71576


namespace area_of_rectangle_l715_71517

theorem area_of_rectangle (width length : ℝ) (h_width : width = 5.4) (h_length : length = 2.5) : width * length = 13.5 :=
by
  -- We are given that the width is 5.4 and the length is 2.5
  -- We need to show that the area (width * length) is 13.5
  sorry

end area_of_rectangle_l715_71517


namespace avg_age_10_students_l715_71553

-- Defining the given conditions
def avg_age_15_students : ℕ := 15
def total_students : ℕ := 15
def avg_age_4_students : ℕ := 14
def num_4_students : ℕ := 4
def age_15th_student : ℕ := 9

-- Calculating the total age based on given conditions
def total_age_15_students : ℕ := avg_age_15_students * total_students
def total_age_4_students : ℕ := avg_age_4_students * num_4_students
def total_age_10_students : ℕ := total_age_15_students - total_age_4_students - age_15th_student

-- Problem to be proved
theorem avg_age_10_students : total_age_10_students / 10 = 16 := 
by sorry

end avg_age_10_students_l715_71553


namespace increasing_function_on_R_l715_71518

theorem increasing_function_on_R (x1 x2 : ℝ) (h : x1 < x2) : 3 * x1 + 2 < 3 * x2 + 2 := 
by
  sorry

end increasing_function_on_R_l715_71518


namespace sequence_eventually_periodic_l715_71533

-- Definitions based on the conditions
def positive_int_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < a n

def satisfies_condition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = a (n + 2) * a (n + 3)

-- Assertion to prove based on the question
theorem sequence_eventually_periodic (a : ℕ → ℕ) 
  (h1 : positive_int_sequence a) 
  (h2 : satisfies_condition a) : 
  ∃ p : ℕ, ∃ k : ℕ, ∀ n : ℕ, a (n + k) = a n :=
sorry

end sequence_eventually_periodic_l715_71533


namespace smallest_value_am_hm_inequality_l715_71578

theorem smallest_value_am_hm_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end smallest_value_am_hm_inequality_l715_71578


namespace unique_function_satisfies_condition_l715_71525

theorem unique_function_satisfies_condition :
  ∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x * Real.sin y) + f (x * Real.sin z) -
    f x * f (Real.sin y * Real.sin z) + Real.sin (Real.pi * x) ≥ 1 := sorry

end unique_function_satisfies_condition_l715_71525


namespace range_of_m_l715_71507

open Set

def M (m : ℝ) : Set ℝ := {x | x ≤ m}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}

theorem range_of_m (m : ℝ) : (M m ∩ N).Nonempty ↔ m > 0 := sorry

end range_of_m_l715_71507


namespace cos_angle_value_l715_71543

noncomputable def cos_angle := Real.cos (19 * Real.pi / 4)

theorem cos_angle_value : cos_angle = -Real.sqrt 2 / 2 := by
  sorry

end cos_angle_value_l715_71543


namespace stamps_per_light_envelope_l715_71555

theorem stamps_per_light_envelope 
  (stamps_heavy : ℕ) (stamps_light : ℕ → ℕ) (total_light : ℕ) (total_stamps_light : ℕ)
  (total_envelopes : ℕ) :
  (∀ n, n > 5 → stamps_heavy = 5) →
  (∀ n, n <= 5 → stamps_light n = total_stamps_light / total_light) →
  total_light = 6 →
  total_stamps_light = 52 →
  total_envelopes = 14 →
  stamps_light 5 = 9 :=
by
  sorry

end stamps_per_light_envelope_l715_71555


namespace compute_sum_l715_71559

open BigOperators

theorem compute_sum : 
  (1 / 2 ^ 2010 : ℝ) * ∑ n in Finset.range 1006, (-3 : ℝ) ^ n * (Nat.choose 2010 (2 * n)) = -1 / 2 :=
by
  sorry

end compute_sum_l715_71559


namespace cos_third_quadrant_l715_71536

theorem cos_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l715_71536


namespace g_periodic_6_l715_71531

def g (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a + b, b + c, a + c)

def g_iter (n : Nat) (triple : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match n with
  | 0 => triple
  | n + 1 => g (g_iter n triple).1 (g_iter n triple).2.1 (g_iter n triple).2.2

theorem g_periodic_6 {a b c : ℝ} (h : ∃ n : Nat, n > 0 ∧ g_iter n (a, b, c) = (a, b, c))
  (h' : (a, b, c) ≠ (0, 0, 0)) : g_iter 6 (a, b, c) = (a, b, c) :=
by
  sorry

end g_periodic_6_l715_71531


namespace derivative_at_one_l715_71510

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem derivative_at_one : deriv f 1 = 2 :=
by sorry

end derivative_at_one_l715_71510


namespace beth_sheep_l715_71574

-- Definition: number of sheep Beth has (B)
variable (B : ℕ)

-- Condition 1: Aaron has 7 times as many sheep as Beth
def Aaron_sheep (B : ℕ) := 7 * B

-- Condition 2: Together, Aaron and Beth have 608 sheep
axiom together_sheep : B + Aaron_sheep B = 608

-- Theorem: Prove that Beth has 76 sheep
theorem beth_sheep : B = 76 :=
sorry

end beth_sheep_l715_71574


namespace average_scissors_correct_l715_71509

-- Definitions for the initial number of scissors in each drawer
def initial_scissors_first_drawer : ℕ := 39
def initial_scissors_second_drawer : ℕ := 27
def initial_scissors_third_drawer : ℕ := 45

-- Definitions for the new scissors added by Dan
def added_scissors_first_drawer : ℕ := 13
def added_scissors_second_drawer : ℕ := 7
def added_scissors_third_drawer : ℕ := 10

-- Calculate the final number of scissors after Dan's addition
def final_scissors_first_drawer : ℕ := initial_scissors_first_drawer + added_scissors_first_drawer
def final_scissors_second_drawer : ℕ := initial_scissors_second_drawer + added_scissors_second_drawer
def final_scissors_third_drawer : ℕ := initial_scissors_third_drawer + added_scissors_third_drawer

-- Statement to prove the average number of scissors in all three drawers
theorem average_scissors_correct :
  (final_scissors_first_drawer + final_scissors_second_drawer + final_scissors_third_drawer) / 3 = 47 := by
  sorry

end average_scissors_correct_l715_71509


namespace sqrt_difference_square_l715_71541

theorem sqrt_difference_square (a b : ℝ) (h₁ : a = Real.sqrt 3 + Real.sqrt 2) (h₂ : b = Real.sqrt 3 - Real.sqrt 2) : a^2 - b^2 = 4 * Real.sqrt 6 := by
  sorry

end sqrt_difference_square_l715_71541


namespace feet_more_than_heads_l715_71563

def num_hens := 50
def num_goats := 45
def num_camels := 8
def num_keepers := 15

def feet_per_hen := 2
def feet_per_goat := 4
def feet_per_camel := 4
def feet_per_keeper := 2

def total_heads := num_hens + num_goats + num_camels + num_keepers
def total_feet := (num_hens * feet_per_hen) + (num_goats * feet_per_goat) + (num_camels * feet_per_camel) + (num_keepers * feet_per_keeper)

-- Theorem to prove:
theorem feet_more_than_heads : total_feet - total_heads = 224 := by
  -- proof goes here
  sorry

end feet_more_than_heads_l715_71563


namespace dog_tail_length_l715_71586

theorem dog_tail_length (b h t : ℝ) 
  (h_head : h = b / 6) 
  (h_tail : t = b / 2) 
  (h_total : b + h + t = 30) : 
  t = 9 :=
by
  sorry

end dog_tail_length_l715_71586


namespace find_positive_x_l715_71571

theorem find_positive_x (x y z : ℝ) 
  (h1 : x * y = 15 - 3 * x - 2 * y)
  (h2 : y * z = 8 - 2 * y - 4 * z)
  (h3 : x * z = 56 - 5 * x - 6 * z) : x = 8 := 
sorry

end find_positive_x_l715_71571


namespace problem_geometric_sequence_l715_71565

variable {α : Type*} [LinearOrderedField α]

noncomputable def geom_sequence_5_8 (a : α) (h : a + 8 * a = 2) : α :=
  (a * 2^4 + a * 2^7)

theorem problem_geometric_sequence : ∃ (a : α), (a + 8 * a = 2) ∧ geom_sequence_5_8 a (sorry) = 32 := 
by sorry

end problem_geometric_sequence_l715_71565


namespace intersecting_chords_theorem_l715_71514

theorem intersecting_chords_theorem
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18)
  (c d k : ℝ) (h3 : c = 3 * k) (h4 : d = 8 * k) :
  (a * b = c * d) → (k = 3) → (c + d = 33) :=
by 
  sorry

end intersecting_chords_theorem_l715_71514


namespace shortest_chord_line_through_P_longest_chord_line_through_P_l715_71580

theorem shortest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = 1/2 * x + 5/2 → a * x + b * y + c = 0)
  ∧ (a = 1) ∧ (b = -2) ∧ (c = 5) := sorry

theorem longest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = -2 * x → a * x + b * y + c = 0)
  ∧ (a = 2) ∧ (b = 1) ∧ (c = 0) := sorry

end shortest_chord_line_through_P_longest_chord_line_through_P_l715_71580


namespace percentage_rotten_bananas_l715_71511

theorem percentage_rotten_bananas :
  let total_oranges := 600
  let total_bananas := 400
  let rotten_oranges_percentage := 0.15
  let good_condition_percentage := 0.878
  let total_fruits := total_oranges + total_bananas 
  let rotten_oranges := rotten_oranges_percentage * total_oranges 
  let good_fruits := good_condition_percentage * total_fruits
  let rotten_fruits := total_fruits - good_fruits
  let rotten_bananas := rotten_fruits - rotten_oranges
  (rotten_bananas / total_bananas) * 100 = 8 := by
  {
    -- Calculations and simplifications go here
    sorry
  }

end percentage_rotten_bananas_l715_71511


namespace total_spokes_is_60_l715_71524

def num_spokes_front : ℕ := 20
def num_spokes_back : ℕ := 2 * num_spokes_front
def total_spokes : ℕ := num_spokes_front + num_spokes_back

theorem total_spokes_is_60 : total_spokes = 60 :=
by
  sorry

end total_spokes_is_60_l715_71524


namespace cost_one_dozen_pens_l715_71568

variable (cost_of_pen cost_of_pencil : ℝ)
variable (ratio : ℝ)
variable (dozen_pens_cost : ℝ)

axiom cost_equation : 3 * cost_of_pen + 5 * cost_of_pencil = 200
axiom ratio_pen_pencil : cost_of_pen = 5 * cost_of_pencil

theorem cost_one_dozen_pens : dozen_pens_cost = 12 * cost_of_pen := 
  by
    sorry

end cost_one_dozen_pens_l715_71568


namespace minimum_value_of_f_l715_71545

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ 5/2) ∧ (f 1 = 5/2) := by
  sorry

end minimum_value_of_f_l715_71545


namespace condition_holds_l715_71579

theorem condition_holds 
  (a b c d : ℝ) 
  (h : (a^2 + b^2) / (b^2 + c^2) = (c^2 + d^2) / (d^2 + a^2)) : 
  (a = c ∨ a = -c) ∨ (a^2 - c^2 + d^2 = b^2) :=
by
  sorry

end condition_holds_l715_71579


namespace calculate_annual_rent_l715_71521

-- Defining the conditions
def num_units : ℕ := 100
def occupancy_rate : ℚ := 3 / 4
def monthly_rent : ℚ := 400

-- Defining the target annual rent
def annual_rent (units : ℕ) (occupancy : ℚ) (rent : ℚ) : ℚ :=
  let occupied_units := occupancy * units
  let monthly_revenue := occupied_units * rent
  monthly_revenue * 12

-- Proof problem statement
theorem calculate_annual_rent :
  annual_rent num_units occupancy_rate monthly_rent = 360000 := by
  sorry

end calculate_annual_rent_l715_71521


namespace pencils_loss_equates_20_l715_71560

/--
Patrick purchased 70 pencils and sold them at a loss equal to the selling price of some pencils. The cost of 70 pencils is 1.2857142857142856 times the selling price of 70 pencils. Prove that the loss equates to the selling price of 20 pencils.
-/
theorem pencils_loss_equates_20 
  (C S : ℝ) 
  (h1 : C = 1.2857142857142856 * S) :
  (70 * C - 70 * S) = 20 * S :=
by
  sorry

end pencils_loss_equates_20_l715_71560


namespace intersect_P_M_l715_71573

def P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x | |x| ≤ 3}

theorem intersect_P_M : (P ∩ M) = {x | 0 ≤ x ∧ x < 3} := by
  sorry

end intersect_P_M_l715_71573


namespace sequence_bound_l715_71503

theorem sequence_bound (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) (h2 : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
by
  sorry

end sequence_bound_l715_71503


namespace negation_universal_proposition_l715_71526

theorem negation_universal_proposition :
  (¬∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end negation_universal_proposition_l715_71526


namespace total_sand_l715_71569

variable (capacity_per_bag : ℕ) (number_of_bags : ℕ)

theorem total_sand (h1 : capacity_per_bag = 65) (h2 : number_of_bags = 12) : capacity_per_bag * number_of_bags = 780 := by
  sorry

end total_sand_l715_71569


namespace exists_arithmetic_seq_perfect_powers_l715_71551

def is_perfect_power (x : ℕ) : Prop := ∃ (a k : ℕ), k > 1 ∧ x = a^k

theorem exists_arithmetic_seq_perfect_powers (n : ℕ) (hn : n > 1) :
  ∃ (a d : ℕ) (seq : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → seq i = a + (i - 1) * d)
  ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → is_perfect_power (seq i)
  ∧ d ≠ 0 :=
sorry

end exists_arithmetic_seq_perfect_powers_l715_71551


namespace sqrt_of_9_l715_71513

theorem sqrt_of_9 (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
by {
  sorry
}

end sqrt_of_9_l715_71513


namespace sum_first_10_mod_8_is_7_l715_71539

-- Define the sum of the first 10 positive integers
def sum_first_10 : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10

-- Define the divisor
def divisor : ℕ := 8

-- Prove that the remainder of the sum of the first 10 positive integers divided by 8 is 7
theorem sum_first_10_mod_8_is_7 : sum_first_10 % divisor = 7 :=
by
  sorry

end sum_first_10_mod_8_is_7_l715_71539


namespace decreasing_power_function_l715_71538

theorem decreasing_power_function (n : ℝ) (f : ℝ → ℝ) 
    (h : ∀ x > 0, f x = (n^2 - n - 1) * x^n) 
    (h_decreasing : ∀ x > 0, f x > f (x + 1)) : n = -1 :=
sorry

end decreasing_power_function_l715_71538


namespace set_intersection_complement_eq_l715_71583

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Complement of B in U
def complement_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- The theorem statement
theorem set_intersection_complement_eq :
  A ∩ complement_B = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_complement_eq_l715_71583


namespace positive_y_equals_32_l715_71532

theorem positive_y_equals_32 (y : ℝ) (h : y^2 = 1024) (hy : 0 < y) : y = 32 :=
sorry

end positive_y_equals_32_l715_71532


namespace parallel_lines_m_eq_one_l715_71529

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, x + (1 + m) * y + (m - 2) = 0 ∧ 2 * m * x + 4 * y + 16 = 0 → m = 1) :=
by
  sorry

end parallel_lines_m_eq_one_l715_71529


namespace gloria_pencils_total_l715_71537

-- Define the number of pencils Gloria initially has.
def pencils_gloria_initial : ℕ := 2

-- Define the number of pencils Lisa initially has.
def pencils_lisa_initial : ℕ := 99

-- Define the final number of pencils Gloria will have after receiving all of Lisa's pencils.
def pencils_gloria_final : ℕ := pencils_gloria_initial + pencils_lisa_initial

-- Prove that the final number of pencils Gloria will have is 101.
theorem gloria_pencils_total : pencils_gloria_final = 101 :=
by sorry

end gloria_pencils_total_l715_71537


namespace starting_number_of_three_squares_less_than_2300_l715_71544

theorem starting_number_of_three_squares_less_than_2300 : 
  ∃ n1 n2 n3 : ℕ, n1 < n2 ∧ n2 < n3 ∧ n3^2 < 2300 ∧ n2^2 < 2300 ∧ n1^2 < 2300 ∧ n3^2 ≥ 2209 ∧ n2^2 ≥ 2116 ∧ n1^2 = 2025 :=
by {
  sorry
}

end starting_number_of_three_squares_less_than_2300_l715_71544


namespace smallest_b_no_inverse_mod75_and_mod90_l715_71523

theorem smallest_b_no_inverse_mod75_and_mod90 :
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, n > 0 → n < b →  ¬ (n.gcd 75 > 1 ∧ n.gcd 90 > 1)) ∧ 
  (b.gcd 75 > 1 ∧ b.gcd 90 > 1) ∧ 
  b = 15 := 
by
  sorry

end smallest_b_no_inverse_mod75_and_mod90_l715_71523


namespace minimum_value_C2_minus_D2_l715_71558

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 11))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 4)) + (Real.sqrt (z + 9))

theorem minimum_value_C2_minus_D2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (C x y z)^2 - (D x y z)^2 ≥ 36 := by
  sorry

end minimum_value_C2_minus_D2_l715_71558


namespace max_m_value_min_value_expression_l715_71542

-- Define the conditions for the inequality where the solution is the entire real line
theorem max_m_value (x m : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
sorry

-- Define the conditions for a, b, c > 0 and their sum equal to 1
-- and prove the minimum value of 4a^2 + 9b^2 + c^2
theorem min_value_expression (a b c : ℝ) (hpos1 : a > 0) (hpos2 : b > 0) (hpos3 : c > 0) (hsum : a + b + c = 1) :
  4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧ (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 → a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
sorry

end max_m_value_min_value_expression_l715_71542


namespace probability_multiple_of_4_l715_71534

theorem probability_multiple_of_4 :
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  prob_end_multiple_of_4 = 7 / 64 :=
by
  let num_cards := 12
  let num_multiple_of_4 := 3
  let prob_start_multiple_of_4 := (num_multiple_of_4 : ℚ) / num_cards
  let prob_RR := (1 / 2 : ℚ) * (1 / 2)
  let prob_L2R := (1 / 4 : ℚ) * (1 / 4)
  let prob_RL := (1 / 2 : ℚ) * (1 / 4)
  let total_prob_stay_multiple_of_4 := prob_RR + prob_L2R + prob_RL
  let prob_end_multiple_of_4 := prob_start_multiple_of_4 * total_prob_stay_multiple_of_4
  have h : prob_end_multiple_of_4 = 7 / 64 := by sorry
  exact h

end probability_multiple_of_4_l715_71534


namespace length_segment_ZZ_l715_71599

variable (Z : ℝ × ℝ) (Z' : ℝ × ℝ)

theorem length_segment_ZZ' 
  (h_Z : Z = (-5, 3)) (h_Z' : Z' = (5, 3)) : 
  dist Z Z' = 10 := by
  sorry

end length_segment_ZZ_l715_71599


namespace value_of_x_l715_71567

theorem value_of_x (b x : ℝ) (h₀ : 1 < b) (h₁ : 0 < x) (h₂ : (2 * x) ^ (Real.logb b 2) - (3 * x) ^ (Real.logb b 3) = 0) : x = 1 / 6 :=
by {
  sorry
}

end value_of_x_l715_71567


namespace hyperbola_center_l715_71594

theorem hyperbola_center (x y : ℝ) :
  ∃ h k : ℝ, (∃ a b : ℝ, a = 9/4 ∧ b = 7/2) ∧ (h, k) = (-2, 3) ∧ 
  (4*x + 8)^2 / 81 - (2*y - 6)^2 / 49 = 1 :=
by
  sorry

end hyperbola_center_l715_71594


namespace evaluate_expression_l715_71512

theorem evaluate_expression : 
  ( (7 : ℝ) ^ (1 / 4) / (7 : ℝ) ^ (1 / 7) ) = 7 ^ (3 / 28) := 
by {
  sorry
}

end evaluate_expression_l715_71512


namespace radius_larger_circle_l715_71549

theorem radius_larger_circle (r : ℝ) (AC BC : ℝ) (h1 : 5 * r = AC / 2) (h2 : 15 = BC) : 
  5 * r = 18.75 :=
by
  sorry

end radius_larger_circle_l715_71549


namespace probability_two_green_apples_l715_71590

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_green_apples :
  ∀ (total_apples green_apples choose_apples : ℕ),
    total_apples = 7 →
    green_apples = 3 →
    choose_apples = 2 →
    (binom green_apples choose_apples : ℝ) / binom total_apples choose_apples = 1 / 7 :=
by
  intro total_apples green_apples choose_apples
  intro h_total h_green h_choose
  rw [h_total, h_green, h_choose]
  -- The proof would go here
  sorry

end probability_two_green_apples_l715_71590


namespace part1_part2_l715_71562

variables {A B C : ℝ} {a b c : ℝ} -- Angles and sides of the triangle
variable (h1 : (a - b + c) * (a - b - c) + a * b = 0)
variable (h2 : b * c * Real.sin C = 3 * c * Real.cos A + 3 * a * Real.cos C)

theorem part1 : c = 2 * Real.sqrt 3 :=
by
  sorry

theorem part2 : 6 < a + b ∧ a + b <= 4 * Real.sqrt 3 :=
by
  sorry

end part1_part2_l715_71562


namespace proof_1_proof_2_l715_71546

-- Definitions of propositions p, q, and r

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (x^2 + (a - 1) * x + a^2 ≤ 0)

def q (a : ℝ) : Prop :=
  2 * a^2 - a > 1

def r (a : ℝ) : Prop :=
  (2 * a - 1) / (a - 2) ≤ 1

-- The given proof problem statement 1
theorem proof_1 (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → (a ∈ Set.Icc (-1) (-1/2) ∪ Set.Ioo (1/3) 1) :=
sorry

-- The given proof problem statement 2
theorem proof_2 (a : ℝ) : ¬ p a → r a :=
sorry

end proof_1_proof_2_l715_71546


namespace cubes_sum_l715_71515

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l715_71515


namespace triangles_in_divided_square_l715_71577

theorem triangles_in_divided_square (V E F : ℕ) 
  (hV : V = 24) 
  (h1 : 3 * F + 1 = 2 * E) 
  (h2 : V - E + F = 2) : F = 43 ∧ (F - 1 = 42) := 
by 
  have hF : F = 43 := sorry
  have hTriangles : F - 1 = 42 := sorry
  exact ⟨hF, hTriangles⟩

end triangles_in_divided_square_l715_71577


namespace linear_inequality_m_eq_zero_l715_71598

theorem linear_inequality_m_eq_zero (m : ℝ) (x : ℝ) : 
  ((m - 2) * x ^ |m - 1| - 3 > 6) → abs (m - 1) = 1 → m ≠ 2 → m = 0 := by
  intros h1 h2 h3
  -- Proof of m = 0 based on given conditions
  sorry

end linear_inequality_m_eq_zero_l715_71598


namespace find_x_l715_71500

variable (x : ℕ)  -- we'll use natural numbers to avoid negative values

-- initial number of children
def initial_children : ℕ := 21

-- number of children who got off
def got_off : ℕ := 10

-- total children after some got on
def total_children : ℕ := 16

-- statement to prove x is the number of children who got on the bus
theorem find_x : initial_children - got_off + x = total_children → x = 5 :=
by
  sorry

end find_x_l715_71500


namespace apple_juice_cost_l715_71502

noncomputable def cost_of_apple_juice (cost_per_orange_juice : ℝ) (total_bottles : ℕ) (total_cost : ℝ) (orange_juice_bottles : ℕ) : ℝ :=
  (total_cost - cost_per_orange_juice * orange_juice_bottles) / (total_bottles - orange_juice_bottles)

theorem apple_juice_cost :
  let cost_per_orange_juice := 0.7
  let total_bottles := 70
  let total_cost := 46.2
  let orange_juice_bottles := 42
  cost_of_apple_juice cost_per_orange_juice total_bottles total_cost orange_juice_bottles = 0.6 := by
    sorry

end apple_juice_cost_l715_71502


namespace carl_cost_l715_71506

theorem carl_cost (property_damage medical_bills : ℝ) (insurance_coverage : ℝ) (carl_coverage : ℝ) (H1 : property_damage = 40000) (H2 : medical_bills = 70000) (H3 : insurance_coverage = 0.80) (H4 : carl_coverage = 0.20) :
  carl_coverage * (property_damage + medical_bills) = 22000 :=
by
  sorry

end carl_cost_l715_71506


namespace coordinates_F_l715_71591

-- Definition of point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Reflection over the y-axis
def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

-- Reflection over the x-axis
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Original point F
def F : Point := { x := 3, y := 3 }

-- First reflection over the y-axis
def F' := reflect_y F

-- Second reflection over the x-axis
def F'' := reflect_x F'

-- Goal: Coordinates of F'' after both reflections
theorem coordinates_F'' : F'' = { x := -3, y := -3 } :=
by
  -- Proof would go here
  sorry

end coordinates_F_l715_71591


namespace smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l715_71550

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - (1 / 2) * Real.cos (2 * x)

theorem smallest_positive_period_and_range :
  (∀ x, f (x + Real.pi) = f x) ∧ (Set.range f = Set.Icc (-3 / 2) (5 / 2)) :=
by
  sorry

theorem sin_2x0_if_zero_of_f (x0 : ℝ) (hx0 : 0 ≤ x0 ∧ x0 ≤ Real.pi / 2)
  (hf : f x0 = 0) : Real.sin (2 * x0) = (Real.sqrt 15 - Real.sqrt 3) / 8 :=
by
  sorry

end smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l715_71550


namespace largest_n_satisfying_expression_l715_71527

theorem largest_n_satisfying_expression :
  ∃ n < 100000, (n - 3)^5 - n^2 + 10 * n - 30 ≡ 0 [MOD 3] ∧ 
  (∀ m, m < 100000 → (m - 3)^5 - m^2 + 10 * m - 30 ≡ 0 [MOD 3] → m ≤ 99998) := sorry

end largest_n_satisfying_expression_l715_71527


namespace general_term_sequence_l715_71584

theorem general_term_sequence (a : ℕ → ℝ) (h₁ : a 1 = 1) (hn : ∀ (n : ℕ), a (n + 1) = (10 + 4 * a n) / (1 + a n)) :
  ∀ n : ℕ, a n = 5 - 7 / (1 + (3 / 4) * (-6)^(n - 1)) := 
sorry

end general_term_sequence_l715_71584


namespace range_of_a_l715_71585

theorem range_of_a (a : ℝ) (in_fourth_quadrant : (a+2 > 0) ∧ (a-3 < 0)) : -2 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l715_71585


namespace perimeter_of_original_square_l715_71530

-- Definitions
variables {x : ℝ}
def rect_width := x
def rect_length := 4 * x
def rect_perimeter := 56
def original_square_perimeter := 32

-- Statement
theorem perimeter_of_original_square (x : ℝ) (h : 28 * x = 56) : 4 * (4 * x) = 32 :=
by
  -- Since the proof is not required, we apply sorry to end the theorem.
  sorry

end perimeter_of_original_square_l715_71530


namespace sum_shade_length_l715_71595

-- Define the arithmetic sequence and the given conditions
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (is_arithmetic : ∀ n, a (n + 1) = a n + d)

-- Define the shadow lengths for each term using the arithmetic progression properties
def shade_length_seq (seq : ArithmeticSequence) : ℕ → ℝ := seq.a

variables (seq : ArithmeticSequence)

-- Given conditions
axiom sum_condition_1 : seq.a 1 + seq.a 4 + seq.a 7 = 31.5
axiom sum_condition_2 : seq.a 2 + seq.a 5 + seq.a 8 = 28.5

-- Question to prove
theorem sum_shade_length : seq.a 3 + seq.a 6 + seq.a 9 = 25.5 :=
by
  -- proof to be filled in later
  sorry

end sum_shade_length_l715_71595


namespace total_square_footage_after_expansion_l715_71540

-- Definitions from the conditions
def size_smaller_house_initial : ℕ := 5200
def size_larger_house : ℕ := 7300
def expansion_smaller_house : ℕ := 3500

-- The new size of the smaller house after expansion
def size_smaller_house_after_expansion : ℕ :=
  size_smaller_house_initial + expansion_smaller_house

-- The new total square footage
def new_total_square_footage : ℕ :=
  size_smaller_house_after_expansion + size_larger_house

-- Goal statement: Prove the total new square footage is 16000 sq. ft.
theorem total_square_footage_after_expansion : new_total_square_footage = 16000 := by
  sorry

end total_square_footage_after_expansion_l715_71540


namespace set_union_covers_real_line_l715_71505

open Set

def M := {x : ℝ | x < 0 ∨ 2 < x}
def N := {x : ℝ | -Real.sqrt 5 < x ∧ x < Real.sqrt 5}

theorem set_union_covers_real_line : M ∪ N = univ := sorry

end set_union_covers_real_line_l715_71505


namespace union_intersection_l715_71504

-- Define the sets M, N, and P
def M := ({1} : Set Nat)
def N := ({1, 2} : Set Nat)
def P := ({1, 2, 3} : Set Nat)

-- Prove that (M ∪ N) ∩ P = {1, 2}
theorem union_intersection : (M ∪ N) ∩ P = ({1, 2} : Set Nat) := 
by 
  sorry

end union_intersection_l715_71504


namespace refill_cost_calculation_l715_71548

variables (total_spent : ℕ) (refills : ℕ)

def one_refill_cost (total_spent refills : ℕ) : ℕ := total_spent / refills

theorem refill_cost_calculation (h1 : total_spent = 40) (h2 : refills = 4) :
  one_refill_cost total_spent refills = 10 :=
by
  sorry

end refill_cost_calculation_l715_71548


namespace num_ordered_pairs_no_real_solution_l715_71575

theorem num_ordered_pairs_no_real_solution : 
  {n : ℕ // ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 - 4*c < 0 ∨ c^2 - 4*b < 0) ∧ n = 6 } := by
sorry

end num_ordered_pairs_no_real_solution_l715_71575


namespace ratio_proof_l715_71587

theorem ratio_proof (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 := by
  sorry

end ratio_proof_l715_71587


namespace cream_cheese_cost_l715_71596

theorem cream_cheese_cost
  (B C : ℝ)
  (h1 : 2 * B + 3 * C = 12)
  (h2 : 4 * B + 2 * C = 14) :
  C = 2.5 :=
by
  sorry

end cream_cheese_cost_l715_71596


namespace range_h_l715_71561

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 5 * x^2)

theorem range_h (a b : ℝ) (h_range : Set.Ioo a b = Set.Icc 0 1) : a + b = 1 := by
  sorry

end range_h_l715_71561


namespace special_divisors_count_of_20_30_l715_71554

def prime_number (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def number_of_divisors (a : ℕ) (α β : ℕ) : ℕ := (α + 1) * (β + 1)

def count_special_divisors (m n : ℕ) : ℕ :=
  let total_divisors_m := (m + 1) * (n + 1)
  let total_divisors_n := (n + 1) * (n / 2 + 1)
  (total_divisors_m - 1) / 2 - total_divisors_n + 1

theorem special_divisors_count_of_20_30 (d_20_30 d_20_15 : ℕ) :
  let α := 60
  let β := 30
  let γ := 30
  let δ := 15
  prime_number 2 ∧ prime_number 5 ∧
  count_special_divisors α β = 1891 ∧
  count_special_divisors γ δ = 496 →
  d_20_30 = 2 * 1891 / 2 ∧
  d_20_15 = 2 * 496 →
  count_special_divisors 60 30 - count_special_divisors 30 15 + 1 = 450
:= by
  sorry

end special_divisors_count_of_20_30_l715_71554


namespace complement_of_A_in_U_l715_71592

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def complement_U_A : Set ℝ := {x | x ≤ 1 ∨ x > 3}

theorem complement_of_A_in_U : (U \ A) = complement_U_A := by
  simp only [U, A, complement_U_A]
  sorry

end complement_of_A_in_U_l715_71592


namespace shorter_side_of_room_l715_71501

theorem shorter_side_of_room
  (P : ℕ) (A : ℕ) (a b : ℕ)
  (perimeter_eq : 2 * a + 2 * b = P)
  (area_eq : a * b = A) (partition_len : ℕ) (partition_cond : partition_len = 5)
  (room_perimeter : P = 60)
  (room_area : A = 200) :
  b = 10 := 
by
  sorry

end shorter_side_of_room_l715_71501


namespace find_xy_l715_71581

theorem find_xy (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end find_xy_l715_71581


namespace four_digit_multiples_of_7_l715_71566

theorem four_digit_multiples_of_7 : 
  let smallest_four_digit := 1000
  let largest_four_digit := 9999
  let smallest_multiple_of_7 := (Nat.ceil (smallest_four_digit / 7)) * 7
  let largest_multiple_of_7 := (Nat.floor (largest_four_digit / 7)) * 7
  let count_of_multiples := (Nat.floor (largest_four_digit / 7)) - (Nat.ceil (smallest_four_digit / 7)) + 1
  count_of_multiples = 1286 :=
by
  sorry

end four_digit_multiples_of_7_l715_71566


namespace colored_shirts_count_l715_71557

theorem colored_shirts_count (n : ℕ) (h1 : 6 = 6) (h2 : (1 / (n : ℝ)) ^ 6 = 1 / 120) : n = 2 := 
sorry

end colored_shirts_count_l715_71557


namespace min_value_arith_seq_l715_71519

theorem min_value_arith_seq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + c = 2 * b) :
  (a + c) / b + b / (a + c) ≥ 5 / 2 := 
sorry

end min_value_arith_seq_l715_71519


namespace compare_neg_fractions_l715_71593

theorem compare_neg_fractions : (- (3 / 2) < -1) :=
by sorry

end compare_neg_fractions_l715_71593


namespace part1_part2_l715_71535

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def set_B : Set ℝ := {x : ℝ | x < -1 ∨ x > 1}

theorem part1 (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a > 3) :=
by sorry

theorem part2 (a : ℝ) : (set_A a ∪ set_B = Set.univ) ↔ (-2 ≤ a ∧ a ≤ -1 / 2) :=
by sorry

end part1_part2_l715_71535


namespace minimize_surface_area_l715_71508

-- Define the problem conditions
def volume (x y : ℝ) : ℝ := 2 * x^2 * y
def surface_area (x y : ℝ) : ℝ := 2 * (2 * x^2 + 2 * x * y + x * y)

theorem minimize_surface_area :
  ∃ (y : ℝ), 
  (∀ (x : ℝ), volume x y = 72) → 
  1 * 2 * y = 4 :=
by
  sorry

end minimize_surface_area_l715_71508


namespace celine_smartphones_l715_71588

-- Definitions based on the conditions
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops_bought : ℕ := 2
def initial_amount : ℕ := 3000
def change_received : ℕ := 200

-- The proof goal is to show that the number of smartphones bought is 4
theorem celine_smartphones (laptop_cost smartphone_cost num_laptops_bought initial_amount change_received : ℕ)
  (h1 : laptop_cost = 600)
  (h2 : smartphone_cost = 400)
  (h3 : num_laptops_bought = 2)
  (h4 : initial_amount = 3000)
  (h5 : change_received = 200) :
  (initial_amount - change_received - num_laptops_bought * laptop_cost) / smartphone_cost = 4 := 
by
  sorry

end celine_smartphones_l715_71588


namespace rhombus_area_l715_71564

theorem rhombus_area 
  (a : ℝ) (d1 d2 : ℝ)
  (h_side : a = Real.sqrt 113)
  (h_diagonal_diff : abs (d1 - d2) = 8)
  (h_geq : d1 ≠ d2) : 
  (a^2 * d1 * d2 / 2 = 194) :=
sorry -- Proof to be completed

end rhombus_area_l715_71564


namespace tea_bags_count_l715_71556

-- Definitions based on the given problem
def valid_bags (b : ℕ) : Prop :=
  ∃ (a c d : ℕ), a + b - a = b ∧ c + d = b ∧ 3 * c + 2 * d = 41 ∧ 3 * a + 2 * (b - a) = 58

-- Statement of the problem, confirming the proof condition
theorem tea_bags_count (b : ℕ) : valid_bags b ↔ b = 20 :=
by {
  -- The proof is left for completion
  sorry
}

end tea_bags_count_l715_71556


namespace commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l715_71552

def binary_star (x y : ℝ) : ℝ := (x - 1) * (y - 1) - 1

-- Statement (A): Commutativity
theorem commutative_star (x y : ℝ) : binary_star x y = binary_star y x := sorry

-- Statement (B): Distributivity (proving it's not distributive)
theorem not_distributive_star (x y z : ℝ) : ¬(binary_star x (y + z) = binary_star x y + binary_star x z) := sorry

-- Statement (C): Special case
theorem special_case_star (x : ℝ) : binary_star (x + 1) (x - 1) = binary_star x x - 1 := sorry

-- Statement (D): Identity element
theorem no_identity_star (x e : ℝ) : ¬(binary_star x e = x ∧ binary_star e x = x) := sorry

-- Statement (E): Associativity (proving it's not associative)
theorem not_associative_star (x y z : ℝ) : ¬(binary_star x (binary_star y z) = binary_star (binary_star x y) z) := sorry

end commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l715_71552


namespace tom_remaining_balloons_l715_71528

theorem tom_remaining_balloons (initial_balloons : ℕ) (balloons_given : ℕ) (balloons_remaining : ℕ) 
  (h1 : initial_balloons = 30) (h2 : balloons_given = 16) : balloons_remaining = 14 := 
by
  sorry

end tom_remaining_balloons_l715_71528


namespace general_formula_a_n_sum_first_n_terms_T_n_l715_71570

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Condition: S_n = 2a_n - 3
axiom condition_S (n : ℕ) : S_n n = 2 * (a_n n) - 3

-- (I) General formula for a_n
theorem general_formula_a_n (n : ℕ) : a_n n = 3 * 2^(n - 1) := 
sorry

-- (II) General formula for T_n
theorem sum_first_n_terms_T_n (n : ℕ) : T_n n = 3 * (n - 1) * 2^n + 3 := 
sorry

end general_formula_a_n_sum_first_n_terms_T_n_l715_71570


namespace candy_bar_cost_l715_71547

/-- Problem statement:
Todd had 85 cents and spent 53 cents in total on a candy bar and a box of cookies.
The box of cookies cost 39 cents. How much did the candy bar cost? --/
theorem candy_bar_cost (t c s b : ℕ) (ht : t = 85) (hc : c = 39) (hs : s = 53) (h_total : s = b + c) : b = 14 :=
by
  sorry

end candy_bar_cost_l715_71547


namespace susannah_swims_more_than_camden_l715_71516

-- Define the given conditions
def camden_total_swims : ℕ := 16
def susannah_total_swims : ℕ := 24
def number_of_weeks : ℕ := 4

-- State the theorem
theorem susannah_swims_more_than_camden :
  (susannah_total_swims / number_of_weeks) - (camden_total_swims / number_of_weeks) = 2 :=
by
  sorry

end susannah_swims_more_than_camden_l715_71516


namespace slope_of_line_through_points_l715_71582

theorem slope_of_line_through_points 
  (t : ℝ) 
  (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 12 * t + 6) 
  (h2 : 2 * x + 3 * y = 8 * t - 1) : 
  ∃ m b : ℝ, (∀ t : ℝ, y = m * x + b) ∧ m = 0 :=
by 
  sorry

end slope_of_line_through_points_l715_71582


namespace total_lunch_cost_l715_71522

/-- Janet, a third grade teacher, is picking up the sack lunch order from a local deli for 
the field trip she is taking her class on. There are 35 children in her class, 5 volunteer 
chaperones, and herself. She also ordered three additional sack lunches, just in case 
there was a problem. Each sack lunch costs $7. --/
theorem total_lunch_cost :
  let children := 35
  let chaperones := 5
  let janet := 1
  let additional_lunches := 3
  let price_per_lunch := 7
  let total_lunches := children + chaperones + janet + additional_lunches
  total_lunches * price_per_lunch = 308 :=
by
  sorry

end total_lunch_cost_l715_71522


namespace largest_four_digit_negative_integer_congruent_to_2_mod_17_l715_71520

theorem largest_four_digit_negative_integer_congruent_to_2_mod_17 :
  ∃ (n : ℤ), (n % 17 = 2 ∧ n > -10000 ∧ n < -999) ∧ ∀ m : ℤ, (m % 17 = 2 ∧ m > -10000 ∧ m < -999) → m ≤ n :=
sorry

end largest_four_digit_negative_integer_congruent_to_2_mod_17_l715_71520
