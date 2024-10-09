import Mathlib

namespace complement_intersection_l1377_137745

def U : Set ℤ := Set.univ
def A : Set ℤ := {1, 2}
def B : Set ℤ := {3, 4}

-- A ∪ B should equal {1, 2, 3, 4}
axiom AUeq : A ∪ B = {1, 2, 3, 4}

theorem complement_intersection : (U \ A) ∩ B = {3, 4} :=
by
  sorry

end complement_intersection_l1377_137745


namespace smallest_common_term_larger_than_2023_l1377_137708

noncomputable def a_seq (n : ℕ) : ℤ :=
  3 * n - 2

noncomputable def b_seq (m : ℕ) : ℤ :=
  10 * m - 8

theorem smallest_common_term_larger_than_2023 :
  ∃ (n m : ℕ), a_seq n = b_seq m ∧ a_seq n > 2023 ∧ a_seq n = 2032 :=
by {
  sorry
}

end smallest_common_term_larger_than_2023_l1377_137708


namespace paul_peaches_l1377_137787

theorem paul_peaches (P : ℕ) (h1 : 26 - P = 22) : P = 4 :=
by {
  sorry
}

end paul_peaches_l1377_137787


namespace problem_statement_l1377_137722

variables {a b c : ℝ}

theorem problem_statement 
  (h1 : a^2 + a * b + b^2 = 9)
  (h2 : b^2 + b * c + c^2 = 52)
  (h3 : c^2 + c * a + a^2 = 49) : 
  (49 * b^2 - 33 * b * c + 9 * c^2) / a^2 = 52 :=
by
  sorry

end problem_statement_l1377_137722


namespace length_is_62_l1377_137733

noncomputable def length_of_plot (b : ℝ) := b + 24

theorem length_is_62 (b : ℝ) (h1 : length_of_plot b = b + 24) 
  (h2 : 2 * (length_of_plot b + b) = 200) : 
  length_of_plot b = 62 :=
by sorry

end length_is_62_l1377_137733


namespace total_pens_l1377_137747

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l1377_137747


namespace table_height_l1377_137714

theorem table_height (l w h : ℝ) (h1 : l + h - w = 38) (h2 : w + h - l = 34) : h = 36 :=
by
  sorry

end table_height_l1377_137714


namespace find_m_l1377_137776

theorem find_m (x1 x2 m : ℝ) (h1 : 2 * x1^2 - 3 * x1 + m = 0) (h2 : 2 * x2^2 - 3 * x2 + m = 0) (h3 : 8 * x1 - 2 * x2 = 7) :
  m = 1 :=
sorry

end find_m_l1377_137776


namespace ratio_implies_sum_ratio_l1377_137724

theorem ratio_implies_sum_ratio (x y : ℝ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_implies_sum_ratio_l1377_137724


namespace tom_balloons_count_l1377_137741

-- Define the number of balloons Tom initially has
def balloons_initial : Nat := 30

-- Define the number of balloons Tom gave away
def balloons_given : Nat := 16

-- Define the number of balloons Tom now has
def balloons_remaining : Nat := balloons_initial - balloons_given

theorem tom_balloons_count :
  balloons_remaining = 14 := by
  sorry

end tom_balloons_count_l1377_137741


namespace premium_rate_l1377_137768

theorem premium_rate (P : ℝ) : (14400 / (100 + P)) * 5 = 600 → P = 20 :=
by
  intro h
  sorry

end premium_rate_l1377_137768


namespace find_q_l1377_137785

def polynomial_q (x p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h₀ : r = 3)
  (h₁ : (-p / 3) = -r)
  (h₂ : (-r) = 1 + p + q + r) :
  q = -16 :=
by
  -- h₀ implies r = 3
  -- h₁ becomes (-p / 3) = -3
  -- which results in p = 9
  -- h₂ becomes -3 = 1 + 9 + q + 3
  -- leading to q = -16
  sorry

end find_q_l1377_137785


namespace sister_weight_difference_is_12_l1377_137746

-- Define Antonio's weight
def antonio_weight : ℕ := 50

-- Define the combined weight of Antonio and his sister
def combined_weight : ℕ := 88

-- Define the weight of Antonio's sister
def sister_weight : ℕ := combined_weight - antonio_weight

-- Define the weight difference
def weight_difference : ℕ := antonio_weight - sister_weight

-- Theorem statement to prove the weight difference is 12 kg
theorem sister_weight_difference_is_12 : weight_difference = 12 := by
  sorry

end sister_weight_difference_is_12_l1377_137746


namespace polly_breakfast_minutes_l1377_137717
open Nat

theorem polly_breakfast_minutes (B : ℕ) 
  (lunch_minutes : ℕ)
  (dinner_4_days_minutes : ℕ)
  (dinner_3_days_minutes : ℕ)
  (total_minutes : ℕ)
  (h1 : lunch_minutes = 5 * 7)
  (h2 : dinner_4_days_minutes = 10 * 4)
  (h3 : dinner_3_days_minutes = 30 * 3)
  (h4 : total_minutes = 305) 
  (h5 : 7 * B + lunch_minutes + dinner_4_days_minutes + dinner_3_days_minutes = total_minutes) :
  B = 20 :=
by
  -- proof omitted
  sorry

end polly_breakfast_minutes_l1377_137717


namespace Yasmin_children_count_l1377_137719

theorem Yasmin_children_count (Y : ℕ) (h1 : 2 * Y + Y = 6) : Y = 2 :=
by
  sorry

end Yasmin_children_count_l1377_137719


namespace tangent_line_circle_l1377_137788

theorem tangent_line_circle : 
  ∃ (k : ℚ), (∀ x y : ℚ, ((x - 3) ^ 2 + (y - 4) ^ 2 = 25) 
               → (3 * x + 4 * y - 25 = 0)) :=
sorry

end tangent_line_circle_l1377_137788


namespace quadratic_inequality_solution_l1377_137777

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l1377_137777


namespace equal_real_roots_eq_one_l1377_137779

theorem equal_real_roots_eq_one (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y * y = x) ∧ (∀ x y : ℝ, x^2 - 2 * x + m = 0 ↔ (x = y) → b^2 - 4 * a * c = 0) → m = 1 := 
sorry

end equal_real_roots_eq_one_l1377_137779


namespace students_in_class_l1377_137784

theorem students_in_class (total_spent: ℝ) (packs_per_student: ℝ) (sausages_per_student: ℝ) (cost_pack_noodles: ℝ) (cost_sausage: ℝ) (cost_per_student: ℝ) (num_students: ℝ):
  total_spent = 290 → 
  packs_per_student = 2 → 
  sausages_per_student = 1 → 
  cost_pack_noodles = 3.5 → 
  cost_sausage = 7.5 → 
  cost_per_student = packs_per_student * cost_pack_noodles + sausages_per_student * cost_sausage →
  total_spent = cost_per_student * num_students →
  num_students = 20 := 
by
  sorry

end students_in_class_l1377_137784


namespace find_a_plus_b_l1377_137736

theorem find_a_plus_b :
  let A := {x : ℝ | -1 < x ∧ x < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  let S := {x : ℝ | -1 < x ∧ x < 2}
  ∃ (a b : ℝ), (∀ x, S x ↔ (x^2 + a * x + b < 0)) ∧ a + b = -3 :=
by
  sorry

end find_a_plus_b_l1377_137736


namespace number_of_shelves_l1377_137728

-- Define the initial conditions and required values
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Define the result we want to prove
theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 4 :=
by
    -- Proof steps go here
    sorry

end number_of_shelves_l1377_137728


namespace total_cost_six_years_l1377_137754

variable {fees : ℕ → ℝ}

-- Conditions
def fee_first_year : fees 1 = 80 := sorry

def fee_increase (n : ℕ) : fees (n + 1) = fees n + (10 + 2 * (n - 1)) := 
sorry

-- Proof problem: Prove that the total cost is 670
theorem total_cost_six_years : (fees 1 + fees 2 + fees 3 + fees 4 + fees 5 + fees 6) = 670 :=
by sorry

end total_cost_six_years_l1377_137754


namespace alice_paid_percentage_of_srp_l1377_137778

theorem alice_paid_percentage_of_srp
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ := P * 0.60) -- Marked Price (MP) is 40% less than SRP
  (price_alice_paid : ℝ := MP * 0.60) -- Alice purchased the book for 40% off the marked price
  : (price_alice_paid / P) * 100 = 36 :=
by
  -- only the statement is required, so proof is omitted
  sorry

end alice_paid_percentage_of_srp_l1377_137778


namespace graphs_relative_position_and_intersection_l1377_137762

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 5

theorem graphs_relative_position_and_intersection :
  (1 > -1.5) ∧ ( ∃ y, f 0 = y ∧ g 0 = y ) ∧ f 0 = 5 :=
by
  -- sorry to skip the proof
  sorry

end graphs_relative_position_and_intersection_l1377_137762


namespace minimum_distance_focus_to_circle_point_l1377_137782

def focus_of_parabola : ℝ × ℝ := (1, 0)
def center_of_circle : ℝ × ℝ := (4, 4)
def radius_of_circle : ℝ := 4
def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16

theorem minimum_distance_focus_to_circle_point :
  ∃ P : ℝ × ℝ, circle_equation P.1 P.2 ∧ dist focus_of_parabola P = 5 :=
sorry

end minimum_distance_focus_to_circle_point_l1377_137782


namespace quadratic_eq_solution_1_quadratic_eq_solution_2_l1377_137731

theorem quadratic_eq_solution_1 :
    ∀ (x : ℝ), x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by 
  sorry

theorem quadratic_eq_solution_2 :
    ∀ (x : ℝ), x * (x - 2) - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
by 
  sorry

end quadratic_eq_solution_1_quadratic_eq_solution_2_l1377_137731


namespace fly_distance_from_ceiling_l1377_137704

theorem fly_distance_from_ceiling (x y z : ℝ) (hx : x = 2) (hy : y = 6) (hP : x^2 + y^2 + z^2 = 100) : z = 2 * Real.sqrt 15 :=
by
  sorry

end fly_distance_from_ceiling_l1377_137704


namespace grandchildren_ages_l1377_137796

theorem grandchildren_ages (x : ℕ) (y : ℕ) :
  (x + y = 30) →
  (5 * (x * (x + 1) + (30 - x) * (31 - x)) = 2410) →
  (x = 16 ∧ y = 14) ∨ (x = 14 ∧ y = 16) :=
by
  sorry

end grandchildren_ages_l1377_137796


namespace find_a_l1377_137765

-- Define the quadratic equation with the root condition
def quadratic_with_root_zero (a : ℝ) : Prop :=
  (a - 1) * 0^2 + 0 + a - 2 = 0

-- State the theorem to be proved
theorem find_a (a : ℝ) (h : quadratic_with_root_zero a) : a = 2 :=
by
  -- Statement placeholder, proof omitted
  sorry

end find_a_l1377_137765


namespace no_zero_terms_in_arithmetic_progression_l1377_137795

theorem no_zero_terms_in_arithmetic_progression (a d : ℤ) (h : ∃ (n : ℕ), 2 * a + (2 * n - 1) * d = ((3 * n - 1) * (2 * a + (3 * n - 2) * d)) / 2) :
  ∀ (m : ℕ), a + (m - 1) * d ≠ 0 :=
by
  sorry

end no_zero_terms_in_arithmetic_progression_l1377_137795


namespace alice_prob_after_three_turns_l1377_137727

/-
Definition of conditions:
 - Alice starts with the ball.
 - If Alice has the ball, there is a 1/3 chance that she will toss it to Bob and a 2/3 chance that she will keep the ball.
 - If Bob has the ball, there is a 1/4 chance that he will toss it to Alice and a 3/4 chance that he keeps the ball.
-/

def alice_to_bob : ℚ := 1/3
def alice_keeps : ℚ := 2/3
def bob_to_alice : ℚ := 1/4
def bob_keeps : ℚ := 3/4

theorem alice_prob_after_three_turns :
  alice_to_bob * bob_keeps * bob_to_alice +
  alice_keeps * alice_keeps * alice_keeps +
  alice_to_bob * bob_to_alice * alice_keeps = 179/432 :=
by
  sorry

end alice_prob_after_three_turns_l1377_137727


namespace probability_same_color_set_l1377_137763

theorem probability_same_color_set 
  (black_pairs blue_pairs : ℕ)
  (green_pairs : {g : Finset (ℕ × ℕ) // g.card = 3})
  (total_pairs := 15)
  (total_shoes := total_pairs * 2) :
  2 * black_pairs + 2 * blue_pairs + green_pairs.val.card * 2 = total_shoes →
  ∃ probability : ℚ, 
    probability = 89 / 435 :=
by
  intro h_total_shoes
  let black_shoes := black_pairs * 2
  let blue_shoes := blue_pairs * 2
  let green_shoes := green_pairs.val.card * 2
  
  have h_black_probability : ℚ := (black_shoes / total_shoes) * (black_pairs / (total_shoes - 1))
  have h_blue_probability : ℚ := (blue_shoes / total_shoes) * (blue_pairs / (total_shoes - 1))
  have h_green_probability : ℚ := (green_shoes / total_shoes) * (green_pairs.val.card / (total_shoes - 1))
  
  have h_total_probability : ℚ := h_black_probability + h_blue_probability + h_green_probability
  
  use h_total_probability
  sorry

end probability_same_color_set_l1377_137763


namespace negation_of_proposition_l1377_137740

-- Definitions of the conditions
variables (a b c : ℝ) 

-- Prove the mathematically equivalent statement:
theorem negation_of_proposition :
  (a + b + c ≠ 1) → (a^2 + b^2 + c^2 > 1 / 9) :=
sorry

end negation_of_proposition_l1377_137740


namespace correct_propositions_l1377_137786

variables (a b : ℝ) (x : ℝ) (a_max : ℝ)

/-- Given propositions to analyze. -/
noncomputable def propositions :=
  ((a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3) ∧
  ((¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ∧
  (a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max))

/-- The main theorem stating which propositions are correct -/
theorem correct_propositions (h1 : a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3)
                            (h2 : (¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0)
                            (h3 : a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max) :
  propositions a b a_max :=
by
  sorry

end correct_propositions_l1377_137786


namespace combined_salaries_of_A_B_C_E_is_correct_l1377_137734

-- Given conditions
def D_salary : ℕ := 7000
def average_salary : ℕ := 8800
def n_individuals : ℕ := 5

-- Combined salary of A, B, C, and E
def combined_salaries : ℕ := 37000

theorem combined_salaries_of_A_B_C_E_is_correct :
  (average_salary * n_individuals - D_salary) = combined_salaries :=
by
  sorry

end combined_salaries_of_A_B_C_E_is_correct_l1377_137734


namespace zack_initial_marbles_l1377_137774

theorem zack_initial_marbles :
  let a1 := 20
  let a2 := 30
  let a3 := 35
  let a4 := 25
  let a5 := 28
  let a6 := 40
  let r := 7
  let T := a1 + a2 + a3 + a4 + a5 + a6 + r
  T = 185 :=
by
  sorry

end zack_initial_marbles_l1377_137774


namespace current_speed_correct_l1377_137744

noncomputable def boat_upstream_speed : ℝ := (1 / 20) * 60
noncomputable def boat_downstream_speed : ℝ := (1 / 9) * 60
noncomputable def speed_of_current : ℝ := (boat_downstream_speed - boat_upstream_speed) / 2

theorem current_speed_correct :
  speed_of_current = 1.835 :=
by
  sorry

end current_speed_correct_l1377_137744


namespace polygon_angle_arithmetic_progression_l1377_137710

theorem polygon_angle_arithmetic_progression
  (h1 : ∀ {n : ℕ}, n ≥ 3)   -- The polygon is convex and n-sided
  (h2 : ∀ (angles : Fin n → ℝ), (∀ i j, i < j → angles i + 5 = angles j))   -- The interior angles form an arithmetic progression with a common difference of 5°
  (h3 : ∀ (angles : Fin n → ℝ), (∃ i, angles i = 160))  -- The largest angle is 160°
  : n = 9 := sorry

end polygon_angle_arithmetic_progression_l1377_137710


namespace lcm_of_numbers_is_750_l1377_137703

-- Define the two numbers x and y
variables (x y : ℕ)

-- Given conditions as hypotheses
def product_of_numbers := 18750
def hcf_of_numbers := 25

-- The proof problem statement
theorem lcm_of_numbers_is_750 (h_product : x * y = product_of_numbers) 
                              (h_hcf : Nat.gcd x y = hcf_of_numbers) : Nat.lcm x y = 750 :=
by
  sorry

end lcm_of_numbers_is_750_l1377_137703


namespace seq_is_geometric_from_second_l1377_137709

namespace sequence_problem

-- Definitions and conditions
def S : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| (n + 1) => 3 * S n - 2 * S (n - 1)

-- Recursive definition for sum of sequence terms
axiom S_rec_relation (n : ℕ) (h : n ≥ 2) : 
  S (n + 1) - 3 * S n + 2 * S (n - 1) = 0

-- Prove the sequence is geometric from the second term
theorem seq_is_geometric_from_second :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 2, a (n + 1) = 2 * a n) ∧ 
  (a 1 = 1) ∧ 
  (a 2 = 1) :=
by
  sorry

end sequence_problem

end seq_is_geometric_from_second_l1377_137709


namespace no_four_distinct_real_roots_l1377_137742

theorem no_four_distinct_real_roots (a b : ℝ) : ¬ (∃ (x1 x2 x3 x4 : ℝ), 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧ 
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧ 
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧ 
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0)) :=
by
  sorry

end no_four_distinct_real_roots_l1377_137742


namespace triangle_equilateral_from_condition_l1377_137780

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
a = b ∧ b = c

theorem triangle_equilateral_from_condition (a b c h_a h_b h_c : ℝ)
  (h : a + h_a = b + h_b ∧ b + h_b = c + h_c) :
  is_equilateral a b c :=
sorry

end triangle_equilateral_from_condition_l1377_137780


namespace jason_initial_cards_l1377_137766

-- Conditions
def cards_given_away : ℕ := 9
def cards_left : ℕ := 4

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 13 :=
by
  sorry

end jason_initial_cards_l1377_137766


namespace circle_radius_l1377_137794

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 120 * π) : r = 10 :=
sorry

end circle_radius_l1377_137794


namespace find_w_l1377_137771

theorem find_w (k : ℝ) (h1 : z * Real.sqrt w = k)
  (z_w3 : z = 6) (w3 : w = 3) :
  z = 3 / 2 → w = 48 := sorry

end find_w_l1377_137771


namespace necessary_not_sufficient_condition_l1377_137758

theorem necessary_not_sufficient_condition (x : ℝ) :
  ((-6 ≤ x ∧ x ≤ 3) → (-5 ≤ x ∧ x ≤ 3)) ∧
  (¬ ((-5 ≤ x ∧ x ≤ 3) → (-6 ≤ x ∧ x ≤ 3))) :=
by
  -- Need proof steps here
  sorry

end necessary_not_sufficient_condition_l1377_137758


namespace negate_prop_l1377_137713

theorem negate_prop :
  ¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
by
  sorry

end negate_prop_l1377_137713


namespace smallest_d_l1377_137700

theorem smallest_d (d : ℕ) (h_pos : 0 < d) (h_square : ∃ k : ℕ, 3150 * d = k^2) : d = 14 :=
sorry

end smallest_d_l1377_137700


namespace geometric_progression_common_ratio_l1377_137761

theorem geometric_progression_common_ratio (a r : ℝ) (h_pos : a > 0)
  (h_eq : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) :
  r = 1/2 :=
sorry

end geometric_progression_common_ratio_l1377_137761


namespace frankie_pets_l1377_137729

variable {C S P D : ℕ}

theorem frankie_pets (h1 : S = C + 6) (h2 : P = C - 1) (h3 : C + D = 6) (h4 : C + S + P + D = 19) : 
  C + S + P + D = 19 :=
  by sorry

end frankie_pets_l1377_137729


namespace distinct_sequences_ten_flips_l1377_137716

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l1377_137716


namespace terminating_decimal_l1377_137715

theorem terminating_decimal : (47 : ℚ) / (2 * 5^4) = 376 / 10^4 :=
by sorry

end terminating_decimal_l1377_137715


namespace find_multiple_l1377_137726

theorem find_multiple (x m : ℝ) (h₁ : 10 * x = m * x - 36) (h₂ : x = -4.5) : m = 2 :=
by
  sorry

end find_multiple_l1377_137726


namespace time_spent_on_marketing_posts_l1377_137772

-- Bryan's conditions
def hours_customer_outreach : ℕ := 4
def hours_advertisement : ℕ := hours_customer_outreach / 2
def total_hours_worked : ℕ := 8

-- Proof statement: Bryan spends 2 hours each day on marketing posts
theorem time_spent_on_marketing_posts : 
  total_hours_worked - (hours_customer_outreach + hours_advertisement) = 2 := by
  sorry

end time_spent_on_marketing_posts_l1377_137772


namespace value_of_x_l1377_137725

theorem value_of_x (x : ℝ) (hx_pos : 0 < x) (hx_eq : x^2 = 1024) : x = 32 := 
by
  sorry

end value_of_x_l1377_137725


namespace min_value_abs_ab_l1377_137759

theorem min_value_abs_ab (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) 
(h_perpendicular : - 1 / (a^2) * (a^2 + 1) / b = -1) :
|a * b| = 2 :=
sorry

end min_value_abs_ab_l1377_137759


namespace calculate_paintable_area_l1377_137705

noncomputable def bedroom_length : ℝ := 15
noncomputable def bedroom_width : ℝ := 11
noncomputable def bedroom_height : ℝ := 9
noncomputable def door_window_area : ℝ := 70
noncomputable def num_bedrooms : ℝ := 3

theorem calculate_paintable_area :
  (num_bedrooms * ((2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height) - door_window_area)) = 1194 := 
by
  -- conditions as definitions
  let total_wall_area := (2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height)
  let paintable_wall_in_bedroom := total_wall_area - door_window_area
  let total_paintable_area := num_bedrooms * paintable_wall_in_bedroom
  show total_paintable_area = 1194
  sorry

end calculate_paintable_area_l1377_137705


namespace sufficient_but_not_necessary_condition_for_q_l1377_137721

variable (p q r : Prop)

theorem sufficient_but_not_necessary_condition_for_q (hp : p → r) (hq1 : r → q) (hq2 : ¬(q → r)) : 
  (p → q) ∧ ¬(q → p) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_q_l1377_137721


namespace total_shoes_l1377_137769

theorem total_shoes (Brian_shoes : ℕ) (Edward_shoes : ℕ) (Jacob_shoes : ℕ)
  (hBrian : Brian_shoes = 22)
  (hEdward : Edward_shoes = 3 * Brian_shoes)
  (hJacob : Jacob_shoes = Edward_shoes / 2) :
  Brian_shoes + Edward_shoes + Jacob_shoes = 121 :=
by 
  sorry

end total_shoes_l1377_137769


namespace common_ratio_geometric_series_l1377_137799

theorem common_ratio_geometric_series :
  let a := 2 / 3
  let b := 4 / 9
  let c := 8 / 27
  (b / a = 2 / 3) ∧ (c / b = 2 / 3) → 
  ∃ r : ℚ, r = 2 / 3 ∧ ∀ n : ℕ, (a * r^n) = (a * (2 / 3)^n) :=
by
  sorry

end common_ratio_geometric_series_l1377_137799


namespace max_rectangles_3x5_in_17x22_l1377_137767

theorem max_rectangles_3x5_in_17x22 : ∃ n : ℕ, n = 24 ∧ 
  (∀ (cut_3x5_pieces : ℤ), cut_3x5_pieces ≤ n) :=
by
  sorry

end max_rectangles_3x5_in_17x22_l1377_137767


namespace knicks_win_tournament_probability_l1377_137737

noncomputable def knicks_win_probability : ℚ :=
  let knicks_win_proba := 2 / 5
  let heat_win_proba := 3 / 5
  let first_4_games_scenarios := 6 * (knicks_win_proba^2 * heat_win_proba^2)
  first_4_games_scenarios * knicks_win_proba

theorem knicks_win_tournament_probability :
  knicks_win_probability = 432 / 3125 :=
by
  sorry

end knicks_win_tournament_probability_l1377_137737


namespace gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l1377_137790

theorem gcd_3_pow_1007_minus_1_3_pow_1018_minus_1 :
  Nat.gcd (3^1007 - 1) (3^1018 - 1) = 177146 :=
by
  -- Proof follows from the Euclidean algorithm and factoring, skipping the proof here.
  sorry

end gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l1377_137790


namespace smallest_number_of_coins_l1377_137757

theorem smallest_number_of_coins (p n d q h: ℕ) (total: ℕ) 
  (coin_value: ℕ → ℕ)
  (h_p: coin_value 1 = 1) 
  (h_n: coin_value 5 = 5) 
  (h_d: coin_value 10 = 10) 
  (h_q: coin_value 25 = 25) 
  (h_h: coin_value 50 = 50)
  (total_def: total = p * (coin_value 1) + n * (coin_value 5) +
                     d * (coin_value 10) + q * (coin_value 25) + 
                     h * (coin_value 50))
  (h_total: total = 100): 
  p + n + d + q + h = 3 :=
by
  sorry

end smallest_number_of_coins_l1377_137757


namespace algebraic_expression_value_l1377_137749

theorem algebraic_expression_value 
  (θ : ℝ)
  (a := (Real.cos θ, Real.sin θ))
  (b := (1, -2))
  (parallel : ∃ k : ℝ, a = (k * 1, k * -2)) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 := 
by 
  -- proof goes here 
  sorry

end algebraic_expression_value_l1377_137749


namespace percentage_seeds_germinated_l1377_137792

theorem percentage_seeds_germinated :
  let S1 := 300
  let S2 := 200
  let S3 := 150
  let S4 := 250
  let S5 := 100
  let G1 := 0.20
  let G2 := 0.35
  let G3 := 0.45
  let G4 := 0.25
  let G5 := 0.60
  (G1 * S1 + G2 * S2 + G3 * S3 + G4 * S4 + G5 * S5) / (S1 + S2 + S3 + S4 + S5) * 100 = 32 := 
by
  sorry

end percentage_seeds_germinated_l1377_137792


namespace krystian_total_books_borrowed_l1377_137773

/-
Conditions:
1. Krystian starts on Monday by borrowing 40 books.
2. Each day from Tuesday to Thursday, he borrows 5% more books than he did the previous day.
3. On Friday, his number of borrowed books is 40% higher than on Thursday.
4. During weekends, Krystian borrows books for his friends, and he borrows 2 additional books for every 10 books borrowed during the weekdays.

Theorem: Given these conditions, Krystian borrows a total of 283 books from Monday to Sunday.
-/
theorem krystian_total_books_borrowed : 
  let mon := 40
  let tue := mon + (5 * mon / 100)
  let wed := tue + (5 * tue / 100)
  let thu := wed + (5 * wed / 100)
  let fri := thu + (40 * thu / 100)
  let weekday_total := mon + tue + wed + thu + fri
  let weekend := 2 * (weekday_total / 10)
  weekday_total + weekend = 283 := 
by
  sorry

end krystian_total_books_borrowed_l1377_137773


namespace actual_distance_between_towns_l1377_137751

def map_distance := 20 -- distance between towns on the map in inches
def scale := 10 -- scale: 1 inch = 10 miles

theorem actual_distance_between_towns : map_distance * scale = 200 := by
  sorry

end actual_distance_between_towns_l1377_137751


namespace compound_interest_calculation_l1377_137781

theorem compound_interest_calculation : 
  ∀ (x y T SI: ℝ), 
  x = 5000 → T = 2 → SI = 500 → 
  (y = SI * 100 / (x * T)) → 
  (5000 * (1 + (y / 100))^T - 5000 = 512.5) :=
by 
  intros x y T SI hx hT hSI hy
  sorry

end compound_interest_calculation_l1377_137781


namespace find_a_l1377_137706

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 :=
by
  sorry

end find_a_l1377_137706


namespace no_solution_inequalities_l1377_137775

theorem no_solution_inequalities (m : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 < 3) → (x > m) → false) ↔ (m ≥ 2) :=
by 
  sorry

end no_solution_inequalities_l1377_137775


namespace molecular_weight_one_mole_of_AlPO4_l1377_137702

theorem molecular_weight_one_mole_of_AlPO4
  (molecular_weight_4_moles : ℝ)
  (h : molecular_weight_4_moles = 488) :
  molecular_weight_4_moles / 4 = 122 :=
by
  sorry

end molecular_weight_one_mole_of_AlPO4_l1377_137702


namespace gcd_of_polynomial_and_linear_l1377_137723

theorem gcd_of_polynomial_and_linear (b : ℤ) (h1 : b % 2 = 1) (h2 : 1019 ∣ b) : 
  Int.gcd (3 * b ^ 2 + 31 * b + 91) (b + 15) = 1 := 
by 
  sorry

end gcd_of_polynomial_and_linear_l1377_137723


namespace train_length_l1377_137707

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_ms : ℝ) (distance_m : ℝ)
  (h1 : speed_km_hr = 90)
  (h2 : time_seconds = 9)
  (h3 : speed_ms = speed_km_hr * (1000 / 3600))
  (h4 : distance_m = speed_ms * time_seconds) :
  distance_m = 225 :=
by
  sorry

end train_length_l1377_137707


namespace volvox_pentagons_heptagons_diff_l1377_137712

-- Given conditions
variables (V E F f_5 f_6 f_7 : ℕ)

-- Euler's polyhedron formula
axiom euler_formula : V - E + F = 2

-- Each edge is shared by two faces
axiom edge_formula : 2 * E = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Each vertex shared by three faces
axiom vertex_formula : 3 * V = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Total number of faces equals sum of individual face types 
def total_faces : ℕ := f_5 + f_6 + f_7

-- Prove that the number of pentagonal cells exceeds the number of heptagonal cells by 12
theorem volvox_pentagons_heptagons_diff : f_5 - f_7 = 12 := 
sorry

end volvox_pentagons_heptagons_diff_l1377_137712


namespace emily_total_beads_l1377_137793

-- Let's define the given conditions
def necklaces : ℕ := 11
def beads_per_necklace : ℕ := 28

-- The statement to prove
theorem emily_total_beads : (necklaces * beads_per_necklace) = 308 := by
  sorry

end emily_total_beads_l1377_137793


namespace origin_inside_circle_range_l1377_137750

theorem origin_inside_circle_range (m : ℝ) :
  ((0 - m)^2 + (0 + m)^2 < 8) → (-2 < m ∧ m < 2) :=
by
  intros h
  sorry

end origin_inside_circle_range_l1377_137750


namespace analogical_reasoning_ineq_l1377_137743

-- Formalization of the conditions and the theorem to be proved

def positive (a : ℕ → ℝ) (n : ℕ) := ∀ i, 1 ≤ i → i ≤ n → a i > 0

theorem analogical_reasoning_ineq {a : ℕ → ℝ} (hpos : positive a 4) (hsum : a 1 + a 2 + a 3 + a 4 = 1) : 
  (1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4) ≥ 16 := 
sorry

end analogical_reasoning_ineq_l1377_137743


namespace strawberry_cost_l1377_137789

variables (S C : ℝ)

theorem strawberry_cost :
  (C = 6 * S) ∧ (5 * S + 5 * C = 77) → S = 2.2 :=
by
  sorry

end strawberry_cost_l1377_137789


namespace segment_area_l1377_137770

noncomputable def area_segment_above_triangle (a b c : ℝ) (triangle_area : ℝ) (y : ℝ) :=
  let ellipse_area := Real.pi * a * b
  ellipse_area - triangle_area

theorem segment_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) :
  let y := (4 * Real.sqrt 2) / 3
  let triangle_area := (1 / 2) * (2 * (b - y))
  area_segment_above_triangle a b c triangle_area y = 6 * Real.pi - 2 + (4 * Real.sqrt 2) / 3 := by
  sorry

end segment_area_l1377_137770


namespace find_cake_box_width_l1377_137701

-- Define the dimensions of the carton
def carton_length := 25
def carton_width := 42
def carton_height := 60
def carton_volume := carton_length * carton_width * carton_height

-- Define the dimensions of the cake box
def cake_box_length := 8
variable (cake_box_width : ℝ) -- This is the unknown width we need to find
def cake_box_height := 5
def cake_box_volume := cake_box_length * cake_box_width * cake_box_height

-- Maximum number of cake boxes that can be placed in the carton
def max_cake_boxes := 210
def total_cake_boxes_volume := max_cake_boxes * cake_box_volume cake_box_width

-- Theorem to prove
theorem find_cake_box_width : cake_box_width = 7.5 :=
by
  sorry

end find_cake_box_width_l1377_137701


namespace complex_number_on_ray_is_specific_l1377_137748

open Complex

theorem complex_number_on_ray_is_specific (a b : ℝ) (z : ℂ) (h₁ : z = a + b * I) 
  (h₂ : a = b) (h₃ : abs z = 1) : 
  z = (Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * I :=
by
  sorry

end complex_number_on_ray_is_specific_l1377_137748


namespace nonagon_diagonals_count_l1377_137783

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l1377_137783


namespace roots_difference_l1377_137732

theorem roots_difference :
  let a := 2 
  let b := 5 
  let c := -12
  let disc := b*b - 4*a*c
  let root1 := (-b + Real.sqrt disc) / (2 * a)
  let root2 := (-b - Real.sqrt disc) / (2 * a)
  let larger_root := max root1 root2
  let smaller_root := min root1 root2
  larger_root - smaller_root = 5.5 := by
  sorry

end roots_difference_l1377_137732


namespace malachi_selfies_total_l1377_137730

theorem malachi_selfies_total (x y : ℕ) 
  (h_ratio : 10 * y = 17 * x)
  (h_diff : y = x + 630) : 
  x + y = 2430 :=
sorry

end malachi_selfies_total_l1377_137730


namespace prove_triangle_inequality_l1377_137764

def triangle_inequality (a b c a1 a2 b1 b2 c1 c2 : ℝ) : Prop := 
  a * a1 * a2 + b * b1 * b2 + c * c1 * c2 ≥ a * b * c

theorem prove_triangle_inequality 
  (a b c a1 a2 b1 b2 c1 c2 : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c)
  (h4: 0 ≤ a1) (h5: 0 ≤ a2) 
  (h6: 0 ≤ b1) (h7: 0 ≤ b2)
  (h8: 0 ≤ c1) (h9: 0 ≤ c2) : triangle_inequality a b c a1 a2 b1 b2 c1 c2 :=
sorry

end prove_triangle_inequality_l1377_137764


namespace recurrence_relation_l1377_137720

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l1377_137720


namespace algebraic_expression_correct_l1377_137711

variable (x y : ℤ)

theorem algebraic_expression_correct (h : (x - y) / (x + y) = 3) : (2 * (x - y)) / (x + y) - (x + y) / (3 * (x - y)) = 53 / 9 := 
by  
  sorry

end algebraic_expression_correct_l1377_137711


namespace find_f2_l1377_137735

noncomputable def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 0) : f 2 a b = -16 :=
by {
  sorry
}

end find_f2_l1377_137735


namespace x_value_l1377_137797

theorem x_value :
  ∀ (x y : ℝ), x = y - 0.1 * y ∧ y = 125 + 0.1 * 125 → x = 123.75 :=
by
  intros x y h
  sorry

end x_value_l1377_137797


namespace calculate_Y_payment_l1377_137718

theorem calculate_Y_payment (X Y : ℝ) (h1 : X + Y = 600) (h2 : X = 1.2 * Y) : Y = 600 / 2.2 :=
by
  sorry

end calculate_Y_payment_l1377_137718


namespace tangent_expression_equals_two_l1377_137752

noncomputable def eval_tangent_expression : ℝ :=
  (1 + Real.tan (3 * Real.pi / 180)) * (1 + Real.tan (42 * Real.pi / 180))

theorem tangent_expression_equals_two :
  eval_tangent_expression = 2 :=
by sorry

end tangent_expression_equals_two_l1377_137752


namespace time_reduced_fraction_l1377_137798

theorem time_reduced_fraction 
  (S : ℝ) (hs : S = 24.000000000000007) 
  (D : ℝ) : 
  1 - (D / (S + 12) / (D / S)) = 1 / 3 :=
by sorry

end time_reduced_fraction_l1377_137798


namespace B_completion_time_l1377_137791

theorem B_completion_time (A_days : ℕ) (A_efficiency_multiple : ℝ) (B_days_correct : ℝ) :
  A_days = 15 →
  A_efficiency_multiple = 1.8 →
  B_days_correct = 4 + 1 / 6 →
  B_days_correct = 25 / 6 :=
sorry

end B_completion_time_l1377_137791


namespace rectangle_properties_l1377_137739

theorem rectangle_properties :
  ∃ (length width : ℝ),
    (length / width = 3) ∧ 
    (length * width = 75) ∧
    (length = 15) ∧
    (width = 5) ∧
    ∀ (side : ℝ), 
      (side^2 = 75) → 
      (side - width > 3) :=
by
  sorry

end rectangle_properties_l1377_137739


namespace jim_saves_by_buying_gallon_l1377_137753

-- Define the conditions as variables
def cost_per_gallon_costco : ℕ := 8
def ounces_per_gallon : ℕ := 128
def cost_per_16oz_bottle_store : ℕ := 3
def ounces_per_bottle : ℕ := 16

-- Define the theorem that needs to be proven
theorem jim_saves_by_buying_gallon (h1 : cost_per_gallon_costco = 8)
                                    (h2 : ounces_per_gallon = 128)
                                    (h3 : cost_per_16oz_bottle_store = 3)
                                    (h4 : ounces_per_bottle = 16) : 
  (8 * 3 - 8) = 16 :=
by sorry

end jim_saves_by_buying_gallon_l1377_137753


namespace a_3_and_a_4_sum_l1377_137756

theorem a_3_and_a_4_sum (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℚ) :
  (1 - (1 / (2 * x))) ^ 6 = a_0 + a_1 * (1 / x) + a_2 * (1 / x) ^ 2 + a_3 * (1 / x) ^ 3 + 
  a_4 * (1 / x) ^ 4 + a_5 * (1 / x) ^ 5 + a_6 * (1 / x) ^ 6 →
  a_3 + a_4 = -25 / 16 :=
sorry

end a_3_and_a_4_sum_l1377_137756


namespace avg_price_six_toys_l1377_137738

def avg_price_five_toys : ℝ := 10
def price_sixth_toy : ℝ := 16
def total_toys : ℕ := 5 + 1

theorem avg_price_six_toys (avg_price_five_toys price_sixth_toy : ℝ) (total_toys : ℕ) :
  (avg_price_five_toys * 5 + price_sixth_toy) / total_toys = 11 := by
  sorry

end avg_price_six_toys_l1377_137738


namespace average_tickets_sold_by_female_l1377_137760

-- Define the conditions as Lean expressions.

def totalMembers (M : ℕ) : ℕ := M + 2 * M
def totalTickets (F : ℕ) (M : ℕ) : ℕ := 58 * M + F * 2 * M
def averageTicketsPerMember (F : ℕ) (M : ℕ) : ℕ := (totalTickets F M) / (totalMembers M)

theorem average_tickets_sold_by_female (F M : ℕ) 
  (h1 : 66 * (totalMembers M) = totalTickets F M) :
  F = 70 :=
by
  sorry

end average_tickets_sold_by_female_l1377_137760


namespace correctly_transformed_equation_l1377_137755

theorem correctly_transformed_equation (s a b x y : ℝ) :
  (s = a * b → a = s / b ∧ b ≠ 0) ∧
  (1/2 * x = 8 → x = 16) ∧
  (-x - 1 = y - 1 → x = -y) ∧
  (a = b → a + 3 = b + 3) :=
by
  sorry

end correctly_transformed_equation_l1377_137755
