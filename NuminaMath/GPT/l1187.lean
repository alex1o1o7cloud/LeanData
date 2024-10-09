import Mathlib

namespace volume_of_cut_cone_l1187_118795

theorem volume_of_cut_cone (V_frustum : ℝ) (A_bottom : ℝ) (A_top : ℝ) (V_cut_cone : ℝ) :
  V_frustum = 52 ∧ A_bottom = 9 * A_top → V_cut_cone = 54 :=
by
  sorry

end volume_of_cut_cone_l1187_118795


namespace pudding_distribution_l1187_118786

theorem pudding_distribution {puddings students : ℕ} (h1 : puddings = 315) (h2 : students = 218) : 
  ∃ (additional_puddings : ℕ), additional_puddings >= 121 ∧ ∃ (cups_per_student : ℕ), 
  (puddings + additional_puddings) ≥ students * cups_per_student :=
by
  sorry

end pudding_distribution_l1187_118786


namespace Mark_water_balloon_spending_l1187_118760

theorem Mark_water_balloon_spending :
  let budget := 24
  let small_bag_cost := 4
  let small_bag_balloons := 50
  let medium_bag_balloons := 75
  let extra_large_bag_cost := 12
  let extra_large_bag_balloons := 200
  let total_balloons := 400
  (2 * extra_large_bag_balloons = total_balloons) → (2 * extra_large_bag_cost = budget) :=
by
  intros
  sorry

end Mark_water_balloon_spending_l1187_118760


namespace find_P_eq_30_l1187_118747

theorem find_P_eq_30 (P Q R S : ℕ) :
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S ∧
  P * Q = 120 ∧ R * S = 120 ∧ P - Q = R + S → P = 30 :=
by
  sorry

end find_P_eq_30_l1187_118747


namespace closest_perfect_square_to_273_l1187_118787

theorem closest_perfect_square_to_273 : ∃ n : ℕ, (n^2 = 289) ∧ 
  ∀ m : ℕ, (m^2 < 273 → 273 - m^2 ≥ 1) ∧ (m^2 > 273 → m^2 - 273 ≥ 16) :=
by
  sorry

end closest_perfect_square_to_273_l1187_118787


namespace odd_multiple_of_9_implies_multiple_of_3_l1187_118791

theorem odd_multiple_of_9_implies_multiple_of_3 :
  ∀ (S : ℤ), (∀ (n : ℤ), 9 * n = S → ∃ (m : ℤ), 3 * m = S) ∧ (S % 2 ≠ 0) → (∃ (m : ℤ), 3 * m = S) :=
by
  sorry

end odd_multiple_of_9_implies_multiple_of_3_l1187_118791


namespace percentage_decrease_of_y_compared_to_z_l1187_118731

theorem percentage_decrease_of_y_compared_to_z (x y z : ℝ)
  (h1 : x = 1.20 * y)
  (h2 : x = 0.60 * z) :
  (y = 0.50 * z) → (1 - (y / z)) * 100 = 50 :=
by
  sorry

end percentage_decrease_of_y_compared_to_z_l1187_118731


namespace krishan_money_l1187_118775

theorem krishan_money (R G K : ℕ) (h₁ : 7 * G = 17 * R) (h₂ : 7 * K = 17 * G) (h₃ : R = 686) : K = 4046 :=
  by sorry

end krishan_money_l1187_118775


namespace problem_proof_l1187_118758

variable {x y z : ℝ}

theorem problem_proof (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) : 2 * (x + y + z) ≤ 3 := 
sorry

end problem_proof_l1187_118758


namespace arithmetic_sequence_ratio_l1187_118701

theorem arithmetic_sequence_ratio (S T : ℕ → ℕ) (a b : ℕ → ℕ)
  (h : ∀ n, S n / T n = (7 * n + 3) / (n + 3)) :
  a 8 / b 8 = 6 :=
by
  sorry

end arithmetic_sequence_ratio_l1187_118701


namespace find_a1_geometric_sequence_l1187_118773

theorem find_a1_geometric_sequence (a₁ q : ℝ) (h1 : q ≠ 1) 
    (h2 : a₁ * (1 - q^3) / (1 - q) = 7)
    (h3 : a₁ * (1 - q^6) / (1 - q) = 63) :
    a₁ = 1 :=
by
  sorry

end find_a1_geometric_sequence_l1187_118773


namespace triangle_right_angled_l1187_118785

theorem triangle_right_angled (A B C : ℝ) (h : A + B + C = 180) (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x) :
  C = 90 :=
by {
  sorry
}

end triangle_right_angled_l1187_118785


namespace ticket_cost_l1187_118799

theorem ticket_cost
    (rows : ℕ) (seats_per_row : ℕ)
    (fraction_sold : ℚ) (total_earnings : ℚ)
    (N : ℕ := rows * seats_per_row)
    (S : ℚ := fraction_sold * N)
    (C : ℚ := total_earnings / S)
    (h1 : rows = 20) (h2 : seats_per_row = 10)
    (h3 : fraction_sold = 3 / 4) (h4 : total_earnings = 1500) :
    C = 10 :=
by
  sorry

end ticket_cost_l1187_118799


namespace quadratic_roots_and_T_range_l1187_118789

theorem quadratic_roots_and_T_range
  (m : ℝ)
  (h1 : m ≥ -1)
  (x1 x2 : ℝ)
  (h2 : x1^2 + 2*(m-2)*x1 + (m^2 - 3*m + 3) = 0)
  (h3 : x2^2 + 2*(m-2)*x2 + (m^2 - 3*m + 3) = 0)
  (h4 : x1 ≠ x2)
  (h5 : x1^2 + x2^2 = 6) :
  m = (5 - Real.sqrt 17) / 2 ∧ (0 < ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≤ 4 ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≠ 2) :=
by
  sorry

end quadratic_roots_and_T_range_l1187_118789


namespace range_of_a_l1187_118718

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 ≤ 1) (f_def : ∀ x, f x = a * x - x^3)
  (condition : f x2 - f x1 > x2 - x1) :
  a ≥ 4 :=
by sorry

end range_of_a_l1187_118718


namespace TreyHasSevenTimesAsManyTurtles_l1187_118774

variable (Kristen_turtles : ℕ)
variable (Kris_turtles : ℕ)
variable (Trey_turtles : ℕ)

-- Conditions
def KristenHas12 : Kristen_turtles = 12 := sorry
def KrisHasQuarterOfKristen : Kris_turtles = Kristen_turtles / 4 := sorry
def TreyHas9MoreThanKristen : Trey_turtles = Kristen_turtles + 9 := sorry

-- Question: Prove that Trey has 7 times as many turtles as Kris
theorem TreyHasSevenTimesAsManyTurtles :
  Kristen_turtles = 12 → 
  Kris_turtles = Kristen_turtles / 4 → 
  Trey_turtles = Kristen_turtles + 9 → 
  Trey_turtles = 7 * Kris_turtles := sorry

end TreyHasSevenTimesAsManyTurtles_l1187_118774


namespace quadratic_equation_even_coefficient_l1187_118741

-- Define the predicate for a rational root
def has_rational_root (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), (q ≠ 0) ∧ (p.gcd q = 1) ∧ (a * p^2 + b * p * q + c * q^2 = 0)

-- Define the predicate for at least one being even
def at_least_one_even (a b c : ℤ) : Prop :=
  (a % 2 = 0) ∨ (b % 2 = 0) ∨ (c % 2 = 0)

theorem quadratic_equation_even_coefficient 
  (a b c : ℤ) (h_non_zero : a ≠ 0) (h_rational_root : has_rational_root a b c) :
  at_least_one_even a b c :=
sorry

end quadratic_equation_even_coefficient_l1187_118741


namespace swimmer_distance_l1187_118721

theorem swimmer_distance :
  let swimmer_speed : ℝ := 3
  let current_speed : ℝ := 1.7
  let time : ℝ := 2.3076923076923075
  let effective_speed := swimmer_speed - current_speed
  let distance := effective_speed * time
  distance = 3 := by
sorry

end swimmer_distance_l1187_118721


namespace largest_term_quotient_l1187_118736

theorem largest_term_quotient (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_S_def : ∀ n, S n = (n * (a 0 + a n)) / 2)
  (h_S15_pos : S 15 > 0)
  (h_S16_neg : S 16 < 0) :
  ∃ m, 1 ≤ m ∧ m ≤ 15 ∧
       ∀ k, (1 ≤ k ∧ k ≤ 15) → (S m / a m) ≥ (S k / a k) ∧ m = 8 := 
sorry

end largest_term_quotient_l1187_118736


namespace number_of_people_chose_pop_l1187_118792

theorem number_of_people_chose_pop (total_people : ℕ) (angle_pop : ℕ) (h1 : total_people = 540) (h2 : angle_pop = 270) : (total_people * (angle_pop / 360)) = 405 := by
  sorry

end number_of_people_chose_pop_l1187_118792


namespace quad_inequality_solution_set_is_reals_l1187_118776

theorem quad_inequality_solution_set_is_reals (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) := 
sorry

end quad_inequality_solution_set_is_reals_l1187_118776


namespace find_d_e_f_l1187_118727

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 3 + 5 / 3)

theorem find_d_e_f :
  ∃ (d e f : ℕ), (y ^ 50 = 3 * y ^ 48 + 10 * y ^ 45 + 9 * y ^ 43 - y ^ 25 + d * y ^ 21 + e * y ^ 19 + f * y ^ 15) 
    ∧ (d + e + f = 119) :=
sorry

end find_d_e_f_l1187_118727


namespace smallest_result_l1187_118710

-- Define the given set of numbers
def given_set : Set Nat := {3, 4, 7, 11, 13, 14}

-- Define the condition for prime numbers greater than 10
def is_prime_gt_10 (n : Nat) : Prop :=
  Nat.Prime n ∧ n > 10

-- Define the property of choosing three different numbers and computing the result
def compute (a b c : Nat) : Nat :=
  (a + b) * c

-- The main theorem stating the problem and its solution
theorem smallest_result : ∃ (a b c : Nat), 
  a ∈ given_set ∧ b ∈ given_set ∧ c ∈ given_set ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (is_prime_gt_10 a ∨ is_prime_gt_10 b ∨ is_prime_gt_10 c) ∧
  compute a b c = 77 ∧
  ∀ (a' b' c' : Nat), 
    a' ∈ given_set ∧ b' ∈ given_set ∧ c' ∈ given_set ∧
    a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
    (is_prime_gt_10 a' ∨ is_prime_gt_10 b' ∨ is_prime_gt_10 c') →
    compute a' b' c' ≥ 77 :=
by
  -- Proof is not required, hence sorry
  sorry

end smallest_result_l1187_118710


namespace sqrt_comparison_l1187_118781

theorem sqrt_comparison :
  let a := Real.sqrt 2
  let b := Real.sqrt 7 - Real.sqrt 3
  let c := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by
{
  sorry
}

end sqrt_comparison_l1187_118781


namespace servings_in_container_l1187_118722

def convert_to_improper_fraction (whole : ℕ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num / denom)

def servings (container : ℚ) (serving_size : ℚ) : ℚ :=
  container / serving_size

def mixed_number (whole : ℕ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num / denom)

theorem servings_in_container : 
  let container := convert_to_improper_fraction 37 2 3
  let serving_size := convert_to_improper_fraction 1 1 2
  let expected_servings := mixed_number 25 1 9
  servings container serving_size = expected_servings :=
by 
  let container := convert_to_improper_fraction 37 2 3
  let serving_size := convert_to_improper_fraction 1 1 2
  let expected_servings := mixed_number 25 1 9
  sorry

end servings_in_container_l1187_118722


namespace proof_d_e_f_value_l1187_118716

theorem proof_d_e_f_value
  (a b c d e f : ℝ)
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.75) :
  d * e * f = 250 :=
sorry

end proof_d_e_f_value_l1187_118716


namespace total_students_l1187_118798

theorem total_students (third_grade_students fourth_grade_students second_grade_boys second_grade_girls : ℕ)
  (h1 : third_grade_students = 19)
  (h2 : fourth_grade_students = 2 * third_grade_students)
  (h3 : second_grade_boys = 10)
  (h4 : second_grade_girls = 19) :
  third_grade_students + fourth_grade_students + (second_grade_boys + second_grade_girls) = 86 :=
by
  rw [h1, h3, h4, h2]
  norm_num
  sorry

end total_students_l1187_118798


namespace total_sampled_papers_l1187_118714

-- Define the conditions
variables {A B C c : ℕ}
variable (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50)
variable (stratified_sampling : true)   -- We simply denote that stratified sampling method is used

-- Theorem to prove the total number of exam papers sampled
theorem total_sampled_papers {T : ℕ} (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50) (stratified_sampling : true) :
  T = (1260 + 720 + 900) * (50 / 900) := sorry

end total_sampled_papers_l1187_118714


namespace joshInitialMarbles_l1187_118739

-- Let n be the number of marbles Josh initially had
variable (n : ℕ)

-- Condition 1: Jack gave Josh 20 marbles
def jackGaveJoshMarbles : ℕ := 20

-- Condition 2: Now Josh has 42 marbles
def joshCurrentMarbles : ℕ := 42

-- Theorem: prove that the number of marbles Josh had initially was 22
theorem joshInitialMarbles : n + jackGaveJoshMarbles = joshCurrentMarbles → n = 22 :=
by
  intros h
  sorry

end joshInitialMarbles_l1187_118739


namespace oranges_harvest_per_day_l1187_118752

theorem oranges_harvest_per_day (total_sacks : ℕ) (days : ℕ) (sacks_per_day : ℕ) 
  (h1 : total_sacks = 498) (h2 : days = 6) : total_sacks / days = sacks_per_day ∧ sacks_per_day = 83 :=
by
  sorry

end oranges_harvest_per_day_l1187_118752


namespace minimum_questionnaires_l1187_118783

theorem minimum_questionnaires (responses_needed : ℕ) (response_rate : ℝ)
  (h1 : responses_needed = 300) (h2 : response_rate = 0.70) :
  ∃ (n : ℕ), n = Nat.ceil (responses_needed / response_rate) ∧ n = 429 :=
by
  sorry

end minimum_questionnaires_l1187_118783


namespace sum_squares_of_roots_of_quadratic_l1187_118764

theorem sum_squares_of_roots_of_quadratic:
  ∀ (s_1 s_2 : ℝ),
  (s_1 + s_2 = 20) ∧ (s_1 * s_2 = 32) →
  (s_1^2 + s_2^2 = 336) :=
by
  intros s_1 s_2 h
  sorry

end sum_squares_of_roots_of_quadratic_l1187_118764


namespace roots_triple_relation_l1187_118790

theorem roots_triple_relation (a b c : ℤ) (α β : ℤ)
    (h_quad : a ≠ 0)
    (h_roots : α + β = -b / a)
    (h_prod : α * β = c / a)
    (h_triple : β = 3 * α) :
    3 * b^2 = 16 * a * c :=
sorry

end roots_triple_relation_l1187_118790


namespace all_radii_equal_l1187_118700
-- Lean 4 statement

theorem all_radii_equal (r : ℝ) (h : r = 2) : r = 2 :=
by
  sorry

end all_radii_equal_l1187_118700


namespace fruit_salad_cost_3_l1187_118769

def cost_per_fruit_salad (num_people sodas_per_person soda_cost sandwich_cost num_snacks snack_cost total_cost : ℕ) : ℕ :=
  let total_soda_cost := num_people * sodas_per_person * soda_cost
  let total_sandwich_cost := num_people * sandwich_cost
  let total_snack_cost := num_snacks * snack_cost
  let total_known_cost := total_soda_cost + total_sandwich_cost + total_snack_cost
  let total_fruit_salad_cost := total_cost - total_known_cost
  total_fruit_salad_cost / num_people

theorem fruit_salad_cost_3 :
  cost_per_fruit_salad 4 2 2 5 3 4 60 = 3 :=
by
  sorry

end fruit_salad_cost_3_l1187_118769


namespace alicia_gumballs_l1187_118733

theorem alicia_gumballs (A : ℕ) (h1 : 3 * A = 60) : A = 20 := sorry

end alicia_gumballs_l1187_118733


namespace total_length_figure_2_l1187_118763

-- Define the conditions for Figure 1
def left_side_figure_1 := 10
def right_side_figure_1 := 7
def top_side_figure_1 := 3
def bottom_side_figure_1_seg1 := 2
def bottom_side_figure_1_seg2 := 1

-- Define the conditions for Figure 2 after removal
def left_side_figure_2 := left_side_figure_1
def right_side_figure_2 := right_side_figure_1
def top_side_figure_2 := 0
def bottom_side_figure_2 := top_side_figure_1 + bottom_side_figure_1_seg1 + bottom_side_figure_1_seg2

-- The Lean statement proving the total length in Figure 2
theorem total_length_figure_2 : 
  left_side_figure_2 + right_side_figure_2 + top_side_figure_2 + bottom_side_figure_2 = 23 := by
  sorry

end total_length_figure_2_l1187_118763


namespace find_valid_primes_and_integers_l1187_118762

def is_prime (p : ℕ) : Prop := Nat.Prime p

def valid_pair (p x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 2 * p ∧ x^(p-1) ∣ (p-1)^x + 1

theorem find_valid_primes_and_integers (p x : ℕ) (hp : is_prime p) 
  (hx : valid_pair p x) : 
  (p = 2 ∧ x = 1) ∨ 
  (p = 2 ∧ x = 2) ∨ 
  (p = 3 ∧ x = 1) ∨ 
  (p = 3 ∧ x = 3) ∨
  (x = 1) :=
sorry

end find_valid_primes_and_integers_l1187_118762


namespace sum_of_squares_l1187_118711

theorem sum_of_squares (x y z : ℤ) (h1 : x + y + z = 3) (h2 : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 :=
sorry

end sum_of_squares_l1187_118711


namespace ratio_of_area_to_breadth_l1187_118766

variable (l b : ℕ)

theorem ratio_of_area_to_breadth 
  (h1 : b = 14) 
  (h2 : l - b = 10) : 
  (l * b) / b = 24 := by
  sorry

end ratio_of_area_to_breadth_l1187_118766


namespace raju_working_days_l1187_118712

theorem raju_working_days (x : ℕ) 
  (h1: (1 / 10 : ℚ) + 1 / x = 1 / 8) : x = 40 :=
by sorry

end raju_working_days_l1187_118712


namespace avg_growth_rate_leq_half_sum_l1187_118732

theorem avg_growth_rate_leq_half_sum (m n p : ℝ) (hm : 0 ≤ m) (hn : 0 ≤ n)
    (hp : (1 + p / 100)^2 = (1 + m / 100) * (1 + n / 100)) : 
    p ≤ (m + n) / 2 :=
by
  sorry

end avg_growth_rate_leq_half_sum_l1187_118732


namespace geometric_series_sum_l1187_118715

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l1187_118715


namespace determinant_equality_l1187_118725

-- Given values p, q, r, s such that the determinant of the first matrix is 5
variables {p q r s : ℝ}

-- Define the determinant condition
def det_condition (p q r s : ℝ) : Prop := p * s - q * r = 5

-- State the theorem that we need to prove
theorem determinant_equality (h : det_condition p q r s) :
  p * (5*r + 2*s) - r * (5*p + 2*q) = 10 :=
sorry

end determinant_equality_l1187_118725


namespace cube_volume_and_surface_area_l1187_118751

theorem cube_volume_and_surface_area (s : ℝ) (h : 12 * s = 72) :
  s^3 = 216 ∧ 6 * s^2 = 216 :=
by 
  sorry

end cube_volume_and_surface_area_l1187_118751


namespace trigonometric_identity_l1187_118757

theorem trigonometric_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 :=
sorry

end trigonometric_identity_l1187_118757


namespace decorations_cost_correct_l1187_118779

def cost_of_roses_per_centerpiece := 5 * 10
def cost_of_lilies_per_centerpiece := 4 * 15
def cost_of_place_settings_per_table := 4 * 10
def cost_of_tablecloth_per_table := 25
def cost_per_table := cost_of_roses_per_centerpiece + cost_of_lilies_per_centerpiece + cost_of_place_settings_per_table + cost_of_tablecloth_per_table
def number_of_tables := 20
def total_cost_of_decorations := cost_per_table * number_of_tables

theorem decorations_cost_correct :
  total_cost_of_decorations = 3500 := by
  sorry

end decorations_cost_correct_l1187_118779


namespace problem_l1187_118759

theorem problem (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : -3 * a^2 + 6 * a + 5 = 2 := by
  sorry

end problem_l1187_118759


namespace standard_deviation_does_not_require_repair_l1187_118713

-- Definitions based on conditions
def greatest_deviation (d : ℝ) := d = 39
def nominal_mass (M : ℝ) := 0.1 * M = 39
def unreadable_measurement_deviation (d : ℝ) := d < 39

-- Theorems to be proved
theorem standard_deviation (σ : ℝ) (d : ℝ) (M : ℝ) :
  greatest_deviation d →
  nominal_mass M →
  unreadable_measurement_deviation d →
  σ ≤ 39 :=
by
  sorry

theorem does_not_require_repair (σ : ℝ) :
  σ ≤ 39 → ¬(machine_requires_repair) :=
by
  sorry

-- Adding an assumption that if σ ≤ 39, the machine does not require repair
axiom machine_requires_repair : Prop

end standard_deviation_does_not_require_repair_l1187_118713


namespace cannot_be_zero_l1187_118755

-- Define polynomial Q(x)
def Q (x : ℝ) (f g h i j : ℝ) : ℝ := x^5 + f * x^4 + g * x^3 + h * x^2 + i * x + j

-- Define the hypotheses for the proof
def distinct_roots (a b c d e : ℝ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def one_root_is_one (f g h i j : ℝ) := Q 1 f g h i j = 0

-- Statement to prove
theorem cannot_be_zero (f g h i j a b c d : ℝ)
  (h1 : Q 1 f g h i j = 0)
  (h2 : distinct_roots 1 a b c d)
  (h3 : Q 1 f g h i j = (1-a)*(1-b)*(1-c)*(1-d)) :
  i ≠ 0 :=
by
  sorry

end cannot_be_zero_l1187_118755


namespace sum_of_money_l1187_118706

theorem sum_of_money (jimin_100_won : ℕ) (jimin_50_won : ℕ) (seokjin_100_won : ℕ) (seokjin_10_won : ℕ) 
  (h1 : jimin_100_won = 5) (h2 : jimin_50_won = 1) (h3 : seokjin_100_won = 2) (h4 : seokjin_10_won = 7) :
  jimin_100_won * 100 + jimin_50_won * 50 + seokjin_100_won * 100 + seokjin_10_won * 10 = 820 :=
by
  sorry

end sum_of_money_l1187_118706


namespace cube_has_12_edges_l1187_118707

-- Definition of the number of edges in a cube
def number_of_edges_of_cube : Nat := 12

-- The theorem that asserts the cube has 12 edges
theorem cube_has_12_edges : number_of_edges_of_cube = 12 := by
  -- proof to be filled later
  sorry

end cube_has_12_edges_l1187_118707


namespace rectangular_prism_total_count_l1187_118720

-- Define the dimensions of the rectangular prism
def length : ℕ := 4
def width : ℕ := 3
def height : ℕ := 5

-- Define the total count of edges, corners, and faces
def total_count : ℕ := 12 + 8 + 6

-- The proof statement that the total count is 26
theorem rectangular_prism_total_count : total_count = 26 :=
by
  sorry

end rectangular_prism_total_count_l1187_118720


namespace fraction_identity_l1187_118726

variable {n : ℕ}

theorem fraction_identity
  (h1 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → 1 / (n * (n + 1)) = 1 / n - 1 / (n + 1)))
  (h2 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → n ≠ 2 → 1 / (n * (n + 1) * (n + 2)) = 1 / (2 * n * (n + 1)) - 1 / (2 * (n + 1) * (n + 2))))
  : 1 / (n * (n + 1) * (n + 2) * (n + 3)) = 1 / (3 * n * (n + 1) * (n + 2)) - 1 / (3 * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end fraction_identity_l1187_118726


namespace systematic_sampling_twentieth_group_number_l1187_118748

theorem systematic_sampling_twentieth_group_number 
  (total_students : ℕ) 
  (total_groups : ℕ) 
  (first_group_number : ℕ) 
  (interval : ℕ) 
  (n : ℕ) 
  (drawn_number : ℕ) :
  total_students = 400 →
  total_groups = 20 →
  first_group_number = 11 →
  interval = 20 →
  n = 20 →
  drawn_number = 11 + 20 * (n - 1) →
  drawn_number = 391 :=
by
  sorry

end systematic_sampling_twentieth_group_number_l1187_118748


namespace negative_to_zero_power_l1187_118788

theorem negative_to_zero_power (a : ℝ) (h : a ≠ 0) : (-a) ^ 0 = 1 :=
by
  sorry

end negative_to_zero_power_l1187_118788


namespace coloring_ways_l1187_118719

-- Define the function that checks valid coloring
noncomputable def valid_coloring (colors : Fin 6 → Fin 3) : Prop :=
  colors 0 = 0 ∧ -- The central pentagon is colored red
  (colors 1 ≠ colors 0 ∧ colors 2 ≠ colors 1 ∧ 
   colors 3 ≠ colors 2 ∧ colors 4 ≠ colors 3 ∧ 
   colors 5 ≠ colors 4 ∧ colors 1 ≠ colors 5) -- No two adjacent polygons have the same color

-- Define the main theorem
theorem coloring_ways (f : Fin 6 → Fin 3) (h : valid_coloring f) : 
  ∃! (f : Fin 6 → Fin 3), valid_coloring f := by
  sorry

end coloring_ways_l1187_118719


namespace train_speed_A_to_B_l1187_118705

-- Define the constants
def distance : ℝ := 480
def return_speed : ℝ := 120
def return_time_longer : ℝ := 1

-- Define the train's speed function on its way from A to B
noncomputable def train_speed : ℝ := distance / (4 - return_time_longer) -- This simplifies directly to 160 based on the provided conditions.

-- State the theorem
theorem train_speed_A_to_B :
  distance / train_speed + return_time_longer = distance / return_speed :=
by
  -- Result follows from the given conditions directly
  sorry

end train_speed_A_to_B_l1187_118705


namespace tom_has_7_blue_tickets_l1187_118756

def number_of_blue_tickets_needed_for_bible := 10 * 10 * 10
def toms_current_yellow_tickets := 8
def toms_current_red_tickets := 3
def toms_needed_blue_tickets := 163

theorem tom_has_7_blue_tickets : 
  (number_of_blue_tickets_needed_for_bible - 
    (toms_current_yellow_tickets * 10 * 10 + 
     toms_current_red_tickets * 10 + 
     toms_needed_blue_tickets)) = 7 :=
by
  -- Proof can be provided here
  sorry

end tom_has_7_blue_tickets_l1187_118756


namespace chosen_numbers_divisibility_l1187_118744

theorem chosen_numbers_divisibility (n : ℕ) (S : Finset ℕ) (hS : S.card > (n + 1) / 2) :
  ∃ a ∈ S, ∃ b ∈ S, a ≠ b ∧ a ∣ b :=
by sorry

end chosen_numbers_divisibility_l1187_118744


namespace paint_time_l1187_118770

theorem paint_time (n₁ n₂ h: ℕ) (t₁ t₂: ℕ) (constant: ℕ):
  n₁ = 6 → t₁ = 8 → h = 2 → constant = 96 →
  constant = n₁ * t₁ * h → n₂ = 4 → constant = n₂ * t₂ * h →
  t₂ = 12 :=
by
  intros
  sorry

end paint_time_l1187_118770


namespace moon_land_value_l1187_118738

theorem moon_land_value (surface_area_earth : ℕ) (surface_area_moon : ℕ) (total_value_earth : ℕ) (worth_factor : ℕ)
  (h_moon_surface_area : surface_area_moon = surface_area_earth / 5)
  (h_surface_area_earth : surface_area_earth = 200) 
  (h_worth_factor : worth_factor = 6) 
  (h_total_value_earth : total_value_earth = 80) : (total_value_earth / 5) * worth_factor = 96 := 
by 
  -- Simplify using the given conditions
  -- total_value_earth / 5 is the value of the moon's land if it had the same value per square acre as Earth's land
  -- multiplying by worth_factor to get the total value on the moon
  sorry

end moon_land_value_l1187_118738


namespace solve_system1_solve_system2_l1187_118794

theorem solve_system1 (x y : ℚ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) :
  x = 3 / 2 ∧ y = -7 / 2 := 
sorry

theorem solve_system2 (x y : ℚ) (h1 : 3 * x - 2 * y = 1) (h2 : 7 * x + 4 * y = 11) :
  x = 1 ∧ y = 1 := 
sorry

end solve_system1_solve_system2_l1187_118794


namespace value_of_algebraic_expression_l1187_118734

variable {a b : ℝ}

theorem value_of_algebraic_expression (h : b = 4 * a + 3) : 4 * a - b - 2 = -5 := 
by
  sorry

end value_of_algebraic_expression_l1187_118734


namespace next_ten_winners_each_receive_160_l1187_118708

def total_prize : ℕ := 2400
def first_winner_share : ℚ := 1 / 3 * total_prize
def remaining_after_first : ℚ := total_prize - first_winner_share
def next_ten_winners_share : ℚ := remaining_after_first / 10

theorem next_ten_winners_each_receive_160 :
  next_ten_winners_share = 160 := by
sorry

end next_ten_winners_each_receive_160_l1187_118708


namespace sqrt_ab_is_integer_l1187_118704

theorem sqrt_ab_is_integer
  (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
  (h_eq : a * (b^2 + n^2) = b * (a^2 + n^2)) :
  ∃ k : ℕ, k * k = a * b :=
by
  sorry

end sqrt_ab_is_integer_l1187_118704


namespace feathers_per_flamingo_l1187_118753

theorem feathers_per_flamingo (num_boa : ℕ) (feathers_per_boa : ℕ) (num_flamingoes : ℕ) (pluck_rate : ℚ)
  (total_feathers : ℕ) (feathers_per_flamingo : ℕ) :
  num_boa = 12 →
  feathers_per_boa = 200 →
  num_flamingoes = 480 →
  pluck_rate = 0.25 →
  total_feathers = num_boa * feathers_per_boa →
  total_feathers = num_flamingoes * feathers_per_flamingo * pluck_rate →
  feathers_per_flamingo = 20 :=
by
  intros h_num_boa h_feathers_per_boa h_num_flamingoes h_pluck_rate h_total_feathers h_feathers_eq
  sorry

end feathers_per_flamingo_l1187_118753


namespace reserved_fraction_l1187_118735

variable (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ)
variable (f : ℚ)

def mrSalazarFractionReserved (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ) : ℚ :=
  1 - (leftover_oranges + rotten_oranges) * sold_fraction / initial_oranges

theorem reserved_fraction (h1 : initial_oranges = 84) (h2 : sold_fraction = 3 / 7) (h3 : rotten_oranges = 4) (h4 : leftover_oranges = 32) :
  (mrSalazarFractionReserved initial_oranges sold_fraction rotten_oranges leftover_oranges) = 1 / 4 :=
  by
    -- Proof is omitted
    sorry

end reserved_fraction_l1187_118735


namespace ticket_price_for_children_l1187_118765

open Nat

theorem ticket_price_for_children
  (C : ℕ)
  (adult_ticket_price : ℕ := 12)
  (num_adults : ℕ := 3)
  (num_children : ℕ := 3)
  (total_cost : ℕ := 66)
  (H : num_adults * adult_ticket_price + num_children * C = total_cost) :
  C = 10 :=
sorry

end ticket_price_for_children_l1187_118765


namespace jamies_score_l1187_118768

def quiz_score (correct incorrect unanswered : ℕ) : ℚ :=
  (correct * 2) + (incorrect * (-0.5)) + (unanswered * 0.25)

theorem jamies_score :
  quiz_score 16 10 4 = 28 :=
by
  sorry

end jamies_score_l1187_118768


namespace line_parallel_condition_l1187_118750

theorem line_parallel_condition (a : ℝ) :
    (a = 1) → (∀ (x y : ℝ), (ax + 2 * y - 1 = 0) ∧ (x + (a + 1) * y + 4 = 0)) → (a = 1 ∨ a = -2) :=
by
sorry

end line_parallel_condition_l1187_118750


namespace percentage_of_first_relative_to_second_l1187_118797

theorem percentage_of_first_relative_to_second (X : ℝ) 
  (first_number : ℝ := 8/100 * X) 
  (second_number : ℝ := 16/100 * X) :
  (first_number / second_number) * 100 = 50 := 
sorry

end percentage_of_first_relative_to_second_l1187_118797


namespace max_elements_A_union_B_l1187_118767

noncomputable def sets_with_conditions (A B : Finset ℝ ) (n : ℕ) : Prop :=
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ A → s.sum id ∈ B) ∧
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ B → s.prod id ∈ A)

theorem max_elements_A_union_B {A B : Finset ℝ} (n : ℕ) (hn : 1 < n)
    (hA : A.card ≥ n) (hB : B.card ≥ n)
    (h_condition : sets_with_conditions A B n) :
    A.card + B.card ≤ 2 * n :=
  sorry

end max_elements_A_union_B_l1187_118767


namespace game_ends_and_last_numbers_depend_on_start_l1187_118754
-- Given that there are three positive integers a, b, c initially.
variables (a b c : ℕ)
-- Assume that a, b, and c are greater than zero.
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Define the gcd of the three numbers.
def g := gcd (gcd a b) c

-- Define the game step condition.
def step_condition (a b c : ℕ): Prop := a > gcd b c

-- Define the termination condition.
def termination_condition (a b c : ℕ): Prop := ¬ step_condition a b c

-- The main theorem
theorem game_ends_and_last_numbers_depend_on_start (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ n, ∃ b' c', termination_condition n b' c' ∧
  n = g ∧ b' = g ∧ c' = g :=
sorry

end game_ends_and_last_numbers_depend_on_start_l1187_118754


namespace arith_seq_a1_a7_sum_l1187_118743

variable (a : ℕ → ℝ) (d : ℝ)

-- Conditions
def arithmetic_sequence : Prop :=
  ∀ n, a (n + 1) = a n + d

def condition_sum : Prop :=
  a 3 + a 4 + a 5 = 12

-- Equivalent proof problem statement
theorem arith_seq_a1_a7_sum :
  arithmetic_sequence a d →
  condition_sum a →
  a 1 + a 7 = 8 :=
by
  sorry

end arith_seq_a1_a7_sum_l1187_118743


namespace inequality_part1_inequality_part2_l1187_118793

section Proof

variable {x m : ℝ}
def f (x : ℝ) : ℝ := |2 * x + 2| + |2 * x - 3|

-- Part 1: Prove the solution set for the inequality f(x) > 7
theorem inequality_part1 (x : ℝ) :
  f x > 7 ↔ (x < -3 / 2 ∨ x > 2) := 
  sorry

-- Part 2: Prove the range of values for m such that the inequality f(x) ≤ |3m - 2| has a solution
theorem inequality_part2 (m : ℝ) :
  (∃ x, f x ≤ |3 * m - 2|) ↔ (m ≤ -1 ∨ m ≥ 7 / 3) := 
  sorry

end Proof

end inequality_part1_inequality_part2_l1187_118793


namespace prob_log3_integer_l1187_118728

theorem prob_log3_integer : 
  (∃ (N: ℕ), (100 ≤ N ∧ N ≤ 999) ∧ ∃ (k: ℕ), N = 3^k) → 
  (∃ (prob : ℚ), prob = 1 / 450) :=
sorry

end prob_log3_integer_l1187_118728


namespace knight_tour_impossible_49_squares_l1187_118740

-- Define the size of the chessboard
def boardSize : ℕ := 7

-- Define the total number of squares on the chessboard
def totalSquares : ℕ := boardSize * boardSize

-- Define the condition for a knight's tour on the 49-square board
def knight_tour_possible (n : ℕ) : Prop :=
  n = totalSquares ∧ 
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 
  -- add condition representing knight's tour and ending
  -- adjacent condition can be mathematically proved here 
  -- but we'll skip here as we asked just to state the problem not the proof.
  sorry -- Placeholder for the precise condition

-- Define the final theorem statement
theorem knight_tour_impossible_49_squares : ¬ knight_tour_possible totalSquares :=
by sorry

end knight_tour_impossible_49_squares_l1187_118740


namespace exists_unique_decomposition_l1187_118703

theorem exists_unique_decomposition (x : ℕ → ℝ) :
  ∃! (y z : ℕ → ℝ),
    (∀ n, x n = y n - z n) ∧
    (∀ n, y n ≥ 0) ∧
    (∀ n, z n ≥ z (n-1)) ∧
    (∀ n, y n * (z n - z (n-1)) = 0) ∧
    z 0 = 0 :=
sorry

end exists_unique_decomposition_l1187_118703


namespace price_of_sundae_l1187_118771

variable (num_ice_cream_bars num_sundaes : ℕ)
variable (total_price : ℚ)
variable (price_per_ice_cream_bar : ℚ)
variable (price_per_sundae : ℚ)

theorem price_of_sundae :
  num_ice_cream_bars = 125 →
  num_sundaes = 125 →
  total_price = 225 →
  price_per_ice_cream_bar = 0.60 →
  price_per_sundae = (total_price - (num_ice_cream_bars * price_per_ice_cream_bar)) / num_sundaes →
  price_per_sundae = 1.20 :=
by
  intros
  sorry

end price_of_sundae_l1187_118771


namespace three_digit_number_with_ones_digit_5_divisible_by_5_l1187_118796

theorem three_digit_number_with_ones_digit_5_divisible_by_5 (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) (h2 : N % 10 = 5) : N % 5 = 0 :=
sorry

end three_digit_number_with_ones_digit_5_divisible_by_5_l1187_118796


namespace factorial_expression_l1187_118784

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_expression (N : ℕ) (h : N > 0) :
  (factorial (N + 1) + factorial (N - 1)) / factorial (N + 2) = 
  (N^2 + N + 1) / (N^3 + 3 * N^2 + 2 * N) :=
by
  sorry

end factorial_expression_l1187_118784


namespace half_product_two_consecutive_integers_mod_3_l1187_118729

theorem half_product_two_consecutive_integers_mod_3 (A : ℤ) : 
  (A * (A + 1) / 2) % 3 = 0 ∨ (A * (A + 1) / 2) % 3 = 1 :=
sorry

end half_product_two_consecutive_integers_mod_3_l1187_118729


namespace sum_of_arithmetic_sequence_l1187_118702

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (a_6 : a 6 = 2) : 
  (11 * (a 1 + (a 1 + 10 * ((a 6 - a 1) / 5))) / 2) = 22 :=
by
  sorry

end sum_of_arithmetic_sequence_l1187_118702


namespace percent_profit_l1187_118772

theorem percent_profit (C S : ℝ) (h : 60 * C = 40 * S) : (S - C) / C * 100 = 50 := by
  sorry

end percent_profit_l1187_118772


namespace simplify_expression_l1187_118778

-- Define the given expression
def given_expression (x : ℝ) : ℝ := 5 * x + 9 * x^2 + 8 - (6 - 5 * x - 3 * x^2)

-- Define the expected simplified form
def expected_expression (x : ℝ) : ℝ := 12 * x^2 + 10 * x + 2

-- The theorem we want to prove
theorem simplify_expression (x : ℝ) : given_expression x = expected_expression x := by
  sorry

end simplify_expression_l1187_118778


namespace isosceles_triangle_l1187_118724

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b^2 - 2 * b * c + c^2) = 0) : 
  (a = b) ∨ (b = c) :=
by sorry

end isosceles_triangle_l1187_118724


namespace solve_for_x_l1187_118749

theorem solve_for_x (x : ℚ) (h : (x + 8) / (x - 4) = (x - 3) / (x + 6)) : 
  x = -12 / 7 :=
sorry

end solve_for_x_l1187_118749


namespace cristobal_read_more_pages_l1187_118746

-- Defining the given conditions
def pages_beatrix_read : ℕ := 704
def pages_cristobal_read (b : ℕ) : ℕ := 3 * b + 15

-- Stating the problem
theorem cristobal_read_more_pages (b : ℕ) (c : ℕ) (h : b = pages_beatrix_read) (h_c : c = pages_cristobal_read b) :
  (c - b) = 1423 :=
by
  sorry

end cristobal_read_more_pages_l1187_118746


namespace ice_cream_sundaes_l1187_118761

theorem ice_cream_sundaes (flavors : Finset String) (vanilla : String) (h1 : vanilla ∈ flavors) (h2 : flavors.card = 8) :
  let remaining_flavors := flavors.erase vanilla
  remaining_flavors.card = 7 :=
by
  sorry

end ice_cream_sundaes_l1187_118761


namespace total_amount_collected_l1187_118745

theorem total_amount_collected 
  (num_members : ℕ)
  (annual_fee : ℕ)
  (cost_hardcover : ℕ)
  (num_hardcovers : ℕ)
  (cost_paperback : ℕ)
  (num_paperbacks : ℕ)
  (total_collected : ℕ) :
  num_members = 6 →
  annual_fee = 150 →
  cost_hardcover = 30 →
  num_hardcovers = 6 →
  cost_paperback = 12 →
  num_paperbacks = 6 →
  total_collected = (annual_fee + cost_hardcover * num_hardcovers + cost_paperback * num_paperbacks) * num_members →
  total_collected = 2412 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end total_amount_collected_l1187_118745


namespace negation_of_universal_statement_l1187_118717

theorem negation_of_universal_statement:
  (∀ x : ℝ, x ≥ 2) ↔ ¬ (∃ x : ℝ, x < 2) :=
by {
  sorry
}

end negation_of_universal_statement_l1187_118717


namespace last_digit_2008_pow_2008_l1187_118730

theorem last_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := by
  -- Here, the proof would follow the understanding of the cyclic pattern of the last digits of powers of 2008
  sorry

end last_digit_2008_pow_2008_l1187_118730


namespace set_B_correct_l1187_118737

-- Define the set A
def A : Set ℤ := {-1, 0, 1, 2}

-- Define the set B using the given formula
def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2 * x}

-- State the theorem that B is equal to the given set {-1, 0, 3}
theorem set_B_correct : B = {-1, 0, 3} := 
by 
  sorry

end set_B_correct_l1187_118737


namespace find_a_l1187_118780

noncomputable def a_b_c_complex (a b c : ℂ) : Prop :=
  a.re = a ∧ a + b + c = 4 ∧ a * b + b * c + c * a = 6 ∧ a * b * c = 8

theorem find_a (a b c : ℂ) (h : a_b_c_complex a b c) : a = 3 :=
by
  sorry

end find_a_l1187_118780


namespace trainB_speed_l1187_118742

variable (v : ℕ)

def trainA_speed : ℕ := 30
def time_gap : ℕ := 2
def distance_overtake : ℕ := 360

theorem trainB_speed (h :  v > trainA_speed) : v = 42 :=
by
  sorry

end trainB_speed_l1187_118742


namespace total_games_played_in_league_l1187_118723

theorem total_games_played_in_league (n : ℕ) (k : ℕ) (games_per_team : ℕ) 
  (h1 : n = 10) 
  (h2 : k = 4) 
  (h3 : games_per_team = n - 1) 
  : (k * (n * games_per_team) / 2) = 180 :=
by
  -- Definitions and transformations go here
  sorry

end total_games_played_in_league_l1187_118723


namespace find_m_l1187_118782

noncomputable def f : ℝ → ℝ := sorry

theorem find_m (h₁ : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h₂ : f 2 = m) : m = -1 / 2 :=
by
  sorry

end find_m_l1187_118782


namespace only_element_in_intersection_l1187_118709

theorem only_element_in_intersection :
  ∃! (n : ℕ), n = 2500 ∧ ∃ (r : ℚ), r ≠ 2 ∧ r ≠ -2 ∧ 404 / (r^2 - 4) = n := sorry

end only_element_in_intersection_l1187_118709


namespace fraction_before_simplification_is_24_56_l1187_118777

-- Definitions of conditions
def fraction_before_simplification_simplifies_to_3_7 (a b : ℕ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧ Int.gcd a b = 1 ∧ (a = 3 * Int.gcd a b ∧ b = 7 * Int.gcd a b)

def sum_of_numerator_and_denominator_is_80 (a b : ℕ) : Prop :=
  a + b = 80

-- Theorem to prove
theorem fraction_before_simplification_is_24_56 (a b : ℕ) :
  fraction_before_simplification_simplifies_to_3_7 a b →
  sum_of_numerator_and_denominator_is_80 a b →
  (a, b) = (24, 56) :=
sorry

end fraction_before_simplification_is_24_56_l1187_118777
