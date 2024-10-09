import Mathlib

namespace jen_age_proof_l1077_107765

-- Definitions
def son_age := 16
def son_present_age := son_age
def jen_present_age := 41

-- Conditions
axiom jen_older_25 (x : ℕ) : ∀ y : ℕ, x = y + 25 → y = son_present_age
axiom jen_age_formula (j s : ℕ) : j = 3 * s - 7 → j = son_present_age + 25

-- Proof problem statement
theorem jen_age_proof : jen_present_age = 41 :=
by
  -- Declare variables
  let j := jen_present_age
  let s := son_present_age
  -- Apply conditions (in Lean, sorry will skip the proof)
  sorry

end jen_age_proof_l1077_107765


namespace min_positive_announcements_l1077_107728

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 90)
  (h2 : y * (y - 1) + (x - y) * (x - y - 1) = 48) 
  : y = 3 :=
sorry

end min_positive_announcements_l1077_107728


namespace total_amount_after_refunds_l1077_107793

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem total_amount_after_refunds : 
  individual_bookings + group_bookings - refunds = 26400 :=
by 
  -- The proof goes here.
  sorry

end total_amount_after_refunds_l1077_107793


namespace cube_cut_problem_l1077_107722

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l1077_107722


namespace fraction_evaluation_l1077_107712

def number_of_primes_between_10_and_30 : ℕ := 6

theorem fraction_evaluation : (number_of_primes_between_10_and_30^2 - 4) / (number_of_primes_between_10_and_30 + 2) = 4 := by
  sorry

end fraction_evaluation_l1077_107712


namespace simplify_expression_l1077_107767

def expr_initial (y : ℝ) := 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2)
def expr_simplified (y : ℝ) := 8*y^2 + 6*y - 5

theorem simplify_expression (y : ℝ) : expr_initial y = expr_simplified y :=
by
  sorry

end simplify_expression_l1077_107767


namespace number_of_n_l1077_107792

theorem number_of_n (n : ℕ) (hn : n ≤ 500) (hk : ∃ k : ℕ, 21 * n = k^2) : 
  ∃ m : ℕ, m = 4 := by
  sorry

end number_of_n_l1077_107792


namespace simplify_expression_l1077_107764

variable (x y z : ℝ)

theorem simplify_expression (hxz : x > z) (hzy : z > y) (hy0 : y > 0) :
  (x^z * z^y * y^x) / (z^z * y^y * x^x) = x^(z-x) * z^(y-z) * y^(x-y) :=
sorry

end simplify_expression_l1077_107764


namespace ratio_qp_l1077_107716

theorem ratio_qp (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 6 → 
    P / (x + 3) + Q / (x * (x - 6)) = (x^2 - 4 * x + 15) / (x * (x + 3) * (x - 6))) : 
  Q / P = 5 := 
sorry

end ratio_qp_l1077_107716


namespace number_of_goats_l1077_107762

theorem number_of_goats (C G : ℕ) 
  (h1 : C = 2) 
  (h2 : ∀ G : ℕ, 460 * C + 60 * G = 1400) 
  (h3 : 460 = 460) 
  (h4 : 60 = 60) : 
  G = 8 :=
by
  sorry

end number_of_goats_l1077_107762


namespace find_equations_of_lines_l1077_107788

-- Define the given constants and conditions
def point_P := (2, 2)
def line_l1 (x y : ℝ) := 3 * x - 2 * y + 1 = 0
def line_l2 (x y : ℝ) := x + 3 * y + 4 = 0
def intersection_point := (-1, -1)
def slope_perpendicular_line := 3

-- The theorem that we need to prove
theorem find_equations_of_lines :
  (∀ k, k = 0 → line_l1 2 2 → (x = y ∨ x + y = 4)) ∧
  (line_l1 (-1) (-1) ∧ line_l2 (-1) (-1) →
   (3 * x - y + 2 = 0))
:=
sorry

end find_equations_of_lines_l1077_107788


namespace max_abs_x2_is_2_l1077_107755

noncomputable def max_abs_x2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) : ℝ :=
2

theorem max_abs_x2_is_2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) :
  max_abs_x2 h = 2 := 
sorry

end max_abs_x2_is_2_l1077_107755


namespace geometric_progression_fraction_l1077_107763

theorem geometric_progression_fraction (a₁ a₂ a₃ a₄ : ℝ) (h1 : a₂ = 2 * a₁) (h2 : a₃ = 2 * a₂) (h3 : a₄ = 2 * a₃) : 
  (2 * a₁ + a₂) / (2 * a₃ + a₄) = 1 / 4 := 
by 
  sorry

end geometric_progression_fraction_l1077_107763


namespace line_parabola_one_intersection_l1077_107710

theorem line_parabola_one_intersection (k : ℝ) : 
  ((∃ (x y : ℝ), y = k * x - 1 ∧ y^2 = 4 * x ∧ (∀ u v : ℝ, u ≠ x → v = k * u - 1 → v^2 ≠ 4 * u)) ↔ (k = 0 ∨ k = 1)) := 
sorry

end line_parabola_one_intersection_l1077_107710


namespace sum_fiftieth_powers_100_gon_l1077_107735

noncomputable def sum_fiftieth_powers_all_sides_and_diagonals (n : ℕ) (R : ℝ) : ℝ := sorry
-- Define the sum of 50-th powers of all the sides and diagonals for a general n-gon inscribed in a circle of radius R

theorem sum_fiftieth_powers_100_gon (R : ℝ) : 
  sum_fiftieth_powers_all_sides_and_diagonals 100 R = sorry := sorry

end sum_fiftieth_powers_100_gon_l1077_107735


namespace soccer_team_selection_l1077_107766

-- Definitions of the problem
def total_members := 16
def utility_exclusion_cond := total_members - 1

-- Lean statement for the proof problem, using the conditions and answer:
theorem soccer_team_selection :
  (utility_exclusion_cond) * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4) = 409500 :=
by
  sorry

end soccer_team_selection_l1077_107766


namespace extrema_f_unique_solution_F_l1077_107782

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * x^2 - m * Real.log x
noncomputable def F (x : ℝ) (m : ℝ) : ℝ := - (1 / 2) * x^2 + (m + 1) * x - m * Real.log x

theorem extrema_f (m : ℝ) :
  (m ≤ 0 → ∀ x > 0, ∀ y > 0, x ≠ y → f x m ≠ f y m) ∧
  (m > 0 → ∃ x₀ > 0, ∀ x > 0, f x₀ m ≤ f x m) :=
sorry

theorem unique_solution_F (m : ℝ) (h : m ≥ 1) :
  ∃ x₀ > 0, ∀ x > 0, F x₀ m = 0 ∧ (F x m = 0 → x = x₀) :=
sorry

end extrema_f_unique_solution_F_l1077_107782


namespace four_digit_number_divisible_by_11_l1077_107769

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end four_digit_number_divisible_by_11_l1077_107769


namespace children_on_bus_after_stops_l1077_107715

-- Define the initial number of children and changes at each stop
def initial_children := 128
def first_stop_addition := 67
def second_stop_subtraction := 34
def third_stop_addition := 54

-- Prove that the number of children on the bus after all the stops is 215
theorem children_on_bus_after_stops :
  initial_children + first_stop_addition - second_stop_subtraction + third_stop_addition = 215 := by
  -- The proof is omitted
  sorry

end children_on_bus_after_stops_l1077_107715


namespace remainder_when_four_times_n_minus_9_divided_by_11_l1077_107723

theorem remainder_when_four_times_n_minus_9_divided_by_11 
  (n : ℤ) (h : n % 11 = 4) : (4 * n - 9) % 11 = 7 := by
  sorry

end remainder_when_four_times_n_minus_9_divided_by_11_l1077_107723


namespace no_int_a_divisible_289_l1077_107739

theorem no_int_a_divisible_289 : ¬ ∃ a : ℤ, ∃ k : ℤ, a^2 - 3 * a - 19 = 289 * k :=
by
  sorry

end no_int_a_divisible_289_l1077_107739


namespace units_digit_sum_l1077_107789

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum :
  units_digit (24^3 + 17^3) = 7 :=
by
  sorry

end units_digit_sum_l1077_107789


namespace solve_inequality_l1077_107718

theorem solve_inequality :
  {x : ℝ | (x - 1) * (2 * x + 1) ≤ 0} = { x : ℝ | -1/2 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end solve_inequality_l1077_107718


namespace find_b_l1077_107785

theorem find_b (b : ℝ) (tangent_condition : ∀ x y : ℝ, y = -2 * x + b → y^2 = 8 * x) : b = -1 :=
sorry

end find_b_l1077_107785


namespace fiftieth_statement_l1077_107776

-- Define the types
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

-- Define the function telling the statement
def statement (inhabitant : Inhabitant) : String :=
  match inhabitant with
  | Inhabitant.knight => "Knight"
  | Inhabitant.liar => "Liar"

-- Define the condition: knights tell the truth and liars lie
def tells_truth (inhabitant : Inhabitant) (statement_about_neighbor : String) : Prop :=
  match inhabitant with
  | Inhabitant.knight => statement_about_neighbor = "Knight"
  | Inhabitant.liar => statement_about_neighbor ≠ "Knight"

-- Define a function that determines what each inhabitant says about their right-hand neighbor
def what_they_say (idx : ℕ) : String :=
  if idx % 2 = 0 then "Liar" else "Knight"

-- Define the inhabitant pattern
def inhabitant_at (idx : ℕ) : Inhabitant :=
  if idx % 2 = 0 then Inhabitant.liar else Inhabitant.knight

-- The main theorem statement
theorem fiftieth_statement : tells_truth (inhabitant_at 49) (what_they_say 50) :=
by 
  -- This proof outlines the theorem statement only
  sorry

end fiftieth_statement_l1077_107776


namespace shared_friends_count_l1077_107786

theorem shared_friends_count (james_friends : ℕ) (total_combined : ℕ) (john_factor : ℕ) 
  (h1 : james_friends = 75) 
  (h2 : john_factor = 3) 
  (h3 : total_combined = 275) : 
  james_friends + (john_factor * james_friends) - total_combined = 25 := 
by
  sorry

end shared_friends_count_l1077_107786


namespace find_x_l1077_107719

namespace ProofProblem

def δ (x : ℚ) : ℚ := 5 * x + 6
def φ (x : ℚ) : ℚ := 9 * x + 4

theorem find_x (x : ℚ) : (δ (φ x) = 14) ↔ (x = -4 / 15) :=
by
  sorry

end ProofProblem

end find_x_l1077_107719


namespace number_satisfying_condition_l1077_107747

-- The sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main theorem
theorem number_satisfying_condition : ∃ n : ℕ, n * sum_of_digits n = 2008 ∧ n = 251 :=
by
  sorry

end number_satisfying_condition_l1077_107747


namespace power_sum_eq_l1077_107724

theorem power_sum_eq (n : ℕ) : (-2)^2009 + (-2)^2010 = 2^2009 := by
  sorry

end power_sum_eq_l1077_107724


namespace min_value_of_a_l1077_107721

noncomputable def smallest_root_sum : ℕ := 78

theorem min_value_of_a (r s t : ℕ) (h1 : r * s * t = 2310) (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) :
  r + s + t = smallest_root_sum :=
sorry

end min_value_of_a_l1077_107721


namespace evaluate_sum_l1077_107743

theorem evaluate_sum : (-1:ℤ) ^ 2010 + (-1:ℤ) ^ 2011 + (1:ℤ) ^ 2012 - (1:ℤ) ^ 2013 + (-1:ℤ) ^ 2014 = 0 := by
  sorry

end evaluate_sum_l1077_107743


namespace gcd_sequence_terms_l1077_107753

theorem gcd_sequence_terms (d m : ℕ) (hd : d > 1) (hm : m > 0) :
    ∃ k l : ℕ, k ≠ l ∧ gcd (2 ^ (2 ^ k) + d) (2 ^ (2 ^ l) + d) > m := 
sorry

end gcd_sequence_terms_l1077_107753


namespace concert_duration_l1077_107787

def duration_in_minutes (hours : Int) (extra_minutes : Int) : Int :=
  hours * 60 + extra_minutes

theorem concert_duration : duration_in_minutes 7 45 = 465 :=
by
  sorry

end concert_duration_l1077_107787


namespace polynomial_bound_l1077_107749

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (hP : ∀ x : ℝ, |x| < 1 → |P x a b c d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end polynomial_bound_l1077_107749


namespace number_of_integers_with_three_divisors_l1077_107757

def has_exactly_three_positive_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p * p

theorem number_of_integers_with_three_divisors (n : ℕ) :
  n = 2012 → Nat.card { x : ℕ | x ≤ n ∧ has_exactly_three_positive_divisors x } = 14 :=
by
  sorry

end number_of_integers_with_three_divisors_l1077_107757


namespace find_expression_value_l1077_107778

theorem find_expression_value (x : ℝ) (h : 4 * x^2 - 2 * x + 5 = 7) :
  2 * (x^2 - x) - (x - 1) + (2 * x + 3) = 5 := by
  sorry

end find_expression_value_l1077_107778


namespace least_subtraction_to_divisible_by_prime_l1077_107758

theorem least_subtraction_to_divisible_by_prime :
  ∃ k : ℕ, (k = 46) ∧ (856324 - k) % 101 = 0 :=
by
  sorry

end least_subtraction_to_divisible_by_prime_l1077_107758


namespace total_candies_l1077_107774

theorem total_candies (n p r : ℕ) (H1 : n = 157) (H2 : p = 235) (H3 : r = 98) :
  n * p + r = 36993 := by
  sorry

end total_candies_l1077_107774


namespace complement_of_A_in_U_l1077_107751

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | |x - 1| > 2 }

theorem complement_of_A_in_U : 
  ∀ x, x ∈ U → x ∈ U \ A ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

end complement_of_A_in_U_l1077_107751


namespace find_m_of_parallelepiped_volume_l1077_107795

theorem find_m_of_parallelepiped_volume 
  {m : ℝ} 
  (h_pos : m > 0) 
  (h_vol : abs (3 * (m^2 - 9) - 2 * (4 * m - 15) + 2 * (12 - 5 * m)) = 20) : 
  m = (9 + Real.sqrt 249) / 6 :=
sorry

end find_m_of_parallelepiped_volume_l1077_107795


namespace find_other_root_l1077_107754

theorem find_other_root (a b c x : ℝ) (h₁ : a ≠ 0) 
  (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h₄ : a * (b + 2 * c) * x^2 + b * (2 * c - a) * x + c * (2 * a - b) = 0)
  (h₅ : a * (b + 2 * c) - b * (2 * c - a) + c * (2 * a - b) = 0) :
  ∃ y : ℝ, y = - (c * (2 * a - b)) / (a * (b + 2 * c)) :=
sorry

end find_other_root_l1077_107754


namespace negation_equiv_l1077_107750

open Classical

-- Proposition p
def p : Prop := ∃ x : ℝ, x^2 - x + 1 = 0

-- Negation of proposition p
def neg_p : Prop := ∀ x : ℝ, x^2 - x + 1 ≠ 0

-- Statement to prove the equivalence of the negation of p and neg_p
theorem negation_equiv :
  ¬p ↔ neg_p := 
sorry

end negation_equiv_l1077_107750


namespace subway_boarding_probability_l1077_107760

theorem subway_boarding_probability :
  ∀ (total_interval boarding_interval : ℕ),
  total_interval = 10 →
  boarding_interval = 1 →
  (boarding_interval : ℚ) / total_interval = 1 / 10 := by
  intros total_interval boarding_interval ht hb
  rw [hb, ht]
  norm_num

end subway_boarding_probability_l1077_107760


namespace set_difference_NM_l1077_107744

open Set

def setDifference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_NM :
  let M := {1, 2, 3, 4, 5}
  let N := {1, 2, 3, 7}
  setDifference N M = {7} :=
by
  sorry

end set_difference_NM_l1077_107744


namespace total_bill_amount_l1077_107706

theorem total_bill_amount (n : ℕ) (cost_per_meal : ℕ) (gratuity_rate : ℚ) (total_bill_with_gratuity : ℚ)
  (h1 : n = 7) (h2 : cost_per_meal = 100) (h3 : gratuity_rate = 20 / 100) :
  total_bill_with_gratuity = (n * cost_per_meal : ℕ) * (1 + gratuity_rate) :=
sorry

end total_bill_amount_l1077_107706


namespace count_squares_containing_A_l1077_107773

-- Given conditions
def figure_with_squares : Prop := ∃ n : ℕ, n = 20

-- The goal is to prove that the number of squares containing A is 13
theorem count_squares_containing_A (h : figure_with_squares) : ∃ k : ℕ, k = 13 :=
by 
  sorry

end count_squares_containing_A_l1077_107773


namespace minimum_value_f_l1077_107772

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem minimum_value_f (x : ℝ) (h : x > 1) : (∃ y, (f y = 3) ∧ ∀ z, z > 1 → f z ≥ 3) :=
by sorry

end minimum_value_f_l1077_107772


namespace geometric_sum_S12_l1077_107701

theorem geometric_sum_S12 (a r : ℝ) (h₁ : r ≠ 1) (S4_eq : a * (1 - r^4) / (1 - r) = 24) (S8_eq : a * (1 - r^8) / (1 - r) = 36) : a * (1 - r^12) / (1 - r) = 42 := 
sorry

end geometric_sum_S12_l1077_107701


namespace compare_negatives_l1077_107779

theorem compare_negatives : (-1.5 : ℝ) < (-1 + -1/5 : ℝ) :=
by 
  sorry

end compare_negatives_l1077_107779


namespace isabella_exchange_l1077_107720

theorem isabella_exchange (d : ℚ) : 
  (8 * d / 5 - 72 = 4 * d) → d = -30 :=
by
  sorry

end isabella_exchange_l1077_107720


namespace Mary_is_10_years_younger_l1077_107709

theorem Mary_is_10_years_younger
  (betty_age : ℕ)
  (albert_age : ℕ)
  (mary_age : ℕ)
  (h1 : albert_age = 2 * mary_age)
  (h2 : albert_age = 4 * betty_age)
  (h_betty : betty_age = 5) :
  (albert_age - mary_age) = 10 :=
  by
  sorry

end Mary_is_10_years_younger_l1077_107709


namespace seats_capacity_l1077_107790

theorem seats_capacity (x : ℕ) (h1 : 15 * x + 12 * x + 8 = 89) : x = 3 :=
by
  -- proof to be filled in
  sorry

end seats_capacity_l1077_107790


namespace weave_mats_l1077_107703

theorem weave_mats (m n p q : ℕ) (h1 : m * n = p * q) (h2 : ∀ k, k = n → n * 2 = k * 2) :
  (8 * 2 = 16) :=
by
  -- This is where we would traditionally include the proof steps.
  sorry

end weave_mats_l1077_107703


namespace max_value_ab_ac_bc_l1077_107711

open Real

theorem max_value_ab_ac_bc {a b c : ℝ} (h : a + 3 * b + c = 6) : 
  ab + ac + bc ≤ 4 :=
sorry

end max_value_ab_ac_bc_l1077_107711


namespace number_of_B_students_l1077_107777

/-- Let x be the number of students who earn a B. 
    Given the conditions:
    - The number of students who earn an A is 0.5x.
    - The number of students who earn a C is 2x.
    - The number of students who earn a D is 0.3x.
    - The total number of students in the class is 40.
    Prove the number of students who earn a B is 40 / 3.8 = 200 / 19, approximately 11. -/
theorem number_of_B_students (x : ℝ) (h_bA: x * 0.5 + x + x * 2 + x * 0.3 = 40) : 
  x = 40 / 3.8 :=
by 
  sorry

end number_of_B_students_l1077_107777


namespace georgina_teaches_2_phrases_per_week_l1077_107729

theorem georgina_teaches_2_phrases_per_week
    (total_phrases : ℕ) 
    (initial_phrases : ℕ) 
    (days_owned : ℕ)
    (phrases_per_week : ℕ):
    total_phrases = 17 → 
    initial_phrases = 3 → 
    days_owned = 49 → 
    phrases_per_week = (total_phrases - initial_phrases) / (days_owned / 7) → 
    phrases_per_week = 2 := 
by
  intros h_total h_initial h_days h_calc
  rw [h_total, h_initial, h_days] at h_calc
  sorry  -- Proof to be filled

end georgina_teaches_2_phrases_per_week_l1077_107729


namespace work_completion_days_l1077_107748

theorem work_completion_days (A B : ℕ) (hA : A = 20) (hB : B = 20) : A + B / (A + B) / 2 = 10 :=
by 
  rw [hA, hB]
  -- Proof omitted
  sorry

end work_completion_days_l1077_107748


namespace distance_between_centers_of_circles_l1077_107702

theorem distance_between_centers_of_circles (C_1 C_2 : ℝ) : 
  (∀ a : ℝ, (C_1 = a ∧ C_2 = a ∧ (4- a)^2 + (1 - a)^2 = a^2)) → 
  |C_1 - C_2| = 8 :=
by
  sorry

end distance_between_centers_of_circles_l1077_107702


namespace hulk_jump_geometric_sequence_l1077_107730

theorem hulk_jump_geometric_sequence (n : ℕ) (a_n : ℕ) : 
  (a_n = 3 * 2^(n - 1)) → (a_n > 3000) → n = 11 :=
by
  sorry

end hulk_jump_geometric_sequence_l1077_107730


namespace simple_interest_true_discount_l1077_107700

theorem simple_interest_true_discount (P R T : ℝ) 
  (h1 : 85 = (P * R * T) / 100)
  (h2 : 80 = (85 * P) / (P + 85)) : P = 1360 :=
sorry

end simple_interest_true_discount_l1077_107700


namespace product_of_axes_l1077_107770

-- Definitions based on conditions
def ellipse (a b : ℝ) : Prop :=
  a^2 - b^2 = 64

def triangle_incircle_diameter (a b : ℝ) : Prop :=
  b + 8 - a = 4

-- Proving that (AB)(CD) = 240
theorem product_of_axes (a b : ℝ) (h₁ : ellipse a b) (h₂ : triangle_incircle_diameter a b) : 
  (2 * a) * (2 * b) = 240 :=
by
  sorry

end product_of_axes_l1077_107770


namespace problem_A_problem_C_l1077_107708

section
variables {a b : ℝ}

-- A: If a and b are positive real numbers, and a > b, then a^3 + b^3 > a^2 * b + a * b^2.
theorem problem_A (ha : 0 < a) (hb : 0 < b) (h : a > b) : a^3 + b^3 > a^2 * b + a * b^2 := sorry

end

section
variables {a b : ℝ}

-- C: If a and b are real numbers, then "a > b > 0" is a sufficient but not necessary condition for "1/a < 1/b".
theorem problem_C (ha : 0 < a) (hb : 0 < b) (h : a > b) : 1/a < 1/b := sorry

end

end problem_A_problem_C_l1077_107708


namespace find_special_numbers_l1077_107717

def is_digit_sum_equal (n m : Nat) : Prop := 
  (n.digits 10).sum = (m.digits 10).sum

def is_valid_number (n : Nat) : Prop := 
  100 ≤ n ∧ n ≤ 999 ∧ is_digit_sum_equal n (6 * n)

theorem find_special_numbers :
  {n : Nat | is_valid_number n} = {117, 135} :=
sorry

end find_special_numbers_l1077_107717


namespace prove_a2_a3_a4_sum_l1077_107733

theorem prove_a2_a3_a4_sum (a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, a1 * (x-1)^4 + a2 * (x-1)^3 + a3 * (x-1)^2 + a4 * (x-1) + a5 = x^4) :
  a2 + a3 + a4 = 14 :=
sorry

end prove_a2_a3_a4_sum_l1077_107733


namespace son_l1077_107738

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 24) 
  (h2 : M + 2 = 2 * (S + 2)) : S = 22 := 
by 
  sorry

end son_l1077_107738


namespace constant_term_binomial_expansion_l1077_107761

theorem constant_term_binomial_expansion (a : ℝ) (h : 15 * a^2 = 120) : a = 2 * Real.sqrt 2 :=
sorry

end constant_term_binomial_expansion_l1077_107761


namespace curved_surface_area_cone_l1077_107707

theorem curved_surface_area_cone :
  let r := 8  -- base radius in cm
  let l := 19  -- lateral edge length in cm
  let π := Real.pi
  let CSA := π * r * l
  477.5 < CSA ∧ CSA < 478 := by
  sorry

end curved_surface_area_cone_l1077_107707


namespace Ali_winning_strategy_l1077_107705

def Ali_and_Mohammad_game (m n : ℕ) (a : Fin m → ℕ) : Prop :=
∃ (k l : ℕ), k > 0 ∧ l > 0 ∧ (∃ p : ℕ, Nat.Prime p ∧ m = p^k ∧ n = p^l)

theorem Ali_winning_strategy (m n : ℕ) (a : Fin m → ℕ) :
  Ali_and_Mohammad_game m n a :=
sorry

end Ali_winning_strategy_l1077_107705


namespace air_conditioned_rooms_fraction_l1077_107732

theorem air_conditioned_rooms_fraction (R A : ℝ) (h1 : 3/4 * R = 3/4 * R - 1/4 * R)
                                        (h2 : 2/3 * A = 2/3 * A - 1/3 * A)
                                        (h3 : 1/3 * A = 0.8 * 1/4 * R) :
    A / R = 3 / 5 :=
by
  -- Proof content goes here
  sorry

end air_conditioned_rooms_fraction_l1077_107732


namespace min_value_M_l1077_107784

theorem min_value_M (a b c : ℝ) (h1 : a < b) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0): 
  ∃ M : ℝ, M = 8 ∧ M = (a + 2 * b + 4 * c) / (b - a) :=
sorry

end min_value_M_l1077_107784


namespace triangle_inequality_l1077_107742

theorem triangle_inequality 
  (A B C : ℝ) -- angle measures
  (a b c : ℝ) -- side lengths
  (h1 : a = b * (Real.cos C) + c * (Real.cos B)) 
  (cos_half_C_pos : 0 < Real.cos (C/2)) 
  (cos_half_C_lt_one : Real.cos (C/2) < 1)
  (cos_half_B_pos : 0 < Real.cos (B/2)) 
  (cos_half_B_lt_one : Real.cos (B/2) < 1) :
  2 * b * Real.cos (C / 2) + 2 * c * Real.cos (B / 2) > a + b + c :=
by
  sorry

end triangle_inequality_l1077_107742


namespace max_writers_and_editors_l1077_107775

theorem max_writers_and_editors (total_people writers editors x : ℕ) (h_total_people : total_people = 100)
(h_writers : writers = 40) (h_editors : editors > 38) (h_both : 2 * x + (writers + editors - x) = total_people) :
x ≤ 21 := sorry

end max_writers_and_editors_l1077_107775


namespace triangle_is_right_triangle_l1077_107726

theorem triangle_is_right_triangle 
  (A B C : ℝ)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = 180)
  (h3 : A / B = 2 / 3)
  (h4 : A / C = 2 / 5) : 
  A = 36 ∧ B = 54 ∧ C = 90 := 
sorry

end triangle_is_right_triangle_l1077_107726


namespace fabric_area_l1077_107746

theorem fabric_area (length width : ℝ) (h_length : length = 8) (h_width : width = 3) : 
  length * width = 24 := 
by
  rw [h_length, h_width]
  norm_num

end fabric_area_l1077_107746


namespace pat_moved_chairs_l1077_107731

theorem pat_moved_chairs (total_chairs : ℕ) (carey_moved : ℕ) (left_to_move : ℕ) (pat_moved : ℕ) :
  total_chairs = 74 →
  carey_moved = 28 →
  left_to_move = 17 →
  pat_moved = total_chairs - left_to_move - carey_moved →
  pat_moved = 29 :=
by
  intros h_total h_carey h_left h_equation
  rw [h_total, h_carey, h_left] at h_equation
  exact h_equation

end pat_moved_chairs_l1077_107731


namespace avg_salary_rest_of_workers_l1077_107799

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_technicians : ℝ) (total_workers : ℕ) (n_technicians : ℕ) (avg_rest : ℝ) :
  avg_all = 8000 ∧ avg_technicians = 20000 ∧ total_workers = 49 ∧ n_technicians = 7 →
  avg_rest = 6000 :=
by
  sorry

end avg_salary_rest_of_workers_l1077_107799


namespace area_of_parallelogram_l1077_107797

theorem area_of_parallelogram
  (angle_deg : ℝ := 150)
  (side1 : ℝ := 10)
  (side2 : ℝ := 20)
  (adj_angle_deg : ℝ := 180 - angle_deg)
  (angle_rad : ℝ := (adj_angle_deg * Real.pi) / 180) :
  let height := side1 * (Real.sqrt 3 / 2)
  let area := side2 * height
  area = 100 * Real.sqrt 3 :=
by
  /- Proof skipped -/
  sorry

end area_of_parallelogram_l1077_107797


namespace area_excluding_garden_proof_l1077_107783

noncomputable def area_land_excluding_garden (length width r : ℝ) : ℝ :=
  let area_rec := length * width
  let area_circle := Real.pi * (r ^ 2)
  area_rec - area_circle

theorem area_excluding_garden_proof :
  area_land_excluding_garden 8 12 3 = 96 - 9 * Real.pi :=
by
  unfold area_land_excluding_garden
  sorry

end area_excluding_garden_proof_l1077_107783


namespace age_of_B_l1077_107768

variable (A B C : ℕ)

theorem age_of_B (h1 : A + B + C = 84) (h2 : A + C = 58) : B = 26 := by
  sorry

end age_of_B_l1077_107768


namespace Johnny_is_8_l1077_107725

-- Define Johnny's current age
def johnnys_age (x : ℕ) : Prop :=
  x + 2 = 2 * (x - 3)

theorem Johnny_is_8 (x : ℕ) (h : johnnys_age x) : x = 8 :=
sorry

end Johnny_is_8_l1077_107725


namespace solve_for_a_l1077_107704

theorem solve_for_a (a : Real) (h_pos : a > 0) (h_eq : (fun x => x^2 + 4) ((fun x => x^2 - 2) a) = 18) : 
  a = Real.sqrt (Real.sqrt 14 + 2) := by 
  sorry

end solve_for_a_l1077_107704


namespace intersection_is_empty_l1077_107794

-- Define the domain and range sets
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | 0 < x}

-- The Lean theorem to prove that the intersection of A and B is the empty set
theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end intersection_is_empty_l1077_107794


namespace four_edge_trips_count_l1077_107756

-- Defining points and edges of the cube
inductive Point
| A | B | C | D | E | F | G | H

open Point

-- Edges of the cube are connections between points
def Edge (p1 p2 : Point) : Prop :=
  ∃ (edges : List (Point × Point)), 
    edges = [(A, B), (A, D), (A, E), (B, C), (B, E), (B, F), (C, D), (C, F), (C, G), (D, E), (D, F), (D, H), (E, F), (E, H), (F, G), (F, H), (G, H)] ∧ 
    ((p1, p2) ∈ edges ∨ (p2, p1) ∈ edges)

-- Define the proof statement
theorem four_edge_trips_count : 
  ∃ (num_paths : ℕ), num_paths = 12 :=
sorry

end four_edge_trips_count_l1077_107756


namespace icosahedron_path_count_l1077_107759

noncomputable def icosahedron_paths : ℕ := 
  sorry

theorem icosahedron_path_count : icosahedron_paths = 45 :=
  sorry

end icosahedron_path_count_l1077_107759


namespace Emily_sixth_score_l1077_107798

theorem Emily_sixth_score :
  let scores := [91, 94, 88, 90, 101]
  let current_sum := scores.sum
  let desired_average := 95
  let num_quizzes := 6
  let total_score_needed := num_quizzes * desired_average
  let sixth_score := total_score_needed - current_sum
  sixth_score = 106 :=
by
  sorry

end Emily_sixth_score_l1077_107798


namespace correct_proposition_l1077_107781

theorem correct_proposition (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end correct_proposition_l1077_107781


namespace no_solution_iff_n_eq_neg2_l1077_107736

noncomputable def has_no_solution (n : ℝ) : Prop :=
  ∀ x y z : ℝ, ¬ (n * x + y + z = 2 ∧ 
                  x + n * y + z = 2 ∧ 
                  x + y + n * z = 2)

theorem no_solution_iff_n_eq_neg2 (n : ℝ) : has_no_solution n ↔ n = -2 := by
  sorry

end no_solution_iff_n_eq_neg2_l1077_107736


namespace pet_store_dogs_l1077_107780

theorem pet_store_dogs (cats dogs : ℕ) (h1 : 18 = cats) (h2 : 3 * dogs = 4 * cats) : dogs = 24 :=
by
  sorry

end pet_store_dogs_l1077_107780


namespace cost_jam_l1077_107737

noncomputable def cost_of_jam (N B J : ℕ) : ℝ :=
  N * J * 5 / 100

theorem cost_jam (N B J : ℕ) (h₁ : N > 1) (h₂ : 4 * N + 20 = 414) :
  cost_of_jam N B J = 2.25 := by
  sorry

end cost_jam_l1077_107737


namespace coin_value_l1077_107741

variables (n d q : ℕ)  -- Number of nickels, dimes, and quarters
variable (total_coins : n + d + q = 30)  -- Total coins condition

-- Original value in cents
def original_value : ℕ := 5 * n + 10 * d + 25 * q

-- Swapped values in cents
def swapped_value : ℕ := 10 * n + 25 * d + 5 * q

-- Condition given about the value difference
variable (value_difference : swapped_value = original_value + 150)

-- Prove the total value of coins is $5.00 (500 cents)
theorem coin_value : original_value = 500 :=
by
  sorry

end coin_value_l1077_107741


namespace gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l1077_107734

def gervais_distance_miles_per_day : Real := 315
def gervais_days : Real := 3
def gervais_km_per_mile : Real := 1.60934

def henri_total_miles : Real := 1250
def madeleine_distance_miles_per_day : Real := 100
def madeleine_days : Real := 5

def gervais_total_km := gervais_distance_miles_per_day * gervais_days * gervais_km_per_mile
def henri_total_km := henri_total_miles * gervais_km_per_mile
def madeleine_total_km := madeleine_distance_miles_per_day * madeleine_days * gervais_km_per_mile

def combined_total_km := gervais_total_km + henri_total_km + madeleine_total_km

theorem gervais_km_correct : gervais_total_km = 1520.82405 := sorry
theorem henri_km_correct : henri_total_km = 2011.675 := sorry
theorem madeleine_km_correct : madeleine_total_km = 804.67 := sorry
theorem total_km_correct : combined_total_km = 4337.16905 := sorry
theorem henri_drove_farthest : henri_total_km = 2011.675 := sorry

end gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l1077_107734


namespace range_of_m_l1077_107752

def proposition_p (m : ℝ) : Prop := (m^2 - 4 ≥ 0)
def proposition_q (m : ℝ) : Prop := (4 - 4 * m < 0)
def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def not_p (m : ℝ) : Prop := ¬ proposition_p m

theorem range_of_m (m : ℝ) (h1 : p_or_q m) (h2 : not_p m) : 1 < m ∧ m < 2 :=
sorry

end range_of_m_l1077_107752


namespace arithmetic_sequence_difference_l1077_107714

theorem arithmetic_sequence_difference (a d : ℕ) (n m : ℕ) (hnm : m > n) (h_a : a = 3) (h_d : d = 7) (h_n : n = 1001) (h_m : m = 1004) :
  (a + (m - 1) * d) - (a + (n - 1) * d) = 21 :=
by
  sorry

end arithmetic_sequence_difference_l1077_107714


namespace factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l1077_107771

-- Problem 1
theorem factorize_x_squared_minus_4 (x : ℝ) :
  x^2 - 4 = (x + 2) * (x - 2) :=
by { 
  sorry
}

-- Problem 2
theorem factorize_2mx_squared_minus_4mx_plus_2m (x m : ℝ) :
  2 * m * x^2 - 4 * m * x + 2 * m = 2 * m * (x - 1)^2 :=
by { 
  sorry
}

-- Problem 3
theorem factorize_y_quad (y : ℝ) :
  (y^2 - 1)^2 - 6 * (y^2 - 1) + 9 = (y + 2)^2 * (y - 2)^2 :=
by { 
  sorry
}

end factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l1077_107771


namespace stickers_per_page_l1077_107727

theorem stickers_per_page (total_pages total_stickers : ℕ) (h1 : total_pages = 22) (h2 : total_stickers = 220) : (total_stickers / total_pages) = 10 :=
by
  sorry

end stickers_per_page_l1077_107727


namespace min_value_a_plus_b_l1077_107796

theorem min_value_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Real.sqrt (3^a * 3^b) = 3^((a + b) / 2)) : a + b = 4 := by
  sorry

end min_value_a_plus_b_l1077_107796


namespace right_triangle_power_inequality_l1077_107791

theorem right_triangle_power_inequality {a b c x : ℝ} (hpos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a^2 = b^2 + c^2) (h_longest : a > b ∧ a > c) :
  (x > 2) → (a^x > b^x + c^x) :=
by sorry

end right_triangle_power_inequality_l1077_107791


namespace permits_increase_l1077_107713

theorem permits_increase :
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  new_permits = 67600 * old_permits :=
by
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  exact sorry

end permits_increase_l1077_107713


namespace taller_pot_shadow_length_l1077_107740

theorem taller_pot_shadow_length
  (height1 shadow1 height2 : ℝ)
  (h1 : height1 = 20)
  (h2 : shadow1 = 10)
  (h3 : height2 = 40) :
  ∃ shadow2 : ℝ, height2 / shadow2 = height1 / shadow1 ∧ shadow2 = 20 :=
by
  -- Since Lean requires proofs for existential statements,
  -- we add "sorry" to skip the proof.
  sorry

end taller_pot_shadow_length_l1077_107740


namespace cube_edge_length_l1077_107745

theorem cube_edge_length (sum_edges length_edge : ℝ) (cube_has_12_edges : 12 * length_edge = sum_edges) (sum_edges_eq_144 : sum_edges = 144) : length_edge = 12 :=
by
  sorry

end cube_edge_length_l1077_107745
