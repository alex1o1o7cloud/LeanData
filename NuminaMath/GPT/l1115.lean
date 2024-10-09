import Mathlib

namespace resistor_problem_l1115_111592

theorem resistor_problem 
  {x y r : ℝ}
  (h1 : 1 / r = 1 / x + 1 / y)
  (h2 : r = 2.9166666666666665)
  (h3 : y = 7) : 
  x = 5 :=
by
  sorry

end resistor_problem_l1115_111592


namespace manuscript_fee_3800_l1115_111509

theorem manuscript_fee_3800 (tax_fee manuscript_fee : ℕ) 
  (h1 : tax_fee = 420) 
  (h2 : (0 < manuscript_fee) ∧ 
        (manuscript_fee ≤ 4000) → 
        tax_fee = (14 * (manuscript_fee - 800)) / 100) 
  (h3 : (manuscript_fee > 4000) → 
        tax_fee = (11 * manuscript_fee) / 100) : manuscript_fee = 3800 :=
by
  sorry

end manuscript_fee_3800_l1115_111509


namespace negative_values_count_l1115_111576

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l1115_111576


namespace find_k_l1115_111598

-- Definitions for the vectors and collinearity condition.

def vector := ℝ × ℝ

def collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Given vectors a and b.
def a (k : ℝ) : vector := (1, k)
def b : vector := (2, 2)

-- Vector addition.
def add (v1 v2 : vector) : vector := (v1.1 + v2.1, v1.2 + v2.2)

-- Problem statement
theorem find_k (k : ℝ) (h : collinear (add (a k) b) (a k)) : k = 1 :=
by
  sorry

end find_k_l1115_111598


namespace larrys_correct_substitution_l1115_111548

noncomputable def lucky_larry_expression (a b c d e f : ℤ) : ℤ :=
  a + (b - (c + (d - (e + f))))

noncomputable def larrys_substitution (a b c d e f : ℤ) : ℤ :=
  a + b - c + d - e + f

theorem larrys_correct_substitution : 
  (lucky_larry_expression 2 4 6 8 e 5 = larrys_substitution 2 4 6 8 e 5) ↔ (e = 8) :=
by
  sorry

end larrys_correct_substitution_l1115_111548


namespace common_ratio_arithmetic_progression_l1115_111580

theorem common_ratio_arithmetic_progression (a3 q : ℝ) (h1 : a3 = 9) (h2 : a3 + a3 * q + 9 = 27) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end common_ratio_arithmetic_progression_l1115_111580


namespace zoey_finishes_on_monday_l1115_111560

def total_reading_days (books : ℕ) : ℕ :=
  (books * (books + 1)) / 2 + books

def day_of_week (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem zoey_finishes_on_monday : 
  day_of_week 2 (total_reading_days 20) = 1 :=
by
  -- Definitions
  let books := 20
  let start_day := 2 -- Corresponding to Tuesday
  let days := total_reading_days books
  
  -- Prove day_of_week 2 (total_reading_days 20) = 1
  sorry

end zoey_finishes_on_monday_l1115_111560


namespace solve_for_a_l1115_111543

theorem solve_for_a (x a : ℝ) (h1 : x + 2 * a - 6 = 0) (h2 : x = -2) : a = 4 :=
by
  sorry

end solve_for_a_l1115_111543


namespace find_valid_pairs_l1115_111502

-- Definitions and conditions:
def satisfies_equation (a b : ℤ) : Prop := a^2 + a * b - b = 2018

-- Correct answers:
def valid_pairs : List (ℤ × ℤ) :=
  [(2, 2014), (0, -2018), (2018, -2018), (-2016, 2014)]

-- Statement to prove:
theorem find_valid_pairs :
  ∀ (a b : ℤ), satisfies_equation a b ↔ (a, b) ∈ valid_pairs.toFinset := by
  sorry

end find_valid_pairs_l1115_111502


namespace Lynne_bought_3_magazines_l1115_111535

open Nat

def books_about_cats : Nat := 7
def books_about_solar_system : Nat := 2
def book_cost : Nat := 7
def magazine_cost : Nat := 4
def total_spent : Nat := 75

theorem Lynne_bought_3_magazines:
  let total_books := books_about_cats + books_about_solar_system
  let total_cost_books := total_books * book_cost
  let total_cost_magazines := total_spent - total_cost_books
  total_cost_magazines / magazine_cost = 3 :=
by sorry

end Lynne_bought_3_magazines_l1115_111535


namespace range_of_a_plus_2014b_l1115_111531

theorem range_of_a_plus_2014b (a b : ℝ) (h1 : a < b) (h2 : |(Real.log a) / (Real.log 2)| = |(Real.log b) / (Real.log 2)|) :
  ∃ c : ℝ, c > 2015 ∧ ∀ x : ℝ, a + 2014 * b = x → x > 2015 := by
  sorry

end range_of_a_plus_2014b_l1115_111531


namespace odd_numbers_not_dividing_each_other_l1115_111571

theorem odd_numbers_not_dividing_each_other (n : ℕ) (hn : n ≥ 4) :
  ∃ (a b : ℕ), a ≠ b ∧ (2 ^ (2 * n) < a ∧ a < 2 ^ (3 * n)) ∧ 
  (2 ^ (2 * n) < b ∧ b < 2 ^ (3 * n)) ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ 
  ¬ (a ∣ b * b) ∧ ¬ (b ∣ a * a) := by
sorry

end odd_numbers_not_dividing_each_other_l1115_111571


namespace common_ratio_l1115_111522

variable {a : ℕ → ℝ} -- Define a as a sequence of real numbers

-- Define the conditions as hypotheses
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

variables (q : ℝ) (h1 : a 2 = 2) (h2 : a 5 = 1 / 4)

-- Define the theorem to prove the common ratio
theorem common_ratio (h_geom : is_geometric_sequence a q) : q = 1 / 2 :=
  sorry

end common_ratio_l1115_111522


namespace coins_remainder_l1115_111591

theorem coins_remainder 
  (n : ℕ)
  (h₁ : n % 8 = 6)
  (h₂ : n % 7 = 2)
  (h₃ : n = 30) :
  n % 9 = 3 :=
sorry

end coins_remainder_l1115_111591


namespace find_couples_l1115_111546

theorem find_couples (n p q : ℕ) (hn : 0 < n) (hp : 0 < p) (hq : 0 < q)
    (h_gcd : Nat.gcd p q = 1)
    (h_eq : p + q^2 = (n^2 + 1) * p^2 + q) : 
    (p = n + 1 ∧ q = n^2 + n + 1) :=
by 
  sorry

end find_couples_l1115_111546


namespace arithmetic_progression_rth_term_l1115_111588

theorem arithmetic_progression_rth_term (S : ℕ → ℕ) (hS : ∀ n, S n = 5 * n + 4 * n ^ 2) 
  (r : ℕ) : S r - S (r - 1) = 8 * r + 1 :=
by
  sorry

end arithmetic_progression_rth_term_l1115_111588


namespace inversely_proportional_y_l1115_111512

theorem inversely_proportional_y (k : ℚ) (x y : ℚ) (hx_neg_10 : x = -10) (hy_5 : y = 5) (hprop : y * x = k) (hx_neg_4 : x = -4) : 
  y = 25 / 2 := 
by
  sorry

end inversely_proportional_y_l1115_111512


namespace no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l1115_111503

theorem no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100 :
  ¬ ∃ (a b c d : ℕ), a + b + c + d = 2^100 ∧ a * b * c * d = 17^100 :=
by
  sorry

end no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l1115_111503


namespace probability_of_perpendicular_edges_l1115_111550

def is_perpendicular_edge (e1 e2 : ℕ) : Prop :=
-- Define the logic for identifying perpendicular edges here
sorry

def total_outcomes : ℕ := 81

def favorable_outcomes : ℕ :=
-- Calculate the number of favorable outcomes here
20 + 6 + 18

theorem probability_of_perpendicular_edges : 
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 44 / 81 := by
-- Proof for calculating the probability
sorry

end probability_of_perpendicular_edges_l1115_111550


namespace max_value_AMC_l1115_111590

theorem max_value_AMC (A M C : ℕ) (h : A + M + C = 15) : 
  2 * (A * M * C) + A * M + M * C + C * A ≤ 325 := 
sorry

end max_value_AMC_l1115_111590


namespace algebraic_identity_example_l1115_111567

-- Define the variables a and b
def a : ℕ := 287
def b : ℕ := 269

-- State the problem and the expected result
theorem algebraic_identity_example :
  a * a + b * b - 2 * a * b = 324 :=
by
  -- Since the proof is not required, we insert sorry here
  sorry

end algebraic_identity_example_l1115_111567


namespace math_problem_l1115_111553

theorem math_problem (f : ℕ → Prop) (m : ℕ) 
  (h1 : f 1) (h2 : f 2) (h3 : f 3)
  (h_implies : ∀ k : ℕ, f k → f (k + m)) 
  (h_max : m = 3):
  ∀ n : ℕ, 0 < n → f n :=
by
  sorry

end math_problem_l1115_111553


namespace total_pins_cardboard_l1115_111565

theorem total_pins_cardboard {length width pins : ℕ} (h_length : length = 34) (h_width : width = 14) (h_pins : pins = 35) :
  2 * pins * (length + width) / (length + width) = 140 :=
by
  sorry

end total_pins_cardboard_l1115_111565


namespace total_get_well_cards_l1115_111527

-- Definitions for the number of cards received in each place
def cardsInHospital : ℕ := 403
def cardsAtHome : ℕ := 287

-- Theorem statement:
theorem total_get_well_cards : cardsInHospital + cardsAtHome = 690 := by
  sorry

end total_get_well_cards_l1115_111527


namespace smallest_integer_switch_add_l1115_111575

theorem smallest_integer_switch_add (a b: ℕ) (h1: n = 10 * a + b) 
  (h2: 3 * n = 10 * b + a + 5)
  (h3: 0 ≤ b) (h4: b < 10) (h5: 1 ≤ a) (h6: a < 10): n = 47 :=
by
  sorry

end smallest_integer_switch_add_l1115_111575


namespace find_m_l1115_111593

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9

def symmetric_line (x y m : ℝ) : Prop := x + m * y + 4 = 0

theorem find_m (m : ℝ) (h1 : circle_equation (-1) 3) (h2 : symmetric_line (-1) 3 m) : m = -1 := by
  sorry

end find_m_l1115_111593


namespace min_value_eq_18sqrt3_l1115_111547

noncomputable def min_value (x y : ℝ) (h : x + y = 5) : ℝ := 3^x + 3^y

theorem min_value_eq_18sqrt3 {x y : ℝ} (h : x + y = 5) : min_value x y h ≥ 18 * Real.sqrt 3 := 
sorry

end min_value_eq_18sqrt3_l1115_111547


namespace scale_drawing_represents_line_segment_l1115_111530

-- Define the given conditions
def scale_factor : ℝ := 800
def line_segment_length_inch : ℝ := 4.75

-- Prove the length in feet
theorem scale_drawing_represents_line_segment :
  line_segment_length_inch * scale_factor = 3800 :=
by
  sorry

end scale_drawing_represents_line_segment_l1115_111530


namespace geometric_sequence_alpha5_eq_three_l1115_111523

theorem geometric_sequence_alpha5_eq_three (α : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, α (n + 1) = α n * r) 
  (h2 : α 4 * α 5 * α 6 = 27) : α 5 = 3 := 
by
  sorry

end geometric_sequence_alpha5_eq_three_l1115_111523


namespace prob_at_least_one_wrong_l1115_111563

-- Defining the conditions in mathlib
def prob_wrong : ℝ := 0.1
def num_questions : ℕ := 3

-- Proving the main statement
theorem prob_at_least_one_wrong : 1 - (1 - prob_wrong) ^ num_questions = 0.271 := by
  sorry

end prob_at_least_one_wrong_l1115_111563


namespace plane_intersect_probability_l1115_111578

-- Define the vertices of the rectangular prism
def vertices : List (ℝ × ℝ × ℝ) := 
  [(0,0,0), (2,0,0), (2,2,0), (0,2,0), 
   (0,0,1), (2,0,1), (2,2,1), (0,2,1)]

-- Calculate total number of ways to choose 3 vertices out of 8
def total_ways : ℕ := Nat.choose 8 3

-- Calculate the number of planes that do not intersect the interior of the prism
def non_intersecting_planes : ℕ := 6 * Nat.choose 4 3

-- Calculate the probability as a fraction
def probability_of_intersecting (total non_intersecting : ℕ) : ℚ :=
  1 - (non_intersecting : ℚ) / (total : ℚ)

-- The main theorem to state the probability is 4/7
theorem plane_intersect_probability : 
  probability_of_intersecting total_ways non_intersecting_planes = 4 / 7 := 
  by
    -- Skipping the proof
    sorry

end plane_intersect_probability_l1115_111578


namespace factorization_theorem_l1115_111596

-- Define the polynomial p(x, y)
def p (x y k : ℝ) : ℝ := x^2 - 2*x*y + k*y^2 + 3*x - 5*y + 2

-- Define the condition for factorization into two linear factors
def can_be_factored (x y m n : ℝ) : Prop :=
  (p x y (m * n)) = ((x + m * y + 1) * (x + n * y + 2))

-- The main theorem proving that k = -3 is the value for factorizability
theorem factorization_theorem (k : ℝ) : (∃ m n : ℝ, can_be_factored x y m n) ↔ k = -3 := by sorry

end factorization_theorem_l1115_111596


namespace intercept_sum_l1115_111513

-- Define the equation of the line and the condition on the intercepts.
theorem intercept_sum (c : ℚ) (x y : ℚ) (h1 : 3 * x + 5 * y + c = 0) (h2 : x + y = 55/4) : 
  c = 825/32 :=
sorry

end intercept_sum_l1115_111513


namespace maximum_smallest_triplet_sum_l1115_111525

theorem maximum_smallest_triplet_sum (circle : Fin 10 → ℕ) (h : ∀ i : Fin 10, 1 ≤ circle i ∧ circle i ≤ 10 ∧ ∀ j k, j ≠ k → circle j ≠ circle k):
  ∃ (i : Fin 10), ∀ j ∈ ({i, i + 1, i + 2} : Finset (Fin 10)), circle i + circle (i + 1) + circle (i + 2) ≤ 15 :=
sorry

end maximum_smallest_triplet_sum_l1115_111525


namespace largest_possible_number_of_pencils_in_a_box_l1115_111561

/-- Olivia bought 48 pencils -/
def olivia_pencils : ℕ := 48
/-- Noah bought 60 pencils -/
def noah_pencils : ℕ := 60
/-- Liam bought 72 pencils -/
def liam_pencils : ℕ := 72

/-- The GCD of the number of pencils bought by Olivia, Noah, and Liam is 12 -/
theorem largest_possible_number_of_pencils_in_a_box :
  gcd olivia_pencils (gcd noah_pencils liam_pencils) = 12 :=
by {
  sorry
}

end largest_possible_number_of_pencils_in_a_box_l1115_111561


namespace problem_f_2009_plus_f_2010_l1115_111510

theorem problem_f_2009_plus_f_2010 (f : ℝ → ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (2 * x + 1) = f (2 * (x + 5 / 2) + 1))
  (h_f1 : f 1 = 5) :
  f 2009 + f 2010 = 0 :=
sorry

end problem_f_2009_plus_f_2010_l1115_111510


namespace original_group_size_l1115_111504

theorem original_group_size (M : ℕ) (R : ℕ) :
  (M * R * 40 = (M - 5) * R * 50) → M = 25 :=
by
  sorry

end original_group_size_l1115_111504


namespace point_transformation_l1115_111514

theorem point_transformation : ∀ (P : ℝ×ℝ), P = (1, -2) → P = (-1, 2) :=
by
  sorry

end point_transformation_l1115_111514


namespace find_ab_minus_a_neg_b_l1115_111562

variable (a b : ℝ)
variables (h₀ : a > 1) (h₁ : b > 0) (h₂ : a^b + a^(-b) = 2 * Real.sqrt 2)

theorem find_ab_minus_a_neg_b : a^b - a^(-b) = 2 := by
  sorry

end find_ab_minus_a_neg_b_l1115_111562


namespace renovation_days_l1115_111520

/-
Conditions:
1. Cost to hire a company: 50000 rubles
2. Cost of buying materials: 20000 rubles
3. Husband's daily wage: 2000 rubles
4. Wife's daily wage: 1500 rubles
Question:
How many workdays can they spend on the renovation to make it more cost-effective?
-/

theorem renovation_days (cost_hire_company cost_materials : ℕ) 
  (husband_daily_wage wife_daily_wage : ℕ) 
  (more_cost_effective_days : ℕ) :
  cost_hire_company = 50000 → 
  cost_materials = 20000 → 
  husband_daily_wage = 2000 → 
  wife_daily_wage = 1500 → 
  more_cost_effective_days = 8 :=
by
  intros
  sorry

end renovation_days_l1115_111520


namespace ratio_malt_to_coke_l1115_111581

-- Definitions from conditions
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_choose_malt : ℕ := 6
def females_choose_malt : ℕ := 8

-- Derived values
def total_cheerleaders : ℕ := total_males + total_females
def total_malt : ℕ := males_choose_malt + females_choose_malt
def total_coke : ℕ := total_cheerleaders - total_malt

-- The theorem to be proved
theorem ratio_malt_to_coke : (total_malt / total_coke) = (7 / 6) :=
  by
    -- skipped proof
    sorry

end ratio_malt_to_coke_l1115_111581


namespace gcd_sum_lcm_eq_gcd_l1115_111594

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
by 
  sorry

end gcd_sum_lcm_eq_gcd_l1115_111594


namespace workload_increase_l1115_111557

theorem workload_increase (a b c d p : ℕ) (h : p ≠ 0) :
  let total_workload := a + b + c + d
  let workload_per_worker := total_workload / p
  let absent_workers := p / 4
  let remaining_workers := p - absent_workers
  let workload_per_remaining_worker := total_workload / (3 * p / 4)
  workload_per_remaining_worker = (a + b + c + d) * 4 / (3 * p) :=
by
  sorry

end workload_increase_l1115_111557


namespace total_coins_is_twenty_l1115_111518

def piles_of_quarters := 2
def piles_of_dimes := 3
def coins_per_pile := 4

theorem total_coins_is_twenty : piles_of_quarters * coins_per_pile + piles_of_dimes * coins_per_pile = 20 :=
by sorry

end total_coins_is_twenty_l1115_111518


namespace determine_omega_phi_l1115_111529

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem determine_omega_phi (ω φ : ℝ) (x : ℝ)
  (h₁ : 0 < ω) (h₂ : |φ| < Real.pi)
  (h₃ : f ω φ (5 * Real.pi / 8) = 2)
  (h₄ : f ω φ (11 * Real.pi / 8) = 0)
  (h₅ : (2 * Real.pi / ω) > 2 * Real.pi) :
  ω = 2 / 3 ∧ φ = Real.pi / 12 :=
sorry

end determine_omega_phi_l1115_111529


namespace charlie_cookies_l1115_111532

theorem charlie_cookies (father_cookies mother_cookies total_cookies charlie_cookies : ℕ)
  (h1 : father_cookies = 10) (h2 : mother_cookies = 5) (h3 : total_cookies = 30) :
  father_cookies + mother_cookies + charlie_cookies = total_cookies → charlie_cookies = 15 :=
by
  intros h
  sorry

end charlie_cookies_l1115_111532


namespace polynomial_divisibility_a_l1115_111597

theorem polynomial_divisibility_a (n : ℕ) : 
  (n % 3 = 1 ∨ n % 3 = 2) ↔ (x^2 + x + 1 ∣ x^(2*n) + x^n + 1) :=
sorry

end polynomial_divisibility_a_l1115_111597


namespace natural_numbers_fitting_description_l1115_111556

theorem natural_numbers_fitting_description (n : ℕ) (h : 1 / (n : ℚ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) : n = 2 ∨ n = 3 :=
by
  sorry

end natural_numbers_fitting_description_l1115_111556


namespace smith_family_mean_age_l1115_111569

theorem smith_family_mean_age :
  let children_ages := [8, 8, 8, 12, 11]
  let dogs_ages := [3, 4]
  let all_ages := children_ages ++ dogs_ages
  let total_ages := List.sum all_ages
  let total_individuals := List.length all_ages
  (total_ages : ℚ) / (total_individuals : ℚ) = 7.71 :=
by
  sorry

end smith_family_mean_age_l1115_111569


namespace alice_bob_total_dollars_l1115_111519

-- Define Alice's amount in dollars
def alice_amount : ℚ := 5 / 8

-- Define Bob's amount in dollars
def bob_amount : ℚ := 3 / 5

-- Define the total amount in dollars
def total_amount : ℚ := alice_amount + bob_amount

theorem alice_bob_total_dollars : (alice_amount + bob_amount : ℚ) = 1.225 := by
    sorry

end alice_bob_total_dollars_l1115_111519


namespace jellybean_count_l1115_111577

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l1115_111577


namespace simplify_expression_l1115_111515

variable (b : ℝ)

theorem simplify_expression (h : b ≠ 2) : (2 - 1 / (1 + b / (2 - b))) = 1 + b / 2 := 
sorry

end simplify_expression_l1115_111515


namespace count_integer_points_l1115_111589

-- Define the conditions: the parabola P with focus at (0,0) and passing through (6,4) and (-6,-4)
def parabola (P : ℝ × ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
  (∀ x y : ℝ, P (x, y) ↔ y = a*x^2 + b) ∧ 
  P (6, 4) ∧ P (-6, -4)

-- Define the main theorem to be proved: the count of integer points satisfying the inequality
theorem count_integer_points (P : ℝ × ℝ → Prop) (hP : parabola P) :
  ∃ n : ℕ, n = 45 ∧ ∀ (x y : ℤ), P (x, y) → |6 * x + 4 * y| ≤ 1200 :=
sorry

end count_integer_points_l1115_111589


namespace simplify_expression_l1115_111572

variable (x : ℝ)

theorem simplify_expression : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x :=
by
  sorry

end simplify_expression_l1115_111572


namespace equation_of_line_l1115_111507

theorem equation_of_line (x_intercept slope : ℝ)
  (hx : x_intercept = 2) (hm : slope = 1) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -1 ∧ c = -2 ∧ (∀ x y : ℝ, y = slope * (x - x_intercept) ↔ a * x + b * y + c = 0) := sorry

end equation_of_line_l1115_111507


namespace distance_focus_to_asymptote_of_hyperbola_l1115_111537

open Real

noncomputable def distance_from_focus_to_asymptote_of_hyperbola : ℝ :=
  let a := 2
  let b := 1
  let c := sqrt (a^2 + b^2)
  let foci1 := (sqrt (a^2 + b^2), 0)
  let foci2 := (-sqrt (a^2 + b^2), 0)
  let asymptote_slope := a / b
  let distance_formula := (|abs (sqrt 5)|) / (sqrt (1 + asymptote_slope^2))
  distance_formula

theorem distance_focus_to_asymptote_of_hyperbola :
  distance_from_focus_to_asymptote_of_hyperbola = 1 :=
sorry

end distance_focus_to_asymptote_of_hyperbola_l1115_111537


namespace intersection_is_ge_negative_one_l1115_111542

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4*x + 3}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem intersection_is_ge_negative_one : M ∩ N = {y | y ≥ -1} := by
  sorry

end intersection_is_ge_negative_one_l1115_111542


namespace sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1_sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1_l1115_111583

open Real

-- Problem (a)
theorem sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1 (n k : Nat) :
  (sqrt 2 - 1)^n = sqrt k - sqrt (k - 1) :=
sorry

-- Problem (b)
theorem sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1 (m n k : Nat) :
  (sqrt m - sqrt (m - 1))^n = sqrt k - sqrt (k - 1) :=
sorry

end sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1_sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1_l1115_111583


namespace sum_of_extreme_values_l1115_111582

theorem sum_of_extreme_values (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (5 - Real.sqrt 34) / 3
  let M := (5 + Real.sqrt 34) / 3
  m + M = 10 / 3 :=
by
  sorry

end sum_of_extreme_values_l1115_111582


namespace problem_statement_l1115_111570

theorem problem_statement (h : 36 = 6^2) : 6^15 / 36^5 = 7776 := by
  sorry

end problem_statement_l1115_111570


namespace union_of_A_and_B_l1115_111517

-- Definitions for sets A and B
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- Theorem statement to prove the union of A and B
theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := by
  sorry

end union_of_A_and_B_l1115_111517


namespace total_daisies_l1115_111599

theorem total_daisies (white pink red : ℕ) (h1 : pink = 9 * white) (h2 : red = 4 * pink - 3) (h3 : white = 6) : 
    white + pink + red = 273 :=
by
  sorry

end total_daisies_l1115_111599


namespace color_opposite_lightgreen_is_red_l1115_111555

-- Define the colors
inductive Color
| Red | White | Green | Brown | LightGreen | Purple

open Color

-- Define the condition
def is_opposite (a b : Color) : Prop := sorry

-- Main theorem
theorem color_opposite_lightgreen_is_red :
  is_opposite LightGreen Red :=
sorry

end color_opposite_lightgreen_is_red_l1115_111555


namespace find_a_l1115_111516

theorem find_a (a : ℝ) : 
  (∃ r : ℕ, (10 - 3 * r = 1 ∧ (-a)^r * (Nat.choose 5 r) *  x^(10 - 2 * r - r) = x ∧ -10 = (-a)^3 * (Nat.choose 5 3)))
  → a = 1 :=
sorry

end find_a_l1115_111516


namespace ratio_c_d_l1115_111564

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x - 2 * y = c)
  (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : c / d = -1 / 3 := by
  sorry

end ratio_c_d_l1115_111564


namespace total_cost_of_cultivating_field_l1115_111534

theorem total_cost_of_cultivating_field 
  (base height : ℕ) 
  (cost_per_hectare : ℝ) 
  (base_eq: base = 3 * height) 
  (height_eq: height = 300) 
  (cost_eq: cost_per_hectare = 24.68) 
  : (1/2 : ℝ) * base * height / 10000 * cost_per_hectare = 333.18 :=
by
  sorry

end total_cost_of_cultivating_field_l1115_111534


namespace xy_sum_is_one_l1115_111505

theorem xy_sum_is_one (x y : ℝ) (h : x^2 + y^2 + x * y = 12 * x - 8 * y + 2) : x + y = 1 :=
sorry

end xy_sum_is_one_l1115_111505


namespace boris_number_of_bowls_l1115_111549

-- Definitions from the conditions
def total_candies : ℕ := 100
def daughter_eats : ℕ := 8
def candies_per_bowl_after_removal : ℕ := 20
def candies_removed_per_bowl : ℕ := 3

-- Derived definitions
def remaining_candies : ℕ := total_candies - daughter_eats
def candies_per_bowl_orig : ℕ := candies_per_bowl_after_removal + candies_removed_per_bowl

-- Statement to prove
theorem boris_number_of_bowls : remaining_candies / candies_per_bowl_orig = 4 :=
by sorry

end boris_number_of_bowls_l1115_111549


namespace lemonade_water_cups_l1115_111573

theorem lemonade_water_cups
  (W S L : ℕ)
  (h1 : W = 5 * S)
  (h2 : S = 3 * L)
  (h3 : L = 5) :
  W = 75 :=
by {
  sorry
}

end lemonade_water_cups_l1115_111573


namespace smallest_mn_sum_l1115_111551

theorem smallest_mn_sum (m n : ℕ) (h : 3 * n ^ 3 = 5 * m ^ 2) : m + n = 60 :=
sorry

end smallest_mn_sum_l1115_111551


namespace integer_part_of_result_is_40_l1115_111568

noncomputable def numerator : ℝ := 0.1 + 1.2 + 2.3 + 3.4 + 4.5 + 5.6 + 6.7 + 7.8 + 8.9
noncomputable def denominator : ℝ := 0.01 + 0.03 + 0.05 + 0.07 + 0.09 + 0.11 + 0.13 + 0.15 + 0.17 + 0.19
noncomputable def result : ℝ := numerator / denominator

theorem integer_part_of_result_is_40 : ⌊result⌋ = 40 := 
by
  -- proof goes here
  sorry

end integer_part_of_result_is_40_l1115_111568


namespace division_of_polynomials_l1115_111552

theorem division_of_polynomials (a b : ℝ) :
  (18 * a^2 * b - 9 * a^5 * b^2) / (-3 * a * b) = -6 * a + 3 * a^4 * b :=
by
  sorry

end division_of_polynomials_l1115_111552


namespace positive_number_divisible_by_4_l1115_111506

theorem positive_number_divisible_by_4 (N : ℕ) (h1 : N % 4 = 0) (h2 : (2 + 4 + N + 3) % 2 = 1) : N = 4 := 
by 
  sorry

end positive_number_divisible_by_4_l1115_111506


namespace inclination_angle_of_line_l1115_111554

theorem inclination_angle_of_line (m : ℝ) (b : ℝ) (h : b = -3) (h_line : ∀ x : ℝ, x - 3 = m * x + b) : 
  (Real.arctan m * 180 / Real.pi) = 45 := 
by sorry

end inclination_angle_of_line_l1115_111554


namespace problem_solution_l1115_111574

noncomputable def complex_expression : ℝ :=
  (-(1/2) * (1/100))^5 * ((2/3) * (2/100))^4 * (-(3/4) * (3/100))^3 * ((4/5) * (4/100))^2 * (-(5/6) * (5/100)) * 10^30

theorem problem_solution : complex_expression = -48 :=
by
  sorry

end problem_solution_l1115_111574


namespace probability_of_first_three_red_cards_l1115_111526

theorem probability_of_first_three_red_cards :
  let total_cards := 60
  let red_cards := 36
  let black_cards := total_cards - red_cards
  let total_ways := total_cards * (total_cards - 1) * (total_cards - 2)
  let red_ways := red_cards * (red_cards - 1) * (red_cards - 2)
  (red_ways / total_ways) = 140 / 673 :=
by
  sorry

end probability_of_first_three_red_cards_l1115_111526


namespace correct_exponent_calculation_l1115_111538

theorem correct_exponent_calculation (x : ℝ) : (-x^3)^4 = x^12 := 
by sorry

end correct_exponent_calculation_l1115_111538


namespace theater_ticket_sales_l1115_111521

theorem theater_ticket_sales (R H : ℕ) (h1 : R = 25) (h2 : H = 3 * R + 18) : H = 93 :=
by
  sorry

end theater_ticket_sales_l1115_111521


namespace fraction_equation_solution_l1115_111595

theorem fraction_equation_solution (x y : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 5) (hy1 : y ≠ 0) (hy2 : y ≠ 7)
  (h : (3 / x) + (2 / y) = 1 / 3) : 
  x = (9 * y) / (y - 6) :=
sorry

end fraction_equation_solution_l1115_111595


namespace proof_problem_l1115_111579

variable {a b : ℤ}

theorem proof_problem (h1 : ∃ k : ℤ, a = 4 * k) (h2 : ∃ l : ℤ, b = 8 * l) : 
  (∃ m : ℤ, b = 4 * m) ∧
  (∃ n : ℤ, a - b = 4 * n) ∧
  (∃ p : ℤ, a + b = 2 * p) := 
by
  sorry

end proof_problem_l1115_111579


namespace new_weights_inequality_l1115_111508

theorem new_weights_inequality (W : ℝ) (x y : ℝ) (h_avg_increase : (8 * W - 2 * 68 + x + y) / 8 = W + 5.5)
  (h_sum_new_weights : x + y ≤ 180) : x > W ∧ y > W :=
by {
  sorry
}

end new_weights_inequality_l1115_111508


namespace papers_left_after_giving_away_l1115_111541

variable (x : ℕ)

-- Given conditions:
def sheets_in_desk : ℕ := 50
def sheets_in_backpack : ℕ := 41
def total_initial_sheets := sheets_in_desk + sheets_in_backpack

-- Prove that Maria has 91 - x sheets left after giving away x sheets
theorem papers_left_after_giving_away (h : total_initial_sheets = 91) : 
  ∀ d b : ℕ, d = sheets_in_desk → b = sheets_in_backpack → 91 - x = total_initial_sheets - x :=
by
  sorry

end papers_left_after_giving_away_l1115_111541


namespace radius_of_cylinder_is_correct_l1115_111528

/-- 
  A right circular cylinder is inscribed in a right circular cone such that:
  - The diameter of the cylinder is equal to its height.
  - The cone has a diameter of 8.
  - The cone has an altitude of 10.
  - The axes of the cylinder and cone coincide.
  Prove that the radius of the cylinder is 20/9.
-/
theorem radius_of_cylinder_is_correct :
  ∀ (r : ℚ), 
    (2 * r = 8 - 2 * r ∧ 10 - 2 * r = (10 / 4) * r) → 
    r = 20 / 9 :=
by
  intro r
  intro h
  sorry

end radius_of_cylinder_is_correct_l1115_111528


namespace shorter_leg_of_right_triangle_l1115_111501

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l1115_111501


namespace cone_volume_l1115_111539

theorem cone_volume (central_angle : ℝ) (sector_area : ℝ) (h1 : central_angle = 120) (h2 : sector_area = 3 * Real.pi) :
  ∃ V : ℝ, V = (2 * Real.sqrt 2 * Real.pi) / 3 :=
by
  -- We acknowledge the input condition where the angle is 120° and sector area is 3π
  -- The problem requires proving the volume of the cone
  sorry

end cone_volume_l1115_111539


namespace females_with_advanced_degrees_l1115_111533

theorem females_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_advanced_degrees : ℕ)
  (males_college_degree_only : ℕ)
  (h1 : total_employees = 200)
  (h2 : total_females = 120)
  (h3 : total_advanced_degrees = 100)
  (h4 : males_college_degree_only = 40) :
  (total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60) :=
by
  -- proof will go here
  sorry

end females_with_advanced_degrees_l1115_111533


namespace original_sticker_price_l1115_111584

theorem original_sticker_price (S : ℝ) (h1 : 0.80 * S - 120 = 0.65 * S - 10) : S = 733 := 
by
  sorry

end original_sticker_price_l1115_111584


namespace inequality_subtraction_l1115_111566

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_l1115_111566


namespace probability_jack_queen_king_l1115_111511

theorem probability_jack_queen_king :
  let deck_size := 52
  let jacks := 4
  let queens := 4
  let kings := 4
  let remaining_after_jack := deck_size - 1
  let remaining_after_queen := deck_size - 2
  (jacks / deck_size) * (queens / remaining_after_jack) * (kings / remaining_after_queen) = 8 / 16575 :=
by
  sorry

end probability_jack_queen_king_l1115_111511


namespace number_of_bead_necklaces_sold_is_3_l1115_111545

-- Definitions of the given conditions
def total_earnings : ℕ := 36
def gemstone_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 6

-- Define the earnings from gemstone necklaces as a separate definition
def earnings_gemstone_necklaces : ℕ := gemstone_necklaces * cost_per_necklace

-- Define the earnings from bead necklaces based on total earnings and earnings from gemstone necklaces
def earnings_bead_necklaces : ℕ := total_earnings - earnings_gemstone_necklaces

-- Define the number of bead necklaces sold
def bead_necklaces_sold : ℕ := earnings_bead_necklaces / cost_per_necklace

-- The theorem we want to prove
theorem number_of_bead_necklaces_sold_is_3 : bead_necklaces_sold = 3 :=
by
  sorry

end number_of_bead_necklaces_sold_is_3_l1115_111545


namespace moles_of_KOH_combined_l1115_111500

theorem moles_of_KOH_combined (H2O_formed : ℕ) (NH4I_used : ℕ) (ratio_KOH_H2O : ℕ) : H2O_formed = 54 → NH4I_used = 3 → ratio_KOH_H2O = 1 → H2O_formed = NH4I_used := 
by 
  intro H2O_formed_eq NH4I_used_eq ratio_eq 
  sorry

end moles_of_KOH_combined_l1115_111500


namespace max_next_person_weight_l1115_111585

def avg_weight_adult := 150
def avg_weight_child := 70
def max_weight_elevator := 1500
def num_adults := 7
def num_children := 5

def total_weight_adults := num_adults * avg_weight_adult
def total_weight_children := num_children * avg_weight_child
def current_weight := total_weight_adults + total_weight_children

theorem max_next_person_weight : 
  max_weight_elevator - current_weight = 100 := 
by 
  sorry

end max_next_person_weight_l1115_111585


namespace admission_price_for_adults_l1115_111586

theorem admission_price_for_adults (A : ℕ) (ticket_price_children : ℕ) (total_children_tickets : ℕ) 
    (total_amount : ℕ) (total_tickets : ℕ) (children_ticket_costs : ℕ) 
    (adult_tickets : ℕ) (adult_ticket_costs : ℕ) :
    ticket_price_children = 5 → 
    total_children_tickets = 21 → 
    total_amount = 201 → 
    total_tickets = 33 → 
    children_ticket_costs = 21 * 5 → 
    adult_tickets = 33 - 21 → 
    adult_ticket_costs = 201 - 21 * 5 → 
    A = (201 - 21 * 5) / (33 - 21) → 
    A = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end admission_price_for_adults_l1115_111586


namespace balloons_initial_count_l1115_111587

theorem balloons_initial_count (B : ℕ) (G : ℕ) : ∃ G : ℕ, B = 7 * G + 4 := sorry

end balloons_initial_count_l1115_111587


namespace ratio_of_ages_l1115_111558

open Real

theorem ratio_of_ages (father_age son_age : ℝ) (h1 : father_age = 45) (h2 : son_age = 15) :
  father_age / son_age = 3 :=
by
  sorry

end ratio_of_ages_l1115_111558


namespace rectangle_area_l1115_111544

theorem rectangle_area (x : ℝ) (w : ℝ) (h : ℝ) (H1 : x^2 = w^2 + h^2) (H2 : h = 3 * w) : 
  (w * h = (3 * x^2) / 10) :=
by sorry

end rectangle_area_l1115_111544


namespace domain_of_function_is_all_real_l1115_111540

def domain_function : Prop :=
  ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 6 ≠ 0

theorem domain_of_function_is_all_real :
  domain_function :=
by
  intros t
  sorry

end domain_of_function_is_all_real_l1115_111540


namespace closest_integer_to_cube_root_of_150_l1115_111524

theorem closest_integer_to_cube_root_of_150 : ∃ (n : ℤ), abs ((n: ℝ)^3 - 150) ≤ abs (((n + 1 : ℤ) : ℝ)^3 - 150) ∧
  abs ((n: ℝ)^3 - 150) ≤ abs (((n - 1 : ℤ) : ℝ)^3 - 150) ∧ n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l1115_111524


namespace cost_of_acai_berry_juice_l1115_111559

theorem cost_of_acai_berry_juice 
  (cost_per_litre_cocktail : ℝ) 
  (cost_per_litre_mixed_fruit : ℝ)
  (volume_mixed_fruit : ℝ)
  (volume_acai_berry : ℝ)
  (total_volume : ℝ) 
  (total_cost_of_mixed_fruit : ℝ)
  (total_cost_cocktail : ℝ)
  : cost_per_litre_cocktail = 1399.45 ∧ 
    cost_per_litre_mixed_fruit = 262.85 ∧ 
    volume_mixed_fruit = 37 ∧ 
    volume_acai_berry = 24.666666666666668 ∧ 
    total_volume = 61.666666666666668 ∧ 
    total_cost_of_mixed_fruit = volume_mixed_fruit * cost_per_litre_mixed_fruit ∧
    total_cost_of_mixed_fruit = 9725.45 ∧
    total_cost_cocktail = total_volume * cost_per_litre_cocktail ∧ 
    total_cost_cocktail = 86327.77 
    → 24.666666666666668 * 3105.99 + 9725.45 = 86327.77 :=
sorry

end cost_of_acai_berry_juice_l1115_111559


namespace train_length_l1115_111536

theorem train_length (L : ℝ) 
  (equal_length : ∀ (A B : ℝ), A = B → L = A)
  (same_direction : ∀ (dir1 dir2 : ℤ), dir1 = 1 → dir2 = 1)
  (speed_faster : ℝ := 50) (speed_slower : ℝ := 36)
  (time_to_pass : ℝ := 36)
  (relative_speed := speed_faster - speed_slower)
  (relative_speed_km_per_sec := relative_speed / 3600)
  (distance_covered := relative_speed_km_per_sec * time_to_pass)
  (total_distance := distance_covered)
  (length_per_train := total_distance / 2)
  (length_in_meters := length_per_train * 1000): 
  L = 70 := 
by 
  sorry

end train_length_l1115_111536
