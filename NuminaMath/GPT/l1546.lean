import Mathlib

namespace part_one_part_two_l1546_154628

noncomputable def f (a x : ℝ) := a * Real.log x - x + 1

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≤ 0) : a = 1 := 
sorry

theorem part_two (h₁ : ∀ x > 0, f 1 x ≤ 0) (x : ℝ) (h₂ : 0 < x) (h₃ : x < Real.pi / 2) :
  Real.exp x * Real.sin x - x > f 1 x :=
sorry

end part_one_part_two_l1546_154628


namespace gcd_lcm_product_75_90_l1546_154649

theorem gcd_lcm_product_75_90 :
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 6750 :=
by
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  sorry

end gcd_lcm_product_75_90_l1546_154649


namespace rectangular_prism_volume_l1546_154634

theorem rectangular_prism_volume 
(l w h : ℝ) 
(h1 : l * w = 18) 
(h2 : w * h = 32) 
(h3 : l * h = 48) : 
l * w * h = 288 :=
sorry

end rectangular_prism_volume_l1546_154634


namespace third_vertex_coordinates_l1546_154605

theorem third_vertex_coordinates (x : ℝ) (h : 6 * |x| = 96) : x = 16 ∨ x = -16 :=
by
  sorry

end third_vertex_coordinates_l1546_154605


namespace profit_percentage_of_cp_is_75_percent_of_sp_l1546_154673

/-- If the cost price (CP) is 75% of the selling price (SP), then the profit percentage is 33.33% -/
theorem profit_percentage_of_cp_is_75_percent_of_sp (SP : ℝ) (h : SP > 0) (CP : ℝ) (hCP : CP = 0.75 * SP) :
  (SP - CP) / CP * 100 = 33.33 :=
by
  sorry

end profit_percentage_of_cp_is_75_percent_of_sp_l1546_154673


namespace squirrel_acorns_left_l1546_154630

noncomputable def acorns_per_winter_month (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) : ℕ :=
  let per_month := total_acorns / months
  let acorns_taken_per_month := acorns_taken_total / months
  per_month - acorns_taken_per_month

theorem squirrel_acorns_left (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) :
  total_acorns = 210 → months = 3 → acorns_taken_total = 30 → acorns_per_winter_month total_acorns months acorns_taken_total = 60 :=
by intros; sorry

end squirrel_acorns_left_l1546_154630


namespace value_of_S_l1546_154682

theorem value_of_S (x R S : ℝ) (h1 : x + 1/x = R) (h2 : R = 6) : x^3 + 1/x^3 = 198 :=
by
  sorry

end value_of_S_l1546_154682


namespace three_scientists_same_topic_l1546_154609

theorem three_scientists_same_topic
  (scientists : Finset ℕ)
  (h_size : scientists.card = 17)
  (topics : Finset ℕ)
  (h_topics : topics.card = 3)
  (communicates : ℕ → ℕ → ℕ)
  (h_communicate : ∀ a b : ℕ, a ≠ b → b ∈ scientists → communicates a b ∈ topics) :
  ∃ (a b c : ℕ), a ∈ scientists ∧ b ∈ scientists ∧ c ∈ scientists ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  communicates a b = communicates b c ∧ communicates b c = communicates a c := 
sorry

end three_scientists_same_topic_l1546_154609


namespace simplify_expression_l1546_154650

theorem simplify_expression (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ 2) (h3 : x ≠ 4) (h4 : x ≠ 5) :
  ( (x^2 - 4*x + 3) / (x^2 - 6*x + 9) ) / ( (x^2 - 6*x + 8) / (x^2 - 8*x + 15) ) =
  ( (x - 1) * (x - 5) ) / ( (x - 3) * (x - 4) * (x - 2) ) :=
by
  sorry

end simplify_expression_l1546_154650


namespace interest_rate_is_10_perc_l1546_154691

noncomputable def interest_rate (P : ℝ) (R : ℝ) (T : ℝ := 2) : ℝ := (P * R * T) / 100

theorem interest_rate_is_10_perc (P : ℝ) : 
  (interest_rate P 10) = P / 5 :=
by
  sorry

end interest_rate_is_10_perc_l1546_154691


namespace swim_team_girls_l1546_154619

-- Definitions using the given conditions
variables (B G : ℕ)
theorem swim_team_girls (h1 : G = 5 * B) (h2 : G + B = 96) : G = 80 :=
sorry

end swim_team_girls_l1546_154619


namespace incorrect_operation_l1546_154629

noncomputable def a : ℤ := -2

def operation_A (a : ℤ) : ℤ := abs a
def operation_B (a : ℤ) : ℤ := abs (a - 2) + abs (a + 1)
def operation_C (a : ℤ) : ℤ := -a ^ 3 + a + (-a) ^ 2
def operation_D (a : ℤ) : ℤ := abs a ^ 2

theorem incorrect_operation :
  operation_D a ≠ abs 4 :=
by
  sorry

end incorrect_operation_l1546_154629


namespace difference_in_pennies_l1546_154622

theorem difference_in_pennies (p : ℤ) : 
  let alice_nickels := 3 * p + 2
  let bob_nickels := 2 * p + 6
  let difference_nickels := alice_nickels - bob_nickels
  let difference_in_pennies := difference_nickels * 5
  difference_in_pennies = 5 * p - 20 :=
by
  sorry

end difference_in_pennies_l1546_154622


namespace problem1_problem2_l1546_154652

def setA : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | (x - m + 2) * (x - m - 2) ≤ 0}

-- Problem 1: prove that if A ∩ B = {x | 0 ≤ x ≤ 3}, then m = 2
theorem problem1 (m : ℝ) : (setA ∩ setB m = {x | 0 ≤ x ∧ x ≤ 3}) → m = 2 :=
by
  sorry

-- Problem 2: prove that if A ⊆ complement of B, then m ∈ (-∞, -3) ∪ (5, +∞)
theorem problem2 (m : ℝ) : (setA ⊆ (fun x => x ∉ setB m)) → (m < -3 ∨ m > 5) :=
by
  sorry

end problem1_problem2_l1546_154652


namespace quadratic_roots_range_l1546_154696

theorem quadratic_roots_range (k : ℝ) : (x^2 - 6*x + k = 0) → k < 9 := 
by
  sorry

end quadratic_roots_range_l1546_154696


namespace last_three_digits_of_2_pow_15000_l1546_154617

-- We need to define the given condition as a hypothesis and then state the goal.
theorem last_three_digits_of_2_pow_15000 :
  (2 ^ 500 ≡ 1 [MOD 1250]) → (2 ^ 15000 ≡ 1 [MOD 1000]) := by
  sorry

end last_three_digits_of_2_pow_15000_l1546_154617


namespace RebeccaHasTwentyMarbles_l1546_154656

variable (groups : ℕ) (marbles_per_group : ℕ) (total_marbles : ℕ)

def totalMarbles (g m : ℕ) : ℕ :=
  g * m

theorem RebeccaHasTwentyMarbles
  (h1 : groups = 5)
  (h2 : marbles_per_group = 4)
  (h3 : total_marbles = totalMarbles groups marbles_per_group) :
  total_marbles = 20 :=
by {
  sorry
}

end RebeccaHasTwentyMarbles_l1546_154656


namespace sum_of_number_and_square_eq_132_l1546_154679

theorem sum_of_number_and_square_eq_132 (x : ℝ) (h : x + x^2 = 132) : x = 11 ∨ x = -12 :=
by
  sorry

end sum_of_number_and_square_eq_132_l1546_154679


namespace debby_candy_problem_l1546_154607

theorem debby_candy_problem (D : ℕ) (sister_candy : ℕ) (eaten : ℕ) (remaining : ℕ) 
  (h1 : sister_candy = 42) (h2 : eaten = 35) (h3 : remaining = 39) :
  D + sister_candy - eaten = remaining ↔ D = 32 :=
by
  sorry

end debby_candy_problem_l1546_154607


namespace solve_inequalities_l1546_154648

theorem solve_inequalities :
  {x : ℝ | -3 < x ∧ x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 2} =
  { x : ℝ | (5 / (x + 3) ≥ 1) ∧ (x^2 + x - 2 ≥ 0) } :=
sorry

end solve_inequalities_l1546_154648


namespace picnic_students_count_l1546_154654

theorem picnic_students_count (x : ℕ) (h1 : (x / 2) + (x / 3) + (x / 4) = 65) : x = 60 :=
by
  -- Proof goes here
  sorry

end picnic_students_count_l1546_154654


namespace alpha_eq_one_l1546_154633

-- Definitions based on conditions from the problem statement.
variable (α : ℝ) 
variable (f : ℝ → ℝ)

-- The conditions defined as hypotheses
axiom functional_eq (x y : ℝ) : f (α * (x + y)) = f x + f y
axiom non_constant : ∃ x y : ℝ, f x ≠ 0

-- The statement to prove
theorem alpha_eq_one : (∃ f : ℝ → ℝ, (∀ x y : ℝ, f (α * (x + y)) = f x + f y) ∧ (∃ x y : ℝ, f x ≠ f y)) → α = 1 :=
by
  sorry

end alpha_eq_one_l1546_154633


namespace net_moles_nh3_after_reactions_l1546_154621

/-- Define the stoichiometry of the reactions and available amounts of reactants -/
def step1_reaction (nh4cl na2co3 : ℕ) : ℕ :=
  if nh4cl / 2 >= na2co3 then 
    2 * na2co3
  else 
    2 * (nh4cl / 2)

def step2_reaction (koh h3po4 : ℕ) : ℕ :=
  0  -- No NH3 produced in this step

theorem net_moles_nh3_after_reactions :
  let nh4cl := 3
  let na2co3 := 1
  let koh := 3
  let h3po4 := 1
  let nh3_after_step1 := step1_reaction nh4cl na2co3
  let nh3_after_step2 := step2_reaction koh h3po4
  nh3_after_step1 + nh3_after_step2 = 2 :=
by
  sorry

end net_moles_nh3_after_reactions_l1546_154621


namespace maximum_a_value_l1546_154692

theorem maximum_a_value :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a + 1)*x^2 - (a + 1)*x + 2022 ∧ (a + 1)*x^2 - (a + 1)*x + 2022 ≤ 2022) →
  a ≤ 16175 := 
by {
  sorry
}

end maximum_a_value_l1546_154692


namespace range_of_a_l1546_154671

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (e^x + 1) * (a * x + 2 * a - 2) < 2) → a < 4 / 3 :=
by
  sorry

end range_of_a_l1546_154671


namespace intersection_of_A_and_B_l1546_154614

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = expected_intersection :=
by
  sorry

end intersection_of_A_and_B_l1546_154614


namespace max_value_of_PQ_l1546_154666

noncomputable def maxDistance (P Q : ℝ × ℝ) : ℝ :=
  let dist (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)
  let O1 : ℝ × ℝ := (0, 4)
  dist P Q

theorem max_value_of_PQ:
  ∀ (P Q : ℝ × ℝ),
    (P.1 ^ 2 + (P.2 - 4) ^ 2 = 1) →
    (Q.1 ^ 2 / 9 + Q.2 ^ 2 = 1) →
    maxDistance P Q ≤ 1 + 3 * Real.sqrt 3 :=
by
  sorry

end max_value_of_PQ_l1546_154666


namespace quadratic_has_one_solution_at_zero_l1546_154651

theorem quadratic_has_one_solution_at_zero (k : ℝ) :
  ((k - 2) * (0 : ℝ)^2 + 3 * (0 : ℝ) + k^2 - 4 = 0) →
  (3^2 - 4 * (k - 2) * (k^2 - 4) = 0) → k = -2 :=
by
  intro h1 h2
  sorry

end quadratic_has_one_solution_at_zero_l1546_154651


namespace trigonometric_identity1_trigonometric_identity2_l1546_154643

theorem trigonometric_identity1 (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin (Real.pi - θ) + Real.cos (θ - Real.pi)) / (Real.sin (θ + Real.pi) + Real.cos (θ + Real.pi)) = -1/3 :=
by
  sorry

theorem trigonometric_identity2 (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ) = 4/5 :=
by
  sorry

end trigonometric_identity1_trigonometric_identity2_l1546_154643


namespace duration_of_each_class_is_3_l1546_154640

theorem duration_of_each_class_is_3
    (weeks : ℕ) 
    (x : ℝ) 
    (weekly_additional_class_hours : ℝ) 
    (homework_hours_per_week : ℝ) 
    (total_hours : ℝ) 
    (h1 : weeks = 24)
    (h2 : weekly_additional_class_hours = 4)
    (h3 : homework_hours_per_week = 4)
    (h4 : total_hours = 336) :
    (2 * x + weekly_additional_class_hours + homework_hours_per_week) * weeks = total_hours → x = 3 := 
by 
  sorry

end duration_of_each_class_is_3_l1546_154640


namespace divide_payment_correctly_l1546_154657

-- Define the number of logs contributed by each person
def logs_troikin : ℕ := 3
def logs_pyaterkin : ℕ := 5
def logs_bestoplivny : ℕ := 0

-- Define the total number of logs
def total_logs : ℕ := logs_troikin + logs_pyaterkin + logs_bestoplivny

-- Define the total number of logs used equally
def logs_per_person : ℚ := total_logs / 3

-- Define the total payment made by Bestoplivny 
def total_payment : ℕ := 80

-- Define the cost per log
def cost_per_log : ℚ := total_payment / logs_per_person

-- Define the contribution of each person to Bestoplivny
def bestoplivny_from_troikin : ℚ := logs_troikin - logs_per_person
def bestoplivny_from_pyaterkin : ℚ := logs_pyaterkin - (logs_per_person - bestoplivny_from_troikin)

-- Define the kopecks received by Troikina and Pyaterkin
def kopecks_troikin : ℚ := bestoplivny_from_troikin * cost_per_log
def kopecks_pyaterkin : ℚ := bestoplivny_from_pyaterkin * cost_per_log

-- Main theorem to prove the correct division of kopecks
theorem divide_payment_correctly : kopecks_troikin = 10 ∧ kopecks_pyaterkin = 70 :=
by
  -- ... Proof goes here
  sorry

end divide_payment_correctly_l1546_154657


namespace polygon_interior_angle_eq_l1546_154618

theorem polygon_interior_angle_eq (n : ℕ) (h : ∀ i, 1 ≤ i → i ≤ n → (interior_angle : ℝ) = 108) : n = 5 := 
sorry

end polygon_interior_angle_eq_l1546_154618


namespace desired_cost_per_pound_l1546_154661

/-- 
Let $p_1 = 8$, $w_1 = 25$, $p_2 = 5$, and $w_2 = 50$ represent the prices and weights of two types of candies.
Calculate the desired cost per pound $p_m$ of the mixture.
-/
theorem desired_cost_per_pound 
  (p1 : ℝ) (w1 : ℝ) (p2 : ℝ) (w2 : ℝ) (p_m : ℝ) 
  (h1 : p1 = 8) (h2 : w1 = 25) (h3 : p2 = 5) (h4 : w2 = 50) :
  p_m = (p1 * w1 + p2 * w2) / (w1 + w2) → p_m = 6 :=
by 
  intros
  sorry

end desired_cost_per_pound_l1546_154661


namespace blake_total_expenditure_l1546_154641

noncomputable def total_cost (rooms : ℕ) (primer_cost : ℝ) (paint_cost : ℝ) (primer_discount : ℝ) : ℝ :=
  let primer_needed := rooms
  let paint_needed := rooms
  let discounted_primer_cost := primer_cost * (1 - primer_discount)
  let total_primer_cost := primer_needed * discounted_primer_cost
  let total_paint_cost := paint_needed * paint_cost
  total_primer_cost + total_paint_cost

theorem blake_total_expenditure :
  total_cost 5 30 25 0.20 = 245 := 
by
  sorry

end blake_total_expenditure_l1546_154641


namespace hyperbola_range_m_l1546_154626

theorem hyperbola_range_m (m : ℝ) : (m - 2) * (m - 6) < 0 ↔ 2 < m ∧ m < 6 :=
by sorry

end hyperbola_range_m_l1546_154626


namespace cost_price_per_meter_of_cloth_l1546_154677

theorem cost_price_per_meter_of_cloth 
  (total_meters : ℕ)
  (selling_price : ℝ)
  (profit_per_meter : ℝ) 
  (total_profit : ℝ)
  (cp_45 : ℝ)
  (cp_per_meter: ℝ) :
  total_meters = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * total_meters →
  cp_45 = selling_price - total_profit →
  cp_per_meter = cp_45 / total_meters →
  cp_per_meter = 86 :=
by
  -- your proof here
  sorry

end cost_price_per_meter_of_cloth_l1546_154677


namespace sum_in_base5_correct_l1546_154683

-- Define numbers in base 5
def n1 : ℕ := 231
def n2 : ℕ := 414
def n3 : ℕ := 123

-- Function to convert a number from base 5 to base 10
def base5_to_base10(n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100)
  d0 * 1 + d1 * 5 + d2 * 25

-- Convert the given numbers from base 5 to base 10
def n1_base10 : ℕ := base5_to_base10 n1
def n2_base10 : ℕ := base5_to_base10 n2
def n3_base10 : ℕ := base5_to_base10 n3

-- Base 10 sum
def sum_base10 : ℕ := n1_base10 + n2_base10 + n3_base10

-- Function to convert a number from base 10 to base 5
def base10_to_base5(n : ℕ) : ℕ :=
  let d0 := n % 5
  let d1 := (n / 5) % 5
  let d2 := (n / 25) % 5
  let d3 := (n / 125)
  d0 * 1 + d1 * 10 + d2 * 100 + d3 * 1000

-- Convert the sum from base 10 to base 5
def sum_base5 : ℕ := base10_to_base5 sum_base10

-- The theorem to prove the sum in base 5 is 1323_5
theorem sum_in_base5_correct : sum_base5 = 1323 := by
  -- Proof steps would go here, but we insert sorry to skip it
  sorry

end sum_in_base5_correct_l1546_154683


namespace repeating_decimal_to_fraction_l1546_154611

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + (6 / 10) / 9) : x = 11 / 30 :=
by
  sorry

end repeating_decimal_to_fraction_l1546_154611


namespace parabola_line_intersection_l1546_154644

theorem parabola_line_intersection :
  ∀ (x y : ℝ), 
  (y = 20 * x^2 + 19 * x) ∧ (y = 20 * x + 19) →
  y = 20 * x^3 + 19 * x^2 :=
by sorry

end parabola_line_intersection_l1546_154644


namespace line_circle_intersect_l1546_154680

theorem line_circle_intersect {a : ℝ} :
  ∃ P : ℝ × ℝ, (P.1, P.2) = (-2, 0) ∧ (a * P.1 - P.2 + 2 * a = 0) ∧ (P.1^2 + P.2^2 < 9) :=
by
  use (-2, 0)
  sorry

end line_circle_intersect_l1546_154680


namespace pete_and_raymond_spent_together_l1546_154663

    def value_nickel : ℕ := 5
    def value_dime : ℕ := 10
    def value_quarter : ℕ := 25

    def pete_nickels_spent : ℕ := 4
    def pete_dimes_spent : ℕ := 3
    def pete_quarters_spent : ℕ := 2

    def raymond_initial : ℕ := 250
    def raymond_nickels_left : ℕ := 5
    def raymond_dimes_left : ℕ := 7
    def raymond_quarters_left : ℕ := 4
    
    def total_spent : ℕ := 155

    theorem pete_and_raymond_spent_together :
      (pete_nickels_spent * value_nickel + pete_dimes_spent * value_dime + pete_quarters_spent * value_quarter)
      + (raymond_initial - (raymond_nickels_left * value_nickel + raymond_dimes_left * value_dime + raymond_quarters_left * value_quarter))
      = total_spent :=
      by
        sorry
    
end pete_and_raymond_spent_together_l1546_154663


namespace angle_R_in_triangle_l1546_154684

theorem angle_R_in_triangle (P Q R : ℝ) 
  (hP : P = 90)
  (hQ : Q = 4 * R - 10)
  (angle_sum : P + Q + R = 180) 
  : R = 20 := by 
sorry

end angle_R_in_triangle_l1546_154684


namespace brittany_second_test_grade_l1546_154699

theorem brittany_second_test_grade
  (first_test_grade second_test_grade : ℕ) 
  (average_after_second : ℕ)
  (h1 : first_test_grade = 78)
  (h2 : average_after_second = 81) 
  (h3 : (first_test_grade + second_test_grade) / 2 = average_after_second) :
  second_test_grade = 84 :=
by
  sorry

end brittany_second_test_grade_l1546_154699


namespace sum_of_consecutive_even_numbers_l1546_154624

theorem sum_of_consecutive_even_numbers (n : ℤ) 
  (h : n + 4 = 14) : n + (n + 2) + (n + 4) + (n + 6) = 52 :=
by
  sorry

end sum_of_consecutive_even_numbers_l1546_154624


namespace prove_n_eq_one_l1546_154602

-- Definitions of the vectors a and b
def vector_a (n : ℝ) : ℝ × ℝ := (1, n)
def vector_b (n : ℝ) : ℝ × ℝ := (-1, n - 2)

-- Definition of collinearity between two vectors
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem to prove that if a and b are collinear, then n = 1
theorem prove_n_eq_one (n : ℝ) (h_collinear : collinear (vector_a n) (vector_b n)) : n = 1 :=
sorry

end prove_n_eq_one_l1546_154602


namespace iced_coffee_cost_is_2_l1546_154600

def weekly_latte_cost := 4 * 5
def annual_latte_cost := weekly_latte_cost * 52
def weekly_iced_coffee_cost (x : ℝ) := x * 3
def annual_iced_coffee_cost (x : ℝ) := weekly_iced_coffee_cost x * 52
def total_annual_coffee_cost (x : ℝ) := annual_latte_cost + annual_iced_coffee_cost x
def reduced_spending_goal (x : ℝ) := 0.75 * total_annual_coffee_cost x
def saved_amount := 338

theorem iced_coffee_cost_is_2 :
  ∃ x : ℝ, (total_annual_coffee_cost x - reduced_spending_goal x = saved_amount) → x = 2 :=
by
  sorry

end iced_coffee_cost_is_2_l1546_154600


namespace vasya_expected_area_greater_l1546_154601

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l1546_154601


namespace sin_law_of_sines_l1546_154615

theorem sin_law_of_sines (a b : ℝ) (sin_A sin_B : ℝ)
  (h1 : a = 3)
  (h2 : b = 4)
  (h3 : sin_A = 3 / 5) :
  sin_B = 4 / 5 := 
sorry

end sin_law_of_sines_l1546_154615


namespace collinear_points_d_value_l1546_154606

theorem collinear_points_d_value (a b c d : ℚ)
  (h1 : b = a)
  (h2 : c = -(a+1)/2)
  (collinear : (4 * d * (4 * a + 5) + a + 1 = 0)) :
  d = 9/20 :=
by {
  sorry
}

end collinear_points_d_value_l1546_154606


namespace total_weight_correct_l1546_154674

def Marco_strawberry_weight : ℕ := 15
def Dad_strawberry_weight : ℕ := 22
def total_strawberry_weight : ℕ := Marco_strawberry_weight + Dad_strawberry_weight

theorem total_weight_correct :
  total_strawberry_weight = 37 :=
by
  sorry

end total_weight_correct_l1546_154674


namespace max_volume_of_open_top_box_l1546_154685

noncomputable def box_max_volume (x : ℝ) : ℝ :=
  (10 - 2 * x) * (16 - 2 * x) * x

theorem max_volume_of_open_top_box : ∃ x : ℝ, 0 < x ∧ x < 5 ∧ box_max_volume x = 144 :=
by
  sorry

end max_volume_of_open_top_box_l1546_154685


namespace largest_x_plus_y_l1546_154690

theorem largest_x_plus_y (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) : x + y ≤ 18 / 7 :=
by
  sorry

end largest_x_plus_y_l1546_154690


namespace value_of_x_l1546_154635

theorem value_of_x (x : ℝ) (h : 0.5 * x = 0.25 * 1500 - 30) : x = 690 :=
by
  sorry

end value_of_x_l1546_154635


namespace abc_equality_l1546_154681

noncomputable def abc_value (a b c : ℝ) : ℝ := (11 + Real.sqrt 117) / 2

theorem abc_equality (a b c : ℝ) (h1 : a + 1/b = 5) (h2 : b + 1/c = 2) (h3 : (c + 1/a)^2 = 4) :
  a * b * c = abc_value a b c := 
sorry

end abc_equality_l1546_154681


namespace arthur_first_day_spending_l1546_154688

-- Define the costs of hamburgers and hot dogs.
variable (H D : ℝ)
-- Given conditions
axiom hot_dog_cost : D = 1
axiom second_day_purchase : 2 * H + 3 * D = 7

-- Goal: How much did Arthur spend on the first day?
-- We need to verify that 3H + 4D = 10
theorem arthur_first_day_spending : 3 * H + 4 * D = 10 :=
by
  -- Validating given conditions
  have h1 := hot_dog_cost
  have h2 := second_day_purchase
  -- Insert proof here
  sorry

end arthur_first_day_spending_l1546_154688


namespace reduced_price_per_kg_of_oil_l1546_154693

/-- The reduced price per kg of oil is approximately Rs. 48 -
given a 30% reduction in price and the ability to buy 5 kgs more
for Rs. 800. -/
theorem reduced_price_per_kg_of_oil
  (P R : ℝ)
  (h1 : R = 0.70 * P)
  (h2 : 800 / R = (800 / P) + 5) : 
  R = 48 :=
sorry

end reduced_price_per_kg_of_oil_l1546_154693


namespace second_bucket_capacity_l1546_154689

-- Define the initial conditions as given in the problem.
def tank_capacity : ℕ := 48
def bucket1_capacity : ℕ := 4

-- Define the number of times the 4-liter bucket is used.
def bucket1_uses : ℕ := tank_capacity / bucket1_capacity

-- Define a condition related to bucket uses.
def buckets_use_relation (x : ℕ) : Prop :=
  bucket1_uses = (tank_capacity / x) - 4

-- Formulate the theorem that states the capacity of the second bucket.
theorem second_bucket_capacity (x : ℕ) (h : buckets_use_relation x) : x = 3 :=
by {
  sorry
}

end second_bucket_capacity_l1546_154689


namespace arithmetic_sequence_sum_nine_l1546_154639

variable {α : Type*} [LinearOrderedField α]

/-- An arithmetic sequence (a_n) is defined by a starting term a_1 and a common difference d. -/
def arithmetic_seq (a d n : α) : α := a + (n - 1) * d

/-- The sum of the first n terms of an arithmetic sequence. -/
def arithmetic_sum (a d n : α) : α := n / 2 * (2 * a + (n - 1) * d)

/-- Prove that for a given arithmetic sequence where a_2 + a_4 + a_9 = 24, the sum of the first 9 terms is 72. -/
theorem arithmetic_sequence_sum_nine 
  {a d : α}
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 4 + arithmetic_seq a d 9 = 24) :
  arithmetic_sum a d 9 = 72 := 
by
  sorry

end arithmetic_sequence_sum_nine_l1546_154639


namespace comparison_among_abc_l1546_154659

noncomputable def a : ℝ := 2^(1/5)
noncomputable def b : ℝ := (1/5)^2
noncomputable def c : ℝ := Real.log (1/5) / Real.log 2

theorem comparison_among_abc : a > b ∧ b > c :=
by
  -- Assume the necessary conditions and the conclusion.
  sorry

end comparison_among_abc_l1546_154659


namespace dolls_given_to_girls_correct_l1546_154672

-- Define the total number of toys given
def total_toys_given : ℕ := 403

-- Define the number of toy cars given to boys
def toy_cars_given_to_boys : ℕ := 134

-- Define the number of dolls given to girls
def dolls_given_to_girls : ℕ := total_toys_given - toy_cars_given_to_boys

-- State the theorem to prove the number of dolls given to girls
theorem dolls_given_to_girls_correct : dolls_given_to_girls = 269 := by
  sorry

end dolls_given_to_girls_correct_l1546_154672


namespace five_a_squared_plus_one_divisible_by_three_l1546_154646

theorem five_a_squared_plus_one_divisible_by_three (a : ℤ) (h : a % 3 ≠ 0) : (5 * a^2 + 1) % 3 = 0 :=
sorry

end five_a_squared_plus_one_divisible_by_three_l1546_154646


namespace exactly_one_wins_at_most_two_win_l1546_154678

def prob_A : ℚ := 4 / 5 
def prob_B : ℚ := 3 / 5 
def prob_C : ℚ := 7 / 10

theorem exactly_one_wins :
  (prob_A * (1 - prob_B) * (1 - prob_C) + 
   (1 - prob_A) * prob_B * (1 - prob_C) + 
   (1 - prob_A) * (1 - prob_B) * prob_C) = 47 / 250 := 
by sorry

theorem at_most_two_win :
  (1 - (prob_A * prob_B * prob_C)) = 83 / 125 :=
by sorry

end exactly_one_wins_at_most_two_win_l1546_154678


namespace no_solutions_l1546_154612

theorem no_solutions (x y : ℤ) (h : 8 * x + 3 * y^2 = 5) : False :=
by
  sorry

end no_solutions_l1546_154612


namespace johnson_class_more_students_l1546_154627

theorem johnson_class_more_students
  (finley_class_students : ℕ)
  (johnson_class_students : ℕ)
  (h_finley : finley_class_students = 24)
  (h_johnson : johnson_class_students = 22) :
  johnson_class_students - finley_class_students / 2 = 10 :=
  sorry

end johnson_class_more_students_l1546_154627


namespace investment_time_l1546_154603

theorem investment_time (P R diff : ℝ) (T : ℕ) 
  (hP : P = 1500)
  (hR : R = 0.10)
  (hdiff : diff = 15)
  (h1 : P * ((1 + R) ^ T - 1) - (P * R * T) = diff) 
  : T = 2 := 
by
  -- proof steps here
  sorry

end investment_time_l1546_154603


namespace transformed_center_is_correct_l1546_154660

-- Definition for transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (dx : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2)

def translate_up (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Given conditions
def initial_center : ℝ × ℝ := (4, -3)
def reflection_center := reflect_x initial_center
def translated_right_center := translate_right reflection_center 5
def final_center := translate_up translated_right_center 3

-- The statement to be proved
theorem transformed_center_is_correct : final_center = (9, 6) :=
by
  sorry

end transformed_center_is_correct_l1546_154660


namespace min_value_x2_plus_y2_l1546_154616

theorem min_value_x2_plus_y2 :
  ∀ x y : ℝ, (x + 5)^2 + (y - 12)^2 = 196 → x^2 + y^2 ≥ 1 :=
by
  intros x y h
  sorry

end min_value_x2_plus_y2_l1546_154616


namespace sum_of_sequences_l1546_154694

-- Definition of the problem conditions
def seq1 := [2, 12, 22, 32, 42]
def seq2 := [10, 20, 30, 40, 50]
def sum_seq1 := 2 + 12 + 22 + 32 + 42
def sum_seq2 := 10 + 20 + 30 + 40 + 50

-- Lean statement of the problem
theorem sum_of_sequences :
  sum_seq1 + sum_seq2 = 260 := by
  sorry

end sum_of_sequences_l1546_154694


namespace problem_solution_l1546_154698

def expr := 1 + 1 / (1 + 1 / (1 + 1))
def answer : ℚ := 5 / 3

theorem problem_solution : expr = answer :=
by
  sorry

end problem_solution_l1546_154698


namespace sum_series_eq_two_l1546_154686

theorem sum_series_eq_two : ∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1)) = 2 :=
sorry

end sum_series_eq_two_l1546_154686


namespace find_m_value_l1546_154687

theorem find_m_value (m : ℝ) 
  (first_term : ℝ := 18) (second_term : ℝ := 6)
  (second_term_2 : ℝ := 6 + m) 
  (S1 : ℝ := first_term / (1 - second_term / first_term))
  (S2 : ℝ := first_term / (1 - second_term_2 / first_term))
  (eq_sum : S2 = 3 * S1) :
  m = 8 := by
  sorry

end find_m_value_l1546_154687


namespace percent_calculation_l1546_154669

theorem percent_calculation (y : ℝ) : (0.3 * 0.7 * y - 0.1 * y) = 0.11 * y ∧ (0.11 * y / y * 100 = 11) := by
  sorry

end percent_calculation_l1546_154669


namespace y_run_time_l1546_154667

theorem y_run_time (t : ℕ) (h_avg : (t + 26) / 2 = 42) : t = 58 :=
by
  sorry

end y_run_time_l1546_154667


namespace work_completion_time_l1546_154697

-- Define the constants for work rates and times
def W : ℚ := 1
def P_rate : ℚ := W / 20
def Q_rate : ℚ := W / 12
def initial_days : ℚ := 4

-- Define the amount of work done by P in the initial 4 days
def work_done_initial : ℚ := initial_days * P_rate

-- Define the remaining work after initial 4 days
def remaining_work : ℚ := W - work_done_initial

-- Define the combined work rate of P and Q
def combined_rate : ℚ := P_rate + Q_rate

-- Define the time taken to complete the remaining work
def remaining_days : ℚ := remaining_work / combined_rate

-- Define the total time taken to complete the work
def total_days : ℚ := initial_days + remaining_days

-- The theorem to prove
theorem work_completion_time :
  total_days = 10 := 
by
  -- these term can be the calculation steps
  sorry

end work_completion_time_l1546_154697


namespace smallest_next_divisor_l1546_154613

theorem smallest_next_divisor (m : ℕ) (h_digit : 10000 ≤ m ∧ m < 100000) (h_odd : m % 2 = 1) (h_div : 437 ∣ m) :
  ∃ d : ℕ, 437 < d ∧ d ∣ m ∧ (∀ e : ℕ, 437 < e ∧ e < d → ¬ e ∣ m) ∧ d = 475 := 
sorry

end smallest_next_divisor_l1546_154613


namespace wine_remaining_percentage_l1546_154608

theorem wine_remaining_percentage :
  let initial_wine := 250.0 -- initial wine in liters
  let daily_fraction := (249.0 / 250.0)
  let days := 50
  let remaining_wine := (daily_fraction ^ days) * initial_wine
  let percentage_remaining := (remaining_wine / initial_wine) * 100
  percentage_remaining = 81.846 :=
by
  sorry

end wine_remaining_percentage_l1546_154608


namespace child_haircut_cost_l1546_154638

/-
Problem Statement:
- Women's haircuts cost $48.
- Tayzia and her two daughters get haircuts.
- Tayzia wants to give a 20% tip to the hair stylist, which amounts to $24.
Question: How much does a child's haircut cost?
-/

noncomputable def cost_of_child_haircut (C : ℝ) : Prop :=
  let women's_haircut := 48
  let tip := 24
  let total_cost_before_tip := women's_haircut + 2 * C
  total_cost_before_tip * 0.20 = tip ∧ total_cost_before_tip = 120 ∧ C = 36

theorem child_haircut_cost (C : ℝ) (h1 : cost_of_child_haircut C) : C = 36 :=
  by sorry

end child_haircut_cost_l1546_154638


namespace andy_more_candies_than_caleb_l1546_154620

theorem andy_more_candies_than_caleb :
  let billy_initial := 6
  let caleb_initial := 11
  let andy_initial := 9
  let father_packet := 36
  let billy_additional := 8
  let caleb_additional := 11
  let billy_total := billy_initial + billy_additional
  let caleb_total := caleb_initial + caleb_additional
  let total_given := billy_additional + caleb_additional
  let andy_additional := father_packet - total_given
  let andy_total := andy_initial + andy_additional
  andy_total - caleb_total = 4 :=
by {
  sorry
}

end andy_more_candies_than_caleb_l1546_154620


namespace price_after_discount_eq_cost_price_l1546_154676

theorem price_after_discount_eq_cost_price (m : Real) :
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  price_after_discount = m :=
by
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  sorry

end price_after_discount_eq_cost_price_l1546_154676


namespace length_OP_l1546_154625

noncomputable def right_triangle_length_OP (XY XZ YZ : ℝ) (rO rP : ℝ) : ℝ :=
  let O := rO
  let P := rP
  -- Coordinates of point Y and Z can be O = (0, r), P = (OP, r)
  25 -- directly from the given correct answer

theorem length_OP (XY XZ YZ : ℝ) (rO rP : ℝ) (hXY : XY = 7) (hXZ : XZ = 24) (hYZ : YZ = 25) 
  (hO : rO = YZ - rO) (hP : rP = YZ - rP) : 
  right_triangle_length_OP XY XZ YZ rO rP = 25 :=
sorry

end length_OP_l1546_154625


namespace eq_of_divisibility_l1546_154636

theorem eq_of_divisibility (a b : ℕ) (h : (a^2 + b^2) ∣ (a * b)) : a = b :=
  sorry

end eq_of_divisibility_l1546_154636


namespace minimum_possible_value_of_Box_l1546_154658

theorem minimum_possible_value_of_Box :
  ∃ a b : ℤ, a ≠ b ∧ a * b = 45 ∧ 
    (∀ c d : ℤ, c * d = 45 → c^2 + d^2 ≥ 106) ∧ a^2 + b^2 = 106 :=
by
  sorry

end minimum_possible_value_of_Box_l1546_154658


namespace alternating_sign_max_pos_l1546_154675

theorem alternating_sign_max_pos (x : ℕ → ℝ) 
  (h_nonzero : ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n ≠ 0)
  (h_condition : ∀ k, 1 ≤ k ∧ k ≤ 2022 → x k + (1 / x (k + 1)) < 0)
  (h_periodic : x 2023 = x 1) :
  ∃ m, m = 1011 ∧ ( ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n > 0 → n ≤ m ∧ m ≤ 2022 ) := 
sorry

end alternating_sign_max_pos_l1546_154675


namespace shopkeeper_sold_articles_l1546_154662

theorem shopkeeper_sold_articles (C : ℝ) (N : ℕ) 
  (h1 : (35 * C = N * C + (1/6) * (N * C))) : 
  N = 30 :=
by
  sorry

end shopkeeper_sold_articles_l1546_154662


namespace josh_total_payment_with_tax_and_discount_l1546_154642

-- Definitions
def total_string_cheeses (pack1 : ℕ) (pack2 : ℕ) (pack3 : ℕ) : ℕ :=
  pack1 + pack2 + pack3

def total_cost_before_tax_and_discount (n : ℕ) (cost_per_cheese : ℚ) : ℚ :=
  n * cost_per_cheese

def discount_amount (cost : ℚ) (discount_rate : ℚ) : ℚ :=
  cost * discount_rate

def discounted_cost (cost : ℚ) (discount : ℚ) : ℚ :=
  cost - discount

def sales_tax_amount (cost : ℚ) (tax_rate : ℚ) : ℚ :=
  cost * tax_rate

def total_cost (cost : ℚ) (tax : ℚ) : ℚ :=
  cost + tax

-- The statement
theorem josh_total_payment_with_tax_and_discount :
  let cost_per_cheese := 0.10
  let discount_rate := 0.05
  let tax_rate := 0.12
  total_cost (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                              (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate))
             (sales_tax_amount (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                                               (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate)) tax_rate) = 6.81 := 
  sorry

end josh_total_payment_with_tax_and_discount_l1546_154642


namespace coeff_of_x_pow_4_in_expansion_l1546_154665

theorem coeff_of_x_pow_4_in_expansion : 
  (∃ c : ℤ, c = (-1)^3 * Nat.choose 8 3 ∧ c = -56) :=
by
  sorry

end coeff_of_x_pow_4_in_expansion_l1546_154665


namespace min_inv_sum_l1546_154655

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ 1 = 2*a + b

theorem min_inv_sum (a b : ℝ) (h : minimum_value_condition a b) : 
  ∃ a b : ℝ, (1 / a + 1 / b = 3 + 2 * Real.sqrt 2) := 
by 
  have h1 : a > 0 := h.1;
  have h2 : b > 0 := h.2.1;
  have h3 : 1 = 2 * a + b := h.2.2;
  sorry

end min_inv_sum_l1546_154655


namespace range_of_a_function_greater_than_exp_neg_x_l1546_154632

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a / x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ f x a = 0) → (0 < a ∧ a ≤ 1 / Real.exp 1) :=
sorry

theorem function_greater_than_exp_neg_x (a : ℝ) (h : a ≥ 2 / Real.exp 1) (x : ℝ) (hx : 0 < x) : f x a > Real.exp (-x) :=
sorry

end range_of_a_function_greater_than_exp_neg_x_l1546_154632


namespace option_D_is_divisible_by_9_l1546_154637

theorem option_D_is_divisible_by_9 (k : ℕ) (hk : k > 0) : 9 ∣ 3 * (2 + 7^k) := 
sorry

end option_D_is_divisible_by_9_l1546_154637


namespace SavingsInequality_l1546_154668

theorem SavingsInequality (n : ℕ) : 52 + 15 * n > 70 + 12 * n := 
by sorry

end SavingsInequality_l1546_154668


namespace imaginary_unit_cubic_l1546_154653

def imaginary_unit_property (i : ℂ) : Prop :=
  i^2 = -1

theorem imaginary_unit_cubic (i : ℂ) (h : imaginary_unit_property i) : 1 + i^3 = 1 - i :=
  sorry

end imaginary_unit_cubic_l1546_154653


namespace martin_discounted_tickets_l1546_154647

-- Definitions of the problem conditions
def total_tickets (F D : ℕ) := F + D = 10
def total_cost (F D : ℕ) := 2 * F + (16/10) * D = 184/10

-- Statement of the proof
theorem martin_discounted_tickets (F D : ℕ) (h1 : total_tickets F D) (h2 : total_cost F D) :
  D = 4 :=
sorry

end martin_discounted_tickets_l1546_154647


namespace frog_reaches_top_l1546_154695

theorem frog_reaches_top (x : ℕ) (h1 : ∀ d ≤ x - 1, 3 * d + 5 ≥ 50) : x = 16 := by
  sorry

end frog_reaches_top_l1546_154695


namespace arithmetic_sequence_a5_l1546_154631

theorem arithmetic_sequence_a5 {a : ℕ → ℝ} (h₁ : a 2 + a 8 = 16) : a 5 = 8 :=
sorry

end arithmetic_sequence_a5_l1546_154631


namespace sequence_n_5_l1546_154670

theorem sequence_n_5 (a : ℤ) (n : ℕ → ℤ) 
  (h1 : ∀ i > 1, n i = 2 * n (i - 1) + a)
  (h2 : n 2 = 5)
  (h3 : n 8 = 257) : n 5 = 33 :=
by
  sorry

end sequence_n_5_l1546_154670


namespace box_surface_area_l1546_154623

variables (a b c : ℝ)

noncomputable def sum_edges : ℝ := 4 * (a + b + c)
noncomputable def diagonal_length : ℝ := Real.sqrt (a^2 + b^2 + c^2)
noncomputable def surface_area : ℝ := 2 * (a * b + b * c + c * a)

/- The problem states that the sum of the lengths of the edges and the diagonal length gives us these values. -/
theorem box_surface_area (h1 : sum_edges a b c = 168) (h2 : diagonal_length a b c = 25) : surface_area a b c = 1139 :=
sorry

end box_surface_area_l1546_154623


namespace min_value_of_a_plus_b_l1546_154664

theorem min_value_of_a_plus_b 
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_eq : 1 / a + 2 / b = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_of_a_plus_b_l1546_154664


namespace sin_double_angle_solution_l1546_154604

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l1546_154604


namespace totalProblemsSolved_l1546_154645

-- Given conditions
def initialProblemsSolved : Nat := 45
def additionalProblemsSolved : Nat := 18

-- Statement to prove the total problems solved equals 63
theorem totalProblemsSolved : initialProblemsSolved + additionalProblemsSolved = 63 := 
by
  sorry

end totalProblemsSolved_l1546_154645


namespace passengers_on_plane_l1546_154610

variables (P : ℕ) (fuel_per_mile : ℕ := 20) (fuel_per_person : ℕ := 3) (fuel_per_bag : ℕ := 2)
variables (num_crew : ℕ := 5) (bags_per_person : ℕ := 2) (trip_distance : ℕ := 400)
variables (total_fuel : ℕ := 106000)

def total_people := P + num_crew
def total_bags := bags_per_person * total_people
def total_fuel_per_mile := fuel_per_mile + fuel_per_person * P + fuel_per_bag * total_bags
def total_trip_fuel := trip_distance * total_fuel_per_mile

theorem passengers_on_plane : total_trip_fuel = total_fuel → P = 33 := 
by
  sorry

end passengers_on_plane_l1546_154610
