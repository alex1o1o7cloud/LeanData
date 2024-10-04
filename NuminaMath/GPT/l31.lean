import Mathlib

namespace arithmetic_seq_8th_term_l31_31148

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l31_31148


namespace tip_percentage_is_20_l31_31895

theorem tip_percentage_is_20 (total_spent price_before_tax_and_tip : ℝ) (sales_tax_rate : ℝ) (h1 : total_spent = 158.40) (h2 : price_before_tax_and_tip = 120) (h3 : sales_tax_rate = 0.10) :
  ((total_spent - (price_before_tax_and_tip * (1 + sales_tax_rate))) / (price_before_tax_and_tip * (1 + sales_tax_rate))) * 100 = 20 :=
by
  sorry

end tip_percentage_is_20_l31_31895


namespace relay_race_total_time_is_correct_l31_31832

-- Define the time taken by each runner
def time_Ainslee : ℕ := 72
def time_Bridget : ℕ := (10 * time_Ainslee) / 9
def time_Cecilia : ℕ := (3 * time_Bridget) / 4
def time_Dana : ℕ := (5 * time_Cecilia) / 6

-- Define the total time and convert to minutes and seconds
def total_time_seconds : ℕ := time_Ainslee + time_Bridget + time_Cecilia + time_Dana
def total_time_minutes := total_time_seconds / 60
def total_time_remainder := total_time_seconds % 60

theorem relay_race_total_time_is_correct :
  total_time_minutes = 4 ∧ total_time_remainder = 22 :=
by
  -- All intermediate values can be calculated using the definitions
  -- provided above correctly.
  sorry

end relay_race_total_time_is_correct_l31_31832


namespace arithmetic_sequence_8th_term_is_71_l31_31180

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l31_31180


namespace age_of_older_teenager_l31_31401

theorem age_of_older_teenager
  (a b : ℕ) 
  (h1 : a^2 - b^2 = 4 * (a + b)) 
  (h2 : a + b = 8 * (a - b)) 
  (h3 : a > b) : 
  a = 18 :=
sorry

end age_of_older_teenager_l31_31401


namespace total_population_l31_31834

variable (b g t s : ℕ)

theorem total_population (hb : b = 4 * g) (hg : g = 8 * t) (ht : t = 2 * s) :
  b + g + t + s = (83 * g) / 16 :=
by sorry

end total_population_l31_31834


namespace total_surface_area_excluding_bases_l31_31267

def lower_base_radius : ℝ := 8
def upper_base_radius : ℝ := 5
def frustum_height : ℝ := 6
def cylinder_section_height : ℝ := 2
def cylinder_section_radius : ℝ := 5

theorem total_surface_area_excluding_bases :
  let l := Real.sqrt (frustum_height ^ 2 + (lower_base_radius - upper_base_radius) ^ 2)
  let lateral_surface_area_frustum := π * (lower_base_radius + upper_base_radius) * l
  let lateral_surface_area_cylinder := 2 * π * cylinder_section_radius * cylinder_section_height
  lateral_surface_area_frustum + lateral_surface_area_cylinder = 39 * π * Real.sqrt 5 + 20 * π :=
by
  sorry

end total_surface_area_excluding_bases_l31_31267


namespace arithmetic_sequence_8th_term_l31_31199

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l31_31199


namespace compute_series_l31_31846

noncomputable def sum_series (c d : ℝ) : ℝ :=
  ∑' n, 1 / ((n-1) * d - (n-2) * c) / (n * d - (n-1) * c)

theorem compute_series (c d : ℝ) (hc_pos : 0 < c) (hd_pos : 0 < d) (hcd : d < c) : 
  sum_series c d = 1 / ((d - c) * c) :=
sorry

end compute_series_l31_31846


namespace jessica_balloon_count_l31_31711

theorem jessica_balloon_count :
  (∀ (joan_initial_balloon_count sally_popped_balloon_count total_balloon_count: ℕ),
  joan_initial_balloon_count = 9 →
  sally_popped_balloon_count = 5 →
  total_balloon_count = 6 →
  ∃ (jessica_balloon_count: ℕ),
    jessica_balloon_count = total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count) →
    jessica_balloon_count = 2) :=
by
  intros joan_initial_balloon_count sally_popped_balloon_count total_balloon_count j1 j2 t1
  use total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count)
  sorry

end jessica_balloon_count_l31_31711


namespace negation_example_l31_31384

theorem negation_example :
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
sorry

end negation_example_l31_31384


namespace evaluate_expression_l31_31669

theorem evaluate_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end evaluate_expression_l31_31669


namespace complex_division_identity_l31_31624

noncomputable def left_hand_side : ℂ := (-2 : ℂ) + (5 : ℂ) * Complex.I / (6 : ℂ) - (3 : ℂ) * Complex.I
noncomputable def right_hand_side : ℂ := - (9 : ℂ) / 15 + (8 : ℂ) / 15 * Complex.I

theorem complex_division_identity : left_hand_side = right_hand_side := 
by
  sorry

end complex_division_identity_l31_31624


namespace problem_l31_31396

def a : ℝ := (-2)^2002
def b : ℝ := (-2)^2003

theorem problem : a + b = -2^2002 := by
  sorry

end problem_l31_31396


namespace CarlyWorkedOnElevenDogs_l31_31528

-- Given conditions
def CarlyTrimmedNails : ℕ := 164
def DogsWithThreeLegs : ℕ := 3
def NailsPerPaw : ℕ := 4
def PawsPerThreeLeggedDog : ℕ := 3
def PawsPerFourLeggedDog : ℕ := 4

-- Deduction steps
def TotalPawsWorkedOn := CarlyTrimmedNails / NailsPerPaw
def PawsOnThreeLeggedDogs := DogsWithThreeLegs * PawsPerThreeLeggedDog
def PawsOnFourLeggedDogs := TotalPawsWorkedOn - PawsOnThreeLeggedDogs
def CountFourLeggedDogs := PawsOnFourLeggedDogs / PawsPerFourLeggedDog

-- Total dogs Carly worked on
def TotalDogsCarlyWorkedOn := CountFourLeggedDogs + DogsWithThreeLegs

-- The statement we need to prove
theorem CarlyWorkedOnElevenDogs : TotalDogsCarlyWorkedOn = 11 := by
  sorry

end CarlyWorkedOnElevenDogs_l31_31528


namespace problem_remainder_P2017_mod_1000_l31_31981

def P (x : ℤ) : ℤ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem problem_remainder_P2017_mod_1000 :
  (P 2017) % 1000 = 167 :=
by
  -- this proof examines \( P(2017) \) modulo 1000
  sorry

end problem_remainder_P2017_mod_1000_l31_31981


namespace arithmetic_seq_8th_term_l31_31147

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l31_31147


namespace polynomial_remainder_l31_31544

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ), (3 * X^5 - 2 * X^3 + 5 * X - 9) = (X - 1) * (X - 2) * q + (92 * X - 95) :=
by
  intro q
  sorry

end polynomial_remainder_l31_31544


namespace total_wings_of_birds_l31_31579

def total_money_from_grandparents (gift : ℕ) (grandparents : ℕ) : ℕ := gift * grandparents

def number_of_birds (total_money : ℕ) (bird_cost : ℕ) : ℕ := total_money / bird_cost

def total_wings (birds : ℕ) (wings_per_bird : ℕ) : ℕ := birds * wings_per_bird

theorem total_wings_of_birds : 
  ∀ (gift amount : ℕ) (grandparents bird_cost wings_per_bird : ℕ),
  gift = 50 → 
  amount = 200 →
  grandparents = 4 → 
  bird_cost = 20 → 
  wings_per_bird = 2 → 
  total_wings (number_of_birds amount bird_cost) wings_per_bird = 20 :=
by {
  intros gift amount grandparents bird_cost wings_per_bird gift_eq amount_eq grandparents_eq bird_cost_eq wings_per_bird_eq,
  rw [gift_eq, amount_eq, grandparents_eq, bird_cost_eq, wings_per_bird_eq],
  simp [total_wings, total_money_from_grandparents, number_of_birds],
  sorry
}

end total_wings_of_birds_l31_31579


namespace arithmetic_sequence_8th_term_is_71_l31_31178

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l31_31178


namespace find_number_l31_31806

theorem find_number (x : ℕ) (h : x / 4 + 15 = 27) : x = 48 :=
sorry

end find_number_l31_31806


namespace Fr_zero_for_all_r_l31_31845

noncomputable def F (r : ℕ) (x y z A B C : ℝ) : ℝ :=
  x^r * Real.sin (r * A) + y^r * Real.sin (r * B) + z^r * Real.sin (r * C)

theorem Fr_zero_for_all_r
  (x y z A B C : ℝ)
  (h_sum : ∃ k : ℤ, A + B + C = k * Real.pi)
  (hF1 : F 1 x y z A B C = 0)
  (hF2 : F 2 x y z A B C = 0)
  : ∀ r : ℕ, F r x y z A B C = 0 :=
sorry

end Fr_zero_for_all_r_l31_31845


namespace CarlyWorkedOnElevenDogs_l31_31529

-- Given conditions
def CarlyTrimmedNails : ℕ := 164
def DogsWithThreeLegs : ℕ := 3
def NailsPerPaw : ℕ := 4
def PawsPerThreeLeggedDog : ℕ := 3
def PawsPerFourLeggedDog : ℕ := 4

-- Deduction steps
def TotalPawsWorkedOn := CarlyTrimmedNails / NailsPerPaw
def PawsOnThreeLeggedDogs := DogsWithThreeLegs * PawsPerThreeLeggedDog
def PawsOnFourLeggedDogs := TotalPawsWorkedOn - PawsOnThreeLeggedDogs
def CountFourLeggedDogs := PawsOnFourLeggedDogs / PawsPerFourLeggedDog

-- Total dogs Carly worked on
def TotalDogsCarlyWorkedOn := CountFourLeggedDogs + DogsWithThreeLegs

-- The statement we need to prove
theorem CarlyWorkedOnElevenDogs : TotalDogsCarlyWorkedOn = 11 := by
  sorry

end CarlyWorkedOnElevenDogs_l31_31529


namespace C_neither_necessary_nor_sufficient_for_A_l31_31788

theorem C_neither_necessary_nor_sufficient_for_A 
  (A B C : Prop) 
  (h1 : B → C)
  (h2 : B → A) : 
  ¬(A → C) ∧ ¬(C → A) :=
by
  sorry

end C_neither_necessary_nor_sufficient_for_A_l31_31788


namespace composite_sum_l31_31138

theorem composite_sum (x y n : ℕ) (hx : x > 1) (hy : y > 1) (h : x^2 + x * y - y = n^2) :
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = x + y + 1 :=
sorry

end composite_sum_l31_31138


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31459

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l31_31459


namespace sum_of_three_integers_l31_31004

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 125) : a + b + c = 31 :=
sorry

end sum_of_three_integers_l31_31004


namespace largest_divisor_of_5_consecutive_integers_l31_31477

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31477


namespace min_y_value_l31_31305

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 7*x + 10) / (x + 1)

theorem min_y_value : ∀ x > -1, f x ≥ 9 :=
by sorry

end min_y_value_l31_31305


namespace sector_to_cone_height_l31_31493

-- Definitions based on the conditions
def circle_radius : ℝ := 8
def num_sectors : ℝ := 4
def sector_angle : ℝ := 2 * Real.pi / num_sectors
def circumference_of_sector : ℝ := 2 * Real.pi * circle_radius / num_sectors
def radius_of_base : ℝ := circumference_of_sector / (2 * Real.pi)
def slant_height : ℝ := circle_radius

-- Assertion to prove
theorem sector_to_cone_height : 
  let h := Real.sqrt (slant_height^2 - radius_of_base^2) 
  in h = 2 * Real.sqrt 15 :=
by {
  sorry
}

end sector_to_cone_height_l31_31493


namespace number_of_pairs_exterior_angles_l31_31096

theorem number_of_pairs_exterior_angles (m n : ℕ) :
  (3 ≤ m ∧ 3 ≤ n ∧ 360 = m * n) ↔ 20 = 20 := 
by sorry

end number_of_pairs_exterior_angles_l31_31096


namespace min_value_of_quadratic_l31_31559

def quadratic_function (x : ℝ) : ℝ := x^2 + 6 * x + 13

theorem min_value_of_quadratic :
  (∃ x : ℝ, quadratic_function x = 4) ∧ (∀ y : ℝ, quadratic_function y ≥ 4) :=
sorry

end min_value_of_quadratic_l31_31559


namespace arithmetic_sequence_eighth_term_l31_31171

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l31_31171


namespace arrange_magnitudes_l31_31685

theorem arrange_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y := by
  sorry

end arrange_magnitudes_l31_31685


namespace pure_alcohol_addition_l31_31890

theorem pure_alcohol_addition (x : ℝ) (h1 : 3 / 10 * 10 = 3)
    (h2 : 60 / 100 * (10 + x) = (3 + x) ) : x = 7.5 :=
sorry

end pure_alcohol_addition_l31_31890


namespace students_taking_music_l31_31772

theorem students_taking_music
  (total_students : Nat)
  (students_taking_art : Nat)
  (students_taking_both : Nat)
  (students_taking_neither : Nat)
  (total_eq : total_students = 500)
  (art_eq : students_taking_art = 20)
  (both_eq : students_taking_both = 10)
  (neither_eq : students_taking_neither = 440) :
  ∃ M : Nat, M = 50 := by
  sorry

end students_taking_music_l31_31772


namespace distance_after_second_sign_l31_31717

-- Define the known conditions
def total_distance_ridden : ℕ := 1000
def distance_to_first_sign : ℕ := 350
def distance_between_signs : ℕ := 375

-- The distance Matt rode after passing the second sign
theorem distance_after_second_sign :
  total_distance_ridden - (distance_to_first_sign + distance_between_signs) = 275 := by
  sorry

end distance_after_second_sign_l31_31717


namespace fraction_of_visitors_l31_31570

theorem fraction_of_visitors
  (total_visitors did_not_enjoy_and_understand enjoyed_and_understood : ℕ)
  [h1 : total_visitors = 400]
  [h2 : did_not_enjoy_and_understand = 100]
  [h3 : enjoyed_and_understood = (total_visitors - did_not_enjoy_and_understand) / 2] :
  (enjoyed_and_understood + did_not_enjoy_and_understand = total_visitors) ∧
  (enjoyed_and_understood / total_visitors = (3 : ℤ) / 8) := by
  sorry

end fraction_of_visitors_l31_31570


namespace mirasol_balance_l31_31990

/-- Given Mirasol initially had $50, spends $10 on coffee beans, and $30 on a tumbler,
    prove that the remaining balance in her account is $10. -/
theorem mirasol_balance (initial_balance spent_coffee spent_tumbler remaining_balance : ℕ)
  (h1 : initial_balance = 50)
  (h2 : spent_coffee = 10)
  (h3 : spent_tumbler = 30)
  (h4 : remaining_balance = initial_balance - (spent_coffee + spent_tumbler)) :
  remaining_balance = 10 :=
sorry

end mirasol_balance_l31_31990


namespace betty_berries_july_five_l31_31050
open Nat

def betty_bear_berries : Prop :=
  ∃ (b : ℕ), (5 * b + 100 = 150) ∧ (b + 40 = 50)

theorem betty_berries_july_five : betty_bear_berries :=
  sorry

end betty_berries_july_five_l31_31050


namespace geometric_sequence_a5_l31_31971

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6)
  (h2 : a 3 + a 5 + a 7 = 78)
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  a 5 = 18 :=
by sorry

end geometric_sequence_a5_l31_31971


namespace x_squared_plus_y_squared_l31_31692

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 :=
by
  sorry

end x_squared_plus_y_squared_l31_31692


namespace outstanding_consumer_installment_credit_l31_31887

-- Given conditions
def total_consumer_installment_credit (C : ℝ) : Prop :=
  let automobile_installment_credit := 0.36 * C
  let automobile_finance_credit := 75
  let total_automobile_credit := 2 * automobile_finance_credit
  automobile_installment_credit = total_automobile_credit

-- Theorem to prove
theorem outstanding_consumer_installment_credit : ∃ (C : ℝ), total_consumer_installment_credit C ∧ C = 416.67 := 
by
  sorry

end outstanding_consumer_installment_credit_l31_31887


namespace arithmetic_sequence_8th_term_l31_31202

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l31_31202


namespace product_of_solutions_l31_31538

theorem product_of_solutions (x : ℝ) :
  ∃ (α β : ℝ), (x^2 - 4*x - 21 = 0) ∧ α * β = -21 := sorry

end product_of_solutions_l31_31538


namespace arithmetic_seq_sum_l31_31330

variable {a_n : ℕ → ℕ}
variable (S_n : ℕ → ℕ)
variable (q : ℕ)
variable (a_1 : ℕ)

axiom h1 : a_n 2 = 2
axiom h2 : a_n 6 = 32
axiom h3 : ∀ n, S_n n = a_1 * (1 - q ^ n) / (1 - q)

theorem arithmetic_seq_sum : S_n 100 = 2^100 - 1 :=
by
  sorry

end arithmetic_seq_sum_l31_31330


namespace circles_intersect_l31_31231

def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 3 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem circles_intersect :
  (∃ (x y : ℝ), circle_eq1 x y ∧ circle_eq2 x y) :=
sorry

end circles_intersect_l31_31231


namespace polynomial_factor_l31_31750

theorem polynomial_factor (a b : ℝ) : 
  (∃ c d : ℝ, (5 * c = a) ∧ (5 * d - 3 * c = b) ∧ (2 * c - 3 * d + 25 = 45) ∧ (2 * d - 15 = -18)) 
  → (a = 151.25 ∧ b = -98.25) :=
by
  sorry

end polynomial_factor_l31_31750


namespace relationship_among_p_q_a_b_l31_31124

open Int

variables (a b p q : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = p) (h3 : Nat.lcm a b = q)

theorem relationship_among_p_q_a_b : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end relationship_among_p_q_a_b_l31_31124


namespace smallest_n_l31_31995

def in_interval (x y z : ℝ) (n : ℕ) : Prop :=
  2 ≤ x ∧ x ≤ n ∧ 2 ≤ y ∧ y ≤ n ∧ 2 ≤ z ∧ z ≤ n

def no_two_within_one_unit (x y z : ℝ) : Prop :=
  abs (x - y) ≥ 1 ∧ abs (y - z) ≥ 1 ∧ abs (z - x) ≥ 1

def more_than_two_units_apart (x y z : ℝ) (n : ℕ) : Prop :=
  x > 2 ∧ x < n - 2 ∧ y > 2 ∧ y < n - 2 ∧ z > 2 ∧ z < n - 2

def probability_condition (n : ℕ) : Prop :=
  (n-4)^3 / (n-2)^3 > 1/3

theorem smallest_n (n : ℕ) : 11 = n → (∃ x y z : ℝ, in_interval x y z n ∧ no_two_within_one_unit x y z ∧ more_than_two_units_apart x y z n ∧ probability_condition n) :=
by
  sorry

end smallest_n_l31_31995


namespace benny_added_march_l31_31048

theorem benny_added_march :
  let january := 19 
  let february := 19
  let march_total := 46
  (march_total - (january + february) = 8) :=
by
  let january := 19
  let february := 19
  let march_total := 46
  sorry

end benny_added_march_l31_31048


namespace cone_height_is_correct_l31_31491

noncomputable def cone_height (r_circle: ℝ) (num_sectors: ℝ) : ℝ :=
  let C := 2 * real.pi * r_circle in
  let sector_circumference := C / num_sectors in
  let base_radius := sector_circumference / (2 * real.pi) in
  let slant_height := r_circle in
  real.sqrt (slant_height^2 - base_radius^2)

theorem cone_height_is_correct :
  cone_height 8 4 = 2 * real.sqrt 15 :=
by
  rw cone_height
  norm_num
  sorry

end cone_height_is_correct_l31_31491


namespace xiao_wang_ways_to_make_8_cents_l31_31485

theorem xiao_wang_ways_to_make_8_cents :
  let one_cent_coins := 8
  let two_cent_coins := 4
  let five_cent_coin := 1
  ∃ ways, ways = 7 ∧ (
       (ways = 8 ∧ one_cent_coins >= 8) ∨
       (ways = 4 ∧ two_cent_coins >= 4) ∨
       (ways = 2 ∧ one_cent_coins >= 2 ∧ two_cent_coins >= 3) ∨
       (ways = 4 ∧ one_cent_coins >= 4 ∧ two_cent_coins >= 2) ∨
       (ways = 6 ∧ one_cent_coins >= 6 ∧ two_cent_coins >= 1) ∨
       (ways = 3 ∧ one_cent_coins >= 3 ∧ five_cent_coin >= 1) ∨
       (ways = 1 ∧ one_cent_coins >= 1 ∧ two_cent_coins >= 1 ∧ five_cent_coin >= 1)
   ) :=
  sorry

end xiao_wang_ways_to_make_8_cents_l31_31485


namespace total_production_l31_31563

theorem total_production (S : ℝ) 
  (h1 : 4 * S = 4400) : 
  4400 + S = 5500 := 
by
  sorry

end total_production_l31_31563


namespace ashok_average_marks_l31_31286

theorem ashok_average_marks (avg_6 : ℝ) (marks_6 : ℝ) (total_sub : ℕ) (sub_6 : ℕ)
  (h1 : avg_6 = 75) (h2 : marks_6 = 80) (h3 : total_sub = 6) (h4 : sub_6 = 5) :
  (avg_6 * total_sub - marks_6) / sub_6 = 74 :=
by
  sorry

end ashok_average_marks_l31_31286


namespace leon_total_payment_l31_31122

noncomputable def total_payment : ℕ :=
let toy_organizers_cost := 78 * 3 in
let gaming_chairs_cost := 83 * 2 in
let total_orders := toy_organizers_cost + gaming_chairs_cost in
let delivery_fee := total_orders * 5 / 100 in
total_orders + delivery_fee

theorem leon_total_payment : total_payment = 420 :=
by
  sorry

end leon_total_payment_l31_31122


namespace total_surface_area_of_cylinder_l31_31506

noncomputable def rectangle_length : ℝ := 4 * Real.pi
noncomputable def rectangle_width : ℝ := 2

noncomputable def cylinder_radius (length : ℝ) : ℝ := length / (2 * Real.pi)
noncomputable def cylinder_height (width : ℝ) : ℝ := width

noncomputable def cylinder_surface_area (radius height : ℝ) : ℝ :=
  2 * Real.pi * radius^2 + 2 * Real.pi * radius * height

theorem total_surface_area_of_cylinder :
  cylinder_surface_area (cylinder_radius rectangle_length) (cylinder_height rectangle_width) = 16 * Real.pi :=
by
  sorry

end total_surface_area_of_cylinder_l31_31506


namespace zookeeper_configurations_l31_31043

theorem zookeeper_configurations :
  ∃ (configs : ℕ), configs = 3 ∧ 
  (∀ (r p : ℕ), 
    30 * r + 35 * p = 1400 ∧ p ≥ r → 
    ((r, p) = (7, 34) ∨ (r, p) = (14, 28) ∨ (r, p) = (21, 22))) :=
sorry

end zookeeper_configurations_l31_31043


namespace min_value_4x2_plus_y2_l31_31942

theorem min_value_4x2_plus_y2 {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 6) : 
  4 * x^2 + y^2 ≥ 18 := by
  sorry

end min_value_4x2_plus_y2_l31_31942


namespace initial_trucks_l31_31997

def trucks_given_to_Jeff : ℕ := 13
def trucks_left_with_Sarah : ℕ := 38

theorem initial_trucks (initial_trucks_count : ℕ) :
  initial_trucks_count = trucks_given_to_Jeff + trucks_left_with_Sarah → initial_trucks_count = 51 :=
by
  sorry

end initial_trucks_l31_31997


namespace adjacent_abby_bridget_probability_l31_31283
open Nat

-- Define the conditions
def total_kids := 6
def grid_rows := 3
def grid_cols := 2
def middle_row := 2
def abby_and_bridget := 2

-- Define the probability calculation
theorem adjacent_abby_bridget_probability :
  let total_arrangements := 6!
  let num_ways_adjacent :=
    (2 * abby_and_bridget) * (total_kids - abby_and_bridget)!
  let total_outcomes := total_arrangements
  (num_ways_adjacent / total_outcomes : ℚ) = 4 / 15
:= sorry

end adjacent_abby_bridget_probability_l31_31283


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31435

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l31_31435


namespace arithmetic_sequence_eighth_term_l31_31173

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l31_31173


namespace smallest_three_digit_number_with_property_l31_31928

theorem smallest_three_digit_number_with_property :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 999 ∧ (∃ (n : ℕ), 317 ≤ n ∧ n ≤ 999 ∧ 1001 * a + 1 = n^2) ∧ a = 183 :=
by
  sorry

end smallest_three_digit_number_with_property_l31_31928


namespace arithmetic_sequence_8th_term_l31_31200

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l31_31200


namespace A_should_shoot_air_l31_31616

-- Define the problem conditions
def hits_A : ℝ := 0.3
def hits_B : ℝ := 1
def hits_C : ℝ := 0.5

-- Define turns
inductive Turn
| A | B | C

-- Define the strategic choice
inductive Strategy
| aim_C | aim_B | shoot_air

-- Define the outcome structure
structure DuelOutcome where
  winner : Option Turn
  probability : ℝ

-- Noncomputable definition given the context of probabilistic reasoning
noncomputable def maximize_survival : Strategy := 
sorry

-- Main theorem to prove the optimal strategy
theorem A_should_shoot_air : maximize_survival = Strategy.shoot_air := 
sorry

end A_should_shoot_air_l31_31616


namespace arithmetic_sequence_8th_term_l31_31157

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l31_31157


namespace derivative_at_neg_one_l31_31965

def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 1

theorem derivative_at_neg_one : deriv f (-1) = -1 :=
by
  -- definition of the function
  -- proof of the statement
  sorry

end derivative_at_neg_one_l31_31965


namespace beta_max_success_ratio_l31_31702

-- Define Beta's score conditions
variables (a b c d : ℕ)
def beta_score_conditions :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) ∧
  (a * 25 < b * 9) ∧
  (c * 25 < d * 17) ∧
  (b + d = 600)

-- Define Beta's success ratio
def beta_success_ratio :=
  (a + c) / 600

theorem beta_max_success_ratio :
  beta_score_conditions a b c d →
  beta_success_ratio a c ≤ 407 / 600 :=
sorry

end beta_max_success_ratio_l31_31702


namespace right_triangle_hypotenuse_l31_31276

noncomputable def triangle_hypotenuse (a b c : ℝ) : Prop :=
(a + b + c = 40) ∧
(a * b = 48) ∧
(a^2 + b^2 = c^2) ∧
(c = 18.8)

theorem right_triangle_hypotenuse :
  ∃ (a b c : ℝ), triangle_hypotenuse a b c :=
by
  sorry

end right_triangle_hypotenuse_l31_31276


namespace number_is_48_l31_31803

theorem number_is_48 (x : ℝ) (h : (1/4) * x + 15 = 27) : x = 48 :=
by sorry

end number_is_48_l31_31803


namespace park_area_l31_31746

theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w = 80) (h2 : l = 3 * w) : l * w = 300 :=
sorry

end park_area_l31_31746


namespace range_of_a_l31_31931

theorem range_of_a (a : ℝ) : (∀ x > 0, a - x - |Real.log x| ≤ 0) → a ≤ 1 := by
  sorry

end range_of_a_l31_31931


namespace smallest_positive_period_of_f_max_min_values_of_f_l31_31338

noncomputable def f (x : ℝ) : ℝ := sin (x / 2) ^ 2 + sqrt 3 * sin (x / 2) * cos (x / 2)

theorem smallest_positive_period_of_f : ∀ (x : ℝ), f (x + 2 * π) = f x := by
  sorry

theorem max_min_values_of_f : 
  ∀ (x : ℝ), (π / 2 ≤ x ∧ x ≤ π) → (1 ≤ f x ∧ f x ≤ 3 / 2) := by
  sorry

end smallest_positive_period_of_f_max_min_values_of_f_l31_31338


namespace tank_capacity_l31_31641

theorem tank_capacity (fill_rate drain_rate1 drain_rate2 : ℝ)
  (initial_fullness : ℝ) (time_to_fill : ℝ) (capacity_in_liters : ℝ) :
  fill_rate = 1 / 2 ∧
  drain_rate1 = 1 / 4 ∧
  drain_rate2 = 1 / 6 ∧ 
  initial_fullness = 1 / 2 ∧ 
  time_to_fill = 60 →
  capacity_in_liters = 10000 :=
by {
  sorry
}

end tank_capacity_l31_31641


namespace probability_three_heads_l31_31020

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability (n : ℕ) (k : ℕ) : ℚ :=
  (binom n k) / (2 ^ n)

theorem probability_three_heads : probability 12 3 = 55 / 1024 := 
by
  sorry

end probability_three_heads_l31_31020


namespace min_sales_required_l31_31517

-- Definitions from conditions
def old_salary : ℝ := 75000
def new_base_salary : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750

-- Statement to be proven
theorem min_sales_required (n : ℕ) :
  n ≥ ⌈(old_salary - new_base_salary) / (commission_rate * sale_amount)⌉₊ :=
sorry

end min_sales_required_l31_31517


namespace part1_part2_part3_l31_31319

open Real

-- Definition of "$k$-derived point"
def k_derived_point (P : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (P.1 + k * P.2, k * P.1 + P.2)

-- Problem statements to prove
theorem part1 :
  k_derived_point (-2, 3) 2 = (4, -1) :=
sorry

theorem part2 (P : ℝ × ℝ) (h : k_derived_point P 3 = (9, 11)) :
  P = (3, 2) :=
sorry

theorem part3 (b k : ℝ) (h1 : b > 0) (h2 : |k * b| ≥ 5 * b) :
  k ≥ 5 ∨ k ≤ -5 :=
sorry

end part1_part2_part3_l31_31319


namespace product_of_five_consecutive_is_divisible_by_sixty_l31_31417

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l31_31417


namespace max_tiles_l31_31598

/--
Given a rectangular floor of size 180 cm by 120 cm
and rectangular tiles of size 25 cm by 16 cm, prove that the maximum number of tiles
that can be accommodated on the floor without overlapping, where the tiles' edges
are parallel and abutting the edges of the floor and with no tile overshooting the edges,
is 49 tiles.
-/
theorem max_tiles (floor_len floor_wid tile_len tile_wid : ℕ) (h1 : floor_len = 180)
  (h2 : floor_wid = 120) (h3 : tile_len = 25) (h4 : tile_wid = 16) :
  ∃ max_tiles : ℕ, max_tiles = 49 :=
by
  sorry

end max_tiles_l31_31598


namespace divide_angle_into_parts_l31_31081

-- Definitions based on the conditions
def given_angle : ℝ := 19

/-- 
Theorem: An angle of 19 degrees can be divided into 19 equal parts using a compass and a ruler,
and each part will measure 1 degree.
-/
theorem divide_angle_into_parts (angle : ℝ) (n : ℕ) (h1 : angle = given_angle) (h2 : n = 19) : angle / n = 1 :=
by
  -- Proof to be filled out
  sorry

end divide_angle_into_parts_l31_31081


namespace curlers_count_l31_31310

theorem curlers_count (T P B G : ℕ) 
  (hT : T = 16)
  (hP : P = T / 4)
  (hB : B = 2 * P)
  (hG : G = T - (P + B)) : 
  G = 4 :=
by
  sorry

end curlers_count_l31_31310


namespace no_distinct_integers_cycle_l31_31136

theorem no_distinct_integers_cycle (p : ℤ → ℤ) 
  (x : ℕ → ℤ) (h_distinct : ∀ i j, i ≠ j → x i ≠ x j)
  (n : ℕ) (h_n_ge_3 : n ≥ 3)
  (hx_cycle : ∀ i, i < n → p (x i) = x (i + 1) % n) :
  false :=
sorry

end no_distinct_integers_cycle_l31_31136


namespace sean_whistles_l31_31730

def charles_whistles : ℕ := 128
def sean_more_whistles : ℕ := 95

theorem sean_whistles : charles_whistles + sean_more_whistles = 223 :=
by {
  sorry
}

end sean_whistles_l31_31730


namespace trees_occupy_area_l31_31502

theorem trees_occupy_area
  (length : ℕ) (width : ℕ) (number_of_trees : ℕ)
  (h_length : length = 1000)
  (h_width : width = 2000)
  (h_trees : number_of_trees = 100000) :
  (length * width) / number_of_trees = 20 := 
by
  sorry

end trees_occupy_area_l31_31502


namespace triangle_inscribed_angle_l31_31238

theorem triangle_inscribed_angle 
  (y : ℝ)
  (arc_PQ arc_QR arc_RP : ℝ)
  (h1 : arc_PQ = 2 * y + 40)
  (h2 : arc_QR = 3 * y + 15)
  (h3 : arc_RP = 4 * y - 40)
  (h4 : arc_PQ + arc_QR + arc_RP = 360) :
  ∃ angle_P : ℝ, angle_P = 64.995 := 
by 
  sorry

end triangle_inscribed_angle_l31_31238


namespace sum_of_roots_l31_31844

theorem sum_of_roots (y1 y2 k m : ℝ) (h1 : y1 ≠ y2) (h2 : 5 * y1^2 - k * y1 = m) (h3 : 5 * y2^2 - k * y2 = m) : 
  y1 + y2 = k / 5 := 
by
  sorry

end sum_of_roots_l31_31844


namespace number_of_B_students_l31_31701

-- Conditions
def prob_A (prob_B : ℝ) := 0.6 * prob_B
def prob_C (prob_B : ℝ) := 1.6 * prob_B
def prob_D (prob_B : ℝ) := 0.3 * prob_B

-- Total students
def total_students : ℝ := 50

-- Main theorem statement
theorem number_of_B_students (x : ℝ) (h1 : prob_A x + x + prob_C x + prob_D x = total_students) :
  x = 14 :=
  by
-- Proof skipped
  sorry

end number_of_B_students_l31_31701


namespace arithmetic_seq_8th_term_l31_31144

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l31_31144


namespace circle_condition_l31_31104

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  -- Define constants and equation representation
  let d : ℝ := -2
  let e : ℝ := -4
  let f : ℝ := m
  -- Use the condition for the circle equation
  have h : d^2 + e^2 - 4*f > 0 := sorry
  -- Prove the inequality
  sorry

end circle_condition_l31_31104


namespace cosine_identity_l31_31939

theorem cosine_identity (alpha : ℝ) (h1 : -180 < alpha ∧ alpha < -90)
  (cos_75_alpha : Real.cos (75 * Real.pi / 180 + alpha) = 1 / 3) :
  Real.cos (15 * Real.pi / 180 - alpha) = -2 * Real.sqrt 2 / 3 := by
sorry

end cosine_identity_l31_31939


namespace general_form_of_quadratic_equation_l31_31226

noncomputable def quadratic_equation_general_form (x : ℝ) : Prop :=
  (x + 3) * (x - 1) = 2 * x - 4

theorem general_form_of_quadratic_equation (x : ℝ) :
  quadratic_equation_general_form x → x^2 + 1 = 0 :=
sorry

end general_form_of_quadratic_equation_l31_31226


namespace cost_price_books_l31_31574

def cost_of_type_A (cost_A cost_B : ℝ) : Prop :=
  cost_A = cost_B + 15

def quantity_equal (cost_A cost_B : ℝ) : Prop :=
  675 / cost_A = 450 / cost_B

theorem cost_price_books (cost_A cost_B : ℝ) (h1 : cost_of_type_A cost_A cost_B) (h2 : quantity_equal cost_A cost_B) : 
  cost_A = 45 ∧ cost_B = 30 :=
by
  -- Proof omitted
  sorry

end cost_price_books_l31_31574


namespace sum_of_x_for_sqrt_eq_nine_l31_31251

theorem sum_of_x_for_sqrt_eq_nine :
  (∑ x in Finset.filter (λ x, (abs (x - 2) = 9)) (Finset.range 100), x) = 4 :=
sorry

end sum_of_x_for_sqrt_eq_nine_l31_31251


namespace contradiction_example_l31_31758

theorem contradiction_example (a b c : ℕ) : (¬ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) → (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) :=
by
  sorry

end contradiction_example_l31_31758


namespace fewer_bands_l31_31356

theorem fewer_bands (J B Y : ℕ) (h1 : J = B + 10) (h2 : B - 4 = 8) (h3 : Y = 24) :
  Y - J = 2 :=
sorry

end fewer_bands_l31_31356


namespace math_problem_l31_31960

theorem math_problem 
  (x y : ℝ) 
  (h : x^2 + y^2 - x * y = 1) 
  : (-2 ≤ x + y) ∧ (x^2 + y^2 ≤ 2) :=
by
  sorry

end math_problem_l31_31960


namespace work_completion_time_l31_31632

theorem work_completion_time (d : ℚ) : 
  (∀ (A B : ℚ), A = 30 ∧ B = 55 → d ≈ 330 / 17) :=
by
  intros A B h
  cases h with hA hB
  have A_work_rate : ℚ := 1 / A
  have B_work_rate : ℚ := 1 / B
  have combined_work_rate : ℚ := A_work_rate + B_work_rate
  have := calc 
    1 / combined_work_rate 
      = d : by  
        linarith 
        sorry 
  exact this

end work_completion_time_l31_31632


namespace number_of_people_in_group_l31_31269

-- Definitions and conditions
def total_cost : ℕ := 94
def mango_juice_cost : ℕ := 5
def pineapple_juice_cost : ℕ := 6
def pineapple_cost_total : ℕ := 54

-- Theorem statement to prove
theorem number_of_people_in_group : 
  ∃ M P : ℕ, 
    mango_juice_cost * M + pineapple_juice_cost * P = total_cost ∧ 
    pineapple_juice_cost * P = pineapple_cost_total ∧ 
    M + P = 17 := 
by 
  sorry

end number_of_people_in_group_l31_31269


namespace birds_are_crows_l31_31968

theorem birds_are_crows (total_birds pigeons crows sparrows parrots non_pigeons: ℕ)
    (h1: pigeons = 20)
    (h2: crows = 40)
    (h3: sparrows = 15)
    (h4: parrots = total_birds - pigeons - crows - sparrows)
    (h5: total_birds = pigeons + crows + sparrows + parrots)
    (h6: non_pigeons = total_birds - pigeons) :
    (crows * 100 / non_pigeons = 50) :=
by sorry

end birds_are_crows_l31_31968


namespace number_of_players_in_each_game_l31_31236

theorem number_of_players_in_each_game 
  (n : ℕ) (Hn : n = 30)
  (total_games : ℕ) (Htotal : total_games = 435) :
  2 = 2 :=
sorry

end number_of_players_in_each_game_l31_31236


namespace largest_divisor_of_five_consecutive_integers_l31_31427

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l31_31427


namespace half_of_number_l31_31889

theorem half_of_number (N : ℕ) (h : (4 / 15 * 5 / 7 * N) - (4 / 9 * 2 / 5 * N) = 24) : N / 2 = 945 :=
by
  sorry

end half_of_number_l31_31889


namespace pure_imaginary_number_l31_31984

open Complex -- Use the Complex module for complex numbers

theorem pure_imaginary_number (a : ℝ) (h : (a - 1 : ℂ).re = 0) : a = 1 :=
by
  -- This part of the proof is omitted hence we put sorry
  sorry

end pure_imaginary_number_l31_31984


namespace simplify_expr_to_polynomial_l31_31255

namespace PolynomialProof

-- Define the given polynomial expressions
def expr1 (x : ℕ) := (3 * x^2 + 4 * x + 8) * (x - 2)
def expr2 (x : ℕ) := (x - 2) * (x^2 + 5 * x - 72)
def expr3 (x : ℕ) := (4 * x - 15) * (x - 2) * (x + 6)

-- Define the full polynomial expression
def full_expr (x : ℕ) := expr1 x - expr2 x + expr3 x

-- Our goal is to prove that full_expr == 6 * x^3 - 4 * x^2 - 26 * x + 20
theorem simplify_expr_to_polynomial (x : ℕ) : 
  full_expr x = 6 * x^3 - 4 * x^2 - 26 * x + 20 := by
  sorry

end PolynomialProof

end simplify_expr_to_polynomial_l31_31255


namespace variance_transformed_list_l31_31951

noncomputable def stddev (xs : List ℝ) : ℝ := sorry
noncomputable def variance (xs : List ℝ) : ℝ := sorry

theorem variance_transformed_list :
  ∀ (a_1 a_2 a_3 a_4 a_5 : ℝ),
  stddev [a_1, a_2, a_3, a_4, a_5] = 2 →
  variance [3 * a_1 - 2, 3 * a_2 - 2, 3 * a_3 - 2, 3 * a_4 - 2, 3 * a_5 - 2] = 36 :=
by
  intros
  sorry

end variance_transformed_list_l31_31951


namespace min_blocks_to_remove_l31_31765

theorem min_blocks_to_remove (n : ℕ) (h : n = 59) : 
  ∃ (k : ℕ), k = 32 ∧ (∃ m, n = m^3 + k ∧ m^3 ≤ n) :=
by {
  sorry
}

end min_blocks_to_remove_l31_31765


namespace amaya_movie_watching_time_l31_31649

theorem amaya_movie_watching_time :
  let t1 := 30 + 5
  let t2 := 20 + 7
  let t3 := 10 + 12
  let t4 := 15 + 8
  let t5 := 25 + 15
  let t6 := 15 + 10
  t1 + t2 + t3 + t4 + t5 + t6 = 172 :=
by
  sorry

end amaya_movie_watching_time_l31_31649


namespace jeff_stars_l31_31663

noncomputable def eric_stars : ℕ := 4
noncomputable def chad_initial_stars : ℕ := 2 * eric_stars
noncomputable def chad_stars_after_sale : ℕ := chad_initial_stars - 2
noncomputable def total_stars : ℕ := 16
noncomputable def stars_eric_and_chad : ℕ := eric_stars + chad_stars_after_sale

theorem jeff_stars :
  total_stars - stars_eric_and_chad = 6 := 
by 
  sorry

end jeff_stars_l31_31663


namespace sqrt_sq_eq_l31_31621

theorem sqrt_sq_eq (x : ℝ) : (Real.sqrt x) ^ 2 = x := by
  sorry

end sqrt_sq_eq_l31_31621


namespace probability_final_roll_six_l31_31033

def roll_die : Int → Bool
| n => n >= 1 ∧ n <= 6

theorem probability_final_roll_six
    (p : Fin 6 → ℝ)
    (h : p 0 + p 1 + p 2 + p 3 + p 4 + p 5 = 1)
    (S : Fin 6 → ℝ)
    (n : ℕ)
    (Y : ℕ → ℝ)
    (H : Y n + S 6 >= 2019) :
  (∑ k in (Finset.range 6).map Fin.mk, (p k) / (7 - (k + 1))) > 1/6 :=
by
  sorry

end probability_final_roll_six_l31_31033


namespace fifth_power_last_digit_l31_31728

theorem fifth_power_last_digit (n : ℕ) : 
  (n % 10)^5 % 10 = n % 10 :=
by sorry

end fifth_power_last_digit_l31_31728


namespace standard_deviation_calculation_l31_31740

theorem standard_deviation_calculation : 
  let mean := 16.2 
  let stddev := 2.3 
  mean - 2 * stddev = 11.6 :=
by
  sorry

end standard_deviation_calculation_l31_31740


namespace baseball_fans_count_l31_31569

theorem baseball_fans_count
  (Y M R : ℕ) 
  (h1 : Y = (3 * M) / 2)
  (h2 : R = (5 * M) / 4)
  (hM : M = 104) :
  Y + M + R = 390 :=
by
  sorry 

end baseball_fans_count_l31_31569


namespace domain_of_f_l31_31216

def domain_f (x : ℝ) : Prop := x ≤ 4 ∧ x ≠ 1

theorem domain_of_f :
  {x : ℝ | ∃(h1 : 4 - x ≥ 0) (h2 : x - 1 ≠ 0), true} = {x : ℝ | domain_f x} :=
by
  sorry

end domain_of_f_l31_31216


namespace final_price_of_hat_is_correct_l31_31779

-- Definitions capturing the conditions.
def original_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

-- Calculations for the intermediate prices.
def price_after_first_discount : ℝ := original_price * (1 - first_discount_rate)
def final_price : ℝ := price_after_first_discount * (1 - second_discount_rate)

-- The theorem we need to prove.
theorem final_price_of_hat_is_correct : final_price = 9 := by
  sorry

end final_price_of_hat_is_correct_l31_31779


namespace birds_in_trees_l31_31391

def number_of_stones := 40
def number_of_trees := number_of_stones + 3 * number_of_stones
def combined_number := number_of_trees + number_of_stones
def number_of_birds := 2 * combined_number

theorem birds_in_trees : number_of_birds = 400 := by
  sorry

end birds_in_trees_l31_31391


namespace seatingArrangementsAreSix_l31_31794

-- Define the number of seating arrangements for 4 people around a round table
def numSeatingArrangements : ℕ :=
  3 * 2 * 1 -- Following the condition that the narrator's position is fixed

-- The main theorem stating the number of different seating arrangements
theorem seatingArrangementsAreSix : numSeatingArrangements = 6 :=
  by
    -- This is equivalent to following the explanation of solution which is just multiplying the numbers
    sorry

end seatingArrangementsAreSix_l31_31794


namespace carly_dog_count_l31_31533

theorem carly_dog_count (total_nails : ℕ) (three_legged_dogs : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) 
  (h2 : three_legged_dogs = 3) 
  (h3 : total_dogs * 4 - three_legged_dogs = 41 - 3 * three_legged_dogs) 
  : total_dogs = 11 :=
sorry

end carly_dog_count_l31_31533


namespace arithmetic_sequence_8th_term_l31_31164

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l31_31164


namespace arithmetic_sequence_8th_term_l31_31197

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l31_31197


namespace jinho_initial_money_l31_31360

variable (M : ℝ)

theorem jinho_initial_money :
  (M / 2 + 300) + (((M / 2 - 300) / 2) + 400) = M :=
by
  -- This proof is yet to be completed.
  sorry

end jinho_initial_money_l31_31360


namespace factorization_l31_31297

theorem factorization (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := 
by 
  sorry

end factorization_l31_31297


namespace alice_walk_time_l31_31648

theorem alice_walk_time (bob_time : ℝ) 
  (bob_distance : ℝ) 
  (alice_distance1 : ℝ) 
  (alice_distance2 : ℝ) 
  (time_ratio : ℝ) 
  (expected_alice_time : ℝ) :
  bob_time = 36 →
  bob_distance = 6 →
  alice_distance1 = 4 →
  alice_distance2 = 7 →
  time_ratio = 1 / 3 →
  expected_alice_time = 21 →
  (expected_alice_time = alice_distance2 / (alice_distance1 / (bob_time * time_ratio))) := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h3, h5]
  have h_speed : ℝ := alice_distance1 / (bob_time * time_ratio)
  rw [h4, h6]
  linarith [h_speed]

end alice_walk_time_l31_31648


namespace arithmetic_sequence_8th_term_l31_31159

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l31_31159


namespace probability_at_least_four_same_is_correct_l31_31074

noncomputable def probability_at_least_four_same (dice : Fin 5 → Fin 6) : ℚ :=
  -- Probability that all five dice show the same value
  (1 : ℚ) * (1/6 : ℚ) * (1/6 : ℚ) * (1/6 : ℚ) * (1/6 : ℚ) +
  -- Probability that exact four dice show the same value and the fifth is different
  5 * ((1/6 : ℚ) * (1/6 : ℚ) * (1/6 : ℚ)) * (5/6 : ℚ)

theorem probability_at_least_four_same_is_correct :
  ∀ (dice : Fin 5 → Fin 6), probability_at_least_four_same dice = 13/648 :=
by
  intro dice
  -- The proof would go here
  sorry

end probability_at_least_four_same_is_correct_l31_31074


namespace probability_roll_6_final_l31_31030

variable {Ω : Type*} [ProbabilitySpace Ω]

/-- Define the outcomes of rolls -/
noncomputable def diceRollPMF : PMF (Fin 6) :=
  PMF.ofFinset (finset.univ) (by simp; exact λ i, 1/6)

/-- Define the scenario and prove the required probability -/
theorem probability_roll_6_final {sum : ℕ} (h_sum : sum ≥ 2019) :
  (PMF.cond diceRollPMF (λ x, x = 5)).prob > 5/6 :=
sorry

end probability_roll_6_final_l31_31030


namespace soap_bubble_radius_l31_31639

/-- Given a spherical soap bubble that divides into two equal hemispheres, 
    each having a radius of 6 * (2 ^ (1 / 3)) cm, 
    show that the radius of the original bubble is also 6 * (2 ^ (1 / 3)) cm. -/
theorem soap_bubble_radius (r : ℝ) (R : ℝ) (π : ℝ) 
  (h_r : r = 6 * (2 ^ (1 / 3)))
  (h_volume_eq : (4 / 3) * π * R^3 = (4 / 3) * π * r^3) : 
  R = 6 * (2 ^ (1 / 3)) :=
by
  sorry

end soap_bubble_radius_l31_31639


namespace largest_divisor_of_5_consecutive_integers_l31_31478

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31478


namespace grain_distance_l31_31756

theorem grain_distance
    (d : ℝ) (v_church : ℝ) (v_cathedral : ℝ)
    (h_d : d = 400) (h_v_church : v_church = 20) (h_v_cathedral : v_cathedral = 25) :
    ∃ x : ℝ, x = 1600 / 9 ∧ v_church * x = v_cathedral * (d - x) :=
by
  sorry

end grain_distance_l31_31756


namespace envelope_weight_l31_31513

-- Define the conditions as constants
def total_weight_kg : ℝ := 7.48
def num_envelopes : ℕ := 880
def kg_to_g_conversion : ℝ := 1000

-- Calculate the total weight in grams
def total_weight_g : ℝ := total_weight_kg * kg_to_g_conversion

-- Define the expected weight of one envelope in grams
def expected_weight_one_envelope_g : ℝ := 8.5

-- The proof statement
theorem envelope_weight :
  total_weight_g / num_envelopes = expected_weight_one_envelope_g := by
  sorry

end envelope_weight_l31_31513


namespace arithmetic_seq_8th_term_l31_31146

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l31_31146


namespace simplify_radical_expr_l31_31789

-- Define the variables and expressions
variables {x : ℝ} (hx : 0 ≤ x) 

-- State the problem
theorem simplify_radical_expr (hx : 0 ≤ x) :
  (Real.sqrt (100 * x)) * (Real.sqrt (3 * x)) * (Real.sqrt (18 * x)) = 30 * x * Real.sqrt (6 * x) :=
sorry

end simplify_radical_expr_l31_31789


namespace arithmetic_sequence_8th_term_l31_31163

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l31_31163


namespace greatest_possible_integer_l31_31839

theorem greatest_possible_integer 
  (n k l : ℕ) 
  (h1 : n < 150) 
  (h2 : n = 9 * k - 2) 
  (h3 : n = 6 * l - 4) : 
  n = 146 := 
sorry

end greatest_possible_integer_l31_31839


namespace initial_birds_correct_l31_31614

def flown_away : ℝ := 8.0
def left_on_fence : ℝ := 4.0
def initial_birds : ℝ := flown_away + left_on_fence

theorem initial_birds_correct : initial_birds = 12.0 := by
  sorry

end initial_birds_correct_l31_31614


namespace test_question_count_l31_31235

theorem test_question_count :
  ∃ (x : ℕ), 
    (20 / x: ℚ) > 0.60 ∧ 
    (20 / x: ℚ) < 0.70 ∧ 
    (4 ∣ x) ∧ 
    x = 32 := 
by
  sorry

end test_question_count_l31_31235


namespace volume_of_revolution_l31_31929

theorem volume_of_revolution (a : ℝ) (h : 0 < a) :
  let x (θ : ℝ) := a * (1 + Real.cos θ) * Real.cos θ
  let y (θ : ℝ) := a * (1 + Real.cos θ) * Real.sin θ
  V = (8 / 3) * π * a^3 :=
sorry

end volume_of_revolution_l31_31929


namespace difference_high_low_score_l31_31768

theorem difference_high_low_score :
  ∀ (num_innings : ℕ) (total_runs : ℕ) (exc_total_runs : ℕ) (high_score : ℕ) (low_score : ℕ),
  num_innings = 46 →
  total_runs = 60 * 46 →
  exc_total_runs = 58 * 44 →
  high_score = 194 →
  total_runs - exc_total_runs = high_score + low_score →
  high_score - low_score = 180 :=
by
  intros num_innings total_runs exc_total_runs high_score low_score h_innings h_total h_exc_total h_high_sum h_difference
  sorry

end difference_high_low_score_l31_31768


namespace original_stickers_l31_31875

theorem original_stickers (x : ℕ) (h₁ : x * 3 / 4 * 4 / 5 = 45) : x = 75 :=
by
  sorry

end original_stickers_l31_31875


namespace shaded_hexagons_are_balanced_l31_31757

-- Definitions and conditions from the problem
def is_balanced (a b c : ℕ) : Prop :=
  (a = b ∧ b = c) ∨ (a ≠ b ∧ b ≠ c ∧ a ≠ c)

def hexagon_grid_balanced (grid : ℕ × ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ),
  (i % 2 = 0 ∧ grid (i, j) = grid (i, j + 1) ∧ grid (i, j + 1) = grid (i + 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i, j + 1) ∧ grid (i, j + 1) ≠ grid (i + 1, j + 1) ∧ grid (i, j) ≠ grid (i + 1, j + 1))
  ∨ (i % 2 ≠ 0 ∧ grid (i, j) = grid (i - 1, j) ∧ grid (i - 1, j) = grid (i - 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i - 1, j) ∧ grid (i - 1, j) ≠ grid (i - 1, j + 1) ∧ grid (i, j) ≠ grid (i - 1, j + 1))

theorem shaded_hexagons_are_balanced (grid : ℕ × ℕ → ℕ) (h_balanced : hexagon_grid_balanced grid) :
  is_balanced (grid (1, 1)) (grid (1, 10)) (grid (10, 10)) :=
sorry

end shaded_hexagons_are_balanced_l31_31757


namespace angelina_speed_l31_31653

theorem angelina_speed (v : ℝ) (h₁ : ∀ t : ℝ, t = 100 / v) (h₂ : ∀ t : ℝ, t = 180 / (2 * v)) 
  (h₃ : ∀ d t : ℝ, 100 / v - 40 = 180 / (2 * v)) : 
  2 * v = 1 / 2 :=
by
  sorry

end angelina_speed_l31_31653


namespace union_of_sets_l31_31946

def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B : Set ℝ := {x | x > 0}
def C : Set ℝ := {x | x > -2}

theorem union_of_sets (A B : Set ℝ) : (A ∪ B) = C :=
  sorry

end union_of_sets_l31_31946


namespace find_value_l31_31693

variable (y : ℝ) (Q : ℝ)
axiom condition : 5 * (3 * y + 7 * Real.pi) = Q

theorem find_value : 10 * (6 * y + 14 * Real.pi) = 4 * Q :=
by
  sorry

end find_value_l31_31693


namespace rectangle_area_l31_31505

-- Definitions from conditions:
def side_length : ℕ := 16 / 4
def area_B : ℕ := side_length * side_length
def probability_not_within_B : ℝ := 0.4666666666666667

-- Main statement to prove
theorem rectangle_area (A : ℝ) (h1 : side_length = 4)
 (h2 : area_B = 16)
 (h3 : probability_not_within_B = 0.4666666666666667) :
   A * 0.5333333333333333 = 16 → A = 30 :=
by
  intros h
  sorry


end rectangle_area_l31_31505


namespace min_value_of_f_solution_set_of_inequality_l31_31586

-- Define the given function f
def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- (1) Prove that the minimum value of y = f(x) is 3
theorem min_value_of_f : ∃ x : ℝ, f x = 3 := 
sorry

-- (2) Prove that the solution set of the inequality |f(x) - 6| ≤ 1 is [-10/3, -8/3] ∪ [0, 4/3]
theorem solution_set_of_inequality : 
  {x | |f x - 6| ≤ 1} = {x | -(10/3) ≤ x ∧ x ≤ -(8/3) ∨ 0 ≤ x ∧ x ≤ (4/3)} :=
sorry

end min_value_of_f_solution_set_of_inequality_l31_31586


namespace factorization_of_x10_minus_1024_l31_31299

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end factorization_of_x10_minus_1024_l31_31299


namespace largest_divisor_of_five_consecutive_integers_l31_31425

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l31_31425


namespace find_coefficient_c_l31_31657

theorem find_coefficient_c (c : ℚ) :
  (x : ℚ) → (P : ℚ → ℚ) → P x = x^4 + 3*x^3 + c*x^2 + 15*x + 20 → (P 3 = 0 → c = -227/9) :=
by
  sorry

end find_coefficient_c_l31_31657


namespace coin_order_correct_l31_31814

-- Define the coins
inductive Coin
| A | B | C | D | E
deriving DecidableEq

open Coin

-- Define the conditions
def covers (x y : Coin) : Prop :=
  (x = A ∧ y = B) ∨
  (x = C ∧ (y = A ∨ y = D)) ∨
  (x = D ∧ y = B) ∨
  (y = E ∧ x = C)

-- Define the order of coins from top to bottom as a list
def coinOrder : List Coin := [C, E, A, D, B]

-- Prove that the order is correct
theorem coin_order_correct :
  ∀ c₁ c₂ : Coin, c₁ ≠ c₂ → List.indexOf c₁ coinOrder < List.indexOf c₂ coinOrder ↔ covers c₁ c₂ :=
by
  sorry

end coin_order_correct_l31_31814


namespace simplify_sqrt_product_l31_31523

theorem simplify_sqrt_product (y : ℝ) (hy : 0 ≤ y) : 
  (√(48 * y) * √(18 * y) * √(50 * y)) = 120 * y * √(3 * y) := 
by
  sorry

end simplify_sqrt_product_l31_31523


namespace product_of_5_consecutive_integers_divisible_by_60_l31_31460

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l31_31460


namespace children_playing_tennis_l31_31697

theorem children_playing_tennis
  (Total : ℕ) (S : ℕ) (N : ℕ) (B : ℕ) (T : ℕ) 
  (hTotal : Total = 38) (hS : S = 21) (hN : N = 10) (hB : B = 12) :
  T = 38 - 21 + 12 - 10 :=
by
  sorry

end children_playing_tennis_l31_31697


namespace solve_inequality_l31_31856

variable {a x : ℝ}

theorem solve_inequality (h : a > 0) : 
  (ax^2 - (a + 1)*x + 1 < 0) ↔ 
    (if 0 < a ∧ a < 1 then 1 < x ∧ x < 1/a else 
     if a = 1 then false else 
     if a > 1 then 1/a < x ∧ x < 1 else true) :=
  sorry

end solve_inequality_l31_31856


namespace area_of_rectangular_field_l31_31722

theorem area_of_rectangular_field (W D : ℝ) (hW : W = 15) (hD : D = 17) :
  ∃ L : ℝ, (W * L = 120) ∧ D^2 = L^2 + W^2 :=
by 
  use 8
  sorry

end area_of_rectangular_field_l31_31722


namespace book_area_correct_l31_31770

def book_length : ℝ := 5
def book_width : ℝ := 10
def book_area (length : ℝ) (width : ℝ) : ℝ := length * width

theorem book_area_correct :
  book_area book_length book_width = 50 :=
by
  sorry

end book_area_correct_l31_31770


namespace range_of_k_l31_31680

-- Define the set M
def M := {x : ℝ | -1 ≤ x ∧ x ≤ 7}

-- Define the set N based on k
def N (k : ℝ) := {x : ℝ | k + 1 ≤ x ∧ x ≤ 2 * k - 1}

-- The main statement to prove
theorem range_of_k (k : ℝ) : M ∩ N k = ∅ → 6 < k :=
by
  -- skipping the proof as instructed
  sorry

end range_of_k_l31_31680


namespace law_school_student_count_l31_31633

theorem law_school_student_count 
    (business_students : ℕ)
    (sibling_pairs : ℕ)
    (selection_probability : ℚ)
    (L : ℕ)
    (h1 : business_students = 500)
    (h2 : sibling_pairs = 30)
    (h3 : selection_probability = 7.500000000000001e-5) :
    L = 8000 :=
by
  sorry

end law_school_student_count_l31_31633


namespace ratio_of_millipedes_l31_31769

-- Define the given conditions
def total_segments_needed : ℕ := 800
def first_millipede_segments : ℕ := 60
def millipedes_segments (x : ℕ) : ℕ := x
def ten_millipedes_segments : ℕ := 10 * 50

-- State the main theorem
theorem ratio_of_millipedes (x : ℕ) : 
  total_segments_needed = 60 + 2 * x + 10 * 50 →
  2 * x / 60 = 4 :=
sorry

end ratio_of_millipedes_l31_31769


namespace solve_for_x_l31_31739

-- Define the custom operation
def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Main statement to prove
theorem solve_for_x : (∃ x : ℝ, custom_mul 3 (custom_mul 4 x) = 10) ↔ (x = 7.5) :=
by
  sorry

end solve_for_x_l31_31739


namespace triangle_similarity_proof_l31_31848

-- Define a structure for points in a geometric space
structure Point : Type where
  x : ℝ
  y : ℝ
  deriving Inhabited

-- Define the conditions provided in the problem
variables (A B C D E H : Point)
variables (HD HE : ℝ)

-- Condition statements
def HD_dist := HD = 6
def HE_dist := HE = 3

-- Main theorem statement
theorem triangle_similarity_proof (BD DC AE EC BH AH : ℝ) 
  (h1 : HD = 6) (h2 : HE = 3) 
  (h3 : 2 * BH = AH) : 
  (BD * DC - AE * EC = 9 * BH + 27) :=
sorry

end triangle_similarity_proof_l31_31848


namespace sum_of_triangles_l31_31738

def triangle (a b c : ℕ) : ℕ := a * b + c

theorem sum_of_triangles :
  triangle 3 2 5 + triangle 4 1 7 = 22 :=
by
  sorry

end sum_of_triangles_l31_31738


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31456

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l31_31456


namespace gcd_lcm_product_l31_31543

theorem gcd_lcm_product (a b : ℕ) (ha : a = 18) (hb : b = 42) :
  Nat.gcd a b * Nat.lcm a b = 756 :=
by
  rw [ha, hb]
  sorry

end gcd_lcm_product_l31_31543


namespace plane_triangle_coverage_l31_31608

noncomputable def percentage_triangles_covered (a : ℝ) : ℝ :=
  let total_area := (4 * a) ^ 2
  let triangle_area := 10 * (1 / 2 * a^2)
  (triangle_area / total_area) * 100

theorem plane_triangle_coverage (a : ℝ) :
  abs (percentage_triangles_covered a - 31.25) < 0.75 :=
  sorry

end plane_triangle_coverage_l31_31608


namespace large_green_curlers_l31_31308

-- Define the number of total curlers
def total_curlers : ℕ := 16

-- Define the fraction for pink curlers
def pink_fraction : ℕ := 1 / 4

-- Define the number of pink curlers
def pink_curlers : ℕ := pink_fraction * total_curlers

-- Define the number of blue curlers
def blue_curlers : ℕ := 2 * pink_curlers

-- Define the total number of pink and blue curlers
def pink_and_blue_curlers : ℕ := pink_curlers + blue_curlers

-- Define the number of green curlers
def green_curlers : ℕ := total_curlers - pink_and_blue_curlers

-- Theorem stating the number of green curlers is 4
theorem large_green_curlers : green_curlers = 4 := by
  -- Proof would go here
  sorry

end large_green_curlers_l31_31308


namespace correct_quadratic_equation_l31_31572

-- Definitions based on conditions
def root_sum (α β : ℝ) := α + β = 8
def root_product (α β : ℝ) := α * β = 24

-- Main statement to be proven
theorem correct_quadratic_equation (α β : ℝ) (h1 : root_sum 5 3) (h2 : root_product (-6) (-4)) :
    (α - 5) * (α - 3) = 0 ∧ (α + 6) * (α + 4) = 0 → α * α - 8 * α + 24 = 0 :=
sorry

end correct_quadratic_equation_l31_31572


namespace solve_for_a_l31_31829

theorem solve_for_a
  (a x : ℚ)
  (h1 : (2 * a * x + 3) / (a - x) = 3 / 4)
  (h2 : x = 1) : a = -3 :=
by
  sorry

end solve_for_a_l31_31829


namespace find_number_l31_31805

theorem find_number (x : ℕ) (h : x / 4 + 15 = 27) : x = 48 :=
sorry

end find_number_l31_31805


namespace select_1996_sets_l31_31367

theorem select_1996_sets (k : ℕ) (sets : Finset (Finset ℕ)) (h : k > 1993006) (h_sets : sets.card = k) :
  ∃ (selected_sets : Finset (Finset ℕ)), selected_sets.card = 1996 ∧
  ∀ (x y z : Finset ℕ), x ∈ selected_sets → y ∈ selected_sets → z ∈ selected_sets → z = x ∪ y → false :=
sorry

end select_1996_sets_l31_31367


namespace sqrt_sequence_solution_l31_31312

theorem sqrt_sequence_solution : 
  (∃ x : ℝ, x = sqrt (18 + x) ∧ x > 0) → (∃ x : ℝ, x = 6) :=
by
  assume h,
  sorry

end sqrt_sequence_solution_l31_31312


namespace number_representation_correct_l31_31646

-- Conditions: 5 in both the tenths and hundredths places, 0 in remaining places.
def number : ℝ := 50.05

theorem number_representation_correct :
  number = 50.05 :=
by 
  -- The proof will show that the definition satisfies the condition.
  sorry

end number_representation_correct_l31_31646


namespace number_of_positive_real_solutions_l31_31955

noncomputable def p (x : ℝ) : ℝ := x^12 + 5 * x^11 + 20 * x^10 + 1300 * x^9 - 1105 * x^8

theorem number_of_positive_real_solutions : ∃! x : ℝ, 0 < x ∧ p x = 0 :=
sorry

end number_of_positive_real_solutions_l31_31955


namespace cost_of_one_each_l31_31405

theorem cost_of_one_each (x y z : ℝ) (h1 : 3 * x + 7 * y + z = 24) (h2 : 4 * x + 10 * y + z = 33) :
  x + y + z = 6 :=
sorry

end cost_of_one_each_l31_31405


namespace base9_minus_base6_to_decimal_l31_31070

theorem base9_minus_base6_to_decimal :
  let b9 := 3 * 9^2 + 2 * 9^1 + 1 * 9^0
  let b6 := 2 * 6^2 + 5 * 6^1 + 4 * 6^0
  b9 - b6 = 156 := by
sorry

end base9_minus_base6_to_decimal_l31_31070


namespace dodecahedron_society_proof_l31_31264

open Fin

-- Definitions based on conditions provided
def individuals : Fin 12 := default

def adjacency (i j : Fin 12) : Prop :=
  -- Define adjacency based on adjacency of faces in a dodecahedron.
  sorry

def acquaintances : Fin 12 → Finset (Fin 12) :=
  λ i, {j | adjacency i j}

-- Conditions
def condition_a : Prop :=
  ∀ i : Fin 12, (acquaintances i).card = 5  -- known to exactly 6 people

def condition_b : Prop :=
  ∀ i : Fin 12, ∃ j k : Fin 12, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ adjacency i j ∧ adjacency i k ∧ adjacency j k  -- trio of mutually acquainted people

def condition_c : Prop :=
  ∀ a b c d : Fin 12, not (adjacency a b ∧ adjacency a c ∧ adjacency a d ∧ adjacency b c ∧ adjacency b d ∧ adjacency c d)

def condition_d : Prop :=
  ∀ a b c d : Fin 12, not (not (adjacency a b) ∧ not (adjacency a c) ∧ not (adjacency a d) ∧ not (adjacency b c) ∧ not (adjacency b d) ∧ not (adjacency c d))

def condition_e : Prop :=
  ∀ i : Fin 12, ∃ j k : Fin 12, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ not (adjacency i j) ∧ not (adjacency i k) ∧ not (adjacency j k) -- trio of mutually unacquainted people

def condition_f : Prop :=
  ∀ i : Fin 12, ∃ j : Fin 12, not (adjacency i j) ∧ ∀ k : Fin 12, not (adjacency j k) -- someone who has no mutual acquaintances

-- Theorem to be proven equivalent to the given problem
theorem dodecahedron_society_proof :
  condition_a ∧ condition_b ∧ condition_c ∧ condition_d ∧ condition_e ∧ condition_f :=
sorry

end dodecahedron_society_proof_l31_31264


namespace distance_between_stripes_l31_31281

/-- Given a crosswalk parallelogram with curbs 60 feet apart, a base of 20 feet, 
and each stripe of length 50 feet, show that the distance between the stripes is 24 feet. -/
theorem distance_between_stripes (h : Real) (b : Real) (s : Real) : h = 60 ∧ b = 20 ∧ s = 50 → (b * h) / s = 24 :=
by
  sorry

end distance_between_stripes_l31_31281


namespace solve_for_x_l31_31735

-- We define that the condition and what we need to prove.
theorem solve_for_x (x : ℝ) : (x + 7) / (x - 4) = (x - 3) / (x + 6) → x = -3 / 2 :=
by sorry

end solve_for_x_l31_31735


namespace proof_problem_l31_31114

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {2, 3, 4}

-- The complement of B with respect to U
def complement_U_B : Set ℕ := U \ B

-- The intersection of A and the complement of B with respect to U
def intersection_A_complement_U_B : Set ℕ := A ∩ complement_U_B

-- The statement we want to prove
theorem proof_problem : intersection_A_complement_U_B = {1, 6} :=
by
  sorry

end proof_problem_l31_31114


namespace evaluate_g_at_3_l31_31098

def g (x : ℤ) : ℤ := 3 * x^3 + 5 * x^2 - 2 * x - 7

theorem evaluate_g_at_3 : g 3 = 113 := by
  -- Proof of g(3) = 113 skipped
  sorry

end evaluate_g_at_3_l31_31098


namespace question_2024_polynomials_l31_31584

open Polynomial

noncomputable def P (x : ℝ) : Polynomial ℝ := sorry
noncomputable def Q (x : ℝ) : Polynomial ℝ := sorry

-- Main statement
theorem question_2024_polynomials (P Q : Polynomial ℝ) (hP : P.degree = 2024) (hQ : Q.degree = 2024)
    (hPm : P.leadingCoeff = 1) (hQm : Q.leadingCoeff = 1) (h : ∀ x : ℝ, P.eval x ≠ Q.eval x) :
    ∀ (α : ℝ), α ≠ 0 → ∃ x : ℝ, P.eval (x - α) = Q.eval (x + α) :=
by
  sorry

end question_2024_polynomials_l31_31584


namespace basket_weight_l31_31135

variables 
  (B : ℕ) -- Weight of the basket
  (L : ℕ) -- Lifting capacity of one balloon

-- Condition: One balloon can lift a basket with contents weighing not more than 80 kg
axiom one_balloon_lifts (h1 : B + L ≤ 80) : Prop

-- Condition: Two balloons can lift a basket with contents weighing not more than 180 kg
axiom two_balloons_lift (h2 : B + 2 * L ≤ 180) : Prop

-- The proof problem: Determine B under the given conditions
theorem basket_weight (B : ℕ) (L : ℕ) (h1 : B + L ≤ 80) (h2 : B + 2 * L ≤ 180) : B = 20 :=
  sorry

end basket_weight_l31_31135


namespace expected_digits_on_20_sided_die_l31_31499

theorem expected_digits_on_20_sided_die : 
  let num_faces := 20 in 
  let one_digit_prob := 9 / num_faces in 
  let two_digit_prob := 11 / num_faces in 
  let expected_value := (one_digit_prob * 1) + (two_digit_prob * 2) in
  expected_value = 1.55 := 
by
  sorry

end expected_digits_on_20_sided_die_l31_31499


namespace largest_divisor_of_5_consecutive_integers_l31_31479

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31479


namespace number_of_birds_is_400_l31_31390

-- Definitions of the problem
def num_stones : ℕ := 40
def num_trees : ℕ := 3 * num_stones + num_stones
def combined_trees_stones : ℕ := num_trees + num_stones
def num_birds : ℕ := 2 * combined_trees_stones

-- Statement to prove
theorem number_of_birds_is_400 : num_birds = 400 := by
  sorry

end number_of_birds_is_400_l31_31390


namespace area_of_rectangle_l31_31507

-- Define the conditions
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)
def area (length width : ℕ) : ℕ := length * width

-- Assumptions based on the problem conditions
variable (length : ℕ) (width : ℕ) (P : ℕ) (A : ℕ)
variable (h1 : width = 25)
variable (h2 : P = 110)

-- Goal: Prove the area is 750 square meters
theorem area_of_rectangle : 
  ∃ l : ℕ, perimeter l 25 = 110 → area l 25 = 750 :=
by
  sorry

end area_of_rectangle_l31_31507


namespace percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l31_31345

variables (a b c d e : ℝ)

-- Conditions
def condition1 : Prop := c = 0.25 * a
def condition2 : Prop := c = 0.50 * b
def condition3 : Prop := d = 0.40 * a
def condition4 : Prop := d = 0.20 * b
def condition5 : Prop := e = 0.35 * d
def condition6 : Prop := e = 0.15 * c

-- Proof Problem Statements
theorem percent_of_a_is_b (h1 : condition1 a c) (h2 : condition2 c b) : b = 0.5 * a := sorry

theorem percent_of_d_is_c (h1 : condition1 a c) (h3 : condition3 a d) : c = 0.625 * d := sorry

theorem percent_of_d_is_e (h5 : condition5 e d) : e = 0.35 * d := sorry

end percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l31_31345


namespace remainder_when_divided_by_x_minus_3_l31_31881

open Polynomial

noncomputable def p : ℝ[X] := 4 * X^3 - 12 * X^2 + 16 * X - 20

theorem remainder_when_divided_by_x_minus_3 : eval 3 p = 28 := by
  sorry

end remainder_when_divided_by_x_minus_3_l31_31881


namespace range_of_a_l31_31337

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 1 then x else a * x^2 + 2 * x

theorem range_of_a (R : Set ℝ) :
  (∀ x : ℝ, f x a ∈ R) → (a ∈ Set.Icc (-1 : ℝ) 0) :=
sorry

end range_of_a_l31_31337


namespace a_2016_is_neg1_l31_31326

noncomputable def a : ℕ → ℤ
| 0     => 0 -- Arbitrary value for n = 0 since sequences generally start from 1 in Lean
| 1     => 1
| 2     => 2
| n + 1 => a n - a (n - 1)

theorem a_2016_is_neg1 : a 2016 = -1 := sorry

end a_2016_is_neg1_l31_31326


namespace gold_common_difference_l31_31659

theorem gold_common_difference :
  (∃ (a : ℚ) (d : ℚ),
    let a1 := a + 9 * d,
        a2 := a + 8 * d,
        a3 := a + 7 * d,
        a4 := a + 6 * d,
        a5 := a + 5 * d,
        a6 := a + 4 * d,
        a7 := a + 3 * d,
        a8 := a + 2 * d,
        a9 := a + d,
        a10 := a in
    (a8 + a9 + a10 = 4) ∧
    (a1 + a2 + a3 + a4 + a5 + a6 + a7 = 3)) →
  ∃ d : ℚ, d = 7 / 78 :=
by
  intro h
  sorry

end gold_common_difference_l31_31659


namespace actual_average_speed_l31_31761

theorem actual_average_speed (v t : ℝ) (h1 : v > 0) (h2: t > 0) (h3 : (t / (t - (1 / 4) * t)) = ((v + 12) / v)) : v = 36 :=
by
  sorry

end actual_average_speed_l31_31761


namespace median_books_read_l31_31864

def num_students_per_books : List (ℕ × ℕ) := [(2, 8), (3, 5), (4, 9), (5, 3)]

theorem median_books_read 
  (h : num_students_per_books = [(2, 8), (3, 5), (4, 9), (5, 3)]) : 
  ∃ median, median = 3 :=
by {
  sorry
}

end median_books_read_l31_31864


namespace minimum_value_is_six_l31_31712

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z)

theorem minimum_value_is_six
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + z = 9) (h2 : y = 2 * x) :
  minimum_value_expression x y z = 6 :=
by
  sorry

end minimum_value_is_six_l31_31712


namespace proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l31_31708

noncomputable def sin_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A + Real.sin B + Real.sin C) ≤ (3 / 2) * Real.sqrt 3

noncomputable def sin_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A * Real.sin B * Real.sin C) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_sum_double_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C)) ≥ (-3 / 2)

noncomputable def cos_square_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2) ≥ (3 / 4)

noncomputable def cos_half_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A * Real.cos B * Real.cos C) ≤ (1 / 8)

theorem proof_sin_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_sum_ineq A B C hABC := sorry

theorem proof_sin_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_product_ineq A B C hABC := sorry

theorem proof_cos_sum_double_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_sum_double_ineq A B C hABC := sorry

theorem proof_cos_square_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_square_sum_ineq A B C hABC := sorry

theorem proof_cos_half_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_half_product_ineq A B C hABC := sorry

theorem proof_cos_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_product_ineq A B C hABC := sorry

end proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l31_31708


namespace range_of_B_l31_31696

theorem range_of_B (a b c : ℝ) (h : a + c = 2 * b) :
  ∃ B : ℝ, 0 < B ∧ B ≤ π / 3 ∧
  ∃ A C : ℝ, ∃ ha : a = c, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π :=
sorry

end range_of_B_l31_31696


namespace percentage_within_one_standard_deviation_l31_31700

-- Define the constants
def m : ℝ := sorry     -- mean
def g : ℝ := sorry     -- standard deviation
def P : ℝ → ℝ := sorry -- cumulative distribution function

-- The condition that 84% of the distribution is less than m + g
def condition1 : Prop := P (m + g) = 0.84

-- The condition that the distribution is symmetric about the mean
def symmetric_distribution (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P (m + (m - x)) = 1 - P x

-- The problem asks to prove that 68% of the distribution lies within one standard deviation of the mean
theorem percentage_within_one_standard_deviation 
  (h₁ : condition1)
  (h₂ : symmetric_distribution P m) : 
  P (m + g) - P (m - g) = 0.68 :=
sorry

end percentage_within_one_standard_deviation_l31_31700


namespace domain_of_sqrt_log_function_l31_31380

def domain_of_function (x : ℝ) : Prop :=
  (1 ≤ x ∧ x < 2) ∨ (2 < x ∧ x < 3)

theorem domain_of_sqrt_log_function :
  ∀ x : ℝ, (x - 1 ≥ 0) → (x - 2 ≠ 0) → (-x^2 + 2 * x + 3 > 0) →
    domain_of_function x :=
by
  intros x h1 h2 h3
  unfold domain_of_function
  sorry

end domain_of_sqrt_log_function_l31_31380


namespace age_ratio_l31_31935
open Nat

theorem age_ratio (B A x : ℕ) (h1 : B - 4 = 2 * (A - 4)) 
                                (h2 : B - 8 = 3 * (A - 8)) 
                                (h3 : (B + x) / (A + x) = 3 / 2) : 
                                x = 4 :=
by
  sorry

end age_ratio_l31_31935


namespace max_sum_n_value_l31_31082

open Nat

-- Definitions for the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2

-- Statement of the theorem
theorem max_sum_n_value (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : arithmetic_sequence a d) 
  (h_initial : a 0 > 0) (h_condition : 8 * a 4 = 13 * a 10) : 
  ∃ n, sum_of_first_n_terms a n = max (sum_of_first_n_terms a n) ∧ n = 20 :=
sorry

end max_sum_n_value_l31_31082


namespace trapezoid_area_l31_31393

-- Definitions based on the problem conditions
def Vertex := (Real × Real)

structure Triangle :=
(A : Vertex)
(B : Vertex)
(C : Vertex)
(area : Real)

structure Trapezoid :=
(AB : Real)
(CD : Real)
(M : Vertex)
(area_triangle_ABM : Real)
(area_triangle_CDM : Real)

-- The main theorem we want to prove
theorem trapezoid_area (T : Trapezoid)
  (parallel_sides : T.AB < T.CD)
  (intersect_at_M : ∃ M : Vertex, M = T.M)
  (area_ABM : T.area_triangle_ABM = 2)
  (area_CDM : T.area_triangle_CDM = 8) :
  T.AB * T.CD / (T.CD - T.AB) + T.CD * T.AB / (T.CD - T.AB) = 18 :=
sorry

end trapezoid_area_l31_31393


namespace smallest_common_multiple_gt_50_l31_31249

theorem smallest_common_multiple_gt_50 (a b : ℕ) (h1 : a = 15) (h2 : b = 20) : 
    ∃ x, x > 50 ∧ Nat.lcm a b = x := by
  have h_lcm : Nat.lcm a b = 60 := by sorry
  use 60
  exact ⟨by decide, h_lcm⟩

end smallest_common_multiple_gt_50_l31_31249


namespace certain_number_is_l31_31870

theorem certain_number_is (x : ℝ) : 
  x * (-4.5) = 2 * (-4.5) - 36 → x = 10 :=
by
  intro h
  -- proof goes here
  sorry

end certain_number_is_l31_31870


namespace jon_buys_2_coffees_each_day_l31_31840

-- Define the conditions
def cost_per_coffee : ℕ := 2
def total_spent : ℕ := 120
def days_in_april : ℕ := 30

-- Define the total number of coffees bought
def total_coffees_bought : ℕ := total_spent / cost_per_coffee

-- Prove that Jon buys 2 coffees each day
theorem jon_buys_2_coffees_each_day : total_coffees_bought / days_in_april = 2 := by
  sorry

end jon_buys_2_coffees_each_day_l31_31840


namespace complement_union_l31_31822

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) 
  (hM : M = {1, 3, 5, 7}) (hN : N = {5, 6, 7}) : U \ (M ∪ N) = {2, 4, 8} :=
by
  sorry

end complement_union_l31_31822


namespace quadratic_two_equal_real_roots_c_l31_31932

theorem quadratic_two_equal_real_roots_c (c : ℝ) : 
  (∃ x : ℝ, (2*x^2 - x + c = 0) ∧ (∃ y : ℝ, y ≠ x ∧ 2*y^2 - y + c = 0)) →
  c = 1/8 :=
sorry

end quadratic_two_equal_real_roots_c_l31_31932


namespace necklaces_caught_l31_31658

noncomputable def total_necklaces_caught (boudreaux rhonda latch cecilia : ℕ) : ℕ :=
  boudreaux + rhonda + latch + cecilia

theorem necklaces_caught :
  ∃ (boudreaux rhonda latch cecilia : ℕ), 
    boudreaux = 12 ∧
    rhonda = boudreaux / 2 ∧
    latch = 3 * rhonda - 4 ∧
    cecilia = latch + 3 ∧
    total_necklaces_caught boudreaux rhonda latch cecilia = 49 ∧
    (total_necklaces_caught boudreaux rhonda latch cecilia) % 7 = 0 :=
by
  sorry

end necklaces_caught_l31_31658


namespace arithmetic_seq_8th_term_l31_31145

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l31_31145


namespace shaded_area_of_squares_is_20_l31_31640

theorem shaded_area_of_squares_is_20 :
  ∀ (a b : ℝ), a = 2 → b = 6 → 
    (1/2) * a * a + (1/2) * b * b = 20 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end shaded_area_of_squares_is_20_l31_31640


namespace batsman_average_increase_l31_31021

theorem batsman_average_increase :
  ∀ (A : ℝ), (10 * A + 110 = 11 * 60) → (60 - A = 5) :=
by
  intros A h
  -- Proof goes here
  sorry

end batsman_average_increase_l31_31021


namespace gcd_linear_combination_l31_31314

theorem gcd_linear_combination (a b : ℤ) (h : Int.gcd a b = 1) : 
    Int.gcd (11 * a + 2 * b) (18 * a + 5 * b) = 1 := 
by
  sorry

end gcd_linear_combination_l31_31314


namespace new_area_rhombus_l31_31346

theorem new_area_rhombus (d1 d2 : ℝ) (h : (d1 * d2) / 2 = 3) : 
  ((5 * d1) * (5 * d2)) / 2 = 75 := 
by
  sorry

end new_area_rhombus_l31_31346


namespace geo_sequence_ratio_l31_31129

theorem geo_sequence_ratio
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (q : ℝ)
  (hq1 : q = 1 → S_8 = 8 * a_n 0 ∧ S_4 = 4 * a_n 0 ∧ S_8 = 2 * S_4)
  (hq2 : q ≠ 1 → S_8 = 2 * S_4 → false)
  (hS : ∀ n, S_n n = a_n 0 * (1 - q^n) / (1 - q))
  (h_condition : S_8 = 2 * S_4) :
  a_n 2 / a_n 0 = 1 := sorry

end geo_sequence_ratio_l31_31129


namespace cost_difference_proof_l31_31320

-- Define the cost per copy at print shop X
def cost_per_copy_X : ℝ := 1.25

-- Define the cost per copy at print shop Y
def cost_per_copy_Y : ℝ := 2.75

-- Define the number of copies
def number_of_copies : ℝ := 60

-- Define the total cost at print shop X
def total_cost_X : ℝ := cost_per_copy_X * number_of_copies

-- Define the total cost at print shop Y
def total_cost_Y : ℝ := cost_per_copy_Y * number_of_copies

-- Define the difference in cost between print shop Y and print shop X
def cost_difference : ℝ := total_cost_Y - total_cost_X

-- The theorem statement proving the cost difference is $90
theorem cost_difference_proof : cost_difference = 90 := by
  sorry

end cost_difference_proof_l31_31320


namespace function_property_l31_31331

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem function_property
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_property : ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0)
  : f (-4) > f (-6) :=
sorry

end function_property_l31_31331


namespace Kim_nail_polishes_l31_31121

-- Define the conditions
variable (K : ℕ)
def Heidi_nail_polishes (K : ℕ) : ℕ := K + 5
def Karen_nail_polishes (K : ℕ) : ℕ := K - 4

-- The main statement to prove
theorem Kim_nail_polishes (K : ℕ) (H : Heidi_nail_polishes K + Karen_nail_polishes K = 25) : K = 12 := by
  sorry

end Kim_nail_polishes_l31_31121


namespace simplify_fraction_l31_31733

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2 - 4) = 272 / 59 :=
by
  sorry

end simplify_fraction_l31_31733


namespace clerical_percentage_after_reduction_l31_31487

-- Define the initial conditions
def total_employees : ℕ := 3600
def clerical_fraction : ℚ := 1/4
def reduction_fraction : ℚ := 1/4

-- Define the intermediate calculations
def initial_clerical_employees : ℚ := clerical_fraction * total_employees
def clerical_reduction : ℚ := reduction_fraction * initial_clerical_employees
def new_clerical_employees : ℚ := initial_clerical_employees - clerical_reduction
def total_employees_after_reduction : ℚ := total_employees - clerical_reduction

-- State the theorem
theorem clerical_percentage_after_reduction :
  (new_clerical_employees / total_employees_after_reduction) * 100 = 20 :=
sorry

end clerical_percentage_after_reduction_l31_31487


namespace intersection_complement_l31_31112

variable U A B : Set ℕ
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (A_def : A = {1, 3, 6})
variable (B_def : B = {2, 3, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 6} :=
by
  rw [U_def, A_def, B_def]
  simp
  sorry

end intersection_complement_l31_31112


namespace largest_angle_of_obtuse_isosceles_triangle_l31_31878

theorem largest_angle_of_obtuse_isosceles_triangle (P Q R : Type) 
  (triangle_PQR : Triangle P Q R)
  (isosceles_PQR : Isosceles triangle_PQR) 
  (obtuse_PQR : Obtuse triangle_PQR)
  (angle_P_30 : angle P triangle_PQR = 30) : 
  ∃ (angle_Q : ℕ), is_largest_angle angle_Q triangle_PQR ∧ angle_Q = 120 := 
by 
  sorry

end largest_angle_of_obtuse_isosceles_triangle_l31_31878


namespace arithmetic_sequence_8th_term_l31_31196

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l31_31196


namespace abs_neg_three_l31_31207

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l31_31207


namespace intersection_of_A_and_B_l31_31681

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 4} := by
  sorry

end intersection_of_A_and_B_l31_31681


namespace summer_discount_percentage_l31_31215

/--
Given:
1. The original cost of the jeans (original_price) is $49.
2. On Wednesdays, there is an additional $10.00 off on all jeans after the summer discount is applied.
3. Before the sales tax applies, the cost of a pair of jeans (final_price) is $14.50.

Prove:
The summer discount percentage (D) is 50%.
-/
theorem summer_discount_percentage (original_price final_price : ℝ) (D : ℝ) :
  original_price = 49 → 
  final_price = 14.50 → 
  (original_price - (original_price * D / 100) - 10 = final_price) → 
  D = 50 :=
by intros h_original h_final h_discount; sorry

end summer_discount_percentage_l31_31215


namespace ratio_difference_l31_31237

theorem ratio_difference (x : ℕ) (h : 7 * x = 70) : 70 - 3 * x = 40 :=
by
  -- proof would go here
  sorry

end ratio_difference_l31_31237


namespace arithmetic_sequence_8th_term_l31_31153

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l31_31153


namespace five_consecutive_product_div_24_l31_31443

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l31_31443


namespace cube_volume_equality_l31_31508

open BigOperators Real

-- Definitions
def initial_volume : ℝ := 1

def removed_volume (x : ℝ) : ℝ := x^2

def removed_volume_with_overlap (x y : ℝ) : ℝ := x^2 - (x^2 * y)

def remaining_volume (a b c : ℝ) : ℝ := 
  initial_volume - removed_volume c - removed_volume_with_overlap b c - removed_volume_with_overlap a c - removed_volume_with_overlap a b + (c^2 * b)

-- Main theorem to prove
theorem cube_volume_equality (c b a : ℝ) (hcb : c < b) (hba : b < a) (ha1 : a < 1):
  (c = 1 / 2) ∧ 
  (b = (1 + Real.sqrt 17) / 8) ∧ 
  (a = (17 + Real.sqrt 17 + Real.sqrt (1202 - 94 * Real.sqrt 17)) / 64) :=
sorry

end cube_volume_equality_l31_31508


namespace determinant_eval_l31_31541

open Matrix

noncomputable def matrix_example (α γ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin α, -Real.cos α],
    ![-Real.sin α, 0, 3 * Real.sin γ],
    ![2 * Real.cos α, -Real.sin γ, 0]]

theorem determinant_eval (α γ : ℝ) :
  det (matrix_example α γ) = 10 * Real.sin α * Real.sin γ * Real.cos α :=
sorry

end determinant_eval_l31_31541


namespace savings_proof_l31_31866

variable (income expenditure savings : ℕ)

def ratio_income_expenditure (i e : ℕ) := i / 10 = e / 7

theorem savings_proof (h : ratio_income_expenditure income expenditure) (hincome : income = 10000) :
  savings = income - expenditure → savings = 3000 :=
by
  sorry

end savings_proof_l31_31866


namespace minimum_attempts_to_make_radio_work_l31_31010

theorem minimum_attempts_to_make_radio_work : 
  ∃ n, n = 12 ∧ (∀ (batteries : Finset ℕ), batteries.card = 8 → 
  (∃ charged_batteries uncharged_batteries, 
    charged_batteries.card = 4 ∧ 
    uncharged_batteries.card = 4 ∧ 
    charged_batteries ∪ uncharged_batteries = batteries) → 
  (∀ attempts : Finset (Finset ℕ), 
    attempts.card = n ∧ 
    (∀ attempt ∈ attempts, attempt.card = 2) → 
    ∃ attempt ∈ attempts, 
      (attempt ⊆ charged_batteries) := 
  sorry

end minimum_attempts_to_make_radio_work_l31_31010


namespace min_value_of_expression_l31_31668

noncomputable def expression (x : ℝ) : ℝ := (15 - x) * (12 - x) * (15 + x) * (12 + x)

theorem min_value_of_expression :
  ∃ x : ℝ, (expression x) = -1640.25 :=
sorry

end min_value_of_expression_l31_31668


namespace trigonometric_identity_l31_31090

theorem trigonometric_identity (m : ℝ) (h : m < 0) :
  2 * (3 / -5) + 4 / -5 = -2 / 5 :=
by
  sorry

end trigonometric_identity_l31_31090


namespace internal_angle_sine_l31_31689

theorem internal_angle_sine (α : ℝ) (h1 : α > 0 ∧ α < 180) (h2 : Real.sin (α * (Real.pi / 180)) = 1 / 2) : α = 30 ∨ α = 150 :=
sorry

end internal_angle_sine_l31_31689


namespace brandon_businesses_l31_31289

theorem brandon_businesses (total_businesses: ℕ) (fire_fraction: ℚ) (quit_fraction: ℚ) 
  (h_total: total_businesses = 72) 
  (h_fire_fraction: fire_fraction = 1/2) 
  (h_quit_fraction: quit_fraction = 1/3) : 
  total_businesses - (total_businesses * fire_fraction + total_businesses * quit_fraction) = 12 :=
by 
  sorry

end brandon_businesses_l31_31289


namespace bill_apples_left_l31_31910

-- Definitions based on the conditions
def total_apples : Nat := 50
def apples_per_child : Nat := 3
def number_of_children : Nat := 2
def apples_per_pie : Nat := 10
def number_of_pies : Nat := 2

-- The main statement to prove
theorem bill_apples_left : total_apples - ((apples_per_child * number_of_children) + (apples_per_pie * number_of_pies)) = 24 := by
sorry

end bill_apples_left_l31_31910


namespace arithmetic_seq_8th_term_l31_31149

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l31_31149


namespace eq_frac_l31_31865

noncomputable def g : ℝ → ℝ := sorry

theorem eq_frac (h1 : ∀ c d : ℝ, c^3 * g d = d^3 * g c)
                (h2 : g 3 ≠ 0) : (g 7 - g 4) / g 3 = 279 / 27 :=
by
  sorry

end eq_frac_l31_31865


namespace product_of_five_consecutive_is_divisible_by_sixty_l31_31415

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l31_31415


namespace arrange_x_y_z_l31_31093

theorem arrange_x_y_z (x : ℝ) (hx : 0.9 < x ∧ x < 1) :
  let y := x^(1/x)
  let z := x^y
  x < z ∧ z < y :=
by
  let y := x^(1/x)
  let z := x^y
  have : 0.9 < x ∧ x < 1 := hx
  sorry

end arrange_x_y_z_l31_31093


namespace function_passes_through_point_l31_31118

theorem function_passes_through_point :
  (∃ (a : ℝ), a = 1 ∧ (∀ (x y : ℝ), y = a * x + a → y = x + 1)) →
  ∃ x y : ℝ, x = -2 ∧ y = -1 ∧ y = x + 1 :=
by
  sorry

end function_passes_through_point_l31_31118


namespace factorization_of_x10_minus_1024_l31_31298

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end factorization_of_x10_minus_1024_l31_31298


namespace cylinder_volume_increase_factor_l31_31885

theorem cylinder_volume_increase_factor
    (π : Real)
    (r h : Real)
    (V_original : Real := π * r^2 * h)
    (new_height : Real := 3 * h)
    (new_radius : Real := 4 * r)
    (V_new : Real := π * (new_radius)^2 * new_height) :
    V_new / V_original = 48 :=
by
  sorry

end cylinder_volume_increase_factor_l31_31885


namespace range_a_l31_31115

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - a| + |x - 1| ≤ 3

theorem range_a (a : ℝ) : range_of_a a → -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_a_l31_31115


namespace simplify_expression_l31_31302

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  (3 * x - 1 - 5 * x) / 3 = -(2 / 3) * x - (1 / 3) := 
by
  sorry

end simplify_expression_l31_31302


namespace comparison_of_logs_l31_31333

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem comparison_of_logs : a > b ∧ b > c :=
by
  sorry

end comparison_of_logs_l31_31333


namespace maximal_q_for_broken_line_l31_31854

theorem maximal_q_for_broken_line :
  ∃ q : ℝ, (∀ i : ℕ, 0 ≤ i → i < 5 → ∀ A_i : ℝ, (A_i = q ^ i)) ∧ 
  (q = (1 + Real.sqrt 5) / 2) := sorry

end maximal_q_for_broken_line_l31_31854


namespace multiplication_result_l31_31099

theorem multiplication_result
  (h : 16 * 21.3 = 340.8) :
  213 * 16 = 3408 :=
sorry

end multiplication_result_l31_31099


namespace remainder_sum_div_7_l31_31811

theorem remainder_sum_div_7 :
  (8145 + 8146 + 8147 + 8148 + 8149) % 7 = 4 :=
by
  sorry

end remainder_sum_div_7_l31_31811


namespace barrels_of_pitch_needed_on_third_day_l31_31278

def road_length : ℕ := 16
def truckloads_per_mile : ℕ := 3
def bags_of_gravel_per_truckload : ℕ := 2
def gravel_bags_to_pitch_ratio : ℕ := 5
def paved_distance_day1 : ℕ := 4
def paved_distance_day2 : ℕ := 2 * paved_distance_day1 - 1

theorem barrels_of_pitch_needed_on_third_day :
  let paved_distance_first_two_days := paved_distance_day1 + paved_distance_day2 in
  let remaining_distance := road_length - paved_distance_first_two_days in
  let truckloads_needed := remaining_distance * truckloads_per_mile in
  let barrels_per_truckload := (bags_of_gravel_per_truckload : ℚ) / gravel_bags_to_pitch_ratio in
  let total_barrels_needed := truckloads_needed * barrels_per_truckload in
  total_barrels_needed = 6 := 
by
  sorry

end barrels_of_pitch_needed_on_third_day_l31_31278


namespace number_of_elements_in_M_l31_31386

theorem number_of_elements_in_M :
  (∃! (M : Finset ℕ), M = {m | ∃ (n : ℕ), n > 0 ∧ m = 2*n - 1 ∧ m < 60 } ∧ M.card = 30) :=
sorry

end number_of_elements_in_M_l31_31386


namespace solve_for_x_l31_31825

theorem solve_for_x (x : ℝ) : 
  x - 3 * x + 5 * x = 150 → x = 50 :=
by
  intro h
  -- sorry to skip the proof
  sorry

end solve_for_x_l31_31825


namespace no_nonzero_integers_satisfy_conditions_l31_31372

theorem no_nonzero_integers_satisfy_conditions :
  ¬ ∃ a b x y : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0) ∧ (a * x - b * y = 16) ∧ (a * y + b * x = 1) :=
by
  sorry

end no_nonzero_integers_satisfy_conditions_l31_31372


namespace arithmetic_sequence_8th_term_is_71_l31_31183

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l31_31183


namespace intersection_of_A_and_B_l31_31339

def U := Set ℝ
def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := {x : ℝ | x < -1}
def C := {x : ℝ | -2 ≤ x ∧ x < -1}

theorem intersection_of_A_and_B : A ∩ B = C :=
by sorry

end intersection_of_A_and_B_l31_31339


namespace abs_neg_three_l31_31205

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l31_31205


namespace clock_hands_overlap_l31_31259

theorem clock_hands_overlap (t : ℝ) :
  (∀ (h_angle m_angle : ℝ), h_angle = 30 + 0.5 * t ∧ m_angle = 6 * t ∧ h_angle = m_angle ∧ h_angle = 45) → t = 8 :=
by
  intro h
  sorry

end clock_hands_overlap_l31_31259


namespace total_adults_wearing_hats_l31_31721

theorem total_adults_wearing_hats (total_adults : ℕ) (men_percentage : ℝ) (men_hats_percentage : ℝ) 
  (women_hats_percentage : ℝ) (total_men_wearing_hats : ℕ) (total_women_wearing_hats : ℕ) : 
  (total_adults = 1200) ∧ (men_percentage = 0.60) ∧ (men_hats_percentage = 0.15) 
  ∧ (women_hats_percentage = 0.10)
     → total_men_wearing_hats + total_women_wearing_hats = 156 :=
by
  -- Definitions
  let total_men := total_adults * men_percentage
  let total_women := total_adults - total_men
  let men_wearing_hats := total_men * men_hats_percentage
  let women_wearing_hats := total_women * women_hats_percentage
  sorry

end total_adults_wearing_hats_l31_31721


namespace area_covered_three_layers_l31_31397

noncomputable def auditorium_width : ℕ := 10
noncomputable def auditorium_height : ℕ := 10

noncomputable def first_rug_width : ℕ := 6
noncomputable def first_rug_height : ℕ := 8
noncomputable def second_rug_width : ℕ := 6
noncomputable def second_rug_height : ℕ := 6
noncomputable def third_rug_width : ℕ := 5
noncomputable def third_rug_height : ℕ := 7

-- Prove that the area of part of the auditorium covered with rugs in three layers is 6 square meters.
theorem area_covered_three_layers : 
  let horizontal_overlap_second_third := 5
  let vertical_overlap_second_third := 3
  let area_overlap_second_third := horizontal_overlap_second_third * vertical_overlap_second_third
  let horizontal_overlap_all := 3
  let vertical_overlap_all := 2
  let area_overlap_all := horizontal_overlap_all * vertical_overlap_all
  area_overlap_all = 6 := 
by
  sorry

end area_covered_three_layers_l31_31397


namespace weight_of_second_piece_l31_31041

-- Define the uniform density of the metal.
def density : ℝ := 0.5  -- ounces per square inch

-- Define the side lengths of the two pieces of metal.
def side_length1 : ℝ := 4  -- inches
def side_length2 : ℝ := 7  -- inches

-- Define the weights of the first piece of metal.
def weight1 : ℝ := 8  -- ounces

-- Define the areas of the pieces of metal.
def area1 : ℝ := side_length1^2  -- square inches
def area2 : ℝ := side_length2^2  -- square inches

-- The theorem to prove: the weight of the second piece of metal.
theorem weight_of_second_piece : (area2 * density) = 24.5 :=
by
  sorry

end weight_of_second_piece_l31_31041


namespace largest_divisor_of_5_consecutive_integers_l31_31470

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31470


namespace student_C_has_sweetest_water_l31_31406

-- Define concentrations for each student
def concentration_A : ℚ := 35 / 175 * 100
def concentration_B : ℚ := 45 / 175 * 100
def concentration_C : ℚ := 65 / 225 * 100

-- Prove that Student C has the highest concentration
theorem student_C_has_sweetest_water :
  concentration_C > concentration_B ∧ concentration_C > concentration_A :=
by
  -- By direct calculation from the provided conditions
  sorry

end student_C_has_sweetest_water_l31_31406


namespace ratio_of_w_to_y_l31_31229

theorem ratio_of_w_to_y (w x y z : ℚ)
  (h1 : w / x = 5 / 4)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 4) :
  w / y = 10 / 3 :=
sorry

end ratio_of_w_to_y_l31_31229


namespace largest_divisor_of_5_consecutive_integers_l31_31475

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31475


namespace units_digit_S7890_l31_31938

noncomputable def c : ℝ := 4 + 3 * Real.sqrt 2
noncomputable def d : ℝ := 4 - 3 * Real.sqrt 2
noncomputable def S (n : ℕ) : ℝ := (1/2:ℝ) * (c^n + d^n)

theorem units_digit_S7890 : (S 7890) % 10 = 8 :=
sorry

end units_digit_S7890_l31_31938


namespace solve_inequality_l31_31601

theorem solve_inequality (a x : ℝ) : 
  if a > 0 then -a < x ∧ x < 2*a else if a < 0 then 2*a < x ∧ x < -a else False :=
by sorry

end solve_inequality_l31_31601


namespace factorization_l31_31296

theorem factorization (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := 
by 
  sorry

end factorization_l31_31296


namespace domain_of_function_l31_31057

def domain_conditions (x : ℝ) : Prop :=
  (1 - x ≥ 0) ∧ (x + 2 > 0)

theorem domain_of_function :
  {x : ℝ | domain_conditions x} = {x : ℝ | -2 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l31_31057


namespace find_multiple_l31_31234

-- Defining the conditions
variables (A B k : ℕ)

-- Given conditions
def sum_condition : Prop := A + B = 77
def bigger_number_condition : Prop := A = 42

-- Using the conditions and aiming to prove that k = 5
theorem find_multiple
  (h1 : sum_condition A B)
  (h2 : bigger_number_condition A) :
  6 * B = k * A → k = 5 :=
by
  sorry

end find_multiple_l31_31234


namespace fish_filets_total_l31_31519

/- Define the number of fish caught by each family member -/
def ben_fish : ℕ := 4
def judy_fish : ℕ := 1
def billy_fish : ℕ := 3
def jim_fish : ℕ := 2
def susie_fish : ℕ := 5

/- Define the number of fish thrown back -/
def fish_thrown_back : ℕ := 3

/- Define the number of filets per fish -/
def filets_per_fish : ℕ := 2

/- Calculate the number of fish filets -/
theorem fish_filets_total : ℕ :=
  let total_fish_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let fish_kept := total_fish_caught - fish_thrown_back
  fish_kept * filets_per_fish

example : fish_filets_total = 24 :=
by {
  /- This 'sorry' placeholder indicates that a proof should be here -/
  sorry
}

end fish_filets_total_l31_31519


namespace max_value_of_quadratic_l31_31073

theorem max_value_of_quadratic : ∃ x : ℝ, (∀ y : ℝ, (-3 * y^2 + 9 * y - 1) ≤ (-3 * (3/2)^2 + 9 * (3/2) - 1)) ∧ x = 3/2 :=
by
  sorry

end max_value_of_quadratic_l31_31073


namespace find_s_l31_31983

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := x^4 + p * x^3 + q * x^2 + r * x + s

theorem find_s (p q r s : ℝ)
  (h1 : ∀ (x : ℝ), g x p q r s = (x + 1) * (x + 10) * (x + 10) * (x + 10))
  (h2 : p + q + r + s = 2673) :
  s = 1000 := 
  sorry

end find_s_l31_31983


namespace number_of_apps_needed_l31_31714

-- Definitions based on conditions
variable (cost_per_app : ℕ) (total_money : ℕ) (remaining_money : ℕ)

-- Assume the conditions given
axiom cost_app_eq : cost_per_app = 4
axiom total_money_eq : total_money = 66
axiom remaining_money_eq : remaining_money = 6

-- The goal is to determine the number of apps Lidia needs to buy
theorem number_of_apps_needed (n : ℕ) (h : total_money - remaining_money = cost_per_app * n) :
  n = 15 :=
by
  sorry

end number_of_apps_needed_l31_31714


namespace c_alone_finishes_job_in_7_5_days_l31_31760

theorem c_alone_finishes_job_in_7_5_days (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : A + B + C = 1 / 5) :
  1 / C = 7.5 :=
by
  -- The proof is omitted
  sorry

end c_alone_finishes_job_in_7_5_days_l31_31760


namespace barrels_of_pitch_needed_l31_31277

-- Define the basic properties and conditions
def total_length_road := 16
def truckloads_per_mile := 3
def bags_of_gravel_per_truckload := 2
def gravel_to_pitch_ratio := 5
def miles_paved_first_day := 4
def miles_paved_second_day := 2 * miles_paved_first_day - 1
def miles_already_paved := miles_paved_first_day + miles_paved_second_day
def remaining_miles := total_length_road - miles_already_paved
def total_truckloads := truckloads_per_mile * remaining_miles
def total_bags_of_gravel := bags_of_gravel_per_truckload * total_truckloads
def barrels_of_pitch := total_bags_of_gravel / gravel_to_pitch_ratio

-- State the theorem to prove the number of barrels of pitch needed
theorem barrels_of_pitch_needed :
    barrels_of_pitch = 6 :=
by
    sorry

end barrels_of_pitch_needed_l31_31277


namespace number_of_chocolate_bars_by_theresa_l31_31581

-- Define the number of chocolate bars and soda cans that Kayla bought
variables (C S : ℕ)

-- Assume the total number of chocolate bars and soda cans Kayla bought is 15
axiom total_purchased_by_kayla : C + S = 15

-- Define the number of chocolate bars Theresa bought as twice the number Kayla bought
def chocolate_bars_purchased_by_theresa := 2 * C

-- The theorem to prove
theorem number_of_chocolate_bars_by_theresa : chocolate_bars_purchased_by_theresa = 2 * C :=
by
  -- The proof is omitted as instructed
  sorry

end number_of_chocolate_bars_by_theresa_l31_31581


namespace health_risk_factor_prob_l31_31063

noncomputable def find_p_q_sum (p q: ℕ) : ℕ :=
if h1 : p.gcd q = 1 then
  31
else 
  sorry

theorem health_risk_factor_prob (p q : ℕ) (h1 : p.gcd q = 1) 
                                (h2 : (p : ℚ) / q = 5 / 26) :
  find_p_q_sum p q = 31 :=
sorry

end health_risk_factor_prob_l31_31063


namespace product_of_5_consecutive_integers_divisible_by_60_l31_31462

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l31_31462


namespace radii_difference_of_concentric_circles_l31_31230

theorem radii_difference_of_concentric_circles 
  (r : ℝ) 
  (h_area_ratio : (π * (2 * r)^2) / (π * r^2) = 4) : 
  (2 * r) - r = r :=
by
  sorry

end radii_difference_of_concentric_circles_l31_31230


namespace find_a₈_l31_31949

noncomputable def a₃ : ℝ := -11 / 6
noncomputable def a₅ : ℝ := -13 / 7

theorem find_a₈ (h : ∃ d : ℝ, ∀ n : ℕ, (1 / (a₃ + 2)) + (n-2) * d = (1 / (a_n + 2)))
  : a_n = -32 / 17 := sorry

end find_a₈_l31_31949


namespace number_of_distinct_b_values_l31_31808

theorem number_of_distinct_b_values : 
  ∃ (b : ℝ) (p q : ℤ), (∀ (x : ℝ), x*x + b*x + 12*b = 0) ∧ 
                        p + q = -b ∧ 
                        p * q = 12 * b ∧ 
                        ∃ n : ℤ, 1 ≤ n ∧ n ≤ 15 :=
sorry

end number_of_distinct_b_values_l31_31808


namespace simplify_polynomial_l31_31373

def p (x : ℝ) : ℝ := 3 * x^5 - x^4 + 2 * x^3 + 5 * x^2 - 3 * x + 7
def q (x : ℝ) : ℝ := -x^5 + 4 * x^4 + x^3 - 6 * x^2 + 5 * x - 4
def r (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + 4 * x^3 - x^2 - x + 2

theorem simplify_polynomial (x : ℝ) :
  (p x) + (q x) - (r x) = 6 * x^4 - x^3 + 3 * x + 1 :=
by sorry

end simplify_polynomial_l31_31373


namespace eight_digit_not_perfect_square_l31_31727

theorem eight_digit_not_perfect_square : ∀ x : ℕ, 0 ≤ x ∧ x ≤ 9999 → ¬ ∃ y : ℤ, (99990000 + x) = y * y := 
by
  intros x hx
  intro h
  obtain ⟨y, hy⟩ := h
  sorry

end eight_digit_not_perfect_square_l31_31727


namespace hypotenuse_square_l31_31353

-- Define the right triangle property and the consecutive integer property
variables (a b c : ℤ)

-- Noncomputable definition will be used as we are proving a property related to integers
noncomputable def consecutive_integers (a b : ℤ) : Prop := b = a + 1

-- Define the statement to prove
theorem hypotenuse_square (h_consec : consecutive_integers a b) (h_right_triangle : a * a + b * b = c * c) : 
  c * c = 2 * a * a + 2 * a + 1 :=
by {
  -- We only need to state the theorem
  sorry
}

end hypotenuse_square_l31_31353


namespace zachary_additional_money_needed_l31_31672

noncomputable def total_cost : ℝ := 3.756 + 2 * 2.498 + 11.856 + 4 * 1.329 + 7.834
noncomputable def zachary_money : ℝ := 24.042
noncomputable def money_needed : ℝ := total_cost - zachary_money

theorem zachary_additional_money_needed : money_needed = 9.716 := 
by 
  sorry

end zachary_additional_money_needed_l31_31672


namespace arithmetic_sequence_8th_term_l31_31165

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l31_31165


namespace find_x_l31_31263

theorem find_x (x : ℝ) (h : 2 * x = 26 - x + 19) : x = 15 :=
by
  sorry

end find_x_l31_31263


namespace arithmetic_sequence_8th_term_l31_31160

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l31_31160


namespace polynomial_inequality_l31_31034

theorem polynomial_inequality (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, (r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3) ∧ 
    (∀ t : ℝ, (t - r1) * (t - r2) * (t - r3) = t^3 + a*t^2 + b*t + c))
  (h2 : ¬ ∃ x : ℝ, (x^2 + x + 2013)^3 + a*(x^2 + x + 2013)^2 + b*(x^2 + x + 2013) + c = 0) :
  t^3 + a*2013^2 + b*2013 + c > 1 / 64 :=
sorry

end polynomial_inequality_l31_31034


namespace total_puzzle_pieces_l31_31577

theorem total_puzzle_pieces : 
  ∀ (p1 p2 p3 : ℕ), 
  p1 = 1000 → 
  p2 = p1 + p1 / 2 → 
  p3 = p1 + p1 / 2 → 
  p1 + p2 + p3 = 4000 := 
by 
  intros p1 p2 p3 
  intro h1 
  intro h2 
  intro h3 
  rw [h1, h2, h3] 
  norm_num
  sorry

end total_puzzle_pieces_l31_31577


namespace dogs_running_l31_31613

theorem dogs_running (total_dogs playing_with_toys barking not_doing_anything running : ℕ)
  (h1 : total_dogs = 88)
  (h2 : playing_with_toys = total_dogs / 2)
  (h3 : barking = total_dogs / 4)
  (h4 : not_doing_anything = 10)
  (h5 : running = total_dogs - playing_with_toys - barking - not_doing_anything) :
  running = 12 :=
sorry

end dogs_running_l31_31613


namespace number_of_slices_per_pizza_l31_31710

-- Given conditions as definitions in Lean 4
def total_pizzas := 2
def total_slices_per_pizza (S : ℕ) : ℕ := total_pizzas * S
def james_portion : ℚ := 2 / 3
def james_ate_slices (S : ℕ) : ℚ := james_portion * (total_slices_per_pizza S)
def james_ate_exactly := 8

-- The main theorem to prove
theorem number_of_slices_per_pizza (S : ℕ) (h : james_ate_slices S = james_ate_exactly) : S = 6 :=
sorry

end number_of_slices_per_pizza_l31_31710


namespace probability_xi_greater_than_2_l31_31695

noncomputable def normalDist := MeasureTheory.ProbabilityDistribution.normal 0 σ^2

variables (ξ : ℝ) (σ : ℝ)
  [fact (σ > 0)]

theorem probability_xi_greater_than_2 :
  MeasureTheory.Probability.mass (λ x, ξ > 2) normalDist = 0.1 :=
by
  have h1 : MeasureTheory.Probability.mass (λ x, -2 < ξ ≤ 0) normalDist = 0.4 := sorry
  have h2 : MeasureTheory.Probability.mass (λ x, 0 < ξ ≤ 2) normalDist = 0.4 := sorry
  have h3 : MeasureTheory.Probability.mass (λ x, -2 ≤ ξ ≤ 2) normalDist = 0.8 := sorry
  have h4 : MeasureTheory.Probability.mass (λ x, |ξ| > 2) normalDist = 0.2 := sorry
  have h5 : MeasureTheory.Probability.mass (λ x, ξ > 2) normalDist = 0.1 := sorry
  exact h5

end probability_xi_greater_than_2_l31_31695


namespace binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l31_31127

-- Definition of power of two
def is_power_of_two (n : ℕ) := ∃ m : ℕ, n = 2^m

-- Theorems to be proven
theorem binom_even_if_power_of_two (n : ℕ) (h : is_power_of_two n) :
  ∀ k : ℕ, 1 ≤ k ∧ k < n → Nat.choose n k % 2 = 0 := sorry

theorem binom_odd_if_not_power_of_two (n : ℕ) (h : ¬ is_power_of_two n) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ Nat.choose n k % 2 = 1 := sorry

end binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l31_31127


namespace polar_r_eq_3_is_circle_l31_31667

theorem polar_r_eq_3_is_circle :
  ∀ θ : ℝ, ∃ x y : ℝ, (x, y) = (3 * Real.cos θ, 3 * Real.sin θ) ∧ x^2 + y^2 = 9 :=
by
  sorry

end polar_r_eq_3_is_circle_l31_31667


namespace setB_forms_right_triangle_l31_31784

-- Define the sets of side lengths
def setA : (ℕ × ℕ × ℕ) := (2, 3, 4)
def setB : (ℕ × ℕ × ℕ) := (3, 4, 5)
def setC : (ℕ × ℕ × ℕ) := (5, 6, 7)
def setD : (ℕ × ℕ × ℕ) := (7, 8, 9)

-- Define the Pythagorean theorem condition
def isRightTriangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- The specific proof goal
theorem setB_forms_right_triangle : isRightTriangle 3 4 5 := by
  sorry

end setB_forms_right_triangle_l31_31784


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31438

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l31_31438


namespace intersect_complement_l31_31110

open Finset

-- Define the universal set U, set A, and set B
def U := {1, 2, 3, 4, 5, 6} : Finset ℕ
def A := {1, 3, 6} : Finset ℕ
def B := {2, 3, 4} : Finset ℕ

-- Define the complement of B in U
def complement_U_B := U \ B

-- The statement to prove
theorem intersect_complement : A ∩ complement_U_B = {1, 6} :=
by sorry

end intersect_complement_l31_31110


namespace scientific_notation_of_61345_05_billion_l31_31851

theorem scientific_notation_of_61345_05_billion :
  ∃ x : ℝ, (61345.05 * 10^9) = x ∧ x = 6.134505 * 10^12 :=
by
  sorry

end scientific_notation_of_61345_05_billion_l31_31851


namespace arithmetic_sequence_eighth_term_l31_31175

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l31_31175


namespace james_profit_l31_31357

/--
  Prove that James's profit from buying 200 lotto tickets at $2 each, given the 
  conditions about winning tickets, is $4,830.
-/
theorem james_profit 
  (total_tickets : ℕ := 200)
  (cost_per_ticket : ℕ := 2)
  (winner_percentage : ℝ := 0.2)
  (five_dollar_win_pct : ℝ := 0.8)
  (grand_prize : ℝ := 5000)
  (average_other_wins : ℝ := 10) :
  let total_cost := total_tickets * cost_per_ticket 
  let total_winners := winner_percentage * total_tickets
  let five_dollar_winners := five_dollar_win_pct * total_winners
  let total_five_dollar := five_dollar_winners * 5
  let remaining_winners := total_winners - 1 - five_dollar_winners
  let total_remaining_winners := remaining_winners * average_other_wins
  let total_winnings := total_five_dollar + grand_prize + total_remaining_winners
  let profit := total_winnings - total_cost
  profit = 4830 :=
by
  sorry

end james_profit_l31_31357


namespace find_m_n_l31_31564

theorem find_m_n (a b : ℝ) (m n : ℤ) :
  (a^m * b * b^n)^3 = a^6 * b^15 → m = 2 ∧ n = 4 :=
by
  sorry

end find_m_n_l31_31564


namespace prove_avg_mark_of_batch3_l31_31755

noncomputable def avg_mark_of_batch3 (A1 A2 A3 : ℕ) (Marks1 Marks2 Marks3 : ℚ) : Prop :=
  A1 = 40 ∧ A2 = 50 ∧ A3 = 60 ∧ Marks1 = 45 ∧ Marks2 = 55 ∧ 
  (A1 * Marks1 + A2 * Marks2 + A3 * Marks3) / (A1 + A2 + A3) = 56.333333333333336 → 
  Marks3 = 65

theorem prove_avg_mark_of_batch3 : avg_mark_of_batch3 40 50 60 45 55 65 :=
by
  unfold avg_mark_of_batch3
  sorry

end prove_avg_mark_of_batch3_l31_31755


namespace five_consecutive_product_div_24_l31_31441

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l31_31441


namespace large_circle_radius_l31_31322

noncomputable def radius_of_large_circle : ℝ :=
  let r_small := 1
  let side_length := 2 * r_small
  let diagonal_length := Real.sqrt (side_length ^ 2 + side_length ^ 2)
  let radius_large := (diagonal_length / 2) + r_small
  radius_large + r_small

theorem large_circle_radius :
  radius_of_large_circle = Real.sqrt 2 + 2 :=
by
  sorry

end large_circle_radius_l31_31322


namespace complement_of_M_with_respect_to_U_l31_31403

namespace Complements

open Set

def U : Set Int := {1, -2, 3, -4, 5, -6}
def M : Set Int := {1, -2, 3, -4}

theorem complement_of_M_with_respect_to_U :
  U \ M = {5, -6} :=
by
  sorry

end Complements

end complement_of_M_with_respect_to_U_l31_31403


namespace factorization_l31_31295

theorem factorization (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := 
by 
  sorry

end factorization_l31_31295


namespace area_of_rectangular_field_l31_31723

theorem area_of_rectangular_field (W D : ℝ) (hW : W = 15) (hD : D = 17) :
  ∃ L : ℝ, (W * L = 120) ∧ D^2 = L^2 + W^2 :=
by 
  use 8
  sorry

end area_of_rectangular_field_l31_31723


namespace product_of_five_consecutive_integers_divisible_by_120_l31_31423

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l31_31423


namespace bisecting_lines_upper_bound_l31_31335

variable (n : ℕ) (P : Type) [ConvexPolygon P] (O : Point)

theorem bisecting_lines_upper_bound (h : ConvexPolygon.sides P = n) :
  ∀ k > n, ¬(∃ L : Finset (Line P),
    (∀ l ∈ L, BisectsArea l P O) ∧ 
    L.card = k) := by sorry

end bisecting_lines_upper_bound_l31_31335


namespace mirasol_account_balance_l31_31992

theorem mirasol_account_balance :
  ∀ (initial_amount spent_coffee spent_tumbler : ℕ), 
  initial_amount = 50 → 
  spent_coffee = 10 → 
  spent_tumbler = 30 → 
  initial_amount - (spent_coffee + spent_tumbler) = 10 :=
by
  intros initial_amount spent_coffee spent_tumbler
  intro h_initial_amount
  intro h_spent_coffee
  intro h_spent_tumbler
  rw [h_initial_amount, h_spent_coffee, h_spent_tumbler]
  simp
  done

end mirasol_account_balance_l31_31992


namespace solution_set_of_inequality_l31_31233

theorem solution_set_of_inequality :
  {x : ℝ | |x + 1| - |x - 5| < 4} = {x : ℝ | x < 4} :=
sorry

end solution_set_of_inequality_l31_31233


namespace remainder_45_to_15_l31_31773

theorem remainder_45_to_15 : ∀ (N : ℤ) (k : ℤ), N = 45 * k + 31 → N % 15 = 1 :=
by
  intros N k h
  sorry

end remainder_45_to_15_l31_31773


namespace find_y_l31_31545

theorem find_y : ∃ y : ℕ, y > 0 ∧ (y + 3050) % 15 = 1234 % 15 ∧ y = 14 := 
by
  sorry

end find_y_l31_31545


namespace dice_probability_l31_31253

noncomputable def probability_same_face (throws : ℕ) (dice : ℕ) : ℚ :=
  1 - (1 - (1 / 6) ^ dice) ^ throws

theorem dice_probability : 
  probability_same_face 5 10 = 1 - (1 - (1 / 6) ^ 10) ^ 5 :=
by 
  sorry

end dice_probability_l31_31253


namespace parallel_lines_k_value_l31_31883

theorem parallel_lines_k_value (k : ℝ) 
  (line1 : ∀ x : ℝ, y = 5 * x + 3) 
  (line2 : ∀ x : ℝ, y = (3 * k) * x + 1) 
  (parallel : ∀ x : ℝ, (5 = 3 * k)) : 
  k = 5 / 3 := 
begin
  sorry
end

end parallel_lines_k_value_l31_31883


namespace single_bill_value_l31_31046

theorem single_bill_value 
  (total_amount : ℕ) 
  (num_5_dollar_bills : ℕ) 
  (amount_5_dollar_bills : ℕ) 
  (single_bill : ℕ) : 
  total_amount = 45 → 
  num_5_dollar_bills = 7 → 
  amount_5_dollar_bills = 5 → 
  total_amount = num_5_dollar_bills * amount_5_dollar_bills + single_bill → 
  single_bill = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end single_bill_value_l31_31046


namespace five_consecutive_product_div_24_l31_31439

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l31_31439


namespace smallest_c_l31_31362

theorem smallest_c {a b c : ℤ} (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c)
  (h4 : a^2 = c * b) : c = 4 :=
by
  -- We state the theorem here without proof. 
  -- The actual proof steps are omitted and replaced by sorry.
  sorry

end smallest_c_l31_31362


namespace five_consecutive_product_div_24_l31_31444

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l31_31444


namespace triangle_area_l31_31967

open Real

-- Define the angles A and C, side a, and state the goal as proving the area
theorem triangle_area (A C : ℝ) (a : ℝ) (hA : A = 30 * (π / 180)) (hC : C = 45 * (π / 180)) (ha : a = 2) : 
  (1 / 2) * ((sqrt 6 + sqrt 2) * (2 * sqrt 2) * sin (30 * (π / 180))) = sqrt 3 + 1 := 
by
  sorry

end triangle_area_l31_31967


namespace no_such_convex_polyhedron_exists_l31_31369

-- Definitions of convex polyhedron and the properties related to its faces and vertices.
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  -- Additional properties and constraints can be added if necessary

-- Definition that captures the condition where each face has more than 5 sides.
def each_face_has_more_than_five_sides (P : ConvexPolyhedron) : Prop :=
  ∀ f, f > 5 -- Simplified assumption

-- Definition that captures the condition where more than five edges meet at each vertex.
def more_than_five_edges_meet_each_vertex (P : ConvexPolyhedron) : Prop :=
  ∀ v, v > 5 -- Simplified assumption

-- The statement to be proven
theorem no_such_convex_polyhedron_exists :
  ¬ ∃ (P : ConvexPolyhedron), (each_face_has_more_than_five_sides P) ∨ (more_than_five_edges_meet_each_vertex P) := by
  -- Proof of this theorem is omitted with "sorry"
  sorry

end no_such_convex_polyhedron_exists_l31_31369


namespace rectangle_area_increase_l31_31213

variable {L W : ℝ} -- Define variables for length and width

theorem rectangle_area_increase (p : ℝ) (hW : W' = 0.4 * W) (hA : A' = 1.36 * (L * W)) :
  L' = L + (240 / 100) * L :=
by
  sorry

end rectangle_area_increase_l31_31213


namespace geometric_sequence_sum_l31_31548

-- Definition of the sum of the first n terms of a geometric sequence
variable (S : ℕ → ℝ)

-- Conditions given in the problem
def S_n_given (n : ℕ) : Prop := S n = 36
def S_2n_given (n : ℕ) : Prop := S (2 * n) = 42

-- Theorem to prove
theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) 
    (h1 : S n = 36) (h2 : S (2 * n) = 42) : S (3 * n) = 48 := sorry

end geometric_sequence_sum_l31_31548


namespace intersection_M_N_l31_31684

-- Define the universe U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set M based on the condition x^2 <= x
def M : Set ℤ := {x ∈ U | x^2 ≤ x}

-- Define the set N based on the condition x^3 - 3x^2 + 2x = 0
def N : Set ℤ := {x ∈ U | x^3 - 3*x^2 + 2*x = 0}

-- State the theorem to be proven
theorem intersection_M_N : M ∩ N = {0, 1} :=
by
  sorry

end intersection_M_N_l31_31684


namespace product_of_five_consecutive_integers_divisible_by_120_l31_31420

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l31_31420


namespace speed_of_faster_train_approx_l31_31880

noncomputable def speed_of_slower_train_kmph : ℝ := 40
noncomputable def speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * 1000 / 3600
noncomputable def distance_train1 : ℝ := 250
noncomputable def distance_train2 : ℝ := 500
noncomputable def total_distance : ℝ := distance_train1 + distance_train2
noncomputable def crossing_time : ℝ := 26.99784017278618
noncomputable def relative_speed_train_crossing : ℝ := total_distance / crossing_time
noncomputable def speed_of_faster_train_mps : ℝ := relative_speed_train_crossing - speed_of_slower_train_mps
noncomputable def speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * 3600 / 1000

theorem speed_of_faster_train_approx : abs (speed_of_faster_train_kmph - 60.0152) < 0.001 :=
by 
  sorry

end speed_of_faster_train_approx_l31_31880


namespace max_divisor_of_five_consecutive_integers_l31_31452

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l31_31452


namespace find_integer_n_l31_31243

theorem find_integer_n : ∃ n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n) % 151 = 93 :=
by
  sorry

end find_integer_n_l31_31243


namespace cars_people_equation_l31_31568

-- Define the first condition
def condition1 (x : ℕ) : ℕ := 4 * (x - 1)

-- Define the second condition
def condition2 (x : ℕ) : ℕ := 2 * x + 8

-- Main theorem which states that the conditions lead to the equation
theorem cars_people_equation (x : ℕ) : condition1 x = condition2 x :=
by
  sorry

end cars_people_equation_l31_31568


namespace trader_sold_bags_l31_31644

-- Define the conditions as constants
def initial_bags : ℕ := 55
def restocked_bags : ℕ := 132
def current_bags : ℕ := 164

-- Define a function to calculate the number of bags sold
def bags_sold (initial restocked current : ℕ) : ℕ :=
  initial + restocked - current

-- Statement of the proof problem
theorem trader_sold_bags : bags_sold initial_bags restocked_bags current_bags = 23 :=
by
  -- Proof is omitted
  sorry

end trader_sold_bags_l31_31644


namespace arithmetic_sequence_8th_term_is_71_l31_31184

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l31_31184


namespace fish_filets_total_l31_31518

/- Define the number of fish caught by each family member -/
def ben_fish : ℕ := 4
def judy_fish : ℕ := 1
def billy_fish : ℕ := 3
def jim_fish : ℕ := 2
def susie_fish : ℕ := 5

/- Define the number of fish thrown back -/
def fish_thrown_back : ℕ := 3

/- Define the number of filets per fish -/
def filets_per_fish : ℕ := 2

/- Calculate the number of fish filets -/
theorem fish_filets_total : ℕ :=
  let total_fish_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let fish_kept := total_fish_caught - fish_thrown_back
  fish_kept * filets_per_fish

example : fish_filets_total = 24 :=
by {
  /- This 'sorry' placeholder indicates that a proof should be here -/
  sorry
}

end fish_filets_total_l31_31518


namespace intersection_complement_eq_l31_31954

def U : Set Int := { -2, -1, 0, 1, 2, 3 }
def M : Set Int := { 0, 1, 2 }
def N : Set Int := { 0, 1, 2, 3 }

noncomputable def C_U (A : Set Int) := U \ A

theorem intersection_complement_eq :
  (C_U M ∩ N) = {3} :=
by
  sorry

end intersection_complement_eq_l31_31954


namespace always_real_roots_range_of_b_analytical_expression_parabola_l31_31886

-- Define the quadratic equation with parameter m
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (5 * m - 1) * x + 4 * m - 4

-- Part 1: Prove the equation always has real roots
theorem always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 := 
sorry

-- Part 2: Find the range of b such that the line intersects the parabola at two distinct points
theorem range_of_b (b : ℝ) : 
  (∀ m : ℝ, m = 1 → (b > -25/4 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = (x1 + b) ∧ quadratic_eq m x2 = (x2 + b)))) :=
sorry

-- Part 3: Find the analytical expressions of the parabolas given the distance condition
theorem analytical_expression_parabola (m : ℝ) : 
  (∀ x1 x2 : ℝ, (|x1 - x2| = 2 → quadratic_eq m x1 = 0 → quadratic_eq m x2 = 0) → 
  (m = -1 ∨ m = -1/5) → 
  ((quadratic_eq (-1) x = -x^2 + 6*x - 8) ∨ (quadratic_eq (-1/5) x = -1/5*x^2 + 2*x - 24/5))) :=
sorry

end always_real_roots_range_of_b_analytical_expression_parabola_l31_31886


namespace real_y_iff_x_interval_l31_31737

theorem real_y_iff_x_interval (x : ℝ) :
  (∃ y : ℝ, 3*y^2 + 2*x*y + x + 5 = 0) ↔ (x ≤ -3 ∨ x ≥ 5) :=
by
  sorry

end real_y_iff_x_interval_l31_31737


namespace range_of_a_l31_31547

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) ↔ (-1 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l31_31547


namespace jimmy_max_loss_l31_31358

-- Definition of the conditions
def exam_points : ℕ := 20
def number_of_exams : ℕ := 3
def points_lost_for_behavior : ℕ := 5
def passing_score : ℕ := 50

-- Total points Jimmy has earned and lost
def total_points : ℕ := (number_of_exams * exam_points) - points_lost_for_behavior

-- The maximum points Jimmy can lose and still pass
def max_points_jimmy_can_lose : ℕ := total_points - passing_score

-- Statement to prove
theorem jimmy_max_loss : max_points_jimmy_can_lose = 5 := 
by
  sorry

end jimmy_max_loss_l31_31358


namespace largest_divisor_of_five_consecutive_integers_l31_31430

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l31_31430


namespace fraction_complex_eq_l31_31964

theorem fraction_complex_eq (z : ℂ) (h : z = 2 + I) : 2 * I / (z - 1) = 1 + I := by
  sorry

end fraction_complex_eq_l31_31964


namespace abs_neg_three_l31_31209

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l31_31209


namespace probability_distribution_correct_l31_31905

noncomputable def X_possible_scores : Set ℤ := {-90, -30, 30, 90}

def prob_correct : ℚ := 0.8
def prob_incorrect : ℚ := 1 - prob_correct

def P_X_neg90 : ℚ := prob_incorrect ^ 3
def P_X_neg30 : ℚ := 3 * prob_correct * prob_incorrect ^ 2
def P_X_30 : ℚ := 3 * prob_correct ^ 2 * prob_incorrect
def P_X_90 : ℚ := prob_correct ^ 3

def P_advance : ℚ := P_X_30 + P_X_90

theorem probability_distribution_correct :
  (P_X_neg90 = (1/125) ∧ P_X_neg30 = (12/125) ∧ P_X_30 = (48/125) ∧ P_X_90 = (64/125)) ∧ 
  P_advance = (112/125) := 
by
  sorry

end probability_distribution_correct_l31_31905


namespace sum_intercepts_of_line_l31_31500

theorem sum_intercepts_of_line (x y : ℝ) (h_eq : y - 6 = -2 * (x - 3)) :
  (∃ x_int : ℝ, (0 - 6 = -2 * (x_int - 3)) ∧ x_int = 6) ∧
  (∃ y_int : ℝ, (y_int - 6 = -2 * (0 - 3)) ∧ y_int = 12) →
  6 + 12 = 18 :=
by sorry

end sum_intercepts_of_line_l31_31500


namespace maximize_f_l31_31328

noncomputable def f (x y z : ℝ) := x * y^2 * z^3

theorem maximize_f :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 1 →
  f x y z ≤ 1 / 432 ∧ (f x y z = 1 / 432 → x = 1/6 ∧ y = 1/3 ∧ z = 1/2) :=
by
  sorry

end maximize_f_l31_31328


namespace num_candidates_l31_31762

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end num_candidates_l31_31762


namespace range_of_k_l31_31961

theorem range_of_k 
  (h : ∀ x : ℝ, x^2 + 2 * k * x - (k - 2) > 0) : -2 < k ∧ k < 1 := 
sorry

end range_of_k_l31_31961


namespace chocolate_milk_tea_cups_l31_31501

-- Defining the conditions as constants or variables
def total_sales : ℕ := 50
def fraction_winter_melon : ℚ := 2/5
def fraction_okinawa : ℚ := 3/10

-- Calculating the number of winter melon and Okinawa flavored cups.
def cups_winter_melon : ℕ := (fraction_winter_melon * total_sales).natAbs
def cups_okinawa : ℕ := (fraction_okinawa * total_sales).natAbs

-- The theorem to prove the number of chocolate-flavored milk tea cups.
theorem chocolate_milk_tea_cups : 
  total_sales - (cups_winter_melon + cups_okinawa) = 15 :=
by
  -- Placeholder for the proof
  sorry

end chocolate_milk_tea_cups_l31_31501


namespace carly_dog_count_l31_31531

theorem carly_dog_count (total_nails : ℕ) (three_legged_dogs : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) 
  (h2 : three_legged_dogs = 3) 
  (h3 : total_dogs * 4 - three_legged_dogs = 41 - 3 * three_legged_dogs) 
  : total_dogs = 11 :=
sorry

end carly_dog_count_l31_31531


namespace stamps_total_l31_31849

theorem stamps_total (x y : ℕ) (hx : x = 34) (hy : y = x + 44) : x + y = 112 :=
by sorry

end stamps_total_l31_31849


namespace total_seeds_l31_31258

-- Definitions and conditions
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds
def Eun_seeds : ℕ := 2 * Gwi_seeds

-- Theorem statement
theorem total_seeds : Bom_seeds + Gwi_seeds + Yeon_seeds + Eun_seeds = 2340 :=
by
  -- Skipping the proof steps with sorry
  sorry

end total_seeds_l31_31258


namespace jimmy_can_lose_5_more_points_l31_31359

theorem jimmy_can_lose_5_more_points (min_points_to_pass : ℕ) (points_per_exam : ℕ) (number_of_exams : ℕ) (points_lost : ℕ) : 
  min_points_to_pass = 50 → 
  points_per_exam = 20 → 
  number_of_exams = 3 → 
  points_lost = 5 → 
  (points_per_exam * number_of_exams - points_lost - 5) = min_points_to_pass :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end jimmy_can_lose_5_more_points_l31_31359


namespace sector_to_cone_height_l31_31494

-- Definitions based on the conditions
def circle_radius : ℝ := 8
def num_sectors : ℝ := 4
def sector_angle : ℝ := 2 * Real.pi / num_sectors
def circumference_of_sector : ℝ := 2 * Real.pi * circle_radius / num_sectors
def radius_of_base : ℝ := circumference_of_sector / (2 * Real.pi)
def slant_height : ℝ := circle_radius

-- Assertion to prove
theorem sector_to_cone_height : 
  let h := Real.sqrt (slant_height^2 - radius_of_base^2) 
  in h = 2 * Real.sqrt 15 :=
by {
  sorry
}

end sector_to_cone_height_l31_31494


namespace product_of_roots_l31_31059

theorem product_of_roots (a b c : ℝ) (h_eq : 24 * a^2 + 36 * a - 648 = 0) : a * c = -27 := 
by
  have h_root_product : (24 * a^2 + 36 * a - 648) = 0 ↔ a = -27 := sorry
  exact sorry

end product_of_roots_l31_31059


namespace find_x2_y2_and_xy_l31_31815

-- Problem statement
theorem find_x2_y2_and_xy (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 9) : 
  x^2 + y^2 = 5 ∧ x * y = -2 :=
by
  sorry -- Proof omitted

end find_x2_y2_and_xy_l31_31815


namespace fries_sold_l31_31897

theorem fries_sold (small_fries large_fries : ℕ) (h1 : small_fries = 4) (h2 : large_fries = 5 * small_fries) :
  small_fries + large_fries = 24 :=
  by
    sorry

end fries_sold_l31_31897


namespace circle_range_of_m_l31_31863

theorem circle_range_of_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x + 2 * m * y + 2 * m^2 + m - 1 = 0 → (2 * m^2 + m - 1 = 0)) → (-2 < m) ∧ (m < 2/3) :=
by
  sorry

end circle_range_of_m_l31_31863


namespace highest_probability_of_red_ball_l31_31354

theorem highest_probability_of_red_ball (red yellow white blue : ℕ) (H1 : red = 5) (H2 : yellow = 4) (H3 : white = 1) (H4 : blue = 3) :
  (red : ℚ) / (red + yellow + white + blue) > (yellow : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (white : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (blue : ℚ) / (red + yellow + white + blue) := 
by {
  sorry
}

end highest_probability_of_red_ball_l31_31354


namespace compound_interest_for_2_years_l31_31591

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

noncomputable def compound_interest (P R T : ℝ) : ℝ := P * (1 + R / 100)^T - P

theorem compound_interest_for_2_years 
  (P : ℝ) (R : ℝ) (T : ℝ) (S : ℝ)
  (h1 : S = 600)
  (h2 : R = 5)
  (h3 : T = 2)
  (h4 : simple_interest P R T = S)
  : compound_interest P R T = 615 := 
sorry

end compound_interest_for_2_years_l31_31591


namespace determine_var_phi_l31_31306

open Real

theorem determine_var_phi (φ : ℝ) (h₀ : 0 ≤ φ ∧ φ ≤ 2 * π) :
  (∀ x, sin (x + φ) = sin (x - π / 6)) → φ = 11 * π / 6 :=
by
  sorry

end determine_var_phi_l31_31306


namespace balance_scale_measurements_l31_31777

theorem balance_scale_measurements {a b c : ℕ}
    (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
    ∀ w : ℕ, 1 ≤ w ∧ w ≤ 11 → ∃ (x y z : ℤ), w = abs (x * a + y * b + z * c) :=
sorry

end balance_scale_measurements_l31_31777


namespace nectar_water_percentage_l31_31482

-- Definitions as per conditions
def nectar_weight : ℝ := 1.2
def honey_weight : ℝ := 1
def honey_water_ratio : ℝ := 0.4

-- Final statement to prove
theorem nectar_water_percentage : (honey_weight * honey_water_ratio + (nectar_weight - honey_weight)) / nectar_weight = 0.5 := by
  sorry

end nectar_water_percentage_l31_31482


namespace range_of_m_l31_31087

variable {x m : ℝ}

def quadratic (x m : ℝ) : ℝ := x^2 + (m - 1) * x + (m^2 - 3 * m + 1)

def absolute_quadratic (x m : ℝ) : ℝ := abs (quadratic x m)

theorem range_of_m (h : ∀ x ∈ Set.Icc (-1 : ℝ) 0, absolute_quadratic x m ≥ absolute_quadratic (x - 1) m) :
  m = 1 ∨ m ≥ 3 :=
sorry

end range_of_m_l31_31087


namespace intersection_x_coordinate_l31_31304

-- Definitions based on conditions
def line1 (x : ℝ) : ℝ := 3 * x + 5
def line2 (x : ℝ) : ℝ := 35 - 5 * x

-- Proof statement
theorem intersection_x_coordinate : ∃ x : ℝ, line1 x = line2 x ∧ x = 15 / 4 :=
by
  use 15 / 4
  sorry

end intersection_x_coordinate_l31_31304


namespace part1_part2_part3_l31_31130

open Set

variable (x : ℝ)

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 2 < x ∧ x < 10}

theorem part1 : A ∩ B = {x | 3 ≤ x ∧ x < 7} :=
sorry

theorem part2 : (Aᶜ : Set ℝ) = {x | x < 3 ∨ x ≥ 7} :=
sorry

theorem part3 : (A ∪ B)ᶜ = {x | x ≤ 2 ∨ x ≥ 10} :=
sorry

end part1_part2_part3_l31_31130


namespace cone_height_l31_31496

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end cone_height_l31_31496


namespace sum_of_reciprocals_of_squares_l31_31016

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 3) :
  (1 / (a : ℚ)^2) + (1 / (b : ℚ)^2) = 10 / 9 :=
sorry

end sum_of_reciprocals_of_squares_l31_31016


namespace find_coordinates_of_P_l31_31725

structure Point where
  x : Int
  y : Int

def symmetric_origin (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = -A.y

def symmetric_y_axis (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = A.y

theorem find_coordinates_of_P :
  ∀ M N P : Point, 
  M = Point.mk (-4) 3 →
  symmetric_origin M N →
  symmetric_y_axis N P →
  P = Point.mk 4 3 := 
by 
  intros M N P hM hSymN hSymP
  sorry

end find_coordinates_of_P_l31_31725


namespace max_divisor_of_five_consecutive_integers_l31_31449

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l31_31449


namespace part1_part2_l31_31332

def A (x : ℝ) : Prop := x < -3 ∨ x > 7
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def complement_R_A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 7

theorem part1 (m : ℝ) :
  (∀ x, complement_R_A x ∨ B m x → complement_R_A x) →
  m ≤ 4 :=
by
  sorry

theorem part2 (m : ℝ) (a b : ℝ) :
  (∀ x, complement_R_A x ∧ B m x ↔ (a ≤ x ∧ x ≤ b)) ∧ (b - a ≥ 1) →
  3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end part1_part2_l31_31332


namespace arithmetic_seq_8th_term_l31_31190

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l31_31190


namespace find_y_l31_31119

theorem find_y (y : ℚ) (h : 6 * y + 3 * y + 4 * y + 2 * y + 1 * y + 5 * y = 360) : y = 120 / 7 := 
sorry

end find_y_l31_31119


namespace spears_per_sapling_l31_31716

/-- Given that a log can produce 9 spears and 6 saplings plus a log produce 27 spears,
prove that a single sapling can produce 3 spears (S = 3). -/
theorem spears_per_sapling (L S : ℕ) (hL : L = 9) (h: 6 * S + L = 27) : S = 3 :=
by
  sorry

end spears_per_sapling_l31_31716


namespace complex_multiplication_l31_31078

-- Definition of the imaginary unit i
def i : ℂ := Complex.I

-- The theorem stating the equality
theorem complex_multiplication : (2 + i) * (3 + i) = 5 + 5 * i := 
sorry

end complex_multiplication_l31_31078


namespace arithmetic_seq_8th_term_l31_31189

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l31_31189


namespace find_n_l31_31573

variable {a : ℕ → ℝ} (h1 : a 4 = 7) (h2 : a 3 + a 6 = 16)

theorem find_n (n : ℕ) (h3 : a n = 31) : n = 16 := by
  sorry

end find_n_l31_31573


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31437

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l31_31437


namespace wyatt_headmaster_duration_l31_31622

def duration_of_wyatt_job (start_month end_month total_months : ℕ) : Prop :=
  start_month <= end_month ∧ total_months = end_month - start_month + 1

theorem wyatt_headmaster_duration : duration_of_wyatt_job 3 12 9 :=
by
  sorry

end wyatt_headmaster_duration_l31_31622


namespace construction_company_order_l31_31636

def concrete_weight : ℝ := 0.17
def bricks_weight : ℝ := 0.17
def stone_weight : ℝ := 0.5
def total_weight : ℝ := 0.84

theorem construction_company_order :
  concrete_weight + bricks_weight + stone_weight = total_weight :=
by
  -- The proof would go here but is omitted per instructions.
  sorry

end construction_company_order_l31_31636


namespace remaining_balance_is_correct_l31_31987

def initial_balance : ℕ := 50
def spent_coffee : ℕ := 10
def spent_tumbler : ℕ := 30

theorem remaining_balance_is_correct : initial_balance - (spent_coffee + spent_tumbler) = 10 := by
  sorry

end remaining_balance_is_correct_l31_31987


namespace range_of_m_l31_31108

variable (x m : ℝ)
hypothesis : (x + m) / (x - 2) + (2 * m) / (2 - x) = 3
hypothesis_pos : 0 < x

theorem range_of_m :
  m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_l31_31108


namespace dot_product_correct_l31_31340

-- Define the vectors as given conditions
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (1, -2)

-- State the theorem to prove the dot product
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = -4 := by
  -- Proof steps go here
  sorry

end dot_product_correct_l31_31340


namespace overall_average_score_l31_31993

theorem overall_average_score 
  (M : ℝ) (E : ℝ) (m e : ℝ)
  (hM : M = 82)
  (hE : E = 75)
  (hRatio : m / e = 5 / 3) :
  (M * m + E * e) / (m + e) = 79.375 := 
by
  sorry

end overall_average_score_l31_31993


namespace arithmetic_seq_8th_term_l31_31194

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l31_31194


namespace evaluate_expression_l31_31069

theorem evaluate_expression : 
  (1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))) = 5 / 7 :=
by
  sorry

end evaluate_expression_l31_31069


namespace two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l31_31585

theorem two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one
  (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n) 
  (h : 2 ^ p + 3 ^ p = a ^ n) : n = 1 :=
sorry

end two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l31_31585


namespace N_is_composite_l31_31061

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ (Nat.Prime N) :=
by
  have h_mod : N % 2027 = 0 := 
    sorry
  intro h_prime
  have h_div : 2027 ∣ N := by
    rw [Nat.dvd_iff_mod_eq_zero, h_mod]
  exact Nat.Prime.not_dvd_one h_prime h_div

end N_is_composite_l31_31061


namespace place_balls_in_boxes_l31_31724

-- Definitions based on conditions in the problem
def num_balls : ℕ := 5
def num_boxes : ℕ := 4

-- Statement of the problem
theorem place_balls_in_boxes : ∃ (ways : ℕ), (ways = 240) :=
by {
  have balls := num_balls,
  have boxes := num_boxes,
  sorry
}

end place_balls_in_boxes_l31_31724


namespace monthly_income_ratio_l31_31868

noncomputable def A_annual_income : ℝ := 571200
noncomputable def C_monthly_income : ℝ := 17000
noncomputable def B_monthly_income : ℝ := C_monthly_income * 1.12
noncomputable def A_monthly_income : ℝ := A_annual_income / 12

theorem monthly_income_ratio :
  (A_monthly_income / B_monthly_income) = 2.5 :=
by
  sorry

end monthly_income_ratio_l31_31868


namespace range_of_a_l31_31682

theorem range_of_a (x y a : ℝ): 
  (x + 3 * y = 3 - a) ∧ (2 * x + y = 1 + 3 * a) ∧ (x + y > 3 * a + 4) ↔ (a < -3 / 2) :=
sorry

end range_of_a_l31_31682


namespace product_of_five_consecutive_is_divisible_by_sixty_l31_31412

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l31_31412


namespace fraction_to_decimal_l31_31799

theorem fraction_to_decimal : (45 : ℝ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end fraction_to_decimal_l31_31799


namespace arithmetic_sequence_eighth_term_l31_31174

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l31_31174


namespace b_95_mod_49_l31_31125

def b (n : ℕ) : ℕ := 5^n + 7^n + 3

theorem b_95_mod_49 : b 95 % 49 = 5 := 
by sorry

end b_95_mod_49_l31_31125


namespace roots_expression_value_l31_31085

theorem roots_expression_value {m n : ℝ} (h₁ : m^2 - 3 * m - 2 = 0) (h₂ : n^2 - 3 * n - 2 = 0) : 
  (7 * m^2 - 21 * m - 3) * (3 * n^2 - 9 * n + 5) = 121 := 
by 
  sorry

end roots_expression_value_l31_31085


namespace probability_fav_song_not_fully_played_l31_31028

theorem probability_fav_song_not_fully_played :
  let song_lengths := List.range 12 |>.map (λ n => 40 * (n + 1))
  let fav_song_idx := 7 -- index of the favourite song (8th song)
  60 * 6 = 360 -- total seconds in 6 minutes
  fav_song_length = 300 -- length of the favourite song in seconds (5 minutes)
  num_songs := 12
  in song_lengths.nth fav_song_idx = some fav_song_length →
      (1 - (1 / (12 * real.to_rat (num_songs.factorial)) *
        ((num_songs - 1).factorial + 3 * (num_songs - 2).factorial))) = 43 / 48 :=
by sorry

end probability_fav_song_not_fully_played_l31_31028


namespace percentage_calculation_l31_31247

def part : ℝ := 12.356
def whole : ℝ := 12356
def expected_percentage : ℝ := 0.1

theorem percentage_calculation (p w : ℝ) (h_p : p = part) (h_w : w = whole) : 
  (p / w) * 100 = expected_percentage :=
sorry

end percentage_calculation_l31_31247


namespace renata_lottery_winnings_l31_31132

def initial_money : ℕ := 10
def donation : ℕ := 4
def prize_won : ℕ := 90
def water_cost : ℕ := 1
def lottery_ticket_cost : ℕ := 1
def final_money : ℕ := 94

theorem renata_lottery_winnings :
  ∃ (lottery_winnings : ℕ), 
  initial_money - donation + prize_won 
  - water_cost - lottery_ticket_cost 
  = final_money ∧ 
  lottery_winnings = 2 :=
by
  -- Proof steps will go here
  sorry

end renata_lottery_winnings_l31_31132


namespace product_of_5_consecutive_integers_divisible_by_60_l31_31465

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l31_31465


namespace arithmetic_sequence_8th_term_l31_31167

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l31_31167


namespace expression_odd_if_p_q_odd_l31_31847

variable (p q : ℕ)

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem expression_odd_if_p_q_odd (hp : is_odd p) (hq : is_odd q) : is_odd (5 * p * q) :=
sorry

end expression_odd_if_p_q_odd_l31_31847


namespace yanna_baked_butter_cookies_in_morning_l31_31486

-- Define the conditions
def biscuits_morning : ℕ := 40
def biscuits_afternoon : ℕ := 20
def cookies_afternoon : ℕ := 10
def total_more_biscuits : ℕ := 30

-- Define the statement to be proved
theorem yanna_baked_butter_cookies_in_morning (B : ℕ) : 
  (biscuits_morning + biscuits_afternoon = (B + cookies_afternoon) + total_more_biscuits) → B = 20 :=
by
  sorry

end yanna_baked_butter_cookies_in_morning_l31_31486


namespace multiple_of_3804_l31_31726

theorem multiple_of_3804 (n : ℕ) (hn : 0 < n) : 
  ∃ k : ℕ, (n^3 - n) * (5^(8*n+4) + 3^(4*n+2)) = k * 3804 :=
by
  sorry

end multiple_of_3804_l31_31726


namespace area_of_bounded_region_l31_31218

open Real

noncomputable def bounded_region_area : ℝ :=
  let f := λ x y : ℝ, y^2 + 4 * x * y + 80 * (abs x) = 800
  -- Assuming the graph defined by the equation forms a bounded region.
  let vertices := [(0, 20), (0, -20), (20, -20), (-20, 20)]
  let height := dist (0, 20) (0, -20)
  let base := dist (0, 20) (-20, 20)
  height * base

theorem area_of_bounded_region :
  bounded_region_area = 1600 := by
  sorry

end area_of_bounded_region_l31_31218


namespace opposite_of_2023_is_neg_2023_l31_31745

theorem opposite_of_2023_is_neg_2023 : (2023 + (-2023) = 0) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l31_31745


namespace find_x_l31_31619

theorem find_x (x : ℝ) (h : 65 + 5 * 12 / (x / 3) = 66) : x = 180 :=
by
  sorry

end find_x_l31_31619


namespace solve_equation_l31_31374

theorem solve_equation {n k l m : ℕ} (h_l : l > 1) :
  (1 + n^k)^l = 1 + n^m ↔ (n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3) :=
sorry

end solve_equation_l31_31374


namespace fraction_power_equals_l31_31051

theorem fraction_power_equals :
  (5 / 7) ^ 7 = (78125 : ℚ) / 823543 := 
by
  sorry

end fraction_power_equals_l31_31051


namespace probability_of_winning_correct_l31_31699

noncomputable def probability_of_winning (P_L : ℚ) (P_T : ℚ) : ℚ :=
  1 - (P_L + P_T)

theorem probability_of_winning_correct :
  probability_of_winning (3/7) (2/21) = 10/21 :=
by
  sorry

end probability_of_winning_correct_l31_31699


namespace abs_neg_three_l31_31208

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l31_31208


namespace product_of_reds_is_red_sum_of_reds_is_red_l31_31798

noncomputable def color := ℕ → Prop

variables (white red : color)
variable (r : ℕ)

axiom coloring : ∀ n, white n ∨ red n
axiom exists_white : ∃ n, white n
axiom exists_red : ∃ n, red n
axiom sum_of_white_red_is_white : ∀ m n, white m → red n → white (m + n)
axiom prod_of_white_red_is_red : ∀ m n, white m → red n → red (m * n)

theorem product_of_reds_is_red (m n : ℕ) : red m → red n → red (m * n) :=
sorry

theorem sum_of_reds_is_red (m n : ℕ) : red m → red n → red (m + n) :=
sorry

end product_of_reds_is_red_sum_of_reds_is_red_l31_31798


namespace slope_angle_of_line_l31_31882

theorem slope_angle_of_line (α : ℝ) (hα : 0 ≤ α ∧ α < 180) 
    (slope_eq_tan : Real.tan α = 1) : α = 45 :=
by
  sorry

end slope_angle_of_line_l31_31882


namespace max_brownies_l31_31823

-- Definitions for the conditions given in the problem
def is_interior_pieces (m n : ℕ) : ℕ := (m - 2) * (n - 2)
def is_perimeter_pieces (m n : ℕ) : ℕ := 2 * m + 2 * n - 4

-- The assertion that the number of brownies along the perimeter is twice the number in the interior
def condition (m n : ℕ) : Prop := 2 * is_interior_pieces m n = is_perimeter_pieces m n

-- The statement that the maximum number of brownies under the given condition is 84
theorem max_brownies : ∃ (m n : ℕ), condition m n ∧ m * n = 84 := by
  sorry

end max_brownies_l31_31823


namespace megatek_manufacturing_percentage_proof_l31_31999

def megatek_employee_percentage
  (total_degrees_in_circle : ℕ)
  (manufacturing_degrees : ℕ) : ℚ :=
  (manufacturing_degrees / total_degrees_in_circle : ℚ) * 100

theorem megatek_manufacturing_percentage_proof (h1 : total_degrees_in_circle = 360)
  (h2 : manufacturing_degrees = 54) :
  megatek_employee_percentage total_degrees_in_circle manufacturing_degrees = 15 := 
by
  sorry

end megatek_manufacturing_percentage_proof_l31_31999


namespace least_faces_combined_l31_31239

noncomputable def num_faces_dice_combined : ℕ :=
  let a := 11
  let b := 7
  a + b

/-- Given the conditions on the dice setups for sums of 8, 11, and 15,
the least number of faces on the two dice combined is 18. -/
theorem least_faces_combined (a b : ℕ) (h1 : 6 < a) (h2 : 6 < b)
  (h_sum_8 : ∃ (p : ℕ), p = 7)  -- 7 ways to roll a sum of 8
  (h_sum_11 : ∃ (q : ℕ), q = 14)  -- half probability means 14 ways to roll a sum of 11
  (h_sum_15 : ∃ (r : ℕ), r = 2) : a + b = 18 :=
by
  sorry

end least_faces_combined_l31_31239


namespace midpoint_polygon_area_half_l31_31368

noncomputable def polygon_area (n : ℕ) (vertices : List (ℝ × ℝ)) : ℝ := sorry

theorem midpoint_polygon_area_half
  (n : ℕ)
  (hn : n ≥ 4)
  (P : Fin n → (ℝ × ℝ))  -- Original n-gon as a function from Fin n to points
  (convex : Convex ℝ (ConvexHull ℝ (Set.range P))) : -- Convexity condition
  let midpoints := λ i : Fin n, ((P i.fst + P i.snd) / 2) -- Midpoints of sides
  polygon_area n midpoints ≥ (polygon_area n P) / 2 :=
sorry

end midpoint_polygon_area_half_l31_31368


namespace cannot_make_it_in_time_l31_31709

theorem cannot_make_it_in_time (time_available : ℕ) (distance_to_station : ℕ) (v1 : ℕ) :
  time_available = 2 ∧ distance_to_station = 2 ∧ v1 = 30 → 
  ¬ ∃ v2, (time_available - (distance_to_station / v1)) * v2 ≥ 1 :=
by
  sorry

end cannot_make_it_in_time_l31_31709


namespace trader_sold_bags_l31_31645

-- Define the conditions as constants
def initial_bags : ℕ := 55
def restocked_bags : ℕ := 132
def current_bags : ℕ := 164

-- Define a function to calculate the number of bags sold
def bags_sold (initial restocked current : ℕ) : ℕ :=
  initial + restocked - current

-- Statement of the proof problem
theorem trader_sold_bags : bags_sold initial_bags restocked_bags current_bags = 23 :=
by
  -- Proof is omitted
  sorry

end trader_sold_bags_l31_31645


namespace calculation_result_l31_31524

theorem calculation_result :
  -Real.sqrt 4 + abs (-Real.sqrt 2 - 1) + (Real.pi - 2013) ^ 0 - (1/5) ^ 0 = Real.sqrt 2 - 1 :=
by
  sorry

end calculation_result_l31_31524


namespace cupcakes_left_at_home_correct_l31_31732

-- Definitions of the conditions
def total_cupcakes_baked : ℕ := 53
def boxes_given_away : ℕ := 17
def cupcakes_per_box : ℕ := 3

-- Calculate the total number of cupcakes given away
def total_cupcakes_given_away := boxes_given_away * cupcakes_per_box

-- Calculate the number of cupcakes left at home
def cupcakes_left_at_home := total_cupcakes_baked - total_cupcakes_given_away

-- Prove that the number of cupcakes left at home is 2
theorem cupcakes_left_at_home_correct : cupcakes_left_at_home = 2 := by
  sorry

end cupcakes_left_at_home_correct_l31_31732


namespace maria_final_bottle_count_l31_31986

-- Define the initial conditions
def initial_bottles : ℕ := 14
def bottles_drunk : ℕ := 8
def bottles_bought : ℕ := 45

-- State the theorem to prove
theorem maria_final_bottle_count : initial_bottles - bottles_drunk + bottles_bought = 51 :=
by
  sorry

end maria_final_bottle_count_l31_31986


namespace conditional_probability_age_30_40_female_l31_31831

noncomputable def total_people : ℕ := 350
noncomputable def total_females : ℕ := 180
noncomputable def females_30_40 : ℕ := 50

theorem conditional_probability_age_30_40_female :
  (females_30_40 : ℚ) / total_females = 5 / 18 :=
by
  sorry

end conditional_probability_age_30_40_female_l31_31831


namespace exists_close_ratios_l31_31597

theorem exists_close_ratios (S : Finset ℝ) (h : S.card = 2000) :
  ∃ (a b c d : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧
  abs ((a - b) / (c - d) - 1) < 1 / 100000 :=
sorry

end exists_close_ratios_l31_31597


namespace carly_dog_count_l31_31532

theorem carly_dog_count (total_nails : ℕ) (three_legged_dogs : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) 
  (h2 : three_legged_dogs = 3) 
  (h3 : total_dogs * 4 - three_legged_dogs = 41 - 3 * three_legged_dogs) 
  : total_dogs = 11 :=
sorry

end carly_dog_count_l31_31532


namespace sum_f_neg12_to_13_l31_31820

noncomputable def f (x : ℝ) := 1 / (3^x + Real.sqrt 3)

theorem sum_f_neg12_to_13 : 
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6)
  + f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0
  + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10
  + f 11 + f 12 + f 13) = (13 * Real.sqrt 3 / 3) :=
sorry

end sum_f_neg12_to_13_l31_31820


namespace product_of_five_consecutive_integers_divisible_by_120_l31_31424

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l31_31424


namespace find_a_l31_31561

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (5, 1)
noncomputable def C (a : ℝ) : ℝ × ℝ := (-4, 2 * a)

def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_a (a : ℝ) : collinear (A a) B (C a) ↔ a = 4 :=
by
  sorry

end find_a_l31_31561


namespace geometric_sequence_fifth_term_l31_31217

theorem geometric_sequence_fifth_term (x y : ℝ) (r : ℝ) 
  (h1 : x + y ≠ 0) (h2 : x - y ≠ 0) (h3 : x ≠ 0) (h4 : y ≠ 0)
  (h_ratio_1 : (x - y) / (x + y) = r)
  (h_ratio_2 : (x^2 * y) / (x - y) = r)
  (h_ratio_3 : (x * y^2) / (x^2 * y) = r) :
  (x * y^2 * ((y / x) * r)) = y^3 := 
by 
  sorry

end geometric_sequence_fifth_term_l31_31217


namespace paving_path_DE_time_l31_31618

-- Define the conditions
variable (v : ℝ) -- Speed of Worker 1
variable (x : ℝ) -- Total distance for Worker 1
variable (d2 : ℝ) -- Total distance for Worker 2
variable (AD DE EF FC : ℝ) -- Distances in the path of Worker 2

-- Define the statement
theorem paving_path_DE_time :
  (AD + DE + EF + FC) = d2 ∧
  x = 9 * v ∧
  d2 = 10.8 * v ∧
  d2 = AD + DE + EF + FC ∧
  (∀ t, t = (DE / (1.2 * v)) * 60) ∧
  t = 45 :=
by
  sorry

end paving_path_DE_time_l31_31618


namespace dan_must_exceed_speed_to_arrive_before_cara_l31_31862

noncomputable def minimum_speed_for_dan (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) : ℕ :=
  (distance / (distance / cara_speed - dan_delay)) + 1

theorem dan_must_exceed_speed_to_arrive_before_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 180 →
  cara_speed = 30 →
  dan_delay = 1 →
  minimum_speed_for_dan distance cara_speed dan_delay > 36 :=
by
  sorry

end dan_must_exceed_speed_to_arrive_before_cara_l31_31862


namespace arithmetic_sequence_8th_term_l31_31156

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l31_31156


namespace factorize_x4_minus_4x2_l31_31922

theorem factorize_x4_minus_4x2 (x : ℝ) : 
  x^4 - 4 * x^2 = x^2 * (x - 2) * (x + 2) :=
by
  sorry

end factorize_x4_minus_4x2_l31_31922


namespace complex_addition_l31_31940

def imag_unit_squared (i : ℂ) : Prop := i * i = -1

theorem complex_addition (a b : ℝ) (i : ℂ)
  (h1 : a + b * i = i * i)
  (h2 : imag_unit_squared i) : a + b = -1 := 
sorry

end complex_addition_l31_31940


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31457

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l31_31457


namespace max_log_sum_l31_31343

noncomputable def log (x : ℝ) : ℝ := Real.log x

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 
  ∃ L, (∀ x y, x > 0 → y > 0 → x + y = 4 → log x + log y ≤ L) ∧ L = log 4 :=
by
  sorry

end max_log_sum_l31_31343


namespace arithmetic_seq_8th_term_l31_31142

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l31_31142


namespace large_green_curlers_l31_31309

-- Define the number of total curlers
def total_curlers : ℕ := 16

-- Define the fraction for pink curlers
def pink_fraction : ℕ := 1 / 4

-- Define the number of pink curlers
def pink_curlers : ℕ := pink_fraction * total_curlers

-- Define the number of blue curlers
def blue_curlers : ℕ := 2 * pink_curlers

-- Define the total number of pink and blue curlers
def pink_and_blue_curlers : ℕ := pink_curlers + blue_curlers

-- Define the number of green curlers
def green_curlers : ℕ := total_curlers - pink_and_blue_curlers

-- Theorem stating the number of green curlers is 4
theorem large_green_curlers : green_curlers = 4 := by
  -- Proof would go here
  sorry

end large_green_curlers_l31_31309


namespace books_loaned_out_l31_31503

theorem books_loaned_out (initial_books returned_percent final_books : ℕ) (h1 : initial_books = 75) (h2 : returned_percent = 65) (h3 : final_books = 61) : 
  ∃ x : ℕ, initial_books - final_books = x - (returned_percent * x / 100) ∧ x = 40 :=
by {
  sorry 
}

end books_loaned_out_l31_31503


namespace find_y_of_x_pow_l31_31344

theorem find_y_of_x_pow (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y - 1) = 8) : y = 4 / 3 :=
by
  -- skipping proof
  sorry

end find_y_of_x_pow_l31_31344


namespace man_speed_3_kmph_l31_31022

noncomputable def bullet_train_length : ℝ := 200 -- The length of the bullet train in meters
noncomputable def bullet_train_speed_kmph : ℝ := 69 -- The speed of the bullet train in km/h
noncomputable def time_to_pass_man : ℝ := 10 -- The time taken to pass the man in seconds
noncomputable def conversion_factor_kmph_to_mps : ℝ := 1000 / 3600 -- Conversion factor from km/h to m/s
noncomputable def bullet_train_speed_mps : ℝ := bullet_train_speed_kmph * conversion_factor_kmph_to_mps -- Speed of the bullet train in m/s
noncomputable def relative_speed : ℝ := bullet_train_length / time_to_pass_man -- Relative speed at which train passes the man
noncomputable def speed_of_man_mps : ℝ := relative_speed - bullet_train_speed_mps -- Speed of the man in m/s
noncomputable def conversion_factor_mps_to_kmph : ℝ := 3.6 -- Conversion factor from m/s to km/h
noncomputable def speed_of_man_kmph : ℝ := speed_of_man_mps * conversion_factor_mps_to_kmph -- Speed of the man in km/h

theorem man_speed_3_kmph :
  speed_of_man_kmph = 3 :=
by
  sorry

end man_speed_3_kmph_l31_31022


namespace range_of_2x_minus_y_l31_31323

variable {x y : ℝ}

theorem range_of_2x_minus_y (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3) :
  ∃ (a b : ℝ), (1 < a) ∧ (a < 2 * x - y) ∧ (2 * x - y < b) ∧ (b < 9) :=
by
  sorry

end range_of_2x_minus_y_l31_31323


namespace prob_memes_given_m_l31_31307

open Probability Theory

theorem prob_memes_given_m :
  let P_word : ProbabilitySpace (Word → ℝ) := sorry
  (P_word.event {w | w = "MATHEMATICS"} = 1 / 2) →
  (P_word.event {w | w = "MEMES"} = 1 / 2) →
  let P_M_math : ProbabilitySpace (Letter → ℝ) := sorry
  (P_M_math.event {l | l = "M"} = 2 / 11) →
  let P_M_memes : ProbabilitySpace (Letter → ℝ) := sorry
  (P_M_memes.event {l | l = "M"} = 2 / 5) →
  conditional_probability (P_word) ("MEMES") (P_M_memes)
    (⋃ word ≃ "MEMES" , (word.weight "M")) = 11 / 16 :=
begin
  sorry
end

end prob_memes_given_m_l31_31307


namespace caterpillar_prob_A_l31_31783

-- Define the probabilities involved
def prob_move_to_A_from_1 (x y z : ℚ) : ℚ :=
  (1/3 : ℚ) * 1 + (1/3 : ℚ) * y + (1/3 : ℚ) * z

def prob_move_to_A_from_2 (x y u : ℚ) : ℚ :=
  (1/3 : ℚ) * 0 + (1/3 : ℚ) * x + (1/3 : ℚ) * u

def prob_move_to_A_from_0 (x y : ℚ) : ℚ :=
  (2/3 : ℚ) * x + (1/3 : ℚ) * y

def prob_move_to_A_from_3 (y u : ℚ) : ℚ :=
  (2/3 : ℚ) * y + (1/3 : ℚ) * u

theorem caterpillar_prob_A :
  exists (x y z u : ℚ), 
    x = prob_move_to_A_from_1 x y z ∧
    y = prob_move_to_A_from_2 x y y ∧
    z = prob_move_to_A_from_0 x y ∧
    u = prob_move_to_A_from_3 y y ∧
    u = y ∧
    x = 9/14 :=
sorry

end caterpillar_prob_A_l31_31783


namespace seed_mixture_percentage_l31_31371

theorem seed_mixture_percentage (x y : ℝ) 
  (hx : 0.4 * x + 0.25 * y = 30)
  (hxy : x + y = 100) :
  x / 100 = 0.3333 :=
by 
  sorry

end seed_mixture_percentage_l31_31371


namespace probability_same_color_ball_draw_l31_31350

theorem probability_same_color_ball_draw (red white : ℕ) 
    (h_red : red = 2) (h_white : white = 2) : 
    let total_outcomes := (red + white) * (red + white)
    let same_color_outcomes := 2 * (red * red + white * white)
    same_color_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end probability_same_color_ball_draw_l31_31350


namespace arithmetic_sequence_8th_term_l31_31195

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l31_31195


namespace rectangle_area_l31_31626

theorem rectangle_area (P L W : ℝ) (hP : P = 2 * (L + W)) (hRatio : L / W = 5 / 2) (hP_val : P = 280) : 
  L * W = 4000 :=
by 
  sorry

end rectangle_area_l31_31626


namespace option_a_solution_l31_31366

theorem option_a_solution (x y : ℕ) (h₁: x = 2) (h₂: y = 2) : 2 * x + y = 6 := by
sorry

end option_a_solution_l31_31366


namespace number_of_levels_l31_31970

theorem number_of_levels (total_capacity : ℕ) (additional_cars : ℕ) (already_parked_cars : ℕ) (n : ℕ) :
  total_capacity = 425 →
  additional_cars = 62 →
  already_parked_cars = 23 →
  n = total_capacity / (already_parked_cars + additional_cars) →
  n = 5 :=
by
  intros
  sorry

end number_of_levels_l31_31970


namespace line_intersects_circle_l31_31079

variable (x0 y0 R : ℝ)

theorem line_intersects_circle (h : x0^2 + y0^2 > R^2) :
  ∃ (x y : ℝ), (x^2 + y^2 = R^2) ∧ (x0 * x + y0 * y = R^2) :=
sorry

end line_intersects_circle_l31_31079


namespace parallel_lines_k_value_l31_31884

theorem parallel_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = 5 * x + 3 → y = (3 * k) * x + 1 → true) → k = 5 / 3 :=
by
  intros
  sorry

end parallel_lines_k_value_l31_31884


namespace simplify_expr_to_polynomial_l31_31254

namespace PolynomialProof

-- Define the given polynomial expressions
def expr1 (x : ℕ) := (3 * x^2 + 4 * x + 8) * (x - 2)
def expr2 (x : ℕ) := (x - 2) * (x^2 + 5 * x - 72)
def expr3 (x : ℕ) := (4 * x - 15) * (x - 2) * (x + 6)

-- Define the full polynomial expression
def full_expr (x : ℕ) := expr1 x - expr2 x + expr3 x

-- Our goal is to prove that full_expr == 6 * x^3 - 4 * x^2 - 26 * x + 20
theorem simplify_expr_to_polynomial (x : ℕ) : 
  full_expr x = 6 * x^3 - 4 * x^2 - 26 * x + 20 := by
  sorry

end PolynomialProof

end simplify_expr_to_polynomial_l31_31254


namespace largest_divisor_of_5_consecutive_integers_l31_31469

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31469


namespace circle_tangent_radii_l31_31665

theorem circle_tangent_radii (a b c : ℝ) (A : ℝ) (p : ℝ)
  (r r_a r_b r_c : ℝ)
  (h1 : p = (a + b + c) / 2)
  (h2 : r = A / p)
  (h3 : r_a = A / (p - a))
  (h4 : r_b = A / (p - b))
  (h5 : r_c = A / (p - c))
  : 1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  sorry

end circle_tangent_radii_l31_31665


namespace evaluate_nested_square_root_l31_31313

-- Define the condition
def pos_real_solution (x : ℝ) : Prop := x = Real.sqrt (18 + x)

-- State the theorem
theorem evaluate_nested_square_root :
  ∃ (x : ℝ), pos_real_solution x ∧ x = (1 + Real.sqrt 73) / 2 :=
sorry

end evaluate_nested_square_root_l31_31313


namespace arithmetic_seq_8th_term_l31_31188

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l31_31188


namespace log_five_fraction_l31_31065

theorem log_five_fraction : log 5 (1 / real.sqrt 5) = -1/2 := by
  sorry

end log_five_fraction_l31_31065


namespace arithmetic_seq_8th_term_l31_31192

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l31_31192


namespace arithmetic_sequence_8th_term_l31_31203

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l31_31203


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31455

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l31_31455


namespace megan_initial_markers_l31_31718

theorem megan_initial_markers (gave : ℕ) (total : ℕ) (initial : ℕ) 
  (h1 : gave = 109) 
  (h2 : total = 326) 
  (h3 : initial + gave = total) : 
  initial = 217 := 
by 
  sorry

end megan_initial_markers_l31_31718


namespace find_number_l31_31375

theorem find_number (x : ℕ) (hx : (x / 100) * 100 = 20) : x = 20 :=
sorry

end find_number_l31_31375


namespace pickle_to_tomato_ratio_l31_31287

theorem pickle_to_tomato_ratio 
  (mushrooms : ℕ) 
  (cherry_tomatoes : ℕ) 
  (pickles : ℕ) 
  (bacon_bits : ℕ) 
  (red_bacon_bits : ℕ) 
  (h1 : mushrooms = 3) 
  (h2 : cherry_tomatoes = 2 * mushrooms)
  (h3 : red_bacon_bits = 32)
  (h4 : bacon_bits = 3 * red_bacon_bits)
  (h5 : bacon_bits = 4 * pickles) : 
  pickles/cherry_tomatoes = 4 :=
by
  sorry

end pickle_to_tomato_ratio_l31_31287


namespace sequence_twice_square_l31_31018

theorem sequence_twice_square (n : ℕ) (a : ℕ → ℕ) :
    (∀ i : ℕ, a i = 0) →
    (∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
        ∀ i : ℕ, i % (2 * m) = 0 → 
            a i = if a i = 0 then 1 else 0) →
    (∀ i : ℕ, a i = 1 ↔ ∃ k : ℕ, i = 2 * k^2) :=
by
  sorry

end sequence_twice_square_l31_31018


namespace range_sinx_pow6_cosx_pow4_l31_31285

open Real

-- Define the function f(x) = sin^6(x) + cos^4(x)
noncomputable def f (x : ℝ) : ℝ := (sin x) ^ 6 + (cos x) ^ 4

-- Prove that the range of f(x) is [0, 1]
theorem range_sinx_pow6_cosx_pow4 : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
by
  intro x
  -- The actual proof will be here
  sorry

end range_sinx_pow6_cosx_pow4_l31_31285


namespace product_of_5_consecutive_integers_divisible_by_60_l31_31466

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l31_31466


namespace expression_value_l31_31948

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 3) :
  (a + b) / m - c * d + m = 2 ∨ (a + b) / m - c * d + m = -4 := 
by
  sorry

end expression_value_l31_31948


namespace even_sum_probability_l31_31858

theorem even_sum_probability : 
  let tiles := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} in
  let players := {1, 2, 3} in
  let combinations (S : Set ℕ) (k : ℕ) := {c : Set ℕ | c ⊆ S ∧ c.card = k} in
  let is_even_sum (s : Set ℕ) := (s.sum % 2 = 0) in
  let ways_even_sum (S : Set ℕ) (k : ℕ) := (combinations S k).filter is_even_sum in
  let p := (ways_even_sum tiles 3).card * (ways_even_sum (tiles \ players) 3).card * (ways_even_sum ((tiles \ players) \ players) 3).card in
  let q := (combinations tiles 3).card * (combinations (tiles \ players) 3).card * (combinations ((tiles \ players) \ players) 3).card in
  p = 4000 ∧ q = 16800 →
  p.gcd q = 1 →
  ∃ (p' q' : ℤ), p' + q' = 26 := 
sorry

end even_sum_probability_l31_31858


namespace rectangle_area_constant_l31_31871

theorem rectangle_area_constant (d : ℝ) (length width : ℝ) (h_ratio : length / width = 5 / 2) (h_diag : d = Real.sqrt (length^2 + width^2)) :
  ∃ k : ℝ, (length * width) = k * d^2 ∧ k = 10 / 29 :=
by
  use 10 / 29
  sorry

end rectangle_area_constant_l31_31871


namespace not_possible_values_l31_31225

theorem not_possible_values (t h d : ℕ) (ht : 3 * t - 6 * h = 2001) (hd : t - h = d) (hh : 6 * h > 0) :
  ∃ n, n = 667 ∧ ∀ d : ℕ, d ≤ 667 → ¬ (t = h + d ∧ 3 * (h + d) - 6 * h = 2001) :=
by
  sorry

end not_possible_values_l31_31225


namespace correct_triangle_set_l31_31220

/-- Definition of triangle inequality -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Sets of lengths for checking the triangle inequality -/
def Set1 : ℝ × ℝ × ℝ := (5, 8, 2)
def Set2 : ℝ × ℝ × ℝ := (5, 8, 13)
def Set3 : ℝ × ℝ × ℝ := (5, 8, 5)
def Set4 : ℝ × ℝ × ℝ := (2, 7, 5)

/-- The correct set of lengths that can form a triangle according to the triangle inequality -/
theorem correct_triangle_set : satisfies_triangle_inequality 5 8 5 :=
by
  -- Proof would be here
  sorry

end correct_triangle_set_l31_31220


namespace sphere_surface_area_ratios_l31_31045

theorem sphere_surface_area_ratios
  (s : ℝ)
  (r1 : ℝ)
  (r2 : ℝ)
  (r3 : ℝ)
  (h1 : r1 = s / 4 * Real.sqrt 6)
  (h2 : r2 = s / 4 * Real.sqrt 2)
  (h3 : r3 = s / 12 * Real.sqrt 6) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r3^2) = 9 ∧
  (4 * Real.pi * r2^2) / (4 * Real.pi * r3^2) = 3 ∧
  (4 * Real.pi * r3^2) / (4 * Real.pi * r3^2) = 1 := 
by
  sorry

end sphere_surface_area_ratios_l31_31045


namespace probability_two_dice_same_number_l31_31595

theorem probability_two_dice_same_number : 
  let dice_sides := 8 in
  let total_outcomes := dice_sides ^ 8 in
  let different_outcomes := (fact dice_sides) / (fact (dice_sides - 8)) in
  (1 - (different_outcomes / total_outcomes)) = (1291 / 1296) :=
by
  sorry

end probability_two_dice_same_number_l31_31595


namespace product_of_five_consecutive_is_divisible_by_sixty_l31_31411

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l31_31411


namespace determinant_of_projection_matrix_l31_31713

noncomputable def projection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of ![![a^2, a*b], ![a*b, b^2]] / (a^2 + b^2)

theorem determinant_of_projection_matrix :
  let vector := ![3, 5]
  let a := (3 : ℝ)
  let b := (5 : ℝ)
  let norm_sq := a^2 + b^2
  let Q := projection_matrix a b in
  Matrix.det Q = 0 :=
by
  let vector := ![3, 5]
  let a := (3 : ℝ)
  let b := (5 : ℝ)
  let norm_sq := a^2 + b^2
  let Q := projection_matrix a b
  have Q_is_projection_matrix : Q = Matrix.of ![![9 / 34, 15 / 34], ![15 / 34, 25 / 34]],
  sorry
  show Matrix.det Q = 0,
  sorry

end determinant_of_projection_matrix_l31_31713


namespace find_x_l31_31683

theorem find_x (x : ℝ) (h1 : |x + 7| = 3) (h2 : x^2 + 2*x - 3 = 5) : x = -4 :=
by
  sorry

end find_x_l31_31683


namespace arithmetic_sequence_8th_term_l31_31152

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l31_31152


namespace circle_radius_zero_l31_31810

-- Theorem statement
theorem circle_radius_zero :
  ∀ (x y : ℝ), 4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → 
  ∃ (c : ℝ) (r : ℝ), r = 0 ∧ (x - 1)^2 + (y + 2)^2 = r^2 :=
by sorry

end circle_radius_zero_l31_31810


namespace remainder_when_divided_by_11_l31_31555

theorem remainder_when_divided_by_11 (n : ℕ) 
  (h1 : 10 ≤ n ∧ n < 100) 
  (h2 : n % 9 = 1) 
  (h3 : n % 10 = 3) : 
  n % 11 = 7 := 
sorry

end remainder_when_divided_by_11_l31_31555


namespace find_natural_number_l31_31923

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_natural_number (n : ℕ) : sum_of_digits (2 ^ n) = 5 ↔ n = 5 := by
  sorry

end find_natural_number_l31_31923


namespace surface_area_eighth_block_l31_31042

theorem surface_area_eighth_block {A B C D E F G H : ℕ} 
  (blockA : A = 148) 
  (blockB : B = 46) 
  (blockC : C = 72) 
  (blockD : D = 28) 
  (blockE : E = 88) 
  (blockF : F = 126) 
  (blockG : G = 58) 
  : H = 22 :=
by 
  sorry

end surface_area_eighth_block_l31_31042


namespace cannot_cover_completely_with_dominoes_l31_31793

theorem cannot_cover_completely_with_dominoes :
  ¬ (∃ f : Fin 5 × Fin 3 → Fin 5 × Fin 3, 
      (∀ p q, f p = f q → p = q) ∧ 
      (∀ p, ∃ q, f q = p) ∧ 
      (∀ p, (f p).1 = p.1 + 1 ∨ (f p).2 = p.2 + 1)) := 
sorry

end cannot_cover_completely_with_dominoes_l31_31793


namespace range_of_a_l31_31686

noncomputable def P (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x + 1 > 0

noncomputable def Q (a : ℝ) : Prop :=
(∃ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1)) ∧ ∀ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1) → (a * (a - 3) < 0)

theorem range_of_a (a : ℝ) (h1 : P a ∨ Q a) (h2 : ¬ (P a ∧ Q a)) : a = 0 ∨ (3 ≤ a ∧ a < 4) := 
sorry

end range_of_a_l31_31686


namespace probability_at_least_two_same_l31_31596

theorem probability_at_least_two_same (n : ℕ) (s : ℕ) (h_n : n = 8) (h_s : s = 8) :
  let total_outcomes := s ^ n
      different_outcomes := Nat.factorial s
      prob_all_different := different_outcomes / total_outcomes
      prob_at_least_two_same := 1 - prob_all_different
  in prob_at_least_two_same = 1291 / 1296 :=
by
  -- Define values
  have h_total_outcomes : total_outcomes = 16777216 := by sorry
  have h_different_outcomes : different_outcomes = 40320 := by sorry
  have h_prob_all_different : prob_all_different = 5 / 1296 := by sorry
  -- Calculate probability of at least two dice showing the same number
  have h_prob_at_least_two_same : prob_at_least_two_same = 1 - (5 / 1296) := by
    unfold prob_at_least_two_same prob_all_different
    rw h_different_outcomes
    rw h_total_outcomes
    rw h_prob_all_different
  -- Simplify
  calc
    prob_at_least_two_same = 1 - (5 / 1296) : by rw h_prob_at_least_two_same
    ... = 1291 / 1296 : by sorry

end probability_at_least_two_same_l31_31596


namespace fraction_to_decimal_l31_31800

theorem fraction_to_decimal : (45 : ℝ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end fraction_to_decimal_l31_31800


namespace peaches_division_l31_31744

theorem peaches_division (n k r : ℕ) 
  (h₁ : 100 = n * k + 10)
  (h₂ : 1000 = n * k * 11 + r) :
  r = 10 :=
by sorry

end peaches_division_l31_31744


namespace max_value_x_2y_2z_l31_31941

theorem max_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : x + 2*y + 2*z ≤ 15 :=
sorry

end max_value_x_2y_2z_l31_31941


namespace arithmetic_sequence_8th_term_l31_31150

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l31_31150


namespace product_of_five_consecutive_is_divisible_by_sixty_l31_31416

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l31_31416


namespace determine_a_for_nonnegative_function_l31_31557

def function_positive_on_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → a * x^3 - 3 * x + 1 ≥ 0

theorem determine_a_for_nonnegative_function :
  ∀ (a : ℝ), function_positive_on_interval a ↔ a = 4 :=
by
  sorry

end determine_a_for_nonnegative_function_l31_31557


namespace arithmetic_sequence_eighth_term_l31_31176

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l31_31176


namespace no_natural_number_solution_for_divisibility_by_2020_l31_31796

theorem no_natural_number_solution_for_divisibility_by_2020 :
  ¬ ∃ k : ℕ, (k^3 - 3 * k^2 + 2 * k + 2) % 2020 = 0 :=
sorry

end no_natural_number_solution_for_divisibility_by_2020_l31_31796


namespace total_degree_difference_l31_31293

-- Definitions based on conditions
def timeStart : ℕ := 12 * 60  -- noon in minutes
def timeEnd : ℕ := 14 * 60 + 30  -- 2:30 PM in minutes
def numTimeZones : ℕ := 3  -- Three time zones
def degreesInCircle : ℕ := 360  -- Degrees in a full circle

-- Calculate degrees moved by each hand
def degreesMovedByHourHand : ℚ := (timeEnd - timeStart) / (12 * 60) * degreesInCircle
def degreesMovedByMinuteHand : ℚ := (timeEnd - timeStart) % 60 * (degreesInCircle / 60)
def degreesMovedBySecondHand : ℕ := 0  -- At 2:30 PM, second hand is at initial position

-- Calculate total degree difference for all three hands and time zones
def totalDegrees : ℚ := 
  (degreesMovedByHourHand + degreesMovedByMinuteHand + degreesMovedBySecondHand) * numTimeZones

-- Theorem statement to prove
theorem total_degree_difference :
  totalDegrees = 765 := by
  sorry

end total_degree_difference_l31_31293


namespace sixty_percent_of_number_l31_31852

theorem sixty_percent_of_number (N : ℚ) (h : ((1 / 6) * (2 / 3) * (3 / 4) * (5 / 7) * N = 25)) :
  0.60 * N = 252 := sorry

end sixty_percent_of_number_l31_31852


namespace complete_the_square_example_l31_31850

theorem complete_the_square_example (x : ℝ) : 
  ∃ c d : ℝ, (x^2 - 6 * x + 5 = 0) ∧ ((x + c)^2 = d) ∧ (d = 4) :=
sorry

end complete_the_square_example_l31_31850


namespace find_subtracted_value_l31_31038

-- Define the conditions
def chosen_number := 124
def result := 110

-- Lean statement to prove
theorem find_subtracted_value (x : ℕ) (y : ℕ) (h1 : x = chosen_number) (h2 : 2 * x - y = result) : y = 138 :=
by
  sorry

end find_subtracted_value_l31_31038


namespace road_length_10_trees_10_intervals_l31_31592

theorem road_length_10_trees_10_intervals 
  (n_trees : ℕ) (n_intervals : ℕ) (tree_interval : ℕ) 
  (h_trees : n_trees = 10) (h_intervals : n_intervals = 9) (h_interval_length : tree_interval = 10) : 
  n_intervals * tree_interval = 90 := 
by 
  sorry

end road_length_10_trees_10_intervals_l31_31592


namespace fish_filets_total_l31_31520

def fish_caught_by_ben : ℕ := 4
def fish_caught_by_judy : ℕ := 1
def fish_caught_by_billy : ℕ := 3
def fish_caught_by_jim : ℕ := 2
def fish_caught_by_susie : ℕ := 5
def fish_thrown_back : ℕ := 3
def filets_per_fish : ℕ := 2

theorem fish_filets_total : 
  (fish_caught_by_ben + fish_caught_by_judy + fish_caught_by_billy + fish_caught_by_jim + fish_caught_by_susie - fish_thrown_back) * filets_per_fish = 24 := 
by
  sorry

end fish_filets_total_l31_31520


namespace value_of_k_l31_31347

open Real

theorem value_of_k {k : ℝ} : 
  (∃ x : ℝ, k * x ^ 2 - 2 * k * x + 4 = 0 ∧ (∀ y : ℝ, k * y ^ 2 - 2 * k * y + 4 = 0 → x = y)) → k = 4 := 
by
  intros h
  sorry

end value_of_k_l31_31347


namespace solve_z_l31_31566

open Complex

theorem solve_z (z : ℂ) (h : z^2 = 3 - 4 * I) : z = 1 - 2 * I ∨ z = -1 + 2 * I :=
by
  sorry

end solve_z_l31_31566


namespace gambler_A_shares_event_A_rare_l31_31575

-- Definitions
def a : ℕ := 243
def k : ℕ := 4
def m : ℕ := 2
def n : ℕ := 1
def p : ℚ := 2 / 3

-- Function to compute the probability of Gambler A winning all remaining rounds
noncomputable def prob_A_wins_all : ℚ :=
  let X2 := p^2 in
  let X3 := (2.choose 1) * p^2 * (1 - p) in
  let X4 := (3.choose 1) * p^2 * (1 - p)^2 in
  X2 + X3 + X4

-- Lean statement for Part 1
theorem gambler_A_shares : prob_A_wins_all * a = 216 := sorry

-- Function to compute the probability of Gambler B winning all remaining rounds (event A)
noncomputable def prob_B_wins_all (p : ℚ) : ℚ :=
  let Y3 := (1 - p)^3 in
  let Y4 := (3.choose 1) * p * (1 - p)^3 in
  Y3 + Y4

-- Function to compute the probability of Gambler A winning all rounds
noncomputable def f (p : ℚ) : ℚ :=
  1 - prob_B_wins_all p

-- Lean statement for Part 2
theorem event_A_rare (p : ℚ) (hp : p ≥ 4 / 5) : f(p) > 0.95 :=
  by
    -- proof would go here
    sorry

end gambler_A_shares_event_A_rare_l31_31575


namespace max_divisor_of_five_consecutive_integers_l31_31450

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l31_31450


namespace abs_val_neg_three_l31_31212

-- Definition section: stating the conditions
def abs_val (x : Int) : Int := if x < 0 then -x else x

-- Statement of the proof problem
theorem abs_val_neg_three : abs_val (-3) = 3 := by
  sorry

end abs_val_neg_three_l31_31212


namespace range_of_f_l31_31227

noncomputable def f : ℝ → ℝ := sorry -- Define f appropriately

theorem range_of_f : Set.range f = {y : ℝ | 0 < y} :=
sorry

end range_of_f_l31_31227


namespace product_of_five_consecutive_integers_divisible_by_120_l31_31421

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l31_31421


namespace center_of_circle_l31_31316

theorem center_of_circle : ∀ (x y : ℝ), x^2 + y^2 = 4 * x - 6 * y + 9 → (x, y) = (2, -3) :=
by
sorry

end center_of_circle_l31_31316


namespace height_of_cone_formed_by_rolling_sector_l31_31498

theorem height_of_cone_formed_by_rolling_sector :
  let r_circle := 8 in
  let n_sectors := 4 in
  let l_cone := r_circle in
  let c_circle := 2 * Real.pi * r_circle in
  let c_base := c_circle / n_sectors in
  let r_base := c_base / (2 * Real.pi) in
  sqrt (l_cone^2 - r_base^2) = 2 * sqrt 15 :=
by
  sorry

end height_of_cone_formed_by_rolling_sector_l31_31498


namespace five_consecutive_product_div_24_l31_31442

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l31_31442


namespace prob_not_rain_correct_l31_31052

noncomputable def prob_not_rain_each_day (prob_rain : ℚ) : ℚ :=
  1 - prob_rain

noncomputable def prob_not_rain_four_days (prob_not_rain : ℚ) : ℚ :=
  prob_not_rain ^ 4

theorem prob_not_rain_correct :
  prob_not_rain_four_days (prob_not_rain_each_day (2/3)) = 1 / 81 :=
by 
  sorry

end prob_not_rain_correct_l31_31052


namespace cost_price_equivalence_l31_31273

theorem cost_price_equivalence (list_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) (cost_price : ℝ) :
  list_price = 132 → discount_rate = 0.1 → profit_rate = 0.1 → 
  (list_price * (1 - discount_rate)) = cost_price * (1 + profit_rate) →
  cost_price = 108 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_price_equivalence_l31_31273


namespace negation_of_proposition_l31_31607

open Classical

theorem negation_of_proposition :
  (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 + 2 * x + 5 > 0) := by
  sorry

end negation_of_proposition_l31_31607


namespace elsa_ends_with_145_marbles_l31_31661

theorem elsa_ends_with_145_marbles :
  let initial := 150
  let after_breakfast := initial - 7
  let after_lunch := after_breakfast - 57
  let after_afternoon := after_lunch + 25
  let after_evening := after_afternoon + 85
  let after_exchange := after_evening - 9 + 6
  let final := after_exchange - 48
  final = 145 := by
    sorry

end elsa_ends_with_145_marbles_l31_31661


namespace max_area_of_sector_l31_31945

theorem max_area_of_sector (α R C : Real) (hC : C > 0) (h : C = 2 * R + α * R) : 
  ∃ S_max : Real, S_max = (C^2) / 16 :=
by
  sorry

end max_area_of_sector_l31_31945


namespace measure_of_angle_C_l31_31973

variable (a b c : ℝ) (S : ℝ)

-- Conditions
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom area_equation : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)

-- The problem
theorem measure_of_angle_C (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  ∃ C : ℝ, C = Real.arctan (Real.sqrt 3) ∧ C = Real.pi / 3 :=
by
  sorry

end measure_of_angle_C_l31_31973


namespace inequality_solution_l31_31916

theorem inequality_solution (x : ℝ) : (x ≠ -2) ↔ (0 ≤ x^2 / (x + 2)^2) := by
  sorry

end inequality_solution_l31_31916


namespace range_of_a_l31_31348

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 - 2 * x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic a x > 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l31_31348


namespace more_tvs_sold_l31_31719

variable (T x : ℕ)

theorem more_tvs_sold (h1 : T + x = 327) (h2 : T + 3 * x = 477) : x = 75 := by
  sorry

end more_tvs_sold_l31_31719


namespace time_after_1450_minutes_l31_31402

theorem time_after_1450_minutes (initial_time_in_minutes : ℕ := 360) (minutes_to_add : ℕ := 1450) : 
  (initial_time_in_minutes + minutes_to_add) % (24 * 60) = 370 :=
by
  -- Given (initial_time_in_minutes = 360 which is 6:00 a.m., minutes_to_add = 1450)
  -- Compute the time in minutes after 1450 minutes
  -- 24 hours = 1440 minutes, so (360 + 1450) % 1440 should equal 370
  sorry

end time_after_1450_minutes_l31_31402


namespace simplify_and_evaluate_l31_31734

theorem simplify_and_evaluate :
  let x := -2 in
  (2 * x + 1) * (x - 2) - (2 - x) ^ 2 = -4 := by
  sorry

end simplify_and_evaluate_l31_31734


namespace problem_condition_l31_31958

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l31_31958


namespace slope_of_line_with_sine_of_angle_l31_31035

theorem slope_of_line_with_sine_of_angle (α : ℝ) 
  (hα₁ : 0 ≤ α) (hα₂ : α < Real.pi) 
  (h_sin : Real.sin α = Real.sqrt 3 / 2) : 
  ∃ k : ℝ, k = Real.tan α ∧ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end slope_of_line_with_sine_of_angle_l31_31035


namespace chili_problem_l31_31791

def cans_of_chili (x y z : ℕ) : Prop := x + 2 * y + z = 6

def percentage_more_tomatoes_than_beans (x y z : ℕ) : ℕ :=
  100 * (z - 2 * y) / (2 * y)

theorem chili_problem (x y z : ℕ) (h1 : cans_of_chili x y z) (h2 : x = 1) (h3 : y = 1) : 
  percentage_more_tomatoes_than_beans x y z = 50 :=
by
  sorry

end chili_problem_l31_31791


namespace find_price_per_backpack_l31_31094

noncomputable def original_price_of_each_backpack
  (total_backpacks : ℕ)
  (monogram_cost : ℕ)
  (total_cost : ℕ)
  (backpacks_cost_before_discount : ℕ) : ℕ :=
total_cost - (total_backpacks * monogram_cost)

theorem find_price_per_backpack
  (total_backpacks : ℕ := 5)
  (monogram_cost : ℕ := 12)
  (total_cost : ℕ := 140)
  (expected_price_per_backpack : ℕ := 16) :
  original_price_of_each_backpack total_backpacks monogram_cost total_cost / total_backpacks = expected_price_per_backpack :=
by
  sorry

end find_price_per_backpack_l31_31094


namespace common_area_of_rectangle_and_circle_l31_31284

theorem common_area_of_rectangle_and_circle (r : ℝ) (a b : ℝ) (h_center : r = 5) (h_dim : a = 10 ∧ b = 4) :
  let sector_area := (25 * Real.pi) / 2 
  let triangle_area := 4 * Real.sqrt 21 
  let result := sector_area + triangle_area 
  result = (25 * Real.pi) / 2 + 4 * Real.sqrt 21 := 
by
  sorry

end common_area_of_rectangle_and_circle_l31_31284


namespace product_of_5_consecutive_integers_divisible_by_60_l31_31463

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l31_31463


namespace circle_radius_l31_31894

/-- Consider a square ABCD with a side length of 4 cm. A circle touches the extensions 
of sides AB and AD. From point C, two tangents are drawn to this circle, 
and the angle between the tangents is 60 degrees. -/
theorem circle_radius (side_length : ℝ) (angle_between_tangents : ℝ) : 
  side_length = 4 ∧ angle_between_tangents = 60 → 
  ∃ (radius : ℝ), radius = 4 * (Real.sqrt 2 + 1) :=
by
  sorry

end circle_radius_l31_31894


namespace arithmetic_sequence_8th_term_is_71_l31_31185

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l31_31185


namespace joan_bought_72_eggs_l31_31974

def dozen := 12
def joan_eggs (dozens: Nat) := dozens * dozen

theorem joan_bought_72_eggs : joan_eggs 6 = 72 :=
by
  sorry

end joan_bought_72_eggs_l31_31974


namespace prob_at_least_two_same_l31_31594

theorem prob_at_least_two_same (h : 8 > 0) : 
  (1 - (Nat.factorial 8 / (8^8) : ℚ) = 2043 / 2048) :=
by
  sorry

end prob_at_least_two_same_l31_31594


namespace fencing_cost_approx_122_52_l31_31072

noncomputable def circumference (d : ℝ) : ℝ := Real.pi * d

noncomputable def fencing_cost (d rate : ℝ) : ℝ := circumference d * rate

theorem fencing_cost_approx_122_52 :
  let d := 26
  let rate := 1.50
  abs (fencing_cost d rate - 122.52) < 1 :=
by
  let d : ℝ := 26
  let rate : ℝ := 1.50
  let cost := fencing_cost d rate
  sorry

end fencing_cost_approx_122_52_l31_31072


namespace trader_sold_23_bags_l31_31642

theorem trader_sold_23_bags
    (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) (x : ℕ)
    (h_initial : initial_stock = 55)
    (h_restocked : restocked = 132)
    (h_final : final_stock = 164)
    (h_equation : initial_stock - x + restocked = final_stock) :
    x = 23 :=
by
    -- Here will be the proof of the theorem
    sorry

end trader_sold_23_bags_l31_31642


namespace prob_Y_gt_4_l31_31966

noncomputable def p : ℝ := 1 - (0.343)^(1/3)

def X : ℕ → ℤ := λ n, Binomial n p

def Y : ℝ → ℝ := λ y, Normal 2 (δ ^ 2)

axiom P_X_ge_1 : (X ≥ 1) = 0.657
axiom P_0_lt_Y_lt_2 : (0 < Y < 2) = p

theorem prob_Y_gt_4 : (Y > 4) = 0.2 := by
  sorry

end prob_Y_gt_4_l31_31966


namespace work_completion_time_l31_31892

theorem work_completion_time :
  let work_rate_A := 1 / 8
  let work_rate_B := 1 / 6
  let work_rate_C := 1 / 4.8
  (work_rate_A + work_rate_B + work_rate_C) = 1 / 2 :=
by
  sorry

end work_completion_time_l31_31892


namespace simplify_expression_l31_31855

-- Define constants
variables (z : ℝ)

-- Define the problem and its solution
theorem simplify_expression :
  (5 - 2 * z) - (4 + 5 * z) = 1 - 7 * z := 
sorry

end simplify_expression_l31_31855


namespace valid_k_l31_31605

theorem valid_k (k : ℕ) (h_pos : k ≥ 1) (h : 10^k - 1 = 9 * k^2) : k = 1 := by
  sorry

end valid_k_l31_31605


namespace valid_sequences_count_l31_31704

noncomputable def number_of_valid_sequences
(strings : List (List Nat))
(ball_A_shot : Nat)
(ball_B_shot : Nat) : Nat := 144

theorem valid_sequences_count :
  let strings := [[1, 2], [3, 4, 5], [6, 7, 8, 9]];
  let ball_A := 1;  -- Assuming A is the first ball in the first string
  let ball_B := 3;  -- Assuming B is the first ball in the second string
  ball_A = 1 →
  ball_B = 3 →
  ball_A_shot = 5 →
  ball_B_shot = 6 →
  number_of_valid_sequences strings ball_A_shot ball_B_shot = 144 :=
by
  intros strings ball_A ball_B hA hB hAShot hBShot
  sorry

end valid_sequences_count_l31_31704


namespace log_comparison_l31_31817

theorem log_comparison (a b c : ℝ) 
  (h₁ : a = Real.log 6 / Real.log 3)
  (h₂ : b = Real.log 10 / Real.log 5)
  (h₃ : c = Real.log 14 / Real.log 7) :
  a > b ∧ b > c :=
  sorry

end log_comparison_l31_31817


namespace typing_speed_equation_l31_31013

theorem typing_speed_equation (x : ℕ) (h_pos : x > 0) :
  120 / x = 180 / (x + 6) :=
sorry

end typing_speed_equation_l31_31013


namespace largest_divisor_of_5_consecutive_integers_l31_31467

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31467


namespace fish_filets_total_l31_31521

def fish_caught_by_ben : ℕ := 4
def fish_caught_by_judy : ℕ := 1
def fish_caught_by_billy : ℕ := 3
def fish_caught_by_jim : ℕ := 2
def fish_caught_by_susie : ℕ := 5
def fish_thrown_back : ℕ := 3
def filets_per_fish : ℕ := 2

theorem fish_filets_total : 
  (fish_caught_by_ben + fish_caught_by_judy + fish_caught_by_billy + fish_caught_by_jim + fish_caught_by_susie - fish_thrown_back) * filets_per_fish = 24 := 
by
  sorry

end fish_filets_total_l31_31521


namespace birds_in_trees_l31_31392

def number_of_stones := 40
def number_of_trees := number_of_stones + 3 * number_of_stones
def combined_number := number_of_trees + number_of_stones
def number_of_birds := 2 * combined_number

theorem birds_in_trees : number_of_birds = 400 := by
  sorry

end birds_in_trees_l31_31392


namespace solution_set_of_x_squared_gt_x_l31_31232

theorem solution_set_of_x_squared_gt_x :
  { x : ℝ | x^2 > x } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end solution_set_of_x_squared_gt_x_l31_31232


namespace height_of_cone_formed_by_rolling_sector_l31_31497

theorem height_of_cone_formed_by_rolling_sector :
  let r_circle := 8 in
  let n_sectors := 4 in
  let l_cone := r_circle in
  let c_circle := 2 * Real.pi * r_circle in
  let c_base := c_circle / n_sectors in
  let r_base := c_base / (2 * Real.pi) in
  sqrt (l_cone^2 - r_base^2) = 2 * sqrt 15 :=
by
  sorry

end height_of_cone_formed_by_rolling_sector_l31_31497


namespace number_of_elements_in_M_l31_31387

def positive_nats : Set ℕ := {n | n > 0}
def M : Set ℕ := {m | ∃ n ∈ positive_nats, m = 2 * n - 1 ∧ m < 60}

theorem number_of_elements_in_M : ∃ s : Finset ℕ, (∀ x, x ∈ s ↔ x ∈ M) ∧ s.card = 30 := 
by
  sorry

end number_of_elements_in_M_l31_31387


namespace odd_function_periodic_value_l31_31364

noncomputable def f : ℝ → ℝ := sorry  -- Define f

theorem odd_function_periodic_value:
  (∀ x, f (-x) = - f x) →  -- f is odd
  (∀ x, f (x + 3) = f x) → -- f has period 3
  f 1 = 2014 →            -- given f(1) = 2014
  f 2013 + f 2014 + f 2015 = 0 := by
  intros h_odd h_period h_f1
  sorry

end odd_function_periodic_value_l31_31364


namespace fermats_little_theorem_l31_31567

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) (hgcd : gcd a p = 1) : (a^(p-1) - 1) % p = 0 := by
  sorry

end fermats_little_theorem_l31_31567


namespace abs_neg_three_l31_31206

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l31_31206


namespace soccer_uniform_probability_l31_31780

-- Definitions for the conditions of the problem
def colorsSocks : List String := ["red", "blue"]
def colorsShirts : List String := ["red", "blue", "green"]

noncomputable def differentColorConfigurations : Nat :=
  let validConfigs := [("red", "blue"), ("red", "green"), ("blue", "red"), ("blue", "green")]
  validConfigs.length

noncomputable def totalConfigurations : Nat :=
  colorsSocks.length * colorsShirts.length

noncomputable def probabilityDifferentColors : ℚ :=
  (differentColorConfigurations : ℚ) / (totalConfigurations : ℚ)

-- The theorem to prove
theorem soccer_uniform_probability :
  probabilityDifferentColors = 2 / 3 :=
by
  sorry

end soccer_uniform_probability_l31_31780


namespace arithmetic_sequence_geometric_condition_l31_31821

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 1) (h_d_nonzero : d ≠ 0)
  (h_geom : (1 + d) * (1 + d) = 1 * (1 + 4 * d)) : a 2013 = 4025 := by sorry

end arithmetic_sequence_geometric_condition_l31_31821


namespace perimeter_of_nonagon_l31_31097

-- Definitions based on the conditions
def sides := 9
def side_length : ℝ := 2

-- The problem statement in Lean
theorem perimeter_of_nonagon : sides * side_length = 18 := 
by sorry

end perimeter_of_nonagon_l31_31097


namespace teams_played_same_matches_l31_31766

theorem teams_played_same_matches (n : ℕ) (h : n = 30)
  (matches_played : Fin n → ℕ) :
  ∃ (i j : Fin n), i ≠ j ∧ matches_played i = matches_played j :=
by
  sorry

end teams_played_same_matches_l31_31766


namespace sum_of_three_integers_l31_31001

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 5^3) : a + b + c = 31 := by
  sorry

end sum_of_three_integers_l31_31001


namespace N_is_composite_l31_31062

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ (Nat.Prime N) :=
by
  sorry

end N_is_composite_l31_31062


namespace arithmetic_sequence_8th_term_l31_31162

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l31_31162


namespace abs_neg_three_l31_31204

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l31_31204


namespace acute_triangle_angle_measure_acute_triangle_side_range_l31_31551

theorem acute_triangle_angle_measure (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) : B = π / 3 :=
by
  sorry

theorem acute_triangle_side_range (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) (hB : B = π / 3) (hb : b = 3) :
  3 * Real.sqrt 3 < a + c ∧ a + c ≤ 6 :=
by
  sorry

end acute_triangle_angle_measure_acute_triangle_side_range_l31_31551


namespace largest_divisor_of_5_consecutive_integers_l31_31472

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31472


namespace math_problem_l31_31959

theorem math_problem 
  (x y : ℝ) 
  (h : x^2 + y^2 - x * y = 1) 
  : (-2 ≤ x + y) ∧ (x^2 + y^2 ≤ 2) :=
by
  sorry

end math_problem_l31_31959


namespace intersection_complement_l31_31111

variable U A B : Set ℕ
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (A_def : A = {1, 3, 6})
variable (B_def : B = {2, 3, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 6} :=
by
  rw [U_def, A_def, B_def]
  simp
  sorry

end intersection_complement_l31_31111


namespace carly_dogs_total_l31_31525

theorem carly_dogs_total (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) (h2 : three_legged_dogs = 3) (h3 : nails_per_paw = 4) : total_dogs = 11 :=
by
  sorry

end carly_dogs_total_l31_31525


namespace selena_taco_packages_l31_31731

-- Define the problem conditions
def tacos_per_package : ℕ := 4
def shells_per_package : ℕ := 6
def min_tacos : ℕ := 60
def min_shells : ℕ := 60

-- Lean statement to prove the smallest number of taco packages needed
theorem selena_taco_packages :
  ∃ n : ℕ, (n * tacos_per_package ≥ min_tacos) ∧ (∃ m : ℕ, (m * shells_per_package ≥ min_shells) ∧ (n * tacos_per_package = m * shells_per_package) ∧ n = 15) := 
by {
  sorry
}

end selena_taco_packages_l31_31731


namespace factorization_of_x10_minus_1024_l31_31300

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end factorization_of_x10_minus_1024_l31_31300


namespace largest_divisor_of_5_consecutive_integers_l31_31474

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31474


namespace pool_length_calc_l31_31920

variable (total_water : ℕ) (drinking_cooking_water : ℕ) (shower_water : ℕ) (shower_count : ℕ)
variable (pool_width : ℕ) (pool_height : ℕ) (pool_volume : ℕ)

theorem pool_length_calc (h1 : total_water = 1000)
    (h2 : drinking_cooking_water = 100)
    (h3 : shower_water = 20)
    (h4 : shower_count = 15)
    (h5 : pool_width = 10)
    (h6 : pool_height = 6)
    (h7 : pool_volume = total_water - (drinking_cooking_water + shower_water * shower_count)) :
    pool_volume = 600 →
    pool_volume = 60 * length →
    length = 10 :=
by
  sorry

end pool_length_calc_l31_31920


namespace zero_point_exists_between_2_and_3_l31_31872

noncomputable def f (x : ℝ) := 2^(x-1) + x - 5

theorem zero_point_exists_between_2_and_3 :
  ∃ x₀ ∈ Set.Ioo (2 : ℝ) 3, f x₀ = 0 :=
sorry

end zero_point_exists_between_2_and_3_l31_31872


namespace monkey_farm_l31_31969

theorem monkey_farm (x y : ℕ) 
  (h1 : y = 14 * x + 48) 
  (h2 : y = 18 * x - 64) : 
  x = 28 ∧ y = 440 := 
by 
  sorry

end monkey_farm_l31_31969


namespace largest_divisor_of_5_consecutive_integers_l31_31473

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31473


namespace line_equation_passing_through_point_and_equal_intercepts_l31_31334

theorem line_equation_passing_through_point_and_equal_intercepts :
    (∃ k: ℝ, ∀ x y: ℝ, (2, 5) = (x, k * x) ∨ x + y = 7) :=
by
  sorry

end line_equation_passing_through_point_and_equal_intercepts_l31_31334


namespace log_base_5_sqrt_inverse_l31_31068

theorem log_base_5_sqrt_inverse (x : ℝ) (hx : 5 ^ x = 1 / real.sqrt 5) : 
  x = -1 / 2 := 
by
  sorry

end log_base_5_sqrt_inverse_l31_31068


namespace number_of_odd_palindromes_l31_31589

def is_palindrome (n : ℕ) : Prop :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := n / 100
  n < 1000 ∧ n >= 100 ∧ d0 = d2

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem number_of_odd_palindromes : ∃ n : ℕ, is_palindrome n ∧ is_odd n → n = 50 :=
by
  sorry

end number_of_odd_palindromes_l31_31589


namespace rectangular_plot_area_l31_31383

-- Define the conditions
def breadth := 11  -- breadth in meters
def length := 3 * breadth  -- length is thrice the breadth

-- Define the function to calculate area
def area (length breadth : ℕ) := length * breadth

-- The theorem to prove
theorem rectangular_plot_area : area length breadth = 363 := by
  sorry

end rectangular_plot_area_l31_31383


namespace find_abcde_l31_31673

noncomputable def find_five_digit_number (a b c d e : ℕ) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem find_abcde
  (a b c d e : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 0 ≤ e ∧ e ≤ 9)
  (h6 : a ≠ 0)
  (h7 : (10 * a + b + 10 * b + c) * (10 * b + c + 10 * c + d) * (10 * c + d + 10 * d + e) = 157605) :
  find_five_digit_number a b c d e = 12345 ∨ find_five_digit_number a b c d e = 21436 :=
sorry

end find_abcde_l31_31673


namespace find_speed_grocery_to_gym_l31_31785

variables (v : ℝ) (speed_grocery_to_gym : ℝ)
variables (d_home_to_grocery : ℝ) (d_grocery_to_gym : ℝ)
variables (time_diff : ℝ)

def problem_conditions : Prop :=
  d_home_to_grocery = 840 ∧
  d_grocery_to_gym = 480 ∧
  time_diff = 40 ∧
  speed_grocery_to_gym = 2 * v

def correct_answer : Prop :=
  speed_grocery_to_gym = 30

theorem find_speed_grocery_to_gym :
  problem_conditions v speed_grocery_to_gym d_home_to_grocery d_grocery_to_gym time_diff →
  correct_answer speed_grocery_to_gym :=
by
  sorry

end find_speed_grocery_to_gym_l31_31785


namespace indefinite_integral_l31_31925

noncomputable def integral : ℝ → ℝ := λ x,
  ∫ (x : ℝ) in -∞..∞, (4 * x^2 + 3 * x + 4) / ((x^2 + 1) * (x^2 + x + 1)) 

theorem indefinite_integral :
  ∃ C : ℝ, 
  (λ x : ℝ, ∫ (t : ℝ) in 0..x, (4 * t^2 + 3 * t + 4) / ((t^2 + 1) * (t^2 + t + 1)) ) = 
  (λ x : ℝ, 3 * arctan x + (2 / real.sqrt 3) * arctan ((2 * x + 1) / real.sqrt 3) + C) :=
sorry

end indefinite_integral_l31_31925


namespace bus_distance_time_relation_l31_31023

theorem bus_distance_time_relation (t : ℝ) :
    (0 ≤ t ∧ t ≤ 1 → s = 60 * t) ∧
    (1 < t ∧ t ≤ 1.5 → s = 60) ∧
    (1.5 < t ∧ t ≤ 2.5 → s = 80 * (t - 1.5) + 60) :=
sorry

end bus_distance_time_relation_l31_31023


namespace regular_polygon_interior_angle_integer_l31_31365

theorem regular_polygon_interior_angle_integer :
  ∃ l : List ℕ, l.length = 9 ∧ ∀ n ∈ l, 3 ≤ n ∧ n ≤ 15 ∧ (180 * (n - 2)) % n = 0 :=
by
  sorry

end regular_polygon_interior_angle_integer_l31_31365


namespace tangent_line_at_point_l31_31924

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
x + 4 * y - 3 = 0

theorem tangent_line_at_point (x y : ℝ) (h₁ : y = 1 / x^2) (h₂ : x = 2) (h₃ : y = 1/4) :
  tangent_line_equation x y :=
by 
  sorry

end tangent_line_at_point_l31_31924


namespace area_of_right_triangle_l31_31833

variables {x y : ℝ} (r : ℝ)

theorem area_of_right_triangle (hx : ∀ r, r * (x + y + r) = x * y) :
  1 / 2 * (x + r) * (y + r) = x * y :=
by sorry

end area_of_right_triangle_l31_31833


namespace part1_solution_set_part2_a_range_l31_31558

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := abs x + 2 * abs (x + 2 - a)

-- Part 1: When a = 3, solving the inequality
theorem part1_solution_set (x : ℝ) : g x 3 ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 :=
by
  sorry

-- Part 2: Finding the range of a such that f(x) = g(x-2) >= 1 for all x in ℝ
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := g (x - 2) a

theorem part2_a_range : (∀ x : ℝ, f x a ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 :=
by
  sorry

end part1_solution_set_part2_a_range_l31_31558


namespace volume_of_convex_solid_l31_31609

variables {m V t6 T t3 : ℝ} 

-- Definition of the distance between the two parallel planes
def distance_between_planes (m : ℝ) : Prop := m > 0

-- Areas of the two parallel faces
def area_hexagon_face (t6 : ℝ) : Prop := t6 > 0
def area_triangle_face (t3 : ℝ) : Prop := t3 > 0

-- Area of the cross-section of the solid with a plane perpendicular to the height at its midpoint
def area_cross_section (T : ℝ) : Prop := T > 0

-- Volume of the convex solid
def volume_formula_holds (V m t6 T t3 : ℝ) : Prop :=
  V = (m / 6) * (t6 + 4 * T + t3)

-- Formal statement of the problem
theorem volume_of_convex_solid
  (m t6 t3 T V : ℝ)
  (h₁ : distance_between_planes m)
  (h₂ : area_hexagon_face t6)
  (h₃ : area_triangle_face t3)
  (h₄ : area_cross_section T) :
  volume_formula_holds V m t6 T t3 :=
by
  sorry

end volume_of_convex_solid_l31_31609


namespace cylinder_new_volume_l31_31395

-- Definitions based on conditions
def original_volume_r_h (π R H : ℝ) : ℝ := π * R^2 * H

def new_volume (π R H : ℝ) : ℝ := π * (3 * R)^2 * (2 * H)

theorem cylinder_new_volume (π R H : ℝ) (h_original_volume : original_volume_r_h π R H = 15) :
  new_volume π R H = 270 :=
by sorry

end cylinder_new_volume_l31_31395


namespace find_distance_d_l31_31703

theorem find_distance_d (d : ℝ) (XR : ℝ) (YP : ℝ) (XZ : ℝ) (YZ : ℝ) (XY : ℝ) (h1 : XR = 3) (h2 : YP = 12) (h3 : XZ = 3 + d) (h4 : YZ = 12 + d) (h5 : XY = 15) (h6 : (XZ)^2 + (XY)^2 = (YZ)^2) : d = 5 :=
sorry

end find_distance_d_l31_31703


namespace projectile_reaches_24m_at_12_7_seconds_l31_31602

theorem projectile_reaches_24m_at_12_7_seconds :
  ∃ t : ℝ, (y = -4.9 * t^2 + 25 * t) ∧ y = 24 ∧ t = 12 / 7 :=
by
  use 12 / 7
  sorry

end projectile_reaches_24m_at_12_7_seconds_l31_31602


namespace star_operation_l31_31671

def new_op (a b : ℝ) : ℝ :=
  a^2 + b^2 - a * b

theorem star_operation (x y : ℝ) : 
  new_op (x + 2 * y) (y + 3 * x) = 7 * x^2 + 3 * y^2 + 3 * (x * y) :=
by
  sorry

end star_operation_l31_31671


namespace find_sp_l31_31221

theorem find_sp (s p : ℝ) (t x y : ℝ) (h1 : x = 3 + 5 * t) (h2 : y = 3 + p * t) 
  (h3 : y = 4 * x - 9) : 
  s = 3 ∧ p = 20 := 
by
  -- Proof goes here
  sorry

end find_sp_l31_31221


namespace arithmetic_sequence_8th_term_is_71_l31_31177

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l31_31177


namespace total_toys_l31_31873

theorem total_toys (n : ℕ) (h1 : 3 * (n / 4) = 18) : n = 24 :=
by
  sorry

end total_toys_l31_31873


namespace bill_apples_left_l31_31911

-- Definitions based on the conditions
def total_apples : Nat := 50
def apples_per_child : Nat := 3
def number_of_children : Nat := 2
def apples_per_pie : Nat := 10
def number_of_pies : Nat := 2

-- The main statement to prove
theorem bill_apples_left : total_apples - ((apples_per_child * number_of_children) + (apples_per_pie * number_of_pies)) = 24 := by
sorry

end bill_apples_left_l31_31911


namespace big_rectangle_width_l31_31778

theorem big_rectangle_width
  (W : ℝ)
  (h₁ : ∃ l w : ℝ, l = 40 ∧ w = W)
  (h₂ : ∃ l' w' : ℝ, l' = l / 2 ∧ w' = w / 2)
  (h_area : 200 = l' * w') :
  W = 20 :=
by sorry

end big_rectangle_width_l31_31778


namespace pairs_of_values_l31_31926

theorem pairs_of_values (x y : ℂ) :
  (y = (x + 2)^3 ∧ x * y + 2 * y = 2) →
  (∃ (r1 r2 i1 i2 : ℂ), (r1.im = 0 ∧ r2.im = 0) ∧ (i1.im ≠ 0 ∧ i2.im ≠ 0) ∧ 
    ((r1, (r1 + 2)^3) = (x, y) ∨ (r2, (r2 + 2)^3) = (x, y) ∨
     (i1, (i1 + 2)^3) = (x, y) ∨ (i2, (i2 + 2)^3) = (x, y))) :=
sorry

end pairs_of_values_l31_31926


namespace relationship_between_trigonometric_functions_l31_31675

open Real

theorem relationship_between_trigonometric_functions :
  let a := cos 2
  let b := sin 3
  let c := tan 4
  in a < b ∧ b < c := by
  -- Definitions based on conditions
  let a := cos 2
  let b := sin 3
  let c := tan 4
  -- Sorry to skip the proof
  sorry

end relationship_between_trigonometric_functions_l31_31675


namespace arithmetic_sequence_8th_term_is_71_l31_31182

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l31_31182


namespace largest_angle_in_pentagon_l31_31240

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
    (hA : A = 60) 
    (hB : B = 85) 
    (hCD : C = D) 
    (hE : E = 2 * C + 15) 
    (sum_angles : A + B + C + D + E = 540) : 
    E = 205 := 
by 
    sorry

end largest_angle_in_pentagon_l31_31240


namespace print_shop_X_charge_l31_31075

-- Define the given conditions
def cost_per_copy_X (x : ℝ) : Prop := x > 0
def cost_per_copy_Y : ℝ := 2.75
def total_copies : ℕ := 40
def extra_cost_Y : ℝ := 60

-- Define the main problem
theorem print_shop_X_charge (x : ℝ) (h : cost_per_copy_X x) :
  total_copies * cost_per_copy_Y = total_copies * x + extra_cost_Y → x = 1.25 :=
by
  sorry

end print_shop_X_charge_l31_31075


namespace total_travel_time_l31_31120

-- Define the given conditions
def speed_jogging : ℝ := 5
def speed_bus : ℝ := 30
def distance_to_school : ℝ := 6.857142857142858

-- State the theorem to prove
theorem total_travel_time :
  (distance_to_school / speed_jogging) + (distance_to_school / speed_bus) = 1.6 :=
by
  sorry

end total_travel_time_l31_31120


namespace abs_val_neg_three_l31_31211

-- Definition section: stating the conditions
def abs_val (x : Int) : Int := if x < 0 then -x else x

-- Statement of the proof problem
theorem abs_val_neg_three : abs_val (-3) = 3 := by
  sorry

end abs_val_neg_three_l31_31211


namespace b_greater_than_neg3_l31_31950

def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem b_greater_than_neg3 (b : ℝ) :
  (∀ (n : ℕ), 0 < n → a_n (n + 1) b > a_n n b) → b > -3 :=
by
  sorry

end b_greater_than_neg3_l31_31950


namespace emily_eggs_collected_l31_31921

theorem emily_eggs_collected :
  let number_of_baskets := 1525
  let eggs_per_basket := 37.5
  let total_eggs := number_of_baskets * eggs_per_basket
  total_eggs = 57187.5 :=
by
  sorry

end emily_eggs_collected_l31_31921


namespace apples_distribution_l31_31539

variable (x : ℕ)

theorem apples_distribution :
  0 ≤ 5 * x + 12 - 8 * (x - 1) ∧ 5 * x + 12 - 8 * (x - 1) < 8 :=
sorry

end apples_distribution_l31_31539


namespace sum_of_200_terms_l31_31677

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (a1 a200 : ℝ)

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = (n * (a 1 + a n)) / 2

def collinearity_condition (a1 a200 : ℝ) : Prop :=
a1 + a200 = 1

-- Proof statement
theorem sum_of_200_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 a200 : ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_collinear : collinearity_condition a1 a200) : 
  S 200 = 100 := 
sorry

end sum_of_200_terms_l31_31677


namespace arithmetic_sequence_8th_term_is_71_l31_31179

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l31_31179


namespace triangle_right_triangle_l31_31089

theorem triangle_right_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 3) (h2 : b = 4) (h3 : c^2 - 10 * c + 25 = 0) : 
  a^2 + b^2 = c^2 :=
by
  -- We know the values of a, b, and c by the conditions
  sorry

end triangle_right_triangle_l31_31089


namespace factorize_x2_add_2x_sub_3_l31_31655

theorem factorize_x2_add_2x_sub_3 :
  (x^2 + 2 * x - 3) = (x + 3) * (x - 1) :=
by
  sorry

end factorize_x2_add_2x_sub_3_l31_31655


namespace inflated_cost_per_person_l31_31377

def estimated_cost : ℝ := 30e9
def people_sharing : ℝ := 200e6
def inflation_rate : ℝ := 0.05

theorem inflated_cost_per_person :
  (estimated_cost * (1 + inflation_rate)) / people_sharing = 157.5 := by
  sorry

end inflated_cost_per_person_l31_31377


namespace largest_interior_angle_obtuse_isosceles_triangle_l31_31879

theorem largest_interior_angle_obtuse_isosceles_triangle :
  ∀ (P Q R : Type) (α β γ : ℝ), α + β + γ = 180 ∧ γ = 120 ∧ α = 30 ∧ β = 30 →
  (α = 30 ∧ β = 30 ∧ γ = 120) ∨
  (α = 30 ∧ γ = 30 ∧ β = 120) ∨
  (β = 30 ∧ γ = 30 ∧ α = 120) → 
  γ = max α (max β γ) :=
by {
  intros P Q R α β γ h1 h2,
  repeat { rw h1 at * },
  rw h2,
  sorry
}

end largest_interior_angle_obtuse_isosceles_triangle_l31_31879


namespace min_value_trig_expression_l31_31807

theorem min_value_trig_expression : (∃ x : ℝ, 3 * Real.cos x - 4 * Real.sin x = -5) :=
by
  sorry

end min_value_trig_expression_l31_31807


namespace Chad_saves_40_percent_of_his_earnings_l31_31055

theorem Chad_saves_40_percent_of_his_earnings :
  let earnings_mow := 600
  let earnings_birthday := 250
  let earnings_games := 150
  let earnings_oddjobs := 150
  let amount_saved := 460
  (amount_saved / (earnings_mow + earnings_birthday + earnings_games + earnings_oddjobs) * 100) = 40 :=
by
  sorry

end Chad_saves_40_percent_of_his_earnings_l31_31055


namespace f_is_odd_max_min_values_l31_31842

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
variable (f_one : f 1 = -2)
variable (f_neg : ∀ x > 0, f x < 0)

-- Define the statement in Lean for Part 1: proving the function is odd
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f (x) := by sorry

-- Define the statement in Lean for Part 2: proving the max and min values on [-3, 3]
theorem max_min_values : 
  ∃ max_value min_value : ℝ, 
  (max_value = f (-3) ∧ max_value = 6) ∧ 
  (min_value = f (3) ∧ min_value = -6) := by sorry

end f_is_odd_max_min_values_l31_31842


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31436

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l31_31436


namespace triangle_number_placement_l31_31915

theorem triangle_number_placement
  (A B C D E F : ℕ)
  (h1 : A + B + C = 6)
  (h2 : D = 5)
  (h3 : E = 6)
  (h4 : D + E + F = 14)
  (h5 : B = 3) : 
  (A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4) :=
by {
  sorry
}

end triangle_number_placement_l31_31915


namespace inequality_for_positive_integers_l31_31137

theorem inequality_for_positive_integers (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b + b * c + a * c ≤ 3 * a * b * c :=
sorry

end inequality_for_positive_integers_l31_31137


namespace value_of_x_l31_31620

theorem value_of_x (x : ℤ) (h : 3 * x / 7 = 21) : x = 49 :=
sorry

end value_of_x_l31_31620


namespace cone_height_l31_31495

theorem cone_height (R : ℝ) (h : ℝ) (r : ℝ) : 
  R = 8 → r = 2 → h = 2 * Real.sqrt 15 :=
by
  intro hR hr
  sorry

end cone_height_l31_31495


namespace number_is_48_l31_31804

theorem number_is_48 (x : ℝ) (h : (1/4) * x + 15 = 27) : x = 48 :=
by sorry

end number_is_48_l31_31804


namespace pizza_combinations_l31_31274

/-- The number of unique pizzas that can be made with exactly 5 toppings from a selection of 8 is 56. -/
theorem pizza_combinations : (Nat.choose 8 5) = 56 := by
  sorry

end pizza_combinations_l31_31274


namespace angle_sum_l31_31706

theorem angle_sum (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h_triangle : A + B + C = 180) (h_complement : 180 - C = 130) :
  A + B = 130 :=
by
  sorry

end angle_sum_l31_31706


namespace trader_sold_23_bags_l31_31643

theorem trader_sold_23_bags
    (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) (x : ℕ)
    (h_initial : initial_stock = 55)
    (h_restocked : restocked = 132)
    (h_final : final_stock = 164)
    (h_equation : initial_stock - x + restocked = final_stock) :
    x = 23 :=
by
    -- Here will be the proof of the theorem
    sorry

end trader_sold_23_bags_l31_31643


namespace total_homework_pages_l31_31853

theorem total_homework_pages (R : ℕ) (H1 : R + 3 = 8) : R + (R + 3) = 13 :=
by sorry

end total_homework_pages_l31_31853


namespace juice_bar_group_total_l31_31270

theorem juice_bar_group_total (total_spent : ℕ) (mango_cost : ℕ) (pineapple_cost : ℕ) 
  (spent_on_pineapple : ℕ) (num_people_total : ℕ) :
  total_spent = 94 →
  mango_cost = 5 →
  pineapple_cost = 6 →
  spent_on_pineapple = 54 →
  num_people_total = (40 / 5) + (54 / 6) →
  num_people_total = 17 :=
by {
  intros h_total h_mango h_pineapple h_pineapple_spent h_calc,
  sorry
}

end juice_bar_group_total_l31_31270


namespace customer_paid_correct_amount_l31_31223

theorem customer_paid_correct_amount (cost_price : ℕ) (markup_percentage : ℕ) (total_price : ℕ) :
  cost_price = 6500 → 
  markup_percentage = 30 → 
  total_price = cost_price + (cost_price * markup_percentage / 100) → 
  total_price = 8450 :=
by
  intros h_cost_price h_markup_percentage h_total_price
  sorry

end customer_paid_correct_amount_l31_31223


namespace diagonal_of_rectangular_prism_l31_31612

theorem diagonal_of_rectangular_prism (x y z : ℝ) (d : ℝ)
  (h_surface_area : 2 * x * y + 2 * x * z + 2 * y * z = 22)
  (h_edge_length : x + y + z = 6) :
  d = Real.sqrt 14 :=
by
  sorry

end diagonal_of_rectangular_prism_l31_31612


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31453

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l31_31453


namespace largest_divisor_of_5_consecutive_integers_l31_31468

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31468


namespace prove_pattern_example_l31_31687

noncomputable def pattern_example : Prop :=
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  (123456 * 9 + 7 = 1111111)

theorem prove_pattern_example : pattern_example := by
  sorry

end prove_pattern_example_l31_31687


namespace largest_angle_in_triangle_PQR_l31_31877

-- Definitions
def is_isosceles_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  (α = β) ∨ (β = γ) ∨ (γ = α)

def is_obtuse_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

variables (P Q R : ℝ)
variables (angleP angleQ angleR : ℝ)

-- Condition: PQR is an obtuse and isosceles triangle, and angle P measures 30 degrees
axiom h1 : is_isosceles_triangle P Q R angleP angleQ angleR
axiom h2 : is_obtuse_triangle P Q R angleP angleQ angleR
axiom h3 : angleP = 30

-- Theorem: The measure of the largest interior angle of triangle PQR is 120 degrees
theorem largest_angle_in_triangle_PQR : max angleP (max angleQ angleR) = 120 :=
  sorry

end largest_angle_in_triangle_PQR_l31_31877


namespace arithmetic_mean_of_18_24_42_l31_31244

-- Define the numbers a, b, c
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 42

-- Define the arithmetic mean
def mean (x y z : ℕ) : ℕ := (x + y + z) / 3

-- State the theorem to be proved
theorem arithmetic_mean_of_18_24_42 : mean a b c = 28 :=
by
  sorry

end arithmetic_mean_of_18_24_42_l31_31244


namespace quadratic_real_roots_l31_31321

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k + 1) * x^2 - 2 * x + 1 = 0) → (k ≤ 0 ∧ k ≠ -1) :=
by
  sorry

end quadratic_real_roots_l31_31321


namespace total_payment_l31_31729

theorem total_payment (rahul_days : ℕ) (rajesh_days : ℕ) (rahul_share : ℚ) (total_payment : ℚ) 
  (h_rahul_days : rahul_days = 3) 
  (h_rajesh_days : rajesh_days = 2) 
  (h_rahul_share : rahul_share = 42)
  (work_per_day_rahul : ℚ := 1 / rahul_days)
  (work_per_day_rajesh : ℚ := 1 / rajesh_days)
  (total_work_per_day : ℚ := work_per_day_rahul + work_per_day_rajesh) 
  (work_completed_together : total_work_per_day = 5 / 6)
  (h_proportion : rahul_share / work_per_day_rahul = total_payment / 1) :
  total_payment = 126 := 
by
  sorry

end total_payment_l31_31729


namespace original_number_l31_31131

theorem original_number (n : ℚ) (h : (3 * (n + 3) - 2) / 3 = 10) : n = 23 / 3 := 
sorry

end original_number_l31_31131


namespace sum_of_solutions_l31_31252

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 2)^2 = 81) (h2 : (x2 - 2)^2 = 81) :
  x1 + x2 = 4 := by
  sorry

end sum_of_solutions_l31_31252


namespace white_fraction_of_large_cube_l31_31025

-- Conditions
def largeCubeEdge : Nat := 4
def smallCubeEdge : Nat := 1
def totalCubes : Nat := 64
def whiteCubes : Nat := 48
def blackCubes : Nat := 16
def blackCorners : Nat := 8 -- This is inferred: 8 corners, each with a black cube
def blackEdges : Nat := 12 -- This is inferred: 12 edges

-- Surface area of a cube with given edge length
def surfaceArea (edge : Nat) : Nat := 6 * (edge * edge)

-- Number of black cubic faces exposed
def blackFacesExposed : Nat :=
  blackCorners * 3 + (blackEdges - blackCorners) -- 3 faces per cube at corners and 1 face per cube on edges excluding corners

-- Number of white cubic faces exposed
def whiteFacesExposed (totalSurfaceArea : Nat) (blackFaces : Nat) : Nat :=
  totalSurfaceArea - blackFaces

-- Fraction of white surface area
def whiteSurfaceFraction (totalSurfaceArea whiteSurfaceArea : Nat) : Rat :=
  whiteSurfaceArea (totalSurfaceArea : ℚ)

theorem white_fraction_of_large_cube :
  whiteSurfaceFraction (surfaceArea largeCubeEdge) (whiteFacesExposed (surfaceArea largeCubeEdge) blackFacesExposed) = 5 8 :=
by
  sorry

end white_fraction_of_large_cube_l31_31025


namespace sum_of_first_15_terms_l31_31625

theorem sum_of_first_15_terms (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 24) : 
  (15 / 2) * (2 * a + 14 * d) = 180 :=
by
  sorry

end sum_of_first_15_terms_l31_31625


namespace ways_to_divide_day_l31_31027

theorem ways_to_divide_day (n m : ℕ) (h : n * m = 86400) : 
  (∃ k : ℕ, k = 96) :=
  sorry

end ways_to_divide_day_l31_31027


namespace find_number_of_values_l31_31615

theorem find_number_of_values (n S : ℕ) (h1 : S / n = 250) (h2 : S + 30 = 251 * n) : n = 30 :=
sorry

end find_number_of_values_l31_31615


namespace functions_satisfying_equation_are_constants_l31_31315

theorem functions_satisfying_equation_are_constants (f g : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (x + y)) = x * f y + g x) → ∃ k : ℝ, (∀ x : ℝ, f x = k) ∧ (∀ x : ℝ, g x = k * (1 - x)) :=
by
  sorry

end functions_satisfying_equation_are_constants_l31_31315


namespace hired_year_l31_31771

theorem hired_year (A W : ℕ) (Y : ℕ) (retire_year : ℕ) 
    (hA : A = 30) 
    (h_rule : A + W = 70) 
    (h_retire : retire_year = 2006) 
    (h_employment : retire_year - Y = W) 
    : Y = 1966 := 
by 
  -- proofs are skipped with 'sorry'
  sorry

end hired_year_l31_31771


namespace mirasol_account_balance_l31_31991

theorem mirasol_account_balance :
  ∀ (initial_amount spent_coffee spent_tumbler : ℕ), 
  initial_amount = 50 → 
  spent_coffee = 10 → 
  spent_tumbler = 30 → 
  initial_amount - (spent_coffee + spent_tumbler) = 10 :=
by
  intros initial_amount spent_coffee spent_tumbler
  intro h_initial_amount
  intro h_spent_coffee
  intro h_spent_tumbler
  rw [h_initial_amount, h_spent_coffee, h_spent_tumbler]
  simp
  done

end mirasol_account_balance_l31_31991


namespace fraction_to_decimal_representation_l31_31801

/-- Determine the decimal representation of a given fraction. -/
theorem fraction_to_decimal_representation : (45 / (2 ^ 3 * 5 ^ 4) = 0.0090) :=
sorry

end fraction_to_decimal_representation_l31_31801


namespace find_k_l31_31556

theorem find_k 
  (h : ∀ x k : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0):
  ∃ (k : ℝ), k = -2 :=
sorry

end find_k_l31_31556


namespace rectangular_field_area_l31_31214

theorem rectangular_field_area (L B : ℝ) (h1 : B = 0.6 * L) (h2 : 2 * L + 2 * B = 800) : L * B = 37500 :=
by
  -- Proof will go here
  sorry

end rectangular_field_area_l31_31214


namespace abs_val_neg_three_l31_31210

-- Definition section: stating the conditions
def abs_val (x : Int) : Int := if x < 0 then -x else x

-- Statement of the proof problem
theorem abs_val_neg_three : abs_val (-3) = 3 := by
  sorry

end abs_val_neg_three_l31_31210


namespace four_digit_numbers_with_8_or_3_l31_31688

theorem four_digit_numbers_with_8_or_3 :
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  total_four_digit_numbers - numbers_without_8_or_3 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  sorry

end four_digit_numbers_with_8_or_3_l31_31688


namespace emily_can_see_emerson_l31_31662

theorem emily_can_see_emerson : 
  ∀ (emily_speed emerson_speed : ℝ) 
    (initial_distance final_distance : ℝ), 
  emily_speed = 15 → 
  emerson_speed = 9 → 
  initial_distance = 1 → 
  final_distance = 1 →
  (initial_distance / (emily_speed - emerson_speed) + final_distance / (emily_speed - emerson_speed)) * 60 = 20 :=
by
  intros emily_speed emerson_speed initial_distance final_distance
  sorry

end emily_can_see_emerson_l31_31662


namespace complex_fraction_calculation_l31_31963

theorem complex_fraction_calculation (z : ℂ) (h : z = 2 + 1 * complex.I) : (2 * complex.I) / (z - 1) = 1 + complex.I :=
by
  sorry

end complex_fraction_calculation_l31_31963


namespace midpoint_AB_find_Q_find_H_l31_31787

-- Problem 1: Midpoint of AB
theorem midpoint_AB (x1 y1 x2 y2 : ℝ) : 
  let A := (x1, y1)
  let B := (x2, y2)
  let M := ( (x1 + x2) / 2, (y1 + y2) / 2 )
  M = ( (x1 + x2) / 2, (y1 + y2) / 2 )
:= 
  -- The lean statement that shows the midpoint formula is correct.
  sorry

-- Problem 2: Coordinates of Q given midpoint
theorem find_Q (px py mx my : ℝ) : 
  let P := (px, py)
  let M := (mx, my)
  let Q := (2 * mx - px, 2 * my - py)
  ( (px + Q.1) / 2 = mx ∧ (py + Q.2) / 2 = my )
:= 
  -- Lean statement to find Q
  sorry

-- Problem 3: Coordinates of H given midpoints coinciding
theorem find_H (xE yE xF yF xG yG : ℝ) :
  let E := (xE, yE)
  let F := (xF, yF)
  let G := (xG, yG)
  ∃ xH yH : ℝ, 
    ( (xE + xH) / 2 = (xF + xG) / 2 ∧ (yE + yH) / 2 = (yF + yG) / 2 ) ∨
    ( (xF + xH) / 2 = (xE + xG) / 2 ∧ (yF + yH) / 2 = (yE + yG) / 2 ) ∨
    ( (xG + xH) / 2 = (xE + xF) / 2 ∧ (yG + yH) / 2 = (yE + yF) / 2 )
:=
  -- Lean statement to find H
  sorry

end midpoint_AB_find_Q_find_H_l31_31787


namespace range_for_a_l31_31553

variable (a : ℝ)

theorem range_for_a (h : ∀ x : ℝ, x^2 + 2 * x + a > 0) : 1 < a := 
sorry

end range_for_a_l31_31553


namespace remaining_volume_after_pours_l31_31026

-- Definitions based on the problem conditions
def initial_volume_liters : ℝ := 2
def initial_volume_milliliters : ℝ := initial_volume_liters * 1000
def pour_amount (x : ℝ) : ℝ := x

-- Statement of the problem as a theorem in Lean 4
theorem remaining_volume_after_pours (x : ℝ) : 
  ∃ remaining_volume : ℝ, remaining_volume = initial_volume_milliliters - 4 * pour_amount x :=
by
  -- To be filled with the proof
  sorry

end remaining_volume_after_pours_l31_31026


namespace odd_palindrome_count_l31_31588

theorem odd_palindrome_count :
  (∃ A B : ℕ, 1 ≤ A ∧ A ≤ 9 ∧ A % 2 = 1 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 9 * B + A = 50) :=
begin
  sorry
end

end odd_palindrome_count_l31_31588


namespace sum_of_arithmetic_sequence_l31_31956

-- Given conditions in the problem
axiom arithmetic_sequence (a : ℕ → ℤ): Prop
axiom are_roots (a b : ℤ): ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a = p ∨ a = q) ∧ (b = p ∨ b = q)

-- The equivalent proof problem statement
theorem sum_of_arithmetic_sequence (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a)
  (h2 : ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a 2 = p ∨ a 2 = q) ∧ (a 11 = p ∨ a 11 = q)):
  a 5 + a 8 = 3 :=
sorry

end sum_of_arithmetic_sequence_l31_31956


namespace remainder_calculation_l31_31481

theorem remainder_calculation 
  (dividend divisor quotient : ℕ)
  (h1 : dividend = 140)
  (h2 : divisor = 15)
  (h3 : quotient = 9) :
  dividend = (divisor * quotient) + (dividend - (divisor * quotient)) := by
sorry

end remainder_calculation_l31_31481


namespace max_divisor_of_five_consecutive_integers_l31_31446

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l31_31446


namespace Harkamal_purchase_grapes_l31_31095

theorem Harkamal_purchase_grapes
  (G : ℕ) -- The number of kilograms of grapes
  (cost_grapes_per_kg : ℕ := 70)
  (kg_mangoes : ℕ := 9)
  (cost_mangoes_per_kg : ℕ := 55)
  (total_paid : ℕ := 1195) :
  70 * G + 55 * 9 = 1195 → G = 10 := 
by
  sorry

end Harkamal_purchase_grapes_l31_31095


namespace percentage_difference_l31_31763

theorem percentage_difference :
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  a1 - a2 = 1.484 := 
by
  -- Definitions
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  -- Proof body (skipped for this task)
  sorry

end percentage_difference_l31_31763


namespace solve_triples_l31_31489

theorem solve_triples (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  a^2 + b^2 = n * Nat.lcm a b + n^2 ∧
  b^2 + c^2 = n * Nat.lcm b c + n^2 ∧
  c^2 + a^2 = n * Nat.lcm c a + n^2 →
  ∃ k : ℕ, 0 < k ∧ a = k ∧ b = k ∧ c = k :=
by
  intros h
  sorry

end solve_triples_l31_31489


namespace largest_among_a_b_c_d_l31_31363

noncomputable def a : ℝ := Real.sin (Real.cos (2015 * Real.pi / 180))
noncomputable def b : ℝ := Real.sin (Real.sin (2015 * Real.pi / 180))
noncomputable def c : ℝ := Real.cos (Real.sin (2015 * Real.pi / 180))
noncomputable def d : ℝ := Real.cos (Real.cos (2015 * Real.pi / 180))

theorem largest_among_a_b_c_d : c = max a (max b (max c d)) := by
  sorry

end largest_among_a_b_c_d_l31_31363


namespace imaginary_part_of_complex_l31_31382

open Complex -- Opens the complex numbers namespace

theorem imaginary_part_of_complex:
  ∀ (a b c d : ℂ), (a = (2 + I) / (1 - I) - (2 - I) / (1 + I)) → (a.im = 3) :=
by
  sorry

end imaginary_part_of_complex_l31_31382


namespace cone_height_is_correct_l31_31492

noncomputable def cone_height (r_circle: ℝ) (num_sectors: ℝ) : ℝ :=
  let C := 2 * real.pi * r_circle in
  let sector_circumference := C / num_sectors in
  let base_radius := sector_circumference / (2 * real.pi) in
  let slant_height := r_circle in
  real.sqrt (slant_height^2 - base_radius^2)

theorem cone_height_is_correct :
  cone_height 8 4 = 2 * real.sqrt 15 :=
by
  rw cone_height
  norm_num
  sorry

end cone_height_is_correct_l31_31492


namespace fraction_of_shoppers_avoiding_checkout_l31_31893

theorem fraction_of_shoppers_avoiding_checkout 
  (total_shoppers : ℕ) 
  (shoppers_at_checkout : ℕ) 
  (h1 : total_shoppers = 480) 
  (h2 : shoppers_at_checkout = 180) : 
  (total_shoppers - shoppers_at_checkout) / total_shoppers = 5 / 8 :=
by
  sorry

end fraction_of_shoppers_avoiding_checkout_l31_31893


namespace arithmetic_sequence_third_term_l31_31603

theorem arithmetic_sequence_third_term
  (a d : ℤ)
  (h_fifteenth_term : a + 14 * d = 15)
  (h_sixteenth_term : a + 15 * d = 21) :
  a + 2 * d = -57 :=
by
  sorry

end arithmetic_sequence_third_term_l31_31603


namespace zorbs_of_60_deg_l31_31134

-- Define the measurement on Zorblat
def zorbs_in_full_circle := 600
-- Define the Earth angle in degrees
def earth_degrees_full_circle := 360
def angle_in_degrees := 60
-- Calculate the equivalent angle in zorbs
def zorbs_in_angle := zorbs_in_full_circle * angle_in_degrees / earth_degrees_full_circle

theorem zorbs_of_60_deg (h1 : zorbs_in_full_circle = 600)
                        (h2 : earth_degrees_full_circle = 360)
                        (h3 : angle_in_degrees = 60) :
  zorbs_in_angle = 100 :=
by sorry

end zorbs_of_60_deg_l31_31134


namespace resistor_problem_l31_31751

theorem resistor_problem (R : ℝ)
  (initial_resistance : ℝ := 3 * R)
  (parallel_resistance : ℝ := R / 3)
  (resistance_change : ℝ := initial_resistance - parallel_resistance)
  (condition : resistance_change = 10) : 
  R = 3.75 := by
  sorry

end resistor_problem_l31_31751


namespace typist_salary_proof_l31_31610

noncomputable def original_salary (x : ℝ) : Prop :=
  1.10 * x * 0.95 = 1045

theorem typist_salary_proof (x : ℝ) (H : original_salary x) : x = 1000 :=
sorry

end typist_salary_proof_l31_31610


namespace intersection_point_sum_l31_31088

theorem intersection_point_sum {h j : ℝ → ℝ} 
    (h3: h 3 = 3) (j3: j 3 = 3) 
    (h6: h 6 = 9) (j6: j 6 = 9)
    (h9: h 9 = 18) (j9: j 9 = 18)
    (h12: h 12 = 18) (j12: j 12 = 18) :
    ∃ a b, (h (3 * a) = 3 * j a ∧ a + b = 22) := 
sorry

end intersection_point_sum_l31_31088


namespace price_of_second_box_l31_31044

noncomputable def price_of_first_box : ℝ := 25
noncomputable def contacts_in_first_box : ℕ := 50
noncomputable def contacts_in_second_box : ℕ := 99
noncomputable def price_per_contact_first_box : ℝ := price_of_first_box / contacts_in_first_box
noncomputable def chosen_price_per_contact : ℝ := 1 / 3

theorem price_of_second_box :
  chosen_price_per_contact < price_per_contact_first_box →
  let price_per_contact_second_box := chosen_price_per_contact in
  let total_price_second_box := price_per_contact_second_box * contacts_in_second_box in
  total_price_second_box = 32.67 :=
by
  intros h
  let price_per_contact_second_box := chosen_price_per_contact
  let total_price_second_box := price_per_contact_second_box * contacts_in_second_box
  have : total_price_second_box = 32.67 := sorry
  exact this

end price_of_second_box_l31_31044


namespace min_abs_sum_l31_31583

theorem min_abs_sum (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a^2 + b * c = 9) (h2 : b * c + d^2 = 9) (h3 : a * b + b * d = 0) (h4 : a * c + c * d = 0) :
  |a| + |b| + |c| + |d| = 8 :=
sorry

end min_abs_sum_l31_31583


namespace initial_deadline_l31_31590

theorem initial_deadline (W : ℕ) (R : ℕ) (D : ℕ) :
    100 * 25 * 8 = (1/3 : ℚ) * W →
    (2/3 : ℚ) * W = 160 * R * 10 →
    D = 25 + R →
    D = 50 := 
by
  intros h1 h2 h3
  sorry

end initial_deadline_l31_31590


namespace sum_of_largest_smallest_angles_l31_31336

noncomputable section

def sides_ratio (a b c : ℝ) : Prop := a / 5 = b / 7 ∧ b / 7 = c / 8

theorem sum_of_largest_smallest_angles (a b c : ℝ) (θA θB θC : ℝ) 
  (h1 : sides_ratio a b c) 
  (h2 : a^2 + b^2 - c^2 = 2 * a * b * Real.cos θC)
  (h3 : b^2 + c^2 - a^2 = 2 * b * c * Real.cos θA)
  (h4 : c^2 + a^2 - b^2 = 2 * c * a * Real.cos θB)
  (h5 : θA + θB + θC = 180) :
  θA + θC = 120 :=
sorry

end sum_of_largest_smallest_angles_l31_31336


namespace find_divisor_l31_31106

def remainder : Nat := 1
def quotient : Nat := 54
def dividend : Nat := 217

theorem find_divisor : ∃ divisor : Nat, (dividend = divisor * quotient + remainder) ∧ divisor = 4 :=
by
  sorry

end find_divisor_l31_31106


namespace problem_I_problem_II_l31_31943

open Set

variable (a x : ℝ)

def p : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem problem_I (hp : p a) : a ≤ 1 :=
  sorry

theorem problem_II (hpq : ¬ (p a ∧ q a)) : a ∈ Ioo (-2 : ℝ) (1 : ℝ) ∪ Ioi 1 :=
  sorry

end problem_I_problem_II_l31_31943


namespace find_larger_number_l31_31611

theorem find_larger_number (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x * y = 375) (hx : x > y) : x = 25 :=
sorry

end find_larger_number_l31_31611


namespace arithmetic_sequence_eighth_term_l31_31169

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l31_31169


namespace promotional_price_difference_l31_31907

theorem promotional_price_difference
  (normal_price : ℝ)
  (months : ℕ)
  (issues_per_month : ℕ)
  (discount_per_issue : ℝ)
  (h1 : normal_price = 34)
  (h2 : months = 18)
  (h3 : issues_per_month = 2)
  (h4 : discount_per_issue = 0.25) : 
  normal_price - (months * issues_per_month * discount_per_issue) = 9 := 
by 
  sorry

end promotional_price_difference_l31_31907


namespace largest_angle_in_triangle_PQR_is_75_degrees_l31_31576

noncomputable def largest_angle (p q r : ℝ) : ℝ :=
  if p + q + 2 * r = p^2 ∧ p + q - 2 * r = -1 then 
    Real.arccos ((p^2 + q^2 - (p^2 + p*q + (1/2)*q^2)/2) / (2 * p * q)) * (180/Real.pi)
  else 
    0

theorem largest_angle_in_triangle_PQR_is_75_degrees (p q r : ℝ) (h1 : p + q + 2 * r = p^2) (h2 : p + q - 2 * r = -1) :
  largest_angle p q r = 75 :=
by sorry

end largest_angle_in_triangle_PQR_is_75_degrees_l31_31576


namespace greatest_value_x_l31_31317

theorem greatest_value_x (x : ℝ) : 
  (x ≠ 9) → 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) →
  x ≤ -2 :=
by
  sorry

end greatest_value_x_l31_31317


namespace walking_rate_on_escalator_l31_31652

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 160)
  (time_taken : ℝ := 8)
  (distance_eq : escalator_length = (v + escalator_speed) * time_taken) :
  v = 8 :=
by
  sorry

end walking_rate_on_escalator_l31_31652


namespace quadratic_has_minimum_l31_31504

theorem quadratic_has_minimum 
  (a b : ℝ) (h : a ≠ 0) (g : ℝ → ℝ) 
  (H : ∀ x, g x = a * x^2 + b * x + (b^2 / a)) :
  ∃ x₀, ∀ x, g x ≥ g x₀ :=
by sorry

end quadratic_has_minimum_l31_31504


namespace ant_path_count_l31_31651

theorem ant_path_count :
  let binom := Nat.choose 4020 1005 in
  ∃ f : Fin 2 → ℕ, 
  (f 0 = 2010 ∧ f 1 = 2010 ∧ (∑ x : Fin 4020, if x.mod 2 = 0 then 1 else -1) = 4020) →
  binom * binom = (Nat.choose 4020 1005) ^ 2 := 
by
  sorry

end ant_path_count_l31_31651


namespace parabola_focus_line_intersection_l31_31679

theorem parabola_focus_line_intersection :
  let F := (1, 0)
  let parabola (x y : ℝ) := y^2 = 4 * x
  let line (x y : ℝ) := y = sqrt(3) * (x - 1)
  let A_x := 3
  let B_x := 1 / 3
  let A := (A_x, sqrt(3) * (A_x - 1))
  let B := (B_x, sqrt(3) * (B_x - 1))
  dist_sq (1, 0) A - dist_sq (1, 0) B = 128 / 9 := sorry

def dist_sq (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

end parabola_focus_line_intersection_l31_31679


namespace luisa_trip_l31_31715

noncomputable def additional_miles (d1: ℝ) (s1: ℝ) (s2: ℝ) (desired_avg_speed: ℝ) : ℝ := 
  let t1 := d1 / s1
  let t := (d1 * (desired_avg_speed - s1)) / (s2 * (s1 - desired_avg_speed))
  s2 * t

theorem luisa_trip :
  additional_miles 18 36 60 45 = 18 :=
by
  sorry

end luisa_trip_l31_31715


namespace binary_multiplication_correct_l31_31809

theorem binary_multiplication_correct:
  let n1 := 29 -- binary 11101 is decimal 29
  let n2 := 13 -- binary 1101 is decimal 13
  let result := 303 -- binary 100101111 is decimal 303
  n1 * n2 = result :=
by
  -- Proof goes here
  sorry

end binary_multiplication_correct_l31_31809


namespace oscar_leap_vs_elmer_stride_l31_31797

theorem oscar_leap_vs_elmer_stride :
  ∀ (num_poles : ℕ) (distance : ℝ) (elmer_strides_per_gap : ℕ) (oscar_leaps_per_gap : ℕ)
    (elmer_stride_time_mult : ℕ) (total_distance_poles : ℕ)
    (elmer_total_strides : ℕ) (oscar_total_leaps : ℕ) (elmer_stride_length : ℝ)
    (oscar_leap_length : ℝ) (expected_diff : ℝ),
    num_poles = 81 →
    distance = 10560 →
    elmer_strides_per_gap = 60 →
    oscar_leaps_per_gap = 15 →
    elmer_stride_time_mult = 2 →
    total_distance_poles = 2 →
    elmer_total_strides = elmer_strides_per_gap * (num_poles - 1) →
    oscar_total_leaps = oscar_leaps_per_gap * (num_poles - 1) →
    elmer_stride_length = distance / elmer_total_strides →
    oscar_leap_length = distance / oscar_total_leaps →
    expected_diff = oscar_leap_length - elmer_stride_length →
    expected_diff = 6.6
:= sorry

end oscar_leap_vs_elmer_stride_l31_31797


namespace max_divisor_of_five_consecutive_integers_l31_31447

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l31_31447


namespace coin_overlap_black_region_cd_sum_l31_31535

noncomputable def black_region_probability : ℝ := 
  let square_side := 10
  let triangle_leg := 3
  let diamond_side := 3 * Real.sqrt 2
  let coin_diameter := 2
  let coin_radius := coin_diameter / 2
  let reduced_square_side := square_side - coin_diameter
  let reduced_square_area := reduced_square_side * reduced_square_side
  let triangle_area := 4 * ((triangle_leg * triangle_leg) / 2)
  let extra_triangle_area := 4 * (Real.pi / 4 + 3)
  let diamond_area := (diamond_side * diamond_side) / 2
  let extra_diamond_area := Real.pi + 12 * Real.sqrt 2
  let total_black_area := triangle_area + extra_triangle_area + diamond_area + extra_diamond_area

  total_black_area / reduced_square_area

theorem coin_overlap_black_region: 
  black_region_probability = (1 / 64) * (30 + 12 * Real.sqrt 2 + Real.pi) := 
sorry

theorem cd_sum: 
  let c := 30
  let d := 12
  c + d = 42 := 
by
  trivial

end coin_overlap_black_region_cd_sum_l31_31535


namespace bench_cost_l31_31268

theorem bench_cost (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
by {
  sorry
}

end bench_cost_l31_31268


namespace square_roots_of_x_l31_31565

theorem square_roots_of_x (a x : ℝ) 
    (h1 : (2 * a - 1) ^ 2 = x) 
    (h2 : (-a + 2) ^ 2 = x)
    (hx : 0 < x) 
    : x = 9 ∨ x = 1 := 
by sorry

end square_roots_of_x_l31_31565


namespace remainder_equality_l31_31552

theorem remainder_equality (P P' : ℕ) (h1 : P = P' + 10) 
  (h2 : P % 10 = 0) (h3 : P' % 10 = 0) : 
  ((P^2 - P'^2) % 10 = 0) :=
by
  sorry

end remainder_equality_l31_31552


namespace fraction_inequality_solution_l31_31000

open Set

theorem fraction_inequality_solution :
  {x : ℝ | 7 * x - 3 ≥ x^2 - x - 12 ∧ x ≠ 3 ∧ x ≠ -4} = Icc (-1 : ℝ) 3 ∪ Ioo (3 : ℝ) 4 ∪ Icc 4 9 :=
by
  sorry

end fraction_inequality_solution_l31_31000


namespace polynomial_representation_l31_31257

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 6)

theorem polynomial_representation (x : ℝ) :
  given_expression x = 6 * x^3 - 4 * x^2 - 26 * x + 20 :=
sorry

end polynomial_representation_l31_31257


namespace exterior_angle_BAC_eq_162_l31_31781

noncomputable def measure_of_angle_BAC : ℝ := 360 - 108 - 90

theorem exterior_angle_BAC_eq_162 :
  measure_of_angle_BAC = 162 := by
  sorry

end exterior_angle_BAC_eq_162_l31_31781


namespace minimum_cost_l31_31896

noncomputable def volume : ℝ := 4800
noncomputable def depth : ℝ := 3
noncomputable def base_cost_per_sqm : ℝ := 150
noncomputable def wall_cost_per_sqm : ℝ := 120
noncomputable def base_area (volume depth : ℝ) : ℝ := volume / depth
noncomputable def wall_surface_area (x : ℝ) : ℝ :=
  6 * x + (2 * (volume * depth / x))

noncomputable def construction_cost (x : ℝ) : ℝ :=
  wall_surface_area x * wall_cost_per_sqm + base_area volume depth * base_cost_per_sqm

theorem minimum_cost :
  ∃(x : ℝ), x = 40 ∧ construction_cost x = 297600 := by
  sorry

end minimum_cost_l31_31896


namespace problem_l31_31327

theorem problem (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (17 * x)) (x : ℝ) :
  f (Real.cos x) ^ 2 + f (Real.sin x) ^ 2 = 1 :=
sorry

end problem_l31_31327


namespace arithmetic_sequence_eighth_term_l31_31172

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l31_31172


namespace remaining_balance_is_correct_l31_31988

def initial_balance : ℕ := 50
def spent_coffee : ℕ := 10
def spent_tumbler : ℕ := 30

theorem remaining_balance_is_correct : initial_balance - (spent_coffee + spent_tumbler) = 10 := by
  sorry

end remaining_balance_is_correct_l31_31988


namespace arithmetic_sequence_8th_term_l31_31201

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l31_31201


namespace stuffed_animal_cost_is_6_l31_31516

-- Definitions for the costs of items
def sticker_cost (s : ℕ) := s
def magnet_cost (m : ℕ) := m
def stuffed_animal_cost (a : ℕ) := a

-- Conditions given in the problem
def conditions (m s a : ℕ) :=
  (m = 3) ∧
  (m = 3 * s) ∧
  (m = (2 * a) / 4)

-- The theorem stating the cost of a single stuffed animal
theorem stuffed_animal_cost_is_6 (s m a : ℕ) (h : conditions m s a) : a = 6 :=
by
  sorry

end stuffed_animal_cost_is_6_l31_31516


namespace benny_march_savings_l31_31049

theorem benny_march_savings :
  (january_add : ℕ) (february_add : ℕ) (march_total : ℕ) 
  (H1 : january_add = 19) (H2 : february_add = 19) (H3 : march_total = 46) :
  march_total - (january_add + february_add) = 8 := 
by
  sorry

end benny_march_savings_l31_31049


namespace margin_in_terms_of_selling_price_l31_31510

variable (C S M : ℝ) (n : ℕ) (h : M = (1 / 2) * (S - (1 / n) * C))

theorem margin_in_terms_of_selling_price :
  M = ((n - 1) / (2 * n - 1)) * S :=
sorry

end margin_in_terms_of_selling_price_l31_31510


namespace problem_condition_l31_31957

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l31_31957


namespace curlers_count_l31_31311

theorem curlers_count (T P B G : ℕ) 
  (hT : T = 16)
  (hP : P = T / 4)
  (hB : B = 2 * P)
  (hG : G = T - (P + B)) : 
  G = 4 :=
by
  sorry

end curlers_count_l31_31311


namespace range_of_m_l31_31107

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (x + m) / (x - 2) + 2 * m / (2 - x) = 3) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_l31_31107


namespace park_area_l31_31747

theorem park_area (l w : ℝ) (h1 : 2 * l + 2 * w = 80) (h2 : l = 3 * w) : l * w = 300 :=
sorry

end park_area_l31_31747


namespace find_abcde_l31_31674

noncomputable def find_five_digit_number (a b c d e : ℕ) : ℕ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem find_abcde
  (a b c d e : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 0 ≤ e ∧ e ≤ 9)
  (h6 : a ≠ 0)
  (h7 : (10 * a + b + 10 * b + c) * (10 * b + c + 10 * c + d) * (10 * c + d + 10 * d + e) = 157605) :
  find_five_digit_number a b c d e = 12345 ∨ find_five_digit_number a b c d e = 21436 :=
sorry

end find_abcde_l31_31674


namespace bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l31_31265

variables {a b c : ℝ}
-- Given conditions from Vieta's formulas for the polynomial x^3 - 20x^2 + 22
axiom vieta1 : a + b + c = 20
axiom vieta2 : a * b + b * c + c * a = 0
axiom vieta3 : a * b * c = -22

theorem bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3 (a b c : ℝ)
  (h1 : a + b + c = 20)
  (h2 : a * b + b * c + c * a = 0)
  (h3 : a * b * c = -22) :
  (b * c / a^2) + (a * c / b^2) + (a * b / c^2) = 3 := 
  sorry

end bc_over_a_sq_plus_ac_over_b_sq_plus_ab_over_c_sq_eq_3_l31_31265


namespace time_to_pass_jogger_l31_31900

noncomputable def jogger_speed_kmh : ℕ := 9
noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_length : ℕ := 130
noncomputable def jogger_ahead_distance : ℕ := 240
noncomputable def train_speed_kmh : ℕ := 45
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover : ℕ := jogger_ahead_distance + train_length
noncomputable def time_taken_to_pass : ℝ := total_distance_to_cover / relative_speed

theorem time_to_pass_jogger : time_taken_to_pass = 37 := sorry

end time_to_pass_jogger_l31_31900


namespace calculate_total_students_l31_31698

/-- Define the number of students who like basketball, cricket, and soccer. -/
def likes_basketball : ℕ := 7
def likes_cricket : ℕ := 10
def likes_soccer : ℕ := 8
def likes_all_three : ℕ := 2
def likes_basketball_and_cricket : ℕ := 5
def likes_basketball_and_soccer : ℕ := 4
def likes_cricket_and_soccer : ℕ := 3

/-- Calculate the number of students who like at least one sport using the principle of inclusion-exclusion. -/
def students_who_like_at_least_one_sport (b c s bc bs cs bcs : ℕ) : ℕ :=
  b + c + s - (bc + bs + cs) + bcs

theorem calculate_total_students :
  students_who_like_at_least_one_sport likes_basketball likes_cricket likes_soccer 
    (likes_basketball_and_cricket - likes_all_three) 
    (likes_basketball_and_soccer - likes_all_three) 
    (likes_cricket_and_soccer - likes_all_three) 
    likes_all_three = 21 := 
by
  sorry

end calculate_total_students_l31_31698


namespace arithmetic_seq_8th_term_l31_31187

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l31_31187


namespace CarlyWorkedOnElevenDogs_l31_31530

-- Given conditions
def CarlyTrimmedNails : ℕ := 164
def DogsWithThreeLegs : ℕ := 3
def NailsPerPaw : ℕ := 4
def PawsPerThreeLeggedDog : ℕ := 3
def PawsPerFourLeggedDog : ℕ := 4

-- Deduction steps
def TotalPawsWorkedOn := CarlyTrimmedNails / NailsPerPaw
def PawsOnThreeLeggedDogs := DogsWithThreeLegs * PawsPerThreeLeggedDog
def PawsOnFourLeggedDogs := TotalPawsWorkedOn - PawsOnThreeLeggedDogs
def CountFourLeggedDogs := PawsOnFourLeggedDogs / PawsPerFourLeggedDog

-- Total dogs Carly worked on
def TotalDogsCarlyWorkedOn := CountFourLeggedDogs + DogsWithThreeLegs

-- The statement we need to prove
theorem CarlyWorkedOnElevenDogs : TotalDogsCarlyWorkedOn = 11 := by
  sorry

end CarlyWorkedOnElevenDogs_l31_31530


namespace number_of_elements_in_M_l31_31388

def positive_nats : Set ℕ := {n | n > 0}
def M : Set ℕ := {m | ∃ n ∈ positive_nats, m = 2 * n - 1 ∧ m < 60}

theorem number_of_elements_in_M : ∃ s : Finset ℕ, (∀ x, x ∈ s ↔ x ∈ M) ∧ s.card = 30 := 
by
  sorry

end number_of_elements_in_M_l31_31388


namespace bao_interest_l31_31908

noncomputable def initial_amount : ℝ := 1000
noncomputable def interest_rate : ℝ := 0.05
noncomputable def periods : ℕ := 6
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ periods
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem bao_interest :
  interest_earned = 340.095 := by
  sorry

end bao_interest_l31_31908


namespace find_focus_of_parabola_l31_31861

-- Define the given parabola equation
def parabola_eqn (x : ℝ) : ℝ := -4 * x^2

-- Define a predicate to check if the point is the focus
def is_focus (x y : ℝ) := x = 0 ∧ y = -1 / 16

theorem find_focus_of_parabola :
  is_focus 0 (parabola_eqn 0) :=
sorry

end find_focus_of_parabola_l31_31861


namespace evaluate_expression_l31_31540

def expression (x y : ℤ) : ℤ :=
  y * (y - 2 * x) ^ 2

theorem evaluate_expression : 
  expression 4 2 = 72 :=
by
  -- Proof will go here
  sorry

end evaluate_expression_l31_31540


namespace peter_total_dogs_l31_31996

def num_german_shepherds_sam : ℕ := 3
def num_french_bulldogs_sam : ℕ := 4
def num_german_shepherds_peter := 3 * num_german_shepherds_sam
def num_french_bulldogs_peter := 2 * num_french_bulldogs_sam

theorem peter_total_dogs : num_german_shepherds_peter + num_french_bulldogs_peter = 17 :=
by {
  -- adding proofs later
  sorry
}

end peter_total_dogs_l31_31996


namespace necessary_but_not_sufficient_condition_l31_31985

variable (x y : ℝ)

theorem necessary_but_not_sufficient_condition :
  (x ≠ 1 ∨ y ≠ 1) ↔ (xy ≠ 1) :=
sorry

end necessary_but_not_sufficient_condition_l31_31985


namespace prove_a2_a3_a4_sum_l31_31325

theorem prove_a2_a3_a4_sum (a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, a1 * (x-1)^4 + a2 * (x-1)^3 + a3 * (x-1)^2 + a4 * (x-1) + a5 = x^4) :
  a2 + a3 + a4 = 14 :=
sorry

end prove_a2_a3_a4_sum_l31_31325


namespace original_inhabitants_l31_31647

theorem original_inhabitants (X : ℝ) 
  (h1 : 10 ≤ X) 
  (h2 : 0.9 * X * 0.75 + 0.225 * X * 0.15 = 5265) : 
  X = 7425 := 
sorry

end original_inhabitants_l31_31647


namespace five_consecutive_product_div_24_l31_31445

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l31_31445


namespace meeting_success_probability_l31_31901

noncomputable def meeting_probability : ℝ :=
  let totalVolume := 1.5 ^ 3
  let z_gt_x_y := (1.5 * 1.5 * 1.5) / 3
  let assistants_leave := 2 * ((1.5 * 0.5 / 2) / 3 * 0.5)
  let effectiveVolume := z_gt_x_y - assistants_leave
  let probability := effectiveVolume / totalVolume
  probability

theorem meeting_success_probability :
  meeting_probability = 8 / 27 := by
  sorry

end meeting_success_probability_l31_31901


namespace center_of_circle_l31_31058

theorem center_of_circle (x y : ℝ) : 
    (∃ x y : ℝ, x^2 + y^2 = 4*x - 6*y + 9) → (x, y) = (2, -3) := 
by sorry

end center_of_circle_l31_31058


namespace josh_bottle_caps_l31_31977

/--
Suppose:
1. 7 bottle caps weigh exactly one ounce.
2. Josh's entire bottle cap collection weighs 18 pounds exactly.
3. There are 16 ounces in 1 pound.
We aim to show that Josh has 2016 bottle caps in his collection.
-/
theorem josh_bottle_caps :
  (7 : ℕ) * (1 : ℕ) = (7 : ℕ) → 
  (18 : ℕ) * (16 : ℕ) = (288 : ℕ) →
  (288 : ℕ) * (7 : ℕ) = (2016 : ℕ) :=
by
  intros h1 h2;
  exact sorry

end josh_bottle_caps_l31_31977


namespace valid_paths_from_P_to_Q_l31_31899

-- Define the grid dimensions and alternate coloring conditions
def grid_width := 10
def grid_height := 8
def is_white_square (r c : ℕ) : Bool :=
  (r + c) % 2 = 1

-- Define the starting and ending squares P and Q
def P : ℕ × ℕ := (0, grid_width / 2)
def Q : ℕ × ℕ := (grid_height - 1, grid_width / 2)

-- Define a function to count valid 9-step paths from P to Q
noncomputable def count_valid_paths : ℕ :=
  -- Here the function to compute valid paths would be defined
  -- This is broad outline due to lean's framework missing specific combinatorial functions
  245

-- The theorem to state the proof problem
theorem valid_paths_from_P_to_Q : count_valid_paths = 245 :=
sorry

end valid_paths_from_P_to_Q_l31_31899


namespace product_of_5_consecutive_integers_divisible_by_60_l31_31464

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l31_31464


namespace exponentiation_equality_l31_31054

theorem exponentiation_equality :
  3^12 * 8^12 * 3^3 * 8^8 = 24 ^ 15 * 32768 := by
  sorry

end exponentiation_equality_l31_31054


namespace eight_in_M_nine_in_M_ten_not_in_M_l31_31514

def M (a : ℤ) : Prop := ∃ b c : ℤ, a = b^2 - c^2

theorem eight_in_M : M 8 := by
  sorry

theorem nine_in_M : M 9 := by
  sorry

theorem ten_not_in_M : ¬ M 10 := by
  sorry

end eight_in_M_nine_in_M_ten_not_in_M_l31_31514


namespace average_temperature_problem_l31_31741

variable {T W Th F : ℝ}

theorem average_temperature_problem (h1 : (W + Th + 44) / 3 = 34) (h2 : T = 38) : 
  (T + W + Th) / 3 = 32 := by
  sorry

end average_temperature_problem_l31_31741


namespace area_of_rectangular_park_l31_31749

theorem area_of_rectangular_park
  (l w : ℕ) 
  (h_perimeter : 2 * l + 2 * w = 80)
  (h_length : l = 3 * w) :
  l * w = 300 :=
sorry

end area_of_rectangular_park_l31_31749


namespace phase_shift_equivalence_l31_31874

noncomputable def y_original (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def y_target (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1
noncomputable def phase_shift : ℝ := 5 * Real.pi / 12

theorem phase_shift_equivalence : 
  ∀ x : ℝ, y_original x = y_target (x - phase_shift) :=
sorry

end phase_shift_equivalence_l31_31874


namespace arithmetic_sequence_8th_term_l31_31198

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l31_31198


namespace math_problem_l31_31582

theorem math_problem (a b c k : ℝ) (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h2 : a + b + c = 0) (h3 : a^2 = k * b^2) (hk : k ≠ 0) :
  (a^2 * b^2) / ((a^2 - b * c) * (b^2 - a * c)) + (a^2 * c^2) / ((a^2 - b * c) * (c^2 - a * b)) + (b^2 * c^2) / ((b^2 - a * c) * (c^2 - a * b)) = 1 :=
by
  sorry

end math_problem_l31_31582


namespace daniel_waist_size_correct_l31_31930

noncomputable def Daniel_waist_size_cm (inches_to_feet : ℝ) (feet_to_cm : ℝ) (waist_size_in_inches : ℝ) : ℝ := 
  (waist_size_in_inches * feet_to_cm) / inches_to_feet

theorem daniel_waist_size_correct :
  Daniel_waist_size_cm 12 30.5 34 = 86.4 :=
by
  -- This skips the proof for now
  sorry

end daniel_waist_size_correct_l31_31930


namespace number_of_children_l31_31351

theorem number_of_children (C : ℝ) 
  (h1 : 0.30 * C >= 0)
  (h2 : 0.20 * C >= 0)
  (h3 : 0.50 * C >= 0)
  (h4 : 0.70 * C = 42) : 
  C = 60 := by
  sorry

end number_of_children_l31_31351


namespace speed_in_still_water_l31_31014

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 25) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 45 :=
by
  sorry

end speed_in_still_water_l31_31014


namespace remainder_when_divided_by_29_l31_31903

theorem remainder_when_divided_by_29 (N : ℤ) (k : ℤ) (h : N = 751 * k + 53) : 
  N % 29 = 24 := 
by 
  sorry

end remainder_when_divided_by_29_l31_31903


namespace largest_divisor_of_5_consecutive_integers_l31_31480

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31480


namespace arithmetic_sequence_8th_term_l31_31166

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l31_31166


namespace problem1_problem2_l31_31678

-- Define the sets P and Q
def set_P : Set ℝ := {x | 2 * x^2 - 5 * x - 3 < 0}
def set_Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Problem (1): P ∩ Q = Q implies a ∈ (-1/2, 2)
theorem problem1 (a : ℝ) : (set_Q a) ⊆ set_P → -1/2 < a ∧ a < 2 :=
by 
  sorry

-- Problem (2): P ∩ Q = ∅ implies a ∈ (-∞, -3/2] ∪ [3, ∞)
theorem problem2 (a : ℝ) : (set_Q a) ∩ set_P = ∅ → a ≤ -3/2 ∨ a ≥ 3 :=
by 
  sorry

end problem1_problem2_l31_31678


namespace trucks_and_goods_l31_31637

variable (x : ℕ) -- Number of trucks
variable (goods : ℕ) -- Total tons of goods

-- Conditions
def condition1 : Prop := goods = 3 * x + 5
def condition2 : Prop := goods = 4 * (x - 5)

theorem trucks_and_goods (h1 : condition1 x goods) (h2 : condition2 x goods) : x = 25 ∧ goods = 80 :=
by
  sorry

end trucks_and_goods_l31_31637


namespace external_angle_at_C_l31_31835

-- Definitions based on conditions
def angleA : ℝ := 40
def B := 2 * angleA
def sum_of_angles_in_triangle (A B C : ℝ) : Prop := A + B + C = 180
def external_angle (C : ℝ) : ℝ := 180 - C

-- Theorem statement
theorem external_angle_at_C :
  ∃ C : ℝ, sum_of_angles_in_triangle angleA B C ∧ external_angle C = 120 :=
sorry

end external_angle_at_C_l31_31835


namespace mirasol_balance_l31_31989

/-- Given Mirasol initially had $50, spends $10 on coffee beans, and $30 on a tumbler,
    prove that the remaining balance in her account is $10. -/
theorem mirasol_balance (initial_balance spent_coffee spent_tumbler remaining_balance : ℕ)
  (h1 : initial_balance = 50)
  (h2 : spent_coffee = 10)
  (h3 : spent_tumbler = 30)
  (h4 : remaining_balance = initial_balance - (spent_coffee + spent_tumbler)) :
  remaining_balance = 10 :=
sorry

end mirasol_balance_l31_31989


namespace largest_divisor_of_5_consecutive_integers_l31_31471

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31471


namespace percent_y_of_x_l31_31888

theorem percent_y_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y / x = 1 / 3 :=
by
  -- proof steps would be provided here
  sorry

end percent_y_of_x_l31_31888


namespace remainders_of_65_powers_l31_31918

theorem remainders_of_65_powers (n : ℕ) :
  (65 ^ (6 * n)) % 9 = 1 ∧
  (65 ^ (6 * n + 1)) % 9 = 2 ∧
  (65 ^ (6 * n + 2)) % 9 = 4 ∧
  (65 ^ (6 * n + 3)) % 9 = 8 :=
by
  sorry

end remainders_of_65_powers_l31_31918


namespace max_divisor_of_five_consecutive_integers_l31_31448

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l31_31448


namespace sandy_friday_hours_l31_31139

-- Define the conditions
def hourly_rate := 15
def saturday_hours := 6
def sunday_hours := 14
def total_earnings := 450

-- Define the proof problem
theorem sandy_friday_hours (F : ℝ) (h1 : F * hourly_rate + saturday_hours * hourly_rate + sunday_hours * hourly_rate = total_earnings) : F = 10 :=
sorry

end sandy_friday_hours_l31_31139


namespace b_minus_a_less_zero_l31_31342

-- Given conditions
variables {a b : ℝ}

-- Define the condition
def a_greater_b (a b : ℝ) : Prop := a > b

-- Lean 4 proof problem statement
theorem b_minus_a_less_zero (a b : ℝ) (h : a_greater_b a b) : b - a < 0 := 
sorry

end b_minus_a_less_zero_l31_31342


namespace polynomial_strictly_monotone_l31_31394

def strictly_monotone (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem polynomial_strictly_monotone
  (P : ℝ → ℝ)
  (H1 : strictly_monotone (P ∘ P))
  (H2 : strictly_monotone (P ∘ P ∘ P)) :
  strictly_monotone P :=
sorry

end polynomial_strictly_monotone_l31_31394


namespace rowing_speed_in_still_water_l31_31775

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.4) (t : ℝ)
  (h2 : (v + c) * t = (v - c) * (2 * t)) : 
  v = 4.2 :=
by
  sorry

end rowing_speed_in_still_water_l31_31775


namespace questions_for_second_project_l31_31272

open Nat

theorem questions_for_second_project (days_per_week : ℕ) (first_project_q : ℕ) (questions_per_day : ℕ) 
  (total_questions : ℕ) (second_project_q : ℕ) 
  (h1 : days_per_week = 7)
  (h2 : first_project_q = 518)
  (h3 : questions_per_day = 142)
  (h4 : total_questions = days_per_week * questions_per_day)
  (h5 : second_project_q = total_questions - first_project_q) :
  second_project_q = 476 :=
by
  -- we assume the solution steps as correct
  sorry

end questions_for_second_project_l31_31272


namespace right_triangle_side_81_exists_arithmetic_progression_l31_31007

theorem right_triangle_side_81_exists_arithmetic_progression :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a - d)^2 + a^2 = (a + d)^2 ∧ (3*d = 81 ∨ 4*d = 81 ∨ 5*d = 81) :=
sorry

end right_triangle_side_81_exists_arithmetic_progression_l31_31007


namespace proof_problem_l31_31113

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {2, 3, 4}

-- The complement of B with respect to U
def complement_U_B : Set ℕ := U \ B

-- The intersection of A and the complement of B with respect to U
def intersection_A_complement_U_B : Set ℕ := A ∩ complement_U_B

-- The statement we want to prove
theorem proof_problem : intersection_A_complement_U_B = {1, 6} :=
by
  sorry

end proof_problem_l31_31113


namespace three_digit_even_two_odd_no_repetition_l31_31936

-- Define sets of digits
def digits : List ℕ := [0, 1, 3, 4, 5, 6]
def evens : List ℕ := [0, 4, 6]
def odds : List ℕ := [1, 3, 5]

noncomputable def total_valid_numbers : ℕ :=
  let choose_0 := 12 -- Given by A_{2}^{1} A_{3}^{2} = 12
  let without_0 := 36 -- Given by C_{2}^{1} * C_{3}^{2} * A_{3}^{3} = 36
  choose_0 + without_0

theorem three_digit_even_two_odd_no_repetition : total_valid_numbers = 48 :=
by
  -- Proof would be provided here
  sorry

end three_digit_even_two_odd_no_repetition_l31_31936


namespace max_acute_triangles_l31_31361

theorem max_acute_triangles (n : ℕ) (hn : n ≥ 3) :
  (∃ k, k = if n % 2 = 0 then (n * (n-2) * (n+2)) / 24 else (n * (n-1) * (n+1)) / 24) :=
by 
  sorry

end max_acute_triangles_l31_31361


namespace positivity_of_fraction_l31_31086

theorem positivity_of_fraction
  (a b c d x1 x2 x3 x4 : ℝ)
  (h_neg_a : a < 0)
  (h_neg_b : b < 0)
  (h_neg_c : c < 0)
  (h_neg_d : d < 0)
  (h_abs : |x1 - a| + |x2 + b| + |x3 - c| + |x4 + d| = 0) :
  (x1 * x2 / (x3 * x4) > 0) := by
  sorry

end positivity_of_fraction_l31_31086


namespace smallest_number_of_sparrows_in_each_flock_l31_31522

theorem smallest_number_of_sparrows_in_each_flock (P : ℕ) (H : 14 * P ≥ 182) : 
  ∃ S : ℕ, S = 14 ∧ S ∣ 182 ∧ (∃ P : ℕ, S ∣ (14 * P)) := 
by 
  sorry

end smallest_number_of_sparrows_in_each_flock_l31_31522


namespace cost_price_to_selling_price_ratio_l31_31752

variable (CP SP : ℝ)
variable (profit_percent : ℝ)

theorem cost_price_to_selling_price_ratio
  (h1 : profit_percent = 0.25)
  (h2 : SP = (1 + profit_percent) * CP) :
  (CP / SP) = 4 / 5 := by
  sorry

end cost_price_to_selling_price_ratio_l31_31752


namespace Johnson_potatoes_left_l31_31580

noncomputable def Gina_potatoes : ℝ := 93.5
noncomputable def Tom_potatoes : ℝ := 3.2 * Gina_potatoes
noncomputable def Anne_potatoes : ℝ := (2/3) * Tom_potatoes
noncomputable def Jack_potatoes : ℝ := (1/7) * (Gina_potatoes + Anne_potatoes)
noncomputable def Total_given_away : ℝ := Gina_potatoes + Tom_potatoes + Anne_potatoes + Jack_potatoes
noncomputable def Initial_potatoes : ℝ := 1250
noncomputable def Potatoes_left : ℝ := Initial_potatoes - Total_given_away

theorem Johnson_potatoes_left : Potatoes_left = 615.98 := 
  by
    sorry

end Johnson_potatoes_left_l31_31580


namespace speed_second_half_l31_31593

theorem speed_second_half (H : ℝ) (S1 S2 : ℝ) (T : ℝ) : T = 11 → S1 = 30 → S1 * T1 = 150 → S1 * T1 + S2 * T2 = 300 → S2 = 25 :=
by
  intro hT hS1 hD1 hTotal
  sorry

end speed_second_half_l31_31593


namespace periodic_decimal_to_fraction_l31_31664

theorem periodic_decimal_to_fraction : (0.7 + 0.32 : ℝ) == (1013 / 990 : ℝ) := by
  sorry

end periodic_decimal_to_fraction_l31_31664


namespace prob_draw_l31_31827

theorem prob_draw (p_not_losing p_winning p_drawing : ℝ) (h1 : p_not_losing = 0.6) (h2 : p_winning = 0.5) :
  p_drawing = 0.1 :=
by
  sorry

end prob_draw_l31_31827


namespace work_completion_l31_31024

theorem work_completion (days_A : ℕ) (days_B : ℕ) (hA : days_A = 14) (hB : days_B = 35) :
  let rate_A := 1 / (days_A : ℚ)
  let rate_B := 1 / (days_B : ℚ)
  let combined_rate := rate_A + rate_B
  let days_AB := 1 / combined_rate
  days_AB = 10 := by
  sorry

end work_completion_l31_31024


namespace minimum_value_of_quadratic_l31_31867

theorem minimum_value_of_quadratic (x : ℝ) : ∃ (y : ℝ), (∀ x : ℝ, y ≤ x^2 + 2) ∧ (y = 2) :=
by
  sorry

end minimum_value_of_quadratic_l31_31867


namespace log_base_5_of_inv_sqrt_5_l31_31067

theorem log_base_5_of_inv_sqrt_5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
sorry

end log_base_5_of_inv_sqrt_5_l31_31067


namespace M_inter_N_l31_31816

def M : Set ℝ := {x | abs (x - 1) < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

theorem M_inter_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 3} :=
by
  sorry

end M_inter_N_l31_31816


namespace symmetry_of_transformed_graphs_l31_31670

noncomputable def y_eq_f_x_symmetric_line (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ), f (x - 19) = f (99 - x) ↔ x = 59

theorem symmetry_of_transformed_graphs (f : ℝ → ℝ) :
  y_eq_f_x_symmetric_line f :=
by {
  sorry
}

end symmetry_of_transformed_graphs_l31_31670


namespace intersection_A_B_l31_31841

open Set

def A : Set ℕ := {x | -2 < (x : ℤ) ∧ (x : ℤ) < 2}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ {x : ℕ | (x : ℤ) ∈ B} = {0, 1} := by
  sorry

end intersection_A_B_l31_31841


namespace log_base2_probability_l31_31012

theorem log_base2_probability (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : ∃ k : ℕ, n = 2^k) : 
  ∃ p : ℚ, p = 1/300 :=
  sorry

end log_base2_probability_l31_31012


namespace arithmetic_sequence_eighth_term_l31_31168

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l31_31168


namespace compute_expression_l31_31792

theorem compute_expression : 
  let x := 19
  let y := 15
  (x + y)^2 - (x - y)^2 = 1140 :=
by
  sorry

end compute_expression_l31_31792


namespace largest_root_in_range_l31_31656

-- Define the conditions for the equation parameters
variables (a0 a1 a2 : ℝ)
-- Define the conditions for the absolute value constraints
variables (h0 : |a0| < 2) (h1 : |a1| < 2) (h2 : |a2| < 2)

-- Define the equation
def cubic_equation (x : ℝ) : ℝ := x^3 + a2 * x^2 + a1 * x + a0

-- Define the property we want to prove about the largest positive root r
theorem largest_root_in_range :
  ∃ r > 0, (∃ x, cubic_equation a0 a1 a2 x = 0 ∧ r = x) ∧ (5 / 2 < r ∧ r < 3) :=
by sorry

end largest_root_in_range_l31_31656


namespace eliot_account_balance_l31_31261

theorem eliot_account_balance 
  (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A = 1.20 * E + 20) : 
  E = 200 :=
by 
  sorry

end eliot_account_balance_l31_31261


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31458

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l31_31458


namespace sum_a5_a6_a7_l31_31819

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = q * a n

variables (a : ℕ → ℤ)
variables (h_geo : geometric_sequence a)
variables (h1 : a 2 + a 3 = 1)
variables (h2 : a 3 + a 4 = -2)

theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 24 :=
by
  sorry

end sum_a5_a6_a7_l31_31819


namespace max_divisor_of_five_consecutive_integers_l31_31451

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l31_31451


namespace notebook_problem_l31_31975

theorem notebook_problem :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 2 * x + 5 * y + 6 * z = 62 ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x = 14 :=
by
  sorry

end notebook_problem_l31_31975


namespace largest_angle_of_isosceles_obtuse_30_deg_l31_31876

def is_isosceles (T : Triangle) : Prop :=
  T.A = T.B ∨ T.B = T.C ∨ T.C = T.A

def is_obtuse (T : Triangle) : Prop :=
  T.A > 90 ∨ T.B > 90 ∨ T.C > 90

def T : Type := {P Q R : ℝ}

noncomputable def largest_angle (T : Triangle) : ℝ :=
  max T.A (max T.B T.C)

theorem largest_angle_of_isosceles_obtuse_30_deg :
  ∀ (T : Triangle), is_isosceles T → is_obtuse T → T.A = 30 → largest_angle T = 120 :=
by
  intro T h_iso h_obt h_A30
  sorry

end largest_angle_of_isosceles_obtuse_30_deg_l31_31876


namespace arithmetic_seq_8th_term_l31_31141

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l31_31141


namespace area_bounded_by_parabola_and_x_axis_l31_31790

/-- Define the parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x - x^2

/-- The function for the x-axis -/
def x_axis : ℝ := 0

/-- Prove that the area bounded by the parabola and x-axis between x = 0 and x = 2 is 4/3 -/
theorem area_bounded_by_parabola_and_x_axis : 
  (∫ x in (0 : ℝ)..(2 : ℝ), parabola x) = 4 / 3 := by
    sorry

end area_bounded_by_parabola_and_x_axis_l31_31790


namespace fraction_to_decimal_representation_l31_31802

/-- Determine the decimal representation of a given fraction. -/
theorem fraction_to_decimal_representation : (45 / (2 ^ 3 * 5 ^ 4) = 0.0090) :=
sorry

end fraction_to_decimal_representation_l31_31802


namespace monochromatic_triangle_in_K9_l31_31550

/-- 
Theorem: In a complete graph \( K_9 \) (with 9 vertices) where each pair of vertices 
is connected by an edge, if at least 33 edges are colored either red or blue, 
then there must exist a monochromatic triangle (a triangle with all three edges the same color).
-/
theorem monochromatic_triangle_in_K9 
    (K9 : SimpleGraph (Fin 9)) : 
  ∀ (E : Finset (Sym2 (Fin 9))), E.card ≥ 33 → 
  (∀ (coloring : Sym2 (Fin 9) → Prop), 
    ∃ (triangle : Finset (Sym2 (Fin 9))), triangle.card = 3 ∧ 
    triangle ⊆ E ∧ 
    (∀ e ∈ triangle, coloring e) ∨ 
    (∀ e ∈ triangle, ¬coloring e)) :=
begin
  sorry
end

end monochromatic_triangle_in_K9_l31_31550


namespace set_difference_equals_six_l31_31303

-- Set Operations definitions used
def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Define sets M and N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 6}

-- Problem statement to prove
theorem set_difference_equals_six : set_difference N M = {6} :=
  sorry

end set_difference_equals_six_l31_31303


namespace find_point_C_on_z_axis_l31_31705

noncomputable def point_c_condition (C : ℝ × ℝ × ℝ) (A B : ℝ × ℝ × ℝ) : Prop :=
  dist C A = dist C B

theorem find_point_C_on_z_axis :
  ∃ C : ℝ × ℝ × ℝ, C = (0, 0, 1) ∧ point_c_condition C (1, 0, 2) (1, 1, 1) :=
by
  use (0, 0, 1)
  simp [point_c_condition]
  sorry

end find_point_C_on_z_axis_l31_31705


namespace problem_1_minimum_value_problem_2_range_of_a_l31_31944

noncomputable def e : ℝ := Real.exp 1  -- Definition of e as exp(1)

-- Question I:
-- Prove that the minimum value of the function f(x) = e^x - e*x - e is -e.
theorem problem_1_minimum_value :
  ∃ x : ℝ, (∀ y : ℝ, (Real.exp x - e * x - e) ≤ (Real.exp y - e * y - e))
  ∧ (Real.exp x - e * x - e) = -e := 
sorry

-- Question II:
-- Prove that the range of values for a such that f(x) = e^x - a*x - a >= 0 for all x is [0, 1].
theorem problem_2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, (Real.exp x - a * x - a) ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

end problem_1_minimum_value_problem_2_range_of_a_l31_31944


namespace arithmetic_sequence_sum_l31_31329

variable {a : ℕ → ℝ}

noncomputable def sum_of_first_ten_terms (a : ℕ → ℝ) : ℝ :=
  (10 / 2) * (a 1 + a 10)

theorem arithmetic_sequence_sum (h : a 5 + a 6 = 28) :
  sum_of_first_ten_terms a = 140 :=
by
  sorry

end arithmetic_sequence_sum_l31_31329


namespace initial_units_of_phones_l31_31509

theorem initial_units_of_phones
  (X : ℕ) 
  (h1 : 5 = 5) 
  (h2 : X - 5 = 3 + 5 + 7) : 
  X = 20 := 
by
  sorry

end initial_units_of_phones_l31_31509


namespace largest_decimal_of_4bit_binary_l31_31246

-- Define the maximum 4-bit binary number and its interpretation in base 10
def max_4bit_binary_value : ℕ := 2^4 - 1

-- The theorem to prove the statement
theorem largest_decimal_of_4bit_binary : max_4bit_binary_value = 15 :=
by
  -- Lean tactics or explicitly writing out the solution steps can be used here.
  -- Skipping proof as instructed.
  sorry

end largest_decimal_of_4bit_binary_l31_31246


namespace inverse_variation_example_l31_31376

theorem inverse_variation_example
  (k : ℝ)
  (h1 : ∀ (c d : ℝ), (c^2) * (d^4) = k)
  (h2 : ∃ (c : ℝ), c = 8 ∧ (∀ (d : ℝ), d = 2 → (c^2) * (d^4) = k)) : 
  (∀ (d : ℝ), d = 4 → (∃ (c : ℝ), (c^2) = 4)) := 
by 
  sorry

end inverse_variation_example_l31_31376


namespace percentage_increase_formula_l31_31998

theorem percentage_increase_formula (A B C : ℝ) (h1 : A = 3 * B) (h2 : C = B - 30) :
  100 * ((A - C) / C) = 200 + 9000 / C := 
by 
  sorry

end percentage_increase_formula_l31_31998


namespace sin_2017pi_over_6_l31_31546

theorem sin_2017pi_over_6 : Real.sin (2017 * Real.pi / 6) = 1 / 2 := 
by 
  -- Proof to be filled in later
  sorry

end sin_2017pi_over_6_l31_31546


namespace arithmetic_seq_8th_term_l31_31191

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l31_31191


namespace ellipse_foci_distance_l31_31511

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), a = 6 → b = 3 → distance_between_foci a b = 3 * Real.sqrt 3 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l31_31511


namespace cubic_identity_l31_31826

theorem cubic_identity (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : xy + xz + yz = 30) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 648 :=
sorry

end cubic_identity_l31_31826


namespace leon_total_payment_l31_31123

-- Define the constants based on the problem conditions
def cost_toy_organizer : ℝ := 78
def num_toy_organizers : ℝ := 3
def cost_gaming_chair : ℝ := 83
def num_gaming_chairs : ℝ := 2
def delivery_fee_rate : ℝ := 0.05

-- Calculate the cost for each category and the total cost
def total_cost_toy_organizers : ℝ := num_toy_organizers * cost_toy_organizer
def total_cost_gaming_chairs : ℝ := num_gaming_chairs * cost_gaming_chair
def total_sales : ℝ := total_cost_toy_organizers + total_cost_gaming_chairs
def delivery_fee : ℝ := delivery_fee_rate * total_sales
def total_amount_paid : ℝ := total_sales + delivery_fee

-- State the theorem for the total amount Leon has to pay
theorem leon_total_payment :
  total_amount_paid = 420 := by
  sorry

end leon_total_payment_l31_31123


namespace Q1_Q2_l31_31091

open Set

-- Definitions of sets A and B and the "length" of an interval
def setA (t : ℝ) : Set ℝ := {2, real.log2 t}
def setB : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}

-- Length of an interval
def interval_length (a b : ℝ) : ℝ := b - a

-- Question 1: Determine the value of t when the length of set A is 3
theorem Q1 (t : ℝ) (H : interval_length 2 (real.log2 t) = 3) : t = 32 := by
  sorry

-- Question 2: Determine the range of values of t such that A is a subset of B
theorem Q2 (t : ℝ) (H : setA t ⊆ setB) : 4 < t ∧ t < 32 := by
  sorry

end Q1_Q2_l31_31091


namespace min_m_n_sum_divisible_by_27_l31_31830

theorem min_m_n_sum_divisible_by_27 (m n : ℕ) (h : 180 * m * (n - 2) % 27 = 0) : m + n = 6 :=
sorry

end min_m_n_sum_divisible_by_27_l31_31830


namespace number_of_toys_sold_l31_31271

theorem number_of_toys_sold (n : ℕ) 
  (sell_price : ℕ) (gain_price : ℕ) (cost_price_per_toy : ℕ) :
  sell_price = 27300 → 
  gain_price = 3 * cost_price_per_toy → 
  cost_price_per_toy = 1300 →
  n * cost_price_per_toy + gain_price = sell_price → 
  n = 18 :=
by sorry

end number_of_toys_sold_l31_31271


namespace difference_eq_neg_subtrahend_implies_minuend_zero_l31_31103

theorem difference_eq_neg_subtrahend_implies_minuend_zero {x y : ℝ} (h : x - y = -y) : x = 0 :=
sorry

end difference_eq_neg_subtrahend_implies_minuend_zero_l31_31103


namespace number_of_bottle_caps_l31_31979

-- Condition: 7 bottle caps weigh exactly one ounce
def weight_of_7_caps : ℕ := 1 -- ounce

-- Condition: Josh's collection weighs 18 pounds, and 1 pound = 16 ounces
def weight_of_collection_pounds : ℕ := 18 -- pounds
def weight_of_pound_in_ounces : ℕ := 16 -- ounces per pound

-- Condition: Question translated to proof statement
theorem number_of_bottle_caps :
  (weight_of_collection_pounds * weight_of_pound_in_ounces * 7 = 2016) :=
by
  sorry

end number_of_bottle_caps_l31_31979


namespace area_of_rectangular_park_l31_31748

theorem area_of_rectangular_park
  (l w : ℕ) 
  (h_perimeter : 2 * l + 2 * w = 80)
  (h_length : l = 3 * w) :
  l * w = 300 :=
sorry

end area_of_rectangular_park_l31_31748


namespace set_intersection_example_l31_31560

def universal_set := Set ℝ

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}

def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x + 1 ∧ -2 ≤ x ∧ x ≤ 1}

def C : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

def complement (A : Set ℝ) : Set ℝ := {x : ℝ | x ∉ A}

def difference (A B : Set ℝ) : Set ℝ := A \ B

def union (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∨ x ∈ B}

def intersection (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection_example :
  intersection (complement A) (union B C) = {x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 4)} :=
by
  sorry

end set_intersection_example_l31_31560


namespace find_pairs_xy_l31_31542

theorem find_pairs_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : 7^x - 3 * 2^y = 1) : 
  (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
sorry

end find_pairs_xy_l31_31542


namespace initial_number_of_balls_l31_31404

theorem initial_number_of_balls (T B : ℕ) (P : ℚ) (after3_blue : ℕ) (prob : ℚ) :
  B = 7 → after3_blue = B - 3 → prob = after3_blue / T → prob = 1/3 → T = 15 :=
by
  sorry

end initial_number_of_balls_l31_31404


namespace max_value_of_expression_max_value_achieved_l31_31128

theorem max_value_of_expression (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
    8 * x + 3 * y + 10 * z ≤ Real.sqrt 173 :=
sorry

theorem max_value_achieved (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)
    (hx : x = Real.sqrt 173 / 30)
    (hy : y = Real.sqrt 173 / 20)
    (hz : z = Real.sqrt 173 / 50) :
    8 * x + 3 * y + 10 * z = Real.sqrt 173 :=
sorry

end max_value_of_expression_max_value_achieved_l31_31128


namespace tetrahedron_edge_length_of_tangent_spheres_l31_31934

theorem tetrahedron_edge_length_of_tangent_spheres (r : ℝ) (h₁ : r = 2) :
  ∃ s : ℝ, s = 4 :=
by
  sorry

end tetrahedron_edge_length_of_tangent_spheres_l31_31934


namespace score_difference_l31_31040

theorem score_difference 
  (x y z w : ℝ)
  (h1 : x = 2 + (y + z + w) / 3)
  (h2 : y = (x + z + w) / 3 - 3)
  (h3 : z = 3 + (x + y + w) / 3) :
  (x + y + z) / 3 - w = 2 :=
by {
  sorry
}

end score_difference_l31_31040


namespace escalator_times_comparison_l31_31036

variable (v v1 v2 l : ℝ)
variable (h_v_lt_v1 : v < v1)
variable (h_v1_lt_v2 : v1 < v2)

theorem escalator_times_comparison
  (h_cond : v < v1 ∧ v1 < v2) :
  (l / (v1 + v) + l / (v2 - v)) < (l / (v1 - v) + l / (v2 + v)) :=
  sorry

end escalator_times_comparison_l31_31036


namespace gain_percent_correct_l31_31764

theorem gain_percent_correct (C S : ℝ) (h : 50 * C = 28 * S) : 
  ( (S - C) / C ) * 100 = 1100 / 14 :=
by
  sorry

end gain_percent_correct_l31_31764


namespace find_k_l31_31630

variable (k : ℕ) (hk : k > 0)

theorem find_k (h : (24 - k) / (8 + k) = 1) : k = 8 :=
by sorry

end find_k_l31_31630


namespace product_of_five_consecutive_integers_divisible_by_120_l31_31422

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l31_31422


namespace arithmetic_sequence_8th_term_l31_31158

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l31_31158


namespace probability_six_greater_than_five_over_six_l31_31032

noncomputable def sumBeforeLastRoll (n : ℕ) (Y : ℕ → ℕ) : Prop :=
  Y (n - 1) + 6 >= 2019

noncomputable def probabilityLastRollSix (n : ℕ) (S : ℕ) : Prop :=
  S = 6

theorem probability_six_greater_than_five_over_six (n : ℕ) :
  ∀ (Y : ℕ → ℕ) (S : ℕ), sumBeforeLastRoll n Y →
  probabilityLastRollSix n S →
  (∑ k in range 1 7, probabilityLastRollSix (n - k)) > (5 / 6) :=
begin
  -- Proof would go here
  sorry
end

end probability_six_greater_than_five_over_six_l31_31032


namespace negation_of_proposition_l31_31606

open Classical

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l31_31606


namespace total_distance_travelled_l31_31015

-- Definitions and propositions
def distance_first_hour : ℝ := 15
def distance_second_hour : ℝ := 18
def distance_third_hour : ℝ := 1.25 * distance_second_hour

-- Conditions based on the problem
axiom second_hour_distance : distance_second_hour = 18
axiom second_hour_20_percent_more : distance_second_hour = 1.2 * distance_first_hour
axiom third_hour_25_percent_more : distance_third_hour = 1.25 * distance_second_hour

-- Proof of the total distance James traveled
theorem total_distance_travelled : 
  distance_first_hour + distance_second_hour + distance_third_hour = 55.5 :=
by
  sorry

end total_distance_travelled_l31_31015


namespace imo_1990_q31_l31_31017

def A (n : ℕ) : ℕ := sorry -- definition of A(n)
def B (n : ℕ) : ℕ := sorry -- definition of B(n)
def f (n : ℕ) : ℕ := if B n = 1 then 1 else -- largest prime factor of B(n)
  sorry -- logic to find the largest prime factor of B(n)

theorem imo_1990_q31 :
  ∃ (M : ℕ), (∀ n : ℕ, f n ≤ M) ∧ (∀ N, (∀ n, f n ≤ N) → M ≤ N) ∧ M = 1999 :=
by sorry

end imo_1990_q31_l31_31017


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31454

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ k : ℤ, k = 60 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 60
  split
  { refl }
  { sorry }

end largest_divisor_of_product_of_five_consecutive_integers_l31_31454


namespace number_of_elements_in_M_l31_31385

theorem number_of_elements_in_M :
  (∃! (M : Finset ℕ), M = {m | ∃ (n : ℕ), n > 0 ∧ m = 2*n - 1 ∧ m < 60 } ∧ M.card = 30) :=
sorry

end number_of_elements_in_M_l31_31385


namespace xyz_squared_sum_l31_31818

theorem xyz_squared_sum (x y z : ℤ) 
  (h1 : |x + y| + |y + z| + |z + x| = 4)
  (h2 : |x - y| + |y - z| + |z - x| = 2) :
  x^2 + y^2 + z^2 = 2 := 
by 
  sorry

end xyz_squared_sum_l31_31818


namespace cylinder_volume_transformation_l31_31898

-- Define the original volume of the cylinder
def original_volume (V: ℝ) := V = 5

-- Define the transformation of quadrupling the dimensions of the cylinder
def new_volume (V V': ℝ) := V' = 64 * V

-- The goal is to show that under these conditions, the new volume is 320 gallons
theorem cylinder_volume_transformation (V V': ℝ) (h: original_volume V) (h': new_volume V V'):
  V' = 320 :=
by
  -- Proof is left as an exercise
  sorry

end cylinder_volume_transformation_l31_31898


namespace determine_base_solution_l31_31056

theorem determine_base_solution :
  ∃ (h : ℕ), 
  h > 8 ∧ 
  (8 * h^3 + 6 * h^2 + 7 * h + 4) + (4 * h^3 + 3 * h^2 + 2 * h + 9) = 1 * h^4 + 3 * h^3 + 0 * h^2 + 0 * h + 3 ∧
  (9 + 4) = 13 ∧
  1 * h + 3 = 13 ∧
  (7 + 2 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (6 + 3 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (8 + 4 + 1) = 13 ∧
  1 * h + 3 = 13 ∧
  h = 10 :=
by
  sorry

end determine_base_solution_l31_31056


namespace S_2011_value_l31_31836

-- Definitions based on conditions provided in the problem
def arithmetic_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a_n (n + 1) = a_n n + d

def sum_seq (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Problem statement
theorem S_2011_value
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (h_arith : arithmetic_seq a_n)
  (h_sum : sum_seq S_n a_n)
  (h_init : a_n 1 = -2011)
  (h_cond : (S_n 2010) / 2010 - (S_n 2008) / 2008 = 2) :
  S_n 2011 = -2011 := 
sorry

end S_2011_value_l31_31836


namespace football_team_goal_l31_31266

-- Definitions of the conditions
def L1 : ℤ := -5
def G2 : ℤ := 13
def L3 : ℤ := -(L1 ^ 2)
def G4 : ℚ := - (L3 : ℚ) / 2

def total_yardage : ℚ := L1 + G2 + L3 + G4

-- The statement to be proved
theorem football_team_goal : total_yardage < 30 := by
  -- sorry for now since no proof is needed
  sorry

end football_team_goal_l31_31266


namespace shortest_distance_between_stations_l31_31399

/-- 
Given two vehicles A and B shuttling between two locations,
with Vehicle A stopping every 0.5 kilometers and Vehicle B stopping every 0.8 kilometers,
prove that the shortest distance between two stations where Vehicles A and B do not stop at the same place is 0.1 kilometers.
-/
theorem shortest_distance_between_stations :
  ∀ (dA dB : ℝ), (dA = 0.5) → (dB = 0.8) → ∃ δ : ℝ, (δ = 0.1) ∧ (∀ n m : ℕ, dA * n ≠ dB * m → abs ((dA * n) - (dB * m)) = δ) :=
by
  intros dA dB hA hB
  use 0.1
  sorry

end shortest_distance_between_stations_l31_31399


namespace option_d_satisfies_equation_l31_31484

theorem option_d_satisfies_equation (x y z : ℤ) (h1 : x = z) (h2 : y = x + 1) : x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  sorry

end option_d_satisfies_equation_l31_31484


namespace carly_dogs_total_l31_31527

theorem carly_dogs_total (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) (h2 : three_legged_dogs = 3) (h3 : nails_per_paw = 4) : total_dogs = 11 :=
by
  sorry

end carly_dogs_total_l31_31527


namespace ant_ways_to_reach_l31_31650

theorem ant_ways_to_reach : 
  let n := 4020
  let a := 2010
  let b := 1005
  (nat.choose n b)^2 = nat.choose n (n - b) * nat.choose n b := by
sorry

end ant_ways_to_reach_l31_31650


namespace intersect_complement_l31_31109

open Finset

-- Define the universal set U, set A, and set B
def U := {1, 2, 3, 4, 5, 6} : Finset ℕ
def A := {1, 3, 6} : Finset ℕ
def B := {2, 3, 4} : Finset ℕ

-- Define the complement of B in U
def complement_U_B := U \ B

-- The statement to prove
theorem intersect_complement : A ∩ complement_U_B = {1, 6} :=
by sorry

end intersect_complement_l31_31109


namespace product_of_5_consecutive_integers_divisible_by_60_l31_31461

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end product_of_5_consecutive_integers_divisible_by_60_l31_31461


namespace arithmetic_sequence_8th_term_l31_31161

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l31_31161


namespace valid_integer_lattice_points_count_l31_31891

def point := (ℤ × ℤ)
def A : point := (-4, 3)
def B : point := (4, -3)

def manhattan_distance (p1 p2 : point) : ℤ :=
  abs (p2.1 - p1.1) + abs (p2.2 - p1.2)

def valid_path_length (p1 p2 : point) : Prop :=
  manhattan_distance p1 p2 ≤ 18

def does_not_cross_y_eq_x (p1 p2 : point) : Prop :=
  ∀ x y, (x, y) ∈ [(p1, p2)] → y ≠ x

def integer_lattice_points_on_path (p1 p2 : point) : ℕ := sorry

theorem valid_integer_lattice_points_count :
  integer_lattice_points_on_path A B = 112 :=
sorry

end valid_integer_lattice_points_count_l31_31891


namespace group_size_l31_31537

noncomputable def total_cost : ℤ := 13500
noncomputable def cost_per_person : ℤ := 900

theorem group_size : total_cost / cost_per_person = 15 :=
by {
  sorry
}

end group_size_l31_31537


namespace five_consecutive_product_div_24_l31_31440

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l31_31440


namespace siblings_age_problem_l31_31378

variable {x y z : ℕ}

theorem siblings_age_problem
  (h1 : x - y = 3)
  (h2 : z - 1 = 2 * (x + y))
  (h3 : z + 20 = x + y + 40) :
  x = 11 ∧ y = 8 ∧ z = 39 :=
by
  sorry

end siblings_age_problem_l31_31378


namespace class_B_more_uniform_than_class_A_l31_31634

-- Definitions based on the given problem
def class_height_variance (class_name : String) : ℝ :=
  if class_name = "A" then 3.24 else if class_name = "B" then 1.63 else 0

-- The theorem statement proving that Class B has more uniform heights (smaller variance)
theorem class_B_more_uniform_than_class_A :
  class_height_variance "B" < class_height_variance "A" :=
by
  sorry

end class_B_more_uniform_than_class_A_l31_31634


namespace arithmetic_sequence_8th_term_l31_31151

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l31_31151


namespace find_intersection_l31_31077

noncomputable def f (n : ℕ) : ℕ := 2 * n + 1

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 4, 5, 6, 7}

def f_set (s : Set ℕ) : Set ℕ := {n | f n ∈ s}

theorem find_intersection : f_set A ∩ f_set B = {1, 2} := 
by {
  sorry
}

end find_intersection_l31_31077


namespace decagon_diagonals_l31_31912

-- The condition for the number of diagonals in a polygon
def number_of_diagonals (n : Nat) : Nat :=
  n * (n - 3) / 2

-- The specific proof statement for a decagon
theorem decagon_diagonals : number_of_diagonals 10 = 35 := by
  -- The proof would go here
  sorry

end decagon_diagonals_l31_31912


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31433

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l31_31433


namespace tank_ratio_l31_31039

theorem tank_ratio (V1 V2 : ℝ) (h1 : 0 < V1) (h2 : 0 < V2) (h1_full : 3 / 4 * V1 - 7 / 20 * V2 = 0) (h2_full : 1 / 4 * V2 + 7 / 20 * V2 = 3 / 5 * V2) :
  V1 / V2 = 7 / 9 :=
by
  sorry

end tank_ratio_l31_31039


namespace part_a_part_b_l31_31919

theorem part_a (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^a - 1)) :=
sorry

theorem part_b (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^(a + 1) - 1)) :=
sorry

end part_a_part_b_l31_31919


namespace value_of_expression_l31_31084

theorem value_of_expression (x y : ℝ) (h1 : 4 * x + y = 20) (h2 : x + 4 * y = 16) : 
  17 * x ^ 2 + 20 * x * y + 17 * y ^ 2 = 656 :=
sorry

end value_of_expression_l31_31084


namespace animal_eyes_count_l31_31352

noncomputable def total_animal_eyes (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ) : ℕ :=
frogs * eyes_per_frog + crocodiles * eyes_per_crocodile

theorem animal_eyes_count (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ):
  frogs = 20 → crocodiles = 10 → eyes_per_frog = 2 → eyes_per_crocodile = 2 → total_animal_eyes frogs crocodiles eyes_per_frog eyes_per_crocodile = 60 :=
by
  sorry

end animal_eyes_count_l31_31352


namespace necessary_but_not_sufficient_condition_l31_31019

variable (a b : ℝ)

theorem necessary_but_not_sufficient_condition : (a > b) → ((a > b) ↔ ((a - b) * b^2 > 0)) :=
sorry

end necessary_but_not_sufficient_condition_l31_31019


namespace remainder_division_l31_31776

/-- A number when divided by a certain divisor left a remainder, 
when twice the number was divided by the same divisor, the remainder was 112. 
The divisor is 398.
Prove that the remainder when the original number is divided by the divisor is 56. -/
theorem remainder_division (N R : ℤ) (D : ℕ) (Q Q' : ℤ)
  (hD : D = 398)
  (h1 : N = D * Q + R)
  (h2 : 2 * N = D * Q' + 112) :
  R = 56 :=
sorry

end remainder_division_l31_31776


namespace absolute_value_condition_l31_31962

theorem absolute_value_condition (x : ℝ) (h : |x| = 32) : x = 32 ∨ x = -32 :=
sorry

end absolute_value_condition_l31_31962


namespace height_pillar_D_correct_l31_31512

def height_of_pillar_at_D (h_A h_B h_C : ℕ) (side_length : ℕ) : ℕ :=
17

theorem height_pillar_D_correct :
  height_of_pillar_at_D 15 10 12 10 = 17 := 
by sorry

end height_pillar_D_correct_l31_31512


namespace allan_balloons_l31_31782

theorem allan_balloons (x : ℕ) : 
  (2 + x) + 1 = 6 → x = 3 :=
by
  intro h
  linarith

end allan_balloons_l31_31782


namespace simplified_expression_term_count_l31_31743

def even_exponents_terms_count : ℕ :=
  let n := 2008
  let k := 1004
  Nat.choose (k + 2) 2

theorem simplified_expression_term_count :
  even_exponents_terms_count = 505815 :=
sorry

end simplified_expression_term_count_l31_31743


namespace arithmetic_sequence_8th_term_l31_31155

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l31_31155


namespace product_of_pairs_l31_31008

theorem product_of_pairs (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2015)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2014)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2015)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2014)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2015)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2014):
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -4 / 1007 :=
sorry

end product_of_pairs_l31_31008


namespace cos_seven_pi_over_six_l31_31071

theorem cos_seven_pi_over_six :
  Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
sorry

end cos_seven_pi_over_six_l31_31071


namespace probability_six_on_final_roll_l31_31031

theorem probability_six_on_final_roll (n : ℕ) (h : n ≥ 2019) :
  (∃ p : ℚ, p > 5 / 6 ∧ 
  (∀ roll : ℕ, roll <= n → roll mod 6 = 0 → roll / n > p)) :=
sorry

end probability_six_on_final_roll_l31_31031


namespace sin_B_sin_C_l31_31707

open Real

noncomputable def triangle_condition (A B C : ℝ) (a b c : ℝ) : Prop :=
  cos (2 * A) - 3 * cos (B + C) = 1 ∧
  (1 / 2) * b * c * sin A = 5 * sqrt 3 ∧
  b = 5

theorem sin_B_sin_C {A B C a b c : ℝ} (h : triangle_condition A B C a b c) :
  (sin B) * (sin C) = 5 / 7 := 
sorry

end sin_B_sin_C_l31_31707


namespace baron_munchausen_incorrect_l31_31909

theorem baron_munchausen_incorrect : 
  ∀ (n : ℕ) (ab : ℕ), 10 ≤ n → n ≤ 99 → 0 ≤ ab → ab ≤ 99 
  → ¬ (∃ (m : ℕ), n * 100 + ab = m * m) := 
by
  intros n ab n_lower_bound n_upper_bound ab_lower_bound ab_upper_bound
  sorry

end baron_munchausen_incorrect_l31_31909


namespace largest_divisor_of_five_consecutive_integers_l31_31426

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l31_31426


namespace quadratic_form_rewrite_l31_31370

theorem quadratic_form_rewrite (x : ℝ) : 2 * x ^ 2 + 7 = 4 * x → 2 * x ^ 2 - 4 * x + 7 = 0 :=
by
    intro h
    linarith

end quadratic_form_rewrite_l31_31370


namespace rectangle_area_l31_31219

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 250) : l * w = 2500 :=
  sorry

end rectangle_area_l31_31219


namespace manuscript_fee_tax_l31_31222

theorem manuscript_fee_tax (fee : ℕ) (tax_paid : ℕ) :
  (tax_paid = 0 ∧ fee ≤ 800) ∨ 
  (tax_paid = (14 * (fee - 800) / 100) ∧ 800 < fee ∧ fee ≤ 4000) ∨ 
  (tax_paid = 11 * fee / 100 ∧ fee > 4000) →
  tax_paid = 420 →
  fee = 3800 :=
by 
  intro h_eq h_tax;
  sorry

end manuscript_fee_tax_l31_31222


namespace annual_interest_rate_is_correct_l31_31753

-- Definitions of the conditions
def true_discount : ℚ := 210
def bill_amount : ℚ := 1960
def time_period_years : ℚ := 3 / 4

-- The present value of the bill
def present_value : ℚ := bill_amount - true_discount

-- The formula for simple interest given principal, rate, and time
def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T / 100

-- Proof statement
theorem annual_interest_rate_is_correct : 
  ∃ (R : ℚ), simple_interest present_value R time_period_years = true_discount ∧ R = 16 :=
by
  use 16
  sorry

end annual_interest_rate_is_correct_l31_31753


namespace polynomial_representation_l31_31256

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 6)

theorem polynomial_representation (x : ℝ) :
  given_expression x = 6 * x^3 - 4 * x^2 - 26 * x + 20 :=
sorry

end polynomial_representation_l31_31256


namespace polygon_diagonals_formula_l31_31037

theorem polygon_diagonals_formula (n : ℕ) (h₁ : n = 5) (h₂ : 2 * n = (n * (n - 3)) / 2) :
  ∃ D : ℕ, D = n * (n - 3) / 2 :=
by
  sorry

end polygon_diagonals_formula_l31_31037


namespace solve_for_N_l31_31005

theorem solve_for_N (a b c N : ℝ) 
  (h1 : a + b + c = 72) 
  (h2 : a - 7 = N) 
  (h3 : b + 7 = N) 
  (h4 : 2 * c = N) : 
  N = 28.8 := 
sorry

end solve_for_N_l31_31005


namespace revenue_change_l31_31006

theorem revenue_change (T C : ℝ) (T_new C_new : ℝ)
  (h1 : T_new = 0.81 * T)
  (h2 : C_new = 1.15 * C)
  (R : ℝ := T * C) : 
  ((T_new * C_new - R) / R) * 100 = -6.85 :=
by
  sorry

end revenue_change_l31_31006


namespace question_1_question_2_l31_31982

def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem question_1 (m : ℝ) :
  (∀ x : ℝ, f x ≤ -m^2 + 6 * m) ↔ (1 ≤ m ∧ m ≤ 5) :=
by
  sorry

theorem question_2 (a b c : ℝ) (h1 : 3 * a + 4 * b + 5 * c = 5) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 ≥ 1 / 2 :=
by
  sorry

end question_1_question_2_l31_31982


namespace x_squared_plus_y_squared_l31_31691

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 :=
by
  sorry

end x_squared_plus_y_squared_l31_31691


namespace line_equation_l31_31381

theorem line_equation (m : ℝ) (x1 y1 : ℝ) (b : ℝ) :
  m = -3 → x1 = -2 → y1 = 0 → 
  (∀ x y, y - y1 = m * (x - x1) ↔ 3 * x + y + 6 = 0) :=
sorry

end line_equation_l31_31381


namespace josh_bottle_caps_l31_31976

/--
Suppose:
1. 7 bottle caps weigh exactly one ounce.
2. Josh's entire bottle cap collection weighs 18 pounds exactly.
3. There are 16 ounces in 1 pound.
We aim to show that Josh has 2016 bottle caps in his collection.
-/
theorem josh_bottle_caps :
  (7 : ℕ) * (1 : ℕ) = (7 : ℕ) → 
  (18 : ℕ) * (16 : ℕ) = (288 : ℕ) →
  (288 : ℕ) * (7 : ℕ) = (2016 : ℕ) :=
by
  intros h1 h2;
  exact sorry

end josh_bottle_caps_l31_31976


namespace Debby_bought_bottles_l31_31914

theorem Debby_bought_bottles :
  (5 : ℕ) * (71 : ℕ) = 355 :=
by
  -- Math proof goes here
  sorry

end Debby_bought_bottles_l31_31914


namespace five_more_than_three_in_pages_l31_31660

def pages := (List.range 512).map (λ n => n + 1)

def count_digit (d : Nat) (n : Nat) : Nat :=
  if n = 0 then 0
  else if n % 10 = d then 1 + count_digit d (n / 10)
  else count_digit d (n / 10)

def total_digit_count (d : Nat) (l : List Nat) : Nat :=
  l.foldl (λ acc x => acc + count_digit d x) 0

theorem five_more_than_three_in_pages :
  total_digit_count 5 pages - total_digit_count 3 pages = 22 := 
by 
  sorry

end five_more_than_three_in_pages_l31_31660


namespace factor_equivalence_l31_31654

noncomputable def given_expression (x : ℝ) :=
  (3 * x^3 + 70 * x^2 - 5) - (-4 * x^3 + 2 * x^2 - 5)

noncomputable def target_form (x : ℝ) :=
  7 * x^2 * (x + 68 / 7)

theorem factor_equivalence (x : ℝ) : given_expression x = target_form x :=
by
  sorry

end factor_equivalence_l31_31654


namespace distance_between_stripes_l31_31282

/-- Given a crosswalk parallelogram with curbs 60 feet apart, a base of 20 feet, 
and each stripe of length 50 feet, show that the distance between the stripes is 24 feet. -/
theorem distance_between_stripes (h : Real) (b : Real) (s : Real) : h = 60 ∧ b = 20 ∧ s = 50 → (b * h) / s = 24 :=
by
  sorry

end distance_between_stripes_l31_31282


namespace not_divisible_1978_1000_l31_31994

theorem not_divisible_1978_1000 (m : ℕ) : ¬ ∃ m : ℕ, (1000^m - 1) ∣ (1978^m - 1) := sorry

end not_divisible_1978_1000_l31_31994


namespace max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l31_31483

noncomputable def max_perimeter_of_right_angled_quadrilateral (r : ℝ) : ℝ :=
  4 * r * Real.sqrt 2

theorem max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2
  (r : ℝ) :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 4 * r^2 → 2 * (x + y) ≤ max_perimeter_of_right_angled_quadrilateral r)
  ∧ (k = max_perimeter_of_right_angled_quadrilateral r) :=
sorry

end max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l31_31483


namespace problem1_problem2_l31_31937

noncomputable def part1 (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4} ∩ {x | x ≤ 2 * a - 5}
noncomputable def part2 (a : ℝ) : Prop := ∀ x : ℝ, (-2 ≤ x ∧ x ≤ 4) → (x ≤ 2 * a - 5)

theorem problem1 : part1 3 = {x | -2 ≤ x ∧ x ≤ 1} :=
by { sorry }

theorem problem2 : ∀ a : ℝ, (part2 a) ↔ (a ≥ 9/2) :=
by { sorry }

end problem1_problem2_l31_31937


namespace initial_students_per_class_l31_31859

theorem initial_students_per_class (students_per_class initial_classes additional_classes total_students : ℕ) 
  (h1 : initial_classes = 15) 
  (h2 : additional_classes = 5) 
  (h3 : total_students = 400) 
  (h4 : students_per_class * (initial_classes + additional_classes) = total_students) : 
  students_per_class = 20 := 
by 
  -- Proof goes here
  sorry

end initial_students_per_class_l31_31859


namespace no_three_digit_number_l31_31341

theorem no_three_digit_number (N : ℕ) : 
  (100 ≤ N ∧ N < 1000 ∧ 
   (∀ k, k ∈ [1,2,3] → 5 < (N / 10^(k - 1) % 10)) ∧ 
   (N % 6 = 0) ∧ (N % 5 = 0)) → 
  false :=
by
sorry

end no_three_digit_number_l31_31341


namespace carpet_needed_l31_31275

/-- A rectangular room with dimensions 15 feet by 9 feet has a non-carpeted area occupied by 
a table with dimensions 3 feet by 2 feet. We want to prove that the number of square yards 
of carpet needed to cover the rest of the floor is 15. -/
theorem carpet_needed
  (room_length : ℝ) (room_width : ℝ) (table_length : ℝ) (table_width : ℝ)
  (h_room : room_length = 15) (h_room_width : room_width = 9)
  (h_table : table_length = 3) (h_table_width : table_width = 2) : 
  (⌈(((room_length * room_width) - (table_length * table_width)) / 9 : ℝ)⌉ = 15) := 
by
  sorry

end carpet_needed_l31_31275


namespace ratio_sheep_to_horses_is_correct_l31_31228

-- Definitions of given conditions
def ounces_per_horse := 230
def total_ounces_per_day := 12880
def number_of_sheep := 16

-- Express the number of horses and the ratio of sheep to horses
def number_of_horses : ℕ := total_ounces_per_day / ounces_per_horse
def ratio_sheep_to_horses := number_of_sheep / number_of_horses

-- The main statement to be proved
theorem ratio_sheep_to_horses_is_correct : ratio_sheep_to_horses = 2 / 7 :=
by
  sorry

end ratio_sheep_to_horses_is_correct_l31_31228


namespace num_dogs_correct_l31_31754

-- Definitions based on conditions
def total_animals : ℕ := 17
def number_of_cats : ℕ := 8

-- Definition based on required proof
def number_of_dogs : ℕ := total_animals - number_of_cats

-- Proof statement
theorem num_dogs_correct : number_of_dogs = 9 :=
by
  sorry

end num_dogs_correct_l31_31754


namespace Q_equals_10_04_l31_31410
-- Import Mathlib for mathematical operations and equivalence checking

-- Define the given conditions
def a := 6
def b := 3
def c := 2

-- Define the expression to be evaluated
def Q : ℚ := (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2)

-- Prove that the expression equals 10.04
theorem Q_equals_10_04 : Q = 10.04 := by
  -- Proof goes here
  sorry

end Q_equals_10_04_l31_31410


namespace subcommittees_with_at_least_one_coach_l31_31398

-- Definitions based on conditions
def total_members : ℕ := 12
def total_coaches : ℕ := 5
def subcommittee_size : ℕ := 5

-- Lean statement of the problem
theorem subcommittees_with_at_least_one_coach :
  (Nat.choose total_members subcommittee_size) - (Nat.choose (total_members - total_coaches) subcommittee_size) = 771 := by
  sorry

end subcommittees_with_at_least_one_coach_l31_31398


namespace quadratic_root_condition_l31_31102

theorem quadratic_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Ici 10 ∪ Set.Iic (-10) :=
by 
  sorry

end quadratic_root_condition_l31_31102


namespace sum_of_first_n_terms_l31_31080

-- Definitions for the sequences and the problem conditions.
def a (n : ℕ) : ℕ := 2 ^ n
def b (n : ℕ) : ℕ := 2 * n - 1
def c (n : ℕ) : ℕ := a n * b n
def T (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ (n + 1) + 6

-- The theorem statement
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range n).sum c = T n :=
  sorry

end sum_of_first_n_terms_l31_31080


namespace find_ax5_by5_l31_31690

variables (a b x y: ℝ)

theorem find_ax5_by5 (h1 : a * x + b * y = 5)
                      (h2 : a * x^2 + b * y^2 = 11)
                      (h3 : a * x^3 + b * y^3 = 24)
                      (h4 : a * x^4 + b * y^4 = 56) :
                      a * x^5 + b * y^5 = 180.36 :=
sorry

end find_ax5_by5_l31_31690


namespace correct_intersection_l31_31092

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem correct_intersection : M ∩ N = {2, 3} := by sorry

end correct_intersection_l31_31092


namespace value_of_coins_is_77_percent_l31_31241

theorem value_of_coins_is_77_percent :
  let pennies := 2 * 1  -- value of two pennies in cents
  let nickel := 5       -- value of one nickel in cents
  let dimes := 2 * 10   -- value of two dimes in cents
  let half_dollar := 50 -- value of one half-dollar in cents
  let total_cents := pennies + nickel + dimes + half_dollar
  let dollar_in_cents := 100
  (total_cents / dollar_in_cents) * 100 = 77 :=
by
  sorry

end value_of_coins_is_77_percent_l31_31241


namespace largest_divisor_of_five_consecutive_integers_l31_31431

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l31_31431


namespace men_wages_l31_31628

-- Conditions
variable (M W B : ℝ)
variable (h1 : 15 * M = W)
variable (h2 : W = 12 * B)
variable (h3 : 15 * M + W + B = 432)

-- Statement to prove
theorem men_wages : 15 * M = 144 :=
by
  sorry

end men_wages_l31_31628


namespace equation_is_hyperbola_l31_31795

theorem equation_is_hyperbola : 
  ∀ x y : ℝ, (x^2 - 25*y^2 - 10*x + 50 = 0) → 
  (∃ a b h k : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (x - h)^2 / a^2 - (y - k)^2 / b^2 = -1)) :=
by
  sorry

end equation_is_hyperbola_l31_31795


namespace solution_product_l31_31843

theorem solution_product (p q : ℝ) (hpq : p ≠ q) (h1 : (x-3)*(3*x+18) = x^2-15*x+54) (hp : (x - p) * (x - q) = x^2 - 12 * x + 54) :
  (p + 2) * (q + 2) = -80 := sorry

end solution_product_l31_31843


namespace cross_country_meet_winning_scores_l31_31117

theorem cross_country_meet_winning_scores :
  ∃ (scores : Finset ℕ), scores.card = 13 ∧
    ∀ s ∈ scores, s ≥ 15 ∧ s ≤ 27 :=
by
  sorry

end cross_country_meet_winning_scores_l31_31117


namespace carly_dogs_total_l31_31526

theorem carly_dogs_total (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) (h2 : three_legged_dogs = 3) (h3 : nails_per_paw = 4) : total_dogs = 11 :=
by
  sorry

end carly_dogs_total_l31_31526


namespace largest_divisor_of_five_consecutive_integers_l31_31428

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l31_31428


namespace files_rem_nat_eq_two_l31_31627

-- Conditions
def initial_music_files : ℕ := 4
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23

-- Correct Answer
def files_remaining : ℕ := initial_music_files + initial_video_files - files_deleted

theorem files_rem_nat_eq_two : files_remaining = 2 := by
  sorry

end files_rem_nat_eq_two_l31_31627


namespace standard_equation_of_ellipse_l31_31828

-- Definitions for clarity
def is_ellipse (E : Type) := true
def major_axis (e : is_ellipse E) : ℝ := sorry
def minor_axis (e : is_ellipse E) : ℝ := sorry
def focus (e : is_ellipse E) : ℝ := sorry

theorem standard_equation_of_ellipse (E : Type)
  (e : is_ellipse E)
  (major_sum : major_axis e + minor_axis e = 9)
  (focus_position : focus e = 3) :
  ∀ x y, (x^2 / 25) + (y^2 / 16) = 1 :=
by sorry

end standard_equation_of_ellipse_l31_31828


namespace collinear_points_l31_31562

variable (a : ℝ) (A B C : ℝ × ℝ)

-- Conditions given in the problem
def point_A := (a, 2 : ℝ)
def point_B := (5, 1 : ℝ)
def point_C := (-4, 2 * a : ℝ)

-- Collinearity condition
def collinear (x y z : ℝ): Prop :=
  (x.1 - y.1) * (y.2 - z.2) = (y.1 - z.1) * (x.2 - y.2)

theorem collinear_points :
  collinear (point_A a) (point_B) (point_C a) →
  a = 5 + sqrt 21 ∨ a = 5 - sqrt 21 :=
by {
  sorry,
}

end collinear_points_l31_31562


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31432

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l31_31432


namespace find_x_l31_31857

theorem find_x (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) 
  (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := 
sorry

end find_x_l31_31857


namespace correct_operation_l31_31902

variable (N : ℚ) -- Original number (assumed rational for simplicity)
variable (x : ℚ) -- Unknown multiplier

theorem correct_operation (h : (N / 10) = (5 / 100) * (N * x)) : x = 2 :=
by
  sorry

end correct_operation_l31_31902


namespace number_properties_l31_31838

def number : ℕ := 52300600

def position_of_2 : ℕ := 10^6

def value_of_2 : ℕ := 20000000

def position_of_5 : ℕ := 10^7

def value_of_5 : ℕ := 50000000

def read_number : String := "five hundred twenty-three million six hundred"

theorem number_properties : 
  position_of_2 = (10^6) ∧ value_of_2 = 20000000 ∧ 
  position_of_5 = (10^7) ∧ value_of_5 = 50000000 ∧ 
  read_number = "five hundred twenty-three million six hundred" :=
by sorry

end number_properties_l31_31838


namespace hall_length_width_difference_l31_31488

theorem hall_length_width_difference : 
  ∃ L W : ℝ, W = (1 / 2) * L ∧ L * W = 450 ∧ L - W = 15 :=
sorry

end hall_length_width_difference_l31_31488


namespace tangent_and_normal_at_t_eq_pi_div4_l31_31076

def tangent_line_equation (t: ℝ) := - (4 / 3) * t + 4 * Real.sqrt 2
def normal_line_equation (t: ℝ) := (3 / 4) * t + (7 * Real.sqrt 2) / 8

theorem tangent_and_normal_at_t_eq_pi_div4 :
  (tangent_line_equation (3 * Real.cos (Real.pi / 4)) = 4 * Real.sqrt 2) ∧
  (normal_line_equation (3 * Real.cos (Real.pi / 4)) = (7 * Real.sqrt 2) / 8) :=
by
  sorry

end tangent_and_normal_at_t_eq_pi_div4_l31_31076


namespace cos_triple_angle_l31_31101

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l31_31101


namespace div_neg_forty_five_l31_31913

theorem div_neg_forty_five : (-40 / 5) = -8 :=
by
  sorry

end div_neg_forty_five_l31_31913


namespace find_d_l31_31126

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3
def h (x : ℝ) (c : ℝ) (d : ℝ) : Prop := f (g x c) c = 15 * x + d

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l31_31126


namespace point_side_opposite_l31_31906

def equation_lhs (x y : ℝ) : ℝ := 2 * y - 6 * x + 1

theorem point_side_opposite : 
  (equation_lhs 0 0 * equation_lhs 2 1 < 0) := 
by 
   sorry

end point_side_opposite_l31_31906


namespace sum_of_three_integers_l31_31002

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 5^3) : a + b + c = 31 := by
  sorry

end sum_of_three_integers_l31_31002


namespace sum_of_fractions_l31_31053

theorem sum_of_fractions :
  (3 / 9) + (7 / 12) = (11 / 12) :=
by 
  sorry

end sum_of_fractions_l31_31053


namespace evaluate_fractions_l31_31064

theorem evaluate_fractions : (7 / 3 : ℚ) + (11 / 5) + (19 / 9) + (37 / 17) - 8 = 628 / 765 := by
  sorry

end evaluate_fractions_l31_31064


namespace handshakes_at_networking_event_l31_31786

noncomputable def total_handshakes (n : ℕ) (exclude : ℕ) : ℕ :=
  (n * (n - 1 - exclude)) / 2

theorem handshakes_at_networking_event : total_handshakes 12 1 = 60 := by
  sorry

end handshakes_at_networking_event_l31_31786


namespace arithmetic_seq_8th_term_l31_31143

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l31_31143


namespace product_of_five_consecutive_is_divisible_by_sixty_l31_31414

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l31_31414


namespace arithmetic_sequence_eighth_term_l31_31170

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l31_31170


namespace sum_of_vertices_l31_31812

theorem sum_of_vertices (rect_verts: Nat) (pent_verts: Nat) (h1: rect_verts = 4) (h2: pent_verts = 5) : rect_verts + pent_verts = 9 :=
by
  sorry

end sum_of_vertices_l31_31812


namespace pinedale_bus_speed_l31_31860

theorem pinedale_bus_speed 
  (stops_every_minutes : ℕ)
  (num_stops : ℕ)
  (distance_km : ℕ)
  (time_per_stop_minutes : stops_every_minutes = 5)
  (dest_stops : num_stops = 8)
  (dest_distance : distance_km = 40) 
  : (distance_km / (num_stops * stops_every_minutes / 60)) = 60 := 
by
  sorry

end pinedale_bus_speed_l31_31860


namespace tax_collected_from_village_l31_31666

-- Definitions according to the conditions in the problem
def MrWillamTax : ℝ := 500
def MrWillamPercentage : ℝ := 0.21701388888888893

-- The theorem to prove the total tax collected
theorem tax_collected_from_village : ∃ (total_collected : ℝ), MrWillamPercentage * total_collected = MrWillamTax ∧ total_collected = 2303.7037037037035 :=
sorry

end tax_collected_from_village_l31_31666


namespace log_base5_of_inverse_sqrt5_l31_31066

theorem log_base5_of_inverse_sqrt5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
begin
  sorry
end

end log_base5_of_inverse_sqrt5_l31_31066


namespace tetrahedron_coloring_l31_31837

noncomputable def count_distinct_tetrahedron_colorings : ℕ :=
  sorry

theorem tetrahedron_coloring :
  count_distinct_tetrahedron_colorings = 6 :=
  sorry

end tetrahedron_coloring_l31_31837


namespace work_completion_l31_31490

/-- 
  Let A, B, and C have work rates where:
  1. A completes the work in 4 days (work rate: 1/4 per day)
  2. C completes the work in 12 days (work rate: 1/12 per day)
  3. Together with B, they complete the work in 2 days (combined work rate: 1/2 per day)
  Prove that B alone can complete the work in 6 days.
--/
theorem work_completion (A B C : ℝ) (x : ℝ)
  (hA : A = 1/4)
  (hC : C = 1/12)
  (h_combined : A + 1/x + C = 1/2) :
  x = 6 := sorry

end work_completion_l31_31490


namespace number_of_birds_is_400_l31_31389

-- Definitions of the problem
def num_stones : ℕ := 40
def num_trees : ℕ := 3 * num_stones + num_stones
def combined_trees_stones : ℕ := num_trees + num_stones
def num_birds : ℕ := 2 * combined_trees_stones

-- Statement to prove
theorem number_of_birds_is_400 : num_birds = 400 := by
  sorry

end number_of_birds_is_400_l31_31389


namespace min_distance_eq_5_l31_31105

-- Define the conditions
def condition1 (a b : ℝ) : Prop := b = 4 * Real.log a - a^2
def condition2 (c d : ℝ) : Prop := d = 2 * c + 2

-- Define the function to prove the minimum value
def minValue (a b c d : ℝ) : ℝ := (a - c)^2 + (b - d)^2

-- The main theorem statement
theorem min_distance_eq_5 (a b c d : ℝ) (ha : a > 0) (h1: condition1 a b) (h2: condition2 c d) : 
  ∃ a c b d, minValue a b c d = 5 := 
sorry

end min_distance_eq_5_l31_31105


namespace ellipse_foci_y_axis_range_l31_31694

theorem ellipse_foci_y_axis_range (k : ℝ) :
  (∃ x y : ℝ, x^2 + k * y^2 = 4 ∧ (∃ c1 c2 : ℝ, y = 0 → c1^2 + c2^2 = 4)) ↔ 0 < k ∧ k < 1 :=
by
  sorry

end ellipse_foci_y_axis_range_l31_31694


namespace problem1_problem2_l31_31294

theorem problem1 : 1 - 2 + 3 + (-4) = -2 :=
sorry

theorem problem2 : (-6) / 3 - (-10) - abs (-8) = 0 :=
sorry

end problem1_problem2_l31_31294


namespace range_of_k_l31_31676

noncomputable def triangle_range (A B C : ℝ) (a b c k : ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (B = Real.pi / 3) ∧       -- From arithmetic sequence and solving for B
  a^2 + c^2 = k * b^2 ∧
  (1 < k ∧ k <= 2)

theorem range_of_k (A B C a b c k : ℝ) :
  A + B + C = Real.pi →
  (B = Real.pi - (A + C)) →
  (B = Real.pi / 3) →
  a^2 + c^2 = k * b^2 →
  0 < A ∧ A < 2*Real.pi/3 →
  1 < k ∧ k <= 2 :=
by
  sorry

end range_of_k_l31_31676


namespace minimum_apples_l31_31631

theorem minimum_apples (n : ℕ) : 
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 9 = 7 → n = 97 := 
by 
  -- To be proved
  sorry

end minimum_apples_l31_31631


namespace brandon_businesses_l31_31288

theorem brandon_businesses (total_businesses: ℕ) (fire_fraction: ℚ) (quit_fraction: ℚ) 
  (h_total: total_businesses = 72) 
  (h_fire_fraction: fire_fraction = 1/2) 
  (h_quit_fraction: quit_fraction = 1/3) : 
  total_businesses - (total_businesses * fire_fraction + total_businesses * quit_fraction) = 12 :=
by 
  sorry

end brandon_businesses_l31_31288


namespace rate_grapes_l31_31292

/-- Given that Bruce purchased 8 kg of grapes at a rate G per kg, 8 kg of mangoes at the rate of 55 per kg, 
and paid a total of 1000 to the shopkeeper, prove that the rate per kg for the grapes (G) is 70. -/
theorem rate_grapes (G : ℝ) (h1 : 8 * G + 8 * 55 = 1000) : G = 70 :=
by 
  sorry

end rate_grapes_l31_31292


namespace speed_of_second_part_of_trip_l31_31029

-- Given conditions
def total_distance : Real := 50
def first_part_distance : Real := 25
def first_part_speed : Real := 66
def average_speed : Real := 44.00000000000001

-- The statement we want to prove
theorem speed_of_second_part_of_trip :
  ∃ second_part_speed : Real, second_part_speed = 33 :=
by
  sorry

end speed_of_second_part_of_trip_l31_31029


namespace trapezoid_area_l31_31972

open Real

theorem trapezoid_area 
  (r : ℝ) (BM CD AB : ℝ) (radius_nonneg : 0 ≤ r) 
  (BM_positive : 0 < BM) (CD_positive : 0 < CD) (AB_positive : 0 < AB)
  (circle_radius : r = 4) (BM_length : BM = 16) (CD_length : CD = 3) :
  let height := 2 * r
  let base_sum := AB + CD
  let area := height * base_sum / 2
  AB = BM + 8 → area = 108 :=
by
  intro hyp
  sorry

end trapezoid_area_l31_31972


namespace union_dues_proof_l31_31408

noncomputable def h : ℕ := 42
noncomputable def r : ℕ := 10
noncomputable def tax_rate : ℝ := 0.20
noncomputable def insurance_rate : ℝ := 0.05
noncomputable def take_home_pay : ℝ := 310

noncomputable def gross_earnings : ℝ := h * r
noncomputable def tax_deduction : ℝ := tax_rate * gross_earnings
noncomputable def insurance_deduction : ℝ := insurance_rate * gross_earnings
noncomputable def total_deductions : ℝ := tax_deduction + insurance_deduction
noncomputable def net_earnings_before_union_dues : ℝ := gross_earnings - total_deductions
noncomputable def union_dues_deduction : ℝ := net_earnings_before_union_dues - take_home_pay

theorem union_dues_proof : union_dues_deduction = 5 := 
by sorry

end union_dues_proof_l31_31408


namespace Brandon_can_still_apply_l31_31291

-- Definitions based on the given conditions
def total_businesses : ℕ := 72
def fired_businesses : ℕ := total_businesses / 2
def quit_businesses : ℕ := total_businesses / 3
def businesses_restricted : ℕ := fired_businesses + quit_businesses

-- The final proof statement
theorem Brandon_can_still_apply : total_businesses - businesses_restricted = 12 :=
by
  -- Note: Proof is omitted; replace sorry with detailed proof in practice.
  sorry

end Brandon_can_still_apply_l31_31291


namespace hypotenuse_is_2_l31_31869

noncomputable def quadratic_trinomial_hypotenuse (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let xv := -b / (2 * a)
  let yv := a * xv^2 + b * xv + c
  if xv = (x1 + x2) / 2 then
    Real.sqrt 2 * abs (-b / a)
  else 0

theorem hypotenuse_is_2 {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  quadratic_trinomial_hypotenuse a b c = 2 := by
  sorry

end hypotenuse_is_2_l31_31869


namespace largest_divisor_of_product_of_five_consecutive_integers_l31_31434

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l31_31434


namespace pyramid_base_is_octagon_l31_31379
-- Import necessary library

-- Declare the problem
theorem pyramid_base_is_octagon (A : Nat) (h : A = 8) : A = 8 :=
by
  -- Proof goes here
  sorry

end pyramid_base_is_octagon_l31_31379


namespace amy_red_balloons_l31_31047

theorem amy_red_balloons (total_balloons green_balloons blue_balloons : ℕ) (h₁ : total_balloons = 67) (h₂: green_balloons = 17) (h₃ : blue_balloons = 21) : (total_balloons - (green_balloons + blue_balloons)) = 29 :=
by
  sorry

end amy_red_balloons_l31_31047


namespace tip_percentage_l31_31617

variable (L : ℝ) (T : ℝ)
 
theorem tip_percentage (h : L = 60.50) (h1 : T = 72.6) :
  ((T - L) / L) * 100 = 20 :=
by
  sorry

end tip_percentage_l31_31617


namespace product_of_five_consecutive_integers_divisible_by_120_l31_31419

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l31_31419


namespace find_perp_line_eq_l31_31742

-- Line equation in the standard form
def line_eq (x y : ℝ) : Prop := 4 * x - 3 * y - 12 = 0

-- Equation of the required line that is perpendicular to the given line and has the same y-intercept
def perp_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 16 = 0

theorem find_perp_line_eq (x y : ℝ) :
  (∃ k : ℝ, line_eq 0 k ∧ perp_line_eq 0 k) →
  (∃ a b c : ℝ, perp_line_eq a b) :=
by
  sorry

end find_perp_line_eq_l31_31742


namespace equidistant_points_l31_31635

theorem equidistant_points (r d1 d2 : ℝ) (d1_eq : d1 = r) (d2_eq : d2 = 6) : 
  ∃ p : ℝ, p = 2 := 
sorry

end equidistant_points_l31_31635


namespace blackRhinoCount_correct_l31_31629

noncomputable def numberOfBlackRhinos : ℕ :=
  let whiteRhinoCount := 7
  let whiteRhinoWeight := 5100
  let blackRhinoWeightInTons := 1
  let totalWeight := 51700
  let oneTonInPounds := 2000
  let totalWhiteRhinoWeight := whiteRhinoCount * whiteRhinoWeight
  let totalBlackRhinoWeight := totalWeight - totalWhiteRhinoWeight
  totalBlackRhinoWeight / (blackRhinoWeightInTons * oneTonInPounds)

theorem blackRhinoCount_correct : numberOfBlackRhinos = 8 := by
  sorry

end blackRhinoCount_correct_l31_31629


namespace arithmetic_seq_8th_term_l31_31193

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l31_31193


namespace mutually_exclusive_not_opposed_l31_31009

-- Define the types for cards and people
inductive Card
| red : Card
| white : Card
| black : Card

inductive Person
| A : Person
| B : Person
| C : Person

-- Define the event that a person receives a specific card
def receives (p : Person) (c : Card) : Prop := sorry

-- Conditions
axiom A_receives_red : receives Person.A Card.red → ¬ receives Person.B Card.red
axiom B_receives_red : receives Person.B Card.red → ¬ receives Person.A Card.red

-- The proof problem statement
theorem mutually_exclusive_not_opposed :
  (receives Person.A Card.red → ¬ receives Person.B Card.red) ∧
  (¬(receives Person.A Card.red ∧ receives Person.B Card.red)) ∧
  (¬∀ p : Person, receives p Card.red) :=
sorry

end mutually_exclusive_not_opposed_l31_31009


namespace largest_divisor_of_five_consecutive_integers_l31_31429

open Nat

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k ∈ {n, n+1, n+2, n+3, n+4} ∧
    ∀ m ∈ {2, 3, 4, 5}, m ∣ k → 60 ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) := 
sorry

end largest_divisor_of_five_consecutive_integers_l31_31429


namespace volume_of_largest_sphere_from_cube_l31_31534

theorem volume_of_largest_sphere_from_cube : 
  (∃ (V : ℝ), 
    (∀ (l : ℝ), l = 1 → (V = (4 / 3) * π * ((l / 2)^3)) → V = π / 6)) :=
sorry

end volume_of_largest_sphere_from_cube_l31_31534


namespace finished_in_6th_l31_31116

variable (p : ℕ → Prop)
variable (Sana Max Omar Jonah Leila : ℕ)

-- Conditions
def condition1 : Prop := Omar = Jonah - 7
def condition2 : Prop := Sana = Max - 2
def condition3 : Prop := Leila = Jonah + 3
def condition4 : Prop := Max = Omar + 1
def condition5 : Prop := Sana = 4

-- Conclusion
theorem finished_in_6th (h1 : condition1 Omar Jonah)
                         (h2 : condition2 Sana Max)
                         (h3 : condition3 Leila Jonah)
                         (h4 : condition4 Max Omar)
                         (h5 : condition5 Sana) :
  Max = 6 := by
  sorry

end finished_in_6th_l31_31116


namespace heidi_zoe_paint_wall_l31_31100

theorem heidi_zoe_paint_wall :
  let heidi_rate := (1 : ℚ) / 60
  let zoe_rate := (1 : ℚ) / 90
  let combined_rate := heidi_rate + zoe_rate
  let painted_fraction_15_minutes := combined_rate * 15
  painted_fraction_15_minutes = (5 : ℚ) / 12 :=
by
  sorry

end heidi_zoe_paint_wall_l31_31100


namespace stellar_hospital_multiple_births_l31_31515

/-- At Stellar Hospital, in a particular year, the multiple-birth statistics were such that sets of twins, triplets, and quintuplets accounted for 1200 of the babies born. 
There were twice as many sets of triplets as sets of quintuplets, and there were twice as many sets of twins as sets of triplets.
Determine how many of these 1200 babies were in sets of quintuplets. -/
theorem stellar_hospital_multiple_births 
    (a b c : ℕ)
    (h1 : b = 2 * c)
    (h2 : a = 2 * b)
    (h3 : 2 * a + 3 * b + 5 * c = 1200) :
    5 * c = 316 :=
by sorry

end stellar_hospital_multiple_births_l31_31515


namespace tan_alpha_plus_pi_over_4_sin_2alpha_expr_l31_31324

open Real

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
by
  sorry

theorem sin_2alpha_expr (α : ℝ) (h : tan α = 2) :
  (sin (2 * α)) / (sin (α) ^ 2 + sin (α) * cos (α)) = 2 / 3 :=
by
  sorry

end tan_alpha_plus_pi_over_4_sin_2alpha_expr_l31_31324


namespace sally_more_cards_than_dan_l31_31600

theorem sally_more_cards_than_dan :
  let sally_initial := 27
  let sally_bought := 20
  let dan_cards := 41
  sally_initial + sally_bought - dan_cards = 6 :=
by
  sorry

end sally_more_cards_than_dan_l31_31600


namespace min_distinct_values_l31_31774

theorem min_distinct_values (n : ℕ) (mode_freq : ℕ) (total : ℕ)
  (h1 : total = 3000) (h2 : mode_freq = 15) :
  n = 215 :=
by
  sorry

end min_distinct_values_l31_31774


namespace value_of_q_when_p_is_smallest_l31_31407

-- Definitions of primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m > 1, m < n → ¬ (n % m = 0)

-- smallest prime number
def smallest_prime : ℕ := 2

-- Given conditions
def p : ℕ := 3
def q : ℕ := 2 + 13 * p

-- The theorem to prove
theorem value_of_q_when_p_is_smallest :
  is_prime smallest_prime →
  is_prime q →
  smallest_prime = 2 →
  p = 3 →
  q = 41 :=
by sorry

end value_of_q_when_p_is_smallest_l31_31407


namespace sum_all_values_x_l31_31250

-- Define the problem's condition
def condition (x : ℝ) : Prop := Real.sqrt ((x - 2) ^ 2) = 9

-- Define the theorem to prove the sum of all solutions equals 4
theorem sum_all_values_x : ∑ x in {x : ℝ | condition x}, x = 4 := by
  -- Introduce the definition of condition
  sorry

end sum_all_values_x_l31_31250


namespace bill_score_l31_31133

theorem bill_score
  (J B S : ℕ)
  (h1 : B = J + 20)
  (h2 : B = S / 2)
  (h3 : J + B + S = 160) : 
  B = 45 := 
by 
  sorry

end bill_score_l31_31133


namespace leap_year_hours_l31_31824

theorem leap_year_hours (days_in_regular_year : ℕ) (hours_in_day : ℕ) (is_leap_year : Bool) : 
  is_leap_year = true ∧ days_in_regular_year = 365 ∧ hours_in_day = 24 → 
  366 * hours_in_day = 8784 :=
by
  intros
  sorry

end leap_year_hours_l31_31824


namespace books_problem_l31_31409

theorem books_problem
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) :
  M = 10 :=
by
  sorry

end books_problem_l31_31409


namespace parabola_intersection_sum_l31_31224

theorem parabola_intersection_sum : 
  ∃ x_0 y_0 : ℝ, (y_0 = x_0^2 + 15 * x_0 + 32) ∧ (x_0 = y_0^2 + 49 * y_0 + 593) ∧ (x_0 + y_0 = -33) :=
by
  sorry

end parabola_intersection_sum_l31_31224


namespace value_of_PQRS_l31_31947

theorem value_of_PQRS : 
  let P := 2 * (Real.sqrt 2010 + Real.sqrt 2011)
  let Q := 3 * (-Real.sqrt 2010 - Real.sqrt 2011)
  let R := 2 * (Real.sqrt 2010 - Real.sqrt 2011)
  let S := 3 * (Real.sqrt 2011 - Real.sqrt 2010)
  P * Q * R * S = -36 :=
by
  sorry

end value_of_PQRS_l31_31947


namespace system1_solution_l31_31140

theorem system1_solution (x y : ℝ) (h1 : 4 * x - 3 * y = 1) (h2 : 3 * x - 2 * y = -1) : x = -5 ∧ y = 7 :=
sorry

end system1_solution_l31_31140


namespace sandwiches_difference_l31_31720

theorem sandwiches_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let total_mw := monday_total + tuesday_total + wednesday_total

  let thursday_lunch := 3 * 2
  let thursday_dinner := 5
  let thursday_total := thursday_lunch + thursday_dinner

  total_mw - thursday_total = 24 :=
by
  sorry

end sandwiches_difference_l31_31720


namespace number_of_bottle_caps_l31_31978

-- Condition: 7 bottle caps weigh exactly one ounce
def weight_of_7_caps : ℕ := 1 -- ounce

-- Condition: Josh's collection weighs 18 pounds, and 1 pound = 16 ounces
def weight_of_collection_pounds : ℕ := 18 -- pounds
def weight_of_pound_in_ounces : ℕ := 16 -- ounces per pound

-- Condition: Question translated to proof statement
theorem number_of_bottle_caps :
  (weight_of_collection_pounds * weight_of_pound_in_ounces * 7 = 2016) :=
by
  sorry

end number_of_bottle_caps_l31_31978


namespace conclusion_friendly_not_large_l31_31587

variable {Snake : Type}
variable (isLarge isFriendly canClimb canSwim : Snake → Prop)
variable (marysSnakes : Finset Snake)
variable (h1 : marysSnakes.card = 16)
variable (h2 : (marysSnakes.filter isLarge).card = 6)
variable (h3 : (marysSnakes.filter isFriendly).card = 7)
variable (h4 : ∀ s, isFriendly s → canClimb s)
variable (h5 : ∀ s, isLarge s → ¬ canSwim s)
variable (h6 : ∀ s, ¬ canSwim s → ¬ canClimb s)

theorem conclusion_friendly_not_large :
  ∀ s, isFriendly s → ¬ isLarge s :=
by
  sorry

end conclusion_friendly_not_large_l31_31587


namespace tea_maker_capacity_l31_31604

theorem tea_maker_capacity (x : ℝ) (h : 0.45 * x = 54) : x = 120 :=
by
  sorry

end tea_maker_capacity_l31_31604


namespace probability_event_A_probability_event_B_probability_event_C_l31_31933

section

variables (Ω : Type) [Fintype Ω] (cards : Set {n : ℕ | n ∈ {1, 2, 3, 4}}) (draws : Π (i : Fin 3), Ω)

def event_A : Set (Π (i : Fin 3), Ω) := {draws | draws 0 = draws 1 ∧ draws 1 = draws 2}
def event_B : Set (Π (i : Fin 3), Ω) := {draws | ¬ (draws 0 = draws 1 ∧ draws 1 = draws 2)}
def event_C : Set (Π (i : Fin 3), Ω) := {draws | draws 0 * draws 1 = draws 2}

theorem probability_event_A : (Finset.card (event_A draws).to_finset : ℚ) / (Finset.card (@Set.univ (Π (i : Fin 3), Ω)).to_finset : ℚ) = 1 / 16 := 
sorry

theorem probability_event_B : (Finset.card (event_B draws).to_finset : ℚ) / (Finset.card (@Set.univ (Π (i : Fin 3), Ω)).to_finset : ℚ) = 15 / 16 := 
sorry

theorem probability_event_C : (Finset.card (event_C draws).to_finset : ℚ) / (Finset.card (@Set.univ (Π (i : Fin 3), Ω)).to_finset : ℚ) = 1 / 8 := 
sorry

end

end probability_event_A_probability_event_B_probability_event_C_l31_31933


namespace angle_APB_60_l31_31536

variable (P : ℝ × ℝ) (A B : ℝ × ℝ)
variable (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop)
variable (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop)

-- Define the circle C: (x - 6)^2 + (y - 2)^2 = 5
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + (y - 2)^2 = 5

-- Define the line l: y = 2x
def line_l (P : ℝ × ℝ) : Prop := P.2 = 2 * P.1

-- Condition: Tangents from P to the circle C are symmetric with respect to l
def symmetric_tangents (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  ∀ Q : ℝ × ℝ, l1 P Q → l2 P (reflection_over_line_l P Q)

-- Define the reflection of a point over the line y = 2x
def reflection_over_line_l (P Q : ℝ × ℝ) : ℝ × ℝ :=
  let a := (2 * Q.1 + Q.2) / 5
  let b := (4 * Q.1 - 2 * Q.2) / 5
  in (a, b)

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem angle_APB_60 (P A B : ℝ × ℝ) (hC : ∀ x y, C x y → circle_C x y)
  (hl : line_l P) (hl12 : symmetric_tangents l1 l2 P) :
  angle A P B = 60 :=
sorry

end angle_APB_60_l31_31536


namespace total_wings_l31_31578

-- Conditions
def money_per_grandparent : ℕ := 50
def number_of_grandparents : ℕ := 4
def bird_cost : ℕ := 20
def wings_per_bird : ℕ := 2

-- Calculate the total amount of money John received:
def total_money_received : ℕ := number_of_grandparents * money_per_grandparent

-- Determine the number of birds John can buy:
def number_of_birds : ℕ := total_money_received / bird_cost

-- Prove that the total number of wings all the birds have is 20:
theorem total_wings : number_of_birds * wings_per_bird = 20 :=
by
  sorry

end total_wings_l31_31578


namespace neg_p_equivalence_l31_31952

theorem neg_p_equivalence:
  (∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
sorry

end neg_p_equivalence_l31_31952


namespace sum_of_squares_l31_31400

variable (a b c : ℝ)
variable (S : ℝ)

theorem sum_of_squares (h1 : ab + bc + ac = 131)
                       (h2 : a + b + c = 22) :
  a^2 + b^2 + c^2 = 222 :=
by
  -- Proof would be placed here
  sorry

end sum_of_squares_l31_31400


namespace girls_not_playing_soccer_l31_31260

-- Define the given conditions
def students_total : Nat := 420
def boys_total : Nat := 312
def soccer_players_total : Nat := 250
def percent_boys_playing_soccer : Float := 0.78

-- Define the main goal based on the question and correct answer
theorem girls_not_playing_soccer : 
  students_total = 420 → 
  boys_total = 312 → 
  soccer_players_total = 250 → 
  percent_boys_playing_soccer = 0.78 → 
  ∃ (girls_not_playing_soccer : Nat), girls_not_playing_soccer = 53 :=
by 
  sorry

end girls_not_playing_soccer_l31_31260


namespace arithmetic_geometric_sequence_l31_31083

theorem arithmetic_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) 
(h1 : S 3 = 2) 
(h2 : S 6 = 18) 
(h3 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
: S 10 / S 5 = 33 := 
sorry

end arithmetic_geometric_sequence_l31_31083


namespace basket_weight_l31_31767

variable (B P : ℕ)

theorem basket_weight (h1 : B + P = 62) (h2 : B + P / 2 = 34) : B = 6 :=
by
  sorry

end basket_weight_l31_31767


namespace fruiting_plants_given_away_l31_31599

noncomputable def roxy_fruiting_plants_given_away 
  (N_f : ℕ) -- initial flowering plants
  (N_ft : ℕ) -- initial fruiting plants
  (N_bsf : ℕ) -- flowering plants bought on Saturday
  (N_bst : ℕ) -- fruiting plants bought on Saturday
  (N_gsf : ℕ) -- flowering plant given away on Sunday
  (N_total_remaining : ℕ) -- total plants remaining 
  (H₁ : N_ft = 2 * N_f) -- twice as many fruiting plants
  (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) -- total plants equation
  : ℕ :=
  4

theorem fruiting_plants_given_away (N_f : ℕ) (N_ft : ℕ) (N_bsf : ℕ) (N_bst : ℕ) (N_gsf : ℕ) (N_total_remaining : ℕ)
  (H₁ : N_ft = 2 * N_f) (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) : N_ft - (N_total_remaining - (N_f + N_bsf - N_gsf)) = 4 := 
by
  sorry

end fruiting_plants_given_away_l31_31599


namespace find_side_y_l31_31917

noncomputable def side_length_y : ℝ :=
  let AB := 10 / Real.sqrt 2
  let AD := 10
  let CD := AD / 2
  CD * Real.sqrt 3

theorem find_side_y : side_length_y = 5 * Real.sqrt 3 := by
  let AB : ℝ := 10 / Real.sqrt 2
  let AD : ℝ := 10
  let CD : ℝ := AD / 2
  have h1 : CD * Real.sqrt 3 = 5 * Real.sqrt 3 := by sorry
  exact h1

end find_side_y_l31_31917


namespace original_plan_was_to_produce_125_sets_per_day_l31_31759

-- We state our conditions
def plans_to_complete_in_days : ℕ := 30
def produces_sets_per_day : ℕ := 150
def finishes_days_ahead_of_schedule : ℕ := 5

-- Calculations based on conditions
def actual_days_used : ℕ := plans_to_complete_in_days - finishes_days_ahead_of_schedule
def total_production : ℕ := produces_sets_per_day * actual_days_used
def original_planned_production_per_day : ℕ := total_production / plans_to_complete_in_days

-- Claim we want to prove
theorem original_plan_was_to_produce_125_sets_per_day :
  original_planned_production_per_day = 125 :=
by
  sorry

end original_plan_was_to_produce_125_sets_per_day_l31_31759


namespace dice_faces_l31_31318

theorem dice_faces (n : ℕ) (h : (1 / (n : ℝ)) ^ 5 = 0.0007716049382716049) : n = 10 := sorry

end dice_faces_l31_31318


namespace product_of_five_consecutive_is_divisible_by_sixty_l31_31413

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l31_31413


namespace expressions_equal_l31_31060

variable (a b c : ℝ)

theorem expressions_equal (h : a + 2 * b + 2 * c = 0) : a + 2 * b * c = (a + 2 * b) * (a + 2 * c) := 
by 
  sorry

end expressions_equal_l31_31060


namespace arithmetic_seq_8th_term_l31_31186

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l31_31186


namespace percent_nonunion_women_l31_31355

variable (E : ℝ) -- Total number of employees

-- Definitions derived from the problem conditions
def menPercent : ℝ := 0.46
def unionPercent : ℝ := 0.60
def nonUnionPercent : ℝ := 1 - unionPercent
def nonUnionWomenPercent : ℝ := 0.90

theorem percent_nonunion_women :
  nonUnionWomenPercent = 0.90 :=
by
  sorry

end percent_nonunion_women_l31_31355


namespace locus_of_Q_l31_31623

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2/a^2 + y^2/b^2 = 1

def A_vertice (a b : ℝ) (x y : ℝ) : Prop :=
  (x = a ∧ y = 0) ∨ (x = -a ∧ y = 0)

def chord_parallel_y_axis (x : ℝ) : Prop :=
  -- Assuming chord's x coordinate is given
  True

def lines_intersect_at_Q (a b Qx Qy : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse a b x y ∧
  A_vertice a b x y ∧
  chord_parallel_y_axis x ∧
  (
    ( (Qy - y) / (Qx - (-a)) = (Qy - 0) / (Qx - a) ) ∨ -- A'P slope-comp
    ( (Qy - (-y)) / (Qx - a) = (Qy - 0) / (Qx - (-a)) ) -- AP' slope-comp
  )

theorem locus_of_Q (a b Qx Qy : ℝ) :
  (lines_intersect_at_Q a b Qx Qy) →
  (Qx^2 / a^2 - Qy^2 / b^2 = 1) := by
  sorry

end locus_of_Q_l31_31623


namespace Brandon_can_still_apply_l31_31290

-- Definitions based on the given conditions
def total_businesses : ℕ := 72
def fired_businesses : ℕ := total_businesses / 2
def quit_businesses : ℕ := total_businesses / 3
def businesses_restricted : ℕ := fired_businesses + quit_businesses

-- The final proof statement
theorem Brandon_can_still_apply : total_businesses - businesses_restricted = 12 :=
by
  -- Note: Proof is omitted; replace sorry with detailed proof in practice.
  sorry

end Brandon_can_still_apply_l31_31290


namespace first_month_sale_l31_31638

theorem first_month_sale 
(sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
(avg_sale : ℕ) 
(h_avg: avg_sale = 6500)
(h_sale2: sale_2 = 6927)
(h_sale3: sale_3 = 6855)
(h_sale4: sale_4 = 7230)
(h_sale5: sale_5 = 6562)
(h_sale6: sale_6 = 4791)
: sale_1 = 6635 := by
  sorry

end first_month_sale_l31_31638


namespace opposite_numbers_l31_31349

theorem opposite_numbers (a b : ℝ) (h : a = -b) : b = -a := 
by 
  sorry

end opposite_numbers_l31_31349


namespace find_number_of_students_l31_31262

theorem find_number_of_students 
    (N T : ℕ) 
    (h1 : T = 80 * N)
    (h2 : (T - 350) / (N - 5) = 90) : 
    N = 10 :=
sorry

end find_number_of_students_l31_31262


namespace simultaneous_equations_solution_l31_31011

theorem simultaneous_equations_solution (x y : ℚ) :
  3 * x^2 + x * y - 2 * y^2 = -5 ∧ x^2 + 2 * x * y + y^2 = 1 ↔ 
  (x = 3/5 ∧ y = -8/5) ∨ (x = -3/5 ∧ y = 8/5) :=
by
  sorry

end simultaneous_equations_solution_l31_31011


namespace average_annual_growth_rate_sales_revenue_2018_l31_31571

-- Define the conditions as hypotheses
def initial_sales := 200000
def final_sales := 800000
def years := 2
def growth_rate := 1.0 -- representing 100%

theorem average_annual_growth_rate (x : ℝ) :
  (initial_sales : ℝ) * (1 + x)^years = final_sales → x = 1 :=
by
  intro h1
  sorry

theorem sales_revenue_2018 (x : ℝ) (revenue_2017 : ℝ) :
  x = 1 → revenue_2017 = final_sales → revenue_2017 * (1 + x) = 1600000 :=
by
  intros h1 h2
  sorry

end average_annual_growth_rate_sales_revenue_2018_l31_31571


namespace sum_of_three_integers_l31_31003

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 125) : a + b + c = 31 :=
sorry

end sum_of_three_integers_l31_31003


namespace arithmetic_sequence_8th_term_l31_31154

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l31_31154


namespace nat_gt_10_is_diff_of_hypotenuse_numbers_l31_31242

def is_hypotenuse_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem nat_gt_10_is_diff_of_hypotenuse_numbers (n : ℕ) (h : n > 10) : 
  ∃ (n₁ n₂ : ℕ), is_hypotenuse_number n₁ ∧ is_hypotenuse_number n₂ ∧ n = n₁ - n₂ :=
by
  sorry

end nat_gt_10_is_diff_of_hypotenuse_numbers_l31_31242


namespace distance_between_stripes_l31_31279

/-- Define the parallel curbs and stripes -/
structure Crosswalk where
  distance_between_curbs : ℝ
  curb_distance_between_stripes : ℝ
  stripe_length : ℝ
  stripe_cross_distance : ℝ
  
open Crosswalk

/-- Conditions given in the problem -/
def crosswalk : Crosswalk where
  distance_between_curbs := 60 -- feet
  curb_distance_between_stripes := 20 -- feet
  stripe_length := 50 -- feet
  stripe_cross_distance := 50 -- feet

/-- Theorem to prove the distance between stripes -/
theorem distance_between_stripes (cw : Crosswalk) :
  2 * (cw.curb_distance_between_stripes * cw.distance_between_curbs) / cw.stripe_length = 24 := sorry

end distance_between_stripes_l31_31279


namespace arithmetic_mean_of_18_24_42_l31_31245

-- Define the numbers a, b, c
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 42

-- Define the arithmetic mean
def mean (x y z : ℕ) : ℕ := (x + y + z) / 3

-- State the theorem to be proved
theorem arithmetic_mean_of_18_24_42 : mean a b c = 28 :=
by
  sorry

end arithmetic_mean_of_18_24_42_l31_31245


namespace product_of_five_consecutive_integers_divisible_by_120_l31_31418

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l31_31418


namespace largest_divisor_of_5_consecutive_integers_l31_31476

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (n : ℤ), ∃ d, d = 120 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intro n
  use 120
  split
  exact rfl
  sorry

end largest_divisor_of_5_consecutive_integers_l31_31476


namespace solve_rational_equation_solve_quadratic_equation_l31_31736

-- Statement for the first equation
theorem solve_rational_equation (x : ℝ) (h : x ≠ 1) : 
  (x / (x - 1) + 2 / (1 - x) = 2) → (x = 0) :=
by intro h1; sorry

-- Statement for the second equation
theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 + 6 * x - 3 = 0) → (x = 1/2 ∨ x = -3) :=
by intro h1; sorry

end solve_rational_equation_solve_quadratic_equation_l31_31736


namespace series_sum_eq_negative_one_third_l31_31301

noncomputable def series_sum : ℝ :=
  ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2) * (n + 3))

theorem series_sum_eq_negative_one_third : series_sum = -1 / 3 := sorry

end series_sum_eq_negative_one_third_l31_31301


namespace first_reduction_is_12_percent_l31_31904

theorem first_reduction_is_12_percent (P : ℝ) (x : ℝ) (h1 : (1 - x / 100) * 0.9 * P = 0.792 * P) : x = 12 :=
by
  sorry

end first_reduction_is_12_percent_l31_31904


namespace find_m_l31_31953

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

-- The intersection condition
def intersect_condition (m : ℝ) : Prop := A m ∩ B = {3}

-- The statement to prove
theorem find_m : ∃ m : ℝ, intersect_condition m → m = 3 :=
by {
  use 3,
  sorry
}

end find_m_l31_31953


namespace problem_part_I_problem_part_II_l31_31554

-- Define the problem and the proof requirements in Lean 4
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (sinB_nonneg : 0 ≤ Real.sin B) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) 
(h_a : a = 2) (h_b : b = 2) : 
Real.cos B = 1/4 :=
sorry

theorem problem_part_II (a b c : ℝ) (A B C : ℝ) (h_B : B = π / 2) 
(h_a : a = Real.sqrt 2) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) :
1/2 * a * c = 1 :=
sorry

end problem_part_I_problem_part_II_l31_31554


namespace distance_between_stripes_l31_31280

/-- Define the parallel curbs and stripes -/
structure Crosswalk where
  distance_between_curbs : ℝ
  curb_distance_between_stripes : ℝ
  stripe_length : ℝ
  stripe_cross_distance : ℝ
  
open Crosswalk

/-- Conditions given in the problem -/
def crosswalk : Crosswalk where
  distance_between_curbs := 60 -- feet
  curb_distance_between_stripes := 20 -- feet
  stripe_length := 50 -- feet
  stripe_cross_distance := 50 -- feet

/-- Theorem to prove the distance between stripes -/
theorem distance_between_stripes (cw : Crosswalk) :
  2 * (cw.curb_distance_between_stripes * cw.distance_between_curbs) / cw.stripe_length = 24 := sorry

end distance_between_stripes_l31_31280


namespace A_superset_C_l31_31549

-- Definitions of the sets as given in the problem statement
def U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {-1, 3}
def C : Set ℝ := {x | -1 < x ∧ x < 3}

-- Statement to be proved: A ⊇ C
theorem A_superset_C : A ⊇ C :=
by sorry

end A_superset_C_l31_31549


namespace arithmetic_sequence_8th_term_is_71_l31_31181

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l31_31181


namespace child_ticket_cost_l31_31813

variable (A C : ℕ) -- A stands for the number of adults, C stands for the cost of one child's ticket

theorem child_ticket_cost 
  (number_of_adults : ℕ) 
  (number_of_children : ℕ) 
  (cost_concessions : ℕ) 
  (total_cost_trip : ℕ)
  (cost_adult_ticket : ℕ) 
  (ticket_costs : ℕ) 
  (total_adult_cost : ℕ) 
  (remaining_ticket_cost : ℕ) 
  (child_ticket : ℕ) :
  number_of_adults = 5 →
  number_of_children = 2 →
  cost_concessions = 12 →
  total_cost_trip = 76 →
  cost_adult_ticket = 10 →
  ticket_costs = total_cost_trip - cost_concessions →
  total_adult_cost = number_of_adults * cost_adult_ticket →
  remaining_ticket_cost = ticket_costs - total_adult_cost →
  child_ticket = remaining_ticket_cost / number_of_children →
  C = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- Adding sorry since the proof is not required
  sorry

end child_ticket_cost_l31_31813


namespace kyunghoon_time_to_go_down_l31_31980

theorem kyunghoon_time_to_go_down (d : ℕ) (t_up t_down total_time : ℕ) : 
  ((t_up = d / 3) ∧ (t_down = (d + 2) / 4) ∧ (total_time = 4) → (t_up + t_down = total_time) → (t_down = 2)) := 
by
  sorry

end kyunghoon_time_to_go_down_l31_31980


namespace exists_smallest_n_l31_31927

theorem exists_smallest_n :
  ∃ n : ℕ, (n^2 + 20 * n + 19) % 2019 = 0 ∧ n = 2000 :=
sorry

end exists_smallest_n_l31_31927


namespace probability_of_yellow_or_green_l31_31248

def bag : List (String × Nat) := [("yellow", 4), ("green", 3), ("red", 2), ("blue", 1)]

def total_marbles (bag : List (String × Nat)) : Nat := bag.foldr (fun (_, n) acc => n + acc) 0

def favorable_outcomes (bag : List (String × Nat)) : Nat :=
  (bag.filter (fun (color, _) => color = "yellow" ∨ color = "green")).foldr (fun (_, n) acc => n + acc) 0

theorem probability_of_yellow_or_green :
  (favorable_outcomes bag : ℚ) / (total_marbles bag : ℚ) = 7 / 10 := by
  sorry

end probability_of_yellow_or_green_l31_31248
