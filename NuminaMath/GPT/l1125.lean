import Mathlib

namespace NUMINAMATH_GPT_smallest_possible_AAB_l1125_112595

-- Definitions of the digits A and B
def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

-- Definition of the condition AB equals 1/7 of AAB
def condition (A B : ℕ) : Prop := 10 * A + B = (1 / 7) * (110 * A + B)

theorem smallest_possible_AAB (A B : ℕ) : is_valid_digit A ∧ is_valid_digit B ∧ condition A B → 110 * A + B = 664 := sorry

end NUMINAMATH_GPT_smallest_possible_AAB_l1125_112595


namespace NUMINAMATH_GPT_exists_subset_sum_2n_l1125_112560

theorem exists_subset_sum_2n (n : ℕ) (h : n > 3) (s : Finset ℕ)
  (hs : ∀ x ∈ s, x < 2 * n) (hs_card : s.card = 2 * n)
  (hs_sum : s.sum id = 4 * n) :
  ∃ t ⊆ s, t.sum id = 2 * n :=
by sorry

end NUMINAMATH_GPT_exists_subset_sum_2n_l1125_112560


namespace NUMINAMATH_GPT_largest_n_for_factored_quad_l1125_112545

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end NUMINAMATH_GPT_largest_n_for_factored_quad_l1125_112545


namespace NUMINAMATH_GPT_percentage_difference_l1125_112570

theorem percentage_difference :
  (0.50 * 56 - 0.30 * 50) = 13 := 
by
  -- sorry is used to skip the actual proof steps
  sorry 

end NUMINAMATH_GPT_percentage_difference_l1125_112570


namespace NUMINAMATH_GPT_correct_equation_l1125_112537

theorem correct_equation (x : ℝ) : (-x^2)^2 = x^4 := by sorry

end NUMINAMATH_GPT_correct_equation_l1125_112537


namespace NUMINAMATH_GPT_auction_site_TVs_correct_l1125_112509

-- Define the number of TVs Beatrice looked at in person
def in_person_TVs : Nat := 8

-- Define the number of TVs Beatrice looked at online
def online_TVs : Nat := 3 * in_person_TVs

-- Define the total number of TVs Beatrice looked at
def total_TVs : Nat := 42

-- Define the number of TVs Beatrice looked at on the auction site
def auction_site_TVs : Nat := total_TVs - (in_person_TVs + online_TVs)

-- Prove that the number of TVs Beatrice looked at on the auction site is 10
theorem auction_site_TVs_correct : auction_site_TVs = 10 :=
by
  sorry

end NUMINAMATH_GPT_auction_site_TVs_correct_l1125_112509


namespace NUMINAMATH_GPT_roots_polynomial_value_l1125_112508

theorem roots_polynomial_value (r s t : ℝ) (h₁ : r + s + t = 15) (h₂ : r * s + s * t + t * r = 25) (h₃ : r * s * t = 10) :
  (1 + r) * (1 + s) * (1 + t) = 51 :=
by
  sorry

end NUMINAMATH_GPT_roots_polynomial_value_l1125_112508


namespace NUMINAMATH_GPT_amelia_drove_tuesday_l1125_112576

-- Define the known quantities
def total_distance : ℕ := 8205
def distance_monday : ℕ := 907
def remaining_distance : ℕ := 6716

-- Define the distance driven on Tuesday and state the theorem
def distance_tuesday : ℕ := total_distance - (distance_monday + remaining_distance)

-- Theorem stating the distance driven on Tuesday is 582 kilometers
theorem amelia_drove_tuesday : distance_tuesday = 582 := 
by
  -- We skip the proof for now
  sorry

end NUMINAMATH_GPT_amelia_drove_tuesday_l1125_112576


namespace NUMINAMATH_GPT_incorrect_scientific_statement_is_D_l1125_112521

-- Define the number of colonies screened by Student A and other students
def studentA_colonies := 150
def other_students_colonies := 50

-- Define the descriptions
def descriptionA := "The reason Student A had such results could be due to different soil samples or problems in the experimental operation."
def descriptionB := "Student A's prepared culture medium could be cultured without adding soil as a blank control, to demonstrate whether the culture medium is contaminated."
def descriptionC := "If other students use the same soil as Student A for the experiment and get consistent results with Student A, it can be proven that Student A's operation was without error."
def descriptionD := "Both experimental approaches described in options B and C follow the principle of control in the experiment."

-- The incorrect scientific statement identified
def incorrect_statement := descriptionD

-- The main theorem statement
theorem incorrect_scientific_statement_is_D : incorrect_statement = descriptionD := by
  sorry

end NUMINAMATH_GPT_incorrect_scientific_statement_is_D_l1125_112521


namespace NUMINAMATH_GPT_compound_interest_1200_20percent_3years_l1125_112516

noncomputable def compoundInterest (P r : ℚ) (n t : ℕ) : ℚ :=
  let A := P * (1 + r / n) ^ (n * t)
  A - P

theorem compound_interest_1200_20percent_3years :
  compoundInterest 1200 0.20 1 3 = 873.6 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_1200_20percent_3years_l1125_112516


namespace NUMINAMATH_GPT_total_trip_length_l1125_112554

theorem total_trip_length :
  ∀ (d : ℝ), 
    (∀ fuel_per_mile : ℝ, fuel_per_mile = 0.03 →
      ∀ battery_miles : ℝ, battery_miles = 50 →
      ∀ avg_miles_per_gallon : ℝ, avg_miles_per_gallon = 50 →
      (d / (fuel_per_mile * (d - battery_miles))) = avg_miles_per_gallon →
      d = 150) := 
by
  intros d fuel_per_mile fuel_per_mile_eq battery_miles battery_miles_eq avg_miles_per_gallon avg_miles_per_gallon_eq trip_condition
  sorry

end NUMINAMATH_GPT_total_trip_length_l1125_112554


namespace NUMINAMATH_GPT_julien_contribution_l1125_112513

def exchange_rate : ℝ := 1.5
def cost_of_pie : ℝ := 12
def lucas_cad : ℝ := 10

theorem julien_contribution : (cost_of_pie - lucas_cad / exchange_rate) = 16 / 3 := by
  sorry

end NUMINAMATH_GPT_julien_contribution_l1125_112513


namespace NUMINAMATH_GPT_womenInBusinessClass_l1125_112511

-- Given conditions
def totalPassengers : ℕ := 300
def percentageWomen : ℚ := 70 / 100
def percentageWomenBusinessClass : ℚ := 15 / 100

def numberOfWomen (totalPassengers : ℕ) (percentageWomen : ℚ) : ℚ := 
  totalPassengers * percentageWomen

def numberOfWomenBusinessClass (numberOfWomen : ℚ) (percentageWomenBusinessClass : ℚ) : ℚ := 
  numberOfWomen * percentageWomenBusinessClass

-- Theorem to prove
theorem womenInBusinessClass (totalPassengers : ℕ) (percentageWomen : ℚ) (percentageWomenBusinessClass : ℚ) :
  numberOfWomenBusinessClass (numberOfWomen totalPassengers percentageWomen) percentageWomenBusinessClass = 32 := 
by 
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_womenInBusinessClass_l1125_112511


namespace NUMINAMATH_GPT_commission_amount_l1125_112565

theorem commission_amount 
  (new_avg_commission : ℤ) (increase_in_avg : ℤ) (sales_count : ℤ) 
  (total_commission_before : ℤ) (total_commission_after : ℤ) : 
  new_avg_commission = 400 → increase_in_avg = 150 → sales_count = 6 → 
  total_commission_before = (sales_count - 1) * (new_avg_commission - increase_in_avg) → 
  total_commission_after = sales_count * new_avg_commission → 
  total_commission_after - total_commission_before = 1150 :=
by 
  sorry

end NUMINAMATH_GPT_commission_amount_l1125_112565


namespace NUMINAMATH_GPT_range_of_a_l1125_112596

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1125_112596


namespace NUMINAMATH_GPT_pebble_sequence_10_l1125_112582

-- A definition for the sequence based on the given conditions and pattern.
def pebble_sequence : ℕ → ℕ
| 0 => 1
| 1 => 5
| 2 => 12
| 3 => 22
| (n + 4) => pebble_sequence (n + 3) + (3 * (n + 1) + 1)

-- Theorem that states the value at the 10th position in the sequence.
theorem pebble_sequence_10 : pebble_sequence 9 = 145 :=
sorry

end NUMINAMATH_GPT_pebble_sequence_10_l1125_112582


namespace NUMINAMATH_GPT_petya_winning_probability_l1125_112501

noncomputable def petya_wins_probability : ℚ :=
  (1 / 4) ^ 4

-- The main theorem statement
theorem petya_winning_probability :
  petya_wins_probability = 1 / 256 :=
by sorry

end NUMINAMATH_GPT_petya_winning_probability_l1125_112501


namespace NUMINAMATH_GPT_rice_and_wheat_grains_division_l1125_112577

-- Definitions for the conditions in the problem
def total_grains : ℕ := 1534
def sample_size : ℕ := 254
def wheat_in_sample : ℕ := 28

-- Proving the approximate amount of wheat grains in the batch  
theorem rice_and_wheat_grains_division : total_grains * (wheat_in_sample / sample_size) = 169 := by 
  sorry

end NUMINAMATH_GPT_rice_and_wheat_grains_division_l1125_112577


namespace NUMINAMATH_GPT_train_speed_proof_l1125_112552

def identical_trains_speed : Real :=
  11.11

theorem train_speed_proof :
  ∀ (v : ℝ),
  (∀ (t t' : ℝ), 
  (t = 150 / v) ∧ 
  (t' = 300 / v) ∧ 
  ((t' + 100 / v) = 36)) → v = identical_trains_speed :=
by
  sorry

end NUMINAMATH_GPT_train_speed_proof_l1125_112552


namespace NUMINAMATH_GPT_remainder_of_exponentiation_is_correct_l1125_112563

-- Define the given conditions
def modulus := 500
def exponent := 5 ^ (5 ^ 5)
def carmichael_500 := 100
def carmichael_100 := 20

-- Prove the main theorem
theorem remainder_of_exponentiation_is_correct :
  (5 ^ exponent) % modulus = 125 := 
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_remainder_of_exponentiation_is_correct_l1125_112563


namespace NUMINAMATH_GPT_min_time_one_ball_l1125_112542

noncomputable def children_circle_min_time (n : ℕ) := 98

theorem min_time_one_ball (n : ℕ) (h1 : n = 99) : 
  children_circle_min_time n = 98 := 
by 
  sorry

end NUMINAMATH_GPT_min_time_one_ball_l1125_112542


namespace NUMINAMATH_GPT_two_solutions_for_positive_integer_m_l1125_112543

theorem two_solutions_for_positive_integer_m :
  ∃ k : ℕ, k = 2 ∧ (∀ m : ℕ, 0 < m → 990 % (m^2 - 2) = 0 → m = 2 ∨ m = 3) := 
sorry

end NUMINAMATH_GPT_two_solutions_for_positive_integer_m_l1125_112543


namespace NUMINAMATH_GPT_simplify_expression_l1125_112572

theorem simplify_expression (x y : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 20 + 4 * y = 45 * x + 20 + 4 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1125_112572


namespace NUMINAMATH_GPT_winning_candidate_percentage_l1125_112586

theorem winning_candidate_percentage (total_membership: ℕ)
  (votes_cast: ℕ) (winning_percentage: ℝ) (h1: total_membership = 1600)
  (h2: votes_cast = 525) (h3: winning_percentage = 19.6875)
  : (winning_percentage / 100 * total_membership / votes_cast * 100 = 60) :=
by
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l1125_112586


namespace NUMINAMATH_GPT_range_of_a_l1125_112504

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (2 * a + 1) * x + a^2 + a < 0 → 0 < 2 * x - 1 ∧ 2 * x - 1 ≤ 10) →
  (∃ l u : ℝ, (l = 1/2) ∧ (u = 9/2) ∧ (l ≤ a ∧ a ≤ u)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1125_112504


namespace NUMINAMATH_GPT_soda_choosers_l1125_112562

-- Definitions based on conditions
def total_people := 600
def soda_angle := 108
def full_circle := 360

-- Statement to prove the number of people who referred to soft drinks as "Soda"
theorem soda_choosers : total_people * (soda_angle / full_circle) = 180 :=
by
  sorry

end NUMINAMATH_GPT_soda_choosers_l1125_112562


namespace NUMINAMATH_GPT_solve_for_x_l1125_112583

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 4 * x = 0) (h₁ : x ≠ 0) : x = 4 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1125_112583


namespace NUMINAMATH_GPT_mandatory_state_tax_rate_l1125_112522

theorem mandatory_state_tax_rate 
  (MSRP : ℝ) (total_paid : ℝ) (insurance_rate : ℝ) (tax_rate : ℝ) 
  (insurance_cost : ℝ := insurance_rate * MSRP)
  (cost_before_tax : ℝ := MSRP + insurance_cost)
  (tax_amount : ℝ := total_paid - cost_before_tax) :
  MSRP = 30 → total_paid = 54 → insurance_rate = 0.2 → 
  tax_amount / cost_before_tax * 100 = tax_rate →
  tax_rate = 50 :=
by
  intros MSRP_val paid_val ins_rate_val comp_tax_rate
  sorry

end NUMINAMATH_GPT_mandatory_state_tax_rate_l1125_112522


namespace NUMINAMATH_GPT_problem_1_problem_2_l1125_112550

-- First Problem
theorem problem_1 (f : ℝ → ℝ) (a : ℝ) (h : ∃ x : ℝ, f x - 2 * |x - 7| ≤ 0) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → a ≥ -12 :=
by
  intros
  sorry

-- Second Problem
theorem problem_2 (f : ℝ → ℝ) (a m : ℝ) (h1 : a = 1) 
  (h2 : ∀ x : ℝ, f x + |x + 7| ≥ m) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → m ≤ 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1125_112550


namespace NUMINAMATH_GPT_circle_through_points_eq_l1125_112506

noncomputable def circle_eqn (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_through_points_eq {h k r : ℝ} :
  circle_eqn h k r (-1) 0 ∧
  circle_eqn h k r 0 2 ∧
  circle_eqn h k r 2 0 → 
  (h = 2 / 3 ∧ k = 2 / 3 ∧ r^2 = 29 / 9) :=
sorry

end NUMINAMATH_GPT_circle_through_points_eq_l1125_112506


namespace NUMINAMATH_GPT_plus_signs_count_l1125_112533

theorem plus_signs_count (total_symbols : ℕ) (n_plus n_minus : ℕ)
  (h1 : total_symbols = 23)
  (h2 : ∀ s : Finset ℕ, s.card = 10 → ∃ i ∈ s, i ≤ n_plus)
  (h3 : ∀ s : Finset ℕ, s.card = 15 → ∃ i ∈ s, i > n_plus) :
  n_plus = 14 :=
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_plus_signs_count_l1125_112533


namespace NUMINAMATH_GPT_card_problem_l1125_112532

-- Define the variables
variables (x y : ℕ)

-- Conditions given in the problem
theorem card_problem 
  (h1 : x - 1 = y + 1) 
  (h2 : x + 1 = 2 * (y - 1)) : 
  x + y = 12 :=
sorry

end NUMINAMATH_GPT_card_problem_l1125_112532


namespace NUMINAMATH_GPT_rectangle_area_correct_l1125_112527

noncomputable def rectangle_area (x: ℚ) : ℚ :=
  let length := 5 * x - 18
  let width := 25 - 4 * x
  length * width

theorem rectangle_area_correct (x: ℚ) (h1: 3.6 < x) (h2: x < 6.25) :
  rectangle_area (43 / 9) = (2809 / 81) := 
  by
    sorry

end NUMINAMATH_GPT_rectangle_area_correct_l1125_112527


namespace NUMINAMATH_GPT_place_sweet_hexagons_l1125_112530

def sweetHexagon (h : ℝ) : Prop := h = 1
def convexPolygon (A : ℝ) : Prop := A ≥ 1900000
def hexagonPlacementPossible (N : ℕ) : Prop := N ≤ 2000000

theorem place_sweet_hexagons:
  (∀ h, sweetHexagon h) →
  (∃ A, convexPolygon A) →
  (∃ N, hexagonPlacementPossible N) →
  True :=
by
  intros _ _ _ 
  exact True.intro

end NUMINAMATH_GPT_place_sweet_hexagons_l1125_112530


namespace NUMINAMATH_GPT_total_rehabilitation_centers_l1125_112585

def lisa_visits : ℕ := 6
def jude_visits (lisa : ℕ) : ℕ := lisa / 2
def han_visits (jude : ℕ) : ℕ := 2 * jude - 2
def jane_visits (han : ℕ) : ℕ := 2 * han + 6
def total_visits (lisa jude han jane : ℕ) : ℕ := lisa + jude + han + jane

theorem total_rehabilitation_centers :
  total_visits lisa_visits (jude_visits lisa_visits) (han_visits (jude_visits lisa_visits)) 
    (jane_visits (han_visits (jude_visits lisa_visits))) = 27 :=
by
  sorry

end NUMINAMATH_GPT_total_rehabilitation_centers_l1125_112585


namespace NUMINAMATH_GPT_contradiction_assumption_l1125_112529

theorem contradiction_assumption (a b : ℝ) (h : a ≤ 2 ∧ b ≤ 2) : (a > 2 ∨ b > 2) -> false :=
by
  sorry

end NUMINAMATH_GPT_contradiction_assumption_l1125_112529


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1125_112502

def Q (x : ℝ) (n : ℕ) : ℝ := (x + 1) ^ n - x ^ n - 1
def P (x : ℝ) : ℝ := x ^ 2 + x + 1

theorem part_a (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1 ∨ n = 6 * k - 1) ↔ (∀ x : ℝ, P x ∣ Q x n) := sorry

theorem part_b (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1) ↔ (∀ x : ℝ, (P x)^2 ∣ Q x n) := sorry

theorem part_c (n : ℕ) : 
  n = 1 ↔ (∀ x : ℝ, (P x)^3 ∣ Q x n) := sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1125_112502


namespace NUMINAMATH_GPT_intersection_complement_l1125_112531

def set_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def set_B := {x : ℝ | x < 1}
def complement_B := {x : ℝ | x ≥ 1}

theorem intersection_complement :
  (set_A ∩ complement_B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1125_112531


namespace NUMINAMATH_GPT_arrows_from_530_to_535_l1125_112578

def cyclic_arrows (n : Nat) : Nat :=
  n % 5

theorem arrows_from_530_to_535 : 
  cyclic_arrows 530 = 0 ∧ cyclic_arrows 531 = 1 ∧ cyclic_arrows 532 = 2 ∧
  cyclic_arrows 533 = 3 ∧ cyclic_arrows 534 = 4 ∧ cyclic_arrows 535 = 0 :=
by
  sorry

end NUMINAMATH_GPT_arrows_from_530_to_535_l1125_112578


namespace NUMINAMATH_GPT_base_of_second_term_l1125_112588

theorem base_of_second_term (e : ℕ) (base : ℝ) 
  (h1 : e = 35) 
  (h2 : (1/5)^e * base^18 = 1 / (2 * (10)^35)) : 
  base = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_base_of_second_term_l1125_112588


namespace NUMINAMATH_GPT_greatest_n_for_xy_le_0_l1125_112566

theorem greatest_n_for_xy_le_0
  (a b : ℕ) (coprime_ab : Nat.gcd a b = 1) :
  ∃ n : ℕ, (n = a * b ∧ ∃ x y : ℤ, n = a * x + b * y ∧ x * y ≤ 0) :=
sorry

end NUMINAMATH_GPT_greatest_n_for_xy_le_0_l1125_112566


namespace NUMINAMATH_GPT_seq_prime_l1125_112580

/-- A strictly increasing sequence of positive integers. -/
def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m, n < m → a n < a m

/-- An infinite strictly increasing sequence of positive integers. -/
def strictly_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, 0 < a n ∧ is_strictly_increasing a

/-- A sequence of distinct primes. -/
def distinct_primes (p : ℕ → ℕ) : Prop :=
  ∀ m n, m ≠ n → p m ≠ p n ∧ Nat.Prime (p n)

/-- The main theorem to be proved. -/
theorem seq_prime (a p : ℕ → ℕ) (h1 : strictly_increasing_sequence a) (h2 : distinct_primes p)
  (h3 : ∀ n, p n ∣ a n) (h4 : ∀ n k, a n - a k = p n - p k) : ∀ n, Nat.Prime (a n) := 
by
  sorry

end NUMINAMATH_GPT_seq_prime_l1125_112580


namespace NUMINAMATH_GPT_range_of_a_l1125_112518

theorem range_of_a (a : ℝ) (h : ∅ ⊂ {x : ℝ | x^2 ≤ a}) : 0 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1125_112518


namespace NUMINAMATH_GPT_maximum_value_abs_difference_l1125_112551

theorem maximum_value_abs_difference (x y : ℝ) 
  (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : 
  |x - y + 1| ≤ 2 :=
sorry

end NUMINAMATH_GPT_maximum_value_abs_difference_l1125_112551


namespace NUMINAMATH_GPT_mary_circus_change_l1125_112555

theorem mary_circus_change :
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  change = 15 :=
by
  let mary_ticket := 2
  let child_ticket := 1
  let num_children := 3
  let total_cost := mary_ticket + num_children * child_ticket
  let amount_paid := 20
  let change := amount_paid - total_cost
  sorry

end NUMINAMATH_GPT_mary_circus_change_l1125_112555


namespace NUMINAMATH_GPT_num_chickens_is_one_l1125_112544

-- Define the number of dogs and the number of total legs
def num_dogs := 2
def total_legs := 10

-- Define the number of legs per dog and per chicken
def legs_per_dog := 4
def legs_per_chicken := 2

-- Define the number of chickens
def num_chickens := (total_legs - num_dogs * legs_per_dog) / legs_per_chicken

-- Prove that the number of chickens is 1
theorem num_chickens_is_one : num_chickens = 1 := by
  -- This is the proof placeholder
  sorry

end NUMINAMATH_GPT_num_chickens_is_one_l1125_112544


namespace NUMINAMATH_GPT_find_m_l1125_112505

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end NUMINAMATH_GPT_find_m_l1125_112505


namespace NUMINAMATH_GPT_water_needed_quarts_l1125_112590

-- Definitions from conditions
def ratio_water : ℕ := 8
def ratio_lemon : ℕ := 1
def total_gallons : ℚ := 1.5
def gallons_to_quarts : ℚ := 4

-- State what needs to be proven
theorem water_needed_quarts : 
  (total_gallons * gallons_to_quarts * (ratio_water / (ratio_water + ratio_lemon))) = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_water_needed_quarts_l1125_112590


namespace NUMINAMATH_GPT_new_assistant_draw_time_l1125_112548

-- Definitions based on conditions
def capacity : ℕ := 36
def halfway : ℕ := capacity / 2
def rate_top : ℕ := 1 / 6
def rate_bottom : ℕ := 1 / 4
def extra_time : ℕ := 24

-- The proof statement
theorem new_assistant_draw_time : 
  ∃ t : ℕ, ((capacity - (extra_time * rate_bottom * 1)) - halfway) = (t * rate_bottom * 1) ∧ t = 48 := by
sorry

end NUMINAMATH_GPT_new_assistant_draw_time_l1125_112548


namespace NUMINAMATH_GPT_Mikail_birthday_money_l1125_112594

theorem Mikail_birthday_money (x : ℕ) (h1 : x = 3 + 3 * 3) : 5 * x = 60 := 
by 
  sorry

end NUMINAMATH_GPT_Mikail_birthday_money_l1125_112594


namespace NUMINAMATH_GPT_divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l1125_112573

theorem divides_2_pow_n_sub_1 (n : ℕ) : 7 ∣ (2 ^ n - 1) ↔ 3 ∣ n := by
  sorry

theorem no_n_divides_2_pow_n_add_1 (n : ℕ) : ¬ 7 ∣ (2 ^ n + 1) := by
  sorry

end NUMINAMATH_GPT_divides_2_pow_n_sub_1_no_n_divides_2_pow_n_add_1_l1125_112573


namespace NUMINAMATH_GPT_divisors_congruent_mod8_l1125_112510

theorem divisors_congruent_mod8 (n : ℕ) (hn : n % 2 = 1) :
  ∀ d, d ∣ (2^n - 1) → d % 8 = 1 ∨ d % 8 = 7 :=
by
  sorry

end NUMINAMATH_GPT_divisors_congruent_mod8_l1125_112510


namespace NUMINAMATH_GPT_number_of_common_points_l1125_112575

-- Define the circle equation
def is_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

-- Define the vertical line equation
def is_on_line (x : ℝ) : Prop :=
  x = 3

-- Prove that the number of distinct points common to both graphs is two
theorem number_of_common_points : 
  ∃ y1 y2 : ℝ, is_on_circle 3 y1 ∧ is_on_circle 3 y2 ∧ y1 ≠ y2 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_common_points_l1125_112575


namespace NUMINAMATH_GPT_quadratic_inequality_m_range_l1125_112500

theorem quadratic_inequality_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ (m ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_m_range_l1125_112500


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1125_112579

theorem quadratic_no_real_roots (m : ℝ) : (∀ x, x^2 - 2 * x + m ≠ 0) ↔ m > 1 := 
by sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1125_112579


namespace NUMINAMATH_GPT_compound_interest_time_period_l1125_112591

theorem compound_interest_time_period (P r I : ℝ) (n A t : ℝ) 
(hP : P = 6000) 
(hr : r = 0.10) 
(hI : I = 1260.000000000001) 
(hn : n = 1)
(hA : A = P + I)
(ht_eqn: (A / P) = (1 + r / n) ^ t) :
t = 2 := 
by sorry

end NUMINAMATH_GPT_compound_interest_time_period_l1125_112591


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1125_112523

theorem solution_set_of_inequality:
  {x : ℝ | 3 ≤ |2 - x| ∧ |2 - x| < 9} = {x : ℝ | (-7 < x ∧ x ≤ -1) ∨ (5 ≤ x ∧ x < 11)} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1125_112523


namespace NUMINAMATH_GPT_pages_wed_calculation_l1125_112503

def pages_mon : ℕ := 23
def pages_tue : ℕ := 38
def pages_thu : ℕ := 12
def pages_fri : ℕ := 2 * pages_thu
def total_pages : ℕ := 158

theorem pages_wed_calculation (pages_wed : ℕ) : 
  pages_mon + pages_tue + pages_wed + pages_thu + pages_fri = total_pages → pages_wed = 61 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_pages_wed_calculation_l1125_112503


namespace NUMINAMATH_GPT_probability_first_grade_probability_at_least_one_second_grade_l1125_112536

-- Define conditions
def total_products : ℕ := 10
def first_grade_products : ℕ := 8
def second_grade_products : ℕ := 2
def inspected_products : ℕ := 2
def total_combinations : ℕ := Nat.choose total_products inspected_products
def first_grade_combinations : ℕ := Nat.choose first_grade_products inspected_products
def mixed_combinations : ℕ := first_grade_products * second_grade_products
def second_grade_combinations : ℕ := Nat.choose second_grade_products inspected_products

-- Define probabilities
def P_A : ℚ := first_grade_combinations / total_combinations
def P_B1 : ℚ := mixed_combinations / total_combinations
def P_B2 : ℚ := second_grade_combinations / total_combinations
def P_B : ℚ := P_B1 + P_B2

-- Statements
theorem probability_first_grade : P_A = 28 / 45 := sorry
theorem probability_at_least_one_second_grade : P_B = 17 / 45 := sorry

end NUMINAMATH_GPT_probability_first_grade_probability_at_least_one_second_grade_l1125_112536


namespace NUMINAMATH_GPT_correct_quotient_is_48_l1125_112541

theorem correct_quotient_is_48 (dividend : ℕ) (incorrect_divisor : ℕ) (incorrect_quotient : ℕ) (correct_divisor : ℕ) (correct_quotient : ℕ) :
  incorrect_divisor = 72 → 
  incorrect_quotient = 24 → 
  correct_divisor = 36 →
  dividend = incorrect_divisor * incorrect_quotient →
  correct_quotient = dividend / correct_divisor →
  correct_quotient = 48 :=
by
  sorry

end NUMINAMATH_GPT_correct_quotient_is_48_l1125_112541


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l1125_112524

theorem trajectory_of_midpoint (x y x₀ y₀ : ℝ) :
  (y₀ = 2 * x₀ ^ 2 + 1) ∧ (x = (x₀ + 0) / 2) ∧ (y = (y₀ + 1) / 2) →
  y = 4 * x ^ 2 + 1 :=
by sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l1125_112524


namespace NUMINAMATH_GPT_meetings_percent_l1125_112515

/-- Define the lengths of the meetings and total workday in minutes -/
def first_meeting : ℕ := 40
def second_meeting : ℕ := 80
def second_meeting_overlap : ℕ := 10
def third_meeting : ℕ := 30
def workday_minutes : ℕ := 8 * 60

/-- Define the effective duration of the second meeting -/
def effective_second_meeting : ℕ := second_meeting - second_meeting_overlap

/-- Define the total time spent in meetings -/
def total_meeting_time : ℕ := first_meeting + effective_second_meeting + third_meeting

/-- Define the percentage of the workday spent in meetings -/
noncomputable def percent_meeting_time : ℚ := (total_meeting_time * 100 : ℕ) / workday_minutes

/-- Theorem: Given Laura's workday and meeting durations, prove that the percent of her workday spent in meetings is approximately 29.17%. -/
theorem meetings_percent {epsilon : ℚ} (h : epsilon = 0.01) : abs (percent_meeting_time - 29.17) < epsilon :=
sorry

end NUMINAMATH_GPT_meetings_percent_l1125_112515


namespace NUMINAMATH_GPT_democrats_ratio_l1125_112547

variable (F M D_F D_M TotalParticipants : ℕ)

-- Assume the following conditions
variables (H1 : F + M = 660)
variables (H2 : D_F = 1 / 2 * F)
variables (H3 : D_F = 110)
variables (H4 : D_M = 1 / 4 * M)
variables (H5 : TotalParticipants = 660)

theorem democrats_ratio 
  (H1 : F + M = 660)
  (H2 : D_F = 1 / 2 * F)
  (H3 : D_F = 110)
  (H4 : D_M = 1 / 4 * M)
  (H5 : TotalParticipants = 660) :
  (D_F + D_M) / TotalParticipants = 1 / 3
:= 
  sorry

end NUMINAMATH_GPT_democrats_ratio_l1125_112547


namespace NUMINAMATH_GPT_maya_total_pages_read_l1125_112553

def last_week_books : ℕ := 5
def pages_per_book : ℕ := 300
def this_week_multiplier : ℕ := 2

theorem maya_total_pages_read : 
  (last_week_books * pages_per_book * (1 + this_week_multiplier)) = 4500 :=
by
  sorry

end NUMINAMATH_GPT_maya_total_pages_read_l1125_112553


namespace NUMINAMATH_GPT_find_common_difference_l1125_112561

noncomputable def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 4 = 7 ∧ a 3 + a 6 = 16

theorem find_common_difference (a : ℕ → ℝ) (d : ℝ) (h : common_difference a d) : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_difference_l1125_112561


namespace NUMINAMATH_GPT_counterexample_to_prime_condition_l1125_112517

theorem counterexample_to_prime_condition :
  ¬(Prime 54) ∧ ¬(Prime 52) ∧ ¬(Prime 51) := by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_counterexample_to_prime_condition_l1125_112517


namespace NUMINAMATH_GPT_find_y_payment_l1125_112574

-- Definitions for the conditions in the problem
def total_payment (X Y : ℝ) : Prop := X + Y = 560
def x_is_120_percent_of_y (X Y : ℝ) : Prop := X = 1.2 * Y

-- Problem statement converted to a Lean proof problem
theorem find_y_payment (X Y : ℝ) (h1 : total_payment X Y) (h2 : x_is_120_percent_of_y X Y) : Y = 255 := 
by sorry

end NUMINAMATH_GPT_find_y_payment_l1125_112574


namespace NUMINAMATH_GPT_part1_profit_in_april_part2_price_reduction_l1125_112535

-- Given conditions
def cost_per_bag : ℕ := 16
def original_price_per_bag : ℕ := 30
def reduction_amount : ℕ := 5
def increase_in_sales_rate : ℕ := 20
def original_sales_volume : ℕ := 200
def target_profit : ℕ := 2860

-- Part 1: When the price per bag of noodles is reduced by 5 yuan
def profit_in_april_when_reduced_by_5 (cost_per_bag original_price_per_bag reduction_amount increase_in_sales_rate original_sales_volume : ℕ) : ℕ := 
  let new_price := original_price_per_bag - reduction_amount
  let new_sales_volume := original_sales_volume + (increase_in_sales_rate * reduction_amount)
  let profit_per_bag := new_price - cost_per_bag
  profit_per_bag * new_sales_volume

theorem part1_profit_in_april :
  profit_in_april_when_reduced_by_5 16 30 5 20 200 = 2700 :=
sorry

-- Part 2: Determine the price reduction for a specific target profit
def price_reduction_for_profit (cost_per_bag original_price_per_bag increase_in_sales_rate original_sales_volume target_profit : ℕ) : ℕ :=
  let x := (target_profit - (original_sales_volume * (original_price_per_bag - cost_per_bag))) / (increase_in_sales_rate * (original_price_per_bag - cost_per_bag) - increase_in_sales_rate - original_price_per_bag)
  x

theorem part2_price_reduction :
  price_reduction_for_profit 16 30 20 200 2860 = 3 :=
sorry

end NUMINAMATH_GPT_part1_profit_in_april_part2_price_reduction_l1125_112535


namespace NUMINAMATH_GPT_polyhedron_calculation_l1125_112520

def faces := 32
def triangular := 10
def pentagonal := 8
def hexagonal := 14
def edges := 79
def vertices := 49
def T := 1
def P := 2

theorem polyhedron_calculation : 
  100 * P + 10 * T + vertices = 249 := 
sorry

end NUMINAMATH_GPT_polyhedron_calculation_l1125_112520


namespace NUMINAMATH_GPT_moles_of_NaCl_l1125_112546

def moles_of_reactants (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

theorem moles_of_NaCl (NaCl KNO3 NaNO3 KCl : ℕ) 
  (h : moles_of_reactants NaCl KNO3 NaNO3 KCl) 
  (h2 : KNO3 = 1)
  (h3 : NaNO3 = 1) :
  NaCl = 1 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_NaCl_l1125_112546


namespace NUMINAMATH_GPT_part1_l1125_112519

variable (a b c : ℝ) (A B : ℝ)
variable (triangle_abc : Triangle ABC)
variable (cos : ℝ → ℝ)

axiom law_of_cosines : ∀ {a b c A : ℝ}, a^2 = b^2 + c^2 - 2 * b * c * cos A

theorem part1 (h1 : b^2 + 3 * a * c * (a^2 + c^2 - b^2) / (2 * a * c) = 2 * c^2) (h2 : a = c) : A = π / 4 := 
sorry

end NUMINAMATH_GPT_part1_l1125_112519


namespace NUMINAMATH_GPT_sampling_probabilities_equal_l1125_112571

-- Definitions according to the problem conditions
def population_size := ℕ
def sample_size := ℕ
def simple_random_sampling (N n : ℕ) : Prop := sorry
def systematic_sampling (N n : ℕ) : Prop := sorry
def stratified_sampling (N n : ℕ) : Prop := sorry

-- Probabilities
def P1 : ℝ := sorry -- Probability for simple random sampling
def P2 : ℝ := sorry -- Probability for systematic sampling
def P3 : ℝ := sorry -- Probability for stratified sampling

-- Each definition directly corresponds to a condition in the problem statement.
-- Now, we summarize the equivalent proof problem in Lean.

theorem sampling_probabilities_equal (N n : ℕ) (h1 : simple_random_sampling N n) (h2 : systematic_sampling N n) (h3 : stratified_sampling N n) :
  P1 = P2 ∧ P2 = P3 :=
by sorry

end NUMINAMATH_GPT_sampling_probabilities_equal_l1125_112571


namespace NUMINAMATH_GPT_series_remainder_is_zero_l1125_112534

theorem series_remainder_is_zero :
  let a : ℕ := 4
  let d : ℕ := 6
  let n : ℕ := 17
  let l : ℕ := a + d * (n - 1) -- last term
  let S : ℕ := n * (a + l) / 2 -- sum of the series
  S % 17 = 0 := by
  sorry

end NUMINAMATH_GPT_series_remainder_is_zero_l1125_112534


namespace NUMINAMATH_GPT_probability_correct_guesses_l1125_112538

theorem probability_correct_guesses:
  let p_wrong := (5/6 : ℚ)
  let p_miss_all := p_wrong ^ 5
  let p_at_least_one_correct := 1 - p_miss_all
  p_at_least_one_correct = 4651/7776 := by
  sorry

end NUMINAMATH_GPT_probability_correct_guesses_l1125_112538


namespace NUMINAMATH_GPT_simplify_expression_l1125_112507

theorem simplify_expression (x : ℤ) : (3 * x) ^ 3 + (2 * x) * (x ^ 4) = 27 * x ^ 3 + 2 * x ^ 5 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1125_112507


namespace NUMINAMATH_GPT_orthogonal_lines_solution_l1125_112556

theorem orthogonal_lines_solution (a b c d : ℝ)
  (h1 : b - a = 0)
  (h2 : c - a = 2)
  (h3 : 12 * d - a = 1)
  : d = 3 / 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_orthogonal_lines_solution_l1125_112556


namespace NUMINAMATH_GPT_probability_escher_consecutive_l1125_112558

def total_pieces : Nat := 12
def escher_pieces : Nat := 4

theorem probability_escher_consecutive :
  (Nat.factorial 9 * Nat.factorial 4 : ℚ) / Nat.factorial 12 = 1 / 55 := 
sorry

end NUMINAMATH_GPT_probability_escher_consecutive_l1125_112558


namespace NUMINAMATH_GPT_employee_B_payment_l1125_112593

theorem employee_B_payment (x : ℝ) (h1 : ∀ A B : ℝ, A + B = 580) (h2 : A = 1.5 * B) : B = 232 :=
by
  sorry

end NUMINAMATH_GPT_employee_B_payment_l1125_112593


namespace NUMINAMATH_GPT_book_sale_revenue_l1125_112540

noncomputable def total_amount_received (price_per_book : ℝ) (B : ℕ) (sold_fraction : ℝ) :=
  sold_fraction * B * price_per_book

theorem book_sale_revenue (B : ℕ) (price_per_book : ℝ) (unsold_books : ℕ) (sold_fraction : ℝ) :
  (1 / 3 : ℝ) * B = unsold_books →
  price_per_book = 3.50 →
  unsold_books = 36 →
  sold_fraction = 2 / 3 →
  total_amount_received price_per_book B sold_fraction = 252 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_book_sale_revenue_l1125_112540


namespace NUMINAMATH_GPT_percentage_of_y_l1125_112568

theorem percentage_of_y (y : ℝ) (h : y > 0) : (9 * y) / 20 + (3 * y) / 10 = 0.75 * y :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_y_l1125_112568


namespace NUMINAMATH_GPT_find_pairs_of_positive_numbers_l1125_112567

theorem find_pairs_of_positive_numbers
  (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (exists_triangle : ∃ (C D E A B : ℝ), true)
  (points_on_hypotenuse : ∀ (C D E A B : ℝ), A ∈ [D, E] ∧ B ∈ [D, E]) 
  (equal_vectors : ∀ (D A B E : ℝ), (D - A) = (A - B) ∧ (A - B) = (B - E))
  (AC_eq_a : (C - A) = a)
  (BC_eq_b : (C - B) = b) :
  (1 / 2) < (a / b) ∧ (a / b) < 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_pairs_of_positive_numbers_l1125_112567


namespace NUMINAMATH_GPT_digit_7_occurrences_in_range_20_to_199_l1125_112584

open Set

noncomputable def countDigitOccurrences (low high : ℕ) (digit : ℕ) : ℕ :=
  sorry

theorem digit_7_occurrences_in_range_20_to_199 : 
  countDigitOccurrences 20 199 7 = 38 := 
by
  sorry

end NUMINAMATH_GPT_digit_7_occurrences_in_range_20_to_199_l1125_112584


namespace NUMINAMATH_GPT_expression_for_an_l1125_112597

noncomputable def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  2 + (n - 1) * d

theorem expression_for_an (d : ℕ) (n : ℕ) 
  (h1 : d > 0)
  (h2 : (arithmetic_sequence d 1) = 2)
  (h3 : (arithmetic_sequence d 1) < (arithmetic_sequence d 2))
  (h4 : (arithmetic_sequence d 2)^2 = 2 * (arithmetic_sequence d 4)) :
  arithmetic_sequence d n = 2 * n := sorry

end NUMINAMATH_GPT_expression_for_an_l1125_112597


namespace NUMINAMATH_GPT_smaller_angle_at_3_15_l1125_112525

theorem smaller_angle_at_3_15 
  (hours_on_clock : ℕ := 12) 
  (degree_per_hour : ℝ := 360 / hours_on_clock) 
  (minute_hand_position : ℝ := 3) 
  (hour_progress_per_minute : ℝ := 1 / 60 * degree_per_hour) : 
  ∃ angle : ℝ, angle = 7.5 := by
  let hour_hand_position := 3 + (15 * hour_progress_per_minute)
  let angle_diff := abs (minute_hand_position * degree_per_hour - hour_hand_position)
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  use smaller_angle
  sorry

end NUMINAMATH_GPT_smaller_angle_at_3_15_l1125_112525


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1125_112557

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 5 + a 6 + a 7 = 1) : a 3 + a 9 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1125_112557


namespace NUMINAMATH_GPT_initial_number_of_boarders_l1125_112587

theorem initial_number_of_boarders (B D : ℕ) (h1 : B / D = 2 / 5) (h2 : (B + 15) / D = 1 / 2) : B = 60 :=
by
  -- Proof needs to be provided here
  sorry

end NUMINAMATH_GPT_initial_number_of_boarders_l1125_112587


namespace NUMINAMATH_GPT_fraction_difference_is_correct_l1125_112528

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end NUMINAMATH_GPT_fraction_difference_is_correct_l1125_112528


namespace NUMINAMATH_GPT_max_full_pikes_l1125_112599

theorem max_full_pikes (initial_pikes : ℕ) (pike_full_condition : ℕ → Prop) (remaining_pikes : ℕ) 
  (h_initial : initial_pikes = 30)
  (h_condition : ∀ n, pike_full_condition n → n ≥ 3)
  (h_remaining : remaining_pikes ≥ 1) :
    ∃ max_full : ℕ, max_full ≤ 9 := 
sorry

end NUMINAMATH_GPT_max_full_pikes_l1125_112599


namespace NUMINAMATH_GPT_star_value_when_c_2_d_3_l1125_112598

def star (c d : ℕ) : ℕ := c^3 + 3*c^2*d + 3*c*d^2 + d^3

theorem star_value_when_c_2_d_3 :
  star 2 3 = 125 :=
by
  sorry

end NUMINAMATH_GPT_star_value_when_c_2_d_3_l1125_112598


namespace NUMINAMATH_GPT_arithmetic_mean_18_27_45_l1125_112589

theorem arithmetic_mean_18_27_45 : (18 + 27 + 45) / 3 = 30 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_18_27_45_l1125_112589


namespace NUMINAMATH_GPT_LCM_of_fractions_l1125_112592

noncomputable def LCM (a b : Rat) : Rat :=
  a * b / (gcd a.num b.num / gcd a.den b.den : Int)

theorem LCM_of_fractions (x : ℤ) (h : x ≠ 0) :
  LCM (1 / (4 * x : ℚ)) (LCM (1 / (6 * x : ℚ)) (1 / (9 * x : ℚ))) = 1 / (36 * x) :=
by
  sorry

end NUMINAMATH_GPT_LCM_of_fractions_l1125_112592


namespace NUMINAMATH_GPT_greatest_length_of_cords_l1125_112569

theorem greatest_length_of_cords (a b c : ℝ) (h₁ : a = Real.sqrt 20) (h₂ : b = Real.sqrt 50) (h₃ : c = Real.sqrt 98) :
  ∃ (d : ℝ), d = 1 ∧ ∀ (k : ℝ), (k = a ∨ k = b ∨ k = c) → ∃ (n m : ℕ), k = d * (n : ℝ) ∧ d * (m : ℝ) = (m : ℝ) := by
sorry

end NUMINAMATH_GPT_greatest_length_of_cords_l1125_112569


namespace NUMINAMATH_GPT_length_MN_of_circle_l1125_112512

def point := ℝ × ℝ

def circle_passing_through (A B C: point) :=
  ∃ (D E F : ℝ), ∀ (p : point), p = A ∨ p = B ∨ p = C →
    (p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0)

theorem length_MN_of_circle (A B C : point) (H : circle_passing_through A B C) :
  A = (1, 3) → B = (4, 2) → C = (1, -7) →
  ∃ M N : ℝ, (A.1 * 0 + N^2 + D * 0 + E * N + F = 0) ∧ (A.1 * 0 + M^2 + D * 0 + E * M + F = 0) ∧
  abs (M - N) = 4 * Real.sqrt 6 := 
sorry

end NUMINAMATH_GPT_length_MN_of_circle_l1125_112512


namespace NUMINAMATH_GPT_simplify_expression_l1125_112549

variable (x : ℝ)

theorem simplify_expression : (5 * x + 2 * (4 + x)) = (7 * x + 8) := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1125_112549


namespace NUMINAMATH_GPT_sugar_amount_l1125_112564

theorem sugar_amount (S F B : ℝ) 
    (h_ratio1 : S = F) 
    (h_ratio2 : F = 10 * B) 
    (h_ratio3 : F / (B + 60) = 8) : S = 2400 := 
by
  sorry

end NUMINAMATH_GPT_sugar_amount_l1125_112564


namespace NUMINAMATH_GPT_percentage_reduction_l1125_112581

-- Define the problem within given conditions
def original_length := 30 -- original length in seconds
def new_length := 21 -- new length in seconds

-- State the theorem that needs to be proved
theorem percentage_reduction (original_length new_length : ℕ) : 
  original_length = 30 → 
  new_length = 21 → 
  ((original_length - new_length) / original_length: ℚ) * 100 = 30 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_reduction_l1125_112581


namespace NUMINAMATH_GPT_digits_of_2_120_l1125_112514

theorem digits_of_2_120 (h : ∀ n : ℕ, (10 : ℝ)^(n - 1) ≤ (2 : ℝ)^200 ∧ (2 : ℝ)^200 < (10 : ℝ)^n → n = 61) :
  ∀ m : ℕ, (10 : ℝ)^(m - 1) ≤ (2 : ℝ)^120 ∧ (2 : ℝ)^120 < (10 : ℝ)^m → m = 37 :=
by
  sorry

end NUMINAMATH_GPT_digits_of_2_120_l1125_112514


namespace NUMINAMATH_GPT_find_k_for_equation_l1125_112539

theorem find_k_for_equation : 
  ∃ k : ℤ, -x^2 - (k + 7) * x - 8 = -(x - 2) * (x - 4) → k = -13 := 
by
  sorry

end NUMINAMATH_GPT_find_k_for_equation_l1125_112539


namespace NUMINAMATH_GPT_dilation_0_minus_2i_to_neg3_minus_14i_l1125_112526

open Complex

def dilation_centered (z_center z zk : ℂ) (factor : ℝ) : ℂ :=
  z_center + factor * (zk - z_center)

theorem dilation_0_minus_2i_to_neg3_minus_14i :
  dilation_centered (1 + 2 * I) (0 - 2 * I) (1 + 2 * I) 4 = -3 - 14 * I :=
by
  sorry

end NUMINAMATH_GPT_dilation_0_minus_2i_to_neg3_minus_14i_l1125_112526


namespace NUMINAMATH_GPT_how_many_cubes_needed_l1125_112559

def cube_volume (side_len : ℕ) : ℕ :=
  side_len ^ 3

theorem how_many_cubes_needed (Vsmall Vlarge Vsmall_cube num_small_cubes : ℕ) 
  (h1 : Vsmall = cube_volume 8) 
  (h2 : Vlarge = cube_volume 12) 
  (h3 : Vsmall_cube = cube_volume 2) 
  (h4 : num_small_cubes = (Vlarge - Vsmall) / Vsmall_cube) :
  num_small_cubes = 152 :=
by
  sorry

end NUMINAMATH_GPT_how_many_cubes_needed_l1125_112559
