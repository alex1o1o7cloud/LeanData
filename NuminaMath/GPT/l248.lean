import Mathlib

namespace NUMINAMATH_GPT_combined_tax_rate_l248_24818

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 3 * Mork_income)
  (tax_Mork tax_Mindy : ℝ) (h2 : tax_Mork = 0.10 * Mork_income) (h3 : tax_Mindy = 0.20 * Mindy_income)
  : (tax_Mork + tax_Mindy) / (Mork_income + Mindy_income) = 0.175 :=
by
  sorry

end NUMINAMATH_GPT_combined_tax_rate_l248_24818


namespace NUMINAMATH_GPT_isabella_more_than_sam_l248_24872

variable (I S G : ℕ)

def Giselle_money : G = 120 := by sorry
def Isabella_more_than_Giselle : I = G + 15 := by sorry
def total_donation : I + S + G = 345 := by sorry

theorem isabella_more_than_sam : I - S = 45 := by
sorry

end NUMINAMATH_GPT_isabella_more_than_sam_l248_24872


namespace NUMINAMATH_GPT_days_with_equal_sun_tue_l248_24870

theorem days_with_equal_sun_tue (days_in_month : ℕ) (weekdays : ℕ) (d1 d2 : ℕ) (h1 : days_in_month = 30)
  (h2 : weekdays = 7) (h3 : d1 = 4) (h4 : d2 = 2) :
  ∃ count, count = 3 := by
  sorry

end NUMINAMATH_GPT_days_with_equal_sun_tue_l248_24870


namespace NUMINAMATH_GPT_find_a_from_inclination_l248_24828

open Real

theorem find_a_from_inclination (a : ℝ) :
  (∃ (k : ℝ), k = (2 - (-3)) / (1 - a) ∧ k = tan (135 * pi / 180)) → a = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a_from_inclination_l248_24828


namespace NUMINAMATH_GPT_max_gcd_sequence_l248_24890

noncomputable def a (n : ℕ) : ℕ := n^3 + 4
noncomputable def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_sequence : (∀ n : ℕ, 0 < n → d n ≤ 433) ∧ (∃ n : ℕ, 0 < n ∧ d n = 433) :=
by sorry

end NUMINAMATH_GPT_max_gcd_sequence_l248_24890


namespace NUMINAMATH_GPT_total_handshakes_l248_24839

-- There are 5 members on each of the two basketball teams.
def teamMembers : Nat := 5

-- There are 2 referees.
def referees : Nat := 2

-- Each player from one team shakes hands with each player from the other team.
def handshakesBetweenTeams : Nat := teamMembers * teamMembers

-- Each player shakes hands with each referee.
def totalPlayers : Nat := 2 * teamMembers
def handshakesWithReferees : Nat := totalPlayers * referees

-- Prove that the total number of handshakes is 45.
theorem total_handshakes : handshakesBetweenTeams + handshakesWithReferees = 45 := by
  -- Total handshakes is the sum of handshakes between teams and handshakes with referees.
  sorry

end NUMINAMATH_GPT_total_handshakes_l248_24839


namespace NUMINAMATH_GPT_range_of_abs_function_l248_24836

theorem range_of_abs_function:
  (∀ y, ∃ x : ℝ, y = |x + 3| - |x - 5|) → ∀ y, y ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_abs_function_l248_24836


namespace NUMINAMATH_GPT_watch_cost_price_l248_24896

open Real

theorem watch_cost_price (CP SP1 SP2 : ℝ)
    (h1 : SP1 = CP * 0.85)
    (h2 : SP2 = CP * 1.10)
    (h3 : SP2 = SP1 + 450) : CP = 1800 :=
by
  sorry

end NUMINAMATH_GPT_watch_cost_price_l248_24896


namespace NUMINAMATH_GPT_max_xyz_l248_24845

theorem max_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2) 
(h5 : x^2 + y^2 + z^2 = x * z + y * z + x * y) : xyz ≤ (8 / 27) :=
sorry

end NUMINAMATH_GPT_max_xyz_l248_24845


namespace NUMINAMATH_GPT_trinomials_real_roots_inequality_l248_24884

theorem trinomials_real_roots_inequality :
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ¬ (∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q))) >
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q)) :=
sorry

end NUMINAMATH_GPT_trinomials_real_roots_inequality_l248_24884


namespace NUMINAMATH_GPT_solution_to_abs_eq_l248_24892

theorem solution_to_abs_eq :
  ∀ x : ℤ, abs ((-5) + x) = 11 → (x = 16 ∨ x = -6) :=
by sorry

end NUMINAMATH_GPT_solution_to_abs_eq_l248_24892


namespace NUMINAMATH_GPT_correct_sampling_methods_l248_24831

-- Define the surveys with their corresponding conditions
structure Survey1 where
  high_income : Nat
  middle_income : Nat
  low_income : Nat
  total_households : Nat

structure Survey2 where
  total_students : Nat
  sample_students : Nat
  differences_small : Bool
  sizes_small : Bool

-- Define the conditions
def survey1_conditions (s : Survey1) : Prop :=
  s.high_income = 125 ∧ s.middle_income = 280 ∧ s.low_income = 95 ∧ s.total_households = 100

def survey2_conditions (s : Survey2) : Prop :=
  s.total_students = 15 ∧ s.sample_students = 3 ∧ s.differences_small = true ∧ s.sizes_small = true

-- Define the answer predicate
def correct_answer (method1 method2 : String) : Prop :=
  method1 = "stratified sampling" ∧ method2 = "simple random sampling"

-- The theorem statement
theorem correct_sampling_methods (s1 : Survey1) (s2 : Survey2) :
  survey1_conditions s1 → survey2_conditions s2 → correct_answer "stratified sampling" "simple random sampling" :=
by
  -- Proof skipped for problem statement purpose
  sorry

end NUMINAMATH_GPT_correct_sampling_methods_l248_24831


namespace NUMINAMATH_GPT_adult_ticket_cost_l248_24867

-- Definitions based on given conditions.
def children_ticket_cost : ℝ := 7.5
def total_bill : ℝ := 138
def total_tickets : ℕ := 12
def additional_children_tickets : ℕ := 8

-- Proof statement: Prove the cost of each adult ticket.
theorem adult_ticket_cost (x : ℕ) (A : ℝ)
  (h1 : x + (x + additional_children_tickets) = total_tickets)
  (h2 : x * A + (x + additional_children_tickets) * children_ticket_cost = total_bill) :
  A = 31.50 :=
  sorry

end NUMINAMATH_GPT_adult_ticket_cost_l248_24867


namespace NUMINAMATH_GPT_area_of_shaded_rectangle_l248_24879

-- Definition of side length of the squares
def side_length : ℕ := 12

-- Definition of the dimensions of the overlapped rectangle
def rectangle_length : ℕ := 20
def rectangle_width : ℕ := side_length

-- Theorem stating the area of the shaded rectangle PBCS
theorem area_of_shaded_rectangle
  (squares_identical : ∀ (a b c d p q r s : ℕ),
    a = side_length → b = side_length →
    p = side_length → q = side_length →
    rectangle_width * (rectangle_length - side_length) = 48) :
  rectangle_width * (rectangle_length - side_length) = 48 :=
by sorry -- Proof omitted

end NUMINAMATH_GPT_area_of_shaded_rectangle_l248_24879


namespace NUMINAMATH_GPT_increase_in_daily_mess_expenses_l248_24893

theorem increase_in_daily_mess_expenses (A X : ℝ)
  (h1 : 35 * A = 420)
  (h2 : 42 * (A - 1) = 420 + X) :
  X = 42 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_daily_mess_expenses_l248_24893


namespace NUMINAMATH_GPT_fibonacci_arithmetic_sequence_l248_24858

def fibonacci : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_arithmetic_sequence (a b c n : ℕ) 
  (h1 : fibonacci 1 = 1)
  (h2 : fibonacci 2 = 1)
  (h3 : ∀ n ≥ 3, fibonacci n = fibonacci (n - 1) + fibonacci (n - 2))
  (h4 : a + b + c = 2500)
  (h5 : (a, b, c) = (n, n + 3, n + 5)) :
  a = 831 := 
sorry

end NUMINAMATH_GPT_fibonacci_arithmetic_sequence_l248_24858


namespace NUMINAMATH_GPT_scientific_notation_of_0_0000021_l248_24852

theorem scientific_notation_of_0_0000021 :
  0.0000021 = 2.1 * 10 ^ (-6) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_0_0000021_l248_24852


namespace NUMINAMATH_GPT_freight_capacity_equation_l248_24899

theorem freight_capacity_equation
  (x : ℝ)
  (h1 : ∀ (capacity_large capacity_small : ℝ), capacity_large = capacity_small + 4)
  (h2 : ∀ (n_large n_small : ℕ), (n_large : ℝ) = 80 / (x + 4) ∧ (n_small : ℝ) = 60 / x → n_large = n_small) :
  (80 / (x + 4)) = (60 / x) :=
by
  sorry

end NUMINAMATH_GPT_freight_capacity_equation_l248_24899


namespace NUMINAMATH_GPT_relay_team_orders_l248_24806

noncomputable def jordan_relay_orders : Nat :=
  let friends := [1, 2, 3] -- Differentiate friends; let's represent A by 1, B by 2, C by 3
  let choices_for_jordan_third := 2 -- Ways if Jordan runs third
  let choices_for_jordan_fourth := 2 -- Ways if Jordan runs fourth
  choices_for_jordan_third + choices_for_jordan_fourth

theorem relay_team_orders :
  jordan_relay_orders = 4 :=
by
  sorry

end NUMINAMATH_GPT_relay_team_orders_l248_24806


namespace NUMINAMATH_GPT_river_ratio_l248_24809

theorem river_ratio (total_length straight_length crooked_length : ℕ) 
  (h1 : total_length = 80) (h2 : straight_length = 20) 
  (h3 : crooked_length = total_length - straight_length) : 
  (straight_length / Nat.gcd straight_length crooked_length) = 1 ∧ (crooked_length / Nat.gcd straight_length crooked_length) = 3 := 
by
  sorry

end NUMINAMATH_GPT_river_ratio_l248_24809


namespace NUMINAMATH_GPT_math_problem_l248_24851

theorem math_problem : (300 + 5 * 8) / (2^3) = 42.5 := by
  sorry

end NUMINAMATH_GPT_math_problem_l248_24851


namespace NUMINAMATH_GPT_smallest_a_l248_24820

theorem smallest_a (a : ℕ) (h1 : a > 0) (h2 : (∀ b : ℕ, b > 0 → b < a → ∀ h3 : b > 0, ¬ (gcd b 72 > 1 ∧ gcd b 90 > 1)))
  (h3 : gcd a 72 > 1) (h4 : gcd a 90 > 1) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_l248_24820


namespace NUMINAMATH_GPT_total_servings_l248_24855

-- Definitions for the conditions

def servings_per_carrot : ℕ := 4
def plants_per_plot : ℕ := 9
def servings_multiplier_corn : ℕ := 5
def servings_multiplier_green_bean : ℤ := 2

-- Proof statement
theorem total_servings : 
  (plants_per_plot * servings_per_carrot) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn)) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn / servings_multiplier_green_bean)) = 
  306 :=
by
  sorry

end NUMINAMATH_GPT_total_servings_l248_24855


namespace NUMINAMATH_GPT_find_a7_of_arithmetic_sequence_l248_24807

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + d * (n - 1)

theorem find_a7_of_arithmetic_sequence (a d : ℤ)
  (h : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 +
       arithmetic_sequence a d 12 + arithmetic_sequence a d 13 = 24) :
  arithmetic_sequence a d 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a7_of_arithmetic_sequence_l248_24807


namespace NUMINAMATH_GPT_find_constant_l248_24801

theorem find_constant (n : ℤ) (c : ℝ) (h1 : ∀ n ≤ 10, c * (n : ℝ)^2 ≤ 12100) : c ≤ 121 :=
sorry

end NUMINAMATH_GPT_find_constant_l248_24801


namespace NUMINAMATH_GPT_closest_to_zero_is_13_l248_24887

noncomputable def a (n : ℕ) : ℤ := 88 - 7 * n

theorem closest_to_zero_is_13 : ∀ (n : ℕ), 1 ≤ n → 81 + (n - 1) * (-7) = a n →
  (∀ m : ℕ, (m : ℤ) ≤ (88 : ℤ) / 7 → abs (a m) > abs (a 13)) :=
  sorry

end NUMINAMATH_GPT_closest_to_zero_is_13_l248_24887


namespace NUMINAMATH_GPT_real_values_satisfying_inequality_l248_24816

theorem real_values_satisfying_inequality :
  { x : ℝ | (x^2 + 2*x^3 - 3*x^4) / (2*x + 3*x^2 - 4*x^3) ≥ -1 } =
  Set.Icc (-1 : ℝ) ((-3 - Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 - Real.sqrt 41) / -8) ((-3 + Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 + Real.sqrt 41) / -8) 0 ∪ 
  Set.Ioi 0 :=
by
  sorry

end NUMINAMATH_GPT_real_values_satisfying_inequality_l248_24816


namespace NUMINAMATH_GPT_equal_12_mn_P_2n_Q_m_l248_24842

-- Define P and Q based on given conditions
def P (m : ℕ) : ℕ := 2 ^ m
def Q (n : ℕ) : ℕ := 3 ^ n

-- The theorem to prove
theorem equal_12_mn_P_2n_Q_m (m n : ℕ) : (12 ^ (m * n)) = (P m ^ (2 * n)) * (Q n ^ m) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_equal_12_mn_P_2n_Q_m_l248_24842


namespace NUMINAMATH_GPT_jerry_can_throw_things_l248_24888

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25
def office_points_threshold : ℕ := 100
def interruptions : ℕ := 2
def insults : ℕ := 4

theorem jerry_can_throw_things : 
  (office_points_threshold - (points_for_interrupting * interruptions + points_for_insulting * insults)) / points_for_throwing = 2 :=
by 
  sorry

end NUMINAMATH_GPT_jerry_can_throw_things_l248_24888


namespace NUMINAMATH_GPT_min_value_of_expression_l248_24853

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, x = 6 * (12 : ℝ)^(1/6) ∧
  (∀ a b c, 0 < a ∧ 0 < b ∧ 0 < c → 
  x ≤ (a + 2 * b) / c + (2 * a + c) / b + (b + 3 * c) / a) :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l248_24853


namespace NUMINAMATH_GPT_contradiction_proof_l248_24880

theorem contradiction_proof (a b : ℝ) (h : a ≥ b) (h_pos : b > 0) (h_contr : a^2 < b^2) : false :=
by {
  sorry
}

end NUMINAMATH_GPT_contradiction_proof_l248_24880


namespace NUMINAMATH_GPT_fraction_of_work_completed_in_25_days_l248_24886

def men_init : ℕ := 100
def days_total : ℕ := 50
def hours_per_day_init : ℕ := 8
def days_first : ℕ := 25
def men_add : ℕ := 60
def hours_per_day_later : ℕ := 10

theorem fraction_of_work_completed_in_25_days : 
  (men_init * days_first * hours_per_day_init) / (men_init * days_total * hours_per_day_init) = 1 / 2 :=
  by sorry

end NUMINAMATH_GPT_fraction_of_work_completed_in_25_days_l248_24886


namespace NUMINAMATH_GPT_FlyersDistributon_l248_24847

variable (total_flyers ryan_flyers alyssa_flyers belinda_percentage : ℕ)
variable (scott_flyers : ℕ)

theorem FlyersDistributon (H : total_flyers = 200)
  (H1 : ryan_flyers = 42)
  (H2 : alyssa_flyers = 67)
  (H3 : belinda_percentage = 20)
  (H4 : scott_flyers = total_flyers - (ryan_flyers + alyssa_flyers + (belinda_percentage * total_flyers) / 100)) :
  scott_flyers = 51 :=
by
  simp [H, H1, H2, H3] at H4
  exact H4

end NUMINAMATH_GPT_FlyersDistributon_l248_24847


namespace NUMINAMATH_GPT_union_sets_l248_24821

def A : Set ℝ := { x | (2 / x) > 1 }
def B : Set ℝ := { x | Real.log x < 0 }

theorem union_sets : (A ∪ B) = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_union_sets_l248_24821


namespace NUMINAMATH_GPT_correct_calculation_l248_24873

theorem correct_calculation (x : ℝ) : 
(x + x = 2 * x) ∧
(x * x = x^2) ∧
(2 * x * x^2 = 2 * x^3) ∧
(x^6 / x^3 = x^3) →
(2 * x * x^2 = 2 * x^3) := 
by
  intro h
  exact h.2.2.1

end NUMINAMATH_GPT_correct_calculation_l248_24873


namespace NUMINAMATH_GPT_parallel_lines_m_eq_l248_24891

theorem parallel_lines_m_eq (m : ℝ) : 
  (∃ k : ℝ, (x y : ℝ) → 2 * x + (m + 1) * y + 4 = k * (m * x + 3 * y - 2)) → 
  (m = 2 ∨ m = -3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parallel_lines_m_eq_l248_24891


namespace NUMINAMATH_GPT_union_sets_l248_24878

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5, 6}

theorem union_sets : (A ∪ B) = {1, 2, 3, 4, 5, 6} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l248_24878


namespace NUMINAMATH_GPT_binary_addition_is_correct_l248_24849

theorem binary_addition_is_correct :
  (0b101101 + 0b1011 + 0b11001 + 0b1110101 + 0b1111) = 0b10010001 :=
by sorry

end NUMINAMATH_GPT_binary_addition_is_correct_l248_24849


namespace NUMINAMATH_GPT_brads_running_speed_proof_l248_24803

noncomputable def brads_speed (distance_between_homes : ℕ) (maxwells_speed : ℕ) (maxwells_time : ℕ) (brad_start_delay : ℕ) : ℕ :=
  let distance_covered_by_maxwell := maxwells_speed * maxwells_time
  let distance_covered_by_brad := distance_between_homes - distance_covered_by_maxwell
  let brads_time := maxwells_time - brad_start_delay
  distance_covered_by_brad / brads_time

theorem brads_running_speed_proof :
  brads_speed 54 4 6 1 = 6 := 
by
  unfold brads_speed
  rfl

end NUMINAMATH_GPT_brads_running_speed_proof_l248_24803


namespace NUMINAMATH_GPT_polynomial_bound_swap_l248_24856

variable (a b c : ℝ)

theorem polynomial_bound_swap (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ (x : ℝ), |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2 := by
  sorry

end NUMINAMATH_GPT_polynomial_bound_swap_l248_24856


namespace NUMINAMATH_GPT_calculate_expression_l248_24850

theorem calculate_expression :
  (56 * 0.57 * 0.85) / (2.8 * 19 * 1.7) = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l248_24850


namespace NUMINAMATH_GPT_satisfies_equation_l248_24830

theorem satisfies_equation : 
  { (x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y } = 
  { (0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2), (-6, 2) } :=
by
  sorry

end NUMINAMATH_GPT_satisfies_equation_l248_24830


namespace NUMINAMATH_GPT_geography_book_price_l248_24881

open Real

-- Define the problem parameters
def num_english_books : ℕ := 35
def num_geography_books : ℕ := 35
def cost_english : ℝ := 7.50
def total_cost : ℝ := 630.00

-- Define the unknown we need to prove
def cost_geography : ℝ := 10.50

theorem geography_book_price :
  num_english_books * cost_english + num_geography_books * cost_geography = total_cost :=
by
  -- No need to include the proof steps
  sorry

end NUMINAMATH_GPT_geography_book_price_l248_24881


namespace NUMINAMATH_GPT_wall_bricks_count_l248_24817

def alice_rate (y : ℕ) : ℕ := y / 8
def bob_rate (y : ℕ) : ℕ := y / 12
def combined_rate (y : ℕ) : ℕ := (5 * y) / 24 - 12
def effective_working_time : ℕ := 6

theorem wall_bricks_count :
  ∃ y : ℕ, (combined_rate y * effective_working_time = y) ∧ y = 288 :=
by
  sorry

end NUMINAMATH_GPT_wall_bricks_count_l248_24817


namespace NUMINAMATH_GPT_find_a_and_solve_inequalities_l248_24834

-- Definitions as per conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a*x^2 + 5*x - 2 > 0
def inequality2 (a : ℝ) (x : ℝ) : Prop := a*x^2 - 5*x + a^2 - 1 > 0

-- Statement of the theorem
theorem find_a_and_solve_inequalities :
  ∀ (a : ℝ),
    (∀ x, (1/2 < x ∧ x < 2) ↔ inequality1 a x) →
    a = -2 ∧
    (∀ x, (-1/2 < x ∧ x < 3) ↔ inequality2 (-2) x) :=
by
  intros a h
  sorry

end NUMINAMATH_GPT_find_a_and_solve_inequalities_l248_24834


namespace NUMINAMATH_GPT_solution_correct_l248_24854

def mixed_number_to_fraction (a b c : ℕ) : ℚ :=
  (a * b + c) / b

def percentage_to_decimal (fraction : ℚ) : ℚ :=
  fraction / 100

def evaluate_expression : ℚ :=
  let part1 := 63 * 5 + 4
  let part2 := 48 * 7 + 3
  let part3 := 17 * 3 + 2
  let term1 := (mixed_number_to_fraction 63 5 4) * 3150
  let term2 := (mixed_number_to_fraction 48 7 3) * 2800
  let term3 := (mixed_number_to_fraction 17 3 2) * 945 / 2
  term1 - term2 + term3

theorem solution_correct :
  (percentage_to_decimal (mixed_number_to_fraction 63 5 4) * 3150) -
  (percentage_to_decimal (mixed_number_to_fraction 48 7 3) * 2800) +
  (percentage_to_decimal (mixed_number_to_fraction 17 3 2) * 945 / 2) = 737.175 := 
sorry

end NUMINAMATH_GPT_solution_correct_l248_24854


namespace NUMINAMATH_GPT_complex_number_simplification_l248_24863

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : 
  (↑(1 : ℂ) - i) / (↑(1 : ℂ) + i) ^ 2017 = -i :=
sorry

end NUMINAMATH_GPT_complex_number_simplification_l248_24863


namespace NUMINAMATH_GPT_train_crossing_time_l248_24810

theorem train_crossing_time (length_of_train : ℕ) (speed_kmh : ℕ) (speed_ms : ℕ) 
  (conversion_factor : speed_kmh * 1000 / 3600 = speed_ms) 
  (H1 : length_of_train = 180) 
  (H2 : speed_kmh = 72) 
  (H3 : speed_ms = 20) 
  : length_of_train / speed_ms = 9 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l248_24810


namespace NUMINAMATH_GPT_length_of_PQ_l248_24875

-- Definitions for the problem conditions
variable (XY UV PQ : ℝ)
variable (hXY_fixed : XY = 120)
variable (hUV_fixed : UV = 90)
variable (hParallel : XY = UV ∧ UV = PQ) -- Ensures XY || UV || PQ

-- The statement to prove
theorem length_of_PQ : PQ = 360 / 7 := by
  -- Definitions for similarity ratios and solving steps can be assumed here
  sorry

end NUMINAMATH_GPT_length_of_PQ_l248_24875


namespace NUMINAMATH_GPT_oil_flow_relationship_l248_24826

theorem oil_flow_relationship (t : ℝ) (Q : ℝ) (initial_quantity : ℝ) (flow_rate : ℝ)
  (h_initial : initial_quantity = 20) (h_flow : flow_rate = 0.2) :
  Q = initial_quantity - flow_rate * t :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_oil_flow_relationship_l248_24826


namespace NUMINAMATH_GPT_congruent_triangles_count_l248_24864

open Set

variables (g l : Line) (A B C : Point)

def number_of_congruent_triangles (g l : Line) (A B C : Point) : ℕ :=
  16

theorem congruent_triangles_count (g l : Line) (A B C : Point) :
  number_of_congruent_triangles g l A B C = 16 :=
sorry

end NUMINAMATH_GPT_congruent_triangles_count_l248_24864


namespace NUMINAMATH_GPT_option_a_is_correct_l248_24804

theorem option_a_is_correct (a b : ℝ) : 
  (a^2 + a * b) / a = a + b := 
by sorry

end NUMINAMATH_GPT_option_a_is_correct_l248_24804


namespace NUMINAMATH_GPT_solution_of_system_l248_24860

theorem solution_of_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 1) : x + y = 3 :=
sorry

end NUMINAMATH_GPT_solution_of_system_l248_24860


namespace NUMINAMATH_GPT_steve_height_end_second_year_l248_24815

noncomputable def initial_height_ft : ℝ := 5
noncomputable def initial_height_inch : ℝ := 6
noncomputable def inch_to_cm : ℝ := 2.54

noncomputable def initial_height_cm : ℝ :=
  (initial_height_ft * 12 + initial_height_inch) * inch_to_cm

noncomputable def first_growth_spurt : ℝ := 0.15
noncomputable def second_growth_spurt : ℝ := 0.07
noncomputable def height_decrease : ℝ := 0.04

noncomputable def height_after_growths : ℝ :=
  let height_after_first_growth := initial_height_cm * (1 + first_growth_spurt)
  height_after_first_growth * (1 + second_growth_spurt)

noncomputable def final_height_cm : ℝ :=
  height_after_growths * (1 - height_decrease)

theorem steve_height_end_second_year : final_height_cm = 198.03 :=
  sorry

end NUMINAMATH_GPT_steve_height_end_second_year_l248_24815


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l248_24885

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1/2) (h2 : y = -2) :
  ((x + 2 * y) ^ 2 - (x + y) * (x - y)) / (2 * y) = -4 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l248_24885


namespace NUMINAMATH_GPT_range_of_a_l248_24848

theorem range_of_a (a : ℝ) : 
  (∀ x : ℕ, (1 ≤ x ∧ x ≤ 4) → ax + 4 ≥ 0) → (-1 ≤ a ∧ a < -4/5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l248_24848


namespace NUMINAMATH_GPT_arrangement_count_l248_24857

def no_adjacent_students_arrangements (teachers students : ℕ) : ℕ :=
  if teachers = 3 ∧ students = 3 then 144 else 0

theorem arrangement_count :
  no_adjacent_students_arrangements 3 3 = 144 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_count_l248_24857


namespace NUMINAMATH_GPT_work_rate_problem_l248_24824

theorem work_rate_problem :
  ∃ (x : ℝ), 
    (0 < x) ∧ 
    (10 * (1 / x + 1 / 40) = 0.5833333333333334) ∧ 
    (x = 30) :=
by
  sorry

end NUMINAMATH_GPT_work_rate_problem_l248_24824


namespace NUMINAMATH_GPT_coeff_x4_in_expansion_correct_l248_24813

noncomputable def coeff_x4_in_expansion (f g : ℕ → ℤ) := 
  ∀ (c : ℤ), c = 80 → f 4 + g 1 * g 3 = c

-- Definitions of the individual polynomials
def poly1 (x : ℤ) : ℤ := 4 * x^2 - 2 * x + 1
def poly2 (x : ℤ) : ℤ := 2 * x + 1

-- Expanded form coefficients
def coeff_poly1 : ℕ → ℤ
  | 0       => 1
  | 1       => -2
  | 2       => 4
  | _       => 0

def coeff_poly2_pow4 : ℕ → ℤ
  | 0       => 1
  | 1       => 8
  | 2       => 24
  | 3       => 32
  | 4       => 16
  | _       => 0

-- The theorem we want to prove
theorem coeff_x4_in_expansion_correct :
  coeff_x4_in_expansion coeff_poly1 coeff_poly2_pow4 := 
by
  sorry

end NUMINAMATH_GPT_coeff_x4_in_expansion_correct_l248_24813


namespace NUMINAMATH_GPT_certain_number_divided_by_two_l248_24874

theorem certain_number_divided_by_two (x : ℝ) (h : x / 2 + x + 2 = 62) : x = 40 :=
sorry

end NUMINAMATH_GPT_certain_number_divided_by_two_l248_24874


namespace NUMINAMATH_GPT_relationship_between_k_and_c_l248_24877

-- Define the functions and given conditions
def y1 (x : ℝ) (c : ℝ) : ℝ := x^2 + 2*x + c
def y2 (x : ℝ) (k : ℝ) : ℝ := k*x + 2

-- Define the vertex of y1
def vertex_y1 (c : ℝ) : ℝ × ℝ := (-1, c - 1)

-- State the main theorem
theorem relationship_between_k_and_c (k c : ℝ) (hk : k ≠ 0) :
  y2 (vertex_y1 c).1 k = (vertex_y1 c).2 → c + k = 3 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_k_and_c_l248_24877


namespace NUMINAMATH_GPT_remainder_when_four_times_n_minus_nine_divided_by_7_l248_24833

theorem remainder_when_four_times_n_minus_nine_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (4 * n - 9) % 7 = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_when_four_times_n_minus_nine_divided_by_7_l248_24833


namespace NUMINAMATH_GPT_determine_xy_l248_24861

noncomputable section

open Real

def op_defined (ab xy : ℝ × ℝ) : ℝ × ℝ :=
  (ab.1 * xy.1 + ab.2 * xy.2, ab.1 * xy.2 + ab.2 * xy.1)

theorem determine_xy (x y : ℝ) :
  (∀ (a b : ℝ), op_defined (a, b) (x, y) = (a, b)) → (x = 1 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_determine_xy_l248_24861


namespace NUMINAMATH_GPT_csc_neg_45_eq_neg_sqrt_2_l248_24895

noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

theorem csc_neg_45_eq_neg_sqrt_2 :
  csc (-Real.pi / 4) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_csc_neg_45_eq_neg_sqrt_2_l248_24895


namespace NUMINAMATH_GPT_f_bounded_by_inverse_l248_24862

theorem f_bounded_by_inverse (f : ℕ → ℝ) (h_pos : ∀ n, 0 < f n) (h_rec : ∀ n, (f n)^2 ≤ f n - f (n + 1)) :
  ∀ n, f n < 1 / (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_f_bounded_by_inverse_l248_24862


namespace NUMINAMATH_GPT_range_of_a_l248_24840

noncomputable def problem (x y z : ℝ) (a : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y + z = 1) ∧ 
  (a / (x * y * z) = 1/x + 1/y + 1/z - 2) 

theorem range_of_a (x y z a : ℝ) (h : problem x y z a) : 
  0 < a ∧ a ≤ 7/27 :=
sorry

end NUMINAMATH_GPT_range_of_a_l248_24840


namespace NUMINAMATH_GPT_megatek_manufacturing_percentage_l248_24844

theorem megatek_manufacturing_percentage (total_degrees sector_degrees : ℝ)
    (h_circle: total_degrees = 360)
    (h_sector: sector_degrees = 252) :
    (sector_degrees / total_degrees) * 100 = 70 :=
by
  sorry

end NUMINAMATH_GPT_megatek_manufacturing_percentage_l248_24844


namespace NUMINAMATH_GPT_unique_friends_count_l248_24897

-- Definitions from conditions
def M : ℕ := 10
def P : ℕ := 20
def G : ℕ := 5
def M_P : ℕ := 4
def M_G : ℕ := 2
def P_G : ℕ := 0
def M_P_G : ℕ := 2

-- Theorem we need to prove
theorem unique_friends_count : (M + P + G - M_P - M_G - P_G + M_P_G) = 31 := by
  sorry

end NUMINAMATH_GPT_unique_friends_count_l248_24897


namespace NUMINAMATH_GPT_train_length_l248_24829

theorem train_length 
  (t1 t2 : ℕ) 
  (d2 : ℕ) 
  (V L : ℝ) 
  (h1 : t1 = 11)
  (h2 : t2 = 22)
  (h3 : d2 = 120)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) : 
  L = 120 := 
by 
  sorry

end NUMINAMATH_GPT_train_length_l248_24829


namespace NUMINAMATH_GPT_parcel_cost_l248_24866

theorem parcel_cost (P : ℤ) (hP : P ≥ 1) : 
  (P ≤ 5 → C = 15 + 4 * (P - 1)) ∧ (P > 5 → C = 15 + 4 * (P - 1) - 10) :=
sorry

end NUMINAMATH_GPT_parcel_cost_l248_24866


namespace NUMINAMATH_GPT_one_of_18_consecutive_is_divisible_l248_24871

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define what it means for one number to be divisible by another
def divisible (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- The main theorem
theorem one_of_18_consecutive_is_divisible : 
  ∀ (n : ℕ), 100 ≤ n ∧ n + 17 ≤ 999 → ∃ (k : ℕ), n ≤ k ∧ k ≤ (n + 17) ∧ divisible k (sum_of_digits k) :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_one_of_18_consecutive_is_divisible_l248_24871


namespace NUMINAMATH_GPT_divide_milk_l248_24838

theorem divide_milk : (3 / 5 : ℚ) = 3 / 5 := by {
    sorry
}

end NUMINAMATH_GPT_divide_milk_l248_24838


namespace NUMINAMATH_GPT_solve_quadratic_l248_24832

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_l248_24832


namespace NUMINAMATH_GPT_range_of_t_l248_24889

theorem range_of_t (a b : ℝ) 
  (h1 : a^2 + a * b + b^2 = 1) 
  (h2 : ∃ t : ℝ, t = a * b - a^2 - b^2) : 
  ∀ t, t = a * b - a^2 - b^2 → -3 ≤ t ∧ t ≤ -1/3 :=
by sorry

end NUMINAMATH_GPT_range_of_t_l248_24889


namespace NUMINAMATH_GPT_angle_intersecting_lines_l248_24827

/-- 
Given three lines intersecting at a point forming six equal angles 
around the point, each angle equals 60 degrees.
-/
theorem angle_intersecting_lines (x : ℝ) (h : 6 * x = 360) : x = 60 := by
  sorry

end NUMINAMATH_GPT_angle_intersecting_lines_l248_24827


namespace NUMINAMATH_GPT_log_eval_l248_24812

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end NUMINAMATH_GPT_log_eval_l248_24812


namespace NUMINAMATH_GPT_total_area_of_rectangles_l248_24869

/-- The combined area of two adjacent rectangular regions given their conditions -/
theorem total_area_of_rectangles (u v w z : ℝ) 
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hz : w < z) : 
  (u + v) * z = (u + v) * w + (u + v) * (z - w) :=
by
  sorry

end NUMINAMATH_GPT_total_area_of_rectangles_l248_24869


namespace NUMINAMATH_GPT_find_other_number_l248_24822

theorem find_other_number (x y : ℕ) (h_gcd : Nat.gcd x y = 22) (h_lcm : Nat.lcm x y = 5940) (h_x : x = 220) :
  y = 594 :=
sorry

end NUMINAMATH_GPT_find_other_number_l248_24822


namespace NUMINAMATH_GPT_subsets_bound_l248_24841

variable {n : ℕ} (S : Finset (Fin n)) (m : ℕ) (A : ℕ → Finset (Fin n))

theorem subsets_bound {n : ℕ} (hn : n ≥ 2) (hA : ∀ i, 1 ≤ i ∧ i ≤ m → (A i).card ≥ 2)
  (h_inter : ∀ i j k, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → 1 ≤ k ∧ k ≤ m →
    (A i) ∩ (A j) ≠ ∅ ∧ (A i) ∩ (A k) ≠ ∅ ∧ (A j) ∩ (A k) ≠ ∅ → (A i) ∩ (A j) ∩ (A k) ≠ ∅) :
  m ≤ 2 ^ (n - 1) - 1 := 
sorry

end NUMINAMATH_GPT_subsets_bound_l248_24841


namespace NUMINAMATH_GPT_garden_area_l248_24802

-- Given conditions:
def width := 16
def length (W : ℕ) := 3 * W

-- Proof statement:
theorem garden_area (W : ℕ) (hW : W = width) : length W * W = 768 :=
by
  rw [hW]
  exact rfl

end NUMINAMATH_GPT_garden_area_l248_24802


namespace NUMINAMATH_GPT_exists_a_b_k_l248_24865

theorem exists_a_b_k (m : ℕ) (hm : 0 < m) : 
  ∃ a b k : ℤ, 
    (a % 2 = 1) ∧ 
    (b % 2 = 1) ∧ 
    (0 ≤ k) ∧ 
    (2 * m = a^19 + b^99 + k * 2^1999) :=
sorry

end NUMINAMATH_GPT_exists_a_b_k_l248_24865


namespace NUMINAMATH_GPT_conic_curve_focus_eccentricity_l248_24823

theorem conic_curve_focus_eccentricity (m : ℝ) 
  (h : ∀ x y : ℝ, x^2 + m * y^2 = 1)
  (eccentricity_eq : ∀ a b : ℝ, a > b → m = 4/3) : m = 4/3 :=
by
  sorry

end NUMINAMATH_GPT_conic_curve_focus_eccentricity_l248_24823


namespace NUMINAMATH_GPT_germination_percentage_l248_24805

theorem germination_percentage (total_seeds_plot1 total_seeds_plot2 germinated_plot2_percentage total_germinated_percentage germinated_plot1_percentage : ℝ) 
  (plant1 : total_seeds_plot1 = 300) 
  (plant2 : total_seeds_plot2 = 200) 
  (germination2 : germinated_plot2_percentage = 0.35) 
  (total_germination : total_germinated_percentage = 0.23)
  (germinated_plot1 : germinated_plot1_percentage = 0.15) :
  (total_germinated_percentage * (total_seeds_plot1 + total_seeds_plot2) = 
    (germinated_plot2_percentage * total_seeds_plot2) + (germinated_plot1_percentage * total_seeds_plot1)) :=
by
  sorry

end NUMINAMATH_GPT_germination_percentage_l248_24805


namespace NUMINAMATH_GPT_ten_percent_of_x_l248_24800

theorem ten_percent_of_x
  (x : ℝ)
  (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 17.85 :=
by
  -- theorem proof goes here
  sorry

end NUMINAMATH_GPT_ten_percent_of_x_l248_24800


namespace NUMINAMATH_GPT_tank_capacity_l248_24843

theorem tank_capacity
  (w c : ℝ)
  (h1 : w / c = 1 / 3)
  (h2 : (w + 5) / c = 2 / 5) :
  c = 75 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l248_24843


namespace NUMINAMATH_GPT_hours_felt_good_l248_24868

variable (x : ℝ)

theorem hours_felt_good (h1 : 15 * x + 10 * (8 - x) = 100) : x == 4 := 
by
  sorry

end NUMINAMATH_GPT_hours_felt_good_l248_24868


namespace NUMINAMATH_GPT_a_13_eq_30_l248_24811

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Define arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a_5_eq_6 : a 5 = 6
axiom a_8_eq_15 : a 8 = 15

-- Required proof
theorem a_13_eq_30 (h : arithmetic_sequence a d) : a 13 = 30 :=
  sorry

end NUMINAMATH_GPT_a_13_eq_30_l248_24811


namespace NUMINAMATH_GPT_like_terms_sum_three_l248_24876

theorem like_terms_sum_three (m n : ℤ) (h1 : 2 * m = 4 - n) (h2 : m = n - 1) : m + n = 3 :=
sorry

end NUMINAMATH_GPT_like_terms_sum_three_l248_24876


namespace NUMINAMATH_GPT_group_product_number_l248_24837

theorem group_product_number (a : ℕ) (group_size : ℕ) (interval : ℕ) (fifth_group_product : ℕ) :
  fifth_group_product = a + 4 * interval → fifth_group_product = 94 → group_size = 5 → interval = 20 →
  (a + (1 - 1) * interval + 1 * interval) = 34 :=
by
  intros fifth_group_eq fifth_group_is_94 group_size_is_5 interval_is_20
  -- Missing steps are handled by sorry
  sorry

end NUMINAMATH_GPT_group_product_number_l248_24837


namespace NUMINAMATH_GPT_incorrect_statement_l248_24846

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b : ℝ × ℝ := (2, 1)
noncomputable def c : ℝ × ℝ := (-4, -2)

-- Define the incorrect vector statement D
theorem incorrect_statement :
  ¬ ∀ (d : ℝ × ℝ), ∃ (k1 k2 : ℝ), d = (k1 * b.1 + k2 * c.1, k1 * b.2 + k2 * c.2) := sorry

end NUMINAMATH_GPT_incorrect_statement_l248_24846


namespace NUMINAMATH_GPT_proof_problem_l248_24882

variables (a b : ℝ)
variable (h : a ≠ b)
variable (h1 : a * Real.exp a = b * Real.exp b)
variable (p : Prop := Real.log a + a = Real.log b + b)
variable (q : Prop := (a + 1) * (b + 1) < 0)

theorem proof_problem : p ∨ q :=
sorry

end NUMINAMATH_GPT_proof_problem_l248_24882


namespace NUMINAMATH_GPT_gasoline_tank_capacity_l248_24859

theorem gasoline_tank_capacity
  (y : ℝ)
  (h_initial: y * (5 / 6) - y * (1 / 3) = 20) :
  y = 40 :=
sorry

end NUMINAMATH_GPT_gasoline_tank_capacity_l248_24859


namespace NUMINAMATH_GPT_Caitlin_correct_age_l248_24819

def Aunt_Anna_age := 48
def Brianna_age := Aunt_Anna_age / 2
def Caitlin_age := Brianna_age - 7

theorem Caitlin_correct_age : Caitlin_age = 17 := by
  /- Condon: Aunt Anna is 48 years old. -/
  let ha := Aunt_Anna_age
  /- Condon: Brianna is half as old as Aunt Anna. -/
  let hb := Brianna_age
  /- Condon: Caitlin is 7 years younger than Brianna. -/
  let hc := Caitlin_age
  /- Question: How old is Caitlin? Proof: -/
  sorry

end NUMINAMATH_GPT_Caitlin_correct_age_l248_24819


namespace NUMINAMATH_GPT_congruent_triangle_sides_l248_24814

variable {x y : ℕ}

theorem congruent_triangle_sides (h_congruent : ∃ (a b c d e f : ℕ), (a = x) ∧ (b = 2) ∧ (c = 6) ∧ (d = 5) ∧ (e = 6) ∧ (f = y) ∧ (a = d) ∧ (b = f) ∧ (c = e)) : 
  x + y = 7 :=
sorry

end NUMINAMATH_GPT_congruent_triangle_sides_l248_24814


namespace NUMINAMATH_GPT_three_digit_integer_condition_l248_24894

theorem three_digit_integer_condition (n a b c : ℕ) (hn : 100 ≤ n ∧ n < 1000)
  (hdigits : n = 100 * a + 10 * b + c)
  (hdadigits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (fact_condition : 2 * n / 3 = a.factorial * b.factorial * c.factorial) :
  n = 432 := sorry

end NUMINAMATH_GPT_three_digit_integer_condition_l248_24894


namespace NUMINAMATH_GPT_no_sol_x_y_pos_int_eq_2015_l248_24835

theorem no_sol_x_y_pos_int_eq_2015 (x y : ℕ) (hx : x > 0) (hy : y > 0) : ¬ (x^2 - y! = 2015) :=
sorry

end NUMINAMATH_GPT_no_sol_x_y_pos_int_eq_2015_l248_24835


namespace NUMINAMATH_GPT_license_plate_increase_l248_24898

theorem license_plate_increase :
  let old_license_plates := 26^2 * 10^3
  let new_license_plates := 26^2 * 10^4
  new_license_plates / old_license_plates = 10 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_increase_l248_24898


namespace NUMINAMATH_GPT_find_x_l248_24825

theorem find_x (x : ℕ) : 
  (∃ (students : ℕ), students = 10) ∧ 
  (∃ (selected : ℕ), selected = 6) ∧ 
  (¬ (∃ (k : ℕ), k = 5 ∧ k = x) ) ∧ 
  (1 ≤ 10 - x) ∧
  (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l248_24825


namespace NUMINAMATH_GPT_shaded_area_of_square_with_circles_l248_24808

theorem shaded_area_of_square_with_circles :
  let side_length_square := 12
  let radius_quarter_circle := 6
  let radius_center_circle := 3
  let area_square := side_length_square * side_length_square
  let area_quarter_circles := 4 * (1 / 4) * Real.pi * (radius_quarter_circle ^ 2)
  let area_center_circle := Real.pi * (radius_center_circle ^ 2)
  area_square - area_quarter_circles - area_center_circle = 144 - 45 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_square_with_circles_l248_24808


namespace NUMINAMATH_GPT_simplify_expression_l248_24883

theorem simplify_expression {x a : ℝ} (h1 : x > a) (h2 : x ≠ 0) (h3 : a ≠ 0) :
  (x * (x^2 - a^2)⁻¹ + 1) / (a * (x - a)⁻¹ + (x - a)^(1 / 2))
  / ((a^2 * (x + a)^(1 / 2)) / (x - (x^2 - a^2)^(1 / 2)) + 1 / (x^2 - a * x))
  = 2 / (x^2 - a^2) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l248_24883
