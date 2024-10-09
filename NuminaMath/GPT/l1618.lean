import Mathlib

namespace equal_costs_at_45_students_l1618_161808

def ticket_cost_option1 (x : ℕ) : ℝ :=
  x * 30 * 0.8

def ticket_cost_option2 (x : ℕ) : ℝ :=
  (x - 5) * 30 * 0.9

theorem equal_costs_at_45_students : ∀ x : ℕ, ticket_cost_option1 x = ticket_cost_option2 x ↔ x = 45 := 
by
  intro x
  sorry

end equal_costs_at_45_students_l1618_161808


namespace sum_of_two_numbers_l1618_161823

theorem sum_of_two_numbers (x y : ℝ) 
  (h1 : x^2 + y^2 = 220) 
  (h2 : x * y = 52) : 
  x + y = 18 :=
by
  sorry

end sum_of_two_numbers_l1618_161823


namespace PolygonNumberSides_l1618_161806

theorem PolygonNumberSides (n : ℕ) (h : n - (1 / 2 : ℝ) * (n * (n - 3)) / 2 = 0) : n = 7 :=
by
  sorry

end PolygonNumberSides_l1618_161806


namespace input_statement_is_INPUT_l1618_161844

-- Define the type for statements
inductive Statement
| PRINT
| INPUT
| IF
| END

-- Define roles for the types of statements
def isOutput (s : Statement) : Prop := s = Statement.PRINT
def isInput (s : Statement) : Prop := s = Statement.INPUT
def isConditional (s : Statement) : Prop := s = Statement.IF
def isTermination (s : Statement) : Prop := s = Statement.END

-- Theorem to prove INPUT is the input statement
theorem input_statement_is_INPUT :
  isInput Statement.INPUT := by
  -- Proof to be provided
  sorry

end input_statement_is_INPUT_l1618_161844


namespace positive_difference_abs_eq_15_l1618_161804

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l1618_161804


namespace contrapositive_statement_l1618_161816

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬is_even n

theorem contrapositive_statement (a b : ℕ) :
  (¬(is_odd a ∧ is_odd b) ∧ ¬(is_even a ∧ is_even b)) → ¬is_even (a + b) :=
by
  sorry

end contrapositive_statement_l1618_161816


namespace largest_y_l1618_161898

theorem largest_y : ∃ (y : ℤ), (y ≤ 3) ∧ (∀ (z : ℤ), (z > y) → ¬ (z / 4 + 6 / 7 < 7 / 4)) :=
by
  -- There exists an integer y such that y <= 3 and for all integers z greater than y, the inequality does not hold
  sorry

end largest_y_l1618_161898


namespace tree_height_at_end_of_2_years_l1618_161857

-- Conditions:
-- 1. The tree tripled its height every year.
-- 2. The tree reached a height of 243 feet at the end of 5 years.
theorem tree_height_at_end_of_2_years (h5 : ℕ) (H5 : h5 = 243) : 
  ∃ h2, h2 = 9 := 
by sorry

end tree_height_at_end_of_2_years_l1618_161857


namespace contractor_absent_days_l1618_161815

theorem contractor_absent_days (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 25 * x - 7.5 * y = 685) : 
  y = 2 :=
by
  sorry

end contractor_absent_days_l1618_161815


namespace unique_solutions_of_system_l1618_161883

theorem unique_solutions_of_system (a : ℝ) :
  (∃! (x y : ℝ), a^2 - 2 * a * x - 6 * y + x^2 + y^2 = 0 ∧ (|x| - 4)^2 + (|y| - 3)^2 = 25) ↔
  (a ∈ Set.union (Set.Ioo (-12) (-6)) (Set.union {0} (Set.Ioo 6 12))) :=
by
  sorry

end unique_solutions_of_system_l1618_161883


namespace sequence_monotonic_decreasing_l1618_161851

theorem sequence_monotonic_decreasing (t : ℝ) :
  (∀ n : ℕ, n > 0 → (- (n + 1) ^ 2 + t * (n + 1)) - (- n ^ 2 + t * n) < 0) ↔ (t < 3) :=
by 
  sorry

end sequence_monotonic_decreasing_l1618_161851


namespace integer_pairs_l1618_161811

def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem integer_pairs (a b : ℤ) :
  (is_perfect_square (a^2 + 4 * b) ∧ is_perfect_square (b^2 + 4 * a)) ↔ 
  (a = 0 ∧ b = 0) ∨ (a = -4 ∧ b = -4) ∨ (a = 4 ∧ b = -4) ∨
  (∃ (k : ℕ), a = k^2 ∧ b = 0) ∨ (∃ (k : ℕ), a = 0 ∧ b = k^2) ∨
  (a = -6 ∧ b = -5) ∨ (a = -5 ∧ b = -6) ∨
  (∃ (t : ℕ), a = t ∧ b = 1 - t) ∨ (∃ (t : ℕ), a = 1 - t ∧ b = t) :=
sorry

end integer_pairs_l1618_161811


namespace solve_inequality_system_l1618_161876

theorem solve_inequality_system (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1)) →
  ((1 / 2) * x - 1 ≤ 7 - (3 / 2) * x) →
  (2 < x ∧ x ≤ 4) :=
by
  intro h1 h2
  sorry

end solve_inequality_system_l1618_161876


namespace no_infinite_prime_sequence_l1618_161859

theorem no_infinite_prime_sequence (p : ℕ → ℕ)
  (h : ∀ k : ℕ, Nat.Prime (p k) ∧ p (k + 1) = 5 * p k + 4) :
  ¬ ∀ n : ℕ, Nat.Prime (p n) :=
by
  sorry

end no_infinite_prime_sequence_l1618_161859


namespace total_area_correct_l1618_161896

-- Define the conditions from the problem
def side_length_small : ℕ := 2
def side_length_medium : ℕ := 4
def side_length_large : ℕ := 8

-- Define the areas of individual squares
def area_small : ℕ := side_length_small * side_length_small
def area_medium : ℕ := side_length_medium * side_length_medium
def area_large : ℕ := side_length_large * side_length_large

-- Define the additional areas as suggested by vague steps in the solution
def area_term1 : ℕ := 4 * 4 / 2 * 2
def area_term2 : ℕ := 2 * 2 / 2
def area_term3 : ℕ := (8 + 2) * 2 / 2 * 2

-- Define the total area as the sum of all calculated parts
def total_area : ℕ := area_large + (area_medium * 3) + area_small + area_term1 + area_term2 + area_term3

-- The theorem to prove total area is 150 square centimeters
theorem total_area_correct : total_area = 150 :=
by
  -- Proof goes here (steps from the solution)...
  sorry

end total_area_correct_l1618_161896


namespace express_in_scientific_notation_l1618_161881

theorem express_in_scientific_notation (n : ℝ) (h : n = 456.87 * 10^6) : n = 4.5687 * 10^8 :=
by 
  -- sorry to skip the proof
  sorry

end express_in_scientific_notation_l1618_161881


namespace min_repetitions_2002_div_by_15_l1618_161848

-- Define the function that generates the number based on repetitions of "2002" and appending "15"
def generate_number (n : ℕ) : ℕ :=
  let repeated := (List.replicate n 2002).foldl (λ acc x => acc * 10000 + x) 0
  repeated * 100 + 15

-- Define the minimum n for which the generated number is divisible by 15
def min_n_divisible_by_15 : ℕ := 3

-- The theorem stating the problem with its conditions (divisibility by 15)
theorem min_repetitions_2002_div_by_15 :
  ∀ n : ℕ, (generate_number n % 15 = 0) ↔ (n ≥ min_n_divisible_by_15) :=
sorry

end min_repetitions_2002_div_by_15_l1618_161848


namespace minimum_value_l1618_161882

open Real

theorem minimum_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h : x * y - 2 * x - y + 1 = 0) :
  ∃ z : ℝ, (z = (3 / 2) * x^2 + y^2) ∧ z = 15 :=
by
  sorry

end minimum_value_l1618_161882


namespace non_participating_members_l1618_161854

noncomputable def members := 35
noncomputable def badminton_players := 15
noncomputable def tennis_players := 18
noncomputable def both_players := 3

theorem non_participating_members : 
  members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end non_participating_members_l1618_161854


namespace intersection_of_A_and_B_l1618_161820

open Set

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x^2 - x ≤ 0}
  let B := ({0, 1, 2} : Set ℝ)
  A ∩ B = ({0, 1} : Set ℝ) :=
by
  sorry

end intersection_of_A_and_B_l1618_161820


namespace chord_square_length_l1618_161861

theorem chord_square_length
    (r1 r2 r3 L1 L2 L3 : ℝ)
    (h1 : r1 = 4) 
    (h2 : r2 = 8) 
    (h3 : r3 = 12) 
    (tangent1 : ∀ x, (L1 - x)^2 + (L2 - x)^2 = (r1 + r2)^2)
    (tangent2 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r2)^2) 
    (tangent3 : ∀ x, x^2 + (L3 - x)^2 = (r3 - r1)^2) : L1^2 = 3584 / 9 :=
by
  sorry

end chord_square_length_l1618_161861


namespace daisy_dog_toys_l1618_161853

theorem daisy_dog_toys (X : ℕ) (lost_toys : ℕ) (total_toys_after_found : ℕ) : 
    (X - lost_toys + (3 + 3) - lost_toys + 5 = total_toys_after_found) → total_toys_after_found = 13 → X = 5 :=
by
  intros h1 h2
  sorry

end daisy_dog_toys_l1618_161853


namespace smallest_unfound_digit_in_odd_units_l1618_161850

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l1618_161850


namespace train_a_speed_54_l1618_161807

noncomputable def speed_of_train_A (length_A length_B : ℕ) (speed_B : ℕ) (time_to_cross : ℕ) : ℕ :=
  let total_distance := length_A + length_B
  let relative_speed := total_distance / time_to_cross
  let relative_speed_km_per_hr := relative_speed * 36 / 10
  let speed_A := relative_speed_km_per_hr - speed_B
  speed_A

theorem train_a_speed_54 
  (length_A length_B : ℕ)
  (speed_B : ℕ)
  (time_to_cross : ℕ)
  (h_length_A : length_A = 150)
  (h_length_B : length_B = 150)
  (h_speed_B : speed_B = 36)
  (h_time_to_cross : time_to_cross = 12) :
  speed_of_train_A length_A length_B speed_B time_to_cross = 54 := by
  sorry

end train_a_speed_54_l1618_161807


namespace smallest_among_5_8_4_l1618_161809

theorem smallest_among_5_8_4 : ∀ (x y z : ℕ), x = 5 → y = 8 → z = 4 → z ≤ x ∧ z ≤ y :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  exact ⟨by norm_num, by norm_num⟩

end smallest_among_5_8_4_l1618_161809


namespace right_angle_vertex_trajectory_l1618_161856

theorem right_angle_vertex_trajectory (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  let P := (x, y)
  (∃ (x y : ℝ), (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16) →
  x ≠ 2 ∧ x ≠ -2 →
  x^2 + y^2 = 4 :=
by
  intro h₁ h₂
  sorry

end right_angle_vertex_trajectory_l1618_161856


namespace tens_digit_of_13_pow_2023_l1618_161839

theorem tens_digit_of_13_pow_2023 :
  ∀ (n : ℕ), (13 ^ (2023 % 20) ≡ 13 ^ n [MOD 100]) ∧ (13 ^ n ≡ 97 [MOD 100]) → (13 ^ 2023) % 100 / 10 % 10 = 9 :=
by
sorry

end tens_digit_of_13_pow_2023_l1618_161839


namespace simplify_expression_eq_l1618_161887

theorem simplify_expression_eq (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) : 
  (a - 1/a) / ((a^2 - 2 * a + 1) / a) = (a + 1) / (a - 1) :=
by
  sorry

end simplify_expression_eq_l1618_161887


namespace solution_of_inequality_l1618_161826

theorem solution_of_inequality (x : ℝ) : -2 * x - 1 < -1 → x > 0 :=
by
  sorry

end solution_of_inequality_l1618_161826


namespace find_polynomial_l1618_161828

noncomputable def polynomial_p (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ t x y a b c : ℝ,
    (P (t * x) (t * y) = t ^ n * P x y) ∧
    (P (a + b) c + P (b + c) a + P (c + a) b = 0) ∧
    (P 1 0 = 1)

theorem find_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) (h : polynomial_p n P) :
  ∀ x y : ℝ, P x y = x^n - y^n :=
sorry

end find_polynomial_l1618_161828


namespace exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l1618_161846

theorem exists_half_perimeter_area_rectangle_6x1 :
  ∃ x₁ x₂ : ℝ, (6 * 1 / 2 = (6 + 1) / 2) ∧
                x₁ * x₂ = 3 ∧
                (x₁ + x₂ = 3.5) ∧
                (x₁ = 2 ∨ x₁ = 1.5) ∧
                (x₂ = 2 ∨ x₂ = 1.5)
:= by
  sorry

theorem not_exists_half_perimeter_area_rectangle_2x1 :
  ¬(∃ x : ℝ, x * (1.5 - x) = 1)
:= by
  sorry

end exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l1618_161846


namespace gcd_245_1001_l1618_161824

-- Definitions based on the given conditions

def fact245 : ℕ := 5 * 7^2
def fact1001 : ℕ := 7 * 11 * 13

-- Lean 4 statement of the proof problem
theorem gcd_245_1001 : Nat.gcd fact245 fact1001 = 7 :=
by
  -- Add the prime factorizations as assumptions
  have h1: fact245 = 245 := by sorry
  have h2: fact1001 = 1001 := by sorry
  -- The goal is to prove the GCD
  sorry

end gcd_245_1001_l1618_161824


namespace distance_traveled_by_car_l1618_161880

theorem distance_traveled_by_car :
  let total_distance := 90
  let distance_by_foot := (1 / 5 : ℝ) * total_distance
  let distance_by_bus := (2 / 3 : ℝ) * total_distance
  let distance_by_car := total_distance - (distance_by_foot + distance_by_bus)
  distance_by_car = 12 :=
by
  sorry

end distance_traveled_by_car_l1618_161880


namespace find_x_l1618_161800

theorem find_x 
  (x : ℝ)
  (h : 120 + 80 + x + x = 360) : 
  x = 80 :=
sorry

end find_x_l1618_161800


namespace find_m_l1618_161889

theorem find_m (m : ℝ) :
  (∀ x y : ℝ, (3 * x + (m + 1) * y - (m - 7) = 0) → 
              (m * x + 2 * y + 3 * m = 0)) →
  (m + 1 ≠ 0) →
  m = -3 :=
by
  sorry

end find_m_l1618_161889


namespace principal_amount_l1618_161812

theorem principal_amount (P : ℝ) (r t : ℝ) (d : ℝ) 
  (h1 : r = 7)
  (h2 : t = 2)
  (h3 : d = 49)
  (h4 : P * ((1 + r / 100) ^ t - 1) - P * (r * t / 100) = d) :
  P = 10000 :=
by sorry

end principal_amount_l1618_161812


namespace probability_two_boys_and_three_girls_l1618_161836

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_boys_and_three_girls :
  binomial_probability 5 2 0.5 = 0.3125 :=
by
  sorry

end probability_two_boys_and_three_girls_l1618_161836


namespace apples_distribution_l1618_161863

theorem apples_distribution (total_apples : ℕ) (rotten_apples : ℕ) (boxes : ℕ) (remaining_apples : ℕ) (apples_per_box : ℕ) :
  total_apples = 40 →
  rotten_apples = 4 →
  boxes = 4 →
  remaining_apples = total_apples - rotten_apples →
  apples_per_box = remaining_apples / boxes →
  apples_per_box = 9 :=
by
  intros
  sorry

end apples_distribution_l1618_161863


namespace certain_number_is_correct_l1618_161832

def m : ℕ := 72483

theorem certain_number_is_correct : 9999 * m = 724827405 := by
  sorry

end certain_number_is_correct_l1618_161832


namespace four_faucets_fill_time_correct_l1618_161813

-- Define the parameters given in the conditions
def three_faucets_rate (volume : ℕ) (time : ℕ) := volume / time
def one_faucet_rate (rate : ℕ) := rate / 3
def four_faucets_rate (rate : ℕ) := 4 * rate
def fill_time (volume : ℕ) (rate : ℕ) := volume / rate

-- Given problem parameters
def volume_large_tub : ℕ := 100
def time_large_tub : ℕ := 6
def volume_small_tub : ℕ := 50

-- Theorem to be proven
theorem four_faucets_fill_time_correct :
  fill_time volume_small_tub (four_faucets_rate (one_faucet_rate (three_faucets_rate volume_large_tub time_large_tub))) * 60 = 135 :=
sorry

end four_faucets_fill_time_correct_l1618_161813


namespace customOp_eval_l1618_161818

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- State the theorem we need to prove
theorem customOp_eval : customOp 4 (-1) = -4 :=
  by
    sorry

end customOp_eval_l1618_161818


namespace height_on_fifth_bounce_l1618_161821

-- Define initial conditions
def initial_height : ℝ := 96
def initial_efficiency : ℝ := 0.5
def efficiency_decrease : ℝ := 0.05
def air_resistance_loss : ℝ := 0.02

-- Recursive function to compute the height after each bounce
def bounce_height (height : ℝ) (efficiency : ℝ) : ℝ :=
  let height_after_bounce := height * efficiency
  height_after_bounce - (height_after_bounce * air_resistance_loss)

-- Function to compute the bounce efficiency after each bounce
def bounce_efficiency (initial_efficiency : ℝ) (n : ℕ) : ℝ :=
  initial_efficiency - n * efficiency_decrease

-- Function to calculate the height after n-th bounce
def height_after_n_bounces (n : ℕ) : ℝ :=
  match n with
  | 0     => initial_height
  | n + 1 => bounce_height (height_after_n_bounces n) (bounce_efficiency initial_efficiency n)

-- Lean statement to prove the problem
theorem height_on_fifth_bounce :
  height_after_n_bounces 5 = 0.82003694685696 := by
  sorry

end height_on_fifth_bounce_l1618_161821


namespace max_difference_in_masses_of_two_flour_bags_l1618_161843

theorem max_difference_in_masses_of_two_flour_bags :
  ∀ (x y : ℝ), (24.8 ≤ x ∧ x ≤ 25.2) → (24.8 ≤ y ∧ y ≤ 25.2) → |x - y| ≤ 0.4 :=
by
  sorry

end max_difference_in_masses_of_two_flour_bags_l1618_161843


namespace domain_of_f_l1618_161855

def domain (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
∀ x, f x ∈ D

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 2)

theorem domain_of_f :
  domain f {y | y ≠ -2} :=
by sorry

end domain_of_f_l1618_161855


namespace line_intersects_circle_l1618_161867

noncomputable def diameter : ℝ := 8
noncomputable def radius : ℝ := diameter / 2
noncomputable def center_to_line_distance : ℝ := 3

theorem line_intersects_circle :
  center_to_line_distance < radius → True :=
by {
  /- The proof would go here, but for now, we use sorry. -/
  sorry
}

end line_intersects_circle_l1618_161867


namespace factorization_identity_l1618_161834

theorem factorization_identity (a b : ℝ) : 
  -a^3 + 12 * a^2 * b - 36 * a * b^2 = -a * (a - 6 * b)^2 :=
by 
  sorry

end factorization_identity_l1618_161834


namespace find_p_and_q_solution_set_l1618_161871

theorem find_p_and_q (p q : ℝ) (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) : 
  p = 5 ∧ q = -6 :=
sorry

theorem solution_set (p q : ℝ) (h_p : p = 5) (h_q : q = -6) : 
  ∀ x : ℝ, q * x^2 - p * x - 1 > 0 ↔ - (1 / 2) < x ∧ x < - (1 / 3) :=
sorry

end find_p_and_q_solution_set_l1618_161871


namespace additional_cost_per_person_l1618_161825

-- Define the initial conditions and variables used in the problem
def base_cost := 1700
def discount_per_person := 50
def car_wash_earnings := 500
def initial_friends := 6
def final_friends := initial_friends - 1

-- Calculate initial cost per person with all friends
def discounted_base_cost_initial := base_cost - (initial_friends * discount_per_person)
def total_cost_after_car_wash_initial := discounted_base_cost_initial - car_wash_earnings
def cost_per_person_initial := total_cost_after_car_wash_initial / initial_friends

-- Calculate final cost per person after Brad leaves
def discounted_base_cost_final := base_cost - (final_friends * discount_per_person)
def total_cost_after_car_wash_final := discounted_base_cost_final - car_wash_earnings
def cost_per_person_final := total_cost_after_car_wash_final / final_friends

-- Proving the amount each friend has to pay more after Brad leaves
theorem additional_cost_per_person : cost_per_person_final - cost_per_person_initial = 40 := 
by
  sorry

end additional_cost_per_person_l1618_161825


namespace sum_of_x_y_l1618_161878

theorem sum_of_x_y (x y : ℝ) (h1 : 3 * x + 2 * y = 10) (h2 : 2 * x + 3 * y = 5) : x + y = 3 := 
by
  sorry

end sum_of_x_y_l1618_161878


namespace find_m_value_l1618_161829

def power_function_increasing (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2*m - 1 > 0)

theorem find_m_value (m : ℝ) (h : power_function_increasing m) : m = -1 :=
  sorry

end find_m_value_l1618_161829


namespace avg_first_12_even_is_13_l1618_161877

-- Definition of the first 12 even numbers
def first_12_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- The sum of the first 12 even numbers
def sum_first_12_even_numbers : ℕ := first_12_even_numbers.sum

-- Number of first 12 even numbers
def count_12_even_numbers : ℕ := first_12_even_numbers.length

-- The average of the first 12 even numbers
def average_12_even_numbers : ℕ := sum_first_12_even_numbers / count_12_even_numbers

-- Proof statement that the average of the first 12 even numbers is 13
theorem avg_first_12_even_is_13 : average_12_even_numbers = 13 := by
  sorry

end avg_first_12_even_is_13_l1618_161877


namespace gcd_102_238_l1618_161879

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l1618_161879


namespace man_l1618_161802

theorem man's_salary 
  (food_fraction : ℚ := 1/5) 
  (rent_fraction : ℚ := 1/10) 
  (clothes_fraction : ℚ := 3/5) 
  (remaining_money : ℚ := 15000) 
  (S : ℚ) :
  (S * (1 - (food_fraction + rent_fraction + clothes_fraction)) = remaining_money) →
  S = 150000 := 
by
  intros h1
  sorry

end man_l1618_161802


namespace two_pow_gt_twice_n_plus_one_l1618_161801

theorem two_pow_gt_twice_n_plus_one (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
sorry

end two_pow_gt_twice_n_plus_one_l1618_161801


namespace stream_current_l1618_161830

noncomputable def solve_stream_current : Prop :=
  ∃ (r w : ℝ), (24 / (r + w) + 6 = 24 / (r - w)) ∧ (24 / (3 * r + w) + 2 = 24 / (3 * r - w)) ∧ (w = 2)

theorem stream_current : solve_stream_current :=
  sorry

end stream_current_l1618_161830


namespace harry_terry_difference_l1618_161847

-- Define Harry's answer
def H : ℤ := 8 - (2 + 5)

-- Define Terry's answer
def T : ℤ := 8 - 2 + 5

-- State the theorem to prove H - T = -10
theorem harry_terry_difference : H - T = -10 := by
  sorry

end harry_terry_difference_l1618_161847


namespace sequence_a_n_sum_T_n_l1618_161841

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (T : ℕ → ℕ)

theorem sequence_a_n (n : ℕ) (hS : ∀ n, S n = 2 * a n - n) :
  a n = 2 ^ n - 1 :=
sorry

theorem sum_T_n (n : ℕ) (hb : ∀ n, b n = (2 * n + 1) * (a n + 1)) 
  (ha : ∀ n, a n = 2 ^ n - 1) :
  T n = 2 + (2 * n - 1) * 2 ^ (n + 1) :=
sorry

end sequence_a_n_sum_T_n_l1618_161841


namespace polygon_length_l1618_161845

noncomputable def DE : ℝ := 3
noncomputable def EF : ℝ := 6
noncomputable def DE_plus_EF : ℝ := DE + EF

theorem polygon_length 
  (area_ABCDEF : ℝ)
  (AB BC FA : ℝ)
  (A B C D E F : ℝ × ℝ) :
  area_ABCDEF = 60 →
  AB = 10 →
  BC = 7 →
  FA = 6 →
  A = (0, 10) →
  B = (10, 10) →
  C = (10, 0) →
  D = (6, 0) →
  E = (6, 3) →
  F = (0, 3) →
  DE_plus_EF = 9 :=
by
  intros
  sorry

end polygon_length_l1618_161845


namespace cary_net_calorie_deficit_is_250_l1618_161838

-- Define the conditions
def miles_walked : ℕ := 3
def candy_bar_calories : ℕ := 200
def calories_per_mile : ℕ := 150

-- Define the function to calculate total calories burned
def total_calories_burned (miles : ℕ) (calories_per_mile : ℕ) : ℕ :=
  miles * calories_per_mile

-- Define the function to calculate net calorie deficit
def net_calorie_deficit (total_calories : ℕ) (candy_calories : ℕ) : ℕ :=
  total_calories - candy_calories

-- The statement to be proven
theorem cary_net_calorie_deficit_is_250 :
  net_calorie_deficit (total_calories_burned miles_walked calories_per_mile) candy_bar_calories = 250 :=
  by sorry

end cary_net_calorie_deficit_is_250_l1618_161838


namespace converse_of_implication_l1618_161864

-- Given propositions p and q
variables (p q : Prop)

-- Proving the converse of "if p then q" is "if q then p"

theorem converse_of_implication (h : p → q) : q → p :=
sorry

end converse_of_implication_l1618_161864


namespace mean_value_of_quadrilateral_angles_l1618_161822

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l1618_161822


namespace problem_solution_l1618_161868

theorem problem_solution (x y : ℚ) (h1 : |x| + x + y - 2 = 14) (h2 : x + |y| - y + 3 = 20) : 
  x + y = 31/5 := 
by
  -- It remains to prove
  sorry

end problem_solution_l1618_161868


namespace magic_triangle_max_sum_l1618_161894

theorem magic_triangle_max_sum :
  ∃ (a b c d e f : ℕ), ((a = 5 ∨ a = 6 ∨ a = 7 ∨ a = 8 ∨ a = 9 ∨ a = 10) ∧
                        (b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8 ∨ b = 9 ∨ b = 10) ∧
                        (c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8 ∨ c = 9 ∨ c = 10) ∧
                        (d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9 ∨ d = 10) ∧
                        (e = 5 ∨ e = 6 ∨ e = 7 ∨ e = 8 ∨ e = 9 ∨ e = 10) ∧
                        (f = 5 ∨ f = 6 ∨ f = 7 ∨ f = 8 ∨ f = 9 ∨ f = 10) ∧
                        (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧
                        (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧
                        (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧
                        (d ≠ e) ∧ (d ≠ f) ∧
                        (e ≠ f) ∧
                        (a + b + c = 24) ∧ (c + d + e = 24) ∧ (e + f + a = 24)) :=
sorry

end magic_triangle_max_sum_l1618_161894


namespace sum_of_S_values_l1618_161891

noncomputable def a : ℕ := 32
noncomputable def b1 : ℕ := 16 -- When M = 73
noncomputable def c : ℕ := 25
noncomputable def b2 : ℕ := 89 -- When M = 146
noncomputable def x1 : ℕ := 14 -- When M = 73
noncomputable def x2 : ℕ := 7 -- When M = 146
noncomputable def y1 : ℕ := 3 -- When M = 73
noncomputable def y2 : ℕ := 54 -- When M = 146
noncomputable def z1 : ℕ := 8 -- When M = 73
noncomputable def z2 : ℕ := 4 -- When M = 146

theorem sum_of_S_values :
  let M1 := a + b1 + c
  let M2 := a + b2 + c
  let S1 := M1 + x1 + y1 + z1
  let S2 := M2 + x2 + y2 + z2
  (S1 = 98) ∧ (S2 = 211) ∧ (S1 + S2 = 309) := by
  sorry

end sum_of_S_values_l1618_161891


namespace joan_gave_apples_l1618_161837

theorem joan_gave_apples (initial_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : initial_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  -- Show that given_apples is obtained by subtracting remaining_apples from initial_apples
  sorry

end joan_gave_apples_l1618_161837


namespace loan_period_l1618_161805

theorem loan_period (principal : ℝ) (rate_A rate_C : ℝ) (gain : ℝ) (years : ℝ) :
  principal = 3500 ∧ rate_A = 0.1 ∧ rate_C = 0.12 ∧ gain = 210 →
  (rate_C * principal * years - rate_A * principal * years) = gain →
  years = 3 :=
by
  sorry

end loan_period_l1618_161805


namespace number_of_people_joining_group_l1618_161858

theorem number_of_people_joining_group (x : ℕ) (h1 : 180 / 18 = 10) 
  (h2 : 180 / (18 + x) = 9) : x = 2 :=
by
  sorry

end number_of_people_joining_group_l1618_161858


namespace mary_average_speed_l1618_161895

noncomputable def average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) : ℝ :=
  (d1 + d2) / ((t1 + t2) / 60)

theorem mary_average_speed :
  average_speed 1.5 1.5 45 15 = 3 := by
  sorry

end mary_average_speed_l1618_161895


namespace least_multiple_of_25_gt_475_l1618_161862

theorem least_multiple_of_25_gt_475 : ∃ n : ℕ, n > 475 ∧ n % 25 = 0 ∧ ∀ m : ℕ, (m > 475 ∧ m % 25 = 0) → n ≤ m := 
  sorry

end least_multiple_of_25_gt_475_l1618_161862


namespace inscribed_sphere_to_cube_volume_ratio_l1618_161852

theorem inscribed_sphere_to_cube_volume_ratio :
  let s := 8
  let r := s / 2
  let V_sphere := (4/3) * Real.pi * r^3
  let V_cube := s^3
  (V_sphere / V_cube) = Real.pi / 6 :=
by
  let s := 8
  let r := s / 2
  let V_sphere := (4/3) * Real.pi * r^3
  let V_cube := s^3
  sorry

end inscribed_sphere_to_cube_volume_ratio_l1618_161852


namespace smallest_b_in_arithmetic_series_l1618_161840

theorem smallest_b_in_arithmetic_series (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_arith_series : a = b - d ∧ c = b + d) (h_product : a * b * c = 125) : b ≥ 5 :=
sorry

end smallest_b_in_arithmetic_series_l1618_161840


namespace initial_number_of_men_l1618_161819

theorem initial_number_of_men (M A : ℕ) : 
  (∀ (M A : ℕ), ((M * A) - 40 + 61) / M = (A + 3)) ∧ (30.5 = 30.5) → 
  M = 7 :=
by
  sorry

end initial_number_of_men_l1618_161819


namespace number_of_boys_in_class_l1618_161870

theorem number_of_boys_in_class
  (g_ratio : ℕ) (b_ratio : ℕ) (total_students : ℕ)
  (h_ratio : g_ratio / b_ratio = 4 / 3)
  (h_total_students : g_ratio + b_ratio = 7 * (total_students / 56)) :
  total_students = 56 → 3 * (total_students / (4 + 3)) = 24 :=
by
  intros total_students_56
  sorry

end number_of_boys_in_class_l1618_161870


namespace find_a4_b4_l1618_161885

theorem find_a4_b4
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end find_a4_b4_l1618_161885


namespace power_function_even_l1618_161874

-- Define the function and its properties
def f (x : ℝ) (α : ℤ) : ℝ := x ^ (Int.toNat α)

-- State the theorem with given conditions
theorem power_function_even (α : ℤ) 
    (h : f 1 α ^ 2 + f (-1) α ^ 2 = 2 * (f 1 α + f (-1) α - 1)) : 
    ∀ x : ℝ, f x α = f (-x) α :=
by
  sorry

end power_function_even_l1618_161874


namespace simplify_exponent_l1618_161884

theorem simplify_exponent :
  2000 * 2000^2000 = 2000^2001 :=
by
  sorry

end simplify_exponent_l1618_161884


namespace canary_possible_distances_l1618_161897

noncomputable def distance_from_bus_stop (bus_stop swallow sparrow canary : ℝ) : Prop :=
  swallow = 380 ∧
  sparrow = 450 ∧
  (sparrow - swallow) = (canary - sparrow) ∨
  (swallow - sparrow) = (sparrow - canary)

theorem canary_possible_distances (swallow sparrow canary : ℝ) :
  distance_from_bus_stop 0 swallow sparrow canary →
  canary = 520 ∨ canary = 1280 :=
by
  sorry

end canary_possible_distances_l1618_161897


namespace cupcakes_sold_l1618_161803

theorem cupcakes_sold (initial additional final sold : ℕ) (h1 : initial = 14) (h2 : additional = 17) (h3 : final = 25) :
  initial + additional - final = sold :=
by
  sorry

end cupcakes_sold_l1618_161803


namespace min_n_satisfies_inequality_l1618_161814

theorem min_n_satisfies_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2) ≤ n * (x^4 + y^4 + z^4)) ∧ (n = 3) :=
by
  sorry

end min_n_satisfies_inequality_l1618_161814


namespace geometric_sequence_a3_is_15_l1618_161842

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
a1 * q^(n - 1)

theorem geometric_sequence_a3_is_15 (q : ℝ) (a1 : ℝ) (a5 : ℝ) 
  (h1 : a1 = 3) (h2 : a5 = 75) (h_seq : ∀ n, a5 = geometric_sequence a1 q n) :
  geometric_sequence a1 q 3 = 15 :=
by 
  sorry

end geometric_sequence_a3_is_15_l1618_161842


namespace probability_of_picking_dumpling_with_egg_l1618_161872

-- Definitions based on the conditions
def total_dumplings : ℕ := 10
def dumplings_with_eggs : ℕ := 3

-- The proof statement
theorem probability_of_picking_dumpling_with_egg :
  (dumplings_with_eggs : ℚ) / total_dumplings = 3 / 10 :=
by
  sorry

end probability_of_picking_dumpling_with_egg_l1618_161872


namespace walter_zoo_time_l1618_161886

theorem walter_zoo_time (S: ℕ) (H1: S + 8 * S + 13 = 130) : S = 13 :=
by sorry

end walter_zoo_time_l1618_161886


namespace peter_contains_five_l1618_161860

theorem peter_contains_five (N : ℕ) (hN : N > 0) :
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → ∃ i : ℕ, 5 ≤ 10^i * (N * 5^m / 10^i) % 10 :=
sorry

end peter_contains_five_l1618_161860


namespace largest_is_three_l1618_161893

variable (p q r : ℝ)

def cond1 : Prop := p + q + r = 3
def cond2 : Prop := p * q + p * r + q * r = 1
def cond3 : Prop := p * q * r = -6

theorem largest_is_three
  (h1 : cond1 p q r)
  (h2 : cond2 p q r)
  (h3 : cond3 p q r) :
  p = 3 ∨ q = 3 ∨ r = 3 := sorry

end largest_is_three_l1618_161893


namespace arithmetic_sequence_probability_l1618_161810

theorem arithmetic_sequence_probability (n p : ℕ) (h_cond : n + p = 2008) (h_neg : n = 161) (h_pos : p = 2008 - 161) :
  ∃ a b : ℕ, (a = 1715261 ∧ b = 2016024 ∧ a + b = 3731285) ∧ (a / b = 1715261 / 2016024) := by
  sorry

end arithmetic_sequence_probability_l1618_161810


namespace single_elimination_games_l1618_161875

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  (n - 1) = 511 :=
by
  sorry

end single_elimination_games_l1618_161875


namespace equation_of_line_passing_through_point_with_slope_l1618_161869

theorem equation_of_line_passing_through_point_with_slope :
  ∃ (l : ℝ → ℝ), l 0 = -1 ∧ ∀ (x y : ℝ), y = l x ↔ y + 1 = 2 * x :=
sorry

end equation_of_line_passing_through_point_with_slope_l1618_161869


namespace total_distance_traveled_l1618_161833

theorem total_distance_traveled
  (r1 r2 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25):
  let arc_outer := 1/4 * 2 * Real.pi * r2
  let radial := r2 - r1
  let circ_inner := 2 * Real.pi * r1
  let return_radial := radial
  let total_distance := arc_outer + radial + circ_inner + return_radial
  total_distance = 42.5 * Real.pi + 20 := 
by
  sorry

end total_distance_traveled_l1618_161833


namespace solution_l1618_161835

def money_problem (x y : ℝ) : Prop :=
  (x + y / 2 = 50) ∧ (y + 2 * x / 3 = 50)

theorem solution :
  ∃ x y : ℝ, money_problem x y ∧ x = 37.5 ∧ y = 25 :=
by
  use 37.5, 25
  sorry

end solution_l1618_161835


namespace max_a_for_integer_roots_l1618_161899

theorem max_a_for_integer_roots (a : ℕ) :
  (∀ x : ℤ, x^2 - 2 * (a : ℤ) * x + 64 = 0 → (∃ y : ℤ, x = y)) →
  (∀ x1 x2 : ℤ, x1 * x2 = 64 ∧ x1 + x2 = 2 * (a : ℤ)) →
  a ≤ 17 := 
sorry

end max_a_for_integer_roots_l1618_161899


namespace permutations_without_HMMT_l1618_161866

noncomputable def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem permutations_without_HMMT :
  let total_permutations := multinomial 8 2 2 4
  let block_permutations := multinomial 5 1 1 2
  (total_permutations - block_permutations + 1) = 361 :=
by
  sorry

end permutations_without_HMMT_l1618_161866


namespace restaurant_chili_paste_needs_l1618_161827

theorem restaurant_chili_paste_needs:
  let large_can_volume := 25
  let small_can_volume := 15
  let large_cans_required := 45
  let total_volume := large_cans_required * large_can_volume
  let small_cans_needed := total_volume / small_can_volume
  small_cans_needed - large_cans_required = 30 :=
by
  sorry

end restaurant_chili_paste_needs_l1618_161827


namespace h_plus_k_l1618_161849

theorem h_plus_k :
  ∀ h k : ℝ, (∀ x : ℝ, x^2 + 4 * x + 4 = (x + h) ^ 2 - k) → h + k = 2 :=
by
  intro h k H
  -- using sorry to indicate the proof is omitted
  sorry

end h_plus_k_l1618_161849


namespace necessary_and_sufficient_condition_for_parallel_lines_l1618_161890

theorem necessary_and_sufficient_condition_for_parallel_lines (a l : ℝ) :
  (a = -1) ↔ (∀ x y : ℝ, ax + 3 * y + 3 = 0 → x + (a - 2) * y + l = 0) := 
sorry

end necessary_and_sufficient_condition_for_parallel_lines_l1618_161890


namespace remainder_when_112222333_divided_by_37_l1618_161865

theorem remainder_when_112222333_divided_by_37 : 112222333 % 37 = 0 :=
by
  sorry

end remainder_when_112222333_divided_by_37_l1618_161865


namespace johns_weekly_allowance_l1618_161873

theorem johns_weekly_allowance (A : ℝ) (h1: A - (3/5) * A = (2/5) * A)
  (h2: (2/5) * A - (1/3) * (2/5) * A = (4/15) * A)
  (h3: (4/15) * A = 0.92) : A = 3.45 :=
by {
  sorry
}

end johns_weekly_allowance_l1618_161873


namespace min_value_M_proof_l1618_161892

noncomputable def min_value_M (a b c d e f g M : ℝ) : Prop :=
  (∀ (a b c d e f g : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0 ∧ 
    a + b + c + d + e + f + g = 1 ∧ 
    M = max (max (max (max (a + b + c) (b + c + d)) (c + d + e)) (d + e + f)) (e + f + g)
  → M ≥ (1 / 3))

theorem min_value_M_proof : min_value_M a b c d e f g M :=
by
  sorry

end min_value_M_proof_l1618_161892


namespace minimum_value_of_f_l1618_161888

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x + 1 / x) + 1 / (x^2 + 1 / x^2)

theorem minimum_value_of_f :
  (∀ x > 0, f x ≥ 3) ∧ (f 1 = 3) :=
by
  sorry

end minimum_value_of_f_l1618_161888


namespace find_p_q_d_l1618_161831

noncomputable def cubic_polynomial_real_root (p q d : ℕ) (x : ℝ) : Prop :=
  27 * x^3 - 12 * x^2 - 4 * x - 1 = 0 ∧ x = (p^(1/3) + q^(1/3) + 1) / d ∧
  p > 0 ∧ q > 0 ∧ d > 0

theorem find_p_q_d :
  ∃ (p q d : ℕ), cubic_polynomial_real_root p q d 1 ∧ p + q + d = 3 :=
by
  sorry

end find_p_q_d_l1618_161831


namespace popsicle_stick_count_l1618_161817

variable (Sam Sid Steve : ℕ)

def number_of_sticks (Sam Sid Steve : ℕ) : ℕ :=
  Sam + Sid + Steve

theorem popsicle_stick_count 
  (h1 : Sam = 3 * Sid)
  (h2 : Sid = 2 * Steve)
  (h3 : Steve = 12) :
  number_of_sticks Sam Sid Steve = 108 :=
by
  sorry

end popsicle_stick_count_l1618_161817
