import Mathlib

namespace quadratic_negative_roots_pq_value_l0_121

theorem quadratic_negative_roots_pq_value (r : ℝ) :
  (∃ p q : ℝ, p = -87 ∧ q = -23 ∧ x^2 - (r + 7)*x + r + 87 = 0 ∧ p < r ∧ r < q)
  → ((-87)^2 + (-23)^2 = 8098) :=
by
  sorry

end quadratic_negative_roots_pq_value_l0_121


namespace arithmetic_series_sum_l0_560

theorem arithmetic_series_sum :
  let a1 := 5
  let an := 105
  let d := 1
  let n := (an - a1) / d + 1
  (n * (a1 + an) / 2) = 5555 := by
  sorry

end arithmetic_series_sum_l0_560


namespace sum_of_squares_l0_715

theorem sum_of_squares (x y z : ℤ) (h1 : x + y + z = 3) (h2 : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 :=
sorry

end sum_of_squares_l0_715


namespace second_hose_correct_l0_567

/-- Define the problem parameters -/
def first_hose_rate : ℕ := 50
def initial_hours : ℕ := 3
def additional_hours : ℕ := 2
def total_capacity : ℕ := 390

/-- Define the total hours the first hose was used -/
def total_hours (initial_hours additional_hours : ℕ) : ℕ := initial_hours + additional_hours

/-- Define the amount of water sprayed by the first hose -/
def first_hose_total (first_hose_rate initial_hours additional_hours : ℕ) : ℕ :=
  first_hose_rate * (initial_hours + additional_hours)

/-- Define the remaining water needed to fill the pool -/
def remaining_water (total_capacity first_hose_total : ℕ) : ℕ :=
  total_capacity - first_hose_total

/-- Define the additional water sprayed by the first hose during the last 2 hours -/
def additional_first_hose (first_hose_rate additional_hours : ℕ) : ℕ :=
  first_hose_rate * additional_hours

/-- Define the water sprayed by the second hose -/
def second_hose_total (remaining_water additional_first_hose : ℕ) : ℕ :=
  remaining_water - additional_first_hose

/-- Define the rate of the second hose (output) -/
def second_hose_rate (second_hose_total additional_hours : ℕ) : ℕ :=
  second_hose_total / additional_hours

/-- Define the theorem we want to prove -/
theorem second_hose_correct :
  second_hose_rate
    (second_hose_total
        (remaining_water total_capacity (first_hose_total first_hose_rate initial_hours additional_hours))
        (additional_first_hose first_hose_rate additional_hours))
    additional_hours = 20 := by
  sorry

end second_hose_correct_l0_567


namespace no_real_solutions_cubic_eq_l0_862

theorem no_real_solutions_cubic_eq : ∀ x : ℝ, ¬ (∃ (y : ℝ), y = x^(1/3) ∧ y = 15 / (6 - y)) :=
by
  intro x
  intro hexist
  obtain ⟨y, hy1, hy2⟩ := hexist
  have h_cubic : y * (6 - y) = 15 := by sorry -- from y = 15 / (6 - y)
  have h_quad : y^2 - 6 * y + 15 = 0 := by sorry -- after expanding y(6 - y) = 15
  sorry -- remainder to show no real solution due to negative discriminant

end no_real_solutions_cubic_eq_l0_862


namespace equation_solution_l0_470

theorem equation_solution : ∃ x : ℝ, (3 / 20) + (3 / x) = (8 / x) + (1 / 15) ∧ x = 60 :=
by
  use 60
  -- skip the proof
  sorry

end equation_solution_l0_470


namespace greater_combined_area_l0_996

noncomputable def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def combined_area (length : ℝ) (width : ℝ) : ℝ :=
  2 * (area_of_rectangle length width)

theorem greater_combined_area 
  (length1 width1 length2 width2 : ℝ)
  (h1 : length1 = 11) (h2 : width1 = 13)
  (h3 : length2 = 6.5) (h4 : width2 = 11) :
  combined_area length1 width1 - combined_area length2 width2 = 143 :=
by
  rw [h1, h2, h3, h4]
  sorry

end greater_combined_area_l0_996


namespace lcm_of_product_of_mutually_prime_l0_751

theorem lcm_of_product_of_mutually_prime (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.lcm a b = a * b :=
by
  sorry

end lcm_of_product_of_mutually_prime_l0_751


namespace price_correct_l0_41

noncomputable def price_per_glass_on_second_day 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * P) 
  : ℝ := 0.40

theorem price_correct 
  (O : ℝ) 
  (price_first_day : ℝ) 
  (revenue_equal : 2 * O * price_first_day = 3 * O * 0.40) 
  : price_per_glass_on_second_day O price_first_day revenue_equal = 0.40 := 
by 
  sorry

end price_correct_l0_41


namespace min_voters_tall_giraffe_win_l0_942

-- Definitions from the problem statement as conditions
def precinct_voters := 3
def precincts_per_district := 9
def districts := 5
def majority_precincts(p : ℕ) := p / 2 + 1  -- Minimum precincts won in a district 
def majority_districts(d : ℕ) := d / 2 + 1  -- Minimum districts won in the final

-- Condition: majority precincts to win a district
def precinct_votes_to_win := majority_precincts precinct_voters

-- Condition: majority districts to win the final
def district_wins_to_win_final := majority_districts districts

-- Minimum precincts the Tall giraffe needs to win overall
def total_precincts_to_win := district_wins_to_win_final * majority_precincts precincts_per_district

-- Proof that the minimum number of voters who could have voted for the Tall giraffe is 30
theorem min_voters_tall_giraffe_win :
  precinct_votes_to_win * total_precincts_to_win = 30 :=
sorry

end min_voters_tall_giraffe_win_l0_942


namespace proposition_p_neither_sufficient_nor_necessary_l0_594

-- Define propositions p and q
def p (m : ℝ) : Prop := m = -1
def q (m : ℝ) : Prop := ∀ x y : ℝ, (x - 1 = 0) ∧ (x + m^2 * y = 0) → ∀ x' y' : ℝ, x' = x ∧ y' = y → (x - 1) * (x + m^2 * y) = 0

-- Main theorem statement
theorem proposition_p_neither_sufficient_nor_necessary (m : ℝ) : ¬ (p m → q m) ∧ ¬ (q m → p m) :=
by
  sorry

end proposition_p_neither_sufficient_nor_necessary_l0_594


namespace ratio_of_boys_l0_272

theorem ratio_of_boys 
  (p : ℚ) 
  (h : p = (3/4) * (1 - p)) : 
  p = 3 / 7 :=
by
  sorry

end ratio_of_boys_l0_272


namespace line_through_point_trangle_area_line_with_given_slope_l0_103

theorem line_through_point_trangle_area (k : ℝ) (b : ℝ) : 
  (∃ k, (∀ x y, y = k * (x + 3) + 4 ∧ (1 / 2) * (abs (3 * k + 4) * abs (-4 / k - 3)) = 3)) → 
  (∃ k₁ k₂, k₁ = -2/3 ∧ k₂ = -8/3 ∧ 
    (∀ x y, y = k₁ * (x + 3) + 4 → 2 * x + 3 * y - 6 = 0) ∧ 
    (∀ x y, y = k₂ * (x + 3) + 4 → 8 * x + 3 * y + 12 = 0)) := 
sorry

theorem line_with_given_slope (b : ℝ) : 
  (∀ x y, y = (1 / 6) * x + b) → (1 / 2) * abs (6 * b * b) = 3 → 
  (b = 1 ∨ b = -1) → (∀ x y, (b = 1 → x - 6 * y + 6 = 0 ∧ b = -1 → x - 6 * y - 6 = 0)) := 
sorry

end line_through_point_trangle_area_line_with_given_slope_l0_103


namespace gambler_received_max_2240_l0_843

def largest_amount_received_back (x y l : ℕ) : ℕ :=
  if 2 * l + 2 = 14 ∨ 2 * l - 2 = 14 then 
    let lost_value_1 := (6 * 100 + 8 * 20)
    let lost_value_2 := (8 * 100 + 6 * 20)
    max (3000 - lost_value_1) (3000 - lost_value_2)
  else 0

theorem gambler_received_max_2240 {x y : ℕ} (hx : 20 * x + 100 * y = 3000)
  (hl : ∃ l : ℕ, (l + (l + 2) = 14 ∨ l + (l - 2) = 14)) :
  largest_amount_received_back x y 6 = 2240 ∧ largest_amount_received_back x y 8 = 2080 := by
  sorry

end gambler_received_max_2240_l0_843


namespace percentage_of_sikhs_is_10_l0_651

-- Definitions based on the conditions
def total_boys : ℕ := 850
def percent_muslims : ℕ := 34
def percent_hindus : ℕ := 28
def other_community_boys : ℕ := 238

-- The problem statement to prove
theorem percentage_of_sikhs_is_10 :
  ((total_boys - ((percent_muslims * total_boys / 100) + (percent_hindus * total_boys / 100) + other_community_boys))
  * 100 / total_boys) = 10 := 
by
  sorry

end percentage_of_sikhs_is_10_l0_651


namespace bookstore_floor_l0_904

theorem bookstore_floor (academy_floor reading_room_floor bookstore_floor : ℤ)
  (h1: academy_floor = 7)
  (h2: reading_room_floor = academy_floor + 4)
  (h3: bookstore_floor = reading_room_floor - 9) :
  bookstore_floor = 2 :=
by
  sorry

end bookstore_floor_l0_904


namespace eight_xyz_le_one_equality_conditions_l0_416

theorem eight_xyz_le_one (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_conditions (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨
                   (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end eight_xyz_le_one_equality_conditions_l0_416


namespace matrix_power_application_l0_555

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
variable (v : Fin 2 → ℝ := ![4, -3])

theorem matrix_power_application :
  (B.mulVec v = ![8, -6]) →
  (B ^ 4).mulVec v = ![64, -48] :=
by
  intro h
  sorry

end matrix_power_application_l0_555


namespace smallest_odd_prime_factor_2021_8_plus_1_l0_42

noncomputable def least_odd_prime_factor (n : ℕ) : ℕ :=
  if 2021^8 + 1 = 0 then 2021^8 + 1 else sorry 

theorem smallest_odd_prime_factor_2021_8_plus_1 :
  least_odd_prime_factor (2021^8 + 1) = 97 :=
  by
    sorry

end smallest_odd_prime_factor_2021_8_plus_1_l0_42


namespace cost_of_advanced_purchase_ticket_l0_476

theorem cost_of_advanced_purchase_ticket
  (x : ℝ)
  (door_cost : ℝ := 14)
  (total_tickets : ℕ := 140)
  (total_money : ℝ := 1720)
  (advanced_tickets_sold : ℕ := 100)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold)
  (advanced_revenue : ℝ := advanced_tickets_sold * x)
  (door_revenue : ℝ := door_tickets_sold * door_cost)
  (total_revenue : ℝ := advanced_revenue + door_revenue) :
  total_revenue = total_money → x = 11.60 :=
by
  intro h
  sorry

end cost_of_advanced_purchase_ticket_l0_476


namespace simple_interest_rate_l0_960

theorem simple_interest_rate (P : ℝ) (R : ℝ) (SI : ℝ) (T : ℝ) (h1 : T = 4) (h2 : SI = P / 5) (h3 : SI = (P * R * T) / 100) : R = 5 := by
  sorry

end simple_interest_rate_l0_960


namespace find_a_l0_413

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (x + 1) = 3 * x + 2) (h2 : f a = 5) : a = 2 :=
sorry

end find_a_l0_413


namespace max_marks_is_667_l0_323

-- Definitions based on the problem's conditions
def pass_threshold (M : ℝ) : ℝ := 0.45 * M
def student_score : ℝ := 225
def failed_by : ℝ := 75
def passing_marks := student_score + failed_by

-- The actual theorem stating that if the conditions are met, then the maximum marks M is 667
theorem max_marks_is_667 : ∃ M : ℝ, pass_threshold M = passing_marks ∧ M = 667 :=
by
  sorry -- Proof is omitted as per the instructions

end max_marks_is_667_l0_323


namespace adam_age_is_8_l0_374

variables (A : ℕ) -- Adam's current age
variable (tom_age : ℕ) -- Tom's current age
variable (combined_age : ℕ) -- Their combined age in 12 years

theorem adam_age_is_8 (h1 : tom_age = 12) -- Tom is currently 12 years old
                    (h2 : combined_age = 44) -- In 12 years, their combined age will be 44 years old
                    (h3 : A + 12 + (tom_age + 12) = combined_age) -- Equation representing the combined age in 12 years
                    : A = 8 :=
by
  sorry

end adam_age_is_8_l0_374


namespace aisha_additional_miles_l0_505

theorem aisha_additional_miles
  (D : ℕ) (d : ℕ) (v1 : ℕ) (v2 : ℕ) (v_avg : ℕ)
  (h1 : D = 18) (h2 : v1 = 36) (h3 : v2 = 60) (h4 : v_avg = 48)
  (h5 : d = 30) :
  (D + d) / ((D / v1) + (d / v2)) = v_avg :=
  sorry

end aisha_additional_miles_l0_505


namespace Drew_older_than_Maya_by_5_l0_632

variable (Maya Drew Peter John Jacob : ℕ)
variable (h1 : John = 30)
variable (h2 : John = 2 * Maya)
variable (h3 : Jacob = 11)
variable (h4 : Jacob + 2 = (Peter + 2) / 2)
variable (h5 : Peter = Drew + 4)

theorem Drew_older_than_Maya_by_5 : Drew = Maya + 5 :=
by
  have Maya_age : Maya = 30 / 2 := by sorry
  have Jacob_age_in_2_years : Jacob + 2 = 13 := by sorry
  have Peter_age_in_2_years : Peter + 2 = 2 * 13 := by sorry
  have Peter_age : Peter = 26 - 2 := by sorry
  have Drew_age : Drew = Peter - 4 := by sorry
  have Drew_older_than_Maya : Drew = Maya + 5 := by sorry
  exact Drew_older_than_Maya

end Drew_older_than_Maya_by_5_l0_632


namespace smallest_integer_solution_l0_61

theorem smallest_integer_solution :
  ∃ y : ℤ, (5 / 8 < (y - 3) / 19) ∧ ∀ z : ℤ, (5 / 8 < (z - 3) / 19) → y ≤ z :=
sorry

end smallest_integer_solution_l0_61


namespace molecular_weight_of_compound_l0_675

def atomic_weight (count : ℕ) (atomic_mass : ℝ) : ℝ :=
  count * atomic_mass

def molecular_weight (C_atom_count H_atom_count O_atom_count : ℕ)
  (C_atomic_weight H_atomic_weight O_atomic_weight : ℝ) : ℝ :=
  (atomic_weight C_atom_count C_atomic_weight) +
  (atomic_weight H_atom_count H_atomic_weight) +
  (atomic_weight O_atom_count O_atomic_weight)

theorem molecular_weight_of_compound :
  molecular_weight 3 6 1 12.01 1.008 16.00 = 58.078 :=
by
  sorry

end molecular_weight_of_compound_l0_675


namespace parabola_complementary_slope_l0_584

theorem parabola_complementary_slope
  (p x0 y0 x1 y1 x2 y2 : ℝ)
  (hp : p > 0)
  (hy0 : y0 > 0)
  (hP : y0^2 = 2 * p * x0)
  (hA : y1^2 = 2 * p * x1)
  (hB : y2^2 = 2 * p * x2)
  (h_slopes : (y1 - y0) / (x1 - x0) = - (2 * p / (y2 + y0))) :
  (y1 + y2) / y0 = -2 :=
by
  sorry

end parabola_complementary_slope_l0_584


namespace percent_increase_first_quarter_l0_127

theorem percent_increase_first_quarter (P : ℝ) (X : ℝ) (h1 : P > 0) 
  (end_of_second_quarter : P * 1.8 = P*(1 + X / 100) * 1.44) : 
  X = 25 :=
by
  sorry

end percent_increase_first_quarter_l0_127


namespace problem_l0_268

noncomputable def f (x : ℝ) := Real.log x + (x + 1) / x

noncomputable def g (x : ℝ) := x - 1/x - 2 * Real.log x

theorem problem 
  (x : ℝ) (hx : x > 0) (hxn1 : x ≠ 1) :
  f x > (x + 1) * Real.log x / (x - 1) :=
by
  sorry

end problem_l0_268


namespace min_value_of_z_l0_65

theorem min_value_of_z (x y : ℝ) (h : y^2 = 4 * x) : 
  ∃ (z : ℝ), z = 3 ∧ ∀ (x' : ℝ) (hx' : x' ≥ 0), ∃ (y' : ℝ), y'^2 = 4 * x' → z ≤ (1/2) * y'^2 + x'^2 + 3 :=
by sorry

end min_value_of_z_l0_65


namespace Jamie_needs_to_climb_40_rungs_l0_198

-- Define the conditions
def height_of_new_tree : ℕ := 20
def rungs_climbed_previous : ℕ := 12
def height_of_previous_tree : ℕ := 6
def rungs_per_foot := rungs_climbed_previous / height_of_previous_tree

-- Define the theorem
theorem Jamie_needs_to_climb_40_rungs :
  height_of_new_tree * rungs_per_foot = 40 :=
by
  -- Proof placeholder
  sorry

end Jamie_needs_to_climb_40_rungs_l0_198


namespace inequality_proof_l0_246

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (b + c) * (c + a) * (a + b) ≥ 4 * ((a + b + c) * ((a + b + c) / 3)^(1 / 8) - 1) :=
by
  sorry

end inequality_proof_l0_246


namespace female_managers_count_l0_542

variable (E M F FM : ℕ)

-- Conditions
def female_employees : Prop := F = 750
def fraction_managers : Prop := (2 / 5 : ℚ) * E = FM + (2 / 5 : ℚ) * M
def total_employees : Prop := E = M + F

-- Proof goal
theorem female_managers_count (h1 : female_employees F) 
                              (h2 : fraction_managers E M FM) 
                              (h3 : total_employees E M F) : 
  FM = 300 := 
sorry

end female_managers_count_l0_542


namespace max_marks_l0_973

theorem max_marks (M : ℝ) (h1 : 0.45 * M = 225) : M = 500 :=
by {
sorry
}

end max_marks_l0_973


namespace exists_unique_decomposition_l0_730

theorem exists_unique_decomposition (x : ℕ → ℝ) :
  ∃! (y z : ℕ → ℝ),
    (∀ n, x n = y n - z n) ∧
    (∀ n, y n ≥ 0) ∧
    (∀ n, z n ≥ z (n-1)) ∧
    (∀ n, y n * (z n - z (n-1)) = 0) ∧
    z 0 = 0 :=
sorry

end exists_unique_decomposition_l0_730


namespace fg_of_5_eq_140_l0_285

def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 10

theorem fg_of_5_eq_140 : f (g 5) = 140 := by
  sorry

end fg_of_5_eq_140_l0_285


namespace factor_expression_l0_893

theorem factor_expression (x : ℝ) : 
  (10 * x^3 + 45 * x^2 - 5 * x) - (-5 * x^3 + 10 * x^2 - 5 * x) = 5 * x^2 * (3 * x + 7) :=
by 
  sorry

end factor_expression_l0_893


namespace cakes_donated_l0_634
-- Import necessary libraries for arithmetic operations and proofs

-- Define the conditions and required proof in Lean
theorem cakes_donated (c : ℕ) (h : 8 * c + 4 * c + 2 * c = 140) : c = 10 :=
by
  sorry

end cakes_donated_l0_634


namespace investment_accumulation_l0_64

variable (P : ℝ) -- Initial investment amount
variable (r1 r2 r3 : ℝ) -- Interest rates for the first 3 years
variable (r4 : ℝ) -- Interest rate for the fourth year
variable (r5 : ℝ) -- Interest rate for the fifth year

-- Conditions
def conditions : Prop :=
  r1 = 0.07 ∧ 
  r2 = 0.08 ∧
  r3 = 0.10 ∧
  r4 = r3 + r3 * 0.12 ∧
  r5 = r4 - r4 * 0.08

-- The accumulated amount after 5 years
def accumulated_amount : ℝ :=
  P * (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5)

-- Proof problem
theorem investment_accumulation (P : ℝ) :
  conditions r1 r2 r3 r4 r5 → 
  accumulated_amount P r1 r2 r3 r4 r5 = 1.8141 * P := by
  sorry

end investment_accumulation_l0_64


namespace sequence_formula_l0_677

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 1) (h_recurrence : ∀ n : ℕ, 2 * n * a n + 1 = (n + 1) * a n) :
  ∀ n : ℕ, a n = n / 2^(n - 1) :=
sorry

end sequence_formula_l0_677


namespace find_two_digit_number_t_l0_157

theorem find_two_digit_number_t (t : ℕ) (ht1 : 10 ≤ t) (ht2 : t ≤ 99) (ht3 : 13 * t % 100 = 52) : t = 12 := 
sorry

end find_two_digit_number_t_l0_157


namespace room_length_l0_581

theorem room_length (w : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (h : w = 4) (h1 : cost_rate = 800) (h2 : total_cost = 17600) : 
  let L := total_cost / (w * cost_rate)
  L = 5.5 :=
by
  sorry

end room_length_l0_581


namespace pen_count_l0_871

-- Define the conditions
def total_pens := 140
def difference := 20

-- Define the quantities to be proven
def ballpoint_pens := (total_pens - difference) / 2
def fountain_pens := total_pens - ballpoint_pens

-- The theorem to be proved
theorem pen_count :
  ballpoint_pens = 60 ∧ fountain_pens = 80 :=
by
  -- Proof omitted
  sorry

end pen_count_l0_871


namespace number_of_subsets_l0_451

def num_subsets (n : ℕ) : ℕ := 2 ^ n

theorem number_of_subsets (A : Finset α) (n : ℕ) (h : A.card = n) : A.powerset.card = num_subsets n :=
by
  have : A.powerset.card = 2 ^ A.card := sorry -- Proof omitted
  rw [h] at this
  exact this

end number_of_subsets_l0_451


namespace tank_salt_solution_l0_889

theorem tank_salt_solution (x : ℝ) (h1 : (0.20 * x + 14) / ((3 / 4) * x + 21) = 1 / 3) : x = 140 :=
sorry

end tank_salt_solution_l0_889


namespace four_p_plus_one_composite_l0_766

theorem four_p_plus_one_composite (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) (h2p_plus1_prime : Nat.Prime (2 * p + 1)) : ¬ Nat.Prime (4 * p + 1) :=
sorry

end four_p_plus_one_composite_l0_766


namespace linear_combination_value_l0_980

theorem linear_combination_value (x y : ℝ) (h₁ : 2 * x + y = 8) (h₂ : x + 2 * y = 10) :
  8 * x ^ 2 + 10 * x * y + 8 * y ^ 2 = 164 :=
sorry

end linear_combination_value_l0_980


namespace parts_drawn_l0_204

-- Given that a sample of 30 parts is drawn and each part has a 25% chance of being drawn,
-- prove that the total number of parts N is 120.

theorem parts_drawn (N : ℕ) (h : (30 : ℚ) / N = 0.25) : N = 120 :=
sorry

end parts_drawn_l0_204


namespace min_arithmetic_series_sum_l0_101

-- Definitions from the conditions
def arithmetic_sequence (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def arithmetic_series_sum (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * (a1 + (n-1) * d / 2)

-- Theorem statement
theorem min_arithmetic_series_sum (a2 a7 : ℤ) (h1 : a2 = -7) (h2 : a7 = 3) :
  ∃ n, (n * (a2 + (n - 1) * 2 / 2) = (n * n) - 10 * n) ∧
  (∀ m, n* (a2 + (m - 1) * 2 / 2) ≥ n * (n * n - 10 * n)) :=
sorry

end min_arithmetic_series_sum_l0_101


namespace compare_exp_square_l0_263

theorem compare_exp_square (n : ℕ) : 
  (n ≥ 3 → 2^(2 * n) > (2 * n + 1)^2) ∧ ((n = 1 ∨ n = 2) → 2^(2 * n) < (2 * n + 1)^2) :=
by
  sorry

end compare_exp_square_l0_263


namespace probability_MAME_top_l0_954

-- Conditions
def paper_parts : ℕ := 8
def desired_top : ℕ := 1

-- Question and Proof Problem (Probability calculation)
theorem probability_MAME_top : (1 : ℚ) / paper_parts = 1 / 8 :=
by
  sorry

end probability_MAME_top_l0_954


namespace time_reading_per_week_l0_195

-- Define the given conditions
def time_meditating_per_day : ℕ := 1
def time_reading_per_day : ℕ := 2 * time_meditating_per_day
def days_in_week : ℕ := 7

-- Define the target property to prove
theorem time_reading_per_week : time_reading_per_day * days_in_week = 14 :=
by
  sorry

end time_reading_per_week_l0_195


namespace domain_of_function_l0_779

theorem domain_of_function (x : ℝ) (k : ℤ) :
  ∃ x, (2 * Real.sin x + 1 ≥ 0) ↔ (- (Real.pi / 6) + 2 * k * Real.pi ≤ x ∧ x ≤ (7 * Real.pi / 6) + 2 * k * Real.pi) :=
sorry

end domain_of_function_l0_779


namespace arcsin_sqrt_three_over_two_l0_573

theorem arcsin_sqrt_three_over_two : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  -- The proof is omitted
  sorry

end arcsin_sqrt_three_over_two_l0_573


namespace fraction_addition_l0_976

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l0_976


namespace GCD_180_252_315_l0_1

theorem GCD_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by
  sorry

end GCD_180_252_315_l0_1


namespace probability_all_truth_l0_737

noncomputable def probability_A : ℝ := 0.55
noncomputable def probability_B : ℝ := 0.60
noncomputable def probability_C : ℝ := 0.45
noncomputable def probability_D : ℝ := 0.70

theorem probability_all_truth : 
  (probability_A * probability_B * probability_C * probability_D = 0.10395) := 
by 
  sorry

end probability_all_truth_l0_737


namespace compound_interest_principal_l0_310

theorem compound_interest_principal (CI t : ℝ) (r n : ℝ) (P : ℝ) : CI = 630 ∧ t = 2 ∧ r = 0.10 ∧ n = 1 → P = 3000 :=
by
  -- Proof to be provided
  sorry

end compound_interest_principal_l0_310


namespace geometric_progression_first_term_l0_342

theorem geometric_progression_first_term (a r : ℝ) 
    (h_sum_inf : a / (1 - r) = 8)
    (h_sum_two : a * (1 + r) = 5) :
    a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) :=
sorry

end geometric_progression_first_term_l0_342


namespace Beth_bought_10_cans_of_corn_l0_385

theorem Beth_bought_10_cans_of_corn (a b : ℕ) (h1 : b = 15 + 2 * a) (h2 : b = 35) : a = 10 := by
  sorry

end Beth_bought_10_cans_of_corn_l0_385


namespace calculation_l0_312

theorem calculation : (1 / 2) ^ (-2 : ℤ) + (-1 : ℝ) ^ (2022 : ℤ) = 5 := by
  sorry

end calculation_l0_312


namespace train_speed_A_to_B_l0_703

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

end train_speed_A_to_B_l0_703


namespace make_tea_time_efficiently_l0_95

theorem make_tea_time_efficiently (minutes_kettle minutes_boil minutes_teapot minutes_teacups minutes_tea_leaves total_estimate total_time : ℕ)
  (h1 : minutes_kettle = 1)
  (h2 : minutes_boil = 15)
  (h3 : minutes_teapot = 1)
  (h4 : minutes_teacups = 1)
  (h5 : minutes_tea_leaves = 2)
  (h6 : total_estimate = 20)
  (h_total_time : total_time = minutes_kettle + minutes_boil) :
  total_time = 16 :=
by
  sorry

end make_tea_time_efficiently_l0_95


namespace distinct_sum_l0_736

theorem distinct_sum (a b c d e : ℤ) (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 0)
  (h2 : a ≠ b) (h3 : a ≠ c) (h4 : a ≠ d) (h5 : a ≠ e) (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ e) (h9 : c ≠ d) (h10 : c ≠ e) (h11 : d ≠ e) :
  a + b + c + d + e = 35 :=
sorry

end distinct_sum_l0_736


namespace perimeter_of_original_rectangle_l0_803

theorem perimeter_of_original_rectangle
  (s : ℕ)
  (h1 : 4 * s = 24)
  (l w : ℕ)
  (h2 : l = 3 * s)
  (h3 : w = s) :
  2 * (l + w) = 48 :=
by
  sorry

end perimeter_of_original_rectangle_l0_803


namespace line_to_slope_intercept_l0_999

noncomputable def line_equation (v p q : ℝ × ℝ) : Prop :=
  (v.1 * (p.1 - q.1) + v.2 * (p.2 - q.2)) = 0

theorem line_to_slope_intercept (x y m b : ℝ) :
  line_equation (3, -4) (x, y) (2, 8) → (m, b) = (3 / 4, 6.5) :=
  by
    sorry

end line_to_slope_intercept_l0_999


namespace min_value_problem1_min_value_problem2_l0_811

-- Problem 1: Prove that the minimum value of the function y = x + 4/(x + 1) + 6 is 9 given x > -1
theorem min_value_problem1 (x : ℝ) (h : x > -1) : (x + 4 / (x + 1) + 6) ≥ 9 := 
sorry

-- Problem 2: Prove that the minimum value of the function y = (x^2 + 8) / (x - 1) is 8 given x > 1
theorem min_value_problem2 (x : ℝ) (h : x > 1) : ((x^2 + 8) / (x - 1)) ≥ 8 :=
sorry

end min_value_problem1_min_value_problem2_l0_811


namespace alicia_gumballs_l0_706

theorem alicia_gumballs (A : ℕ) (h1 : 3 * A = 60) : A = 20 := sorry

end alicia_gumballs_l0_706


namespace novel_pages_total_l0_508

-- Definitions based on conditions
def pages_first_two_days : ℕ := 2 * 50
def pages_next_four_days : ℕ := 4 * 25
def pages_six_days : ℕ := pages_first_two_days + pages_next_four_days
def pages_seventh_day : ℕ := 30
def total_pages : ℕ := pages_six_days + pages_seventh_day

-- Statement of the problem as a theorem in Lean 4
theorem novel_pages_total : total_pages = 230 := by
  sorry

end novel_pages_total_l0_508


namespace number_of_stacks_l0_740

theorem number_of_stacks (total_coins stacks coins_per_stack : ℕ) (h1 : coins_per_stack = 3) (h2 : total_coins = 15) (h3 : total_coins = stacks * coins_per_stack) : stacks = 5 :=
by
  sorry

end number_of_stacks_l0_740


namespace expression_equivalence_l0_877

theorem expression_equivalence (a b : ℝ) : 2 * a * b - a^2 - b^2 = -((a - b)^2) :=
by {
  sorry
}

end expression_equivalence_l0_877


namespace yarn_length_proof_l0_237

def green_length := 156
def total_length := 632

noncomputable def red_length (x : ℕ) := green_length * x + 8

theorem yarn_length_proof (x : ℕ) (green_length_eq : green_length = 156)
  (total_length_eq : green_length + red_length x = 632) : x = 3 :=
by {
  sorry
}

end yarn_length_proof_l0_237


namespace mean_of_six_numbers_l0_435

theorem mean_of_six_numbers (sum_six_numbers : ℚ) (h : sum_six_numbers = 3/4) : 
  (sum_six_numbers / 6) = 1/8 := by
  -- proof can be filled in here
  sorry

end mean_of_six_numbers_l0_435


namespace number_of_girls_l0_391

theorem number_of_girls
  (total_pupils : ℕ)
  (boys : ℕ)
  (teachers : ℕ)
  (girls : ℕ)
  (h1 : total_pupils = 626)
  (h2 : boys = 318)
  (h3 : teachers = 36)
  (h4 : girls = total_pupils - boys - teachers) :
  girls = 272 :=
by
  rw [h1, h2, h3] at h4
  exact h4

-- Proof is not required, hence 'sorry' can be used for practical purposes
-- exact sorry

end number_of_girls_l0_391


namespace pictures_per_album_l0_947

-- Define the problem conditions
def picturesFromPhone : Nat := 35
def picturesFromCamera : Nat := 5
def totalAlbums : Nat := 5

-- Define the total number of pictures
def totalPictures : Nat := picturesFromPhone + picturesFromCamera

-- Define what we need to prove
theorem pictures_per_album :
  totalPictures / totalAlbums = 8 := by
  sorry

end pictures_per_album_l0_947


namespace octal_to_base12_conversion_l0_39

-- Define the computation functions required
def octalToDecimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 64 + d1 * 8 + d0

def decimalToBase12 (n : ℕ) : List ℕ :=
  let d0 := n % 12
  let n1 := n / 12
  let d1 := n1 % 12
  let n2 := n1 / 12
  let d2 := n2 % 12
  [d2, d1, d0]

-- The main theorem that combines both conversions
theorem octal_to_base12_conversion :
  decimalToBase12 (octalToDecimal 563) = [2, 6, 11] :=
sorry

end octal_to_base12_conversion_l0_39


namespace circle_tangent_lines_l0_159

theorem circle_tangent_lines (h k : ℝ) (r : ℝ) (h_gt_10 : h > 10) (k_gt_10 : k > 10)
  (tangent_y_eq_10 : k - 10 = r)
  (tangent_y_eq_x : r = (|h - k| / Real.sqrt 2)) :
  (h, k) = (10 + (1 + Real.sqrt 2) * r, 10 + r) :=
by
  sorry

end circle_tangent_lines_l0_159


namespace angle_of_parallel_l0_303

-- Define a line and a plane
variable {L : Type} (l : L)
variable {P : Type} (β : P)

-- Define the parallel condition
def is_parallel (l : L) (β : P) : Prop := sorry

-- Define the angle function between a line and a plane
def angle (l : L) (β : P) : ℝ := sorry

-- The theorem stating that if l is parallel to β, then the angle is 0
theorem angle_of_parallel (h : is_parallel l β) : angle l β = 0 := sorry

end angle_of_parallel_l0_303


namespace sin_alpha_l0_97

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2 - Real.pi / 4)

theorem sin_alpha (α : ℝ) (h : f α = 1 / 3) : Real.sin α = -7 / 9 :=
by 
  sorry

end sin_alpha_l0_97


namespace rectangle_ratio_l0_229

theorem rectangle_ratio {l w : ℕ} (h_w : w = 5) (h_A : 50 = l * w) : l / w = 2 := by 
  sorry

end rectangle_ratio_l0_229


namespace find_radius_l0_402

theorem find_radius 
  (r : ℝ)
  (h1 : ∀ (x y : ℝ), ((x - r) ^ 2 + y ^ 2 = r ^ 2) → (4 * x ^ 2 + 9 * y ^ 2 = 36)) 
  (h2 : (4 * r ^ 2 + 9 * 0 ^ 2 = 36)) 
  (h3 : ∃ r : ℝ, r > 0) : 
  r = (2 * Real.sqrt 5) / 3 :=
sorry

end find_radius_l0_402


namespace number_of_students_in_range_l0_880

-- Define the basic variables and conditions
variable (a b : ℝ) -- Heights of the rectangles in the histogram

-- Define the total number of surveyed students
def total_students : ℝ := 1500

-- Define the width of each histogram group
def group_width : ℝ := 5

-- State the theorem with the conditions and the expected result
theorem number_of_students_in_range (a b : ℝ) :
    5 * (a + b) * total_students = 7500 * (a + b) :=
by
  -- Proof will be added here
  sorry

end number_of_students_in_range_l0_880


namespace full_tank_cost_l0_124

-- Definitions from the conditions
def total_liters_given := 36
def total_cost_given := 18
def tank_capacity := 64

-- Hypothesis based on the conditions
def price_per_liter := total_cost_given / total_liters_given

-- Conclusion we need to prove
theorem full_tank_cost: price_per_liter * tank_capacity = 32 :=
  sorry

end full_tank_cost_l0_124


namespace simplify_and_evaluate_l0_494

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) (h3 : x ≠ 1) :
  (x = -1) → ( (x-1) / (x^2 - 2*x + 1) / ((x^2 + x - 1) / (x-1) - (x + 1)) - 1 / (x - 2) = -2 / 3 ) :=
by 
  intro hx
  rw [hx]
  sorry

end simplify_and_evaluate_l0_494


namespace probability_at_least_one_black_ball_l0_347

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4
def selected_balls : ℕ := 4

theorem probability_at_least_one_black_ball :
  (∃ (p : ℚ), p = 13 / 14 ∧ 
  (number_of_ways_to_choose4_balls_has_at_least_1_black / number_of_ways_to_choose4_balls) = p) :=
by
  sorry

end probability_at_least_one_black_ball_l0_347


namespace arithmetic_seq_50th_term_l0_480

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_50th_term : 
  arithmetic_seq_nth_term 3 7 50 = 346 :=
by
  -- Intentionally left as sorry
  sorry

end arithmetic_seq_50th_term_l0_480


namespace distance_midpoint_AB_to_y_axis_l0_333

def parabola := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }

variable (A B : parabola)
variable (x1 x2 : ℝ)
variable (y1 y2 : ℝ)

open scoped Classical

noncomputable def midpoint_x (x1 x2 : ℝ) : ℝ :=
  (x1 + x2) / 2

theorem distance_midpoint_AB_to_y_axis 
  (h1 : x1 + x2 = 3) 
  (hA : A.val = (x1, y1))
  (hB : B.val = (x2, y2)) : 
  midpoint_x x1 x2 = 3 / 2 := 
by
  sorry

end distance_midpoint_AB_to_y_axis_l0_333


namespace molecular_weight_is_171_35_l0_801

def atomic_weight_ba : ℝ := 137.33
def atomic_weight_o : ℝ := 16.00
def atomic_weight_h : ℝ := 1.01

def molecular_weight : ℝ :=
  (1 * atomic_weight_ba) + (2 * atomic_weight_o) + (2 * atomic_weight_h)

-- The goal is to prove that the molecular weight is 171.35
theorem molecular_weight_is_171_35 : molecular_weight = 171.35 :=
by
  sorry

end molecular_weight_is_171_35_l0_801


namespace problem_l0_133

theorem problem (a b c k : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hk : k ≠ 0)
  (h1 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (k * (b - c)^2) + b / (k * (c - a)^2) + c / (k * (a - b)^2) = 0 :=
by
  sorry

end problem_l0_133


namespace john_reads_days_per_week_l0_149

-- Define the conditions
def john_reads_books_per_day := 4
def total_books_read := 48
def total_weeks := 6

-- Theorem statement
theorem john_reads_days_per_week :
  (total_books_read / john_reads_books_per_day) / total_weeks = 2 :=
by
  sorry

end john_reads_days_per_week_l0_149


namespace train_cross_time_l0_789

open Real

noncomputable def length_train1 := 190 -- in meters
noncomputable def length_train2 := 160 -- in meters
noncomputable def speed_train1 := 60 * (5/18) --speed_kmhr_to_msec 60 km/hr to m/s
noncomputable def speed_train2 := 40 * (5/18) -- speed_kmhr_to_msec 40 km/hr to m/s
noncomputable def relative_speed := speed_train1 + speed_train2 -- relative speed

theorem train_cross_time :
  (length_train1 + length_train2) / relative_speed = 350 / ((60 * (5/18)) + (40 * (5/18))) :=
by
  sorry -- The proof will be here initially just to validate the Lean statement

end train_cross_time_l0_789


namespace trajectory_equation_l0_368

variable (x y a b : ℝ)
variable (P : ℝ × ℝ := (0, -3))
variable (A : ℝ × ℝ := (a, 0))
variable (Q : ℝ × ℝ := (0, b))
variable (M : ℝ × ℝ := (x, y))

theorem trajectory_equation
  (h1 : A.1 = a)
  (h2 : A.2 = 0)
  (h3 : Q.1 = 0)
  (h4 : Q.2 > 0)
  (h5 : (P.1 - A.1) * (x - A.1) + (P.2 - A.2) * y = 0)
  (h6 : (x - A.1, y) = (-3/2 * (-x, b - y))) :
  y = (1 / 4) * x ^ 2 ∧ x ≠ 0 := by
    -- Sorry, proof omitted
    sorry

end trajectory_equation_l0_368


namespace evaluate_expression_l0_66

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : 
  (6 * a ^ 2 - 15 * a + 5) * (3 * a - 4) = 0 := by
  sorry

end evaluate_expression_l0_66


namespace value_of_expression_l0_828

theorem value_of_expression : (7^2 - 6^2)^4 = 28561 :=
by sorry

end value_of_expression_l0_828


namespace javier_needs_10_dozen_l0_104

def javier_goal : ℝ := 96
def cost_per_dozen : ℝ := 2.40
def selling_price_per_donut : ℝ := 1

theorem javier_needs_10_dozen : (javier_goal / ((selling_price_per_donut - (cost_per_dozen / 12)) * 12)) = 10 :=
by
  sorry

end javier_needs_10_dozen_l0_104


namespace sum_of_five_consecutive_even_integers_l0_805

theorem sum_of_five_consecutive_even_integers (a : ℤ) 
  (h : a + (a + 4) = 144) : a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 370 := by
  sorry

end sum_of_five_consecutive_even_integers_l0_805


namespace intersection_is_expected_l0_262

open Set

def setA : Set ℝ := { x | (x + 1) / (x - 2) ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def expectedIntersection : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_is_expected :
  (setA ∩ setB) = expectedIntersection := by
  sorry

end intersection_is_expected_l0_262


namespace isosceles_triangle_area_l0_354

-- Define the conditions for the isosceles triangle
def is_isosceles_triangle (a b c : ℝ) : Prop := a = b ∨ b = c ∨ a = c 

-- Define the side lengths
def side_length_1 : ℝ := 15
def side_length_2 : ℝ := 15
def side_length_3 : ℝ := 24

-- State the theorem
theorem isosceles_triangle_area :
  is_isosceles_triangle side_length_1 side_length_2 side_length_3 →
  side_length_1 = 15 →
  side_length_2 = 15 →
  side_length_3 = 24 →
  ∃ A : ℝ, (A = (1 / 2) * 24 * 9) ∧ A = 108 :=
sorry

end isosceles_triangle_area_l0_354


namespace mushrooms_used_by_Karla_correct_l0_525

-- Given conditions
def mushrooms_cut_each_mushroom : ℕ := 4
def mushrooms_cut_total : ℕ := 22 * mushrooms_cut_each_mushroom
def mushrooms_used_by_Kenny : ℕ := 38
def mushrooms_remaining : ℕ := 8
def mushrooms_total_used_by_Kenny_and_remaining : ℕ := mushrooms_used_by_Kenny + mushrooms_remaining
def mushrooms_used_by_Karla : ℕ := mushrooms_cut_total - mushrooms_total_used_by_Kenny_and_remaining

-- Statement to prove
theorem mushrooms_used_by_Karla_correct :
  mushrooms_used_by_Karla = 42 :=
by
  sorry

end mushrooms_used_by_Karla_correct_l0_525


namespace range_of_a_l0_254

variable (a b c : ℝ)

def condition1 := a^2 - b * c - 8 * a + 7 = 0

def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
  sorry

end range_of_a_l0_254


namespace perimeter_shaded_region_l0_819

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def arc_length_per_circle (C : ℝ) : ℝ := C / 4

theorem perimeter_shaded_region (C : ℝ) (hC : C = 48) : 
  3 * arc_length_per_circle C = 36 := by
  sorry

end perimeter_shaded_region_l0_819


namespace find_y_l0_595

-- Define the problem conditions
variable (x y : ℕ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (rem_eq : x % y = 3)
variable (div_eq : (x : ℝ) / y = 96.12)

-- The theorem to prove
theorem find_y : y = 25 :=
sorry

end find_y_l0_595


namespace sum_of_values_k_l0_946

theorem sum_of_values_k (k : ℕ) : 
  (∀ x y : ℤ, (3 * x * x - k * x + 12 = 0) ∧ (3 * y * y - k * y + 12 = 0) ∧ x ≠ y) → k = 0 :=
by
  sorry

end sum_of_values_k_l0_946


namespace standard_deviation_does_not_require_repair_l0_683

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

end standard_deviation_does_not_require_repair_l0_683


namespace andrew_permit_rate_l0_380

def permits_per_hour (a h_a H T : ℕ) : ℕ :=
  T / (H - (a * h_a))

theorem andrew_permit_rate :
  permits_per_hour 2 3 8 100 = 50 := by
  sorry

end andrew_permit_rate_l0_380


namespace chocolate_bars_partial_boxes_l0_364

-- Define the total number of bars for each type
def totalA : ℕ := 853845
def totalB : ℕ := 537896
def totalC : ℕ := 729763

-- Define the box capacities for each type
def capacityA : ℕ := 9
def capacityB : ℕ := 11
def capacityC : ℕ := 15

-- State the theorem we want to prove
theorem chocolate_bars_partial_boxes :
  totalA % capacityA = 4 ∧
  totalB % capacityB = 3 ∧
  totalC % capacityC = 8 :=
by
  -- Proof omitted for this task
  sorry

end chocolate_bars_partial_boxes_l0_364


namespace poly_has_one_positive_and_one_negative_root_l0_245

theorem poly_has_one_positive_and_one_negative_root :
  ∃! r1, r1 > 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) ∧ 
  ∃! r2, r2 < 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) := by
sorry

end poly_has_one_positive_and_one_negative_root_l0_245


namespace angle_sum_around_point_l0_643

theorem angle_sum_around_point (y : ℝ) (h1 : 150 + y + y = 360) : y = 105 :=
by sorry

end angle_sum_around_point_l0_643


namespace rectangle_to_square_y_l0_582

theorem rectangle_to_square_y (y : ℝ) (a b : ℝ) (s : ℝ) (h1 : a = 7) (h2 : b = 21)
  (h3 : s^2 = a * b) (h4 : y = s / 2) : y = 7 * Real.sqrt 3 / 2 :=
by
  -- proof skipped
  sorry

end rectangle_to_square_y_l0_582


namespace vector_equation_l0_258

variable {V : Type} [AddCommGroup V]

variables (A B C : V)

theorem vector_equation :
  (B - A) - 2 • (C - A) + (C - B) = (A - C) :=
by
  sorry

end vector_equation_l0_258


namespace find_ratio_l0_572

variables {a b c d : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
variables (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
variables (h6 : (7 * a + b) / (7 * c + d) = 9)

theorem find_ratio (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
    (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
    (h6 : (7 * a + b) / (7 * c + d) = 9) :
    (9 * a + b) / (9 * c + d) = 9 := 
by {
    sorry
}

end find_ratio_l0_572


namespace base_length_of_parallelogram_l0_7

theorem base_length_of_parallelogram (A h : ℝ) (hA : A = 44) (hh : h = 11) :
  ∃ b : ℝ, b = 4 ∧ A = b * h :=
by
  sorry

end base_length_of_parallelogram_l0_7


namespace subtraction_result_l0_630

theorem subtraction_result: (3.75 - 1.4 = 2.35) :=
by
  sorry

end subtraction_result_l0_630


namespace problem_3000_mod_1001_l0_126

theorem problem_3000_mod_1001 : (300 ^ 3000 - 1) % 1001 = 0 := 
by
  have h1: (300 ^ 3000) % 7 = 1 := sorry
  have h2: (300 ^ 3000) % 11 = 1 := sorry
  have h3: (300 ^ 3000) % 13 = 1 := sorry
  sorry

end problem_3000_mod_1001_l0_126


namespace train_cross_time_approx_24_seconds_l0_509

open Real

noncomputable def time_to_cross (train_length : ℝ) (train_speed_km_h : ℝ) (man_speed_km_h : ℝ) : ℝ :=
  let train_speed_m_s := train_speed_km_h * (1000 / 3600)
  let man_speed_m_s := man_speed_km_h * (1000 / 3600)
  let relative_speed := train_speed_m_s - man_speed_m_s
  train_length / relative_speed

theorem train_cross_time_approx_24_seconds : 
  abs (time_to_cross 400 63 3 - 24) < 1 :=
by
  sorry

end train_cross_time_approx_24_seconds_l0_509


namespace ralph_did_not_hit_110_balls_l0_304

def tennis_problem : Prop :=
  ∀ (total_balls first_batch second_batch hit_first hit_second not_hit_first not_hit_second not_hit_total : ℕ),
  total_balls = 175 →
  first_batch = 100 →
  second_batch = 75 →
  hit_first = 2/5 * first_batch →
  hit_second = 1/3 * second_batch →
  not_hit_first = first_batch - hit_first →
  not_hit_second = second_batch - hit_second →
  not_hit_total = not_hit_first + not_hit_second →
  not_hit_total = 110

theorem ralph_did_not_hit_110_balls : tennis_problem := by
  unfold tennis_problem
  intros
  sorry

end ralph_did_not_hit_110_balls_l0_304


namespace points_on_fourth_board_l0_986

-- Definition of the points scored on each dartboard
def points_board_1 : ℕ := 30
def points_board_2 : ℕ := 38
def points_board_3 : ℕ := 41

-- Statement to prove that points on the fourth board are 34
theorem points_on_fourth_board : (points_board_1 + points_board_2) / 2 = 34 :=
by
  -- Given points on first and second boards
  have h1 : points_board_1 + points_board_2 = 68 := by rfl
  sorry

end points_on_fourth_board_l0_986


namespace alpha_beta_sum_l0_748

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, x ≠ 54 → x ≠ -60 → (x - α) / (x + β) = (x^2 - 72 * x + 945) / (x^2 + 45 * x - 3240)) :
  α + β = 81 :=
sorry

end alpha_beta_sum_l0_748


namespace third_dog_average_daily_miles_l0_192

/-- Bingo has three dogs. On average, they walk a total of 100 miles a week.

    The first dog walks an average of 2 miles a day.

    The second dog walks 1 mile if it is an odd day of the month and 3 miles if it is an even day of the month.

    Considering a 30-day month, the goal is to find the average daily miles of the third dog. -/
theorem third_dog_average_daily_miles :
  let total_dogs := 3
  let weekly_total_miles := 100
  let first_dog_daily_miles := 2
  let second_dog_odd_day_miles := 1
  let second_dog_even_day_miles := 3
  let days_in_month := 30
  let odd_days_in_month := 15
  let even_days_in_month := 15
  let weeks_in_month := days_in_month / 7
  let first_dog_monthly_miles := days_in_month * first_dog_daily_miles
  let second_dog_monthly_miles := (second_dog_odd_day_miles * odd_days_in_month) + (second_dog_even_day_miles * even_days_in_month)
  let third_dog_monthly_miles := (weekly_total_miles * weeks_in_month) - (first_dog_monthly_miles + second_dog_monthly_miles)
  let third_dog_daily_miles := third_dog_monthly_miles / days_in_month
  third_dog_daily_miles = 10.33 :=
by
  sorry

end third_dog_average_daily_miles_l0_192


namespace total_sampled_papers_l0_684

-- Define the conditions
variables {A B C c : ℕ}
variable (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50)
variable (stratified_sampling : true)   -- We simply denote that stratified sampling method is used

-- Theorem to prove the total number of exam papers sampled
theorem total_sampled_papers {T : ℕ} (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50) (stratified_sampling : true) :
  T = (1260 + 720 + 900) * (50 / 900) := sorry

end total_sampled_papers_l0_684


namespace senior_citizen_tickets_l0_654

theorem senior_citizen_tickets (A S : ℕ) 
  (h1 : A + S = 510) 
  (h2 : 21 * A + 15 * S = 8748) : 
  S = 327 :=
by 
  -- Proof steps are omitted as instructed
  sorry

end senior_citizen_tickets_l0_654


namespace shorter_piece_length_l0_408

theorem shorter_piece_length (x : ℝ) (h : 3 * x = 60) : x = 20 :=
by
  sorry

end shorter_piece_length_l0_408


namespace product_of_three_numbers_l0_34

theorem product_of_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 30) 
  (h2 : a = 5 * (b + c)) 
  (h3 : b = 9 * c) : 
  a * b * c = 56.25 := 
by 
  sorry

end product_of_three_numbers_l0_34


namespace dasha_meeting_sasha_l0_250

def stripes_on_zebra : ℕ := 360

variables {v : ℝ} -- speed of Masha
def dasha_speed (v : ℝ) : ℝ := 2 * v -- speed of Dasha (twice Masha's speed)

def masha_distance_before_meeting_sasha : ℕ := 180
def total_stripes_met : ℕ := stripes_on_zebra
def relative_speed_masha_sasha (v : ℝ) : ℝ := v + v -- combined speed of Masha and Sasha
def relative_speed_dasha_sasha (v : ℝ) : ℝ := 3 * v -- combined speed of Dasha and Sasha

theorem dasha_meeting_sasha (v : ℝ) (hv : 0 < v) :
  ∃ t' t'', 
  (t'' = 120 / v) ∧ (dasha_speed v * t' = 240) :=
by {
  sorry
}

end dasha_meeting_sasha_l0_250


namespace part_a_part_b_l0_52

-- Part (a)
theorem part_a (x y z : ℤ) : (x^2 + y^2 + z^2 = 2 * x * y * z) → (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

-- Part (b)
theorem part_b : ∃ (x y z v : ℤ), (x^2 + y^2 + z^2 + v^2 = 2 * x * y * z * v) → (x = 0 ∧ y = 0 ∧ z = 0 ∧ v = 0) :=
by
  sorry

end part_a_part_b_l0_52


namespace determine_h_l0_426

open Polynomial

noncomputable def f (x : ℚ) : ℚ := x^2

theorem determine_h (h : ℚ → ℚ) : 
  (∀ x, f (h x) = 9 * x^2 + 6 * x + 1) ↔ 
  (∀ x, h x = 3 * x + 1 ∨ h x = - (3 * x + 1)) :=
by
  sorry

end determine_h_l0_426


namespace bob_weight_l0_673

theorem bob_weight (j b : ℝ) (h1 : j + b = 220) (h2 : b - 2 * j = b / 3) : b = 165 :=
  sorry

end bob_weight_l0_673


namespace sum_of_squares_of_ages_l0_466

theorem sum_of_squares_of_ages {a b c : ℕ} (h1 : 5 * a + b = 3 * c) (h2 : 3 * c^2 = 2 * a^2 + b^2) 
  (relatively_prime : Nat.gcd (Nat.gcd a b) c = 1) : 
  a^2 + b^2 + c^2 = 374 :=
by
  sorry

end sum_of_squares_of_ages_l0_466


namespace negative_integer_solution_l0_423

theorem negative_integer_solution (N : ℤ) (hN : N^2 + N = -12) : N = -3 ∨ N = -4 :=
sorry

end negative_integer_solution_l0_423


namespace grades_with_fewer_students_l0_882

-- Definitions of the involved quantities
variables (G1 G2 G5 G1_2 : ℕ)
variables (Set_X : ℕ)

-- Conditions given in the problem
theorem grades_with_fewer_students (h1: G1_2 = Set_X + 30) (h2: G5 = G1 - 30) :
  exists Set_X, G1_2 - Set_X = 30 :=
by 
  sorry

end grades_with_fewer_students_l0_882


namespace union_sets_l0_962

-- Define the sets A and B using their respective conditions.
def A : Set ℝ := {x : ℝ | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 4 < x ∧ x ≤ 10}

-- The theorem we aim to prove.
theorem union_sets : A ∪ B = {x : ℝ | 3 < x ∧ x ≤ 10} := 
by
  sorry

end union_sets_l0_962


namespace correct_answer_l0_166

variable (x : ℝ)

theorem correct_answer : {x : ℝ | x^2 + 2*x + 1 = 0} = {-1} :=
by sorry -- the actual proof is not required, just the statement

end correct_answer_l0_166


namespace katya_needs_at_least_ten_l0_875

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l0_875


namespace solve_for_y_l0_619

theorem solve_for_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y) = 8) : y = 1 :=
by {
  -- Apply the conditions and derive the proof
  sorry
}

end solve_for_y_l0_619


namespace common_difference_l0_6

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
axiom h1 : a 3 + a 7 = 10
axiom h2 : a 8 = 8

-- Statement to prove
theorem common_difference (h : ∀ n, a (n + 1) = a n + d) : d = 1 :=
  sorry

end common_difference_l0_6


namespace no_integer_solutions_l0_373

theorem no_integer_solutions :
  ¬ (∃ a b : ℤ, 3 * a^2 = b^2 + 1) :=
by 
  sorry

end no_integer_solutions_l0_373


namespace sixty_percent_of_40_greater_than_four_fifths_of_25_l0_184

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  let x := (60 / 100 : ℝ) * 40
  let y := (4 / 5 : ℝ) * 25
  x - y = 4 :=
by
  sorry

end sixty_percent_of_40_greater_than_four_fifths_of_25_l0_184


namespace henry_total_payment_l0_53

-- Define the conditions
def painting_payment : ℕ := 5
def selling_extra_payment : ℕ := 8
def total_payment_per_bike : ℕ := painting_payment + selling_extra_payment  -- 13

-- Define the quantity of bikes
def bikes_count : ℕ := 8

-- Calculate the total payment for painting and selling 8 bikes
def total_payment : ℕ := bikes_count * total_payment_per_bike  -- 144

-- The statement to prove
theorem henry_total_payment : total_payment = 144 :=
by
  -- Proof goes here
  sorry

end henry_total_payment_l0_53


namespace percentage_decrease_of_y_compared_to_z_l0_700

theorem percentage_decrease_of_y_compared_to_z (x y z : ℝ)
  (h1 : x = 1.20 * y)
  (h2 : x = 0.60 * z) :
  (y = 0.50 * z) → (1 - (y / z)) * 100 = 50 :=
by
  sorry

end percentage_decrease_of_y_compared_to_z_l0_700


namespace solutions_to_quadratic_l0_235

noncomputable def a : ℝ := (6 + Real.sqrt 92) / 2
noncomputable def b : ℝ := (6 - Real.sqrt 92) / 2

theorem solutions_to_quadratic :
  a ≥ b ∧ ((∀ x : ℝ, x^2 - 6 * x + 11 = 25 → x = a ∨ x = b) → 3 * a + 2 * b = 15 + Real.sqrt 92 / 2) := by
  sorry

end solutions_to_quadratic_l0_235


namespace triangle_middle_side_at_least_sqrt_two_l0_50

theorem triangle_middle_side_at_least_sqrt_two
    (a b c : ℝ)
    (h1 : a ≥ b) (h2 : b ≥ c)
    (h3 : ∃ α : ℝ, 0 < α ∧ α < π ∧ 1 = 1/2 * b * c * Real.sin α) :
  b ≥ Real.sqrt 2 :=
sorry

end triangle_middle_side_at_least_sqrt_two_l0_50


namespace simplify_expression_l0_951

variables {a b : ℝ}

-- Define the conditions
def condition (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a^4 + b^4 = a + b)

-- Define the target goal
def goal (a b : ℝ) : Prop := 
  (a / b + b / a - 1 / (a * b^2)) = (-a - b) / (a * b^2)

-- Statement of the theorem
theorem simplify_expression (h : condition a b) : goal a b :=
by 
  sorry

end simplify_expression_l0_951


namespace cost_of_each_pair_of_socks_eq_2_l0_460

-- Definitions and conditions
def cost_of_shoes : ℤ := 74
def cost_of_bag : ℤ := 42
def paid_amount : ℤ := 118
def discount_rate : ℚ := 0.10

-- Given the conditions
def total_cost (x : ℚ) : ℚ := cost_of_shoes + 2 * x + cost_of_bag
def discount (x : ℚ) : ℚ := if total_cost x > 100 then discount_rate * (total_cost x - 100) else 0
def total_cost_after_discount (x : ℚ) : ℚ := total_cost x - discount x

-- Theorem to prove
theorem cost_of_each_pair_of_socks_eq_2 : 
  ∃ x : ℚ, total_cost_after_discount x = paid_amount ∧ 2 * x = 4 :=
by
  sorry

end cost_of_each_pair_of_socks_eq_2_l0_460


namespace number_of_true_propositions_l0_193

-- Define the original condition
def original_proposition (a b : ℝ) : Prop := (a + b = 1) → (a * b ≤ 1 / 4)

-- Define contrapositive
def contrapositive (a b : ℝ) : Prop := (a * b > 1 / 4) → (a + b ≠ 1)

-- Define inverse
def inverse (a b : ℝ) : Prop := (a * b ≤ 1 / 4) → (a + b = 1)

-- Define converse
def converse (a b : ℝ) : Prop := (a + b ≠ 1) → (a * b > 1 / 4)

-- State the problem
theorem number_of_true_propositions (a b : ℝ) :
  (original_proposition a b ∧ contrapositive a b ∧ ¬inverse a b ∧ ¬converse a b) → 
  (∃ n : ℕ, n = 1) :=
by sorry

end number_of_true_propositions_l0_193


namespace johns_family_total_members_l0_82

theorem johns_family_total_members (n_f : ℕ) (h_f : n_f = 10) (n_m : ℕ) (h_m : n_m = (13 * n_f) / 10) :
  n_f + n_m = 23 := by
  rw [h_f, h_m]
  norm_num
  sorry

end johns_family_total_members_l0_82


namespace prob_two_more_heads_than_tails_eq_210_1024_l0_417

-- Let P be the probability of getting exactly two more heads than tails when flipping 10 coins.
def P (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2^n : ℚ)

theorem prob_two_more_heads_than_tails_eq_210_1024 :
  P 10 6 = 210 / 1024 :=
by
  -- The steps leading to the proof are omitted and hence skipped
  sorry

end prob_two_more_heads_than_tails_eq_210_1024_l0_417


namespace sum_of_cubes_l0_933

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l0_933


namespace min_max_values_on_interval_l0_753

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) + (x + 1)*(Real.sin x) + 1

theorem min_max_values_on_interval :
  (∀ x ∈ Set.Icc 0 (2*Real.pi), f x ≥ -(3*Real.pi/2) ∧ f x ≤ (Real.pi/2 + 2)) ∧
  ( ∃ a ∈ Set.Icc 0 (2*Real.pi), f a = -(3*Real.pi/2) ) ∧
  ( ∃ b ∈ Set.Icc 0 (2*Real.pi), f b = (Real.pi/2 + 2) ) :=
by
  sorry

end min_max_values_on_interval_l0_753


namespace deposit_percentage_l0_163

noncomputable def last_year_cost : ℝ := 250
noncomputable def increase_percentage : ℝ := 0.40
noncomputable def amount_paid_at_pickup : ℝ := 315
noncomputable def total_cost := last_year_cost * (1 + increase_percentage)
noncomputable def deposit := total_cost - amount_paid_at_pickup
noncomputable def percentage_deposit := deposit / total_cost * 100

theorem deposit_percentage :
  percentage_deposit = 10 := 
  by
    sorry

end deposit_percentage_l0_163


namespace jasmine_max_cards_l0_392

-- Define constants and conditions
def initial_card_price : ℝ := 0.95
def discount_card_price : ℝ := 0.85
def budget : ℝ := 9.00
def threshold : ℕ := 6

-- Define the condition for the total cost if more than 6 cards are bought
def total_cost (n : ℕ) : ℝ :=
  if n ≤ threshold then initial_card_price * n
  else initial_card_price * threshold + discount_card_price * (n - threshold)

-- Define the condition for the maximum number of cards Jasmine can buy 
def max_cards (n : ℕ) : Prop :=
  total_cost n ≤ budget ∧ ∀ m : ℕ, total_cost m ≤ budget → m ≤ n

-- Theore statement stating Jasmine can buy a maximum of 9 cards
theorem jasmine_max_cards : max_cards 9 :=
sorry

end jasmine_max_cards_l0_392


namespace intersection_complement_eq_l0_139

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x < 0 }

def B : Set ℝ := { x | x ≤ -1 }

theorem intersection_complement_eq : A ∩ (U \ B) = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end intersection_complement_eq_l0_139


namespace well_depth_is_correct_l0_835

noncomputable def depth_of_well : ℝ :=
  122500

theorem well_depth_is_correct (d t1 : ℝ) : 
  t1 = Real.sqrt (d / 20) ∧ 
  (d / 1100) + t1 = 10 →
  d = depth_of_well := 
by
  sorry

end well_depth_is_correct_l0_835


namespace product_of_roots_of_quadratic_equation_l0_874

theorem product_of_roots_of_quadratic_equation :
  ∀ (x : ℝ), (x^2 + 14 * x + 48 = -4) → (-6) * (-8) = 48 :=
by
  sorry

end product_of_roots_of_quadratic_equation_l0_874


namespace determinant_equality_l0_718

-- Given values p, q, r, s such that the determinant of the first matrix is 5
variables {p q r s : ℝ}

-- Define the determinant condition
def det_condition (p q r s : ℝ) : Prop := p * s - q * r = 5

-- State the theorem that we need to prove
theorem determinant_equality (h : det_condition p q r s) :
  p * (5*r + 2*s) - r * (5*p + 2*q) = 10 :=
sorry

end determinant_equality_l0_718


namespace washing_machine_heavy_wash_usage_l0_534

-- Definition of variables and constants
variables (H : ℕ)                           -- Amount of water used for a heavy wash
def regular_wash : ℕ := 10                   -- Gallons used for a regular wash
def light_wash : ℕ := 2                      -- Gallons used for a light wash
def extra_light_wash : ℕ := light_wash       -- Extra light wash due to bleach

-- Number of each type of wash
def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_washes : ℕ := 2

-- Total water usage
def total_water_usage : ℕ := 
  num_heavy_washes * H + 
  num_regular_washes * regular_wash + 
  num_light_washes * light_wash + 
  num_bleached_washes * extra_light_wash

-- Given total water usage
def given_total_water_usage : ℕ := 76

-- Lean statement to prove the amount of water used for a heavy wash
theorem washing_machine_heavy_wash_usage : total_water_usage H = given_total_water_usage → H = 20 :=
by
  sorry

end washing_machine_heavy_wash_usage_l0_534


namespace emily_gardens_and_seeds_l0_768

variables (total_seeds planted_big_garden tom_seeds lettuce_seeds pepper_seeds tom_gardens lettuce_gardens pepper_gardens : ℕ)

def seeds_left (total_seeds planted_big_garden : ℕ) : ℕ :=
  total_seeds - planted_big_garden

def seeds_used_tomatoes (tom_seeds tom_gardens : ℕ) : ℕ :=
  tom_seeds * tom_gardens

def seeds_used_lettuce (lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  lettuce_seeds * lettuce_gardens

def seeds_used_peppers (pepper_seeds pepper_gardens : ℕ) : ℕ :=
  pepper_seeds * pepper_gardens

def remaining_seeds (total_seeds planted_big_garden tom_seeds tom_gardens lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  seeds_left total_seeds planted_big_garden - (seeds_used_tomatoes tom_seeds tom_gardens + seeds_used_lettuce lettuce_seeds lettuce_gardens)

def total_small_gardens (tom_gardens lettuce_gardens pepper_gardens : ℕ) : ℕ :=
  tom_gardens + lettuce_gardens + pepper_gardens

theorem emily_gardens_and_seeds :
  total_seeds = 42 ∧
  planted_big_garden = 36 ∧
  tom_seeds = 4 ∧
  lettuce_seeds = 3 ∧
  pepper_seeds = 2 ∧
  tom_gardens = 3 ∧
  lettuce_gardens = 2 →
  seeds_used_peppers pepper_seeds pepper_gardens = 0 ∧
  total_small_gardens tom_gardens lettuce_gardens pepper_gardens = 5 :=
by
  sorry

end emily_gardens_and_seeds_l0_768


namespace solve_log_eq_l0_500

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_log_eq (x : ℝ) (hx1 : x + 1 > 0) (hx2 : x - 1 > 0) :
  log_base (x + 1) (x^3 - 9 * x + 8) * log_base (x - 1) (x + 1) = 3 ↔ x = 3 := by
  sorry

end solve_log_eq_l0_500


namespace D_times_C_eq_l0_475

-- Define the matrices C and D
variable (C D : Matrix (Fin 2) (Fin 2) ℚ)

-- Add the conditions
axiom h1 : C * D = C + D
axiom h2 : C * D = ![![15/2, 9/2], ![-6/2, 12/2]]

-- Define the goal
theorem D_times_C_eq : D * C = ![![15/2, 9/2], ![-6/2, 12/2]] :=
sorry

end D_times_C_eq_l0_475


namespace min_value_of_reciprocal_sum_l0_21

variables {m n : ℝ}
variables (h1 : m > 0)
variables (h2 : n > 0)
variables (h3 : m + n = 1)

theorem min_value_of_reciprocal_sum : 
  (1 / m + 1 / n) = 4 :=
by
  sorry

end min_value_of_reciprocal_sum_l0_21


namespace sin_double_angle_half_pi_l0_396

theorem sin_double_angle_half_pi (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1 / 3) : 
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by
  sorry

end sin_double_angle_half_pi_l0_396


namespace natural_number_increased_by_one_l0_970

theorem natural_number_increased_by_one (a : ℕ) 
  (h : (a + 1) ^ 2 - a ^ 2 = 1001) : 
  a = 500 := 
sorry

end natural_number_increased_by_one_l0_970


namespace village_population_l0_829

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 :=
sorry

end village_population_l0_829


namespace find_d_e_f_l0_695

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 3 + 5 / 3)

theorem find_d_e_f :
  ∃ (d e f : ℕ), (y ^ 50 = 3 * y ^ 48 + 10 * y ^ 45 + 9 * y ^ 43 - y ^ 25 + d * y ^ 21 + e * y ^ 19 + f * y ^ 15) 
    ∧ (d + e + f = 119) :=
sorry

end find_d_e_f_l0_695


namespace systematic_sampling_interval_l0_664

-- Definitions based on conditions
def population_size : ℕ := 1000
def sample_size : ℕ := 40

-- Theorem statement 
theorem systematic_sampling_interval :
  population_size / sample_size = 25 :=
by
  sorry

end systematic_sampling_interval_l0_664


namespace penny_frogs_count_l0_28

theorem penny_frogs_count :
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  tree_frogs + poison_frogs + wood_frogs = 78 :=
by
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  show tree_frogs + poison_frogs + wood_frogs = 78
  sorry

end penny_frogs_count_l0_28


namespace convex_polygon_diagonals_l0_794

theorem convex_polygon_diagonals (n : ℕ) (h_n : n = 25) : 
  (n * (n - 3)) / 2 = 275 :=
by
  sorry

end convex_polygon_diagonals_l0_794


namespace inequality_proof_l0_591

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : y > 2 / (x - y)) : x^2 > y^2 + 4 :=
by
  sorry

end inequality_proof_l0_591


namespace slope_intercept_form_of_line_l0_236

theorem slope_intercept_form_of_line :
  ∀ (x y : ℝ), (∀ (a b : ℝ), (a, b) = (0, 4) ∨ (a, b) = (3, 0) → y = - (4 / 3) * x + 4) := 
by
  sorry

end slope_intercept_form_of_line_l0_236


namespace savings_account_after_8_weeks_l0_642

noncomputable def initial_amount : ℕ := 43
noncomputable def weekly_allowance : ℕ := 10
noncomputable def comic_book_cost : ℕ := 3
noncomputable def saved_per_week : ℕ := weekly_allowance - comic_book_cost
noncomputable def weeks : ℕ := 8
noncomputable def savings_in_8_weeks : ℕ := saved_per_week * weeks
noncomputable def total_piggy_bank_after_8_weeks : ℕ := initial_amount + savings_in_8_weeks

theorem savings_account_after_8_weeks : total_piggy_bank_after_8_weeks = 99 :=
by
  have h1 : saved_per_week = 7 := rfl
  have h2 : savings_in_8_weeks = 56 := rfl
  have h3 : total_piggy_bank_after_8_weeks = 99 := rfl
  exact h3

end savings_account_after_8_weeks_l0_642


namespace minimum_value_ineq_l0_357

theorem minimum_value_ineq (x : ℝ) (hx : x >= 4) : x + 4 / (x - 1) >= 5 := by
  sorry

end minimum_value_ineq_l0_357


namespace book_cost_l0_56

-- Definitions from conditions
def priceA : ℝ := 340
def priceB : ℝ := 350
def gain_percent_more : ℝ := 0.05

-- proof problem
theorem book_cost (C : ℝ) (G : ℝ) :
  (priceA - C = G) →
  (priceB - C = (1 + gain_percent_more) * G) →
  C = 140 :=
by
  intros
  sorry

end book_cost_l0_56


namespace abs_neg_three_l0_918

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l0_918


namespace part1_part2_l0_901

variable (a b c : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom eq_sum_square : a^2 + b^2 + 4*c^2 = 3

-- Part 1
theorem part1 : a + b + 2 * c ≤ 3 := 
by
  sorry

-- Additional condition for Part 2
variable (hc : b = 2*c)

-- Part 2
theorem part2 : (1 / a) + (1 / c) ≥ 3 :=
by
  sorry

end part1_part2_l0_901


namespace unique_number_not_in_range_l0_845

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : g a b c d 13 = 13)
  (h2 : g a b c d 31 = 31)
  (h3 : ∀ x, x ≠ -d / c → g a b c d (g a b c d x) = x) :
  ∀ y, ∃! x, g a b c d x = y :=
by {
  sorry
}

end unique_number_not_in_range_l0_845


namespace quadratic_roots_l0_920

theorem quadratic_roots (a b c : ℝ) :
  (∀ (x y : ℝ), ((x, y) = (-2, 12) ∨ (x, y) = (0, -8) ∨ (x, y) = (1, -12) ∨ (x, y) = (3, -8)) → y = a * x^2 + b * x + c) →
  (a * 0^2 + b * 0 + c + 8 = 0) ∧ (a * 3^2 + b * 3 + c + 8 = 0) :=
by sorry

end quadratic_roots_l0_920


namespace inequality_solution_l0_640

theorem inequality_solution {x : ℝ} (h : 2 * x + 1 > x + 2) : x > 1 :=
by
  sorry

end inequality_solution_l0_640


namespace fraction_correct_l0_848

theorem fraction_correct (x : ℚ) (h : (5 / 6) * 576 = x * 576 + 300) : x = 5 / 16 := 
sorry

end fraction_correct_l0_848


namespace remaining_hard_hats_l0_317

theorem remaining_hard_hats 
  (pink_initial : ℕ)
  (green_initial : ℕ)
  (yellow_initial : ℕ)
  (carl_takes_pink : ℕ)
  (john_takes_pink : ℕ)
  (john_takes_green : ℕ) :
  john_takes_green = 2 * john_takes_pink →
  pink_initial = 26 →
  green_initial = 15 →
  yellow_initial = 24 →
  carl_takes_pink = 4 →
  john_takes_pink = 6 →
  ∃ pink_remaining green_remaining yellow_remaining total_remaining, 
    pink_remaining = pink_initial - carl_takes_pink - john_takes_pink ∧
    green_remaining = green_initial - john_takes_green ∧
    yellow_remaining = yellow_initial ∧
    total_remaining = pink_remaining + green_remaining + yellow_remaining ∧
    total_remaining = 43 :=
by
  sorry

end remaining_hard_hats_l0_317


namespace geometric_sequence_terms_l0_100

theorem geometric_sequence_terms
  (a_3 : ℝ) (a_4 : ℝ)
  (h1 : a_3 = 12)
  (h2 : a_4 = 18) :
  ∃ (a_1 a_2 : ℝ) (q: ℝ), 
    a_1 = 16 / 3 ∧ a_2 = 8 ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 := 
by
  sorry

end geometric_sequence_terms_l0_100


namespace inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l0_974

theorem inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d
  (a b c d : ℚ) 
  (h1 : a * d > b * c) 
  (h2 : (a : ℚ) / b > (c : ℚ) / d) : 
  (a / b > (a + c) / (b + d)) ∧ ((a + c) / (b + d) > c / d) :=
by 
  sorry

end inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l0_974


namespace zero_points_of_gx_l0_220

noncomputable def fx (a x : ℝ) : ℝ := (1 / 2) * x^2 - abs (x - 2 * a)
noncomputable def gx (a x : ℝ) : ℝ := 4 * a * x^2 + 2 * x + 1

theorem zero_points_of_gx (a : ℝ) (h : -1 / 4 ≤ a ∧ a ≤ 1 / 4) : 
  ∃ n, (n = 1 ∨ n = 2) ∧ (∃ x1 x2, gx a x1 = 0 ∧ gx a x2 = 0) := 
sorry

end zero_points_of_gx_l0_220


namespace least_integer_in_ratio_1_3_5_l0_674

theorem least_integer_in_ratio_1_3_5 (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 90) (h_ratio : a * 3 = b ∧ a * 5 = c) : a = 10 :=
sorry

end least_integer_in_ratio_1_3_5_l0_674


namespace integer_solution_zero_l0_612

theorem integer_solution_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end integer_solution_zero_l0_612


namespace cindy_pens_ratio_is_one_l0_975

noncomputable def pens_owned_initial : ℕ := 25
noncomputable def pens_given_by_mike : ℕ := 22
noncomputable def pens_given_to_sharon : ℕ := 19
noncomputable def pens_owned_final : ℕ := 75

def pens_before_cindy (initial_pens mike_pens : ℕ) : ℕ := initial_pens + mike_pens
def pens_before_sharon (final_pens sharon_pens : ℕ) : ℕ := final_pens + sharon_pens
def pens_given_by_cindy (pens_before_sharon pens_before_cindy : ℕ) : ℕ := pens_before_sharon - pens_before_cindy
def ratio_pens_given_cindy (cindy_pens pens_before_cindy : ℕ) : ℚ := cindy_pens / pens_before_cindy

theorem cindy_pens_ratio_is_one :
    ratio_pens_given_cindy
        (pens_given_by_cindy (pens_before_sharon pens_owned_final pens_given_to_sharon)
                             (pens_before_cindy pens_owned_initial pens_given_by_mike))
        (pens_before_cindy pens_owned_initial pens_given_by_mike) = 1 := by
    sorry

end cindy_pens_ratio_is_one_l0_975


namespace repetitive_decimals_subtraction_correct_l0_360

noncomputable def repetitive_decimals_subtraction : Prop :=
  let a : ℚ := 4567 / 9999
  let b : ℚ := 1234 / 9999
  let c : ℚ := 2345 / 9999
  a - b - c = 988 / 9999

theorem repetitive_decimals_subtraction_correct : repetitive_decimals_subtraction :=
  by sorry

end repetitive_decimals_subtraction_correct_l0_360


namespace minimum_value_S15_minus_S10_l0_131

theorem minimum_value_S15_minus_S10 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom_seq : ∀ n, S (n + 1) = S n * a (n + 1))
  (h_pos_terms : ∀ n, a n > 0)
  (h_arith_seq : S 10 - 2 * S 5 = 3)
  (h_geom_sub_seq : (S 10 - S 5) * (S 10 - S 5) = S 5 * (S 15 - S 10)) :
  ∃ m, m = 12 ∧ (S 15 - S 10) ≥ m := sorry

end minimum_value_S15_minus_S10_l0_131


namespace yura_picture_dimensions_l0_183

-- Definitions based on the problem conditions
variable {a b : ℕ} -- dimensions of the picture
variable (hasFrame : ℕ × ℕ → Prop) -- definition sketch

-- The main statement to prove
theorem yura_picture_dimensions (h : (a + 2) * (b + 2) - a * b = 2 * a * b) :
  (a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) :=
  sorry

end yura_picture_dimensions_l0_183


namespace total_cost_supplies_l0_964

-- Definitions based on conditions
def cost_bow : ℕ := 5
def cost_vinegar : ℕ := 2
def cost_baking_soda : ℕ := 1
def cost_per_student : ℕ := cost_bow + cost_vinegar + cost_baking_soda
def number_of_students : ℕ := 23

-- Statement to be proven
theorem total_cost_supplies : cost_per_student * number_of_students = 184 := by
  sorry

end total_cost_supplies_l0_964


namespace average_rainfall_l0_814

theorem average_rainfall (r d h : ℕ) (rainfall_eq : r = 450) (days_eq : d = 30) (hours_eq : h = 24) :
  r / (d * h) = 25 / 16 := 
  by 
    -- Insert appropriate proof here
    sorry

end average_rainfall_l0_814


namespace combined_molecular_weight_l0_561

theorem combined_molecular_weight {m1 m2 : ℕ} 
  (MW_C : ℝ) (MW_H : ℝ) (MW_O : ℝ)
  (Butanoic_acid : ℕ × ℕ × ℕ)
  (Propanoic_acid : ℕ × ℕ × ℕ)
  (MW_Butanoic_acid : ℝ)
  (MW_Propanoic_acid : ℝ)
  (weight_Butanoic_acid : ℝ)
  (weight_Propanoic_acid : ℝ)
  (total_weight : ℝ) :
MW_C = 12.01 → MW_H = 1.008 → MW_O = 16.00 →
Butanoic_acid = (4, 8, 2) → MW_Butanoic_acid = (4 * MW_C) + (8 * MW_H) + (2 * MW_O) →
Propanoic_acid = (3, 6, 2) → MW_Propanoic_acid = (3 * MW_C) + (6 * MW_H) + (2 * MW_O) →
m1 = 9 → weight_Butanoic_acid = m1 * MW_Butanoic_acid →
m2 = 5 → weight_Propanoic_acid = m2 * MW_Propanoic_acid →
total_weight = weight_Butanoic_acid + weight_Propanoic_acid →
total_weight = 1163.326 :=
by {
  intros;
  sorry
}

end combined_molecular_weight_l0_561


namespace range_of_m_intersection_l0_926

noncomputable def f (x m : ℝ) : ℝ := (1/x) - (m/(x^2)) - (x/3)

theorem range_of_m_intersection (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m ∈ (Set.Iic 0 ∪ {2/3}) :=
sorry

end range_of_m_intersection_l0_926


namespace domain_lg_sqrt_l0_628

def domain_of_function (x : ℝ) : Prop :=
  1 - x > 0 ∧ x + 2 > 0

theorem domain_lg_sqrt (x : ℝ) : 
  domain_of_function x ↔ -2 < x ∧ x < 1 :=
sorry

end domain_lg_sqrt_l0_628


namespace math_problem_l0_282

noncomputable def alpha_condition (α : ℝ) : Prop :=
  4 * Real.cos α - 2 * Real.sin α = 0

theorem math_problem (α : ℝ) (h : alpha_condition α) :
  (Real.sin α)^3 + (Real.cos α)^3 / (Real.sin α - Real.cos α) = 9 / 5 :=
  sorry

end math_problem_l0_282


namespace unique_solution_k_l0_479

theorem unique_solution_k (k : ℚ) : (∀ x : ℚ, x ≠ -2 → (x + 3)/(k*x - 2) = x) ↔ k = -3/4 :=
sorry

end unique_solution_k_l0_479


namespace find_d_over_a_l0_571

variable (a b c d : ℚ)

-- Conditions
def condition1 : Prop := a / b = 8
def condition2 : Prop := c / b = 4
def condition3 : Prop := c / d = 2 / 3

-- Theorem statement
theorem find_d_over_a (h1 : condition1 a b) (h2 : condition2 c b) (h3 : condition3 c d) : d / a = 3 / 4 :=
by
  -- Proof is omitted
  sorry

end find_d_over_a_l0_571


namespace percentage_of_teachers_with_neither_issue_l0_566

theorem percentage_of_teachers_with_neither_issue 
  (total_teachers : ℕ)
  (teachers_with_bp : ℕ)
  (teachers_with_stress : ℕ)
  (teachers_with_both : ℕ)
  (h1 : total_teachers = 150)
  (h2 : teachers_with_bp = 90)
  (h3 : teachers_with_stress = 60)
  (h4 : teachers_with_both = 30) :
  let neither_issue_teachers := total_teachers - (teachers_with_bp + teachers_with_stress - teachers_with_both)
  let percentage := (neither_issue_teachers * 100) / total_teachers
  percentage = 20 :=
by
  -- skipping the proof
  sorry

end percentage_of_teachers_with_neither_issue_l0_566


namespace min_value_l0_452

theorem min_value (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 1) :
  (a + 1)^2 + 4 * b^2 + 9 * c^2 ≥ 144 / 49 :=
sorry

end min_value_l0_452


namespace dealership_truck_sales_l0_369

theorem dealership_truck_sales (SUVs Trucks : ℕ) (h1 : SUVs = 45) (h2 : 3 * Trucks = 5 * SUVs) : Trucks = 75 :=
by
  sorry

end dealership_truck_sales_l0_369


namespace calculation_2015_l0_12

theorem calculation_2015 :
  2015 ^ 2 - 2016 * 2014 = 1 :=
by
  sorry

end calculation_2015_l0_12


namespace workshop_processing_equation_l0_10

noncomputable def process_equation (x : ℝ) : Prop :=
  (4000 / x - 4200 / (1.5 * x) = 3)

theorem workshop_processing_equation (x : ℝ) (hx : x > 0) :
  process_equation x :=
by
  sorry

end workshop_processing_equation_l0_10


namespace amount_coach_mike_gave_l0_924

-- Definitions from conditions
def cost_of_lemonade : ℕ := 58
def change_received : ℕ := 17

-- Theorem stating the proof problem
theorem amount_coach_mike_gave : cost_of_lemonade + change_received = 75 := by
  sorry

end amount_coach_mike_gave_l0_924


namespace cubic_roots_sum_of_cubes_l0_407

theorem cubic_roots_sum_of_cubes :
  ∀ (a b c : ℝ), 
  (∀ x : ℝ, 9 * x^3 + 14 * x^2 + 2047 * x + 3024 = 0 → (x = a ∨ x = b ∨ x = c)) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = -58198 / 729 :=
by
  intros a b c roota_eqn
  sorry

end cubic_roots_sum_of_cubes_l0_407


namespace percentage_decrease_l0_763

theorem percentage_decrease (original_price new_price : ℝ) (h1 : original_price = 1400) (h2 : new_price = 1064) :
  ((original_price - new_price) / original_price * 100) = 24 :=
by
  sorry

end percentage_decrease_l0_763


namespace arithmetic_sequence_z_value_l0_148

theorem arithmetic_sequence_z_value :
  ∃ z : ℤ, (3 ^ 2 = 9 ∧ 3 ^ 4 = 81) ∧ z = (9 + 81) / 2 :=
by
  -- the proof goes here
  sorry

end arithmetic_sequence_z_value_l0_148


namespace strawberries_taken_out_l0_436

theorem strawberries_taken_out : 
  ∀ (initial_total_strawberries buckets strawberries_left_per_bucket : ℕ),
  initial_total_strawberries = 300 → 
  buckets = 5 → 
  strawberries_left_per_bucket = 40 → 
  (initial_total_strawberries / buckets - strawberries_left_per_bucket = 20) :=
by
  intros initial_total_strawberries buckets strawberries_left_per_bucket h1 h2 h3
  sorry

end strawberries_taken_out_l0_436


namespace cranberries_left_l0_179

def initial_cranberries : ℕ := 60000
def harvested_by_humans : ℕ := initial_cranberries * 40 / 100
def eaten_by_elk : ℕ := 20000

theorem cranberries_left (c : ℕ) : c = initial_cranberries - harvested_by_humans - eaten_by_elk → c = 16000 := by
  sorry

end cranberries_left_l0_179


namespace max_product_of_three_numbers_l0_351

theorem max_product_of_three_numbers (n : ℕ) (h_n_pos : 0 < n) :
  ∃ a b c : ℕ, (a + b + c = 3 * n + 1) ∧ (∀ a' b' c' : ℕ,
        (a' + b' + c' = 3 * n + 1) →
        a' * b' * c' ≤ a * b * c) ∧
    (a * b * c = n^3 + n^2) :=
by
  sorry

end max_product_of_three_numbers_l0_351


namespace line_through_points_l0_646

theorem line_through_points (x1 y1 x2 y2 : ℕ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (3, 8)) : 
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = 2 := 
by
  sorry

end line_through_points_l0_646


namespace population_in_2001_l0_313

-- Define the populations at specific years
def pop_2000 := 50
def pop_2002 := 146
def pop_2003 := 350

-- Define the population difference condition
def pop_condition (n : ℕ) (pop : ℕ → ℕ) :=
  pop (n + 3) - pop n = 3 * pop (n + 2)

-- Given that the population condition holds, and specific populations are known,
-- the population in the year 2001 is 100
theorem population_in_2001 :
  (∃ (pop : ℕ → ℕ), pop 2000 = pop_2000 ∧ pop 2002 = pop_2002 ∧ pop 2003 = pop_2003 ∧ 
    pop_condition 2000 pop) → ∃ (pop : ℕ → ℕ), pop 2001 = 100 :=
by
  -- Placeholder for the actual proof
  sorry

end population_in_2001_l0_313


namespace fifth_number_21st_row_is_809_l0_427

-- Define the sequence of positive odd numbers
def nth_odd_number (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the last odd number in the nth row
def last_odd_number_in_row (n : ℕ) : ℕ :=
  nth_odd_number (n * n)

-- Define the position of the 5th number in the 21st row
def pos_5th_in_21st_row : ℕ :=
  let sum_first_20_rows := 400
  sum_first_20_rows + 5

-- The 5th number from the left in the 21st row
def fifth_number_in_21st_row : ℕ :=
  nth_odd_number pos_5th_in_21st_row

-- The proof statement
theorem fifth_number_21st_row_is_809 : fifth_number_in_21st_row = 809 :=
by
  -- proof omitted
  sorry

end fifth_number_21st_row_is_809_l0_427


namespace gcd_459_357_l0_453

-- Define the numbers involved
def num1 := 459
def num2 := 357

-- State the proof problem
theorem gcd_459_357 : Int.gcd num1 num2 = 51 := by
  sorry

end gcd_459_357_l0_453


namespace negation_of_existential_statement_l0_471

theorem negation_of_existential_statement :
  (¬ ∃ x : ℝ, x ≥ 1 ∨ x > 2) ↔ ∀ x : ℝ, x < 1 :=
by
  sorry

end negation_of_existential_statement_l0_471


namespace initial_investment_proof_l0_979

-- Definitions for the conditions
def initial_investment_A : ℝ := sorry
def contribution_B : ℝ := 15750
def profit_ratio_A : ℝ := 2
def profit_ratio_B : ℝ := 3
def time_A : ℝ := 12
def time_B : ℝ := 4

-- Lean statement to prove
theorem initial_investment_proof : initial_investment_A * time_A * profit_ratio_B = contribution_B * time_B * profit_ratio_A → initial_investment_A = 1750 :=
by
  sorry

end initial_investment_proof_l0_979


namespace sum_powers_mod_7_l0_709

theorem sum_powers_mod_7 :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
  sorry

end sum_powers_mod_7_l0_709


namespace monotonic_conditions_fixed_point_property_l0_786

noncomputable
def f (x a b c : ℝ) : ℝ := x^3 - a * x^2 - b * x + c

theorem monotonic_conditions (a b c : ℝ) :
  a = 0 ∧ c = 0 ∧ b ≤ 3 ↔ ∀ x : ℝ, (x ≥ 1 → (f x a b c) ≥ 1) → ∀ x y: ℝ, (x ≥ y ↔ f x a b c ≤ f y a b c) := sorry

theorem fixed_point_property (a b c : ℝ) :
  (∀ x : ℝ, (x ≥ 1 ∧ (f x a b c) ≥ 1) → f (f x a b c) a b c = x) ↔ (f x 0 b 0 = x) := sorry

end monotonic_conditions_fixed_point_property_l0_786


namespace total_candles_in_small_boxes_l0_744

-- Definitions of the conditions
def num_small_boxes_per_big_box := 4
def num_big_boxes := 50
def candles_per_small_box := 40

-- The total number of small boxes
def total_small_boxes : Nat := num_small_boxes_per_big_box * num_big_boxes

-- The statement to prove the total number of candles in all small boxes is 8000
theorem total_candles_in_small_boxes : candles_per_small_box * total_small_boxes = 8000 :=
by 
  sorry

end total_candles_in_small_boxes_l0_744


namespace total_kids_played_tag_with_l0_503

theorem total_kids_played_tag_with : 
  let kids_mon : Nat := 12
  let kids_tues : Nat := 7
  let kids_wed : Nat := 15
  let kids_thurs : Nat := 10
  let kids_fri : Nat := 18
  (kids_mon + kids_tues + kids_wed + kids_thurs + kids_fri) = 62 := by
  sorry

end total_kids_played_tag_with_l0_503


namespace binary_division_remainder_correct_l0_36

-- Define the last two digits of the binary number
def b_1 : ℕ := 1
def b_0 : ℕ := 1

-- Define the function to calculate the remainder when dividing by 4
def binary_remainder (b1 b0 : ℕ) : ℕ := 2 * b1 + b0

-- Expected remainder in binary form
def remainder_in_binary : ℕ := 0b11  -- '11' in binary is 3 in decimal

-- The theorem to prove
theorem binary_division_remainder_correct :
  binary_remainder b_1 b_0 = remainder_in_binary :=
by
  -- Proof goes here
  sorry

end binary_division_remainder_correct_l0_36


namespace additional_time_to_walk_1_mile_l0_449

open Real

noncomputable def additional_time_per_mile
  (distance_child : ℝ) (time_child : ℝ)
  (distance_elderly : ℝ) (time_elderly : ℝ)
  : ℝ :=
  let speed_child := distance_child / time_child
  let time_per_mile_child := (time_child * 60) / distance_child
  let speed_elderly := distance_elderly / time_elderly
  let time_per_mile_elderly := (time_elderly * 60) / distance_elderly
  time_per_mile_elderly - time_per_mile_child

theorem additional_time_to_walk_1_mile
  (h1 : 15 = 15) (h2 : 3.5 = 3.5)
  (h3 : 10 = 10) (h4 : 4 = 4)
  : additional_time_per_mile 15 3.5 10 4 = 10 :=
  by
    sorry

end additional_time_to_walk_1_mile_l0_449


namespace a10_is_b55_l0_288

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ := 2 * n - 1

-- Define the new sequence b_n according to the given insertion rules
def b (k : ℕ) : ℕ := sorry

-- Prove that if a_10 = 19, then 19 is the 55th term in the new sequence b_n
theorem a10_is_b55 : b 55 = a 10 := sorry

end a10_is_b55_l0_288


namespace percentage_increase_l0_212

theorem percentage_increase (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 480) :
  ((new_price - original_price) / original_price) * 100 = 60 :=
by
  -- Proof goes here
  sorry

end percentage_increase_l0_212


namespace common_ratio_geometric_sequence_l0_916

theorem common_ratio_geometric_sequence 
  (a1 : ℝ) 
  (q : ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h2 : 8 * S 6 = 7 * S 3) 
  (hq : q ≠ 1) : 
  q = -1 / 2 := 
sorry

end common_ratio_geometric_sequence_l0_916


namespace least_value_y_l0_376

theorem least_value_y : ∃ y : ℝ, (3 * y ^ 3 + 3 * y ^ 2 + 5 * y + 1 = 5) ∧ ∀ z : ℝ, (3 * z ^ 3 + 3 * z ^ 2 + 5 * z + 1 = 5) → y ≤ z :=
sorry

end least_value_y_l0_376


namespace initial_men_l0_824

variable (x : ℕ)

-- Conditions
def condition1 (x : ℕ) : Prop :=
  -- The hostel had provisions for x men for 28 days.
  true

def condition2 (x : ℕ) : Prop :=
  -- If 50 men left, the food would last for 35 days for the remaining x - 50 men.
  (x - 50) * 35 = x * 28

-- Theorem to prove
theorem initial_men (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 250 :=
by
  sorry

end initial_men_l0_824


namespace leap_day_2040_is_tuesday_l0_743

def days_in_non_leap_year := 365
def days_in_leap_year := 366
def leap_years_between_2000_and_2040 := 10

def total_days_between_2000_and_2040 := 
  30 * days_in_non_leap_year + leap_years_between_2000_and_2040 * days_in_leap_year

theorem leap_day_2040_is_tuesday :
  (total_days_between_2000_and_2040 % 7) = 0 :=
by
  sorry

end leap_day_2040_is_tuesday_l0_743


namespace inequality_proof_l0_820

theorem inequality_proof (n : ℕ) (a : ℝ) (h₀ : n > 1) (h₁ : 0 < a) (h₂ : a < 1) : 
  1 + a < (1 + a / n) ^ n ∧ (1 + a / n) ^ n < (1 + a / (n + 1)) ^ (n + 1) := 
sorry

end inequality_proof_l0_820


namespace twenty_five_percent_of_x_l0_568

-- Define the number x and the conditions
variable (x : ℝ)
variable (h : x - (3/4) * x = 100)

-- The theorem statement
theorem twenty_five_percent_of_x : (1/4) * x = 100 :=
by 
  -- Assume x satisfies the given condition
  sorry

end twenty_five_percent_of_x_l0_568


namespace boat_distance_downstream_l0_418

theorem boat_distance_downstream
  (speed_boat : ℕ)
  (speed_stream : ℕ)
  (time_downstream : ℕ)
  (h1 : speed_boat = 22)
  (h2 : speed_stream = 5)
  (h3 : time_downstream = 8) :
  speed_boat + speed_stream * time_downstream = 216 :=
by
  sorry

end boat_distance_downstream_l0_418


namespace largest_number_after_removal_l0_585

theorem largest_number_after_removal :
  ∀ (s : Nat), s = 1234567891011121314151617181920 -- representing the start of the sequence
  → true
  := by
    sorry

end largest_number_after_removal_l0_585


namespace total_hours_is_900_l0_538

-- Definitions for the video length, speeds, and number of videos watched
def video_length : ℕ := 100
def lila_speed : ℕ := 2
def roger_speed : ℕ := 1
def num_videos : ℕ := 6

-- Definition of total hours watched
def total_hours_watched : ℕ :=
  let lila_time_per_video := video_length / lila_speed
  let roger_time_per_video := video_length / roger_speed
  (lila_time_per_video * num_videos) + (roger_time_per_video * num_videos)

-- Prove that the total hours watched is 900
theorem total_hours_is_900 : total_hours_watched = 900 :=
by
  -- Proving the equation step-by-step
  sorry

end total_hours_is_900_l0_538


namespace mr_hernandez_tax_l0_301

theorem mr_hernandez_tax :
  let taxable_income := 42500
  let resident_months := 9
  let standard_deduction := if resident_months > 6 then 5000 else 0
  let adjusted_income := taxable_income - standard_deduction
  let tax_bracket_1 := min adjusted_income 10000 * 0.01
  let tax_bracket_2 := min (max (adjusted_income - 10000) 0) 20000 * 0.03
  let tax_bracket_3 := min (max (adjusted_income - 30000) 0) 30000 * 0.05
  let total_tax_before_credit := tax_bracket_1 + tax_bracket_2 + tax_bracket_3
  let tax_credit := if resident_months < 10 then 500 else 0
  total_tax_before_credit - tax_credit = 575 := 
by
  sorry
  
end mr_hernandez_tax_l0_301


namespace charge_per_call_proof_l0_981

-- Define the conditions as given in the problem
def fixed_rental : ℝ := 350
def free_calls_per_month : ℕ := 200
def charge_per_call_exceed_200 (x : ℝ) (calls : ℕ) : ℝ := 
  if calls > 200 then (calls - 200) * x else 0

def charge_per_call_exceed_400 : ℝ := 1.6
def discount_rate : ℝ := 0.28
def february_calls : ℕ := 150
def march_calls : ℕ := 250
def march_discount (x : ℝ) : ℝ := x * (1 - discount_rate)
def total_march_charge (x : ℝ) : ℝ := 
  fixed_rental + charge_per_call_exceed_200 (march_discount x) march_calls

-- Prove the correct charge per call when calls exceed 200 per month
theorem charge_per_call_proof (x : ℝ) : 
  charge_per_call_exceed_200 x february_calls = 0 ∧ 
  total_march_charge x = fixed_rental + (march_calls - free_calls_per_month) * (march_discount x) → 
  x = x := 
by { 
  sorry 
}

end charge_per_call_proof_l0_981


namespace play_number_of_children_l0_370

theorem play_number_of_children (A C : ℕ) (ticket_price_adult : ℕ) (ticket_price_child : ℕ)
    (total_people : ℕ) (total_money : ℕ)
    (h1 : ticket_price_adult = 8)
    (h2 : ticket_price_child = 1)
    (h3 : total_people = 22)
    (h4 : total_money = 50)
    (h5 : A + C = total_people)
    (h6 : ticket_price_adult * A + ticket_price_child * C = total_money) :
    C = 18 := sorry

end play_number_of_children_l0_370


namespace cookies_initial_count_l0_383

theorem cookies_initial_count (C : ℕ) (h1 : C / 8 = 8) : C = 64 :=
by
  sorry

end cookies_initial_count_l0_383


namespace ratio_of_m_l0_91

theorem ratio_of_m (a b m m1 m2 : ℝ)
  (h1 : a * m^2 + b * m + c = 0)
  (h2 : (a / b + b / a) = 3 / 7)
  (h3 : a + b = (3 * m - 2) / m)
  (h4 : a * b = 7 / m)
  (h5 : (a + b)^2 = ab / (m * (7/ m)) - 2) :
  (m1 + m2 = 21) ∧ (m1 * m2 = 4) → 
  (m1/m2 + m2/m1 = 108.25) := sorry

end ratio_of_m_l0_91


namespace find_g_two_fifths_l0_467

noncomputable def g : ℝ → ℝ :=
sorry -- The function g(x) is not explicitly defined.

theorem find_g_two_fifths :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g x = 0 → g 0 = 0) ∧
  (∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 5) = g x / 3)
  → g (2 / 5) = 1 / 3 :=
sorry

end find_g_two_fifths_l0_467


namespace prime_p_sum_of_squares_l0_872

theorem prime_p_sum_of_squares (p : ℕ) (hp : p.Prime) 
  (h : ∃ (a : ℕ), 2 * p = a^2 + (a + 1)^2 + (a + 2)^2 + (a + 3)^2) : 
  36 ∣ (p - 7) :=
by 
  sorry

end prime_p_sum_of_squares_l0_872


namespace divisibility_problem_l0_202

theorem divisibility_problem (q : ℕ) (hq : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬((q + 2)^(q - 3) + 1) % (q - 4) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % q = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 6) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 3) = 0 := sorry

end divisibility_problem_l0_202


namespace investment_C_120000_l0_533

noncomputable def investment_C (P_B P_A_difference : ℕ) (investment_A investment_B : ℕ) : ℕ :=
  let P_A := (P_B * investment_A) / investment_B
  let P_C := P_A + P_A_difference
  (P_C * investment_B) / P_B

theorem investment_C_120000
  (investment_A investment_B P_B P_A_difference : ℕ)
  (hA : investment_A = 8000)
  (hB : investment_B = 10000)
  (hPB : P_B = 1400)
  (hPA_difference : P_A_difference = 560) :
  investment_C P_B P_A_difference investment_A investment_B = 120000 :=
by
  sorry

end investment_C_120000_l0_533


namespace husband_age_l0_535

theorem husband_age (a b : ℕ) (w_age h_age : ℕ) (ha : a > 0) (hb : b > 0) 
  (hw_age : w_age = 10 * a + b) 
  (hh_age : h_age = 10 * b + a) 
  (h_older : h_age > w_age)
  (h_difference : 9 * (b - a) = a + b) :
  h_age = 54 :=
by
  sorry

end husband_age_l0_535


namespace pass_in_both_subjects_l0_27

variable (F_H F_E F_HE : ℝ)

theorem pass_in_both_subjects (h1 : F_H = 20) (h2 : F_E = 70) (h3 : F_HE = 10) :
  100 - ((F_H + F_E) - F_HE) = 20 :=
by
  sorry

end pass_in_both_subjects_l0_27


namespace solution_2016_121_solution_2016_144_l0_20

-- Definitions according to the given conditions
def delta_fn (f : ℕ → ℕ → ℕ) :=
  (∀ a b : ℕ, f (a + b) b = f a b + 1) ∧ (∀ a b : ℕ, f a b * f b a = 0)

-- Proof objectives
theorem solution_2016_121 (f : ℕ → ℕ → ℕ) (h : delta_fn f) : f 2016 121 = 16 :=
sorry

theorem solution_2016_144 (f : ℕ → ℕ → ℕ) (h : delta_fn f) : f 2016 144 = 13 :=
sorry

end solution_2016_121_solution_2016_144_l0_20


namespace black_female_pigeons_more_than_males_l0_626

theorem black_female_pigeons_more_than_males:
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  black_female_pigeons - black_male_pigeons = 21 := by
{
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  show black_female_pigeons - black_male_pigeons = 21
  sorry
}

end black_female_pigeons_more_than_males_l0_626


namespace period_of_f_l0_659

noncomputable def f (x : ℝ) : ℝ := (Real.tan (x/3)) + (Real.sin x)

theorem period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
  sorry

end period_of_f_l0_659


namespace total_pizza_pieces_l0_70

-- Definitions based on the conditions
def pieces_per_pizza : Nat := 6
def pizzas_per_student : Nat := 20
def number_of_students : Nat := 10

-- Statement of the theorem
theorem total_pizza_pieces :
  pieces_per_pizza * pizzas_per_student * number_of_students = 1200 :=
by
  -- Placeholder for the proof
  sorry

end total_pizza_pieces_l0_70


namespace cubic_sum_divisible_by_9_l0_249

theorem cubic_sum_divisible_by_9 (n : ℕ) (hn : n > 0) : 
  ∃ k, n^3 + (n+1)^3 + (n+2)^3 = 9*k := by
  sorry

end cubic_sum_divisible_by_9_l0_249


namespace pascal_50_5th_element_is_22050_l0_214

def pascal_fifth_element (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_50_5th_element_is_22050 :
  pascal_fifth_element 50 4 = 22050 :=
by
  -- Calculation steps would go here
  sorry

end pascal_50_5th_element_is_22050_l0_214


namespace discriminant_is_four_l0_621

-- Define the quadratic equation components
def quadratic_a (a : ℝ) := 1
def quadratic_b (a : ℝ) := 2 * a
def quadratic_c (a : ℝ) := a^2 - 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) := quadratic_b a ^ 2 - 4 * quadratic_a a * quadratic_c a

-- Statement to prove: The discriminant is 4
theorem discriminant_is_four (a : ℝ) : discriminant a = 4 :=
by {
  sorry
}

end discriminant_is_four_l0_621


namespace equivalent_functions_l0_754

theorem equivalent_functions :
  ∀ (x t : ℝ), (x^2 - 2*x - 1 = t^2 - 2*t + 1) := 
by
  intros x t
  sorry

end equivalent_functions_l0_754


namespace eggs_left_for_sunny_side_up_l0_424

-- Given conditions:
def ordered_dozen_eggs : ℕ := 3 * 12
def eggs_used_for_crepes (total_eggs : ℕ) : ℕ := total_eggs * 1 / 4
def eggs_after_crepes (total_eggs : ℕ) (used_for_crepes : ℕ) : ℕ := total_eggs - used_for_crepes
def eggs_used_for_cupcakes (remaining_eggs : ℕ) : ℕ := remaining_eggs * 2 / 3
def eggs_left (remaining_eggs : ℕ) (used_for_cupcakes : ℕ) : ℕ := remaining_eggs - used_for_cupcakes

-- Proposition:
theorem eggs_left_for_sunny_side_up : 
  eggs_left (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs)) 
            (eggs_used_for_cupcakes (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs))) = 9 :=
sorry

end eggs_left_for_sunny_side_up_l0_424


namespace coloring_integers_l0_168

theorem coloring_integers 
  (color : ℤ → ℕ) 
  (x y : ℤ) 
  (hx : x % 2 = 1) 
  (hy : y % 2 = 1) 
  (h_neq : |x| ≠ |y|) 
  (h_color_range : ∀ n : ℤ, color n < 4) :
  ∃ a b : ℤ, color a = color b ∧ (a - b = x ∨ a - b = y ∨ a - b = x + y ∨ a - b = x - y) :=
sorry

end coloring_integers_l0_168


namespace hawkeye_fewer_mainecoons_than_gordon_l0_43

-- Definitions based on conditions
def JamiePersians : ℕ := 4
def JamieMaineCoons : ℕ := 2
def GordonPersians : ℕ := JamiePersians / 2
def GordonMaineCoons : ℕ := JamieMaineCoons + 1
def TotalCats : ℕ := 13
def JamieTotalCats : ℕ := JamiePersians + JamieMaineCoons
def GordonTotalCats : ℕ := GordonPersians + GordonMaineCoons
def JamieAndGordonTotalCats : ℕ := JamieTotalCats + GordonTotalCats
def HawkeyeTotalCats : ℕ := TotalCats - JamieAndGordonTotalCats
def HawkeyePersians : ℕ := 0
def HawkeyeMaineCoons : ℕ := HawkeyeTotalCats - HawkeyePersians

-- Theorem statement to prove: Hawkeye owns 1 fewer Maine Coon than Gordon
theorem hawkeye_fewer_mainecoons_than_gordon : HawkeyeMaineCoons + 1 = GordonMaineCoons :=
by
  sorry

end hawkeye_fewer_mainecoons_than_gordon_l0_43


namespace max_sum_of_factors_l0_635

theorem max_sum_of_factors (h k : ℕ) (h_even : Even h) (prod_eq : h * k = 24) : h + k ≤ 14 :=
sorry

end max_sum_of_factors_l0_635


namespace xyz_expression_l0_190

theorem xyz_expression (x y z : ℝ) 
  (h1 : x^2 - y * z = 2)
  (h2 : y^2 - z * x = 2)
  (h3 : z^2 - x * y = 2) :
  x * y + y * z + z * x = -2 :=
sorry

end xyz_expression_l0_190


namespace range_of_set_l0_530

theorem range_of_set (a b c : ℕ) (h1 : a = 2) (h2 : b = 6) (h3 : 2 ≤ c ∧ c ≤ 10) (h4 : (a + b + c) / 3 = 6) : (c - a) = 8 :=
by
  sorry

end range_of_set_l0_530


namespace james_toys_l0_956

-- Define the conditions and the problem statement
theorem james_toys (x : ℕ) (h1 : ∀ x, 2 * x = 60 - x) : x = 20 :=
sorry

end james_toys_l0_956


namespace value_two_std_dev_less_than_mean_l0_923

-- Define the given conditions for the problem.
def mean : ℝ := 15
def std_dev : ℝ := 1.5

-- Define the target value that should be 2 standard deviations less than the mean.
def target_value := mean - 2 * std_dev

-- State the theorem that represents the proof problem.
theorem value_two_std_dev_less_than_mean : target_value = 12 := by
  sorry

end value_two_std_dev_less_than_mean_l0_923


namespace div_by_27_l0_406

theorem div_by_27 (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
sorry

end div_by_27_l0_406


namespace proof_d_e_f_value_l0_685

theorem proof_d_e_f_value
  (a b c d e f : ℝ)
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.75) :
  d * e * f = 250 :=
sorry

end proof_d_e_f_value_l0_685


namespace maximum_candies_karlson_l0_473

theorem maximum_candies_karlson (n : ℕ) (h_n : n = 40) :
  ∃ k, k = 780 :=
by
  sorry

end maximum_candies_karlson_l0_473


namespace geometric_sequence_proof_l0_616

-- Define a geometric sequence with first term 1 and common ratio q with |q| ≠ 1
noncomputable def geometric_sequence (q : ℝ) (n : ℕ) : ℝ :=
  if h : |q| ≠ 1 then (1 : ℝ) * q ^ (n - 1) else 0

-- m should be 11 given the conditions
theorem geometric_sequence_proof (q : ℝ) (m : ℕ) (h : |q| ≠ 1) 
  (hm : geometric_sequence q m = geometric_sequence q 1 * geometric_sequence q 2 * geometric_sequence q 3 * geometric_sequence q 4 * geometric_sequence q 5 ) : 
  m = 11 :=
by
  sorry

end geometric_sequence_proof_l0_616


namespace sum_a4_a5_a6_l0_813

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 3 = -10

-- Definition of arithmetic sequence
axiom h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- Proof problem statement
theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = -66 :=
by
  sorry

end sum_a4_a5_a6_l0_813


namespace find_x_y_sum_of_squares_l0_445

theorem find_x_y_sum_of_squares :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (xy + x + y = 47) ∧ (x^2 * y + x * y^2 = 506) ∧ (x^2 + y^2 = 101) :=
by {
  sorry
}

end find_x_y_sum_of_squares_l0_445


namespace find_A_l0_529

theorem find_A (A B : ℕ) (h1 : A + B = 1149) (h2 : A = 8 * B + 24) : A = 1024 :=
by
  sorry

end find_A_l0_529


namespace isosceles_obtuse_triangle_angle_correct_l0_764

noncomputable def isosceles_obtuse_triangle_smallest_angle (A B C : ℝ) (h1 : A = 1.3 * 90) (h2 : B = C) (h3 : A + B + C = 180) : ℝ :=
  (180 - A) / 2

theorem isosceles_obtuse_triangle_angle_correct 
  (A B C : ℝ)
  (h1 : A = 1.3 * 90)
  (h2 : B = C)
  (h3 : A + B + C = 180) :
  isosceles_obtuse_triangle_smallest_angle A B C h1 h2 h3 = 31.5 :=
sorry

end isosceles_obtuse_triangle_angle_correct_l0_764


namespace smallest_constant_inequality_l0_905

theorem smallest_constant_inequality :
  ∀ (x y : ℝ), 1 + (x + y)^2 ≤ (4 / 3) * (1 + x^2) * (1 + y^2) :=
by
  intro x y
  sorry

end smallest_constant_inequality_l0_905


namespace scientific_notation_correct_l0_990

theorem scientific_notation_correct :
  0.000000007 = 7 * 10^(-9) :=
by
  sorry

end scientific_notation_correct_l0_990


namespace incorrect_correlation_coefficient_range_l0_255

noncomputable def regression_analysis_conditions 
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) : Prop :=
  non_deterministic_relationship ∧
  correlation_coefficient_range ∧
  perfect_correlation ∧
  correlation_coefficient_sign

theorem incorrect_correlation_coefficient_range
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) :
  regression_analysis_conditions 
    non_deterministic_relationship 
    correlation_coefficient_range 
    perfect_correlation 
    correlation_coefficient_sign →
  ¬ correlation_coefficient_range :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end incorrect_correlation_coefficient_range_l0_255


namespace lashawn_three_times_kymbrea_l0_161

-- Definitions based on the conditions
def kymbrea_collection (months : ℕ) : ℕ := 50 + 3 * months
def lashawn_collection (months : ℕ) : ℕ := 20 + 5 * months

-- Theorem stating the core of the problem
theorem lashawn_three_times_kymbrea (x : ℕ) 
  (h : lashawn_collection x = 3 * kymbrea_collection x) : x = 33 := 
sorry

end lashawn_three_times_kymbrea_l0_161


namespace coloring_ways_l0_682

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

end coloring_ways_l0_682


namespace sum_of_squares_l0_863

theorem sum_of_squares (x y z w a b c d : ℝ) (h1: x * y = a) (h2: x * z = b) (h3: y * z = c) (h4: x * w = d) :
  x^2 + y^2 + z^2 + w^2 = (ab + bd + da)^2 / abd := 
by
  sorry

end sum_of_squares_l0_863


namespace total_pennies_donated_l0_856

theorem total_pennies_donated:
  let cassandra_pennies := 5000
  let james_pennies := cassandra_pennies - 276
  let stephanie_pennies := 2 * james_pennies
  cassandra_pennies + james_pennies + stephanie_pennies = 19172 :=
by
  sorry

end total_pennies_donated_l0_856


namespace max_sum_11xy_3x_2012yz_l0_631

theorem max_sum_11xy_3x_2012yz (x y z : ℕ) (h : x + y + z = 1000) : 
  11 * x * y + 3 * x + 2012 * y * z ≤ 503000000 :=
sorry

end max_sum_11xy_3x_2012yz_l0_631


namespace arithmetic_sequence_fifth_term_l0_419

theorem arithmetic_sequence_fifth_term (a1 d : ℕ) (a_n : ℕ → ℕ) 
  (h_a1 : a1 = 2) (h_d : d = 1) (h_a_n : ∀ n : ℕ, a_n n = a1 + (n-1) * d) : 
  a_n 5 = 6 := 
    by
    -- Given the conditions, we need to prove a_n evaluated at 5 is equal to 6.
    sorry

end arithmetic_sequence_fifth_term_l0_419


namespace island_not_Maya_l0_662

variable (A B : Prop)
variable (IslandMaya : Prop)
variable (Liar : Prop → Prop)
variable (TruthTeller : Prop → Prop)

-- A's statement: "We are both liars, and this island is called Maya."
axiom A_statement : Liar A ∧ Liar B ∧ IslandMaya

-- B's statement: "At least one of us is a liar, and this island is not called Maya."
axiom B_statement : (Liar A ∨ Liar B) ∧ ¬IslandMaya

theorem island_not_Maya : ¬IslandMaya := by
  sorry

end island_not_Maya_l0_662


namespace boys_count_in_dance_class_l0_337

theorem boys_count_in_dance_class
  (total_students : ℕ) 
  (ratio_girls_to_boys : ℕ) 
  (ratio_boys_to_girls: ℕ)
  (total_students_eq : total_students = 35)
  (ratio_eq : ratio_girls_to_boys = 3 ∧ ratio_boys_to_girls = 4) : 
  ∃ boys : ℕ, boys = 20 :=
by
  let k := total_students / (ratio_girls_to_boys + ratio_boys_to_girls)
  have girls := ratio_girls_to_boys * k
  have boys := ratio_boys_to_girls * k
  use boys
  sorry

end boys_count_in_dance_class_l0_337


namespace remainder_of_2_pow_33_mod_9_l0_606

theorem remainder_of_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end remainder_of_2_pow_33_mod_9_l0_606


namespace isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l0_987

open Real

theorem isosceles_triangle_of_sine_ratio (a b c : ℝ) (A B C : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h1 : a = b * sin C + c * cos B) :
  C = π / 4 :=
sorry

theorem obtuse_triangle_of_tan_sum_neg (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h_tan_sum : tan A + tan B + tan C < 0) :
  ∃ (E : ℝ), (A = E ∨ B = E ∨ C = E) ∧ π / 2 < E :=
sorry

end isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l0_987


namespace diff_square_mental_math_l0_487

theorem diff_square_mental_math :
  75 ^ 2 - 45 ^ 2 = 3600 :=
by
  -- The proof would go here
  sorry

end diff_square_mental_math_l0_487


namespace perimeter_of_figure_l0_345

theorem perimeter_of_figure (x : ℕ) (h : x = 3) : 
  let sides := [x, x + 1, 6, 10]
  (sides.sum = 23) := by 
  sorry

end perimeter_of_figure_l0_345


namespace largest_x_value_l0_37

theorem largest_x_value (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + x * z + y * z = 9) : x ≤ 4 := 
sorry

end largest_x_value_l0_37


namespace garden_length_l0_478

-- Define the perimeter and breadth
def perimeter : ℕ := 900
def breadth : ℕ := 190

-- Define a function to calculate the length using given conditions
def length (P : ℕ) (B : ℕ) : ℕ := (P / 2) - B

-- Theorem stating that for the given perimeter and breadth, the length is 260.
theorem garden_length : length perimeter breadth = 260 :=
by
  -- placeholder for proof
  sorry

end garden_length_l0_478


namespace last_two_digits_of_7_pow_5_pow_6_l0_219

theorem last_two_digits_of_7_pow_5_pow_6 : (7 ^ (5 ^ 6)) % 100 = 7 := 
  sorry

end last_two_digits_of_7_pow_5_pow_6_l0_219


namespace binary_arithmetic_l0_76

theorem binary_arithmetic :
  let a := 0b1101
  let b := 0b0110
  let c := 0b1011
  let d := 0b1001
  a + b - c + d = 0b10001 := by
sorry

end binary_arithmetic_l0_76


namespace neha_mother_age_l0_620

variable (N M : ℕ)

theorem neha_mother_age (h1 : M - 12 = 4 * (N - 12)) (h2 : M + 12 = 2 * (N + 12)) : M = 60 := by
  sorry

end neha_mother_age_l0_620


namespace origami_papers_per_cousin_l0_605

theorem origami_papers_per_cousin (total_papers : ℕ) (num_cousins : ℕ) (same_papers_each : ℕ) 
  (h1 : total_papers = 48) 
  (h2 : num_cousins = 6) 
  (h3 : same_papers_each = total_papers / num_cousins) : 
  same_papers_each = 8 := 
by 
  sorry

end origami_papers_per_cousin_l0_605


namespace kathryn_gave_56_pencils_l0_728

-- Define the initial and total number of pencils
def initial_pencils : ℕ := 9
def total_pencils : ℕ := 65

-- Define the number of pencils Kathryn gave to Anthony
def pencils_given : ℕ := total_pencils - initial_pencils

-- Prove that Kathryn gave Anthony 56 pencils
theorem kathryn_gave_56_pencils : pencils_given = 56 :=
by
  -- Proof is omitted as per the requirement
  sorry

end kathryn_gave_56_pencils_l0_728


namespace kataleya_total_amount_paid_l0_663

/-- A store offers a $2 discount for every $10 purchase on any item in the store.
Kataleya went to the store and bought 400 peaches sold at forty cents each.
Prove that the total amount of money she paid at the store for the fruits is $128. -/
theorem kataleya_total_amount_paid : 
  let price_per_peach : ℝ := 0.40
  let number_of_peaches : ℝ := 400 
  let total_cost : ℝ := number_of_peaches * price_per_peach
  let discount_per_10_dollars : ℝ := 2
  let number_of_discounts := total_cost / 10
  let total_discount := number_of_discounts * discount_per_10_dollars
  let amount_paid := total_cost - total_discount
  amount_paid = 128 :=
by
  sorry

end kataleya_total_amount_paid_l0_663


namespace european_stamps_cost_l0_502

def prices : String → ℕ 
| "Italy"   => 7
| "Japan"   => 7
| "Germany" => 5
| "China"   => 5
| _ => 0

def stamps_1950s : String → ℕ 
| "Italy"   => 5
| "Germany" => 8
| "China"   => 10
| "Japan"   => 6
| _ => 0

def stamps_1960s : String → ℕ 
| "Italy"   => 9
| "Germany" => 12
| "China"   => 5
| "Japan"   => 10
| _ => 0

def total_cost (stamps : String → ℕ) (price : String → ℕ) : ℕ :=
  (stamps "Italy" * price "Italy" +
   stamps "Germany" * price "Germany") 

theorem european_stamps_cost : total_cost stamps_1950s prices + total_cost stamps_1960s prices = 198 :=
by
  sorry

end european_stamps_cost_l0_502


namespace total_vehicles_in_lanes_l0_14

theorem total_vehicles_in_lanes :
  ∀ (lanes : ℕ) (trucks_per_lane cars_total trucks_total : ℕ),
  lanes = 4 →
  trucks_per_lane = 60 →
  trucks_total = trucks_per_lane * lanes →
  cars_total = 2 * trucks_total →
  (trucks_total + cars_total) = 2160 :=
by intros lanes trucks_per_lane cars_total trucks_total hlanes htrucks_per_lane htrucks_total hcars_total
   -- sorry added to skip the proof
   sorry

end total_vehicles_in_lanes_l0_14


namespace sqrt_diff_eq_neg_sixteen_l0_156

theorem sqrt_diff_eq_neg_sixteen : 
  (Real.sqrt (16 - 8 * Real.sqrt 2) - Real.sqrt (16 + 8 * Real.sqrt 2)) = -16 := 
  sorry

end sqrt_diff_eq_neg_sixteen_l0_156


namespace initial_candies_is_720_l0_576

-- Definitions according to the conditions
def candies_remaining_after_day_n (initial_candies : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 1 => initial_candies / 2
  | 2 => (initial_candies / 2) / 3
  | 3 => (initial_candies / 2) / 3 / 4
  | 4 => (initial_candies / 2) / 3 / 4 / 5
  | 5 => (initial_candies / 2) / 3 / 4 / 5 / 6
  | _ => 0 -- For days beyond the fifth, this is nonsensical

-- Proof statement
theorem initial_candies_is_720 : ∀ (initial_candies : ℕ), candies_remaining_after_day_n initial_candies 5 = 1 → initial_candies = 720 :=
by
  intros initial_candies h
  sorry

end initial_candies_is_720_l0_576


namespace student_marks_l0_18

theorem student_marks (T P F M : ℕ)
  (hT : T = 600)
  (hP : P = 33)
  (hF : F = 73)
  (hM : M = (P * T / 100) - F) : M = 125 := 
by 
  sorry

end student_marks_l0_18


namespace points_on_same_line_l0_83

theorem points_on_same_line (k : ℤ) : 
  (∃ m b : ℤ, ∀ p : ℤ × ℤ, p = (1, 4) ∨ p = (3, -2) ∨ p = (6, k / 3) → p.2 = m * p.1 + b) ↔ k = -33 :=
by
  sorry

end points_on_same_line_l0_83


namespace polynomial_roots_identity_l0_420

variables {c d : ℂ}

theorem polynomial_roots_identity (hc : c + d = 5) (hd : c * d = 6) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 :=
by {
  sorry
}

end polynomial_roots_identity_l0_420


namespace pairs_equality_l0_94

-- Define all the pairs as given in the problem.
def pairA_1 : ℤ := - (2^7)
def pairA_2 : ℤ := (-2)^7
def pairB_1 : ℤ := - (3^2)
def pairB_2 : ℤ := (-3)^2
def pairC_1 : ℤ := -3 * (2^3)
def pairC_2 : ℤ := - (3^2) * 2
def pairD_1 : ℤ := -((-3)^2)
def pairD_2 : ℤ := -((-2)^3)

-- The problem statement.
theorem pairs_equality :
  pairA_1 = pairA_2 ∧ ¬ (pairB_1 = pairB_2) ∧ ¬ (pairC_1 = pairC_2) ∧ ¬ (pairD_1 = pairD_2) := by
  sorry

end pairs_equality_l0_94


namespace greatest_prime_divisor_digits_sum_l0_603

theorem greatest_prime_divisor_digits_sum (h : 8191 = 2^13 - 1) : (1 + 2 + 7) = 10 :=
by
  sorry

end greatest_prime_divisor_digits_sum_l0_603


namespace last_bead_color_is_blue_l0_609

def bead_color_cycle := ["Red", "Orange", "Yellow", "Yellow", "Green", "Blue", "Purple"]

def bead_color (n : Nat) : String :=
  bead_color_cycle.get! (n % bead_color_cycle.length)

theorem last_bead_color_is_blue :
  bead_color 82 = "Blue" := 
by
  sorry

end last_bead_color_is_blue_l0_609


namespace arithmetic_sequence_150th_term_l0_116

open Nat

-- Define the nth term of an arithmetic sequence
def nth_term_arithmetic (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Theorem to prove
theorem arithmetic_sequence_150th_term (a1 d n : ℕ) (h1 : a1 = 3) (h2 : d = 7) (h3 : n = 150) :
  nth_term_arithmetic a1 d n = 1046 :=
by
  sorry

end arithmetic_sequence_150th_term_l0_116


namespace total_students_correct_l0_548

theorem total_students_correct (H : ℕ)
  (B : ℕ := 2 * H)
  (P : ℕ := H + 5)
  (S : ℕ := 3 * (H + 5))
  (h1 : B = 30)
  : (B + H + P + S) = 125 := by
  sorry

end total_students_correct_l0_548


namespace centroid_coordinates_of_tetrahedron_l0_804

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (O A B C G G1 : V) (OG1_subdivides : G -ᵥ O = 3 • (G1 -ᵥ G))
variable (A_centroid : G1 -ᵥ O = (1/3 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O))

-- The main proof problem
theorem centroid_coordinates_of_tetrahedron :
  G -ᵥ O = (1/4 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O) :=
sorry

end centroid_coordinates_of_tetrahedron_l0_804


namespace smallest_pretty_num_l0_645

-- Define the notion of a pretty number
def is_pretty (n : ℕ) : Prop :=
  ∃ d1 d2 : ℕ, (1 ≤ d1 ∧ d1 ≤ n) ∧ (1 ≤ d2 ∧ d2 ≤ n) ∧ d2 - d1 ∣ n ∧ (1 < d1)

-- Define the statement to prove that 160400 is the smallest pretty number greater than 401 that is a multiple of 401
theorem smallest_pretty_num (n : ℕ) (hn1 : n > 401) (hn2 : n % 401 = 0) : n = 160400 :=
  sorry

end smallest_pretty_num_l0_645


namespace functional_eq_l0_247

theorem functional_eq (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x * f y + y) = f (x * y) + f y) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) :=
sorry

end functional_eq_l0_247


namespace necessary_but_not_sufficient_l0_404

variables {R : Type*} [Field R] (a b c : R)

def condition1 : Prop := (a / b) = (b / c)
def condition2 : Prop := b^2 = a * c

theorem necessary_but_not_sufficient :
  (condition1 a b c → condition2 a b c) ∧ ¬ (condition2 a b c → condition1 a b c) :=
by
  sorry

end necessary_but_not_sufficient_l0_404


namespace tan_diff_pi_over_4_l0_81

theorem tan_diff_pi_over_4 (α : ℝ) (hα1 : π < α) (hα2 : α < 3 / 2 * π) (hcos : Real.cos α = -4 / 5) :
  Real.tan (π / 4 - α) = 1 / 7 := by
  sorry

end tan_diff_pi_over_4_l0_81


namespace range_of_x_l0_652

theorem range_of_x (S : ℕ → ℕ) (a : ℕ → ℕ) (x : ℕ) :
  (∀ n, n ≥ 2 → S (n - 1) + S n = 2 * n^2 + 1) →
  S 0 = 0 →
  a 1 = x →
  (∀ n, a n ≤ a (n + 1)) →
  2 < x ∧ x < 3 := 
sorry

end range_of_x_l0_652


namespace ratio_of_vegetables_to_beef_l0_557

variable (amountBeefInitial : ℕ) (amountBeefUnused : ℕ) (amountVegetables : ℕ)

def amount_beef_used (initial unused : ℕ) : ℕ := initial - unused
def ratio_vegetables_beef (vegetables beef : ℕ) : ℚ := vegetables / beef

theorem ratio_of_vegetables_to_beef 
  (h1 : amountBeefInitial = 4)
  (h2 : amountBeefUnused = 1)
  (h3 : amountVegetables = 6) :
  ratio_vegetables_beef amountVegetables (amount_beef_used amountBeefInitial amountBeefUnused) = 2 :=
by
  sorry

end ratio_of_vegetables_to_beef_l0_557


namespace gift_distribution_l0_267

noncomputable section

structure Recipients :=
  (ondra : String)
  (matej : String)
  (kuba : String)

structure PetrStatements :=
  (ondra_fire_truck : Bool)
  (kuba_no_fire_truck : Bool)
  (matej_no_merkur : Bool)

def exactly_one_statement_true (s : PetrStatements) : Prop :=
  (s.ondra_fire_truck && ¬s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && ¬s.kuba_no_fire_truck && s.matej_no_merkur)

def correct_recipients (r : Recipients) : Prop :=
  r.kuba = "fire truck" ∧ r.matej = "helicopter" ∧ r.ondra = "Merkur"

theorem gift_distribution
  (r : Recipients)
  (s : PetrStatements)
  (h : exactly_one_statement_true s)
  (h0 : ¬exactly_one_statement_true ⟨r.ondra = "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  (h1 : ¬exactly_one_statement_true ⟨r.ondra ≠ "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  : correct_recipients r := by
  -- Proof is omitted as per the instructions
  sorry

end gift_distribution_l0_267


namespace fraction_identity_l0_687

variable {n : ℕ}

theorem fraction_identity
  (h1 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → 1 / (n * (n + 1)) = 1 / n - 1 / (n + 1)))
  (h2 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → n ≠ 2 → 1 / (n * (n + 1) * (n + 2)) = 1 / (2 * n * (n + 1)) - 1 / (2 * (n + 1) * (n + 2))))
  : 1 / (n * (n + 1) * (n + 2) * (n + 3)) = 1 / (3 * n * (n + 1) * (n + 2)) - 1 / (3 * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end fraction_identity_l0_687


namespace y1_less_than_y2_l0_158

noncomputable def y1 : ℝ := 2 * (-5) + 1
noncomputable def y2 : ℝ := 2 * 3 + 1

theorem y1_less_than_y2 : y1 < y2 := by
  sorry

end y1_less_than_y2_l0_158


namespace exists_f_prime_eq_inverses_l0_622

theorem exists_f_prime_eq_inverses (f : ℝ → ℝ) (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : ContinuousOn f (Set.Icc a b))
  (h4 : DifferentiableOn ℝ f (Set.Ioo a b)) :
  ∃ c ∈ Set.Ioo a b, (deriv f c) = (1 / (a - c)) + (1 / (b - c)) + (1 / (a + b)) :=
by
  sorry

end exists_f_prime_eq_inverses_l0_622


namespace words_per_page_l0_68

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 90) : p = 90 :=
sorry

end words_per_page_l0_68


namespace coordinates_of_C_are_correct_l0_77

noncomputable section 

def Point := (ℝ × ℝ)

def A : Point := (1, 3)
def B : Point := (13, 9)

def vector_AB (A B : Point) : Point :=
  (B.1 - A.1, B.2 - A.2)

def scalar_mult (s : ℝ) (v : Point) : Point :=
  (s * v.1, s * v.2)

def add_vectors (v1 v2 : Point) : Point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def C : Point :=
  let AB := vector_AB A B
  add_vectors B (scalar_mult (1 / 2) AB)

theorem coordinates_of_C_are_correct : C = (19, 12) := by sorry

end coordinates_of_C_are_correct_l0_77


namespace plot_length_l0_431

-- Define the conditions
def rent_per_acre_per_month : ℝ := 30
def total_rent_per_month : ℝ := 300
def width_feet : ℝ := 1210
def area_acres : ℝ := 10
def square_feet_per_acre : ℝ := 43560

-- Prove that the length of the plot is 360 feet
theorem plot_length (h1 : rent_per_acre_per_month = 30)
                    (h2 : total_rent_per_month = 300)
                    (h3 : width_feet = 1210)
                    (h4 : area_acres = 10)
                    (h5 : square_feet_per_acre = 43560) :
  (area_acres * square_feet_per_acre) / width_feet = 360 := 
by {
  sorry
}

end plot_length_l0_431


namespace fraction_halfway_between_l0_170

theorem fraction_halfway_between (a b : ℚ) (h₁ : a = 1 / 6) (h₂ : b = 2 / 5) : (a + b) / 2 = 17 / 60 :=
by {
  sorry
}

end fraction_halfway_between_l0_170


namespace figure_can_be_cut_and_reassembled_into_square_l0_437

-- Define the conditions
def is_square_area (n: ℕ) : Prop := ∃ k: ℕ, k * k = n

def can_form_square (area: ℕ) : Prop :=
area = 18 ∧ ¬ is_square_area area

-- The proof statement
theorem figure_can_be_cut_and_reassembled_into_square (area: ℕ) (hf: area = 18): 
  can_form_square area → ∃ (part1 part2 part3: Set (ℕ × ℕ)), true :=
by
  sorry

end figure_can_be_cut_and_reassembled_into_square_l0_437


namespace cost_price_per_meter_l0_216

-- Definitions for conditions
def total_length : ℝ := 9.25
def total_cost : ℝ := 416.25

-- The theorem to be proved
theorem cost_price_per_meter : total_cost / total_length = 45 := by
  sorry

end cost_price_per_meter_l0_216


namespace perimeter_square_III_l0_527

theorem perimeter_square_III (perimeter_I perimeter_II : ℕ) (hI : perimeter_I = 12) (hII : perimeter_II = 24) : 
  let side_I := perimeter_I / 4 
  let side_II := perimeter_II / 4 
  let side_III := side_I + side_II 
  4 * side_III = 36 :=
by
  sorry

end perimeter_square_III_l0_527


namespace sqrt_expr_eq_two_l0_969

noncomputable def expr := Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2)

theorem sqrt_expr_eq_two : expr = 2 := 
by
  sorry

end sqrt_expr_eq_two_l0_969


namespace intersection_or_parallel_lines_l0_497

structure Triangle (Point : Type) :=
  (A B C : Point)

structure Plane (Point : Type) :=
  (P1 P2 P3 P4 : Point)

variables {Point : Type}
variables (triABC triA1B1C1 : Triangle Point)
variables (plane1 plane2 plane3 : Plane Point)

-- Intersection conditions
variable (AB_intersects_A1B1 : (triABC.A, triABC.B) = (triA1B1C1.A, triA1B1C1.B))
variable (BC_intersects_B1C1 : (triABC.B, triABC.C) = (triA1B1C1.B, triA1B1C1.C))
variable (CA_intersects_C1A1 : (triABC.C, triABC.A) = (triA1B1C1.C, triA1B1C1.A))

theorem intersection_or_parallel_lines :
  ∃ P : Point, (
    (∃ A1 : Point, (triABC.A, A1) = (P, P)) ∧
    (∃ B1 : Point, (triABC.B, B1) = (P, P)) ∧
    (∃ C1 : Point, (triABC.C, C1) = (P, P))
  ) ∨ (
    (∃ d1 d2 d3 : Point, 
      (∀ A1 B1 C1 : Point,
        (triABC.A, A1) = (d1, d1) ∧ 
        (triABC.B, B1) = (d2, d2) ∧ 
        (triABC.C, C1) = (d3, d3)
      )
    )
  ) := by
  sorry

end intersection_or_parallel_lines_l0_497


namespace initial_sum_of_money_l0_847

theorem initial_sum_of_money (A2 A7 : ℝ) (H1 : A2 = 520) (H2 : A7 = 820) :
  ∃ P : ℝ, P = 400 :=
by
  -- Proof starts here
  sorry

end initial_sum_of_money_l0_847


namespace find_sheets_used_l0_565

variable (x y : ℕ) -- define variables for x and y
variable (h₁ : 82 - x = y) -- 82 - x = number of sheets left
variable (h₂ : y = x - 6) -- number of sheets left = number of sheets used - 6

theorem find_sheets_used (h₁ : 82 - x = x - 6) : x = 44 := 
by
  sorry

end find_sheets_used_l0_565


namespace shekar_biology_marks_l0_815

variable (M S SS E A : ℕ)

theorem shekar_biology_marks (hM : M = 76) (hS : S = 65) (hSS : SS = 82) (hE : E = 67) (hA : A = 77) :
  let total_marks := M + S + SS + E
  let total_average_marks := A * 5
  let biology_marks := total_average_marks - total_marks
  biology_marks = 95 :=
by
  sorry

end shekar_biology_marks_l0_815


namespace smaller_square_area_l0_384

theorem smaller_square_area (A_L : ℝ) (h : A_L = 100) : ∃ A_S : ℝ, A_S = 50 := 
by
  sorry

end smaller_square_area_l0_384


namespace altitude_change_correct_l0_63

noncomputable def altitude_change (T_ground T_high : ℝ) (deltaT_per_km : ℝ) : ℝ :=
  (T_high - T_ground) / deltaT_per_km

theorem altitude_change_correct :
  altitude_change 18 (-48) (-6) = 11 :=
by 
  sorry

end altitude_change_correct_l0_63


namespace olivine_more_stones_l0_745

theorem olivine_more_stones (x O D : ℕ) (h1 : O = 30 + x) (h2 : D = O + 11)
  (h3 : 30 + O + D = 111) : x = 5 :=
by
  sorry

end olivine_more_stones_l0_745


namespace length_AB_indeterminate_l0_785

theorem length_AB_indeterminate
  (A B C : Type)
  (AC : ℝ) (BC : ℝ)
  (AC_eq_1 : AC = 1)
  (BC_eq_3 : BC = 3) :
  (2 < AB ∧ AB < 4) ∨ (AB = 2 ∨ AB = 4) → false :=
by sorry

end length_AB_indeterminate_l0_785


namespace second_person_avg_pages_per_day_l0_911

def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def average_book_pages : ℕ := 320
def closest_person_percentage : ℝ := 0.75

theorem second_person_avg_pages_per_day :
  (deshaun_books * average_book_pages * closest_person_percentage) / summer_days = 180 := by
sorry

end second_person_avg_pages_per_day_l0_911


namespace single_intersection_not_necessarily_tangent_l0_248

structure Hyperbola where
  -- Placeholder for hyperbola properties
  axis1 : Real
  axis2 : Real

def is_tangent (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for tangency
  ∃ p : Real × Real, l = { p }

def is_parallel_to_asymptote (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for parallelism to asymptote 
  ∃ A : Real, l = { (x, A * x) | x : Real }

theorem single_intersection_not_necessarily_tangent
  (l : Set (Real × Real)) (H : Hyperbola) (h : ∃ p : Real × Real, l = { p }) :
  ¬ is_tangent l H ∨ is_parallel_to_asymptote l H :=
sorry

end single_intersection_not_necessarily_tangent_l0_248


namespace cos_values_l0_517

theorem cos_values (n : ℤ) : (0 ≤ n ∧ n ≤ 360) ∧ (Real.cos (n * Real.pi / 180) = Real.cos (310 * Real.pi / 180)) ↔ (n = 50 ∨ n = 310) :=
by
  sorry

end cos_values_l0_517


namespace exists_nat_lt_100_two_different_squares_l0_816

theorem exists_nat_lt_100_two_different_squares :
  ∃ n : ℕ, n < 100 ∧ 
    ∃ a b c d : ℕ, a^2 + b^2 = n ∧ c^2 + d^2 = n ∧ (a ≠ c ∨ b ≠ d) ∧ a ≠ b ∧ c ≠ d :=
by
  sorry

end exists_nat_lt_100_two_different_squares_l0_816


namespace cookie_combinations_l0_319

theorem cookie_combinations (total_cookies kinds : Nat) (at_least_one : kinds > 0 ∧ ∀ k : Nat, k < kinds → k > 0) : 
  (total_cookies = 8 ∧ kinds = 4) → 
  (∃ comb : Nat, comb = 34) := 
by 
  -- insert proof here 
  sorry

end cookie_combinations_l0_319


namespace problem1_problem2_l0_464

theorem problem1 : ( (2 / 3 - 1 / 4 - 5 / 6) * 12 = -5 ) :=
by sorry

theorem problem2 : ( (-3)^2 * 2 + 4 * (-3) - 28 / (7 / 4) = -10 ) :=
by sorry

end problem1_problem2_l0_464


namespace volume_of_inscribed_sphere_l0_526

noncomputable def volume_of_tetrahedron (R : ℝ) (S1 S2 S3 S4 : ℝ) : ℝ :=
  R * (S1 + S2 + S3 + S4)

theorem volume_of_inscribed_sphere (R S1 S2 S3 S4 V : ℝ) :
  V = R * (S1 + S2 + S3 + S4) :=
sorry

end volume_of_inscribed_sphere_l0_526


namespace image_preimage_f_l0_25

-- Defining the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Given conditions
def A : Set (ℝ × ℝ) := {p | True}
def B : Set (ℝ × ℝ) := {p | True}

-- Proof statement
theorem image_preimage_f :
  f (1, 3) = (4, -2) ∧ ∃ x y : ℝ, f (x, y) = (1, 3) ∧ (x, y) = (2, -1) :=
by
  sorry

end image_preimage_f_l0_25


namespace calorie_allowance_correct_l0_647

-- Definitions based on the problem's conditions
def daily_calorie_allowance : ℕ := 2000
def weekly_calorie_allowance : ℕ := 10500
def days_in_week : ℕ := 7

-- The statement to be proven
theorem calorie_allowance_correct :
  daily_calorie_allowance * days_in_week = weekly_calorie_allowance :=
by
  sorry

end calorie_allowance_correct_l0_647


namespace min_value_of_expression_l0_488

theorem min_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + a * b + a * c + b * c = 4) : 2 * a + b + c ≥ 4 :=
sorry

end min_value_of_expression_l0_488


namespace total_cakes_served_l0_994

def Cakes_Monday_Lunch : ℕ := 5
def Cakes_Monday_Dinner : ℕ := 6
def Cakes_Sunday : ℕ := 3
def cakes_served_twice (n : ℕ) : ℕ := 2 * n
def cakes_thrown_away : ℕ := 4

theorem total_cakes_served : 
  Cakes_Sunday + Cakes_Monday_Lunch + Cakes_Monday_Dinner + 
  (cakes_served_twice (Cakes_Monday_Lunch + Cakes_Monday_Dinner) - cakes_thrown_away) = 32 := 
by 
  sorry

end total_cakes_served_l0_994


namespace Kiran_money_l0_180

theorem Kiran_money (R G K : ℕ) (h1 : R / G = 6 / 7) (h2 : G / K = 6 / 15) (h3 : R = 36) : K = 105 := by
  sorry

end Kiran_money_l0_180


namespace factor_expression_l0_474

theorem factor_expression (x : ℝ) : 2 * x * (x + 3) + (x + 3) = (2 * x + 1) * (x + 3) :=
by
  sorry

end factor_expression_l0_474


namespace smallest_f1_value_l0_206

noncomputable def polynomial := 
  fun (f : ℝ → ℝ) (r s : ℝ) => 
    f = λ x => (x - r) * (x - s) * (x - ((r + s)/2))

def distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ s ∧ polynomial f r s ∧ 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (f ∘ f) a = 0 ∧ (f ∘ f) b = 0 ∧ (f ∘ f) c = 0)

theorem smallest_f1_value
  (f : ℝ → ℝ)
  (hf : distinct_real_roots f) :
  ∃ r s : ℝ, r ≠ s ∧ f 1 = 3/8 :=
sorry

end smallest_f1_value_l0_206


namespace servings_in_container_l0_698

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

end servings_in_container_l0_698


namespace factorize_polynomial_l0_425

variable (x : ℝ)

theorem factorize_polynomial : 4 * x^3 - 8 * x^2 + 4 * x = 4 * x * (x - 1)^2 := 
by 
  sorry

end factorize_polynomial_l0_425


namespace rita_canoe_trip_distance_l0_130

theorem rita_canoe_trip_distance 
  (D : ℝ)
  (h_upstream : ∃ t1, t1 = D / 3)
  (h_downstream : ∃ t2, t2 = D / 9)
  (h_total_time : ∃ t1 t2, t1 + t2 = 8) :
  D = 18 :=
by
  sorry

end rita_canoe_trip_distance_l0_130


namespace no_multiple_of_2310_in_2_j_minus_2_i_l0_306

theorem no_multiple_of_2310_in_2_j_minus_2_i (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 50) :
  ¬ ∃ k : ℕ, 2^j - 2^i = 2310 * k :=
by 
  sorry

end no_multiple_of_2310_in_2_j_minus_2_i_l0_306


namespace find_w_l0_919

theorem find_w (u v w : ℝ) (h1 : 10 * u + 8 * v + 5 * w = 160)
  (h2 : v = u + 3) (h3 : w = 2 * v) : w = 13.5714 := by
  -- The proof would go here, but we leave it empty as per instructions.
  sorry

end find_w_l0_919


namespace sticks_form_triangle_l0_113

theorem sticks_form_triangle (a b c d e : ℝ) 
  (h1 : 2 < a) (h2 : a < 8)
  (h3 : 2 < b) (h4 : b < 8)
  (h5 : 2 < c) (h6 : c < 8)
  (h7 : 2 < d) (h8 : d < 8)
  (h9 : 2 < e) (h10 : e < 8) :
  ∃ x y z, 
    (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
    (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
    (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
    x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    x + y > z ∧ x + z > y ∧ y + z > x :=
by sorry

end sticks_form_triangle_l0_113


namespace find_interest_rate_l0_869

theorem find_interest_rate (initial_investment : ℚ) (duration_months : ℚ) 
  (first_rate : ℚ) (final_value : ℚ) (s : ℚ) :
  initial_investment = 15000 →
  duration_months = 9 →
  first_rate = 0.09 →
  final_value = 17218.50 →
  (∃ s : ℚ, 16012.50 * (1 + (s * 0.75) / 100) = final_value) →
  s = 10 := 
by
  sorry

end find_interest_rate_l0_869


namespace quadratic_general_form_l0_539

theorem quadratic_general_form (x : ℝ) :
  x * (x + 2) = 5 * (x - 2) → x^2 - 3 * x - 10 = 0 := by
  sorry

end quadratic_general_form_l0_539


namespace playground_width_l0_146

open Nat

theorem playground_width (garden_width playground_length perimeter_garden : ℕ) (garden_area_eq_playground_area : Bool) :
  garden_width = 8 →
  playground_length = 16 →
  perimeter_garden = 64 →
  garden_area_eq_playground_area →
  ∃ (W : ℕ), W = 12 :=
by
  intros h_t1 h_t2 h_t3 h_t4
  sorry

end playground_width_l0_146


namespace quarters_to_dollars_l0_215

theorem quarters_to_dollars (total_quarters : ℕ) (quarters_per_dollar : ℕ) (h1 : total_quarters = 8) (h2 : quarters_per_dollar = 4) : total_quarters / quarters_per_dollar = 2 :=
by {
  sorry
}

end quarters_to_dollars_l0_215


namespace value_of_x_l0_44

theorem value_of_x (g : ℝ → ℝ) (h : ∀ x, g (5 * x + 2) = 3 * x - 4) : g (-13) = -13 :=
by {
  sorry
}

end value_of_x_l0_44


namespace speeds_of_bus_and_car_l0_335

theorem speeds_of_bus_and_car
  (d t : ℝ) (v1 v2 : ℝ)
  (h1 : 1.5 * v1 + 1.5 * v2 = d)
  (h2 : 2.5 * v1 + 1 * v2 = d) :
  v1 = 40 ∧ v2 = 80 :=
by sorry

end speeds_of_bus_and_car_l0_335


namespace shaded_region_area_l0_222

-- Define the problem conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def area_of_shaded_region : ℝ := 50

-- State the theorem to prove the area of the shaded region
theorem shaded_region_area (n : ℕ) (d : ℝ) (area : ℝ) (h1 : n = num_squares) (h2 : d = diagonal_length) : 
  area = area_of_shaded_region :=
sorry

end shaded_region_area_l0_222


namespace symmetric_axis_of_parabola_l0_51

theorem symmetric_axis_of_parabola :
  (∃ x : ℝ, x = 6 ∧ (∀ y : ℝ, y = 1/2 * x^2 - 6 * x + 21)) :=
sorry

end symmetric_axis_of_parabola_l0_51


namespace max_full_box_cards_l0_327

-- Given conditions
def total_cards : ℕ := 94
def unfilled_box_cards : ℕ := 6

-- Define the number of cards that are evenly distributed into full boxes
def evenly_distributed_cards : ℕ := total_cards - unfilled_box_cards

-- Prove that the maximum number of cards a full box can hold is 22
theorem max_full_box_cards (h : evenly_distributed_cards = 88) : ∃ x : ℕ, evenly_distributed_cards % x = 0 ∧ x = 22 :=
by 
  -- Proof goes here
  sorry

end max_full_box_cards_l0_327


namespace hyperbola_constant_ellipse_constant_l0_519

variables {a b : ℝ} (a_pos_b_gt_a : 0 < a ∧ a < b)
variables {A B : ℝ × ℝ} (on_hyperbola_A : A.1^2 / a^2 - A.2^2 / b^2 = 1)
variables (on_hyperbola_B : B.1^2 / a^2 - B.2^2 / b^2 = 1) (perp_OA_OB : A.1 * B.1 + A.2 * B.2 = 0)

-- Hyperbola statement
theorem hyperbola_constant :
  (1 / (A.1^2 + A.2^2)) + (1 / (B.1^2 + B.2^2)) = 1 / a^2 - 1 / b^2 :=
sorry

variables {C D : ℝ × ℝ} (on_ellipse_C : C.1^2 / a^2 + C.2^2 / b^2 = 1)
variables (on_ellipse_D : D.1^2 / a^2 + D.2^2 / b^2 = 1) (perp_OC_OD : C.1 * D.1 + C.2 * D.2 = 0)

-- Ellipse statement
theorem ellipse_constant :
  (1 / (C.1^2 + C.2^2)) + (1 / (D.1^2 + D.2^2)) = 1 / a^2 + 1 / b^2 :=
sorry

end hyperbola_constant_ellipse_constant_l0_519


namespace unique_digits_addition_l0_241

theorem unique_digits_addition :
  ∃ (X Y B M C : ℕ), 
    -- Conditions
    X ≠ 0 ∧ Y ≠ 0 ∧ B ≠ 0 ∧ M ≠ 0 ∧ C ≠ 0 ∧
    X ≠ Y ∧ X ≠ B ∧ X ≠ M ∧ X ≠ C ∧ Y ≠ B ∧ Y ≠ M ∧ Y ≠ C ∧ B ≠ M ∧ B ≠ C ∧ M ≠ C ∧
    -- Addition equation with distinct digits
    (X * 1000 + Y * 100 + 70) + (B * 100 + M * 10 + C) = (B * 1000 + M * 100 + C * 10 + 0) ∧
    -- Correct Answer
    X = 9 ∧ Y = 8 ∧ B = 3 ∧ M = 8 ∧ C = 7 :=
sorry

end unique_digits_addition_l0_241


namespace largest_of_five_consecutive_ints_15120_l0_925

theorem largest_of_five_consecutive_ints_15120 :
  ∃ (a b c d e : ℕ), 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a * b * c * d * e = 15120 ∧ 
  e = 10 := 
sorry

end largest_of_five_consecutive_ints_15120_l0_925


namespace center_of_circle_l0_649

theorem center_of_circle :
  ∀ (x y : ℝ), (x^2 - 8 * x + y^2 - 4 * y = 16) → (x, y) = (4, 2) :=
by
  sorry

end center_of_circle_l0_649


namespace sum_of_sequences_l0_321

noncomputable def arithmetic_sequence (a b : ℤ) : Prop :=
  ∃ k : ℤ, a = 6 + k ∧ b = 6 + 2 * k

noncomputable def geometric_sequence (c d : ℤ) : Prop :=
  ∃ q : ℤ, c = 6 * q ∧ d = 6 * q^2

theorem sum_of_sequences (a b c d : ℤ) 
  (h_arith : arithmetic_sequence a b) 
  (h_geom : geometric_sequence c d) 
  (hb : b = 48) (hd : 6 * c^2 = 48): 
  a + b + c + d = 111 := 
sorry

end sum_of_sequences_l0_321


namespace larger_number_ratio_l0_778

theorem larger_number_ratio (x : ℕ) (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 8 * x) 
(h3 : (a - 24) * 9 = (b - 24) * 4) : b = 192 :=
sorry

end larger_number_ratio_l0_778


namespace people_lost_l0_16

-- Define the given conditions
def ratio_won_to_lost : ℕ × ℕ := (4, 1)
def people_won : ℕ := 28

-- Define the proof problem
theorem people_lost (L : ℕ) (h_ratio : ratio_won_to_lost = (4, 1)) (h_won : people_won = 28) : L = 7 :=
by
  -- Skip the proof
  sorry

end people_lost_l0_16


namespace value_of_x_l0_637

theorem value_of_x (x : ℤ) (h : 3 * x = (26 - x) + 26) : x = 13 :=
by
  sorry

end value_of_x_l0_637


namespace n_squared_divisible_by_12_l0_230

theorem n_squared_divisible_by_12 (n : ℕ) : 12 ∣ n^2 * (n^2 - 1) :=
  sorry

end n_squared_divisible_by_12_l0_230


namespace parabola_intersects_xaxis_at_least_one_l0_580

theorem parabola_intersects_xaxis_at_least_one {a b c : ℝ} (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x1 x2, x1 ≠ x2 ∧ (a * x1^2 + 2 * b * x1 + c = 0) ∧ (a * x2^2 + 2 * b * x2 + c = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (b * x1^2 + 2 * c * x1 + a = 0) ∧ (b * x2^2 + 2 * c * x2 + a = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (c * x1^2 + 2 * a * x1 + b = 0) ∧ (c * x2^2 + 2 * a * x2 + b = 0)) :=
by
  sorry

end parabola_intersects_xaxis_at_least_one_l0_580


namespace fraction_equivalence_l0_58

-- Given fractions
def frac1 : ℚ := 3 / 7
def frac2 : ℚ := 4 / 5
def frac3 : ℚ := 5 / 12
def frac4 : ℚ := 2 / 9

-- Expectation
def result : ℚ := 1548 / 805

-- Theorem to prove the equality
theorem fraction_equivalence : ((frac1 + frac2) / (frac3 + frac4)) = result := by
  sorry

end fraction_equivalence_l0_58


namespace geometric_sequence_problem_l0_458

noncomputable def q : ℝ := 1 + Real.sqrt 2

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = (q : ℝ) * a n)
  (h_cond : a 2 = a 0 + 2 * a 1) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := 
sorry

end geometric_sequence_problem_l0_458


namespace tangent_line_to_parabola_l0_982

theorem tangent_line_to_parabola (r : ℝ) :
  (∃ x : ℝ, 2 * x^2 - x - r = 0) ∧
  (∀ x1 x2 : ℝ, (2 * x1^2 - x1 - r = 0) ∧ (2 * x2^2 - x2 - r = 0) → x1 = x2) →
  r = -1 / 8 :=
sorry

end tangent_line_to_parabola_l0_982


namespace steve_pie_difference_l0_112

-- Definitions of conditions
def apple_pie_days : Nat := 3
def cherry_pie_days : Nat := 2
def pies_per_day : Nat := 12

-- Theorem statement
theorem steve_pie_difference : 
  (apple_pie_days * pies_per_day) - (cherry_pie_days * pies_per_day) = 12 := 
by
  sorry

end steve_pie_difference_l0_112


namespace Adam_total_cost_l0_867

theorem Adam_total_cost :
  let laptop1_cost := 500
  let laptop2_base_cost := 3 * laptop1_cost
  let discount := 0.15 * laptop2_base_cost
  let laptop2_cost := laptop2_base_cost - discount
  let external_hard_drive := 80
  let mouse := 20
  let software1 := 120
  let software2 := 2 * 120
  let insurance1 := 0.10 * laptop1_cost
  let insurance2 := 0.10 * laptop2_cost
  let total_cost1 := laptop1_cost + external_hard_drive + mouse + software1 + insurance1
  let total_cost2 := laptop2_cost + external_hard_drive + mouse + software2 + insurance2
  total_cost1 + total_cost2 = 2512.5 :=
by
  sorry

end Adam_total_cost_l0_867


namespace simplify_expression_l0_547

variable (x y z : ℝ)

noncomputable def expr1 := (3 * x + y / 3 + 2 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (2 * z)⁻¹)
noncomputable def expr2 := (2 * y + 18 * x * z + 3 * z * x) / (6 * x * y * z * (9 * x + y + 6 * z))

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxyz : 3 * x + y / 3 + 2 * z ≠ 0) :
  expr1 x y z = expr2 x y z := by 
  sorry

end simplify_expression_l0_547


namespace bottles_purchased_l0_226

/-- Given P bottles can be bought for R dollars, determine how many bottles can be bought for M euros
    if 1 euro is worth 1.2 dollars and there is a 10% discount when buying with euros. -/
theorem bottles_purchased (P R M : ℝ) (hR : R > 0) (hP : P > 0) :
  let euro_to_dollars := 1.2
  let discount := 0.9
  let dollars := euro_to_dollars * M * discount
  (P / R) * dollars = (1.32 * P * M) / R :=
by
  sorry

end bottles_purchased_l0_226


namespace loss_due_to_simple_interest_l0_899

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r)^t

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem loss_due_to_simple_interest (P : ℝ) (r : ℝ) (t : ℝ)
  (hP : P = 2500) (hr : r = 0.04) (ht : t = 2) :
  let CI := compound_interest P r t
  let SI := simple_interest P r t
  ∃ loss : ℝ, loss = CI - SI ∧ loss = 4 :=
by
  sorry

end loss_due_to_simple_interest_l0_899


namespace log_three_nine_cubed_l0_624

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l0_624


namespace mayor_cup_num_teams_l0_618

theorem mayor_cup_num_teams (x : ℕ) (h : x * (x - 1) / 2 = 21) : 
    ∃ x, x * (x - 1) / 2 = 21 := 
by
  sorry

end mayor_cup_num_teams_l0_618


namespace martha_jar_spices_cost_l0_300

def price_per_jar_spices (p_beef p_fv p_oj : ℕ) (price_spices : ℕ) :=
  let total_spent := (3 * p_beef) + (8 * p_fv) + p_oj + (3 * price_spices)
  let total_points := (total_spent / 10) * 50 + if total_spent > 100 then 250 else 0
  total_points

theorem martha_jar_spices_cost (price_spices : ℕ) :
  price_per_jar_spices 11 4 37 price_spices = 850 → price_spices = 6 := by
  sorry

end martha_jar_spices_cost_l0_300


namespace part1_solution_part2_solution_l0_771

-- Definition for part 1
noncomputable def f_part1 (x : ℝ) := abs (x - 3) + 2 * x

-- Proof statement for part 1
theorem part1_solution (x : ℝ) : (f_part1 x ≥ 3) ↔ (x ≥ 0) :=
by sorry

-- Definition for part 2
noncomputable def f_part2 (x a : ℝ) := abs (x - a) + 2 * x

-- Proof statement for part 2
theorem part2_solution (a : ℝ) : 
  (∀ x, f_part2 x a ≤ 0 ↔ x ≤ -2) → (a = 2 ∨ a = -6) :=
by sorry

end part1_solution_part2_solution_l0_771


namespace value_of_expression_l0_691

theorem value_of_expression (x y : ℤ) (h1 : x = -6) (h2 : y = -3) : 4 * (x - y) ^ 2 - x * y = 18 :=
by sorry

end value_of_expression_l0_691


namespace triangle_inequality_l0_315

-- Define the lengths of the existing sticks
def a := 4
def b := 7

-- Define the list of potential third sticks
def potential_sticks := [3, 6, 11, 12]

-- Define the triangle inequality conditions
def valid_length (c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Prove that the valid length satisfying these conditions is 6
theorem triangle_inequality : ∃ c ∈ potential_sticks, valid_length c ∧ c = 6 :=
by
  sorry

end triangle_inequality_l0_315


namespace tire_cost_l0_629

theorem tire_cost (total_cost : ℕ) (number_of_tires : ℕ) (cost_per_tire : ℕ) 
    (h1 : total_cost = 240) 
    (h2 : number_of_tires = 4)
    (h3 : cost_per_tire = total_cost / number_of_tires) : 
    cost_per_tire = 60 :=
sorry

end tire_cost_l0_629


namespace last_date_in_2011_divisible_by_101_is_1221_l0_358

def is_valid_date (a b c d : ℕ) : Prop :=
  (10 * a + b) ≤ 12 ∧ (10 * c + d) ≤ 31

def date_as_number (a b c d : ℕ) : ℕ :=
  20110000 + 1000 * a + 100 * b + 10 * c + d

theorem last_date_in_2011_divisible_by_101_is_1221 :
  ∃ (a b c d : ℕ), is_valid_date a b c d ∧ date_as_number a b c d % 101 = 0 ∧ date_as_number a b c d = 20111221 :=
by
  sorry

end last_date_in_2011_divisible_by_101_is_1221_l0_358


namespace last_digit_2008_pow_2008_l0_699

theorem last_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := by
  -- Here, the proof would follow the understanding of the cyclic pattern of the last digits of powers of 2008
  sorry

end last_digit_2008_pow_2008_l0_699


namespace mike_initial_marbles_l0_260

theorem mike_initial_marbles (n : ℕ) 
  (gave_to_sam : ℕ) (left_with_mike : ℕ)
  (h1 : gave_to_sam = 4)
  (h2 : left_with_mike = 4)
  (h3 : n = gave_to_sam + left_with_mike) : n = 8 := 
by
  sorry

end mike_initial_marbles_l0_260


namespace hall_length_l0_136

theorem hall_length
  (width : ℝ)
  (stone_length : ℝ)
  (stone_width : ℝ)
  (num_stones : ℕ)
  (h₁ : width = 15)
  (h₂ : stone_length = 0.8)
  (h₃ : stone_width = 0.5)
  (h₄ : num_stones = 1350) :
  ∃ length : ℝ, length = 36 :=
by
  sorry

end hall_length_l0_136


namespace base_nine_to_mod_five_l0_239

-- Define the base-nine number N
def N : ℕ := 2 * 9^10 + 7 * 9^9 + 0 * 9^8 + 0 * 9^7 + 6 * 9^6 + 0 * 9^5 + 0 * 9^4 + 0 * 9^3 + 0 * 9^2 + 5 * 9^1 + 2 * 9^0

-- Theorem statement
theorem base_nine_to_mod_five : N % 5 = 3 :=
by
  sorry

end base_nine_to_mod_five_l0_239


namespace cone_sphere_ratio_l0_733

/-- A right circular cone and a sphere have bases with the same radius r. 
If the volume of the cone is one-third that of the sphere, find the ratio of 
the altitude of the cone to the radius of its base. -/
theorem cone_sphere_ratio (r h : ℝ) (h_pos : 0 < r) 
    (volume_cone : ℝ) (volume_sphere : ℝ)
    (cone_volume_formula : volume_cone = (1 / 3) * π * r^2 * h) 
    (sphere_volume_formula : volume_sphere = (4 / 3) * π * r^3) 
    (volume_relation : volume_cone = (1 / 3) * volume_sphere) : 
    h / r = 4 / 3 :=
by
    sorry

end cone_sphere_ratio_l0_733


namespace part1_part2_l0_860

noncomputable def A (a : ℝ) := { x : ℝ | x^2 - a * x + a^2 - 19 = 0 }
def B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
def C := { x : ℝ | x^2 + 2 * x - 8 = 0 }

-- Proof Problem 1: Prove that if A ∩ B ≠ ∅ and A ∩ C = ∅, then a = -2
theorem part1 (a : ℝ) (h1 : (A a ∩ B) ≠ ∅) (h2 : (A a ∩ C) = ∅) : a = -2 :=
sorry

-- Proof Problem 2: Prove that if A ∩ B = A ∩ C ≠ ∅, then a = -3
theorem part2 (a : ℝ) (h1 : (A a ∩ B = A a ∩ C) ∧ (A a ∩ B) ≠ ∅) : a = -3 :=
sorry

end part1_part2_l0_860


namespace min_mn_value_l0_879

theorem min_mn_value
  (a : ℝ) (m : ℝ) (n : ℝ)
  (ha_pos : a > 0) (ha_ne_one : a ≠ 1) (hm_pos : m > 0) (hn_pos : n > 0)
  (H : (1 : ℝ) / m + (1 : ℝ) / n = 4) :
  m + n ≥ 1 :=
sorry

end min_mn_value_l0_879


namespace arithmetic_mean_is_correct_l0_907

open Nat

noncomputable def arithmetic_mean_of_two_digit_multiples_of_9 : ℝ :=
  let smallest := 18
  let largest := 99
  let n := 10
  let sum := (n / 2 : ℝ) * (smallest + largest)
  sum / n

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_two_digit_multiples_of_9 = 58.5 :=
by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l0_907


namespace translate_triangle_vertex_l0_289

theorem translate_triangle_vertex 
    (a b : ℤ) 
    (hA : (-3, a) = (-1, 2) + (-2, a - 2)) 
    (hB : (b, 3) = (1, -1) + (b - 1, 4)) :
    (2 + (-3 - (-1)), 1 + (3 - (-1))) = (0, 5) :=
by 
  -- proof is omitted as instructed
  sorry

end translate_triangle_vertex_l0_289


namespace all_radii_equal_l0_681
-- Lean 4 statement

theorem all_radii_equal (r : ℝ) (h : r = 2) : r = 2 :=
by
  sorry

end all_radii_equal_l0_681


namespace largest_term_quotient_l0_697

theorem largest_term_quotient (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_S_def : ∀ n, S n = (n * (a 0 + a n)) / 2)
  (h_S15_pos : S 15 > 0)
  (h_S16_neg : S 16 < 0) :
  ∃ m, 1 ≤ m ∧ m ≤ 15 ∧
       ∀ k, (1 ≤ k ∧ k ≤ 15) → (S m / a m) ≥ (S k / a k) ∧ m = 8 := 
sorry

end largest_term_quotient_l0_697


namespace quadratic_equation_even_coefficient_l0_693

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

end quadratic_equation_even_coefficient_l0_693


namespace arithmetic_progression_product_difference_le_one_l0_886

theorem arithmetic_progression_product_difference_le_one 
  (a b : ℝ) :
  ∃ (m n k l : ℤ), |(a + b * m) * (a + b * n) - (a + b * k) * (a + b * l)| ≤ 1 :=
sorry

end arithmetic_progression_product_difference_le_one_l0_886


namespace tan_sum_eq_one_l0_577

theorem tan_sum_eq_one (a b : ℝ) (h1 : Real.tan a = 1 / 2) (h2 : Real.tan b = 1 / 3) :
    Real.tan (a + b) = 1 := 
by
  sorry

end tan_sum_eq_one_l0_577


namespace Inequality_Solution_Set_Range_of_c_l0_868

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x

noncomputable def g (x : ℝ) : ℝ := -(((-x)^2) + 2 * (-x))

theorem Inequality_Solution_Set (x : ℝ) :
  (g x ≥ f x - |x - 1|) ↔ (-1 ≤ x ∧ x ≤ 1/2) :=
by
  sorry

theorem Range_of_c (c : ℝ) :
  (∀ x : ℝ, g x + c ≤ f x - |x - 1|) ↔ (c ≤ -9/8) :=
by
  sorry

end Inequality_Solution_Set_Range_of_c_l0_868


namespace quadratic_real_roots_l0_86

theorem quadratic_real_roots (k : ℝ) : 
  (∀ x : ℝ, (2 * x^2 + 4 * x + k - 1 = 0) → ∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) → 
  k ≤ 3 :=
by
  intro h
  have h_discriminant : 16 - 8 * k >= 0 := sorry
  linarith

end quadratic_real_roots_l0_86


namespace complement_intersection_l0_196

open Set

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 2, 5}

theorem complement_intersection :
  ((U \ A) ∩ B) = {1, 5} :=
by
  sorry

end complement_intersection_l0_196


namespace all_d_zero_l0_564

def d (n m : ℕ) : ℤ := sorry -- or some explicit initial definition

theorem all_d_zero (n m : ℕ) (h₁ : n ≥ 0) (h₂ : 0 ≤ m) (h₃ : m ≤ n) :
  (m = 0 ∨ m = n → d n m = 0) ∧
  (0 < m ∧ m < n → m * d n m = m * d (n - 1) m + (2 * n - m) * d (n - 1) (m - 1))
:=
  sorry

end all_d_zero_l0_564


namespace numberOfValidFiveDigitNumbers_l0_142

namespace MathProof

def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

def firstAndLastDigitsEqual (n : ℕ) : Prop := 
  let firstDigit := (n / 10000) % 10
  let lastDigit := n % 10
  firstDigit = lastDigit

def sumOfDigitsDivisibleBy5 (n : ℕ) : Prop := 
  let d1 := (n / 10000) % 10
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 + d2 + d3 + d4 + d5) % 5 = 0

theorem numberOfValidFiveDigitNumbers :
  ∃ (count : ℕ), count = 200 ∧ 
  count = Nat.card {n : ℕ // isFiveDigitNumber n ∧ 
                                isDivisibleBy5 n ∧ 
                                firstAndLastDigitsEqual n ∧ 
                                sumOfDigitsDivisibleBy5 n} :=
by
  sorry

end MathProof

end numberOfValidFiveDigitNumbers_l0_142


namespace geometric_sum_common_ratio_l0_443

theorem geometric_sum_common_ratio (a₁ a₂ : ℕ) (q : ℕ) (S₃ : ℕ)
  (h1 : S₃ = a₁ + 3 * a₂)
  (h2: S₃ = a₁ * (1 + q + q^2)) :
  q = 2 :=
by
  sorry

end geometric_sum_common_ratio_l0_443


namespace Mrs_Amaro_roses_l0_610

theorem Mrs_Amaro_roses :
  ∀ (total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses : ℕ),
    total_roses = 500 →
    5 * total_roses % 8 = 0 →
    red_roses = total_roses * 5 / 8 →
    yellow_roses = (total_roses - red_roses) * 1 / 8 →
    pink_roses = (total_roses - red_roses) * 2 / 8 →
    remaining_roses = total_roses - red_roses - yellow_roses - pink_roses →
    remaining_roses % 2 = 0 →
    white_roses = remaining_roses / 2 →
    purple_roses = remaining_roses / 2 →
    red_roses + white_roses + purple_roses = 430 :=
by
  intros total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses
  intro total_roses_eq
  intro red_roses_divisible
  intro red_roses_def
  intro yellow_roses_def
  intro pink_roses_def
  intro remaining_roses_def
  intro remaining_roses_even
  intro white_roses_def
  intro purple_roses_def
  sorry

end Mrs_Amaro_roses_l0_610


namespace crayon_difference_l0_463

theorem crayon_difference:
  let karen := 639
  let cindy := 504
  let peter := 752
  let rachel := 315
  max karen (max cindy (max peter rachel)) - min karen (min cindy (min peter rachel)) = 437 :=
by
  sorry

end crayon_difference_l0_463


namespace sum_remainders_mod_13_l0_171

theorem sum_remainders_mod_13 :
  ∀ (a b c d e : ℕ),
  a % 13 = 3 →
  b % 13 = 5 →
  c % 13 = 7 →
  d % 13 = 9 →
  e % 13 = 11 →
  (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end sum_remainders_mod_13_l0_171


namespace sunday_price_correct_l0_749

def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.60
def second_discount_rate : ℝ := 0.25
def discounted_price : ℝ := original_price * (1 - first_discount_rate)
def sunday_price : ℝ := discounted_price * (1 - second_discount_rate)

theorem sunday_price_correct :
  sunday_price = 75 := by
  sorry

end sunday_price_correct_l0_749


namespace multiply_by_11_l0_521

theorem multiply_by_11 (A B : ℕ) (h : A + B < 10) : 
  (10 * A + B) * 11 = 100 * A + 10 * (A + B) + B :=
by
  sorry

end multiply_by_11_l0_521


namespace negation_of_existential_square_inequality_l0_152

theorem negation_of_existential_square_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_square_inequality_l0_152


namespace joan_books_correct_l0_477

def sam_books : ℕ := 110
def total_books : ℕ := 212

def joan_books : ℕ := total_books - sam_books

theorem joan_books_correct : joan_books = 102 := by
  sorry

end joan_books_correct_l0_477


namespace cos_alpha_value_l0_355

theorem cos_alpha_value (α β γ: ℝ) (h1: β = 2 * α) (h2: γ = 4 * α)
 (h3: 2 * (Real.sin β) = (Real.sin α + Real.sin γ)) : Real.cos α = -1/2 := 
by
  sorry

end cos_alpha_value_l0_355


namespace max_gold_coins_l0_798

-- Define the conditions as predicates
def divides_with_remainder (n : ℕ) (d r : ℕ) : Prop := n % d = r
def less_than (n k : ℕ) : Prop := n < k

-- Main statement incorporating the conditions and the conclusion
theorem max_gold_coins (n : ℕ) :
  divides_with_remainder n 15 3 ∧ less_than n 120 → n ≤ 105 :=
by
  sorry

end max_gold_coins_l0_798


namespace gcd_987654_876543_eq_3_l0_966

theorem gcd_987654_876543_eq_3 :
  Nat.gcd 987654 876543 = 3 :=
sorry

end gcd_987654_876543_eq_3_l0_966


namespace sum_of_arithmetic_sequence_l0_729

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (a_6 : a 6 = 2) : 
  (11 * (a 1 + (a 1 + 10 * ((a 6 - a 1) / 5))) / 2) = 22 :=
by
  sorry

end sum_of_arithmetic_sequence_l0_729


namespace total_cups_l0_341

theorem total_cups (m c s : ℕ) (h1 : 3 * c = 2 * m) (h2 : 2 * c = 6) : m + c + s = 18 :=
by
  sorry

end total_cups_l0_341


namespace assignment_plan_count_l0_807

noncomputable def number_of_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let tasks := ["translation", "tour guide", "etiquette", "driver"]
  let v1 := ["Xiao Zhang", "Xiao Zhao"]
  let v2 := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Condition: Xiao Zhang and Xiao Zhao can only take positions for translation and tour guide
  -- Calculate the number of ways to assign based on the given conditions
  -- 36 is the total number of assignment plans
  36

theorem assignment_plan_count :
  number_of_assignment_plans = 36 :=
  sorry

end assignment_plan_count_l0_807


namespace laborer_savings_l0_132

theorem laborer_savings
  (monthly_expenditure_first6 : ℕ := 70)
  (monthly_expenditure_next4 : ℕ := 60)
  (monthly_income : ℕ := 69)
  (expenditure_first6 := 6 * monthly_expenditure_first6)
  (income_first6 := 6 * monthly_income)
  (debt : ℕ := expenditure_first6 - income_first6)
  (expenditure_next4 := 4 * monthly_expenditure_next4)
  (income_next4 := 4 * monthly_income)
  (savings : ℕ := income_next4 - (expenditure_next4 + debt)) :
  savings = 30 := 
by
  sorry

end laborer_savings_l0_132


namespace simplify_expression_l0_89

def expression (x y : ℤ) : ℤ := 
  ((2 * x + y) * (2 * x - y) - (2 * x - 3 * y)^2) / (-2 * y)

theorem simplify_expression {x y : ℤ} (hx : x = 1) (hy : y = -2) :
  expression x y = -16 :=
by 
  -- This proof will involve algebraic manipulation and substitution.
  sorry

end simplify_expression_l0_89


namespace tank_capacity_l0_738

theorem tank_capacity (C : ℝ) 
  (h1 : 10 > 0) 
  (h2 : 16 > (10 : ℝ))
  (h3 : ((C/10) - 480 = (C/16))) : C = 1280 := 
by 
  sorry

end tank_capacity_l0_738


namespace probability_of_circle_in_square_l0_167

open Real Set

theorem probability_of_circle_in_square :
  ∃ (p : ℝ), (∀ x y : ℝ, x ∈ Icc (-1 : ℝ) 1 → y ∈ Icc (-1 : ℝ) 1 → (x^2 + y^2 < 1/4) → True)
  → p = π / 16 :=
by
  use π / 16
  sorry

end probability_of_circle_in_square_l0_167


namespace brad_reads_more_pages_l0_927

-- Definitions based on conditions
def greg_pages_per_day : ℕ := 18
def brad_pages_per_day : ℕ := 26

-- Statement to prove
theorem brad_reads_more_pages : brad_pages_per_day - greg_pages_per_day = 8 :=
by
  -- sorry is used here to indicate the absence of a proof
  sorry

end brad_reads_more_pages_l0_927


namespace priyas_age_l0_484

/-- 
  Let P be Priya's current age, and F be her father's current age. 
  Given:
  1. F = P + 31
  2. (P + 8) + (F + 8) = 69
  Prove: Priya's current age P is 11.
-/
theorem priyas_age 
  (P F : ℕ) 
  (h1 : F = P + 31) 
  (h2 : (P + 8) + (F + 8) = 69) 
  : P = 11 :=
by
  sorry

end priyas_age_l0_484


namespace S_div_T_is_one_half_l0_429

def T (x y z : ℝ) := x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x + y + z = 1

def supports (a b c x y z : ℝ) := 
  (x >= a ∧ y >= b ∧ z < c) ∨ 
  (x >= a ∧ z >= c ∧ y < b) ∨ 
  (y >= b ∧ z >= c ∧ x < a)

def S (x y z : ℝ) := T x y z ∧ supports (1/4) (1/4) (1/2) x y z

theorem S_div_T_is_one_half :
  let area_T := 1 -- Normalizing since area of T is in fact √3 / 2 but we care about ratios
  let area_S := 1/2 * area_T -- Given by the problem solution
  area_S / area_T = 1/2 := 
sorry

end S_div_T_is_one_half_l0_429


namespace games_within_division_l0_587

/-- 
Given a baseball league with two four-team divisions,
where each team plays N games against other teams in its division,
and M games against teams in the other division.
Given that N > 2M and M > 6, and each team plays a total of 92 games in a season,
prove that each team plays 60 games within its own division.
-/
theorem games_within_division (N M : ℕ) (hN : N > 2 * M) (hM : M > 6) (h_total : 3 * N + 4 * M = 92) :
  3 * N = 60 :=
by
  -- The proof is omitted.
  sorry

end games_within_division_l0_587


namespace parabola_equation_l0_213

-- Defining the point F and the line
def F : ℝ × ℝ := (0, 4)

def line_eq (y : ℝ) : Prop := y = -5

-- Defining the condition that point M is closer to F(0, 4) than to the line y = -5 by less than 1
def condition (M : ℝ × ℝ) : Prop :=
  let dist_to_F := (M.1 - F.1)^2 + (M.2 - F.2)^2
  let dist_to_line := abs (M.2 - (-5))
  abs (dist_to_F - dist_to_line) < 1

-- The equation we need to prove under the given condition
theorem parabola_equation (M : ℝ × ℝ) (h : condition M) : M.1^2 = 16 * M.2 := 
sorry

end parabola_equation_l0_213


namespace probability_of_same_length_l0_937

-- Define the set T and its properties
def T : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def sides : ℕ := 6
def diagonals : ℕ := 9
def total_elements : ℕ := 15
def probability_same_length : ℚ := 17 / 35

-- Lean 4 statement (theorem)
theorem probability_of_same_length: 
    ( ∃ (sides diagonals : ℕ), sides = 6 ∧ diagonals = 9 ∧ sides + diagonals = 15 →
                                              ∃ probability_same_length : ℚ, probability_same_length = 17 / 35) := 
sorry

end probability_of_same_length_l0_937


namespace two_pow_15000_mod_1250_l0_544

theorem two_pow_15000_mod_1250 (h : 2 ^ 500 ≡ 1 [MOD 1250]) :
  2 ^ 15000 ≡ 1 [MOD 1250] :=
sorry

end two_pow_15000_mod_1250_l0_544


namespace abs_m_plus_one_l0_617

theorem abs_m_plus_one (m : ℝ) (h : |m| = m + 1) : (4 * m - 1) ^ 4 = 81 := by
  sorry

end abs_m_plus_one_l0_617


namespace find_xyz_l0_783

def divisible_by (n k : ℕ) : Prop := k % n = 0

def is_7_digit_number (a b c d e f g : ℕ) : ℕ := 
  10^6 * a + 10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + g

theorem find_xyz
  (x y z : ℕ)
  (h : divisible_by 792 (is_7_digit_number 1 4 x y 7 8 z))
  : (100 * x + 10 * y + z) = 644 :=
by
  sorry

end find_xyz_l0_783


namespace inverse_of_true_implies_negation_true_l0_253

variable (P : Prop)
theorem inverse_of_true_implies_negation_true (h : ¬ P) : ¬ P :=
by 
  exact h

end inverse_of_true_implies_negation_true_l0_253


namespace point_3_units_away_l0_705

theorem point_3_units_away (x : ℤ) (h : abs (x + 1) = 3) : x = 2 ∨ x = -4 :=
by
  sorry

end point_3_units_away_l0_705


namespace remainder_of_power_division_l0_823

theorem remainder_of_power_division :
  (2^222 + 222) % (2^111 + 2^56 + 1) = 218 :=
by sorry

end remainder_of_power_division_l0_823


namespace female_kittens_count_l0_596

theorem female_kittens_count (initial_cats total_cats male_kittens female_kittens : ℕ)
  (h1 : initial_cats = 2)
  (h2 : total_cats = 7)
  (h3 : male_kittens = 2)
  (h4 : female_kittens = total_cats - initial_cats - male_kittens) :
  female_kittens = 3 :=
by
  sorry

end female_kittens_count_l0_596


namespace gcd_of_54000_and_36000_l0_485

theorem gcd_of_54000_and_36000 : Nat.gcd 54000 36000 = 18000 := 
by sorry

end gcd_of_54000_and_36000_l0_485


namespace find_x_if_perpendicular_l0_457

-- Given definitions and conditions
def a : ℝ × ℝ := (-5, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Statement to be proved
theorem find_x_if_perpendicular (x : ℝ) :
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 10 :=
by
  sorry

end find_x_if_perpendicular_l0_457


namespace books_not_sold_l0_280

-- Definitions capturing the conditions
variable (B : ℕ)
variable (books_price : ℝ := 3.50)
variable (total_received : ℝ := 252)

-- Lean statement to capture the proof problem
theorem books_not_sold (h : (2 / 3 : ℝ) * B * books_price = total_received) :
  B / 3 = 36 :=
by
  sorry

end books_not_sold_l0_280


namespace pieces_from_rod_l0_256

theorem pieces_from_rod (length_of_rod : ℝ) (length_of_piece : ℝ) 
  (h_rod : length_of_rod = 42.5) 
  (h_piece : length_of_piece = 0.85) :
  length_of_rod / length_of_piece = 50 :=
by
  rw [h_rod, h_piece]
  calc
    42.5 / 0.85 = 50 := by norm_num

end pieces_from_rod_l0_256


namespace shirt_cost_l0_773

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 86) : S = 24 :=
by
  sorry

end shirt_cost_l0_773


namespace michelle_total_payment_l0_562
noncomputable def michelle_base_cost := 25
noncomputable def included_talk_time := 40 -- in hours
noncomputable def text_cost := 10 -- in cents per message
noncomputable def extra_talk_cost := 15 -- in cents per minute
noncomputable def february_texts_sent := 200
noncomputable def february_talk_time := 41 -- in hours

theorem michelle_total_payment : 
  25 + ((200 * 10) / 100) + (((41 - 40) * 60 * 15) / 100) = 54 := by
  sorry

end michelle_total_payment_l0_562


namespace alyssa_puppies_l0_54

-- Definitions from the problem conditions
def initial_puppies (P x : ℕ) : ℕ := P + x

-- Lean 4 Statement of the problem
theorem alyssa_puppies (P x : ℕ) (given_aw: 7 = 7) (remaining: 5 = 5) :
  initial_puppies P x = 12 :=
sorry

end alyssa_puppies_l0_54


namespace min_expression_value_l0_381

theorem min_expression_value (m n : ℝ) (h : m - n^2 = 1) : ∃ min_val : ℝ, min_val = 4 ∧ (∀ x y, x - y^2 = 1 → m^2 + 2 * y^2 + 4 * x - 1 ≥ min_val) :=
by
  sorry

end min_expression_value_l0_381


namespace proper_fraction_cubed_numerator_triples_denominator_add_three_l0_0

theorem proper_fraction_cubed_numerator_triples_denominator_add_three
  (a b : ℕ)
  (h1 : a < b)
  (h2 : (a^3 : ℚ) / (b + 3) = 3 * (a : ℚ) / b) : 
  a = 2 ∧ b = 9 :=
by
  sorry

end proper_fraction_cubed_numerator_triples_denominator_add_three_l0_0


namespace part1_part2_l0_483

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : (set_A ∪ set_B a = set_A ∩ set_B a) → a = 1 :=
sorry

theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -1 ∨ a = 1) :=
sorry

end part1_part2_l0_483


namespace count_distinct_digits_l0_579

theorem count_distinct_digits (n : ℕ) (h1 : ∃ (n : ℕ), n^3 = 125) : 
  n = 5 :=
by
  sorry

end count_distinct_digits_l0_579


namespace total_distance_walked_l0_780

noncomputable def hazel_total_distance : ℕ := 3

def distance_first_hour := 2  -- The distance traveled in the first hour (in kilometers)
def distance_second_hour := distance_first_hour * 2  -- The distance traveled in the second hour
def distance_third_hour := distance_second_hour / 2  -- The distance traveled in the third hour, with a 50% speed decrease

theorem total_distance_walked :
  distance_first_hour + distance_second_hour + distance_third_hour = 8 :=
  by
    sorry

end total_distance_walked_l0_780


namespace least_perimeter_of_triangle_l0_782

theorem least_perimeter_of_triangle (cosA cosB cosC : ℝ)
  (h₁ : cosA = 13 / 16)
  (h₂ : cosB = 4 / 5)
  (h₃ : cosC = -3 / 5) :
  ∃ a b c : ℕ, a + b + c = 28 ∧ 
  a^2 + b^2 - c^2 = 2 * a * b * cosC ∧ 
  b^2 + c^2 - a^2 = 2 * b * c * cosA ∧ 
  c^2 + a^2 - b^2 = 2 * c * a * cosB :=
sorry

end least_perimeter_of_triangle_l0_782


namespace largest_mersenne_prime_less_than_500_l0_135

def mersenne_prime (n : ℕ) : ℕ := 2^n - 1

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem largest_mersenne_prime_less_than_500 :
  ∃ n, is_prime n ∧ mersenne_prime n < 500 ∧ ∀ m, is_prime m ∧ mersenne_prime m < 500 → mersenne_prime m ≤ mersenne_prime n :=
  sorry

end largest_mersenne_prime_less_than_500_l0_135


namespace largest_constant_C_l0_96

theorem largest_constant_C (C : ℝ) : C = 2 / Real.sqrt 3 ↔ ∀ (x y z : ℝ), x^2 + y^2 + 2 * z^2 + 1 ≥ C * (x + y + z) :=
by
  sorry

end largest_constant_C_l0_96


namespace brianne_savings_ratio_l0_191

theorem brianne_savings_ratio
  (r : ℝ)
  (H1 : 10 * r^4 = 160) :
  r = 2 :=
by 
  sorry

end brianne_savings_ratio_l0_191


namespace infinite_geometric_series_correct_l0_752

noncomputable def infinite_geometric_series_sum : ℚ :=
  let a : ℚ := 5 / 3
  let r : ℚ := -9 / 20
  a / (1 - r)

theorem infinite_geometric_series_correct : infinite_geometric_series_sum = 100 / 87 := 
by
  sorry

end infinite_geometric_series_correct_l0_752


namespace arithmetic_sequence_minimum_value_S_l0_995

noncomputable def S (n : ℕ) : ℤ := sorry -- The sum of the first n terms of the sequence a_n

def a (n : ℕ) : ℤ := sorry -- Defines a_n

axiom condition1 (n : ℕ) : (2 * S n / n + n = 2 * a n + 1)

theorem arithmetic_sequence (n : ℕ) : ∃ d : ℤ, ∀ k : ℕ, a (k + 1) = a k + d := sorry

axiom geometric_sequence : a 7 ^ 2 = a 4 * a 9

theorem minimum_value_S : ∀ n : ℕ, (a 4 < a 7 ∧ a 7 < a 9) → S n ≥ -78 := sorry

end arithmetic_sequence_minimum_value_S_l0_995


namespace Blair_17th_turn_l0_679

/-
  Jo begins counting by saying "5". Blair then continues the sequence, each time saying a number that is 2 more than the last number Jo said. Jo increments by 1 each turn after Blair. They alternate turns.
  Prove that Blair says the number 55 on her 17th turn.
-/

def Jo_initial := 5
def increment_Jo := 1
def increment_Blair := 2

noncomputable def blair_sequence (n : ℕ) : ℕ :=
  Jo_initial + increment_Blair + (n - 1) * (increment_Jo + increment_Blair)

theorem Blair_17th_turn : blair_sequence 17 = 55 := by
    sorry

end Blair_17th_turn_l0_679


namespace average_age_of_two_women_is_30_l0_985

-- Given definitions
def avg_age_before_replacement (A : ℝ) := 8 * A
def avg_age_after_increase (A : ℝ) := 8 * (A + 2)
def ages_of_men_replaced := 20 + 24

-- The theorem to prove: the average age of the two women is 30 years
theorem average_age_of_two_women_is_30 (A : ℝ) :
  (avg_age_after_increase A) - (avg_age_before_replacement A) = 16 →
  (ages_of_men_replaced + 16) / 2 = 30 :=
by
  sorry

end average_age_of_two_women_is_30_l0_985


namespace sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l0_935

theorem sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3 :
  let largestThreeDigitMultipleOf4 := 996
  let smallestFourDigitMultipleOf3 := 1002
  largestThreeDigitMultipleOf4 + smallestFourDigitMultipleOf3 = 1998 :=
by
  sorry

end sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l0_935


namespace vinegar_used_is_15_l0_348

noncomputable def vinegar_used (T : ℝ) : ℝ :=
  let water := (3 / 5) * 20
  let total_volume := 27
  let vinegar := total_volume - water
  vinegar

theorem vinegar_used_is_15 (T : ℝ) (h1 : (3 / 5) * 20 = 12) (h2 : 27 - 12 = 15) (h3 : (5 / 6) * T = 15) : vinegar_used T = 15 :=
by
  sorry

end vinegar_used_is_15_l0_348


namespace percentage_increase_l0_611

theorem percentage_increase (original_interval : ℕ) (new_interval : ℕ) 
  (h1 : original_interval = 30) (h2 : new_interval = 45) :
  ((new_interval - original_interval) / original_interval) * 100 = 50 := 
by 
  -- Provide the proof here
  sorry

end percentage_increase_l0_611


namespace chess_tournament_games_l0_74

def num_games (n : Nat) : Nat := n * (n - 1) * 2

theorem chess_tournament_games : num_games 7 = 84 :=
by
  sorry

end chess_tournament_games_l0_74


namespace julie_upstream_distance_l0_784

noncomputable def speed_of_stream : ℝ := 0.5
noncomputable def distance_downstream : ℝ := 72
noncomputable def time_spent : ℝ := 4
noncomputable def speed_of_julie_in_still_water : ℝ := 17.5
noncomputable def distance_upstream : ℝ := 68

theorem julie_upstream_distance :
  (distance_upstream / (speed_of_julie_in_still_water - speed_of_stream) = time_spent) ∧
  (distance_downstream / (speed_of_julie_in_still_water + speed_of_stream) = time_spent) →
  distance_upstream = 68 :=
by 
  sorry

end julie_upstream_distance_l0_784


namespace average_mileage_highway_l0_896

theorem average_mileage_highway (H : Real) : 
  (∀ d : Real, (d / 7.6) > 23 → false) → 
  (280.6 / 23 = H) → 
  H = 12.2 := by
  sorry

end average_mileage_highway_l0_896


namespace petrov_vasechkin_boards_l0_155

theorem petrov_vasechkin_boards:
  ∃ n : ℕ, 
  (∃ x y : ℕ, 2 * x + 3 * y = 87 ∧ x + y = n) ∧ 
  (∃ u v : ℕ, 3 * u + 5 * v = 94 ∧ u + v = n) ∧ 
  n = 30 := 
sorry

end petrov_vasechkin_boards_l0_155


namespace max_power_sum_l0_17

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l0_17


namespace nonagon_isosceles_triangle_count_l0_109

theorem nonagon_isosceles_triangle_count (N : ℕ) (hN : N = 9) : 
  ∃(k : ℕ), k = 30 := 
by 
  have h := hN
  sorry      -- Solution steps would go here if we were proving it

end nonagon_isosceles_triangle_count_l0_109


namespace octal_to_decimal_l0_2

theorem octal_to_decimal (d0 d1 : ℕ) (n8 : ℕ) (n10 : ℕ) 
  (h1 : d0 = 3) (h2 : d1 = 5) (h3 : n8 = 53) (h4 : n10 = 43) : 
  (d1 * 8^1 + d0 * 8^0 = n10) :=
by
  sorry

end octal_to_decimal_l0_2


namespace discriminant_formula_l0_853

def discriminant_cubic_eq (x1 x2 x3 p q : ℝ) : ℝ :=
  (x1 - x2)^2 * (x2 - x3)^2 * (x3 - x1)^2

theorem discriminant_formula (x1 x2 x3 p q : ℝ)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : x1 * x2 + x1 * x3 + x2 * x3 = p)
  (h3 : x1 * x2 * x3 = -q) :
  discriminant_cubic_eq x1 x2 x3 p q = -4 * p^3 - 27 * q^2 :=
by sorry

end discriminant_formula_l0_853


namespace x_eq_zero_sufficient_not_necessary_l0_972

theorem x_eq_zero_sufficient_not_necessary (x : ℝ) : 
  (x = 0 → x^2 - 2 * x = 0) ∧ (x^2 - 2 * x = 0 → x = 0 ∨ x = 2) :=
by
  sorry

end x_eq_zero_sufficient_not_necessary_l0_972


namespace range_of_x_if_p_and_q_range_of_a_if_not_p_sufficient_for_not_q_l0_765

variable (x a : ℝ)

-- Condition p
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

-- Condition q
def q (x : ℝ) : Prop :=
  (x^2 - x - 6 ≤ 0) ∧ (x^2 + 3*x - 10 > 0)

-- Proof problem for question (1)
theorem range_of_x_if_p_and_q (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 :=
  sorry

-- Proof problem for question (2)
theorem range_of_a_if_not_p_sufficient_for_not_q (h : (¬p x a) → (¬q x)) : 1 < a ∧ a ≤ 2 :=
  sorry

end range_of_x_if_p_and_q_range_of_a_if_not_p_sufficient_for_not_q_l0_765


namespace find_angle_D_l0_558

theorem find_angle_D (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : A + B + C + D = 360) : D = 60 :=
sorry

end find_angle_D_l0_558


namespace expression_equals_6_l0_540

-- Define the expression as a Lean definition.
def expression : ℤ := 1 - (-2) - 3 - (-4) - 5 - (-6) - 7 - (-8)

-- The statement to prove that the expression equals 6.
theorem expression_equals_6 : expression = 6 := 
by 
  -- This is a placeholder for the actual proof.
  sorry

end expression_equals_6_l0_540


namespace basketball_scores_l0_111

theorem basketball_scores (n : ℕ) (h : n = 7) : 
  ∃ (k : ℕ), k = 8 :=
by {
  sorry
}

end basketball_scores_l0_111


namespace work_completion_together_l0_465

theorem work_completion_together (man_days : ℕ) (son_days : ℕ) (together_days : ℕ) 
  (h_man : man_days = 10) (h_son : son_days = 10) : together_days = 5 :=
by sorry

end work_completion_together_l0_465


namespace peter_total_spent_l0_233

/-
Peter bought a scooter for a certain sum of money. He spent 5% of the cost on the first round of repairs, another 10% on the second round of repairs, and 7% on the third round of repairs. After this, he had to pay a 12% tax on the original cost. Also, he offered a 15% holiday discount on the scooter's selling price. Despite the discount, he still managed to make a profit of $2000. How much did he spend in total, including repairs, tax, and discount if his profit percentage was 30%?
-/

noncomputable def total_spent (C S P : ℝ) : Prop :=
    (0.3 * C = P) ∧
    (0.85 * S = 1.34 * C + P) ∧
    (C = 2000 / 0.3) ∧
    (1.34 * C = 8933.33)

theorem peter_total_spent
  (C S P : ℝ)
  (h1 : 0.3 * C = P)
  (h2 : 0.85 * S = 1.34 * C + P)
  (h3 : C = 2000 / 0.3)
  : 1.34 * C = 8933.33 := by 
  sorry

end peter_total_spent_l0_233


namespace total_dividends_received_l0_200

theorem total_dividends_received
  (investment : ℝ)
  (share_price : ℝ)
  (nominal_value : ℝ)
  (dividend_rate_year1 : ℝ)
  (dividend_rate_year2 : ℝ)
  (dividend_rate_year3 : ℝ)
  (num_shares : ℝ)
  (total_dividends : ℝ) :
  investment = 14400 →
  share_price = 120 →
  nominal_value = 100 →
  dividend_rate_year1 = 0.07 →
  dividend_rate_year2 = 0.09 →
  dividend_rate_year3 = 0.06 →
  num_shares = investment / share_price → 
  total_dividends = (dividend_rate_year1 * nominal_value * num_shares) +
                    (dividend_rate_year2 * nominal_value * num_shares) +
                    (dividend_rate_year3 * nominal_value * num_shares) →
  total_dividends = 2640 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end total_dividends_received_l0_200


namespace complex_quadrant_l0_128

theorem complex_quadrant (z : ℂ) (h : z = (↑(1/2) : ℂ) + (↑(1/2) : ℂ) * I ) : 
  0 < z.re ∧ 0 < z.im :=
by {
sorry -- Proof goes here
}

end complex_quadrant_l0_128


namespace longest_segment_in_cylinder_l0_332

noncomputable def cylinder_diagonal (radius height : ℝ) : ℝ :=
  Real.sqrt (height^2 + (2 * radius)^2)

theorem longest_segment_in_cylinder :
  cylinder_diagonal 4 10 = 2 * Real.sqrt 41 :=
by
  -- Proof placeholder
  sorry

end longest_segment_in_cylinder_l0_332


namespace C_should_pay_correct_amount_l0_583

def A_oxen_months : ℕ := 10 * 7
def B_oxen_months : ℕ := 12 * 5
def C_oxen_months : ℕ := 15 * 3
def D_oxen_months : ℕ := 20 * 6

def total_rent : ℚ := 225

def C_share_of_rent : ℚ :=
  total_rent * (C_oxen_months : ℚ) / (A_oxen_months + B_oxen_months + C_oxen_months + D_oxen_months)

theorem C_should_pay_correct_amount : C_share_of_rent = 225 * (45 : ℚ) / 295 := by
  sorry

end C_should_pay_correct_amount_l0_583


namespace next_ten_winners_each_receive_160_l0_689

def total_prize : ℕ := 2400
def first_winner_share : ℚ := 1 / 3 * total_prize
def remaining_after_first : ℚ := total_prize - first_winner_share
def next_ten_winners_share : ℚ := remaining_after_first / 10

theorem next_ten_winners_each_receive_160 :
  next_ten_winners_share = 160 := by
sorry

end next_ten_winners_each_receive_160_l0_689


namespace set_B_correct_l0_722

-- Define the set A
def A : Set ℤ := {-1, 0, 1, 2}

-- Define the set B using the given formula
def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2 * x}

-- State the theorem that B is equal to the given set {-1, 0, 3}
theorem set_B_correct : B = {-1, 0, 3} := 
by 
  sorry

end set_B_correct_l0_722


namespace decrease_A_share_l0_761

theorem decrease_A_share :
  ∃ (a b x : ℝ),
    a + b + 495 = 1010 ∧
    (a - x) / 3 = 96 ∧
    (b - 10) / 2 = 96 ∧
    x = 25 :=
by
  sorry

end decrease_A_share_l0_761


namespace crayons_in_new_set_l0_991

theorem crayons_in_new_set (initial_crayons : ℕ) (half_loss : ℕ) (total_after_purchase : ℕ) (initial_crayons_eq : initial_crayons = 18) (half_loss_eq : half_loss = initial_crayons / 2) (total_eq : total_after_purchase = 29) :
  total_after_purchase - (initial_crayons - half_loss) = 20 :=
by
  sorry

end crayons_in_new_set_l0_991


namespace probability_six_distinct_numbers_l0_532

theorem probability_six_distinct_numbers :
  let total_outcomes := 6^6
  let distinct_outcomes := Nat.factorial 6
  let probability := (distinct_outcomes:ℚ) / (total_outcomes:ℚ)
  probability = 5 / 324 :=
sorry

end probability_six_distinct_numbers_l0_532


namespace find_value_of_ratio_l0_67

theorem find_value_of_ratio (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x / y + y / x = 4) :
  (x + 2 * y) / (x - 2 * y) = Real.sqrt 33 / 3 := 
  sorry

end find_value_of_ratio_l0_67


namespace number_of_valid_ns_l0_578

theorem number_of_valid_ns :
  ∃ (S : Finset ℕ), S.card = 13 ∧ ∀ n ∈ S, n ≤ 1000 ∧ Nat.floor (995 / n) + Nat.floor (996 / n) + Nat.floor (997 / n) % 4 ≠ 0 :=
by
  sorry

end number_of_valid_ns_l0_578


namespace circle_equation_standard_l0_563

open Real

noncomputable def equation_of_circle : Prop :=
  ∃ R : ℝ, R = sqrt 2 ∧ 
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → x + y - 2 = 0 → 0 ≤ x ∧ x ≤ 2)

theorem circle_equation_standard :
    equation_of_circle := sorry

end circle_equation_standard_l0_563


namespace find_rectangle_width_l0_913

noncomputable def area_of_square_eq_5times_area_of_rectangle (s l : ℝ) (w : ℝ) :=
  s^2 = 5 * (l * w)

noncomputable def perimeter_of_square_eq_160 (s : ℝ) :=
  4 * s = 160

theorem find_rectangle_width : ∃ w : ℝ, ∀ l : ℝ, 
  area_of_square_eq_5times_area_of_rectangle 40 l w ∧
  perimeter_of_square_eq_160 40 → 
  w = 10 :=
by
  sorry

end find_rectangle_width_l0_913


namespace triangle_angles_ratios_l0_394

def angles_of_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

theorem triangle_angles_ratios (α β γ : ℝ)
  (h1 : α + β + γ = 180) 
  (h2 : β = 2 * α)
  (h3 : γ = 3 * α) : 
  angles_of_triangle 60 45 75 ∨ angles_of_triangle 45 22.5 112.5 :=
by
  sorry

end triangle_angles_ratios_l0_394


namespace two_workers_two_hours_holes_l0_802

theorem two_workers_two_hours_holes
    (workers1: ℝ) (holes1: ℝ) (hours1: ℝ)
    (workers2: ℝ) (hours2: ℝ)
    (h1: workers1 = 1.5)
    (h2: holes1 = 1.5)
    (h3: hours1 = 1.5)
    (h4: workers2 = 2)
    (h5: hours2 = 2)
    : (workers2 * (holes1 / (workers1 * hours1)) * hours2 = 8 / 3) := 
by {
   -- To be filled with proof, currently a placeholder.
  sorry
}

end two_workers_two_hours_holes_l0_802


namespace value_of_algebraic_expression_l0_707

variable {a b : ℝ}

theorem value_of_algebraic_expression (h : b = 4 * a + 3) : 4 * a - b - 2 = -5 := 
by
  sorry

end value_of_algebraic_expression_l0_707


namespace exists_tangent_inequality_l0_294

theorem exists_tangent_inequality {x : Fin 8 → ℝ} (h : Function.Injective x) :
  ∃ (i j : Fin 8), i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (Real.pi / 7) :=
by
  sorry

end exists_tangent_inequality_l0_294


namespace part1_inequality_l0_356

noncomputable def f (x : ℝ) : ℝ := x - 2
noncomputable def g (x m : ℝ) : ℝ := x^2 - 2 * m * x + 4

theorem part1_inequality (m : ℝ) : (∀ x : ℝ, g x m > f x) ↔ (m ∈ Set.Ioo (-Real.sqrt 6 - (1/2)) (Real.sqrt 6 - (1/2))) :=
sorry

end part1_inequality_l0_356


namespace boat_travel_distance_downstream_l0_176

-- Define the conditions given in the problem
def speed_boat_still_water := 22 -- in km/hr
def speed_stream := 5 -- in km/hr
def time_downstream := 2 -- in hours

-- Define a function to compute the effective speed downstream
def effective_speed_downstream (speed_boat: ℝ) (speed_stream: ℝ) : ℝ :=
  speed_boat + speed_stream

-- Define a function to compute the distance travelled downstream
def distance_downstream (speed: ℝ) (time: ℝ) : ℝ :=
  speed * time

-- The main theorem to prove
theorem boat_travel_distance_downstream :
  distance_downstream (effective_speed_downstream speed_boat_still_water speed_stream) time_downstream = 54 :=
by
  -- Proof is to be filled in later
  sorry

end boat_travel_distance_downstream_l0_176


namespace bananas_unit_measurement_l0_30

-- Definition of given conditions
def units_per_day : ℕ := 13
def total_bananas : ℕ := 9828
def total_weeks : ℕ := 9
def days_per_week : ℕ := 7
def total_days : ℕ := total_weeks * days_per_week
def bananas_per_day : ℕ := total_bananas / total_days
def bananas_per_unit : ℕ := bananas_per_day / units_per_day

-- Main theorem statement
theorem bananas_unit_measurement :
  bananas_per_unit = 12 := sorry

end bananas_unit_measurement_l0_30


namespace contrapositive_of_implication_l0_908

theorem contrapositive_of_implication (a : ℝ) (h : a > 0 → a > 1) : a ≤ 1 → a ≤ 0 :=
by
  sorry

end contrapositive_of_implication_l0_908


namespace vote_majority_is_160_l0_338

-- Define the total number of votes polled
def total_votes : ℕ := 400

-- Define the percentage of votes polled by the winning candidate
def winning_percentage : ℝ := 0.70

-- Define the percentage of votes polled by the losing candidate
def losing_percentage : ℝ := 0.30

-- Define the number of votes gained by the winning candidate
def winning_votes := winning_percentage * total_votes

-- Define the number of votes gained by the losing candidate
def losing_votes := losing_percentage * total_votes

-- Define the vote majority
def vote_majority := winning_votes - losing_votes

-- Prove that the vote majority is 160 votes
theorem vote_majority_is_160 : vote_majority = 160 :=
sorry

end vote_majority_is_160_l0_338


namespace num_ordered_triples_unique_l0_790

theorem num_ordered_triples_unique : 
  (∃! (x y z : ℝ), x + y = 2 ∧ xy - z^2 = 1) := 
by 
  sorry 

end num_ordered_triples_unique_l0_790


namespace initial_percentage_increase_l0_887

-- Given conditions
def S_original : ℝ := 4000.0000000000005
def S_final : ℝ := 4180
def reduction : ℝ := 5

-- Predicate to prove the initial percentage increase is 10%
theorem initial_percentage_increase (x : ℝ) 
  (hx : (95/100) * (S_original * (1 + x / 100)) = S_final) : 
  x = 10 :=
sorry

end initial_percentage_increase_l0_887


namespace zero_in_M_l0_902

-- Define the set M
def M : Set ℕ := {0, 1, 2}

-- State the theorem to be proved
theorem zero_in_M : 0 ∈ M := 
  sorry

end zero_in_M_l0_902


namespace cost_of_seven_CDs_l0_514

theorem cost_of_seven_CDs (cost_per_two : ℝ) (h1 : cost_per_two = 32) : (7 * (cost_per_two / 2)) = 112 :=
by
  sorry

end cost_of_seven_CDs_l0_514


namespace each_piglet_ate_9_straws_l0_3

theorem each_piglet_ate_9_straws (t : ℕ) (h_t : t = 300)
                                 (p : ℕ) (h_p : p = 20)
                                 (f : ℕ) (h_f : f = (3 * t / 5)) :
  f / p = 9 :=
by
  sorry

end each_piglet_ate_9_straws_l0_3


namespace main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l0_434

-- Problem Statement in Lean 4

theorem main_diagonal_squares (k : ℕ) : ∃ m : ℕ, (4 * k * (k + 1) + 1 = m * m) := 
sorry

theorem second_diagonal_composite (k : ℕ) (hk : k ≥ 1) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * (2 * k * (2 * k - 1) - 1) + 1 = a * b) :=
sorry

theorem third_diagonal_composite (k : ℕ) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * ((4 * k + 3) * (4 * k - 1)) + 1 = a * b) :=
sorry

end main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l0_434


namespace arithmetic_sequence_ratio_l0_711

theorem arithmetic_sequence_ratio (S T : ℕ → ℕ) (a b : ℕ → ℕ)
  (h : ∀ n, S n / T n = (7 * n + 3) / (n + 3)) :
  a 8 / b 8 = 6 :=
by
  sorry

end arithmetic_sequence_ratio_l0_711


namespace four_digit_number_condition_l0_859

theorem four_digit_number_condition (x n : ℕ) (h1 : n = 2000 + x) (h2 : 10 * x + 2 = 2 * n + 66) : n = 2508 :=
sorry

end four_digit_number_condition_l0_859


namespace actual_distance_between_towns_l0_33

-- Definitions based on conditions
def scale_inch_to_miles : ℚ := 8
def map_distance_inches : ℚ := 27 / 8

-- Proof statement
theorem actual_distance_between_towns : scale_inch_to_miles * map_distance_inches / (1 / 4) = 108 := by
  sorry

end actual_distance_between_towns_l0_33


namespace sum_of_angles_l0_210

theorem sum_of_angles (a b : ℝ) (ha : a = 45) (hb : b = 225) : a + b = 270 :=
by
  rw [ha, hb]
  norm_num -- Lean's built-in tactic to normalize numerical expressions

end sum_of_angles_l0_210


namespace stocks_higher_price_l0_993

theorem stocks_higher_price (total_stocks lower_price higher_price: ℝ)
  (h_total: total_stocks = 8000)
  (h_ratio: higher_price = 1.5 * lower_price)
  (h_sum: lower_price + higher_price = total_stocks) :
  higher_price = 4800 :=
by
  sorry

end stocks_higher_price_l0_993


namespace geometric_series_sum_l0_688

theorem geometric_series_sum : 
  let a : ℝ := 1 / 4
  let r : ℝ := 1 / 2
  |r| < 1 → (∑' n:ℕ, a * r^n) = 1 / 2 :=
by
  sorry

end geometric_series_sum_l0_688


namespace rectangular_prism_total_count_l0_725

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

end rectangular_prism_total_count_l0_725


namespace kafelnikov_served_in_first_game_l0_656

theorem kafelnikov_served_in_first_game (games : ℕ) (kafelnikov_wins : ℕ) (becker_wins : ℕ)
  (server_victories : ℕ) (x y : ℕ) 
  (h1 : kafelnikov_wins = 6)
  (h2 : becker_wins = 3)
  (h3 : server_victories = 5)
  (h4 : games = 9)
  (h5 : kafelnikov_wins + becker_wins = games)
  (h6 : (5 - x) + y = 5) 
  (h7 : x + y = 6):
  x = 3 :=
by
  sorry

end kafelnikov_served_in_first_game_l0_656


namespace bead_problem_l0_756

theorem bead_problem 
  (x y : ℕ) 
  (hx : 19 * x + 17 * y = 2017): 
  (x + y = 107) ∨ (x + y = 109) ∨ (x + y = 111) ∨ (x + y = 113) ∨ (x + y = 115) ∨ (x + y = 117) := 
sorry

end bead_problem_l0_756


namespace students_in_class_l0_559

/-- Conditions:
1. 20 hands in Peter’s class, not including his.
2. Every student in the class has 2 hands.

Prove that the number of students in Peter’s class including him is 11.
-/
theorem students_in_class (hands_without_peter : ℕ) (hands_per_student : ℕ) (students_including_peter : ℕ) :
  hands_without_peter = 20 →
  hands_per_student = 2 →
  students_including_peter = (hands_without_peter + hands_per_student) / hands_per_student →
  students_including_peter = 11 :=
by
  intros h₁ h₂ h₃
  sorry

end students_in_class_l0_559


namespace evaluate_expression_l0_144

-- Definitions for conditions
def x := (1 / 4 : ℚ)
def y := (1 / 2 : ℚ)
def z := (3 : ℚ)

-- Statement of the problem
theorem evaluate_expression : 
  4 * (x^3 * y^2 * z^2) = 9 / 64 :=
by
  sorry

end evaluate_expression_l0_144


namespace product_of_two_numbers_l0_755

theorem product_of_two_numbers (a b : ℝ)
  (h1 : a + b = 8 * (a - b))
  (h2 : a * b = 30 * (a - b)) :
  a * b = 400 / 7 :=
by
  sorry

end product_of_two_numbers_l0_755


namespace domain_of_f_eq_l0_998

def domain_of_fractional_function : Set ℝ := 
  { x : ℝ | x > -1 }

theorem domain_of_f_eq : 
  ∀ x : ℝ, x ∈ domain_of_fractional_function ↔ x > -1 :=
by
  sorry -- Proof this part in Lean 4. The domain of f(x) is (-1, +∞)

end domain_of_f_eq_l0_998


namespace tangent_line_eq_l0_386

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

def derivative_curve (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

-- Define the problem as a theorem statement
theorem tangent_line_eq (L : ℝ → ℝ) (hL : ∀ x, L x = 2 * x ∨ L x = - x/4) :
  (∀ x, x = 0 → L x = 0) →
  (∀ x x0, L x = curve x → derivative_curve x0 = derivative_curve 0 → x0 = 0 ∨ x0 = 3/2) →
  (L x = 2 * x - curve x ∨ L x = 4 * x + curve x) :=
by
  sorry

end tangent_line_eq_l0_386


namespace total_distinct_symbols_l0_855

def numSequences (n : ℕ) : ℕ := 3^n

theorem total_distinct_symbols :
  numSequences 1 + numSequences 2 + numSequences 3 + numSequences 4 = 120 :=
by
  sorry

end total_distinct_symbols_l0_855


namespace find_q_zero_l0_129

-- Assuming the polynomials p, q, and r are defined, and their relevant conditions are satisfied.

def constant_term (f : ℕ → ℝ) : ℝ := f 0

theorem find_q_zero (p q r : ℕ → ℝ)
  (h : p * q = r)
  (h_p_const : constant_term p = 5)
  (h_r_const : constant_term r = -10) :
  q 0 = -2 :=
sorry

end find_q_zero_l0_129


namespace total_trees_planted_l0_32

theorem total_trees_planted :
  let fourth_graders := 30
  let fifth_graders := 2 * fourth_graders
  let sixth_graders := 3 * fifth_graders - 30
  fourth_graders + fifth_graders + sixth_graders = 240 :=
by
  sorry

end total_trees_planted_l0_32


namespace compute_fraction_power_l0_945

theorem compute_fraction_power (a b : ℕ) (ha : a = 123456) (hb : b = 41152) : (a ^ 5 / b ^ 5) = 243 := by
  sorry

end compute_fraction_power_l0_945


namespace sum_n_10_terms_progression_l0_781

noncomputable def sum_arith_progression (n a d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_n_10_terms_progression :
  ∃ (a : ℕ), (∃ (n : ℕ), sum_arith_progression n a 3 = 220) ∧
  (2 * a + (10 - 1) * 3) = 43 ∧
  sum_arith_progression 10 a 3 = 215 :=
by sorry

end sum_n_10_terms_progression_l0_781


namespace rectangle_length_l0_178

theorem rectangle_length (P W : ℝ) (hP : P = 40) (hW : W = 8) : ∃ L : ℝ, 2 * (L + W) = P ∧ L = 12 := 
by 
  sorry

end rectangle_length_l0_178


namespace cooper_needs_1043_bricks_l0_878

def wall1_length := 15
def wall1_height := 6
def wall1_depth := 3

def wall2_length := 20
def wall2_height := 4
def wall2_depth := 2

def wall3_length := 25
def wall3_height := 5
def wall3_depth := 3

def wall4_length := 17
def wall4_height := 7
def wall4_depth := 2

def bricks_needed_for_wall (length height depth: Nat) : Nat :=
  length * height * depth

def total_bricks_needed : Nat :=
  bricks_needed_for_wall wall1_length wall1_height wall1_depth +
  bricks_needed_for_wall wall2_length wall2_height wall2_depth +
  bricks_needed_for_wall wall3_length wall3_height wall3_depth +
  bricks_needed_for_wall wall4_length wall4_height wall4_depth

theorem cooper_needs_1043_bricks : total_bricks_needed = 1043 := by
  sorry

end cooper_needs_1043_bricks_l0_878


namespace divisors_count_30_l0_411

theorem divisors_count_30 : 
  (∃ n : ℤ, n > 1 ∧ 30 % n = 0) 
  → 
  (∃ k : ℕ, k = 14) :=
by
  sorry

end divisors_count_30_l0_411


namespace total_faces_is_198_l0_545

-- Definitions for the number of dice and geometrical shapes brought by each person:
def TomDice : ℕ := 4
def TimDice : ℕ := 5
def TaraDice : ℕ := 3
def TinaDice : ℕ := 2
def TonyCubes : ℕ := 1
def TonyTetrahedrons : ℕ := 3
def TonyIcosahedrons : ℕ := 2

-- Definitions for the number of faces for each type of dice or shape:
def SixSidedFaces : ℕ := 6
def EightSidedFaces : ℕ := 8
def TwelveSidedFaces : ℕ := 12
def TwentySidedFaces : ℕ := 20
def CubeFaces : ℕ := 6
def TetrahedronFaces : ℕ := 4
def IcosahedronFaces : ℕ := 20

-- We want to prove that the total number of faces is 198:
theorem total_faces_is_198 : 
  (TomDice * SixSidedFaces) + 
  (TimDice * EightSidedFaces) + 
  (TaraDice * TwelveSidedFaces) + 
  (TinaDice * TwentySidedFaces) + 
  (TonyCubes * CubeFaces) + 
  (TonyTetrahedrons * TetrahedronFaces) + 
  (TonyIcosahedrons * IcosahedronFaces) 
  = 198 := 
by {
  sorry
}

end total_faces_is_198_l0_545


namespace knight_tour_impossible_49_squares_l0_692

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

end knight_tour_impossible_49_squares_l0_692


namespace net_change_in_price_net_change_percentage_l0_218

theorem net_change_in_price (P : ℝ) :
  0.80 * P * 1.55 - P = 0.24 * P :=
by sorry

theorem net_change_percentage (P : ℝ) :
  ((0.80 * P * 1.55 - P) / P) * 100 = 24 :=
by sorry


end net_change_in_price_net_change_percentage_l0_218


namespace Bhupathi_amount_l0_806

variable (A B : ℝ)

theorem Bhupathi_amount
  (h1 : A + B = 1210)
  (h2 : (4 / 15) * A = (2 / 5) * B) :
  B = 484 := by
  sorry

end Bhupathi_amount_l0_806


namespace problem_equivalent_l0_461

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l0_461


namespace range_3x_plus_2y_l0_903

theorem range_3x_plus_2y (x y : ℝ) : -1 < x + y ∧ x + y < 4 → 2 < x - y ∧ x - y < 3 → 
  -3/2 < 3*x + 2*y ∧ 3*x + 2*y < 23/2 :=
by
  sorry

end range_3x_plus_2y_l0_903


namespace triangle_properties_l0_543

theorem triangle_properties
  (a b : ℝ)
  (C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hC : C = Real.pi / 3)
  :
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)
  let area := (1 / 2) * a * b * Real.sin C
  let sin2A := 2 * (a * Real.sin C / c) * Real.sqrt (1 - (a * Real.sin C / c)^2)
  c = Real.sqrt 7 
  ∧ area = (3 * Real.sqrt 3) / 2 
  ∧ sin2A = (4 * Real.sqrt 3) / 7 :=
by
  sorry

end triangle_properties_l0_543


namespace solve_quadratic_equation_l0_444

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2 := by
sorry

end solve_quadratic_equation_l0_444


namespace find_a_even_function_l0_800

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_domain : ∀ x, 2 * a + 1 ≤ x ∧ x ≤ a + 5) :
  a = -2 :=
sorry

end find_a_even_function_l0_800


namespace range_of_values_for_a_l0_409

theorem range_of_values_for_a 
  (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x = x - 1/x - a * Real.log x)
  (h2 : ∀ x > 0, (x^2 - a * x + 1) ≥ 0) : 
  a ≤ 2 :=
sorry

end range_of_values_for_a_l0_409


namespace parabola_vertex_above_x_axis_l0_211

theorem parabola_vertex_above_x_axis (k : ℝ) (h : k > 9 / 4) : 
  ∃ y : ℝ, ∀ x : ℝ, y = (x - 3 / 2) ^ 2 + k - 9 / 4 ∧ y > 0 := 
by
  sorry

end parabola_vertex_above_x_axis_l0_211


namespace proof_problem_l0_353

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := 2 * x + 4

theorem proof_problem : (f (g 3))^2 - (g (f 3))^2 = 28 := by
  sorry

end proof_problem_l0_353


namespace boxes_containing_neither_l0_344

theorem boxes_containing_neither
  (total_boxes : ℕ)
  (boxes_with_stickers : ℕ)
  (boxes_with_cards : ℕ)
  (boxes_with_both : ℕ)
  (h1 : total_boxes = 15)
  (h2 : boxes_with_stickers = 8)
  (h3 : boxes_with_cards = 5)
  (h4 : boxes_with_both = 3) :
  (total_boxes - (boxes_with_stickers + boxes_with_cards - boxes_with_both)) = 5 :=
by
  sorry

end boxes_containing_neither_l0_344


namespace solution_set_l0_895

def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

theorem solution_set : { x : ℝ | f x > 1 } = Set.Ioo (2/3) 2 :=
by
  sorry

end solution_set_l0_895


namespace inequality_solution_l0_864

theorem inequality_solution (x : ℝ) : (3 * x - 1) / (x - 2) > 0 ↔ x < 1 / 3 ∨ x > 2 :=
sorry

end inequality_solution_l0_864


namespace not_cube_of_sum_l0_85

theorem not_cube_of_sum (a b : ℕ) : ¬ ∃ (k : ℤ), a^3 + b^3 + 4 = k^3 :=
by
  sorry

end not_cube_of_sum_l0_85


namespace trigonometric_identity_l0_866

theorem trigonometric_identity (α : ℝ) (h : Real.tan (α + π / 4) = -3) :
  2 * Real.cos (2 * α) + 3 * Real.sin (2 * α) - Real.sin α ^ 2 = 2 / 5 :=
by
  sorry

end trigonometric_identity_l0_866


namespace part1_part2_l0_735

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0 }

noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0 }

-- Part (1): Prove a = 1 given A ∪ B = B
theorem part1 (a : ℝ) (h : A ∪ B a = B a) : a = 1 :=
sorry

-- Part (2): Prove the set C composed of the values of a given A ∩ B = B
def C : Set ℝ := {a | a ≤ -1 ∨ a = 1}

theorem part2 (h : ∀ a, A ∩ B a = B a ↔ a ∈ C) : forall a, A ∩ B a = B a ↔ a ∈ C :=
sorry

end part1_part2_l0_735


namespace arithmetic_sequence_sum_9_is_36_l0_965

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = r * (a n)
noncomputable def Sn (b : ℕ → ℝ) (n : ℕ) : ℝ := n * (b 1 + b n) / 2

theorem arithmetic_sequence_sum_9_is_36 (a b : ℕ → ℝ) (h_geom : geometric_sequence a) 
    (h_cond : a 4 * a 6 = 2 * a 5) (h_b5 : b 5 = 2 * a 5) : Sn b 9 = 36 :=
by
  sorry

end arithmetic_sequence_sum_9_is_36_l0_965


namespace prove_correct_y_l0_906

noncomputable def find_larger_y (x y : ℕ) : Prop :=
  y - x = 1365 ∧ y = 6 * x + 15

noncomputable def correct_y : ℕ := 1635

theorem prove_correct_y (x y : ℕ) (h : find_larger_y x y) : y = correct_y :=
by
  sorry

end prove_correct_y_l0_906


namespace expression_value_at_2_l0_515

theorem expression_value_at_2 : (2^2 - 3 * 2 + 2) = 0 :=
by
  sorry

end expression_value_at_2_l0_515


namespace largest_angle_measure_l0_597

theorem largest_angle_measure (v : ℝ) (h : v > 3/2) :
  ∃ θ, θ = Real.arccos ((4 * v - 4) / (2 * Real.sqrt ((2 * v - 3) * (4 * v - 4)))) ∧
       θ = π - θ ∧
       θ = Real.arccos ((2 * v - 3) / (2 * Real.sqrt ((2 * v + 3) * (4 * v - 4)))) := 
sorry

end largest_angle_measure_l0_597


namespace nelly_payment_is_correct_l0_439

-- Given definitions and conditions
def joes_bid : ℕ := 160000
def additional_amount : ℕ := 2000

-- Nelly's total payment
def nellys_payment : ℕ := (3 * joes_bid) + additional_amount

-- The proof statement we need to prove that Nelly's payment equals 482000 dollars
theorem nelly_payment_is_correct : nellys_payment = 482000 :=
by
  -- This is a placeholder for the actual proof.
  -- You can fill in the formal proof here.
  sorry

end nelly_payment_is_correct_l0_439


namespace smallest_positive_integer_satisfying_conditions_l0_774

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (x : ℕ),
    x % 4 = 1 ∧
    x % 5 = 2 ∧
    x % 7 = 3 ∧
    ∀ y : ℕ, (y % 4 = 1 ∧ y % 5 = 2 ∧ y % 7 = 3) → y ≥ x ∧ x = 93 :=
by
  sorry

end smallest_positive_integer_satisfying_conditions_l0_774


namespace avg_tickets_per_member_is_66_l0_279

-- Definitions based on the problem's conditions
def avg_female_tickets : ℕ := 70
def male_to_female_ratio : ℕ := 2
def avg_male_tickets : ℕ := 58

-- Let the number of male members be M and number of female members be F
variables (M : ℕ) (F : ℕ)
def num_female_members : ℕ := male_to_female_ratio * M

-- Total tickets sold by males
def total_male_tickets : ℕ := avg_male_tickets * M

-- Total tickets sold by females
def total_female_tickets : ℕ := avg_female_tickets * num_female_members M

-- Total tickets sold by all members
def total_tickets_sold : ℕ := total_male_tickets M + total_female_tickets M

-- Total number of members
def total_members : ℕ := M + num_female_members M

-- Statement to prove: the average number of tickets sold per member is 66
theorem avg_tickets_per_member_is_66 : total_tickets_sold M / total_members M = 66 :=
by 
  sorry

end avg_tickets_per_member_is_66_l0_279


namespace sam_found_pennies_l0_29

-- Define the function that computes the number of pennies Sam found given the initial and current amounts of pennies
def find_pennies (initial_pennies current_pennies : Nat) : Nat :=
  current_pennies - initial_pennies

-- Define the main proof problem
theorem sam_found_pennies : find_pennies 98 191 = 93 := by
  -- Proof steps would go here
  sorry

end sam_found_pennies_l0_29


namespace total_games_played_in_league_l0_696

theorem total_games_played_in_league (n : ℕ) (k : ℕ) (games_per_team : ℕ) 
  (h1 : n = 10) 
  (h2 : k = 4) 
  (h3 : games_per_team = n - 1) 
  : (k * (n * games_per_team) / 2) = 180 :=
by
  -- Definitions and transformations go here
  sorry

end total_games_played_in_league_l0_696


namespace average_weight_of_all_children_l0_123

theorem average_weight_of_all_children 
    (boys_weight_avg : ℕ)
    (number_of_boys : ℕ)
    (girls_weight_avg : ℕ)
    (number_of_girls : ℕ)
    (tall_boy_weight : ℕ)
    (ht1 : boys_weight_avg = 155)
    (ht2 : number_of_boys = 8)
    (ht3 : girls_weight_avg = 130)
    (ht4 : number_of_girls = 6)
    (ht5 : tall_boy_weight = 175)
    : (boys_weight_avg * (number_of_boys - 1) + tall_boy_weight + girls_weight_avg * number_of_girls) / (number_of_boys + number_of_girls) = 146 :=
by
  sorry

end average_weight_of_all_children_l0_123


namespace dalton_needs_more_money_l0_377

-- Definitions based on the conditions
def jumpRopeCost : ℕ := 7
def boardGameCost : ℕ := 12
def ballCost : ℕ := 4
def savedAllowance : ℕ := 6
def moneyFromUncle : ℕ := 13

-- Computation of how much more money is needed
theorem dalton_needs_more_money : 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  totalCost - totalMoney = 4 := 
by 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  have h1 : totalCost = 23 := by rfl
  have h2 : totalMoney = 19 := by rfl
  calc
    totalCost - totalMoney = 23 - 19 := by rw [h1, h2]
    _ = 4 := by rfl

end dalton_needs_more_money_l0_377


namespace find_m_l0_493

variables (a : ℕ → ℝ) (r : ℝ) (m : ℕ)

-- Define the conditions of the problem
def exponential_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

def condition_1 (a : ℕ → ℝ) (r : ℝ) : Prop :=
  a 5 * a 6 + a 4 * a 7 = 18

def condition_2 (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a 1 * a m = 9

-- The theorem to prove based on the given conditions
theorem find_m
  (h_exp : exponential_sequence a r)
  (h_r_ne_1 : r ≠ 1)
  (h_cond1 : condition_1 a r)
  (h_cond2 : condition_2 a m) :
  m = 10 :=
sorry

end find_m_l0_493


namespace lemons_count_l0_836

def total_fruits (num_baskets : ℕ) (total : ℕ) : Prop := num_baskets = 5 ∧ total = 58
def basket_contents (basket : ℕ → ℕ) : Prop := 
  basket 1 = 18 ∧ -- mangoes
  basket 2 = 10 ∧ -- pears
  basket 3 = 12 ∧ -- pawpaws
  (∀ i, (i = 4 ∨ i = 5) → basket i = (basket 4 + basket 5) / 2)

theorem lemons_count (num_baskets : ℕ) (total : ℕ) (basket : ℕ → ℕ) : 
  total_fruits num_baskets total ∧ basket_contents basket → basket 5 = 9 :=
by
  sorry

end lemons_count_l0_836


namespace can_cover_101x101_with_102_cells_100_times_l0_296

theorem can_cover_101x101_with_102_cells_100_times :
  ∃ f : Fin 100 → Fin 101 → Fin 101 → Bool,
  (∀ i j : Fin 101, (i ≠ 100 ∨ j ≠ 100) → ∃ t : Fin 100, 
    f t i j = true) :=
sorry

end can_cover_101x101_with_102_cells_100_times_l0_296


namespace intersection_of_prime_and_even_is_two_l0_55

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  ∃ k : ℤ, n = 2 * k

theorem intersection_of_prime_and_even_is_two :
  {n : ℕ | is_prime n} ∩ {n : ℕ | is_even n} = {2} :=
by
  sorry

end intersection_of_prime_and_even_is_two_l0_55


namespace solve_system_eq_l0_462

theorem solve_system_eq (x y z t : ℕ) : 
  ((x^2 + t^2) * (z^2 + y^2) = 50) ↔
    (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
    (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
    (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
    (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
by 
  sorry

end solve_system_eq_l0_462


namespace egg_count_l0_482

theorem egg_count :
  ∃ x : ℕ, 
    (∀ e1 e10 e100 : ℤ, 
      (e1 = 1 ∨ e1 = -1) →
      (e10 = 10 ∨ e10 = -10) →
      (e100 = 100 ∨ e100 = -100) →
      7 * x + e1 + e10 + e100 = 3162) → 
    x = 439 :=
by 
  sorry

end egg_count_l0_482


namespace find_initial_days_provisions_last_l0_992

def initial_days_provisions_last (initial_men reinforcements days_after_reinforcement : ℕ) (x : ℕ) : Prop :=
  initial_men * (x - 15) = (initial_men + reinforcements) * days_after_reinforcement

theorem find_initial_days_provisions_last
  (initial_men reinforcements days_after_reinforcement x : ℕ)
  (h1 : initial_men = 2000)
  (h2 : reinforcements = 1900)
  (h3 : days_after_reinforcement = 20)
  (h4 : initial_days_provisions_last initial_men reinforcements days_after_reinforcement x) :
  x = 54 :=
by
  sorry


end find_initial_days_provisions_last_l0_992


namespace price_per_liter_after_discount_l0_177

-- Define the initial conditions
def num_bottles : ℕ := 6
def liters_per_bottle : ℝ := 2
def original_total_cost : ℝ := 15
def discounted_total_cost : ℝ := 12

-- Calculate the total number of liters
def total_liters : ℝ := num_bottles * liters_per_bottle

-- Define the expected price per liter after discount
def expected_price_per_liter : ℝ := 1

-- Lean query to verify the expected price per liter
theorem price_per_liter_after_discount : (discounted_total_cost / total_liters) = expected_price_per_liter := by
  sorry

end price_per_liter_after_discount_l0_177


namespace quotient_unchanged_l0_892

-- Define the variables
variables (a b k : ℝ)

-- Condition: k ≠ 0
theorem quotient_unchanged (h : k ≠ 0) : (a * k) / (b * k) = a / b := by
  sorry

end quotient_unchanged_l0_892


namespace jane_earnings_l0_676

def age_of_child (jane_start_age : ℕ) (child_factor : ℕ) : ℕ :=
  jane_start_age / child_factor

def babysit_rate (age : ℕ) : ℕ :=
  if age < 2 then 5
  else if age <= 5 then 7
  else 8

def amount_earned (hours rate : ℕ) : ℕ := 
  hours * rate

def total_earnings (earnings : List ℕ) : ℕ :=
  earnings.foldl (·+·) 0

theorem jane_earnings
  (jane_start_age : ℕ := 18)
  (child_A_hours : ℕ := 50)
  (child_B_hours : ℕ := 90)
  (child_C_hours : ℕ := 130)
  (child_D_hours : ℕ := 70) :
  let child_A_age := age_of_child jane_start_age 2
  let child_B_age := child_A_age - 2
  let child_C_age := child_B_age + 3
  let child_D_age := child_C_age
  let earnings_A := amount_earned child_A_hours (babysit_rate child_A_age)
  let earnings_B := amount_earned child_B_hours (babysit_rate child_B_age)
  let earnings_C := amount_earned child_C_hours (babysit_rate child_C_age)
  let earnings_D := amount_earned child_D_hours (babysit_rate child_D_age)
  total_earnings [earnings_A, earnings_B, earnings_C, earnings_D] = 2720 :=
by
  sorry

end jane_earnings_l0_676


namespace lines_passing_through_neg1_0_l0_266

theorem lines_passing_through_neg1_0 (k : ℝ) :
  ∀ x y : ℝ, (y = k * (x + 1)) ↔ (x = -1 → y = 0 ∧ k ≠ 0) :=
by
  sorry

end lines_passing_through_neg1_0_l0_266


namespace haircuts_away_from_next_free_l0_275

def free_haircut (total_paid : ℕ) : ℕ := total_paid / 14

theorem haircuts_away_from_next_free (total_haircuts : ℕ) (free_haircuts : ℕ) (haircuts_per_free : ℕ) :
  total_haircuts = 79 → free_haircuts = 5 → haircuts_per_free = 14 → 
  (haircuts_per_free - (total_haircuts - free_haircuts)) % haircuts_per_free = 10 :=
by
  intros h1 h2 h3
  sorry

end haircuts_away_from_next_free_l0_275


namespace find_m_l0_328

-- Definitions from conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := x^2 + m * y^2 = 1
def major_axis_twice_minor_axis (a b : ℝ) : Prop := a = 2 * b

-- Main statement
theorem find_m (m : ℝ) (h1 : ellipse_eq 0 0 m) (h2 : 0 < m) (h3 : 0 < m ∧ m < 1) :
  m = 1 / 4 :=
by
  sorry

end find_m_l0_328


namespace car_rental_cost_per_mile_l0_489

theorem car_rental_cost_per_mile
    (daily_rental_cost : ℕ)
    (daily_budget : ℕ)
    (miles_limit : ℕ)
    (cost_per_mile : ℕ) :
    daily_rental_cost = 30 →
    daily_budget = 76 →
    miles_limit = 200 →
    cost_per_mile = (daily_budget - daily_rental_cost) * 100 / miles_limit →
    cost_per_mile = 23 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end car_rental_cost_per_mile_l0_489


namespace monthly_average_growth_rate_eq_l0_678

theorem monthly_average_growth_rate_eq (x : ℝ) :
  16 * (1 + x)^2 = 25 :=
sorry

end monthly_average_growth_rate_eq_l0_678


namespace project_completion_equation_l0_281

variables (x : ℕ)

-- Project completion conditions
def person_A_time : ℕ := 12
def person_B_time : ℕ := 8
def A_initial_work_days : ℕ := 3

-- Work done by Person A when working alone for 3 days
def work_A_initial := (A_initial_work_days:ℚ) / person_A_time

-- Work done by Person A and B after the initial 3 days until completion
def combined_work_remaining := 
  (λ x:ℕ => ((x - A_initial_work_days):ℚ) * (1/person_A_time + 1/person_B_time))

-- The equation representing the total work done equals 1
theorem project_completion_equation (x : ℕ) : 
  (x:ℚ) / person_A_time + (x - A_initial_work_days:ℚ) / person_B_time = 1 :=
sorry

end project_completion_equation_l0_281


namespace total_cookies_l0_441

-- Definitions from conditions
def cookies_per_guest : ℕ := 2
def number_of_guests : ℕ := 5

-- Theorem statement that needs to be proved
theorem total_cookies : cookies_per_guest * number_of_guests = 10 := by
  -- We skip the proof since only the statement is required
  sorry

end total_cookies_l0_441


namespace cube_has_12_edges_l0_719

-- Definition of the number of edges in a cube
def number_of_edges_of_cube : Nat := 12

-- The theorem that asserts the cube has 12 edges
theorem cube_has_12_edges : number_of_edges_of_cube = 12 := by
  -- proof to be filled later
  sorry

end cube_has_12_edges_l0_719


namespace problem_statement_l0_117

def a : ℤ := 2020
def b : ℤ := 2022

theorem problem_statement : b^3 - a * b^2 - a^2 * b + a^3 = 16168 := by
  sorry

end problem_statement_l0_117


namespace pipe_filling_problem_l0_71

theorem pipe_filling_problem (x : ℝ) (h : (2 / 15) * x + (1 / 20) * (10 - x) = 1) : x = 6 :=
sorry

end pipe_filling_problem_l0_71


namespace snacks_in_3h40m_l0_440

def minutes_in_hours (hours : ℕ) : ℕ := hours * 60

def snacks_in_time (total_minutes : ℕ) (snack_interval : ℕ) : ℕ := total_minutes / snack_interval

theorem snacks_in_3h40m : snacks_in_time (minutes_in_hours 3 + 40) 20 = 11 :=
by
  sorry

end snacks_in_3h40m_l0_440


namespace sqrt_ab_is_integer_l0_708

theorem sqrt_ab_is_integer
  (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
  (h_eq : a * (b^2 + n^2) = b * (a^2 + n^2)) :
  ∃ k : ℕ, k * k = a * b :=
by
  sorry

end sqrt_ab_is_integer_l0_708


namespace math_problem_l0_122

theorem math_problem : 12 - (- 18) + (- 7) - 15 = 8 :=
by
  sorry

end math_problem_l0_122


namespace minute_hand_coincides_hour_hand_11_times_l0_400

noncomputable def number_of_coincidences : ℕ := 11

theorem minute_hand_coincides_hour_hand_11_times :
  ∀ (t : ℝ), (0 < t ∧ t < 12) → ∃(n : ℕ), (1 ≤ n ∧ n ≤ 11) ∧ t = (n * 1 + n * (5 / 11)) :=
sorry

end minute_hand_coincides_hour_hand_11_times_l0_400


namespace angle_remains_unchanged_l0_569

-- Definition of magnification condition (though it does not affect angle in mathematics, we state it as given)
def magnifying_glass (magnification : ℝ) (initial_angle : ℝ) : ℝ := 
  initial_angle  -- Magnification does not change the angle in this context.

-- Given condition
def initial_angle : ℝ := 30

-- Theorem we want to prove
theorem angle_remains_unchanged (magnification : ℝ) (h_magnify : magnification = 100) :
  magnifying_glass magnification initial_angle = initial_angle :=
by
  sorry

end angle_remains_unchanged_l0_569


namespace geometric_sequence_ratio_l0_795

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def condition_1 (n : ℕ) : Prop := S n = 2 * a n - 2

theorem geometric_sequence_ratio (h : ∀ n, condition_1 a S n) : (a 8 / a 6 = 4) :=
sorry

end geometric_sequence_ratio_l0_795


namespace committee_of_4_from_10_eq_210_l0_448

theorem committee_of_4_from_10_eq_210 :
  (Nat.choose 10 4) = 210 :=
by
  sorry

end committee_of_4_from_10_eq_210_l0_448


namespace call_processing_ratio_l0_199

variables (A B C : ℝ)
variable (total_calls : ℝ)
variable (calls_processed_by_A_per_member calls_processed_by_B_per_member : ℝ)

-- Given conditions
def team_A_agents_ratio : Prop := A = (5 / 8) * B
def team_B_calls_ratio : Prop := calls_processed_by_B_per_member * B = (4 / 7) * total_calls
def team_A_calls_ratio : Prop := calls_processed_by_A_per_member * A = (3 / 7) * total_calls

-- Proving the ratio of calls processed by each member
theorem call_processing_ratio
    (hA : team_A_agents_ratio A B)
    (hB_calls : team_B_calls_ratio B total_calls calls_processed_by_B_per_member)
    (hA_calls : team_A_calls_ratio A total_calls calls_processed_by_A_per_member) :
  calls_processed_by_A_per_member / calls_processed_by_B_per_member = 6 / 5 :=
by
  sorry

end call_processing_ratio_l0_199


namespace obtuse_right_triangle_cannot_exist_l0_851

-- Definitions of various types of triangles

def is_acute (θ : ℕ) : Prop := θ < 90
def is_right (θ : ℕ) : Prop := θ = 90
def is_obtuse (θ : ℕ) : Prop := θ > 90

def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ a = c
def is_scalene (a b c : ℕ) : Prop := ¬ (a = b) ∧ ¬ (b = c) ∧ ¬ (a = c)
def is_triangle (a b c : ℕ) : Prop := a + b + c = 180

-- Propositions for the types of triangles given in the problem

def acute_isosceles_triangle (a b : ℕ) : Prop :=
  is_triangle a a (180 - 2 * a) ∧ is_acute a ∧ is_isosceles a a (180 - 2 * a)

def isosceles_right_triangle (a : ℕ) : Prop :=
  is_triangle a a 90 ∧ is_right 90 ∧ is_isosceles a a 90

def obtuse_right_triangle (a b : ℕ) : Prop :=
  is_triangle a 90 (180 - 90 - a) ∧ is_right 90 ∧ is_obtuse (180 - 90 - a)

def scalene_right_triangle (a b : ℕ) : Prop :=
  is_triangle a b 90 ∧ is_right 90 ∧ is_scalene a b 90

def scalene_obtuse_triangle (a b : ℕ) : Prop :=
  is_triangle a b (180 - a - b) ∧ is_obtuse (180 - a - b) ∧ is_scalene a b (180 - a - b)

-- The final theorem stating that obtuse right triangle cannot exist

theorem obtuse_right_triangle_cannot_exist (a b : ℕ) :
  ¬ exists (a b : ℕ), obtuse_right_triangle a b :=
by
  sorry

end obtuse_right_triangle_cannot_exist_l0_851


namespace reserved_fraction_l0_701

variable (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ)
variable (f : ℚ)

def mrSalazarFractionReserved (initial_oranges : ℕ) (sold_fraction : ℚ) (rotten_oranges : ℕ) (leftover_oranges : ℕ) : ℚ :=
  1 - (leftover_oranges + rotten_oranges) * sold_fraction / initial_oranges

theorem reserved_fraction (h1 : initial_oranges = 84) (h2 : sold_fraction = 3 / 7) (h3 : rotten_oranges = 4) (h4 : leftover_oranges = 32) :
  (mrSalazarFractionReserved initial_oranges sold_fraction rotten_oranges leftover_oranges) = 1 / 4 :=
  by
    -- Proof is omitted
    sorry

end reserved_fraction_l0_701


namespace intersection_expression_value_l0_667

theorem intersection_expression_value
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : x₁ * y₁ = 1)
  (h₂ : x₂ * y₂ = 1)
  (h₃ : x₁ = -x₂)
  (h₄ : y₁ = -y₂) :
  x₁ * y₂ + x₂ * y₁ = -2 :=
by
  sorry

end intersection_expression_value_l0_667


namespace projected_percent_increase_l0_352

theorem projected_percent_increase (R : ℝ) (p : ℝ) 
  (h1 : 0.7 * R = R * 0.7) 
  (h2 : 0.7 * R = 0.5 * (R + p * R)) : 
  p = 0.4 :=
by
  sorry

end projected_percent_increase_l0_352


namespace exists_xy_for_cube_difference_l0_339

theorem exists_xy_for_cube_difference (a : ℕ) (h : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 :=
sorry

end exists_xy_for_cube_difference_l0_339


namespace no_member_of_T_is_divisible_by_4_or_5_l0_340

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4_or_5 :
  ∀ (n : ℤ), ¬ (∃ (T : ℤ), T = sum_of_squares_of_four_consecutive_integers n ∧ (T % 4 = 0 ∨ T % 5 = 0)) :=
by
  sorry

end no_member_of_T_is_divisible_by_4_or_5_l0_340


namespace vector_calc_l0_672

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Statement to prove that 2a - b = (5, 7)
theorem vector_calc : 2 • a - b = (5, 7) :=
by {
  -- Proof will be filled here
  sorry
}

end vector_calc_l0_672


namespace infinite_series_value_l0_844

noncomputable def infinite_series : ℝ :=
  ∑' n, if n ≥ 2 then (n^4 + 5 * n^2 + 8 * n + 8) / (2^(n + 1) * (n^4 + 4)) else 0

theorem infinite_series_value :
  infinite_series = 3 / 10 :=
by
  sorry

end infinite_series_value_l0_844


namespace jerry_remaining_debt_l0_608

variable (two_months_ago_payment last_month_payment total_debt : ℕ)

def remaining_debt : ℕ := total_debt - (two_months_ago_payment + last_month_payment)

theorem jerry_remaining_debt :
  two_months_ago_payment = 12 →
  last_month_payment = 12 + 3 →
  total_debt = 50 →
  remaining_debt two_months_ago_payment last_month_payment total_debt = 23 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jerry_remaining_debt_l0_608


namespace david_still_has_less_than_750_l0_291

theorem david_still_has_less_than_750 (S R : ℝ) 
  (h1 : S + R = 1500)
  (h2 : R < S) : 
  R < 750 :=
by 
  sorry

end david_still_has_less_than_750_l0_291


namespace min_ratio_ax_l0_826

theorem min_ratio_ax (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) 
: y^2 - 1 = a^2 * (x^2 - 1) → ∃ (k : ℕ), k = 2 ∧ (a = k * x) := 
sorry

end min_ratio_ax_l0_826


namespace x_plus_p_l0_175

theorem x_plus_p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 :=
by
  sorry

end x_plus_p_l0_175


namespace hadassah_additional_paintings_l0_767

noncomputable def hadassah_initial_paintings : ℕ := 12
noncomputable def hadassah_initial_hours : ℕ := 6
noncomputable def hadassah_total_hours : ℕ := 16

theorem hadassah_additional_paintings 
  (initial_paintings : ℕ)
  (initial_hours : ℕ)
  (total_hours : ℕ) :
  initial_paintings = hadassah_initial_paintings →
  initial_hours = hadassah_initial_hours →
  total_hours = hadassah_total_hours →
  let additional_hours := total_hours - initial_hours
  let painting_rate := initial_paintings / initial_hours
  let additional_paintings := painting_rate * additional_hours
  additional_paintings = 20 :=
by
  sorry

end hadassah_additional_paintings_l0_767


namespace find_number_l0_469

theorem find_number (x : ℝ) : 2.75 + 0.003 + x = 2.911 -> x = 0.158 := 
by
  intros h
  sorry

end find_number_l0_469


namespace incorrect_inequality_given_conditions_l0_809

variable {a b x y : ℝ}

theorem incorrect_inequality_given_conditions 
  (h1 : a > b) (h2 : x > y) : ¬ (|a| * x > |a| * y) :=
sorry

end incorrect_inequality_given_conditions_l0_809


namespace scorpion_millipedes_needed_l0_468

theorem scorpion_millipedes_needed 
  (total_segments_required : ℕ)
  (eaten_millipede_1_segments : ℕ)
  (eaten_millipede_2_segments : ℕ)
  (segments_per_millipede : ℕ)
  (n_millipedes_needed : ℕ)
  (total_segments : total_segments_required = 800) 
  (segments_1 : eaten_millipede_1_segments = 60)
  (segments_2 : eaten_millipede_2_segments = 2 * 60)
  (needed_segments_calculation : 800 - (60 + 2 * (2 * 60)) = n_millipedes_needed * 50) 
  : n_millipedes_needed = 10 :=
by
  sorry

end scorpion_millipedes_needed_l0_468


namespace remaining_oil_quantity_check_remaining_oil_quantity_l0_261

def initial_oil_quantity : Real := 40
def outflow_rate : Real := 0.2

theorem remaining_oil_quantity (t : Real) : Real :=
  initial_oil_quantity - outflow_rate * t

theorem check_remaining_oil_quantity (t : Real) : remaining_oil_quantity t = 40 - 0.2 * t := 
by 
  sorry

end remaining_oil_quantity_check_remaining_oil_quantity_l0_261


namespace square_field_area_l0_293

/-- 
  Statement: Prove that the area of the square field is 69696 square meters 
  given that the wire goes around the square field 15 times and the total 
  length of the wire is 15840 meters.
-/
theorem square_field_area (rounds : ℕ) (total_length : ℕ) (area : ℕ) 
  (h1 : rounds = 15) (h2 : total_length = 15840) : 
  area = 69696 := 
by 
  sorry

end square_field_area_l0_293


namespace pascal_triangle_count_30_rows_l0_884

def pascal_row_count (n : Nat) := n + 1

def sum_arithmetic_sequence (a₁ an n : Nat) : Nat :=
  n * (a₁ + an) / 2

theorem pascal_triangle_count_30_rows :
  sum_arithmetic_sequence (pascal_row_count 0) (pascal_row_count 29) 30 = 465 :=
by
  sorry

end pascal_triangle_count_30_rows_l0_884


namespace simplify_expression_l0_362

variable (a b : ℝ)

theorem simplify_expression :
  (a^3 - b^3) / (a * b) - (ab - b^2) / (ab - a^3) = (a^2 + ab + b^2) / b :=
by
  sorry

end simplify_expression_l0_362


namespace rational_sum_is_negative_then_at_most_one_positive_l0_154

theorem rational_sum_is_negative_then_at_most_one_positive (a b : ℚ) (h : a + b < 0) :
  (a > 0 ∧ b ≤ 0) ∨ (a ≤ 0 ∧ b > 0) ∨ (a ≤ 0 ∧ b ≤ 0) :=
by
  sorry

end rational_sum_is_negative_then_at_most_one_positive_l0_154


namespace quadratic_function_expression_l0_512

theorem quadratic_function_expression : 
  ∃ (a : ℝ), (a ≠ 0) ∧ (∀ x : ℝ, x = -1 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ x : ℝ, x = 2 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ y : ℝ, ∃ x : ℝ, x = 0 ∧ y = -2 → y = a * (x + 1) * (x - 2)) 
  → (∀ x : ℝ, ∃ y : ℝ, y = x^2 - x - 2) := 
sorry

end quadratic_function_expression_l0_512


namespace width_of_first_sheet_l0_240

theorem width_of_first_sheet (w : ℝ) (h : 2 * (w * 17) = 2 * (8.5 * 11) + 100) : w = 287 / 34 :=
by
  sorry

end width_of_first_sheet_l0_240


namespace range_of_a_l0_556

variables {a x : ℝ}

def P (a : ℝ) : Prop := ∀ x, ¬ (x^2 - (a + 1) * x + 1 ≤ 0)

def Q (a : ℝ) : Prop := ∀ x, |x - 1| ≥ a + 2

theorem range_of_a (a : ℝ) : 
  (¬ P a ∧ ¬ Q a) → a ≥ 1 :=
by
  sorry

end range_of_a_l0_556


namespace trainB_speed_l0_710

variable (v : ℕ)

def trainA_speed : ℕ := 30
def time_gap : ℕ := 2
def distance_overtake : ℕ := 360

theorem trainB_speed (h :  v > trainA_speed) : v = 42 :=
by
  sorry

end trainB_speed_l0_710


namespace boxes_needed_to_complete_flooring_l0_963

-- Definitions of given conditions
def length_of_living_room : ℕ := 16
def width_of_living_room : ℕ := 20
def sq_ft_per_box : ℕ := 10
def sq_ft_already_covered : ℕ := 250

-- Statement to prove
theorem boxes_needed_to_complete_flooring : 
  (length_of_living_room * width_of_living_room - sq_ft_already_covered) / sq_ft_per_box = 7 :=
by
  sorry

end boxes_needed_to_complete_flooring_l0_963


namespace find_c_gen_formula_l0_367

noncomputable def seq (a : ℕ → ℕ) (c : ℕ) : Prop :=
a 1 = 2 ∧
(∀ n, a (n + 1) = a n + c * n) ∧
(2 + c) * (2 + c) = 2 * (2 + 3 * c)

theorem find_c (a : ℕ → ℕ) : ∃ c, seq a c :=
by
  sorry

theorem gen_formula (a : ℕ → ℕ) (c : ℕ) (h : seq a c) : (∀ n, a n = n^2 - n + 2) :=
by
  sorry

end find_c_gen_formula_l0_367


namespace parabola_directrix_eq_l0_78

def parabola_directrix (p : ℝ) : ℝ := -p

theorem parabola_directrix_eq (x y p : ℝ) (h : y ^ 2 = 8 * x) (hp : 2 * p = 8) : 
  parabola_directrix p = -2 :=
by
  sorry

end parabola_directrix_eq_l0_78


namespace problem_statement_l0_660

theorem problem_statement (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (∃ φn : ℕ, φn = Nat.totient n ∧ p ∣ φn ∧ (∀ a : ℕ, Nat.gcd a n = 1 → n ∣ a ^ (φn / p) - 1)) ↔ 
  (∃ q1 q2 : ℕ, q1 ≠ q2 ∧ Nat.Prime q1 ∧ Nat.Prime q2 ∧ q1 ≡ 1 [MOD p] ∧ q2 ≡ 1 [MOD p] ∧ q1 ∣ n ∧ q2 ∣ n ∨ 
  (∃ q : ℕ, Nat.Prime q ∧ q ≡ 1 [MOD p] ∧ q ∣ n ∧ p ^ 2 ∣ n)) :=
by {
  sorry
}

end problem_statement_l0_660


namespace monthly_income_of_P_l0_271

-- Define variables and assumptions
variables (P Q R : ℝ)
axiom avg_P_Q : (P + Q) / 2 = 5050
axiom avg_Q_R : (Q + R) / 2 = 6250
axiom avg_P_R : (P + R) / 2 = 5200

-- Prove that the monthly income of P is 4000
theorem monthly_income_of_P : P = 4000 :=
by
  sorry

end monthly_income_of_P_l0_271


namespace sum_of_angles_is_360_l0_977

-- Let's define the specific angles within our geometric figure
variables (A B C D F G : ℝ)

-- Define a condition stating that these angles form a quadrilateral inside a geometric figure, such that their sum is valid
def angles_form_quadrilateral (A B C D F G : ℝ) : Prop :=
  (A + B + C + D + F + G = 360)

-- Finally, we declare the theorem we want to prove
theorem sum_of_angles_is_360 (A B C D F G : ℝ) (h : angles_form_quadrilateral A B C D F G) : A + B + C + D + F + G = 360 :=
  h


end sum_of_angles_is_360_l0_977


namespace find_single_digit_l0_455

def isSingleDigit (n : ℕ) : Prop := n < 10

def repeatedDigitNumber (A : ℕ) : ℕ := 10 * A + A 

theorem find_single_digit (A : ℕ) (h1 : isSingleDigit A) (h2 : repeatedDigitNumber A + repeatedDigitNumber A = 132) : A = 6 :=
by
  sorry

end find_single_digit_l0_455


namespace problem1_problem2_l0_958

theorem problem1 : 27^((2:ℝ)/(3:ℝ)) - 2^(Real.logb 2 3) * Real.logb 2 (1/8) = 18 := 
by
  sorry -- proof omitted

theorem problem2 : 1/(Real.sqrt 5 - 2) - (Real.sqrt 5 + 2)^0 - Real.sqrt ((2 - Real.sqrt 5)^2) = 2*(Real.sqrt 5 - 1) := 
by
  sorry -- proof omitted

end problem1_problem2_l0_958


namespace find_number_l0_486

theorem find_number (x : ℤ) (h : 300 + 8 * x = 340) : x = 5 := by
  sorry

end find_number_l0_486


namespace solve_eq_l0_934

theorem solve_eq {x y z : ℕ} :
  2^x + 3^y - 7 = z! ↔ (x = 2 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 4) :=
by
  sorry -- Proof should be provided here

end solve_eq_l0_934


namespace max_xy_l0_537

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy <= 81 :=
by {
  sorry
}

end max_xy_l0_537


namespace no_real_solution_to_system_l0_944

theorem no_real_solution_to_system :
  ∀ (x y z : ℝ), (x + y - 2 - 4 * x * y = 0) ∧
                 (y + z - 2 - 4 * y * z = 0) ∧
                 (z + x - 2 - 4 * z * x = 0) → false := 
by 
    intros x y z h
    rcases h with ⟨h1, h2, h3⟩
    -- Here would be the proof steps, which are omitted.
    sorry

end no_real_solution_to_system_l0_944


namespace unique_friendly_determination_l0_518

def is_friendly (a b : ℕ → ℕ) : Prop :=
∀ n : ℕ, ∃ i j : ℕ, n = a i * b j ∧ ∀ (k l : ℕ), n = a k * b l → (i = k ∧ j = l)

theorem unique_friendly_determination {a b c : ℕ → ℕ} 
  (h_friend_a_b : is_friendly a b) 
  (h_friend_a_c : is_friendly a c) :
  b = c :=
sorry

end unique_friendly_determination_l0_518


namespace find_f_of_3_l0_46

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_of_3 :
  (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intros h
  have : f 3 = 1 / 2 := sorry
  exact this

end find_f_of_3_l0_46


namespace rounding_example_l0_788

theorem rounding_example (x : ℝ) (h : x = 8899.50241201) : round x = 8900 :=
by
  sorry

end rounding_example_l0_788


namespace students_in_class_l0_898

theorem students_in_class (n : ℕ) (T : ℕ) 
  (average_age_students : T = 16 * n)
  (staff_age : ℕ)
  (increased_average_age : (T + staff_age) / (n + 1) = 17)
  (staff_age_val : staff_age = 49) : n = 32 := 
by
  sorry

end students_in_class_l0_898


namespace greatest_multiple_of_5_and_7_less_than_800_l0_153

theorem greatest_multiple_of_5_and_7_less_than_800 : 
    ∀ n : ℕ, (n < 800 ∧ 35 ∣ n) → n ≤ 770 := 
by
  -- Proof steps go here
  sorry

end greatest_multiple_of_5_and_7_less_than_800_l0_153


namespace total_classic_books_l0_343

-- Definitions for the conditions
def authors := 6
def books_per_author := 33

-- Statement of the math proof problem
theorem total_classic_books : authors * books_per_author = 198 := by
  sorry  -- Proof to be filled in

end total_classic_books_l0_343


namespace delta_delta_delta_45_l0_19

def delta (P : ℚ) : ℚ := (2 / 3) * P + 2

theorem delta_delta_delta_45 :
  delta (delta (delta 45)) = 158 / 9 :=
by sorry

end delta_delta_delta_45_l0_19


namespace vincent_total_cost_l0_209

theorem vincent_total_cost :
  let day1_packs := 15
  let day1_pack_cost := 2.50
  let discount_percent := 0.10
  let day2_packs := 25
  let day2_pack_cost := 3.00
  let tax_percent := 0.05
  let day1_total_cost_before_discount := day1_packs * day1_pack_cost
  let day1_discount_amount := discount_percent * day1_total_cost_before_discount
  let day1_total_cost_after_discount := day1_total_cost_before_discount - day1_discount_amount
  let day2_total_cost_before_tax := day2_packs * day2_pack_cost
  let day2_tax_amount := tax_percent * day2_total_cost_before_tax
  let day2_total_cost_after_tax := day2_total_cost_before_tax + day2_tax_amount
  let total_cost := day1_total_cost_after_discount + day2_total_cost_after_tax
  total_cost = 112.50 :=
by 
  -- Mathlib can be used for floating point calculations, if needed
  -- For the purposes of this example, we assume calculations are correct.
  sorry

end vincent_total_cost_l0_209


namespace add_ab_values_l0_747

theorem add_ab_values (a b : ℝ) (h1 : ∀ x : ℝ, (x^2 + 4*x + 3) = (a*x + b)^2 + 4*(a*x + b) + 3) :
  a + b = -8 ∨ a + b = 4 :=
  by sorry

end add_ab_values_l0_747


namespace complement_of_union_eq_l0_721

-- Define the universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define the subset A
def A : Set ℤ := {-1, 0, 1}

-- Define the subset B
def B : Set ℤ := {0, 1, 2, 3}

-- Define the union of A and B
def A_union_B : Set ℤ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℤ := U \ A_union_B

-- State the theorem to be proved
theorem complement_of_union_eq {U A B : Set ℤ} :
  U = {-1, 0, 1, 2, 3, 4} →
  A = {-1, 0, 1} →
  B = {0, 1, 2, 3} →
  complement_U_A_union_B = {4} :=
by
  intros hU hA hB
  sorry

end complement_of_union_eq_l0_721


namespace find_mn_l0_286

theorem find_mn
  (AB BC : ℝ) -- Lengths of AB and BC
  (m n : ℝ)   -- Coefficients of the quadratic equation
  (h_perimeter : 2 * (AB + BC) = 12)
  (h_area : AB * BC = 5)
  (h_roots_sum : AB + BC = -m)
  (h_roots_product : AB * BC = n) :
  m * n = -30 :=
by
  sorry

end find_mn_l0_286


namespace ali_less_nada_l0_499

variable (Ali Nada John : ℕ)

theorem ali_less_nada
  (h_total : Ali + Nada + John = 67)
  (h_john_nada : John = 4 * Nada)
  (h_john : John = 48) :
  Nada - Ali = 5 :=
by
  sorry

end ali_less_nada_l0_499


namespace milton_sold_total_pies_l0_978

-- Definitions for the given conditions.
def apple_pie_slices : ℕ := 8
def peach_pie_slices : ℕ := 6
def cherry_pie_slices : ℕ := 10

def apple_slices_ordered : ℕ := 88
def peach_slices_ordered : ℕ := 78
def cherry_slices_ordered : ℕ := 45

-- Function to compute the number of pies, rounding up as necessary
noncomputable def pies_sold (ordered : ℕ) (slices : ℕ) : ℕ :=
  (ordered + slices - 1) / slices  -- Using integer division to round up

-- The theorem asserting the total number of pies sold 
theorem milton_sold_total_pies : 
  pies_sold apple_slices_ordered apple_pie_slices +
  pies_sold peach_slices_ordered peach_pie_slices +
  pies_sold cherry_slices_ordered cherry_pie_slices = 29 :=
by sorry

end milton_sold_total_pies_l0_978


namespace possible_values_of_expression_l0_598

theorem possible_values_of_expression (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  ∃ (vals : Finset ℤ), vals = {6, 2, 0, -2, -6} ∧
  (∃ val ∈ vals, val = (if p > 0 then 1 else -1) + 
                         (if q > 0 then 1 else -1) + 
                         (if r > 0 then 1 else -1) + 
                         (if s > 0 then 1 else -1) + 
                         (if (p * q * r) > 0 then 1 else -1) + 
                         (if (p * r * s) > 0 then 1 else -1)) :=
by
  sorry

end possible_values_of_expression_l0_598


namespace base8_subtraction_correct_l0_40

theorem base8_subtraction_correct :
  ∀ (a b : ℕ) (h1 : a = 7534) (h2 : b = 3267),
      (a - b) % 8 = 4243 % 8 := by
  sorry

end base8_subtraction_correct_l0_40


namespace siblings_of_kevin_l0_185

-- Define traits of each child
structure Child where
  eye_color : String
  hair_color : String

def Oliver : Child := ⟨"Green", "Red"⟩
def Kevin : Child := ⟨"Grey", "Brown"⟩
def Lily : Child := ⟨"Grey", "Red"⟩
def Emma : Child := ⟨"Green", "Brown"⟩
def Noah : Child := ⟨"Green", "Red"⟩
def Mia : Child := ⟨"Green", "Brown"⟩

-- Define the condition that siblings must share at least one trait
def share_at_least_one_trait (c1 c2 : Child) : Prop :=
  c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color

-- Prove that Emma and Mia are Kevin's siblings
theorem siblings_of_kevin : share_at_least_one_trait Kevin Emma ∧ share_at_least_one_trait Kevin Mia ∧ share_at_least_one_trait Emma Mia :=
  sorry

end siblings_of_kevin_l0_185


namespace volume_tetrahedron_constant_l0_326

theorem volume_tetrahedron_constant (m n h : ℝ) (ϕ : ℝ) :
  ∃ V : ℝ, V = (1 / 6) * m * n * h * Real.sin ϕ :=
by
  sorry

end volume_tetrahedron_constant_l0_326


namespace value_of_x2_plus_9y2_l0_953

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l0_953


namespace valid_passwords_l0_968

theorem valid_passwords (total_passwords restricted_passwords : Nat) 
  (h_total : total_passwords = 10^4)
  (h_restricted : restricted_passwords = 8) : 
  total_passwords - restricted_passwords = 9992 := by
  sorry

end valid_passwords_l0_968


namespace weight_of_B_l0_793

theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 31 :=
sorry

end weight_of_B_l0_793


namespace remainder_when_divided_by_x_minus_2_l0_234

def polynomial (x : ℝ) := x^5 + 2 * x^3 - x + 4

theorem remainder_when_divided_by_x_minus_2 :
  polynomial 2 = 50 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l0_234


namespace parallelogram_area_correct_l0_671

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_correct (base height : ℝ) (h_base : base = 30) (h_height : height = 12) : parallelogram_area base height = 360 :=
by
  rw [h_base, h_height]
  simp [parallelogram_area]
  sorry

end parallelogram_area_correct_l0_671


namespace piravena_flight_cost_l0_35

noncomputable def cost_of_flight (distance_km : ℕ) (booking_fee : ℕ) (rate_per_km : ℕ) : ℕ :=
  booking_fee + (distance_km * rate_per_km / 100)

def check_cost_of_flight : Prop :=
  let distance_bc := 1000
  let booking_fee := 100
  let rate_per_km := 10
  cost_of_flight distance_bc booking_fee rate_per_km = 200

theorem piravena_flight_cost : check_cost_of_flight := 
by {
  sorry
}

end piravena_flight_cost_l0_35


namespace average_marks_increase_ratio_l0_938

theorem average_marks_increase_ratio
  (T : ℕ)  -- The correct total marks of the class
  (n : ℕ)  -- The number of pupils in the class
  (h_n : n = 16) (wrong_mark : ℕ) (correct_mark : ℕ)  -- The wrong and correct marks
  (h_wrong : wrong_mark = 73) (h_correct : correct_mark = 65) :
  (8 : ℚ) / T = (wrong_mark - correct_mark : ℚ) / n * (n / T) :=
by
  sorry

end average_marks_increase_ratio_l0_938


namespace sum_first_75_terms_arith_seq_l0_5

theorem sum_first_75_terms_arith_seq (a_1 d : ℕ) (n : ℕ) (h_a1 : a_1 = 3) (h_d : d = 4) (h_n : n = 75) : 
  (n * (2 * a_1 + (n - 1) * d)) / 2 = 11325 := 
by
  subst h_a1
  subst h_d
  subst h_n
  sorry

end sum_first_75_terms_arith_seq_l0_5


namespace train_speed_excluding_stoppages_l0_295

theorem train_speed_excluding_stoppages 
    (speed_including_stoppages : ℕ)
    (stoppage_time_per_hour : ℕ)
    (running_time_per_hour : ℚ)
    (h1 : speed_including_stoppages = 36)
    (h2 : stoppage_time_per_hour = 20)
    (h3 : running_time_per_hour = 2 / 3) :
    ∃ S : ℕ, S = 54 :=
by 
  sorry

end train_speed_excluding_stoppages_l0_295


namespace younger_by_17_l0_165

variables (A B C : ℕ)

-- Given condition
axiom age_condition : A + B = B + C + 17

-- To show
theorem younger_by_17 : A - C = 17 :=
by
  sorry

end younger_by_17_l0_165


namespace sam_initial_money_l0_363

theorem sam_initial_money :
  (9 * 7 + 16 = 79) :=
by
  sorry

end sam_initial_money_l0_363


namespace three_pow_zero_eq_one_l0_861

theorem three_pow_zero_eq_one : 3^0 = 1 :=
by {
  -- Proof would go here
  sorry
}

end three_pow_zero_eq_one_l0_861


namespace merchant_gross_profit_l0_516

-- Define the purchase price and markup rate
def purchase_price : ℝ := 42
def markup_rate : ℝ := 0.30
def discount_rate : ℝ := 0.20

-- Define the selling price equation given the purchase price and markup rate
def selling_price (S : ℝ) : Prop := S = purchase_price + markup_rate * S

-- Define the discounted selling price given the selling price and discount rate
def discounted_selling_price (S : ℝ) : ℝ := S - discount_rate * S

-- Define the gross profit as the difference between the discounted selling price and purchase price
def gross_profit (S : ℝ) : ℝ := discounted_selling_price S - purchase_price

theorem merchant_gross_profit : ∃ S : ℝ, selling_price S ∧ gross_profit S = 6 :=
by
  sorry

end merchant_gross_profit_l0_516


namespace sum_series_eq_3_over_4_l0_45

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l0_45


namespace shaded_area_calc_l0_625

theorem shaded_area_calc (r1_area r2_area overlap_area circle_area : ℝ)
  (h_r1_area : r1_area = 36)
  (h_r2_area : r2_area = 28)
  (h_overlap_area : overlap_area = 21)
  (h_circle_area : circle_area = Real.pi) : 
  (r1_area + r2_area - overlap_area - circle_area) = 64 - Real.pi :=
by
  sorry

end shaded_area_calc_l0_625


namespace group_for_2019_is_63_l0_311

def last_term_of_group (n : ℕ) : ℕ := (n * (n + 1)) / 2 + n

theorem group_for_2019_is_63 :
  ∃ n : ℕ, (2015 < 2019 ∧ 2019 ≤ 2079) :=
by
  sorry

end group_for_2019_is_63_l0_311


namespace solve_for_y_l0_259

theorem solve_for_y (y : ℕ) (h : 2^y + 8 = 4 * 2^y - 40) : y = 4 :=
by
  sorry

end solve_for_y_l0_259


namespace series_converges_to_three_fourths_l0_831

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l0_831


namespace leftover_balls_when_placing_60_in_tetrahedral_stack_l0_143

def tetrahedral_number (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) / 6

/--
  When placing 60 balls in a tetrahedral stack, the number of leftover balls is 4.
-/
theorem leftover_balls_when_placing_60_in_tetrahedral_stack :
  ∃ n, tetrahedral_number n ≤ 60 ∧ 60 - tetrahedral_number n = 4 := by
  sorry

end leftover_balls_when_placing_60_in_tetrahedral_stack_l0_143


namespace neither_sufficient_nor_necessary_l0_909

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem neither_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  ¬ ((q > 1) ↔ is_increasing_sequence a) :=
sorry

end neither_sufficient_nor_necessary_l0_909


namespace jamie_total_balls_after_buying_l0_822

theorem jamie_total_balls_after_buying (red_balls : ℕ) (blue_balls : ℕ) (yellow_balls : ℕ) (lost_red_balls : ℕ) (final_red_balls : ℕ) (total_balls : ℕ)
  (h1 : red_balls = 16)
  (h2 : blue_balls = 2 * red_balls)
  (h3 : lost_red_balls = 6)
  (h4 : final_red_balls = red_balls - lost_red_balls)
  (h5 : yellow_balls = 32)
  (h6 : total_balls = final_red_balls + blue_balls + yellow_balls) :
  total_balls = 74 := by
    sorry

end jamie_total_balls_after_buying_l0_822


namespace eiffel_tower_height_l0_415

-- Define the constants for heights and difference
def BurjKhalifa : ℝ := 830
def height_difference : ℝ := 506

-- The goal: Prove that the height of the Eiffel Tower is 324 m.
theorem eiffel_tower_height : BurjKhalifa - height_difference = 324 := 
by 
sorry

end eiffel_tower_height_l0_415


namespace find_ratio_l0_984

open Real

theorem find_ratio (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + 2 * y) / (x - 2 * y) = -4 / sqrt 7 :=
by
  sorry

end find_ratio_l0_984


namespace win_sector_area_l0_73

theorem win_sector_area (r : ℝ) (P : ℝ) (h0 : r = 8) (h1 : P = 3 / 8) :
    let area_total := Real.pi * r ^ 2
    let area_win := P * area_total
    area_win = 24 * Real.pi :=
by 
  sorry

end win_sector_area_l0_73


namespace average_students_per_bus_l0_546

-- Definitions
def total_students : ℕ := 396
def students_in_cars : ℕ := 18
def number_of_buses : ℕ := 7

-- Proof problem statement
theorem average_students_per_bus : (total_students - students_in_cars) / number_of_buses = 54 := by
  sorry

end average_students_per_bus_l0_546


namespace euclid_middle_school_math_students_l0_225

theorem euclid_middle_school_math_students
  (students_Germain : ℕ)
  (students_Newton : ℕ)
  (students_Young : ℕ)
  (students_Euler : ℕ)
  (h_Germain : students_Germain = 12)
  (h_Newton : students_Newton = 10)
  (h_Young : students_Young = 7)
  (h_Euler : students_Euler = 6) :
  students_Germain + students_Newton + students_Young + students_Euler = 35 :=
by {
  sorry
}

end euclid_middle_school_math_students_l0_225


namespace amount_after_2_years_l0_760

noncomputable def amount_after_n_years (present_value : ℝ) (rate_of_increase : ℝ) (years : ℕ) : ℝ :=
  present_value * (1 + rate_of_increase)^years

theorem amount_after_2_years :
  amount_after_n_years 6400 (1/8) 2 = 8100 :=
by
  sorry

end amount_after_2_years_l0_760


namespace NOQZ_has_same_product_as_MNOQ_l0_145

/-- Each letter of the alphabet is assigned a value (A=1, B=2, C=3, ..., Z=26). -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13
  | 'N' => 14 | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19
  | 'T' => 20 | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _   => 0  -- We'll assume only uppercase letters are inputs

/-- The product of a four-letter list is the product of the values of its four letters. -/
def list_product (lst : List Char) : ℕ :=
  lst.map letter_value |>.foldl (· * ·) 1

/-- The product of the list MNOQ is calculated. -/
def product_MNOQ : ℕ := list_product ['M', 'N', 'O', 'Q']
/-- The product of the list BEHK is calculated. -/
def product_BEHK : ℕ := list_product ['B', 'E', 'H', 'K']
/-- The product of the list NOQZ is calculated. -/
def product_NOQZ : ℕ := list_product ['N', 'O', 'Q', 'Z']

theorem NOQZ_has_same_product_as_MNOQ :
  product_NOQZ = product_MNOQ := by
  sorry

end NOQZ_has_same_product_as_MNOQ_l0_145


namespace urn_gold_coins_percentage_l0_772

theorem urn_gold_coins_percentage (obj_perc_beads : ℝ) (coins_perc_gold : ℝ) : 
    obj_perc_beads = 0.15 → coins_perc_gold = 0.65 → 
    (1 - obj_perc_beads) * coins_perc_gold = 0.5525 := 
by
  intros h_obj_perc_beads h_coins_perc_gold
  sorry

end urn_gold_coins_percentage_l0_772


namespace has_two_zeros_of_f_l0_141

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.exp x - a

theorem has_two_zeros_of_f (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (-1 / Real.exp 2 < a ∧ a < 0) := by
sorry

end has_two_zeros_of_f_l0_141


namespace minimum_value_of_y_at_l0_23

noncomputable def y (x : ℝ) : ℝ := x * 2^x

theorem minimum_value_of_y_at :
  ∃ x : ℝ, (∀ x' : ℝ, y x ≤ y x') ∧ x = -1 / Real.log 2 :=
by
  sorry

end minimum_value_of_y_at_l0_23


namespace initial_deadlift_weight_l0_372

theorem initial_deadlift_weight
    (initial_squat : ℕ := 700)
    (initial_bench : ℕ := 400)
    (D : ℕ)
    (squat_loss : ℕ := 30)
    (deadlift_loss : ℕ := 200)
    (new_total : ℕ := 1490) :
    (initial_squat * (100 - squat_loss) / 100) + initial_bench + (D - deadlift_loss) = new_total → D = 800 :=
by
  sorry

end initial_deadlift_weight_l0_372


namespace table_runner_combined_area_l0_120

theorem table_runner_combined_area
    (table_area : ℝ) (cover_percentage : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) (A : ℝ) :
    table_area = 175 →
    cover_percentage = 0.8 →
    area_two_layers = 24 →
    area_three_layers = 28 →
    A = (cover_percentage * table_area - area_two_layers - area_three_layers) + area_two_layers + 2 * area_three_layers →
    A = 168 :=
by
  intros h_table_area h_cover_percentage h_area_two_layers h_area_three_layers h_A
  sorry

end table_runner_combined_area_l0_120


namespace value_of_composed_operations_l0_79

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_composed_operations : op2 (op1 15) = -15 :=
by
  sorry

end value_of_composed_operations_l0_79


namespace find_c_value_l0_599

-- Given condition: x^2 + 300x + c = (x + a)^2
-- Problem statement: Prove that c = 22500 for the given conditions
theorem find_c_value (x a c : ℝ) : (x^2 + 300 * x + c = (x + 150)^2) → (c = 22500) :=
by
  intro h
  sorry

end find_c_value_l0_599


namespace lines_intersect_not_perpendicular_l0_223

noncomputable def slopes_are_roots (m k1 k2 : ℝ) : Prop :=
  k1^2 + m*k1 - 2 = 0 ∧ k2^2 + m*k2 - 2 = 0

theorem lines_intersect_not_perpendicular (m k1 k2 : ℝ) (h : slopes_are_roots m k1 k2) : (k1 * k2 = -2 ∧ k1 ≠ k2) → ∃ l1 l2 : ℝ, l1 ≠ l2 ∧ l1 = k1 ∧ l2 = k2 :=
by
  sorry

end lines_intersect_not_perpendicular_l0_223


namespace ratio_of_blue_marbles_l0_742

theorem ratio_of_blue_marbles {total_marbles red_marbles orange_marbles blue_marbles : ℕ} 
  (h_total : total_marbles = 24)
  (h_red : red_marbles = 6)
  (h_orange : orange_marbles = 6)
  (h_blue : blue_marbles = total_marbles - red_marbles - orange_marbles) : 
  (blue_marbles : ℚ) / (total_marbles : ℚ) = 1 / 2 := 
by
  sorry -- the proof is omitted as per instructions

end ratio_of_blue_marbles_l0_742


namespace factorize_expression_1_factorize_expression_2_l0_422

theorem factorize_expression_1 (m : ℤ) : 
  m^3 - 2 * m^2 - 4 * m + 8 = (m - 2)^2 * (m + 2) := 
sorry

theorem factorize_expression_2 (x y : ℤ) : 
  x^2 - 2 * x * y + y^2 - 9 = (x - y + 3) * (x - y - 3) :=
sorry

end factorize_expression_1_factorize_expression_2_l0_422


namespace four_digit_number_properties_l0_825

theorem four_digit_number_properties :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 8 ∧ 
    a = 3 * b ∧ 
    d = 4 * c ∧ 
    1000 * a + 100 * b + 10 * c + d = 6200 :=
by
  sorry

end four_digit_number_properties_l0_825


namespace arithmetic_mean_difference_l0_388

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
by sorry

end arithmetic_mean_difference_l0_388


namespace range_of_k_l0_361

noncomputable def point_satisfies_curve (a k : ℝ) : Prop :=
(-a)^2 - a * (-a) + 2 * a + k = 0

theorem range_of_k (a k : ℝ) (h : point_satisfies_curve a k) : k ≤ 1 / 2 :=
by
  sorry

end range_of_k_l0_361


namespace find_other_number_l0_252

theorem find_other_number (a b : ℕ) (h1 : (a + b) / 2 = 7) (h2 : a = 5) : b = 9 :=
by
  sorry

end find_other_number_l0_252


namespace central_angle_of_sector_l0_821

theorem central_angle_of_sector (r l : ℝ) (h1 : l + 2 * r = 4) (h2 : (1 / 2) * l * r = 1) : l / r = 2 :=
by
  -- The proof should be provided here
  sorry

end central_angle_of_sector_l0_821


namespace percentage_increase_l0_189

theorem percentage_increase
  (W R : ℝ)
  (H1 : 0.70 * R = 1.04999999999999982 * W) :
  (R - W) / W * 100 = 50 :=
by
  sorry

end percentage_increase_l0_189


namespace find_sales_tax_percentage_l0_397

noncomputable def salesTaxPercentage (price_with_tax : ℝ) (price_difference : ℝ) : ℝ :=
  (price_difference * 100) / (price_with_tax - price_difference)

theorem find_sales_tax_percentage :
  salesTaxPercentage 2468 161.46 = 7 := by
  sorry

end find_sales_tax_percentage_l0_397


namespace probability_same_color_two_dice_l0_201

theorem probability_same_color_two_dice :
  let total_sides : ℕ := 30
  let maroon_sides : ℕ := 5
  let teal_sides : ℕ := 10
  let cyan_sides : ℕ := 12
  let sparkly_sides : ℕ := 3
  (maroon_sides / total_sides)^2 + (teal_sides / total_sides)^2 + (cyan_sides / total_sides)^2 + (sparkly_sides / total_sides)^2 = 139 / 450 :=
by
  sorry

end probability_same_color_two_dice_l0_201


namespace colten_chickens_l0_134

variable (Colten Skylar Quentin : ℕ)

def chicken_problem_conditions :=
  (Skylar = 3 * Colten - 4) ∧
  (Quentin = 6 * Skylar + 17) ∧
  (Colten + Skylar + Quentin = 383)

theorem colten_chickens (h : chicken_problem_conditions Colten Skylar Quentin) : Colten = 37 :=
sorry

end colten_chickens_l0_134


namespace remaining_speed_20_kmph_l0_432

theorem remaining_speed_20_kmph
  (D T : ℝ)
  (H1 : (2/3 * D) / (1/3 * T) = 80)
  (H2 : T = D / 40) :
  (D / 3) / (2/3 * T) = 20 :=
by 
  sorry

end remaining_speed_20_kmph_l0_432


namespace length_of_faster_train_l0_438

/-- Define the speeds of the trains in kmph -/
def speed_faster_train := 180 -- in kmph
def speed_slower_train := 90  -- in kmph

/-- Convert speeds to m/s -/
def kmph_to_mps (speed : ℕ) : ℕ := speed * 5 / 18

/-- Define the relative speed in m/s -/
def relative_speed := kmph_to_mps speed_faster_train - kmph_to_mps speed_slower_train

/-- Define the time it takes for the faster train to cross the man in seconds -/
def crossing_time := 15 -- in seconds

/-- Define the length of the train calculation in meters -/
noncomputable def length_faster_train := relative_speed * crossing_time

theorem length_of_faster_train :
  length_faster_train = 375 :=
by
  sorry

end length_of_faster_train_l0_438


namespace number_of_rallies_l0_741

open Nat

def X_rallies : Nat := 10
def O_rallies : Nat := 100
def sequence_Os : Nat := 3
def sequence_Xs : Nat := 7

theorem number_of_rallies : 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≤ 379) ∧ 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≥ 370) := 
by
  sorry

end number_of_rallies_l0_741


namespace meeting_time_eqn_l0_224

-- Mathematical definitions derived from conditions:
def distance := 270 -- Cities A and B are 270 kilometers apart.
def speed_fast_train := 120 -- Speed of the fast train is 120 km/h.
def speed_slow_train := 75 -- Speed of the slow train is 75 km/h.
def time_head_start := 1 -- Slow train departs 1 hour before the fast train.

-- Let x be the number of hours it takes for the two trains to meet after the fast train departs
def x : Real := sorry

-- Proving the equation representing the situation:
theorem meeting_time_eqn : 75 * 1 + (120 + 75) * x = 270 :=
by
  sorry

end meeting_time_eqn_l0_224


namespace option_b_correct_l0_604

theorem option_b_correct (a : ℝ) : (-a)^3 / (-a)^2 = -a :=
by sorry

end option_b_correct_l0_604


namespace probability_at_least_one_black_eq_seven_tenth_l0_912

noncomputable def probability_drawing_at_least_one_black_ball : ℚ :=
  let total_ways := Nat.choose 5 2
  let ways_no_black := Nat.choose 3 2
  1 - (ways_no_black / total_ways)

theorem probability_at_least_one_black_eq_seven_tenth :
  probability_drawing_at_least_one_black_ball = 7 / 10 :=
by
  sorry

end probability_at_least_one_black_eq_seven_tenth_l0_912


namespace negation_of_universal_statement_l0_731

theorem negation_of_universal_statement:
  (∀ x : ℝ, x ≥ 2) ↔ ¬ (∃ x : ℝ, x < 2) :=
by {
  sorry
}

end negation_of_universal_statement_l0_731


namespace time_upstream_equal_nine_hours_l0_501

noncomputable def distance : ℝ := 126
noncomputable def time_downstream : ℝ := 7
noncomputable def current_speed : ℝ := 2
noncomputable def downstream_speed := distance / time_downstream
noncomputable def boat_speed := downstream_speed - current_speed
noncomputable def upstream_speed := boat_speed - current_speed

theorem time_upstream_equal_nine_hours : (distance / upstream_speed) = 9 := by
  sorry

end time_upstream_equal_nine_hours_l0_501


namespace distinct_students_l0_858

theorem distinct_students 
  (students_euler : ℕ) (students_gauss : ℕ) (students_fibonacci : ℕ) (overlap_euler_gauss : ℕ)
  (h_euler : students_euler = 15) 
  (h_gauss : students_gauss = 10) 
  (h_fibonacci : students_fibonacci = 12) 
  (h_overlap : overlap_euler_gauss = 3) 
  : students_euler + students_gauss + students_fibonacci - overlap_euler_gauss = 34 :=
by
  sorry

end distinct_students_l0_858


namespace central_angle_of_spherical_sector_l0_203

theorem central_angle_of_spherical_sector (R α r m : ℝ) (h1 : R * Real.pi * r = 2 * R * Real.pi * m) (h2 : R^2 = r^2 + (R - m)^2) :
  α = 2 * Real.arccos (3 / 5) :=
by
  sorry

end central_angle_of_spherical_sector_l0_203


namespace k_bounds_inequality_l0_274

open Real

theorem k_bounds_inequality (k : ℝ) :
  (∀ x : ℝ, abs ((x^2 - k * x + 1) / (x^2 + x + 1)) < 3) ↔ -5 ≤ k ∧ k ≤ 1 := 
sorry

end k_bounds_inequality_l0_274


namespace arithmetic_geometric_mean_l0_520

theorem arithmetic_geometric_mean (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l0_520


namespace arithmetic_sum_first_11_terms_l0_818

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

variable (a : ℕ → ℝ)

theorem arithmetic_sum_first_11_terms (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_sum_condition : a 2 + a 6 + a 10 = 6) :
  sum_first_n_terms a 11 = 22 :=
sorry

end arithmetic_sum_first_11_terms_l0_818


namespace range_of_x_l0_633

-- Let p and q be propositions regarding the range of x:
def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

-- Main theorem statement
theorem range_of_x 
  (h1 : ∀ x : ℝ, p x ∨ q x)
  (h2 : ∀ x : ℝ, ¬ q x) :
  ∀ x : ℝ, (x ≤ 0 ∨ x ≥ 4) := by
  sorry

end range_of_x_l0_633


namespace probability_of_selecting_one_is_correct_l0_187

-- Define the number of elements in the first 20 rows of Pascal's triangle
def totalElementsInPascalFirst20Rows : ℕ := 210

-- Define the number of ones in the first 20 rows of Pascal's triangle
def totalOnesInPascalFirst20Rows : ℕ := 39

-- The probability as a rational number
def probabilityOfSelectingOne : ℚ := totalOnesInPascalFirst20Rows / totalElementsInPascalFirst20Rows

theorem probability_of_selecting_one_is_correct :
  probabilityOfSelectingOne = 13 / 70 :=
by
  -- Proof is omitted
  sorry

end probability_of_selecting_one_is_correct_l0_187


namespace constant_expression_l0_147

theorem constant_expression 
  (x y : ℝ) 
  (h₁ : x + y = 1) 
  (h₂ : x ≠ 1) 
  (h₃ : y ≠ 1) : 
  (x / (y^3 - 1) + y / (1 - x^3) + 2 * (x - y) / (x^2 * y^2 + 3)) = 0 :=
by 
  sorry

end constant_expression_l0_147


namespace time_to_reach_julia_via_lee_l0_389

theorem time_to_reach_julia_via_lee (d1 d2 d3 : ℕ) (t1 t2 : ℕ) :
  d1 = 2 → 
  t1 = 6 → 
  d3 = 3 → 
  (∀ v, v = d1 / t1) → 
  t2 = d3 / v → 
  t2 = 9 :=
by
  intros h1 h2 h3 hv ht2
  sorry

end time_to_reach_julia_via_lee_l0_389


namespace intersection_points_of_circle_and_vertical_line_l0_575

theorem intersection_points_of_circle_and_vertical_line :
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (3, y1) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y1) ≠ (3, y2)) := 
by
  sorry

end intersection_points_of_circle_and_vertical_line_l0_575


namespace rearrange_circles_sums13_l0_952

def isSum13 (a b c d x y z w : ℕ) : Prop :=
  (a + 4 + b = 13) ∧ (b + 2 + d = 13) ∧ (d + 1 + c = 13) ∧ (c + 3 + a = 13)

theorem rearrange_circles_sums13 : 
  ∃ (a b c d x y z w : ℕ), 
  a = 4 ∧ b = 5 ∧ c = 6 ∧ d = 6 ∧ 
  a + b = 9 ∧ b + z = 11 ∧ z + c = 12 ∧ c + a = 10 ∧ 
  isSum13 a b c d x y z w :=
by {
  sorry
}

end rearrange_circles_sums13_l0_952


namespace constant_term_of_binomial_expansion_l0_125

noncomputable def constant_in_binomial_expansion (a : ℝ) : ℝ := 
  if h : a = ∫ (x : ℝ) in (0)..(1), 2 * x 
  then ((1 : ℝ) - (a : ℝ)^(-1 : ℝ))^6
  else 0

theorem constant_term_of_binomial_expansion : 
  ∃ a : ℝ, (a = ∫ (x : ℝ) in (0)..(1), 2 * x) → constant_in_binomial_expansion a = (15 : ℝ) := sorry

end constant_term_of_binomial_expansion_l0_125


namespace polar_to_rectangular_l0_318

theorem polar_to_rectangular (r θ : ℝ) (h1 : r = 3 * Real.sqrt 2) (h2 : θ = Real.pi / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (3, 3) :=
by
  -- Proof goes here
  sorry

end polar_to_rectangular_l0_318


namespace horizontal_length_tv_screen_l0_160

theorem horizontal_length_tv_screen : 
  ∀ (a b : ℝ), (a / b = 4 / 3) → (a ^ 2 + b ^ 2 = 27 ^ 2) → a = 21.5 := 
by 
  sorry

end horizontal_length_tv_screen_l0_160


namespace daily_earnings_r_l0_446

theorem daily_earnings_r (p q r s : ℝ)
  (h1 : p + q + r + s = 300)
  (h2 : p + r = 120)
  (h3 : q + r = 130)
  (h4 : s + r = 200)
  (h5 : p + s = 116.67) : 
  r = 75 :=
by
  sorry

end daily_earnings_r_l0_446


namespace product_of_tangents_l0_72

theorem product_of_tangents : 
  (Real.tan (Real.pi / 8) * Real.tan (3 * Real.pi / 8) * 
   Real.tan (5 * Real.pi / 8) * Real.tan (7 * Real.pi / 8) = -2 * Real.sqrt 2) :=
sorry

end product_of_tangents_l0_72


namespace three_digit_number_divisibility_four_digit_number_divisibility_l0_513

-- Definition of three-digit number
def is_three_digit_number (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

-- Definition of four-digit number
def is_four_digit_number (b : ℕ) : Prop := 1000 ≤ b ∧ b ≤ 9999

-- First proof problem
theorem three_digit_number_divisibility (a : ℕ) (h : is_three_digit_number a) : 
  (1001 * a) % 7 = 0 ∧ (1001 * a) % 11 = 0 ∧ (1001 * a) % 13 = 0 := 
sorry

-- Second proof problem
theorem four_digit_number_divisibility (b : ℕ) (h : is_four_digit_number b) : 
  (10001 * b) % 73 = 0 ∧ (10001 * b) % 137 = 0 := 
sorry

end three_digit_number_divisibility_four_digit_number_divisibility_l0_513


namespace correct_population_l0_290

variable (P : ℕ) (S : ℕ)
variable (math_scores : ℕ → Type)

-- Assume P is the total number of students who took the exam.
-- Let math_scores(P) represent the math scores of P students.

def population_data (P : ℕ) : Prop := 
  P = 50000

def sample_data (S : ℕ) : Prop :=
  S = 2000

theorem correct_population (P : ℕ) (S : ℕ) (math_scores : ℕ → Type)
  (hP : population_data P) (hS : sample_data S) : 
  math_scores P = math_scores 50000 :=
by {
  sorry
}

end correct_population_l0_290


namespace tangent_line_exponential_passing_through_origin_l0_574

theorem tangent_line_exponential_passing_through_origin :
  ∃ (p : ℝ × ℝ) (m : ℝ), 
  (p = (1, Real.exp 1)) ∧ (m = Real.exp 1) ∧ 
  (∀ x : ℝ, x ≠ 1 → ¬ (∃ k : ℝ, k = (Real.exp x - 0) / (x - 0) ∧ k = Real.exp x)) :=
by 
  sorry

end tangent_line_exponential_passing_through_origin_l0_574


namespace minimum_value_expr_l0_496

noncomputable def expr (x y z : ℝ) : ℝ :=
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) +
  (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2)))

theorem minimum_value_expr : ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) →
  expr x y z ≥ 2 :=
by sorry

end minimum_value_expr_l0_496


namespace completing_square_correctness_l0_636

theorem completing_square_correctness :
  (2 * x^2 - 4 * x - 7 = 0) ->
  ((x - 1)^2 = 9 / 2) :=
sorry

end completing_square_correctness_l0_636


namespace smallest_possible_n_l0_644

theorem smallest_possible_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 60) : n = 16 :=
sorry

end smallest_possible_n_l0_644


namespace mrs_hilt_total_candy_l0_734

theorem mrs_hilt_total_candy :
  (2 * 3) + (4 * 2) + (6 * 4) = 38 :=
by
  -- here, skip the proof as instructed
  sorry

end mrs_hilt_total_candy_l0_734


namespace polynomial_factorization_l0_320

noncomputable def polynomial_equivalence : Prop :=
  ∀ x : ℂ, (x^12 - 3*x^9 + 3*x^3 + 1) = (x + 1)^4 * (x^2 - x + 1)^4

theorem polynomial_factorization : polynomial_equivalence := by
  sorry

end polynomial_factorization_l0_320


namespace correct_calculation_l0_523

theorem correct_calculation (n : ℕ) (h : n - 59 = 43) : n - 46 = 56 :=
by
  sorry

end correct_calculation_l0_523


namespace dot_product_of_a_and_b_l0_393

noncomputable def vector_a (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
a

noncomputable def vector_b (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
b

theorem dot_product_of_a_and_b {a b : ℝ × ℝ} 
  (h1 : a + b = (1, -3)) 
  (h2 : a - b = (3, 7)) : 
  (a.1 * b.1 + a.2 * b.2) = -12 := 
sorry

end dot_product_of_a_and_b_l0_393


namespace calculate_sum_l0_395

theorem calculate_sum (P r : ℝ) (h1 : 2 * P * r = 10200) (h2 : P * ((1 + r) ^ 2 - 1) = 11730) : P = 17000 :=
sorry

end calculate_sum_l0_395


namespace arithmetic_sequence_l0_776

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) (d a1 : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) (d a1 : ℤ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Given conditions
theorem arithmetic_sequence (n : ℕ) (d a1 : ℤ) (S3 : ℤ) (h1 : a1 = 10) (h2 : S_n 3 d a1 = 24) :
  (a_n n d a1 = 12 - 2 * n) ∧ (S_n n (-2) 12 = -n^2 + 11 * n) ∧ (∀ k, S_n k (-2) 12 ≤ 30) :=
by
  sorry

end arithmetic_sequence_l0_776


namespace ratio_of_areas_l0_840

theorem ratio_of_areas 
  (a b c : ℕ) (d e f : ℕ)
  (hABC : a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2)
  (hDEF : d = 8 ∧ e = 15 ∧ f = 17 ∧ d^2 + e^2 = f^2) :
  (1/2 * a * b) / (1/2 * d * e) = 2 / 5 :=
by
  sorry

end ratio_of_areas_l0_840


namespace max_self_intersection_points_13_max_self_intersection_points_1950_l0_657

def max_self_intersection_points (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n - 3) / 2 else n * (n - 4) / 2 + 1

theorem max_self_intersection_points_13 : max_self_intersection_points 13 = 65 :=
by sorry

theorem max_self_intersection_points_1950 : max_self_intersection_points 1950 = 1897851 :=
by sorry

end max_self_intersection_points_13_max_self_intersection_points_1950_l0_657


namespace moon_land_value_l0_723

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

end moon_land_value_l0_723


namespace geometric_sequence_sum_l0_334

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (a1 : a 1 = 3)
  (a4 : a 4 = 24)
  (h_geo : ∃ q : ℝ, ∀ n : ℕ, a n = 3 * q^(n - 1)) :
  a 3 + a 4 + a 5 = 84 :=
by
  sorry

end geometric_sequence_sum_l0_334


namespace fill_bathtub_time_l0_941

def rate_cold_water : ℚ := 3 / 20
def rate_hot_water : ℚ := 1 / 8
def rate_drain : ℚ := 3 / 40
def net_rate : ℚ := rate_cold_water + rate_hot_water - rate_drain

theorem fill_bathtub_time :
  net_rate = 1/5 → (1 / net_rate) = 5 := by
  sorry

end fill_bathtub_time_l0_941


namespace competitive_exam_candidates_l0_182

theorem competitive_exam_candidates (x : ℝ)
  (A_selected : ℝ := 0.06 * x) 
  (B_selected : ℝ := 0.07 * x) 
  (h : B_selected = A_selected + 81) :
  x = 8100 := by
  sorry

end competitive_exam_candidates_l0_182


namespace find_number_of_students_l0_873

theorem find_number_of_students
    (S N : ℕ) 
    (h₁ : 4 * S + 3 = N)
    (h₂ : 5 * S = N + 6) : 
  S = 9 :=
by
  sorry

end find_number_of_students_l0_873


namespace only_element_in_intersection_l0_712

theorem only_element_in_intersection :
  ∃! (n : ℕ), n = 2500 ∧ ∃ (r : ℚ), r ≠ 2 ∧ r ≠ -2 ∧ 404 / (r^2 - 4) = n := sorry

end only_element_in_intersection_l0_712


namespace convex_polyhedron_same_number_of_sides_l0_346

theorem convex_polyhedron_same_number_of_sides {N : ℕ} (hN : N ≥ 4): 
  ∃ (f1 f2 : ℕ), (f1 >= 3 ∧ f1 < N ∧ f2 >= 3 ∧ f2 < N) ∧ f1 = f2 :=
by
  sorry

end convex_polyhedron_same_number_of_sides_l0_346


namespace radius_of_larger_circle_l0_244

theorem radius_of_larger_circle (r : ℝ) (r_pos : r > 0)
    (ratio_condition : ∀ (rs : ℝ), rs = 3 * r)
    (diameter_condition : ∀ (ac : ℝ), ac = 6 * r)
    (chord_tangent_condition : ∀ (ab : ℝ), ab = 12) :
     (radius : ℝ) = 3 * r :=
by
  sorry

end radius_of_larger_circle_l0_244


namespace stratified_sampling_example_l0_739

theorem stratified_sampling_example 
    (high_school_students : ℕ)
    (junior_high_students : ℕ) 
    (sampled_high_school_students : ℕ)
    (sampling_ratio : ℚ)
    (total_students : ℕ)
    (n : ℕ)
    (h1 : high_school_students = 3500)
    (h2 : junior_high_students = 1500)
    (h3 : sampled_high_school_students = 70)
    (h4 : sampling_ratio = sampled_high_school_students / high_school_students)
    (h5 : total_students = high_school_students + junior_high_students) :
    n = total_students * sampling_ratio → 
    n = 100 :=
by
  sorry

end stratified_sampling_example_l0_739


namespace union_of_sets_l0_883

variable (x : ℝ)

def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | -1 < x ∧ x < 2}
def target : Set ℝ := {x | -1 < x ∧ x < 3}

theorem union_of_sets : (A ∪ B) = target :=
by
  sorry

end union_of_sets_l0_883


namespace number_of_subsets_of_P_l0_830

noncomputable def P : Set ℝ := {x | x^2 - 2*x + 1 = 0}

theorem number_of_subsets_of_P : ∃ (n : ℕ), n = 2 ∧ ∀ S : Set ℝ, S ⊆ P → S = ∅ ∨ S = {1} := by
  sorry

end number_of_subsets_of_P_l0_830


namespace trig_identity_proof_l0_590

theorem trig_identity_proof : 
  (Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end trig_identity_proof_l0_590


namespace red_red_pairs_l0_914

theorem red_red_pairs (green_shirts red_shirts total_students total_pairs green_green_pairs : ℕ)
    (hg1 : green_shirts = 64)
    (hr1 : red_shirts = 68)
    (htotal : total_students = 132)
    (htotal_pairs : total_pairs = 66)
    (hgreen_green_pairs : green_green_pairs = 28) :
    (total_students = green_shirts + red_shirts) ∧
    (green_green_pairs ≤ total_pairs) ∧
    (∃ red_red_pairs, red_red_pairs = 30) :=
by
  sorry

end red_red_pairs_l0_914


namespace smallest_result_l0_713

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

end smallest_result_l0_713


namespace price_per_glass_first_day_l0_197

theorem price_per_glass_first_day
    (O W : ℝ) (P1 P2 : ℝ)
    (h1 : O = W)
    (h2 : P2 = 0.40)
    (h3 : 2 * O * P1 = 3 * O * P2) :
    P1 = 0.60 :=
by
    sorry

end price_per_glass_first_day_l0_197


namespace find_vector_at_t5_l0_13

def vector_on_line (t : ℝ) : ℝ × ℝ := 
  let a := (0, 11) -- From solving the system of equations
  let d := (2, -4) -- From solving the system of equations
  (a.1 + t * d.1, a.2 + t * d.2)

theorem find_vector_at_t5 : vector_on_line 5 = (10, -9) := 
by 
  sorry

end find_vector_at_t5_l0_13


namespace f_is_odd_l0_238

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2 * x

-- State the problem
theorem f_is_odd :
  ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

end f_is_odd_l0_238


namespace zero_in_interval_l0_205

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem zero_in_interval : 
  ∃ x₀, f x₀ = 0 ∧ (2 : ℝ) < x₀ ∧ x₀ < (3 : ℝ) :=
by
  sorry

end zero_in_interval_l0_205


namespace rectangle_ratio_l0_669

-- Define the width of the rectangle
def width : ℕ := 7

-- Define the area of the rectangle
def area : ℕ := 196

-- Define that the length is a multiple of the width
def length_is_multiple_of_width (l w : ℕ) : Prop := ∃ k : ℕ, l = k * w

-- Define that the ratio of the length to the width is 4:1
def ratio_is_4_to_1 (l w : ℕ) : Prop := l / w = 4

theorem rectangle_ratio (l w : ℕ) (h1 : w = width) (h2 : area = l * w) (h3 : length_is_multiple_of_width l w) : ratio_is_4_to_1 l w :=
by
  sorry

end rectangle_ratio_l0_669


namespace ball_bounces_to_C_l0_854

/--
On a rectangular table with dimensions 9 cm in length and 7 cm in width, a small ball is shot from point A at a 45-degree angle. Upon reaching point E, it bounces off at a 45-degree angle and continues to roll forward. Throughout its motion, the ball bounces off the table edges at a 45-degree angle each time. Prove that, starting from point A, the ball first reaches point C after exactly 14 bounces.
-/
theorem ball_bounces_to_C (length width : ℝ) (angle : ℝ) (bounce_angle : ℝ) :
  length = 9 ∧ width = 7 ∧ angle = 45 ∧ bounce_angle = 45 → bounces_to_C = 14 :=
by
  intros
  sorry

end ball_bounces_to_C_l0_854


namespace cab_speed_fraction_l0_379

theorem cab_speed_fraction :
  ∀ (S R : ℝ),
    (75 * S = 90 * R) →
    (R / S = 5 / 6) :=
by
  intros S R h
  sorry

end cab_speed_fraction_l0_379


namespace proof_l0_592

noncomputable def question (a b c : ℂ) : ℂ := (a^3 + b^3 + c^3) / (a * b * c)

theorem proof (a b c : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 15)
  (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2 * a * b * c) :
  question a b c = 18 :=
by
  sorry

end proof_l0_592


namespace sum_of_next_17_consecutive_integers_l0_796

theorem sum_of_next_17_consecutive_integers (x : ℤ) (h₁ : (List.range 17).sum + 17 * x = 306) :
  (List.range 17).sum + 17 * (x + 17)  = 595 := 
sorry

end sum_of_next_17_consecutive_integers_l0_796


namespace min_value_of_quadratic_l0_827

theorem min_value_of_quadratic (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2 * x + m) 
  (min_val : ∀ x ≥ 2, f x ≥ -2) : m = -2 := 
by
  sorry

end min_value_of_quadratic_l0_827


namespace product_mod_17_eq_zero_l0_536

theorem product_mod_17_eq_zero :
    (2001 * 2002 * 2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end product_mod_17_eq_zero_l0_536


namespace yangmei_1_yangmei_2i_yangmei_2ii_l0_242

-- Problem 1: Prove that a = 20
theorem yangmei_1 (a : ℕ) (h : 160 * a + 270 * a = 8600) : a = 20 := by
  sorry

-- Problem 2 (i): Prove x = 44 and y = 36
theorem yangmei_2i (x y : ℕ) (h1 : 160 * x + 270 * y = 16760) (h2 : 8 * x + 18 * y = 1000) : x = 44 ∧ y = 36 := by
  sorry

-- Problem 2 (ii): Prove b = 9 or 18
theorem yangmei_2ii (m n b : ℕ) (h1 : 8 * (m + b) + 18 * n = 1000) (h2 : 160 * m + 270 * n = 16760) (h3 : 0 < b)
: b = 9 ∨ b = 18 := by
  sorry

end yangmei_1_yangmei_2i_yangmei_2ii_l0_242


namespace sector_area_l0_38

theorem sector_area (theta l : ℝ) (h_theta : theta = 2) (h_l : l = 2) :
    let r := l / theta
    let S := 1 / 2 * l * r
    S = 1 := by
  sorry

end sector_area_l0_38


namespace total_grandchildren_l0_75

-- Define the conditions 
def daughters := 5
def sons := 4
def children_per_daughter := 8 + 7
def children_per_son := 6 + 3

-- State the proof problem
theorem total_grandchildren : daughters * children_per_daughter + sons * children_per_son = 111 :=
by
  sorry

end total_grandchildren_l0_75


namespace window_dimensions_l0_162

-- Given conditions
def panes := 12
def rows := 3
def columns := 4
def height_to_width_ratio := 3
def border_width := 2

-- Definitions based on given conditions
def width_per_pane (x : ℝ) := x
def height_per_pane (x : ℝ) := 3 * x

def total_width (x : ℝ) := columns * width_per_pane x + (columns + 1) * border_width
def total_height (x : ℝ) := rows * height_per_pane x + (rows + 1) * border_width

-- Theorem statement: width and height of the window
theorem window_dimensions (x : ℝ) : 
  total_width x = 4 * x + 10 ∧ 
  total_height x = 9 * x + 8 := by
  sorry

end window_dimensions_l0_162


namespace cooking_oil_distribution_l0_732

theorem cooking_oil_distribution (total_oil : ℝ) (oil_A : ℝ) (oil_B : ℝ) (oil_C : ℝ)
    (h_total_oil : total_oil = 3 * 1000) -- Total oil is 3000 milliliters
    (h_A_B : oil_A = oil_B + 200) -- A receives 200 milliliters more than B
    (h_B_C : oil_B = oil_C + 200) -- B receives 200 milliliters more than C
    : oil_B = 1000 :=              -- We need to prove B receives 1000 milliliters
by
  sorry

end cooking_oil_distribution_l0_732


namespace cos_half_angle_l0_371

theorem cos_half_angle (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : 0 < α ∧ α < Real.pi / 2) : 
    Real.cos (α / 2) = 2 * Real.sqrt 5 / 5 := 
by 
    sorry

end cos_half_angle_l0_371


namespace overall_average_score_l0_627

noncomputable def average_score (scores : List ℝ) : ℝ :=
  scores.sum / (scores.length)

theorem overall_average_score :
  let male_scores_avg := 82
  let female_scores_avg := 92
  let num_male_students := 8
  let num_female_students := 32
  let total_students := num_male_students + num_female_students
  let combined_scores_total := num_male_students * male_scores_avg + num_female_students * female_scores_avg
  average_score ([combined_scores_total]) / total_students = 90 :=
by 
  sorry

end overall_average_score_l0_627


namespace roots_quadratic_l0_850

theorem roots_quadratic (a b : ℝ) (h : ∀ x : ℝ, x^2 - 7 * x + 7 = 0 → (x = a) ∨ (x = b)) :
  a^2 + b^2 = 35 :=
sorry

end roots_quadratic_l0_850


namespace james_total_cost_l0_47

def milk_cost : ℝ := 4.50
def milk_tax_rate : ℝ := 0.20
def banana_cost : ℝ := 3.00
def banana_tax_rate : ℝ := 0.15
def baguette_cost : ℝ := 2.50
def baguette_tax_rate : ℝ := 0.0
def cereal_cost : ℝ := 6.00
def cereal_discount_rate : ℝ := 0.20
def cereal_tax_rate : ℝ := 0.12
def eggs_cost : ℝ := 3.50
def eggs_coupon : ℝ := 1.00
def eggs_tax_rate : ℝ := 0.18

theorem james_total_cost :
  let milk_total := milk_cost * (1 + milk_tax_rate)
  let banana_total := banana_cost * (1 + banana_tax_rate)
  let baguette_total := baguette_cost * (1 + baguette_tax_rate)
  let cereal_discounted := cereal_cost * (1 - cereal_discount_rate)
  let cereal_total := cereal_discounted * (1 + cereal_tax_rate)
  let eggs_discounted := eggs_cost - eggs_coupon
  let eggs_total := eggs_discounted * (1 + eggs_tax_rate)
  milk_total + banana_total + baguette_total + cereal_total + eggs_total = 19.68 := 
by
  sorry

end james_total_cost_l0_47


namespace least_subtraction_l0_930

theorem least_subtraction (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 45678) (h2 : d = 47) (h3 : n % d = r) : r = 35 :=
by {
  sorry
}

end least_subtraction_l0_930


namespace anna_pizza_fraction_l0_639

theorem anna_pizza_fraction :
  let total_slices := 16
  let anna_eats := 2
  let shared_slices := 1
  let anna_share := shared_slices / 3
  let fraction_alone := anna_eats / total_slices
  let fraction_shared := anna_share / total_slices
  fraction_alone + fraction_shared = 7 / 48 :=
by
  sorry

end anna_pizza_fraction_l0_639


namespace total_money_l0_839

-- Define the problem statement
theorem total_money (n : ℕ) (hn : 3 * n = 75) : (n * 1 + n * 5 + n * 10) = 400 :=
by sorry

end total_money_l0_839


namespace range_of_sum_l0_90

theorem range_of_sum (a b c d : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
sorry

end range_of_sum_l0_90


namespace pizza_slices_have_both_cheese_and_bacon_l0_110

theorem pizza_slices_have_both_cheese_and_bacon:
  ∀ (total_slices cheese_slices bacon_slices n : ℕ),
  total_slices = 15 →
  cheese_slices = 8 →
  bacon_slices = 13 →
  (total_slices = cheese_slices + bacon_slices - n) →
  n = 6 :=
by {
  -- proof skipped
  sorry
}

end pizza_slices_have_both_cheese_and_bacon_l0_110


namespace prob_log3_integer_l0_727

theorem prob_log3_integer : 
  (∃ (N: ℕ), (100 ≤ N ∧ N ≤ 999) ∧ ∃ (k: ℕ), N = 3^k) → 
  (∃ (prob : ℚ), prob = 1 / 450) :=
sorry

end prob_log3_integer_l0_727


namespace problem_power_function_l0_613

-- Defining the conditions
variable {f : ℝ → ℝ}
variable (a : ℝ)
variable (h₁ : ∀ x, f x = x^a)
variable (h₂ : f 2 = Real.sqrt 2)

-- Stating what we need to prove
theorem problem_power_function : f 4 = 2 :=
by sorry

end problem_power_function_l0_613


namespace EF_side_length_l0_309

def square_side_length (n : ℝ) : Prop := n = 10

def distance_parallel_line (d : ℝ) : Prop := d = 6.5

def area_difference (a : ℝ) : Prop := a = 13.8

theorem EF_side_length :
  ∃ (x : ℝ), square_side_length 10 ∧ distance_parallel_line 6.5 ∧ area_difference 13.8 ∧ x = 5.4 :=
sorry

end EF_side_length_l0_309


namespace adamek_marbles_l0_949

theorem adamek_marbles : ∃ n : ℕ, (∀ k : ℕ, n = 4 * k ∧ n = 3 * (k + 8)) → n = 96 :=
by
  sorry

end adamek_marbles_l0_949


namespace incorrect_statement_B_l0_833

noncomputable def y (x : ℝ) : ℝ := 2 / x 

theorem incorrect_statement_B :
  ¬ ∀ x > 0, ∀ y1 y2 : ℝ, x < y1 → y1 < y2 → y x < y y2 := sorry

end incorrect_statement_B_l0_833


namespace tangent_sphere_surface_area_l0_528

noncomputable def cube_side_length (V : ℝ) : ℝ := V^(1/3)
noncomputable def sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem tangent_sphere_surface_area (V : ℝ) (hV : V = 64) : 
  sphere_surface_area (sphere_radius (cube_side_length V)) = 16 * Real.pi :=
by
  sorry

end tangent_sphere_surface_area_l0_528


namespace sum_of_digits_of_N_l0_498

theorem sum_of_digits_of_N :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ N.digits 10 = [7, 7] :=
by
  sorry

end sum_of_digits_of_N_l0_498


namespace closest_point_to_line_l0_746

theorem closest_point_to_line {x y : ℝ} :
  (y = 2 * x - 7) → (∃ p : ℝ × ℝ, p.1 = 5 ∧ p.2 = 3 ∧ (p.1, p.2) ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} ∧ (∀ q : ℝ × ℝ, q ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} → dist ⟨x, y⟩ p ≤ dist ⟨x, y⟩ q)) :=
by
  -- proof goes here
  sorry

end closest_point_to_line_l0_746


namespace prime_pairs_perfect_square_l0_173

theorem prime_pairs_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ k : ℕ, p^(q-1) + q^(p-1) = k^2 ↔ (p = 2 ∧ q = 2) :=
by
  sorry

end prime_pairs_perfect_square_l0_173


namespace sum_of_reciprocal_roots_l0_917

theorem sum_of_reciprocal_roots (r s α β : ℝ) (h1 : 7 * r^2 - 8 * r + 6 = 0) (h2 : 7 * s^2 - 8 * s + 6 = 0) (h3 : α = 1 / r) (h4 : β = 1 / s) :
  α + β = 4 / 3 := 
sorry

end sum_of_reciprocal_roots_l0_917


namespace bus_ticket_probability_l0_876

theorem bus_ticket_probability :
  let total_tickets := 10 ^ 6
  let choices := Nat.choose 10 6 * 2
  (choices : ℝ) / total_tickets = 0.00042 :=
by
  sorry

end bus_ticket_probability_l0_876


namespace fish_weight_l0_950

variables (W G T : ℕ)

-- Define the known conditions
axiom tail_weight : W = 1
axiom head_weight : G = W + T / 2
axiom torso_weight : T = G + W

-- Define the proof statement
theorem fish_weight : W + G + T = 8 :=
by
  sorry

end fish_weight_l0_950


namespace car_average_speed_is_correct_l0_524

noncomputable def average_speed_of_car : ℝ :=
  let d1 := 30
  let s1 := 30
  let d2 := 35
  let s2 := 55
  let t3 := 0.5
  let s3 := 70
  let t4 := 40 / 60 -- 40 minutes converted to hours
  let s4 := 36
  let t1 := d1 / s1
  let t2 := d2 / s2
  let d3 := s3 * t3
  let d4 := s4 * t4
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  total_distance / total_time

theorem car_average_speed_is_correct :
  average_speed_of_car = 44.238 := 
sorry

end car_average_speed_is_correct_l0_524


namespace isosceles_triangle_area_l0_57

open Real

noncomputable def area_of_isosceles_triangle (b : ℝ) (h : ℝ) : ℝ :=
  (1/2) * b * h

theorem isosceles_triangle_area :
  ∃ (b : ℝ) (l : ℝ), h = 8 ∧ (2 * l + b = 32) ∧ (area_of_isosceles_triangle b h = 48) :=
by
  sorry

end isosceles_triangle_area_l0_57


namespace number_of_puppies_with_4_spots_is_3_l0_481

noncomputable def total_puppies : Nat := 10
noncomputable def puppies_with_5_spots : Nat := 6
noncomputable def puppies_with_2_spots : Nat := 1
noncomputable def puppies_with_4_spots : Nat := total_puppies - puppies_with_5_spots - puppies_with_2_spots

theorem number_of_puppies_with_4_spots_is_3 :
  puppies_with_4_spots = 3 := 
sorry

end number_of_puppies_with_4_spots_is_3_l0_481


namespace find_k_l0_787

theorem find_k (k t : ℝ) (h1 : t = 5) (h2 : (1/2) * (t^2) / ((k-1) * (k+1)) = 10) : 
  k = 3/2 := 
  sorry

end find_k_l0_787


namespace max_is_twice_emily_probability_l0_31

noncomputable def probability_event_max_gt_twice_emily : ℝ :=
  let total_area := 1000 * 3000
  let triangle_area := 1/2 * 1000 * 1000
  let rectangle_area := 1000 * (3000 - 2000)
  let favorable_area := triangle_area + rectangle_area
  favorable_area / total_area

theorem max_is_twice_emily_probability :
  probability_event_max_gt_twice_emily = 1 / 2 :=
by
  sorry

end max_is_twice_emily_probability_l0_31


namespace walking_west_negation_l0_118

theorem walking_west_negation (distance_east distance_west : Int) (h_east : distance_east = 6) (h_west : distance_west = -10) : 
    (10 : Int) = - distance_west := by
  sorry

end walking_west_negation_l0_118


namespace cost_of_country_cd_l0_971

theorem cost_of_country_cd
  (cost_rock_cd : ℕ) (cost_pop_cd : ℕ) (cost_dance_cd : ℕ)
  (num_each : ℕ) (julia_has : ℕ) (julia_short : ℕ)
  (total_cost : ℕ) (total_other_cds : ℕ) (cost_country_cd : ℕ) :
  cost_rock_cd = 5 →
  cost_pop_cd = 10 →
  cost_dance_cd = 3 →
  num_each = 4 →
  julia_has = 75 →
  julia_short = 25 →
  total_cost = julia_has + julia_short →
  total_other_cds = num_each * cost_rock_cd + num_each * cost_pop_cd + num_each * cost_dance_cd →
  total_cost = total_other_cds + num_each * cost_country_cd →
  cost_country_cd = 7 :=
by
  intros cost_rock_cost_pop_cost_dance_num julia_diff 
         calc_total_total_other sub_total total_cds
  sorry

end cost_of_country_cd_l0_971


namespace problem_statement_l0_150

noncomputable def decimalPartSqrtFive : ℝ := Real.sqrt 5 - 2
def integerPartSqrtThirteen : ℕ := 3

theorem problem_statement :
  decimalPartSqrtFive + integerPartSqrtThirteen - Real.sqrt 5 = 1 :=
by
  sorry

end problem_statement_l0_150


namespace avg_growth_rate_leq_half_sum_l0_714

theorem avg_growth_rate_leq_half_sum (m n p : ℝ) (hm : 0 ≤ m) (hn : 0 ≤ n)
    (hp : (1 + p / 100)^2 = (1 + m / 100) * (1 + n / 100)) : 
    p ≤ (m + n) / 2 :=
by
  sorry

end avg_growth_rate_leq_half_sum_l0_714


namespace gcd_2015_15_l0_838

theorem gcd_2015_15 : Nat.gcd 2015 15 = 5 :=
by
  have h1 : 2015 = 15 * 134 + 5 := by rfl
  have h2 : 15 = 5 * 3 := by rfl
  sorry

end gcd_2015_15_l0_838


namespace mean_of_remaining_three_l0_495

theorem mean_of_remaining_three (a b c : ℝ) (h₁ : (a + b + c + 105) / 4 = 93) : (a + b + c) / 3 = 89 :=
  sorry

end mean_of_remaining_three_l0_495


namespace complex_division_l0_891

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (1 + i) / (1 - i) = i :=
by
  sorry

end complex_division_l0_891


namespace closest_to_sin_2016_deg_is_neg_half_l0_92

/-- Given the value of \( \sin 2016^\circ \), show that the closest number from the given options is \( -\frac{1}{2} \).
Options:
A: \( \frac{11}{2} \)
B: \( -\frac{1}{2} \)
C: \( \frac{\sqrt{2}}{2} \)
D: \( -1 \)
-/
theorem closest_to_sin_2016_deg_is_neg_half :
  let sin_2016 := Real.sin (2016 * Real.pi / 180)
  |sin_2016 - (-1 / 2)| < |sin_2016 - 11 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - Real.sqrt 2 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - (-1)| :=
by
  sorry

end closest_to_sin_2016_deg_is_neg_half_l0_92


namespace max_expr_value_l0_717

theorem max_expr_value (a b c d : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) (hc : 0 ≤ c) (hc1 : c ≤ 1) (hd : 0 ≤ d) (hd1 : d ≤ 1) : 
  a + b + c + d - a * b - b * c - c * d - d * a ≤ 2 :=
sorry

end max_expr_value_l0_717


namespace correct_operation_l0_865

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end correct_operation_l0_865


namespace find_k_value_l0_257

theorem find_k_value (k : ℝ) : 
  5 + ∑' n : ℕ, (5 + k + n) / 5^(n+1) = 12 → k = 18.2 :=
by 
  sorry

end find_k_value_l0_257


namespace correct_conclusions_l0_390

-- Definitions based on conditions
def condition_1 (x : ℝ) : Prop := x ≠ 0 → x + |x| > 0
def condition_3 (a b c : ℝ) (Δ : ℝ) : Prop := a > 0 ∧ Δ ≤ 0 ∧ Δ = b^2 - 4*a*c → 
  ∀ x, a*x^2 + b*x + c ≥ 0

-- Stating the proof problem
theorem correct_conclusions (x a b c Δ : ℝ) :
  (condition_1 x) ∧ (condition_3 a b c Δ) :=
sorry

end correct_conclusions_l0_390


namespace find_x_orthogonal_l0_762

theorem find_x_orthogonal :
  ∃ x : ℝ, (2 * x + 5 * (-3) = 0) ∧ x = 15 / 2 :=
by
  sorry

end find_x_orthogonal_l0_762


namespace points_on_line_l0_378

-- Define the two points the line connects
def P1 : (ℝ × ℝ) := (8, 10)
def P2 : (ℝ × ℝ) := (2, -2)

-- Define the candidate points
def A : (ℝ × ℝ) := (5, 4)
def E : (ℝ × ℝ) := (1, -4)

-- Define the line equation, given the slope and y-intercept
def line (x : ℝ) : ℝ := 2 * x - 6

theorem points_on_line :
  (A.snd = line A.fst) ∧ (E.snd = line E.fst) :=
by
  sorry

end points_on_line_l0_378


namespace max_value_x_plus_y_max_value_x_plus_y_achieved_l0_287

theorem max_value_x_plus_y (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : x + y ≤ 6 * Real.sqrt 5 :=
by
  sorry

theorem max_value_x_plus_y_achieved (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : ∃ x y, x + y = 6 * Real.sqrt 5 :=
by
  sorry

end max_value_x_plus_y_max_value_x_plus_y_achieved_l0_287


namespace william_farm_tax_l0_251

theorem william_farm_tax :
  let total_tax_collected := 3840
  let william_land_percentage := 0.25
  william_land_percentage * total_tax_collected = 960 :=
by sorry

end william_farm_tax_l0_251


namespace Tom_spends_375_dollars_l0_890

noncomputable def totalCost (numBricks : ℕ) (halfDiscount : ℚ) (fullPrice : ℚ) : ℚ :=
  let halfBricks := numBricks / 2
  let discountedPrice := fullPrice * halfDiscount
  (halfBricks * discountedPrice) + (halfBricks * fullPrice)

theorem Tom_spends_375_dollars : 
  ∀ (numBricks : ℕ) (halfDiscount fullPrice : ℚ), 
  numBricks = 1000 → halfDiscount = 0.5 → fullPrice = 0.5 → totalCost numBricks halfDiscount fullPrice = 375 := 
by
  intros numBricks halfDiscount fullPrice hnumBricks hhalfDiscount hfullPrice
  rw [hnumBricks, hhalfDiscount, hfullPrice]
  sorry

end Tom_spends_375_dollars_l0_890


namespace comparison_of_powers_l0_164

theorem comparison_of_powers : 6 ^ 0.7 > 0.7 ^ 6 ∧ 0.7 ^ 6 > 0.6 ^ 7 := by
  sorry

end comparison_of_powers_l0_164


namespace initial_investment_l0_366

theorem initial_investment (P r : ℝ) 
  (h1 : 600 = P * (1 + 0.02 * r)) 
  (h2 : 850 = P * (1 + 0.07 * r)) : 
  P = 500 :=
sorry

end initial_investment_l0_366


namespace incorrect_statement_count_l0_658

theorem incorrect_statement_count :
  let statements := ["Every number has a square root",
                     "The square root of a number must be positive",
                     "The square root of a^2 is a",
                     "The square root of (π - 4)^2 is π - 4",
                     "A square root cannot be negative"]
  let incorrect := [statements.get! 0, statements.get! 1, statements.get! 2, statements.get! 3]
  incorrect.length = 4 :=
by
  sorry

end incorrect_statement_count_l0_658


namespace action_figure_cost_l0_454

def initial_figures : ℕ := 7
def total_figures_needed : ℕ := 16
def total_cost : ℕ := 72

theorem action_figure_cost :
  total_cost / (total_figures_needed - initial_figures) = 8 := by
  sorry

end action_figure_cost_l0_454


namespace least_positive_integer_with_12_factors_is_972_l0_102

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l0_102


namespace product_of_consecutive_natural_numbers_l0_316

theorem product_of_consecutive_natural_numbers (n : ℕ) : 
  (∃ t : ℕ, n = t * (t + 1) - 1) ↔ ∃ x : ℕ, n^2 - 1 = x * (x + 1) * (x + 2) * (x + 3) := 
sorry

end product_of_consecutive_natural_numbers_l0_316


namespace probability_at_least_one_white_ball_l0_810

/-
  We define the conditions:
  - num_white: the number of white balls,
  - num_red: the number of red balls,
  - total_balls: the total number of balls,
  - num_drawn: the number of balls drawn.
-/
def num_white : ℕ := 5
def num_red : ℕ := 4
def total_balls : ℕ := num_white + num_red
def num_drawn : ℕ := 3

/-
  Given the conditions, we need to prove that the probability of drawing at least one white ball is 20/21.
-/
theorem probability_at_least_one_white_ball :
  (1 : ℚ) - (4 / 84) = 20 / 21 :=
by
  sorry

end probability_at_least_one_white_ball_l0_810


namespace grasshopper_position_after_100_jumps_l0_108

theorem grasshopper_position_after_100_jumps :
  let start_pos := 1
  let jumps (n : ℕ) := n
  let total_positions := 6
  let total_distance := (100 * (100 + 1)) / 2
  (start_pos + (total_distance % total_positions)) % total_positions = 5 :=
by
  sorry

end grasshopper_position_after_100_jumps_l0_108


namespace ellipse_foci_coordinates_l0_792

/-- Define the parameters for the ellipse. -/
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 169 = 1

/-- Prove the coordinates of the foci of the given ellipse. -/
theorem ellipse_foci_coordinates :
  (∀ (x y : ℝ), ellipse_eq x y → False) →
  ∃ (c : ℝ), c = 12 ∧ 
  ((0, c) = (0, 12) ∧ (0, -c) = (0, -12)) := 
by
  sorry

end ellipse_foci_coordinates_l0_792


namespace tangent_line_ratio_l0_217

variables {x1 x2 : ℝ}

theorem tangent_line_ratio (h1 : 2 * x1 = 3 * x2^2) (h2 : x1^2 = 2 * x2^3) : (x1 / x2) = 4 / 3 :=
by sorry

end tangent_line_ratio_l0_217


namespace Carol_cleaning_time_l0_948

theorem Carol_cleaning_time 
(Alice_time : ℕ) 
(Bob_time : ℕ) 
(Carol_time : ℕ) 
(h1 : Alice_time = 40) 
(h2 : Bob_time = 3 * Alice_time / 4) 
(h3 : Carol_time = 2 * Bob_time) :
  Carol_time = 60 := 
sorry

end Carol_cleaning_time_l0_948


namespace geometric_sequence_sum_l0_593

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) ^ 2 = a n * a (n + 2))
  (h_pos : ∀ n, 0 < a n) (h_given : a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) :
  a 3 + a 6 = 4 := 
sorry

end geometric_sequence_sum_l0_593


namespace quadratic_non_real_roots_l0_770

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l0_770


namespace polygon_area_is_12_l0_307

def polygon_vertices := [(0,0), (4,0), (4,4), (2,4), (2,2), (0,2)]

def area_of_polygon (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to compute the area (stub here for now)
  sorry

theorem polygon_area_is_12 :
  area_of_polygon polygon_vertices = 12 :=
by
  sorry

end polygon_area_is_12_l0_307


namespace seeds_planted_l0_277

theorem seeds_planted (seeds_per_bed : ℕ) (beds : ℕ) (total_seeds : ℕ) :
  seeds_per_bed = 10 → beds = 6 → total_seeds = seeds_per_bed * beds → total_seeds = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end seeds_planted_l0_277


namespace percentage_chromium_first_alloy_l0_989

theorem percentage_chromium_first_alloy
  (x : ℝ) (h : (x / 100) * 15 + (8 / 100) * 35 = (9.2 / 100) * 50) : x = 12 :=
sorry

end percentage_chromium_first_alloy_l0_989


namespace four_gt_sqrt_fifteen_l0_799

theorem four_gt_sqrt_fifteen : 4 > Real.sqrt 15 := 
sorry

end four_gt_sqrt_fifteen_l0_799


namespace calc_value_l0_817

theorem calc_value : (3000 * (3000 ^ 2999) * 2 = 2 * 3000 ^ 3000) := 
by
  sorry

end calc_value_l0_817


namespace oil_bill_for_january_l0_852

-- Definitions and conditions
def ratio_F_J (F J : ℝ) : Prop := F / J = 3 / 2
def ratio_F_M (F M : ℝ) : Prop := F / M = 4 / 5
def ratio_F_J_modified (F J : ℝ) : Prop := (F + 20) / J = 5 / 3
def ratio_F_M_modified (F M : ℝ) : Prop := (F + 20) / M = 2 / 3

-- The main statement to prove
theorem oil_bill_for_january (J F M : ℝ) 
  (h1 : ratio_F_J F J)
  (h2 : ratio_F_M F M)
  (h3 : ratio_F_J_modified F J)
  (h4 : ratio_F_M_modified F M) :
  J = 120 :=
sorry

end oil_bill_for_january_l0_852


namespace swimmer_distance_l0_726

theorem swimmer_distance :
  let swimmer_speed : ℝ := 3
  let current_speed : ℝ := 1.7
  let time : ℝ := 2.3076923076923075
  let effective_speed := swimmer_speed - current_speed
  let distance := effective_speed * time
  distance = 3 := by
sorry

end swimmer_distance_l0_726


namespace xander_pages_left_to_read_l0_915

theorem xander_pages_left_to_read :
  let total_pages := 500
  let read_first_night := 0.2 * 500
  let read_second_night := 0.2 * 500
  let read_third_night := 0.3 * 500
  total_pages - (read_first_night + read_second_night + read_third_night) = 150 :=
by 
  sorry

end xander_pages_left_to_read_l0_915


namespace bananas_per_box_l0_308

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) (h1 : total_bananas = 40) (h2 : num_boxes = 8) :
  total_bananas / num_boxes = 5 := by
  sorry

end bananas_per_box_l0_308


namespace odd_product_probability_lt_one_eighth_l0_881

theorem odd_product_probability_lt_one_eighth : 
  (∃ p : ℝ, p = (500 / 1000) * (499 / 999) * (498 / 998)) → p < 1 / 8 :=
by
  sorry

end odd_product_probability_lt_one_eighth_l0_881


namespace perimeter_triangle_l0_26

-- Definitions and conditions
def side1 : ℕ := 2
def side2 : ℕ := 5
def is_odd (n : ℕ) : Prop := n % 2 = 1
def valid_third_side (x : ℕ) : Prop := 3 < x ∧ x < 7 ∧ is_odd x

-- Theorem statement
theorem perimeter_triangle : ∃ (x : ℕ), valid_third_side x ∧ (side1 + side2 + x = 12) :=
by 
  sorry

end perimeter_triangle_l0_26


namespace trackball_mice_count_l0_959

theorem trackball_mice_count (total_sales wireless_share optical_share : ℕ) 
    (h_total : total_sales = 80)
    (h_wireless : wireless_share = total_sales / 2)
    (h_optical : optical_share = total_sales / 4):
    total_sales - (wireless_share + optical_share) = 20 :=
by
  sorry

end trackball_mice_count_l0_959


namespace george_final_score_l0_329

-- Definitions for points in the first half
def first_half_odd_points (questions : Nat) := 5 * 2
def first_half_even_points (questions : Nat) := 5 * 4
def first_half_bonus_points (questions : Nat) := 3 * 5
def first_half_points := first_half_odd_points 5 + first_half_even_points 5 + first_half_bonus_points 3

-- Definitions for points in the second half
def second_half_odd_points (questions : Nat) := 6 * 3
def second_half_even_points (questions : Nat) := 6 * 5
def second_half_bonus_points (questions : Nat) := 4 * 5
def second_half_points := second_half_odd_points 6 + second_half_even_points 6 + second_half_bonus_points 4

-- Definition of the total points
def total_points := first_half_points + second_half_points

-- The theorem statement to prove the total points
theorem george_final_score : total_points = 113 := by
  unfold total_points
  unfold first_half_points
  unfold second_half_points
  unfold first_half_odd_points first_half_even_points first_half_bonus_points
  unfold second_half_odd_points second_half_even_points second_half_bonus_points
  sorry

end george_final_score_l0_329


namespace thirty_percent_greater_l0_243

theorem thirty_percent_greater (x : ℝ) (h : x = 1.3 * 88) : x = 114.4 :=
sorry

end thirty_percent_greater_l0_243


namespace nonnegative_difference_of_roots_l0_93

theorem nonnegative_difference_of_roots :
  ∀ (x : ℝ), x^2 + 40 * x + 300 = -50 → (∃ a b : ℝ, x^2 + 40 * x + 350 = 0 ∧ x = a ∧ x = b ∧ |a - b| = 25) := 
by 
sorry

end nonnegative_difference_of_roots_l0_93


namespace water_level_drop_l0_84

theorem water_level_drop :
  (∀ x : ℝ, x > 0 → (x = 4) → (x > 0 → x = 4)) →
  ∃ y : ℝ, y < 0 ∧ (y = -1) :=
by
  sorry

end water_level_drop_l0_84


namespace wire_division_l0_298

theorem wire_division (L leftover total_length : ℝ) (seg1 seg2 : ℝ)
  (hL : L = 120 * 2)
  (hleftover : leftover = 2.4)
  (htotal : total_length = L + leftover)
  (hseg1 : seg1 = total_length / 3)
  (hseg2 : seg2 = total_length / 3) :
  seg1 = 80.8 ∧ seg2 = 80.8 := by
  sorry

end wire_division_l0_298


namespace rabbit_toy_cost_l0_414

theorem rabbit_toy_cost 
  (cost_pet_food : ℝ) 
  (cost_cage : ℝ) 
  (found_dollar : ℝ)
  (total_cost : ℝ) 
  (h1 : cost_pet_food = 5.79) 
  (h2 : cost_cage = 12.51)
  (h3 : found_dollar = 1.00)
  (h4 : total_cost = 24.81):
  ∃ (cost_rabbit_toy : ℝ), cost_rabbit_toy = 7.51 := by
  let cost_rabbit_toy := total_cost - (cost_pet_food + cost_cage) + found_dollar
  use cost_rabbit_toy
  sorry

end rabbit_toy_cost_l0_414


namespace find_other_integer_l0_232

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 150) (h2 : x = 15 ∨ y = 15) : y = 30 :=
by
  sorry

end find_other_integer_l0_232


namespace collinear_points_count_l0_570

-- Definitions for the problem conditions
def vertices_count := 8
def midpoints_count := 12
def face_centers_count := 6
def cube_center_count := 1
def total_points_count := vertices_count + midpoints_count + face_centers_count + cube_center_count

-- Lean statement to express the proof problem
theorem collinear_points_count :
  (total_points_count = 27) →
  (vertices_count = 8) →
  (midpoints_count = 12) →
  (face_centers_count = 6) →
  (cube_center_count = 1) →
  ∃ n, n = 49 :=
by
  intros
  existsi 49
  sorry

end collinear_points_count_l0_570


namespace no_rain_four_days_l0_921

-- Define the probability of rain on any given day
def prob_rain : ℚ := 2/3

-- Define the probability that it does not rain on any given day
def prob_no_rain : ℚ := 1 - prob_rain

-- Define the probability that it does not rain at all over four days
def prob_no_rain_four_days : ℚ := prob_no_rain^4

theorem no_rain_four_days : prob_no_rain_four_days = 1/81 := by
  sorry

end no_rain_four_days_l0_921


namespace anna_walk_distance_l0_983

theorem anna_walk_distance (d: ℚ) 
  (hd: 22 * 1.25 - 4 * 1.25 = d)
  (d2: d = 3.7): d = 3.7 :=
by 
  sorry

end anna_walk_distance_l0_983


namespace r_amount_l0_99

-- Let p, q, and r be the amounts of money p, q, and r have, respectively
variables (p q r : ℝ)

-- Given conditions: p + q + r = 5000 and r = (2 / 3) * (p + q)
theorem r_amount (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) :
  r = 2000 :=
sorry

end r_amount_l0_99


namespace product_of_x_y_l0_350

theorem product_of_x_y (x y : ℝ) (h1 : 3 * x + 4 * y = 60) (h2 : 6 * x - 4 * y = 12) : x * y = 72 :=
by
  sorry

end product_of_x_y_l0_350


namespace Carver_school_earnings_l0_265

noncomputable def total_earnings_Carver_school : ℝ :=
  let base_payment := 20
  let total_payment := 900
  let Allen_days := 7 * 3
  let Balboa_days := 5 * 6
  let Carver_days := 4 * 10
  let total_student_days := Allen_days + Balboa_days + Carver_days
  let adjusted_total_payment := total_payment - 3 * base_payment
  let daily_wage := adjusted_total_payment / total_student_days
  daily_wage * Carver_days

theorem Carver_school_earnings : 
  total_earnings_Carver_school = 369.6 := 
by 
  sorry

end Carver_school_earnings_l0_265


namespace dot_product_a_b_l0_648

-- Definitions for unit vectors e1 and e2 with given conditions
variables (e1 e2 : ℝ × ℝ)
variables (h_norm_e1 : e1.1^2 + e1.2^2 = 1) -- e1 is a unit vector
variables (h_norm_e2 : e2.1^2 + e2.2^2 = 1) -- e2 is a unit vector
variables (h_angle : e1.1 * e2.1 + e1.2 * e2.2 = -1 / 2) -- angle between e1 and e2 is 120 degrees

-- Definitions for vectors a and b
def a : ℝ × ℝ := (e1.1 + e2.1, e1.2 + e2.2)
def b : ℝ × ℝ := (e1.1 - 3 * e2.1, e1.2 - 3 * e2.2)

-- Theorem to prove
theorem dot_product_a_b : (a e1 e2) • (b e1 e2) = -1 :=
by
  sorry

end dot_product_a_b_l0_648


namespace correct_proposition_l0_929

def curve_is_ellipse (k : ℝ) : Prop :=
  9 < k ∧ k < 25

def curve_is_hyperbola_on_x_axis (k : ℝ) : Prop :=
  k < 9

theorem correct_proposition (k : ℝ) :
  (curve_is_ellipse k ∨ ¬ curve_is_ellipse k) ∧ 
  (curve_is_hyperbola_on_x_axis k ∨ ¬ curve_is_hyperbola_on_x_axis k) →
  (9 < k ∧ k < 25 → curve_is_ellipse k) ∧ 
  (curve_is_ellipse k ↔ (9 < k ∧ k < 25)) ∧ 
  (curve_is_hyperbola_on_x_axis k ↔ k < 9) → 
  (curve_is_ellipse k ∧ curve_is_hyperbola_on_x_axis k) :=
by
  sorry

end correct_proposition_l0_929


namespace must_be_true_if_not_all_electric_l0_511

variable (P : Type) (ElectricCar : P → Prop)

theorem must_be_true_if_not_all_electric (h : ¬ ∀ x : P, ElectricCar x) : 
  ∃ x : P, ¬ ElectricCar x :=
by 
sorry

end must_be_true_if_not_all_electric_l0_511


namespace pizza_diameter_increase_l0_888

theorem pizza_diameter_increase :
  ∀ (d D : ℝ), 
    (D / d)^2 = 1.96 → D = 1.4 * d := by
  sorry

end pizza_diameter_increase_l0_888


namespace percentage_of_360_is_165_6_l0_961

theorem percentage_of_360_is_165_6 :
  (165.6 / 360) * 100 = 46 :=
by
  sorry

end percentage_of_360_is_165_6_l0_961


namespace half_product_two_consecutive_integers_mod_3_l0_690

theorem half_product_two_consecutive_integers_mod_3 (A : ℤ) : 
  (A * (A + 1) / 2) % 3 = 0 ∨ (A * (A + 1) / 2) % 3 = 1 :=
sorry

end half_product_two_consecutive_integers_mod_3_l0_690


namespace tan_ratio_l0_757

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 2 := 
by
  sorry 

end tan_ratio_l0_757


namespace cost_price_article_l0_387
-- Importing the required library

-- Definition of the problem
theorem cost_price_article
  (C S C_new S_new : ℝ)
  (h1 : S = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : S_new = S - 1)
  (h4 : S_new = 1.045 * C) :
  C = 200 :=
by
  -- The proof is omitted
  sorry

end cost_price_article_l0_387


namespace distance_between_wheels_l0_666

theorem distance_between_wheels 
  (D : ℕ) 
  (back_perimeter : ℕ) (front_perimeter : ℕ) 
  (more_revolutions : ℕ)
  (h1 : back_perimeter = 9)
  (h2 : front_perimeter = 7)
  (h3 : more_revolutions = 10)
  (h4 : D / front_perimeter = D / back_perimeter + more_revolutions) : 
  D = 315 :=
by
  sorry

end distance_between_wheels_l0_666


namespace soda_cost_is_2_l0_922

noncomputable def cost_per_soda (total_bill : ℕ) (num_adults : ℕ) (num_children : ℕ) 
  (adult_meal_cost : ℕ) (child_meal_cost : ℕ) (num_sodas : ℕ) : ℕ :=
  (total_bill - (num_adults * adult_meal_cost + num_children * child_meal_cost)) / num_sodas

theorem soda_cost_is_2 :
  let total_bill := 60
  let num_adults := 6
  let num_children := 2
  let adult_meal_cost := 6
  let child_meal_cost := 4
  let num_sodas := num_adults + num_children
  cost_per_soda total_bill num_adults num_children adult_meal_cost child_meal_cost num_sodas = 2 :=
by
  -- proof goes here
  sorry

end soda_cost_is_2_l0_922


namespace simplify_fraction_l0_264

theorem simplify_fraction (d : ℝ) : (6 - 5 * d) / 9 - 3 = (-21 - 5 * d) / 9 :=
by
  sorry

end simplify_fraction_l0_264


namespace mingi_initial_tomatoes_l0_600

theorem mingi_initial_tomatoes (n m r : ℕ) (h1 : n = 15) (h2 : m = 20) (h3 : r = 6) : n * m + r = 306 := by
  sorry

end mingi_initial_tomatoes_l0_600


namespace sum_of_fractions_equals_16_l0_837

def list_of_fractions : List (ℚ) := [
  2 / 10,
  4 / 10,
  6 / 10,
  8 / 10,
  10 / 10,
  15 / 10,
  20 / 10,
  25 / 10,
  30 / 10,
  40 / 10
]

theorem sum_of_fractions_equals_16 : list_of_fractions.sum = 16 := by
  sorry

end sum_of_fractions_equals_16_l0_837


namespace product_closest_to_106_l0_405

theorem product_closest_to_106 :
  let product := (2.1 : ℝ) * (50.8 - 0.45)
  abs (product - 106) < abs (product - 105) ∧
  abs (product - 106) < abs (product - 107) ∧
  abs (product - 106) < abs (product - 108) ∧
  abs (product - 106) < abs (product - 110) :=
by
  sorry

end product_closest_to_106_l0_405


namespace manage_committee_combination_l0_324

theorem manage_committee_combination : (Nat.choose 20 3) = 1140 := by
  sorry

end manage_committee_combination_l0_324


namespace mean_score_74_l0_623

theorem mean_score_74 
  (M SD : ℝ)
  (h1 : 58 = M - 2 * SD)
  (h2 : 98 = M + 3 * SD) : 
  M = 74 :=
by
  sorry

end mean_score_74_l0_623


namespace minimum_cost_l0_22

noncomputable def f (x : ℝ) : ℝ := (1000 / (x + 5)) + 5 * x + (1 / 2) * (x^2 + 25)

theorem minimum_cost :
  (2 ≤ x ∧ x ≤ 8) →
  (f 5 = 150 ∧ (∀ y, 2 ≤ y ∧ y ≤ 8 → f y ≥ f 5)) :=
by
  intro h
  have f_exp : f x = (1000 / (x+5)) + 5*x + (1/2)*(x^2 + 25) := rfl
  sorry

end minimum_cost_l0_22


namespace eq_sqrt_pattern_l0_398

theorem eq_sqrt_pattern (a t : ℝ) (ha : a = 6) (ht : t = a^2 - 1) (h_pos : 0 < a ∧ 0 < t) :
  a + t = 41 := by
  sorry

end eq_sqrt_pattern_l0_398


namespace find_positive_m_has_exactly_single_solution_l0_655

theorem find_positive_m_has_exactly_single_solution :
  ∃ m : ℝ, 0 < m ∧ (∀ x : ℝ, 16 * x^2 + m * x + 4 = 0 → x = 16) :=
sorry

end find_positive_m_has_exactly_single_solution_l0_655


namespace initially_marked_points_l0_412

theorem initially_marked_points (k : ℕ) (h : 4 * k - 3 = 101) : k = 26 :=
by
  sorry

end initially_marked_points_l0_412


namespace conscript_from_western_village_l0_589

/--
Given:
- The population of the northern village is 8758
- The population of the western village is 7236
- The population of the southern village is 8356
- The total number of conscripts needed is 378

Prove that the number of people to be conscripted from the western village is 112.
-/
theorem conscript_from_western_village (hnorth : ℕ) (hwest : ℕ) (hsouth : ℕ) (hconscripts : ℕ)
    (htotal : hnorth + hwest + hsouth = 24350) :
    let prop := (hwest / (hnorth + hwest + hsouth)) * hconscripts
    hnorth = 8758 → hwest = 7236 → hsouth = 8356 → hconscripts = 378 → prop = 112 :=
by
  intros
  simp_all
  sorry

end conscript_from_western_village_l0_589


namespace flour_in_cupboard_l0_151

theorem flour_in_cupboard :
  let flour_on_counter := 100
  let flour_in_pantry := 100
  let flour_per_loaf := 200
  let loaves := 2
  let total_flour_needed := loaves * flour_per_loaf
  let flour_outside_cupboard := flour_on_counter + flour_in_pantry
  let flour_in_cupboard := total_flour_needed - flour_outside_cupboard
  flour_in_cupboard = 200 :=
by
  sorry

end flour_in_cupboard_l0_151


namespace find_n_cubes_l0_359

theorem find_n_cubes (n : ℕ) (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h1 : 837 + n = y^3) (h2 : 837 - n = x^3) : n = 494 :=
by {
  sorry
}

end find_n_cubes_l0_359


namespace monotonic_intervals_range_of_a_min_value_of_c_l0_842

noncomputable def f (a c x : ℝ) : ℝ :=
  a * Real.log x + (x - c) * abs (x - c)

-- 1. Monotonic intervals
theorem monotonic_intervals (a c : ℝ) (ha : a = -3 / 4) (hc : c = 1 / 4) :
  ((∀ x, 0 < x ∧ x < 3 / 4 → f a c x > f a c (x - 1)) ∧ (∀ x, 3 / 4 < x → f a c x > f a c (x - 1))) :=
sorry

-- 2. Range of values for a
theorem range_of_a (a c : ℝ) (hc : c = a / 2 + 1) (h : ∀ x > c, f a c x ≥ 1 / 4) :
  -2 < a ∧ a ≤ -1 :=
sorry

-- 3. Minimum value of c
theorem min_value_of_c (a c x1 x2 : ℝ) (hx1 : x1 = Real.sqrt (-a / 2)) (hx2 : x2 = c)
  (h_tangents_perpendicular : f a c x1 * f a c x2 = -1) :
  c = 3 * Real.sqrt 3 / 2 :=
sorry

end monotonic_intervals_range_of_a_min_value_of_c_l0_842


namespace cubic_inequality_l0_491

theorem cubic_inequality (x p q : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 := 
  sorry

end cubic_inequality_l0_491


namespace average_of_consecutive_numbers_l0_62

-- Define the 7 consecutive numbers and their properties
variables (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (e : ℝ) (f : ℝ) (g : ℝ)

-- Conditions given in the problem
def consecutive_numbers (a b c d e f g : ℝ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 ∧ g = a + 6

def percent_relationship (a g : ℝ) : Prop :=
  g = 1.5 * a

-- The proof problem
theorem average_of_consecutive_numbers (a b c d e f g : ℝ)
  (h1 : consecutive_numbers a b c d e f g)
  (h2 : percent_relationship a g) :
  (a + b + c + d + e + f + g) / 7 = 15 :=
by {
  sorry -- Proof goes here
}

-- To ensure it passes the type checker but without providing the actual proof, we use sorry.

end average_of_consecutive_numbers_l0_62


namespace average_first_21_multiples_of_4_l0_957

-- Define conditions
def n : ℕ := 21
def a1 : ℕ := 4
def an : ℕ := 4 * n
def sum_series (n a1 an : ℕ) : ℕ := (n * (a1 + an)) / 2

-- The problem statement in Lean 4
theorem average_first_21_multiples_of_4 : 
    (sum_series n a1 an) / n = 44 :=
by
  -- skipping the proof
  sorry

end average_first_21_multiples_of_4_l0_957


namespace joshInitialMarbles_l0_724

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

end joshInitialMarbles_l0_724


namespace triangle_DFG_area_l0_931

theorem triangle_DFG_area (a b x y : ℝ) (h_ab : a * b = 20) (h_xy : x * y = 8) : 
  (a * b - x * y) / 2 = 6 := 
by
  sorry

end triangle_DFG_area_l0_931


namespace count_integers_between_bounds_l0_504

theorem count_integers_between_bounds : 
  ∃ n : ℤ, n = 15 ∧ ∀ x : ℤ, 3 < Real.sqrt (x : ℝ) ∧ Real.sqrt (x : ℝ) < 5 → 10 ≤ x ∧ x ≤ 24 :=
by
  sorry

end count_integers_between_bounds_l0_504


namespace largest_mersenne_prime_less_than_500_l0_284

-- Define what it means for a number to be prime
def is_prime (p : ℕ) : Prop :=
p > 1 ∧ ∀ (n : ℕ), n > 1 ∧ n < p → ¬ (p % n = 0)

-- Define what a Mersenne prime is
def is_mersenne_prime (m : ℕ) : Prop :=
∃ n : ℕ, is_prime n ∧ m = 2^n - 1

-- We state the main theorem we want to prove
theorem largest_mersenne_prime_less_than_500 : ∀ (m : ℕ), is_mersenne_prime m ∧ m < 500 → m ≤ 127 :=
by 
  sorry

end largest_mersenne_prime_less_than_500_l0_284


namespace solve_equation_l0_278

theorem solve_equation (x : ℝ) : 
  (x - 1)^2 + 2 * x * (x - 1) = 0 ↔ x = 1 ∨ x = 1 / 3 :=
by sorry

end solve_equation_l0_278


namespace bank_teller_rolls_of_coins_l0_910

theorem bank_teller_rolls_of_coins (tellers : ℕ) (coins_per_roll : ℕ) (total_coins : ℕ) (h_tellers : tellers = 4) (h_coins_per_roll : coins_per_roll = 25) (h_total_coins : total_coins = 1000) : 
  (total_coins / tellers) / coins_per_roll = 10 :=
by 
  sorry

end bank_teller_rolls_of_coins_l0_910


namespace random_point_between_R_S_l0_777

theorem random_point_between_R_S {P Q R S : ℝ} (PQ PR RS : ℝ) (h1 : PQ = 4 * PR) (h2 : PQ = 8 * RS) :
  let PS := PR + RS
  let probability := RS / PQ
  probability = 5 / 8 :=
by
  let PS := PR + RS
  let probability := RS / PQ
  sorry

end random_point_between_R_S_l0_777


namespace square_table_production_l0_812

theorem square_table_production (x y : ℝ) :
  x + y = 5 ∧ 50 * x * 4 = 300 * y → 
  x = 3 ∧ y = 2 ∧ 50 * x = 150 :=
by
  sorry

end square_table_production_l0_812


namespace fifth_rectangle_is_square_l0_60

-- Define the conditions
variables (s : ℝ) (a b : ℝ)
variables (R1 R2 R3 R4 : Set (ℝ × ℝ))
variables (R5 : Set (ℝ × ℝ))

-- Assume the areas of the corner rectangles are equal
def equal_area (R : Set (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), R = {p | p.1 < a ∧ p.2 < b} ∧ a * b = k

-- State the conditions
axiom h1 : equal_area R1 a
axiom h2 : equal_area R2 a
axiom h3 : equal_area R3 a
axiom h4 : equal_area R4 a

axiom h5 : ∀ (p : ℝ × ℝ), p ∈ R5 → p.1 ≠ 0 → p.2 ≠ 0

-- Prove that the fifth rectangle is a square
theorem fifth_rectangle_is_square : ∃ c : ℝ, ∀ r1 r2, r1 ∈ R5 → r2 ∈ R5 → r1.1 - r2.1 = c ∧ r1.2 - r2.2 = c :=
by sorry

end fifth_rectangle_is_square_l0_60


namespace work_completion_time_l0_997

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 5) (hC : C = 1 / 20) :
  1 / (A + B + C) = 2 :=
by
  -- Proof goes here
  sorry

end work_completion_time_l0_997


namespace determine_y_value_l0_900

theorem determine_y_value {k y : ℕ} (h1 : k > 0) (h2 : y > 0) (hk : k < 10) (hy : y < 10) :
  (8 * 100 + k * 10 + 8) + (k * 100 + 8 * 10 + 8) - (1 * 100 + 6 * 10 + y * 1) = 8 * 100 + k * 10 + 8 → 
  y = 9 :=
by
  sorry

end determine_y_value_l0_900


namespace tingting_solution_correct_l0_928

noncomputable def product_of_square_roots : ℝ :=
  (Real.sqrt 8) * (Real.sqrt 18)

theorem tingting_solution_correct : product_of_square_roots = 12 := by
  sorry

end tingting_solution_correct_l0_928


namespace dolphins_trained_next_month_l0_283

theorem dolphins_trained_next_month
  (total_dolphins : ℕ) 
  (one_fourth_fully_trained : ℚ) 
  (two_thirds_in_training : ℚ)
  (h1 : total_dolphins = 20)
  (h2 : one_fourth_fully_trained = 1 / 4) 
  (h3 : two_thirds_in_training = 2 / 3) :
  (total_dolphins - total_dolphins * one_fourth_fully_trained) * two_thirds_in_training = 10 := 
by 
  sorry

end dolphins_trained_next_month_l0_283


namespace youtube_dislikes_l0_638

def initial_dislikes (likes : ℕ) : ℕ := (likes / 2) + 100

def new_dislikes (initial : ℕ) : ℕ := initial + 1000

theorem youtube_dislikes
  (likes : ℕ)
  (h_likes : likes = 3000) :
  new_dislikes (initial_dislikes likes) = 2600 :=
by
  sorry

end youtube_dislikes_l0_638


namespace container_volume_ratio_l0_653

theorem container_volume_ratio (A B : ℕ) 
  (h1 : (3 / 4 : ℚ) * A = (5 / 8 : ℚ) * B) :
  (A : ℚ) / B = 5 / 6 :=
by
  admit
-- sorry

end container_volume_ratio_l0_653


namespace maximum_value_expression_l0_208

theorem maximum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
by
  sorry

end maximum_value_expression_l0_208


namespace find_value_of_a_l0_661

-- Define variables and constants
variable (a : ℚ)
variable (b : ℚ := 3 * a)
variable (c : ℚ := 4 * b)
variable (d : ℚ := 6 * c)
variable (total : ℚ := 186)

-- State the theorem
theorem find_value_of_a (h : a + b + c + d = total) : a = 93 / 44 := by
  sorry

end find_value_of_a_l0_661


namespace sin_double_angle_identity_l0_105

theorem sin_double_angle_identity (α : ℝ) (h : Real.cos α = 1 / 4) : 
  Real.sin (π / 2 - 2 * α) = -7 / 8 :=
by 
  sorry

end sin_double_angle_identity_l0_105


namespace problem1_problem2_problem3_problem4_l0_365

theorem problem1 : (70.8 - 1.25 - 1.75 = 67.8) := sorry

theorem problem2 : ((8 + 0.8) * 1.25 = 11) := sorry

theorem problem3 : (125 * 0.48 = 600) := sorry

theorem problem4 : (6.7 * (9.3 * (6.2 + 1.7)) = 554.559) := sorry

end problem1_problem2_problem3_problem4_l0_365


namespace no_integer_solutions_for_2891_l0_106

theorem no_integer_solutions_for_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) :=
sorry

end no_integer_solutions_for_2891_l0_106


namespace find_number_l0_665

theorem find_number (number : ℝ) : 469138 * number = 4690910862 → number = 10000.1 :=
by
  sorry

end find_number_l0_665


namespace math_problem_l0_492

noncomputable def proof : Prop :=
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  ( (1 / a + 1 / b) / (1 / a - 1 / b) = 1001 ) →
  ((a + b) / (a - b) = 1001)

theorem math_problem : proof := 
  by
    intros a b h₁ h₂ h₃
    sorry

end math_problem_l0_492


namespace cone_bead_path_l0_188

theorem cone_bead_path (r h : ℝ) (h_sqrt : h / r = 3 * Real.sqrt 11) : 3 + 11 = 14 := by
  sorry

end cone_bead_path_l0_188


namespace skeleton_ratio_l0_541

theorem skeleton_ratio (W M C : ℕ) 
  (h1 : W + M + C = 20)
  (h2 : M = C)
  (h3 : 20 * W + 25 * M + 10 * C = 375) :
  (W : ℚ) / (W + M + C) = 1 / 2 :=
by
  sorry

end skeleton_ratio_l0_541


namespace parallel_lines_l0_169

def line1 (x : ℝ) : ℝ := 5 * x + 3
def line2 (x k : ℝ) : ℝ := 3 * k * x + 7

theorem parallel_lines (k : ℝ) : (∀ x : ℝ, line1 x = line2 x k) → k = 5 / 3 := 
by
  intros h_parallel
  sorry

end parallel_lines_l0_169


namespace least_common_multiple_first_ten_integers_l0_59

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l0_59


namespace vertical_asymptote_condition_l0_273

theorem vertical_asymptote_condition (c : ℝ) :
  (∀ x : ℝ, (x = 3 ∨ x = -6) → (x^2 - x + c = 0)) → 
  (c = -6 ∨ c = -42) :=
by
  sorry

end vertical_asymptote_condition_l0_273


namespace solve_correct_problems_l0_15

theorem solve_correct_problems (x : ℕ) (h1 : 3 * x + x = 120) : x = 30 :=
by
  sorry

end solve_correct_problems_l0_15


namespace perfect_square_iff_n_eq_one_l0_857

theorem perfect_square_iff_n_eq_one (n : ℕ) : ∃ m : ℕ, n^2 + 3 * n = m^2 ↔ n = 1 := by
  sorry

end perfect_square_iff_n_eq_one_l0_857


namespace fraction_representation_of_2_375_l0_490

theorem fraction_representation_of_2_375 : 2.375 = 19 / 8 := by
  sorry

end fraction_representation_of_2_375_l0_490


namespace functional_eq_app_only_solutions_l0_841

noncomputable def f : Real → Real := sorry

theorem functional_eq_app (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) :
  f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2 :=
sorry

theorem only_solutions (f : ℝ → ℝ) (hf : ∀ n : ℕ, ∀ x : Fin n → ℝ, (∀ i, 0 ≤ x i) → f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2) :
  f = (fun x => 0) ∨ f = (fun x => x) :=
sorry

end functional_eq_app_only_solutions_l0_841


namespace remainder_pow_2023_l0_834

theorem remainder_pow_2023 (a b : ℕ) (h : b = 2023) : (3 ^ b) % 11 = 5 :=
by
  sorry

end remainder_pow_2023_l0_834


namespace greatest_multiple_less_150_l0_399

/-- Define the LCM of two natural numbers -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem greatest_multiple_less_150 (x y : ℕ) (h1 : x = 15) (h2 : y = 20) : 
  (∃ m : ℕ, LCM x y * m < 150 ∧ ∀ n : ℕ, LCM x y * n < 150 → LCM x y * n ≤ LCM x y * m) ∧ 
  (∃ m : ℕ, LCM x y * m = 120) :=
by
  sorry

end greatest_multiple_less_150_l0_399


namespace central_cell_value_l0_430

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l0_430


namespace transaction_loss_l0_330

theorem transaction_loss 
  (sell_price_house sell_price_store : ℝ)
  (cost_price_house cost_price_store : ℝ)
  (house_loss_percent store_gain_percent : ℝ)
  (house_loss_eq : sell_price_house = (4/5) * cost_price_house)
  (store_gain_eq : sell_price_store = (6/5) * cost_price_store)
  (sell_prices_eq : sell_price_house = 12000 ∧ sell_price_store = 12000)
  (house_loss_percent_eq : house_loss_percent = 0.20)
  (store_gain_percent_eq : store_gain_percent = 0.20) :
  cost_price_house + cost_price_store - (sell_price_house + sell_price_store) = 1000 :=
by
  sorry

end transaction_loss_l0_330


namespace find_a_given_conditions_l0_137

theorem find_a_given_conditions (a : ℤ)
  (hA : ∃ (x : ℤ), x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)
  (hA_contains_minus3 : ∃ (x : ℤ), (-3 = x) ∧ (x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)) : a = -3 := 
by
  sorry

end find_a_given_conditions_l0_137


namespace loop_execution_count_l0_80

theorem loop_execution_count : 
  ∀ (a b : ℤ), a = 2 → b = 20 → (b - a + 1) = 19 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- Here, we explicitly compute (20 - 2 + 1) = 19
  exact rfl

end loop_execution_count_l0_80


namespace earthquake_relief_team_selection_l0_472

theorem earthquake_relief_team_selection : 
    ∃ (ways : ℕ), ways = 590 ∧ 
      ∃ (orthopedic neurosurgeon internist : ℕ), 
      orthopedic + neurosurgeon + internist = 5 ∧ 
      1 ≤ orthopedic ∧ 1 ≤ neurosurgeon ∧ 1 ≤ internist ∧
      orthopedic ≤ 3 ∧ neurosurgeon ≤ 4 ∧ internist ≤ 5 := 
  sorry

end earthquake_relief_team_selection_l0_472


namespace find_number_of_children_l0_955

theorem find_number_of_children (N : ℕ) (B : ℕ) 
    (h1 : B = 2 * N) 
    (h2 : B = 4 * (N - 160)) 
    : N = 320 := 
by
  sorry

end find_number_of_children_l0_955


namespace sin_690_eq_neg_half_l0_832

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end sin_690_eq_neg_half_l0_832


namespace lending_period_C_l0_988

theorem lending_period_C (R : ℝ) (P_B P_C T_B I : ℝ) (h1 : R = 13.75) (h2 : P_B = 4000) (h3 : P_C = 2000) (h4 : T_B = 2) (h5 : I = 2200) : 
  ∃ T_C : ℝ, T_C = 4 :=
by
  -- Definitions and known facts
  let I_B := (P_B * R * T_B) / 100
  let I_C := I - I_B
  let T_C := I_C / ((P_C * R) / 100)
  -- Prove the target
  use T_C
  sorry

end lending_period_C_l0_988


namespace max_third_side_of_triangle_l0_510

theorem max_third_side_of_triangle (a b : ℕ) (h₁ : a = 7) (h₂ : b = 11) : 
  ∃ c : ℕ, c < a + b ∧ c = 17 :=
by 
  sorry

end max_third_side_of_triangle_l0_510


namespace lori_beanie_babies_times_l0_522

theorem lori_beanie_babies_times (l s : ℕ) (h1 : l = 300) (h2 : l + s = 320) : l = 15 * s :=
by
  sorry

end lori_beanie_babies_times_l0_522


namespace total_weight_of_rice_l0_181

theorem total_weight_of_rice :
  (29 * 4) / 16 = 7.25 := by
sorry

end total_weight_of_rice_l0_181


namespace incorrect_statement_D_l0_932

theorem incorrect_statement_D : ∃ a : ℝ, a > 0 ∧ (1 - 1 / (2 * a) < 0) := by
  sorry

end incorrect_statement_D_l0_932


namespace nina_expected_tomato_harvest_l0_9

noncomputable def expected_tomato_harvest 
  (garden_length : ℝ) (garden_width : ℝ) 
  (plants_per_sq_ft : ℝ) (tomatoes_per_plant : ℝ) : ℝ :=
  garden_length * garden_width * plants_per_sq_ft * tomatoes_per_plant

theorem nina_expected_tomato_harvest : 
  expected_tomato_harvest 10 20 5 10 = 10000 :=
by
  -- Proof would go here
  sorry

end nina_expected_tomato_harvest_l0_9


namespace Monica_tiles_count_l0_750

noncomputable def total_tiles (length width : ℕ) := 
  let double_border_tiles := (2 * ((length - 4) + (width - 4)) + 8)
  let inner_area := (length - 4) * (width - 4)
  let three_foot_tiles := (inner_area + 8) / 9
  double_border_tiles + three_foot_tiles

theorem Monica_tiles_count : total_tiles 18 24 = 183 := 
by
  sorry

end Monica_tiles_count_l0_750


namespace cookie_baking_time_l0_846

theorem cookie_baking_time 
  (total_time : ℕ) 
  (white_icing_time: ℕ)
  (chocolate_icing_time: ℕ) 
  (total_icing_time : white_icing_time + chocolate_icing_time = 60)
  (total_cooking_time : total_time = 120):

  (total_time - (white_icing_time + chocolate_icing_time) = 60) :=
by
  sorry

end cookie_baking_time_l0_846


namespace find_A_l0_349

def hash_relation (A B : ℕ) : ℕ := A^2 + B^2

theorem find_A (A : ℕ) (h1 : hash_relation A 7 = 218) : A = 13 := 
by sorry

end find_A_l0_349


namespace solve_stream_speed_l0_607

noncomputable def boat_travel (v : ℝ) : Prop :=
  let downstream_speed := 12 + v
  let upstream_speed := 12 - v
  let downstream_time := 60 / downstream_speed
  let upstream_time := 60 / upstream_speed
  upstream_time - downstream_time = 2

theorem solve_stream_speed : ∃ v : ℝ, boat_travel v ∧ v = 2.31 :=
by {
  sorry
}

end solve_stream_speed_l0_607


namespace fourth_grade_students_l0_870

theorem fourth_grade_students (initial_students left_students new_students final_students : ℕ) 
    (h1 : initial_students = 33) 
    (h2 : left_students = 18) 
    (h3 : new_students = 14) 
    (h4 : final_students = initial_students - left_students + new_students) :
    final_students = 29 := 
by 
    sorry

end fourth_grade_students_l0_870


namespace element_in_set_l0_410

def M : Set (ℤ × ℤ) := {(1, 2)}

theorem element_in_set : (1, 2) ∈ M :=
by
  sorry

end element_in_set_l0_410


namespace combined_cost_l0_808

theorem combined_cost (wallet_cost : ℕ) (purse_cost : ℕ)
    (h_wallet_cost : wallet_cost = 22)
    (h_purse_cost : purse_cost = 4 * wallet_cost - 3) :
    wallet_cost + purse_cost = 107 :=
by
  rw [h_wallet_cost, h_purse_cost]
  norm_num
  sorry

end combined_cost_l0_808


namespace zoey_holidays_l0_758

def visits_per_year (visits_per_month : ℕ) (months_per_year : ℕ) : ℕ :=
  visits_per_month * months_per_year

def visits_every_two_months (months_per_year : ℕ) : ℕ :=
  months_per_year / 2

def visits_every_four_months (visits_per_period : ℕ) (periods_per_year : ℕ) : ℕ :=
  visits_per_period * periods_per_year

theorem zoey_holidays (visits_per_month_first : ℕ) 
                      (months_per_year : ℕ) 
                      (visits_per_period_third : ℕ) 
                      (periods_per_year : ℕ) : 
  visits_per_year visits_per_month_first months_per_year 
  + visits_every_two_months months_per_year 
  + visits_every_four_months visits_per_period_third periods_per_year = 39 := 
  by 
  sorry

end zoey_holidays_l0_758


namespace find_values_l0_614

noncomputable def value_of_a (a : ℚ) : Prop :=
  4 + a = 2

noncomputable def value_of_b (b : ℚ) : Prop :=
  b^2 - 2 * b = 24 ∧ 4 * b^2 - 2 * b = 72

theorem find_values (a b : ℚ) (h1 : value_of_a a) (h2 : value_of_b b) :
  a = -2 ∧ b = -4 :=
by
  sorry

end find_values_l0_614


namespace paperclips_in_64_volume_box_l0_428

def volume_16 : ℝ := 16
def volume_32 : ℝ := 32
def volume_64 : ℝ := 64
def paperclips_50 : ℝ := 50
def paperclips_100 : ℝ := 100

theorem paperclips_in_64_volume_box :
  ∃ (k p : ℝ), 
  (paperclips_50 = k * volume_16^p) ∧ 
  (paperclips_100 = k * volume_32^p) ∧ 
  (200 = k * volume_64^p) :=
by
  sorry

end paperclips_in_64_volume_box_l0_428


namespace accuracy_l0_375

-- Given number and accuracy statement
def given_number : ℝ := 3.145 * 10^8
def expanded_form : ℕ := 314500000

-- Proof statement: the number is accurate to the hundred thousand's place
theorem accuracy (h : given_number = expanded_form) : 
  ∃ n : ℕ, expanded_form = n * 10^5 ∧ (n % 10) ≠ 0 := 
by
  sorry

end accuracy_l0_375


namespace min_inv_sum_four_l0_670

theorem min_inv_sum_four (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  4 ≤ (1 / a + 1 / b) := 
sorry

end min_inv_sum_four_l0_670


namespace alfonso_initial_money_l0_940

def daily_earnings : ℕ := 6
def days_per_week : ℕ := 5
def total_weeks : ℕ := 10
def cost_of_helmet : ℕ := 340

theorem alfonso_initial_money :
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  cost_of_helmet - total_earnings = 40 :=
by
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  show cost_of_helmet - total_earnings = 40
  sorry

end alfonso_initial_money_l0_940


namespace Gwendolyn_will_take_50_hours_to_read_l0_299

def GwendolynReadingTime (sentences_per_hour : ℕ) (sentences_per_paragraph : ℕ) (paragraphs_per_page : ℕ) (pages : ℕ) : ℕ :=
  (sentences_per_paragraph * paragraphs_per_page * pages) / sentences_per_hour

theorem Gwendolyn_will_take_50_hours_to_read 
  (h1 : 200 = 200)
  (h2 : 10 = 10)
  (h3 : 20 = 20)
  (h4 : 50 = 50) :
  GwendolynReadingTime 200 10 20 50 = 50 := by
  sorry

end Gwendolyn_will_take_50_hours_to_read_l0_299


namespace find_x_l0_4

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 1)

theorem find_x (x : ℝ) (h : deriv f x = x) : x = 0 ∨ x = 1 :=
by
  sorry

end find_x_l0_4


namespace sum_of_money_l0_702

theorem sum_of_money (jimin_100_won : ℕ) (jimin_50_won : ℕ) (seokjin_100_won : ℕ) (seokjin_10_won : ℕ) 
  (h1 : jimin_100_won = 5) (h2 : jimin_50_won = 1) (h3 : seokjin_100_won = 2) (h4 : seokjin_10_won = 7) :
  jimin_100_won * 100 + jimin_50_won * 50 + seokjin_100_won * 100 + seokjin_10_won * 10 = 820 :=
by
  sorry

end sum_of_money_l0_702


namespace at_least_one_closed_l0_401

theorem at_least_one_closed {T V : Set ℤ} (hT : T.Nonempty) (hV : V.Nonempty) (h_disjoint : ∀ x, x ∈ T → x ∉ V)
  (h_union : ∀ x, x ∈ T ∨ x ∈ V)
  (hT_closed : ∀ a b c, a ∈ T → b ∈ T → c ∈ T → a * b * c ∈ T)
  (hV_closed : ∀ x y z, x ∈ V → y ∈ V → z ∈ V → x * y * z ∈ V) :
  (∀ a b, a ∈ T → b ∈ T → a * b ∈ T) ∨ (∀ x y, x ∈ V → y ∈ V → x * y ∈ V) := sorry

end at_least_one_closed_l0_401


namespace number_of_DVDs_sold_l0_694

theorem number_of_DVDs_sold (C D: ℤ) (h₁ : D = 16 * C / 10) (h₂ : D + C = 273) : D = 168 := 
sorry

end number_of_DVDs_sold_l0_694


namespace bread_carriers_l0_549

-- Definitions for the number of men, women, and children
variables (m w c : ℕ)

-- Conditions from the problem
def total_people := m + w + c = 12
def total_bread := 8 * m + 2 * w + c = 48

-- Theorem to prove the correct number of men, women, and children
theorem bread_carriers (h1 : total_people m w c) (h2 : total_bread m w c) : 
  m = 5 ∧ w = 1 ∧ c = 6 :=
sorry

end bread_carriers_l0_549


namespace simplify_and_evaluate_l0_943

theorem simplify_and_evaluate (a : ℕ) (h : a = 2) : 
  (1 - (1 : ℚ) / (a + 1)) / (a / ((a * a) - 1)) = 1 := by
  sorry

end simplify_and_evaluate_l0_943


namespace fourth_person_height_is_82_l0_207

theorem fourth_person_height_is_82 (H : ℕ)
    (h1: (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 76)
    (h_diff1: H + 2 - H = 2)
    (h_diff2: H + 4 - (H + 2) = 2)
    (h_diff3: H + 10 - (H + 4) = 6) :
  (H + 10) = 82 := 
sorry

end fourth_person_height_is_82_l0_207


namespace sum_of_digits_base2_310_l0_849

-- We define what it means to convert a number to binary and sum its digits.
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (Nat.digits 2 n).sum

-- The main statement of the problem.
theorem sum_of_digits_base2_310 :
  sum_of_binary_digits 310 = 5 :=
by
  sorry

end sum_of_digits_base2_310_l0_849


namespace find_people_got_off_at_first_stop_l0_269

def total_seats (rows : ℕ) (seats_per_row : ℕ) : ℕ :=
  rows * seats_per_row

def occupied_seats (total_seats : ℕ) (initial_people : ℕ) : ℕ :=
  total_seats - initial_people

def occupied_seats_after_first_stop (initial_people : ℕ) (boarded_first_stop : ℕ) (got_off_first_stop : ℕ) : ℕ :=
  (initial_people + boarded_first_stop) - got_off_first_stop

def occupied_seats_after_second_stop (occupied_after_first_stop : ℕ) (boarded_second_stop : ℕ) (got_off_second_stop : ℕ) : ℕ :=
  (occupied_after_first_stop + boarded_second_stop) - got_off_second_stop

theorem find_people_got_off_at_first_stop
  (initial_people : ℕ := 16)
  (boarded_first_stop : ℕ := 15)
  (total_rows : ℕ := 23)
  (seats_per_row : ℕ := 4)
  (boarded_second_stop : ℕ := 17)
  (got_off_second_stop : ℕ := 10)
  (empty_seats_after_second_stop : ℕ := 57)
  : ∃ x, (occupied_seats_after_second_stop (occupied_seats_after_first_stop initial_people boarded_first_stop x) boarded_second_stop got_off_second_stop) = total_seats total_rows seats_per_row - empty_seats_after_second_stop :=
by
  sorry

end find_people_got_off_at_first_stop_l0_269


namespace shop_dimension_is_100_l0_450

-- Given conditions
def monthly_rent : ℕ := 1300
def annual_rent_per_sqft : ℕ := 156

-- Define annual rent
def annual_rent : ℕ := monthly_rent * 12

-- Define dimension to prove
def dimension_of_shop : ℕ := annual_rent / annual_rent_per_sqft

-- The theorem statement
theorem shop_dimension_is_100 :
  dimension_of_shop = 100 :=
by
  sorry

end shop_dimension_is_100_l0_450


namespace fewer_onions_grown_l0_174

def num_tomatoes := 2073
def num_cobs_of_corn := 4112
def num_onions := 985

theorem fewer_onions_grown : num_tomatoes + num_cobs_of_corn - num_onions = 5200 := by
  sorry

end fewer_onions_grown_l0_174


namespace max_consecutive_irreducible_l0_507

-- Define what it means for a five-digit number to be irreducible
def is_irreducible (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ¬∃ x y : ℕ, 100 ≤ x ∧ x < 1000 ∧ 100 ≤ y ∧ y < 1000 ∧ x * y = n

-- Prove the maximum number of consecutive irreducible five-digit numbers is 99
theorem max_consecutive_irreducible : ∃ m : ℕ, m = 99 ∧ 
  (∀ n : ℕ, (n ≤ 99901) → (∀ k : ℕ, (n ≤ k ∧ k < n + m) → is_irreducible k)) ∧
  (∀ x y : ℕ, x > 99 → ∀ n : ℕ, (n ≤ 99899) → (∀ k : ℕ, (n ≤ k ∧ k < n + x) → is_irreducible k) → x = 99) :=
by
  sorry

end max_consecutive_irreducible_l0_507


namespace strictly_decreasing_exponential_l0_403

theorem strictly_decreasing_exponential (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2*a - 1)^x > (2*a - 1)^y) → (1/2 < a ∧ a < 1) :=
by
  sorry

end strictly_decreasing_exponential_l0_403


namespace equal_real_roots_iff_c_is_nine_l0_447

theorem equal_real_roots_iff_c_is_nine (c : ℝ) : (∃ x : ℝ, x^2 + 6 * x + c = 0 ∧ ∃ Δ, Δ = 6^2 - 4 * 1 * c ∧ Δ = 0) ↔ c = 9 :=
by
  sorry

end equal_real_roots_iff_c_is_nine_l0_447


namespace product_of_two_numbers_l0_270

theorem product_of_two_numbers (a b : ℤ) (h1 : lcm a b = 72) (h2 : gcd a b = 8) :
  a * b = 576 :=
sorry

end product_of_two_numbers_l0_270


namespace Mary_work_days_l0_939

theorem Mary_work_days :
  ∀ (M : ℝ), (∀ R : ℝ, R = M / 1.30) → (R = 20) → M = 26 :=
by
  intros M h1 h2
  sorry

end Mary_work_days_l0_939


namespace trig_inequality_l0_49

theorem trig_inequality : Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := by
  sorry

end trig_inequality_l0_49


namespace time_since_production_approximate_l0_276

noncomputable def solve_time (N N₀ : ℝ) (t : ℝ) : Prop :=
  N = N₀ * (1 / 2) ^ (t / 5730) ∧
  N / N₀ = 3 / 8 ∧
  t = 8138

theorem time_since_production_approximate
  (N N₀ : ℝ)
  (h_decay : N = N₀ * (1 / 2) ^ (t / 5730))
  (h_ratio : N / N₀ = 3 / 8) :
  t = 8138 := 
sorry

end time_since_production_approximate_l0_276


namespace johns_out_of_pocket_l0_302

noncomputable def total_cost_after_discounts (computer_cost gaming_chair_cost accessories_cost : ℝ) 
  (comp_discount gaming_discount : ℝ) (tax : ℝ) : ℝ :=
  let comp_price := computer_cost * (1 - comp_discount)
  let chair_price := gaming_chair_cost * (1 - gaming_discount)
  let pre_tax_total := comp_price + chair_price + accessories_cost
  pre_tax_total * (1 + tax)

noncomputable def total_selling_price (playstation_value playstation_discount bicycle_price : ℝ) (exchange_rate : ℝ) : ℝ :=
  let playstation_price := playstation_value * (1 - playstation_discount)
  (playstation_price * exchange_rate) / exchange_rate + bicycle_price

theorem johns_out_of_pocket (computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax 
  playstation_value playstation_discount bicycle_price exchange_rate : ℝ) :
  computer_cost = 1500 →
  gaming_chair_cost = 400 →
  accessories_cost = 300 →
  comp_discount = 0.2 →
  gaming_discount = 0.1 →
  tax = 0.05 →
  playstation_value = 600 →
  playstation_discount = 0.2 →
  bicycle_price = 200 →
  exchange_rate = 100 →
  total_cost_after_discounts computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax -
  total_selling_price playstation_value playstation_discount bicycle_price exchange_rate = 1273 := by
  intros
  sorry

end johns_out_of_pocket_l0_302


namespace econ_not_feasible_l0_119

theorem econ_not_feasible (x y p q: ℕ) (h_xy : 26 * x + 29 * y = 687) (h_pq : 27 * p + 31 * q = 687) : p + q ≥ x + y := by
  sorry

end econ_not_feasible_l0_119


namespace ratio_of_b_to_a_l0_322

variable (V A B : ℝ)

def ten_pours_of_a_cup : Prop := 10 * A = V
def five_pours_of_b_cup : Prop := 5 * B = V

theorem ratio_of_b_to_a (h1 : ten_pours_of_a_cup V A) (h2 : five_pours_of_b_cup V B) : B / A = 2 :=
sorry

end ratio_of_b_to_a_l0_322


namespace factor_difference_of_squares_l0_894

theorem factor_difference_of_squares (y : ℝ) : 81 - 16 * y^2 = (9 - 4 * y) * (9 + 4 * y) :=
by
  sorry

end factor_difference_of_squares_l0_894


namespace batsman_percentage_running_between_wickets_l0_114

def boundaries : Nat := 6
def runs_per_boundary : Nat := 4
def sixes : Nat := 4
def runs_per_six : Nat := 6
def no_balls : Nat := 8
def runs_per_no_ball : Nat := 1
def wide_balls : Nat := 5
def runs_per_wide_ball : Nat := 1
def leg_byes : Nat := 2
def runs_per_leg_bye : Nat := 1
def total_score : Nat := 150

def runs_from_boundaries : Nat := boundaries * runs_per_boundary
def runs_from_sixes : Nat := sixes * runs_per_six
def runs_not_off_bat : Nat := no_balls * runs_per_no_ball + wide_balls * runs_per_wide_ball + leg_byes * runs_per_leg_bye

def runs_running_between_wickets : Nat := total_score - runs_not_off_bat - runs_from_boundaries - runs_from_sixes

def percentage_runs_running_between_wickets : Float := 
  (runs_running_between_wickets.toFloat / total_score.toFloat) * 100

theorem batsman_percentage_running_between_wickets : percentage_runs_running_between_wickets = 58 := sorry

end batsman_percentage_running_between_wickets_l0_114


namespace weights_less_than_90_l0_172

variable (a b c : ℝ)
-- conditions
axiom h1 : a + b = 100
axiom h2 : a + c = 101
axiom h3 : b + c = 102

theorem weights_less_than_90 (a b c : ℝ) (h1 : a + b = 100) (h2 : a + c = 101) (h3 : b + c = 102) : a < 90 ∧ b < 90 ∧ c < 90 := 
by sorry

end weights_less_than_90_l0_172


namespace solve_swim_problem_l0_602

/-- A man swims downstream 36 km and upstream some distance taking 3 hours each time. 
The speed of the man in still water is 9 km/h. -/
def swim_problem : Prop :=
  ∃ (v : ℝ) (d : ℝ),
    (9 + v) * 3 = 36 ∧ -- effective downstream speed and distance condition
    (9 - v) * 3 = d ∧ -- effective upstream speed and distance relation
    d = 18            -- required distance upstream is 18 km

theorem solve_swim_problem : swim_problem :=
  sorry

end solve_swim_problem_l0_602


namespace area_of_park_l0_421

theorem area_of_park (x : ℕ) (rate_per_meter : ℝ) (total_cost : ℝ)
  (ratio_len_wid : ℕ × ℕ)
  (h_ratio : ratio_len_wid = (3, 2))
  (h_cost : total_cost = 140)
  (unit_rate : rate_per_meter = 0.50)
  (h_perimeter : 10 * x * rate_per_meter = total_cost) :
  6 * x^2 = 4704 :=
by
  sorry

end area_of_park_l0_421


namespace smallest_value_of_x_l0_305

theorem smallest_value_of_x (x : ℝ) (h : |x - 3| = 8) : x = -5 :=
sorry

end smallest_value_of_x_l0_305


namespace least_prime_factor_of_expr_l0_69

theorem least_prime_factor_of_expr : ∀ n : ℕ, n = 11^5 - 11^2 → (∃ p : ℕ, Nat.Prime p ∧ p ≤ 2 ∧ p ∣ n) :=
by
  intros n h
  -- here will be proof steps, currently skipped
  sorry

end least_prime_factor_of_expr_l0_69


namespace marbles_solution_l0_531

def marbles_problem : Prop :=
  let total_marbles := 20
  let blue_marbles := 6
  let red_marbles := 9
  let total_prob_red_white := 0.7
  let white_marbles := 5
  total_marbles = blue_marbles + red_marbles + white_marbles ∧
  (white_marbles / total_marbles + red_marbles / total_marbles = total_prob_red_white)

theorem marbles_solution : marbles_problem :=
by {
  sorry
}

end marbles_solution_l0_531


namespace num_integers_satisfying_inequality_l0_221

theorem num_integers_satisfying_inequality : 
  ∃ (xs : Finset ℤ), (∀ x ∈ xs, -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9) ∧ xs.card = 5 := 
by 
  sorry

end num_integers_satisfying_inequality_l0_221


namespace rhombus_area_l0_769

-- Definition of a rhombus with given conditions
structure Rhombus where
  side : ℝ
  d1 : ℝ
  d2 : ℝ

noncomputable def Rhombus.area (r : Rhombus) : ℝ :=
  (r.d1 * r.d2) / 2

noncomputable example : Rhombus :=
{ side := 20,
  d1 := 16,
  d2 := 8 * Real.sqrt 21 }

theorem rhombus_area : 
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  Rhombus.area r = 64 * Real.sqrt 21 :=
by
  let r : Rhombus := { side := 20, d1 := 16, d2 := 8 * Real.sqrt 21 }
  sorry

end rhombus_area_l0_769


namespace range_of_a_l0_720

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 ≤ 1) (f_def : ∀ x, f x = a * x - x^3)
  (condition : f x2 - f x1 > x2 - x1) :
  a ≥ 4 :=
by sorry

end range_of_a_l0_720


namespace max_S_n_l0_588

noncomputable def S (n : ℕ) : ℝ := sorry  -- Definition of the sum of the first n terms

theorem max_S_n (S : ℕ → ℝ) (h16 : S 16 > 0) (h17 : S 17 < 0) : ∃ n, S n = S 8 :=
sorry

end max_S_n_l0_588


namespace average_of_first_6_numbers_l0_550

-- Definitions extracted from conditions
def average_of_11_numbers := 60
def average_of_last_6_numbers := 65
def sixth_number := 258
def total_sum := 11 * average_of_11_numbers
def sum_of_last_6_numbers := 6 * average_of_last_6_numbers

-- Lean 4 statement for the proof problem
theorem average_of_first_6_numbers :
  (∃ A, 6 * A = (total_sum - (sum_of_last_6_numbers - sixth_number))) →
  (∃ A, 6 * A = 528) :=
by
  intro h
  exact h

end average_of_first_6_numbers_l0_550


namespace range_of_m_l0_967

theorem range_of_m (m : ℝ) :
  (∃ x y, y = x^2 + m * x + 2 ∧ x - y + 1 = 0 ∧ 0 ≤ x ∧ x ≤ 2) → m ≤ -1 :=
by
  sorry

end range_of_m_l0_967


namespace diagonal_BD_size_cos_A_value_l0_231

noncomputable def AB := 250
noncomputable def CD := 250
noncomputable def angle_A := 120
noncomputable def angle_C := 120
noncomputable def AD := 150
noncomputable def BC := 150
noncomputable def perimeter := 800

/-- The size of the diagonal BD in isosceles trapezoid ABCD is 350, given the conditions -/
theorem diagonal_BD_size (AB CD AD BC : ℕ) (angle_A angle_C : ℝ) :
  AB = 250 → CD = 250 → AD = 150 → BC = 150 →
  angle_A = 120 → angle_C = 120 →
  ∃ BD : ℝ, BD = 350 :=
by
  sorry

/-- The cosine of angle A is -0.5, given the angle is 120 degrees -/
theorem cos_A_value (angle_A : ℝ) :
  angle_A = 120 → ∃ cos_A : ℝ, cos_A = -0.5 :=
by
  sorry

end diagonal_BD_size_cos_A_value_l0_231


namespace usual_time_cover_journey_l0_24

theorem usual_time_cover_journey (S T : ℝ) (H : S / T = (5/6 * S) / (T + 8)) : T = 48 :=
by
  sorry

end usual_time_cover_journey_l0_24


namespace x_intercept_is_3_l0_885

-- Define the given points
def point1 : ℝ × ℝ := (2, -2)
def point2 : ℝ × ℝ := (6, 6)

-- Prove the x-intercept is 3
theorem x_intercept_is_3 (x : ℝ) :
  (∃ m b : ℝ, (∀ x1 y1 x2 y2 : ℝ, (y1 = m * x1 + b) ∧ (x1, y1) = point1 ∧ (x2, y2) = point2) ∧ y = 0 ∧ x = -b / m) → x = 3 :=
sorry

end x_intercept_is_3_l0_885


namespace sales_tax_difference_l0_459

theorem sales_tax_difference (rate1 rate2 : ℝ) (price : ℝ) (h1 : rate1 = 0.075) (h2 : rate2 = 0.0625) (hprice : price = 50) : 
  rate1 * price - rate2 * price = 0.625 :=
by
  sorry

end sales_tax_difference_l0_459


namespace circle_center_radius_sum_l0_704

theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), c = -6 ∧ d = -7 ∧ s = Real.sqrt 13 ∧
  (x^2 + 14 * y + 72 = -y^2 - 12 * x → c + d + s = -13 + Real.sqrt 13) :=
sorry

end circle_center_radius_sum_l0_704


namespace cos_double_angle_l0_615

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2*n) = 8) :
  Real.cos (2 * θ) = 3 / 4 :=
sorry

end cos_double_angle_l0_615


namespace range_of_function_l0_456

open Set

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_of_function (S : Set ℝ) : 
    S = {y : ℝ | ∃ x : ℝ, x ≥ 1 ∧ y = 2 + log_base_2 x} 
    ↔ S = {y : ℝ | y ≥ 2} :=
by 
  sorry

end range_of_function_l0_456


namespace equation_has_one_real_root_l0_336

noncomputable def f (x : ℝ) : ℝ :=
  (3 / 11)^x + (5 / 11)^x + (7 / 11)^x - 1

theorem equation_has_one_real_root :
  ∃! x : ℝ, f x = 0 := sorry

end equation_has_one_real_root_l0_336


namespace a_n3_l0_650

def right_angled_triangle_array (a : ℕ → ℕ → ℚ) : Prop :=
  ∀ i j, 1 ≤ j ∧ j ≤ i →
    (j = 1 → a i j = 1 / 4 + (i - 1) / 4) ∧
    (i ≥ 3 → (1 < j → a i j = a i 1 * (1 / 2)^(j - 1)))

theorem a_n3 (a : ℕ → ℕ → ℚ) (n : ℕ) (h : right_angled_triangle_array a) : a n 3 = n / 16 :=
sorry

end a_n3_l0_650


namespace number_of_ants_l0_11

def spiders := 8
def spider_legs := 8
def ants := 12
def ant_legs := 6
def total_legs := 136

theorem number_of_ants :
  spiders * spider_legs + ants * ant_legs = total_legs → ants = 12 :=
by
  sorry

end number_of_ants_l0_11


namespace finish_11th_l0_314

noncomputable def place_in_race (place: Fin 15) := ℕ

variables (Dana Ethan Alice Bob Chris Flora : Fin 15)

def conditions := 
  Dana.val + 3 = Ethan.val ∧
  Alice.val = Bob.val - 2 ∧
  Chris.val = Flora.val - 5 ∧
  Flora.val = Dana.val + 2 ∧
  Ethan.val = Alice.val - 3 ∧
  Bob.val = 6

theorem finish_11th (h : conditions Dana Ethan Alice Bob Chris Flora) : Flora.val = 10 :=
  by sorry

end finish_11th_l0_314


namespace triangle_subsegment_length_l0_641

theorem triangle_subsegment_length (DF DE EF DG GF : ℚ)
  (h_ratio : ∃ x : ℚ, DF = 3 * x ∧ DE = 4 * x ∧ EF = 5 * x)
  (h_EF_len : EF = 20)
  (h_angle_bisector : DG + GF = DE ∧ DG / GF = DE / DF) :
  DF < DE ∧ DE < EF →
  min DG GF = 48 / 7 :=
by
  sorry

end triangle_subsegment_length_l0_641


namespace radio_selling_price_l0_586

noncomputable def sellingPrice (costPrice : ℝ) (lossPercentage : ℝ) : ℝ :=
  costPrice - (lossPercentage / 100 * costPrice)

theorem radio_selling_price :
  sellingPrice 490 5 = 465.5 :=
by
  sorry

end radio_selling_price_l0_586


namespace part1_part2_l0_553

open Real

noncomputable def curve_parametric (α : ℝ) : ℝ × ℝ :=
  (2 + sqrt 10 * cos α, sqrt 10 * sin α)

noncomputable def curve_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * cos θ - 6 = 0

noncomputable def line_polar (ρ θ : ℝ) : Prop :=
  ρ * cos θ + 2 * ρ * sin θ - 12 = 0

theorem part1 (α : ℝ) : ∃ ρ θ : ℝ, curve_polar ρ θ :=
  sorry

theorem part2 : ∃ ρ1 ρ2 : ℝ, curve_polar ρ1 (π / 4) ∧ line_polar ρ2 (π / 4) ∧ abs (ρ1 - ρ2) = sqrt 2 :=
  sorry

end part1_part2_l0_553


namespace quarters_value_percentage_l0_138

theorem quarters_value_percentage (dimes_count quarters_count dimes_value quarters_value : ℕ) (h1 : dimes_count = 75)
    (h2 : quarters_count = 30) (h3 : dimes_value = 10) (h4 : quarters_value = 25) :
    (quarters_count * quarters_value * 100) / (dimes_count * dimes_value + quarters_count * quarters_value) = 50 := 
by
    sorry

end quarters_value_percentage_l0_138


namespace abs_diff_of_pq_eq_6_and_pq_sum_7_l0_759

variable (p q : ℝ)

noncomputable def abs_diff (a b : ℝ) := |a - b|

theorem abs_diff_of_pq_eq_6_and_pq_sum_7 (hpq : p * q = 6) (hpq_sum : p + q = 7) : abs_diff p q = 5 :=
by
  sorry

end abs_diff_of_pq_eq_6_and_pq_sum_7_l0_759


namespace lcm_gcd_48_180_l0_331

theorem lcm_gcd_48_180 :
  Nat.lcm 48 180 = 720 ∧ Nat.gcd 48 180 = 12 :=
by
  sorry

end lcm_gcd_48_180_l0_331


namespace total_adults_across_all_three_buses_l0_668

def total_passengers : Nat := 450
def bus_A_passengers : Nat := 120
def bus_B_passengers : Nat := 210
def bus_C_passengers : Nat := 120
def children_ratio_A : ℚ := 1/3
def children_ratio_B : ℚ := 2/5
def children_ratio_C : ℚ := 3/8

theorem total_adults_across_all_three_buses :
  let children_A := bus_A_passengers * children_ratio_A
  let children_B := bus_B_passengers * children_ratio_B
  let children_C := bus_C_passengers * children_ratio_C
  let adults_A := bus_A_passengers - children_A
  let adults_B := bus_B_passengers - children_B
  let adults_C := bus_C_passengers - children_C
  (adults_A + adults_B + adults_C) = 281 := by {
    -- The proof steps will go here
    sorry
}

end total_adults_across_all_three_buses_l0_668


namespace palindrome_probability_divisible_by_11_l0_797

namespace PalindromeProbability

-- Define the concept of a five-digit palindrome and valid digits
def is_five_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 10001 * a + 1010 * b + 100 * c

-- Define the condition for a number being divisible by 11
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Count all five-digit palindromes
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10  -- There are 9 choices for a (1-9), and 10 choices for b and c (0-9)

-- Count five-digit palindromes that are divisible by 11
def count_divisible_by_11_five_digit_palindromes : ℕ :=
  9 * 10  -- There are 9 choices for a, and 10 valid (b, c) pairs for divisibility by 11

-- Calculate the probability
theorem palindrome_probability_divisible_by_11 :
  (count_divisible_by_11_five_digit_palindromes : ℚ) / count_five_digit_palindromes = 1 / 10 :=
  by sorry -- Proof goes here

end PalindromeProbability

end palindrome_probability_divisible_by_11_l0_797


namespace sally_cards_final_count_l0_552

def initial_cards : ℕ := 27
def cards_from_Dan : ℕ := 41
def cards_bought : ℕ := 20
def cards_traded : ℕ := 15
def cards_lost : ℕ := 7

def final_cards (initial : ℕ) (from_Dan : ℕ) (bought : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial + from_Dan + bought - traded - lost

theorem sally_cards_final_count :
  final_cards initial_cards cards_from_Dan cards_bought cards_traded cards_lost = 66 := by
  sorry

end sally_cards_final_count_l0_552


namespace problem_l0_140

-- Definitions and conditions
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n, 2 ≤ n → 2 * a n / (a n * (Finset.sum (Finset.range n) a) - (Finset.sum (Finset.range n) a) ^ 2) = 1)

-- Sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := Finset.sum (Finset.range n) a

-- The proof statement
theorem problem (a : ℕ → ℚ) (h : seq a) : S a 2017 = 1 / 1009 := sorry

end problem_l0_140


namespace cafeteria_problem_l0_292

theorem cafeteria_problem (C : ℕ) 
    (h1 : ∃ h : ℕ, h = 4 * C)
    (h2 : 5 = 5)
    (h3 : C + 4 * C + 5 = 40) : 
    C = 7 := sorry

end cafeteria_problem_l0_292


namespace raju_working_days_l0_716

theorem raju_working_days (x : ℕ) 
  (h1: (1 / 10 : ℚ) + 1 / x = 1 / 8) : x = 40 :=
by sorry

end raju_working_days_l0_716


namespace calculateRemainingMoney_l0_194

def initialAmount : ℝ := 100
def actionFiguresCount : ℕ := 3
def actionFigureOriginalPrice : ℝ := 12
def actionFigureDiscount : ℝ := 0.25
def boardGamesCount : ℕ := 2
def boardGamePrice : ℝ := 11
def puzzleSetsCount : ℕ := 4
def puzzleSetPrice : ℝ := 6
def salesTax : ℝ := 0.05

theorem calculateRemainingMoney :
  initialAmount - (
    (actionFigureOriginalPrice * (1 - actionFigureDiscount) * actionFiguresCount) +
    (boardGamePrice * boardGamesCount) +
    (puzzleSetPrice * puzzleSetsCount)
  ) * (1 + salesTax) = 23.35 :=
by
  sorry

end calculateRemainingMoney_l0_194


namespace total_students_is_45_l0_98

theorem total_students_is_45
  (students_burgers : ℕ) 
  (total_students : ℕ) 
  (hb : students_burgers = 30) 
  (ht : total_students = 45) : 
  total_students = 45 :=
by
  sorry

end total_students_is_45_l0_98


namespace shaded_figure_perimeter_l0_186

theorem shaded_figure_perimeter (a b : ℝ) (area_overlap : ℝ) (side_length : ℝ) (side_length_overlap : ℝ):
    a = 5 → b = 5 → area_overlap = 4 → side_length_overlap * side_length_overlap = area_overlap →
    side_length_overlap = 2 →
    ((4 * a) + (4 * b) - (4 * side_length_overlap)) = 32 :=
by
  intros
  sorry

end shaded_figure_perimeter_l0_186


namespace tangent_vertical_y_axis_iff_a_gt_0_l0_115

theorem tangent_vertical_y_axis_iff_a_gt_0 {a : ℝ} (f : ℝ → ℝ) 
    (hf : ∀ x > 0, f x = a * x^2 - Real.log x)
    (h_tangent_vertical : ∃ x > 0, (deriv f x) = 0) :
    a > 0 := 
sorry

end tangent_vertical_y_axis_iff_a_gt_0_l0_115


namespace ray_has_4_nickels_left_l0_8

theorem ray_has_4_nickels_left (initial_cents : ℕ) (given_to_peter : ℕ)
    (given_to_randi : ℕ) (value_of_nickel : ℕ) (remaining_cents : ℕ) 
    (remaining_nickels : ℕ) :
    initial_cents = 95 →
    given_to_peter = 25 →
    given_to_randi = 2 * given_to_peter →
    value_of_nickel = 5 →
    remaining_cents = initial_cents - given_to_peter - given_to_randi →
    remaining_nickels = remaining_cents / value_of_nickel →
    remaining_nickels = 4 :=
by
  intros
  sorry

end ray_has_4_nickels_left_l0_8


namespace symmetric_points_subtraction_l0_88

theorem symmetric_points_subtraction (a b : ℝ) (h1 : -2 = -a) (h2 : b = -3) : a - b = 5 :=
by {
  sorry
}

end symmetric_points_subtraction_l0_88


namespace isosceles_triangle_l0_680

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b^2 - 2 * b * c + c^2) = 0) : 
  (a = b) ∨ (b = c) :=
by sorry

end isosceles_triangle_l0_680


namespace per_capita_income_growth_l0_506

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end per_capita_income_growth_l0_506


namespace Cameron_books_proof_l0_554

noncomputable def Cameron_initial_books :=
  let B : ℕ := 24
  let B_donated := B / 4
  let B_left := B - B_donated
  let C_donated (C : ℕ) := C / 3
  let C_left (C : ℕ) := C - C_donated C
  ∃ C : ℕ, B_left + C_left C = 38 ∧ C = 30

-- Note that we use sorry to indicate the proof is omitted.
theorem Cameron_books_proof : Cameron_initial_books :=
by {
  sorry
}

end Cameron_books_proof_l0_554


namespace value_g2_l0_433

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (g (x - y)) = g x * g y - g x + g y - x^3 * y^3

theorem value_g2 : g 2 = 8 :=
by sorry

end value_g2_l0_433


namespace condition_on_a_b_l0_897

theorem condition_on_a_b (a b : ℝ) (h : a^2 * b^2 + 5 > 2 * a * b - a^2 - 4 * a) : ab ≠ 1 ∨ a ≠ -2 :=
by
  sorry

end condition_on_a_b_l0_897


namespace a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l0_227

def a_n (n : ℕ) : ℕ := 10^(3*n+2) + 2 * 10^(2*n+1) + 2 * 10^(n+1) + 1

theorem a_n_div_3_sum_two_cubes (n : ℕ) : ∃ x y : ℤ, (x > 0) ∧ (y > 0) ∧ (a_n n / 3 = x^3 + y^3) := sorry

theorem a_n_div_3_not_sum_two_squares (n : ℕ) : ¬ (∃ x y : ℤ, a_n n / 3 = x^2 + y^2) := sorry

end a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l0_227


namespace shara_shells_l0_382

def initial_shells : ℕ := 20
def first_vacation_day1_3 : ℕ := 5 * 3
def first_vacation_day4 : ℕ := 6
def second_vacation_day1_2 : ℕ := 4 * 2
def second_vacation_day3 : ℕ := 7
def third_vacation_day1 : ℕ := 8
def third_vacation_day2 : ℕ := 4
def third_vacation_day3_4 : ℕ := 3 * 2

def total_shells : ℕ :=
  initial_shells + 
  (first_vacation_day1_3 + first_vacation_day4) +
  (second_vacation_day1_2 + second_vacation_day3) + 
  (third_vacation_day1 + third_vacation_day2 + third_vacation_day3_4)

theorem shara_shells : total_shells = 74 :=
by
  sorry

end shara_shells_l0_382


namespace x729_minus_inverse_l0_936

theorem x729_minus_inverse (x : ℂ) (h : x - x⁻¹ = 2 * Complex.I) : x ^ 729 - x⁻¹ ^ 729 = 2 * Complex.I := 
by 
  sorry

end x729_minus_inverse_l0_936


namespace tan_arith_seq_l0_601

theorem tan_arith_seq (x y z : ℝ)
  (h₁ : y = x + π / 3)
  (h₂ : z = x + 2 * π / 3) :
  (Real.tan x * Real.tan y) + (Real.tan y * Real.tan z) + (Real.tan z * Real.tan x) = -3 :=
sorry

end tan_arith_seq_l0_601


namespace fraction_of_males_on_time_l0_87

theorem fraction_of_males_on_time (A : ℕ) :
  (2 / 9 : ℚ) * A = (2 / 9 : ℚ) * A → 
  (2 / 3 : ℚ) * A = (2 / 3 : ℚ) * A → 
  (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) = (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) → 
  ((7 / 9 : ℚ) * A - (5 / 18 : ℚ) * A) / ((2 / 3 : ℚ) * A) = (1 / 2 : ℚ) :=
by
  intros h1 h2 h3
  sorry

end fraction_of_males_on_time_l0_87


namespace rebecca_less_than_toby_l0_107

-- Define the conditions
variable (x : ℕ) -- Thomas worked x hours
variable (tobyHours : ℕ := 2 * x - 10) -- Toby worked 10 hours less than twice what Thomas worked
variable (rebeccaHours : ℕ := 56) -- Rebecca worked 56 hours

-- Define the total hours worked in one week
axiom total_hours_worked : x + tobyHours + rebeccaHours = 157

-- The proof goal
theorem rebecca_less_than_toby : tobyHours - rebeccaHours = 8 := 
by
  -- (proof steps would go here)
  sorry

end rebecca_less_than_toby_l0_107


namespace percentage_decrease_in_speed_l0_325

variable (S : ℝ) (S' : ℝ) (T T' : ℝ)

noncomputable def percentageDecrease (originalSpeed decreasedSpeed : ℝ) : ℝ :=
  ((originalSpeed - decreasedSpeed) / originalSpeed) * 100

theorem percentage_decrease_in_speed :
  T = 40 ∧ T' = 50 ∧ S' = (4 / 5) * S →
  percentageDecrease S S' = 20 :=
by sorry

end percentage_decrease_in_speed_l0_325


namespace find_diameter_C_l0_791

noncomputable def diameter_of_circle_C (diameter_of_D : ℝ) (ratio_shaded_to_C : ℝ) : ℝ :=
  let radius_D := diameter_of_D / 2
  let radius_C := radius_D / (2 * Real.sqrt ratio_shaded_to_C)
  2 * radius_C

theorem find_diameter_C :
  let diameter_D := 20
  let ratio_shaded_area_to_C := 7
  diameter_of_circle_C diameter_D ratio_shaded_area_to_C = 5 * Real.sqrt 2 :=
by
  -- The proof is omitted.
  sorry

end find_diameter_C_l0_791


namespace solve_a_minus_b_l0_686

theorem solve_a_minus_b (a b : ℝ) (h1 : 2010 * a + 2014 * b = 2018) (h2 : 2012 * a + 2016 * b = 2020) : a - b = -3 :=
sorry

end solve_a_minus_b_l0_686


namespace time_for_machine_A_l0_551

theorem time_for_machine_A (x : ℝ) (T : ℝ) (A B : ℝ) :
  (B = 2 * x / 5) → 
  (A + B = x / 2) → 
  (A = x / T) → 
  T = 10 := 
by 
  intros hB hAB hA
  sorry

end time_for_machine_A_l0_551


namespace find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l0_775

def U := Set ℝ
def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

theorem find_intersection (x : ℝ) : x ∈ A ∧ x ∈ B ↔ 4 < x ∧ x < 6 :=
by
  sorry

theorem complement_B (x : ℝ) : x ∉ B ↔ x ≥ 6 ∨ x ≤ -6 :=
by
  sorry

def A_minus_B : Set ℝ := {x | x ∈ A ∧ x ∉ B}

theorem find_A_minus_B (x : ℝ) : x ∈ A_minus_B ↔ x ≥ 6 :=
by
  sorry

theorem find_A_minus_A_minus_B (x : ℝ) : x ∈ (A \ A_minus_B) ↔ 4 < x ∧ x < 6 :=
by
  sorry

end find_intersection_complement_B_find_A_minus_B_find_A_minus_A_minus_B_l0_775


namespace trajectory_of_point_P_l0_297

open Real

theorem trajectory_of_point_P (a : ℝ) (ha : a > 0) :
  (∀ x y : ℝ, (a = 1 → x = 0) ∧ 
    (a ≠ 1 → (x - (a^2 + 1) / (a^2 - 1))^2 + y^2 = 4 * a^2 / (a^2 - 1)^2)) := 
by 
  sorry

end trajectory_of_point_P_l0_297


namespace tom_walking_distance_l0_48

noncomputable def walking_rate_miles_per_minute : ℝ := 1 / 18
def walking_time_minutes : ℝ := 15
def expected_distance_miles : ℝ := 0.8

theorem tom_walking_distance :
  walking_rate_miles_per_minute * walking_time_minutes = expected_distance_miles :=
by
  -- Calculation steps and conversion to decimal are skipped
  sorry

end tom_walking_distance_l0_48


namespace divisibility_by_six_l0_228

theorem divisibility_by_six (n : ℤ) : 6 ∣ (n^3 - n) := 
sorry

end divisibility_by_six_l0_228


namespace find_a_l0_442

theorem find_a (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_b : b = 1)
    (h_ab_ccb : (10 * a + b)^2 = 100 * c + 10 * c + b) (h_ccb_gt_300 : 100 * c + 10 * c + b > 300) :
    a = 2 :=
sorry

end find_a_l0_442
