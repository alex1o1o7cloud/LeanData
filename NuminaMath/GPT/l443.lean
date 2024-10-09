import Mathlib

namespace calculation_equivalence_l443_44372

theorem calculation_equivalence : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := 
by
  sorry

end calculation_equivalence_l443_44372


namespace boat_speed_in_still_water_l443_44335

theorem boat_speed_in_still_water 
  (rate_of_current : ℝ) 
  (time_in_hours : ℝ) 
  (distance_downstream : ℝ)
  (h_rate : rate_of_current = 5) 
  (h_time : time_in_hours = 15 / 60) 
  (h_distance : distance_downstream = 6.25) : 
  ∃ x : ℝ, (distance_downstream = (x + rate_of_current) * time_in_hours) ∧ x = 20 :=
by 
  -- Main theorem statement, proof omitted for brevity.
  sorry

end boat_speed_in_still_water_l443_44335


namespace hyperbola_same_foci_l443_44317

-- Define the conditions for the ellipse and hyperbola
def ellipse (x y : ℝ) : Prop := (x^2 / 12) + (y^2 / 4) = 1
def hyperbola (x y m : ℝ) : Prop := (x^2 / m) - y^2 = 1

-- Statement to be proved in Lean 4
theorem hyperbola_same_foci : ∃ m : ℝ, ∀ x y : ℝ, ellipse x y → hyperbola x y m :=
by
  have a_squared := 12
  have b_squared := 4
  have c_squared := a_squared - b_squared
  have c := Real.sqrt c_squared
  have c_value : c = 2 * Real.sqrt 2 := by sorry
  let m := c^2 - 1
  exact ⟨m, by sorry⟩

end hyperbola_same_foci_l443_44317


namespace compute_fraction_l443_44365

theorem compute_fraction : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64) / (2 - 4 + 8 - 16 + 32 - 64 + 128) = 1 / 2 := 
by
  sorry

end compute_fraction_l443_44365


namespace problem_l443_44395

-- Definitions and hypotheses based on the given conditions
variable (a b : ℝ)
def sol_set := {x : ℝ | -1/2 < x ∧ x < 1/3}
def quadratic_inequality (x : ℝ) := a * x^2 + b * x + 2

-- Statement expressing that the inequality holds for the given solution set
theorem problem
  (h : ∀ (x : ℝ), x ∈ sol_set → quadratic_inequality a b x > 0) :
  a - b = -10 :=
sorry

end problem_l443_44395


namespace frame_interior_edge_sum_l443_44353

theorem frame_interior_edge_sum (y : ℝ) :
  ( ∀ outer_edge1 : ℝ, outer_edge1 = 7 →
    ∀ frame_width : ℝ, frame_width = 2 →
    ∀ frame_area : ℝ, frame_area = 30 →
    7 * y - (3 * (y - 4)) = 30) → 
  (7 * y - (4 * y - 12) ) / 4 = 4.5 → 
  (3 + (y - 4)) * 2 = 7 :=
sorry

end frame_interior_edge_sum_l443_44353


namespace negation_of_p_is_correct_l443_44352

variable (c : ℝ)

-- Proposition p defined as: there exists c > 0 such that x^2 - x + c = 0 has a solution
def proposition_p : Prop :=
  ∃ c > 0, ∃ x : ℝ, x^2 - x + c = 0

-- Negation of proposition p
def neg_proposition_p : Prop :=
  ∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0

-- The Lean statement to prove
theorem negation_of_p_is_correct :
  neg_proposition_p ↔ (∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end negation_of_p_is_correct_l443_44352


namespace part1_growth_rate_part2_new_price_l443_44393

-- Definitions based on conditions
def purchase_price : ℕ := 30
def selling_price : ℕ := 40
def january_sales : ℕ := 400
def march_sales : ℕ := 576
def growth_rate (x : ℝ) : Prop := january_sales * (1 + x)^2 = march_sales

-- Part (1): Prove the monthly average growth rate
theorem part1_growth_rate : 
  ∃ (x : ℝ), growth_rate x ∧ x = 0.2 :=
by
  sorry

-- Definitions for part (2) - based on the second condition
def price_reduction (y : ℝ) : Prop := (selling_price - y - purchase_price) * (march_sales + 12 * y) = 4800

-- Part (2): Prove the new price for April
theorem part2_new_price :
  ∃ (y : ℝ), price_reduction y ∧ (selling_price - y) = 38 :=
by
  sorry

end part1_growth_rate_part2_new_price_l443_44393


namespace arithmetic_sequence_a14_l443_44397

theorem arithmetic_sequence_a14 (a : ℕ → ℤ) (h1 : a 4 = 5) (h2 : a 9 = 17) (h3 : 2 * a 9 = a 14 + a 4) : a 14 = 29 := sorry

end arithmetic_sequence_a14_l443_44397


namespace sand_removal_l443_44373

theorem sand_removal :
  let initial_weight := (8 / 3 : ℚ)
  let first_removal := (1 / 4 : ℚ)
  let second_removal := (5 / 6 : ℚ)
  initial_weight - (first_removal + second_removal) = (13 / 12 : ℚ) := by
  -- sorry is used here to skip the proof as instructed
  sorry

end sand_removal_l443_44373


namespace origin_not_in_A_point_M_in_A_l443_44311

def set_A : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ x + 2 * y - 1 ≥ 0 ∧ y ≤ x + 2 ∧ 2 * x + y - 5 ≤ 0}

theorem origin_not_in_A : (0, 0) ∉ set_A := by
  sorry

theorem point_M_in_A : (1, 1) ∈ set_A := by
  sorry

end origin_not_in_A_point_M_in_A_l443_44311


namespace probability_remainder_is_4_5_l443_44367

def probability_remainder_1 (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1)

theorem probability_remainder_is_4_5 : 
  ∀ N, N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1) → (number_of_successful_outcomes / total_outcomes = 4 / 5) :=
sorry

end probability_remainder_is_4_5_l443_44367


namespace a5_is_16_S8_is_255_l443_44396

-- Define the sequence
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * seq n

-- Definition of the geometric sum
def geom_sum (n : ℕ) : ℕ :=
  (2 ^ (n + 1) - 1)

-- Prove that a₅ = 16
theorem a5_is_16 : seq 5 = 16 :=
  by
  unfold seq
  sorry

-- Prove that the sum of the first 8 terms, S₈ = 255
theorem S8_is_255 : geom_sum 7 = 255 :=
  by 
  unfold geom_sum
  sorry

end a5_is_16_S8_is_255_l443_44396


namespace find_a_b_largest_x_l443_44388

def polynomial (a b x : ℤ) : ℤ := 2 * (a * x - 3) - 3 * (b * x + 5)

-- Given conditions
variables (a b : ℤ)
#check polynomial

-- Part 1: Prove the values of a and b
theorem find_a_b (h1 : polynomial a b 2 = -31) (h2 : a + b = 0) : a = -1 ∧ b = 1 :=
by sorry

-- Part 2: Given a and b found in Part 1, find the largest integer x such that P > 0
noncomputable def P (x : ℤ) : ℤ := -5 * x - 21

theorem largest_x {a b : ℤ} (ha : a = -1) (hb : b = 1) : ∃ x : ℤ, P x > 0 ∧ ∀ y : ℤ, (P y > 0 → y ≤ x) :=
by sorry

end find_a_b_largest_x_l443_44388


namespace group_division_l443_44322

theorem group_division (total_students groups_per_group : ℕ) (h1 : total_students = 30) (h2 : groups_per_group = 5) : 
  (total_students / groups_per_group) = 6 := 
by 
  sorry

end group_division_l443_44322


namespace students_with_no_preference_l443_44313

def total_students : ℕ := 210
def prefer_mac : ℕ := 60
def equally_prefer_both (x : ℕ) : ℕ := x / 3

def no_preference_students : ℕ :=
  total_students - (prefer_mac + equally_prefer_both prefer_mac)

theorem students_with_no_preference :
  no_preference_students = 130 :=
by
  sorry

end students_with_no_preference_l443_44313


namespace equation_represents_pair_of_lines_l443_44391

theorem equation_represents_pair_of_lines : ∀ x y : ℝ, 9 * x^2 - 25 * y^2 = 0 → 
                    (x = (5/3) * y ∨ x = -(5/3) * y) :=
by sorry

end equation_represents_pair_of_lines_l443_44391


namespace initial_house_cats_l443_44387

theorem initial_house_cats (H : ℕ) 
  (siamese_cats : ℕ := 38) 
  (cats_sold : ℕ := 45) 
  (cats_left : ℕ := 18) 
  (initial_total_cats : ℕ := siamese_cats + H) 
  (after_sale_cats : ℕ := initial_total_cats - cats_sold) : 
  after_sale_cats = cats_left → H = 25 := 
by
  intro h
  sorry

end initial_house_cats_l443_44387


namespace express_as_terminating_decimal_l443_44331

section terminating_decimal

theorem express_as_terminating_decimal
  (a b : ℚ)
  (h1 : a = 125)
  (h2 : b = 144)
  (h3 : b = 2^4 * 3^2): 
  a / b = 0.78125 := 
by 
  sorry

end terminating_decimal

end express_as_terminating_decimal_l443_44331


namespace find_j_of_scaled_quadratic_l443_44357

/- Define the given condition -/
def quadratic_expressed (p q r : ℝ) : Prop :=
  ∀ x : ℝ, p * x^2 + q * x + r = 5 * (x - 3)^2 + 15

/- State the theorem to be proved -/
theorem find_j_of_scaled_quadratic (p q r m j l : ℝ) (h_quad : quadratic_expressed p q r) :
  (∀ x : ℝ, 2 * p * x^2 + 2 * q * x + 2 * r = m * (x - j)^2 + l) → j = 3 :=
by
  intro h
  sorry

end find_j_of_scaled_quadratic_l443_44357


namespace quadratic_inequality_solution_range_l443_44305

theorem quadratic_inequality_solution_range (a : ℝ) :
  (¬ ∃ x : ℝ, 4 * x^2 + (a - 2) * x + 1 / 4 ≤ 0) ↔ 0 < a ∧ a < 4 :=
by
  sorry

end quadratic_inequality_solution_range_l443_44305


namespace squares_are_equal_l443_44383

theorem squares_are_equal (a b c d : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) 
    (h₄ : a * (b + c + d) = b * (a + c + d)) 
    (h₅ : a * (b + c + d) = c * (a + b + d)) 
    (h₆ : a * (b + c + d) = d * (a + b + c)) : 
    a^2 = b^2 ∧ b^2 = c^2 ∧ c^2 = d^2 := 
by
  sorry

end squares_are_equal_l443_44383


namespace find_sets_l443_44302

theorem find_sets (a b c d : ℕ) (h₁ : 1 < a) (h₂ : a < b) (h₃ : b < c) (h₄ : c < d)
  (h₅ : (abcd - 1) % ((a-1) * (b-1) * (c-1) * (d-1)) = 0) :
  (a = 3 ∧ b = 5 ∧ c = 17 ∧ d = 255) ∨ (a = 2 ∧ b = 4 ∧ c = 10 ∧ d = 80) :=
by
  sorry

end find_sets_l443_44302


namespace rope_segment_equation_l443_44362

theorem rope_segment_equation (x : ℝ) (h1 : 2 - x > 0) :
  x^2 = 2 * (2 - x) :=
by
  sorry

end rope_segment_equation_l443_44362


namespace toys_secured_in_25_minutes_l443_44363

def net_toy_gain_per_minute (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  toys_mom_puts - toys_mia_takes

def total_minutes (total_toys : ℕ) (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  (total_toys - 1) / net_toy_gain_per_minute toys_mom_puts toys_mia_takes + 1

theorem toys_secured_in_25_minutes :
  total_minutes 50 5 3 = 25 :=
by
  sorry

end toys_secured_in_25_minutes_l443_44363


namespace servings_in_box_l443_44312

theorem servings_in_box (total_cereal : ℕ) (serving_size : ℕ) (total_cereal_eq : total_cereal = 18) (serving_size_eq : serving_size = 2) :
  total_cereal / serving_size = 9 :=
by
  sorry

end servings_in_box_l443_44312


namespace number_of_girls_in_school_l443_44325

theorem number_of_girls_in_school (total_students : ℕ) (sample_size : ℕ) (x : ℕ) :
  total_students = 2400 →
  sample_size = 200 →
  2 * x + 10 = sample_size →
  (95 / 200 : ℚ) * (total_students : ℚ) = 1140 :=
by
  intros h_total h_sample h_sampled
  rw [h_total, h_sample] at *
  sorry

end number_of_girls_in_school_l443_44325


namespace card_prob_ace_of_hearts_l443_44382

def problem_card_probability : Prop :=
  let deck_size := 52
  let draw_size := 2
  let ace_hearts := 1
  let total_combinations := Nat.choose deck_size draw_size
  let favorable_combinations := deck_size - ace_hearts
  let probability := favorable_combinations / total_combinations
  probability = 1 / 26

theorem card_prob_ace_of_hearts : problem_card_probability := by
  sorry

end card_prob_ace_of_hearts_l443_44382


namespace min_value_expression_l443_44338

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) ≥ 4 := by
  sorry

end min_value_expression_l443_44338


namespace find_primes_a_l443_44348

theorem find_primes_a :
  ∀ (a : ℕ), (∀ n : ℕ, n < a → Nat.Prime (4 * n * n + a)) → (a = 3 ∨ a = 7) :=
by
  sorry

end find_primes_a_l443_44348


namespace inequality_proof_l443_44307

theorem inequality_proof (a b : ℝ) : 
  a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 := 
by 
  sorry

end inequality_proof_l443_44307


namespace roots_opposite_eq_minus_one_l443_44345

theorem roots_opposite_eq_minus_one (k : ℝ) 
  (h_real_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + x₂ = 0 ∧ x₁ * x₂ = k + 1) :
  k = -1 :=
by
  sorry

end roots_opposite_eq_minus_one_l443_44345


namespace correct_quotient_is_32_l443_44328

-- Definitions based on the conditions
def incorrect_divisor := 12
def correct_divisor := 21
def incorrect_quotient := 56
def dividend := incorrect_divisor * incorrect_quotient -- Given as 672

-- Statement of the theorem
theorem correct_quotient_is_32 :
  dividend / correct_divisor = 32 :=
by
  -- skip the proof
  sorry

end correct_quotient_is_32_l443_44328


namespace sum_of_common_ratios_of_sequences_l443_44301

def arithmetico_geometric_sequence (a b c : ℕ → ℝ) (r : ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n + d ∧ b (n + 1) = r * b n + d

theorem sum_of_common_ratios_of_sequences {m n : ℝ}
    {a1 a2 a3 b1 b2 b3 : ℝ}
    (p q : ℝ)
    (h_a1 : a1 = m)
    (h_a2 : a2 = m * p + 5)
    (h_a3 : a3 = m * p^2 + 5 * p + 5)
    (h_b1 : b1 = n)
    (h_b2 : b2 = n * q + 5)
    (h_b3 : b3 = n * q^2 + 5 * q + 5)
    (h_cond : a3 - b3 = 3 * (a2 - b2)) :
    p + q = 4 :=
by
  sorry

end sum_of_common_ratios_of_sequences_l443_44301


namespace range_of_m_l443_44358

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + (m + 2) * x + (m + 5) = 0 → 0 < x) → (-5 < m ∧ m ≤ -4) :=
by
  sorry

end range_of_m_l443_44358


namespace bruce_goals_l443_44371

theorem bruce_goals (B M : ℕ) (h1 : M = 3 * B) (h2 : B + M = 16) : B = 4 :=
by {
  -- Omitted proof
  sorry
}

end bruce_goals_l443_44371


namespace negation_of_square_zero_l443_44310

variable {m : ℝ}

def is_positive (m : ℝ) : Prop := m > 0
def square_is_zero (m : ℝ) : Prop := m^2 = 0

theorem negation_of_square_zero (h : ∀ m, is_positive m → square_is_zero m) :
  ∀ m, ¬ is_positive m → ¬ square_is_zero m := 
sorry

end negation_of_square_zero_l443_44310


namespace cauliflower_production_proof_l443_44347

theorem cauliflower_production_proof (x y : ℕ) 
  (h1 : y^2 - x^2 = 401)
  (hx : x > 0)
  (hy : y > 0) :
  y^2 = 40401 :=
by
  sorry

end cauliflower_production_proof_l443_44347


namespace hyperbola_eccentricity_l443_44315

theorem hyperbola_eccentricity (a b : ℝ) (h : ∃ P : ℝ × ℝ, ∃ A : ℝ × ℝ, ∃ F : ℝ × ℝ, 
  (∃ c : ℝ, F = (c, 0) ∧ A = (-a, 0) ∧ P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1 ∧ 
  (F.fst - P.fst) ^ 2 + P.snd ^ 2 = (F.fst + a) ^ 2 ∧ (F.fst - A.fst) ^ 2 + (F.snd - A.snd) ^ 2 = (F.fst + a) ^ 2 ∧ 
  (P.snd = F.snd) ∧ (abs (F.fst - A.fst) = abs (F.fst - P.fst)))) : 
∃ e : ℝ, e = 2 :=
by
  sorry

end hyperbola_eccentricity_l443_44315


namespace sector_angle_degree_measure_l443_44344

-- Define the variables and conditions
variables (θ r : ℝ)
axiom h1 : (1 / 2) * θ * r^2 = 1
axiom h2 : 2 * r + θ * r = 4

-- Define the theorem to be proved
theorem sector_angle_degree_measure (θ r : ℝ) (h1 : (1 / 2) * θ * r^2 = 1) (h2 : 2 * r + θ * r = 4) : θ = 2 :=
sorry

end sector_angle_degree_measure_l443_44344


namespace correct_operation_l443_44351

theorem correct_operation :
  (3 * m^2 + 4 * m^2 ≠ 7 * m^4) ∧
  (4 * m^3 * 5 * m^3 ≠ 20 * m^3) ∧
  ((-2 * m)^3 ≠ -6 * m^3) ∧
  (m^10 / m^5 = m^5) :=
by
  sorry

end correct_operation_l443_44351


namespace additional_charge_per_2_5_mile_l443_44306

theorem additional_charge_per_2_5_mile (x : ℝ) : 
  (∀ (total_charge distance charge_per_segment initial_fee : ℝ),
    total_charge = 5.65 →
    initial_fee = 2.5 →
    distance = 3.6 →
    charge_per_segment = (3.6 / (2/5)) →
    total_charge = initial_fee + charge_per_segment * x → 
    x = 0.35) :=
by
  intros total_charge distance charge_per_segment initial_fee
  intros h_total_charge h_initial_fee h_distance h_charge_per_segment h_eq
  sorry

end additional_charge_per_2_5_mile_l443_44306


namespace calculate_expression_l443_44374

theorem calculate_expression (y : ℤ) (hy : y = 2) : (3 * y + 4)^2 = 100 :=
by
  sorry

end calculate_expression_l443_44374


namespace trig_identity_proof_l443_44364

theorem trig_identity_proof
  (α : ℝ)
  (h : Real.sin (α - π / 6) = 3 / 5) :
  Real.cos (2 * π / 3 - α) = 3 / 5 :=
sorry

end trig_identity_proof_l443_44364


namespace wire_cut_l443_44300

theorem wire_cut (x : ℝ) (h1 : x + (100 - x) = 100) (h2 : x = (7/13) * (100 - x)) : x = 35 :=
sorry

end wire_cut_l443_44300


namespace cubic_solution_identity_l443_44377

theorem cubic_solution_identity {a b c : ℕ} 
  (h1 : a + b + c = 6) 
  (h2 : ab + bc + ca = 11) 
  (h3 : abc = 6) : 
  (ab / c) + (bc / a) + (ca / b) = 49 / 6 := 
by 
  sorry

end cubic_solution_identity_l443_44377


namespace arithmetic_seq_k_l443_44342

theorem arithmetic_seq_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) 
  (h1 : a 1 = -3)
  (h2 : a (k + 1) = 3 / 2)
  (h3 : S k = -12)
  (h4 : ∀ n, S n = n * (a 1 + a (n+1)) / 2):
  k = 13 :=
sorry

end arithmetic_seq_k_l443_44342


namespace ellipse_focal_distance_l443_44332

theorem ellipse_focal_distance :
  let a := 9
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 4 * Real.sqrt 14 :=
by
  sorry

end ellipse_focal_distance_l443_44332


namespace inequality_solution_l443_44381

theorem inequality_solution :
  ∀ x : ℝ, ( (x - 3) / ( (x - 2) ^ 2 ) < 0 ) ↔ ( x < 2 ∨ (2 < x ∧ x < 3) ) :=
by
  sorry

end inequality_solution_l443_44381


namespace G5_units_digit_is_0_l443_44392

def power_mod (base : ℕ) (exp : ℕ) (modulus : ℕ) : ℕ :=
  (base ^ exp) % modulus

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 2

theorem G5_units_digit_is_0 : (G 5) % 10 = 0 :=
by
  sorry

end G5_units_digit_is_0_l443_44392


namespace jellybean_removal_l443_44309

theorem jellybean_removal 
    (initial_count : ℕ) 
    (first_removal : ℕ) 
    (added_back : ℕ) 
    (final_count : ℕ)
    (initial_count_eq : initial_count = 37)
    (first_removal_eq : first_removal = 15)
    (added_back_eq : added_back = 5)
    (final_count_eq : final_count = 23) :
    (initial_count - first_removal + added_back - final_count) = 4 :=
by 
    sorry

end jellybean_removal_l443_44309


namespace scientific_notation_of_graphene_l443_44318

theorem scientific_notation_of_graphene :
  0.00000000034 = 3.4 * 10^(-10) :=
sorry

end scientific_notation_of_graphene_l443_44318


namespace pow_two_gt_cube_l443_44376

theorem pow_two_gt_cube (n : ℕ) (h : 10 ≤ n) : 2^n > n^3 := sorry

end pow_two_gt_cube_l443_44376


namespace original_number_is_correct_l443_44356

theorem original_number_is_correct (x : ℝ) (h : 10 * x = x + 34.65) : x = 3.85 :=
sorry

end original_number_is_correct_l443_44356


namespace degrees_of_interior_angles_l443_44379

-- Definitions for the problem conditions
variables {a b c h_a h_b S : ℝ} 
variables (ABC : Triangle) 
variables (height_to_bc height_to_ac : ℝ)
variables (le_a_ha : a ≤ height_to_bc)
variables (le_b_hb : b ≤ height_to_ac)
variables (area : S = 1 / 2 * a * height_to_bc)
variables (area_eq : S = 1 / 2 * b * height_to_ac)
variables (ha_eq : height_to_bc = 2 * S / a)
variables (hb_eq : height_to_ac = 2 * S / b)
variables (height_pos : 0 < 2 * S)
variables (length_pos : 0 < a ∧ 0 < b ∧ 0 < c)

-- Conclude the degrees of the interior angles
theorem degrees_of_interior_angles : 
  ∃ A B C : ℝ, A = 45 ∧ B = 45 ∧ C = 90 :=
sorry

end degrees_of_interior_angles_l443_44379


namespace initial_distance_proof_l443_44330

noncomputable def initial_distance (V_A V_B T : ℝ) : ℝ :=
  (V_A * T) + (V_B * T)

theorem initial_distance_proof 
  (V_A V_B : ℝ) 
  (T : ℝ) 
  (h1 : V_A / V_B = 5 / 6)
  (h2 : V_B = 90)
  (h3 : T = 8 / 15) :
  initial_distance V_A V_B T = 88 := 
by
  -- proof goes here
  sorry

end initial_distance_proof_l443_44330


namespace sum_of_D_coordinates_l443_44304

noncomputable def sum_of_coordinates_of_D (D : ℝ × ℝ) (M C : ℝ × ℝ) : ℝ :=
  D.1 + D.2

theorem sum_of_D_coordinates (D M C : ℝ × ℝ) (H_M_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) 
                             (H_M_value : M = (5, 9)) (H_C_value : C = (11, 5)) : 
                             sum_of_coordinates_of_D D M C = 12 :=
sorry

end sum_of_D_coordinates_l443_44304


namespace garden_area_in_square_meters_l443_44394

def garden_width_cm : ℕ := 500
def garden_length_cm : ℕ := 800
def conversion_factor_cm2_to_m2 : ℕ := 10000

theorem garden_area_in_square_meters : (garden_length_cm * garden_width_cm) / conversion_factor_cm2_to_m2 = 40 :=
by
  sorry

end garden_area_in_square_meters_l443_44394


namespace evaluate_expression_l443_44308

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l443_44308


namespace sandy_books_from_first_shop_l443_44320

theorem sandy_books_from_first_shop 
  (cost_first_shop : ℕ)
  (books_second_shop : ℕ)
  (cost_second_shop : ℕ)
  (average_price : ℕ)
  (total_cost : ℕ)
  (total_books : ℕ)
  (num_books_first_shop : ℕ) :
  cost_first_shop = 1480 →
  books_second_shop = 55 →
  cost_second_shop = 920 →
  average_price = 20 →
  total_cost = cost_first_shop + cost_second_shop →
  total_books = total_cost / average_price →
  num_books_first_shop + books_second_shop = total_books →
  num_books_first_shop = 65 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end sandy_books_from_first_shop_l443_44320


namespace cookie_count_l443_44368

theorem cookie_count (C : ℕ) 
  (h1 : 3 * C / 4 + 1 * (C / 4) / 5 + 1 * (C / 4) * 4 / 20 = 10) 
  (h2: 1 * (5 * 4 / 20) / 10 = 1): 
  C = 100 :=
by 
sorry

end cookie_count_l443_44368


namespace number_of_seniors_in_statistics_l443_44327

theorem number_of_seniors_in_statistics (total_students : ℕ) (half_enrolled_in_statistics : ℕ) (percentage_seniors : ℚ) (students_in_statistics seniors_in_statistics : ℕ) 
(h1 : total_students = 120)
(h2 : half_enrolled_in_statistics = total_students / 2)
(h3 : students_in_statistics = half_enrolled_in_statistics)
(h4 : percentage_seniors = 0.90)
(h5 : seniors_in_statistics = students_in_statistics * percentage_seniors) : 
seniors_in_statistics = 54 := 
by sorry

end number_of_seniors_in_statistics_l443_44327


namespace min_value_expr_l443_44361

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (a / c) ≥ 4 := 
sorry

end min_value_expr_l443_44361


namespace largest_of_decimals_l443_44303

theorem largest_of_decimals :
  let a := 0.993
  let b := 0.9899
  let c := 0.990
  let d := 0.989
  let e := 0.9909
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by
  sorry

end largest_of_decimals_l443_44303


namespace eunseo_change_correct_l443_44359

-- Define the given values
def r : ℕ := 3
def p_r : ℕ := 350
def b : ℕ := 2
def p_b : ℕ := 180
def P : ℕ := 2000

-- Define the total cost of candies and the change
def total_cost := r * p_r + b * p_b
def change := P - total_cost

-- Theorem statement
theorem eunseo_change_correct : change = 590 := by
  -- proof not required, so using sorry
  sorry

end eunseo_change_correct_l443_44359


namespace condition_A_is_necessary_but_not_sufficient_for_condition_B_l443_44324

-- Define conditions
variables (a b : ℝ)

-- Condition A: ab > 0
def condition_A : Prop := a * b > 0

-- Condition B: a > 0 and b > 0
def condition_B : Prop := a > 0 ∧ b > 0

-- Prove that condition_A is a necessary but not sufficient condition for condition_B
theorem condition_A_is_necessary_but_not_sufficient_for_condition_B :
  (condition_A a b → condition_B a b) ∧ ¬(condition_B a b → condition_A a b) :=
by
  sorry

end condition_A_is_necessary_but_not_sufficient_for_condition_B_l443_44324


namespace find_A_l443_44366

theorem find_A (A : ℤ) (h : A + 10 = 15) : A = 5 :=
sorry

end find_A_l443_44366


namespace rankings_are_correct_l443_44326

-- Define teams:
inductive Team
| A | B | C | D

-- Define the type for ranking
structure Ranking :=
  (first : Team)
  (second : Team)
  (third : Team)
  (last : Team)

-- Define the predictions of Jia, Yi, and Bing
structure Predictions := 
  (Jia : Ranking)
  (Yi : Ranking)
  (Bing : Ranking)

-- Define the condition that each prediction is half right, half wrong
def isHalfRightHalfWrong (pred : Ranking) (actual : Ranking) : Prop :=
  (pred.first = actual.first ∨ pred.second = actual.second ∨ pred.third = actual.third ∨ pred.last = actual.last) ∧
  (pred.first ≠ actual.first ∨ pred.second ≠ actual.second ∨ pred.third ≠ actual.third ∨ pred.last ≠ actual.last)

-- Define the actual rankings
def actualRanking : Ranking := { first := Team.C, second := Team.A, third := Team.D, last := Team.B }

-- Define Jia's Predictions 
def JiaPrediction : Ranking := { first := Team.C, second := Team.C, third := Team.D, last := Team.D }

-- Define Yi's Predictions 
def YiPrediction : Ranking := { first := Team.B, second := Team.A, third := Team.C, last := Team.D }

-- Define Bing's Predictions 
def BingPrediction : Ranking := { first := Team.C, second := Team.B, third := Team.A, last := Team.D }

-- Create an instance of predictions
def pred : Predictions := { Jia := JiaPrediction, Yi := YiPrediction, Bing := BingPrediction }

-- The theorem to be proved
theorem rankings_are_correct :
  isHalfRightHalfWrong pred.Jia actualRanking ∧ 
  isHalfRightHalfWrong pred.Yi actualRanking ∧ 
  isHalfRightHalfWrong pred.Bing actualRanking →
  actualRanking.first = Team.C ∧ actualRanking.second = Team.A ∧ actualRanking.third = Team.D ∧ 
  actualRanking.last = Team.B :=
by
  sorry -- Proof is not required.

end rankings_are_correct_l443_44326


namespace girls_more_than_boys_l443_44355

/-- 
In a class with 42 students, where the ratio of boys to girls is 3:4, 
prove that there are 6 more girls than boys.
-/
theorem girls_more_than_boys (students total_students : ℕ) (boys girls : ℕ) (ratio_boys_girls : 3 * girls = 4 * boys)
  (total_students_count : boys + girls = total_students)
  (total_students_value : total_students = 42) : girls - boys = 6 :=
by
  sorry

end girls_more_than_boys_l443_44355


namespace continuity_at_three_l443_44385

noncomputable def f (x : ℝ) : ℝ := -2 * x ^ 2 - 4

theorem continuity_at_three (ε : ℝ) (hε : 0 < ε) :
  ∃ δ > 0, ∀ x : ℝ, |x - 3| < δ → |f x - f 3| < ε :=
sorry

end continuity_at_three_l443_44385


namespace zoo_rabbits_count_l443_44350

theorem zoo_rabbits_count (parrots rabbits : ℕ) (h_ratio : parrots * 4 = rabbits * 3) (h_parrots_count : parrots = 21) : rabbits = 28 :=
by
  sorry

end zoo_rabbits_count_l443_44350


namespace roots_of_unity_cubic_l443_44314

noncomputable def countRootsOfUnityCubic (c d e : ℤ) : ℕ := sorry

theorem roots_of_unity_cubic :
  ∃ (z : ℂ) (n : ℕ), (z^n = 1) ∧ (∃ (c d e : ℤ), z^3 + c * z^2 + d * z + e = 0)
  ∧ countRootsOfUnityCubic c d e = 12 :=
sorry

end roots_of_unity_cubic_l443_44314


namespace surface_area_of_4cm_cube_after_corner_removal_l443_44334

noncomputable def surface_area_after_corner_removal (cube_side original_surface_length corner_cube_side : ℝ) : ℝ := 
  let num_faces : ℕ := 6
  let num_corners : ℕ := 8
  let surface_area_one_face := cube_side * cube_side
  let original_surface_area := num_faces * surface_area_one_face
  let corner_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let exposed_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let net_change_per_corner_cube := -corner_surface_area_one_face + exposed_surface_area_one_face
  let total_change := num_corners * net_change_per_corner_cube
  original_surface_area + total_change

theorem surface_area_of_4cm_cube_after_corner_removal : 
  ∀ (cube_side original_surface_length corner_cube_side : ℝ), 
  cube_side = 4 ∧ original_surface_length = 4 ∧ corner_cube_side = 2 →
  surface_area_after_corner_removal cube_side original_surface_length corner_cube_side = 96 :=
by
  intros cube_side original_surface_length corner_cube_side h
  rcases h with ⟨hs, ho, hc⟩
  rw [hs, ho, hc]
  sorry

end surface_area_of_4cm_cube_after_corner_removal_l443_44334


namespace factor_expression_l443_44386

theorem factor_expression (x : ℝ) : 
  4 * x * (x - 5) + 6 * (x - 5) = (4 * x + 6) * (x - 5) :=
by 
  sorry

end factor_expression_l443_44386


namespace match_Tile_C_to_Rectangle_III_l443_44333

-- Define the structure for a Tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the given tiles
def Tile_A : Tile := { top := 5, right := 3, bottom := 7, left := 2 }
def Tile_B : Tile := { top := 3, right := 6, bottom := 2, left := 8 }
def Tile_C : Tile := { top := 7, right := 9, bottom := 1, left := 3 }
def Tile_D : Tile := { top := 1, right := 8, bottom := 5, left := 9 }

-- The proof problem: Prove that Tile C should be matched to Rectangle III
theorem match_Tile_C_to_Rectangle_III : (Tile_C = { top := 7, right := 9, bottom := 1, left := 3 }) → true := 
by
  intros
  sorry

end match_Tile_C_to_Rectangle_III_l443_44333


namespace decimal_to_fraction_l443_44337

theorem decimal_to_fraction : 2.36 = 59 / 25 :=
by
  sorry

end decimal_to_fraction_l443_44337


namespace find_a_for_exponential_function_l443_44323

theorem find_a_for_exponential_function (a : ℝ) :
  a - 2 = 1 ∧ a > 0 ∧ a ≠ 1 → a = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end find_a_for_exponential_function_l443_44323


namespace profit_starts_from_third_year_most_beneficial_option_l443_44390

-- Define the conditions of the problem
def investment_cost := 144
def maintenance_cost (n : ℕ) := 4 * n^2 + 20 * n
def revenue_per_year := 1

-- Define the net profit function
def net_profit (n : ℕ) : ℤ :=
(revenue_per_year * n : ℤ) - (maintenance_cost n) - investment_cost

-- Question 1: Prove the project starts to make a profit from the 3rd year
theorem profit_starts_from_third_year (n : ℕ) (h : 2 < n ∧ n < 18) : 
net_profit n > 0 ↔ 3 ≤ n := sorry

-- Question 2: Prove the most beneficial option for company's development
theorem most_beneficial_option : (∃ o, o = 1) ∧ (∃ t1 t2, t1 = 264 ∧ t2 = 264 ∧ t1 < t2) := sorry

end profit_starts_from_third_year_most_beneficial_option_l443_44390


namespace combined_river_length_estimate_l443_44339

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end combined_river_length_estimate_l443_44339


namespace rick_group_division_l443_44354

theorem rick_group_division :
  ∀ (total_books : ℕ), total_books = 400 → 
  (∃ n : ℕ, (∀ (books_per_category : ℕ) (divisions : ℕ), books_per_category = total_books / (2 ^ divisions) → books_per_category = 25 → divisions = n) ∧ n = 4) :=
by
  sorry

end rick_group_division_l443_44354


namespace amount_used_to_pay_l443_44370

noncomputable def the_cost_of_football : ℝ := 9.14
noncomputable def the_cost_of_baseball : ℝ := 6.81
noncomputable def the_change_received : ℝ := 4.05

theorem amount_used_to_pay : 
    (the_cost_of_football + the_cost_of_baseball + the_change_received) = 20.00 := 
by
  sorry

end amount_used_to_pay_l443_44370


namespace simplify_expression_l443_44384

theorem simplify_expression : ( (2^8 + 4^5) * (2^3 - (-2)^3) ^ 8 ) = 0 := 
by sorry

end simplify_expression_l443_44384


namespace Amanda_tickets_third_day_l443_44329

theorem Amanda_tickets_third_day :
  (let total_tickets := 80
   let first_day_tickets := 5 * 4
   let second_day_tickets := 32

   total_tickets - (first_day_tickets + second_day_tickets) = 28) :=
by
  sorry

end Amanda_tickets_third_day_l443_44329


namespace man_is_older_by_l443_44321

theorem man_is_older_by :
  ∀ (M S : ℕ), S = 22 → (M + 2) = 2 * (S + 2) → (M - S) = 24 :=
by
  intros M S h1 h2
  sorry

end man_is_older_by_l443_44321


namespace find_constant_l443_44343

theorem find_constant (N : ℝ) (C : ℝ) (h1 : N = 12.0) (h2 : C + 0.6667 * N = 0.75 * N) : C = 0.9996 :=
by
  sorry

end find_constant_l443_44343


namespace can_still_row_probability_l443_44336

/-- Define the probabilities for the left and right oars --/
def P_left1_work : ℚ := 3 / 5
def P_left2_work : ℚ := 2 / 5
def P_right1_work : ℚ := 4 / 5 
def P_right2_work : ℚ := 3 / 5

/-- Define the probabilities of the failures as complementary probabilities --/
def P_left1_fail : ℚ := 1 - P_left1_work
def P_left2_fail : ℚ := 1 - P_left2_work
def P_right1_fail : ℚ := 1 - P_right1_work
def P_right2_fail : ℚ := 1 - P_right2_work

/-- Define the probability of both left oars failing --/
def P_both_left_fail : ℚ := P_left1_fail * P_left2_fail

/-- Define the probability of both right oars failing --/
def P_both_right_fail : ℚ := P_right1_fail * P_right2_fail

/-- Define the probability of all four oars failing --/
def P_all_fail : ℚ := P_both_left_fail * P_both_right_fail

/-- Calculate the probability that at least one oar on each side works --/
def P_can_row : ℚ := 1 - (P_both_left_fail + P_both_right_fail - P_all_fail)

theorem can_still_row_probability :
  P_can_row = 437 / 625 :=
by {
  -- The proof is to be completed
  sorry
}

end can_still_row_probability_l443_44336


namespace minimize_quadratic_function_l443_44399

def quadratic_function (x : ℝ) : ℝ := x^2 + 8*x + 7

theorem minimize_quadratic_function : ∃ x : ℝ, (∀ y : ℝ, quadratic_function y ≥ quadratic_function x) ∧ x = -4 :=
by
  sorry

end minimize_quadratic_function_l443_44399


namespace honors_students_count_l443_44378

variable {total_students : ℕ}
variable {total_girls total_boys : ℕ}
variable {honors_girls honors_boys : ℕ}

axiom class_size_constraint : total_students < 30
axiom prob_girls_honors : (honors_girls : ℝ) / total_girls = 3 / 13
axiom prob_boys_honors : (honors_boys : ℝ) / total_boys = 4 / 11
axiom total_students_eq : total_students = total_girls + total_boys
axiom honors_girls_value : honors_girls = 3
axiom honors_boys_value : honors_boys = 4

theorem honors_students_count : 
  honors_girls + honors_boys = 7 :=
by
  sorry

end honors_students_count_l443_44378


namespace extra_flowers_correct_l443_44360

variable (pickedTulips : ℕ) (pickedRoses : ℕ) (usedFlowers : ℕ)

def totalFlowers : ℕ := pickedTulips + pickedRoses
def extraFlowers : ℕ := totalFlowers pickedTulips pickedRoses - usedFlowers

theorem extra_flowers_correct : 
  pickedTulips = 39 → pickedRoses = 49 → usedFlowers = 81 → extraFlowers pickedTulips pickedRoses usedFlowers = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end extra_flowers_correct_l443_44360


namespace range_of_m_l443_44346

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x^2 - x - m = 0) : m ≥ -1/4 :=
by
  sorry

end range_of_m_l443_44346


namespace star_polygon_internal_angles_sum_l443_44319

-- Define the core aspects of the problem using type defintions and axioms.
def n_star_polygon_total_internal_angle_sum (n : ℕ) : ℝ :=
  180 * (n - 4)

theorem star_polygon_internal_angles_sum (n : ℕ) (h : n ≥ 6) :
  n_star_polygon_total_internal_angle_sum n = 180 * (n - 4) :=
by
  -- This step would involve the formal proof using Lean
  sorry

end star_polygon_internal_angles_sum_l443_44319


namespace number_power_eq_l443_44349

theorem number_power_eq (x : ℕ) (h : x^10 = 16^5) : x = 4 :=
by {
  -- Add supporting calculations here if needed
  sorry
}

end number_power_eq_l443_44349


namespace max_good_triplets_l443_44341

-- Define the problem's conditions
variables (k : ℕ) (h_pos : 0 < k)

-- The statement to be proven
theorem max_good_triplets : ∃ T, T = 12 * k ^ 4 := 
sorry

end max_good_triplets_l443_44341


namespace solve_diophantine_l443_44375

theorem solve_diophantine : ∃ (x y : ℕ) (t : ℤ), x = 4 - 43 * t ∧ y = 6 - 65 * t ∧ t ≤ 0 ∧ 65 * x - 43 * y = 2 :=
by
  sorry

end solve_diophantine_l443_44375


namespace race_distance_l443_44369

/-- Given that Sasha, Lesha, and Kolya start a 100m race simultaneously and run at constant velocities,
when Sasha finishes, Lesha is 10m behind, and when Lesha finishes, Kolya is 10m behind.
Prove that the distance between Sasha and Kolya when Sasha finishes is 19 meters. -/
theorem race_distance
    (v_S v_L v_K : ℝ)
    (h1 : 100 / v_S - 100 / v_L = 10 / v_L)
    (h2 : 100 / v_L - 100 / v_K = 10 / v_K) :
    100 - 81 = 19 :=
by
  sorry

end race_distance_l443_44369


namespace cycle_time_to_library_l443_44398

theorem cycle_time_to_library 
  (constant_speed : Prop)
  (time_to_park : ℕ)
  (distance_to_park : ℕ)
  (distance_to_library : ℕ)
  (h1 : constant_speed)
  (h2 : time_to_park = 30)
  (h3 : distance_to_park = 5)
  (h4 : distance_to_library = 3) :
  (18 : ℕ) = (30 * distance_to_library / distance_to_park) :=
by
  intros
  -- The proof would go here
  sorry

end cycle_time_to_library_l443_44398


namespace train_length_l443_44389

/-- A train crosses a tree in 120 seconds. It takes 230 seconds to pass a platform 1100 meters long.
    How long is the train? -/
theorem train_length (L : ℝ) (V : ℝ)
    (h1 : V = L / 120)
    (h2 : V = (L + 1100) / 230) :
    L = 1200 :=
by
  sorry

end train_length_l443_44389


namespace fifth_term_in_geometric_sequence_l443_44316

variable (y : ℝ)

def geometric_sequence : ℕ → ℝ
| 0       => 3
| (n + 1) => geometric_sequence n * (3 * y)

theorem fifth_term_in_geometric_sequence (y : ℝ) : 
  geometric_sequence y 4 = 243 * y^4 :=
sorry

end fifth_term_in_geometric_sequence_l443_44316


namespace maxValue_is_6084_over_17_l443_44380

open Real

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem maxValue_is_6084_over_17 (x y : ℝ) (h : x + y = 5) :
  maxValue x y h ≤ 6084 / 17 := 
sorry

end maxValue_is_6084_over_17_l443_44380


namespace Jake_has_fewer_peaches_l443_44340

def Steven_peaches := 14
def Jill_peaches := 5
def Jake_peaches := Jill_peaches + 3

theorem Jake_has_fewer_peaches : Steven_peaches - Jake_peaches = 6 :=
by
  sorry

end Jake_has_fewer_peaches_l443_44340
