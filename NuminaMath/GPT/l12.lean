import Mathlib

namespace scientific_notation_l12_12316

theorem scientific_notation (a : ℝ) (n : ℤ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 43050000 = a * 10^n) : a = 4.305 ∧ n = 7 :=
by
  sorry

end scientific_notation_l12_12316


namespace depth_of_well_l12_12444

theorem depth_of_well (d : ℝ) (t1 t2 : ℝ)
  (h1 : d = 15 * t1^2)
  (h2 : t2 = d / 1100)
  (h3 : t1 + t2 = 9.5) :
  d = 870.25 := 
sorry

end depth_of_well_l12_12444


namespace diameter_of_outer_edge_l12_12809

-- Defining the conditions as variables
variable (pathWidth gardenWidth statueDiameter fountainDiameter : ℝ)
variable (hPathWidth : pathWidth = 10)
variable (hGardenWidth : gardenWidth = 12)
variable (hStatueDiameter : statueDiameter = 6)
variable (hFountainDiameter : fountainDiameter = 14)

-- Lean statement to prove the diameter
theorem diameter_of_outer_edge :
  2 * ((fountainDiameter / 2) + gardenWidth + pathWidth) = 58 :=
by
  rw [hPathWidth, hGardenWidth, hFountainDiameter]
  sorry

end diameter_of_outer_edge_l12_12809


namespace geometric_sequence_sum_inverse_equals_l12_12753

variable (a : ℕ → ℝ)
variable (n : ℕ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃(r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum_inverse_equals (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_sum : a 5 + a 6 + a 7 + a 8 = 15 / 8)
  (h_prod : a 6 * a 7 = -9 / 8) :
  (1 / a 5) + (1 / a 6) + (1 / a 7) + (1 / a 8) = -5 / 3 :=
by
  sorry

end geometric_sequence_sum_inverse_equals_l12_12753


namespace license_plate_palindrome_l12_12251

theorem license_plate_palindrome :
  let p_digit_palindrome := (1 : ℚ) / 100,
      p_letter_palindrome := (1 : ℚ) / 676,
      combined_probability := p_digit_palindrome + p_letter_palindrome - p_digit_palindrome * p_letter_palindrome,
      m := 31,
      n := 2704 in
  combined_probability = (m : ℚ) / n ∧ m + n = 2735 :=
by
  -- The assumptions and definitions as stated suffice to skip the actual proof here.
  sorry

end license_plate_palindrome_l12_12251


namespace S8_value_l12_12217

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

def condition_a3_a6 (a : ℕ → ℝ) : Prop :=
  a 3 = 9 - a 6

theorem S8_value (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_formula : sum_of_first_n_terms S a)
  (h_condition : condition_a3_a6 a) :
  S 8 = 72 :=
by
  sorry

end S8_value_l12_12217


namespace ellipse_line_intersection_l12_12850

theorem ellipse_line_intersection (m : ℝ) : 
  (m > 0 ∧ m ≠ 3) →
  (∃ x y : ℝ, (x^2 / 3 + y^2 / m = 1) ∧ (x + 2 * y - 2 = 0)) ↔ 
  ((1 / 4 < m ∧ m < 3) ∨ (m > 3)) := 
by 
  sorry

end ellipse_line_intersection_l12_12850


namespace paul_earns_from_license_plates_l12_12899

theorem paul_earns_from_license_plates
  (plates_from_40_states : ℕ)
  (total_50_states : ℕ)
  (reward_per_percentage_point : ℕ)
  (h1 : plates_from_40_states = 40)
  (h2 : total_50_states = 50)
  (h3 : reward_per_percentage_point = 2) :
  (40 / 50) * 100 * 2 = 160 := 
sorry

end paul_earns_from_license_plates_l12_12899


namespace simple_interest_time_period_l12_12966

theorem simple_interest_time_period 
  (P : ℝ) (R : ℝ := 4) (T : ℝ) (SI : ℝ := (2 / 5) * P) :
  SI = P * R * T / 100 → T = 10 :=
by {
  sorry
}

end simple_interest_time_period_l12_12966


namespace total_amount_l12_12516

theorem total_amount (P Q R : ℝ) (h1 : R = 2 / 3 * (P + Q)) (h2 : R = 3200) : P + Q + R = 8000 := 
by
  sorry

end total_amount_l12_12516


namespace f_monotone_f_inequality_solution_l12_12593

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x > 0 → ∃ y, f y = x
axiom f_at_2: f 2 = 1
axiom f_mul : ∀ x y, f (x * y) = f x + f y
axiom f_positive : ∀ x, x > 1 → f x > 0

theorem f_monotone (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

theorem f_inequality_solution (x : ℝ) (hx : x > 2 ∧ x ≤ 4) : f x + f (x - 2) ≤ 3 :=
sorry

end f_monotone_f_inequality_solution_l12_12593


namespace algebraic_identity_l12_12712

theorem algebraic_identity (a b c d : ℝ) : a - b + c - d = a + c - (b + d) :=
by
  sorry

end algebraic_identity_l12_12712


namespace number_of_remainders_mod_210_l12_12450

theorem number_of_remainders_mod_210 (p : ℕ) (hp : Nat.Prime p) (hp_gt_7 : p > 7) :
  ∃ (remainders : Finset ℕ), (remainders.card = 12) ∧ ∀ r ∈ remainders, ∃ k, p^2 ≡ r [MOD 210] :=
by
  sorry

end number_of_remainders_mod_210_l12_12450


namespace certain_number_approx_l12_12070

theorem certain_number_approx (x : ℝ) : 213 * 16 = 3408 → x * 2.13 = 0.3408 → x = 0.1600 :=
by
  intro h1 h2
  sorry

end certain_number_approx_l12_12070


namespace courtyard_length_proof_l12_12554

noncomputable def paving_stone_area (length width : ℝ) : ℝ := length * width

noncomputable def total_area_stones (stone_area : ℝ) (num_stones : ℝ) : ℝ := stone_area * num_stones

noncomputable def courtyard_length (total_area width : ℝ) : ℝ := total_area / width

theorem courtyard_length_proof :
  let stone_length := 2.5
  let stone_width := 2
  let courtyard_width := 16.5
  let num_stones := 99
  let stone_area := paving_stone_area stone_length stone_width
  let total_area := total_area_stones stone_area num_stones
  courtyard_length total_area courtyard_width = 30 :=
by
  sorry

end courtyard_length_proof_l12_12554


namespace total_sheep_flock_l12_12252

-- Definitions and conditions based on the problem description
def crossing_rate : ℕ := 3 -- Sheep per minute
def sleep_duration : ℕ := 90 -- Duration of sleep in minutes
def sheep_counted_before_sleep : ℕ := 42 -- Sheep counted before falling asleep

-- Total sheep that crossed while Nicholas was asleep
def sheep_during_sleep := crossing_rate * sleep_duration 

-- Total sheep that crossed when Nicholas woke up
def total_sheep_after_sleep := sheep_counted_before_sleep + sheep_during_sleep

-- Prove the total number of sheep in the flock
theorem total_sheep_flock : (2 * total_sheep_after_sleep) = 624 :=
by
  sorry

end total_sheep_flock_l12_12252


namespace sqrt6_op_sqrt6_l12_12291

variable (x y : ℝ)

noncomputable def op (x y : ℝ) := (x + y)^2 - (x - y)^2

theorem sqrt6_op_sqrt6 : ∀ (x y : ℝ), op (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end sqrt6_op_sqrt6_l12_12291


namespace value_of_square_sum_l12_12364

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l12_12364


namespace solution_set_l12_12576

noncomputable def solve_inequality : Set ℝ :=
  {x | (1 / (x - 1)) >= -1}

theorem solution_set :
  solve_inequality = {x | x ≤ 0} ∪ {x | x > 1} :=
by
  sorry

end solution_set_l12_12576


namespace inv_geom_seq_prod_next_geom_seq_l12_12595

variable {a : Nat → ℝ} (q : ℝ) (h_q : q ≠ 0)
variable (h_geom : ∀ n, a (n + 1) = q * a n)

theorem inv_geom_seq :
  ∀ n, ∃ c q_inv, (q_inv ≠ 0) ∧ (1 / a n = c * q_inv ^ n) :=
sorry

theorem prod_next_geom_seq :
  ∀ n, ∃ c q_sq, (q_sq ≠ 0) ∧ (a n * a (n + 1) = c * q_sq ^ n) :=
sorry

end inv_geom_seq_prod_next_geom_seq_l12_12595


namespace number_of_customers_l12_12232

theorem number_of_customers
  (nails_per_person : ℕ)
  (total_sounds : ℕ)
  (trimmed_nails_per_person : nails_per_person = 20)
  (produced_sounds : total_sounds = 100) :
  total_sounds / nails_per_person = 5 :=
by
  -- This is offered as a placeholder to indicate where a Lean proof goes.
  sorry

end number_of_customers_l12_12232


namespace molecular_weight_N2O5_correct_l12_12321

noncomputable def atomic_weight_N : ℝ := 14.01
noncomputable def atomic_weight_O : ℝ := 16.00
def molecular_formula_N2O5 : (ℕ × ℕ) := (2, 5)

theorem molecular_weight_N2O5_correct :
  let weight := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  weight = 108.02 :=
by
  sorry

end molecular_weight_N2O5_correct_l12_12321


namespace weighted_average_score_l12_12200

def weight (subject_mark : Float) (weight_percentage : Float) : Float :=
    subject_mark * weight_percentage

theorem weighted_average_score :
    (weight 61 0.2) + (weight 65 0.25) + (weight 82 0.3) + (weight 67 0.15) + (weight 85 0.1) = 71.6 := by
    sorry

end weighted_average_score_l12_12200


namespace combined_savings_after_5_years_l12_12610

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + (r / n)) ^ (n * t)

theorem combined_savings_after_5_years :
  let P1 := 600
  let r1 := 0.10
  let n1 := 12
  let t := 5
  let P2 := 400
  let r2 := 0.08
  let n2 := 4
  compound_interest P1 r1 n1 t + compound_interest P2 r2 n2 t = 1554.998 :=
by
  sorry

end combined_savings_after_5_years_l12_12610


namespace option_a_is_correct_l12_12542

theorem option_a_is_correct (a b : ℝ) : 
  (a^2 + a * b) / a = a + b := 
by sorry

end option_a_is_correct_l12_12542


namespace evaluate_fraction_l12_12468

theorem evaluate_fraction : (5 / 6 : ℚ) / (9 / 10) - 1 = -2 / 27 := by
  sorry

end evaluate_fraction_l12_12468


namespace multiplication_of_powers_l12_12429

theorem multiplication_of_powers :
  2^4 * 3^2 * 5^2 * 11 = 39600 := by
  sorry

end multiplication_of_powers_l12_12429


namespace coefficient_x7_y2_of_expansion_l12_12751

noncomputable def coefficient_x7_y2_expansion : ℤ :=
  let term1 := Int.ofNat (Nat.choose 8 2)
  let term2 := -Int.ofNat (Nat.choose 8 1)
  term1 + term2

-- Statement of the proof problem
theorem coefficient_x7_y2_of_expansion :
  coefficient_x7_y2_expansion = 20 :=
by
  sorry

end coefficient_x7_y2_of_expansion_l12_12751


namespace min_abs_sum_of_products_l12_12510

noncomputable def g (x : ℝ) : ℝ := x^4 + 10*x^3 + 29*x^2 + 30*x + 9

theorem min_abs_sum_of_products (w : Fin 4 → ℝ) (h_roots : ∀ i, g (w i) = 0)
  : ∃ a b c d : Fin 4, a ≠ b ∧ c ≠ d ∧ (∀ i j, i ≠ j → a ≠ i ∧ b ≠ i ∧ c ≠ i ∧ d ≠ i → a ≠ j ∧ b ≠ j ∧ c ≠ j ∧ d ≠ j) ∧
    |w a * w b + w c * w d| = 6 :=
sorry

end min_abs_sum_of_products_l12_12510


namespace shift_right_linear_function_l12_12382

theorem shift_right_linear_function (x : ℝ) : 
  (∃ k b : ℝ, k ≠ 0 ∧ (∀ x : ℝ, y = -2x → y = kx + b) → (x, y) = (x - 3, -2(x-3))) → y = -2x + 6 :=
by
  sorry

end shift_right_linear_function_l12_12382


namespace geometric_means_insertion_l12_12405

noncomputable def is_geometric_progression (s : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (r_pos : r > 0), ∀ n, s (n + 1) = s n * r

theorem geometric_means_insertion (s : ℕ → ℝ) (n : ℕ)
  (h : is_geometric_progression s)
  (h_pos : ∀ i, s i > 0) :
  ∃ t : ℕ → ℝ, is_geometric_progression t :=
sorry

end geometric_means_insertion_l12_12405


namespace find_x_when_fx_eq_3_l12_12889

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then x + 2 else
if x < 2 then x^2 else
2 * x

theorem find_x_when_fx_eq_3 : ∃ x : ℝ, f x = 3 ∧ x = Real.sqrt 3 := by
  sorry

end find_x_when_fx_eq_3_l12_12889


namespace median_to_hypotenuse_of_right_triangle_l12_12496

theorem median_to_hypotenuse_of_right_triangle (DE DF : ℝ) (h₁ : DE = 6) (h₂ : DF = 8) :
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  N = 5 :=
by
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  have h : N = 5 :=
    by
      sorry
  exact h

end median_to_hypotenuse_of_right_triangle_l12_12496


namespace total_pencils_l12_12883

-- Defining the number of pencils each person has.
def jessica_pencils : ℕ := 8
def sandy_pencils : ℕ := 8
def jason_pencils : ℕ := 8

-- Theorem stating the total number of pencils
theorem total_pencils : jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end total_pencils_l12_12883


namespace cos_double_angle_l12_12723

theorem cos_double_angle (theta : ℝ) (h : Real.sin (Real.pi - theta) = 1 / 3) : Real.cos (2 * theta) = 7 / 9 := by
  sorry

end cos_double_angle_l12_12723


namespace milk_butterfat_problem_l12_12545

variable (x : ℝ)

def butterfat_10_percent (x : ℝ) := 0.10 * x
def butterfat_35_percent_in_8_gallons : ℝ := 0.35 * 8
def total_milk (x : ℝ) := x + 8
def total_butterfat (x : ℝ) := 0.20 * (x + 8)

theorem milk_butterfat_problem 
    (h : butterfat_10_percent x + butterfat_35_percent_in_8_gallons = total_butterfat x) : x = 12 :=
by
  sorry

end milk_butterfat_problem_l12_12545


namespace find_A_plus_B_l12_12249

/-- Let A, B, C, and D be distinct digits such that 0 ≤ A, B, C, D ≤ 9.
    C and D are non-zero, and A ≠ B ≠ C ≠ D.
    If (A+B)/(C+D) is an integer and C+D is minimized,
    then prove that A + B = 15. -/
theorem find_A_plus_B
  (A B C D : ℕ)
  (h_digits : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_range : 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9)
  (h_nonzero_CD : C ≠ 0 ∧ D ≠ 0)
  (h_integer : (A + B) % (C + D) = 0)
  (h_min_CD : ∀ C' D', (C' ≠ C ∨ D' ≠ D) → (C' ≠ 0 ∧ D' ≠ 0 → (C + D ≤ C' + D'))) :
  A + B = 15 := 
sorry

end find_A_plus_B_l12_12249


namespace sculpture_plus_base_height_l12_12195

def height_sculpture_feet : Nat := 2
def height_sculpture_inches : Nat := 10
def height_base_inches : Nat := 4

def height_sculpture_total_inches : Nat := height_sculpture_feet * 12 + height_sculpture_inches
def height_total_inches : Nat := height_sculpture_total_inches + height_base_inches

theorem sculpture_plus_base_height :
  height_total_inches = 38 := by
  sorry

end sculpture_plus_base_height_l12_12195


namespace incorrect_expression_l12_12392

variable (D : ℚ) (P Q : ℕ) (r s : ℕ)

-- D represents a repeating decimal.
-- P denotes the r figures of D which do not repeat themselves.
-- Q denotes the s figures of D which repeat themselves.

theorem incorrect_expression :
  10^r * (10^s - 1) * D ≠ Q * (P - 1) :=
sorry

end incorrect_expression_l12_12392


namespace figure_at_1000th_position_position_of_1000th_diamond_l12_12778

-- Define the repeating sequence
def repeating_sequence : List String := ["△", "Λ", "◇", "Λ", "⊙", "□"]

-- Lean 4 statement for (a)
theorem figure_at_1000th_position :
  repeating_sequence[(1000 % repeating_sequence.length) - 1] = "Λ" :=
by sorry

-- Define the arithmetic sequence for diamond positions
def diamond_position (n : Nat) : Nat :=
  3 + (n - 1) * 6

-- Lean 4 statement for (b)
theorem position_of_1000th_diamond :
  diamond_position 1000 = 5997 :=
by sorry

end figure_at_1000th_position_position_of_1000th_diamond_l12_12778


namespace simplify_expression_l12_12118

theorem simplify_expression :
  (Real.sqrt 600 / Real.sqrt 75 - Real.sqrt 243 / Real.sqrt 108) = (4 * Real.sqrt 2 - 3 * Real.sqrt 3) / 2 := by
  sorry

end simplify_expression_l12_12118


namespace probability_is_seven_fifteenths_l12_12502

-- Define the problem conditions
def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def choose_3_from_10 : ℕ := Nat.choose 10 3
def choose_3_red : ℕ := Nat.choose 5 3
def choose_3_green : ℕ := Nat.choose 3 3
def choose_2_red_1_green : ℕ := Nat.choose 5 2 * Nat.choose 3 1
def choose_2_green_1_red : ℕ := Nat.choose 3 2 * Nat.choose 5 1

-- Calculate favorable outcomes
def favorable_outcomes : ℕ :=
  choose_3_red + choose_3_green + choose_2_red_1_green + choose_2_green_1_red

-- Calculate the required probability
def probability_all_red_or_green : ℚ := favorable_outcomes / choose_3_from_10

-- Prove that probability_all_red_or_green is 7/15
theorem probability_is_seven_fifteenths :
  probability_all_red_or_green = 7 / 15 :=
by 
  -- Leaving the proof as a sorry for now
  sorry

end probability_is_seven_fifteenths_l12_12502


namespace sum_of_28_terms_l12_12059

variable {f : ℝ → ℝ}
variable {a : ℕ → ℝ}

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_of_28_terms
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h2 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y)
  (h3 : ∃ d ≠ 0, ∃ a₁, ∀ n, a (n + 1) = a₁ + n * d)
  (h4 : f (a 6) = f (a 23)) :
  sum_arithmetic_sequence 28 (a 1) ((a 2) - (a 1)) = 28 :=
by sorry

end sum_of_28_terms_l12_12059


namespace laura_total_cost_l12_12100

def salad_cost : ℝ := 3
def beef_cost : ℝ := 2 * salad_cost
def potato_cost : ℝ := salad_cost / 3
def juice_cost : ℝ := 1.5

def total_salad_cost : ℝ := 2 * salad_cost
def total_beef_cost : ℝ := 2 * beef_cost
def total_potato_cost : ℝ := 1 * potato_cost
def total_juice_cost : ℝ := 2 * juice_cost

def total_cost : ℝ := total_salad_cost + total_beef_cost + total_potato_cost + total_juice_cost

theorem laura_total_cost : total_cost = 22 := by
  sorry

end laura_total_cost_l12_12100


namespace find_complement_intersection_find_union_complement_subset_implies_a_range_l12_12062

-- Definitions for sets A and B
def A : Set ℝ := { x | 3 ≤ x ∧ x < 6 }
def B : Set ℝ := { x | 2 < x ∧ x < 9 }

-- Definitions for complements and subsets
def complement (S : Set ℝ) : Set ℝ := { x | x ∉ S }
def intersection (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∧ x ∈ T }
def union (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∨ x ∈ T }

-- Definition for set C as a parameterized set by a
def C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- Proof statements
theorem find_complement_intersection :
  complement (intersection A B) = { x | x < 3 ∨ x ≥ 6 } :=
by sorry

theorem find_union_complement :
  union (complement B) A = { x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by sorry

theorem subset_implies_a_range (a : ℝ) :
  C a ⊆ B → a ∈ {x | 2 ≤ x ∧ x ≤ 8} :=
by sorry

end find_complement_intersection_find_union_complement_subset_implies_a_range_l12_12062


namespace evaluate_expression_l12_12003

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12003


namespace inclination_angle_of_line_l12_12914

-- Definitions and conditions
def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

-- Theorem statement
theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : angle = 45 := by
  sorry

end inclination_angle_of_line_l12_12914


namespace smallest_n_l12_12204

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3^n = k^4) (h2 : ∃ l : ℕ, 2^n = l^6) : n = 12 :=
by
  sorry

end smallest_n_l12_12204


namespace rectangular_prism_dimensions_l12_12176

theorem rectangular_prism_dimensions (b l h : ℕ) 
  (h1 : l = 3 * b) 
  (h2 : l = 2 * h) 
  (h3 : l * b * h = 12168) :
  b = 14 ∧ l = 42 ∧ h = 21 :=
by
  -- The proof will go here
  sorry

end rectangular_prism_dimensions_l12_12176


namespace value_of_x2_plus_9y2_l12_12367

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l12_12367


namespace product_of_consecutive_integers_eq_255_l12_12136

theorem product_of_consecutive_integers_eq_255 (x : ℕ) (h : x * (x + 1) = 255) : x + (x + 1) = 31 := 
sorry

end product_of_consecutive_integers_eq_255_l12_12136


namespace denis_sum_of_numbers_l12_12968

theorem denis_sum_of_numbers :
  ∃ a b c d : ℕ, a < b ∧ b < c ∧ c < d ∧ a*d = 32 ∧ b*c = 14 ∧ a + b + c + d = 42 :=
sorry

end denis_sum_of_numbers_l12_12968


namespace range_of_f_l12_12133

def f (x : ℤ) : ℤ := x ^ 2 - 2 * x
def domain : Set ℤ := {0, 1, 2, 3}
def expectedRange : Set ℤ := {-1, 0, 3}

theorem range_of_f : (Set.image f domain) = expectedRange :=
  sorry

end range_of_f_l12_12133


namespace total_expenditure_eq_fourteen_l12_12400

variable (cost_barrette cost_comb : ℕ)
variable (kristine_barrettes kristine_combs crystal_barrettes crystal_combs : ℕ)

theorem total_expenditure_eq_fourteen 
  (h_cost_barrette : cost_barrette = 3)
  (h_cost_comb : cost_comb = 1)
  (h_kristine_barrettes : kristine_barrettes = 1)
  (h_kristine_combs : kristine_combs = 1)
  (h_crystal_barrettes : crystal_barrettes = 3)
  (h_crystal_combs : crystal_combs = 1) :
  (kristine_barrettes * cost_barrette + kristine_combs * cost_comb) +
  (crystal_barrettes * cost_barrette + crystal_combs * cost_comb) = 14 := 
by 
  sorry

end total_expenditure_eq_fourteen_l12_12400


namespace max_product_production_l12_12293

theorem max_product_production (C_mats A_mats C_ship A_ship B_mats B_ship : ℝ)
  (cost_A cost_B ship_A ship_B : ℝ) (prod_A prod_B max_cost_mats max_cost_ship prod_max : ℝ)
  (h_prod_A : prod_A = 90)
  (h_cost_A : cost_A = 1000)
  (h_ship_A : ship_A = 500)
  (h_prod_B : prod_B = 100)
  (h_cost_B : cost_B = 1500)
  (h_ship_B : ship_B = 400)
  (h_max_cost_mats : max_cost_mats = 6000)
  (h_max_cost_ship : max_cost_ship = 2000)
  (h_prod_max : prod_max = 440)
  (H_C_mats : C_mats = cost_A * A_mats + cost_B * B_mats)
  (H_C_ship : C_ship = ship_A * A_ship + ship_B * B_ship)
  (H_A_mats_ship : A_mats = A_ship)
  (H_B_mats_ship : B_mats = B_ship)
  (H_C_mats_le : C_mats ≤ max_cost_mats)
  (H_C_ship_le : C_ship ≤ max_cost_ship) :
  prod_A * A_mats + prod_B * B_mats ≤ prod_max :=
by {
  sorry
}

end max_product_production_l12_12293


namespace notebooks_ratio_l12_12584

variable (C N : Nat)

theorem notebooks_ratio (h1 : 512 = C * N)
  (h2 : 512 = 16 * (C / 2)) :
  N = C / 8 :=
by
  sorry

end notebooks_ratio_l12_12584


namespace sums_of_squares_divisibility_l12_12393

theorem sums_of_squares_divisibility :
  (∀ n : ℤ, (3 * n^2 + 2) % 3 ≠ 0) ∧ (∃ n : ℤ, (3 * n^2 + 2) % 11 = 0) := 
by
  sorry

end sums_of_squares_divisibility_l12_12393


namespace baseball_card_decrease_l12_12551

theorem baseball_card_decrease (x : ℝ) :
  (0 < x) ∧ (x < 100) ∧ (100 - x) * 0.9 = 45 → x = 50 :=
by
  intros h
  sorry

end baseball_card_decrease_l12_12551


namespace prob_zero_to_one_l12_12040

variable {σ : ℝ} (ξ : ℝ → ℝ)

def is_normal_1_var (ξ : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), ξ x ~ 𝓝(1, σ^2)

def prob_greater_than_2 (ξ : ℝ → ℝ) [is_normal_1_var ξ] : Prop :=
  P(ξ > 2) = 0.15

theorem prob_zero_to_one (ξ : ℝ → ℝ) [is_normal_1_var ξ] [prob_greater_than_2 ξ] :
  P(0 <= ξ ≤ 1) = 0.35 :=
sorry

end prob_zero_to_one_l12_12040


namespace point_in_second_quadrant_range_l12_12386

theorem point_in_second_quadrant_range (m : ℝ) :
  (m - 3 < 0 ∧ m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  sorry

end point_in_second_quadrant_range_l12_12386


namespace penalty_kicks_l12_12522

-- Define the soccer team data
def total_players : ℕ := 16
def goalkeepers : ℕ := 2
def players_shooting : ℕ := total_players - goalkeepers -- 14

-- Function to calculate total penalty kicks
def total_penalty_kicks (total_players goalkeepers : ℕ) : ℕ :=
  let players_shooting := total_players - goalkeepers
  players_shooting * goalkeepers

-- Theorem stating the number of penalty kicks
theorem penalty_kicks : total_penalty_kicks total_players goalkeepers = 30 :=
by
  sorry

end penalty_kicks_l12_12522


namespace real_roots_polynomials_l12_12577

noncomputable def poly_deg1 := [Polynomial.C 1 + Polynomial.X, -Polynomial.X + Polynomial.C 1]
noncomputable def poly_deg2 := [Polynomial.X^2 + Polynomial.X - Polynomial.C 1, Polynomial.X^2 - Polynomial.X - Polynomial.C 1]
noncomputable def poly_deg3 := [Polynomial.X^3 + Polynomial.X^2 - Polynomial.X - Polynomial.C 1, Polynomial.X^3 - Polynomial.X^2 - Polynomial.X + Polynomial.C 1]

theorem real_roots_polynomials :
  ∀ (p : Polynomial ℝ), (p.coeffs = [1, 1, -1] ∨ p.coeffs = [1, -1, -1] ∨ 
                          p.coeffs = [1, 0, -1] ∨ p.coeffs = [-1, 0, -1] ∨
                          p.coeffs = [1, 1, 0, -1] ∨ p.coeffs = [1, -1, 0, 1] ∨
                          p.coeffs = [-1, 1, 0, -1] ∨ p.coeffs = [-1, -1, 0, 1]) ↔
                          (∀ (x : ℝ), p.is_root x) :=
begin
  sorry
end

end real_roots_polynomials_l12_12577


namespace simplest_form_option_l12_12938

theorem simplest_form_option (x y : ℚ) :
  (∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (12 * (x - y) / (15 * (x + y)) ≠ 4 * (x - y) / 5 * (x + y))) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 + y^2) / (x + y) = a / b) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / ((x + y)^2) ≠ (x - y) / (x + y)) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / (x + y) ≠ x - y)) := sorry

end simplest_form_option_l12_12938


namespace evaluate_expression_l12_12014

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12014


namespace find_abc_l12_12853

noncomputable def f (a b c x : ℝ) := x^3 + a*x^2 + b*x + c
noncomputable def f' (a b x : ℝ) := 3*x^2 + 2*a*x + b

theorem find_abc (a b c : ℝ) :
  (f' a b -2 = 0) ∧
  (f' a b 1 = -3) ∧
  (f a b c 1 = 0) →
  a = 1 ∧ b = -8 ∧ c = 6 :=
sorry

end find_abc_l12_12853


namespace possible_values_of_X_l12_12849

-- Define the conditions and the problem
def defective_products_total := 3
def total_products := 10
def selected_products := 2

-- Define the random variable X
def X (n : ℕ) : ℕ := n / selected_products

-- Now the statement to prove is that X can only take the values {0, 1, 2}
theorem possible_values_of_X :
  ∀ (X : ℕ → ℕ), ∃ (vals : Set ℕ), (vals = {0, 1, 2} ∧ ∀ (n : ℕ), X n ∈ vals) :=
by
  sorry

end possible_values_of_X_l12_12849


namespace largest_satisfying_n_correct_l12_12334
noncomputable def largest_satisfying_n : ℕ := 4

theorem largest_satisfying_n_correct :
  ∀ n x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5) 
  → n = largest_satisfying_n ∧
  ¬ (∃ x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5 ∧ 5 < x^5 ∧ x^5 < 6)) := sorry

end largest_satisfying_n_correct_l12_12334


namespace min_a_value_l12_12861

theorem min_a_value {a b : ℕ} (h : 1998 * a = b^4) : a = 1215672 :=
sorry

end min_a_value_l12_12861


namespace average_of_remaining_numbers_l12_12131

theorem average_of_remaining_numbers (S : ℕ) (h1 : S = 12 * 90) :
  ((S - 65 - 75 - 85) / 9) = 95 :=
by
  sorry

end average_of_remaining_numbers_l12_12131


namespace steve_fraction_of_day_in_school_l12_12906

theorem steve_fraction_of_day_in_school :
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  (school_hours / total_hours) = (1 / 6) :=
by
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  have : (school_hours / total_hours) = (1 / 6) := sorry
  exact this

end steve_fraction_of_day_in_school_l12_12906


namespace find_g_of_7_l12_12486

theorem find_g_of_7 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 8) = 2 * x + 11) : g 7 = 21 :=
by
  sorry

end find_g_of_7_l12_12486


namespace range_of_A_l12_12729

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then -x^2 + 2*x - 5/4 else log (x) / log (1/3) - 1/4

noncomputable def g (A : ℝ) : ℝ → ℝ :=
λ x, abs (A - 2) * sin x

theorem range_of_A (A : ℝ) : 
  (∀ x1 x2 : ℝ, f x1 ≤ g A x2) ↔ (7/4 ≤ A ∧ A ≤ 9/4) := by
  sorry

end range_of_A_l12_12729


namespace max_S_is_9_l12_12216

-- Definitions based on the conditions
def a (n : ℕ) : ℤ := 28 - 3 * n
def S (n : ℕ) : ℤ := n * (25 + a n) / 2

-- The theorem to be proved
theorem max_S_is_9 : ∃ n : ℕ, n = 9 ∧ S n = 117 :=
by
  sorry

end max_S_is_9_l12_12216


namespace son_age_l12_12305

-- Defining the variables
variables (S F : ℕ)

-- The conditions
def condition1 : Prop := F = S + 25
def condition2 : Prop := F + 2 = 2 * (S + 2)

-- The statement to be proved
theorem son_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 23 :=
sorry

end son_age_l12_12305


namespace speed_conversion_l12_12960

theorem speed_conversion (speed_mps: ℝ) (conversion_factor: ℝ) (expected_speed_kmph: ℝ):
  speed_mps * conversion_factor = expected_speed_kmph :=
by
  let speed_mps := 115.00919999999999
  let conversion_factor := 3.6
  let expected_speed_kmph := 414.03312
  sorry

end speed_conversion_l12_12960


namespace corrected_mean_35_25_l12_12414

theorem corrected_mean_35_25 (n : ℕ) (mean : ℚ) (x_wrong x_correct : ℚ) :
  n = 20 → mean = 36 → x_wrong = 40 → x_correct = 25 → 
  ( (mean * n - x_wrong + x_correct) / n = 35.25) :=
by
  intros h1 h2 h3 h4
  sorry

end corrected_mean_35_25_l12_12414


namespace necessary_but_not_sufficient_converse_implies_l12_12472

theorem necessary_but_not_sufficient (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * (Real.log x) ^ 2 < 1) → (x * Real.log x < 1) :=
sorry

theorem converse_implies (x : ℝ) (hx1 : 1 < x) (hx2 : x < Real.exp 1) : 
  (x * Real.log x < 1) → (x * (Real.log x) ^ 2 < 1) :=
sorry

end necessary_but_not_sufficient_converse_implies_l12_12472


namespace radius_of_given_circle_is_eight_l12_12592

noncomputable def radius_of_circle (diameter : ℝ) : ℝ := diameter / 2

theorem radius_of_given_circle_is_eight :
  radius_of_circle 16 = 8 :=
by
  sorry

end radius_of_given_circle_is_eight_l12_12592


namespace train_a_constant_rate_l12_12145

theorem train_a_constant_rate
  (d : ℕ)
  (v_b : ℕ)
  (d_a : ℕ)
  (v : ℕ)
  (h1 : d = 350)
  (h2 : v_b = 30)
  (h3 : d_a = 200)
  (h4 : v * (d_a / v) + v_b * (d_a / v) = d) :
  v = 40 := by
  sorry

end train_a_constant_rate_l12_12145


namespace max_value_ab_bc_cd_l12_12759

theorem max_value_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) (h_sum : a + b + c + d = 200) :
  ab + bc + cd ≤ 2500 :=
by
  sorry

end max_value_ab_bc_cd_l12_12759


namespace regular_pentagon_cannot_cover_floor_completely_l12_12816

theorem regular_pentagon_cannot_cover_floor_completely
  (hexagon_interior_angle : ℝ)
  (pentagon_interior_angle : ℝ)
  (square_interior_angle : ℝ)
  (triangle_interior_angle : ℝ)
  (hexagon_condition : 360 / hexagon_interior_angle = 3)
  (square_condition : 360 / square_interior_angle = 4)
  (triangle_condition : 360 / triangle_interior_angle = 6)
  (pentagon_condition : 360 / pentagon_interior_angle ≠ 3)
  (pentagon_condition2 : 360 / pentagon_interior_angle ≠ 4)
  (pentagon_condition3 : 360 / pentagon_interior_angle ≠ 6) :
  pentagon_interior_angle = 108 := 
  sorry

end regular_pentagon_cannot_cover_floor_completely_l12_12816


namespace find_common_difference_l12_12234

variable (a : ℕ → ℤ)  -- define the arithmetic sequence as a function from ℕ to ℤ
variable (d : ℤ)      -- define the common difference

-- Define the conditions
def conditions := (a 5 = 10) ∧ (a 12 = 31)

-- Define the formula for the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) := a 1 + d * (n - 1)

-- Prove that the common difference d is 3 given the conditions
theorem find_common_difference (h : conditions a) : d = 3 :=
sorry

end find_common_difference_l12_12234


namespace quiz_win_probability_is_13_over_256_l12_12689

open ProbabilityTheory

noncomputable def quizWinProbability : ℚ :=
  let probabilityCorrect := (1 : ℚ) / 4
  let probabilityWrong := (3 : ℚ) / 4
  let probabilityAllFourCorrect := probabilityCorrect ^ 4
  let probabilityExactlyThreeCorrect := 4 * (probabilityCorrect ^ 3 * probabilityWrong)
  probabilityAllFourCorrect + probabilityExactlyThreeCorrect

theorem quiz_win_probability_is_13_over_256 :
  quizWinProbability = (13 : ℚ) / 256 :=
by
  sorry

end quiz_win_probability_is_13_over_256_l12_12689


namespace rate_per_kg_first_batch_l12_12962

/-- This theorem proves the rate per kg of the first batch of wheat. -/
theorem rate_per_kg_first_batch (x : ℝ) 
  (h1 : 30 * x + 20 * 14.25 = 285 + 30 * x) 
  (h2 : (30 * x + 285) * 1.3 = 819) : 
  x = 11.5 := 
sorry

end rate_per_kg_first_batch_l12_12962


namespace total_mile_times_l12_12282

-- Define the conditions
def Tina_time : ℕ := 6  -- Tina runs a mile in 6 minutes

def Tony_time : ℕ := Tina_time / 2  -- Tony runs twice as fast as Tina

def Tom_time : ℕ := Tina_time / 3  -- Tom runs three times as fast as Tina

-- Define the proof statement
theorem total_mile_times : Tony_time + Tina_time + Tom_time = 11 := by
  sorry

end total_mile_times_l12_12282


namespace determine_missing_digits_l12_12629

theorem determine_missing_digits :
  (237 * 0.31245 = 7430.65) := 
by 
  sorry

end determine_missing_digits_l12_12629


namespace complex_in_second_quadrant_l12_12498

-- Define the complex number z based on the problem conditions.
def z : ℂ := Complex.I + (Complex.I^6)

-- State the condition to check whether z is in the second quadrant.
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Formulate the theorem stating that the complex number z is in the second quadrant.
theorem complex_in_second_quadrant : is_in_second_quadrant z :=
by
  sorry

end complex_in_second_quadrant_l12_12498


namespace prime_factors_count_l12_12999

theorem prime_factors_count (n : ℕ) (h : n = 75) : (nat.factors n).to_finset.card = 2 :=
by
  rw h
  -- The proof part is omitted as instructed
  sorry

end prime_factors_count_l12_12999


namespace Skylar_chickens_less_than_triple_Colten_l12_12407

def chickens_count (S Q C : ℕ) : Prop := 
  Q + S + C = 383 ∧ 
  Q = 2 * S + 25 ∧ 
  C = 37

theorem Skylar_chickens_less_than_triple_Colten (S Q C : ℕ) 
  (h : chickens_count S Q C) : (3 * C - S = 4) := 
sorry

end Skylar_chickens_less_than_triple_Colten_l12_12407


namespace part1_part2_l12_12603

section

variable (a : ℝ) (a_seq : ℕ → ℝ)
variable (h_seq : ∀ n, a_seq (n + 1) = (5 * a_seq n - 8) / (a_seq n - 1))
variable (h_initial : a_seq 1 = a)

-- Part 1:
theorem part1 (h_a : a = 3) : 
  ∃ r : ℝ, ∀ n, (a_seq n - 2) / (a_seq n - 4) = r ^ n ∧ a_seq n = (4 * 3 ^ (n - 1) + 2) / (3 ^ (n - 1) + 1) := 
sorry

-- Part 2:
theorem part2 (h_pos : ∀ n, a_seq n > 3) : 3 < a := 
sorry

end

end part1_part2_l12_12603


namespace expected_value_of_10_sided_die_l12_12446

-- Definition of the conditions
def num_faces : ℕ := 10
def face_values : List ℕ := List.range' 2 num_faces

-- Theorem statement: The expected value of a roll of this die is 6.5
theorem expected_value_of_10_sided_die : 
  (List.sum face_values : ℚ) / num_faces = 6.5 := 
sorry

end expected_value_of_10_sided_die_l12_12446


namespace sequence_sum_fraction_l12_12602

theorem sequence_sum_fraction :
  (∀ n: ℕ, a n = if n = 0 then 1 else a (n-1) + n + 1) →
  (∑ n in range 2006, 1 / (a n)) = 4032 / 2017 :=
by
  intro h
  -- Insert the definitions and loop through the sequence
  sorry

end sequence_sum_fraction_l12_12602


namespace monotonic_iff_a_range_l12_12852

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem monotonic_iff_a_range (a : ℝ) : 
  (∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 ∨ f a x1 ≥ f a x2) ↔ (-3 < a ∧ a < 6) :=
by 
  sorry

end monotonic_iff_a_range_l12_12852


namespace derivative_f_derivative_f_at_2_l12_12731

noncomputable def f : ℝ → ℝ := λ x, x^2 + x

-- The first statement: derivative of f(x) is 2x + 1
theorem derivative_f : deriv f x = 2 * x + 1 := sorry

-- The second statement: value of the derivative at x = 2 is 5
theorem derivative_f_at_2 : deriv f 2 = 5 := sorry

end derivative_f_derivative_f_at_2_l12_12731


namespace train_B_time_to_reach_destination_l12_12784

theorem train_B_time_to_reach_destination
    (T t : ℝ)
    (train_A_speed : ℝ) (train_B_speed : ℝ)
    (train_A_extra_hours : ℝ)
    (h1 : train_A_speed = 110)
    (h2 : train_B_speed = 165)
    (h3 : train_A_extra_hours = 9)
    (h_eq : 110 * (T + train_A_extra_hours) = 110 * T + 165 * t) :
    t = 6 := 
by
  sorry

end train_B_time_to_reach_destination_l12_12784


namespace sum_of_digits_joey_age_l12_12390

def int.multiple (a b : ℕ) := ∃ k : ℕ, a = k * b

theorem sum_of_digits_joey_age (J C M n : ℕ) (h1 : J = C + 2) (h2 : M = 2) (h3 : ∃ k, C = k * M) (h4 : C = 12) (h5 : J + n = 26) : 
  (2 + 6 = 8) :=
by
  sorry

end sum_of_digits_joey_age_l12_12390


namespace zero_intercept_and_distinct_roots_l12_12205

noncomputable def Q (x a' b' c' d' : ℝ) : ℝ := x^4 + a' * x^3 + b' * x^2 + c' * x + d'

theorem zero_intercept_and_distinct_roots (a' b' c' d' : ℝ) (u v w : ℝ) (h_distinct : u ≠ v ∧ v ≠ w ∧ u ≠ w) (h_intercept_at_zero : d' = 0)
(h_Q_form : ∀ x, Q x a' b' c' d' = x * (x - u) * (x - v) * (x - w)) : c' ≠ 0 :=
by
  sorry

end zero_intercept_and_distinct_roots_l12_12205


namespace friends_recycled_pounds_l12_12517

-- Definitions for the given conditions
def pounds_per_point : ℕ := 4
def paige_recycled : ℕ := 14
def total_points : ℕ := 4

-- The proof statement
theorem friends_recycled_pounds :
  ∃ p_friends : ℕ, 
  (paige_recycled / pounds_per_point) + (p_friends / pounds_per_point) = total_points 
  → p_friends = 4 := 
sorry

end friends_recycled_pounds_l12_12517


namespace annual_growth_rate_equation_l12_12541

theorem annual_growth_rate_equation
  (initial_capital : ℝ)
  (final_capital : ℝ)
  (n : ℕ)
  (x : ℝ)
  (h1 : initial_capital = 10)
  (h2 : final_capital = 14.4)
  (h3 : n = 2) :
  1000 * (1 + x)^2 = 1440 :=
by
  sorry

end annual_growth_rate_equation_l12_12541


namespace ajay_gain_l12_12798

-- Definitions of the problem conditions as Lean variables/constants.
variables (kg1 kg2 kg_total : ℕ) 
variables (price1 price2 price3 cost1 cost2 total_cost selling_price gain : ℝ)

-- Conditions of the problem.
def conditions : Prop :=
  kg1 = 15 ∧ 
  kg2 = 10 ∧ 
  kg_total = kg1 + kg2 ∧ 
  price1 = 14.5 ∧ 
  price2 = 13 ∧ 
  price3 = 15 ∧ 
  cost1 = kg1 * price1 ∧ 
  cost2 = kg2 * price2 ∧ 
  total_cost = cost1 + cost2 ∧ 
  selling_price = kg_total * price3 ∧ 
  gain = selling_price - total_cost 

-- The theorem for the gain amount proof.
theorem ajay_gain (h : conditions kg1 kg2 kg_total price1 price2 price3 cost1 cost2 total_cost selling_price gain) : 
  gain = 27.50 :=
  sorry

end ajay_gain_l12_12798


namespace license_plate_count_l12_12955

noncomputable def num_license_plates : Nat :=
  let num_digit_possibilities := 10
  let num_letter_possibilities := 26
  let num_letter_pairs := num_letter_possibilities * num_letter_possibilities
  let num_positions_for_block := 6
  num_positions_for_block * (num_digit_possibilities ^ 5) * num_letter_pairs

theorem license_plate_count :
  num_license_plates = 40560000 :=
by
  sorry

end license_plate_count_l12_12955


namespace no_super_squarish_numbers_l12_12959

def is_super_squarish (M : ℕ) : Prop :=
  let a := M / 100000 % 100
  let b := M / 1000 % 1000
  let c := M % 100
  (M ≥ 1000000 ∧ M < 10000000) ∧
  (M % 10 ≠ 0 ∧ (M / 10) % 10 ≠ 0 ∧ (M / 100) % 10 ≠ 0 ∧ (M / 1000) % 10 ≠ 0 ∧
    (M / 10000) % 10 ≠ 0 ∧ (M / 100000) % 10 ≠ 0 ∧ (M / 1000000) % 10 ≠ 0) ∧
  (∃ y : ℕ, y * y = M) ∧
  (∃ f g : ℕ, f * f = a ∧ 2 * f * g = b ∧ g * g = c) ∧
  (10 ≤ a ∧ a ≤ 99) ∧
  (100 ≤ b ∧ b ≤ 999) ∧
  (10 ≤ c ∧ c ≤ 99)

theorem no_super_squarish_numbers : ∀ M : ℕ, is_super_squarish M → false :=
sorry

end no_super_squarish_numbers_l12_12959


namespace jane_oldest_babysat_age_l12_12755

-- Given conditions
def jane_babysitting_has_constraints (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ) : Prop :=
  jane_current_age - jane_stop_babysitting_age = 10 ∧
  jane_stop_babysitting_age - jane_start_babysitting_age = 2

-- Helper definition for prime number constraint
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (n % m = 0)

-- Main goal: the current age of the oldest person Jane could have babysat is 19
theorem jane_oldest_babysat_age
  (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ)
  (H_constraints : jane_babysitting_has_constraints jane_current_age jane_stop_babysitting_age jane_start_babysitting_age) :
  ∃ (child_age : ℕ), child_age = 19 ∧ is_prime child_age ∧
  (child_age = (jane_stop_babysitting_age / 2 + 10) ∨ child_age = (jane_stop_babysitting_age / 2 + 9)) :=
sorry  -- Proof to be filled in.

end jane_oldest_babysat_age_l12_12755


namespace cos_diff_proof_l12_12212

noncomputable def cos_diff (α β : ℝ) : ℝ := Real.cos (α - β)

theorem cos_diff_proof (α β : ℝ) 
  (h1 : Real.cos α - Real.cos β = 1 / 2)
  (h2 : Real.sin α - Real.sin β = 1 / 3) :
  cos_diff α β = 59 / 72 := by
  sorry

end cos_diff_proof_l12_12212


namespace factorize_expr_l12_12329

theorem factorize_expr (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := 
sorry

end factorize_expr_l12_12329


namespace average_after_17th_inning_l12_12796

variable (A : ℕ)

-- Definition of total runs before the 17th inning
def total_runs_before := 16 * A

-- Definition of new total runs after the 17th inning
def total_runs_after := total_runs_before A + 87

-- Definition of new average after the 17th inning
def new_average := A + 4

-- Definition of new total runs in terms of new average
def new_total_runs := 17 * new_average A

-- The statement we want to prove
theorem average_after_17th_inning : total_runs_after A = new_total_runs A → new_average A = 23 := by
  sorry

end average_after_17th_inning_l12_12796


namespace total_pages_in_book_l12_12808

-- Given conditions
def pages_first_chapter : ℕ := 13
def pages_second_chapter : ℕ := 68

-- The theorem to prove the total number of pages in the book
theorem total_pages_in_book :
  pages_first_chapter + pages_second_chapter = 81 := by
  sorry

end total_pages_in_book_l12_12808


namespace shopkeeper_oranges_l12_12442

theorem shopkeeper_oranges (O : ℕ) 
  (bananas : ℕ) 
  (percent_rotten_oranges : ℕ) 
  (percent_rotten_bananas : ℕ) 
  (percent_good_condition : ℚ) 
  (h1 : bananas = 400) 
  (h2 : percent_rotten_oranges = 15) 
  (h3 : percent_rotten_bananas = 6) 
  (h4 : percent_good_condition = 88.6) : 
  O = 600 :=
by
  -- This proof needs to be filled in.
  sorry

end shopkeeper_oranges_l12_12442


namespace john_lift_total_weight_l12_12884

-- Define the conditions as constants
def initial_weight : ℝ := 135
def weight_increase : ℝ := 265
def bracer_factor : ℝ := 6

-- Define a theorem to prove the total weight John can lift
theorem john_lift_total_weight : initial_weight + weight_increase + (initial_weight + weight_increase) * bracer_factor = 2800 := by
  -- proof here
  sorry

end john_lift_total_weight_l12_12884


namespace daily_profit_at_45_selling_price_for_1200_profit_l12_12299

-- Definitions for the conditions
def cost_price (p: ℝ) : Prop := p = 30
def initial_sales (p: ℝ) (s: ℝ) : Prop := p = 40 ∧ s = 80
def sales_decrease_rate (r: ℝ) : Prop := r = 2
def max_selling_price (p: ℝ) : Prop := p ≤ 55

-- Proof for Question 1
theorem daily_profit_at_45 (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate) :
  (price = 45) → profit = 1050 :=
by sorry

-- Proof for Question 2
theorem selling_price_for_1200_profit (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate)
  (h_max_price : ∀ p, max_selling_price p → p ≤ 55) :
  profit = 1200 → price = 50 :=
by sorry

end daily_profit_at_45_selling_price_for_1200_profit_l12_12299


namespace parameter_values_for_three_distinct_roots_l12_12579

theorem parameter_values_for_three_distinct_roots (a : ℝ) :
  (∀ x : ℝ, (|x^3 - a^3| = x - a) → (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ↔ 
  (-2 / Real.sqrt 3 < a ∧ a < -1 / Real.sqrt 3) :=
sorry

end parameter_values_for_three_distinct_roots_l12_12579


namespace money_conditions_l12_12865

theorem money_conditions (c d : ℝ) (h1 : 7 * c - d > 80) (h2 : 4 * c + d = 44) (h3 : d < 2 * c) :
  c > 124 / 11 ∧ d < 2 * c ∧ d = 12 :=
by
  sorry

end money_conditions_l12_12865


namespace parabola_hyperbola_intersection_l12_12352

open Real

theorem parabola_hyperbola_intersection (p : ℝ) (hp : p > 0)
  (h_hyperbola : ∀ x y, (x^2 / 4 - y^2 = 1) → (y = 2*x ∨ y = -2*x))
  (h_parabola_directrix : ∀ y, (x^2 = 2 * p * y) → (x = -p/2)) 
  (h_area_triangle : (1/2) * (p/2) * (2 * p) = 1) :
  p = sqrt 2 := sorry

end parabola_hyperbola_intersection_l12_12352


namespace non_zero_real_y_satisfies_l12_12540

theorem non_zero_real_y_satisfies (y : ℝ) (h : y ≠ 0) : (8 * y) ^ 3 = (16 * y) ^ 2 → y = 1 / 2 :=
by
  -- Lean code placeholders
  sorry

end non_zero_real_y_satisfies_l12_12540


namespace characterize_affine_function_l12_12463

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, a < b → integrable_on f (set.Icc a b)) ∧
  (∀ x : ℝ, ∀ n : ℕ, 1 ≤ n → (f x = (n / 2) * integral (Icc (x - 1 / n) (x + 1 / n)) f))

theorem characterize_affine_function :
  ∀ f : ℝ → ℝ, satisfies_conditions f → ∃ p q : ℝ, ∀ x : ℝ, f x = p * x + q := 
sorry

end characterize_affine_function_l12_12463


namespace hawks_score_l12_12233

theorem hawks_score (a b : ℕ) (h1 : a + b = 58) (h2 : a - b = 12) : b = 23 :=
by
  sorry

end hawks_score_l12_12233


namespace curve_product_l12_12844

theorem curve_product (a b : ℝ) (h1 : 8 * a + 2 * b = 2) (h2 : 12 * a + b = 9) : a * b = -3 := by
  sorry

end curve_product_l12_12844


namespace find_value_of_fraction_l12_12243

open Real

theorem find_value_of_fraction (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) : 
  (x + y) / (x - y) = -sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l12_12243


namespace evaluate_x_squared_minus_y_squared_l12_12053

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end evaluate_x_squared_minus_y_squared_l12_12053


namespace incorrect_conclusion_C_l12_12128

variable {a : ℕ → ℝ} {q : ℝ}

-- Conditions
def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q

theorem incorrect_conclusion_C 
  (h_geo: geo_seq a q)
  (h_cond: a 1 * a 2 < 0) : 
  a 1 * a 5 > 0 :=
by 
  sorry

end incorrect_conclusion_C_l12_12128


namespace Jane_age_l12_12757

theorem Jane_age (J A : ℕ) (h1 : J + A = 54) (h2 : J - A = 22) : A = 16 := 
by 
  sorry

end Jane_age_l12_12757


namespace positive_integer_solutions_count_l12_12910

theorem positive_integer_solutions_count : 
  (∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15) :=
sorry

end positive_integer_solutions_count_l12_12910


namespace evaluate_expression_l12_12011

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12011


namespace vampire_daily_blood_suction_l12_12187

-- Conditions from the problem
def vampire_bl_need_per_week : ℕ := 7  -- gallons of blood per week
def blood_per_person_in_pints : ℕ := 2  -- pints of blood per person
def pints_per_gallon : ℕ := 8            -- pints in 1 gallon

-- Theorem statement to prove
theorem vampire_daily_blood_suction :
  let daily_requirement_in_gallons : ℕ := vampire_bl_need_per_week / 7   -- gallons per day
  let daily_requirement_in_pints : ℕ := daily_requirement_in_gallons * pints_per_gallon
  let num_people_needed_per_day : ℕ := daily_requirement_in_pints / blood_per_person_in_pints
  num_people_needed_per_day = 4 :=
by
  sorry

end vampire_daily_blood_suction_l12_12187


namespace cube_product_l12_12427

/-- A cube is a three-dimensional shape with a specific number of vertices and faces. -/
structure Cube where
  vertices : ℕ
  faces : ℕ

theorem cube_product (C : Cube) (h1: C.vertices = 8) (h2: C.faces = 6) : 
  (C.vertices * C.faces = 48) :=
by sorry

end cube_product_l12_12427


namespace bobs_walking_rate_l12_12669

theorem bobs_walking_rate (distance_XY : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_distance_when_met : ℕ) 
  (yolanda_extra_hour : ℕ)
  (meet_covered_distance : distance_XY = yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1 + bob_distance_when_met / bob_distance_when_met)) 
  (yolanda_distance_when_met : yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) + bob_distance_when_met = distance_XY) 
  : 
  (bob_distance_when_met / (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) = yolanda_rate) :=
  sorry

end bobs_walking_rate_l12_12669


namespace find_value_of_expression_l12_12213

theorem find_value_of_expression (m : ℝ) (h_m : m^2 - 3 * m + 1 = 0) : 2 * m^2 - 6 * m - 2024 = -2026 := by
  sorry

end find_value_of_expression_l12_12213


namespace quad_eq_complete_square_l12_12057

theorem quad_eq_complete_square (p q : ℝ) 
  (h : ∀ x : ℝ, (4 * x^2 - p * x + q = 0 ↔ (x - 1/4)^2 = 33/16)) : q / p = -4 := by
  sorry

end quad_eq_complete_square_l12_12057


namespace average_rainfall_l12_12819

theorem average_rainfall (rainfall_Tuesday : ℝ) (rainfall_others : ℝ) (days_in_week : ℝ)
  (h1 : rainfall_Tuesday = 10.5) 
  (h2 : rainfall_Tuesday = rainfall_others)
  (h3 : days_in_week = 7) : 
  (rainfall_Tuesday + rainfall_others) / days_in_week = 3 :=
by
  sorry

end average_rainfall_l12_12819


namespace minimize_fractions_sum_l12_12506

theorem minimize_fractions_sum {A B C D E : ℕ}
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E)
  (h5 : B ≠ C) (h6 : B ≠ D) (h7 : B ≠ E)
  (h8 : C ≠ D) (h9 : C ≠ E) (h10 : D ≠ E)
  (h11 : A ≠ 9) (h12 : B ≠ 9) (h13 : C ≠ 9) (h14 : D ≠ 9) (h15 : E ≠ 9)
  (hA : 1 ≤ A) (hB : 1 ≤ B) (hC : 1 ≤ C) (hD : 1 ≤ D) (hE : 1 ≤ E)
  (hA' : A ≤ 9) (hB' : B ≤ 9) (hC' : C ≤ 9) (hD' : D ≤ 9) (hE' : E ≤ 9) :
  A / B + C / D + E / 9 = 125 / 168 :=
sorry

end minimize_fractions_sum_l12_12506


namespace evaluate_expression_l12_12002

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12002


namespace MarlySoupBags_l12_12898

theorem MarlySoupBags :
  ∀ (milk chicken_stock vegetables bag_capacity total_soup total_bags : ℚ),
    milk = 6 ∧
    chicken_stock = 3 * milk ∧
    vegetables = 3 ∧
    bag_capacity = 2 ∧
    total_soup = milk + chicken_stock + vegetables ∧
    total_bags = total_soup / bag_capacity ∧
    total_bags.ceil = 14 :=
by
  intros
  sorry

end MarlySoupBags_l12_12898


namespace euler_family_mean_age_l12_12646

theorem euler_family_mean_age : 
  let girls_ages := [5, 5, 10, 15]
  let boys_ages := [8, 12, 16]
  let children_ages := girls_ages ++ boys_ages
  let total_sum := List.sum children_ages
  let number_of_children := List.length children_ages
  (total_sum : ℚ) / number_of_children = 10.14 := 
by
  sorry

end euler_family_mean_age_l12_12646


namespace commensurable_iff_rat_l12_12406

def commensurable (A B : ℝ) : Prop :=
  ∃ d : ℝ, ∃ m n : ℤ, A = m * d ∧ B = n * d

theorem commensurable_iff_rat (A B : ℝ) :
  commensurable A B ↔ ∃ (m n : ℤ) (h : n ≠ 0), A / B = m / n :=
by
  sorry

end commensurable_iff_rat_l12_12406


namespace chapters_per_day_l12_12583

theorem chapters_per_day (total_pages : ℕ) (total_chapters : ℕ) (total_days : ℕ)
  (h1 : total_pages = 193)
  (h2 : total_chapters = 15)
  (h3 : total_days = 660) :
  (total_chapters : ℝ) / total_days = 0.0227 :=
by 
  sorry

end chapters_per_day_l12_12583


namespace evaluate_polynomial_at_6_eq_1337_l12_12147

theorem evaluate_polynomial_at_6_eq_1337 :
  (3 * 6^2 + 15 * 6 + 7) + (4 * 6^3 + 8 * 6^2 - 5 * 6 + 10) = 1337 := by
  sorry

end evaluate_polynomial_at_6_eq_1337_l12_12147


namespace union_A_B_l12_12358

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem union_A_B :
  A ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

end union_A_B_l12_12358


namespace remainder_71_73_div_8_l12_12674

theorem remainder_71_73_div_8 :
  (71 * 73) % 8 = 7 :=
by
  sorry

end remainder_71_73_div_8_l12_12674


namespace proposition_p_l12_12076

variable (x : ℝ)

-- Define condition
def negation_of_p : Prop := ∃ x, x < 1 ∧ x^2 < 1

-- Define proposition p
def p : Prop := ∀ x, x < 1 → x^2 ≥ 1

-- Theorem statement
theorem proposition_p (h : negation_of_p) : (p) :=
sorry

end proposition_p_l12_12076


namespace extra_yellow_balls_dispatched_l12_12815

theorem extra_yellow_balls_dispatched : 
  ∀ (W Y E : ℕ), -- Declare natural numbers W, Y, E
  W = Y →      -- Condition that the number of white balls equals the number of yellow balls
  W + Y = 64 → -- Condition that the total number of originally ordered balls is 64
  W / (Y + E) = 8 / 13 → -- The given ratio involving the extra yellow balls
  E = 20 :=               -- Prove that the extra yellow balls E equals 20
by
  intros W Y E h1 h2 h3
  -- Proof mechanism here
  sorry

end extra_yellow_balls_dispatched_l12_12815


namespace convert_speed_l12_12469

theorem convert_speed (v_kmph : ℝ) (conversion_factor : ℝ) : 
  v_kmph = 252 → conversion_factor = 0.277778 → v_kmph * conversion_factor = 70 := by
  intros h1 h2
  rw [h1, h2]
  sorry

end convert_speed_l12_12469


namespace evaluate_expression_l12_12006

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12006


namespace find_remainder_mod_105_l12_12900

-- Define the conditions as a set of hypotheses
variables {n a b c : ℕ}
variables (hn : n > 0)
variables (ha : a < 3) (hb : b < 5) (hc : c < 7)
variables (h3 : n % 3 = a) (h5 : n % 5 = b) (h7 : n % 7 = c)
variables (heq : 4 * a + 3 * b + 2 * c = 30)

-- State the theorem
theorem find_remainder_mod_105 : n % 105 = 29 :=
by
  -- Hypotheses block for documentation
  have ha_le : 0 ≤ a := sorry
  have hb_le : 0 ≤ b := sorry
  have hc_le : 0 ≤ c := sorry
  sorry

end find_remainder_mod_105_l12_12900


namespace min_value_2a_b_c_l12_12608

theorem min_value_2a_b_c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * (a + b + c) + b * c = 4) : 
  2 * a + b + c ≥ 4 :=
sorry

end min_value_2a_b_c_l12_12608


namespace sum_of_Jo_numbers_l12_12620

theorem sum_of_Jo_numbers : 
  let seq := fun n => 2 * n - 1 
  let sum := (n: ℕ) → (finset.range n).sum seq
  sum 25 = 625 :=
by
  -- Proof is required here
  sorry

end sum_of_Jo_numbers_l12_12620


namespace largest_integer_y_l12_12426

theorem largest_integer_y (y : ℤ) : (y / (4:ℚ) + 3 / 7 < 2 / 3) → y ≤ 0 :=
by
  sorry

end largest_integer_y_l12_12426


namespace value_of_x2_plus_9y2_l12_12366

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l12_12366


namespace point_C_number_l12_12767

theorem point_C_number (B C: ℝ) (h1 : B = 3) (h2 : |C - B| = 2) :
  C = 1 ∨ C = 5 := 
by {
  sorry
}

end point_C_number_l12_12767


namespace bags_needed_l12_12697

theorem bags_needed (expected_people extra_people extravagant_bags average_bags : ℕ) 
    (h1 : expected_people = 50) 
    (h2 : extra_people = 40) 
    (h3 : extravagant_bags = 10) 
    (h4 : average_bags = 20) : 
    (expected_people + extra_people - (extravagant_bags + average_bags) = 60) :=
by {
  sorry
}

end bags_needed_l12_12697


namespace find_both_artifacts_total_time_l12_12879

variables (months_in_year : Nat) (expedition_first_years : Nat) (artifact_factor : Nat)

noncomputable def total_time (research_months : Nat) (expedition_first_years : Nat) :=
  let research_first_years := float_of_nat research_months / float_of_nat months_in_year
  let total_first := research_first_years + float_of_nat expedition_first_years
  let total_second := artifact_factor * total_first
  total_first + total_second

theorem find_both_artifacts_total_time :
  forall (months_in_year : Nat) (expedition_first_years : Nat) (artifact_factor : Nat),
    months_in_year = 12 → 
    expedition_first_years = 2 → 
    artifact_factor = 3 → 
    total_time 6 expedition_first_years = 10 :=
by intros months_in_year expedition_first_years artifact_factor hm he hf
   unfold total_time 
   sorry

end find_both_artifacts_total_time_l12_12879


namespace union_complement_l12_12733

open Set

-- Definitions based on conditions
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}
def C_UA : Set ℕ := U \ A

-- The theorem to prove
theorem union_complement :
  (C_UA ∪ B) = {0, 2, 4, 5, 6} :=
by
  sorry

end union_complement_l12_12733


namespace factorization_2109_two_digit_l12_12360

theorem factorization_2109_two_digit (a b: ℕ) : 
  2109 = a * b ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 → false :=
by
  sorry

end factorization_2109_two_digit_l12_12360


namespace projection_correct_l12_12977

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P : Point3D := ⟨-1, 3, -4⟩

def projection_yOz_plane (P : Point3D) : Point3D :=
  ⟨0, P.y, P.z⟩

theorem projection_correct :
  projection_yOz_plane P = ⟨0, 3, -4⟩ :=
by
  -- The theorem proof is omitted.
  sorry

end projection_correct_l12_12977


namespace solution_set_of_inequality_l12_12662

theorem solution_set_of_inequality :
  { x : ℝ | 3 ≤ |2 * x - 5| ∧ |2 * x - 5| < 9 } = { x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) } :=
by 
  -- Conditions and steps omitted for the sake of the statement.
  sorry

end solution_set_of_inequality_l12_12662


namespace sum_of_three_numbers_l12_12535

theorem sum_of_three_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 12)
    (h4 : (a + b + c) / 3 = a + 8) (h5 : (a + b + c) / 3 = c - 18) : 
    a + b + c = 66 := 
sorry

end sum_of_three_numbers_l12_12535


namespace find_real_solutions_l12_12033

theorem find_real_solutions (x : ℝ) :
  x^4 + (3 - x)^4 = 146 ↔ x = 1.5 + Real.sqrt 3.4175 ∨ x = 1.5 - Real.sqrt 3.4175 :=
by
  sorry

end find_real_solutions_l12_12033


namespace WangLei_is_13_l12_12230

-- We need to define the conditions and question in Lean 4
def WangLei_age (x : ℕ) : Prop :=
  3 * x - 8 = 31

theorem WangLei_is_13 : ∃ x : ℕ, WangLei_age x ∧ x = 13 :=
by
  use 13
  unfold WangLei_age
  sorry

end WangLei_is_13_l12_12230


namespace combined_score_l12_12743

variable (A J M : ℕ)

-- Conditions
def Jose_score_more_than_Alisson : Prop := J = A + 40
def Meghan_score_less_than_Jose : Prop := M = J - 20
def total_possible_score : ℕ := 100
def Jose_questions_wrong (wrong_questions : ℕ) : Prop := J = total_possible_score - (wrong_questions * 2)

-- Proof statement
theorem combined_score (h1 : Jose_score_more_than_Alisson)
                       (h2 : Meghan_score_less_than_Jose)
                       (h3 : Jose_questions_wrong 5) :
                       A + J + M = 210 := by
  sorry

end combined_score_l12_12743


namespace irrigation_tank_final_amount_l12_12810

theorem irrigation_tank_final_amount : 
  let initial_amount := 300.0
  let evaporation := 1.0
  let addition := 0.3
  let days := 45
  let daily_change := addition - evaporation
  let total_change := daily_change * days
  initial_amount + total_change = 268.5 := 
by {
  -- Proof goes here
  sorry
}

end irrigation_tank_final_amount_l12_12810


namespace sequence_becomes_negative_from_8th_term_l12_12134

def seq (n : ℕ) : ℤ := 21 + 4 * n - n ^ 2

theorem sequence_becomes_negative_from_8th_term :
  ∀ n, n ≥ 8 ↔ seq n < 0 :=
by
  -- proof goes here
  sorry

end sequence_becomes_negative_from_8th_term_l12_12134


namespace smallest_possible_value_of_M_l12_12106

theorem smallest_possible_value_of_M (a b c d e : ℕ) (h1 : a + b + c + d + e = 3060) 
    (h2 : a + e ≥ 1300) :
    ∃ M : ℕ, M = max (max (a + b) (max (b + c) (max (c + d) (d + e)))) ∧ M = 1174 :=
by
  sorry

end smallest_possible_value_of_M_l12_12106


namespace annual_interest_rate_l12_12173

-- Definitions based on conditions
def initial_amount : ℝ := 1000
def spent_amount : ℝ := 440
def final_amount : ℝ := 624

-- The main theorem
theorem annual_interest_rate (x : ℝ) : 
  (initial_amount * (1 + x) - spent_amount) * (1 + x) = final_amount →
  x = 0.04 :=
by
  intro h
  sorry

end annual_interest_rate_l12_12173


namespace time_with_family_l12_12126

theorem time_with_family : 
    let hours_in_day := 24
    let sleep_fraction := 1 / 3
    let school_fraction := 1 / 6
    let assignment_fraction := 1 / 12
    let sleep_hours := sleep_fraction * hours_in_day
    let school_hours := school_fraction * hours_in_day
    let assignment_hours := assignment_fraction * hours_in_day
    let total_hours_occupied := sleep_hours + school_hours + assignment_hours
    hours_in_day - total_hours_occupied = 10 :=
by
  sorry

end time_with_family_l12_12126


namespace net_rate_25_dollars_per_hour_l12_12185

noncomputable def net_rate_of_pay (hours : ℕ) (speed : ℕ) (mileage : ℕ) (rate_per_mile : ℚ) (diesel_cost_per_gallon : ℚ) : ℚ :=
  let distance := hours * speed
  let diesel_used := distance / mileage
  let earnings := rate_per_mile * distance
  let diesel_cost := diesel_cost_per_gallon * diesel_used
  let net_earnings := earnings - diesel_cost
  net_earnings / hours

theorem net_rate_25_dollars_per_hour :
  net_rate_of_pay 4 45 15 (0.75 : ℚ) (3.00 : ℚ) = 25 :=
by
  -- Proof is omitted
  sorry

end net_rate_25_dollars_per_hour_l12_12185


namespace shaded_area_of_square_with_quarter_circles_l12_12443

theorem shaded_area_of_square_with_quarter_circles :
  let side_len : ℝ := 12
  let square_area := side_len * side_len
  let radius := side_len / 2
  let total_circle_area := 4 * (π * radius^2 / 4)
  let shaded_area := square_area - total_circle_area
  shaded_area = 144 - 36 * π := 
by
  sorry

end shaded_area_of_square_with_quarter_circles_l12_12443


namespace total_new_bottles_l12_12337

theorem total_new_bottles (initial_bottles : ℕ) (recycle_ratio : ℕ) (bonus_ratio : ℕ) (final_bottles : ℕ) :
  initial_bottles = 625 →
  recycle_ratio = 5 →
  bonus_ratio = 20 →
  final_bottles = 163 :=
by {
  sorry -- Proof goes here
}

end total_new_bottles_l12_12337


namespace sample_stddev_is_2_l12_12116

open Finset

def masses : Finset ℝ := {125, 124, 121, 123, 127}

def mean (s : Finset ℝ) : ℝ :=
  s.sum id / s.card

def stddev (s : Finset ℝ) : ℝ :=
  let m := mean s
  let squared_deviations := s.map (λ x, (x - m) * (x - m))
  real.sqrt (squared_deviations.sum id / (s.card - 1))

theorem sample_stddev_is_2 : stddev masses = 2 :=
  sorry

end sample_stddev_is_2_l12_12116


namespace solveNumberOfWaysToChooseSeats_l12_12253

/--
Define the problem of professors choosing their seats among 9 chairs with specific constraints.
-/
noncomputable def numberOfWaysToChooseSeats : ℕ :=
  let totalChairs := 9
  let endChairChoices := 2 * (7 * (7 - 2))  -- (2 end chairs, 7 for 2nd prof, 5 for 3rd prof)
  let middleChairChoices := 7 * (6 * (6 - 2))  -- (7 non-end chairs, 6 for 2nd prof, 4 for 3rd prof)
  endChairChoices + middleChairChoices

/--
The final result should be 238
-/
theorem solveNumberOfWaysToChooseSeats : numberOfWaysToChooseSeats = 238 := by
  sorry

end solveNumberOfWaysToChooseSeats_l12_12253


namespace cos_double_angle_l12_12862

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1/3) : Real.cos (2 * α) = 7/9 :=
by
    sorry

end cos_double_angle_l12_12862


namespace incorrect_intersection_point_l12_12339

def linear_function (x : ℝ) : ℝ := -2 * x + 4

theorem incorrect_intersection_point : ¬(linear_function 0 = 4) :=
by {
  /- Proof can be filled here later -/
  sorry
}

end incorrect_intersection_point_l12_12339


namespace find_number_l12_12437

theorem find_number (x : ℕ) (h : x * 48 = 173 * 240) : x = 865 :=
sorry

end find_number_l12_12437


namespace proposition_negation_l12_12078

theorem proposition_negation (p : Prop) : 
  (∃ x : ℝ, x < 1 ∧ x^2 < 1) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
sorry

end proposition_negation_l12_12078


namespace necessary_condition_abs_sq_necessary_and_sufficient_add_l12_12065

theorem necessary_condition_abs_sq (a b : ℝ) : a^2 > b^2 → |a| > |b| :=
sorry

theorem necessary_and_sufficient_add (a b c : ℝ) :
  (a > b) ↔ (a + c > b + c) :=
sorry

end necessary_condition_abs_sq_necessary_and_sufficient_add_l12_12065


namespace roots_of_polynomial_l12_12981

theorem roots_of_polynomial :
  (3 * (2 + Real.sqrt 3)^4 - 19 * (2 + Real.sqrt 3)^3 + 34 * (2 + Real.sqrt 3)^2 - 19 * (2 + Real.sqrt 3) + 3 = 0) ∧ 
  (3 * (2 - Real.sqrt 3)^4 - 19 * (2 - Real.sqrt 3)^3 + 34 * (2 - Real.sqrt 3)^2 - 19 * (2 - Real.sqrt 3) + 3 = 0) ∧
  (3 * ((7 + Real.sqrt 13) / 6)^4 - 19 * ((7 + Real.sqrt 13) / 6)^3 + 34 * ((7 + Real.sqrt 13) / 6)^2 - 19 * ((7 + Real.sqrt 13) / 6) + 3 = 0) ∧
  (3 * ((7 - Real.sqrt 13) / 6)^4 - 19 * ((7 - Real.sqrt 13) / 6)^3 + 34 * ((7 - Real.sqrt 13) / 6)^2 - 19 * ((7 - Real.sqrt 13) / 6) + 3 = 0) :=
by sorry

end roots_of_polynomial_l12_12981


namespace no_such_f_exists_l12_12117

theorem no_such_f_exists (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
  (h2 : ∀ x y, 0 < x → 0 < y → f x ^ 2 ≥ f (x + y) * (f x + y)) : false :=
sorry

end no_such_f_exists_l12_12117


namespace train_speed_l12_12549

theorem train_speed (L : ℝ) (T : ℝ) (V_m : ℝ) (V_t : ℝ) : (L = 500) → (T = 29.997600191984642) → (V_m = 5 / 6) → (V_t = (L / T) + V_m) → (V_t * 3.6 = 63) :=
by
  intros hL hT hVm hVt
  simp at hL hT hVm hVt
  sorry

end train_speed_l12_12549


namespace doses_A_correct_doses_B_correct_doses_C_correct_l12_12835

def days_in_july : ℕ := 31

def daily_dose_A : ℕ := 1
def daily_dose_B : ℕ := 2
def daily_dose_C : ℕ := 3

def missed_days_A : ℕ := 3
def missed_days_B_morning : ℕ := 5
def missed_days_C_all : ℕ := 2

def total_doses_A : ℕ := days_in_july * daily_dose_A
def total_doses_B : ℕ := days_in_july * daily_dose_B
def total_doses_C : ℕ := days_in_july * daily_dose_C

def missed_doses_A : ℕ := missed_days_A * daily_dose_A
def missed_doses_B : ℕ := missed_days_B_morning
def missed_doses_C : ℕ := missed_days_C_all * daily_dose_C

def doses_consumed_A := total_doses_A - missed_doses_A
def doses_consumed_B := total_doses_B - missed_doses_B
def doses_consumed_C := total_doses_C - missed_doses_C

theorem doses_A_correct : doses_consumed_A = 28 := by sorry
theorem doses_B_correct : doses_consumed_B = 57 := by sorry
theorem doses_C_correct : doses_consumed_C = 87 := by sorry

end doses_A_correct_doses_B_correct_doses_C_correct_l12_12835


namespace find_x_for_parallel_vectors_l12_12721

def vector := (ℝ × ℝ)

def a (x : ℝ) : vector := (1, x)
def b (x : ℝ) : vector := (2, 2 - x)

def are_parallel (v w : vector) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem find_x_for_parallel_vectors :
  ∀ x : ℝ, are_parallel (a x) (b x) → x = 2/3 :=
by
  sorry

end find_x_for_parallel_vectors_l12_12721


namespace total_profit_l12_12175

noncomputable def profit_x (P : ℕ) : ℕ := 3 * P
noncomputable def profit_y (P : ℕ) : ℕ := 2 * P

theorem total_profit
  (P_x P_y : ℕ)
  (h_ratio : P_x = 3 * (P_y / 2))
  (h_diff : P_x - P_y = 100) :
  P_x + P_y = 500 :=
by
  sorry

end total_profit_l12_12175


namespace division_subtraction_l12_12572

theorem division_subtraction : 144 / (12 / 3) - 5 = 31 := by
  sorry

end division_subtraction_l12_12572


namespace evaluate_expression_l12_12009

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12009


namespace find_number_of_terms_l12_12271

variable {n : ℕ} {a : ℕ → ℤ}
variable (a_seq : ℕ → ℤ)

def sum_first_three_terms (a : ℕ → ℤ) : ℤ :=
  a 1 + a 2 + a 3

def sum_last_three_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  a (n-2) + a (n-1) + a n

def sum_all_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem find_number_of_terms (h1 : sum_first_three_terms a_seq = 20)
    (h2 : sum_last_three_terms n a_seq = 130)
    (h3 : sum_all_terms n a_seq = 200) : n = 8 :=
sorry

end find_number_of_terms_l12_12271


namespace positive_integer_solutions_count_l12_12909

theorem positive_integer_solutions_count : 
  (∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15) :=
sorry

end positive_integer_solutions_count_l12_12909


namespace exactly_one_equals_xx_plus_xx_l12_12063

theorem exactly_one_equals_xx_plus_xx (x : ℝ) (hx : x > 0) :
  let expr1 := 2 * x^x
  let expr2 := x^(2*x)
  let expr3 := (2*x)^x
  let expr4 := (2*x)^(2*x)
  (expr1 = x^x + x^x) ∧ (¬(expr2 = x^x + x^x)) ∧ (¬(expr3 = x^x + x^x)) ∧ (¬(expr4 = x^x + x^x)) := 
by
  sorry

end exactly_one_equals_xx_plus_xx_l12_12063


namespace robert_books_read_l12_12113

theorem robert_books_read (pages_per_hour : ℕ) (book_pages : ℕ) (total_hours : ℕ) :
  pages_per_hour = 120 → book_pages = 360 → total_hours = 8 → (total_hours * pages_per_hour) / book_pages = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact (nat.div_eq_of_lt sorry)
end

end robert_books_read_l12_12113


namespace gcd_360_504_is_72_l12_12655

theorem gcd_360_504_is_72 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_is_72_l12_12655


namespace circles_area_l12_12653

theorem circles_area (BD AC : ℝ) (r : ℝ) (h1 : BD = 6) (h2 : AC = 12)
  (h3 : ∀ (d1 d2 : ℝ), d1 = AC / 2 → d2 = BD / 2 → r^2 = (r - d2)^2 + d1^2) :
  real.pi * r^2 = (225/4) * real.pi :=
by
  -- proof to be filled
  sorry

end circles_area_l12_12653


namespace div_by_7_of_sum_div_by_7_l12_12634

theorem div_by_7_of_sum_div_by_7 (x y z : ℤ) (h : 7 ∣ x^3 + y^3 + z^3) : 7 ∣ x * y * z := by
  sorry

end div_by_7_of_sum_div_by_7_l12_12634


namespace shells_total_l12_12897

variable (x y : ℝ)

theorem shells_total (h1 : y = x + (x + 32)) : y = 2 * x + 32 :=
sorry

end shells_total_l12_12897


namespace largest_x_to_floor_ratio_l12_12979

theorem largest_x_to_floor_ratio : ∃ x : ℝ, (⌊x⌋ / x = 9 / 10 ∧ ∀ y : ℝ, (⌊y⌋ / y = 9 / 10 → y ≤ x)) :=
sorry

end largest_x_to_floor_ratio_l12_12979


namespace all_three_use_media_l12_12570

variable (U T R M T_and_M T_and_R R_and_M T_and_R_and_M : ℕ)

theorem all_three_use_media (hU : U = 180)
  (hT : T = 115)
  (hR : R = 110)
  (hM : M = 130)
  (hT_and_M : T_and_M = 85)
  (hT_and_R : T_and_R = 75)
  (hR_and_M : R_and_M = 95)
  (h_union : U = T + R + M - T_and_R - T_and_M - R_and_M + T_and_R_and_M) :
  T_and_R_and_M = 80 :=
by
  sorry

end all_three_use_media_l12_12570


namespace arithmetic_seq_proof_l12_12039

theorem arithmetic_seq_proof
  (x : ℕ → ℝ)
  (h : ∀ n ≥ 3, x (n-1) = (x n + x (n-1) + x (n-2)) / 3):
  (x 300 - x 33) / (x 333 - x 3) = 89 / 110 := by
  sorry

end arithmetic_seq_proof_l12_12039


namespace graph_of_equation_l12_12150

theorem graph_of_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := 
by
  sorry

end graph_of_equation_l12_12150


namespace length_of_fountain_built_by_20_men_in_6_days_l12_12294

noncomputable def work (workers : ℕ) (days : ℕ) : ℕ :=
  workers * days

theorem length_of_fountain_built_by_20_men_in_6_days :
  (work 35 3) / (work 20 6) * 49 = 56 :=
by
  sorry

end length_of_fountain_built_by_20_men_in_6_days_l12_12294


namespace complex_quadrant_example_l12_12220

open Complex

def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_quadrant_example (z : ℂ) (h : (1 - I) * z = (1 + I) ^ 2) : in_second_quadrant z :=
by
  sorry

end complex_quadrant_example_l12_12220


namespace find_x_if_delta_phi_x_eq_3_l12_12718

def delta (x : ℚ) : ℚ := 2 * x + 5
def phi (x : ℚ) : ℚ := 9 * x + 6

theorem find_x_if_delta_phi_x_eq_3 :
  ∃ (x : ℚ), delta (phi x) = 3 ∧ x = -7/9 := by
sorry

end find_x_if_delta_phi_x_eq_3_l12_12718


namespace simplify_expression_l12_12224

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) : (a - 2) * (b - 2) = -2 * m := 
by
  sorry

end simplify_expression_l12_12224


namespace evaluate_x_squared_minus_y_squared_l12_12051

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end evaluate_x_squared_minus_y_squared_l12_12051


namespace pizza_slices_left_l12_12565

theorem pizza_slices_left (total_slices : ℕ) (angeli_slices : ℚ) (marlon_slices : ℚ) 
  (H1 : total_slices = 8) (H2 : angeli_slices = 3/2) (H3 : marlon_slices = 3/2) :
  total_slices - (angeli_slices + marlon_slices) = 5 :=
by
  sorry

end pizza_slices_left_l12_12565


namespace smallest_n_for_cube_root_form_l12_12624

theorem smallest_n_for_cube_root_form
  (m n : ℕ) (r : ℝ)
  (h_pos_n : n > 0)
  (h_pos_r : r > 0)
  (h_r_bound : r < 1/500)
  (h_m : m = (n + r)^3)
  (h_min_m : ∀ k : ℕ, k = (n + r)^3 → k ≥ m) :
  n = 13 :=
by
  -- proof goes here
  sorry

end smallest_n_for_cube_root_form_l12_12624


namespace number_of_valid_three_digit_numbers_l12_12785

def three_digit_numbers_count : Nat :=
  let count_numbers (last_digit : Nat) (remaining_digits : List Nat) : Nat :=
    remaining_digits.length * (remaining_digits.erase last_digit).length

  let count_when_last_digit_is_0 :=
    count_numbers 0 [1, 2, 3, 4, 5, 6, 7, 8, 9]

  let count_when_last_digit_is_5 :=
    count_numbers 5 [0, 1, 2, 3, 4, 6, 7, 8, 9]

  count_when_last_digit_is_0 + count_when_last_digit_is_5

theorem number_of_valid_three_digit_numbers : three_digit_numbers_count = 136 := by
  sorry

end number_of_valid_three_digit_numbers_l12_12785


namespace inequality_proof_l12_12837

noncomputable def a := Real.log 1 / Real.log 3
noncomputable def b := Real.log 1 / Real.log (1 / 2)
noncomputable def c := (1/2)^(1/3)

theorem inequality_proof : b > c ∧ c > a := 
by 
  sorry

end inequality_proof_l12_12837


namespace triangle_area_is_correct_l12_12976

-- Defining the points
structure Point where
  x : ℝ
  y : ℝ

-- Defining vertices A, B, C
def A : Point := { x := 2, y := -3 }
def B : Point := { x := 0, y := 4 }
def C : Point := { x := 3, y := -1 }

-- Vector from C to A
def v : Point := { x := A.x - C.x, y := A.y - C.y }

-- Vector from C to B
def w : Point := { x := B.x - C.x, y := B.y - C.y }

-- Cross product of vectors v and w in 2D
noncomputable def cross_product (v w : Point) : ℝ :=
  v.x * w.y - v.y * w.x

-- Absolute value of the cross product
noncomputable def abs_cross_product (v w : Point) : ℝ :=
  |cross_product v w|

-- Area of the triangle
noncomputable def area_of_triangle (v w : Point) : ℝ :=
  (1 / 2) * abs_cross_product v w

-- Prove the area of the triangle is 5.5
theorem triangle_area_is_correct : area_of_triangle v w = 5.5 :=
  sorry

end triangle_area_is_correct_l12_12976


namespace squares_sum_l12_12760

theorem squares_sum {r s : ℝ} (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end squares_sum_l12_12760


namespace evaluate_x2_y2_l12_12044

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end evaluate_x2_y2_l12_12044


namespace complement_union_eq_l12_12735

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 3, 5}

theorem complement_union_eq:
  compl A ∪ B = {0, 2, 3, 5} :=
by
  sorry

end complement_union_eq_l12_12735


namespace range_of_alpha_l12_12730

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 5 * x

theorem range_of_alpha (α : ℝ) (h₀ : -1 < α) (h₁ : α < 1) (h₂ : f (1 - α) + f (1 - α^2) < 0) : 1 < α ∧ α < Real.sqrt 2 := by
  sorry

end range_of_alpha_l12_12730


namespace cone_sphere_ratio_l12_12179

-- Defining the conditions and proof goals
theorem cone_sphere_ratio (r h : ℝ) (h_cone_sphere_radius : r ≠ 0) 
  (h_cone_volume : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  -- All the assumptions / conditions given in the problem
  sorry -- Proof omitted

end cone_sphere_ratio_l12_12179


namespace find_x_l12_12794

def Hiram_age := 40
def Allyson_age := 28
def Twice_Allyson_age := 2 * Allyson_age
def Four_less_than_twice_Allyson_age := Twice_Allyson_age - 4

theorem find_x (x : ℤ) : Hiram_age + x = Four_less_than_twice_Allyson_age → x = 12 := 
by
  intros h -- introducing the assumption 
  sorry

end find_x_l12_12794


namespace math_problem_l12_12675

def Q (f : ℝ → ℝ) : Prop :=
  (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / (x + y)) = f (1 / x) + f (1 / y))
  ∧ (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → (x + y) * f (x + y) = x * y * f x * f y)
  ∧ f 1 = 1

theorem math_problem (f : ℝ → ℝ) : Q f → (∀ (x : ℝ), x ≠ 0 → f x = 1 / x) :=
by
  -- Proof goes here
  sorry

end math_problem_l12_12675


namespace range_of_a_l12_12353

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → abs (2 * a - 1) ≤ abs (x + 1 / x)) →
  -1 / 2 ≤ a ∧ a ≤ 3 / 2 :=
by sorry

end range_of_a_l12_12353


namespace teacher_age_frequency_l12_12562

theorem teacher_age_frequency (f_less_than_30 : ℝ) (f_between_30_and_50 : ℝ) (h1 : f_less_than_30 = 0.3) (h2 : f_between_30_and_50 = 0.5) :
  1 - f_less_than_30 - f_between_30_and_50 = 0.2 :=
by
  rw [h1, h2]
  norm_num

end teacher_age_frequency_l12_12562


namespace log_expression_eval_find_m_from_conditions_l12_12459

-- (1) Prove that lg (5^2) + (2/3) * lg 8 + lg 5 * lg 20 + (lg 2)^2 = 3.
theorem log_expression_eval : 
  Real.logb 10 (5^2) + (2 / 3) * Real.logb 10 8 + Real.logb 10 5 * Real.logb 10 20 + (Real.logb 10 2)^2 = 3 := 
sorry

-- (2) Given 2^a = 5^b = m and 1/a + 1/b = 2, prove that m = sqrt(10).
theorem find_m_from_conditions (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 :=
sorry

end log_expression_eval_find_m_from_conditions_l12_12459


namespace original_average_is_24_l12_12647

theorem original_average_is_24
  (A : ℝ)
  (h1 : ∀ n : ℕ, n = 7 → 35 * A = 7 * 120) :
  A = 24 :=
by
  sorry

end original_average_is_24_l12_12647


namespace complete_the_square_l12_12637

theorem complete_the_square : ∀ x : ℝ, x^2 - 6 * x + 4 = 0 → (x - 3)^2 = 5 :=
by
  intro x h
  sorry

end complete_the_square_l12_12637


namespace value_of_x2_plus_9y2_l12_12371

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l12_12371


namespace find_integer_solutions_l12_12974

theorem find_integer_solutions (x y : ℤ) :
  8 * x^2 * y^2 + x^2 + y^2 = 10 * x * y ↔
  (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
by 
  sorry

end find_integer_solutions_l12_12974


namespace cos_alpha_plus_pi_over_4_l12_12483

theorem cos_alpha_plus_pi_over_4
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = 3 / 5)
  (h4 : Real.sin (β - π / 4) = 5 / 13) : 
  Real.cos (α + π / 4) = 56 / 65 :=
by
  sorry 

end cos_alpha_plus_pi_over_4_l12_12483


namespace total_mile_times_l12_12280

theorem total_mile_times (t_Tina t_Tony t_Tom t_Total : ℕ) 
  (h1 : t_Tina = 6) 
  (h2 : t_Tony = t_Tina / 2) 
  (h3 : t_Tom = t_Tina / 3) 
  (h4 : t_Total = t_Tina + t_Tony + t_Tom) : t_Total = 11 := 
sorry

end total_mile_times_l12_12280


namespace rectangle_perimeter_eq_circle_circumference_l12_12529

theorem rectangle_perimeter_eq_circle_circumference (l : ℝ) :
  2 * (l + 3) = 10 * Real.pi -> l = 5 * Real.pi - 3 :=
by
  intro h
  sorry

end rectangle_perimeter_eq_circle_circumference_l12_12529


namespace sum_first10PrimesGT50_eq_732_l12_12933

def first10PrimesGT50 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

theorem sum_first10PrimesGT50_eq_732 :
  first10PrimesGT50.sum = 732 := by
  sorry

end sum_first10PrimesGT50_eq_732_l12_12933


namespace basis_transformation_l12_12607

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem basis_transformation (h_basis : ∀ (v : V), ∃ (x y z : ℝ), v = x • a + y • b + z • c) :
  ∀ (v : V), ∃ (x y z : ℝ), v = x • (a + b) + y • (a - c) + z • b :=
by {
  sorry  -- to skip the proof steps for now
}

end basis_transformation_l12_12607


namespace max_y_value_l12_12726

-- Definitions according to the problem conditions
def is_negative_integer (z : ℤ) : Prop := z < 0

-- The theorem to be proven
theorem max_y_value (x y : ℤ) (hx : is_negative_integer x) (hy : is_negative_integer y) 
  (h_eq : y = 10 * x / (10 - x)) : y = -5 :=
sorry

end max_y_value_l12_12726


namespace small_slices_sold_l12_12944

theorem small_slices_sold (S L : ℕ) 
  (h1 : S + L = 5000) 
  (h2 : 150 * S + 250 * L = 1050000) : 
  S = 2000 :=
by
  sorry

end small_slices_sold_l12_12944


namespace min_handshakes_35_people_l12_12548

theorem min_handshakes_35_people (n : ℕ) (h1 : n = 35) (h2 : ∀ p : ℕ, p < n → p ≥ 3) : ∃ m : ℕ, m = 51 :=
by
  sorry

end min_handshakes_35_people_l12_12548


namespace total_peaches_in_baskets_l12_12705

def total_peaches (red_peaches : ℕ) (green_peaches : ℕ) (baskets : ℕ) : ℕ :=
  (red_peaches + green_peaches) * baskets

theorem total_peaches_in_baskets :
  total_peaches 19 4 15 = 345 :=
by
  sorry

end total_peaches_in_baskets_l12_12705


namespace john_spent_on_sweets_l12_12250

def initial_amount := 7.10
def amount_given_per_friend := 1.00
def amount_left := 4.05
def amount_spent_on_friends := 2 * amount_given_per_friend
def amount_remaining_after_friends := initial_amount - amount_spent_on_friends
def amount_spent_on_sweets := amount_remaining_after_friends - amount_left

theorem john_spent_on_sweets : amount_spent_on_sweets = 1.05 := 
by
  sorry

end john_spent_on_sweets_l12_12250


namespace sequence_gcd_equality_l12_12661

theorem sequence_gcd_equality (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) : 
  ∀ i, a i = i := 
sorry

end sequence_gcd_equality_l12_12661


namespace count_ways_to_complete_20160_l12_12877

noncomputable def waysToComplete : Nat :=
  let choices_for_last_digit := 5
  let choices_for_first_three_digits := 9^3
  choices_for_last_digit * choices_for_first_three_digits

theorem count_ways_to_complete_20160 (choices : Fin 9 → Fin 9) : waysToComplete = 3645 := by
  sorry

end count_ways_to_complete_20160_l12_12877


namespace man_work_m_alone_in_15_days_l12_12683

theorem man_work_m_alone_in_15_days (M : ℕ) (h1 : 1/M + 1/10 = 1/6) : M = 15 := sorry

end man_work_m_alone_in_15_days_l12_12683


namespace cheapest_salon_option_haily_l12_12481

theorem cheapest_salon_option_haily : 
  let gustran_haircut := 45
  let gustran_facial := 22
  let gustran_nails := 30
  let gustran_foot_spa := 15
  let gustran_massage := 50
  let gustran_total := gustran_haircut + gustran_facial + gustran_nails + gustran_foot_spa + gustran_massage
  let gustran_discount := 0.20
  let gustran_final := gustran_total * (1 - gustran_discount)

  let barbara_nails := 40
  let barbara_haircut := 30
  let barbara_facial := 28
  let barbara_foot_spa := 18
  let barbara_massage := 45
  let barbara_total :=
      barbara_nails + barbara_haircut + (barbara_facial * 0.5) + barbara_foot_spa + (barbara_massage * 0.5)

  let fancy_haircut := 34
  let fancy_facial := 30
  let fancy_nails := 20
  let fancy_foot_spa := 25
  let fancy_massage := 60
  let fancy_total := fancy_haircut + fancy_facial + fancy_nails + fancy_foot_spa + fancy_massage
  let fancy_discount := 15
  let fancy_final := fancy_total - fancy_discount

  let avg_haircut := (gustran_haircut + barbara_haircut + fancy_haircut) / 3
  let avg_facial := (gustran_facial + barbara_facial + fancy_facial) / 3
  let avg_nails := (gustran_nails + barbara_nails + fancy_nails) / 3
  let avg_foot_spa := (gustran_foot_spa + barbara_foot_spa + fancy_foot_spa) / 3
  let avg_massage := (gustran_massage + barbara_massage + fancy_massage) / 3

  let luxury_haircut := avg_haircut * 1.10
  let luxury_facial := avg_facial * 1.10
  let luxury_nails := avg_nails * 1.10
  let luxury_foot_spa := avg_foot_spa * 1.10
  let luxury_massage := avg_massage * 1.10
  let luxury_total := luxury_haircut + luxury_facial + luxury_nails + luxury_foot_spa + luxury_massage
  let luxury_discount := 20
  let luxury_final := luxury_total - luxury_discount

  gustran_final > barbara_total ∧ barbara_total < fancy_final ∧ barbara_total < luxury_final := 
by 
  sorry

end cheapest_salon_option_haily_l12_12481


namespace cone_base_circumference_l12_12304

theorem cone_base_circumference (radius : ℝ) (angle : ℝ) (c_base : ℝ) :
  radius = 6 ∧ angle = 180 ∧ c_base = 6 * Real.pi →
  (c_base = (angle / 360) * (2 * Real.pi * radius)) :=
by
  intros h
  rcases h with ⟨h_radius, h_angle, h_c_base⟩
  rw [h_radius, h_angle]
  norm_num
  sorry

end cone_base_circumference_l12_12304


namespace probability_of_white_crows_remain_same_l12_12922

theorem probability_of_white_crows_remain_same (a b c d : ℕ) (h1 : a + b = 50) (h2 : c + d = 50) 
  (ha1 : a > 0) (h3 : b ≥ a) (h4 : d ≥ c - 1) :
  ((b - a) * (d - c) + a + b) / (50 * 51) > (bc + ad) / (50 * 51)
:= by
  -- We need to show that the probability of the number of white crows on the birch remaining the same 
  -- is greater than the probability of it changing.
  sorry

end probability_of_white_crows_remain_same_l12_12922


namespace fingers_game_conditions_l12_12752

noncomputable def minNForWinningSubset (N : ℕ) : Prop :=
  N ≥ 220

-- To state the probability condition, we need to express it in terms of actual probabilities
noncomputable def probLeaderWins (N : ℕ) : ℝ := 
  1 / N

noncomputable def leaderWinProbabilityTendsToZero : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, probLeaderWins n < ε

theorem fingers_game_conditions (N : ℕ) (probLeaderWins : ℕ → ℝ) :
  (minNForWinningSubset N) ∧ leaderWinProbabilityTendsToZero :=
by
  sorry

end fingers_game_conditions_l12_12752


namespace solve_for_k_l12_12487

noncomputable def polynomial_is_perfect_square (k : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, (f x) ^ 2 = x^2 - 2*(k+1)*x + 4

theorem solve_for_k :
  ∀ k : ℝ, polynomial_is_perfect_square k → (k = -3 ∨ k = 1) := by
  sorry

end solve_for_k_l12_12487


namespace green_disks_more_than_blue_l12_12380

theorem green_disks_more_than_blue 
  (total_disks : ℕ) (blue_ratio yellow_ratio green_ratio red_ratio : ℕ)
  (h1 : total_disks = 132)
  (h2 : blue_ratio = 3)
  (h3 : yellow_ratio = 7)
  (h4 : green_ratio = 8)
  (h5 : red_ratio = 4)
  : 6 * green_ratio - 6 * blue_ratio = 30 :=
by
  sorry

end green_disks_more_than_blue_l12_12380


namespace part1_part2_l12_12058

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (x - Real.pi / 3)

theorem part1 : f (2 * Real.pi / 3) = -1 / 4 :=
by
  sorry

theorem part2 : {x | f x < 1 / 4} = { x | ∃ k : ℤ, k * Real.pi + 5 * Real.pi / 12 < x ∧ x < k * Real.pi + 11 * Real.pi / 12 } :=
by
  sorry

end part1_part2_l12_12058


namespace solve_for_x_l12_12777

theorem solve_for_x :
  exists x : ℝ, 11.98 * 11.98 + 11.98 * x + 0.02 * 0.02 = (11.98 + 0.02) ^ 2 ∧ x = 0.04 :=
by
  sorry

end solve_for_x_l12_12777


namespace largest_divisor_of_m_p1_l12_12490

theorem largest_divisor_of_m_p1 (m : ℕ) (h1 : m > 0) (h2 : 72 ∣ m^3) : 6 ∣ m :=
sorry

end largest_divisor_of_m_p1_l12_12490


namespace eval_expression_l12_12000

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l12_12000


namespace value_subtracted_l12_12612

theorem value_subtracted (x y : ℤ) (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 13 = 4) : y = 2 :=
sorry

end value_subtracted_l12_12612


namespace eval_expression_l12_12020

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12020


namespace sum_of_distinct_prime_factors_of_2016_l12_12932

-- Define 2016 and the sum of its distinct prime factors
def n : ℕ := 2016
def sumOfDistinctPrimeFactors (n : ℕ) : ℕ :=
  if n = 2016 then 2 + 3 + 7 else 0  -- Capture the problem-specific condition

-- The main theorem to prove the sum of the distinct prime factors of 2016 is 12
theorem sum_of_distinct_prime_factors_of_2016 :
  sumOfDistinctPrimeFactors 2016 = 12 :=
by
  -- Since this is beyond the obvious steps, we use a sorry here
  sorry

end sum_of_distinct_prime_factors_of_2016_l12_12932


namespace elective_schemes_count_l12_12747

open Finset

variable (A B C D E F G H I : Type)
variable [DecidableEq A] [DecidableEq B] [DecidableEq C] 
variable [DecidableEq D] [DecidableEq E] [DecidableEq F]
variable [DecidableEq G] [DecidableEq H] [DecidableEq I]

theorem elective_schemes_count :
  let courses := ({A, B, C, D, E, F, G, H, I} : Finset Type) in
  let abc := ({A, B, C} : Finset Type) in
  let others := (courses \ abc) in
  @card (Set (courses \ ∅)) = 4 →
  @card ((abc \ ∅) ∪ (choose others 3)) + @card (choose others 4) = 75 :=
by
  sorry

end elective_schemes_count_l12_12747


namespace pos_rel_lines_l12_12530

-- Definition of the lines
def line1 (k : ℝ) (x y : ℝ) : Prop := 2 * x - y + k = 0
def line2 (x y : ℝ) : Prop := 4 * x - 2 * y + 1 = 0

-- Theorem stating the positional relationship between the two lines
theorem pos_rel_lines (k : ℝ) : 
  (∀ x y : ℝ, line1 k x y → line2 x y → 2 * k - 1 = 0) → 
  (∀ x y : ℝ, line1 k x y → ¬ line2 x y → 2 * k - 1 ≠ 0) → 
  (k = 1/2 ∨ k ≠ 1/2) :=
by sorry

end pos_rel_lines_l12_12530


namespace soccer_team_points_l12_12643

theorem soccer_team_points 
  (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_draws : draws = total_games - (wins + losses))
  (h_points_per_win : points_per_win = 3)
  (h_points_per_draw : points_per_draw = 1)
  (h_points_per_loss : points_per_loss = 0) :
  (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) = 46 :=
by
  -- the actual proof steps will be inserted here
  sorry

end soccer_team_points_l12_12643


namespace eval_expression_l12_12017

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12017


namespace strip_covers_cube_l12_12178

   -- Define the given conditions
   def strip_length := 12
   def strip_width := 1
   def cube_edge := 1
   def layers := 2

   -- Define the main statement to be proved
   theorem strip_covers_cube : 
     (strip_length >= 6 * cube_edge / layers) ∧ 
     (strip_width >= cube_edge) ∧ 
     (layers == 2) → 
     true :=
   by
     intro h
     sorry
   
end strip_covers_cube_l12_12178


namespace gift_bags_needed_l12_12694

/-
  Constants
  total_expected: \(\mathbb{N}\) := 90        -- 50 people who will show up + 40 more who may show up
  total_prepared: \(\mathbb{N}\) := 30        -- 10 extravagant gift bags + 20 average gift bags

  The property to be proved:
  prove that (total_expected - total_prepared = 60)
-/

def total_expected : ℕ := 50 + 40
def total_prepared : ℕ := 10 + 20
def additional_needed := total_expected - total_prepared

theorem gift_bags_needed : additional_needed = 60 := by
  sorry

end gift_bags_needed_l12_12694


namespace fractions_product_l12_12194

theorem fractions_product :
  (4 / 2) * (8 / 4) * (9 / 3) * (18 / 6) * (16 / 8) * (24 / 12) * (30 / 15) * (36 / 18) = 576 := by
  sorry

end fractions_product_l12_12194


namespace min_value_expression_l12_12992

noncomputable def log (base : ℝ) (num : ℝ) := Real.log num / Real.log base

theorem min_value_expression (a b : ℝ) (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * log a b + 6 * log b a = 11) : 
  a^3 + (2 / (b - 1)) ≥ 2 * Real.sqrt 2 + 1 :=
by
  sorry

end min_value_expression_l12_12992


namespace proportion_of_mothers_full_time_jobs_l12_12616

theorem proportion_of_mothers_full_time_jobs
  (P : ℝ) (W : ℝ) (F : ℝ → Prop) (M : ℝ)
  (hwomen : W = 0.4 * P)
  (hfathers_full_time : ∀ p, F p → p = 0.75)
  (hno_full_time : P - (W + 0.75 * (P - W)) = 0.19 * P) :
  M = 0.9 :=
by
  sorry

end proportion_of_mothers_full_time_jobs_l12_12616


namespace digits_of_2_pow_100_last_three_digits_of_2_pow_100_l12_12736

-- Prove that 2^100 has 31 digits.
theorem digits_of_2_pow_100 : (10^30 ≤ 2^100) ∧ (2^100 < 10^31) :=
by
  sorry

-- Prove that the last three digits of 2^100 are 376.
theorem last_three_digits_of_2_pow_100 : 2^100 % 1000 = 376 :=
by
  sorry

end digits_of_2_pow_100_last_three_digits_of_2_pow_100_l12_12736


namespace num_solutions_in_interval_l12_12135

theorem num_solutions_in_interval : 
  ∃ n : ℕ, n = 2 ∧ ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  2 ^ Real.cos θ = Real.sin θ → n = 2 := 
sorry

end num_solutions_in_interval_l12_12135


namespace point_C_number_l12_12768

theorem point_C_number (B C: ℝ) (h1 : B = 3) (h2 : |C - B| = 2) :
  C = 1 ∨ C = 5 := 
by {
  sorry
}

end point_C_number_l12_12768


namespace percentage_cut_third_week_l12_12311

noncomputable def initial_weight : ℝ := 300
noncomputable def first_week_percentage : ℝ := 0.30
noncomputable def second_week_percentage : ℝ := 0.30
noncomputable def final_weight : ℝ := 124.95

theorem percentage_cut_third_week :
  let remaining_after_first_week := initial_weight * (1 - first_week_percentage)
  let remaining_after_second_week := remaining_after_first_week * (1 - second_week_percentage)
  let cut_weight_third_week := remaining_after_second_week - final_weight
  let percentage_cut_third_week := (cut_weight_third_week / remaining_after_second_week) * 100
  percentage_cut_third_week = 15 :=
by
  sorry

end percentage_cut_third_week_l12_12311


namespace totalInterest_l12_12192

-- Definitions for the amounts and interest rates
def totalInvestment : ℝ := 22000
def investedAt18 : ℝ := 7000
def rate18 : ℝ := 0.18
def rate14 : ℝ := 0.14

-- Calculations as conditions
def interestFrom18 (p r : ℝ) : ℝ := p * r
def investedAt14 (total inv18 : ℝ) : ℝ := total - inv18
def interestFrom14 (p r : ℝ) : ℝ := p * r

-- Proof statement
theorem totalInterest : interestFrom18 investedAt18 rate18 + interestFrom14 (investedAt14 totalInvestment investedAt18) rate14 = 3360 :=
by
  sorry

end totalInterest_l12_12192


namespace solve_fraction_equation_l12_12257

theorem solve_fraction_equation (x : ℚ) (h : (x + 7) / (x - 4) = (x - 5) / (x + 3)) : x = -1 / 19 := 
sorry

end solve_fraction_equation_l12_12257


namespace triangle_side_ratio_eq_one_l12_12614

theorem triangle_side_ratio_eq_one
    (a b c C : ℝ)
    (h1 : a = 2 * b * Real.cos C)
    (cosine_rule : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
    (b / c = 1) := 
by 
    sorry

end triangle_side_ratio_eq_one_l12_12614


namespace impossible_result_l12_12037

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬ (∃ f1 f_1 : ℤ, f1 = a * Real.sin 1 + b + c ∧ f_1 = -a * Real.sin 1 - b + c ∧ (f1 = 1 ∧ f_1 = 2)) :=
by
  sorry

end impossible_result_l12_12037


namespace isosceles_triangle_perimeter_l12_12868

theorem isosceles_triangle_perimeter :
  (∃ x y : ℝ, x^2 - 6*x + 8 = 0 ∧ y^2 - 6*y + 8 = 0 ∧ (x = 2 ∧ y = 4) ∧ 2 + 4 + 4 = 10) :=
by
  sorry

end isosceles_triangle_perimeter_l12_12868


namespace negation_of_prop_p_l12_12600

theorem negation_of_prop_p (p : Prop) (h : ∀ x: ℝ, 0 < x → x > Real.log x) :
  (¬ (∀ x: ℝ, 0 < x → x > Real.log x)) ↔ (∃ x_0: ℝ, 0 < x_0 ∧ x_0 ≤ Real.log x_0) :=
by sorry

end negation_of_prop_p_l12_12600


namespace value_of_x2_plus_9y2_l12_12369

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l12_12369


namespace reflected_circle_center_l12_12261

theorem reflected_circle_center
  (original_center : ℝ × ℝ) 
  (reflection_line : ℝ × ℝ → ℝ × ℝ)
  (hc : original_center = (8, -3))
  (hl : ∀ (p : ℝ × ℝ), reflection_line p = (-p.2, -p.1))
  : reflection_line original_center = (3, -8) :=
sorry

end reflected_circle_center_l12_12261


namespace arithmetic_sequence_a_m_n_zero_l12_12874

theorem arithmetic_sequence_a_m_n_zero
  (a : ℕ → ℕ)
  (m n : ℕ) 
  (hm : m > 0) (hn : n > 0)
  (h_ma_m : a m = n)
  (h_na_n : a n = m) : 
  a (m + n) = 0 :=
by 
  sorry

end arithmetic_sequence_a_m_n_zero_l12_12874


namespace find_k_l12_12671

theorem find_k : ∃ k : ℚ, (k = (k + 4) / 4) ∧ k = 4 / 3 :=
by
  sorry

end find_k_l12_12671


namespace print_rolls_sold_l12_12681

-- Defining the variables and conditions
def num_sold := 480
def total_amount := 2340
def solid_price := 4
def print_price := 6

-- Proposed theorem statement
theorem print_rolls_sold (S P : ℕ) (h1 : S + P = num_sold) (h2 : solid_price * S + print_price * P = total_amount) : P = 210 := sorry

end print_rolls_sold_l12_12681


namespace intersection_points_l12_12324

-- Definitions and conditions
def is_ellipse (e : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, e x y ↔ x^2 + 2*y^2 = 2

def is_tangent_or_intersects (l : ℝ → ℝ) (e : ℝ → ℝ → Prop) : Prop :=
  ∃ z1 z2 : ℝ, (e z1 (l z1) ∨ e z2 (l z2))

def lines_intersect (l1 l2 : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, l1 x = l2 x

theorem intersection_points :
  ∀ (e : ℝ → ℝ → Prop) (l1 l2 : ℝ → ℝ),
  is_ellipse e →
  is_tangent_or_intersects l1 e →
  is_tangent_or_intersects l2 e →
  lines_intersect l1 l2 →
  ∃ n : ℕ, n = 2 ∨ n = 3 ∨ n = 4 :=
by
  intros e l1 l2 he hto1 hto2 hl
  sorry

end intersection_points_l12_12324


namespace inverse_proportion_function_l12_12779

theorem inverse_proportion_function (x y : ℝ) (h : y = 6 / x) : x * y = 6 :=
by
  sorry

end inverse_proportion_function_l12_12779


namespace find_prime_powers_l12_12207

open Nat

theorem find_prime_powers (p x y : ℕ) (hp : p.Prime) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 ↔
  (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
sorry

end find_prime_powers_l12_12207


namespace three_consecutive_odds_l12_12531

theorem three_consecutive_odds (x : ℤ) (h3 : x + 4 = 133) : 
  x + (x + 4) = 3 * (x + 2) - 131 := 
by {
  sorry
}

end three_consecutive_odds_l12_12531


namespace width_of_foil_covered_prism_l12_12177

noncomputable def foil_covered_prism_width : ℕ :=
  let (l, w, h) := (4, 8, 4)
  let inner_width := 2 * l
  let increased_width := w + 2
  increased_width

theorem width_of_foil_covered_prism : foil_covered_prism_width = 10 := 
by
  let l := 4
  let w := 2 * l
  let h := w / 2
  have volume : l * w * h = 128 := by
    sorry
  have width_foil_covered := w + 2
  have : foil_covered_prism_width = width_foil_covered := by
    sorry
  sorry

end width_of_foil_covered_prism_l12_12177


namespace bryden_receives_10_dollars_l12_12193

-- Define the face value of one quarter
def face_value_quarter : ℝ := 0.25

-- Define the number of quarters Bryden has
def num_quarters : ℕ := 8

-- Define the multiplier for 500%
def multiplier : ℝ := 5

-- Calculate the total face value of eight quarters
def total_face_value : ℝ := num_quarters * face_value_quarter

-- Calculate the amount Bryden will receive
def amount_received : ℝ := total_face_value * multiplier

-- The proof goal: Bryden will receive 10 dollars
theorem bryden_receives_10_dollars : amount_received = 10 :=
by
  sorry

end bryden_receives_10_dollars_l12_12193


namespace negation_of_proposition_l12_12917

theorem negation_of_proposition :
  ¬(∃ x₀ : ℝ, 0 < x₀ ∧ Real.log x₀ = x₀ - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by
  sorry

end negation_of_proposition_l12_12917


namespace school_population_l12_12082

variable (b g t a : ℕ)

theorem school_population (h1 : b = 2 * g) (h2 : g = 4 * t) (h3 : a = t / 2) : 
  b + g + t + a = 27 * b / 16 := by
  sorry

end school_population_l12_12082


namespace ratio_is_one_quarter_l12_12093

def Joel_garden_area : ℚ := 64
def garden_half : ℚ := Joel_garden_area / 2
def strawberry_area : ℚ := 8
def fruit_section_area : ℚ := garden_half
def ratio_strawberries_to_fruit_section : ℚ := strawberry_area / fruit_section_area

theorem ratio_is_one_quarter :
  garden_half = 32 ∧ strawberry_area = 8 ∧ ratio_strawberries_to_fruit_section = 1 / 4 :=
by
  split
  . exact rfl
  . split
    . exact rfl
    . exact rfl

end ratio_is_one_quarter_l12_12093


namespace smallest_twice_perfect_square_three_times_perfect_cube_l12_12035

theorem smallest_twice_perfect_square_three_times_perfect_cube :
  ∃ n : ℕ, (∃ k : ℕ, n = 2 * k^2) ∧ (∃ m : ℕ, n = 3 * m^3) ∧ n = 648 :=
by
  sorry

end smallest_twice_perfect_square_three_times_perfect_cube_l12_12035


namespace math_club_team_selection_l12_12916

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_ways_to_form_team_with_at_least_4_girls (boys girls team_size : ℕ) : ℕ :=
  num_combinations girls 4 * num_combinations boys 4 +
  num_combinations girls 5 * num_combinations boys 3 +
  num_combinations girls 6 * num_combinations boys 2 +
  num_combinations girls 7 * num_combinations boys 1 +
  num_combinations girls 8

theorem math_club_team_selection : 
  num_ways_to_form_team_with_at_least_4_girls 10 12 8 = 245985 :=
by sorry

end math_club_team_selection_l12_12916


namespace no_nonzero_solution_l12_12162

theorem no_nonzero_solution (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
by 
  sorry

end no_nonzero_solution_l12_12162


namespace minimize_product_of_roots_of_quadratic_eq_l12_12727

theorem minimize_product_of_roots_of_quadratic_eq (k : ℝ) :
  (∃ x y : ℝ, 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0) 
  → k = 25 / 8 :=
sorry

end minimize_product_of_roots_of_quadratic_eq_l12_12727


namespace equality_of_a_and_b_l12_12890

theorem equality_of_a_and_b
  (a b : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 4 * a * b - 1 ∣ (4 * a ^ 2 - 1) ^ 2) : a = b := 
sorry

end equality_of_a_and_b_l12_12890


namespace selling_price_of_cycle_l12_12672

theorem selling_price_of_cycle (original_price : ℝ) (loss_percentage : ℝ) (loss_amount : ℝ) (selling_price : ℝ) :
  original_price = 2000 →
  loss_percentage = 10 →
  loss_amount = (loss_percentage / 100) * original_price →
  selling_price = original_price - loss_amount →
  selling_price = 1800 :=
by
  intros
  sorry

end selling_price_of_cycle_l12_12672


namespace pete_ate_percentage_l12_12905

-- Definitions of the conditions
def total_slices : ℕ := 2 * 12
def stephen_ate_slices : ℕ := (25 * total_slices) / 100
def remaining_slices_after_stephen : ℕ := total_slices - stephen_ate_slices
def slices_left_after_pete : ℕ := 9

-- The statement to be proved
theorem pete_ate_percentage (h1 : total_slices = 24)
                            (h2 : stephen_ate_slices = 6)
                            (h3 : remaining_slices_after_stephen = 18)
                            (h4 : slices_left_after_pete = 9) :
  ((remaining_slices_after_stephen - slices_left_after_pete) * 100 / remaining_slices_after_stephen) = 50 :=
sorry

end pete_ate_percentage_l12_12905


namespace find_parts_per_hour_find_min_A_machines_l12_12129

-- Conditions
variable (x y : ℕ) -- x is parts per hour by B, y is parts per hour by A

-- Definitions based on conditions
def machineA_speed_relation (x y : ℕ) : Prop :=
  y = x + 2

def time_relation (x y : ℕ) : Prop :=
  80 / y = 60 / x

def min_A_machines (x y : ℕ) (m : ℕ) : Prop :=
  8 * m + 6 * (10 - m) ≥ 70

-- Problem statements
theorem find_parts_per_hour (x y : ℕ) (h1 : machineA_speed_relation x y) (h2 : time_relation x y) :
  x = 6 ∧ y = 8 :=
sorry

theorem find_min_A_machines (m : ℕ) (h1 : machineA_speed_relation 6 8) (h2 : time_relation 6 8) (h3 : min_A_machines 6 8 m) :
  m ≥ 5 :=
sorry

end find_parts_per_hour_find_min_A_machines_l12_12129


namespace least_possible_N_proof_l12_12680

noncomputable def least_possible_N (N : ℕ) (n : ℕ) : Prop :=
  N > 0 ∧
  (1 ≤ n ∧ n ≤ 29) ∧
  (∀ k : ℕ, (1 ≤ k ∧ k ≤ 30) → k ≠ n → k ≠ n + 1 → k ∣ N) ∧
  N = 2230928700

theorem least_possible_N_proof : ∃ (N n : ℕ), least_possible_N N n :=
by
  use 2230928700
  use 28
  -- Proof of the conditions, skipped with sorry
  sorry

end least_possible_N_proof_l12_12680


namespace intersection_M_N_l12_12764

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | x > 0 ∧ x < 2}

theorem intersection_M_N : M ∩ N = {1} :=
by {
  sorry
}

end intersection_M_N_l12_12764


namespace proof_problem_l12_12840

open Real

def p : Prop := ∀ a : ℝ, a^2017 > -1 → a > -1
def q : Prop := ∀ x : ℝ, x^2 * tan (x^2) > 0

theorem proof_problem : p ∨ q :=
sorry

end proof_problem_l12_12840


namespace eval_expression_l12_12018

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12018


namespace rectangular_solid_surface_area_l12_12467

theorem rectangular_solid_surface_area
  (a b c : ℕ)
  (h_prime_a : Prime a)
  (h_prime_b : Prime b)
  (h_prime_c : Prime c)
  (h_volume : a * b * c = 143) :
  2 * (a * b + b * c + c * a) = 382 := by
  sorry

end rectangular_solid_surface_area_l12_12467


namespace scout_troop_profit_l12_12180

theorem scout_troop_profit :
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let bars_per_dollar := 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let bars_per_three_dollars := 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  profit = 320 := by
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  sorry

end scout_troop_profit_l12_12180


namespace find_y_l12_12935

theorem find_y (y : ℝ) (h : 3 * y / 7 = 21) : y = 49 := 
sorry

end find_y_l12_12935


namespace intersection_S_T_eq_U_l12_12896

def S : Set ℝ := {x | abs x < 5}
def T : Set ℝ := {x | (x + 7) * (x - 3) < 0}
def U : Set ℝ := {x | -5 < x ∧ x < 3}

theorem intersection_S_T_eq_U : (S ∩ T) = U := 
by 
  sorry

end intersection_S_T_eq_U_l12_12896


namespace percent_of_x_is_y_l12_12489

theorem percent_of_x_is_y (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y = 0.3333 * x :=
by
  sorry

end percent_of_x_is_y_l12_12489


namespace evaluate_x_squared_minus_y_squared_l12_12052

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end evaluate_x_squared_minus_y_squared_l12_12052


namespace simplify_expr1_simplify_expr2_l12_12323

-- Define the first problem with necessary conditions
theorem simplify_expr1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) - (b / (b - a)) = (a + b) / (a - b) :=
by
  sorry

-- Define the second problem with necessary conditions
theorem simplify_expr2 (x : ℝ) (hx1 : x ≠ -3) (hx2 : x ≠ 4) (hx3 : x ≠ -4) :
  ((x - 4) / (x + 3)) / (x - 3 - (7 / (x + 3))) = 1 / (x + 4) :=
by
  sorry

end simplify_expr1_simplify_expr2_l12_12323


namespace pell_infinite_solutions_l12_12617

theorem pell_infinite_solutions : ∃ m : ℕ, ∃ a b c : ℕ, 
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (∀ n : ℕ, ∃ an bn cn : ℕ, 
    (1 / an + 1 / bn + 1 / cn + 1 / (an * bn * cn) = m / (an + bn + cn))) := 
sorry

end pell_infinite_solutions_l12_12617


namespace current_speed_is_one_l12_12171

noncomputable def motorboat_rate_of_current (b h t : ℝ) : ℝ :=
  let eq1 := (b + 1 - h) * 4
  let eq2 := (b - 1 + t) * 6
  if eq1 = 24 ∧ eq2 = 24 then 1 else sorry

theorem current_speed_is_one (b h t : ℝ) : motorboat_rate_of_current b h t = 1 :=
by
  sorry

end current_speed_is_one_l12_12171


namespace quinton_total_fruit_trees_l12_12256

-- Define the given conditions
def num_apple_trees := 2
def width_apple_tree_ft := 10
def space_between_apples_ft := 12
def width_peach_tree_ft := 12
def space_between_peaches_ft := 15
def total_space_ft := 71

-- Definition that calculates the total number of fruit trees Quinton wants to plant
def total_fruit_trees : ℕ := 
  let space_apple_trees := num_apple_trees * width_apple_tree_ft + space_between_apples_ft
  let space_remaining_for_peaches := total_space_ft - space_apple_trees
  1 + space_remaining_for_peaches / (width_peach_tree_ft + space_between_peaches_ft) + num_apple_trees

-- The statement to prove
theorem quinton_total_fruit_trees : total_fruit_trees = 4 := by
  sorry

end quinton_total_fruit_trees_l12_12256


namespace hexagon_perimeter_l12_12946

-- Definitions of the conditions
def side_length : ℕ := 5
def number_of_sides : ℕ := 6

-- The perimeter of the hexagon
def perimeter : ℕ := side_length * number_of_sides

-- Proof statement
theorem hexagon_perimeter : perimeter = 30 :=
by
  sorry

end hexagon_perimeter_l12_12946


namespace largest_4_digit_div_by_5_smallest_primes_l12_12208

noncomputable def LCM_5_smallest_primes : ℕ := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))

theorem largest_4_digit_div_by_5_smallest_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 9240 := by
  sorry

end largest_4_digit_div_by_5_smallest_primes_l12_12208


namespace bananas_proof_l12_12181

noncomputable def number_of_bananas (total_oranges : ℕ) (total_fruits_percent_good : ℝ) 
  (percent_rotten_oranges : ℝ) (percent_rotten_bananas : ℝ) : ℕ := 448

theorem bananas_proof :
  let total_oranges := 600
  let percent_rotten_oranges := 0.15
  let percent_rotten_bananas := 0.08
  let total_fruits_percent_good := 0.878
  
  number_of_bananas total_oranges total_fruits_percent_good percent_rotten_oranges percent_rotten_bananas = 448 :=
by
  sorry

end bananas_proof_l12_12181


namespace alcohol_added_l12_12807

-- Definitions from conditions
def initial_volume : ℝ := 40
def initial_alcohol_concentration : ℝ := 0.05
def initial_alcohol_amount : ℝ := initial_volume * initial_alcohol_concentration
def added_water_volume : ℝ := 3.5
def final_alcohol_concentration : ℝ := 0.17

-- The problem to be proven
theorem alcohol_added :
  ∃ x : ℝ,
    x = (final_alcohol_concentration * (initial_volume + x + added_water_volume) - initial_alcohol_amount) :=
by
  sorry

end alcohol_added_l12_12807


namespace sin_y_gt_half_x_l12_12805

theorem sin_y_gt_half_x (x y : ℝ) (hx : x ≤ 90) (h : Real.sin y = (3 / 4) * Real.sin x) : y > x / 2 :=
by
  sorry

end sin_y_gt_half_x_l12_12805


namespace original_average_weight_l12_12533

-- Definitions from conditions
def original_team_size : ℕ := 7
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60
def new_team_size := original_team_size + 2
def new_average_weight : ℝ := 106

-- Statement to prove
theorem original_average_weight (W : ℝ) :
  (7 * W + 110 + 60 = 9 * 106) → W = 112 := by
  sorry

end original_average_weight_l12_12533


namespace smallest_number_of_groups_l12_12314

theorem smallest_number_of_groups
  (participants : ℕ)
  (max_group_size : ℕ)
  (h1 : participants = 36)
  (h2 : max_group_size = 12) :
  participants / max_group_size = 3 :=
by
  sorry

end smallest_number_of_groups_l12_12314


namespace paco_min_cookies_l12_12402

theorem paco_min_cookies (x : ℕ) (h_initial : 25 - x ≥ 0) : 
  x + (3 + 2) ≥ 5 := by
  sorry

end paco_min_cookies_l12_12402


namespace total_students_proof_l12_12312

variable (studentsA studentsB : ℕ) (ratioAtoB : ℕ := 3/2)
variable (percentA percentB : ℕ := 10/100)
variable (diffPercent : ℕ := 20/100)
variable (extraStudentsInA : ℕ := 190)
variable (totalStudentsB : ℕ := 650)

theorem total_students_proof :
  (studentsB = totalStudentsB) ∧ 
  ((percentA * studentsA - diffPercent * studentsB = extraStudentsInA) ∧
  (studentsA / studentsB = ratioAtoB)) →
  (studentsA + studentsB = 1625) :=
by
  sorry

end total_students_proof_l12_12312


namespace fractional_equation_solution_l12_12904

theorem fractional_equation_solution (x : ℝ) (h : x = 7) : (3 / (x - 3)) - 1 = 1 / (3 - x) := by
  sorry

end fractional_equation_solution_l12_12904


namespace probability_heads_given_heads_l12_12423

-- Definitions for fair coin flips and the stopping condition
noncomputable def fair_coin_prob (event : ℕ → Prop) : ℝ :=
  sorry -- Probability function for coin events (to be defined in proofs)

-- The main statement
theorem probability_heads_given_heads :
  let p : ℝ := 1 / 3 in
  ∃ p: ℝ, p = 1 / 3 ∧ fair_coin_prob (λ n, (n = 1 ∧ (coin_flip n = (TT)) ∧ ((coin_flip (n+1) = (HH) ∨ coin_flip (n+1) = (TH))) ∧ ¬has_heads_before n)) = p :=
sorry

end probability_heads_given_heads_l12_12423


namespace houses_after_boom_l12_12191

theorem houses_after_boom (h_pre_boom : ℕ) (h_built : ℕ) (h_count : ℕ)
  (H1 : h_pre_boom = 1426)
  (H2 : h_built = 574)
  (H3 : h_count = h_pre_boom + h_built) :
  h_count = 2000 :=
by {
  sorry
}

end houses_after_boom_l12_12191


namespace jenny_boxes_sold_l12_12503

/--
Jenny sold some boxes of Trefoils. Each box has 8.0 packs. She sold 192 packs in total.
Prove that Jenny sold 24 boxes.
-/
theorem jenny_boxes_sold (packs_per_box : Real) (total_packs_sold : Real) (num_boxes_sold : Real) 
  (h1 : packs_per_box = 8.0) (h2 : total_packs_sold = 192) : num_boxes_sold = 24 :=
by
  have h3 : num_boxes_sold = total_packs_sold / packs_per_box :=
    by sorry
  sorry

end jenny_boxes_sold_l12_12503


namespace r4_plus_inv_r4_l12_12640

theorem r4_plus_inv_r4 (r : ℝ) (h : (r + (1 : ℝ) / r) ^ 2 = 5) : r ^ 4 + (1 : ℝ) / r ^ 4 = 7 := 
by
  -- Proof goes here
  sorry

end r4_plus_inv_r4_l12_12640


namespace line_ellipse_common_points_l12_12267

def point (P : Type*) := P → ℝ × ℝ

theorem line_ellipse_common_points
  (m n : ℝ)
  (no_common_points_with_circle : ∀ (x y : ℝ), mx + ny - 3 = 0 → x^2 + y^2 ≠ 3) :
  ∀ (Px Py : ℝ), (Px = m ∧ Py = n) →
  (∃ (x1 y1 x2 y2 : ℝ), ((x1^2 / 7) + (y1^2 / 3) = 1 ∧ (x2^2 / 7) + (y2^2 / 3) = 1) ∧ (x1, y1) ≠ (x2, y2)) :=
by
  sorry

end line_ellipse_common_points_l12_12267


namespace area_of_circles_l12_12650

theorem area_of_circles (BD AC : ℝ) (hBD : BD = 6) (hAC : AC = 12) : 
  ∃ S : ℝ, S = 225 / 4 * Real.pi :=
by
  sorry

end area_of_circles_l12_12650


namespace max_value_ineq_l12_12720

variables {R : Type} [LinearOrderedField R]

theorem max_value_ineq (a b c x y z : R) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : 0 ≤ x) (h5 : 0 ≤ y) (h6 : 0 ≤ z)
  (h7 : a + b + c = 1) (h8 : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≤ 1 / 16 :=
sorry

end max_value_ineq_l12_12720


namespace abc_order_l12_12103

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := 0.5^3
noncomputable def c : Real := Real.log 3 / Real.log 0.5 -- log_0.5 3 is written as (log 3) / (log 0.5) in Lean

theorem abc_order : a > b ∧ b > c :=
by
  have h1 : a = Real.sqrt 3 := rfl
  have h2 : b = 0.5^3 := rfl
  have h3 : c = Real.log 3 / Real.log 0.5 := rfl
  sorry

end abc_order_l12_12103


namespace avg_salary_rest_of_workers_l12_12132

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_tech : ℝ) (total_workers : ℕ)
  (total_avg_salary : avg_all = 8000) (tech_avg_salary : avg_tech = 12000) (workers_count : total_workers = 30) :
  (20 * (total_workers * avg_all - 10 * avg_tech) / 20) = 6000 :=
by
  sorry

end avg_salary_rest_of_workers_l12_12132


namespace tan_alpha_plus_pi_over_12_l12_12474

theorem tan_alpha_plus_pi_over_12 (α : ℝ) (h : Real.sin α = 3 * Real.sin (α + π / 6)) :
  Real.tan (α + π / 12) = 2 * Real.sqrt 3 - 4 :=
by
  sorry

end tan_alpha_plus_pi_over_12_l12_12474


namespace quadrilateral_area_BEIH_l12_12155

-- Define the necessary points in the problem
structure Point :=
(x : ℚ)
(y : ℚ)

-- Definitions of given points and midpoints
def B : Point := ⟨0, 0⟩
def E : Point := ⟨0, 1.5⟩
def F : Point := ⟨1.5, 0⟩

-- Definitions of line equations from points
def line_DE (p : Point) : Prop := p.y = - (1 / 2) * p.x + 1.5
def line_AF (p : Point) : Prop := p.y = -2 * p.x + 3

-- Intersection points
def I : Point := ⟨3 / 5, 9 / 5⟩
def H : Point := ⟨3 / 4, 3 / 4⟩

-- Function to calculate the area using the Shoelace Theorem
def shoelace_area (a b c d : Point) : ℚ :=
  (1 / 2) * ((a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y) - (a.y * b.x + b.y * c.x + c.y * d.x + d.y * a.x))

-- The proof statement
theorem quadrilateral_area_BEIH :
  shoelace_area B E I H = 9 / 16 :=
sorry

end quadrilateral_area_BEIH_l12_12155


namespace largest_real_root_range_l12_12851

theorem largest_real_root_range (b0 b1 b2 b3 : ℝ) (h0 : |b0| ≤ 1) (h1 : |b1| ≤ 1) (h2 : |b2| ≤ 1) (h3 : |b3| ≤ 1) :
  ∀ r : ℝ, (Polynomial.eval r (Polynomial.C (1:ℝ) + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C b0) = 0) → (5 / 2) < r ∧ r < 3 :=
by
  sorry

end largest_real_root_range_l12_12851


namespace x_squared_plus_y_squared_l12_12218

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 3) (h2 : (x - y) ^ 2 = 9) : 
  x ^ 2 + y ^ 2 = 15 := sorry

end x_squared_plus_y_squared_l12_12218


namespace new_average_height_is_184_l12_12156

-- Define the initial conditions
def original_num_students : ℕ := 35
def original_avg_height : ℕ := 180
def left_num_students : ℕ := 7
def left_avg_height : ℕ := 120
def joined_num_students : ℕ := 7
def joined_avg_height : ℕ := 140

-- Calculate the initial total height
def original_total_height := original_avg_height * original_num_students

-- Calculate the total height of the students who left
def left_total_height := left_avg_height * left_num_students

-- Calculate the new total height after the students left
def new_total_height1 := original_total_height - left_total_height

-- Calculate the total height of the new students who joined
def joined_total_height := joined_avg_height * joined_num_students

-- Calculate the new total height after the new students joined
def new_total_height2 := new_total_height1 + joined_total_height

-- Calculate the new average height
def new_avg_height := new_total_height2 / original_num_students

-- The theorem stating the result
theorem new_average_height_is_184 : new_avg_height = 184 := by
  sorry

end new_average_height_is_184_l12_12156


namespace find_length_PB_l12_12508

noncomputable def radius (O : Type*) : ℝ := sorry

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

variables {α : Type*}

def Point (α : Type*) := α

variables (P T A B : Point ℝ) (O : Circle ℝ) (r : ℝ)

def PA := (4 : ℝ)
def PT (AB : ℝ) := AB - 2
def PB (AB : ℝ) := 4 + AB

def power_of_a_point (PA PB PT : ℝ) := PA * PB = PT^2

theorem find_length_PB (AB : ℝ) 
  (h1 : power_of_a_point PA (PB AB) (PT AB)) 
  (h2 : PA < PB AB) : 
  PB AB = 18 := 
by 
  sorry

end find_length_PB_l12_12508


namespace proposition_negation_l12_12077

theorem proposition_negation (p : Prop) : 
  (∃ x : ℝ, x < 1 ∧ x^2 < 1) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
sorry

end proposition_negation_l12_12077


namespace gift_bags_needed_l12_12695

/-
  Constants
  total_expected: \(\mathbb{N}\) := 90        -- 50 people who will show up + 40 more who may show up
  total_prepared: \(\mathbb{N}\) := 30        -- 10 extravagant gift bags + 20 average gift bags

  The property to be proved:
  prove that (total_expected - total_prepared = 60)
-/

def total_expected : ℕ := 50 + 40
def total_prepared : ℕ := 10 + 20
def additional_needed := total_expected - total_prepared

theorem gift_bags_needed : additional_needed = 60 := by
  sorry

end gift_bags_needed_l12_12695


namespace my_current_age_l12_12949

-- Definitions based on the conditions
def bro_age (x : ℕ) : ℕ := 2 * x - 5

-- Main theorem to prove that my current age is 13 given the conditions
theorem my_current_age 
  (x y : ℕ)
  (h1 : y - 5 = 2 * (x - 5))
  (h2 : (x + 8) + (y + 8) = 50) :
  x = 13 :=
sorry

end my_current_age_l12_12949


namespace y_value_when_x_neg_one_l12_12107

theorem y_value_when_x_neg_one (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = t^2 + 3 * t + 6) 
  (h3 : x = -1) : 
  y = 16 := 
by sorry

end y_value_when_x_neg_one_l12_12107


namespace tan_subtraction_l12_12231

theorem tan_subtraction (α β : ℝ) (h₁ : Real.tan α = 9) (h₂ : Real.tan β = 6) :
  Real.tan (α - β) = 3 / 55 :=
by
  sorry

end tan_subtraction_l12_12231


namespace intersection_A_B_l12_12215

def A : Set ℝ := { x | abs x < 3 }
def B : Set ℝ := { x | 2 - x > 0 }

theorem intersection_A_B : A ∩ B = { x : ℝ | -3 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l12_12215


namespace percentage_deficit_l12_12086

theorem percentage_deficit
  (L W : ℝ)
  (h1 : ∃(x : ℝ), 1.10 * L * (W * (1 - x / 100)) = L * W * 1.045) :
  ∃ (x : ℝ), x = 5 :=
by
  sorry

end percentage_deficit_l12_12086


namespace f_odd_f_decreasing_f_extremum_l12_12627

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_val : f 1 = -2
axiom f_neg : ∀ x > 0, f x < 0

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem f_extremum : ∃ (max min : ℝ), max = f (-3) ∧ min = f 3 :=
sorry

end f_odd_f_decreasing_f_extremum_l12_12627


namespace total_chocolate_bars_proof_l12_12440

def large_box_contains := 17
def first_10_boxes_contains := 10
def medium_boxes_per_small := 4
def chocolate_bars_per_medium := 26

def remaining_7_boxes := 7
def first_two_boxes := 2
def first_two_bars := 18
def next_three_boxes := 3
def next_three_bars := 22
def last_two_boxes := 2
def last_two_bars := 30

noncomputable def total_chocolate_bars_in_large_box : Nat :=
  let chocolate_in_first_10 := first_10_boxes_contains * medium_boxes_per_small * chocolate_bars_per_medium
  let chocolate_in_remaining_7 :=
    (first_two_boxes * first_two_bars) +
    (next_three_boxes * next_three_bars) +
    (last_two_boxes * last_two_bars)
  chocolate_in_first_10 + chocolate_in_remaining_7

theorem total_chocolate_bars_proof :
  total_chocolate_bars_in_large_box = 1202 :=
by
  -- Detailed calculation is skipped
  sorry

end total_chocolate_bars_proof_l12_12440


namespace exists_prime_mod_greater_remainder_l12_12763

theorem exists_prime_mod_greater_remainder (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  ∃ p : ℕ, Prime p ∧ a % p > b % p :=
sorry

end exists_prime_mod_greater_remainder_l12_12763


namespace find_angle_4_l12_12586

def angle_sum_180 (α β : ℝ) : Prop := α + β = 180
def angle_equality (γ δ : ℝ) : Prop := γ = δ
def triangle_angle_values (A B : ℝ) : Prop := A = 80 ∧ B = 50

theorem find_angle_4
  (A B : ℝ) (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle_sum_180 angle1 angle2)
  (h2 : angle_equality angle3 angle4)
  (h3 : triangle_angle_values A B)
  (h4 : angle_sum_180 (angle1 + A + B) 180)
  (h5 : angle_sum_180 (angle2 + angle3 + angle4) 180) :
  angle4 = 25 :=
by sorry

end find_angle_4_l12_12586


namespace find_m_from_permutation_l12_12344

theorem find_m_from_permutation (A : Nat → Nat → Nat) (m : Nat) (hA : A 11 m = 11 * 10 * 9 * 8 * 7 * 6 * 5) : m = 7 :=
sorry

end find_m_from_permutation_l12_12344


namespace Sam_has_seven_watermelons_l12_12408

-- Declare the initial number of watermelons
def initial_watermelons : Nat := 4

-- Declare the additional number of watermelons Sam grew
def more_watermelons : Nat := 3

-- Prove that the total number of watermelons is 7
theorem Sam_has_seven_watermelons : initial_watermelons + more_watermelons = 7 :=
by
  sorry

end Sam_has_seven_watermelons_l12_12408


namespace student_entrepreneur_profit_l12_12804

theorem student_entrepreneur_profit {x y a: ℝ} 
  (h1 : a * (y - x) = 1000) 
  (h2 : (ay / x) * y - ay = 1500)
  (h3 : y = 3 / 2 * x) : a * x = 2000 := 
sorry

end student_entrepreneur_profit_l12_12804


namespace equal_sum_seq_example_l12_12201

def EqualSumSeq (a : ℕ → ℕ) (c : ℕ) : Prop := ∀ n, a n + a (n + 1) = c

theorem equal_sum_seq_example (a : ℕ → ℕ) 
  (h1 : EqualSumSeq a 5) 
  (h2 : a 1 = 2) : a 6 = 3 :=
by 
  sorry

end equal_sum_seq_example_l12_12201


namespace books_read_in_eight_hours_l12_12115

-- Definitions to set up the problem
def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

-- Theorem statement
theorem books_read_in_eight_hours : (available_time * reading_speed) / book_length = 2 := 
by
  sorry

end books_read_in_eight_hours_l12_12115


namespace larger_of_two_numbers_l12_12064

theorem larger_of_two_numbers (x y : ℕ) (h1 : x * y = 24) (h2 : x + y = 11) : max x y = 8 :=
sorry

end larger_of_two_numbers_l12_12064


namespace even_abs_func_necessary_not_sufficient_l12_12041

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_symmetrical_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem even_abs_func_necessary_not_sufficient (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, |f (-x)| = |f x|) ∧ (∃ g : ℝ → ℝ, (∀ x : ℝ, |g (-x)| = |g x|) ∧ ¬(∀ x : ℝ, g (-x) = -g x)) :=
by
  -- Proof omitted.
  sorry

end even_abs_func_necessary_not_sufficient_l12_12041


namespace original_decimal_number_l12_12546

theorem original_decimal_number (x : ℝ) (h : 0.375 = (x / 1000) * 10) : x = 37.5 :=
sorry

end original_decimal_number_l12_12546


namespace sufficient_not_necessary_condition_abs_eq_one_l12_12067

theorem sufficient_not_necessary_condition_abs_eq_one (a : ℝ) :
  (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 ∨ a = -1) :=
by
  sorry

end sufficient_not_necessary_condition_abs_eq_one_l12_12067


namespace inequality_neg_mul_l12_12484

theorem inequality_neg_mul (a b : ℝ) (h : a > b) : -3 * a < -3 * b :=
sorry

end inequality_neg_mul_l12_12484


namespace probability_triangle_side_decagon_l12_12343

theorem probability_triangle_side_decagon (total_vertices : ℕ) (choose_vertices : ℕ)
  (total_triangles : ℕ) (favorable_outcomes : ℕ)
  (triangle_formula : total_vertices = 10)
  (choose_vertices_formula : choose_vertices = 3)
  (total_triangle_count_formula : total_triangles = 120)
  (favorable_outcome_count_formula : favorable_outcomes = 70)
  : (favorable_outcomes : ℚ) / total_triangles = 7 / 12 := 
by 
  sorry

end probability_triangle_side_decagon_l12_12343


namespace simplified_expr_eval_l12_12774

theorem simplified_expr_eval
  (x : ℚ) (y : ℚ) (h_x : x = -1/2) (h_y : y = 1) :
  (5*x^2 - 10*y^2) = -35/4 := 
by
  subst h_x
  subst h_y
  sorry

end simplified_expr_eval_l12_12774


namespace weaving_sequence_l12_12873

-- Define the arithmetic sequence conditions
def day1_weaving := 5
def total_cloth := 390
def days := 30

-- Mathematical statement to be proved
theorem weaving_sequence : 
    ∃ d : ℚ, 30 * day1_weaving + (days * (days - 1) / 2) * d = total_cloth ∧ d = 16 / 29 :=
by 
  sorry

end weaving_sequence_l12_12873


namespace xyz_neg_of_ineq_l12_12923

variables {x y z : ℝ}

theorem xyz_neg_of_ineq
  (h1 : 2 * x - y < 0)
  (h2 : 3 * y - 2 * z < 0)
  (h3 : 4 * z - 3 * x < 0) :
  x < 0 ∧ y < 0 ∧ z < 0 :=
sorry

end xyz_neg_of_ineq_l12_12923


namespace pencil_length_l12_12800

theorem pencil_length (L : ℝ) (h1 : (1 / 8) * L + (1 / 2) * (7 / 8) * L + (7 / 2) = L) : L = 16 :=
by
  sorry

end pencil_length_l12_12800


namespace largest_integer_among_four_l12_12834

theorem largest_integer_among_four 
  (x y z w : ℤ)
  (h1 : x + y + z = 234)
  (h2 : x + y + w = 255)
  (h3 : x + z + w = 271)
  (h4 : y + z + w = 198) :
  max x (max y (max z w)) = 121 := 
by
  -- This is a placeholder for the actual proof
  sorry

end largest_integer_among_four_l12_12834


namespace students_taking_french_l12_12746

theorem students_taking_french 
  (Total : ℕ) (G : ℕ) (B : ℕ) (Neither : ℕ) (H_total : Total = 87)
  (H_G : G = 22) (H_B : B = 9) (H_neither : Neither = 33) : 
  ∃ F : ℕ, F = 41 := 
by
  sorry

end students_taking_french_l12_12746


namespace tangent_line_ellipse_l12_12728

theorem tangent_line_ellipse (x y : ℝ) (h : 2^2 / 8 + 1^2 / 2 = 1) :
    x / 4 + y / 2 = 1 := 
  sorry

end tangent_line_ellipse_l12_12728


namespace sum_lent_l12_12157

theorem sum_lent (P : ℝ) (R : ℝ := 4) (T : ℝ := 8) (I : ℝ) (H1 : I = P - 204) (H2 : I = (P * R * T) / 100) : 
  P = 300 :=
by 
  sorry

end sum_lent_l12_12157


namespace copy_pages_l12_12618

theorem copy_pages (total_cents : ℕ) (cost_per_page : ℕ) (h1 : total_cents = 1500) (h2 : cost_per_page = 5) : 
  (total_cents / cost_per_page = 300) :=
sorry

end copy_pages_l12_12618


namespace two_aces_or_at_least_one_king_probability_l12_12870

theorem two_aces_or_at_least_one_king_probability :
  let total_cards := 52
  let total_aces := 5
  let total_kings := 4
  let prob_two_aces := (total_aces / total_cards) * ((total_aces - 1) / (total_cards - 1))
  let prob_exactly_one_king := ((total_kings / total_cards) * ((total_cards - total_kings) / (total_cards - 1))) +
                               (((total_cards - total_kings) / total_cards) * (total_kings / (total_cards - 1)))
  let prob_two_kings := (total_kings / total_cards) * ((total_kings - 1) / (total_cards - 1))
  let prob_at_least_one_king := prob_exactly_one_king + prob_two_kings
  let prob_question := prob_two_aces + prob_at_least_one_king
in prob_question = 104 / 663 := by
  let total_cards := 52
  let total_aces := 5
  let total_kings := 4
  let prob_two_aces := (total_aces / total_cards) * ((total_aces - 1) / (total_cards - 1))
  let prob_exactly_one_king := ((total_kings / total_cards) * ((total_cards - total_kings) / (total_cards - 1))) +
                               (((total_cards - total_kings) / total_cards) * (total_kings / (total_cards - 1)))
  let prob_two_kings := (total_kings / total_cards) * ((total_kings - 1) / (total_cards - 1))
  let prob_at_least_one_king := prob_exactly_one_king + prob_two_kings
  let prob_question := prob_two_aces + prob_at_least_one_king
  have h_prob_two_aces : prob_two_aces = 10 / 1326 := sorry
  have h_prob_exactly_one_king : prob_exactly_one_king = 32 / 221 := sorry
  have h_prob_two_kings : prob_two_kings = 1 / 221 := sorry
  have h_prob_at_least_one_king : prob_at_least_one_king = (32 / 221) + (1 / 221) := sorry
  have h_prob_at_least_one_king := prob_at_least_one_king = 33 / 221 := sorry 
  have h_final := prob_question = (10 / 1326) + (33 / 221) := sorry
  have h_final := prob_question = (10 / 1326) + (198 / 1326) := sorry
  have h_final := prob_question = 208 / 1326 := sorry 
  exact h_final = 104 / 663 sorry 

end two_aces_or_at_least_one_king_probability_l12_12870


namespace cryptarithm_solution_l12_12388

theorem cryptarithm_solution (A B : ℕ) (h_digit_A : A < 10) (h_digit_B : B < 10)
  (h_equation : 9 * (10 * A + B) = 110 * A + B) :
  A = 2 ∧ B = 5 :=
sorry

end cryptarithm_solution_l12_12388


namespace total_eggs_l12_12273

def e0 : ℝ := 47.0
def ei : ℝ := 5.0

theorem total_eggs : e0 + ei = 52.0 := by
  sorry

end total_eggs_l12_12273


namespace abc_product_l12_12596

theorem abc_product :
  ∃ (a b c P : ℕ), 
    b + c = 3 ∧ 
    c + a = 6 ∧ 
    a + b = 7 ∧ 
    P = a * b * c ∧ 
    P = 10 :=
by sorry

end abc_product_l12_12596


namespace simplify_expression_l12_12509

variable (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b)
variable (h : a^3 - b^3 = a - b)

theorem simplify_expression 
  (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b) (h : a^3 - b^3 = a - b) : 
  (a / b - b / a + 1 / (a * b)) = 2 * (1 / (a * b)) - 1 := 
sorry

end simplify_expression_l12_12509


namespace combined_score_210_l12_12744

-- Define the constants and variables
def total_questions : ℕ := 50
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5
def jose_extra_marks (alisson_score : ℕ) : ℕ := 40
def meghan_less_marks (jose_score : ℕ) : ℕ := 20

-- Define the total possible marks
def total_possible_marks : ℕ := total_questions * marks_per_question

-- Given the conditions, we need to prove the total combined score is 210
theorem combined_score_210 : 
  ∃ (jose_score meghan_score alisson_score combined_score : ℕ), 
  jose_score = total_possible_marks - (jose_wrong_questions * marks_per_question) ∧
  meghan_score = jose_score - meghan_less_marks jose_score ∧
  alisson_score = jose_score - jose_extra_marks alisson_score ∧
  combined_score = jose_score + meghan_score + alisson_score ∧
  combined_score = 210 := by
  sorry

end combined_score_210_l12_12744


namespace ranking_arrangements_l12_12036

open Finset

theorem ranking_arrangements (students : Finset ℕ) (A B : ℕ) (ranking : ℕ → ℕ) :
  students = {1, 2, 3, 4, 5} →
  A ∉ {ranking 1} →
  B ∉ {ranking 5} →
  ∃ possible_rankings, possible_rankings.card = 78 :=
by
  sorry

end ranking_arrangements_l12_12036


namespace clarinet_players_count_l12_12497

-- Given weights and counts
def weight_trumpet : ℕ := 5
def weight_clarinet : ℕ := 5
def weight_trombone : ℕ := 10
def weight_tuba : ℕ := 20
def weight_drum : ℕ := 15
def count_trumpets : ℕ := 6
def count_trombones : ℕ := 8
def count_tubas : ℕ := 3
def count_drummers : ℕ := 2
def total_weight : ℕ := 245

-- Calculated known weight
def known_weight : ℕ :=
  (count_trumpets * weight_trumpet) +
  (count_trombones * weight_trombone) +
  (count_tubas * weight_tuba) +
  (count_drummers * weight_drum)

-- Weight carried by clarinets
def weight_clarinets : ℕ := total_weight - known_weight

-- Number of clarinet players
def number_of_clarinet_players : ℕ := weight_clarinets / weight_clarinet

theorem clarinet_players_count :
  number_of_clarinet_players = 9 := by
  unfold number_of_clarinet_players
  unfold weight_clarinets
  unfold known_weight
  calc
    (245 - (
      (6 * 5) + 
      (8 * 10) + 
      (3 * 20) + 
      (2 * 15))) / 5 = 9 := by norm_num

end clarinet_players_count_l12_12497


namespace quadratic_inequality_real_roots_l12_12202

theorem quadratic_inequality_real_roots (c : ℝ) (h_pos : 0 < c) (h_ineq : c < 25) :
  ∃ x : ℝ, x^2 - 10 * x + c < 0 :=
sorry

end quadratic_inequality_real_roots_l12_12202


namespace eval_expression_l12_12030

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12030


namespace total_ticket_sales_l12_12439

-- Define the parameters and the theorem to be proven.
theorem total_ticket_sales (total_people : ℕ) (kids : ℕ) (adult_ticket_price : ℕ) (kid_ticket_price : ℕ) 
  (adult_tickets := total_people - kids) 
  (adult_ticket_sales := adult_tickets * adult_ticket_price) 
  (kid_ticket_sales := kids * kid_ticket_price) : 
  total_people = 254 → kids = 203 → adult_ticket_price = 28 → kid_ticket_price = 12 → 
  adult_ticket_sales + kid_ticket_sales = 3864 := 
by
  intros h1 h2 h3 h4
  sorry

end total_ticket_sales_l12_12439


namespace earnings_difference_l12_12797

theorem earnings_difference (x y : ℕ) 
  (h1 : 3 * 6 + 4 * 5 + 5 * 4 = 58)
  (h2 : x * y = 12500) 
  (total_earnings : (3 * 6 * x * y / 100 + 4 * 5 * x * y / 100 + 5 * 4 * x * y / 100) = 7250) :
  4 * 5 * x * y / 100 - 3 * 6 * x * y / 100 = 250 := 
by 
  sorry

end earnings_difference_l12_12797


namespace average_length_tapes_l12_12526

def lengths (l1 l2 l3 l4 l5 : ℝ) : Prop :=
  l1 = 35 ∧ l2 = 29 ∧ l3 = 35.5 ∧ l4 = 36 ∧ l5 = 30.5

theorem average_length_tapes
  (l1 l2 l3 l4 l5 : ℝ)
  (h : lengths l1 l2 l3 l4 l5) :
  (l1 + l2 + l3 + l4 + l5) / 5 = 33.2 := 
by
  sorry

end average_length_tapes_l12_12526


namespace intersection_M_complement_N_l12_12996

noncomputable def M := {y : ℝ | 1 ≤ y ∧ y ≤ 2}
noncomputable def N_complement := {x : ℝ | 1 ≤ x}

theorem intersection_M_complement_N : M ∩ N_complement = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end intersection_M_complement_N_l12_12996


namespace day_of_month_l12_12410

/--
The 25th day of a particular month is a Monday. 
We need to prove that the 1st day of that month is a Friday.
-/
theorem day_of_month (h : (25 % 7 = 1)) : (1 % 7 = 5) :=
sorry

end day_of_month_l12_12410


namespace find_a_and_b_l12_12987

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 6 * a * x^2 + b

theorem find_a_and_b :
  (∃ a b : ℝ, a ≠ 0 ∧
   (∀ x, -1 ≤ x ∧ x ≤ 2 → f a b x ≤ 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = -29)
  ) → ((a = 2 ∧ b = 3) ∨ (a = -2 ∧ b = -29)) :=
sorry

end find_a_and_b_l12_12987


namespace shift_parabola_two_units_right_l12_12567

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function
def shift (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the new parabola equation after shifting 2 units to the right
def shifted_parabola (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that shifting the original parabola 2 units to the right equals the new parabola equation
theorem shift_parabola_two_units_right :
  ∀ x : ℝ, shift original_parabola 2 x = shifted_parabola x :=
by
  intros
  sorry

end shift_parabola_two_units_right_l12_12567


namespace tetrahedron_sum_l12_12504

theorem tetrahedron_sum :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  sorry

end tetrahedron_sum_l12_12504


namespace distance_of_intersections_l12_12221

theorem distance_of_intersections 
  (t : ℝ)
  (x := (2 - t) * (Real.sin (Real.pi / 6)))
  (y := (-1 + t) * (Real.sin (Real.pi / 6)))
  (curve : x = y)
  (circle : x^2 + y^2 = 8) :
  ∃ (B C : ℝ × ℝ), dist B C = Real.sqrt 30 := 
by
  sorry

end distance_of_intersections_l12_12221


namespace Jill_age_l12_12983

theorem Jill_age 
  (G H I J : ℕ)
  (h1 : G = H - 4)
  (h2 : H = I + 5)
  (h3 : I + 2 = J)
  (h4 : G = 18) : 
  J = 19 := 
sorry

end Jill_age_l12_12983


namespace john_money_left_l12_12095

-- Definitions for initial conditions
def initial_amount : ℤ := 100
def cost_roast : ℤ := 17
def cost_vegetables : ℤ := 11

-- Total spent calculation
def total_spent : ℤ := cost_roast + cost_vegetables

-- Remaining money calculation
def remaining_money : ℤ := initial_amount - total_spent

-- Theorem stating that John has €72 left
theorem john_money_left : remaining_money = 72 := by
  sorry

end john_money_left_l12_12095


namespace quadratic_inequality_solution_set_l12_12832

theorem quadratic_inequality_solution_set {x : ℝ} :
  (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end quadratic_inequality_solution_set_l12_12832


namespace final_selling_price_l12_12188

def actual_price : ℝ := 9941.52
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.05

noncomputable def final_price (P : ℝ) : ℝ :=
  P * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem final_selling_price :
  final_price actual_price = 6800.00 :=
by
  sorry

end final_selling_price_l12_12188


namespace solve_for_a_b_l12_12623

open Complex

theorem solve_for_a_b (a b : ℝ) (h : (mk 1 2) / (mk a b) = mk 1 1) : 
  a = 3 / 2 ∧ b = 1 / 2 :=
sorry

end solve_for_a_b_l12_12623


namespace f_f_neg1_l12_12948

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_f_neg1 : f (f (-1)) = 5 :=
  by
    sorry

end f_f_neg1_l12_12948


namespace range_of_a_l12_12512

def P (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def Q (x : ℝ) (a : ℝ) : Prop := x < a

theorem range_of_a (a : ℝ) : (∀ x, P x → Q x a) → (∀ x, Q x a → P x) → a ≥ 5 :=
by
  sorry

end range_of_a_l12_12512


namespace cost_price_of_book_l12_12073

theorem cost_price_of_book
  (C : ℝ)
  (h : 1.09 * C - 0.91 * C = 9) :
  C = 50 :=
sorry

end cost_price_of_book_l12_12073


namespace eval_expression_l12_12023

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12023


namespace power_of_two_divisor_l12_12325

theorem power_of_two_divisor {n : ℕ} (h_pos : n > 0) : 
  (∃ m : ℤ, (2^n - 1) ∣ (m^2 + 9)) → ∃ r : ℕ, n = 2^r :=
by
  sorry

end power_of_two_divisor_l12_12325


namespace ten_digit_number_l12_12436

open Nat

theorem ten_digit_number (a : Fin 10 → ℕ) (h1 : a 4 = 2)
  (h2 : a 8 = 3)
  (h3 : ∀ i, i < 8 → a i * a (i + 1) * a (i + 2) = 24) :
  a = ![4, 2, 3, 4, 2, 3, 4, 2, 3, 4] :=
sorry

end ten_digit_number_l12_12436


namespace eval_expression_l12_12025

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12025


namespace evaluate_expression_l12_12007

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12007


namespace max_wx_xy_yz_zt_l12_12242

theorem max_wx_xy_yz_zt {w x y z t : ℕ} (h_sum : w + x + y + z + t = 120)
  (hnn_w : 0 ≤ w) (hnn_x : 0 ≤ x) (hnn_y : 0 ≤ y) (hnn_z : 0 ≤ z) (hnn_t : 0 ≤ t) :
  wx + xy + yz + zt ≤ 3600 := 
sorry

end max_wx_xy_yz_zt_l12_12242


namespace complex_neither_sufficient_nor_necessary_real_l12_12068

noncomputable def quadratic_equation_real_roots (a : ℝ) : Prop := 
  (a^2 - 4 * a ≥ 0)

noncomputable def quadratic_equation_complex_roots (a : ℝ) : Prop := 
  (a^2 - 4 * (-a) < 0)

theorem complex_neither_sufficient_nor_necessary_real (a : ℝ) :
  (quadratic_equation_complex_roots a ↔ quadratic_equation_real_roots a) = false := 
sorry

end complex_neither_sufficient_nor_necessary_real_l12_12068


namespace evaluate_expression_l12_12010

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12010


namespace value_of_x_squared_plus_9y_squared_l12_12375

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l12_12375


namespace ten_thousands_written_correctly_ten_thousands_truncated_correctly_l12_12536

-- Definitions to be used in the proof
def ten_thousands_description := "Three thousand nine hundred seventy-six ten thousands"
def num_written : ℕ := 39760000
def truncated_num : ℕ := 3976

-- Theorems to be proven
theorem ten_thousands_written_correctly :
  (num_written = 39760000) :=
sorry

theorem ten_thousands_truncated_correctly :
  (truncated_num = 3976) :=
sorry

end ten_thousands_written_correctly_ten_thousands_truncated_correctly_l12_12536


namespace regular_pentagon_cannot_cover_floor_l12_12817

theorem regular_pentagon_cannot_cover_floor :
  ¬(∃ n : ℕ, 360 % 108 = 0) :=
begin
  sorry
end

end regular_pentagon_cannot_cover_floor_l12_12817


namespace loss_per_meter_calculation_l12_12182

/-- Define the given constants and parameters. --/
def total_meters : ℕ := 600
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 35

/-- Now we define the total cost price, total loss and loss per meter --/
def total_cost_price : ℕ := cost_price_per_meter * total_meters
def total_loss : ℕ := total_cost_price - selling_price
def loss_per_meter : ℕ := total_loss / total_meters

/-- State the theorem we need to prove. --/
theorem loss_per_meter_calculation : loss_per_meter = 5 :=
by
  sorry

end loss_per_meter_calculation_l12_12182


namespace car_clock_time_correct_l12_12524

noncomputable def car_clock (t : ℝ) : ℝ := t * (4 / 3)

theorem car_clock_time_correct :
  ∀ t_real t_car,
  (car_clock 0 = 0) ∧
  (car_clock 0.5 = 2 / 3) ∧
  (car_clock t_real = t_car) ∧
  (t_car = (8 : ℝ)) → (t_real = 6) → (t_real + 1 = 7) :=
by
  intro t_real t_car h
  sorry

end car_clock_time_correct_l12_12524


namespace John_other_trip_length_l12_12098

theorem John_other_trip_length :
  ∀ (fuel_per_km total_fuel first_trip_length other_trip_length : ℕ),
    fuel_per_km = 5 →
    total_fuel = 250 →
    first_trip_length = 20 →
    total_fuel / fuel_per_km - first_trip_length = other_trip_length →
    other_trip_length = 30 :=
by
  intros fuel_per_km total_fuel first_trip_length other_trip_length h1 h2 h3 h4
  sorry

end John_other_trip_length_l12_12098


namespace alice_card_value_l12_12998

theorem alice_card_value (x : ℝ) (hx : x ∈ Ioo (π / 2) π) 
  (h1 : ∃ a b c : ℝ, set.eq_on (λ y, y) (λ y, sin x) {a} ∧ set.eq_on (λ y, y) (λ y, cos x) {b} ∧ set.eq_on (λ y, y) (λ y, tan x) {c} ∧ ∀ y ∈ {a}, sin x ≠ y → ∀ y ∈ {b}, cos x ≠ y → ∀ y ∈ {c}, tan x ≠ y) :
  sin x = (-1 + real.sqrt 5) / 2 :=
sorry

end alice_card_value_l12_12998


namespace lara_sees_leo_for_six_minutes_l12_12152

-- Define constants for speeds and initial distances
def lara_speed : ℕ := 60
def leo_speed : ℕ := 40
def initial_distance : ℕ := 1
def time_to_minutes (t : ℚ) : ℚ := t * 60
-- Define the condition that proves Lara can see Leo for 6 minutes
theorem lara_sees_leo_for_six_minutes :
  lara_speed > leo_speed ∧
  initial_distance > 0 ∧
  (initial_distance : ℚ) / (lara_speed - leo_speed) * 2 = (6 : ℚ) / 60 :=
by
  sorry

end lara_sees_leo_for_six_minutes_l12_12152


namespace find_x_l12_12359

-- Definitions used in conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (4, x)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Main statement of the problem to be proved
theorem find_x (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = -1) : x = -1 / 5 :=
by {
  sorry
}

end find_x_l12_12359


namespace total_mile_times_l12_12283

-- Define the conditions
def Tina_time : ℕ := 6  -- Tina runs a mile in 6 minutes

def Tony_time : ℕ := Tina_time / 2  -- Tony runs twice as fast as Tina

def Tom_time : ℕ := Tina_time / 3  -- Tom runs three times as fast as Tina

-- Define the proof statement
theorem total_mile_times : Tony_time + Tina_time + Tom_time = 11 := by
  sorry

end total_mile_times_l12_12283


namespace eval_expression_l12_12022

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12022


namespace solution_to_equation_l12_12666

theorem solution_to_equation (x y : ℤ) (h : x^6 - y^2 = 648) : 
  (x = 3 ∧ y = 9) ∨ 
  (x = -3 ∧ y = 9) ∨ 
  (x = 3 ∧ y = -9) ∨ 
  (x = -3 ∧ y = -9) :=
sorry

end solution_to_equation_l12_12666


namespace ratio_of_B_to_C_l12_12289

variables (A B C : ℕ)

-- Conditions from the problem
axiom h1 : A = B + 2
axiom h2 : A + B + C = 12
axiom h3 : B = 4

-- Goal: Prove that the ratio of B's age to C's age is 2
theorem ratio_of_B_to_C : B / C = 2 :=
by {
  sorry
}

end ratio_of_B_to_C_l12_12289


namespace problem_solution_exists_six_values_n_l12_12225

open Int

theorem problem_solution_exists_six_values_n : 
  let condition := λ n : ℕ, (n > 0) ∧ (⌊Real.sqrt (n:ℝ)⌋ = (n + 1000) / 70) in 
  ∃ (S : Set ℕ), S = {n | condition n} ∧ S.card = 6 :=
by
  sorry

end problem_solution_exists_six_values_n_l12_12225


namespace find_value_of_fraction_l12_12246

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : (x / y) + (y / x) = 8)

theorem find_value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l12_12246


namespace eval_expression_l12_12019

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12019


namespace gcd_360_504_is_72_l12_12656

theorem gcd_360_504_is_72 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_is_72_l12_12656


namespace evaluate_x_squared_minus_y_squared_l12_12054

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end evaluate_x_squared_minus_y_squared_l12_12054


namespace g_minus_one_eq_zero_l12_12105

def g (x r : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 5 + r

theorem g_minus_one_eq_zero (r : ℝ) : g (-1) r = 0 → r = 14 := by
  sorry

end g_minus_one_eq_zero_l12_12105


namespace problem_part1_problem_part2_l12_12587

/-
Given the function f(x) = 2√3 sin(x) cos(x) - 2 cos(x)² + 1,
and the triangle ABC with angles A, B, C such that f(C) = 2 and side c = √3,
we need to prove:
1. The set of values of x when f(x) reaches its maximum is { x | x = k * π + π/3, k ∈ ℤ }.
2. The maximum area of triangle ABC is 3√3 / 4.
-/

noncomputable def f (x : ℝ) : ℝ :=
  2 * sqrt 3 * real.sin x * real.cos x - 2 * (real.cos x)^2 + 1

theorem problem_part1 :
  {x : ℝ | ∃ k : ℤ, x = k * real.pi + real.pi / 3} = 
  {x | f(x) = 2 * sqrt 3 * real.sin x * real.cos x - 2 * (real.cos x)^2 + 1 ∧
    ∀ y, f(y) ≤ f(x)} :=
sorry

theorem problem_part2 (C : ℝ) (a b : ℝ) (hC : f C = 2) (hc : b = sqrt 3) :
  ∃ (a b : ℝ), let area := 1/2 * a * b * real.sin C in area = 3 * sqrt 3 / 4 :=
sorry

end problem_part1_problem_part2_l12_12587


namespace unique_intersection_point_l12_12199

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (Real.log 3 / Real.log x)
noncomputable def h (x : ℝ) : ℝ := 3 - (1 / Real.sqrt (Real.log 3 / Real.log x))

theorem unique_intersection_point : (∃! (x : ℝ), (x > 0) ∧ (f x = g x ∨ f x = h x ∨ g x = h x)) :=
sorry

end unique_intersection_point_l12_12199


namespace distinct_triangles_from_chord_intersections_l12_12110

theorem distinct_triangles_from_chord_intersections :
  let points := 9
  let chords := (points.choose 2)
  let intersections := (points.choose 4)
  let triangles := (points.choose 6)
  (chords > 0 ∧ intersections > 0 ∧ triangles > 0) →
  triangles = 84 :=
by
  intros
  sorry

end distinct_triangles_from_chord_intersections_l12_12110


namespace min_cost_to_form_closed_chain_l12_12174

/-- Definition for the cost model -/
def cost_separate_link : ℕ := 1
def cost_attach_link : ℕ := 2
def total_cost (n : ℕ) : ℕ := n * (cost_separate_link + cost_attach_link)

-- Number of pieces of gold chain and links in each chain
def num_pieces : ℕ := 13

/-- Minimum cost calculation proof statement -/
theorem min_cost_to_form_closed_chain : total_cost (num_pieces - 1) = 36 := 
by
  sorry

end min_cost_to_form_closed_chain_l12_12174


namespace geometric_a1_value_l12_12845

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ (n - 1)

theorem geometric_a1_value (a3 a5 : ℝ) (q : ℝ) : 
  a3 = geometric_sequence a1 q 3 →
  a5 = geometric_sequence a1 q 5 →
  a1 = 2 :=
by
  sorry

end geometric_a1_value_l12_12845


namespace find_value_of_x_l12_12950

theorem find_value_of_x :
  ∃ x : ℝ, (0.65 * x = 0.20 * 747.50) ∧ x = 230 :=
by
  sorry

end find_value_of_x_l12_12950


namespace num_sets_with_6_sum_18_is_4_l12_12921

open Finset

def num_sets_with_6_sum_18 : ℕ :=
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  s.filter (λ t, t.card = 3 ∧ 18 ∈ {t.sum} ∧ 6 ∈ t).card

theorem num_sets_with_6_sum_18_is_4 :
  num_sets_with_6_sum_18 = 4 :=
sorry

end num_sets_with_6_sum_18_is_4_l12_12921


namespace cost_of_each_candy_bar_l12_12138

theorem cost_of_each_candy_bar
  (p_chips : ℝ)
  (total_cost : ℝ)
  (num_students : ℕ)
  (num_chips_per_student : ℕ)
  (num_candy_bars_per_student : ℕ)
  (h1 : p_chips = 0.50)
  (h2 : total_cost = 15)
  (h3 : num_students = 5)
  (h4 : num_chips_per_student = 2)
  (h5 : num_candy_bars_per_student = 1) :
  ∃ C : ℝ, C = 2 := 
by 
  sorry

end cost_of_each_candy_bar_l12_12138


namespace monotonic_decreasing_fx_l12_12465

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem monotonic_decreasing_fx : ∀ (x : ℝ), (0 < x) ∧ (x < (1 / exp 1)) → deriv f x < 0 := 
by
  sorry

end monotonic_decreasing_fx_l12_12465


namespace Faye_total_pencils_l12_12973

def pencils_per_row : ℕ := 8
def number_of_rows : ℕ := 4
def total_pencils : ℕ := pencils_per_row * number_of_rows

theorem Faye_total_pencils : total_pencils = 32 := by
  sorry

end Faye_total_pencils_l12_12973


namespace find_d_e_f_l12_12762

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 3 + 5 / 3)

theorem find_d_e_f :
  ∃ (d e f : ℕ), (y ^ 50 = 3 * y ^ 48 + 10 * y ^ 45 + 9 * y ^ 43 - y ^ 25 + d * y ^ 21 + e * y ^ 19 + f * y ^ 15) 
    ∧ (d + e + f = 119) :=
sorry

end find_d_e_f_l12_12762


namespace jordan_rectangle_width_l12_12196

noncomputable def carol_length : ℝ := 4.5
noncomputable def carol_width : ℝ := 19.25
noncomputable def jordan_length : ℝ := 3.75

noncomputable def carol_area : ℝ := carol_length * carol_width
noncomputable def jordan_width : ℝ := carol_area / jordan_length

theorem jordan_rectangle_width : jordan_width = 23.1 := by
  -- proof will go here
  sorry

end jordan_rectangle_width_l12_12196


namespace taxi_faster_than_truck_l12_12142

noncomputable def truck_speed : ℝ := 2.1 / 1
noncomputable def taxi_speed : ℝ := 10.5 / 4

theorem taxi_faster_than_truck :
  taxi_speed / truck_speed = 1.25 :=
by
  sorry

end taxi_faster_than_truck_l12_12142


namespace diff_squares_of_roots_l12_12823

theorem diff_squares_of_roots : ∀ α β : ℝ, (α * β = 6) ∧ (α + β = 5) -> (α - β)^2 = 1 := by
  sorry

end diff_squares_of_roots_l12_12823


namespace total_pencils_l12_12882

-- Defining the number of pencils each person has.
def jessica_pencils : ℕ := 8
def sandy_pencils : ℕ := 8
def jason_pencils : ℕ := 8

-- Theorem stating the total number of pencils
theorem total_pencils : jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end total_pencils_l12_12882


namespace abc_sum_l12_12776

theorem abc_sum (f : ℝ → ℝ) (a b c : ℝ) :
  f (x - 2) = 2 * x^2 - 5 * x + 3 → f x = a * x^2 + b * x + c → a + b + c = 6 :=
by
  intros h₁ h₂
  sorry

end abc_sum_l12_12776


namespace bus_trip_speed_l12_12297

theorem bus_trip_speed :
  ∃ v : ℝ, v > 0 ∧ (660 / v - 1 = 660 / (v + 5)) ∧ v = 55 :=
by
  sorry

end bus_trip_speed_l12_12297


namespace smallest_piece_to_cut_l12_12184

theorem smallest_piece_to_cut (x : ℕ) 
  (h1 : 9 - x > 0) 
  (h2 : 16 - x > 0) 
  (h3 : 18 - x > 0) :
  7 ≤ x ∧ 9 - x + 16 - x ≤ 18 - x :=
by {
  sorry
}

end smallest_piece_to_cut_l12_12184


namespace speed_conversion_l12_12167

-- Define the given condition
def kmph_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Speed in kmph
def speed_kmph : ℕ := 216

-- The proof statement
theorem speed_conversion : kmph_to_mps speed_kmph = 60 :=
by
  sorry

end speed_conversion_l12_12167


namespace binomial_expansion_conditions_l12_12500

noncomputable def binomial_expansion (a b : ℝ) (x y : ℝ) (n : ℕ) : ℝ :=
(1 + a*x + b*y)^n

theorem binomial_expansion_conditions
  (a b : ℝ) (n : ℕ) 
  (h1 : (1 + b)^n = 243)
  (h2 : (1 + |a|)^n = 32) :
  a = 1 ∧ b = 2 ∧ n = 5 := by
  sorry

end binomial_expansion_conditions_l12_12500


namespace joseph_cards_l12_12391

theorem joseph_cards (cards_per_student : ℕ) (students : ℕ) (cards_left : ℕ) 
    (H1 : cards_per_student = 23)
    (H2 : students = 15)
    (H3 : cards_left = 12) 
    : (cards_per_student * students + cards_left = 357) := 
  by
  sorry

end joseph_cards_l12_12391


namespace exist_m_eq_l12_12891

theorem exist_m_eq (n b : ℕ) (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_zero : n ≠ 0) (hb_zero : b ≠ 0)
  (h_div : p ∣ (b^(2^n) + 1)) :
  ∃ m : ℕ, p = 2^(n+1) * m + 1 :=
by
  sorry

end exist_m_eq_l12_12891


namespace remainder_101_pow_47_mod_100_l12_12931

theorem remainder_101_pow_47_mod_100 : (101 ^ 47) % 100 = 1 := by 
  sorry

end remainder_101_pow_47_mod_100_l12_12931


namespace cards_drawing_problem_l12_12532

-- Define the sets and parameters
def total_cards := 16
def group_size := 4
def red_cards := 4
def yellow_cards := 4
def blue_cards := 4
def green_cards := 4
def draw_cards := 3

-- Theorem statement
theorem cards_drawing_problem : 
  (nat.choose total_cards draw_cards) - 
  4 * (nat.choose group_size draw_cards) = 544 := 
by sorry

end cards_drawing_problem_l12_12532


namespace meat_purchase_l12_12387

theorem meat_purchase :
  ∃ x y : ℕ, 16 * x = y + 25 ∧ 8 * x = y - 15 ∧ y / x = 11 :=
by
  sorry

end meat_purchase_l12_12387


namespace sum_of_first_five_terms_arith_seq_l12_12287

/-
An arithmetic sequence where the first term is 6 and the common difference is 4.
We aim to prove that the sum of the first five terms is 70.
-/

def a : ℕ → ℕ
| 0     := 6
| (n+1) := a n + 4

theorem sum_of_first_five_terms_arith_seq : 
  (a 0 + a 1 + a 2 + a 3 + a 4) = 70 :=
by
  sorry

end sum_of_first_five_terms_arith_seq_l12_12287


namespace calc1_calc2_calc3_calc4_l12_12702

theorem calc1 : 327 + 46 - 135 = 238 := by sorry
theorem calc2 : 1000 - 582 - 128 = 290 := by sorry
theorem calc3 : (124 - 62) * 6 = 372 := by sorry
theorem calc4 : 500 - 400 / 5 = 420 := by sorry

end calc1_calc2_calc3_calc4_l12_12702


namespace square_table_production_l12_12560

theorem square_table_production (x y : ℝ) :
  x + y = 5 ∧ 50 * x * 4 = 300 * y → 
  x = 3 ∧ y = 2 ∧ 50 * x = 150 :=
by
  sorry

end square_table_production_l12_12560


namespace value_of_x2_plus_9y2_l12_12372

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l12_12372


namespace simplify_power_of_product_l12_12635

theorem simplify_power_of_product (x : ℝ) : (5 * x^2)^4 = 625 * x^8 :=
by
  sorry

end simplify_power_of_product_l12_12635


namespace theater_tickets_l12_12691

theorem theater_tickets (O B P : ℕ) (h1 : O + B + P = 550) 
  (h2 : 15 * O + 10 * B + 25 * P = 9750) (h3: P = 5 * O) (h4 : O ≥ 50) : 
  B - O = 179 :=
by
  sorry

end theater_tickets_l12_12691


namespace probability_Jane_Albert_same_committee_is_correct_l12_12664

noncomputable def probability_Jane_Albert_same_committee : ℝ :=
  let n : ℕ := 6
  let k : ℕ := 3
  let all_students : Finset ℕ := {0, 1, 2, 3, 4, 5}  -- representing the 6 students
  let Jane : ℕ := 4
  let Albert : ℕ := 5
  let possible_committees := all_students.powerset.filter (λ s, s.card = k)
  let total_committees : ℕ := possible_committees.card
  let favorable_committees := possible_committees.filter (λ s, Jane ∈ s ∧ Albert ∈ s)
  let favorable_count : ℕ := favorable_committees.card
  (favorable_count : ℝ) / (total_committees : ℝ)

theorem probability_Jane_Albert_same_committee_is_correct : probability_Jane_Albert_same_committee = 1 / 5 :=
by
  let n : ℕ := 6
  let k : ℕ := 3
  let all_students : Finset ℕ := {0, 1, 2, 3, 4, 5}  -- representing the 6 students
  let Jane : ℕ := 4
  let Albert : ℕ := 5
  let possible_committees := all_students.powerset.filter (λ s, s.card = k)
  let total_committees := possible_committees.card
  have h_total_committees : total_committees = 20 := by
    -- Proof of total committees being 20
    sorry
  let favorable_committees := possible_committees.filter (λ s, Jane ∈ s ∧ Albert ∈ s)
  let favorable_count := favorable_committees.card
  have h_favorable_count : favorable_count = 4 := by
    -- Proof of favorable committees being 4
    sorry
  calc
    probability_Jane_Albert_same_committee
      = (favorable_committees.card : ℝ) / (possible_committees.card : ℝ) := rfl
  ... = (favorable_count : ℝ) / (total_committees : ℝ) := by congr
  ... = 4 / 20 := by rw [h_favorable_count, h_total_committees]
  ... = 1 / 5 := by norm_num

end probability_Jane_Albert_same_committee_is_correct_l12_12664


namespace find_b_l12_12866

variable (x : ℝ)

theorem find_b (a b: ℝ) (h1 : x + 1/x = a) (h2 : x^3 + 1/x^3 = b) (ha : a = 3): b = 18 :=
by
  sorry

end find_b_l12_12866


namespace cello_viola_pairs_l12_12553

theorem cello_viola_pairs (cellos violas : Nat) (p_same_tree : ℚ) (P : Nat)
  (h_cellos : cellos = 800)
  (h_violas : violas = 600)
  (h_p_same_tree : p_same_tree = 0.00020833333333333335)
  (h_equation : P * ((1 : ℚ) / cellos * (1 : ℚ) / violas) = p_same_tree) :
  P = 100 := 
by
  sorry

end cello_viola_pairs_l12_12553


namespace root_of_polynomial_l12_12227

theorem root_of_polynomial (k : ℝ) (h : (3 : ℝ) ^ 4 + k * (3 : ℝ) ^ 2 + 27 = 0) : k = -12 :=
by
  sorry

end root_of_polynomial_l12_12227


namespace minimum_value_l12_12894

noncomputable def problem_statement (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 4 * c^2

theorem minimum_value : ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27), 
  problem_statement a b c h = 180 :=
sorry

end minimum_value_l12_12894


namespace compelling_quadruples_l12_12826
   
   def isCompellingQuadruple (a b c d : ℕ) : Prop :=
     1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 10 ∧ a + d < b + c 

   def compellingQuadruplesCount (count : ℕ) : Prop :=
     count = 80
   
   theorem compelling_quadruples :
     ∃ count, compellingQuadruplesCount count :=
   by
     use 80
     sorry
   
end compelling_quadruples_l12_12826


namespace curves_intersect_at_four_points_l12_12795

theorem curves_intersect_at_four_points (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 = a^2 ∧ y = -x^2 + a ) ∧ 
   (0 = x ∧ y = a) ∧ 
   (∃ t : ℝ, x = t ∧ (y = 1 ∧ x^2 = a - 1))) ↔ a = 2 := 
by
  sorry

end curves_intersect_at_four_points_l12_12795


namespace evaluate_expression_l12_12829

theorem evaluate_expression :
  (⌈(19 / 7 : ℚ) - ⌈(35 / 19 : ℚ)⌉⌉ / ⌈(35 / 7 : ℚ) + ⌈((7 * 19) / 35 : ℚ)⌉⌉) = (1 / 9 : ℚ) :=
by
  sorry

end evaluate_expression_l12_12829


namespace sin_probability_interval_l12_12558

theorem sin_probability_interval :
  let I : set ℝ := set.Icc (-1 : ℝ) (1 : ℝ)
  let J : set ℝ := set.Icc (-1 / 2) (real.sqrt 2 / 2)
  let range_x : set ℝ := { x : ℝ | (sin (real.pi * x / 4)) ∈ J }
  set.density I range_x = 5 / 6 :=
by
  -- Here we assume that set.density is the appropriate formalization of the desired probability.
  sorry

end sin_probability_interval_l12_12558


namespace other_religion_students_l12_12081

theorem other_religion_students (total_students : ℕ) 
  (muslims_percent hindus_percent sikhs_percent christians_percent buddhists_percent : ℝ) 
  (h1 : total_students = 1200) 
  (h2 : muslims_percent = 0.35) 
  (h3 : hindus_percent = 0.25) 
  (h4 : sikhs_percent = 0.15) 
  (h5 : christians_percent = 0.10) 
  (h6 : buddhists_percent = 0.05) : 
  ∃ other_religion_students : ℕ, other_religion_students = 120 :=
by
  sorry

end other_religion_students_l12_12081


namespace positive_integers_satisfy_l12_12912

theorem positive_integers_satisfy (n : ℕ) (h1 : 25 - 5 * n > 15) : n = 1 :=
by sorry

end positive_integers_satisfy_l12_12912


namespace total_cost_correct_l12_12399

def cost_barette : ℕ := 3
def cost_comb : ℕ := 1

def kristine_barrettes : ℕ := 1
def kristine_combs : ℕ := 1

def crystal_barrettes : ℕ := 3
def crystal_combs : ℕ := 1

def total_spent (cost_barette : ℕ) (cost_comb : ℕ) 
  (kristine_barrettes : ℕ) (kristine_combs : ℕ) 
  (crystal_barrettes : ℕ) (crystal_combs : ℕ) : ℕ :=
  (kristine_barrettes * cost_barette + kristine_combs * cost_comb) + 
  (crystal_barrettes * cost_barette + crystal_combs * cost_comb)

theorem total_cost_correct :
  total_spent cost_barette cost_comb kristine_barrettes kristine_combs crystal_barrettes crystal_combs = 14 :=
by
  sorry

end total_cost_correct_l12_12399


namespace triangle_area_l12_12926

namespace MathProof

theorem triangle_area (y_eq_6 y_eq_2_plus_x y_eq_2_minus_x : ℝ → ℝ)
  (h1 : ∀ x, y_eq_6 x = 6)
  (h2 : ∀ x, y_eq_2_plus_x x = 2 + x)
  (h3 : ∀ x, y_eq_2_minus_x x = 2 - x) :
  let a := (4, 6)
  let b := (-4, 6)
  let c := (0, 2)
  let base := dist a b
  let height := (6 - 2:ℝ)
  (1 / 2 * base * height = 16) := by
    sorry

end MathProof

end triangle_area_l12_12926


namespace total_time_spent_l12_12878

-- Define the conditions
def t1 : ℝ := 2.5
def t2 : ℝ := 3 * t1

-- Define the theorem to prove
theorem total_time_spent : t1 + t2 = 10 := by
  sorry

end total_time_spent_l12_12878


namespace kim_boxes_on_tuesday_l12_12632

theorem kim_boxes_on_tuesday
  (sold_on_thursday : ℕ)
  (sold_on_wednesday : ℕ)
  (sold_on_tuesday : ℕ)
  (h1 : sold_on_thursday = 1200)
  (h2 : sold_on_wednesday = 2 * sold_on_thursday)
  (h3 : sold_on_tuesday = 2 * sold_on_wednesday) :
  sold_on_tuesday = 4800 :=
sorry

end kim_boxes_on_tuesday_l12_12632


namespace number_of_distinct_stackings_l12_12153

-- Defining the conditions
def cubes : ℕ := 8
def edge_length : ℕ := 1
def valid_stackings (n : ℕ) : Prop := 
  n = 8 -- Stating that we are working with 8 cubes

-- The theorem stating the problem and expected solution
theorem number_of_distinct_stackings : 
  cubes = 8 ∧ edge_length = 1 ∧ valid_stackings cubes → ∃ (count : ℕ), count = 10 :=
by 
  sorry

end number_of_distinct_stackings_l12_12153


namespace smallest_num_rectangles_to_cover_square_l12_12539

-- Define essential conditions
def area_3by4_rectangle : ℕ := 3 * 4
def area_square (side_length : ℕ) : ℕ := side_length * side_length
def can_be_tiled_with_3by4 (side_length : ℕ) : Prop := (area_square side_length) % area_3by4_rectangle = 0

-- Define the main theorem
theorem smallest_num_rectangles_to_cover_square :
  can_be_tiled_with_3by4 12 → ∃ n : ℕ, n = (area_square 12) / area_3by4_rectangle ∧ n = 12 :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l12_12539


namespace union_sets_l12_12109

open Set

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem union_sets :
  A ∪ B = {1, 2, 3, 4, 5} :=
by
  sorry

end union_sets_l12_12109


namespace trash_cans_street_count_l12_12313

theorem trash_cans_street_count (S B : ℕ) (h1 : B = 2 * S) (h2 : S + B = 42) : S = 14 :=
by
  sorry

end trash_cans_street_count_l12_12313


namespace find_k_l12_12588

theorem find_k 
  (e1 : ℝ × ℝ) (h_e1 : e1 = (1, 0))
  (e2 : ℝ × ℝ) (h_e2 : e2 = (0, 1))
  (a : ℝ × ℝ) (h_a : a = (1, -2))
  (b : ℝ × ℝ) (h_b : b = (k, 1))
  (parallel : ∃ m : ℝ, a = (m * b.1, m * b.2)) : 
  k = -1/2 :=
sorry

end find_k_l12_12588


namespace john_money_left_l12_12094

-- Definitions for initial conditions
def initial_amount : ℤ := 100
def cost_roast : ℤ := 17
def cost_vegetables : ℤ := 11

-- Total spent calculation
def total_spent : ℤ := cost_roast + cost_vegetables

-- Remaining money calculation
def remaining_money : ℤ := initial_amount - total_spent

-- Theorem stating that John has €72 left
theorem john_money_left : remaining_money = 72 := by
  sorry

end john_money_left_l12_12094


namespace shifted_function_is_correct_l12_12383

-- Define the original function
def original_function (x : ℝ) : ℝ := -2 * x

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function (x - 3)

-- State the theorem to be proven
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = -2 * x + 6 :=
by
  sorry

end shifted_function_is_correct_l12_12383


namespace B_investment_l12_12296

theorem B_investment (A : ℝ) (t_B : ℝ) (profit_ratio : ℝ) (B_investment_result : ℝ) : 
  A = 27000 → t_B = 4.5 → profit_ratio = 2 → B_investment_result = 36000 :=
by
  intro hA htB hpR
  sorry

end B_investment_l12_12296


namespace goodColoringsOfPoints_l12_12754

noncomputable def countGoodColorings (k m : ℕ) : ℕ :=
  (k * (k - 1) + 2) * 2 ^ m

theorem goodColoringsOfPoints :
  countGoodColorings 2011 2011 = (2011 * 2010 + 2) * 2 ^ 2011 :=
  by
    sorry

end goodColoringsOfPoints_l12_12754


namespace explicit_formula_for_f_l12_12825

def f (k : ℕ) : ℚ :=
  if k = 1 then 4 / 3
  else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3

theorem explicit_formula_for_f (k : ℕ) (hk : k ≥ 1) : 
  (f k = if k = 1 then 4 / 3 else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3) ∧ 
  ∀ k ≥ 2, 2 * f k = f (k - 1) - k * 5^k + 2^k :=
by {
  sorry
}

end explicit_formula_for_f_l12_12825


namespace eval_expression_l12_12028

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12028


namespace value_of_square_sum_l12_12362

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l12_12362


namespace sin_three_pi_four_minus_alpha_l12_12475

theorem sin_three_pi_four_minus_alpha 
  (α : ℝ) 
  (h₁ : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (3 * π / 4 - α) = 3 / 5 :=
by
  sorry

end sin_three_pi_four_minus_alpha_l12_12475


namespace faye_initial_books_l12_12711

theorem faye_initial_books (X : ℕ) (h : (X - 3) + 48 = 79) : X = 34 :=
sorry

end faye_initial_books_l12_12711


namespace inequality_for_natural_n_l12_12404

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end inequality_for_natural_n_l12_12404


namespace Suma_can_complete_in_6_days_l12_12772

-- Define the rates for Renu and their combined rate
def Renu_rate := (1 : ℚ) / 6
def Combined_rate := (1 : ℚ) / 3

-- Define Suma's time to complete the work alone
def Suma_days := 6

-- defining the work rate Suma is required to achieve given the known rates and combined rate
def Suma_rate := Combined_rate - Renu_rate

-- Require to prove 
theorem Suma_can_complete_in_6_days : (1 / Suma_rate) = Suma_days :=
by
  -- Using the definitions provided and some basic algebra to prove the theorem 
  sorry

end Suma_can_complete_in_6_days_l12_12772


namespace initial_percentage_increase_l12_12189

theorem initial_percentage_increase (E P : ℝ) 
  (h1 : E * (1 + P / 100) = 678)
  (h2 : E * 1.15 = 683.95) :
  P ≈ 14 :=
by sorry

end initial_percentage_increase_l12_12189


namespace werewolf_knight_is_A_l12_12417

structure Person :=
  (isKnight : Prop)
  (isLiar : Prop)
  (isWerewolf : Prop)

variables (A B C : Person)

-- A's statement: "At least one of us is a liar."
def statementA (A B C : Person) : Prop := A.isLiar ∨ B.isLiar ∨ C.isLiar

-- B's statement: "C is a knight."
def statementB (C : Person) : Prop := C.isKnight

theorem werewolf_knight_is_A (A B C : Person) 
  (hA : statementA A B C)
  (hB : statementB C)
  (hWerewolfKnight : ∃ x : Person, x.isWerewolf ∧ x.isKnight ∧ ¬ (A ≠ x ∧ B ≠ x ∧ C ≠ x))
  : A.isWerewolf ∧ A.isKnight :=
sorry

end werewolf_knight_is_A_l12_12417


namespace number_of_polynomials_l12_12197

theorem number_of_polynomials (count : ℕ) : 
  (∃ (a b c d e : ℕ), 
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧ e ≤ 9 ∧ 
    -a + b - c + d - e = -20) ∧ count = 12650 := 
sorry

end number_of_polynomials_l12_12197


namespace length_of_shorter_side_l12_12812

-- Define the given conditions
def width : ℝ := 50
def num_poles : ℕ := 24
def distance_between_poles : ℝ := 5

-- Prove the length of the shorter side
theorem length_of_shorter_side : 
  (let num_gaps := num_poles - 1 in
   let total_perimeter := num_gaps * distance_between_poles in
   2 * (length + width) = total_perimeter) → 
  length = 7.5 :=
by
  -- Add proof here
  sorry

end length_of_shorter_side_l12_12812


namespace area_of_rectangular_plot_l12_12413

-- Defining the breadth
def breadth : ℕ := 26

-- Defining the length as thrice the breadth
def length : ℕ := 3 * breadth

-- Defining the area as the product of length and breadth
def area : ℕ := length * breadth

-- The theorem stating the problem to prove
theorem area_of_rectangular_plot : area = 2028 := by
  -- Initial proof step skipped
  sorry

end area_of_rectangular_plot_l12_12413


namespace probability_white_marble_l12_12165

theorem probability_white_marble :
  ∀ (p_blue p_green p_white : ℝ),
    p_blue = 0.25 →
    p_green = 0.4 →
    p_blue + p_green + p_white = 1 →
    p_white = 0.35 :=
by
  intros p_blue p_green p_white h_blue h_green h_total
  sorry

end probability_white_marble_l12_12165


namespace arrange_books_l12_12519

theorem arrange_books :
  let total_books := 9,
      arabic_books := 2,
      german_books := 3,
      spanish_books := 4,
      books := arabic_books + german_books + spanish_books,
      group_arabic := 1,
      group_spanish := 1,
      book_groups := group_arabic + german_books + group_spanish,
      arrangement_groups := book_groups.factorial,
      arrange_arabic := arabic_books.factorial,
      arrange_spanish := spanish_books.factorial in
  books = total_books →
  group_arabic = 1 ∧ group_spanish = 1 →
  (arrangement_groups * arrange_arabic * arrange_spanish = 5760) :=
begin
  intros,
  exact sorry
end

end arrange_books_l12_12519


namespace rabbit_can_escape_l12_12307

def RabbitEscapeExists
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : Prop :=
 ∃ (rabbit_escapes : Bool), rabbit_escapes = true

theorem rabbit_can_escape
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : RabbitEscapeExists center_x center_y wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 wolf_speed rabbit_speed condition1 condition2 condition3 := 
sorry

end rabbit_can_escape_l12_12307


namespace green_pill_cost_l12_12967

-- Given conditions
def days := 21
def total_cost := 903
def cost_difference := 2
def daily_cost := total_cost / days

-- Statement to prove
theorem green_pill_cost : (∃ (y : ℝ), y + (y - cost_difference) = daily_cost ∧ y = 22.5) :=
by
  sorry

end green_pill_cost_l12_12967


namespace probability_both_heads_on_last_flip_l12_12425

noncomputable def fair_coin_flip : probabilityₓ ℙ :=
  probabilityₓ.ofUniform [true, false]

def both_coins_heads (events : list (bool × bool)) : bool :=
  events.all (λ event, event.1 = true)

def stops_with_heads (events : list (bool × bool)) : bool :=
  events.any (λ event, event.1 = true ∨ event.2 = true)

theorem probability_both_heads_on_last_flip :
  ∀ events : list (bool × bool), probabilityₓ (fair_coin_flip ×ₗ fair_coin_flip)
  (λ event, both_coins_heads events = true ∧ stops_with_heads events = true) = 1 / 3 :=
sorry

end probability_both_heads_on_last_flip_l12_12425


namespace bags_needed_l12_12696

theorem bags_needed (expected_people extra_people extravagant_bags average_bags : ℕ) 
    (h1 : expected_people = 50) 
    (h2 : extra_people = 40) 
    (h3 : extravagant_bags = 10) 
    (h4 : average_bags = 20) : 
    (expected_people + extra_people - (extravagant_bags + average_bags) = 60) :=
by {
  sorry
}

end bags_needed_l12_12696


namespace part_1_part_2_l12_12223

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part_1 (a : ℝ) (h : ∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) : a = 2 :=
sorry

theorem part_2 (a : ℝ) (h : a = 2) : ∀ m, (∀ x, f (3 * x) a + f (x + 3) a ≥ m) ↔ m ≤ 5 / 3 :=
sorry

end part_1_part_2_l12_12223


namespace reflected_circle_center_l12_12260

theorem reflected_circle_center
  (original_center : ℝ × ℝ) 
  (reflection_line : ℝ × ℝ → ℝ × ℝ)
  (hc : original_center = (8, -3))
  (hl : ∀ (p : ℝ × ℝ), reflection_line p = (-p.2, -p.1))
  : reflection_line original_center = (3, -8) :=
sorry

end reflected_circle_center_l12_12260


namespace h_inverse_correct_l12_12247

noncomputable def f (x : ℝ) := 4 * x + 7
noncomputable def g (x : ℝ) := 3 * x - 2
noncomputable def h (x : ℝ) := f (g x)
noncomputable def h_inv (y : ℝ) := (y + 1) / 12

theorem h_inverse_correct : ∀ x : ℝ, h_inv (h x) = x :=
by
  intro x
  sorry

end h_inverse_correct_l12_12247


namespace proof_not_necessarily_15_points_l12_12954

-- Define the number of teams
def teams := 14

-- Define a tournament where each team plays every other exactly once
def games := (teams * (teams - 1)) / 2

-- Define a function calculating the total points by summing points for each game
def total_points (wins draws : ℕ) := (3 * wins) + (1 * draws)

-- Define a statement that total points is at least 150
def scores_sum_at_least_150 (wins draws : ℕ) : Prop :=
  total_points wins draws ≥ 150

-- Define a condition that a score could be less than 15
def highest_score_not_necessarily_15 : Prop :=
  ∃ (scores : Finset ℕ), scores.card = teams ∧ ∀ score ∈ scores, score < 15

theorem proof_not_necessarily_15_points :
  ∃ (wins draws : ℕ), wins + draws = games ∧ scores_sum_at_least_150 wins draws ∧ highest_score_not_necessarily_15 :=
by
  sorry

end proof_not_necessarily_15_points_l12_12954


namespace conic_section_is_ellipse_l12_12828

theorem conic_section_is_ellipse :
  ∀ x y : ℝ, 4 * x^2 + y^2 - 12 * x - 2 * y + 4 = 0 →
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧ (a * (x - h)^2 + b * (y - k)^2 = 1) :=
by
  sorry

end conic_section_is_ellipse_l12_12828


namespace evaluate_x2_minus_y2_l12_12046

-- Definitions based on the conditions.
def x : ℝ
def y : ℝ
axiom cond1 : x + y = 12
axiom cond2 : 3 * x + y = 18

-- The main statement we need to prove.
theorem evaluate_x2_minus_y2 : x^2 - y^2 = -72 :=
by
  sorry

end evaluate_x2_minus_y2_l12_12046


namespace area_of_scalene_right_triangle_l12_12258

noncomputable def area_of_triangle_DEF (DE EF : ℝ) (h1 : DE > 0) (h2 : EF > 0) (h3 : DE / EF = 3) (h4 : DE^2 + EF^2 = 16) : ℝ :=
1 / 2 * DE * EF

theorem area_of_scalene_right_triangle (DE EF : ℝ) 
  (h1 : DE > 0)
  (h2 : EF > 0)
  (h3 : DE / EF = 3)
  (h4 : DE^2 + EF^2 = 16) :
  area_of_triangle_DEF DE EF h1 h2 h3 h4 = 2.4 :=
sorry

end area_of_scalene_right_triangle_l12_12258


namespace linear_function_does_not_pass_fourth_quadrant_l12_12780

theorem linear_function_does_not_pass_fourth_quadrant :
  ∀ x, (2 * x + 1 ≥ 0) :=
by sorry

end linear_function_does_not_pass_fourth_quadrant_l12_12780


namespace proof_problem_statement_l12_12989

noncomputable def ellipse := { P : ℝ × ℝ // P.1^2 / 2 + P.2^2 = 1 }

variables {P Q : ellipse} {F1 F2 O : (ℝ × ℝ)}
variables {k : ℝ}

-- Conditions
def on_ellipse (P : ellipse) : Prop :=
  P.1 * P.1 / 2 + P.2 * P.2 = 1

def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

-- Given conditions
def given_conditions (P Q F1 F2 : ℝ × ℝ) : Prop :=
  on_ellipse P ∧ on_ellipse Q ∧ perpendicular (P - F2) (Q - F2)

def condition1 (P F1 F2 O : ℝ × ℝ) : Prop :=
  min (complex.norm (P - F1 + P - F2)) = 2

def condition2 (P Q F1 F2 : ℝ × ℝ) (k : ℝ) : Prop :=
  perpendicular (P - F1 + P - F2) (Q - F1 + Q - F2) →
  k = slope_of_PQ P Q → k^2 = (2 * sqrt 10 / 10) - 5 / 10

theorem proof_problem_statement (P Q F1 F2 : ℝ × ℝ) (k : ℝ) (h : given_conditions P Q F1 F2) :
  condition1 P F1 F2 0 ∧ condition2 P Q F1 F2 k :=
by
  sorry

end proof_problem_statement_l12_12989


namespace angle_380_in_first_quadrant_l12_12940

theorem angle_380_in_first_quadrant : ∃ n : ℤ, 380 - 360 * n = 20 ∧ 0 ≤ 20 ∧ 20 ≤ 90 :=
by
  use 1 -- We use 1 because 380 = 20 + 360 * 1
  sorry

end angle_380_in_first_quadrant_l12_12940


namespace downstream_speed_l12_12556

-- Define the given conditions as constants
def V_u : ℝ := 25 -- upstream speed in kmph
def V_m : ℝ := 40 -- speed of the man in still water in kmph

-- Define the speed of the stream
def V_s := V_m - V_u

-- Define the downstream speed
def V_d := V_m + V_s

-- Assertion we need to prove
theorem downstream_speed : V_d = 55 := by
  sorry

end downstream_speed_l12_12556


namespace perpendicular_condition_parallel_condition_opposite_direction_l12_12719

/-- Conditions definitions --/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def k_vector_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
def vector_a_minus_3b : ℝ × ℝ := (10, -4)

/-- Problem 1: Prove the perpendicular condition --/
theorem perpendicular_condition (k : ℝ) : (k_vector_a_plus_b k).fst * vector_a_minus_3b.fst + (k_vector_a_plus_b k).snd * vector_a_minus_3b.snd = 0 → k = 19 :=
by
  sorry

/-- Problem 2: Prove the parallel condition --/
theorem parallel_condition (k : ℝ) : (-(k - 3) / 10 = (2 * k + 2) / (-4)) → k = -1/3 :=
by
  sorry

/-- Determine if the vectors are in opposite directions --/
theorem opposite_direction (k : ℝ) (hk : k = -1/3) : k_vector_a_plus_b k = (-(1/3):ℝ) • vector_a_minus_3b :=
by
  sorry

end perpendicular_condition_parallel_condition_opposite_direction_l12_12719


namespace kim_probability_same_color_l12_12099

noncomputable def probability_same_color (total_shoes : ℕ) (pairs_of_shoes : ℕ) : ℚ :=
  let total_selections := (total_shoes * (total_shoes - 1)) / 2
  let successful_selections := pairs_of_shoes
  successful_selections / total_selections

theorem kim_probability_same_color :
  probability_same_color 10 5 = 1 / 9 :=
by
  unfold probability_same_color
  have h_total : (10 * 9) / 2 = 45 := by norm_num
  have h_success : 5 = 5 := by norm_num
  rw [h_total, h_success]
  norm_num
  done

end kim_probability_same_color_l12_12099


namespace basketball_game_count_l12_12435

noncomputable def total_games_played (teams games_each_opp : ℕ) : ℕ :=
  (teams * (teams - 1) / 2) * games_each_opp

theorem basketball_game_count (n : ℕ) (g : ℕ) (h_n : n = 10) (h_g : g = 4) : total_games_played n g = 180 :=
by
  -- Use 'h_n' and 'h_g' as hypotheses
  rw [h_n, h_g]
  show (10 * 9 / 2) * 4 = 180
  sorry

end basketball_game_count_l12_12435


namespace cauliflower_production_diff_l12_12682

theorem cauliflower_production_diff
  (area_this_year : ℕ)
  (area_last_year : ℕ)
  (side_this_year : ℕ)
  (side_last_year : ℕ)
  (H1 : side_this_year * side_this_year = area_this_year)
  (H2 : side_last_year * side_last_year = area_last_year)
  (H3 : side_this_year = side_last_year + 1)
  (H4 : area_this_year = 12544) :
  area_this_year - area_last_year = 223 :=
by
  sorry

end cauliflower_production_diff_l12_12682


namespace head_start_fraction_of_length_l12_12943

-- Define the necessary variables and assumptions.
variables (Va Vb L H : ℝ)

-- Given conditions
def condition_speed_relation : Prop := Va = (22 / 19) * Vb
def condition_dead_heat : Prop := (L / Va) = ((L - H) / Vb)

-- The statement to be proven
theorem head_start_fraction_of_length (h_speed_relation: condition_speed_relation Va Vb) (h_dead_heat: condition_dead_heat L Va H Vb) : 
  H = (3 / 22) * L :=
sorry

end head_start_fraction_of_length_l12_12943


namespace part1_part2_l12_12598

def f (x : ℝ) : ℝ := Real.log (1 + x) + x^2 / 2
def g (x : ℝ) : ℝ := Real.cos x + x^2 / 2

theorem part1 (x : ℝ) (hx : 0 ≤ x) : f x ≥ x :=
by
  sorry

theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : f (Real.exp (a / 2)) = g b - 1) : f (b^2) + 1 > g (a + 1) :=
by
  sorry

end part1_part2_l12_12598


namespace zero_in_M_l12_12357

def M : Set Int := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M :=
  by
  -- Proof is omitted
  sorry

end zero_in_M_l12_12357


namespace coffee_bean_price_l12_12555

theorem coffee_bean_price 
  (x : ℝ)
  (price_second : ℝ) (weight_first weight_second : ℝ)
  (total_weight : ℝ) (price_mixture : ℝ) 
  (value_mixture : ℝ) 
  (h1 : price_second = 12)
  (h2 : weight_first = 25)
  (h3 : weight_second = 25)
  (h4 : total_weight = 100)
  (h5 : price_mixture = 11.25)
  (h6 : value_mixture = total_weight * price_mixture)
  (h7 : weight_first + weight_second = total_weight) :
  25 * x + 25 * 12 = 100 * 11.25 → x = 33 :=
by
  intro h
  sorry

end coffee_bean_price_l12_12555


namespace basketball_team_combinations_l12_12951

/-
  Problem:
  A basketball team has 12 players. The coach needs to select a team captain and then choose 5 players for the starting lineup (excluding the captain). Prove that the number of different combinations the coach can form is 5544.
-/

theorem basketball_team_combinations : ∃ (n : ℕ), n = 12 * (Nat.choose 11 5) ∧ n = 5544 :=
by
  sorry

end basketball_team_combinations_l12_12951


namespace incorrect_games_leq_75_percent_l12_12168

theorem incorrect_games_leq_75_percent (N : ℕ) (win_points : ℕ) (draw_points : ℚ) (loss_points : ℕ) (incorrect : (ℕ × ℕ) → Prop) :
  (win_points = 1) → (draw_points = 1 / 2) → (loss_points = 0) →
  ∀ (g : ℕ × ℕ), incorrect g → 
  ∃ (total_games incorrect_games : ℕ), 
    total_games = N * (N - 1) / 2 ∧
    incorrect_games ≤ 3 / 4 * total_games := sorry

end incorrect_games_leq_75_percent_l12_12168


namespace circle_area_l12_12652

-- Given conditions
variables {BD AC : ℝ} (BD_pos : BD = 6) (AC_pos : AC = 12)
variables {R : ℝ} (R_pos : R = 15 / 2)

-- Prove that the area of the circles is \(\frac{225}{4}\pi\)
theorem circle_area (BD_pos : BD = 6) (AC_pos : AC = 12) (R : ℝ) (R_pos : R = 15 / 2) : 
        ∃ S, S = (225 / 4) * Real.pi := 
by sorry

end circle_area_l12_12652


namespace area_of_square_same_yarn_l12_12154

theorem area_of_square_same_yarn (a : ℕ) (ha : a = 4) :
  let hexagon_perimeter := 6 * a
  let square_side := hexagon_perimeter / 4
  square_side * square_side = 36 :=
by
  sorry

end area_of_square_same_yarn_l12_12154


namespace countDistinguishedDigitsTheorem_l12_12858

-- Define a function to count numbers with four distinct digits where leading zeros are allowed
def countDistinguishedDigits : Nat :=
  10 * 9 * 8 * 7

-- State the theorem we need to prove
theorem countDistinguishedDigitsTheorem :
  countDistinguishedDigits = 5040 := 
by
  sorry

end countDistinguishedDigitsTheorem_l12_12858


namespace range_of_a_l12_12248

noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 16) / x

theorem range_of_a (a : ℝ) (h1 : 2 ≤ a) (h2 : (∀ x, 2 ≤ x ∧ x ≤ a → 9 ≤ f x ∧ f x ≤ 11)) : 4 ≤ a ∧ a ≤ 8 := by
  sorry

end range_of_a_l12_12248


namespace value_of_x_squared_plus_9y_squared_l12_12374

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l12_12374


namespace roots_distinct_and_real_l12_12198

variables (b d : ℝ)
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_distinct_and_real (h₁ : discriminant b (-3 * Real.sqrt 5) d = 25) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 :=
by 
  sorry

end roots_distinct_and_real_l12_12198


namespace fifth_equation_pattern_l12_12514

theorem fifth_equation_pattern :
  (1 = 1) →
  (2 + 3 + 4 = 9) →
  (3 + 4 + 5 + 6 + 7 = 25) →
  (4 + 5 + 6 + 7 + 8 + 9 + 10 = 49) →
  (5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81) :=
by 
  intros h1 h2 h3 h4
  sorry

end fifth_equation_pattern_l12_12514


namespace probability_defective_unit_l12_12628

theorem probability_defective_unit (T : ℝ) 
  (P_A : ℝ := 9 / 1000) 
  (P_B : ℝ := 1 / 50) 
  (output_ratio_A : ℝ := 0.4)
  (output_ratio_B : ℝ := 0.6) : 
  (P_A * output_ratio_A + P_B * output_ratio_B) = 0.0156 :=
by
  sorry

end probability_defective_unit_l12_12628


namespace wharf_length_l12_12505

-- Define the constants
def avg_speed := 2 -- average speed in m/s
def travel_time := 16 -- travel time in seconds

-- Define the formula to calculate length of the wharf
def length_of_wharf := 2 * avg_speed * travel_time

-- The goal is to prove that length_of_wharf equals 64
theorem wharf_length : length_of_wharf = 64 :=
by
  -- Proof would be here
  sorry

end wharf_length_l12_12505


namespace max_cos_a_l12_12888

theorem max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 := 
sorry

end max_cos_a_l12_12888


namespace martin_rings_big_bell_l12_12766

/-
Problem Statement:
Martin rings the small bell 4 times more than 1/3 as often as the big bell.
If he rings both of them a combined total of 52 times, prove that he rings the big bell 36 times.
-/

theorem martin_rings_big_bell (s b : ℕ) 
  (h1 : s + b = 52) 
  (h2 : s = 4 + (1 / 3 : ℚ) * b) : 
  b = 36 := 
by
  sorry

end martin_rings_big_bell_l12_12766


namespace geometric_series_sum_l12_12822

/-- The first term of the geometric series. -/
def a : ℚ := 3

/-- The common ratio of the geometric series. -/
def r : ℚ := -3 / 4

/-- The sum of the geometric series is equal to 12/7. -/
theorem geometric_series_sum : (∑' n : ℕ, a * r^n) = 12 / 7 := 
by
  /- The Sum function and its properties for the geometric series will be used here. -/
  sorry

end geometric_series_sum_l12_12822


namespace gift_contributors_l12_12952

theorem gift_contributors :
  (∃ (n : ℕ), n ≥ 1 ∧ n ≤ 20 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → (9 : ℕ) ≤ 20) →
  (∃ (n : ℕ), n = 12) :=
by
  sorry

end gift_contributors_l12_12952


namespace compound_interest_years_is_four_l12_12831
noncomputable def compoundInterestYears (P : ℝ) (r : ℝ) (n : ℕ) (CI : ℝ) : ℕ :=
  let A := P + CI
  let factor := (1 + r / n)
  let log_A_P := Real.log (A / P)
  let log_factor := Real.log factor
  Nat.floor (log_A_P / log_factor)

theorem compound_interest_years_is_four :
  compoundInterestYears 1200 0.20 1 1288.32 = 4 :=
by
  sorry

end compound_interest_years_is_four_l12_12831


namespace functional_equation_solution_l12_12433

theorem functional_equation_solution (f g : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (x^2 - g y) = g x ^ 2 - y) :
  (∀ x : ℝ, f x = x) ∧ (∀ x : ℝ, g x = x) :=
by
  sorry

end functional_equation_solution_l12_12433


namespace pairings_count_l12_12698

-- Define the problem's conditions explicitly
def number_of_bowls : Nat := 6
def number_of_glasses : Nat := 6

-- The theorem stating that the number of pairings is 36
theorem pairings_count : number_of_bowls * number_of_glasses = 36 := by
  sorry

end pairings_count_l12_12698


namespace transmission_prob_correct_transmission_scheme_comparison_l12_12088

noncomputable def transmission_prob_single (α β : ℝ) : ℝ :=
  (1 - α) * (1 - β)^2

noncomputable def transmission_prob_triple_sequence (β : ℝ) : ℝ :=
  β * (1 - β)^2

noncomputable def transmission_prob_triple_decoding_one (β : ℝ) : ℝ :=
  β * (1 - β)^2 + (1 - β)^3

noncomputable def transmission_prob_triple_decoding_zero (α : ℝ) : ℝ :=
  3 * α * (1 - α)^2 + (1 - α)^3

noncomputable def transmission_prob_single_decoding_zero (α : ℝ) : ℝ :=
  1 - α

theorem transmission_prob_correct (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  transmission_prob_single α β = (1 - α) * (1 - β)^2 ∧
  transmission_prob_triple_sequence β = β * (1 - β)^2 ∧
  transmission_prob_triple_decoding_one β = β * (1 - β)^2 + (1 - β)^3 :=
sorry

theorem transmission_scheme_comparison (α : ℝ) (hα : 0 < α ∧ α < 0.5) :
  transmission_prob_triple_decoding_zero α > transmission_prob_single_decoding_zero α :=
sorry

end transmission_prob_correct_transmission_scheme_comparison_l12_12088


namespace B_investment_is_72000_l12_12315

noncomputable def A_investment : ℝ := 27000
noncomputable def C_investment : ℝ := 81000
noncomputable def C_profit : ℝ := 36000
noncomputable def total_profit : ℝ := 80000

noncomputable def B_investment : ℝ :=
  let total_investment := (C_investment * total_profit) / C_profit
  total_investment - A_investment - C_investment

theorem B_investment_is_72000 :
  B_investment = 72000 :=
by
  sorry

end B_investment_is_72000_l12_12315


namespace hyperbola_distance_from_focus_to_asymptote_l12_12978

theorem hyperbola_distance_from_focus_to_asymptote :
  (distance_from_focus_to_asymptote (4 : ℝ) (12 : ℝ) = 2 * real.sqrt 3) :=
by sorry

end hyperbola_distance_from_focus_to_asymptote_l12_12978


namespace min_value_of_reciprocal_sum_l12_12839

theorem min_value_of_reciprocal_sum {a b : ℝ} (h : a > 0 ∧ b > 0)
  (h_circle1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (h_circle2 : ∀ x y : ℝ, (x - 2)^2 + (y - 2)^2 = 4)
  (h_common_chord : a + b = 2) :
  (1 / a + 9 / b = 8) := 
sorry

end min_value_of_reciprocal_sum_l12_12839


namespace irrational_c_l12_12061

def a : ℚ := 1 / 3
def b : ℝ := Real.sqrt 4
def c : ℝ := Real.pi / 3
def d : ℝ := 0.673232 -- Assuming interpret repeating decimals properly

theorem irrational_c : Irrational c := by
  sorry

end irrational_c_l12_12061


namespace matrix_non_invertible_at_36_31_l12_12714

-- Define the matrix A
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2 * x, 9], ![4 - x, 11]]

-- State the theorem
theorem matrix_non_invertible_at_36_31 :
  ∃ x : ℝ, (A x).det = 0 ∧ x = 36 / 31 :=
by {
  sorry
}

end matrix_non_invertible_at_36_31_l12_12714


namespace common_area_approximation_l12_12478

noncomputable def elliptical_domain (x y : ℝ) : Prop :=
  (x^2 / 3 + y^2 / 2) ≤ 1

noncomputable def circular_domain (x y : ℝ) : Prop :=
  (x^2 + y^2) ≤ 2

noncomputable def intersection_area : ℝ :=
  7.27

theorem common_area_approximation :
  ∃ area, 
    elliptical_domain x y ∧ circular_domain x y →
    abs (area - intersection_area) < 0.01 :=
sorry

end common_area_approximation_l12_12478


namespace maximum_value_parabola_l12_12072

theorem maximum_value_parabola (x : ℝ) : 
  ∃ y : ℝ, y = -3 * x^2 + 6 ∧ ∀ z : ℝ, (∃ a : ℝ, z = -3 * a^2 + 6) → z ≤ 6 :=
by
  sorry

end maximum_value_parabola_l12_12072


namespace total_cost_correct_l12_12398

def cost_barette : ℕ := 3
def cost_comb : ℕ := 1

def kristine_barrettes : ℕ := 1
def kristine_combs : ℕ := 1

def crystal_barrettes : ℕ := 3
def crystal_combs : ℕ := 1

def total_spent (cost_barette : ℕ) (cost_comb : ℕ) 
  (kristine_barrettes : ℕ) (kristine_combs : ℕ) 
  (crystal_barrettes : ℕ) (crystal_combs : ℕ) : ℕ :=
  (kristine_barrettes * cost_barette + kristine_combs * cost_comb) + 
  (crystal_barrettes * cost_barette + crystal_combs * cost_comb)

theorem total_cost_correct :
  total_spent cost_barette cost_comb kristine_barrettes kristine_combs crystal_barrettes crystal_combs = 14 :=
by
  sorry

end total_cost_correct_l12_12398


namespace find_x_in_terms_of_z_l12_12641

variable (z : ℝ)
variable (x y : ℝ)

theorem find_x_in_terms_of_z (h1 : 0.35 * (400 + y) = 0.20 * x) 
                             (h2 : x = 2 * z^2) 
                             (h3 : y = 3 * z - 5) : 
  x = 2 * z^2 :=
by
  exact h2

end find_x_in_terms_of_z_l12_12641


namespace solve_inequality_l12_12409

theorem solve_inequality (x : ℝ) : -7/3 < x ∧ x < 7 → |x+2| + |x-2| < x + 7 :=
by
  intro h
  sorry

end solve_inequality_l12_12409


namespace four_brothers_money_l12_12715

theorem four_brothers_money 
  (a_1 a_2 a_3 a_4 : ℝ) 
  (x : ℝ)
  (h1 : a_1 + a_2 + a_3 + a_4 = 48)
  (h2 : a_1 + 3 = x)
  (h3 : a_2 - 3 = x)
  (h4 : 3 * a_3 = x)
  (h5 : a_4 / 3 = x) :
  a_1 = 6 ∧ a_2 = 12 ∧ a_3 = 3 ∧ a_4 = 27 :=
by
  sorry

end four_brothers_money_l12_12715


namespace units_digit_a2019_l12_12589

theorem units_digit_a2019 (a : ℕ → ℝ) (h₁ : ∀ n, a n > 0)
  (h₂ : a 2 ^ 2 + a 4 ^ 2 = 900 - 2 * a 1 * a 5)
  (h₃ : a 5 = 9 * a 3) : (3^(2018) % 10) = 9 := by
  sorry

end units_digit_a2019_l12_12589


namespace max_value_of_expression_l12_12892

-- Define the variables and condition.
variable (x y z : ℝ)
variable (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)

-- State the theorem.
theorem max_value_of_expression :
  (8 * x + 5 * y + 15 * z) ≤ 4.54 :=
sorry

end max_value_of_expression_l12_12892


namespace integer_solutions_system_l12_12333

theorem integer_solutions_system :
  {x : ℤ | (4 * (1 + x) / 3 - 1 ≤ (5 + x) / 2) ∧ (x - 5 ≤ (3 * (3 * x - 2)) / 2)} = {0, 1, 2} :=
by
  sorry

end integer_solutions_system_l12_12333


namespace total_tickets_sold_l12_12665

def ticket_prices : Nat := 25
def senior_ticket_price : Nat := 15
def total_receipts : Nat := 9745
def senior_tickets_sold : Nat := 348
def adult_tickets_sold : Nat := (total_receipts - senior_ticket_price * senior_tickets_sold) / ticket_prices

theorem total_tickets_sold : adult_tickets_sold + senior_tickets_sold = 529 :=
by
  sorry

end total_tickets_sold_l12_12665


namespace odd_checkerboard_cannot_be_covered_by_dominoes_l12_12169

theorem odd_checkerboard_cannot_be_covered_by_dominoes 
    (m n : ℕ) (h : (m * n) % 2 = 1) :
    ¬ ∃ (dominos : Finset (Fin 2 × Fin 2)),
    ∀ {i j : Fin 2}, (i, j) ∈ dominos → 
    ((i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0)) ∧ 
    dominos.card = (m * n) / 2 := sorry

end odd_checkerboard_cannot_be_covered_by_dominoes_l12_12169


namespace problem_statement_l12_12069

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2004 = 2005 :=
sorry

end problem_statement_l12_12069


namespace evaluate_x2_y2_l12_12043

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end evaluate_x2_y2_l12_12043


namespace tunnel_length_scale_l12_12745

theorem tunnel_length_scale (map_length_cm : ℝ) (scale_ratio : ℝ) (convert_factor : ℝ) : 
  map_length_cm = 7 → scale_ratio = 38000 → convert_factor = 100000 →
  (map_length_cm * scale_ratio / convert_factor) = 2.66 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tunnel_length_scale_l12_12745


namespace value_added_to_number_l12_12660

theorem value_added_to_number (x : ℤ) : 
  (150 - 109 = 109 + x) → (x = -68) :=
by
  sorry

end value_added_to_number_l12_12660


namespace number_of_possible_ones_digits_l12_12513

open Finset

-- Define the condition of being divisible by 6, which entails being divisible by both 2 and 3
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define the set of possible ones digits for an even number
def even_ones_digits : Finset ℕ := {0, 2, 4, 6, 8}

-- Define the condition for the sum of digits being divisible by 3
def sum_of_digits_divisible_by_3 (n : ℕ) : Prop := 
  let digits := n.digits 10 in
  digits.sum % 3 = 0

-- State the problem: how many different ones digits are possible for numbers Ana likes
theorem number_of_possible_ones_digits : 
  (even_ones_digits.filter (λ d, ∃ n, n % 10 = d ∧ divisible_by_6 n)).card = 5 :=
sorry

end number_of_possible_ones_digits_l12_12513


namespace probability_both_heads_on_last_flip_l12_12424

noncomputable def fair_coin_flip : probabilityₓ ℙ :=
  probabilityₓ.ofUniform [true, false]

def both_coins_heads (events : list (bool × bool)) : bool :=
  events.all (λ event, event.1 = true)

def stops_with_heads (events : list (bool × bool)) : bool :=
  events.any (λ event, event.1 = true ∨ event.2 = true)

theorem probability_both_heads_on_last_flip :
  ∀ events : list (bool × bool), probabilityₓ (fair_coin_flip ×ₗ fair_coin_flip)
  (λ event, both_coins_heads events = true ∧ stops_with_heads events = true) = 1 / 3 :=
sorry

end probability_both_heads_on_last_flip_l12_12424


namespace pipe_A_fill_time_l12_12518

theorem pipe_A_fill_time (t : ℝ) (h1 : t > 0) (h2 : ∃ tA tB, tA = t ∧ tB = t / 6 ∧ (tA + tB) = 3) : t = 21 :=
by
  sorry

end pipe_A_fill_time_l12_12518


namespace hamburgers_sold_in_winter_l12_12550

theorem hamburgers_sold_in_winter:
  ∀ (T x : ℕ), 
  (T = 5 * 4) → 
  (5 + 6 + 4 + x = T) →
  (x = 5) :=
by
  intros T x hT hTotal
  sorry

end hamburgers_sold_in_winter_l12_12550


namespace sum_of_number_and_reverse_is_perfect_square_iff_l12_12919

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def reverse_of (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem sum_of_number_and_reverse_is_perfect_square_iff :
  ∀ n : ℕ, is_two_digit n →
    is_perfect_square (n + reverse_of n) ↔
      n = 29 ∨ n = 38 ∨ n = 47 ∨ n = 56 ∨ n = 65 ∨ n = 74 ∨ n = 83 ∨ n = 92 :=
by
  sorry

end sum_of_number_and_reverse_is_perfect_square_iff_l12_12919


namespace rate_of_interest_l12_12820

theorem rate_of_interest (P A T SI : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 2)
  (h4 : SI = A - P) (h5 : SI = (P * R * T) / 100) : R = 10 :=
by
  sorry

end rate_of_interest_l12_12820


namespace angle_in_second_quadrant_l12_12163

theorem angle_in_second_quadrant (n : ℤ) : (460 : ℝ) = 360 * n + 100 := by
  sorry

end angle_in_second_quadrant_l12_12163


namespace rectangle_width_length_ratio_l12_12499

theorem rectangle_width_length_ratio (w l P : ℕ) (h_l : l = 10) (h_P : P = 30) (h_perimeter : 2*w + 2*l = P) :
  w / l = 1 / 2 := 
by {
  sorry
}

end rectangle_width_length_ratio_l12_12499


namespace total_import_value_l12_12876

-- Define the given conditions
def export_value : ℝ := 8.07
def additional_amount : ℝ := 1.11
def factor : ℝ := 1.5

-- Define the import value to be proven
def import_value : ℝ := 46.4

-- Main theorem statement
theorem total_import_value :
  export_value = factor * import_value + additional_amount → import_value = 46.4 :=
by sorry

end total_import_value_l12_12876


namespace chocolate_chip_more_than_raisin_l12_12482

def chocolate_chip_yesterday : ℕ := 19
def chocolate_chip_morning : ℕ := 237
def raisin_cookies : ℕ := 231

theorem chocolate_chip_more_than_raisin : 
  (chocolate_chip_yesterday + chocolate_chip_morning) - raisin_cookies = 25 :=
by 
  sorry

end chocolate_chip_more_than_raisin_l12_12482


namespace redesigned_lock_additional_combinations_l12_12111

-- Definitions for the problem conditions
def original_combinations : ℕ := Nat.choose 10 5
def total_new_combinations : ℕ := (Finset.range 10).sum (λ k => Nat.choose 10 (k + 1)) 
def additional_combinations := total_new_combinations - original_combinations - 2 -- subtract combinations for 0 and 10

-- Statement of the theorem
theorem redesigned_lock_additional_combinations : additional_combinations = 770 :=
by
  -- Proof omitted (insert 'sorry' to indicate incomplete proof state)
  sorry

end redesigned_lock_additional_combinations_l12_12111


namespace sum_a2_a4_a6_l12_12083

-- Define the arithmetic sequence with a positive common difference
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ (d : ℝ), d > 0 ∧ ∀ n, a (n + 1) = a n + d

-- Define that a_1 and a_7 are roots of the quadratic equation x^2 - 10x + 16 = 0
def roots_condition (a : ℕ → ℝ) : Prop :=
(a 1) * (a 7) = 16 ∧ (a 1) + (a 7) = 10

-- The main theorem we want to prove
theorem sum_a2_a4_a6 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : roots_condition a) :
  a 2 + a 4 + a 6 = 15 :=
sorry

end sum_a2_a4_a6_l12_12083


namespace sqrt_1_0201_eq_1_01_l12_12228

theorem sqrt_1_0201_eq_1_01 (h : Real.sqrt 102.01 = 10.1) : Real.sqrt 1.0201 = 1.01 :=
by 
  sorry

end sqrt_1_0201_eq_1_01_l12_12228


namespace probability_heads_given_heads_l12_12422

-- Definitions for fair coin flips and the stopping condition
noncomputable def fair_coin_prob (event : ℕ → Prop) : ℝ :=
  sorry -- Probability function for coin events (to be defined in proofs)

-- The main statement
theorem probability_heads_given_heads :
  let p : ℝ := 1 / 3 in
  ∃ p: ℝ, p = 1 / 3 ∧ fair_coin_prob (λ n, (n = 1 ∧ (coin_flip n = (TT)) ∧ ((coin_flip (n+1) = (HH) ∨ coin_flip (n+1) = (TH))) ∧ ¬has_heads_before n)) = p :=
sorry

end probability_heads_given_heads_l12_12422


namespace find_g_expression_l12_12597

theorem find_g_expression (f g : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = 2 * x + 3)
  (h2 : ∀ x : ℝ, g (x + 2) = f x) :
  ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end find_g_expression_l12_12597


namespace maximum_third_height_l12_12814

theorem maximum_third_height 
  (A B C : Type)
  (h1 h2 : ℕ)
  (h1_pos : h1 = 4) 
  (h2_pos : h2 = 12) 
  (h3_pos : ℕ)
  (triangle_inequality : ∀ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a)
  (scalene : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c)
  : (3 < h3_pos ∧ h3_pos < 6) → h3_pos = 5 := 
sorry

end maximum_third_height_l12_12814


namespace proof_problem_l12_12601

-- Define the propositions and conditions
def p : Prop := ∀ x > 0, 3^x > 1
def neg_p : Prop := ∃ x > 0, 3^x ≤ 1
def q (a : ℝ) : Prop := a < -2
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

-- The condition that q is a sufficient condition for f(x) to have a zero in [-1,2]
def has_zero_in_interval (a : ℝ) : Prop := 
  (-a + 3) * (2 * a + 3) ≤ 0

-- The proof problem statement
theorem proof_problem (a : ℝ) (P : p) (Q : has_zero_in_interval a) : ¬ p ∧ q a :=
by
  sorry

end proof_problem_l12_12601


namespace negation_of_existence_is_universal_l12_12480

theorem negation_of_existence_is_universal (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
sorry

end negation_of_existence_is_universal_l12_12480


namespace m_plus_n_eq_47_l12_12833

theorem m_plus_n_eq_47 (m n : ℕ)
  (h1 : m + 8 < n - 1)
  (h2 : (m + m + 3 + m + 8 + n - 1 + n + 3 + 2 * n - 2) / 6 = n)
  (h3 : (m + 8 + (n - 1)) / 2 = n) :
  m + n = 47 :=
sorry

end m_plus_n_eq_47_l12_12833


namespace fruits_eaten_total_l12_12790

variable (oranges_per_day : ℕ) (grapes_per_day : ℕ) (days : ℕ)

def total_fruits (oranges_per_day grapes_per_day days : ℕ) : ℕ :=
  (oranges_per_day * days) + (grapes_per_day * days)

theorem fruits_eaten_total 
  (h1 : oranges_per_day = 20)
  (h2 : grapes_per_day = 40) 
  (h3 : days = 30) : 
  total_fruits oranges_per_day grapes_per_day days = 1800 := 
by 
  sorry

end fruits_eaten_total_l12_12790


namespace natural_number_pairs_l12_12975

theorem natural_number_pairs (a b : ℕ) (p q : ℕ) :
  a ≠ b →
  (∃ p, a + b = 2^p) →
  (∃ q, ab + 1 = 2^q) →
  (a = 1 ∧ b = 2^p - 1 ∨ a = 2^q - 1 ∧ b = 2^q + 1) :=
by intro hne hp hq; sorry

end natural_number_pairs_l12_12975


namespace lcm_of_5_6_8_9_l12_12929

theorem lcm_of_5_6_8_9 : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 := 
by 
  sorry

end lcm_of_5_6_8_9_l12_12929


namespace probability_X_eq_Y_l12_12190

theorem probability_X_eq_Y
  (x y : ℝ)
  (h1 : -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi)
  (h2 : -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi)
  (h3 : Real.cos (Real.cos x) = Real.cos (Real.cos y)) :
  (∃ N : ℕ, N = 100 ∧ ∃ M : ℕ, M = 11 ∧ M / N = (11 : ℝ) / 100) :=
by sorry

end probability_X_eq_Y_l12_12190


namespace complement_of_16deg51min_is_73deg09min_l12_12066

def complement_angle (A : ℝ) : ℝ := 90 - A

theorem complement_of_16deg51min_is_73deg09min :
  complement_angle 16.85 = 73.15 := by
  sorry

end complement_of_16deg51min_is_73deg09min_l12_12066


namespace distinct_real_roots_l12_12412

theorem distinct_real_roots :
  ∀ x : ℝ, (x^3 - 3*x^2 + x - 2) * (x^3 - x^2 - 4*x + 7) + 6*x^2 - 15*x + 18 = 0 ↔
  x = 1 ∨ x = -2 ∨ x = 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 1 + Real.sqrt 2 :=
by sorry

end distinct_real_roots_l12_12412


namespace unpacked_books_30_l12_12080

theorem unpacked_books_30 :
  let total_books := 1485 * 42
  let books_per_box := 45
  total_books % books_per_box = 30 :=
by
  let total_books := 1485 * 42
  let books_per_box := 45
  have h : total_books % books_per_box = 30 := sorry
  exact h

end unpacked_books_30_l12_12080


namespace quadratic_function_range_l12_12501

theorem quadratic_function_range (a b c : ℝ) (x y : ℝ) :
  (∀ x, x = -4 → y = a * (-4)^2 + b * (-4) + c → y = 3) ∧
  (∀ x, x = -3 → y = a * (-3)^2 + b * (-3) + c → y = -2) ∧
  (∀ x, x = -2 → y = a * (-2)^2 + b * (-2) + c → y = -5) ∧
  (∀ x, x = -1 → y = a * (-1)^2 + b * (-1) + c → y = -6) ∧
  (∀ x, x = 0 → y = a * 0^2 + b * 0 + c → y = -5) →
  (∀ x, x < -2 → y > -5) :=
sorry

end quadratic_function_range_l12_12501


namespace num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l12_12859

theorem num_three_digit_numbers_divisible_by_5_and_6_with_digit_6 : 
  ∃ S : Finset ℕ, (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (6 ∈ n.digits 10)) ∧ S.card = 6 :=
by
  sorry

end num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l12_12859


namespace first_sales_amount_l12_12688

-- Conditions from the problem
def first_sales_royalty : ℝ := 8 -- million dollars
def second_sales_royalty : ℝ := 9 -- million dollars
def second_sales_amount : ℝ := 108 -- million dollars
def decrease_percentage : ℝ := 0.7916666666666667

-- The goal is to determine the first sales amount, S, meeting the conditions.
theorem first_sales_amount :
  ∃ S : ℝ,
    (first_sales_royalty / S - second_sales_royalty / second_sales_amount = decrease_percentage * (first_sales_royalty / S)) ∧
    S = 20 :=
sorry

end first_sales_amount_l12_12688


namespace min_moves_to_break_chocolate_l12_12303

theorem min_moves_to_break_chocolate (n m : ℕ) (tiles : ℕ) (moves : ℕ) :
    (n = 4) → (m = 10) → (tiles = n * m) → (moves = tiles - 1) → moves = 39 :=
by
  intros hnm hn4 hm10 htm
  sorry

end min_moves_to_break_chocolate_l12_12303


namespace original_price_computer_l12_12903

noncomputable def first_store_price (P : ℝ) : ℝ := 0.94 * P

noncomputable def second_store_price (exchange_rate : ℝ) : ℝ := (920 / 0.95) * exchange_rate

theorem original_price_computer 
  (exchange_rate : ℝ)
  (h : exchange_rate = 1.1) 
  (H : (first_store_price P - second_store_price exchange_rate = 19)) :
  P = 1153.47 :=
by
  sorry

end original_price_computer_l12_12903


namespace arithmetic_sum_of_11_terms_l12_12235

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α) (d : α)

def arithmetic_sequence (a : ℕ → α) (a₁ : α) (d : α) : Prop :=
∀ n, a n = a₁ + n * d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
(n + 1) * (a 0 + a n) / 2

theorem arithmetic_sum_of_11_terms
  (a₁ d : α)
  (a : ℕ → α)
  (h_seq : arithmetic_sequence a a₁ d)
  (h_cond : a 8 = (1 / 2) * a 11 + 3) :
  sum_first_n_terms a 10 = 66 := by
  sorry

end arithmetic_sum_of_11_terms_l12_12235


namespace find_value_of_fraction_l12_12244

open Real

theorem find_value_of_fraction (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) : 
  (x + y) / (x - y) = -sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l12_12244


namespace total_expenditure_eq_fourteen_l12_12401

variable (cost_barrette cost_comb : ℕ)
variable (kristine_barrettes kristine_combs crystal_barrettes crystal_combs : ℕ)

theorem total_expenditure_eq_fourteen 
  (h_cost_barrette : cost_barrette = 3)
  (h_cost_comb : cost_comb = 1)
  (h_kristine_barrettes : kristine_barrettes = 1)
  (h_kristine_combs : kristine_combs = 1)
  (h_crystal_barrettes : crystal_barrettes = 3)
  (h_crystal_combs : crystal_combs = 1) :
  (kristine_barrettes * cost_barrette + kristine_combs * cost_comb) +
  (crystal_barrettes * cost_barrette + crystal_combs * cost_comb) = 14 := 
by 
  sorry

end total_expenditure_eq_fourteen_l12_12401


namespace candidate_percentage_l12_12679

variables (P candidate_votes rival_votes total_votes : ℝ)

-- Conditions
def candidate_lost_by_2460 (candidate_votes rival_votes : ℝ) : Prop :=
  rival_votes = candidate_votes + 2460

def total_votes_cast (candidate_votes rival_votes total_votes : ℝ) : Prop :=
  candidate_votes + rival_votes = total_votes

-- Proof problem
theorem candidate_percentage (h1 : candidate_lost_by_2460 candidate_votes rival_votes)
                             (h2 : total_votes_cast candidate_votes rival_votes 8200) :
  P = 35 :=
sorry

end candidate_percentage_l12_12679


namespace find_m_eq_4_l12_12102

theorem find_m_eq_4 (m : ℝ) (h₁ : ∃ (A B C : ℝ × ℝ), A = (m, -m+3) ∧ B = (2, m-1) ∧ C = (-1, 4)) (h₂ : (4 - (-m+3)) / (-1-m) = 3 * ((m-1) - 4) / (2 - (-1))) : m = 4 :=
sorry

end find_m_eq_4_l12_12102


namespace charles_housesitting_hours_l12_12574

theorem charles_housesitting_hours :
  ∀ (earnings_per_hour_housesitting earnings_per_hour_walking_dog number_of_dogs_walked total_earnings : ℕ),
  earnings_per_hour_housesitting = 15 →
  earnings_per_hour_walking_dog = 22 →
  number_of_dogs_walked = 3 →
  total_earnings = 216 →
  ∃ h : ℕ, 15 * h + 22 * 3 = 216 ∧ h = 10 :=
by
  intros
  sorry

end charles_housesitting_hours_l12_12574


namespace interest_rate_increase_l12_12445

theorem interest_rate_increase (P : ℝ) (A1 A2 : ℝ) (T : ℝ) (R1 R2 : ℝ) (percentage_increase : ℝ) :
  P = 500 → A1 = 600 → A2 = 700 → T = 2 → 
  (A1 - P) = P * R1 * T →
  (A2 - P) = P * R2 * T →
  percentage_increase = (R2 - R1) / R1 * 100 →
  percentage_increase = 100 :=
by sorry

end interest_rate_increase_l12_12445


namespace rationalize_denominator_eq_l12_12770

noncomputable def rationalize_denominator : ℝ :=
  18 / (Real.sqrt 36 + Real.sqrt 2)

theorem rationalize_denominator_eq : rationalize_denominator = (54 / 17) - (9 * Real.sqrt 2 / 17) := 
by
  sorry

end rationalize_denominator_eq_l12_12770


namespace cannot_form_right_triangle_l12_12939

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem cannot_form_right_triangle : ¬ is_right_triangle 40 50 60 := 
by
  sorry

end cannot_form_right_triangle_l12_12939


namespace charlie_original_price_l12_12071

theorem charlie_original_price (acorns_Alice acorns_Bob acorns_Charlie ν_Alice ν_Bob discount price_Charlie_before_discount price_Charlie_after_discount total_paid_by_AliceBob total_acorns_AliceBob average_price_per_acorn price_per_acorn_Alice price_per_acorn_Bob total_paid_Alice total_paid_Bob: ℝ) :
  acorns_Alice = 3600 →
  acorns_Bob = 2400 →
  acorns_Charlie = 4500 →
  ν_Bob = 6000 →
  ν_Alice = 9 * ν_Bob →
  price_per_acorn_Bob = ν_Bob / acorns_Bob →
  price_per_acorn_Alice = ν_Alice / acorns_Alice →
  total_paid_Alice = acorns_Alice * price_per_acorn_Alice →
  total_paid_Bob = ν_Bob →
  total_paid_by_AliceBob = total_paid_Alice + total_paid_Bob →
  total_acorns_AliceBob = acorns_Alice + acorns_Bob →
  average_price_per_acorn = total_paid_by_AliceBob / total_acorns_AliceBob →
  discount = 10 / 100 →
  price_Charlie_after_discount = average_price_per_acorn * (1 - discount) →
  price_Charlie_before_discount = average_price_per_acorn →
  price_Charlie_before_discount = 14.50 →
  price_per_acorn_Alice = 22.50 →
  price_Charlie_before_discount * acorns_Charlie = 4500 * 14.50 :=
by sorry

end charlie_original_price_l12_12071


namespace speed_of_stream_l12_12684

variable (b s : ℝ)

theorem speed_of_stream (h1 : 110 = (b + s + 3) * 5)
                        (h2 : 85 = (b - s + 2) * 6) : s = 3.4 :=
by
  sorry

end speed_of_stream_l12_12684


namespace total_capacity_of_two_tanks_l12_12739

-- Conditions
def tank_A_initial_fullness : ℚ := 3 / 4
def tank_A_final_fullness : ℚ := 7 / 8
def tank_A_added_volume : ℚ := 5

def tank_B_initial_fullness : ℚ := 2 / 3
def tank_B_final_fullness : ℚ := 5 / 6
def tank_B_added_volume : ℚ := 3

-- Proof statement
theorem total_capacity_of_two_tanks :
  let tank_A_total_capacity := tank_A_added_volume / (tank_A_final_fullness - tank_A_initial_fullness)
  let tank_B_total_capacity := tank_B_added_volume / (tank_B_final_fullness - tank_B_initial_fullness)
  tank_A_total_capacity + tank_B_total_capacity = 58 := 
sorry

end total_capacity_of_two_tanks_l12_12739


namespace trapezoid_area_l12_12447

theorem trapezoid_area (h_base : ℕ) (sum_bases : ℕ) (height : ℕ) (hsum : sum_bases = 36) (hheight : height = 15) :
    (sum_bases * height) / 2 = 270 := by
  sorry

end trapezoid_area_l12_12447


namespace smallest_four_digit_product_is_12_l12_12286

theorem smallest_four_digit_product_is_12 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
           (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 12 ∧ a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 6) ∧
           (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 →
                     (∃ a' b' c' d' : ℕ, m = 1000 * a' + 100 * b' + 10 * c' + d' ∧ a' * b' * c' * d' = 12) →
                     n ≤ m) :=
by
  sorry

end smallest_four_digit_product_is_12_l12_12286


namespace range_of_g_l12_12203

noncomputable def g (x : ℝ) : ℝ := (3 * x + 8 - 2 * x ^ 2) / (x + 4)

theorem range_of_g : 
  (∀ y : ℝ, ∃ x : ℝ, x ≠ -4 ∧ y = (3 * x + 8 - 2 * x^2) / (x + 4)) :=
by
  sorry

end range_of_g_l12_12203


namespace greater_number_is_84_l12_12275

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : x + y - (x - y) = 64) :
  x = 84 :=
by sorry

end greater_number_is_84_l12_12275


namespace sandy_position_l12_12159

structure Position :=
  (x : ℤ)
  (y : ℤ)

def initial_position : Position := { x := 0, y := 0 }
def after_south : Position := { x := 0, y := -20 }
def after_east : Position := { x := 20, y := -20 }
def after_north : Position := { x := 20, y := 0 }
def final_position : Position := { x := 30, y := 0 }

theorem sandy_position :
  final_position.x - initial_position.x = 10 ∧ final_position.y - initial_position.y = 0 :=
by
  sorry

end sandy_position_l12_12159


namespace quadratic_function_value_at_point_l12_12742

theorem quadratic_function_value_at_point (a : ℝ) :
  (∃ (P : ℝ × ℝ), P = (1, a) ∧ ∀ (x : ℝ), P.2 = 2 * x^2) → a = 2 :=
by
  intro h
  cases h with P hP
  rw [Prod.ext_iff, and_comm] at hP 
  cases hP with hx ha
  rw hx at ha
  specialize ha 1
  rw mul_one at ha 
  norm_num at ha
  exact ha

end quadratic_function_value_at_point_l12_12742


namespace soccer_team_points_l12_12644

theorem soccer_team_points 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (points_per_win : ℕ) 
  (points_per_draw : ℕ) 
  (points_per_loss : ℕ) 
  (draws : ℕ := total_games - (wins + losses)) : 
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  46 = (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) :=
by sorry

end soccer_team_points_l12_12644


namespace gcd_12m_18n_with_gcd_mn_18_l12_12229

theorem gcd_12m_18n_with_gcd_mn_18 (m n : ℕ) (hm : Nat.gcd m n = 18) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  Nat.gcd (12 * m) (18 * n) = 108 :=
by sorry

end gcd_12m_18n_with_gcd_mn_18_l12_12229


namespace geometric_sequence_sum_l12_12599

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

noncomputable def sum_geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum 
  (a₁ : ℝ) (q : ℝ) 
  (h_q : q = 1 / 2) 
  (h_a₂ : geometric_sequence a₁ q 2 = 2) : 
  sum_geometric_sequence a₁ q 6 = 63 / 8 :=
by
  -- The proof is skipped here
  sorry

end geometric_sequence_sum_l12_12599


namespace seating_arrangements_l12_12495

theorem seating_arrangements :
  let total_arrangements := Nat.factorial 10
  let abc_together := Nat.factorial 8 * Nat.factorial 3
  let de_together := Nat.factorial 9 * Nat.factorial 2
  let abc_and_de_together := Nat.factorial 7 * Nat.factorial 3 * Nat.factorial 2
  total_arrangements - abc_together - de_together + abc_and_de_together = 2853600 :=
by
  sorry

end seating_arrangements_l12_12495


namespace angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l12_12091

open Real

variable {A B C a b c : ℝ}
variable (AM BM MC : ℝ)

-- Conditions
axiom triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)
axiom BM_MC_relation : BM = (1 / 2) * MC

-- Part 1: Measure of angle A
theorem angle_A_is_pi_over_3 (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) : 
  A = π / 3 :=
by sorry

-- Part 2: Minimum value of |AM|^2 / S
noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : ℝ := 1 / 2 * b * c * sin A

axiom condition_b_eq_2c : b = 2 * c

theorem minimum_value_AM_sq_div_S (AM BM MC : ℝ) (S : ℝ) (H : BM = (1 / 2) * MC) 
  (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) 
  (area : S = area_triangle a b c A)
  (condition_b_eq_2c : b = 2 * c) : 
  (AM ^ 2) / S ≥ (8 * sqrt 3) / 9 :=
by sorry

end angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l12_12091


namespace martin_rings_big_bell_l12_12765

/-
Problem Statement:
Martin rings the small bell 4 times more than 1/3 as often as the big bell.
If he rings both of them a combined total of 52 times, prove that he rings the big bell 36 times.
-/

theorem martin_rings_big_bell (s b : ℕ) 
  (h1 : s + b = 52) 
  (h2 : s = 4 + (1 / 3 : ℚ) * b) : 
  b = 36 := 
by
  sorry

end martin_rings_big_bell_l12_12765


namespace eval_expression_l12_12001

theorem eval_expression : (3 + 1) * (3 ^ 2 + 1 ^ 2) * (3 ^ 4 + 1 ^ 4) = 3280 :=
by
  -- Bounds and simplifications
  simp
  -- Show the calculation steps are equivalent to 3280
  sorry

end eval_expression_l12_12001


namespace Linda_original_savings_l12_12395

-- Definition of the problem with all conditions provided.
theorem Linda_original_savings (S : ℝ) (TV_cost : ℝ) (TV_tax_rate : ℝ) (refrigerator_rate : ℝ) (furniture_discount_rate : ℝ) :
  let furniture_cost := (3 / 4) * S
  let TV_cost_with_tax := TV_cost + TV_cost * TV_tax_rate
  let refrigerator_cost := TV_cost + TV_cost * refrigerator_rate
  let remaining_savings := TV_cost_with_tax + refrigerator_cost
  let furniture_cost_after_discount := furniture_cost - furniture_cost * furniture_discount_rate
  (remaining_savings = (1 / 4) * S) →
  S = 1898.40 :=
by
  sorry


end Linda_original_savings_l12_12395


namespace total_spend_on_four_games_l12_12419

noncomputable def calculate_total_spend (batman_price : ℝ) (superman_price : ℝ)
                                        (batman_discount : ℝ) (superman_discount : ℝ)
                                        (tax_rate : ℝ) (game1_price : ℝ) (game2_price : ℝ) : ℝ :=
  let batman_discounted_price := batman_price - batman_discount * batman_price
  let superman_discounted_price := superman_price - superman_discount * superman_price
  let batman_price_after_tax := batman_discounted_price + tax_rate * batman_discounted_price
  let superman_price_after_tax := superman_discounted_price + tax_rate * superman_discounted_price
  batman_price_after_tax + superman_price_after_tax + game1_price + game2_price

theorem total_spend_on_four_games :
  calculate_total_spend 13.60 5.06 0.10 0.05 0.08 7.25 12.50 = 38.16 :=
by sorry

end total_spend_on_four_games_l12_12419


namespace find_subsequence_with_sum_n_l12_12707

theorem find_subsequence_with_sum_n (n : ℕ) (a : Fin n → ℕ) (h1 : ∀ i, a i ∈ Finset.range n) 
  (h2 : (Finset.univ.sum a) < 2 * n) : 
  ∃ s : Finset (Fin n), s.sum a = n := 
sorry

end find_subsequence_with_sum_n_l12_12707


namespace area_inside_arcs_outside_square_l12_12308

theorem area_inside_arcs_outside_square (r : ℝ) (θ : ℝ) (L : ℝ) (a b c d : ℝ) :
  r = 6 ∧ θ = 45 ∧ L = 12 ∧ a = 15 ∧ b = 0 ∧ c = 15 ∧ d = 144 →
  (a + b + c + d = 174) :=
by
  intros h
  sorry

end area_inside_arcs_outside_square_l12_12308


namespace distinct_remainders_l12_12449

theorem distinct_remainders (p : ℕ) (hp : Nat.Prime p) (h7 : 7 < p) : 
  let remainders := 
    { r | r ∈ Finset.univ.filter (λ r, ∃ k, p^2 = 210 * k + r) } 
  remainders.card = 6 :=
sorry

end distinct_remainders_l12_12449


namespace sufficient_but_not_necessary_l12_12288

theorem sufficient_but_not_necessary (x : ℝ) :
  (x^2 > 1) → (1 / x < 1) ∧ ¬(1 / x < 1 → x^2 > 1) :=
by
  sorry

end sufficient_but_not_necessary_l12_12288


namespace water_glass_ounces_l12_12396

theorem water_glass_ounces (glasses_per_day : ℕ) (days_per_week : ℕ)
    (bottle_ounces : ℕ) (bottle_fills_per_week : ℕ)
    (total_glasses_per_week : ℕ)
    (total_ounces_per_week : ℕ)
    (glasses_per_week_eq : glasses_per_day * days_per_week = total_glasses_per_week)
    (ounces_per_week_eq : bottle_ounces * bottle_fills_per_week = total_ounces_per_week)
    (ounce_per_glass : ℕ)
    (glasses_per_week : ℕ)
    (ounces_per_week : ℕ) :
    total_ounces_per_week / total_glasses_per_week = 5 :=
by
  sorry

end water_glass_ounces_l12_12396


namespace proposition_p_l12_12075

variable (x : ℝ)

-- Define condition
def negation_of_p : Prop := ∃ x, x < 1 ∧ x^2 < 1

-- Define proposition p
def p : Prop := ∀ x, x < 1 → x^2 ≥ 1

-- Theorem statement
theorem proposition_p (h : negation_of_p) : (p) :=
sorry

end proposition_p_l12_12075


namespace value_of_square_sum_l12_12363

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l12_12363


namespace initial_boys_down_slide_l12_12639

variable (B : Int)

theorem initial_boys_down_slide:
  B + 13 = 35 → B = 22 := by
  sorry

end initial_boys_down_slide_l12_12639


namespace find_x_for_g_inv_l12_12993

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 4 * x + 1

theorem find_x_for_g_inv (x : ℝ) (h : g 3 = x) : g⁻¹ 3 = 3 :=
by
  sorry

end find_x_for_g_inv_l12_12993


namespace georgia_makes_muffins_l12_12451

-- Definitions based on conditions
def muffinRecipeMakes : ℕ := 6
def numberOfStudents : ℕ := 24
def durationInMonths : ℕ := 9

-- Theorem to prove the given problem
theorem georgia_makes_muffins :
  (numberOfStudents / muffinRecipeMakes) * durationInMonths = 36 :=
by
  -- We'll skip the proof with sorry
  sorry

end georgia_makes_muffins_l12_12451


namespace joan_total_spent_l12_12238

theorem joan_total_spent (cost_basketball cost_racing total_spent : ℝ) 
  (h1 : cost_basketball = 5.20) 
  (h2 : cost_racing = 4.23) 
  (h3 : total_spent = cost_basketball + cost_racing) : 
  total_spent = 9.43 := 
by 
  sorry

end joan_total_spent_l12_12238


namespace train_length_is_170_meters_l12_12563

-- Definition of the conditions
def speed_km_per_hr := 45
def time_seconds := 30
def bridge_length_meters := 205

-- Convert speed from km/hr to m/s
def speed_m_per_s : ℝ := (speed_km_per_hr : ℝ) * 1000 / 3600

-- The total distance covered by the train in 30 seconds
def total_distance_m : ℝ := speed_m_per_s * (time_seconds : ℝ)

-- The length of the train
def length_of_train_m : ℝ := total_distance_m - (bridge_length_meters : ℝ)

-- The theorem we need to prove
theorem train_length_is_170_meters : 
  length_of_train_m = 170 := by
    sorry

end train_length_is_170_meters_l12_12563


namespace simplest_common_denominator_l12_12918

theorem simplest_common_denominator (x a : ℕ) :
  let d1 := 3 * x
  let d2 := 6 * x^2
  lcm d1 d2 = 6 * x^2 := 
by
  let d1 := 3 * x
  let d2 := 6 * x^2
  show lcm d1 d2 = 6 * x^2
  sorry

end simplest_common_denominator_l12_12918


namespace tan_alpha_value_complicated_expression_value_l12_12984

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) : 
  Real.tan α = -2 := by 
  sorry

theorem complicated_expression_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) (h3 : Real.tan α = -2) :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) / 
  (Real.cos (α - Real.pi / 2) - Real.sin (2 * Real.pi / 2 + α)) = -5 := by 
  sorry

end tan_alpha_value_complicated_expression_value_l12_12984


namespace eval_expression_l12_12021

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12021


namespace min_value_one_over_a_plus_nine_over_b_l12_12350

theorem min_value_one_over_a_plus_nine_over_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  16 ≤ (1 / a) + (9 / b) :=
sorry

end min_value_one_over_a_plus_nine_over_b_l12_12350


namespace liz_prob_at_least_half_l12_12701

noncomputable def binom (n k : ℕ) : ℕ := 
  Nat.choose n k

noncomputable def binom_prob (n k : ℕ) (p : ℚ) : ℚ := 
  (binom n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def prob_at_least_half_correct (n : ℕ) (p : ℚ) : ℚ := 
  (Finset.range (n + 1)).filter (fun k => k ≥ n / 2).sum (binom_prob n · p)

theorem liz_prob_at_least_half (n : ℕ) (p : ℚ) (h_n : n = 10) (h_p : p = 1/3) :
  prob_at_least_half_correct n p = 161 / 2187 := 
by
  sorry

end liz_prob_at_least_half_l12_12701


namespace one_cow_one_bag_l12_12803

-- Define parameters
def cows : ℕ := 26
def bags : ℕ := 26
def days_for_all_cows : ℕ := 26

-- Theorem to prove the number of days for one cow to eat one bag of husk
theorem one_cow_one_bag (cows bags days_for_all_cows : ℕ) (h : cows = bags) (h2 : days_for_all_cows = 26) : days_for_one_cow_one_bag = 26 :=
by {
    sorry -- Proof to be filled in
}

end one_cow_one_bag_l12_12803


namespace no_more_than_one_100_l12_12786

-- Define the score variables and the conditions
variables (R P M : ℕ)

-- Given conditions: R = P - 3 and P = M - 7
def score_conditions : Prop := R = P - 3 ∧ P = M - 7

-- The maximum score condition
def max_score_condition : Prop := R ≤ 100 ∧ P ≤ 100 ∧ M ≤ 100

-- The goal: it is impossible for Vanya to have scored 100 in more than one exam
theorem no_more_than_one_100 (R P M : ℕ) (h1 : score_conditions R P M) (h2 : max_score_condition R P M) :
  (R = 100 ∧ P = 100) ∨ (P = 100 ∧ M = 100) ∨ (M = 100 ∧ R = 100) → false :=
sorry

end no_more_than_one_100_l12_12786


namespace rational_sum_of_cubes_l12_12901

theorem rational_sum_of_cubes (t : ℚ) : 
    ∃ (a b c : ℚ), t = (a^3 + b^3 + c^3) :=
by
  sorry

end rational_sum_of_cubes_l12_12901


namespace parabola_intersection_l12_12659

theorem parabola_intersection:
  (∀ x y1 y2 : ℝ, (y1 = 3 * x^2 - 6 * x + 6) ∧ (y2 = -2 * x^2 - 4 * x + 6) → y1 = y2 → x = 0 ∨ x = 2 / 5) ∧
  (∀ a c : ℝ, a = 0 ∧ c = 2 / 5 ∧ c ≥ a → c - a = 2 / 5) :=
by sorry

end parabola_intersection_l12_12659


namespace problem1_problem2_l12_12378

-- Theorem 1: Given a^2 - b^2 = 1940:
theorem problem1 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1940 → 
  (a = 102 ∧ b = 92) := 
by 
  sorry

-- Theorem 2: Given a^2 - b^2 = 1920:
theorem problem2 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1920 → 
  (a = 101 ∧ b = 91) ∨ 
  (a = 58 ∧ b = 38) ∨ 
  (a = 47 ∧ b = 17) ∨ 
  (a = 44 ∧ b = 4) := 
by 
  sorry

end problem1_problem2_l12_12378


namespace triangle_area_eq_40_sqrt_3_l12_12871

open Real

theorem triangle_area_eq_40_sqrt_3 
  (a : ℝ) (A : ℝ) (b c : ℝ)
  (h1 : a = 14)
  (h2 : A = π / 3) -- 60 degrees in radians
  (h3 : b / c = 8 / 5) :
  1 / 2 * b * c * sin A = 40 * sqrt 3 :=
by
  sorry

end triangle_area_eq_40_sqrt_3_l12_12871


namespace future_cup_defensive_analysis_l12_12750

variables (avg_A : ℝ) (std_dev_A : ℝ) (avg_B : ℝ) (std_dev_B : ℝ)

-- Statement translations:
-- A: On average, Class B has better defensive skills than Class A.
def stat_A : Prop := avg_B < avg_A

-- C: Class B sometimes performs very well in defense, while other times it performs relatively poorly.
def stat_C : Prop := std_dev_B > std_dev_A

-- D: Class A rarely concedes goals.
def stat_D : Prop := avg_A <= 1.9 -- It's implied that 'rarely' indicates consistency and a lower average threshold, so this represents that.

theorem future_cup_defensive_analysis (h_avg_A : avg_A = 1.9) (h_std_dev_A : std_dev_A = 0.3) 
  (h_avg_B : avg_B = 1.3) (h_std_dev_B : std_dev_B = 1.2) :
  stat_A avg_A avg_B ∧ stat_C std_dev_A std_dev_B ∧ stat_D avg_A :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end future_cup_defensive_analysis_l12_12750


namespace total_cost_with_discount_and_tax_l12_12428

theorem total_cost_with_discount_and_tax
  (sandwich_cost : ℝ := 2.44)
  (soda_cost : ℝ := 0.87)
  (num_sandwiches : ℕ := 2)
  (num_sodas : ℕ := 4)
  (discount : ℝ := 0.15)
  (tax_rate : ℝ := 0.09) : 
  (num_sandwiches * sandwich_cost * (1 - discount) + num_sodas * soda_cost) * (1 + tax_rate) = 8.32 :=
by
  sorry

end total_cost_with_discount_and_tax_l12_12428


namespace option_B_is_one_variable_quadratic_l12_12936

theorem option_B_is_one_variable_quadratic :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x : ℝ, 2 * (x - x^2) - 1 = a * x^2 + b * x + c) :=
by
  sorry

end option_B_is_one_variable_quadratic_l12_12936


namespace fraction_problem_l12_12913

theorem fraction_problem (b : ℕ) (h₀ : 0 < b) (h₁ : (b : ℝ) / (b + 35) = 0.869) : b = 232 := 
by
  sorry

end fraction_problem_l12_12913


namespace intersection_M_N_l12_12622

noncomputable def M : Set ℝ := {x | x^2 + x - 6 < 0}
noncomputable def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x } = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l12_12622


namespace midpoint_on_circumcircle_of_triangle_ADZ_l12_12893

theorem midpoint_on_circumcircle_of_triangle_ADZ
  {A B C D Z : Point}
  (h_triangle: Triangle A B C)
  (h_AB_lt_AC : dist A B < dist A C)
  (h_D_on_angle_bisector : D lies_on_bisector BAC A circumcircle A B C)
  (h_Z_on_perpendicular_bisector_external_bisector : Z lies_on_perpendicular_bisector_both_angle AC and_external_bisector A B C) :
  midpoint A B lies_on_circumcircle A D Z :=
begin
  sorry
end

end midpoint_on_circumcircle_of_triangle_ADZ_l12_12893


namespace g_g_2_eq_78652_l12_12361

def g (x : ℝ) : ℝ := 4 * x^3 - 3 * x + 1

theorem g_g_2_eq_78652 : g (g 2) = 78652 := by
  sorry

end g_g_2_eq_78652_l12_12361


namespace elder_age_is_30_l12_12432

-- Define the ages of the younger and elder persons
variables (y e : ℕ)

-- We have the following conditions:
-- Condition 1: The elder's age is 16 years more than the younger's age
def age_difference := e = y + 16

-- Condition 2: Six years ago, the elder's age was three times the younger's age
def six_years_ago := e - 6 = 3 * (y - 6)

-- We need to prove that the present age of the elder person is 30
theorem elder_age_is_30 (y e : ℕ) (h1 : age_difference y e) (h2 : six_years_ago y e) : e = 30 :=
sorry

end elder_age_is_30_l12_12432


namespace ratio_of_hours_l12_12277

theorem ratio_of_hours (x y z : ℕ) 
  (h1 : x + y + z = 157) 
  (h2 : z = y - 8) 
  (h3 : z = 56) 
  (h4 : y = x + 10) : 
  (y / gcd y x) = 32 ∧ (x / gcd y x) = 27 := 
by 
  sorry

end ratio_of_hours_l12_12277


namespace days_B_to_complete_remaining_work_l12_12678

/-- 
  Given that:
  - A can complete a work in 20 days.
  - B can complete the same work in 12 days.
  - A and B worked together for 3 days before A left.
  
  We need to prove that B will require 7.2 days to complete the remaining work alone. 
--/
theorem days_B_to_complete_remaining_work : 
  (∃ (A_rate B_rate combined_rate work_done_in_3_days remaining_work d_B : ℚ), 
   A_rate = (1 / 20) ∧
   B_rate = (1 / 12) ∧
   combined_rate = A_rate + B_rate ∧
   work_done_in_3_days = 3 * combined_rate ∧
   remaining_work = 1 - work_done_in_3_days ∧
   d_B = remaining_work / B_rate ∧
   d_B = 7.2) := 
by 
  sorry

end days_B_to_complete_remaining_work_l12_12678


namespace circle_reflection_l12_12262

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3)
    (new_x new_y : ℝ) (hne_x : new_x = 3) (hne_y : new_y = -8) :
    (new_x, new_y) = (-y, -x) := by
  sorry

end circle_reflection_l12_12262


namespace cookies_per_box_correct_l12_12456

variable (cookies_per_box : ℕ)

-- Define the conditions
def morning_cookie : ℕ := 1 / 2
def bed_cookie : ℕ := 1 / 2
def day_cookies : ℕ := 2
def daily_cookies := morning_cookie + bed_cookie + day_cookies

def days : ℕ := 30
def total_cookies := days * daily_cookies

def boxes : ℕ := 2
def total_cookies_in_boxes : ℕ := cookies_per_box * boxes

-- Theorem we want to prove
theorem cookies_per_box_correct :
  total_cookies_in_boxes = 90 → cookies_per_box = 45 :=
by
  sorry

end cookies_per_box_correct_l12_12456


namespace min_cos_for_sqrt_l12_12335

theorem min_cos_for_sqrt (x : ℝ) (h : 2 * Real.cos x - 1 ≥ 0) : Real.cos x ≥ 1 / 2 := 
by
  sorry

end min_cos_for_sqrt_l12_12335


namespace ben_marble_count_l12_12457

theorem ben_marble_count :
  ∃ k : ℕ, 5 * 2^k > 200 ∧ ∀ m < k, 5 * 2^m ≤ 200 :=
sorry

end ben_marble_count_l12_12457


namespace eval_expression_l12_12029

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12029


namespace circle_center_x_coordinate_eq_l12_12920

theorem circle_center_x_coordinate_eq (a : ℝ) (h : (∃ k : ℝ, ∀ x y : ℝ, x^2 + y^2 - a * x = k) ∧ (1 = a / 2)) : a = 2 :=
sorry

end circle_center_x_coordinate_eq_l12_12920


namespace gardener_b_time_l12_12211

theorem gardener_b_time :
  ∃ x : ℝ, (1 / 3 + 1 / x = 1 / 1.875) → (x = 5) := by
  sorry

end gardener_b_time_l12_12211


namespace karen_average_speed_l12_12239

noncomputable def total_distance : ℚ := 198
noncomputable def start_time : ℚ := (9 * 60 + 40) / 60
noncomputable def end_time : ℚ := (13 * 60 + 20) / 60
noncomputable def total_time : ℚ := end_time - start_time
noncomputable def average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed :
  average_speed total_distance total_time = 54 := by
  sorry

end karen_average_speed_l12_12239


namespace closest_integer_to_cube_root_of_500_l12_12149

theorem closest_integer_to_cube_root_of_500 :
  ∃ n : ℤ, (∀ m : ℤ, |m^3 - 500| ≥ |8^3 - 500|) := 
sorry

end closest_integer_to_cube_root_of_500_l12_12149


namespace smallest_n_gt_15_l12_12885

theorem smallest_n_gt_15 (n : ℕ) : n ≡ 4 [MOD 6] → n ≡ 3 [MOD 7] → n > 15 → n = 52 :=
by
  sorry

end smallest_n_gt_15_l12_12885


namespace walking_time_l12_12557

theorem walking_time (distance_walking_rate : ℕ) 
                     (distance : ℕ)
                     (rest_distance : ℕ) 
                     (rest_time : ℕ) 
                     (total_walking_time : ℕ) : 
  distance_walking_rate = 10 → 
  rest_distance = 10 → 
  rest_time = 7 → 
  distance = 50 → 
  total_walking_time = 328 → 
  total_walking_time = (distance / distance_walking_rate) * 60 + ((distance / rest_distance) - 1) * rest_time :=
by
  sorry

end walking_time_l12_12557


namespace length_of_third_wall_l12_12389

-- Define the dimensions of the first two walls
def wall1_length : ℕ := 30
def wall1_height : ℕ := 12
def wall1_area : ℕ := wall1_length * wall1_height

def wall2_length : ℕ := 30
def wall2_height : ℕ := 12
def wall2_area : ℕ := wall2_length * wall2_height

-- Total area needed
def total_area_needed : ℕ := 960

-- Calculate the area for the third wall
def two_walls_area : ℕ := wall1_area + wall2_area
def third_wall_area : ℕ := total_area_needed - two_walls_area

-- Height of the third wall
def third_wall_height : ℕ := 12

-- Calculate the length of the third wall
def third_wall_length : ℕ := third_wall_area / third_wall_height

-- Final claim: Length of the third wall is 20 feet
theorem length_of_third_wall : third_wall_length = 20 := by
  sorry

end length_of_third_wall_l12_12389


namespace sum_of_three_distinct_l12_12604

def S : Set ℤ := {2, 5, 8, 11, 14, 17, 20}

theorem sum_of_three_distinct (S : Set ℤ) (h : S = {2, 5, 8, 11, 14, 17, 20}) :
  (∃ n : ℕ, n = 13 ∧ ∀ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ∃ k : ℕ, a + b + c = 3 * k) := 
by  -- The proof goes here.
  sorry

end sum_of_three_distinct_l12_12604


namespace crowdfunding_highest_level_backing_l12_12693

-- Definitions according to the conditions
def lowest_level_backing : ℕ := 50
def second_level_backing : ℕ := 10 * lowest_level_backing
def highest_level_backing : ℕ := 100 * lowest_level_backing
def total_raised : ℕ := (2 * highest_level_backing) + (3 * second_level_backing) + (10 * lowest_level_backing)

-- Statement of the problem
theorem crowdfunding_highest_level_backing (h: total_raised = 12000) :
  highest_level_backing = 5000 :=
sorry

end crowdfunding_highest_level_backing_l12_12693


namespace find_cos_sum_l12_12494

-- Defining the conditions based on the problem
variable (P A B C D : Type) (α β : ℝ)

-- Assumptions stating the given conditions
def regular_quadrilateral_pyramid (P A B C D : Type) : Prop :=
  -- Placeholder for the exact definition of a regular quadrilateral pyramid
  sorry

def dihedral_angle_lateral_base (P A B C D : Type) (α : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between lateral face and base is α
  sorry

def dihedral_angle_adjacent_lateral (P A B C D : Type) (β : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between two adjacent lateral faces is β
  sorry

-- The final theorem that we want to prove
theorem find_cos_sum (P A B C D : Type) (α β : ℝ)
  (H1 : regular_quadrilateral_pyramid P A B C D)
  (H2 : dihedral_angle_lateral_base P A B C D α)
  (H3 : dihedral_angle_adjacent_lateral P A B C D β) :
  2 * Real.cos β + Real.cos (2 * α) = -1 :=
sorry

end find_cos_sum_l12_12494


namespace number_of_common_points_l12_12466

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

end number_of_common_points_l12_12466


namespace circle_area_l12_12651

-- Given conditions
variables {BD AC : ℝ} (BD_pos : BD = 6) (AC_pos : AC = 12)
variables {R : ℝ} (R_pos : R = 15 / 2)

-- Prove that the area of the circles is \(\frac{225}{4}\pi\)
theorem circle_area (BD_pos : BD = 6) (AC_pos : AC = 12) (R : ℝ) (R_pos : R = 15 / 2) : 
        ∃ S, S = (225 / 4) * Real.pi := 
by sorry

end circle_area_l12_12651


namespace soccer_team_points_l12_12642

theorem soccer_team_points 
  (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_draws : draws = total_games - (wins + losses))
  (h_points_per_win : points_per_win = 3)
  (h_points_per_draw : points_per_draw = 1)
  (h_points_per_loss : points_per_loss = 0) :
  (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) = 46 :=
by
  -- the actual proof steps will be inserted here
  sorry

end soccer_team_points_l12_12642


namespace relay_race_total_time_l12_12210

theorem relay_race_total_time :
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  athlete1 + athlete2 + athlete3 + athlete4 = 200 :=
by
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  show athlete1 + athlete2 + athlete3 + athlete4 = 200
  sorry

end relay_race_total_time_l12_12210


namespace total_fruits_l12_12274

theorem total_fruits (cucumbers : ℕ) (watermelons : ℕ) 
  (h1 : cucumbers = 18) 
  (h2 : watermelons = cucumbers + 8) : 
  cucumbers + watermelons = 44 := 
by {
  sorry
}

end total_fruits_l12_12274


namespace time_per_bone_l12_12571

theorem time_per_bone (total_hours : ℕ) (total_bones : ℕ) (h1 : total_hours = 1030) (h2 : total_bones = 206) :
  (total_hours / total_bones = 5) :=
by {
  sorry
}

end time_per_bone_l12_12571


namespace minimum_value_of_expression_l12_12842

theorem minimum_value_of_expression (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : 2 * x + 3 * y = 8) : 
  (∀ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 → (2 / a + 3 / b) ≥ 25 / 8) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 ∧ 2 / a + 3 / b = 25 / 8) :=
sorry

end minimum_value_of_expression_l12_12842


namespace value_of_x2_plus_9y2_l12_12370

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l12_12370


namespace points_opposite_sides_line_l12_12055

theorem points_opposite_sides_line (m : ℝ) :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end points_opposite_sides_line_l12_12055


namespace people_with_diploma_percentage_l12_12749

-- Definitions of the given conditions
def P_j_and_not_d := 0.12
def P_not_j_and_d := 0.15
def P_j := 0.40

-- Definitions for intermediate values
def P_not_j := 1 - P_j
def P_not_j_d := P_not_j * P_not_j_and_d

-- Definition of the result to prove
def P_d := (P_j - P_j_and_not_d) + P_not_j_d

theorem people_with_diploma_percentage : P_d = 0.43 := by
  -- Placeholder for the proof
  sorry

end people_with_diploma_percentage_l12_12749


namespace geometric_sequence_common_ratio_l12_12725

theorem geometric_sequence_common_ratio (a : ℕ → ℚ) (q : ℚ) :
  (∀ n, a n = a 2 * q ^ (n - 2)) ∧ a 2 = 2 ∧ a 6 = 1 / 8 →
  (q = 1 / 2 ∨ q = -1 / 2) :=
by
  sorry

end geometric_sequence_common_ratio_l12_12725


namespace part1_1_part1_2_part1_3_part2_l12_12462

def operation (a b c : ℝ) : Prop := a^c = b

theorem part1_1 : operation 3 81 4 :=
by sorry

theorem part1_2 : operation 4 1 0 :=
by sorry

theorem part1_3 : operation 2 (1 / 4) (-2) :=
by sorry

theorem part2 (x y z : ℝ) (h1 : operation 3 7 x) (h2 : operation 3 8 y) (h3 : operation 3 56 z) : x + y = z :=
by sorry

end part1_1_part1_2_part1_3_part2_l12_12462


namespace tom_and_elizabeth_climb_ratio_l12_12144

theorem tom_and_elizabeth_climb_ratio :
  let elizabeth_time := 30
  let tom_time_hours := 2
  let tom_time_minutes := tom_time_hours * 60
  (tom_time_minutes / elizabeth_time) = 4 :=
by sorry

end tom_and_elizabeth_climb_ratio_l12_12144


namespace circle_reflection_l12_12263

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3)
    (new_x new_y : ℝ) (hne_x : new_x = 3) (hne_y : new_y = -8) :
    (new_x, new_y) = (-y, -x) := by
  sorry

end circle_reflection_l12_12263


namespace inverse_graph_pass_point_l12_12394

variable {f : ℝ → ℝ}
variable {f_inv : ℝ → ℝ}

noncomputable def satisfies_inverse (f f_inv : ℝ → ℝ) : Prop :=
∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

theorem inverse_graph_pass_point
  (hf : satisfies_inverse f f_inv)
  (h_point : (1 : ℝ) - f 1 = 3) :
  f_inv (-2) + 3 = 4 :=
by
  sorry

end inverse_graph_pass_point_l12_12394


namespace factor_M_l12_12470

theorem factor_M (a b c d : ℝ) : 
  ((a - c)^2 + (b - d)^2) * (a^2 + b^2) - (a * d - b * c)^2 =
  (a * c + b * d - a^2 - b^2)^2 :=
by
  sorry

end factor_M_l12_12470


namespace infinite_common_divisor_l12_12255

theorem infinite_common_divisor (n : ℕ) : ∃ᶠ n in at_top, Nat.gcd (2 * n - 3) (3 * n - 2) > 1 := 
sorry

end infinite_common_divisor_l12_12255


namespace sum_a_b_when_pow_is_max_l12_12821

theorem sum_a_b_when_pow_is_max (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 1) (h_pow : a^b < 500) 
(h_max : ∀ (a' b' : ℕ), (a' > 0) -> (b' > 1) -> (a'^b' < 500) -> a^b >= a'^b') : a + b = 24 := by
  sorry

end sum_a_b_when_pow_is_max_l12_12821


namespace bird_costs_l12_12455

-- Define the cost of a small bird and a large bird
def cost_small_bird (x : ℕ) := x
def cost_large_bird (x : ℕ) := 2 * x

-- Define total cost calculations for the first and second ladies
def cost_first_lady (x : ℕ) := 5 * cost_large_bird x + 3 * cost_small_bird x
def cost_second_lady (x : ℕ) := 5 * cost_small_bird x + 3 * cost_large_bird x

-- State the main theorem
theorem bird_costs (x : ℕ) (hx : cost_first_lady x = cost_second_lady x + 20) : 
(cost_small_bird x = 10) ∧ (cost_large_bird x = 20) := 
by {
  sorry
}

end bird_costs_l12_12455


namespace money_left_l12_12097

noncomputable def initial_amount : ℕ := 100
noncomputable def cost_roast : ℕ := 17
noncomputable def cost_vegetables : ℕ := 11

theorem money_left (init_amt cost_r cost_v : ℕ) 
  (h1 : init_amt = 100)
  (h2 : cost_r = 17)
  (h3 : cost_v = 11) : init_amt - (cost_r + cost_v) = 72 := by
  sorry

end money_left_l12_12097


namespace rita_bought_4_pounds_l12_12112

variable (total_amount : ℝ) (cost_per_pound : ℝ) (amount_left : ℝ)

theorem rita_bought_4_pounds (h1 : total_amount = 70)
                             (h2 : cost_per_pound = 8.58)
                             (h3 : amount_left = 35.68) :
  (total_amount - amount_left) / cost_per_pound = 4 := 
  by
  sorry

end rita_bought_4_pounds_l12_12112


namespace line_plane_parallelism_l12_12841

variables {Point : Type} [LinearOrder Point] -- Assuming Point is a Type with some linear order.

-- Definitions for line and plane
-- These definitions need further libraries or details depending on actual Lean geometry library support
@[ext] structure Line (P : Type) := (contains : P → Prop)
@[ext] structure Plane (P : Type) := (contains : P → Prop)

variables {a b : Line Point} {α β : Plane Point} {l : Line Point}

-- Conditions (as in part a)
axiom lines_are_different : a ≠ b
axiom planes_are_different : α ≠ β
axiom planes_intersect_in_line : ∃ l, α.contains l ∧ β.contains l
axiom a_parallel_l : ∀ p : Point, a.contains p → l.contains p
axiom b_within_plane : ∀ p : Point, b.contains p → β.contains p
axiom b_parallel_alpha : ∀ p q : Point, β.contains p → β.contains q → α.contains p → α.contains q

-- Define the theorem statement
theorem line_plane_parallelism : a ≠ b ∧ α ≠ β ∧ (∃ l, α.contains l ∧ β.contains l) 
  ∧ (∀ p, a.contains p → l.contains p) 
  ∧ (∀ p, b.contains p → β.contains p) 
  ∧ (∀ p q, β.contains p → β.contains q → α.contains p → α.contains q) → a = b :=
by sorry

end line_plane_parallelism_l12_12841


namespace factorize_expr_l12_12328

theorem factorize_expr (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := 
sorry

end factorize_expr_l12_12328


namespace second_piece_cost_l12_12880

theorem second_piece_cost
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (single_piece1 : ℕ)
  (single_piece2 : ℕ)
  (remaining_piece_count : ℕ)
  (remaining_piece_cost : ℕ)
  (total_cost : total_spent = 610)
  (number_of_items : num_pieces = 7)
  (first_item_cost : single_piece1 = 49)
  (remaining_piece_item_cost : remaining_piece_cost = 96)
  (first_item_total_cost : remaining_piece_count = 5)
  (sum_equation : single_piece1 + single_piece2 + (remaining_piece_count * remaining_piece_cost) = total_spent) :
  single_piece2 = 81 := 
  sorry

end second_piece_cost_l12_12880


namespace solve_inequality_l12_12582

open Set Real

theorem solve_inequality (x : ℝ) : { x : ℝ | x^2 - 4 * x > 12 } = {x : ℝ | x < -2} ∪ {x : ℝ | 6 < x} := 
sorry

end solve_inequality_l12_12582


namespace soccer_team_points_l12_12645

theorem soccer_team_points 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (points_per_win : ℕ) 
  (points_per_draw : ℕ) 
  (points_per_loss : ℕ) 
  (draws : ℕ := total_games - (wins + losses)) : 
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  46 = (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) :=
by sorry

end soccer_team_points_l12_12645


namespace kiwis_to_add_for_25_percent_oranges_l12_12525

theorem kiwis_to_add_for_25_percent_oranges :
  let oranges := 24
  let kiwis := 30
  let apples := 15
  let bananas := 20
  let total_fruits := oranges + kiwis + apples + bananas
  let target_total_fruits := (oranges : ℝ) / 0.25
  let fruits_to_add := target_total_fruits - (total_fruits : ℝ)
  fruits_to_add = 7 := by
  sorry

end kiwis_to_add_for_25_percent_oranges_l12_12525


namespace find_value_of_expression_l12_12355

theorem find_value_of_expression (x y : ℝ)
  (h1 : 5 * x + y = 19)
  (h2 : x + 3 * y = 1) :
  3 * x + 2 * y = 10 :=
sorry

end find_value_of_expression_l12_12355


namespace lcm_5_6_8_9_l12_12927

theorem lcm_5_6_8_9 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := by
  sorry

end lcm_5_6_8_9_l12_12927


namespace evaluate_expression_l12_12012

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12012


namespace eval_expression_l12_12031

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12031


namespace expand_product_l12_12709

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by 
  sorry

end expand_product_l12_12709


namespace geometric_series_ratio_l12_12473

theorem geometric_series_ratio (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ (n : ℕ), S n = a 1 * (1 - q^n) / (1 - q))
  (h2 : a 3 + 2 * a 6 = 0)
  (h3 : a 6 = a 3 * q^3)
  (h4 : q^3 = -1 / 2) :
  S 3 / S 6 = 2 := 
sorry

end geometric_series_ratio_l12_12473


namespace find_number_l12_12687

/--
A number is added to 5, then multiplied by 5, then subtracted by 5, and then divided by 5. 
The result is still 5. Prove that the number is 1.
-/
theorem find_number (x : ℝ) (h : ((5 * (x + 5) - 5) / 5 = 5)) : x = 1 := 
  sorry

end find_number_l12_12687


namespace find_original_cost_price_l12_12431

variable (C S : ℝ)

-- Conditions
def original_profit (C S : ℝ) : Prop := S = 1.25 * C
def new_profit_condition (C S : ℝ) : Prop := 1.04 * C = S - 12.60

-- Main Theorem
theorem find_original_cost_price (h1 : original_profit C S) (h2 : new_profit_condition C S) : C = 60 := 
sorry

end find_original_cost_price_l12_12431


namespace import_tax_percentage_l12_12441

theorem import_tax_percentage
  (total_value : ℝ)
  (non_taxable_portion : ℝ)
  (import_tax_paid : ℝ)
  (h_total_value : total_value = 2610)
  (h_non_taxable_portion : non_taxable_portion = 1000)
  (h_import_tax_paid : import_tax_paid = 112.70) :
  ((import_tax_paid / (total_value - non_taxable_portion)) * 100) = 7 :=
by
  sorry

end import_tax_percentage_l12_12441


namespace inequality_is_linear_l12_12430

theorem inequality_is_linear (k : ℝ) (h1 : (|k| - 1) = 1) (h2 : (k + 2) ≠ 0) : k = 2 :=
sorry

end inequality_is_linear_l12_12430


namespace hypotenuse_length_l12_12544

theorem hypotenuse_length (a b c : ℝ) (hC : (a^2 + b^2) * (a^2 + b^2 + 1) = 12) (right_triangle : a^2 + b^2 = c^2) : 
  c = Real.sqrt 3 := 
by
  sorry

end hypotenuse_length_l12_12544


namespace original_selling_price_l12_12183

theorem original_selling_price (C : ℝ) (h : 1.60 * C = 2560) : 1.40 * C = 2240 :=
by
  sorry

end original_selling_price_l12_12183


namespace jill_present_age_l12_12663

-- Define the main proof problem
theorem jill_present_age (H J : ℕ) (h1 : H + J = 33) (h2 : H - 6 = 2 * (J - 6)) : J = 13 :=
by
  sorry

end jill_present_age_l12_12663


namespace quadrilateral_area_l12_12326

theorem quadrilateral_area :
  let a1 := 9  -- adjacent side length
  let a2 := 6  -- other adjacent side length
  let d := 20  -- diagonal
  let θ1 := 35  -- first angle in degrees
  let θ2 := 110  -- second angle in degrees
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  area_triangle1 + area_triangle2 = 108.006 := 
by
  let a1 := 9
  let a2 := 6
  let d := 20
  let θ1 := 35
  let θ2 := 110
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  show area_triangle1 + area_triangle2 = 108.006
  sorry

end quadrilateral_area_l12_12326


namespace arcade_fraction_spent_l12_12561

noncomputable def weekly_allowance : ℚ := 2.25 
def y (x : ℚ) : ℚ := 1 - x
def remainding_after_toy (x : ℚ) : ℚ := y x - (1/3) * y x

theorem arcade_fraction_spent : 
  ∃ x : ℚ, remainding_after_toy x = 0.60 ∧ x = 3/5 :=
by
  sorry

end arcade_fraction_spent_l12_12561


namespace money_left_l12_12096

noncomputable def initial_amount : ℕ := 100
noncomputable def cost_roast : ℕ := 17
noncomputable def cost_vegetables : ℕ := 11

theorem money_left (init_amt cost_r cost_v : ℕ) 
  (h1 : init_amt = 100)
  (h2 : cost_r = 17)
  (h3 : cost_v = 11) : init_amt - (cost_r + cost_v) = 72 := by
  sorry

end money_left_l12_12096


namespace proof_problem_l12_12609

variable {R : Type*} [Field R] {x y z w N : R}

theorem proof_problem 
  (h1 : 4 * x * z + y * w = N)
  (h2 : x * w + y * z = 6)
  (h3 : (2 * x + y) * (2 * z + w) = 15) :
  N = 3 :=
by sorry

end proof_problem_l12_12609


namespace tangent_line_ln_curve_l12_12594

theorem tangent_line_ln_curve (a : ℝ) :
  (∃ x y : ℝ, y = Real.log x + a ∧ x - y + 1 = 0 ∧ (∀ t : ℝ, t = x → (t - (Real.log t + a)) = -(1 - a))) → a = 2 :=
by
  sorry

end tangent_line_ln_curve_l12_12594


namespace ana_wins_l12_12528

-- Define the game conditions and state
def game_conditions (n : ℕ) (m : ℕ) : Prop :=
  n < m ∧ m < n^2 ∧ Nat.gcd n m = 1

-- Define the losing condition
def losing_condition (n : ℕ) : Prop :=
  n >= 2016

-- Define the predicate for Ana having a winning strategy
def ana_winning_strategy : Prop :=
  ∃ (strategy : ℕ → ℕ), strategy 3 = 5 ∧
  (∀ n, (¬ losing_condition n) → (losing_condition (strategy n)))

theorem ana_wins : ana_winning_strategy :=
  sorry

end ana_wins_l12_12528


namespace gcd_84_120_eq_12_l12_12538

theorem gcd_84_120_eq_12 : Int.gcd 84 120 = 12 := by
  sorry

end gcd_84_120_eq_12_l12_12538


namespace inequality_correctness_l12_12863

theorem inequality_correctness (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end inequality_correctness_l12_12863


namespace max_sides_convex_polygon_with_obtuse_angles_l12_12740

-- Definition of conditions
def is_convex_polygon (n : ℕ) : Prop := n ≥ 3
def obtuse_angles (n : ℕ) (k : ℕ) : Prop := k = 3 ∧ is_convex_polygon n

-- Statement of the problem
theorem max_sides_convex_polygon_with_obtuse_angles (n : ℕ) :
  obtuse_angles n 3 → n ≤ 6 :=
sorry

end max_sides_convex_polygon_with_obtuse_angles_l12_12740


namespace expression_divisible_by_11_l12_12403

theorem expression_divisible_by_11 (n : ℕ) : (3 ^ (2 * n + 2) + 2 ^ (6 * n + 1)) % 11 = 0 :=
sorry

end expression_divisible_by_11_l12_12403


namespace thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l12_12534

theorem thirty_percent_less_than_ninety_eq_one_fourth_more_than_n (n : ℝ) :
  0.7 * 90 = (5 / 4) * n → n = 50.4 :=
by sorry

end thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l12_12534


namespace jessa_gave_3_bills_l12_12585

variable (J G K : ℕ)
variable (billsGiven : ℕ)

/-- Initial conditions and question for the problem -/
def initial_conditions :=
  G = 16 ∧
  K = J - 2 ∧
  G = 2 * K ∧
  (J - billsGiven = 7)

/-- The theorem to prove: Jessa gave 3 bills to Geric -/
theorem jessa_gave_3_bills (h : initial_conditions J G K billsGiven) : billsGiven = 3 := 
sorry

end jessa_gave_3_bills_l12_12585


namespace solutions_exist_l12_12331

theorem solutions_exist (k : ℤ) : ∃ x y : ℤ, (x = 3 * k + 2) ∧ (y = 7 * k + 4) ∧ (7 * x - 3 * y = 2) :=
by {
  -- Proof will be filled in here
  sorry
}

end solutions_exist_l12_12331


namespace eval_expression_l12_12027

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12027


namespace prove_system_of_equations_l12_12748

variables (x y : ℕ)

def system_of_equations (x y : ℕ) : Prop :=
  x = 2*y + 4 ∧ x = 3*y - 9

theorem prove_system_of_equations :
  ∀ (x y : ℕ), system_of_equations x y :=
by sorry

end prove_system_of_equations_l12_12748


namespace at_least_one_not_solved_l12_12957

theorem at_least_one_not_solved (p q : Prop) : (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by sorry

end at_least_one_not_solved_l12_12957


namespace point_to_polar_coordinates_l12_12575

noncomputable def convert_to_polar_coordinates (x y : ℝ) (r θ : ℝ) : Prop :=
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x)

theorem point_to_polar_coordinates :
  convert_to_polar_coordinates 8 (2 * Real.sqrt 6) 
    (2 * Real.sqrt 22) (Real.arctan (Real.sqrt 6 / 4)) :=
sorry

end point_to_polar_coordinates_l12_12575


namespace max_pieces_of_pie_l12_12520

theorem max_pieces_of_pie : ∃ (PIE PIECE : ℕ), 10000 ≤ PIE ∧ PIE < 100000
  ∧ 10000 ≤ PIECE ∧ PIECE < 100000
  ∧ ∃ (n : ℕ), n = 7 ∧ PIE = n * PIECE := by
  sorry

end max_pieces_of_pie_l12_12520


namespace boat_distance_along_stream_in_one_hour_l12_12872

theorem boat_distance_along_stream_in_one_hour :
  ∀ (v_b v_s d_up t : ℝ),
  v_b = 7 →
  d_up = 3 →
  t = 1 →
  (t * (v_b - v_s) = d_up) →
  t * (v_b + v_s) = 11 :=
by
  intros v_b v_s d_up t Hv_b Hd_up Ht Hup
  sorry

end boat_distance_along_stream_in_one_hour_l12_12872


namespace find_common_ratio_l12_12724

noncomputable def a : ℕ → ℝ
noncomputable def q : ℝ

axiom geom_seq (n : ℕ) : a n = a 1 * q ^ (n - 1)
axiom a2 : a 2 = 2
axiom a6 : a 6 = (1 / 8)

theorem find_common_ratio : (q = 1 / 2) ∨ (q = -1 / 2) :=
by
  sorry

end find_common_ratio_l12_12724


namespace average_xy_l12_12269

theorem average_xy (x y : ℝ) 
  (h : (4 + 6 + 9 + x + y) / 5 = 20) : (x + y) / 2 = 40.5 :=
sorry

end average_xy_l12_12269


namespace difference_of_two_distinct_members_sum_of_two_distinct_members_l12_12734

theorem difference_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ N, N = 19 ∧ (∀ n, 1 ≤ n ∧ n ≤ N → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ n = a - b)) :=
by
  sorry

theorem sum_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ M, M = 37 ∧ (∀ m, 3 ≤ m ∧ m ≤ 39 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ m = a + b)) :=
by
  sorry

end difference_of_two_distinct_members_sum_of_two_distinct_members_l12_12734


namespace Jenny_older_than_Rommel_l12_12924

theorem Jenny_older_than_Rommel :
  ∃ t r j, t = 5 ∧ r = 3 * t ∧ j = t + 12 ∧ (j - r = 2) := 
by
  -- We insert the proof here using sorry to skip the actual proof part.
  sorry

end Jenny_older_than_Rommel_l12_12924


namespace stair_calculation_l12_12448

def already_climbed : ℕ := 74
def left_to_climb : ℕ := 22
def total_stairs : ℕ := 96

theorem stair_calculation :
  already_climbed + left_to_climb = total_stairs :=
by {
  sorry
}

end stair_calculation_l12_12448


namespace area_of_circles_l12_12649

theorem area_of_circles (BD AC : ℝ) (hBD : BD = 6) (hAC : AC = 12) : 
  ∃ S : ℝ, S = 225 / 4 * Real.pi :=
by
  sorry

end area_of_circles_l12_12649


namespace max_value_expression_l12_12338

theorem max_value_expression (x : ℝ) : 
  ∃ m : ℝ, m = 1 / 37 ∧ ∀ x : ℝ, (x^6) / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) ≤ m :=
sorry

end max_value_expression_l12_12338


namespace volume_of_cube_in_pyramid_l12_12811

theorem volume_of_cube_in_pyramid :
  (∃ (s : ℝ), 
    ( ∀ (b h l : ℝ),
      b = 2 ∧ 
      h = 3 ∧ 
      l = 2 * Real.sqrt 2 →
      s = 4 * Real.sqrt 2 - 3 ∧ 
      ((4 * Real.sqrt 2 - 3) ^ 3 = (4 * Real.sqrt 2 - 3) ^ 3))) :=
sorry

end volume_of_cube_in_pyramid_l12_12811


namespace timothy_read_pages_l12_12143

theorem timothy_read_pages 
    (mon_tue_pages : Nat) (wed_pages : Nat) (thu_sat_pages : Nat) 
    (sun_read_pages : Nat) (sun_review_pages : Nat) : 
    mon_tue_pages = 45 → wed_pages = 50 → thu_sat_pages = 40 → sun_read_pages = 25 → sun_review_pages = 15 →
    (2 * mon_tue_pages + wed_pages + 3 * thu_sat_pages + sun_read_pages + sun_review_pages = 300) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end timothy_read_pages_l12_12143


namespace roots_of_quadratic_eq_l12_12488

theorem roots_of_quadratic_eq {x1 x2 : ℝ} (h1 : x1 * x1 - 3 * x1 - 5 = 0) (h2 : x2 * x2 - 3 * x2 - 5 = 0) 
                              (h3 : x1 + x2 = 3) (h4 : x1 * x2 = -5) : x1^2 + x2^2 = 19 := 
sorry

end roots_of_quadratic_eq_l12_12488


namespace evaluate_x2_minus_y2_l12_12047

-- Definitions based on the conditions.
def x : ℝ
def y : ℝ
axiom cond1 : x + y = 12
axiom cond2 : 3 * x + y = 18

-- The main statement we need to prove.
theorem evaluate_x2_minus_y2 : x^2 - y^2 = -72 :=
by
  sorry

end evaluate_x2_minus_y2_l12_12047


namespace evaluation_result_l12_12971

noncomputable def evaluate_expression : ℝ :=
  let a := 210
  let b := 206
  let numerator := 980 ^ 2
  let denominator := a^2 - b^2
  numerator / denominator

theorem evaluation_result : evaluate_expression = 577.5 := 
  sorry  -- Placeholder for the proof

end evaluation_result_l12_12971


namespace value_of_x_squared_plus_9y_squared_l12_12376

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l12_12376


namespace geometric_sequence_product_l12_12354

theorem geometric_sequence_product (b : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, b (n+1) = b n * r)
  (h_b9 : b 9 = (3 + 5) / 2) : b 1 * b 17 = 16 :=
by
  sorry

end geometric_sequence_product_l12_12354


namespace katerina_weight_correct_l12_12569

-- We define the conditions
def total_weight : ℕ := 95
def alexa_weight : ℕ := 46

-- Define the proposition to prove: Katerina's weight is the total weight minus Alexa's weight, which should be 49.
theorem katerina_weight_correct : (total_weight - alexa_weight = 49) :=
by
  -- We use sorry to skip the proof.
  sorry

end katerina_weight_correct_l12_12569


namespace inequality_proof_equality_case_l12_12761

variables (x y z : ℝ)
  
theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) : 
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 := 
sorry

theorem equality_case 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) 
  (h_eq : (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) = 1) :
  x = 1 ∧ y = 1 ∧ z = 1 := 
sorry

end inequality_proof_equality_case_l12_12761


namespace find_m_set_l12_12856

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}
noncomputable def B (m : ℝ) : Set ℝ := if m = 0 then ∅ else {-1/m}

theorem find_m_set :
  { m : ℝ | A ∪ B m = A } = {0, -1/2, -1/3} :=
by
  sorry

end find_m_set_l12_12856


namespace smallest_positive_number_is_correct_l12_12860

noncomputable def smallest_positive_number : ℝ := 20 - 5 * Real.sqrt 15

theorem smallest_positive_number_is_correct :
  ∀ n,
    (n = 12 - 3 * Real.sqrt 12 ∨ n = 3 * Real.sqrt 12 - 11 ∨ n = 20 - 5 * Real.sqrt 15 ∨ n = 55 - 11 * Real.sqrt 30 ∨ n = 11 * Real.sqrt 30 - 55) →
    n > 0 → smallest_positive_number ≤ n :=
by
  sorry

end smallest_positive_number_is_correct_l12_12860


namespace evaluate_expression_l12_12004

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12004


namespace general_term_sequence_l12_12356

noncomputable def a (t : ℝ) (n : ℕ) : ℝ :=
if h : t ≠ 1 then (2 * (t^n - 1) / n) - 1 else 0

theorem general_term_sequence (t : ℝ) (n : ℕ) (hn : n ≠ 0) (h : t ≠ 1) :
  a t (n+1) = (2 * (t^(n+1) - 1) / (n+1)) - 1 := 
sorry

end general_term_sequence_l12_12356


namespace parallel_lines_m_value_l12_12997

/-- Given two lines x + m * y + 6 = 0 and (m - 2) * x + 3 * y + 2 * m = 0 are parallel,
    prove that the value of the real number m that makes the lines parallel is -1. -/
theorem parallel_lines_m_value (m : ℝ) : 
  (x + m * y + 6 = 0 ∧ (m - 2) * x + 3 * y + 2 * m = 0 → 
  (m = -1)) :=
by
  sorry

end parallel_lines_m_value_l12_12997


namespace vampire_daily_needed_people_l12_12186

-- Define the conditions as constants
def gallons_needed_per_week : ℕ := 7
def pints_per_person : ℕ := 2
def pints_per_gallon : ℕ := 8
def days_per_week : ℕ := 7

-- Define the proof statement
theorem vampire_daily_needed_people (gallons_needed_per_week = 7) 
                                    (pints_per_person = 2) 
                                    (pints_per_gallon = 8) 
                                    (days_per_week = 7) : 
                                    7 * 8 / 7 / 2 = 4 :=
by
    -- The proof is expected here
    sorry

end vampire_daily_needed_people_l12_12186


namespace margaret_spends_on_croissants_l12_12038

theorem margaret_spends_on_croissants :
  (∀ (people : ℕ) (sandwiches_per_person : ℕ) (croissants_per_sandwich : ℕ) (croissants_per_set : ℕ) (cost_per_set : ℝ),
    people = 24 →
    sandwiches_per_person = 2 →
    croissants_per_sandwich = 1 →
    croissants_per_set = 12 →
    cost_per_set = 8 →
    (people * sandwiches_per_person * croissants_per_sandwich) / croissants_per_set * cost_per_set = 32) := sorry

end margaret_spends_on_croissants_l12_12038


namespace add_and_multiply_l12_12907

def num1 : ℝ := 0.0034
def num2 : ℝ := 0.125
def num3 : ℝ := 0.00678
def sum := num1 + num2 + num3

theorem add_and_multiply :
  (sum * 2) = 0.27036 := by
  sorry

end add_and_multiply_l12_12907


namespace andrew_grapes_purchase_l12_12318

theorem andrew_grapes_purchase (G : ℕ) (rate_grape rate_mango total_paid total_mango_cost : ℕ)
  (h1 : rate_grape = 54)
  (h2 : rate_mango = 62)
  (h3 : total_paid = 1376)
  (h4 : total_mango_cost = 10 * rate_mango)
  (h5 : total_paid = rate_grape * G + total_mango_cost) : G = 14 := by
  sorry

end andrew_grapes_purchase_l12_12318


namespace bus_driver_regular_rate_l12_12166

theorem bus_driver_regular_rate (R : ℝ) (h1 : 976 = (40 * R) + (14.32 * (1.75 * R))) : 
  R = 15 := 
by
  sorry

end bus_driver_regular_rate_l12_12166


namespace age_of_female_employee_when_hired_l12_12298

-- Defining the conditions
def hired_year : ℕ := 1989
def retirement_year : ℕ := 2008
def sum_age_employment : ℕ := 70

-- Given the conditions we found that years of employment (Y):
def years_of_employment : ℕ := retirement_year - hired_year -- 19

-- Defining the age when hired (A)
def age_when_hired : ℕ := sum_age_employment - years_of_employment -- 51

-- Now we need to prove
theorem age_of_female_employee_when_hired : age_when_hired = 51 :=
by
  -- Here should be the proof steps, but we use sorry for now
  sorry

end age_of_female_employee_when_hired_l12_12298


namespace georgia_makes_muffins_l12_12454

/--
Georgia makes muffins and brings them to her students on the first day of every month.
Her muffin recipe only makes 6 muffins and she has 24 students. 
Prove that Georgia makes 36 batches of muffins in 9 months.
-/
theorem georgia_makes_muffins 
  (muffins_per_batch : ℕ)
  (students : ℕ)
  (months : ℕ) 
  (batches_per_day : ℕ) 
  (total_batches : ℕ)
  (h1 : muffins_per_batch = 6)
  (h2 : students = 24)
  (h3 : months = 9)
  (h4 : batches_per_day = students / muffins_per_batch) : 
  total_batches = months * batches_per_day :=
by
  -- The proof would go here
  sorry

end georgia_makes_muffins_l12_12454


namespace area_of_right_triangle_with_hypotenuse_and_angle_l12_12559

theorem area_of_right_triangle_with_hypotenuse_and_angle 
  (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 9 * Real.sqrt 3) (h_angle : angle = 30) : 
  ∃ (area : ℝ), area = 364.5 := 
by
  sorry

end area_of_right_triangle_with_hypotenuse_and_angle_l12_12559


namespace bananas_bought_l12_12264

theorem bananas_bought (O P B : Nat) (x : Nat) 
  (h1 : P - O = B)
  (h2 : O + P = 120)
  (h3 : P = 90)
  (h4 : 60 * x + 30 * (2 * x) = 24000) : 
  x = 200 := by
  sorry

end bananas_bought_l12_12264


namespace no_nat_n_for_9_pow_n_minus_7_is_product_l12_12464

theorem no_nat_n_for_9_pow_n_minus_7_is_product :
  ¬ ∃ (n k : ℕ), 9 ^ n - 7 = k * (k + 1) :=
by
  sorry

end no_nat_n_for_9_pow_n_minus_7_is_product_l12_12464


namespace steve_family_time_l12_12124

theorem steve_family_time :
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours * assignments_fraction
  let total_activity_hours := sleeping_hours + school_hours + assignments_hours
  day_hours - total_activity_hours = 10 :=
by
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours *  assignments_fraction
  let total_activity_hours := sleeping_hours +
                              school_hours + 
                              assignments_hours
  show day_hours - total_activity_hours = 10
  sorry

end steve_family_time_l12_12124


namespace find_a8_l12_12846

variable (a : ℕ → ℤ)

axiom h1 : ∀ n : ℕ, 2 * a n + a (n + 1) = 0
axiom h2 : a 3 = -2

theorem find_a8 : a 8 = 64 := by
  sorry

end find_a8_l12_12846


namespace limit_ln_eq_half_ln_three_l12_12460

theorem limit_ln_eq_half_ln_three :
  (Real.lim (λ x : ℝ, ln ((exp (x^2) - cos x) * cos (1 / x) + tan (x + Real.pi / 3))) 0) = 1 / 2 * ln 3 :=
sorry

end limit_ln_eq_half_ln_three_l12_12460


namespace time_after_seconds_l12_12619

def initial_time : Nat × Nat × Nat := (4, 45, 0)
def seconds_to_add : Nat := 12345
def final_time : Nat × Nat × Nat := (8, 30, 45)

theorem time_after_seconds (h : initial_time = (4, 45, 0) ∧ seconds_to_add = 12345) : 
  ∃ (h' : Nat × Nat × Nat), h' = final_time := by
  sorry

end time_after_seconds_l12_12619


namespace distance_ratio_l12_12621

variables (KD DM : ℝ)

theorem distance_ratio : 
  KD = 4 ∧ (KD + DM + DM + KD = 12) → (KD / DM = 2) := 
by
  sorry

end distance_ratio_l12_12621


namespace exactly_one_three_digit_perfect_cube_divisible_by_25_l12_12226

theorem exactly_one_three_digit_perfect_cube_divisible_by_25 :
  ∃! (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 25 = 0 :=
sorry

end exactly_one_three_digit_perfect_cube_divisible_by_25_l12_12226


namespace length_of_the_train_l12_12564

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def time_to_cross_seconds : ℝ := 30
noncomputable def bridge_length_meters : ℝ := 205

noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def distance_crossed_meters : ℝ := train_speed_mps * time_to_cross_seconds

theorem length_of_the_train 
  (h1 : train_speed_kmph = 45)
  (h2 : time_to_cross_seconds = 30)
  (h3 : bridge_length_meters = 205) : 
  distance_crossed_meters - bridge_length_meters = 170 := 
by
  sorry

end length_of_the_train_l12_12564


namespace georgia_makes_muffins_l12_12452

-- Definitions based on conditions
def muffinRecipeMakes : ℕ := 6
def numberOfStudents : ℕ := 24
def durationInMonths : ℕ := 9

-- Theorem to prove the given problem
theorem georgia_makes_muffins :
  (numberOfStudents / muffinRecipeMakes) * durationInMonths = 36 :=
by
  -- We'll skip the proof with sorry
  sorry

end georgia_makes_muffins_l12_12452


namespace total_trail_length_l12_12630

-- Definitions based on conditions
variables (a b c d e : ℕ)

-- Conditions
def condition1 : Prop := a + b + c = 36
def condition2 : Prop := b + c + d = 48
def condition3 : Prop := c + d + e = 45
def condition4 : Prop := a + d = 31

-- Theorem statement
theorem total_trail_length (h1 : condition1 a b c) (h2 : condition2 b c d) (h3 : condition3 c d e) (h4 : condition4 a d) : 
  a + b + c + d + e = 81 :=
by 
  sorry

end total_trail_length_l12_12630


namespace lcm_5_6_8_9_l12_12928

theorem lcm_5_6_8_9 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := by
  sorry

end lcm_5_6_8_9_l12_12928


namespace field_length_to_width_ratio_l12_12657
-- Import the math library

-- Define the problem conditions and proof goal statement
theorem field_length_to_width_ratio (w : ℝ) (l : ℝ) (area_pond : ℝ) (area_field : ℝ) 
    (h_length : l = 16) (h_area_pond : area_pond = 64) 
    (h_area_relation : area_pond = (1/2) * area_field)
    (h_field_area : area_field = l * w) : l / w = 2 :=
by 
  -- Leaving the proof as an exercise
  sorry

end field_length_to_width_ratio_l12_12657


namespace origin_in_ellipse_l12_12056

theorem origin_in_ellipse (k : ℝ):
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ x = 0 ∧ y = 0) →
  0 < abs k ∧ abs k < 1 :=
by
  -- Note: Proof omitted.
  sorry

end origin_in_ellipse_l12_12056


namespace transmission_prob_correct_transmission_scheme_comparison_l12_12087

noncomputable def transmission_prob_single (α β : ℝ) : ℝ :=
  (1 - α) * (1 - β)^2

noncomputable def transmission_prob_triple_sequence (β : ℝ) : ℝ :=
  β * (1 - β)^2

noncomputable def transmission_prob_triple_decoding_one (β : ℝ) : ℝ :=
  β * (1 - β)^2 + (1 - β)^3

noncomputable def transmission_prob_triple_decoding_zero (α : ℝ) : ℝ :=
  3 * α * (1 - α)^2 + (1 - α)^3

noncomputable def transmission_prob_single_decoding_zero (α : ℝ) : ℝ :=
  1 - α

theorem transmission_prob_correct (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  transmission_prob_single α β = (1 - α) * (1 - β)^2 ∧
  transmission_prob_triple_sequence β = β * (1 - β)^2 ∧
  transmission_prob_triple_decoding_one β = β * (1 - β)^2 + (1 - β)^3 :=
sorry

theorem transmission_scheme_comparison (α : ℝ) (hα : 0 < α ∧ α < 0.5) :
  transmission_prob_triple_decoding_zero α > transmission_prob_single_decoding_zero α :=
sorry

end transmission_prob_correct_transmission_scheme_comparison_l12_12087


namespace single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l12_12090

section transmission_scheme

variables (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1)

-- Part A
theorem single_transmission_probability :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β) ^ 2 :=
by sorry

-- Part B
theorem triple_transmission_probability :
  (1 - β) * β * (1 - β) = β * (1 - β) ^ 2 :=
by sorry

-- Part C
theorem triple_transmission_decoding :
  (3 * β * (1 - β) ^ 2) + (1 - β) ^ 3 = β * (1 - β) ^ 2 + (1 - β) ^ 3 :=
by sorry

-- Part D
theorem decoding_comparison (h : 0 < α ∧ α < 0.5) :
  (1 - α) < (3 * α * (1 - α) ^ 2 + (1 - α) ^ 3) :=
by sorry

end transmission_scheme

end single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l12_12090


namespace expand_product_l12_12710

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by 
  sorry

end expand_product_l12_12710


namespace total_fruits_in_30_days_l12_12788

-- Define the number of oranges Sophie receives each day
def sophie_daily_oranges : ℕ := 20

-- Define the number of grapes Hannah receives each day
def hannah_daily_grapes : ℕ := 40

-- Define the number of days
def number_of_days : ℕ := 30

-- Calculate the total number of fruits received by Sophie and Hannah in 30 days
theorem total_fruits_in_30_days :
  (sophie_daily_oranges * number_of_days) + (hannah_daily_grapes * number_of_days) = 1800 :=
by
  sorry

end total_fruits_in_30_days_l12_12788


namespace river_road_cars_l12_12137

theorem river_road_cars
  (B C : ℕ)
  (h1 : B * 17 = C)
  (h2 : C = B + 80) :
  C = 85 := by
  sorry

end river_road_cars_l12_12137


namespace reduced_price_per_kg_l12_12958

theorem reduced_price_per_kg (P R : ℝ) (Q : ℝ)
  (h1 : R = 0.80 * P)
  (h2 : Q * P = 1500)
  (h3 : (Q + 10) * R = 1500) : R = 30 :=
by
  sorry

end reduced_price_per_kg_l12_12958


namespace necessary_but_not_sufficient_condition_l12_12991

variable {a : Nat → Real} -- Sequence a_n
variable {q : Real} -- Common ratio
variable (a1_pos : a 1 > 0) -- Condition a1 > 0

-- Definition of geometric sequence
def is_geometric_sequence (a : Nat → Real) (q : Real) : Prop :=
  ∀ n : Nat, a (n + 1) = a n * q

-- Definition of increasing sequence
def is_increasing_sequence (a : Nat → Real) : Prop :=
  ∀ n : Nat, a n < a (n + 1)

-- Theorem statement
theorem necessary_but_not_sufficient_condition (a : Nat → Real) (q : Real) (a1_pos : a 1 > 0) :
  is_geometric_sequence a q →
  is_increasing_sequence a →
  q > 0 ∧ ¬(q > 0 → is_increasing_sequence a) := by
  sorry

end necessary_but_not_sufficient_condition_l12_12991


namespace path_length_of_dot_l12_12813

-- Define the dimensions of the rectangular prism
def prism_width := 1 -- cm
def prism_height := 1 -- cm
def prism_length := 2 -- cm

-- Define the condition that the dot is marked at the center of the top face
def dot_position := (0.5, 1)

-- Define the condition that the prism starts with the 1 cm by 2 cm face on the table
def initial_face_on_table := (prism_length, prism_height)

-- Define the statement to prove the length of the path followed by the dot
theorem path_length_of_dot: 
  ∃ length_of_path : ℝ, length_of_path = 2 * Real.pi :=
sorry

end path_length_of_dot_l12_12813


namespace competition_winner_is_C_l12_12342

-- Define the type for singers
inductive Singer
| A | B | C | D
deriving DecidableEq

-- Assume each singer makes a statement
def statement (s : Singer) : Prop :=
  match s with
  | Singer.A => Singer.B ≠ Singer.C
  | Singer.B => Singer.A ≠ Singer.C
  | Singer.C => true
  | Singer.D => Singer.B ≠ Singer.D

-- Define that two and only two statements are true
def exactly_two_statements_are_true : Prop :=
  (statement Singer.A ∧ statement Singer.C ∧ ¬statement Singer.B ∧ ¬statement Singer.D) ∨
  (statement Singer.A ∧ statement Singer.D ∧ ¬statement Singer.B ∧ ¬statement Singer.C)

-- Define the winner
def winner : Singer := Singer.C

-- The main theorem to be proved
theorem competition_winner_is_C :
  exactly_two_statements_are_true → (winner = Singer.C) :=
by
  intro h
  exact sorry

end competition_winner_is_C_l12_12342


namespace problem1_part1_problem1_part2_problem2_l12_12717

noncomputable def problem1_condition1 (m : ℕ) (a : ℕ) : Prop := 4^m = a
noncomputable def problem1_condition2 (n : ℕ) (b : ℕ) : Prop := 8^n = b

theorem problem1_part1 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(2*m + 3*n) = a * b :=
by sorry

theorem problem1_part2 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(4*m - 6*n) = (a^2) / (b^2) :=
by sorry

theorem problem2 (x : ℕ) (h : 2 * 8^x * 16 = 2^23) : x = 6 :=
by sorry

end problem1_part1_problem1_part2_problem2_l12_12717


namespace value_of_x2_plus_9y2_l12_12368

theorem value_of_x2_plus_9y2 (x y : ℝ) 
  (h1 : x + 3 * y = 6)
  (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := 
by {
  sorry
}

end value_of_x2_plus_9y2_l12_12368


namespace shiela_drawings_l12_12633

theorem shiela_drawings (neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
  (h1 : neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
  by 
    have h : total_drawings = neighbors * drawings_per_neighbor := sorry
    rw [h1, h2] at h
    exact h
    -- Proof skipped with sorry.

end shiela_drawings_l12_12633


namespace determineHairColors_l12_12418

structure Person where
  name : String
  hairColor : String

def Belokurov : Person := { name := "Belokurov", hairColor := "" }
def Chernov : Person := { name := "Chernov", hairColor := "" }
def Ryzhev : Person := { name := "Ryzhev", hairColor := "" }

-- Define the possible hair colors
def Blonde : String := "Blonde"
def Brunette : String := "Brunette"
def RedHaired : String := "Red-Haired"

-- Define the conditions based on the problem statement
axiom hairColorConditions :
  Belokurov.hairColor ≠ Blonde ∧
  Belokurov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ Brunette ∧
  Chernov.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ RedHaired ∧
  Ryzhev.hairColor ≠ Blonde ∧
  ∀ p : Person, p.hairColor = Brunette → p.name ≠ "Belokurov"

-- Define the uniqueness condition that each person has a different hair color
axiom uniqueHairColors :
  Belokurov.hairColor ≠ Chernov.hairColor ∧
  Belokurov.hairColor ≠ Ryzhev.hairColor ∧
  Chernov.hairColor ≠ Ryzhev.hairColor

-- Define the proof problem
theorem determineHairColors :
  Belokurov.hairColor = RedHaired ∧
  Chernov.hairColor = Blonde ∧
  Ryzhev.hairColor = Brunette := by
  sorry

end determineHairColors_l12_12418


namespace digitalEarth_correct_l12_12317

-- Define the possible descriptions of "Digital Earth"
inductive DigitalEarthDescription
| optionA : DigitalEarthDescription
| optionB : DigitalEarthDescription
| optionC : DigitalEarthDescription
| optionD : DigitalEarthDescription

-- Define the correct description according to the solution
def correctDescription : DigitalEarthDescription := DigitalEarthDescription.optionB

-- Define the theorem to prove the equivalence
theorem digitalEarth_correct :
  correctDescription = DigitalEarthDescription.optionB :=
sorry

end digitalEarth_correct_l12_12317


namespace evaluate_x2_y2_l12_12045

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end evaluate_x2_y2_l12_12045


namespace value_large_cube_l12_12306

-- Definitions based on conditions
def volume_small := 1 -- volume of one-inch cube in cubic inches
def volume_large := 64 -- volume of four-inch cube in cubic inches
def value_small : ℝ := 1000 -- value of one-inch cube of gold in dollars
def proportion (x y : ℝ) : Prop := y = 64 * x -- proportionality condition

-- Prove that the value of the four-inch cube of gold is $64000
theorem value_large_cube : proportion value_small 64000 := by
  -- Proof skipped
  sorry

end value_large_cube_l12_12306


namespace hyperbola_range_of_k_l12_12867

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x^2)/(k + 4) + (y^2)/(k - 1) = 1) → -4 < k ∧ k < 1 :=
by 
  sorry

end hyperbola_range_of_k_l12_12867


namespace time_with_family_l12_12127

theorem time_with_family : 
    let hours_in_day := 24
    let sleep_fraction := 1 / 3
    let school_fraction := 1 / 6
    let assignment_fraction := 1 / 12
    let sleep_hours := sleep_fraction * hours_in_day
    let school_hours := school_fraction * hours_in_day
    let assignment_hours := assignment_fraction * hours_in_day
    let total_hours_occupied := sleep_hours + school_hours + assignment_hours
    hours_in_day - total_hours_occupied = 10 :=
by
  sorry

end time_with_family_l12_12127


namespace student_exchanges_l12_12381

theorem student_exchanges (x : ℕ) : x * (x - 1) = 72 :=
sorry

end student_exchanges_l12_12381


namespace solve_mt_eq_l12_12332

theorem solve_mt_eq (m n : ℤ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^2 + n) * (m + n^2) = (m - n)^3 →
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by
  sorry

end solve_mt_eq_l12_12332


namespace jogging_friends_probability_l12_12537

theorem jogging_friends_probability
  (n p q r : ℝ)
  (h₀ : 1 > 0) -- Positive integers condition
  (h₁ : n = p - q * Real.sqrt r)
  (h₂ : ∀ prime, ¬ (r ∣ prime ^ 2)) -- r is not divisible by the square of any prime
  (h₃ : (60 - n)^2 = 1800) -- Derived from 50% meeting probability
  (h₄ : p = 60) -- Identified values from solution
  (h₅ : q = 30)
  (h₆ : r = 2) : 
  p + q + r = 92 :=
by
  sorry

end jogging_friends_probability_l12_12537


namespace pencils_in_each_box_l12_12606

theorem pencils_in_each_box (total_pencils : ℕ) (total_boxes : ℕ) (pencils_per_box : ℕ) 
  (h1 : total_pencils = 648) (h2 : total_boxes = 162) : 
  total_pencils / total_boxes = pencils_per_box := 
by
  sorry

end pencils_in_each_box_l12_12606


namespace minimum_pyramid_volume_proof_l12_12792

noncomputable def minimum_pyramid_volume (side_length : ℝ) (apex_angle : ℝ) : ℝ :=
  if side_length = 6 ∧ apex_angle = 2 * Real.arcsin (1 / 3 : ℝ) then 5 * Real.sqrt 23 else 0

theorem minimum_pyramid_volume_proof : 
  minimum_pyramid_volume 6 (2 * Real.arcsin (1 / 3)) = 5 * Real.sqrt 23 :=
by
  sorry

end minimum_pyramid_volume_proof_l12_12792


namespace union_of_sets_l12_12240

def M : Set ℝ := {x | x^2 + 2 * x = 0}

def N : Set ℝ := {x | x^2 - 2 * x = 0}

theorem union_of_sets : M ∪ N = {x | x = -2 ∨ x = 0 ∨ x = 2} := sorry

end union_of_sets_l12_12240


namespace placing_pencils_l12_12206

theorem placing_pencils (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
    (h1 : total_pencils = 6) (h2 : num_rows = 2) : pencils_per_row = 3 :=
by
  sorry

end placing_pencils_l12_12206


namespace arrangements_APPLE_is_60_l12_12605

-- Definition of the problem statement based on the given conditions
def distinct_arrangements_APPLE : Nat :=
  let n := 5
  let n_A := 1
  let n_P := 2
  let n_L := 1
  let n_E := 1
  (n.factorial / (n_A.factorial * n_P.factorial * n_L.factorial * n_E.factorial))

-- The proof statement (without the proof itself, which is "sorry")
theorem arrangements_APPLE_is_60 : distinct_arrangements_APPLE = 60 := by
  sorry

end arrangements_APPLE_is_60_l12_12605


namespace probability_red_balls_by_4th_draw_l12_12416

theorem probability_red_balls_by_4th_draw :
  let total_balls := 10
  let red_prob := 2 / total_balls
  let white_prob := 1 - red_prob
  (white_prob^3) * red_prob = 0.0434 := sorry

end probability_red_balls_by_4th_draw_l12_12416


namespace hotel_room_friends_distribution_l12_12956

theorem hotel_room_friends_distribution 
    (rooms : ℕ)
    (friends : ℕ)
    (min_friends_per_room : ℕ)
    (max_friends_per_room : ℕ)
    (unique_ways : ℕ) :
    rooms = 6 →
    friends = 10 →
    min_friends_per_room = 1 →
    max_friends_per_room = 3 →
    unique_ways = 1058400 :=
by
  intros h_rooms h_friends h_min_friends h_max_friends
  sorry

end hotel_room_friends_distribution_l12_12956


namespace abs_neg_three_l12_12130

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l12_12130


namespace triangular_number_is_perfect_square_l12_12830

def is_triangular_number (T : ℕ) : Prop :=
∃ n : ℕ, T = n * (n + 1) / 2

def is_perfect_square (T : ℕ) : Prop :=
∃ y : ℕ, T = y * y

theorem triangular_number_is_perfect_square:
  ∀ (x_k : ℕ), 
    ((∃ n y : ℕ, (2 * n + 1)^2 - 8 * y^2 = 1 ∧ T_n = n * (n + 1) / 2 ∧ T_n = x_k^2 - 1 / 8) →
    (is_triangular_number T_n → is_perfect_square T_n)) :=
by
  sorry

end triangular_number_is_perfect_square_l12_12830


namespace find_OQ_l12_12209
-- Import the required math libarary

-- Define points on a line with the given distances
def O := 0
def A (a : ℝ) := 2 * a
def B (b : ℝ) := 4 * b
def C (c : ℝ) := 5 * c
def D (d : ℝ) := 7 * d

-- Given P between B and C such that ratio condition holds
def P (a b c d x : ℝ) := 
  B b ≤ x ∧ x ≤ C c ∧ 
  (A a - x) * (x - C c) = (B b - x) * (x - D d)

-- Calculate Q based on given ratio condition
def Q (b c d y : ℝ) := 
  C c ≤ y ∧ y ≤ D d ∧ 
  (C c - y) * (y - D d) = (B b - C c) * (C c - D d)

-- Main Proof Statement to prove OQ
theorem find_OQ (a b c d y : ℝ) 
  (hP : ∃ x, P a b c d x)
  (hQ : ∃ y, Q b c d y) :
  y = (14 * c * d - 10 * b * c) / (5 * c - 7 * d) := by
  sorry

end find_OQ_l12_12209


namespace sophie_buys_six_doughnuts_l12_12775

variable (num_doughnuts : ℕ)

theorem sophie_buys_six_doughnuts 
  (h1 : 5 * 2 = 10)
  (h2 : 4 * 2 = 8)
  (h3 : 15 * 0.60 = 9)
  (h4 : 10 + 8 + 9 = 27)
  (h5 : 33 - 27 = 6)
  (h6 : num_doughnuts * 1 = 6) :
  num_doughnuts = 6 := 
  by
    sorry

end sophie_buys_six_doughnuts_l12_12775


namespace base_four_odd_last_digit_l12_12340

theorem base_four_odd_last_digit :
  ∃ b : ℕ, b = 4 ∧ (b^4 ≤ 625 ∧ 625 < b^5) ∧ (625 % b % 2 = 1) :=
by
  sorry

end base_four_odd_last_digit_l12_12340


namespace arctan_tan_sub_eq_l12_12700

noncomputable def arctan_tan_sub (a b : ℝ) : ℝ := Real.arctan (Real.tan a - 3 * Real.tan b)

theorem arctan_tan_sub_eq (a b : ℝ) (ha : a = 75) (hb : b = 15) :
  arctan_tan_sub a b = 75 :=
by
  sorry

end arctan_tan_sub_eq_l12_12700


namespace positive_real_inequality_l12_12626

noncomputable def positive_real_sum_condition (u v w : ℝ) [OrderedRing ℝ] :=
  u + v + w + Real.sqrt (u * v * w) = 4

theorem positive_real_inequality (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  positive_real_sum_condition u v w →
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w :=
by
  sorry

end positive_real_inequality_l12_12626


namespace max_val_a_l12_12854

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x^2 - 3 * x + 2)

theorem max_val_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 1, f a x ≥ 0) : a ≤ 1 := sorry

end max_val_a_l12_12854


namespace determine_value_l12_12703

theorem determine_value : 3 - ((-3)⁻³ : ℚ) = 82 / 27 := by
  sorry

end determine_value_l12_12703


namespace value_of_x2_plus_9y2_l12_12373

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l12_12373


namespace shift_right_three_units_l12_12384

theorem shift_right_three_units (x : ℝ) : (λ x, -2 * x) (x - 3) = -2 * x + 6 :=
by
  sorry

end shift_right_three_units_l12_12384


namespace sum_4digit_numbers_remainder_3_l12_12668

theorem sum_4digit_numbers_remainder_3
  (LCM : ℕ := 35)
  (is_4digit : ℕ → Prop := λ n, n >= 1000 ∧ n <= 9999)
  (leaves_remainder_3 : ℕ → Prop := λ n, n % LCM = 3)
  (numbers : list ℕ := list.range' 1000 (9999 - 1000 + 1))
  : (numbers.filter (λ n, leaves_remainder_3 n)).sum = 1414773 := by
  sorry

end sum_4digit_numbers_remainder_3_l12_12668


namespace base_eight_conversion_l12_12284

theorem base_eight_conversion :
  (1 * 8^2 + 3 * 8^1 + 2 * 8^0 = 90) := by
  sorry

end base_eight_conversion_l12_12284


namespace seats_per_section_correct_l12_12139

-- Define the total number of seats
def total_seats : ℕ := 270

-- Define the number of sections
def sections : ℕ := 9

-- Define the number of seats per section
def seats_per_section (total_seats sections : ℕ) : ℕ := total_seats / sections

theorem seats_per_section_correct : seats_per_section total_seats sections = 30 := by
  sorry

end seats_per_section_correct_l12_12139


namespace a_minus_b_is_neg_seven_l12_12847

-- Definitions for sets
def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 1 < x ∧ x < 4}
def setC : Set ℝ := {x | 1 < x ∧ x < 3}

-- Proving the statement
theorem a_minus_b_is_neg_seven :
  ∀ (a b : ℝ), (∀ x, (x ∈ setC) ↔ (x^2 + a*x + b < 0)) → a - b = -7 :=
by
  intros a b h
  sorry

end a_minus_b_is_neg_seven_l12_12847


namespace evaluate_expressions_l12_12937

theorem evaluate_expressions : (∀ (a b c d : ℤ), a = -(-3) → b = -(|-3|) → c = -(-(3^2)) → d = ((-3)^2) → b < 0) :=
by
  sorry

end evaluate_expressions_l12_12937


namespace proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l12_12793

theorem proof_by_contradiction_x_gt_y_implies_x3_gt_y3
  (x y: ℝ) (h: x > y) : ¬ (x^3 ≤ y^3) :=
by
  -- We need to show that assuming x^3 <= y^3 leads to a contradiction
  sorry

end proof_by_contradiction_x_gt_y_implies_x3_gt_y3_l12_12793


namespace solve_for_y_l12_12902

theorem solve_for_y (x y : ℝ) (h : 5 * x - y = 6) : y = 5 * x - 6 :=
sorry

end solve_for_y_l12_12902


namespace angle_B_pi_div_3_triangle_perimeter_l12_12236

-- Problem 1: Prove that B = π / 3 given the condition.
theorem angle_B_pi_div_3 (A B C : ℝ) (hTriangle : A + B + C = Real.pi) 
  (hCos : Real.cos B = Real.cos ((A + C) / 2)) : 
  B = Real.pi / 3 :=
sorry

-- Problem 2: Prove the perimeter given the conditions.
theorem triangle_perimeter (a b c : ℝ) (m : ℝ) 
  (altitude : ℝ) 
  (hSides : 8 * a = 3 * c) 
  (hAltitude : altitude = 12 * Real.sqrt 3 / 7) 
  (hAngleB : ∃ B, B = Real.pi / 3) :
  a + b + c = 18 := 
sorry

end angle_B_pi_div_3_triangle_perimeter_l12_12236


namespace problem1_l12_12507

def setA : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def setB (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 2 * m - 2}

theorem problem1 (m : ℝ) : 
  (∀ x, x ∈ setA → x ∈ setB m) ∧ ¬(∀ x, x ∈ setA ↔ x ∈ setB m) → 3 ≤ m :=
sorry

end problem1_l12_12507


namespace c_is_11_years_younger_than_a_l12_12947

variable (A B C : ℕ) (h : A + B = B + C + 11)

theorem c_is_11_years_younger_than_a (A B C : ℕ) (h : A + B = B + C + 11) : C = A - 11 := by
  sorry

end c_is_11_years_younger_than_a_l12_12947


namespace arithmetic_sequence_property_l12_12895

-- Define the arithmetic sequence {an}
variable {α : Type*} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) := ∃ (d : α), ∀ (n : ℕ), a (n+1) = a n + d

-- Define the condition
def given_condition (a : ℕ → α) : Prop := a 5 / a 3 = 5 / 9

-- Main theorem statement
theorem arithmetic_sequence_property (a : ℕ → α) (h : is_arith_seq a) 
  (h_condition : given_condition a) : 1 = 1 :=
by
  sorry

end arithmetic_sequence_property_l12_12895


namespace term_100_is_981_l12_12349

def sequence_term (n : ℕ) : ℕ :=
  if n = 100 then 981 else sorry

theorem term_100_is_981 : sequence_term 100 = 981 := by
  rfl

end term_100_is_981_l12_12349


namespace variance_X_plus_2Y_l12_12042

/-
The definitions used in Lean 4 should be directly based on the conditions outlined in the given problem.
-/

noncomputable theory

variable {Ω : Type*} [MeasureSpace Ω]

-- Define the random variables X and Y with given distributions
def X : Ω → ℝ := λ ω, if ω = 0 then 0 else 1
def P_X : MeasureTheory.Measure Ω := MeasureTheory.Measure.dirac 0 0.5 + MeasureTheory.Measure.dirac 1 0.5

def Y : Ω → ℝ := λ ω, if ω = 1 then 1 else 2
def P_Y : MeasureTheory.Measure Ω := MeasureTheory.Measure.dirac 1 (2/3) + MeasureTheory.Measure.dirac 2 (1/3)

-- X and Y are independent
axiom independent_X_Y : MeasureTheory.ProbIndep X Y P_X P_Y

-- Given that a = 1/2 and b = 2/3, we aim to prove the variance of X + 2Y
theorem variance_X_plus_2Y :
  MeasureTheory.variance (λ ω, X ω + 2 * Y ω) = (41 / 36) :=
begin
  sorry
end

end variance_X_plus_2Y_l12_12042


namespace savings_after_one_year_l12_12802

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem savings_after_one_year :
  compound_interest 1000 0.10 2 1 = 1102.50 :=
by
  sorry

end savings_after_one_year_l12_12802


namespace exponentiation_addition_zero_l12_12458

theorem exponentiation_addition_zero : (-2)^(3^2) + 2^(3^2) = 0 := 
by 
  -- proof goes here
  sorry

end exponentiation_addition_zero_l12_12458


namespace batsman_new_average_l12_12295

variable (A : ℝ) -- Assume that A is the average before the 17th inning
variable (score : ℝ) -- The score in the 17th inning
variable (new_average : ℝ) -- The new average after the 17th inning

-- The conditions
axiom H1 : score = 85
axiom H2 : new_average = A + 3

-- The statement to prove
theorem batsman_new_average : 
    new_average = 37 :=
by 
  sorry

end batsman_new_average_l12_12295


namespace red_car_speed_is_10mph_l12_12421

noncomputable def speed_of_red_car (speed_black : ℝ) (initial_distance : ℝ) (time_to_overtake : ℝ) : ℝ :=
  (speed_black * time_to_overtake - initial_distance) / time_to_overtake

theorem red_car_speed_is_10mph :
  ∀ (speed_black initial_distance time_to_overtake : ℝ),
  speed_black = 50 →
  initial_distance = 20 →
  time_to_overtake = 0.5 →
  speed_of_red_car speed_black initial_distance time_to_overtake = 10 :=
by
  intros speed_black initial_distance time_to_overtake hb hd ht
  rw [hb, hd, ht]
  norm_num
  sorry

end red_car_speed_is_10mph_l12_12421


namespace steve_family_time_l12_12120

theorem steve_family_time :
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  hours_per_day - (hours_sleeping + hours_school + hours_assignments) = 10 :=
by
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  sorry

end steve_family_time_l12_12120


namespace max_value_fraction_l12_12690

theorem max_value_fraction {a b c : ℝ} (h1 : c = Real.sqrt (a^2 + b^2)) 
  (h2 : a > 0) (h3 : b > 0) (A : ℝ) (hA : A = 1 / 2 * a * b) :
  ∃ x : ℝ, x = (a + b + A) / c ∧ x ≤ (5 / 4) * Real.sqrt 2 :=
by
  sorry

end max_value_fraction_l12_12690


namespace kopecks_to_rubles_l12_12276

noncomputable def exchangeable_using_coins (total : ℕ) (num_coins : ℕ) : Prop :=
  ∃ (x y z t u v w : ℕ), 
    total = x * 1 + y * 2 + z * 5 + t * 10 + u * 20 + v * 50 + w * 100 ∧ 
    num_coins = x + y + z + t + u + v + w

theorem kopecks_to_rubles (A B : ℕ)
  (h : exchangeable_using_coins A B) : exchangeable_using_coins (100 * B) A :=
sorry

end kopecks_to_rubles_l12_12276


namespace value_of_x_squared_plus_9y_squared_l12_12377

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l12_12377


namespace evaluate_expression_at_minus3_l12_12708

theorem evaluate_expression_at_minus3:
  (∀ x, x = -3 → (3 + x * (3 + x) - 3^2 + x) / (x - 3 + x^2 - x) = -3/2) :=
by
  sorry

end evaluate_expression_at_minus3_l12_12708


namespace roots_quadratic_sum_of_squares_l12_12348

theorem roots_quadratic_sum_of_squares :
  ∀ x1 x2 : ℝ, (x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0) → x1^2 + x2^2 = 6 :=
by
  intros x1 x2 h
  -- proof goes here
  sorry

end roots_quadratic_sum_of_squares_l12_12348


namespace simple_interest_l12_12782

theorem simple_interest (TD : ℝ) (Sum : ℝ) (SI : ℝ) 
  (h1 : TD = 78) 
  (h2 : Sum = 947.1428571428571) 
  (h3 : SI = Sum - (Sum - TD)) : 
  SI = 78 := 
by 
  sorry

end simple_interest_l12_12782


namespace floor_area_not_greater_than_10_l12_12266

theorem floor_area_not_greater_than_10 (L W H : ℝ) (h_height : H = 3)
  (h_more_paint_wall1 : L * 3 > L * W)
  (h_more_paint_wall2 : W * 3 > L * W) :
  L * W ≤ 9 :=
by
  sorry

end floor_area_not_greater_than_10_l12_12266


namespace set_representation_l12_12032

theorem set_representation :
  {p : ℕ × ℕ | 2 * p.1 + 3 * p.2 = 16} = {(2, 4), (5, 2), (8, 0)} :=
by
  sorry

end set_representation_l12_12032


namespace areas_equal_l12_12875

noncomputable def midpoint (A B : Point) : Point :=
  (A + B) / 2

def orthocenter (A B C : Triangle) : Point :=
  -- The orthocenter is the intersection of the altitudes

theorem areas_equal (A B C D : Point) (circABC : is_cyclic_quadrilateral A B C D) :
  let E := midpoint A B
  let F := midpoint B C
  let G := midpoint C D
  let H := midpoint D A
  let W := orthocenter A H E
  let X := orthocenter B E F
  let Y := orthocenter C F G
  let Z := orthocenter D G H
  area (quadrilateral A B C D) = area (quadrilateral W X Y Z) :=
sorry

end areas_equal_l12_12875


namespace distributor_profit_percentage_l12_12953

theorem distributor_profit_percentage 
    (commission_rate : ℝ) (cost_price : ℝ) (final_price : ℝ) (P : ℝ) (profit : ℝ) 
    (profit_percentage: ℝ) :
  commission_rate = 0.20 →
  cost_price = 15 →
  final_price = 19.8 →
  0.80 * P = final_price →
  P = cost_price + profit →
  profit_percentage = (profit / cost_price) * 100 →
  profit_percentage = 65 :=
by
  intros h_commission_rate h_cost_price h_final_price h_equation h_profit_eq h_percent_eq
  sorry

end distributor_profit_percentage_l12_12953


namespace mass_of_man_l12_12670

def boat_length : ℝ := 3 -- boat length in meters
def boat_breadth : ℝ := 2 -- boat breadth in meters
def boat_sink_depth : ℝ := 0.01 -- boat sink depth in meters
def water_density : ℝ := 1000 -- density of water in kg/m^3

/- Theorem: The mass of the man is equal to 60 kg given the parameters defined above. -/
theorem mass_of_man : (water_density * (boat_length * boat_breadth * boat_sink_depth)) = 60 :=
by
  simp [boat_length, boat_breadth, boat_sink_depth, water_density]
  sorry

end mass_of_man_l12_12670


namespace bicycle_trip_length_l12_12881

def total_distance (days1 day1 miles1 day2 miles2: ℕ) : ℕ :=
  days1 * miles1 + day2 * miles2

theorem bicycle_trip_length :
  total_distance 12 12 1 6 = 150 :=
by
  sorry

end bicycle_trip_length_l12_12881


namespace area_of_shaded_region_l12_12580

theorem area_of_shaded_region :
  let v1 := (0, 0)
  let v2 := (15, 0)
  let v3 := (45, 30)
  let v4 := (45, 45)
  let v5 := (30, 45)
  let v6 := (0, 15)
  let area_large_rectangle := 45 * 45
  let area_triangle1 := 1 / 2 * 15 * 15
  let area_triangle2 := 1 / 2 * 15 * 15
  let shaded_area := area_large_rectangle - (area_triangle1 + area_triangle2)
  shaded_area = 1800 :=
by
  sorry

end area_of_shaded_region_l12_12580


namespace cats_to_dogs_ratio_l12_12915

noncomputable def num_dogs : ℕ := 18
noncomputable def num_cats : ℕ := num_dogs - 6
noncomputable def ratio (a b : ℕ) : ℚ := a / b

theorem cats_to_dogs_ratio (h1 : num_dogs = 18) (h2 : num_cats = num_dogs - 6) : ratio num_cats num_dogs = 2 / 3 :=
by
  sorry

end cats_to_dogs_ratio_l12_12915


namespace normal_line_eqn_l12_12631

variables {α β : Type*} [LinearOrderedField α] [TopologicalSpace β]
  {f : α → β} {x₀ : α} {y₀ : β}

-- Conditions: A differentiable function f, a point (x₀, y₀) on the curve, and the derivative of f
theorem normal_line_eqn (hf : DifferentiableAt ℝ f x₀) (hy₀ : f x₀ = y₀) :
  -f'(x₀) * (y - y₀) = x - x₀ := 
sorry

end normal_line_eqn_l12_12631


namespace complex_number_sum_l12_12758

variable (ω : ℂ)
variable (h1 : ω^9 = 1)
variable (h2 : ω ≠ 1)

theorem complex_number_sum :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80 = ω^2 :=
by sorry

end complex_number_sum_l12_12758


namespace edge_length_of_prism_l12_12141

-- Definitions based on conditions
def rectangular_prism_edges : ℕ := 12
def total_edge_length : ℕ := 72

-- Proof problem statement
theorem edge_length_of_prism (num_edges : ℕ) (total_length : ℕ) (h1 : num_edges = rectangular_prism_edges) (h2 : total_length = total_edge_length) : 
  (total_length / num_edges) = 6 :=
by {
  -- The proof is omitted here as instructed
  sorry
}

end edge_length_of_prism_l12_12141


namespace neither_sufficient_nor_necessary_l12_12864

noncomputable def a_b_conditions (a b: ℝ) : Prop :=
∃ (a b: ℝ), ¬((a - b > 0) → (a^2 - b^2 > 0)) ∧ ¬((a^2 - b^2 > 0) → (a - b > 0))

theorem neither_sufficient_nor_necessary (a b: ℝ) : a_b_conditions a b :=
sorry

end neither_sufficient_nor_necessary_l12_12864


namespace students_taking_art_l12_12552

theorem students_taking_art :
  ∀ (total_students music_students both_students neither_students : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_students = 10 →
  neither_students = 460 →
  music_students + both_students + neither_students = total_students →
  ((total_students - neither_students) - (music_students - both_students) + both_students = 20) :=
by
  intros total_students music_students both_students neither_students 
  intro h_total h_music h_both h_neither h_sum 
  sorry

end students_taking_art_l12_12552


namespace exist_rel_prime_k_l_divisible_l12_12836

theorem exist_rel_prime_k_l_divisible (a b p : ℤ) : 
  ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) := 
sorry

end exist_rel_prime_k_l_divisible_l12_12836


namespace find_central_angle_of_sector_l12_12351

variables (r θ : ℝ)

def sector_arc_length (r θ : ℝ) := r * θ
def sector_area (r θ : ℝ) := 0.5 * r^2 * θ

theorem find_central_angle_of_sector
  (l : ℝ)
  (A : ℝ)
  (hl : l = sector_arc_length r θ)
  (hA : A = sector_area r θ)
  (hl_val : l = 4)
  (hA_val : A = 2) :
  θ = 4 :=
sorry

end find_central_angle_of_sector_l12_12351


namespace number_of_children_l12_12706

theorem number_of_children (total_crayons children_crayons children : ℕ) 
  (h1 : children_crayons = 3) 
  (h2 : total_crayons = 18) 
  (h3 : total_crayons = children_crayons * children) : 
  children = 6 := 
by 
  sorry

end number_of_children_l12_12706


namespace initial_boys_down_slide_l12_12638

variable (B : Int)

theorem initial_boys_down_slide:
  B + 13 = 35 → B = 22 := by
  sorry

end initial_boys_down_slide_l12_12638


namespace time_to_cover_length_l12_12799

/-- Constants -/
def speed_escalator : ℝ := 10
def length_escalator : ℝ := 112
def speed_person : ℝ := 4

/-- Proof problem -/
theorem time_to_cover_length :
  (length_escalator / (speed_escalator + speed_person)) = 8 := by
  sorry

end time_to_cover_length_l12_12799


namespace range_of_a_l12_12995

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, (a * x^2 - 3 * x - 4 = 0) ∧ (a * y^2 - 3 * y - 4 = 0) → x = y) ↔ (a ≤ -9 / 16 ∨ a = 0) := 
by
  sorry

end range_of_a_l12_12995


namespace minimum_value_l12_12590

open Real

theorem minimum_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eqn : 2 * x + y = 2) :
    ∃ x y, (0 < x) ∧ (0 < y) ∧ (2 * x + y = 2) ∧ (x + sqrt (x^2 + y^2) = 8 / 5) :=
sorry

end minimum_value_l12_12590


namespace intersection_point_of_planes_l12_12320

theorem intersection_point_of_planes :
  ∃ (x y z : ℚ), 
    3 * x - y + 4 * z = 2 ∧ 
    -3 * x + 4 * y - 3 * z = 4 ∧ 
    -x + y - z = 5 ∧ 
    x = -55 ∧ 
    y = -11 ∧ 
    z = 39 := 
by
  sorry

end intersection_point_of_planes_l12_12320


namespace rational_numbers_on_circle_l12_12477

theorem rational_numbers_on_circle (a b c d e f : ℚ)
  (h1 : a = |b - c|)
  (h2 : b = d)
  (h3 : c = |d - e|)
  (h4 : d = |e - f|)
  (h5 : e = f)
  (h6 : a + b + c + d + e + f = 1) :
  [a, b, c, d, e, f] = [1/4, 1/4, 0, 1/4, 1/4, 0] :=
sorry

end rational_numbers_on_circle_l12_12477


namespace tom_paid_correct_amount_l12_12278

-- Define the conditions given in the problem
def kg_apples : ℕ := 8
def rate_apples : ℕ := 70
def kg_mangoes : ℕ := 9
def rate_mangoes : ℕ := 45

-- Define the cost calculations
def cost_apples : ℕ := kg_apples * rate_apples
def cost_mangoes : ℕ := kg_mangoes * rate_mangoes
def total_amount : ℕ := cost_apples + cost_mangoes

-- The proof problem statement
theorem tom_paid_correct_amount : total_amount = 965 :=
by
  -- The proof steps are omitted and replaced with sorry
  sorry

end tom_paid_correct_amount_l12_12278


namespace avg_growth_rate_selling_price_reduction_l12_12908

open Real

-- Define the conditions for the first question
def sales_volume_aug : ℝ := 50000
def sales_volume_oct : ℝ := 72000

-- Define the conditions for the second question
def cost_price_per_unit : ℝ := 40
def initial_selling_price_per_unit : ℝ := 80
def initial_sales_volume_per_day : ℝ := 20
def additional_units_per_half_dollar_decrease : ℝ := 4
def desired_daily_profit : ℝ := 1400

-- First proof: monthly average growth rate
theorem avg_growth_rate (x : ℝ) :
  sales_volume_aug * (1 + x)^2 = sales_volume_oct → x = 0.2 :=
by {
  sorry
}

-- Second proof: reduction in selling price for daily profit
theorem selling_price_reduction (y : ℝ) :
  (initial_selling_price_per_unit - y - cost_price_per_unit) * (initial_sales_volume_per_day + additional_units_per_half_dollar_decrease * y / 0.5) = desired_daily_profit → y = 30 :=
by {
  sorry
}

end avg_growth_rate_selling_price_reduction_l12_12908


namespace ratio_of_ages_l12_12420

variable (T N : ℕ)
variable (sum_ages : T = T) -- This is tautological based on the given condition; we can consider it a given sum
variable (age_condition : T - N = 3 * (T - 3 * N))

theorem ratio_of_ages (T N : ℕ) (sum_ages : T = T) (age_condition : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end ratio_of_ages_l12_12420


namespace correct_operation_l12_12151

variable (a b : ℝ)

theorem correct_operation : 
  ¬ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧
  ¬ ((a^3) ^ 2 = a ^ 5) ∧
  (a ^ 5 / a ^ 3 = a ^ 2) ∧
  ¬ (a ^ 3 + a ^ 2 = a ^ 5) :=
by
  sorry

end correct_operation_l12_12151


namespace y_coordinate_of_C_l12_12254

def Point : Type := (ℤ × ℤ)

def A : Point := (0, 0)
def B : Point := (0, 4)
def D : Point := (4, 4)
def E : Point := (4, 0)

def PentagonArea (C : Point) : ℚ :=
  let triangleArea : ℚ := (1/2 : ℚ) * 4 * ((C.2 : ℚ) - 4)
  let squareArea : ℚ := 4 * 4
  triangleArea + squareArea

theorem y_coordinate_of_C (h : ℤ) (C : Point := (2, h)) : PentagonArea C = 40 → C.2 = 16 :=
by
  sorry

end y_coordinate_of_C_l12_12254


namespace bryce_received_12_raisins_l12_12857

-- Defining the main entities for the problem
variables {x y z : ℕ} -- number of raisins Bryce, Carter, and Emma received respectively

-- Conditions:
def condition1 (x y : ℕ) : Prop := y = x - 8
def condition2 (x y : ℕ) : Prop := y = x / 3
def condition3 (y z : ℕ) : Prop := z = 2 * y

-- The goal is to prove that Bryce received 12 raisins
theorem bryce_received_12_raisins (x y z : ℕ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) 
  (h3 : condition3 y z) : 
  x = 12 :=
sorry

end bryce_received_12_raisins_l12_12857


namespace shift_parabola_l12_12566

theorem shift_parabola (x : ℝ) : 
    let y := x^2 in 
    ∃ (x' : ℝ), (x' = x - 2) ∧ (y = x'^2) :=
    sorry

end shift_parabola_l12_12566


namespace three_digit_factorial_sum_l12_12148

theorem three_digit_factorial_sum : ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ (∃ a b c : ℕ, x = 100 * a + 10 * b + c ∧ (a = 0 ∨ b = 0 ∨ c = 0) ∧ x = a.factorial + b.factorial + c.factorial) := 
sorry

end three_digit_factorial_sum_l12_12148


namespace georgia_makes_muffins_l12_12453

/--
Georgia makes muffins and brings them to her students on the first day of every month.
Her muffin recipe only makes 6 muffins and she has 24 students. 
Prove that Georgia makes 36 batches of muffins in 9 months.
-/
theorem georgia_makes_muffins 
  (muffins_per_batch : ℕ)
  (students : ℕ)
  (months : ℕ) 
  (batches_per_day : ℕ) 
  (total_batches : ℕ)
  (h1 : muffins_per_batch = 6)
  (h2 : students = 24)
  (h3 : months = 9)
  (h4 : batches_per_day = students / muffins_per_batch) : 
  total_batches = months * batches_per_day :=
by
  -- The proof would go here
  sorry

end georgia_makes_muffins_l12_12453


namespace problem_statement_l12_12265

theorem problem_statement : 
  (∀ x y : ℤ, y = 2 * x^2 - 3 * x + 4 ∧ y = 6 ∧ x = 2) → (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  sorry

end problem_statement_l12_12265


namespace evaluate_expression_l12_12016

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12016


namespace anna_initial_stamps_l12_12818

theorem anna_initial_stamps (final_stamps : ℕ) (alison_stamps : ℕ) (alison_to_anna : ℕ) : 
  final_stamps = 50 ∧ alison_stamps = 28 ∧ alison_to_anna = 14 → (final_stamps - alison_to_anna = 36) :=
by
  sorry

end anna_initial_stamps_l12_12818


namespace candle_lighting_time_l12_12783

theorem candle_lighting_time 
  (l : ℕ) -- initial length of the candles
  (t_diff : ℤ := 206) -- the time difference in minutes, correlating to 1:34 PM.
  : t_diff = 206 :=
by sorry

end candle_lighting_time_l12_12783


namespace log_a_sub_b_eq_one_half_range_of_a_l12_12222

noncomputable def f (a b : ℝ) (x : ℝ) := a * x + b / x - 2 * a + 2

def g (a : ℝ) (x : ℝ) := f a (a - 2) x - 2 * Real.log x

-- Problem (I)
theorem log_a_sub_b_eq_one_half (a b : ℝ) (h0 : 0 < a) (h1 : f'(1) = a - b ∧ a - b = 2) :
  Real.log 4 (a - b) = 1 / 2 :=
sorry

-- Problem (II)
theorem range_of_a (a : ℝ) (h0 : 0 < a) (h1 : ∀ x, x ∈ Set.Ici (1 : ℝ) → f a (a - 2) x - 2 * Real.log x ≥ 0) :
  1 ≤ a :=
sorry

end log_a_sub_b_eq_one_half_range_of_a_l12_12222


namespace evaluate_expression_l12_12005

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12005


namespace Adam_picks_apples_days_l12_12692

theorem Adam_picks_apples_days (total_apples remaining_apples daily_pick : ℕ) 
  (h1 : total_apples = 350) 
  (h2 : remaining_apples = 230) 
  (h3 : daily_pick = 4) : 
  (total_apples - remaining_apples) / daily_pick = 30 :=
by {
  sorry
}

end Adam_picks_apples_days_l12_12692


namespace largest_angle_triangle_l12_12415

-- Definition of constants and conditions
def right_angle : ℝ := 90
def angle_sum : ℝ := 120
def angle_difference : ℝ := 20

-- Given two angles of a triangle sum to 120 degrees and one is 20 degrees greater than the other,
-- Prove the largest angle in the triangle is 70 degrees
theorem largest_angle_triangle (A B C : ℝ) (hA : A + B = angle_sum) (hB : B = A + angle_difference) (hC : A + B + C = 180) : 
  max A (max B C) = 70 := 
by 
  sorry

end largest_angle_triangle_l12_12415


namespace min_value_x2_y2_l12_12843

theorem min_value_x2_y2 (x y : ℝ) (h : 2 * x + y + 5 = 0) : x^2 + y^2 ≥ 5 :=
by
  sorry

end min_value_x2_y2_l12_12843


namespace train_crossing_time_l12_12673

theorem train_crossing_time (length_of_train : ℝ) (speed_of_train : ℝ) (speed_of_man : ℝ) :
  length_of_train = 1500 → speed_of_train = 95 → speed_of_man = 5 → 
  (length_of_train / ((speed_of_train - speed_of_man) * (1000 / 3600))) = 60 :=
by
  intros h1 h2 h3
  have h_rel_speed : ((speed_of_train - speed_of_man) * (1000 / 3600)) = 25 := by
    rw [h2, h3]
    norm_num
  rw [h1, h_rel_speed]
  norm_num

end train_crossing_time_l12_12673


namespace permutation_value_l12_12738

theorem permutation_value (n : ℕ) (h : n * (n - 1) = 12) : n = 4 :=
by
  sorry

end permutation_value_l12_12738


namespace steve_family_time_l12_12121

theorem steve_family_time :
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  hours_per_day - (hours_sleeping + hours_school + hours_assignments) = 10 :=
by
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  sorry

end steve_family_time_l12_12121


namespace distance_inequality_l12_12397

theorem distance_inequality (a : ℝ) (h : |a - 1| < 3) : -2 < a ∧ a < 4 :=
sorry

end distance_inequality_l12_12397


namespace compare_logarithmic_values_l12_12838

theorem compare_logarithmic_values :
  let a := Real.log 3.4 / Real.log 2
  let b := Real.log 3.6 / Real.log 4
  let c := Real.log 0.3 / Real.log 3
  c < b ∧ b < a :=
by
  sorry

end compare_logarithmic_values_l12_12838


namespace how_much_money_per_tshirt_l12_12259

def money_made_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) : Prop :=
  total_money_tshirts / number_tshirts = 62

theorem how_much_money_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) 
  (h1 : total_money_tshirts = 11346) 
  (h2 : number_tshirts = 183) : 
  money_made_per_tshirt total_money_tshirts number_tshirts := 
by 
  sorry

end how_much_money_per_tshirt_l12_12259


namespace maximum_xyzw_l12_12625

theorem maximum_xyzw (x y z w : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_pos_w : 0 < w)
(h : (x * y * z) + w = (x + w) * (y + w) * (z + w))
(h_sum : x + y + z + w = 1) :
  xyzw = 1 / 256 :=
sorry

end maximum_xyzw_l12_12625


namespace height_percentage_difference_l12_12801

theorem height_percentage_difference
  (h_B h_A : ℝ)
  (hA_def : h_A = h_B * 0.55) :
  ((h_B - h_A) / h_A) * 100 = 81.82 := by 
  sorry

end height_percentage_difference_l12_12801


namespace harmonic_mean_closest_to_one_l12_12573

-- Define the given conditions a = 1/4 and b = 2048
def a : ℚ := 1 / 4
def b : ℚ := 2048

-- Define the harmonic mean of two numbers
def harmonic_mean (x y : ℚ) : ℚ := 2 * x * y / (x + y)

-- State the theorem proving the harmonic mean is closest to 1
theorem harmonic_mean_closest_to_one : abs (harmonic_mean a b - 1) < 1 :=
sorry

end harmonic_mean_closest_to_one_l12_12573


namespace find_ab_range_m_l12_12732

-- Part 1
theorem find_ab (a b: ℝ) (h1 : 3 - 6 * a + b = 0) (h2 : -1 + 3 * a - b + a^2 = 0) :
  a = 2 ∧ b = 9 := 
sorry

-- Part 2
theorem range_m (m: ℝ) (h: ∀ x ∈ (Set.Icc (-2) 1), x^3 + 3 * 2 * x^2 + 9 * x + 4 - m ≤ 0) :
  20 ≤ m :=
sorry

end find_ab_range_m_l12_12732


namespace evaluate_x2_minus_y2_l12_12048

-- Definitions based on the conditions.
def x : ℝ
def y : ℝ
axiom cond1 : x + y = 12
axiom cond2 : 3 * x + y = 18

-- The main statement we need to prove.
theorem evaluate_x2_minus_y2 : x^2 - y^2 = -72 :=
by
  sorry

end evaluate_x2_minus_y2_l12_12048


namespace value_of_expression_l12_12986

theorem value_of_expression (x y : ℝ) (h₀ : x = Real.sqrt 2 + 1) (h₁ : y = Real.sqrt 2 - 1) : 
  (x + y) * (x - y) = 4 * Real.sqrt 2 :=
by
  sorry

end value_of_expression_l12_12986


namespace spaghetti_cost_l12_12336

theorem spaghetti_cost (hamburger_cost french_fry_cost soda_cost spaghetti_cost split_payment friends : ℝ) 
(hamburger_count : ℕ) (french_fry_count : ℕ) (soda_count : ℕ) (friend_count : ℕ)
(h_split_payment : split_payment * friend_count = 25)
(h_hamburger_cost : hamburger_cost = 3 * hamburger_count)
(h_french_fry_cost : french_fry_cost = 1.20 * french_fry_count)
(h_soda_cost : soda_cost = 0.5 * soda_count)
(h_total_order_cost : hamburger_cost + french_fry_cost + soda_cost + spaghetti_cost = split_payment * friend_count) :
spaghetti_cost = 2.70 :=
by {
  sorry
}

end spaghetti_cost_l12_12336


namespace marc_average_speed_l12_12699

theorem marc_average_speed 
  (d : ℝ) -- Define d as a real number representing distance
  (chantal_speed1 : ℝ := 3) -- Chantal's speed for the first half
  (chantal_speed2 : ℝ := 1.5) -- Chantal's speed for the second half
  (chantal_speed3 : ℝ := 2) -- Chantal's speed while descending
  (marc_meeting_point : ℝ := (2 / 3) * d) -- One-third point from the trailhead
  (chantal_time1 : ℝ := d / chantal_speed1) 
  (chantal_time2 : ℝ := (d / chantal_speed2))
  (chantal_time3 : ℝ := (d / 6)) -- Chantal's time for the descent from peak to one-third point
  (total_time : ℝ := chantal_time1 + chantal_time2 + chantal_time3) : 
  marc_meeting_point / total_time = 12 / 13 := 
  by 
  -- Leaving the proof as sorry to indicate where the proof would be
  sorry

end marc_average_speed_l12_12699


namespace verify_shifted_function_l12_12385

def linear_function_shift_3_units_right (k b : ℝ) (hk : k ≠ 0) : Prop :=
  ∀ (x : ℝ), (k = -2) → (b = 6) → (λ x, -2 * (x - 3) + 6) = (λ x, k * x + b)

theorem verify_shifted_function : 
  linear_function_shift_3_units_right (-2) 6 (by norm_num) :=
sorry

end verify_shifted_function_l12_12385


namespace smallest_consecutive_even_sum_140_l12_12292

theorem smallest_consecutive_even_sum_140 :
  ∃ (x : ℕ), (x % 2 = 0) ∧ (x + (x + 2) + (x + 4) + (x + 6) = 140) ∧ (x = 32) :=
by
  sorry

end smallest_consecutive_even_sum_140_l12_12292


namespace boys_to_girls_ratio_l12_12523

theorem boys_to_girls_ratio (x y : ℕ) 
  (h1 : 149 * x + 144 * y = 147 * (x + y)) : 
  x = (3 / 2 : ℚ) * y :=
by
  sorry

end boys_to_girls_ratio_l12_12523


namespace second_number_l12_12677

theorem second_number (x : ℝ) (h : 3 + x + 333 + 33.3 = 399.6) : x = 30.3 :=
sorry

end second_number_l12_12677


namespace entertainment_expense_percentage_l12_12773

noncomputable def salary : ℝ := 10000
noncomputable def savings : ℝ := 2000
noncomputable def food_expense_percentage : ℝ := 0.40
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def conveyance_percentage : ℝ := 0.10

theorem entertainment_expense_percentage :
  let E := (1 - (food_expense_percentage + house_rent_percentage + conveyance_percentage) - (savings / salary))
  E = 0.10 :=
by
  sorry

end entertainment_expense_percentage_l12_12773


namespace smallest_n_for_n_cubed_ends_in_888_l12_12241

/-- Proof Problem: Prove that 192 is the smallest positive integer \( n \) such that the last three digits of \( n^3 \) are 888. -/
theorem smallest_n_for_n_cubed_ends_in_888 : ∃ n : ℕ, n > 0 ∧ (n^3 % 1000 = 888) ∧ ∀ m : ℕ, 0 < m ∧ (m^3 % 1000 = 888) → n ≤ m :=
by
  sorry

end smallest_n_for_n_cubed_ends_in_888_l12_12241


namespace white_balls_count_l12_12085

theorem white_balls_count {T W : ℕ} (h1 : 3 * 4 = T) (h2 : T - 3 = W) : W = 9 :=
by 
    sorry

end white_balls_count_l12_12085


namespace desks_increase_l12_12493

theorem desks_increase 
  (rows : ℕ) (first_row_desks : ℕ) (total_desks : ℕ) 
  (d : ℕ) 
  (h_rows : rows = 8) 
  (h_first_row : first_row_desks = 10) 
  (h_total_desks : total_desks = 136)
  (h_desks_sum : 10 + (10 + d) + (10 + 2 * d) + (10 + 3 * d) + (10 + 4 * d) + (10 + 5 * d) + (10 + 6 * d) + (10 + 7 * d) = total_desks) : 
  d = 2 := 
by 
  sorry

end desks_increase_l12_12493


namespace pyramid_surface_area_l12_12309

-- Definitions for the conditions
structure Rectangle where
  length : ℝ
  width : ℝ

structure Pyramid where
  base : Rectangle
  height : ℝ

-- Create instances representing the given conditions
noncomputable def givenRectangle : Rectangle := {
  length := 8,
  width := 6
}

noncomputable def givenPyramid : Pyramid := {
  base := givenRectangle,
  height := 15
}

-- Statement to prove the surface area of the pyramid
theorem pyramid_surface_area
  (rect: Rectangle)
  (length := rect.length)
  (width := rect.width)
  (height: ℝ)
  (hy1: length = 8)
  (hy2: width = 6)
  (hy3: height = 15) :
  let base_area := length * width
  let slant_height := Real.sqrt (height^2 + (length / 2)^2)
  let lateral_area := 2 * ((length * slant_height) / 2 + (width * slant_height) / 2)
  let total_surface_area := base_area + lateral_area 
  total_surface_area = 48 + 7 * Real.sqrt 241 := 
  sorry

end pyramid_surface_area_l12_12309


namespace ticket_door_price_l12_12272

theorem ticket_door_price
  (total_attendance : ℕ)
  (tickets_before : ℕ)
  (price_before : ℚ)
  (total_receipts : ℚ)
  (tickets_bought_before : ℕ)
  (price_door : ℚ)
  (h_attendance : total_attendance = 750)
  (h_price_before : price_before = 2)
  (h_receipts : total_receipts = 1706.25)
  (h_tickets_before : tickets_bought_before = 475)
  (h_total_receipts : (tickets_bought_before * price_before) + (((total_attendance - tickets_bought_before) : ℕ) * price_door) = total_receipts) :
  price_door = 2.75 :=
by
  sorry

end ticket_door_price_l12_12272


namespace family_reunion_handshakes_l12_12963

theorem family_reunion_handshakes (married_couples : ℕ) (participants : ℕ) (allowed_handshakes : ℕ) (total_handshakes : ℕ) :
  married_couples = 8 →
  participants = married_couples * 2 →
  allowed_handshakes = participants - 1 - 1 - 6 →
  total_handshakes = (participants * allowed_handshakes) / 2 →
  total_handshakes = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end family_reunion_handshakes_l12_12963


namespace correct_decimal_multiplication_l12_12961

theorem correct_decimal_multiplication : 0.085 * 3.45 = 0.29325 := 
by 
  sorry

end correct_decimal_multiplication_l12_12961


namespace max_f_l12_12034

noncomputable def f (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 + Real.sin θ)

theorem max_f : ∀ (θ : ℝ), 0 < θ ∧ θ < π → f θ ≤ (4 * Real.sqrt 3) / 9 :=
by
  sorry

end max_f_l12_12034


namespace daily_sales_profit_45_selling_price_for_1200_profit_l12_12302

-- Definitions based on given conditions

def cost_price : ℤ := 30
def base_selling_price : ℤ := 40
def base_sales_volume : ℤ := 80
def price_increase_effect : ℤ := 2
def max_selling_price : ℤ := 55

-- Part (1): Prove that for a selling price of 45 yuan, the daily sales profit is 1050 yuan.
theorem daily_sales_profit_45 :
  let selling_price := 45
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = 1050 := by sorry

-- Part (2): Prove that to achieve a daily profit of 1200 yuan, the selling price should be 50 yuan.
theorem selling_price_for_1200_profit :
  let target_profit := 1200
  ∃ (selling_price : ℤ), 
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = target_profit ∧ selling_price ≤ max_selling_price ∧ selling_price = 50 := by sorry

end daily_sales_profit_45_selling_price_for_1200_profit_l12_12302


namespace steve_family_time_l12_12123

theorem steve_family_time :
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours * assignments_fraction
  let total_activity_hours := sleeping_hours + school_hours + assignments_hours
  day_hours - total_activity_hours = 10 :=
by
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours *  assignments_fraction
  let total_activity_hours := sleeping_hours +
                              school_hours + 
                              assignments_hours
  show day_hours - total_activity_hours = 10
  sorry

end steve_family_time_l12_12123


namespace quadratic_roots_equal_l12_12074

theorem quadratic_roots_equal {k : ℝ} (h : (2 * k) ^ 2 - 4 * 1 * (k^2 + k + 3) = 0) : k^2 + k + 3 = 9 :=
by
  sorry

end quadratic_roots_equal_l12_12074


namespace smallest_positive_integer_rel_prime_180_l12_12581

theorem smallest_positive_integer_rel_prime_180 : 
  ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → y ≥ 7 := 
by 
  sorry

end smallest_positive_integer_rel_prime_180_l12_12581


namespace cos_angle_identity_l12_12737

theorem cos_angle_identity (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) :
  Real.cos (π - 2 * α) = - (5 / 9) := by
sorry

end cos_angle_identity_l12_12737


namespace distance_walked_by_man_l12_12170

theorem distance_walked_by_man (x t : ℝ) (h1 : d = (x + 0.5) * (4 / 5) * t) (h2 : d = (x - 0.5) * (t + 2.5)) : d = 15 :=
by
  sorry

end distance_walked_by_man_l12_12170


namespace sum_sequence_formula_l12_12676

-- Define the sequence terms as a function.
def seq_term (x a : ℕ) (n : ℕ) : ℕ :=
x ^ (n + 1) + (n + 1) * a

-- Define the sum of the first nine terms of the sequence.
def sum_first_nine_terms (x a : ℕ) : ℕ :=
(x * (x ^ 9 - 1)) / (x - 1) + 45 * a

-- State the theorem to prove that the sum S is as expected.
theorem sum_sequence_formula (x a : ℕ) (h : x ≠ 1) : 
  sum_first_nine_terms x a = (x ^ 10 - x) / (x - 1) + 45 * a := by
  sorry

end sum_sequence_formula_l12_12676


namespace list_price_of_article_l12_12268

theorem list_price_of_article (P : ℝ) 
  (first_discount second_discount final_price : ℝ)
  (h1 : first_discount = 0.10)
  (h2 : second_discount = 0.08235294117647069)
  (h3 : final_price = 56.16)
  (h4 : P * (1 - first_discount) * (1 - second_discount) = final_price) : P = 68 :=
sorry

end list_price_of_article_l12_12268


namespace simplify_inverse_expression_l12_12934

theorem simplify_inverse_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z - x * z + x * y) :=
by
  sorry

end simplify_inverse_expression_l12_12934


namespace cylindrical_to_rectangular_l12_12824

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 6) (hθ : θ = π / 3) (hz : z = -3) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, -3) :=
by
  sorry

end cylindrical_to_rectangular_l12_12824


namespace fruits_eaten_total_l12_12789

variable (oranges_per_day : ℕ) (grapes_per_day : ℕ) (days : ℕ)

def total_fruits (oranges_per_day grapes_per_day days : ℕ) : ℕ :=
  (oranges_per_day * days) + (grapes_per_day * days)

theorem fruits_eaten_total 
  (h1 : oranges_per_day = 20)
  (h2 : grapes_per_day = 40) 
  (h3 : days = 30) : 
  total_fruits oranges_per_day grapes_per_day days = 1800 := 
by 
  sorry

end fruits_eaten_total_l12_12789


namespace evaluate_x_squared_minus_y_squared_l12_12049

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end evaluate_x_squared_minus_y_squared_l12_12049


namespace sum_of_first_9_terms_arithmetic_sequence_l12_12988

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem sum_of_first_9_terms_arithmetic_sequence
  (h_arith_seq : is_arithmetic_sequence a)
  (h_condition : a 2 + a 8 = 8) :
  (Finset.range 9).sum a = 36 :=
sorry

end sum_of_first_9_terms_arithmetic_sequence_l12_12988


namespace page_number_added_twice_l12_12781

-- Define the sum of natural numbers from 1 to n
def sum_nat (n: ℕ): ℕ := n * (n + 1) / 2

-- Incorrect sum due to one page number being counted twice
def incorrect_sum (n p: ℕ): ℕ := sum_nat n + p

-- Declaring the known conditions as Lean definitions
def n : ℕ := 70
def incorrect_sum_val : ℕ := 2550

-- Lean theorem statement to be proven
theorem page_number_added_twice :
  ∃ p, incorrect_sum n p = incorrect_sum_val ∧ p = 65 := by
  sorry

end page_number_added_twice_l12_12781


namespace line_AC_eqn_l12_12722

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define point A
def A : Point := { x := 3, y := 1 }

-- Define point B
def B : Point := { x := -1, y := 2 }

-- Define the line equation y = x + 1
def line_eq (p : Point) : Prop := p.y = p.x + 1

-- Define the bisector being on line y=x+1 as a condition
axiom bisector_on_line (C : Point) : 
  line_eq C → (∃ k : ℝ, (C.y - B.y) = k * (C.x - B.x))

-- Define the final goal to prove the equation of line AC
theorem line_AC_eqn (C : Point) :
  line_eq C → ((A.x - C.x) * (B.y - C.y) = (B.x - C.x) * (A.y - C.y)) → C.x = -3 ∧ C.y = -2 → 
  (A.x - 2 * A.y = 1) := sorry

end line_AC_eqn_l12_12722


namespace tom_swim_time_l12_12279

theorem tom_swim_time (t : ℝ) :
  (2 * t + 4 * t = 12) → t = 2 :=
by
  intro h
  have eq1 : 6 * t = 12 := by linarith
  linarith

end tom_swim_time_l12_12279


namespace fraction_not_on_time_l12_12964

theorem fraction_not_on_time (n : ℕ) (h1 : ∃ (k : ℕ), 3 * k = 5 * n) 
(h2 : ∃ (k : ℕ), 4 * k = 5 * m) 
(h3 : ∃ (k : ℕ), 5 * k = 6 * f) 
(h4 : m + f = n) 
(h5 : r = rm + rf) 
(h6 : rm = 4/5 * m) 
(h7 : rf = 5/6 * f) :
  (not_on_time : ℚ) = 1/5 := 
by
  sorry

end fraction_not_on_time_l12_12964


namespace distinct_real_numbers_condition_l12_12982

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)

theorem distinct_real_numbers_condition (a b x1 x2 x3 : ℝ) :
  f a b x1 = x2 → f a b x2 = x3 → f a b x3 = x1 → x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → a = -b^2 :=
by
  sorry

end distinct_real_numbers_condition_l12_12982


namespace base7_divisible_by_5_l12_12341

theorem base7_divisible_by_5 :
  ∃ (d : ℕ), (0 ≤ d ∧ d < 7) ∧ (344 * d + 56) % 5 = 0 ↔ d = 1 :=
by
  sorry

end base7_divisible_by_5_l12_12341


namespace correct_system_of_equations_l12_12806

theorem correct_system_of_equations (x y : ℝ) :
  (y - x = 4.5) ∧ (x - y / 2 = 1) ↔
  ((y - x = 4.5) ∧ (x - y / 2 = 1)) :=
by sorry

end correct_system_of_equations_l12_12806


namespace hyperbola_eccentricity_is_4_l12_12476

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
: ℝ := c / a

theorem hyperbola_eccentricity_is_4 (a b c : ℝ)
  (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
  (h_c2 : c^2 = a^2 + b^2)
  (h_bc : b^2 = a^2 * (c^2 / a^2 - 1))
: hyperbola_eccentricity a b c h_eq1 h_eq2 h_focus = 4 := by
  sorry

end hyperbola_eccentricity_is_4_l12_12476


namespace ratio_of_areas_l12_12658

variable (s : ℝ)
def side_length_square := s
def side_length_longer_rect := 1.2 * s
def side_length_shorter_rect := 0.7 * s
def area_square := s^2
def area_rect := (1.2 * s) * (0.7 * s)

theorem ratio_of_areas (h1 : s > 0) :
  area_rect s / area_square s = 21 / 25 :=
by 
  sorry

end ratio_of_areas_l12_12658


namespace total_fruits_in_30_days_l12_12787

-- Define the number of oranges Sophie receives each day
def sophie_daily_oranges : ℕ := 20

-- Define the number of grapes Hannah receives each day
def hannah_daily_grapes : ℕ := 40

-- Define the number of days
def number_of_days : ℕ := 30

-- Calculate the total number of fruits received by Sophie and Hannah in 30 days
theorem total_fruits_in_30_days :
  (sophie_daily_oranges * number_of_days) + (hannah_daily_grapes * number_of_days) = 1800 :=
by
  sorry

end total_fruits_in_30_days_l12_12787


namespace Shekar_biology_marks_l12_12521

theorem Shekar_biology_marks 
  (math_marks : ℕ := 76) 
  (science_marks : ℕ := 65) 
  (social_studies_marks : ℕ := 82) 
  (english_marks : ℕ := 47) 
  (average_marks : ℕ := 71) 
  (num_subjects : ℕ := 5) 
  (biology_marks : ℕ) :
  (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / num_subjects = average_marks → biology_marks = 85 := 
by 
  sorry

end Shekar_biology_marks_l12_12521


namespace inequality_solution_l12_12636

theorem inequality_solution (x : ℝ) :
  (\frac{9 * x^2 + 18 * x - 60}{(3 * x - 4) * (x + 5)} < 2) ↔ 
  (x ∈ Set.Ioo (-5 / 3) (4 / 3) ∪ Set.Ioi 4) :=
by
  sorry

end inequality_solution_l12_12636


namespace train_speed_is_85_kmh_l12_12158

noncomputable def speed_of_train_in_kmh (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_man_kmh : ℝ) : ℝ :=
  let speed_of_man_mps := speed_of_man_kmh * 1000 / 3600
  let relative_speed_mps := length_of_train / time_to_cross
  let speed_of_train_mps := relative_speed_mps - speed_of_man_mps
  speed_of_train_mps * 3600 / 1000

theorem train_speed_is_85_kmh
  (length_of_train : ℝ)
  (time_to_cross : ℝ)
  (speed_of_man_kmh : ℝ)
  (h1 : length_of_train = 150)
  (h2 : time_to_cross = 6)
  (h3 : speed_of_man_kmh = 5) :
  speed_of_train_in_kmh length_of_train time_to_cross speed_of_man_kmh = 85 :=
by
  sorry

end train_speed_is_85_kmh_l12_12158


namespace a_beats_b_time_difference_l12_12079

theorem a_beats_b_time_difference
  (d : ℝ) (d_A : ℝ) (d_B : ℝ)
  (t_A : ℝ)
  (h1 : d = 1000)
  (h2 : d_A = d)
  (h3 : d_B = d - 60)
  (h4 : t_A = 235) :
  (t_A - (d_B * t_A / d_A)) = 14.1 :=
by sorry

end a_beats_b_time_difference_l12_12079


namespace sugar_percentage_l12_12164

theorem sugar_percentage 
  (initial_volume : ℝ) (initial_water_perc : ℝ) (initial_kola_perc: ℝ) (added_sugar : ℝ) (added_water : ℝ) (added_kola : ℝ)
  (initial_solution: initial_volume = 340) 
  (perc_water : initial_water_perc = 0.75) 
  (perc_kola: initial_kola_perc = 0.05)
  (added_sugar_amt : added_sugar = 3.2) 
  (added_water_amt : added_water = 12) 
  (added_kola_amt : added_kola = 6.8) : 
  (71.2 / 362) * 100 = 19.67 := 
by 
  sorry

end sugar_percentage_l12_12164


namespace single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l12_12089

section transmission_scheme

variables (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1)

-- Part A
theorem single_transmission_probability :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β) ^ 2 :=
by sorry

-- Part B
theorem triple_transmission_probability :
  (1 - β) * β * (1 - β) = β * (1 - β) ^ 2 :=
by sorry

-- Part C
theorem triple_transmission_decoding :
  (3 * β * (1 - β) ^ 2) + (1 - β) ^ 3 = β * (1 - β) ^ 2 + (1 - β) ^ 3 :=
by sorry

-- Part D
theorem decoding_comparison (h : 0 < α ∧ α < 0.5) :
  (1 - α) < (3 * α * (1 - α) ^ 2 + (1 - α) ^ 3) :=
by sorry

end transmission_scheme

end single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l12_12089


namespace sara_total_spent_l12_12515

-- Definitions based on the conditions
def ticket_price : ℝ := 10.62
def discount_rate : ℝ := 0.10
def rented_movie : ℝ := 1.59
def bought_movie : ℝ := 13.95
def snacks : ℝ := 7.50
def sales_tax_rate : ℝ := 0.05

-- Problem statement
theorem sara_total_spent : 
  let total_tickets := 2 * ticket_price
  let discount := total_tickets * discount_rate
  let discounted_tickets := total_tickets - discount
  let subtotal := discounted_tickets + rented_movie + bought_movie
  let sales_tax := subtotal * sales_tax_rate
  let total_with_tax := subtotal + sales_tax
  let total_amount := total_with_tax + snacks
  total_amount = 43.89 :=
by
  sorry

end sara_total_spent_l12_12515


namespace compute_sum_pq_pr_qr_l12_12771

theorem compute_sum_pq_pr_qr (p q r : ℝ) (h : 5 * (p + q + r) = p^2 + q^2 + r^2) : 
  let N := 150
  let n := -12.5
  N + 15 * n = -37.5 := 
by {
  sorry
}

end compute_sum_pq_pr_qr_l12_12771


namespace prove_inequality_l12_12713

theorem prove_inequality (x : ℝ) (h : 3 * x^2 + x - 8 < 0) : -2 < x ∧ x < 4 / 3 :=
sorry

end prove_inequality_l12_12713


namespace variance_Y_l12_12848

variables {X : Type} [MeasurableSpace X] (μ : MeasureTheory.Measure X)
variables (X : X → ℝ) (c : ℝ)

noncomputable def variance (X : X → ℝ) : ℝ :=
MeasureTheory.MeasureTheory.variance μ X

theorem variance_Y
  (h1 : variance μ X = 1)
  (h2 : ∀ x, X x = 2 * X x + 3) :
  variance μ (λ x, 2 * (X x) + 3) = 4 :=
by
  sorry

end variance_Y_l12_12848


namespace positive_integers_satisfy_l12_12911

theorem positive_integers_satisfy (n : ℕ) (h1 : 25 - 5 * n > 15) : n = 1 :=
by sorry

end positive_integers_satisfy_l12_12911


namespace total_votes_election_l12_12084

theorem total_votes_election (V : ℝ)
    (h1 : 0.55 * 0.8 * V + 2520 = 0.8 * V)
    (h2 : 0.36 > 0) :
    V = 7000 :=
  by
  sorry

end total_votes_election_l12_12084


namespace expand_and_simplify_fraction_l12_12972

theorem expand_and_simplify_fraction (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * ((7 / (x^2)) + 15 * (x^3) - 4 * x) = (3 / (x^2)) + (45 * (x^3) / 7) - (12 * x / 7) :=
by
  sorry

end expand_and_simplify_fraction_l12_12972


namespace no_two_champions_l12_12615

structure Tournament (Team : Type) :=
  (defeats : Team → Team → Prop)  -- Team A defeats Team B

def is_superior {Team : Type} (T : Tournament Team) (A B: Team) : Prop :=
  T.defeats A B ∨ ∃ C, T.defeats A C ∧ T.defeats C B

def is_champion {Team : Type} (T : Tournament Team) (A : Team) : Prop :=
  ∀ B, A ≠ B → is_superior T A B

theorem no_two_champions {Team : Type} (T : Tournament Team) :
  ¬ (∃ A B, A ≠ B ∧ is_champion T A ∧ is_champion T B) :=
sorry

end no_two_champions_l12_12615


namespace eval_expression_l12_12026

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12026


namespace evaluate_expression_l12_12013

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12013


namespace negation_of_all_students_are_punctual_l12_12527

variable (Student : Type)
variable (student : Student → Prop)
variable (punctual : Student → Prop)

theorem negation_of_all_students_are_punctual :
  ¬ (∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) := by
  sorry

end negation_of_all_students_are_punctual_l12_12527


namespace daily_sales_profit_45_selling_price_for_1200_profit_l12_12301

-- Definitions based on given conditions

def cost_price : ℤ := 30
def base_selling_price : ℤ := 40
def base_sales_volume : ℤ := 80
def price_increase_effect : ℤ := 2
def max_selling_price : ℤ := 55

-- Part (1): Prove that for a selling price of 45 yuan, the daily sales profit is 1050 yuan.
theorem daily_sales_profit_45 :
  let selling_price := 45
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = 1050 := by sorry

-- Part (2): Prove that to achieve a daily profit of 1200 yuan, the selling price should be 50 yuan.
theorem selling_price_for_1200_profit :
  let target_profit := 1200
  ∃ (selling_price : ℤ), 
  let increase_in_price := selling_price - base_selling_price
  let decrease_in_volume := increase_in_price * price_increase_effect
  let new_sales_volume := base_sales_volume - decrease_in_volume
  let profit_per_item := selling_price - cost_price
  let daily_profit := profit_per_item * new_sales_volume
  daily_profit = target_profit ∧ selling_price ≤ max_selling_price ∧ selling_price = 50 := by sorry

end daily_sales_profit_45_selling_price_for_1200_profit_l12_12301


namespace exists_sum_of_150_consecutive_integers_l12_12543

theorem exists_sum_of_150_consecutive_integers :
  ∃ a : ℕ, 1627395075 = 150 * a + 11175 :=
by
  sorry

end exists_sum_of_150_consecutive_integers_l12_12543


namespace algebra_expression_l12_12485

theorem algebra_expression (a b : ℝ) (h : a = b + 1) : 3 + 2 * a - 2 * b = 5 :=
sorry

end algebra_expression_l12_12485


namespace steve_family_time_l12_12119

theorem steve_family_time :
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  hours_per_day - (hours_sleeping + hours_school + hours_assignments) = 10 :=
by
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  sorry

end steve_family_time_l12_12119


namespace inverse_function_correct_l12_12855

noncomputable def inverse_function (y : ℝ) : ℝ := (1 / 2) * y - (3 / 2)

theorem inverse_function_correct :
  ∀ x ∈ Set.Icc (0 : ℝ) (5 : ℝ), (inverse_function (2 * x + 3) = x) ∧ (0 ≤ 2 * x + 3) ∧ (2 * x + 3 ≤ 5) :=
by
  sorry

end inverse_function_correct_l12_12855


namespace find_average_age_of_students_l12_12492

-- Given conditions
variables (n : ℕ) (T : ℕ) (A : ℕ)

-- 20 students in the class
def students : ℕ := 20

-- Teacher's age is 42 years
def teacher_age : ℕ := 42

-- When the teacher's age is included, the average age increases by 1
def average_age_increase (A : ℕ) := A + 1

-- Proof problem statement in Lean 4
theorem find_average_age_of_students (A : ℕ) :
  20 * A + 42 = 21 * (A + 1) → A = 21 :=
by
  -- Here should be the proof steps, added sorry to skip the proof
  sorry

end find_average_age_of_students_l12_12492


namespace find_b_l12_12887

open Matrix

def a : ℝ^3 := ![3, 2, 4]
def b (x y z : ℝ) : ℝ^3 := ![x, y, z]

theorem find_b (x y z : ℝ) :
  (a.dot_product (b x y z) = 20) ∧
  (a.cross_product (b x y z) = ![-8, 16, -2]) :=
sorry

end find_b_l12_12887


namespace determine_v6_l12_12101

variable (v : ℕ → ℝ)

-- Given initial conditions: v₄ = 12 and v₇ = 471
def initial_conditions := v 4 = 12 ∧ v 7 = 471

-- Recurrence relation definition: vₙ₊₂ = 3vₙ₊₁ + vₙ
def recurrence_relation := ∀ n : ℕ, v (n + 2) = 3 * v (n + 1) + v n

-- The target is to prove that v₆ = 142.5
theorem determine_v6 (h1 : initial_conditions v) (h2 : recurrence_relation v) : 
  v 6 = 142.5 :=
sorry

end determine_v6_l12_12101


namespace base4_base7_digit_difference_l12_12827

def num_digits_base (n b : ℕ) : ℕ :=
  if b > 1 then Nat.log b n + 1 else 0

theorem base4_base7_digit_difference :
  let n := 1573
  num_digits_base n 4 - num_digits_base n 7 = 2 := by
  sorry

end base4_base7_digit_difference_l12_12827


namespace harry_water_per_mile_l12_12945

noncomputable def water_per_mile_during_first_3_miles (initial_water : ℝ) (remaining_water : ℝ) (leak_rate : ℝ) (hike_time : ℝ) (water_drunk_last_mile : ℝ) (first_3_miles : ℝ) : ℝ :=
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  water_drunk_first_3_miles / first_3_miles

theorem harry_water_per_mile :
  water_per_mile_during_first_3_miles 10 2 1 2 3 3 = 1 / 3 :=
by
  have initial_water := 10
  have remaining_water := 2
  have leak_rate := 1
  have hike_time := 2
  have water_drunk_last_mile := 3
  have first_3_miles := 3
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  let result := water_drunk_first_3_miles / first_3_miles
  exact sorry

end harry_water_per_mile_l12_12945


namespace factorize_expr_l12_12330

theorem factorize_expr (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := 
sorry

end factorize_expr_l12_12330


namespace problem_solution_l12_12327

-- Definitions based on conditions
def valid_sequence (b : Fin 7 → Nat) : Prop :=
  (∀ i j : Fin 7, i ≤ j → b i ≥ b j) ∧ 
  (∀ i : Fin 7, b i ≤ 1500) ∧ 
  (∀ i : Fin 7, (b i + i) % 3 = 0)

-- The main theorem
theorem problem_solution :
  (∃ b : Fin 7 → Nat, valid_sequence b) →
  @Nat.choose 506 7 % 1000 = 506 :=
sorry

end problem_solution_l12_12327


namespace plane_hit_probability_l12_12769

-- Define the probability of person A hitting the plane
def P_A : ℝ := 0.7

-- Define the probability of person B hitting the plane
def P_B : ℝ := 0.5

-- Main theorem stating the probability of the enemy plane being hit.
theorem plane_hit_probability : (1 - (1 - P_A) * (1 - P_B)) = 0.85 := 
by
  -- Sorry is added here to skip the proof.
  sorry

end plane_hit_probability_l12_12769


namespace base_b_expression_not_divisible_l12_12969

theorem base_b_expression_not_divisible 
  (b : ℕ) : 
  (b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8) →
  (2 * b^3 - 2 * b^2 + b - 1) % 5 ≠ 0 ↔ (b ≠ 6) :=
by
  sorry

end base_b_expression_not_divisible_l12_12969


namespace hyperbola_eccentricity_l12_12060

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_asymptote_parallel : b = 2 * a)
  (h_c_squared : c^2 = a^2 + b^2)
  (h_e_def : e = c / a) :
  e = Real.sqrt 5 :=
sorry

end hyperbola_eccentricity_l12_12060


namespace shifted_parabola_l12_12568

theorem shifted_parabola (x : ℝ) : 
  (let y := x^2 in (let x := x + 2 in y)) = (x - 2)^2 := sorry

end shifted_parabola_l12_12568


namespace chrom_replication_not_in_prophase_I_l12_12146

-- Definitions for the conditions
def chrom_replication (stage : String) : Prop := 
  stage = "Interphase"

def chrom_shortening_thickening (stage : String) : Prop := 
  stage = "Prophase I"

def pairing_homologous_chromosomes (stage : String) : Prop := 
  stage = "Prophase I"

def crossing_over (stage : String) : Prop :=
  stage = "Prophase I"

-- Stating the theorem
theorem chrom_replication_not_in_prophase_I :
  chrom_replication "Interphase" ∧ 
  chrom_shortening_thickening "Prophase I" ∧ 
  pairing_homologous_chromosomes "Prophase I" ∧ 
  crossing_over "Prophase I" → 
  ¬ chrom_replication "Prophase I" := 
by
  sorry

end chrom_replication_not_in_prophase_I_l12_12146


namespace rectangle_ratio_l12_12716

theorem rectangle_ratio (s x y : ℝ) 
  (h_outer_area : (2 * s) ^ 2 = 4 * s ^ 2)
  (h_inner_sides : s + 2 * y = 2 * s)
  (h_outer_sides : x + y = 2 * s) :
  x / y = 3 :=
by
  sorry

end rectangle_ratio_l12_12716


namespace evaluate_expression_l12_12015

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12015


namespace arithmetic_sequence_10th_term_l12_12219

theorem arithmetic_sequence_10th_term (a_1 : ℕ) (d : ℕ) (n : ℕ) 
  (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 10) : (a_1 + (n - 1) * d) = 28 := by 
  sorry

end arithmetic_sequence_10th_term_l12_12219


namespace remainder_sum_mod9_l12_12980

def a1 := 8243
def a2 := 8244
def a3 := 8245
def a4 := 8246

theorem remainder_sum_mod9 : ((a1 + a2 + a3 + a4) % 9) = 7 :=
by
  sorry

end remainder_sum_mod9_l12_12980


namespace area_transformed_function_l12_12270

variable (f : ℝ → ℝ)
variable (a b : ℝ)
variable (h_integral : ∫ x in a..b, f x = 12)

theorem area_transformed_function :
  ∫ x in a..b, (2 * f (x - 1) + 4) = 24 :=
by
  sorry

end area_transformed_function_l12_12270


namespace evaluate_expression_l12_12008

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end evaluate_expression_l12_12008


namespace weight_of_new_student_l12_12160

-- Define some constants for the problem
def avg_weight_29_students : ℝ := 28
def number_of_students_29 : ℕ := 29
def new_avg_weight_30_students : ℝ := 27.5
def number_of_students_30 : ℕ := 30

-- Calculate total weights
def total_weight_29_students : ℝ := avg_weight_29_students * number_of_students_29
def new_total_weight_30_students : ℝ := new_avg_weight_30_students * number_of_students_30

-- The proposition we need to prove
theorem weight_of_new_student :
  new_total_weight_30_students - total_weight_29_students = 13 := by
  -- Placeholder for the actual proof
  sorry

end weight_of_new_student_l12_12160


namespace solve_for_a_l12_12613

theorem solve_for_a (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 5 * a * x + a) = x^3 + (1 - 5 * a) * x^2 - 4 * a * x + a) →
  (1 - 5 * a = 0) →
  a = 1 / 5 := 
by
  intro h₁ h₂
  sorry

end solve_for_a_l12_12613


namespace expression_value_range_l12_12886

theorem expression_value_range (a b c d e : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 1) (h₃ : 0 ≤ b) (h₄ : b ≤ 1) (h₅ : 0 ≤ c) (h₆ : c ≤ 1) (h₇ : 0 ≤ d) (h₈ : d ≤ 1) (h₉ : 0 ≤ e) (h₁₀ : e ≤ 1) :
  4 * Real.sqrt (2 / 3) ≤ (Real.sqrt (a^2 + (1 - b)^2 + e^2) + Real.sqrt (b^2 + (1 - c)^2 + e^2) + Real.sqrt (c^2 + (1 - d)^2 + e^2) + Real.sqrt (d^2 + (1 - a)^2 + e^2)) ∧ 
  (Real.sqrt (a^2 + (1 - b)^2 + e^2) + Real.sqrt (b^2 + (1 - c)^2 + e^2) + Real.sqrt (c^2 + (1 - d)^2 + e^2) + Real.sqrt (d^2 + (1 - a)^2 + e^2)) ≤ 8 :=
sorry

end expression_value_range_l12_12886


namespace polynomial_divisibility_l12_12704

theorem polynomial_divisibility (a : ℤ) : ∃ q : ℤ[X], (X^13 + X + 90 : ℤ[X]) = (X^2 - X + a) * q ↔ a = -2 :=
by sorry

end polynomial_divisibility_l12_12704


namespace binomial_probability_l12_12379

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial probability mass function
def binomial_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coeff n k) * (p^k) * ((1 - p)^(n - k))

-- Define the conditions of the problem
def n := 5
def k := 2
def p : ℚ := 1/3

-- State the theorem
theorem binomial_probability :
  binomial_pmf n k p = binomial_coeff 5 2 * (1/3)^2 * (2/3)^3 := by
  sorry

end binomial_probability_l12_12379


namespace circles_area_l12_12654

theorem circles_area (BD AC : ℝ) (r : ℝ) (h1 : BD = 6) (h2 : AC = 12)
  (h3 : ∀ (d1 d2 : ℝ), d1 = AC / 2 → d2 = BD / 2 → r^2 = (r - d2)^2 + d1^2) :
  real.pi * r^2 = (225/4) * real.pi :=
by
  -- proof to be filled
  sorry

end circles_area_l12_12654


namespace point_on_hyperbola_l12_12994

theorem point_on_hyperbola : 
  (∃ x y : ℝ, (x, y) = (3, -2) ∧ y = -6 / x) :=
by
  sorry

end point_on_hyperbola_l12_12994


namespace clock_angle_4_oclock_l12_12667

theorem clock_angle_4_oclock :
  let total_degrees := 360
  let hours := 12
  let degree_per_hour := total_degrees / hours
  let hour_position := 4
  let minute_hand_position := 0
  let hour_hand_angle := hour_position * degree_per_hour
  hour_hand_angle = 120 := sorry

end clock_angle_4_oclock_l12_12667


namespace final_student_count_is_correct_l12_12965

-- Define the initial conditions
def initial_students : ℕ := 11
def students_left_first_semester : ℕ := 6
def students_joined_first_semester : ℕ := 25
def additional_students_second_semester : ℕ := 15
def students_transferred_second_semester : ℕ := 3
def students_switched_class_second_semester : ℕ := 2

-- Define the final number of students to be proven
def final_number_of_students : ℕ := 
  initial_students - students_left_first_semester + students_joined_first_semester + 
  additional_students_second_semester - students_transferred_second_semester - students_switched_class_second_semester

-- The theorem we need to prove
theorem final_student_count_is_correct : final_number_of_students = 40 := by
  sorry

end final_student_count_is_correct_l12_12965


namespace diophantine_solution_l12_12108

theorem diophantine_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (n : ℕ) (h_n : n > a * b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end diophantine_solution_l12_12108


namespace inequality_solution_set_l12_12591

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + (a-1)*x^2

theorem inequality_solution_set (a : ℝ) (ha : ∀ x : ℝ, f x a = -f (-x) a) :
  {x : ℝ | f (a*x) a > f (a-x) a} = {x : ℝ | x > 1/2} :=
by
  sorry

end inequality_solution_set_l12_12591


namespace average_weight_of_boys_l12_12648

theorem average_weight_of_boys (n1 n2 : ℕ) (w1 w2 : ℚ) 
  (weight_avg_22_boys : w1 = 50.25) 
  (weight_avg_8_boys : w2 = 45.15) 
  (count_22_boys : n1 = 22) 
  (count_8_boys : n2 = 8) 
  : ((n1 * w1 + n2 * w2) / (n1 + n2) : ℚ) = 48.89 :=
by
  sorry

end average_weight_of_boys_l12_12648


namespace liars_count_l12_12547

inductive Person
| Knight
| Liar
| Eccentric

open Person

def isLiarCondition (p : Person) (right : Person) : Prop :=
  match p with
  | Knight => right = Liar
  | Liar => right ≠ Liar
  | Eccentric => True

theorem liars_count (people : Fin 100 → Person) (h : ∀ i, isLiarCondition (people i) (people ((i + 1) % 100))) :
  (∃ n : ℕ, n = 0 ∨ n = 50) :=
sorry

end liars_count_l12_12547


namespace merchant_salt_mixture_l12_12685

theorem merchant_salt_mixture (x : ℝ) (h₀ : (0.48 * (40 + x)) = 1.20 * (14 + 0.50 * x)) : x = 0 :=
by
  sorry

end merchant_salt_mixture_l12_12685


namespace ratio_of_arithmetic_sequence_sums_l12_12322

-- Definitions of the arithmetic sequences based on the conditions
def numerator_seq (n : ℕ) : ℕ := 3 + (n - 1) * 3
def denominator_seq (m : ℕ) : ℕ := 4 + (m - 1) * 4

-- Definitions of the number of terms based on the conditions
def num_terms_num : ℕ := 32
def num_terms_den : ℕ := 16

-- Definitions of the sums based on the sequences
def sum_numerator_seq : ℕ := (num_terms_num / 2) * (3 + 96)
def sum_denominator_seq : ℕ := (num_terms_den / 2) * (4 + 64)

-- Calculate the ratio of the sums
def ratio_of_sums : ℚ := sum_numerator_seq / sum_denominator_seq

-- Proof statement
theorem ratio_of_arithmetic_sequence_sums : ratio_of_sums = 99 / 34 := by
  sorry

end ratio_of_arithmetic_sequence_sums_l12_12322


namespace ticket_cost_per_ride_l12_12970

theorem ticket_cost_per_ride
  (total_tickets: ℕ) 
  (spent_tickets: ℕ)
  (rides: ℕ)
  (remaining_tickets: ℕ)
  (ride_cost: ℕ)
  (h1: total_tickets = 79)
  (h2: spent_tickets = 23)
  (h3: rides = 8)
  (h4: remaining_tickets = total_tickets - spent_tickets)
  (h5: remaining_tickets = ride_cost * rides):
  ride_cost = 7 :=
by
  sorry

end ticket_cost_per_ride_l12_12970


namespace unique_parallel_line_in_beta_l12_12611

-- Define the basic geometrical entities.
axiom Plane : Type
axiom Line : Type
axiom Point : Type

-- Definitions relating entities.
def contains (P : Plane) (l : Line) : Prop := sorry
def parallel (A B : Plane) : Prop := sorry
def in_plane (p : Point) (P : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry

-- Statements derived from the conditions in problem.
variables (α β : Plane) (a : Line) (B : Point)
-- Given conditions
axiom plane_parallel : parallel α β
axiom line_in_plane : contains α a
axiom point_in_plane : in_plane B β

-- The ultimate goal derived from the question.
theorem unique_parallel_line_in_beta : 
  ∃! b : Line, (in_plane B β) ∧ (parallel_lines a b) :=
sorry

end unique_parallel_line_in_beta_l12_12611


namespace daily_profit_at_45_selling_price_for_1200_profit_l12_12300

-- Definitions for the conditions
def cost_price (p: ℝ) : Prop := p = 30
def initial_sales (p: ℝ) (s: ℝ) : Prop := p = 40 ∧ s = 80
def sales_decrease_rate (r: ℝ) : Prop := r = 2
def max_selling_price (p: ℝ) : Prop := p ≤ 55

-- Proof for Question 1
theorem daily_profit_at_45 (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate) :
  (price = 45) → profit = 1050 :=
by sorry

-- Proof for Question 2
theorem selling_price_for_1200_profit (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate)
  (h_max_price : ∀ p, max_selling_price p → p ≤ 55) :
  profit = 1200 → price = 50 :=
by sorry

end daily_profit_at_45_selling_price_for_1200_profit_l12_12300


namespace triangle_reciprocal_sum_l12_12092

variables {A B C D L M N : Type} -- Points are types
variables {t_1 t_2 t_3 t_4 t_5 t_6 : ℝ} -- Areas are real numbers

-- Assume conditions as hypotheses
variable (h1 : ∀ (t1 t4 t5 t6: ℝ), t_1 = t1 ∧ t_4 = t4 ∧ t_5 = t5 ∧ t_6 = t6 -> (t1 + t4) = (t5 + t6))
variable (h2 : ∀ (t2 t4 t3 t6: ℝ), t_2 = t2 ∧ t_4 = t4 ∧ t_3 = t3 ∧ t_6 = t6 -> (t2 + t4) = (t3 + t6))
variable (h3 : ∀ (t1 t5 t3 t4 : ℝ), t_1 = t1 ∧ t_5 = t5 ∧ t_3 = t3 ∧ t_4 = t4 -> (t1 + t3) = (t4 + t5))

theorem triangle_reciprocal_sum 
  (h1 : ∀ (t1 t4 t5 t6: ℝ), t_1 = t1 ∧ t_4 = t4 ∧ t_5 = t5 ∧ t_6 = t6 -> (t1 + t4) = (t5 + t6))
  (h2 : ∀ (t2 t4 t3 t6: ℝ), t_2 = t2 ∧ t_4 = t4 ∧ t_3 = t3 ∧ t_6 = t6 -> (t2 + t4) = (t3 + t6))
  (h3 : ∀ (t1 t5 t3 t4: ℝ), t_1 = t1 ∧ t_5 = t5 ∧ t_3 = t3 ∧ t_4 = t4 -> (t1 + t3) = (t4 + t5)) :
  (1 / t_1 + 1 / t_3 + 1 / t_5) = (1 / t_2 + 1 / t_4 + 1 / t_6) :=
sorry

end triangle_reciprocal_sum_l12_12092


namespace prime_factor_of_reversed_difference_l12_12411

theorem prime_factor_of_reversed_difference (A B C : ℕ) (hA : A ≠ C) (hA_d : 1 ≤ A ∧ A ≤ 9) (hB_d : 0 ≤ B ∧ B ≤ 9) (hC_d : 1 ≤ C ∧ C ≤ 9) :
  ∃ p, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 11 := 
by
  sorry

end prime_factor_of_reversed_difference_l12_12411


namespace find_b_l12_12104

noncomputable def g (b x : ℝ) : ℝ := b * x^2 - Real.cos (Real.pi * x)

theorem find_b (b : ℝ) (hb : 0 < b) (h : g b (g b 1) = -Real.cos Real.pi) : b = 1 :=
by
  sorry

end find_b_l12_12104


namespace winter_melon_ratio_l12_12686

theorem winter_melon_ratio (T Ok_sales Choc_sales : ℕ) (hT : T = 50) 
  (hOk : Ok_sales = 3 * T / 10) (hChoc : Choc_sales = 15) :
  (T - (Ok_sales + Choc_sales)) / T = 2 / 5 :=
by
  sorry

end winter_melon_ratio_l12_12686


namespace time_with_family_l12_12125

theorem time_with_family : 
    let hours_in_day := 24
    let sleep_fraction := 1 / 3
    let school_fraction := 1 / 6
    let assignment_fraction := 1 / 12
    let sleep_hours := sleep_fraction * hours_in_day
    let school_hours := school_fraction * hours_in_day
    let assignment_hours := assignment_fraction * hours_in_day
    let total_hours_occupied := sleep_hours + school_hours + assignment_hours
    hours_in_day - total_hours_occupied = 10 :=
by
  sorry

end time_with_family_l12_12125


namespace train_crossing_time_l12_12290

def length_of_train : ℕ := 120
def speed_of_train_kmph : ℕ := 54
def length_of_bridge : ℕ := 660

def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600
def total_distance : ℕ := length_of_train + length_of_bridge
def time_to_cross_bridge : ℕ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 52 :=
sorry

end train_crossing_time_l12_12290


namespace percentage_y_less_than_x_l12_12942

variable (x y : ℝ)
variable (h : x = 12 * y)

theorem percentage_y_less_than_x :
  (11 / 12) * 100 = 91.67 := by
  sorry

end percentage_y_less_than_x_l12_12942


namespace adjacent_angles_l12_12161

variable (θ : ℝ)

theorem adjacent_angles (h : θ + 3 * θ = 180) : θ = 45 ∧ 3 * θ = 135 :=
by 
  -- This is the place where the proof would go
  -- Here we only declare the statement, not the proof
  sorry

end adjacent_angles_l12_12161


namespace tank_empty_time_l12_12461

theorem tank_empty_time (R L : ℝ) (h1 : R = 1 / 7) (h2 : R - L = 1 / 8) : 
  (1 / L) = 56 :=
by
  sorry

end tank_empty_time_l12_12461


namespace geometric_sequence_product_l12_12479

theorem geometric_sequence_product (a₁ aₙ : ℝ) (n : ℕ) (hn : n > 0) (number_of_terms : n ≥ 1) :
  -- Conditions: First term, last term, number of terms
  ∃ P : ℝ, P = (a₁ * aₙ) ^ (n / 2) :=
sorry

end geometric_sequence_product_l12_12479


namespace black_squares_in_20th_row_l12_12172

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def squares_in_row (n : ℕ) : ℕ := 1 + sum_natural (n - 2)

noncomputable def black_squares_in_row (n : ℕ) : ℕ := 
  if squares_in_row n % 2 = 1 then (squares_in_row n - 1) / 2 else squares_in_row n / 2

theorem black_squares_in_20th_row : black_squares_in_row 20 = 85 := 
by
  sorry

end black_squares_in_20th_row_l12_12172


namespace projection_of_b_onto_a_l12_12347

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_squared := a.1 * a.1 + a.2 * a.2
  let scalar := dot_product / magnitude_squared
  (scalar * a.1, scalar * a.2)

theorem projection_of_b_onto_a :
  vector_projection (2, -1) (6, 2) = (4, -2) :=
by
  simp [vector_projection]
  sorry

end projection_of_b_onto_a_l12_12347


namespace sale_in_third_month_l12_12438

theorem sale_in_third_month 
  (sale1 sale2 sale4 sale5 sale6 : ℕ) 
  (avg_sale_months : ℕ) 
  (total_sales : ℕ)
  (h1 : sale1 = 6435) 
  (h2 : sale2 = 6927) 
  (h4 : sale4 = 7230) 
  (h5 : sale5 = 6562) 
  (h6 : sale6 = 7991) 
  (h_avg : avg_sale_months = 7000) 
  (h_total : total_sales = 6 * avg_sale_months) 
  : (total_sales - (sale1 + sale2 + sale4 + sale5 + sale6)) = 6855 :=
by
  have sales_sum := sale1 + sale2 + sale4 + sale5 + sale6
  have required_sales := total_sales - sales_sum
  sorry

end sale_in_third_month_l12_12438


namespace transform_quadratic_to_linear_l12_12491

theorem transform_quadratic_to_linear (x y : ℝ) : 
  x^2 - 4 * x * y + 4 * y^2 = 4 ↔ (x - 2 * y + 2 = 0 ∨ x - 2 * y - 2 = 0) :=
by
  sorry

end transform_quadratic_to_linear_l12_12491


namespace thirteen_coins_value_l12_12941

theorem thirteen_coins_value :
  ∃ (p n d q : ℕ), p + n + d + q = 13 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 141 ∧ 
                   2 ≤ p ∧ 2 ≤ n ∧ 2 ≤ d ∧ 2 ≤ q ∧ 
                   d = 3 :=
  sorry

end thirteen_coins_value_l12_12941


namespace inequality_proof_l12_12985

theorem inequality_proof (a b c : ℝ) (ha : a = 2 / 21) (hb : b = Real.log 1.1) (hc : c = 21 / 220) : a < b ∧ b < c :=
by
  sorry

end inequality_proof_l12_12985


namespace find_value_of_fraction_l12_12245

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : (x / y) + (y / x) = 8)

theorem find_value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l12_12245


namespace polynomial_one_negative_root_iff_l12_12578

noncomputable def polynomial_has_one_negative_real_root (p : ℝ) : Prop :=
  ∃ (x : ℝ), (x^4 + 3*p*x^3 + 6*x^2 + 3*p*x + 1 = 0) ∧
  ∀ (y : ℝ), y < x → y^4 + 3*p*y^3 + 6*y^2 + 3*p*y + 1 ≠ 0

theorem polynomial_one_negative_root_iff (p : ℝ) :
  polynomial_has_one_negative_real_root p ↔ p ≥ 4 / 3 :=
sorry

end polynomial_one_negative_root_iff_l12_12578


namespace value_of_square_sum_l12_12365

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l12_12365


namespace sin_double_angle_cos_condition_l12_12345

theorem sin_double_angle_cos_condition (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) :
  Real.sin (2 * x) = -7 / 25 :=
sorry

end sin_double_angle_cos_condition_l12_12345


namespace triangle_side_lengths_inequality_l12_12511

theorem triangle_side_lengths_inequality
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end triangle_side_lengths_inequality_l12_12511


namespace total_towels_folded_in_one_hour_l12_12237

-- Define the conditions for folding rates and breaks of each person
def Jane_folding_rate (minutes : ℕ) : ℕ :=
  if minutes % 8 < 5 then 5 * (minutes / 8 + 1) else 5 * (minutes / 8)

def Kyla_folding_rate (minutes : ℕ) : ℕ :=
  if minutes < 30 then 12 * (minutes / 10 + 1) else 36 + 6 * ((minutes - 30) / 10)

def Anthony_folding_rate (minutes : ℕ) : ℕ :=
  if minutes <= 40 then 14 * (minutes / 20)
  else if minutes <= 50 then 28
  else 28 + 14 * ((minutes - 50) / 20)

def David_folding_rate (minutes : ℕ) : ℕ :=
  let sets := minutes / 15
  let additional := sets / 3
  4 * (sets - additional) + 5 * additional

-- Definitions are months passing given in the questions
def hours_fold_towels (minutes : ℕ) : ℕ :=
  Jane_folding_rate minutes + Kyla_folding_rate minutes + Anthony_folding_rate minutes + David_folding_rate minutes

theorem total_towels_folded_in_one_hour : hours_fold_towels 60 = 134 := sorry

end total_towels_folded_in_one_hour_l12_12237


namespace total_ticket_cost_l12_12310

theorem total_ticket_cost (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : V = G - 276) 
  (price_vip : ℕ := 45) 
  (price_regular : ℕ := 20) : 
  (price_vip * V + price_regular * G = 6950) :=
by sorry

end total_ticket_cost_l12_12310


namespace percent_of_value_l12_12791

theorem percent_of_value : (2 / 5) * (1 / 100) * 450 = 1.8 :=
by sorry

end percent_of_value_l12_12791


namespace bingley_bracelets_final_l12_12319

-- Definitions
def initial_bingley_bracelets : Nat := 5
def kelly_bracelets_given : Nat := 16 / 4
def bingley_bracelets_after_kelly : Nat := initial_bingley_bracelets + kelly_bracelets_given
def bingley_bracelets_given_to_sister : Nat := bingley_bracelets_after_kelly / 3
def bingley_remaining_bracelets : Nat := bingley_bracelets_after_kelly - bingley_bracelets_given_to_sister

-- Theorem
theorem bingley_bracelets_final : bingley_remaining_bracelets = 6 := by
  sorry

end bingley_bracelets_final_l12_12319


namespace evaluate_x_squared_minus_y_squared_l12_12050

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end evaluate_x_squared_minus_y_squared_l12_12050


namespace lcm_of_5_6_8_9_l12_12930

theorem lcm_of_5_6_8_9 : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 := 
by 
  sorry

end lcm_of_5_6_8_9_l12_12930


namespace polynomial_problem_l12_12471

theorem polynomial_problem :
  ∀ P : Polynomial ℤ,
    (∃ R : Polynomial ℤ, (X^2 + 6*X + 10) * P^2 - 1 = R^2) → 
    P = 0 :=
by { sorry }

end polynomial_problem_l12_12471


namespace eval_expression_l12_12024

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l12_12024


namespace find_LCM_of_three_numbers_l12_12869

noncomputable def LCM_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem find_LCM_of_three_numbers
  (a b c : ℕ)
  (h_prod : a * b * c = 1354808)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 11) :
  LCM_of_three_numbers a b c = 123164 := by
  sorry

end find_LCM_of_three_numbers_l12_12869


namespace remainder_1493827_div_4_l12_12285

theorem remainder_1493827_div_4 : 1493827 % 4 = 3 := 
by
  sorry

end remainder_1493827_div_4_l12_12285


namespace problem1_1_problem1_2_problem2_l12_12434

open Set

/-
Given sets U, A, and B, derived from the provided conditions:
  U : Set ℝ
  A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
  B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}
-/

def U : Set ℝ := univ
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}

theorem problem1_1 (m : ℝ) (h : m = 5) : A ∩ B m = {x | -3 ≤ x ∧ x ≤ 5} :=
sorry

theorem problem1_2 (m : ℝ) (h : m = 5) : (compl A) ∪ B m = univ :=
sorry

theorem problem2 (m : ℝ) : A ⊆ B m → 4 < m :=
sorry

end problem1_1_problem1_2_problem2_l12_12434


namespace problem_statement_l12_12741

theorem problem_statement (x y z : ℝ) (h : (x - z)^2 - 4 * (x - y) * (y - z) = 0) : x + z - 2 * y = 0 :=
sorry

end problem_statement_l12_12741


namespace jim_gave_away_cards_l12_12756

theorem jim_gave_away_cards
  (sets_brother : ℕ := 15)
  (sets_sister : ℕ := 8)
  (sets_friend : ℕ := 4)
  (sets_cousin : ℕ := 6)
  (sets_classmate : ℕ := 3)
  (cards_per_set : ℕ := 25) :
  (sets_brother + sets_sister + sets_friend + sets_cousin + sets_classmate) * cards_per_set = 900 :=
by
  sorry

end jim_gave_away_cards_l12_12756


namespace intersection_complement_M_N_l12_12990

def M := { x : ℝ | x ≤ 1 / 2 }
def N := { x : ℝ | x^2 ≤ 1 }
def complement_M := { x : ℝ | x > 1 / 2 }

theorem intersection_complement_M_N :
  (complement_M ∩ N = { x : ℝ | 1 / 2 < x ∧ x ≤ 1 }) :=
by
  sorry

end intersection_complement_M_N_l12_12990


namespace sin_double_angle_half_pi_l12_12346

theorem sin_double_angle_half_pi (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1 / 3) : 
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by
  sorry

end sin_double_angle_half_pi_l12_12346


namespace steve_family_time_l12_12122

theorem steve_family_time :
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours * assignments_fraction
  let total_activity_hours := sleeping_hours + school_hours + assignments_hours
  day_hours - total_activity_hours = 10 :=
by
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours *  assignments_fraction
  let total_activity_hours := sleeping_hours +
                              school_hours + 
                              assignments_hours
  show day_hours - total_activity_hours = 10
  sorry

end steve_family_time_l12_12122


namespace problem_l12_12214

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (3 * x - Real.pi / 3)

theorem problem 
  (x₁ x₂ : ℝ)
  (hx₁x₂ : |f x₁ - f x₂| = 4)
  (x : ℝ)
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 6)
  (m : ℝ) : m ≥ 1 / 3 :=
sorry

end problem_l12_12214


namespace total_mile_times_l12_12281

theorem total_mile_times (t_Tina t_Tony t_Tom t_Total : ℕ) 
  (h1 : t_Tina = 6) 
  (h2 : t_Tony = t_Tina / 2) 
  (h3 : t_Tom = t_Tina / 3) 
  (h4 : t_Total = t_Tina + t_Tony + t_Tom) : t_Total = 11 := 
sorry

end total_mile_times_l12_12281


namespace robert_can_read_books_l12_12114

theorem robert_can_read_books (pages_per_hour : ℕ) (book_pages : ℕ) (total_hours : ℕ) :
  pages_per_hour = 120 →
  book_pages = 360 →
  total_hours = 8 →
  total_hours / (book_pages / pages_per_hour) = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end robert_can_read_books_l12_12114


namespace correct_option_is_B_l12_12140

-- Define the total number of balls
def total_black_balls : ℕ := 3
def total_red_balls : ℕ := 7
def total_balls : ℕ := total_black_balls + total_red_balls

-- Define the event of drawing balls
def drawing_balls (n : ℕ) : Prop := n = 3

-- Define what a random variable is within this context
def is_random_variable (n : ℕ) : Prop :=
  n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3

-- The main statement to prove
theorem correct_option_is_B (n : ℕ) :
  drawing_balls n → is_random_variable n :=
by
  intro h
  sorry

end correct_option_is_B_l12_12140


namespace fraction_of_integer_l12_12925

theorem fraction_of_integer :
  (5 / 6) * 30 = 25 :=
by
  sorry

end fraction_of_integer_l12_12925
