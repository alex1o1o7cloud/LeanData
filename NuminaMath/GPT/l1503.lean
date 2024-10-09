import Mathlib

namespace length_of_bridge_is_l1503_150365

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 21.998240140788738
noncomputable def speed_kmph : ℝ := 36
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is : bridge_length = 119.98240140788738 :=
by
  have speed_mps_val : speed_mps = 10 := by
    norm_num [speed_kmph, speed_mps]
  have total_distance_val : total_distance = 219.98240140788738 := by
    norm_num [total_distance, speed_mps_val, time_to_cross_bridge]
  have bridge_length_val : bridge_length = 119.98240140788738 := by
    norm_num [bridge_length, total_distance_val, train_length]
  exact bridge_length_val

end length_of_bridge_is_l1503_150365


namespace cost_of_product_l1503_150376

theorem cost_of_product (x : ℝ) (a : ℝ) (h : a > 0) :
  (1 + a / 100) * (x / (1 + a / 100)) = x :=
by
  field_simp [ne_of_gt h]
  sorry

end cost_of_product_l1503_150376


namespace floor_painting_cost_l1503_150392

noncomputable def floor_painting_problem : Prop := 
  ∃ (B L₁ L₂ B₂ Area₁ Area₂ CombinedCost : ℝ),
  L₁ = 2 * B ∧
  Area₁ = L₁ * B ∧
  484 = Area₁ * 3 ∧
  L₂ = 0.8 * L₁ ∧
  B₂ = 1.3 * B ∧
  Area₂ = L₂ * B₂ ∧
  CombinedCost = 484 + (Area₂ * 5) ∧
  CombinedCost = 1320.8

theorem floor_painting_cost : floor_painting_problem :=
by
  sorry

end floor_painting_cost_l1503_150392


namespace island_count_l1503_150330

-- Defining the conditions
def lakes := 7
def canals := 10

-- Euler's formula for connected planar graph
def euler_characteristic (V E F : ℕ) := V - E + F = 2

-- Determine the number of faces using Euler's formula
def faces (V E : ℕ) :=
  let F := V - E + 2
  F

-- The number of islands is the number of faces minus one for the outer face
def number_of_islands (F : ℕ) :=
  F - 1

-- The given proof problem to be converted to Lean
theorem island_count :
  number_of_islands (faces lakes canals) = 4 :=
by
  unfold lakes canals faces number_of_islands
  sorry

end island_count_l1503_150330


namespace triangle_similarity_proof_l1503_150308

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

end triangle_similarity_proof_l1503_150308


namespace even_number_divisible_by_8_l1503_150358

theorem even_number_divisible_by_8 {n : ℤ} (h : ∃ k : ℤ, n = 2 * k) : 
  (n * (n^2 + 20)) % 8 = 0 ∧ 
  (n * (n^2 - 20)) % 8 = 0 ∧ 
  (n * (n^2 + 4)) % 8 = 0 ∧ 
  (n * (n^2 - 4)) % 8 = 0 :=
by
  sorry

end even_number_divisible_by_8_l1503_150358


namespace circle_diameter_in_feet_l1503_150309

/-- Given: The area of a circle is 25 * pi square inches.
    Prove: The diameter of the circle in feet is 5/6 feet. -/
theorem circle_diameter_in_feet (A : ℝ) (hA : A = 25 * Real.pi) :
  ∃ d : ℝ, d = (5 / 6) :=
by
  -- The proof goes here
  sorry

end circle_diameter_in_feet_l1503_150309


namespace actual_revenue_is_60_percent_of_projected_l1503_150360

variable (R : ℝ)

-- Condition: Projected revenue is 25% more than last year's revenue
def projected_revenue (R : ℝ) : ℝ := 1.25 * R

-- Condition: Actual revenue decreased by 25% compared to last year's revenue
def actual_revenue (R : ℝ) : ℝ := 0.75 * R

-- Theorem: Prove that the actual revenue is 60% of the projected revenue
theorem actual_revenue_is_60_percent_of_projected :
  (actual_revenue R) = 0.6 * (projected_revenue R) :=
  sorry

end actual_revenue_is_60_percent_of_projected_l1503_150360


namespace sum_squares_nonpositive_l1503_150320

theorem sum_squares_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ac ≤ 0 :=
by {
  sorry
}

end sum_squares_nonpositive_l1503_150320


namespace christine_needs_32_tbs_aquafaba_l1503_150391

-- Definitions for the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

def total_egg_whites : ℕ := egg_whites_per_cake * number_of_cakes
def total_tbs_aquafaba : ℕ := tablespoons_per_egg_white * total_egg_whites

-- Theorem statement
theorem christine_needs_32_tbs_aquafaba :
  total_tbs_aquafaba = 32 :=
by sorry

end christine_needs_32_tbs_aquafaba_l1503_150391


namespace diane_total_harvest_l1503_150383

def total_harvest (h1 i1 i2 : Nat) : Nat :=
  h1 + (h1 + i1) + ((h1 + i1) + i2)

theorem diane_total_harvest :
  total_harvest 2479 6085 7890 = 27497 := 
by 
  sorry

end diane_total_harvest_l1503_150383


namespace find_prime_triple_l1503_150353

def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_triple :
  ∃ (I M C : ℕ), is_prime I ∧ is_prime M ∧ is_prime C ∧ I ≤ M ∧ M ≤ C ∧ 
  I * M * C = I + M + C + 1007 ∧ (I = 2 ∧ M = 2 ∧ C = 337) :=
by
  sorry

end find_prime_triple_l1503_150353


namespace carla_drive_distance_l1503_150384

theorem carla_drive_distance
    (d1 d3 : ℕ) (gpm : ℕ) (gas_price total_cost : ℕ) 
    (x : ℕ)
    (hx : 2 * gas_price = 1)
    (gallon_cost : ℕ := total_cost / gas_price)
    (total_distance   : ℕ := gallon_cost * gpm)
    (total_errand_distance : ℕ := d1 + x + d3 + 2 * x)
    (h_distance : total_distance = total_errand_distance) :
  x = 10 :=
by
  -- begin
  -- proof construction
  sorry

end carla_drive_distance_l1503_150384


namespace min_purchase_amount_is_18_l1503_150301

def burger_cost := 2 * 3.20
def fries_cost := 2 * 1.90
def milkshake_cost := 2 * 2.40
def current_total := burger_cost + fries_cost + milkshake_cost
def additional_needed := 3.00
def min_purchase_amount_for_free_delivery := current_total + additional_needed

theorem min_purchase_amount_is_18 : min_purchase_amount_for_free_delivery = 18 := by
  sorry

end min_purchase_amount_is_18_l1503_150301


namespace irrational_sqrt_2023_l1503_150312

theorem irrational_sqrt_2023 (A B C D : ℝ) :
  A = -2023 → B = Real.sqrt 2023 → C = 0 → D = 1 / 2023 →
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ B = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ A = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ C = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ D = p / q) := 
by
  intro hA hB hC hD
  sorry

end irrational_sqrt_2023_l1503_150312


namespace negate_exists_l1503_150399

theorem negate_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x < Real.sin x ∨ x > Real.tan x) ↔ (∀ x : ℝ, x ≥ Real.sin x ∨ x ≤ Real.tan x) :=
by
  sorry

end negate_exists_l1503_150399


namespace marbles_problem_l1503_150317

def marbles_total : ℕ := 30
def prob_black_black : ℚ := 14 / 25
def prob_white_white : ℚ := 16 / 225

theorem marbles_problem (total_marbles : ℕ) (prob_bb prob_ww : ℚ) 
  (h_total : total_marbles = 30)
  (h_prob_bb : prob_bb = 14 / 25)
  (h_prob_ww : prob_ww = 16 / 225) :
  let m := 16
  let n := 225
  m.gcd n = 1 ∧ m + n = 241 :=
by {
  sorry
}

end marbles_problem_l1503_150317


namespace find_total_cards_l1503_150351

def numCardsInStack (n : ℕ) : Prop :=
  let cards : List ℕ := List.range' 1 (2 * n + 1)
  let pileA := cards.take n
  let pileB := cards.drop n
  let restack := List.zipWith (fun x y => [y, x]) pileA pileB |> List.join
  (restack.take 13).getLastD 0 = 13 ∧ 2 * n = 26

theorem find_total_cards : ∃ (n : ℕ), numCardsInStack n :=
sorry

end find_total_cards_l1503_150351


namespace ratio_of_unit_prices_l1503_150364

def volume_y (v : ℝ) : ℝ := v
def price_y (p : ℝ) : ℝ := p
def volume_x (v : ℝ) : ℝ := 1.3 * v
def price_x (p : ℝ) : ℝ := 0.8 * p

theorem ratio_of_unit_prices (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (0.8 * p / (1.3 * v)) / (p / v) = 8 / 13 :=
by 
  sorry

end ratio_of_unit_prices_l1503_150364


namespace length_OR_coordinates_Q_area_OPQR_8_p_value_l1503_150302

noncomputable def point_R : (ℝ × ℝ) := (0, 4)

noncomputable def OR_distance : ℝ := 0 - 4 -- the vertical distance from O to R

theorem length_OR : OR_distance = 4 := sorry

noncomputable def point_Q (p : ℝ) : (ℝ × ℝ) := (p, 2 * p + 4)

theorem coordinates_Q (p : ℝ) : point_Q p = (p, 2 * p + 4) := sorry

noncomputable def area_OPQR (p : ℝ) : ℝ := 
  let OR : ℝ := 4
  let PQ : ℝ := 2 * p + 4
  let OP : ℝ := p
  1 / 2 * (OR + PQ) * OP

theorem area_OPQR_8 : area_OPQR 8 = 96 := sorry

theorem p_value (h : area_OPQR p = 77) : p = 7 := sorry

end length_OR_coordinates_Q_area_OPQR_8_p_value_l1503_150302


namespace prob1_prob2_prob3_l1503_150345

-- Problem 1
theorem prob1 (a b c : ℝ) : ((-8 * a^4 * b^5 * c / (4 * a * b^5)) * (3 * a^3 * b^2)) = -6 * a^6 * b^2 :=
by
  sorry

-- Problem 2
theorem prob2 (a : ℝ) : (2 * a + 1)^2 - (2 * a + 1) * (2 * a - 1) = 4 * a + 2 :=
by
  sorry

-- Problem 3
theorem prob3 (x y : ℝ) : (x - y - 2) * (x - y + 2) - (x + 2 * y) * (x - 3 * y) = 7 * y^2 - x * y - 4 :=
by
  sorry

end prob1_prob2_prob3_l1503_150345


namespace sin_double_angle_l1503_150356

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l1503_150356


namespace machine_work_time_today_l1503_150314

theorem machine_work_time_today :
  let shirts_today := 40
  let pants_today := 50
  let shirt_rate := 5
  let pant_rate := 3
  let time_for_shirts := shirts_today / shirt_rate
  let time_for_pants := pants_today / pant_rate
  time_for_shirts + time_for_pants = 24.67 :=
by
  sorry

end machine_work_time_today_l1503_150314


namespace binary_to_decimal_conversion_l1503_150311

theorem binary_to_decimal_conversion : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) := by
  sorry

end binary_to_decimal_conversion_l1503_150311


namespace possible_values_of_n_l1503_150385

theorem possible_values_of_n (E M n : ℕ) (h1 : M + 3 = n * (E - 3)) (h2 : E + n = 3 * (M - n)) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end possible_values_of_n_l1503_150385


namespace total_holes_dug_l1503_150333

theorem total_holes_dug :
  (Pearl_digging_rate * 21 + Miguel_digging_rate * 21) = 26 :=
by
  -- Definitions based on conditions
  let Pearl_digging_rate := 4 / 7
  let Miguel_digging_rate := 2 / 3
  -- Sorry placeholder for the proof
  sorry

end total_holes_dug_l1503_150333


namespace first_candidate_more_gain_l1503_150331

-- Definitions for the salaries, revenues, training costs, and bonuses
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_cost_per_month : ℕ := 1200
def training_months : ℕ := 3

def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2_percentage : ℕ := 1

-- Calculate net gains
def net_gain1 : ℕ :=
  revenue1 - salary1 - (training_cost_per_month * training_months)

def net_gain2 : ℕ :=
  revenue2 - salary2 - (salary2 * bonus2_percentage / 100)

def difference_in_gain : ℕ :=
  net_gain1 - net_gain2

-- Theorem statement
theorem first_candidate_more_gain :
  difference_in_gain = 850 :=
by
  -- Proof goes here
  sorry

end first_candidate_more_gain_l1503_150331


namespace power_equation_l1503_150393

theorem power_equation (x a : ℝ) (h : x^(-a) = 3) : x^(2 * a) = 1 / 9 :=
sorry

end power_equation_l1503_150393


namespace total_students_l1503_150335

theorem total_students (x : ℕ) (h1 : 3 * x + 8 = 3 * x + 5) (h2 : 5 * (x - 1) + 3 > 3 * x + 8) : x = 6 :=
sorry

end total_students_l1503_150335


namespace quadratic_has_negative_root_condition_l1503_150324

theorem quadratic_has_negative_root_condition (a : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, ax^2 + 2*x + 1 = 0 ∧ x < 0) ↔ (a < 0 ∨ (0 < a ∧ a ≤ 1)) :=
by
  sorry

end quadratic_has_negative_root_condition_l1503_150324


namespace find_m_for_q_find_m_for_pq_l1503_150382

variable (m : ℝ)

-- Statement q: The equation represents a hyperbola if and only if m > 3
def q (m : ℝ) : Prop := m > 3

-- Statement p: The inequality holds if and only if m >= 1
def p (m : ℝ) : Prop := m ≥ 1

-- 1. If statement q is true, find the range of values for m.
theorem find_m_for_q (h : q m) : m > 3 := by
  exact h

-- 2. If (p ∨ q) is true and (p ∧ q) is false, find the range of values for m.
theorem find_m_for_pq (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end find_m_for_q_find_m_for_pq_l1503_150382


namespace claire_photos_l1503_150321

variable (C : ℕ) -- Claire's photos
variable (L : ℕ) -- Lisa's photos
variable (R : ℕ) -- Robert's photos

-- Conditions
axiom Lisa_photos : L = 3 * C
axiom Robert_photos : R = C + 16
axiom Lisa_Robert_same : L = R

-- Proof Goal
theorem claire_photos : C = 8 :=
by
  -- Sorry skips the proof and allows the theorem to compile
  sorry

end claire_photos_l1503_150321


namespace initial_boys_count_l1503_150342

variable (q : ℕ) -- total number of children initially in the group
variable (b : ℕ) -- number of boys initially in the group

-- Initial condition: 60% of the group are boys initially
def initial_boys (q : ℕ) : ℕ := 6 * q / 10

-- Change after event: three boys leave, three girls join
def boys_after_event (b : ℕ) : ℕ := b - 3

-- After the event, the number of boys is 50% of the total group
def boys_percentage_after_event (b : ℕ) (q : ℕ) : Prop :=
  boys_after_event b = 5 * q / 10

theorem initial_boys_count :
  ∃ b q : ℕ, b = initial_boys q ∧ boys_percentage_after_event b q → b = 18 := 
sorry

end initial_boys_count_l1503_150342


namespace new_mix_concentration_l1503_150366

theorem new_mix_concentration 
  (capacity1 capacity2 capacity_mix : ℝ)
  (alc_percent1 alc_percent2 : ℝ)
  (amount1 amount2 : capacity1 = 3 ∧ capacity2 = 5 ∧ capacity_mix = 10)
  (percent1: alc_percent1 = 0.25)
  (percent2: alc_percent2 = 0.40)
  (total_volume : ℝ)
  (eight_liters : total_volume = 8) :
  (alc_percent1 * capacity1 + alc_percent2 * capacity2) / total_volume * 100 = 34.375 :=
by
  sorry

end new_mix_concentration_l1503_150366


namespace king_and_queen_ages_l1503_150341

variable (K Q : ℕ)

theorem king_and_queen_ages (h1 : K = 2 * (Q - (K - Q)))
                            (h2 : K + (K + (K - Q)) = 63) :
                            K = 28 ∧ Q = 21 := by
  sorry

end king_and_queen_ages_l1503_150341


namespace range_of_m_l1503_150395

open Set

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 7 }
def B (m : ℝ) : Set ℝ := { x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1) }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → m ≤ 4 :=
by
  intro h
  sorry

end range_of_m_l1503_150395


namespace no_common_root_l1503_150323

theorem no_common_root (a b c d : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 ∧ x^2 + a * x + d = 0 :=
by
  sorry

end no_common_root_l1503_150323


namespace arithmetic_expression_evaluation_l1503_150325

theorem arithmetic_expression_evaluation :
  3^2 + 4 * 2 - 6 / 3 + 7 = 22 :=
by 
  -- Use tactics to break down the arithmetic expression evaluation (steps are abstracted)
  sorry

end arithmetic_expression_evaluation_l1503_150325


namespace determinant_of_tan_matrix_l1503_150369

theorem determinant_of_tan_matrix
  (A B C : ℝ)
  (h₁ : A = π / 4)
  (h₂ : A + B + C = π)
  : (Matrix.det ![
      ![Real.tan A, 1, 1],
      ![1, Real.tan B, 1],
      ![1, 1, Real.tan C]
    ]) = 2 :=
  sorry

end determinant_of_tan_matrix_l1503_150369


namespace smallest_sum_l1503_150396

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  (∀ A B C D : ℕ, 
    5 * A = 25 * A - 27 * B ∧
    5 * B = 15 * A - 16 * B ∧
    3 * C = 25 * C - 27 * D ∧
    3 * D = 15 * C - 16 * D) ∧
  a = 4 ∧ b = 3 ∧ c = 27 ∧ d = 22 ∧ a + b + c + d = 56

theorem smallest_sum : problem_statement :=
  sorry

end smallest_sum_l1503_150396


namespace calc_7_op_4_minus_4_op_7_l1503_150373

def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

theorem calc_7_op_4_minus_4_op_7 : (op 7 4) - (op 4 7) = -12 := by
  sorry

end calc_7_op_4_minus_4_op_7_l1503_150373


namespace find_a0_find_a2_find_sum_a1_a2_a3_a4_l1503_150390

lemma problem_conditions (x : ℝ) : 
  (x - 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 :=
sorry

theorem find_a0 :
  a_0 = 16 :=
sorry

theorem find_a2 :
  a_2 = 24 :=
sorry

theorem find_sum_a1_a2_a3_a4 :
  a_1 + a_2 + a_3 + a_4 = -15 :=
sorry

end find_a0_find_a2_find_sum_a1_a2_a3_a4_l1503_150390


namespace odd_function_inequality_l1503_150329

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_inequality
  (f : ℝ → ℝ) (h1 : is_odd_function f)
  (a b : ℝ) (h2 : f a > f b) :
  f (-a) < f (-b) :=
by
  sorry

end odd_function_inequality_l1503_150329


namespace arithmetic_expression_equality_l1503_150339

theorem arithmetic_expression_equality : 18 * 36 - 27 * 18 = 162 := by
  sorry

end arithmetic_expression_equality_l1503_150339


namespace age_of_15th_student_l1503_150315

theorem age_of_15th_student (avg15: ℕ) (avg5: ℕ) (avg9: ℕ) (x: ℕ)
  (h1: avg15 = 15) (h2: avg5 = 14) (h3: avg9 = 16)
  (h4: 15 * avg15 = x + 5 * avg5 + 9 * avg9) : x = 11 :=
by
  -- Proof will be added here
  sorry

end age_of_15th_student_l1503_150315


namespace sum_of_values_l1503_150347

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 5 * x - 3 else x^2 - 4 * x + 3

theorem sum_of_values (s : Finset ℝ) : 
  (∀ x ∈ s, f x = 2) → s.sum id = 4 :=
by 
  sorry

end ProofProblem

end sum_of_values_l1503_150347


namespace cos_alpha_value_l1503_150316

theorem cos_alpha_value (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 3 / 5) :
  Real.cos α = 4 / 5 :=
by
  sorry

end cos_alpha_value_l1503_150316


namespace triangle_BC_length_l1503_150326

theorem triangle_BC_length (A : ℝ) (AC : ℝ) (S : ℝ) (BC : ℝ)
  (h1 : A = 60) (h2 : AC = 16) (h3 : S = 220 * Real.sqrt 3) :
  BC = 49 :=
by
  sorry

end triangle_BC_length_l1503_150326


namespace totalPawnsLeft_l1503_150398

def sophiaInitialPawns := 8
def chloeInitialPawns := 8
def sophiaLostPawns := 5
def chloeLostPawns := 1

theorem totalPawnsLeft : (sophiaInitialPawns - sophiaLostPawns) + (chloeInitialPawns - chloeLostPawns) = 10 := by
  sorry

end totalPawnsLeft_l1503_150398


namespace sum_b_div_5_pow_eq_l1503_150336

namespace SequenceSumProblem

-- Define the sequence b_n
def b : ℕ → ℝ
| 0       => 2
| 1       => 3
| (n + 2) => b (n + 1) + b n

-- The infinite series sum we need to prove
noncomputable def sum_b_div_5_pow (Y : ℝ) : Prop :=
  Y = ∑' n : ℕ, (b n) / (5 ^ (n + 1))

-- The statement of the problem
theorem sum_b_div_5_pow_eq : sum_b_div_5_pow (2 / 25) :=
sorry

end SequenceSumProblem

end sum_b_div_5_pow_eq_l1503_150336


namespace area_spot_can_reach_l1503_150354

noncomputable def area_reachable_by_spot (s : ℝ) (r : ℝ) : ℝ := 
  if s = 1 ∧ r = 3 then 6.5 * Real.pi else 0

theorem area_spot_can_reach : area_reachable_by_spot 1 3 = 6.5 * Real.pi :=
by
  -- The theorem proof should go here.
  sorry

end area_spot_can_reach_l1503_150354


namespace correct_propositions_l1503_150352

-- Definitions of relations between lines and planes
variable {Line : Type}
variable {Plane : Type}

-- Definition of relationships
variable (parallel_lines : Line → Line → Prop)
variable (parallel_plane_with_plane : Plane → Plane → Prop)
variable (parallel_line_with_plane : Line → Plane → Prop)
variable (perpendicular_plane_with_plane : Plane → Plane → Prop)
variable (perpendicular_line_with_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (distinct_lines : Line → Line → Prop)
variable (distinct_planes : Plane → Plane → Prop)

-- The main theorem we are proving with the given conditions
theorem correct_propositions (m n : Line) (α β γ : Plane)
  (hmn : distinct_lines m n) (hαβ : distinct_planes α β) (hαγ : distinct_planes α γ)
  (hβγ : distinct_planes β γ) :
  -- Statement 1
  (parallel_plane_with_plane α β → parallel_plane_with_plane α γ → parallel_plane_with_plane β γ) ∧
  -- Statement 3
  (perpendicular_line_with_plane m α → parallel_line_with_plane m β → perpendicular_plane_with_plane α β) :=
by
  sorry

end correct_propositions_l1503_150352


namespace inequality_solution_addition_eq_seven_l1503_150377

theorem inequality_solution_addition_eq_seven (b c : ℝ) :
  (∀ x : ℝ, -5 < 2 * x - 3 ∧ 2 * x - 3 < 5 → -1 < x ∧ x < 4) →
  (∀ x : ℝ, -x^2 + b * x + c = 0 ↔ (x = -1 ∨ x = 4)) →
  b + c = 7 :=
by
  intro h1 h2
  sorry

end inequality_solution_addition_eq_seven_l1503_150377


namespace fraction_min_sum_l1503_150375

theorem fraction_min_sum (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) 
  (h_ineq : 45 * b < 110 * a ∧ 110 * a < 50 * b) :
  a = 3 ∧ b = 7 :=
sorry

end fraction_min_sum_l1503_150375


namespace binary_to_decimal_110011_l1503_150306

theorem binary_to_decimal_110011 :
  1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 51 :=
by
  sorry

end binary_to_decimal_110011_l1503_150306


namespace photo_counts_correct_l1503_150387

open Real

-- Definitions based on the conditions from step a)
def animal_photos : ℕ := 20
def flower_photos : ℕ := 30 -- 1.5 * 20
def total_animal_flower_photos : ℕ := animal_photos + flower_photos
def scenery_abstract_photos_combined : ℕ := (4 / 10) * total_animal_flower_photos -- 40% of total_animal_flower_photos

def x : ℕ := scenery_abstract_photos_combined / 5
def scenery_photos : ℕ := 3 * x
def abstract_photos : ℕ := 2 * x
def total_photos : ℕ := animal_photos + flower_photos + scenery_photos + abstract_photos

-- The statement to prove
theorem photo_counts_correct :
  animal_photos = 20 ∧
  flower_photos = 30 ∧
  total_animal_flower_photos = 50 ∧
  scenery_abstract_photos_combined = 20 ∧
  scenery_photos = 12 ∧
  abstract_photos = 8 ∧
  total_photos = 70 :=
by
  sorry

end photo_counts_correct_l1503_150387


namespace simplify_expression_l1503_150340

open Real

theorem simplify_expression :
    (3 * (sqrt 5 + sqrt 7) / (4 * sqrt (3 + sqrt 5))) = sqrt (414 - 98 * sqrt 35) / 8 :=
by
  sorry

end simplify_expression_l1503_150340


namespace remainder_when_divided_by_10_l1503_150313

theorem remainder_when_divided_by_10 :
  (2457 * 6291 * 9503) % 10 = 1 :=
by
  sorry

end remainder_when_divided_by_10_l1503_150313


namespace geometric_series_sum_l1503_150318

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end geometric_series_sum_l1503_150318


namespace room_dimension_l1503_150379

theorem room_dimension
  (x : ℕ)
  (cost_per_sqft : ℕ := 4)
  (dimension_1 : ℕ := 15)
  (dimension_2 : ℕ := 12)
  (door_width : ℕ := 6)
  (door_height : ℕ := 3)
  (num_windows : ℕ := 3)
  (window_width : ℕ := 4)
  (window_height : ℕ := 3)
  (total_cost : ℕ := 3624) :
  (2 * (x * dimension_1) + 2 * (x * dimension_2) - (door_width * door_height + num_windows * (window_width * window_height))) * cost_per_sqft = total_cost →
  x = 18 :=
by
  sorry

end room_dimension_l1503_150379


namespace simplify_expression_l1503_150368

noncomputable def i : ℂ := Complex.I

theorem simplify_expression : 7*(4 - 2*i) + 4*i*(3 - 2*i) = 36 - 2*i :=
by
  sorry

end simplify_expression_l1503_150368


namespace total_weight_correct_l1503_150305

def weight_male_clothes : ℝ := 2.6
def weight_female_clothes : ℝ := 5.98
def total_weight_clothes : ℝ := weight_male_clothes + weight_female_clothes

theorem total_weight_correct : total_weight_clothes = 8.58 := by
  sorry

end total_weight_correct_l1503_150305


namespace factorial_not_div_by_two_pow_l1503_150372

theorem factorial_not_div_by_two_pow (n : ℕ) : ¬ (2^n ∣ n!) :=
sorry

end factorial_not_div_by_two_pow_l1503_150372


namespace tangent_intersection_x_l1503_150371

theorem tangent_intersection_x :
  ∃ x : ℝ, 
    0 < x ∧ (∃ r1 r2 : ℝ, 
     (r1 = 3) ∧ 
     (r2 = 8) ∧ 
     (0, 0) = (0, 0) ∧ 
     (18, 0) = (18, 0) ∧
     (∀ t : ℝ, t > 0 → t = x / (18 - x) → t = r1 / r2) ∧ 
      x = 54 / 11) := 
sorry

end tangent_intersection_x_l1503_150371


namespace find_angle_and_sum_of_sides_l1503_150327

noncomputable def triangle_conditions 
    (a b c : ℝ) (C : ℝ)
    (area : ℝ) : Prop :=
  a^2 + b^2 - c^2 = a * b ∧
  c = Real.sqrt 7 ∧
  area = (3 * Real.sqrt 3) / 2 

theorem find_angle_and_sum_of_sides
    (a b c C : ℝ)
    (area : ℝ)
    (h : triangle_conditions a b c C area) :
    C = Real.pi / 3 ∧ a + b = 5 := by
  sorry

end find_angle_and_sum_of_sides_l1503_150327


namespace polynomial_sum_l1503_150310

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def j (x : ℝ) : ℝ := x^2 - x - 3

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = -3 * x^2 + 11 * x - 15 := by
  sorry

end polynomial_sum_l1503_150310


namespace fred_seashells_l1503_150334

-- Define the initial number of seashells Fred found.
def initial_seashells : ℕ := 47

-- Define the number of seashells Fred gave to Jessica.
def seashells_given : ℕ := 25

-- Prove that Fred now has 22 seashells.
theorem fred_seashells : initial_seashells - seashells_given = 22 :=
by
  sorry

end fred_seashells_l1503_150334


namespace total_cupcakes_l1503_150388

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (total_cupcakes : ℕ) 
  (h1 : children = 8) (h2 : cupcakes_per_child = 12) : total_cupcakes = 96 := 
by
  sorry

end total_cupcakes_l1503_150388


namespace incorrect_statement_B_l1503_150348

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x - 1)^3 - a * x - b + 2

-- Condition for statement B
axiom eqn_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1

-- The theorem to prove:
theorem incorrect_statement_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1 := by
  exact eqn_B a b

end incorrect_statement_B_l1503_150348


namespace find_other_root_l1503_150355

theorem find_other_root (x : ℚ) (h: 63 * x^2 - 100 * x + 45 = 0) (hx: x = 5 / 7) : x = 1 ∨ x = 5 / 7 :=
by 
  -- Insert the proof steps here if needed.
  sorry

end find_other_root_l1503_150355


namespace inequality_ab_bc_ca_max_l1503_150337

theorem inequality_ab_bc_ca_max (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|))
  ≤ 1 + (1 / 3) * (a + b + c)^2 := sorry

end inequality_ab_bc_ca_max_l1503_150337


namespace perpendicular_lines_m_value_l1503_150322

theorem perpendicular_lines_m_value (m : ℝ) (l1_perp_l2 : (m ≠ 0) → (m * (-1 / m^2)) = -1) : m = 0 ∨ m = 1 :=
sorry

end perpendicular_lines_m_value_l1503_150322


namespace find_d_l1503_150338

theorem find_d 
    (a b c d : ℝ) 
    (h_a_pos : 0 < a)
    (h_b_pos : 0 < b)
    (h_c_pos : 0 < c)
    (h_d_pos : 0 < d)
    (max_val : d + a = 7)
    (min_val : d - a = 1) :
    d = 4 :=
by
  sorry

end find_d_l1503_150338


namespace smallest_x_exists_l1503_150361

theorem smallest_x_exists (x k m : ℤ) 
    (h1 : x + 3 = 7 * k) 
    (h2 : x - 5 = 8 * m) 
    (h3 : ∀ n : ℤ, ((n + 3) % 7 = 0) ∧ ((n - 5) % 8 = 0) → x ≤ n) : 
    x = 53 := by
  sorry

end smallest_x_exists_l1503_150361


namespace B_time_to_complete_work_l1503_150374

variable {W : ℝ} {R_b : ℝ} {T_b : ℝ}

theorem B_time_to_complete_work (h1 : 3 * R_b * (T_b - 10) = R_b * T_b) : T_b = 15 :=
by
  sorry

end B_time_to_complete_work_l1503_150374


namespace train_people_count_l1503_150303

theorem train_people_count :
  let initial := 332
  let first_station_on := 119
  let first_station_off := 113
  let second_station_off := 95
  let second_station_on := 86
  initial + first_station_on - first_station_off - second_station_off + second_station_on = 329 := 
by
  sorry

end train_people_count_l1503_150303


namespace jack_needs_more_money_l1503_150343

/--
Jack is a soccer player. He needs to buy two pairs of socks, a pair of soccer shoes, a soccer ball, and a sports bag.
Each pair of socks costs $12.75, the shoes cost $145, the soccer ball costs $38, and the sports bag costs $47.
Jack has a 5% discount coupon for the shoes and a 10% discount coupon for the sports bag.
He currently has $25. How much more money does Jack need to buy all the items?
-/
theorem jack_needs_more_money :
  let socks_cost : ℝ := 12.75
  let shoes_cost : ℝ := 145
  let ball_cost : ℝ := 38
  let bag_cost : ℝ := 47
  let shoes_discount : ℝ := 0.05
  let bag_discount : ℝ := 0.10
  let money_jack_has : ℝ := 25
  let total_cost := 2 * socks_cost + (shoes_cost - shoes_cost * shoes_discount) + ball_cost + (bag_cost - bag_cost * bag_discount)
  total_cost - money_jack_has = 218.55 :=
by
  sorry

end jack_needs_more_money_l1503_150343


namespace abs_diff_condition_l1503_150397

theorem abs_diff_condition {a b : ℝ} (h1 : |a| = 1) (h2 : |b - 1| = 2) (h3 : a > b) : a - b = 2 := 
sorry

end abs_diff_condition_l1503_150397


namespace max_min_value_l1503_150389

noncomputable def f (A B x a b : ℝ) : ℝ :=
  A * Real.sqrt (x - a) + B * Real.sqrt (b - x)

theorem max_min_value (A B a b : ℝ) (hA : A > 0) (hB : B > 0) (ha_lt_b : a < b) :
  (∀ x, a ≤ x ∧ x ≤ b → f A B x a b ≤ Real.sqrt ((A^2 + B^2) * (b - a))) ∧
  min (f A B a a b) (f A B b a b) ≤ f A B x a b :=
  sorry

end max_min_value_l1503_150389


namespace find_symmetric_point_l1503_150394

def slope_angle (l : ℝ → ℝ → Prop) (θ : ℝ) := ∃ m, m = Real.tan θ ∧ ∀ x y, l x y ↔ y = m * (x - 1) + 1
def passes_through (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) := l P.fst P.snd
def symmetric_point (A A' : ℝ × ℝ) (l : ℝ → ℝ → Prop) := 
  (A'.snd - A.snd = A'.fst - A.fst) ∧ 
  ((A'.fst + A.fst) / 2 + (A'.snd + A.snd) / 2 - 2 = 0)

theorem find_symmetric_point :
  ∃ l : ℝ → ℝ → Prop, 
    slope_angle l (135 : ℝ) ∧ 
    passes_through l (1, 1) ∧ 
    (∀ x y, l x y ↔ x + y = 2) ∧ 
    symmetric_point (3, 4) (-2, -1) l :=
by sorry

end find_symmetric_point_l1503_150394


namespace totalCostOfFencing_l1503_150381

def numberOfSides : ℕ := 4
def costPerSide : ℕ := 79

theorem totalCostOfFencing (n : ℕ) (c : ℕ) (hn : n = numberOfSides) (hc : c = costPerSide) : n * c = 316 :=
by 
  rw [hn, hc]
  exact rfl

end totalCostOfFencing_l1503_150381


namespace correct_sum_of_integers_l1503_150350

theorem correct_sum_of_integers :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a - b = 3 ∧ a * b = 63 ∧ a + b = 17 :=
by 
  sorry

end correct_sum_of_integers_l1503_150350


namespace eval_expression_l1503_150304

def a : ℕ := 4 * 5 * 6
def b : ℚ := 1/4 + 1/5 - 1/10

theorem eval_expression : a * b = 42 := by
  sorry

end eval_expression_l1503_150304


namespace number_of_pens_each_student_gets_l1503_150346

theorem number_of_pens_each_student_gets 
    (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ)
    (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) :
  (total_pens / Nat.gcd total_pens total_pencils) = 11 :=
by
  sorry

end number_of_pens_each_student_gets_l1503_150346


namespace unique_function_l1503_150378

-- Define the function in the Lean environment
def f (n : ℕ) : ℕ := n

-- State the theorem with the given conditions and expected answer
theorem unique_function (f : ℕ → ℕ) : 
  (∀ x y : ℕ, 0 < x → 0 < y → f x + y * f (f x) < x * (1 + f y) + 2021) → (∀ x : ℕ, f x = x) :=
by
  intros h x
  -- Placeholder for the proof
  sorry

end unique_function_l1503_150378


namespace gcd_78_182_l1503_150359

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end gcd_78_182_l1503_150359


namespace dot_product_of_ab_ac_l1503_150362

def vec_dot (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem dot_product_of_ab_ac :
  vec_dot (1, -2) (2, -2) = 6 := by
  sorry

end dot_product_of_ab_ac_l1503_150362


namespace researcher_can_cross_desert_l1503_150332

structure Condition :=
  (distance_to_oasis : ℕ)  -- total distance to be covered
  (travel_per_day : ℕ)     -- distance covered per day
  (carry_capacity : ℕ)     -- maximum days of supplies they can carry
  (ensure_return : Bool)   -- flag to ensure porters can return
  (cannot_store_food : Bool) -- flag indicating no food storage in desert

def condition_instance : Condition :=
{ distance_to_oasis := 380,
  travel_per_day := 60,
  carry_capacity := 4,
  ensure_return := true,
  cannot_store_food := true }

theorem researcher_can_cross_desert (cond : Condition) : cond.distance_to_oasis = 380 
  ∧ cond.travel_per_day = 60 
  ∧ cond.carry_capacity = 4 
  ∧ cond.ensure_return = true 
  ∧ cond.cannot_store_food = true 
  → true := 
by 
  sorry

end researcher_can_cross_desert_l1503_150332


namespace triangle_side_lengths_values_l1503_150367

theorem triangle_side_lengths_values :
  ∃ (m_values : Finset ℕ), m_values = {m ∈ Finset.range 750 | m ≥ 4} ∧ m_values.card = 746 :=
by
  sorry

end triangle_side_lengths_values_l1503_150367


namespace range_of_a_for_distinct_real_roots_l1503_150363

theorem range_of_a_for_distinct_real_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔ (a < 2 ∧ a ≠ 1) :=
by
  sorry

end range_of_a_for_distinct_real_roots_l1503_150363


namespace f_neg_val_is_minus_10_l1503_150357
-- Import the necessary Lean library

-- Define the function f with the given conditions
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 + 3

-- Define the specific values
def x_val : ℝ := 2023
def x_neg_val : ℝ := -2023
def f_pos_val : ℝ := 16

-- Theorem to prove
theorem f_neg_val_is_minus_10 (a b : ℝ)
  (h : f a b x_val = f_pos_val) : 
  f a b x_neg_val = -10 :=
by
  -- Sorry placeholder for proof
  sorry

end f_neg_val_is_minus_10_l1503_150357


namespace highest_number_paper_l1503_150300

theorem highest_number_paper (n : ℕ) (h : 1 / (n : ℝ) = 0.01020408163265306) : n = 98 :=
sorry

end highest_number_paper_l1503_150300


namespace parallelogram_ratio_l1503_150319

-- Definitions based on given conditions
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_ratio (A : ℝ) (B : ℝ) (h : ℝ) (H1 : A = 242) (H2 : B = 11) (H3 : A = parallelogram_area B h) :
  h / B = 2 :=
by
  -- Proof goes here
  sorry

end parallelogram_ratio_l1503_150319


namespace quad_eq_sum_ab_l1503_150370

theorem quad_eq_sum_ab {a b : ℝ} (h1 : a < 0)
  (h2 : ∀ x : ℝ, (x = -1 / 2 ∨ x = 1 / 3) ↔ ax^2 + bx + 2 = 0) :
  a + b = -14 :=
by
  sorry

end quad_eq_sum_ab_l1503_150370


namespace quadratic_solution_sum_l1503_150344

theorem quadratic_solution_sum
  (x : ℚ)
  (m n p : ℕ)
  (h_eq : (5 * x - 11) * x = -6)
  (h_form : ∃ m n p, x = (m + Real.sqrt n) / p ∧ x = (m - Real.sqrt n) / p)
  (h_gcd : Nat.gcd (Nat.gcd m n) p = 1) :
  m + n + p = 22 := 
sorry

end quadratic_solution_sum_l1503_150344


namespace sum_of_solutions_eq_seven_l1503_150380

theorem sum_of_solutions_eq_seven : 
  ∃ x : ℝ, x + 49/x = 14 ∧ (∀ y : ℝ, y + 49 / y = 14 → y = x) → x = 7 :=
by {
  sorry
}

end sum_of_solutions_eq_seven_l1503_150380


namespace find_angle_A_l1503_150349

theorem find_angle_A (A B : ℝ) (a b : ℝ) (h1 : b = 2 * a * Real.sin B) (h2 : a ≠ 0) :
  A = 30 ∨ A = 150 :=
by
  sorry

end find_angle_A_l1503_150349


namespace pastries_average_per_day_l1503_150328

theorem pastries_average_per_day :
  let monday_sales := 2
  let tuesday_sales := monday_sales + 1
  let wednesday_sales := tuesday_sales + 1
  let thursday_sales := wednesday_sales + 1
  let friday_sales := thursday_sales + 1
  let saturday_sales := friday_sales + 1
  let sunday_sales := saturday_sales + 1
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
  let days := 7
  total_sales / days = 5 := by
  sorry

end pastries_average_per_day_l1503_150328


namespace attendance_second_day_l1503_150307

theorem attendance_second_day (total_attendance first_day_attendance second_day_attendance third_day_attendance : ℕ) 
  (h_total : total_attendance = 2700)
  (h_second_day : second_day_attendance = first_day_attendance / 2)
  (h_third_day : third_day_attendance = 3 * first_day_attendance) :
  second_day_attendance = 300 :=
by
  sorry

end attendance_second_day_l1503_150307


namespace arctan_sum_zero_l1503_150386
open Real

variable (a b c : ℝ)
variable (h : a^2 + b^2 = c^2)

theorem arctan_sum_zero (h : a^2 + b^2 = c^2) :
  arctan (a / (b + c)) + arctan (b / (a + c)) + arctan (c / (a + b)) = 0 := 
sorry

end arctan_sum_zero_l1503_150386
