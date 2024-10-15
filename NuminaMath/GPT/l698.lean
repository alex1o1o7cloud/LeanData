import Mathlib

namespace NUMINAMATH_GPT_toy_cost_l698_69869

-- Conditions
def initial_amount : ℕ := 3
def allowance : ℕ := 7
def total_amount : ℕ := initial_amount + allowance
def number_of_toys : ℕ := 2

-- Question and Proof
theorem toy_cost :
  total_amount / number_of_toys = 5 :=
by
  sorry

end NUMINAMATH_GPT_toy_cost_l698_69869


namespace NUMINAMATH_GPT_replaced_person_age_is_40_l698_69808

def average_age_decrease_replacement (T age_of_replaced: ℕ) : Prop :=
  let original_average := T / 10
  let new_total_age := T - age_of_replaced + 10
  let new_average := new_total_age / 10
  original_average - 3 = new_average

theorem replaced_person_age_is_40 (T : ℕ) (h : average_age_decrease_replacement T 40) : Prop :=
  ∀ age_of_replaced, age_of_replaced = 40 → average_age_decrease_replacement T age_of_replaced

-- To actually formalize the proof, you can use the following structure:
-- proof by calculation omitted
lemma replaced_person_age_is_40_proof (T : ℕ) (h : average_age_decrease_replacement T 40) : 
  replaced_person_age_is_40 T h :=
by
  sorry

end NUMINAMATH_GPT_replaced_person_age_is_40_l698_69808


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_eight_l698_69856

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sum (a₁ a₈ : α) (n : α) : α := (n * (a₁ + a₈)) / 2

theorem arithmetic_sequence_sum_eight {a₄ a₅ : α} (h₄₅ : a₄ + a₅ = 10) :
  let a₁ := a₄ - 3 * ((a₅ - a₄) / 1) -- a₁ in terms of a₄ and a₅
  let a₈ := a₄ + 4 * ((a₅ - a₄) / 1) -- a₈ in terms of a₄ and a₅
  arithmetic_sum a₁ a₈ 8 = 40 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_eight_l698_69856


namespace NUMINAMATH_GPT_find_current_listens_l698_69851

theorem find_current_listens (x : ℕ) (h : 15 * x = 900000) : x = 60000 :=
by
  sorry

end NUMINAMATH_GPT_find_current_listens_l698_69851


namespace NUMINAMATH_GPT_min_value_of_quadratic_l698_69824

-- Define the given quadratic function
def quadratic (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

-- Define the assertion that the minimum value of the quadratic function is 29/3
theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 29/3 ∧ ∀ y : ℝ, quadratic y ≥ 29/3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l698_69824


namespace NUMINAMATH_GPT_find_x_l698_69867

variable (x : ℝ)
def vector_a : ℝ × ℝ := (x, 2)
def vector_b : ℝ × ℝ := (x - 1, 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (h1 : dot_product (vector_a x + vector_b x) (vector_a x - vector_b x) = 0) : x = -1 := by 
  sorry

end NUMINAMATH_GPT_find_x_l698_69867


namespace NUMINAMATH_GPT_complement_intersection_l698_69870

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def set_B : Set ℕ := {2, 3, 6}

theorem complement_intersection :
  ((universal_set \ set_A) ∩ set_B) = {2, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l698_69870


namespace NUMINAMATH_GPT_compute_b_l698_69899

theorem compute_b (x y b : ℚ) (h1 : 5 * x - 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hy : y = 3) :
  b = 13 / 2 :=
sorry

end NUMINAMATH_GPT_compute_b_l698_69899


namespace NUMINAMATH_GPT_inverse_proportion_l698_69893

theorem inverse_proportion {x y : ℝ} :
  (y = (3 / x)) -> ¬(y = x / 3) ∧ ¬(y = 3 / (x + 1)) ∧ ¬(y = 3 * x) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_l698_69893


namespace NUMINAMATH_GPT_balance_balls_l698_69836

theorem balance_balls (G Y B W : ℝ) (h₁ : 4 * G = 10 * B) (h₂ : 3 * Y = 8 * B) (h₃ : 8 * B = 6 * W) :
  5 * G + 5 * Y + 4 * W = 31.1 * B :=
by
  sorry

end NUMINAMATH_GPT_balance_balls_l698_69836


namespace NUMINAMATH_GPT_find_D_l698_69879

theorem find_D (P Q : ℕ) (h_pos : 0 < P ∧ 0 < Q) (h_eq : P + Q + P * Q = 90) : P + Q = 18 := by
  sorry

end NUMINAMATH_GPT_find_D_l698_69879


namespace NUMINAMATH_GPT_combined_mpg_l698_69803

theorem combined_mpg :
  let mR := 150 -- miles Ray drives
  let mT := 300 -- miles Tom drives
  let mpgR := 50 -- miles per gallon for Ray's car
  let mpgT := 20 -- miles per gallon for Tom's car
  -- Total gasoline used by Ray and Tom
  let gR := mR / mpgR
  let gT := mT / mpgT
  -- Total distance driven
  let total_distance := mR + mT
  -- Total gasoline used
  let total_gasoline := gR + gT
  -- Combined miles per gallon
  let combined_mpg := total_distance / total_gasoline
  combined_mpg = 25 := by
    sorry

end NUMINAMATH_GPT_combined_mpg_l698_69803


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l698_69841

theorem quadratic_inequality_solution_set :
  {x : ℝ | - x ^ 2 + 4 * x + 12 > 0} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l698_69841


namespace NUMINAMATH_GPT_compound_interest_correct_l698_69897
noncomputable def compound_interest_proof : Prop :=
  let si := 55
  let r := 5
  let t := 2
  let p := si * 100 / (r * t)
  let ci := p * ((1 + r / 100)^t - 1)
  ci = 56.375

theorem compound_interest_correct : compound_interest_proof :=
by {
  sorry
}

end NUMINAMATH_GPT_compound_interest_correct_l698_69897


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l698_69863

theorem quadratic_inequality_solution_set (x : ℝ) :
  (x^2 - 3 * x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l698_69863


namespace NUMINAMATH_GPT_simple_interest_rate_l698_69814

theorem simple_interest_rate (P : ℝ) (T : ℝ) (r : ℝ) (h1 : T = 10) (h2 : (3 / 5) * P = (P * r * T) / 100) : r = 6 := by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l698_69814


namespace NUMINAMATH_GPT_max_value_abs_diff_PQ_PR_l698_69830

-- Definitions for the points on the given curves
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1
def circle1 (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 1

-- Statement of the problem as a theorem
theorem max_value_abs_diff_PQ_PR (P Q R : ℝ × ℝ)
(hyp_P : hyperbola P.1 P.2)
(hyp_Q : circle1 Q.1 Q.2)
(hyp_R : circle2 R.1 R.2) :
  max (abs (dist P Q - dist P R)) = 10 :=
sorry

end NUMINAMATH_GPT_max_value_abs_diff_PQ_PR_l698_69830


namespace NUMINAMATH_GPT_min_area_of_triangle_l698_69848

noncomputable def area_of_triangle (p q : ℤ) : ℚ :=
  (1 / 2 : ℚ) * abs (3 * p - 5 * q)

theorem min_area_of_triangle :
  (∀ p q : ℤ, p ≠ 0 ∨ q ≠ 0 → area_of_triangle p q ≥ (1 / 2 : ℚ)) ∧
  (∃ p q : ℤ, p ≠ 0 ∨ q ≠ 0 ∧ area_of_triangle p q = (1 / 2 : ℚ)) := 
by { 
  sorry 
}

end NUMINAMATH_GPT_min_area_of_triangle_l698_69848


namespace NUMINAMATH_GPT_unique_intersection_point_l698_69894

def line1 (x y : ℝ) : Prop := 3 * x + 2 * y = 9
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 10
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1
def line5 (x y : ℝ) : Prop := x + y = 4

theorem unique_intersection_point :
  ∃! (p : ℝ × ℝ), 
     line1 p.1 p.2 ∧ 
     line2 p.1 p.2 ∧ 
     line3 p.1 ∧ 
     line4 p.2 ∧ 
     line5 p.1 p.2 :=
sorry

end NUMINAMATH_GPT_unique_intersection_point_l698_69894


namespace NUMINAMATH_GPT_range_of_y_l698_69817

theorem range_of_y (y: ℝ) (hy: y > 0) (h_eq: ⌈y⌉ * ⌊y⌋ = 72) : 8 < y ∧ y < 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_y_l698_69817


namespace NUMINAMATH_GPT_convex_polygon_quadrilateral_division_l698_69885

open Nat

theorem convex_polygon_quadrilateral_division (n : ℕ) : ℕ :=
  if h : n > 0 then
    1 / (2 * n - 1) * (Nat.choose (3 * n - 3) (n - 1))
  else
    0

end NUMINAMATH_GPT_convex_polygon_quadrilateral_division_l698_69885


namespace NUMINAMATH_GPT_maximize_prob_l698_69829

-- Define the probability of correctly answering each question
def prob_A : ℝ := 0.6
def prob_B : ℝ := 0.8
def prob_C : ℝ := 0.5

-- Define the probability of getting two questions correct in a row for each order
def prob_A_first : ℝ := (prob_A * prob_B * (1 - prob_C) + (1 - prob_A) * prob_B * prob_C) +
                        (prob_A * prob_C * (1 - prob_B) + (1 - prob_A) * prob_C * prob_B)
def prob_B_first : ℝ := (prob_B * prob_A * (1 - prob_C) + (1 - prob_B) * prob_A * prob_C) +
                        (prob_B * prob_C * (1 - prob_A) + (1 - prob_B) * prob_C * prob_A)
def prob_C_first : ℝ := (prob_C * prob_A * (1 - prob_B) + (1 - prob_C) * prob_A * prob_B) +
                        (prob_C * prob_B * (1 - prob_A) + (1 - prob_C) * prob_B * prob_A)

-- Prove that the maximum probability is obtained when question C is answered first
theorem maximize_prob : prob_C_first > prob_A_first ∧ prob_C_first > prob_B_first :=
by
  -- Add the proof details here
  sorry

end NUMINAMATH_GPT_maximize_prob_l698_69829


namespace NUMINAMATH_GPT_rate_per_kg_of_grapes_l698_69825

-- Define the conditions 
namespace Problem

-- Given conditions
variables (G : ℝ) (rate_mangoes : ℝ := 55) (cost_paid : ℝ := 1055)
variables (kg_grapes : ℝ := 8) (kg_mangoes : ℝ := 9)

-- Statement to prove
theorem rate_per_kg_of_grapes : 8 * G + 9 * rate_mangoes = cost_paid → G = 70 := 
by
  intro h
  sorry -- proof goes here

end Problem

end NUMINAMATH_GPT_rate_per_kg_of_grapes_l698_69825


namespace NUMINAMATH_GPT_correct_option_is_D_l698_69818

noncomputable def expression1 (a b : ℝ) : Prop := a + b > 2 * b^2
noncomputable def expression2 (a b : ℝ) : Prop := a^5 + b^5 > a^3 * b^2 + a^2 * b^3
noncomputable def expression3 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * (a - b - 1)
noncomputable def expression4 (a b : ℝ) : Prop := (b / a) + (a / b) > 2

theorem correct_option_is_D (a b : ℝ) (h : a ≠ b) : 
  (expression3 a b ∧ ¬expression1 a b ∧ ¬expression2 a b ∧ ¬expression4 a b) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_D_l698_69818


namespace NUMINAMATH_GPT_custom_deck_card_selection_l698_69801

theorem custom_deck_card_selection :
  let cards := 60
  let suits := 4
  let cards_per_suit := 15
  let red_suits := 2
  let black_suits := 2
  -- Total number of ways to pick two cards with the second of a different color
  ∃ (ways : ℕ), ways = 60 * 30 ∧ ways = 1800 := by
  sorry

end NUMINAMATH_GPT_custom_deck_card_selection_l698_69801


namespace NUMINAMATH_GPT_max_projection_area_of_tetrahedron_l698_69828

theorem max_projection_area_of_tetrahedron (a : ℝ) (h1 : a > 0) :
  ∃ (A : ℝ), (A = a^2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_max_projection_area_of_tetrahedron_l698_69828


namespace NUMINAMATH_GPT_rectangular_prism_length_l698_69842

theorem rectangular_prism_length (w l h : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : h = 3 * w) 
  (h3 : 4 * l + 4 * w + 4 * h = 256) : 
  l = 32 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_length_l698_69842


namespace NUMINAMATH_GPT_find_x_l698_69820

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vectors_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, (u.1 * k = v.1) ∧ (u.2 * k = v.2)

theorem find_x :
  let a := (1, -2)
  let b := (3, -1)
  let c := (x, 4)
  vectors_parallel (vector_add a c) (vector_add b c) → x = 3 :=
by intros; sorry

end NUMINAMATH_GPT_find_x_l698_69820


namespace NUMINAMATH_GPT_xiaoxiao_age_in_2015_l698_69838

-- Definitions for conditions
variables (x : ℕ) (T : ℕ)

-- The total age of the family in 2015 was 7 times Xiaoxiao's age
axiom h1 : T = 7 * x

-- The total age of the family in 2020 after the sibling is 6 times Xiaoxiao's age in 2020
axiom h2 : T + 19 = 6 * (x + 5)

-- Proof goal: Xiaoxiao’s age in 2015 is 11
theorem xiaoxiao_age_in_2015 : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_xiaoxiao_age_in_2015_l698_69838


namespace NUMINAMATH_GPT_percentage_of_number_l698_69809

theorem percentage_of_number (N P : ℝ) (h1 : 0.60 * N = 240) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_number_l698_69809


namespace NUMINAMATH_GPT_det_scaled_matrices_l698_69855

variable (a b c d : ℝ)

-- Given condition: determinant of the original matrix
def det_A : ℝ := Matrix.det ![![a, b], ![c, d]]

-- Problem statement: determinants of the scaled matrices
theorem det_scaled_matrices
    (h: det_A a b c d = 3) :
  Matrix.det ![![3 * a, 3 * b], ![3 * c, 3 * d]] = 27 ∧
  Matrix.det ![![4 * a, 2 * b], ![4 * c, 2 * d]] = 24 :=
by
  sorry

end NUMINAMATH_GPT_det_scaled_matrices_l698_69855


namespace NUMINAMATH_GPT_distinct_powers_exist_l698_69881

theorem distinct_powers_exist :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
    (∃ n, a1 = n^2) ∧ (∃ m, a2 = m^2) ∧
    (∃ p, b1 = p^3) ∧ (∃ q, b2 = q^3) ∧
    (∃ r, c1 = r^5) ∧ (∃ s, c2 = s^5) ∧
    (∃ t, d1 = t^7) ∧ (∃ u, d2 = u^7) ∧
    a1 - a2 = b1 - b2 ∧ b1 - b2 = c1 - c2 ∧ c1 - c2 = d1 - d2 ∧
    a1 ≠ b1 ∧ a1 ≠ c1 ∧ a1 ≠ d1 ∧ b1 ≠ c1 ∧ b1 ≠ d1 ∧ c1 ≠ d1 := 
sorry

end NUMINAMATH_GPT_distinct_powers_exist_l698_69881


namespace NUMINAMATH_GPT_percentage_deficit_l698_69887

theorem percentage_deficit
  (L W : ℝ)
  (h1 : ∃(x : ℝ), 1.10 * L * (W * (1 - x / 100)) = L * W * 1.045) :
  ∃ (x : ℝ), x = 5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_deficit_l698_69887


namespace NUMINAMATH_GPT_g_range_excludes_zero_l698_69886

noncomputable def g (x : ℝ) : ℤ :=
if x > -1 then ⌈1 / (x + 1)⌉
else ⌊1 / (x + 1)⌋

theorem g_range_excludes_zero : ¬ ∃ x : ℝ, g x = 0 := 
by 
  sorry

end NUMINAMATH_GPT_g_range_excludes_zero_l698_69886


namespace NUMINAMATH_GPT_total_selling_price_l698_69832

theorem total_selling_price (profit_per_meter cost_price_per_meter meters : ℕ)
  (h_profit : profit_per_meter = 20)
  (h_cost : cost_price_per_meter = 85)
  (h_meters : meters = 85) :
  (cost_price_per_meter + profit_per_meter) * meters = 8925 :=
by
  sorry

end NUMINAMATH_GPT_total_selling_price_l698_69832


namespace NUMINAMATH_GPT_find_other_person_weight_l698_69873

noncomputable def other_person_weight (n avg new_avg W1 : ℕ) : ℕ :=
  let total_initial := n * avg
  let new_n := n + 2
  let total_new := new_n * new_avg
  total_new - total_initial - W1

theorem find_other_person_weight:
  other_person_weight 23 48 51 78 = 93 := by
  sorry

end NUMINAMATH_GPT_find_other_person_weight_l698_69873


namespace NUMINAMATH_GPT_find_angle_A_and_triangle_perimeter_l698_69846

-- Declare the main theorem using the provided conditions and the desired results
theorem find_angle_A_and_triangle_perimeter
  (a b c : ℝ) (A B : ℝ)
  (h1 : 0 < A ∧ A < Real.pi)
  (h2 : (Real.sqrt 3) * b * c * (Real.cos A) = a * (Real.sin B))
  (h3 : a = Real.sqrt 2)
  (h4 : (c / a) = (Real.sin A / Real.sin B)) :
  (A = Real.pi / 3) ∧ (a + b + c = 3 * Real.sqrt 2) :=
  sorry -- Proof is left as an exercise

end NUMINAMATH_GPT_find_angle_A_and_triangle_perimeter_l698_69846


namespace NUMINAMATH_GPT_option_D_not_right_angled_l698_69843

def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def option_A (a b c : ℝ) : Prop :=
  b^2 = a^2 - c^2

def option_B (a b c : ℝ) : Prop :=
  a = 3 * c / 5 ∧ b = 4 * c / 5

def option_C (A B C : ℝ) : Prop :=
  C = A - B ∧ A + B + C = 180

def option_D (A B C : ℝ) : Prop :=
  A / 3 = B / 4 ∧ B / 4 = C / 5

theorem option_D_not_right_angled (a b c A B C : ℝ) :
  ¬ is_right_angled_triangle a b c ↔ option_D A B C :=
  sorry

end NUMINAMATH_GPT_option_D_not_right_angled_l698_69843


namespace NUMINAMATH_GPT_missed_bus_time_l698_69812

theorem missed_bus_time (T: ℕ) (speed_ratio: ℚ) (T_slow: ℕ) (missed_time: ℕ) : 
  T = 16 → speed_ratio = 4/5 → T_slow = (5/4) * T → missed_time = T_slow - T → missed_time = 4 :=
by
  sorry

end NUMINAMATH_GPT_missed_bus_time_l698_69812


namespace NUMINAMATH_GPT_weight_of_person_being_replaced_l698_69807

variable (W_old : ℝ)

theorem weight_of_person_being_replaced :
  (W_old : ℝ) = 35 :=
by
  -- Given: The average weight of 8 persons increases by 5 kg.
  -- The weight of the new person is 75 kg.
  -- The total weight increase is 40 kg.
  -- Prove that W_old = 35 kg.
  sorry

end NUMINAMATH_GPT_weight_of_person_being_replaced_l698_69807


namespace NUMINAMATH_GPT_sum_at_simple_interest_l698_69849

theorem sum_at_simple_interest 
  (P R : ℕ)
  (h : ((P * (R + 1) * 3) / 100) - ((P * R * 3) / 100) = 69) : 
  P = 2300 :=
by sorry

end NUMINAMATH_GPT_sum_at_simple_interest_l698_69849


namespace NUMINAMATH_GPT_aluminum_percentage_in_new_alloy_l698_69811

theorem aluminum_percentage_in_new_alloy :
  ∀ (x1 x2 x3 : ℝ),
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  x1 + x2 + x3 = 1 ∧
  0.15 * x1 + 0.3 * x2 = 0.2 →
  0.15 ≤ 0.6 * x1 + 0.45 * x3 ∧ 0.6 * x1 + 0.45 * x3 ≤ 0.40 :=
by
  -- The proof will be inserted here
  sorry

end NUMINAMATH_GPT_aluminum_percentage_in_new_alloy_l698_69811


namespace NUMINAMATH_GPT_max_value_sine_cosine_l698_69847

/-- If the maximum value of the function f(x) = 4 * sin x + a * cos x is 5, then a = ±3. -/
theorem max_value_sine_cosine (a : ℝ) :
  (∀ x : ℝ, 4 * Real.sin x + a * Real.cos x ≤ 5) →
  (∃ x : ℝ, 4 * Real.sin x + a * Real.cos x = 5) →
  a = 3 ∨ a = -3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_sine_cosine_l698_69847


namespace NUMINAMATH_GPT_supplement_of_angle_l698_69872

theorem supplement_of_angle (A : ℝ) (h : 90 - A = A - 18) : 180 - A = 126 := by
    sorry

end NUMINAMATH_GPT_supplement_of_angle_l698_69872


namespace NUMINAMATH_GPT_unique_pos_neg_roots_of_poly_l698_69858

noncomputable def poly : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^4 + Polynomial.C 5 * Polynomial.X^3 + Polynomial.C 15 * Polynomial.X - Polynomial.C 9

theorem unique_pos_neg_roots_of_poly : 
  (∃! x : ℝ, (0 < x) ∧ poly.eval x = 0) ∧ (∃! x : ℝ, (x < 0) ∧ poly.eval x = 0) :=
  sorry

end NUMINAMATH_GPT_unique_pos_neg_roots_of_poly_l698_69858


namespace NUMINAMATH_GPT_roots_of_unity_sum_l698_69802

theorem roots_of_unity_sum (x y z : ℂ) (n m p : ℕ)
  (hx : x^n = 1) (hy : y^m = 1) (hz : z^p = 1) :
  (∃ k : ℕ, (x + y + z)^k = 1) ↔ (x + y = 0 ∨ y + z = 0 ∨ z + x = 0) :=
sorry

end NUMINAMATH_GPT_roots_of_unity_sum_l698_69802


namespace NUMINAMATH_GPT_perfect_squares_between_50_and_1000_l698_69861

theorem perfect_squares_between_50_and_1000 :
  ∃ (count : ℕ), count = 24 ∧ ∀ (n : ℕ), 50 < n * n ∧ n * n < 1000 ↔ 8 ≤ n ∧ n ≤ 31 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_perfect_squares_between_50_and_1000_l698_69861


namespace NUMINAMATH_GPT_negation_of_exists_l698_69882

variable (a : ℝ)

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) : ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l698_69882


namespace NUMINAMATH_GPT_tan_beta_value_l698_69853

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 1 / 3) (h2 : Real.tan (α + β) = 1 / 2) : Real.tan β = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_beta_value_l698_69853


namespace NUMINAMATH_GPT_work_together_days_l698_69898

theorem work_together_days (hA : ∃ d : ℝ, d > 0 ∧ d = 15)
                          (hB : ∃ d : ℝ, d > 0 ∧ d = 20)
                          (hfrac : ∃ f : ℝ, f = (23 / 30)) :
  ∃ d : ℝ, d = 2 := by
  sorry

end NUMINAMATH_GPT_work_together_days_l698_69898


namespace NUMINAMATH_GPT_jellybean_problem_l698_69890

theorem jellybean_problem 
    (T L A : ℕ) 
    (h1 : T = L + 24) 
    (h2 : A = L / 2) 
    (h3 : T = 34) : 
    A = 5 := 
by 
  sorry

end NUMINAMATH_GPT_jellybean_problem_l698_69890


namespace NUMINAMATH_GPT_units_digit_product_even_composite_l698_69857

/-- The units digit of the product of the first three even composite numbers greater than 10 is 8. -/
theorem units_digit_product_even_composite :
  let a := 12
  let b := 14
  let c := 16
  (a * b * c) % 10 = 8 :=
by
  let a := 12
  let b := 14
  let c := 16
  have h : (a * b * c) % 10 = 8
  { sorry }
  exact h

end NUMINAMATH_GPT_units_digit_product_even_composite_l698_69857


namespace NUMINAMATH_GPT_Q_evaluation_at_2_l698_69837

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end NUMINAMATH_GPT_Q_evaluation_at_2_l698_69837


namespace NUMINAMATH_GPT_ratio_of_ages_l698_69823

theorem ratio_of_ages (M : ℕ) (S : ℕ) (h1 : M = 24) (h2 : S + 6 = 38) : 
  (S / M : ℚ) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l698_69823


namespace NUMINAMATH_GPT_luke_hotdogs_ratio_l698_69866

-- Definitions
def hotdogs_per_sister : ℕ := 2
def total_sisters_hotdogs : ℕ := 2 * 2 -- Ella and Emma together
def hunter_hotdogs : ℕ := 6 -- 1.5 times the total of sisters' hotdogs
def total_hotdogs : ℕ := 14

-- Ratio proof problem statement
theorem luke_hotdogs_ratio :
  ∃ x : ℕ, total_hotdogs = total_sisters_hotdogs + 4 * x + hunter_hotdogs ∧ 
    (4 * x = 2 * 1 ∧ x = 1) := 
by 
  sorry

end NUMINAMATH_GPT_luke_hotdogs_ratio_l698_69866


namespace NUMINAMATH_GPT_probability_of_selecting_two_girls_l698_69871

def total_students : ℕ := 5
def boys : ℕ := 2
def girls : ℕ := 3
def selected_students : ℕ := 2

theorem probability_of_selecting_two_girls :
  (Nat.choose girls selected_students : ℝ) / (Nat.choose total_students selected_students : ℝ) = 0.3 := by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_two_girls_l698_69871


namespace NUMINAMATH_GPT_daily_savings_amount_l698_69839

theorem daily_savings_amount (total_savings : ℕ) (days : ℕ) (daily_savings : ℕ)
  (h1 : total_savings = 12410)
  (h2 : days = 365)
  (h3 : total_savings = daily_savings * days) :
  daily_savings = 34 :=
sorry

end NUMINAMATH_GPT_daily_savings_amount_l698_69839


namespace NUMINAMATH_GPT_max_workers_l698_69888

variable {n : ℕ} -- number of workers on the smaller field
variable {S : ℕ} -- area of the smaller field
variable (a : ℕ) -- productivity of each worker

theorem max_workers 
  (h_area : ∀ large small : ℕ, large = 2 * small) 
  (h_workers : ∀ large small : ℕ, large = small + 4) 
  (h_inequality : ∀ (S : ℕ) (n a : ℕ), S / (a * n) > (2 * S) / (a * (n + 4))) :
  2 * n + 4 ≤ 10 :=
by
  -- h_area implies the area requirement
  -- h_workers implies the worker requirement
  -- h_inequality implies the time requirement
  sorry

end NUMINAMATH_GPT_max_workers_l698_69888


namespace NUMINAMATH_GPT_remainder_when_divided_by_3x_minus_6_l698_69875

def polynomial (x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 9 * x^4 + 3 * x^3 - 7

def evaluate_at (f : ℝ → ℝ) (a : ℝ) : ℝ := f a

theorem remainder_when_divided_by_3x_minus_6 :
  evaluate_at polynomial 2 = 897 :=
by
  -- Compute this value manually or use automated tools
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_3x_minus_6_l698_69875


namespace NUMINAMATH_GPT_smallest_non_factor_product_l698_69826

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 18 :=
by
  -- proof intentionally omitted
  sorry

end NUMINAMATH_GPT_smallest_non_factor_product_l698_69826


namespace NUMINAMATH_GPT_xy_in_A_l698_69876

def A : Set ℤ :=
  {z | ∃ (a b k n : ℤ), z = a^2 + k * a * b + n * b^2}

theorem xy_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := sorry

end NUMINAMATH_GPT_xy_in_A_l698_69876


namespace NUMINAMATH_GPT_find_a_for_square_of_binomial_l698_69819

theorem find_a_for_square_of_binomial (a : ℝ) :
  (∃ r s : ℝ, (r * x + s)^2 = a * x^2 + 18 * x + 9) ↔ a = 9 := 
sorry

end NUMINAMATH_GPT_find_a_for_square_of_binomial_l698_69819


namespace NUMINAMATH_GPT_square_area_l698_69816

theorem square_area (x : ℝ) (side_length : ℝ) 
  (h1_side_length : side_length = 5 * x - 10)
  (h2_side_length : side_length = 3 * (x + 4)) :
  side_length ^ 2 = 2025 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l698_69816


namespace NUMINAMATH_GPT_range_of_a_l698_69822

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → x + (4 / x) - 1 - a^2 + 2 * a > 0) : -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l698_69822


namespace NUMINAMATH_GPT_side_length_S2_l698_69834

-- Define the variables
variables (r s : ℕ)

-- Given conditions
def condition1 : Prop := 2 * r + s = 2300
def condition2 : Prop := 2 * r + 3 * s = 4000

-- The main statement to be proven
theorem side_length_S2 (h1 : condition1 r s) (h2 : condition2 r s) : s = 850 := sorry

end NUMINAMATH_GPT_side_length_S2_l698_69834


namespace NUMINAMATH_GPT_sum_of_consecutive_ints_product_eq_336_l698_69889

def consecutive_ints_sum (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

theorem sum_of_consecutive_ints_product_eq_336 (a b c : ℤ) (h1 : consecutive_ints_sum a b c) (h2 : a * b * c = 336) :
  a + b + c = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_ints_product_eq_336_l698_69889


namespace NUMINAMATH_GPT_total_weight_of_10_moles_l698_69891

theorem total_weight_of_10_moles
  (molecular_weight : ℕ)
  (moles : ℕ)
  (h_molecular_weight : molecular_weight = 2670)
  (h_moles : moles = 10) :
  moles * molecular_weight = 26700 := by
  -- By substituting the values from the hypotheses:
  -- We will get:
  -- 10 * 2670 = 26700
  sorry

end NUMINAMATH_GPT_total_weight_of_10_moles_l698_69891


namespace NUMINAMATH_GPT_train_speed_l698_69850

theorem train_speed (length_train length_bridge time_crossing speed : ℝ)
  (h1 : length_train = 100)
  (h2 : length_bridge = 300)
  (h3 : time_crossing = 24)
  (h4 : speed = (length_train + length_bridge) / time_crossing) :
  speed = 16.67 := 
sorry

end NUMINAMATH_GPT_train_speed_l698_69850


namespace NUMINAMATH_GPT_interest_rate_per_annum_l698_69868

def principal : ℝ := 8945
def simple_interest : ℝ := 4025.25
def time : ℕ := 5

theorem interest_rate_per_annum : (simple_interest * 100) / (principal * time) = 9 := by
  sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l698_69868


namespace NUMINAMATH_GPT_bucket_initial_amount_l698_69860

theorem bucket_initial_amount (A B : ℝ) 
  (h1 : A - 6 = (1 / 3) * (B + 6)) 
  (h2 : B - 6 = (1 / 2) * (A + 6)) : 
  A = 13.2 := 
sorry

end NUMINAMATH_GPT_bucket_initial_amount_l698_69860


namespace NUMINAMATH_GPT_second_machine_completion_time_l698_69813

variable (time_first_machine : ℝ) (rate_first_machine : ℝ) (rate_combined : ℝ)
variable (rate_second_machine: ℝ) (y : ℝ)

def processing_rate_first_machine := rate_first_machine = 100
def processing_rate_combined := rate_combined = 1000 / 3
def processing_rate_second_machine := rate_second_machine = rate_combined - rate_first_machine
def completion_time_second_machine := y = 1000 / rate_second_machine

theorem second_machine_completion_time
  (h1: processing_rate_first_machine rate_first_machine)
  (h2: processing_rate_combined rate_combined)
  (h3: processing_rate_second_machine rate_combined rate_first_machine rate_second_machine)
  (h4: completion_time_second_machine rate_second_machine y) :
  y = 30 / 7 :=
sorry

end NUMINAMATH_GPT_second_machine_completion_time_l698_69813


namespace NUMINAMATH_GPT_probability_of_type_A_probability_of_different_type_l698_69833

def total_questions : ℕ := 6
def type_A_questions : ℕ := 4
def type_B_questions : ℕ := 2
def select_questions : ℕ := 2

def total_combinations := Nat.choose total_questions select_questions
def type_A_combinations := Nat.choose type_A_questions select_questions
def different_type_combinations := Nat.choose type_A_questions 1 * Nat.choose type_B_questions 1

theorem probability_of_type_A : (type_A_combinations : ℚ) / total_combinations = 2 / 5 := by
  sorry

theorem probability_of_different_type : (different_type_combinations : ℚ) / total_combinations = 8 / 15 := by
  sorry

end NUMINAMATH_GPT_probability_of_type_A_probability_of_different_type_l698_69833


namespace NUMINAMATH_GPT_A_days_l698_69805

theorem A_days (B_days : ℕ) (total_wage A_wage : ℕ) (h_B_days : B_days = 15) (h_total_wage : total_wage = 3000) (h_A_wage : A_wage = 1800) :
  ∃ A_days : ℕ, A_days = 10 := by
  sorry

end NUMINAMATH_GPT_A_days_l698_69805


namespace NUMINAMATH_GPT_series_result_l698_69821

noncomputable def series_sum (u : ℕ → ℚ) (s : ℚ) : Prop :=
  ∑' n, u n = s

def nth_term (n : ℕ) : ℚ := (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_result : series_sum nth_term (1 / 200) := by
  sorry

end NUMINAMATH_GPT_series_result_l698_69821


namespace NUMINAMATH_GPT_ball_falls_total_distance_l698_69806

noncomputable def total_distance : ℕ → ℤ → ℤ → ℤ
| 0, a, _ => 0
| (n+1), a, d => a + total_distance n (a + d) d

theorem ball_falls_total_distance :
  total_distance 5 30 (-6) = 90 :=
by
  sorry

end NUMINAMATH_GPT_ball_falls_total_distance_l698_69806


namespace NUMINAMATH_GPT_train_speed_l698_69864

theorem train_speed (d t s : ℝ) (h1 : d = 320) (h2 : t = 6) (h3 : s = 53.33) :
  s = d / t :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_train_speed_l698_69864


namespace NUMINAMATH_GPT_find_Tom_favorite_numbers_l698_69896

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_multiple_of (n k : ℕ) : Prop :=
  n % k = 0

def Tom_favorite_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧
  is_multiple_of n 13 ∧
  ¬ is_multiple_of n 3 ∧
  is_multiple_of (sum_of_digits n) 4

theorem find_Tom_favorite_numbers :
  ∃ n : ℕ, Tom_favorite_number n ∧ (n = 130 ∨ n = 143) :=
by
  sorry

end NUMINAMATH_GPT_find_Tom_favorite_numbers_l698_69896


namespace NUMINAMATH_GPT_area_is_rational_l698_69880

-- Definitions of the vertices of the triangle
def point1 : (ℤ × ℤ) := (2, 3)
def point2 : (ℤ × ℤ) := (5, 7)
def point3 : (ℤ × ℤ) := (3, 4)

-- Define a function to calculate the area of a triangle given vertices with integer coordinates
def triangle_area (A B C: (ℤ × ℤ)) : ℚ :=
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

-- Define the area of our specific triangle
noncomputable def area_of_triangle_with_given_vertices := triangle_area point1 point2 point3

-- Proof statement
theorem area_is_rational : ∃ (Q : ℚ), Q = area_of_triangle_with_given_vertices := 
sorry

end NUMINAMATH_GPT_area_is_rational_l698_69880


namespace NUMINAMATH_GPT_investor_receives_7260_l698_69895

-- Define the initial conditions
def principal : ℝ := 6000
def annual_rate : ℝ := 0.10
def compoundings_per_year : ℝ := 1
def years : ℝ := 2

-- Define the compound interest formula
noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem: The investor will receive $7260 after two years
theorem investor_receives_7260 : compound_interest principal annual_rate compoundings_per_year years = 7260 := by
  sorry

end NUMINAMATH_GPT_investor_receives_7260_l698_69895


namespace NUMINAMATH_GPT_max_marks_l698_69844

theorem max_marks (M : ℕ) (h_pass : 55 / 100 * M = 510) : M = 928 :=
sorry

end NUMINAMATH_GPT_max_marks_l698_69844


namespace NUMINAMATH_GPT_find_b_in_triangle_l698_69840

theorem find_b_in_triangle (c : ℝ) (B C : ℝ) (h1 : c = Real.sqrt 3)
  (h2 : B = Real.pi / 4) (h3 : C = Real.pi / 3) : ∃ b : ℝ, b = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_in_triangle_l698_69840


namespace NUMINAMATH_GPT_negation_of_existence_l698_69854

variable (x : ℝ)

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2 * x₀ - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_existence_l698_69854


namespace NUMINAMATH_GPT_kids_in_group_l698_69845

theorem kids_in_group :
  ∃ (K : ℕ), (∃ (A : ℕ), A + K = 9 ∧ 2 * A = 14) ∧ K = 2 :=
by
  sorry

end NUMINAMATH_GPT_kids_in_group_l698_69845


namespace NUMINAMATH_GPT_measure_of_MNP_l698_69862

-- Define the conditions of the pentagon
variables {M N P Q S : Type} -- Define the vertices of the pentagon
variables {MN NP PQ QS SM : ℝ} -- Define the lengths of the sides
variables (MNP QNS : ℝ) -- Define the measures of the involved angles

-- State the conditions
-- Pentagon sides are equal
axiom equal_sides : MN = NP ∧ NP = PQ ∧ PQ = QS ∧ QS = SM ∧ SM = MN 
-- Angle relation
axiom angle_relation : MNP = 2 * QNS

-- The goal is to prove that measure of angle MNP is 60 degrees
theorem measure_of_MNP : MNP = 60 :=
by {
  sorry -- The proof goes here
}

end NUMINAMATH_GPT_measure_of_MNP_l698_69862


namespace NUMINAMATH_GPT_holiday_not_on_22nd_l698_69835

def isThirdWednesday (d : ℕ) : Prop :=
  d = 15 ∨ d = 16 ∨ d = 17 ∨ d = 18 ∨ d = 19 ∨ d = 20 ∨ d = 21

theorem holiday_not_on_22nd :
  ¬ isThirdWednesday 22 :=
by
  intro h
  cases h
  repeat { contradiction }

end NUMINAMATH_GPT_holiday_not_on_22nd_l698_69835


namespace NUMINAMATH_GPT_find_abc_value_l698_69883

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom h1 : a + 1 / b = 5
axiom h2 : b + 1 / c = 2
axiom h3 : c + 1 / a = 9 / 4

theorem find_abc_value : a * b * c = (7 + Real.sqrt 21) / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_abc_value_l698_69883


namespace NUMINAMATH_GPT_probability_smallest_divides_larger_two_l698_69865

noncomputable def number_of_ways := 20

noncomputable def successful_combinations := 11

theorem probability_smallest_divides_larger_two : (successful_combinations : ℚ) / number_of_ways = 11 / 20 :=
by
  sorry

end NUMINAMATH_GPT_probability_smallest_divides_larger_two_l698_69865


namespace NUMINAMATH_GPT_greatest_difference_l698_69877

def difference_marbles : Nat :=
  let A_diff := 4 - 2
  let B_diff := 6 - 1
  let C_diff := 9 - 3
  max (max A_diff B_diff) C_diff

theorem greatest_difference :
  difference_marbles = 6 :=
by
  sorry

end NUMINAMATH_GPT_greatest_difference_l698_69877


namespace NUMINAMATH_GPT_inequality_holds_iff_even_l698_69884

theorem inequality_holds_iff_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∀ x y z : ℝ, (x - y) ^ a * (x - z) ^ b * (y - z) ^ c ≥ 0) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_iff_even_l698_69884


namespace NUMINAMATH_GPT_garden_area_l698_69874

theorem garden_area (P : ℝ) (hP : P = 72) (l w : ℝ) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 243 := 
by
  sorry

end NUMINAMATH_GPT_garden_area_l698_69874


namespace NUMINAMATH_GPT_total_tickets_l698_69892

-- Define the initial number of tickets Tate has.
def tate_initial_tickets : ℕ := 32

-- Define the number of tickets Tate buys additionally.
def additional_tickets : ℕ := 2

-- Define the total number of tickets Tate has after buying more.
def tate_total_tickets : ℕ := tate_initial_tickets + additional_tickets

-- Define the total number of tickets Peyton has.
def peyton_tickets : ℕ := tate_total_tickets / 2

-- State the theorem to prove the total number of tickets Tate and Peyton have together.
theorem total_tickets : tate_total_tickets + peyton_tickets = 51 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_tickets_l698_69892


namespace NUMINAMATH_GPT_area_of_10th_square_l698_69831

noncomputable def area_of_square (n: ℕ) : ℚ :=
  if n = 1 then 4
  else 2 * (1 / 2)^(n - 1)

theorem area_of_10th_square : area_of_square 10 = 1 / 256 := 
  sorry

end NUMINAMATH_GPT_area_of_10th_square_l698_69831


namespace NUMINAMATH_GPT_equilateral_triangle_M_properties_l698_69804

-- Define the points involved
variables (A B C M P Q R : ℝ)
-- Define distances from M to the sides as given by perpendiculars
variables (d_AP d_BQ d_CR d_PB d_QC d_RA : ℝ)

-- Equilateral triangle assumption and perpendiculars from M to sides
def equilateral_triangle (A B C : ℝ) : Prop := sorry
def perpendicular_from_point (M P R : ℝ) (line : ℝ) : Prop := sorry

-- Problem statement encapsulating the given conditions and what needs to be proved:
theorem equilateral_triangle_M_properties
  (h_triangle: equilateral_triangle A B C)
  (h_perp_AP: perpendicular_from_point M P A B)
  (h_perp_BQ: perpendicular_from_point M Q B C)
  (h_perp_CR: perpendicular_from_point M R C A) :
  (d_AP^2 + d_BQ^2 + d_CR^2 = d_PB^2 + d_QC^2 + d_RA^2) ∧ 
  (d_AP + d_BQ + d_CR = d_PB + d_QC + d_RA) := sorry

end NUMINAMATH_GPT_equilateral_triangle_M_properties_l698_69804


namespace NUMINAMATH_GPT_coordinate_system_and_parametric_equations_l698_69878

/-- Given the parametric equation of curve C1 is 
  x = 2 * cos φ and y = 3 * sin φ (where φ is the parameter)
  and a coordinate system with the origin as the pole and the positive half-axis of x as the polar axis.
  The polar equation of curve C2 is ρ = 2.
  The vertices of square ABCD are all on C2, arranged counterclockwise,
  with the polar coordinates of point A being (2, π/3).
  Find the Cartesian coordinates of A, B, C, and D, and prove that
  for any point P on C1, |PA|^2 + |PB|^2 + |PC|^2 + |PD|^2 is within the range [32, 52]. -/
theorem coordinate_system_and_parametric_equations
  (φ : ℝ)
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)
  (P : ℝ → ℝ × ℝ)
  (A B C D : ℝ × ℝ)
  (t : ℝ)
  (H1 : ∀ φ, P φ = (2 * Real.cos φ, 3 * Real.sin φ))
  (H2 : A = (1, Real.sqrt 3) ∧ B = (-Real.sqrt 3, 1) ∧ C = (-1, -Real.sqrt 3) ∧ D = (Real.sqrt 3, -1))
  (H3 : ∀ p : ℝ × ℝ, ∃ φ, p = P φ)
  : ∀ x y, ∃ (φ : ℝ), P φ = (x, y) →
    ∃ t, t = |P φ - A|^2 + |P φ - B|^2 + |P φ - C|^2 + |P φ - D|^2 ∧ 32 ≤ t ∧ t ≤ 52 := 
sorry

end NUMINAMATH_GPT_coordinate_system_and_parametric_equations_l698_69878


namespace NUMINAMATH_GPT_min_exponent_binomial_l698_69800

theorem min_exponent_binomial (n : ℕ) (h1 : n > 0)
  (h2 : ∃ r : ℕ, (n.choose r) / (n.choose (r + 1)) = 5 / 7) : n = 11 :=
by {
-- Note: We are merely stating the theorem here according to the instructions,
-- the proof body is omitted and hence the use of 'sorry'.
sorry
}

end NUMINAMATH_GPT_min_exponent_binomial_l698_69800


namespace NUMINAMATH_GPT_uncovered_area_is_52_l698_69827

-- Define the dimensions of the rectangles
def smaller_rectangle_length : ℕ := 4
def smaller_rectangle_width : ℕ := 2
def larger_rectangle_length : ℕ := 10
def larger_rectangle_width : ℕ := 6

-- Define the areas of both rectangles
def area_larger_rectangle : ℕ := larger_rectangle_length * larger_rectangle_width
def area_smaller_rectangle : ℕ := smaller_rectangle_length * smaller_rectangle_width

-- Define the area of the uncovered region
def area_uncovered_region : ℕ := area_larger_rectangle - area_smaller_rectangle

-- State the theorem
theorem uncovered_area_is_52 : area_uncovered_region = 52 := by sorry

end NUMINAMATH_GPT_uncovered_area_is_52_l698_69827


namespace NUMINAMATH_GPT_sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l698_69852

variable (α : ℝ)

-- Given conditions
def α_condition (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) : Prop := 
  true

-- Prove the first part: sin(π / 6 + α) = (3 + 4 * real.sqrt 3) / 10
theorem sin_pi_over_6_plus_α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.sin (π / 6 + α) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  sorry

-- Prove the second part: cos(π / 3 + 2 * α) = -(7 + 24 * real.sqrt 3) / 50
theorem cos_pi_over_3_plus_2α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.cos (π / 3 + 2 * α) = -(7 + 24 * Real.sqrt 3) / 50 :=
by
  sorry

end NUMINAMATH_GPT_sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l698_69852


namespace NUMINAMATH_GPT_simplify_frac_l698_69859

theorem simplify_frac : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_frac_l698_69859


namespace NUMINAMATH_GPT_clare_remaining_money_l698_69815

-- Definitions based on conditions
def clare_initial_money : ℕ := 47
def bread_quantity : ℕ := 4
def milk_quantity : ℕ := 2
def bread_cost : ℕ := 2
def milk_cost : ℕ := 2

-- The goal is to prove that Clare has $35 left after her purchases.
theorem clare_remaining_money : 
  clare_initial_money - (bread_quantity * bread_cost + milk_quantity * milk_cost) = 35 := 
sorry

end NUMINAMATH_GPT_clare_remaining_money_l698_69815


namespace NUMINAMATH_GPT_solve_for_a_l698_69810

theorem solve_for_a (a : ℝ) (h : 4 * a + 9 + (3 * a + 5) = 0) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l698_69810
