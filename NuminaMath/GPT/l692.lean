import Mathlib

namespace NUMINAMATH_GPT_ratio_of_segments_of_hypotenuse_l692_69200

theorem ratio_of_segments_of_hypotenuse (k : Real) :
  let AB := 3 * k
  let BC := 2 * k
  let AC := Real.sqrt (AB^2 + BC^2)
  ∃ D : Real, 
    let BD := (2 / 3) * D
    let AD := (4 / 9) * D
    let CD := D
    ∀ AD CD, AD / CD = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_segments_of_hypotenuse_l692_69200


namespace NUMINAMATH_GPT_ROI_difference_l692_69245

-- Definitions based on the conditions
def Emma_investment : ℝ := 300
def Briana_investment : ℝ := 500
def Emma_yield : ℝ := 0.15
def Briana_yield : ℝ := 0.10
def years : ℕ := 2

-- The goal is to prove that the difference between their 2-year ROI is $10
theorem ROI_difference :
  let Emma_ROI := Emma_investment * Emma_yield * years
  let Briana_ROI := Briana_investment * Briana_yield * years
  (Briana_ROI - Emma_ROI) = 10 :=
by
  sorry

end NUMINAMATH_GPT_ROI_difference_l692_69245


namespace NUMINAMATH_GPT_find_y_l692_69286

theorem find_y (x y : ℝ) (h1 : x = 100) (h2 : x^3 * y - 3 * x^2 * y + 3 * x * y = 3000000) : 
  y = 3000000 / (100^3 - 3 * 100^2 + 3 * 100 * 1) :=
by sorry

end NUMINAMATH_GPT_find_y_l692_69286


namespace NUMINAMATH_GPT_monotonically_increasing_sequence_l692_69282

theorem monotonically_increasing_sequence (k : ℝ) : (∀ n : ℕ+, n^2 + k * n < (n + 1)^2 + k * (n + 1)) ↔ k > -3 := by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_sequence_l692_69282


namespace NUMINAMATH_GPT_find_b_l692_69289

noncomputable def g (b x : ℝ) : ℝ := b * x^2 - Real.cos (Real.pi * x)

theorem find_b (b : ℝ) (hb : 0 < b) (h : g b (g b 1) = -Real.cos Real.pi) : b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l692_69289


namespace NUMINAMATH_GPT_probability_same_color_given_first_red_l692_69262

-- Definitions of events
def event_A (draw1 : ℕ) : Prop := draw1 = 1 -- Event A: the first ball drawn is red (drawing 1 means the first ball is red)

def event_B (draw1 draw2 : ℕ) : Prop := -- Event B: the two balls drawn are of the same color
  (draw1 = 1 ∧ draw2 = 1) ∨ (draw1 = 2 ∧ draw2 = 2)

-- Given probabilities
def P_A : ℚ := 2 / 5
def P_AB : ℚ := (2 / 5) * (1 / 4)

-- The conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem probability_same_color_given_first_red : P_B_given_A = 1 / 4 := 
by 
  unfold P_B_given_A P_A P_AB
  sorry

end NUMINAMATH_GPT_probability_same_color_given_first_red_l692_69262


namespace NUMINAMATH_GPT_min_blue_edges_l692_69241

def tetrahedron_min_blue_edges : ℕ := sorry

theorem min_blue_edges (edges_colored : ℕ → Bool) (face_has_blue_edge : ℕ → Bool) 
    (H1 : ∀ face, face_has_blue_edge face)
    (H2 : ∀ edge, face_has_blue_edge edge = True → edges_colored edge = True) : 
    tetrahedron_min_blue_edges = 2 := 
sorry

end NUMINAMATH_GPT_min_blue_edges_l692_69241


namespace NUMINAMATH_GPT_hire_charges_paid_by_b_l692_69279

theorem hire_charges_paid_by_b (total_cost : ℕ) (hours_a : ℕ) (hours_b : ℕ) (hours_c : ℕ) 
  (total_hours : ℕ) (cost_per_hour : ℕ) : 
  total_cost = 520 → hours_a = 7 → hours_b = 8 → hours_c = 11 → total_hours = hours_a + hours_b + hours_c 
  → cost_per_hour = total_cost / total_hours → 
  (hours_b * cost_per_hour) = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_hire_charges_paid_by_b_l692_69279


namespace NUMINAMATH_GPT_calculate_expression_value_l692_69271

theorem calculate_expression_value (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 8) :
  (7 * x + 5 * y) / (70 * x * y) = 57 / 400 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_value_l692_69271


namespace NUMINAMATH_GPT_even_function_a_value_l692_69294

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x^2 + (a^2 - 1) * x + (a - 1)) = ((-x)^2 + (a^2 - 1) * (-x) + (a - 1))) → (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_even_function_a_value_l692_69294


namespace NUMINAMATH_GPT_probability_of_green_ball_l692_69242

def total_balls : ℕ := 3 + 3 + 6
def green_balls : ℕ := 3

theorem probability_of_green_ball : (green_balls : ℚ) / total_balls = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_green_ball_l692_69242


namespace NUMINAMATH_GPT_bowling_ball_weight_l692_69219

-- Define the weights of the bowling balls and canoes
variables (b c : ℝ)

-- Conditions provided by the problem statement
axiom eq1 : 8 * b = 4 * c
axiom eq2 : 3 * c = 108

-- Prove that one bowling ball weighs 18 pounds
theorem bowling_ball_weight : b = 18 :=
by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l692_69219


namespace NUMINAMATH_GPT_intersection_eq_l692_69202

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 - x ≤ 0}

theorem intersection_eq : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l692_69202


namespace NUMINAMATH_GPT_base6_addition_correct_l692_69274

-- Define a function to convert a base 6 digit to its base 10 equivalent
def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | d => 0 -- for illegal digits, fallback to 0

-- Define a function to convert a number in base 6 to base 10
def convert_base6_to_base10 (n : Nat) : Nat :=
  let units := base6_to_base10 (n % 10)
  let tens := base6_to_base10 ((n / 10) % 10)
  let hundreds := base6_to_base10 ((n / 100) % 10)
  units + 6 * tens + 6 * 6 * hundreds

-- Define a function to convert a base 10 number to a base 6 number
def base10_to_base6 (n : Nat) : Nat :=
  (n % 6) + 10 * ((n / 6) % 6) + 100 * ((n / (6 * 6)) % 6)

theorem base6_addition_correct : base10_to_base6 (convert_base6_to_base10 35 + convert_base6_to_base10 25) = 104 := by
  sorry

end NUMINAMATH_GPT_base6_addition_correct_l692_69274


namespace NUMINAMATH_GPT_pairs_xy_solution_sum_l692_69260

theorem pairs_xy_solution_sum :
  ∃ (x y : ℝ) (a b c d : ℕ), 
    x + y = 5 ∧ 2 * x * y = 5 ∧ 
    (x = (5 + Real.sqrt 15) / 2 ∨ x = (5 - Real.sqrt 15) / 2) ∧ 
    a = 5 ∧ b = 1 ∧ c = 15 ∧ d = 2 ∧ a + b + c + d = 23 :=
by
  sorry

end NUMINAMATH_GPT_pairs_xy_solution_sum_l692_69260


namespace NUMINAMATH_GPT_min_pq_sq_min_value_l692_69292

noncomputable def min_pq_sq (α : ℝ) : ℝ :=
  let p := α - 2
  let q := -(α + 1)
  (p + q)^2 - 2 * (p * q)

theorem min_pq_sq_min_value : 
  (∃ (α : ℝ), ∀ p q : ℝ, 
    p^2 + q^2 = (p + q)^2 - 2 * p * q ∧ 
    (p + q = α - 2 ∧ p * q = -(α + 1))) → 
  (min_pq_sq 1) = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_pq_sq_min_value_l692_69292


namespace NUMINAMATH_GPT_spending_required_for_free_shipping_l692_69237

def shampoo_cost : ℕ := 10
def conditioner_cost : ℕ := 10
def lotion_cost : ℕ := 6
def shampoo_count : ℕ := 1
def conditioner_count : ℕ := 1
def lotion_count : ℕ := 3
def additional_spending_needed : ℕ := 12
def current_spending : ℕ := (shampoo_cost * shampoo_count) + (conditioner_cost * conditioner_count) + (lotion_cost * lotion_count)

theorem spending_required_for_free_shipping : current_spending + additional_spending_needed = 50 := by
  sorry

end NUMINAMATH_GPT_spending_required_for_free_shipping_l692_69237


namespace NUMINAMATH_GPT_sum_of_primes_less_than_10_is_17_l692_69265

-- Definition of prime numbers less than 10
def primes_less_than_10 : List ℕ := [2, 3, 5, 7]

-- Sum of the prime numbers less than 10
def sum_primes_less_than_10 : ℕ := List.sum primes_less_than_10

theorem sum_of_primes_less_than_10_is_17 : sum_primes_less_than_10 = 17 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_primes_less_than_10_is_17_l692_69265


namespace NUMINAMATH_GPT_mph_to_fps_l692_69276

theorem mph_to_fps (C G : ℝ) (x : ℝ) (hC : C = 60 * x) (hG : G = 40 * x) (h1 : 7 * C - 7 * G = 210) :
  x = 1.5 :=
by {
  -- Math proof here, but we insert sorry for now
  sorry
}

end NUMINAMATH_GPT_mph_to_fps_l692_69276


namespace NUMINAMATH_GPT_intercept_sum_l692_69232

theorem intercept_sum {x y : ℝ} 
  (h : y - 3 = -3 * (x - 5)) 
  (hx : x = 6) 
  (hy : y = 18) 
  (intercept_sum_eq : x + y = 24) : 
  x + y = 24 :=
by
  sorry

end NUMINAMATH_GPT_intercept_sum_l692_69232


namespace NUMINAMATH_GPT_find_x_l692_69284

def delta (x : ℝ) : ℝ := 4 * x + 9
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x (x : ℝ) (h : delta (phi x) = 10) : x = -23 / 36 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l692_69284


namespace NUMINAMATH_GPT_sign_of_f_based_on_C_l692_69293

def is_triangle (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem sign_of_f_based_on_C (a b c : ℝ) (R r : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) 
  (h3 : c = 2 * R * Real.sin C)
  (h4 : r = 4 * R * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2))
  (h5 : A + B + C = Real.pi)
  (h_triangle : is_triangle a b c)
  : (a + b - 2 * R - 2 * r > 0 ↔ C < Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r = 0 ↔ C = Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r < 0 ↔ C > Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_sign_of_f_based_on_C_l692_69293


namespace NUMINAMATH_GPT_min_value_fraction_l692_69238

theorem min_value_fraction (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∃c, (c = 8) ∧ (∀z w : ℝ, z > 1 → w > 1 → ((z^3 / (w - 1) + w^3 / (z - 1)) ≥ c))) :=
by 
  sorry

end NUMINAMATH_GPT_min_value_fraction_l692_69238


namespace NUMINAMATH_GPT_solve_system_of_equations_l692_69277

theorem solve_system_of_equations
  {a b c d x y z : ℝ}
  (h1 : x + y + z = 1)
  (h2 : a * x + b * y + c * z = d)
  (h3 : a^2 * x + b^2 * y + c^2 * z = d^2)
  (hne1 : a ≠ b)
  (hne2 : a ≠ c)
  (hne3 : b ≠ c) :
  x = (d - b) * (d - c) / ((a - b) * (a - c)) ∧
  y = (d - a) * (d - c) / ((b - a) * (b - c)) ∧
  z = (d - a) * (d - b) / ((c - a) * (c - b)) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l692_69277


namespace NUMINAMATH_GPT_MinTransportCost_l692_69287

noncomputable def TruckTransportOptimization :=
  ∃ (x y : ℕ), x + y = 6 ∧ 45 * x + 30 * y ≥ 240 ∧ 400 * x + 300 * y ≤ 2300 ∧ (∃ (min_cost : ℕ), min_cost = 2200 ∧ x = 4 ∧ y = 2)
  
theorem MinTransportCost : TruckTransportOptimization :=
sorry

end NUMINAMATH_GPT_MinTransportCost_l692_69287


namespace NUMINAMATH_GPT_find_annual_interest_rate_l692_69224

theorem find_annual_interest_rate 
  (TD : ℝ) (FV : ℝ) (T : ℝ) (expected_R: ℝ)
  (hTD : TD = 189)
  (hFV : FV = 1764)
  (hT : T = 9 / 12)
  (hExpected : expected_R = 16) : 
  ∃ R : ℝ, 
  (TD = (FV - (FV - TD)) * R * T / 100) ∧ 
  R = expected_R := 
by 
  sorry

end NUMINAMATH_GPT_find_annual_interest_rate_l692_69224


namespace NUMINAMATH_GPT_sixth_inequality_l692_69226

theorem sixth_inequality :
  (1 + 1/2^2 + 1/3^2 + 1/4^2 + 1/5^2 + 1/6^2 + 1/7^2) < 13/7 :=
  sorry

end NUMINAMATH_GPT_sixth_inequality_l692_69226


namespace NUMINAMATH_GPT_mono_increasing_m_value_l692_69214

theorem mono_increasing_m_value (m : ℝ) :
  (∀ x : ℝ, 0 ≤ 3 * x ^ 2 + 4 * x + m) → (m ≥ 4 / 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mono_increasing_m_value_l692_69214


namespace NUMINAMATH_GPT_ab_gt_c_l692_69239

theorem ab_gt_c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 4 / b = 1) (hc : c < 9) : a + b > c :=
sorry

end NUMINAMATH_GPT_ab_gt_c_l692_69239


namespace NUMINAMATH_GPT_geom_sequence_ratio_and_fifth_term_l692_69281

theorem geom_sequence_ratio_and_fifth_term 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 10) 
  (h₂ : a₂ = -15) 
  (h₃ : a₃ = 22.5) 
  (h₄ : a₄ = -33.75) : 
  ∃ r a₅, r = -1.5 ∧ a₅ = 50.625 ∧ (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ (a₄ = r * a₃) ∧ (a₅ = r * a₄) := 
by
  sorry

end NUMINAMATH_GPT_geom_sequence_ratio_and_fifth_term_l692_69281


namespace NUMINAMATH_GPT_value_of_x_l692_69268

theorem value_of_x (a x y : ℝ) 
  (h1 : a^(x - y) = 343) 
  (h2 : a^(x + y) = 16807) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l692_69268


namespace NUMINAMATH_GPT_women_fraction_l692_69235

/-- In a room with 100 people, 1/4 of whom are married, the maximum number of unmarried women is 40.
    We need to prove that the fraction of women in the room is 2/5. -/
theorem women_fraction (total_people : ℕ) (married_fraction : ℚ) (unmarried_women : ℕ) (W : ℚ) 
  (h1 : total_people = 100) 
  (h2 : married_fraction = 1 / 4) 
  (h3 : unmarried_women = 40) 
  (hW : W = 2 / 5) : 
  W = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_women_fraction_l692_69235


namespace NUMINAMATH_GPT_tennis_tournament_matches_l692_69208

theorem tennis_tournament_matches (num_players : ℕ) (total_days : ℕ) (rest_days : ℕ)
  (num_matches_per_day : ℕ) (matches_per_player : ℕ)
  (h1 : num_players = 10)
  (h2 : total_days = 9)
  (h3 : rest_days = 1)
  (h4 : num_matches_per_day = 5)
  (h5 : matches_per_player = 1)
  : (num_players * (num_players - 1) / 2) - (num_matches_per_day * (total_days - rest_days)) = 40 :=
by
  sorry

end NUMINAMATH_GPT_tennis_tournament_matches_l692_69208


namespace NUMINAMATH_GPT_inequality_proof_l692_69215

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : -b > 0) (h3 : a > -b) (h4 : c < 0) : 
  a * (1 - c) > b * (c - 1) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l692_69215


namespace NUMINAMATH_GPT_factor_is_two_l692_69250

theorem factor_is_two (n f : ℤ) (h1 : n = 121) (h2 : n * f - 140 = 102) : f = 2 :=
by
  sorry

end NUMINAMATH_GPT_factor_is_two_l692_69250


namespace NUMINAMATH_GPT_haley_laundry_loads_l692_69263

theorem haley_laundry_loads (shirts sweaters pants socks : ℕ) 
    (machine_capacity total_pieces : ℕ)
    (sum_of_clothing : 6 + 28 + 10 + 9 = total_pieces)
    (machine_capacity_eq : machine_capacity = 5) :
  ⌈(total_pieces:ℚ) / machine_capacity⌉ = 11 :=
by
  sorry

end NUMINAMATH_GPT_haley_laundry_loads_l692_69263


namespace NUMINAMATH_GPT_part_1_property_part_2_property_part_3_geometric_l692_69204

-- Defining properties
def prop1 (a : ℕ → ℕ) (i j m: ℕ) : Prop := i > j ∧ (a i)^2 / (a j) = a m
def prop2 (a : ℕ → ℕ) (n k l: ℕ) : Prop := n ≥ 3 ∧ k > l ∧ (a n) = (a k)^2 / (a l)

-- Part I: Sequence {a_n = n} check for property 1
theorem part_1_property (a : ℕ → ℕ) (h : ∀ n, a n = n) : ¬∃ i j m, prop1 a i j m := by
  sorry

-- Part II: Sequence {a_n = 2^(n-1)} check for property 1 and 2
theorem part_2_property (a : ℕ → ℕ) (h : ∀ n, a n = 2^(n-1)) : 
  (∀ i j, ∃ m, prop1 a i j m) ∧ (∀ n k l, prop2 a n k l) := by
  sorry

-- Part III: Increasing sequence that satisfies both properties is a geometric sequence
theorem part_3_geometric (a : ℕ → ℕ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_prop1 : ∀ i j, i > j → ∃ m, prop1 a i j m)
  (h_prop2 : ∀ n, n ≥ 3 → ∃ k l, k > l ∧ (a n) = (a k)^2 / (a l)) : 
  ∃ r, ∀ n, a (n + 1) = r * a n := by
  sorry

end NUMINAMATH_GPT_part_1_property_part_2_property_part_3_geometric_l692_69204


namespace NUMINAMATH_GPT_find_percentage_l692_69233

theorem find_percentage (P : ℕ) (h: (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 18) : P = 15 :=
sorry

end NUMINAMATH_GPT_find_percentage_l692_69233


namespace NUMINAMATH_GPT_total_games_won_l692_69244

-- Define the number of games won by the Chicago Bulls
def bulls_games : ℕ := 70

-- Define the number of games won by the Miami Heat
def heat_games : ℕ := bulls_games + 5

-- Define the total number of games won by both the Bulls and the Heat
def total_games : ℕ := bulls_games + heat_games

-- The theorem stating that the total number of games won by both teams is 145
theorem total_games_won : total_games = 145 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_games_won_l692_69244


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l692_69295

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : (a^2 + b^2 < 1) → (ab + 1 > a + b) ∧ ¬(ab + 1 > a + b ↔ a^2 + b^2 < 1) := 
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l692_69295


namespace NUMINAMATH_GPT_yasmine_chocolate_beverage_l692_69206

theorem yasmine_chocolate_beverage :
  ∃ (m s : ℕ), (∀ k : ℕ, k > 0 → (∃ n : ℕ, 4 * n = 7 * k) → (m, s) = (7 * k, 4 * k)) ∧
  (2 * 7 * 1 + 1.4 * 4 * 1) = 19.6 := by
sorry

end NUMINAMATH_GPT_yasmine_chocolate_beverage_l692_69206


namespace NUMINAMATH_GPT_correct_average_marks_l692_69272

theorem correct_average_marks :
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  correct_avg = 63.125 :=
by
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  sorry

end NUMINAMATH_GPT_correct_average_marks_l692_69272


namespace NUMINAMATH_GPT_find_value_l692_69273

theorem find_value :
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l692_69273


namespace NUMINAMATH_GPT_correct_decision_box_l692_69243

theorem correct_decision_box (a b c : ℝ) (x : ℝ) : 
  x = a ∨ x = b → (x = b → b > a) →
  (c > x) ↔ (max (max a b) c = c) :=
by sorry

end NUMINAMATH_GPT_correct_decision_box_l692_69243


namespace NUMINAMATH_GPT_prove_a2_l692_69269

def arithmetic_seq (a d : ℕ → ℝ) : Prop :=
  ∀ n m, a n + d (n - m) = a m

theorem prove_a2 (a : ℕ → ℝ) (d : ℕ → ℝ) :
  (∀ n, a n = a 0 + (n - 1) * 2) → 
  (a 1 + 4) / a 1 = (a 1 + 6) / (a 1 + 4) →
  (d 1 = 2) →
  a 2 = -6 :=
by
  intros h_seq h_geo h_common_diff
  sorry

end NUMINAMATH_GPT_prove_a2_l692_69269


namespace NUMINAMATH_GPT_intersection_is_one_l692_69285

def M : Set ℝ := {x | x - 1 = 0}
def N : Set ℝ := {x | x^2 - 3 * x + 2 = 0}

theorem intersection_is_one : M ∩ N = {1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_one_l692_69285


namespace NUMINAMATH_GPT_num_black_balls_l692_69217

theorem num_black_balls 
  (R W B : ℕ) 
  (R_eq : R = 30) 
  (prob_white : (W : ℝ) / 100 = 0.47) 
  (total_balls : R + W + B = 100) : B = 23 := 
by 
  sorry

end NUMINAMATH_GPT_num_black_balls_l692_69217


namespace NUMINAMATH_GPT_survey_households_selected_l692_69259

theorem survey_households_selected 
    (total_households : ℕ) 
    (middle_income_families : ℕ) 
    (low_income_families : ℕ) 
    (high_income_selected : ℕ)
    (total_high_income_families : ℕ)
    (total_selected_households : ℕ) 
    (H1 : total_households = 480)
    (H2 : middle_income_families = 200)
    (H3 : low_income_families = 160)
    (H4 : high_income_selected = 6)
    (H5 : total_high_income_families = total_households - (middle_income_families + low_income_families))
    (H6 : total_selected_households * total_high_income_families = high_income_selected * total_households) :
    total_selected_households = 24 :=
by
  -- The actual proof will go here:
  sorry

end NUMINAMATH_GPT_survey_households_selected_l692_69259


namespace NUMINAMATH_GPT_common_rational_root_is_negative_non_integer_l692_69278

theorem common_rational_root_is_negative_non_integer 
    (a b c d e f g : ℤ)
    (p : ℚ)
    (h1 : 90 * p^4 + a * p^3 + b * p^2 + c * p + 15 = 0)
    (h2 : 15 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 90 = 0)
    (h3 : ¬ (∃ k : ℤ, p = k))
    (h4 : p < 0) : 
  p = -1 / 3 := 
sorry

end NUMINAMATH_GPT_common_rational_root_is_negative_non_integer_l692_69278


namespace NUMINAMATH_GPT_line_equation_of_intersection_points_l692_69223

theorem line_equation_of_intersection_points (x y : ℝ) :
  (x^2 + y^2 - 6*x - 7 = 0) ∧ (x^2 + y^2 - 6*y - 27 = 0) → (3*x - 3*y = 10) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_of_intersection_points_l692_69223


namespace NUMINAMATH_GPT_problem_statement_l692_69257

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x
noncomputable def F (x : ℝ) : ℝ := f x - g x
noncomputable def m (x x₀ : ℝ) : ℝ := if x ≤ x₀ then f x else g x

-- Statement of the theorem
theorem problem_statement (x₀ x₁ x₂ n : ℝ) (hx₀ : x₀ ∈ Set.Ioo 1 2)
  (hF_root : F x₀ = 0)
  (hm_roots : m x₁ x₀ = n ∧ m x₂ x₀ = n ∧ 1 < x₁ ∧ x₁ < x₀ ∧ x₀ < x₂) :
  x₁ + x₂ > 2 * x₀ :=
sorry

end NUMINAMATH_GPT_problem_statement_l692_69257


namespace NUMINAMATH_GPT_not_car_probability_l692_69212

-- Defining the probabilities of taking different modes of transportation.
def P_train : ℝ := 0.5
def P_car : ℝ := 0.2
def P_plane : ℝ := 0.3

-- Defining the event that these probabilities are for mutually exclusive events
axiom mutually_exclusive_events : P_train + P_car + P_plane = 1

-- Statement of the theorem to prove
theorem not_car_probability : P_train + P_plane = 0.8 := 
by 
  -- Use the definitions and axiom provided
  sorry

end NUMINAMATH_GPT_not_car_probability_l692_69212


namespace NUMINAMATH_GPT_games_played_so_far_l692_69266

-- Definitions based on conditions
def total_matches := 20
def points_for_victory := 3
def points_for_draw := 1
def points_for_defeat := 0
def points_scored_so_far := 14
def points_needed := 40
def required_wins := 6

-- The proof problem
theorem games_played_so_far : 
  ∃ W D L : ℕ, 3 * W + D + 0 * L = points_scored_so_far ∧ 
  ∃ W' D' L' : ℕ, 3 * W' + D' + 0 * L' + 3 * required_wins = points_needed ∧ 
  (total_matches - required_wins = 14) :=
by 
  sorry

end NUMINAMATH_GPT_games_played_so_far_l692_69266


namespace NUMINAMATH_GPT_find_x_l692_69222

def vec_a : ℝ × ℝ × ℝ := (-2, 1, 3)
def vec_b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

theorem find_x (x : ℝ) (h : (-2) * 1 + 1 * x + 3 * (-1) = 0) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l692_69222


namespace NUMINAMATH_GPT_property_related_only_to_temperature_l692_69230

-- The conditions given in the problem
def solubility_of_ammonia_gas (T P : Prop) : Prop := T ∧ P
def ion_product_of_water (T : Prop) : Prop := T
def oxidizing_property_of_pp (T C A : Prop) : Prop := T ∧ C ∧ A
def degree_of_ionization_of_acetic_acid (T C : Prop) : Prop := T ∧ C

-- The statement to prove
theorem property_related_only_to_temperature
  (T P C A : Prop)
  (H1 : solubility_of_ammonia_gas T P)
  (H2 : ion_product_of_water T)
  (H3 : oxidizing_property_of_pp T C A)
  (H4 : degree_of_ionization_of_acetic_acid T C) :
  ∃ T, ion_product_of_water T ∧
        ¬solubility_of_ammonia_gas T P ∧
        ¬oxidizing_property_of_pp T C A ∧
        ¬degree_of_ionization_of_acetic_acid T C :=
by
  sorry

end NUMINAMATH_GPT_property_related_only_to_temperature_l692_69230


namespace NUMINAMATH_GPT_max_value_of_h_l692_69218

noncomputable def f (x : ℝ) : ℝ := -x + 3
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := min (f x) (g x)

theorem max_value_of_h : ∃ x : ℝ, h x = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_h_l692_69218


namespace NUMINAMATH_GPT_part1_part2_l692_69297

-- Define the coordinates of point P as functions of n
def pointP (n : ℝ) : ℝ × ℝ := (n + 3, 2 - 3 * n)

-- Condition 1: Point P is in the fourth quadrant
def inFourthQuadrant (n : ℝ) : Prop :=
  let point := pointP n
  point.1 > 0 ∧ point.2 < 0

-- Condition 2: Distance from P to the x-axis is 1 greater than the distance to the y-axis
def distancesCondition (n : ℝ) : Prop :=
  abs (2 - 3 * n) + 1 = abs (n + 3)

-- Definition of point Q
def pointQ (n : ℝ) : ℝ × ℝ := (n, -4)

-- Condition 3: PQ is parallel to the x-axis
def pqParallelX (n : ℝ) : Prop :=
  (pointP n).2 = (pointQ n).2

-- Theorems to prove the coordinates of point P and the length of PQ
theorem part1 (n : ℝ) (h1 : inFourthQuadrant n) (h2 : distancesCondition n) : pointP n = (6, -7) :=
sorry

theorem part2 (n : ℝ) (h1 : pqParallelX n) : abs ((pointP n).1 - (pointQ n).1) = 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l692_69297


namespace NUMINAMATH_GPT_scatter_plot_role_regression_analysis_l692_69280

theorem scatter_plot_role_regression_analysis :
  ∀ (role : String), 
  (role = "Finding the number of individuals" ∨ 
   role = "Comparing the size relationship of individual data" ∨ 
   role = "Exploring individual classification" ∨ 
   role = "Roughly judging whether variables are linearly related")
  → role = "Roughly judging whether variables are linearly related" :=
by
  intros role h
  sorry

end NUMINAMATH_GPT_scatter_plot_role_regression_analysis_l692_69280


namespace NUMINAMATH_GPT_hans_room_count_l692_69270

theorem hans_room_count :
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  available_floors * rooms_per_floor = 90 := by
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  show available_floors * rooms_per_floor = 90
  sorry

end NUMINAMATH_GPT_hans_room_count_l692_69270


namespace NUMINAMATH_GPT_discount_difference_is_correct_l692_69240

-- Define the successive discounts in percentage
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10

-- Define the store's claimed discount
def claimed_discount : ℝ := 0.45

-- Calculate the true discount
def true_discount : ℝ := 1 - ((1 - discount1) * (1 - discount2) * (1 - discount3))

-- Calculate the difference between the true discount and the claimed discount
def discount_difference : ℝ := claimed_discount - true_discount

-- State the theorem to be proved
theorem discount_difference_is_correct : discount_difference = 2.375 / 100 := by
  sorry

end NUMINAMATH_GPT_discount_difference_is_correct_l692_69240


namespace NUMINAMATH_GPT_cos_double_angle_l692_69236

theorem cos_double_angle (theta : ℝ) (h : Real.sin (Real.pi - theta) = 1 / 3) : Real.cos (2 * theta) = 7 / 9 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l692_69236


namespace NUMINAMATH_GPT_javier_savings_l692_69255

theorem javier_savings (regular_price : ℕ) (discount1 : ℕ) (discount2 : ℕ) : 
  (regular_price = 50) 
  ∧ (discount1 = 40)
  ∧ (discount2 = 50) 
  → (30 = (100 * (regular_price * 3 - (regular_price + (regular_price * (100 - discount1) / 100) + regular_price / 2)) / (regular_price * 3))) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_javier_savings_l692_69255


namespace NUMINAMATH_GPT_convex_pentagon_probability_l692_69249

-- Defining the number of chords and the probability calculation as per the problem's conditions
def number_of_chords (n : ℕ) : ℕ := (n * (n - 1)) / 2
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem conditions
def eight_points_on_circle : ℕ := 8
def chords_chosen : ℕ := 5

-- Total number of chords from eight points
def total_chords : ℕ := number_of_chords eight_points_on_circle

-- The probability calculation
def probability_convex_pentagon :=
  binom 8 5 / binom total_chords chords_chosen

-- Statement to be proven
theorem convex_pentagon_probability :
  probability_convex_pentagon = 1 / 1755 := sorry

end NUMINAMATH_GPT_convex_pentagon_probability_l692_69249


namespace NUMINAMATH_GPT_simplify_expression_l692_69234

theorem simplify_expression (a c b : ℝ) (h1 : a > c) (h2 : c ≥ 0) (h3 : b > 0) :
  (a * b^2 * (1 / (a + c)^2 + 1 / (a - c)^2) = a - b) → (2 * a * b = a^2 - c^2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l692_69234


namespace NUMINAMATH_GPT_y_pow_one_div_x_neq_x_pow_y_l692_69248

theorem y_pow_one_div_x_neq_x_pow_y (t : ℝ) (ht : t > 1) : 
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  (y ^ (1 / x) ≠ x ^ y) :=
by
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  sorry

end NUMINAMATH_GPT_y_pow_one_div_x_neq_x_pow_y_l692_69248


namespace NUMINAMATH_GPT_hanoi_moves_correct_l692_69201

def hanoi_moves (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * hanoi_moves (n - 1) + 1

theorem hanoi_moves_correct (n : ℕ) : hanoi_moves n = 2^n - 1 := by
  sorry

end NUMINAMATH_GPT_hanoi_moves_correct_l692_69201


namespace NUMINAMATH_GPT_arithmetic_sequence_term_l692_69231

theorem arithmetic_sequence_term (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 3 = 4) : a 10 = 18 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_l692_69231


namespace NUMINAMATH_GPT_total_earnings_l692_69228

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def num_phone_repairs : ℕ := 5
def num_laptop_repairs : ℕ := 2
def num_computer_repairs : ℕ := 2

theorem total_earnings :
  phone_repair_cost * num_phone_repairs
  + laptop_repair_cost * num_laptop_repairs
  + computer_repair_cost * num_computer_repairs = 121 := by
  sorry

end NUMINAMATH_GPT_total_earnings_l692_69228


namespace NUMINAMATH_GPT_wholesale_prices_l692_69264

-- Definitions for the problem conditions
variable (p1 p2 d k : ℝ)
variable (h_d : d > 0)
variable (h_k : k > 1)
variable (prices : Finset ℝ)
variable (h_prices : prices = {64, 64, 70, 72})

-- The theorem statement to prove
theorem wholesale_prices :
  ∃ p1 p2, (p1 + d ∈ prices ∧ k * p1 ∈ prices) ∧ 
           (p2 + d ∈ prices ∧ k * p2 ∈ prices) ∧ 
           p1 ≠ p2
:= sorry

end NUMINAMATH_GPT_wholesale_prices_l692_69264


namespace NUMINAMATH_GPT_find_values_l692_69220

theorem find_values (x : ℝ) (h : 2 * Real.cos x - 5 * Real.sin x = 3) :
  3 * Real.sin x + 2 * Real.cos x = ( -21 + 13 * Real.sqrt 145 ) / 58 ∨
  3 * Real.sin x + 2 * Real.cos x = ( -21 - 13 * Real.sqrt 145 ) / 58 := sorry

end NUMINAMATH_GPT_find_values_l692_69220


namespace NUMINAMATH_GPT_expression_value_l692_69283

theorem expression_value :
  let x := (3 + 1 : ℚ)⁻¹ * 2
  let y := x⁻¹ * 2
  let z := y⁻¹ * 2
  z = (1 / 2 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l692_69283


namespace NUMINAMATH_GPT_factorial_division_l692_69211

theorem factorial_division :
  (Nat.factorial 4) / (Nat.factorial (4 - 3)) = 24 :=
by
  sorry

end NUMINAMATH_GPT_factorial_division_l692_69211


namespace NUMINAMATH_GPT_books_already_read_l692_69252

def total_books : ℕ := 20
def unread_books : ℕ := 5

theorem books_already_read : (total_books - unread_books = 15) :=
by
 -- Proof goes here
 sorry

end NUMINAMATH_GPT_books_already_read_l692_69252


namespace NUMINAMATH_GPT_cost_per_page_of_notebooks_l692_69209

-- Define the conditions
def notebooks : Nat := 2
def pages_per_notebook : Nat := 50
def cost_in_dollars : Nat := 5

-- Define the conversion constants
def dollars_to_cents : Nat := 100

-- Define the correct answer
def expected_cost_per_page := 5

-- State the theorem to prove the cost per page
theorem cost_per_page_of_notebooks :
  let total_pages := notebooks * pages_per_notebook
  let total_cost_in_cents := cost_in_dollars * dollars_to_cents
  let cost_per_page := total_cost_in_cents / total_pages
  cost_per_page = expected_cost_per_page :=
by
  -- Skip the proof with sorry
  sorry

end NUMINAMATH_GPT_cost_per_page_of_notebooks_l692_69209


namespace NUMINAMATH_GPT_albert_complete_laps_l692_69261

theorem albert_complete_laps (D L : ℝ) (I : ℕ) (hD : D = 256.5) (hL : L = 9.7) (hI : I = 6) :
  ⌊(D - I * L) / L⌋ = 20 :=
by
  sorry

end NUMINAMATH_GPT_albert_complete_laps_l692_69261


namespace NUMINAMATH_GPT_reduced_travel_time_l692_69256

-- Definition of conditions as given in part a)
def initial_speed := 48 -- km/h
def initial_time := 50/60 -- hours (50 minutes)
def required_speed := 60 -- km/h
def reduced_time := 40/60 -- hours (40 minutes)

-- Problem statement
theorem reduced_travel_time :
  ∃ t2, (initial_speed * initial_time = required_speed * t2) ∧ (t2 = reduced_time) :=
by
  sorry

end NUMINAMATH_GPT_reduced_travel_time_l692_69256


namespace NUMINAMATH_GPT_parking_spaces_in_the_back_l692_69207

theorem parking_spaces_in_the_back
  (front_spaces : ℕ)
  (cars_parked : ℕ)
  (half_back_filled : ℕ → ℚ)
  (spaces_available : ℕ)
  (B : ℕ)
  (h1 : front_spaces = 52)
  (h2 : cars_parked = 39)
  (h3 : half_back_filled B = B / 2)
  (h4 : spaces_available = 32) :
  B = 38 :=
by
  -- Here you can provide the proof steps.
  sorry

end NUMINAMATH_GPT_parking_spaces_in_the_back_l692_69207


namespace NUMINAMATH_GPT_g_ten_l692_69299

-- Define the function g and its properties
def g : ℝ → ℝ := sorry

axiom g_property1 : ∀ x y : ℝ, g (x * y) = 2 * g x * g y
axiom g_property2 : g 0 = 2

-- Prove that g 10 = 1 / 2
theorem g_ten : g 10 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_g_ten_l692_69299


namespace NUMINAMATH_GPT_rationalize_denominator_l692_69210

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  (4 * Real.sqrt 7 + 3 * Real.sqrt 13) ≠ 0 →
  B < D →
  ∀ (x : ℝ), x = (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) →
    A + B + C + D + E = 22 := 
by
  intros
  -- Provide the actual theorem statement here
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l692_69210


namespace NUMINAMATH_GPT_contradiction_example_l692_69225

theorem contradiction_example 
  (a b c : ℝ) 
  (h : (a - 1) * (b - 1) * (c - 1) > 0) : 
  (1 < a) ∨ (1 < b) ∨ (1 < c) :=
by
  sorry

end NUMINAMATH_GPT_contradiction_example_l692_69225


namespace NUMINAMATH_GPT_find_number_l692_69267

variable (x : ℝ)

theorem find_number (h : 2 * x - 6 = (1/4) * x + 8) : x = 8 :=
sorry

end NUMINAMATH_GPT_find_number_l692_69267


namespace NUMINAMATH_GPT_bottles_needed_l692_69296

theorem bottles_needed (runners : ℕ) (bottles_needed_per_runner : ℕ) (bottles_available : ℕ)
  (h_runners : runners = 14)
  (h_bottles_needed_per_runner : bottles_needed_per_runner = 5)
  (h_bottles_available : bottles_available = 68) :
  runners * bottles_needed_per_runner - bottles_available = 2 :=
by
  sorry

end NUMINAMATH_GPT_bottles_needed_l692_69296


namespace NUMINAMATH_GPT_polynomial_never_33_l692_69290

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_never_33_l692_69290


namespace NUMINAMATH_GPT_arithmetic_sequence_nth_term_l692_69205

theorem arithmetic_sequence_nth_term (x n : ℕ) (a1 a2 a3 : ℚ) (a_n : ℕ) :
  a1 = 3 * x - 5 ∧ a2 = 7 * x - 17 ∧ a3 = 4 * x + 3 ∧ a_n = 4033 →
  n = 641 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_nth_term_l692_69205


namespace NUMINAMATH_GPT_minimum_value_fraction_l692_69253

theorem minimum_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  ∃ (x : ℝ), (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 → x ≤ (a + b) / (a * b * c)) ∧ x = 16 / 9 := 
sorry

end NUMINAMATH_GPT_minimum_value_fraction_l692_69253


namespace NUMINAMATH_GPT_width_of_jesses_room_l692_69254

theorem width_of_jesses_room (length : ℝ) (tile_area : ℝ) (num_tiles : ℕ) (total_area : ℝ) (width : ℝ) :
  length = 2 → tile_area = 4 → num_tiles = 6 → total_area = (num_tiles * tile_area : ℝ) → (length * width) = total_area → width = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_width_of_jesses_room_l692_69254


namespace NUMINAMATH_GPT_value_range_of_2_sin_x_minus_1_l692_69288

theorem value_range_of_2_sin_x_minus_1 :
  (∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) →
  (∀ y : ℝ, y = 2 * Real.sin y - 1 → -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_value_range_of_2_sin_x_minus_1_l692_69288


namespace NUMINAMATH_GPT_zero_if_sum_of_squares_eq_zero_l692_69298

theorem zero_if_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_if_sum_of_squares_eq_zero_l692_69298


namespace NUMINAMATH_GPT_comparison_l692_69275

open Real

noncomputable def a := 5 * log (2 ^ exp 1)
noncomputable def b := 2 * log (5 ^ exp 1)
noncomputable def c := 10

theorem comparison : c > a ∧ a > b :=
by
  have a_def : a = 5 * log (2 ^ exp 1) := rfl
  have b_def : b = 2 * log (5 ^ exp 1) := rfl
  have c_def : c = 10 := rfl
  sorry -- Proof goes here

end NUMINAMATH_GPT_comparison_l692_69275


namespace NUMINAMATH_GPT_sin_15_mul_sin_75_l692_69246

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_15_mul_sin_75_l692_69246


namespace NUMINAMATH_GPT_find_f_3_l692_69213

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_find_f_3_l692_69213


namespace NUMINAMATH_GPT_triangle_area_difference_l692_69221

-- Definitions based on given lengths and right angles.
def GH : ℝ := 5
def HI : ℝ := 7
def FG : ℝ := 9

-- Note: Right angles are implicitly used in the area calculations and do not need to be represented directly in Lean.
-- Define areas for triangles involved.
def area_FGH : ℝ := 0.5 * FG * GH
def area_GHI : ℝ := 0.5 * GH * HI
def area_FHI : ℝ := 0.5 * FG * HI

-- Define areas of the triangles FGJ and HJI using variables.
variable (x y z : ℝ)
axiom area_FGJ : x = area_FHI - z
axiom area_HJI : y = area_GHI - z

-- The main proof statement involving the difference.
theorem triangle_area_difference : (x - y) = 14 := by
  sorry

end NUMINAMATH_GPT_triangle_area_difference_l692_69221


namespace NUMINAMATH_GPT_B_pow_48_l692_69229

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 0],
  ![0, 0, 1],
  ![0, -1, 0]
]

theorem B_pow_48 :
  B^48 = ![
    ![0, 0, 0],
    ![0, 1, 0],
    ![0, 0, 1]
  ] := by sorry

end NUMINAMATH_GPT_B_pow_48_l692_69229


namespace NUMINAMATH_GPT_calculate_expression_l692_69258

def f (x : ℝ) := 2 * x^2 - 3 * x + 1
def g (x : ℝ) := x + 2

theorem calculate_expression : f (1 + g 3) = 55 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l692_69258


namespace NUMINAMATH_GPT_james_pays_660_for_bed_and_frame_l692_69227

theorem james_pays_660_for_bed_and_frame :
  let bed_frame_price := 75
  let bed_price := 10 * bed_frame_price
  let total_price_before_discount := bed_frame_price + bed_price
  let discount := 0.20 * total_price_before_discount
  let final_price := total_price_before_discount - discount
  final_price = 660 := 
by
  sorry

end NUMINAMATH_GPT_james_pays_660_for_bed_and_frame_l692_69227


namespace NUMINAMATH_GPT_purely_imaginary_subtraction_l692_69291

-- Definition of the complex number z.
def z : ℂ := Complex.mk 2 (-1)

-- Statement to prove
theorem purely_imaginary_subtraction (h: z = Complex.mk 2 (-1)) : ∃ (b : ℝ), z - 2 = Complex.im b :=
by {
    sorry
}

end NUMINAMATH_GPT_purely_imaginary_subtraction_l692_69291


namespace NUMINAMATH_GPT_forum_members_l692_69203

theorem forum_members (M : ℕ)
  (h1 : ∀ q a, a = 3 * q)
  (h2 : ∀ h d, q = 3 * h * d)
  (h3 : 24 * (M * 3 * (24 + 3 * 72)) = 57600) : M = 200 :=
by
  sorry

end NUMINAMATH_GPT_forum_members_l692_69203


namespace NUMINAMATH_GPT_nat_power_of_p_iff_only_prime_factor_l692_69251

theorem nat_power_of_p_iff_only_prime_factor (p n : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℕ, n = p^k) ↔ (∀ q : ℕ, Nat.Prime q → q ∣ n → q = p) := 
sorry

end NUMINAMATH_GPT_nat_power_of_p_iff_only_prime_factor_l692_69251


namespace NUMINAMATH_GPT_geom_seq_sum_is_15_l692_69216

theorem geom_seq_sum_is_15 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) (hq : q = -2) (h_geom : ∀ n, a (n + 1) = a n * q) :
  a 1 + |a 2| + a 3 + |a 4| = 15 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_is_15_l692_69216


namespace NUMINAMATH_GPT_ternary_to_decimal_l692_69247

theorem ternary_to_decimal (n : ℕ) (h : n = 121) : 
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 16 :=
by sorry

end NUMINAMATH_GPT_ternary_to_decimal_l692_69247
