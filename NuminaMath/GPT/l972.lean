import Mathlib

namespace cos_150_eq_neg_half_l972_97219

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l972_97219


namespace bags_of_oranges_l972_97251

-- Define the total number of oranges in terms of bags B
def totalOranges (B : ℕ) : ℕ := 30 * B

-- Define the number of usable oranges left after considering rotten oranges
def usableOranges (B : ℕ) : ℕ := totalOranges B - 50

-- Define the oranges to be sold after keeping some for juice
def orangesToBeSold (B : ℕ) : ℕ := usableOranges B - 30

-- The theorem to state that given 220 oranges will be sold,
-- we need to find B, the number of bags of oranges
theorem bags_of_oranges (B : ℕ) : orangesToBeSold B = 220 → B = 10 :=
by
  sorry

end bags_of_oranges_l972_97251


namespace middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l972_97283

noncomputable def term_in_expansion (n k : ℕ) : ℚ :=
  (Nat.choose n k) * ((-1/2) ^ k)

theorem middle_term_in_expansion :
  term_in_expansion 8 4 = 35 / 8 := by
  sorry

theorem sum_of_odd_coefficients :
  (term_in_expansion 8 1 + term_in_expansion 8 3 + term_in_expansion 8 5 + term_in_expansion 8 7) = -(205 / 16) := by
  sorry

theorem weighted_sum_of_coefficients :
  ((1 * term_in_expansion 8 1) + (2 * term_in_expansion 8 2) + (3 * term_in_expansion 8 3) + (4 * term_in_expansion 8 4) +
  (5 * term_in_expansion 8 5) + (6 * term_in_expansion 8 6) + (7 * term_in_expansion 8 7) + (8 * term_in_expansion 8 8)) =
  -(1 / 32) := by
  sorry

end middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l972_97283


namespace base3_addition_l972_97292

theorem base3_addition :
  (2 + 1 * 3 + 2 * 9 + 1 * 27 + 2 * 81) + (1 + 1 * 3 + 2 * 9 + 2 * 27) + (2 * 9 + 1 * 27 + 0 * 81 + 2 * 243) + (1 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81) = 
  2 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81 + 1 * 243 + 1 * 729 := sorry

end base3_addition_l972_97292


namespace linear_inequalities_solution_l972_97277

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l972_97277


namespace worker_bees_hive_empty_l972_97289

theorem worker_bees_hive_empty:
  ∀ (initial_worker: ℕ) (leave_nectar: ℕ) (reassign_guard: ℕ) (return_trip: ℕ) (multiplier: ℕ),
  initial_worker = 400 →
  leave_nectar = 28 →
  reassign_guard = 30 →
  return_trip = 15 →
  multiplier = 5 →
  ((initial_worker - leave_nectar - reassign_guard + return_trip) * (1 - multiplier)) = 0 :=
by
  intros initial_worker leave_nectar reassign_guard return_trip multiplier
  sorry

end worker_bees_hive_empty_l972_97289


namespace satisfies_equation_l972_97239

theorem satisfies_equation (a b c : ℤ) (h₁ : a = b) (h₂ : b = c + 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 3 := 
by 
  sorry

end satisfies_equation_l972_97239


namespace inequality_transformation_l972_97212

theorem inequality_transformation (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) : 
  x + (n^n) / (x^n) ≥ n + 1 := 
sorry

end inequality_transformation_l972_97212


namespace find_x_l972_97206

theorem find_x (x : ℕ) (h : (85 + 32 / x : ℝ) * x = 9637) : x = 113 :=
sorry

end find_x_l972_97206


namespace probability_odd_multiple_of_5_l972_97259

theorem probability_odd_multiple_of_5 :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ 1 ≤ c ∧ c ≤ 100 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c) % 2 = 1 ∧ (a * b * c) % 5 = 0) → 
  p = 3 / 125 := 
sorry

end probability_odd_multiple_of_5_l972_97259


namespace truck_travel_distance_l972_97282

theorem truck_travel_distance (b t : ℝ) (h1 : t > 0) :
  (300 * (b / 4) / t) / 3 = (25 * b) / t :=
by
  sorry

end truck_travel_distance_l972_97282


namespace problem_equivalence_l972_97261

-- Define the given circles and their properties
def E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1

-- Define the curve C as the trajectory of the center of the moving circle P
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l intersecting curve C at points A and B with midpoint M(1,1)
def M (A B : ℝ × ℝ) : Prop := (A.1 + B.1 = 2) ∧ (A.2 + B.2 = 2)
def l (x y : ℝ) : Prop := x + 4 * y - 5 = 0

theorem problem_equivalence :
  (∀ x y, E x y ∧ F x y → C x y) ∧
  (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧ M A B → (∀ x y, l x y)) :=
sorry

end problem_equivalence_l972_97261


namespace binomial_coefficient_example_l972_97296

theorem binomial_coefficient_example :
  2 * (Nat.choose 7 4) = 70 := 
sorry

end binomial_coefficient_example_l972_97296


namespace train_speed_proof_l972_97297

theorem train_speed_proof
  (length_of_train : ℕ)
  (length_of_bridge : ℕ)
  (time_to_cross_bridge : ℕ)
  (h_train_length : length_of_train = 145)
  (h_bridge_length : length_of_bridge = 230)
  (h_time : time_to_cross_bridge = 30) :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 18 / 5 = 45 :=
by
  sorry

end train_speed_proof_l972_97297


namespace solution_set_quadratic_inequality_l972_97288

def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem solution_set_quadratic_inequality :
  {x : ℝ | quadratic_inequality_solution x} = {x : ℝ | x < -2 ∨ x > 1} :=
by
  sorry

end solution_set_quadratic_inequality_l972_97288


namespace added_number_after_doubling_l972_97256

theorem added_number_after_doubling (original_number : ℕ) (result : ℕ) (added_number : ℕ) 
  (h1 : original_number = 7)
  (h2 : 3 * (2 * original_number + added_number) = result)
  (h3 : result = 69) :
  added_number = 9 :=
by
  sorry

end added_number_after_doubling_l972_97256


namespace triangle_side_lengths_l972_97269

theorem triangle_side_lengths {x : ℤ} (h₁ : x + 4 > 10) (h₂ : x + 10 > 4) (h₃ : 10 + 4 > x) :
  ∃ (n : ℕ), n = 7 :=
by
  sorry

end triangle_side_lengths_l972_97269


namespace smallest_q_p_l972_97226

noncomputable def q_p_difference : ℕ := 3

theorem smallest_q_p (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : 5 * q < 9 * p) (h2 : 9 * p < 5 * q) : q - p = q_p_difference → q = 7 :=
by
  sorry

end smallest_q_p_l972_97226


namespace inequality_proof_equality_case_l972_97231

-- Defining that a, b, c are positive real numbers
variables (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The main theorem statement
theorem inequality_proof :
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 >= 6 * Real.sqrt 3 :=
sorry

-- Equality case
theorem equality_case :
  a = b ∧ b = c ∧ a = Real.sqrt 3^(1/4) →
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 = 6 * Real.sqrt 3 :=
sorry

end inequality_proof_equality_case_l972_97231


namespace find_digit_P_l972_97270

theorem find_digit_P (P Q R S T : ℕ) (digits : Finset ℕ) (h1 : digits = {1, 2, 3, 6, 8}) 
(h2 : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T)
(h3 : P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ T ∈ digits)
(hPQR_div_6 : (100 * P + 10 * Q + R) % 6 = 0)
(hQRS_div_8 : (100 * Q + 10 * R + S) % 8 = 0)
(hRST_div_3 : (100 * R + 10 * S + T) % 3 = 0) : 
P = 2 := 
sorry

end find_digit_P_l972_97270


namespace wam_gm_gt_hm_l972_97246

noncomputable def wam (w v a b : ℝ) : ℝ := w * a + v * b
noncomputable def gm (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def hm (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

theorem wam_gm_gt_hm
  (a b w v : ℝ)
  (h1 : 0 < a ∧ 0 < b)
  (h2 : 0 < w ∧ 0 < v)
  (h3 : w + v = 1)
  (h4 : a ≠ b) :
  wam w v a b > gm a b ∧ gm a b > hm a b :=
by
  -- Proof omitted
  sorry

end wam_gm_gt_hm_l972_97246


namespace find_coordinates_of_P_l972_97266

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

end find_coordinates_of_P_l972_97266


namespace discount_equivalence_l972_97234

theorem discount_equivalence :
  ∀ (p d1 d2 : ℝ) (d : ℝ),
    p = 800 →
    d1 = 0.15 →
    d2 = 0.10 →
    p * (1 - d1) * (1 - d2) = p * (1 - d) →
    d = 0.235 := by
  intros p d1 d2 d hp hd1 hd2 heq
  sorry

end discount_equivalence_l972_97234


namespace right_triangle_hypotenuse_inequality_l972_97233

theorem right_triangle_hypotenuse_inequality
  (a b c m : ℝ)
  (h_right_triangle : c^2 = a^2 + b^2)
  (h_area_relation : a * b = c * m) :
  m + c > a + b :=
by
  sorry

end right_triangle_hypotenuse_inequality_l972_97233


namespace games_left_is_correct_l972_97227

-- Define the initial number of DS games
def initial_games : ℕ := 98

-- Define the number of games given away
def games_given_away : ℕ := 7

-- Define the number of games left
def games_left : ℕ := initial_games - games_given_away

-- Theorem statement to prove that the number of games left is 91
theorem games_left_is_correct : games_left = 91 :=
by
  -- Currently, we use sorry to skip the actual proof part.
  sorry

end games_left_is_correct_l972_97227


namespace initial_bird_families_l972_97290

/- Definitions: -/
def birds_away_africa : ℕ := 23
def birds_away_asia : ℕ := 37
def birds_left_mountain : ℕ := 25

/- Theorem (Question and Correct Answer): -/
theorem initial_bird_families : birds_away_africa + birds_away_asia + birds_left_mountain = 85 := by
  sorry

end initial_bird_families_l972_97290


namespace seq_v13_eq_b_l972_97291

noncomputable def seq (v : ℕ → ℝ) (b : ℝ) : Prop :=
v 1 = b ∧ ∀ n ≥ 1, v (n + 1) = -1 / (v n + 2)

theorem seq_v13_eq_b (b : ℝ) (hb : 0 < b) (v : ℕ → ℝ) (hs : seq v b) : v 13 = b := by
  sorry

end seq_v13_eq_b_l972_97291


namespace system_equations_solution_exists_l972_97263

theorem system_equations_solution_exists (m : ℝ) :
  (∃ x y : ℝ, y = 3 * m * x + 6 ∧ y = (4 * m - 3) * x + 9) ↔ m ≠ 3 :=
by
  sorry

end system_equations_solution_exists_l972_97263


namespace equation_of_circle_O2_equation_of_tangent_line_l972_97264

-- Define circle O1
def circle_O1 (x y : ℝ) : Prop :=
  x^2 + (y + 1)^2 = 4

-- Define the center and radius of circle O2 given that they are externally tangent
def center_O2 : ℝ × ℝ := (3, 3)
def radius_O2 : ℝ := 3

-- Prove the equation of circle O2
theorem equation_of_circle_O2 :
  ∀ (x y : ℝ), (x - 3)^2 + (y - 3)^2 = 9 := by
  intro x y
  sorry

-- Prove the equation of the common internal tangent line to circles O1 and O2
theorem equation_of_tangent_line :
  ∀ (x y : ℝ), 3 * x + 4 * y - 21 = 0 := by
  intro x y
  sorry

end equation_of_circle_O2_equation_of_tangent_line_l972_97264


namespace time_to_fill_pool_l972_97247

variable (pool_volume : ℝ) (fill_rate : ℝ) (leak_rate : ℝ)

theorem time_to_fill_pool (h_pool_volume : pool_volume = 60)
    (h_fill_rate : fill_rate = 1.6)
    (h_leak_rate : leak_rate = 0.1) :
    (pool_volume / (fill_rate - leak_rate)) = 40 :=
by
  -- We skip the proof step, only the theorem statement is required
  sorry

end time_to_fill_pool_l972_97247


namespace square_side_length_l972_97216

theorem square_side_length (length width : ℕ) (h1 : length = 10) (h2 : width = 5) (cut_across_length : length % 2 = 0) :
  ∃ square_side : ℕ, square_side = 5 := by
  sorry

end square_side_length_l972_97216


namespace drink_total_amount_l972_97278

theorem drink_total_amount (total_amount: ℝ) (grape_juice: ℝ) (grape_proportion: ℝ) 
  (h1: grape_proportion = 0.20) (h2: grape_juice = 40) : total_amount = 200 :=
by
  -- Definitions and assumptions
  let calculation := grape_juice / grape_proportion
  -- Placeholder for the proof
  sorry

end drink_total_amount_l972_97278


namespace min_value_expression_l972_97221

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 ≥ 18 :=
by
  sorry

end min_value_expression_l972_97221


namespace average_difference_l972_97241

theorem average_difference :
  let a1 := 20
  let a2 := 40
  let a3 := 60
  let b1 := 10
  let b2 := 70
  let b3 := 13
  (a1 + a2 + a3) / 3 - (b1 + b2 + b3) / 3 = 9 := by
sorry

end average_difference_l972_97241


namespace election_votes_l972_97249

theorem election_votes (total_votes : ℕ) (h1 : (4 / 15) * total_votes = 48) : total_votes = 180 :=
sorry

end election_votes_l972_97249


namespace five_n_minus_twelve_mod_nine_l972_97229

theorem five_n_minus_twelve_mod_nine (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end five_n_minus_twelve_mod_nine_l972_97229


namespace total_notes_proof_l972_97285

variable (x : Nat)

def total_money := 10350
def fifty_notes_count := 17
def fifty_notes_value := 850  -- 17 * 50
def five_hundred_notes_value := 500 * x
def total_value_proposition := fifty_notes_value + five_hundred_notes_value = total_money

theorem total_notes_proof :
  total_value_proposition -> (fifty_notes_count + x) = 36 :=
by
  intros h
  -- The proof steps would go here, but we use sorry for now.
  sorry

end total_notes_proof_l972_97285


namespace sum_of_roots_l972_97220

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end sum_of_roots_l972_97220


namespace equivalent_operation_l972_97214

theorem equivalent_operation (x : ℚ) : 
  (x * (5 / 6) / (2 / 7)) = x * (35 / 12) :=
by
  sorry

end equivalent_operation_l972_97214


namespace rectangle_area_l972_97286

theorem rectangle_area (L B r s : ℝ) (h1 : L = 5 * r)
                       (h2 : r = s)
                       (h3 : s^2 = 16)
                       (h4 : B = 11) :
  (L * B = 220) :=
by
  sorry

end rectangle_area_l972_97286


namespace division_exponent_rule_l972_97232

theorem division_exponent_rule (a : ℝ) (h : a ≠ 0) : (a^8) / (a^2) = a^6 :=
sorry

end division_exponent_rule_l972_97232


namespace solve_investment_problem_l972_97250

def investment_problem
  (total_investment : ℝ) (etf_investment : ℝ) (mutual_funds_factor : ℝ) (mutual_funds_investment : ℝ) : Prop :=
  total_investment = etf_investment + mutual_funds_factor * etf_investment →
  mutual_funds_factor * etf_investment = mutual_funds_investment

theorem solve_investment_problem :
  investment_problem 210000 46666.67 3.5 163333.35 :=
by
  sorry

end solve_investment_problem_l972_97250


namespace denise_crayons_l972_97243

theorem denise_crayons (c : ℕ) :
  (∀ f p : ℕ, f = 30 ∧ p = 7 → c = f * p) → c = 210 :=
by
  intro h
  specialize h 30 7 ⟨rfl, rfl⟩
  exact h

end denise_crayons_l972_97243


namespace positive_difference_between_median_and_mode_l972_97211

-- Definition of the data as provided in the stem and leaf plot
def data : List ℕ := [
  21, 21, 21, 24, 25, 25,
  33, 33, 36, 37,
  40, 43, 44, 47, 49, 49,
  52, 56, 56, 58, 
  59, 59, 60, 63
]

-- Definition of mode and median calculations
def mode (l : List ℕ) : ℕ := 49  -- As determined, 49 is the mode
def median (l : List ℕ) : ℚ := (43 + 44) / 2  -- Median determined from the sorted list

-- The main theorem to prove
theorem positive_difference_between_median_and_mode (l : List ℕ) :
  abs (median l - mode l) = 5.5 := by
  sorry

end positive_difference_between_median_and_mode_l972_97211


namespace disjoint_subsets_with_same_sum_l972_97276

theorem disjoint_subsets_with_same_sum :
  ∀ (S : Finset ℕ), S.card = 10 ∧ (∀ x ∈ S, x ∈ Finset.range 101) →
  ∃ A B : Finset ℕ, A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end disjoint_subsets_with_same_sum_l972_97276


namespace units_digit_42_pow_4_add_24_pow_4_l972_97254

-- Define a function to get the units digit of a number.
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_42_pow_4_add_24_pow_4 : units_digit (42^4 + 24^4) = 2 := by
  sorry

end units_digit_42_pow_4_add_24_pow_4_l972_97254


namespace horizon_distance_ratio_l972_97222

def R : ℝ := 6000000
def h1 : ℝ := 1
def h2 : ℝ := 2

noncomputable def distance_to_horizon (R h : ℝ) : ℝ :=
  Real.sqrt (2 * R * h)

noncomputable def d1 : ℝ := distance_to_horizon R h1
noncomputable def d2 : ℝ := distance_to_horizon R h2

theorem horizon_distance_ratio : d2 / d1 = Real.sqrt 2 :=
  sorry

end horizon_distance_ratio_l972_97222


namespace max_students_distributing_items_l972_97284

-- Define the given conditions
def pens : Nat := 1001
def pencils : Nat := 910

-- Define the statement
theorem max_students_distributing_items :
  Nat.gcd pens pencils = 91 :=
by
  sorry

end max_students_distributing_items_l972_97284


namespace paint_cans_used_l972_97201

theorem paint_cans_used (initial_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) 
    (h1 : initial_rooms = 50) (h2 : lost_cans = 5) (h3 : remaining_rooms = 40) : 
    (remaining_rooms / (initial_rooms - remaining_rooms) / lost_cans) = 20 :=
by
  sorry

end paint_cans_used_l972_97201


namespace find_k_l972_97200

theorem find_k (k : ℝ) (x₁ x₂ : ℝ) (h_distinct_roots : (2*k + 3)^2 - 4*k^2 > 0)
  (h_roots : ∀ (x : ℝ), x^2 + (2*k + 3)*x + k^2 = 0 ↔ x = x₁ ∨ x = x₂)
  (h_reciprocal_sum : 1/x₁ + 1/x₂ = -1) : k = 3 :=
by
  sorry

end find_k_l972_97200


namespace minimize_quadratic_l972_97294

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l972_97294


namespace range_of_x_l972_97281

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (x / (1 + 2 * x))

theorem range_of_x (x : ℝ) :
  f (x * (3 * x - 2)) < -1 / 3 ↔ (-(1 / 3) < x ∧ x < 0) ∨ ((2 / 3) < x ∧ x < 1) :=
by
  sorry

end range_of_x_l972_97281


namespace distance_between_petya_and_misha_l972_97230

theorem distance_between_petya_and_misha 
  (v1 v2 v3 : ℝ) -- Speeds of Misha, Dima, and Petya
  (t1 : ℝ) -- Time taken by Misha to finish the race
  (d : ℝ := 1000) -- Distance of the race
  (h1 : d - (v1 * (d / v1)) = 0)
  (h2 : d - 0.9 * v1 * (d / v1) = 100)
  (h3 : d - 0.81 * v1 * (d / v1) = 100) :
  (d - 0.81 * v1 * (d / v1) = 190) := 
sorry

end distance_between_petya_and_misha_l972_97230


namespace old_clock_slow_by_12_minutes_l972_97273

theorem old_clock_slow_by_12_minutes (overlap_interval: ℕ) (standard_day_minutes: ℕ)
  (h1: overlap_interval = 66) (h2: standard_day_minutes = 24 * 60):
  standard_day_minutes - 24 * 60 / 66 * 66 = 12 :=
by
  sorry

end old_clock_slow_by_12_minutes_l972_97273


namespace conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l972_97248

theorem conversion_7_dms_to_cms :
  7 * 100 = 700 :=
by
  sorry

theorem conversion_5_hectares_to_sms :
  5 * 10000 = 50000 :=
by
  sorry

theorem conversion_600_hectares_to_sqkms :
  600 / 100 = 6 :=
by
  sorry

theorem conversion_200_sqsmeters_to_smeters :
  200 / 100 = 2 :=
by
  sorry

end conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l972_97248


namespace sum_of_squares_of_consecutive_integers_l972_97265

theorem sum_of_squares_of_consecutive_integers (a : ℕ) (h : (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2))) :
  (a - 1)^2 + a^2 + (a + 1)^2 + (a + 2)^2 = 86 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l972_97265


namespace number_of_solutions_l972_97257

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3 - 12 * x^2 + 12

theorem number_of_solutions : ∃ x1 x2, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 ∧
  ∀ x, f x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end number_of_solutions_l972_97257


namespace decimal_expansion_of_fraction_l972_97255

/-- 
Theorem: The decimal expansion of 13 / 375 is 0.034666...
-/
theorem decimal_expansion_of_fraction : 
  let numerator := 13
  let denominator := 375
  let resulting_fraction := (numerator * 2^3) / (denominator * 2^3)
  let decimal_expansion := 0.03466666666666667
  (resulting_fraction : ℝ) = decimal_expansion :=
sorry

end decimal_expansion_of_fraction_l972_97255


namespace calculation_eq_990_l972_97244

theorem calculation_eq_990 : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 :=
by
  sorry

end calculation_eq_990_l972_97244


namespace min_value_of_f_l972_97203

noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

theorem min_value_of_f (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : c * y + b * z = a) (h2 : a * z + c * x = b) (h3 : b * x + a * y = c) :
  ∃ x y z : ℝ, f x y z = 1 / 2 := sorry

end min_value_of_f_l972_97203


namespace cube_face_coloring_l972_97271

-- Define the type of a cube's face coloring
inductive FaceColor
| black
| white

open FaceColor

def countDistinctColorings : Nat :=
  -- Function to count the number of distinct colorings considering rotational symmetry
  10

theorem cube_face_coloring :
  countDistinctColorings = 10 :=
by
  -- Skip the proof, indicating it should be proved.
  sorry

end cube_face_coloring_l972_97271


namespace find_a_range_l972_97207

variable (a k : ℝ)
variable (x : ℝ) (hx : x > 0)

def p := ∀ x > 0, x + a / x ≥ 2
def q := ∀ k : ℝ, ∃ x y : ℝ, k * x - y + 2 = 0 ∧ x^2 + y^2 / a^2 = 1

theorem find_a_range :
  (a > 0) →
  ((p a) ∨ (q a)) ∧ ¬ ((p a) ∧ (q a)) ↔ 1 ≤ a ∧ a < 2 :=
sorry

end find_a_range_l972_97207


namespace parabola_position_l972_97293

-- Define the two parabolas as functions
def parabola1 (x : ℝ) : ℝ := x^2 - 2 * x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + 2 * x + 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (1, parabola1 1) -- (1, 2)
def vertex2 : ℝ × ℝ := (-1, parabola2 (-1)) -- (-1, 0)

-- Define the proof problem where we show relative positions
theorem parabola_position :
  (vertex1.1 > vertex2.1) ∧ (vertex1.2 > vertex2.2) :=
by
  sorry

end parabola_position_l972_97293


namespace units_digit_7_pow_5_pow_3_l972_97275

theorem units_digit_7_pow_5_pow_3 : (7 ^ (5 ^ 3)) % 10 = 7 := by
  sorry

end units_digit_7_pow_5_pow_3_l972_97275


namespace miles_driven_l972_97223

theorem miles_driven (rental_fee charge_per_mile total_amount_paid : ℝ) (h₁ : rental_fee = 20.99) (h₂ : charge_per_mile = 0.25) (h₃ : total_amount_paid = 95.74) :
  (total_amount_paid - rental_fee) / charge_per_mile = 299 :=
by
  -- Placeholder for proof
  sorry

end miles_driven_l972_97223


namespace winning_percentage_is_62_l972_97295

-- Definitions based on given conditions
def candidate_winner_votes : ℕ := 992
def candidate_win_margin : ℕ := 384
def total_votes : ℕ := candidate_winner_votes + (candidate_winner_votes - candidate_win_margin)

-- The key proof statement
theorem winning_percentage_is_62 :
  ((candidate_winner_votes : ℚ) / total_votes) * 100 = 62 := 
sorry

end winning_percentage_is_62_l972_97295


namespace sumsquare_properties_l972_97228

theorem sumsquare_properties {a b c d e f g h i : ℕ} (hc1 : a + b + c = d + e + f) 
(hc2 : d + e + f = g + h + i) 
(hc3 : a + e + i = d + e + f) 
(hc4 : c + e + g = d + e + f) : 
∃ m : ℕ, m % 3 = 0 ∧ (a ≤ (2 * m / 3 - 1)) ∧ (b ≤ (2 * m / 3 - 1)) ∧ (c ≤ (2 * m / 3 - 1)) ∧ (d ≤ (2 * m / 3 - 1)) ∧ (e ≤ (2 * m / 3 - 1)) ∧ (f ≤ (2 * m / 3 - 1)) ∧ (g ≤ (2 * m / 3 - 1)) ∧ (h ≤ (2 * m / 3 - 1)) ∧ (i ≤ (2 * m / 3 - 1)) := 
by {
  sorry
}

end sumsquare_properties_l972_97228


namespace jackson_paintable_area_l972_97215

namespace PaintWallCalculation

def length := 14
def width := 11
def height := 9
def windowArea := 70
def bedrooms := 4

def area_one_bedroom : ℕ :=
  2 * (length * height) + 2 * (width * height)

def paintable_area_one_bedroom : ℕ :=
  area_one_bedroom - windowArea

def total_paintable_area : ℕ :=
  bedrooms * paintable_area_one_bedroom

theorem jackson_paintable_area :
  total_paintable_area = 1520 :=
sorry

end PaintWallCalculation

end jackson_paintable_area_l972_97215


namespace wheels_in_garage_l972_97268

-- Definitions of the entities within the problem
def cars : Nat := 2
def car_wheels : Nat := 4

def riding_lawnmower : Nat := 1
def lawnmower_wheels : Nat := 4

def bicycles : Nat := 3
def bicycle_wheels : Nat := 2

def tricycle : Nat := 1
def tricycle_wheels : Nat := 3

def unicycle : Nat := 1
def unicycle_wheels : Nat := 1

-- The total number of wheels in the garage
def total_wheels :=
  (cars * car_wheels) +
  (riding_lawnmower * lawnmower_wheels) +
  (bicycles * bicycle_wheels) +
  (tricycle * tricycle_wheels) +
  (unicycle * unicycle_wheels)

-- The theorem we wish to prove
theorem wheels_in_garage : total_wheels = 22 := by
  sorry

end wheels_in_garage_l972_97268


namespace fa_plus_fb_gt_zero_l972_97218

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Define the conditions for a and b
variables (a b : ℝ)
axiom ab_pos : a + b > 0

-- State the theorem
theorem fa_plus_fb_gt_zero : f a + f b > 0 :=
sorry

end fa_plus_fb_gt_zero_l972_97218


namespace first_team_more_points_l972_97260

/-
Conditions:
  - Beth scored 12 points.
  - Jan scored 10 points.
  - Judy scored 8 points.
  - Angel scored 11 points.
Question:
  - How many more points did the first team get than the second team?
Prove that the first team scored 3 points more than the second team.
-/

theorem first_team_more_points
  (Beth_score : ℕ)
  (Jan_score : ℕ)
  (Judy_score : ℕ)
  (Angel_score : ℕ)
  (First_team_total : ℕ := Beth_score + Jan_score)
  (Second_team_total : ℕ := Judy_score + Angel_score)
  (Beth_score_val : Beth_score = 12)
  (Jan_score_val : Jan_score = 10)
  (Judy_score_val : Judy_score = 8)
  (Angel_score_val : Angel_score = 11)
  : First_team_total - Second_team_total = 3 := by
  sorry

end first_team_more_points_l972_97260


namespace solve_for_q_l972_97225

theorem solve_for_q (q : ℝ) (p : ℝ) (h : p = 15 * q^2 - 5) : p = 40 → q = Real.sqrt 3 :=
by
  sorry

end solve_for_q_l972_97225


namespace quadratic_equal_roots_l972_97279

theorem quadratic_equal_roots (a : ℝ) : (∀ x : ℝ, x * (x + 1) + a * x = 0) → a = -1 :=
by sorry

end quadratic_equal_roots_l972_97279


namespace complex_expression_ab_l972_97237

open Complex

theorem complex_expression_ab :
  ∀ (a b : ℝ), (2 + 3 * I) / I = a + b * I → a * b = 6 :=
by
  intros a b h
  sorry

end complex_expression_ab_l972_97237


namespace f_1001_value_l972_97209

noncomputable def f : ℕ → ℝ := sorry

theorem f_1001_value :
  (∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) →
  f 1 = 1 →
  f 1001 = 83 :=
by
  intro h₁ h₂
  sorry

end f_1001_value_l972_97209


namespace wrapping_paper_cost_l972_97236

theorem wrapping_paper_cost :
  let cost_design1 := 4 * 4 -- 20 shirt boxes / 5 shirt boxes per roll * $4.00 per roll
  let cost_design2 := 3 * 8 -- 12 XL boxes / 4 XL boxes per roll * $8.00 per roll
  let cost_design3 := 3 * 12-- 6 XXL boxes / 2 XXL boxes per roll * $12.00 per roll
  cost_design1 + cost_design2 + cost_design3 = 76
:= by
  -- Definitions
  let cost_design1 := 4 * 4
  let cost_design2 := 3 * 8
  let cost_design3 := 3 * 12
  -- Proof (To be implemented)
  sorry

end wrapping_paper_cost_l972_97236


namespace greatest_a_l972_97274

theorem greatest_a (a : ℝ) : a^2 - 14*a + 45 ≤ 0 → a ≤ 9 :=
by
  -- placeholder for the actual proof
  sorry

end greatest_a_l972_97274


namespace eval_expression_l972_97217

theorem eval_expression (x y z : ℝ) 
  (h1 : z = y - 11) 
  (h2 : y = x + 3) 
  (h3 : x = 5)
  (h4 : x + 2 ≠ 0) 
  (h5 : y - 3 ≠ 0) 
  (h6 : z + 7 ≠ 0) : 
  ( (x + 3) / (x + 2) * (y - 1) / (y - 3) * (z + 9) / (z + 7) ) = 2.4 := 
by
  sorry

end eval_expression_l972_97217


namespace find_number_l972_97210

-- Define the condition
def is_number (x : ℝ) : Prop :=
  0.15 * x = 0.25 * 16 + 2

-- The theorem statement: proving the number is 40
theorem find_number (x : ℝ) (h : is_number x) : x = 40 :=
by
  -- We would insert the proof steps here
  sorry

end find_number_l972_97210


namespace sheep_problem_system_l972_97240

theorem sheep_problem_system :
  (∃ (x y : ℝ), 5 * x - y = -90 ∧ 50 * x - y = 0) ↔ 
  (5 * x - y = -90 ∧ 50 * x - y = 0) := 
by
  sorry

end sheep_problem_system_l972_97240


namespace ratio_proof_l972_97213

-- Definitions and conditions
variables {A B C : ℕ}

-- Given condition: A : B : C = 3 : 2 : 5
def ratio_cond (A B C : ℕ) := 3 * B = 2 * A ∧ 5 * B = 2 * C

-- Theorem statement
theorem ratio_proof (h : ratio_cond A B C) : (2 * A + 3 * B) / (A + 5 * C) = 3 / 7 :=
by sorry

end ratio_proof_l972_97213


namespace remaining_blocks_correct_l972_97208

-- Define the initial number of blocks
def initial_blocks : ℕ := 59

-- Define the number of blocks used
def used_blocks : ℕ := 36

-- Define the remaining blocks equation
def remaining_blocks : ℕ := initial_blocks - used_blocks

-- Prove that the number of remaining blocks is 23
theorem remaining_blocks_correct : remaining_blocks = 23 := by
  sorry

end remaining_blocks_correct_l972_97208


namespace perp_condition_l972_97258

def a (x : ℝ) : ℝ × ℝ := (x-1, 2)
def b : ℝ × ℝ := (2, 1)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perp_condition (x : ℝ) : dot_product (a x) b = 0 ↔ x = 0 :=
by 
  sorry

end perp_condition_l972_97258


namespace smallest_possible_sum_l972_97242

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l972_97242


namespace smallest_angle_of_triangle_l972_97267

theorem smallest_angle_of_triangle (a b c : ℕ) 
    (h1 : a = 60) (h2 : b = 70) (h3 : a + b + c = 180) : 
    c = 50 ∧ min a (min b c) = 50 :=
by {
    sorry
}

end smallest_angle_of_triangle_l972_97267


namespace num_values_sum_l972_97202

noncomputable def g : ℝ → ℝ :=
sorry

theorem num_values_sum (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - 2 * x + 2) :
  ∃ n s : ℕ, (n = 1 ∧ s = 3 ∧ n * s = 3) :=
sorry

end num_values_sum_l972_97202


namespace range_of_a_l972_97252

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.log x / Real.log 2 else Real.log (-x) / Real.log (1/2)

theorem range_of_a (a : ℝ) (h : f a > f (-a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (1 : ℝ) :=
by
  sorry

end range_of_a_l972_97252


namespace machines_solution_l972_97238

theorem machines_solution (x : ℝ) (h : x > 0) :
  (1 / (x + 10) + 1 / (x + 3) + 1 / (2 * x) = 1 / x) → x = 3 / 2 := 
by
  sorry

end machines_solution_l972_97238


namespace people_per_table_l972_97253

theorem people_per_table (initial_customers left_customers tables remaining_customers : ℕ) 
  (h1 : initial_customers = 21) 
  (h2 : left_customers = 12) 
  (h3 : tables = 3) 
  (h4 : remaining_customers = initial_customers - left_customers) 
  : remaining_customers / tables = 3 :=
by
  sorry

end people_per_table_l972_97253


namespace geometric_sequence_problem_l972_97224

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_problem (a : ℕ → ℝ) (ha : geometric_sequence a) (h : a 4 + a 8 = 1 / 2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 1 / 4 :=
sorry

end geometric_sequence_problem_l972_97224


namespace solve_equation_l972_97205

theorem solve_equation (x : ℝ) (h : (3*x^2 + 2*x + 1) / (x - 1) = 3*x + 1) : x = -1/2 :=
sorry

end solve_equation_l972_97205


namespace value_of_square_sum_l972_97245

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l972_97245


namespace minimize_expression_l972_97235

theorem minimize_expression (x : ℝ) : 
  ∃ (m : ℝ), m = 2023 ∧ ∀ x, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ m :=
sorry

end minimize_expression_l972_97235


namespace systematic_sampling_employee_l972_97280

theorem systematic_sampling_employee
    (n : ℕ)
    (employees : Finset ℕ)
    (sample : Finset ℕ)
    (h_n_52 : n = 52)
    (h_employees : employees = Finset.range 52)
    (h_sample_size : sample.card = 4)
    (h_systematic_sample : sample ⊆ employees)
    (h_in_sample : {6, 32, 45} ⊆ sample) :
    19 ∈ sample :=
by
  -- conditions 
  have h0 : 6 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h1 : 32 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h2 : 45 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h_arith : 6 + 45 = 32 + 19 :=
    by linarith
  sorry

end systematic_sampling_employee_l972_97280


namespace original_price_eq_600_l972_97204

theorem original_price_eq_600 (P : ℝ) (h1 : 300 = P * 0.5) : 
  P = 600 :=
sorry

end original_price_eq_600_l972_97204


namespace sum_of_fractions_l972_97298

theorem sum_of_fractions :
  (3 / 9) + (6 / 12) = 5 / 6 := by
  sorry

end sum_of_fractions_l972_97298


namespace smallest_circle_equation_l972_97272

-- Definitions of the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- The statement of the problem
theorem smallest_circle_equation : ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ 
  A.1 = -3 ∧ A.2 = 0 ∧ B.1 = 3 ∧ B.2 = 0 ∧ ((x - 0)^2 + (y - 0)^2 = 9) :=
by
  sorry

end smallest_circle_equation_l972_97272


namespace john_total_beats_l972_97262

noncomputable def minutes_in_hour : ℕ := 60
noncomputable def hours_per_day : ℕ := 2
noncomputable def days_played : ℕ := 3
noncomputable def beats_per_minute : ℕ := 200

theorem john_total_beats :
  (beats_per_minute * hours_per_day * minutes_in_hour * days_played) = 72000 :=
by
  -- we will implement the proof here
  sorry

end john_total_beats_l972_97262


namespace students_in_classroom_l972_97299

/-- There are some students in a classroom. Half of them have 5 notebooks each and the other half have 3 notebooks each. There are 112 notebooks in total in the classroom. Prove the number of students is 28. -/
theorem students_in_classroom (S : ℕ) (h1 : (S / 2) * 5 + (S / 2) * 3 = 112) : S = 28 := 
sorry

end students_in_classroom_l972_97299


namespace quadratic_always_real_roots_rhombus_area_when_m_minus_7_l972_97287

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := 2 * x^2 + (m - 2) * x - m

-- Statement 1: For any real number m, the quadratic equation always has real roots.
theorem quadratic_always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
by {
  -- Proof omitted
  sorry
}

-- Statement 2: When m = -7, the area of the rhombus whose diagonals are the roots of the quadratic equation is 7/4.
theorem rhombus_area_when_m_minus_7 : (∃ x1 x2 : ℝ, quadratic_eq (-7) x1 = 0 ∧ quadratic_eq (-7) x2 = 0 ∧ (1 / 2) * x1 * x2 = 7 / 4) :=
by {
  -- Proof omitted
  sorry
}

end quadratic_always_real_roots_rhombus_area_when_m_minus_7_l972_97287
