import Mathlib

namespace determine_p_l503_503179

variable (x y z p : ℝ)

theorem determine_p (h1 : 8 / (x + y) = p / (x + z)) (h2 : p / (x + z) = 12 / (z - y)) : p = 20 :=
sorry

end determine_p_l503_503179


namespace infinite_warriors_ordered_by_height_l503_503157

variable {W : Type*} [Infinite W] [LinearOrder W]

theorem infinite_warriors_ordered_by_height : 
  ∃ (subset : Set W), Infinite subset ∧ ∀ w1 w2 ∈ subset, w1 ≤ w2 ∨ w2 ≤ w1 := 
by 
  sorry

end infinite_warriors_ordered_by_height_l503_503157


namespace reflected_curve_equation_l503_503972

-- Define the original curve equation
def original_curve (x y : ℝ) : Prop :=
  2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0

-- Define the line of reflection
def line_of_reflection (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

-- Define the equation of the reflected curve
def reflected_curve (x y : ℝ) : Prop :=
  146 * x^2 - 44 * x * y + 29 * y^2 + 152 * x - 64 * y - 494 = 0

-- Problem: Prove the equation of the reflected curve is as given
theorem reflected_curve_equation (x y : ℝ) :
  (∃ x1 y1 : ℝ, original_curve x1 y1 ∧ line_of_reflection x1 y1 ∧ (x, y) = (x1, y1)) →
  reflected_curve x y :=
by
  intros
  sorry

end reflected_curve_equation_l503_503972


namespace odd_power_of_7_plus_1_divisible_by_8_l503_503006

theorem odd_power_of_7_plus_1_divisible_by_8 (n : ℕ) (h : n % 2 = 1) : (7 ^ n + 1) % 8 = 0 :=
by
  sorry

end odd_power_of_7_plus_1_divisible_by_8_l503_503006


namespace tax_paid_at_fifth_checkpoint_l503_503701

variable {x : ℚ}

theorem tax_paid_at_fifth_checkpoint (x : ℚ) (h : (x / 2) + (x / 2 * 1 / 3) + (x / 3 * 1 / 4) + (x / 4 * 1 / 5) + (x / 5 * 1 / 6) = 1) :
  (x / 5 * 1 / 6) = 1 / 25 :=
sorry

end tax_paid_at_fifth_checkpoint_l503_503701


namespace sum_Tn_l503_503221

-- Define the geometric sequence a_n
def geom_seq (a : ℕ → ℝ) := a 1 = 1 ∧ 2 * a 3 = a 2

-- Define the arithmetic sequence b_n
def arith_seq (b : ℕ → ℝ) (S : ℕ → ℝ) := b 1 = 2 ∧ S 3 = b 2 + 6

-- Define the sum of the product sequence
def sum_product_seq (a b : ℕ → ℝ) (T : ℕ → ℝ) := 
  T n = ∑ i in finset.range n, a (i + 1) * b (i + 1)

-- The theorem to be proven
theorem sum_Tn (a b : ℕ → ℝ) (S T : ℕ → ℝ)
  (ha : geom_seq a) (hb : arith_seq b S) :
  T n = 6 - (n + 3) * (1 / 2)^(n - 1) := 
sorry

end sum_Tn_l503_503221


namespace vitya_needs_58_offers_l503_503084

noncomputable def smallest_integer_k (P : ℝ → ℝ) : ℝ :=
  if H : ∃ k, k > P (100), then classical.some H else 0

theorem vitya_needs_58_offers :
  ∀ n : ℕ, n ≥ 13 → 
  (12:ℝ/13:ℝ) ^ smallest_integer_k (fun x => Real.log x / (Real.log 13 - Real.log 12)) < 0.01 :=
begin
  assume n h,
  rw smallest_integer_k,
  split_ifs,
  { sorry }, -- proof would go here
  { exfalso, exact sorry }, -- no proof steps provided
end

end vitya_needs_58_offers_l503_503084


namespace correct_statement_l503_503890

variables {Line Plane : Type} 
variable [LinearGeometry Line Plane] 
variables (m n : Line) (α β : Plane)

theorem correct_statement :
  (m ∥ α) → (m ⊥ β) → (α ⊥ β) :=
by
  -- Proof is omitted.
  sorry

end correct_statement_l503_503890


namespace boxes_same_number_oranges_l503_503931

theorem boxes_same_number_oranges 
  (total_boxes : ℕ) (min_oranges : ℕ) (max_oranges : ℕ) 
  (boxes : ℕ) (range_oranges : ℕ) :
  total_boxes = 150 →
  min_oranges = 130 →
  max_oranges = 160 →
  range_oranges = max_oranges - min_oranges + 1 →
  boxes = total_boxes / range_oranges →
  31 = range_oranges →
  4 ≤ boxes :=
by sorry

end boxes_same_number_oranges_l503_503931


namespace am_gm_hm_inequality_l503_503170

variable {x y : ℝ}

-- Conditions: x and y are positive real numbers and x < y
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x < y

-- Proof statement: A.M. > G.M. > H.M. under given conditions
theorem am_gm_hm_inequality (x y : ℝ) (h : conditions x y) :
  (x + y) / 2 > Real.sqrt (x * y) ∧ Real.sqrt (x * y) > (2 * x * y) / (x + y) :=
sorry

end am_gm_hm_inequality_l503_503170


namespace no_real_solution_condition_l503_503209

def no_real_solution (k : ℝ) : Prop :=
  let discriminant := 25 + 4 * k
  discriminant < 0

theorem no_real_solution_condition (k : ℝ) : no_real_solution k ↔ k < -25 / 4 := 
sorry

end no_real_solution_condition_l503_503209


namespace evaluate_function_l503_503271

def f : ℝ → ℝ
| x => if x ≥ 0 then Math.tan x else Real.log 10 (-x)

theorem evaluate_function :
  (f (Real.pi / 4 + 2) * f (-98)) = 2 :=
by
  sorry

end evaluate_function_l503_503271


namespace axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l503_503333

-- (1) Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, (y = x^2 - 2*t*x + 1) → (x = t) := sorry

-- (2) Comparison of m and n
theorem compare_m_n (t m n : ℝ) :
  (t - 2)^2 - 2*t*(t - 2) + 1 = m*1 →
  (t + 3)^2 - 2*t*(t + 3) + 1 = n*1 →
  n > m := sorry

-- (3) Range of t for y₁ ≤ y₂
theorem range_of_t_for_y1_leq_y2 (t x1 x2 y1 y2 : ℝ) :
  (-1 ≤ x1) → (x1 < 3) → (x2 = 3) → 
  (y1 = x1^2 - 2*t*x1 + 1) → 
  (y2 = x2^2 - 2*t*x2 + 1) → 
  y1 ≤ y2 →
  t ≤ 1 := sorry

-- (4) Maximum value of t
theorem maximum_value_of_t (t y1 y2 : ℝ) :
  (y1 = (t + 1)^2 - 2*t*(t + 1) + 1) →
  (y2 = (2*t - 4)^2 - 2*t*(2*t - 4) + 1) →
  y1 ≥ y2 →
  t = 5 := sorry

end axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l503_503333


namespace find_minimal_sum_l503_503605

theorem find_minimal_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * (x + 1)) ∣ (y * (y + 1)) →
  ¬(x ∣ y ∨ x ∣ (y + 1)) →
  ¬((x + 1) ∣ y ∨ (x + 1) ∣ (y + 1)) →
  x = 14 ∧ y = 35 ∧ x^2 + y^2 = 1421 :=
sorry

end find_minimal_sum_l503_503605


namespace charles_earnings_l503_503545

def housesit_rate : ℝ := 15
def dog_walk_rate : ℝ := 22
def hours_housesit : ℝ := 10
def num_dogs : ℝ := 3

theorem charles_earnings :
  housesit_rate * hours_housesit + dog_walk_rate * num_dogs = 216 :=
by
  sorry

end charles_earnings_l503_503545


namespace g_odd_l503_503734

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_odd {x₁ x₂ : ℝ} 
  (h₁ : |f x₁ + f x₂| ≥ |g x₁ + g x₂|)
  (hf_odd : ∀ x, f x = -f (-x)) : ∀ x, g x = -g (-x) :=
by
  -- The proof would go here, but it's omitted for the purpose of this translation.
  sorry

end g_odd_l503_503734


namespace min_a2_b2_l503_503610

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + 2 * x^2 + b * x + 1 = 0) : a^2 + b^2 ≥ 8 :=
sorry

end min_a2_b2_l503_503610


namespace problem_statement_l503_503120

theorem problem_statement : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end problem_statement_l503_503120


namespace good_subset_divisible_by_5_l503_503282

noncomputable def num_good_subsets : ℕ :=
  (Nat.factorial 1000) / ((Nat.factorial 201) * (Nat.factorial (1000 - 201)))

theorem good_subset_divisible_by_5 : num_good_subsets / 5 = (1 / 5) * num_good_subsets := 
sorry

end good_subset_divisible_by_5_l503_503282


namespace alpha_plus_beta_l503_503885

theorem alpha_plus_beta (a b c λ : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) (h_sum : a + b + c = 3) :
  let f := a^2 + b^2 + c^2 + λ * a * b * c
  ∃ α β : ℝ, (f ∈ set.Icc α β) ∧ 
    (α + β = if λ ≤ 3/2 then λ + 12 else if λ ≤ 6 then 27/2 else λ + 15/2) :=
by
  -- Conditions and assumptions are used directly from the problem statement
  sorry

end alpha_plus_beta_l503_503885


namespace river_crossing_possible_l503_503832

-- Define the weights of the individuals
def weight (person : String) : Nat :=
  if person = "Andrey" then 54
  else if person = "Oleg" then 46
  else if person = "Misha" then 98
  else 0 -- Assuming no other people are involved

-- Define the boat capacity
def boat_capacity : Nat := 100

-- Define a function to check if a given set of people can safely cross the river together
def can_cross (people : List String) : Prop :=
  people.map weight |>.sum ≤ boat_capacity

-- Define the theorem to be proved
theorem river_crossing_possible :
  ∃ steps : List (List String),
    (can_cross steps.head ∧
     can_cross steps[1] ∧
     can_cross steps[2] ∧
     can_cross steps[3] ∧
     can_cross steps[4] ∧
     can_cross steps[5])
    ∧ steps.length = 6
    ∧ steps[0] = ["Andrey", "Oleg"]
    ∧ steps[1] = ["Andrey"]
    ∧ steps[2] = ["Misha"]
    ∧ steps[3] = ["Oleg"]
    ∧ steps[4] = ["Andrey", "Oleg"]
    ∧ steps[5] = [] := sorry

end river_crossing_possible_l503_503832


namespace part1_part2_part3_l503_503264

open Real

def f (x : ℝ) : ℝ := 2 * (cos x)^2 - 2 * sqrt 3 * (sin x) * (cos x) - 1

theorem part1 : f (-π / 12) = sqrt 3 :=
by {
  sorry
}

theorem part2 : ∃ (p : ℝ), 0 < p ∧ ∀ (x : ℝ), f (x + p) = f x :=
by {
  use π,
  sorry
}

theorem part3 : ∀ (k : ℤ), k * π - (2 * π) / 3 ≤ x → x ≤ k * π - π / 6 → 
  ∃ (a b : ℝ), k * π - (2 * π) / 3 = a ∧ k * π - π / 6 = b ∧ 
  ∀ (x : ℝ), a ≤ x ∧ x ≤ b → f(x) is_strictly_increasing_on (set.Icc a b) :=
by {
  intros k x H1 H2,
  use [(k : ℝ) * π - (2 * π) / 3, (k : ℝ) * π - π / 6],
  sorry
}

end part1_part2_part3_l503_503264


namespace max_points_vasilisa_can_guarantee_l503_503402

structure Deck :=
  (cards : Finset (Suit × Rank))
  (cards_length : cards.cardinality = 36)
  (suits : Finset Suit)
  (suit_count : suits.cardinality = 4)
  (ranks : Finset Rank)
  (rank_count : ranks.cardinality = 9)

structure Player :=
  (hand : Finset (Suit × Rank))
  (hand_length : hand.cardinality = 18)
  (player_type : PlayerType)

inductive PlayerType
| Polina
| Vasilisa

noncomputable def game : PlayerType → Finset (Suit × Rank) → ℕ
| PlayerType.Polina, _ := 0
| PlayerType.Vasilisa, hand :=
    let score := 
      -- function to calculate Vasilisa's guaranteed points goes here
      sorry
    score

theorem max_points_vasilisa_can_guarantee (d : Deck) (p_hand v_hand : Finset (Suit × Rank))
  (h_p_hand : p_hand.cardinality = 18)
  (h_v_hand : v_hand.cardinality = 18)
  (h_split : p_hand ∪ v_hand = d.cards ∧ p_hand ∩ v_hand = ∅):
  game PlayerType.Vasilisa v_hand ≥ 15 :=
sorry

end max_points_vasilisa_can_guarantee_l503_503402


namespace events_are_mutually_exclusive_l503_503385

/-- Let events A and B be given -/
variables {Ω : Type*} {α β : Ω → Prop}
variables [ProbabilitySpace Ω]
noncomputable theory

/-- Define the probabilities given in the conditions -/
variables (P_A : ℝ) (P_B : ℝ) (P_A_union_B : ℝ)
hypothesis h1 : P_A = 1/5
hypothesis h2 : P_B = 1/3
hypothesis h3 : P_A_union_B = 8/15

/-- Define event mutually exclusive -/
def mutually_exclusive (α β : Ω → Prop) := Pr(α ∧ β) = 0

/-- Prove that A and B are mutually exclusive given the probability conditions -/
theorem events_are_mutually_exclusive : mutually_exclusive α β :=
begin
  have h_add : P_A + P_B = 8/15,
  { rw [h1, h2], norm_num },
  have h_union : P_A_union_B = P_A + P_B,
  { exact h3 },
  unfold mutually_exclusive,
  -- Add sorry to skip the proof.
  sorry
end

end events_are_mutually_exclusive_l503_503385


namespace rewrite_sum_l503_503218

theorem rewrite_sum (S_b S : ℕ → ℕ) (n S_1 : ℕ) (a b c : ℕ) :
  b = 4 → (a + b + c) / 3 = 6 →
  S_b n = b * n + (a + b + c) / 3 * (S n - n * S_1) →
  S_b n = 4 * n + 6 * (S n - n * S_1) := by
sorry

end rewrite_sum_l503_503218


namespace candies_remaining_l503_503874

theorem candies_remaining 
    (red_candies : ℕ)
    (yellow_candies : ℕ)
    (blue_candies : ℕ)
    (yellow_condition : yellow_candies = 3 * red_candies - 20)
    (blue_condition : blue_candies = yellow_candies / 2)
    (initial_red_candies : red_candies = 40) :
    (red_candies + yellow_candies + blue_candies - yellow_candies) = 90 := 
by
  sorry

end candies_remaining_l503_503874


namespace num_triangles_from_decagon_l503_503658

theorem num_triangles_from_decagon (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ c ≠ a → ¬(collinear a b c)) :
  (nat.choose 10 3) = 120 :=
by
  sorry

end num_triangles_from_decagon_l503_503658


namespace pass_through_walls_k_10_l503_503312

theorem pass_through_walls_k_10 (n : ℕ) : (10 * Real.sqrt (10 / n) = Real.sqrt (10 * (10 / n))) ↔ n = 99 :=
by
  -- The problem states such equations hold through a pattern
  have key_equation : ∀ k : ℕ, (k * Real.sqrt (k / (k^2 - 1))) = Real.sqrt (k * (k / (k^2 - 1))) :=
    -- We'll prove using the pattern described
    sorry
  -- Now applying it for k = 10
  rw [key_equation 10]
  -- We need to show n = 99
  split
  -- Forward implication
  { intros h
    exact sorry }
  -- Backward implication
  { intros h
    rw h
    exact sorry }

end pass_through_walls_k_10_l503_503312


namespace part1_part2_l503_503994

-- Problem (1)
def trajectory_eq (P : ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  (A = (2, 0)) ∧ ∃ y, (y^2 = 4 * fst P)

-- Problem (2)
def max_area_triangle (A B Q : ℝ × ℝ) : Prop :=
  ∀ (x1 x2 y1 y2 : ℝ), (x1 + x2 = 4) ∧ (x1 ≠ x2) ∧ (y1^2 = 4 * x1) ∧ (y2^2 = 4 * x2) →
  let M := ((x1 + x2) / 2, (y1 + y2) / 2)
      AB := (B.1 - A.1, B.2 - A.2)
      bisector := (4, 0)
      Q := (4, 0)
      area := 1/2 * abs (det (AB.1, AB.2) (Q.1, Q.2))
  in area ≤ 8

theorem part1 {P : ℝ × ℝ} (A : ℝ × ℝ) : trajectory_eq P A := sorry

theorem part2 {A B Q : ℝ × ℝ} : max_area_triangle A B Q := sorry

end part1_part2_l503_503994


namespace log_0_333_eq_neg1_l503_503977

theorem log_0_333_eq_neg1 : log 3 0.333 = -1 := by
  have h1 : 0.333 = 1 / 3 := sorry -- This would be assured by a separate lemma about decimal fractions
  have h2 : log 3 (1 / 3) = log 3 1 - log 3 3 := by
    rw [log_div]
  have h3 : log 3 1 = 0 := by
    exact log_one_eq_zero
  have h4 : log 3 3 = 1 := by
    exact log_self_eq_one
  rw [←h1, h2, h3, h4]
  exact by norm_num

end log_0_333_eq_neg1_l503_503977


namespace intersection_points_l503_503635

variables {α β : Type}
variable  (f : ℝ → ℝ)
variables (a b : ℝ)

-- Defining the set representing the graph of the function y = f(x) for x in [a, b]
def graph := {p : ℝ × ℝ | p.1 ∈ set.Icc a b ∧ p.2 = f p.1}

-- Defining the set representing the line x = 2
def line := {p : ℝ × ℝ | p.1 = 2}

-- Theorem stating the number of elements in the intersection set based on the condition 2 ∈ [a, b]
theorem intersection_points :
  fintype.card (graph f a b ∩ line) = 
    if 2 ∈ set.Icc a b then 1 else 0 :=
sorry

end intersection_points_l503_503635


namespace sin_beta_symmetric_yaxis_l503_503328

theorem sin_beta_symmetric_yaxis (α β : ℝ) (k : ℤ)
(h1 : ∃ k : ℤ, β = π + 2 * k * π - α)
(h2 : sin α = 1 / 3) :
  sin β = 1 / 3 :=
sorry

end sin_beta_symmetric_yaxis_l503_503328


namespace percentage_less_than_l503_503682

variable (x : ℝ)

-- Provided conditions
def y := 1.40 * x
def z := 1.50 * y
def w := 0.70 * z

-- The proof statement to establish x is approximately 31.97% less than w
theorem percentage_less_than (h_y : y = 1.40 * x)
                             (h_z : z = 1.50 * y)
                             (h_w : w = 0.70 * z) :
  abs ((w - x) / w - 0.47 / 1.47) < 0.0001 := sorry

end percentage_less_than_l503_503682


namespace dinos_win_all_games_l503_503776

def prob_win_single_game : ℚ := 3 / 5

def prob_win_all_games : ℚ := prob_win_single_game ^ 6

theorem dinos_win_all_games :
  prob_win_all_games = 729 / 15625 :=
by
  -- Definition of probability of winning a single game
  have h1 : prob_win_single_game = 3 / 5 := rfl
  -- Definition of probability of winning all 6 games
  have h2 : prob_win_all_games = prob_win_single_game ^ 6 := rfl
  -- Calculation of the probability of winning all 6 games
  calc
    prob_win_all_games
      = (3 / 5) ^ 6 : by rw h2; rw h1
  ... = 729 / 15625 : sorry -- This is where the proof would go

end dinos_win_all_games_l503_503776


namespace sum_of_unique_remainders_544_l503_503779

theorem sum_of_unique_remainders_544 :
  let S := {n | ∃ a b : ℕ, a ∈ {0,1,2,3,4,5,6,7,8} ∧ b ∈ {0,1,2,3,4,5,6,7,8} ∧
                               n = (1100 * a + 1010 + 11 * b) % 39}
  in (∑ x in S, x) = 544 := 
sorry

end sum_of_unique_remainders_544_l503_503779


namespace Vitya_needs_58_offers_l503_503092

theorem Vitya_needs_58_offers :
  ∃ k : ℕ, (log 0.01 / log (12 / 13) < k) ∧ k = 58 :=
by
  sorry

end Vitya_needs_58_offers_l503_503092


namespace equation_roots_l503_503816

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l503_503816


namespace evaluate_expression_l503_503186

theorem evaluate_expression (a x : ℤ) (h : x = a + 7) : x - a + 3 = 10 := by
  sorry

end evaluate_expression_l503_503186


namespace closest_point_on_line_l503_503974

open Real

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem closest_point_on_line (P Q : ℝ × ℝ) : 
  (P = (3, 1)) → 
  (∀ (x : ℝ), Q = (x, 4 * x - 3) → 
  ∀ (R S : ℝ × ℝ),
    R = (19 / 17, 41 / 17) → 
    (S = (x, 4 * x - 3) → 
    distance P R ≤ distance P S)) :=
 by
  intros
  sorry

end closest_point_on_line_l503_503974


namespace identity_product_l503_503962

theorem identity_product :
  ∃ N1 N2 : ℤ, (∀ x : ℤ, x ≠ 1 ∧ x ≠ 3 →
    (56 * x - 14) * ((x - 1) * (x - 3)) = (N1 * (x - 1) + N2 * (x - 3)) * ((x - 1) * (x - 3))) ∧
    N1 * N2 = -1617 :=
by
  use [-21, 77]
  constructor
  sorry -- Proof that the identity holds for all x
  calc
    (-21) * 77 = -1617 : by norm_num

end identity_product_l503_503962


namespace smallest_of_seven_consecutive_even_numbers_l503_503771

theorem smallest_of_seven_consecutive_even_numbers (a b c d e f g : ℤ)
  (h₁ : a + b + c + d + e + f + g = 700)
  (h₂ : b = a + 2)
  (h₃ : c = a + 4)
  (h₄ : d = a + 6)
  (h₅ : e = a + 8)
  (h₆ : f = a + 10)
  (h₇ : g = a + 12)
  : a = 94 :=
by
  -- Proof is omitted, this is just the statement.
  sorry

end smallest_of_seven_consecutive_even_numbers_l503_503771


namespace number_of_children_l503_503453
-- Import the entirety of the Mathlib library

-- Define the conditions and the theorem to be proven
theorem number_of_children (C n : ℕ) 
  (h1 : C = 8 * n + 4) 
  (h2 : C = 11 * (n - 1)) : 
  n = 5 :=
by sorry

end number_of_children_l503_503453


namespace trigonometric_identity_l503_503205

theorem trigonometric_identity 
  (deg7 deg37 deg83 : ℝ)
  (h7 : deg7 = 7) 
  (h37 : deg37 = 37) 
  (h83 : deg83 = 83) 
  : (Real.sin (deg7 * Real.pi / 180) * Real.cos (deg37 * Real.pi / 180) - Real.sin (deg83 * Real.pi / 180) * Real.sin (deg37 * Real.pi / 180) = -1/2) :=
sorry

end trigonometric_identity_l503_503205


namespace polynomial_identity_l503_503594

theorem polynomial_identity :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 a_12 a_13 a_14 a_15 a_16 a_17 a_18 a_19 a_20
     a_21 a_22 a_23 a_24 a_25 a_26 a_27 a_28 a_29 a_30 a_31 a_32 a_33 a_34 a_35 a_36 a_37 a_38 a_39
     a_40 a_41 a_42 a_43 a_44 a_45 a_46 a_47 a_48 a_49 a_50 : ℝ),
  (2 - real.sqrt 3) ^ 50 = a_0 + a_1*(1:ℝ) + a_2*(1:ℝ) ^ 2 + a_3*(1:ℝ) ^ 3 + a_4*(1:ℝ) ^ 4 + a_5*(1:ℝ) ^ 5 + a_6*(1:ℝ) ^ 6 + a_7*(1:ℝ)^ 7 +
                          a_8*(1:ℝ) ^ 8 + a_9*(1:ℝ) ^ 9 + a_10*(1:ℝ) ^ 10 + a_11*(1:ℝ) ^ 11 + a_12*(1:ℝ) ^ 12 + 
                          a_13*(1:ℝ) ^ 13 + a_14*(1:ℝ) ^ 14 + a_15*(1:ℝ) ^ 15 + a_16*(1:ℝ)^ 16 + a_17*(1:ℝ) ^ 17 +
                          a_18*(1:ℝ) ^ 18 + a_19*(1:ℝ) ^ 19 + a_20*(1:ℝ) ^ 20 + a_21*(1:ℝ) ^ 21 + a_22*(1:ℝ) ^
                          22 + a_23*(1:ℝ) ^ 23 + a_24*(1:ℝ)^ 24 + a_25*(1:ℝ) ^ 25 + a_26*(1:ℝ)^ 26 +
                          a_27*(1:ℝ)^ 27 + a_28*(1:ℝ)^ 28 + a_29*(1:ℝ)^ 29 + a_30*(1:ℝ)^ 30 + a_31*(1:ℝ)^ 31 +
                          a_32*(1:ℝ)^ 32 + a_33*(1:ℝ) ^ 33 + a_34*(1:ℝ)^ 34 + a_35*(1:ℝ) ^ 35 + a_36*(1:ℝ)^ 36 +
                          a_37*(1:ℝ) ^ 37 + a_38*(1:ℝ)^ 38 + a_39*(1:ℝ) ^ 39 + a_40*(1:ℝ)^ 40 + a_41*(1:ℝ)^ 41 +
                          a_42*(1:ℝ)^ 42 + a_43*(1:ℝ) ^ 43 + a_44*(1:ℝ)^ 44 + a_45*(1:ℝ) ^ 45 + a_46*(1:ℝ)^ 46 +
                          a_47*(1:ℝ)^ 47 + a_48*(1:ℝ) ^ 48 + a_49*(1:ℝ)^ 49 + a_50*(1:ℝ) ^ 50 ∧
  (2 + real.sqrt 3) ^ 50 = a_0 - a_1*(1:ℝ) + a_2*(1:ℝ)^ 2 - a_3*(1:ℝ)^ 3 + a_4*(1:ℝ) ^ 4 - a_5*(1:ℝ) ^ 5 +
                              a_6*(1:ℝ) ^ 6 - a_7*(1:ℝ)^ 7 + a_8*(1:ℝ) ^ 8 - a_9*(1:ℝ) ^ 9 + a_10*(1:ℝ) ^ 10 - 
                              a_11*(1:ℝ) ^ 11 + a_12*(1:ℝ) ^ 12 - a_13*(1:ℝ) ^ 13 + a_14*(1:ℝ) ^ 14 - a_15*(1:ℝ) ^ 15 +
                              a_16*(1:ℝ) ^ 16 - a_17*(1:ℝ) ^ 17 + a_18*(1:ℝ) ^ 18 - a_19*(1:ℝ) ^ 19 + a_20*(1:ℝ) ^ 20 -
                              a_21*(1:ℝ) ^ 21 + a_22*(1:ℝ) ^ 22 - a_23*(1:ℝ) ^ 23 + a_24*(1:ℝ) ^ 24 - a_25*(1:ℝ) ^ 25 +
                              a_26*(1:ℝ) ^ 26 - a_27*(1:ℝ) ^ 27 + a_28*(1:ℝ)^ 28 - a_29*(1:ℝ)^ 29 + a_30*(1:ℝ)^ 30 -
                              a_31*(1:ℝ) ^ 31 + a_32*(1:ℝ) ^ 32 - a_33*(1:ℝ) ^ 33 + a_34*(1:ℝ) ^ 34 - a_35*(1:ℝ) ^ 35 +
                              a_36*(1:ℝ)^ 36 - a_37*(1:ℝ)^ 37 + a_38*(1:ℝ)^ 38 - a_39*(1:ℝ) ^ 39 + a_40*(1:ℝ)^ 40 -
                              a_41*(1:ℝ)^ 41 + a_42*(1:ℝ) ^ 42 - a_43*(1:ℝ)^ 43 + a_44*(1:ℝ)^ 44 - a_45*(1:ℝ)^ 45 +
                              a_46*(1:ℝ)^ 46 - a_47*(1:ℝ)^ 47 + a_48*(1:ℝ)^ 48 - a_49*(1:ℝ)^ 49 + a_50*(1:ℝ)^ 50 → 
  (a_0 + a_2 + a_4 + a_6 + a_8 + a_10 + a_12 + a_14 + a_16 + a_18 + a_20 + a_22 + a_24 +
   a_26 + a_28 + a_30 + a_32 + a_34 + a_36 + a_38 + a_40 + a_42 + a_44 + a_46 + a_48 + a_50) ^ 2 - 
  (a_1 + a_3 + a_5 + a_7 + a_9 + a_11 + a_13 + a_15 + a_17 + a_19 + a_21 + a_23 + a_25 + a_27 +
   a_29 + a_31 + a_33 + a_35 + a_37 + a_39 + a_41 + a_43 + a_45 + a_47 + a_49) ^ 2 = 1 :=
 by sorry

end polynomial_identity_l503_503594


namespace find_finleys_class_students_l503_503390

-- Conditions described in the problem
def johnsons_class (F : ℕ) : ℕ := (1 / 2 : ℚ) * F + 10

theorem find_finleys_class_students (F : ℕ) 
  (h1 : johnsons_class F = 22) : F = 24 :=
by 
  sorry

end find_finleys_class_students_l503_503390


namespace roots_of_quadratic_eq_l503_503809

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l503_503809


namespace tom_picked_undetermined_plums_l503_503934

theorem tom_picked_undetermined_plums (Alyssa_limes Mike_limes Tom_plums total_limes : ℕ) 
  (hA : Alyssa_limes = 25) 
  (hM : Mike_limes = 32) 
  (hT : total_limes = 57) : 
  Tom_plums ∈ set.univ :=
by
  have : Alyssa_limes + Mike_limes = 57,
  { rw [hA, hM],
    norm_num, },
  have : total_limes = 57,
  { assumption, },
  sorry

end tom_picked_undetermined_plums_l503_503934


namespace equation_roots_l503_503817

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l503_503817


namespace time_to_pass_platform_l503_503506

-- Conditions of the problem
def length_of_train : ℕ := 1500
def time_to_cross_tree : ℕ := 100
def length_of_platform : ℕ := 500

-- Derived values according to solution steps
def speed_of_train : ℚ := length_of_train / time_to_cross_tree
def total_distance_to_pass_platform : ℕ := length_of_train + length_of_platform

-- The theorem to be proved
theorem time_to_pass_platform :
  (total_distance_to_pass_platform / speed_of_train : ℚ) = 133.33 := sorry

end time_to_pass_platform_l503_503506


namespace vitya_needs_58_offers_l503_503086

noncomputable def smallest_integer_k (P : ℝ → ℝ) : ℝ :=
  if H : ∃ k, k > P (100), then classical.some H else 0

theorem vitya_needs_58_offers :
  ∀ n : ℕ, n ≥ 13 → 
  (12:ℝ/13:ℝ) ^ smallest_integer_k (fun x => Real.log x / (Real.log 13 - Real.log 12)) < 0.01 :=
begin
  assume n h,
  rw smallest_integer_k,
  split_ifs,
  { sorry }, -- proof would go here
  { exfalso, exact sorry }, -- no proof steps provided
end

end vitya_needs_58_offers_l503_503086


namespace solve_for_y_l503_503285

theorem solve_for_y (x y : ℝ) (hx : x > 1) (hy : y > 1) (h1 : 1 / x + 1 / y = 1) (h2 : x * y = 9) :
  y = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end solve_for_y_l503_503285


namespace books_on_each_shelf_l503_503161

-- Define the conditions and the problem statement
theorem books_on_each_shelf :
  ∀ (M P : ℕ), 
  -- Conditions
  (5 * M + 4 * P = 72) ∧ (M = P) ∧ (∃ B : ℕ, M = B ∧ P = B) ->
  -- Conclusion
  (∃ B : ℕ, B = 8) :=
by
  sorry

end books_on_each_shelf_l503_503161


namespace intersection_A_B_l503_503232

-- Definition of sets A and B
def A := {x : ℝ | x > 2}
def B := { x : ℝ | (x - 1) * (x - 3) < 0 }

-- Claim that A ∩ B = {x : ℝ | 2 < x < 3}
theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l503_503232


namespace lars_breads_per_day_l503_503355

noncomputable theory

def loaves_per_hour := 10
def baguettes_per_2hours := 30
def baking_hours_per_day := 6

theorem lars_breads_per_day : 
  (loaves_per_hour * baking_hours_per_day) + (baguettes_per_2hours * (baking_hours_per_day / 2)) = 105 :=
by
  let loaves := loaves_per_hour * baking_hours_per_day
  let baguettes := baguettes_per_2hours * (baking_hours_per_day / 2)
  have h_loaves : loaves = 60 := sorry  -- skipped proof
  have h_baguettes : baguettes = 45 := sorry  -- skipped proof
  show loaves + baguettes = 105 by rw [h_loaves, h_baguettes]; exact rfl

end lars_breads_per_day_l503_503355


namespace evaluate_expression_l503_503192

theorem evaluate_expression :
  8^6 * 27^6 * 8^15 * 27^15 = 216^21 :=
by
  sorry

end evaluate_expression_l503_503192


namespace problem_l503_503262

def f : ℝ → ℝ :=
λ x, if 0 ≤ x ∧ x < 1 then 2^x else sin (π * x / 4)

theorem problem :
  f 2 + f (3 - real.logb 2 7) = 15 / 7 :=
by
  sorry

end problem_l503_503262


namespace female_athletes_in_sample_l503_503525

theorem female_athletes_in_sample (M F S : ℕ) (hM : M = 56) (hF : F = 42) (hS : S = 28) :
  (F * (S / (M + F))) = 12 :=
by
  rw [hM, hF, hS]
  norm_num
  sorry

end female_athletes_in_sample_l503_503525


namespace polar_to_cartesian_standard_eqs_and_chord_length_l503_503638

theorem polar_to_cartesian_standard_eqs_and_chord_length :
  (∀ t : ℝ, ∃ (x y : ℝ), x = t + 1 ∧ y = t - 2) →
  (∀ θ : ℝ, ∃ (ρ : ℝ), ρ = 4 * Real.cos θ) →
  (∃ (l_eq : ℝ → ℝ → Prop), l_eq = (λ x y, x - y - 3 = 0)) ∧
  (∃ (c_eq : ℝ → ℝ → Prop), c_eq = (λ x y, (x - 2)^2 + y^2 = 4)) ∧
  (∃ chord_length : ℝ, chord_length = Real.sqrt 14) :=
by
  intros line_param polar_param
  sorry

end polar_to_cartesian_standard_eqs_and_chord_length_l503_503638


namespace interval_intersection_l503_503973

theorem interval_intersection :
  {x : ℝ | 1 < 3 * x ∧ 3 * x < 2 ∧ 1 < 5 * x ∧ 5 * x < 2} =
  {x : ℝ | (1 / 3 : ℝ) < x ∧ x < (2 / 5 : ℝ)} :=
by
  -- Need a proof here
  sorry

end interval_intersection_l503_503973


namespace general_terms_sum_first_n_terms_l503_503249

noncomputable theory

open Real

-- Declare the initial conditions
variables (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ)

-- Initial conditions for sequence {a_n}
axiom a1 : a 1 = 1 / 2
axiom a_decreasing : ∀ n ≥ 1, a (n + 1) < a n
axiom a_arith_seq : ∀ n ≥ 1, 3 * a 2 = a 1 + 2 * a 3

-- Initial condition for sequence {b_n}
axiom S_def : ∀ n ∈ ℕ, S n = n^2 + n
axiom b_def : b 1 = 2 ∧ ∀ n ≥ 2, b n = S n - S (n - 1)

-- Defining c and T
def c (n : ℕ) : ℝ := (b (n + 1)) / 2 * log (a n)
def T (n : ℕ) : ℝ := -(∑ i in finset.range n, (1 / c i))

-- The general term formulas
theorem general_terms :
  ∀ n ∈ ℕ, a n = (1 / 2)^n ∧ b n = 2 * n :=
sorry

-- The sum of the first n terms of the sequence {1/c_n}
theorem sum_first_n_terms:
  ∀ n ∈ ℕ, T n = -n / (n + 1) :=
sorry

end general_terms_sum_first_n_terms_l503_503249


namespace sum_of_interior_angles_of_regular_polygon_l503_503789

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : n = 360 / 20) :
  (∑ i in finset.range n, 180 - 360 / n) = 2880 :=
by
  sorry

end sum_of_interior_angles_of_regular_polygon_l503_503789


namespace alice_prank_combinations_l503_503941

theorem alice_prank_combinations : 
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 60 :=
by
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  exact (show 1 * 3 * 5 * 4 * 1 = 60 from sorry)

end alice_prank_combinations_l503_503941


namespace convinced_of_twelve_models_vitya_review_58_offers_l503_503089

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end convinced_of_twelve_models_vitya_review_58_offers_l503_503089


namespace youngest_is_dan_l503_503778

notation "alice" => 21
notation "bob" => 18
notation "clare" => 22
notation "dan" => 16
notation "eve" => 28

theorem youngest_is_dan :
  let a := alice
  let b := bob
  let c := clare
  let d := dan
  let e := eve
  a + b = 39 ∧
  b + c = 40 ∧
  c + d = 38 ∧
  d + e = 44 ∧
  a + b + c + d + e = 105 →
  min (min (min (min a b) c) d) e = d :=
by {
  sorry
}

end youngest_is_dan_l503_503778


namespace sum_of_non_palindromes_taking_eight_steps_is_187_l503_503753

-- Define the range and properties
def is_positive_integer_between_10_and_100 (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 100

def is_non_palindrome (n : ℕ) : Prop :=
  let digits : List ℕ := n.digits 10
  digits ≠ digits.reverse

def takes_eight_steps_to_palindrome (n : ℕ) : Prop := 
  -- The precise definition should reflect the process of checking 
  -- the number of steps needed to reach a palindrome, this is a placeholder:
  sorry 

-- The theorem statement proving the problem
theorem sum_of_non_palindromes_taking_eight_steps_is_187 : 
  ∑ n in {n : ℕ | is_positive_integer_between_10_and_100 n ∧ is_non_palindrome n ∧ takes_eight_steps_to_palindrome n}, n = 187 := 
sorry

end sum_of_non_palindromes_taking_eight_steps_is_187_l503_503753


namespace trigonometric_solution_l503_503980

-- mathematical problem definition in Lean

theorem trigonometric_solution (n : ℕ) (x : ℝ) (m k l : ℤ) :
  (∃ n : ℕ, n > 0) →
  (∀ n x, (∏ i in finset.range n, sin (i * x + x)) + (∏ i in finset.range n, cos (i * x + x)) = 1) →
  (if n = 1 then 
      (∃ m k : ℤ, x = 2 * m * real.pi ∨ x = 2 * k * real.pi + real.pi / 2)
   else if n ≥ 2 then 
     (∃ m l : ℤ, 
       (n = 4 * l - 2 ∨ n = 4 * l + 1 → x = 2 * m * real.pi) ∨ 
       (n = 4 * l ∨ n = 4 * l - 1 → x = m * real.pi)))
:= sorry

end trigonometric_solution_l503_503980


namespace a_n_repeat_every_3_a_2024_val_l503_503957

def reciprocal_difference (a : ℚ) : ℚ := 1 / (1 - a)

def a_seq : ℕ → ℚ
| 0     := -3 -- Given a₁ = -3
| (n+1) := reciprocal_difference (a_seq n)

theorem a_n_repeat_every_3 : ∀ n, a_seq (n + 3) = a_seq n :=
by
  -- since only the statement is required, the proof is skipped
  sorry

theorem a_2024_val : a_seq 2023 = 1 / 4 :=
by 
  -- since only the statement is required, the proof is skipped
  sorry

end a_n_repeat_every_3_a_2024_val_l503_503957


namespace area_of_intersection_of_circles_l503_503847

-- Definition of the two circles
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 ≤ 9
def circle2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 ≤ 9

-- Definition of the intersection area of two circles
def intersection_area : ℝ := 9 * (Real.pi - 2)/2

-- Statement to prove
theorem area_of_intersection_of_circles :
  (∫ x in -Real.pi..Real.pi, ∫ y in -Real.pi..Real.pi, dichotomy (circle1 x y) (circle2 x y)) = intersection_area := 
sorry

end area_of_intersection_of_circles_l503_503847


namespace percent_decrease_l503_503415

variable (c_2010 c_2020 : ℝ)
variable (h2010 : c_2010 = 45)
variable (h2020 : c_2020 = 12)

theorem percent_decrease (h2010 : c_2010 = 45) (h2020 : c_2020 = 12) :
  (c_2010 - c_2020) / c_2010 * 100 ≈ 73 := by
  sorry

end percent_decrease_l503_503415


namespace num_combinations_l503_503705

theorem num_combinations (backpacks pencil_cases : ℕ) (h_backpacks : backpacks = 2) (h_pencil_cases : pencil_cases = 2) : 
  backpacks * pencil_cases = 4 :=
by
  rw [h_backpacks, h_pencil_cases]
  exact Nat.mul_self_eq 2

end num_combinations_l503_503705


namespace find_a_l503_503297

theorem find_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 1) : a = 3 :=
by
  sorry

end find_a_l503_503297


namespace largest_integer_with_unique_prime_pairs_l503_503520

-- Define helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the function to extract two-digit pairs from an integer (represented as a list of digits)
def two_digit_primes (L : List ℕ) : List ℕ :=
  (L.zipWith (λ x y => x * 10 + y) (L.dropLast 1) (L.drop 1)).filter is_prime

-- Define the condition for a number to have all unique two-digit prime pairs
def has_unique_prime_pairs (n : ℕ) : Prop :=
  let digits := n.digits 10
  let primes := two_digit_primes digits
  primes.nodup

-- Define the desired property
theorem largest_integer_with_unique_prime_pairs : ∃ N, has_unique_prime_pairs N ∧ N = 619737131179 := sorry

end largest_integer_with_unique_prime_pairs_l503_503520


namespace median_of_fifteen_is_eight_l503_503480

def median_of_first_fifteen_positive_integers : ℝ :=
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median_pos := (list.length lst + 1) / 2  
  lst.get (median_pos - 1)

theorem median_of_fifteen_is_eight : median_of_first_fifteen_positive_integers = 8.0 := 
  by 
    -- Proof omitted    
    sorry

end median_of_fifteen_is_eight_l503_503480


namespace centroid_expression_l503_503455

-- Define the points P, Q, and R as given.
structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 5 }
def Q : Point := { x := 0, y := -1 }
def R : Point := { x := 7, y := 2 }

-- Define the coordinates of the centroid S of triangle PQR
def S : Point :=
  { x := (P.x + Q.x + R.x) / 3, y := (P.y + Q.y + R.y) / 3 }

-- Prove that the expression 5m + 3n equals 21 when m and n are the coordinates of the centroid of PQR
theorem centroid_expression : (5 * S.x + 3 * S.y = 21) :=
by
  -- Coordinates of S are (3, 2)
  -- Thus the value of 5m + 3n should be 5*3 + 3*2 = 21
  sorry

end centroid_expression_l503_503455


namespace distance_from_company_total_fuel_consumption_total_fare_received_l503_503524

def distances : List Int := [5, 2, -4, -3, 6]

noncomputable def fuel_rate : ℚ := 0.3
noncomputable def fare_base : ℚ := 8
noncomputable def fare_extra_rate : ℚ := 1.6

theorem distance_from_company :
  List.sum distances = 6 := by
  sorry

theorem total_fuel_consumption :
  List.sum (distances.map Int.natAbs) * fuel_rate = 6 := by
  sorry

theorem total_fare_received :
  let fare := distances.map (λ d =>
    if d ≤ 3 then fare_base
    else fare_base + fare_extra_rate * (d - 3)) in
  List.sum fare = 49.6 := by
  sorry

end distance_from_company_total_fuel_consumption_total_fare_received_l503_503524


namespace volunteer_assignment_count_l503_503978

theorem volunteer_assignment_count :
  ∃ (assignments : ℕ), assignments = 240 :=
by
  -- Let n be the number of volunteers
  let n := 5
  -- Let k be the number of events
  let k := 4
  -- We need to prove the number of ways to assign the volunteers to events such that each event gets at least one volunteer is equal to 240
  let number_of_ways := Nat.choose 5 2 * Nat.factorial 4
  have h : number_of_ways = 240 := by sorry
  use number_of_ways
  exact h

end volunteer_assignment_count_l503_503978


namespace fraction_of_crop_to_longest_side_l503_503603

def is_isosceles_trapezoid (a b c d α β : ℝ) : Prop :=
  a = c ∧ b = 200 ∧ d = 100 ∧ α = 45 ∧ β = 135 ∧ 
  α + β = 180 - (45 + 135)

theorem fraction_of_crop_to_longest_side 
  (a b c d α β : ℝ) 
  (h_trapezoid : is_isosceles_trapezoid a b c d α β) :
  (let A := (1/2) * (100 + 200) * (50 * real.sqrt 2),
       A1 := (1/2) * (150 + 200) * (25 * real.sqrt 2) in
      A1 / A) = (7/12) :=
sorry

end fraction_of_crop_to_longest_side_l503_503603


namespace sum_of_positive_integers_n_l503_503585

theorem sum_of_positive_integers_n (n : ℕ) (h1 : ∃ n : ℕ, 91 = nat.greatest_unformable_postage 5 n (n+1)) 
(h2 : ∀ n : ℕ, nat.greatest_unformable_postage 5 n (n+1) > 91 → n = 0):
  n = 24 ∨ n = 47 → 24 + 47 = 71 :=
by 
  sorry

end sum_of_positive_integers_n_l503_503585


namespace find_number_l503_503436

theorem find_number (x : ℝ) (h : (3.242 * 16) / x = 0.051871999999999995) : x = 1000 :=
by
  sorry

end find_number_l503_503436


namespace problem_f_of_f1_l503_503625

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then Real.log 2 (5 - x) else 2^x + 2

theorem problem_f_of_f1 : f (f 1) = 6 :=
by
  sorry

end problem_f_of_f1_l503_503625


namespace find_a_value_l503_503983

theorem find_a_value 
  (a : ℝ) 
  (h : (∀ x : ℂ, x^2 - 8 * x + a = 0 ↔ (x - 4)^2 = 1)) 
  : a = 15 := 
begin
  sorry
end

end find_a_value_l503_503983


namespace smallest_of_seven_consecutive_even_numbers_l503_503772

theorem smallest_of_seven_consecutive_even_numbers (a b c d e f g : ℤ)
  (h₁ : a + b + c + d + e + f + g = 700)
  (h₂ : b = a + 2)
  (h₃ : c = a + 4)
  (h₄ : d = a + 6)
  (h₅ : e = a + 8)
  (h₆ : f = a + 10)
  (h₇ : g = a + 12)
  : a = 94 :=
by
  -- Proof is omitted, this is just the statement.
  sorry

end smallest_of_seven_consecutive_even_numbers_l503_503772


namespace median_of_first_fifteen_integers_l503_503483

theorem median_of_first_fifteen_integers : 
  let L := (list.range 15).map (λ n, n + 1)
  in list.median L = 8.0 :=
by 
  sorry

end median_of_first_fifteen_integers_l503_503483


namespace number_of_incorrect_relations_l503_503936

theorem number_of_incorrect_relations :
  let relations := [1 ⊆ {0, 1, 2}, {1} ∈ {0, 1, 2}, {0, 1, 2} ⊆ {0, 1, 2}, ∅ ⊂ {0}]
  (relations.filter (λ r, ¬ r)).length = 2 :=
by
  sorry

end number_of_incorrect_relations_l503_503936


namespace reflection_combination_l503_503580

-- Define the matrix type
def Matrix2x2 : Type := Matrix (Fin 2) (Fin 2) ℝ

-- Define the reflection matrices
def reflectX : Matrix2x2 := !![1, 0; 0, -1]
def reflectY : Matrix2x2 := !![-1, 0; 0, 1]

-- The target matrix for sequential reflection over x-axis and y-axis
def reflectXY : Matrix2x2 := !![-1, 0; 0, -1]

-- The theorem statement
theorem reflection_combination :
  reflectX ⬝ reflectY = reflectXY :=
by
  sorry

end reflection_combination_l503_503580


namespace computation_square_root_l503_503953

theorem computation_square_root (x : ℕ) (h : x = 30) :
  sqrt ((x + 4) * (x + 2) * (x - 2) * (x - 4) + 1) = 170 := by
  sorry

end computation_square_root_l503_503953


namespace determine_right_triangle_l503_503331

variable (A B C : ℝ)
variable (AB BC AC : ℝ)

-- Conditions as definitions
def condition1 : Prop := A + C = B
def condition2 : Prop := A = 30 ∧ B = 60 ∧ C = 90 -- Since ratio 1:2:3 means A = 30, B = 60, C = 90

-- Proof problem statement
theorem determine_right_triangle (h1 : condition1 A B C) (h2 : condition2 A B C) : (B = 90) :=
sorry

end determine_right_triangle_l503_503331


namespace area_of_smaller_rhombus_l503_503672

theorem area_of_smaller_rhombus {r : ℝ} (radius : r = 10) :
  let d := 2 * r in 
  let area_of_larger_rhombus := (d * d) / 2 in
  let side_of_smaller_rhombus := d / 2 in
  let area_of_smaller_rhombus := side_of_smaller_rhombus ^ 2 in
  area_of_smaller_rhombus = 100 :=
by 
  sorry

end area_of_smaller_rhombus_l503_503672


namespace length_of_train_is_correct_l503_503526

noncomputable def speed_in_m_per_s (speed_in_km_per_hr : ℝ) : ℝ := speed_in_km_per_hr * 1000 / 3600

noncomputable def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

noncomputable def length_of_train (total_distance : ℝ) (length_of_bridge : ℝ) : ℝ := total_distance - length_of_bridge

theorem length_of_train_is_correct :
  ∀ (speed_in_km_per_hr : ℝ) (time_to_cross_bridge : ℝ) (length_of_bridge : ℝ),
  speed_in_km_per_hr = 72 →
  time_to_cross_bridge = 12.199024078073753 →
  length_of_bridge = 134 →
  length_of_train (total_distance (speed_in_m_per_s speed_in_km_per_hr) time_to_cross_bridge) length_of_bridge = 110.98048156147506 :=
by 
  intros speed_in_km_per_hr time_to_cross_bridge length_of_bridge hs ht hl;
  rw [hs, ht, hl];
  sorry

end length_of_train_is_correct_l503_503526


namespace solveForK_l503_503208

theorem solveForK :
  (∃ k : ℚ, (∀ x : ℝ, ((x + 6) * (x + 2) = k + 3 * x) ↔ x^2 + 5 * x + (12 - k) = 0) 
  ∧ k = 23/4) :=
begin
  -- this is where the proof would go
  sorry
end

end solveForK_l503_503208


namespace evaluate_floor_sum_l503_503566

-- Definitions from conditions
def floor (x : Real) : Int := Int.floor x

-- Statement requiring proof
theorem evaluate_floor_sum : (floor 12.7) + (floor (-12.7)) = -1 := by
  sorry

end evaluate_floor_sum_l503_503566


namespace sin_and_tan_alpha_l503_503254

variable (x : ℝ) (α : ℝ)

-- Conditions
def vertex_is_origin : Prop := true
def initial_side_is_non_negative_half_axis : Prop := true
def terminal_side_passes_through_P : Prop := ∃ (P : ℝ × ℝ), P = (x, -Real.sqrt 2)
def cos_alpha_eq : Prop := x ≠ 0 ∧ Real.cos α = (Real.sqrt 3 / 6) * x

-- Proof Problem Statement
theorem sin_and_tan_alpha (h1 : vertex_is_origin) 
                         (h2 : initial_side_is_non_negative_half_axis) 
                         (h3 : terminal_side_passes_through_P x) 
                         (h4 : cos_alpha_eq x α) 
                         : Real.sin α = -Real.sqrt 6 / 6 ∧ (Real.tan α = Real.sqrt 5 / 5 ∨ Real.tan α = -Real.sqrt 5 / 5) := 
sorry

end sin_and_tan_alpha_l503_503254


namespace horner_operations_count_l503_503072

def polynomial (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

def horner_polynomial (x : ℝ) := (((((5*x + 4)*x + 3)*x + 2)*x + 1)*x + 1)

theorem horner_operations_count (x : ℝ) : 
    (polynomial x = horner_polynomial x) → 
    (x = 2) → 
    (mul_ops : ℕ) = 5 → 
    (add_ops : ℕ) = 5 := 
by 
  sorry

end horner_operations_count_l503_503072


namespace geom_seq_a_sum_first_n_terms_l503_503996

noncomputable def a (n : ℕ) : ℕ := 2^(n + 1)

def b (n : ℕ) : ℕ := 3 * (n + 1) - 2

def a_b_product (n : ℕ) : ℕ := (3 * (n + 1) - 2) * 2^(n + 1)

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => a_b_product k)

theorem geom_seq_a (n : ℕ) : a (n + 1) = 2 * a n :=
by sorry

theorem sum_first_n_terms (n : ℕ) : S n = 10 + (3 * n - 5) * 2^(n + 1) :=
by sorry

end geom_seq_a_sum_first_n_terms_l503_503996


namespace even_numbers_average_18_l503_503031

variable (n : ℕ)
variable (avg : ℕ)

theorem even_numbers_average_18 (h : avg = 18) : n = 17 := 
    sorry

end even_numbers_average_18_l503_503031


namespace shaded_to_non_shaded_ratio_l503_503069

-- Define Triangle ABC and its properties
def triangle (A B C : Type) : Prop :=
  ∀ P Q R : A, 
  (side_length P Q = 6 ∧ side_length Q R = 6 ∧ side_length R P = 6)

-- Define midpoints D, E, F
def midpoint (P Q : Type) : Type := sorry
def D := midpoint A B
def E := midpoint B C
def F := midpoint C A

-- Define midpoints G and H
def G := midpoint D F
def H := midpoint F E

-- Define areas
def area (triangle : Type) : ℝ := sorry -- Placeholder for area computation formula
def area_ABC := area (triangle A B C)
def area_DGF := area (triangle D G F)
def area_EHF := area (triangle E H F)

-- Calculate total shaded area
def shaded_area := area_DGF + area_EHF

-- Calculate non-shaded area
def non_shaded_area := area_ABC - shaded_area

-- Define ratio of shaded to non-shaded area
def ratio := shaded_area / non_shaded_area

-- Prove the required ratio
theorem shaded_to_non_shaded_ratio : ratio = 1 / 23 :=
by
  -- Proof steps would go here
  sorry

end shaded_to_non_shaded_ratio_l503_503069


namespace axis_of_symmetry_range_b_plus_c_l503_503213

variables {x A B C : ℝ} {a b c : ℝ}
variables {k : ℤ}
noncomputable def m := (1/2 * Real.sin x, √3/2)
noncomputable def n := (Real.cos x, Real.cos x ^ 2 - 1/2)
noncomputable def f := m.1 * n.1 + m.2 * n.2

-- Equation of the axis of symmetry for f(x)
theorem axis_of_symmetry (k : ℤ) : 
  ∃ k : ℤ, x = 1/2 * k * Real.pi + Real.pi / 12 :=
sorry

-- Range of values for b + c
theorem range_b_plus_c (hA : f A = 0) (ha : a = √3) (h_acute : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) : 
  3 < b + c ∧ b + c ≤ 2 * √3 :=
sorry

end axis_of_symmetry_range_b_plus_c_l503_503213


namespace number_of_large_posters_is_5_l503_503923

theorem number_of_large_posters_is_5
  (total_posters : ℕ)
  (small_posters_ratio : ℚ)
  (medium_posters_ratio : ℚ)
  (h_total : total_posters = 50)
  (h_small_ratio : small_posters_ratio = 2 / 5)
  (h_medium_ratio : medium_posters_ratio = 1 / 2) :
  (total_posters * (1 - small_posters_ratio - medium_posters_ratio)) = 5 :=
by sorry

end number_of_large_posters_is_5_l503_503923


namespace sin_2alpha_eq_2_minus_2_sqrt_2_l503_503237

theorem sin_2alpha_eq_2_minus_2_sqrt_2
  (α t : ℝ)
  (h1 : Polynomial.has_root (Polynomial.C 1 * Polynomial.X^2 - Polynomial.C t * Polynomial.X + Polynomial.C t) (cos α))
  (h2 : Polynomial.has_root (Polynomial.C 1 * Polynomial.X^2 - Polynomial.C t * Polynomial.X + Polynomial.C t) (sin α))
  (h_sum : sin α + cos α = t)
  (h_product : sin α * cos α = t) :
  sin (2 * α) = 2 - 2 * Real.sqrt 2 := by
  sorry

end sin_2alpha_eq_2_minus_2_sqrt_2_l503_503237


namespace interior_angles_sum_l503_503794

theorem interior_angles_sum (h : ∀ (n : ℕ), n = 360 / 20) : 
  180 * (h 18 - 2) = 2880 :=
by
  sorry

end interior_angles_sum_l503_503794


namespace right_triangle_circles_l503_503708

theorem right_triangle_circles {A B C D : Type*} {O1 O2 : Type*}
    (hABC : ∀ {x y z : Type*}, Triangle x y z)
    (r1 r2 r : ℝ)
    (HA : A = Pt) (HB : B = Pt) (HC : C = Pt)
    (HCD_perp_AB : IsPerpendicular CD AB)
    (HangleC : ∠ A B C = 90)
    (HO1_tangent_AD : IsTangent O1 AD)
    (HO1_tangent_DB : IsTangent O1 DB)
    (HO1_tangent_CD : IsTangent O1 CD)
    (HO2_tangent_AD : IsTangent O2 AD)
    (HO2_tangent_DB : IsTangent O2 DB)
    (HO2_tangent_circumcircle : IsTangent O2 (Circumcircle ABC))
    (Hincircle_radii : IsIncircle ABC r)
    (HO1_radius : Radius O1 = r1)
    (HO2_radius : Radius O2 = r2) :
    r1 + r2 = 2 * r := 
sorry

end right_triangle_circles_l503_503708


namespace incenter_distances_l503_503111

variable (a b c d₁ d₂ d₃ : ℝ)

theorem incenter_distances (h : (d₁^2 / (b * c)) + (d₂^2 / (c * a)) + (d₃^2 / (a * b)) = 1) :
  ∃ d₁ d₂ d₃ : ℝ, 
    (d₁^2 / (b * c)) + (d₂^2 / (c * a)) + (d₃^2 / (a * b)) = 1 :=
sory

end incenter_distances_l503_503111


namespace line_curve_common_points_eq_one_l503_503117

def parametric_line (t : ℝ) : ℝ × ℝ :=
  (t, 4 + t)

def polar_curve (θ : ℝ) : ℝ :=
  4 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

theorem line_curve_common_points_eq_one : ∃! (p : ℝ × ℝ), 
  ∃ t θ, parametric_line t = p ∧ polar_curve θ = Real.sqrt (p.1 ^ 2 + p.2 ^ 2) ∧ 
  θ = Real.atan2 p.2 p.1 :=
sorry

end line_curve_common_points_eq_one_l503_503117


namespace unit_digit_b2017_l503_503642

/-- Definitions and conditions setup --/
def sequence_a (n : ℕ) (h : n > 0) : ℕ := Nat.floor (Real.sqrt (5 * n - 1))

def sequence_b : ℕ → ℕ := 
  (λ n : ℕ, 
     if h : n = 2017 then 2 
     else sorry)
 
/-- The theorem statement --/
theorem unit_digit_b2017 : 
  sequence_b 2017 = 2 := sorry

end unit_digit_b2017_l503_503642


namespace Auston_height_in_cm_l503_503160

-- Define the conversions and Auston's height
def foot_to_inches (feet: ℝ): ℝ := feet * 12
def inch_to_cm (inches: ℝ): ℝ := inches * 2.54
def total_inches (feet inches: ℝ): ℝ := foot_to_inches feet + inches
def round_to_nearest_tenth (num : ℝ) : ℝ := (Real.floor (num * 10) + 0.5) / 10

theorem Auston_height_in_cm (feet inches : ℝ) (h: feet = 5) (i: inches = 2) : 
  round_to_nearest_tenth (inch_to_cm (total_inches feet inches)) = 157.5 := 
by
  sorry

end Auston_height_in_cm_l503_503160


namespace max_digit_sum_in_24hr_format_l503_503754

theorem max_digit_sum_in_24hr_format : 
  let hours := [(x, y) | x <- [0, 1, 2], y <- if x = 2 then [0, 1, 2, 3] else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
      minutes := [(x, y) | x <- [0, 1, 2, 3, 4, 5], y <- [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
      sum_digits := λ (t: ℕ × ℕ), t.1 + t.2 
  in max (hours.map sum_digits).maximum + max (minutes.map sum_digits).maximum = 19 :=
sorry

end max_digit_sum_in_24hr_format_l503_503754


namespace exists_positive_n_l503_503723

-- Define the polynomial P(x)
variable (P : Polynomial ℝ)
-- Define the hypotheses: P(x) > 0 for all x ≥ 0
def positive_polynomial (P : Polynomial ℝ) : Prop := ∀ x : ℝ, 0 ≤ x → P.eval x > 0

-- Main theorem statement
theorem exists_positive_n (hP : positive_polynomial P) : ∃ n : ℕ, ∀ x : ℝ, 0 ≤ x → (Polynomial.eval (Polynomial.X + 1 : Polynomial ℝ) x) ^ n * P.eval x ≥ 0 :=
sorry

end exists_positive_n_l503_503723


namespace polynomial_evaluation_l503_503379

def p (x : ℝ) (a b c d : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_evaluation
  (a b c d : ℝ)
  (h1 : p 1 a b c d = 1993)
  (h2 : p 2 a b c d = 3986)
  (h3 : p 3 a b c d = 5979) :
  (1 / 4 : ℝ) * (p 11 a b c d + p (-7) a b c d) = 5233 := by
  sorry

end polynomial_evaluation_l503_503379


namespace find_m_eq_2_l503_503773

variables {ℝ : Type*} [normed_ring ℝ] [is_R_or_C ℝ]
variables (a b c : ℝ → ℝ) (m : ℝ)

theorem find_m_eq_2 (h₁ : ∀ a b c, a + 2 * b + c = 0)
    (h₂ : ∀ a b c, m * (a × c) + 2 * (b × c) + c × a = 0) :
    m = 2 :=
sorry

end find_m_eq_2_l503_503773


namespace eq_solutions_l503_503194

theorem eq_solutions (x : ℝ) :
  (∃ x : ℝ, (real.root 4 (47 - 2 * x) + real.root 4 (35 + 2 * x) = 4) -> (x = 23 ∨ x = -17)) :=
begin
  sorry
end

end eq_solutions_l503_503194


namespace find_linear_function_decreasing_l503_503899

theorem find_linear_function_decreasing (f : ℝ → ℝ) 
  (h1 : ∀ x y, x < y → f(x) > f(y))
  (h2 : ∀ x, f(f(x)) = 4 * x - 1) : 
  f = λ x, -2 * x + 1 :=
by
  sorry

end find_linear_function_decreasing_l503_503899


namespace area_of_intersection_of_circles_l503_503846

-- Definition of the two circles
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 ≤ 9
def circle2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 ≤ 9

-- Definition of the intersection area of two circles
def intersection_area : ℝ := 9 * (Real.pi - 2)/2

-- Statement to prove
theorem area_of_intersection_of_circles :
  (∫ x in -Real.pi..Real.pi, ∫ y in -Real.pi..Real.pi, dichotomy (circle1 x y) (circle2 x y)) = intersection_area := 
sorry

end area_of_intersection_of_circles_l503_503846


namespace points_form_line_l503_503684

-- Define the points (x, y) based on the function (cos^4 t, sin^4 t) for real numbers t
def point (t : ℝ) : ℝ × ℝ := (Real.cos t ^ 4, Real.sin t ^ 4)

-- The theorem statement to prove the set of points forms a line
theorem points_form_line (t : ℝ) : 
  let (x, y) := point t 
  in x + y + 2 * Real.sqrt (x * y) = 1 :=
by
  let ⟨x, y⟩ := point t
  sorry

end points_form_line_l503_503684


namespace tangent_line_circle_l503_503558

theorem tangent_line_circle {a b : ℝ} : 
  let circle_eq := λ (x y : ℝ), x^2 + y^2 + a*x + b*y = 0 
  let line_eq := λ (x y : ℝ), a*x + b*y = 0 
  (∀ x y, line_eq x y → circle_eq x y → (x+ (a/2))^2 + (y + (b/2))^2 = (a^2 + b^2) / 4) →
  (a ≠ 0 ∨ b ≠ 0) →
  let center := (-(a/2), -(b/2))
  let radius := (real.sqrt (a^2 + b^2)) / 2
  let distance := (a^2 + b^2) / (real.sqrt (a^2 + b^2))
  distance = radius :=
sorry

end tangent_line_circle_l503_503558


namespace polynomial_max_value_greater_than_inverse_exp_n_l503_503725

noncomputable def e : ℝ := Real.exp 1

theorem polynomial_max_value_greater_than_inverse_exp_n
  (n : ℕ) (hn : 0 < n)
  (p : ℝ[X]) (hp : p.degree = n ∧ ∀ (coeff : ℕ), p.coeff coeff ∈ Int) :
  ∃ x ∈ Icc (0 : ℝ) 1, |p.eval x| > 1 / e ^ n := sorry

end polynomial_max_value_greater_than_inverse_exp_n_l503_503725


namespace set_representation_equiv_l503_503159

open Nat

theorem set_representation_equiv :
  {x : ℕ | (0 < x) ∧ (x - 3 < 2)} = {1, 2, 3, 4} :=
by
  sorry

end set_representation_equiv_l503_503159


namespace sum_g_reciprocal_l503_503372

def g (n : ℕ) : ℝ :=
  let m := Real.floor (Real.cbrt n) + 1
  if (m - 0.5) < Real.cbrt n ∧ Real.cbrt n < (m + 0.5) then m else m - 1

theorem sum_g_reciprocal :
  (∑ k in Finset.range 2500, 1 / g k) = 700.875 := sorry

end sum_g_reciprocal_l503_503372


namespace find_c_and_d_l503_503300

theorem find_c_and_d :
  ∀ (y c d : ℝ), (y^2 - 5 * y + 5 / y + 1 / (y^2) = 17) ∧ (y = c - Real.sqrt d) ∧ (0 < c) ∧ (0 < d) → (c + d = 106) :=
by
  intros y c d h
  sorry

end find_c_and_d_l503_503300


namespace inverse_of_g_l503_503782

noncomputable def u (x : ℝ) : ℝ := sorry
noncomputable def v (x : ℝ) : ℝ := sorry
noncomputable def w (x : ℝ) : ℝ := sorry

noncomputable def u_inv (x : ℝ) : ℝ := sorry
noncomputable def v_inv (x : ℝ) : ℝ := sorry
noncomputable def w_inv (x : ℝ) : ℝ := sorry

lemma u_inverse : ∀ x, u_inv (u x) = x ∧ u (u_inv x) = x := sorry
lemma v_inverse : ∀ x, v_inv (v x) = x ∧ v (v_inv x) = x := sorry
lemma w_inverse : ∀ x, w_inv (w x) = x ∧ w (w_inv x) = x := sorry

noncomputable def g (x : ℝ) : ℝ := v (u (w x))

noncomputable def g_inv (x : ℝ) : ℝ := w_inv (u_inv (v_inv x))

theorem inverse_of_g :
  ∀ x : ℝ, g_inv (g x) = x ∧ g (g_inv x) = x :=
by
  intro x
  -- proof omitted
  sorry

end inverse_of_g_l503_503782


namespace total_surface_area_correct_l503_503906

noncomputable def total_surface_area_of_cylinder (radius height : ℝ) : ℝ :=
  let lateral_surface_area := 2 * Real.pi * radius * height
  let top_and_bottom_area := 2 * Real.pi * radius^2
  lateral_surface_area + top_and_bottom_area

theorem total_surface_area_correct : total_surface_area_of_cylinder 3 10 = 78 * Real.pi :=
by
  sorry

end total_surface_area_correct_l503_503906


namespace smallest_number_conditions_l503_503863

theorem smallest_number_conditions :
  ∃ b : ℕ, 
    (b % 3 = 2) ∧ 
    (b % 4 = 2) ∧
    (b % 5 = 3) ∧
    (∀ b' : ℕ, 
      (b' % 3 = 2) ∧ 
      (b' % 4 = 2) ∧
      (b' % 5 = 3) → b ≤ b') :=
begin
  use 38,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros b' hb',
    have h3 := (hb'.left),
    have h4 := (hb'.right.left),
    have h5 := (hb'.right.right),
    -- The raw machinery for showing that 38 is the smallest may require more definition
    sorry
  }
end

end smallest_number_conditions_l503_503863


namespace trig_identity_condition_l503_503681

open Real

theorem trig_identity_condition (a : Real) (h : ∃ x ≥ 0, (tan a = -1 ∧ cos a ≠ 0)) :
  (sin a / sqrt (1 - sin a ^ 2) + sqrt (1 - cos a ^ 2) / cos a) = 0 :=
by
  sorry

end trig_identity_condition_l503_503681


namespace magnitude_vector_sub_l503_503646

def vector3 := (ℝ × ℝ × ℝ)

def a : vector3 := (2, -3, 5)
def b : vector3 := (-3, 1, -4)

noncomputable def vector_magnitude (v : vector3) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def v_sub_scalar_mul (v1 v2 : vector3) (k : ℝ) : vector3 :=
  (v1.1 - k * v2.1, v1.2 - k * v2.2, v1.3 - k * v2.3)

theorem magnitude_vector_sub : 
  vector_magnitude (v_sub_scalar_mul a b 2) = real.sqrt 258 := by
  sorry

end magnitude_vector_sub_l503_503646


namespace problem1_problem2_problem3_l503_503947

theorem problem1 : 2013^2 - 2012 * 2014 = 1 := 
by 
  sorry

variables (m n : ℤ)

theorem problem2 : ((m-n)^6 / (n-m)^4) * (m-n)^3 = (m-n)^5 :=
by 
  sorry

variables (a b c : ℤ)

theorem problem3 : (a - 2*b + 3*c) * (a - 2*b - 3*c) = a^2 - 4*a*b + 4*b^2 - 9*c^2 :=
by 
  sorry

end problem1_problem2_problem3_l503_503947


namespace consecutive_even_numbers_divisible_by_384_l503_503434

theorem consecutive_even_numbers_divisible_by_384 (n : Nat) (h1 : n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) = 384) : n = 6 :=
sorry

end consecutive_even_numbers_divisible_by_384_l503_503434


namespace initial_people_count_l503_503894

-- Define the conditions
def days_to_complete_total := 50
def days_after := 25
def work_completed_partial := 0.4
def additional_people := 60

-- Define the initial number of people as P and total number of people as T
variable (P : ℕ)
variable (T : ℕ)

-- Define the equations based on conditions
def work_time_initial := (days_after / work_completed_partial)
def total_people_needed (P : ℕ) := P + additional_people
def proportion_relation (P : ℕ) (T : ℕ) := P * work_time_initial = T * days_to_complete_total

-- State the main theorem to prove
theorem initial_people_count :
  ∃ (P : ℕ),  (P * work_time_initial = (P + additional_people) * days_to_complete_total) ∧ P = 240 := 
sorry

end initial_people_count_l503_503894


namespace AB_less_than_AC_l503_503381

theorem AB_less_than_AC (AB AC BD CD: ℝ) (ABCD_convex: Convex ABCD) (h: AB + BD ≤ AC + CD) : AB < AC := 
by 
  sorry

end AB_less_than_AC_l503_503381


namespace AM_eq_2_OF_l503_503376

-- Define the points and conditions for the acute-angled triangle, orthocenter, circumcenter, and midpoint
variables {A B C : Type*}
variables [linear_ordered_field ABC]
variables {M O F : ABC}
variables (triangle_ABC : is_acute_triangle A B C)
variables (orthocenter_M : is_orthocenter M A B C)
variables (circumcenter_O : is_circumcenter O A B C)
variables (midpoint_F : is_midpoint F B C)

-- Define the theorem to prove the stated relationship
theorem AM_eq_2_OF
  (h_triangle : triangle_ABC)
  (h_orthocenter : orthocenter_M)
  (h_circumcenter : circumcenter_O)
  (h_midpoint : midpoint_F) :
  dist A M = 2 * dist O F :=
sorry

end AM_eq_2_OF_l503_503376


namespace female_athletes_selected_l503_503144

/-!
# Number of Female Athletes Selected
Given:
- 56 male athletes.
- 42 female athletes.
- Using stratified sampling, 8 male athletes are selected.
Prove that the number of female athletes selected is 6.
-/

theorem female_athletes_selected :
  ∀ (male_total female_total male_selected : ℕ),
    male_total = 56 →
    female_total = 42 →
    male_selected = 8 →
    ∃ female_selected : ℕ, 
      (female_selected : ℕ) * male_total = (male_selected : ℕ) * female_total ∧ 
      female_selected = 6 :=
by
  intros male_total female_total male_selected h1 h2 h3
  use 6
  split
  sorry
  refl

end female_athletes_selected_l503_503144


namespace magician_trick_succeeds_l503_503900

theorem magician_trick_succeeds :
  ∀ (coins : list ℕ) (k : ℕ), 
    (coins.length = 2) →
    (∀ (x ∈ coins), x ∈ list.range 12) →
    (∀ (y : ℕ), (y ∉ coins) → y = k) →
    (∃ (a b : ℕ), a ≠ b ∧ a ∈ [k+1 % 12, k+2 % 12, k+5 % 12, k+7 % 12] ∧ b ∈ [k+1 % 12, k+2 % 12, k+5 % 12, k+7 % 12] ∧ a ∈ coins ∧ b ∈ coins) :=
by {
  sorry
}

end magician_trick_succeeds_l503_503900


namespace number_of_triangles_in_regular_decagon_l503_503660

noncomputable def number_of_triangles_in_decagon : ℕ :=
∑ i in (finset.range 10).powerset_len 3, 1

theorem number_of_triangles_in_regular_decagon :
  number_of_triangles_in_decagon = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l503_503660


namespace solution_set_of_inequality_l503_503212

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 3 * x else -x^2 + 3 * x

theorem solution_set_of_inequality :
  {x : ℝ | f(x - 2) + f(x^2 - 4) < 0} = set.Ioo (-3 : ℝ) (2 : ℝ) :=
sorry

end solution_set_of_inequality_l503_503212


namespace unique_k_for_triangle_inequality_l503_503572

theorem unique_k_for_triangle_inequality (k : ℕ) (h : 0 < k) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a * b + b * b + c * c) → a + b > c ∧ b + c > a ∧ c + a > b) ↔ (k = 6) :=
by
  sorry

end unique_k_for_triangle_inequality_l503_503572


namespace a_values_unique_solution_l503_503645

theorem a_values_unique_solution :
  (∀ a : ℝ, ∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) →
  (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end a_values_unique_solution_l503_503645


namespace average_speed_l503_503055

theorem average_speed (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 120) (h2 : d2 = 60) (h3 : t = 2) :
  (d1 + d2) / t = 90 := 
by
  simp [h1, h2, h3]
  sorry

end average_speed_l503_503055


namespace fraction_evaluation_l503_503494

theorem fraction_evaluation :
  (1 * 2 * 3 * 4) / (1 + 2 + 3 + 6) = 2 :=
by
  calc
    (1 * 2 * 3 * 4 : ℚ) / (1 + 2 + 3 + 6 : ℚ) = 24 / 12 : by norm_num
    ... = 2 : by norm_num

end fraction_evaluation_l503_503494


namespace parabola_distance_l503_503398

noncomputable def focus := (0, 1) -- Focus of the parabola x^2 = 4y is at (0,1)

theorem parabola_distance 
  (P : ℝ × ℝ) 
  (hP : ∃ x, P = (x, (x^2)/4)) 
  (h_dist_focus : dist P focus = 8) : 
  abs (P.snd) = 7 := sorry

end parabola_distance_l503_503398


namespace perpendicular_vectors_exist_circle_l503_503373

variables {R : Type*} [LinearOrderedField R]

noncomputable def trajectory (m x y : R) :=
  m * x ^ 2 + y ^ 2 = 1

noncomputable def circle (x y : R) :=
  x ^ 2 + y ^ 2 = 4 / 5

theorem perpendicular_vectors (m x y : R) (hm : trajectory m x y) :
  m * x ^ 2 + y ^ 2 = 1 :=
begin
  exact hm,
end

theorem exist_circle (x y : R) (h : trajectory (1 / 4) x y) :
  ∃! C, (∀ t, (circle).intersect (linear t)) ∧ ( ∀ v w, (x₁, y₁) = C.v ∧ (x₂, y₂) = C.w → perp (0: (R, R)) ∧ OA = |⟨x₁ y₁⟩ - ⟨0 0⟩| ∧ OB = |⟨x₂ y₂⟩ - ⟨0 0⟩| ) :=
begin
  sorry,
end

#align perpendicular_vectors perpendicular_vectors
#align exist_circle exist_circle

end perpendicular_vectors_exist_circle_l503_503373


namespace square_area_l503_503500

theorem square_area (x y : ℝ) 
  (h1 : x = 20 ∧ y = 20)
  (h2 : x = 20 ∧ y = 5)
  (h3 : x = x ∧ y = 5)
  (h4 : x = x ∧ y = 20)
  : (∃ a : ℝ, a = 225) :=
sorry

end square_area_l503_503500


namespace lava_lamps_probability_l503_503408

noncomputable def lampProbability : ℚ :=
  let total_arrangements := (Nat.choose 8 4) * (Nat.choose 8 4)
  let favorable_arrangements := (Nat.choose 6 3) * (Nat.choose 6 3)
  favorable_arrangements / total_arrangements

theorem lava_lamps_probability : lampProbability = 4 / 49 := by
  sorry

end lava_lamps_probability_l503_503408


namespace MN_vector_l503_503697

variable (a b : ℝ)
variable (A B C D N M : ℝ) -- Representing points in ℝ for simplicity
variable (AB AD AN MN AM NC : ℝ → ℝ) -- Mapping points to their vector representations

-- Conditions as hypotheses
def parallelogram : Prop := true
def AB_eq_a : AB = a := sorry
def AD_eq_b : AD = b := sorry
def AN_eq_3NC : AN = 3 * NC := sorry
def M_midpoint_BC : M = (B + C) / 2 := sorry

-- Prove that MN = -1/4 * a + 1/4 * b
theorem MN_vector :
  parallelogram →
  AB_eq_a →
  AD_eq_b →
  AN_eq_3NC →
  M_midpoint_BC →
  MN = -1/4 * a + 1/4 * b := sorry

end MN_vector_l503_503697


namespace solution_is_unique_l503_503600

noncomputable def solution (f : ℝ → ℝ) (α : ℝ) :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

theorem solution_is_unique (f : ℝ → ℝ) (α : ℝ)
  (h : solution f α) :
  f = id ∧ α = -1 :=
sorry

end solution_is_unique_l503_503600


namespace inequality_ab5_bc5_ca5_l503_503403

theorem inequality_ab5_bc5_ca5 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b^5 + b * c^5 + c * a^5 ≥ a * b * c * (a^2 * b + b^2 * c + c^2 * a) :=
sorry

end inequality_ab5_bc5_ca5_l503_503403


namespace equilibrium_ladder_mu_l503_503147

theorem equilibrium_ladder_mu :
  ∃ (μ : ℝ), (0.678 - μ).abs < 0.001 :=
by
  sorry

end equilibrium_ladder_mu_l503_503147


namespace minimum_sum_abc_l503_503803

theorem minimum_sum_abc (a b c : ℕ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (h : a * b * c = 2310) (prime_factors_c : ∃ p q : ℕ, prime p ∧ prime q ∧ p ≠ q ∧ c = p * q) : 
  a + b + c ≥ 88 :=
sorry

end minimum_sum_abc_l503_503803


namespace sum_of_interior_angles_l503_503787

theorem sum_of_interior_angles (h : ∀ (n : ℕ), 360 / 20 = n) : 
  ∃ (s : ℕ), s = 2880 :=
by
  have n := 360 / 20
  have sum := 180 * (n - 2)
  use sum
  sorry

end sum_of_interior_angles_l503_503787


namespace vitya_convinced_of_12_models_l503_503079

noncomputable def min_offers_needed (n : ℕ) (k : ℕ) : ℕ :=
  if h : n = 13 then
    let ln100 := Real.log 100
    let ln13 := Real.log 13
    let ln12 := Real.log 12
    let req_k := Real.log 100 / (Real.log 13 - Real.log 12)
    if k > req_k then k else req_k.toNat + 1
  else k

theorem vitya_convinced_of_12_models (k : ℕ) : ∀ n, (n >= 13) → (min_offers_needed n k > 58) :=
by
  intros n h
  apply sorry

end vitya_convinced_of_12_models_l503_503079


namespace exist_five_connected_points_splitting_edges_l503_503542

def graph (V : Type) := V × (V → V → Prop)

def connected {V : Type} (g : graph V) : Prop :=
  ∀ v w : V, v ≠ w → ∃ p : list V, 
    (∀ i ∈ p, i ∈ V) ∧ 
    (head p = v) ∧ 
    (last p = some w) ∧ 
    (∀ (a b : V), (a, b) ∈ p.zip (p.tail) → g.2 a b)

def is_bridge {V : Type} (g : graph V) (e : V × V) : Prop :=
  connected g ∧
  ¬ connected (V, λ u v, g.2 u v ∧ (u ≠ e.1 ∨ v ≠ e.2)) 

theorem exist_five_connected_points_splitting_edges :
  ∃ (V : Type) (g : graph V),
    (is_bridge g) ∧ (∃ (v : list V), length v = 5) :=
sorry

end exist_five_connected_points_splitting_edges_l503_503542


namespace cone_surface_area_eq_3pi_l503_503310

noncomputable def coneSurfaceArea (r l : ℝ) : ℝ :=
  let lateralSurface := π * r * l
  let baseSurface := π * r^2
  baseSurface + lateralSurface

theorem cone_surface_area_eq_3pi (r : ℝ) (h : r = 1) : coneSurfaceArea r 2 = 3 * π := by
  sorry

end cone_surface_area_eq_3pi_l503_503310


namespace rectangle_diagonal_bisect_l503_503327

theorem rectangle_diagonal_bisect
  (A B C D O : Type)
  (AB AD : ℝ)
  (h1 : AB = 4)
  (h2 : AD = 3)
  (h3 : ∀ P : Type, (P = A ∨ P = B ∨ P = C ∨ P = D) ∧
        (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) ∧
        (AB = 4 ∧ AD = 3) ∧ -- sides are defined
        -- define properties of rectangle
         (∀ (a b c : Type), a ≠ b → a ≠ c → a ⋀ c ≠ b ⋀ c) ∧
        (∀ (angle : ℕ), angle = 90) ∧ -- right angle properties
        ∃ O, O = midpoint A C ∧ -- O is the midpoint of diagonal AC
        (AC = sqrt (AD ^ 2 + AB ^ 2))
  )
  : OA = 2.5 := 
begin
  sorry
end

end rectangle_diagonal_bisect_l503_503327


namespace fruit_difference_l503_503749

noncomputable def apples : ℕ := 60
noncomputable def peaches : ℕ := 3 * apples

theorem fruit_difference : peaches - apples = 120 :=
by
  have h1 : apples = 60 := rfl
  have h2 : peaches = 3 * apples := rfl
  calc
    peaches - apples = 3 * apples - apples : by rw [h2]
                ... = 3 * 60 - 60        : by rw [h1]
                ... = 180 - 60           : by norm_num
                ... = 120                : by norm_num

end fruit_difference_l503_503749


namespace trig_value_ordering_l503_503952

theorem trig_value_ordering : 
  let a := Real.sin (17 * Real.pi / 12)
  let b := Real.cos (4 * Real.pi / 9)
  let c := Real.tan (7 * Real.pi / 4)
  c < a ∧ a < b := 
by 
  -- Definitions based on given conditions
  let a := Real.sin (17 * Real.pi / 12)
  let b := Real.cos (4 * Real.pi / 9)
  let c := Real.tan (7 * Real.pi / 4)
  -- The proof will go here
  sorry

end trig_value_ordering_l503_503952


namespace math_problem_l503_503215

theorem math_problem (c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ) (φ: ℝ → ℝ) (λ: ℝ) :
  (f(x) = x^2 + c) ∧ (∀ x, f[f(x)] = f(x^2 + 1)) ∧ 
  (λ = 4) → 
  (g(x) = x^4 + 2x^2 + 2) ∧
  (λ = 4) ∧
  (∀ x < -1, deriv φ x < 0) ∧ (∀ x, -1 < x ∧ x < 0 → deriv φ x > 0)
:= by sorry

end math_problem_l503_503215


namespace number_divisible_by_itself_l503_503137

theorem number_divisible_by_itself (a b : ℕ) (d : ℕ) (number : ℕ) 
  (ha : a = 761) (hb : b = 829) (hd : d = 3) 
  (hnumber: number = 763829): 
  number % number = 0 := 
by {
  rw hnumber,
  exact nat.mod_self number
}

end number_divisible_by_itself_l503_503137


namespace smallest_AAB_value_exists_l503_503930

def is_consecutive_digits (A B : ℕ) : Prop :=
  (B = A + 1 ∨ A = B + 1) ∧ 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9

def two_digit_to_int (A B : ℕ) : ℕ :=
  10 * A + B

def three_digit_to_int (A B : ℕ) : ℕ :=
  110 * A + B

theorem smallest_AAB_value_exists :
  ∃ (A B: ℕ), is_consecutive_digits A B ∧ two_digit_to_int A B = (1 / 7 : ℝ) * ↑(three_digit_to_int A B) ∧ three_digit_to_int A B = 889 :=
sorry

end smallest_AAB_value_exists_l503_503930


namespace Vitya_needs_58_offers_l503_503094

theorem Vitya_needs_58_offers :
  ∃ k : ℕ, (log 0.01 / log (12 / 13) < k) ∧ k = 58 :=
by
  sorry

end Vitya_needs_58_offers_l503_503094


namespace backpack_pencil_case_combinations_l503_503703

theorem backpack_pencil_case_combinations (backpacks pencil_cases : Fin 2) : 
  (backpacks * pencil_cases) = 4 :=
by 
  sorry

end backpack_pencil_case_combinations_l503_503703


namespace find_number_l503_503902

theorem find_number : ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 51 ∧ x = 50 :=
by
  sorry

end find_number_l503_503902


namespace number_of_zeros_of_abs_f_minus_one_l503_503263

def f (x : ℝ) (k : ℝ) : ℝ :=
  if x > 0 then Real.log x else k * x + 2

theorem number_of_zeros_of_abs_f_minus_one (k : ℝ) (hk : k > 0) : 
  ∃ xs : Finset ℝ, (∀ x ∈ xs, |f x k| - 1 = 0) ∧ xs.card = 4 :=
by
  sorry

end number_of_zeros_of_abs_f_minus_one_l503_503263


namespace function_properties_l503_503250

def is_periodic_of_min (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x ∧ (∀ t > 0, (∀ y, f (y + t) = f y) → t ≥ T)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f x > f y

theorem function_properties (f : ℝ → ℝ) :
  is_periodic_of_min f π →
  (∀ x, f (x - π / 4) + f (-x) = 0) →
  is_decreasing_on f (π / 4) (π / 2) →
  f = (λ x, sin (2 * x) + cos (2 * x)) :=
sorry

end function_properties_l503_503250


namespace friends_pay_equally_in_usd_after_discount_l503_503119

theorem friends_pay_equally_in_usd_after_discount:
  ∀ (num_friends : ℕ) (total_bill_eur discount_percentage conversion_rate : ℝ) (correct_amount_usd : ℝ),
  num_friends = 10 →
  total_bill_eur = 150 →
  discount_percentage = 0.12 →
  conversion_rate = 1.12 →
  correct_amount_usd = 14.78 →
  (total_bill_eur * (1 - discount_percentage) / num_friends * conversion_rate ≈ correct_amount_usd) :=
begin
  intros num_friends total_bill_eur discount_percentage conversion_rate correct_amount_usd,
  assume h_num_friends h_total_bill h_discount_percentage h_conversion_rate h_correct_amount,
  sorry,
end

end friends_pay_equally_in_usd_after_discount_l503_503119


namespace solve_for_y_l503_503411

theorem solve_for_y : ∀ y : ℚ, (8 * y^2 + 78 * y + 5) / (2 * y + 19) = 4 * y + 2 → y = -16.5 :=
by
  intro y
  intro h
  sorry

end solve_for_y_l503_503411


namespace median_of_fifteen_is_eight_l503_503481

def median_of_first_fifteen_positive_integers : ℝ :=
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median_pos := (list.length lst + 1) / 2  
  lst.get (median_pos - 1)

theorem median_of_fifteen_is_eight : median_of_first_fifteen_positive_integers = 8.0 := 
  by 
    -- Proof omitted    
    sorry

end median_of_fifteen_is_eight_l503_503481


namespace median_first_fifteen_integers_l503_503471

theorem median_first_fifteen_integers :
  let l := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] in
  let seventh := l.nth 6 in
  let eighth := l.nth 7 in
  (seventh.is_some ∧ eighth.is_some) →
  (seventh.get_or_else 0 + eighth.get_or_else 0) / 2 = 7.5 :=
by
  sorry

end median_first_fifteen_integers_l503_503471


namespace area_of_trapezoid_EFGH_l503_503859

/-- The coordinates of the vertices of the trapezoid EFGH. --/
structure Point where
  x : ℝ
  y : ℝ

def E : Point := ⟨0, 0⟩
def F : Point := ⟨0, -3⟩
def G : Point := ⟨5, 0⟩
def H : Point := ⟨5, 8⟩

/-- Calculate the vertical and horizontal distances between two points. --/
def vertical_distance (P Q : Point) : ℝ := abs (P.y - Q.y)
def horizontal_distance (P Q : Point) : ℝ := abs (P.x - Q.x)

/-- Calculate the area of the trapezoid given the lengths of the two bases and the height. --/
def trapezoid_area (base1 base2 height : ℝ) : ℝ :=
  (base1 + base2) / 2 * height

theorem area_of_trapezoid_EFGH : trapezoid_area (vertical_distance E F) (vertical_distance G H) (horizontal_distance E G) = 27.5 := 
by sorry

end area_of_trapezoid_EFGH_l503_503859


namespace convinced_of_twelve_models_vitya_review_58_offers_l503_503090

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end convinced_of_twelve_models_vitya_review_58_offers_l503_503090


namespace sum_of_fourth_powers_l503_503057

theorem sum_of_fourth_powers (n : ℤ) (h1 : n > 0) (h2 : (n - 1)^2 + n^2 + (n + 1)^2 = 9458) :
  (n - 1)^4 + n^4 + (n + 1)^4 = 30212622 :=
by sorry

end sum_of_fourth_powers_l503_503057


namespace winning_strategy_first_player_l503_503954

-- Define the problem parameters
def game_condition (n i : ℕ) := i ≠ 0 ∧ i ≤ 3

-- Define the game state
def game_state (n : ℕ) :=  ∀ i ∈ (finset.range n), ∃ k_i : ℕ, i + 1 = k_i

-- Define the main theorem using the conditions identified
theorem winning_strategy_first_player (n : ℕ) :
  game_condition n ∧ game_state n → (n % 4 = 1 ∨ n % 4 = 2) :=
by
  intros
  sorry

end winning_strategy_first_player_l503_503954


namespace second_number_desc_diff_fourth_eighth_desc_l503_503985

noncomputable def digit_cards : List ℕ := [0, 1, 2, 2]

def valid_three_digit_numbers : List ℕ := 
  let nums := List.permutations digit_cards 
  List.filter (λ num, num.length = 3 ∧ num.head ≠ 0) nums

def descending_numbers : List ℕ := List.sort (λ a b, b < a) valid_three_digit_numbers

theorem second_number_desc : descending_numbers.nth 1 = some 220 := sorry

theorem diff_fourth_eighth_desc : descending_numbers.nth 3 - descending_numbers.nth 7 = some 90 := sorry

end second_number_desc_diff_fourth_eighth_desc_l503_503985


namespace convinced_of_twelve_models_vitya_review_58_offers_l503_503087

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end convinced_of_twelve_models_vitya_review_58_offers_l503_503087


namespace flowchart_makes_algorithms_intuitive_l503_503873

def flowchart_is_intuitive : Prop :=
  "Flowcharts make the expression of algorithms and their steps more intuitive"

theorem flowchart_makes_algorithms_intuitive : flowchart_is_intuitive :=
by
  sorry

end flowchart_makes_algorithms_intuitive_l503_503873


namespace hexagon_diagonal_length_l503_503201

theorem hexagon_diagonal_length (s : ℝ) (h : s = 8) : 
  let a := 8 in 
  let d := a * sqrt 3 in 
  d = 8 * sqrt 3 :=
by 
  sorry

end hexagon_diagonal_length_l503_503201


namespace profit_per_meter_is_correct_l503_503145

def meters_sold : ℕ := 80
def selling_price : ℝ := 10000
def cost_price_per_meter : ℝ := 118
def total_cost_price : ℝ := cost_price_per_meter * meters_sold
def total_profit : ℝ := selling_price - total_cost_price
def profit_per_meter : ℝ := total_profit / meters_sold

theorem profit_per_meter_is_correct : profit_per_meter = 7 := 
by
  -- Proof omitted
  sorry

end profit_per_meter_is_correct_l503_503145


namespace general_term_formula_sum_of_terms_l503_503998

noncomputable def a_seq : ℕ → ℕ 
| 1       => 1
| (n + 1) => 2 * a_seq n

def S (n : ℕ) : ℕ := 2 * a_seq n - 1

theorem general_term_formula (n : ℕ) (h : n ≠ 0) : a_seq n = 2^(n - 1) := 
by 
  sorry

noncomputable def na_seq (n : ℕ) : ℕ := (n + 1) * a_seq (n + 1)

noncomputable def T (n : ℕ) : ℕ := (finset.range n).sum na_seq

theorem sum_of_terms (n : ℕ) : T n = (n-1) * 2^n + 1 := 
by 
  sorry

end general_term_formula_sum_of_terms_l503_503998


namespace trapezoid_area_efgh_l503_503856

theorem trapezoid_area_efgh :
  let E := (0, 0)
  let F := (0, -3)
  let G := (5, 0)
  let H := (5, 8)
  ∃ (area : ℝ), 
    area = (1 / 2) * ((3 : ℝ) + 8) * 5 ∧ 
    area = 27.5 :=
by
  let E := (0 : ℝ, 0 : ℝ)
  let F := (0 : ℝ, -3 : ℝ)
  let G := (5 : ℝ, 0 : ℝ)
  let H := (5 : ℝ, 8 : ℝ)
  use (1 / 2) * (3 + 8) * 5
  split
  · exact rfl
  · norm_num
  · sorry

end trapezoid_area_efgh_l503_503856


namespace first_digit_base_9_of_y_l503_503032

def base_3_to_base_10 (digits : List ℕ) : ℕ :=
  digits.reverse.foldl (λ (num pair) → num + pair.1 * 3^pair.2) 0 (List.enumFrom 0 digits.length)

def first_digit_base_9 (n : ℕ) : ℕ :=
  let digits := n.toDigits 9
  digits.headD 0

theorem first_digit_base_9_of_y :
  let y := base_3_to_base_10 [2, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 1]
  first_digit_base_9 y = 3 :=
by
  let y := base_3_to_base_10 [2, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 1]
  have : first_digit_base_9 y = 3
  sorry

end first_digit_base_9_of_y_l503_503032


namespace jessica_final_balance_l503_503501

variable (B : ℚ) -- Original balance
variable (withdrawal : ℚ) -- Amount withdrawn, $200
variable (decrease_fraction : ℚ) -- Decrease fraction, 2/5
variable (remaining_fraction : ℚ) -- Fraction of remaining balance deposited, 1/2

-- Define the problem conditions
variable (condition1 : withdrawal = 200)
variable (condition2 : decrease_fraction = (2 / 5 : ℚ))
variable (condition3 : remaining_fraction = (1 / 2 : ℚ))
variable (balance_after_withdrawal : ℚ) (balance_after_deposit : ℚ)

-- Define the remaining balance and final balance based on conditions
def original_balance := B = (withdrawal / decrease_fraction)
def remaining_balance := balance_after_withdrawal = (B - withdrawal)
def deposit := (remaining_fraction * balance_after_withdrawal)
def final_balance := (balance_after_withdrawal + deposit)

theorem jessica_final_balance :
  original_balance B withdrawal decrease_fraction → 
  remaining_balance balance_after_withdrawal B withdrawal → 
  final_balance balance_after_deposit balance_after_withdrawal remaining_fraction → 
  balance_after_deposit = 450 := by
  intros
  sorry

end jessica_final_balance_l503_503501


namespace sum_smallest_largest_primes_between_30_and_60_l503_503746

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b : ℕ) : set ℕ := {p | p ≥ a ∧ p ≤ b ∧ is_prime p}

theorem sum_smallest_largest_primes_between_30_and_60 :
  let S := primes_in_range 30 60 in (∀ p ∈ S, is_prime p) ∧ (∃ x y, x ∈ S ∧ y ∈ S ∧ x = 31 ∧ y = 59 ∧ x + y = 90) :=
begin
  let S := primes_in_range 30 60,
  have h1 : ∀ p ∈ S, is_prime p, by { intro p, intro hp, cases hp with hpa hpr, exact hpr },
  have h2 : ∃ x y, x ∈ S ∧ y ∈ S ∧ x = 31 ∧ y = 59 ∧ x + y = 90, by { sorry }, -- detailed proof omitted
  exact ⟨h1, h2⟩,
end

end sum_smallest_largest_primes_between_30_and_60_l503_503746


namespace weighted_average_sales_l503_503722

-- Define the sales data for the whole week
def sales_data : ℕ → (ℕ × ℕ × ℕ)
| 0 := (15, 10, 10) -- Monday
| 1 := (17, 13, 20) -- Tuesday
| 2 := (22, 18, 15) -- Wednesday
| 3 := (25, 12, 21) -- Thursday
| 4 := (17, 20, 24) -- Friday
| 5 := (20, 30, 25) -- Saturday
| 6 := (14, 10, 20) -- Sunday
| _ := (0, 0, 0)  -- Default (invalid index)

-- Calculate the total hamburgers and total food items sold on each day
def total_food_items (sales : (ℕ × ℕ × ℕ)) : ℕ :=
  sales.1 + sales.2 + sales.3

def total_statistics : (ℕ × ℕ) :=
  List.foldl (λ (acc : ℕ × ℕ) (i : ℕ) , 
                let daily_total := total_food_items (sales_data i) in
                (acc.1 + (sales_data i).1 * daily_total, acc.2 + daily_total))
             (0, 0) (List.range 7)

theorem weighted_average_sales : (total_statistics.1 : ℚ) / (total_statistics.2 : ℚ) ≈ 19.01 :=
by sorry

end weighted_average_sales_l503_503722


namespace probability_all_yellow_l503_503022

theorem probability_all_yellow (total_apples red_apples yellow_apples select_apples : ℕ)
  (h1 : total_apples = 10)
  (h2 : red_apples = 6)
  (h3 : yellow_apples = 4)
  (h4 : select_apples = 3)
  : (Nat.choose yellow_apples select_apples : ℚ) / (Nat.choose total_apples select_apples : ℚ) = 1 / 30 :=
by
  -- Proof omitted
  sorry

end probability_all_yellow_l503_503022


namespace count_expressible_integers_l503_503653

theorem count_expressible_integers :
  ∃ (x : Set ℝ), (∀ n (0 < n ∧ n ≤ 1500), 
  ∃ x, x ∈ Set.Icc (0 : ℝ) 1 ∧ ∀ n, 
  (n = ⌊3 * x⌋ + ⌊6 * x⌋ + ⌊9 * x⌋ + ⌊12 * x⌋) → 
  ∃ k ∈ (finset.range 1500), n = 900 :=
sorry

end count_expressible_integers_l503_503653


namespace area_of_ring_between_concentric_circles_l503_503850

theorem area_of_ring_between_concentric_circles :
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  area_ring = 95 * Real.pi :=
by
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  show area_ring = 95 * Real.pi
  sorry

end area_of_ring_between_concentric_circles_l503_503850


namespace equation_roots_l503_503819

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l503_503819


namespace num_combinations_l503_503704

theorem num_combinations (backpacks pencil_cases : ℕ) (h_backpacks : backpacks = 2) (h_pencil_cases : pencil_cases = 2) : 
  backpacks * pencil_cases = 4 :=
by
  rw [h_backpacks, h_pencil_cases]
  exact Nat.mul_self_eq 2

end num_combinations_l503_503704


namespace solution_set_of_inequality_l503_503503

def f : ℝ → ℝ := sorry

axiom f_value_at_1 : f 1 = 3
axiom f_derivative : ∀ x : ℝ, (∂/∂x : ℝ → ℝ) f x < 2

theorem solution_set_of_inequality :
  {x : ℝ | f x < 2 * x + 1} = {x ∈ set.Ioi 1 | true} :=
begin
  sorry
end

end solution_set_of_inequality_l503_503503


namespace area_of_triangle_EFC_l503_503706

theorem area_of_triangle_EFC :
  ∃ (E F C : Point), 
    (circle O with radius 5) ∧ 
    (is_tangent AC O) ∧ (is_tangent BC O) ∧ 
    (is_tangent EF O) ∧ (points E in_line AC) ∧ (points F in_line BC) ∧ 
    (perpendicular_line EF FC) ∧ 
    (distance C O = 13) ∧ 
    (area_of_triangle E F C = 420/17) :=
sorry

end area_of_triangle_EFC_l503_503706


namespace distinct_positive_integers_sums_are_kth_powers_l503_503999

open Nat

theorem distinct_positive_integers_sums_are_kth_powers (k : ℕ) (hk : k > 1) :
  ∃ (n : ℕ) (a : ℕ → ℕ), n > 1 ∧ (∀ i j, i ≠ j → a i ≠ a j) ∧
  (∀ i, 1 < a i) ∧ 
  let sum_a := ∑ i in finset.range n, a i
  let sum_phi_a := ∑ i in finset.range n, φ (a i)
  (∃ m l : ℕ, sum_a = m^k ∧ sum_phi_a = l^k) :=
sorry

end distinct_positive_integers_sums_are_kth_powers_l503_503999


namespace tom_speed_bc_l503_503109

-- Definitions from conditions
def distance_wb := 2 * distance_bc
def speed_wb := 60
def average_speed := 36

-- Question and conditions translation to prove statement
theorem tom_speed_bc (distance_bc : ℝ) : ∃ (speed_bc : ℝ), speed_bc = 20 :=
by
  have total_distance := 2 * distance_bc + distance_bc
  have total_time := (2 * distance_bc) / speed_wb + distance_bc / speed_bc
  have avg_speed_eq := average_speed = total_distance / total_time

  -- Now we can state the value of speed_bc
  use 20
  sorry

end tom_speed_bc_l503_503109


namespace f_neither_odd_nor_even_l503_503180

def f (x : ℝ) : ℝ := (Real.sin x)^2 + Real.tan x

theorem f_neither_odd_nor_even : ¬ (∀ x, f (-x) = f x) ∧ ¬ (∀ x, f (-x) = -f x) :=
by
  sorry

end f_neither_odd_nor_even_l503_503180


namespace geometric_sequence_sum_range_l503_503679

theorem geometric_sequence_sum_range (a b c : ℝ) 
  (h1 : ∃ q : ℝ, q ≠ 0 ∧ a = b * q ∧ c = b / q) 
  (h2 : a + b + c = 1) : 
  a + c ∈ (Set.Icc (2 / 3 : ℝ) 1 \ Set.Iio 1) ∪ (Set.Ioo 1 2) :=
sorry

end geometric_sequence_sum_range_l503_503679


namespace robot_path_length_l503_503907

/--
A robot moves in the plane in a straight line, but every one meter it turns 90° to the right or to the left. At some point it reaches its starting point without having visited any other point more than once, and stops immediately. Prove that the possible path lengths of the robot are 4k for some integer k with k >= 3.
-/
theorem robot_path_length (n : ℕ) (h : n > 0) (Movement : n % 4 = 0) :
  ∃ k : ℕ, n = 4 * k ∧ k ≥ 3 :=
sorry

end robot_path_length_l503_503907


namespace identical_rows_eventually_eleventh_row_identical_to_twelfth_tenth_and_eleventh_row_differ_l503_503875

-- Define the concept of rows and how they are generated
def generateRow (row : List ℕ) : List ℕ :=
  row.map (λ a => row.countp (λ x => x = a))

-- Define the main theorem
theorem identical_rows_eventually (init_row : List ℕ) (h_len : init_row.length = 1000) :
  ∃ n, generateRow (nthRow init_row n) = generateRow (nthRow init_row (n+1)) :=
  sorry

-- Prove that the 11th row will be identical to the 12th row given the initial row
theorem eleventh_row_identical_to_twelfth (init_row : List ℕ) (h_len : init_row.length = 1000) :
  generateRow (nthRow init_row 11) = generateRow (nthRow init_row 12) :=
  sorry

-- This would require constructing an example, which isn't formalized in Lean but would be a core idea
-- showing that two rows aren't identical
theorem tenth_and_eleventh_row_differ (init_row : List ℕ) (h_len : init_row.length = 1000) :
  ∃ r10 r11, generateRow (nthRow init_row 10) = r10 ∧ generateRow (nthRow init_row 11) = r11 ∧ r10 ≠ r11 :=
  sorry

-- Function to repetitively generate the nth row
def nthRow (row : List ℕ) (n : ℕ) : List ℕ :=
  if n = 0 then row else nthRow (generateRow row) (n-1)

end identical_rows_eventually_eleventh_row_identical_to_twelfth_tenth_and_eleventh_row_differ_l503_503875


namespace median_of_first_fifteen_integers_l503_503484

theorem median_of_first_fifteen_integers : 
  let L := (list.range 15).map (λ n, n + 1)
  in list.median L = 8.0 :=
by 
  sorry

end median_of_first_fifteen_integers_l503_503484


namespace probability_even_sum_l503_503969

theorem probability_even_sum : 
  let balls := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
  let total_outcomes := 15 * 14
  let even_balls := {x ∈ balls | x % 2 = 0}
  let odd_balls := {x ∈ balls | x % 2 = 1}
  let favorable_even_even := (even_balls.card * (even_balls.card - 1))
  let favorable_odd_odd := (odd_balls.card * (odd_balls.card - 1))
  let favorable_outcomes := favorable_even_even + favorable_odd_odd
  let probability := favorable_outcomes / total_outcomes
  in probability = 7 / 15 := sorry

end probability_even_sum_l503_503969


namespace arithmetic_sequence_common_difference_l503_503329

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_a2 : a 2 = 10) (h_a4 : a 4 = 18) : 
  d = 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l503_503329


namespace solve_x4_plus_81_eq_zero_l503_503193

theorem solve_x4_plus_81_eq_zero :
  (∀ x : ℂ, x^4 + 81 = 0 ↔ (x = 1.5 * √2 + 1.5 * √2 * Complex.i
                         ∨ x = -1.5 * √2 - 1.5 * √2 * Complex.i
                         ∨ x = 1.5 * √2 * Complex.i - 1.5 * √2
                         ∨ x = -1.5 * √2 * Complex.i + 1.5 * √2)) :=
by sorry

end solve_x4_plus_81_eq_zero_l503_503193


namespace vitya_convinced_of_12_models_l503_503081

noncomputable def min_offers_needed (n : ℕ) (k : ℕ) : ℕ :=
  if h : n = 13 then
    let ln100 := Real.log 100
    let ln13 := Real.log 13
    let ln12 := Real.log 12
    let req_k := Real.log 100 / (Real.log 13 - Real.log 12)
    if k > req_k then k else req_k.toNat + 1
  else k

theorem vitya_convinced_of_12_models (k : ℕ) : ∀ n, (n >= 13) → (min_offers_needed n k > 58) :=
by
  intros n h
  apply sorry

end vitya_convinced_of_12_models_l503_503081


namespace hexagon_area_inscribed_circle_l503_503488

theorem hexagon_area_inscribed_circle (r : ℝ) (hex_is_regular : r = 4) :
  let s := r in
  let area_tri := (sqrt 3 / 4) * s^2 in
  let area_hex := 6 * area_tri in
  area_hex = 24 * sqrt 3 := by
  sorry

end hexagon_area_inscribed_circle_l503_503488


namespace constant_of_zero_derivative_l503_503036

theorem constant_of_zero_derivative {a b : ℝ} {f : ℝ → ℝ} 
  (h_diff : ∀ x ∈ set.Icc a b, differentiable_at ℝ f x)
  (h_deriv_zero : ∀ x ∈ set.Icc a b, deriv f x = 0) :
  ∀ x ∈ set.Icc a b, f x = f a :=
by
  sorry

end constant_of_zero_derivative_l503_503036


namespace distance_M₀_to_plane_M₁_M₂_M₃_l503_503195

noncomputable def M₀ : ℝ × ℝ × ℝ := (10, -8, -7)
noncomputable def M₁ : ℝ × ℝ × ℝ := (-1, -5, 2)
noncomputable def M₂ : ℝ × ℝ × ℝ := (-6, 0, -3)
noncomputable def M₃ : ℝ × ℝ × ℝ := (3, 6, -3)

def distance_point_to_plane (p₀ p₁ p₂ p₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := p₀ in
  let (x₁, y₁, z₁) := p₁ in
  let (x₂, y₂, z₂) := p₂ in
  let (x₃, y₃, z₃) := p₃ in
  let A := (y₂ - y₁) * (z₃ - z₁) - (y₃ - y₁) * (z₂ - z₁) in
  let B := (z₂ - z₁) * (x₃ - x₁) - (z₃ - z₁) * (x₂ - x₁) in
  let C := (x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁) in
  let D := - (A * x₁ + B * y₁ + C * z₁) in
  (A * x₀ + B * y₀ + C * z₀ + D).natAbs / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_M₀_to_plane_M₁_M₂_M₃ :
  distance_point_to_plane M₀ M₁ M₂ M₃ = 2 * Real.sqrt 38 :=
by 
  sorry

end distance_M₀_to_plane_M₁_M₂_M₃_l503_503195


namespace omega_sets_l503_503226

def satisfies_omega_set_condition (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ y₁ : ℝ), y₁ = f x₁ → ∃ (x₂ y₂ : ℝ), y₂ = f x₂ ∧ x₁*x₂ + y₁*y₂ = 0

def M1 (x : ℝ) : ℝ := if x ≠ 0 then 1 / x else 0
def M2 (x : ℝ) : ℝ := (x - 1) / Real.exp x
def M3 (x : ℝ) : ℝ := Real.sqrt (1 - x^2)
def M4 (x : ℝ) : ℝ := x^2 - 2 * x + 2
def M5 (x : ℝ) : ℝ := Real.cos x + Real.sin x

theorem omega_sets :
  (satisfies_omega_set_condition M2) ∧ 
  (satisfies_omega_set_condition M3) ∧ 
  (satisfies_omega_set_condition M5) ∧
  ¬ (satisfies_omega_set_condition M1) ∧ 
  ¬ (satisfies_omega_set_condition M4) :=
by
  -- Proof to be filled here
  sorry

end omega_sets_l503_503226


namespace invest_amount_a_l503_503882

noncomputable def capital_invested_by_a
  (P_b P_diff : ℝ) (C_b C_c : ℝ)
  (h1 : P_b = 2000) 
  (h2 : P_diff = 799.9999999999998) 
  (h3 : C_b = 10000) 
  (h4 : C_c = 12000) 
  : ℝ :=
  let P_c := 1.2 * P_b in
  let P_a := P_c + P_diff in
  let C_a := (P_a / P_b) * C_b in
  C_a

theorem invest_amount_a 
  : capital_invested_by_a 2000 799.9999999999998 10000 12000 = 16000 :=
by
  simp [capital_invested_by_a]
  sorry

end invest_amount_a_l503_503882


namespace smallest_positive_e_l503_503047

-- Define the polynomial and roots condition
def polynomial (a b c d e : ℤ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

def has_integer_roots (p : ℝ → ℝ) (roots : List ℝ) : Prop :=
  ∀ r ∈ roots, p r = 0

def polynomial_with_given_roots (a b c d e : ℤ) : Prop :=
  has_integer_roots (polynomial a b c d e) [-3, 4, 11, -(1/4)]

-- Main theorem to prove the smallest positive integer e
theorem smallest_positive_e (a b c d : ℤ) :
  ∃ e : ℤ, e > 0 ∧ polynomial_with_given_roots a b c d e ∧
            (∀ e' : ℤ, e' > 0 ∧ polynomial_with_given_roots a b c d e' → e ≤ e') :=
  sorry

end smallest_positive_e_l503_503047


namespace find_some_number_l503_503497

-- Define the values involved based on the conditions
def expr_val : ℝ := 3.242 * 14 / 1000

-- Statement for the problem
theorem find_some_number : 3.242 * 14 / 1000 = 0.045388 :=
by 
  -- Proof can be added here
  sorry

end find_some_number_l503_503497


namespace f_g_relationship_l503_503992

-- Definitions based on conditions
def f (x : ℝ) : ℝ := 3 * x^2 - x + 1
def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

-- Proving the relationship f(x) > g(x)
theorem f_g_relationship (x : ℝ) : f(x) > g(x) :=
by {
  sorry
}

end f_g_relationship_l503_503992


namespace vitya_needs_58_offers_l503_503082

noncomputable def smallest_integer_k (P : ℝ → ℝ) : ℝ :=
  if H : ∃ k, k > P (100), then classical.some H else 0

theorem vitya_needs_58_offers :
  ∀ n : ℕ, n ≥ 13 → 
  (12:ℝ/13:ℝ) ^ smallest_integer_k (fun x => Real.log x / (Real.log 13 - Real.log 12)) < 0.01 :=
begin
  assume n h,
  rw smallest_integer_k,
  split_ifs,
  { sorry }, -- proof would go here
  { exfalso, exact sorry }, -- no proof steps provided
end

end vitya_needs_58_offers_l503_503082


namespace problem1_solution_problem2_solution_problem3_solution_problem4_solution_l503_503164

noncomputable def problem1 : Int := 1 + (-2) + abs (-2 - 3) - 5
theorem problem1_solution : problem1 = -1 := by
    sorry

noncomputable def problem2 : Rational := (-2) * (3 / 2) / (-(3 / 4)) * 4
theorem problem2_solution : problem2 = 16 := by
    sorry

noncomputable def problem3_inner : Rational := (7 / 9) - (11 / 12) + (1 / 6)
noncomputable def problem3 : Rational := ((-2) ^ 2 - problem3_inner * 36) / 5
theorem problem3_solution : problem3 = -1 := by
    sorry

noncomputable def problem4_inner : Rational := (-2) ^ 5 - (3 ^ 2) - (5 / 14) / (-(1 / 7))
noncomputable def problem4 : Rational := -1 ^ 2016 * problem4_inner - 2.5
theorem problem4_solution : problem4 = 36 := by
    sorry

end problem1_solution_problem2_solution_problem3_solution_problem4_solution_l503_503164


namespace find_second_expression_l503_503414

theorem find_second_expression (a : ℕ) (x : ℕ) 
  (h1 : (2 * a + 16 + x) / 2 = 74) (h2 : a = 28) : x = 76 := 
by
  sorry

end find_second_expression_l503_503414


namespace number_selection_probability_l503_503149

theorem number_selection_probability :
  let total_ways := 12 * 11 * 10 * 9 in
  let valid_ways := 182 in
  (valid_ways : ℚ) / total_ways = 13 / 845 :=
by
  -- The proof is omitted as per instructions
  sorry

end number_selection_probability_l503_503149


namespace farmer_harvested_correctly_l503_503132

def estimated_harvest : ℕ := 213489
def additional_harvest : ℕ := 13257
def total_harvest : ℕ := 226746

theorem farmer_harvested_correctly :
  estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvested_correctly_l503_503132


namespace kim_average_increase_l503_503933

noncomputable def avg (scores : List ℚ) : ℚ :=
  (scores.sum) / (scores.length)

theorem kim_average_increase :
  let scores_initial := [85, 89, 90, 92]  -- Initial scores
  let score_fifth := 95  -- Fifth score
  let original_average := avg scores_initial
  let new_average := avg (scores_initial ++ [score_fifth])
  new_average - original_average = 1.2 := by
  let scores_initial : List ℚ := [85, 89, 90, 92]
  let score_fifth : ℚ := 95
  let original_average : ℚ := avg scores_initial
  let new_average : ℚ := avg (scores_initial ++ [score_fifth])
  have : new_average - original_average = 1.2 := sorry
  exact this

end kim_average_increase_l503_503933


namespace kendall_nickels_count_l503_503350

theorem kendall_nickels_count :
  ∃ (n : ℕ), n * 0.05 = 4 - (10 * 0.25 + 12 * 0.10) ∧ n = 6 :=
by
  have quarters_value : ℝ := 10 * 0.25
  have dimes_value : ℝ := 12 * 0.10
  have total_value : ℝ := 4
  have nickels_value : ℝ := total_value - (quarters_value + dimes_value)
  use 6
  split
  sorry
  sorry

end kendall_nickels_count_l503_503350


namespace present_value_annuity_l503_503021

theorem present_value_annuity :
  let r := 0.04
      FV := 2400
      n := 8
      k := 3
      PV := 2400 * (1 - (1 + r)^(-k*n)) / ((1 + r)^k - 1) in
  PV = 11772.71 := by
    sorry

end present_value_annuity_l503_503021


namespace parabola_equation_and_slopes_l503_503639

theorem parabola_equation_and_slopes 
  (p : ℝ) (p_pos : p > 0) 
  (A : ℝ × ℝ) (A_eq : A = (1, Real.sqrt (2 * p))) 
  (B : ℝ × ℝ) (B_eq : B = (p / 2, 0)) 
  (dist_AB : Real.dist (1, Real.sqrt (2 * p)) (p / 2, 0) = 2 * Real.sqrt 2) :
  (p = 2) ∧ (∀ (x_0 y_0 x_1 y_1 x_2 y_2 : ℝ) (k_1 k_2 k_3 : ℝ),
    (y_0^2 = 4 * x_0) → 
    (y_1 = (4 / k_1) - y_0) →
    (x_1 = ((4 - y_0 * k_1)^2 / (4 * k_1^2))) →
    (y_2 = (4 / k_2) - y_0) →
    (x_2 = ((4 - y_0 * k_2)^2 / (4 * k_2^2))) →
    k_3 = ((4 / k_1 - y_0) - (4 / k_2 - y_0)) / ((x_1 - x_2)) →
    1 / k_3 = 1 / k_1 + 1 / k_2 - y_0 / 2) := 
begin
  sorry
end

end parabola_equation_and_slopes_l503_503639


namespace sample_size_correct_l503_503118

def total_students (freshmen sophomores juniors : ℕ) : ℕ :=
  freshmen + sophomores + juniors

def sample_size (total : ℕ) (prob : ℝ) : ℝ :=
  total * prob

theorem sample_size_correct (f : ℕ) (s : ℕ) (j : ℕ) (p : ℝ) (h_f : f = 400) (h_s : s = 320) (h_j : j = 280) (h_p : p = 0.2) :
  sample_size (total_students f s j) p = 200 :=
by
  sorry

end sample_size_correct_l503_503118


namespace cole_round_trip_time_l503_503951

-- Define the relevant quantities
def speed_to_work : ℝ := 70 -- km/h
def speed_to_home : ℝ := 105 -- km/h
def time_to_work_mins : ℝ := 72 -- minutes

-- Define the theorem to be proved
theorem cole_round_trip_time : 
  (time_to_work_mins / 60 + (speed_to_work * time_to_work_mins / 60) / speed_to_home) = 2 :=
by
  sorry

end cole_round_trip_time_l503_503951


namespace trisect_angle_l503_503007

noncomputable def can_trisect_with_ruler_and_compasses (n : ℕ) : Prop :=
  ¬(3 ∣ n) → ∃ a b : ℤ, 3 * a + n * b = 1

theorem trisect_angle (n : ℕ) (h : ¬(3 ∣ n)) :
  can_trisect_with_ruler_and_compasses n :=
sorry

end trisect_angle_l503_503007


namespace solve_equation_l503_503018

theorem solve_equation (x : ℝ) : 
  (x - 1) / 2 - (2 * x + 3) / 3 = 1 ↔ 3 * (x - 1) - 2 * (2 * x + 3) = 6 := 
sorry

end solve_equation_l503_503018


namespace leak_empties_cistern_in_12_hours_l503_503881

theorem leak_empties_cistern_in_12_hours 
  (R : ℝ) (L : ℝ)
  (h1 : R = 1 / 4) 
  (h2 : R - L = 1 / 6) : 
  1 / L = 12 := 
by
  -- proof will go here
  sorry

end leak_empties_cistern_in_12_hours_l503_503881


namespace proof_problem_l503_503982

noncomputable def problem_statement (a b : ℕ) : Prop :=
  let product := List.prod (List.map (λ i, real.log ((i : ℝ) + 1) / real.log (i : ℝ)) (list.iota 1000).map (λ i, a + i))
  in (product = 3) ∧ (List.length (List.map (λ i, a + i) (list.iota 1000)) = 1000)

theorem proof_problem (a b : ℕ) (h1 : a > 0) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 1000 → a + i > 0) (h3 : problem_statement a b) : a + b = 1010 := sorry

end proof_problem_l503_503982


namespace square_side_length_l503_503013

/-- A theorem stating the length of a side of the square constructed inside the right triangle -/
theorem square_side_length 
  (PQ PR : ℝ) 
  (hPQ : PQ = 9) 
  (hPR : PR = 12) 
  (hRight : ∀ P Q R, angle PQR = π / 2) 
  (s : ℝ) :
  s = 45 / 7 :=
by 
  -- We will prove this theorem using the given conditions (hPQ, hPR, hRight)
  sorry

end square_side_length_l503_503013


namespace salary_increase_correct_l503_503777

noncomputable def old_average_salary : ℕ := 1500
noncomputable def number_of_employees : ℕ := 24
noncomputable def manager_salary : ℕ := 11500
noncomputable def new_total_salary := (number_of_employees * old_average_salary) + manager_salary
noncomputable def new_number_of_people := number_of_employees + 1
noncomputable def new_average_salary := new_total_salary / new_number_of_people
noncomputable def salary_increase := new_average_salary - old_average_salary

theorem salary_increase_correct : salary_increase = 400 := by
sorry

end salary_increase_correct_l503_503777


namespace sum_formula_max_sum_value_l503_503255

def arithmetic_seq (n : ℕ) : ℤ := -2 * n + 11

noncomputable def sum_n_terms (n : ℕ) : ℤ := n * (arithmetic_seq 1 + arithmetic_seq n) / 2

theorem sum_formula (n : ℕ) : sum_n_terms n = -n^2 + 10 * n := by 
  sorry -- The detailed proof steps are omitted here.

theorem max_sum_value : ∃ n, sum_n_terms n = 25 ∧ ∀ m, sum_n_terms m ≤ 25 := by
  use 5
  split
  · sorry -- Proof that at n = 5, the sum reaches 25
  · sorry -- Proof that for all other m, the sum is less than or equal to 25

end sum_formula_max_sum_value_l503_503255


namespace sum_of_interior_angles_l503_503798

theorem sum_of_interior_angles (ext_angle : ℝ) (h : ext_angle = 20) : 
  let n := 360 / ext_angle in
  let int_sum := 180 * (n - 2) in
  int_sum = 2880 := 
by 
  sorry

end sum_of_interior_angles_l503_503798


namespace crescent_moon_area_is_2pi_l503_503336

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 ≥ 0 ∧ p.2 ≥ 0

def arc_subtype (c : Circle) (filter: (ℝ × ℝ) → Prop) : set (ℝ × ℝ) :=
  { p | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧ filter p }

def crescent_moon_area :=
  let large_circle := Circle.mk (0, 0) 4
  let small_circle := Circle.mk (0, 1) 2
  let quarter_large_arc := arc_subtype large_circle first_quadrant
  let semi_small_arc := { p : ℝ × ℝ | (p.1 - small_circle.center.1)^2 + (p.2 - small_circle.center.2)^2 = small_circle.radius^2 ∧ p.2 ≥ 0 }
  (4 * π * 4^2 / 4) - (π * 2^2 / 2)

theorem crescent_moon_area_is_2pi : crescent_moon_area = 2 * π :=
by
  sorry

end crescent_moon_area_is_2pi_l503_503336


namespace negation_of_exponential_prop_l503_503043

theorem negation_of_exponential_prop :
  (¬∃ x0 : ℝ, x0 > 0 ∧ 2^x0 ≤ 0) ↔ (∀ x : ℝ, x > 0 → 2^x > 0) :=
sorry

end negation_of_exponential_prop_l503_503043


namespace _l503_503995

noncomputable theorem solve_inequality (s x : ℝ) (hs_pos : s > 0) (hs_ne_one : s ≠ 1) (hx_pos : x > 0) (h_log_pos : real.log s x > 0) :
  (real.log (1 / s) (real.log s x) > real.log s (real.log s x)) ↔
  ((s > 1 ∧ 1 < x ∧ x < s) ∨ (0 < s ∧ s < 1 ∧ 0 < x ∧ x < s)) :=
by 
  sorry

end _l503_503995


namespace inequality_proof_l503_503230

noncomputable def proof_problem (n : ℕ) (a : ℕ → ℝ) :=
  (∀ i, 1 ≤ i ∧ i ≤ n → a i > 0) →
  (finset.univ.sum (λ i : fin n, a (i + 1)) = 1) →
  (finset.univ.prod (λ i : fin n, a (i + 1) + 1 / a (i + 1)) ≥ (n + 1 / n) ^ n)

theorem inequality_proof (n : ℕ) (a : ℕ → ℝ) : proof_problem n a :=
begin
  intros h1 h2,
  sorry
end

end inequality_proof_l503_503230


namespace max_triangle_area_l503_503737

/-- Let A = (1,0), B = (4,3), and C = (p, q) be three points on the parabola y = -x^2 + 8x - 12, 
    where 1 ≤ p ≤ 4. The largest possible area of triangle ABC is 15/8. -/
theorem max_triangle_area : ∃ p : ℝ, (1 ≤ p ∧ p ≤ 4) ∧ 
  let q := -p^2 + 8 * p - 12 in ((3 : ℝ) / 2) * abs ((p - 7 / 2)^2 - 1 / 4) = 15 / 8 :=
by
  sorry

end max_triangle_area_l503_503737


namespace probability_of_different_groups_is_correct_l503_503828

-- Define the number of total members and groups
def num_groups : ℕ := 6
def members_per_group : ℕ := 3
def total_members : ℕ := num_groups * members_per_group

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function to select 3 people from different groups
noncomputable def probability_different_groups : ℚ :=
  binom num_groups 3 / binom total_members 3

-- State the theorem we want to prove
theorem probability_of_different_groups_is_correct :
  probability_different_groups = 5 / 204 :=
by
  sorry

end probability_of_different_groups_is_correct_l503_503828


namespace smallest_n_divides_l503_503360

theorem smallest_n_divides (m : ℕ) (h1 : m % 2 = 1) (h2 : m > 2) :
  ∃ n : ℕ, 2^(1988) = n ∧ 2^1989 ∣ m^n - 1 :=
by
  sorry

end smallest_n_divides_l503_503360


namespace minimum_cubes_required_l503_503515

def box_length := 12
def box_width := 16
def box_height := 6
def cube_volume := 3

def volume_box := box_length * box_width * box_height

theorem minimum_cubes_required : volume_box / cube_volume = 384 := by
  sorry

end minimum_cubes_required_l503_503515


namespace reflections_on_circumcircle_l503_503727

open EuclideanGeometry

structure Triangle :=
(A B C : Point)

def orthocenter (t : Triangle) : Point := sorry

def is_reflection (p q : Point) (l : Line) : Prop := sorry

def circumscribed_circle (t : Triangle) : Circle := sorry

theorem reflections_on_circumcircle (t : Triangle) (H_h : Point) 
  (orthocenter_def : orthocenter t = H_h) 
  (acute_triangle : ∀ (A B C : Point), is_acute △ABC) :
  ∀ (H_A H_B H_C : Point), 
    (is_reflection H_h H_A (Line.mk t.B t.C)) ∧ 
    (is_reflection H_h H_B (Line.mk t.C t.A)) ∧ 
    (is_reflection H_h H_C (Line.mk t.A t.B)) →
    H_A ∈ circumscribed_circle t ∧ 
    H_B ∈ circumscribed_circle t ∧ 
    H_C ∈ circumscribed_circle t := 
  sorry

end reflections_on_circumcircle_l503_503727


namespace proof_problem_l503_503258

theorem proof_problem (a b : ℝ) (h : a^2 + b^2 + 2*a - 4*b + 5 = 0) : 2*a^2 + 4*b - 3 = 7 :=
sorry

end proof_problem_l503_503258


namespace ten_player_round_robin_matches_l503_503910

theorem ten_player_round_robin_matches :
  (∑ i in finset.range 10, i) = 45 :=
by
  sorry

end ten_player_round_robin_matches_l503_503910


namespace sum_of_interior_angles_l503_503784

theorem sum_of_interior_angles (h : ∀ (n : ℕ), 360 / 20 = n) : 
  ∃ (s : ℕ), s = 2880 :=
by
  have n := 360 / 20
  have sum := 180 * (n - 2)
  use sum
  sorry

end sum_of_interior_angles_l503_503784


namespace expr_is_integer_l503_503175

noncomputable def binomial (n k : ℕ) : ℕ :=
  n.choose k

def expr (n k : ℕ) : ℤ :=
  ((n - 3 * k - 2) / (k + 2)) * binomial n k

theorem expr_is_integer (k n : ℕ) (hk : 1 ≤ k) (hkn : k < n) : 
  (expr n k).denom = 1 :=
by
  sorry

end expr_is_integer_l503_503175


namespace area_of_inscribed_rectangle_l503_503406

noncomputable def area_rectangle_ABCD 
  (AD AB : ℝ) 
  (EG : ℝ := 15)
  (altitude_F_to_EG : ℝ := 10)
  (h1: AB = (1/3) * AD)
  (h2: ∃ AE DG, (AE + DG = EG ∧ ((1.5 : ℝ) = (15 - AD) / AB))) : 
  ℝ :=
AD * AB

theorem area_of_inscribed_rectangle (AD : ℝ)
  (AB : ℝ := (1 / 3) * AD)
  (EG : ℝ := 15)
  (altitude_F_to_EG : ℝ := 10)
  (h1: AB = (1/3) * AD)
  (h2: ∃ AE DG, (AE + DG = EG ∧ ((1.5 : ℝ) = (15 - AD) / AB))):
  area_rectangle_ABCD AD AB = 100 / 3 := 
by 
  sorry

end area_of_inscribed_rectangle_l503_503406


namespace Maria_needs_more_l503_503688

def num_mechanics : Nat := 20
def num_thermodynamics : Nat := 50
def num_optics : Nat := 30
def total_questions : Nat := num_mechanics + num_thermodynamics + num_optics

def correct_mechanics : Nat := (80 * num_mechanics) / 100
def correct_thermodynamics : Nat := (50 * num_thermodynamics) / 100
def correct_optics : Nat := (70 * num_optics) / 100
def correct_total : Nat := correct_mechanics + correct_thermodynamics + correct_optics

def correct_for_passing : Nat := (65 * total_questions) / 100
def additional_needed : Nat := correct_for_passing - correct_total

theorem Maria_needs_more:
  additional_needed = 3 := by
  sorry

end Maria_needs_more_l503_503688


namespace polynomial_simplification_l503_503017

theorem polynomial_simplification (p : ℤ) :
  (5 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 2) + (-3 * p^4 + 4 * p^3 + 8 * p^2 - 2 * p + 6) = 
  2 * p^4 + 6 * p^3 + p^2 + p + 4 :=
by
  sorry

end polynomial_simplification_l503_503017


namespace system_solution_l503_503020

noncomputable def x : ℝ := 2
noncomputable def y : ℝ := 1 / 2

theorem system_solution : 
  (3 * x - 2 * y = 5) ∧ (x + 4 * y = 4) :=
by
  unfold x y
  split
  sorry
  sorry

end system_solution_l503_503020


namespace bottle_caps_per_group_l503_503718

theorem bottle_caps_per_group (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) 
  (h1 : total_caps = 12) (h2 : num_groups = 6) : 
  total_caps / num_groups = caps_per_group := by
  sorry

end bottle_caps_per_group_l503_503718


namespace smallest_class_size_l503_503691

/--
In a science class, students are separated into five rows for an experiment. 
The class size must be greater than 50. 
Three rows have the same number of students, one row has two more students than the others, 
and another row has three more students than the others.
Prove that the smallest possible class size for this science class is 55.
-/
theorem smallest_class_size (class_size : ℕ) (n : ℕ) 
  (h1 : class_size = 3 * n + (n + 2) + (n + 3))
  (h2 : class_size > 50) :
  class_size = 55 :=
sorry

end smallest_class_size_l503_503691


namespace intersection_points_at_most_one_l503_503431

-- Define the function and the line
variable {α β : Type*} [linear_ordered_field α] [order_topology α] [topological_space β]
variable (f : α → β) (a : α)

-- Lean theorem statement
theorem intersection_points_at_most_one :
  ∃ y, y = f a → a ∈ set_of (λ x, x = a) → ∀ y1 y2, y1 = f a → y2 = f a → y1 = y2 := 
sorry

end intersection_points_at_most_one_l503_503431


namespace juan_max_error_l503_503346

noncomputable def maxPercentError (d : ℝ) (error : ℝ): ℝ :=
  let min_d := d * (1 - error)
  let max_d := d * (1 + error)
  let true_area := Real.pi * (d / 2) ^ 2
  let min_area := Real.pi * (min_d / 2) ^ 2
  let max_area := Real.pi * (max_d / 2) ^ 2
  let min_error := (true_area - min_area) / true_area * 100
  let max_error := (max_area - true_area) / true_area * 100
  min max_error min_error

theorem juan_max_error :
  maxPercentError 30 0.30 = 69 :=
by
  sorry

end juan_max_error_l503_503346


namespace price_of_other_stamp_l503_503139

-- Define the conditions
def total_stamps : ℕ := 75
def total_value_cents : ℕ := 480
def known_stamp_price : ℕ := 8
def known_stamp_count : ℕ := 40
def unknown_stamp_count : ℕ := total_stamps - known_stamp_count

-- The problem to solve
theorem price_of_other_stamp (x : ℕ) :
  (known_stamp_count * known_stamp_price) + (unknown_stamp_count * x) = total_value_cents → x = 5 :=
by
  sorry

end price_of_other_stamp_l503_503139


namespace nice_function_count_at_least_l503_503363

open scoped Classical

-- Defining the conditions
variables (m n : ℕ) (X : Type*) [Fintype X] [Card X = n]
variables (X_1 X_2 ... X_m : set X) 
  (H_X1 : ∀ i j, 1 ≤ i, j ≤ m → i ≠ j → X_i ≠ X_j) -- pairwise distinct subsets
  (H_X1_nonempty : ∀ i, 1 ≤ i ≤ m → X_i ≠ ∅) -- non-empty subsets
  (f : X → ℕ) (H_f_range : ∀ x, f x ∈ set.range (Finset.finRange (n + 1))) -- f maps X to {1, 2, ..., n+1}

-- Definition of a nice function
def is_nice (f : X → ℕ) : Prop := 
  ∃ k, 1 ≤ k ≤ m ∧ ∀ i, i ≠ k → (Finset.sum (Finset.filter (λ x, x ∈ X_k) (Fintype.elems X)) f) > 
  (Finset.sum (Finset.filter (λ x, x ∈ X_i) (Fintype.elems X)) f)

-- Statement that we need to prove (no proof required)
theorem nice_function_count_at_least : 
  ∃ Φ : (X → { i : ℕ // i < n + 1 }) → (X → ℕ), 
    (Injective Φ) ∧ (∀ f ∈ X → { i : ℕ // i < n + 1 }, is_nice (Φ f)) ∧ (Finset.card (X → { i : ℕ // i < n + 1 })) ≥ n ^ n := 
sorry

end nice_function_count_at_least_l503_503363


namespace large_posters_count_l503_503915

theorem large_posters_count (total_posters small_ratio medium_ratio : ℕ) (h_total : total_posters = 50) (h_small_ratio : small_ratio = 2/5) (h_medium_ratio : medium_ratio = 1/2) :
  let small_posters := (small_ratio * total_posters) in
  let medium_posters := (medium_ratio * total_posters) in
  let large_posters := total_posters - (small_posters + medium_posters) in
  large_posters = 5 := by
{
  sorry
}

end large_posters_count_l503_503915


namespace probability_condition_is_one_l503_503519

-- Define the square and the properties of the point (x, y) chosen uniformly at random
def square_region (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2

-- Define the condition that we are interested in
def condition (x y : ℝ) : Prop :=
  x + y < 4

-- Prove that the probability of the condition holding within the square region is 1
theorem probability_condition_is_one :
  let P := {p : ℝ × ℝ | square_region p.1 p.2} in
  let Q := {p : ℝ × ℝ | square_region p.1 p.2 ∧ condition p.1 p.2} in
  ∃! (p : ℝ × ℝ), square_region p.1 p.2 →
  (set.finite Q ∧ set.fin Q = 1) :=
sorry

end probability_condition_is_one_l503_503519


namespace AE_perp_CE_l503_503313

-- Definitions based on conditions from the problem
variables {A B C D E : Type}
variables [Decision : DecidableEq A] [LinearOrder A] [Field A]
variables 
  (triangleABC : Triangle A)
  (AB AC : A)
  (angleBAC : Real.Angle)
  (midpointE : Midpoint A)

-- Given conditions
def is_isosceles_triangle (TAB : triangleABC): Prop := AB = AC

def angle_BAC_eq : angleBAC = 108 := sorry

def D_on_extended_AC (AD_eq_BC : A) : D := sorry

def E_midpoint_BD (BD : A) : E := sorry

-- The statement to be proven
theorem AE_perp_CE 
  (h1 : is_isosceles_triangle triangleABC)
  (h2 : angle_BAC_eq)
  (h3 : ∃ D, D_on_extended_AC AD_eq_BC)
  (h4 : ∃ E, E_midpoint_BD BD):
  Perpendicular AE CE := sorry

end AE_perp_CE_l503_503313


namespace density_difference_of_cubes_l503_503452

theorem density_difference_of_cubes (a m : ℝ) (h_a_pos : a > 0) (h_m_pos : m > 0) :
  let a₂ := 1.25 * a,
      m₂ := 0.75 * m,
      V₁ := a ^ 3,
      V₂ := a₂ ^ 3,
      ρ₁ := m / V₁,
      ρ₂ := m₂ / V₂ in
  ((ρ₁ - ρ₂) / ρ₁) * 100 = 61.6 :=
by
  sorry

end density_difference_of_cubes_l503_503452


namespace integral_sqrt_1_minus_x_sq_half_circle_l503_503966

theorem integral_sqrt_1_minus_x_sq_half_circle :
  ∫ x in -1..1, sqrt (1 - x^2) = (π / 2) :=
sorry

end integral_sqrt_1_minus_x_sq_half_circle_l503_503966


namespace length_PT_l503_503712

noncomputable def triangle_problem : ℕ :=
  let e := 1797
  let f := 500
  let g := 1
  let h := 100
  e + f + g + h

theorem length_PT (PQ QR PR : ℝ) (PS PT : ℝ) (S U : ℝ) (SU TU : ℝ) 
  (PQ_lt_PS : PQ < PS) (PS_lt_PT : PS < PT) (U_ne_R : U ≠ R)
  (SU_eq_3 : SU = 3) (TU_eq_8 : TU = 8) : 
  PQ = 5 → QR = 6 → PR = 7 → 
  triangle_problem = 2398 := 
by
  intros hPQ hQR hPR
  unfold triangle_problem
  simp
  sorry

end length_PT_l503_503712


namespace parabola_distance_to_xaxis_l503_503396

open Real

noncomputable def parabola (x y : ℝ) : Prop :=
  x^2 = 4 * y

def focus : ℝ × ℝ := (0, 1)
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_distance_to_xaxis
  (x y : ℝ)
  (h₁ : parabola x y)
  (h₂ : distance (x, y) focus = 8) :
  y = 7 :=
sorry

end parabola_distance_to_xaxis_l503_503396


namespace find_valid_n_l503_503570

def is_valid_n (n : ℕ) : Prop :=
  (n % 10 = 0) ∨
  (2 ∣ n ∧ ¬(5 ∣ n) ∧ (nat.factorization n).find 2 ≥ 4) ∨
  (5 ∣ n ∧ ¬(2 ∣ n) ∧ (nat.factorization n).find 5 ≥ 2)

theorem find_valid_n (n : ℕ) (h1 : 10 ≤ n) (h2 : n ≤ 2019) :
  (∀ m, m % n = 0 → has_at_least_two_distinct_digits m) ↔ is_valid_n n :=
sorry

end find_valid_n_l503_503570


namespace sum_of_evens_and_odds_l503_503827

theorem sum_of_evens_and_odds (N : ℕ) (h_even_sum : ∑ k in (finset.filter (λ x, x % 2 = 0) (finset.range (N + 1))), k = 90)
    (h_odd_sum : ∑ k in (finset.filter (λ x, x % 2 = 1) (finset.range (N + 1))), k = 100) : N = 19 := by
  sorry

end sum_of_evens_and_odds_l503_503827


namespace min_bottles_required_to_fill_container_l503_503717

-- Definitions based on conditions
def capacity_large_container := 1125
def capacity_small_bottle_1 := 45
def capacity_small_bottle_2 := 75

-- Problem statement translated to Lean 4
theorem min_bottles_required_to_fill_container : 
  min_bottles_needed capacity_large_container capacity_small_bottle_1 capacity_small_bottle_2 = 15 := 
sorry

end min_bottles_required_to_fill_container_l503_503717


namespace median_first_fifteen_positive_integers_l503_503466

-- Define the list of the first fifteen positive integers
def first_fifteen_positive_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

-- Define the property that the median of the list is 8.0
theorem median_first_fifteen_positive_integers : median(first_fifteen_positive_integers) = 8.0 := 
sorry

end median_first_fifteen_positive_integers_l503_503466


namespace roots_of_equation_l503_503806

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l503_503806


namespace max_value_of_xyz_l503_503375

theorem max_value_of_xyz (x y z : ℝ) (h : x + 3 * y + z = 5) : xy + xz + yz ≤ 125 / 4 := 
sorry

end max_value_of_xyz_l503_503375


namespace roots_of_quadratic_eq_l503_503815

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_eq_l503_503815


namespace isosceles_triangle_area_l503_503843

theorem isosceles_triangle_area 
  (P Q R : Type) [Triangle P Q R]
  (h1 : PQ = 13)
  (h2 : PR = 13)
  (h3 : QR = 24) :
  area P Q R = 60 :=
sorry

end isosceles_triangle_area_l503_503843


namespace problem1_problem2_l503_503216

def f (x : ℝ) := |x + 1| + |x - 1|

theorem problem1 (x : ℝ) : f x ≤ x + 2 ↔ 0 ≤ x ∧ x ≤ 2 :=
by
  sorry

theorem problem2 (x : ℝ) (a : ℝ) : 
  (∀ a, f x ≤ Real.log 2 (a^2 - 4 * a + 12)) ↔ -1 < x ∧ x ≤ 1.5 :=
by
  sorry

end problem1_problem2_l503_503216


namespace tangent_line_equation_l503_503035

noncomputable def f (x : ℝ) : ℝ := Real.log ((2 - x) / (2 + x))

-- The point (x0, y0)
def x0 : ℝ := -1
def y0 : ℝ := f x0

-- The slope of the tangent line at x = -1
def slope : ℝ := -4 / 3

-- The equation of the tangent line at (x0, y0)
def tangent_line (x : ℝ) : ℝ := slope * x + (Real.log 3 - 4 / 3)

-- Problem statement: verifying the equation of the tangent line at the given point
theorem tangent_line_equation :
  ∀ x : ℝ, y0 = Real.log 3 → 
 (∀ x : ℝ, f x = Real.log (2 - x) - Real.log (2 + x)) →
 (∀ x : ℝ, y - y0 = slope * (x - x0) ↔ y = slope * x + (Real.log 3 - 4 / 3)) := by
  sorry

end tangent_line_equation_l503_503035


namespace abs_pi_squared_l503_503548

theorem abs_pi_squared (π : ℝ) (h : π ≈ 3.14) : |3 - |9 - π^2|| = 12 - π^2 := by 
  sorry

end abs_pi_squared_l503_503548


namespace andy_wrong_questions_l503_503324

theorem andy_wrong_questions (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 3) : a = 6 := by
  sorry

end andy_wrong_questions_l503_503324


namespace dodecahedron_equilateral_triangles_l503_503650

-- Definitions reflecting the conditions
def vertices_of_dodecahedron := 20
def faces_of_dodecahedron := 12
def vertices_per_face := 5
def equilateral_triangles_per_face := 5

theorem dodecahedron_equilateral_triangles :
  (faces_of_dodecahedron * equilateral_triangles_per_face) = 60 := by
  sorry

end dodecahedron_equilateral_triangles_l503_503650


namespace triangles_from_decagon_l503_503667

-- Define the parameters for the problem
def n : ℕ := 10
def k : ℕ := 3

-- Define the combination formula
def combination (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- State the theorem we want to prove
theorem triangles_from_decagon : combination n k = 120 := by
  -- Proof steps would go here
  sorry

end triangles_from_decagon_l503_503667


namespace builder_total_amount_paid_l503_503127

theorem builder_total_amount_paid :
  let cost_drill_bits := 5 * 6
  let tax_drill_bits := 0.10 * cost_drill_bits
  let total_cost_drill_bits := cost_drill_bits + tax_drill_bits

  let cost_hammers := 3 * 8
  let discount_hammers := 0.05 * cost_hammers
  let total_cost_hammers := cost_hammers - discount_hammers

  let cost_toolbox := 25
  let tax_toolbox := 0.15 * cost_toolbox
  let total_cost_toolbox := cost_toolbox + tax_toolbox

  let total_amount_paid := total_cost_drill_bits + total_cost_hammers + total_cost_toolbox

  total_amount_paid = 84.55 :=
by
  sorry

end builder_total_amount_paid_l503_503127


namespace probability_of_consecutive_letters_l503_503451

theorem probability_of_consecutive_letters :
  ∀ (cards : finset char), 
  cards = {'A', 'B', 'C', 'D', 'E'} →
  (finset.card (finset.filter (λ (s : finset char), ∃ a b, s = {a, b} ∧ (a.ord - b.ord).abs = 1 % (cards.card - 1)) 
                             (finsets_univ (finset.card 'A') (finset.card 'E')))) = 4 →
  (finsets_univ (2 // 5)) :=
by
  sorry

end probability_of_consecutive_letters_l503_503451


namespace find_m_l503_503562

-- Definitions for conditions
def line_parametric (m : ℝ) (t : ℝ) : (ℝ × ℝ) :=
  (√3 / 2 * t + m, 1 / 2 * t)

def curve_polar (θ : ℝ) : (ℝ × ℝ) :=
  let ρ := 2 * Real.cos θ in
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Proposition for proof
theorem find_m (m : ℝ) :
  (∃ t1 t2 : ℝ, (√3 / 2 * t1 + m) ^ 2 + (1 / 2 * t1) ^ 2 = 2 * (√3 / 2 * t1 + m) ∧
               (√3 / 2 * t2 + m) ^ 2 + (1 / 2 * t2) ^ 2 = 2 * (√3 / 2 * t2 + m) ∧ 
                t1 * t2 = 1) →
  (m = 1 + Real.sqrt 2 ∨ m = 1 - Real.sqrt 2 ∨ m = 1) := by
  sorry

end find_m_l503_503562


namespace integral_of_f_from_0_to_1_l503_503270

def f (x : ℝ) : ℝ := x^2 - x + 2

theorem integral_of_f_from_0_to_1 : ∫ x in 0..1, f x = 11 / 6 :=
by
  sorry

end integral_of_f_from_0_to_1_l503_503270


namespace math_marks_is_95_l503_503173

-- Define the conditions as Lean assumptions
variables (english_marks math_marks physics_marks chemistry_marks biology_marks : ℝ)
variable (average_marks : ℝ)
variable (num_subjects : ℝ)

-- State the conditions
axiom h1 : english_marks = 96
axiom h2 : physics_marks = 82
axiom h3 : chemistry_marks = 97
axiom h4 : biology_marks = 95
axiom h5 : average_marks = 93
axiom h6 : num_subjects = 5

-- Formalize the problem: Prove that math_marks = 95
theorem math_marks_is_95 : math_marks = 95 :=
by
  sorry

end math_marks_is_95_l503_503173


namespace median_salary_is_28000_l503_503527

-- Definition of each position's salary and count
def ceo_count : ℕ := 1
def senior_vp_count : ℕ := 4
def manager_count : ℕ := 15
def assistant_manager_count : ℕ := 10
def clerk_count : ℕ := 45

def ceo_salary : ℕ := 150000
def senior_vp_salary : ℕ := 110000
def manager_salary : ℕ := 80000
def assistant_manager_salary : ℕ := 55000
def clerk_salary : ℕ := 28000

-- Total number of employees
def total_employees : ℕ :=
  ceo_count + senior_vp_count + manager_count + assistant_manager_count + clerk_count

-- The definition of the median salary based on conditions
def median_salary : ℕ :=
  if (ceo_count + senior_vp_count + manager_count + assistant_manager_count + clerk_count) % 2 = 1 then
    -- The number of employees is odd, so median is the middle element
    -- Given conditions, median falls under the clerk's salary
    clerk_salary
  else
    -- This branch won't be reached as number of employees is 75 (odd)
    sorry

-- Problem statement in Lean 4
theorem median_salary_is_28000 : median_salary = 28000 := by
  have total_employees_75 : total_employees = 75 := by
    unfold total_employees ceo_count senior_vp_count manager_count assistant_manager_count clerk_count
    norm_num
  have median_position : total_employees // 2 + 1 = 38 := by
    calc
      75 // 2 + 1 = 37 + 1 : by norm_num
      _ = 38 : by norm_num
  have clerk_count_surpasses_38th : clerk_count ≥ 38 := by
    unfold clerk_count
    norm_num
  unfold median_salary
  simp [total_employees_75, clerk_count_surpasses_38th]
  exact rfl

end median_salary_is_28000_l503_503527


namespace ordered_pairs_sol_3_over_m_plus_6_over_n_eq_1_l503_503289

theorem ordered_pairs_sol_3_over_m_plus_6_over_n_eq_1 :
  {p : ℕ × ℕ // 3 / p.1 + 6 / p.2 = 1}.toFinset.card = 6 :=
sorry

end ordered_pairs_sol_3_over_m_plus_6_over_n_eq_1_l503_503289


namespace sin_le_cos_iff_l503_503761

theorem sin_le_cos_iff (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : 
  (sin x ≤ cos x) ↔ (x ≤ π / 4) := 
by 
  sorry

end sin_le_cos_iff_l503_503761


namespace smallest_number_remainder_l503_503861

open Nat

theorem smallest_number_remainder
  (b : ℕ)
  (h1 : b % 4 = 2)
  (h2 : b % 3 = 2)
  (h3 : b % 5 = 3) :
  b = 38 :=
sorry

end smallest_number_remainder_l503_503861


namespace equation_roots_l503_503818

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l503_503818


namespace stars_total_is_correct_l503_503061

-- Define the given conditions
def number_of_stars_per_student : ℕ := 6
def number_of_students : ℕ := 210

-- Define total number of stars calculation
def total_number_of_stars : ℕ := number_of_stars_per_student * number_of_students

-- Proof statement that the total number of stars is correct
theorem stars_total_is_correct : total_number_of_stars = 1260 := by
  sorry

end stars_total_is_correct_l503_503061


namespace M_is_orthocenter_l503_503207

variables {A B C M : Type}
variables [triangle ABC : Type] [interior_point M ABC : Type]
variables [angle_equality : ∀ (AM BC BM CA CM AB: Line), equal_angle(AM, BC) ∧ equal_angle(BM, CA) ∧ equal_angle(CM, AB)]

theorem M_is_orthocenter 
  (h1 : M ∈ interior_point ABC)
  (h2 : ∃ ϕ, (ϕ ≤ 90) ∧ (∀ (AM BC BM CA CM AB : Line), equal_angle(AM, BC) = ϕ ∧ equal_angle(BM, CA) = ϕ ∧ equal_angle(CM, AB) = ϕ)) :
  is_orthocenter M ABC :=
sorry

end M_is_orthocenter_l503_503207


namespace base5_sum_correct_l503_503423

def int_to_base5 (n : ℕ) : List ℕ := 
  if n = 0 then [0] else
    let rec aux (n : ℕ) : List ℕ :=
      if n = 0 then [] else (n % 5) :: (aux (n / 5))
    aux n

def base5_sum (n1 n2 : ℕ) : ℕ := 
  let b1 := int_to_base5 n1
  let b2 := int_to_base5 n2
  -- Evaluation of the sum in base 5, converting the list representation back to a number in base 10
  let rec to_base10 (l : List ℕ) : ℕ :=
    match l with
    | [] => 0
    | h::t => h + 5 * (to_base10 t)
  let sum_b5 := (List.zipWith (λ x y, x + y) b1 b2)   -- assuming same length, carry logic omitted for simplicity
  to_base10 sum_b5

theorem base5_sum_correct : base5_sum 243 62 = 2170 :=
  by
  -- Here we'd state and prove that converting 243 and 62 to base 5 and summing
  -- their representations gives 2170 in base 5.
  sorry

end base5_sum_correct_l503_503423


namespace problem_statement_l503_503238

-- We define the conditions of the problem
variable (α : Real)
hypothesis (h : sin α + 3 * cos α = 0)

-- The statement we want to prove
theorem problem_statement : 2 * sin (2 * α) - cos α ^ 2 = -13 / 10 :=
by
  sorry

end problem_statement_l503_503238


namespace sum_divisors_of_24_is_60_and_not_prime_l503_503945

def divisors (n : Nat) : List Nat :=
  List.filter (λ d => n % d = 0) (List.range (n + 1))

def sum_divisors (n : Nat) : Nat :=
  (divisors n).sum

def is_prime (n : Nat) : Bool :=
  n > 1 ∧ (List.filter (λ d => d > 1 ∧ d < n ∧ n % d = 0) (List.range (n + 1))).length = 0

theorem sum_divisors_of_24_is_60_and_not_prime :
  sum_divisors 24 = 60 ∧ ¬ is_prime 60 := 
by
  sorry

end sum_divisors_of_24_is_60_and_not_prime_l503_503945


namespace cosine_transformation_equivalence_l503_503836

theorem cosine_transformation_equivalence :
  ∀ (x : ℝ), cos (2 * x - π / 3) = cos ((x - π / 3) * 2) :=
begin
  intro x,
  sorry
end

end cosine_transformation_equivalence_l503_503836


namespace factorable_polynomial_l503_503012

theorem factorable_polynomial (a b : ℝ) :
  (∀ x y : ℝ, ∃ u v p q : ℝ, (x + uy + v) * (x + py + q) = x * (x + 4) + a * (y^2 - 1) + 2 * b * y) ↔
  (a + 2)^2 + b^2 = 4 :=
  sorry

end factorable_polynomial_l503_503012


namespace sum_of_interior_angles_l503_503796

theorem sum_of_interior_angles (ext_angle : ℝ) (h : ext_angle = 20) : 
  let n := 360 / ext_angle in
  let int_sum := 180 * (n - 2) in
  int_sum = 2880 := 
by 
  sorry

end sum_of_interior_angles_l503_503796


namespace older_brother_has_17_stamps_l503_503045

def stamps_problem (y : ℕ) : Prop := y + (2 * y + 1) = 25

theorem older_brother_has_17_stamps (y : ℕ) (h : stamps_problem y) : 2 * y + 1 = 17 :=
by
  sorry

end older_brother_has_17_stamps_l503_503045


namespace interior_angles_sum_l503_503795

theorem interior_angles_sum (h : ∀ (n : ℕ), n = 360 / 20) : 
  180 * (h 18 - 2) = 2880 :=
by
  sorry

end interior_angles_sum_l503_503795


namespace even_function_monotonicity_l503_503241

theorem even_function_monotonicity (f : ℝ → ℝ)
  (hf_even : ∀ x, f (x) = f (-x))
  (hf_monotone : ∀ x1 x2 ∈ set.Iic (-1), (x2 - x1) * (f x2 - f x1) < 0) :
  f (-1) < f (-3/2) ∧ f (-3/2) < f 2 :=
begin
  -- Proof required
  sorry
end

end even_function_monotonicity_l503_503241


namespace average_score_of_entire_class_l503_503315

theorem average_score_of_entire_class :
  ∀ (num_students num_boys : ℕ) (avg_score_girls avg_score_boys : ℝ),
  num_students = 50 →
  num_boys = 20 →
  avg_score_girls = 85 →
  avg_score_boys = 80 →
  (avg_score_boys * num_boys + avg_score_girls * (num_students - num_boys)) / num_students = 83 :=
by
  intros num_students num_boys avg_score_girls avg_score_boys
  sorry

end average_score_of_entire_class_l503_503315


namespace jihye_triangle_area_multiple_l503_503964

/-!
  Problem Statement:
  During the origami class, Donggeon folded the paper in a triangular shape with a base of 3 centimeters (cm) 
  and a height of 2 centimeters (cm), and Jihye folded the paper in a triangular shape with 
  a base of 3 centimeters (cm) and a height of 6.02 centimeters (cm). 
  Prove that the area of the triangle folded by Jihye is 3.01 times the area of the triangle folded by Donggeon.
-/

noncomputable def area_of_triangle (base height : ℝ) : ℝ :=
  0.5 * base * height

theorem jihye_triangle_area_multiple :
  let base_D := 3 : ℝ
  let height_D := 2 : ℝ
  let base_J := 3 : ℝ
  let height_J := 6.02 : ℝ
  area_of_triangle base_J height_J = 3.01 * area_of_triangle base_D height_D := 
by
  sorry

end jihye_triangle_area_multiple_l503_503964


namespace algebraic_expressions_additive_inverse_quadratic_surds_same_type_l503_503114

-- Additive inverses imply given x values
theorem algebraic_expressions_additive_inverse (x : ℝ) (h : (x^2 + 3 * x - 6) + (-x + 1) = 0) :
  x = -1 + real.sqrt 6 ∨ x = -1 - real.sqrt 6 :=
sorry

-- Same type quadratic surds imply given m values
theorem quadratic_surds_same_type (m : ℝ) (h : real.sqrt (m^2 - 6) = real.sqrt (6 * m + 1)) :
  m = 7 :=
sorry

end algebraic_expressions_additive_inverse_quadratic_surds_same_type_l503_503114


namespace unit_digit_div_l503_503115

theorem unit_digit_div (n : ℕ) : (33 * 10) % (2 ^ 1984) = n % 10 :=
by
  have h := 2 ^ 1984
  have u_digit_2_1984 := 6 -- Since 1984 % 4 = 0, last digit in the cycle of 2^n for n ≡ 0 [4] is 6
  sorry
  
example : (33 * 10) / (2 ^ 1984) % 10 = 6 :=
by sorry

end unit_digit_div_l503_503115


namespace atomic_weight_Ca_l503_503581

def molecular_weight_CaH2 : ℝ := 42
def atomic_weight_H : ℝ := 1.008

theorem atomic_weight_Ca : atomic_weight_H * 2 < molecular_weight_CaH2 :=
by sorry

end atomic_weight_Ca_l503_503581


namespace final_number_is_odd_l503_503459

theorem final_number_is_odd :
  (∃ n : ℕ, n = 1 ∧ n % 2 = 1) ∧
  (∀ m : ℕ, (m = 2 → m - 2 ∈ (1, 2, ..., 1978))) :=
by
  -- Initial condition: numbers 1 to 1978
  let numbers : List ℕ := List.range 1978 |>.map (. + 1)
  
  -- Induction hypothesis: the count of odd numbers is always odd
  induction numbers using nat.strong_induction_on with k hk
  exact
    -- Base case: since there are 989 odd numbers, final number has to be odd
    sorry

end final_number_is_odd_l503_503459


namespace range_of_a_l503_503053

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem range_of_a (a : ℝ) (h₁ : 0 ≤ a) (h₂ : a ∈ set.Icc 2 4) : 
  set.range (λ x, f x) = set.Icc (-4 : ℝ) (0 : ℝ) :=
sorry

end range_of_a_l503_503053


namespace closest_to_100000_l503_503153

theorem closest_to_100000 (a b c d : ℕ) (h1 : a = 100260) (h2 : b = 99830) (h3 : c = 98900) (h4 : d = 100320) :
  (∀ x ∈ {a, b, c, d}, |100000 - x| ≥ |100000 - b|) :=
by
  sorry

end closest_to_100000_l503_503153


namespace remaining_two_odds_probability_l503_503151

-- Define the set of numbers as a list
def number_set : List ℕ := [1, 2, 3, 4, 5]

-- Define the function to compute n choose k
def choose (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Calculate combinations and probability
def probability_of_two_remaining_odds : ℚ :=
  let total_outcomes := choose 5 3
  let favorable_outcomes := choose 2 2 * choose 3 1
  favorable_outcomes.to_rat / total_outcomes.to_rat

-- Statement of the problem
theorem remaining_two_odds_probability : 
  probability_of_two_remaining_odds = 0.3 := 
sorry

end remaining_two_odds_probability_l503_503151


namespace max_volume_cube_in_sphere_l503_503845

theorem max_volume_cube_in_sphere {R : ℝ} (hR : R > 0) :
  ∃(V : ℝ), V = (8 * real.sqrt 3 * R^3) / 9 := sorry

end max_volume_cube_in_sphere_l503_503845


namespace round_robin_tournament_matches_l503_503909

theorem round_robin_tournament_matches (n : ℕ) (h : n = 10) :
  let matches := n * (n - 1) / 2
  matches = 45 :=
by
  intros
  rw [h]
  dsimp
  norm_num
  sorry

end round_robin_tournament_matches_l503_503909


namespace range_of_m_in_first_quadrant_l503_503621

noncomputable def range_m (m : ℝ) : Prop :=
  (m^2 + m - 1 > 0) ∧ (4 * m^2 - 8 * m + 3 < 0)

theorem range_of_m_in_first_quadrant :
  set_of range_m = { m : ℝ | (↑(-1 + real.sqrt 5) / 2 : ℝ) < m ∧ m < ↑(3 / 2 : ℝ) } :=
sorry

end range_of_m_in_first_quadrant_l503_503621


namespace charles_total_earnings_l503_503543

def charles_earnings (house_rate dog_rate : ℝ) (house_hours dog_count dog_hours : ℝ) : ℝ :=
  (house_rate * house_hours) + (dog_rate * dog_count * dog_hours)

theorem charles_total_earnings :
  charles_earnings 15 22 10 3 1 = 216 := by
  sorry

end charles_total_earnings_l503_503543


namespace final_percentage_is_approx_58_l503_503505

-- Define the conditions for the problem
def amount_60_solution := 26 -- gallons
def percentage_60_solution := 0.60
def total_solution := 39 -- gallons
def amount_54_solution := total_solution - amount_60_solution -- 13 gallons
def percentage_54_solution := 0.54

-- Define the total antifreeze content
def total_antifreeze := (percentage_60_solution * amount_60_solution) + (percentage_54_solution * amount_54_solution)

-- Calculate the final percentage of the antifreeze solution
def final_percentage := (total_antifreeze / total_solution) * 100

-- Prove that the final_percentage is approximately 58.0%
theorem final_percentage_is_approx_58 :
  final_percentage ≈ 58.0 := -- Use RealApproxEqual operator to denote approximation
by
  sorry

end final_percentage_is_approx_58_l503_503505


namespace angle_ABF_45_degrees_l503_503001

theorem angle_ABF_45_degrees
  (A B C D E F : Point)
  (hABC : right_triangle A B C)
  (hAD_perp_BC : Perpendicular A D B C)
  (hAD_eq_DE : distance A D = distance D E)
  (hEF_perp_BC : Perpendicular E F B C)
  (hF_on_AC : OnLineSegment F A C) :
  angle A B F = 45 := 
sorry

end angle_ABF_45_degrees_l503_503001


namespace hyperbola_imaginary_axis_twice_real_axis_l503_503677

theorem hyperbola_imaginary_axis_twice_real_axis (m : ℝ) :
  (∃ a b : ℝ, m < 0 ∧ a = 1 ∧ b = 2 ∧ (mx^2 + y^2 = 1) = (- (x^2) / (b^2) + (y^2) / (a^2) = 1) )
  → m = -1/4 :=
by
  -- Applying the conditions:
  assume h,
  -- Detailed proof would go here
  sorry

end hyperbola_imaginary_axis_twice_real_axis_l503_503677


namespace john_calculation_correct_l503_503345

theorem john_calculation_correct (x y : ℕ) (h1 : x = 125) (h2 : y = 384) :
  let product := x * y in
  let correct_product := (product : ℝ) / 10^5 in
  correct_product = 0.48 :=
by
  sorry

end john_calculation_correct_l503_503345


namespace zero_points_count_l503_503612

theorem zero_points_count 
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_symm : ∀ x, f (5 + x) = f (5 - x))
  (h_zero : f 1 = 0)
  (h_unique : ∀ x, 0 ≤ x ∧ x ≤ 5 → f x = 0 → x = 1) : 
  let zero_count : ℕ := List.length ((List.range (2 * 2013 + 1)).filter (λ k, f (-2013 + k) = 0)) 
  in zero_count = 806 :=
by
  sorry

end zero_points_count_l503_503612


namespace installment_value_approx_l503_503049

-- Define the given conditions
def price_of_tv : ℝ := 15000
def total_installments : ℕ := 20
def interest_rate : ℝ := 0.06
def first_installment_paid := true
def last_installment : ℝ := 13000

-- Define the function to calculate the installment value
def installment_value (I : ℝ) : ℝ :=
  let unpaid_balance_avg := price_of_tv / 2
  let interest := unpaid_balance_avg * interest_rate
  let total_amount := price_of_tv + interest
  total_amount - last_installment - I * 19

-- Define approximation tolerance
def tolerance : ℝ := 1

-- Lean 4 theorem statement to prove that the approximate value of each installment (excluding the last one) is Rs. 129.
theorem installment_value_approx : 
  ∃ I : ℝ, abs (I - 129) < tolerance ∧ installment_value I = 0 :=
begin
  existsi 129,
  split,
  { simp,
    norm_num,
  },
  { sorry }
end

end installment_value_approx_l503_503049


namespace line_of_intersection_l503_503876

theorem line_of_intersection :
  ∀ (x y z : ℝ),
    (3 * x + 4 * y - 2 * z + 1 = 0) ∧ (2 * x - 4 * y + 3 * z + 4 = 0) →
    (∃ t : ℝ, x = -1 + 4 * t ∧ y = 1 / 2 - 13 * t ∧ z = -20 * t) :=
by
  intro x y z
  intro h
  cases h
  sorry

end line_of_intersection_l503_503876


namespace value_of_expression_l503_503869

theorem value_of_expression (b : ℝ) (h : b = 3) : 
    (3 * b^(-2) + (b^(-2) / 3)) / b^2 = 10 / 243 :=
by
  sorry

end value_of_expression_l503_503869


namespace largest_c_value_l503_503579

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 3 * x + c

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, f x c = -2) ↔ c ≤ 1/4 := by
sorry

end largest_c_value_l503_503579


namespace partial_fraction_sum_l503_503540

theorem partial_fraction_sum :
  (∑ k in Finset.range 99, 1 / ((k + 1) * (k + 2) : ℝ)) = 0.99 := 
by 
  sorry

end partial_fraction_sum_l503_503540


namespace sum_of_interior_angles_l503_503797

theorem sum_of_interior_angles (ext_angle : ℝ) (h : ext_angle = 20) : 
  let n := 360 / ext_angle in
  let int_sum := 180 * (n - 2) in
  int_sum = 2880 := 
by 
  sorry

end sum_of_interior_angles_l503_503797


namespace sum_of_sequence_l503_503539

theorem sum_of_sequence : 
  (∑ i in (Finset.range 10).map (λ i, i + 1) , (i : ℚ) / 20) + (100 / 20) = 7.75 := 
by
  sorry

end sum_of_sequence_l503_503539


namespace trig_expression_identity_l503_503946

theorem trig_expression_identity :
  sin (14 * Real.pi / 3) + cos (-25 * Real.pi / 4) = (Real.sqrt 3 + Real.sqrt 2) / 2 :=
by
  sorry

end trig_expression_identity_l503_503946


namespace eccentricity_of_hyperbola_l503_503274

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
  (F1 F2 : ℝ × ℝ) (hF1 : F1 = (-c, 0)) (hF2 : F2 = (c, 0))
  (line : ℝ × ℝ → ℝ × ℝ → Prop) (hline : line (a, 0) ((a + b) / 2, (a * b - b^2) / (2 * a)))
  (distance : ℝ → ℝ) (h_distance : distance 0 = sqrt 3 * c / 4)
  : ℝ :=
sqrt (1 + (b^2 / a^2))

theorem eccentricity_of_hyperbola (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
  (F1 F2 : ℝ × ℝ) (hF1 : F1 = (-c, 0)) (hF2 : F2 = (c, 0))
  (line : ℝ × ℝ → ℝ × ℝ → Prop) (hline : line (a, 0) ((a + b) / 2, (a * b - b^2) / (2 * a)))
  (distance : ℝ → ℝ) (h_distance : distance 0 = sqrt 3 * c / 4) :
  hyperbola_eccentricity a b c h1 h2 F1 hF1 F2 hF2 line hline distance h_distance = 2 * sqrt 3 / 3 :=
sorry

end eccentricity_of_hyperbola_l503_503274


namespace value_of_a_l503_503283

-- Define the sets A and B
def A : set (ℝ × ℝ) := { p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ y = 2 * x + 1 }
def B : set (ℝ × ℝ) := { p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ y = x + 3 }

-- Define the proposition to prove
theorem value_of_a (a : ℝ × ℝ) (hA : a ∈ A) (hB : a ∈ B) : a = (2, 5) :=
sorry

end value_of_a_l503_503283


namespace no_rational_solution_l503_503016

/-- Prove that the only rational solution to the equation x^3 + 3y^3 + 9z^3 = 9xyz is x = y = z = 0. -/
theorem no_rational_solution : ∀ (x y z : ℚ), x^3 + 3 * y^3 + 9 * z^3 = 9 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro x y z h
  sorry

end no_rational_solution_l503_503016


namespace factory_toys_production_l503_503131

theorem factory_toys_production
  (line_a_toys_per_day : ℕ)
  (line_b_toys_per_day : ℕ)
  (line_c_toys_per_day : ℕ)
  (days_per_week : ℕ)
  (weekly_production : line_a_toys_per_day * days_per_week + line_b_toys_per_day * days_per_week + line_c_toys_per_day * days_per_week = 27500) :
  line_a_toys_per_day = 1500 → 
  line_b_toys_per_day = 1800 →
  line_c_toys_per_day = 2200 →
  days_per_week = 5 →
  weekly_production = 27500 :=
by
  intros
  sorry

end factory_toys_production_l503_503131


namespace range_of_a_l503_503629

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + Real.exp x - 1 / Real.exp x

theorem range_of_a (a : ℝ) (h : f(a-1) + f(2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l503_503629


namespace sum_of_interior_angles_of_regular_polygon_l503_503790

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : n = 360 / 20) :
  (∑ i in finset.range n, 180 - 360 / n) = 2880 :=
by
  sorry

end sum_of_interior_angles_of_regular_polygon_l503_503790


namespace vitya_convinced_of_12_models_l503_503080

noncomputable def min_offers_needed (n : ℕ) (k : ℕ) : ℕ :=
  if h : n = 13 then
    let ln100 := Real.log 100
    let ln13 := Real.log 13
    let ln12 := Real.log 12
    let req_k := Real.log 100 / (Real.log 13 - Real.log 12)
    if k > req_k then k else req_k.toNat + 1
  else k

theorem vitya_convinced_of_12_models (k : ℕ) : ∀ n, (n >= 13) → (min_offers_needed n k > 58) :=
by
  intros n h
  apply sorry

end vitya_convinced_of_12_models_l503_503080


namespace isosceles_triangle_area_l503_503844

theorem isosceles_triangle_area 
  (P Q R : Type) [Triangle P Q R]
  (h1 : PQ = 13)
  (h2 : PR = 13)
  (h3 : QR = 24) :
  area P Q R = 60 :=
sorry

end isosceles_triangle_area_l503_503844


namespace eric_six_digit_code_l503_503563

theorem eric_six_digit_code :
  let digits := {1, 2, 3, 4, 5, 6}
  let even_digits := {2, 4, 6}
  let odd_digits := {1, 3, 5}
  ∃ code : vector ℕ 6,
    (∀ i, code.nth i ∈ digits) ∧
    (∀ i, i < 5 → (code.nth i ∈ even_digits ↔ code.nth (i+1) ∈ odd_digits)) ∧
    (∀ i, i < 5 → (code.nth i ∈ odd_digits ↔ code.nth (i+1) ∈ even_digits)) ∧
    (∑ c in (even_digits.product odd_digits).product_digits 6, 1) +
    (∑ c in (odd_digits.product even_digits).product_digits 6, 1) = 1458 := sorry

end eric_six_digit_code_l503_503563


namespace speed_of_current_is_2_l503_503901

noncomputable def speed_current : ℝ :=
  let still_water_speed := 14  -- kmph
  let distance_m := 40         -- meters
  let time_s := 8.9992800576   -- seconds
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let downstream_speed := distance_km / time_h
  downstream_speed - still_water_speed

theorem speed_of_current_is_2 :
  speed_current = 2 :=
by
  sorry

end speed_of_current_is_2_l503_503901


namespace true_propositions_l503_503242

variables (m n : ℝ^3 → Prop) (α β : ∀ {P : Prop}, Prop)

-- Assume m and n are distinct lines
axiom distinct_lines : m ≠ n

-- Assume α and β are non-coincident planes
axiom non_coincident_planes : α ≠ β

-- Proposition 1 conditions
def prop1 : Prop :=
  α = β ∧ (∀ x, m x → α x) ∧ (∀ x, n x → β x) → (∀ x, m x → n x)

-- Proposition 2 conditions
def prop2 : Prop :=
  (∀ x, m x → α x) ∧ (∀ x, n x → α x) ∧ (∀ x, m x → β x) ∧ (∀ x, n x → β x) → α = β

-- Proposition 3 conditions
def prop3 : Prop :=
  (∀ x, m x → ¬ α x) ∧ (∀ x, n x → ¬ β x) ∧ (∀ x, m x → n x) → α = β

-- Proposition 4 conditions
def prop4 : Prop :=
  (¬ ∃ x, m x ∧ n x) ∧ (∀ x, m x → α x) ∧ (∀ x, m x → β x) ∧ (∀ x, n x → α x) ∧ (∀ x, n x → β x) → α = β

theorem true_propositions :
  prop3 m n α β ∧ prop4 m n α β :=
by
  sorry

end true_propositions_l503_503242


namespace flooring_cost_l503_503041

theorem flooring_cost (L W : ℝ) (r : ℝ) (Cost : ℝ) 
    (hL : L = 5.5) (hW : W = 3.75) (hr : r = 600) 
    (hCost : Cost = L * W * r) : Cost = 12375 := 
by
    rw [hL, hW, hr, hCost]
    norm_num
    sorry

end flooring_cost_l503_503041


namespace max_area_region_T_l503_503321

open Real
open Set

/-- Define the circles with specified radii -/
def circle (r : ℝ) : Set (ℝ × ℝ) := {p | sqrt (p.1 ^ 2 + p.2 ^ 2) < r}

/-- The total area function for non-overlapping circles -/
def circle_area (r : ℝ) : ℝ := π * r^2

/-- Define the region T as the union of points inside exactly one circle -/
def region_T (radii : List ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ r ∈ radii, p ∈ circle r ∧ ∀ r' ∈ radii, r ≠ r' → p ∉ circle r'}

def radii := [2, 4, 6, 8, 10]

/-- Compute the expected area of the union of circles -/
noncomputable def expected_area : ℝ := (circle_area 2) + (circle_area 4) + (circle_area 6) + (circle_area 8) + (circle_area 10)

theorem max_area_region_T : expected_area = 220 * π :=
  by
    -- Axioms or assumptions of the problem can be placed here
  sorry

end max_area_region_T_l503_503321


namespace roots_of_quadratic_eq_l503_503808

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l503_503808


namespace original_price_l503_503155

theorem original_price (P : ℝ) (h1 : 0.76 * P = 820) : P = 1079 :=
by
  sorry

end original_price_l503_503155


namespace backyard_area_l503_503745

theorem backyard_area {length perimeter : ℝ} 
  (h1 : 30 * length = 1200) 
  (h2 : 12 * perimeter = 1200) : 
  (let width := (perimeter - 2 * length) / 2
   in length * width = 400) :=
by 
  sorry

end backyard_area_l503_503745


namespace vitya_convinced_of_12_models_l503_503078

noncomputable def min_offers_needed (n : ℕ) (k : ℕ) : ℕ :=
  if h : n = 13 then
    let ln100 := Real.log 100
    let ln13 := Real.log 13
    let ln12 := Real.log 12
    let req_k := Real.log 100 / (Real.log 13 - Real.log 12)
    if k > req_k then k else req_k.toNat + 1
  else k

theorem vitya_convinced_of_12_models (k : ℕ) : ∀ n, (n >= 13) → (min_offers_needed n k > 58) :=
by
  intros n h
  apply sorry

end vitya_convinced_of_12_models_l503_503078


namespace num_pairs_satisfying_equation_l503_503557

-- Define the given problem conditions in Lean
def satisfies_equation (x y : ℕ) : Prop :=
  x^2 + y^2 = x^5

-- Define the theorem as per the equivalent problem
theorem num_pairs_satisfying_equation :
  (∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ satisfies_equation x y) = 2 :=
sorry

end num_pairs_satisfying_equation_l503_503557


namespace height_of_frustum_l503_503138

-- Definitions based on the given conditions
def cuts_parallel_to_base (height: ℕ) (ratio: ℕ) : ℕ := 
  height * ratio

-- Define the problem
theorem height_of_frustum 
  (height_smaller_pyramid : ℕ) 
  (ratio_upper_to_lower: ℕ) 
  (h : height_smaller_pyramid = 3) 
  (r : ratio_upper_to_lower = 4) 
  : (cuts_parallel_to_base 3 2) - height_smaller_pyramid = 3 := 
by
  sorry

end height_of_frustum_l503_503138


namespace smallest_number_remainder_l503_503862

open Nat

theorem smallest_number_remainder
  (b : ℕ)
  (h1 : b % 4 = 2)
  (h2 : b % 3 = 2)
  (h3 : b % 5 = 3) :
  b = 38 :=
sorry

end smallest_number_remainder_l503_503862


namespace marked_price_is_300_max_discount_is_50_l503_503143

-- Definition of the conditions given in the problem:
def loss_condition (x : ℝ) : Prop := 0.4 * x - 30 = 0.7 * x - 60
def profit_condition (x : ℝ) : Prop := 0.7 * x - 60 - (0.4 * x - 30) = 90

-- Statement for the first problem: Prove the marked price is 300 yuan.
theorem marked_price_is_300 : ∃ x : ℝ, loss_condition x ∧ profit_condition x ∧ x = 300 := by
  exists 300
  simp [loss_condition, profit_condition]
  sorry

noncomputable def max_discount (x : ℝ) : ℝ := 100 - (30 + 0.4 * x) / x * 100

def no_loss_max_discount (d : ℝ) : Prop := d = 50

-- Statement for the second problem: Prove the maximum discount is 50%.
theorem max_discount_is_50 (x : ℝ) (h_loss : loss_condition x) (h_profit : profit_condition x) : no_loss_max_discount (max_discount x) := by
  simp [max_discount, no_loss_max_discount]
  sorry

end marked_price_is_300_max_discount_is_50_l503_503143


namespace deriv_at_minus_one_l503_503632

variable {a b c : ℝ}

noncomputable def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

lemma deriv_of_poly (x : ℝ) : deriv (f x) = 4 * a * x^3 + 2 * b * x :=
begin
  sorry
end

theorem deriv_at_minus_one (h : deriv (f 1) = 2) : deriv (f (-1)) = -2 :=
begin
  have H1 : deriv (f 1) = 4 * a + 2 * b,
  { rw deriv_of_poly 1 },
  rw H1 at h,
  have : 4 * a + 2 * b = 2 := h,
  have H2 : deriv (f (-1)) = 4 * a * (-1)^3 + 2 * b * (-1),
  { rw deriv_of_poly (-1) },
  rw [pow_succ, pow_one, neg_one_mul, neg_one_mul, one_mul, neg_add_eq_sub, sub_neg_eq_add, ←neg_add, neg_one_mul, ←neg_eq_iff_add_eq_zero],
  exact eq.subst this.symm (neg_two_eq_iff_eq_add'.mpr this),
end

end deriv_at_minus_one_l503_503632


namespace find_a_and_circle_eq_l503_503239

-- Define the conditions
variable (a : ℝ)
def l1 (x y : ℝ) : Prop := (2 * a + 1) * x + 2 * y - a + 2 = 0
def l2 (x y : ℝ) : Prop := 2 * x - 3 * a * y - 3 * a - 5 = 0
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 9 = 0

-- Auxiliary statements
def perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

theorem find_a_and_circle_eq (h_perp : perpendicular (- (2 * a + 1) / 2) (2 / (3 * a)))
  (a_eq : a = 1)
  (h_l1_perp_l2 : ∀ x y, l1 a x y → l2 x y → x = 1 ∧ y = -2) :
  ∃ (x0 y0 r : ℝ), l1 a x0 y0 ∧ l2 x0 y0 ∧ line 1 (-2) / r = 4
  ∧ ((x - 1)^2 + (y + 2)^2 = 16) := sorry

end find_a_and_circle_eq_l503_503239


namespace base9_multiplication_l503_503583

theorem base9_multiplication :
  ∃ x : ℕ, x = nat.of_digits 9 [3, 2, 7] * nat.of_digits 9 [3] ∧ nat.to_digits 9 x = [1, 0, 8, 3] :=
by
  sorry

end base9_multiplication_l503_503583


namespace range_of_constant_c_in_quadrant_I_l503_503260

theorem range_of_constant_c_in_quadrant_I (c : ℝ) (x y : ℝ)
  (h1 : x - 2 * y = 4)
  (h2 : 2 * c * x + y = 5)
  (hx_pos : x > 0)
  (hy_pos : y > 0) : 
  -1 / 4 < c ∧ c < 5 / 8 := 
sorry

end range_of_constant_c_in_quadrant_I_l503_503260


namespace smaller_value_r_plus_s_l503_503733

theorem smaller_value_r_plus_s :
  ∃ (x : ℝ) (r s : ℤ), (∃ (a b : ℝ), a = x^(1/3) ∧ b = (30 - x)^(1/3) ∧ a + b = 3 ∧ a^3 + b^3 = 30)
  ∧ x = r - sqrt (s : ℝ) ∧ r + s = 96 :=
sorry

end smaller_value_r_plus_s_l503_503733


namespace median_of_fifteen_is_eight_l503_503478

def median_of_first_fifteen_positive_integers : ℝ :=
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median_pos := (list.length lst + 1) / 2  
  lst.get (median_pos - 1)

theorem median_of_fifteen_is_eight : median_of_first_fifteen_positive_integers = 8.0 := 
  by 
    -- Proof omitted    
    sorry

end median_of_fifteen_is_eight_l503_503478


namespace boat_shipments_divisor_l503_503454

/-- 
Given:
1. There exists an integer B representing the number of boxes that can be divided into S equal shipments by boat.
2. B can be divided into 24 equal shipments by truck.
3. The smallest number of boxes B is 120.
Prove that S, the number of equal shipments by boat, is 60.
--/
theorem boat_shipments_divisor (B S : ℕ) (h1 : B % S = 0) (h2 : B % 24 = 0) (h3 : B = 120) : S = 60 := 
sorry

end boat_shipments_divisor_l503_503454


namespace cosine_of_half_pi_minus_double_alpha_l503_503236

theorem cosine_of_half_pi_minus_double_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (π / 2 - 2 * α) = 4 / 5 :=
sorry

end cosine_of_half_pi_minus_double_alpha_l503_503236


namespace coefficients_sum_l503_503291

-- Define the polynomial (1 + 2x)(1 - 2x)^7
def polynomial (x : ℝ) := (1 + 2 * x) * (1 - 2 * x) ^ 7

-- Expand the polynomial and define the coefficients a_0, a_1, ..., a_8 in terms of ℝ
axiom coefficients : (ℝ → ℝ) → list ℝ
noncomputable def a : list ℝ := coefficients polynomial

-- Define the term we're interested in
def sum_of_coefficients_up_to_n : ℕ → list ℝ → ℝ
| 0     a := 0
| (n+1) a := a.head + sum_of_coefficients_up_to_n n a.tail

-- Statement to prove
theorem coefficients_sum :
  sum_of_coefficients_up_to_n 8 a = 253 := 
sorry

end coefficients_sum_l503_503291


namespace square_feet_per_acre_l503_503509

-- Define the conditions
def rent_per_acre_per_month : ℝ := 60
def total_rent_per_month : ℝ := 600
def length_of_plot : ℝ := 360
def width_of_plot : ℝ := 1210

-- Translate the problem to a Lean theorem
theorem square_feet_per_acre :
  (length_of_plot * width_of_plot) / (total_rent_per_month / rent_per_acre_per_month) = 43560 :=
by {
  -- skipping the proof steps
  sorry
}

end square_feet_per_acre_l503_503509


namespace milk_butterfat_problem_l503_503651

-- Define the values given in the problem
def b1 : ℝ := 0.35  -- butterfat percentage of initial milk
def v1 : ℝ := 8     -- volume of initial milk in gallons
def b2 : ℝ := 0.10  -- butterfat percentage of milk to be added
def bf : ℝ := 0.20  -- desired butterfat percentage of the final mixture

-- Define the proof statement
theorem milk_butterfat_problem :
  ∃ x : ℝ, (2.8 + 0.1 * x) / (v1 + x) = bf ↔ x = 12 :=
by {
  sorry
}

end milk_butterfat_problem_l503_503651


namespace problem_B_correct_l503_503101

def necessary_but_not_sufficient (k : ℤ) : Prop :=
  let x := (k * Real.pi) / 4
  (∃ k : ℤ, tan x = 1) ∧ ∀ k : ℤ, tan x ≠ -1

theorem problem_B_correct : ∀ (k : ℤ), necessary_but_not_sufficient k :=
by
  -- statement of theorem without its proof
  sorry

end problem_B_correct_l503_503101


namespace sum_of_possible_values_of_abs_w_minus_z_l503_503374

theorem sum_of_possible_values_of_abs_w_minus_z 
  (w x y z : ℝ)
  (h₁ : |w - x| = 3)
  (h₂ : |x - y| = 2)
  (h₃ : |y - z| = 5)
  (h₄ : w ≥ z) : 
  ({|w - z| | (|w - x| = 3) ∧ (|x - y| = 2) ∧ (|y - z| = 5) ∧ (w ≥ z)}.sum = 20) :=
sorry

end sum_of_possible_values_of_abs_w_minus_z_l503_503374


namespace perpendicular_MP_AD_l503_503112

variables (A B C D M K P : Point) (ω : Circle)

-- Given conditions
variables (h1 : CyclicQuadrilateral A B C D)
variables (M_midpoint : Midpoint M B C)
variables (h2 : Perpendicular (Line.mk M K) (Line.mk B C))
variables (h3 : IntersectsCircle (Circle.mk K C) (Seg.mk C D) P)
variables (P_not_C : P ≠ C)

-- Goal: Prove that lines MP and AD are perpendicular.
theorem perpendicular_MP_AD :
  Perpendicular (Line.mk M P) (Line.mk A D) :=
begin
  sorry
end

end perpendicular_MP_AD_l503_503112


namespace equidistant_point_unique_l503_503893

namespace CircleProblem

-- Define the circle
def circle (O : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) :=
  { p | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2 }

-- Point of interest
def point (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define tangents locations
def upper_tangent_y : ℝ := 8
def lower_tangent_y : ℝ := -12

-- Midline point definition
def midline_y : ℝ := (upper_tangent_y + lower_tangent_y) / 2 -- which is -2 in this context

-- Proposition stating that there is exactly one point equidistant from the given circle and the tangents
theorem equidistant_point_unique (O : ℝ × ℝ) (r : ℝ) :
  O = (0, 0) ∧ r = 5 → 
  upper_tangent_y = 8 ∧ lower_tangent_y = -12 →
  ∃! p : ℝ × ℝ, p.2 = midline_y ∧ dist p O = 5 + upper_tangent_y - midline_y :=
sorry

end CircleProblem

end equidistant_point_unique_l503_503893


namespace num_large_posters_l503_503919

-- Define the constants
def total_posters : ℕ := 50
def small_posters : ℕ := total_posters * 2 / 5
def medium_posters : ℕ := total_posters / 2
def large_posters : ℕ := total_posters - (small_posters + medium_posters)

-- Theorem to prove the number of large posters
theorem num_large_posters : large_posters = 5 :=
by
  sorry

end num_large_posters_l503_503919


namespace circle_equation_l503_503624

theorem circle_equation (a : ℤ) (h : a < 2): 
  ∃ x y : ℝ, (x - 1)^2 + (y + 3)^2 = 10 - 5 * a :=
begin
  use [1, -3],  -- Provide specific x and y that satisfy the equation for particular integer a = 1
  subst h,
  sorry
end

end circle_equation_l503_503624


namespace journey_time_ratio_l503_503507

theorem journey_time_ratio (D : ℝ) (hD_pos : D > 0) :
  let T1 := D / 45
  let T2 := D / 30
  (T2 / T1) = (3 / 2) := 
by
  sorry

end journey_time_ratio_l503_503507


namespace zero_in_interval_l503_503448

noncomputable def f : ℝ → ℝ := λ x, 2^x + 2 * x - 6

theorem zero_in_interval : ∃ c ∈ Ioo 1 2, f c = 0 :=
by {
  have h1 : f 1 < 0,
  { sorry },
  have h2 : f 2 > 0,
  { sorry },
  have continuous_f : continuous f,
  { sorry },
  exact intermediate_value_Ioo 1 2 h1 h2 continuous_f,
}

end zero_in_interval_l503_503448


namespace product_value_l503_503380

variables {A B R M L : ℝ}

theorem product_value (h1 : log10 (A * B * L) + log10 (A * B * M) = 3)
                      (h2 : log10 (B * M * L) + log10 (B * M * R) = 4)
                      (h3 : log10 (R * A * B) + log10 (R * B * L) = 5) :
  A * B * R * M * L = 10^6 :=
sorry

end product_value_l503_503380


namespace non_arithmetic_sequence_l503_503056

theorem non_arithmetic_sequence (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) :
    (∀ n, S_n n = n^2 + 2 * n - 1) →
    (∀ n, a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1)) →
    ¬(∀ d, ∀ n, a_n (n+1) = a_n n + d) :=
by
  intros hS ha
  sorry

end non_arithmetic_sequence_l503_503056


namespace find_annual_income_l503_503821

noncomputable def annual_income (q : ℝ) : ℝ :=
  let I := 36000
  -- Condition 1 and 2: Tax rate split at $30000
      T := (0.01 * q) * 30000 + (0.01 * (q + 3)) * (I - 30000)
  -- Condition 3: Total tax paid is $4500
      total_tax := 4500
  -- Condition 4: Total tax as percentage of income
      percentage_tax := 0.01 * (q + 0.5) * I
  T

theorem find_annual_income (q : ℝ) : annual_income q = 4500 → 0.01 * (q + 0.5) * 36000 = 4500 :=
by
  intro h1
  apply congr
  exact h1||
  sorry

end find_annual_income_l503_503821


namespace cone_surface_area_l503_503309

theorem cone_surface_area (slant_height : ℝ) (lateral_surface_is_semicircle : bool) (radius : ℝ)
  (h1 : slant_height = 2)
  (h2 : lateral_surface_is_semicircle = true)
  (h3 : 2 * real.pi * radius = 2 * real.pi) :
  pi * radius * slant_height + pi * radius ^ 2 = 3 * pi :=
by {
  -- Extract radius from the given relationship
  have r_eq : radius = 1, from (eq_of_mul_eq_mul_left (ne_of_gt real.pi_pos) h3).mpr,
  -- Substitute the obtained radius into the surface area formula
  rw [r_eq, h1],
  -- Simplify the surface area formula
  ring
}

end cone_surface_area_l503_503309


namespace division_theorem_l503_503592

noncomputable def f : ℤ[X] := X^13 - X^5 + 90
noncomputable def g : ℤ[X] := X^2 - X + 2

theorem division_theorem : g ∣ f := sorry

end division_theorem_l503_503592


namespace backpack_pencil_case_combinations_l503_503702

theorem backpack_pencil_case_combinations (backpacks pencil_cases : Fin 2) : 
  (backpacks * pencil_cases) = 4 :=
by 
  sorry

end backpack_pencil_case_combinations_l503_503702


namespace find_unique_k_l503_503514

-- Define constants and conditions based on the problem
constant num_dalmatians : ℕ := 101
constant voting_probability : ℚ := 1 / 2

-- Definitions used in expectation calculations
 -- Here we use rational numbers to safely handle the probabilities involved.
noncomputable def expected_X_squared : ℚ := 
  let i_square := (voting_probability)
  let i_cross_term := (voting_probability * (1 + (num_dalmatians.choose num_dalmatians/2)) / 2^(num_dalmatians + 1))
  (num_dalmatians : ℚ) * i_square + 2 * (num_dalmatians.choose 2 * i_cross_term)

noncomputable def a_b : (ℚ × ℚ) := sorry -- Substitute this with actual calculations.

constant a : ℤ := a_b.fst.num       -- Numerator of expected value
constant b : ℤ := a_b.snd.denom   -- Denominator of expected value

-- Given expected value's numerator and denominator are coprime
axiom gcd_a_b : (a : ℤ).gcd b = 1

-- The target statement where k is the unique positive integer ≤ 103 such that 103 | a - b * k
theorem find_unique_k : ∃! k : ℕ, k ≤ 103 ∧ 103 ∣ a - b * k := sorry

end find_unique_k_l503_503514


namespace triangle_ABC_properties_l503_503622

open Real

theorem triangle_ABC_properties
  (a b c : ℝ) 
  (A B C : ℝ) 
  (A_eq : A = π / 3) 
  (b_eq : b = sqrt 2) 
  (cond1 : b^2 + sqrt 2 * a * c = a^2 + c^2) 
  (cond2 : a * cos B = b * sin A) 
  (cond3 : sin B + cos B = sqrt 2) : 
  B = π / 4 ∧ (1 / 2) * a * b * sin (π - A - B) = (3 + sqrt 3) / 4 := 
by 
  sorry

end triangle_ABC_properties_l503_503622


namespace trigonometric_identities_l503_503235

theorem trigonometric_identities
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (sinα : Real.sin α = 4 / 5)
  (cosβ : Real.cos β = 12 / 13) :
  Real.sin (α + β) = 63 / 65 ∧ Real.tan (α - β) = 33 / 56 := by
  sorry

end trigonometric_identities_l503_503235


namespace GIMPS_meaning_l503_503462

/--
  Curtis Cooper's team discovered the largest prime number known as \( 2^{74,207,281} - 1 \), which is a Mersenne prime.
  GIMPS stands for "Great Internet Mersenne Prime Search."

  Prove that GIMPS means "Great Internet Mersenne Prime Search".
-/
theorem GIMPS_meaning : GIMPS = "Great Internet Mersenne Prime Search" :=
  sorry

end GIMPS_meaning_l503_503462


namespace vitya_needs_58_offers_l503_503085

noncomputable def smallest_integer_k (P : ℝ → ℝ) : ℝ :=
  if H : ∃ k, k > P (100), then classical.some H else 0

theorem vitya_needs_58_offers :
  ∀ n : ℕ, n ≥ 13 → 
  (12:ℝ/13:ℝ) ^ smallest_integer_k (fun x => Real.log x / (Real.log 13 - Real.log 12)) < 0.01 :=
begin
  assume n h,
  rw smallest_integer_k,
  split_ifs,
  { sorry }, -- proof would go here
  { exfalso, exact sorry }, -- no proof steps provided
end

end vitya_needs_58_offers_l503_503085


namespace rug_area_calculation_l503_503904

theorem rug_area_calculation (length_floor width_floor strip_width : ℕ)
  (h_length : length_floor = 10)
  (h_width : width_floor = 8)
  (h_strip : strip_width = 2) :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := by
  sorry

end rug_area_calculation_l503_503904


namespace arithmetic_mean_of_lambda_l503_503644

def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2020}

def λ (A : Set ℕ) : ℕ := Nat.max (A.toFinset.max' (by sorry)) + Nat.min (A.toFinset.min' (by sorry))

theorem arithmetic_mean_of_lambda :
  let S := { A : Set ℕ | A ⊆ M ∧ A ≠ ∅ } in
  (∑ A in S.toFinset, λ A) / S.toFinset.card = 2021 :=
sorry

end arithmetic_mean_of_lambda_l503_503644


namespace male_percentage_l503_503696

theorem male_percentage (total_employees : ℕ)
  (males_below_50 : ℕ)
  (percentage_males_at_least_50 : ℝ)
  (male_percentage : ℝ) :
  total_employees = 2200 →
  males_below_50 = 616 →
  percentage_males_at_least_50 = 0.3 → 
  male_percentage = 40 :=
by
  sorry

end male_percentage_l503_503696


namespace sequence_product_2017_l503_503280

noncomputable def sequence (n : ℕ) : ℚ :=
if n = 0 then 2
else let a : ℕ → ℚ := λ n, if n = 1 then 2 else 
  match n with
  | Nat.succ n' => (1 + sequence n') / (1 - sequence n')
  end
a n

theorem sequence_product_2017 : (∏ i in Finset.range 2017.succ, sequence i) = 2 := 
sorry

end sequence_product_2017_l503_503280


namespace infinite_series_sum_l503_503183

theorem infinite_series_sum :
  let S := ∑' n : ℕ, (ite (n % 4 = 0) (1 / (n * n)) (if n % 2 = 0 then -(1 / (n * n)) else (-1 / (n * n * n))))
  in S = 0.470275 :=
begin
  sorry
end

end infinite_series_sum_l503_503183


namespace problem_statement_l503_503279

def p := ∃ x : ℝ, x^2 < 0
def q := ∀ x : ℝ, x > 2 → log (1/2) x < 0

theorem problem_statement : ¬ p ∧ q :=
by
  sorry

end problem_statement_l503_503279


namespace constant_term_in_binomial_expansion_l503_503822

theorem constant_term_in_binomial_expansion :
  let n := 4
  let P := 4^n
  let Q := 2^n
  P + Q = 272 → 
  ∃ r, (C(4, r) * 3^(4-r)) = 108 ∧ (4 - 4 * r) / 3 = 0 := 
begin
  sorry
end

end constant_term_in_binomial_expansion_l503_503822


namespace factory_production_eq_l503_503508

theorem factory_production_eq (x : ℝ) (h1 : x > 50) : 450 / (x - 50) - 400 / x = 1 := 
by 
  sorry

end factory_production_eq_l503_503508


namespace find_range_m_l503_503641

noncomputable def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, m * x₀^2 + 1 < 1

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem find_range_m (m : ℝ) : ¬ (p m ∨ ¬ q m) ↔ -2 ≤ m ∧ m ≤ 2 :=
  sorry

end find_range_m_l503_503641


namespace correct_statement_l503_503102

theorem correct_statement :
  (∀ x k : ℤ, x = (k * π / 4) → tan x = 1 → x = (4 * k + 1) * π / 4) ∧
  ¬ (∀ x k : ℤ, (x = (k * π / 4) → (x = (4 * k + 1) * π / 4))) :=
begin
  sorry
end

end correct_statement_l503_503102


namespace sandy_books_second_shop_l503_503409

theorem sandy_books_second_shop (x : ℕ) (h1 : 65 = 1080 / 16) 
                                (h2 : x * 16 = 840) 
                                (h3 : (1080 + 840) / 16 = 120) : 
                                x = 55 :=
by
  sorry

end sandy_books_second_shop_l503_503409


namespace quadratic_roots_relation_l503_503590

noncomputable def roots_relation (a b c : ℝ) : Prop :=
  ∃ α β : ℝ, (α * β = c / a) ∧ (α + β = -b / a) ∧ β = 3 * α

theorem quadratic_roots_relation (a b c : ℝ) (h : roots_relation a b c) : 3 * b^2 = 16 * a * c :=
by
  sorry

end quadratic_roots_relation_l503_503590


namespace mn_parallel_to_bases_l503_503340

variables {A B C D M N : Type} [ordered_comm_ring A]
variables (AD BC AM CN BM DN : A)
variables (trapezoid_ABCD : A) (cyclic_AMND cyclic_BMNC : Prop)

-- Given that the quadrilaterals AMND and BMNC are cyclic
-- And given that the trapezoid is ABCD with bases AD and BC
-- And given that AM = CN and BM = DN
-- Prove that MN is parallel to the bases AD and BC

theorem mn_parallel_to_bases 
  (h1: trapezoid_ABCD)
  (h2: cyclic_AMND)
  (h3: cyclic_BMNC)
  (h4: AM = CN)
  (h5: BM = DN) : 
  (∃MN : A, parallel MN AD ∧ parallel MN BC) := 
sorry

end mn_parallel_to_bases_l503_503340


namespace initial_num_nuts_l503_503063

theorem initial_num_nuts (total_nuts : ℕ) (h1 : 1/6 * total_nuts = 5) : total_nuts = 30 := 
sorry

end initial_num_nuts_l503_503063


namespace general_term_inequality_l503_503643

-- Define the sequence
def a : ℕ → ℚ
| 0       := 3 / 2
| (n + 1) := 3 * (n + 1) * a n / (2 * a n + n)

-- Prove the general term of the sequence
theorem general_term (n : ℕ) : a n = n / (1 - 1 / 3 ^ n) := by sorry

-- Prove the inequality for the sequence product
theorem inequality (n : ℕ) : ∏ i in (finset.range n).succ, a i < 2 * n.factorial := by sorry

end general_term_inequality_l503_503643


namespace calculate_expression_l503_503010

theorem calculate_expression 
  : (- 21 * (2 : ℚ)/ 3 + 3 * (1 : ℚ) / 4 - (-2 / 3) - (+1 / 4) = -18) := 
by 
  sorry

end calculate_expression_l503_503010


namespace integer_right_triangle_non_grid_aligned_l503_503948

theorem integer_right_triangle_non_grid_aligned :
  ∃ A B C : ℤ × ℤ,
    -- Define the vertices
    A = (0, 0) ∧ B = (12, 16) ∧ C = (-12, 9) ∧
    -- Verify side slopes are not integers
    let m_AB := (B.2 - A.2) / (B.1 - A.1),
        m_AC := (C.2 - A.2) / (C.1 - A.1),
        m_BC := (C.2 - B.2) / (C.1 - B.1) in
    (m_AB ∉ ℤ) ∧ (m_AC ∉ ℤ) ∧ (m_BC ∉ ℤ) := by
    let A := (0, 0)
    let B := (12, 16)
    let C := (-12, 9)
    let m_AB := (B.2 - A.2) / (B.1 - A.1)
    let m_AC := (C.2 - A.2) / (C.1 - A.1)
    let m_BC := (C.2 - B.2) / (C.1 - B.1)
    have h_AB : m_AB = (16 - 0) / (12 - 0) := rfl
    have h_AC : m_AC = (9 - 0) / (-12 - 0) := rfl
    have h_BC : m_BC = (9 - 16) / (-12 - 12) := rfl
    show (m_AB ∉ ℤ) ∧ (m_AC ∉ ℤ) ∧ (m_BC ∉ ℤ) from 
        by simp [m_AB, m_AC, m_BC]
    sorry

end integer_right_triangle_non_grid_aligned_l503_503948


namespace gasoline_price_equiv_l503_503510

theorem gasoline_price_equiv (x : ℝ) (P_0 P_1 P_2 P_3 P_4 P_5 : ℝ) : 
    P_0 = 100 ∧ 
    P_1 = P_0 * 1.30 ∧ 
    P_2 = P_1 * 0.75 ∧ 
    P_3 = P_2 * 1.20 ∧ 
    P_4 = P_3 * (1 - x / 100) ∧ 
    P_5 = P_4 * 1.15 ∧ 
    P_5 = 100
    → x ≈ 26 :=
by sorry

end gasoline_price_equiv_l503_503510


namespace curve_is_circle_l503_503577

theorem curve_is_circle (r θ : ℝ) (h : r = 4 * cot θ * csc θ) : 
  ∃ k : ℝ, (∀ x y : ℝ, (x ^ 2 + y ^ 2 = k ^ 2) ∧ (k > 0)) :=
by
  sorry

end curve_is_circle_l503_503577


namespace number_of_large_posters_is_5_l503_503922

theorem number_of_large_posters_is_5
  (total_posters : ℕ)
  (small_posters_ratio : ℚ)
  (medium_posters_ratio : ℚ)
  (h_total : total_posters = 50)
  (h_small_ratio : small_posters_ratio = 2 / 5)
  (h_medium_ratio : medium_posters_ratio = 1 / 2) :
  (total_posters * (1 - small_posters_ratio - medium_posters_ratio)) = 5 :=
by sorry

end number_of_large_posters_is_5_l503_503922


namespace symmetric_point_proof_l503_503709

def symmetric_point_xOy (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1, P.2, -P.3)

theorem symmetric_point_proof :
  symmetric_point_xOy (1, 2, -3) = (1, 2, 3) :=
by
  sorry

end symmetric_point_proof_l503_503709


namespace det_A2_B3_l503_503171

variable {A B : Matrix} -- Assuming Matrix is defined properly in Mathlib
variable (det_A : det A = 3) (det_B : det B = -2)

theorem det_A2_B3 : det (A^2 * B^3) = -72 :=
by 
  -- We skip the proof.
  sorry

end det_A2_B3_l503_503171


namespace diagonals_from_vertex_of_decagon_l503_503211

theorem diagonals_from_vertex_of_decagon :
  ∀ (n : ℕ), n = 10 → (n - 3) = 7 :=
by
  intros
  subst_vars
  calc
    10 - 3 = 7 : by sorry

end diagonals_from_vertex_of_decagon_l503_503211


namespace g_properties_l503_503217

noncomputable def g (ω x : ℝ) : ℝ := 2 * sin (ω * x + π / 12) * cos (ω * x + π / 12)

theorem g_properties (ω : ℝ) (hω : ω > 0) : 
  (g ω 0 = 1 / 2) ∧
  (∀ t : ℝ, -π/6 ≤ t ∧ t ≤ π/4 → 0 < ω ∧ ω ≤ 2/3) ∧
  (∀ x1 x2 : ℝ, g ω x1 = 1 ∧ g ω x2 = -1 ∧ |x1 - x2| = π → ω = 1 / 2) ∧
  (∃ ω : ℝ, 1 < ω ∧ ω < 3 ∧ ∀ x : ℝ, g ω (x - π/6) = g ω (π/6 - x)) :=
sorry

end g_properties_l503_503217


namespace equal_angles_l503_503130

variables {V : Type*} [normed_group V] [inner_product_space ℝ V]

def is_convex_hexagon (A B C D E F : V) : Prop :=
  -- Add appropriate convexity condition for the hexagon with vertices A, B, C, D, E, F
  sorry

def opposite_sides_condition (A B C D E F : V) : Prop :=
  ∀ (U V : V), 
    (U = A - B ∨ U = C - D ∨ U = E - F) ∧ 
    (V = D - E ∨ V = A - F ∨ V = B - C) →
      ∥(U + V) / 2∥ = (1/2) * real.sqrt(3) * (∥U∥ + ∥V∥)

theorem equal_angles 
  {A B C D E F : V}
  (h_convex : is_convex_hexagon A B C D E F)
  (h_opposite : opposite_sides_condition A B C D E F) :
  ∀ (θ : ℝ), angle A B C = θ ∧ angle B C D = θ ∧ angle C D E = θ ∧ 
              angle D E F = θ ∧ angle E F A = θ ∧ angle F A B = θ :=
sorry

end equal_angles_l503_503130


namespace verify_extending_points_l503_503741

noncomputable def verify_P_and_Q (A B P Q : ℝ → ℝ → ℝ) : Prop := 
  let vector_relation_P := P = - (2/5) • A + (7/5) • B
  let vector_relation_Q := Q = - (1/4) • A + (5/4) • B 
  vector_relation_P ∧ vector_relation_Q

theorem verify_extending_points 
  (A B P Q : ℝ → ℝ → ℝ)
  (h1 : 7 • (P - A) = 2 • (B - P))
  (h2 : 5 • (Q - A) = 1 • (Q - B)) :
  verify_P_and_Q A B P Q := 
by
  sorry  

end verify_extending_points_l503_503741


namespace find_f_neg_pi_over_6_l503_503025

variable (m : ℝ)
variable (h₀ : m ≠ 0)

def f (x : ℝ) : ℝ :=
  m * x^3 + x * Real.sin x

theorem find_f_neg_pi_over_6 (h₁ : f m (Real.pi / 6) = - Real.pi / 3) :
  f m (- Real.pi / 6) = Real.pi / 2 :=
  sorry

end find_f_neg_pi_over_6_l503_503025


namespace triangle_third_side_length_l503_503458

theorem triangle_third_side_length (a b : ℝ) (alpha : ℝ)
    (ha : a = 5) (hb : b = 8)
    (angle_relation : sin(alpha) * b = sin(2 * alpha) * a) :
    let c := sqrt (a^2 + b^2 - 2 * a * b * cos (2 * alpha)) in
    c ≈ 7.8 := 
by {
    sorry
}

end triangle_third_side_length_l503_503458


namespace sum_S_1_to_299_eq_150_l503_503281

def S (n : ℕ) : ℤ := (-1)^(n+1) * ((n+1) / 2)

theorem sum_S_1_to_299_eq_150 :
  ∑ i in Finset.range 299, S (i + 1) = 150 := by sorry

end sum_S_1_to_299_eq_150_l503_503281


namespace proof_fathers_current_age_l503_503513

-- Define the son and father's age according to the conditions
variables (S F : ℕ) (son_age_5_years_back : ℕ) (father_at_son_birth : ℕ)

-- Initial conditions
def condition_1 : Prop := son_age_5_years_back = 26
def condition_2 : Prop := father_at_son_birth = S
def condition_3 : Prop := S = son_age_5_years_back + 5

-- The statement to be proven
def fathers_current_age : Prop := F = S + (S - son_age_5_years_back)

theorem proof_fathers_current_age
  (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : fathers_current_age :=
by
  -- Skipping the proof steps
  sorry

end proof_fathers_current_age_l503_503513


namespace sum_of_solutions_sqrt_eq_7_l503_503975

theorem sum_of_solutions_sqrt_eq_7 :
  let f (x : ℝ) := sqrt (x + 1) + sqrt (9 / (x + 1)) + sqrt (x + 1 + 9 / (x + 1))
  sum_of_real_solutions (f x = 7) = (331.25 / 49) :=
  by
  -- The let f and sum_of_real_solutions definitions need appropriate context and implementation.
  sorry

end sum_of_solutions_sqrt_eq_7_l503_503975


namespace same_last_k_digits_pow_l503_503305

theorem same_last_k_digits_pow (A B : ℤ) (k n : ℕ) 
  (h : A % 10^k = B % 10^k) : 
  (A^n % 10^k = B^n % 10^k) := 
by
  sorry

end same_last_k_digits_pow_l503_503305


namespace sin_cos_value_l503_503986

noncomputable theory

variable (α : ℝ)

def sin_value : ℝ := real.sin α
def cos_value : ℝ := real.cos α

def given_condition : sin_value α = (real.sqrt 5 / 5) := sorry

def target_expression : ℝ := (sin_value α) ^ 4 - (cos_value α) ^ 4

theorem sin_cos_value (α : ℝ)
  (h : sin_value α = (real.sqrt 5 / 5)) :
  target_expression α = -((3 : ℝ) / 5) :=
sorry

end sin_cos_value_l503_503986


namespace large_posters_count_l503_503916

theorem large_posters_count (total_posters small_ratio medium_ratio : ℕ) (h_total : total_posters = 50) (h_small_ratio : small_ratio = 2/5) (h_medium_ratio : medium_ratio = 1/2) :
  let small_posters := (small_ratio * total_posters) in
  let medium_posters := (medium_ratio * total_posters) in
  let large_posters := total_posters - (small_posters + medium_posters) in
  large_posters = 5 := by
{
  sorry
}

end large_posters_count_l503_503916


namespace total_students_appeared_l503_503325

variable (T : ℝ) -- total number of students

def fraction_failed := 0.65
def num_failed := 546

theorem total_students_appeared :
  0.65 * T = 546 → T = 840 :=
by
  intro h
  sorry

end total_students_appeared_l503_503325


namespace inclination_angle_of_line_l503_503046

theorem inclination_angle_of_line :
  (∃ t : ℝ, (x = -2 + t * real.cos (real.pi / 6)) ∧ (y = 3 - t * real.sin (real.pi / 3))) →
  abs (real.atan (-(1:ℝ))) = real.pi * (3/4) :=
by
  sorry

end inclination_angle_of_line_l503_503046


namespace second_interest_rate_exists_l503_503855

theorem second_interest_rate_exists (X Y : ℝ) (H : 0 < X ∧ X ≤ 10000) : ∃ Y, 8 * X + Y * (10000 - X) = 85000 :=
by
  sorry

end second_interest_rate_exists_l503_503855


namespace roots_of_quadratic_eq_l503_503814

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_eq_l503_503814


namespace value_of_g_at_1_l503_503762

def f (x : ℝ) : ℝ := log x / log 3

def reflected (x : ℝ) : ℝ := 3 ^ x

def g (x : ℝ) : ℝ := reflected (x + 1)

theorem value_of_g_at_1 : g 1 = 9 := by
  -- conditions and proof to be written here
  sorry

end value_of_g_at_1_l503_503762


namespace incorrect_statement_count_l503_503154

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

end incorrect_statement_count_l503_503154


namespace area_enclosed_by_equation_is_96_l503_503538

-- Definitions based on the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- The theorem to prove the area enclosed by the graph is 96 square units
theorem area_enclosed_by_equation_is_96 :
  (∃ x y : ℝ, equation x y) → ∃ A : ℝ, A = 96 :=
sorry

end area_enclosed_by_equation_is_96_l503_503538


namespace part1_part2_min_g_l503_503269

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := sin x - sqrt 3 * cos x

-- Statement for Part (1)
theorem part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (hfx : f x = 2 / 3) : cos (2 * π / 3 + x) = -2 * sqrt 2 / 3 :=
sorry

-- Define the function g(x) with the given transformations
def g (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

-- Statement for Part (2)
theorem part2_min_g (hx : 0 ≤ x ∧ x ≤ π / 2) : ∃ y ∈ Icc (0:ℝ) (π / 2), g y = -2 :=
sorry

end part1_part2_min_g_l503_503269


namespace minimum_f_value_g_ge_f_implies_a_ge_4_l503_503988

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3

theorem minimum_f_value : (∃ x : ℝ, f x = 2 / Real.exp 1) :=
  sorry

theorem g_ge_f_implies_a_ge_4 (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f x ≤ g x a) → a ≥ 4 :=
  sorry

end minimum_f_value_g_ge_f_implies_a_ge_4_l503_503988


namespace geometric_series_sum_frac_l503_503609

open BigOperators

theorem geometric_series_sum_frac (q : ℚ) (a1 : ℚ) (a_list: List ℚ) (h_theta : q = 1 / 2) 
(h_a_list : a_list ⊆ [-4, -3, -2, 0, 1, 23, 4]) : 
  a1 * (1 + q^5) / (1 - q) = 33 / 4 := by
  sorry

end geometric_series_sum_frac_l503_503609


namespace hyperbola_range_of_m_l503_503673

theorem hyperbola_range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (1 + m) + (y^2) / (1 - m) = 1) → 
  (m < -1 ∨ m > 1) :=
by 
sorry

end hyperbola_range_of_m_l503_503673


namespace stratified_sampling_grade10_students_l503_503133

-- Definitions based on the given problem
def total_students := 900
def grade10_students := 300
def sample_size := 45

-- Calculation of the number of Grade 10 students in the sample
theorem stratified_sampling_grade10_students : (grade10_students * sample_size) / total_students = 15 := by
  sorry

end stratified_sampling_grade10_students_l503_503133


namespace area_of_trapezoid_EFGH_l503_503858

/-- The coordinates of the vertices of the trapezoid EFGH. --/
structure Point where
  x : ℝ
  y : ℝ

def E : Point := ⟨0, 0⟩
def F : Point := ⟨0, -3⟩
def G : Point := ⟨5, 0⟩
def H : Point := ⟨5, 8⟩

/-- Calculate the vertical and horizontal distances between two points. --/
def vertical_distance (P Q : Point) : ℝ := abs (P.y - Q.y)
def horizontal_distance (P Q : Point) : ℝ := abs (P.x - Q.x)

/-- Calculate the area of the trapezoid given the lengths of the two bases and the height. --/
def trapezoid_area (base1 base2 height : ℝ) : ℝ :=
  (base1 + base2) / 2 * height

theorem area_of_trapezoid_EFGH : trapezoid_area (vertical_distance E F) (vertical_distance G H) (horizontal_distance E G) = 27.5 := 
by sorry

end area_of_trapezoid_EFGH_l503_503858


namespace fruit_difference_l503_503747

/-- Mr. Connell harvested 60 apples and 3 times as many peaches. The difference 
    between the number of peaches and apples is 120. -/
theorem fruit_difference (apples peaches : ℕ) (h1 : apples = 60) (h2 : peaches = 3 * apples) :
  peaches - apples = 120 :=
sorry

end fruit_difference_l503_503747


namespace median_of_first_fifteen_positive_integers_l503_503473

theorem median_of_first_fifteen_positive_integers : ∃ m : ℝ, m = 7.5 :=
by
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median := (seq[6] + seq[7]) / 2
  use median
  sorry

end median_of_first_fifteen_positive_integers_l503_503473


namespace probability_of_arrangement_XXOXOO_l503_503774

noncomputable def probability_of_XXOXOO : ℚ :=
  let total_arrangements := (Nat.factorial 6) / ((Nat.factorial 4) * (Nat.factorial 2))
  let favorable_arrangements := 1
  favorable_arrangements / total_arrangements

theorem probability_of_arrangement_XXOXOO :
  probability_of_XXOXOO = 1 / 15 :=
by
  sorry

end probability_of_arrangement_XXOXOO_l503_503774


namespace yara_total_earnings_l503_503560

-- Lean code to represent the conditions and the proof statement

theorem yara_total_earnings
  (x : ℕ)  -- Yara's hourly wage
  (third_week_hours : ℕ := 18)
  (previous_week_hours : ℕ := 12)
  (extra_earnings : ℕ := 36)
  (third_week_earning : ℕ := third_week_hours * x)
  (previous_week_earning : ℕ := previous_week_hours * x)
  (total_earning : ℕ := third_week_earning + previous_week_earning) :
  third_week_earning = previous_week_earning + extra_earnings → 
  total_earning = 180 := 
by
  -- Proof here
  sorry

end yara_total_earnings_l503_503560


namespace find_equation_of_line_l503_503197

def is_on_line (l : ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop := l p

def slope (m : ℝ) (l : ℝ × ℝ → Prop) : Prop := 
  ∀ (p1 p2 : ℝ × ℝ), p1 ≠ p2 → l p1 → l p2 → (p2.2 - p1.2)/(p2.1 - p1.1) = m

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

noncomputable def circle_center : ℝ × ℝ := (-1, 0)

noncomputable def circle : ℝ × ℝ → Prop := 
  λ p, (p.1 + 1)^2 + p.2^2 = 1

noncomputable def line1 : ℝ × ℝ → Prop := 
  λ p, p.1 + p.2 - 2 = 0

noncomputable def line2 : ℝ × ℝ → Prop := 
  λ p, p.1 - p.2 + 1 = 0

theorem find_equation_of_line :
  is_on_line circle (circle_center) →
  is_on_line line1 (0, 2) →
  slope (-1) line1 →
  perpendicular (1) (-1) →
  is_on_line line2 (circle_center) →
  slope 1 line2 :=
by
  sorry

end find_equation_of_line_l503_503197


namespace find_d_l503_503219

theorem find_d (c a m d : ℝ) (h : m = (c * a * d) / (a - d)) : d = (m * a) / (m + c * a) :=
by sorry

end find_d_l503_503219


namespace find_f_2017_l503_503631

noncomputable def f (x : ℤ) (a α b β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem find_f_2017
(x : ℤ)
(a α b β : ℝ)
(h : f 4 a α b β = 3) :
f 2017 a α b β = -3 := 
sorry

end find_f_2017_l503_503631


namespace baba_yaga_powder_problem_l503_503110

theorem baba_yaga_powder_problem (A B d : ℝ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 := 
sorry

end baba_yaga_powder_problem_l503_503110


namespace age_ratio_l503_503834

theorem age_ratio (s a : ℕ) (h1 : s - 3 = 2 * (a - 3)) (h2 : s - 7 = 3 * (a - 7)) :
  ∃ x : ℕ, (x = 23) ∧ (s + x) / (a + x) = 3 / 2 :=
by
  sorry

end age_ratio_l503_503834


namespace triangle_area_via_line_eq_l503_503029

theorem triangle_area_via_line_eq (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  area = 1 / (2 * |a * b|) :=
by
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  sorry

end triangle_area_via_line_eq_l503_503029


namespace range_of_d_l503_503620

theorem range_of_d {x y : ℝ} (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
    (on_circle : (P.1 - 3)^2 + (P.2 - 4)^2 = 1)
    (A_def : A = (0, -1))
    (B_def : B = (0, 1)) :
    let d : ℝ := (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2
    in 32 ≤ d ∧ d ≤ 72 :=
sorry

end range_of_d_l503_503620


namespace triangle_area_example_l503_503842

def isosceles_triangle_area (a b c : ℝ) (h1 : a = b) (h2 : c = 24) : ℝ :=
  let half_c := c / 2
  let h := real.sqrt (a ^ 2 - half_c ^ 2)
  0.5 * c * h

theorem triangle_area_example :
  isosceles_triangle_area 13 13 24 13 rfl :=
by
  simp only [isosceles_triangle_area, real.sqrt, pow_two, bit0, add_zero, mul_one]
  norm_num
  sorry -- The proof will be inserted here.

end triangle_area_example_l503_503842


namespace root_of_equation_l503_503054

theorem root_of_equation : ∀ x : ℝ, 10^(x + real.log 2) = 2000 → x = 3 :=
by
  intro x
  intro h
  sorry

end root_of_equation_l503_503054


namespace roots_of_quadratic_eq_l503_503811

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l503_503811


namespace f_monotonically_increasing_l503_503424

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem f_monotonically_increasing : 
  ∀ x : ℝ, 1 < x → (∀ y : ℝ, x < y → f(x) < f(y)) := 
by 
  sorry

end f_monotonically_increasing_l503_503424


namespace num_large_posters_l503_503920

-- Define the constants
def total_posters : ℕ := 50
def small_posters : ℕ := total_posters * 2 / 5
def medium_posters : ℕ := total_posters / 2
def large_posters : ℕ := total_posters - (small_posters + medium_posters)

-- Theorem to prove the number of large posters
theorem num_large_posters : large_posters = 5 :=
by
  sorry

end num_large_posters_l503_503920


namespace find_D_l503_503710

-- Definitions from conditions
def is_different (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- The proof problem
theorem find_D (A B C D : ℕ) (h_diff: is_different A B C D) (h_eq : 700 + 10 * A + 5 + 100 * B + 70 + C = 100 * D + 38) : D = 9 :=
sorry

end find_D_l503_503710


namespace andre_tuesday_ladybugs_l503_503158

theorem andre_tuesday_ladybugs (M T : ℕ) (dots_per_ladybug total_dots monday_dots tuesday_dots : ℕ)
  (h1 : M = 8)
  (h2 : dots_per_ladybug = 6)
  (h3 : total_dots = 78)
  (h4 : monday_dots = M * dots_per_ladybug)
  (h5 : tuesday_dots = total_dots - monday_dots)
  (h6 : tuesday_dots = T * dots_per_ladybug) :
  T = 5 :=
sorry

end andre_tuesday_ladybugs_l503_503158


namespace wanda_treats_jane_treats_ratio_l503_503715

-- Definitions based on the conditions
variables (J_t J_b W_t W_b : ℕ)

-- Conditions given in the problem
def condition1 : Prop := J_b = 3 / 4 * J_t
def condition2 : Prop := W_b = 3 * W_t
def condition3 : Prop := W_b = 90
def condition4 : Prop := J_b + J_t + W_b + W_t = 225

-- Theorem statement based on the problem
theorem wanda_treats_jane_treats_ratio (J_t J_b W_t W_b : ℕ)
  (h1 : condition1 J_t J_b)
  (h2 : condition2 W_t W_b)
  (h3 : condition3 W_b)
  (h4 : condition4 J_t J_b W_t W_b) :
  W_t / J_t = 1 / 2 :=
begin
  sorry
end

end wanda_treats_jane_treats_ratio_l503_503715


namespace align_two_pieces_l503_503395

/-- There are four pieces on a coordinate plane, their centers having integer coordinates.
    It is allowed to move any piece by the vector connecting the centers of any two
    of the other pieces. Prove that it is possible to align any two pre-designated pieces. -/
theorem align_two_pieces (pieces : fin 4 → ℤ × ℤ) :
  (∃ i j : fin 4, i ≠ j ∧ 
  ∀ k : fin 4, k ≠ i ∨ k ≠ j 
  → ∃ m n : fin 4, m ≠ n ∧ pieces m = pieces n) := 
sorry

end align_two_pieces_l503_503395


namespace area_of_pivot_region_l503_503601

noncomputable def regular_pentagon (A : ℝ) := { ... }  -- defines a regular pentagon with the given area A

def pivot_lines (p : regular_pentagon 1) := {
  lines | line not passing through any vertex of p ∧ 
         ∃ t : list vertex, length t = 3 ∧ ∀ v in t, v on one side of line 
         ∧ ∀ v in vertices \ t, v on the other side
}

def pivot_point (p : regular_pentagon 1) (Pt : point) := {
  Pt inside p ∧ ∃ finite S, ∀ l in all_lines \ S, ¬ pivot_line l
}

theorem area_of_pivot_region : 
  ∃ (p : regular_pentagon 1), 
    (pivot_points_area p = (7 - 3 * real.sqrt 5) / 2) := 
sorry

end area_of_pivot_region_l503_503601


namespace find_area_of_triangle_MSN_l503_503246

-- Define the conditions and the statement
theorem find_area_of_triangle_MSN (m n k : ℝ) (h1 : m^2 + n^2 + k^2 ≤ 12) (h2 : ∃ (angle_MSN : ℝ), angle_MSN = 30) (h3 : m = n ∧ n = k) :
  (1 / 2) * m * n * real.sin ((angle_MSN.to_real * π) / 180) = 1 :=
by
  sorry

end find_area_of_triangle_MSN_l503_503246


namespace frog_vertical_side_probability_l503_503896

-- Definitions of the problem, frog jumps, and bounds.
def square_bounded (x y : ℕ) : Prop := (0 ≤ x ∧ x ≤ 5) ∧ (0 ≤ y ∧ y ≤ 5)

def frog_starting_point (x y : ℕ) : Prop := (x = 2) ∧ (y = 3)

-- Definition of probability P_{(x,y)}
-- Here P_x_y represents the probability starting at (x, y)
noncomputable def P : ℕ × ℕ → ℚ
| (0, y) := 1
| (5, y) := 1
| (x, 0) := 0
| (x, 5) := 0
| (x, y) := (1 / 4) * (P (x-1, y) + P (x+1, y) + P (x, y-1) + P (x, y+1))

-- Theorem to prove the probability
theorem frog_vertical_side_probability :
  square_bounded 2 3 →
  frog_starting_point 2 3 →
  P (2, 3) = 2 / 5 :=
by
  intros h1 h2
  sorry

end frog_vertical_side_probability_l503_503896


namespace alex_quiz_scores_l503_503438

theorem alex_quiz_scores :
  ∃ scores : List ℕ,
    scores = [94, 92, 85, 78, 74, 69] ∧
    (let first_four := [92, 85, 78, 74] in
     let mean := 82 in
     (List.sum scores = 6 * mean) ∧
     (List.sum first_four + List.sum (scores.diff first_four) = 6 * mean) ∧
     ∀ score ∈ scores, score < 95 ∧
     scores.nodup ∧
     scores.toFinset.card = 6) := by
{
  sorry
}

end alex_quiz_scores_l503_503438


namespace triangle_ADE_is_right_l503_503361

theorem triangle_ADE_is_right (A B C D E: Point)
  (h1: Triangle A B C)
  (h2: Acute (Triangle A B C))
  (h3: Scalene (Triangle A B C))
  (h4: InInterior D (Triangle A B C))
  (h5: InInterior E (Triangle A B C))
  (h6: Angle D A B = Angle D C B)
  (h7: Angle D A C = Angle D B C)
  (h8: Angle E A B = Angle E B C)
  (h9: Angle E A C = Angle E C B)
  : ∠ (Triangle A D E) = 90 :=
sorry

end triangle_ADE_is_right_l503_503361


namespace tile_probability_l503_503071

open Nat

theorem tile_probability : 
  let A := {n | 1 ≤ n ∧ n ≤ 20}
  let B := {n | 11 ≤ n ∧ n ≤ 30}
  let probA := (∑ n in A, if n < 15 then 1 else 0) / (∑ n in A, 1)
  let probB := (∑ n in B, if n % 2 = 0 ∨ n > 25 then 1 else 0) / (∑ n in B, 1)
  (probA * probB = 21 / 50) :=
by
  -- The proof goes here
  sorry

end tile_probability_l503_503071


namespace median_first_fifteen_positive_integers_l503_503464

-- Define the list of the first fifteen positive integers
def first_fifteen_positive_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

-- Define the property that the median of the list is 8.0
theorem median_first_fifteen_positive_integers : median(first_fifteen_positive_integers) = 8.0 := 
sorry

end median_first_fifteen_positive_integers_l503_503464


namespace solution_inequality_l503_503203

theorem solution_inequality (θ x : ℝ)
  (h : |x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) : 
  -1 ≤ x ∧ x ≤ -Real.cos (2 * θ) :=
sorry

end solution_inequality_l503_503203


namespace nine_digit_symmetrical_numbers_count_l503_503938

def is_symmetrical (n : ℕ) : Prop :=
  let s := n.digits 10;
  s = s.reverse

theorem nine_digit_symmetrical_numbers_count :
  {n : ℕ // 10^8 ≤ n ∧ n < 10^9 ∧ is_symmetrical n ∧ is_symmetrical (n + 11000)}.card = 8100 :=
by
  sorry

end nine_digit_symmetrical_numbers_count_l503_503938


namespace exists_positive_integer_solution_l503_503015

theorem exists_positive_integer_solution (a : ℕ) (ha : 0 < a) :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^3 + x + a^2 = y^2 :=
by
  let x := 4 * a^2 * (16 * a^4 + 2)
  let y := 2 * a * (16 * a^4 + 2) * (16 * a^4 + 1) - a
  have hx : 0 < x := sorry
  have hy : 0 < y := sorry
  use x, y
  split, 
  exact hx,
  split,
  exact hy,
  sorry

end exists_positive_integer_solution_l503_503015


namespace log_squared_eqn_l503_503493

noncomputable def log_base_10 : ℝ → ℝ := real.logb 10

theorem log_squared_eqn : [log_base_10 (7 * log_base_10 1000)]^2 = (log_base_10 21)^2 :=
by
  -- Proof goes here
  sorry

end log_squared_eqn_l503_503493


namespace magnitude_of_difference_l503_503247

open Real

variables (a b : EuclideanSpace ℝ (Fin 3))
variable (angle_ab : ℝ)
variable (mag_a : ℝ)
variable (mag_b : ℝ)
variable (dot_ab : ℝ)

-- Assume the given conditions
axiom angle_between_vectors : angle_ab = 120 * π / 180
axiom magnitude_a : mag_a = 1
axiom magnitude_b : mag_b = 3
axiom dot_product_ab : dot_ab = mag_a * mag_b * cos angle_ab

-- Define the vectors with the above conditions
def scaled_a := 2 • a
def diff_vector := scaled_a - b

def magnitude_diff_vector (a b : EuclideanSpace ℝ (Fin 3)) :=
  sqrt (4 * (mag_a * mag_a) - 4 * dot_ab + mag_b * mag_b)

theorem magnitude_of_difference :
  |2 • a - b| = sqrt 19 := by 
  sorry

end magnitude_of_difference_l503_503247


namespace Vitya_needs_58_offers_l503_503093

theorem Vitya_needs_58_offers :
  ∃ k : ℕ, (log 0.01 / log (12 / 13) < k) ∧ k = 58 :=
by
  sorry

end Vitya_needs_58_offers_l503_503093


namespace sum_squares_solutions_l503_503976

theorem sum_squares_solutions :
  let a := 2
  let b := -3
  let c := 1/1004
  let d_pos := 1/502
  let d_neg := -1/502
  let sum_squares (roots : List ℚ) : ℚ := roots.foldr (λ x acc => x^2 + acc) 0
  let poly (d : ℚ) (x : ℚ) : ℚ := a * x^2 + b * x + c - d
  sum_squares [x | x ∈ [1/1,1/2,1/3,1/4] | poly d_pos x = 0]
  + sum_squares [x | x ∈ [1/1,1/2,1/3,1/4] | poly d_neg x = 0]
  = 40373/4032 := sorry

end sum_squares_solutions_l503_503976


namespace base_conversion_difference_l503_503163

-- Definitions
def base9_to_base10 (n : ℕ) : ℕ := 3 * (9^2) + 2 * (9^1) + 7 * (9^0)
def base8_to_base10 (m : ℕ) : ℕ := 2 * (8^2) + 5 * (8^1) + 3 * (8^0)

-- Statement
theorem base_conversion_difference :
  base9_to_base10 327 - base8_to_base10 253 = 97 :=
by sorry

end base_conversion_difference_l503_503163


namespace least_number_to_add_l503_503870

theorem least_number_to_add (n : ℕ) (sum_digits : ℕ) (next_multiple : ℕ) 
  (h1 : n = 51234) 
  (h2 : sum_digits = 5 + 1 + 2 + 3 + 4) 
  (h3 : next_multiple = 18) :
  ∃ k, (k = next_multiple - sum_digits) ∧ (n + k) % 9 = 0 :=
sorry

end least_number_to_add_l503_503870


namespace arc_length_proof_l503_503252

def radius : ℝ := 3
def central_angle_degrees : ℝ := 30
def degrees_to_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)
def arc_length (r : ℝ) (θ_rad : ℝ) : ℝ := r * θ_rad

theorem arc_length_proof : arc_length radius (degrees_to_radians central_angle_degrees) = (1/2) * Real.pi := by
  sorry

end arc_length_proof_l503_503252


namespace files_per_folder_l503_503002

theorem files_per_folder (total_files deleted_files folders : ℕ)
    (h1 : total_files = 27)
    (h2 : deleted_files = 9)
    (h3 : folders = 3) :
    (total_files - deleted_files) / folders = 6 :=
begin
    sorry
end

end files_per_folder_l503_503002


namespace sampling_methods_correct_l503_503067

theorem sampling_methods_correct :
  (community : List (String × Nat)) →
  (students : List String) →
  (community = [("high_income", 125), ("middle_income", 280), ("low_income", 95)] ∧ length students = 12) →
  (select_households_using : String) →
  (select_students_using : String) →
  (select_households_using = "stratified" ∧ select_students_using = "random") :=
by
  intros community students Hmethods select_households_using select_students_using
  sorry

end sampling_methods_correct_l503_503067


namespace minimum_box_value_l503_503295

def is_valid_pair (a b : ℤ) : Prop :=
  a * b = 15 ∧ (a^2 + b^2 ≥ 34)

theorem minimum_box_value :
  ∃ (a b : ℤ), is_valid_pair a b ∧ (∀ (a' b' : ℤ), is_valid_pair a' b' → a^2 + b^2 ≤ a'^2 + b'^2) ∧ a^2 + b^2 = 34 :=
by
  sorry

end minimum_box_value_l503_503295


namespace find_principal_sum_l503_503925

theorem find_principal_sum
  (R : ℝ) (P : ℝ)
  (H1 : 0 < R)
  (H2 : 8 * 10 * P / 100 = 150) :
  P = 187.50 :=
by
  sorry

end find_principal_sum_l503_503925


namespace sum_of_interior_angles_l503_503786

theorem sum_of_interior_angles (h : ∀ (n : ℕ), 360 / 20 = n) : 
  ∃ (s : ℕ), s = 2880 :=
by
  have n := 360 / 20
  have sum := 180 * (n - 2)
  use sum
  sorry

end sum_of_interior_angles_l503_503786


namespace pencils_left_l503_503446

-- Definitions based on the given conditions
def students_first_classroom : ℕ := 30
def students_second_classroom : ℕ := 20
def total_pencils : ℕ := 210
def pencils_per_student : ℕ := 4 \-- (after rounding down from 4.2)

-- Statement of the theorem
theorem pencils_left : 
  let total_students := students_first_classroom + students_second_classroom,
      pencils_used_first := students_first_classroom * pencils_per_student,
      pencils_used_second := students_second_classroom * pencils_per_student,
      total_pencils_used := pencils_used_first + pencils_used_second
  in total_pencils - total_pencils_used = 10 :=
by
  sorry

end pencils_left_l503_503446


namespace x_intercept_of_quadratic_l503_503591

theorem x_intercept_of_quadratic (a b c : ℝ) (h_vertex : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 4 ∧ y = -2) 
(h_intercept : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 1 ∧ y = 0) : 
∃ x : ℝ, x = 7 ∧ ∃ y : ℝ, y = a * x^2 + b * x + c ∧ y = 0 :=
sorry

end x_intercept_of_quadratic_l503_503591


namespace sum_of_rational_coeff_l503_503330

theorem sum_of_rational_coeff (x : ℝ) :
  let s := (sqrt x + 2)^19 in
  (∑ r in Finset.range 20, if (19 - r) % 2 = 0 then nat.choose 19 r * 2^r else 0) = (3^19 + 1) / 2 :=
by
  sorry

end sum_of_rational_coeff_l503_503330


namespace stickers_spent_correct_l503_503014

def initial_pennies := 2476
def spent_on_toy := 1145
def spent_on_candy := 781
def total_spent := spent_on_toy + spent_on_candy
def spent_on_stickers := initial_pennies - total_spent

theorem stickers_spent_correct : spent_on_stickers = 550 := by
  simp [initial_pennies, spent_on_toy, spent_on_candy, total_spent, spent_on_stickers]
  done

end stickers_spent_correct_l503_503014


namespace negation_statement_l503_503802

theorem negation_statement (a b : ℝ) :
  ¬ (a > b → 2 ^ a > 2 ^ b) ↔ (a ≤ b → 2 ^ a ≤ 2 ^ b) :=
by {
  sorry
}

end negation_statement_l503_503802


namespace derivative_f_at_0_l503_503531

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else x + Real.arcsin (x^2 * Real.sin (6 / x))

theorem derivative_f_at_0 : 
  (Real.deriv f 0) = 1 := by 
  sorry

end derivative_f_at_0_l503_503531


namespace max_value_of_expression_l503_503595

theorem max_value_of_expression (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁^2 / 4 + 9 * y₁^2 / 4 = 1) 
  (h₂ : x₂^2 / 4 + 9 * y₂^2 / 4 = 1) 
  (h₃ : x₁ * x₂ + 9 * y₁ * y₂ = -2) :
  (|2 * x₁ + 3 * y₁ - 3| + |2 * x₂ + 3 * y₂ - 3|) ≤ 6 + 2 * Real.sqrt 5 :=
sorry

end max_value_of_expression_l503_503595


namespace numTrianglesFromDecagon_is_120_l503_503666

noncomputable def numTrianglesFromDecagon : ℕ := 
  nat.choose 10 3

theorem numTrianglesFromDecagon_is_120 : numTrianglesFromDecagon = 120 := 
  by
    -- Form the combination
    have : numTrianglesFromDecagon = nat.choose 10 3 := rfl

    -- Calculate
    have calc₁ : nat.choose 10 3 = 10 * 9 * 8 / (3 * 2 * 1) := by 
      exact nat.choose_eq_div
      simp

    -- Simplify the calculation to 120
    have : 10 * 9 * 8 / (3 * 2 * 1) = 120 := by 
      norm_num 

    exact eq.trans this.symm calc₁.symm ⟩

end numTrianglesFromDecagon_is_120_l503_503666


namespace median_of_first_fifteen_positive_integers_l503_503474

theorem median_of_first_fifteen_positive_integers : ∃ m : ℝ, m = 7.5 :=
by
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median := (seq[6] + seq[7]) / 2
  use median
  sorry

end median_of_first_fifteen_positive_integers_l503_503474


namespace complex_modulus_to_real_l503_503678

theorem complex_modulus_to_real (a : ℝ) (h : (a + 1)^2 + (1 - a)^2 = 10) : a = 2 ∨ a = -2 :=
sorry

end complex_modulus_to_real_l503_503678


namespace ramu_selling_price_l503_503009

theorem ramu_selling_price (P R : ℝ) (profit_percent : ℝ) 
  (P_def : P = 42000)
  (R_def : R = 13000)
  (profit_percent_def : profit_percent = 17.272727272727273) :
  let total_cost := P + R
  let selling_price := total_cost * (1 + (profit_percent / 100))
  selling_price = 64500 := 
by
  sorry

end ramu_selling_price_l503_503009


namespace player_c_wins_l503_503066

theorem player_c_wins :
  ∀ (A_wins A_losses B_wins B_losses C_losses C_wins : ℕ),
  A_wins = 4 →
  A_losses = 2 →
  B_wins = 3 →
  B_losses = 3 →
  C_losses = 3 →
  A_wins + B_wins + C_wins = A_losses + B_losses + C_losses →
  C_wins = 2 :=
by
  intros A_wins A_losses B_wins B_losses C_losses C_wins
  sorry

end player_c_wins_l503_503066


namespace collinear_centers_l503_503735

-- Definitions of conditions
variables (A B C D E F X Y Z : Type)
           [add_comm_group A] [add_comm_group B] [add_comm_group C]
           [add_comm_group D] [add_comm_group E] [add_comm_group F]
           [add_comm_group X] [add_comm_group Y] [add_comm_group Z]

-- Hypotheses based on given problem
variables (AB DE BC EF CD FA : ℝ)
variables (h1 : ∥A - B∥ = ∥D - E∥)
          (h2 : ∥B - C∥ = ∥E - F∥)
          (h3 : ∥C - D∥ = ∥F - A∥)
          (h4 : (AB * DE) = (BC * EF) ∧ (BC * EF) = (CD * FA))

-- Define the calculation to find midpoints
noncomputable def midpoint (X Y : Type) [has_add X] [has_inv X] : Type :=
(X + Y) / 2

-- Midpoints of AD, BE, CF
variables (M_X : X = midpoint A D)
          (M_Y : Y = midpoint B E)
          (M_Z : Z = midpoint C F)

-- Statement of the proof
theorem collinear_centers :
  collinear [circumcenter (triangle ACE), circumcenter (triangle BDF), orthocenter (triangle XYZ)] :=
sorry

end collinear_centers_l503_503735


namespace probability_union_l503_503123

variables (A B : Prop)
noncomputable def P : Prop → ℝ := sorry

axiom P_A : P A = 0.34
axiom P_B : P B = 0.32
axiom P_AB : P (A ∧ B) = 0.31

theorem probability_union : P (A ∨ B) = 0.35 :=
by
    -- Use the probability formula for the union of two events
    have h : P (A ∨ B) = P A + P B - P (A ∧ B), from sorry,
    -- Substitute the values and verify
    rw [P_A, P_B, P_AB],
    -- Conclude the proof
    exact sorry

end probability_union_l503_503123


namespace sunscreen_bottle_contains_l503_503835

-- Definitions
def t_r := 2 -- re-application interval in hours
def s_a := 3 -- amount of sunscreen per application in ounces
def c_b := 3.5 -- bottle cost in dollars
def t_t := 16 -- total time at the beach in hours
def c_s := 7 -- cost of sunscreen in dollars

-- Theorem to prove that a bottle of sunscreen contains 12 ounces
theorem sunscreen_bottle_contains (s_b : ℝ) :
  s_b = 12 :=
by
  let applications := t_t / t_r
  let total_sunscreen_needed := applications * s_a
  let number_of_bottles := c_s / c_b
  have h1 : total_sunscreen_needed = 24 := by norm_num [t_t, t_r, s_a]
  have h2 : number_of_bottles = 2 := by norm_num [c_s, c_b]
  have h3 : s_b = total_sunscreen_needed / number_of_bottles := by norm_num [h1, h2]
  exact h3 

sunscreen_bottle_contains sorry -- Placeholder to indicate the need to provide proof

end sunscreen_bottle_contains_l503_503835


namespace geometric_sequence_a9_l503_503273

open Nat

theorem geometric_sequence_a9 (a : ℕ → ℝ) (h1 : a 3 = 20) (h2 : a 6 = 5) 
  (h_geometric : ∀ m n, a ((m + n) / 2) ^ 2 = a m * a n) : 
  a 9 = 5 / 4 := 
by
  sorry

end geometric_sequence_a9_l503_503273


namespace unique_ordered_triple_l503_503654

theorem unique_ordered_triple :
  ∃! a b c : ℤ, a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ real.log b / real.log a = c ^ 3 ∧ a + b + c = 100 := sorry

end unique_ordered_triple_l503_503654


namespace sum_of_angles_l503_503926

theorem sum_of_angles (A B C D E F : Type) [Inhabited A]
  (h1 : is_isosceles A B C)
  (h2 : is_isosceles D E F)
  (h3 : angle A B C = 30)
  (h4 : angle D E F = 20) :
  angle D A C + angle A D E = 155 := 
sorry

end sum_of_angles_l503_503926


namespace unique_real_solution_l503_503574

theorem unique_real_solution (x y : ℝ) (h1 : x^3 = 2 - y) (h2 : y^3 = 2 - x) : x = 1 ∧ y = 1 :=
sorry

end unique_real_solution_l503_503574


namespace police_officers_on_duty_l503_503401

theorem police_officers_on_duty
  (female_officers : ℕ)
  (percent_female_on_duty : ℚ)
  (total_female_on_duty : ℕ)
  (total_officers_on_duty : ℕ)
  (H1 : female_officers = 1000)
  (H2 : percent_female_on_duty = 15 / 100)
  (H3 : total_female_on_duty = percent_female_on_duty * female_officers)
  (H4 : 2 * total_female_on_duty = total_officers_on_duty) :
  total_officers_on_duty = 300 :=
by
  sorry

end police_officers_on_duty_l503_503401


namespace ten_player_round_robin_matches_l503_503911

theorem ten_player_round_robin_matches :
  (∑ i in finset.range 10, i) = 45 :=
by
  sorry

end ten_player_round_robin_matches_l503_503911


namespace sin_cos_ratio_problem_l503_503365

theorem sin_cos_ratio_problem (x y : Real) (h1 : sin x / sin y = 3) (h2 : cos x / cos y = 1 / 2) :
  (sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 49 / 58) :=
by
  sorry

#eval 49 + 58 -- Ensure the constants are correct

end sin_cos_ratio_problem_l503_503365


namespace xiaoming_bus_time_l503_503106

-- Definitions derived from the conditions:
def total_time : ℕ := 40
def transfer_time : ℕ := 6
def subway_time : ℕ := 30
def bus_time : ℕ := 50

-- Theorem statement to prove the bus travel time equals 10 minutes
theorem xiaoming_bus_time : (total_time - transfer_time = 34) ∧ (subway_time = 30 ∧ bus_time = 50) → 
  ∃ (T_bus : ℕ), T_bus = 10 := by
  sorry

end xiaoming_bus_time_l503_503106


namespace angle_between_hands_at_15_45_l503_503074

-- Define the time 15:45 in a 24-hour format 
def hour := 15
def minute := 45

-- Define the 24-hour clock and the calculation of angles
def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60

-- Calculate the angle of the hour hand
def hour_angle := (hour % 12) * degrees_per_hour + (minute / 60) * degrees_per_hour

-- Calculate the angle of the minute hand
def minute_angle := minute * degrees_per_minute

-- Calculate the absolute difference between the two angles
def angle_diff := abs (minute_angle - hour_angle)

-- Statement: Prove that the angle between the hour and minute hand at 15:45 is 157.5 degrees
theorem angle_between_hands_at_15_45 : angle_diff = 157.5 := sorry

end angle_between_hands_at_15_45_l503_503074


namespace number_of_divisors_l503_503736

theorem number_of_divisors {n : ℕ} (h : n = (2^31) * (3^19) * (5^7)) :
  let n_squared := n * n in
  let divisors_count := (62 + 1) * (38 + 1) * (14 + 1) in
  let less_than_n_divisors := (divisors_count - 1) / 2 in
  let n_divisors := (31 + 1) * (19 + 1) * (7 + 1) in
  let result := less_than_n_divisors - n_divisors in
  result = 13307 :=
by
  sorry

end number_of_divisors_l503_503736


namespace axis_of_symmetry_compare_m_n_range_t_max_t_l503_503334

-- Condition: Definition of the parabola
def parabola (t x : ℝ) := x^2 - 2 * t * x + 1

-- Problem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ (y x : ℝ), parabola t x = y -> x = t :=
sorry

-- Problem 2: Comparing m and n
theorem compare_m_n (t m n : ℝ) :
  parabola t (t - 2) = m ∧ parabola t (t + 3) = n -> n > m := 
sorry

-- Problem 3: Range of t for y₁ ≤ y₂
theorem range_t (t x₁ y₁ y₂ : ℝ) :
  -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ = y₁ ∧ parabola t 3 = y₂ -> y₁ ≤ y₂ → t ≤ 1 :=
sorry

-- Problem 4: Maximum t for y₁ ≥ y₂
theorem max_t (t y₁ y₂ : ℝ) :
  (parabola t (t + 1) = y₁ ∧ parabola t (2 * t - 4) = y₂) → y₁ ≥ y₂ → t ≤ 5 :=
sorry

end axis_of_symmetry_compare_m_n_range_t_max_t_l503_503334


namespace percent_of_g_is_h_l503_503738

variable (a b c d e f g h : ℝ)

-- Conditions
def cond1a : f = 0.60 * a := sorry
def cond1b : f = 0.45 * b := sorry
def cond2a : g = 0.70 * b := sorry
def cond2b : g = 0.30 * c := sorry
def cond3a : h = 0.80 * c := sorry
def cond3b : h = 0.10 * f := sorry
def cond4a : c = 0.30 * a := sorry
def cond4b : c = 0.25 * b := sorry
def cond5a : d = 0.40 * a := sorry
def cond5b : d = 0.35 * b := sorry
def cond6a : e = 0.50 * b := sorry
def cond6b : e = 0.20 * c := sorry

-- Theorem to prove
theorem percent_of_g_is_h (h_percent_g : ℝ) 
  (h_formula : h = h_percent_g * g) : 
  h = 0.285714 * g :=
by
  sorry

end percent_of_g_is_h_l503_503738


namespace product_of_approx_numbers_l503_503584

theorem product_of_approx_numbers : 
  (let x := 0.3862 in let y := 0.85 in let result := (x * y).round (2 : ℕ) in result = 0.33) := sorry

end product_of_approx_numbers_l503_503584


namespace arithmetic_sequences_ratio_l503_503647

theorem arithmetic_sequences_ratio (a b S T : ℕ → ℕ) (h : ∀ n, S n / T n = 2 * n / (3 * n + 1)) :
  (a 2) / (b 3 + b 7) + (a 8) / (b 4 + b 6) = 9 / 14 :=
  sorry

end arithmetic_sequences_ratio_l503_503647


namespace smallest_number_conditions_l503_503864

theorem smallest_number_conditions :
  ∃ b : ℕ, 
    (b % 3 = 2) ∧ 
    (b % 4 = 2) ∧
    (b % 5 = 3) ∧
    (∀ b' : ℕ, 
      (b' % 3 = 2) ∧ 
      (b' % 4 = 2) ∧
      (b' % 5 = 3) → b ≤ b') :=
begin
  use 38,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros b' hb',
    have h3 := (hb'.left),
    have h4 := (hb'.right.left),
    have h5 := (hb'.right.right),
    -- The raw machinery for showing that 38 is the smallest may require more definition
    sorry
  }
end

end smallest_number_conditions_l503_503864


namespace eval_expression_l503_503191

theorem eval_expression : 5^(Real.log 9 / Real.log 5) = 9 := by 
  sorry

end eval_expression_l503_503191


namespace ratio_of_girls_to_boys_l503_503751

variable (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : g + b = 36)
                               (h₂ : g = b + 6) : g / b = 7 / 5 :=
by sorry

end ratio_of_girls_to_boys_l503_503751


namespace incorrect_choice_l503_503950

-- Definitions for the conditions (options) in Lean
def optA : Prop := ∃ assumptions, ¬(provable assumptions)
def optB : Prop := ∃ proofs (method1 method2 : proofs), method1 ≠ method2 ∧ conclusion method1 = conclusion method2
def optC : Prop := ∀ (variables constants : Type), known (variables.constants := true) ∧ defined (variables.constants := true)
def optD : Prop := ∀ (falseAssumption : Prop), falseAssumption → (accurateReasoning falseAssumption) → ¬trueConclusion
def optE : Prop := ∀ (statement : Prop), ¬provedDirectly statement → counterExampleNeeded statement

-- Incorrect statement
def incorrectStatement : Prop := optE

-- The theorem
theorem incorrect_choice : incorrectStatement := sorry

end incorrect_choice_l503_503950


namespace lines_are_skew_if_and_only_if_b_ne_40_div_19_l503_503553

/--
Given the parametric equations of two lines:
L1: (2, 3, b) + t * (3, 4, 5)
L2: (5, 3, 1) + u * (7, 3, 2)

Prove that these lines are skew if and only if b ≠ 40/19.
-/
theorem lines_are_skew_if_and_only_if_b_ne_40_div_19 (b : ℚ) :
  let r1 (t : ℚ) := (2 : ℚ, 3, b) + t * (3, 4, 5)
  let r2 (u : ℚ) := (5 : ℚ, 3, 1) + u * (7, 3, 2)
  (∀ t u : ℚ, r1 t ≠ r2 u) ↔ b ≠ 40 / 19 :=
sorry

end lines_are_skew_if_and_only_if_b_ne_40_div_19_l503_503553


namespace process_stops_after_k_assignments_l503_503626

def f (x : ℝ) : ℝ := 2 * x + 1

/-- The main theorem that states the range of x if the process stops after k assignments. -/
theorem process_stops_after_k_assignments (x : ℝ) (k : ℕ) (hk : k > 0):
  (∀ n : ℕ, 1 ≤ n → n ≤ k → f^[n] x ≤ 255) ∧ (f^[k+1] x > 255) ↔
  x ∈ Ioo (2^(8-k : ℝ) - 1) (2^(9-k : ℝ) - 1) :=
sorry

end process_stops_after_k_assignments_l503_503626


namespace num_triangles_from_decagon_l503_503657

theorem num_triangles_from_decagon (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ c ≠ a → ¬(collinear a b c)) :
  (nat.choose 10 3) = 120 :=
by
  sorry

end num_triangles_from_decagon_l503_503657


namespace students_play_at_least_one_sport_l503_503316

def B := 12
def C := 10
def S := 9
def Ba := 6

def B_and_C := 5
def B_and_S := 4
def B_and_Ba := 3
def C_and_S := 2
def C_and_Ba := 3
def S_and_Ba := 2

def B_and_C_and_S_and_Ba := 1

theorem students_play_at_least_one_sport : 
  B + C + S + Ba - B_and_C - B_and_S - B_and_Ba - C_and_S - C_and_Ba - S_and_Ba + B_and_C_and_S_and_Ba = 19 :=
by
  sorry

end students_play_at_least_one_sport_l503_503316


namespace find_b_value_l503_503259

theorem find_b_value (a b : ℤ) (h₁ : a + 2 * b = 32) (h₂ : |a| > 2) (h₃ : a = 4) : b = 14 :=
by
  -- proof goes here
  sorry

end find_b_value_l503_503259


namespace probability_of_specific_event_l503_503148

noncomputable def adam_probability := 1 / 5
noncomputable def beth_probability := 2 / 9
noncomputable def jack_probability := 1 / 6
noncomputable def jill_probability := 1 / 7
noncomputable def sandy_probability := 1 / 8

theorem probability_of_specific_event :
  (1 - adam_probability) * beth_probability * (1 - jack_probability) * jill_probability * sandy_probability = 1 / 378 := by
  sorry

end probability_of_specific_event_l503_503148


namespace probability_other_side_red_l503_503126

theorem probability_other_side_red
    (total_cards : ℕ)
    (black_black_cards : ℕ)
    (black_red_cards : ℕ)
    (red_red_cards : ℕ)
    (total_sides : ℕ)
    (red_sides_black_red : ℕ)
    (red_sides_red_red : ℕ)
    (observed_red_sides : ℕ)
    (observed_black_red_sides : ℕ)
    (observed_red_red_sides : ℕ)
    (observed_red : nat)
    (total_red_sides : nat)
    (prob : nat) :
    total_cards = 8 →
    black_black_cards = 4 →
    black_red_cards = 2 →
    red_red_cards = 2 →
    total_sides = total_cards * 2 →
    red_sides_black_red = black_red_cards * 1 →
    red_sides_red_red = red_red_cards * 2 →
    observed_red_sides = red_sides_black_red + red_sides_red_red →
    observed_black_red_sides = red_sides_black_red →
    observed_red_red_sides = red_sides_red_red →
    observed_red = observed_red_red_sides →
    total_red_sides = red_sides_black_red + red_sides_red_red →
    prob = observed_red_red_sides / total_red_sides →
    prob = 2 / 3 :=
sorry

end probability_other_side_red_l503_503126


namespace find_u_v_sum_l503_503961

variable (u v : ℕ)

def is_arithmetic_sequence (a d : ℕ) (terms : List ℕ) := 
  ∀ i j, i < j → j < terms.length → terms.get i + (j - i) * d = terms.get j

theorem find_u_v_sum :
  is_arithmetic_sequence 3 6 [3, 9, 15, u, v, 39] → u + v = 60 :=
by
  sorry

end find_u_v_sum_l503_503961


namespace total_resistance_l503_503695

theorem total_resistance (x y z w : ℝ) (hx : x = 5) (hy : y = 6) (hz : z = 9) (hw : w = 7) : 
  let R_parallel := (1/x + 1/y + 1/z)⁻¹ in
  let R := R_parallel + w in
  R = 391/43 :=
by
  sorry

end total_resistance_l503_503695


namespace find_center_of_circle_l503_503511

noncomputable def center_of_circle_tangent_to_parabola : (ℝ × ℝ) :=
  let a : ℝ := 3 in
  let b : ℝ := 97 / 10 in
  (a, b)

theorem find_center_of_circle :
  ∃ a b : ℝ, (0, 3) ∈ {p : ℝ × ℝ | (p.1 - a) ^ 2 + (p.2 - b) ^ 2 = (3 - a) ^ 2 + (9 - b) ^ 2} ∧
              ∀ y, y = x^2 → deriv y 3 = 6 →
              a = 6b - 57 ∧
              (0 - a)^2 + (3 - b)^2 = (3 - a)^2 + (9 - b)^2 ∧
              (a, b) = (3, 97/10) :=
by
  sorry

end find_center_of_circle_l503_503511


namespace number_of_squares_in_H_l503_503176

/-!
Define the set H as consisting of points (x, y) with integer coordinates where 2 ≤ |x| ≤ 6 and 
2 ≤ |y| ≤ 6. We want to prove that the number of squares with side length at least 4 can have 
all their four vertices in H is 16.
-/

/-- A point in the plane -/
structure Point where
  x : ℤ
  y : ℤ
  deriving DecidableEq

/-- The set H consists of points (x, y) such that 2 ≤ |x| ≤ 6 and 2 ≤ |y| ≤ 6 -/
def H : set Point := {p : Point | (2 ≤ |p.x| ∧ |p.x| ≤ 6) ∧ (2 ≤ |p.y| ∧ |p.y| ≤ 6)}

-- We aim to prove that there are 16 squares of side length at least 4 whose vertices are in H.
theorem number_of_squares_in_H : 
  (∃ H : set Point, (∀ p : Point, p ∈ H ↔ (2 ≤ |p.x| ∧ |p.x| ≤ 6) ∧ (2 ≤ |p.y| ∧ |p.y| ≤ 6)) 
  → ∑ x in H, ∑ y in H, 4 = 16 :=
by
  sorry

end number_of_squares_in_H_l503_503176


namespace arithmetic_sequence_sum_l503_503693

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 4 + a 8 = 4 →
  S 11 + a 6 = 24 :=
by
  intros a S h1 h2
  sorry

end arithmetic_sequence_sum_l503_503693


namespace sum_of_integer_pair_l503_503871

theorem sum_of_integer_pair (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 10) (h3 : 1 ≤ b) (h4 : b ≤ 10) (h5 : a * b = 14) : a + b = 9 := 
sorry

end sum_of_integer_pair_l503_503871


namespace positional_relation_MN_and_beta_l503_503676

variables {α : Type*} [euclidean_space α] 

def midpoint (A B : α) : α := (1/2) • A + (1/2) • B
def line (A B : α) : set α := {P | ∃ t : ℝ, P = A + t • (B - A)}
def plane (A B C : α) : set α := {P | ∃ u v : ℝ, P = A + u • (B - A) + v • (C - A)}

variables 
(A B C M N : α) 
(β : set α)

/-- M and N are the midpoints of sides AB and AC of triangle ABC -/
def M_is_mid_AB : Prop := M = midpoint A B
def N_is_mid_AC : Prop := N = midpoint A C

/-- Plane β passes through line BC -/
def plane_contains_line_BC : Prop := ∀ P, P ∈ line B C → P ∈ β

/-- MN is parallel to β or MN is contained in β -/
def target_statement : Prop := (∀ P, P ∈ line M N → P ∈ β) ∨ (∀ P ∈ line M N, ∃ Q ∈ β, P - Q = P - P)

theorem positional_relation_MN_and_beta
  (hm : M_is_mid_AB M A B)
  (hn : N_is_mid_AC N A C)
  (hb : plane_contains_line_BC β B C) :
  target_statement M N β := 
sorry

end positional_relation_MN_and_beta_l503_503676


namespace fruit_drink_total_l503_503498

def total_ounces_of_drink (T : ℕ) :=
  (0.15 * T) + (0.60 * T) + 30 = T

theorem fruit_drink_total (T : ℕ) (h1 : 0.15 * T = 0.15 * T) (h2 : 0.60 * T = 0.60 * T) (h3 : 30 = 30) :
  total_ounces_of_drink T → T = 120 :=
sorry

end fruit_drink_total_l503_503498


namespace numTrianglesFromDecagon_is_120_l503_503663

noncomputable def numTrianglesFromDecagon : ℕ := 
  nat.choose 10 3

theorem numTrianglesFromDecagon_is_120 : numTrianglesFromDecagon = 120 := 
  by
    -- Form the combination
    have : numTrianglesFromDecagon = nat.choose 10 3 := rfl

    -- Calculate
    have calc₁ : nat.choose 10 3 = 10 * 9 * 8 / (3 * 2 * 1) := by 
      exact nat.choose_eq_div
      simp

    -- Simplify the calculation to 120
    have : 10 * 9 * 8 / (3 * 2 * 1) = 120 := by 
      norm_num 

    exact eq.trans this.symm calc₁.symm ⟩

end numTrianglesFromDecagon_is_120_l503_503663


namespace length_OB_l503_503700

-- Definitions of points A, B, and the origin O
def A : ℝ × ℝ × ℝ := (1, 2, 3)
def B : ℝ × ℝ × ℝ := (0, 2, 3)
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Definition of the Euclidean distance function in 3D
def Euclidean_distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

-- Specification of the problem statement
theorem length_OB : Euclidean_distance O B = Real.sqrt 13 := by
  sorry

end length_OB_l503_503700


namespace spinner_sections_equal_size_l503_503829

theorem spinner_sections_equal_size 
  (p : ℕ → Prop)
  (h1 : ∀ n, p n ↔ (1 - (1: ℝ) / n) ^ 2 = 0.5625) : 
  p 4 :=
by
  sorry

end spinner_sections_equal_size_l503_503829


namespace total_season_cost_l503_503344

noncomputable def production_cost_first_half := 11 * 1000
noncomputable def production_cost_second_half := 11 * (1000 + (1000 * 120/100))

noncomputable def total_production_cost := production_cost_first_half + production_cost_second_half

noncomputable def advertising_cost_first_10 := 10 * 500
noncomputable def r := 1.10
noncomputable def advertising_cost_rem_12 := 
  let a := 550
  let n := 12
  a * (1 - r^n) / (1 - r)

noncomputable def total_advertising_cost := advertising_cost_first_10 + advertising_cost_rem_12

noncomputable def actors_fee_first_half := 11 * 2000
noncomputable def actors_fee_second_half := 11 * 2500

noncomputable def total_actors_fee := actors_fee_first_half + actors_fee_second_half

noncomputable def final_total_cost := total_production_cost + total_advertising_cost + total_actors_fee

theorem total_season_cost : final_total_cost ≈ 101461.36 := by
  sorry

end total_season_cost_l503_503344


namespace locus_of_M_lies_on_circle_l503_503648

-- Definitions for the given conditions
variable (A B M : Point)
variable (k : ℝ) (h_k : k ≠ 0)
-- Assuming points A and B are distinct and AB = 2a
variable (a : ℝ) (h_a : a > 0) (h_dist : dist A B = 2 * a)
-- Midpoint O of A and B
def O : Point := midpoint A B

-- Main theorem to prove
theorem locus_of_M_lies_on_circle (M : Point) (A B : Point) (k : ℝ) (h_k : k ≠ 0) (a : ℝ) (h_a : a > 0) (h_dist : dist A B = 2 * a) :
  -- Condition that must be satisfied for M
  (vector MA) • (vector MB) = k^2 ↔ dist O M = sqrt (k^2 + a^2) :=
sorry

end locus_of_M_lies_on_circle_l503_503648


namespace simon_change_l503_503765

def pansy_price : ℝ := 2.50
def pansy_count : ℕ := 5
def hydrangea_price : ℝ := 12.50
def hydrangea_count : ℕ := 1
def petunia_price : ℝ := 1.00
def petunia_count : ℕ := 5
def discount_rate : ℝ := 0.10
def initial_payment : ℝ := 50.00

theorem simon_change : 
  let total_cost := (pansy_count * pansy_price) + (hydrangea_count * hydrangea_price) + (petunia_count * petunia_price)
  let discount := total_cost * discount_rate
  let cost_after_discount := total_cost - discount
  let change := initial_payment - cost_after_discount
  change = 23.00 :=
by
  sorry

end simon_change_l503_503765


namespace compare_abc_l503_503987

noncomputable def a : ℝ := Real.log 0.5 / Real.log 5
noncomputable def b : ℝ := 5 ^ 0.5
noncomputable def c : ℝ := 0.5 ^ 0.6

theorem compare_abc : a < c ∧ c < b := by
  sorry

end compare_abc_l503_503987


namespace carvings_per_shelf_l503_503024

def total_wood_carvings := 56
def num_shelves := 7

theorem carvings_per_shelf : total_wood_carvings / num_shelves = 8 := by
  sorry

end carvings_per_shelf_l503_503024


namespace retailer_profit_percentage_l503_503523

theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (h_wholesale_price : wholesale_price = 108)
  (h_retail_price : retail_price = 144)
  (h_discount_rate : discount_rate = 0.10) :
  (retail_price * (1 - discount_rate) - wholesale_price) / wholesale_price * 100 = 20 :=
by
  sorry

end retailer_profit_percentage_l503_503523


namespace evaluate_expression_l503_503564

theorem evaluate_expression :
  2 * (Nat.floor 1.999 + Nat.ceil 3.005) = 10 :=
by
  have h1 : Nat.floor (1.999 : ℝ) = 1 := sorry
  have h2 : Nat.ceil (3.005 : ℝ) = 4 := sorry
  calc
    2 * (Nat.floor 1.999 + Nat.ceil 3.005) = 2 * (1 + 4) : by rw [h1, h2]
    ... = 2 * 5 : rfl
    ... = 10 : rfl

end evaluate_expression_l503_503564


namespace distribute_paper_clips_l503_503740

theorem distribute_paper_clips (total_clips : ℕ) (boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : boxes = 9) :
  total_clips / boxes = clips_per_box ↔ clips_per_box = 9 :=
by
  sorry

end distribute_paper_clips_l503_503740


namespace slope_point_line_l503_503135

theorem slope_point_line (m b : ℝ) (h_m : m = 8) (h_point : ∃ b, ∀ x y, (x, y) = (-2, 4) → y = m * x + b) : m + b = 28 := by
  obtain ⟨b, h_eq⟩ := h_point
  specialize h_eq (-2) 4 rfl
  rw [h_m, mul_neg, neg_mul_eq_neg_mul]
  linarith
  sorry

end slope_point_line_l503_503135


namespace original_employees_l503_503107

variable (x : ℝ)

theorem original_employees (h : 0.87 * x = 181) : x ≈ 208 := by
  sorry

end original_employees_l503_503107


namespace divisors_64n2_l503_503981

def prime_factors (n : ℕ) : List (ℕ × ℕ) :=
  sorry -- Assuming there is a function to get prime factors

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc (p, k) => acc * (k + 1)) 1

variable {n : ℕ}

theorem divisors_64n2 (hn : 0 < n) (hdiv : num_divisors (prime_factors (150 * n ^ 3)) = 150) :
  num_divisors (prime_factors (64 * n ^ 2)) = 153 :=
begin
  sorry
end

end divisors_64n2_l503_503981


namespace probability_triangle_nonagon_l503_503210

-- Define the total number of ways to choose 3 vertices from 9 vertices
def total_ways_to_choose_triangle : ℕ := Nat.choose 9 3

-- Define the number of favorable outcomes
def favorable_outcomes_one_side : ℕ := 9 * 5
def favorable_outcomes_two_sides : ℕ := 9

def total_favorable_outcomes : ℕ := favorable_outcomes_one_side + favorable_outcomes_two_sides

-- Define the probability as a rational number
def probability_at_least_one_side_nonagon (total: ℕ) (favorable: ℕ) : ℚ :=
  favorable / total
  
-- Theorem stating the probability
theorem probability_triangle_nonagon :
  probability_at_least_one_side_nonagon total_ways_to_choose_triangle total_favorable_outcomes = 9 / 14 :=
by
  sorry

end probability_triangle_nonagon_l503_503210


namespace find_x_when_w_is_10_l503_503877

noncomputable def prop_proportional_rels (x w : ℝ) : Prop :=
  ∃ (m n k : ℝ), (x = m * (n / 5) ^ 3 / 125 / k ^ (3/2) ) ∧ (w = 5) ∧ (x = 8)

theorem find_x_when_w_is_10 :
  prop_proportional_rels 8 5 → (∃ m n k : ℝ, x = m * (n / 10) ^ 3 / 1000 / k ^ (3/2) ∧ w=10) → x = 1 / 8 :=
begin
  sorry
end

end find_x_when_w_is_10_l503_503877


namespace administrators_rotation_l503_503065

variables {α : Type} [linear_ordered_field α]

-- Let ∆ABC be an equilateral triangle with side length s
noncomputable def is_equilateral_triangle (A B C: α × α) (s: α) := 
    dist A B = s ∧ dist B C = s ∧ dist C A = s

-- Path of administrator A: A -> B -> C -> A
noncomputable def path_A (A B C: α × α) := 
    [A, B, C, A]

-- Path of administrator B: D -> C -> A -> B -> D
noncomputable def path_B (D C A B: α × α) :=
    [D, C, A, B, D]

-- External angle sum property of triangle 
noncomputable def external_angle_sum_triangle := (3 : α) * 120 = 360

theorem administrators_rotation {A B C D: α × α} {s: α}
  (h_eq_triangle : is_equilateral_triangle A B C s) 
  (h_path_A : path_A A B C)
  (h_path_B : path_B D C A B)
  (h_external_angle_sum: external_angle_sum_triangle) :
  A_rotation = 240 ∧ B_rotation = 360 :=
sorry

end administrators_rotation_l503_503065


namespace triangle_existence_areas_l503_503713

-- Definitions from the conditions
namespace TriangleABC

variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to angles A, B, C
variables {AD: ℝ} -- Median length

-- Given conditions
def condition_1 : Prop := a = Real.sqrt 6
def condition_2 : Prop := Real.sin B ^ 2 + Real.sin C ^ 2 = Real.sin A ^ 2 + (2 * Real.sqrt 3 / 3) * Real.sin A * Real.sin B * Real.sin C
def median_AD : Prop := AD = Real.sqrt 10 / 2
def sum_bc : Prop := b + c = 2 * Real.sqrt 3
def cos_B : Prop := Real.cos B = -3 / 5

-- Equivalent proof problem
theorem triangle_existence_areas :
  (condition_1 ∧ condition_2 ∧ median_AD → ∃ (area: ℝ), area = Real.sqrt 3 / 2) ∧
  (condition_1 ∧ condition_2 ∧ sum_bc → ∃ (area: ℝ), area = Real.sqrt 3 / 2) ∧
  (condition_1 ∧ condition_2 ∧ cos_B → False) :=
by sorry

end TriangleABC

end triangle_existence_areas_l503_503713


namespace division_remainder_l503_503755

theorem division_remainder (dividend quotient divisor remainder : ℕ) 
  (h_dividend : dividend = 12401) 
  (h_quotient : quotient = 76) 
  (h_divisor : divisor = 163) 
  (h_remainder : dividend = quotient * divisor + remainder) : 
  remainder = 13 := 
by
  sorry

end division_remainder_l503_503755


namespace log_sum_roots_l503_503619

theorem log_sum_roots (α β : ℝ) (h1 : Math.log(α) ^ 2 - Math.log(α) - 2 = 0)
  (h2 : Math.log(β) ^ 2 - Math.log(β) - 2 = 0) :
  Math.logBase α β + Math.logBase β α = -5 / 2 :=
sorry

end log_sum_roots_l503_503619


namespace integer_values_of_expression_l503_503199

theorem integer_values_of_expression (k : ℤ) :
  ∃ n : ℤ, n = 20 * k - 5 ∧ 
  (∃ m : ℤ, m = (1 / 12) * (8 * sin (π * n / 10) - sin (3 * π * n / 10) + 4 * cos (π * n / 5) + 1)) :=
begin
  sorry
end

end integer_values_of_expression_l503_503199


namespace pokemon_cards_count_l503_503407

theorem pokemon_cards_count :
  let sally_initial := 27
  let dan_gives := 41
  let sally_after_dan := sally_initial + dan_gives
  let percent_dup := 30
  let dup_cards := (percent_dup * dan_gives) / 100
  let sally_after_giving_away := sally_after_dan - real.floor dup_cards
  let sally_wins := 15
  let sally_after_winning := sally_after_giving_away + sally_wins
  let sally_loses := 20
  let sally_final := sally_after_winning - sally_loses
  in sally_final = 51 := by
  sorry

end pokemon_cards_count_l503_503407


namespace negation_universal_proposition_l503_503430

theorem negation_universal_proposition :
  (¬∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end negation_universal_proposition_l503_503430


namespace base5_division_example_l503_503536

  theorem base5_division_example : 
    (convert_from_base 5 2013) / (convert_from_base 5 23) = convert_from_base 5 34 :=
  by
    sorry

  def convert_from_base (b : ℕ) (n : ℕ) : ℕ :=
    let digits := digit_list b n
    list.foldl (λ sum digit, sum * b + digit) 0 digits

  def digit_list (b : ℕ) (n : ℕ) : list ℕ :=
    if n < b then [n] else (n % b) :: digit_list b (n / b)
  
end base5_division_example_l503_503536


namespace parabola_equation_l503_503038

theorem parabola_equation : 
  (∃ (a b c d e f : ℤ), 
    c > 0 ∧
    Int.gcd (Int.natAbs a) (Int.gcd (Int.natAbs b) (Int.gcd (Int.natAbs c) (Int.gcd (Int.natAbs d) (Int.gcd (Int.natAbs e) (Int.natAbs f))))) = 1 ∧ 
    a * 2^2 + b * 2 * 7 + c * 7^2 + d * 2 + e * 7 + f = 0 ∧
    a * 0^2 + b * 0 * 5 + c * 5^2 + d * 0 + e * 5 + f = 0 ∧
    (∃ (k : ℚ), ∀ x y, x = k * (y - 5)^2 → a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0) ↔ 
  (∃ (a b c d e f : ℤ), a = 0 ∧ b = 0 ∧ c = 1 ∧ d = -2 ∧ e = -10 ∧ f = 25) := 
sorry

end parabola_equation_l503_503038


namespace find_MP_l503_503362

-- Definitions based on problem conditions
variables {A B C D M P : Point}
variables {AB CD AP MP : ℝ}

-- Given trapezoid conditions
axiom AB_parallel_CD : ∀ (trapezoid : Trapezoid), parallel (AB) (CD)
axiom AD_eq_BD : AD = BD
axiom midpoint_AB : midpoint M AB

-- Given circumcircle condition
axiom P_second_intersection : is_circumcircle_intersection P C (triangle B C D) (diagonal A C)

-- Given lengths
axiom BC_length : BC = 27
axiom CD_length : CD = 25
axiom AP_length : AP = 10

-- Theorem to prove
theorem find_MP (MP : ℝ) : MP = 27 / 5 ∧ 100 * 27 + 5 = 2705 := 
by
  sorry

end find_MP_l503_503362


namespace num_large_posters_l503_503918

-- Define the constants
def total_posters : ℕ := 50
def small_posters : ℕ := total_posters * 2 / 5
def medium_posters : ℕ := total_posters / 2
def large_posters : ℕ := total_posters - (small_posters + medium_posters)

-- Theorem to prove the number of large posters
theorem num_large_posters : large_posters = 5 :=
by
  sorry

end num_large_posters_l503_503918


namespace sum_first_100_terms_l503_503439

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 0
  else match n with
       | 1 => 2
       | k+1 => sequence k / (sequence k - 1)
       end

def sum_first_n_terms (n : ℕ) : ℝ :=
  (Finset.range n).sum sequence

theorem sum_first_100_terms :
  sum_first_n_terms 100 = 103 / 2 :=
sorry

end sum_first_100_terms_l503_503439


namespace initial_rate_is_30_l503_503156

-- Defining the initial conditions and the problem
def initial_rate (x : ℝ) :=
  let t := 60 / x in
  let total_time := t + 1 in
  40 = 120 / total_time

-- Stating the theorem we need to prove
theorem initial_rate_is_30 : initial_rate 30 :=
by
  sorry

end initial_rate_is_30_l503_503156


namespace complex_number_quadrant_l503_503960

def quadrant_of_complex (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First Quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second Quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third Quadrant"
  else if z.re > 0 ∧ z.im < 0 then "Fourth Quadrant"
  else "On an Axis"

noncomputable def z : ℂ := (2*complex.I) / (2 - complex.I)

theorem complex_number_quadrant : quadrant_of_complex z = "Second Quadrant" := by
  sorry

end complex_number_quadrant_l503_503960


namespace brenda_travel_distance_l503_503534

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

theorem brenda_travel_distance :
  let p1 := (-3 : ℝ, 6 : ℝ)
  let p2 := (1 : ℝ, 1 : ℝ)
  let p3 := (6 : ℝ, -3 : ℝ)
  distance p1 p2 + distance p2 p3 = 2 * real.sqrt 41 := by
  sorry

end brenda_travel_distance_l503_503534


namespace kaleb_initial_cherries_l503_503347

/-- Kaleb's initial number of cherries -/
def initial_cherries : ℕ := 67

/-- Cherries that Kaleb ate -/
def eaten_cherries : ℕ := 25

/-- Cherries left after eating -/
def left_cherries : ℕ := 42

/-- Prove that the initial number of cherries is 67 given the conditions. -/
theorem kaleb_initial_cherries :
  eaten_cherries + left_cherries = initial_cherries :=
by
  sorry

end kaleb_initial_cherries_l503_503347


namespace base_case_0_l503_503299

theorem base_case_0 : 
  let b (n : ℕ) := match n with
  | 0     => Real.cos (Real.pi / 30 ) ^ 2
  | n + 1 => 4 * b n * (1 - b n)
  (n : ℕ) : 
  (∃ n, b n = b 0) :=
by
  sorry

end base_case_0_l503_503299


namespace remainder_p_q_l503_503724

noncomputable def i : ℂ := Complex.I
def S := Finset.range 2014

theorem remainder_p_q :
  let p := 0
  let q := 2^1007
  (|p| + |q|) % 1000 = 872 :=
by
  let p := 0
  let q := 2^1007
  have hp : |p| = 0 := abs_zero
  have hq : |q| = q := abs_of_nonneg (pow_pos zero_lt_two 1007)
  have hq_mod : q % 1000 = 872 := sorry
  show (|p| + |q|) % 1000 = 872 from by
    rw [hp, hq, add_zero, hq_mod]

end remainder_p_q_l503_503724


namespace exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l503_503177

def omega (n : Nat) : Nat :=
  if n = 1 then 0 else n.factors.toFinset.card

theorem exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2 :
  ∃ᶠ n in atTop, ∃ k : Nat, n = 2^k ∧
    omega n < omega (n + 1) ∧
    omega (n + 1) < omega (n + 2) :=
sorry

end exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l503_503177


namespace f_has_zero_point_l503_503425

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x - 1

theorem f_has_zero_point : 
    (∃ c ∈ Ioo (1 / 4 : ℝ) (1 / 2 : ℝ), f c = 0) :=
by
  have h1 : continuous_on f (Ioi (0 : ℝ)) := sorry,
  have h2 : ∀ x y, x < y → f x < f y := sorry,
  have h3 : f (1 / 4) < 0 := by
    norm_num,
    linarith,
  have h4 : f (1 / 2) > 0 := by
    norm_num,
    linarith,
  exact sorry

end f_has_zero_point_l503_503425


namespace triangle_congruence_sum_l503_503442

theorem triangle_congruence_sum (m n : ℝ) (h1 : Set {2, 5, m} = Set {n, 6, 2}) : m + n = 11 :=
by
  sorry

end triangle_congruence_sum_l503_503442


namespace fixed_monthly_fee_l503_503949

theorem fixed_monthly_fee (x y : ℝ)
  (h₁ : x + y = 18.70)
  (h₂ : x + 3 * y = 34.10) : x = 11.00 :=
by sorry

end fixed_monthly_fee_l503_503949


namespace flask_forces_l503_503586

theorem flask_forces (r : ℝ) (ρ g h_A h_B h_C V : ℝ) (A : ℝ) (FA FB FC : ℝ) (h1 : r = 2)
  (h2 : A = π * r^2)
  (h3 : V = A * h_A ∧ V = A * h_B ∧ V = A * h_C)
  (h4 : FC = ρ * g * h_C * A)
  (h5 : FA = ρ * g * h_A * A)
  (h6 : FB = ρ * g * h_B * A)
  (h7 : h_C > h_A ∧ h_A > h_B) : FC > FA ∧ FA > FB := 
sorry

end flask_forces_l503_503586


namespace Vitya_needs_58_offers_l503_503095

theorem Vitya_needs_58_offers :
  ∃ k : ℕ, (log 0.01 / log (12 / 13) < k) ∧ k = 58 :=
by
  sorry

end Vitya_needs_58_offers_l503_503095


namespace number_of_real_z10_l503_503062

theorem number_of_real_z10 (z : ℂ) (h : z^30 = 1) : 
  (∃ k : ℕ, k < 30 ∧ z = exp (2 * real.pi * complex.I * k / 30)) → 
  set.card ((λ k : ℕ, k < 30 ∧ z = exp (2 * real.pi * complex.I * k / 3)) '' {k | k < 30 ∧ 3 ∣ k }) = 10 := 
begin 
  sorry
end

end number_of_real_z10_l503_503062


namespace smallest_b_for_undefined_inverse_mod_70_77_l503_503490

theorem smallest_b_for_undefined_inverse_mod_70_77 (b : ℕ) :
  (∀ k, k < b → k * 1 % 70 ≠ 1 ∧ k * 1 % 77 ≠ 1) ∧ (b * 1 % 70 ≠ 1) ∧ (b * 1 % 77 ≠ 1) → b = 7 :=
by sorry

end smallest_b_for_undefined_inverse_mod_70_77_l503_503490


namespace floor_of_3_9_l503_503565

theorem floor_of_3_9 : Real.floor 3.9 = 3 :=
by
  sorry

end floor_of_3_9_l503_503565


namespace problem_statement_l503_503167

def p (f : ℝ → ℝ) : ℕ :=
  if h : ∃ x, f x = ⊥ then 1 else 0

def q (f : ℝ → ℝ) : ℕ :=
  if h : ∃ x, ∂ (f x) (x → x) = ⊤ then 2 else 0

def r (f : ℝ → ℝ) : ℕ :=
  if h : ∃ x, ∂ (f x) (x → x) = 0 then 1 else 0

def s (f : ℝ → ℝ) : ℕ :=
  if h : ∃ x, ∂ (f x) (x → x) ∈ ℤ then 0 else 0

noncomputable def f : ℝ → ℝ :=
  λ x, (x^2 + 5 * x + 6) / (x^3 - 2 * x^2 - x + 2)

theorem problem_statement : p f + 2 * q f + 3 * r f + 4 * s f = 8 :=
by sorry

end problem_statement_l503_503167


namespace median_first_fifteen_integers_l503_503472

theorem median_first_fifteen_integers :
  let l := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] in
  let seventh := l.nth 6 in
  let eighth := l.nth 7 in
  (seventh.is_some ∧ eighth.is_some) →
  (seventh.get_or_else 0 + eighth.get_or_else 0) / 2 = 7.5 :=
by
  sorry

end median_first_fifteen_integers_l503_503472


namespace johns_age_in_8_years_l503_503979

theorem johns_age_in_8_years :
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  current_age + 8 = twice_age_five_years_ago :=
by
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  sorry

end johns_age_in_8_years_l503_503979


namespace max_value_expression_l503_503253

theorem max_value_expression (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 48) :
  sqrt (x^2 + y^2 - 4 * x + 4) + sqrt (x^2 + y^2 - 2 * x + 4 * y + 5) = 8 + sqrt 13 :=
sorry

end max_value_expression_l503_503253


namespace particular_integral_eq1_particular_integral_eq2_particular_integral_eq3_particular_integral_eq4_l503_503853

-- Definition and conditions for equation 1
def eq1 (y : ℝ → ℝ) :=
  ∀ x, (deriv^[2] y x) + 4 * (deriv y x) + 4 * y x = real.exp (-9 * x) * (real.sec x)^2

def sol1 (y : ℝ → ℝ) :=
  ∀ x, y x = -real.exp (-2 * x) * real.log (abs (real.cos x))

-- Proof statement for equation 1
theorem particular_integral_eq1 : ∃ y, eq1 y ∧ sol1 y := by
  sorry

-- Definition and conditions for equation 2
def eq2 (y : ℝ → ℝ) :=
  ∀ x, (deriv^[2] y x) + 5 * (deriv y x) + 6 * y x = (real.exp (2 * x) + 1) ^ (-3/2)

def sol2 (y : ℝ → ℝ) :=
  ∀ x, y x = -real.exp (-3 * x) * real.log(real.exp x + real.sqrt (real.exp (2 * x) + 1))

-- Proof statement for equation 2
theorem particular_integral_eq2 : ∃ y, eq2 y ∧ sol2 y := by
  sorry

-- Definition and conditions for equation 3
def eq3 (y : ℝ → ℝ) :=
  ∀ x, (deriv^[3] y x) - 3 * (deriv^[2] y x) + 3 * (deriv y x) - y x = 5 * x^3 * real.exp x + 3 * real.exp (2 * x)

def sol3 (y : ℝ → ℝ) :=
  ∀ x, y x = real.exp x * (x^6 / 24 + 3 * real.exp x)

-- Proof statement for equation 3
theorem particular_integral_eq3 : ∃ y, eq3 y ∧ sol3 y := by
  sorry

-- Definition and conditions for equation 4
def eq4 (y : ℝ → ℝ) :=
  ∀ x, (deriv@[2] y x) + 4 * y x = (real.cos x)^3

def sol4 (y : ℝ → ℝ) :=
  ∀ x, y x = (1/4) * (real.cos x - (1/5) * real.cos(3 * x))

-- Proof statement for equation 4
theorem particular_integral_eq4 : ∃ y, eq4 y ∧ sol4 y := by
  sorry

end particular_integral_eq1_particular_integral_eq2_particular_integral_eq3_particular_integral_eq4_l503_503853


namespace eccentricity_of_hyperbola_l503_503276

-- Definition of the hyperbola and associated variables
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Definition of the eccentricity given by the solution
def hyperbola_eccentricity (a b : ℝ) (c: ℝ) : ℝ := c / a

-- Statement of the proof problem
theorem eccentricity_of_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (c : ℝ) (h1 : b^2 = a * c) (h2 : c^2 = a^2 + b^2) :
  hyperbola_eccentricity a b c = (Real.sqrt 5 + 1) / 2 := by
  -- The following line indicates the need for a proof
  sorry

end eccentricity_of_hyperbola_l503_503276


namespace gcd_of_polynomials_l503_503240

theorem gcd_of_polynomials (b : ℤ) (h : 2460 ∣ b) : 
  Int.gcd (b^2 + 6 * b + 30) (b + 5) = 30 :=
sorry

end gcd_of_polynomials_l503_503240


namespace perimeter_excentral_triangle_leq_l503_503593

variables (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
variables (R q : ℝ)

-- Assuming the circumradius and excentral triangle perimeter as conditions.
axiom circumradius_triangle_ABC : ℝ
axiom perimeter_excentral_triangle : ℝ

theorem perimeter_excentral_triangle_leq : 
  (q ≤ 6 * real.sqrt 3 * R) := 
sorry

end perimeter_excentral_triangle_leq_l503_503593


namespace evaluate_log_subtraction_l503_503187

theorem evaluate_log_subtraction : 
  log 3 243 - log 3 (1/27) = 8 :=
by
  sorry

end evaluate_log_subtraction_l503_503187


namespace zero_in_interval_l503_503556

noncomputable def g (x : ℝ) : ℝ := 2 * Real.exp(x) + x - 7

theorem zero_in_interval : ∃ x ∈ Ioo (1 : ℝ) 2, g x = 0 := by
  sorry

end zero_in_interval_l503_503556


namespace smallest_t_l503_503589

def p (k : ℕ) : ℕ := sorry -- define the smallest prime that does not divide k

def X (k : ℕ) : ℕ :=
  if p k > 2 then ∏ (i : ℕ) in Finset.filter (λ p, Prime p ∧ p < (p k)) (Finset.range (p k)), i
  else 1

def x_seq : ℕ → ℕ
| 0 => 1
| n+1 => (x_seq n * p (x_seq n)) / X (x_seq n)

theorem smallest_t : ∃ t : ℕ, t = 149 ∧ x_seq t = 2090 := sorry

end smallest_t_l503_503589


namespace rohan_savings_l503_503763

-- Definitions for conditions
def salary := 12500
def food_expense_percent := 0.40
def house_rent_expense_percent := 0.20
def entertainment_expense_percent := 0.10
def conveyance_expense_percent := 0.10

-- Lean statement to prove
theorem rohan_savings : 
  let total_expense_percent := food_expense_percent + house_rent_expense_percent + entertainment_expense_percent + conveyance_expense_percent in
  let savings_percent := 1 - total_expense_percent in
  savings_percent * salary = 2500 := by
    sorry

end rohan_savings_l503_503763


namespace monotonicity_and_range_of_k_l503_503633

noncomputable def f (x k : ℝ) := 2 * real.exp x - k * x - 2

theorem monotonicity_and_range_of_k (k : ℝ) :
  (∀ x > 0, (2 * real.exp x - k) > 0 ∨ (2 * real.exp x - k) < 0) ∧
  (∃ m > 0, ∀ x ∈ set.Ioo 0 m, |f x k| > 2 * x) → k > 4 :=
by
  sorry

end monotonicity_and_range_of_k_l503_503633


namespace hexagon_twice_triangle_area_l503_503912

variable {R : Type} [OrderedCommRing R]

structure Point (R : Type) := 
  (x : R) (y : R)

structure Triangle (R : Type) :=
  (A B C : Point R)
  (inscribed : ∃ O : Point R, ∀ p : Point R, (p = A ∨ p = B ∨ p = C) → dist O p = dist O A)

def reflect (O p : Point R) : Point R := 
  Point.mk (2 * O.x - p.x) (2 * O.y - p.y)

def hexagon_area (A B C A1 B1 C1 : Point R) : R := 
  (tri_area A B C) + (tri_area A1 B1 C1) 
  + (tri_area A B1 A1) + (tri_area B C1 B1) + (tri_area C A1 C1) + (tri_area A B C1)

def tri_area (A B C : Point R) : R := 
  abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2

theorem hexagon_twice_triangle_area (T : Triangle R) :
  ∀ (O : Point R) A1 B1 C1,
    T.inscribed O →
    A1 = reflect O T.A → B1 = reflect O T.B → C1 = reflect O T.C →
    hexagon_area T.A T.B T.C A1 B1 C1 = 2 * tri_area T.A T.B T.C :=
by
  intro O A1 B1 C1 inscribed hA1 hB1 hC1
  sorry

end hexagon_twice_triangle_area_l503_503912


namespace monotonically_increasing_interval_minimum_value_a_l503_503266

noncomputable def f (x : ℝ) := 4 * sin x * sin (x + π / 3)

theorem monotonically_increasing_interval :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 6) ≤ x ∧ x ≤ (k * π + π / 3) → monotone_on f (Set.Icc (k * π - π / 6) (k * π + π / 3)) :=
by sorry

theorem minimum_value_a :
  ∀ (A B C a b c : ℝ), 
  b + c = 6 → 
  f A = 3 → 
  A = π / 3 →
  triangle_correct A B C a b c → 
  a = 3 :=
by sorry

end monotonically_increasing_interval_minimum_value_a_l503_503266


namespace area_of_closed_figure_l503_503028

-- Defining necessary conditions
def y1 (y : ℝ) : Prop := y = 1 / 2
def y2 (y : ℝ) : Prop := y = 2
def curve (x y : ℝ) : Prop := y = 1 / x

-- The integral calculation for the area
def S : ℝ := ∫ (y : ℝ) in (1/2)..2, (1 / y)

-- The proof statement
theorem area_of_closed_figure : S = 2 * Real.log 2 :=
by
  sorry

end area_of_closed_figure_l503_503028


namespace triangle_area_ratio_l503_503711

theorem triangle_area_ratio (P Q R S : Type) [triangle P Q R]
  (hpq : PQ = 21) (hpr : PR = 28) (hqr : QR = 37)
  (hps : is_angle_bisector PS) :
  ratio_of_areas PQS PRS = 3 / 4 :=
sorry

end triangle_area_ratio_l503_503711


namespace slope_of_AB_l503_503229

theorem slope_of_AB (A B : (ℕ × ℕ)) (hA : A = (3, 4)) (hB : B = (2, 3)) : 
  (B.2 - A.2) / (B.1 - A.1) = 1 := 
by 
  sorry

end slope_of_AB_l503_503229


namespace algae_colony_growth_l503_503146

def initial_cells : ℕ := 5
def days : ℕ := 10
def tripling_period : ℕ := 3
def cell_growth_ratio : ℕ := 3

noncomputable def cells_after_n_days (init_cells : ℕ) (day_count : ℕ) (period : ℕ) (growth_ratio : ℕ) : ℕ :=
  let steps := day_count / period
  init_cells * growth_ratio^steps

theorem algae_colony_growth : cells_after_n_days initial_cells days tripling_period cell_growth_ratio = 135 :=
  by sorry

end algae_colony_growth_l503_503146


namespace hyperbola_eccentricity_l503_503426

def f1 := (-1 : ℝ, 0 : ℝ)
def f2 := (1 : ℝ, 0 : ℝ)
def parabola : ℝ → ℝ → Prop := λ x y, y^2 = 4 * x
def condition (P : ℝ × ℝ) : Prop :=
  let F2P := (P.1 - f2.1, P.2 - f2.2)
      F2F1 := (f1.1 - f2.1, f1.2 - f2.2)
  in (F2P.1 + F2F1.1, F2P.2 + F2F1.2) ⋅ (F2P.1 - F2F1.1, F2P.2 - F2F1.2) = 0

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity :
  ∃ P : ℝ × ℝ, 
  parabola P.1 P.2 ∧ condition P ∧ (eccentricity (real.sqrt 2 - 1) 1 = 1 + real.sqrt 2) :=
sorry

end hyperbola_eccentricity_l503_503426


namespace boris_stopped_saving_in_may_2020_l503_503943

theorem boris_stopped_saving_in_may_2020 :
  ∀ (B V : ℕ) (start_date_B start_date_V stop_date : ℕ), 
    (∀ t, start_date_B + t ≤ stop_date → B = 200 * t) →
    (∀ t, start_date_V + t ≤ stop_date → V = 300 * t) → 
    V = 6 * B →
    stop_date = 17 → 
    B / 200 = 4 → 
    stop_date - B/200 = 2020 * 12 + 5 :=
by
  sorry

end boris_stopped_saving_in_may_2020_l503_503943


namespace calculate_T_l503_503377

noncomputable def T (S : ℕ) : ℝ := Real.sqrt (Real.log (1 + (∑ i in Finset.range (S + 1), (2 : ℝ) ^ i)) / Real.log 2)

theorem calculate_T : T 120 = 11 :=
by
  have F : ℝ := ∑ i in Finset.range 121, (2 : ℝ) ^ i
  have log2 := Real.log (2 : ℝ)
  have logF := Real.log (F + 1)
  have T_val := Real.sqrt (logF / log2)
  have simplified_F : F = 2 ^ 121 - 1 := sorry
  rw [simplified_F, ←Real.log_pow (2 : ℝ) 121, Real.log_mul_self log2 121] at logF
  rw [Real.sqrt_eq_iff_sq_eq, Real.pow_two, Real.div_self log2] at T_val
  exact T_val
  linarith
  exact_mod_cast Real.log_pos one_lt_two
  linarith

end calculate_T_l503_503377


namespace eccentricity_of_ellipse_l503_503257

theorem eccentricity_of_ellipse 
  (a b : ℝ) (h_ab : a > b) (h_b0 : b > 0)
  (M N P : ℝ × ℝ) 
  (hM : M.1^2 / a^2 + M.2^2 / b^2 = 1)
  (hN : N.1^2 / a^2 + N.2^2 / b^2 = 1)
  (hN_sym : N = (-M.1, -M.2))
  (hP : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (k1 k2 : ℝ)
  (h_slope_PM : k1 = (M.2 - P.2) / (M.1 - P.1))
  (h_slope_PN : k2 = (N.2 - P.2) / (N.1 - P.1))
  (h_k1k2 : abs(k1 * k2) = 1 / 4) :
  sqrt(3) / 2 = (sqrt(a^2 - b^2) / a) :=
sorry

end eccentricity_of_ellipse_l503_503257


namespace democrats_ratio_l503_503826

variables (M F : ℕ)
variables (total_participants : ℕ := 870)
variables (female_democrats : ℕ := 145)

-- Given conditions
def condition1 : Prop := M + F = total_participants
def condition2 : Prop := M / 4 + female_democrats = total_participants / 3

-- Proving the ratio of female democrats to total female participants
def ratio_to_prove : Prop := (female_democrats / F) = 1 / 2

theorem democrats_ratio
  (h1 : condition1)
  (h2 : condition2)
  : ratio_to_prove :=
sorry

end democrats_ratio_l503_503826


namespace minimum_possible_value_of_Box_l503_503293

theorem minimum_possible_value_of_Box : 
  ∃ (a b Box : ℤ), 
    (a ≠ b) ∧ (a ≠ Box) ∧ (b ≠ Box) ∧
    (a * b = 15) ∧ 
    (∀ x : ℤ, (a * x + b) * (b * x + a) = 15 * x ^ 2 + Box * x + 15) ∧ 
    (∃ p q : ℤ, (p * q = 15 ∧ p ≠ q ∧ p ≠ 34 ∧ q ≠ 34) → (Box = p^2 + q^2)) ∧ 
    Box = 34 :=
by
  sorry

end minimum_possible_value_of_Box_l503_503293


namespace number_of_large_posters_is_5_l503_503921

theorem number_of_large_posters_is_5
  (total_posters : ℕ)
  (small_posters_ratio : ℚ)
  (medium_posters_ratio : ℚ)
  (h_total : total_posters = 50)
  (h_small_ratio : small_posters_ratio = 2 / 5)
  (h_medium_ratio : medium_posters_ratio = 1 / 2) :
  (total_posters * (1 - small_posters_ratio - medium_posters_ratio)) = 5 :=
by sorry

end number_of_large_posters_is_5_l503_503921


namespace sum_ratio_l503_503366

variable (a1 q : ℝ) -- Defining the first term and the common ratio as real numbers
variable (S : ℕ → ℝ) -- Defining the sum of the first n terms of the sequence

-- Condition: S_n is the sum of the first n terms of a geometric sequence
def sum_geometric_sequence (a1 q: ℝ) (n: ℕ) : ℝ := 
  if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

-- Condition: 8a2 + a5 = 0
axiom geom_cond : 8 * (a1 * q) + a1 * q^4 = 0

-- Prove: S3/S2 = -3
theorem sum_ratio (a1 q: ℝ) (S : ℕ → ℝ) : 
  (S 3) / (S 2) = -3
  :=
sorry

end sum_ratio_l503_503366


namespace pencil_cost_l503_503140

theorem pencil_cost (P : ℝ) : 
  (∀ pen_cost total : ℝ, pen_cost = 3.50 → total = 291 → 38 * P + 56 * pen_cost = total → P = 2.50) :=
by
  intros pen_cost total h1 h2 h3
  sorry

end pencil_cost_l503_503140


namespace volume_of_cylinder_and_radius_of_sphere_of_melted_cylinder_l503_503924

/-
Given:
1. Base radius of the cylinder, r = 4.
2. Lateral area of the cylinder, L = (16/3)π.

We need to prove:
1. The volume of the cylinder is (32/3)π.
2. The radius of the sphere that is formed by melting and casting the cylinder is 2.
-/

def base_radius := 4
def lateral_area := (16 / 3) * Real.pi

theorem volume_of_cylinder_and_radius_of_sphere_of_melted_cylinder :
  ∃ (V_cylinder : ℝ) (R_sphere : ℝ), 
    V_cylinder = (32 / 3) * Real.pi ∧ 
    R_sphere = 2 :=
by
  sorry

end volume_of_cylinder_and_radius_of_sphere_of_melted_cylinder_l503_503924


namespace valid_4_digit_numbers_count_l503_503288

/-- The number of 4-digit numbers greater than 1000 that can be formed using the digits of 2013 is 18. -/
theorem valid_4_digit_numbers_count : 
  let digits := [2, 0, 1, 3] in
  let count := {n // n >= 1000 ∧ (n.digits 10).perm digits}.to_finset.card in
  count = 18 :=
by
  let digits := [2, 0, 1, 3]
  let count := {n // n >= 1000 ∧ (n.digits 10).perm digits}.to_finset.card
  sorry

end valid_4_digit_numbers_count_l503_503288


namespace angle_sum_gt_180_l503_503714

noncomputable def triangle (A B C : Type) := sorry -- Assuming the definition of a triangle

variables {A B C : Type}  -- Points A, B, C
variable (D : Type)       -- Point D

-- Conditions
def BD_add_AC_lt_BC (BD AC BC : ℝ) : Prop := BD + AC < BC 

-- Theorem and proof statement
theorem angle_sum_gt_180 (h : triangle A B C) (h1 : BD_add_AC_lt_BC BD AC BC) : 
  ∠ DAC + ∠ ADB > 180 := sorry

end angle_sum_gt_180_l503_503714


namespace area_of_intersection_is_correct_l503_503848

-- Define the radius and centers of the circles
def radius := 3
def center1 := (3 : ℝ, 0 : ℝ)
def center2 := (0 : ℝ, 3 : ℝ)

-- Define the equations of the circles
def circle1 (x y : ℝ) : Prop := (x - center1.1)^2 + y^2 = radius^2
def circle2 (x y : ℝ) : Prop := x^2 + (y - center2.2)^2 = radius^2

-- Define the intersection points
def intersection_points : set (ℝ × ℝ) := {p | circle1 p.1 p.2 ∧ circle2 p.1 p.2}

-- Theorem to prove the area of intersection of two circles
theorem area_of_intersection_is_correct : 
  let area_of_intersection := 9 * (Real.pi / 2 - 1) in
  area_of_intersection = 9 * (Real.pi / 2 - 1) :=
by
  sorry

end area_of_intersection_is_correct_l503_503848


namespace number_of_triangles_in_regular_decagon_l503_503662

noncomputable def number_of_triangles_in_decagon : ℕ :=
∑ i in (finset.range 10).powerset_len 3, 1

theorem number_of_triangles_in_regular_decagon :
  number_of_triangles_in_decagon = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l503_503662


namespace average_speed_interval_l503_503302

theorem average_speed_interval :
  let s (t : ℝ) := 3 + t^2 in
  (s 2.1 - s 2) / (2.1 - 2) = 4.1 :=
by
  sorry

end average_speed_interval_l503_503302


namespace sum_of_interior_angles_l503_503799

theorem sum_of_interior_angles (ext_angle : ℝ) (h : ext_angle = 20) : 
  let n := 360 / ext_angle in
  let int_sum := 180 * (n - 2) in
  int_sum = 2880 := 
by 
  sorry

end sum_of_interior_angles_l503_503799


namespace hyperbola_eccentricity_l503_503275

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (c : ℝ) (h3 : a^2 + b^2 = c^2) 
  (h4 : ∃ M : ℝ × ℝ, (M.fst^2 / a^2 - M.snd^2 / b^2 = 1) ∧ (M.snd^2 = 8 * M.fst)
    ∧ (|M.fst - 2| + |M.snd| = 5)) : 
  (c / a = 2) :=
by
  sorry

end hyperbola_eccentricity_l503_503275


namespace bianca_picture_books_shelves_l503_503533

theorem bianca_picture_books_shelves (total_shelves : ℕ) (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 8 →
  mystery_shelves = 5 →
  total_books = 72 →
  total_shelves = (total_books - (mystery_shelves * books_per_shelf)) / books_per_shelf →
  total_shelves = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end bianca_picture_books_shelves_l503_503533


namespace median_of_fifteen_is_eight_l503_503482

def median_of_first_fifteen_positive_integers : ℝ :=
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median_pos := (list.length lst + 1) / 2  
  lst.get (median_pos - 1)

theorem median_of_fifteen_is_eight : median_of_first_fifteen_positive_integers = 8.0 := 
  by 
    -- Proof omitted    
    sorry

end median_of_fifteen_is_eight_l503_503482


namespace circle_equation_of_tangent_circle_l503_503244

theorem circle_equation_of_tangent_circle
  (h : ∀ x y: ℝ, x^2/4 - y^2 = 1 → (x = 2 ∨ x = -2) → y = 0)
  (asymptote : ∀ x y : ℝ, (y = (1/2)*x ∨ y = -(1/2)*x) → (x - 2)^2 + y^2 = (4/5))
  : ∃ k : ℝ, (∀ x y : ℝ, (x - 2)^2 + y^2 = k) → k = 4/5 := by
  sorry

end circle_equation_of_tangent_circle_l503_503244


namespace find_fifth_month_sales_correct_l503_503898

theorem find_fifth_month_sales_correct:
  let S1 := 5266
  let S2 := 5768
  let S3 := 5922
  let S4 := 5678
  let S6 := 4937
  let average := 5600
  let num_months := 6
  let total_sales := average * num_months in
  let fifth_month_sales := total_sales - (S1 + S2 + S3 + S4 + S6) in
  fifth_month_sales = 6029 :=
by {
  sorry
}

end find_fifth_month_sales_correct_l503_503898


namespace magic_king_total_episodes_l503_503441

def total_episodes_first_three_seasons := 3 * 20
def total_episodes_seasons_4_to_8 := (8 - 4 + 1) * 25
def total_episodes_seasons_9_to_11 := (11 - 9 + 1) * 30
def total_episodes_last_three_seasons := 3 * 15
def holiday_specials := 5

def total_episodes :=
  total_episodes_first_three_seasons +
  total_episodes_seasons_4_to_8 +
  total_episodes_seasons_9_to_11 +
  total_episodes_last_three_seasons +
  holiday_specials

theorem magic_king_total_episodes : total_episodes = 325 :=
by
  calc
    total_episodes
        = 3 * 20 + (8 - 4 + 1) * 25 + (11 - 9 + 1) * 30 + 3 * 15 + 5 : by sorry
    ... = 60 + 125 + 90 + 45 + 5 : by sorry
    ... = 325 : by sorry

end magic_king_total_episodes_l503_503441


namespace last_digit_of_large_power_l503_503165

def last_digit (n : ℕ) : ℕ := n % 10

lemma last_digit_power_cycle_five (n : ℕ) : last_digit (5^n) = 5 :=
by simp [last_digit, pow_mod, Nat.mod_eq_of_lt 5 (by norm_num : 5 < 10)]

lemma last_digit_power_cycle_six (n : ℕ) : last_digit (6^n) = 6 :=
by simp [last_digit, pow_mod, Nat.mod_eq_of_lt 6 (by norm_num : 6 < 10)]

lemma last_digit_power_cycle_seven (n : ℕ) : last_digit (7^1) = 7 ∧ 
                                             last_digit (7^2) = 9 ∧
                                             last_digit (7^3) = 3 ∧
                                             last_digit (7^4) = 1 :=
by norm_num [last_digit]

theorem last_digit_of_large_power : 
  last_digit (5^555 + 6^666 + 7^777) = 8 :=
sorry

end last_digit_of_large_power_l503_503165


namespace eccentricity_of_ellipse_l503_503614

theorem eccentricity_of_ellipse (a c : ℝ) (h : 4 * a = 7 * 2 * (a - c)) : 
    c / a = 5 / 7 :=
by {
  sorry
}

end eccentricity_of_ellipse_l503_503614


namespace median_of_first_fifteen_positive_integers_l503_503476

theorem median_of_first_fifteen_positive_integers : ∃ m : ℝ, m = 7.5 :=
by
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median := (seq[6] + seq[7]) / 2
  use median
  sorry

end median_of_first_fifteen_positive_integers_l503_503476


namespace train_length_approx_l503_503928

-- Define the speed of the train in km/h
def speed_kmh : ℝ := 32

-- Define the conversion factor from km/h to m/s
def conversion_factor : ℝ := 5 / 18

-- Define the speed of the train in m/s
def speed_ms : ℝ := speed_kmh * conversion_factor

-- Define the time taken for the train to cross the man in seconds
def time_seconds : ℝ := 18

-- Define the length of the train in meters
def train_length : ℝ := speed_ms * time_seconds

-- The statement we want to prove
theorem train_length_approx : train_length ≈ 160.02 :=
by
  sorry

end train_length_approx_l503_503928


namespace number_of_rational_numbers_is_two_l503_503150

/-- The set of numbers to analyze -/
def numbers : set ℝ := {-3.5, 22 / 7, (0.161161116 : ℝ), real.pi / 2}

/-- Definition of rational numbers in the given set -/
def rational_numbers : set ℝ := {x ∈ numbers | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

/-- The main statement: To prove that there are exactly two rational numbers in the given set. -/
theorem number_of_rational_numbers_is_two : rational_numbers.card = 2 := sorry

end number_of_rational_numbers_is_two_l503_503150


namespace count_no_repeat_count_with_repeat_l503_503075

-- Define the set of digits and properties we are working with
def digits := {0, 2, 3, 5, 7}

noncomputable def count_no_repeat_div_by_5 : Nat := 42
noncomputable def count_with_repeat_div_by_5 : Nat := 200

-- First problem: Digits do not repeat
theorem count_no_repeat :
  (∃(a b c d : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ (d = 0 ∨ d = 5) ∧ (a * 1000 + b * 100 + c * 10 + d) % 5 = 0) →
  count_no_repeat_div_by_5 = 42 :=
by sorry

-- Second problem: Digits can repeat
theorem count_with_repeat :
  (∃(a b c d : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ (d = 0 ∨ d = 5) ∧ (a * 1000 + b * 100 + c * 10 + d) % 5 = 0) →
  count_with_repeat_div_by_5 = 200 :=
by sorry

end count_no_repeat_count_with_repeat_l503_503075


namespace tan_expression_val_l503_503674

theorem tan_expression_val (A B : ℝ) (hA : A = 30) (hB : B = 15) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
by
  sorry

end tan_expression_val_l503_503674


namespace correct_statements_in_space_l503_503529

noncomputable theory
open_locale classical

variables {L : Type} [linear_ordered_field L]

structure line (L : Type) :=
  (point1 point2 : L)

def parallel (l1 l2 : line L) : Prop :=
  ∃ (v : L), ∀ (p1 p2 : L), l1.point1 ≠ l2.point1 → (p2 - p1) / (l2.point2 - l2.point1) = v

def perpendicular (l1 l2 : line L) : Prop :=
  ∃ (v : L), v ≠ 0 ∧ v * v = -1

def intersects (l1 l2 : line L) : Prop := 
  ∃ (p : L), l1.point1 = p ∨ l1.point2 = p

theorem correct_statements_in_space :
  ∃ (l1 l2 l3 l4 l5 : line L),
    (parallel l1 l2) ∧
    (¬parallel l3 l4 → perpendicular l3 l5 → false) ∧
    (perpendicular l1 l2 ∧ parallel l1 l3 → perpendicular l2 l3) ∧
    (parallel l1 l2 ∧ intersects l3 l1 → ¬intersects l3 l2) → 
    true :=
by { sorry }

end correct_statements_in_space_l503_503529


namespace nick_pennsylvania_state_quarters_l503_503391

theorem nick_pennsylvania_state_quarters:
  let quarters : ℕ := 35
  let state_fraction : ℚ := 2 / 5
  let penn_fraction : ℚ := 0.5
  let state_quarters := quarters * state_fraction
  let penn_quarters := state_quarters * penn_fraction
  penn_quarters = 7 := by
  -- definitions
  let quarters := 35
  let state_fraction := 2 / 5
  let penn_fraction := 0.5
  let state_quarters := quarters * state_fraction
  let penn_quarters := state_quarters * penn_fraction
  -- definitions and simplification show that total Pennsylvania state quarters should be 7
  show state_quarters * penn_fraction = 7
  sorry

end nick_pennsylvania_state_quarters_l503_503391


namespace count_multiples_of_73_in_array_l503_503929

theorem count_multiples_of_73_in_array :
  let a (n k : ℕ) := 2^(n-1) * (n + 2*k - 2),
      total_entries := 22 in
  (∑ n in finset.range 44, if n % 2 = 1 then 1 else 0) = total_entries :=
by
  let a := λ (n k : ℕ), 2^(n-1) * (n + 2*k - 2)
  let total_entries := 22
  sorry

end count_multiples_of_73_in_array_l503_503929


namespace mean_mode_equal_x_l503_503042

variable {x : ℝ}
variable {data : List ℝ}

def mean (l : List ℝ) : ℝ := l.sum / l.length

def mode (l : List ℝ) : ℝ :=
  l.groupBy (λ x y => x = y) 
   |>.maxBy (λ g => g.length) 
   |>.head! 
   |>.toReal

theorem mean_mode_equal_x (h_mean : mean [x, 70, x, 55, x, 110, 180, 75] = x)
                         (h_mode : mode [x, 70, x, 55, x, 110, 180, 75] = x) :
  x = 98 :=
  sorry

end mean_mode_equal_x_l503_503042


namespace roots_of_quadratic_eq_l503_503813

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_eq_l503_503813


namespace median_of_first_fifteen_positive_integers_l503_503475

theorem median_of_first_fifteen_positive_integers : ∃ m : ℝ, m = 7.5 :=
by
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median := (seq[6] + seq[7]) / 2
  use median
  sorry

end median_of_first_fifteen_positive_integers_l503_503475


namespace sum_of_interior_angles_of_regular_polygon_l503_503791

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : n = 360 / 20) :
  (∑ i in finset.range n, 180 - 360 / n) = 2880 :=
by
  sorry

end sum_of_interior_angles_of_regular_polygon_l503_503791


namespace necklace_last_bead_color_l503_503342

theorem necklace_last_bead_color :
  (∃ pattern : list string, pattern = ["Red", "Orange", "Yellow", "Yellow", "Green", "Blue"] ∧ 
  (∃ n : ℕ, n = 81 ∧ list.nth (list.cycle pattern) (n - 1) = some "Yellow")) :=
by 
  sorry

end necklace_last_bead_color_l503_503342


namespace interior_angles_sum_l503_503793

theorem interior_angles_sum (h : ∀ (n : ℕ), n = 360 / 20) : 
  180 * (h 18 - 2) = 2880 :=
by
  sorry

end interior_angles_sum_l503_503793


namespace length_of_side_of_largest_square_l503_503878

-- Definitions based on the conditions
def string_length : ℕ := 24

-- The main theorem corresponding to the problem statement.
theorem length_of_side_of_largest_square (h: string_length = 24) : 24 / 4 = 6 :=
by
  sorry

end length_of_side_of_largest_square_l503_503878


namespace sequence_an_sequence_Tn_l503_503617

theorem sequence_an (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, 2 * S n = a n ^ 2 + a n):
  ∀ n, a n = n :=
sorry

theorem sequence_Tn (b : ℕ → ℕ) (T : ℕ → ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, 2 * S n = a n ^ 2 + a n) (h2 : ∀ n, a n = n) (h3 : ∀ n, b n = 2^n * a n):
  ∀ n, T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end sequence_an_sequence_Tn_l503_503617


namespace geom_sequence_sum_l503_503317

theorem geom_sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (r : ℤ) 
    (h1 : ∀ n : ℕ, n ≥ 1 → S n = 3^n + r) 
    (h2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)) 
    (h3 : a 1 = S 1) :
  r = -1 := 
sorry

end geom_sequence_sum_l503_503317


namespace symmetry_axis_of_f_l503_503419

-- Define the function f(x), as given in the problem
def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3)

-- Statement to prove that the axis of symmetry for f(x) is at x = 5π/3
theorem symmetry_axis_of_f : ∃ (x : ℝ), x = 5 * Real.pi / 3 ∧ ∀ y : ℝ, f (2 * x - y) = f y :=
sorry

end symmetry_axis_of_f_l503_503419


namespace units_digits_no_match_l503_503124

theorem units_digits_no_match : ∀ x : ℕ, 1 ≤ x ∧ x ≤ 100 → (x % 10 ≠ (101 - x) % 10) :=
by
  intro x hx
  sorry

end units_digits_no_match_l503_503124


namespace numTrianglesFromDecagon_is_120_l503_503664

noncomputable def numTrianglesFromDecagon : ℕ := 
  nat.choose 10 3

theorem numTrianglesFromDecagon_is_120 : numTrianglesFromDecagon = 120 := 
  by
    -- Form the combination
    have : numTrianglesFromDecagon = nat.choose 10 3 := rfl

    -- Calculate
    have calc₁ : nat.choose 10 3 = 10 * 9 * 8 / (3 * 2 * 1) := by 
      exact nat.choose_eq_div
      simp

    -- Simplify the calculation to 120
    have : 10 * 9 * 8 / (3 * 2 * 1) = 120 := by 
      norm_num 

    exact eq.trans this.symm calc₁.symm ⟩

end numTrianglesFromDecagon_is_120_l503_503664


namespace find_k_values_l503_503573

-- Definitions of parameters in given conditions
def A := Matrix.ofVecs 2 2 ([[3, 4], [6, -1]] : List (List ℝ))
def b := EuclideanSpace.vec2 2 (-2)
def k_values := {-9, 3}

theorem find_k_values (k : ℝ) (v : EuclideanSpace ℝ (Fin 2)) :
  (A * v = k • v + b) → v ≠ 0 → k ∈ k_values :=
sorry

end find_k_values_l503_503573


namespace max_of_2x_plus_y_l503_503243

theorem max_of_2x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y / 2 + 1 / x + 8 / y = 10) : 
  2 * x + y ≤ 18 :=
sorry

end max_of_2x_plus_y_l503_503243


namespace max_value_condition_l503_503604

variable {m n : ℝ}

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  m * n > 0 ∧ m + n = -1

-- Statement of the proof problem
theorem max_value_condition (h : conditions m n) : (1/m + 1/n) ≤ 4 :=
sorry

end max_value_condition_l503_503604


namespace sum_of_interior_angles_of_regular_polygon_l503_503788

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : n = 360 / 20) :
  (∑ i in finset.range n, 180 - 360 / n) = 2880 :=
by
  sorry

end sum_of_interior_angles_of_regular_polygon_l503_503788


namespace carwash_problem_l503_503358

theorem carwash_problem
(h1 : ∀ (n : ℕ), 5 * n + 6 * 5 + 7 * 5 = 100)
(h2 : 5 * 5 = 25)
(h3 : 7 * 5 = 35)
(h4 : 100 - 35 - 30 = 35):
(n = 7) :=
by
  have h : 5 * n = 35 := by sorry
  exact eq_of_mul_eq_mul_left (by sorry) h

end carwash_problem_l503_503358


namespace find_mn_sum_of_parallel_segment_l503_503070

theorem find_mn_sum_of_parallel_segment (PQ PR QR : ℕ) 
  (PQ_eq : PQ = 26) (PR_eq : PR = 28) (QR_eq : QR = 30) 
  (S T : Point) (HS : S ∈ line_segment P Q) (HT : T ∈ line_segment P R) 
  (H_parallel : parallel (line_segment S T) (line_segment Q R))
  (H_incenter : incenter (triangle P Q R) ∈ line_segment S T) :
  let ST := (135 : ℚ) / 7 in
  ∃ m n : ℕ, nat.coprime m n ∧ ST = m / n ∧ m + n = 142 :=
by sorry

end find_mn_sum_of_parallel_segment_l503_503070


namespace geometric_representation_of_S_l503_503440

def S (a : ℝ) : set ℂ := { w | ∃ z : ℂ, arg z = a ∧ w = conj (z ^ 2) }

theorem geometric_representation_of_S (a : ℝ) : S a = { w : ℂ | ∃ r : ℝ, r > 0 ∧ w = r * complex.exp (complex.I * (-2 * a)) } :=
by sorry

end geometric_representation_of_S_l503_503440


namespace sum_alternating_series_l503_503113

theorem sum_alternating_series : ∑ k in Finset.range 2010, (-1) ^ (k + 1) = 0 :=
sorry

end sum_alternating_series_l503_503113


namespace find_x_squared_plus_y_squared_l503_503233

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = -8) : x^2 + y^2 = 33 := 
by 
  sorry

end find_x_squared_plus_y_squared_l503_503233


namespace distance_from_point_to_line_l503_503690

theorem distance_from_point_to_line :
  let x1 := -3
  let y1 := 8
  let A := 5
  let B := -1
  let C := 10
  d = (abs (A * x1 + B * y1 + C) / real.sqrt (A^2 + B^2)) :=
  d = 13 * real.sqrt 26 / 26 :=
begin
  -- Proof goes here
  sorry
end

end distance_from_point_to_line_l503_503690


namespace original_weight_of_apple_box_l503_503450

theorem original_weight_of_apple_box:
  ∀ (x : ℕ), (3 * x - 12 = x) → x = 6 :=
by
  intros x h
  sorry

end original_weight_of_apple_box_l503_503450


namespace modulus_of_z_l503_503220

def z (z : ℂ) : Prop := (1 + complex.i) / (1 - complex.i) * z = 3 + 4 * complex.i

theorem modulus_of_z {z : ℂ} (h : z (z)) : abs z = 5 := by
  sorry

end modulus_of_z_l503_503220


namespace quadratic_rewriting_l503_503052

theorem quadratic_rewriting :
  ∃ d e : ℤ, (∀ x : ℝ, x^2 - 16 * x + 15 = (x + d)^2 + e) ∧ d + e = -57 :=
begin
  sorry
end

end quadratic_rewriting_l503_503052


namespace num_triangles_from_decagon_l503_503655

theorem num_triangles_from_decagon (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ c ≠ a → ¬(collinear a b c)) :
  (nat.choose 10 3) = 120 :=
by
  sorry

end num_triangles_from_decagon_l503_503655


namespace first_digit_of_sum_l503_503003

theorem first_digit_of_sum (n : ℕ) (a : ℕ) (hs : 9 * a = n)
  (h_sum : n = 43040102 - (10^7 * d - 10^7 * 4)) : 
  (10^7 * d - 10^7 * 4) / 10^7 = 8 :=
by
  sorry

end first_digit_of_sum_l503_503003


namespace vector_parallel_dot_product_l503_503287

theorem vector_parallel_dot_product (x : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (x, 1))
  (h2 : b = (4, 2))
  (h3 : x / 4 = 1 / 2) : 
  (a.1 * (b.1 - a.1) + a.2 * (b.2 - a.2)) = 5 := 
by 
  sorry

end vector_parallel_dot_product_l503_503287


namespace factor_difference_of_squares_l503_503968

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l503_503968


namespace real_numbers_correspond_to_number_line_l503_503781

noncomputable def number_line := ℝ

def real_numbers := ℝ

theorem real_numbers_correspond_to_number_line :
  ∀ (p : ℝ), ∃ (r : real_numbers), r = p ∧ ∀ (r : real_numbers), ∃ (p : ℝ), p = r :=
by
  sorry

end real_numbers_correspond_to_number_line_l503_503781


namespace value_of_a_squared_b_plus_ab_squared_eq_4_l503_503597

variable (a b : ℝ)
variable (h_a : a = 2 + Real.sqrt 3)
variable (h_b : b = 2 - Real.sqrt 3)

theorem value_of_a_squared_b_plus_ab_squared_eq_4 :
  a^2 * b + a * b^2 = 4 := by
  sorry

end value_of_a_squared_b_plus_ab_squared_eq_4_l503_503597


namespace quadratic_inequality_solution_l503_503640

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x ^ 2 + 4 * x + 1 > 0) ↔ (a > 4) :=
sorry

end quadratic_inequality_solution_l503_503640


namespace median_of_first_fifteen_integers_l503_503486

theorem median_of_first_fifteen_integers : 
  let L := (list.range 15).map (λ n, n + 1)
  in list.median L = 8.0 :=
by 
  sorry

end median_of_first_fifteen_integers_l503_503486


namespace number_of_ways_to_choose_team_l503_503756

theorem number_of_ways_to_choose_team 
  (players : Finset ℕ) 
  (quadruplets : Finset ℕ)
  (H_total : players.card = 15)
  (H_quadruplets : quadruplets.card = 4)
  (quadruplet_inclusion : quadruplets ⊆ players)
  (team_size : nat) 
  (H_team_size : team_size = 7) : 
  ∃ ways : ℕ, ways = (players.filter (λ x, x ∉ quadruplets)).card.choose (team_size - quadruplets.card) := 
begin
  -- The number of remaining players is 11
  have H_remaining : (players.filter (λ x, x ∉ quadruplets)).card = 11,
  {
    rw Finset.card_filter,
    have H_disjoint : quadruplets ∩ (players.filter (λ x, x ∉ quadruplets)) = ∅,
    {
      apply Finset.disjoint_filter,
    },
    rw Finset.filter_false_of_mem H_disjoint,
    rw Finset.card_disjoint_union,
    rw H_quadruplets,
    rw H_total,
    exact rfl,
  },
  -- The number of ways to choose the remaining players is 165
  use nat.choose 11 3,
  refl,
end

#print number_of_ways_to_choose_team

end number_of_ways_to_choose_team_l503_503756


namespace smallest_even_number_of_seven_l503_503770

-- Conditions: The sum of seven consecutive even numbers is 700.
-- We need to prove that the smallest of these numbers is 94.

theorem smallest_even_number_of_seven (n : ℕ) (hn : 7 * n = 700) :
  ∃ (a b c d e f g : ℕ), 
  (2 * a + 4 * b + 6 * c + 8 * d + 10 * e + 12 * f + 14 * g = 700) ∧ 
  (a = b - 1) ∧ (b = c - 1) ∧ (c = d - 1) ∧ (d = e - 1) ∧ (e = f - 1) ∧ 
  (f = g - 1) ∧ (g = 100) ∧ (a = 94) :=
by
  -- This is the theorem statement. 
  sorry

end smallest_even_number_of_seven_l503_503770


namespace smallest_number_remainder_l503_503860

open Nat

theorem smallest_number_remainder
  (b : ℕ)
  (h1 : b % 4 = 2)
  (h2 : b % 3 = 2)
  (h3 : b % 5 = 3) :
  b = 38 :=
sorry

end smallest_number_remainder_l503_503860


namespace horner_method_v2_l503_503073

noncomputable def f (x : ℤ) : ℤ := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldr (λ a b, a + x * b) 0
  
theorem horner_method_v2 (x : ℤ) (h : x = -4) :
  let coeffs := [3, 5, 6, 79, -8, 35, 12]
  let v2 := ((-7 * (coe (-4))) + 6)
  f x = horner_eval coeffs x ∧ v2 = 34 :=
by
  sorry

end horner_method_v2_l503_503073


namespace graph_shift_l503_503068

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem graph_shift :
  ∀ (x : ℝ), f x = g (x - Real.pi / 4) :=
by
  intros
  sorry

end graph_shift_l503_503068


namespace max_value_x2_y2_l503_503296

variable (x y : ℝ)

theorem max_value_x2_y2 :
  (5 * x^2 - 10 * x + 4 * y^2 = 0) → (0 ≤ x ∧ x ≤ 2) → x^2 + y^2 ≤ 4 :=
begin
  sorry
end

end max_value_x2_y2_l503_503296


namespace average_of_seven_consecutive_integers_b_l503_503410

-- Define the initial setup: seven consecutive integers starting from a
def a := ℕ
def b := a + 3

-- Prove the average of seven consecutive integers starting from b equals a + 6
theorem average_of_seven_consecutive_integers_b (a b : ℕ) (h : b = a + 3) : 
  (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7 = a + 6 :=
by
  sorry


end average_of_seven_consecutive_integers_b_l503_503410


namespace triangle_inequality_equilateral_equality_l503_503382

noncomputable def HeronArea (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2
in real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_inequality (a b c : ℝ) (A : ℝ) (hA : A = HeronArea a b c) :
  a^2 + b^2 + c^2 >= 4 * real.sqrt 3 * A :=
sorry

theorem equilateral_equality (a : ℝ) : 
  let A := (a^2 * real.sqrt 3) / 4
  in a^2 + a^2 + a^2 = 4 * real.sqrt 3 * A :=
by
  let A := (a^2 * real.sqrt 3) / 4
  have : a^2 + a^2 + a^2 = 3 * a^2 := by ring
  rw [this, A]
  simp [mul_assoc]
  norm_num
  sorry

end triangle_inequality_equilateral_equality_l503_503382


namespace selection_methods_12_l503_503141

def num_selection_methods (teachers : Finset ℕ) (phases : ℕ) : ℕ :=
  teachers.choose phases.card

theorem selection_methods_12 :
  let teachers := {0, 1, 2, 3}, -- Representing Teacher A, B, C, D
      forbidden := {0, 1}, -- Representing Teacher A and B cannot attend the first phase
      phases := 3,
      valid_first_phase := teachers \ forbidden in
  num_selection_methods valid_first_phase 1 * (teachers \ num_selection_methods valid_first_phase 1).choose 2 = 12 :=
sorry

#print axioms selection_methods_12  -- Just a sanity check.

end selection_methods_12_l503_503141


namespace lines_of_code_thursday_l503_503903

theorem lines_of_code_thursday (k : ℕ) (l_wed : ℕ) (c_wed : ℕ) (c_thu : ℕ) :
  (l_wed = k * c_wed) → (c_wed = 3) → (l_wed = 150) → (c_thu = 5) → (k = 50) → (k * c_thu = 250) :=
by
  intros h1 h2 h3 h4 h5
  rw [←h5, ←h4]
  sorry

end lines_of_code_thursday_l503_503903


namespace unique_integer_3_5_l503_503571

theorem unique_integer_3_5 (n : ℕ) (h : n ≥ 1) : 3^{n-1} + 5^{n-1} ∣ 3^n + 5^n → n = 1 := by
  sorry

end unique_integer_3_5_l503_503571


namespace spacy_subsets_count_l503_503959

-- Define the recurrence relation for c_n
def c : ℕ → ℕ
| 0 => 1  -- This is not needed, but required for completeness
| 1 => 2
| 2 => 3
| 3 => 4
| n => c (n - 1) + c (n - 3)

-- Define the theorem to prove c 15 = 406
theorem spacy_subsets_count : c 15 = 406 :=
by
  -- the proof goes here
  sorry

end spacy_subsets_count_l503_503959


namespace number_of_possible_sums_l503_503886

open Finset
open BigOperators

theorem number_of_possible_sums (A : Finset ℕ) (hA : A ⊆ range 1 51) (h_card : A.card = 40) :
  ∃ n : ℕ, n = 401 :=
by
  sorry

end number_of_possible_sums_l503_503886


namespace possible_values_of_d_l503_503405

theorem possible_values_of_d (u v : ℝ) (c d : ℝ) (hu : Polynomial.roots (Polynomial.C d + Polynomial.C c * X + X^3) = {u, v, -u-v})
(hq1 : Polynomial.has_root (Polynomial.C (d - 270) + Polynomial.C c * X + X^3) (u + 3))
(hq2 : Polynomial.has_root (Polynomial.C (d - 270) + Polynomial.C c * X + X^3) (v - 2)) :
d = -6 ∨ d = -120 :=
sorry

end possible_values_of_d_l503_503405


namespace film_finishes_earlier_on_first_channel_l503_503895

-- Definitions based on conditions
def DurationSegmentFirstChannel (n : ℕ) : ℝ := n * 22
def DurationSegmentSecondChannel (k : ℕ) : ℝ := k * 11

-- The time when first channel starts the n-th segment
def StartNthSegmentFirstChannel (n : ℕ) : ℝ := (n - 1) * 22

-- The number of segments second channel shows by the time first channel starts the n-th segment
def SegmentsShownSecondChannel (n : ℕ) : ℕ := ((n - 1) * 22) / 11

-- If first channel finishes earlier than second channel
theorem film_finishes_earlier_on_first_channel (n : ℕ) (hn : 1 < n) :
  DurationSegmentFirstChannel n < DurationSegmentSecondChannel (SegmentsShownSecondChannel n + 1) :=
sorry

end film_finishes_earlier_on_first_channel_l503_503895


namespace log_base_1_over_4_increasing_l503_503801

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- The statement we want to prove
theorem log_base_1_over_4_increasing :
  ∀ x : ℝ, log (1/4) (f x) is_monotonic_increasing_on I := sorry

end log_base_1_over_4_increasing_l503_503801


namespace largest_expression_is_E_l503_503496

def expression_A : ℕ := 3 + 1 + 2 + 8
def expression_B : ℕ := 3 * 1 + 2 + 8
def expression_C : ℕ := 3 + 1 * 2 + 8
def expression_D : ℕ := 3 + 1 + 2 * 8
def expression_E : ℕ := 3 * 1 * 2 * 8

theorem largest_expression_is_E :
  expression_E = 48 ∧
  expression_E > expression_A ∧
  expression_E > expression_B ∧
  expression_E > expression_C ∧
  expression_E > expression_D :=
by
  have hA : expression_A = 14 := by sorry
  have hB : expression_B = 13 := by sorry
  have hC : expression_C = 13 := by sorry
  have hD : expression_D = 20 := by sorry
  have hE : expression_E = 48 := by sorry
  exact ⟨hE, by rw [hE, hA]; exact Nat.gt_succ_self, by rw [hE, hB]; exact Nat.gt_succ_self.succ, by rw [hE, hC]; exact Nat.gt_succ_self.succ, by rw [hE, hD]; exact Nat.gt_succ_self.succ_succ⟩

end largest_expression_is_E_l503_503496


namespace cost_of_pencil_and_pen_l503_503033

variable (p q : ℝ)

axiom condition1 : 4 * p + 3 * q = 4.20
axiom condition2 : 3 * p + 4 * q = 4.55

theorem cost_of_pencil_and_pen : p + q = 1.25 :=
by
  sorry

end cost_of_pencil_and_pen_l503_503033


namespace second_discount_percentage_l503_503437

theorem second_discount_percentage
    (original_price : ℝ)
    (first_discount : ℝ)
    (final_sale_price : ℝ)
    (second_discount : ℝ)
    (h1 : original_price = 390)
    (h2 : first_discount = 14)
    (h3 : final_sale_price = 285.09) :
    second_discount = 15 :=
by
  -- Since we are not providing the full proof, we assume the steps to be correct
  sorry

end second_discount_percentage_l503_503437


namespace inequality_property_l503_503535

theorem inequality_property (a b : ℝ) (h : a > b) : -5 * a < -5 * b := sorry

end inequality_property_l503_503535


namespace roots_of_quadratic_eq_l503_503812

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_eq_l503_503812


namespace incorrect_rounding_statement_l503_503099

def rounded_to_nearest (n : ℝ) (accuracy : ℝ) : Prop :=
  ∃ (k : ℤ), abs (n - k * accuracy) < accuracy / 2

theorem incorrect_rounding_statement :
  ¬ rounded_to_nearest 23.9 10 :=
sorry

end incorrect_rounding_statement_l503_503099


namespace lake_view_population_l503_503825

-- Define the populations of the cities
def population_of_Seattle : ℕ := 20000 -- Derived from the solution
def population_of_Boise : ℕ := (3 / 5) * population_of_Seattle
def population_of_Lake_View : ℕ := population_of_Seattle + 4000
def total_population : ℕ := population_of_Seattle + population_of_Boise + population_of_Lake_View

-- Statement to prove
theorem lake_view_population :
  total_population = 56000 →
  population_of_Lake_View = 24000 :=
sorry

end lake_view_population_l503_503825


namespace point_on_x_axis_l503_503611

theorem point_on_x_axis (a : ℝ) (h : a + 2 = 0) : (a - 1, a + 2) = (-3, 0) :=
by
  sorry

end point_on_x_axis_l503_503611


namespace solve_problem_l503_503575

theorem solve_problem (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  7^m - 3 * 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) := sorry

end solve_problem_l503_503575


namespace remainder_of_M_div_500_l503_503367

-- Define the sequence T in Lean
def T : ℕ → ℕ := λ n, 
  let binom := λ n k, nat.choose n k in
  -- Find the nth number in the sequence of numbers with 7 ones in binary
  if h : n > 0 then
    let idx := n - 1 in
    let sum_binom := (λ m sum, if sum ≤ idx then (m, sum + binom m 7) else (m, sum)) in
    let m := (nat.lt_wf {n // n > 0}).fix (λ m sum_binom, let ⟨m, sum⟩ := sum_binom m 0 in m)
    in 
    (nat.lt_wf m).fix (λ k remain, if remain ≤ idx then 0 else (2^k) + remain)
  else
    0

-- Define M as the 500th element in the sequence T
def M : ℕ := T 500

-- Define the final property to prove
theorem remainder_of_M_div_500 : M % 500 = 64 := 
by sorry

end remainder_of_M_div_500_l503_503367


namespace smallest_number_l503_503867

theorem smallest_number (b : ℕ) :
  (b % 3 = 2) ∧ (b % 4 = 2) ∧ (b % 5 = 3) → b = 38 :=
by
  sorry

end smallest_number_l503_503867


namespace probability_of_5_pieces_of_candy_l503_503891

-- Define the conditions
def total_eggs : ℕ := 100 -- Assume total number of eggs is 100 for simplicity
def blue_eggs : ℕ := 4 * total_eggs / 5
def purple_eggs : ℕ := total_eggs / 5
def blue_eggs_with_5_candies : ℕ := blue_eggs / 4
def purple_eggs_with_5_candies : ℕ := purple_eggs / 2
def total_eggs_with_5_candies : ℕ := blue_eggs_with_5_candies + purple_eggs_with_5_candies

-- The proof problem
theorem probability_of_5_pieces_of_candy : (total_eggs_with_5_candies : ℚ) / (total_eggs : ℚ) = 3 / 10 := 
by
  sorry

end probability_of_5_pieces_of_candy_l503_503891


namespace conditional_probability_PB_given_A_l503_503125

def num_products := 4
def num_first_class := 3
def num_second_class := 1

def event_A : probability_space.event :=
  -- Event A: A first-class product is taken on the first draw.
  λ s, s.first_draw = 'first_class'

def event_B : probability_space.event :=
  -- Event B: A first-class product is taken on the second draw.
  λ s, s.second_draw = 'first_class'

def P (e : probability_space.event) : ℚ := sorry -- Define the probability function

axiom draws_without_replacement : -- Defining the problem condition
  ∀ s, ('first_class' \in s.remaining_products_after_first_draw) ↔ (s = first_draw = 'first_class')

theorem conditional_probability_PB_given_A :
  P(event_B | event_A) = 2 / 3 :=
by
  sorry

end conditional_probability_PB_given_A_l503_503125


namespace flip_theorem_l503_503824

open Nat

-- Define the problem with 2015 coins
def coins := Fin 2015

-- Define the flipping rule
def flip (coins : Fin 2015) (i : Nat) : Nat := sorry

-- Main theorem statement
theorem flip_theorem (coins : Fin 2015) :
  ∃ final_state : Bool, (∀ i < 2015, flip coins i = final_state) ∧ (final_state = tt ∨ final_state = ff) :=
sorry

end flip_theorem_l503_503824


namespace median_first_fifteen_integers_l503_503470

theorem median_first_fifteen_integers :
  let l := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] in
  let seventh := l.nth 6 in
  let eighth := l.nth 7 in
  (seventh.is_some ∧ eighth.is_some) →
  (seventh.get_or_else 0 + eighth.get_or_else 0) / 2 = 7.5 :=
by
  sorry

end median_first_fifteen_integers_l503_503470


namespace products_of_three_distinct_elements_of_T_l503_503378

def is_divisor (n d : ℕ) : Prop := d ∣ n

def elements_of_T (t : ℕ → Prop) : Prop :=
  ∃ a b : ℕ, 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 9 ∧ t = 2^a * 5^b

def three_distinct_product (a b c : ℕ) : Prop :=
  elements_of_T a ∧ elements_of_T b ∧ elements_of_T c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

def exactly_three_distinct_products_count : ℕ := 350

theorem products_of_three_distinct_elements_of_T :
  (T : set ℕ) := { d | is_divisor 500000 d } →
  is_divisor 500000 500000 →
  exactly_three_distinct_products_count = 350 :=
by
  sorry

end products_of_three_distinct_elements_of_T_l503_503378


namespace round_robin_tournament_matches_l503_503908

theorem round_robin_tournament_matches (n : ℕ) (h : n = 10) :
  let matches := n * (n - 1) / 2
  matches = 45 :=
by
  intros
  rw [h]
  dsimp
  norm_num
  sorry

end round_robin_tournament_matches_l503_503908


namespace probability_juliet_in_capulet_l503_503892

-- Definitions for the conditions

variables (P : ℝ) (hP_pos : P > 0)

-- Define populations
def MontaguePopulation := (3 / 4) * P
def CapuletPopulation := (1 / 4) * P

-- Support for Romeo and Juliet in each province
def RomeoSupportersMontague := 0.8 * MontaguePopulation
def JulietSupportersMontague := 0.2 * MontaguePopulation
def JulietSupportersCapulet := 0.7 * CapuletPopulation

-- Total number of Juliet supporters in Venezia
def TotalJulietSupporters := JulietSupportersMontague + JulietSupportersCapulet

-- The probability we need to prove
theorem probability_juliet_in_capulet :
  let juliet_supporters_capulet := JulietSupportersCapulet in
  let total_juliet_supporters := TotalJulietSupporters in
  (juliet_supporters_capulet / total_juliet_supporters) * 100 ≈ 54 :=
sorry

end probability_juliet_in_capulet_l503_503892


namespace cookies_batches_needed_l503_503530

noncomputable def number_of_recipes (total_students : ℕ) (attendance_drop : ℝ) (cookies_per_batch : ℕ) : ℕ :=
  let remaining_students := (total_students : ℝ) * (1 - attendance_drop)
  let total_cookies := remaining_students * 2
  let recipes_needed := total_cookies / cookies_per_batch
  (Nat.ceil recipes_needed : ℕ)

theorem cookies_batches_needed :
  number_of_recipes 150 0.40 18 = 10 :=
by
  sorry

end cookies_batches_needed_l503_503530


namespace ratio_sum_eq_three_l503_503228

theorem ratio_sum_eq_three
(a1 b1 c1 a2 b2 c2 x1 y1 x2 y2 : ℝ)
(h1 : a1 * x1 + b1 * y1 = c1)
(h2 : a2 * x2 + b2 * y2 = c2)
(h3 : a1 + b1 = c1)
(h4 : a2 + b2 = 2 * c2)
(h5 : real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) ≥ real.sqrt (2) / 2) :
  c1 / a1 + a2 / c2 = 3 :=
by
  sorry

end ratio_sum_eq_three_l503_503228


namespace A_beats_B_by_meters_l503_503687

theorem A_beats_B_by_meters:
  ∀ (d race_time_A diff_seconds : ℝ),
  d = 1000 → race_time_A = 490 → diff_seconds = 10 →
  (d / race_time_A) * diff_seconds ≈ 20.408 :=
by
  intros d race_time_A diff_seconds d_eq race_time_A_eq diff_seconds_eq
  rw [d_eq, race_time_A_eq, diff_seconds_eq]
  convert rfl
  simp
  sorry

end A_beats_B_by_meters_l503_503687


namespace cuberoot_sum_eq_three_l503_503730

theorem cuberoot_sum_eq_three (x r s : ℝ) (h : (∛x) + ∛(30 - x) = 3) 
  (hx : x = (r - Real.sqrt s)^3) : r + s = 102 :=
sorry

end cuberoot_sum_eq_three_l503_503730


namespace part_one_a_part_one_b_part_two_l503_503628

noncomputable def f (a x : ℝ) := x^3 + (4 - a) * x^2 - 15 * x + a

-- Problem (I) Part 1: Prove that P(0, -2) implies a = -2
theorem part_one_a (a : ℝ) (h₁ : f a 0 = -2) : a = -2 := sorry

-- Problem (I) Part 2: Given a = -2, prove the minimum value of f(x)
theorem part_one_b (h₂ : f (-2) 1 = -10) : ∀ x : ℝ, f (-2) x ≥ -10 := sorry

-- Problem (II): Prove the maximum value of a given monotonic decreasing condition
theorem part_two (a : ℝ) (h₃ : ∀ x ∈ Ioc (-1:ℝ) 1, f a' x ≤ 0) : a ≤ 10 := sorry

end part_one_a_part_one_b_part_two_l503_503628


namespace find_A_l503_503944

theorem find_A (A B C : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9)
  (h3 : A * 10 + B + B * 10 + C = B * 100 + C * 10 + B) : 
  A = 9 :=
  sorry

end find_A_l503_503944


namespace impossible_to_convince_logical_jury_of_innocence_if_guilty_l503_503027

theorem impossible_to_convince_logical_jury_of_innocence_if_guilty :
  (guilty : Prop) →
  (jury_is_logical : Prop) →
  guilty →
  (∀ statement : Prop, (logical_deduction : Prop) → (logical_deduction → ¬guilty)) →
  False :=
by
  intro guilty jury_is_logical guilty_premise logical_argument
  sorry

end impossible_to_convince_logical_jury_of_innocence_if_guilty_l503_503027


namespace evaluate_log_subtraction_l503_503188

theorem evaluate_log_subtraction : 
  log 3 243 - log 3 (1/27) = 8 :=
by
  sorry

end evaluate_log_subtraction_l503_503188


namespace avg_of_9_observations_l503_503416

variable {α : Type*} [Field α]

def avg (l : List α) : α :=
  l.sum / l.length

theorem avg_of_9_observations (l : List α) (h_length : l.length = 9)
  (h_first_5_avg : avg (l.take 5) = 10)
  (h_last_5_avg : avg (l.drop 4) = 8)
  (h_fifth_obs : l.nth 4 = some 18) :
  avg l = 8 :=
sorry

end avg_of_9_observations_l503_503416


namespace tunnel_length_general_tunnel_length_specific_l503_503456

-- Define the variables and conditions
variables (a b : ℕ) (T : ℕ)
variable (hT : T = 120)

-- Define the tunnel length function L
def tunnel_length (a b T : ℕ) : ℕ := (a + b) * T

-- General proof statement
theorem tunnel_length_general : tunnel_length a b 120 = (a + b) * 120 :=
by
  rfl

-- Specific proof statement when a = 11 and b = 9
theorem tunnel_length_specific (h_a : a = 11) (h_b : b = 9) : tunnel_length a b 120 = 2400 :=
by
  rw [tunnel_length, h_a, h_b]
  norm_num
  rfl

end tunnel_length_general_tunnel_length_specific_l503_503456


namespace equal_chords_angle_bisector_l503_503185

variable {O : Type} [metric_space O] [has_norm O] 
variables {A B C D M : O}
variable {radius : ℝ}

-- Define the circle with center O and radius
def is_circle (O : O) (radius : ℝ) (X : O) : Prop := dist O X = radius

-- Circles with chords AB and CD meeting at M
def equal_chords_intersecting_at (O : O) (A B C D M : O) 
  (h1 : is_circle O radius A) (h2 : is_circle O radius B)
  (h3 : is_circle O radius C) (h4 : is_circle O radius D)
  (h_eq : dist A B = dist C D) (h_intersect : dist O M ≠ 0) : Prop :=
∃ X Y : O, ⟪X, M⟫ ⟪Y, M⟫ ⟪O, M⟫

-- Defining the property that MO is the angle bisector between the chords AB and CD
def is_angle_bisector (O A B C D M : O) : Prop :=
angle (O, M, A) = angle (O, M, C)

-- Statement
theorem equal_chords_angle_bisector (O A B C D M : O) (radius : ℝ) :
  (is_circle O radius A) →
  (is_circle O radius B) →
  (is_circle O radius C) →
  (is_circle O radius D) →
  (dist A B = dist C D) →
  (dist O M ≠ 0) →
  is_angle_bisector O A B C D M :=
by sorry

end equal_chords_angle_bisector_l503_503185


namespace slope_of_line_l503_503227

open Real

structure Point :=
(x : ℝ)
(y : ℝ)

def Circle (center : Point) (radius : ℝ) := 
  ∀ (p : Point), (p.x - center.x) ^ 2 + (p.y - center.y) ^ 2 = radius ^ 2

def Line := Point → Point → Point

def isMidpoint (A B M : Point) : Prop := 
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

noncomputable def slope (P1 P2 : Point) : ℝ :=
  (P2.y - P1.y) / (P2.x - P1.x)

def CircleC : Circle ⟨3, 5⟩ (sqrt 5) :=
  λ p, (p.x - 3) ^ 2 + (p.y - 5) ^ 2 = 5

def line_through_center_intersects_Y (l : Line) (P C : Point) : Prop :=
  l C P = ⟨0, P.y⟩

def two_PA_eq_PB (P A B : Point) : Prop :=
  2 * A.x = B.x ∧ 2 * A.y = B.y

theorem slope_of_line (l : Line) (A B P C : Point) 
  (h_circle : CircleC C)
  (h_line : l C A = A ∧ l C B = B ∧ l C P = ⟨0, P.y⟩)
  (h_midpoint : isMidpoint P B A)
  (h_slope_cond : two_PA_eq_PB P A B) : 
  slope C P = 2 ∨ slope C P = -2 :=
by 
  sorry

end slope_of_line_l503_503227


namespace nick_pennsylvania_state_quarters_l503_503392

theorem nick_pennsylvania_state_quarters:
  let quarters : ℕ := 35
  let state_fraction : ℚ := 2 / 5
  let penn_fraction : ℚ := 0.5
  let state_quarters := quarters * state_fraction
  let penn_quarters := state_quarters * penn_fraction
  penn_quarters = 7 := by
  -- definitions
  let quarters := 35
  let state_fraction := 2 / 5
  let penn_fraction := 0.5
  let state_quarters := quarters * state_fraction
  let penn_quarters := state_quarters * penn_fraction
  -- definitions and simplification show that total Pennsylvania state quarters should be 7
  show state_quarters * penn_fraction = 7
  sorry

end nick_pennsylvania_state_quarters_l503_503392


namespace triangles_from_decagon_l503_503670

-- Define the parameters for the problem
def n : ℕ := 10
def k : ℕ := 3

-- Define the combination formula
def combination (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- State the theorem we want to prove
theorem triangles_from_decagon : combination n k = 120 := by
  -- Proof steps would go here
  sorry

end triangles_from_decagon_l503_503670


namespace converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l503_503105

variable (a b : ℝ)

theorem converse_of_proposition :
  (ab > 0 → a > 0 ∧ b > 0) = false := sorry

theorem inverse_of_proposition :
  (a ≤ 0 ∨ b ≤ 0 → ab ≤ 0) = false := sorry

theorem contrapositive_of_proposition :
  (ab ≤ 0 → a ≤ 0 ∨ b ≤ 0) = true := sorry

end converse_of_proposition_inverse_of_proposition_contrapositive_of_proposition_l503_503105


namespace initial_pages_l503_503449

variable (P : ℕ)
variable (h : 20 * P - 20 = 220)

theorem initial_pages (h : 20 * P - 20 = 220) : P = 12 := by
  sorry

end initial_pages_l503_503449


namespace compute_Q_at_1_l503_503168

noncomputable def P (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x + 10

def roots (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
a + b + c = -3 ∧ ab + bc + ca = 6 ∧ abc = -10

def Q (x : ℝ) (a b c : ℝ) : ℝ :=
(x - a * b) * (x - b * c) * (x - c * a)

theorem compute_Q_at_1
  (a b c : ℝ)
  (h : roots P a b c) :
  |Q 1 a b c| = 75 :=
sorry

end compute_Q_at_1_l503_503168


namespace sum_of_A_and_C_l503_503561

theorem sum_of_A_and_C :
  ∀ (A B C D : ℕ), A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A ∈ {1, 2, 3, 4, 5} → B ∈ {1, 2, 3, 4, 5} → C ∈ {1, 2, 3, 4, 5} → D ∈ {1, 2, 3, 4, 5} →
  (A / B + C / D = 2) → (A + C = 6) :=
by
  intros A B C D hAB hAC hAD hBC hBD hCD hA hB hC hD hABEquality
  sorry

end sum_of_A_and_C_l503_503561


namespace problem_solution_l503_503026

variables (p q : Prop)

theorem problem_solution (h1 : ¬ (p ∧ q)) (h2 : p ∨ q) : ¬ p ∨ ¬ q := by
  sorry

end problem_solution_l503_503026


namespace length_of_mini_train_l503_503098

-- Define the given conditions
def speed_kmph : ℝ := 75
def time_seconds : ℝ := 3
def conversion_factor : ℝ := 1000 / 3600

-- Convert speed from kmph to m/s
def speed_mps : ℝ := speed_kmph * conversion_factor

-- State the theorem to prove the length of the mini-train
theorem length_of_mini_train :
  speed_mps * time_seconds = 62.5 :=
by
  sorry

end length_of_mini_train_l503_503098


namespace kendall_nickels_l503_503349

def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

theorem kendall_nickels (q d : ℕ) (total : ℝ) (hq : q = 10) (hd : d = 12) (htotal : total = 4) : 
  ∃ n : ℕ, value_of_nickels n = total - (value_of_quarters q + value_of_dimes d) ∧ n = 6 :=
by
  sorry

end kendall_nickels_l503_503349


namespace last_number_in_sequence_is_123_l503_503914

theorem last_number_in_sequence_is_123 : 
  ∀ (a b c d e f : ℕ), a = 2 ∧ b = 3 ∧ c = 6 ∧ d = 15 ∧ e = 33 ∧ f = 123 → f = 123 :=
by
  intros a b c d e f h
  cases h
  cases h_left
  cases h_right
  cases h_right_right
  cases h_right_right_right
  cases h_right_right_right_right
  apply h_right_right_right_right_right
  sorry

end last_number_in_sequence_is_123_l503_503914


namespace charles_earnings_l503_503546

def housesit_rate : ℝ := 15
def dog_walk_rate : ℝ := 22
def hours_housesit : ℝ := 10
def num_dogs : ℝ := 3

theorem charles_earnings :
  housesit_rate * hours_housesit + dog_walk_rate * num_dogs = 216 :=
by
  sorry

end charles_earnings_l503_503546


namespace tomato_count_after_harvest_l503_503716

theorem tomato_count_after_harvest :
  let plant_A_initial := 150
  let plant_B_initial := 200
  let plant_C_initial := 250
  -- Day 1
  let plant_A_after_day1 := plant_A_initial - (plant_A_initial * 3 / 10)
  let plant_B_after_day1 := plant_B_initial - (plant_B_initial * 1 / 4)
  let plant_C_after_day1 := plant_C_initial - (plant_C_initial * 4 / 25)
  -- Day 7
  let plant_A_after_day7 := plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)
  let plant_B_after_day7 := plant_B_after_day1 - (plant_B_after_day1 * 1 / 5)
  let plant_C_after_day7 := plant_C_after_day1 - ((plant_C_initial * 4 / 25) * 2)
  -- Day 14
  let plant_A_after_day14 := plant_A_after_day7 - ((plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)) * 3)
  let plant_B_after_day14 := plant_B_after_day7 - ((plant_B_after_day1 * 1 / 5) + 15)
  let plant_C_after_day14 := plant_C_after_day7 - (plant_C_after_day7 * 1 / 5)
  (plant_A_after_day14 = 0) ∧ (plant_B_after_day14 = 75) ∧ (plant_C_after_day14 = 104) :=
by
  sorry

end tomato_count_after_harvest_l503_503716


namespace find_vertices_of_triangle_l503_503172

/-- 
Given the center of the circumscribed circle $O$, the orthocenter $M$, 
and one vertex of the triangle $A$, the vertices $B$ and $C$ of the triangle 
are at the intersections of the perpendicular bisector of the 
reflection of the orthocenter $M$ across the line $OA$ with the circumcircle 
centered at $O$ and radius $OA$.
-/
theorem find_vertices_of_triangle
  (O M A : Point)
  (circumcircle : Circle O (dist O A))
  (M1 : Point := reflect M (line_through O A))
  (P : Point := midpoint M M1)
  (bisector_intersections : (perpendicular_bisector M M1).intersections circumcircle)
  : ∃ (B C : Point), B ∈ bisector_intersections ∧ C ∈ bisector_intersections ∧ 
                      (triangle A B C).circumcenter = O ∧ (triangle A B C).orthocenter = M :=
sorry

end find_vertices_of_triangle_l503_503172


namespace complete_graph_triangle_inequality_l503_503008

open_locale big_operators

theorem complete_graph_triangle_inequality (n : ℕ) (h : n > 1000) :
  ∃ (f : finset (fin (n.choose 2)) → ℕ), 
  (∀ (p : finset (fin n)) (hp : p.card = 3), 
  (∑ e in p, f e) ≥ nat.floor (3 * n - 1000 * real.log (real.log n))) :=
sorry

end complete_graph_triangle_inequality_l503_503008


namespace polynomial_divisibility_l503_503386

theorem polynomial_divisibility
  (a b c : ℤ)
  (h_root : ∃ x₁ x₂ x₃ : ℤ, x₁ * x₂ * x₃ = -c ∧ x₁ + x₂ + x₃ = -a ∧ x₁ = x₂ * x₃)
  : ∃ k : ℤ, 2f(-1) = k * (f(1) + f(-1) - 2 * (1 + f(0)))
where
  f (x : ℤ) : ℤ := x^3 + a * x^2 + b * x + c := sorry

end polynomial_divisibility_l503_503386


namespace ac_bisects_bd_l503_503224

noncomputable theory

variables (A B C D E F M N : Type) [linear_ordered_comm_group A] [linear_ordered_comm_group B]
[linear_ordered_comm_group C] [linear_ordered_comm_group D] [linear_ordered_comm_group E]
[linear_ordered_comm_group F] [linear_ordered_comm_group M] [linear_ordered_comm_group N]

-- Given data setup for the quadrilateral and its properties
variables (AB CD BC AD BD EF AC : Set (A × B))
variables (hBD_parallel_EF : parallel BD EF)
variables (hE_intersection : intersection AB CD E)
variables (hF_intersection : intersection BC AD F)
variables (hAC_intersects_BD_at_M : intersection AC BD M)
variables (hAC_intersects_EF_at_N : intersection AC EF N)

-- Proof goal
theorem ac_bisects_bd (h : quadrilateral A B C D ∧ parallel BD EF ∧
  intersection AB CD E ∧ intersection BC AD F ∧
  intersection AC BD M ∧ intersection AC EF N) :
  segment_bisector AC BD :=
by { sorry }

end ac_bisects_bd_l503_503224


namespace smallest_natural_number_meeting_conditions_l503_503852

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem smallest_natural_number_meeting_conditions :
  ∃ (a : ℕ), (a + 11 = 19000000000) ∧ (digit_sum a % 11 = 0) ∧ (digit_sum (a + 11) % 11 = 0) ∧ (a = 18999999999) := by
  exists 18999999999
  split
  sorry

end smallest_natural_number_meeting_conditions_l503_503852


namespace roller_speed_range_l503_503913

theorem roller_speed_range
  (roller_width : ℝ) (overlap : ℝ) (section_length : ℝ) (section_width : ℝ)
  (time_min : ℝ) (time_max : ℝ) :
  roller_width = 0.85 →
  overlap = 0.25 →
  section_length = 750 →
  section_width = 6.5 →
  time_min = 5 →
  time_max = 6 →
  let effective_width := roller_width * (1 - overlap) in
  let num_passes := (section_width / effective_width).ceil.to_nat in
  let total_distance := 2 * section_length * num_passes in
  let speed_min := total_distance / time_max in
  let speed_max := total_distance / time_min in
  2.75 ≤ speed_min ∧ speed_max ≤ 3.3 :=
by
  intros
  assume h0 h1 h2 h3 h4 h5
  let effective_width := roller_width * (1 - overlap)
  let num_passes := (section_width / effective_width).ceil.to_nat
  let total_distance := 2 * section_length * num_passes
  let speed_min := total_distance / time_max
  let speed_max := total_distance / time_min
  sorry

end roller_speed_range_l503_503913


namespace _l503_503547

noncomputable theorem sum_of_squares_of_roots :
  (∀ (a b c : ℂ), (∃ (h : IsRoot (3 * X ^ 3 + 2 * X ^ 2 - 5 * X - 15) a)
                      (h : IsRoot (3 * X ^ 3 + 2 * X ^ 2 - 5 * X - 15) b)
                      (h : IsRoot (3 * X ^ 3 + 2 * X ^ 2 - 5 * X - 15) c),
                      a ≠ b ∧ b ≠ c ∧ a ≠ c ) →
    a^2 + b^2 + c^2 = -26 / 9) := sorry

end _l503_503547


namespace log_difference_l503_503189

theorem log_difference : 
  log 3 243 - log 3 (1 / 27) = 8 := by sorry

end log_difference_l503_503189


namespace lars_breads_per_day_l503_503354

noncomputable theory

def loaves_per_hour := 10
def baguettes_per_2hours := 30
def baking_hours_per_day := 6

theorem lars_breads_per_day : 
  (loaves_per_hour * baking_hours_per_day) + (baguettes_per_2hours * (baking_hours_per_day / 2)) = 105 :=
by
  let loaves := loaves_per_hour * baking_hours_per_day
  let baguettes := baguettes_per_2hours * (baking_hours_per_day / 2)
  have h_loaves : loaves = 60 := sorry  -- skipped proof
  have h_baguettes : baguettes = 45 := sorry  -- skipped proof
  show loaves + baguettes = 105 by rw [h_loaves, h_baguettes]; exact rfl

end lars_breads_per_day_l503_503354


namespace subset_count_eq_sixteen_l503_503044

-- Definition of the universal set
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Defining the given subset
def A : Set ℕ := {1, 2, 3}

-- Goal: prove that the number of subsets of U that include A is 16
theorem subset_count_eq_sixteen : 
  {X : Set ℕ // A ⊆ X ∧ X ⊆ U}.card = 16 := 
sorry

end subset_count_eq_sixteen_l503_503044


namespace smallest_n_modified_special_sum_l503_503174

def is_modified_special (x : ℝ) : Prop :=
  ∃ (digits : ℕ → ℕ), (∀ n, digits n ∈ {0, 3}) ∧ x = (∑ i in (finset.range digits.length), digits i * (10 : ℝ)^(- i : ℤ))

theorem smallest_n_modified_special_sum :
  ∃ n, (∀ (l : list ℝ), list.length l = n → (∀ x ∈ l, is_modified_special x) → (list.sum l = 1)) :=
  sorry

end smallest_n_modified_special_sum_l503_503174


namespace percent_of_12356_equals_1_2356_l503_503460

theorem percent_of_12356_equals_1_2356 (p : ℝ) (h : p * 12356 = 1.2356) : p = 0.0001 := sorry

end percent_of_12356_equals_1_2356_l503_503460


namespace total_weight_lifted_l503_503837

-- Definitions based on conditions
def original_lift : ℝ := 80
def after_training : ℝ := original_lift * 2
def specialization_increment : ℝ := after_training * 0.10
def specialized_lift : ℝ := after_training + specialization_increment

-- Statement of the theorem to prove total weight lifted
theorem total_weight_lifted : 
  (specialized_lift * 2) = 352 :=
sorry

end total_weight_lifted_l503_503837


namespace factor_difference_of_squares_l503_503967

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l503_503967


namespace graphs_symmetric_l503_503588

theorem graphs_symmetric : ∀ (f : ℝ → ℝ), 
  (∀ x, f(x-1) = f(1-x)) ↔ (∃ (l : ℝ), l = 1) :=
by 
  intro f
  sorry

end graphs_symmetric_l503_503588


namespace relative_error_comparison_l503_503528

theorem relative_error_comparison :
  ∀ (error1 len1 error2 len2 : ℝ), 
    error1 = 0.02 → len1 = 10 → 
    error2 = 0.2 → len2 = 100 →
    (error1 / len1) = (error2 / len2) := 
by {
  intros error1 len1 error2 len2 h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  simp,
  sorry
}

end relative_error_comparison_l503_503528


namespace geometric_series_second_term_l503_503937

theorem geometric_series_second_term (a : ℝ) (r : ℝ) (sum : ℝ) 
  (h1 : r = 1/4) 
  (h2 : sum = 40) 
  (sum_formula : sum = a / (1 - r)) : a * r = 7.5 :=
by {
  -- Proof to be filled in later
  sorry
}

end geometric_series_second_term_l503_503937


namespace knight_tour_impossible_4x4_l503_503005

theorem knight_tour_impossible_4x4 :
  ¬ ∃ tour : ℕ → (ℕ × ℕ), 
    (∀ i < 16, ∀ j < 16, i ≠ j → tour i ≠ tour j) ∧
    (∀ k < 15, let (x1, y1) := tour k, (x2, y2) := tour (k + 1) 
              in (x1 ≠ x2 ∧ y1 ≠ y2 ∧ (abs (x1 - x2) = 1 ∧ abs (y1 - y2) = 2) ∨
                  abs (x1 - x2) = 2 ∧ abs (y1 - y2) = 1)) ∧
    (∀ k < 16, let (x, y) := tour k 
              in 1 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 4) :=
by
  sorry

end knight_tour_impossible_4x4_l503_503005


namespace age_of_child_l503_503889

theorem age_of_child 
  (avg_age_3_years_ago : ℕ)
  (family_size_3_years_ago : ℕ)
  (current_family_size : ℕ)
  (current_avg_age : ℕ)
  (h1 : avg_age_3_years_ago = 17)
  (h2 : family_size_3_years_ago = 5)
  (h3 : current_family_size = 6)
  (h4 : current_avg_age = 17)
  : ∃ age_of_baby : ℕ, age_of_baby = 2 := 
by
  sorry

end age_of_child_l503_503889


namespace bulls_win_series_seven_games_l503_503413

open Nat

def probability_bulls_win_game : ℚ := 2 / 5
def probability_knicks_win_game : ℚ := 3 / 5

theorem bulls_win_series_seven_games :
  let probability_all_sequences := (binomial 6 3 : ℚ) * (probability_bulls_win_game ^ 3) * (probability_knicks_win_game ^ 3)
  let probability_seven_games := probability_all_sequences * probability_bulls_win_game
  probability_seven_games = 864 / 15625 :=
by
  sorry

end bulls_win_series_seven_games_l503_503413


namespace smallest_number_l503_503868

theorem smallest_number (b : ℕ) :
  (b % 3 = 2) ∧ (b % 4 = 2) ∧ (b % 5 = 3) → b = 38 :=
by
  sorry

end smallest_number_l503_503868


namespace miller_rabin_prime_indicator_miller_rabin_composite_indicator_l503_503764

theorem miller_rabin_prime_indicator (n : ℕ) (h_prime : Prime n) :
  ∀ (u : ℕ), u^(n-1) ≡ 1 [MOD n] :=
sorry

theorem miller_rabin_composite_indicator (n : ℕ) (h_composite : ¬ Prime n) :
  ∃ (u : ℕ), (u^(n-1) ≠ 1 [MOD n] ∨ ∃ (r : ℕ), (u^(2^r * (n - 1)/2^r) ≠ -1 [MOD n])) :=
sorry

end miller_rabin_prime_indicator_miller_rabin_composite_indicator_l503_503764


namespace rhombus_area_l503_503162

theorem rhombus_area (circumradius_EFG : ℝ) (circumradius_EHG : ℝ) (h₁ : circumradius_EFG = 15) (h₂ : circumradius_EHG = 30) : 
  let x := 12 in let y := 2 * x in (2 * x) * (2 * y) / 2 = 576 :=
by
  sorry

end rhombus_area_l503_503162


namespace decreasing_digits_between_200_250_l503_503652

def is_decreasing_digits (n : ℕ) : Prop :=
  let digits := List.reverse (Nat.digits 10 n) in
  digits.length = 3 ∧ digits.nth 0 > digits.nth 1 ∧ digits.nth 1 > digits.nth 2

def count_decreasing_digits_ints (l u : ℕ) : ℕ :=
  (List.range' l (u - l + 1)).count is_decreasing_digits

theorem decreasing_digits_between_200_250 : count_decreasing_digits_ints 200 250 = 1 :=
  by
    sorry

end decreasing_digits_between_200_250_l503_503652


namespace inequality_solution_l503_503820

theorem inequality_solution (x : ℝ) : x + 1 < (4 + 3 * x) / 2 → x > -2 :=
by
  intros h
  sorry

end inequality_solution_l503_503820


namespace math_problem_l503_503541

-- Define the expression
def e : ℤ :=
  (-1)^2 - abs (-3) + (-5) / (-5 / (3 : ℚ))

-- State the theorem
theorem math_problem : e = 1 :=
  sorry

end math_problem_l503_503541


namespace part_I_part_II_part_III_l503_503225

-- Definitions and conditions:
def seq_a (n : ℕ) : ℕ := 2^(n + 1) - 2
def S (n : ℕ) := 2 * (seq_a n) - 2

-- Part I: Prove the general term formula:
theorem part_I (n : ℕ) : seq_a n = 2^n := 
sorry

-- Definitions for Part II:
def b (n : ℕ) := Real.logb 2 (seq_a n)
def c (n : ℕ) := 1 / (b n * b (n + 1))
def T (n : ℕ) := ∑ i in Finset.range (n + 1), c i

-- Part II: Prove the sum of c_n:
theorem part_II (n : ℕ) : T n = n / (n + 1) := 
sorry

-- Definitions for Part III:
def d (n : ℕ) : ℕ := n * seq_a n
def G (n : ℕ) := ∑ i in Finset.range (n + 1), d i

-- Part III: Prove the sum of d_n:
theorem part_III (n : ℕ) : G n = (n - 1) * 2^(n + 1) + 2 := 
sorry

end part_I_part_II_part_III_l503_503225


namespace point_in_third_quadrant_l503_503880

theorem point_in_third_quadrant (θ : ℝ) (hθ1 : θ = 2014) :
  let A : ℝ × ℝ := (Real.sin (θ * Real.pi / 180), Real.cos (θ * Real.pi / 180)) in
  A.1 < 0 ∧ A.2 < 0 :=
by
  sorry

end point_in_third_quadrant_l503_503880


namespace ball_hits_ground_time_l503_503417

noncomputable def find_time_when_ball_hits_ground (a b c : ℝ) : ℝ :=
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)

theorem ball_hits_ground_time :
  find_time_when_ball_hits_ground (-16) 40 50 = (5 + 5 * Real.sqrt 3) / 4 :=
by
  sorry

end ball_hits_ground_time_l503_503417


namespace ramu_profit_percent_l503_503108

noncomputable def profitPercent
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (selling_price : ℝ) : ℝ :=
  ((selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost)) * 100

theorem ramu_profit_percent :
  profitPercent 42000 13000 61900 = 12.55 :=
by
  sorry

end ramu_profit_percent_l503_503108


namespace square_of_binomial_coefficient_l503_503970

theorem square_of_binomial_coefficient :
  ∃ t u : ℚ, (bx^2 + 21x + 9 = (t * x + u) ^ 2) → b = 49 / 4 :=
begin
  sorry
end

end square_of_binomial_coefficient_l503_503970


namespace find_principal_l503_503499

def si (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

def ci (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem find_principal :
  ∀ (P R T : ℝ), 
  (ci P R T - si P R T = 20) → (R = 10) → (T = 2) → P = 2000 :=
by
  intro P R T h1 h2 h3
  sorry

end find_principal_l503_503499


namespace num_triangles_from_decagon_l503_503656

theorem num_triangles_from_decagon (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ c ≠ a → ¬(collinear a b c)) :
  (nat.choose 10 3) = 120 :=
by
  sorry

end num_triangles_from_decagon_l503_503656


namespace infinite_a_without_solution_l503_503502

def tau (n : ℕ) : ℕ := {
  count := λ (d : ℕ), ∃ (h : d ∣ n), ()
}

theorem infinite_a_without_solution :
  ∃ (a_seq : ℕ → ℕ), (∀ k : ℕ, ∀ n : ℕ, tau (a_seq k * n) ≠ n) ∧ (∀ k : ℕ, a_seq k > 0) :=
sorry

end infinite_a_without_solution_l503_503502


namespace equation_of_tangent_line_l503_503627

noncomputable def f (m x : ℝ) := m * Real.exp x - x - 1

def passes_through_P (m : ℝ) : Prop :=
  f m 0 = 1

theorem equation_of_tangent_line (m : ℝ) (h : passes_through_P m) :
  (f m) 0 = 1 → (2 - 1 = 1) ∧ ((y - 1 = x) → (x - y + 1 = 0)) :=
by
  intro h
  sorry

end equation_of_tangent_line_l503_503627


namespace solve_system_l503_503169

theorem solve_system :
  ∀ x y : ℚ, (3 * x + 4 * y = 12) ∧ (9 * x - 12 * y = -24) →
  (x = 2 / 3) ∧ (y = 5 / 2) :=
by
  intro x y
  intro h
  sorry

end solve_system_l503_503169


namespace inverse_function_correct_l503_503040

noncomputable def func (x : ℝ) : ℝ :=
if x < 0 then x else x^2

noncomputable def inv_func (y : ℝ) : ℝ :=
if y < 0 then y else sqrt y

theorem inverse_function_correct : 
  ∀ (x : ℝ), (inv_func (func x)) = x :=
by
  intro x
  unfold func inv_func
  split_ifs
  { -- Case: x < 0
    simp [h]
  }
  { -- Case: x ≥ 0
    have hy : sqrt (x^2) = x := by
      rw sqrt_sq_eq_abs x
      rw abs_of_nonneg h
    exact hy
  }

end inverse_function_correct_l503_503040


namespace range_of_f_is_0_to_1_l503_503608

def f (x : ℝ) (k : ℝ) := x^k

noncomputable def range_of_f_on_interval (k : ℝ) (h : k < 0) : Set ℝ :=
  (λ y, ∃ x : ℝ, 1 ≤ x ∧ f x k = y) '' (Set.Ici (1 : ℝ))

theorem range_of_f_is_0_to_1 (k : ℝ) (h : k < 0) :
  range_of_f_on_interval k h = Set.Ioc (0 : ℝ) (1 : ℝ) := 
  sorry

end range_of_f_is_0_to_1_l503_503608


namespace volleyball_team_total_score_l503_503685

-- Define the conditions
def LizzieScore := 4
def NathalieScore := LizzieScore + 3
def CombinedLizzieNathalieScore := LizzieScore + NathalieScore
def AimeeScore := 2 * CombinedLizzieNathalieScore
def TeammatesScore := 17

-- Prove that the total team score is 50
theorem volleyball_team_total_score :
  LizzieScore + NathalieScore + AimeeScore + TeammatesScore = 50 :=
by
  sorry

end volleyball_team_total_score_l503_503685


namespace find_r_in_geometric_sequence_l503_503319

noncomputable def sum_of_geometric_sequence (n : ℕ) (r : ℝ) := 3^n + r

theorem find_r_in_geometric_sequence:
  ∃ r : ℝ, 
  (∀ n : ℕ, n >= 1 → sum_of_geometric_sequence n r = 3^n +  r) ∧
  (∀ n : ℕ, n >= 2 → 
    let S_n := sum_of_geometric_sequence n r in
    let S_n_minus_1 := sum_of_geometric_sequence (n - 1) r in
    let a_n := S_n - S_n_minus_1 in
    a_n = 2 * 3^(n - 1)) ∧
  (∃ a1 : ℝ, a1 = 3 + r ∧ a1 * 3 = 6) ∧
  r = -1 :=
by sorry

end find_r_in_geometric_sequence_l503_503319


namespace minimum_box_value_l503_503294

def is_valid_pair (a b : ℤ) : Prop :=
  a * b = 15 ∧ (a^2 + b^2 ≥ 34)

theorem minimum_box_value :
  ∃ (a b : ℤ), is_valid_pair a b ∧ (∀ (a' b' : ℤ), is_valid_pair a' b' → a^2 + b^2 ≤ a'^2 + b'^2) ∧ a^2 + b^2 = 34 :=
by
  sorry

end minimum_box_value_l503_503294


namespace hyperbola_vertex_to_asymptote_distance_l503_503578

noncomputable def distance_from_vertex_to_asymptote :
    ℝ :=
  let a := 1
  let b := real.sqrt 2
  let point := (1, 0)
  let line_A := real.sqrt 2
  let line_B := -1
  let line_C := 0
  real.abs ((line_A * point.1 + line_B * point.2 + line_C) / (real.sqrt (line_A^2 + line_B^2)))

theorem hyperbola_vertex_to_asymptote_distance :
  distance_from_vertex_to_asymptote = (real.sqrt 6) / 3 :=
  sorry

end hyperbola_vertex_to_asymptote_distance_l503_503578


namespace kendall_nickels_l503_503348

def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

theorem kendall_nickels (q d : ℕ) (total : ℝ) (hq : q = 10) (hd : d = 12) (htotal : total = 4) : 
  ∃ n : ℕ, value_of_nickels n = total - (value_of_quarters q + value_of_dimes d) ∧ n = 6 :=
by
  sorry

end kendall_nickels_l503_503348


namespace trapezoid_area_efgh_l503_503857

theorem trapezoid_area_efgh :
  let E := (0, 0)
  let F := (0, -3)
  let G := (5, 0)
  let H := (5, 8)
  ∃ (area : ℝ), 
    area = (1 / 2) * ((3 : ℝ) + 8) * 5 ∧ 
    area = 27.5 :=
by
  let E := (0 : ℝ, 0 : ℝ)
  let F := (0 : ℝ, -3 : ℝ)
  let G := (5 : ℝ, 0 : ℝ)
  let H := (5 : ℝ, 8 : ℝ)
  use (1 / 2) * (3 + 8) * 5
  split
  · exact rfl
  · norm_num
  · sorry

end trapezoid_area_efgh_l503_503857


namespace negation_of_P_equiv_l503_503278

-- Define the proposition P
def P : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of P equivalently
theorem negation_of_P_equiv :
  ¬ P ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := 
sorry

end negation_of_P_equiv_l503_503278


namespace fraction_not_declared_major_23_over_60_l503_503314

variable (N : ℕ)
variable (total_students : ℕ := 3 * N)

def first_year_students := N
def second_year_students := N
def third_year_students := N

def not_declared_major_first_year := 4 / 5 * first_year_students
def declared_major_first_year := 1 / 5 * first_year_students
def declared_major_second_year := 1 / 2 * declared_major_first_year
def not_declared_major_second_year := second_year_students - declared_major_second_year
def not_declared_major_third_year := 1 / 4 * third_year_students

def fraction_of_students_not_declared_major_second_and_third_year :=
  (not_declared_major_second_year + not_declared_major_third_year) / total_students

theorem fraction_not_declared_major_23_over_60 :
  fraction_of_students_not_declared_major_second_and_third_year = 23 / 60 := sorry

end fraction_not_declared_major_23_over_60_l503_503314


namespace d_mu_M_l503_503343

section leap_year_data

-- Definition of the dataset
def dataset : list ℕ :=
  (list.replicate 12 [1, 2, ..., 29]).bind id ++ 
  list.replicate 11 30 ++ 
  list.replicate 7 31

-- Mean (μ) calculation
def mean (data : list ℕ) : ℚ :=
  (data.sum : ℚ) / data.length

-- Median calculation
def median (data : list ℕ) : ℚ :=
  let sorted_data := data.qsort(≤)
  if sorted_data.length % 2 = 1 then
    sorted_data.nth_le (sorted_data.length / 2) sorry
  else
    (sorted_data.nth_le (sorted_data.length / 2 - 1) sorry + 
      sorted_data.nth_le (sorted_data.length / 2) sorry) / 2

-- Median of the modes
def median_of_modes (data : list ℕ) : ℕ :=
  let modes := data.filter (λ x, data.count x = data.foldr (λ y acc, max (data.count y) acc) 0)
  let mode_median_idx := modes.length / 2
  modes.nth_le mode_median_idx sorry

-- Theorem: Proving the relationship d < μ < M
theorem d_mu_M (data : list ℕ) (h : dataset = data) :
  let μ := mean data
  let M := median data
  let d := median_of_modes data
  d < μ ∧ μ < M :=
by 
  sorry

end leap_year_data

end d_mu_M_l503_503343


namespace semicircle_circumference_is_correct_l503_503884

def rectangle_length : ℝ := 4
def rectangle_breadth : ℝ := 2
def rectangle_perimeter := 2 * (rectangle_length + rectangle_breadth)
def square_perimeter := rectangle_perimeter
def square_side := square_perimeter / 4
def semicircle_diameter := square_side
def semicircle_circumference := (Real.pi * semicircle_diameter) / 2 + semicircle_diameter
def semicircle_circumference_rounded := (Float.round (semicircle_circumference * 100) / 100).toReal

theorem semicircle_circumference_is_correct :
  semicircle_circumference_rounded = 7.71 := by
  sorry

end semicircle_circumference_is_correct_l503_503884


namespace triangle_area_l503_503369

-- Define the vectors a and b as given in the condition
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (-5, 2)

-- Using the mathematical formula for determinant and area calculation, 
-- state the problem to prove the area of the triangle
theorem triangle_area : 
  let parallelogram_area := abs ((a.1 * b.2) - (a.2 * b.1)) in
  (parallelogram_area / 2) = 11 / 2 :=
by
  sorry

end triangle_area_l503_503369


namespace tens_digit_of_8_l503_503461

def last_two_digits_of_powers_of_8 : List (Fin 100) :=
  [08, 64, 12, 96, 68, 44, 52, 16, 28, 24, 92, 36, 88, 04, 32, 56, 48, 84, 72, 76]

def tens_digit (n : Fin 100) : Nat := n / 10

theorem tens_digit_of_8^1998 : tens_digit (last_two_digits_of_powers_of_8.get (1998 % 20)) = 8 := by
  sorry

end tens_digit_of_8_l503_503461


namespace max_mag_of_diff_l503_503286

noncomputable def a (theta : ℝ) : ℝ × ℝ := (Real.cos theta, Real.sin theta)
def b : ℝ × ℝ := (Real.sqrt 3, -1)

def mag_of_diff (theta : ℝ) : ℝ :=
  Real.sqrt ((2 * Real.cos theta - Real.sqrt 3)^2 + (2 * Real.sin theta + 1)^2)

theorem max_mag_of_diff (theta : ℝ) : ∃ θ : ℝ, mag_of_diff θ = 4 :=
by
  sorry

end max_mag_of_diff_l503_503286


namespace math_problem_solution_l503_503990

noncomputable def math_problem_statement (a b c : ℝ) (h_a : a < 0) (h_b : b < 0) (h_c : c < 0) : Prop :=
  (sqrt (a / (b + c)) + (1 / sqrt 2)) ^ 2 +
  (sqrt (b / (c + a)) + (1 / sqrt 2)) ^ 2 +
  (sqrt (c / (a + b)) + (1 / sqrt 2)) ^ 2 ≥ 6

theorem math_problem_solution (a b c : ℝ) (h_a : a < 0) (h_b : b < 0) (h_c : c < 0) : 
  math_problem_statement a b c h_a h_b h_c :=
sorry

end math_problem_solution_l503_503990


namespace tourists_escape_l503_503775

theorem tourists_escape (T : ℕ) (hT : T = 10) (hats : Fin T → Bool) (could_see : ∀ (i : Fin T), Fin (i) → Bool) :
  ∃ strategy : (Fin T → Bool), (∀ (i : Fin T), (strategy i = hats i) ∨ (strategy i ≠ hats i)) →
  (∀ (i : Fin T), (∀ (j : Fin T), i < j → strategy i = hats i) → ∃ count : ℕ, count ≥ 9 ∧ ∀ (i : Fin T), count ≥ i → strategy i = hats i) := sorry

end tourists_escape_l503_503775


namespace rhombus_area_l503_503555
open Complex

noncomputable def poly := (z : ℂ) => z^4 + 4*I*z^3 + (1 + I)*z^2 + (2 + 4*I)*z + (1 - 4*I)

theorem rhombus_area :
  let roots := [a, b, c, d] where ⟦a, b, c, d⟧ ⊆ spectrum ℂ poly
  let center := (a + b + c + d) / 4
  let p := |a - center|
  let q := |b - center|
  area (rhombus formed by roots) = sqrt 29 :=
  sorry

end rhombus_area_l503_503555


namespace not_possible_to_obtain_l503_503393

noncomputable def can_obtain (S : List ℕ) (target : ℕ) : Bool :=
  if target = 1 then true else -- case for multiplication beginning with 1
  if S = [] then false else
  (target ∈ S) ∨ (∃ x y ∈ S, can_obtain ((x * y) :: S) target)

theorem not_possible_to_obtain :
  ∀ (S : List ℕ), 
    S = [20, 100] → 
    ¬ can_obtain S (10 ^ 2015) := 
by
  sorry

end not_possible_to_obtain_l503_503393


namespace problem_bc_d_l503_503606

noncomputable def floor : ℝ → ℤ :=
  λ x, if 0 ≤ x then nat.floor x else - int.of_nat (nat.ceil (-x))

theorem problem_bc_d (x y : ℝ) : 
  (∀ x, x - 1 < floor x) ∧ 
  (∀ x y, floor x - floor y - 1 < x - y ∧ x - y < floor x - floor y + 1) ∧ 
  (∀ x, x^2 + 1/3 > floor x) := 
by 
  split 
  sorry 
  split 
  sorry 
  sorry

end problem_bc_d_l503_503606


namespace problem_B_correct_l503_503100

def necessary_but_not_sufficient (k : ℤ) : Prop :=
  let x := (k * Real.pi) / 4
  (∃ k : ℤ, tan x = 1) ∧ ∀ k : ℤ, tan x ≠ -1

theorem problem_B_correct : ∀ (k : ℤ), necessary_but_not_sufficient k :=
by
  -- statement of theorem without its proof
  sorry

end problem_B_correct_l503_503100


namespace Vitya_needs_58_offers_l503_503096

theorem Vitya_needs_58_offers :
  ∃ k : ℕ, (log 0.01 / log (12 / 13) < k) ∧ k = 58 :=
by
  sorry

end Vitya_needs_58_offers_l503_503096


namespace shaded_area_ratio_l503_503550

theorem shaded_area_ratio (s : ℝ) (h : s = 8) :
  let r := s / 2 in
  let A_semicircles := 2 * (1/2 * Real.pi * r^2) in
  let A_quarter_circle := 1/4 * Real.pi * r^2 in
  let A_full_circle := Real.pi * r^2 in
  let A_shaded := A_semicircles - A_quarter_circle in
  A_shaded / A_full_circle = 3 / 4 :=
by 
  sorry

end shaded_area_ratio_l503_503550


namespace trap_area_BCDK_l503_503129

-- Definitions
variables {A B C D E K M : Point}
variable {circle : Circle}
variable {rect : Rectangle}
variable {ab_len ke_ka_ratio : ℝ}
variable {area_BCKD : ℝ}

noncomputable def circle_pass_through_A_B_and_touch_CD_midpoint
  (hA : A ∈ circle)
  (hB : B ∈ circle)
  (hMidCD : M ∈ circle)
  (hMidCDCD : M.is_midpoint CD) : Prop := sorry

noncomputable def tangent_line_from_D
  (h_tangent_circle : is_tangent D E circle)
  (h_extend_AB : is_intersection K (extension AB) E) : Prop := sorry

-- Hypotheses
def problem_hypotheses : Prop :=
  circle_pass_through_A_B_and_touch_CD_midpoint
    (by simp [A])
    (by simp [B])
    (by simp [M])
    (by simp)
  ∧ tangent_line_from_D
    (by simp)
    (by simp)
  ∧ AB = 10
  ∧ KE / KA = 3 / 2

-- Goal: area_BCDK == 210
theorem trap_area_BCDK
  (h : problem_hypotheses)
  : area_BCKD = 210 :=
begin
  sorry
end

end trap_area_BCDK_l503_503129


namespace Lars_bake_total_breads_l503_503353

theorem Lars_bake_total_breads :
  (Lars_bakes_10_loaves_per_hour : ℕ → ℕ) → (Lars_bakes_30_baguettes_per_2_hours : ℕ → ℕ) →
  (Lars_bakes_for_6_hours : ℕ) →
  (total_breads_in_a_day : ℕ) :=
by
  let Lars_bakes_10_loaves_per_hour := (n : ℕ) → 10 * n
  let Lars_bakes_30_baguettes_per_2_hours := (n : ℕ) → 30 * n
  let Lars_bakes_for_6_hours := 6
  let total_loaves := Lars_bakes_10_loaves_per_hour Lars_bakes_for_6_hours
  let total_baguettes := Lars_bakes_30_baguettes_per_2_hours (Lars_bakes_for_6_hours / 2)
  let total_breads_in_a_day := total_loaves + total_baguettes
  have : total_breads_in_a_day = 150 := by
    sorry
  exact total_breads_in_a_day

end Lars_bake_total_breads_l503_503353


namespace correct_statement_l503_503103

theorem correct_statement :
  (∀ x k : ℤ, x = (k * π / 4) → tan x = 1 → x = (4 * k + 1) * π / 4) ∧
  ¬ (∀ x k : ℤ, (x = (k * π / 4) → (x = (4 * k + 1) * π / 4))) :=
begin
  sorry
end

end correct_statement_l503_503103


namespace range_of_k_l503_503615

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ (x, y) = (0, 0)) →
  0 < |k| ∧ |k| < 1 :=
by
  intros
  sorry

end range_of_k_l503_503615


namespace average_and_range_unchanged_l503_503303

/- Definitions for the given conditions -/

variables {n : ℕ} -- n should be a natural number
variables {x : ℕ → ℚ} -- x is a sequence of rational numbers
variable (x̄ : ℚ) -- average of initial data set
variable (a : ℚ) -- range of initial data set
variable (b : ℚ) -- median of initial data set
variable (s : ℚ) -- variance of initial data set

/- Definitions for the new data set after adding x̄ to original set -/
variable (x̄' : ℚ) -- new average
variable (a' : ℚ) -- new range
variable (b' : ℚ) -- new median
variable (s' : ℚ) -- new variance
variable data_new : ℕ → ℚ := λ i, if i = 0 then x̄ else x (i - 1)

/- Proof statement -/
theorem average_and_range_unchanged :
  x̄' = x̄ ∧ a' = a :=
sorry -- Proof is omitted as per instructions

end average_and_range_unchanged_l503_503303


namespace maximum_perimeter_trapezoid_l503_503963

theorem maximum_perimeter_trapezoid (R x y : ℝ)
  (h1 : 0 < R)
  (h2 : x = R)
  (h3 : 2R + (2R^2 - (x^2))/R + 2x = (2R + x) * 2 + 2R^2)  :
  x = R := 
sorry

end maximum_perimeter_trapezoid_l503_503963


namespace net_moles_nh3_after_reactions_l503_503551

/-- Define the stoichiometry of the reactions and available amounts of reactants -/
def step1_reaction (nh4cl na2co3 : ℕ) : ℕ :=
  if nh4cl / 2 >= na2co3 then 
    2 * na2co3
  else 
    2 * (nh4cl / 2)

def step2_reaction (koh h3po4 : ℕ) : ℕ :=
  0  -- No NH3 produced in this step

theorem net_moles_nh3_after_reactions :
  let nh4cl := 3
  let na2co3 := 1
  let koh := 3
  let h3po4 := 1
  let nh3_after_step1 := step1_reaction nh4cl na2co3
  let nh3_after_step2 := step2_reaction koh h3po4
  nh3_after_step1 + nh3_after_step2 = 2 :=
by
  sorry

end net_moles_nh3_after_reactions_l503_503551


namespace probability_no_dessert_l503_503427

theorem probability_no_dessert (p_DC : ℝ) (p_D : ℝ) (p_C : ℝ) (p_AD : ℝ) (p_AC : ℝ) (p_ADC : ℝ)
                            (h1 : p_DC = 0.60) (h2 : p_D = 0.15) (h3 : p_C = 0.10) 
                            (h4 : p_AD = 0.05) (h5 : p_AC = 0.08) (h6 : p_ADC = 0.03) :
  (p_C + p_AC + (1 - (p_DC + p_D + p_AD + p_ADC))) = 0.35 :=
by
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end probability_no_dessert_l503_503427


namespace cone_surface_area_l503_503308

theorem cone_surface_area (slant_height : ℝ) (lateral_surface_is_semicircle : bool) (radius : ℝ)
  (h1 : slant_height = 2)
  (h2 : lateral_surface_is_semicircle = true)
  (h3 : 2 * real.pi * radius = 2 * real.pi) :
  pi * radius * slant_height + pi * radius ^ 2 = 3 * pi :=
by {
  -- Extract radius from the given relationship
  have r_eq : radius = 1, from (eq_of_mul_eq_mul_left (ne_of_gt real.pi_pos) h3).mpr,
  -- Substitute the obtained radius into the surface area formula
  rw [r_eq, h1],
  -- Simplify the surface area formula
  ring
}

end cone_surface_area_l503_503308


namespace fibonacci_determinant_identity_fibonacci_1020_1022_1021_l503_503166

/-- Condition: Matrix identity for Fibonacci numbers -/
def fibo_matrix_identity (n : ℕ) : matrix (fin 2) (fin 2) ℤ :=
  ![(F (n+1), F n), (F n, F (n-1))]

/-- Condition: Determinant property -/
theorem fibonacci_determinant_identity (n : ℕ) :
  (F (n + 1) * F (n - 1) - F n * F n) = (-1)^n :=
sorry

/-- Proof of the main problem -/
theorem fibonacci_1020_1022_1021 :
  F 1020 * F 1022 - F 1021^2 = -1 :=
by
  have fib_det := fibonacci_determinant_identity 1021
  sorry

end fibonacci_determinant_identity_fibonacci_1020_1022_1021_l503_503166


namespace three_digit_number_constraint_l503_503447

theorem three_digit_number_constraint (B : ℕ) (h1 : 30 ≤ B ∧ B < 40) (h2 : (330 + B) % 3 = 0) (h3 : (330 + B) % 7 = 0) : B = 6 :=
sorry

end three_digit_number_constraint_l503_503447


namespace speed_of_A_l503_503142

theorem speed_of_A (V_B : ℝ) (h_VB : V_B = 4.555555555555555)
  (h_B_overtakes: ∀ (t_A t_B : ℝ), t_A = t_B + 0.5 → t_B = 1.8) 
  : ∃ V_A : ℝ, V_A = 3.57 :=
by
  sorry

end speed_of_A_l503_503142


namespace sequence_property_l503_503599

noncomputable theory

-- Define the sequence and the given condition 
def sequence (n : ℕ) : ℝ := sorry  -- placeholder for a_n sequence

-- Define the partial sums
def partial_sum (n : ℕ) : ℝ := ∑ i in finset.range n, sequence i

-- Define S_n in terms of a_n
def S (n : ℕ) : ℝ := 0.5 * (sequence n + 1 / sequence n)

-- Equivalent proof problem in Lean
theorem sequence_property :
  (sequence 1 = 1) ∧ 
  (sequence 2 = real.sqrt 2 - 1) ∧ 
  (sequence 3 = real.sqrt 3 - real.sqrt 2) ∧ 
  (∀ n : ℕ+, sequence n = real.sqrt (n : ℝ) - real.sqrt (n - 1)) :=
sorry

end sequence_property_l503_503599


namespace interior_angles_sum_l503_503792

theorem interior_angles_sum (h : ∀ (n : ℕ), n = 360 / 20) : 
  180 * (h 18 - 2) = 2880 :=
by
  sorry

end interior_angles_sum_l503_503792


namespace log_difference_l503_503190

theorem log_difference : 
  log 3 243 - log 3 (1 / 27) = 8 := by sorry

end log_difference_l503_503190


namespace car_robot_collections_l503_503839

variable (t m b s j : ℕ)

axiom tom_has_15 : t = 15
axiom michael_robots : m = 3 * t - 5
axiom bob_robots : b = 8 * (t + m)
axiom sarah_robots : s = b / 2 - 7
axiom jane_robots : j = (s - t) / 3

theorem car_robot_collections :
  t = 15 ∧
  m = 40 ∧
  b = 440 ∧
  s = 213 ∧
  j = 66 :=
  by
    sorry

end car_robot_collections_l503_503839


namespace part1_part2_l503_503004

-- Define the cooling formula
def cooling (θ θ1 k t : ℝ) : ℝ :=
  θ + (θ1 - θ) * Real.exp (-k * t)

-- Part 1: Prove k = 0.007 given the conditions.
theorem part1 (θ θ1 θ_t : ℝ) (k t : ℝ)
  (h_θ : θ = 20)
  (h_θ1 : θ1 = 98)
  (h_θ_t : θ_t = 71.2)
  (h_t : t = 60):
  k = 0.007 :=
by {
  sorry
}

-- Part 2: Prove room temperature θ = 22.8 given the conditions.
theorem part2 (θ1 θ_t θ : ℝ) (k t : ℝ)
  (h_k : k = 0.01)
  (h_θ1 : θ1 = 100)
  (h_θ_t : θ_t = 40)
  (h_t : t = 150):
  θ = 22.8 :=
by {
  sorry
}

end part1_part2_l503_503004


namespace groom_age_proof_l503_503060

theorem groom_age_proof (G B : ℕ) (h1 : B = G + 19) (h2 : G + B = 185) : G = 83 :=
by
  sorry

end groom_age_proof_l503_503060


namespace ratio_of_speeds_l503_503457

theorem ratio_of_speeds (v1 v2 : ℝ) (h1 : v1 > v2) (h2 : 8 = (v1 + v2) * 2) (h3 : 8 = (v1 - v2) * 4) : v1 / v2 = 3 :=
by
  sorry

end ratio_of_speeds_l503_503457


namespace tin_amount_new_alloy_l503_503121

/-
Given:
  1. 90 kg of alloy A is mixed with 140 kg of alloy B.
  2. Alloy A has lead and tin in the ratio 3 : 4.
  3. Alloy B has tin and copper in the ratio 2 : 5.

Prove:
  The amount of tin in the new alloy is approximately 91.43 kg.
-/

theorem tin_amount_new_alloy : 
  let alloy_A_weight := 90
      alloy_B_weight := 140
      ratio_lead_tin_A := (3, 4)
      ratio_tin_copper_B := (2, 5)
      total_tin := ((4 / 7) * alloy_A_weight + (2 / 7) * alloy_B_weight) in
  abs (total_tin - 91.43) < 0.01 := 
by 
  admit

end tin_amount_new_alloy_l503_503121


namespace tan_2alpha_eq_24_over_7_l503_503607

theorem tan_2alpha_eq_24_over_7
  (α : ℝ)
  (h1 : sin α + cos α = 1 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  tan (2 * α) = 24 / 7 := sorry

end tan_2alpha_eq_24_over_7_l503_503607


namespace smallest_number_conditions_l503_503865

theorem smallest_number_conditions :
  ∃ b : ℕ, 
    (b % 3 = 2) ∧ 
    (b % 4 = 2) ∧
    (b % 5 = 3) ∧
    (∀ b' : ℕ, 
      (b' % 3 = 2) ∧ 
      (b' % 4 = 2) ∧
      (b' % 5 = 3) → b ≤ b') :=
begin
  use 38,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros b' hb',
    have h3 := (hb'.left),
    have h4 := (hb'.right.left),
    have h5 := (hb'.right.right),
    -- The raw machinery for showing that 38 is the smallest may require more definition
    sorry
  }
end

end smallest_number_conditions_l503_503865


namespace distance_between_foci_l503_503037

-- Define the hyperbola as a set of points satisfying xy = 1
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the distance between two points in ℝ²
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

-- Define the statement asserting the distance between the foci of the hyperbola xy = 1 
theorem distance_between_foci (t : ℝ) (ht : 0 < t) :
  distance (t, t) (-t, -t) = 4 :=
sorry

end distance_between_foci_l503_503037


namespace value_of_expression_l503_503097

theorem value_of_expression : 3 - (-3)^{-3} = 82 / 27 := by
  sorry

end value_of_expression_l503_503097


namespace sequence_formula_minimum_m_l503_503997

variable (a_n : ℕ → ℕ) (S_n : ℕ → ℕ)

/-- The sequence a_n with sum of its first n terms S_n, the first term a_1 = 1, and the terms
   1, a_n, S_n forming an arithmetic sequence, satisfies a_n = 2^(n-1). -/
theorem sequence_formula (h1 : a_n 1 = 1)
    (h2 : ∀ n : ℕ, 1 + n * (a_n n - 1) = S_n n) :
    ∀ n : ℕ, a_n n = 2 ^ (n - 1) := by
  sorry

/-- T_n being the sum of the sequence {n / a_n}, if T_n < (m - 4) / 3 for all n in ℕ*, 
    then the minimum value of m is 16. -/
theorem minimum_m (T_n : ℕ → ℝ) (m : ℕ)
    (hT : ∀ n : ℕ, n > 0 → T_n n < (m - 4) / 3) :
    m ≥ 16 := by
  sorry

end sequence_formula_minimum_m_l503_503997


namespace find_minimal_positive_n_l503_503694

-- Define the arithmetic sequence
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the conditions
variables (a1 d : ℤ)
axiom condition_1 : arithmetic_seq a1 d 11 / arithmetic_seq a1 d 10 < -1
axiom condition_2 : ∃ n : ℕ, ∀ k : ℕ, k ≤ n → sum_arithmetic_seq a1 d k ≤ sum_arithmetic_seq a1 d n

-- Prove the statement
theorem find_minimal_positive_n : ∃ n : ℕ, n = 19 ∧ sum_arithmetic_seq a1 d n = 0 ∧
  (∀ m : ℕ, 0 < sum_arithmetic_seq a1 d m ∧ sum_arithmetic_seq a1 d m < sum_arithmetic_seq a1 d n) :=
sorry

end find_minimal_positive_n_l503_503694


namespace proof_inequality_equality_condition_l503_503991

theorem proof_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 2) :
  3 * x + 8 * x * y + 16 * x * y * z ≤ 12 := sorry

theorem equality_condition :
  ∃ (x y z : ℝ), x = 1 ∧ y = 3/4 ∧ z = 1/4 ∧ 3 * x + 8 * x * y + 16 * x * y * z = 12 := 
begin
  use [1, 3/4, 1/4],
  split,
  { refl },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end proof_inequality_equality_condition_l503_503991


namespace milk_level_lowered_approx_6_0168_inches_l503_503428

noncomputable def milkLoweringInches := 
  let box_length : ℝ := 58
  let box_width : ℝ := 25
  let gallons : ℝ := 5437.5
  let gallon_to_cubic_feet : ℝ := 0.133681
  let cubic_feet : ℝ := gallons * gallon_to_cubic_feet
  let base_area : ℝ := box_length * box_width
  let height_feet : ℝ := cubic_feet / base_area
  let height_inches : ℝ := height_feet * 12
  height_inches

theorem milk_level_lowered_approx_6_0168_inches :
  milkLoweringInches ≈ 6.0168 := 
  sorry

end milk_level_lowered_approx_6_0168_inches_l503_503428


namespace probability_two_red_chips_l503_503768

def total_chips := 6
def red_chips := 4
def white_chips := 2
def draw_chips := 3
def probability_red := 2 / 5

theorem probability_two_red_chips (Ethan: {...} chip_sets: List (List Chip)) :
  (* Assuming chip_sets represent all possible configurations of chips in the urns *)
  ∀ (distribution_1 distribution_2 : set Chip), 
  set.card (distribution_1 ∪ distribution_2) = total_chips ∧
  set.card (distribution_1) = draw_chips ∧
  set.card (distribution_2) = draw_chips ∧
  list.count chip_sets.get_red distribution_1 = 2 ->
  list.count chip_sets.get_red distribution_2 = 2 ->
  calculate_probability chip_sets = probability_red :=
 sorry

end probability_two_red_chips_l503_503768


namespace sum_of_numbers_l503_503888

theorem sum_of_numbers : 3 + 33 + 333 + 33.3 = 402.3 :=
  by
    sorry

end sum_of_numbers_l503_503888


namespace area_of_intersection_is_correct_l503_503849

-- Define the radius and centers of the circles
def radius := 3
def center1 := (3 : ℝ, 0 : ℝ)
def center2 := (0 : ℝ, 3 : ℝ)

-- Define the equations of the circles
def circle1 (x y : ℝ) : Prop := (x - center1.1)^2 + y^2 = radius^2
def circle2 (x y : ℝ) : Prop := x^2 + (y - center2.2)^2 = radius^2

-- Define the intersection points
def intersection_points : set (ℝ × ℝ) := {p | circle1 p.1 p.2 ∧ circle2 p.1 p.2}

-- Theorem to prove the area of intersection of two circles
theorem area_of_intersection_is_correct : 
  let area_of_intersection := 9 * (Real.pi / 2 - 1) in
  area_of_intersection = 9 * (Real.pi / 2 - 1) :=
by
  sorry

end area_of_intersection_is_correct_l503_503849


namespace ellipse_standard_equation_max_area_triangle_OPQ_l503_503623

theorem ellipse_standard_equation (a b : ℝ) (P : ℝ × ℝ)
  (ha : a > b) (hb : b > 0) (hP : P = (1, sqrt 3 / 2))
  (he : (sqrt (a^2 - b^2)) / a = sqrt 3 / 2) :
  (a = 2 ∧ b = 1) → (∀ (x y : ℝ), (x, y) ∈ set_of (λ p, (p.1 / a)^2 + (p.2 / b)^2 = 1) → 
     y^2 = 1 - (x^2 / 4)) :=
begin
  sorry
end

theorem max_area_triangle_OPQ (k : ℝ) (l : ℝ → ℝ)
  (hl : ∀ x, l x = k * x - 2)
  (hE : l 0 = -2)
  (intersection : ∀ x, (x, l x) ∈ ({p | (p.1 / 2)^2 + p.2^2 = 1}))
  (hm : ∃ k, k ≠ 0 ∧ sqrt (4 * k^2 - 3) > 0) :
  ∃ A : ℝ, A = 1 :=
begin
  sorry
end

end ellipse_standard_equation_max_area_triangle_OPQ_l503_503623


namespace audio_distribution_l503_503927

theorem audio_distribution 
  (total_audio : ℕ)
  (disc_capacity : ℕ)
  (min_discs_needed : ℕ)
  (uniform_audio : ℝ) :
  total_audio = 620 ∧ 
  disc_capacity = 70 ∧ 
  min_discs_needed = 9 ∧
  uniform_audio = total_audio / min_discs_needed 
  → 
  uniform_audio ≈ 68.89 :=
by
  sorry

end audio_distribution_l503_503927


namespace sum_T_19_34_51_l503_503537

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2 : ℕ) else (n + 1) / 2

def T (n : ℕ) : ℤ :=
  2 + S n

theorem sum_T_19_34_51 : T 19 + T 34 + T 51 = 25 := 
by
  -- Add the steps here
  sorry

end sum_T_19_34_51_l503_503537


namespace minimum_possible_value_of_Box_l503_503292

theorem minimum_possible_value_of_Box : 
  ∃ (a b Box : ℤ), 
    (a ≠ b) ∧ (a ≠ Box) ∧ (b ≠ Box) ∧
    (a * b = 15) ∧ 
    (∀ x : ℤ, (a * x + b) * (b * x + a) = 15 * x ^ 2 + Box * x + 15) ∧ 
    (∃ p q : ℤ, (p * q = 15 ∧ p ≠ q ∧ p ≠ 34 ∧ q ≠ 34) → (Box = p^2 + q^2)) ∧ 
    Box = 34 :=
by
  sorry

end minimum_possible_value_of_Box_l503_503292


namespace slope_of_tangent_at_A_l503_503443

theorem slope_of_tangent_at_A :
  let y := λ x : ℝ, x^2 + 3 * x
  let y' := deriv y
  y' 2 = 7 :=
by
  intro y y' -- introduce the definitions
  have h1 : y = λ x : ℝ, x^2 + 3 * x := rfl
  have h2 : y' = deriv y := rfl
  sorry

end slope_of_tangent_at_A_l503_503443


namespace number_of_cars_washed_l503_503357

theorem number_of_cars_washed (cars trucks suvs total raised_per_car raised_per_truck raised_per_suv : ℕ)
  (hc : cars = 5)
  (ht : trucks = 5)
  (ha : cars + trucks + suvs = total)
  (h_cost_car : raised_per_car = 5)
  (h_cost_truck : raised_per_truck = 6)
  (h_cost_suv : raised_per_suv = 7)
  (h_amount_total : total = 100)
  (h_raised_trucks : trucks * raised_per_truck = 30)
  (h_raised_suvs : suvs * raised_per_suv = 35) :
  suvs + trucks + cars = 7 :=
by
  sorry

end number_of_cars_washed_l503_503357


namespace parabola_distance_l503_503399

noncomputable def focus := (0, 1) -- Focus of the parabola x^2 = 4y is at (0,1)

theorem parabola_distance 
  (P : ℝ × ℝ) 
  (hP : ∃ x, P = (x, (x^2)/4)) 
  (h_dist_focus : dist P focus = 8) : 
  abs (P.snd) = 7 := sorry

end parabola_distance_l503_503399


namespace mia_socks_l503_503388

-- Defining the number of each type of socks
variables {a b c : ℕ}

-- Conditions and constraints
def total_pairs (a b c : ℕ) : Prop := a + b + c = 15
def total_cost (a b c : ℕ) : Prop := 2 * a + 3 * b + 5 * c = 35
def at_least_one (a b c : ℕ) : Prop := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- Main theorem to prove the number of 2-dollar pairs of socks
theorem mia_socks : 
  ∀ (a b c : ℕ), 
  total_pairs a b c → 
  total_cost a b c → 
  at_least_one a b c → 
  a = 12 :=
by
  sorry

end mia_socks_l503_503388


namespace ratio_proof_l503_503504

theorem ratio_proof :
  ∃ (x y z w : ℕ), (15 : x) ∧ (y ÷ 8 = 3 ÷ 4) ∧ (3 ÷ 4 = 0.75) ∧ (0.75 = w / 100) ∧ (x = 20) ∧ (y = 6) ∧ (w = 75) :=
  sorry

end ratio_proof_l503_503504


namespace relatively_prime_days_in_april_l503_503522

-- Define a function to check if two numbers are relatively prime
def are_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the month number and days in April
def month_number := 4
def total_days := 31

-- Define the set of days in April
def days_in_april : List ℕ := List.range (total_days + 1)

-- Filter the days that are relatively prime to the month number
def rel_prime_days := List.filter (λ day, are_rel_prime day month_number) days_in_april

-- Assert that the number of relatively prime days is 16
theorem relatively_prime_days_in_april : List.length rel_prime_days = 16 := by
  sorry

end relatively_prime_days_in_april_l503_503522


namespace fraction_of_smart_integers_divisible_by_27_l503_503721

def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum
def is_smart_integer (n : ℕ) : Prop :=
  is_even n ∧ 20 < n ∧ n < 120 ∧ sum_of_digits n = 10

theorem fraction_of_smart_integers_divisible_by_27 : 
  (∃ k : ℕ, k = 0) / (∑ n in (finset.filter is_smart_integer (finset.range 120)), 1) = 0 :=
by
  sorry

end fraction_of_smart_integers_divisible_by_27_l503_503721


namespace ninth_term_arithmetic_sequence_l503_503445

variable (a d : ℕ)

def arithmetic_sequence_sum (a d : ℕ) : ℕ :=
  a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d)

theorem ninth_term_arithmetic_sequence (h1 : arithmetic_sequence_sum a d = 21) (h2 : a + 6 * d = 7) : a + 8 * d = 9 :=
by
  sorry

end ninth_term_arithmetic_sequence_l503_503445


namespace Lars_bake_total_breads_l503_503352

theorem Lars_bake_total_breads :
  (Lars_bakes_10_loaves_per_hour : ℕ → ℕ) → (Lars_bakes_30_baguettes_per_2_hours : ℕ → ℕ) →
  (Lars_bakes_for_6_hours : ℕ) →
  (total_breads_in_a_day : ℕ) :=
by
  let Lars_bakes_10_loaves_per_hour := (n : ℕ) → 10 * n
  let Lars_bakes_30_baguettes_per_2_hours := (n : ℕ) → 30 * n
  let Lars_bakes_for_6_hours := 6
  let total_loaves := Lars_bakes_10_loaves_per_hour Lars_bakes_for_6_hours
  let total_baguettes := Lars_bakes_30_baguettes_per_2_hours (Lars_bakes_for_6_hours / 2)
  let total_breads_in_a_day := total_loaves + total_baguettes
  have : total_breads_in_a_day = 150 := by
    sorry
  exact total_breads_in_a_day

end Lars_bake_total_breads_l503_503352


namespace tan_2x_is_odd_l503_503630

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem tan_2x_is_odd :
  is_odd f :=
by
  sorry

end tan_2x_is_odd_l503_503630


namespace t_range_l503_503956

noncomputable def f : ℝ → ℝ :=
λ x, if 0 ≤ x ∧ x < 1 then x^2 - x
     else if 1 ≤ x ∧ x < 2 then -((1/2)^abs(x - 3/2))
     else if x < 0 then 2^(⌊x/2⌋ + 1) * f (x - 2 * ⌊x/2⌋ - 2)
     else f (x - 2)

theorem t_range (t : ℝ) : (∀ x, -4 ≤ x ∧ x < -2 → f x ≥ t^2 / 4 - t + 1 / 2) →
 1 ≤ t ∧ t ≤ 3 :=
begin
  sorry
end

end t_range_l503_503956


namespace sum_of_interior_angles_l503_503785

theorem sum_of_interior_angles (h : ∀ (n : ℕ), 360 / 20 = n) : 
  ∃ (s : ℕ), s = 2880 :=
by
  have n := 360 / 20
  have sum := 180 * (n - 2)
  use sum
  sorry

end sum_of_interior_angles_l503_503785


namespace distance_to_school_l503_503719

/-- Jeremy's father usually drives him to school in rush hour traffic in 30 minutes.
    One exceptional day when traffic is unusually light, he is able to drive at a speed 15 miles per hour faster,
    therefore reaching the school in just 18 minutes. 
    Prove that the distance to the school is 11.25 miles under these conditions. -/
theorem distance_to_school
  (d : ℕ) -- distance in miles
  (v : ℕ) -- usual speed in miles per hour
  (h1 : 30 * v = 60 * d)
  (h2 : 18 * (v + 15) = 60 * d) :
  d = 11.25 := 
  sorry

end distance_to_school_l503_503719


namespace most_likely_outcome_l503_503323

-- Define the probabilities
def prob_girl : ℚ := 3/5
def prob_boy : ℚ := 2/5

-- Compute the probability for exactly 2 girls and 1 boy
def prob_2_girls_1_boy : ℚ := (3.choose 2) * (prob_girl^2) * (prob_boy^1)

-- Define other probability outcomes for reference
def prob_all_boys : ℚ := prob_boy^3
def prob_all_girls : ℚ := prob_girl^3
def prob_1_girl_2_boys : ℚ := (3.choose 1) * (prob_girl^1) * (prob_boy^2)

-- Theorem to state the answer
theorem most_likely_outcome :
  prob_2_girls_1_boy > prob_all_boys ∧
  prob_2_girls_1_boy > prob_all_girls ∧
  prob_2_girls_1_boy > prob_1_girl_2_boys :=
begin
  sorry
end

end most_likely_outcome_l503_503323


namespace two_digit_prime_number_count_is_seven_l503_503671

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_prime_count : ℕ :=
  (∑ tens in {1, 2, 5, 7, 8, 9},
    ∑ units in {1, 2, 5, 7, 8, 9}.erase tens,
      if is_prime (10 * tens + units) then 1 else 0)

theorem two_digit_prime_number_count_is_seven : two_digit_prime_count = 7 :=
by sorry

end two_digit_prime_number_count_is_seven_l503_503671


namespace equation_1_solution_set_equation_2_solution_set_l503_503019

open Real

theorem equation_1_solution_set (x : ℝ) : x^2 - 4 * x - 8 = 0 ↔ (x = 2 * sqrt 3 + 2 ∨ x = -2 * sqrt 3 + 2) :=
by sorry

theorem equation_2_solution_set (x : ℝ) : 3 * x - 6 = x * (x - 2) ↔ (x = 2 ∨ x = 3) :=
by sorry

end equation_1_solution_set_equation_2_solution_set_l503_503019


namespace two_digit_count_four_digit_count_l503_503887

-- Define the sets of digits
def digits := {1, 2, 3, 4}

-- Statement for the number of two-digit numbers
theorem two_digit_count : (digits.card = 4) → ∃ n, n = 12 :=
by {
  intros,
  use 12,
  sorry
}

-- Statement for the number of four-digit numbers without repetition
theorem four_digit_count : (digits.card = 4) → ∃ n, n = 24 :=
by {
  intros,
  use 24,
  sorry
}

end two_digit_count_four_digit_count_l503_503887


namespace vitya_convinced_of_12_models_l503_503077

noncomputable def min_offers_needed (n : ℕ) (k : ℕ) : ℕ :=
  if h : n = 13 then
    let ln100 := Real.log 100
    let ln13 := Real.log 13
    let ln12 := Real.log 12
    let req_k := Real.log 100 / (Real.log 13 - Real.log 12)
    if k > req_k then k else req_k.toNat + 1
  else k

theorem vitya_convinced_of_12_models (k : ℕ) : ∀ n, (n >= 13) → (min_offers_needed n k > 58) :=
by
  intros n h
  apply sorry

end vitya_convinced_of_12_models_l503_503077


namespace inequality_division_by_two_l503_503298

theorem inequality_division_by_two (x y : ℝ) (h : x > y) : (x / 2) > (y / 2) := 
sorry

end inequality_division_by_two_l503_503298


namespace max_positive_numbers_in_filled_table_l503_503122

theorem max_positive_numbers_in_filled_table :
  ∀ (table : ℕ × ℕ), table = (7, 7) →
  ∀ (values : ℕ → ℕ → ℤ), 
  (∀ i j, (i = 0 ∨ i = 6 ∨ j = 0 ∨ j = 6) → values i j <  0) →
  (∀ i j, (i > 0 ∧ i < 6 ∧ j > 0 ∧ j < 6) → 
  (∃ (row: ℕ), values i j = (values row j * values i (j-1) ∨ values row j * values i (j+1)) ∨ 
                  ∃ (col: ℕ), values i j = (values (i-1) col * values (i+1) col))) →
  ∃ count, count = 24 ∧ count = ∑ i in (finset.range 7), ∑ j in (finset.range 7), if values i j > 0 then 1 else 0 .

end max_positive_numbers_in_filled_table_l503_503122


namespace cone_surface_area_eq_3pi_l503_503311

noncomputable def coneSurfaceArea (r l : ℝ) : ℝ :=
  let lateralSurface := π * r * l
  let baseSurface := π * r^2
  baseSurface + lateralSurface

theorem cone_surface_area_eq_3pi (r : ℝ) (h : r = 1) : coneSurfaceArea r 2 = 3 * π := by
  sorry

end cone_surface_area_eq_3pi_l503_503311


namespace candy_bar_price_increase_l503_503128

theorem candy_bar_price_increase (W P : ℝ) (hW : W > 0) (hP : P > 0) :
  let original_price_per_ounce := P / W,
      new_weight := 0.60 * W,
      new_price_per_ounce := P / new_weight,
      percent_increase := ((new_price_per_ounce - original_price_per_ounce) / original_price_per_ounce) * 100
  in percent_increase = 200 / 3 := by 
  sorry

end candy_bar_price_increase_l503_503128


namespace part_a_part_b_l503_503707

-- Definitions to use in proof problems
def plane : Type := ℝ × ℝ
def line (p : plane) (q : plane) : Type := { r : plane // ∃ a b : ℝ, r = (a * p.1 + b * q.1, a * p.2 + b * q.2) }
def lines (n : ℕ) : Type := Vector (line ((0, 0) : plane) ((1, 1) : plane)) n

def unique_pairs (n : ℕ) := (finset.range n).pow 2

def num_intersections (l : lines 6) : ℕ :=
  (unique_pairs 6).card - ∑ p in unique_pairs 6, if (∃ r, p = (l.1, l.2)) then 1 else 0

-- Proof statements
theorem part_a (l : lines 6) (H : ∀ p q r : line ((0, 0) : plane) ((1, 1) : plane), p ≠ q ∧ p ≠ r ∧ q ≠ r): 
  ∃ l : lines 6, num_intersections l = 12 := sorry

theorem part_b (l : lines 6) (H : ∀ p q r : line ((0, 0) : plane) ((1, 1) : plane), p ≠ q ∧ p ≠ r ∧ q ≠ r):
  ¬ (∃ l : lines 6, num_intersections l = 16) := sorry

end part_a_part_b_l503_503707


namespace john_can_buy_10_packets_of_corn_chips_l503_503422

theorem john_can_buy_10_packets_of_corn_chips:
  (∀ (chip_cost corn_chip_cost : ℝ) (number_of_chips budget : ℝ),
    chip_cost = 2 ∧ corn_chip_cost = 1.5 ∧ number_of_chips = 15 ∧ budget = 45 →
    (budget - number_of_chips * chip_cost) / corn_chip_cost = 10) :=
by
  intros chip_cost corn_chip_cost number_of_chips budget h
  cases h with h1 h' 
  cases h' with h2 h'' 
  cases h'' with h3 h4 
  rw [h1, h2, h3, h4]
  simp
  norm_num
  sorry

end john_can_buy_10_packets_of_corn_chips_l503_503422


namespace angle_theta_value_l503_503076

theorem angle_theta_value (A B C D E F G : Type) -- Vertices in the zigzag drawing
  (angleACB : ℝ) (angleFEG : ℝ) (angleDCE : ℝ) (angleDEC : ℝ) 
  (sum_tri_CDE : angleDEC + angleDCE = 180) : 
  angle DEC = 83 ∧ angleDCE = 86 → angleDEC + angleDCE - 180 = 11 :=
by
  sorry

end angle_theta_value_l503_503076


namespace domain_of_function_l503_503780

theorem domain_of_function :
  (∀ x, x ∈ (1, 2) ↔ (x^2 - 1 > 0 ∧ -x^2 + x + 2 > 0)) :=
by sorry

end domain_of_function_l503_503780


namespace sum_of_squares_of_roots_l503_503051

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) (h₀ : polynomial.eval x₁ (5 * polynomial.X^2 + 20 * polynomial.X - 25) = 0)
    (h₁ : polynomial.eval x₂ (5 * polynomial.X^2 + 20 * polynomial.X - 25) = 0) :
    x₁^2 + x₂^2 = 26 :=
sorry

end sum_of_squares_of_roots_l503_503051


namespace locus_of_C_on_union_circles_l503_503726

-- Define the points A and B as fixed points in the plane
variables {A B : Point}

-- Define the condition that the altitude from C to AB is equal to b
variables {C : Point} (h_b : Real) [Altitude (Triangle A B C) C h_b] (h_b_eq_b : h_b = b)

-- Define the circles S1 and S2 as the images of circle with diameter AB
def circle_with_diameter (X Y : Point) : Circle := sorry -- Placeholder for the actual circle definition

def S1 : Circle := rotate_by_90_degrees (circle_with_diameter A B) A
def S2 : Circle := rotate_by_minus_90_degrees (circle_with_diameter A B) A

-- The locus of C lies on the union of the circle S1 and S2
theorem locus_of_C_on_union_circles :
  (C ∈ S1 ∪ S2) :=
sorry

end locus_of_C_on_union_circles_l503_503726


namespace range_of_f2_l503_503613

noncomputable theory

variables {a b c x1 x2 : ℝ}

/-- Define the cubic function f(x) -/
def f (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

/-- Given conditions -/
def odd_func_cond (g : ℝ → ℝ) [function.odd g] : Prop := ∀ x, g (x + 1) = - g (-(x + 1))
def root_cond : Prop := f 1 = 0 ∧ f x1 = 0 ∧ f x2 = 0 ∧ 0 < x1 ∧ x1 < x2
def f_x_plus_1_odd : Prop := odd_func_cond f

/-- The main theorem -/
theorem range_of_f2 (h1 : root_cond) (h2 : f_x_plus_1_odd) : Ioo 0 1 :=
  sorry

end range_of_f2_l503_503613


namespace arithmetic_seq_b3_b6_l503_503370

theorem arithmetic_seq_b3_b6 (b : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, b n = b 1 + n * d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_b4_b5 : b 4 * b 5 = 30) :
  b 3 * b 6 = 28 := 
sorry

end arithmetic_seq_b3_b6_l503_503370


namespace jessie_final_position_l503_503000

theorem jessie_final_position :
  ∃ y : ℕ,
  (0 + 6 * 4 = 24) ∧
  (y = 24) :=
by
  sorry

end jessie_final_position_l503_503000


namespace find_common_ratio_l503_503222

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_seq a q)
  (h2 : tendsto (λ n, ∑ i in range(n), a (4 * i)) at_top (𝓝 4))
  (h3 : tendsto (λ n, ∑ i in range(n), (a i) ^ 2) at_top (𝓝 8)) :
  q = 1 / 3 :=
sorry

end find_common_ratio_l503_503222


namespace median_first_fifteen_positive_integers_l503_503463

-- Define the list of the first fifteen positive integers
def first_fifteen_positive_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

-- Define the property that the median of the list is 8.0
theorem median_first_fifteen_positive_integers : median(first_fifteen_positive_integers) = 8.0 := 
sorry

end median_first_fifteen_positive_integers_l503_503463


namespace ellipse_and_triangle_area_l503_503256

-- Define the conditions
def is_ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def minor_axis_length (b : ℝ) : Prop :=
  b = 1

def eccentricity (a c : ℝ) : Prop :=
  c = sqrt (a^2 - 1) ∧ c / a = sqrt 6 / 3

def line_through_point (m : ℝ) : ℝ → ℝ :=
  λ y, m * y - 1

def line_intersects_ellipse (l : ℝ → ℝ) (E : ℝ → ℝ → Prop) : Prop :=
  ∀ y, E (l y) y

-- Define the problem
theorem ellipse_and_triangle_area (a b : ℝ) (c : ℝ) (l : ℝ → ℝ) :
  is_ellipse a b ∧ minor_axis_length b ∧ eccentricity a c →
  (∀ x y : ℝ, x^2 / 3 + y^2 = 1) ∧ (∃ (A B : ℝ × ℝ), True) → -- Definition to be expanded as necessary
  (l = line_through_point 0) ∧
  True :=  -- ─ Adding this dummy condition for the skipped proof area computation
sorry

end ellipse_and_triangle_area_l503_503256


namespace smallest_k_no_real_roots_l503_503489

theorem smallest_k_no_real_roots :
  let f (k x : ℝ) := 3 * x * (k * x - 5) - x^2 + 8 - x^3 in
  ∀ (x : ℝ), ¬ ∃ k : ℤ, f k x = 0 ∧ k = 1 :=
by
  sorry

end smallest_k_no_real_roots_l503_503489


namespace number_of_triangles_in_regular_decagon_l503_503659

noncomputable def number_of_triangles_in_decagon : ℕ :=
∑ i in (finset.range 10).powerset_len 3, 1

theorem number_of_triangles_in_regular_decagon :
  number_of_triangles_in_decagon = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l503_503659


namespace number_of_cars_washed_l503_503356

theorem number_of_cars_washed (cars trucks suvs total raised_per_car raised_per_truck raised_per_suv : ℕ)
  (hc : cars = 5)
  (ht : trucks = 5)
  (ha : cars + trucks + suvs = total)
  (h_cost_car : raised_per_car = 5)
  (h_cost_truck : raised_per_truck = 6)
  (h_cost_suv : raised_per_suv = 7)
  (h_amount_total : total = 100)
  (h_raised_trucks : trucks * raised_per_truck = 30)
  (h_raised_suvs : suvs * raised_per_suv = 35) :
  suvs + trucks + cars = 7 :=
by
  sorry

end number_of_cars_washed_l503_503356


namespace inclination_angle_x_equals_3_is_90_l503_503039

-- Define the condition that line x = 3 is vertical
def is_vertical_line (x : ℝ) : Prop := x = 3

-- Define the inclination angle property for a vertical line
def inclination_angle_of_vertical_line_is_90 (x : ℝ) (h : is_vertical_line x) : ℝ :=
90   -- The angle is 90 degrees

-- Theorem statement to prove the inclination angle of the line x = 3 is 90 degrees
theorem inclination_angle_x_equals_3_is_90 :
  inclination_angle_of_vertical_line_is_90 3 (by simp [is_vertical_line]) = 90 :=
sorry  -- proof goes here


end inclination_angle_x_equals_3_is_90_l503_503039


namespace penelope_markup_rate_l503_503757

theorem penelope_markup_rate (S : ℝ) (hS : S = 10) :
  let C := 0.70 * S in
  let markup := (S - C) / C * 100 in
  markup = 42.86 :=
by
  let C := 0.70 * S
  let markup := (S - C) / C * 100
  have hS1 : S = 10 := hS
  -- Simplifying calculations and proving the final result
  sorry

end penelope_markup_rate_l503_503757


namespace probability_consecutive_majors_l503_503831

-- Define the conditions
def thirteen_people_at_table : ℕ := 13
def math_majors : ℕ := 6
def physics_majors : ℕ := 3
def biology_majors : ℕ := 4

-- Define the target probability
def target_probability : ℚ := 1 / 21952

theorem probability_consecutive_majors :
  thirteen_people_at_table = 13 →
  math_majors + physics_majors + biology_majors = 13 →
  math_majors = 6 →
  physics_majors = 3 →
  biology_majors = 4 →
  (∀ seats : ℕ, ∃ ways : ℚ, (ways = target_probability)) :=
by
  intros h1 h2 h3 h4 h5
  existsi target_probability
  sorry

end probability_consecutive_majors_l503_503831


namespace power_function_solution_l503_503637

theorem power_function_solution :
  ∃ a : ℝ, (∀ x, f x = x ^ a) ∧ f 2 = 8 ∧ (∃ x : ℝ, f x = 27 ∧ x = 3) := 
  by
  sorry

end power_function_solution_l503_503637


namespace vitya_needs_58_offers_l503_503083

noncomputable def smallest_integer_k (P : ℝ → ℝ) : ℝ :=
  if H : ∃ k, k > P (100), then classical.some H else 0

theorem vitya_needs_58_offers :
  ∀ n : ℕ, n ≥ 13 → 
  (12:ℝ/13:ℝ) ^ smallest_integer_k (fun x => Real.log x / (Real.log 13 - Real.log 12)) < 0.01 :=
begin
  assume n h,
  rw smallest_integer_k,
  split_ifs,
  { sorry }, -- proof would go here
  { exfalso, exact sorry }, -- no proof steps provided
end

end vitya_needs_58_offers_l503_503083


namespace smallest_number_l503_503866

theorem smallest_number (b : ℕ) :
  (b % 3 = 2) ∧ (b % 4 = 2) ∧ (b % 5 = 3) → b = 38 :=
by
  sorry

end smallest_number_l503_503866


namespace shaded_area_parallelogram_WXYZ_l503_503326

/-- 
In parallelogram WXYZ, WZ has length 14 units 
and the height from point X to side WZ is 10 units.
Point T is on WZ such that WT = 9 units.
The shaded region is TXZY.
The area of the shaded region TXZY is 95 square units.
 -/
theorem shaded_area_parallelogram_WXYZ
  (WZ_length : ℝ) (ht_WZ : ℝ) (WT_length : ℝ)
  (h_WZ_length : WZ_length = 14)
  (h_ht_WZ : ht_WZ = 10)
  (h_WT_length : WT_length = 9) :
  [WZ_length * ht_WZ - (1 / 2 * WT_length * ht_WZ)] = 95 := by
  sorry

end shaded_area_parallelogram_WXYZ_l503_503326


namespace polyhedron_edges_faces_vertices_l503_503384

theorem polyhedron_edges_faces_vertices
  (E F V n m : ℕ)
  (h1 : n * F = 2 * E)
  (h2 : m * V = 2 * E)
  (h3 : V + F = E + 2) :
  ¬(m * F = 2 * E) :=
sorry

end polyhedron_edges_faces_vertices_l503_503384


namespace find_f6_l503_503897

noncomputable def f (x : ℝ) : ℝ := 
  let u := 4 * x - 2 in (u^2 - 4 * u + 52) / 16

theorem find_f6 : 
  (∀ x : ℝ, f (4 * x - 2) = x^2 - 2 * x + 4) → f 6 = 4 :=
by
  intro h
  sorry

end find_f6_l503_503897


namespace outfits_count_l503_503412

theorem outfits_count (shirts ties : ℕ) (h_shirts : shirts = 7) (h_ties : ties = 6) : 
  (shirts * (ties + 1) = 49) :=
by
  sorry

end outfits_count_l503_503412


namespace normals_intersect_at_single_point_l503_503116

-- Definitions of points on the parabola and distinct condition
variables {a b c : ℝ}

-- Condition stating that A, B, C are distinct points
def distinct_points (a b c : ℝ) : Prop :=
  (a - b) ≠ 0 ∧ (b - c) ≠ 0 ∧ (c - a) ≠ 0

-- Statement to be proved
theorem normals_intersect_at_single_point (habc : distinct_points a b c) :
  a + b + c = 0 :=
sorry

end normals_intersect_at_single_point_l503_503116


namespace power_difference_divisible_by_35_l503_503759

theorem power_difference_divisible_by_35 (n : ℕ) : (3^(6*n) - 2^(6*n)) % 35 = 0 := 
by sorry

end power_difference_divisible_by_35_l503_503759


namespace multiplicative_inverse_137_391_l503_503549

theorem multiplicative_inverse_137_391 :
  ∃ (b : ℕ), (b ≤ 390) ∧ (137 * b) % 391 = 1 :=
sorry

end multiplicative_inverse_137_391_l503_503549


namespace sales_increased_most_in_2004_l503_503432

theorem sales_increased_most_in_2004 :
  let sales := [10, 12, 25, 27.5, 40]
  let years := [2000, 2002, 2004, 2006, 2008]
  let diffs := List.zipWith (· - ·) (List.tail sales) sales in
  List.maximum diffs = some (25 - 12) →
  years[(List.indexOf? diffs (some (25 - 12))).getD 0] = 2004 :=
by sorry

end sales_increased_most_in_2004_l503_503432


namespace price_of_orange_is_60_l503_503942

-- Given: 
-- 1. The price of each apple is 40 cents.
-- 2. Mary selects 10 pieces of fruit in total.
-- 3. The average price of these 10 pieces is 56 cents.
-- 4. Mary must put back 6 oranges so that the remaining average price is 50 cents.
-- Prove: The price of each orange is 60 cents.

theorem price_of_orange_is_60 (a o : ℕ) (x : ℕ) 
  (h1 : a + o = 10)
  (h2 : 40 * a + x * o = 560)
  (h3 : 40 * a + x * (o - 6) = 200) : 
  x = 60 :=
by
  have eq1 : 40 * a + x * o = 560 := h2
  have eq2 : 40 * a + x * (o - 6) = 200 := h3
  sorry

end price_of_orange_is_60_l503_503942


namespace large_posters_count_l503_503917

theorem large_posters_count (total_posters small_ratio medium_ratio : ℕ) (h_total : total_posters = 50) (h_small_ratio : small_ratio = 2/5) (h_medium_ratio : medium_ratio = 1/2) :
  let small_posters := (small_ratio * total_posters) in
  let medium_posters := (medium_ratio * total_posters) in
  let large_posters := total_posters - (small_posters + medium_posters) in
  large_posters = 5 := by
{
  sorry
}

end large_posters_count_l503_503917


namespace multiplication_in_P_l503_503387

-- Define the set P as described in the problem
def P := {x : ℕ | ∃ n : ℕ, x = n^2}

-- Prove that for all a, b in P, a * b is also in P
theorem multiplication_in_P {a b : ℕ} (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P :=
sorry

end multiplication_in_P_l503_503387


namespace roots_of_equation_l503_503805

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l503_503805


namespace rhombus_area_l503_503783

theorem rhombus_area : 
  let d1 d2 : ℝ := 
    let roots := ((21 + Real.sqrt(21 * 21 - 4 * 1 * 30)) / (2 * 1), (21 - Real.sqrt(21 * 21 - 4 * 1 * 30)) / (2 * 1)) in
    (roots.1, roots.2)
  in
  (1 / 2) * d1 * d2 = 15 :=
by
  sorry

end rhombus_area_l503_503783


namespace problem_1_problem_2_l503_503272

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^3 + b * x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x + g (4 * a^2 / 4) x

theorem problem_1 (a b : ℝ) (h1 : a > 0) (h2 : f a 1 = g b 1) 
  (h3 : deriv (f a) 1 = deriv (g b) 1) : 
  a = 3 ∧ b = 3 :=
sorry

theorem problem_2 (a : ℝ) (h4 : a > 0) (h5 : a^2 = 4 * (4 * a^2 / 4)) : 
  (∀ x, x < -a / 2 → deriv (h a) x < 0) ∧ 
  (∀ x, x > -a / 2 ∧ x < -a / 6 → deriv (h a) x > 0) ∧ 
  (∀ x, x > -a / 6 → deriv (h a) x < 0) ∧ 
  (∀ x, x ≤ -1 → h(a) x ≤ h(a) (-a / 2) ∧ x = -a / 2 → h(a) x = 1) :=
sorry

end problem_1_problem_2_l503_503272


namespace axis_of_symmetry_compare_m_n_range_t_max_t_l503_503335

-- Condition: Definition of the parabola
def parabola (t x : ℝ) := x^2 - 2 * t * x + 1

-- Problem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ (y x : ℝ), parabola t x = y -> x = t :=
sorry

-- Problem 2: Comparing m and n
theorem compare_m_n (t m n : ℝ) :
  parabola t (t - 2) = m ∧ parabola t (t + 3) = n -> n > m := 
sorry

-- Problem 3: Range of t for y₁ ≤ y₂
theorem range_t (t x₁ y₁ y₂ : ℝ) :
  -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ = y₁ ∧ parabola t 3 = y₂ -> y₁ ≤ y₂ → t ≤ 1 :=
sorry

-- Problem 4: Maximum t for y₁ ≥ y₂
theorem max_t (t y₁ y₂ : ℝ) :
  (parabola t (t + 1) = y₁ ∧ parabola t (2 * t - 4) = y₂) → y₁ ≥ y₂ → t ≤ 5 :=
sorry

end axis_of_symmetry_compare_m_n_range_t_max_t_l503_503335


namespace line_parallel_condition_l503_503729

theorem line_parallel_condition (a : ℝ) :
    (a = 1) → (∀ (x y : ℝ), (ax + 2 * y - 1 = 0) ∧ (x + (a + 1) * y + 4 = 0)) → (a = 1 ∨ a = -2) :=
by
sorry

end line_parallel_condition_l503_503729


namespace directrix_of_parabola_l503_503034

theorem directrix_of_parabola : (∀ x : ℝ, y = (1 / 4) * x^2) → directrix_eq : (y = -1) :=
by
  sorry

end directrix_of_parabola_l503_503034


namespace jam_cost_is_162_l503_503184

theorem jam_cost_is_162 (N B J : ℕ) (h1 : N > 1) (h2 : 4 * B + 6 * J = 39) (h3 : N = 9) : 
  6 * N * J = 162 := 
by sorry

end jam_cost_is_162_l503_503184


namespace smallest_even_number_of_seven_l503_503769

-- Conditions: The sum of seven consecutive even numbers is 700.
-- We need to prove that the smallest of these numbers is 94.

theorem smallest_even_number_of_seven (n : ℕ) (hn : 7 * n = 700) :
  ∃ (a b c d e f g : ℕ), 
  (2 * a + 4 * b + 6 * c + 8 * d + 10 * e + 12 * f + 14 * g = 700) ∧ 
  (a = b - 1) ∧ (b = c - 1) ∧ (c = d - 1) ∧ (d = e - 1) ∧ (e = f - 1) ∧ 
  (f = g - 1) ∧ (g = 100) ∧ (a = 94) :=
by
  -- This is the theorem statement. 
  sorry

end smallest_even_number_of_seven_l503_503769


namespace worst_player_is_son_l503_503932

-- Define the types of players and relationships
inductive Sex
| male
| female

structure Player where
  name : String
  sex : Sex
  age : Nat

-- Define the four players
def woman := Player.mk "woman" Sex.female 30  -- Age is arbitrary
def brother := Player.mk "brother" Sex.male 30
def son := Player.mk "son" Sex.male 10
def daughter := Player.mk "daughter" Sex.female 10

-- Define the conditions
def opposite_sex (p1 p2 : Player) : Prop := p1.sex ≠ p2.sex
def same_age (p1 p2 : Player) : Prop := p1.age = p2.age

-- Define the worst player and the best player
variable (worst_player : Player) (best_player : Player)

-- Conditions as hypotheses
axiom twin_condition : ∃ twin : Player, (twin ≠ worst_player) ∧ (opposite_sex twin best_player)
axiom age_condition : same_age worst_player best_player
axiom not_same_player : worst_player ≠ best_player

-- Prove that the worst player is the son
theorem worst_player_is_son : worst_player = son :=
by
  sorry

end worst_player_is_son_l503_503932


namespace median_first_fifteen_integers_l503_503469

theorem median_first_fifteen_integers :
  let l := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] in
  let seventh := l.nth 6 in
  let eighth := l.nth 7 in
  (seventh.is_some ∧ eighth.is_some) →
  (seventh.get_or_else 0 + eighth.get_or_else 0) / 2 = 7.5 :=
by
  sorry

end median_first_fifteen_integers_l503_503469


namespace fractional_inequality_l503_503364

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem fractional_inequality (x : ℝ) (hx : x > 1) (hnint : ¬(∃ (n : ℤ), x = n)) :
  ( ( (x + fractional_part x) / floor x ) - (floor x / (x + fractional_part x)) + 
    ( (x + floor x) / fractional_part x ) - (fractional_part x / (x + floor x)) ) 
  > 9 / 2 := 
by
  sorry

end fractional_inequality_l503_503364


namespace find_a_for_even_function_l503_503268

theorem find_a_for_even_function :
  (∀ x : ℝ, f x = x^2 * 2^x / (4^(1*x) + 1)) ↔ a = 1 := by
  let f : ℝ → ℝ := λ x, x^2 * 2^x / (4^(a*x) + 1)
  have even_function : ∀ x : ℝ, f(-x) = f(x) := sorry
  sorry

end find_a_for_even_function_l503_503268


namespace P7_eq_P1_l503_503739

noncomputable def P1 : Complex := sorry
noncomputable def P2 : Complex := sorry
noncomputable def P3 : Complex := sorry
noncomputable def P4 : Complex := sorry
noncomputable def P5 : Complex := sorry
noncomputable def P6 : Complex := sorry
noncomputable def P7 : Complex := sorry

-- Condition 1: Points P1, P2, P3, P4 are on the unit circle
def on_unit_circle (z : Complex) : Prop := Complex.abs z = 1
axiom P1_on_circle : on_unit_circle P1
axiom P2_on_circle : on_unit_circle P2
axiom P3_on_circle : on_unit_circle P3
axiom P4_on_circle : on_unit_circle P4

-- Condition 2: P4P5 is parallel to P1P2
axiom P4P5_parallel_P1P2 : ∀ (d e a b : Complex), (e = P5) → (P4 = d) → 
  (P1 = a) → (P2 = b) → ((d - e) / (a - b) = (Complex.conj d - Complex.conj e) / (Complex.conj a - Complex.conj b))

-- Condition 3: P5P6 is parallel to P2P3
axiom P5P6_parallel_P2P3 : ∀ (e f b c : Complex), (f = P6) → (P5 = e) → 
  (P2 = b) → (P3 = c) → ((e - f) / (b - c) = (Complex.conj e - Complex.conj f) / (Complex.conj b - Complex.conj c))

-- Condition 4: P6P7 is parallel to P3P4
axiom P6P7_parallel_P3P4 : ∀ (f g c d : Complex), (g = P7) → (P6 = f) → 
  (P3 = c) → (P4 = d) → ((f - g) / (c - d) = (Complex.conj f - Complex.conj g) / (Complex.conj c - Complex.conj d))

theorem P7_eq_P1 : P7 = P1 := sorry

end P7_eq_P1_l503_503739


namespace find_value_b_in_geometric_sequence_l503_503433

theorem find_value_b_in_geometric_sequence
  (b : ℝ)
  (h1 : 15 ≠ 0) -- to ensure division by zero does not occur
  (h2 : b ≠ 0)  -- to ensure division by zero does not occur
  (h3 : 15 * (b / 15) = b) -- 15 * r = b
  (h4 : b * (b / 15) = 45 / 4) -- b * r = 45 / 4
  : b = 15 * Real.sqrt 3 / 2 :=
sorry

end find_value_b_in_geometric_sequence_l503_503433


namespace sixth_root_68968845601_l503_503182

theorem sixth_root_68968845601 :
  ∃ x : ℕ, x^6 = 68968845601 ∧ x = 51 :=
by {
  existsi 51,
  split,
  calc 51^6 = (50 + 1)^6 : by rw [Nat.add_comm 50 1]
        ... = 68968845601 : by norm_num,
  refl
}

end sixth_root_68968845601_l503_503182


namespace operation_result_l503_503552

-- Define the new operation x # y
def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

-- Prove that (6 # 4) - (4 # 6) = -8
theorem operation_result : op 6 4 - op 4 6 = -8 :=
by
  sorry

end operation_result_l503_503552


namespace median_first_fifteen_positive_integers_l503_503467

-- Define the list of the first fifteen positive integers
def first_fifteen_positive_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

-- Define the property that the median of the list is 8.0
theorem median_first_fifteen_positive_integers : median(first_fifteen_positive_integers) = 8.0 := 
sorry

end median_first_fifteen_positive_integers_l503_503467


namespace median_first_fifteen_integers_l503_503468

theorem median_first_fifteen_integers :
  let l := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] in
  let seventh := l.nth 6 in
  let eighth := l.nth 7 in
  (seventh.is_some ∧ eighth.is_some) →
  (seventh.get_or_else 0 + eighth.get_or_else 0) / 2 = 7.5 :=
by
  sorry

end median_first_fifteen_integers_l503_503468


namespace power_function_value_l503_503421

noncomputable def f (x : ℝ) : ℝ := x^2

theorem power_function_value :
  f 3 = 9 :=
by
  -- Since f(x) = x^2 and f passes through (-2, 4)
  -- f(x) = x^2, so f(3) = 3^2 = 9
  sorry

end power_function_value_l503_503421


namespace triangle_angle_contradiction_l503_503872

theorem triangle_angle_contradiction (a b c : ℝ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : a + b + c = 180) (h₃ : 60 < a ∧ 60 < b ∧ 60 < c) : false :=
by
  sorry

end triangle_angle_contradiction_l503_503872


namespace num_satisfying_integers_l503_503023

def count_satisfying_integers : ℕ := 
  let numbers := [n | n ∈ finset.range 1001 5000 \ finset.range 1000 5000 1000, 
                       ∃ m ∈ (finset.range 1 1000), m ∣ (n % 1000)] in
  numbers.card

theorem num_satisfying_integers : count_satisfying_integers = 68 :=
by
  sorry

end num_satisfying_integers_l503_503023


namespace car_winning_probability_l503_503689

noncomputable def probability_of_winning (P_X P_Y P_Z : ℚ) : ℚ :=
  P_X + P_Y + P_Z

theorem car_winning_probability :
  let P_X := (1 : ℚ) / 6
  let P_Y := (1 : ℚ) / 10
  let P_Z := (1 : ℚ) / 8
  probability_of_winning P_X P_Y P_Z = 47 / 120 :=
by
  sorry

end car_winning_probability_l503_503689


namespace pieces_brought_to_school_on_friday_l503_503743

def pieces_of_fruit_mark_had := 10
def pieces_eaten_first_four_days := 5
def pieces_kept_for_next_week := 2

theorem pieces_brought_to_school_on_friday :
  pieces_of_fruit_mark_had - pieces_eaten_first_four_days - pieces_kept_for_next_week = 3 :=
by
  sorry

end pieces_brought_to_school_on_friday_l503_503743


namespace unique_solution_range_l503_503636
-- import relevant libraries

-- define the functions
def f (a x : ℝ) : ℝ := 2 * a * x ^ 3 + 3
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 2

-- state and prove the main theorem (statement only)
theorem unique_solution_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x = g x ∧ ∀ y : ℝ, y > 0 → f a y = g y → y = x) ↔ a ∈ Set.Iio (-1) :=
sorry

end unique_solution_range_l503_503636


namespace charles_total_earnings_l503_503544

def charles_earnings (house_rate dog_rate : ℝ) (house_hours dog_count dog_hours : ℝ) : ℝ :=
  (house_rate * house_hours) + (dog_rate * dog_count * dog_hours)

theorem charles_total_earnings :
  charles_earnings 15 22 10 3 1 = 216 := by
  sorry

end charles_total_earnings_l503_503544


namespace problem_solution_l503_503341

theorem problem_solution (x y : ℝ) :
  (x ≥ 0 → y > 0) ∧ (x < 0 → y > -3 * x) → (2 * y + 3 * x > 3 * |x|) :=
begin
  sorry
end

end problem_solution_l503_503341


namespace number_of_ways_to_get_max_label_three_l503_503686

-- Define the set of labels
inductive Label
| one : Label
| two : Label
| three : Label

-- Define the set of draws
def Draws := (Label × Label × Label)

-- Define a function that checks if a draw contains the label three
def contains_three (d : Draws) : Prop :=
  d.1 = Label.three ∨ d.2 = Label.three ∨ d.3 = Label.three

-- Define the total number of possible draws
def total_draws : ℕ := 3 * 3 * 3

-- Define the total number of draws not containing the label three
def draws_not_containing_three : ℕ := 8

-- The problem statement to prove
theorem number_of_ways_to_get_max_label_three : 
  ∃ (n : ℕ), n = (total_draws - draws_not_containing_three) :=
by
  existsi 19
  sorry

end number_of_ways_to_get_max_label_three_l503_503686


namespace number_of_triangles_in_regular_decagon_l503_503661

noncomputable def number_of_triangles_in_decagon : ℕ :=
∑ i in (finset.range 10).powerset_len 3, 1

theorem number_of_triangles_in_regular_decagon :
  number_of_triangles_in_decagon = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l503_503661


namespace part_a_part_b_part_c_l503_503050

variables (p : ℝ) (q := 1 - p) (N : ℕ)

-- Part a
theorem part_a (h : 0 < p ∧ p < 1) :
  let prob := 2 * p / (p + 1)
  in prob = 2 * p / (p + 1) :=
sorry

-- Part b
theorem part_b (h : 0 < p ∧ p < 1) :
  let prob := 2 * p / (2 * p + (1 - p)^2)
  in prob = 2 * p / (2 * p + (1 - p)^2) :=
sorry

-- Part c
theorem part_c (h : 0 < p ∧ p < 1) :
  let expected_pairs := N * p / (p + 1)
  in expected_pairs = N * p / (p + 1) :=
sorry

end part_a_part_b_part_c_l503_503050


namespace common_ratio_value_l503_503616

variable {α : Type*} [Field α]
variable {a : ℕ → α} {q : α} (neq1 : q ≠ 1)

-- Definition for geometric sequence
def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Definition for arithmetic sequence condition
def is_arithmetic_sequence (a1 a2 a3 : α) : Prop :=
  2 * a3 = a1 + a2

-- Main theorem
theorem common_ratio_value (h1 : is_geometric_sequence a q) (h2 : is_arithmetic_sequence (a 1) (a 2) (a 3)) : q = - (1 / 2) :=
by
  sorry

end common_ratio_value_l503_503616


namespace arithmetic_progression_other_term_position_l503_503058

noncomputable def arithmetic_progression_position : ℕ := 
  let a : ℝ := 10 - 4.5 * (2 * arithmetic_progression_position.d a_);
  let d := (100 - 10 * a) / 45 in
  let x  : ℕ :=  9 - 1 in x

-- Define terms of an arithmetic progression
def nth_term (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define sum of the first n terms of an arithmetic progression
def sum_of_first_n_terms (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)

theorem arithmetic_progression_other_term_position
  (a d : ℝ)
  (h1 : nth_term a d 12 + nth_term a d 8 = 20)
  (h2 : sum_of_first_n_terms a d 10 = 100) :
  nth_term a d 12 := 0:
  
  let l:= (10 - 4.5 * d, d)
  
  let c :=( (20 - 9*n + (10+x )-9) in 
  let p :  d*(x + 1 == 9):
  
 sorry 

end arithmetic_progression_other_term_position_l503_503058


namespace median_first_fifteen_positive_integers_l503_503465

-- Define the list of the first fifteen positive integers
def first_fifteen_positive_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

-- Define the property that the median of the list is 8.0
theorem median_first_fifteen_positive_integers : median(first_fifteen_positive_integers) = 8.0 := 
sorry

end median_first_fifteen_positive_integers_l503_503465


namespace cubes_with_no_colored_faces_l503_503521

theorem cubes_with_no_colored_faces (width length height : ℕ) (total_cubes cube_side : ℕ) :
  width = 6 ∧ length = 5 ∧ height = 4 ∧ total_cubes = 120 ∧ cube_side = 1 →
  (width - 2) * (length - 2) * (height - 2) = 24 :=
by
  intros h
  sorry

end cubes_with_no_colored_faces_l503_503521


namespace infinite_solutions_imply_values_l503_503304

theorem infinite_solutions_imply_values (a b : ℝ) :
  (∀ x : ℝ, a * (2 * x + b) = 12 * x + 5) ↔ (a = 6 ∧ b = 5 / 6) :=
by
  sorry

end infinite_solutions_imply_values_l503_503304


namespace units_digit_17_pow_31_l503_503492

theorem units_digit_17_pow_31 : (17 ^ 31) % 10 = 3 := by
  sorry

end units_digit_17_pow_31_l503_503492


namespace conic_section_is_ellipse_l503_503554

theorem conic_section_is_ellipse (x y : ℝ) : (x^2 + 2 * y^2 - 6 * x - 8 * y + 9 = 0) → 
  (∃ (h k : ℝ) (a b : ℝ), a > 0 ∧ b > 0 ∧ ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)) :=
by
  intros h,
  use 3, 2, 2, 1,
  sorry

end conic_section_is_ellipse_l503_503554


namespace initial_paintings_l503_503136

theorem initial_paintings (x : ℕ) (h : x - 3 = 95) : x = 98 :=
sorry

end initial_paintings_l503_503136


namespace roots_of_quadratic_eq_l503_503810

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l503_503810


namespace linear_function_not_in_second_quadrant_l503_503420

-- Define the linear function y = x - 1.
def linear_function (x : ℝ) : ℝ := x - 1

-- Define the condition for a point to be in the second quadrant.
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State that for any point (x, y) in the second quadrant, it does not satisfy y = x - 1.
theorem linear_function_not_in_second_quadrant {x y : ℝ} (h : in_second_quadrant x y) : linear_function x ≠ y :=
sorry

end linear_function_not_in_second_quadrant_l503_503420


namespace solve_integer_pairs_l503_503883

-- Definition of the predicate that (m, n) satisfies the given equation
def satisfies_equation (m n : ℤ) : Prop :=
  m * n^2 = 2009 * (n + 1)

-- Theorem stating that the only solutions are (4018, 1) and (0, -1)
theorem solve_integer_pairs :
  ∀ (m n : ℤ), satisfies_equation m n ↔ (m = 4018 ∧ n = 1) ∨ (m = 0 ∧ n = -1) :=
by
  sorry

end solve_integer_pairs_l503_503883


namespace tangent_points_are_on_locus_l503_503322

noncomputable def tangent_points_locus (d : ℝ) : Prop :=
∀ (x y : ℝ), 
((x ≠ 0 ∨ y ≠ 0) ∧ (x-d ≠ 0)) ∧ (y = x) 
→ (y^2 - x*y + d*(x + y) = 0)

theorem tangent_points_are_on_locus (d : ℝ) : 
  tangent_points_locus d :=
by sorry

end tangent_points_are_on_locus_l503_503322


namespace convinced_of_twelve_models_vitya_review_58_offers_l503_503091

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end convinced_of_twelve_models_vitya_review_58_offers_l503_503091


namespace rectangular_prism_edge_sum_l503_503905

theorem rectangular_prism_edge_sum
  (V A : ℝ)
  (hV : V = 8)
  (hA : A = 32)
  (l w h : ℝ)
  (geom_prog : l = w / h ∧ w = l * h ∧ h = l * (w / l)) :
  4 * (l + w + h) = 28 :=
by 
  sorry

end rectangular_prism_edge_sum_l503_503905


namespace mason_savings_fraction_l503_503744

theorem mason_savings_fraction (M p b : ℝ) (h : (1 / 4) * M = (2 / 5) * b * p) : 
  (M - b * p) / M = 3 / 8 :=
by 
  sorry

end mason_savings_fraction_l503_503744


namespace normal_map_three_colorable_iff_even_sided_countries_l503_503758

/--
Prove that a normal map can be properly colored with three colors if and only if all its countries are even-sided polygons.
-/
theorem normal_map_three_colorable_iff_even_sided_countries 
  (K : Type) 
  [normal_map K] 
  (countries_even_sided : ∀ c : K, even (num_borders c)) :
  (three_colorable K ↔ ∀ (c : K), even (num_borders c)) := 
sorry

end normal_map_three_colorable_iff_even_sided_countries_l503_503758


namespace median_of_first_fifteen_positive_integers_l503_503477

theorem median_of_first_fifteen_positive_integers : ∃ m : ℝ, m = 7.5 :=
by
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median := (seq[6] + seq[7]) / 2
  use median
  sorry

end median_of_first_fifteen_positive_integers_l503_503477


namespace infinitely_many_not_2a_3b_5c_l503_503760

theorem infinitely_many_not_2a_3b_5c : ∃ᶠ x : ℤ in Filter.cofinite, ∀ a b c : ℕ, x % 120 ≠ (2^a + 3^b - 5^c) % 120 :=
by
  sorry

end infinitely_many_not_2a_3b_5c_l503_503760


namespace expected_number_of_hits_l503_503698

variable (W : ℝ) (n : ℕ)
def expected_hits (W : ℝ) (n : ℕ) : ℝ := W * n

theorem expected_number_of_hits :
  W = 0.75 → n = 40 → expected_hits W n = 30 :=
by
  intros hW hn
  rw [hW, hn]
  norm_num
  sorry

end expected_number_of_hits_l503_503698


namespace sum_to_product_identity_l503_503568

theorem sum_to_product_identity (a b : ℝ) : 
  cos (a + b) + cos (a - b) = 2 * cos a * cos b := 
by {
  sorry
}

end sum_to_product_identity_l503_503568


namespace amount_spent_on_candy_l503_503742

-- Define the given conditions
def amount_from_mother := 80
def amount_from_father := 40
def amount_from_uncle := 70
def final_amount := 140 

-- Define the initial amount
def initial_amount := amount_from_mother + amount_from_father 

-- Prove the amount spent on candy
theorem amount_spent_on_candy : 
  initial_amount - (final_amount - amount_from_uncle) = 50 := 
by
  -- Placeholder for proof
  sorry

end amount_spent_on_candy_l503_503742


namespace median_of_first_fifteen_integers_l503_503487

theorem median_of_first_fifteen_integers : 
  let L := (list.range 15).map (λ n, n + 1)
  in list.median L = 8.0 :=
by 
  sorry

end median_of_first_fifteen_integers_l503_503487


namespace infinite_solutions_factorial_equation_l503_503404

-- Define the factorial function (already present in Lean)
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Lean theorem statement to demonstrate the equation x! y! = z! has infinitely many solutions
theorem infinite_solutions_factorial_equation :
  ∃ (m : ℕ) (x y z : ℕ), m > 1 ∧ y = m ∧ x = factorial m - 1 ∧ z = factorial m ∧ (factorial x * factorial y = factorial z) :=
by
  sorry

end infinite_solutions_factorial_equation_l503_503404


namespace mila_calculator_sum_l503_503389

theorem mila_calculator_sum :
  let n := 60
  let calc1_start := 2
  let calc2_start := 0
  let calc3_start := -1
  calc1_start^(3^n) + calc2_start^2^(n) + (-calc3_start)^n = 2^(3^60) + 1 :=
by {
  sorry
}

end mila_calculator_sum_l503_503389


namespace graph_does_not_pass_through_third_quadrant_l503_503495

theorem graph_does_not_pass_through_third_quadrant (k x y : ℝ) (hk : k < 0) :
  y = k * x - k → (¬ (x < 0 ∧ y < 0)) :=
by
  sorry

end graph_does_not_pass_through_third_quadrant_l503_503495


namespace find_positive_n_for_quadratic_l503_503181

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

-- Define the condition: the quadratic equation has exactly one real root if its discriminant is zero
def has_one_real_root (a b c : ℝ) : Prop := discriminant a b c = 0

-- The specific quadratic equation y^2 + 6ny + 9n
def my_quadratic (n : ℝ) : Prop := has_one_real_root 1 (6 * n) (9 * n)

-- The statement to be proven: for the quadratic equation y^2 + 6ny + 9n to have one real root, n must be 1
theorem find_positive_n_for_quadratic : ∃ (n : ℝ), my_quadratic n ∧ n > 0 ∧ n = 1 := 
by
  sorry

end find_positive_n_for_quadratic_l503_503181


namespace monotonicity_f_inequality_f_l503_503267

-- Define the function f(x) given m
def f (m : ℝ) (x : ℝ) : ℝ := m / x + Real.log x

-- Part 1: Monotonicity of f(x)
theorem monotonicity_f (m : ℝ) :
  (m ≤ 0 → ∀ x : ℝ, 0 < x → f m x ≥ f m (x + 1)) ∧
  (m > 0 → (∀ x : ℝ, 0 < x ∧ x < m → f m x ≥ f m (x + 0.1)) ∧
  (∀ x : ℝ, x > m → f m x ≤ f m (x + 0.1))) :=
sorry

-- Part 2: Prove that mf(x) ≥ 2m - 1 for m > 0
theorem inequality_f (m : ℝ) (x : ℝ) (h : m > 0) (hx : 0 < x) : m * f m x ≥ 2 * m - 1 :=
sorry

end monotonicity_f_inequality_f_l503_503267


namespace problem_1_problem_2_l503_503596

variables {α : ℝ}
noncomputable def tan (x : ℝ) : ℝ := (sin x) / (cos x)

-- Given tan(α) = 2, prove the first equation
theorem problem_1 (h : tan α = 2) : 
  (sin (α - 3 * π) + cos (π + α)) / (sin (-α) - cos (π + α)) = 3 := 
by
  sorry

-- Given tan(α) = 2, prove the second equation
theorem problem_2 (h : tan α = 2) : 
  cos α ^ 2 - 2 * (sin α * cos α) = 2 / 5 := 
by 
  sorry

end problem_1_problem_2_l503_503596


namespace spiders_cannot_catch_fly_l503_503833

-- Define the conditions for the cube, initial positions, and speed ratios
def spiders_and_fly (v_s v_f a : ℝ) : Prop :=
  (v_f = 3 * v_s) ∧
  ∀ t : ℝ, t ≥ 0 → (∀ p_s p_f : ℝ, p_f = v_f * t → p_s = v_s * t → 
  (p_f ≠ (a * (3)³)))

-- Define the main theorem
theorem spiders_cannot_catch_fly (a v_s v_f : ℝ) : 
  spiders_and_fly v_s v_f a → 
  (∀ t : ℝ, t ≥ 0 → ¬(∃ t_c : ℝ, ∀ t : ℝ, t ≥ t_c → ∃ p_s p_f : ℝ, 
  p_f = v_f * t ∧ p_s = v_s * t ∧ p_s = p_f)) :=
by 
  sorry

end spiders_cannot_catch_fly_l503_503833


namespace carwash_problem_l503_503359

theorem carwash_problem
(h1 : ∀ (n : ℕ), 5 * n + 6 * 5 + 7 * 5 = 100)
(h2 : 5 * 5 = 25)
(h3 : 7 * 5 = 35)
(h4 : 100 - 35 - 30 = 35):
(n = 7) :=
by
  have h : 5 * n = 35 := by sorry
  exact eq_of_mul_eq_mul_left (by sorry) h

end carwash_problem_l503_503359


namespace choose_three_collinear_points_choose_four_points_with_three_collinear_l503_503752

-- Define the points and collinearity conditions
constant num_points : ℕ := 9
constant collinear_groups : ℕ := 8
constant points_per_group : ℕ := 3

-- Problem a: Prove the number of ways to choose three collinear points
theorem choose_three_collinear_points (h : collinear_groups = 8) : 
  (choose_three_collinear_points == 8) :=
sorry

-- Problem b: Prove the number of ways to choose four points such that three of them are collinear
theorem choose_four_points_with_three_collinear (h : collinear_groups = 8) (remaining_points : ℕ := num_points - (points_per_group - 1)) :
  (choose_three_collinear_points * remaining_points == 48) :=
sorry

end choose_three_collinear_points_choose_four_points_with_three_collinear_l503_503752


namespace smallest_enclosing_sphere_radius_l503_503965

theorem smallest_enclosing_sphere_radius:
  let r := 1 -- Radius of each sphere
  let centers := {p : ℝ × ℝ × ℝ | ∃ (x y z : ℝ), (abs x = r) ∧ (abs y = r ∧ abs z = r)} in
  let max_distance := sup (λ (p : ℝ × ℝ × ℝ), real.sqrt (p.1 ^ 2 + p.2 ^ 2 + p.3 ^ 2)) in
  let enclosing_radius := max_distance + r in
  enclosing_radius = real.sqrt 6 :=
begin
  sorry
end

end smallest_enclosing_sphere_radius_l503_503965


namespace range_of_a_l503_503435

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ∈ set.Iio 1 → 
  (∀ y ∈ set.Iio 1, (log (x^2 - a * x + 3) / log 0.5 < log (y^2 - a*y + 3) / log 0.5) ↔ (x < y))) ↔ 
  (2 ≤ a ∧ a ≤ 4) :=
begin
  sorry
end

end range_of_a_l503_503435


namespace median_of_fifteen_is_eight_l503_503479

def median_of_first_fifteen_positive_integers : ℝ :=
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median_pos := (list.length lst + 1) / 2  
  lst.get (median_pos - 1)

theorem median_of_fifteen_is_eight : median_of_first_fifteen_positive_integers = 8.0 := 
  by 
    -- Proof omitted    
    sorry

end median_of_fifteen_is_eight_l503_503479


namespace triangles_from_decagon_l503_503669

-- Define the parameters for the problem
def n : ℕ := 10
def k : ℕ := 3

-- Define the combination formula
def combination (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- State the theorem we want to prove
theorem triangles_from_decagon : combination n k = 120 := by
  -- Proof steps would go here
  sorry

end triangles_from_decagon_l503_503669


namespace triangles_from_decagon_l503_503668

-- Define the parameters for the problem
def n : ℕ := 10
def k : ℕ := 3

-- Define the combination formula
def combination (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- State the theorem we want to prove
theorem triangles_from_decagon : combination n k = 120 := by
  -- Proof steps would go here
  sorry

end triangles_from_decagon_l503_503668


namespace xy_z_sum_l503_503879

theorem xy_z_sum : 
  let X := 0.20 * 50 in
  let Y := 40 / 0.20 in
  let Z := (40 * 100) / 50 in
  X + Y + Z = 290 :=
by
  let X := 0.20 * 50
  let Y := 40 / 0.20
  let Z := (40 * 100) / 50
  sorry

end xy_z_sum_l503_503879


namespace largest_n_l503_503178

theorem largest_n {x y z n : ℕ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (n:ℤ)^2 = (x:ℤ)^2 + (y:ℤ)^2 + (z:ℤ)^2 + 2*(x:ℤ)*(y:ℤ) + 2*(y:ℤ)*(z:ℤ) + 2*(z:ℤ)*(x:ℤ) + 6*(x:ℤ) + 6*(y:ℤ) + 6*(z:ℤ) - 12
  → n = 13 :=
sorry

end largest_n_l503_503178


namespace tangent_line_eq_monotonicity_intervals_l503_503371

noncomputable def f (a b x : ℝ) : ℝ := a * x * Real.log x + b

theorem tangent_line_eq (b := 3) : 
    (f 2 b (1 : ℝ) = 3) → 
    (∀ x, 2 * x - (f 2 b x) + 1 = 2 * (x - 1)) →
    True  := 
by
  intros
  sorry

theorem monotonicity_intervals (a b : ℝ) (h_nonzero : a ≠ 0) :
    (∀ x, x > 0 → (a * (Real.log x + 1)) > 0 → x > 1 / Real.exp 1) →
    (∀ x, x > 0 → (a * (Real.log x + 1)) < 0 → x < 1 / Real.exp 1) →
    a > 0 → 
    (∀ x, x ∈ (Set.Ioo 0 (1 / Real.exp 1)) → Function.StrictMonoOn (f a b) (Set.Ioo 0 (1 / Real.exp 1))) ∧ 
    (∀ x, x ∈ (Set.Ioi (1 / Real.exp 1)) → Function.monotone_on (f a b) (Set.Ioi (1 / Real.exp 1))) →
    a < 0 → 
    ((∀ x, x ∈ (Set.Ioo 0 (1 / Real.exp 1)) → Function.monotone_on (f a b) (Set.Ioo 0 (1 / Real.exp 1))) ∧ 
    (∀ x, x ∈ (Set.Ioi (1 / Real.exp 1)) → Function.StrictMonoOn (f a b) (Set.Ioi (1 / Real.exp 1)))) := 
by
  intros
  sorry

end tangent_line_eq_monotonicity_intervals_l503_503371


namespace solve_for_x_l503_503984

theorem solve_for_x : ∀ x : ℚ, (3 ^ (2 * x^2 - 6 * x + 2) = 3 ^ (2 * x^2 + 8 * x - 4)) → x = 3 / 7 := by
  intros x h
  sorry

end solve_for_x_l503_503984


namespace locus_of_M_l503_503993

variable {O P Q M M₀ : Point}
variable {circle : Circle}
variable [Inhabited circle]

-- Definitions from conditions
def in_circle (P : Point) (circle : Circle) : Prop := P ∈ circle.interior
def on_circumference (Q : Point) (circle : Circle) : Prop := Q ∈ circle.boundary
def tangent (e : Line) (circle : Circle) (Q : Point) : Prop := e.is_tangent_to circle at Q
def perpendicular_from_center (O : Point) (PQ : Line) (e : Line) (M : Point) : Prop :=
  M ∈ e ∧ e.is_perpendicular_to (Line.mk O (intersection_point_with_line O PQ))

-- The statement proving the locus
theorem locus_of_M (hO_center : IsCenter O circle)
                   (hP_inside : in_circle P circle)
                   (hQ_on_bound : ∀ Q, on_circumference Q circle)
                   (he_tangent : ∀ Q, ∃ e, tangent e circle Q)
                   (hM_inter : ∀ Q, ∃ M, perpendicular_from_center O (Line.mk P Q) e M) :
  ∀ Q, on_circumference Q circle → 
  ∃ M₀, (∃ Q₀, on_circumference Q₀ circle ∧ (dist M₀ O) * (dist P O) = (dist Q₀ O)^2) ∧ 
        ∀ M, intersect (Line.mk M₀ (translate_onto_perpendicular O P Q)) :=
sorry

end locus_of_M_l503_503993


namespace sum_of_absolute_differences_equal_n_squared_l503_503429

variable (n : ℕ)

-- Assume a and b are sequences representing groups with the given properties
variable (a b : Fin n → ℕ)
variable (combined: Fin (2 * n) → ℕ)

-- Definitions for the sequences a and b having increasing and decreasing orders respectively.
axiom (a_increasing : ∀ i j : Fin n, i < j → a i < a j)
axiom (b_decreasing : ∀ i j : Fin n, i < j → b i > b j)

-- Definition ensuring that combined contains all elements 1 to 2*n.
axiom (combined_elements : ∀ (k : Fin (2 * n)), combined k = k.1 + 1)

-- Definition that (a i) and (b i) comprise a partition of combined.
axiom (partition : ∀ i : Fin (2 * n), i < n → ∃ j : Fin n, (combined i = a j) ∨ (combined i = b j))

theorem sum_of_absolute_differences_equal_n_squared (h : (...)) :
  (Finset.univ.sum (λ i, Nat.abs (a i - b i))) = n^2 :=
sorry

end sum_of_absolute_differences_equal_n_squared_l503_503429


namespace problem1_general_formula_problem2_range_of_a_problem3_sum_of_first_n_terms_l503_503265

noncomputable def f (x : ℝ) : ℝ :=
  2 * |x + 2| - |x + 1|

noncomputable def a_seq (n : ℕ) : ℝ := by
  apply ite (n = 0) 0
  apply f (n - 1)

theorem problem1_general_formula (n : ℕ) (h : n ≠ 0) :
  a_seq n = n + 3 := by
  sorry

theorem problem2_range_of_a (a : ℝ) :
  (∀n > 1, a_seq (n + 1) = f (a_seq n) →
      (∃ d, a_seq (n + 1) = a_seq n + d)) →
  a ≥ -1 ∨ a = -3 := by
  sorry

theorem problem3_sum_of_first_n_terms (a : ℝ) (n : ℕ) :
  S_n a n = ∑ i in range n, a_seq i =
    if a ≥ -1 then
      (3/2 : ℝ) * (n ^ 2) + (a - 3 / 2) * n
    else if -2 < a ∧ a ≤ -1 then
      (3/2 : ℝ) * (n ^ 2) + (1/2 + 3 * a) * n - 2 * a - 2
    else
      (3/2 : ℝ) * (n ^ 2) - (a + 15 / 2) * n + 2 * a + 6 := by
  sorry

end problem1_general_formula_problem2_range_of_a_problem3_sum_of_first_n_terms_l503_503265


namespace smaller_value_r_plus_s_l503_503732

theorem smaller_value_r_plus_s :
  ∃ (x : ℝ) (r s : ℤ), (∃ (a b : ℝ), a = x^(1/3) ∧ b = (30 - x)^(1/3) ∧ a + b = 3 ∧ a^3 + b^3 = 30)
  ∧ x = r - sqrt (s : ℝ) ∧ r + s = 96 :=
sorry

end smaller_value_r_plus_s_l503_503732


namespace smallest_is_B_l503_503064

def A : ℕ := 32 + 7
def B : ℕ := (3 * 10) + 3
def C : ℕ := 50 - 9

theorem smallest_is_B : min A (min B C) = B := 
by 
  have hA : A = 39 := by rfl
  have hB : B = 33 := by rfl
  have hC : C = 41 := by rfl
  rw [hA, hB, hC]
  exact sorry

end smallest_is_B_l503_503064


namespace characterize_additive_function_l503_503971

def additivity (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f(x + y) = f(x) + f(y)

theorem characterize_additive_function (f : ℤ → ℤ) (h : additivity f) :
  ∃ a : ℤ, ∀ x : ℤ, f(x) = a * x :=
sorry

end characterize_additive_function_l503_503971


namespace absolute_value_expression_evaluation_l503_503567

theorem absolute_value_expression_evaluation : abs (-2) * (abs (-Real.sqrt 25) - abs (Real.sin (5 * Real.pi / 2))) = 8 := by
  sorry

end absolute_value_expression_evaluation_l503_503567


namespace numTrianglesFromDecagon_is_120_l503_503665

noncomputable def numTrianglesFromDecagon : ℕ := 
  nat.choose 10 3

theorem numTrianglesFromDecagon_is_120 : numTrianglesFromDecagon = 120 := 
  by
    -- Form the combination
    have : numTrianglesFromDecagon = nat.choose 10 3 := rfl

    -- Calculate
    have calc₁ : nat.choose 10 3 = 10 * 9 * 8 / (3 * 2 * 1) := by 
      exact nat.choose_eq_div
      simp

    -- Simplify the calculation to 120
    have : 10 * 9 * 8 / (3 * 2 * 1) = 120 := by 
      norm_num 

    exact eq.trans this.symm calc₁.symm ⟩

end numTrianglesFromDecagon_is_120_l503_503665


namespace reverse_greater_count_l503_503532

theorem reverse_greater_count : 
  (∃ (count : ℕ), count = 36 ∧ ∀ (n : ℕ), (10 ≤ n ∧ n < 100 ∧ let a := n / 10 in let b := n % 10 in b > a) → ∃ (m : ℕ), m = (b * 10 + a) ∧ m > n) :=
by
  sorry

end reverse_greater_count_l503_503532


namespace graph_sequence_periodic_l503_503830

open Finset

/-- Given a graph G_0 on vertices A_1, A_2, ..., A_n,
    and a sequence G_{n+1} constructed such that A_i and A_j are joined only if
    in G_n there is a vertex A_k ≠ A_i, A_j such that A_k is joined with both A_i and A_j,
    prove that the sequence {G_n} is periodic after some term with period T ≤ 2^n. -/
theorem graph_sequence_periodic (n : ℕ) (G_0 : SimpleGraph (Fin n))
  (G_seq : ℕ → SimpleGraph (Fin n))
  (h : ∀ k, ∀ (A_i A_j : Fin n), (A_i ∈ G_seq k.edges) → (A_j ∈ G_seq k.edges) →
  ∃ (A_k : Fin n), A_k ≠ A_i ∧ A_k ≠ A_j ∧ A_k ∈ G_seq k.edges ∧
  A_i ∈ G_seq (k+1).edges ∧ A_j ∈ G_seq (k+1).edges)
  : ∃ T ≤ 2^n, ∀ t ≥ T, G_seq t = G_seq T := sorry

end graph_sequence_periodic_l503_503830


namespace range_of_m_l503_503231

theorem range_of_m {m : ℝ} (h1 : ∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1)))
                   (h2 : ∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))
                   (h3 : ¬(∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1))) ∧
                           (∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))) :
  m > 1 :=
by
  sorry

end range_of_m_l503_503231


namespace convinced_of_twelve_models_vitya_review_58_offers_l503_503088

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end convinced_of_twelve_models_vitya_review_58_offers_l503_503088


namespace pushing_car_effort_l503_503418

theorem pushing_car_effort (effort constant : ℕ) (people1 people2 : ℕ) 
  (h1 : constant = people1 * effort)
  (h2 : people1 = 4)
  (h3 : effort = 120)
  (h4 : people2 = 6) :
  effort * people1 = constant → constant = people2 * 80 :=
by
  sorry

end pushing_car_effort_l503_503418


namespace vector_relationships_l503_503284

def is_parallel (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  ∀ k : ℝ, k ≠ 0 → v₁ = (k * v₂.1, k * v₂.2, k * v₂.3)

def is_orthogonal (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3 = 0

theorem vector_relationships :
  let a := (-2, -3, 1)
  let b := (2, 0, 4)
  let c := (4, 6, -2)
  is_parallel a c ∧ is_orthogonal a b := by
  sorry

end vector_relationships_l503_503284


namespace find_bd_l503_503206

noncomputable def b : ℚ := 128 / 15
noncomputable def d : ℚ := 7 / 5

def vec1 : ℤ × ℤ × ℤ := (4, 5, 3)
def vec2 : ℚ × ℤ × ℚ := (b, -2, d)
def vec_result : ℤ × ℤ × ℤ := (-13, -20, 23)

def cross_product (u v : ℚ × ℤ × ℚ) : ℚ × ℚ × ℚ :=
  (u.2.1 * v.2 - u.2 * v.2.1,
   u.2 * v.1 - u.1 * v.2,
   u.1 * v.2.1 - u.2.1 * v.1)

theorem find_bd :
  cross_product vec2 vec1 = (vec_result.1, vec_result.2.1, vec_result.2) :=
by
auto

end find_bd_l503_503206


namespace average_of_second_class_l503_503030

variable (average1 : ℝ) (average2 : ℝ) (combined_average : ℝ) (n1 : ℕ) (n2 : ℕ)

theorem average_of_second_class
  (h1 : n1 = 25) 
  (h2 : average1 = 40) 
  (h3 : n2 = 30) 
  (h4 : combined_average = 50.90909090909091) 
  (h5 : n1 + n2 = 55) 
  (h6 : n2 * average2 = 55 * combined_average - n1 * average1) :
  average2 = 60 := by
  sorry

end average_of_second_class_l503_503030


namespace equation_of_line_l503_503198

theorem equation_of_line (a b : ℝ) (h1 : a = -2) (h2 : b = 2) :
  (∀ x y : ℝ, (x / a + y / b = 1) → x - y + 2 = 0) :=
by
  sorry

end equation_of_line_l503_503198


namespace range_of_x_l503_503728

noncomputable def T (x : ℝ) : ℝ := |(2 * x - 1)|

theorem range_of_x (x : ℝ) (h : ∀ a : ℝ, T x ≥ |1 + a| - |2 - a|) : 
  x ≤ -1 ∨ 2 ≤ x :=
by
  sorry

end range_of_x_l503_503728


namespace arithmetic_sequence_sum_S9_l503_503823

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Definition of an arithmetic sequence (general term formula)
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of first n terms of an arithmetic sequence
def S (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_sum_S9 :
  is_arithmetic_sequence a →
  (a 4 + a 6 = 12) →
  (S 9 = 54) :=
by {
  intro h_seq,
  intro h_eq,
  sorry
}

end arithmetic_sequence_sum_S9_l503_503823


namespace proposition_pairing_even_l503_503935

theorem proposition_pairing_even (P Q : Prop) (h1 : (P → Q) = (¬Q → ¬P)) (h2 : (¬P → ¬Q) = (Q → P)) :
  ∃ n, n % 2 = 0 ∧ (∃ p1 p2 p3 p4, ⟦ [p1, p2, p3, p4] = [P → Q, Q → P, ¬Q → ¬P, ¬P → ¬Q] ⟧ ∧ ∑ i in [p1, p2, p3, p4], ite i 1 0 = n) :=
sorry

end proposition_pairing_even_l503_503935


namespace kendall_nickels_count_l503_503351

theorem kendall_nickels_count :
  ∃ (n : ℕ), n * 0.05 = 4 - (10 * 0.25 + 12 * 0.10) ∧ n = 6 :=
by
  have quarters_value : ℝ := 10 * 0.25
  have dimes_value : ℝ := 12 * 0.10
  have total_value : ℝ := 4
  have nickels_value : ℝ := total_value - (quarters_value + dimes_value)
  use 6
  split
  sorry
  sorry

end kendall_nickels_count_l503_503351


namespace hyperbola_theorem_angle_bisector_theorem_l503_503134

namespace HyperbolaProblem

-- Conditions of the problem
def hyperbola_condition (A: ℝ × ℝ) (foci: ℝ × ℝ × ℝ × ℝ) (e: ℝ) : Prop :=
  let (F1, F2) := foci in
  A = (4, 6) ∧
  (F1.1 = -4 ∧ F1.2 = 0) ∧ (F2.1 = 4 ∧ F2.2 = 0) ∧
  e = 2

-- Equation of the hyperbola
def hyperbola_equation (x y a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Conditions mentioning point on hyperbola and eccentricity
def hyperbola_specific_condition (A : ℝ × ℝ) (a b c e : ℝ) : Prop :=
  (4^2 / a^2 - 6^2 / b^2 = 1) ∧ (c^2 = a^2 + b^2) ∧ (c / a = e) ∧ (e = 2)

-- Proof of hyperbola equation
noncomputable def proof_hyperbola_equation (a b : ℝ) : Prop :=
  hyperbola_equation 4 6 a b ∧ a^2 = 4 ∧ b^2 = 12 ∧ 
  (∀ x y, hyperbola_equation x y 2 sqrt 12)

-- Equation of the angle bisector line
def angle_bisector_equation (m : ℝ) : Prop :=
  m = 1 ∧ (∀ x, y = 2 * x - 2)

-- Main theorem to be proven
theorem hyperbola_theorem : 
  ∃ a b, hyperbola_condition (4, 6) ((-4, 0), (4, 0)) 2 → proof_hyperbola_equation a b := sorry

-- Main theorem to be proven for angle bisector
theorem angle_bisector_theorem : 
  angle_bisector_equation 1 := sorry

end HyperbolaProblem

end hyperbola_theorem_angle_bisector_theorem_l503_503134


namespace image_digit_sum_l503_503011

theorem image_digit_sum :
  ∃ cat chicken crab bear goat : ℕ,
  ((5 * crab = 10) ∧
   (4 * crab + goat = 11) ∧
   (2 * goat + crab + 2 * bear = 16) ∧
   (cat + bear + 2 * goat + crab = 13) ∧
   (2 * crab + 2 * chicken + goat = 17)) ∧
   10000 * cat + 1000 * chicken + 100 * crab + 10 * bear + goat = 15243 :=
begin
  use [1, 5, 2, 4, 3],
  split,
  { split; try {linarith},
    split; try {linarith},
    split; linarith },
  exact rfl,
end

end image_digit_sum_l503_503011


namespace period_of_function_is_2pi_over_3_l503_503444

noncomputable def period_of_f (x : ℝ) : ℝ :=
  4 * (Real.sin x)^3 - Real.sin x + 2 * (Real.sin (x / 2) - Real.cos (x / 2))^2

theorem period_of_function_is_2pi_over_3 : ∀ x, period_of_f (x + (2 * Real.pi) / 3) = period_of_f x :=
by sorry

end period_of_function_is_2pi_over_3_l503_503444


namespace median_of_first_fifteen_integers_l503_503485

theorem median_of_first_fifteen_integers : 
  let L := (list.range 15).map (λ n, n + 1)
  in list.median L = 8.0 :=
by 
  sorry

end median_of_first_fifteen_integers_l503_503485


namespace sequence_periodicity_l503_503338

theorem sequence_periodicity : 
  let a : ℕ → ℤ := λ n, 
    if n = 1 then 13 
    else if n = 2 then 56 
    else if n > 2 ∧ n % 6 = 5 then a (n - 1) + a (n - 2) 
    else a (n - 1) - a (n - 2)
  in a 1934 = 56 :=
by
  sorry

end sequence_periodicity_l503_503338


namespace partial_fraction_decomposition_l503_503576

theorem partial_fraction_decomposition :
  ∃ A B C : ℚ, (∀ x : ℚ, x ≠ -1 ∧ x^2 - x + 2 ≠ 0 →
          (x^2 + 2 * x - 8) / (x^3 - x - 2) = A / (x + 1) + (B * x + C) / (x^2 - x + 2)) ∧
          A = -9/4 ∧ B = 13/4 ∧ C = -7/2 :=
sorry

end partial_fraction_decomposition_l503_503576


namespace max_profit_production_volume_and_value_l503_503512

noncomputable def profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then - (1/2) * x^2 + 300 * x - 20000
  else 60000 - 100 * x

theorem max_profit_production_volume_and_value :
  ∃ x : ℝ, x = 300 ∧ profit x = 25000 :=
by
  use 300
  split
  { refl }
  { sorry }

end max_profit_production_volume_and_value_l503_503512


namespace count_valid_initial_values_l503_503383

def sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if a (n - 1) % 2 = 0 then a (n - 1) / 2
  else 5 * a (n - 1) + 1

theorem count_valid_initial_values :
  let a := λ (n : ℕ), sequence (λ n, a n) n
  let valid_initial (a1 : ℕ) := a1 ≤ 1000 ∧ a1 % 2 = 1 ∧
    a1 < sequence (λ n, if n = 0 then a1 else sequence (λ n, a n) n) 1 ∧
    a1 < sequence (λ n, if n = 0 then a1 else sequence (λ n, a n) n) 2 ∧
    a1 < sequence (λ n, if n = 0 then a1 else sequence (λ n, a n) n) 3
  in (finset.range 1001).filter valid_initial).card = 500 :=
begin
  sorry
end

end count_valid_initial_values_l503_503383


namespace fruit_difference_l503_503750

noncomputable def apples : ℕ := 60
noncomputable def peaches : ℕ := 3 * apples

theorem fruit_difference : peaches - apples = 120 :=
by
  have h1 : apples = 60 := rfl
  have h2 : peaches = 3 * apples := rfl
  calc
    peaches - apples = 3 * apples - apples : by rw [h2]
                ... = 3 * 60 - 60        : by rw [h1]
                ... = 180 - 60           : by norm_num
                ... = 120                : by norm_num

end fruit_difference_l503_503750


namespace hot_drink_price_control_l503_503940

theorem hot_drink_price_control (p_0 v_0 c : ℝ) 
  (h_sales : p_0 = 1.5 ∧ v_0 = 800 ∧ c = 0.9 ∧ 
    ∀ x : ℝ, ((p_0 + x - c) * (v_0 - 20 * x) ≥ 720 ↔ 0.4 ≤ x ∧ x ≤ 3)) :
  ∀ x : ℝ, ((1.5 + x) ∈ set.Icc 1.9 4.5 ↔ 0.4 ≤ x ∧ x ≤ 3) :=
by
  intros x
  specialize h_sales.installation_of_theorem
  sorry

end hot_drink_price_control_l503_503940


namespace product_of_solutions_l503_503202

theorem product_of_solutions : 
  ∀ x₁ x₂ : ℝ, (|6 * x₁| + 5 = 47) ∧ (|6 * x₂| + 5 = 47) → x₁ * x₂ = -49 :=
by
  sorry

end product_of_solutions_l503_503202


namespace sum_inverse_products_l503_503587

noncomputable def product_of_elements (C : Finset ℕ) : ℕ :=
  if C = ∅ then 1 else C.prod id

theorem sum_inverse_products (n : ℕ) :
  ∑ (C : Finset ℕ) in (Finset.powerset (Finset.range (n+1))), (1 / (product_of_elements C : ℚ)) = n + 1 :=
  sorry

end sum_inverse_products_l503_503587


namespace binom_coeff_expansion_l503_503680

theorem binom_coeff_expansion (n : ℕ) 
  (h_sum : (2:ℕ)^n = 32) : 
  (∃ r : ℕ, (5 - 2 * r = 3) ∧ (binom 5 r * 2^(5-r) * (-1)^r = -80)) := 
by 
  have h_n: n = 5 := by 
    sorry 
  use 1 
  split 
  case h_left 
  · 
    simp 
  case h_right 
  · 
    calc 
      binom 5 1 * 2^(5-1) * (-1)^1 
      = 5 * 2^4 * (-1) : by 
        simp 
      ... = -80 : by
        simp

end binom_coeff_expansion_l503_503680


namespace butterfly_eq_roots_l503_503958

variable (a b c : ℝ)

def is_butterfly (a b c : ℝ) : Prop := (a ≠ 0) ∧ (a - b + c = 0)

theorem butterfly_eq_roots (h : is_butterfly a b c) (roots_eq : b^2 - 4*a*c = 0) : a = c :=
by {
    sorry, -- Proof will be provided here
}

end butterfly_eq_roots_l503_503958


namespace andrew_age_l503_503939

theorem andrew_age :
  ∃ a g : ℕ, g = 15 * a ∧ g - a = 70 ∧ a = 5 :=
by
  use 5
  use 75
  split
  · exact rfl
  split
  · exact rfl
  · exact rfl

end andrew_age_l503_503939


namespace ball_distribution_ways_l503_503290

theorem ball_distribution_ways :
  ∃ (n : ℕ), n = 56 ∧ ∀ (b : ℕ) (k : ℕ), b = 5 ∧ k = 4 → 
    (∑ i in (Finset.partitions b k), (Finset.choose_multiset (multiset.of_finset i))).card = n :=
by
  sorry

end ball_distribution_ways_l503_503290


namespace solve_for_x_l503_503234

theorem solve_for_x (x : ℝ) (h : 2^x + 2^x + 2^x + 2^x = 512) : x = 7 := by
  sorry

end solve_for_x_l503_503234


namespace find_larger_number_l503_503200

theorem find_larger_number 
  (L S : ℕ) 
  (h1 : L - S = 2342) 
  (h2 : L = 9 * S + 23) : 
  L = 2624 := 
sorry

end find_larger_number_l503_503200


namespace range_of_a_l503_503675

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x - 2| ≤ a) ↔ a ≥ 1 :=
sorry

end range_of_a_l503_503675


namespace graduation_ceremony_sum_x_l503_503838

theorem graduation_ceremony_sum_x :
  (∑ x in Finset.filter (λ x : ℕ, 15 ≤ x ∧ ∃ y, 10 ≤ y ∧ x * y = 288) (Finset.range 289)) = 58 :=
by {
  sorry
}

end graduation_ceremony_sum_x_l503_503838


namespace max_value_f_when_a_4_value_of_a_two_zeros_inequality_proof_l503_503261

-- 1. Maximum value of f(x) when a = 4
theorem max_value_f_when_a_4 : 
  ∀ x : ℝ, x ∈ set.Icc 0 (4/3) → (2 * x ^ 3 - 4 * x ^ 2 + 1 ≤ 1) :=
  sorry

-- 2. Value of a for exactly two zeros for f(x)
theorem value_of_a_two_zeros :
  ∀ f (a : ℝ), (f = 2 * x ^ 3 - a * x ^ 2 + 1) → 
  (∃ z : set ℝ, z.card = 2 ∧ (∀ x ∈ z, f x = 0)) → a = 3 :=
  sorry

-- 3. Inequality proof
theorem inequality_proof (n : ℕ) (h : 2 ≤ n) : 
  ∑ k in (finset.range n).filter (λ k, 2 ≤ k), (1 / k^3 : ℝ) <
  1 / 3 - 1 / (2 * n + 1) :=
  sorry

end max_value_f_when_a_4_value_of_a_two_zeros_inequality_proof_l503_503261


namespace polynomial_solution_l503_503582

noncomputable def p (x : ℝ) : ℝ := (1 + Real.sqrt 109) / 2

theorem polynomial_solution :
  ∀ x : ℝ, p(x^2) - p(x^2 - 3) = (p(x))^2 + 27 :=
  by
    intros
    have h : p(x) = (1 + Real.sqrt 109) / 2 := by sorry
    rw [h]
    sorry

end polynomial_solution_l503_503582


namespace sum_of_solutions_l503_503491

theorem sum_of_solutions : ∑ x in {x : ℝ | |x^2 - 15 * x + 58| = 3}, x = 30 :=
by
  sorry

end sum_of_solutions_l503_503491


namespace parabola_distance_to_xaxis_l503_503397

open Real

noncomputable def parabola (x y : ℝ) : Prop :=
  x^2 = 4 * y

def focus : ℝ × ℝ := (0, 1)
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_distance_to_xaxis
  (x y : ℝ)
  (h₁ : parabola x y)
  (h₂ : distance (x, y) focus = 8) :
  y = 7 :=
sorry

end parabola_distance_to_xaxis_l503_503397


namespace final_sum_l503_503955

theorem final_sum (n : ℕ)
  (h : n = 51)
  (start_values : list ℤ)
  (h_start : start_values = [2, 0, -2])
  (operation_2 : ℤ → ℤ := λ x, 2 * x)
  (operation_0 : ℤ → ℤ := λ x, x * x)
  (operation_neg2 : ℤ → ℤ := λ x, -x)
  (next_value : ℕ → ℤ → ℤ := λ i x, match i, x with
    | 0, x => operation_2 x
    | 1, x => operation_0 x
    | 2, x => operation_neg2 x
    | _, x => x
    end) :
  (list.sum (list.map (λ x, (next_value (51 % 3) (next_value (50 % 3) ... (next_value (0 % 3) x)))) start_values)) = 2^52 + 2 := sorry

end final_sum_l503_503955


namespace find_length_AB_l503_503337

noncomputable def length_of_AB (DE DF : ℝ) (AC : ℝ) : ℝ :=
  (AC * DE) / DF

theorem find_length_AB (DE DF AC : ℝ) (pro1 : DE = 9) (pro2 : DF = 17) (pro3 : AC = 10) :
    length_of_AB DE DF AC = 90 / 17 :=
  by
    rw [pro1, pro2, pro3]
    unfold length_of_AB
    norm_num

end find_length_AB_l503_503337


namespace angle_equality_l503_503692

variables {P A B E D Q C H : Type*}
-- Let ∆PAB be an acute-angled triangle
-- Let there be a semicircle with diameter AB intersecting PA at E and PB at D
-- AB intersects ED at Q
-- AD and BE intersect at C
-- PC intersects AB at H
-- O is the center of the semicircle

theorem angle_equality
  (h_triangle : ∀ P A B : Type*, acute_triangle P A B)
  (h_segments : semicircle_with_diameter AB)
  (h_inter_PA_E : intersects PA E)
  (h_inter_PB_D : intersects PB D)
  (h_inter_AB_Q : intersects AB Q)
  (h_inter_AD_BE_C : intersects AD BE C)
  (h_inter_PC_H : intersects PC H)
  : ∠OEH = ∠ODH ∧ ∠ODH = ∠EQO :=
sorry

end angle_equality_l503_503692


namespace part_a_proof_l503_503223

-- Define given conditions
variables (A B M : Point) [Plane]
variables (AMCD MBEF : Square) 
variables (circumcircle1 circumcircle2: Circle)
variables (N: Point)
variables (AF BC: Line)

-- Define the relationships
axioms
  (on_segment_AB : M ∈ segment A B)
  (square_AMCD : square AMCD A M)
  (square_MBEF : square MBEF M B)
  (circumcircle1_def : circumcircle1 = circumcircle AMCD)
  (circumcircle2_def : circumcircle2 = circumcircle MBEF)
  (intersect_at_M_N : N ∈ circumcircle1 ∧ N ∈ circumcircle2 ∧ M ∈ circumcircle1 ∧ M ∈ circumcircle2)
  (line_AF : line A F = AF)
  (line_BC : line B C = BC)

-- State the theorem
theorem part_a_proof (A B M F C N : Point) :
  intersect (line A F) (line B C) N :=
sorry

end part_a_proof_l503_503223


namespace sum_of_coefficients_l503_503559

theorem sum_of_coefficients : 
  (∑ i in Finset.range (11), (Binomial 10 i * (-2)^i)) = 1 :=
by
  sorry

end sum_of_coefficients_l503_503559


namespace find_r_in_geometric_sequence_l503_503320

noncomputable def sum_of_geometric_sequence (n : ℕ) (r : ℝ) := 3^n + r

theorem find_r_in_geometric_sequence:
  ∃ r : ℝ, 
  (∀ n : ℕ, n >= 1 → sum_of_geometric_sequence n r = 3^n +  r) ∧
  (∀ n : ℕ, n >= 2 → 
    let S_n := sum_of_geometric_sequence n r in
    let S_n_minus_1 := sum_of_geometric_sequence (n - 1) r in
    let a_n := S_n - S_n_minus_1 in
    a_n = 2 * 3^(n - 1)) ∧
  (∃ a1 : ℝ, a1 = 3 + r ∧ a1 * 3 = 6) ∧
  r = -1 :=
by sorry

end find_r_in_geometric_sequence_l503_503320


namespace condition_a_condition_b_l503_503394

/-- A given figure can be cut into at most four parts and reassembled into a square. -/
theorem condition_a : ∃ (parts : List (Set (ℝ × ℝ))), parts.length ≤ 4 ∧ 
                     (∀ p ∈ parts, measurable_set p) ∧ 
                     (⋃₀ parts = given_figure) ∧ 
                     (∃ f : ℝ × ℝ → ℝ × ℝ, bijective f ∧ 
                     ∀ p ∈ parts, isometry (f '' p) square) :=
sorry

/-- A given figure can be cut into at most five triangular parts and reassembled into a square. -/
theorem condition_b : ∃ (parts : List (Set (ℝ × ℝ))), parts.length = 5 ∧ 
                     (∀ p ∈ parts, measurable_set p ∧ is_triangle p) ∧ 
                     (⋃₀ parts = given_figure) ∧ 
                     (∃ f : ℝ × ℝ → ℝ × ℝ, bijective f ∧ 
                     ∀ p ∈ parts, isometry (f '' p) square) :=
sorry

end condition_a_condition_b_l503_503394


namespace unique_pos_int_sum_of_digits_l503_503204

def unique_positive_integer (n : ℕ) : Prop :=
  0 < n ∧ (Real.log 3 (Real.log 27 n) = Real.log 3 (Real.log 9 n))

theorem unique_pos_int_sum_of_digits :
  ∃! (n : ℕ), unique_positive_integer n ∧ (n.digits.sum = 1) :=
sorry

end unique_pos_int_sum_of_digits_l503_503204


namespace sum_of_other_endpoint_coordinates_l503_503400

theorem sum_of_other_endpoint_coordinates {x y : ℝ} :
  let P1 := (1, 2)
  let M := (5, 6)
  let P2 := (x, y)
  (M.1 = (P1.1 + P2.1) / 2 ∧ M.2 = (P1.2 + P2.2) / 2) → (x + y) = 19 :=
by
  intros P1 M P2 h
  sorry

end sum_of_other_endpoint_coordinates_l503_503400


namespace range_of_t_l503_503634

noncomputable theory

def g (x : ℝ) := Real.log x + (3 / (4 * x)) - (1 / 4) * x - 1
def f (x t : ℝ) := x^2 - 2 * t * x + 4

theorem range_of_t (t : ℝ) :
  (∀ x1 ∈ (Set.Ioo 0 2), ∃ x2 ∈ (Set.Icc 1 2), g x1 ≥ f x2 t) ↔ t ≥ 17 / 8 :=
  sorry

end range_of_t_l503_503634


namespace closest_integer_ratio_l503_503248

theorem closest_integer_ratio (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_gt : a > b) 
    (h_mean : (a + b) / 2 = 3 * Real.sqrt(a * b)) : Int.ceil (a / b) = 34 := 
sorry

end closest_integer_ratio_l503_503248


namespace triangle_area_example_l503_503841

def isosceles_triangle_area (a b c : ℝ) (h1 : a = b) (h2 : c = 24) : ℝ :=
  let half_c := c / 2
  let h := real.sqrt (a ^ 2 - half_c ^ 2)
  0.5 * c * h

theorem triangle_area_example :
  isosceles_triangle_area 13 13 24 13 rfl :=
by
  simp only [isosceles_triangle_area, real.sqrt, pow_two, bit0, add_zero, mul_one]
  norm_num
  sorry -- The proof will be inserted here.

end triangle_area_example_l503_503841


namespace projection_of_a_in_direction_of_b_eq_neg_half_l503_503649

open RealEuclideanSpace

namespace ProjectionProof

variables {a b : ℝ²}

-- Condition definitions
def a_norm : ℝ := 1
def b_norm : ℝ := 2
def a_plus_b_norm : ℝ := sqrt 3

-- Main statement
theorem projection_of_a_in_direction_of_b_eq_neg_half
  (ha : ∥a∥ = a_norm)
  (hb : ∥b∥ = b_norm)
  (hab : ∥a + b∥ = a_plus_b_norm) :
  (a • b) / ∥b∥ = -1 / 2 :=
by
  sorry

end ProjectionProof

end projection_of_a_in_direction_of_b_eq_neg_half_l503_503649


namespace complementary_angle_difference_l503_503800

theorem complementary_angle_difference (x : ℝ) (h : 3 * x + 5 * x = 90) : 
    abs ((5 * x) - (3 * x)) = 22.5 :=
by
  -- placeholder proof
  sorry

end complementary_angle_difference_l503_503800


namespace solve_equation_l503_503766

theorem solve_equation :
  ∃ x : ℝ, x = 25 ∧ (sqrt (1 + sqrt (2 + x^2)) = real.cbrt (3 + sqrt x)) :=
begin
  use 25,
  split,
  refl,
  sorry
end

end solve_equation_l503_503766


namespace sin_alpha_plus_beta_l503_503699

theorem sin_alpha_plus_beta (α β : ℝ) (A : ℝ × ℝ) (hA : A = (1, 2)) (hβ : β = α + π/2) :
  let r := Real.sqrt (1^2 + 2^2),
      sin_α := 2 / r,
      cos_α := 1 / r in
  sin (α + β) = -3/5 :=
by
  -- Defining the necessary constants
  let r := Real.sqrt (1^2 + 2^2)
  let sin_α := 2 / r
  let cos_α := 1 / r
  -- Defining the given condition for β
  have hβ : β = α + π/2 := sorry
  -- We skip the proof here because we're focusing on the statement setup
  sorry

end sin_alpha_plus_beta_l503_503699


namespace max_value_carried_alice_max_value_l503_503104

structure Rock :=
  (weight : ℕ)
  (value : ℕ)
  (available : ℕ)

def rocks : list Rock :=
  [ {weight := 6, value := 18, available := 15},
    {weight := 3, value := 9, available := 15},
    {weight := 2, value := 6, available := 15} ]

def max_carry_weight : ℕ := 21

theorem max_value_carried (rocks : list Rock) (max_carry_weight : ℕ) : Prop :=
  ∃ (r1 r2 r3 : ℕ),
    r1 * (rocks.nth_le 0 (by simp)).weight + r2 * (rocks.nth_le 1 (by simp)).weight + r3 * (rocks.nth_le 2 (by simp)).weight ≤ max_carry_weight ∧
    r1 * (rocks.nth_le 0 (by simp)).value + r2 * (rocks.nth_le 1 (by simp)).value + r3 * (rocks.nth_le 2 (by simp)).value = 63

theorem alice_max_value : max_value_carried rocks max_carry_weight :=
  sorry

end max_value_carried_alice_max_value_l503_503104


namespace solve_system_of_inequalities_l503_503767

variable (x : Real)

theorem solve_system_of_inequalities :
  (x - 1 ≤ x / 2 ∧ x + 2 > 3 * (x - 2)) → x ≤ 2 :=
by
  intro h
  cases h with h₁ h₂
  sorry

end solve_system_of_inequalities_l503_503767


namespace log_normal_expected_value_log_normal_variance_l503_503339

def log_normal_pdf (x a σ : ℝ) : ℝ :=
  if x ≤ 0 then 0 else (1 / (σ * x * Real.sqrt (2 * Real.pi))) * Real.exp (-(Real.log x - a)^2 / (2 * σ^2))

def expected_value (X : ℝ → ℝ) (a σ : ℝ) : ℝ :=
  ∫ x in 0..∞, x * log_normal_pdf x a σ

def variance (X : ℝ → ℝ) (a σ : ℝ) : ℝ :=
  let EX := expected_value X a σ
  in (∫ x in 0..∞, x^2 * log_normal_pdf x a σ) - EX^2

theorem log_normal_expected_value (a σ : ℝ) :
  expected_value (λ x, x) a σ = Real.exp (a + σ^2 / 2) :=
sorry

theorem log_normal_variance (a σ : ℝ) :
  variance (λ x, x) a σ = Real.exp (2 * a) * (Real.exp (2 * σ^2) - Real.exp σ^2) :=
sorry

end log_normal_expected_value_log_normal_variance_l503_503339


namespace sin_pi_minus_α_minus_sin_pi_over_2_plus_α_value_set_S_of_angle_α_l503_503618

noncomputable def α_satisfying_conditions : Prop :=
  ∃ α : ℝ, α ∈ { α | cos α = 1 / 2 ∧ sin α = sqrt 3 / 2 }

-- Problem Part (1)
theorem sin_pi_minus_α_minus_sin_pi_over_2_plus_α_value : α_satisfying_conditions → 
  (∀ α : ℝ, cos α = 1 / 2 ∧ sin α = sqrt 3 / 2) →
    (sin (π - α) - sin (π / 2 + α) = (sqrt 3 - 1) / 2) := 
by
  intro hα_cond h_trig_defs
  sorry

-- Problem Part (2)
theorem set_S_of_angle_α : α_satisfying_conditions → 
  let S := { α | ∃ k : ℤ, α = 2 * k * π + π / 3 } in
  α ∈ S :=
by
  simp only [α_satisfying_conditions]
  intro hα_cond
  sorry

end sin_pi_minus_α_minus_sin_pi_over_2_plus_α_value_set_S_of_angle_α_l503_503618


namespace geom_sequence_sum_l503_503318

theorem geom_sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (r : ℤ) 
    (h1 : ∀ n : ℕ, n ≥ 1 → S n = 3^n + r) 
    (h2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)) 
    (h3 : a 1 = S 1) :
  r = -1 := 
sorry

end geom_sequence_sum_l503_503318


namespace volume_of_tetrahedron_l503_503569

theorem volume_of_tetrahedron 
  (angle_ABC_BCD : Real) 
  (area_ABC area_BCD : Real) 
  (side_BC : Real) 
  (h : angle_ABC_BCD = pi / 4) 
  (h1 : area_ABC = 150) 
  (h2 : area_BCD = 100) 
  (h3 : side_BC = 12) : 
  ∃ (volume : Real), volume = (1250 * sqrt 2) / 3 :=
sorry

end volume_of_tetrahedron_l503_503569


namespace roots_of_equation_l503_503804

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l503_503804


namespace parabola_ellipse_intersection_fixed_point_pass_l503_503277

def parabola_standard_form (p : ℝ) (hp : p > 0) (C : ℝ → ℝ → Prop) : Prop :=
∀ x y, C x y ↔ y^2 = 2 * p * x

def intersection_points (C : ℝ → ℝ → Prop) (C' : ℝ → ℝ → Prop) : Prop :=
∃ x1 y1 x2 y2, C x1 y1 ∧ C x2 y2 ∧ C' x1 y1 ∧ C' x2 y2 ∧ y1 > 0 ∧ y2 < 0

theorem parabola_ellipse_intersection (p : ℝ) (hp : p > 0) :
  parabola_standard_form p hp (λ x y, y^2 = 2 * p * x) →
  intersection_points (λ x y, y^2 = 2 * p * x) 
                     (λ x y, (x^2) / 4 + (15 * y^2) / 16 = 1) →
  p = 1 := sorry

def line_through_points (A B : ℝ × ℝ) : Prop :=
∃ (k m : ℝ), k ≠ 0 ∧ m ≠ 0 ∧
  A.2 = k * A.1 ∧ A.2^2 = 2 * A.1 ∧
  B.2 = m * B.1 ∧ B.2^2 = 2 * B.1

def line_inclination_fixed_point (θ : ℝ) (tθ : tan θ = 2) (A B : ℝ × ℝ) : Prop :=
∃ α β, α + β = θ ∧ ∃ k m,
k = (2 - m) / (1 + 2 * m) ∧ A = (2 / k^2, 2 / k) ∧
B = (2 / m^2, 2 / m) →
  (λ x y, ∃ m, y = (m * (2 - m))/(2 * (m^2 + 1)) * (x + 2) + 1) (-2, 1)

theorem fixed_point_pass (θ : ℝ) (tθ : tan θ = 2) :
  ∀ A B, A ≠ B →
  line_through_points A B →
  line_inclination_fixed_point θ tθ A B :=
sorry

end parabola_ellipse_intersection_fixed_point_pass_l503_503277


namespace triangle_PQR_incenter_perimeter_l503_503840

theorem triangle_PQR_incenter_perimeter (PQ QR PR : ℝ) 
  (hPQ : PQ = 10) (hQR : QR = 20) (hPR : PR = 16) :
  let I := incenter (triangle P Q R),
      X := line_through I ∥ QR ∩ PQ,
      Y := line_through I ∥ QR ∩ PR in
  perimeter (triangle P X Y) = 26 :=
by
  sorry

end triangle_PQR_incenter_perimeter_l503_503840


namespace mode_and_median_of_data_set_l503_503683

theorem mode_and_median_of_data_set :
  let data := [176, 178, 178, 180, 182, 185, 189]
  in mode data = 178 ∧ median data = 180 :=
by
  sorry

end mode_and_median_of_data_set_l503_503683


namespace tan_alpha_proof_l503_503602

def right_triangle (ABC : Type) := sorry -- definition of a right triangle

noncomputable def hypotenuse_div (BC : ℝ) (n : ℕ) (a : ℝ) := 
  BC = a ∧ n % 2 = 1 -- BC is divided into n equal parts, n is odd, and BC has length a

def altitude (h : ℝ) := h -- altitude of the right triangle

def angle (α : ℝ) := α -- angle formed

theorem tan_alpha_proof (ABC : Type) (BC : ℝ) (a : ℝ) (n : ℕ) (h : ℝ) (α : ℝ)
  [ht : right_triangle ABC]
  [hd : hypotenuse_div BC n a]
  [alt : altitude h]
  [ang : angle α] :
  tan α = 4 * n * h / ((n^2 - 1) * a) := 
by
  sorry

end tan_alpha_proof_l503_503602


namespace vasya_figure_cells_l503_503854

theorem vasya_figure_cells (n : ℕ) : 
  (∀ (m : ℕ), m ∈ (2 * 2) ∨ m ∈ 4 → n ≥ 16 ∧ n % 8 = 0) :=
sorry

end vasya_figure_cells_l503_503854


namespace squares_sharing_vertices_with_PQRS_l503_503368

-- Definition: A square PQRS
structure Square (P Q R S : Type) := 
  (side : ℝ)

-- Condition: PQRS is a square
variables (P Q R S : Type) (PQRS : Square P Q R S)

-- Theorem: There are 8 squares in the same plane as PQRS that share two vertices with PQRS
theorem squares_sharing_vertices_with_PQRS : 
  ∃ (n : ℕ), n = 8 ∧ 
  (∃ (squares : list (Square P Q R S)), 
    ∀ (square ∈ squares), (∃ (P Q : P), P ∈ square.verts ∧ Q ∈ square.verts)) := 
by
  sorry

end squares_sharing_vertices_with_PQRS_l503_503368


namespace no_multiples_of_7_l503_503989

-- Define the sequence of 2008 integers
def seq (a : ℕ → ℤ) := ∀ i : ℕ, i < 2008 → ℤ

-- Condition: The sum of any subset is not a multiple of 2009
def sum_not_multiple_of_2009 (a : ℕ → ℤ) : Prop :=
  ∀ (s : finset ℕ) (H : ∀ i ∈ s, i < 2008), (s.sum a) % 2009 ≠ 0

-- Theorem: None of the 2008 integers are multiples of 7
theorem no_multiples_of_7 (a : ℕ → ℤ) (H : sum_not_multiple_of_2009 a) : 
  ∀ i : ℕ, i < 2008 → a i % 7 ≠ 0 :=
sorry

end no_multiples_of_7_l503_503989


namespace new_machine_rate_l503_503518

def old_machine_rate : ℕ := 100
def total_bolts : ℕ := 500
def time_hours : ℕ := 2

theorem new_machine_rate (R : ℕ) : 
  (old_machine_rate * time_hours + R * time_hours = total_bolts) → 
  R = 150 := 
by
  sorry

end new_machine_rate_l503_503518


namespace axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l503_503332

-- (1) Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, (y = x^2 - 2*t*x + 1) → (x = t) := sorry

-- (2) Comparison of m and n
theorem compare_m_n (t m n : ℝ) :
  (t - 2)^2 - 2*t*(t - 2) + 1 = m*1 →
  (t + 3)^2 - 2*t*(t + 3) + 1 = n*1 →
  n > m := sorry

-- (3) Range of t for y₁ ≤ y₂
theorem range_of_t_for_y1_leq_y2 (t x1 x2 y1 y2 : ℝ) :
  (-1 ≤ x1) → (x1 < 3) → (x2 = 3) → 
  (y1 = x1^2 - 2*t*x1 + 1) → 
  (y2 = x2^2 - 2*t*x2 + 1) → 
  y1 ≤ y2 →
  t ≤ 1 := sorry

-- (4) Maximum value of t
theorem maximum_value_of_t (t y1 y2 : ℝ) :
  (y1 = (t + 1)^2 - 2*t*(t + 1) + 1) →
  (y2 = (2*t - 4)^2 - 2*t*(2*t - 4) + 1) →
  y1 ≥ y2 →
  t = 5 := sorry

end axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l503_503332


namespace parallel_planes_l503_503152

-- Definitions as per given conditions
def cond1 (α β : plane) : Prop := 
  ∃ l : line, l ⊆ α ∧ l ∥ β

def cond2 (α β : plane) : Prop := 
  ∀ l : line, l ⊆ α → l ∥ β

def cond3 (α β : plane) (a : line) (b : line) : Prop := 
  a ⊆ α ∧ b ⊆ β ∧ a ∥ β ∧ b ∥ α

def cond4 (α β : plane) (a : line) (b : line) : Prop := 
  a ∥ b ∧ a ⊥ α ∧ b ⊥ β

-- Theorem statement: conditions 2 and 4 imply parallelism of planes α and β
theorem parallel_planes (α β : plane) (a b : line) :
  (cond2 α β) ∨ (cond4 α β a b) → α ∥ β :=
by sorry

end parallel_planes_l503_503152


namespace P_sufficient_but_not_necessary_for_Q_l503_503598

def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) := 
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l503_503598


namespace john_saves_1200_yearly_l503_503720

noncomputable def former_rent_per_month (sq_ft_cost : ℝ) (sq_ft : ℝ) : ℝ :=
  sq_ft_cost * sq_ft

noncomputable def new_rent_per_month (total_cost : ℝ) (roommates : ℝ) : ℝ :=
  total_cost / roommates

noncomputable def monthly_savings (former_rent : ℝ) (new_rent : ℝ) : ℝ :=
  former_rent - new_rent

noncomputable def annual_savings (monthly_savings : ℝ) : ℝ :=
  monthly_savings * 12

theorem john_saves_1200_yearly :
  let former_rent := former_rent_per_month 2 750
  let new_rent := new_rent_per_month 2800 2
  let monthly_savings := monthly_savings former_rent new_rent
  annual_savings monthly_savings = 1200 := 
by 
  sorry

end john_saves_1200_yearly_l503_503720


namespace population_change_l503_503048

theorem population_change :
  let initial_population := 1 in
  let change_factor := (6 / 5) * (9 / 10) * (13 / 10) * (4 / 5) in
  let net_change := (change_factor - 1) * 100 in
  net_change = 12 :=
by
  let initial_population := 1
  let change_factor := (6 / 5) * (9 / 10) * (13 / 10) * (4 / 5)
  let net_change := (change_factor - 1) * 100
  show net_change = 12 from sorry

end population_change_l503_503048


namespace central_angle_constant_l503_503306

theorem central_angle_constant {r θ : ℝ} (h : r > 0) (hθ : θ > 0) :
  let s := r * θ,
      new_r := 2 * r,
      new_s := 2 * s in
  new_s / new_r = θ :=
by 
  sorry

end central_angle_constant_l503_503306


namespace cuberoot_sum_eq_three_l503_503731

theorem cuberoot_sum_eq_three (x r s : ℝ) (h : (∛x) + ∛(30 - x) = 3) 
  (hx : x = (r - Real.sqrt s)^3) : r + s = 102 :=
sorry

end cuberoot_sum_eq_three_l503_503731


namespace profit_percentage_is_23_16_l503_503516

   noncomputable def cost_price (mp : ℝ) : ℝ := 95 * mp
   noncomputable def selling_price (mp : ℝ) : ℝ := 120 * (mp - (0.025 * mp))
   noncomputable def profit_percent (cp sp : ℝ) : ℝ := ((sp - cp) / cp) * 100

   theorem profit_percentage_is_23_16 
     (mp : ℝ) (h_mp_gt_zero : mp > 0) : 
       profit_percent (cost_price mp) (selling_price mp) = 23.16 :=
   by 
     sorry
   
end profit_percentage_is_23_16_l503_503516


namespace perpendicular_tangent_l503_503059

noncomputable def f (x a : ℝ) := (x + a) * Real.exp x -- Defines the function

theorem perpendicular_tangent (a : ℝ) : 
  ∀ (tangent_slope perpendicular_slope : ℝ), 
  (tangent_slope = 1) → 
  (perpendicular_slope = -1) →
  tangent_slope = Real.exp 0 * (a + 1) →
  tangent_slope + perpendicular_slope = 0 → 
  a = 0 := by 
  intros tangent_slope perpendicular_slope htangent hperpendicular hderiv hperpendicular_slope
  sorry

end perpendicular_tangent_l503_503059


namespace l1_perpendicular_l2_l1_parallel_l2_l1_inclination_45_degrees_l503_503245

-- Condition definitions for the lines
def line1 := (A : ℝ × ℝ) → (B : ℝ × ℝ) → (l1_slope : ℝ) :=
  ∀ m : ℝ, A = (m, 1) ∧ B = (-3, 4) → l1_slope = (4 - 1) / (-3 - m)

def line2 := (C : ℝ × ℝ) → (D : ℝ × ℝ) → (l2_slope : ℝ) :=
  ∀ m : ℝ, C = (1, m) ∧ D = (-1, m + 1) → l2_slope = (m + 1 - m) / (-1 - 1)

-- Questions rewritten in Lean 4 statements
theorem l1_perpendicular_l2 (m : ℝ) (l1_slope l2_slope : ℝ) (h1 : line1 (m, 1) (-3, 4) l1_slope) (h2 : line2 (1, m) (-1, m + 1) l2_slope) :
  l1_slope * l2_slope = -1 → m = -9 / 2 := sorry

theorem l1_parallel_l2 (m : ℝ) (l1_slope l2_slope : ℝ) (h1 : line1 (m, 1) (-3, 4) l1_slope) (h2 : line2 (1, m) (-1, m + 1) l2_slope) :
  l1_slope = l2_slope → m = 3 := sorry

theorem l1_inclination_45_degrees (m : ℝ) (l1_slope : ℝ) (h1 : line1 (m, 1) (-3, 4) l1_slope) :
  l1_slope = 1 → m = -6 := sorry

end l1_perpendicular_l2_l1_parallel_l2_l1_inclination_45_degrees_l503_503245


namespace prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l503_503851

def num_outcomes := 36

def same_points_events := 6
def less_than_seven_events := 15
def greater_than_or_equal_eleven_events := 3

def prob_same_points := (same_points_events : ℚ) / num_outcomes
def prob_less_than_seven := (less_than_seven_events : ℚ) / num_outcomes
def prob_greater_or_equal_eleven := (greater_than_or_equal_eleven_events : ℚ) / num_outcomes

theorem prob_same_points_eq : prob_same_points = 1 / 6 := by
  sorry

theorem prob_less_than_seven_eq : prob_less_than_seven = 5 / 12 := by
  sorry

theorem prob_greater_or_equal_eleven_eq : prob_greater_or_equal_eleven = 1 / 12 := by
  sorry

end prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l503_503851


namespace roots_of_equation_l503_503807

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l503_503807


namespace parallel_lines_conditions_l503_503214

-- Define the Points and Conditions
structure Triangle where
  A B C : Point

structure Triangle1 where
  A1 B1 C1 : Point

structure Triangle2 where
  A2 B2 C2 : Point

variables {n : ℝ} (T : Triangle) (T1 : Triangle1) (T2 : Triangle2)

-- Given conditions as definitions
def condition1 (T : Triangle) (T1 : Triangle1) (n : ℝ) :=
  (T.AC_1 / T.C_1B = 1 / n) ∧
  (T.CA_1 / T.A_1B = 1 / n) ∧
  (T.BA_1 / T.A_1C = 1 / n)

def condition2 (T1 : Triangle1) (T2 : Triangle2) (n : ℝ) :=
  (T1.A_1C_2 / T2.C_2B_1 = n) ∧
  (T1.B_1A_2 / T2.A_2C_1 = n) ∧
  (T1.C_1B_2 / T2.B_2A_1 = n)

-- The theorem to prove
theorem parallel_lines_conditions (T : Triangle) (T1 : Triangle1) (T2 : Triangle2) (n : ℝ) :
  condition1 T T1 n →
  condition2 T1 T2 n →
  (Parallel T2.A2C2 T.AC) ∧
  (Parallel T2.C2B2 T.CB) ∧
  (Parallel T2.B2A2 T.BA) :=
by sorry

end parallel_lines_conditions_l503_503214


namespace inequality_holds_l503_503301

theorem inequality_holds 
  (a b c : ℝ) 
  (h1 : a > 0)
  (h2 : b < 0) 
  (h3 : b > c) : 
  (a / (c^2)) > (b / (c^2)) :=
by
  sorry

end inequality_holds_l503_503301


namespace female_student_weight_l503_503307

theorem female_student_weight (x : ℝ) (e : ℝ) (hx : x = 160) (he : |e| ≤ 4) :
    0.85 * x - 88 + e ≥ 44 := by
    -- sorry

end female_student_weight_l503_503307


namespace fruit_difference_l503_503748

/-- Mr. Connell harvested 60 apples and 3 times as many peaches. The difference 
    between the number of peaches and apples is 120. -/
theorem fruit_difference (apples peaches : ℕ) (h1 : apples = 60) (h2 : peaches = 3 * apples) :
  peaches - apples = 120 :=
sorry

end fruit_difference_l503_503748


namespace maximum_value_of_f_l503_503517

def operation (a b : ℝ) : ℝ := if a ≤ b then a else b

def f (x : ℝ) : ℝ := operation (Real.sin x) (Real.cos x)

theorem maximum_value_of_f : ∃ x : ℝ, f x = (Real.sqrt 2) / 2 := sorry

end maximum_value_of_f_l503_503517


namespace find_circle_equation_l503_503196

-- Define the conditions
def line (x y : ℝ) : Prop := x + y = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Define the proof problem
theorem find_circle_equation (x y : ℝ) :
  (line x y) →
  (∃ (x1 y1 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1) →
  ∃ (h k r : ℝ), h = -3 ∧ k = 3 ∧ r = √10 ∧ (x + h)^2 + (y + k)^2 = r^2 :=
begin
  sorry
end

end find_circle_equation_l503_503196


namespace tangent_line_equations_l503_503251

theorem tangent_line_equations (x y : ℝ) (hP : (2, -2) : ℝ × ℝ)
  (hC : y = (1/3) * x^3 - x) :
  (y = -x) ∨ (y = 8 * x - 18) :=
sorry

end tangent_line_equations_l503_503251
