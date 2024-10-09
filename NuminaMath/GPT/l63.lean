import Mathlib

namespace find_new_person_age_l63_6375

variables (A X : ℕ) -- A is the original average age, X is the age of the new person

def original_total_age (A : ℕ) := 10 * A
def new_total_age (A X : ℕ) := 10 * (A - 3)

theorem find_new_person_age (A : ℕ) (h : new_total_age A X = original_total_age A - 45 + X) : X = 15 :=
by
  sorry

end find_new_person_age_l63_6375


namespace biker_distance_and_speed_l63_6328

variable (D V : ℝ)

theorem biker_distance_and_speed (h1 : D / 2 = V * 2.5)
                                  (h2 : D / 2 = (V + 2) * (7 / 3)) :
  D = 140 ∧ V = 28 :=
by
  sorry

end biker_distance_and_speed_l63_6328


namespace tangent_ellipse_hyperbola_l63_6352

theorem tangent_ellipse_hyperbola {m : ℝ} :
    (∀ x y : ℝ, x^2 + 9*y^2 = 9 → x^2 - m*(y + 1)^2 = 1 → false) →
    m = 72 :=
sorry

end tangent_ellipse_hyperbola_l63_6352


namespace Ivan_bought_10_cards_l63_6358

-- Define variables and conditions
variables (x : ℕ) -- Number of Uno Giant Family Cards bought
def original_price : ℕ := 12
def discount_per_card : ℕ := 2
def discounted_price := original_price - discount_per_card
def total_paid : ℕ := 100

-- Lean 4 theorem statement
theorem Ivan_bought_10_cards (h : discounted_price * x = total_paid) : x = 10 := by
  -- proof goes here
  sorry

end Ivan_bought_10_cards_l63_6358


namespace marcus_leah_together_l63_6313

def num_games_with_combination (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_games_together (total_players players_per_game : ℕ) (games_with_each_combination: ℕ) : ℕ :=
  total_players / players_per_game * games_with_each_combination

/-- Prove that Marcus and Leah play 210 games together. -/
theorem marcus_leah_together :
  let total_players := 12
  let players_per_game := 6
  let total_games := num_games_with_combination total_players players_per_game
  let marc_per_game := total_games / 2
  let together_pcnt := 5 / 11
  together_pcnt * marc_per_game = 210 :=
by
  sorry

end marcus_leah_together_l63_6313


namespace tangent_line_at_origin_l63_6334

noncomputable def f (x : ℝ) := Real.log (1 + x) + x * Real.exp (-x)

theorem tangent_line_at_origin : 
  ∀ (x : ℝ), (1 : ℝ) * x + (0 : ℝ) = 2 * x := 
sorry

end tangent_line_at_origin_l63_6334


namespace corn_harvest_l63_6348

theorem corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) (total_bushels : ℕ) :
  rows = 5 →
  stalks_per_row = 80 →
  stalks_per_bushel = 8 →
  total_bushels = (rows * stalks_per_row) / stalks_per_bushel →
  total_bushels = 50 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, mul_comm 5 80] at h4
  norm_num at h4
  exact h4

end corn_harvest_l63_6348


namespace least_common_multiple_of_wang_numbers_l63_6326

noncomputable def wang_numbers (n : ℕ) : List ℕ :=
  -- A function that returns the wang numbers in the set from 1 to n
  sorry

noncomputable def LCM (list : List ℕ) : ℕ :=
  -- A function that computes the least common multiple of a list of natural numbers
  sorry

theorem least_common_multiple_of_wang_numbers :
  LCM (wang_numbers 100) = 10080 :=
sorry

end least_common_multiple_of_wang_numbers_l63_6326


namespace sally_picked_peaches_l63_6343

variable (p_initial p_current p_picked : ℕ)

theorem sally_picked_peaches (h1 : p_initial = 13) (h2 : p_current = 55) :
  p_picked = p_current - p_initial → p_picked = 42 :=
by
  intros
  sorry

end sally_picked_peaches_l63_6343


namespace cannot_determine_right_triangle_l63_6303

/-- Proof that the condition \(a^2 = 5\), \(b^2 = 12\), \(c^2 = 13\) cannot determine that \(\triangle ABC\) is a right triangle. -/
theorem cannot_determine_right_triangle (a b c : ℝ) (ha : a^2 = 5) (hb : b^2 = 12) (hc : c^2 = 13) : 
  ¬(a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := 
by
  sorry

end cannot_determine_right_triangle_l63_6303


namespace parabola_focus_coords_l63_6312

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus coordinates
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- The math proof problem statement
theorem parabola_focus_coords :
  ∀ x y, parabola x y → focus x y :=
by
  intros x y hp
  sorry

end parabola_focus_coords_l63_6312


namespace youseff_distance_l63_6324

theorem youseff_distance (x : ℕ) 
  (walk_time_per_block : ℕ := 1)
  (bike_time_per_block_secs : ℕ := 20)
  (time_difference : ℕ := 12) :
  (x : ℕ) = 18 :=
by
  -- walking time
  let walk_time := x * walk_time_per_block
  
  -- convert bike time per block to minutes
  let bike_time_per_block := (bike_time_per_block_secs : ℚ) / 60

  -- biking time
  let bike_time := x * bike_time_per_block

  -- set up the equation for time difference
  have time_eq := walk_time - bike_time = time_difference
  
  -- from here, the actual proof steps would follow, 
  -- but we include "sorry" as a placeholder since the focus is on the statement.
  sorry

end youseff_distance_l63_6324


namespace trains_crossing_time_l63_6371

noncomputable def length_first_train : ℝ := 120
noncomputable def length_second_train : ℝ := 160
noncomputable def speed_first_train_kmph : ℝ := 60
noncomputable def speed_second_train_kmph : ℝ := 40
noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

noncomputable def speed_first_train : ℝ := kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train : ℝ := kmph_to_mps speed_second_train_kmph
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def crossing_time : ℝ := total_distance / relative_speed

theorem trains_crossing_time :
  crossing_time = 10.08 := by
  sorry

end trains_crossing_time_l63_6371


namespace polynomial_smallest_e_l63_6362

theorem polynomial_smallest_e :
  ∃ (a b c d e : ℤ), (a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ∧ a ≠ 0 ∧ e > 0 ∧ (x + 3) * (x - 6) * (x - 10) * (2 * x + 1) = 0) 
  ∧ e = 180 :=
by
  sorry

end polynomial_smallest_e_l63_6362


namespace rope_subdivision_length_l63_6390

theorem rope_subdivision_length 
  (initial_length : ℕ) 
  (num_parts : ℕ) 
  (num_subdivided_parts : ℕ) 
  (final_subdivision_factor : ℕ) 
  (initial_length_eq : initial_length = 200) 
  (num_parts_eq : num_parts = 4) 
  (num_subdivided_parts_eq : num_subdivided_parts = num_parts / 2) 
  (final_subdivision_factor_eq : final_subdivision_factor = 2) :
  initial_length / num_parts / final_subdivision_factor = 25 := 
by 
  sorry

end rope_subdivision_length_l63_6390


namespace remainder_of_exp_l63_6330

theorem remainder_of_exp (x : ℝ) :
  (x + 1) ^ 2100 % (x^4 - x^2 + 1) = x^2 := 
sorry

end remainder_of_exp_l63_6330


namespace gcd_45736_123456_l63_6322

theorem gcd_45736_123456 : Nat.gcd 45736 123456 = 352 :=
by sorry

end gcd_45736_123456_l63_6322


namespace union_cardinality_inequality_l63_6320

open Set

/-- Given three finite sets A, B, and C such that A ∩ B ∩ C = ∅,
prove that |A ∪ B ∪ C| ≥ 1/2 (|A| + |B| + |C|) -/
theorem union_cardinality_inequality (A B C : Finset ℕ)
  (h : (A ∩ B ∩ C) = ∅) : (A ∪ B ∪ C).card ≥ (A.card + B.card + C.card) / 2 := sorry

end union_cardinality_inequality_l63_6320


namespace initial_price_of_TV_l63_6338

theorem initial_price_of_TV (T : ℤ) (phone_price_increase : ℤ) (total_amount : ℤ) 
    (h1 : phone_price_increase = (400: ℤ) + (40 * 400 / 100)) 
    (h2 : total_amount = T + (2 * T / 5) + phone_price_increase) 
    (h3 : total_amount = 1260) : 
    T = 500 := by
  sorry

end initial_price_of_TV_l63_6338


namespace fraction_twins_l63_6355

variables (P₀ I E P_f f : ℕ) (x : ℚ)

def initial_population := P₀ = 300000
def immigrants := I = 50000
def emigrants := E = 30000
def pregnant_fraction := f = 1 / 8
def final_population := P_f = 370000

theorem fraction_twins :
  initial_population P₀ ∧ immigrants I ∧ emigrants E ∧ pregnant_fraction f ∧ final_population P_f →
  x = 1 / 4 :=
by
  sorry

end fraction_twins_l63_6355


namespace weight_of_second_square_l63_6399

noncomputable def weight_of_square (side_length : ℝ) (density : ℝ) : ℝ :=
  side_length^2 * density

theorem weight_of_second_square :
  let s1 := 4
  let m1 := 20
  let s2 := 7
  let density := m1 / (s1 ^ 2)
  ∃ (m2 : ℝ), m2 = 61.25 :=
by
  have s1 := 4
  have m1 := 20
  have s2 := 7
  let density := m1 / (s1 ^ 2)
  have m2 := weight_of_square s2 density
  use m2
  sorry

end weight_of_second_square_l63_6399


namespace factor_expression_l63_6397

-- Define the expression to be factored
def expr (b : ℝ) := 348 * b^2 + 87 * b + 261

-- Define the supposedly factored form of the expression
def factored_expr (b : ℝ) := 87 * (4 * b^2 + b + 3)

-- The theorem stating that the original expression is equal to its factored form
theorem factor_expression (b : ℝ) : expr b = factored_expr b := 
by
  unfold expr factored_expr
  sorry

end factor_expression_l63_6397


namespace range_of_expression_l63_6318

theorem range_of_expression (x : ℝ) (h1 : 1 - 3 * x ≥ 0) (h2 : 2 * x ≠ 0) : x ≤ 1 / 3 ∧ x ≠ 0 := by
  sorry

end range_of_expression_l63_6318


namespace square_side_length_l63_6310

theorem square_side_length (A : ℝ) (h : A = 169) : ∃ s : ℝ, s^2 = A ∧ s = 13 := by
  sorry

end square_side_length_l63_6310


namespace age_is_nine_l63_6306

-- Define the conditions
def current_age (X : ℕ) :=
  X = 3 * (X - 6)

-- The theorem: Prove that the age X is equal to 9 under the conditions given
theorem age_is_nine (X : ℕ) (h : current_age X) : X = 9 :=
by
  -- The proof is omitted
  sorry

end age_is_nine_l63_6306


namespace find_min_max_l63_6335

noncomputable def f (x y : ℝ) : ℝ := Real.sin x + Real.sin y - Real.sin (x + y)

theorem find_min_max :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y ≤ 2 * Real.pi → 
    (0 ≤ f x y ∧ f x y ≤ 3 * Real.sqrt 3 / 2)) :=
sorry

end find_min_max_l63_6335


namespace math_problem_l63_6384

theorem math_problem (x y : Int)
  (hx : x = 2 - 4 + 6)
  (hy : y = 1 - 3 + 5) :
  x - y = 1 :=
by
  sorry

end math_problem_l63_6384


namespace three_pow_2040_mod_5_l63_6311

theorem three_pow_2040_mod_5 : (3^2040) % 5 = 1 := by
  sorry

end three_pow_2040_mod_5_l63_6311


namespace total_tickets_l63_6382

-- Definitions based on given conditions
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def additional_tickets : ℕ := 6

-- Proof statement (only statement, proof is not required)
theorem total_tickets : (initial_tickets - spent_tickets + additional_tickets = 30) :=
  sorry

end total_tickets_l63_6382


namespace arithmetic_sequence_seventh_term_l63_6315

noncomputable def a3 := (2 : ℚ) / 11
noncomputable def a11 := (5 : ℚ) / 6

noncomputable def a7 := (a3 + a11) / 2

theorem arithmetic_sequence_seventh_term :
  a7 = 67 / 132 := by
  sorry

end arithmetic_sequence_seventh_term_l63_6315


namespace place_value_ratio_l63_6305

theorem place_value_ratio :
  let d8_place := 0.1
  let d7_place := 10
  d8_place / d7_place = 0.01 :=
by
  -- proof skipped
  sorry

end place_value_ratio_l63_6305


namespace marbles_total_l63_6357

theorem marbles_total (r b g y : ℝ) 
  (h1 : r = 1.30 * b)
  (h2 : g = 1.50 * r)
  (h3 : y = 0.80 * g) :
  r + b + g + y = 4.4692 * r :=
by
  sorry

end marbles_total_l63_6357


namespace Ginger_sold_10_lilacs_l63_6372

variable (R L G : ℕ)

def condition1 := R = 3 * L
def condition2 := G = L / 2
def condition3 := L + R + G = 45

theorem Ginger_sold_10_lilacs
    (h1 : condition1 R L)
    (h2 : condition2 G L)
    (h3 : condition3 L R G) :
  L = 10 := 
  sorry

end Ginger_sold_10_lilacs_l63_6372


namespace student_correct_sums_l63_6339

theorem student_correct_sums (x wrong total : ℕ) (h1 : wrong = 2 * x) (h2 : total = x + wrong) (h3 : total = 54) : x = 18 :=
by
  sorry

end student_correct_sums_l63_6339


namespace minimum_value_of_expression_l63_6340

noncomputable def min_value (a b : ℝ) : ℝ :=
  a^2 + (1 / (a * b)) + (1 / (a * (a - b)))

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) : min_value a b >= 4 := by
  sorry

end minimum_value_of_expression_l63_6340


namespace triangle_interior_angles_l63_6356

theorem triangle_interior_angles (E1 E2 E3 : ℝ) (I1 I2 I3 : ℝ) (x : ℝ)
  (h1 : E1 = 12 * x) 
  (h2 : E2 = 13 * x) 
  (h3 : E3 = 15 * x)
  (h4 : E1 + E2 + E3 = 360) 
  (h5 : I1 = 180 - E1) 
  (h6 : I2 = 180 - E2) 
  (h7 : I3 = 180 - E3) :
  I1 = 72 ∧ I2 = 63 ∧ I3 = 45 :=
by
  sorry

end triangle_interior_angles_l63_6356


namespace student_chose_number_l63_6351

theorem student_chose_number (x : ℤ) (h : 2 * x - 148 = 110) : x = 129 := 
by
  sorry

end student_chose_number_l63_6351


namespace simplify_fraction_l63_6308

variable (x y : ℝ)
variable (h1 : x ≠ 0)
variable (h2 : y ≠ 0)
variable (h3 : x - y^2 ≠ 0)

theorem simplify_fraction :
  (y^2 - 1/x) / (x - y^2) = (x * y^2 - 1) / (x^2 - x * y^2) :=
by
  sorry

end simplify_fraction_l63_6308


namespace seq_general_formula_l63_6378

open Nat

def seq (a : ℕ+ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, a (n + 1) = 2 * a n / (2 + a n)

theorem seq_general_formula (a : ℕ+ → ℝ) (h : seq a) :
  ∀ n : ℕ+, a n = 2 / (n + 1) :=
by
  sorry

end seq_general_formula_l63_6378


namespace option_one_better_than_option_two_l63_6379

/-- Define the probability of winning in the first lottery option (drawing two red balls from a box
containing 4 red balls and 2 white balls). -/
def probability_option_one : ℚ := 2 / 5

/-- Define the probability of winning in the second lottery option (rolling two dice and having at least one die show a four). -/
def probability_option_two : ℚ := 11 / 36

/-- Prove that the probability of winning in the first lottery option is greater than the probability of winning in the second lottery option. -/
theorem option_one_better_than_option_two : probability_option_one > probability_option_two :=
by sorry

end option_one_better_than_option_two_l63_6379


namespace smallest_c_inequality_l63_6389

theorem smallest_c_inequality (x : ℕ → ℝ) (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10) :
  ∃ c : ℝ, (∀ x : ℕ → ℝ, x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10 →
    |x 0| + |x 1| + |x 2| + |x 3| + |x 4| + |x 5| + |x 6| + |x 7| + |x 8| ≥ c * |x 4|) ∧ c = 9 := 
by
  sorry

end smallest_c_inequality_l63_6389


namespace bandit_showdown_l63_6392

theorem bandit_showdown :
  ∃ b : ℕ, b ≥ 8 ∧ b < 50 ∧
         ∀ i j : ℕ, i ≠ j → (i < 50 ∧ j < 50) →
         ∃ k : ℕ, k < 50 ∧
         ∀ b : ℕ, b < 50 → 
         ∃ l m : ℕ, l ≠ m ∧ l < 50 ∧ m < 50 ∧ l ≠ b ∧ m ≠ b :=
sorry

end bandit_showdown_l63_6392


namespace president_vice_president_count_l63_6377

/-- The club consists of 24 members, split evenly with 12 boys and 12 girls. 
    There are also two classes, each containing 6 boys and 6 girls. 
    Prove that the number of ways to choose a president and a vice-president 
    if they must be of the same gender and from different classes is 144. -/
theorem president_vice_president_count :
  ∃ n : ℕ, n = 144 ∧ 
  (∀ (club : Finset ℕ) (boys girls : Finset ℕ) 
     (class1_boys class1_girls class2_boys class2_girls : Finset ℕ),
     club.card = 24 →
     boys.card = 12 → girls.card = 12 →
     class1_boys.card = 6 → class1_girls.card = 6 →
     class2_boys.card = 6 → class2_girls.card = 6 →
     (∃ president vice_president : ℕ,
     president ∈ club ∧ vice_president ∈ club ∧
     ((president ∈ boys ∧ vice_president ∈ boys) ∨ 
      (president ∈ girls ∧ vice_president ∈ girls)) ∧
     ((president ∈ class1_boys ∧ vice_president ∈ class2_boys) ∨
      (president ∈ class2_boys ∧ vice_president ∈ class1_boys) ∨
      (president ∈ class1_girls ∧ vice_president ∈ class2_girls) ∨
      (president ∈ class2_girls ∧ vice_president ∈ class1_girls)) →
     n = 144)) :=
by
  sorry

end president_vice_president_count_l63_6377


namespace union_complement_A_eq_l63_6336

open Set

universe u

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ (x : ℝ), y = x^2 + 1 }

theorem union_complement_A_eq :
  A ∪ ((U \ B : Set ℝ) : Set ℝ) = { x | x < 2 } := by
  sorry

end union_complement_A_eq_l63_6336


namespace undefined_expression_value_l63_6307

theorem undefined_expression_value {a : ℝ} : (a^3 - 8 = 0) ↔ (a = 2) :=
by sorry

end undefined_expression_value_l63_6307


namespace carriages_per_train_l63_6337

variable (c : ℕ)

theorem carriages_per_train :
  (∃ c : ℕ, (25 + 10) * c * 3 = 420) → c = 4 :=
by
  sorry

end carriages_per_train_l63_6337


namespace exists_another_nice_triple_l63_6394

noncomputable def is_nice_triple (a b c : ℕ) : Prop :=
  (a ≤ b ∧ b ≤ c ∧ (b - a) = (c - b)) ∧
  (Nat.gcd b a = 1 ∧ Nat.gcd b c = 1) ∧ 
  (∃ k, a * b * c = k^2)

theorem exists_another_nice_triple (a b c : ℕ) 
  (h : is_nice_triple a b c) : ∃ a' b' c', 
  (is_nice_triple a' b' c') ∧ 
  (a' = a ∨ a' = b ∨ a' = c ∨ 
   b' = a ∨ b' = b ∨ b' = c ∨ 
   c' = a ∨ c' = b ∨ c' = c) :=
by sorry

end exists_another_nice_triple_l63_6394


namespace rectangle_perimeter_l63_6393

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def satisfies_relations (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  b1 + b2 = b3 ∧
  b1 + b3 = b4 ∧
  b3 + b4 = b5 ∧
  b4 + b5 = b6 ∧
  b2 + b5 = b7

def non_overlapping_squares (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  -- Placeholder for expressing that the squares are non-overlapping.
  true -- This is assumed as given in the problem.

theorem rectangle_perimeter (b1 b2 b3 b4 b5 b6 b7 : ℕ)
  (h1 : b1 = 1) (h2 : b2 = 2)
  (h_relations : satisfies_relations b1 b2 b3 b4 b5 b6 b7)
  (h_non_overlapping : non_overlapping_squares b1 b2 b3 b4 b5 b6 b7)
  (h_rel_prime : relatively_prime b6 b7) :
  2 * (b6 + b7) = 46 := by
  sorry

end rectangle_perimeter_l63_6393


namespace xyz_ratio_l63_6350

theorem xyz_ratio (k x y z : ℝ) (h1 : x + k * y + 3 * z = 0)
                                (h2 : 3 * x + k * y - 2 * z = 0)
                                (h3 : 2 * x + 4 * y - 3 * z = 0)
                                (x_ne_zero : x ≠ 0)
                                (y_ne_zero : y ≠ 0)
                                (z_ne_zero : z ≠ 0) :
  (k = 11) → (x * z) / (y ^ 2) = 10 := by
  sorry

end xyz_ratio_l63_6350


namespace negation_of_proposition_p_is_false_l63_6359

variable (p : Prop)

theorem negation_of_proposition_p_is_false
  (h : ¬p) : ¬(¬p) :=
by
  sorry

end negation_of_proposition_p_is_false_l63_6359


namespace matrices_commute_l63_6398

variable {n : Nat}
variable (A B X : Matrix (Fin n) (Fin n) ℝ)

theorem matrices_commute (h : A * X * B + A + B = 0) : A * X * B = B * X * A :=
by
  sorry

end matrices_commute_l63_6398


namespace cone_prism_ratio_is_pi_over_16_l63_6370

noncomputable def cone_prism_volume_ratio 
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ) 
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) : ℝ :=
  (1/3) * Real.pi * cone_base_radius^2 * cone_height / (prism_length * prism_width * prism_height)

theorem cone_prism_ratio_is_pi_over_16
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ)
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) :
  cone_prism_volume_ratio prism_length prism_width prism_height cone_base_radius cone_height
    h_length h_width h_height h_radius_cone h_cone_height = Real.pi / 16 := 
by
  sorry

end cone_prism_ratio_is_pi_over_16_l63_6370


namespace y_in_terms_of_w_l63_6383

theorem y_in_terms_of_w (y w : ℝ) (h1 : y = 3^2 - 1) (h2 : w = 2) : y = 4 * w :=
by
  sorry

end y_in_terms_of_w_l63_6383


namespace distinct_real_roots_l63_6333

def otimes (a b : ℝ) : ℝ := b^2 - a * b

theorem distinct_real_roots (m x : ℝ) :
  otimes (m - 2) x = m -> ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x^2 - (m - 2) * x - m = 0) := by
  sorry

end distinct_real_roots_l63_6333


namespace profit_percentage_is_10_percent_l63_6323

theorem profit_percentage_is_10_percent
  (market_price_per_pen : ℕ)
  (retailer_buys_40_pens_for_36_price : 40 * market_price_per_pen = 36 * market_price_per_pen)
  (discount_percentage : ℕ)
  (selling_price_with_discount : ℕ) :
  discount_percentage = 1 →
  selling_price_with_discount = market_price_per_pen - (market_price_per_pen / 100) →
  (selling_price_with_discount * 40 - 36 * market_price_per_pen) / (36 * market_price_per_pen) * 100 = 10 :=
by
  sorry

end profit_percentage_is_10_percent_l63_6323


namespace find_x_for_parallel_vectors_l63_6304

noncomputable def vector_m : (ℝ × ℝ) := (1, 2)
noncomputable def vector_n (x : ℝ) : (ℝ × ℝ) := (x, 2 - 2 * x)

theorem find_x_for_parallel_vectors :
  ∀ x : ℝ, (1, 2).fst * (2 - 2 * x) - (1, 2).snd * x = 0 → x = 1 / 2 :=
by
  intros
  exact sorry

end find_x_for_parallel_vectors_l63_6304


namespace new_average_l63_6367

theorem new_average (n : ℕ) (average : ℝ) (new_average : ℝ) 
  (h1 : n = 10)
  (h2 : average = 80)
  (h3 : new_average = (2 * average * n) / n) : 
  new_average = 160 := 
by 
  simp [h1, h2, h3]
  sorry

end new_average_l63_6367


namespace maximum_xy_value_l63_6381

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end maximum_xy_value_l63_6381


namespace mo_tea_cups_l63_6363

theorem mo_tea_cups (n t : ℤ) 
  (h1 : 2 * n + 5 * t = 26) 
  (h2 : 5 * t = 2 * n + 14) :
  t = 4 :=
sorry

end mo_tea_cups_l63_6363


namespace cathy_wins_probability_l63_6368

theorem cathy_wins_probability : 
  (∑' (n : ℕ), (1 / 6 : ℚ)^3 * (5 / 6)^(3 * n)) = 1 / 91 
:= by sorry

end cathy_wins_probability_l63_6368


namespace minimum_value_768_l63_6317

noncomputable def min_value_expression (a b c : ℝ) := a^2 + 8 * a * b + 16 * b^2 + 2 * c^5

theorem minimum_value_768 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_condition : a * b^2 * c^3 = 256) : 
  min_value_expression a b c = 768 :=
sorry

end minimum_value_768_l63_6317


namespace sum_first_13_terms_l63_6332

variable {a_n : ℕ → ℝ} (S : ℕ → ℝ)
variable (a_1 d : ℝ)

-- Arithmetic sequence properties
axiom arithmetic_sequence (n : ℕ) : a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms
axiom sum_of_terms (n : ℕ) : S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom sum_specific_terms : a_n 2 + a_n 7 + a_n 12 = 30

-- Theorem to prove
theorem sum_first_13_terms : S 13 = 130 := sorry

end sum_first_13_terms_l63_6332


namespace polynomial_non_negative_l63_6344

theorem polynomial_non_negative (x : ℝ) : x^8 + x^6 - 4*x^4 + x^2 + 1 ≥ 0 := 
sorry

end polynomial_non_negative_l63_6344


namespace calculate_expression_l63_6388

theorem calculate_expression : 56.8 * 35.7 + 56.8 * 28.5 + 64.2 * 43.2 = 6420 := 
by sorry

end calculate_expression_l63_6388


namespace distance_between_parallel_lines_l63_6361

theorem distance_between_parallel_lines (a d : ℝ) (d_pos : 0 ≤ d) (a_pos : 0 ≤ a) :
  {d_ | d_ = d + a ∨ d_ = |d - a|} = {d + a, abs (d - a)} :=
by
  sorry

end distance_between_parallel_lines_l63_6361


namespace find_alpha_plus_beta_l63_6329

theorem find_alpha_plus_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos α = (Real.sqrt 5) / 5) (h4 : Real.sin β = (3 * Real.sqrt 10) / 10) : 
  α + β = 3 * π / 4 :=
sorry

end find_alpha_plus_beta_l63_6329


namespace systematic_sampling_interval_l63_6331

-- Definitions based on the conditions in part a)
def total_students : ℕ := 1500
def sample_size : ℕ := 30

-- The goal is to prove that the interval k in systematic sampling equals 50
theorem systematic_sampling_interval :
  (total_students / sample_size = 50) :=
by
  sorry

end systematic_sampling_interval_l63_6331


namespace value_of_x_l63_6325

theorem value_of_x (x : ℝ) (h : (x / 5 / 3) = (5 / (x / 3))) : x = 15 ∨ x = -15 := 
by sorry

end value_of_x_l63_6325


namespace total_cost_of_fruit_l63_6385

theorem total_cost_of_fruit (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 58) 
  (h2 : 3 * x + 2 * y = 72) : 
  3 * x + 3 * y = 78 := 
by
  sorry

end total_cost_of_fruit_l63_6385


namespace fraction_of_passengers_from_Africa_l63_6345

theorem fraction_of_passengers_from_Africa :
  (1/4 + 1/8 + 1/6 + A + 36/96 = 1) → (96 - 36) = (11/24 * 96) → 
  A = 1/12 :=
by
  sorry

end fraction_of_passengers_from_Africa_l63_6345


namespace percentage_died_by_bombardment_l63_6316

noncomputable def initial_population : ℕ := 8515
noncomputable def final_population : ℕ := 6514

theorem percentage_died_by_bombardment :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ 100) ∧
  8515 - ((x / 100) * 8515) - (15 / 100) * (8515 - ((x / 100) * 8515)) = 6514 ∧
  x = 10 :=
by
  sorry

end percentage_died_by_bombardment_l63_6316


namespace sum_x_y_eq_two_l63_6300

theorem sum_x_y_eq_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 28) : x + y = 2 :=
sorry

end sum_x_y_eq_two_l63_6300


namespace find_coefficients_l63_6346

theorem find_coefficients (a1 a2 : ℚ) :
  (4 * a1 + 5 * a2 = 9) ∧ (-a1 + 3 * a2 = 4) ↔ (a1 = 181 / 136) ∧ (a2 = 25 / 68) := 
sorry

end find_coefficients_l63_6346


namespace solve_boys_left_l63_6301

--given conditions
variable (boys_initial girls_initial boys_left girls_entered children_end: ℕ)
variable (h_boys_initial : boys_initial = 5)
variable (h_girls_initial : girls_initial = 4)
variable (h_girls_entered : girls_entered = 2)
variable (h_children_end : children_end = 8)

-- Problem definition
def boys_left_proof : Prop :=
  ∃ (B : ℕ), boys_left = B ∧ boys_initial - B + girls_initial + girls_entered = children_end ∧ B = 3

-- The statement to be proven
theorem solve_boys_left : boys_left_proof boys_initial girls_initial boys_left girls_entered children_end := by
  -- Proof will be provided here
  sorry

end solve_boys_left_l63_6301


namespace range_of_m_l63_6373

theorem range_of_m {m : ℝ} : (-1 ≤ 1 - m ∧ 1 - m ≤ 1) ↔ (0 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l63_6373


namespace product_of_two_consecutive_even_numbers_is_divisible_by_8_l63_6342

theorem product_of_two_consecutive_even_numbers_is_divisible_by_8 (n : ℤ) : (4 * n * (n + 1)) % 8 = 0 :=
sorry

end product_of_two_consecutive_even_numbers_is_divisible_by_8_l63_6342


namespace positional_relationship_l63_6366

-- Definitions of skew_lines and parallel_lines
def skew_lines (a b : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, ¬ (a x y ∨ b x y) 

def parallel_lines (a c : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x y, c x y = a (k * x) (k * y)

-- Theorem statement
theorem positional_relationship (a b c : ℝ → ℝ → Prop) 
  (h1 : skew_lines a b) 
  (h2 : parallel_lines a c) : 
  skew_lines c b ∨ (∃ x y, c x y ∧ b x y) :=
sorry

end positional_relationship_l63_6366


namespace melanie_total_amount_l63_6309

theorem melanie_total_amount :
  let g1 := 12
  let g2 := 15
  let g3 := 8
  let g4 := 10
  let g5 := 20
  g1 + g2 + g3 + g4 + g5 = 65 :=
by
  sorry

end melanie_total_amount_l63_6309


namespace train_passing_time_l63_6360

noncomputable def first_train_length : ℝ := 270
noncomputable def first_train_speed_kmh : ℝ := 108
noncomputable def second_train_length : ℝ := 360
noncomputable def second_train_speed_kmh : ℝ := 72

noncomputable def convert_speed_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def first_train_speed_mps : ℝ := convert_speed_to_mps first_train_speed_kmh
noncomputable def second_train_speed_mps : ℝ := convert_speed_to_mps second_train_speed_kmh

noncomputable def relative_speed_mps : ℝ := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance : ℝ := first_train_length + second_train_length
noncomputable def time_to_pass : ℝ := total_distance / relative_speed_mps

theorem train_passing_time : time_to_pass = 12.6 :=
by 
  sorry

end train_passing_time_l63_6360


namespace mean_calculation_incorrect_l63_6395

theorem mean_calculation_incorrect (a b c : ℝ) (h : a < b) (h1 : b < c) :
  let x := (a + b) / 2
  let y := (x + c) / 2
  y < (a + b + c) / 3 :=
by 
  let x := (a + b) / 2
  let y := (x + c) / 2
  sorry

end mean_calculation_incorrect_l63_6395


namespace tiling_tetromino_divisibility_l63_6396

theorem tiling_tetromino_divisibility (n : ℕ) : 
  (∃ (t : ℕ), n = 4 * t) ↔ (∃ (k : ℕ), n * n = 4 * k) :=
by
  sorry

end tiling_tetromino_divisibility_l63_6396


namespace rectangle_area_l63_6369

theorem rectangle_area (y : ℝ) (h_rect : (5 - (-3)) * (y - (-1)) = 48) (h_pos : 0 < y) : y = 5 :=
by
  sorry

end rectangle_area_l63_6369


namespace graph_does_not_pass_through_quadrant_II_l63_6347

noncomputable def linear_function (x : ℝ) : ℝ := 3 * x - 4

def passes_through_quadrant_I (x : ℝ) : Prop := x > 0 ∧ linear_function x > 0
def passes_through_quadrant_II (x : ℝ) : Prop := x < 0 ∧ linear_function x > 0
def passes_through_quadrant_III (x : ℝ) : Prop := x < 0 ∧ linear_function x < 0
def passes_through_quadrant_IV (x : ℝ) : Prop := x > 0 ∧ linear_function x < 0

theorem graph_does_not_pass_through_quadrant_II :
  ¬(∃ x : ℝ, passes_through_quadrant_II x) :=
sorry

end graph_does_not_pass_through_quadrant_II_l63_6347


namespace english_score_is_96_l63_6386

variable (Science_score : ℕ) (Social_studies_score : ℕ) (English_score : ℕ)

/-- Jimin's social studies score is 6 points higher than his science score -/
def social_studies_score_condition := Social_studies_score = Science_score + 6

/-- The science score is 87 -/
def science_score_condition := Science_score = 87

/-- The average score for science, social studies, and English is 92 -/
def average_score_condition := (Science_score + Social_studies_score + English_score) / 3 = 92

theorem english_score_is_96
  (h1 : social_studies_score_condition Science_score Social_studies_score)
  (h2 : science_score_condition Science_score)
  (h3 : average_score_condition Science_score Social_studies_score English_score) :
  English_score = 96 :=
  by
    sorry

end english_score_is_96_l63_6386


namespace chickens_do_not_lay_eggs_l63_6374

theorem chickens_do_not_lay_eggs (total_chickens : ℕ) 
  (roosters : ℕ) (hens : ℕ) (hens_lay_eggs : ℕ) (hens_do_not_lay_eggs : ℕ) 
  (chickens_do_not_lay_eggs : ℕ) :
  total_chickens = 80 →
  roosters = total_chickens / 4 →
  hens = total_chickens - roosters →
  hens_lay_eggs = 3 * hens / 4 →
  hens_do_not_lay_eggs = hens - hens_lay_eggs →
  chickens_do_not_lay_eggs = hens_do_not_lay_eggs + roosters →
  chickens_do_not_lay_eggs = 35 :=
by
  intros h0 h1 h2 h3 h4 h5
  sorry

end chickens_do_not_lay_eggs_l63_6374


namespace average_price_of_5_baskets_l63_6327

/-- Saleem bought 4 baskets with an average cost of $4 each. --/
def average_cost_first_4_baskets : ℝ := 4

/-- Saleem buys the fifth basket with the price of $8. --/
def price_fifth_basket : ℝ := 8

/-- Prove that the average price of the 5 baskets is $4.80. --/
theorem average_price_of_5_baskets :
  (4 * average_cost_first_4_baskets + price_fifth_basket) / 5 = 4.80 := 
by
  sorry

end average_price_of_5_baskets_l63_6327


namespace other_acute_angle_of_right_triangle_l63_6349

theorem other_acute_angle_of_right_triangle (a : ℝ) (h₀ : 0 < a ∧ a < 90) (h₁ : a = 20) :
  ∃ b, b = 90 - a ∧ b = 70 := by
    sorry

end other_acute_angle_of_right_triangle_l63_6349


namespace inequality_solution_sets_l63_6391

theorem inequality_solution_sets (a : ℝ)
  (h1 : ∀ x : ℝ, (1/2) < x ∧ x < 2 ↔ ax^2 + 5*x - 2 > 0) :
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) ↔ ax^2 - 5*x + a^2 - 1 > 0) :=
by {
  sorry
}

end inequality_solution_sets_l63_6391


namespace train_b_speed_l63_6353

/-- Given:
    1. Length of train A: 150 m
    2. Length of train B: 150 m
    3. Speed of train A: 54 km/hr
    4. Time taken to cross train B: 12 seconds
    Prove: The speed of train B is 36 km/hr
-/
theorem train_b_speed (l_A l_B : ℕ) (V_A : ℕ) (t : ℕ) (h1 : l_A = 150) (h2 : l_B = 150) (h3 : V_A = 54) (h4 : t = 12) :
  ∃ V_B : ℕ, V_B = 36 := sorry

end train_b_speed_l63_6353


namespace determine_m_l63_6321

-- Definition of complex numbers z1 and z2
def z1 (m : ℝ) : ℂ := m + 2 * Complex.I
def z2 : ℂ := 2 + Complex.I

-- Condition that the product z1 * z2 is a pure imaginary number
def pure_imaginary (c : ℂ) : Prop := c.re = 0 

-- The proof statement
theorem determine_m (m : ℝ) : pure_imaginary (z1 m * z2) → m = 1 := 
sorry

end determine_m_l63_6321


namespace find_alpha_l63_6364

theorem find_alpha (α : ℝ) (h1 : Real.tan α = -1) (h2 : 0 < α ∧ α ≤ Real.pi) : α = 3 * Real.pi / 4 :=
sorry

end find_alpha_l63_6364


namespace repeating_decimal_to_fraction_l63_6341

theorem repeating_decimal_to_fraction : (6 + 81 / 99) = 75 / 11 := 
by 
  sorry

end repeating_decimal_to_fraction_l63_6341


namespace total_pens_bought_l63_6302

-- Define the problem conditions
def pens_given_to_friends : ℕ := 22
def pens_kept_for_herself : ℕ := 34

-- Theorem statement
theorem total_pens_bought : pens_given_to_friends + pens_kept_for_herself = 56 := by
  sorry

end total_pens_bought_l63_6302


namespace increasing_sequence_k_range_l63_6354

theorem increasing_sequence_k_range (k : ℝ) (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = n^2 + k * n) :
  (∀ n : ℕ, a (n + 1) > a n) → (k ≥ -3) :=
  sorry

end increasing_sequence_k_range_l63_6354


namespace even_suff_not_nec_l63_6365

theorem even_suff_not_nec (f g : ℝ → ℝ) 
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hg_even : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x + g x) = ((f + g) x) ∧ (∀ h : ℝ → ℝ, ∃ f g : ℝ → ℝ, h = f + g ∧ ∀ x : ℝ, (h (-x) = h x) ↔ (f (-x) = f x ∧ g (-x) = g x)) :=
by 
  sorry

end even_suff_not_nec_l63_6365


namespace principal_made_mistake_l63_6387

-- Definitions based on given conditions
def students_per_class (x : ℤ) : Prop := x > 0
def total_students (x : ℤ) : ℤ := 2 * x
def non_failing_grades (y : ℤ) : ℤ := y
def failing_grades (y : ℤ) : ℤ := y + 11
def total_grades (x y : ℤ) : Prop := total_students x = non_failing_grades y + failing_grades y

-- Proposition stating the principal made a mistake
theorem principal_made_mistake (x y : ℤ) (hx : students_per_class x) : ¬ total_grades x y :=
by
  -- Assume the proof for the hypothesis is required here
  sorry

end principal_made_mistake_l63_6387


namespace no_such_function_exists_l63_6314

theorem no_such_function_exists :
  ¬(∃ (f : ℝ → ℝ), ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end no_such_function_exists_l63_6314


namespace jamie_dimes_l63_6319

theorem jamie_dimes (p n d : ℕ) (h1 : p + n + d = 50) (h2 : p + 5 * n + 10 * d = 240) : d = 10 :=
sorry

end jamie_dimes_l63_6319


namespace probability_black_ball_l63_6380

theorem probability_black_ball :
  let P_red := 0.41
  let P_white := 0.27
  let P_black := 1 - P_red - P_white
  P_black = 0.32 :=
by
  sorry

end probability_black_ball_l63_6380


namespace find_a_c_pair_l63_6376

-- Given conditions in the problem
variable (a c : ℝ)

-- First condition: The quadratic equation has exactly one solution
def quadratic_eq_has_one_solution : Prop :=
  let discriminant := (30:ℝ)^2 - 4 * a * c
  discriminant = 0

-- Second condition: Sum of a and c
def sum_eq_41 : Prop := a + c = 41

-- Third condition: a is less than c
def a_lt_c : Prop := a < c

-- State the proof problem
theorem find_a_c_pair (a c : ℝ) (h1 : quadratic_eq_has_one_solution a c) (h2 : sum_eq_41 a c) (h3 : a_lt_c a c) : (a, c) = (6.525, 34.475) :=
sorry

end find_a_c_pair_l63_6376
