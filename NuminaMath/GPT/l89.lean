import Mathlib

namespace not_irrational_B_l89_89799

-- Definition of each number as per the problem statement
def A := -π / 3
def B := 2.7182 -- denoting repeating decimal 2.71 overline 82
def C := 3.12345 -- denoting the decimal representation
def D := Real.sqrt 2

-- Prove that B is rational and others are irrational
theorem not_irrational_B : ¬ irrational B :=
by sorry

end not_irrational_B_l89_89799


namespace sanjay_homework_fraction_l89_89196

theorem sanjay_homework_fraction (h1 : (3 : ℝ) / 5 ) (h2 : (4 : ℝ) / 15 ) :
  (6 : ℝ) / 15 - (4 : ℝ) / 15 = (2 : ℝ) / 15 :=
by
  -- The fraction of the homework Sanjay did on Tuesday.
  sorry

end sanjay_homework_fraction_l89_89196


namespace num_cells_after_10_moves_l89_89613

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 + 1 = p2.1))

def num_reachable_cells (n m k : ℕ) (start : ℕ × ℕ) : ℕ :=
  sorry -- Calculation of the number of reachable cells after k moves

theorem num_cells_after_10_moves :
  let board_size := 21
  let start := (11, 11)
  let moves := 10
  num_reachable_cells board_size board_size moves start = 121 :=
sorry

end num_cells_after_10_moves_l89_89613


namespace football_players_count_l89_89006

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def softball_players : ℕ := 13
def total_players : ℕ := 59

theorem football_players_count :
  total_players - (cricket_players + hockey_players + softball_players) = 18 :=
by 
  sorry

end football_players_count_l89_89006


namespace isosceles_triangle_dot_product_equal_l89_89018

noncomputable def vector := ℝ × ℝ × ℝ

def is_isosceles_triangle (A B C : vector) : Prop :=
  dist A C = dist B C

def M_condition (A B C M : vector) : Prop :=
  (2:ℝ) • (M - A) = (B - M)

def dist (u v : vector) : ℝ := norm (u - v)

def norm (v : vector) : ℝ := real.sqrt (inner v v)

def inner (u v : vector) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def angle_ACB (A B C : vector) : Prop :=
  inner (A - C) (B - C) = (dist A C) * (dist B C) * real.cos (real.pi / 3)

def CM_dot_CB (A B C M : vector) : ℝ :=
  inner (M - C) (B - C)

def proof_question (A B C M : vector) : Prop :=
  CM_dot_CB A B C M = 12

theorem isosceles_triangle_dot_product_equal (A B C M : vector)
  (h_iso : is_isosceles_triangle A B C)
  (h_ACB : angle_ACB A B C)
  (h_M_cond : M_condition A B C M) :
  proof_question A B C M :=
sorry

end isosceles_triangle_dot_product_equal_l89_89018


namespace copper_percentage_alloy_l89_89203

theorem copper_percentage_alloy (x : ℝ) :
  (x / 100 * 45 + 0.21 * (108 - 45) = 0.1975 * 108) → x = 18 :=
by 
  sorry

end copper_percentage_alloy_l89_89203


namespace inequality_in_set_A_l89_89154

def A := {p : ℝ × ℝ × ℝ | let (a, b, c) := p in 
  a + b + c = 3 ∧ (6*a + b^2 + c^2) * (6*b + c^2 + a^2) * (6*c + a^2 + b^2) ≠ 0}

theorem inequality_in_set_A (a b c : ℝ) (h : (a, b, c) ∈ A) :
  (a / (6 * a + b^2 + c^2) + b / (6 * b + c^2 + a^2) + c / (6 * c + a^2 + b^2)) ≤ (3 / 8) :=
sorry

end inequality_in_set_A_l89_89154


namespace pinwheel_angle_sum_l89_89134

theorem pinwheel_angle_sum (AF BG CH DI EJ : Line) (O : Point)
  (h_intersect : ∀ P, P = O → 
      (P ∈ (AF ∩ BG) ∧ P ∈ (BG ∩ CH) ∧ P ∈ (CH ∩ DI) ∧ P ∈ (DI ∩ EJ)))
  : ∀ (A B C D E F G H I J : ℝ), 
      (\ang A + \ang B + \ang C + \ang D + \ang E + \ang F + \ang G + \ang H + \ang I + \ang J) = 360 :=
by
  -- The formal proof would go here.
  sorry

end pinwheel_angle_sum_l89_89134


namespace tan_315_eq_neg1_l89_89257

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89257


namespace number_of_men_in_first_group_l89_89561

theorem number_of_men_in_first_group 
    (x : ℕ) (H1 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate x 5 = 1 / (5 * x))
    (H2 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate 15 12 = 1 / (15 * 12))
    (H3 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate x 5 = work_rate 15 12) 
    : x = 36 := 
by {
    sorry
}

end number_of_men_in_first_group_l89_89561


namespace tan_315_eq_neg1_l89_89388

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89388


namespace sandy_correct_sums_l89_89160

theorem sandy_correct_sums
  (c i : ℕ)
  (h1 : c + i = 30)
  (h2 : 3 * c - 2 * i = 45) :
  c = 21 :=
by
  sorry

end sandy_correct_sums_l89_89160


namespace radius_of_two_new_cookies_l89_89847

noncomputable def radius_large : ℝ := 4
noncomputable def radius_small : ℝ := 1

theorem radius_of_two_new_cookies :
  let A_large := π * radius_large^2 in
  let A_small := π * radius_small^2 in
  let A_total_small := 6 * A_small in
  let A_scrap := A_large - A_total_small in
  let A_each_scrap := A_scrap / 2 in
  ∃ r : ℝ, π * r^2 = A_each_scrap ∧ r = sqrt 5 := by
  sorry

end radius_of_two_new_cookies_l89_89847


namespace tan_315_proof_l89_89311

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89311


namespace min_2a_b_c_l89_89484

theorem min_2a_b_c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : (a + b) * b * c = 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 := sorry

end min_2a_b_c_l89_89484


namespace math_olympiad_scores_l89_89599

theorem math_olympiad_scores (a : Fin 20 → ℕ) 
  (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → j ≠ k → i ≠ k → a i < a j + a k) :
  ∀ i : Fin 20, a i > 18 := 
sorry

end math_olympiad_scores_l89_89599


namespace arithmetic_sum_l89_89739

theorem arithmetic_sum (k : ℕ) :
  let a₁ := k^2 + 1 in
  let d := 1 in
  let n := 2 * k + 1 in
  let an := a₁ + (n - 1) * d in
  let S := (n * (a₁ + an)) / 2 in
  S = k^3 + (k + 1)^3 :=
by
  sorry

end arithmetic_sum_l89_89739


namespace medians_parallel_to_sides_of_triangle_l89_89756

-- Definitions and assumptions
variables {V : Type} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (T T₁ : Type) 
variables [Triangle T V] [Triangle T₁ V]

-- Side vectors of triangle T
variables (a b c : V)
-- Median vectors of triangle T₁
variables (median1 median2 median3 : V)

-- Assume sides of T are parallel to medians of T₁
variables (hp1 : a ∥ median1) (hp2 : b ∥ median2) (hp3 : c ∥ median3)

-- The Lean statement to prove
theorem medians_parallel_to_sides_of_triangle :
  (med (b - a) ∥ a) ∧ (med (a - c) ∥ b) ∧ (med (c - b) ∥ c) := 
by sorry

end medians_parallel_to_sides_of_triangle_l89_89756


namespace arithmetic_sum_l89_89209

theorem arithmetic_sum : (∑ k in Finset.range 10, (1 + 2 * k)) = 100 :=
by
  sorry

end arithmetic_sum_l89_89209


namespace trigonometric_identity_l89_89482

-- The main statement to prove
theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 :=
by
  sorry

end trigonometric_identity_l89_89482


namespace parameter_range_exists_solution_l89_89892

theorem parameter_range_exists_solution :
  {a : ℝ | ∃ b : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * a * (a + y - x) = 49 ∧
    y = 15 * Real.cos (x - b) - 8 * Real.sin (x - b)
  } = {a : ℝ | -24 ≤ a ∧ a ≤ 24} :=
sorry

end parameter_range_exists_solution_l89_89892


namespace tangent_315_deg_l89_89279

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89279


namespace fredek_correctness_l89_89478

open Finset

section FredekHotel

variables {G : Type} [Fintype G] [DecidableEq G] (R : G → G → Prop)
variable (n : ℕ)

def symmetric_relation (R : G → G → Prop) : Prop := ∀ a b, R a b → R b a

noncomputable def fredek_claim := 
  ∀ {G : Type} [Fintype G] [DecidableEq G] (R : G → G → Prop),
  symmetric_relation R →
  (Fintype.card G = n) →
  (n ≥ 3 → n ≠ 4 → ∃ x y : G, x ≠ y ∧ 
    (card (R x) = card (R y)) ∧ 
    (∃ z, R x z ∧ R y z ∨ ¬ R x z ∧ ¬ R y z))

theorem fredek_correctness : 
  ∀ n : ℕ, (fredek_claim R n) :=
by sorry

end FredekHotel

end fredek_correctness_l89_89478


namespace tan_315_eq_neg1_l89_89382

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89382


namespace probability_at_least_one_woman_in_selection_l89_89559

theorem probability_at_least_one_woman_in_selection :
  ∃ (P : ℚ), P = 85 / 99 :=
by 
  -- Define variables
  let total_people := 12
  let men := 8
  let women := 4
  let selection := 4

  -- Calculate the probability of selecting four men
  let P_all_men := (men / total_people) * ((men - 1) / (total_people - 1)) *
                   ((men - 2) / (total_people - 2)) *
                   ((men - 3) / (total_people - 3))

  -- Calculate the probability of at least one woman being selected
  let P_at_least_one_woman := 1 - P_all_men

  -- Verify the result
  have H : P_at_least_one_woman = 85 / 99 := sorry
  use P_at_least_one_woman
  exact H

end probability_at_least_one_woman_in_selection_l89_89559


namespace find_flour_amount_l89_89074

variables (F S C : ℕ)

-- Condition 1: Proportions must remain constant
axiom proportion : 11 * S = 7 * F ∧ 7 * C = 5 * S

-- Condition 2: Mary needs 2 more cups of flour than sugar
axiom flour_sugar : F = S + 2

-- Condition 3: Mary needs 1 more cup of sugar than cocoa powder
axiom sugar_cocoa : S = C + 1

-- Question: How many cups of flour did she put in?
theorem find_flour_amount : F = 8 :=
by
  sorry

end find_flour_amount_l89_89074


namespace tan_315_eq_neg1_l89_89385

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89385


namespace distinct_factors_of_given_number_l89_89429

-- Definition of the given number in its prime factorized form
def given_number : ℕ := 2^6 * 7^3 * 5^5

-- Theorem statement: the number of distinct natural-number factors of given_number
theorem distinct_factors_of_given_number : 
  ∃ (n : ℕ), n = 7 * 4 * 6 ∧ 
    (∀ (a b c : ℕ), 
      (0 ≤ a ∧ a ≤ 6) → 
      (0 ≤ b ∧ b ≤ 3) → 
      (0 ≤ c ∧ c ≤ 5) → 
      2^a * 7^b * 5^c ∣ given_number) :=
by
  have num_factors : ∀ (a b c : ℕ), 
    (0 ≤ a ∧ a ≤ 6) ∧ 
    (0 ≤ b ∧ b ≤ 3) ∧ 
    (0 ≤ c ∧ c ≤ 5) → 
    2^a * 7^b * 5^c ∣ given_number,
  { intros a b c h,
    cases h with ha hbc,
    cases hbc with hb hc,
    sorry },
  existsi 168,
  split,
  exact rfl,
  exact num_factors

end distinct_factors_of_given_number_l89_89429


namespace tan_315_eq_neg1_l89_89247

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89247


namespace loss_due_to_simple_interest_l89_89652

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

end loss_due_to_simple_interest_l89_89652


namespace speed_of_man_l89_89851

-- Converting 25 kmph to meters per second
noncomputable def train_speed_mps : ℝ := (25 * 1000) / 3600

-- Conditions
def train_length : ℝ := 330
def crossing_time : ℝ := 44 
def relative_speed : ℝ := train_length / crossing_time

-- The train's speed and the man's speed add up to the relative speed
def man_speed_mps : ℝ := relative_speed - train_speed_mps

-- Conversion factor from meters per second to kmph
noncomputable def man_speed_kmph : ℝ := man_speed_mps * (3600 / 1000)

theorem speed_of_man :
  man_speed_kmph ≈ 2 :=
sorry

end speed_of_man_l89_89851


namespace proposition_does_not_hold_6_l89_89493

-- Define P as a proposition over positive integers
variable (P : ℕ → Prop)

-- Assumptions
variables (h1 : ∀ k : ℕ, P k → P (k + 1))  
variable (h2 : ¬ P 7)

-- Statement of the Problem
theorem proposition_does_not_hold_6 : ¬ P 6 :=
sorry

end proposition_does_not_hold_6_l89_89493


namespace sum_of_squares_of_roots_correct_l89_89957

noncomputable def sum_of_squares_of_roots (x : ℝ) : ℝ := x^128 - 512^16

theorem sum_of_squares_of_roots_correct :
  (sum_of_squares_of_roots = 0 → (2^((9:ℝ)/4) + 2^((9:ℝ)/4)) = 2^((11:ℝ)/4)) :=
by
  sorry

end sum_of_squares_of_roots_correct_l89_89957


namespace train_time_to_cross_pole_l89_89194

def speed_kmh : ℝ := 40 -- speed of train in km/hr
def length_m : ℝ := 100 -- length of the train in meters

noncomputable def speed_ms (s_kmh : ℝ) : ℝ :=
  s_kmh * (1000 / 3600)

noncomputable def time_to_cross (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_time_to_cross_pole :
  let speed := speed_ms speed_kmh in 
  time_to_cross length_m speed = 9 :=
by
  sorry

end train_time_to_cross_pole_l89_89194


namespace incorrect_statement_e_l89_89420

-- Define each statement as conditions
def statement_a : Prop := "Some foundational statements in mathematics are accepted without proof and called axioms."
def statement_b : Prop := "There are proofs where multiple valid sequences of steps can lead to the conclusion."
def statement_c : Prop := "Each term used in a proof must be precisely defined before its use."
def statement_d : Prop := "A valid conclusion cannot be derived from a false premise using sound logical reasoning."
def statement_e : Prop := "Proof by contradiction necessarily needs multiple contradictory propositions to be effective."

-- The goal is to show that statement E is incorrect
theorem incorrect_statement_e : ¬statement_e :=
by
  -- Here the proof would show why statement_e is false
  sorry

end incorrect_statement_e_l89_89420


namespace count_no_carry_pairs_l89_89475

theorem count_no_carry_pairs : 
  ∃ n, n = 1125 ∧ ∀ (a b : ℕ), (2000 ≤ a ∧ a < 2999 ∧ b = a + 1) → 
  (∀ i, (0 ≤ i ∧ i < 4) → ((a / (10 ^ i) % 10 + b / (10 ^ i) % 10) < 10)) := sorry

end count_no_carry_pairs_l89_89475


namespace find_number_l89_89163

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 8) : x = 4 :=
by
  sorry

end find_number_l89_89163


namespace total_shaded_area_l89_89646

-- The given conditions in the problem
def largest_circle_area : ℝ := 100 * Real.pi
def radius_of_largest_circle : ℝ := Real.sqrt (largest_circle_area / Real.pi)

def medium_circle_area : ℝ := (radius_of_largest_circle / 2)^2 * Real.pi
def radius_of_medium_circle : ℝ := radius_of_largest_circle / 2

def smallest_circle_area : ℝ := (radius_of_medium_circle / 2)^2 * Real.pi
def radius_of_smallest_circle : ℝ := radius_of_medium_circle / 2

-- Define the shaded areas
def shaded_area_of_largest_circle : ℝ := largest_circle_area / 2
def shaded_area_of_medium_circle : ℝ := medium_circle_area / 2
def shaded_area_of_smallest_circle : ℝ := smallest_circle_area / 2

-- The proof problem statement using the given conditions
theorem total_shaded_area :
  shaded_area_of_largest_circle + 
  shaded_area_of_medium_circle + 
  shaded_area_of_smallest_circle = 
  65.625 * Real.pi := 
by
  sorry

end total_shaded_area_l89_89646


namespace one_third_of_1206_is_201_percent_of_200_l89_89809

theorem one_third_of_1206_is_201_percent_of_200 : 
  (1 / 3) * 1206 = 402 ∧ 402 / 200 = 201 / 100 :=
by
  sorry

end one_third_of_1206_is_201_percent_of_200_l89_89809


namespace professor_k_jokes_lectures_l89_89707

theorem professor_k_jokes_lectures (jokes : Finset ℕ) (h_card : jokes.card = 8) :
  let ways_to_choose_3 := jokes.card * (jokes.card - 1) * (jokes.card - 2) / 6
  let ways_to_choose_2 := jokes.card * (jokes.card - 1) / 2
in ways_to_choose_3 + ways_to_choose_2 = 84 :=
by sorry


end professor_k_jokes_lectures_l89_89707


namespace tan_315_degree_l89_89408

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89408


namespace unique_sequence_l89_89059

theorem unique_sequence (n : ℕ) (h : 1 < n)
  (x : Fin (n-1) → ℕ)
  (h_pos : ∀ i, 0 < x i)
  (h_incr : ∀ i j, i < j → x i < x j)
  (h_symm : ∀ i : Fin (n-1), x i + x ⟨n - 2 - i.val, sorry⟩ = 2 * n)
  (h_sum : ∀ i j : Fin (n-1), x i + x j < 2 * n → ∃ k : Fin (n-1), x i + x j = x k) :
  ∀ i : Fin (n-1), x i = 2 * (i + 1) :=
by
  sorry

end unique_sequence_l89_89059


namespace simplify_equation_is_elliptical_l89_89109

-- Define the condition: the original equation.
def equation (x y : ℝ) := sqrt (x^2 + (y + 3)^2) + sqrt (x^2 + (y - 3)^2) = 10

-- Define the target equation to verify.
def target_equation (x y : ℝ) := (x^2) / 25 + (y^2) / 16 = 1

theorem simplify_equation_is_elliptical {x y : ℝ} (h : equation x y) : target_equation x y :=
sorry

end simplify_equation_is_elliptical_l89_89109


namespace tan_315_proof_l89_89314

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89314


namespace tan_315_eq_neg1_l89_89339

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89339


namespace root_integer_l89_89996

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

def is_root (x_0 : ℝ) : Prop := f x_0 = 0

theorem root_integer (x_0 : ℝ) (h : is_root x_0) : Int.floor x_0 = 2 := by
  sorry

end root_integer_l89_89996


namespace determine_m_l89_89562

theorem determine_m (a b : ℝ) (m : ℝ) :
  (2 * (a ^ 2 - 2 * a * b - b ^ 2) - (a ^ 2 + m * a * b + 2 * b ^ 2)) = a ^ 2 - (4 + m) * a * b - 4 * b ^ 2 →
  ¬(∃ (c : ℝ), (a ^ 2 - (4 + m) * a * b - 4 * b ^ 2) = a ^ 2 + c * (a * b) + k) →
  m = -4 :=
sorry

end determine_m_l89_89562


namespace tan_315_degrees_l89_89360

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89360


namespace find_expression_l89_89527

noncomputable theory

-- Given condition
def func_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 1) = x / (x + 1)

-- Statement to prove
theorem find_expression (f : ℝ → ℝ) (h : func_condition f) : 
  ∀ x : ℝ, f x = (x + 1) / (x + 2) :=
by
  sorry

end find_expression_l89_89527


namespace tan_315_eq_neg1_l89_89386

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89386


namespace tan_315_eq_neg1_l89_89272

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89272


namespace at_most_n_fixed_points_l89_89047

/-- Let P(x) be a polynomial of degree n with integer coefficients,
    and let k be a positive integer.
    Consider the polynomial Q(x) = P(P(...P(P(x))...)), where P occurs k times.
    Prove that there exist at most n integers t such that Q(t) = t. -/
theorem at_most_n_fixed_points (P : Polynomial ℤ) (n k : ℕ) (hn : P.degree = n) (hk : k > 0) :
  ∃ m ≤ n, ∀ t : ℤ, t ∈ finset.range m → (Q P k t = t) :=
begin
  have hP_n : P.degree > 1,
  { sorry }, -- Assumed as per problem conditions
  let Q := (λ x, polynomial.iterate P k x),
  sorry
end

end at_most_n_fixed_points_l89_89047


namespace perpendicular_line_and_triangle_area_l89_89896

theorem perpendicular_line_and_triangle_area 
  (m : ℝ) (h : m = 12 * Real.sqrt 2 ∨ m = -12 * Real.sqrt 2)
  (x y : ℝ) :
  4 * x - 3 * y + 5 = 0 → 
  (3 * x + 4 * y + m = 0 ∨ 3 * x + 4 * y - m = 0) →
  (1/2) * abs (-m / 3) * abs (-m / 4) = 24 :=
begin
  sorry
end

end perpendicular_line_and_triangle_area_l89_89896


namespace angle_B_condition_angle_C_condition_l89_89574

theorem angle_B_condition (A B C a b c : ℝ) (h1 : (sin B - sin A) / sin C = (a + c) / (a + b)) : B = (2 * Real.pi) / 3 :=
sorry

theorem angle_C_condition (A C : ℝ) (h2 : sin A * cos C = (sqrt 3 - 1) / 4) : C = Real.pi / 4 :=
sorry

end angle_B_condition_angle_C_condition_l89_89574


namespace iodine_mixture_percentage_l89_89845

theorem iodine_mixture_percentage:
  ∀ (amount1 amount2 total_volume : ℝ),
  (percent1 percent2 final_percent : ℝ),
  (amount1 = 4.5) →
  (amount2 = 4.5) →
  (total_volume = 6) →
  (percent1 = 0.40) →
  (percent2 = 0.80) →
  (final_percent = 90) →
  (percent1 * amount1 + percent2 * amount2 = total_volume * (final_percent / 100)) :=
by
  sorry

end iodine_mixture_percentage_l89_89845


namespace tan_315_proof_l89_89312

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89312


namespace tan_315_eq_neg1_l89_89347

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89347


namespace set_of_points_closer_to_center_l89_89715

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((P1.1 - P2.1) ^ 2 + (P1.2 - P2.2) ^ 2)

def is_closer_to_center (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  let A := (a, b)
  let B := (a, -b)
  let C := (-a, -b)
  let D := (-a, b)
  distance P O < distance P A ∧
  distance P O < distance P B ∧
  distance P O < distance P C ∧
  distance P O < distance P D

theorem set_of_points_closer_to_center (a b : ℝ) : 
  {P | is_closer_to_center a b P} = 
  {P | -- description of the area enclosed by the perpendicular bisectors
        sorry } :=
sorry

end set_of_points_closer_to_center_l89_89715


namespace prf_p1_ge_p2_prf_p3_eq_100p0_prf_p1_le_100p2_l89_89699

-- Define the sound pressure levels using the provided formula
def Lp (p p₀ : ℝ) : ℝ := 20 * Real.log10 (p / p₀)

-- Given constants and conditions
variables (p₀ p₁ p₂ p₃ : ℝ) (h₀ : 0 < p₀)
variables (h₁ : 10^3 * p₀ ≤ p₁ ∧ p₁ ≤ 10^(9/2) * p₀)
variables (h₂ : 10^(5/2) * p₀ ≤ p₂ ∧ p₂ ≤ 10^3 * p₀)
variables (h₃ : p₃ = 10^2 * p₀)

-- Proof statements to be shown
theorem prf_p1_ge_p2 : p₁ ≥ p₂ :=
sorry

theorem prf_p3_eq_100p0 : p₃ = 100 * p₀ :=
sorry

theorem prf_p1_le_100p2 : p₁ ≤ 100 * p₂ :=
sorry

end prf_p1_ge_p2_prf_p3_eq_100p0_prf_p1_le_100p2_l89_89699


namespace probability_of_event_l89_89161

theorem probability_of_event (favorable unfavorable : ℕ) (h : favorable = 3) (h2 : unfavorable = 5) :
  (favorable / (favorable + unfavorable) : ℚ) = 3 / 8 :=
by
  sorry

end probability_of_event_l89_89161


namespace divisible_by_5_last_digit_l89_89788

theorem divisible_by_5_last_digit (B : ℕ) (h : B < 10) : (∃ k : ℕ, 5270 + B = 5 * k) ↔ B = 0 ∨ B = 5 :=
by sorry

end divisible_by_5_last_digit_l89_89788


namespace ellipse_equation_and_max_area_l89_89500

variables (a b c m k : ℝ)
variables (A B O : (ℝ × ℝ))

-- Given conditions
def ellipse_eq : Prop := ∃ a b : ℝ, a > b ∧ b > 0 ∧ (λ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
def eccentricity : Prop := a > 0 ∧ c = (real.sqrt 6) / 3 * a
def distance_minor_axis_focus : Prop := ∃ b c : ℝ, b > 0 ∧ c > 0 ∧ (λ : ℝ, c^2 = a^2 - b^2) ∧ c = real.sqrt b^2 + a^2
def points_A_B : Prop := ∃ A B O : (ℝ × ℝ), O = (0, 0) ∧ A ≠ B ∧ (|O - A| * |O - B| = |AB|)
def distance_origin_line_l : Prop := abs m / real.sqrt (1 + k^2) = real.sqrt 3 / 2

-- Proven results
def equation_of_ellipse : Prop := ∀ x y : ℝ, ellipse_eq x y ↔ (x^2 / 3) + y^2 = 1
def max_area_triangle : Prop := ∃ A B O : (ℝ × ℝ), points_A_B A B O ∧ distance_origin_line_l_dict (A, B, O) = (real.sqrt 3) / 2
def triangle_area : Prop := ∃ |AB| : ℝ, (|AB| = 2) → (real.sqrt 3 / 2 = (1/2) * |AB| * (real.sqrt 3 / 2))

theorem ellipse_equation_and_max_area :
  ellipse_eq a b c (λ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) →
  eccentricity →
  distance_minor_axis_focus →
  equation_of_ellipse ↔ max_area_triangle :=
begin
  sorry
end

end ellipse_equation_and_max_area_l89_89500


namespace relationship_among_abc_l89_89684

noncomputable def a : ℝ := Real.log (1/4) / Real.log 2
noncomputable def b : ℝ := 2.1^(1/3)
noncomputable def c : ℝ := (4/5)^2

theorem relationship_among_abc : a < c ∧ c < b :=
by
  -- Definitions
  have ha : a = Real.log (1/4) / Real.log 2 := rfl
  have hb : b = 2.1^(1/3) := rfl
  have hc : c = (4/5)^2 := rfl
  sorry

end relationship_among_abc_l89_89684


namespace reachable_cells_after_10_moves_l89_89630

def adjacent_cells (x y : ℕ) : set (ℕ × ℕ) :=
  { (x', y') | (x' = x + 1 ∧ y' = y) ∨ (x' = x - 1 ∧ y' = y) 
            ∨ (x' = x ∧ y' = y + 1) ∨ (x' = x ∧ y' = y - 1) }

def in_bounds (x y : ℕ) : Prop :=
  x > 0 ∧ x ≤ 21 ∧ y > 0 ∧ y ≤ 21

theorem reachable_cells_after_10_moves : 
  ∃ cells : set (ℕ × ℕ), ∃ initial_position : (11, 11) ∈ cells ∧ 
  (∀ (x y : ℕ), (x, y) ∈ cells → in_bounds x y ∧
  (∀ n ≤ 10, (x', y') ∈ adjacent_cells x y → (x', y') ∈ cells)) ∧ 
  (set.card cells = 121) :=
sorry

end reachable_cells_after_10_moves_l89_89630


namespace div_factorial_result_l89_89504

-- Define the given condition
def ten_fact : ℕ := 3628800

-- Define four factorial
def four_fact : ℕ := 4 * 3 * 2 * 1

-- State the theorem to be proved
theorem div_factorial_result : ten_fact / four_fact = 151200 :=
by
  -- Sorry is used to skip the proof, only the statement is provided
  sorry

end div_factorial_result_l89_89504


namespace seq_formula_l89_89981

noncomputable def seq (n : ℕ) : ℝ := 
  if n % 2 = 1 then (sqrt 2)^(n - 1) else (sqrt 2)^(n)

theorem seq_formula (n : ℕ) (h₀ : n > 0) 
  (h₁ : seq 1 = 1) 
  (h₂ : ∀ n : ℕ, n > 0 → seq n * seq (n + 1) = 2^n) : 
  seq n = if n % 2 = 1 then (sqrt 2)^(n - 1) else (sqrt 2)^(n) :=
sorry

end seq_formula_l89_89981


namespace quadratic_real_roots_l89_89962

theorem quadratic_real_roots (m : ℝ) : 
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  (discriminant ≥ 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  sorry

end quadratic_real_roots_l89_89962


namespace probability_sum_l89_89751

noncomputable def P (a : ℝ) (n : ℕ) (h : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) : ℝ :=
a / (n * (n + 1))

theorem probability_sum (a : ℝ) (h : ∑ n in (Finset.range 5).filter (λ n, n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4), P a n _ = 1) :
  P (5 / 4) 1 (Or.inl rfl) + P (5 / 4) 2 (Or.inr $ Or.inl rfl) = 5 / 6 :=
sorry

end probability_sum_l89_89751


namespace bounded_seq_implies_m_is_square_or_5k_square_l89_89659

noncomputable def f (x : ℝ) : ℝ := 
  let n := floor (sqrt x)
  let next_square := if x < (n + 1)^2 then n^2 else (n + 1)^2
  |x - next_square|

def alpha : ℝ := (3 + Real.sqrt 5) / 2

def is_bounded_seq (s : ℕ → ℝ) : Prop := 
  ∃ C, ∀ n, abs (s n) ≤ C

theorem bounded_seq_implies_m_is_square_or_5k_square (m : ℤ) 
  (h : ∃ s : ℕ → ℝ, (∀ n, s n = f (m * alpha^n)) ∧ is_bounded_seq s) :
  ∃ k : ℤ, m = k^2 ∨ m = 5 * k^2 := 
sorry

end bounded_seq_implies_m_is_square_or_5k_square_l89_89659


namespace probability_at_least_two_same_post_l89_89774

theorem probability_at_least_two_same_post : 
  let volunteers := 3
  let posts := 4
  let total_assignments := posts ^ volunteers
  let different_post_assignments := Nat.factorial posts / (Nat.factorial (posts - volunteers))
  let probability_all_different := different_post_assignments / total_assignments
  let probability_two_same := 1 - probability_all_different
  (1 - (Nat.factorial posts / (total_assignments * Nat.factorial (posts - volunteers)))) = 5 / 8 :=
by
  sorry

end probability_at_least_two_same_post_l89_89774


namespace find_number_l89_89868

theorem find_number (x : ℝ) (h: 9999 * x = 4690910862): x = 469.1 :=
by
  sorry

end find_number_l89_89868


namespace prove_f_zero_l89_89772

variable {Point : Type} [AddGroup Point] [Module ℝ Point]

noncomputable def centroid (A B C : Point) : Point := (A + B + C) / 3

variable (f : Point → ℝ)
variable (cond : ∀ A B C : Point, f (centroid A B C) = f A + f B + f C)

theorem prove_f_zero (A : Point) : f A = 0 := by
  sorry

end prove_f_zero_l89_89772


namespace digit_for_divisibility_by_5_l89_89790

theorem digit_for_divisibility_by_5 (B : ℕ) (h : B < 10) :
  (∃ (n : ℕ), n = 527 * 10 + B ∧ n % 5 = 0) ↔ (B = 0 ∨ B = 5) :=
by sorry

end digit_for_divisibility_by_5_l89_89790


namespace determine_m_l89_89054

variable (e₁ e₂ : Type) [AddCommGroup e₁] [AddCommGroup e₂]
variable [Module ℝ e₁] [Module ℝ e₂]
variable (e₁_vec e₂_vec : e₁)
variable (a b : e₁)

-- e₁ and e₂ are non-collinear suggests they are not proportional
axiom e₁_e₂_non_collinear : ¬ (∃ k : ℝ, e₁_vec = k • e₂_vec)

def a_vec := 3 • e₁_vec + 5 • e₂_vec
def b_vec := λ m : ℝ, m • e₁_vec - 3 • e₂_vec 

theorem determine_m (m : ℝ) (h : ∃ λ : ℝ, b_vec e₁_vec e₂_vec m = λ • a_vec e₁_vec e₂_vec) :
  m = -9/5 :=
by
  sorry

end determine_m_l89_89054


namespace find_angle_OKC_l89_89651

open EuclideanGeometry

variable {ABC : Triangle}
variable (A B C M K O : Point)
variable (α : ℝ)

-- Conditions
axiom angle_ABC_60 : ∠ B A C = 60
axiom angle_bisector_intersect_M : is_angle_bisector (∠ A B C) A M C
axiom point_K_on_AC : lies_on K (line A C)
axiom angle_AMK_30 : ∠ A M K = 30
axiom O_is_circumcenter_AMC : is_circumcenter O (triangle A M C)

-- The main theorem
theorem find_angle_OKC : ∠ O K C = 30 :=
sorry -- proof to be provided

end find_angle_OKC_l89_89651


namespace michael_choices_l89_89017

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem michael_choices : combination 10 4 = 210 := by
  sorry

end michael_choices_l89_89017


namespace tangent_315_deg_l89_89282

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89282


namespace remainder_123456789012_div_252_l89_89942

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l89_89942


namespace reachable_cells_after_10_moves_l89_89625

def adjacent_cells (x y : ℕ) : set (ℕ × ℕ) :=
  { (x', y') | (x' = x + 1 ∧ y' = y) ∨ (x' = x - 1 ∧ y' = y) 
            ∨ (x' = x ∧ y' = y + 1) ∨ (x' = x ∧ y' = y - 1) }

def in_bounds (x y : ℕ) : Prop :=
  x > 0 ∧ x ≤ 21 ∧ y > 0 ∧ y ≤ 21

theorem reachable_cells_after_10_moves : 
  ∃ cells : set (ℕ × ℕ), ∃ initial_position : (11, 11) ∈ cells ∧ 
  (∀ (x y : ℕ), (x, y) ∈ cells → in_bounds x y ∧
  (∀ n ≤ 10, (x', y') ∈ adjacent_cells x y → (x', y') ∈ cells)) ∧ 
  (set.card cells = 121) :=
sorry

end reachable_cells_after_10_moves_l89_89625


namespace count_no_carry_pairs_l89_89474

theorem count_no_carry_pairs : 
  ∃ n, n = 1125 ∧ ∀ (a b : ℕ), (2000 ≤ a ∧ a < 2999 ∧ b = a + 1) → 
  (∀ i, (0 ≤ i ∧ i < 4) → ((a / (10 ^ i) % 10 + b / (10 ^ i) % 10) < 10)) := sorry

end count_no_carry_pairs_l89_89474


namespace region_outside_circle_inside_square_area_l89_89831

theorem region_outside_circle_inside_square_area
  (C E F H M : Type) (r : ℝ) 
  (hC_𝐫 : ∃ (h_𝐑 : C → ℝ), h_𝐑 = r)
  (h_sq_intersect_circle : ∃ (H M : ℝ), (H ∈ H) ∧ (M ∈ M))
  (h_M_mid_EF : E <=> internal_edges.side_A efH²)
  (h_collinear_CEF : ∃ (collinear : ℝ → ℝ: ℝ → linear_equality: ssl₂), collinearD = tr_A_B__)
  (h_between_CE_F : EMD IntValValue) :
  r_interval = eeRf_math_approx) - (\full_identity⌈(segment_intersect_line_collapse))
  ⧠))

end region_outside_circle_inside_square_area_l89_89831


namespace tan_315_degree_l89_89400

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89400


namespace tan_315_eq_neg1_l89_89348

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89348


namespace incorrect_y_value_l89_89191

noncomputable def y (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem incorrect_y_value 
  (a b c d : ℝ) (h_a : a ≠ 0) 
  (x : Fin 7 → ℝ) (y : Fin 7 → ℝ)
  (h_diff_x : ∀ i : Fin 6, x i.succ - x i = d) 
  (h_y_values : y 0 = 51 ∧ y 1 = 107 ∧ y 2 = 185 ∧ y 3 = 285 ∧ y 4 = 407 ∧ y 5 = 549 ∧ y 6 = 717) :
  y 5 = 551 := 
begin
  sorry
end

end incorrect_y_value_l89_89191


namespace tan_315_eq_neg1_l89_89237

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89237


namespace remainder_div_252_l89_89925

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l89_89925


namespace sin_condition_sufficient_but_not_necessary_l89_89570

variable {A B C : ℝ}

def is_isosceles (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

theorem sin_condition_sufficient_but_not_necessary 
  (h : sin A = sin B) : is_isosceles A B C := by
  intro h
  -- Proof goes here
  sorry

end sin_condition_sufficient_but_not_necessary_l89_89570


namespace tan_315_eq_neg1_l89_89345

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89345


namespace infinite_circumferences_l89_89114

theorem infinite_circumferences (A B C : Type) (r ω : ℝ) (condition1 : ω = 1) :
  let circumferences := (2 * Real.pi) in
  let radius_reduction := 1 / 3 in
  Sum (fun n => circumferences * (radius_reduction ^ n)) = 5 * Real.pi :=
by sorry

end infinite_circumferences_l89_89114


namespace tangent_315_deg_l89_89277

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89277


namespace remainder_123456789012_mod_252_l89_89902

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l89_89902


namespace number_of_digits_in_sum_l89_89541

theorem number_of_digits_in_sum (C D : ℕ) (hC : C ≠ 0 ∧ C < 10) (hD : D % 2 = 0 ∧ D < 10) : 
  (Nat.digits 10 (8765 + (C * 100 + 43) + (D * 10 + 2))).length = 4 := 
by
  sorry

end number_of_digits_in_sum_l89_89541


namespace tan_315_eq_neg_1_l89_89316

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89316


namespace tan_315_eq_neg1_l89_89399

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89399


namespace parallel_lines_intersect_hyperbola_l89_89776

noncomputable def point_A : (ℝ × ℝ) := (0, 14)
noncomputable def point_B : (ℝ × ℝ) := (0, 4)
noncomputable def hyperbola (x : ℝ) : ℝ := 1 / x

theorem parallel_lines_intersect_hyperbola (k : ℝ)
  (x_K x_L x_M x_N : ℝ) 
  (hAK : hyperbola x_K = k * x_K + 14) (hAL : hyperbola x_L = k * x_L + 14)
  (hBM : hyperbola x_M = k * x_M + 4) (hBN : hyperbola x_N = k * x_N + 4)
  (vieta1 : x_K + x_L = -14 / k) (vieta2 : x_M + x_N = -4 / k) :
  (AL - AK) / (BN - BM) = 3.5 :=
by
  sorry

end parallel_lines_intersect_hyperbola_l89_89776


namespace final_price_correct_l89_89423

def original_cost : ℝ := 2.00
def discount : ℝ := 0.57
def final_price : ℝ := 1.43

theorem final_price_correct :
  original_cost - discount = final_price :=
by
  sorry

end final_price_correct_l89_89423


namespace arc_length_of_circle_l89_89001

theorem arc_length_of_circle (r θ L : ℝ) (h₁ : r = 2) (h₂ : θ = π / 7) :
  L = r * θ → L = 2 * (π / 7) := 
by {
  intro h,
  rw [h₁, h₂] at h,
  exact h,
}

end arc_length_of_circle_l89_89001


namespace tan_315_degrees_l89_89362

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89362


namespace tan_315_eq_neg1_l89_89264

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89264


namespace tan_315_eq_neg1_l89_89336

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89336


namespace sum_of_squares_l89_89887

-- Define the triangles and their properties
structure EquilateralTriangle :=
  (a b c: ℝ)
  (side_length: a = side_length ∧ b = side_length ∧ c = side_length)

-- Conditions
def triangle_ABC : EquilateralTriangle := ⟨10, 10, 10, by simp⟩
def triangle_A1B1C1 : EquilateralTriangle := ⟨10, 10, 10, by simp⟩

-- Point properties on arcs
axiom point_A1_on_arc_BC : True
axiom point_B1_on_arc_AC : True
axiom point_C1_on_arc_AB : True

-- Main proposition
theorem sum_of_squares 
  (AA1 BB1 CC1 : ℝ) 
  (h1 : AA1 = (10 * (Math.sin (60 + α)) / (Math.sin 60))^2)
  (h2 : BB1 = (10 * (Math.sin (60 - α)) / (Math.sin 60))^2)
  (h3 : CC1 = (10 * (Math.sin α) / (Math.sin 60))^2)
  : AA1 + BB1 + CC1 = 200 := 
  sorry

end sum_of_squares_l89_89887


namespace polynomial_sum_equals_one_l89_89049

theorem polynomial_sum_equals_one (a a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (2*x + 1)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end polynomial_sum_equals_one_l89_89049


namespace tan_315_eq_neg1_l89_89242

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89242


namespace num_games_last_year_l89_89092

-- Definitions from conditions
def num_games_this_year : ℕ := 14
def total_num_games : ℕ := 43

-- Theorem to prove
theorem num_games_last_year (num_games_last_year : ℕ) : 
  total_num_games - num_games_this_year = num_games_last_year ↔ num_games_last_year = 29 :=
by
  sorry

end num_games_last_year_l89_89092


namespace tan_315_eq_neg1_l89_89263

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89263


namespace balance_blue_balls_l89_89703

variables (G Y W R B : ℕ)

axiom green_balance : 3 * G = 6 * B
axiom yellow_balance : 2 * Y = 5 * B
axiom white_balance : 6 * B = 4 * W
axiom red_balance : 4 * R = 10 * B

theorem balance_blue_balls : 5 * G + 3 * Y + 3 * W + 2 * R = 27 * B :=
  by
  sorry

end balance_blue_balls_l89_89703


namespace _l89_89917

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l89_89917


namespace value_of_central_number_l89_89602

theorem value_of_central_number 
  (cells : ℕ → ℕ → ℕ)
  (sum_all : ∑ i in finset.range 5, ∑ j in finset.range 5, cells i j = 200)
  (sum_1x3_rectangle :
    ∀ i j : ℕ, i < 5 ∧ j < 3 → (cells i j + cells i (j+1) + cells i (j+2) = 23)) :
  cells 2 2 = 16 :=
by
  sorry

end value_of_central_number_l89_89602


namespace sqrt_difference_of_cubes_is_integer_l89_89875

theorem sqrt_difference_of_cubes_is_integer (a b : ℕ) (h1 : a = 105) (h2 : b = 104) :
  (Int.sqrt (a^3 - b^3) = 181) :=
by
  sorry

end sqrt_difference_of_cubes_is_integer_l89_89875


namespace tan_315_degrees_l89_89363

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89363


namespace triangle_side_possible_values_l89_89564

theorem triangle_side_possible_values (m : ℝ) (h1 : 1 < m) (h2 : m < 7) : 
  m = 5 :=
by
  sorry

end triangle_side_possible_values_l89_89564


namespace number_of_reachable_cells_after_10_moves_l89_89637

theorem number_of_reachable_cells_after_10_moves : 
  (let 
    n := 21 
    center := (11, 11)
    moves := 10
  in
  ∃ reachable_cells, reachable_cells = 121) :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89637


namespace tan_315_eq_neg1_l89_89273

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89273


namespace probability_at_least_one_woman_in_selection_l89_89556

theorem probability_at_least_one_woman_in_selection :
  ∃ (P : ℚ), P = 85 / 99 :=
by 
  -- Define variables
  let total_people := 12
  let men := 8
  let women := 4
  let selection := 4

  -- Calculate the probability of selecting four men
  let P_all_men := (men / total_people) * ((men - 1) / (total_people - 1)) *
                   ((men - 2) / (total_people - 2)) *
                   ((men - 3) / (total_people - 3))

  -- Calculate the probability of at least one woman being selected
  let P_at_least_one_woman := 1 - P_all_men

  -- Verify the result
  have H : P_at_least_one_woman = 85 / 99 := sorry
  use P_at_least_one_woman
  exact H

end probability_at_least_one_woman_in_selection_l89_89556


namespace bus_ride_cost_l89_89804

noncomputable def bus_cost : ℝ := 1.75

theorem bus_ride_cost (B T : ℝ) (h1 : T = B + 6.35) (h2 : T + B = 9.85) : B = bus_cost :=
by
  sorry

end bus_ride_cost_l89_89804


namespace determinant_not_sufficient_nor_necessary_l89_89485

-- Definitions of the initial conditions
variables {a1 b1 a2 b2 c1 c2 : ℝ}

-- Conditions given: neither line coefficients form the zero vector
axiom non_zero_1 : a1^2 + b1^2 ≠ 0
axiom non_zero_2 : a2^2 + b2^2 ≠ 0

-- The matrix determinant condition and line parallelism
def determinant_condition (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * b2 - a2 * b1 ≠ 0

def lines_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 - a2 * b1 = 0 ∧ a1 * c2 ≠ a2 * c1

-- Proof problem statement: proving equivalence
theorem determinant_not_sufficient_nor_necessary :
  ¬ (∀ a1 b1 a2 b2 c1 c2, (determinant_condition a1 b1 a2 b2 → lines_parallel a1 b1 c1 a2 b2 c2) ∧
                          (lines_parallel a1 b1 c1 a2 b2 c2 → determinant_condition a1 b1 a2 b2)) :=
sorry

end determinant_not_sufficient_nor_necessary_l89_89485


namespace team_includes_john_peter_mary_prob_l89_89007

/-- In a group of 12 players where John, Peter, and Mary are among them, 
if a coach randomly selects a 6-player team, 
the probability of choosing a team that includes John, Peter, and Mary is 1/11. -/
def probability_team_includes_john_peter_mary (total_players : ℕ) (team_size : ℕ) (selected_players : ℕ) : ℚ :=
  if h1 : total_players = 12 ∧ team_size = 6 ∧ selected_players = 3 then
    let ways_to_choose := Nat.choose 9 3 in
    let total_ways := Nat.choose 12 6 in
    ways_to_choose / total_ways
  else 
    0

theorem team_includes_john_peter_mary_prob :
  probability_team_includes_john_peter_mary 12 6 3 = 1 / 11 :=
by 
  simp [probability_team_includes_john_peter_mary, h1]
  sorry

end team_includes_john_peter_mary_prob_l89_89007


namespace rectangle_area_excluding_hole_l89_89876

theorem rectangle_area_excluding_hole (x : ℝ) (h : x > 5 / 3) :
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  A_large - A_hole = -x^2 + 17 * x + 38 :=
by
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  sorry

end rectangle_area_excluding_hole_l89_89876


namespace six_digit_number_proof_l89_89848

noncomputable def original_number : ℕ :=
  ∃ (N : ℕ), (N < 10^6 ∧ N ≥ 10^5 ∧ (N / 10^5 = 1)) ∧
  (∃ (M : ℕ), M = 3 * N ∧ (M = (N % 10^5) * 10 + 1)) ∧ 
  N = 142857

theorem six_digit_number_proof : original_number := 
begin
  use (142857),
  split,
  { exact dec_trivial, },
  { split,
    { exact dec_trivial, },
    { split,
      { exact dec_trivial, },
      { use (428571),
        split,
        { exact dec_trivial, },
        { exact dec_trivial, } } } } },
  sorry.

end six_digit_number_proof_l89_89848


namespace count_distinct_even_numbers_with_adjacent_4_and_5_l89_89519

noncomputable def distinct_even_numbers_with_adjacent_4_and_5 : Nat :=
  14

theorem count_distinct_even_numbers_with_adjacent_4_and_5 :
  let digits := {1, 2, 3, 4, 5}
  let even (n : Nat) : Prop := n % 2 = 0
  let adjacent (a b : Nat) : Prop := ∃ l, l = [a, b] ∨ l = [b, a]
  ∃ n : Set Nat,
    (4 ∈ n ∧ 5 ∈ n) ∧
    ∀ d ∈ n, d ∈ digits ∧
    List.length n = 4 ∧
    even (List.back n) ∧
    ∀ i, n[i] = 4 → n[i+1] = 5 ∨ n[i] = 5 → n[i+1] = 4 ∧
    ∃ k ≤ 14, k = distinct_even_numbers_with_adjacent_4_and_5
  :=
sorry

end count_distinct_even_numbers_with_adjacent_4_and_5_l89_89519


namespace remainder_123456789012_div_252_l89_89946

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l89_89946


namespace domain_of_h_l89_89895

def h (x : ℝ) : ℝ := (x^3 - 3*x^2 + 2*x + 8) / (x^2 - 9)

theorem domain_of_h :
  {x : ℝ | x^2 - 9 ≠ 0} = (Set.Ioo (Float.negInf) (-3) ∪ Set.Ioo (-3) 3 ∪ Set.Ioo 3 (Float.PosInf)) :=
by
  sorry

end domain_of_h_l89_89895


namespace sum_arithmetic_sequence_l89_89211

theorem sum_arithmetic_sequence : 
  let a1 := 1 
  let d := 2 
  let an := 19 
  let n := 10 
  S_n = n / 2 * (a1 + an)
  S_10 = 100

end sum_arithmetic_sequence_l89_89211


namespace tan_315_eq_neg1_l89_89397

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89397


namespace find_m_n_and_linear_term_coeff_l89_89517

theorem find_m_n_and_linear_term_coeff (m n : ℚ)
  (H1 : (m * x^2 + 2 * m * x - 1) * (x^m + 3 * n * x + 2) = mx^{m+2} + 2 * m * x^{m+1} - x^m + 3 * m * n * x^3 + (2 * m + 6 * m * n) * x^2 + (4 * m - 3 * n) * x - 2)
  (H2 : quartic_polynomial_has_no_quadratic_term : (6 * m * n + 2 * m - 1 = 0)) :
  m = 2 ∧ n = -1 / 4 ∧ (4 * m - 3 * n) = 35 / 4 :=
by
  sorry

end find_m_n_and_linear_term_coeff_l89_89517


namespace _l89_89923

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l89_89923


namespace total_tv_show_cost_correct_l89_89821

noncomputable def total_cost_of_tv_show : ℕ :=
  let cost_per_episode_first_season := 100000
  let episodes_first_season := 12
  let episodes_seasons_2_to_4 := 18
  let cost_per_episode_other_seasons := 2 * cost_per_episode_first_season
  let episodes_last_season := 24
  let number_of_other_seasons := 4
  let total_cost_first_season := episodes_first_season * cost_per_episode_first_season
  let total_cost_other_seasons := (episodes_seasons_2_to_4 * 3 + episodes_last_season) * cost_per_episode_other_seasons
  total_cost_first_season + total_cost_other_seasons

theorem total_tv_show_cost_correct : total_cost_of_tv_show = 16800000 := by
  sorry

end total_tv_show_cost_correct_l89_89821


namespace solve_prime_equation_l89_89724

def is_prime (n : ℕ) : Prop := ∀ k, k < n ∧ k > 1 → n % k ≠ 0

theorem solve_prime_equation (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r)
  (h : 5 * p = q^3 - r^3) : p = 67 ∧ q = 7 ∧ r = 2 :=
sorry

end solve_prime_equation_l89_89724


namespace find_y_l89_89687

theorem find_y (x y : ℝ) (h1 : 3 * x + 2 = 2) (h2 : y - x = 2) : y = 2 :=
by
  sorry

end find_y_l89_89687


namespace find_x_l89_89481

-- Define vectors a and b
def a : ℝ × ℝ × ℝ := (2, -1, x)
def b : ℝ × ℝ × ℝ := (3, 2, -1)

-- Define the condition that a is perpendicular to b
axiom perp : (2 * 3 + -1 * 2 + x * -1) = 0

-- Theorem statement
theorem find_x (x : ℝ) : x = 4 :=
by
  -- Given the perpendicularity condition
  exact (eq_of_sub_eq_zero 
         (by sorry))  -- skipping the actual proof for simplicity

end find_x_l89_89481


namespace tan_315_eq_neg_1_l89_89325

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89325


namespace brightness_ratio_sirius_altair_l89_89778

theorem brightness_ratio_sirius_altair :
  ∀ (m1 m2 E1 E2 : ℝ), 
    m1 = 1.25 → 
    m2 = 1.00 → 
    (m1 - m2) = 2.5 * (Real.log10 E2 - Real.log10 E1) → 
    E1 / E2 ≈ 1.26 :=
by 
  intros m1 m2 E1 E2 h1 h2 h3
  sorry

end brightness_ratio_sirius_altair_l89_89778


namespace determine_N_l89_89428

open Matrix

def mat1 : Matrix (Fin 2) (Fin 2) ℚ := !![2, -3; 4, -1]
def result : Matrix (Fin 2) (Fin 2) ℚ := !![-8, 7; 20, -11]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -10; 24, 38]

theorem determine_N : N ⬝ mat1 = result := by
  -- The proof is omitted
  sorry

end determine_N_l89_89428


namespace animals_left_in_barn_l89_89768

-- Define the conditions
def num_pigs : Nat := 156
def num_cows : Nat := 267
def num_sold : Nat := 115

-- Define the question
def num_left := num_pigs + num_cows - num_sold

-- State the theorem
theorem animals_left_in_barn : num_left = 308 :=
by
  sorry

end animals_left_in_barn_l89_89768


namespace number_of_reachable_cells_after_10_moves_l89_89612

-- Define board size, initial position, and the number of moves
def board_size : ℕ := 21
def initial_position : ℕ × ℕ := (11, 11)
def moves : ℕ := 10

-- Define the main problem statement
theorem number_of_reachable_cells_after_10_moves :
  (reachable_cells board_size initial_position moves).card = 121 :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89612


namespace min_distance_l89_89511

-- Define the parabola and the circle
def parabola (M : Point) : Prop :=
  ∃ (x y : ℝ), M = (x, y) ∧ y^2 = 4 * x

def circle (N : Point) : Prop :=
  ∃ (x y : ℝ), N = (x, y) ∧ (x + 4)^2 + y^2 = 4

-- Define the focus and directrix distance (d)
def focus (F : Point) : Prop := F = (1, 0)
def distance_to_focus (M F : Point) : ℝ := real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2)

-- Define the problem and proof statement
theorem min_distance (M N : Point) (d : ℝ) :
  parabola M → circle N → (d = distance_to_focus M (1, 0)) → (|MN| + d ≥ 3) :=
  sorry

end min_distance_l89_89511


namespace tan_315_proof_l89_89315

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89315


namespace right_triangle_adjacent_side_l89_89846

theorem right_triangle_adjacent_side 
  (h : hypotenuse = 8) 
  (opposite : side_opposite = 5) 
  (adjacent := sqrt 39) : 
  adjacent = sqrt 39 := 
    sorry

end right_triangle_adjacent_side_l89_89846


namespace tan_315_eq_neg1_l89_89338

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89338


namespace sum_of_consecutive_naturals_eq_2009_l89_89891

theorem sum_of_consecutive_naturals_eq_2009 :
  ∃ (a : ℕ), (∑ i in finset.range 7, a + i = 2009) :=
begin
    sorry
end

end sum_of_consecutive_naturals_eq_2009_l89_89891


namespace books_read_l89_89767

-- Definitions
def total_books : ℕ := 13
def unread_books : ℕ := 4

-- Theorem
theorem books_read : total_books - unread_books = 9 :=
by
  sorry

end books_read_l89_89767


namespace tan_315_eq_neg1_l89_89236

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89236


namespace math_problem_l89_89061

noncomputable def alpha : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def beta : ℝ := 3 - 2 * Real.sqrt 2
noncomputable def x : ℝ := alpha ^ 500
noncomputable def n : ℕ := ⌊x⌋
noncomputable def f : ℝ := x - n

theorem math_problem : x * (1 - f) = 1 := sorry

end math_problem_l89_89061


namespace _l89_89920

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l89_89920


namespace discount_discrepancy_l89_89187
-- Import necessary library to avoid import errors

-- Define the conditions
def initial_discount : ℝ := 40 / 100
def additional_discount : ℝ := 10 / 100
def claimed_discount : ℝ := 60 / 100

-- Calculate the true discounted price based on given conditions
def true_discounted_price : ℝ := (1 - initial_discount) * (1 - additional_discount)
def true_discount : ℝ := 1 - true_discounted_price

-- Define discrepancy between true discount and claimed discount
def discrepancy : ℝ := claimed_discount - true_discount

-- The theorem to be proved: the discrepancy is 14%
theorem discount_discrepancy : discrepancy = 0.14 := by
  sorry

end discount_discrepancy_l89_89187


namespace measure_of_angle_E_l89_89090

theorem measure_of_angle_E
  (EFGH : Parallelogram)
  (external_angle_F : ∠external F FH = 50)
  (angle_EGH : ∠ EGH = 70) :
  ∠ E = 90 :=
by 
  sorry

end measure_of_angle_E_l89_89090


namespace simplify_expansion_l89_89718

theorem simplify_expansion (x : ℝ) : 
  (3 * x - 6) * (x + 8) - (x + 6) * (3 * x + 2) = -2 * x - 60 :=
by
  sorry

end simplify_expansion_l89_89718


namespace minimum_area_triangle_PA1C_l89_89008

noncomputable def minimum_triangle_area : ℝ := 
  let A := (0, 0, 0)
  let B := (1, 0, 0)
  let D := (0, 2, 0)
  let A1 := (0, 0, 1)
  let B1 := (1, 0, 1)
  let C := (1, 2, 0)
  let AC := (1, 2, -1)
  
  -- Parametric equation of point P
  let P (t : ℝ) := (t, 0, t)
  -- Cross product and its magnitude
  let cross_product (x y z : ℕ × ℕ × ℕ) :=
    let (x1, y1, z1) := x 
    let (x2, y2, z2) := y
    (y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2)
  
  let magnitude (u : ℕ × ℕ × ℕ) := 
    let (x, y, z) := u
    Real.sqrt (x * x + y * y + z * z)
  
  let PA1 (t : ℝ) := (t, 0, t - 1)
  
  let CP := cross_product (PA1 t) AC
  let magnitude_CP := magnitude CP
  
  let length_AC := Real.sqrt 6
  
  let PH (t : ℝ) := magnitude_CP / length_AC
  
  let area := Real.min (1/2 * length_AC * PH t) (t ∈Icc 0 1)
  
  area

#eval minimum_triangle_area -- Expected \( \frac{\sqrt{2}}{2} \)

theorem minimum_area_triangle_PA1C : minimum_triangle_area = Real.sqrt 2 / 2 :=
sorry

end minimum_area_triangle_PA1C_l89_89008


namespace scores_greater_than_18_l89_89589

noncomputable def olympiad_scores (scores : Fin 20 → ℕ) :=
∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k

theorem scores_greater_than_18 (scores : Fin 20 → ℕ) (h1 : ∀ i j, i < j → scores i < scores j)
  (h2 : olympiad_scores scores) : ∀ i, 18 < scores i :=
by
  intro i
  sorry

end scores_greater_than_18_l89_89589


namespace intersection_of_derived_lines_l89_89497

theorem intersection_of_derived_lines (O : Point) (S : Sphere) (e f g : Line) :
  (∀ e, (¬ ∃ p, p ∈ e ∧ p ∈ S) ∧ (∀ (P : Plane) (e, P tangent_to S), ∃ e', e' connects tangent points)) →
  (∃ P, P ∈ f ∧ P ∈ g) → 
  (∃ P', P' ∈ f' ∧ P' ∈ g') :=
by
  -- Proof omitted for this statement
  sorry

end intersection_of_derived_lines_l89_89497


namespace quadratic_real_roots_range_l89_89965

theorem quadratic_real_roots_range (m : ℝ) :
  ∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0 ↔ m ≤ 4 ∧ m ≠ 3 := 
by
  sorry

end quadratic_real_roots_range_l89_89965


namespace tan_315_eq_neg1_l89_89258

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89258


namespace range_of_a_l89_89507

noncomputable def p (x : ℝ) (a : ℝ) : Prop := x > -1 → (x^2 / (x + 1)) ≥ a

noncomputable def q (a : ℝ) : Prop := ∃ x : ℝ, ax^2 - ax + 1 = 0

noncomputable def pq_false_and (pa qa : Prop) : Prop := ¬pa ∧ ¬qa

theorem range_of_a (a : ℝ) : 
  ¬(∃ x : ℝ, p x a) ∧ ¬q a ∧ (∃ x : ℝ, p x a ∨ q a) → (a = 0 ∨ a ≥ 4) :=
by 
  sorry

end range_of_a_l89_89507


namespace cos_sin_215_deg_pow_36_eq_l89_89414

theorem cos_sin_215_deg_pow_36_eq :
  (complex.cos 215 + complex.sin 215 * complex.i)^36 = 
  complex.cos 300 + complex.sin 300 * complex.i :=
by
  sorry

end cos_sin_215_deg_pow_36_eq_l89_89414


namespace tan_315_proof_l89_89313

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89313


namespace hulk_jump_distance_exceeds_1000_l89_89099

theorem hulk_jump_distance_exceeds_1000 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → 3^m ≤ 1000) ∧ 3^n > 1000 :=
sorry

end hulk_jump_distance_exceeds_1000_l89_89099


namespace olympiad_scores_greater_than_18_l89_89594

open Classical

theorem olympiad_scores_greater_than_18 (n : ℕ) (a : ℕ → ℕ) (h_distinct: ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_ordered: ∀ i j: ℕ, i < j → a i < a j)
  (h_condition: ∀ i j k: ℕ, i ≠ j → i ≠ k → j ≠ k → a i < a j + a k) :
  ∀ i < n, n = 20 ∧ a i > 18 :=
by
  assume i h_i_lt_n h_n_eq_20
  sorry

end olympiad_scores_greater_than_18_l89_89594


namespace value_of_x_l89_89736

theorem value_of_x (p q r x : ℝ)
  (h1 : p = 72)
  (h2 : q = 18)
  (h3 : r = 108)
  (h4 : x = 180 - (q + r)) : 
  x = 54 := by
  sorry

end value_of_x_l89_89736


namespace reachable_cells_after_10_moves_l89_89623

theorem reachable_cells_after_10_moves :
  let board_size := 21
  let central_cell := (11, 11)
  let moves := 10
  (reachable_cells board_size central_cell moves) = 121 :=
by
  sorry

end reachable_cells_after_10_moves_l89_89623


namespace tan_315_degree_l89_89405

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89405


namespace fractional_part_shaded_eq_four_fifths_l89_89179

-- defining the conditions
def large_square_divided : Prop :=
  ∀ (n : ℕ), let area := 1 in area / 16^(n+1)

def shaded_squares_sequence (n : ℕ) : ℚ :=
  12 / 16^(n + 1)

-- proving the fractional part theorem
theorem fractional_part_shaded_eq_four_fifths : 
  (∑' n, shaded_squares_sequence n) = 4 / 5 := 
sorry

end fractional_part_shaded_eq_four_fifths_l89_89179


namespace max_x_minus_y_l89_89674

theorem max_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x^2 + y) :
  x - y ≤ 1 / Real.sqrt 24 :=
sorry

end max_x_minus_y_l89_89674


namespace tan_315_eq_neg1_l89_89244

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89244


namespace tan_315_proof_l89_89306

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89306


namespace geometric_sequence_increasing_neither_sufficient_nor_necessary_l89_89669

-- Definitions based on the conditions
def is_geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop := ∀ n, a (n + 1) = a n * q
def is_increasing_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) > a n

-- Define the main theorem according to the problem statement
theorem geometric_sequence_increasing_neither_sufficient_nor_necessary (a : ℕ → ℝ) (a1 q : ℝ) 
  (h_geom : is_geometric_sequence a a1 q) :
  ¬ ( ( (∀ (h : a1 * q > 0), is_increasing_sequence a) ∨ 
        (∀ (h : is_increasing_sequence a), a1 * q > 0) ) ) :=
sorry

end geometric_sequence_increasing_neither_sufficient_nor_necessary_l89_89669


namespace joy_quadrilateral_rods_l89_89042

/-- 
Given a set of rods with lengths from 1 cm to 35 cm, excluding the rods
of lengths 4 cm, 9 cm, and 18 cm, prove there are 23 suitable rods that
can be chosen as the fourth rod in the quadrilateral such that the 
quadrilateral has a positive area.
-/
theorem joy_quadrilateral_rods : 
  ∀ (rods : set ℕ) (d : ℕ), 
  (∀ n, n ∈ rods ↔ n ∈ (finset.range 36).erase 9.erase 18.erase 4) →
  (∀ d, d ∈ rods → 5 < d ∧ d < 31) →
  (rods.card = 23) :=
by
  intros rods d h_rods h_valid_range
  sorry

end joy_quadrilateral_rods_l89_89042


namespace trains_cross_time_opposite_directions_l89_89780

theorem trains_cross_time_opposite_directions 
  (A B: ℕ) (tA tB: ℕ) (LA 150: ℕ) (LB 90: ℕ) 
  (hA: A = 150) (hB: B = 90)
  (htA: tA = 10)
  (htB: tB = 15) :
  (LA / tA + LB / tB) = 21 :=
by
  sorry

end trains_cross_time_opposite_directions_l89_89780


namespace tan_315_eq_neg_one_l89_89226

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89226


namespace reachable_cells_after_moves_l89_89633

def is_valid_move (n : ℕ) (x y : ℤ) : Prop :=
(abs x ≤ n ∧ abs y ≤ n ∧ (x + y) % 2 = 0)

theorem reachable_cells_after_moves (n : ℕ) :
  n = 10 → ∃ (cells : Finset (ℤ × ℤ)), cells.card = 121 ∧ 
  (∀ (cell : ℤ × ℤ), cell ∈ cells → is_valid_move n cell.1 cell.2) :=
by
  intros h
  use {-10 ≤ x, y, x + y % 2 = 0 & abs x + abs y ≤ n }
  sorry -- proof goes here

end reachable_cells_after_moves_l89_89633


namespace rectangle_vertices_of_nonintersecting_tangent_circles_l89_89217

noncomputable def points_form_rectangle {j k : Type} [euclidean_geometry.circle j] [euclidean_geometry.circle k]
  (O P A B C D : Type)
  (h1 : euclidean_geometry.circle_center j O)
  (h2 : euclidean_geometry.circle_center k P)
  (h3 : ¬ euclidean_geometry.intersect j k)
  (h4 : euclidean_geometry.tangent O k A)
  (h5 : euclidean_geometry.tangent O k B)
  (h6 : euclidean_geometry.tangent P j C)
  (h7 : euclidean_geometry.tangent P j D) : Prop :=
  euclidean_geometry.is_rectangle A B C D

theorem rectangle_vertices_of_nonintersecting_tangent_circles
  {j k : Type} [euclidean_geometry.circle j] [euclidean_geometry.circle k]
  (O P A B C D : Type)
  (h1 : euclidean_geometry.circle_center j O)
  (h2 : euclidean_geometry.circle_center k P)
  (h3 : ¬ euclidean_geometry.intersect j k)
  (h4 : euclidean_geometry.tangent O k A)
  (h5 : euclidean_geometry.tangent O k B)
  (h6 : euclidean_geometry.tangent P j C)
  (h7 : euclidean_geometry.tangent P j D) :
  points_form_rectangle O P A B C D h1 h2 h3 h4 h5 h6 h7 :=
sorry

end rectangle_vertices_of_nonintersecting_tangent_circles_l89_89217


namespace number_of_reachable_cells_after_10_moves_l89_89608

-- Define board size, initial position, and the number of moves
def board_size : ℕ := 21
def initial_position : ℕ × ℕ := (11, 11)
def moves : ℕ := 10

-- Define the main problem statement
theorem number_of_reachable_cells_after_10_moves :
  (reachable_cells board_size initial_position moves).card = 121 :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89608


namespace expression_nonnegative_interval_l89_89879

theorem expression_nonnegative_interval (x : ℝ) :
  (x - 20 * x^2 + 100 * x^3) / (16 - 2 * x^3) ≥ 0 ↔ x ∈ set.Ici 0 ∩ set.Iio 2 :=
by sorry

end expression_nonnegative_interval_l89_89879


namespace probability_winning_pair_l89_89781

theorem probability_winning_pair :
  let cards := [('A', 'R'), ('B', 'R'), ('C', 'R'), ('A', 'G'), ('B', 'G'), ('C', 'G'), ('A', 'B'), ('B', 'B'), ('C', 'B')] in
  let winning_pair := λ (card1 card2 : Char × Char), card1.fst = card2.fst ∨ card1.snd = card2.snd in
  let total_ways := (9.choose 2) in
  let favorable_ways := (3.choose 2 * 3 + 3.choose 2 * 3) in
  (favorable_ways / total_ways) = 1 / 2 := sorry

end probability_winning_pair_l89_89781


namespace total_trees_cut_l89_89655

/-- James cuts 20 trees each day for the first 2 days. Then, for the next 3 days, he and his 2 brothers (each cutting 20% fewer trees per day than James) cut trees together. Prove that they cut 196 trees in total. -/
theorem total_trees_cut :
  let trees_first_2_days := 2 * 20; let trees_per_day_james := 20; let rate_fewer := 0.2;
  let trees_per_day_brother := trees_per_day_james * (1 - rate_fewer);
  let days_with_help := 3;
  let trees_per_day_all := trees_per_day_james + 2 * trees_per_day_brother;
  let total_trees_with_help := days_with_help * trees_per_day_all;
  total_trees_first_2_days + total_trees_with_help = 196 :=
by {
  let trees_first_2_days := 2 * 20;
  let trees_per_day_james := 20;
  let rate_fewer := 0.2;
  let trees_per_day_brother := trees_per_day_james * (1 - rate_fewer);
  let days_with_help := 3;
  let trees_per_day_all := trees_per_day_james + 2 * trees_per_day_brother;
  let total_trees_with_help := days_with_help * trees_per_day_all;

  have h1 : trees_first_2_days = 40 := by norm_num,
  have h2 : trees_per_day_brother = 16 := by norm_num,
  have h3 : trees_per_day_all = 52 := by norm_num,
  have h4 : total_trees_with_help = 156 := by norm_num,
  exact h1 + h4
}

end total_trees_cut_l89_89655


namespace problem_solution_l89_89530

noncomputable def a : ℕ → ℝ
| 0       := 2
| (n + 1) := 1 - (1 / a n)

noncomputable def T (n : ℕ) : ℝ :=
∏ i in finset.range n, a i

theorem problem_solution : T 2016 = 1 :=
sorry

end problem_solution_l89_89530


namespace tangent_315_deg_l89_89284

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89284


namespace reflection_matrix_over_y_equals_x_l89_89449

theorem reflection_matrix_over_y_equals_x :
  let reflection_matrix := λ (v : ℝ × ℝ), (v.2, v.1) in
  (∀ v, reflection_matrix v = (v.2, v.1) →
      ∃ M : Matrix (Fin 2) (Fin 2) ℝ,
      ∀ v : Matrix (Fin 2) (Fin 1) ℝ,
      M ⬝ v = reflection_matrix (v 0 0, v 1 0) ⬝) :=
begin
  sorry
end

end reflection_matrix_over_y_equals_x_l89_89449


namespace maximum_value_decreasing_intervals_range_f_l89_89877

def f (x : ℝ) : ℝ := 2 * sin (2*x + π/6)

theorem maximum_value :
  ∃ x : ℝ, f x = 2 :=
sorry

theorem decreasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, k*π + π/6 ≤ x ∧ x ≤ k*π + 2*π/3 → monotone (fun ivl => f (x - ivl)) :=
sorry

theorem range_f :
  ∀ x : ℝ, x ∈ Icc (-π/6) (π/3) → f x ∈ Icc (-1) 2 :=
sorry

end maximum_value_decreasing_intervals_range_f_l89_89877


namespace at_least_one_woman_selected_probability_l89_89552

-- Define the total number of people, men, and women
def total_people : Nat := 12
def men : Nat := 8
def women : Nat := 4
def selected_people : Nat := 4

-- Define the probability ratio of at least one woman being selected
def probability_at_least_one_woman_selected : ℚ := 85 / 99

-- Prove the probability is correct given the conditions
theorem at_least_one_woman_selected_probability :
  (probability_of_selecting_at_least_one_woman men women selected_people total_people) = probability_at_least_one_woman_selected :=
sorry

end at_least_one_woman_selected_probability_l89_89552


namespace trapezium_distance_l89_89443

theorem trapezium_distance (a b area : ℝ) (h : ℝ) :
  a = 20 ∧ b = 18 ∧ area = 266 ∧
  area = (1/2) * (a + b) * h -> h = 14 :=
by
  sorry

end trapezium_distance_l89_89443


namespace greatest_constant_right_triangle_l89_89446

theorem greatest_constant_right_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2) (K : ℝ) 
    (hK : (a^2 + b^2) / (a^2 + b^2 + c^2) > K) : 
    K ≤ 1 / 2 :=
by 
  sorry

end greatest_constant_right_triangle_l89_89446


namespace triangle_inequality_l89_89151

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : True :=
  sorry

end triangle_inequality_l89_89151


namespace f_at_11_l89_89425

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then Real.log (2^(3 - x)) else f (x - 1) - f (x - 2)

theorem f_at_11 : f 11 = 2 := by
  sorry

end f_at_11_l89_89425


namespace eval_expr_l89_89888

theorem eval_expr : (3 : ℚ) / (2 - (5 / 4)) = 4 := by
  sorry

end eval_expr_l89_89888


namespace reflection_over_y_eq_x_l89_89451

theorem reflection_over_y_eq_x :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, (∀ (v : Vector (Fin 2) ℝ), M.mulVec v = ⟨v.2, v.1⟩) ∧ 
  M = (Matrix.vecCons (Matrix.vecCons 0 1) (Matrix.vecCons 1 0)) :=
begin
  sorry
end

end reflection_over_y_eq_x_l89_89451


namespace min_both_attendees_l89_89004

-- Defining the parameters and conditions
variable (n : ℕ) -- total number of attendees
variable (glasses name_tags both : ℕ) -- attendees wearing glasses, name tags, and both

-- Conditions provided in the problem
def wearing_glasses_condition (n : ℕ) (glasses : ℕ) : Prop := glasses = n / 3
def wearing_name_tags_condition (n : ℕ) (name_tags : ℕ) : Prop := name_tags = n / 2
def total_attendees_condition (n : ℕ) : Prop := n = 6

-- Theorem to prove the minimum attendees wearing both glasses and name tags is 1
theorem min_both_attendees (n glasses name_tags both : ℕ) (h1 : wearing_glasses_condition n glasses) 
  (h2 : wearing_name_tags_condition n name_tags) (h3 : total_attendees_condition n) : 
  both = 1 :=
sorry

end min_both_attendees_l89_89004


namespace lines_parallel_m_values_l89_89560

theorem lines_parallel_m_values (m : ℝ) :
    (∀ x y : ℝ, (m - 2) * x - y - 1 = 0 ↔ 3 * x - m * y = 0) ↔ (m = -1 ∨ m = 3) :=
by
  sorry

end lines_parallel_m_values_l89_89560


namespace vectors_perpendicular_l89_89536

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

def not_parallel (a b : V) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a ≠ k • b

def eq_norm (a b : V) : Prop :=
  ∥a∥ = ∥b∥

theorem vectors_perpendicular 
  (h1 : not_parallel a b) 
  (h2 : eq_norm a b) : 
  ⟪a + b, a - b⟫ = 0 :=
by
  sorry

end vectors_perpendicular_l89_89536


namespace abc_cube_geq_abc_sum_l89_89086

theorem abc_cube_geq_abc_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ a * b ^ b * c ^ c) ^ 3 ≥ (a * b * c) ^ (a + b + c) :=
by
  sorry

end abc_cube_geq_abc_sum_l89_89086


namespace solve_equation_l89_89723

theorem solve_equation (x : ℚ) : 
  (x - 30) / 3 = (5 - 3 * x) / 4 → 
  x = 135 / 13 :=
by
  intro h
  have : 12 * ((x - 30) / 3) = 12 * ((5 - 3 * x) / 4), by rw h
  norm_num at this
  sorry 

end solve_equation_l89_89723


namespace find_values_of_c_x1_x2_l89_89512

theorem find_values_of_c_x1_x2 (x₁ x₂ c : ℝ)
    (h1 : x₁ + x₂ = -2)
    (h2 : x₁ * x₂ = c)
    (h3 : x₁^2 + x₂^2 = c^2 - 2 * c) :
    c = -2 ∧ x₁ = -1 + Real.sqrt 3 ∧ x₂ = -1 - Real.sqrt 3 :=
by
  sorry

end find_values_of_c_x1_x2_l89_89512


namespace volume_uncovered_is_correct_l89_89097

-- Define the volumes of the shoebox and the objects
def volume_shoebox : ℕ := 12 * 6 * 4
def volume_object1 : ℕ := 5 * 3 * 1
def volume_object2 : ℕ := 2 * 2 * 3
def volume_object3 : ℕ := 4 * 2 * 4

-- Define the total volume of the objects
def total_volume_objects : ℕ := volume_object1 + volume_object2 + volume_object3

-- Define the volume left uncovered
def volume_uncovered : ℕ := volume_shoebox - total_volume_objects

-- Prove that the volume left uncovered is 229 cubic inches
theorem volume_uncovered_is_correct : volume_uncovered = 229 := by
  -- This is where the proof would be written
  sorry

end volume_uncovered_is_correct_l89_89097


namespace man_rowing_upstream_speed_l89_89181

theorem man_rowing_upstream_speed 
  (V_m : ℝ) (V_downstream : ℝ) (h1 : V_m = 75) (h2 : V_downstream = 90) : 
  let V_s := V_downstream - V_m in
  let V_upstream := V_m - V_s in
  V_upstream = 60 :=
by
  sorry

end man_rowing_upstream_speed_l89_89181


namespace tan_315_eq_neg1_l89_89393

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89393


namespace Oliver_total_workout_hours_l89_89702

-- Define the working hours for each day
def Monday_hours : ℕ := 4
def Tuesday_hours : ℕ := Monday_hours - 2
def Wednesday_hours : ℕ := 2 * Monday_hours
def Thursday_hours : ℕ := 2 * Tuesday_hours

-- Prove that the total hours Oliver worked out adds up to 18
theorem Oliver_total_workout_hours : Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours = 18 := by
  sorry

end Oliver_total_workout_hours_l89_89702


namespace tan_315_eq_neg1_l89_89240

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89240


namespace isosceles_triangle_vertex_angle_l89_89015

theorem isosceles_triangle_vertex_angle (θ : ℝ) (h₀ : θ = 80) (h₁ : ∃ (x y z : ℝ), (x = y ∨ y = z ∨ z = x) ∧ x + y + z = 180) : θ = 80 ∨ θ = 20 := 
sorry

end isosceles_triangle_vertex_angle_l89_89015


namespace quadratic_vertex_l89_89113

theorem quadratic_vertex (a b c : ℝ) (h1 : ∀ x, (a * x^2 + b * x + c = a * (x + 4)^2 + (-16 * a + c)))
  (h2 : a * (1 + 4)^2 + (-16 * a + c) = -75) :
  a = -3 :=
begin
  sorry
end

end quadratic_vertex_l89_89113


namespace sum_binom_eq_delta_l89_89714

-- Definitions for the problem
def binom (n k : ℕ) : ℕ := nat.choose n k

def delta (i j : ℕ) : ℕ := if i = j then 1 else 0

theorem sum_binom_eq_delta (p n : ℕ) :
  (∑ k in finset.range (n + 1), if k < p then 0 else (-1)^k * binom n k * binom k p) = (-1)^n * delta p n :=
by
  sorry

end sum_binom_eq_delta_l89_89714


namespace average_of_new_set_l89_89102

theorem average_of_new_set (s : List ℝ) (h₁ : s.length = 10) (h₂ : (s.sum / 10) = 7) : 
  ((s.map (λ x => x * 12)).sum / 10) = 84 :=
by
  sorry

end average_of_new_set_l89_89102


namespace sum_ceil_sqrt_from_25_to_100_l89_89437

-- Conditions
def ceil (x : ℝ) : ℤ := ⌈x⌉

-- The main statement
theorem sum_ceil_sqrt_from_25_to_100 : 
  (\sum n in (Finset.range (100 - 25 + 1)).map ((+ 25) ∘ nat.to_finset), ceil (real.sqrt n)) = 554 :=
by
  sorry

end sum_ceil_sqrt_from_25_to_100_l89_89437


namespace min_floor_sum_perm_l89_89492

theorem min_floor_sum_perm (n : ℕ) (h : 0 < n) : 
  let k := Nat.log2 n in
  ∃ p : (Fin n.succ) → (Fin n.succ), 
  (∀ i j : Fin n.succ, p i ≠ p j → i ≠ j) ∧ 
  (∑ i in Finset.range n, p i / (i + 1) = k + 1) := sorry

end min_floor_sum_perm_l89_89492


namespace total_weight_of_carrots_l89_89172

theorem total_weight_of_carrots (average_27_carrots : ℕ) (average_3_carrots : ℕ) 
  (h1 : average_27_carrots = 200) (h2 : average_3_carrots = 180) : 
  (27 * average_27_carrots + 3 * average_3_carrots) / 1000 = 5.94 := 
by 
  sorry

end total_weight_of_carrots_l89_89172


namespace distinct_m_values_l89_89677

theorem distinct_m_values :
  ∃ (S : Finset ℤ), 
  (∀ x₁ x₂ : ℤ, x₁ * x₂ = 36 → Int.em (x₁ % 2 = x₂ % 2) → x₁ + x₂ ∈ S) ∧ 
  S.card = 11 :=
by
  sorry

end distinct_m_values_l89_89677


namespace tan_315_eq_neg1_l89_89373

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89373


namespace prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l89_89173

theorem prime_of_form_4k_plus_1_as_sum_of_two_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 4 * k + 1) :
  ∃ a b : ℤ, p = a^2 + b^2 :=
sorry

theorem prime_of_form_8k_plus_3_as_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 8 * k + 3) :
  ∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
sorry

end prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l89_89173


namespace monochromatic_triangles_l89_89502

theorem monochromatic_triangles (n : ℕ) (points : Fin n → (ℝ × ℝ))
  (color : Fin n → Bool)
  (h_collinear : ∀ (i j k : Fin n), 
    i ≠ j → j ≠ k → i ≠ k → 
    ¬ collinear (points i) (points j) (points k))
  (h_sides : ∀ (i j k l : Fin n), 
    i < j → k < l → 
    (triangle_count (points i) (points j) points) = 
    (triangle_count (points k) (points l) points)) :
  ∃ (T₁ T₂ : Finset (Fin n)),
    T₁.card = 3 ∧ 
    T₂.card = 3 ∧ 
    monochromatic color T₁ ∧ 
    monochromatic color T₂ ∧ 
    T₁ ≠ T₂ := sorry

-- Additional Definitions needed for the theorem to compile
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry
def triangle_count (p1 p2 : ℝ × ℝ) (points : Fin n → (ℝ × ℝ)) : ℕ := sorry
def monochromatic (color : Fin n → Bool) (T : Finset (Fin n)) : Prop := sorry

end monochromatic_triangles_l89_89502


namespace seq_proof_l89_89683

noncomputable def geometric_seq (a : ℕ → ℝ) : Prop :=
∀ n, a n = 3^(n + 1)

noncomputable def b_seq (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n, b n = Real.logb 3 (a (2 * n - 1))

noncomputable def s_seq (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * (b 1 + b n) / 2

noncomputable def c_seq (S : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
∀ n, c n = 1 / (4 * S n - 1)

noncomputable def t_seq (c : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
∀ n, T n = ∑ i in Finset.range n, c (i + 1)

theorem seq_proof (a b c S T : ℕ → ℝ)
  (h1 : ∀ n, a n = 3^(n + 1))
  (h2 : ∀ n, b n = Real.logb 3 (a (2 * n - 1)))
  (h3 : ∀ n, S n = n * (b 1 + b n) / 2)
  (h4 : ∀ n, c n = 1 / (4 * S n - 1))
  (h5 : ∀ n, T n = ∑ i in Finset.range n, c (i + 1)) :
  ∀ n, T n = n / (2 * n + 1) :=
sorry

end seq_proof_l89_89683


namespace quadratic_real_roots_l89_89969

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) →
  (m ≤ 4 ∧ m ≠ 3) :=
sorry

end quadratic_real_roots_l89_89969


namespace tan_315_eq_neg1_l89_89252

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89252


namespace angle_bisector_length_l89_89649

noncomputable def angle_bisector_length_in_triangle
  (A B C : Type)
  [EuclideanSpace A]
  (AB AC : ℝ)
  (hAB : AB = 3)
  (hAC : AC = 6)
  (cosA : ℝ)
  (hcosA : cosA = 1 / 8) :
  ℝ :=
  let BC := real.sqrt (AB^2 + AC^2 - 2 * AB * AC * cosA) in
  let BD := (3 / real.sqrt 2) in
  let AD := real.sqrt (AB^2 + BD^2 - 2 * AB * BD * (real.sqrt 2 / 4)) in
  AD

theorem angle_bisector_length
  {A B C : Type}
  [EuclideanSpace A]
  (h1 : (A B C) = triangle)
  (h2 : side_length A B = 3)
  (h3 : side_length A C = 6)
  (h4 : cos_angle A = 1 / 8) :
  angle_bisector_length_in_triangle A B C 3 6 (1 / 8) 1 / 8 = 3 :=
sorry

end angle_bisector_length_l89_89649


namespace find_equidistant_point_l89_89459

noncomputable def equidistant_point (P Q R : ℝ × ℝ × ℝ) (x y : ℝ) : Prop :=
  let p1 := (x - P.1)^2 + (y - P.2 + 2)^2
  let p2 := (x - Q.1)^2 + (y - Q.2)^2 + (R.3)^2
  let p3 := (x - R.1)^2 + (y - R.2 + 3)^2 + (R.3 - P.3)^2
  p1 = p2 ∧ p1 = p3

#align_import

theorem find_equidistant_point :
  ∃ x y, equidistant_point (0, -2, 0) (3, 0, 2) (4, -3, -2) x y ∧
         x = -1 / 2 ∧ y = -11 / 2 :=
begin
  use [-1 / 2, -11 / 2],
  split,
  { dsimp [equidistant_point],
    sorry },
  { split; refl }
end

end find_equidistant_point_l89_89459


namespace no_solution_eq_l89_89094

theorem no_solution_eq (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (2 / (x + 1) + 3 / (x - 1) = 6 / (x^2 - 1)) → False :=
by
  have h3 : x^2 - 1 = (x + 1) * (x - 1) := by ring
  rw h3 at *
  sorry

end no_solution_eq_l89_89094


namespace parallel_lines_l89_89087

theorem parallel_lines (k1 k2 l1 l2 : ℝ) :
  (∀ x, (k1 ≠ k2) -> (k1 * x + l1 ≠ k2 * x + l2)) ↔ 
  (k1 = k2 ∧ l1 ≠ l2) := 
by sorry

end parallel_lines_l89_89087


namespace fraction_states_1790_1799_l89_89095

theorem fraction_states_1790_1799 (total_states : ℕ) (states_1790_1799 : ℕ) (h₁ : total_states = 25) (h₂ : states_1790_1799 = 10) :
  (states_1790_1799 : ℚ) / (total_states : ℚ) = 2 / 5 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end fraction_states_1790_1799_l89_89095


namespace remainder_123456789012_div_252_l89_89941

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l89_89941


namespace prob_1_prob_2_prob_3_prob_4_l89_89783

-- Definitions based on the conditions from the problem
def not_adjacent (p : List Char) : Prop :=
  list_to_string p != "AAB" ∧ list_to_string p != "BAA"

def a_to_left_of_b (p : List Char) : Prop :=
  (list_to_string p).index_of 'A' < (list_to_string p).index_of 'B'

def far_left_or_right (p : List Char) : Prop :=
  (p.head = 'A' ∨ p.head = 'B') ∧ p.last ≠ 'A'

def teachers_in_middle (p : List Char) : Prop :=
  p.take 2 = ['A', 'B'] ∧ list_to_string p.drop 2 = "21"

noncomputable def num_ways_not_adjacent : ℕ :=
  480

noncomputable def num_ways_a_to_left_of_b : ℕ :=
  360

noncomputable def num_ways_far_left_or_right : ℕ :=
  216

noncomputable def num_ways_teachers_in_middle : ℕ :=
  12

theorem prob_1 : num_ways_not_adjacent = 480 := sorry

theorem prob_2 : num_ways_a_to_left_of_b = 360 := sorry

theorem prob_3 : num_ways_far_left_or_right = 216 := sorry

theorem prob_4 : num_ways_teachers_in_middle = 12 := sorry

end prob_1_prob_2_prob_3_prob_4_l89_89783


namespace real_number_a_l89_89518

-- Definition of the complex number z.
def z (a : ℝ) : Complex := Complex.I * a + 2 / (1 + Complex.I)

-- Condition given: z * conjugate(z) = 2
def condition (a : ℝ) : Prop := (z a) * Complex.conj (z a) = 2

-- Proposition to prove: a = 0 or a = 2
theorem real_number_a (a : ℝ) (h : condition a) : a = 0 ∨ a = 2 := by
  -- Proof stub
  sorry

end real_number_a_l89_89518


namespace poly_div_quotient_l89_89900

/-!
# Problem Statement: 
Find the quotient when \( x^5 - x^4 + x^3 - 9 \) is divided by \( x - 1 \).
-/

theorem poly_div_quotient :
  Polynomial.quotientByMonicPoly (Polynomial.X ^ 5 - Polynomial.X ^ 4 + Polynomial.X ^ 3 - 9 : Polynomial ℝ)
                                 (Polynomial.X - 1 : Polynomial ℝ) = 
  Polynomial.X ^ 4 - Polynomial.X ^ 3 + Polynomial.X ^ 2 - Polynomial.X + 1 :=
sorry

end poly_div_quotient_l89_89900


namespace remainder_when_divided_l89_89939

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l89_89939


namespace sum_of_inverses_squared_lt_l89_89710

theorem sum_of_inverses_squared_lt (n : ℕ) (h₁: 1 ≤ n)  (h₂: 2 ≤ n) :
  (1 + ∑ k in Finset.range (n - 1) + 1, (1 / (k + 2) ^ 2)) < (2 - 1 / ↑n) := sorry

end sum_of_inverses_squared_lt_l89_89710


namespace reachable_cells_after_10_moves_l89_89622

theorem reachable_cells_after_10_moves :
  let board_size := 21
  let central_cell := (11, 11)
  let moves := 10
  (reachable_cells board_size central_cell moves) = 121 :=
by
  sorry

end reachable_cells_after_10_moves_l89_89622


namespace quadratic_inequality_solution_set_l89_89427

theorem quadratic_inequality_solution_set (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ (b^2 - 4 * a * c) < 0) :=
by sorry

end quadratic_inequality_solution_set_l89_89427


namespace remainder_123456789012_mod_252_l89_89904

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l89_89904


namespace tan_315_eq_neg1_l89_89384

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89384


namespace cyclic_iff_equal_areas_l89_89644

variables {A B C D P : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P]

/-- In the convex quadrilateral ABCD, the diagonals AC and BD are perpendicular,
    and the opposite sides AB and DC are not parallel. The point P, where the 
    perpendicular bisectors of AB and DC meet, is inside ABCD. Prove that ABCD is 
    cyclic if and only if the triangles ABP and CDP have equal areas. -/
theorem cyclic_iff_equal_areas
  (convex_ABCD : convex_hull ℝ ({A, B, C, D} : set Type*))
  (perpendicular_AC_BD : ∀ (X : Type*), is_perpendicular (metric_space.dist A C) (metric_space.dist B D))
  (not_parallel_AB_DC : ¬is_parallel (metric_space.dist A B) (metric_space.dist D C))
  (P_meets_bisectors : ∀ (X : Type*), is_inside P (convex_hull ℝ ({A, B, C, D} : set Type*)) ∧
                        ∀ l : set Type*, is_perpendicular_bisector l)
  : cyclic (convex_hull ℝ ({A, B, C, D} : set Type*)) ↔
    (triangle.area A B P = triangle.area C D P) :=
sorry

end cyclic_iff_equal_areas_l89_89644


namespace Sasha_is_girl_l89_89167

-- Definitions for the problem
def children : List String := ["Vanya", "Dima", "Egor", "Inna", "Lesha", "Sasha", "Tanya"]

def boys : List String := ["Vanya", "Dima", "Egor", "Lesha"]
def girls : List String := ["Inna", "Tanya"]

def is_boy (x : String) : Prop := x ∈ boys
def is_girl (x : String) : Prop := x ∈ girls

def response (x : String) (count : Nat) : Prop :=
  if is_boy x then count = 2 ∨ count = 3
  else if is_girl x then count = 6
  else False

-- Theorem to prove
theorem Sasha_is_girl : is_girl "Sasha" :=
sorry

end Sasha_is_girl_l89_89167


namespace remainder_of_large_number_l89_89955

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l89_89955


namespace sequence_general_terms_l89_89531

theorem sequence_general_terms (θ : ℝ) :
  (∀ n, aₙ = a_(n-1) * real.cos θ - b_(n-1) * real.sin θ) ∧ 
  (∀ n, bₙ = a_(n-1) * real.sin θ + b_(n-1) * real.cos θ) ∧ 
  a₁ = 1 ∧ 
  b₁ = real.tan θ →
  (∀ n, aₙ = real.sec θ * real.cos (n * θ)) ∧
  (∀ n, bₙ = real.sec θ * real.sin (n * θ)) := 
sorry

end sequence_general_terms_l89_89531


namespace sphere_surface_area_ratio_l89_89568

theorem sphere_surface_area_ratio (V1 V2 : ℝ) (h1 : V1 = (4 / 3) * π * (r1^3))
  (h2 : V2 = (4 / 3) * π * (r2^3)) (h3 : V1 / V2 = 1 / 27) :
  (4 * π * r1^2) / (4 * π * r2^2) = 1 / 9 := 
sorry

end sphere_surface_area_ratio_l89_89568


namespace tan_315_eq_neg1_l89_89256

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89256


namespace tan_315_eq_neg1_l89_89333

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89333


namespace smallest_integer_ratio_l89_89675

theorem smallest_integer_ratio (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (h_sum : x + y = 120) (h_even : x % 2 = 0) : ∃ (k : ℕ), k = x / y ∧ k = 1 :=
by
  sorry

end smallest_integer_ratio_l89_89675


namespace find_chemistry_marks_l89_89424

theorem find_chemistry_marks 
  (marks_english : ℕ = 96)
  (marks_math : ℕ = 95)
  (marks_physics : ℕ = 82)
  (marks_biology : ℕ = 92)
  (average_marks : ℚ = 90.4) : 
  ∃ C : ℕ, 
    (marks_english + marks_math + marks_physics + marks_biology + C) / 5 = average_marks ∧ C = 87 :=
by
  sorry

end find_chemistry_marks_l89_89424


namespace sqrt_expression_value_l89_89148

theorem sqrt_expression_value :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 * Real.sqrt 3 :=
by
  sorry

end sqrt_expression_value_l89_89148


namespace find_points_B_and_C_l89_89050

theorem find_points_B_and_C
  (A : (ℝ × ℝ))
  (hA1 : A = (2, 4))
  (hA2 : ∃ x, ∃ y, y = 2 * x^2 ∧ A = (x, y))
  (hA3 : ∀ x (h : x ≠ 2), let y := 2 * x^2 in 
    ∀ (normal : ℝ → ℝ),
    (normal x = -1 / 8 * x + 17 / 4) →
    (∃ B, B = (x, y) ∧ ∀ C, C = (34, 0))) :
  ∃ B C, B = (-17 / 16, 289 / 128) ∧ C = (34, 0) := sorry

end find_points_B_and_C_l89_89050


namespace tan_315_eq_neg_one_l89_89231

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89231


namespace range_of_a_for_monotonic_f_l89_89742

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 1 then (a - 2) * x - 1 else (a * x + 1) / (x + a)

theorem range_of_a_for_monotonic_f :
  { a : ℝ | ∀ x y : ℝ, x ≤ y → f a x ≤ f a y } = { a : ℝ | 2 < a ∧ a ≤ 4 } :=
by sorry

end range_of_a_for_monotonic_f_l89_89742


namespace xyz_inequality_l89_89711

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2) :=
sorry

end xyz_inequality_l89_89711


namespace circle_tangency_α_l89_89021

theorem circle_tangency_α (α : ℝ) : 
  let C1 := λ θ, 4 * 2^(1/2) * cos (θ - (Real.pi / 4))
      C2_x θ := -1 + α * cos θ
      C2_y θ := -1 + α * sin θ in
  ∃ r1 r2 x1 y1 x2 y2 : ℝ,
    (r1 = 2 * 2^(1/2)) ∧ (x1 = 2) ∧ (y1 = 2) ∧
    (r2 = |α|) ∧ (x2 = -1) ∧ (y2 = -1) ∧
    ((3 * 2^(1/2) = r1 + r2) ∨ (3 * 2^(1/2) = r1 - r2)) →
  α = 2^(1/2) ∨ α = -2^(1/2) ∨ α = 5 * 2^(1/2) ∨ α = -5 * 2^(1/2) :=
by
  intros
  sorry

end circle_tangency_α_l89_89021


namespace spell_theer_incorrect_probability_l89_89844

theorem spell_theer_incorrect_probability :
  let letters := ['h', 'r', 't', 'e', 'e']
  let target_word := "theer"
  let total_arrangements := 5! / (2! * (5 - 2)!) * 3!
  let correct_arrangements := 1
  let incorrect_probability := (total_arrangements - correct_arrangements) / total_arrangements
  incorrect_probability = 59 / 60 := by
  sorry

end spell_theer_incorrect_probability_l89_89844


namespace tan_315_eq_neg1_l89_89377

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89377


namespace remainder_when_divided_l89_89940

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l89_89940


namespace merchant_profit_percentage_l89_89157

theorem merchant_profit_percentage (C S : ℝ) (h : 24 * C = 16 * S) : ((S - C) / C) * 100 = 50 := by
  -- Adding "by" to denote beginning of proof section
  sorry  -- Proof is skipped

end merchant_profit_percentage_l89_89157


namespace tan_315_eq_neg_one_l89_89299

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89299


namespace tan_315_degrees_l89_89371

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89371


namespace train_pass_bridge_time_l89_89855

theorem train_pass_bridge_time (L_train L_bridge : ℕ) (v_train_kmh : ℕ) :
  L_train = 360 → L_bridge = 140 → v_train_kmh = 36 →
  let total_distance := L_train + L_bridge in
  let v_train_ms := v_train_kmh * 1000 / 3600 in
  let time := total_distance / v_train_ms in
  time = 50 :=
by
  intros h1 h2 h3
  unfold total_distance v_train_ms time
  rw [h1, h2, h3]
  norm_num
  sorry

end train_pass_bridge_time_l89_89855


namespace common_difference_min_Sn_l89_89499

section arithmetic_sequence

variables {a d : ℤ} -- Define variables in ℤ

-- Condition 1: First term of the sequence
def a1 : ℤ := -9

-- Condition 2: Condition involving the sum of first three terms
def condition_S3 (S3 S1 : ℤ) : Prop := (S3 / 3) - S1 = 1

-- Sum of first 'n' terms (Sn) of an arithmetic sequence
def Sn (n : ℤ) (a d : ℤ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

-- Proof statement
theorem common_difference_min_Sn :
  ∃ d Smin, d = 1 ∧ Smin = -45 ∧ 
    (∀ S3 S1, condition_S3 S3 S1 → S1 = a1 → (Sn 3 a1 d = S3)) :=
begin
  sorry -- Proof goes here
end

end arithmetic_sequence

end common_difference_min_Sn_l89_89499


namespace inequality_solution_set_l89_89757

theorem inequality_solution_set :
  { x : ℝ | (x + 2) ^ (-3 / 5) < (5 - 2 * x) ^ (-3 / 5) } =
  { x : ℝ | x < -2 } ∪ { x : ℝ | 1 < x ∧ x < 5 / 2 } := by
sorry

end inequality_solution_set_l89_89757


namespace range_of_m_l89_89990

theorem range_of_m {a b c x0 y0 y1 y2 m : ℝ} (h1 : a ≠ 0)
    (A_on_parabola : y1 = a * m^2 + 4 * a * m + c)
    (B_on_parabola : y2 = a * (m + 2)^2 + 4 * a * (m + 2) + c)
    (C_on_parabola : y0 = a * (-2)^2 + 4 * a * (-2) + c)
    (C_is_vertex : x0 = -2)
    (y_relation : y0 ≥ y2 ∧ y2 > y1) :
    m < -3 := 
sorry

end range_of_m_l89_89990


namespace number_line_problem_l89_89083

theorem number_line_problem (A B C : ℤ) (hA : A = -1) (hB : B = A - 5 + 6) (hC : abs (C - B) = 5) :
  C = 5 ∨ C = -5 :=
by sorry

end number_line_problem_l89_89083


namespace tan_315_eq_neg1_l89_89355

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89355


namespace tan_315_eq_neg1_l89_89270

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89270


namespace correct_option_d_l89_89798

theorem correct_option_d : real.cbrt (-8) = -(real.cbrt 8) :=
by sorry

end correct_option_d_l89_89798


namespace number_of_reachable_cells_after_10_moves_l89_89640

theorem number_of_reachable_cells_after_10_moves : 
  (let 
    n := 21 
    center := (11, 11)
    moves := 10
  in
  ∃ reachable_cells, reachable_cells = 121) :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89640


namespace remainder_div_252_l89_89929

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l89_89929


namespace euler_theorem_l89_89071

theorem euler_theorem 
  (Δ : ℝ) -- Area of triangle ABC
  (O : Point) -- Circumcenter
  (R : ℝ) -- Circumradius
  (P : Point) (d : ℝ) -- Any point P with OP = d
  (A B C : Point) -- Vertices of the triangle ABC
  (A₁ B₁ C₁ : Point) -- Projections of P on sides BC, CA, AB respectively
  (Δ₁ : ℝ) -- Area of triangle A₁B₁C₁
  (hOP : dist O P = d) -- Distance OP = d
  (hΔ₁ : Δ₁ = Δ / 4 * abs (1 - d^2 / R^2)) :
  Δ₁ = Δ / 4 * abs (1 - d^2 / R^2) :=
sorry

end euler_theorem_l89_89071


namespace distance_inequality_l89_89084

-- Define the vertices of the regular tetrahedron
noncomputable def A1 : ℝ × ℝ × ℝ := (0, 0, 1)
noncomputable def A2 : ℝ × ℝ × ℝ := (sqrt 3 / 2, 0, -1 / 2)
noncomputable def A3 : ℝ × ℝ × ℝ := (-sqrt 3 / 2, 0, -1 / 2)
noncomputable def A4 : ℝ × ℝ × ℝ := (0, sqrt 3, 0)

-- Assume B1 and B2 are inside the specified bounded region
variables (B1 B2 : ℝ × ℝ × ℝ)
def is_inside (P : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := P in
  (x^2 + y^2 + z^2 ≤ 1) ∧
  ((x - sqrt 3 / 2)^2 + y^2 + (z + 1 / 2)^2 ≤ 1) ∧
  ((x + sqrt 3 / 2)^2 + y^2 + (z + 1 / 2)^2 ≤ 1)

-- Proving the inequality
theorem distance_inequality (hB1 : is_inside B1) (hB2 : is_inside B2) :
  dist B1 B2 < max (dist B1 A1) (max (dist B1 A2) (max (dist B1 A3) (dist B1 A4))) :=
by
  sorry

end distance_inequality_l89_89084


namespace tan_315_eq_neg1_l89_89271

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89271


namespace total_production_cost_l89_89823

-- Conditions
def first_season_episodes : ℕ := 12
def remaining_season_factor : ℝ := 1.5
def last_season_episodes : ℕ := 24
def first_season_cost_per_episode : ℝ := 100000
def other_season_cost_per_episode : ℝ := first_season_cost_per_episode * 2

-- Number of seasons
def number_of_seasons : ℕ := 5

-- Question: Calculate the total cost
def total_first_season_cost : ℝ := first_season_episodes * first_season_cost_per_episode
def second_season_episodes : ℕ := (first_season_episodes * remaining_season_factor).toNat
def second_season_cost : ℝ := second_season_episodes * other_season_cost_per_episode
def third_and_fourth_seasons_cost : ℝ := 2 * second_season_cost
def last_season_cost : ℝ := last_season_episodes * other_season_cost_per_episode
def total_cost : ℝ := total_first_season_cost + second_season_cost + third_and_fourth_seasons_cost + last_season_cost

-- Proof
theorem total_production_cost :
  total_cost = 16800000 :=
by
  sorry

end total_production_cost_l89_89823


namespace total_amount_shared_l89_89826

-- Given John (J), Jose (Jo), and Binoy (B) and their proportion of money
variables (J Jo B : ℕ)
-- John received 1440 Rs.
variable (John_received : J = 1440)

-- The ratio of their shares is 2:4:6
axiom ratio_condition : J * 2 = Jo * 4 ∧ J * 2 = B * 6

-- The target statement to prove
theorem total_amount_shared : J + Jo + B = 8640 :=
by {
  sorry
}

end total_amount_shared_l89_89826


namespace math_olympiad_scores_l89_89597

theorem math_olympiad_scores (a : Fin 20 → ℕ) 
  (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → j ≠ k → i ≠ k → a i < a j + a k) :
  ∀ i : Fin 20, a i > 18 := 
sorry

end math_olympiad_scores_l89_89597


namespace tan_315_proof_l89_89304

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89304


namespace int_part_div_10_harmonic_sum_7_l89_89792

open Real

noncomputable def harmonic_sum_7 := (1 : ℝ) + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5 + 1 / 6 + 1 / 7

theorem int_part_div_10_harmonic_sum_7 : (⌊10 / harmonic_sum_7⌋ : ℤ) = 3 :=
  sorry

end int_part_div_10_harmonic_sum_7_l89_89792


namespace expression_equals_24_l89_89489

noncomputable def f : ℕ → ℝ := sorry

axiom f_add (m n : ℕ) : f (m + n) = f m * f n
axiom f_one : f 1 = 3

theorem expression_equals_24 :
  (f 1^2 + f 2) / f 1 + (f 2^2 + f 4) / f 3 + (f 3^2 + f 6) / f 5 + (f 4^2 + f 8) / f 7 = 24 :=
by sorry

end expression_equals_24_l89_89489


namespace charlie_has_54_crayons_l89_89689

theorem charlie_has_54_crayons
  (crayons_Billie : ℕ)
  (crayons_Bobbie : ℕ)
  (crayons_Lizzie : ℕ)
  (crayons_Charlie : ℕ)
  (h1 : crayons_Billie = 18)
  (h2 : crayons_Bobbie = 3 * crayons_Billie)
  (h3 : crayons_Lizzie = crayons_Bobbie / 2)
  (h4 : crayons_Charlie = 2 * crayons_Lizzie) : 
  crayons_Charlie = 54 := 
sorry

end charlie_has_54_crayons_l89_89689


namespace merchant_discount_l89_89182

-- Definitions used in Lean 4 statement coming directly from conditions
def initial_cost_price : Real := 100
def marked_up_percentage : Real := 0.80
def profit_percentage : Real := 0.35

-- To prove the percentage discount offered
theorem merchant_discount (cp mp sp discount percentage_discount : Real) 
  (H1 : cp = initial_cost_price)
  (H2 : mp = cp + (marked_up_percentage * cp))
  (H3 : sp = cp + (profit_percentage * cp))
  (H4 : discount = mp - sp)
  (H5 : percentage_discount = (discount / mp) * 100) :
  percentage_discount = 25 := 
sorry

end merchant_discount_l89_89182


namespace number_square_of_digits_l89_89970

theorem number_square_of_digits (x y : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) :
  ∃ n : ℕ, (∃ (k : ℕ), (1001 * x + 110 * y) = k^2) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end number_square_of_digits_l89_89970


namespace min_point_is_centroid_l89_89650

variables {A B C P : Type*}
variables (xa1 xa2 xa3 ya1 ya2 ya3 : ℝ)
variables (x y : ℝ)

-- Definition of the squared distance to each vertex
def squared_distance (xi yi : ℝ) (x y : ℝ) : ℝ := 
  (x - xi)^2 + (y - yi)^2

-- Sum of squared distances definition
def sum_squared_distances (P : ℝ × ℝ) :=
  squared_distance xa1 ya1 P.1 P.2 +
  squared_distance xa2 ya2 P.1 P.2 +
  squared_distance xa3 ya3 P.1 P.2

theorem min_point_is_centroid 
  (xa1 xa2 xa3 ya1 ya2 ya3 : ℝ) :
  ∃ (P : ℝ × ℝ), 
    sum_squared_distances xa1 ya1 xa2 ya2 xa3 ya3 P = (sum_squared_distances xa1 ya1 xa2 ya2 xa3 ya3 ( 
    let x := (xa1 + xa2 + xa3) / 3 in
    let y := (ya1 + ya2 + ya3) / 3 in
    (x, y)) ∧ 
    ∀ (Q : ℝ × ℝ),  sum_squared_distances xa1 ya1 xa2 ya2 xa3 ya3 P ≤ sum_squared_distances xa1 ya1 xa2 ya2 xa3 ya3 Q : 
  ∃ (P : ℝ × ℝ), 
    P =  (( (xa1 + xa2 + xa3) / 3), ((ya1 + ya2 + ya3) / 3)))
:=
by
  intros
  use (( (xa1 + xa2 + xa3) / 3), ((ya1 + ya2 + ya3) / 3))
  sorry -- Details of the proof go here


end min_point_is_centroid_l89_89650


namespace area_shaded_region_l89_89030

theorem area_shaded_region (a b : ℕ) (area_square1 area_triangle : ℕ) (area_shaded : ℕ) 
(h1 : a = 4) (h2 : b = 12) 
(h3 : area_square1 = a * a)
(h4 : let DG = 3 in area_triangle = (1/2) * DG.to_nat * a)
(h5 : DG = 12 / 16 * a.to_nat)
(h6 : area_shaded = area_square1 - area_triangle)
: area_shaded = 10 := 
sorry

end area_shaded_region_l89_89030


namespace no_carry_consecutive_pairs_l89_89472

/-- Consider the range of integers {2000, 2001, ..., 3000}. 
    We determine that the number of pairs of consecutive integers in this range such that their addition requires no carrying is 729. -/
theorem no_carry_consecutive_pairs : 
  ∀ (n : ℕ), (2000 ≤ n ∧ n < 3000) ∧ ((n + 1) ≤ 3000) → 
  ∃ (count : ℕ), count = 729 := 
sorry

end no_carry_consecutive_pairs_l89_89472


namespace number_is_9_l89_89971

theorem number_is_9 : ∃ (x : ℤ), 45 - 3 * x = 18 ∧ x = 9 :=
by {
  use 9,
  split,
  calc
    45 - 3 * 9 = 45 - 27 : by norm_num
    ... = 18 : by norm_num,
  refl,
  sorry
}

end number_is_9_l89_89971


namespace tan_315_degree_l89_89404

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89404


namespace highest_qualification_number_l89_89733

-- Define the conditions of the problem.
def wrestler_wins (n m : ℕ) : Prop :=
  if abs (n - m) > 2 then n < m else true

-- Define the single-elimination tournament configuration.
def single_elimination (n : ℕ) : Prop :=
  ∃ rounds, rounds = 8 ∧ n = 256

-- The main theorem to prove the highest qualification number the winner can have.
theorem highest_qualification_number : 
  (∀ (n m : ℕ), abs (n - m) > 2 → n < m) → 
  (∃ rounds, rounds = 8 ∧ n = 256) → 
  ∃ k, k ≤ 16 :=
by
  intro h,
  intro t,
  use 16,
  sorry

end highest_qualification_number_l89_89733


namespace tan_315_eq_neg1_l89_89381

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89381


namespace gini_coefficient_when_operating_separately_gini_coefficient_change_after_combination_l89_89579

section GiniCalculation

-- Conditions (definitions of populations and production functions)
def n_population := 24
def s_population := n_population / 4
def n_ppf (x : ℝ) := 13.5 - 9 * x
def s_ppf (x : ℝ) := 1.5 * x^2 - 24
def set_price := 2000
def set_y_to_x_ratio := 9

-- questions
def question1 := "What is the Gini coefficient when both regions operate separately?"
def question2 := "How does the Gini coefficient change if the southern region agrees to the conditions of the northern region?"

-- Propositions (mathematically equivalent problems)
def proposition1 : Prop :=
  calc_gini_coefficient (regions_operate_separately n_population s_population n_ppf s_ppf set_y_to_x_ratio set_price) = 0.2

def proposition2 : Prop :=
  calc_gini_change_after_combination (combine_production_resources n_population s_population n_ppf s_ppf set_y_to_x_ratio set_price 661) = 0.001

noncomputable def regions_operate_separately (n_pop s_pop : ℝ) 
  (n_ppf s_ppf : ℝ → ℝ) (ratio price : ℝ) := sorry

noncomputable def combine_production_resources 
  (n_pop s_pop : ℝ) (n_ppf s_ppf : ℝ → ℝ)
  (ratio price : ℝ) (fee : ℝ) := sorry

noncomputable def calc_gini_coefficient : ℝ := sorry

noncomputable def calc_gini_change_after_combination : ℝ := sorry

-- Lean 4 Statements (no proof provided)
theorem gini_coefficient_when_operating_separately : proposition1 := by sorry
theorem gini_coefficient_change_after_combination : proposition2 := by sorry

end GiniCalculation

end gini_coefficient_when_operating_separately_gini_coefficient_change_after_combination_l89_89579


namespace island_challenge_probability_l89_89112
open Nat

theorem island_challenge_probability :
  let total_ways := choose 20 3
  let ways_one_tribe := choose 10 3
  let combined_ways := 2 * ways_one_tribe
  let probability := combined_ways / total_ways
  probability = (20 : ℚ) / 95 :=
by
  sorry

end island_challenge_probability_l89_89112


namespace at_least_one_woman_probability_l89_89551

noncomputable def probability_at_least_one_woman_selected 
  (total_men : ℕ) (total_women : ℕ) (selected_people : ℕ) : ℚ :=
  1 - (8 / 12 * 7 / 11 * 6 / 10 * 5 / 9)

theorem at_least_one_woman_probability :
  probability_at_least_one_woman_selected 8 4 4 = 85 / 99 := 
sorry

end at_least_one_woman_probability_l89_89551


namespace trisect_length_AG_l89_89171

theorem trisect_length_AG (A G E F N : Type) [LinearOrder A] [LinearOrder G] [LinearOrder E] [LinearOrder F] [LinearOrder N]
  (trisect : ∀ (a g e f : LinearOrder), e + f = a + g)
  (midpoint : ∀ (n a g : LinearOrder), n = (a + g) / 2)
  (trisect_length : E = E ∧ F = F ∧ 3 * E = A ∧ 3 * F = G ∧ E = F)
  (NF : F - N = 10) : 
  A + G = 30 := 
by 
  sorry

end trisect_length_AG_l89_89171


namespace vector_zero_conditions_l89_89150

namespace Vectors

variables {V : Type*} [AddCommGroup V]

theorem vector_zero_conditions 
  (MB BO OM AB BC OB OC BO' CO AB' AC BD CD : V) 
  (h1 : MB + BO + OM = 0) 
  (h2 : AB + BC ≠ 0) 
  (h3 : OB + OC + BO' + CO = 0) 
  (h4 : AB' - AC + BD - CD = 0) : 
  MB + BO + OM = 0 ∧ OB + OC + BO' + CO = 0 ∧ AB' - AC + BD - CD = 0 :=
by
  split
  · exact h1
  split
  · exact h3
  · exact h4

end Vectors

end vector_zero_conditions_l89_89150


namespace reachable_cells_after_moves_l89_89634

def is_valid_move (n : ℕ) (x y : ℤ) : Prop :=
(abs x ≤ n ∧ abs y ≤ n ∧ (x + y) % 2 = 0)

theorem reachable_cells_after_moves (n : ℕ) :
  n = 10 → ∃ (cells : Finset (ℤ × ℤ)), cells.card = 121 ∧ 
  (∀ (cell : ℤ × ℤ), cell ∈ cells → is_valid_move n cell.1 cell.2) :=
by
  intros h
  use {-10 ≤ x, y, x + y % 2 = 0 & abs x + abs y ≤ n }
  sorry -- proof goes here

end reachable_cells_after_moves_l89_89634


namespace remainder_div_252_l89_89930

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l89_89930


namespace triangle_angle_A_sixty_l89_89573

theorem triangle_angle_A_sixty
  (A B C : ℝ)
  (BD CE intersect I : Point)
  (D E : Point)
  (I_perpendicular_DE_IQ_2IP : IQ = 2 * IP)
  (H1 : IsTriangle ABC)
  (H2 : AngleBisector BD)
  (H3 : AngleBisector CE)
  (H4 : Intersects I BD CE)
  (H5 : LiesOn D AC)
  (H6 : LiesOn E AB)
  (H7 : Perpendicular I DE)
  (H8 : Intersects P DE)
  (H9 : LiesOn Q PI)
  (H10 : Intersects Q BC) :
  Angle A = 60 :=
sorry

end triangle_angle_A_sixty_l89_89573


namespace cubic_root_expression_l89_89681

theorem cubic_root_expression (p q r : ℝ)
  (h₁ : p + q + r = 8)
  (h₂ : p * q + p * r + q * r = 11)
  (h₃ : p * q * r = 3) :
  p / (q * r + 1) + q / (p * r + 1) + r / (p * q + 1) = 32 / 15 :=
by 
  sorry

end cubic_root_expression_l89_89681


namespace tan_315_eq_neg_one_l89_89225

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89225


namespace g_triple_apply_l89_89070

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_triple_apply : g (g (g 20)) = 1 :=
by
  sorry

end g_triple_apply_l89_89070


namespace ratio_R_l89_89508

variables (a b c : ℝ)
variable (k : ℝ)

-- Conditions: a, b, c are not zero and their ratios are 6:3:1 respectively.
def valid_ratios (a b c : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a / b = 6 / 3 ∧ b / c = 3 / 1

-- The value of R given the conditions.
def R_value (a b c : ℝ) : ℝ := (3 * b^2) / (2 * a^2 + b * c)

-- The theorem to prove
theorem ratio_R (a b c : ℝ) (h : valid_ratios a b c) : R_value a b c = 9 / 25 := by
  sorry

end ratio_R_l89_89508


namespace problem_solution_l89_89993

variable (a b c : ℝ)

theorem problem_solution (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) :
  a + b ≤ 3 * c := 
sorry

end problem_solution_l89_89993


namespace tan_315_eq_neg_one_l89_89219

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89219


namespace aeroplane_distance_l89_89863

theorem aeroplane_distance
  (speed : ℝ) (time : ℝ) (distance : ℝ)
  (h1 : speed = 590)
  (h2 : time = 8)
  (h3 : distance = speed * time) :
  distance = 4720 :=
by {
  -- The proof will contain the steps to show that distance = 4720
  sorry
}

end aeroplane_distance_l89_89863


namespace remainder_when_divided_l89_89934

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l89_89934


namespace solution_m_plus_n_l89_89540

variable (m n : ℝ)

theorem solution_m_plus_n 
  (h₁ : m ≠ 0)
  (h₂ : m^2 + m * n - m = 0) :
  m + n = 1 := by
  sorry

end solution_m_plus_n_l89_89540


namespace isosceles_triangle_vertex_angle_l89_89016

theorem isosceles_triangle_vertex_angle (θ : ℝ) (h₀ : θ = 80) (h₁ : ∃ (x y z : ℝ), (x = y ∨ y = z ∨ z = x) ∧ x + y + z = 180) : θ = 80 ∨ θ = 20 := 
sorry

end isosceles_triangle_vertex_angle_l89_89016


namespace reachable_cells_after_10_moves_l89_89619

theorem reachable_cells_after_10_moves :
  let board_size := 21
  let central_cell := (11, 11)
  let moves := 10
  (reachable_cells board_size central_cell moves) = 121 :=
by
  sorry

end reachable_cells_after_10_moves_l89_89619


namespace length_EF_l89_89575

-- Variables and Definitions
variable (A B C E F : Type) [triangle A B C]
variables (a b c e f : ℝ)

-- Conditions
def isIsosceles (A B C : Type) [triangle A B C] (ab ac : ℝ) := ab = ac
def midpoint (E : Type) [Point E] (B C : Type) [segment B C] := E
def perpendicular (EF AB : Type) [line EF AB] (E F B : Permission.bool) := EF ⊥ AB 

-- Problem Statement
theorem length_EF (A B C E F : Type) [Point A B C E F] [segment A B E] :
  isIsosceles A B C 5 5 → 
  length E (midpoint 8) = 4 → 
  perpendicular E F A B :
  EF = 12 / 5 := sorry

end length_EF_l89_89575


namespace remainder_div_252_l89_89926

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l89_89926


namespace problem1_problem2_problem3_problem4_l89_89213

-- (1) Prove (1 + sqrt 3) * (2 - sqrt 3) = -1 + sqrt 3
theorem problem1 : (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = -1 + Real.sqrt 3 :=
by sorry

-- (2) Prove (sqrt 36 * sqrt 12) / sqrt 3 = 12
theorem problem2 : (Real.sqrt 36 * Real.sqrt 12) / Real.sqrt 3 = 12 :=
by sorry

-- (3) Prove sqrt 18 - sqrt 8 + sqrt (1 / 8) = (5 * sqrt 2) / 4
theorem problem3 : Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1 / 8) = (5 * Real.sqrt 2) / 4 :=
by sorry

-- (4) Prove (3 * sqrt 18 + (1 / 5) * sqrt 50 - 4 * sqrt (1 / 2)) / sqrt 32 = 2
theorem problem4 : (3 * Real.sqrt 18 + (1 / 5) * Real.sqrt 50 - 4 * Real.sqrt (1 / 2)) / Real.sqrt 32 = 2 :=
by sorry

end problem1_problem2_problem3_problem4_l89_89213


namespace no_tiling_10x10_1x4_l89_89037

-- Define the problem using the given conditions
def checkerboard_tiling (n k : ℕ) : Prop :=
  ∃ t : ℕ, t * k = n * n ∧ n % k = 0

-- Prove that it is impossible to tile a 10x10 board with 1x4 tiles
theorem no_tiling_10x10_1x4 : ¬ checkerboard_tiling 10 4 :=
sorry

end no_tiling_10x10_1x4_l89_89037


namespace cone_lateral_surface_area_l89_89487

-- Definitions and conditions
def radius : ℝ := 2
def height : ℝ := 1
def slant_height : ℝ := Real.sqrt (radius^2 + height^2)

-- The question and expected answer
def lateral_surface_area : ℝ := 2 * Real.sqrt 5 * Real.pi

-- The statement to prove
theorem cone_lateral_surface_area :
  ∀ (R h : ℝ), R = radius → h = height → 
  2 * R * Real.sqrt (R^2 + h^2) * Real.pi = lateral_surface_area :=
by
  intros R h hR hH
  rw [hR, hH]
  sorry

end cone_lateral_surface_area_l89_89487


namespace num_cells_after_10_moves_l89_89615

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 + 1 = p2.1))

def num_reachable_cells (n m k : ℕ) (start : ℕ × ℕ) : ℕ :=
  sorry -- Calculation of the number of reachable cells after k moves

theorem num_cells_after_10_moves :
  let board_size := 21
  let start := (11, 11)
  let moves := 10
  num_reachable_cells board_size board_size moves start = 121 :=
sorry

end num_cells_after_10_moves_l89_89615


namespace oakwood_high_school_math_team_l89_89745

open Finset

/-- Define a combinatorial function to calculate combinations -/
def choose (n k : ℕ) : ℕ := nat.choose n k

/-- The problem statement to be proved -/
theorem oakwood_high_school_math_team :
  (choose 4 2) * (choose 6 2) = 90 := 
by
  sorry

end oakwood_high_school_math_team_l89_89745


namespace intersection_point_l89_89647

-- Definitions of the parametric equations of curve C
def curve_C_x (α : ℝ) : ℝ := sin α + cos α
def curve_C_y (α : ℝ) : ℝ := 1 + sin (2 * α)

-- Definition of the rectangular coordinate form of the line l
def line_l (x y : ℝ) : Prop := y - x = 2

-- Definition of the rectangular form of the curve C
def curve_C (x y : ℝ) : Prop := x^2 = y

-- Prove that the intersection point of line l and curve C is (-1, 1)
theorem intersection_point : (line_l (-1) 1) ∧ (curve_C (-1) 1) :=
by
  -- Proof can be filled here
  sorry

end intersection_point_l89_89647


namespace crayons_total_cost_l89_89693

theorem crayons_total_cost :
  let packs_initial := 4
  let packs_to_buy := 2
  let cost_per_pack := 2.5
  let total_packs := packs_initial + packs_to_buy
  let total_cost := total_packs * cost_per_pack
  total_cost = 15 :=
by
  sorry

end crayons_total_cost_l89_89693


namespace correct_answer_is_B_l89_89858

-- Define the conditions
def num_students := 11623
def num_questions := 12
def points_per_question := 5
def difficulty_coefficient := 0.34
def average_score := difficulty_coefficient * points_per_question

-- Define the student answer statistics
def option_A_percentage := 36.21 / 100
def option_B_percentage := 33.85 / 100
def option_C_percentage := 17.7 / 100
def option_D_percentage := 11.96 / 100

-- Define the average score calculation for each option
def average_score_A := option_A_percentage * points_per_question
def average_score_B := option_B_percentage * points_per_question
def average_score_C := option_C_percentage * points_per_question
def average_score_D := option_D_percentage * points_per_question

-- The theorem stating that option B is the correct answer
theorem correct_answer_is_B : average_score = average_score_B :=
by 
  -- calculate and verify the values 
  sorry

end correct_answer_is_B_l89_89858


namespace number_of_reachable_cells_after_10_moves_l89_89642

theorem number_of_reachable_cells_after_10_moves : 
  (let 
    n := 21 
    center := (11, 11)
    moves := 10
  in
  ∃ reachable_cells, reachable_cells = 121) :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89642


namespace equation_of_plane_l89_89606

-- Definitions based on conditions
def line_equation (A B C x y : ℝ) : Prop :=
  A * x + B * y + C = 0

def A_B_nonzero (A B : ℝ) : Prop :=
  A ^ 2 + B ^ 2 ≠ 0

-- Statement for the problem
noncomputable def plane_equation (A B C D x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem equation_of_plane (A B C D : ℝ) :
  (A ^ 2 + B ^ 2 + C ^ 2 ≠ 0) → (∀ x y z : ℝ, plane_equation A B C D x y z) :=
by
  sorry

end equation_of_plane_l89_89606


namespace possible_distribution_iff_odd_l89_89680

def is_magic_pair (a b : ℕ) (n : ℕ) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 1 ∧ b = (n * (n - 1)) / 2) ∨ (b = 1 ∧ a = (n * (n - 1)) / 2)

def can_distribute_magic_pairs (n : ℕ) : Prop :=
  ∃ (stacks : list (list ℕ)), stacks.length = n ∧
  (∀ i j, i ≠ j → ∃ a b, a ∈ stacks.nth_le i sorry ∧ b ∈ stacks.nth_le j sorry ∧ is_magic_pair a b n)

theorem possible_distribution_iff_odd (n : ℕ) (h : n > 2) : can_distribute_magic_pairs n ↔ odd n :=
sorry

end possible_distribution_iff_odd_l89_89680


namespace tan_315_eq_neg1_l89_89394

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89394


namespace tan_315_eq_neg1_l89_89235

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89235


namespace arithmetic_mean_after_removal_l89_89735

theorem arithmetic_mean_after_removal 
  (S : Finset ℝ) (hS : S.card = 60) 
  (mean_S : 42 = S.sum / 60) 
  (a b c : ℝ) (h_vals : a = 50 ∧ b = 60 ∧ c = 70)
  (h_in : a ∈ S ∧ b ∈ S ∧ c ∈ S) :
  (S.erase a).erase b).erase c).sum / 57 ≈ 41.0 :=
  sorry

end arithmetic_mean_after_removal_l89_89735


namespace unsold_fruits_correct_l89_89859

def initial_stock : Type := 
  { kidney_apples : ℕ, golden_apples : ℕ, canada_apples : ℕ, fuji_apples : ℕ, granny_smith_apples : ℕ, 
    valencia_oranges : ℕ, navel_oranges : ℕ, cavendish_bananas : ℕ, ladyfinger_bananas : ℕ }

def sold_fruits : Type :=
  { kidney_apples : ℕ, golden_apples : ℕ, canada_apples : ℕ, fuji_apples : ℕ, granny_smith_apples : ℕ,
    valencia_oranges : ℕ, navel_oranges : ℕ, cavendish_bananas : ℕ, ladyfinger_bananas : ℕ }

def unsold_fruits (initial : initial_stock) (sold : sold_fruits) : Type :=
  { kidney_apples := initial.kidney_apples - sold.kidney_apples,
    golden_apples := initial.golden_apples - sold.golden_apples,
    canada_apples := initial.canada_apples - sold.canada_apples,
    fuji_apples := initial.fuji_apples - sold.fuji_apples,
    granny_smith_apples := initial.granny_smith_apples - sold.granny_smith_apples,
    valencia_oranges := initial.valencia_oranges - sold.valencia_oranges,
    navel_oranges := initial.navel_oranges - sold.navel_oranges,
    cavendish_bananas := initial.cavendish_bananas - sold.cavendish_bananas,
    ladyfinger_bananas := initial.ladyfinger_bananas - sold.ladyfinger_bananas }

theorem unsold_fruits_correct (initial : initial_stock) (sold : sold_fruits) :
  initial.kidney_apples = 26 → sold.kidney_apples = 15 →
  initial.golden_apples = 42 → sold.golden_apples = 28 →
  initial.canada_apples = 19 → sold.canada_apples = 12 →
  initial.fuji_apples = 35 → sold.fuji_apples = 20 →
  initial.granny_smith_apples = 22 → sold.granny_smith_apples = 18 →
  initial.valencia_oranges = 40 → sold.valencia_oranges = 25 →
  initial.navel_oranges = 28 → sold.navel_oranges = 19 →
  initial.cavendish_bananas = 33 → sold.cavendish_bananas = 27 →
  initial.ladyfinger_bananas = 17 → sold.ladyfinger_bananas = 10 →
  unsold_fruits initial sold =
    { kidney_apples := 11,
      golden_apples := 14,
      canada_apples := 7,
      fuji_apples := 15,
      granny_smith_apples := 4,
      valencia_oranges := 15,
      navel_oranges := 9,
      cavendish_bananas := 6,
      ladyfinger_bananas := 7 } :=
by sorry

end unsold_fruits_correct_l89_89859


namespace brad_running_speed_l89_89075

def maxwell_speed : ℝ := 2  -- Maxwell's walking speed in km/h
def distance_between_homes : ℝ := 36  -- Distance between homes in km
def maxwell_distance : ℝ := 12  -- Distance Maxwell has traveled when they meet in km
def meeting_time : ℝ := maxwell_distance / maxwell_speed -- Time taken by Maxwell to meet Brad

theorem brad_running_speed :
  ∃ brad_speed : ℝ, brad_speed = (distance_between_homes - maxwell_distance) / meeting_time :=
by
  use 4
  sorry

end brad_running_speed_l89_89075


namespace teenas_speed_l89_89098

theorem teenas_speed (T : ℝ) :
  (7.5 + 15 + 40 * 1.5 = T * 1.5) → T = 55 := 
by
  intro h
  sorry

end teenas_speed_l89_89098


namespace area_triangle_BMN_of_parallelogram_and_circle_l89_89603

theorem area_triangle_BMN_of_parallelogram_and_circle
  (α r : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hr : 0 < r)
  (M N : Point)
  (h_parallel : parallelogram ABCD)
  (h_circle : circle_of_radius_passing_through_vertices_and_intersects_lines M N α r ABCD) :
  area_triangle_BMN = 2 * r^2 * sin(α)^2 * sin(2 * α) :=
sorry

end area_triangle_BMN_of_parallelogram_and_circle_l89_89603


namespace tan_sum_identity_l89_89959

open Real

theorem tan_sum_identity : 
  tan (80 * π / 180) + tan (40 * π / 180) - sqrt 3 * tan (80 * π / 180) * tan (40 * π / 180) = -sqrt 3 :=
by
  sorry

end tan_sum_identity_l89_89959


namespace number_of_lectures_l89_89709

theorem number_of_lectures (n : ℕ) (h₁ : n = 8) : 
  (Nat.choose n 3) + (Nat.choose n 2) = 84 :=
by
  rw [h₁]
  sorry

end number_of_lectures_l89_89709


namespace final_population_of_bacteria_l89_89120

theorem final_population_of_bacteria (P0 : ℕ) (d t : ℕ) 
  (h1 : P0 = 1000) 
  (h2 : d = 2) 
  (h3 : t = 20) 
  (h4 : ∀ (n : ℕ), P0 * 2 ^ (t / d) = 1024000) :
  P0 * 2 ^ (t / d) = 1024000 := 
by 
  rw [h1, h2, h3];
  exact h4 10

end final_population_of_bacteria_l89_89120


namespace tan_315_eq_neg1_l89_89396

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89396


namespace rational_comparison_correct_l89_89198

-- Definitions based on conditions 
def positive_gt_zero (a : ℚ) : Prop := 0 < a
def negative_lt_zero (a : ℚ) : Prop := a < 0
def positive_gt_negative (a b : ℚ) : Prop := positive_gt_zero a ∧ negative_lt_zero b ∧ a > b
def negative_comparison (a b : ℚ) : Prop := negative_lt_zero a ∧ negative_lt_zero b ∧ abs a > abs b ∧ a < b

-- Theorem to prove
theorem rational_comparison_correct :
  (0 < - (1 / 2)) = false ∧
  ((4 / 5) < - (6 / 7)) = false ∧
  ((9 / 8) > (8 / 9)) = true ∧
  (-4 > -3) = false :=
by
  -- Mark the proof as unfinished.
  sorry

end rational_comparison_correct_l89_89198


namespace race_time_l89_89808

theorem race_time (v_A v_B : ℝ) (t_A t_B : ℝ) (h1 : v_A = 1000 / t_A) (h2 : v_B = 952 / (t_A + 6)) (h3 : v_A = v_B) : t_A = 125 :=
by
  sorry

end race_time_l89_89808


namespace tan_315_eq_neg_1_l89_89324

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89324


namespace solution_set_of_inequality_l89_89759

theorem solution_set_of_inequality (x : ℝ) :
  (x + 2) ^ (-(3 / 5)) < (5 - 2 * x) ^ (-(3 / 5)) ↔ x ∈ (Set.Ioo 1 (5 / 2) ∪ Set.Iio (-2)) := by
  sorry

end solution_set_of_inequality_l89_89759


namespace quadratic_real_roots_range_l89_89966

theorem quadratic_real_roots_range (m : ℝ) :
  ∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0 ↔ m ≤ 4 ∧ m ≠ 3 := 
by
  sorry

end quadratic_real_roots_range_l89_89966


namespace tan_315_eq_neg_1_l89_89318

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89318


namespace system1_solution_system2_solution_l89_89725

theorem system1_solution (x y : ℝ) (h₁ : y = 2 * x) (h₂ : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 := 
by sorry

theorem system2_solution (x y : ℝ) (h₁ : x - 3 * y = -2) (h₂ : 2 * x + 3 * y = 3) : x = (1 / 3) ∧ y = (7 / 9) := 
by sorry

end system1_solution_system2_solution_l89_89725


namespace eval_expression_l89_89438

theorem eval_expression : 
  (8^5) / (4 * 2^5 + 16) = 2^11 / 9 :=
by
  sorry

end eval_expression_l89_89438


namespace inequality_lean_statement_l89_89501

theorem inequality_lean_statement (n : ℕ) (h : 3 ≤ n) (x : Finₓ (n + 2) → ℝ) 
  (hpos : ∀ i, 0 < x i) :
  (∑ i : Fin n, x i / (x (i + 1) + x (i + 2))) ≥ (Real.sqrt 2 - 1) * n :=
by
  sorry

end inequality_lean_statement_l89_89501


namespace vasya_wins_optimal_play_l89_89747

theorem vasya_wins_optimal_play :
  let game_start := 2017
  in (∀ initial_number : ℕ, 
      (initial_number = game_start) ∧ 
      (∀ turns : ℕ, turns ≥ 0 → 
      (∃ moves : ℕ, moves > 0 ∧ moves < initial_number)) →
      (turns % 2 ≠ 0 ∨ initial_number = 1) ∧
      (turns % 2 = 0 → initial_number ≠ 1) 
    ) →
    ("Vasya" wins with optimal play) sorry

end vasya_wins_optimal_play_l89_89747


namespace sum_distances_from_vertex_to_midpoints_of_rectangle_l89_89869

theorem sum_distances_from_vertex_to_midpoints_of_rectangle
  (A B C D : ℝ × ℝ) 
  (h : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 3 ∧ B.2 = 0 ∧ C.1 = 3 ∧ C.2 = 5 ∧ D.1 = 0 ∧ D.2 = 5) :
  let M := (1.5, 0)
    N := (3, 2.5)
    O := (1.5, 5)
    P := (0, 2.5) in
  dist A M + dist A N + dist A O + dist A P = 13.1 := 
by sorry

end sum_distances_from_vertex_to_midpoints_of_rectangle_l89_89869


namespace annette_weights_more_l89_89204

variable (A C S B : ℝ)

theorem annette_weights_more :
  A + C = 95 ∧
  C + S = 87 ∧
  A + S = 97 ∧
  C + B = 100 ∧
  A + C + B = 155 →
  A - S = 8 := by
  sorry

end annette_weights_more_l89_89204


namespace remainder_123456789012_div_252_l89_89914

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l89_89914


namespace tan_315_eq_neg_one_l89_89218

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89218


namespace number_of_reachable_cells_after_10_moves_l89_89610

-- Define board size, initial position, and the number of moves
def board_size : ℕ := 21
def initial_position : ℕ × ℕ := (11, 11)
def moves : ℕ := 10

-- Define the main problem statement
theorem number_of_reachable_cells_after_10_moves :
  (reachable_cells board_size initial_position moves).card = 121 :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89610


namespace inequality_solution_set_l89_89526

def f (x : ℝ) : ℝ := x^3

theorem inequality_solution_set (x : ℝ) :
  (f (2 * x) + f (x - 1) < 0) ↔ (x < (1 / 3)) := 
sorry

end inequality_solution_set_l89_89526


namespace tan_315_eq_neg_one_l89_89222

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89222


namespace prime_product_2002_l89_89096

theorem prime_product_2002 {a b c d : ℕ} (ha_prime : Prime a) (hb_prime : Prime b) (hc_prime : Prime c) (hd_prime : Prime d)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : a + c = d)
  (h2 : a * (a + b + c + d) = c * (d - b))
  (h3 : 1 + b * c + d = b * d) :
  a * b * c * d = 2002 := 
by 
  sorry

end prime_product_2002_l89_89096


namespace circle_radius_is_10_l89_89183

-- Define the conditions using Lean definitions
def area (r : ℝ) : ℝ := π * r^2
def circumference (r : ℝ) : ℝ := 2 * π * r
def sum_x_y (x y : ℝ) : ℝ := x + y

-- State the theorem matching the problem statement
theorem circle_radius_is_10 (x y r : ℝ) (h1 : x = area r) (h2 : y = circumference r) (h3 : sum_x_y x y = 90 * π) : r = 10 := 
  sorry

end circle_radius_is_10_l89_89183


namespace cube_path_length_l89_89834

noncomputable def path_length_dot_cube : ℝ :=
  let edge_length := 2
  let radius1 := Real.sqrt 5
  let radius2 := 1
  (radius1 + radius2) * Real.pi

theorem cube_path_length :
  path_length_dot_cube = (Real.sqrt 5 + 1) * Real.pi :=
by
  sorry

end cube_path_length_l89_89834


namespace segment_AE_length_l89_89005

structure Point (α : Type) := (x : α) (y : α)

noncomputable def AE_length : ℚ := Real.sqrt ((203 / 23)^2 + (1.56 - 4)^2)

def A : Point ℚ := ⟨0, 4⟩
def B : Point ℚ := ⟨7, 0⟩
def C : Point ℚ := ⟨5, 3⟩
def D : Point ℚ := ⟨3, 0⟩

-- Coordinates of intersection E are derived but not directly used in the Lean statement.

theorem segment_AE_length :
  AE_length ≈ 9.16 := sorry -- Using a dummy proof for approximation

end segment_AE_length_l89_89005


namespace find_angle_x_l89_89645

-- Define the angles and parallel lines conditions
def parallel_lines (k l : Prop) (angle1 : Real) (angle2 : Real) : Prop :=
  k ∧ l ∧ angle1 = 30 ∧ angle2 = 90

-- Statement of the problem in Lean syntax
theorem find_angle_x (k l : Prop) (angle1 angle2 : Real) (x : Real) : 
  parallel_lines k l angle1 angle2 → x = 150 :=
by
  -- Assuming conditions are given, prove x = 150
  sorry

end find_angle_x_l89_89645


namespace tan_315_eq_neg1_l89_89241

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89241


namespace problem1_problem2_l89_89523

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) ^ 2 + 2 * (Real.cos x) ^ 2

theorem problem1 :
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi)
  ∧ (∃ k : ℤ, ∀ x ∈ Set.Icc (Real.pi / 8 + k * Real.pi) (5 * Real.pi / 8 + k * Real.pi), f(x) ≤ f(x + ε) → ε ≤ 0) :=
sorry

theorem problem2 :
  ∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi) (Real.pi / 4 + k * Real.pi), f(x) ≥ 3 :=
sorry

end problem1_problem2_l89_89523


namespace tan_315_degrees_l89_89358

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89358


namespace tan_315_eq_neg_one_l89_89300

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89300


namespace tan_315_eq_neg1_l89_89332

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89332


namespace number_of_lectures_l89_89708

theorem number_of_lectures (n : ℕ) (h₁ : n = 8) : 
  (Nat.choose n 3) + (Nat.choose n 2) = 84 :=
by
  rw [h₁]
  sorry

end number_of_lectures_l89_89708


namespace first_player_always_wins_l89_89769

theorem first_player_always_wins :
  ∀ (cards : Finset ℕ) (n : ℕ), 
    (∀ k, k ∈ cards → k ≤ n)
    ∧ (cards.card = 2002)
    ∧ (∀ i, ∃ j, i ∈ cards → j % 10 = i % 10) → 
    (first_player wins regardless of second player's strategy) := by 
    sorry

end first_player_always_wins_l89_89769


namespace necessary_not_sufficient_l89_89509

theorem necessary_not_sufficient (m : ℝ) (x : ℝ) (h₁ : m > 0) (h₂ : 0 < x ∧ x < m) (h₃ : x / (x - 1) < 0) 
: m = 1 / 2 := 
sorry

end necessary_not_sufficient_l89_89509


namespace boundary_polygon_sides_eq_five_l89_89055

variable {a : ℝ} (h_a : 0 < a)

def S (x y : ℝ) : Prop :=
  1 / 2 * a ≤ x ∧ x ≤ 2 * a ∧
  1 / 2 * a ≤ y ∧ y ≤ 2 * a ∧
  x + y ≥ 3 * a ∧
  x + a ≥ y ∧
  y + a ≥ x

theorem boundary_polygon_sides_eq_five : 
  ∀ (x y : ℝ), S a x y → (∃ p : ℕ, p = 5) :=
by
  sorry

end boundary_polygon_sides_eq_five_l89_89055


namespace jerry_added_action_figures_l89_89658

theorem jerry_added_action_figures (x : ℕ) (h1 : 7 + x - 10 = 8) : x = 11 :=
by
  sorry

end jerry_added_action_figures_l89_89658


namespace PX_is_40_l89_89027

def line_segments_parallel (CD WX : Set (ℝ × ℝ)) : Prop :=
  ∃ m b₁ b₂ : ℝ, CD = {p | p.2 = m * p.1 + b₁} ∧ WX = {p | p.2 = m * p.1 + b₂}

def point (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def PX_length : ℝ :=
  let CX := 60
  let DP := 18
  let PW := 36
  let CP := PX / 2
  let PX := (CP + PX) = CX / (3 / 2)
      sorry

theorem PX_is_40 : PX_length = 40 := by
  sorry

end PX_is_40_l89_89027


namespace triangle_area_correct_l89_89135

-- Define the first line equation
def line1 (x : ℝ) : ℝ := (3/4) * x + 9/4

-- Define the second line equation
def line2 (x : ℝ) : ℝ := -2 * x + 5

-- Define the third line equation
def line3 (x y : ℝ) : Prop := x + y = 8

-- Define the area of the triangle using given vertices (1,3), (-3,11), (23/7, 32/7)
def vertex_A := (1 : ℝ, 3 : ℝ)
def vertex_B := (-3 : ℝ, 11 : ℝ)
def vertex_C := (23/7 : ℝ, 32/7 : ℝ)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

-- Statement asserting that the area of the triangle is 86/7
theorem triangle_area_correct : 
  triangle_area vertex_A vertex_B vertex_C = 86 / 7 := 
sorry

end triangle_area_correct_l89_89135


namespace b_2023_value_l89_89056

def sequence (n : ℕ) : ℤ 
| 1       := 1
| 2       := 4
| (n + 1) := sequence n - sequence (n - 1)

theorem b_2023_value : sequence 2023 = -4 :=
by sorry

end b_2023_value_l89_89056


namespace tan_315_eq_neg_1_l89_89328

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89328


namespace probability_even_five_digit_number_l89_89972

theorem probability_even_five_digit_number :
  let total_events := Nat.choose 5 3 * Nat.choose 4 2 * Nat.factorial 5 in
  let favorable_events := Nat.choose 5 3 * Nat.choose 4 2 * 2 * Nat.factorial 4 in
  (favorable_events : ℚ) / (total_events : ℚ) = 2 / 5 :=
by 
  sorry

end probability_even_five_digit_number_l89_89972


namespace find_range_of_r_l89_89029

noncomputable def range_of_r : Set ℝ :=
  {r : ℝ | 3 * Real.sqrt 5 - 3 * Real.sqrt 2 ≤ r ∧ r ≤ 3 * Real.sqrt 5 + 3 * Real.sqrt 2}

theorem find_range_of_r 
  (O : ℝ × ℝ) (A : ℝ × ℝ) (r : ℝ) (h : r > 0)
  (hA : A = (0, 3))
  (C : Set (ℝ × ℝ)) (hC : C = {M : ℝ × ℝ | (M.1 - 3)^2 + (M.2 - 3)^2 = r^2})
  (M : ℝ × ℝ) (hM : M ∈ C)
  (h_cond : (M.1 - 0)^2 + (M.2 - 3)^2 = 2 * ((M.1 - 0)^2 + (M.2 - 0)^2)) :
  r ∈ range_of_r :=
sorry

end find_range_of_r_l89_89029


namespace area_triangle_PQR_l89_89782

theorem area_triangle_PQR (P : ℝ × ℝ) (PQ_slope PR_slope : ℝ) (Q R : ℝ × ℝ)
    (P_coords : P = (2, 5))
    (PQ_line : PQ_slope = 1 / 2)
    (PR_line : PR_slope = 3)
    (Q_coords : Q.2 = 0)
    (R_coords : R.2 = 0)
    (PQ_eq : (P.2 - Q.2) / (P.1 - Q.1) = PQ_slope)
    (PR_eq : (P.2 - R.2) / (P.1 - R.1) = PR_slope) :
    let QR := (R.1 - Q.1).abs in
    let height := (P.2 - 0).abs in
    let area := (1 / 2) * QR * height in
    area = 125 / 6 := sorry

end area_triangle_PQR_l89_89782


namespace ink_cost_computation_l89_89019

def class_A_whiteboards := 3
def class_B_whiteboards := 2
def class_C_whiteboards := 4
def class_D_whiteboards := 1
def class_E_whiteboards := 3

def class_A_ink_usage_per_whiteboard := 20
def class_B_ink_usage_per_whiteboard := 25
def class_C_ink_usage_per_whiteboard := 15
def class_D_ink_usage_per_whiteboard := 30
def class_E_ink_usage_per_whiteboard := 20

def class_A_ink_cost_per_ml := 0.50
def class_B_ink_cost_per_ml := 0.60
def class_C_ink_cost_per_ml := 0.40
def class_D_ink_cost_per_ml := 0.55
def class_E_ink_cost_per_ml := 0.45

def class_A_bottle_size := 100
def class_B_bottle_size := 150
def class_C_bottle_size := 50
def class_D_bottle_size := 75
def class_E_bottle_size := 125

def ink_cost_to_use_whiteboards_for_one_day : ℝ := 277.50

theorem ink_cost_computation :
  (class_A_whiteboards * class_A_ink_usage_per_whiteboard / class_A_bottle_size).ceil * class_A_bottle_size * class_A_ink_cost_per_ml +
  (class_B_whiteboards * class_B_ink_usage_per_whiteboard / class_B_bottle_size).ceil * class_B_bottle_size * class_B_ink_cost_per_ml +
  (class_C_whiteboards * class_C_ink_usage_per_whiteboard / class_C_bottle_size).ceil * class_C_bottle_size * class_C_ink_cost_per_ml +
  (class_D_whiteboards * class_D_ink_usage_per_whiteboard / class_D_bottle_size).ceil * class_D_bottle_size * class_D_ink_cost_per_ml +
  (class_E_whiteboards * class_E_ink_usage_per_whiteboard / class_E_bottle_size).ceil * class_E_bottle_size * class_E_ink_cost_per_ml = 
  ink_cost_to_use_whiteboards_for_one_day := by
  sorry

end ink_cost_computation_l89_89019


namespace binomial_sum_identity_l89_89439

open Nat

theorem binomial_sum_identity : 
  (∑ k in range 25, (-1 : ℤ)^k * (Nat.choose 49 (2 * k))) = -2^24 := by
  sorry

end binomial_sum_identity_l89_89439


namespace angle_AMC_150_l89_89101

/-- The given triangle and conditions -/
variables {A B C M : Type} 
variables (h1 : A ∠= 70)
variables (h2 : C ∠= 80)
variables (h3 : altitudes_from A C intersect_at M)

/-- Proof that the angle AMC is 150 degrees -/
theorem angle_AMC_150 : ∠ AMC = 150 :=
by
  sorry -- The proof is omitted as per instructions.

end angle_AMC_150_l89_89101


namespace num_comfortable_butterflies_final_state_l89_89068

noncomputable def num_comfortable_butterflies (n : ℕ) : ℕ :=
  if h : 0 < n then
    n
  else
    0

theorem num_comfortable_butterflies_final_state {n : ℕ} (h : 0 < n):
  num_comfortable_butterflies n = n := by
  sorry

end num_comfortable_butterflies_final_state_l89_89068


namespace max_value_pq_qr_rs_sp_l89_89119

variable (p q r s : ℕ)

theorem max_value_pq_qr_rs_sp :
  (p = 1 ∨ p = 3 ∨ p = 5 ∨ p = 7) →
  (q = 1 ∨ q = 3 ∨ q = 5 ∨ q = 7) →
  (r = 1 ∨ r = 3 ∨ r = 5 ∨ r = 7) →
  (s = 1 ∨ s = 3 ∨ s = 5 ∨ s = 7) →
  (p ≠ q) →
  (p ≠ r) →
  (p ≠ s) →
  (q ≠ r) →
  (q ≠ s) →
  (r ≠ s) →
  pq + qr + rs + sp ≤ 64 :=
sorry

end max_value_pq_qr_rs_sp_l89_89119


namespace remainder_when_divided_l89_89935

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l89_89935


namespace coins_division_remainder_l89_89174

theorem coins_division_remainder
  (n : ℕ)
  (h1 : n % 6 = 4)
  (h2 : n % 5 = 3)
  (h3 : n = 28) :
  n % 7 = 0 :=
by
  sorry

end coins_division_remainder_l89_89174


namespace book_return_percentage_l89_89843

variable (P : ℕ → ℕ → ℕ → ℕ)

theorem book_return_percentage 
  (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) 
  (h_initial : initial_books = 75) 
  (h_end : end_books = 64) 
  (h_loaned : loaned_books = 55) : 
  P initial_books end_books loaned_books = 80 :=
by 
  -- Define P as the function to calculate the percentage of returned books
  let P (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) := 
    let remaining_books := initial_books - loaned_books in
    let returned_books := end_books - remaining_books in
    returned_books * 100 / loaned_books
  have h_P : P 75 64 55 = (64 - (75 - 55)) * 100 / 55 := rfl
  simp [P, h_P, h_initial, h_end, h_loaned]
  have h_result : (64 - (75 - 55)) * 100 / 55 = 80 := by norm_num
  exact h_result

end book_return_percentage_l89_89843


namespace sqrt_product_simplified_l89_89867

noncomputable def sqrt_product (x : ℝ) (hx : 0 ≤ x) : ℝ :=
  (real.sqrt (50 * x)) * (real.sqrt (18 * x)) * (real.sqrt (8 * x))

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  sqrt_product x hx = 60 * x * real.sqrt (2 * x) :=
by
  sorry

end sqrt_product_simplified_l89_89867


namespace area_of_PQR_l89_89838

noncomputable def area_of_triangle : Real :=
  let P := (2, 5)
  let Line1 := { slope := 1 / 2, point := P }
  let Line2 := { slope := 3, point := P }
  let Q := (0, 1 / 2 * 0 + 4)
  let R := (0, 3 * 0 - 1)
  let base := 4 - (-1)
  let height := 2 
  (1 / 2) * base * height

theorem area_of_PQR : area_of_triangle = 5 :=
by
  sorry

end area_of_PQR_l89_89838


namespace domain_of_function_l89_89529

theorem domain_of_function:
  ∀ x : ℝ, (sqrt (4 - x^2) / log10 (|x| + x)) ∈ ℝ →
  (4 - x^2 ≥ 0) →
  (|x| + x > 0) →
  (|x| + x ≠ 1) →
  (x > 0 ∧ x < 1/2 ∨ x > 1/2 ∧ x ≤ 2) :=
by
  sorry

end domain_of_function_l89_89529


namespace find_m_range_l89_89992

theorem find_m_range
  (m y1 y2 y0 x0 : ℝ)
  (a c : ℝ) (h1 : a ≠ 0)
  (h2 : x0 = -2)
  (h3 : ∀ x, (x, ax^2 + 4*a*x + c) = (m, y1) ∨ (x, ax^2 + 4*a*x + c) = (m + 2, y2) ∨ (x, ax^2 + 4*a*x + c) = (x0, y0))
  (h4 : y0 ≥ y2) (h5 : y2 > y1) :
  m < -3 :=
sorry

end find_m_range_l89_89992


namespace parabola_tangent_to_hyperbola_l89_89461

theorem parabola_tangent_to_hyperbola (m : ℝ) :
  (∀ x y : ℝ, y = x^2 + 4 → y^2 - m * x^2 = 4) ↔ m = 8 := 
sorry

end parabola_tangent_to_hyperbola_l89_89461


namespace correct_systematic_sample_option_l89_89081

def systematic_sampling_interval (N n : Nat) : Nat :=
  N / n

def valid_systematic_sample (start k : Nat) (samples : List Nat) : Prop :=
  samples = List.range n |>.map (fun i => start + i * k)

theorem correct_systematic_sample_option :
  let N := 60
  let n := 6
  let k := systematic_sampling_interval N n
  k = 10 →
  valid_systematic_sample 3 k [3, 13, 23, 33, 43, 53] :=
by
  intros N n k h
  rw [systematic_sampling_interval, h]
  sorry

end correct_systematic_sample_option_l89_89081


namespace olympiad_scores_l89_89586

theorem olympiad_scores (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j : Fin 20, i < j → a i < a j)
  (h_condition : ∀ i j k : Fin 20, i ≠ j ∧ i ≠ k ∧ j ≠ k → a i < a j + a k) : 
  ∀ i : Fin 20, a i > 18 :=
by
  sorry

end olympiad_scores_l89_89586


namespace infinite_divisible_by_prime_l89_89660

noncomputable def S (p : ℕ) (k : ℕ) : ℕ := ∑ j in Finset.range (p), j^k

theorem infinite_divisible_by_prime (p : ℕ) (hp : p ≥ 5) (hprime : Prime p):
  ∃ᶠ n in atTop, p^3 ∣ S p n ∧ p ∣ S p (n - 1) ∧ p ∣ S p (n - 2) :=
sorry

end infinite_divisible_by_prime_l89_89660


namespace digit_for_divisibility_by_5_l89_89789

theorem digit_for_divisibility_by_5 (B : ℕ) (h : B < 10) :
  (∃ (n : ℕ), n = 527 * 10 + B ∧ n % 5 = 0) ↔ (B = 0 ∨ B = 5) :=
by sorry

end digit_for_divisibility_by_5_l89_89789


namespace select_at_least_one_woman_probability_l89_89546

theorem select_at_least_one_woman_probability (men women total selected : ℕ) (h_men : men = 8) (h_women : women = 4) (h_total : total = men + women) (h_selected : selected = 4) :
  let total_prob := 1
  let prob_all_men := (men.to_rat / total) * ((men - 1).to_rat / (total - 1)) * ((men - 2).to_rat / (total - 2)) * ((men - 3).to_rat / (total - 3))
  let prob_at_least_one_woman := total_prob - prob_all_men
  prob_at_least_one_woman = 85 / 99 := by
  sorry

end select_at_least_one_woman_probability_l89_89546


namespace statement_two_correct_statement_three_correct_l89_89976

open Set

variable (ℝ : Type) [LinearOrder ℝ]

-- Define the variables for lines and planes
variables (l m : ℝ → ℝ → Prop) (α β : ℝ → ℝ → ℝ → Prop)

-- Define perpendicularity and parallelism for lines and planes
def perpendicular (x y : ℝ → ℝ → Prop) := sorry -- definition of perpendicular lines
def parallel (x y : ℝ → ℝ → Prop) := sorry -- definition of parallel lines
def plane_perpendicular (x y : ℝ → ℝ → ℝ → Prop) := sorry -- definition of perpendicular planes
def line_within_plane (x : ℝ → ℝ → Prop) (p : ℝ → ℝ → ℝ → Prop) := sorry -- definition of a line within a plane

-- Statements to be proven
theorem statement_two_correct :
  (perpendicular l α ∧ parallel l m) → perpendicular m α := sorry

theorem statement_three_correct :
  (plane_perpendicular α β ∧ perpendicular l α ∧ ¬line_within_plane l β) → parallel l β := sorry

end statement_two_correct_statement_three_correct_l89_89976


namespace tan_315_eq_neg_one_l89_89221

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89221


namespace trigonometric_expression_simplification_l89_89721

theorem trigonometric_expression_simplification :
  (tan 20 + tan 70 + tan 80) / cos 30 
  = (1 + cos 10 * cos 20) / (cos 20 * cos 70 * cos 30) := 
sorry

end trigonometric_expression_simplification_l89_89721


namespace exists_multiple_representations_l89_89067

def V (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

def indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V n ∧ ¬∃ (p q : ℕ), p ∈ V n ∧ q ∈ V n ∧ p * q = m

theorem exists_multiple_representations (n : ℕ) (h : 2 < n) :
  ∃ r ∈ V n, ∃ s t u v : ℕ, 
    indecomposable n s ∧ indecomposable n t ∧ indecomposable n u ∧ indecomposable n v ∧ 
    r = s * t ∧ r = u * v ∧ (s ≠ u ∨ t ≠ v) :=
sorry

end exists_multiple_representations_l89_89067


namespace tan_315_eq_neg1_l89_89392

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89392


namespace tan_315_eq_neg1_l89_89262

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89262


namespace restaurant_earnings_on_weekday_l89_89201

theorem restaurant_earnings_on_weekday (W : ℕ) 
  (H1 : ∀ day : ℕ, (day ≤ 20 → earns day = W) ∧ (day > 20 → earns day = 2 * W)) 
  (H2 : ∑ i in finset.range 30, earns i = 21600) : 
  W = 600 := 
by sorry

end restaurant_earnings_on_weekday_l89_89201


namespace remainder_div_252_l89_89932

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l89_89932


namespace motorcyclist_wait_time_l89_89810

-- Define the constants and variables
def hiker_speed : ℝ := 6 -- in miles per hour
def motorcyclist_speed : ℝ := 30 -- in miles per hour
def wait_time_minutes : ℝ := 12 -- in minutes

-- Define a function to convert minutes to hours
def minutes_to_hours (m : ℝ) : ℝ := m / 60

-- Define a function to calculate distance traveled
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Calculate the distance the motorcyclist travels in 12 minutes
def distance_ahead : ℝ := distance_traveled motorcyclist_speed (minutes_to_hours wait_time_minutes)

-- Calculate the time it takes for the hiker to catch up in hours
def time_to_catch_up_hours : ℝ := distance_ahead / hiker_speed

-- Convert the catch up time to minutes
def time_to_catch_up_minutes : ℝ := time_to_catch_up_hours * 60

theorem motorcyclist_wait_time : time_to_catch_up_minutes = 60 := by
  sorry

end motorcyclist_wait_time_l89_89810


namespace constant_term_binomial_l89_89563

theorem constant_term_binomial :
  ∀ (x : ℝ), (∑ k in finset.range 7, (nat.choose 6 k) * (2:ℝ)^(6-k) * x^k) = 0 → 64 :=
by
  -- We state the sum of the binomial coefficients and derive the constant term
  sorry

end constant_term_binomial_l89_89563


namespace tan_315_degree_l89_89403

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89403


namespace correct_guesser_is_D_l89_89861

variable (A B C D E F : Prop)

def passerby_A_guessed : Prop := D ∨ E
def passerby_B_guessed : Prop := ¬C
def passerby_C_guessed : Prop := A ∨ B ∨ F
def passerby_D_guessed : Prop := ¬D ∧ ¬E ∧ ¬F

def only_one_guessed_correctly (P : Prop) : Prop := 
  (P ∧ ¬ passerby_A_guessed ∧ ¬ passerby_B_guessed ∧ ¬ passerby_C_guessed) ∨
  (¬ P ∧ passerby_A_guessed ∧ ¬ passerby_B_guessed ∧ ¬ passerby_C_guessed) ∨
  (¬ P ∧ ¬ passerby_A_guessed ∧ passerby_B_guessed ∧ ¬ passerby_C_guessed) ∨
  (¬ P ∧ ¬ passerby_A_guessed ∧ ¬ passerby_B_guessed ∧ passerby_C_guessed)

theorem correct_guesser_is_D : (only_one_guessed_correctly (passerby_D_guessed)) → D :=
by sorry

end correct_guesser_is_D_l89_89861


namespace tan_315_eq_neg1_l89_89390

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89390


namespace integer_solutions_of_log_inequality_l89_89754

def log_inequality_solution_set : Set ℤ := {0, 1, 2}

theorem integer_solutions_of_log_inequality (x : ℤ) (h : 2 < Real.log (x + 5) / Real.log 2 ∧ Real.log (x + 5) / Real.log 2 < 3) :
    x ∈ log_inequality_solution_set :=
sorry

end integer_solutions_of_log_inequality_l89_89754


namespace tan_315_proof_l89_89303

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89303


namespace total_tv_show_cost_correct_l89_89819

noncomputable def total_cost_of_tv_show : ℕ :=
  let cost_per_episode_first_season := 100000
  let episodes_first_season := 12
  let episodes_seasons_2_to_4 := 18
  let cost_per_episode_other_seasons := 2 * cost_per_episode_first_season
  let episodes_last_season := 24
  let number_of_other_seasons := 4
  let total_cost_first_season := episodes_first_season * cost_per_episode_first_season
  let total_cost_other_seasons := (episodes_seasons_2_to_4 * 3 + episodes_last_season) * cost_per_episode_other_seasons
  total_cost_first_season + total_cost_other_seasons

theorem total_tv_show_cost_correct : total_cost_of_tv_show = 16800000 := by
  sorry

end total_tv_show_cost_correct_l89_89819


namespace students_playing_football_l89_89003

theorem students_playing_football (L B total neither : ℕ) (hL : L = 20) (hB : B = 17) (htotal : total = 36) (hneither : neither = 7) :
  ∃ F : ℕ, F = 26 ∧ (total - neither) = (F + L - B) :=
by {
  use 26,
  split,
  { refl },
  { exact calc
      total - neither = 29 : by rw [htotal, hneither]; norm_num
      ... = 26 + L - B : by rw [hL, hB]; norm_num }
}
sorry

end students_playing_football_l89_89003


namespace olympiad_scores_l89_89584

theorem olympiad_scores (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j : Fin 20, i < j → a i < a j)
  (h_condition : ∀ i j k : Fin 20, i ≠ j ∧ i ≠ k ∧ j ≠ k → a i < a j + a k) : 
  ∀ i : Fin 20, a i > 18 :=
by
  sorry

end olympiad_scores_l89_89584


namespace number_of_reachable_cells_after_10_moves_l89_89607

-- Define board size, initial position, and the number of moves
def board_size : ℕ := 21
def initial_position : ℕ × ℕ := (11, 11)
def moves : ℕ := 10

-- Define the main problem statement
theorem number_of_reachable_cells_after_10_moves :
  (reachable_cells board_size initial_position moves).card = 121 :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89607


namespace bus_stops_for_15_minutes_l89_89807

-- Define the speeds with and without stoppages
def speed_without_stoppages : ℝ := 80
def speed_with_stoppages : ℝ := 60

-- Define the time the bus stops per hour
def stoppage_time_per_hour (speed_without_stoppages speed_with_stoppages : ℝ) : ℝ :=
  ((speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages) * 60

-- Prove that the stoppage time per hour is 15 minutes
theorem bus_stops_for_15_minutes :
  stoppage_time_per_hour speed_without_stoppages speed_with_stoppages = 15 :=
by
  -- Placeholder for the proof
  sorry

end bus_stops_for_15_minutes_l89_89807


namespace reachable_cells_after_moves_l89_89632

def is_valid_move (n : ℕ) (x y : ℤ) : Prop :=
(abs x ≤ n ∧ abs y ≤ n ∧ (x + y) % 2 = 0)

theorem reachable_cells_after_moves (n : ℕ) :
  n = 10 → ∃ (cells : Finset (ℤ × ℤ)), cells.card = 121 ∧ 
  (∀ (cell : ℤ × ℤ), cell ∈ cells → is_valid_move n cell.1 cell.2) :=
by
  intros h
  use {-10 ≤ x, y, x + y % 2 = 0 & abs x + abs y ≤ n }
  sorry -- proof goes here

end reachable_cells_after_moves_l89_89632


namespace five_digit_palindrome_count_l89_89457

-- Definitions based on conditions
def digits := Fin 10 -- Digits from 0 to 9
def nonzero_digits := Fin 9 -- Digits from 1 to 9

def is_palindrome (n : ℕ) : Prop :=
  let str := n.toString
  str.length = 5 ∧ str = str.reverse

-- Lean statement for the problem
theorem five_digit_palindrome_count : 
  ∃ (count : ℕ), 
    (count = 9 * 10 * 10) ∧ 
    (∀ n, is_palindrome n → (n.toString.length = 5 → ∃ (a b c : ℕ), 
      a ∈ {1,2,3,4,5,6,7,8,9} ∧ 
      b ∈ {0,1,2,3,4,5,6,7,8,9} ∧ 
      c ∈ {0,1,2,3,4,5,6,7,8,9} ∧ 
      n = a * 10000 + b * 1000 + c * 100 + b * 10 + a)) :=
by 
  existsi 900
  split
  · apply rfl -- This confirms count = 900
  · intros n palindrome hyp5
    sorry

end five_digit_palindrome_count_l89_89457


namespace _l89_89919

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l89_89919


namespace remainder_123456789012_mod_252_l89_89903

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l89_89903


namespace exists_order_equal_d_l89_89168

theorem exists_order_equal_d (n d : ℕ) (h1 : ∀ a, gcd a n = 1 → a ^ d ≡ 1 [MOD n]) 
    (h2 : ∀ e : ℕ, (∀ a, gcd a n = 1 → a ^ e ≡ 1 [MOD n]) → e ≥ d) :
  ∃ b, gcd b n = 1 ∧ ord n b = d :=
begin
  sorry
end

end exists_order_equal_d_l89_89168


namespace tan_315_eq_neg1_l89_89374

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89374


namespace tan_315_degree_l89_89412

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89412


namespace square_area_l89_89077

theorem square_area (x : ℝ) (G H : ℝ) (hyp_1 : 0 ≤ G) (hyp_2 : G ≤ x) (hyp_3 : 0 ≤ H) (hyp_4 : H ≤ x) (AG : ℝ) (GH : ℝ) (HD : ℝ)
  (hyp_5 : AG = 20) (hyp_6 : GH = 20) (hyp_7 : HD = 20) (hyp_8 : x = 20 * Real.sqrt 2) :
  x^2 = 800 :=
by
  sorry

end square_area_l89_89077


namespace tan_315_eq_neg_one_l89_89297

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89297


namespace total_trees_cut_l89_89654

/-- James cuts 20 trees each day for the first 2 days. Then, for the next 3 days, he and his 2 brothers (each cutting 20% fewer trees per day than James) cut trees together. Prove that they cut 196 trees in total. -/
theorem total_trees_cut :
  let trees_first_2_days := 2 * 20; let trees_per_day_james := 20; let rate_fewer := 0.2;
  let trees_per_day_brother := trees_per_day_james * (1 - rate_fewer);
  let days_with_help := 3;
  let trees_per_day_all := trees_per_day_james + 2 * trees_per_day_brother;
  let total_trees_with_help := days_with_help * trees_per_day_all;
  total_trees_first_2_days + total_trees_with_help = 196 :=
by {
  let trees_first_2_days := 2 * 20;
  let trees_per_day_james := 20;
  let rate_fewer := 0.2;
  let trees_per_day_brother := trees_per_day_james * (1 - rate_fewer);
  let days_with_help := 3;
  let trees_per_day_all := trees_per_day_james + 2 * trees_per_day_brother;
  let total_trees_with_help := days_with_help * trees_per_day_all;

  have h1 : trees_first_2_days = 40 := by norm_num,
  have h2 : trees_per_day_brother = 16 := by norm_num,
  have h3 : trees_per_day_all = 52 := by norm_num,
  have h4 : total_trees_with_help = 156 := by norm_num,
  exact h1 + h4
}

end total_trees_cut_l89_89654


namespace _l89_89924

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l89_89924


namespace complement_of_A_with_respect_to_U_l89_89533

namespace SetTheory

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def C_UA : Set ℕ := {3, 4, 5}

theorem complement_of_A_with_respect_to_U :
  (U \ A) = C_UA := by
  sorry

end SetTheory

end complement_of_A_with_respect_to_U_l89_89533


namespace kim_saplings_left_l89_89044

def number_of_pits : ℕ := 80
def proportion_sprout : ℚ := 0.25
def saplings_sold : ℕ := 6

theorem kim_saplings_left : 
  (number_of_pits * proportion_sprout - saplings_sold = 14) :=
begin
  sorry
end

end kim_saplings_left_l89_89044


namespace infinite_common_terms_l89_89141

noncomputable def a_seq : ℕ → ℤ
| 0       := 2
| 1       := 14
| (n + 2) := 14 * a_seq n.succ + a_seq n

noncomputable def b_seq : ℕ → ℤ
| 0       := 2
| 1       := 14
| (n + 2) := 6 * b_seq n.succ - b_seq n

theorem infinite_common_terms : 
  ∃ (S : Set ℤ), S.Infinite ∧ 
    (∀ n, a_seq n ∈ S ) ∧ 
    (∀ n, b_seq n ∈ S) :=
sorry

end infinite_common_terms_l89_89141


namespace find_b_l89_89741

-- Define the problem based on the conditions identified
theorem find_b (b : ℕ) (h₁ : b > 0) (h₂ : (b : ℝ)/(b+15) = 0.75) : b = 45 := 
  sorry

end find_b_l89_89741


namespace trajectory_not_on_straight_line_l89_89801

open Real

noncomputable def distance_point_to_line (x y A B C : ℝ) : ℝ :=
  abs (A * x + B * y + C) / sqrt (A^2 + B^2)

theorem trajectory_not_on_straight_line : 
  ∃ M : ℝ × ℝ, 
    let x := M.1 in 
    let y := M.2 in 
    distance_point_to_line x y 4 3 (-5) + distance_point_to_line x y 4 3 10 = 3 ∧ 
    ¬(
      (∃ t : ℝ, 
        let x := t in 
        let y := 0 in 
        distance_point_to_line x y 1 0 (-1) = 2
      ) ∨
      (∃ t : ℝ, 
        let x := 0 in 
        let y := t in 
        distance_point_to_line x y 0 1 (-2) - distance_point_to_line x y 0 1 2 = 4
      ) ∨
      (∃ t : ℝ, 
        let x := 2 in 
        let y := 3 in 
        distance_point_to_line x y 2 (-1) (-1) = sqrt( (x - 2)^2 + (y - 3)^2 )
      )
    )
:=
  sorry

end trajectory_not_on_straight_line_l89_89801


namespace alternating_sum_101_l89_89873

def alternating_sum : ℕ → ℤ
| 0       := 0
| (n + 1) := if (n % 2 = 0) then alternating_sum n + (n + 1) else alternating_sum n - (n + 1)

theorem alternating_sum_101 : alternating_sum 101 = 51 := by
  sorry

end alternating_sum_101_l89_89873


namespace num_cells_after_10_moves_l89_89616

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 + 1 = p2.1))

def num_reachable_cells (n m k : ℕ) (start : ℕ × ℕ) : ℕ :=
  sorry -- Calculation of the number of reachable cells after k moves

theorem num_cells_after_10_moves :
  let board_size := 21
  let start := (11, 11)
  let moves := 10
  num_reachable_cells board_size board_size moves start = 121 :=
sorry

end num_cells_after_10_moves_l89_89616


namespace count_no_carrying_pairs_in_range_l89_89467

def is_consecutive (a b : ℕ) : Prop :=
  b = a + 1

def no_carrying (a b : ℕ) : Prop :=
  ∀ i, ((a / 10^i) % 10 + (b / 10^i) % 10) < 10

def count_no_carrying_pairs (start end_ : ℕ) : ℕ :=
  let pairs := (start to end_).to_list
  (pairs.zip pairs.tail).count (λ (a, b) => is_consecutive a b ∧ no_carrying a b)

theorem count_no_carrying_pairs_in_range :
  count_no_carrying_pairs 2000 3000 = 7290 :=
sorry

end count_no_carrying_pairs_in_range_l89_89467


namespace tan_315_proof_l89_89310

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89310


namespace reachable_cells_after_10_moves_l89_89629

def adjacent_cells (x y : ℕ) : set (ℕ × ℕ) :=
  { (x', y') | (x' = x + 1 ∧ y' = y) ∨ (x' = x - 1 ∧ y' = y) 
            ∨ (x' = x ∧ y' = y + 1) ∨ (x' = x ∧ y' = y - 1) }

def in_bounds (x y : ℕ) : Prop :=
  x > 0 ∧ x ≤ 21 ∧ y > 0 ∧ y ≤ 21

theorem reachable_cells_after_10_moves : 
  ∃ cells : set (ℕ × ℕ), ∃ initial_position : (11, 11) ∈ cells ∧ 
  (∀ (x y : ℕ), (x, y) ∈ cells → in_bounds x y ∧
  (∀ n ≤ 10, (x', y') ∈ adjacent_cells x y → (x', y') ∈ cells)) ∧ 
  (set.card cells = 121) :=
sorry

end reachable_cells_after_10_moves_l89_89629


namespace probability_G_is_one_fourth_l89_89184

-- Definitions and conditions
variables (p_E p_F p_G p_H : ℚ)
axiom probability_E : p_E = 1/3
axiom probability_F : p_F = 1/6
axiom prob_G_eq_H : p_G = p_H
axiom total_prob_sum : p_E + p_F + p_G + p_G = 1

-- Theorem statement
theorem probability_G_is_one_fourth : p_G = 1/4 :=
by 
  -- Lean proof omitted, only the statement required
  sorry

end probability_G_is_one_fourth_l89_89184


namespace g_solution_l89_89671

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 2
axiom g_functional : ∀ x y : ℝ, g (x * y) = g ((x^2 + y^2) / 2) + (x - y)^2 + x^2

theorem g_solution :
  ∀ x : ℝ, g x = 2 - 2 * x := sorry

end g_solution_l89_89671


namespace tan_315_eq_neg1_l89_89340

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89340


namespace value_of_a4_l89_89514

theorem value_of_a4 (a : ℕ → ℕ) (r : ℕ) (h1 : ∀ n, a (n+1) = r * a n) (h2 : a 4 / a 2 - a 3 = 0) (h3 : r = 2) :
  a 4 = 8 :=
sorry

end value_of_a4_l89_89514


namespace remainder_123456789012_div_252_l89_89910

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l89_89910


namespace find_a_l89_89539

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (x + a)^9 = ∑ i in finset.range 10, (a_i : ℝ) * (x+1)^i) ∧
  (a_5 : ℝ = 126) →
  (a = 0 ∨ a = 2) := 
by
  sorry

end find_a_l89_89539


namespace tangent_line_at_point_l89_89738

open Real

def curve (x : ℝ) : ℝ := (1 / x) - sqrt x

def deriv_curve (x : ℝ) : ℝ := -(1 / x^2) - (1 / (2 * sqrt x))

theorem tangent_line_at_point :
  ∀ (x y : ℝ), x = 4 → y = -7 / 4 → (deriv_curve 4 = -5 / 16) → 5 * x + 16 * y + 8 = 0 :=
by
  intros x y hx hy h_deriv
  rw [hx, hy] at *
  -- Skip the detailed proof steps
  sorry

end tangent_line_at_point_l89_89738


namespace side_length_square_l89_89106

-- Define the side length of the square
variables (s : ℝ)

-- Given condition: the diagonal of the square is 2√2 inches long
def diagonal_eq (s : ℝ) : Prop := (2 * real.sqrt 2)

-- Prove that the side length of the square is 2 inches
theorem side_length_square (h : s * real.sqrt 2 = 2 * real.sqrt 2) : s = 2 :=
by sorry

end side_length_square_l89_89106


namespace find_DE_l89_89813

-- Define points A, B, C, D, E on the circle and given angles
variables (A B C D E : Type) [is_circle C 20]
variables (center : center_of_circle C)
variables (radius : radius_of_circle C 20)
variables (on_circle_A : on_circle A C)
variables (on_circle_B : on_circle B C)
variables (on_circle_D : on_circle D C)
variables (angle_ACB : angle A C B = 60)
variables (angle_ACD : angle A C D = 160)
variables (angle_DCB : angle D C B = 100)
variables (is_intersection_E : is_intersection E (line_through A C) (line_through B D))

-- The goal is to prove that DE = 20
theorem find_DE : distance D E = 20 := sorry

end find_DE_l89_89813


namespace reflection_matrix_over_y_equals_x_l89_89448

theorem reflection_matrix_over_y_equals_x :
  let reflection_matrix := λ (v : ℝ × ℝ), (v.2, v.1) in
  (∀ v, reflection_matrix v = (v.2, v.1) →
      ∃ M : Matrix (Fin 2) (Fin 2) ℝ,
      ∀ v : Matrix (Fin 2) (Fin 1) ℝ,
      M ⬝ v = reflection_matrix (v 0 0, v 1 0) ⬝) :=
begin
  sorry
end

end reflection_matrix_over_y_equals_x_l89_89448


namespace find_a_b_c_sum_l89_89442

theorem find_a_b_c_sum (a b c : ℝ) 
  (h_vertex : ∀ x, y = a * x^2 + b * x + c ↔ y = a * (x - 3)^2 + 5)
  (h_passes : a * 1^2 + b * 1 + c = 2) :
  a + b + c = 35 / 4 :=
sorry

end find_a_b_c_sum_l89_89442


namespace num_cells_after_10_moves_l89_89618

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 + 1 = p2.1))

def num_reachable_cells (n m k : ℕ) (start : ℕ × ℕ) : ℕ :=
  sorry -- Calculation of the number of reachable cells after k moves

theorem num_cells_after_10_moves :
  let board_size := 21
  let start := (11, 11)
  let moves := 10
  num_reachable_cells board_size board_size moves start = 121 :=
sorry

end num_cells_after_10_moves_l89_89618


namespace tan_315_degrees_l89_89361

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89361


namespace tangent_315_deg_l89_89276

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89276


namespace proof_problem_l89_89466

def absolute_operation (s : List ℝ) : ℝ :=
  s.pairwise (|· - ·|).sum

def correct_statements_count : ℝ :=
  if 
    absolute_operation [1, 3, 5, 10] = 29 &&
    ¬ (∃ x : ℝ, absolute_operation [x, -2, 5] = 14) &&
    (count_simplified_expressions [a, b, b, c] = 6)
  then 1 else if 
    absolute_operation [1, 3, 5, 10] = 29 ||
    (∃ x : ℝ, absolute_operation [x, -2, 5] = 14) ||
    count_simplified_expressions [a, b, b, c] = 6
  then 0 else 0


theorem proof_problem :
  correct_statements_count = 1 :=
begin
  sorry
end

end proof_problem_l89_89466


namespace math_olympiad_scores_l89_89598

theorem math_olympiad_scores (a : Fin 20 → ℕ) 
  (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → j ≠ k → i ≠ k → a i < a j + a k) :
  ∀ i : Fin 20, a i > 18 := 
sorry

end math_olympiad_scores_l89_89598


namespace concyclic_points_l89_89983

open EuclideanGeometry

variable {A B C X Y A1 B1 C1 : Point} [InTriangle : Triangle A B C]
variable [OnSidesX : OnLine X A B] [OnSidesY : OnLine Y A C]
variable [Altitudes : Perpendicular A1 A (B, C)] [Perpendicular B1 B (A, C)] [Perpendicular C1 C (A, B)]
variable [SymX : Reflection X B (A, C)] [SymY : Reflection Y C (A, B)]

theorem concyclic_points : CyclicQuadrilateral A B1 C1 A1 := by 
  sorry

end concyclic_points_l89_89983


namespace tan_315_eq_neg1_l89_89259

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89259


namespace tan_315_eq_neg1_l89_89335

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89335


namespace remainder_div_252_l89_89931

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l89_89931


namespace man_walking_time_l89_89436

-- Definitions and conditions
variables (T X W D : ℕ)
variable (usual_drive_time : X)
variable (arrives_early : T - 60)
variable (arrives_earlier_home : T + X - 40)
variable (walk_drive_sum : W + D = X)
variable (drive_time_diff : D = X - 40)

-- Proof statement
theorem man_walking_time : W = 40 := by
  -- The theorem we need to prove, assuming the conditions are met
  sorry

end man_walking_time_l89_89436


namespace candle_flower_groupings_l89_89162

theorem candle_flower_groupings : 
  (nat.choose 4 2) * (nat.choose 9 8) = 54 :=
by sorry

end candle_flower_groupings_l89_89162


namespace num_cells_after_10_moves_l89_89617

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 + 1 = p2.1))

def num_reachable_cells (n m k : ℕ) (start : ℕ × ℕ) : ℕ :=
  sorry -- Calculation of the number of reachable cells after k moves

theorem num_cells_after_10_moves :
  let board_size := 21
  let start := (11, 11)
  let moves := 10
  num_reachable_cells board_size board_size moves start = 121 :=
sorry

end num_cells_after_10_moves_l89_89617


namespace distance_from_center_to_surface_l89_89704

theorem distance_from_center_to_surface (R : ℝ) (h : ℝ) 
    (H1 : ∀ (r : ℝ), r = 2 * R) 
    (H2 : ∀ (P Q : ℝ), P = 4 * R) 
    (H3 : ∀ (x y z : ℝ), PythagoreanTheorem  (3 * R) y (2 * R)) : 
    h = Real.sqrt 5 * R :=
sorry

end distance_from_center_to_surface_l89_89704


namespace tan_315_degrees_l89_89367

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89367


namespace gini_coefficient_separate_gini_coefficient_combined_l89_89576

-- Definitions based on provided conditions
def northern_residents : ℕ := 24
def southern_residents : ℕ := 6
def price_per_set : ℝ := 2000
def northern_PPC (x : ℝ) : ℝ := 13.5 - 9 * x
def southern_PPC (x : ℝ) : ℝ := 1.5 * x^2 - 24

-- Gini Coefficient when both regions operate separately
theorem gini_coefficient_separate : 
  ∃ G : ℝ, G = 0.2 :=
  sorry

-- Gini Coefficient change when blending productions as per Northern conditions
theorem gini_coefficient_combined :
  ∃ ΔG : ℝ, ΔG = 0.001 :=
  sorry

end gini_coefficient_separate_gini_coefficient_combined_l89_89576


namespace tan_315_eq_neg1_l89_89269

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89269


namespace tan_315_eq_neg_1_l89_89319

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89319


namespace tan_315_eq_neg_1_l89_89322

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89322


namespace remainder_123456789012_div_252_l89_89944

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l89_89944


namespace min_value_inequality_l89_89676

theorem min_value_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  ∃ n : ℝ, n = 9 / 4 ∧ (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 2 → (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ n) :=
sorry

end min_value_inequality_l89_89676


namespace probability_at_least_one_woman_in_selection_l89_89558

theorem probability_at_least_one_woman_in_selection :
  ∃ (P : ℚ), P = 85 / 99 :=
by 
  -- Define variables
  let total_people := 12
  let men := 8
  let women := 4
  let selection := 4

  -- Calculate the probability of selecting four men
  let P_all_men := (men / total_people) * ((men - 1) / (total_people - 1)) *
                   ((men - 2) / (total_people - 2)) *
                   ((men - 3) / (total_people - 3))

  -- Calculate the probability of at least one woman being selected
  let P_at_least_one_woman := 1 - P_all_men

  -- Verify the result
  have H : P_at_least_one_woman = 85 / 99 := sorry
  use P_at_least_one_woman
  exact H

end probability_at_least_one_woman_in_selection_l89_89558


namespace tangent_to_circle_ACE_l89_89421

theorem tangent_to_circle_ACE 
  (circle1 circle2 : Circle) (A C P D E : Point)
  (H1 : tangent_at circle1 circle2 A)
  (H2 : on_circle C circle1)
  (H3 : line_through A C) 
  (H4 : line_intersects_circle (line_through A C) circle2 P)
  (H5 : tangent_at C circle1)
  (H6 : tangent_intersects_circle_at C circle1 circle2 D)
  (H7 : tangent_intersects_circle_at C circle1 circle2 E)
  : tangent_to (line_through P E) (circumscribed_circle A C E) := 
by
  sorry

end tangent_to_circle_ACE_l89_89421


namespace at_least_one_woman_selected_probability_l89_89553

-- Define the total number of people, men, and women
def total_people : Nat := 12
def men : Nat := 8
def women : Nat := 4
def selected_people : Nat := 4

-- Define the probability ratio of at least one woman being selected
def probability_at_least_one_woman_selected : ℚ := 85 / 99

-- Prove the probability is correct given the conditions
theorem at_least_one_woman_selected_probability :
  (probability_of_selecting_at_least_one_woman men women selected_people total_people) = probability_at_least_one_woman_selected :=
sorry

end at_least_one_woman_selected_probability_l89_89553


namespace tangent_315_deg_l89_89275

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89275


namespace tan_315_eq_neg1_l89_89254

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89254


namespace largest_coefficient_l89_89110

theorem largest_coefficient (t : ℕ) (h_t : t = 6) :
  let a := t^3
  let b := 9 * t^2
  let c := 27 * t
  let d := 27
  max (max a b) (max c d) = 324 := 
by
  let a := t^3
  let b := 9 * t^2
  let c := 27 * t
  let d := 27
  rw [h_t]
  have : a = 216 := by norm_num
  have : b = 324 := by norm_num
  have : c = 162 := by norm_num
  have : d = 27 := by norm_num
  have h1 : max a b = b := by simp [this]
  have h2 : max c d = c := by simp [this]
  have h3 : max b c = b := by simp [this]
  simp [h1, h2, h3]

-- sorry for skipping the proof

end largest_coefficient_l89_89110


namespace circumcircle_tangent_inscribed_circle_l89_89984

-- definitions of the angle, inscribed circle, and tangent line intersections
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (angle : A) (circle : MetricSpace.Circle) (tangent : Line)
variables (intersection1 : Tangent.Intersects tangent (angle.side1 circle))
variables (intersection2 : Tangent.Intersects tangent (angle.side2 circle))

-- theorem to prove circumcircle of triangle ABC is tangent to the inscribed circle
theorem circumcircle_tangent_inscribed_circle
  (h1 : angle.contains_circle circle)
  (h2 : tangent.is_tangent_to circle)
  (h3 : Tangent.intersects_sides tangent (angle.side1 circle) (angle.side2 circle) B C) :
  let circumcircle := Circumcircle A B C in
  circumcircle.is_tangent_to circle :=
sorry

end circumcircle_tangent_inscribed_circle_l89_89984


namespace determine_k_from_line_l89_89883

-- Define the necessary variables and hypotheses
variable (k : ℝ)
variable (x y : ℝ)
variable (L : ℝ → ℝ → Prop)

-- Define the equation of the line
def line (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y, 3 * k * x - 2 = 4 * y

-- Define the specific point
def point : ℝ × ℝ := (-1 / 2, -5)

-- State the main theorem
theorem determine_k_from_line (h : line k (point.1) (point.2)) : k = 12 := by
  dsimp [line, point] at h
  sorry

end determine_k_from_line_l89_89883


namespace sin_cos_diff_value_l89_89516

noncomputable def sin_cos_diff (α : Real) : Real :=
(sin α) - (cos α)

theorem sin_cos_diff_value :
  ∃ (α : Real), (∃ (x y r : Real), x = -2 ∧ y = 4 ∧ r = Real.sqrt (x^2 + y^2) ∧ sin α = y / r ∧ cos α = x / r) ∧ sin_cos_diff α = (3 * Real.sqrt 5) / 5 :=
begin
  sorry
end

end sin_cos_diff_value_l89_89516


namespace probability_of_exactly_5_games_probability_distribution_and_expectation_of_X_l89_89829

-- Define the initial conditions
def best_of_seven_match (pA : ℚ) (pB : ℚ) : Prop :=
  (pA = 1 / 3) ∧ (pB = 2 / 3)

-- Part 1: Prove the probability of exactly 5 games being played when the match ends
theorem probability_of_exactly_5_games {pA pB : ℚ} (h : best_of_seven_match pA pB) :
  pA = 1 / 3 →
  pB = 2 / 3 →
  let P1 := 4.choose 3 * (pA ^ 3) * (pB) * (pA)
  let P2 := 4.choose 3 * (pB ^ 3) * (pA) * (pB)
  let P := P1 + P2
  P = 8 / 27 := sorry

-- Part 2: Prove the probability distribution and expectation of X
theorem probability_distribution_and_expectation_of_X {pA pB : ℚ} (h : best_of_seven_match pA pB) :
  pA = 1 / 3 →
  pB = 2 / 3 →
  let P1 := pA
  let P2 := (pB * pA)
  let P3 := pB ^ 2
  let E := P1 * 1 + P2 * 2 + P3 * 3
  (P1 = 1 / 3) ∧ (P2 = 2 / 9) ∧ (P3 = 4 / 9) ∧ (E = 19 / 9) := sorry

end probability_of_exactly_5_games_probability_distribution_and_expectation_of_X_l89_89829


namespace _l89_89922

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l89_89922


namespace area_ratio_of_squares_l89_89144

def side_length_ratio (a b : ℝ) : Prop :=
  b = 5 * a

def diagonal_length (a : ℝ) : ℝ :=
  a * Real.sqrt 2

def area (a : ℝ) : ℝ :=
  a * a

theorem area_ratio_of_squares 
  (a b : ℝ)
  (h1 : side_length_ratio a b)
  (d1 : diagonal_length a)
  (d2 : diagonal_length b) 
  : b*b = 25*a*a :=
by 
  unfold side_length_ratio at h1
  rw [h1]
  unfold area
  sorry

end area_ratio_of_squares_l89_89144


namespace circle_has_most_axes_of_symmetry_l89_89199

-- Define the number of axes of symmetry for each shape
def axes_of_symmetry (shape : Type) : ℕ ∞ → Prop :=
  | isosceles_triangle => 1
  | square => 4
  | circle => ∞
  | line_segment => 2

-- Define the shapes as types
def isosceles_triangle : Type := sorry
def square : Type := sorry
def circle : Type := sorry
def line_segment : Type := sorry

-- Prove that the circle has the most axes of symmetry 
theorem circle_has_most_axes_of_symmetry
  (h1 : axes_of_symmetry isosceles_triangle 1)
  (h2 : axes_of_symmetry square 4)
  (h3 : axes_of_symmetry circle ∞)
  (h4 : axes_of_symmetry line_segment 2) :
  (∀ s : Type, axes_of_symmetry s ∞ → s = circle) :=
sorry

end circle_has_most_axes_of_symmetry_l89_89199


namespace equilateral_triangle_ratios_l89_89793

open Real

variables (s : ℝ) (A : ℝ) (P : ℝ) (h : ℝ)

noncomputable def side_length := (6 : ℝ)
noncomputable def area := (side_length^2 * sqrt 3 / 4)
noncomputable def perimeter := (3 * side_length)
noncomputable def height := (sqrt 3 / 2 * side_length)

theorem equilateral_triangle_ratios :
  let A := side_length^2 * sqrt 3 / 4,
      P := 3 * side_length,
      h := sqrt 3 / 2 * side_length in
  A / P = sqrt 3 / 2 ∧ h / side_length = sqrt 3 / 2 :=
by
  let A := side_length^2 * sqrt 3 / 4
  let P := 3 * side_length
  let h := sqrt 3 / 2 * side_length

  -- prove the theorem
  sorry

end equilateral_triangle_ratios_l89_89793


namespace tan_315_degrees_l89_89366

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89366


namespace bottle_cap_count_l89_89885

theorem bottle_cap_count (price_per_cap total_cost : ℕ) (h_price : price_per_cap = 2) (h_total : total_cost = 12) : total_cost / price_per_cap = 6 :=
by
  sorry

end bottle_cap_count_l89_89885


namespace max_possible_value_l89_89490

noncomputable def max_ratio (S : set ℕ) (U V : set (set ℕ)) [finite S] : ℚ :=
  let U_size : ℕ := U.to_finset.card
  let V_size : ℕ := V.to_finset.card
  let UV_inter : ℕ := (U ∩ V).to_finset.card
  UV_inter / (U_size * V_size : ℚ)

theorem max_possible_value (S : set ℕ) (U V : set (set ℕ))
  [finite S] (hU : ∀ {A B : set ℕ}, A ∈ U → A ⊆ B → B ∈ U)
  (hV : ∀ {A B : set ℕ}, A ∈ V → B ⊆ A → B ∈ V) :
  max_ratio S U V ≤ 1 / (2 ^ (finite.to_finset S).card : ℚ) :=
sorry

end max_possible_value_l89_89490


namespace vasya_can_form_two_triangles_l89_89140

-- Given vertices and edges of a tetrahedron
variables {D A B C : Type}
variable (edges : set (set (Type)))

-- Condition reflecting the structure of a tetrahedron
def is_tetrahedron (D A B C : Type) (edges : set (set (Type))) : Prop :=
  edges = {{D, A}, {D, B}, {D, C}, {A, B}, {A, C}, {B, C}}

-- The main statement reflecting the fact that Vasya can form two triangles
theorem vasya_can_form_two_triangles(D A B C : Type) (h_tetra : is_tetrahedron D A B C edges) :
  ∃ T1 T2 : set (set (Type)), (∀ e ∈ edges, e ∈ T1 ∨ e ∈ T2) ∧ (T1 ∩ T2 = ∅) ∧ 
  (is_triangle T1) ∧ (is_triangle T2) :=
sorry

-- Definition of a triangle used in the theorem
def is_triangle (T : set (set (Type))) : Prop :=
  ∃ x y z : Type, T = {{x, y}, {y, z}, {z, x}}

end vasya_can_form_two_triangles_l89_89140


namespace reachable_cells_after_10_moves_l89_89628

def adjacent_cells (x y : ℕ) : set (ℕ × ℕ) :=
  { (x', y') | (x' = x + 1 ∧ y' = y) ∨ (x' = x - 1 ∧ y' = y) 
            ∨ (x' = x ∧ y' = y + 1) ∨ (x' = x ∧ y' = y - 1) }

def in_bounds (x y : ℕ) : Prop :=
  x > 0 ∧ x ≤ 21 ∧ y > 0 ∧ y ≤ 21

theorem reachable_cells_after_10_moves : 
  ∃ cells : set (ℕ × ℕ), ∃ initial_position : (11, 11) ∈ cells ∧ 
  (∀ (x y : ℕ), (x, y) ∈ cells → in_bounds x y ∧
  (∀ n ≤ 10, (x', y') ∈ adjacent_cells x y → (x', y') ∈ cells)) ∧ 
  (set.card cells = 121) :=
sorry

end reachable_cells_after_10_moves_l89_89628


namespace correct_statements_l89_89024

noncomputable def f (x y : ℝ) : ℝ := (x - y) / Real.sqrt (x^2 + y^2)

theorem correct_statements 
(x y : ℝ) 
(r : ℝ) 
(h : r = Real.sqrt (x^2 + y^2))
(hr_pos : r > 0)
: ∀ (θ : ℝ), (f(x, y) ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)) ∧
              (f(x, y) = f((3 * Real.pi / 4) - θ)) ∧
              (∀ θ, f(θ) = f(θ + 2 * Real.pi)) := sorry

end correct_statements_l89_89024


namespace remainder_123456789012_mod_252_l89_89906

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l89_89906


namespace instantaneous_velocity_at_2_l89_89746

def displacement (t : ℝ) : ℝ := 2 * t^2 + 3

theorem instantaneous_velocity_at_2 : (deriv displacement 2) = 8 :=
by 
  -- Proof would go here
  sorry

end instantaneous_velocity_at_2_l89_89746


namespace crates_needed_l89_89889

def ceil_div (a b : ℕ) : ℕ := (a + b - 1) / b

theorem crates_needed :
  ceil_div 145 12 + ceil_div 271 8 + ceil_div 419 10 + ceil_div 209 14 = 104 :=
by
  sorry

end crates_needed_l89_89889


namespace tan_315_degree_l89_89401

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89401


namespace tan_315_proof_l89_89307

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89307


namespace smallest_value_of_Q_l89_89104

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 7*x^2 - 2*x + 10

theorem smallest_value_of_Q :
  min (Q 1) (min (10 : ℝ) (min (4 : ℝ) (min (1 - 4 + 7 - 2 + 10 : ℝ) (2.5 : ℝ)))) = 2.5 :=
by sorry

end smallest_value_of_Q_l89_89104


namespace find_emeralds_l89_89178

-- Definitions for the problem.
variable (D E R : ℕ)
variable (box1 box2 box3 box4 box5 box6 : ℕ)
variable (box_contents : Set ℕ := {box1, box2, box3, box4, box5, box6})
variable total_stones : Set ℕ := {13, 8, 7, 5, 4, 2}

-- Problem condition: Total number of stones in boxes.
axiom box_contents_correct : box_contents = total_stones

-- Problem condition: R is 15 more than D.
axiom ruby_diamond_relation : R = D + 15

-- Main theorem: Finding the number of emeralds.
theorem find_emeralds : E = 12 :=
by
  -- Steps will make use of conditions:
  -- R = D + 15, box_distribution, and total counts in the boxes.
  sorry

end find_emeralds_l89_89178


namespace total_fish_purchased_l89_89716

/-- Definition of the conditions based on Roden's visits to the pet shop. -/
def first_visit_goldfish := 15
def first_visit_bluefish := 7
def second_visit_goldfish := 10
def second_visit_bluefish := 12
def second_visit_greenfish := 5
def third_visit_goldfish := 3
def third_visit_bluefish := 7
def third_visit_greenfish := 9

/-- Proof statement in Lean 4. -/
theorem total_fish_purchased :
  first_visit_goldfish + first_visit_bluefish +
  second_visit_goldfish + second_visit_bluefish + second_visit_greenfish +
  third_visit_goldfish + third_visit_bluefish + third_visit_greenfish = 68 :=
by
  sorry

end total_fish_purchased_l89_89716


namespace possible_review_organization_l89_89773

noncomputable def organizing_review : Prop :=
  ∃ (students problems : Finset ℕ) (solved : ℕ → Finset ℕ), 
    students.card = 20 ∧ problems.card = 20 ∧
    (∀ s ∈ students, (solved s).card = 2) ∧ 
    (∀ p ∈ problems, ∃! s ∈ students, p ∈ solved s) ∧ 
    (∀ p ∈ problems, ∃ s1 s2 ∈ students, s1 ≠ s2 ∧ p ∈ solved s1 ∧ p ∈ solved s2) ∧
    ∀ s ∈ students, ∃ p ∈ solved s, p ∈ problems

theorem possible_review_organization : organizing_review :=
  sorry

end possible_review_organization_l89_89773


namespace gcd_13924_32451_eq_one_l89_89445

-- Define the two given integers.
def x : ℕ := 13924
def y : ℕ := 32451

-- State and prove that the greatest common divisor of x and y is 1.
theorem gcd_13924_32451_eq_one : Nat.gcd x y = 1 := by
  sorry

end gcd_13924_32451_eq_one_l89_89445


namespace tangent_315_deg_l89_89274

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89274


namespace sum_S_range_l89_89064

noncomputable def S : ℝ :=
  ∑ x in { x : ℝ | 0 < x ∧ x ^ 2 ^ sqrt 3 = (sqrt 3) ^ 2 ^ x }, x 

theorem sum_S_range : 2 ≤ S ∧ S < 6 :=
by
  sorry

end sum_S_range_l89_89064


namespace solution_set_of_inequality_l89_89760

theorem solution_set_of_inequality (x : ℝ) :
  (x + 2) ^ (-(3 / 5)) < (5 - 2 * x) ^ (-(3 / 5)) ↔ x ∈ (Set.Ioo 1 (5 / 2) ∪ Set.Iio (-2)) := by
  sorry

end solution_set_of_inequality_l89_89760


namespace value_divided_by_3_l89_89730

-- Given condition
def given_condition (x : ℕ) : Prop := x - 39 = 54

-- Correct answer we need to prove
theorem value_divided_by_3 (x : ℕ) (h : given_condition x) : x / 3 = 31 := 
by
  sorry

end value_divided_by_3_l89_89730


namespace rectangle_area_change_l89_89116

theorem rectangle_area_change
  (L B : ℝ)
  (hL : L > 0)
  (hB : B > 0)
  (new_L : ℝ := 1.25 * L)
  (new_B : ℝ := 0.85 * B):
  (new_L * new_B = 1.0625 * (L * B)) :=
by
  sorry

end rectangle_area_change_l89_89116


namespace sequence_problems_l89_89982

noncomputable def sequence_sum (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  if n = 0 then 0 else a n + sequence_sum (n-1) a

theorem sequence_problems (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → S n = 2 * a n - 3 * n) →
  a 1 = 3 ∧ a 2 = 9 ∧ a 3 = 21 ∧
  (∃ λ : ℕ, (a 2 + λ)^2 = (a 1 + λ) * (a 3 + λ) ∧
             λ = 3 ∧ ∀ n : ℕ, n > 0 → a n = 6 * 2^(n-1) - 3) := by
  sorry

end sequence_problems_l89_89982


namespace workers_time_together_l89_89138

theorem workers_time_together (T : ℝ) (h1 : ∀ t : ℝ, (T + 8) = t → 1 / t = 1 / (T + 8))
                                (h2 : ∀ t : ℝ, (T + 4.5) = t → 1 / t = 1 / (T + 4.5))
                                (h3 : 1 / (T + 8) + 1 / (T + 4.5) = 1 / T) : T = 6 :=
sorry

end workers_time_together_l89_89138


namespace num_circles_rectangle_l89_89052

structure Rectangle (α : Type*) [Field α] :=
  (A B C D : α × α)
  (AB_parallel_CD : B.1 = A.1 ∧ D.1 = C.1)
  (AD_parallel_BC : D.2 = A.2 ∧ C.2 = B.2)

def num_circles_with_diameter_vertices (R : Rectangle ℝ) : ℕ :=
  sorry

theorem num_circles_rectangle (R : Rectangle ℝ) : num_circles_with_diameter_vertices R = 5 :=
  sorry

end num_circles_rectangle_l89_89052


namespace part1_part2_l89_89664

theorem part1 (A B C : ℝ) (h1 : A = 2 * B) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) :
  C = 5 / 8 * π :=
sorry

theorem part2 (a b c : ℝ) (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) (h3 : a / sin A = b / sin B) (h4 : b / sin B = c / sin C) :
  2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l89_89664


namespace select_at_least_one_woman_probability_l89_89545

theorem select_at_least_one_woman_probability (men women total selected : ℕ) (h_men : men = 8) (h_women : women = 4) (h_total : total = men + women) (h_selected : selected = 4) :
  let total_prob := 1
  let prob_all_men := (men.to_rat / total) * ((men - 1).to_rat / (total - 1)) * ((men - 2).to_rat / (total - 2)) * ((men - 3).to_rat / (total - 3))
  let prob_at_least_one_woman := total_prob - prob_all_men
  prob_at_least_one_woman = 85 / 99 := by
  sorry

end select_at_least_one_woman_probability_l89_89545


namespace reachable_cells_after_moves_l89_89631

def is_valid_move (n : ℕ) (x y : ℤ) : Prop :=
(abs x ≤ n ∧ abs y ≤ n ∧ (x + y) % 2 = 0)

theorem reachable_cells_after_moves (n : ℕ) :
  n = 10 → ∃ (cells : Finset (ℤ × ℤ)), cells.card = 121 ∧ 
  (∀ (cell : ℤ × ℤ), cell ∈ cells → is_valid_move n cell.1 cell.2) :=
by
  intros h
  use {-10 ≤ x, y, x + y % 2 = 0 & abs x + abs y ≤ n }
  sorry -- proof goes here

end reachable_cells_after_moves_l89_89631


namespace sum_of_first_5_terms_l89_89032

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * a n

def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a i

theorem sum_of_first_5_terms : S 5 = 31 := by
  sorry

end sum_of_first_5_terms_l89_89032


namespace tangent_condition_l89_89605

-- Definitions based on conditions
def line_parametric (t k : ℝ) : ℝ × ℝ := (t, 1 + k * t)

def curve_polar (ρ θ : ℝ) : Prop := ρ * (sin θ)^2 = 4 * cos θ

def line_cartesian (k x y : ℝ) : Prop := k * x - y + 1 = 0

def curve_cartesian (x y : ℝ) : Prop := y^2 = 4 * x

-- The main statement capturing the proof problem
theorem tangent_condition (k : ℝ) :
  (∀ t: ℝ, ∃ x y: ℝ, line_parametric t k = (x, y) ∧ line_cartesian k x y) →
  (∀ ρ θ: ℝ, ∃ x y: ℝ, curve_polar ρ θ → ρ = sqrt (x^2 + y^2) ∧ y = ρ * sin θ ∧ x = ρ * cos θ → curve_cartesian x y) →
  (∃ x: ℝ, ∀ y: ℝ, (line_cartesian k x y) ∧ curve_cartesian x y →
  (k * x - y + 1 = 0 ∧ y^2 = 4 * x → (2 * k - 4)^2 - 4 * k^2 = 0)) →
  k = 1 :=
by
  sorry

end tangent_condition_l89_89605


namespace math_problem_l89_89060

noncomputable def alpha : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def beta : ℝ := 3 - 2 * Real.sqrt 2
noncomputable def x : ℝ := alpha ^ 500
noncomputable def n : ℕ := ⌊x⌋
noncomputable def f : ℝ := x - n

theorem math_problem : x * (1 - f) = 1 := sorry

end math_problem_l89_89060


namespace area_ratio_l89_89510

-- Definitions for the problem
variables {A B C P : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]

-- Condition that point P lies in the plane of triangle ABC and satisfies PA - PB - PC = BC
theorem area_ratio (h : dist A P - dist B P - dist C P = dist B C) :
  let area_ABC := (1 / 2) * (dist A B) * (dist B C),
      area_ABP := (1 / 2) * (dist A B) * (2 * (dist B C)) in
  area_ABP / area_ABC = 2 :=
by
  sorry

end area_ratio_l89_89510


namespace tan_315_eq_neg1_l89_89337

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89337


namespace divisible_by_5_last_digit_l89_89787

theorem divisible_by_5_last_digit (B : ℕ) (h : B < 10) : (∃ k : ℕ, 5270 + B = 5 * k) ↔ B = 0 ∨ B = 5 :=
by sorry

end divisible_by_5_last_digit_l89_89787


namespace kim_saplings_left_l89_89046

def sprouted_pits (total_pits num_sprouted_pits: ℕ) (percent_sprouted: ℝ) : Prop :=
  percent_sprouted * total_pits = num_sprouted_pits

def sold_saplings (total_saplings saplings_sold saplings_left: ℕ) : Prop :=
  total_saplings - saplings_sold = saplings_left

theorem kim_saplings_left
  (total_pits : ℕ) (num_sprouted_pits : ℕ) (percent_sprouted : ℝ)
  (saplings_sold : ℕ) (saplings_left : ℕ) :
  total_pits = 80 →
  percent_sprouted = 0.25 →
  saplings_sold = 6 →
  sprouted_pits total_pits num_sprouted_pits percent_sprouted →
  sold_saplings num_sprouted_pits saplings_sold saplings_left →
  saplings_left = 14 :=
by
  intros
  sorry

end kim_saplings_left_l89_89046


namespace tan_315_eq_neg_one_l89_89224

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89224


namespace number_of_women_l89_89866

theorem number_of_women (M W : ℕ) (hM : M = 9) (h1 : ∀ m, m < M → ∃ w, w < W ∧ m = 4 * w) 
  (h2 : ∀ w, w < W → ∃ m, m < M ∧ w = 3 * m) : W = 12 := 
by 
  have total_pairs := 9 * 4
  have women_paired := total_pairs / 3
  exact W = women_paired


end number_of_women_l89_89866


namespace tangent_315_deg_l89_89283

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89283


namespace height_percentage_l89_89543

theorem height_percentage (a b c : ℝ) 
  (h1 : a = 0.6 * b) 
  (h2 : c = 1.25 * a) : 
  (b - a) / a * 100 = 66.67 ∧ (c - a) / a * 100 = 25 := 
by 
  sorry

end height_percentage_l89_89543


namespace intersection_A_B_l89_89479

-- Definitions based on problem conditions
def setA : Set ℝ := { x | 2 * x ≤ 1 }
def setB : Set ℝ := { -1, 0, 1 }

-- Lean 4 statement to prove the intersection of setA and setB
theorem intersection_A_B : setA ∩ setB = {-1, 0} := by
  sorry

end intersection_A_B_l89_89479


namespace num_proper_subsets_l89_89749

def set_def : Set ℤ := {x | x^2 - 1 = 0}

-- We prove that the number of proper subsets of the given set is 3.
theorem num_proper_subsets : 
  (∃ (s : Set ℤ), s = set_def ∧ (s.elements = {1, -1})) ⟹
  (count_proper_subsets s = 3) := 
sorry

end num_proper_subsets_l89_89749


namespace contractor_net_amount_l89_89833

-- Definitions based on conditions
def total_days : ℕ := 30
def pay_per_day : ℝ := 25
def fine_per_absence_day : ℝ := 7.5
def days_absent : ℕ := 6

-- Calculate days worked
def days_worked : ℕ := total_days - days_absent

-- Calculate total earnings
def earnings : ℝ := days_worked * pay_per_day

-- Calculate total fine
def fine : ℝ := days_absent * fine_per_absence_day

-- Calculate net amount received by the contractor
def net_amount : ℝ := earnings - fine

-- Problem statement: Prove that the net amount is Rs. 555
theorem contractor_net_amount : net_amount = 555 := by
  sorry

end contractor_net_amount_l89_89833


namespace kvass_affordability_l89_89727

theorem kvass_affordability (x y : ℚ) (hx : x + y = 1) (hxy : 1.2 * (0.5 * x + y) = 1) : 1.44 * y ≤ 1 :=
by
  -- Placeholder for proof
  sorry

end kvass_affordability_l89_89727


namespace tan_315_eq_neg1_l89_89239

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89239


namespace tan_315_eq_neg1_l89_89246

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89246


namespace michael_crayon_cost_l89_89691

section
variable (initial_packs : ℕ) (packs_to_buy : ℕ) (cost_per_pack : ℝ) 

-- Given conditions
def michael_initial_packs : ℕ := 4
def michael_packs_to_buy : ℕ := 2
def pack_cost : ℝ := 2.5

-- Theorem statement
theorem michael_crayon_cost :
  let total_packs := michael_initial_packs + michael_packs_to_buy in
  let total_cost := total_packs * pack_cost in
  total_cost = 15 := by
  sorry
end

end michael_crayon_cost_l89_89691


namespace students_wanted_fruit_l89_89123

theorem students_wanted_fruit (red_apples green_apples extra_fruit : ℕ)
  (h_red : red_apples = 42)
  (h_green : green_apples = 7)
  (h_extra : extra_fruit = 40) :
  red_apples + green_apples + extra_fruit - (red_apples + green_apples) = 40 :=
by
  sorry

end students_wanted_fruit_l89_89123


namespace hyperbola_eccentricity_l89_89995

theorem hyperbola_eccentricity (O P A B : Point) (a : ℝ) (h : a > 0) 
  (OnHyperbola : P ∈ { (x, y) : ℝ × ℝ | x^2 / a^2 - y^2 = 1 })
  (ParallelOA : (∃ k : ℝ, A = (k, k / a)) ∧ (∃ k : ℝ, B = (k, -k / a)))
  (AreaParallelogram : 1 = abs ((A.1 - O.1) * (P.2 - O.2))) :
  eccentricity = Real.sqrt(5) / 2 := 
sorry

end hyperbola_eccentricity_l89_89995


namespace sum_of_sequence_is_24_l89_89189

-- Definitions of the sequence b_n
noncomputable def b : ℕ → ℚ
| 0       := 2  -- Since sequences are usually indexed from 0 in Lean, adjust accordingly
| 1       := 3
| (n + 2) := (1 / 2) * b (n + 1) + (1 / 3) * b n

-- Definition of the infinite sum of the sequence b_n
noncomputable def T : ℚ := ∑' n, b n

-- The theorem stating T = 24
theorem sum_of_sequence_is_24 : T = 24 :=
by
  sorry

end sum_of_sequence_is_24_l89_89189


namespace tan_315_eq_neg1_l89_89232

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89232


namespace find_a_l89_89775

noncomputable def direction_vector1 : ℝ × ℝ × ℝ := (4, 1, -6)
noncomputable def direction_vector2 (a : ℝ) : ℝ × ℝ × ℝ := (a, 3, 5)

theorem find_a (a : ℝ) (h : (direction_vector1.1 * a + direction_vector1.2 * (direction_vector2 a).2 + direction_vector1.3 * (direction_vector2 a).3) = 0) :
  a = 27 / 4 :=
sorry

end find_a_l89_89775


namespace line_equation_l89_89897

def P := (1 : ℝ, 2 : ℝ)
def A := (2 : ℝ, 3 : ℝ)
def B := (0 : ℝ, -5 : ℝ)

theorem line_equation
  (h_eq_dist : ∀ (l : ℝ → ℝ), 
    let dist := λ (p : ℝ × ℝ), abs (l p.fst - p.snd) / sqrt (l 1 - l 0)^2 + 1 in
    dist A = dist B)
  : (∀ x y, (4 * x - y - 2 = 0) ↔ (P = (x, y))) ∨ (x = 1) :=
by
  sorry

end line_equation_l89_89897


namespace pond_length_l89_89744

theorem pond_length (L W S : ℝ) (h1 : L = 2 * W) (h2 : L = 80) (h3 : S^2 = (1/50) * (L * W)) : S = 8 := 
by 
  -- Insert proof here 
  sorry

end pond_length_l89_89744


namespace tan_315_eq_neg_one_l89_89229

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89229


namespace sum_arithmetic_sequence_l89_89212

theorem sum_arithmetic_sequence : 
  let a1 := 1 
  let d := 2 
  let an := 19 
  let n := 10 
  S_n = n / 2 * (a1 + an)
  S_10 = 100

end sum_arithmetic_sequence_l89_89212


namespace lcm_15_48_eq_240_l89_89145

def is_least_common_multiple (n a b : Nat) : Prop :=
  n % a = 0 ∧ n % b = 0 ∧ ∀ m, (m % a = 0 ∧ m % b = 0) → n ≤ m

theorem lcm_15_48_eq_240 : is_least_common_multiple 240 15 48 :=
by
  sorry

end lcm_15_48_eq_240_l89_89145


namespace calculation_part_inequality_system_no_solution_l89_89814

theorem calculation_part : (Real.sqrt 3) ^ 2 - (2023 + Real.pi / 2) ^ 0 - (-1) ^ (-1) = 3 := by
  sorry

theorem inequality_system_no_solution :
  ¬(∃ x : ℝ, 5 * x - 4 > 3 * x ∧ (2 * x - 1) / 3 < x / 2) := by
  sorry

end calculation_part_inequality_system_no_solution_l89_89814


namespace repeating_decimal_sum_is_fraction_l89_89441

noncomputable def repeatingDecimalAsFraction (a b : ℕ) : ℚ :=
(a / (10^(b) - 1)).cast ℚ

theorem repeating_decimal_sum_is_fraction :
  let x := repeatingDecimalAsFraction 23 2,
      y := repeatingDecimalAsFraction 56 3,
      z := repeatingDecimalAsFraction 4  3 in
   x + y + z = (28917 : ℚ) / (98901 : ℚ) :=
by 
  sorry

end repeating_decimal_sum_is_fraction_l89_89441


namespace quadratic_real_roots_l89_89968

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) →
  (m ≤ 4 ∧ m ≠ 3) :=
sorry

end quadratic_real_roots_l89_89968


namespace remainder_123456789012_div_252_l89_89912

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l89_89912


namespace absolute_operation_statements_correctness_l89_89464

noncomputable def abs_op (xs : List ℤ) : ℤ :=
  List.sum (List.map (λ p => Int.natAbs (p.1 - p.2)) (List.diag xs))

theorem absolute_operation_statements_correctness :
  abs_op [1, 3, 5, 10] = 29 ∧
  (∀ x : ℤ, abs_op [x, -2, 5] ≥ 14) ∧
  (∃ (a b c : ℤ), abs_op [a, b, b, c] = 6) →
  1 = 1 := sorry

end absolute_operation_statements_correctness_l89_89464


namespace shaded_region_area_l89_89202

def isosceles_triangle (AB AC BC : ℝ) (BAC : ℝ) : Prop :=
  AB = AC ∧ BAC = 120 ∧ BC = 32

def circle_with_diameter (diameter : ℝ) (radius : ℝ) : Prop :=
  radius = diameter / 2

theorem shaded_region_area :
  ∀ (AB AC BC : ℝ) (BAC : ℝ) (O : Type) (a b c : ℕ),
    isosceles_triangle AB AC BC BAC →
    circle_with_diameter BC 8 →
    (a = 43) ∧ (b = 128) ∧ (c = 3) →
    a + b + c = 174 :=
by
  sorry

end shaded_region_area_l89_89202


namespace tan_315_eq_neg_1_l89_89329

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89329


namespace spatial_relationship_l89_89987

variables (a b c d : Type) [linear_ordered_field a] [linear_ordered_field b] [linear_ordered_field c] [linear_ordered_field d]
variables (α β : Type)
variables [plane α] [plane β]

-- a ⊥ b
variable (a_perp_b : orthogonal a b)
-- a ⊥ α
variable (a_perp_alpha : orthogonal a α)
-- c ⊥ α
variable (c_perp_alpha : orthogonal c α)

theorem spatial_relationship : orthogonal c b :=
by sorry

end spatial_relationship_l89_89987


namespace square_side_length_l89_89886

theorem square_side_length (s : ℝ) (h : 8 * s^2 = 3200) : s = 20 :=
by
  sorry

end square_side_length_l89_89886


namespace total_production_cost_l89_89824

-- Conditions
def first_season_episodes : ℕ := 12
def remaining_season_factor : ℝ := 1.5
def last_season_episodes : ℕ := 24
def first_season_cost_per_episode : ℝ := 100000
def other_season_cost_per_episode : ℝ := first_season_cost_per_episode * 2

-- Number of seasons
def number_of_seasons : ℕ := 5

-- Question: Calculate the total cost
def total_first_season_cost : ℝ := first_season_episodes * first_season_cost_per_episode
def second_season_episodes : ℕ := (first_season_episodes * remaining_season_factor).toNat
def second_season_cost : ℝ := second_season_episodes * other_season_cost_per_episode
def third_and_fourth_seasons_cost : ℝ := 2 * second_season_cost
def last_season_cost : ℝ := last_season_episodes * other_season_cost_per_episode
def total_cost : ℝ := total_first_season_cost + second_season_cost + third_and_fourth_seasons_cost + last_season_cost

-- Proof
theorem total_production_cost :
  total_cost = 16800000 :=
by
  sorry

end total_production_cost_l89_89824


namespace num_solutions_fn_l89_89065

-- Define the base function f
def f (x : ℝ) : ℝ := abs (1 - 2 * x)

-- Define the iterative functions f_n
def f_n : ℕ → (ℝ → ℝ)
| 0     := id
| (n+1) := f ∘ f_n n

-- Define the interval [0, 1]
def interval_01 := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- State the theorem to be proven
theorem num_solutions_fn (n : ℕ) : 
  ∃ S : finset ℝ, (∀ x ∈ S, x ∈ interval_01 ∧ f_n n x = (1 / 2) * x) ∧ S.card = 2^n :=
sorry

end num_solutions_fn_l89_89065


namespace tan_315_eq_neg_one_l89_89296

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89296


namespace tan_315_degrees_l89_89364

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89364


namespace volume_truncated_cone_l89_89764

/-- 
Given a truncated right circular cone with a large base radius of 10 cm,
a smaller base radius of 3 cm, and a height of 9 cm, 
prove that the volume of the truncated cone is 417 π cubic centimeters.
-/
theorem volume_truncated_cone :
  let R := 10
  let r := 3
  let h := 9
  let V := (1/3) * Real.pi * h * (R^2 + R*r + r^2)
  V = 417 * Real.pi :=
by 
  sorry

end volume_truncated_cone_l89_89764


namespace tan_315_eq_neg1_l89_89238

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89238


namespace infinite_product_value_l89_89884

noncomputable def infinite_product : ℝ := ∏' (n : ℕ) in Finset.range 1000, ((3 ^ (n + 1)) ^ (1 / (3 ^ (n + 1))))

theorem infinite_product_value :
  infinite_product = real.rpow 27 (1/4) :=
sorry

end infinite_product_value_l89_89884


namespace tan_315_eq_neg1_l89_89261

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89261


namespace isosceles_triangle_ratio_HD_HA_l89_89417

theorem isosceles_triangle_ratio_HD_HA (A B C D H : ℝ) :
  let AB := 13;
  let AC := 13;
  let BC := 10;
  let s := (AB + AC + BC) / 2;
  let area := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC));
  let h := (2 * area) / BC;
  let AD := h;
  let HA := h;
  let HD := 0;
  HD / HA = 0 := sorry

end isosceles_triangle_ratio_HD_HA_l89_89417


namespace tan_315_eq_neg_1_l89_89317

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89317


namespace reachable_cells_after_10_moves_l89_89627

def adjacent_cells (x y : ℕ) : set (ℕ × ℕ) :=
  { (x', y') | (x' = x + 1 ∧ y' = y) ∨ (x' = x - 1 ∧ y' = y) 
            ∨ (x' = x ∧ y' = y + 1) ∨ (x' = x ∧ y' = y - 1) }

def in_bounds (x y : ℕ) : Prop :=
  x > 0 ∧ x ≤ 21 ∧ y > 0 ∧ y ≤ 21

theorem reachable_cells_after_10_moves : 
  ∃ cells : set (ℕ × ℕ), ∃ initial_position : (11, 11) ∈ cells ∧ 
  (∀ (x y : ℕ), (x, y) ∈ cells → in_bounds x y ∧
  (∀ n ≤ 10, (x', y') ∈ adjacent_cells x y → (x', y') ∈ cells)) ∧ 
  (set.card cells = 121) :=
sorry

end reachable_cells_after_10_moves_l89_89627


namespace tan_315_proof_l89_89308

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89308


namespace length_of_bridge_is_correct_l89_89811

def train_length : ℝ := 200 -- in meters

def train_speed : ℝ := 60 * (1000 / 3600) -- converting speed from km/hr to m/s

def crossing_time : ℝ := 40 -- in seconds

def total_distance : ℝ := train_speed * crossing_time

def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is_correct :
  bridge_length = 466.8 := by
  sorry

end length_of_bridge_is_correct_l89_89811


namespace polynomial_at_n_plus_1_l89_89185

variable {n : ℕ}

-- Define the polynomial P and its properties
def P (k : ℕ) : ℝ := (k : ℝ) / ((k : ℝ) + 1)

theorem polynomial_at_n_plus_1 (P : ℕ → ℝ) (h_degree : ∃ (d : ℕ), d = n)
  (h_values : ∀ k : ℕ, k ≤ n → P k = (k : ℝ) / ((k : ℝ) + 1)) :
  P (n + 1) = (n + 1 + (-1)^(n + 1)) / (n + 2) :=
sorry

end polynomial_at_n_plus_1_l89_89185


namespace tyrone_gives_eric_19_marbles_l89_89784

theorem tyrone_gives_eric_19_marbles :
  ∃ x : ℕ, (120 - x = 3 * (15 + x)) ∧ x = 19 :=
by {
  use 19,
  split,
  {
    -- Prove 120 - 19 = 3 * (15 + 19)
    calc 
      120 - 19 = 101 : by norm_num
      ... = 3 * 34  : by norm_num
      ... = 3 * (15 + 19) : by norm_num
  },
  {
    -- Prove x = 19
    refl
  },
}

end tyrone_gives_eric_19_marbles_l89_89784


namespace three_digit_log3_eq_whole_and_log3_log9_eq_whole_l89_89462

noncomputable def logBase (b : ℝ) (x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem three_digit_log3_eq_whole_and_log3_log9_eq_whole (n : ℕ) (hn : 100 ≤ n ∧ n ≤ 999) (hlog3 : ∃ x : ℤ, logBase 3 n = x) (hlog3log9 : ∃ k : ℤ, logBase 3 n + logBase 9 n = k) :
  n = 729 := sorry

end three_digit_log3_eq_whole_and_log3_log9_eq_whole_l89_89462


namespace x_one_minus_f_eq_one_l89_89063

theorem x_one_minus_f_eq_one :
  let x := (3 + 2 * Real.sqrt 2)^500 
  let n := Int.floor x 
  let f := x - n 
  x * (1 - f) = 1 :=
by 
  sorry

end x_one_minus_f_eq_one_l89_89063


namespace x_one_minus_f_eq_one_l89_89062

theorem x_one_minus_f_eq_one :
  let x := (3 + 2 * Real.sqrt 2)^500 
  let n := Int.floor x 
  let f := x - n 
  x * (1 - f) = 1 :=
by 
  sorry

end x_one_minus_f_eq_one_l89_89062


namespace lattice_points_on_circle_l89_89604

theorem lattice_points_on_circle : ∀ (c : Int × Int) (r : Int), c = (199, 0) ∧ r = 199 →
 ∃ (points : Finset (Int × Int)), points.card = 4 ∧ 
 points = {(199, 199), (199, -199), (0, 0), (398, 0)} :=
by
  intros c r hc
  cases hc with c_eq r_eq
  use {(199, 199), (199, -199), (0, 0), (398, 0)}
  split
  · simp
  · assumption
  sorry

end lattice_points_on_circle_l89_89604


namespace tan_315_eq_neg1_l89_89267

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89267


namespace range_of_m_l89_89989

theorem range_of_m {a b c x0 y0 y1 y2 m : ℝ} (h1 : a ≠ 0)
    (A_on_parabola : y1 = a * m^2 + 4 * a * m + c)
    (B_on_parabola : y2 = a * (m + 2)^2 + 4 * a * (m + 2) + c)
    (C_on_parabola : y0 = a * (-2)^2 + 4 * a * (-2) + c)
    (C_is_vertex : x0 = -2)
    (y_relation : y0 ≥ y2 ∧ y2 > y1) :
    m < -3 := 
sorry

end range_of_m_l89_89989


namespace tan_315_eq_neg1_l89_89357

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89357


namespace michael_crayon_cost_l89_89692

section
variable (initial_packs : ℕ) (packs_to_buy : ℕ) (cost_per_pack : ℝ) 

-- Given conditions
def michael_initial_packs : ℕ := 4
def michael_packs_to_buy : ℕ := 2
def pack_cost : ℝ := 2.5

-- Theorem statement
theorem michael_crayon_cost :
  let total_packs := michael_initial_packs + michael_packs_to_buy in
  let total_cost := total_packs * pack_cost in
  total_cost = 15 := by
  sorry
end

end michael_crayon_cost_l89_89692


namespace negation_equiv_l89_89117

def neg_of_at_least_two_negatives (a b c : ℝ) : Prop :=
  ¬ (if h : (a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (b < 0 ∧ c < 0) then true else false)

def at_most_one_negative (a b c : ℝ) : Prop :=
  (if h : (a < 0) + (b < 0) + (c < 0) ≤ 1 then true else false)

theorem negation_equiv (a b c : ℝ) :
  neg_of_at_least_two_negatives a b c ↔ at_most_one_negative a b c :=
sorry

end negation_equiv_l89_89117


namespace at_least_one_woman_selected_probability_l89_89554

-- Define the total number of people, men, and women
def total_people : Nat := 12
def men : Nat := 8
def women : Nat := 4
def selected_people : Nat := 4

-- Define the probability ratio of at least one woman being selected
def probability_at_least_one_woman_selected : ℚ := 85 / 99

-- Prove the probability is correct given the conditions
theorem at_least_one_woman_selected_probability :
  (probability_of_selecting_at_least_one_woman men women selected_people total_people) = probability_at_least_one_woman_selected :=
sorry

end at_least_one_woman_selected_probability_l89_89554


namespace smallest_cheetahs_in_drawers_l89_89870

theorem smallest_cheetahs_in_drawers (x : ℕ) (h : x > 0) : ∃ n : ℕ, n = 7 * x ∧ n % 6 = 0 ∧ n = 42 := by
  use 7 * 6
  split
  case left =>
    rw [mul_comm]
    trivial
  case right =>
    calc
      7 * 6 % 6 = (7 % 6) * (6 % 6) := Nat.mul_mod
          _   = 42 % 6 := by sorry

end smallest_cheetahs_in_drawers_l89_89870


namespace polar_eq_to_cartesian_eq_l89_89422

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
(ρ * cos θ, ρ * sin θ)

theorem polar_eq_to_cartesian_eq (ρ θ : ℝ) (h : ρ^2 * cos θ - ρ = 0) :
  (ρ = 0 ∨ ρ * cos θ = 1) ↔ ((ρ^2 = 0) ∨ (ρ * cos θ = 1)) :=
by
  sorry

end polar_eq_to_cartesian_eq_l89_89422


namespace height_of_recast_cone_l89_89153

noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def volume_cone (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

def radius_sphere (d : ℝ) : ℝ := d / 2
def radius_cone (d : ℝ) : ℝ := d / 2

theorem height_of_recast_cone :
  let d_sphere := 6
  let d_cone := 12
  let r_sphere := radius_sphere d_sphere
  let r_cone := radius_cone d_cone
  let v_sphere := volume_sphere r_sphere
  let v_cone h := volume_cone r_cone h
  v_sphere = v_cone 3 :=
by
  sorry

end height_of_recast_cone_l89_89153


namespace tan_315_proof_l89_89305

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89305


namespace chord_length_l89_89031

noncomputable def parametric_line_l (t : ℝ) : ℝ × ℝ :=
(1 + 4 / 5 * t, -1 - 3 / 5 * t)

def polar_curve_C (theta : ℝ) : ℝ :=
  √2 * Real.cos (theta + Real.pi / 4)

noncomputable def line_l_rect (x y : ℝ) : Prop :=
  3 * x + 4 * y + 1 = 0

noncomputable def curve_C_rect (x y : ℝ) : Prop :=
  x^2 + y^2 - x + y = 0

noncomputable def distance_from_center (x y : ℝ) (a b : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

theorem chord_length :
  let C := 1/2, -1/2, (√2)/2
  let l := 3, 4, 1
  distance_from_center (1/2) (-1/2) 3 4 1 = 1/10 →
  2 * Real.sqrt ((1/2)^2 - (1/10)^2) = 7/5 := 
sorry

end chord_length_l89_89031


namespace exists_set_100_distinct_natural_numbers_l89_89713

theorem exists_set_100_distinct_natural_numbers:
  ∃ (c : Fin 100 → ℕ), (∀ i : Fin (99), ∃ k : ℕ, (c i)^2 + (c (⟨i.1 + 1, by simp [i.2]⟩))^2 = k^2) :=
by
  let c := λ i : Fin 100, 3^(100 - i.1) * 4^(i.1)
  existsi c
  intros i
  use 3^(99 - i.1) * 4^(i.1 - 1) * 5
  sorry

end exists_set_100_distinct_natural_numbers_l89_89713


namespace remainder_of_large_number_l89_89953

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l89_89953


namespace tan_315_eq_neg1_l89_89234

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89234


namespace tan_315_eq_neg_one_l89_89227

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89227


namespace scores_greater_than_18_l89_89590

noncomputable def olympiad_scores (scores : Fin 20 → ℕ) :=
∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k

theorem scores_greater_than_18 (scores : Fin 20 → ℕ) (h1 : ∀ i j, i < j → scores i < scores j)
  (h2 : olympiad_scores scores) : ∀ i, 18 < scores i :=
by
  intro i
  sorry

end scores_greater_than_18_l89_89590


namespace find_f_at_4_l89_89111

noncomputable def f : ℝ → ℝ := sorry -- We assume such a function exists

theorem find_f_at_4:
  (∀ x : ℝ, f (4^x) + x * f (4^(-x)) = 3) → f (4) = 0 := by
  intro h
  -- Proof would go here, but is omitted as per instructions
  sorry

end find_f_at_4_l89_89111


namespace tan_315_degree_l89_89413

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89413


namespace reachable_cells_after_10_moves_l89_89624

theorem reachable_cells_after_10_moves :
  let board_size := 21
  let central_cell := (11, 11)
  let moves := 10
  (reachable_cells board_size central_cell moves) = 121 :=
by
  sorry

end reachable_cells_after_10_moves_l89_89624


namespace arithmetic_sequence_properties_l89_89985

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

def condition_S10_pos (S : ℕ → ℝ) : Prop :=
S 10 > 0

def condition_S11_neg (S : ℕ → ℝ) : Prop :=
S 11 < 0

-- Main statement
theorem arithmetic_sequence_properties {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
  (ar_seq : is_arithmetic_sequence a d)
  (sum_first_n : sum_of_first_n_terms S a)
  (S10_pos : condition_S10_pos S)
  (S11_neg : condition_S11_neg S) :
  (∀ n, (S n) / n = a 1 + (n - 1) / 2 * d) ∧
  (a 2 = 1 → -2 / 7 < d ∧ d < -1 / 4) :=
by
  sorry

end arithmetic_sequence_properties_l89_89985


namespace remainder_123456789012_div_252_l89_89915

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l89_89915


namespace range_of_a_l89_89524

theorem range_of_a (a : ℝ) : 
  (∀ x ≤ 3, (f(x) = x^2 + 2*(a-1)*x + 2) → deriv f x ≤ 0) ↔ 
  a ≤ -2 := 
sorry

end range_of_a_l89_89524


namespace geometric_sequence_sixth_term_l89_89460

theorem geometric_sequence_sixth_term (a1 a2 : ℕ) (r : ℕ) : 
  a1 = 5 → a2 = 15 → r = a2 / a1 → (a1 * r ^ (6 - 1)) = 1215 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end geometric_sequence_sixth_term_l89_89460


namespace tan_315_eq_neg1_l89_89389

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89389


namespace length_MN_l89_89085

theorem length_MN (a b : ℝ) 
    (M N : Point)
    (H1 : MN_parallel_AD)
    (H2 : area_ratio_condition (Trapezoid MBCN) (Trapezoid AMND) (2/3))
    (H3 : BC_length = a)
    (H4 : AD_length = b) 
    : MN_length = sqrt (3 * a^2 + 2 * b^2) / 5 :=
sorry

end length_MN_l89_89085


namespace remainder_123456789012_mod_252_l89_89901

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l89_89901


namespace exists_sequence_composite_sums_l89_89653

def is_composite (n : ℕ) : Prop := 
  ∃ m (k : ℕ), 1 < m ∧ 1 < k ∧ m * k = n

def is_coprime (a b : ℕ) : Prop := 
  Nat.gcd a b = 1

theorem exists_sequence_composite_sums :
  ∃ (a : Fin 2016 → ℕ),
    (∀ r s : Fin 2016, r ≤ s → is_composite (Finset.sum (Finset.Icc r s) (λ i, a i))) ∧
    (∀ i : Fin 2015, is_coprime (a i) (a ⟨i + 1, Nat.lt_succ_of_lt i.prop⟩)) ∧
    (∀ i : Fin 2014, is_coprime (a i) (a ⟨i + 2, Nat.succ_lt_succ i.prop⟩)) :=
sorry

end exists_sequence_composite_sums_l89_89653


namespace triangle_area_l89_89108

theorem triangle_area (a b c : ℝ)
    (h1 : Polynomial.eval a (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h2 : Polynomial.eval b (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h3 : Polynomial.eval c (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (sum_roots : a + b + c = 4)
    (sum_prod_roots : a * b + a * c + b * c = 5)
    (prod_roots : a * b * c = 1):
    Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) = 1 :=
  sorry

end triangle_area_l89_89108


namespace average_speed_1_ms_instantaneous_speed_1_ms_l89_89520

noncomputable def S (t : ℝ) := 3 * t - t^2

theorem average_speed_1_ms :
  (S 2 - S 0) / (2 - 0) = 1 :=
by
  sorry

theorem instantaneous_speed_1_ms :
  |(derivative S) 2| = 1 :=
by
  sorry

end average_speed_1_ms_instantaneous_speed_1_ms_l89_89520


namespace bound_on_P_l89_89678

theorem bound_on_P (S : Finset ℕ) (P : Finset (Fin 100 → ℕ)) :
  S.card = 1990 →
  (∀ p ∈ P, ∀ i j : Fin 100, i ≠ j → (p i ≠ p j ∧ p i ∈ S ∧ p j ∈ S)) →
  (∀ p₁ p₂ ∈ P, p₁ ≠ p₂ → (∀ i j : Fin 100, i < j → (p₁ i, p₁ j) ≠ (p₂ i, p₂ j))) →
  P.card ≤ 800 :=
by
  intros hS hP1 hP2
  sorry

end bound_on_P_l89_89678


namespace sam_overall_average_speed_l89_89091

def time_motorcycling : ℝ := 45 / 60 -- in hours
def speed_motorcycling : ℝ := 30 -- in mph
def time_jogging : ℝ := 60 / 60 -- in hours
def speed_jogging : ℝ := 5 -- in mph

def distance_motorcycling : ℝ := speed_motorcycling * time_motorcycling
def distance_jogging : ℝ := speed_jogging * time_jogging
def total_distance : ℝ := distance_motorcycling + distance_jogging
def total_time : ℝ := time_motorcycling + time_jogging
def average_speed : ℝ := total_distance / total_time

theorem sam_overall_average_speed : average_speed = 15 := 
by
  -- proofs or calculations are intentionally skipped
  sorry

end sam_overall_average_speed_l89_89091


namespace remainder_of_large_number_l89_89951

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l89_89951


namespace remainder_div_2DD_l89_89476

theorem remainder_div_2DD' (P D D' Q R Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = 2 * D' * Q' + R') :
  P % (2 * D * D') = D * R' + R :=
sorry

end remainder_div_2DD_l89_89476


namespace bob_always_wins_l89_89862

open Nat

noncomputable def bob_move (n : ℕ) (a : ℕ) : ℕ := n - a * a
noncomputable def amy_move (n : ℕ) (k : ℕ) : ℕ := n ^ k

def is_valid_bob_move (n : ℕ) (a : ℕ) : Prop := a > 0 ∧ a * a ≤ n
def is_valid_amy_move (n : ℕ) (k : ℕ) : Prop := k > 0

theorem bob_always_wins (n : ℕ) (h : n > 0) : 
(∃ M : ℕ, ∀ (k : ℕ), is_valid_amy_move (M) (k) → (∃ a : ℕ, is_valid_bob_move (amy_move(M)(k)) (a) → bob_move (amy_move (M) (k)) (a) = 0)) :=
sorry

end bob_always_wins_l89_89862


namespace amelia_distance_l89_89107

theorem amelia_distance (total_distance amelia_monday_distance amelia_tuesday_distance : ℕ) 
  (h1 : total_distance = 8205) 
  (h2 : amelia_monday_distance = 907) 
  (h3 : amelia_tuesday_distance = 582) : 
  total_distance - (amelia_monday_distance + amelia_tuesday_distance) = 6716 := 
by 
  sorry

end amelia_distance_l89_89107


namespace simplify_log_expression_l89_89720

theorem simplify_log_expression (x : ℝ) (h : x > 1) :
  log 2 (2 * x^2) + log 2 x * x^(log x (log 2 x + 1)) + (1/2) * log 4 (x^4)^2 + 2^(-3 * log (1/2) (log 2 x)) =
    (log 2 x + 1)^3 := by
  sorry

end simplify_log_expression_l89_89720


namespace measure_angle_y_l89_89899

-- Define the conditions
def angle_BDE : ℝ := 115 -- in degrees
def angle_EBD : ℝ := 30 -- in degrees
def triangle_angle_sum (α β γ : ℝ) : Prop := α + β + γ = 180

-- State the problem and expected answer
theorem measure_angle_y :
  ∃ y : ℝ, y = 35 ∧ triangle_angle_sum angle_BDE angle_EBD y :=
by
  -- We need to prove that there exists an angle y that satisfies both y = 35 and the angle sum property
  use 35
  split
  { refl }
  {
    unfold triangle_angle_sum
    linarith
  }

end measure_angle_y_l89_89899


namespace sum_palindromic_primes_l89_89082

def is_palindromic_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 200 ∧ Nat.Prime n ∧ Nat.Prime (n.reverse_digits)

def reverse_digits (n : ℕ) : ℕ :=
  let rec loop (n acc : ℕ) : ℕ :=
    if n = 0 then acc else loop (n / 10) (acc * 10 + n % 10)
  loop n 0

theorem sum_palindromic_primes : 
  (Finset.filter is_palindromic_prime (Finset.range 200)).sum = 452 :=
by
  sorry

end sum_palindromic_primes_l89_89082


namespace find_a_b_find_arithmetic_square_root_l89_89515

-- Define basic assumptions
variables {a b : ℤ}

-- Define conditions based on the problem statement
def condition1 := (∃ k : ℤ, a + b - 5 = k * k) ∧ ((k = 3) ∨ (k = -3))
def condition2 := ∃ m : ℤ, a - b + 4 = m * m * m ∧ (m = 2)

-- Define the main proof problem statements
theorem find_a_b (h1 : condition1) (h2 : condition2) : a = 9 ∧ b = 5 :=
sorry

theorem find_arithmetic_square_root (h : a = 9 ∧ b = 5) : ∃ x : ℕ, 4 * (a - b) = x * x ∧ x = 4 :=
sorry

end find_a_b_find_arithmetic_square_root_l89_89515


namespace tan_315_eq_neg_one_l89_89230

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89230


namespace equal_max_clique_sizes_l89_89726

-- Definitions based on the problem's conditions
def is_mutual_friendship (participants : Type) (friendship : participants → participants → Prop) :=
  ∀ a b, friendship a b → friendship b a

def is_clique (participants : Type) (friendship : participants → participants → Prop) (clique : set participants) :=
  ∀ a b ∈ clique, friendship a b

def max_clique_size (participants : Type) (friendship : participants → participants → Prop) :=
  max_set size { clique | is_clique participants friendship clique }

-- The main theorem we need to prove
theorem equal_max_clique_sizes (participants : Type) (friendship : participants → participants → Prop) 
  (h_mutual : is_mutual_friendship participants friendship)
  (h_even : ∃ P, max_clique_size participants friendship = 2 * P) :
  ∃ R1 R2, max_clique_size R1 friendship = max_clique_size R2 friendship :=
sorry

end equal_max_clique_sizes_l89_89726


namespace determine_n_l89_89121

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem determine_n :
  (∃ n : ℕ, digit_sum (9 * (10^n - 1)) = 999 ∧ n = 111) :=
sorry

end determine_n_l89_89121


namespace remainder_123456789012_div_252_l89_89947

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l89_89947


namespace ratio_y_coordinates_l89_89521

-- Variables and definitions based on conditions
variable {x y : ℝ}

def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1
def P := (1, 0 : ℝ)
def l1 (x y : ℝ) : Prop := x = -2
def l2 (x y : ℝ) : Prop := x = 2
def line_CD (x y : ℝ) : Prop := x = 1

def is_chord (A B : ℝ × ℝ) : Prop := ellipse A.1 A.2 ∧ ellipse B.1 B.2
def passes_through_P (A B : ℝ × ℝ) : Prop := 
  ∃ t : ℝ, ((A.1 = 1 + t * (P.1 - 1) ∧ A.2 = t * P.2) ∨ (B.1 = 1 + t * (P.1 - 1) ∧ B.2 = t * P.2))

def intersect_l (l : ℝ → Prop) (C D : ℝ × ℝ) (AC_line BD_line : ℝ → Prop) : ℝ × ℝ
 := if H : ∃ y : ℝ, AC_line (2 : ℝ) ∧ l 2 then (2, (some (exists_elim_2 H)))
    else if G : ∃ y : ℝ, BD_line (-2 : ℝ) ∧ l (-2) then (-2, (some (exists_elim_2 G)))
    else (0, 0)
 
def E (AC_line : ℝ → Prop) (C D : ℝ × ℝ) : ℝ × ℝ := intersect_l l2 C D AC_line
def F (BD_line : ℝ → Prop) (C D : ℝ × ℝ) : ℝ × ℝ := intersect_l l1 C D BD_line

-- The final statement to be proven: the ratio of y-coordinates of E and F
theorem ratio_y_coordinates {C D : ℝ × ℝ} (AC_line BD_line : ℝ → Prop) (hC : ellipse C.1 C.2)
  (hD : ellipse D.1 D.2) (hCD : line_CD C.1 C.2) (hCD' : line_CD D.1 D.2) (hPass : passes_through_P C D) :
  (E AC_line C D).snd / (F BD_line C D).snd = -1 / 3 :=
sorry

end ratio_y_coordinates_l89_89521


namespace tan_315_eq_neg_one_l89_89295

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89295


namespace solution_set_of_cx_sq_minus_bx_plus_a_l89_89000

theorem solution_set_of_cx_sq_minus_bx_plus_a (a b c : ℝ) (h1 : a < 0)
(h2 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ 2 < x ∧ x < 3) :
  ∀ x : ℝ, cx^2 - bx + a > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by
  sorry

end solution_set_of_cx_sq_minus_bx_plus_a_l89_89000


namespace part1_part2_l89_89506

noncomputable def f (x : ℝ) :=
  if x ≥ 0 then x^2 - 4 * x else -x^2 - 4 * x

theorem part1 :
  (∀ x, (f x = if x ≥ 0 then x^2 - 4 * x else -x^2 - 4 * x)) :=
begin
  sorry
end

theorem part2 :
  (∀ x, (f x = x + 6 ↔ x = 6 ∨ x = -2 ∨ x = -3)) :=
begin
  sorry
end

end part1_part2_l89_89506


namespace intersection_of_P_and_Q_l89_89662

theorem intersection_of_P_and_Q :
  let P := {-1, 1}
  let Q := {0, 1, 2}
  (P ∩ Q) = {1} :=
by {
  let P := {-1, 1}
  let Q := {0, 1, 2}
  show (P ∩ Q) = {1},
  sorry
}

end intersection_of_P_and_Q_l89_89662


namespace thursday_wednesday_ratio_l89_89840

-- Declaration of constants as given conditions
constant T : ℕ -- Copies sold on Thursday
constant F : ℕ -- Copies sold on Friday

-- Given conditions translated as definitions
def wednesday_sales := 15
def thursday_sales := T
def friday_sales := F
def total_sales := 69

-- Relating friday sales to thursday sales
def friday_sales_eq := friday_sales = thursday_sales / 5

-- Total sales up to Friday
def total_sales_eq := wednesday_sales + thursday_sales + friday_sales = total_sales

-- The target theorem statement proving the ratio
theorem thursday_wednesday_ratio :
  friday_sales_eq ∧ total_sales_eq → thursday_sales / wednesday_sales = 3 :=
by
  sorry

end thursday_wednesday_ratio_l89_89840


namespace at_least_one_woman_probability_l89_89548

noncomputable def probability_at_least_one_woman_selected 
  (total_men : ℕ) (total_women : ℕ) (selected_people : ℕ) : ℚ :=
  1 - (8 / 12 * 7 / 11 * 6 / 10 * 5 / 9)

theorem at_least_one_woman_probability :
  probability_at_least_one_woman_selected 8 4 4 = 85 / 99 := 
sorry

end at_least_one_woman_probability_l89_89548


namespace remainder_123456789012_mod_252_l89_89908

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l89_89908


namespace boy_to_total_ratio_l89_89002

-- Problem Definitions
variables (b g : ℕ) -- number of boys and number of girls

-- Hypothesis: The probability of choosing a boy is (4/5) the probability of choosing a girl
def probability_boy := b / (b + g : ℕ)
def probability_girl := g / (b + g : ℕ)

theorem boy_to_total_ratio (h : probability_boy b g = (4 / 5) * probability_girl b g) : 
  b / (b + g : ℕ) = 4 / 9 :=
sorry

end boy_to_total_ratio_l89_89002


namespace find_m_range_l89_89991

theorem find_m_range
  (m y1 y2 y0 x0 : ℝ)
  (a c : ℝ) (h1 : a ≠ 0)
  (h2 : x0 = -2)
  (h3 : ∀ x, (x, ax^2 + 4*a*x + c) = (m, y1) ∨ (x, ax^2 + 4*a*x + c) = (m + 2, y2) ∨ (x, ax^2 + 4*a*x + c) = (x0, y0))
  (h4 : y0 ≥ y2) (h5 : y2 > y1) :
  m < -3 :=
sorry

end find_m_range_l89_89991


namespace total_perimeter_of_border_l89_89696

theorem total_perimeter_of_border :
  let num_A := 125.0
  let circ_A := 0.5
  let num_B := 64.0
  let circ_B := 0.7
  let total_length_A := num_A * circ_A
  let total_length_B := num_B * circ_B
  total_length_A + total_length_B = 107.3 := by
{
  let num_A := 125.0,
  let circ_A := 0.5,
  let num_B := 64.0,
  let circ_B := 0.7,
  let total_length_A := num_A * circ_A,
  let total_length_B := num_B * circ_B,
  sorry
}

end total_perimeter_of_border_l89_89696


namespace output_S_when_n_eq_3_input_n_when_S_eq_30_l89_89522

def algorithm (n : ℕ) : ℕ :=
  Nat.recOn n 0 (λ i S, S + 2 * (i + 1))

theorem output_S_when_n_eq_3 :
  algorithm 3 = 12 :=
by
  -- proof of the theorem will go here
  sorry

theorem input_n_when_S_eq_30 :
  algorithm 5 = 30 :=
by
  -- proof of the theorem will go here
  sorry

end output_S_when_n_eq_3_input_n_when_S_eq_30_l89_89522


namespace tan_315_eq_neg1_l89_89245

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89245


namespace function_neither_even_nor_odd_l89_89036

noncomputable def f (x : ℝ) : ℝ := (4 * x ^ 3 - 3) / (x ^ 6 + 2)

theorem function_neither_even_nor_odd : 
  (∀ x : ℝ, f (-x) ≠ f x) ∧ (∀ x : ℝ, f (-x) ≠ -f x) :=
by
  sorry

end function_neither_even_nor_odd_l89_89036


namespace philip_school_trip_days_l89_89129

-- Define the distances for the trips
def school_trip_one_way_miles : ℝ := 2.5
def market_trip_one_way_miles : ℝ := 2

-- Define the number of times he makes the trips in a day and in a week
def school_round_trips_per_day : ℕ := 2
def market_round_trips_per_week : ℕ := 1

-- Define the total mileage in a week
def weekly_mileage : ℕ := 44

-- Define the equation based on the given conditions
def weekly_school_trip_distance (d : ℕ) : ℝ :=
  (school_trip_one_way_miles * 2 * school_round_trips_per_day) * d

def weekly_market_trip_distance : ℝ :=
  (market_trip_one_way_miles * 2) * market_round_trips_per_week

-- Define the main theorem to be proved
theorem philip_school_trip_days :
  ∃ d : ℕ, weekly_school_trip_distance d + weekly_market_trip_distance = weekly_mileage ∧ d = 4 :=
by
  sorry

end philip_school_trip_days_l89_89129


namespace simplify_expression_l89_89416

theorem simplify_expression (x : ℤ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 :=
by
  rw h
  norm_num
  sorry

end simplify_expression_l89_89416


namespace alexis_pants_l89_89038

theorem alexis_pants (P D : ℕ) (A_p : ℕ)
  (h1 : P + D = 13)
  (h2 : 3 * D = 18)
  (h3 : A_p = 3 * P) : A_p = 21 :=
  sorry

end alexis_pants_l89_89038


namespace f_expression_f_range_l89_89505

-- Definitions based on conditions
def is_quadratic (f : ℝ → ℝ) : Prop :=
∃ a b c : ℝ, ∀ x, f(x) = a*x^2 + b*x + c

-- Original functions and conditions
axiom f : ℝ → ℝ 
axiom f_is_quadratic : is_quadratic f
axiom f_zero : f(0) = 0
axiom f_x_plus_1 : ∀ x, f(x + 1) = f(x) + x + 1

-- Proving the analytical expression of f(x)
theorem f_expression : f = λ x, (1/2)*x^2 + (1/2)*x :=
by 
  sorry

-- Proving the range of y = f(x^2 - 2)
theorem f_range : ∀ y, (∃ x, y = f(x^2 - 2)) ↔ y ∈ Set.Ici (-1/8 : ℝ) :=
by 
  sorry

end f_expression_f_range_l89_89505


namespace tan_315_eq_neg1_l89_89387

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89387


namespace noise_pollution_l89_89701

variables (p p0 p1 p2 p3 : ℝ)
variable (p0_pos : p0 > 0)
variable (Lp1 Lp2 Lp3 : ℝ)
variable (gasoline_car_condition : 60 ≤ Lp1 ∧ Lp1 ≤ 90)
variable (hybrid_car_condition : 50 ≤ Lp2 ∧ Lp2 ≤ 60)
variable (electric_car_condition : Lp3 = 40)

theorem noise_pollution : 
  (20 * log10 (p1 / p0) = Lp1) ∧ (20 * log10 (p2 / p0) = Lp2) ∧ (20 * log10 (p3 / p0) = Lp3) ∧ 
  (60 ≤ Lp1 ∧ Lp1 ≤ 90) ∧ (50 ≤ Lp2 ∧ Lp2 ≤ 60) ∧ (Lp3 = 40) →
  (p1 ≥ p2) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) :=
by
  sorry

end noise_pollution_l89_89701


namespace remainder_2_power_404_l89_89794

theorem remainder_2_power_404 (y : ℕ) (h_y : y = 2^101) :
  (2^404 + 404) % (2^203 + 2^101 + 1) = 403 := by
sorry

end remainder_2_power_404_l89_89794


namespace C_share_correct_l89_89857

def investment_A := 27000
def investment_B := 72000
def investment_C := 81000
def total_profit := 80000

def gcd_investment : ℕ := Nat.gcd investment_A (Nat.gcd investment_B investment_C)
def ratio_A : ℕ := investment_A / gcd_investment
def ratio_B : ℕ := investment_B / gcd_investment
def ratio_C : ℕ := investment_C / gcd_investment
def total_parts : ℕ := ratio_A + ratio_B + ratio_C

def C_share : ℕ := (ratio_C / total_parts) * total_profit

theorem C_share_correct : C_share = 36000 := 
by sorry

end C_share_correct_l89_89857


namespace remainder_123456789012_div_252_l89_89913

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l89_89913


namespace grass_cut_possible_l89_89080

theorem grass_cut_possible (n : ℕ) : 
  ∀ (blades : Fin n → ℕ), 
  (∃ cuts : List (Fin n × ℕ), 
    cuts.length ≤ n - 1 ∧
    (∀ blade_idx, 
      blades blade_idx = 
      (total_length blades) / n)) → 
  true := 
sorry

end grass_cut_possible_l89_89080


namespace sentence_for_arson_is_36_l89_89040

theorem sentence_for_arson_is_36 (A : ℕ) 
  (arson_counts : ℕ := 3)
  (burglary_charges : ℕ := 2)
  (burglary_sentence : ℕ := 18)
  (petty_larceny_ratio : ℚ := 1/3)
  (total_sentence : ℕ := 216)
  (sentences : ℕ := arson_counts * A + 
                   burglary_charges * burglary_sentence + 
                   (6 * burglary_charges) * (burglary_sentence * petty_larceny_ratio)) :
  sentences = total_sentence → A = 36 :=
by
  -- Proof will be inserted here
  sorry

end sentence_for_arson_is_36_l89_89040


namespace select_at_least_one_woman_probability_l89_89547

theorem select_at_least_one_woman_probability (men women total selected : ℕ) (h_men : men = 8) (h_women : women = 4) (h_total : total = men + women) (h_selected : selected = 4) :
  let total_prob := 1
  let prob_all_men := (men.to_rat / total) * ((men - 1).to_rat / (total - 1)) * ((men - 2).to_rat / (total - 2)) * ((men - 3).to_rat / (total - 3))
  let prob_at_least_one_woman := total_prob - prob_all_men
  prob_at_least_one_woman = 85 / 99 := by
  sorry

end select_at_least_one_woman_probability_l89_89547


namespace tan_315_eq_neg_one_l89_89228

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89228


namespace num_cells_after_10_moves_l89_89614

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 + 1 = p2.1))

def num_reachable_cells (n m k : ℕ) (start : ℕ × ℕ) : ℕ :=
  sorry -- Calculation of the number of reachable cells after k moves

theorem num_cells_after_10_moves :
  let board_size := 21
  let start := (11, 11)
  let moves := 10
  num_reachable_cells board_size board_size moves start = 121 :=
sorry

end num_cells_after_10_moves_l89_89614


namespace max_value_exponential_expression_sup_of_exponential_expression_l89_89456

theorem max_value_exponential_expression :
  ∀ x : ℝ, 
  ∃ y : ℝ, 2^x - 3 * 4^x ≤ y ∧
  (∀ z : ℝ, (∃ x0 : ℝ, z = 2^x0 - 3 * 4^x0) → z ≤ y) :=
begin
  sorry
end

theorem sup_of_exponential_expression :
  ∀ x : ℝ,
  (2^x - 3 * 4^x ≤ 1/12) :=
begin
  sorry
end

end max_value_exponential_expression_sup_of_exponential_expression_l89_89456


namespace part1_part2_l89_89665

theorem part1 (A B C : ℝ) (h1 : A = 2 * B) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) :
  C = 5 / 8 * π :=
sorry

theorem part2 (a b c : ℝ) (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) (h3 : a / sin A = b / sin B) (h4 : b / sin B = c / sin C) :
  2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l89_89665


namespace no_carry_consecutive_pairs_l89_89471

/-- Consider the range of integers {2000, 2001, ..., 3000}. 
    We determine that the number of pairs of consecutive integers in this range such that their addition requires no carrying is 729. -/
theorem no_carry_consecutive_pairs : 
  ∀ (n : ℕ), (2000 ≤ n ∧ n < 3000) ∧ ((n + 1) ≤ 3000) → 
  ∃ (count : ℕ), count = 729 := 
sorry

end no_carry_consecutive_pairs_l89_89471


namespace tan_315_eq_neg1_l89_89351

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89351


namespace hyperbola_eccentricity_l89_89994

-- Define the given conditions
variables {a b c F1 F2 P : ℝ}
variables (h1 : a > 0) (h2 : b > 0)
variables (hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → (x, y) ≠ (0, 0))
variables (circle : ∀ x y, x^2 + y^2 = (c - a)^2 → (x, y) ≠ (F1, 0))
variables (P_first_quadrant : P > 0)
variables (area_triangle : 1 / 2 * F1 * F2 = a^2)

-- The theorem to prove the eccentricity of the hyperbola is sqrt(2)
theorem hyperbola_eccentricity : (c = a * Real.sqrt 2) → (c / a = Real.sqrt 2) := by
  intro hc
  field_simp [hc]
  rw Real.sqrt_div_self
  field_simp
  apply Real.sqrt_eq
  linarith

end hyperbola_eccentricity_l89_89994


namespace max_product_l89_89830

-- Define the unit square and circumscribed circle
structure Square :=
  (A B C D : ℝ)
  (unit : ∀ a b, (a ∈ {A, B, C, D}) ∧ (b ∈ {A, B, C, D}) ∧ a ≠ b → dist a b = 1)

def circumscribed_circle (s : Square) : set ℝ :=
  {m | ∃ r, r ≠ 0 ∧ dist(m, s.A) = r ∧ dist(m, s.B) = r ∧ dist(m, s.C) = r ∧ dist(m, s.D) = r}

def valid_point_circle (s : Square) (M : ℝ) : Prop :=
  M ∈ circumscribed_circle s

noncomputable def product_M (s : Square) (M : ℝ) : ℝ :=
  dist M s.A * dist M s.B * dist M s.C * dist M s.D

-- The final statement to prove
theorem max_product (s : Square) (M : ℝ) (hM : valid_point_circle s M) :
  (product_M s M) ≤ 0.5 :=
sorry

end max_product_l89_89830


namespace faster_train_length_l89_89137

theorem faster_train_length
  (speed_faster : ℝ)
  (speed_slower : ℝ)
  (time_to_cross : ℝ)
  (relative_speed_limit: ℝ)
  (h1 : speed_faster = 108 * 1000 / 3600)
  (h2: speed_slower = 36 * 1000 / 3600)
  (h3: time_to_cross = 17)
  (h4: relative_speed_limit = 2) :
  (speed_faster - speed_slower) * time_to_cross = 340 := 
sorry

end faster_train_length_l89_89137


namespace number_of_reachable_cells_after_10_moves_l89_89638

theorem number_of_reachable_cells_after_10_moves : 
  (let 
    n := 21 
    center := (11, 11)
    moves := 10
  in
  ∃ reachable_cells, reachable_cells = 121) :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89638


namespace lengths_of_bases_equal_l89_89079

theorem lengths_of_bases_equal
  (R : ℝ) (P Q : ℝ → ℝ)
  (H1 : R = sqrt(6))
  (H2 : ∀ x : ℝ, x = x)
  (H3 : PQ_parallel : P = Q - sqrt(2) ∨ P = Q + sqrt(2))
  (H_trapezoid : is_isosceles_trapezoid K L M N)
  (H_incircles : circles_inscribed KPQN PLMQ)
  : length LK = length NM := sorry

end lengths_of_bases_equal_l89_89079


namespace largest_integer_of_consecutive_sequence_l89_89841

theorem largest_integer_of_consecutive_sequence :
  ∀ (a : ℤ) (n : ℕ), a = -11 → n = 40 → (a + (n - 1) : ℤ) = 28 :=
by {
  intros,
  sorry
}

end largest_integer_of_consecutive_sequence_l89_89841


namespace selling_price_correct_l89_89827

-- Define the original price a.
variable (a : ℝ)

-- Define the intermediate price after the first discount.
def price_after_first_discount := a - 100

-- Define the selling price after applying an additional 10% discount.
def selling_price := 0.9 * price_after_first_discount a

-- The theorem states that, given the conditions, the selling price is 0.9 * (a - 100).
theorem selling_price_correct : selling_price a = 0.9 * (a - 100) :=
by
  unfold selling_price
  simp [price_after_first_discount]
  sorry

end selling_price_correct_l89_89827


namespace convex_polygon_homothety_half_l89_89088

noncomputable
def homothety {R : Type*} [linear_ordered_field R] (P : set (R × R)) (k : R) (O : R × R) : set (R × R) :=
  { Z | ∃ (X : R × R), X ∈ P ∧ Z = (k * (X.1 - O.1) + O.1, k * (X.2 - O.2) + O.2)}

theorem convex_polygon_homothety_half (P : set (ℝ × ℝ)) (hP : convex ℝ P) :
  ∃ (O : ℝ × ℝ), (homothety P (1/2) O) ⊆ P :=
sorry

end convex_polygon_homothety_half_l89_89088


namespace each_sister_received_cake_l89_89477

theorem each_sister_received_cake :
  ∀ (total_pieces : ℕ) (square_pieces : ℕ) (triangle_pieces : ℕ)
    (percentage_eaten_square : ℚ) (percentage_eaten_triangle : ℚ)
    (weight_square_piece : ℚ) (weight_triangle_piece : ℚ)
    (percentage_taken_family : ℚ) (percentage_taken_friends : ℚ)
    (number_of_sisters : ℕ),
    total_pieces = 240 ∧ square_pieces = 160 ∧ triangle_pieces = 80 ∧
    percentage_eaten_square = 0.60 ∧ percentage_eaten_triangle = 0.40 ∧
    weight_square_piece = 25 ∧ weight_triangle_piece = 20 ∧
    percentage_taken_family = 0.30 ∧ percentage_taken_friends = 0.25 ∧
    number_of_sisters = 3 →
  let eaten_square := percentage_eaten_square * square_pieces,
      eaten_triangle := percentage_eaten_triangle * triangle_pieces,
      left_square := square_pieces - eaten_square,
      left_triangle := triangle_pieces - eaten_triangle,
      remaining_weight_square := left_square * weight_square_piece,
      remaining_weight_triangle := left_triangle * weight_triangle_piece,
      total_remaining_weight := remaining_weight_square + remaining_weight_triangle,
      weight_taken_family := percentage_taken_family * total_remaining_weight,
      remaining_after_family := total_remaining_weight - weight_taken_family,
      weight_taken_friends := percentage_taken_friends * remaining_after_family,
      final_remaining_weight := remaining_after_family - weight_taken_friends,
      weight_per_sister := final_remaining_weight / number_of_sisters
  in weight_per_sister = 448 :=
begin
  intros,
  sorry
end

end each_sister_received_cake_l89_89477


namespace tan_315_eq_neg_one_l89_89289

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89289


namespace olympiad_scores_greater_than_18_l89_89593

open Classical

theorem olympiad_scores_greater_than_18 (n : ℕ) (a : ℕ → ℕ) (h_distinct: ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_ordered: ∀ i j: ℕ, i < j → a i < a j)
  (h_condition: ∀ i j k: ℕ, i ≠ j → i ≠ k → j ≠ k → a i < a j + a k) :
  ∀ i < n, n = 20 ∧ a i > 18 :=
by
  assume i h_i_lt_n h_n_eq_20
  sorry

end olympiad_scores_greater_than_18_l89_89593


namespace Kataleya_paid_total_l89_89850

/--
A store offers the following variable discounts for different types of fruits:
- A $2 discount for every $10 purchase of peaches.
- A $3 discount for every $15 purchase of apples.
- A $1.5 discount for every $7 purchase of oranges.

Kataleya goes to the store and buys:
- 400 peaches at $0.40 each.
- 150 apples at $0.60 each.
- 200 oranges at $0.50 each.

Prove that the total amount of money she paid for the fruits after applying all discounts is $279.
-/
theorem Kataleya_paid_total : 
  let cost_peach := 0.4 in
  let cost_apple := 0.6 in
  let cost_orange := 0.5 in
  let total_cost_peach := 400 * cost_peach in
  let total_cost_apple := 150 * cost_apple in
  let total_cost_orange := 200 * cost_orange in
  let discount_peach := (total_cost_peach / 10).toNat * 2 in
  let discount_apple := (total_cost_apple / 15).toNat * 3 in
  let discount_orange := (total_cost_orange / 7).toNat * 1.5 in
  let paid_peach := total_cost_peach - discount_peach in
  let paid_apple := total_cost_apple - discount_apple in
  let paid_orange := total_cost_orange - discount_orange in
  paid_peach + paid_apple + paid_orange = 279 :=
by
  sorry

end Kataleya_paid_total_l89_89850


namespace total_journey_distance_l89_89805

theorem total_journey_distance
  (T : ℝ) (D : ℝ)
  (h1 : T = 20)
  (h2 : (D / 2) / 21 + (D / 2) / 24 = 20) :
  D = 448 :=
by
  sorry

end total_journey_distance_l89_89805


namespace distinct_four_digit_numbers_count_l89_89200

theorem distinct_four_digit_numbers_count :
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ i j, i ≠ j → n.digits 10 i ≠ n.digits 10 j)}.card = 4536 :=
by
  sorry

end distinct_four_digit_numbers_count_l89_89200


namespace triangle_area_l89_89142

theorem triangle_area :
  let a := 4
  let c := 5
  let b := Real.sqrt (c^2 - a^2)
  (1 / 2) * a * b = 6 :=
by sorry

end triangle_area_l89_89142


namespace parabola_focus_y²__l89_89881

def parabola_focus (p : ℝ) : ℝ × ℝ :=
  (-p, 0)

theorem parabola_focus_y²_=-8x : parabola_focus 2 = (-2, 0) := 
  sorry

end parabola_focus_y²__l89_89881


namespace exists_common_point_in_square_l89_89010

open Set

variable (A : ℕ → Set (ℝ × ℝ))
variable (S : ℕ → ℝ)

theorem exists_common_point_in_square (hA : ∀ i : ℕ, i < 100 → measurable_set (A i))
    (hS : ∑ i in Finset.range 100, S i > 99) :
    ∃ (p : ℝ × ℝ) (H : p ∈ Icc (0:ℝ) 1 × Icc (0:ℝ) 1), ∀ i < 100, p ∈ A i := 
begin 
  sorry 
end

end exists_common_point_in_square_l89_89010


namespace maximize_revenue_l89_89190

noncomputable def revenue (p : ℝ) : ℝ := 100 * p - 4 * p^2

theorem maximize_revenue : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 20 ∧ (∀ q : ℝ, 0 ≤ q ∧ q ≤ 20 → revenue q ≤ revenue p) ∧ p = 12.5 := by
  sorry

end maximize_revenue_l89_89190


namespace find_y_for_negative_x_l89_89542

-- Given conditions
variables (k : ℝ) (y : ℝ) (x : ℝ)
axiom direct_variation : y = k * x
axiom specific_condition : y = 10 ∧ x = 2.5
axiom negative_x : x = -5

-- Problem statement
theorem find_y_for_negative_x : y = -20 :=
by
  have k_value : k = 4 := sorry
  have specific_y_value : y = 4 * (-5) := sorry
  exact specific_y_value

end find_y_for_negative_x_l89_89542


namespace minimum_adults_attending_l89_89206

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem minimum_adults_attending : 
  ∃ n : ℕ, n > 0 ∧ n % 17 = 0 ∧ n % 15 = 0 ∧ n = 255 :=
by
  use 255
  split
  · exact Nat.zero_lt_succ 254
  split
  · rw Nat.mod_eq_zero_of_dvd (Nat.dvd_of_mod_eq_zero rfl)
  split
  · rw Nat.mod_eq_zero_of_dvd (Nat.dvd_of_mod_eq_zero rfl)
  · rfl
  sorry

end minimum_adults_attending_l89_89206


namespace proof_problem_l89_89034

variable {a b c A B C : ℝ}
variable {ABC : Type} [IsTriangle ABC a b c A B C]

axiom law_of_cosines : ∀ {a b c A}, a^2 = b^2 + c^2 - 2 * b * c * Real.cos A
axiom sin_cos_relation : ∀ {C B}, Real.sin C = 2 * Real.cos B

theorem proof_problem (h1 : a^2 = b^2 + c^2 - √3 * b * c)
  (h2 : Real.sin C = 2 * Real.cos B): b = √3 * a ∧ c = 2 * a :=
by
  sorry

end proof_problem_l89_89034


namespace tan_315_eq_neg_1_l89_89320

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89320


namespace find_courtyard_width_l89_89039

-- Definitions based on conditions
def courtyard_length : ℝ := 10
def tiles_per_sqft : ℝ := 4
def percent_green : ℝ := 0.4
def cost_green_per_tile : ℝ := 3
def cost_red_per_tile : ℝ := 1.5
def total_cost : ℝ := 2100

-- Theorem stating the width of the courtyard
theorem find_courtyard_width (w : ℝ) :
  let area := courtyard_length * w,
      num_tiles := tiles_per_sqft * area,
      num_green_tiles := percent_green * num_tiles,
      num_red_tiles := num_tiles - num_green_tiles,
      cost_green := num_green_tiles * cost_green_per_tile,
      cost_red := num_red_tiles * cost_red_per_tile,
      cost_total := cost_green + cost_red
  in cost_total = total_cost → w = 25 :=
by
  intros
  sorry

end find_courtyard_width_l89_89039


namespace remainder_123456789012_div_252_l89_89909

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l89_89909


namespace tan_315_eq_neg1_l89_89353

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89353


namespace B_pow_3_v_l89_89053

-- Definitions based on the conditions in the problem
def B : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]
def v : Fin 2 → ℝ := ![3, -1]
def v_transformed : Fin 2 → ℝ := ![6, -2]

-- The given condition 
axiom B_v_eq_v_transformed : B.mul_vec v = v_transformed
  
-- The theorem to prove
theorem B_pow_3_v :
  (B ^ 3).mul_vec v = ![24, -8] :=
  sorry

end B_pow_3_v_l89_89053


namespace tan_315_eq_neg_one_l89_89288

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89288


namespace chess_game_probability_l89_89435

theorem chess_game_probability (P_Draw P_A_wins : ℚ) (h1 : P_Draw = 1 / 2) (h2 : P_A_wins = 1 / 3) : 
  P_A_wins + P_Draw = 5 / 6 :=
by
  rw [h1, h2]
  norm_num
  sorry

end chess_game_probability_l89_89435


namespace tan_315_eq_neg1_l89_89344

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89344


namespace precious_mistakes_is_18_l89_89073

noncomputable def number_of_mistakes_precious : ℕ :=
  let total_items : ℕ := 75
  let lyssa_incorrect_rate : ℝ := 0.20
  let lyssa_mistakes : ℕ := (lyssa_incorrect_rate * total_items).toNat
  let lyssa_correct : ℕ := total_items - lyssa_mistakes
  let precious_correct := lyssa_correct - 3
  total_items - precious_correct

theorem precious_mistakes_is_18 : number_of_mistakes_precious = 18 := by
  sorry

end precious_mistakes_is_18_l89_89073


namespace truncated_cone_base_area_l89_89078

-- Definitions for given conditions
def cone (r: ℝ) := r -- representing the radius of the base of cones

def truncatedCone (r: ℝ) := r -- representing the radius of the smaller base of truncated cone

-- Given the radii of three cones
def r1 : ℝ := 10
def r2 : ℝ := 15
def r3 : ℝ := 15

-- Radius of the smaller base to be found
def smaller_base_radius : ℝ := 2

-- Target area of the smaller base (correct answer)
def smaller_base_area (r: ℝ) : ℝ := π * r^2

-- Proof statement
theorem truncated_cone_base_area:
  smaller_base_area smaller_base_radius = 4 * π := by
  -- This is where the proof would go
  sorry

end truncated_cone_base_area_l89_89078


namespace remainder_of_large_number_l89_89952

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l89_89952


namespace prime_factors_count_l89_89958

theorem prime_factors_count :
  let expression := (2^9) * (3^5) * (5^7) * (7^4) * (11^6) * (13^3) * (17^5) * (19^2)
  ∑ e in [9, 5, 7, 4, 6, 3, 5, 2], e = 41 :=
by
  let expression := (2^9) * (3^5) * (5^7) * (7^4) * (11^6) * (13^3) * (17^5) * (19^2)
  have h : ∑ e in [9, 5, 7, 4, 6, 3, 5, 2], e = 41 := by rfl
  exact h

end prime_factors_count_l89_89958


namespace scores_greater_than_18_l89_89588

noncomputable def olympiad_scores (scores : Fin 20 → ℕ) :=
∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k

theorem scores_greater_than_18 (scores : Fin 20 → ℕ) (h1 : ∀ i j, i < j → scores i < scores j)
  (h2 : olympiad_scores scores) : ∀ i, 18 < scores i :=
by
  intro i
  sorry

end scores_greater_than_18_l89_89588


namespace angle_eq_l89_89872

theorem angle_eq (C1 C2 C : Type) (O1 O2 O A B K L D : Type)
  (h_tangent_C1_C2_at_lambda : tangent_at C1 C2 λ)
  (h_touches_C_at_A : touches_at C O C1 A)
  (h_touches_C_at_B : touches_at C O C2 B)
  (h_O1_O2_inside_C : inside O1 C ∧ inside O2 C)
  (h_tangent_intersects_C_at_KL : tangent_intersects C1 C2 λ C K L)
  (h_D_midpoint_KL : is_midpoint D K L) :
  angle_eq (O1 O O2) (A D B) := sorry

end angle_eq_l89_89872


namespace tickets_second_half_l89_89100

theorem tickets_second_half
  (T : ℕ) (F : ℕ) (S : ℕ)
  (hT : T = 9570)
  (hF : F = 3867)
  (hS : S = 5703) :
  S = T - F :=
by
  rw [hT, hF, hS]
  norm_num
  sorry

end tickets_second_half_l89_89100


namespace tan_315_eq_neg_1_l89_89323

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89323


namespace sum_diff_l89_89013

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

noncomputable def arithmetic_sequence (n : ℕ) : Prop :=
  ∀ n, n ≥ 2 → a (n + 1) - a n ^ 2 + a (n - 1) = 0

noncomputable def sum_of_first_n_terms (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

theorem sum_diff (n : ℕ) (hn : n ≥ 2) (has : arithmetic_sequence n) :
  S (2 * n - 1) - 4 * n = -2 :=
sorry

end sum_diff_l89_89013


namespace tan_315_eq_neg1_l89_89383

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89383


namespace math_problem_l89_89486

theorem math_problem (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a*b + b*c + c*a = 1) :
  (Real.cbrt (1/a + 6*b) + Real.cbrt (1/b + 6*c) + Real.cbrt (1/c + 6*a)) ≤ 1/(a*b*c) := 
sorry

end math_problem_l89_89486


namespace max_surface_area_of_cut_l89_89488

noncomputable def max_sum_surface_areas (l w h : ℝ) : ℝ :=
  if l = 5 ∧ w = 4 ∧ h = 3 then 144 else 0

theorem max_surface_area_of_cut (l w h : ℝ) (h_l : l = 5) (h_w : w = 4) (h_h : h = 3) : 
  max_sum_surface_areas l w h = 144 :=
by 
  rw [max_sum_surface_areas, if_pos]
  exact ⟨h_l, h_w, h_h⟩

end max_surface_area_of_cut_l89_89488


namespace f_5_eq_9_l89_89058

def f : ℤ → ℤ
| x := if x ≥ 10 then x - 2 else f (x + 6)

theorem f_5_eq_9 : f 5 = 9 :=
by
  -- The proofs are omitted as per the given instructions.
  -- But we need to carry out the steps based on the given solution:
  -- f(5) = f(5 + 6)
  --      = f(11)
  --      = 11 - 2
  --      = 9
  sorry

end f_5_eq_9_l89_89058


namespace tan_315_eq_neg1_l89_89395

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89395


namespace unique_positive_integer_satisfies_eq_l89_89874

theorem unique_positive_integer_satisfies_eq (n : ℕ) (h : 3 * 2^3 + ∑ i in finset.range (n + 1), (i + 4) * 2^(i + 4) = 2^(n + 12)) : 
  n = 1023 := 
sorry

end unique_positive_integer_satisfies_eq_l89_89874


namespace right_triangle_ratio_l89_89583

theorem right_triangle_ratio (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : a^2 + b^2 = c^2) (r s : ℝ) (h3 : r = a^2 / c) (h4 : s = b^2 / c) : 
  r / s = 9 / 16 := by
 sorry

end right_triangle_ratio_l89_89583


namespace tangent_315_deg_l89_89278

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89278


namespace vectors_parallel_l89_89537

theorem vectors_parallel (x : ℝ) (ha : (1, x)) (hb : (x, 9)) (h : (1 * 9 - x * x = 0)) : x = 3 ∨ x = -3 :=
by sorry

end vectors_parallel_l89_89537


namespace symmetric_ring_possible_values_of_n_l89_89192

theorem symmetric_ring_possible_values_of_n (m n : ℕ) (h : (m - 2) * (n - 6) = 12) (hm : m > 2) :
  n ∈ {7, 8, 9, 10, 12, 18} :=
by {
  sorry
}

end symmetric_ring_possible_values_of_n_l89_89192


namespace triangle_area_l89_89880

open Real

-- Define the vertices
def A : ℝ × ℝ × ℝ := (1, 8, 11)
def B : ℝ × ℝ × ℝ := (-2, 5, 7)
def C : ℝ × ℝ × ℝ := (-5, 8, 7)

-- Helper functions to compute vector operations
def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

-- Define the area of triangle
def area (a b c : ℝ × ℝ × ℝ) : ℝ :=
  (1/2) * magnitude (cross_product (vector_sub b a) (vector_sub c a))

-- The theorem to be proved
theorem triangle_area : area A B C = 3 * sqrt 34 := by
  sorry

end triangle_area_l89_89880


namespace monkey_climb_height_l89_89839

theorem monkey_climb_height (x : ℕ) : 
  tree_height = 17 →
  hours = 15 →
  slip_back = 2 →
  (∀ t < hours, net_climb t = x - slip_back) →
  (net_climb hours = x) →
  (14 * (x - 2) + x = tree_height) →
  x = 3 :=
by
  intros h_tree_height h_hours h_slip_back h_net_climb h_last_hour_climb h_equation
  sorry

end monkey_climb_height_l89_89839


namespace total_tv_show_cost_correct_l89_89820

noncomputable def total_cost_of_tv_show : ℕ :=
  let cost_per_episode_first_season := 100000
  let episodes_first_season := 12
  let episodes_seasons_2_to_4 := 18
  let cost_per_episode_other_seasons := 2 * cost_per_episode_first_season
  let episodes_last_season := 24
  let number_of_other_seasons := 4
  let total_cost_first_season := episodes_first_season * cost_per_episode_first_season
  let total_cost_other_seasons := (episodes_seasons_2_to_4 * 3 + episodes_last_season) * cost_per_episode_other_seasons
  total_cost_first_season + total_cost_other_seasons

theorem total_tv_show_cost_correct : total_cost_of_tv_show = 16800000 := by
  sorry

end total_tv_show_cost_correct_l89_89820


namespace solve_for_x_l89_89722

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2 * x - 25) : x = -20 :=
by
  sorry

end solve_for_x_l89_89722


namespace absolute_operation_statements_correctness_l89_89463

noncomputable def abs_op (xs : List ℤ) : ℤ :=
  List.sum (List.map (λ p => Int.natAbs (p.1 - p.2)) (List.diag xs))

theorem absolute_operation_statements_correctness :
  abs_op [1, 3, 5, 10] = 29 ∧
  (∀ x : ℤ, abs_op [x, -2, 5] ≥ 14) ∧
  (∃ (a b c : ℤ), abs_op [a, b, b, c] = 6) →
  1 = 1 := sorry

end absolute_operation_statements_correctness_l89_89463


namespace _l89_89918

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l89_89918


namespace cos_div_sin_sum_pow_lt_two_l89_89498

/-- Given acute angles α and β and a real number x such that
x * (α + β - π/2) > 0,
prove that (cos α / sin β) ^ x + (cos β / sin α) ^ x < 2. -/
theorem cos_div_sin_sum_pow_lt_two (α β x : ℝ) 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hx : x * (α + β - π / 2) > 0) :
  (cos α / sin β) ^ x + (cos β / sin α) ^ x < 2 := 
sorry

end cos_div_sin_sum_pow_lt_two_l89_89498


namespace tan_315_eq_neg1_l89_89354

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89354


namespace tan_315_eq_neg1_l89_89378

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89378


namespace find_150th_term_in_sequence_l89_89893

def is_power_of_three (n : ℕ) : Prop := ∃ (k : ℕ), n = 3 ^ k
def is_sum_of_distinct_powers_of_three (n : ℕ) : Prop := 
  ∃ (l : list ℕ), (∀ x ∈ l, is_power_of_three x) ∧ (l.nodup) ∧ (n = l.sum)

def in_sequence (n : ℕ) : Prop := is_power_of_three n ∨ is_sum_of_distinct_powers_of_three n

def sequence := { n : ℕ | in_sequence n }

theorem find_150th_term_in_sequence :
  sequence.to_list.nth 149 = some 2280 :=
by
  sorry

end find_150th_term_in_sequence_l89_89893


namespace sqrt_div_l89_89797

theorem sqrt_div (a b : ℝ) (ha : a = 6) (hb : b = 2) : (real.sqrt a / real.sqrt b = real.sqrt 3) :=
by sorry

end sqrt_div_l89_89797


namespace convert_to_polar_l89_89878

-- Define the point in rectangular coordinates
def point_rect := (2, -1 : ℝ × ℝ)

-- Proving that converting the given rectangular coordinates to polar coordinates results in the correct polar coordinates
theorem convert_to_polar : 
  let r := Real.sqrt(5)
  let θ := 2 * Real.pi - Real.arctan (1 / 2)
  (r, θ) = (Real.sqrt (2^2 + (-1)^2), 2 * Real.pi - Real.arctan ((-1) / 2)) :=
by
  let r := Real.sqrt(5)
  let θ := 2 * Real.pi - Real.arctan (1 / 2)
  calc (r, θ) 
      = (√5, 2 * π - Real.arctan ((-1) / 2)) : by sorry

end convert_to_polar_l89_89878


namespace concyclic_points_l89_89571

/- Definitions for triangle vertices and sides -/
variables (A B C D E K L I : Type)
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

/- Hypotheses based on the given conditions -/
axiom (h1 : a + c = 3 * b)

/- Incircle touches sides AB and BC at points D and E -/
axiom (incircle_touches : ∀ (A B C : Type), touches_incircle_at A B C D E)

/- Points K and L are reflections of D and E over the incenter I -/
axiom (reflection_K : reflection_point_over_incenter D I = K)
axiom (reflection_L : reflection_point_over_incenter E I = L)

/- The main theorem to be proved -/
theorem concyclic_points : concyclic A C K L :=
  sorry

end concyclic_points_l89_89571


namespace mountain_elevation_l89_89128

open Real

theorem mountain_elevation
  (T_foot : ℝ)
  (T_summit : ℝ)
  (dT_per_100m : ℝ)
  (Delta_T : T_foot - T_summit)
  (increments : Delta_T / dT_per_100m)
  (elevation_per_increment : ℝ)
  (total_elevation : elevation_per_increment * increments) :
  T_foot = 26 →
  T_summit = 14.1 →
  dT_per_100m = 0.7 →
  elevation_per_increment = 100 →
  total_elevation = 1700 :=
by
  intros h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  subst h4
  sorry

end mountain_elevation_l89_89128


namespace max_real_part_of_sum_w_j_l89_89661

noncomputable def maxRealPartSum (z : ℂ) (i : ℂ) : ℝ :=
  ∑ j in finset.range 12,
    if j < 4 ∨ j >= 10 then (8 : ℝ) * real.cos ((j * real.pi) / 6)
    else (8 : ℝ) * real.cos (((j + 3) % 12 * real.pi) / 6)

theorem max_real_part_of_sum_w_j :
  let z1 := 8 * complex.exp (0 * complex.I * π / 6)
  let z2 := 8 * complex.exp (1 * complex.I * π / 6)
  let z3 := 8 * complex.exp (2 * complex.I * π / 6)
  let z4 := 8 * complex.exp (3 * complex.I * π / 6)
  let z5 := 8 * complex.exp (4 * complex.I * π / 6)
  let z6 := 8 * complex.exp (5 * complex.I * π / 6)
  let z7 := 8 * complex.exp (6 * complex.I * π / 6)
  let z8 := 8 * complex.exp (7 * complex.I * π / 6)
  let z9 := 8 * complex.exp (8 * complex.I * π / 6)
  let z10 := 8 * complex.exp (9 * complex.I * π / 6)
  let z11 := 8 * complex.exp (10 * complex.I * π / 6)
  let z12 := 8 * complex.exp (11 * complex.I * π / 6)
  -- 16 + 768 = 784
  maxRealPartSum z i = (16 + 16 * real.sqrt 3) := by
  sorry

end max_real_part_of_sum_w_j_l89_89661


namespace fencing_cost_l89_89812

theorem fencing_cost (x : ℕ) (h : 3 * 4 * x = 9408) : 
  let length := 4 * x,
      width := 3 * x,
      perimeter := 2 * (length + width),
      cost_per_meter := 0.25,
      total_cost := perimeter * cost_per_meter
  in total_cost = 98 := by
  sorry

end fencing_cost_l89_89812


namespace plywood_cut_difference_l89_89818

theorem plywood_cut_difference :
  let l := 9
  let w := 3
  let num_pieces := 3
  let perimeter1 := 2 * (l / num_pieces + w) -- Cut parallel to the longer side
  let perimeter2 := 2 * (l + w / num_pieces) -- Cut parallel to the shorter side
  let max_perimeter := max perimeter1 perimeter2
  let min_perimeter := min perimeter1 perimeter2
  max_perimeter - min_perimeter = 12 :=
by
  unfold l w num_pieces perimeter1 perimeter2 max_perimeter min_perimeter
  -- Perform the necessary calculations or assertions here, then use sorry to skip the proof
  sorry

end plywood_cut_difference_l89_89818


namespace volleyball_tournament_ranking_l89_89012

-- Definitions of the teams and matches
inductive Team
| A | B | C | D | E | F

inductive Match_Result
| Win | Loss

-- Function that represents matches
def match (team1 team2 : Team) : Match_Result → Match_Result := sorry

-- Function to determine the possible rankings from the given rules
noncomputable def possible_rankings : ℕ := 
  let saturday_matches := 2^3 in   -- 8 possible combinations from Saturday's matches
  let sunday_permutations := 6 * 6 in  -- 36 permutations from Sunday's round-robins
  saturday_matches * sunday_permutations

theorem volleyball_tournament_ranking : possible_rankings = 288 :=
by
  sorry

end volleyball_tournament_ranking_l89_89012


namespace sequence_correct_l89_89124

-- First, define the sequence a_n according to the problem's initial conditions and recurrence relations
noncomputable def a_seq : ℕ → ℝ
| 0 := 2
| 1 := 3
| (2*m+1) := a_seq (2*m) + a_seq (2*m - 1)
| (2*m) := a_seq (2*m - 1) + 2 * a_seq (2*m - 2)

-- Define the closed forms for odd and even indexed terms
noncomputable def a_n_odd (n : ℕ) : ℝ := (4 + Real.sqrt 2) / 4 * (2 + Real.sqrt 2)^(n - 1) + 4 * Real.sqrt 2 / 4 * (2 - Real.sqrt 2)^(n - 1)
noncomputable def a_n_even (n : ℕ) : ℝ := (2 * Real.sqrt 2 + 1) / 4 * (2 + Real.sqrt 2)^n - (2 * Real.sqrt 2 - 1) / 4 * (2 - Real.sqrt 2)^n

-- Prove that a_seq equals the closed forms for odd and even indexed terms
theorem sequence_correct (n : ℕ) : 
  (n % 2 = 1 → a_seq n = a_n_odd n) ∧ (n % 2 = 0 → a_seq n = a_n_even n) :=
by
  sorry -- proof omitted

end sequence_correct_l89_89124


namespace at_least_one_woman_probability_l89_89550

noncomputable def probability_at_least_one_woman_selected 
  (total_men : ℕ) (total_women : ℕ) (selected_people : ℕ) : ℚ :=
  1 - (8 / 12 * 7 / 11 * 6 / 10 * 5 / 9)

theorem at_least_one_woman_probability :
  probability_at_least_one_woman_selected 8 4 4 = 85 / 99 := 
sorry

end at_least_one_woman_probability_l89_89550


namespace tan_315_eq_neg1_l89_89243

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89243


namespace no_carry_consecutive_pairs_l89_89470

/-- Consider the range of integers {2000, 2001, ..., 3000}. 
    We determine that the number of pairs of consecutive integers in this range such that their addition requires no carrying is 729. -/
theorem no_carry_consecutive_pairs : 
  ∀ (n : ℕ), (2000 ≤ n ∧ n < 3000) ∧ ((n + 1) ≤ 3000) → 
  ∃ (count : ℕ), count = 729 := 
sorry

end no_carry_consecutive_pairs_l89_89470


namespace point_count_on_line_2x_plus_5_max_b_for_line_5x3x_plus_b_max_n_for_no_lattice_on_line_mx_plus_1_l89_89753

-- Part (a)
theorem point_count_on_line_2x_plus_5 :
  let A := {p : ℝ × ℝ | 0.5 < p.1 ∧ p.1 < 99.5 ∧ 0.5 < p.2 ∧ p.2 < 99.5}
  in
  let lattice_points := {p : ℝ × ℝ | p.1 = r ∧ p.2 = s ∧ r ∈ ℤ ∧ s ∈ ℤ}
  in
  let line := {p : ℝ × ℝ | p.2 = 2 * p.1 + 5}
  in
  finset.card ({p ∈ A ∩ lattice_points ∩ line | p}) = 47 := sorry

-- Part (b)
theorem max_b_for_line_5x3x_plus_b :
  ∃ b : ℤ,
  let A := {p : ℝ × ℝ | 0.5 < p.1 ∧ p.1 < 99.5 ∧ 0.5 < p.2 ∧ p.2 < 99.5}
  in
  let lattice_points := {p : ℝ × ℝ | p.1 = r ∧ p.2 = s ∧ r ∈ ℤ ∧ s ∈ ℤ}
  in
  let line := {p : ℝ × ℝ | p.2 = 5/3 * p.1 + b}
  in
  finset.card ({p ∈ A ∩ lattice_points ∩ line | p}) ≥ 15 ∧
  ∀ b', finset.card ({p ∈ A ∩ lattice_points ∩ line | p}) ≥ 15 → b ≤ 24 := sorry

-- Part (c)
theorem max_n_for_no_lattice_on_line_mx_plus_1 :
  let A := {p : ℝ × ℝ | 0.5 < p.1 ∧ p.1 < 99.5 ∧ 0.5 < p.2 ∧ p.2 < 99.5}
  in
  let lattice_points := {p : ℝ × ℝ | p.1 = r ∧ p.2 = s ∧ r ∈ ℤ ∧ s ∈ ℤ}
  in
  ∃ n : ℝ,
  (∀ m : ℝ, (2/7 < m ∧ m < n) →
  let line := {p : ℝ × ℝ | p.2 = m * p.1 + 1}
  in
  ∀ p ∈ (A ∩ lattice_points ∩ line), false) ∧ n = 27/94 := sorry

end point_count_on_line_2x_plus_5_max_b_for_line_5x3x_plus_b_max_n_for_no_lattice_on_line_mx_plus_1_l89_89753


namespace remainder_18_l89_89156

theorem remainder_18 (x : ℤ) (k : ℤ) (h : x = 62 * k + 7) :
  (x + 11) % 31 = 18 :=
by
  sorry

end remainder_18_l89_89156


namespace cos_sin_215_deg_pow_36_eq_l89_89415

theorem cos_sin_215_deg_pow_36_eq :
  (complex.cos 215 + complex.sin 215 * complex.i)^36 = 
  complex.cos 300 + complex.sin 300 * complex.i :=
by
  sorry

end cos_sin_215_deg_pow_36_eq_l89_89415


namespace quadratic_real_roots_range_l89_89964

theorem quadratic_real_roots_range (m : ℝ) :
  ∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0 ↔ m ≤ 4 ∧ m ≠ 3 := 
by
  sorry

end quadratic_real_roots_range_l89_89964


namespace select_at_least_one_woman_probability_l89_89544

theorem select_at_least_one_woman_probability (men women total selected : ℕ) (h_men : men = 8) (h_women : women = 4) (h_total : total = men + women) (h_selected : selected = 4) :
  let total_prob := 1
  let prob_all_men := (men.to_rat / total) * ((men - 1).to_rat / (total - 1)) * ((men - 2).to_rat / (total - 2)) * ((men - 3).to_rat / (total - 3))
  let prob_at_least_one_woman := total_prob - prob_all_men
  prob_at_least_one_woman = 85 / 99 := by
  sorry

end select_at_least_one_woman_probability_l89_89544


namespace solution_in_quadrant_I_l89_89672

theorem solution_in_quadrant_I (k : ℝ) :
  ∃ x y : ℝ, (2 * x - y = 5 ∧ k * x + 2 * y = 4 ∧ x > 0 ∧ y > 0) ↔ -4 < k ∧ k < 8 / 5 :=
by
  sorry

end solution_in_quadrant_I_l89_89672


namespace expression_value_l89_89960

theorem expression_value (a b : ℕ) (h₁ : a = 2023) (h₂ : b = 2020) :
  ((
     (3 / (a - b) + (3 * a) / (a^3 - b^3) * ((a^2 + a * b + b^2) / (a + b))) * ((2 * a + b) / (a^2 + 2 * a * b + b^2))
  ) * (3 / (a + b))) = 3 :=
by
  -- Use the provided conditions
  rw [h₁, h₂]
  -- Execute the following steps as per the mathematical solution steps 
  sorry

end expression_value_l89_89960


namespace tan_315_eq_neg1_l89_89249

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89249


namespace professor_k_jokes_lectures_l89_89706

theorem professor_k_jokes_lectures (jokes : Finset ℕ) (h_card : jokes.card = 8) :
  let ways_to_choose_3 := jokes.card * (jokes.card - 1) * (jokes.card - 2) / 6
  let ways_to_choose_2 := jokes.card * (jokes.card - 1) / 2
in ways_to_choose_3 + ways_to_choose_2 = 84 :=
by sorry


end professor_k_jokes_lectures_l89_89706


namespace complement_angle1_l89_89480

def deg : Type := ℕ  -- Representing degrees using natural numbers (0 to 89)
def min : Type := ℕ  -- Representing minutes using natural numbers (0 to 59)

-- Define angles as a pair of degrees and minutes
structure angle :=
(degrees : deg)
(minutes : min)

-- Given angle 1 as 38 degrees and 15 minutes
def angle1 : angle := { degrees := 38, minutes := 15 }

-- Define the complement function
def complement (a : angle) : angle :=
if a.degrees < 90 ∧ a.minutes < 60 then
  { degrees := 89 - a.degrees, minutes := 60 - a.minutes }
else
  a  -- if the angle is out of bounds, return the original angle (this case shouldn't occur for the given problem)

-- Prove the complement of angle1 is 51 degrees and 45 minutes
theorem complement_angle1 : complement angle1 = { degrees := 51, minutes := 45 } :=
by sorry

end complement_angle1_l89_89480


namespace trigonometry_solution_l89_89973

noncomputable def solve_trig_problem : Prop :=
  ∀ (α : ℝ), 
    (sin (π / 4 + α) * sin (π / 4 - α) = 1 / 6) → 
    (π / 2 < α) ∧ (α < π) → 
    tan (4 * α) = 4 * sqrt 2 / 7

theorem trigonometry_solution : solve_trig_problem :=
  sorry

end trigonometry_solution_l89_89973


namespace tangent_315_deg_l89_89286

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89286


namespace second_product_of_digits_98_eq_14_l89_89132

def first_product_of_digits (n : ℕ) : ℕ :=
  let digit1 := n / 10
  let digit2 := n % 10
  digit1 * digit2

def second_product_of_digits (n : ℕ) : ℕ :=
  let first := first_product_of_digits n
  let digit1 := first / 10
  let digit2 := first % 10
  digit1 * digit2

theorem second_product_of_digits_98_eq_14 :
  second_product_of_digits 98 = 14 :=
by
  have h1 : first_product_of_digits 98 = 72 := by 
    unfold first_product_of_digits 
    norm_num
  have h2 : 72 / 10 = 7 := by norm_num
  have h3 : 72 % 10 = 2 := by norm_num
  unfold second_product_of_digits
  rw [h1, h2, h3]
  norm_num

end second_product_of_digits_98_eq_14_l89_89132


namespace maximum_value_m_l89_89525

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

noncomputable def exists_t_and_max_m (m : ℝ) : Prop :=
  ∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) ≤ x

theorem maximum_value_m : ∃ m : ℝ, exists_t_and_max_m m ∧ (∀ m' : ℝ, exists_t_and_max_m m' → m' ≤ 4) :=
by
  sorry

end maximum_value_m_l89_89525


namespace tan_315_eq_neg_one_l89_89223

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89223


namespace calculate_expression_l89_89997

theorem calculate_expression
  (x : ℝ)
  (h1 : cos (π / 4 + x) = 4 / 5)
  (h2 : x ∈ Ioo (-π / 2) (-π / 4)) :
  (sin (2 * x) - 2 * sin x ^ 2) / (1 + tan x) = 28 / 75 := 
sorry

end calculate_expression_l89_89997


namespace total_production_cost_l89_89822

-- Conditions
def first_season_episodes : ℕ := 12
def remaining_season_factor : ℝ := 1.5
def last_season_episodes : ℕ := 24
def first_season_cost_per_episode : ℝ := 100000
def other_season_cost_per_episode : ℝ := first_season_cost_per_episode * 2

-- Number of seasons
def number_of_seasons : ℕ := 5

-- Question: Calculate the total cost
def total_first_season_cost : ℝ := first_season_episodes * first_season_cost_per_episode
def second_season_episodes : ℕ := (first_season_episodes * remaining_season_factor).toNat
def second_season_cost : ℝ := second_season_episodes * other_season_cost_per_episode
def third_and_fourth_seasons_cost : ℝ := 2 * second_season_cost
def last_season_cost : ℝ := last_season_episodes * other_season_cost_per_episode
def total_cost : ℝ := total_first_season_cost + second_season_cost + third_and_fourth_seasons_cost + last_season_cost

-- Proof
theorem total_production_cost :
  total_cost = 16800000 :=
by
  sorry

end total_production_cost_l89_89822


namespace john_probability_l89_89041

theorem john_probability :
  let p := (1 : ℚ) / 2,
      n := 10,
      prob := ∑ k in finset.range (n + 1), if k >= 6 then nat.choose n k * p^k * (1 - p)^(n-k) else 0
  in prob = 193 / 512 := by
  sorry

end john_probability_l89_89041


namespace clock_820_angle_is_130_degrees_l89_89734

def angle_at_8_20 : ℝ :=
  let degrees_per_hour := 30.0
  let degrees_per_minute_hour_hand := 0.5
  let num_hour_sections := 4.0
  let minutes := 20.0
  let hour_angle := num_hour_sections * degrees_per_hour
  let minute_addition := minutes * degrees_per_minute_hour_hand
  hour_angle + minute_addition

theorem clock_820_angle_is_130_degrees :
  angle_at_8_20 = 130 :=
by
  sorry

end clock_820_angle_is_130_degrees_l89_89734


namespace gcd_1722_966_l89_89143

theorem gcd_1722_966 : Nat.gcd 1722 966 = 42 :=
  sorry

end gcd_1722_966_l89_89143


namespace mark_bottle_cap_further_jenny_l89_89657

def bottle_cap_distance_jenny : ℝ :=
  let d1 := 18
  let d2 := (1 / 3) * d1
  let d3 := 1.2 * d2
  let d4 := (1 / 2) * d3
  let d5 := 1.1 * d4
  d1 + d2 + d3 + d4 + d5

def bottle_cap_distance_mark : ℝ :=
  let d1 := 15
  let d2 := 2 * d1
  let d3 := 1.15 * d2
  let d4 := (3 / 4) * d3
  let d5 := 0.95 * d4
  let d6 := 0.3 * d5
  let d7 := 1.25 * d6
  d1 + d2 + d3 + d4 + d5 + d6 + d7

def bottle_cap_difference : ℝ :=
  bottle_cap_distance_mark - bottle_cap_distance_jenny

theorem mark_bottle_cap_further_jenny : bottle_cap_difference = 107.78959375 :=
  by
  sorry

end mark_bottle_cap_further_jenny_l89_89657


namespace g_neither_even_nor_odd_l89_89035

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 2) + 1/3

theorem g_neither_even_nor_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) :=
by
  -- insert proof here
  sorry

end g_neither_even_nor_odd_l89_89035


namespace find_functional_relationship_find_march_cost_unit_find_april_minimum_profit_l89_89828

-- Definitions
def is_linear_relation (points : list (ℝ × ℝ)) (f : ℝ → ℝ) : Prop :=
  ∀ x y, (x, y) ∈ points → y = f x

-- Given data points
def points : list (ℝ × ℝ) := [(30, 40), (32, 36)]

-- Function found in part 1
def linear_function (x : ℝ) : ℝ := -2 * x + 100

-- March Conditions
def march_selling_price := 35
def march_profit := 450
def march_sales_volume := linear_function march_selling_price 

-- April Conditions
def april_cost_reduction := 14
def april_selling_price := {x : ℝ // 25 ≤ x ∧ x ≤ 30}
def april_profit_function (x : ℝ) : ℝ := (-2 * x + 100) * (x - 6) - 450

-- Proof statement for part 1
theorem find_functional_relationship :
  is_linear_relation points linear_function := 
sorry

-- Proof statement for March cost per unit
theorem find_march_cost_unit (m : ℝ) :
  march_profit = march_sales_volume * (march_selling_price - m) → 
  m = 20 := 
sorry

-- Proof statement for April minimum profit
theorem find_april_minimum_profit : 
  ∃ x : ℝ, (25 ≤ x ∧ x ≤ 30) ∧ april_profit_function x = 500 := 
  sorry

end find_functional_relationship_find_march_cost_unit_find_april_minimum_profit_l89_89828


namespace dining_bill_before_tip_l89_89763

noncomputable def total_bill_before_tip (final_share : ℝ) (num_people : ℕ) (tip_rate : ℝ) : ℝ :=
  let total_with_tip := final_share * ↑num_people
  total_with_tip / (1 + tip_rate)

theorem dining_bill_before_tip (h : total_bill_before_tip 40.44 6 0.15 = 210.99) : 
  total_bill_before_tip 40.44 6 0.15 ≈ 210.99 := 
by 
  sorry

end dining_bill_before_tip_l89_89763


namespace tan_315_eq_neg1_l89_89255

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89255


namespace weighings_required_for_counterfeit_coins_l89_89770

-- Definitions for the problem
def num_weighings_to_identify_counterfeit (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 2
  else if n = 9 then 2
  else 0  -- we're defining for 3, 4 and 9 coins only, other cases return 0

-- Theorem statement for the proof problem
theorem weighings_required_for_counterfeit_coins :
  num_weighings_to_identify_counterfeit 3 = 1 ∧
  num_weighings_to_identify_counterfeit 4 = 2 ∧
  num_weighings_to_identify_counterfeit 9 = 2 :=
by
  split;
  (simp; try split; simp)

end weighings_required_for_counterfeit_coins_l89_89770


namespace tan_315_eq_neg1_l89_89342

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89342


namespace travel_options_A_to_C_l89_89766

def travel_options (trains ferries : Nat) : Nat := trains * ferries

theorem travel_options_A_to_C :
  travel_options 5 4 = 20 :=
by
  -- This is the equivalent of the solution steps 4 and the boxed answer
  simp [travel_options]
  rfl

end travel_options_A_to_C_l89_89766


namespace sufficient_not_necessary_condition_for_f_l89_89169

theorem sufficient_not_necessary_condition_for_f (a : ℝ) :
  (∀ x : ℝ, 1 < x → x - a ≥ 0) ∧ (a = 1) ↔ (a = 1 ∧ ∀ y : ℝ, y > 1 → (differentiable ℝ (λ x, (x - a)^2) → ∃ δ > 0, ∀ x > y, (x - a)^2 > (x + δ - a)^2)) := 
begin
  sorry
end

end sufficient_not_necessary_condition_for_f_l89_89169


namespace tan_315_degrees_l89_89365

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89365


namespace exist_infinitely_many_K_l89_89785

variable (f : ℕ → ℕ)

def is_permutation (f : ℕ → ℕ) : Prop :=
  bijective f

def is_involution (f : ℕ → ℕ) : Prop :=
  ∀ x, f (f x) = x

def bounded_distance (f : ℕ → ℕ) : Prop :=
  ∀ x, abs (f x - x) ≤ 3

def M (f : ℕ → ℕ) (n : ℕ) : ℚ :=
  (1 / (n + 1 : ℚ)) * (∑ j in Finset.range (n + 1), abs (f j - j))

def condition_on_M (f : ℕ → ℕ) : Prop :=
  ∀ n > 42, M f n < 2.011

def P (K : ℕ) : Prop :=
  ∀ x, x ≤ K → f x ≤ K

theorem exist_infinitely_many_K
  (hf1 : is_permutation f)
  (hf2 : is_involution f)
  (hf3 : bounded_distance f)
  (hf4 : condition_on_M f) :
  ∃ᶠ K in at_top, P f K := sorry

end exist_infinitely_many_K_l89_89785


namespace reachable_cells_after_moves_l89_89636

def is_valid_move (n : ℕ) (x y : ℤ) : Prop :=
(abs x ≤ n ∧ abs y ≤ n ∧ (x + y) % 2 = 0)

theorem reachable_cells_after_moves (n : ℕ) :
  n = 10 → ∃ (cells : Finset (ℤ × ℤ)), cells.card = 121 ∧ 
  (∀ (cell : ℤ × ℤ), cell ∈ cells → is_valid_move n cell.1 cell.2) :=
by
  intros h
  use {-10 ≤ x, y, x + y % 2 = 0 & abs x + abs y ≤ n }
  sorry -- proof goes here

end reachable_cells_after_moves_l89_89636


namespace remainder_div_252_l89_89927

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l89_89927


namespace tan_315_eq_neg1_l89_89253

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89253


namespace remainder_when_divided_l89_89938

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l89_89938


namespace tan_315_eq_neg_1_l89_89326

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89326


namespace probability_same_color_l89_89771

theorem probability_same_color (r w : ℕ) (h_r : r = 2) (h_w : w = 2) : (2/4 * 2/4 + 2/4 * 2/4) = 1/2 :=
by
  rw [h_r, h_w]
  norm_num
  sorry

end probability_same_color_l89_89771


namespace volume_tetrahedron_K_L_M_D_l89_89765

theorem volume_tetrahedron_K_L_M_D 
  (V : ℝ) 
  (CD DB AB : ℝ)
  (h1 : 2 * (CD / 2) = CD) 
  (h2 : 3 * (DB / 3) = DB) 
  (h3 : 5 * (2 * AB / 5) = 2 * AB) 
  (Vol_ABCD : Volume (tetrahedron A B C D) = V) :
  Volume (tetrahedron K L M D) = V / 15 :=
sorry

end volume_tetrahedron_K_L_M_D_l89_89765


namespace final_answer_after_division_and_subtraction_l89_89842

theorem final_answer_after_division_and_subtraction : 
  let chosen_number := 740 in
  let result_of_division := chosen_number / 4 in
  let final_answer := result_of_division - 175 in
  final_answer = 10 :=
by
  sorry

end final_answer_after_division_and_subtraction_l89_89842


namespace at_least_one_red_l89_89513

-- Definitions
def prob_red_A : ℚ := 1 / 3
def prob_red_B : ℚ := 1 / 2

-- Main theorem
theorem at_least_one_red : 
  let prob_both_not_red := (1 - prob_red_A) * (1 - prob_red_B)
  in 1 - prob_both_not_red = 2 / 3 := 
by
  let prob_both_not_red := (1 - prob_red_A) * (1 - prob_red_B)
  have h : 1 - prob_both_not_red = 2 / 3 := 
    sorry
  exact h

end at_least_one_red_l89_89513


namespace _l89_89921

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l89_89921


namespace tangent_lines_of_parabola_l89_89491

-- Define the given parabola
def parabola (x y : ℝ) := x^2 = 4 * y

-- Points of intersection P(-2, 1), P(2, 1)
def point1 := (-2 : ℝ, 1 : ℝ)
def point2 := (2 : ℝ, 1 : ℝ)

-- Tangent lines at points P1 and P2
def tangent_line1 (x y : ℝ) := x + y + 1 = 0
def tangent_line2 (x y : ℝ) := x - y - 1 = 0

-- Lean theorem statement
theorem tangent_lines_of_parabola :
  (parabola point1.1 point1.2 → tangent_line1 point1.1 point1.2) ∧ 
  (parabola point2.1 point2.2 → tangent_line2 point2.1 point2.2) :=
by
  sorry

end tangent_lines_of_parabola_l89_89491


namespace AD_div_BC_l89_89028

variable {s t : ℝ}

theorem AD_div_BC (h₁ : AB = s) (h₂ : AC = s) (h₃ : BC = t) (h₄ : BD = t) (h₅ : CD = s) :
  (AD / BC) = (sqrt (4 * s ^ 2 - t ^ 2) + sqrt (4 * t ^ 2 - s ^ 2)) / (2 * t) :=
sorry

end AD_div_BC_l89_89028


namespace tan_315_eq_neg1_l89_89331

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89331


namespace rhombus_equal_segments_l89_89122

theorem rhombus_equal_segments
    (A B C D M N P R : Type)
    [IsRhombus A B C D]
    (hacuteA : ∠A < π/2)
    (hM_on_AC : M ∈ Segment A C)
    (hN_on_BC : N ∈ Segment B C)
    (hDM_eq_MN : |DM| = |MN|)
    (hP : P = Intersection AC DN)
    (hR : R = Intersection AB DM) :
    |RP| = |PD| :=
sorry

end rhombus_equal_segments_l89_89122


namespace tan_315_eq_neg_one_l89_89294

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89294


namespace chord_AB_sqrt3_l89_89175

noncomputable def circleEquation (x y : ℝ) : Prop := x^2 + y^2 = 1

def pointP : ℝ × ℝ := (1, Real.sqrt 3)

def isTangent (C : ℝ × ℝ → Prop) (P A : ℝ × ℝ) : Prop :=
  ∃ v : ℝ × ℝ, C A ∧ (A.1 - P.1) * v.1 + (A.2 - P.2) * v.2 = 0

def chordLength (A B P : ℝ × ℝ) (C : ℝ × ℝ → Prop) : ℝ :=
  if h : isTangent C P A ∧ isTangent C P B
  then Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  else 0

theorem chord_AB_sqrt3 : chordLength (1, 0) (-1, 0) pointP circleEquation = Real.sqrt 3 :=
by
  sorry

end chord_AB_sqrt3_l89_89175


namespace probability_even_in_5_of_7_rolls_is_21_over_128_l89_89786

noncomputable def probability_even_in_5_of_7_rolls : ℚ :=
  let n := 7
  let k := 5
  let p := (1:ℚ) / 2
  let binomial (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  (binomial n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_even_in_5_of_7_rolls_is_21_over_128 :
  probability_even_in_5_of_7_rolls = 21 / 128 :=
by
  sorry

end probability_even_in_5_of_7_rolls_is_21_over_128_l89_89786


namespace inequality_solution_set_l89_89758

theorem inequality_solution_set :
  { x : ℝ | (x + 2) ^ (-3 / 5) < (5 - 2 * x) ^ (-3 / 5) } =
  { x : ℝ | x < -2 } ∪ { x : ℝ | 1 < x ∧ x < 5 / 2 } := by
sorry

end inequality_solution_set_l89_89758


namespace probability_of_point_being_in_circle_l89_89133

def event_count_in_circle (n : ℕ) (m : ℕ) : ℕ :=
  if n * n + m * m ≤ 9 then 1 else 0

def total_possible_outcomes : ℕ :=
  36

def count_inside_circle : ℕ :=
  (List.range 1 7).foldl (λ acc x => (List.range 1 7).foldl (λ acc y => acc + event_count_in_circle x y) acc) 0

def probability_in_circle : ℚ :=
  count_inside_circle / total_possible_outcomes

theorem probability_of_point_being_in_circle :
  probability_in_circle = 1 / 9 :=
sorry

end probability_of_point_being_in_circle_l89_89133


namespace part_a_part_b_l89_89164

-- Define the conditions of a special square.
def is_special_square (sq : List (List ℕ)) : Prop :=
  (∀ r, r < 4 → Multiset.card (Multiset.of_list (sq.nth r).getD []) = 4 ∧
    Multiset.card (Multiset.of_list (List.transpose sq).nth r.getD []) = 4) ∧
  (∀ n, n < 4 → Multiset.card (Multiset.of_list (
    sq.drop (n / 2 * 2)).take 2
      .map (fun l => l.drop (n % 2 * 2)).take 2
      .join)) = 4)

-- Part a: prove that the given special square can be completed.
theorem part_a : ∃ sq, is_special_square [[1, 2, _, _], [3, 4, _, _], [_, _, _, _], [_, _, _, 1]] :=
  sorry

-- Part b: prove that it is not possible to complete a given partially filled square.
theorem part_b : ∀ sq, ¬ is_special_square [[_, 3, _, _], [_, 4, _, 3], [_, _, _, _], [_, 3, _, 4]] :=
  sorry

-- Part c: enumerate all possible completions for a given partially filled square.
noncomputable def part_c : List (List (List ℕ)) :=
  sorry

-- Part d: count the total number of special squares.
noncomputable def part_d : ℕ :=
  288

end part_a_part_b_l89_89164


namespace tan_315_eq_neg1_l89_89350

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89350


namespace remainder_when_divided_l89_89936

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l89_89936


namespace tan_315_eq_neg1_l89_89266

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89266


namespace annie_regaining_control_chance_l89_89205

theorem annie_regaining_control_chance
  (temp_drop : ℕ)
  (skid_increase_per_drop : ℕ)
  (degrees_per_increment : ℕ)
  (serious_accident_chance : ℕ) :
  temp_drop = 24 →
  skid_increase_per_drop = 5 →
  degrees_per_increment = 3 →
  serious_accident_chance = 24 →
  let increments := temp_drop / degrees_per_increment in
  let skid_chance := increments * skid_increase_per_drop in
  0.40 * (1 - R) = 0.24 →
  R = 0.40 :=
by
  intros h1 h2 h3 h4 h_eq
  let increments := 24 / 3
  let skid_chance := increments * 5
  have h_solve : 0.40 * (1 - R) = 0.24, from h_eq
  have h1 : 1 - R = 0.60, by sorry
  have h2 : R = 1 - 0.60, by sorry
  have h3 : R = 0.40, by sorry
  exact h3

end annie_regaining_control_chance_l89_89205


namespace remainder_of_large_number_l89_89950

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l89_89950


namespace infinitely_many_distances_l89_89503

-- Assume there are infinitely many points in the plane
noncomputable def infinite_points (E : Type) [metric_space E] : Prop :=
  ∃ (S : set E), infinite S

-- Define the main theorem statement
theorem infinitely_many_distances (P : Type) [metric_space P] :
  infinite_points P → ∃ (d : set ℝ), infinite d :=
by
  -- We assume the proof of this theorem
  sorry

end infinitely_many_distances_l89_89503


namespace geometric_sequence_common_ratio_l89_89836

theorem geometric_sequence_common_ratio :
  ∃ r, (∀ n, r = -3/2) → ∃ a b c d : ℝ, a = 32 ∧ b = -48 ∧ c = 72 ∧ d = -108 ∧
  r = b / a ∧ r = c / b ∧ r = d / c :=
sorry

end geometric_sequence_common_ratio_l89_89836


namespace pentagon_perimeter_l89_89865

def pentagon_side_lengths : list ℕ := [6, 7, 8, 9, 10]

theorem pentagon_perimeter :
  pentagon_side_lengths.sum = 40 :=
by
  sorry

end pentagon_perimeter_l89_89865


namespace proof_problem_l89_89208

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def smallest_two_digit_prime : ℕ :=
  if h: is_prime 11 then 11 else sorry

noncomputable def largest_one_digit_prime : ℕ :=
  if h: is_prime 7 then 7 else sorry

def smallest_one_digit_prime := 2

def calculate_expression (p1 p2 q : ℕ) : ℕ :=
  p1 * (p2 * p2) - q

theorem proof_problem :
  calculate_expression smallest_two_digit_prime largest_one_digit_prime smallest_one_digit_prime = 537 :=
by
  sorry

end proof_problem_l89_89208


namespace interval_of_monotonic_decrease_l89_89115

noncomputable def decreasing_interval : Set ℝ :=
  {x : ℝ | 0 < x ∧ x < 2}

theorem interval_of_monotonic_decrease (x : ℝ) :
  (0 < x ∧ x < 2) ↔ (∃ y : ℝ, y = log 3 (4 - x^2) ∧ y ∈ decreasing_interval) :=
by
  sorry

end interval_of_monotonic_decrease_l89_89115


namespace cheyenne_profit_l89_89216

def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def uncracked_fraction : ℚ := 3/5
def cost_per_pot : ℚ := 15
def selling_price_per_pot : ℚ := 40

def number_of_uncracked_pots : ℕ := (uncracked_fraction * total_pots : ℚ).toNat
def total_production_cost : ℚ := total_pots * cost_per_pot
def revenue_from_uncracked_pots : ℚ := number_of_uncracked_pots * selling_price_per_pot
def profit : ℚ := revenue_from_uncracked_pots - total_production_cost

theorem cheyenne_profit : profit = 720 := by
  -- The proof goes here
  sorry

end cheyenne_profit_l89_89216


namespace alok_age_l89_89155

theorem alok_age (B A C : ℕ) (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  -- proof would go here
  sorry

end alok_age_l89_89155


namespace remainder_div_252_l89_89928

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l89_89928


namespace tan_315_proof_l89_89309

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89309


namespace circle_geometry_max_diameter_intersection_length_l89_89663

theorem circle_geometry_max_diameter_intersection_length :
  ∃ (x y z : ℕ), (∀ p : ℕ, prime p → ¬ p^2 ∣ z) ∧ (∀ A B : Point, circle A B = 2 → A₀ = 1 ∧ B₀ = 1)  ∧
    (A₀ = B₀ = 1 → d = x - y * sqrt z) ∧ x + y + z = 5 :=
by
  sorry

end circle_geometry_max_diameter_intersection_length_l89_89663


namespace smallest_n_for_distances_l89_89978

theorem smallest_n_for_distances : ∃ (n : ℕ), 
  (∀ (points : list (ℝ × ℝ)), points.length = n → 
   ∃ (d : list ℝ), d = [1, 2, 4, 8, 16, 32] ∧ 
   ∀ (i j : ℕ), 0 ≤ i ∧ i < n → 0 ≤ j ∧ j < n → i ≠ j → 
   (let dist := (points.nth_le i sorry).dist (points.nth_le j sorry) in 
    dist = 1 ∨ dist = 2 ∨ dist = 4 ∨ dist = 8 ∨ dist = 16 ∨ dist = 32)
  ) → n = 7 :=
by sorry

end smallest_n_for_distances_l89_89978


namespace crayons_total_cost_l89_89694

theorem crayons_total_cost :
  let packs_initial := 4
  let packs_to_buy := 2
  let cost_per_pack := 2.5
  let total_packs := packs_initial + packs_to_buy
  let total_cost := total_packs * cost_per_pack
  total_cost = 15 :=
by
  sorry

end crayons_total_cost_l89_89694


namespace at_least_one_woman_selected_probability_l89_89555

-- Define the total number of people, men, and women
def total_people : Nat := 12
def men : Nat := 8
def women : Nat := 4
def selected_people : Nat := 4

-- Define the probability ratio of at least one woman being selected
def probability_at_least_one_woman_selected : ℚ := 85 / 99

-- Prove the probability is correct given the conditions
theorem at_least_one_woman_selected_probability :
  (probability_of_selecting_at_least_one_woman men women selected_people total_people) = probability_at_least_one_woman_selected :=
sorry

end at_least_one_woman_selected_probability_l89_89555


namespace no_non_integer_solutions_l89_89433

theorem no_non_integer_solutions (x y : ℝ) (h1 : ¬ (x ∈ ℤ)) (h2 : ¬ (y ∈ ℤ)) 
    (h3 : ∃ (m n : ℤ), 6 * x + 5 * y = m ∧ 13 * x + 11 * y = n) : false := 
by
  sorry

end no_non_integer_solutions_l89_89433


namespace reachable_cells_after_10_moves_l89_89620

theorem reachable_cells_after_10_moves :
  let board_size := 21
  let central_cell := (11, 11)
  let moves := 10
  (reachable_cells board_size central_cell moves) = 121 :=
by
  sorry

end reachable_cells_after_10_moves_l89_89620


namespace tan_315_degrees_l89_89359

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89359


namespace tan_315_degree_l89_89407

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89407


namespace area_triangle_BDA_eq_zero_l89_89580

theorem area_triangle_BDA_eq_zero
  (O A C B D : Type)
  (r : ℝ)
  [metric_space O]
  [has_dist O ℝ]
  (circ : circle (dist O ℝ) r)
  (midpoint_B: is_midpoint B A C)
  (perpendicular_OB_AC: is_perpendicular O B A C)
  (perpendicular_BD_OA: is_perpendicular B D O A)
  (AC_eq_2r: dist A C = 2 * r)
  (O_eq_center: is_center O circ):

  -- The area of triangle BDA is 0.
    area_triangle B D A = 0 :=
sorry

end area_triangle_BDA_eq_zero_l89_89580


namespace tan_315_degree_l89_89409

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89409


namespace tan_315_eq_neg1_l89_89352

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89352


namespace tan_315_degrees_l89_89369

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89369


namespace triangle_inequality_5_l89_89566

def isTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_inequality_5 :
  ∀ (m : ℝ), isTriangle 3 4 m → (m = 5) :=
begin
  intros m h,
  -- Given options are 1, 5, 7, 9
  by_contradiction,
  -- Since 1 < m < 7, the only valid m is m = 5,
  sorry,
end

end triangle_inequality_5_l89_89566


namespace part1_part2_l89_89495

theorem part1 (x : ℝ) (m : ℝ) :
  (∃ x, x^2 - 2*(m-1)*x + m^2 = 0) → (m ≤ 1 / 2) := 
  sorry

theorem part2 (x1 x2 : ℝ) (m : ℝ) :
  (x1^2 - 2*(m-1)*x1 + m^2 = 0) ∧ (x2^2 - 2*(m-1)*x2 + m^2 = 0) ∧ 
  (x1^2 + x2^2 = 8 - 3*x1*x2) → (m = -2 / 5) := 
  sorry

end part1_part2_l89_89495


namespace median_and_angle_bisector_not_divide_altitude_l89_89214

theorem median_and_angle_bisector_not_divide_altitude
  (A B C H F G K : Type)
  [inhabited A] [inhabited B] [inhabited C] [inhabited H] [inhabited F] [inhabited G] [inhabited K]
  (acute : ∀ {X : Type}, X = A ∨ X = B ∨ X = C → ¬ ∃ (α : ℝ), α = 90)
  (median : ∀ {X Y Z : Type}, X = A ∧ Y = midpoint B C ∧ Z = A → Z ∈ line(A, midpoint(B, C)))
  (angle_bisector : ∀ {X Y Z : Type}, X = A ∧ Y = B ∧ Z = H → bisects ∠BAH at K)
  (altitude : BH is altitude from B to AC)
  (equal_parts : BH = 3 * |BH|.half)
  : false := 
by {
  sorry -- Proof not required as per instructions
}

end median_and_angle_bisector_not_divide_altitude_l89_89214


namespace minimum_value_is_two_l89_89979

def minimum_value_frac (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2 * y = 4 → (2 / x + 1 / y ≥ 2)

theorem minimum_value_is_two (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 4) :
  2 / x + 1 / y = 2 :=
begin
  sorry
end

end minimum_value_is_two_l89_89979


namespace range_of_a_l89_89051

def A := {x : ℝ | x * (4 - x) ≥ 3}
def B (a : ℝ) := {x : ℝ | x > a}

theorem range_of_a (a : ℝ) : (A ∩ B a = A) ↔ (a < 1) := by
  sorry

end range_of_a_l89_89051


namespace tan_315_eq_neg_one_l89_89301

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89301


namespace product_of_four_consecutive_add_one_is_square_specific_case_five_six_seven_eight_l89_89440

-- Definitions based on the given conditions
def is_perfect_square (n : ℕ) : Prop := ∃ (k : ℕ), k * k = n

-- Theorem stating the product of four consecutive natural numbers plus one is a perfect square
theorem product_of_four_consecutive_add_one_is_square (n : ℕ) : 
  is_perfect_square (n * (n + 1) * (n + 2) * (n + 3) + 1) := 
begin
  sorry
end

-- Specific case: Prove 5 * 6 * 7 * 8 + 1 is a perfect square
theorem specific_case_five_six_seven_eight : 
  is_perfect_square (5 * 6 * 7 * 8 + 1) :=
begin
  exact product_of_four_consecutive_add_one_is_square 5,
end

end product_of_four_consecutive_add_one_is_square_specific_case_five_six_seven_eight_l89_89440


namespace parallelogram_angle_sum_l89_89020

theorem parallelogram_angle_sum (ABCD : Type) [Parallelogram ABCD]
  (angle_B angle_D angle_A : ℝ)
  (h_parallelogram : IsParallelogram ABCD)
  (h_angles_sum : angle_B + angle_D = 120)
  (h_opposite_angles : angle_B = angle_D)
  (h_adjacent_sum : angle_A + angle_B = 180) :
  angle_A = 120 :=
sorry

end parallelogram_angle_sum_l89_89020


namespace concurrency_of_lines_l89_89779

-- Define the points and conditions
variables {A B C D E F M X : Type*}
variables [affine_space ℝ (affine.coord_proj ℝ 3)]
variables (hBCF : right_triangle B C F)
variables (hA_on_CF : collinear ℝ ({A, C, F}) ∧ dist A F = dist B F ∧ between A F C)
variables (hD : dist D A = dist D C ∧ bisects_line_segment AC D B ∧ bisects_angle A D C B)
variables (hE : dist E A = dist E D ∧ bisects_angle E A C D)
variables (hM_midpoint : midpoint (C, F) M)
variables (hX_parallel : parallel_lines (segment A M) (segment E X) ∧ parallel_lines (segment A E) (segment M X))

-- The theorem stating that lines BD, FX, and ME are concurrent
theorem concurrency_of_lines :
  concurrent {line_through B D, line_through F X, line_through M E} :=
  sorry

end concurrency_of_lines_l89_89779


namespace range_of_a_l89_89483

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + 2 * (1 - x) * Real.sin (a * x)

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → (2 * x - 1 = f a x ↔ x = 1 ∨ 2 * Real.sin (a * x) = 1)) →
  a ∈ Ioo (5 * Real.pi / 6) (13 * Real.pi / 6) :=
sorry

end range_of_a_l89_89483


namespace y_intercept_l89_89130

theorem y_intercept (x y : ℝ) (h : 2 * x - 3 * y = 6) : x = 0 → y = -2 :=
by
  intro h₁
  sorry

end y_intercept_l89_89130


namespace remainder_123456789012_div_252_l89_89948

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l89_89948


namespace train_length_l89_89853

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_seconds : ℝ := 36
noncomputable def speed_m_s := speed_km_hr * (5/18 : ℝ)
noncomputable def distance := speed_m_s * time_seconds

-- Theorem statement
theorem train_length : distance = 600.12 := by
  sorry

end train_length_l89_89853


namespace tan_315_eq_neg1_l89_89349

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89349


namespace line_through_symmetric_points_on_circle_l89_89023

-- Define the center of the circle and its radius
def circle_center : ℝ × ℝ := (0, 1)
def circle_radius : ℝ := 2

-- Define the points A and B lying on the circle symmetric about the point P
variables {A B : ℝ × ℝ}
def symmetric_point (P : ℝ × ℝ) (Q1 Q2 : ℝ × ℝ) : Prop :=
  2 * P.1 = Q1.1 + Q2.1 ∧ 2 * P.2 = Q1.2 + Q2.2

-- Conditions from the problem
def on_circle (P : ℝ × ℝ) : Prop :=
  (P.1 - circle_center.1)^2 + (P.2 - circle_center.2)^2 = circle_radius

def symmetric_about_P (P A B : ℝ × ℝ) : Prop :=
  symmetric_point P A B

-- Definition of the point P
def point_P : ℝ × ℝ := (1, 2)

-- Definition of line_AB in terms of its equation
def line_AB : ℝ × ℝ → Prop := λ (x y : ℝ), x + y - 3 = 0

-- The main statement to be proved
theorem line_through_symmetric_points_on_circle (A B : ℝ × ℝ)
  (hA : on_circle A) (hB : on_circle B) (hSym : symmetric_about_P point_P A B) :
  ∀ x y, line_AB (x, y) :=
sorry

end line_through_symmetric_points_on_circle_l89_89023


namespace max_k_possible_l89_89197

-- Given the sequence formed by writing all three-digit numbers from 100 to 999 consecutively
def digits_sequence : List Nat := List.join (List.map (fun n => [n / 100, (n / 10) % 10, n % 10]) (List.range' 100 (999 - 100 + 1)))

-- Function to get a k-digit number from the sequence
def get_k_digit_number (seq : List Nat) (start k : Nat) : List Nat := seq.drop start |>.take k

-- Statement to prove the maximum k
theorem max_k_possible : ∃ k : Nat, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 5 = get_k_digit_number digits_sequence start2 5) ∧ (¬ ∃ k' > 5, (∀ start1 start2, start1 ≠ start2 → get_k_digit_number digits_sequence start1 k' = get_k_digit_number digits_sequence start2 k')) :=
sorry

end max_k_possible_l89_89197


namespace remainder_123456789012_mod_252_l89_89905

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l89_89905


namespace complementary_angles_l89_89752

/-- 
The ratio of measures of two complementary angles is 2 to 7. The smallest measure is increased by 20%.
Prove that the larger measure should be decreased by approximately 5.71% to ensure that the angles
remain complementary.
-/
theorem complementary_angles {
  α β : ℝ
  (hαβ : α + β = 90)
  (h_ratio : α / β = 2 / 7)
  (h_increase : α_new = α * 1.20)
  (h_sum : 90 - α_new = β_new) :
  (β_new / β) * 100 ≈ 94.29 :=  -- This means a decrease by approximately 5.71%
begin
  sorry
end

end complementary_angles_l89_89752


namespace tangent_315_deg_l89_89285

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89285


namespace reflection_over_y_eq_x_l89_89450

theorem reflection_over_y_eq_x :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, (∀ (v : Vector (Fin 2) ℝ), M.mulVec v = ⟨v.2, v.1⟩) ∧ 
  M = (Matrix.vecCons (Matrix.vecCons 0 1) (Matrix.vecCons 1 0)) :=
begin
  sorry
end

end reflection_over_y_eq_x_l89_89450


namespace tan_315_degree_l89_89402

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89402


namespace remainder_123456789012_div_252_l89_89916

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l89_89916


namespace unique_function_natural_l89_89444

theorem unique_function_natural (f : ℕ → ℕ) (h : ∀ n : ℕ, f(n) + f(f(n)) + f(f(f(n))) = 3n) : ∀ n : ℕ, f(n) = n := 
by
  sorry

end unique_function_natural_l89_89444


namespace arithmetic_sequence_proof_l89_89025

noncomputable def arithmetic_sum_condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d = -2 ∧ (∑ k in Finset.range 11, a (1 + 3 * k) = 50)

theorem arithmetic_sequence_proof (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sum_condition a d →
  (∑ k in Finset.range 12, a (2 + 4 * k)) = -82 :=
by
  sorry

end arithmetic_sequence_proof_l89_89025


namespace semicircle_area_proof_l89_89817

noncomputable def semicircle_area (a b : ℝ) (h : a < b) : ℝ :=
  let r := sqrt ((((b / 2) ^ 2) + ((a / 2) ^ 2))) in (1 / 2) * π * r ^ 2

theorem semicircle_area_proof : semicircle_area 2 3 (by norm_num) = (9 / 4) * π := sorry

end semicircle_area_proof_l89_89817


namespace no_unique_day_in_august_l89_89731

def july_has_five_tuesdays (N : ℕ) : Prop :=
  ∃ (d : ℕ), ∀ k : ℕ, k < 5 → (d + k * 7) ≤ 30

def july_august_have_30_days (N : ℕ) : Prop :=
  true -- We're asserting this unconditionally since both months have exactly 30 days in the problem

theorem no_unique_day_in_august (N : ℕ) (h1 : july_has_five_tuesdays N) (h2 : july_august_have_30_days N) :
  ¬(∃ d : ℕ, ∀ k : ℕ, k < 5 → (d + k * 7) ≤ 30 ∧ ∃! wday : ℕ, (d + k * 7 + wday) % 7 = 0) :=
sorry

end no_unique_day_in_august_l89_89731


namespace relationship_of_abc_l89_89999

theorem relationship_of_abc 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = 2 * Real.log 2 / Real.log 3)
  (h2 : b = -Real.log 2 / (-2 * Real.log 2))
  (h3 : c = 2 ^ (- 1 / 3)) :
  a > c ∧ c > b :=
by
  sorry

end relationship_of_abc_l89_89999


namespace tangent_315_deg_l89_89281

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89281


namespace reflection_over_line_y_eq_x_l89_89454

noncomputable def reflection_matrix_over_y_eq_x : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1;
     1, 0]

theorem reflection_over_line_y_eq_x (x y : ℝ) :
  (reflection_matrix_over_y_eq_x.mul_vec ![x, y]) = ![y, x] :=
by sorry

end reflection_over_line_y_eq_x_l89_89454


namespace reachable_cells_after_10_moves_l89_89621

theorem reachable_cells_after_10_moves :
  let board_size := 21
  let central_cell := (11, 11)
  let moves := 10
  (reachable_cells board_size central_cell moves) = 121 :=
by
  sorry

end reachable_cells_after_10_moves_l89_89621


namespace reachable_cells_after_moves_l89_89635

def is_valid_move (n : ℕ) (x y : ℤ) : Prop :=
(abs x ≤ n ∧ abs y ≤ n ∧ (x + y) % 2 = 0)

theorem reachable_cells_after_moves (n : ℕ) :
  n = 10 → ∃ (cells : Finset (ℤ × ℤ)), cells.card = 121 ∧ 
  (∀ (cell : ℤ × ℤ), cell ∈ cells → is_valid_move n cell.1 cell.2) :=
by
  intros h
  use {-10 ≤ x, y, x + y % 2 = 0 & abs x + abs y ≤ n }
  sorry -- proof goes here

end reachable_cells_after_moves_l89_89635


namespace tan_315_degrees_l89_89370

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89370


namespace exists_sequence_2017_power_l89_89434

theorem exists_sequence_2017_power :
  ∃ (a : ℕ → ℕ), (∀ n, ∃ s t : ℕ, t ≥ 2 ∧ a 1 = s * t ∧ (∀ n ≥ 2, a n = s * (t^n - t^(n-1))) ∧
  (∃ (k : ℕ), ∀ n > 0, (a 1 * a 2 * ... * a n) / (a 1 + a 2 + ... + a n) = k^2017)) :=
by
  sorry

end exists_sequence_2017_power_l89_89434


namespace total_cost_calculation_l89_89777

theorem total_cost_calculation :
  let land_cost := 30 * 20
  let house_cost := 120000
  let cows_cost := 20 * 1000
  let chickens_cost := 100 * 5
  let solar_installation_cost := 6 * 100
  let solar_equipment_cost := 6000
  let total_cost := land_cost + house_cost + cows_cost + chickens_cost + solar_installation_cost + solar_equipment_cost
  total_cost = 147700 := by
  let land_cost := 30 * 20
  let house_cost := 120000
  let cows_cost := 20 * 1000
  let chickens_cost := 100 * 5
  let solar_installation_cost := 6 * 100
  let solar_equipment_cost := 6000
  let total_cost := land_cost + house_cost + cows_cost + chickens_cost + solar_installation_cost + solar_equipment_cost
  show total_cost = 147700 from
    calc total_cost = 600 + 120000 + 20000 + 500 + 600 + 6000 : by {
      unfold land_cost house_cost cows_cost chickens_cost solar_installation_cost solar_equipment_cost total_cost,
      sorry
    }
    ... = 147700 : by {
      sorry
    }

end total_cost_calculation_l89_89777


namespace population_net_change_l89_89131

theorem population_net_change :
  let inc := (5 : ℝ) / 4
  let dec := (3 : ℝ) / 4
  let net_change_factor := inc * inc * dec * dec
  (net_change_factor - 1) * 100 ≈ -12 :=
by
  let initial_population := 1  -- Scale it to initial Population as 1 for simplification
  let final_population := initial_population * net_change_factor
  let net_change := (final_population - initial_population) / initial_population * 100
  have approx_eq : Real.floor (net_change + 0.5) = -12 := sorry
  exact approx_eq

end population_net_change_l89_89131


namespace P_sufficient_for_Q_P_not_necessary_for_Q_l89_89532

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

theorem P_sufficient_for_Q : ∀ x, x ∈ P → x ∈ Q := by
  intros x hx
  have h1 : x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 := by simp [P] at hx; exact hx
  cases h1 with h1 h2
  { rw h1; simp [Q] }
  cases h2 with h2 h3
  { rw h2; simp [Q] }
  cases h3 with h3 h4
  { rw h3; simp [Q] }
  { rw h4; simp [Q] }

theorem P_not_necessary_for_Q : ¬(∀ y, y ∈ Q → y ∈ P) := by
  intro h
  have h_counterexample : 2.5 ∈ Q := by simp [Q]; linarith
  have h_not_in_P : 2.5 ∉ P := by simp [P]; linarith
  exact h_not_in_P (h 2.5 h_counterexample)

end P_sufficient_for_Q_P_not_necessary_for_Q_l89_89532


namespace units_digit_of_factorial_sum_l89_89146

theorem units_digit_of_factorial_sum:
  let u1 := (1! % 10),
      u2 := (2! % 10),
      u3 := (3! % 10),
      u4 := (4! % 10),
      sum_units := (u1 + u2 + u3 + u4) % 10,
      result := (3 * sum_units) % 10
  in result = 9 :=
by
  sorry

end units_digit_of_factorial_sum_l89_89146


namespace tan_315_eq_neg1_l89_89375

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89375


namespace triangle_ratios_sum_eq_two_l89_89648

theorem triangle_ratios_sum_eq_two
  (A B C D E F : Point)
  (B_midpoint_AC : midpoint A C B)
  (D_on_BC : on_segment B C D)
  (ratio_BD_DC : \(\frac{BD}{DC} = \frac{1}{2}\))
  (E_on_AB : on_segment A B E)
  (ratio_AE_EB : \(\frac{AE}{EB} = \frac{2}{1}\))
  : \(\frac{EF}{FC} + \frac{AF}{FD} = 2\) :=
by
  sorry

end triangle_ratios_sum_eq_two_l89_89648


namespace derivative_of_f_l89_89105

noncomputable def f (x : ℝ) : ℝ := sin x - x * cos x

theorem derivative_of_f :
  ∀ x : ℝ,
  deriv f x = x * sin x :=
by
  intro x
  sorry

end derivative_of_f_l89_89105


namespace count_no_carrying_pairs_in_range_l89_89469

def is_consecutive (a b : ℕ) : Prop :=
  b = a + 1

def no_carrying (a b : ℕ) : Prop :=
  ∀ i, ((a / 10^i) % 10 + (b / 10^i) % 10) < 10

def count_no_carrying_pairs (start end_ : ℕ) : ℕ :=
  let pairs := (start to end_).to_list
  (pairs.zip pairs.tail).count (λ (a, b) => is_consecutive a b ∧ no_carrying a b)

theorem count_no_carrying_pairs_in_range :
  count_no_carrying_pairs 2000 3000 = 7290 :=
sorry

end count_no_carrying_pairs_in_range_l89_89469


namespace sebastian_older_than_jeremy_by_4_l89_89126

def J : ℕ := 40
def So : ℕ := 60 - 3
def sum_ages_in_3_years (S : ℕ) : Prop := (J + 3) + (S + 3) + (So + 3) = 150

theorem sebastian_older_than_jeremy_by_4 (S : ℕ) (h : sum_ages_in_3_years S) : S - J = 4 := by
  -- proof will be filled in
  sorry

end sebastian_older_than_jeremy_by_4_l89_89126


namespace arc_MTN_calculation_correct_l89_89014

noncomputable def calculate_arc_MTN_degrees (P Q R T M N : Type) 
  [IsoscelesTriangle P Q R] (s : ℝ) (radius : ℝ) (T_midpoint : IsMidpoint T Q)
  (angle_PQR : ∠ PQR = 40) : Prop :=
  let r := s / 3 in
  let arc_MTN := 60 in -- this is the derived answer, equivalent to the correct answer step
  arc_MTN = 60

theorem arc_MTN_calculation_correct (P Q R T M N : Type) 
  [IsoscelesTriangle P Q R] (s : ℝ) (radius : ℝ) (T_midpoint : IsMidpoint T Q)
  (angle_PQR : ∠ PQR = 40) : calculate_arc_MTN_degrees P Q R T M N s radius T_midpoint angle_PQR = 60 :=
  sorry

end arc_MTN_calculation_correct_l89_89014


namespace count_no_carrying_pairs_in_range_l89_89468

def is_consecutive (a b : ℕ) : Prop :=
  b = a + 1

def no_carrying (a b : ℕ) : Prop :=
  ∀ i, ((a / 10^i) % 10 + (b / 10^i) % 10) < 10

def count_no_carrying_pairs (start end_ : ℕ) : ℕ :=
  let pairs := (start to end_).to_list
  (pairs.zip pairs.tail).count (λ (a, b) => is_consecutive a b ∧ no_carrying a b)

theorem count_no_carrying_pairs_in_range :
  count_no_carrying_pairs 2000 3000 = 7290 :=
sorry

end count_no_carrying_pairs_in_range_l89_89468


namespace sara_red_balloons_l89_89717

theorem sara_red_balloons (initial_red : ℕ) (given_red : ℕ) 
  (h_initial : initial_red = 31) (h_given : given_red = 24) : 
  initial_red - given_red = 7 :=
by {
  sorry
}

end sara_red_balloons_l89_89717


namespace part1_part2_l89_89666

noncomputable def triangle_ABC (A B C a b c : ℝ) : Prop :=
  ∀ A B C a b c : ℝ,
    (sin C * sin (A - B) = sin (B) * sin (C - A)) →
    (A = 2 * B) →
    C = 5 * pi / 8

theorem part1 (A B C a b c : ℝ) (h1 : triangle_ABC A B C a b c) : C = 5 * Real.pi / 8 :=
  by sorry

noncomputable def triangle_ABC_equality (A B C a b c : ℝ) : Prop :=
  ∀ A B C a b c : ℝ,
    (sin C * sin (A - B) = sin (B) * sin (C - A)) →
    2 * a^2 = b^2 + c^2

theorem part2 (A B C a b c : ℝ) (h1 : triangle_ABC_equality A B C a b c) : 2 * a^2 = b^2 + c^2 :=
  by sorry

end part1_part2_l89_89666


namespace tangent_315_deg_l89_89280

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89280


namespace triangle_inequality_l89_89048

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  (a / Real.sqrt (2*b^2 + 2*c^2 - a^2)) + (b / Real.sqrt (2*c^2 + 2*a^2 - b^2)) + 
  (c / Real.sqrt (2*a^2 + 2*b^2 - c^2)) ≥ Real.sqrt 3 := by
  sorry

end triangle_inequality_l89_89048


namespace no_natural_number_divides_Q_by_x_squared_minus_one_l89_89890

def Q (n : ℕ) (x : ℝ) : ℝ := 1 + 5*x^2 + x^4 - (n - 1) * x^(n - 1) + (n - 8) * x^n

theorem no_natural_number_divides_Q_by_x_squared_minus_one :
  ∀ (n : ℕ), n > 0 → ¬ (x^2 - 1 ∣ Q n x) :=
by
  intros n h
  sorry

end no_natural_number_divides_Q_by_x_squared_minus_one_l89_89890


namespace number_of_elements_in_intersection_l89_89686

open Set

def f (x : ℝ) : ℝ := Real.log (abs (x + 1) - 1)

def A : Set ℝ := {x | x < -2 ∨ x > 0}

def U : Set ℝ := univ

def A_complement : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}

def B : Set ℝ := {x : ℝ | ∃ k : ℤ, x = 2 * k}

theorem number_of_elements_in_intersection : 
  ∃ (n : ℕ), n = 2 ∧ 
  {x | x ∈ A_complement ∧ x ∈ B}.finite ∧ 
  Finset.card (({x | x ∈ A_complement ∧ x ∈ B} : Set ℝ).toFinset) = n := sorry

end number_of_elements_in_intersection_l89_89686


namespace number_of_zeros_in_interval_2_5_l89_89750

noncomputable def f (x : ℝ) := -x^2 + 8 * x - 14

theorem number_of_zeros_in_interval_2_5 : 
  ∃ x : ℝ, x ∈ set.Icc 2 5 ∧ f x = 0 :=
sorry

end number_of_zeros_in_interval_2_5_l89_89750


namespace remainder_123456789012_div_252_l89_89911

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l89_89911


namespace S_2015_mod_12_l89_89069

def A : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 2
| (n+3) := A n + A (n + 1) + A (n + 2)

def S (n : ℕ) := 2 * A n

theorem S_2015_mod_12 : S 2015 % 12 = 8 := 
by
  sorry

end S_2015_mod_12_l89_89069


namespace part_a_impossible_to_zero_part_b_possible_to_zero_part_c_possible_to_zero_l89_89166

-- Part (a)
def sequence_a : List ℕ := List.range (1998 + 1)

def operation (x y : ℕ) : ℕ := |x - y|

theorem part_a_impossible_to_zero : ¬ (∃ seq, seq.last = 0 ∧
  seq.head = sequence_a ∧
  (∀ i j, i ≠ j → seq (i + 1) = seq i.erase seq j.erase operation))
  sorry

-- Part (b)
def sequence_b : List ℕ := List.range (1999 + 1)

theorem part_b_possible_to_zero : ∃ seq, seq.last = 0 ∧
  seq.head = sequence_b ∧
  (∀ i j, i ≠ j → seq (i + 1) = seq i.erase seq j.erase operation
  sorry

-- Part (c)
def sequence_c : List ℕ := List.range (2000 + 1)

theorem part_c_possible_to_zero : ∃ seq, seq.last = 0 ∧
  seq.head = sequence_c ∧
  (∀ i j, i ≠ j → seq (i + 1) = seq i.erase seq j.erase operation)
  sorry

end part_a_impossible_to_zero_part_b_possible_to_zero_part_c_possible_to_zero_l89_89166


namespace volume_tetrahedron_proof_l89_89600

noncomputable def volume_of_tetrahedron (p q r : ℝ) : ℝ :=
  let cos_60 : ℝ := Real.cos (Real.pi / 3) in
  let sin_60 : ℝ := Real.sin (Real.pi / 3) in
  let area_BCD : ℝ := (1 / 2) * p * q * sin_60 in
  let height_A_to_BCD : ℝ := r * (Real.sqrt 6 / 3) in
  (1 / 3) * area_BCD * height_A_to_BCD

theorem volume_tetrahedron_proof (p q r : ℝ) (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) :
  ∀ (A B C D : Type) [InnerProductSpace ℝ ℝ] 
  (AB AC AD : ℝ)
  (hAB : AB = p) 
  (hAC : AC = q) 
  (hAD : AD = r) 
  (hCAD : inner_product_space.angle A C A D = Real.pi / 3)
  (hCAB : inner_product_space.angle C A B = Real.pi / 3)
  (hBAD : inner_product_space.angle B A D = Real.pi / 3), 
  volume_of_tetrahedron p q r = (Real.sqrt 2 / 12) * p * q * r := 
sorry

end volume_tetrahedron_proof_l89_89600


namespace area_of_triangle_formed_by_diagonal_l89_89980

theorem area_of_triangle_formed_by_diagonal (area_parallelogram : ℝ) (a b : ℝ) (θ : ℝ) (h : area_parallelogram = 128) :
  ∃ area_triangle, area_triangle = 1 / 2 * area_parallelogram ∧ area_triangle = 64 :=
by {
  use 1 / 2 * area_parallelogram,
  split,
  {
    exact mul_div_cancel' area_parallelogram two_ne_zero,
  },
  {
    rw h,
    norm_num,
  },
}

end area_of_triangle_formed_by_diagonal_l89_89980


namespace tan_315_eq_neg1_l89_89334

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89334


namespace smith_paid_correct_discounted_price_l89_89755

/-- The original selling price of the shirt is Rs. 700 -/
def original_selling_price : ℝ := 700

/-- The shop offered a 20% discount for every shirt -/
def discount_percentage : ℝ := 20

/-- The discounted price Smith paid for the shirt -/
def discounted_price (original_price discount_pct : ℝ) : ℝ :=
  original_price - ((discount_pct / 100) * original_price)

theorem smith_paid_correct_discounted_price :
  discounted_price original_selling_price discount_percentage = 560 := by
  sorry

end smith_paid_correct_discounted_price_l89_89755


namespace problem1_problem2_l89_89719

-- Proving that (3*sqrt(8) - 12*sqrt(1/2) + sqrt(18)) * 2*sqrt(3) = 6*sqrt(6)
theorem problem1 :
  (3 * Real.sqrt 8 - 12 * Real.sqrt (1/2) + Real.sqrt 18) * 2 * Real.sqrt 3 = 6 * Real.sqrt 6 :=
sorry

-- Proving that (6*sqrt(x/4) - 2*x*sqrt(1/x)) / 3*sqrt(x) = 1/3
theorem problem2 (x : ℝ) (hx : 0 < x) :
  (6 * Real.sqrt (x/4) - 2 * x * Real.sqrt (1/x)) / (3 * Real.sqrt x) = 1/3 :=
sorry

end problem1_problem2_l89_89719


namespace noise_pollution_l89_89700

variables (p p0 p1 p2 p3 : ℝ)
variable (p0_pos : p0 > 0)
variable (Lp1 Lp2 Lp3 : ℝ)
variable (gasoline_car_condition : 60 ≤ Lp1 ∧ Lp1 ≤ 90)
variable (hybrid_car_condition : 50 ≤ Lp2 ∧ Lp2 ≤ 60)
variable (electric_car_condition : Lp3 = 40)

theorem noise_pollution : 
  (20 * log10 (p1 / p0) = Lp1) ∧ (20 * log10 (p2 / p0) = Lp2) ∧ (20 * log10 (p3 / p0) = Lp3) ∧ 
  (60 ≤ Lp1 ∧ Lp1 ≤ 90) ∧ (50 ≤ Lp2 ∧ Lp2 ≤ 60) ∧ (Lp3 = 40) →
  (p1 ≥ p2) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) :=
by
  sorry

end noise_pollution_l89_89700


namespace find_percentage_l89_89816

variable (P : ℝ)
variable (num : ℝ := 70)
variable (result : ℝ := 25)

theorem find_percentage (h : ((P / 100) * num) - 10 = result) : P = 50 := by
  sorry

end find_percentage_l89_89816


namespace trapezium_area_l89_89033

variables {A B C D O : Type}
variables (P Q : ℕ)

-- Conditions
def trapezium (ABCD : Type) : Prop := true
def parallel_lines (AB DC : Type) : Prop := true
def intersection (AC BD O : Type) : Prop := true
def area_AOB (P : ℕ) : Prop := P = 16
def area_COD : ℕ := 25

theorem trapezium_area (ABCD AC BD AB DC O : Type) (P Q : ℕ)
  (h1 : trapezium ABCD)
  (h2 : parallel_lines AB DC)
  (h3 : intersection AC BD O)
  (h4 : area_AOB P) 
  (h5 : area_COD = 25) :
  Q = 81 :=
sorry

end trapezium_area_l89_89033


namespace log_equation_solution_l89_89431

theorem log_equation_solution (x : ℝ) :
  (log (x + 5) + log (x - 3) = log (x^2 - 4)) ↔ (x = 11 / 2) :=
by {
  sorry
}

end log_equation_solution_l89_89431


namespace find_x_l89_89988

variable (x : ℝ)
variable (θ : ℝ)
variable (P : ℝ × ℝ := (x, 3))
variable hcosθ : cos θ = -4 / 5

theorem find_x
  (P_condition : P = (x, 3))
  (cos_condition : cos θ = -4 / 5)
  (cos_definition : cos θ = x / real.sqrt (x^2 + 9)) :
  x = -4 :=
  sorry

end find_x_l89_89988


namespace count_lattice_points_hyperbola_line_l89_89748

theorem count_lattice_points_hyperbola_line:
  (∑ x in (range 98).map (λ n, 2*(n+2) - 1)) = 9800 :=
by
  sorry

end count_lattice_points_hyperbola_line_l89_89748


namespace percentage_discount_l89_89695

theorem percentage_discount (individual_payment_without_discount final_payment discount_per_person : ℝ)
  (h1 : 3 * individual_payment_without_discount = final_payment + 3 * discount_per_person)
  (h2 : discount_per_person = 4)
  (h3 : final_payment = 48) :
  discount_per_person / (individual_payment_without_discount * 3) * 100 = 20 :=
by
  -- Proof to be provided here
  sorry

end percentage_discount_l89_89695


namespace tan_315_eq_neg1_l89_89268

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89268


namespace proof_equiv_l89_89076

noncomputable theory

def regression_equation (x : ℝ) : ℝ := -0.32 * x + 40
def correlation_coefficient : ℝ := -0.9923
def data_points_x : List ℝ := [90, 95, 100, 105, 110]
def data_points_y : List ℝ := [11, 10, 8, 6, 5]
def intercept : ℝ := 40
def residual (x y : ℝ) : ℝ := y - regression_equation x

theorem proof_equiv :
  (-0.32 < 0) = True ∧
  (|correlation_coefficient| ≈ 1) ∧
  intercept = 40 ∧
  residual 95 10 ≈ 0.4 :=
by
  sorry

end proof_equiv_l89_89076


namespace no_such_n_exists_l89_89682

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, n * sum_of_digits n = 100200300 :=
by
  sorry

end no_such_n_exists_l89_89682


namespace range_of_a_l89_89430

theorem range_of_a (a : ℝ) : (∃ x : ℝ, cos x ^ 2 - 2 * cos x - a = 0) ↔ -1 ≤ a ∧ a ≤ 3 := by 
sorry

end range_of_a_l89_89430


namespace remainder_123456789012_div_252_l89_89943

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l89_89943


namespace total_legs_in_room_is_71_l89_89011

theorem total_legs_in_room_is_71 :
  let table_4_legs := 4 * 4
  let sofa_4_legs := 1 * 4
  let chair_4_legs := 2 * 4
  let table_3_legs := 3 * 3
  let table_1_leg := 1 * 1
  let rocking_chair_2_legs := 1 * 2
  let bench_6_legs := 1 * 6
  let stool_3_legs := 2 * 3
  let wardrobe_4_3_legs := 1 * 4 + 1 * 3
  let ecko_3_legs := 1 * 3
  let antique_table_remaining_3_legs := 1 * 3
  let damaged_4_leg_table := 1 * 3.5
  let damaged_stool_2_5_legs := 1 * 2.5
  let total_legs :=
    table_4_legs + sofa_4_legs + chair_4_legs +
    table_3_legs + table_1_leg + rocking_chair_2_legs +
    bench_6_legs + stool_3_legs + wardrobe_4_3_legs +
    ecko_3_legs + antique_table_remaining_3_legs +
    damaged_4_leg_table + damaged_stool_2_5_legs
  in total_legs = 71 := by
  have table_4_legs_val : table_4_legs = 16 := rfl
  have sofa_4_legs_val : sofa_4_legs = 4 := rfl
  have chair_4_legs_val : chair_4_legs = 8 := rfl
  have table_3_legs_val : table_3_legs = 9 := rfl
  have table_1_leg_val : table_1_leg = 1 := rfl
  have rocking_chair_2_legs_val : rocking_chair_2_legs = 2 := rfl
  have bench_6_legs_val : bench_6_legs = 6 := rfl
  have stool_3_legs_val : stool_3_legs = 6 := rfl
  have wardrobe_4_3_legs_val : wardrobe_4_3_legs = 7 := rfl
  have ecko_3_legs_val : ecko_3_legs = 3 := rfl
  have antique_table_remaining_3_legs_val : antique_table_remaining_3_legs = 3 := rfl
  have damaged_4_leg_table_val : damaged_4_leg_table = 3.5 := rfl
  have damaged_stool_2_5_legs_val : damaged_stool_2.5 := rfl
  calc total_legs = 16 + 4 + 8 + 9 + 1 + 2 + 6 + 6 + 7 + 3 + 3 + 3.5 + 2.5 : by simp
     ... = 71 : by sorry

end total_legs_in_room_is_71_l89_89011


namespace smallest_consecutive_sum_l89_89127

theorem smallest_consecutive_sum (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by 
  sorry

end smallest_consecutive_sum_l89_89127


namespace person_speed_1440m_in_12min_is_7_2kmh_l89_89803

def meters_to_kilometers (d : ℕ) : ℝ := d / 1000.0
def minutes_to_hours (t : ℕ) : ℝ := t / 60.0
def speed (d : ℝ) (t : ℝ) : ℝ := d / t

theorem person_speed_1440m_in_12min_is_7_2kmh :
  speed (meters_to_kilometers 1440) (minutes_to_hours 12) = 7.2 := by
  sorry

end person_speed_1440m_in_12min_is_7_2kmh_l89_89803


namespace grandma_molly_statues_l89_89538

theorem grandma_molly_statues :
  ∃ F : ℕ, (4 * F + 12 - 3 + 6 = 31) :=
begin
  use 4,
  norm_num,
end

end grandma_molly_statues_l89_89538


namespace tan_315_degree_l89_89411

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89411


namespace educated_employees_count_l89_89159

def daily_wages_decrease (illiterate_avg_before illiterate_avg_after illiterate_count : ℕ) : ℕ :=
  (illiterate_avg_before - illiterate_avg_after) * illiterate_count

def total_employees (total_decreased total_avg_decreased : ℕ) : ℕ :=
  total_decreased / total_avg_decreased

theorem educated_employees_count :
  ∀ (illiterate_avg_before illiterate_avg_after illiterate_count total_avg_decreased : ℕ),
    illiterate_avg_before = 25 →
    illiterate_avg_after = 10 →
    illiterate_count = 20 →
    total_avg_decreased = 10 →
    total_employees (daily_wages_decrease illiterate_avg_before illiterate_avg_after illiterate_count) total_avg_decreased - illiterate_count = 10 :=
by
  intros
  sorry

end educated_employees_count_l89_89159


namespace find_distance_cd_l89_89419

noncomputable def distance_cd : ℝ :=
  let ellipse : (ℝ × ℝ) → Prop := λ p, 16 * (p.1 + 2) ^ 2 + 4 * p.2 ^ 2 = 64
  let a := 4
  let b := 2
  real.sqrt (a ^ 2 + b ^ 2)

theorem find_distance_cd :
  let ellipse (p : ℝ × ℝ) : Prop := 16 * (p.1 + 2) ^ 2 + 4 * p.2 ^ 2 = 64
  let c : ℝ × ℝ := (-2, 4)  -- One of the endpoints of the major axis
  let d : ℝ × ℝ := (0, 0)  -- One of the endpoints of the minor axis
  distance_cd = 2 * real.sqrt 5 :=
by
  sorry

end find_distance_cd_l89_89419


namespace tan_315_eq_neg_one_l89_89293

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89293


namespace tan_315_eq_neg1_l89_89346

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89346


namespace remainder_123456789012_div_252_l89_89945

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l89_89945


namespace tan_315_eq_neg1_l89_89398

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89398


namespace tan_315_eq_neg1_l89_89341

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89341


namespace distinct_symbols_count_l89_89581

/-- A modified Morse code symbol is represented by a sequence of dots, dashes, and spaces, where spaces can only appear between dots and dashes but not at the beginning or end of the sequence. -/
def valid_sequence_length_1 := 2
def valid_sequence_length_2 := 2^2
def valid_sequence_length_3 := 2^3 + 3
def valid_sequence_length_4 := 2^4 + 3 * 2^4 + 3 * 2^4 
def valid_sequence_length_5 := 2^5 + 4 * 2^5 + 6 * 2^5 + 4 * 2^5

theorem distinct_symbols_count : 
  valid_sequence_length_1 + valid_sequence_length_2 + valid_sequence_length_3 + valid_sequence_length_4 + valid_sequence_length_5 = 609 := by
  sorry

end distinct_symbols_count_l89_89581


namespace part_a_l89_89066

variable (f : ℝ → ℝ)
variable (h_cont : ∀ (x : ℝ), 0 ≤ x → 0 ≤ f x)
variable (h_lim : tendsto (λ x, f (f x)) at_top at_top)

theorem part_a (h_cont : continuous_on f (λ x, 0 ≤ x)) :
  tendsto (λ x, f x) at_top at_top :=
sorry

end part_a_l89_89066


namespace triangle_area_13_14_15_l89_89572

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let sin_C := Real.sqrt (1 - cos_C^2)
  (1/2) * a * b * sin_C

theorem triangle_area_13_14_15 : area_of_triangle 13 14 15 = 84 :=
by sorry

end triangle_area_13_14_15_l89_89572


namespace number_of_reachable_cells_after_10_moves_l89_89641

theorem number_of_reachable_cells_after_10_moves : 
  (let 
    n := 21 
    center := (11, 11)
    moves := 10
  in
  ∃ reachable_cells, reachable_cells = 121) :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89641


namespace triangle_side_possible_values_l89_89565

theorem triangle_side_possible_values (m : ℝ) (h1 : 1 < m) (h2 : m < 7) : 
  m = 5 :=
by
  sorry

end triangle_side_possible_values_l89_89565


namespace mean_eq_median_lt_mode_l89_89139

-- Define the daily fish count data
def daily_fish_count : List ℕ := [1, 3, 0, 0, 2, 2, 3, 1, 1, 2, 3, 0]

-- Define the mean, median, and mode calculations
def mean (data : List ℕ) : ℚ :=
  let sum : ℕ := data.foldl (· + ·) 0
  sum.toRat / data.length

def median (data : List ℕ) : ℚ :=
  let sorted_data := data.qsort (· ≤ ·)
  if sorted_data.length % 2 = 0 then
    let mid1 := sorted_data.get! (sorted_data.length / 2 - 1)
    let mid2 := sorted_data.get! (sorted_data.length / 2)
    (mid1 + mid2).toRat / 2
  else
    sorted_data.get! (sorted_data.length / 2).toRat

def mode (data : List ℕ) : List ℕ :=
  let grouped_data := data.groupBy (· = ·)
  let max_count := grouped_data.foldl (λ acc xs, max acc xs.length) 0
  grouped_data.filter (λ xs, xs.length = max_count) |>.map List.head |>.unzip

-- Define the mean, median, and mode values for daily_fish_count
def mean_value : ℚ := mean daily_fish_count
def median_value : ℚ := median daily_fish_count
def mode_values : List ℕ := mode daily_fish_count

-- The statement to prove the relationship between mean, median, and mode
theorem mean_eq_median_lt_mode : mean_value = median_value ∧ ∀ m ∈ mode_values, mean_value < m.toRat := by
  sorry

end mean_eq_median_lt_mode_l89_89139


namespace square_area_l89_89643

open Complex

theorem square_area (z : ℂ) (h1 : z ≠ 0) (h2 : z ≠ 1) (hsquare : ∃ (x₁ x₂ x₃ x₄ : ℂ), {z, z^2, z^4} ⊆ {x₁, x₂, x₃, x₄} ∧ ({x₁, x₂, x₃, x₄}.pairwise (λ a b, |a - b| = |x₁ - x₂|)) ∧ ({x₁, x₂, x₃, x₄}.pairwise (λ a b, (a - b).arg = (x₁ - x₂).arg ± π/2))) : |z^2 - z|^2 = 10 :=
by
  sorry

end square_area_l89_89643


namespace count_no_carry_pairs_l89_89473

theorem count_no_carry_pairs : 
  ∃ n, n = 1125 ∧ ∀ (a b : ℕ), (2000 ≤ a ∧ a < 2999 ∧ b = a + 1) → 
  (∀ i, (0 ≤ i ∧ i < 4) → ((a / (10 ^ i) % 10 + b / (10 ^ i) % 10) < 10)) := sorry

end count_no_carry_pairs_l89_89473


namespace bella_steps_l89_89207

-- Define the conditions and the necessary variables
variable (b : ℝ) (distance : ℝ) (steps_per_foot : ℝ)

-- Given constants
def bella_speed := b
def ella_speed := 4 * b
def combined_speed := bella_speed + ella_speed
def total_distance := 15840
def feet_per_step := 3

-- Define the main theorem to prove the number of steps Bella takes
theorem bella_steps : (total_distance / combined_speed) * bella_speed / feet_per_step = 1056 := by
  sorry

end bella_steps_l89_89207


namespace lisa_mother_twice_age_l89_89705

theorem lisa_mother_twice_age : 
  let (b: ℕ) := 2004 in
  let (lisa_age: ℕ) := 10 in
  let (mother_age: ℕ) := 5 * 10 in
  ∃ (x: ℕ), (mother_age + x = 2 * (lisa_age + x)) ∧ (b + x = 2034) :=
by
  sorry

end lisa_mother_twice_age_l89_89705


namespace distance_between_points_l89_89894

def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2 + (b.3 - a.3) ^ 2)

theorem distance_between_points :
  distance (3, -2, 7) (8, 3, 6) = real.sqrt 51 :=
by sorry

end distance_between_points_l89_89894


namespace max_elements_in_S_l89_89898

open Nat

theorem max_elements_in_S (S : Finset ℕ) 
  (h1 : ∀ a ∈ S, 0 < a ∧ a ≤ 100)
  (h2 : ∀ a b ∈ S, a ≠ b → ∃ c ∈ S, gcd a c = 1 ∧ gcd b c = 1)
  (h3 : ∀ a b ∈ S, a ≠ b → ∃ d ∈ S, d ≠ a ∧ d ≠ b ∧ gcd a d > 1 ∧ gcd b d > 1) :
  S.card ≤ 72 := 
sorry

end max_elements_in_S_l89_89898


namespace exists_triangle_area_leq_sqrt3_l89_89697

theorem exists_triangle_area_leq_sqrt3 (points : Finset (ℝ × ℝ))
  (h₀ : ∀ p ∈ points, polygon.contains (equilateral_triangle_of_side_4) p)
  (h₁ : ∀ (p1 p2 p3 : ℝ × ℝ) (hp1: p1 ∈ points) (hp2: p2 ∈ points) (hp3: p3 ∈ points), 
          ¬ collinear ℝ [p1, p2, p3])
  (h_total : points.card = 9) :
  ∃ (p1 p2 p3 : ℝ × ℝ), 
    p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    triangle_area p1 p2 p3 ≤ sqrt 3 :=
by 
  sorry

end exists_triangle_area_leq_sqrt3_l89_89697


namespace total_spent_target_l89_89871

theorem total_spent_target (face_moisturizer_cost : ℕ) (body_lotion_cost : ℕ) (face_moisturizers_bought : ℕ) (body_lotions_bought : ℕ) (christy_multiplier : ℕ) :
  face_moisturizer_cost = 50 →
  body_lotion_cost = 60 →
  face_moisturizers_bought = 2 →
  body_lotions_bought = 4 →
  christy_multiplier = 2 →
  (face_moisturizers_bought * face_moisturizer_cost + body_lotions_bought * body_lotion_cost) * (1 + christy_multiplier) = 1020 := by
  sorry

end total_spent_target_l89_89871


namespace tan_315_eq_neg1_l89_89251

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89251


namespace sum_difference_even_multiples_l89_89791

theorem sum_difference_even_multiples :
  let E := (finset.range 2025).map (λ n, 2 * (n + 1))
  let M := (finset.range 2025).map (λ n, 3 * (n + 1))
  (E.sum (λ x, x) - M.sum (λ x, x) = -2052155) :=
by
  sorry

end sum_difference_even_multiples_l89_89791


namespace tan_315_proof_l89_89302

noncomputable def tan_315_eq_neg1 : Prop :=
  let θ := 315 : ℝ in
  let x := ((real.sqrt 2) / 2) in
  let y := -((real.sqrt 2) / 2) in
  tan (θ * real.pi / 180) = y / x

theorem tan_315_proof : tan_315_eq_neg1 := by
  sorry

end tan_315_proof_l89_89302


namespace quadratic_real_roots_l89_89967

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) →
  (m ≤ 4 ∧ m ≠ 3) :=
sorry

end quadratic_real_roots_l89_89967


namespace tan_315_eq_neg_1_l89_89321

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89321


namespace tan_315_degree_l89_89410

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89410


namespace range_of_a_solve_log_inequality_log_function_min_value_l89_89975

-- 1. Prove that if a > 0 and 2^{2a+1} > 2^{5a-2}, then a < 1
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : 2^(2 * a + 1) > 2^(5 * a - 2)) : a < 1 :=
by sorry

-- 2. Prove that if 0 < a < 1, then the solution set for log_a(2x-1) < log_a(7-5x) is (8/7, 7/5)
theorem solve_log_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : set_of (λ x : ℝ, log a (2 * x - 1) < log a (7 - 5 * x)) = { x | 8 / 7 < x ∧ x < 7 / 5 } :=
by sorry

-- 3. Prove that if the function y = log_a(2x-1) has a minimum value of -2 in the interval [1, 3], then a = √5 / 5
theorem log_function_min_value (a : ℝ) (h1 : ∀ x ∈ set.Icc 1 3, log a (2 * x - 1) ≥ -2) (h2 : log a 5 = -2) : a = real.sqrt 5 / 5 :=
by sorry

end range_of_a_solve_log_inequality_log_function_min_value_l89_89975


namespace coupon1_max_discount_l89_89835

def coupon1_discount (x : ℝ) : ℝ := if x ≥ 60 then 0.12 * x else 0
def coupon2_discount (x : ℝ) : ℝ := if x ≥ 120 then 25 else 0
def coupon3_discount (x : ℝ) : ℝ := if x ≥ 120 then 0.20 * (x - 120) else 0
def coupon4_discount (x : ℝ) : ℝ := if x ≥ 200 then 35 else 0

theorem coupon1_max_discount (x : ℝ) (h1 : x = 219.95) :
  coupon1_discount x > coupon2_discount x ∧ 
  coupon1_discount x > coupon3_discount x ∧ 
  coupon1_discount x > coupon4_discount x :=
by {
  -- The proof will be filled here.
  sorry
}

end coupon1_max_discount_l89_89835


namespace triangle_perimeter_l89_89856

theorem triangle_perimeter {a b c : ℕ} (ha : a = 10) (hb : b = 6) (hc : c = 7) :
    a + b + c = 23 := by
  sorry

end triangle_perimeter_l89_89856


namespace sum_of_a_b_l89_89685

-- Definitions for the given conditions
def geom_series_sum (a : ℤ) (n : ℕ) : ℤ := 2^n + a
def arith_series_sum (b : ℤ) (n : ℕ) : ℤ := n^2 - 2*n + b

-- Theorem statement
theorem sum_of_a_b (a b : ℤ) (h1 : ∀ n, geom_series_sum a n = 2^n + a)
  (h2 : ∀ n, arith_series_sum b n = n^2 - 2*n + b) :
  a + b = -1 :=
sorry

end sum_of_a_b_l89_89685


namespace minimal_planes_cover_S_l89_89673

theorem minimal_planes_cover_S (n : ℕ) : 
  let S := {p : Fin (n+1) × Fin (n+1) × Fin (n+1) // p.1 + p.2 + p.3 > 0} in
  ∃ (planes : ℕ), 
    (∀ (p ∈ S), ∃ (k ∈ {1, ..., 3*n}), p.1 + p.2 + p.3 = k) ∧
    (∀ (p : Fin (n+1) × Fin (n+1) × Fin (n+1)), p = (0, 0, 0) → ¬∃ k ∈ {1, ..., 3*n}, p.1 + p.2 + p.3 = k) ∧
    planes = 3*n :=
begin
  sorry
end

end minimal_planes_cover_S_l89_89673


namespace probability_sum_odd_l89_89825

theorem probability_sum_odd (balls : Finset ℕ) (odd_balls even_balls : Finset ℕ) :
  balls = Finset.range 14 ∧
  odd_balls = {1, 3, 5, 7, 9, 11, 13} ∧
  even_balls = {2, 4, 6, 8, 10, 12} ∧
  (∀ (draw : Finset ℕ), draw.card = 7 → draw ⊆ balls → 
   (finset.card (draw ∩ odd_balls) % 2 = 1 → 
    ∃ p : ℚ, p = 212 / 429)) :=
begin
  sorry
end

end probability_sum_odd_l89_89825


namespace fifth_student_selected_l89_89177

-- Preconditions
def is_student (n : ℕ) : Prop := n ≤ 55
def selected_students : List ℕ := [5, 16, 27, 49]
def systematic_sampling (students : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → (j < students.length) → (students[j] - students[i]) = 11

-- Main theorem
theorem fifth_student_selected : ∃ (n : ℕ), (n = 38) ∧ (n :: selected_students = List.nth_le selected_students 4 sorry) := 
by 
  sorry

end fifth_student_selected_l89_89177


namespace proof_ellipse_properties_l89_89072

noncomputable def ellipse_eccentricity_distance (a b : ℝ) (e : ℝ) (d : ℝ) : Prop :=
  let c := a * e in
  e = 1 / 2 ∧
  a > b ∧
  b > 0 ∧
  d = √21 / 7 ∧
  a = 2 ∧
  b = √3 ∧
  ∀ (x y : ℝ), ((x^2 / (a^2)) + (y^2 / (b^2)) = 1 ↔ (x^2 / 4 + y^2 / 3 = 1)) ∧
  ∀ (m k : ℝ), Δ := ((k^2 + 1) * (4 * m^2 - 12) / (3 + 4 * k^2) - (8 * k^2 * m^2) / (3 + 4 * k^2) + m^2) = 0 →
    Δ = 7 * m^2 - 12 * (k^2 + 1) ∧
    m / √(k^2 + 1) = 2 * √21 / 7

theorem proof_ellipse_properties : ellipse_eccentricity_distance 2 (√3) (1 / 2) (√21 / 7) := 
  sorry

end proof_ellipse_properties_l89_89072


namespace tan_315_eq_neg_one_l89_89298

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89298


namespace tan_315_eq_neg_one_l89_89290

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89290


namespace average_of_11_numbers_l89_89103

theorem average_of_11_numbers 
  (avg_first6 : ℝ)
  (avg_last6 : ℝ)
  (middle_num : ℝ) 
  (H1 : avg_first6 = 10.5)
  (H2 : avg_last6 = 11.4)
  (H3 : middle_num = 22.5) :
  ((6 * avg_first6) + (6 * avg_last6) - middle_num) / 11 = 9.9 :=
by 
  rw [H1, H2, H3]
  norm_num
  sorry

end average_of_11_numbers_l89_89103


namespace arithmetic_sum_l89_89210

theorem arithmetic_sum : (∑ k in Finset.range 10, (1 + 2 * k)) = 100 :=
by
  sorry

end arithmetic_sum_l89_89210


namespace train_length_l89_89854

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_seconds : ℝ := 36
noncomputable def speed_m_s := speed_km_hr * (5/18 : ℝ)
noncomputable def distance := speed_m_s * time_seconds

-- Theorem statement
theorem train_length : distance = 600.12 := by
  sorry

end train_length_l89_89854


namespace tan_315_eq_neg_one_l89_89220

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_one_l89_89220


namespace tan_315_eq_neg_one_l89_89291

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89291


namespace root_interval_l89_89057

def f (x : ℝ) : ℝ := log x / log 10 + x - 3

theorem root_interval :
  f 2.25 < 0 ∧ f 2.75 > 0 ∧ f 2.5 < 0 ∧ f 3 > 0 →
  ∃ c, 2.5 < c ∧ c < 2.75 ∧ f c = 0 :=
by
  assume h : f 2.25 < 0 ∧ f 2.75 > 0 ∧ f 2.5 < 0 ∧ f 3 > 0
  sorry

end root_interval_l89_89057


namespace train_length_l89_89852

theorem train_length (V L : ℝ) (h₁ : V = L / 18) (h₂ : V = (L + 200) / 30) : L = 300 :=
by
  sorry

end train_length_l89_89852


namespace tan_315_degrees_l89_89368

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l89_89368


namespace number_of_ping_pong_balls_l89_89180

def sales_tax_rate : ℝ := 0.16

def total_cost_with_tax (B x : ℝ) : ℝ := B * x * (1 + sales_tax_rate)

def total_cost_without_tax (B x : ℝ) : ℝ := (B + 3) * x

theorem number_of_ping_pong_balls
  (B x : ℝ) (h₁ : total_cost_with_tax B x = total_cost_without_tax B x) :
  B = 18.75 := 
sorry

end number_of_ping_pong_balls_l89_89180


namespace tan_315_eq_neg1_l89_89265

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89265


namespace propositions_are_3_and_4_l89_89740

-- Conditions
def stmt_1 := "Is it fun to study math?"
def stmt_2 := "Do your homework well and strive to pass the math test next time;"
def stmt_3 := "2 is not a prime number"
def stmt_4 := "0 is a natural number"

-- Representation of a propositional statement
def isPropositional (stmt : String) : Bool :=
  stmt ≠ stmt_1 ∧ stmt ≠ stmt_2

-- The theorem proving the question given the conditions
theorem propositions_are_3_and_4 :
  isPropositional stmt_3 ∧ isPropositional stmt_4 :=
by
  -- Proof to be filled in later
  sorry

end propositions_are_3_and_4_l89_89740


namespace tan_315_eq_neg1_l89_89380

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89380


namespace fraction_equivalence_l89_89426

theorem fraction_equivalence :
    (∃ (C D : ℚ), C = 32/9 ∧ D = 13/9 ∧ (∀ (x : ℚ), x ≠ 7 ∧ x ≠ -2 →
        (5 * x - 3) / (x^2 - 5 * x - 14) = C / (x - 7) + D / (x + 2))) :=
begin
  use [32/9, 13/9],
  split, { refl },
  split, { refl },
  intros x hx,
  have h1 : x^2 - 5 * x - 14 = (x - 7) * (x + 2),
  { ring },
  sorry -- proof omitted
end

end fraction_equivalence_l89_89426


namespace reflection_over_y_eq_x_l89_89452

theorem reflection_over_y_eq_x :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, (∀ (v : Vector (Fin 2) ℝ), M.mulVec v = ⟨v.2, v.1⟩) ∧ 
  M = (Matrix.vecCons (Matrix.vecCons 0 1) (Matrix.vecCons 1 0)) :=
begin
  sorry
end

end reflection_over_y_eq_x_l89_89452


namespace tan_315_eq_neg1_l89_89379

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89379


namespace tan_315_eq_neg1_l89_89356

noncomputable def cosd (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def sind (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def tand (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

theorem tan_315_eq_neg1 : tand 315 = -1 :=
by
  have h1 : 315 = 360 - 45 := by norm_num
  have cos_45 := by norm_num; exact Real.cos (45 * Real.pi / 180)
  have sin_45 := by norm_num; exact Real.sin (45 * Real.pi / 180)
  rw [tand, h1, Real.tan_eq_sin_div_cos, Real.sin_sub, Real.cos_sub]
  rw [Real.sin_pi_div_four]
  rw [Real.cos_pi_div_four]
  norm_num
  sorry -- additional steps are needed but sorrry is used as per instruction

end tan_315_eq_neg1_l89_89356


namespace stacy_days_to_complete_paper_l89_89728

-- Conditions as definitions
def total_pages : ℕ := 63
def pages_per_day : ℕ := 9

-- The problem statement
theorem stacy_days_to_complete_paper : total_pages / pages_per_day = 7 :=
by
  sorry

end stacy_days_to_complete_paper_l89_89728


namespace remainder_123456789012_mod_252_l89_89907

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l89_89907


namespace olympiad_scores_greater_than_18_l89_89595

open Classical

theorem olympiad_scores_greater_than_18 (n : ℕ) (a : ℕ → ℕ) (h_distinct: ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_ordered: ∀ i j: ℕ, i < j → a i < a j)
  (h_condition: ∀ i j k: ℕ, i ≠ j → i ≠ k → j ≠ k → a i < a j + a k) :
  ∀ i < n, n = 20 ∧ a i > 18 :=
by
  assume i h_i_lt_n h_n_eq_20
  sorry

end olympiad_scores_greater_than_18_l89_89595


namespace ratio_of_chickens_l89_89215

theorem ratio_of_chickens (initial_chickens : ℕ) (percent_died : ℚ) (total_chickens_after : ℕ) :
  initial_chickens = 400 →
  percent_died = 40 / 100 →
  total_chickens_after = 1840 →
  let chickens_died := percent_died * initial_chickens in
  let chickens_bought := total_chickens_after - (initial_chickens - chickens_died) in
  chickens_bought / chickens_died = 10 :=
by
  intros
  sorry

end ratio_of_chickens_l89_89215


namespace factor_tree_value_l89_89009

theorem factor_tree_value : 
  let W := 3 * 2,
      Z := 2 * W,
      Y := 7 * 11,
      X := Y * Z 
  in X = 924 := 
by
  -- We will include the necessary let bindings and provide a proof structure that aligns with Lean
  let W := 3 * 2,
      Z := 2 * W,
      Y := 7 * 11,
      X := Y * Z
  show X = 924 from
    sorry -- proof to be filled by the proof generation

end factor_tree_value_l89_89009


namespace total_volume_ice_cream_l89_89188

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

noncomputable def volume_of_hemisphere (r : ℝ) : ℝ :=
  (2 / 3) * π * r^3

theorem total_volume_ice_cream :
  let r := 3
  let h := 12
  volume_of_cone r h + volume_of_hemisphere r = 54 * π := sorry

end total_volume_ice_cream_l89_89188


namespace olympiad_scores_l89_89587

theorem olympiad_scores (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j : Fin 20, i < j → a i < a j)
  (h_condition : ∀ i j k : Fin 20, i ≠ j ∧ i ≠ k ∧ j ≠ k → a i < a j + a k) : 
  ∀ i : Fin 20, a i > 18 :=
by
  sorry

end olympiad_scores_l89_89587


namespace quadratic_real_roots_l89_89963

theorem quadratic_real_roots (m : ℝ) : 
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  (discriminant ≥ 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  sorry

end quadratic_real_roots_l89_89963


namespace tan_315_eq_neg1_l89_89376

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89376


namespace range_of_AC_l89_89601

theorem range_of_AC 
  (A B C : Type)
  (angle : A -> B -> C -> Real) 
  (BC : Real) 
  (BC_eq_1 : BC = 1) 
  (is_acute_triangle : angle B A C < π/2 ∧ angle A C B < π/2 ∧ angle C B A < π/2) 
  (angle_B_eq_2A : angle B A C = 2 * angle A C B): 
  (sqrt(2) < AC ∧ AC < sqrt(3)) :=
sorry

end range_of_AC_l89_89601


namespace remainder_of_large_number_l89_89954

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l89_89954


namespace remainder_when_divided_l89_89937

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l89_89937


namespace trig_identity_l89_89147

theorem trig_identity : (Real.cos (15 * Real.pi / 180))^2 - (Real.sin (15 * Real.pi / 180))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l89_89147


namespace remainder_of_large_number_l89_89949

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l89_89949


namespace triangle_area_l89_89458

def point (x y : ℝ) := (x, y)

/-- The area of the triangle formed by the points (0,0), (0,9), and (-9,0) is 40.5 square units. -/
theorem triangle_area :
  let A := point 0 0,
      B := point 0 9,
      C := point (-9) 0 in
  (1 / 2 : ℝ) * 9 * 9 = 40.5 :=
by
  -- Definitions to ensure the main formula is understood.
  let A := point 0 0
  let B := point 0 9
  let C := point (-9) 0
  -- Area calculation, intentionally left as sorry to indicate a proof is needed.
  sorry

end triangle_area_l89_89458


namespace prob_of_points_on_axes_l89_89026

-- Define the circle equation
def circle_eq (x y : ℚ) : Prop := (x - 6/5)^2 + y^2 = 36/25

-- Define the set of integer points on or inside the circle
def integer_points_on_or_inside_circle : finset (ℚ × ℚ) :=
  [(0,0), (1,-1), (1,0), (1,1), (2,0)].to_finset

-- Define the points on the coordinate axes
def points_on_axes : finset (ℚ × ℚ) :=
  [(0,0), (1,0), (2,0)].to_finset

-- Define the total number of ways to choose 2 points from 5
def total_combinations := nat.choose 5 2 -- 10

-- Define the number of ways to choose 2 points from the points on the axes
def favorable_combinations := nat.choose 3 2 -- 3

-- The probability is the ratio of favorable outcomes to total outcomes
def probability : ℚ := (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem prob_of_points_on_axes :
  probability = 3 / 10 := sorry

end prob_of_points_on_axes_l89_89026


namespace tan_315_degree_l89_89406

theorem tan_315_degree :
  let sin_45 := real.sin (45 * real.pi / 180)
  let cos_45 := real.cos (45 * real.pi / 180)
  let sin_315 := real.sin (315 * real.pi / 180)
  let cos_315 := real.cos (315 * real.pi / 180)
  sin_45 = cos_45 ∧ sin_45 = real.sqrt 2 / 2 ∧ cos_45 = real.sqrt 2 / 2 ∧ sin_315 = -sin_45 ∧ cos_315 = cos_45 → 
  real.tan (315 * real.pi / 180) = -1 :=
by
  intros
  sorry

end tan_315_degree_l89_89406


namespace washington_high_teacher_student_ratio_l89_89761

theorem washington_high_teacher_student_ratio (students teachers : ℕ) (h_students : students = 1155) (h_teachers : teachers = 42) : (students / teachers : ℚ) = 27.5 :=
by
  sorry

end washington_high_teacher_student_ratio_l89_89761


namespace container_volume_ratio_l89_89860

theorem container_volume_ratio
  (A B C : ℚ)  -- A is the volume of the first container, B is the volume of the second container, C is the volume of the third container
  (h1 : (8 / 9) * A = (7 / 9) * B)  -- Condition: First container was 8/9 full and second container gets filled to 7/9 after transfer.
  (h2 : (7 / 9) * B + (1 / 2) * C = C)  -- Condition: Mixing contents from second and third containers completely fill third container.
  : A / C = 63 / 112 := sorry  -- We need to prove this.

end container_volume_ratio_l89_89860


namespace at_least_one_closed_l89_89815

theorem at_least_one_closed {T V : Set ℤ} (hT : T.Nonempty) (hV : V.Nonempty) (h_disjoint : ∀ x, x ∈ T → x ∉ V)
  (h_union : ∀ x, x ∈ T ∨ x ∈ V)
  (hT_closed : ∀ a b c, a ∈ T → b ∈ T → c ∈ T → a * b * c ∈ T)
  (hV_closed : ∀ x y z, x ∈ V → y ∈ V → z ∈ V → x * y * z ∈ V) :
  (∀ a b, a ∈ T → b ∈ T → a * b ∈ T) ∨ (∀ x y, x ∈ V → y ∈ V → x * y ∈ V) := sorry

end at_least_one_closed_l89_89815


namespace number_of_reachable_cells_after_10_moves_l89_89639

theorem number_of_reachable_cells_after_10_moves : 
  (let 
    n := 21 
    center := (11, 11)
    moves := 10
  in
  ∃ reachable_cells, reachable_cells = 121) :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89639


namespace Mia_age_l89_89729

-- Let's define the context and conditions in Lean 4
def students_guesses : List ℕ := [26, 29, 33, 35, 37, 39, 43, 46, 50, 52]

def condition_1 (age : ℕ) : Prop :=
  age ∈ students_guesses ∧ Nat.prime age

def condition_2 (age : ℕ) : Prop :=
  List.filter (λ x => x < age) students_guesses = List.take (students_guesses.length / 2) students_guesses

def condition_3 (age : ℕ) : Prop :=
  List.filter (λ x => x = age - 1 ∨ x = age + 1) students_guesses = [age - 1, age + 1]

theorem Mia_age : (age : ℕ) (age ∈ students_guesses) (Nat.prime age)
  (List.filter (λ x => x < age) students_guesses = [26, 29, 33, 35, 37])
  (List.filter (λ x => x = age - 1 ∨ x = age + 1) students_guesses = [46, 50])
  : age = 47 := by
  sorry

end Mia_age_l89_89729


namespace bad_segments_even_l89_89165

noncomputable def polygon_chain_tangent_circle {A : Type} [DecidableEq A] [LinearOrder A] 
  (vertices : list A) (circle : set A) [IsTangent polygon chain] : Prop :=
∃ n (A_i : ℕ → A), n > 0 ∧ 
  ( ∀ i < n, (circle ∩ segment (A_i i) (A_i (i+1))) ≠ ∅) ∧ 
  ( ∀ i < n, 
    let seg : set A := segment (A_i i) (A_i (i + 1)) 
    in if (circle ∩ seg) = seg then seg is "good" else seg is "bad"
  ) 

theorem bad_segments_even
  (A : Type) [DecidableEq A] [LinearOrder A] 
  {vertices : list A} {circle : set A} 
  (h : ∃ n (A_i : ℕ → A), n > 0 ∧ 
    ( ∀ i < n, (circle ∩ segment (A_i i) (A_i (i+1))) ≠ ∅) ∧ 
    ( ∀ i < n, 
      let seg : set A := segment (A_i i) (A_i (i + 1)) 
      in if (circle ∩ seg) = seg then seg is "good" else seg is "bad"
    )
  ) : 
  ∃ m, m % 2 = 0 :=
sorry

end bad_segments_even_l89_89165


namespace cis_sum_theta_l89_89418

noncomputable def theta_sum : ℝ := 84

theorem cis_sum_theta :
  let angles := list.map (λ n : ℕ, complex.cis (real.of_nat (40 + 8 * n) * (real.pi / 180))) (list.range 12) in
  let sum_cis := list.sum angles in
  ∃ r > 0, θ, θ = theta_sum ∧ sum_cis = complex.cis (θ * real.pi / 180) * r :=
sorry

end cis_sum_theta_l89_89418


namespace ellipse_circle_proof_l89_89986

noncomputable def ellipse_equation (a b x y : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1) ∧ (2^2 / a^2 + (sqrt 2)^2 / b^2 = 1) ∧ (sqrt 6^2 / a^2 + 1^2 / b^2 = 1) ∧ (a > b) ∧ (b > 0)

noncomputable def circle_exists (a b : ℝ) : Prop := 
  ∃ r : ℝ, (r^2 = 8/3) ∧ ∀ (k m x1 x2 : ℝ), 
    ((1 + 2*k^2)*x1^2 + 4*k*m*x1 + 2*m^2 - 8 = 0) ∧ 
    (x1 + x2 = -4*k*m / (1 + 2*k^2)) ∧ 
    (x1*x2 = (2*m^2 - 8) / (1 + 2*k^2)) ∧ 
    ((k^2*x1*x2 + k*m*(x1 + x2) + m^2) = ((m^2 - 8*k^2) / (1 + 2*k^2))) ∧ 
    (x1 = -x2) → ∥r∥ = ∥x1 - x2∥

theorem ellipse_circle_proof :
  (∃ a b : ℝ, ellipse_equation a b 0 0 ∧ circle_exists a b) :=
sorry

end ellipse_circle_proof_l89_89986


namespace log_exp_eval_l89_89170

theorem log_exp_eval : 2 * log 5 10 + log 5 0.25 + (8 : ℝ)^(2 / 3) = 6 := by
  -- Sorry to skip the proof
  sorry

end log_exp_eval_l89_89170


namespace lewis_total_earnings_l89_89688

theorem lewis_total_earnings (earnings_per_week : ℕ) (weeks : ℕ) 
  (h1 : earnings_per_week = 2) (h2 : weeks = 89) : 
  earnings_per_week * weeks = 178 := 
by
  rw [h1, h2]
  norm_num
  sorry

end lewis_total_earnings_l89_89688


namespace proof_problem_l89_89465

def absolute_operation (s : List ℝ) : ℝ :=
  s.pairwise (|· - ·|).sum

def correct_statements_count : ℝ :=
  if 
    absolute_operation [1, 3, 5, 10] = 29 &&
    ¬ (∃ x : ℝ, absolute_operation [x, -2, 5] = 14) &&
    (count_simplified_expressions [a, b, b, c] = 6)
  then 1 else if 
    absolute_operation [1, 3, 5, 10] = 29 ||
    (∃ x : ℝ, absolute_operation [x, -2, 5] = 14) ||
    count_simplified_expressions [a, b, b, c] = 6
  then 0 else 0


theorem proof_problem :
  correct_statements_count = 1 :=
begin
  sorry
end

end proof_problem_l89_89465


namespace angle_PMX_eq_angle_AOI_l89_89974

-- Definitions representing the geometric setup and conditions
variables (A B C O I D M X P : Type)
variables (triangle_ABC : Triangle A B C)
variables (circumcenter_O : Circumcenter O triangle_ABC)
variables (incircle_I : Incircle I triangle_ABC)
variables (D_touches_BC : Touchpoint D incircle_I (Side B C))
variables (midpoint_M : Midpoint M (Segment A I))
variables (diameter_DX : Diameter DX incircle_I)
variables (foot_perpendicular_P : PerpendicularFoot P A DX)

-- The theorem stating the problem
theorem angle_PMX_eq_angle_AOI :
  ∠ PMX = ∠ AOI :=
by sorry

end angle_PMX_eq_angle_AOI_l89_89974


namespace tangent_line_angle_at_point_l89_89125

theorem tangent_line_angle_at_point : (y = (1 / 2) * x^2) →
(point : ℝ × ℝ) (h : point = (1, 1 / 2)) →
(derivative : ℝ → ℝ) (h' : ∀ x, derivative x = x) →
(tangent_slope : ℝ) (h'' : tangent_slope = derivative 1) →
(angle : ℝ) (h''' : tan angle = tangent_slope) →
(0 < angle ∧ angle < π) → angle = π / 4 :=
by
  intros
  sorry

end tangent_line_angle_at_point_l89_89125


namespace number_of_reachable_cells_after_10_moves_l89_89611

-- Define board size, initial position, and the number of moves
def board_size : ℕ := 21
def initial_position : ℕ × ℕ := (11, 11)
def moves : ℕ := 10

-- Define the main problem statement
theorem number_of_reachable_cells_after_10_moves :
  (reachable_cells board_size initial_position moves).card = 121 :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89611


namespace gini_coefficient_when_operating_separately_gini_coefficient_change_after_combination_l89_89578

section GiniCalculation

-- Conditions (definitions of populations and production functions)
def n_population := 24
def s_population := n_population / 4
def n_ppf (x : ℝ) := 13.5 - 9 * x
def s_ppf (x : ℝ) := 1.5 * x^2 - 24
def set_price := 2000
def set_y_to_x_ratio := 9

-- questions
def question1 := "What is the Gini coefficient when both regions operate separately?"
def question2 := "How does the Gini coefficient change if the southern region agrees to the conditions of the northern region?"

-- Propositions (mathematically equivalent problems)
def proposition1 : Prop :=
  calc_gini_coefficient (regions_operate_separately n_population s_population n_ppf s_ppf set_y_to_x_ratio set_price) = 0.2

def proposition2 : Prop :=
  calc_gini_change_after_combination (combine_production_resources n_population s_population n_ppf s_ppf set_y_to_x_ratio set_price 661) = 0.001

noncomputable def regions_operate_separately (n_pop s_pop : ℝ) 
  (n_ppf s_ppf : ℝ → ℝ) (ratio price : ℝ) := sorry

noncomputable def combine_production_resources 
  (n_pop s_pop : ℝ) (n_ppf s_ppf : ℝ → ℝ)
  (ratio price : ℝ) (fee : ℝ) := sorry

noncomputable def calc_gini_coefficient : ℝ := sorry

noncomputable def calc_gini_change_after_combination : ℝ := sorry

-- Lean 4 Statements (no proof provided)
theorem gini_coefficient_when_operating_separately : proposition1 := by sorry
theorem gini_coefficient_change_after_combination : proposition2 := by sorry

end GiniCalculation

end gini_coefficient_when_operating_separately_gini_coefficient_change_after_combination_l89_89578


namespace quadratic_real_roots_l89_89961

theorem quadratic_real_roots (m : ℝ) : 
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  (discriminant ≥ 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  sorry

end quadratic_real_roots_l89_89961


namespace find_obtuse_angle_l89_89186

open Real

structure EquilateralTriangle :=
  (A B C : Point)
  (eq_sides : dist A B = dist B C ∧ dist B C = dist C A)

def ray_divides_base (A L : Point) (B C : Point) (m n : ℝ) : Prop :=
  let BL := dist B L
  let LC := dist L C
  BL / LC = m / n

def obtuse_angle (A L : Point) (B C : Point) (theta : ℝ) : Prop :=
  angle A L B = π - θ

theorem find_obtuse_angle (ABC : EquilateralTriangle) (L : Point) (m n : ℝ)
  (ray_div : ray_divides_base ABC.A L ABC.B ABC.C m n) :
  ∃ θ, obtuse_angle ABC.A L ABC.B ABC.C (atan ( (m + n) * sqrt 3 / (m - n) )) := by
  sorry

end find_obtuse_angle_l89_89186


namespace students_scoring_above_110_l89_89832

theorem students_scoring_above_110 
  (students : ℕ)
  (mu sigma : ℝ)
  (normal_dist : measure_theory.measure_space (ℝ → ℝ))
  (prob_range : ℝ)
  (total_students : students = 50)
  (mean : mu = 100)
  (std_dev : sigma = 10)
  (prob_90_to_100 : prob_range = 0.3) :
  ∃ n, n = 10 ∧ n = students * 0.2 :=
sorry

end students_scoring_above_110_l89_89832


namespace ratio_of_work_completed_by_a_l89_89152

theorem ratio_of_work_completed_by_a (A B W : ℝ) (ha : (A + B) * 6 = W) :
  (A * 3) / W = 1 / 2 :=
by 
  sorry

end ratio_of_work_completed_by_a_l89_89152


namespace tetrahedron_sphere_relations_l89_89668

theorem tetrahedron_sphere_relations 
  (ρ ρ1 ρ2 ρ3 ρ4 m1 m2 m3 m4 : ℝ)
  (hρ_pos : ρ > 0)
  (hρ1_pos : ρ1 > 0)
  (hρ2_pos : ρ2 > 0)
  (hρ3_pos : ρ3 > 0)
  (hρ4_pos : ρ4 > 0)
  (hm1_pos : m1 > 0)
  (hm2_pos : m2 > 0)
  (hm3_pos : m3 > 0)
  (hm4_pos : m4 > 0) : 
  (2 / ρ = 1 / ρ1 + 1 / ρ2 + 1 / ρ3 + 1 / ρ4) ∧
  (1 / ρ = 1 / m1 + 1 / m2 + 1 / m3 + 1 / m4) ∧
  ( 1 / ρ1 = -1 / m1 + 1 / m2 + 1 / m3 + 1 / m4 ) := sorry

end tetrahedron_sphere_relations_l89_89668


namespace find_x_of_line_segment_l89_89837

theorem find_x_of_line_segment (x : ℝ) (h1: real.sqrt ((x - 2)^2 + (7 - 1)^2) = 8) (h2 : x > 0) : x = 2 + 2 * real.sqrt 7 :=
sorry

end find_x_of_line_segment_l89_89837


namespace find_k_l89_89528

noncomputable def enclosed_area (k : ℝ) : ℝ :=
  ∫ x in 0..k, k * x - x^2

theorem find_k (k : ℝ) (h : k > 0) (area_eq : enclosed_area k = 9 / 2) : k = 3 := by
  sorry

end find_k_l89_89528


namespace find_n_l89_89149

theorem find_n (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) = 14) : n = 2 :=
sorry

end find_n_l89_89149


namespace geom_seq_frac_l89_89762

variable {a b : ℝ}
variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}

def geometric_sum (n : ℕ) : ℝ := b * (-2)^(n-1) - a

theorem geom_seq_frac : (∀ n, (∑ i in Finset.range n, a_n i) = geometric_sum n)
                        → a / b = -1 / 2 :=
by
  sorry

end geom_seq_frac_l89_89762


namespace males_in_age_brackets_correct_l89_89582

variables (total_employees : ℕ) (pct_males : ℝ) (pct_18_29 pct_30_39 pct_40_49 pct_50_59 pct_60_above : ℝ)

def num_males := total_employees * pct_males

def num_18_29 := num_males * pct_18_29
def num_30_39 := num_males * pct_30_39
def num_40_49 := num_males * pct_40_49
def num_50_59 := num_males * pct_50_59
def num_60_above := num_males * pct_60_above

theorem males_in_age_brackets_correct :
  total_employees = 4200 →
  pct_males = 0.35 →
  pct_18_29 = 0.10 →
  pct_30_39 = 0.25 →
  pct_40_49 = 0.35 →
  pct_50_59 = 0.20 →
  pct_60_above = 0.10 →
  num_males = 1470 ∧
  num_18_29 = 147 ∧
  num_30_39 = 367 ∧
  num_40_49 = 515 ∧
  num_50_59 = 294 ∧
  num_60_above = 147 :=
by {
  intros,
  simp [num_males, num_18_29, num_30_39, num_40_49, num_50_59, num_60_above],
  norm_num,
  split; norm_num,
  split; norm_num,
  split; norm_num,
  split; norm_num,
  split; norm_num,
  norm_num,
  sorry
}

end males_in_age_brackets_correct_l89_89582


namespace percentage_decrease_l89_89569

theorem percentage_decrease (x y z : ℝ) (h1 : x = 1.30 * y) (h2 : x = 0.65 * z) : 
  ((z - y) / z) * 100 = 50 :=
by
  sorry

end percentage_decrease_l89_89569


namespace sales_proof_l89_89176

noncomputable def sale_in_second_month (a1 a2 a3 a4 a5 a6 : ℕ) : ℕ :=
  a2

theorem sales_proof :
  ∀ (a1 a3 a4 a5 a6 : ℕ),
  (a1 = 6435) →
  (a3 = 6855) →
  (a4 = 7230) →
  (a5 = 6562) →
  (a6 = 4991) →
  (6500 * 6 = a1 + 13782 + a3 + a4 + a5 + a6) →
  (sale_in_second_month a1 13782 a3 a4 a5 a6 = 13782) :=
by
  intros a1 a3 a4 a5 a6 hA1 hA3 hA4 hA5 hA6 hTotal
  rw [sale_in_second_month]
  apply rfl

end sales_proof_l89_89176


namespace reachable_cells_after_10_moves_l89_89626

def adjacent_cells (x y : ℕ) : set (ℕ × ℕ) :=
  { (x', y') | (x' = x + 1 ∧ y' = y) ∨ (x' = x - 1 ∧ y' = y) 
            ∨ (x' = x ∧ y' = y + 1) ∨ (x' = x ∧ y' = y - 1) }

def in_bounds (x y : ℕ) : Prop :=
  x > 0 ∧ x ≤ 21 ∧ y > 0 ∧ y ≤ 21

theorem reachable_cells_after_10_moves : 
  ∃ cells : set (ℕ × ℕ), ∃ initial_position : (11, 11) ∈ cells ∧ 
  (∀ (x y : ℕ), (x, y) ∈ cells → in_bounds x y ∧
  (∀ n ≤ 10, (x', y') ∈ adjacent_cells x y → (x', y') ∈ cells)) ∧ 
  (set.card cells = 121) :=
sorry

end reachable_cells_after_10_moves_l89_89626


namespace tan_315_eq_neg1_l89_89248

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89248


namespace prf_p1_ge_p2_prf_p3_eq_100p0_prf_p1_le_100p2_l89_89698

-- Define the sound pressure levels using the provided formula
def Lp (p p₀ : ℝ) : ℝ := 20 * Real.log10 (p / p₀)

-- Given constants and conditions
variables (p₀ p₁ p₂ p₃ : ℝ) (h₀ : 0 < p₀)
variables (h₁ : 10^3 * p₀ ≤ p₁ ∧ p₁ ≤ 10^(9/2) * p₀)
variables (h₂ : 10^(5/2) * p₀ ≤ p₂ ∧ p₂ ≤ 10^3 * p₀)
variables (h₃ : p₃ = 10^2 * p₀)

-- Proof statements to be shown
theorem prf_p1_ge_p2 : p₁ ≥ p₂ :=
sorry

theorem prf_p3_eq_100p0 : p₃ = 100 * p₀ :=
sorry

theorem prf_p1_le_100p2 : p₁ ≤ 100 * p₂ :=
sorry

end prf_p1_ge_p2_prf_p3_eq_100p0_prf_p1_le_100p2_l89_89698


namespace inequality_proof_for_any_real_l89_89712

theorem inequality_proof_for_any_real (x y : ℝ) : 
  x^2 * real.sqrt(1 + 2 * y^2) + y^2 * real.sqrt(1 + 2 * x^2) >= x * y * (x + y + real.sqrt(2)) :=
  sorry

end inequality_proof_for_any_real_l89_89712


namespace tan_315_eq_neg1_l89_89330

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89330


namespace reflection_over_line_y_eq_x_l89_89455

noncomputable def reflection_matrix_over_y_eq_x : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1;
     1, 0]

theorem reflection_over_line_y_eq_x (x y : ℝ) :
  (reflection_matrix_over_y_eq_x.mul_vec ![x, y]) = ![y, x] :=
by sorry

end reflection_over_line_y_eq_x_l89_89455


namespace cube_prob_sum_numerator_denominator_l89_89118

noncomputable def cube_probability : ℚ :=
  -- Number of valid configurations where no two consecutive numbers are adjacent
  let valid_configs := 12 
  -- Total number of configurations for labeling the cube faces
  let total_configs := 6.factorial 
  -- The desired probability
  (valid_configs : ℚ) / total_configs

theorem cube_prob_sum_numerator_denominator :
  cube_probability.numerator + cube_probability.denominator = 61 :=
by
  rw [cube_probability, Rat.numerator, Rat.denominator]
  have fact_6_val : 6.factorial = 720 := by sorry
  rw [fact_6_val]
  norm_num
  -- manually checking the computation: (12 / 720).num + (12 / 720).denom = 1 + 60
  rw [Rat.add, Rat.num, Rat.denom]
  exact calc
    12 + 720 - 720 + 1 - 1 = 61 : by sorry
    12 / 720 = 1 / 60 : by norm_num

end cube_prob_sum_numerator_denominator_l89_89118


namespace lines_not_parallel_for_a_eq_3_l89_89432

theorem lines_not_parallel_for_a_eq_3 : 
∀ a, ¬(a = 3 → (∃ (k : ℝ), a/3 = k ∧ 2/(a-1) = k ∧ 3a ≠ k * (a^2-a+3))) :=
by
  intro a
  sorry

end lines_not_parallel_for_a_eq_3_l89_89432


namespace identity_proof_l89_89089

theorem identity_proof (n : ℝ) (h1 : n^2 ≥ 4) (h2 : n ≠ 0) :
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) - 2) / 
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) + 2)
    = ((n + 1) * Real.sqrt (n - 2)) / ((n - 1) * Real.sqrt (n + 2)) := by
  sorry

end identity_proof_l89_89089


namespace vec_add_eq_l89_89535

noncomputable section

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

theorem vec_add_eq 
  (m : ℝ) 
  (h_parallel : 1 * m = 2 * (-2)) : 
  3 • vector_a + 2 • (vector_b m) = (-1 : ℝ, -2) :=
by
  sorry

end vec_add_eq_l89_89535


namespace determine_a_plus_b_l89_89670

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b
noncomputable def f_inv (a b x : ℝ) : ℝ := b * x^2 + a

theorem determine_a_plus_b (a b : ℝ) (h: ∀ x : ℝ, f a b (f_inv a b x) = x) : a + b = 1 :=
sorry

end determine_a_plus_b_l89_89670


namespace B_can_complete_work_l89_89802

theorem B_can_complete_work (
  A_days_complete : ℕ,  -- The number of days A can complete the work
  B_days_complete : ℕ,  -- The number of days B needs to complete the work
  A_work_done : ℚ,      -- Fraction of work A completes in given time
  B_work_done : ℚ       -- Fraction of work B completes in given time
) (h1 : A_days_complete = 15)
  (h2 : B_work_done = (10 : ℚ) / (2 / 3))
  (h3 : A_work_done = (5 : ℚ) / 15)
  (h4 : A_work_done + 2 / 3 = 1) :
  B_days_complete = 15 := by
  sorry

end B_can_complete_work_l89_89802


namespace kim_saplings_left_l89_89045

def sprouted_pits (total_pits num_sprouted_pits: ℕ) (percent_sprouted: ℝ) : Prop :=
  percent_sprouted * total_pits = num_sprouted_pits

def sold_saplings (total_saplings saplings_sold saplings_left: ℕ) : Prop :=
  total_saplings - saplings_sold = saplings_left

theorem kim_saplings_left
  (total_pits : ℕ) (num_sprouted_pits : ℕ) (percent_sprouted : ℝ)
  (saplings_sold : ℕ) (saplings_left : ℕ) :
  total_pits = 80 →
  percent_sprouted = 0.25 →
  saplings_sold = 6 →
  sprouted_pits total_pits num_sprouted_pits percent_sprouted →
  sold_saplings num_sprouted_pits saplings_sold saplings_left →
  saplings_left = 14 :=
by
  intros
  sorry

end kim_saplings_left_l89_89045


namespace solution_set_of_f_inequality_l89_89743

open Real

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f(x) = -f(-x)

theorem solution_set_of_f_inequality
  (f : ℝ → ℝ)
  (h_odd : is_odd_function (λ x, f(x + 1)))
  (h_monotone : ∀ x₁ x₂, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0) :
  {x : ℝ | f(1 - x) < 0} = set.Iio 0 :=
sorry

end solution_set_of_f_inequality_l89_89743


namespace tan_difference_l89_89998

theorem tan_difference (x : Real) 
  (h : sin ((π / 3) - x) = (1 / 2) * cos (x - (π / 2))) :
  tan (x - (π / 6)) = (√3 / 9) := 
by
  sorry

end tan_difference_l89_89998


namespace tan_315_eq_neg1_l89_89343

def Q : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem tan_315_eq_neg1 : real.tan (315 * real.pi / 180) = -1 := 
by {
  sorry
}

end tan_315_eq_neg1_l89_89343


namespace jason_total_spent_l89_89656

-- Conditions
def shorts_cost : ℝ := 14.28
def jacket_cost : ℝ := 4.74

-- Statement to prove
theorem jason_total_spent : shorts_cost + jacket_cost = 19.02 := by
  -- Proof to be filled in
  sorry

end jason_total_spent_l89_89656


namespace min_edges_in_graph_l89_89093

theorem min_edges_in_graph (G : SimpleGraph V) (H : Fintype V) (hv : Fintype.card V = 19998)
  (hsubgraph : ∀ (V' : Finset V), V'.card = 9999 → (G.subgraph (G.induced V')).edgeFinset.card ≥ 9999) :
  ∃ (e : ℕ), (G.edgeFinset.card = 49995) :=
begin
  sorry
end

end min_edges_in_graph_l89_89093


namespace toy_train_produces_5_consecutive_same_tune_l89_89193

noncomputable def probability_same_tune (plays : ℕ) (p : ℚ) (tunes : ℕ) : ℚ :=
  p ^ plays

theorem toy_train_produces_5_consecutive_same_tune :
  probability_same_tune 5 (1/3) 3 = 1/243 :=
by
  sorry

end toy_train_produces_5_consecutive_same_tune_l89_89193


namespace calculate_variance_l89_89737

noncomputable def expectation (X : ℕ → ℝ) (P : ℕ → ℝ) : ℝ :=
∑ i, X i * P i

noncomputable def variance (X : ℕ → ℝ) (P : ℕ → ℝ) : ℝ :=
expectation (fun i => (X i) ^ 2) P - (expectation X P) ^ 2

theorem calculate_variance (X : ℕ → ℝ) (P : ℕ → ℝ)
  (h_dist : (P 0 = 0.3) ∧ (P 1 = 0.2) ∧ (P 2 = 0.4) ∧ (P 3 = 0.1))
  (h_sum : ∑ i in Finset.range 4, P i = 1)
  (h_exp : expectation X P = 15 / 8) :
  variance X P = 55 / 64 :=
by sorry

end calculate_variance_l89_89737


namespace area_square_is_41_l89_89849
open Real

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (5, parabola 5)

-- Define the side length of the square
def side_length : ℝ := sqrt ((vertex.1 - 0)^2 + (vertex.2 - 0)^2)

-- Define the area of the square
def area_square : ℝ := side_length^2

-- Prove that the area of the square is 41
theorem area_square_is_41 : area_square = 41 :=
by
sry sorry

end area_square_is_41_l89_89849


namespace perpendicular_vector_zero_dot_product_l89_89534

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (4, -2)
def λ := -1

theorem perpendicular_vector_zero_dot_product
  (λ : ℝ)
  (a b : ℝ × ℝ)
  (h₁ : a = (1, -3))
  (h₂ : b = (4, -2))
  (h₃ : (λ • a + b) • a = 0) :
  λ = -1 :=
by
  sorry

end perpendicular_vector_zero_dot_product_l89_89534


namespace combined_tax_rate_l89_89806

-- Definitions of the problem conditions
def tax_rate_Mork : ℝ := 0.40
def tax_rate_Mindy : ℝ := 0.25

-- Asserts the condition that Mindy earned 4 times as much as Mork
def income_ratio (income_Mindy income_Mork : ℝ) := income_Mindy = 4 * income_Mork

-- The theorem to be proved: The combined tax rate is 28%.
theorem combined_tax_rate (income_Mork income_Mindy total_income total_tax : ℝ)
  (h_income_ratio : income_ratio income_Mindy income_Mork)
  (total_income_eq : total_income = income_Mork + income_Mindy)
  (total_tax_eq : total_tax = tax_rate_Mork * income_Mork + tax_rate_Mindy * income_Mindy) :
  total_tax / total_income = 0.28 := sorry

end combined_tax_rate_l89_89806


namespace tan_315_eq_neg1_l89_89372

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l89_89372


namespace tan_315_eq_neg_one_l89_89292

theorem tan_315_eq_neg_one : real.tan (315 * real.pi / 180) = -1 := by
  -- Definitions based on the conditions
  let Q := ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩
  have ref_angle : 315 = 360 - 45 := sorry
  have coordinates_of_Q : Q = ⟨real.sqrt 2 / 2, - real.sqrt 2 / 2⟩ := sorry
  have Q_x := real.sqrt 2 / 2
  have Q_y := - real.sqrt 2 / 2
  -- Proof
  sorry

end tan_315_eq_neg_one_l89_89292


namespace sum_and_count_evens_20_30_l89_89158

def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_evens_20_30 :
  let x := sum_integers 20 30
  let y := count_even_integers 20 30
  x + y = 281 :=
by
  sorry

end sum_and_count_evens_20_30_l89_89158


namespace remainder_of_large_number_l89_89956

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l89_89956


namespace tan_315_eq_neg1_l89_89260

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg1_l89_89260


namespace smallest_value_x_l89_89795

theorem smallest_value_x : 
  (∃ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 6 ∧ 
  (∀ y : ℝ, ((5*y - 20)/(4*y - 5))^2 + ((5*y - 20)/(4*y - 5)) = 6 → x ≤ y)) → 
  x = 35 / 17 :=
by 
  sorry

end smallest_value_x_l89_89795


namespace reflection_matrix_over_y_equals_x_l89_89447

theorem reflection_matrix_over_y_equals_x :
  let reflection_matrix := λ (v : ℝ × ℝ), (v.2, v.1) in
  (∀ v, reflection_matrix v = (v.2, v.1) →
      ∃ M : Matrix (Fin 2) (Fin 2) ℝ,
      ∀ v : Matrix (Fin 2) (Fin 1) ℝ,
      M ⬝ v = reflection_matrix (v 0 0, v 1 0) ⬝) :=
begin
  sorry
end

end reflection_matrix_over_y_equals_x_l89_89447


namespace sufficient_not_necessary_l89_89800

theorem sufficient_not_necessary (a : ℝ) :
  (∃ x₀ : ℝ, a * Real.cos x₀ + 1 < 0) ↔ (a < -1) :=
by
  split
  {
    -- proving sufficiency:
    intro h
    have h₀ : Real.cos 0 = 1 := by sorry -- proof that cos(0) = 1
    specialize h 0
    calc a * Real.cos 0 + 1 = a * 1 + 1 := by sorry
                      ... < 0          := by sorry -- the given condition
  }
  {
    -- proving necessity:
    intros h
    have h₁ : -1 ≤ Real.cos x₀ ∧ Real.cos x₀ ≤ 1 := by sorry -- the range of cos(x)
    split_ifs
    {
      -- analyzing different cases for a (0, positive, and negative)

    }
  }
  sorry

end sufficient_not_necessary_l89_89800


namespace olympiad_scores_l89_89585

theorem olympiad_scores (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j : Fin 20, i < j → a i < a j)
  (h_condition : ∀ i j k : Fin 20, i ≠ j ∧ i ≠ k ∧ j ≠ k → a i < a j + a k) : 
  ∀ i : Fin 20, a i > 18 :=
by
  sorry

end olympiad_scores_l89_89585


namespace reflection_over_line_y_eq_x_l89_89453

noncomputable def reflection_matrix_over_y_eq_x : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1;
     1, 0]

theorem reflection_over_line_y_eq_x (x y : ℝ) :
  (reflection_matrix_over_y_eq_x.mul_vec ![x, y]) = ![y, x] :=
by sorry

end reflection_over_line_y_eq_x_l89_89453


namespace least_positive_t_arithmetic_progression_l89_89882

theorem least_positive_t_arithmetic_progression (α : ℝ) (hα1 : 0 < α) (hα2 : α < (Real.pi/2)) :
  ∃ (t : ℝ), (0 < t) ∧ (∀ n, arccos (cos(α * n)) = α * n) ∧ t = 7 := 
by 
  sorry

end least_positive_t_arithmetic_progression_l89_89882


namespace scores_greater_than_18_l89_89591

noncomputable def olympiad_scores (scores : Fin 20 → ℕ) :=
∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k

theorem scores_greater_than_18 (scores : Fin 20 → ℕ) (h1 : ∀ i j, i < j → scores i < scores j)
  (h2 : olympiad_scores scores) : ∀ i, 18 < scores i :=
by
  intro i
  sorry

end scores_greater_than_18_l89_89591


namespace tangent_315_deg_l89_89287

theorem tangent_315_deg : Real.tan (315 * (Real.pi / 180)) = -1 :=
by
  sorry

end tangent_315_deg_l89_89287


namespace tan_315_eq_neg1_l89_89391

-- Definitions based on conditions
def angle_315 := 315 * Real.pi / 180  -- 315 degrees in radians
def angle_45 := 45 * Real.pi / 180    -- 45 degrees in radians
def cos_45 := Real.sqrt 2 / 2         -- cos 45 = √2 / 2
def sin_45 := Real.sqrt 2 / 2         -- sin 45 = √2 / 2
def cos_315 := cos_45                 -- cos 315 = cos 45
def sin_315 := -sin_45                -- sin 315 = -sin 45

-- Statement to prove
theorem tan_315_eq_neg1 : Real.tan angle_315 = -1 := by
  -- All definitions should be present and useful within this proof block
  sorry

end tan_315_eq_neg1_l89_89391


namespace probability_of_at_least_one_machine_operating_l89_89022

variables (t : ℝ)
variables (P_A1 P_A2 : ℝ) (P_B : ℝ)
variables (A1 A2 B : Prop)

-- Define probabilities
def P (A : Prop) : ℝ := sorry
def P_not (A : Prop) : ℝ := 1 - P A

-- Conditions
axiom cond1 : P A1 = 0.9
axiom cond2 : P A2 = 0.8
axiom independence : P_not A1 * P_not A2 = P_not B

-- Proof Problem
theorem probability_of_at_least_one_machine_operating : 
  P B = 0.98 :=
by 
  simp only [P, P_not] at independence,
  sorry

end probability_of_at_least_one_machine_operating_l89_89022


namespace find_k_l89_89496

noncomputable def sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  (∀ n : ℕ, 1 ≤ n → a (n + 1) = 3 * S n) ∧
  (∀ n : ℕ, 1 ≤ n → S n = ∑ i in Finset.range (n + 1), a i)

theorem find_k (a S : ℕ → ℕ) (k : ℕ) :
  sequence a S →
  (750 < a k ∧ a k < 900) → k = 6 :=
by
  intro h1 h2
  sorry

end find_k_l89_89496


namespace tan_315_eq_neg_1_l89_89327

theorem tan_315_eq_neg_1 : 
  (real.angle.tan (real.angle.of_deg 315) = -1) := by
{
  sorry
}

end tan_315_eq_neg_1_l89_89327


namespace child_patients_per_hour_l89_89796

-- Define the constants and variables according to the problem description
def daily_revenue (C : ℕ) : ℕ :=
  8 * (200 + 25 * C)

theorem child_patients_per_hour :
  ∃ C : ℕ, daily_revenue C = 2200 ∧ C = 3 :=
by
  exists 3
  split
  case left =>
    unfold daily_revenue
    norm_num
  case right =>
    rfl


end child_patients_per_hour_l89_89796


namespace quadratic_eq_real_roots_l89_89494

theorem quadratic_eq_real_roots (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x^2 + 2 * x - 1 = 0 → ∃ x : ℝ, (m - 2) * x^2 + 2 * x - 1 = 0 → x ∈ ℝ) → m ≥ 1 :=
by
  -- Proof is omitted
  sorry

end quadratic_eq_real_roots_l89_89494


namespace probability_at_least_one_woman_in_selection_l89_89557

theorem probability_at_least_one_woman_in_selection :
  ∃ (P : ℚ), P = 85 / 99 :=
by 
  -- Define variables
  let total_people := 12
  let men := 8
  let women := 4
  let selection := 4

  -- Calculate the probability of selecting four men
  let P_all_men := (men / total_people) * ((men - 1) / (total_people - 1)) *
                   ((men - 2) / (total_people - 2)) *
                   ((men - 3) / (total_people - 3))

  -- Calculate the probability of at least one woman being selected
  let P_at_least_one_woman := 1 - P_all_men

  -- Verify the result
  have H : P_at_least_one_woman = 85 / 99 := sorry
  use P_at_least_one_woman
  exact H

end probability_at_least_one_woman_in_selection_l89_89557


namespace math_olympiad_scores_l89_89596

theorem math_olympiad_scores (a : Fin 20 → ℕ) 
  (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → j ≠ k → i ≠ k → a i < a j + a k) :
  ∀ i : Fin 20, a i > 18 := 
sorry

end math_olympiad_scores_l89_89596


namespace marie_finishes_ninth_task_at_730PM_l89_89690

noncomputable def start_time : ℕ := 8 * 60 -- 8:00 AM in minutes
noncomputable def end_time_task_3 : ℕ := 11 * 60 + 30 -- 11:30 AM in minutes
noncomputable def total_tasks : ℕ := 9
noncomputable def tasks_done_by_1130AM : ℕ := 3
noncomputable def end_time_task_9 : ℕ := 19 * 60 + 30 -- 7:30 PM in minutes

theorem marie_finishes_ninth_task_at_730PM
    (h1 : start_time = 480) -- 8:00 AM
    (h2 : end_time_task_3 = 690) -- 11:30 AM
    (h3 : total_tasks = 9)
    (h4 : tasks_done_by_1130AM = 3)
    (h5 : end_time_task_9 = 1170) -- 7:30 PM
    : end_time_task_9 = start_time + ((end_time_task_3 - start_time) / tasks_done_by_1130AM) * total_tasks :=
sorry

end marie_finishes_ninth_task_at_730PM_l89_89690


namespace exists_a_with_signum_product_l89_89679

variable (f : ℝ → ℝ)

theorem exists_a_with_signum_product {
  continuous_third_derivative : ∀ x : ℝ, Continuous (adsymbol.prefix_third_derivative x)
}: 
∃ a : ℝ, f a * deriv f a * deriv (deriv f) a * deriv (deriv (deriv f)) a ≥ 0 :=
by
  sorry

end exists_a_with_signum_product_l89_89679


namespace olympiad_scores_greater_than_18_l89_89592

open Classical

theorem olympiad_scores_greater_than_18 (n : ℕ) (a : ℕ → ℕ) (h_distinct: ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_ordered: ∀ i j: ℕ, i < j → a i < a j)
  (h_condition: ∀ i j k: ℕ, i ≠ j → i ≠ k → j ≠ k → a i < a j + a k) :
  ∀ i < n, n = 20 ∧ a i > 18 :=
by
  assume i h_i_lt_n h_n_eq_20
  sorry

end olympiad_scores_greater_than_18_l89_89592


namespace blood_expiration_date_blood_expiration_final_date_l89_89195

theorem blood_expiration_date :
  let seconds_per_day := 86400
  let jan_days := 31
  let ten_fact := 10.factorial
  let total_days := ten_fact / seconds_per_day
  (jan_days + total_days) = 42 :=
by
  let seconds_per_day := 86400
  let jan_days := 31
  let ten_fact := 10.factorial
  let total_days := ten_fact / seconds_per_day
  exact Nat.div_add_div 10.factorial 86400 sorry -- where the modular operation ensures the division approximation
  let end_day := jan_days + total_days
  exact end_day = 42

-- Sanity check: Complete the theorem to see the overall goal
theorem blood_expiration_final_date :
  let days_in_jan := 31
  let days_after_jan := 42 - days_in_jan
  days_after_jan = 11 :=
by
  let days_in_jan := 31
  let days_after_jan := 42 - days_in_jan
  exact days_after_jan = 11

end blood_expiration_date_blood_expiration_final_date_l89_89195


namespace number_of_packages_sold_l89_89864

variable (P N : ℕ)
variable (price_per_package reduced_price total_payment first_ten_packages_cost excess_packages : ℕ)

def auto_parts_supplier_conditions : Prop :=
  let price_per_package := 20
  let reduced_price := 4 * price_per_package / 5
  let total_payment := 1096
  let first_ten_packages_cost := 10 * price_per_package
  let excess_packages := total_payment - first_ten_packages_cost
  N = excess_packages / reduced_price

theorem number_of_packages_sold :
  auto_parts_supplier_conditions P N →
  P = 10 + N :=
by
  sorry

end number_of_packages_sold_l89_89864


namespace kim_saplings_left_l89_89043

def number_of_pits : ℕ := 80
def proportion_sprout : ℚ := 0.25
def saplings_sold : ℕ := 6

theorem kim_saplings_left : 
  (number_of_pits * proportion_sprout - saplings_sold = 14) :=
begin
  sorry
end

end kim_saplings_left_l89_89043


namespace gini_coefficient_separate_gini_coefficient_combined_l89_89577

-- Definitions based on provided conditions
def northern_residents : ℕ := 24
def southern_residents : ℕ := 6
def price_per_set : ℝ := 2000
def northern_PPC (x : ℝ) : ℝ := 13.5 - 9 * x
def southern_PPC (x : ℝ) : ℝ := 1.5 * x^2 - 24

-- Gini Coefficient when both regions operate separately
theorem gini_coefficient_separate : 
  ∃ G : ℝ, G = 0.2 :=
  sorry

-- Gini Coefficient change when blending productions as per Northern conditions
theorem gini_coefficient_combined :
  ∃ ΔG : ℝ, ΔG = 0.001 :=
  sorry

end gini_coefficient_separate_gini_coefficient_combined_l89_89577


namespace remainder_when_divided_l89_89933

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l89_89933


namespace trains_crossing_time_l89_89136

-- Definitions for lengths and speeds
def length_train_a : ℝ := 200
def length_train_b : ℝ := 180
def speed_train_a_kmph : ℝ := 40
def speed_train_b_kmph : ℝ := 45

-- Conversions and calculations
def speed_train_a_mps : ℝ := speed_train_a_kmph * (1000 / 3600)
def speed_train_b_mps : ℝ := speed_train_b_kmph * (1000 / 3600)
def relative_speed_mps : ℝ := speed_train_b_mps - speed_train_a_mps
def total_length : ℝ := length_train_a + length_train_b
def crossing_time_seconds : ℝ := total_length / relative_speed_mps
def crossing_time_minutes : ℝ := crossing_time_seconds / 60

theorem trains_crossing_time :
  length_train_a = 200 → length_train_b = 180 →
  speed_train_a_kmph = 40 → speed_train_b_kmph = 45 →
  (crossing_time_minutes ≈ 4.56) :=
by
  intros h1 h2 h3 h4
  sorry

end trains_crossing_time_l89_89136


namespace at_least_one_woman_probability_l89_89549

noncomputable def probability_at_least_one_woman_selected 
  (total_men : ℕ) (total_women : ℕ) (selected_people : ℕ) : ℚ :=
  1 - (8 / 12 * 7 / 11 * 6 / 10 * 5 / 9)

theorem at_least_one_woman_probability :
  probability_at_least_one_woman_selected 8 4 4 = 85 / 99 := 
sorry

end at_least_one_woman_probability_l89_89549


namespace cartesian_equation_of_C_min_AB_distance_l89_89732

-- Define the parametric equation of the line l
def parametric_line := 
  {x : ℝ // ∃ t α : ℝ, 0 < α ∧ α < π ∧ x = 1 + t * cos α}

-- Define the polar equation of the curve C
def polar_curve := 
  {ρ : ℝ // ∃ θ : ℝ, ρ * (sin θ)^2 = 4 * cos θ}

theorem cartesian_equation_of_C : ∀ (x y : ℝ), 
  (polar_curve.1)^2 = 4 * polar_curve.1 → y^2 = 4 * x := 
by sorry

theorem min_AB_distance : ∀ (α : ℝ), 
  (∃ A B : (ℝ × ℝ), parametric_line.1 = A.fst ∧ parametric_line.1 = B.fst ∧ polar_curve.1 = A.snd ∧ polar_curve.1 = B.snd) → 
  ∃ α_min : ℝ, α_min = π/2 ∧ |(A.fst - B.fst)| = 4 := 
by sorry

end cartesian_equation_of_C_min_AB_distance_l89_89732


namespace Helly_Theorem_l89_89977

-- Declare the problem in Lean
theorem Helly_Theorem (n : ℕ) (figures : Fin n → Set (ℝ × ℝ))
  (h_convex : ∀ i, Convex (figures i))
  (h_bounded : ∀ i, Bounded (figures i))
  (h_common_point : ∀ (i j k : Fin n), (figures i ∩ figures j ∩ figures k).Nonempty) :
  (⋂ i : Fin n, figures i).Nonempty :=
sorry

end Helly_Theorem_l89_89977


namespace tan_315_eq_neg1_l89_89250

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := by
  -- The statement means we need to prove that the tangent of 315 degrees is -1
  sorry

end tan_315_eq_neg1_l89_89250


namespace triangle_inequality_5_l89_89567

def isTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_inequality_5 :
  ∀ (m : ℝ), isTriangle 3 4 m → (m = 5) :=
begin
  intros m h,
  -- Given options are 1, 5, 7, 9
  by_contradiction,
  -- Since 1 < m < 7, the only valid m is m = 5,
  sorry,
end

end triangle_inequality_5_l89_89567


namespace number_of_reachable_cells_after_10_moves_l89_89609

-- Define board size, initial position, and the number of moves
def board_size : ℕ := 21
def initial_position : ℕ × ℕ := (11, 11)
def moves : ℕ := 10

-- Define the main problem statement
theorem number_of_reachable_cells_after_10_moves :
  (reachable_cells board_size initial_position moves).card = 121 :=
sorry

end number_of_reachable_cells_after_10_moves_l89_89609


namespace tan_315_eq_neg1_l89_89233

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by 
  sorry

end tan_315_eq_neg1_l89_89233


namespace part1_part2_l89_89667

noncomputable def triangle_ABC (A B C a b c : ℝ) : Prop :=
  ∀ A B C a b c : ℝ,
    (sin C * sin (A - B) = sin (B) * sin (C - A)) →
    (A = 2 * B) →
    C = 5 * pi / 8

theorem part1 (A B C a b c : ℝ) (h1 : triangle_ABC A B C a b c) : C = 5 * Real.pi / 8 :=
  by sorry

noncomputable def triangle_ABC_equality (A B C a b c : ℝ) : Prop :=
  ∀ A B C a b c : ℝ,
    (sin C * sin (A - B) = sin (B) * sin (C - A)) →
    2 * a^2 = b^2 + c^2

theorem part2 (A B C a b c : ℝ) (h1 : triangle_ABC_equality A B C a b c) : 2 * a^2 = b^2 + c^2 :=
  by sorry

end part1_part2_l89_89667
